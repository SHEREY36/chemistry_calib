"""
Export the calibrated power-law chemistry (theta_chem, Phase 4's posterior
mean) to the two forms CFD-ACE+ can actually consume as a surface boundary
condition:

  (1) A UDF (user-defined C function) that evaluates our EXACT calibrated
      power law at whatever local wall temperature/partial pressures the
      flow solver computes each iteration. This is the RECOMMENDED path:
      zero information loss, because it is literally our fitted function,
      not a re-derivation of it.

  (2) An elementary (site-based) surface-reaction mechanism deck, for cases
      where a UDF isn't available/desired (e.g. the solver needs to track
      individual adsorbed-species site fractions for other post-processing).
      This is NECESSARILY LOSSY: a 4-8 parameter lumped power law does not
      uniquely determine 9+ elementary rate constants. What this function
      does is (a) start from literature-seed values for every step (from
      build_steps_and_cfd_integration.md Phase 9.3 -- Coltrin/Kee, Ho &
      Melius, Imai 2008, etc.), all tagged status="seed", and (b) perform
      ONE constrained consistency calibration: the rate-limiting step's
      apparent activation energy is set to reproduce our calibrated
      kappa_GR (tagged status="fit_to_power_law"). It does NOT claim to
      have independently derived all 9 rate constants from 4 numbers --
      that would be a real overclaim. See the docstring on
      `_calibrate_rate_limiting_step` for exactly what is and isn't pinned.

HOW CFD OUTPUTS RE-ENTER THE ML PIPELINE (the other direction of this
module): once CFD-ACE+ runs with either surface BC above, its output
(local wall T, local p_i, resulting GR/Ge profiles across the wafer -- see
cfd/io.py:CFDResult) is consumed by cfd/transfer.py:extract_transfer_priors,
which turns "how much did the local surface conditions differ from the
setpoint" into an INFORMATIVE prior on reactor_transfer.py's alpha_{i,r}/
eta_r -- replacing the current weak N(0,1) prior with a physically-derived
one. That is the single biggest thing CFD buys this pipeline: it resolves
the alpha/eta identifiability problem documented in METHODOLOGY.md sec 8
(posterior correlations up to -0.97) by computing alpha from geometry
instead of trying to infer it blindly from 18-35 wafer outcomes.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from chem_ml.config import R_GAS
from chem_ml.physics_core import destandardize_kappa

log = logging.getLogger("chem_ml")

# ---------------------------------------------------------------------------
# Literature-seed elementary mechanism (build_steps_and_cfd_integration.md
# Sec. 9.3, Tables). Values are representative ORDER-OF-MAGNITUDE seeds from
# the cited source classes, NOT independently re-verified against the
# primary references in this codebase -- do that verification pass before
# using these numerically in a real CFD-ACE+ run (see the CRITICAL HONESTY
# NOTE in the original doc). Ea in kcal/mol, A in consistent Arrhenius units
# (s^-1 for gas/first-order surface steps), gamma = sticking coefficient
# (dimensionless, 0<gamma<=1) for adsorption steps.
# ---------------------------------------------------------------------------
_KCAL_PER_MOL_TO_J_PER_MOL = 4184.0


@dataclass
class MechanismStep:
    step_id: str
    reaction: str
    rate_form: str            # "arrhenius" | "sticking" | "arrhenius_reversible"
    A: float | None = None     # pre-exponential (s^-1), None for sticking-only steps
    Ea_kcal_mol: float | None = None
    gamma: float | None = None  # sticking coefficient, for "sticking" rate_form
    source_class: str = ""
    status: str = "seed"       # "seed" | "fit_to_power_law" | "verified"

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id, "reaction": self.reaction, "rate_form": self.rate_form,
            "A_per_s": self.A, "Ea_kcal_mol": self.Ea_kcal_mol,
            "Ea_J_per_mol": self.Ea_kcal_mol * _KCAL_PER_MOL_TO_J_PER_MOL if self.Ea_kcal_mol else None,
            "gamma_sticking": self.gamma, "source_class": self.source_class, "status": self.status,
        }


def _seed_mechanism_dcs() -> list[MechanismStep]:
    """The DCS/GeH4/HCl/B2H6 mechanism (S1-S9, G1-G3), literature seeds,
    exactly as tabled in build_steps_and_cfd_integration.md Sec. 9.3."""
    return [
        MechanismStep("G1", "SiH2Cl2 <=> SiCl2 + H2", "arrhenius_reversible",
                      Ea_kcal_mol=65.0, source_class="Coltrin/Kee chlorosilane; Ho & Melius"),
        MechanismStep("G2", "GeH4 <=> GeH2 + H2", "arrhenius_reversible",
                      Ea_kcal_mol=47.0, source_class="germane pyrolysis literature"),
        MechanismStep("S1", "SiH2Cl2(g) + 2s -> SiCl2(s) + 2H(s)", "sticking",
                      gamma=0.03, source_class="Coltrin/Kee; Kleijn reactor models"),
        MechanismStep("S2", "SiCl2(s) -> Si(b) + 2Cl(s)", "arrhenius",
                      A=1e13, Ea_kcal_mol=50.0, source_class="chlorosilane surface kinetics"),
        MechanismStep("S3", "2H(s) -> H2(g) + 2s", "arrhenius",
                      A=1e13, Ea_kcal_mol=51.5, source_class="Si(100) H2 TPD literature",
                      status="fit_to_power_law"),  # the rate-limiting step -- see below
        MechanismStep("S4", "2Cl(s) -> Cl2(g) + 2s", "arrhenius",
                      A=1e13, Ea_kcal_mol=72.0, source_class="Cl/Si desorption literature"),
        MechanismStep("S5", "HCl(g) + Si(b) -> SiHCl(s)/SiCl2(g)", "arrhenius",
                      A=1e12, Ea_kcal_mol=30.0, source_class="SEG selectivity literature"),
        MechanismStep("S6", "GeH4(g) + 2s -> Ge(s) + 2H(s)", "sticking",
                      gamma=0.3, source_class="Imai 2008; germane surface"),
        MechanismStep("S7", "Ge(s) -> Ge(b)", "arrhenius",
                      A=1e12, Ea_kcal_mol=15.0, source_class="Imai 2008; Ge/Si competition"),
        MechanismStep("S8", "B2H6(g) + 2s -> 2BH3(s) -> 2B(b)", "sticking",
                      gamma=0.1, source_class="Tomasini DS2; B-doping literature"),
        MechanismStep("S9", "H(s) + Cl(s) -> HCl(g) + 2s", "arrhenius",
                      A=1e13, Ea_kcal_mol=58.0, source_class="surface HCl formation"),
    ]


def _seed_mechanism_silane() -> list[MechanismStep]:
    """UNCALIBRATED alternate Si-source mechanism for a silane (SiH4) process
    -- NO Tomasini data covers silane at all (DS1-DS4 are 100% DCS-based),
    so every value here is a literature seed. Included because the registry
    (chem_ml/registry.py) already declares silane as a valid Si-source
    species and the assembler's anti-contamination gate means adding this
    mechanism cannot leak into or perturb the calibrated DCS chemistry above
    -- they are structurally independent networks selected by chem_class/
    declared species, never blended. Ea in kcal/mol are representative
    silane-surface-chemistry ranges (SiH4 dissociative chemisorption on
    Si(100), H2 desorption still typically rate-limiting at moderate T)."""
    return [
        MechanismStep("G1s", "SiH4 <=> SiH2 + H2", "arrhenius_reversible",
                      Ea_kcal_mol=57.0, source_class="silane pyrolysis literature", status="seed"),
        MechanismStep("S1s", "SiH4(g) + 2s -> SiH2(s) + 2H(s)", "sticking",
                      gamma=0.01, source_class="SiH4/Si(100) dissociative chemisorption lit.",
                      status="seed"),
        MechanismStep("S2s", "SiH2(s) -> Si(b) + 2H(s)", "arrhenius",
                      A=1e13, Ea_kcal_mol=38.0, source_class="silane surface decomposition lit.",
                      status="seed"),
        MechanismStep("S3s", "2H(s) -> H2(g) + 2s", "arrhenius",
                      A=1e13, Ea_kcal_mol=51.5, source_class="Si(100) H2 TPD literature (same H-desorption physics as DCS case)",
                      status="seed"),
        MechanismStep("S6", "GeH4(g) + 2s -> Ge(s) + 2H(s)", "sticking",
                      gamma=0.3, source_class="Imai 2008; germane surface (reused from DCS mechanism)",
                      status="seed"),
        MechanismStep("S7", "Ge(s) -> Ge(b)", "arrhenius",
                      A=1e12, Ea_kcal_mol=15.0, source_class="Imai 2008; Ge/Si competition (reused)",
                      status="seed"),
    ]


def _calibrate_rate_limiting_step(steps: list[MechanismStep], kappa_GR_std: float,
                                  invT_scaler: tuple[float, float]) -> list[MechanismStep]:
    """The ONE consistency calibration this module performs: pin the
    rate-limiting step's (S3, H2 desorption -- the paper's own text
    identifies this as 'often rate-limiting') activation energy so the
    mechanism's overall apparent Ea matches our DATA-CALIBRATED kappa_GR,
    via Ea = -R * kappa_GR (kappa_GR is negative since GR rises with T; see
    METHODOLOGY.md sec 4/5 for the sign derivation). `kappa_GR_std` is the
    RAW posterior value, fit against STANDARDIZED 1/T (see features.py) --
    it must be destandardized back to real K units via `invT_scaler`
    BEFORE this conversion, or the resulting "activation energy" is off by
    the standardization's std (a real bug this module's own test caught:
    kappa_GR_std ~ O(1-2), not ~O(24000), because it multiplies
    (1/T - mu)/sd rather than 1/T directly).

    This is deliberately the ONLY parameter pinned by data. It does NOT
    imply the other 8 steps' Ea/A/gamma values are independently validated
    -- a lumped 4-parameter power law cannot determine 9 elementary rate
    constants (an underdetermined inverse problem); pretending otherwise
    would be exactly the kind of overclaim this project has been explicit
    about avoiding (see METHODOLOGY.md sec 12). Full validation of the
    elementary mechanism requires either independent literature
    verification of each step, or fitting all 9 against a much richer
    dataset than Tomasini's (e.g. isotope-labeled TPD, in-situ FTIR)."""
    kappa_GR_K = destandardize_kappa(kappa_GR_std, invT_scaler)
    Ea_from_data_kcal = -kappa_GR_K * R_GAS / _KCAL_PER_MOL_TO_J_PER_MOL
    out = []
    for s in steps:
        if s.step_id == "S3":
            s = MechanismStep(**{**s.__dict__, "Ea_kcal_mol": Ea_from_data_kcal})
        out.append(s)
    return out


def _udf_c_source(theta_gr: dict, theta_ge: dict, theta_b: dict | None,
                  invT_scaler: tuple[float, float]) -> str:
    """Generate a C source file implementing the EXACT calibrated power law
    as a CFD-ACE+ user surface-flux subroutine. The function signature below
    is illustrative -- CFD-ACE+'s actual UDF calling convention (argument
    order, units, how it's registered in the .DTF/model file) must be
    confirmed against your license's User Subroutines manual; what's exact
    regardless of that wrapper is the arithmetic inside, which is a literal
    transcription of chem_ml/physics_core.py's gr_logmodel/ge_logmodel using
    the Phase 4 posterior-mean parameters below.

    p_HCl, p_GeH4, p_B2H6 passed in are the LOCAL (wall-adjacent) partial
    pressures the CFD flow/species solve computes for this cell/face on
    this iteration -- NOT the inlet setpoint. That's the entire value of
    doing this in CFD rather than table-lookup: the surface BC responds to
    whatever local depletion/mixing the 3D flow actually produces."""
    mu, sd = invT_scaler
    b_block = ""
    if theta_b is not None:
        b_block = f"""
/* B/Si ratio power law (Phase 4, DS2 fit). Returns [B]/[Si]; multiply by
   the local Si atomic density (~5e22 at/cm3, Tomasini's own DS2 convention)
   to get an absolute concentration if the solver needs one. */
double calibrated_B_over_Si(double p_HCl_over_pDCS, double p_GeH4_over_pDCS,
                            double p_B2H6_over_pDCS) {{
    const double lnK_B      = {theta_b['lnK_B']:.10g};
    const double beta_HCl   = {theta_b['beta_HCl']:.10g};
    const double beta_GeH4  = {theta_b['beta_GeH4']:.10g};
    const double beta_B2H6  = {theta_b['beta_B2H6']:.10g};
    double ln_b_over_si = lnK_B
        + beta_HCl  * log(p_HCl_over_pDCS)
        + beta_GeH4 * log(p_GeH4_over_pDCS)
        + beta_B2H6 * log(p_B2H6_over_pDCS);
    return exp(ln_b_over_si);
}}
"""
    return f"""/* AUTO-GENERATED by chem_ml/cfd/mechanism.py -- do not hand-edit.
 * Source: Phase 4 posterior-mean parameters (Tomasini reproduction).
 * Regenerate with: python -m chem_ml.cli export-mechanism --form udf
 *
 * ADAPT: the exact function signature/registration CFD-ACE+ expects for a
 * user surface-reaction subroutine (see your license's User Subroutines
 * manual). The arithmetic below is exact regardless of that wrapper -- it
 * is a literal C transcription of chem_ml/physics_core.py:gr_logmodel /
 * ge_logmodel evaluated at the Phase 4 posterior mean. Units: pressures are
 * DIMENSIONLESS ratios p_i/p_DCS (Tomasini's own normalization convention,
 * see schema.py); T_wall_K is the LOCAL wall temperature CFD-ACE+ computes,
 * not a setpoint. GR is returned in nm/min (Tomasini's native unit) --
 * convert to your solver's deposition-flux units (e.g. mol/m^2/s) using the
 * SiGe molar volume before wiring into the species/energy source term.
 */
#include <math.h>

/* Standardized-1/T scaler from the fit -- MUST match the value baked in
   here (mu={mu:.10g}, sd={sd:.10g}); it is NOT re-derivable from T alone. */
static const double INV_T_MU = {mu:.10g};
static const double INV_T_SD = {sd:.10g};

/* Growth rate power law (Phase 4, DS1 fit). Returns GR in nm/min. */
double calibrated_GR_nm_min(double T_wall_K, double p_HCl_over_pDCS, double p_GeH4_over_pDCS) {{
    const double lnK_GR     = {theta_gr['lnK_GR']:.10g};
    const double kappa_GR   = {theta_gr['kappa_GR']:.10g};   /* coefficient of STANDARDIZED 1/T */
    const double gamma_HCl  = {theta_gr['gamma_HCl']:.10g};
    const double gamma_GeH4 = {theta_gr['gamma_GeH4']:.10g};

    double invT_std = (1.0 / T_wall_K - INV_T_MU) / INV_T_SD;
    double ln_GR = lnK_GR
        + kappa_GR   * invT_std
        + gamma_HCl  * log(p_HCl_over_pDCS)
        + gamma_GeH4 * log(p_GeH4_over_pDCS);
    return exp(ln_GR);
}}

/* Ge atomic fraction power law (Phase 4, DS1 fit). Returns x_Ge in [0,1]. */
double calibrated_xGe(double T_wall_K, double p_HCl_over_pDCS, double p_GeH4_over_pDCS) {{
    const double lnK_Ge      = {theta_ge['lnK_Ge']:.10g};
    const double kappa_Ge    = {theta_ge['kappa_Ge']:.10g};
    const double dgamma_HCl  = {theta_ge['dgamma_HCl']:.10g};
    const double dgamma_GeH4 = {theta_ge['dgamma_GeH4']:.10g};

    double invT_std = (1.0 / T_wall_K - INV_T_MU) / INV_T_SD;
    double ln_ratio = lnK_Ge
        + kappa_Ge    * invT_std
        + dgamma_HCl  * log(p_HCl_over_pDCS)
        + dgamma_GeH4 * log(p_GeH4_over_pDCS);
    double ratio = exp(ln_ratio);       /* x/(1-x) */
    return ratio / (1.0 + ratio);       /* -> x */
}}
{b_block}"""


def export_mechanism_to_cfd(theta_gr: dict, theta_ge: dict, invT_scaler: tuple[float, float],
                            out_dir: str | Path, theta_b: dict | None = None,
                            chem_system: str = "dcs") -> dict:
    """Write BOTH CFD-ACE+ input forms (Phase 9.3/9.4.1):
      out_dir/surface_bc_udf.c          -- exact UDF (recommended path)
      out_dir/elementary_mechanism.json -- literature-seed mechanism table,
                                           with S3's Ea pinned to the
                                           calibrated kappa_GR (dcs system only)
    `chem_system`: "dcs" (the calibrated Tomasini system) or "silane"
    (uncalibrated seed-only alternate Si-source system -- see
    _seed_mechanism_silane docstring). Returns a manifest dict recording
    what's calibrated vs. seed, so downstream users can't lose track."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    udf_src = _udf_c_source(theta_gr, theta_ge, theta_b, invT_scaler)
    udf_path = out_dir / "surface_bc_udf.c"
    udf_path.write_text(udf_src)

    if chem_system == "dcs":
        steps = _calibrate_rate_limiting_step(_seed_mechanism_dcs(), theta_gr["kappa_GR"], invT_scaler)
        calibration_status = "S3 (H2 desorption, rate-limiting) Ea pinned to data; all other steps are literature seeds"
    elif chem_system == "silane":
        steps = _seed_mechanism_silane()
        calibration_status = "ALL STEPS ARE UNCALIBRATED LITERATURE SEEDS -- no Tomasini data covers silane chemistry"
    else:
        raise ValueError(f"Unknown chem_system '{chem_system}', expected 'dcs' or 'silane'")

    mech_path = out_dir / "elementary_mechanism.json"
    manifest = {
        "chem_system": chem_system,
        "calibration_status": calibration_status,
        "site_conservation_constraint": "theta_free + theta_H + theta_Cl + theta_Si + theta_Ge = 1 (enforce as a hard constraint in the CFD surface-chemistry solve)",
        "steps": [s.to_dict() for s in steps],
    }
    mech_path.write_text(json.dumps(manifest, indent=2))

    log.info("Wrote CFD mechanism exports to %s (udf) and %s (elementary, %s)",
             udf_path, mech_path, calibration_status)
    return {"udf_path": str(udf_path), "mechanism_path": str(mech_path), "manifest": manifest}
