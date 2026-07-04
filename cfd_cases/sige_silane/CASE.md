# CFD-ACE+ Case: SiGe via Silane/GeH4 on AMAT's 3D reactor (UNCALIBRATED SEED)

**Calibration status: NOT calibrated. Every rate constant in this case is a
literature seed.** Tomasini's data (DS1-DS4, all of Phases 1-8) is 100%
DCS-based -- no silane (SiH4) chemistry appears anywhere in it. This case
exists because you asked to "simulate these calibrated reactions on CFD-ACE+
to check again" for a precursor family the current data doesn't cover, and
because `chem_ml/registry.py` already declares silane as a valid Si-source
species: the assembler's hard class gate means adding this case cannot
perturb the calibrated DCS network in any way (they are structurally
independent -- see `chem_ml/assembler.py`'s anti-contamination test). Treat
everything below as a **scoping/template exercise**, not a predictive tool.

## 1. Does "SiGe with silane" make sense as a second case?

Yes, as a reactor-engineering comparison -- silane-based SiGe epitaxy is a
real, common alternative to DCS: it decomposes more readily (lower thermal
budget, higher growth rate at a given T, often used for blanket/non-
selective epi since it lacks HCl's inherent etch-back character), so
comparing DCS vs. silane behavior in the SAME 3D geometry is a genuinely
useful CFD study (different diffusivities/reactivities can change wafer-
scale uniformity differently even for a nominally similar target GR/Ge%).
What it does **not** give you is a second calibrated chemistry to compare --
that requires new experimental data (see sec. 4).

## 2. Placeholders -- silane processes typically run cooler than DCS

| Item | Placeholder used below | Rationale |
|---|---|---|
| `geometry_id` | `AMAT_3D_v1` | same mesh as the DCS case, for comparability |
| Susceptor T setpoints | 550-650 C | silane decomposes more readily than DCS; literature SiGe:silane processes commonly run 100-150C cooler than DCS-based ones for a comparable GR -- ADJUST to your actual process |
| Total pressure | 10 Torr | kept equal to the DCS case for apples-to-apples CFD comparison; real silane SiGe can also run at lower pressure (UHV-CVD-style, <1 Torr) depending on tool -- replace with your actual spec |
| Reference SiH4 flow | 50 sccm | arbitrary, mirrors the DCS case's reference DCS flow for comparability |
| HCl | 0 (or trace, if selectivity is needed) | pure silane blanket epi often omits HCl entirely; add it back if your process is selective |

## 3. Surface reaction mechanism (ALL SEED -- verify before quantitative use)

No UDF is provided for this case (`surface_bc_udf.c` alongside this file is
a placeholder explaining why, not a function to call) -- there is no
calibrated power law to transcribe. Use the elementary mechanism below as
CFD-ACE+'s native surface-reaction input instead:

| Step | Reaction | Rate form | A (s^-1) | Ea (kcal/mol) | gamma (sticking) | Status |
|---|---|---|---|---|---|---|
| G1s | SiH4 <=> SiH2 + H2 | Arrhenius (rev.) | -- | 57.0 | -- | seed |
| S1s | SiH4(g) + 2s -> SiH2(s) + 2H(s) | sticking | -- | -- | 0.01 | seed |
| S2s | SiH2(s) -> Si(b) + 2H(s) | Arrhenius | 1e13 | 38.0 | -- | seed |
| S3s | 2H(s) -> H2(g) + 2s | Arrhenius | 1e13 | 51.5 | -- | seed (same H-desorption physics as the DCS case's S3, reused) |
| S6 | GeH4(g) + 2s -> Ge(s) + 2H(s) | sticking | -- | -- | 0.3 | seed (identical to the DCS case -- Ge incorporation physics is precursor-family-independent in this simplified picture) |
| S7 | Ge(s) -> Ge(b) | Arrhenius | 1e12 | 15.0 | -- | seed (identical to the DCS case) |

Site conservation still applies: $\theta_{\text{free}} + \theta_H + \theta_{Si} + \theta_{Ge} = 1$
(no Cl term here, since there's no DCS/HCl in this system).

**None of the Arrhenius parameters above have been independently verified
against their cited source classes in this codebase.** Before trusting a
CFD-ACE+ run's absolute GR/Ge numbers for this system, at minimum: (a)
verify S1s/S2s/S3s against primary silane-surface-kinetics literature
(Ho & Melius-style ab initio work, or TPD studies specific to SiH4/Si(100)),
and (b) run the calibration in sec. 4 below.

## 4. How to actually calibrate this case (the real path forward)

This is a structurally NEW sub-model, not a retrofit of the DCS one --
follow Phase 1-4's exact method on a new precursor:

1. **Collect a DS1-equivalent silane sweep**: on ONE reference reactor,
   vary T (aim for >=6 levels spanning your real process window) and
   GeH4/SiH4 ratio (aim for >=4-5 levels at each T), recording GR (with
   growth time!), Ge%, and B% if doped -- exactly the shape of data that
   made DS1 identifiable in the first place (see METHODOLOGY.md sec 2).
2. **Register it**: `python -m chem_ml.cli add-data --csv silane_sweep.csv --reactor <ref_reactor> --chem-class SiGe --tag silane_sweep_v1` (using the standard intake CSV format in `chem_ml/schema.py:ingest_standard_csv` -- columns `T_C, HCl_over_DCS, GeH4_over_DCS, GR_nm_min, Ge_at_pct`; for a pure-silane process without DCS at all, this requires a small structural change: a new `gr_logmodel_silane`/`ge_logmodel_silane` in `chem_ml/physics_core.py` keyed off `p_GeH4/p_SiH4` rather than `p_i/p_DCS`, plus a new NumPyro model in `calibration.py` -- both following the EXACT pattern of the existing GR/Ge functions, not sharing their parameters (this is what keeps it from contaminating the DCS fit, same discipline as the boron sub-model's isolation).
3. **Calibrate**: run NUTS the same way Phase 4 does, get a real posterior over silane-specific rate-law parameters.
4. **Re-run `export-mechanism --system silane`**: once step 3 exists, extend
   `chem_ml/cfd/mechanism.py` to pin this system's own rate-limiting step
   the same way `_calibrate_rate_limiting_step` does for DCS -- then this
   case graduates from "seed" to "calibrated," and the UDF placeholder
   above gets replaced with a real one.

Until then, use this case for CFD-ACE+ geometry/mixing sensitivity
scoping only (e.g. "how does a silane-scale diffusivity change wafer
uniformity vs. DCS in this specific 3D reactor"), not for absolute GR/Ge
predictions.
