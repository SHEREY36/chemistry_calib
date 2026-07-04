"""Tests for the Phase 9 CFD I/O framework: mechanism export, deck I/O
contract, and the CFD -> reactor_transfer prior hookup. No real CFD-ACE+
license is available, so these exercise the CONTRACT (file formats,
round-trips, numerical correctness of the exported UDF arithmetic) rather
than an actual CFD run."""
import json
import re

import jax.numpy as jnp
import numpy as np
import pytest

from chem_ml.cfd.io import (
    CFDCondition, CFDResult, parse_cfd_output, write_cfd_deck,
    write_output_contract_template,
)
from chem_ml.cfd.mechanism import export_mechanism_to_cfd
from chem_ml.cfd.transfer import extract_transfer_priors
from chem_ml.physics_core import gr_logmodel, ge_logmodel
from chem_ml.reactor_transfer import reactor_transfer_model_gr_ge

_THETA_GR = {"lnK_GR": 6.4, "kappa_GR": -1.5, "gamma_HCl": -0.71, "gamma_GeH4": 1.31}
_THETA_GE = {"lnK_Ge": 0.66, "kappa_Ge": 0.26, "dgamma_HCl": 0.10, "dgamma_GeH4": 0.51}
_INVT_SCALER = (9.1e-4, 5.2e-5)


def test_export_mechanism_dcs_writes_udf_and_mechanism(tmp_path):
    result = export_mechanism_to_cfd(_THETA_GR, _THETA_GE, _INVT_SCALER, tmp_path, chem_system="dcs")
    udf_src = open(result["udf_path"]).read()
    for name in ("calibrated_GR_nm_min", "calibrated_xGe", str(_THETA_GR["gamma_HCl"])):
        assert name in udf_src

    manifest = json.loads(open(result["mechanism_path"]).read())
    assert manifest["chem_system"] == "dcs"
    s3 = next(s for s in manifest["steps"] if s["step_id"] == "S3")
    assert s3["status"] == "fit_to_power_law"
    # Ea = -kappa_GR * R / kcal_to_J; kappa_GR is on STANDARDIZED invT here,
    # so this just checks the pin logic ran and produced a physically
    # sane activation energy (tens of kcal/mol), not a specific number.
    assert 5.0 < s3["Ea_kcal_mol"] < 200.0
    other_seed = next(s for s in manifest["steps"] if s["step_id"] == "S1")
    assert other_seed["status"] == "seed"


def test_export_mechanism_silane_is_all_seed(tmp_path):
    result = export_mechanism_to_cfd(_THETA_GR, _THETA_GE, _INVT_SCALER, tmp_path, chem_system="silane")
    manifest = json.loads(open(result["mechanism_path"]).read())
    assert manifest["chem_system"] == "silane"
    assert all(s["status"] == "seed" for s in manifest["steps"])
    assert "UNCALIBRATED" in manifest["calibration_status"]

    # The silane UDF must NOT be the DCS-calibrated function (there's no
    # calibrated silane power law to transcribe -- writing the DCS one here
    # would be actively misleading, since it takes p_HCl_over_pDCS/
    # p_GeH4_over_pDCS args that don't correspond to a silane process).
    udf_src = open(result["udf_path"]).read()
    assert "calibrated_GR_nm_min" not in udf_src
    assert "NO calibrated power-law UDF exists" in udf_src


def test_udf_arithmetic_matches_python_physics_core(tmp_path):
    """The generated C source's literal constants must match the Python
    physics core it's supposedly a transcription of -- this test doesn't
    compile C, it checks the embedded constants round-trip exactly."""
    result = export_mechanism_to_cfd(_THETA_GR, _THETA_GE, _INVT_SCALER, tmp_path, chem_system="dcs")
    udf_src = open(result["udf_path"]).read()
    for key, val in {**_THETA_GR, **_THETA_GE}.items():
        assert f"{val:.10g}" in udf_src, f"{key}={val} not found verbatim in generated UDF"

    # Cross-check the UDF's documented formula against gr_logmodel directly.
    T_wall, p_hcl, p_geh4 = 1000.0, 0.5, 0.03
    mu, sd = _INVT_SCALER
    invT_std = (1.0 / T_wall - mu) / sd
    X = jnp.array([[invT_std, jnp.log(p_hcl), jnp.log(p_geh4), 0.0]])
    expected_gr = float(jnp.exp(gr_logmodel(_THETA_GR, X))[0])
    # Re-implement the UDF's exact arithmetic in Python to confirm it's the
    # same formula (this is what "exact transcription" is claiming).
    ln_gr = (_THETA_GR["lnK_GR"] + _THETA_GR["kappa_GR"] * invT_std
             + _THETA_GR["gamma_HCl"] * np.log(p_hcl) + _THETA_GR["gamma_GeH4"] * np.log(p_geh4))
    assert np.exp(ln_gr) == pytest.approx(expected_gr, rel=1e-9)


def test_deck_roundtrip_and_output_contract(tmp_path):
    cond = CFDCondition(T_set_K=1023.15, flows_sccm={"DCS": 10.0, "GeH4": 0.3, "HCl": 5.0, "B2H6": 1e-5},
                       P_tot_torr=10.0, geometry_id="AMAT_3D_v1")
    deck_path = write_cfd_deck(cond, tmp_path / "deck.txt")
    deck_text = deck_path.read_text()
    assert "T1023K" in deck_text or str(cond.T_set_K) in deck_text
    assert "surface_bc_udf.c" in deck_text

    template_path = write_output_contract_template(tmp_path / "template.csv")
    header = template_path.read_text().strip()
    assert header == "r_mm,surface_T_K,p_HCl_over_pDCS,p_GeH4_over_pDCS,p_B2H6_over_pDCS,GR_nm_min,Ge_frac"

    # Fill in a small synthetic wafer profile matching the contract and
    # parse it back -- this is standing in for CFD-ACE+'s real output until
    # a license is available.
    out_csv = tmp_path / "profile.csv"
    out_csv.write_text(
        "r_mm,surface_T_K,p_HCl_over_pDCS,p_GeH4_over_pDCS,p_B2H6_over_pDCS,GR_nm_min,Ge_frac\n"
        "0,1020.0,0.55,0.032,0.0,60.0,0.21\n"
        "50,1021.5,0.53,0.031,0.0,58.0,0.205\n"
        "100,1023.0,0.50,0.030,0.0,55.0,0.20\n"
    )
    result = parse_cfd_output(out_csv, cond)
    assert isinstance(result, CFDResult)
    assert result.r_mm.shape == (3,)
    assert result.surface_p_i["HCl"][0] == pytest.approx(0.55)


def test_parse_cfd_output_rejects_missing_columns(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("r_mm,surface_T_K\n0,1000\n")
    cond = CFDCondition(T_set_K=1000.0, flows_sccm={"DCS": 10.0, "GeH4": 0.3, "HCl": 5.0, "B2H6": 0.0},
                       P_tot_torr=10.0, geometry_id="g")
    with pytest.raises(ValueError, match="missing required columns"):
        parse_cfd_output(bad_csv, cond)


def test_extract_transfer_priors_recovers_known_alpha():
    """Synthetic CFD results where the wafer-surface HCl ratio is exactly
    1.5x the inlet setpoint ratio (a made-up but internally consistent
    'CFD run') -- extract_transfer_priors should recover ln(1.5)."""
    cond = CFDCondition(T_set_K=1000.0, flows_sccm={"DCS": 10.0, "GeH4": 0.3, "HCl": 5.0, "B2H6": 0.0},
                       P_tot_torr=10.0, geometry_id="g")
    inlet_hcl_ratio = 5.0 / 10.0
    inlet_geh4_ratio = 0.3 / 10.0
    r = np.array([0.0, 50.0, 100.0])
    result = CFDResult(
        condition=cond, r_mm=r,
        surface_T_K=np.full_like(r, 1005.0),
        surface_p_i={"HCl": np.full_like(r, inlet_hcl_ratio * 1.5),
                    "GeH4": np.full_like(r, inlet_geh4_ratio * 0.8),
                    "B2H6": np.zeros_like(r)},
        GR_profile_nm_min=np.full_like(r, 50.0), Ge_profile_frac=np.full_like(r, 0.2),
    )
    priors = extract_transfer_priors([result])
    assert priors["ln_alpha_HCl"][0] == pytest.approx(np.log(1.5), abs=1e-9)
    assert priors["ln_alpha_GeH4"][0] == pytest.approx(np.log(0.8), abs=1e-9)
    assert priors["dT_r_K_diagnostic_only"][0] == pytest.approx(5.0)
    # std is floored, not zero, even though this synthetic run is noiseless
    assert priors["ln_alpha_HCl"][1] >= 0.05


def test_reactor_transfer_model_accepts_cfd_priors_and_runs():
    """The alpha_priors hookup shouldn't just parse -- NUTS must actually
    run with it and use a DIFFERENT prior than the weak default."""
    import jax
    from numpyro.infer import MCMC, NUTS

    rng = np.random.default_rng(0)
    N = 12
    X = jnp.asarray(np.stack([
        rng.normal(0, 0.3, N), np.log(rng.uniform(0.3, 0.9, N)),
        np.log(rng.uniform(0.02, 0.05, N)), np.zeros(N),
    ], axis=1))
    y_gr_log = jnp.asarray(gr_logmodel(_THETA_GR, X) + rng.normal(0, 0.05, N))
    y_ge_log = jnp.asarray(ge_logmodel(_THETA_GE, X) + rng.normal(0, 0.05, N))

    cfd_priors = {"ln_alpha_HCl": (0.4, 0.1), "ln_alpha_GeH4": (-0.2, 0.1)}
    kernel = NUTS(reactor_transfer_model_gr_ge)
    mcmc = MCMC(kernel, num_warmup=200, num_samples=300, num_chains=1, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(0), X=X, y_gr_log=y_gr_log, y_ge_log=y_ge_log,
             theta_gr=_THETA_GR, theta_ge=_THETA_GE, alpha_priors=cfd_priors)
    s = mcmc.get_samples()
    # With a tight informative prior (std=0.1) and only 12 rows, the
    # posterior should sit close to the CFD-supplied mean, not near 0
    # (which is what the weak default N(0,1) would pull toward).
    assert float(np.mean(s["ln_alpha_HCl"])) == pytest.approx(0.4, abs=0.15)
