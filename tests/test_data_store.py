"""Tests for additive/incremental data ingestion (data_store.py) and
warm-start calibration (pipeline.run_phase4_warm_start) -- specifically the
anti-contamination guarantees the whole design is built around: new data of
a different reactor or chemistry class must NOT perturb the existing
GR/Ge/B fits, while new data of the SAME reactor/class DOES get pooled in."""
import numpy as np
import pytest

from chem_ml.config import Config
from chem_ml.data_store import (
    load_accumulated_dataset, register_new_data, registered_source_tags,
)
from chem_ml.pipeline import run_phase4_calibration, run_phase4_warm_start
from chem_ml.schema import ChemClass, Mode


def _write_standard_csv(path, rows):
    header = "T_C,HCl_over_DCS,GeH4_over_DCS,GR_nm_min,Ge_at_pct\n"
    body = "\n".join(f"{t},{hcl},{geh4},{gr},{ge}" for t, hcl, geh4, gr, ge in rows)
    path.write_text(header + body + "\n")


@pytest.fixture(scope="module")
def phase4(tmp_path_factory):
    return run_phase4_calibration(Config())


def test_register_and_load_accumulated_dataset(tmp_path):
    cfg = Config(data_raw="data/raw", data_processed=str(tmp_path))
    csv_path = tmp_path / "new_sige.csv"
    _write_standard_csv(csv_path, [(730, 0.6, 0.03, 40.0, 20.0), (730, 0.5, 0.035, 45.0, 21.0)])

    register_new_data(cfg, str(csv_path), reactor_id="ASM_Epsilon", chem_class=ChemClass.SIGE,
                      source_tag="test_batch_1")
    assert "test_batch_1" in registered_source_tags(cfg)

    ds = load_accumulated_dataset(cfg)
    new_rows = ds.filter(source_dataset="test_batch_1")
    assert len(new_rows) == 2
    # base Tomasini data is still all there too
    assert len(ds.filter(source_dataset="DS1")) == 70


def test_reregistering_same_tag_raises(tmp_path):
    cfg = Config(data_raw="data/raw", data_processed=str(tmp_path))
    csv_path = tmp_path / "new_sige.csv"
    _write_standard_csv(csv_path, [(730, 0.6, 0.03, 40.0, 20.0)])
    register_new_data(cfg, str(csv_path), reactor_id="ASM_Epsilon", chem_class=ChemClass.SIGE,
                      source_tag="dup_tag")
    with pytest.raises(ValueError, match="already registered"):
        register_new_data(cfg, str(csv_path), reactor_id="ASM_Epsilon", chem_class=ChemClass.SIGE,
                          source_tag="dup_tag")


def test_warm_start_ignores_different_chem_class(tmp_path, phase4):
    """A new SiGe:P (phosphine) dataset must NOT touch the GR/Ge/B fits at
    all -- the assembler's hard class gate plus run_phase4_warm_start's
    (chem_class, reactor_id) filter should exclude it entirely, not just
    'not much effect'. Assert the returned mcmc objects are the SAME
    object (unchanged), not merely numerically close."""
    from chem_ml.schema import Dataset, CanonicalRow

    junk_rows = [CanonicalRow(
        reactor_id="ASM_Epsilon", chem_class=ChemClass.SIGE_P, mode=Mode.BLANKET,
        T_K=1000.0, p_DCS=1.0, p_GeH4=0.9, p_HCl=0.01,  # deliberately wild values
        GR_nm_min=9999.0, Ge_at_frac=0.001, source_dataset="phosphine_probe",
    )]
    new_ds = Dataset(junk_rows)

    updated = run_phase4_warm_start(Config(), phase4, new_ds)
    assert updated["n_new_sige_rows"] == 0
    assert updated["n_new_sigeb_rows"] == 0
    assert updated["mcmc_gr"] is phase4["mcmc_gr"]
    assert updated["mcmc_ge"] is phase4["mcmc_ge"]
    assert updated["mcmc_b"] is phase4["mcmc_b"]


def test_warm_start_ignores_different_reactor(tmp_path, phase4):
    """New SiGe data from a DIFFERENT reactor (e.g. an XYZ tool that
    hasn't been through Phase 7 transfer calibration) must not leak into
    theta_chem via warm-start either -- same filter, same guarantee."""
    from chem_ml.schema import Dataset, CanonicalRow

    other_reactor_rows = [CanonicalRow(
        reactor_id="XYZ_tool_1", chem_class=ChemClass.SIGE, mode=Mode.BLANKET,
        T_K=1000.0, p_DCS=1.0, p_GeH4=0.03, p_HCl=0.5,
        GR_nm_min=45.0, Ge_at_frac=0.20, source_dataset="xyz_probe",
    )]
    new_ds = Dataset(other_reactor_rows)

    updated = run_phase4_warm_start(Config(), phase4, new_ds)
    assert updated["n_new_sige_rows"] == 0
    assert updated["mcmc_gr"] is phase4["mcmc_gr"]
    assert updated["mcmc_ge"] is phase4["mcmc_ge"]


def test_warm_start_pools_matching_new_data_and_stays_stable(phase4):
    """New data from the SAME reactor/class DOES get fit (n_new_sige_rows
    reflects it), and because it's drawn from the ALREADY-fitted power law
    plus small noise, the warm-started posterior should stay close to the
    previous one (a consistency/stability check: the informative prior
    should stop a handful of new points from swinging the fit wildly, which
    is exactly what a from-scratch fit on 6 points with the ORIGINAL weak
    literature priors would risk)."""
    from chem_ml.calibration import posterior_mean_params
    from chem_ml.physics_core import gr_logmodel
    from chem_ml.schema import Dataset, CanonicalRow

    prev_gr_params = posterior_mean_params(phase4["mcmc_gr"], ["lnK_GR", "kappa_GR", "gamma_HCl", "gamma_GeH4"])
    mu, sd = phase4["features_ds1"].invT_scaler
    rng = np.random.default_rng(0)

    new_rows = []
    for T_c in (700.0, 720.0, 740.0):
        T_K = T_c + 273.15
        hcl_ratio, geh4_ratio = 0.5, 0.035
        import jax.numpy as jnp
        invT_std = (1.0 / T_K - mu) / sd
        X = jnp.array([[invT_std, jnp.log(hcl_ratio), jnp.log(geh4_ratio), 0.0]])
        gr_true = float(jnp.exp(gr_logmodel(prev_gr_params, X))[0]) * np.exp(rng.normal(0, 0.05))
        new_rows.append(CanonicalRow(
            reactor_id="ASM_Epsilon", chem_class=ChemClass.SIGE, mode=Mode.BLANKET,
            T_K=T_K, p_DCS=1.0, p_GeH4=geh4_ratio, p_HCl=hcl_ratio,
            GR_nm_min=gr_true, Ge_at_frac=0.20, source_dataset="consistent_probe",
        ))
    new_ds = Dataset(new_rows)

    updated = run_phase4_warm_start(Config(mcmc=Config().mcmc), phase4, new_ds)
    assert updated["n_new_sige_rows"] == 3
    assert updated["mcmc_gr"] is not phase4["mcmc_gr"]  # it DID refit

    new_gr_params = posterior_mean_params(updated["mcmc_gr"], ["lnK_GR", "kappa_GR", "gamma_HCl", "gamma_GeH4"])
    # kappa_GR is the most data-hungry/sloppy param (METHODOLOGY.md sec 7);
    # even so, with an informative prior centered on the previous fit, 3
    # consistent new points shouldn't move it by an order of magnitude.
    assert abs(new_gr_params["kappa_GR"] - prev_gr_params["kappa_GR"]) < 3.0 * abs(prev_gr_params["kappa_GR"])
    assert np.sign(new_gr_params["gamma_HCl"]) == np.sign(prev_gr_params["gamma_HCl"])
