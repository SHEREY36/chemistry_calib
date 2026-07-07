import jax.numpy as jnp
import pytest

from chem_ml.config import Config
from chem_ml.pipeline import run_class_calibration
from chem_ml.schema import CanonicalRow, ChemClass, Dataset, Mode
from chem_ml import pipeline


def _row(**kwargs):
    base = dict(
        reactor_id="XYZ_tool_1",
        chem_class=ChemClass.SIGEC,
        mode=Mode.BLANKET,
        T_K=973.15,
        p_DCS=1.0,
        p_GeH4=0.03,
        p_HCl=0.2,
        p_MMS=0.004,
        GR_nm_min=100.0,
        Ge_at_frac=0.22,
    )
    base.update(kwargs)
    return CanonicalRow(**base)


def test_sigec_rows_do_not_enter_legacy_sige_class_route():
    ds = Dataset([
        _row(C_at_frac=0.006),
        _row(chem_class=ChemClass.SIGE, reactor_id="ASM_Epsilon", C_at_frac=None),
    ])

    out = run_class_calibration(
        Config(),
        ds=ds,
        chem_class=ChemClass.SIGE,
        reference_reactor="ASM_Epsilon",
    )

    assert out["report"]["n_rows"] == 1
    assert out["report"]["carbon_model_trained"] is False


def test_sigec_carbon_model_skips_with_clear_report_when_c_target_absent():
    ds = Dataset([_row(C_at_frac=None), _row(C_at_frac=None)])

    out = run_class_calibration(
        Config(),
        ds=ds,
        chem_class=ChemClass.SIGEC,
        reference_reactor="XYZ_tool_1",
    )

    assert out["report"]["n_rows"] == 2
    assert out["report"]["n_c_rows"] == 0
    assert out["report"]["carbon_model_trained"] is False
    assert "C_at_pct" in out["report"]["carbon_skip_reason"]


def test_sigec_carbon_model_slot_trains_when_c_target_exists(monkeypatch):
    captured = {}

    class FakeMCMC:
        def get_samples(self, *args, **kwargs):
            return {"lnK_C": jnp.array([0.0, 0.0])}

    def fake_run_mcmc(model_fn, X, y_log, cfg):
        captured["X_shape"] = tuple(X.shape)
        captured["y_log"] = y_log
        return FakeMCMC()

    def fake_mu_draws(logmodel, mcmc, X, param_names):
        return jnp.stack([captured["y_log"], captured["y_log"]])

    monkeypatch.setattr(pipeline, "run_mcmc", fake_run_mcmc)
    monkeypatch.setattr(pipeline, "diagnostics", lambda mcmc: {"max_rhat": 1.0})
    monkeypatch.setattr(pipeline, "mu_draws", fake_mu_draws)
    monkeypatch.setattr(pipeline.az, "from_numpyro", lambda mcmc: "idata-c")

    ds = Dataset([_row(C_at_frac=0.006), _row(T_K=983.15, p_MMS=0.005, C_at_frac=0.008)])
    out = run_class_calibration(
        Config(),
        ds=ds,
        chem_class=ChemClass.SIGEC,
        reference_reactor="XYZ_tool_1",
    )

    assert captured["X_shape"] == (2, 10)
    assert out["report"]["carbon_model_trained"] is True
    assert out["report"]["R2_C"] == pytest.approx(1.0)
    assert out["idata_c"] == "idata-c"
