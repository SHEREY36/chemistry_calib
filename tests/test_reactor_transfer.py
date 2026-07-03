"""Phase 7 tests: cross-reactor validation. theta_chem is frozen at its
Phase 4 DS1 posterior mean; only a low-dimensional delta_r is fit per
reactor (build_steps_and_cfd_integration.md Phase 7.2)."""
import pytest

from chem_ml.config import Config
from chem_ml.pipeline import run_phase4_calibration, run_phase7_cross_reactor


@pytest.fixture(scope="module")
def phase7_result():
    cfg = Config()
    p4 = run_phase4_calibration(cfg)
    return run_phase7_cross_reactor(cfg, p4)


def test_delta_r_mcmc_converged(phase7_result):
    for name in ("ds3", "ds4"):
        diag = phase7_result[f"diag_{name}"]
        assert diag["max_rhat"] < 1.01
        assert diag["n_diverging"] == 0


def test_ds3_gr_reproduces_papers_own_limitation(phase7_result):
    """Paper's own DS3 GR fit is R^2=0.844 (Eq. 11) -- this is the dataset
    Fig. 1's Regime-I curvature comes from, so a comparably modest R^2 here
    (not near-DS1's 0.98) is the CORRECT reproduction, not a failure."""
    assert phase7_result["report"]["DS3_GR_within_band"]


def test_ds3_ge_recovered_with_low_dim_delta_r(phase7_result):
    """Paper: Eq. 16, R^2=0.994."""
    assert phase7_result["report"]["DS3_Ge_within_band"]


def test_ds4_ge_recovered_with_low_dim_delta_r(phase7_result):
    assert phase7_result["report"]["DS4_Ge_within_band"]


def test_delta_r_is_low_dimensional(phase7_result):
    """Phase 7.2's whole point: portability with ~3-5 delta_r params, not a
    full re-fit."""
    assert phase7_result["report"]["DS3_n_delta_r_params"] <= 5
    assert phase7_result["report"]["DS4_n_delta_r_params"] <= 5


def test_overall_cross_reactor_gate_passes(phase7_result):
    assert phase7_result["report"]["PASS"], phase7_result["report"]
