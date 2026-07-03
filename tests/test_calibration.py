"""Phase 4 tests: the core Tomasini reproduction gate
(build_steps_and_cfd_integration.md Phase 4.3). This is the headline
deliverable -- if this fails, everything downstream is suspect.
"""
import pytest

from chem_ml.config import Config
from chem_ml.pipeline import run_phase4_calibration


@pytest.fixture(scope="module")
def phase4_result():
    return run_phase4_calibration(Config())


def test_mcmc_converged(phase4_result):
    """Standard Bayesian-workflow convergence gate: R-hat<1.01, healthy ESS,
    and divergences under ~1% of total draws (a handful of divergences on a
    5-parameter model fit to DS2's 11-point boron table is expected small-N
    NUTS geometry, not a sign of a broken model -- see e.g. Stan/PyMC
    guidance on divergence rates; only >1% warrants reparametrizing)."""
    for name in ("gr", "ge", "b"):
        diag = phase4_result[f"diag_{name}"]
        cfg = Config()
        total_draws = cfg.mcmc.num_samples * cfg.mcmc.num_chains
        assert diag["max_rhat"] < 1.01, f"{name}: R-hat {diag['max_rhat']} >= 1.01"
        assert diag["n_diverging"] / total_draws < 0.01, (
            f"{name}: {diag['n_diverging']}/{total_draws} divergent transitions")
        assert diag["min_ess"] > 400, f"{name}: min ESS {diag['min_ess']} too low"


def test_gr_parity_matches_paper(phase4_result):
    """Paper Table 1, Eq. 10 (DS1, 10 Torr): R^2 = 0.985."""
    assert phase4_result["report"]["R2_GR"] >= 0.98


def test_ge_parity_matches_paper(phase4_result):
    """Paper Table 2, Eq. 15 (DS1, 10 Torr): R^2 = 0.988."""
    assert phase4_result["report"]["R2_Ge"] >= 0.98


def test_kappa_gr_matches_paper_within_10pct(phase4_result):
    """Paper's tabulated Ea/R = -24507 K for Eq. 10."""
    assert phase4_result["report"]["kappa_GR_within_10pct"]


def test_reaction_orders_match_paper(phase4_result):
    """Paper: gamma_HCl = -0.7, gamma_GeH4 = 1.3 (Eq. 10)."""
    r = phase4_result["report"]
    assert r["gamma_HCl"] == pytest.approx(-0.7, abs=0.15)
    assert r["gamma_GeH4"] == pytest.approx(1.3, abs=0.2)


def test_boron_scaling_matches_paper(phase4_result):
    """Paper Eq. 20 (DS2): [B] ~ p_B2H6^0.8."""
    assert phase4_result["report"]["beta_B2H6_within_target"]


def test_overall_acceptance_gate_passes(phase4_result):
    assert phase4_result["report"]["PASS"], phase4_result["report"]
