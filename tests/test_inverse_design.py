"""Phase 8 tests: inverse design with confidence-gated refusal."""
import jax.numpy as jnp
import pytest

from chem_ml.config import Config
from chem_ml.inverse_design import posterior_predictive_variance, stack_theta_samples
from chem_ml.pipeline import run_phase4_calibration, run_phase8_inverse_design


def test_posterior_predictive_variance_does_not_pool_across_dims():
    """Regression test: an earlier version pooled both output dims into one
    jnp.var(), so a large FIXED offset between the two channels (here: 100
    vs 0) would swamp the result even with zero actual per-channel spread."""
    def forward_log(theta, X):
        # two outputs with a large fixed offset between them, but each is
        # CONSTANT across samples (theta["a"] doesn't affect the output) ->
        # true per-dimension variance is exactly 0.
        return jnp.array([[100.0, 0.0]])

    theta_stacked = stack_theta_samples([{"a": 1.0}, {"a": 2.0}, {"a": 3.0}])
    u = posterior_predictive_variance(forward_log, theta_stacked, jnp.zeros(4))
    assert float(u) == pytest.approx(0.0, abs=1e-9)


@pytest.fixture(scope="module")
def phase4():
    return run_phase4_calibration(Config())


def test_in_range_target_is_recovered_and_accepted(phase4):
    """Target matches DS1 run #70 (Tg=725C, GR=29.3, Ge=21.73%) almost
    exactly -- the optimizer should land on it with small error and accept."""
    cfg = Config()
    r = run_phase8_inverse_design(cfg, phase4, target_gr_nm_min=29.3, target_ge_frac=0.2173)
    assert r["gr_rel_error"] < 0.05
    assert r["ge_abs_error"] < 0.02
    assert r["accepted"]
    assert not r["at_feasible_boundary"]


def test_extreme_target_is_refused(phase4):
    """GR=500 nm/min at Ge=60% simultaneously is far outside anything DS1
    observed (max DS1 GR is 128 nm/min, max Ge is 40.9%) -- must be flagged,
    not silently extrapolated."""
    cfg = Config()
    r = run_phase8_inverse_design(cfg, phase4, target_gr_nm_min=500.0, target_ge_frac=0.60)
    assert not r["accepted"]
    assert r["low_confidence"]


def test_degenerate_b2h6_column_does_not_trigger_false_boundary_flag(phase4):
    """Regression test: DS1 has no B2H6 (that feature column is constant 0,
    so lo==hi==0 there); the boundary check must not treat every solution as
    'pinned' just because the unused B2H6 column trivially equals its own
    degenerate box."""
    cfg = Config()
    r = run_phase8_inverse_design(cfg, phase4, target_gr_nm_min=29.3, target_ge_frac=0.2173)
    assert r["recipe"]["p_B2H6_over_pDCS"] == pytest.approx(1.0)  # exp(0) = untouched
    assert not r["at_feasible_boundary"]
