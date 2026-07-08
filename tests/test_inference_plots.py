"""Smoke tests for the inference-time visualization suite (posterior
structure, single-query credible intervals, extrapolation comparison)."""
import pytest

from chem_ml.config import Config
from chem_ml.pipeline import run_phase4_calibration
from chem_ml.inference_plots import (
    plot_bayesian_response_envelope, plot_credible_interval_for_query,
    plot_extrapolation_superiority, plot_posterior_pairplot,
    plot_process_window_probability_map, plot_trace,
)


@pytest.fixture(scope="module")
def phase4():
    return run_phase4_calibration(Config())


def _assert_written(p):
    assert p.exists() and p.stat().st_size > 1000


def test_posterior_pairplot_and_trace(tmp_path, phase4):
    p1, p2 = tmp_path / "pair.png", tmp_path / "trace.png"
    plot_posterior_pairplot(phase4, p1)
    plot_trace(phase4, p2)
    _assert_written(p1)
    _assert_written(p2)


def test_credible_interval_query_returns_sane_interval(tmp_path, phase4):
    out = tmp_path / "ci.png"
    summary = plot_credible_interval_for_query(phase4, Config(), T_C=725.0, hcl_ratio=0.5,
                                               geh4_ratio=0.03, out=out)
    _assert_written(out)
    assert summary["p5"] < summary["p50"] < summary["p95"]
    assert summary["p50"] > 0


def test_extrapolation_plot_shows_physics_band_growing_and_rf_frozen(tmp_path, phase4, caplog):
    import logging
    out = tmp_path / "extrap.png"
    with caplog.at_level(logging.INFO, logger="chem_ml"):
        plot_extrapolation_superiority(phase4, out)
    _assert_written(out)
    # The whole point of this plot: physics CI keeps widening past the
    # training range, the RF's naive inter-tree band does not.
    growth_msg = next(r for r in caplog.records if "Band width AT the training edge" in r.message)
    assert "frozen" in growth_msg.message


def test_bayesian_response_envelope_shows_extrapolation_uncertainty(tmp_path, phase4):
    out = tmp_path / "response_envelope.png"
    summary = plot_bayesian_response_envelope(phase4, out)
    _assert_written(out)
    assert summary["relative_width_growth_750C"] > 1.0
    assert summary["max_train_geh4"] > 0


def test_process_window_probability_map_finds_candidate_region(tmp_path, phase4):
    out = tmp_path / "window_map.png"
    summary = plot_process_window_probability_map(phase4, out)
    _assert_written(out)
    assert summary["max_target_probability"] > 0.05
    assert summary["best_hcl_ratio"] > 0
    assert summary["best_geh4_ratio"] > 0
