"""Smoke tests for figure generation -- doesn't check pixel content, just
that every plot function runs against the real fitted model and produces
a non-empty PNG (regression guard against the parity/sensitivity/
calibration plotting code silently breaking)."""
from pathlib import Path

import pytest

from chem_ml.config import Config
from chem_ml.pipeline import run_phase4_calibration
from chem_ml.plots import generate_all_figures


@pytest.fixture(scope="module")
def figures(tmp_path_factory):
    cfg = Config()
    p4 = run_phase4_calibration(cfg)
    import chem_ml.plots as plots_mod
    out_dir = tmp_path_factory.mktemp("figures")
    plots_mod.FIG_DIR = out_dir
    calib = generate_all_figures(cfg, p4)
    return out_dir, calib


def test_all_figures_written_and_nonempty(figures):
    out_dir, _ = figures
    for name in ("fig2_gr_parity.png", "fig3_ge_parity.png", "fig4_gr_sensitivity.png",
                 "fig5_ge_sensitivity.png", "uncertainty_calibration.png"):
        p = out_dir / name
        assert p.exists(), f"{name} was not written"
        assert p.stat().st_size > 1000, f"{name} looks empty/truncated"


def test_calibration_is_not_wildly_overconfident(figures):
    """The dangerous failure mode for UQ is overconfidence (empirical
    coverage far BELOW nominal, i.e. intervals too narrow). Guard against
    that broad regression rather than pinning exact coverage numbers."""
    _, calib = figures
    for lev, cov in zip(calib["levels"], calib["gr_coverage"]):
        assert cov >= lev - 0.15, f"GR model overconfident at level {lev}: coverage {cov}"
    for lev, cov in zip(calib["levels"], calib["ge_coverage"]):
        assert cov >= lev - 0.15, f"Ge model overconfident at level {lev}: coverage {cov}"
