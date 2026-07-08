"""Smoke tests for figure generation -- doesn't check pixel content, just
that every plot function runs against the real fitted model and produces
a non-empty PNG (regression guard against the parity/sensitivity/
calibration plotting code silently breaking)."""
from pathlib import Path

import pytest

from chem_ml.config import Config
from chem_ml.pipeline import run_phase4_calibration
from chem_ml.plots import generate_all_figures, _sensitivity_curves


@pytest.fixture(scope="module")
def figures(tmp_path_factory):
    cfg = Config()
    p4 = run_phase4_calibration(cfg)
    import chem_ml.plots as plots_mod
    out_dir = tmp_path_factory.mktemp("figures")
    plots_mod.FIG_DIR = out_dir
    calib = generate_all_figures(cfg, p4)
    return out_dir, calib, p4


def test_all_figures_written_and_nonempty(figures):
    out_dir, _, _ = figures
    for name in ("fig1_ds3_regime.png", "fig2_gr_parity.png", "fig3_ge_parity.png", "fig4_gr_sensitivity.png",
                 "fig5_ge_sensitivity.png", "uncertainty_calibration.png"):
        p = out_dir / name
        assert p.exists(), f"{name} was not written"
        assert p.stat().st_size > 1000, f"{name} looks empty/truncated"


def test_calibration_is_not_wildly_overconfident(figures):
    """The dangerous failure mode for UQ is overconfidence (empirical
    coverage far BELOW nominal, i.e. intervals too narrow). Guard against
    that broad regression rather than pinning exact coverage numbers."""
    _, calib, _ = figures
    for lev, cov in zip(calib["levels"], calib["gr_coverage"]):
        assert cov >= lev - 0.15, f"GR model overconfident at level {lev}: coverage {cov}"
    for lev, cov in zip(calib["levels"], calib["ge_coverage"]):
        assert cov >= lev - 0.15, f"Ge model overconfident at level {lev}: coverage {cov}"


def test_fig45_sensitivity_curves_are_in_paper_like_units(figures):
    """Guard against regressing to raw ratio-space derivatives, which are
    orders of magnitude larger than Tomasini's sccm axes."""
    _, _, p4 = figures
    rows = _sensitivity_curves(p4)
    row_750 = [r for r in rows if r["T_C"] == 750.0][0]

    assert 0.5 <= abs(row_750["dGR_dHCl_sccm"]) <= 2.0
    assert 0.5 <= abs(row_750["dGR_dGeH4_10pct_sccm"]) <= 3.0
    assert 1.0 <= row_750["dGR_dT"] <= 3.0
    assert 0.05 <= row_750["dGe_dGeH4_10pct_sccm"] <= 0.3
