"""Phase 6 tests: identifiability (posterior covariance / Fisher) and
sensitivity derivatives reproducing Tomasini Fig. 4."""
import numpy as np
import pytest

from chem_ml.config import Config
from chem_ml.pipeline import run_phase4_calibration, run_phase6_identifiability


@pytest.fixture(scope="module")
def phase6_result():
    cfg = Config()
    p4 = run_phase4_calibration(cfg)
    return run_phase6_identifiability(cfg, p4)


def test_eigenspectrum_is_valid_covariance_spectrum(phase6_result):
    eigvals = np.asarray(phase6_result["eigvals"])
    assert eigvals.shape == (4,)
    assert np.all(eigvals > 0), "posterior covariance must be positive definite"
    assert np.all(np.diff(eigvals) >= 0), "eigenspectrum() must return ascending order"


def test_fisher_information_is_symmetric_psd(phase6_result):
    fisher = np.asarray(phase6_result["fisher"])
    assert fisher.shape == (4, 4)
    np.testing.assert_allclose(fisher, fisher.T, rtol=1e-8, atol=1e-8)
    eigvals = np.linalg.eigvalsh(fisher)
    assert np.all(eigvals >= -1e-6), "Fisher information must be PSD"


def test_dgr_dt_increases_with_temperature(phase6_result):
    """Physical sanity: GR's temperature sensitivity itself grows with T
    (matches the paper's own description of the derivative shrinking to
    ~25% per 50 K drop)."""
    table = phase6_result["report"]["sensitivity_table"]
    derivs = [row["dGR_dT_nm_min_per_K"] for row in table]
    assert derivs == sorted(derivs), "dGR/dT should increase monotonically with T"


def test_dgr_dt_at_750c_matches_paper_fig4(phase6_result):
    """Paper Fig. 4: |dGR/dT| in the 1-2 nm/min/K range at 750 C."""
    assert phase6_result["report"]["dGR_dT_in_paper_1_to_2_range"]


def test_reproduces_paper_gr_values_at_fig4_operating_points(phase6_result):
    """Independent check: at Fig. 4's own worked operating points
    (pHCl/pDCS=0.34, pGeH4/pDCS tuned to 20% Ge), the fitted model's GR
    should land within a factor of 2 of the paper's quoted values -- these
    exact points were not part of the Phase 4 fit target."""
    for row in phase6_result["report"]["sensitivity_table"]:
        ratio = row["GR_model_nm_min"] / row["GR_paper_nm_min"]
        assert 0.5 < ratio < 2.0, f"T={row['T_C']}: model/paper GR ratio {ratio}"
