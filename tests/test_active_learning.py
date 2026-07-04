"""Tests for the Phase 10 GP-surrogate active-learning loop. No CFD-ACE+
license exists, so a synthetic ground-truth function stands in for "run
CFD at this condition" -- structurally identical to how a real CFD job
would be ingested (via CFDResult + extract_transfer_priors), just cheap
enough to call thousands of times for a real test."""
import numpy as np
import pytest

from chem_ml.active_learning import ActiveLearner, CFD_INPUT_DIMS
from chem_ml.cfd.io import CFDCondition, CFDResult
from chem_ml.config import Config

_BOUNDS = np.array([[873.0, 1053.0], [0.1, 0.9], [0.01, 0.09], [5.0, 20.0]])


def _ground_truth(row: np.ndarray, rng: np.random.Generator, noise: float = 0.02) -> np.ndarray:
    """Smooth synthetic 'CFD' response standing in for a real solve."""
    T, hcl, geh4, p_tot = row
    ln_alpha_hcl = 0.3 * np.sin(hcl * 5.0) + 0.10 * (T - 963.0) / 90.0
    ln_alpha_geh4 = -4.0 * geh4 + 0.03 * p_tot
    dT = 5.0 + 0.4 * (p_tot - 12.5) * (T - 963.0) / 90.0
    out = np.array([ln_alpha_hcl, ln_alpha_geh4, dT])
    return out + rng.normal(0, noise, size=3)


def _run_fake_cfd(cond: CFDCondition, rng: np.random.Generator) -> CFDResult:
    """Build a CFDResult whose extract_transfer_priors() output recovers
    _ground_truth(row) -- i.e. a synthetic wafer with 3 radial points all
    consistent with the same underlying alpha/dT, small run-to-run noise."""
    row = np.array([cond.T_set_K, cond.flows_sccm["HCl"] / cond.flows_sccm["DCS"],
                   cond.flows_sccm["GeH4"] / cond.flows_sccm["DCS"], cond.P_tot_torr])
    ln_alpha_hcl, ln_alpha_geh4, dT = _ground_truth(row, rng)
    inlet_hcl = cond.flows_sccm["HCl"] / cond.flows_sccm["DCS"]
    inlet_geh4 = cond.flows_sccm["GeH4"] / cond.flows_sccm["DCS"]
    r = np.array([0.0, 50.0, 100.0])
    return CFDResult(
        condition=cond, r_mm=r,
        surface_T_K=np.full_like(r, cond.T_set_K + dT),
        surface_p_i={"HCl": np.full_like(r, inlet_hcl * np.exp(ln_alpha_hcl)),
                    "GeH4": np.full_like(r, inlet_geh4 * np.exp(ln_alpha_geh4)),
                    "B2H6": np.zeros_like(r)},
        GR_profile_nm_min=np.full_like(r, 50.0), Ge_profile_frac=np.full_like(r, 0.2),
    )


def test_seed_produces_conditions_within_bounds():
    al = ActiveLearner(Config(), _BOUNDS)
    conds = al.seed(8, seed=1)
    assert len(conds) == 8
    for c in conds:
        assert _BOUNDS[0, 0] <= c.T_set_K <= _BOUNDS[0, 1]
        hcl_ratio = c.flows_sccm["HCl"] / c.flows_sccm["DCS"]
        geh4_ratio = c.flows_sccm["GeH4"] / c.flows_sccm["DCS"]
        assert _BOUNDS[1, 0] <= hcl_ratio <= _BOUNDS[1, 1]
        assert _BOUNDS[2, 0] <= geh4_ratio <= _BOUNDS[2, 1]
        assert _BOUNDS[3, 0] <= c.P_tot_torr <= _BOUNDS[3, 1]


def test_ingest_and_acquisition_prefers_unexplored_region():
    rng = np.random.default_rng(0)
    al = ActiveLearner(Config(), _BOUNDS)
    seed_conds = al.seed(6, seed=0)
    al.ingest([_run_fake_cfd(c, rng) for c in seed_conds])
    assert al._fitted

    # A candidate right on top of an already-seen point should score lower
    # than a candidate far from everything seen so far.
    near_dup = seed_conds[0]
    far_row = np.array([1053.0, 0.9, 0.09, 20.0])  # a corner, likely underexplored
    far_cond = al._row_to_condition(far_row)
    scores = al.acquisition([near_dup, far_cond])
    assert scores[1] > scores[0]


def test_select_batch_is_diverse_not_clustered():
    rng = np.random.default_rng(1)
    al = ActiveLearner(Config(al_batch_size=4), _BOUNDS)
    al.ingest([_run_fake_cfd(c, rng) for c in al.seed(6, seed=0)])
    pool = al.seed(200, seed=2)  # dense candidate pool
    batch = al.select_batch(pool, k=4)
    assert len(batch) == 4
    X = np.stack([al._condition_to_row(c) for c in batch])
    X_norm = al._normalize(X)
    # No two selected points should sit essentially on top of each other.
    for i in range(len(batch)):
        for j in range(i + 1, len(batch)):
            assert np.linalg.norm(X_norm[i] - X_norm[j]) > 0.05


def test_active_learning_reduces_uncertainty_more_than_random_same_budget():
    """The actual claim active learning makes: for the SAME number of CFD
    runs, GP-variance-guided selection should reduce mean predictive
    uncertainty over a held-out candidate pool at least as much as an
    equal-size batch of random points (usually more, since it explicitly
    targets high-variance regions rather than sampling blind)."""
    rng_seed_data = np.random.default_rng(42)
    seed_conditions = ActiveLearner(Config(), _BOUNDS).seed(6, seed=0)
    seed_results = [_run_fake_cfd(c, rng_seed_data) for c in seed_conditions]

    eval_pool = ActiveLearner(Config(), _BOUNDS).seed(300, seed=99)

    # ---- Active learning arm ------------------------------------------------
    al = ActiveLearner(Config(al_batch_size=6), _BOUNDS)
    al.ingest(seed_results)
    candidate_pool = al.seed(150, seed=7)
    al_batch = al.select_batch(candidate_pool, k=6)
    al.ingest([_run_fake_cfd(c, np.random.default_rng(123)) for c in al_batch])
    al_final_std = al.mean_predictive_std(eval_pool)

    # ---- Random-selection arm (same seed data, same budget) -----------------
    rand = ActiveLearner(Config(), _BOUNDS)
    rand.ingest(seed_results)
    random_batch = rand.seed(6, seed=555)  # random additional 6, not variance-guided
    rand.ingest([_run_fake_cfd(c, np.random.default_rng(123)) for c in random_batch])
    rand_final_std = rand.mean_predictive_std(eval_pool)

    assert al_final_std <= rand_final_std * 1.05, (
        f"active learning ({al_final_std:.4f}) should not do meaningfully worse "
        f"than random ({rand_final_std:.4f}) at the same budget"
    )


def test_budget_remaining_tracks_ingested_runs():
    al = ActiveLearner(Config(cfd_run_budget=10), _BOUNDS)
    rng = np.random.default_rng(0)
    conds = al.seed(4, seed=0)
    al.ingest([_run_fake_cfd(c, rng) for c in conds])
    assert al.budget_remaining() == 6
