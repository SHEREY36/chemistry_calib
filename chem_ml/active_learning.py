"""
Active-learning loop to minimize CFD-ACE+ runs (Phase 10).

ML method: Gaussian Process regression (scikit-learn) as a cheap surrogate
over the CFD input space. This is the ONE place in the whole pipeline a GP
is used -- Phases 0-8 use MCMC (Bayesian calibration of the chemistry) and a
small NN (residual correction, Phase 5); this phase specifically needs a
surrogate that's cheap to query many times over a continuous 4D space and
gives calibrated predictive uncertainty for free, which is exactly what a
GP is for and neither MCMC nor a point-estimate NN naturally gives you
without extra machinery.

No CFD-ACE+ license exists in this environment, so this module is validated
against a SYNTHETIC stand-in function (tests/test_active_learning.py), not
real CFD output. Swapping in real CFDResult objects via `ingest()` requires
no code change here -- only real data from an actual CFD-ACE+ run.

Loop (Phase 10.1-10.5): Sobol seed -> GP surrogate over CFD inputs ->
GP-variance acquisition (D-optimal-flavored, cost-weighted) -> batch of k
with a diversity penalty -> run CFD -> ingest -> repeat until predictive
uncertainty < tol or the run budget (Config.cfd_run_budget) is hit.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from chem_ml.cfd.io import CFDCondition, CFDResult
from chem_ml.cfd.transfer import extract_transfer_priors
from chem_ml.config import Config

log = logging.getLogger("chem_ml")

# x_cfd = [T_set_K, HCl_over_DCS, GeH4_over_DCS, P_tot_torr] -- the 4 knobs
# a CFD-ACE+ input deck (cfd/io.py:CFDCondition) actually varies.
CFD_INPUT_DIMS = ["T_set_K", "HCl_over_DCS", "GeH4_over_DCS", "P_tot_torr"]
# y_cfd = [ln_alpha_HCl, ln_alpha_GeH4, dT_r_K] -- the reactor-transfer
# quantities extract_transfer_priors() derives from one CFD run; these (not
# raw GR/Ge) are what the surrogate targets, since they're what Phase 7
# actually needs a prior on.
CFD_OUTPUT_DIMS = ["ln_alpha_HCl", "ln_alpha_GeH4", "dT_r_K"]


class ActiveLearner:
    def __init__(self, cfg: Config, bounds: np.ndarray, geometry_id: str = "AMAT_3D_v1",
                dcs_flow_sccm: float = 10.0):
        """`bounds`: (4, 2) array of [lo, hi] per CFD_INPUT_DIMS, e.g.
        [[873, 1053], [0.1, 0.9], [0.01, 0.09], [5, 20]] for a 600-780C,
        HCl/DCS 0.1-0.9, GeH4/DCS 0.01-0.09, 5-20 Torr envelope (DS1's own
        range, as a reasonable default search space when nothing else is
        specified)."""
        self.cfg = cfg
        self.bounds = np.asarray(bounds, dtype=float)
        self.geometry_id = geometry_id
        self.dcs_flow_sccm = dcs_flow_sccm
        kernel = (ConstantKernel(1.0, (1e-2, 1e3)) * RBF(length_scale=np.ones(4), length_scale_bounds=(1e-2, 1e2))
                 + WhiteKernel(1e-3, noise_level_bounds=(1e-4, 1e1)))
        self.gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=3)
        self.X_done: list[np.ndarray] = []
        self.Y_done: list[np.ndarray] = []
        self.done: list[CFDResult] = []
        self._fitted = False

    # ---- input-space bookkeeping -------------------------------------------
    def _normalize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])

    def _row_to_condition(self, row: np.ndarray) -> CFDCondition:
        T_set, hcl_ratio, geh4_ratio, p_tot = row
        flows = {"DCS": self.dcs_flow_sccm, "HCl": float(hcl_ratio * self.dcs_flow_sccm),
                 "GeH4": float(geh4_ratio * self.dcs_flow_sccm), "B2H6": 0.0}
        return CFDCondition(T_set_K=float(T_set), flows_sccm=flows, P_tot_torr=float(p_tot),
                            geometry_id=self.geometry_id)

    def _condition_to_row(self, cond: CFDCondition) -> np.ndarray:
        return np.array([cond.T_set_K, cond.flows_sccm["HCl"] / cond.flows_sccm["DCS"],
                         cond.flows_sccm["GeH4"] / cond.flows_sccm["DCS"], cond.P_tot_torr])

    # ---- Phase 10.1: space-filling seed -------------------------------------
    def seed(self, n: int, seed: int = 0) -> list[CFDCondition]:
        """Sobol (not uniform-random) space-filling seed of n conditions --
        important when n is a handful of expensive runs and even coverage
        matters more than i.i.d. randomness."""
        sampler = qmc.Sobol(d=4, scramble=True, seed=seed)
        unit = sampler.random(n)
        scaled = qmc.scale(unit, self.bounds[:, 0], self.bounds[:, 1])
        return [self._row_to_condition(row) for row in scaled]

    # ---- Phase 10.2: fit/update the surrogate ------------------------------
    def ingest(self, results: list[CFDResult]) -> None:
        """Refit the GP on every CFD result seen so far. Each CFDResult ->
        one (x_cfd, y_cfd) training pair via extract_transfer_priors on
        that single run (its own radial spread sets that pair's implicit
        noise level, folded into the GP's WhiteKernel term across all
        pairs rather than per-point -- a per-point heteroscedastic GP
        would be a reasonable upgrade once there's enough real CFD data to
        justify the extra complexity)."""
        for res in results:
            priors = extract_transfer_priors([res])
            y = np.array([priors["ln_alpha_HCl"][0], priors["ln_alpha_GeH4"][0],
                         priors["dT_r_K_diagnostic_only"][0]])
            self.X_done.append(self._condition_to_row(res.condition))
            self.Y_done.append(y)
        self.done.extend(results)
        if len(self.X_done) >= 2:
            self.gp.fit(self._normalize(np.stack(self.X_done)), np.stack(self.Y_done))
            self._fitted = True

    # ---- Phase 10.3: acquisition -------------------------------------------
    def acquisition(self, candidates: list[CFDCondition],
                    cost_fn: Optional[Callable[[CFDCondition], float]] = None) -> np.ndarray:
        """Score each candidate by total GP predictive variance across the 3
        output dims -- a D-optimal-flavored proxy for expected information
        gain (high variance = the surrogate is unsure here = a real CFD run
        here teaches the model the most), divided by a per-condition cost
        if `cost_fn` is given (3D CFD runs are not equal cost: higher
        pressure or lower T can need finer meshing / more solver
        iterations to converge)."""
        X = self._normalize(np.stack([self._condition_to_row(c) for c in candidates]))
        if not self._fitted:
            scores = np.ones(len(candidates))  # no data yet -> everything equally informative
        else:
            _, std = self.gp.predict(X, return_std=True)
            scores = np.sum(std ** 2, axis=1)
        if cost_fn is not None:
            scores = scores / np.array([cost_fn(c) for c in candidates])
        return scores

    # ---- Phase 10.4: batch selection with diversity ------------------------
    def select_batch(self, candidates: list[CFDCondition], k: Optional[int] = None,
                     cost_fn: Optional[Callable[[CFDCondition], float]] = None) -> list[CFDCondition]:
        """Greedy top-k with local-penalization diversity (Gonzalez et al.
        batch-BO style): after picking a candidate, downweight nearby
        candidates in normalized input space so the batch doesn't cluster
        on the one corner of the space that happens to score highest."""
        k = k or self.cfg.al_batch_size
        X_norm = self._normalize(np.stack([self._condition_to_row(c) for c in candidates]))
        scores = self.acquisition(candidates, cost_fn=cost_fn).copy()
        chosen_idx: list[int] = []
        remaining = set(range(len(candidates)))
        length_scale = 0.15  # in normalized [0,1]^4 units
        for _ in range(min(k, len(candidates))):
            best = max(remaining, key=lambda i: scores[i])
            chosen_idx.append(best)
            remaining.discard(best)
            for j in list(remaining):
                d2 = float(np.sum((X_norm[j] - X_norm[best]) ** 2))
                penalty = np.exp(-d2 / (2 * length_scale ** 2))
                scores[j] *= (1.0 - 0.8 * penalty)  # soft repulsion, not exclusion
        return [candidates[i] for i in chosen_idx]

    def step(self, candidate_pool: list[CFDCondition],
            cost_fn: Optional[Callable[[CFDCondition], float]] = None) -> list[CFDCondition]:
        return self.select_batch(candidate_pool, cost_fn=cost_fn)

    def budget_remaining(self) -> int:
        return self.cfg.cfd_run_budget - len(self.done)

    # ---- Phase 10.5: stopping criterion ------------------------------------
    def mean_predictive_std(self, candidates: list[CFDCondition]) -> float:
        """Average GP predictive std across a candidate pool -- the
        stopping-criterion signal: stop when this falls below tolerance or
        the run budget is hit."""
        if not self._fitted:
            return float("inf")
        X = self._normalize(np.stack([self._condition_to_row(c) for c in candidates]))
        _, std = self.gp.predict(X, return_std=True)
        return float(np.mean(std))
