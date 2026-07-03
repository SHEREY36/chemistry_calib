"""
Active-learning loop to minimize CFD-ACE+ runs (Phase 10). OUT OF SCOPE for
Phases 0-8 (STOP GATE in build_steps_and_cfd_integration.md).
"""
from __future__ import annotations

import numpy as np

from chem_ml.cfd.io import CFDCondition, CFDResult
from chem_ml.config import Config


class ActiveLearner:
    """
    Loop: Sobol seed -> GP surrogate over CFD input space -> D-optimal/EIG
    acquisition (with cost) -> batch of k -> run CFD -> update -> repeat until
    delta_r posterior uncertainty < tol OR budget hit.
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.gp = None
        self.done: list[CFDResult] = []

    def seed(self, n: int) -> list[CFDCondition]:
        raise NotImplementedError("Phase 10: not started (STOP GATE after Phase 8)")

    def acquisition(self, candidates: list[CFDCondition]) -> np.ndarray:
        raise NotImplementedError("Phase 10: not started (STOP GATE after Phase 8)")

    def select_batch(self, candidates: list[CFDCondition]) -> list[CFDCondition]:
        raise NotImplementedError("Phase 10: not started (STOP GATE after Phase 8)")

    def step(self, candidate_pool: list[CFDCondition]) -> list[CFDCondition]:
        batch = self.select_batch(candidate_pool)
        return batch

    def ingest(self, results: list[CFDResult]) -> None:
        self.done.extend(results)

    def budget_remaining(self) -> int:
        return self.cfg.cfd_run_budget - len(self.done)
