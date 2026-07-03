"""
Small, strongly-regularized residual MLP: y_log = f_phys + g_NN(phi; x).
Gated hard by declared ChemClass so it never contaminates across chemistries
(INVARIANT 3). Implemented in Phase 5 -- see build_steps_and_cfd_integration.md.
"""
from __future__ import annotations

from chem_ml.schema import ChemClass


class ResidualNN:  # placeholder interface, filled in during Phase 5
    def __init__(self, chem_class: ChemClass, n_out: int = 3):
        self.chem_class = chem_class
        self.n_out = n_out
        self.params = None  # set by fit

    def __call__(self, X_full):
        raise NotImplementedError("Phase 5: implement gated residual MLP")

    def fit(self, X_full, residual_targets, l2: float = 1e-2):
        raise NotImplementedError("Phase 5: implement gated residual MLP")
