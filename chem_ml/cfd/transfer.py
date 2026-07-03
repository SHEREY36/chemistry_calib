"""
Extract the setpoint->surface map from CFD-ACE+ runs, as priors on delta_r.
OUT OF SCOPE for Phases 0-8 (STOP GATE in build_steps_and_cfd_integration.md).
"""
from __future__ import annotations

from chem_ml.cfd.io import CFDResult


def extract_transfer_priors(results: list[CFDResult]) -> dict:
    raise NotImplementedError("Phase 9: not started (STOP GATE after Phase 8)")
