"""
Export the calibrated power-law mechanism to CFD-ACE+ surface-reaction cards.
OUT OF SCOPE for Phases 0-8 (STOP GATE in build_steps_and_cfd_integration.md).
Left as a spec'd stub so Phase 9 has a slot; do not implement before Phase 8
passes.
"""
from __future__ import annotations


def export_mechanism_to_cfd(theta_posterior_mean: dict, path: str) -> None:
    raise NotImplementedError("Phase 9: not started (STOP GATE after Phase 8)")
