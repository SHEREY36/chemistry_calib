"""
CFD-ACE+ input deck writer / output parser. OUT OF SCOPE for Phases 0-8
(STOP GATE in build_steps_and_cfd_integration.md). Left as a spec'd stub.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CFDCondition:
    T_set_K: float
    flows_sccm: dict
    P_tot_torr: float
    geometry_id: str


@dataclass
class CFDResult:
    condition: CFDCondition
    surface_p_i: dict
    surface_T_K: float
    GR_profile_r: np.ndarray
    Ge_profile_r: np.ndarray


def write_cfd_deck(cond: CFDCondition, path: str) -> None:
    raise NotImplementedError("Phase 9: not started (STOP GATE after Phase 8)")


def parse_cfd_output(path: str) -> CFDResult:
    raise NotImplementedError("Phase 9: not started (STOP GATE after Phase 8)")
