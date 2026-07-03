"""
SpeciesRegistry: single source of truth for every precursor/dopant/carrier
the model can use. Adding a species = adding an entry here (Phase 2.1).
No code change elsewhere.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

log = logging.getLogger("chem_ml")


class Role(str, Enum):
    SI_SOURCE = "Si-source"
    GE_SOURCE = "Ge-source"
    C_SOURCE = "C-source"
    DOPANT = "dopant"
    SELECTIVITY = "selectivity-agent"
    CARRIER = "carrier"
    BYPRODUCT = "byproduct"


@dataclass(frozen=True)
class Species:
    canonical_name: str
    formula: str
    role: Role
    family: str                # 'hydride' | 'chlorinated' | 'germane' | 'dopant' | ...
    n_Si: int = 0
    n_Ge: int = 0
    n_C: int = 0
    n_Cl: int = 0
    n_H: int = 0
    produces_HCl: bool = False
    default_prior: Optional[dict] = None  # prior on delivery/decomp params for new species


class SpeciesRegistry:
    """Single source of truth for every precursor/dopant/carrier the model can use.
    Adding a species = adding an entry here (Phase 2.1). No code change elsewhere."""

    def __init__(self) -> None:
        self._db: dict[str, Species] = {}
        self._seed_defaults()

    def _seed_defaults(self) -> None:
        for sp in [
            Species("dichlorosilane", "SiH2Cl2", Role.SI_SOURCE, "chlorinated",
                    n_Si=1, n_Cl=2, n_H=2, produces_HCl=True),
            Species("silane", "SiH4", Role.SI_SOURCE, "hydride", n_Si=1, n_H=4),
            Species("disilane", "Si2H6", Role.SI_SOURCE, "hydride", n_Si=2, n_H=6),
            Species("trisilane", "Si3H8", Role.SI_SOURCE, "hydride", n_Si=3, n_H=8),
            Species("trichlorosilane", "SiHCl3", Role.SI_SOURCE, "chlorinated",
                    n_Si=1, n_Cl=3, n_H=1, produces_HCl=True),
            Species("germane", "GeH4", Role.GE_SOURCE, "germane", n_Ge=1, n_H=4),
            Species("hcl", "HCl", Role.SELECTIVITY, "chlorinated", n_Cl=1, n_H=1,
                    produces_HCl=True),
            Species("diborane", "B2H6", Role.DOPANT, "dopant", n_H=6),
            Species("phosphine", "PH3", Role.DOPANT, "dopant", n_H=3),
            Species("hydrogen", "H2", Role.CARRIER, "carrier", n_H=2),
        ]:
            self._db[sp.canonical_name] = sp

    def get(self, name: str) -> Species:
        if name not in self._db:
            raise KeyError(f"Unknown species '{name}'. Add it to the registry (Phase 2.1).")
        return self._db[name]

    def add(self, sp: Species) -> None:
        if sp.canonical_name in self._db:
            raise ValueError(f"Species {sp.canonical_name} already registered")
        self._db[sp.canonical_name] = sp
        log.info("Registered new species: %s (%s, family=%s)", sp.canonical_name, sp.formula, sp.family)

    def by_role(self, role: Role) -> list[Species]:
        return [s for s in self._db.values() if s.role == role]
