"""
ReactionNetworkAssembler: given a declared class + species list, returns the
set of active model terms. Absent species contribute nothing -- enforced
structurally (INVARIANT 2 / anti-contamination guarantee, Phase 2.2/2.3).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from chem_ml.registry import Role, Species, SpeciesRegistry
from chem_ml.schema import ChemClass, Mode


@dataclass
class ActiveNetwork:
    """The set of active model terms for a declared recipe. Absent species are
    simply not here -> zero dependence by construction (INVARIANT 2)."""
    chem_class: ChemClass
    si_source: Species
    ge_source: Optional[Species]
    c_source: Optional[Species]
    dopant: Optional[Species]
    selectivity_agent: Optional[Species]
    has_chlorine: bool
    mode: Mode

    @property
    def uses_GR_model(self) -> bool:
        return True

    @property
    def uses_Ge_model(self) -> bool:
        return self.ge_source is not None

    @property
    def uses_C_model(self) -> bool:
        return self.c_source is not None or self.chem_class in (ChemClass.SIGEC, ChemClass.SIGEC_X)

    @property
    def uses_B_model(self) -> bool:
        return self.dopant is not None and self.dopant.formula == "B2H6"


class ReactionNetworkAssembler:
    """Given a declared class + species list, returns the ActiveNetwork.
    This is the anti-contamination guarantee (Phase 2.2/2.3)."""

    def __init__(self, registry: SpeciesRegistry) -> None:
        self.reg = registry

    def assemble(self, chem_class: ChemClass, species_names: Sequence[str], mode: Mode) -> ActiveNetwork:
        species = [self.reg.get(n) for n in species_names]
        si = _first(species, Role.SI_SOURCE)
        ge = _first(species, Role.GE_SOURCE, required=False)
        c = _first(species, Role.C_SOURCE, required=False)
        dop = _first(species, Role.DOPANT, required=False)
        sel = _first(species, Role.SELECTIVITY, required=False)
        if si is None:
            raise ValueError("Every recipe needs a Si source.")
        has_cl = any(s.produces_HCl for s in species)
        return ActiveNetwork(chem_class, si, ge, c, dop, sel, has_cl, mode)


def _first(species: Sequence[Species], role: Role, required: bool = True) -> Optional[Species]:
    for s in species:
        if s.role == role:
            return s
    if required:
        return None
    return None
