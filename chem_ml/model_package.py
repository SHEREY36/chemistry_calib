"""Production model package for chamber-agnostic epitaxy chemistry.

This module is the public boundary the refactor pivots around: one
species/target-driven architecture, observable slots for GR/composition/
dopant incorporation, and optional bounded log-space residual networks that
can be transcribed into a deterministic CFD-ACE+ UDF.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import numpy as np

from chem_ml.assembler import ReactionNetworkAssembler
from chem_ml.registry import SpeciesRegistry
from chem_ml.schema import ChemClass, Mode, canonical_chem_class


class Observable(str, Enum):
    GR = "GR"
    GE = "Ge"
    C = "C"
    DOPANT = "dopant"


RESIDUAL_INPUT_NAMES = (
    "invT_std",
    "ln_HCl",
    "ln_GeH4",
    "ln_B2H6",
    "ln_C_source",
    "ln_dopant",
    "ln_H2",
    "ln_N2",
    "XT_H2_minus_N2_scaled",
    "pattern_density",
    "raw_HCl",
    "raw_GeH4",
    "raw_B2H6",
)


def default_species_for_chem_class(chem_class: ChemClass) -> tuple[str, ...]:
    """Conservative default species for clone-and-train CLI workflows."""
    canonical = canonical_chem_class(chem_class)
    species = ["dichlorosilane"]
    if canonical in (ChemClass.SIGE, ChemClass.SIGE_X, ChemClass.SIGEC, ChemClass.SIGEC_X):
        species.append("germane")
    if canonical in (ChemClass.SIGEC, ChemClass.SIGEC_X):
        species.append("methylsilane")
    if canonical in (ChemClass.SI_X, ChemClass.SIGE_X, ChemClass.SIGEC_X):
        species.append("diborane")
    species.extend(["hcl", "hydrogen"])
    return tuple(dict.fromkeys(species))


@dataclass(frozen=True)
class ChemistryModelSpec:
    """Architecture selection for one calibrated chemistry package.

    The architecture is shared across chemistries; the enabled observable
    slots and fitted parameter values are chemistry-specific.
    """

    chem_class: ChemClass
    target_deposit: str
    species_names: tuple[str, ...]
    enabled_observables: tuple[Observable, ...]
    si_source: str
    ge_source: str | None = None
    c_source: str | None = None
    dopant: str | None = None
    selectivity_agent: str | None = None
    mode: Mode = Mode.BLANKET


def build_model_spec(
    chem_class: ChemClass,
    species_names: Sequence[str],
    target_deposit: str | None = None,
    mode: Mode = Mode.BLANKET,
    registry: SpeciesRegistry | None = None,
) -> ChemistryModelSpec:
    """Build the shared model architecture from declared species.

    This is the anti-duplication layer: Si, SiGe, SiGeC, and doped variants
    use the same model object and simply enable different observable slots.
    """
    reg = registry or SpeciesRegistry()
    network = ReactionNetworkAssembler(reg).assemble(chem_class, species_names, mode)
    canonical = canonical_chem_class(chem_class)
    observables: list[Observable] = [Observable.GR]
    if network.uses_Ge_model or canonical in (ChemClass.SIGE, ChemClass.SIGE_X, ChemClass.SIGEC, ChemClass.SIGEC_X):
        observables.append(Observable.GE)
    if network.uses_C_model or canonical in (ChemClass.SIGEC, ChemClass.SIGEC_X):
        observables.append(Observable.C)
    if network.dopant is not None or canonical in (ChemClass.SI_X, ChemClass.SIGE_X, ChemClass.SIGEC_X):
        observables.append(Observable.DOPANT)

    return ChemistryModelSpec(
        chem_class=canonical,
        target_deposit=target_deposit or canonical.value,
        species_names=tuple(species_names),
        enabled_observables=tuple(dict.fromkeys(observables)),
        si_source=network.si_source.canonical_name,
        ge_source=network.ge_source.canonical_name if network.ge_source else None,
        c_source=network.c_source.canonical_name if network.c_source else None,
        dopant=network.dopant.canonical_name if network.dopant else None,
        selectivity_agent=network.selectivity_agent.canonical_name if network.selectivity_agent else None,
        mode=mode,
    )


@dataclass
class BoundedResidualMLP:
    """Small tanh MLP for log-space residual correction.

    Output is bounded in log units, so the residual acts as a controlled
    multiplicative correction instead of a free replacement for the physics
    kernel.
    """

    observable: Observable
    input_names: tuple[str, ...]
    weights: list[np.ndarray]
    biases: list[np.ndarray]
    max_abs_log_correction: float = 0.5

    def __post_init__(self) -> None:
        if len(self.weights) != len(self.biases):
            raise ValueError("weights and biases must have the same number of layers")
        if not self.weights:
            raise ValueError("at least one layer is required")
        prev = len(self.input_names)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            w_arr = np.asarray(w, dtype=float)
            b_arr = np.asarray(b, dtype=float)
            if w_arr.ndim != 2 or b_arr.ndim != 1:
                raise ValueError("weights must be 2D and biases must be 1D")
            if w_arr.shape[1] != prev or w_arr.shape[0] != b_arr.shape[0]:
                raise ValueError(f"layer {i} has incompatible shapes")
            self.weights[i] = w_arr
            self.biases[i] = b_arr
            prev = b_arr.shape[0]
        if self.biases[-1].shape[0] != 1:
            raise ValueError("residual MLP must have a scalar output")
        if self.max_abs_log_correction <= 0:
            raise ValueError("max_abs_log_correction must be positive")

    @classmethod
    def zero(cls, observable: Observable, input_names: Sequence[str]) -> "BoundedResidualMLP":
        """A deterministic zero residual with the same UDF interface."""
        return cls(
            observable=observable,
            input_names=tuple(input_names),
            weights=[np.zeros((1, len(tuple(input_names))))],
            biases=[np.zeros(1)],
            max_abs_log_correction=1.0,
        )

    def __call__(self, x: Sequence[float] | np.ndarray) -> float:
        h = np.asarray(x, dtype=float)
        if h.shape[-1] != len(self.input_names):
            raise ValueError(f"expected {len(self.input_names)} residual inputs, got {h.shape[-1]}")
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            h = np.tanh(w @ h + b)
        raw = float((self.weights[-1] @ h + self.biases[-1])[0])
        return float(self.max_abs_log_correction * np.tanh(raw / self.max_abs_log_correction))

    def to_jsonable(self) -> dict:
        return {
            "observable": self.observable.value,
            "input_names": list(self.input_names),
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
            "max_abs_log_correction": self.max_abs_log_correction,
        }

    @classmethod
    def from_jsonable(cls, payload: dict) -> "BoundedResidualMLP":
        return cls(
            observable=Observable(payload["observable"]),
            input_names=tuple(payload["input_names"]),
            weights=[np.asarray(w, dtype=float) for w in payload["weights"]],
            biases=[np.asarray(b, dtype=float) for b in payload["biases"]],
            max_abs_log_correction=float(payload.get("max_abs_log_correction", 0.5)),
        )


@dataclass
class CalibratedChemistryModel:
    """Everything needed to evaluate or export one calibrated chemistry."""

    spec: ChemistryModelSpec
    theta: dict[Observable, dict[str, float]]
    invT_scaler: tuple[float, float]
    residuals: dict[Observable, BoundedResidualMLP] = field(default_factory=dict)
    training_source: str = "applied_experimental_data"
    transport_deembedding: str = "not_started"

    def enabled(self, observable: Observable) -> bool:
        return observable in self.spec.enabled_observables and observable in self.theta

    def residual_for(self, observable: Observable) -> BoundedResidualMLP | None:
        return self.residuals.get(observable)

    def to_jsonable(self) -> dict:
        return {
            "chem_class": self.spec.chem_class.value,
            "target_deposit": self.spec.target_deposit,
            "species": list(self.spec.species_names),
            "mode": self.spec.mode.value,
            "enabled_observables": [o.value for o in self.spec.enabled_observables],
            "theta": {obs.value: params for obs, params in self.theta.items()},
            "invT_scaler": list(self.invT_scaler),
            "residuals": {obs.value: r.to_jsonable() for obs, r in self.residuals.items()},
            "training_source": self.training_source,
            "transport_deembedding": self.transport_deembedding,
        }

    @classmethod
    def from_jsonable(cls, payload: dict) -> "CalibratedChemistryModel":
        spec = build_model_spec(
            ChemClass(payload["chem_class"]),
            payload["species"],
            target_deposit=payload.get("target_deposit"),
            mode=Mode(payload.get("mode", "blanket")),
        )
        return cls(
            spec=spec,
            theta={Observable(k): v for k, v in payload["theta"].items()},
            invT_scaler=tuple(payload["invT_scaler"]),
            residuals={
                Observable(k): BoundedResidualMLP.from_jsonable(v)
                for k, v in payload.get("residuals", {}).items()
            },
            training_source=payload.get("training_source", "applied_experimental_data"),
            transport_deembedding=payload.get("transport_deembedding", "not_started"),
        )
