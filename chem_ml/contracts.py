"""Public request contracts for intent-based chemistry workflows.

These dataclasses are deliberately thin: they describe what the caller wants
to do, while ``workflows.py`` decides which existing scientific module handles
the work.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from chem_ml.cfd.io import CFDCondition
from chem_ml.schema import ChemClass, Mode


class DataKind(str, Enum):
    SCALAR = "scalar"
    SPATIAL_SCAN = "spatial_scan"
    CFD_PROFILE = "cfd_profile"


class TrainTarget(str, Enum):
    CHEMISTRY = "chemistry"
    REACTOR_TRANSFER = "reactor_transfer"
    SPATIAL_TRANSFER = "spatial_transfer"


class TrainStrategy(str, Enum):
    POOLED = "pooled"
    WARM_START = "warm_start"
    FROZEN_CHEMISTRY = "frozen_chemistry"


class ValidationSuite(str, Enum):
    REPRODUCTION = "reproduction"
    TRANSFER = "transfer"
    SPATIAL = "spatial"
    CFD_CONTRACT = "cfd_contract"
    ALL = "all"


@dataclass(frozen=True)
class RegisterExperimentRequest:
    kind: DataKind
    tag: str = ""
    reactor_id: str = ""
    chem_class: ChemClass = ChemClass.SIGE
    mode: Mode = Mode.BLANKET
    csv_path: Optional[str] = None
    runs_csv: Optional[str] = None
    points_csv: Optional[str] = None
    cfd_output_csv: Optional[str] = None
    cfd_condition: Optional[CFDCondition] = None


@dataclass(frozen=True)
class TrainRequest:
    target: TrainTarget
    strategy: TrainStrategy = TrainStrategy.POOLED
    csv_path: Optional[str] = None
    runs_csv: Optional[str] = None
    points_csv: Optional[str] = None
    reactor_id: str = ""
    reference_reactor: str = "ASM_Epsilon"
    chem_class: ChemClass = ChemClass.SIGE
    mode: Mode = Mode.BLANKET
    tag: str = ""
    widen_factor: float = 2.0
    include_registered: bool = True
    save_posteriors: bool = False
    use_benchmark_data: bool = False
    species_names: tuple[str, ...] = ()
    target_deposit: str = ""
    save_model_package: bool = False
    model_package_path: str = "data/processed/model_package.json"
    fit_residual_nn: bool = True
    residual_steps: int = 2000


@dataclass(frozen=True)
class ValidateRequest:
    suite: ValidationSuite
    tag: str = ""
    cfd_output_csv: Optional[str] = None
    cfd_condition: Optional[CFDCondition] = None
    write_report: bool = False
    report_path: str = "VALIDATION_REPORT.md"
