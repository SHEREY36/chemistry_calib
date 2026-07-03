"""
Config dataclasses: RNG seeds, priors, MCMC settings, tolerances, paths.
Everything downstream reads from Config; nothing hard-codes numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, field

R_GAS = 8.314462618  # J/mol/K (used only if converting kappa <-> Ea; kappa is primary)


@dataclass(frozen=True)
class MCMCConfig:
    num_warmup: int = 1500
    num_samples: int = 2000
    num_chains: int = 4
    target_accept: float = 0.9
    seed: int = 0


@dataclass(frozen=True)
class PriorConfig:
    """Literature-informed priors (means, sds) in the model's native parametrization.
    kappa is the coefficient of 1/T (units K). Orders are dimensionless."""
    # GR model
    lnK_GR: tuple[float, float] = (0.0, 10.0)          # weak; K absorbs scaling
    kappa_GR: tuple[float, float] = (-24507.0, 5000.0)  # GR RISES with T -> negative
    gamma_HCl: tuple[float, float] = (-0.7, 0.3)
    gamma_GeH4: tuple[float, float] = (1.3, 0.3)
    # Ge/Si model
    lnK_Ge: tuple[float, float] = (0.0, 10.0)
    # Ge FALLS with T -> kappa_Ge > 0 in the ln(y)=lnK+kappa/T convention
    # (opposite sign from kappa_GR; see physics_core.py docstring -- this is
    # a corrected sign relative to the original design-doc draft, verified
    # against DS1: Ge% ~33% at 605C vs ~21% at 765C at matched GeH4/DCS).
    kappa_Ge: tuple[float, float] = (4319.0, 3000.0)
    dgamma_HCl: tuple[float, float] = (0.1, 0.2)
    dgamma_GeH4: tuple[float, float] = (0.51, 0.2)
    # Boron model (DS2)
    lnK_B: tuple[float, float] = (0.0, 10.0)
    beta_HCl: tuple[float, float] = (-0.5, 0.3)
    beta_GeH4: tuple[float, float] = (-0.3, 0.3)
    beta_B2H6: tuple[float, float] = (0.8, 0.2)
    # noise
    sigma_halfnormal: float = 0.5  # log-space observation noise scale


@dataclass(frozen=True)
class Config:
    data_raw: str = "data/raw"
    data_processed: str = "data/processed"
    mcmc: MCMCConfig = field(default_factory=MCMCConfig)
    priors: PriorConfig = field(default_factory=PriorConfig)
    # acceptance tolerances (Phase 4.3)
    r2_target: float = 0.98
    kappa_GR_tol_frac: float = 0.10
    # inverse design
    inverse_uq_lambda: float = 1.0
    # active learning
    cfd_run_budget: int = 25
    al_batch_size: int = 3
