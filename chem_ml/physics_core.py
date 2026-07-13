"""
Pure, differentiable JAX forward maps for GR, Ge/Si ratio, and B/Si ratio.

INVARIANT: kappa is the coefficient of 1/T, NOT exp(-Ea/RT). This sidesteps
Tomasini's sign-convention ambiguity in his tabulated "Ea/R" values.

  GR *increases* with T  => kappa_GR < 0, expect kappa_GR ~ -24,507 K
                             (this equals Tomasini's tabulated "Ea/R"; the
                             negative coefficient of 1/T is what makes GR
                             rise with T).
  Ge fraction *decreases* with T => kappa_Ge > 0, expect kappa_Ge ~ +4,319 K.
                             NOTE: this is the OPPOSITE sign from kappa_GR,
                             because GR and the Ge/Si ratio move in OPPOSITE
                             directions with T while sharing the identical
                             ln(y) = lnK + kappa/T functional form (verified
                             directly against DS1: at matched GeH4/DCS ~0.045,
                             Ge% is ~33% at 605 C vs ~21% at 765 C). An
                             earlier draft of this design incorrectly carried
                             the same negative sign over from kappa_GR by
                             analogy -- it does not transfer; the sign must
                             match this dataset's own T-trend for each
                             observable independently, not Tomasini's raw
                             tabulated "Ea/R" column (which the two
                             equations do not use in the same sense -- see
                             build_steps_and_cfd_integration.md Phase 3.1).

This is the single most error-prone point in the whole pipeline -- do not
"fix" the sign without re-reading the docstring above and re-checking
against the raw data.

NOTE: features pass STANDARDIZED invT (see features.build_features); the
model learns kappa_std then destandardize_kappa() converts back to physical
K units for reporting against Tomasini's tables.
"""
from __future__ import annotations

import jax.numpy as jnp


def gr_logmodel(params: dict, X: jnp.ndarray) -> jnp.ndarray:
    """ln(GR). params: lnK_GR, kappa_GR (std units), gamma_HCl, gamma_GeH4."""
    invT, ln_HCl, ln_GeH4 = X[:, 0], X[:, 1], X[:, 2]
    return (params["lnK_GR"]
            + params["kappa_GR"] * invT
            + params["gamma_HCl"] * ln_HCl
            + params["gamma_GeH4"] * ln_GeH4)


def ge_logmodel(params: dict, X: jnp.ndarray) -> jnp.ndarray:
    """ln(x/(1-x)). params: lnK_Ge, kappa_Ge, dgamma_HCl, dgamma_GeH4."""
    invT, ln_HCl, ln_GeH4 = X[:, 0], X[:, 1], X[:, 2]
    return (params["lnK_Ge"]
            + params["kappa_Ge"] * invT
            + params["dgamma_HCl"] * ln_HCl
            + params["dgamma_GeH4"] * ln_GeH4)


def b_logmodel(params: dict, X: jnp.ndarray) -> jnp.ndarray:
    """ln([B]/[Si]). params: lnK_B, beta_HCl, beta_GeH4, beta_B2H6."""
    ln_HCl, ln_GeH4, ln_B2H6 = X[:, 1], X[:, 2], X[:, 3]
    return (params["lnK_B"]
            + params["beta_HCl"] * ln_HCl
            + params["beta_GeH4"] * ln_GeH4
            + params["beta_B2H6"] * ln_B2H6)


def dopant_logmodel(params: dict, X: jnp.ndarray) -> jnp.ndarray:
    """ln([X]/[Si]) for a generic dopant precursor such as PH3 or B2H6."""
    ln_HCl, ln_GeH4, ln_dopant = X[:, 1], X[:, 2], X[:, 5]
    return (params["lnK_X"]
            + params["beta_HCl_X"] * ln_HCl
            + params["beta_GeH4_X"] * ln_GeH4
            + params["beta_dopant_X"] * ln_dopant)


def c_logmodel(params: dict, X: jnp.ndarray) -> jnp.ndarray:
    """ln(x_C/(1-x_C)) for SiGeC carbon incorporation.

    Reads only the appended carbon slot plus legacy HCl/GeH4/invT features,
    so it is inert unless the class-specific training route selects it.
    """
    invT, ln_HCl, ln_GeH4, ln_C_source = X[:, 0], X[:, 1], X[:, 2], X[:, 4]
    return (params["lnK_C"]
            + params["kappa_C"] * invT
            + params["cgamma_HCl"] * ln_HCl
            + params["cgamma_GeH4"] * ln_GeH4
            + params["cgamma_MMS"] * ln_C_source)


def destandardize_kappa(kappa_std: float, invT_scaler: tuple[float, float]) -> float:
    """Convert learned kappa (on standardized 1/T) back to physical K units.
    Report this against Tomasini's tabulated Ea/R (expect ~ -24507 for GR)."""
    _, sd = invT_scaler
    return kappa_std / sd
