"""
Log-space feature builder. The first four columns are a stable public
contract for the legacy SiGe/Tomasini models:
    [1/T, ln(pHCl/pSi), ln(pGeH4/pSi), ln(pB2H6/pSi)]

New class-aware process features are appended after those columns so old
forward maps remain bit-stable by construction.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from chem_ml.schema import Dataset

FEATURE_COLUMNS = [
    "invT",
    "ln_HCl",
    "ln_GeH4",
    "ln_B2H6",
    "ln_C_source",
    "ln_dopant",
    "ln_H2",
    "ln_N2",
    "XT_H2_minus_N2_scaled",
    "pattern_density",
]


@dataclass
class FeatureBundle:
    """Design matrix in log space + bookkeeping to invert scaling."""
    X: jnp.ndarray            # (N, D) features
    col_names: list[str]
    invT_scaler: tuple[float, float]  # (mean, std) used to standardize 1/T


def _safe_ln_ratio(numer: float, denom: float, *, zero_if_absent: bool = False) -> float:
    if numer <= 0.0:
        if zero_if_absent:
            return 0.0
        return float(np.log(1e-30 / denom))
    return float(np.log(numer / denom))


def build_features(ds: Dataset, standardize_invT: bool = True,
                   invT_scaler: tuple[float, float] | None = None) -> FeatureBundle:
    """Build the stable feature matrix.

    p_DCS is retained as the normalized Si-source denominator for backward
    compatibility; for non-DCS Si sources it is still the dimensionless
    p_Si reference and remains 1.0 at intake.

    `invT_scaler`: pass an EXISTING (mean, std) to standardize against,
    instead of computing one from `ds`. Required whenever `ds` doesn't span
    a real temperature range on its own -- e.g. DS3 is a single fixed T, so
    its own std(invT) is 0 and self-standardizing would divide by ~0. Reuse
    the scaler theta_chem was actually fit against (Phase 7)."""
    rows = ds.rows
    invT = np.array([1.0 / r.T_K for r in rows])
    ln_HCl = np.array([_safe_ln_ratio(r.p_HCl, r.p_DCS) for r in rows])
    ln_GeH4 = np.array([_safe_ln_ratio(r.p_GeH4, r.p_DCS) for r in rows])
    # guard log(0) for B2H6 absent -> use -inf-safe: absent B just won't feed B-model
    ln_B2H6 = np.array([_safe_ln_ratio(r.p_B2H6, r.p_DCS, zero_if_absent=True) for r in rows])
    ln_C_source = np.array([_safe_ln_ratio(r.p_MMS, r.p_DCS, zero_if_absent=True) for r in rows])
    ln_dopant = np.array([_safe_ln_ratio(r.p_dopant, r.p_DCS, zero_if_absent=True) for r in rows])
    ln_H2 = np.array([_safe_ln_ratio(r.p_H2, r.p_DCS, zero_if_absent=True) for r in rows])
    ln_N2 = np.array([_safe_ln_ratio(r.p_N2, r.p_DCS, zero_if_absent=True) for r in rows])
    xt_scaled = np.array([r.XT_flow_H2_minus_N2_sccm / 1000.0 for r in rows])
    pattern_density = np.array([r.pattern_density for r in rows])

    if invT_scaler is not None:
        mu, sd = invT_scaler
    elif standardize_invT:
        mu, sd = float(invT.mean()), float(invT.std() + 1e-12)
    else:
        mu, sd = 0.0, 1.0
    invT_s = (invT - mu) / sd

    X = jnp.asarray(np.stack([
        invT_s,
        ln_HCl,
        ln_GeH4,
        ln_B2H6,
        ln_C_source,
        ln_dopant,
        ln_H2,
        ln_N2,
        xt_scaled,
        pattern_density,
    ], axis=1))
    return FeatureBundle(X, FEATURE_COLUMNS.copy(), (mu, sd))
