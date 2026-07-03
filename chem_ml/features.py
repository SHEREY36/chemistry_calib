"""
Log-space feature builder. Columns: [1/T, ln(pHCl/pDCS), ln(pGeH4/pDCS),
ln(pB2H6/pDCS)]. Intercept handled inside the NumPyro model (lnK).
"""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from chem_ml.schema import Dataset


@dataclass
class FeatureBundle:
    """Design matrix in log space + bookkeeping to invert scaling."""
    X: jnp.ndarray            # (N, D) features
    col_names: list[str]
    invT_scaler: tuple[float, float]  # (mean, std) used to standardize 1/T


def build_features(ds: Dataset, standardize_invT: bool = True,
                   invT_scaler: tuple[float, float] | None = None) -> FeatureBundle:
    """Columns: [1/T, ln(pHCl/pDCS), ln(pGeH4/pDCS), ln(pB2H6/pDCS)].
    p_DCS is always 1.0 by the normalization convention in schema.py, so
    ln(p_i/p_DCS) reduces to ln(p_i).

    `invT_scaler`: pass an EXISTING (mean, std) to standardize against,
    instead of computing one from `ds`. Required whenever `ds` doesn't span
    a real temperature range on its own -- e.g. DS3 is a single fixed T, so
    its own std(invT) is 0 and self-standardizing would divide by ~0. Reuse
    the scaler theta_chem was actually fit against (Phase 7)."""
    rows = ds.rows
    invT = np.array([1.0 / r.T_K for r in rows])
    ln_HCl = np.array([np.log(r.p_HCl / r.p_DCS) for r in rows])
    ln_GeH4 = np.array([np.log(r.p_GeH4 / r.p_DCS) for r in rows])
    # guard log(0) for B2H6 absent -> use -inf-safe: absent B just won't feed B-model
    ln_B2H6 = np.array([np.log(r.p_B2H6 / r.p_DCS) if r.p_B2H6 > 0 else 0.0 for r in rows])

    if invT_scaler is not None:
        mu, sd = invT_scaler
    elif standardize_invT:
        mu, sd = float(invT.mean()), float(invT.std() + 1e-12)
    else:
        mu, sd = 0.0, 1.0
    invT_s = (invT - mu) / sd

    X = jnp.asarray(np.stack([invT_s, ln_HCl, ln_GeH4, ln_B2H6], axis=1))
    return FeatureBundle(X, ["invT", "ln_HCl", "ln_GeH4", "ln_B2H6"], (mu, sd))
