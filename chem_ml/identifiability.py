"""
Posterior covariance / Fisher eigenspectrum for parameter identifiability,
plus autodiff sensitivity derivatives (Phase 6, reproduces Tomasini Figs 4-5).
"""
from __future__ import annotations

from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer import MCMC


def parameter_covariance(mcmc: MCMC, param_names: Sequence[str]) -> np.ndarray:
    """Posterior covariance of the named params (Phase 6.1)."""
    s = mcmc.get_samples()
    M = np.stack([np.asarray(s[n]).reshape(-1) for n in param_names], axis=1)
    return np.cov(M, rowvar=False)


def eigenspectrum(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stiff (small variance dir) vs sloppy (large variance dir). Returns
    (eigenvalues ascending, eigenvectors). Rank params by data-constrained-ness."""
    w, V = np.linalg.eigh(cov)
    return w, V


def fisher_information(logmodel: Callable, params: dict, X: jnp.ndarray,
                       sigma: float, param_order: Sequence[str]) -> np.ndarray:
    """I = J^T Sigma^-1 J with J = d(logmodel)/d(theta) via JAX jacfwd (Phase 6.2)."""
    def f(theta_vec):
        p = dict(zip(param_order, theta_vec))
        merged = {**params, **p}
        return logmodel(merged, X)
    theta0 = jnp.array([params[k] for k in param_order])
    J = jax.jacfwd(f)(theta0)                       # (N, P)
    return np.asarray(J.T @ J) / (sigma ** 2)


def sensitivity_derivatives(logmodel: Callable, params: dict, X: jnp.ndarray) -> dict:
    """d(observable)/d(feature) via autodiff, in REAL units (not log).
    observable = exp(logmodel); d(observable)/d(x_j) = observable * d(logmodel)/d(x_j).
    Returns per-row derivatives keyed by feature name; caller maps feature
    derivatives to physical d/dT, d/dp_i using the chain rule through the
    (possibly standardized) feature transform (Phase 6.3)."""
    def f(x_row):
        return logmodel(params, x_row[None, :])[0]

    J = jax.vmap(jax.grad(f))(X)          # (N, D) d(ln y)/d(feature)
    y = jnp.exp(jax.vmap(f)(X))           # (N,)
    dydx = J * y[:, None]                 # (N, D) d(y)/d(feature), chain rule
    return {"y": np.asarray(y), "dlny_dfeature": np.asarray(J), "dy_dfeature": np.asarray(dydx)}
