"""
Differentiable inverse design: given a target (GR, %Ge, [B]) find the recipe
(feature-space point) that achieves it, penalized by posterior-predictive
uncertainty so low-confidence targets are flagged rather than silently
extrapolated (Phase 8).
"""
from __future__ import annotations

from typing import Callable, Optional

import jax
import jax.numpy as jnp
import optax


def inverse_design(forward_log: Callable, theta_bar: dict, y_target_log: jnp.ndarray,
                   x0: jnp.ndarray, feasible_box: tuple[jnp.ndarray, jnp.ndarray],
                   uq_fn: Optional[Callable] = None, lam: float = 1.0,
                   steps: int = 500, lr: float = 1e-2) -> jnp.ndarray:
    """x* = argmin_x || forward_log(theta_bar, x) - y_target_log ||^2 + lam * U(x)
    subject to x in feasible_box. Differentiable; projected gradient descent.
    x here is the FEATURE-SPACE recipe (invT, ln-ratios); map back to flows after.
    (Phase 8.1)"""
    lo, hi = feasible_box
    opt = optax.adam(lr)
    x = x0
    st = opt.init(x)

    def loss(x):
        pred = forward_log(theta_bar, x[None, :])[0]
        data_term = jnp.sum((pred - y_target_log) ** 2)
        uq_term = uq_fn(x) if uq_fn is not None else 0.0
        return data_term + lam * uq_term

    gval = jax.grad(loss)
    for _ in range(steps):
        g = gval(x)
        upd, st = opt.update(g, st)
        x = optax.apply_updates(x, upd)
        x = jnp.clip(x, lo, hi)   # projection onto feasible box
    return x


def posterior_predictive_variance(forward_log: Callable, theta_samples: list[dict],
                                  x: jnp.ndarray) -> jnp.ndarray:
    """U(x): variance across posterior samples of theta at a candidate recipe x.
    Used as the uq_fn passed to inverse_design so the optimizer is penalized
    for drifting into regions the posterior is uncertain about (Phase 8.2)."""
    preds = jnp.stack([forward_log(theta, x[None, :])[0] for theta in theta_samples])
    return jnp.var(preds)


def feature_to_recipe(x: jnp.ndarray, invT_scaler: tuple[float, float]) -> dict:
    """Invert the standardized-invT / log-ratio feature vector back to
    physical (T_K, p_HCl/p_DCS, p_GeH4/p_DCS, p_B2H6/p_DCS)."""
    mu, sd = invT_scaler
    invT = x[0] * sd + mu
    return {
        "T_K": float(1.0 / invT),
        "p_HCl_over_pDCS": float(jnp.exp(x[1])),
        "p_GeH4_over_pDCS": float(jnp.exp(x[2])),
        "p_B2H6_over_pDCS": float(jnp.exp(x[3])) if x.shape[0] > 3 else 0.0,
    }
