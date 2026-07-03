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
    (Phase 8.1)

    The per-step update is JIT-compiled: `steps` calls to a Python-level
    jax.grad(loss) without jit would retrace the whole graph every
    iteration (500 retraces was the difference between this finishing in
    under a second and taking minutes -- see uq_fn's docstring in
    posterior_predictive_variance for the matching fix on the UQ side)."""
    lo, hi = feasible_box
    opt = optax.adam(lr)

    def loss(x):
        pred = forward_log(theta_bar, x[None, :])[0]
        data_term = jnp.sum((pred - y_target_log) ** 2)
        uq_term = uq_fn(x) if uq_fn is not None else 0.0
        return data_term + lam * uq_term

    @jax.jit
    def step(x, st):
        g = jax.grad(loss)(x)
        upd, st = opt.update(g, st)
        x = optax.apply_updates(x, upd)
        x = jnp.clip(x, lo, hi)
        return x, st

    x, st = x0, opt.init(x0)
    for _ in range(steps):
        x, st = step(x, st)
    return x


def stack_theta_samples(theta_samples: list[dict]) -> dict:
    """[{'a': 1.0, 'b': 2.0}, {'a': 1.1, 'b': 2.1}, ...] -> {'a': [1.0, 1.1],
    'b': [2.0, 2.1]}: turns a list of per-sample param dicts into one dict of
    stacked arrays, so posterior_predictive_variance can jax.vmap over the
    sample axis instead of Python-looping over samples."""
    keys = theta_samples[0].keys()
    return {k: jnp.asarray([t[k] for t in theta_samples]) for k in keys}


def posterior_predictive_variance(forward_log: Callable, theta_stacked: dict,
                                  x: jnp.ndarray) -> jnp.ndarray:
    """U(x): variance across posterior samples of theta at a candidate recipe
    x, summed PER OUTPUT DIMENSION (not pooled), vmapped over the sample
    axis. Used as the uq_fn passed to inverse_design so the optimizer is
    penalized for drifting into regions the posterior is uncertain about
    (Phase 8.2).

    Two bugs fixed relative to an earlier version of this function:
    (1) `theta_stacked` must be a dict of STACKED arrays (see
    stack_theta_samples), evaluated via jax.vmap, not a Python list of
    per-sample dicts evaluated via a list comprehension + jnp.stack --
    the latter is untraced Python looping, which is what made
    inverse_design() take minutes instead of under a second once this
    function is called every step of a 500-step optimization loop.
    (2) Pooling all output dims into one jnp.var() is wrong for a
    multi-output forward_log: e.g. ln(GR) sits around 3 and
    ln(x/(1-x)) sits around -1, so a flat pooled variance is dominated by
    that between-channel offset rather than genuine per-channel posterior
    spread, corrupting the gradient inverse_design optimizes against.
    Per-dimension variance, summed, is the correct aggregate penalty."""
    def one_sample(theta_i):
        return forward_log(theta_i, x[None, :])[0]

    preds = jax.vmap(one_sample)(theta_stacked)  # (S, D)
    return jnp.sum(jnp.var(preds, axis=0))


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
