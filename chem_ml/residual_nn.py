"""
Small, strongly-regularized residual MLP: y_log = f_phys + g_NN(phi; x).
Gated hard by declared ChemClass so it never contaminates across chemistries
(INVARIANT 3): each ResidualNN instance is trained on, and only ever
evaluated against, rows of its own declared class -- see
fit_residual_networks() below, which partitions the dataset by class before
training a separate net per class.

Library choice: Equinox over Flax. A 2x16-unit MLP has no need for Flax's
module-collection/mutable-state machinery; Equinox's networks are plain
pytrees that compose directly with the pure-JAX f_phys core and the
existing optax training loop already used for inverse design (Phase 8),
with far less boilerplate for a net this small.
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from chem_ml.schema import ChemClass, Dataset


class ResidualNN:
    """g_NN(x; phi): input = [standardized log-features, raw ratios],
    output = per-observable log-residual. Strongly regularized (weight decay)
    so it shrinks to ~0 wherever the power law already explains the data."""

    def __init__(self, chem_class: ChemClass, in_size: int, n_out: int = 2,
                 width_size: int = 16, depth: int = 2, seed: int = 0):
        self.chem_class = chem_class
        self.n_out = n_out
        self.in_size = in_size
        key = jax.random.PRNGKey(seed)
        self.net = eqx.nn.MLP(
            in_size=in_size, out_size=n_out, width_size=width_size, depth=depth,
            activation=jnp.tanh, key=key,
        )
        self.train_loss_curve: list[float] = []

    def __call__(self, X_full: jnp.ndarray) -> jnp.ndarray:
        """Evaluate g_NN on a (N, in_size) batch -> (N, n_out) log-residuals."""
        if X_full.shape[-1] != self.in_size:
            raise ValueError(f"ResidualNN({self.chem_class}) expects in_size="
                              f"{self.in_size}, got {X_full.shape[-1]}")
        return jax.vmap(self.net)(X_full)

    def fit(self, X_full: jnp.ndarray, residual_targets: jnp.ndarray,
            l2: float = 1e-2, steps: int = 3000, lr: float = 5e-3) -> float:
        """Full-batch optax.adamw training (weight_decay=l2 is the shrinkage
        mechanism -- with the physics core already explaining most of the
        variance, weight decay pulls g_NN toward 0 wherever there's no
        residual signal left to fit)."""
        opt = optax.adamw(lr, weight_decay=l2)
        opt_state = opt.init(eqx.filter(self.net, eqx.is_array))

        def loss_fn(net, X, y):
            pred = jax.vmap(net)(X)
            return jnp.mean((pred - y) ** 2)

        @eqx.filter_jit
        def step(net, opt_state, X, y):
            loss, grads = eqx.filter_value_and_grad(loss_fn)(net, X, y)
            updates, opt_state = opt.update(grads, opt_state, eqx.filter(net, eqx.is_array))
            net = eqx.apply_updates(net, updates)
            return net, opt_state, loss

        net = self.net
        self.train_loss_curve = []
        for i in range(steps):
            net, opt_state, loss = step(net, opt_state, X_full, residual_targets)
            if i % 200 == 0 or i == steps - 1:
                self.train_loss_curve.append(float(loss))
        self.net = net
        return self.train_loss_curve[-1]


def build_residual_input(ds: Dataset, fb) -> jnp.ndarray:
    """Input = [standardized log-features (invT, ln_HCl, ln_GeH4, ln_B2H6),
    raw ratios (p_HCl, p_GeH4, p_B2H6)] -- 7 columns. The raw (non-log)
    ratios let the net represent the Regime-I low-pGeH4/pDCS curvature
    (Tomasini Fig. 1) that the log-linear power law misses, without having
    to relearn the whole log-space trend from scratch."""
    raw = np.array([[r.p_HCl / r.p_DCS, r.p_GeH4 / r.p_DCS, r.p_B2H6 / r.p_DCS] for r in ds.rows])
    return jnp.concatenate([fb.X, jnp.asarray(raw)], axis=1)


def fit_residual_networks(ds: Dataset, fb, residual_targets_by_class: dict[ChemClass, jnp.ndarray],
                          l2: float = 1e-2, steps: int = 3000, lr: float = 5e-3,
                          n_out: int = 2) -> dict[ChemClass, ResidualNN]:
    """Train one ResidualNN per ChemClass present, each on ONLY that class's
    rows (the anti-contamination hard gate). `residual_targets_by_class[c]`
    must already be filtered/ordered to match `ds.filter(chem_class=c)`."""
    nets: dict[ChemClass, ResidualNN] = {}
    X_full = build_residual_input(ds, fb)
    for c, targets in residual_targets_by_class.items():
        mask = np.array([r.chem_class == c for r in ds.rows])
        X_c = X_full[mask]
        net = ResidualNN(chem_class=c, in_size=X_c.shape[1], n_out=n_out)
        net.fit(X_c, targets, l2=l2, steps=steps, lr=lr)
        nets[c] = net
    return nets
