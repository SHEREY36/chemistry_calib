"""
Inference-time visualizations: MCMC posterior structure, single-query
predictive credible intervals, and the head-to-head comparison that answers
"why build a physics+Bayesian model instead of a black-box regressor."

Run: conda activate epitaxy && python -m chem_ml.inference_plots
Writes PNGs to figures/inference_*.png.
"""
from __future__ import annotations

import logging
from pathlib import Path

import arviz as az
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter

from chem_ml.calibration import mu_draws, posterior_predict
from chem_ml.calibration import gr_numpyro_model
from chem_ml.config import Config
from chem_ml.physics_core import gr_logmodel
from chem_ml.pipeline import _GE_PARAM_NAMES, _GR_PARAM_NAMES, load_all_datasets, run_phase4_calibration

log = logging.getLogger("chem_ml")
FIG_DIR = Path("figures")
_PRESENTATION_DRAWS = 1000


def _subset_samples(mcmc, names: list[str], max_draws: int = _PRESENTATION_DRAWS) -> dict[str, np.ndarray]:
    samples = mcmc.get_samples()
    n = len(samples[names[0]])
    idx = np.linspace(0, n - 1, min(max_draws, n), dtype=int)
    return {name: np.asarray(samples[name])[idx] for name in names}


def _make_feature_matrix(invT_scaler: tuple[float, float], T_C, hcl_ratio, geh4_ratio) -> np.ndarray:
    T_C = np.asarray(T_C, dtype=float)
    hcl_ratio = np.asarray(hcl_ratio, dtype=float)
    geh4_ratio = np.asarray(geh4_ratio, dtype=float)
    T_C, hcl_ratio, geh4_ratio = np.broadcast_arrays(T_C, hcl_ratio, geh4_ratio)
    mu, sd = invT_scaler
    invT_std = (1.0 / (T_C + 273.15) - mu) / sd
    return np.stack([
        invT_std.ravel(),
        np.log(hcl_ratio.ravel()),
        np.log(geh4_ratio.ravel()),
        np.zeros(invT_std.size),
    ], axis=1)


def _gr_draws_from_samples(samples: dict[str, np.ndarray], X: np.ndarray) -> np.ndarray:
    return np.exp(
        samples["lnK_GR"][:, None]
        + samples["kappa_GR"][:, None] * X[None, :, 0]
        + samples["gamma_HCl"][:, None] * X[None, :, 1]
        + samples["gamma_GeH4"][:, None] * X[None, :, 2]
    )


def _ge_draws_from_samples(samples: dict[str, np.ndarray], X: np.ndarray) -> np.ndarray:
    ge_ratio = np.exp(
        samples["lnK_Ge"][:, None]
        + samples["kappa_Ge"][:, None] * X[None, :, 0]
        + samples["dgamma_HCl"][:, None] * X[None, :, 1]
        + samples["dgamma_GeH4"][:, None] * X[None, :, 2]
    )
    return ge_ratio / (1.0 + ge_ratio)


def _draw_summary(draws: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(np.percentile(draws, [5, 50, 95], axis=0))


def _set_ratio_ticks(ax, *, x_ticks: list[float] | None = None, y_ticks: list[float] | None = None) -> None:
    if x_ticks is not None:
        ax.xaxis.set_major_locator(FixedLocator(x_ticks))
        ax.xaxis.set_major_formatter(FixedFormatter([f"{t:.2f}" for t in x_ticks]))
        ax.xaxis.set_minor_formatter(NullFormatter())
    if y_ticks is not None:
        ax.yaxis.set_major_locator(FixedLocator(y_ticks))
        ax.yaxis.set_major_formatter(FixedFormatter([f"{t:.2f}" for t in y_ticks]))
        ax.yaxis.set_minor_formatter(NullFormatter())


def plot_posterior_pairplot(p4: dict, out: Path) -> None:
    """Corner-style pairwise posterior scatter for the 4 GR parameters --
    what "MCMC posterior samples" actually look like, including the
    correlation structure between parameters (e.g. lnK_GR trades off
    against gamma_HCl/gamma_GeH4 the way any regression intercept trades
    off against its slopes)."""
    # Avoid ArviZ's marginal KDE path here: on some conda/numba installs it
    # tries to enable a cache for ArviZ package files and fails before any
    # plotting happens. The pairwise hexbin panels carry the correlation
    # structure this diagnostic is meant to show.
    axes = az.plot_pair(p4["idata_gr"], var_names=_GR_PARAM_NAMES, kind="hexbin",
                        marginals=False, figsize=(8, 8))
    fig = np.asarray(axes).flat[0].figure
    fig.suptitle("GR model: MCMC posterior samples (4 chains x 2000 draws, pairwise)", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out)


def plot_trace(p4: dict, out: Path) -> None:
    """Chain-mixing trace plot -- the visual counterpart to the R-hat/ESS
    numbers already gated in Phase 4.2: well-mixed chains look like
    'fuzzy caterpillars' overlapping each other, not separated bands."""
    samples = p4["mcmc_gr"].get_samples(group_by_chain=True)
    fig, axes = plt.subplots(len(_GR_PARAM_NAMES), 1, figsize=(10, 8), sharex=True)
    for ax, name in zip(np.asarray(axes).flat, _GR_PARAM_NAMES):
        vals = np.asarray(samples[name])
        for chain_idx in range(vals.shape[0]):
            label = f"chain {chain_idx + 1}" if name == _GR_PARAM_NAMES[0] else None
            ax.plot(vals[chain_idx], lw=0.5, alpha=0.75, label=label)
        ax.set_ylabel(name)
    axes[-1].set_xlabel("draw")
    axes[0].legend(loc="upper right", ncol=4, fontsize=8)
    fig.suptitle("GR model: MCMC trace (4 chains) -- convergence check", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out)


def plot_credible_interval_for_query(p4: dict, cfg: Config, T_C: float, hcl_ratio: float,
                                     geh4_ratio: float, out: Path) -> dict:
    """Full posterior-PREDICTIVE distribution (parameter uncertainty +
    observation noise) for one specific recipe, as a histogram with the
    mean and 90% credible interval marked -- what an engineer actually
    gets back for a single query, not an aggregate calibration statistic."""
    fb1 = p4["features_ds1"]
    mu, sd = fb1.invT_scaler
    T_K = T_C + 273.15
    invT_std = (1.0 / T_K - mu) / sd
    X = jnp.array([[invT_std, jnp.log(hcl_ratio), jnp.log(geh4_ratio), 0.0]])

    gr_pred_log = np.asarray(posterior_predict(gr_numpyro_model, p4["mcmc_gr"], X, cfg))[:, 0]
    gr_pred = np.exp(gr_pred_log)
    p5, p50, p95 = np.percentile(gr_pred, [5, 50, 95])

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.hist(gr_pred, bins=60, color="tab:blue", alpha=0.7)
    ax.axvline(p50, color="k", lw=2, label=f"median = {p50:.1f} nm/min")
    ax.axvline(p5, color="k", ls="--", lw=1, label=f"90% CI = [{p5:.1f}, {p95:.1f}]")
    ax.axvline(p95, color="k", ls="--", lw=1)
    ax.set_xlabel("GR (nm/min)")
    ax.set_ylabel("posterior predictive draws")
    ax.set_title(f"Single-query prediction: T={T_C:.0f}C, HCl/DCS={hcl_ratio}, GeH4/DCS={geh4_ratio}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("Wrote %s", out)
    return {"p5": float(p5), "p50": float(p50), "p95": float(p95)}


def plot_extrapolation_superiority(p4: dict, out: Path) -> None:
    """The concrete argument for why this is physics+Bayesian, not a
    black-box regressor: sweep pGeH4/pDCS from within DS1's observed range
    out to well beyond it (fixed T=725C, pHCl/pDCS=0.5), and compare:

      (a) THIS model: posterior mean +/- 90% credible interval. Being a
          power law, it extrapolates as a straight line in log-log space,
          and its credible interval WIDENS beyond the training envelope
          (the posterior over gamma_GeH4 gets amplified by how far ln(x)
          is from the data's own range -- exactly what should happen when
          you leave the region data constrains).

      (b) A generic black-box regressor (RandomForestRegressor, scikit-
          learn) fit DIRECTLY on [T_K, HCl_ratio, GeH4_ratio] -> GR_nm_min,
          no log-features, no power-law structure, no Bayesian posterior --
          representative of "just fit a flexible model to the data" with
          zero physics. A random forest cannot extrapolate past the
          training data's feature range BY CONSTRUCTION (every tree's leaf
          prediction is a constant equal to a training-set average; once
          inputs exceed the max split threshold, every tree returns the
          same boundary leaf value) -- so it flatlines exactly at the
          edge of the data, with LOW inter-tree spread (false confidence),
          not a widening interval reflecting genuine doubt.

    This is the concrete, checkable answer to 'why not just curve-fit a
    black box': the black box doesn't know it's extrapolating; the
    physics+Bayesian model does, and says so."""
    from sklearn.ensemble import RandomForestRegressor

    ds1 = load_all_datasets(Config()).filter(source_dataset="DS1")
    fb1 = p4["features_ds1"]
    mu, sd = fb1.invT_scaler

    T_K_arr = np.array([r.T_K for r in ds1.rows])
    hcl_arr = np.array([r.p_HCl for r in ds1.rows])
    geh4_arr = np.array([r.p_GeH4 for r in ds1.rows])
    gr_arr = np.array([r.GR_nm_min for r in ds1.rows])
    max_geh4_ratio = float(geh4_arr.max())

    rf = RandomForestRegressor(n_estimators=300, random_state=0)
    rf.fit(np.stack([T_K_arr, hcl_arr, geh4_arr], axis=1), gr_arr)

    T_C_fixed, hcl_fixed = 725.0, 0.5
    geh4_sweep = np.linspace(0.01, max_geh4_ratio * 3.0, 200)
    T_K_fixed = T_C_fixed + 273.15

    # ---- (a) physics + Bayesian ---------------------------------------------
    invT_std = (1.0 / T_K_fixed - mu) / sd
    X = jnp.asarray(np.stack([np.full_like(geh4_sweep, invT_std), np.full_like(geh4_sweep, np.log(hcl_fixed)),
                             np.log(geh4_sweep), np.zeros_like(geh4_sweep)], axis=1))
    draws = np.asarray(mu_draws(gr_logmodel, p4["mcmc_gr"], X, _GR_PARAM_NAMES))
    phys_mean = np.exp(draws.mean(axis=0))
    phys_lo, phys_hi = np.exp(np.percentile(draws, 5, axis=0)), np.exp(np.percentile(draws, 95, axis=0))

    # ---- (b) black-box RF, with its own naive UQ (inter-tree spread) --------
    X_rf = np.stack([np.full_like(geh4_sweep, T_K_fixed), np.full_like(geh4_sweep, hcl_fixed), geh4_sweep], axis=1)
    per_tree = np.stack([t.predict(X_rf) for t in rf.estimators_], axis=0)  # (n_trees, n_sweep)
    rf_mean = per_tree.mean(axis=0)
    rf_lo, rf_hi = np.percentile(per_tree, 5, axis=0), np.percentile(per_tree, 95, axis=0)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(geh4_sweep, phys_mean, color="tab:blue", label="physics+Bayesian: posterior mean")
    ax.fill_between(geh4_sweep, phys_lo, phys_hi, color="tab:blue", alpha=0.25, label="physics+Bayesian: 90% CI")
    ax.plot(geh4_sweep, rf_mean, color="tab:red", label="black-box RF: mean (inter-tree)")
    ax.fill_between(geh4_sweep, rf_lo, rf_hi, color="tab:red", alpha=0.25, label="black-box RF: inter-tree 90% band")
    ax.axvline(max_geh4_ratio, color="k", ls="--", lw=1.2, label="end of DS1 training range")
    ax.scatter(geh4_arr[np.isclose(hcl_arr, hcl_fixed, atol=0.15)],
              gr_arr[np.isclose(hcl_arr, hcl_fixed, atol=0.15)],
              color="k", marker="x", s=30, label="nearby DS1 points (HCl/DCS~0.5)", zorder=5)
    ax.set_yscale("log")
    ax.set_xlabel("pGeH4/pDCS (swept beyond training range)")
    ax.set_ylabel("GR (nm/min)")
    ax.set_title(f"Extrapolation: physics+Bayesian vs. black-box RF\n(T={T_C_fixed:.0f}C, pHCl/pDCS={hcl_fixed})")
    ax.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("Wrote %s", out)

    # The point of this plot isn't the CI width AT one point -- it's whether
    # the band keeps GROWING past the training edge (honest extrapolation
    # uncertainty) or freezes (false confidence). Compare band width at the
    # training edge vs. at 3x the range for both models.
    edge_idx = int(np.argmin(np.abs(geh4_sweep - max_geh4_ratio)))
    phys_width_edge, phys_width_end = phys_hi[edge_idx] - phys_lo[edge_idx], phys_hi[-1] - phys_lo[-1]
    rf_width_edge, rf_width_end = rf_hi[edge_idx] - rf_lo[edge_idx], rf_hi[-1] - rf_lo[-1]
    log.info("Band width AT the training edge -> AT 3x the range: "
            "physics+Bayesian %.1f -> %.1f nm/min (%.2fx growth); "
            "black-box RF %.1f -> %.1f nm/min (%.2fx growth, i.e. frozen) -- "
            "the RF's band does not widen past its training range at all; the "
            "physics+Bayesian model's does, because it still has real parameter "
            "uncertainty (gamma_GeH4's posterior spread) to propagate.",
            phys_width_edge, phys_width_end, phys_width_end / phys_width_edge,
            rf_width_edge, rf_width_end, rf_width_end / rf_width_edge)


def plot_bayesian_response_envelope(p4: dict, out: Path) -> dict:
    """Presentation plot: sweep GeH4/DCS at fixed HCl/DCS and show the family
    of GR curves allowed by the posterior.

    This is more useful than a raw posterior pairplot for non-statisticians:
    it turns parameter uncertainty into a process statement. Inside the DS1
    envelope the band is tight; beyond the edge, the mean still follows the
    physically constrained power law but the band widens instead of pretending
    the extrapolation is just as well known as the calibration region."""
    ds1 = load_all_datasets(Config()).filter(source_dataset="DS1")
    fb1 = p4["features_ds1"]
    gr_samples = _subset_samples(p4["mcmc_gr"], _GR_PARAM_NAMES)

    geh4_train = np.array([r.p_GeH4 for r in ds1.rows])
    hcl_train = np.array([r.p_HCl for r in ds1.rows])
    gr_train = np.array([r.GR_nm_min for r in ds1.rows])
    max_geh4 = float(geh4_train.max())

    hcl_fixed = 0.34
    T_grid = [650.0, 700.0, 750.0]
    geh4_sweep = np.geomspace(max(0.006, geh4_train.min() * 0.7), max_geh4 * 2.5, 180)
    colors = ["tab:green", "tab:orange", "tab:blue"]

    fig, (ax_pred, ax_uq) = plt.subplots(1, 2, figsize=(12.0, 4.8), sharex=True)
    metrics = {"max_train_geh4": max_geh4, "hcl_fixed": hcl_fixed, "temperatures_C": T_grid}

    for T_C, color in zip(T_grid, colors):
        X = _make_feature_matrix(fb1.invT_scaler, T_C, hcl_fixed, geh4_sweep)
        draws = _gr_draws_from_samples(gr_samples, X)
        lo, med, hi = _draw_summary(draws)
        rel_width = (hi - lo) / med
        edge_idx = int(np.argmin(np.abs(geh4_sweep - max_geh4)))
        metrics[f"relative_width_growth_{int(T_C)}C"] = float(rel_width[-1] / rel_width[edge_idx])

        label = f"{T_C:.0f} C posterior median"
        ax_pred.plot(geh4_sweep, med, color=color, lw=2, label=label)
        ax_pred.fill_between(geh4_sweep, lo, hi, color=color, alpha=0.18)
        ax_uq.plot(geh4_sweep, rel_width, color=color, lw=2, label=f"{T_C:.0f} C")

    near_hcl = np.isclose(hcl_train, hcl_fixed, atol=0.08)
    ax_pred.scatter(geh4_train[near_hcl], gr_train[near_hcl], color="black", marker="x",
                    s=35, label="DS1 points near HCl/DCS=0.34", zorder=5)
    for ax in (ax_pred, ax_uq):
        ax.axvline(max_geh4, color="black", ls="--", lw=1.1)
        ax.axvspan(max_geh4, geh4_sweep[-1], color="0.92", alpha=0.75, zorder=0)
        ax.set_xscale("log")
        _set_ratio_ticks(ax, x_ticks=[0.02, 0.03, 0.05, 0.10, 0.20])
        ax.set_xlabel(r"$p_{\mathrm{GeH4}}/p_{\mathrm{DCS}}$")
    ax_pred.set_yscale("log")
    ax_pred.set_ylabel("GR (nm/min)")
    ax_pred.set_title("Posterior process curves")
    ax_pred.legend(fontsize=7.5, loc="upper left")
    ax_pred.text(max_geh4 * 1.08, ax_pred.get_ylim()[0] * 1.35, "outside DS1\nGeH4 range", fontsize=8)

    ax_uq.set_ylabel("90% band width / median")
    ax_uq.set_title("Uncertainty grows when extrapolating")
    ax_uq.legend(fontsize=8, loc="upper left")
    fig.suptitle("Bayesian physics response envelope at fixed HCl/DCS=0.34", y=1.03)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out)
    return metrics


def plot_process_window_probability_map(p4: dict, out: Path) -> dict:
    """Presentation plot: convert the posterior into a recipe-space decision
    map at a fixed temperature.

    The lower-right panel is deliberately phrased as probability of meeting a
    process target rather than "model score". That is the practical advantage
    over a deterministic Tomasini-style response function: the model gives a
    process window and a confidence level at the same time."""
    ds1 = load_all_datasets(Config()).filter(source_dataset="DS1")
    fb1 = p4["features_ds1"]
    gr_samples = _subset_samples(p4["mcmc_gr"], _GR_PARAM_NAMES)
    ge_samples = _subset_samples(p4["mcmc_ge"], _GE_PARAM_NAMES)

    hcl_train = np.array([r.p_HCl for r in ds1.rows])
    geh4_train = np.array([r.p_GeH4 for r in ds1.rows])
    T_C = 725.0

    hcl_grid = np.geomspace(max(0.02, hcl_train.min() * 0.7), hcl_train.max() * 1.35, 58)
    geh4_grid = np.geomspace(max(0.006, geh4_train.min() * 0.7), geh4_train.max() * 1.6, 64)
    H, G = np.meshgrid(hcl_grid, geh4_grid, indexing="ij")
    X = _make_feature_matrix(fb1.invT_scaler, T_C, H, G)

    gr_draws = _gr_draws_from_samples(gr_samples, X)
    ge_draws = _ge_draws_from_samples(ge_samples, X)
    gr_lo, gr_med, gr_hi = _draw_summary(gr_draws)
    _, ge_med, _ = _draw_summary(100.0 * ge_draws)
    gr_rel_width = (gr_hi - gr_lo) / gr_med

    target = (
        (gr_draws >= 20.0)
        & (gr_draws <= 50.0)
        & (100.0 * ge_draws >= 18.0)
        & (100.0 * ge_draws <= 22.0)
    )
    target_prob = target.mean(axis=0)

    shape = H.shape
    gr_med = gr_med.reshape(shape)
    ge_med = ge_med.reshape(shape)
    gr_rel_width = gr_rel_width.reshape(shape)
    target_prob = target_prob.reshape(shape)

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.2), sharex=True, sharey=True)
    panels = [
        (axes[0, 0], gr_med, "Posterior median GR", "GR (nm/min)", "viridis"),
        (axes[0, 1], gr_rel_width, "GR uncertainty", "90% width / median", "magma"),
        (axes[1, 0], ge_med, "Posterior median Ge", "Ge (at.%)", "cividis"),
        (axes[1, 1], target_prob, "Probability of target window", "Pr(20-50 nm/min and 18-22% Ge)", "YlGn"),
    ]
    for ax, Z, title, cbar_label, cmap in panels:
        pcm = ax.pcolormesh(G, H, Z, shading="auto", cmap=cmap)
        cb = fig.colorbar(pcm, ax=ax)
        cb.set_label(cbar_label)
        ax.scatter(geh4_train, hcl_train, c="white", edgecolors="black", s=22, linewidths=0.6)
        ax.set_xscale("log")
        ax.set_yscale("log")
        _set_ratio_ticks(ax, x_ticks=[0.02, 0.03, 0.05, 0.10], y_ticks=[0.10, 0.20, 0.50, 1.00])
        ax.set_title(title)
        ax.axvline(geh4_train.max(), color="white", ls="--", lw=1.0, alpha=0.9)
        ax.axhline(hcl_train.max(), color="white", ls="--", lw=1.0, alpha=0.9)
    axes[0, 0].contour(G, H, gr_med, levels=[10, 30, 100], colors="white", linewidths=0.9)
    axes[1, 0].contour(G, H, ge_med, levels=[10, 20, 30], colors="white", linewidths=0.9)
    axes[1, 1].contour(G, H, target_prob, levels=[0.25, 0.50, 0.75], colors="black", linewidths=1.0)

    for ax in axes[1, :]:
        ax.set_xlabel(r"$p_{\mathrm{GeH4}}/p_{\mathrm{DCS}}$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$p_{\mathrm{HCl}}/p_{\mathrm{DCS}}$")
    fig.suptitle("Posterior process-window map at T=725 C", y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out)

    best_idx = np.unravel_index(int(np.argmax(target_prob)), target_prob.shape)
    return {
        "T_C": T_C,
        "target": "GR 20-50 nm/min and Ge 18-22 at.%",
        "max_target_probability": float(target_prob[best_idx]),
        "best_hcl_ratio": float(H[best_idx]),
        "best_geh4_ratio": float(G[best_idx]),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    FIG_DIR.mkdir(exist_ok=True)
    cfg = Config()
    p4 = run_phase4_calibration(cfg)

    plot_posterior_pairplot(p4, FIG_DIR / "inference_posterior_pairplot.png")
    plot_trace(p4, FIG_DIR / "inference_mcmc_trace.png")
    plot_credible_interval_for_query(p4, cfg, T_C=725.0, hcl_ratio=0.5, geh4_ratio=0.03,
                                     out=FIG_DIR / "inference_credible_interval_query.png")
    plot_extrapolation_superiority(p4, FIG_DIR / "inference_extrapolation_superiority.png")
    plot_bayesian_response_envelope(p4, FIG_DIR / "inference_bayesian_response_envelope.png")
    plot_process_window_probability_map(p4, FIG_DIR / "inference_process_window_map.png")


if __name__ == "__main__":
    main()
