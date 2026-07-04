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

from chem_ml.calibration import mu_draws, posterior_predict
from chem_ml.calibration import gr_numpyro_model
from chem_ml.config import Config
from chem_ml.physics_core import gr_logmodel
from chem_ml.pipeline import _GR_PARAM_NAMES, load_all_datasets, run_phase4_calibration

log = logging.getLogger("chem_ml")
FIG_DIR = Path("figures")


def plot_posterior_pairplot(p4: dict, out: Path) -> None:
    """Corner-style pairwise posterior scatter for the 4 GR parameters --
    what "MCMC posterior samples" actually look like, including the
    correlation structure between parameters (e.g. lnK_GR trades off
    against gamma_HCl/gamma_GeH4 the way any regression intercept trades
    off against its slopes)."""
    axes = az.plot_pair(p4["idata_gr"], var_names=_GR_PARAM_NAMES, kind="hexbin",
                        marginals=True, figsize=(8, 8))
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
    axes = az.plot_trace(p4["idata_gr"], var_names=_GR_PARAM_NAMES, figsize=(10, 8))
    fig = np.asarray(axes).flat[0].figure
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


if __name__ == "__main__":
    main()
