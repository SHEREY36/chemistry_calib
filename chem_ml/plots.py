"""
Reproduces Tomasini Figs. 2-5 as PNGs against our fitted model, plus a
posterior-predictive calibration plot (how good is the model's own
uncertainty quantification, not just its point predictions).

Run: conda activate epitaxy && python -m chem_ml.plots
Writes PNGs to figures/.
"""
from __future__ import annotations

import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chem_ml.calibration import mu_draws, posterior_mean_params, posterior_predict
from chem_ml.config import Config
from chem_ml.physics_core import ge_logmodel, gr_logmodel
from chem_ml.pipeline import (
    _FIG4_HCL_RATIO,
    _FIG4_TARGET_GE,
    _GE_PARAM_NAMES,
    _GR_PARAM_NAMES,
    load_all_datasets,
    run_phase4_calibration,
    run_phase6_identifiability,
)
from chem_ml.calibration import ge_numpyro_model, gr_numpyro_model

log = logging.getLogger("chem_ml")

FIG_DIR = Path("figures")
_MARKERS = {765: "o", 745: "*", 725: "D", 705: "s", 670: "P", 645: "+", 630: "x", 605: "X"}


def _marker_for(T_C: float) -> str:
    return _MARKERS.get(int(round(T_C)), ".")


def plot_parity(ax_top, ax_bot, T_C: np.ndarray, y_true: np.ndarray, draws: np.ndarray,
                real_space: bool, xlabel: str, ylabel_top: str, ylabel_bot: str) -> None:
    """draws: (num_samples, N) posterior fitted-curve draws in LOG space.
    real_space=True exponentiates (GR); False leaves as-is (not used here,
    kept for generality)."""
    mean = draws.mean(axis=0)
    lo, hi = np.percentile(draws, 5, axis=0), np.percentile(draws, 95, axis=0)
    if real_space:
        mean, lo, hi, y_plot = np.exp(mean), np.exp(lo), np.exp(hi), y_true
    else:
        y_plot = y_true

    for Tc in sorted(set(T_C), reverse=True):
        m = T_C == Tc
        marker = _marker_for(Tc)
        ax_top.errorbar(mean[m], y_plot[m], xerr=[mean[m] - lo[m], hi[m] - mean[m]],
                        fmt=marker, ms=5, capsize=2, alpha=0.8, label=f"{Tc:.0f}°C")
    lims = [min(y_plot.min(), mean.min()) * 0.8, max(y_plot.max(), mean.max()) * 1.2]
    ax_top.plot(lims, lims, "k--", lw=1)
    ax_top.set_xscale("log"); ax_top.set_yscale("log")
    ax_top.set_xlim(lims); ax_top.set_ylim(lims)
    ax_top.set_ylabel(ylabel_top)
    ax_top.legend(fontsize=7, ncol=2, loc="upper left")
    ax_top.set_title("a) experimental vs. predicted (error bars = 90% posterior CI)")

    residual = mean - y_plot
    for Tc in sorted(set(T_C), reverse=True):
        m = T_C == Tc
        ax_bot.plot(mean[m], residual[m], _marker_for(Tc), ms=5, alpha=0.8)
    ax_bot.axhline(0, color="k", ls="--", lw=1)
    ax_bot.set_xscale("log")
    ax_bot.set_xlabel(xlabel)
    ax_bot.set_ylabel(ylabel_bot)
    ax_bot.set_title("b) residual (predicted - experimental) vs. predicted")


def make_fig2_gr_parity(p4: dict, out: Path) -> None:
    ds1 = load_all_datasets(Config()).filter(source_dataset="DS1")
    fb1 = p4["features_ds1"]
    T_C = np.array([r.T_K - 273.15 for r in ds1.rows])
    draws = np.asarray(mu_draws(gr_logmodel, p4["mcmc_gr"], fb1.X, _GR_PARAM_NAMES))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 8), height_ratios=[2, 1])
    plot_parity(ax1, ax2, T_C, p4["y_gr"], draws, real_space=True,
               xlabel="GR calc. (nm/min)", ylabel_top="GR exp. (nm/min)",
               ylabel_bot="Residual (nm/min)")
    fig.suptitle("Fig. 2 reproduction -- GR parity (Eq. 10, DS1)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("Wrote %s", out)


def make_fig3_ge_parity(p4: dict, out: Path) -> None:
    ds1 = load_all_datasets(Config()).filter(source_dataset="DS1")
    fb1 = p4["features_ds1"]
    T_C = np.array([r.T_K - 273.15 for r in ds1.rows])
    y_ge = p4["y_ge"]
    y_ratio = y_ge / (1 - y_ge)
    draws = np.asarray(mu_draws(ge_logmodel, p4["mcmc_ge"], fb1.X, _GE_PARAM_NAMES))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 8), height_ratios=[2, 1])
    plot_parity(ax1, ax2, T_C, y_ratio, draws, real_space=True,
               xlabel="x/(1-x) calc.", ylabel_top="x/(1-x) exp.",
               ylabel_bot="Residual")
    ax1.set_xscale("linear"); ax1.set_yscale("linear")
    ax2.set_xscale("linear")
    fig.suptitle("Fig. 3 reproduction -- Ge/Si ratio parity (Eq. 15, DS1)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("Wrote %s", out)


def _sensitivity_curves(p4: dict):
    """Recompute the Fig. 4/5 operating points (600-750C, pHCl/pDCS=0.34,
    pGeH4/pDCS solved per-T for 20% Ge) and all three partial derivatives
    (dT, dHCl, dGeH4) for both GR and Ge, in ratio-space units (see
    run_phase6_identifiability's unit caveat -- no sccm<->ratio conversion
    is available, so pHCl/pGeH4 derivatives are per-unit-ratio, not
    per-sccm like the paper's axes)."""
    fb1 = p4["features_ds1"]
    mu, sd = fb1.invT_scaler
    gr_params = posterior_mean_params(p4["mcmc_gr"], _GR_PARAM_NAMES)
    ge_params = posterior_mean_params(p4["mcmc_ge"], _GE_PARAM_NAMES)

    def solve_geh4(T_K):
        target_ln_ratio = float(np.log(_FIG4_TARGET_GE / (1 - _FIG4_TARGET_GE)))
        invT_std = (1.0 / T_K - mu) / sd
        ln_geh4 = (target_ln_ratio - ge_params["lnK_Ge"] - ge_params["kappa_Ge"] * invT_std
                   - ge_params["dgamma_HCl"] * np.log(_FIG4_HCL_RATIO)) / ge_params["dgamma_GeH4"]
        return float(np.exp(ln_geh4))

    def gr_of(T_K, hcl, geh4):
        invT_std = (1.0 / T_K - mu) / sd
        X = jnp.array([[invT_std, jnp.log(hcl), jnp.log(geh4), 0.0]])
        return jnp.exp(gr_logmodel(gr_params, X))[0]

    def ge_of(T_K, hcl, geh4):
        invT_std = (1.0 / T_K - mu) / sd
        X = jnp.array([[invT_std, jnp.log(hcl), jnp.log(geh4), 0.0]])
        ratio = jnp.exp(ge_logmodel(ge_params, X))[0]
        return ratio / (1.0 + ratio)

    T_C_grid = np.array([600.0, 650.0, 700.0, 750.0])
    rows = []
    for Tc in T_C_grid:
        T_K = Tc + 273.15
        geh4 = solve_geh4(T_K)
        dGR_dT = float(jax.grad(gr_of, argnums=0)(T_K, _FIG4_HCL_RATIO, geh4))
        dGR_dHCl = float(jax.grad(gr_of, argnums=1)(T_K, _FIG4_HCL_RATIO, geh4))
        dGR_dGeH4 = float(jax.grad(gr_of, argnums=2)(T_K, _FIG4_HCL_RATIO, geh4))
        dGe_dT = float(jax.grad(ge_of, argnums=0)(T_K, _FIG4_HCL_RATIO, geh4))
        dGe_dHCl = float(jax.grad(ge_of, argnums=1)(T_K, _FIG4_HCL_RATIO, geh4))
        dGe_dGeH4 = float(jax.grad(ge_of, argnums=2)(T_K, _FIG4_HCL_RATIO, geh4))
        rows.append(dict(T_C=Tc, geh4_ratio=geh4,
                         dGR_dT=dGR_dT, dGR_dHCl=dGR_dHCl, dGR_dGeH4=dGR_dGeH4,
                         dGe_dT=dGe_dT, dGe_dHCl=dGe_dHCl, dGe_dGeH4=dGe_dGeH4))
    return rows


def make_fig4_gr_sensitivity(p4: dict, out: Path) -> None:
    rows = _sensitivity_curves(p4)
    T_C = [r["T_C"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(T_C, [abs(r["dGR_dHCl"]) for r in rows], "s--", label="|dGR/d(pHCl/pDCS)|")
    ax.plot(T_C, [abs(r["dGR_dT"]) for r in rows], "+--", label="dGR/dT (nm/min/K)")
    ax.plot(T_C, [abs(r["dGR_dGeH4"]) for r in rows], "D--", label="dGR/d(pGeH4/pDCS)")
    ax.set_yscale("log")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("|partial derivative| (ratio-space units -- see caveat)")
    ax.set_title("Fig. 4 reproduction -- GR sensitivity vs. T\n"
                 "(pHCl/pDCS=0.34 fixed, pGeH4/pDCS solved for 20% Ge)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("Wrote %s", out)


def make_fig5_ge_sensitivity(p4: dict, out: Path) -> None:
    """Dual y-axis, matching the paper's own Fig. 5 layout: d(xGe)/d(pGeH4)
    is ~2 orders of magnitude larger than d(xGe)/d(pHCl) and d(xGe)/dT in
    our ratio-space units (pGeH4/pDCS lives on a 0.01-0.04 scale, so its
    derivative is correspondingly inflated) -- a single linear axis
    squashes the other two curves to a flat line at 0, same reason the
    paper itself split this onto two axes."""
    rows = _sensitivity_curves(p4)
    T_C = [r["T_C"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(6.5, 5))
    ax2 = ax1.twinx()
    l1, = ax1.plot(T_C, [r["dGe_dHCl"] for r in rows], "s--", color="tab:blue",
                   label="d(xGe)/d(pHCl/pDCS)  [left axis]")
    l2, = ax1.plot(T_C, [r["dGe_dT"] for r in rows], "+--", color="tab:green",
                   label="d(xGe)/dT (frac/K)  [left axis]")
    l3, = ax2.plot(T_C, [r["dGe_dGeH4"] for r in rows], "D--", color="tab:orange",
                   label="d(xGe)/d(pGeH4/pDCS)  [right axis]")
    ax1.axhline(0, color="gray", lw=0.8)
    ax1.set_xlabel("Temperature (°C)")
    ax1.set_ylabel("d(xGe)/d(pHCl/pDCS), d(xGe)/dT")
    ax2.set_ylabel("d(xGe)/d(pGeH4/pDCS)  [ratio-space, see caveat]")
    ax1.set_title("Fig. 5 reproduction -- %Ge sensitivity vs. T\n"
                  "(pHCl/pDCS=0.34 fixed, pGeH4/pDCS solved for 20% Ge)")
    ax1.legend(handles=[l1, l2, l3], fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("Wrote %s", out)


def make_uncertainty_calibration(p4: dict, cfg: Config, out: Path) -> dict:
    """Posterior-PREDICTIVE (parameter uncertainty + observation noise, via
    calibration.posterior_predict) calibration check: for nominal credible
    levels (50/80/90/95%), what fraction of DS1's actual GR/Ge values fall
    inside that pointwise credible interval? A well-calibrated model traces
    the diagonal. This is an IN-SAMPLE check (DS1 was the training set, not
    held out), so it validates that the model's stated uncertainty is
    internally consistent, not that it generalizes -- Phase 7's DS3/DS4
    results are the actual out-of-sample evidence for that."""
    fb1 = p4["features_ds1"]
    gr_pred_log = np.asarray(posterior_predict(gr_numpyro_model, p4["mcmc_gr"], fb1.X, cfg))
    ge_pred_log = np.asarray(posterior_predict(ge_numpyro_model, p4["mcmc_ge"], fb1.X, cfg))
    y_gr_log = np.log(p4["y_gr"])
    y_ge = p4["y_ge"]
    y_ge_log = np.log(y_ge / (1 - y_ge))

    levels = [0.5, 0.8, 0.9, 0.95]
    gr_coverage, ge_coverage = [], []
    for lev in levels:
        lo_p, hi_p = (1 - lev) / 2 * 100, (1 + lev) / 2 * 100
        gr_lo, gr_hi = np.percentile(gr_pred_log, lo_p, axis=0), np.percentile(gr_pred_log, hi_p, axis=0)
        ge_lo, ge_hi = np.percentile(ge_pred_log, lo_p, axis=0), np.percentile(ge_pred_log, hi_p, axis=0)
        gr_coverage.append(float(np.mean((y_gr_log >= gr_lo) & (y_gr_log <= gr_hi))))
        ge_coverage.append(float(np.mean((y_ge_log >= ge_lo) & (y_ge_log <= ge_hi))))

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    ax.plot(levels, gr_coverage, "o-", label="GR model")
    ax.plot(levels, ge_coverage, "s-", label="Ge/Si model")
    ax.set_xlabel("Nominal credible level")
    ax.set_ylabel("Empirical coverage (DS1, in-sample)")
    ax.set_title("Posterior-predictive calibration")
    ax.legend()
    ax.set_xlim(0.4, 1.0); ax.set_ylim(0.4, 1.05)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("Wrote %s", out)

    return {"levels": levels, "gr_coverage": gr_coverage, "ge_coverage": ge_coverage}


def generate_all_figures(cfg: Config, p4: dict) -> dict:
    """Generate all 5 PNGs from an already-fitted Phase 4 result (reused by
    chem_ml.report so the validation report doesn't re-run MCMC a second
    time just to make plots)."""
    FIG_DIR.mkdir(exist_ok=True)
    make_fig2_gr_parity(p4, FIG_DIR / "fig2_gr_parity.png")
    make_fig3_ge_parity(p4, FIG_DIR / "fig3_ge_parity.png")
    make_fig4_gr_sensitivity(p4, FIG_DIR / "fig4_gr_sensitivity.png")
    make_fig5_ge_sensitivity(p4, FIG_DIR / "fig5_ge_sensitivity.png")
    calib = make_uncertainty_calibration(p4, cfg, FIG_DIR / "uncertainty_calibration.png")
    log.info("Calibration: %s", calib)
    return calib


def main() -> dict:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = Config()
    p4 = run_phase4_calibration(cfg)
    return generate_all_figures(cfg, p4)


if __name__ == "__main__":
    main()
