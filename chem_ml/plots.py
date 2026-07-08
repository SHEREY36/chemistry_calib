"""
Reproduces Tomasini Figs. 1-5 as PNGs against the transcribed data/fitted model, plus a
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
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter

from chem_ml.calibration import mu_draws, posterior_mean_params, posterior_predict
from chem_ml.config import Config
from chem_ml.physics_core import ge_logmodel, gr_logmodel
from chem_ml.pipeline import (
    _FIG4_GR_PAPER,
    _FIG4_HCL_RATIO,
    _FIG4_TARGET_GE,
    _FIG4_T_C,
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

# Tomasini Figs. 4/5 report flow sensitivities, not derivatives per unit
# pressure ratio. DS1's published/transcribed appendix gives ratios, not the
# underlying MFC flow table, so we use one explicit effective DCS-flow
# conversion to put derivatives on the same order-of-magnitude axes as the
# paper: d(p_HCl/p_DCS)/d(HCl sccm)=1/F_DCS and
# d(p_GeH4/p_DCS)/d(10% GeH4-in-H2 sccm)=0.1/F_DCS.
_FIG45_EFFECTIVE_DCS_SCCM = 170.0
_GEH4_DILUTION_FRACTION = 0.10


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


def make_fig1_ds3_regime(out: Path) -> None:
    """Reproduce Tomasini Fig. 1: DS3 experimental GR vs GeH4/DCS, grouped
    by fixed HCl/DCS multiples. This figure is a data/regime illustration,
    not a fitted-model parity plot."""
    ds3 = load_all_datasets(Config()).filter(source_dataset="DS3")
    rows = ds3.rows
    hcl_levels = sorted({round(r.p_HCl, 7) for r in rows})
    f = hcl_levels[0]
    marker_cycle = ["o", "x", "D", "s", "*"]

    fig, ax = plt.subplots(figsize=(6.2, 4.7))
    for i, hcl in enumerate(hcl_levels):
        group = [r for r in rows if round(r.p_HCl, 7) == hcl]
        group = sorted(group, key=lambda r: r.p_GeH4)
        x = np.array([r.p_GeH4 for r in group])
        y = np.array([r.GR_nm_min for r in group])
        multiple = int(round(hcl / f))
        label = "f" if multiple == 1 else f"{multiple}f"
        marker = marker_cycle[i % len(marker_cycle)]
        ax.plot(x, y, "--", color="0.25", lw=1.1, alpha=0.85)
        ax.plot(x, y, marker, color="black", ms=6, label=label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.01, 0.2)
    ax.set_ylim(0.8, 150)
    ax.xaxis.set_major_locator(FixedLocator([0.01, 0.1, 0.2]))
    ax.xaxis.set_major_formatter(FixedFormatter(["0.01", "0.1", "0.2"]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(FixedLocator([1, 10, 100]))
    ax.yaxis.set_major_formatter(FixedFormatter(["1", "10", "100"]))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"$p_{\mathrm{GeH4}}/p_{\mathrm{DCS}}$")
    ax.set_ylabel("GR exp. (nm/min)")
    ax.text(0.014, 88, "Regime II", fontsize=14, weight="bold")
    ax.text(0.075, 4.5, "Regime I", fontsize=14, weight="bold")
    ax.text(0.012, 0.55, f"DS3, T=750°C, f=pHCl/pDCS={f:.4f}", fontsize=8.5,
            transform=ax.get_xaxis_transform())
    ax.legend(loc="lower center", ncol=5, fontsize=8, frameon=True, title="fixed HCl level")
    ax.set_title("Fig. 1 reproduction -- DS3 GR regimes")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("Wrote %s", out)


def _sensitivity_curves(p4: dict):
    """Recompute the Fig. 4/5 operating points (600-750C, pHCl/pDCS=0.34,
    pGeH4/pDCS solved per-T for 20% Ge) and all three partial derivatives
    (dT, dHCl, dGeH4) for both GR and Ge. Raw ratio-space derivatives are
    retained, then converted to the paper's sccm axes with the explicit
    effective-flow convention above."""
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

    hcl_ratio_per_sccm = 1.0 / _FIG45_EFFECTIVE_DCS_SCCM
    geh4_10_ratio_per_sccm = _GEH4_DILUTION_FRACTION / _FIG45_EFFECTIVE_DCS_SCCM

    T_C_grid = np.array(_FIG4_T_C)
    rows = []
    for Tc in T_C_grid:
        T_K = Tc + 273.15
        geh4 = solve_geh4(T_K)
        gr = float(gr_of(T_K, _FIG4_HCL_RATIO, geh4))
        dGR_dT = float(jax.grad(gr_of, argnums=0)(T_K, _FIG4_HCL_RATIO, geh4))
        dGR_dHCl = float(jax.grad(gr_of, argnums=1)(T_K, _FIG4_HCL_RATIO, geh4))
        dGR_dGeH4 = float(jax.grad(gr_of, argnums=2)(T_K, _FIG4_HCL_RATIO, geh4))
        dGe_dT = float(jax.grad(ge_of, argnums=0)(T_K, _FIG4_HCL_RATIO, geh4))
        dGe_dHCl = float(jax.grad(ge_of, argnums=1)(T_K, _FIG4_HCL_RATIO, geh4))
        dGe_dGeH4 = float(jax.grad(ge_of, argnums=2)(T_K, _FIG4_HCL_RATIO, geh4))
        rows.append(dict(
            T_C=Tc,
            geh4_ratio=geh4,
            GR_nm_min=gr,
            GR_paper_nm_min=_FIG4_GR_PAPER[float(Tc)],
            dGR_dT=dGR_dT,
            dGR_dHCl_ratio=dGR_dHCl,
            dGR_dGeH4_ratio=dGR_dGeH4,
            dGR_dHCl_sccm=dGR_dHCl * hcl_ratio_per_sccm,
            dGR_dGeH4_10pct_sccm=dGR_dGeH4 * geh4_10_ratio_per_sccm,
            dGe_dT_atpct=100.0 * dGe_dT,
            dGe_dHCl_ratio=100.0 * dGe_dHCl,
            dGe_dGeH4_ratio=100.0 * dGe_dGeH4,
            dGe_dHCl_sccm=100.0 * dGe_dHCl * hcl_ratio_per_sccm,
            dGe_dGeH4_10pct_sccm=100.0 * dGe_dGeH4 * geh4_10_ratio_per_sccm,
        ))
    return rows


def _add_geh4_top_axis(ax, rows):
    ax_top = ax.twiny()
    temps = [r["T_C"] for r in rows]
    labels = [f"{r['geh4_ratio']:.3f}" for r in rows]
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(temps)
    ax_top.set_xticklabels(labels)
    ax_top.set_xlabel(r"$p_{\mathrm{GeH4}}/p_{\mathrm{DCS}}$")
    return ax_top


def make_fig4_gr_sensitivity(p4: dict, out: Path) -> None:
    rows = _sensitivity_curves(p4)
    T_C = [r["T_C"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.plot(T_C, [abs(r["dGR_dHCl_sccm"]) for r in rows], "s--", color="black",
            label=r"$-\partial GR/\partial HCl$")
    ax.plot(T_C, [abs(r["dGR_dT"]) for r in rows], "P--", color="black",
            label=r"$\partial GR/\partial T$")
    ax.plot(T_C, [abs(r["dGR_dGeH4_10pct_sccm"]) for r in rows], "D--", color="black",
            label=r"$\partial GR/\partial GeH4$")
    ax.set_yscale("log")
    ax.set_xlim(575, 775)
    ax.set_ylim(1e-3, 10)
    ax.set_xticks([575, 625, 675, 725, 775])
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel(r"GR K$^{-1}$ ; GR sccm$^{-1}$")
    _add_geh4_top_axis(ax, rows)
    ax.text(585, 1.2, r"$p_{\mathrm{HCl}}/p_{\mathrm{DCS}}=0.34$", fontsize=11, weight="bold")
    ax.text(675, 0.004, f"flow scale: F_DCS={_FIG45_EFFECTIVE_DCS_SCCM:.0f} sccm", fontsize=7.5, color="0.35")
    ax.set_title("Fig. 4 reproduction -- GR first partial derivatives")
    ax.legend(fontsize=8, loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("Wrote %s", out)


def make_fig5_ge_sensitivity(p4: dict, out: Path) -> None:
    """Dual y-axis, matching the paper's own Fig. 5 layout."""
    rows = _sensitivity_curves(p4)
    T_C = [r["T_C"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(6.2, 4.8))
    ax2 = ax1.twinx()
    l1, = ax1.plot(T_C, [r["dGe_dHCl_sccm"] for r in rows], "s--", color="black",
                   label=r"$\partial x_{Ge}/\partial HCl$")
    l2, = ax1.plot(T_C, [r["dGe_dT_atpct"] for r in rows], "P--", color="black",
                   label=r"$\partial x_{Ge}/\partial T$")
    l3, = ax2.plot(T_C, [r["dGe_dGeH4_10pct_sccm"] for r in rows], "D--", color="black",
                   label=r"$\partial x_{Ge}/\partial GeH4$")
    ax1.axhline(0, color="0.45", lw=0.8)
    ax1.set_xlim(575, 775)
    ax1.set_ylim(-0.10, 0.06)
    ax2.set_ylim(-0.2, 0.8)
    ax1.set_xticks([600, 650, 700, 750])
    ax1.set_xlabel("Temperature (°C)")
    ax1.set_ylabel(r"%Ge K$^{-1}$ ; %Ge sccm$^{-1}$ (HCl)")
    ax2.set_ylabel(r"%Ge sccm$^{-1}$ (GeH4 10%)")
    _add_geh4_top_axis(ax1, rows)
    ax1.text(688, -0.092, r"$p_{\mathrm{HCl}}/p_{\mathrm{DCS}}=0.34$", fontsize=11, weight="bold")
    ax1.set_title("Fig. 5 reproduction -- %Ge first partial derivatives")
    ax1.legend(handles=[l1, l2, l3], fontsize=8, loc="center left", frameon=False)
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
    make_fig1_ds3_regime(FIG_DIR / "fig1_ds3_regime.png")
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
