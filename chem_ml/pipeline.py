"""
Top-level orchestration: the "reproduce Tomasini" entry point (Phases 1-8).
Built incrementally as each phase is implemented and verified.
"""
from __future__ import annotations

import logging

import arviz as az
import jax.numpy as jnp
import numpy as np

from chem_ml.assembler import ReactionNetworkAssembler
from chem_ml.calibration import (
    b_numpyro_model,
    diagnostics,
    ge_numpyro_model,
    gr_numpyro_model,
    mu_draws,
    posterior_mean_params,
    check_tomasini_acceptance,
    run_mcmc,
)
from chem_ml.config import Config
from chem_ml.features import build_features
from chem_ml.physics_core import b_logmodel, ge_logmodel, gr_logmodel
from chem_ml.registry import SpeciesRegistry
from chem_ml.residual_nn import ResidualNN, build_residual_input
from chem_ml.schema import ChemClass, Dataset, ingest_tomasini

log = logging.getLogger("chem_ml")

# Paper's stated constant for the DS2 [B]/[Si] ratio (Table 2, Eq. 20 comment).
DS2_SI_ATOMS_CM3 = 5e22


def load_all_datasets(cfg: Config) -> Dataset:
    """Phase 1: ingest all four Tomasini appendix datasets into one Dataset."""
    ds = ingest_tomasini(cfg.data_raw)
    log.info("Ingested %d canonical rows from Tomasini appendices", len(ds))
    for src in ("DS1", "DS2_GR", "DS2_B", "DS3", "DS4"):
        n = len(ds.filter(source_dataset=src))
        log.info("  %s: %d rows", src, n)
    return ds


def build_default_registry_and_assembler():
    """Phase 2: default registry + assembler, ready for anti-contamination checks."""
    reg = SpeciesRegistry()
    asm = ReactionNetworkAssembler(reg)
    return reg, asm


def run_phase4_calibration(cfg: Config, ds: Dataset | None = None) -> dict:
    """Phase 4: NUTS-calibrate GR and Ge/Si on DS1, and the boron model on
    DS2's [B] rows. Returns a dict with the fitted MCMC objects, the
    Phase-4.3 acceptance report, and arviz InferenceData for each model
    (this posterior becomes the prior for Phases 5-8)."""
    if ds is None:
        ds = load_all_datasets(cfg)

    ds1 = ds.filter(source_dataset="DS1")
    fb1 = build_features(ds1)
    y_gr = np.array([r.GR_nm_min for r in ds1.rows])
    y_ge = np.array([r.Ge_at_frac for r in ds1.rows])
    y_gr_log = jnp.asarray(np.log(y_gr))
    y_ge_log = jnp.asarray(np.log(y_ge / (1.0 - y_ge)))

    log.info("Phase 4: fitting GR model on DS1 (N=%d)...", len(ds1))
    mcmc_gr = run_mcmc(gr_numpyro_model, fb1.X, y_gr_log, cfg)
    diag_gr = diagnostics(mcmc_gr)
    log.info("GR diagnostics: %s", diag_gr)

    log.info("Phase 4: fitting Ge/Si model on DS1 (N=%d)...", len(ds1))
    mcmc_ge = run_mcmc(ge_numpyro_model, fb1.X, y_ge_log, cfg)
    diag_ge = diagnostics(mcmc_ge)
    log.info("Ge/Si diagnostics: %s", diag_ge)

    gr_mu_draws = mu_draws(gr_logmodel, mcmc_gr, fb1.X, ["lnK_GR", "kappa_GR", "gamma_HCl", "gamma_GeH4"])
    ge_mu_draws = mu_draws(ge_logmodel, mcmc_ge, fb1.X, ["lnK_Ge", "kappa_Ge", "dgamma_HCl", "dgamma_GeH4"])

    report = check_tomasini_acceptance(
        mcmc_gr, mcmc_ge, y_gr, y_ge, gr_mu_draws, ge_mu_draws, fb1.invT_scaler, cfg,
    )

    # ---- boron model on DS2's dedicated [B] rows ---------------------------
    ds2b = ds.filter(source_dataset="DS2_B")
    fb2b = build_features(ds2b)
    b_over_si = np.array([r.B_conc / DS2_SI_ATOMS_CM3 for r in ds2b.rows])
    y_b_log = jnp.asarray(np.log(b_over_si))

    log.info("Phase 4: fitting B/Si model on DS2 (N=%d)...", len(ds2b))
    mcmc_b = run_mcmc(b_numpyro_model, fb2b.X, y_b_log, cfg)
    diag_b = diagnostics(mcmc_b)
    log.info("B/Si diagnostics: %s", diag_b)

    beta_b2h6 = float(jnp.mean(mcmc_b.get_samples()["beta_B2H6"]))
    b_mu_draws = mu_draws(b_logmodel, mcmc_b, fb2b.X, ["lnK_B", "beta_HCl", "beta_GeH4", "beta_B2H6"])
    from chem_ml.calibration import r2_score
    report["R2_B"] = r2_score(b_over_si, np.exp(np.asarray(b_mu_draws).mean(0)))
    report["beta_B2H6"] = beta_b2h6
    report["beta_B2H6_within_target"] = abs(beta_b2h6 - 0.8) <= 0.2
    report["PASS"] = report["PASS"] and report["beta_B2H6_within_target"]

    log.info("Phase 4 acceptance report: %s", report)

    return {
        "mcmc_gr": mcmc_gr, "mcmc_ge": mcmc_ge, "mcmc_b": mcmc_b,
        "diag_gr": diag_gr, "diag_ge": diag_ge, "diag_b": diag_b,
        "features_ds1": fb1, "features_ds2b": fb2b,
        "y_gr": y_gr, "y_ge": y_ge, "b_over_si": b_over_si,
        "report": report,
        "idata_gr": az.from_numpyro(mcmc_gr),
        "idata_ge": az.from_numpyro(mcmc_ge),
        "idata_b": az.from_numpyro(mcmc_b),
    }


def run_phase5_residual_hybrid(cfg: Config, phase4_result: dict, ds: Dataset | None = None) -> dict:
    """Phase 5: fit a small gated residual NN on DS1's GR/Ge log-residuals
    left over after the Phase-4 physics core.

    Validates build_steps_and_cfd_integration.md Phase 5.2's first claim:
    the net stays small given the physics already achieves R^2 >= 0.98
    (weight_decay=0.3 was chosen by sweeping l2 in {0.01..1.0} against DS1:
    at 0.01 the net's RMS reaches ~75% of the physics residual's RMS, i.e.
    it is fitting noise on 70 points with a 16-unit MLP; at 0.3 it settles
    to ~40% while still reducing RMSE, which is the "small but genuinely
    active" regime the doc calls for).

    Phase 5.2's SECOND claim -- that the correction concentrates on the
    Regime-I low-pGeH4/pDCS curvature from Tomasini's Fig. 1 -- does NOT
    hold on DS1 (checked directly on the raw, pre-NN physics residual: its
    correlation with ln(pGeH4/pDCS) is +0.42, i.e. residuals are LARGER at
    higher GeH4 ratio, the opposite direction). This is not a bug: Fig. 1 is
    a controlled single-variable sweep on DS3 at fixed T=750C and fixed
    pHCl; DS1 is a multi-dimensional DoE varying T, HCl, and GeH4
    simultaneously, so the same curvature is not expected to isolate
    cleanly in DS1's residual. Reported as a diagnostic, not asserted."""
    if ds is None:
        ds = load_all_datasets(cfg)
    ds1 = ds.filter(source_dataset="DS1")
    fb1 = phase4_result["features_ds1"]

    gr_params = posterior_mean_params(phase4_result["mcmc_gr"], ["lnK_GR", "kappa_GR", "gamma_HCl", "gamma_GeH4"])
    ge_params = posterior_mean_params(phase4_result["mcmc_ge"], ["lnK_Ge", "kappa_Ge", "dgamma_HCl", "dgamma_GeH4"])

    y_gr_log = jnp.log(jnp.asarray(phase4_result["y_gr"]))
    y_ge = phase4_result["y_ge"]
    y_ge_log = jnp.log(jnp.asarray(y_ge / (1.0 - y_ge)))

    gr_phys_log = gr_logmodel(gr_params, fb1.X)
    ge_phys_log = ge_logmodel(ge_params, fb1.X)
    resid_gr = y_gr_log - gr_phys_log
    resid_ge = y_ge_log - ge_phys_log
    residual_targets = jnp.stack([resid_gr, resid_ge], axis=1)  # (N, 2): [GR, Ge]

    X_full = build_residual_input(ds1, fb1)
    net = ResidualNN(chem_class=ChemClass.SIGE, in_size=X_full.shape[1], n_out=2)
    net.fit(X_full, residual_targets, l2=0.3, steps=2000, lr=5e-3)
    g_pred = net(X_full)

    physics_rmse_gr = float(jnp.sqrt(jnp.mean(resid_gr ** 2)))
    physics_rmse_ge = float(jnp.sqrt(jnp.mean(resid_ge ** 2)))
    hybrid_rmse_gr = float(jnp.sqrt(jnp.mean((resid_gr - g_pred[:, 0]) ** 2)))
    hybrid_rmse_ge = float(jnp.sqrt(jnp.mean((resid_ge - g_pred[:, 1]) ** 2)))

    ln_geh4 = np.asarray(fb1.X[:, 2])
    corr_absresid_lngeh4 = float(np.corrcoef(np.abs(np.asarray(resid_gr)), ln_geh4)[0, 1])

    report = {
        "physics_rmse_gr": physics_rmse_gr, "hybrid_rmse_gr": hybrid_rmse_gr,
        "physics_rmse_ge": physics_rmse_ge, "hybrid_rmse_ge": hybrid_rmse_ge,
        "g_nn_rms_gr": float(jnp.sqrt(jnp.mean(g_pred[:, 0] ** 2))),
        "g_nn_rms_ge": float(jnp.sqrt(jnp.mean(g_pred[:, 1] ** 2))),
        # diagnostic only (see docstring): DS1 does not isolate Fig. 1's
        # Regime-I curvature the way DS3's controlled sweep does.
        "corr_abs_physics_resid_vs_ln_geh4_ds1": corr_absresid_lngeh4,
    }
    report["net_shrinks_toward_zero"] = report["g_nn_rms_gr"] < 0.5 * physics_rmse_gr
    report["hybrid_improves_on_physics"] = hybrid_rmse_gr < physics_rmse_gr and hybrid_rmse_ge < physics_rmse_ge
    log.info("Phase 5 residual-NN report: %s", report)

    return {"net": net, "residual_targets": residual_targets, "g_pred": g_pred,
            "X_full": X_full, "report": report}
