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
    check_tomasini_acceptance,
    run_mcmc,
)
from chem_ml.config import Config
from chem_ml.features import build_features
from chem_ml.physics_core import b_logmodel, ge_logmodel, gr_logmodel
from chem_ml.registry import SpeciesRegistry
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
