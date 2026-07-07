"""
NumPyro models + NUTS/MCMC runners for the GR, Ge/Si, and B/Si power-law
cores, plus the Phase-4.3 reproduction acceptance gates.
"""
from __future__ import annotations

from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive

from chem_ml.config import Config, PriorConfig
from chem_ml.physics_core import b_logmodel, c_logmodel, destandardize_kappa, ge_logmodel, gr_logmodel


def _sample_normal(name: str, ms: tuple[float, float]):
    return numpyro.sample(name, dist.Normal(ms[0], ms[1]))


def gr_numpyro_model(X: jnp.ndarray, y_log: Optional[jnp.ndarray], pri: PriorConfig):
    p = {
        "lnK_GR": _sample_normal("lnK_GR", pri.lnK_GR),
        "kappa_GR": _sample_normal("kappa_GR", pri.kappa_GR),
        "gamma_HCl": _sample_normal("gamma_HCl", pri.gamma_HCl),
        "gamma_GeH4": _sample_normal("gamma_GeH4", pri.gamma_GeH4),
    }
    sigma = numpyro.sample("sigma_GR", dist.HalfNormal(pri.sigma_halfnormal))
    mu = gr_logmodel(p, X)
    numpyro.sample("obs_GR", dist.Normal(mu, sigma), obs=y_log)


def ge_numpyro_model(X: jnp.ndarray, y_log: Optional[jnp.ndarray], pri: PriorConfig):
    p = {
        "lnK_Ge": _sample_normal("lnK_Ge", pri.lnK_Ge),
        "kappa_Ge": _sample_normal("kappa_Ge", pri.kappa_Ge),
        "dgamma_HCl": _sample_normal("dgamma_HCl", pri.dgamma_HCl),
        "dgamma_GeH4": _sample_normal("dgamma_GeH4", pri.dgamma_GeH4),
    }
    sigma = numpyro.sample("sigma_Ge", dist.HalfNormal(pri.sigma_halfnormal))
    mu = ge_logmodel(p, X)
    numpyro.sample("obs_Ge", dist.Normal(mu, sigma), obs=y_log)


def b_numpyro_model(X: jnp.ndarray, y_log: Optional[jnp.ndarray], pri: PriorConfig):
    p = {
        "lnK_B": _sample_normal("lnK_B", pri.lnK_B),
        "beta_HCl": _sample_normal("beta_HCl", pri.beta_HCl),
        "beta_GeH4": _sample_normal("beta_GeH4", pri.beta_GeH4),
        "beta_B2H6": _sample_normal("beta_B2H6", pri.beta_B2H6),
    }
    sigma = numpyro.sample("sigma_B", dist.HalfNormal(pri.sigma_halfnormal))
    mu = b_logmodel(p, X)
    numpyro.sample("obs_B", dist.Normal(mu, sigma), obs=y_log)


def c_numpyro_model(X: jnp.ndarray, y_log: Optional[jnp.ndarray], pri: PriorConfig):
    p = {
        "lnK_C": _sample_normal("lnK_C", pri.lnK_C),
        "kappa_C": _sample_normal("kappa_C", pri.kappa_C),
        "cgamma_HCl": _sample_normal("cgamma_HCl", pri.cgamma_HCl),
        "cgamma_GeH4": _sample_normal("cgamma_GeH4", pri.cgamma_GeH4),
        "cgamma_MMS": _sample_normal("cgamma_MMS", pri.cgamma_MMS),
    }
    sigma = numpyro.sample("sigma_C", dist.HalfNormal(pri.sigma_halfnormal))
    mu = c_logmodel(p, X)
    numpyro.sample("obs_C", dist.Normal(mu, sigma), obs=y_log)


def run_mcmc(model_fn: Callable, X: jnp.ndarray, y_log: jnp.ndarray, cfg: Config) -> MCMC:
    """Fit one observable's physics core with NUTS. Returns fitted MCMC object.
    Check R-hat<1.01, ESS, 0 divergences downstream (Phase 4.2)."""
    kernel = NUTS(model_fn, target_accept_prob=cfg.mcmc.target_accept)
    mcmc = MCMC(kernel, num_warmup=cfg.mcmc.num_warmup, num_samples=cfg.mcmc.num_samples,
                num_chains=cfg.mcmc.num_chains, progress_bar=True)
    rng = jax.random.PRNGKey(cfg.mcmc.seed)
    mcmc.run(rng, X=X, y_log=y_log, pri=cfg.priors)
    return mcmc


def posterior_predict(model_fn: Callable, mcmc: MCMC, X: jnp.ndarray, cfg: Config) -> jnp.ndarray:
    """Posterior predictive in log space. Returns (num_samples, N)."""
    pred = Predictive(model_fn, posterior_samples=mcmc.get_samples())
    rng = jax.random.PRNGKey(cfg.mcmc.seed + 1)
    out = pred(rng, X=X, y_log=None, pri=cfg.priors)
    site = [k for k in out if k.startswith("obs_")][0]
    return out[site]


def mu_draws(logmodel: Callable, mcmc: MCMC, X: jnp.ndarray, param_names: list[str]) -> jnp.ndarray:
    """Noiseless fitted-curve draws: logmodel(theta_s, X) for every posterior
    sample s, WITHOUT the observation-noise term that Predictive/posterior_predict
    would add. This is what Tomasini's own Table 1/2 R^2 is computed against
    (the deterministic best-fit curve), not simulated noisy resamples.
    Returns (num_samples, N) in log space."""
    s = mcmc.get_samples()
    n_samples = len(s[param_names[0]])

    def one_draw(i):
        p = {n: s[n][i] for n in param_names}
        return logmodel(p, X)

    return jax.vmap(one_draw)(jnp.arange(n_samples))


def posterior_mean_params(mcmc: MCMC, names: list[str]) -> dict:
    s = mcmc.get_samples()
    return {n: float(jnp.mean(s[n])) for n in names}


def posterior_to_normal_prior(mcmc: MCMC, names: list[str], widen_factor: float = 2.0) -> dict[str, tuple]:
    """Turn a fitted posterior into a NEW Normal prior (mean, std) per
    parameter -- "yesterday's posterior is tomorrow's prior" (Phase 4.4),
    used by pipeline.run_phase4_warm_start for additive/incremental
    calibration on new data only.

    WHY widen_factor (default 2x) instead of using the raw posterior std:
    the GR/Ge/B models are linear-Gaussian in log-space (Normal likelihood,
    Normal prior, log-linear mean function), so *exact* sequential Bayesian
    updating (treating today's posterior as tomorrow's prior) is close to
    correct here -- this is the Normal-Inverse-Gamma conjugate family, not
    an ad hoc approximation, UNLIKE the residual NN (point-estimate,
    non-Bayesian) or a genuinely nonlinear model, where this trick would
    be far shakier. The widening is a separate, deliberate safety margin
    for what NUTS's finite posterior SAMPLE doesn't capture exactly (a
    thousand posterior draws approximate a continuous distribution, they
    don't equal it) and for the fact that this is only ever used between
    periodic full pooled refits (see run_phase4_warm_start), not as a
    permanent substitute for one."""
    s = mcmc.get_samples()
    return {n: (float(jnp.mean(s[n])), float(jnp.std(s[n])) * widen_factor) for n in names}


# ---- convergence diagnostics (Phase 4.2) -----------------------------------
def diagnostics(mcmc: MCMC) -> dict:
    """R-hat / ESS / divergence summary. Gate: max R-hat < 1.01, min ESS
    reasonable, zero divergences."""
    import numpyro.diagnostics as npd
    samples = mcmc.get_samples(group_by_chain=True)
    rhats = {k: float(np.max(np.asarray(npd.gelman_rubin(v)))) for k, v in samples.items()}
    ess = {k: float(np.min(np.asarray(npd.effective_sample_size(v)))) for k, v in samples.items()}
    n_diverging = int(np.sum(mcmc.get_extra_fields().get("diverging", np.array([0]))))
    return {
        "max_rhat": max(rhats.values()) if rhats else float("nan"),
        "min_ess": min(ess.values()) if ess else float("nan"),
        "n_diverging": n_diverging,
        "rhats": rhats,
        "ess": ess,
    }


# ---- acceptance gates (Phase 4.3) ------------------------------------------
def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / (ss_tot + 1e-30)


def check_tomasini_acceptance(mcmc_gr: MCMC, mcmc_ge: MCMC,
                              y_gr_true: np.ndarray, y_ge_true: np.ndarray,
                              gr_pred_log: np.ndarray, ge_pred_log: np.ndarray,
                              invT_scaler: tuple[float, float], cfg: Config) -> dict:
    """Return a report dict of the objective acceptance metrics (Table in
    build_steps_and_cfd_integration.md Phase 4.3). Predictions passed in are
    LOG-SPACE posterior-predictive draws (num_samples, N); we use the
    posterior mean prediction for the parity check."""
    kappa_gr = destandardize_kappa(float(np.mean(mcmc_gr.get_samples()["kappa_GR"])), invT_scaler)
    gr_pred_mean = np.exp(np.asarray(gr_pred_log).mean(0))
    ge_ratio_pred_mean = np.exp(np.asarray(ge_pred_log).mean(0))
    report = {
        "R2_GR": r2_score(y_gr_true, gr_pred_mean),
        "R2_Ge": r2_score(y_ge_true / (1 - y_ge_true), ge_ratio_pred_mean),
        "kappa_GR_K": kappa_gr,
        "kappa_GR_within_10pct": abs(abs(kappa_gr) - 24507.0) / 24507.0 <= cfg.kappa_GR_tol_frac,
        "gamma_HCl": float(np.mean(mcmc_gr.get_samples()["gamma_HCl"])),
        "gamma_GeH4": float(np.mean(mcmc_gr.get_samples()["gamma_GeH4"])),
    }
    report["PASS"] = (report["R2_GR"] >= cfg.r2_target
                      and report["R2_Ge"] >= cfg.r2_target
                      and report["kappa_GR_within_10pct"])
    return report
