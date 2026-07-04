"""
Reactor-transfer random effect (Layer B, data-only version): given theta_chem
FROZEN at its Phase 4 posterior mean (fit on DS1/ASM_Epsilon only), fit a
small per-reactor delta_r = {alpha_HCl, alpha_GeH4, eta_GR, eta_Ge} to
recover DS3 (Hartmann) and DS4 (Tan) using only that low-dimensional offset
(Phase 7.2 cross-reactor validation).

DESIGN NOTE -- dT_r is dropped, not "hierarchical partial pooling":
  The doc's generic delta_r is {dT_r, alpha_{i,r}, eta_r}. dT_r is NOT
  identifiable here:
    - DS3 is measured at a SINGLE fixed T=750C (35 rows, all same T). Any
      dT_r shift is then just 1/(T+dT_r) evaluated at one T -- a constant
      across every row, perfectly collinear with ln(eta_r) in log space.
      There is no way to separate a temperature offset from a scale factor
      with single-temperature data.
    - DS4 has no GR at all (Phase 1 data gap: no growth time in the
      appendix), so only the Ge/Si channel is available, over a narrow
      20 C range (740-760 C) -- too weak to pin dT_r independently of
      eta_Ge.
  So dT_r is fixed at 0 and eta_r absorbs any true reactor-level T offset.
  This is a smaller delta_r (3-4 params) than the generic doc description,
  which is explicitly allowed ("~3-5 delta_r parameters").

Also: DS4 flows trace B2H6, but the frozen Ge/Si theta_chem (ge_logmodel)
structurally never reads the B2H6 feature column (same anti-contamination
guarantee as Phase 2) -- consistent with Tomasini's own DS4 B-order on the
Ge/Si ratio being small (0.007-0.04, Eqs. 18-19).

CFD-INFORMED PRIORS (Phase 9 hookup): both model functions below accept an
optional `alpha_priors` dict -- exactly the shape returned by
cfd/transfer.py:extract_transfer_priors() -- with keys "ln_alpha_HCl" and
"ln_alpha_GeH4", each a (mean, std) tuple. When omitted, they fall back to
the weak Normal(0, 1) used throughout Phases 0-8 (data-only, no CFD
available). Passing CFD-derived priors here is what actually resolves the
alpha/eta near-degeneracy documented in METHODOLOGY.md sec 8 (posterior
correlation up to -0.97 on DS3 under the weak prior) -- CFD supplies
information about alpha that the wafer outcomes structurally cannot.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from chem_ml.physics_core import ge_logmodel, gr_logmodel

_DELTA_R_PRIOR_SD = 1.0  # weak prior on ln_alpha_i, ln_eta -- data-dominated with 18-35 rows
_WEAK_ALPHA_PRIOR = {"ln_alpha_HCl": (0.0, _DELTA_R_PRIOR_SD), "ln_alpha_GeH4": (0.0, _DELTA_R_PRIOR_SD)}


def reactor_transfer_model_gr_ge(X: jnp.ndarray, y_gr_log: jnp.ndarray | None,
                                 y_ge_log: jnp.ndarray | None, theta_gr: dict, theta_ge: dict,
                                 alpha_priors: dict | None = None):
    """DS3-shaped reactor: both GR and Ge/Si channels available, sharing the
    SAME alpha_HCl/alpha_GeH4 (a reactor-level partial-pressure delivery
    offset is a property of the reactor, not of which observable reads it)."""
    ap = alpha_priors or _WEAK_ALPHA_PRIOR
    ln_alpha_HCl = numpyro.sample("ln_alpha_HCl", dist.Normal(*ap["ln_alpha_HCl"]))
    ln_alpha_GeH4 = numpyro.sample("ln_alpha_GeH4", dist.Normal(*ap["ln_alpha_GeH4"]))
    ln_eta_GR = numpyro.sample("ln_eta_GR", dist.Normal(0.0, _DELTA_R_PRIOR_SD))
    ln_eta_Ge = numpyro.sample("ln_eta_Ge", dist.Normal(0.0, _DELTA_R_PRIOR_SD))
    sigma_gr = numpyro.sample("sigma_GR_r", dist.HalfNormal(0.5))
    sigma_ge = numpyro.sample("sigma_Ge_r", dist.HalfNormal(0.5))

    X_eff = X.at[:, 1].add(ln_alpha_HCl).at[:, 2].add(ln_alpha_GeH4)
    mu_gr = ln_eta_GR + gr_logmodel(theta_gr, X_eff)
    mu_ge = ln_eta_Ge + ge_logmodel(theta_ge, X_eff)
    numpyro.sample("obs_GR_r", dist.Normal(mu_gr, sigma_gr), obs=y_gr_log)
    numpyro.sample("obs_Ge_r", dist.Normal(mu_ge, sigma_ge), obs=y_ge_log)


def reactor_transfer_model_ge_only(X: jnp.ndarray, y_ge_log: jnp.ndarray | None, theta_ge: dict,
                                   alpha_priors: dict | None = None):
    """DS4-shaped reactor: no GR available (Phase 1 data gap), Ge/Si only."""
    ap = alpha_priors or _WEAK_ALPHA_PRIOR
    ln_alpha_HCl = numpyro.sample("ln_alpha_HCl", dist.Normal(*ap["ln_alpha_HCl"]))
    ln_alpha_GeH4 = numpyro.sample("ln_alpha_GeH4", dist.Normal(*ap["ln_alpha_GeH4"]))
    ln_eta_Ge = numpyro.sample("ln_eta_Ge", dist.Normal(0.0, _DELTA_R_PRIOR_SD))
    sigma_ge = numpyro.sample("sigma_Ge_r", dist.HalfNormal(0.5))

    X_eff = X.at[:, 1].add(ln_alpha_HCl).at[:, 2].add(ln_alpha_GeH4)
    mu_ge = ln_eta_Ge + ge_logmodel(theta_ge, X_eff)
    numpyro.sample("obs_Ge_r", dist.Normal(mu_ge, sigma_ge), obs=y_ge_log)
