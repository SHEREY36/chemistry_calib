"""
Extract the setpoint->surface map from CFD-ACE+ runs, as INFORMATIVE priors
for reactor_transfer.py's alpha_{i,r} (Phase 9.4.3).

This is the direct payoff of Phase 9 for the identifiability problem
documented in METHODOLOGY.md sec 8: fitting alpha_{i,r}/eta_r from 18-35
wafer outcomes alone leaves them nearly degenerate (posterior correlation
up to -0.97 between ln_alpha_GeH4 and ln_eta_Ge on DS3 -- the data supplies
only ~2 effective numbers per reactor for 4 unknowns). CFD-ACE+ computes
alpha_{i,r} directly from species transport/depletion in the actual
geometry, which is INDEPENDENT information the wafer outcomes alone can't
supply -- feeding it in as a tight prior (rather than the current N(0,1))
breaks the degeneracy instead of just hoping the wafer data resolves it.

BONUS the wafer data can't give you: because CFD-ACE+ runs are virtual
experiments, they can cheaply span MULTIPLE setpoint temperatures even when
your physical wafer dataset (e.g. DS3-style, isothermal) can't -- which is
exactly the missing ingredient for separating a true reactor-level
temperature offset (dT_r) from a rate scale factor (eta_r), see
reactor_transfer.py's docstring on why dT_r is currently dropped. This
module computes dT_r(setpoint) as a diagnostic even though the current
NumPyro reactor-transfer models don't yet consume it (re-enabling it is a
documented future extension, not implemented here without real CFD data to
validate the extension against).
"""
from __future__ import annotations

import logging

import numpy as np

from chem_ml.cfd.io import CFDResult

log = logging.getLogger("chem_ml")

_MIN_STD = 0.05  # floor on the derived prior std so a single/near-identical
                 # CFD run never collapses to a Dirac-delta prior


def extract_transfer_priors(results: list[CFDResult]) -> dict:
    """From a list of CFDResult (one or more CFD-ACE+ runs, ideally spanning
    the operating envelope you plan to calibrate delta_r over), compute:
      ln_alpha_HCl, ln_alpha_GeH4 : (mean, std) -- informative priors to pass
          into reactor_transfer.reactor_transfer_model_gr_ge/ge_only in place
          of the hardcoded Normal(0, 1).
      dT_r_K : (mean, std) -- diagnostic only (see module docstring).

    alpha_i is computed per-run as the ratio of the CFD-predicted LOCAL
    (radially averaged) wafer-surface p_i/p_DCS to the INLET setpoint's
    p_i/p_DCS (from the run's flows_sccm, same ratio convention as
    schema.py's DS4 ingestion: p_i/p_DCS = flow_i/flow_DCS). Averaged in
    log space across all provided runs; the spread across runs/radii sets
    the prior's std (floored at _MIN_STD so a single run doesn't imply
    perfect certainty)."""
    if not results:
        raise ValueError("extract_transfer_priors needs at least one CFDResult")

    ln_alpha_hcl_samples: list[float] = []
    ln_alpha_geh4_samples: list[float] = []
    dT_samples: list[float] = []

    for res in results:
        flows = res.condition.flows_sccm
        inlet_hcl_ratio = flows["HCl"] / flows["DCS"]
        inlet_geh4_ratio = flows["GeH4"] / flows["DCS"]

        local_hcl_ratio = np.asarray(res.surface_p_i["HCl"])
        local_geh4_ratio = np.asarray(res.surface_p_i["GeH4"])

        ln_alpha_hcl_samples.extend(np.log(local_hcl_ratio / inlet_hcl_ratio).tolist())
        ln_alpha_geh4_samples.extend(np.log(local_geh4_ratio / inlet_geh4_ratio).tolist())
        dT_samples.extend((np.asarray(res.surface_T_K) - res.condition.T_set_K).tolist())

    def _mean_std(xs: list[float], floor: float = _MIN_STD) -> tuple[float, float]:
        arr = np.asarray(xs)
        return float(arr.mean()), float(max(arr.std(ddof=1) if len(arr) > 1 else floor, floor))

    priors = {
        "ln_alpha_HCl": _mean_std(ln_alpha_hcl_samples),
        "ln_alpha_GeH4": _mean_std(ln_alpha_geh4_samples),
        "dT_r_K_diagnostic_only": _mean_std(dT_samples, floor=1.0),
        "n_cfd_runs": len(results),
        "n_radial_points_total": len(ln_alpha_hcl_samples),
    }
    log.info("Extracted CFD-informed reactor-transfer priors from %d run(s): %s",
             len(results), priors)
    return priors
