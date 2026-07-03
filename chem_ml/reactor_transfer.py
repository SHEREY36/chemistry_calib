"""
Hierarchical reactor-transfer random effect (Layer B, data-only version):
delta_r = {dT_r, alpha_{i,r}, eta_r}. theta_chem SHARED and frozen from the
Phase 4 DS1 fit; delta_r per-reactor, partially pooled. Implemented in
Phase 7 -- see build_steps_and_cfd_integration.md.
"""
from __future__ import annotations

from chem_ml.config import PriorConfig


def reactor_transfer_model(X_by_reactor: dict, y_by_reactor: dict, pri: PriorConfig):
    """
    Phase 7 TODO:
      - Global chemistry params theta_chem sampled once (shared) -- or frozen
        at Phase 4's posterior mean, per the cross-reactor validation design.
      - Hyperpriors: sigma_dT ~ HalfNormal; sigma_alpha ~ HalfNormal.
      - For each reactor r: dT_r ~ Normal(0, sigma_dT); ln_alpha_{i,r} ~ Normal(0, sigma_alpha);
        ln_eta_r ~ Normal(0, s).
      - Transfer map: invT_eff = 1/(T + dT_r); ln_ratio_eff = ln_ratio + ln_alpha_{i,r};
        GR_eff = eta_r * GR_chem. Feed EFFECTIVE features into gr_logmodel etc.
      - Likelihood per reactor's rows.
    CROSS-REACTOR VALIDATION (Phase 7.2): fit theta_chem on DS1 only; FREEZE it;
    fit ONLY delta_r for DS3/DS4; check GR/Ge recovered within published R^2 band.
    """
    raise NotImplementedError("Phase 7: implement hierarchical reactor transfer model")
