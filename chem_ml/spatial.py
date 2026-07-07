"""
Phase 12: spatial wafer-scan data model + radially-resolved reactor-transfer.

WHY A SIBLING STRUCTURE, NOT NEW CanonicalRow FIELDS: a radial/contour scan
is one recipe -> N spatial measurement points, but schema.py's Dataset /
data_store.py's anti-contamination guarantee assumes one-row-one-independent
-observation. Pooling raw spatial points through today's add-data/add-reactor
unmodified would silently pseudo-replicate one wafer's systematic pattern as
if it were N independent chemistry confirmations. WaferScan.to_canonical_row()
is the deliberate seam that keeps the existing scalar machinery
(run_phase4_calibration, add-reactor, warm-start) fed exactly one point per
wafer, while the full spatial detail lives in this parallel path.

RADIALLY-RESOLVED REACTOR-TRANSFER: reactor_transfer.py's Phase 7 fits ONE
scalar delta_r = {alpha_HCl, alpha_GeH4, eta_GR, eta_Ge} per reactor, frozen
theta_chem. This module generalizes that to a delta_r(r): a LINEAR-IN-(r/R_w)
basis (intercept + slope) for each of those four quantities, fit directly
against one wafer's own radial/contour scan. This is the standalone
capability that makes a real spatial scan useful with zero CFD-ACE+
dependency, and it directly targets what METHODOLOGY.md sec 8 flags as
unresolved ("individual alpha values are not uniquely identified from wafer
data alone") by using the wafer's own radial shape, not just its scalar
average.

Basis is kept deliberately SMALL (2 coefficients per quantity): sec 8 already
documents a -0.97 posterior correlation between alpha and eta at the SCALAR
level. A higher-order radial basis would need many more spatial points than
a single contour scan realistically provides before it's identified by data
rather than by the prior.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from chem_ml.physics_core import ge_logmodel, gr_logmodel
from chem_ml.reactor_transfer import _DELTA_R_PRIOR_SD, _WEAK_ALPHA_PRIOR
from chem_ml.schema import CanonicalRow, ChemClass, Mode


@dataclass
class WaferRunMeta:
    """Per-run condition + instrumentation metadata for one physical wafer.
    Every point in that wafer's WaferScan shares this SAME nominal recipe --
    what varies spatially is the OUTCOME (via a fitted delta_r(r)), not the
    setpoint. nozzle_flows_sccm/probe_temps_K are per-run instrumentation
    (e.g. Stick_1..Stick_n nozzle flows, probe_1..probe_n pyrometer/probe
    temperatures) -- descriptive metadata about the condition, not part of
    the one-to-many spatial-outcome problem WaferPoint solves."""
    run_id: str
    reactor_id: str
    chem_class: ChemClass
    T_set_K: float
    p_DCS: float
    p_GeH4: float
    p_HCl: float
    p_B2H6: float = 0.0
    nozzle_flows_sccm: dict = field(default_factory=dict)
    probe_temps_K: dict = field(default_factory=dict)
    growth_time_s: Optional[float] = None
    source_dataset: str = ""

    def validate(self) -> None:
        assert self.T_set_K > 273.0, f"T_set_K={self.T_set_K} looks like Celsius; convert at ingest"
        assert self.p_DCS > 0.0, "p_DCS must be > 0 (normalization base)"


@dataclass
class WaferPoint:
    """One spatial measurement point on a wafer (one contour-scan location,
    or one point along an XRD radial line scan -- pass y_mm=0.0 for the
    latter, a pure-radial scan is just a subset of the general 2D case)."""
    run_id: str                       # FK into WaferRunMeta.run_id
    x_mm: Optional[float] = None
    y_mm: Optional[float] = None
    r_mm: Optional[float] = None      # derived from (x_mm, y_mm) if not given directly
    theta_deg: Optional[float] = None
    GR_nm_min_local: Optional[float] = None
    Ge_at_frac_local: Optional[float] = None
    thickness_A_local: Optional[float] = None
    measurement_source: str = ""      # e.g. "SE_contour" | "XRD_radial"

    def __post_init__(self) -> None:
        if self.r_mm is None:
            assert self.x_mm is not None and self.y_mm is not None, (
                "WaferPoint needs either r_mm directly, or both x_mm and y_mm"
            )
            self.r_mm = float(np.hypot(self.x_mm, self.y_mm))
        if self.theta_deg is None and self.x_mm is not None and self.y_mm is not None:
            self.theta_deg = float(np.degrees(np.arctan2(self.y_mm, self.x_mm)))

    def validate(self) -> None:
        assert self.r_mm >= 0.0, f"r_mm={self.r_mm} negative"
        assert any(v is not None for v in
                  (self.GR_nm_min_local, self.Ge_at_frac_local, self.thickness_A_local)), (
            "WaferPoint needs at least one of GR_nm_min_local/Ge_at_frac_local/thickness_A_local"
        )
        if self.Ge_at_frac_local is not None:
            assert 0.0 < self.Ge_at_frac_local < 1.0, "Ge fraction out of (0,1)"


@dataclass
class WaferScan:
    """One physical wafer's full spatial dataset: shared recipe metadata +
    N point measurements."""
    meta: WaferRunMeta
    points: list[WaferPoint]

    def validate(self) -> None:
        self.meta.validate()
        assert len(self.points) > 0, "WaferScan needs at least one point"
        for p in self.points:
            assert p.run_id == self.meta.run_id, (
                f"point run_id {p.run_id!r} != meta.run_id {self.meta.run_id!r}"
            )
            p.validate()

    def r_array(self) -> np.ndarray:
        return np.array([p.r_mm for p in self.points])

    def effective_GR_nm_min(self, point: WaferPoint) -> Optional[float]:
        """GR_nm_min_local if measured directly; else derived from
        thickness_A_local / meta.growth_time_s (Zhang et al. 2026 sec 3.4's
        own convention: growth rate = thickness / growth time, for
        approximately-linear-in-time growth) when only thickness was
        measured (e.g. an XRD-only line scan with no direct GR channel)."""
        if point.GR_nm_min_local is not None:
            return point.GR_nm_min_local
        if point.thickness_A_local is not None and self.meta.growth_time_s:
            # thickness_A / 10 -> nm; growth_time_s / 60 -> min
            return point.thickness_A_local * 6.0 / self.meta.growth_time_s
        return None

    def to_canonical_row(self, reduction: str = "center") -> CanonicalRow:
        """Collapse to ONE scalar CanonicalRow for the EXISTING scalar
        chemistry fit (Phase 4 / add-reactor) -- keeps anti-contamination
        and add-reactor working completely unchanged; the full spatial
        detail is only ever consumed by run_phase12_spatial_transfer.

        "center": the point nearest r=0 becomes the scalar GR/Ge.
        "area_weighted_mean": wafer_average() over all points carrying that
          field."""
        if reduction == "center":
            pt = min(self.points, key=lambda p: p.r_mm)
            gr = self.effective_GR_nm_min(pt)
            ge = pt.Ge_at_frac_local
        elif reduction == "area_weighted_mean":
            r_all = self.r_array()
            gr_vals = np.array([self.effective_GR_nm_min(p) for p in self.points])
            ge_vals = np.array([p.Ge_at_frac_local for p in self.points])
            gr_mask = gr_vals != None  # noqa: E711 (elementwise None-check on object array)
            ge_mask = ge_vals != None  # noqa: E711
            gr = wafer_average(r_all[gr_mask], gr_vals[gr_mask].astype(float)) if gr_mask.any() else None
            ge = wafer_average(r_all[ge_mask], ge_vals[ge_mask].astype(float)) if ge_mask.any() else None
        else:
            raise ValueError(f"unknown reduction {reduction!r}")

        return CanonicalRow(
            reactor_id=self.meta.reactor_id, chem_class=self.meta.chem_class, mode=Mode.BLANKET,
            T_K=self.meta.T_set_K, p_DCS=self.meta.p_DCS, p_GeH4=self.meta.p_GeH4,
            p_HCl=self.meta.p_HCl, p_B2H6=self.meta.p_B2H6, growth_time_s=self.meta.growth_time_s,
            GR_nm_min=gr, Ge_at_frac=ge, source_dataset=self.meta.source_dataset,
        )


# ---------------------------------------------------------------------------
# Radial-profile utilities (shared by real wafer scans here AND, later, by
# CFD-simulated CFDResult profiles -- both are r_mm-indexed, so the same
# wafer_average/wiwnu apply to either).
# ---------------------------------------------------------------------------
def radial_profile(r_mm: np.ndarray, values: np.ndarray, decimals: int = 0
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Collapse azimuthally-repeated points at (numerically) the same radius
    into one averaged value per radius -- Jackel et al. (2024)'s radial-
    averaging strategy (their Fig. 2) for turning a full 2D wafer scan into
    a single GR(r)/Ge(r) curve before any area integral is taken. `decimals`
    controls how close two points' radii must be to count as "the same
    ring" (default: nearest whole mm)."""
    r_mm = np.asarray(r_mm, dtype=float)
    values = np.asarray(values, dtype=float)
    r_round = np.round(r_mm, decimals)
    uniq = np.unique(r_round)
    v_avg = np.array([values[r_round == u].mean() for u in uniq])
    order = np.argsort(uniq)
    return uniq[order], v_avg[order]


def wafer_average(r_mm: np.ndarray, values: np.ndarray) -> float:
    """Area-weighted mean over the wafer disk: (2/R_w^2) * integral r*f(r) dr,
    via trapz on the radially-averaged profile (radial_profile above). R_w is
    taken as the outermost MEASURED radius, not a separately-configured
    wafer size, so this works directly off whatever scan extent was
    actually sampled."""
    r_p, v_p = radial_profile(r_mm, values)
    R_w = r_p[-1]
    if R_w <= 0:
        return float(v_p[0])
    return float(2.0 * np.trapezoid(r_p * v_p, r_p) / (R_w ** 2))


def wiwnu(r_mm: np.ndarray, values: np.ndarray) -> float:
    """Within-wafer non-uniformity, Zhang et al. (2026)'s definition:
    (max-min)/2 / mean, computed over the radially-averaged profile (not raw
    scattered points, so repeated azimuthal sampling at one ring doesn't
    over-weight that ring)."""
    _, v_p = radial_profile(r_mm, values)
    return float((v_p.max() - v_p.min()) / 2.0 / v_p.mean())


def build_spatial_features(scan: WaferScan, invT_scaler: tuple[float, float]) -> jnp.ndarray:
    """Build the [invT_std, ln_HCl, ln_GeH4, ln_B2H6] feature row for this
    scan's single shared recipe condition (features.build_features's exact
    transform and normalization convention), TILED once per spatial point --
    every point on one wafer was grown under the same nominal setpoint; what
    varies with position is the OUTCOME (via the fitted delta_r(r) below),
    not the input condition."""
    m = scan.meta
    mu, sd = invT_scaler
    invT_s = (1.0 / m.T_set_K - mu) / sd
    ln_HCl = float(np.log(m.p_HCl / m.p_DCS))
    ln_GeH4 = float(np.log(m.p_GeH4 / m.p_DCS))
    ln_B2H6 = float(np.log(m.p_B2H6 / m.p_DCS)) if m.p_B2H6 > 0 else 0.0
    row = jnp.array([invT_s, ln_HCl, ln_GeH4, ln_B2H6])
    return jnp.tile(row, (len(scan.points), 1))


def apply_chemistry_pointwise(theta_gr: dict, theta_ge: dict, X_at_r: jnp.ndarray,
                              ln_alpha_HCl_r: jnp.ndarray, ln_alpha_GeH4_r: jnp.ndarray,
                              ln_eta_GR_r: jnp.ndarray, ln_eta_Ge_r: jnp.ndarray
                              ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Reapply the FROZEN chemistry core across many spatial rows at once,
    shifting each row's HCl/GeH4 log-features by that row's OWN alpha(r)
    (rather than one scalar alpha for the whole reactor, as
    reactor_transfer.py's Phase-7 fit does) -- same column-shift convention
    as reactor_transfer.py:65, just per-row. Returns (gr_log, ge_log)."""
    X_eff = X_at_r.at[:, 1].add(ln_alpha_HCl_r).at[:, 2].add(ln_alpha_GeH4_r)
    gr_log = ln_eta_GR_r + gr_logmodel(theta_gr, X_eff)
    ge_log = ln_eta_Ge_r + ge_logmodel(theta_ge, X_eff)
    return gr_log, ge_log


# ---------------------------------------------------------------------------
# NumPyro models: radially-resolved generalization of
# reactor_transfer.py's reactor_transfer_model_gr_ge / _ge_only. Same
# frozen-theta_chem structure, same Normal-in-log-space likelihood; the
# single scalar ln_alpha_HCl/ln_alpha_GeH4/ln_eta_GR/ln_eta_Ge become
# linear-in-(r/R_w) functions (intercept `a0_*` + slope `a1_*`).
# ---------------------------------------------------------------------------
def reactor_transfer_model_spatial_gr_ge(r_over_Rw: jnp.ndarray, X: jnp.ndarray,
                                         y_gr_log: jnp.ndarray, y_ge_log: jnp.ndarray,
                                         theta_gr: dict, theta_ge: dict,
                                         alpha_priors: dict | None = None):
    """Full-scan case: both GR and Ge/Si channels measured at every point."""
    ap = alpha_priors or _WEAK_ALPHA_PRIOR
    a0_HCl = numpyro.sample("a0_alpha_HCl", dist.Normal(*ap["ln_alpha_HCl"]))
    a1_HCl = numpyro.sample("a1_alpha_HCl", dist.Normal(0.0, _DELTA_R_PRIOR_SD))
    a0_GeH4 = numpyro.sample("a0_alpha_GeH4", dist.Normal(*ap["ln_alpha_GeH4"]))
    a1_GeH4 = numpyro.sample("a1_alpha_GeH4", dist.Normal(0.0, _DELTA_R_PRIOR_SD))
    a0_eta_GR = numpyro.sample("a0_eta_GR", dist.Normal(0.0, _DELTA_R_PRIOR_SD))
    a1_eta_GR = numpyro.sample("a1_eta_GR", dist.Normal(0.0, _DELTA_R_PRIOR_SD))
    a0_eta_Ge = numpyro.sample("a0_eta_Ge", dist.Normal(0.0, _DELTA_R_PRIOR_SD))
    a1_eta_Ge = numpyro.sample("a1_eta_Ge", dist.Normal(0.0, _DELTA_R_PRIOR_SD))
    sigma_gr = numpyro.sample("sigma_GR_r", dist.HalfNormal(0.5))
    sigma_ge = numpyro.sample("sigma_Ge_r", dist.HalfNormal(0.5))

    ln_alpha_HCl_r = a0_HCl + a1_HCl * r_over_Rw
    ln_alpha_GeH4_r = a0_GeH4 + a1_GeH4 * r_over_Rw
    ln_eta_GR_r = a0_eta_GR + a1_eta_GR * r_over_Rw
    ln_eta_Ge_r = a0_eta_Ge + a1_eta_Ge * r_over_Rw

    mu_gr, mu_ge = apply_chemistry_pointwise(
        theta_gr, theta_ge, X, ln_alpha_HCl_r, ln_alpha_GeH4_r, ln_eta_GR_r, ln_eta_Ge_r,
    )
    numpyro.sample("obs_GR_r", dist.Normal(mu_gr, sigma_gr), obs=y_gr_log)
    numpyro.sample("obs_Ge_r", dist.Normal(mu_ge, sigma_ge), obs=y_ge_log)


def reactor_transfer_model_spatial_ge_only(r_over_Rw: jnp.ndarray, X: jnp.ndarray,
                                           y_ge_log: jnp.ndarray, theta_ge: dict,
                                           alpha_priors: dict | None = None):
    """Ge/Si-only case: e.g. an XRD-only radial scan with no direct GR
    channel and no growth_time_s to derive one from thickness."""
    ap = alpha_priors or _WEAK_ALPHA_PRIOR
    a0_HCl = numpyro.sample("a0_alpha_HCl", dist.Normal(*ap["ln_alpha_HCl"]))
    a1_HCl = numpyro.sample("a1_alpha_HCl", dist.Normal(0.0, _DELTA_R_PRIOR_SD))
    a0_GeH4 = numpyro.sample("a0_alpha_GeH4", dist.Normal(*ap["ln_alpha_GeH4"]))
    a1_GeH4 = numpyro.sample("a1_alpha_GeH4", dist.Normal(0.0, _DELTA_R_PRIOR_SD))
    a0_eta_Ge = numpyro.sample("a0_eta_Ge", dist.Normal(0.0, _DELTA_R_PRIOR_SD))
    a1_eta_Ge = numpyro.sample("a1_eta_Ge", dist.Normal(0.0, _DELTA_R_PRIOR_SD))
    sigma_ge = numpyro.sample("sigma_Ge_r", dist.HalfNormal(0.5))

    ln_alpha_HCl_r = a0_HCl + a1_HCl * r_over_Rw
    ln_alpha_GeH4_r = a0_GeH4 + a1_GeH4 * r_over_Rw
    ln_eta_Ge_r = a0_eta_Ge + a1_eta_Ge * r_over_Rw

    X_eff = X.at[:, 1].add(ln_alpha_HCl_r).at[:, 2].add(ln_alpha_GeH4_r)
    mu_ge = ln_eta_Ge_r + ge_logmodel(theta_ge, X_eff)
    numpyro.sample("obs_Ge_r", dist.Normal(mu_ge, sigma_ge), obs=y_ge_log)
