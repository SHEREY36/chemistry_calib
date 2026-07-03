"""
chemistry_ml_framework.py
==========================

SINGLE-FILE SCAFFOLD for the Physics-ML Chemistry Calibration model.

PURPOSE
-------
This file is the blueprint Claude Code will refactor into the modular package
described in `build_steps_and_cfd_integration.md`. Each `# ===== MODULE: xxx =====`
banner marks an intended file boundary. Fully-implemented sections (schema,
registry, assembler, physics core, Bayesian calibration, identifiability,
inverse-design skeleton) are correct and runnable. Sections that are large
engineering efforts (residual NN, CFD-ACE+ I/O, active learning) are precise,
spec'd STUBS with `TODO(claude-code)` markers describing exactly what to build.

DESIGN INVARIANTS (do not violate when refactoring)
---------------------------------------------------
1. TWO LAYERS STAY SEPARATE:
     Layer A = intrinsic chemistry (theta_chem)  -> reactor-independent
     Layer B = reactor transport (delta_r)       -> reactor-specific (CFD or data)
   Observable = A(B(setpoint, geometry)). Never merge them.
2. ABSENT SPECIES -> ZERO DEPENDENCE, STRUCTURALLY. The reaction-network
   assembler instantiates only present species. There is no learned mask that
   could leak an absent precursor into a prediction.
3. DISCONTINUOUS AXES (class, dopant, precursor-family, mode) ARE HARD-GATED by
   the DECLARED recipe (task identity known). CONTINUOUS AXES (reactor, drift,
   species-within-family) ARE PARTIALLY POOLED (hierarchical Bayes).
4. PARAMETRIZE THE TEMPERATURE TERM AS THE COEFFICIENT OF 1/T (kappa), NOT as
   exp(-Ea/RT). This sidesteps Tomasini's sign-convention ambiguity. Expect
   kappa_GR ~ -24507 K (GR rises with T) and kappa_Ge ~ -4319 K (Ge falls with T).
5. float64 EVERYWHERE. Kinetics fits are ill-conditioned in float32.

Author: (scaffold) | Target: Claude Code refactor
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Sequence

import numpy as np

# --- JAX / NumPyro (Bayesian core) ------------------------------------------
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)  # INVARIANT 5

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("chem_ml")

R_GAS = 8.314462618  # J/mol/K  (used only if converting kappa <-> Ea; kappa is primary)


# ============================================================================
# ===== MODULE: config =======================================================
# ============================================================================
@dataclass(frozen=True)
class MCMCConfig:
    num_warmup: int = 1500
    num_samples: int = 2000
    num_chains: int = 4
    target_accept: float = 0.9
    seed: int = 0


@dataclass(frozen=True)
class PriorConfig:
    """Literature-informed priors (means, sds) in the model's native parametrization.
    kappa is the coefficient of 1/T (units K). Orders are dimensionless."""
    # GR model
    lnK_GR: tuple[float, float] = (0.0, 10.0)          # weak; K absorbs scaling
    kappa_GR: tuple[float, float] = (-24507.0, 5000.0)  # GR RISES with T -> negative
    gamma_HCl: tuple[float, float] = (-0.7, 0.3)
    gamma_GeH4: tuple[float, float] = (1.3, 0.3)
    # Ge/Si model
    lnK_Ge: tuple[float, float] = (0.0, 10.0)
    kappa_Ge: tuple[float, float] = (-4319.0, 3000.0)   # Ge FALLS with T
    dgamma_HCl: tuple[float, float] = (0.1, 0.2)
    dgamma_GeH4: tuple[float, float] = (0.51, 0.2)
    # Boron model (DS2)
    lnK_B: tuple[float, float] = (0.0, 10.0)
    beta_HCl: tuple[float, float] = (-0.5, 0.3)
    beta_GeH4: tuple[float, float] = (-0.3, 0.3)
    beta_B2H6: tuple[float, float] = (0.8, 0.2)
    # noise
    sigma_halfnormal: float = 0.5  # log-space observation noise scale


@dataclass(frozen=True)
class Config:
    data_raw: str = "data/raw"
    data_processed: str = "data/processed"
    mcmc: MCMCConfig = field(default_factory=MCMCConfig)
    priors: PriorConfig = field(default_factory=PriorConfig)
    # acceptance tolerances (Phase 4.3)
    r2_target: float = 0.98
    kappa_GR_tol_frac: float = 0.10
    # inverse design
    inverse_uq_lambda: float = 1.0
    # active learning
    cfd_run_budget: int = 25
    al_batch_size: int = 3


# ============================================================================
# ===== MODULE: schema =======================================================
# ============================================================================
class Mode(str, Enum):
    BLANKET = "blanket"
    SELECTIVE = "selective"


class ChemClass(str, Enum):
    SI = "Si"
    SIGE = "SiGe"
    SIGE_B = "SiGe:B"
    SIGE_P = "SiGe:P"
    SIC = "SiC"
    SIGEC = "SiGeC"


@dataclass
class CanonicalRow:
    """One physical condition (one wafer / one calibration point).
    Partial pressures in Torr; T in KELVIN (convert from C at ingest!)."""
    reactor_id: str
    chem_class: ChemClass
    mode: Mode
    T_K: float
    p_DCS: float
    p_GeH4: float
    p_HCl: float
    p_B2H6: float = 0.0
    p_carrier: float = 0.0
    pattern_density: float = 0.0          # 0 for blanket
    growth_time_s: Optional[float] = None
    GR_nm_min: Optional[float] = None
    Ge_at_frac: Optional[float] = None    # 0..1
    B_conc: Optional[float] = None        # at/cm^3 or [B]/[Si]; document choice
    source_dataset: str = ""

    def validate(self) -> None:
        """Fail-loud validation (Phase 1.3). Raise on physical impossibility."""
        assert self.T_K > 273.0, f"T_K={self.T_K} looks like Celsius; convert at ingest"
        for name in ("p_DCS", "p_GeH4", "p_HCl", "p_B2H6"):
            v = getattr(self, name)
            assert v >= 0.0, f"{name} negative ({v})"
        assert self.p_DCS > 0.0, "p_DCS must be > 0 (normalization base)"
        if self.GR_nm_min is not None:
            assert self.GR_nm_min > 0.0, "GR must be > 0"
        if self.Ge_at_frac is not None:
            assert 0.0 < self.Ge_at_frac < 1.0, "Ge fraction out of (0,1)"


@dataclass
class Dataset:
    rows: list[CanonicalRow]

    def validate(self) -> None:
        for r in self.rows:
            r.validate()

    def filter(self, **kw) -> "Dataset":
        def ok(r: CanonicalRow) -> bool:
            return all(getattr(r, k) == v for k, v in kw.items())
        return Dataset([r for r in self.rows if ok(r)])

    def __len__(self) -> int:
        return len(self.rows)


# TODO(claude-code): implement `ingest_tomasini(appendix_paths) -> Dataset`
#   - parse DS1 (App I i-SiGe, 70 rows), DS2 (App I SiGe:B), DS3 (App II Hartmann),
#     DS4 (App III Tan). Reconstruct p_i from ratios * p_DCS. CONVERT T_C->T_K.
#   - tag reactor_id: DS1/DS2="ASM_Epsilon", DS3="Hartmann", DS4="Tan".
#   - run Dataset.validate(); persist parquet to Config.data_processed.
# TODO(claude-code): implement `ingest_amat(export_paths) -> Dataset`
#   - map ACE+/PX/export columns -> canonical schema; CLEAN negative/near-zero
#     flow columns (interpret as off/offset/noise) BEFORE forming partial pressures;
#     p_i = P_tot * flow_i / sum(flows); GR = XRD_thickness / growth_time.


# ============================================================================
# ===== MODULE: registry =====================================================
# ============================================================================
class Role(str, Enum):
    SI_SOURCE = "Si-source"
    GE_SOURCE = "Ge-source"
    C_SOURCE = "C-source"
    DOPANT = "dopant"
    SELECTIVITY = "selectivity-agent"
    CARRIER = "carrier"
    BYPRODUCT = "byproduct"


@dataclass(frozen=True)
class Species:
    canonical_name: str
    formula: str
    role: Role
    family: str                # 'hydride' | 'chlorinated' | 'germane' | 'dopant' | ...
    n_Si: int = 0
    n_Ge: int = 0
    n_C: int = 0
    n_Cl: int = 0
    n_H: int = 0
    produces_HCl: bool = False
    default_prior: Optional[dict] = None  # prior on delivery/decomp params for new species


class SpeciesRegistry:
    """Single source of truth for every precursor/dopant/carrier the model can use.
    Adding a species = adding an entry here (Phase 2.1). No code change elsewhere."""

    def __init__(self) -> None:
        self._db: dict[str, Species] = {}
        self._seed_defaults()

    def _seed_defaults(self) -> None:
        for sp in [
            Species("dichlorosilane", "SiH2Cl2", Role.SI_SOURCE, "chlorinated",
                    n_Si=1, n_Cl=2, n_H=2, produces_HCl=True),
            Species("silane", "SiH4", Role.SI_SOURCE, "hydride", n_Si=1, n_H=4),
            Species("disilane", "Si2H6", Role.SI_SOURCE, "hydride", n_Si=2, n_H=6),
            Species("trisilane", "Si3H8", Role.SI_SOURCE, "hydride", n_Si=3, n_H=8),
            Species("trichlorosilane", "SiHCl3", Role.SI_SOURCE, "chlorinated",
                    n_Si=1, n_Cl=3, n_H=1, produces_HCl=True),
            Species("germane", "GeH4", Role.GE_SOURCE, "germane", n_Ge=1, n_H=4),
            Species("hcl", "HCl", Role.SELECTIVITY, "chlorinated", n_Cl=1, n_H=1,
                    produces_HCl=True),
            Species("diborane", "B2H6", Role.DOPANT, "dopant", n_H=6),
            Species("phosphine", "PH3", Role.DOPANT, "dopant", n_H=3),
            Species("hydrogen", "H2", Role.CARRIER, "carrier", n_H=2),
        ]:
            self._db[sp.canonical_name] = sp

    def get(self, name: str) -> Species:
        if name not in self._db:
            raise KeyError(f"Unknown species '{name}'. Add it to the registry (Phase 2.1).")
        return self._db[name]

    def add(self, sp: Species) -> None:
        if sp.canonical_name in self._db:
            raise ValueError(f"Species {sp.canonical_name} already registered")
        self._db[sp.canonical_name] = sp
        log.info("Registered new species: %s (%s, family=%s)", sp.canonical_name, sp.formula, sp.family)

    def by_role(self, role: Role) -> list[Species]:
        return [s for s in self._db.values() if s.role == role]


# ============================================================================
# ===== MODULE: assembler ====================================================
# ============================================================================
@dataclass
class ActiveNetwork:
    """The set of active model terms for a declared recipe. Absent species are
    simply not here -> zero dependence by construction (INVARIANT 2)."""
    chem_class: ChemClass
    si_source: Species
    ge_source: Optional[Species]
    dopant: Optional[Species]
    selectivity_agent: Optional[Species]
    has_chlorine: bool
    mode: Mode

    @property
    def uses_GR_model(self) -> bool:
        return True

    @property
    def uses_Ge_model(self) -> bool:
        return self.ge_source is not None

    @property
    def uses_B_model(self) -> bool:
        return self.dopant is not None and self.dopant.formula == "B2H6"


class ReactionNetworkAssembler:
    """Given a declared class + species list, returns the ActiveNetwork.
    This is the anti-contamination guarantee (Phase 2.2/2.3)."""

    def __init__(self, registry: SpeciesRegistry) -> None:
        self.reg = registry

    def assemble(self, chem_class: ChemClass, species_names: Sequence[str], mode: Mode) -> ActiveNetwork:
        species = [self.reg.get(n) for n in species_names]
        si = _first(species, Role.SI_SOURCE)
        ge = _first(species, Role.GE_SOURCE, required=False)
        dop = _first(species, Role.DOPANT, required=False)
        sel = _first(species, Role.SELECTIVITY, required=False)
        if si is None:
            raise ValueError("Every recipe needs a Si source.")
        has_cl = any(s.produces_HCl for s in species)
        return ActiveNetwork(chem_class, si, ge, dop, sel, has_cl, mode)


def _first(species: Sequence[Species], role: Role, required: bool = True) -> Optional[Species]:
    for s in species:
        if s.role == role:
            return s
    if required:
        return None
    return None


# ============================================================================
# ===== MODULE: features =====================================================
# ============================================================================
@dataclass
class FeatureBundle:
    """Design matrix in log space + bookkeeping to invert scaling."""
    X: jnp.ndarray            # (N, D) features
    col_names: list[str]
    invT_scaler: tuple[float, float]  # (mean, std) used to standardize 1/T


def build_features(ds: Dataset, standardize_invT: bool = True) -> FeatureBundle:
    """Columns: [1/T, ln(pHCl/pDCS), ln(pGeH4/pDCS), ln(pB2H6/pDCS)].
    Intercept handled inside the NumPyro model (lnK)."""
    rows = ds.rows
    invT = np.array([1.0 / r.T_K for r in rows])
    ln_HCl = np.array([np.log(r.p_HCl / r.p_DCS) for r in rows])
    ln_GeH4 = np.array([np.log(r.p_GeH4 / r.p_DCS) for r in rows])
    # guard log(0) for B2H6 absent -> use -inf-safe: absent B just won't feed B-model
    ln_B2H6 = np.array([np.log(r.p_B2H6 / r.p_DCS) if r.p_B2H6 > 0 else 0.0 for r in rows])

    if standardize_invT:
        mu, sd = float(invT.mean()), float(invT.std() + 1e-12)
        invT_s = (invT - mu) / sd
    else:
        mu, sd = 0.0, 1.0
        invT_s = invT

    X = jnp.asarray(np.stack([invT_s, ln_HCl, ln_GeH4, ln_B2H6], axis=1))
    return FeatureBundle(X, ["invT", "ln_HCl", "ln_GeH4", "ln_B2H6"], (mu, sd))


# ============================================================================
# ===== MODULE: physics_core =================================================
# ============================================================================
# Pure, differentiable JAX forward maps. INVARIANT 4: kappa is coeff of 1/T.
# NOTE: features pass STANDARDIZED invT; the model learns kappa_std then we
# de-standardize for reporting (kappa_true = kappa_std / sd).

def gr_logmodel(params: dict, X: jnp.ndarray) -> jnp.ndarray:
    """ln(GR). params: lnK_GR, kappa_GR (std units), gamma_HCl, gamma_GeH4."""
    invT, ln_HCl, ln_GeH4 = X[:, 0], X[:, 1], X[:, 2]
    return (params["lnK_GR"]
            + params["kappa_GR"] * invT
            + params["gamma_HCl"] * ln_HCl
            + params["gamma_GeH4"] * ln_GeH4)


def ge_logmodel(params: dict, X: jnp.ndarray) -> jnp.ndarray:
    """ln(x/(1-x)). params: lnK_Ge, kappa_Ge, dgamma_HCl, dgamma_GeH4."""
    invT, ln_HCl, ln_GeH4 = X[:, 0], X[:, 1], X[:, 2]
    return (params["lnK_Ge"]
            + params["kappa_Ge"] * invT
            + params["dgamma_HCl"] * ln_HCl
            + params["dgamma_GeH4"] * ln_GeH4)


def b_logmodel(params: dict, X: jnp.ndarray) -> jnp.ndarray:
    """ln([B]/[Si]). params: lnK_B, beta_HCl, beta_GeH4, beta_B2H6."""
    ln_HCl, ln_GeH4, ln_B2H6 = X[:, 1], X[:, 2], X[:, 3]
    return (params["lnK_B"]
            + params["beta_HCl"] * ln_HCl
            + params["beta_GeH4"] * ln_GeH4
            + params["beta_B2H6"] * ln_B2H6)


def destandardize_kappa(kappa_std: float, invT_scaler: tuple[float, float]) -> float:
    """Convert learned kappa (on standardized 1/T) back to physical K units.
    Report this against Tomasini's tabulated Ea/R (expect ~ -24507 for GR)."""
    _, sd = invT_scaler
    return kappa_std / sd


# ============================================================================
# ===== MODULE: calibration ==================================================
# ============================================================================
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
    site = [k for k in pred(rng, X=X, y_log=None, pri=cfg.priors) if k.startswith("obs_")][0]
    return pred(rng, X=X, y_log=None, pri=cfg.priors)[site]


# ---- acceptance gates (Phase 4.3) ------------------------------------------
def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / (ss_tot + 1e-30)


def check_tomasini_acceptance(mcmc_gr: MCMC, mcmc_ge: MCMC,
                              y_gr_true: np.ndarray, y_ge_true: np.ndarray,
                              gr_pred_log: np.ndarray, ge_pred_log: np.ndarray,
                              invT_scaler: tuple[float, float], cfg: Config) -> dict:
    """Return a report dict of the objective acceptance metrics.
    TODO(claude-code): wire predictions in real units (exponentiate log preds)."""
    kappa_gr = destandardize_kappa(float(np.mean(mcmc_gr.get_samples()["kappa_GR"])), invT_scaler)
    report = {
        "R2_GR_logspace": r2_score(np.log(y_gr_true), gr_pred_log.mean(0)),
        "R2_Ge_logspace": r2_score(np.log(y_ge_true / (1 - y_ge_true)), ge_pred_log.mean(0)),
        "kappa_GR_K": kappa_gr,
        "kappa_GR_within_10pct": abs(abs(kappa_gr) - 24507.0) / 24507.0 <= cfg.kappa_GR_tol_frac,
        "gamma_HCl": float(np.mean(mcmc_gr.get_samples()["gamma_HCl"])),
        "gamma_GeH4": float(np.mean(mcmc_gr.get_samples()["gamma_GeH4"])),
    }
    report["PASS"] = (report["R2_GR_logspace"] >= cfg.r2_target
                      and report["R2_Ge_logspace"] >= cfg.r2_target
                      and report["kappa_GR_within_10pct"])
    return report


# ============================================================================
# ===== MODULE: residual_nn  (STUB) ==========================================
# ============================================================================
# TODO(claude-code): implement g_NN(x; phi) as a SMALL, strongly-regularized MLP.
#   - framework: flax.linen or equinox. Arch: input(features + raw ratios) ->
#     Dense(16) -> tanh -> Dense(16) -> tanh -> Dense(n_obs). Output = log-residual.
#   - HARD GATE by declared ChemClass: a dict of per-class NN params; only the
#     declared class's net is evaluated (INVARIANT 3). Never share across classes.
#   - REGULARIZATION: optax.adamw with weight_decay; plus a Normal(0, small) prior
#     if fit within NumPyro, so g_NN shrinks to ~0 where the power law suffices.
#   - Hybrid forward: y_log = f_phys(theta, X) + g_NN(phi, X_full).
#   - VALIDATION: on DS1, confirm ||g_NN|| small and it only lifts the Regime-I
#     low-pGeH4/pDCS tail that the power law misses.
class ResidualNN:  # placeholder interface
    def __init__(self, chem_class: ChemClass, n_out: int = 3):
        self.chem_class = chem_class
        self.n_out = n_out
        self.params = None  # set by fit

    def __call__(self, X_full: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError("TODO(claude-code): implement gated residual MLP")

    def fit(self, X_full, residual_targets, l2: float = 1e-2):
        raise NotImplementedError


# ============================================================================
# ===== MODULE: identifiability ==============================================
# ============================================================================
def parameter_covariance(mcmc: MCMC, param_names: Sequence[str]) -> np.ndarray:
    """Posterior covariance of the named params (Phase 6.1)."""
    s = mcmc.get_samples()
    M = np.stack([np.asarray(s[n]).reshape(-1) for n in param_names], axis=1)
    return np.cov(M, rowvar=False)


def eigenspectrum(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stiff (small variance dir) vs sloppy (large variance dir). Returns
    (eigenvalues ascending, eigenvectors). Rank params by data-constrained-ness."""
    w, V = np.linalg.eigh(cov)
    return w, V


def fisher_information(logmodel: Callable, params: dict, X: jnp.ndarray,
                       sigma: float, param_order: Sequence[str]) -> np.ndarray:
    """I = J^T Sigma^-1 J with J = d(logmodel)/d(theta) via JAX jacfwd (Phase 6.2)."""
    def f(theta_vec):
        p = dict(zip(param_order, theta_vec))
        # merge with any fixed params if needed
        merged = {**params, **p}
        return logmodel(merged, X)
    theta0 = jnp.array([params[k] for k in param_order])
    J = jax.jacfwd(f)(theta0)                       # (N, P)
    return np.asarray(J.T @ J) / (sigma ** 2)


# ============================================================================
# ===== MODULE: reactor_transfer (Layer B, data-only) ========================
# ============================================================================
# Hierarchical random effect delta_r = {dT_r, alpha_{i,r}, eta_r}.
# theta_chem SHARED; delta_r per-reactor, partially pooled. (Phase 7)
def reactor_transfer_model(X_by_reactor: dict, y_by_reactor: dict, pri: PriorConfig):
    """
    TODO(claude-code): full hierarchical NumPyro model.
      - Global chemistry params theta_chem sampled once (shared).
      - Hyperpriors: sigma_dT ~ HalfNormal; sigma_alpha ~ HalfNormal.
      - For each reactor r: dT_r ~ Normal(0, sigma_dT); ln_alpha_{i,r} ~ Normal(0, sigma_alpha);
        ln_eta_r ~ Normal(0, s).
      - Transfer map: invT_eff = 1/(T + dT_r); ln_ratio_eff = ln_ratio + ln_alpha_{i,r};
        GR_eff = eta_r * GR_chem. Feed EFFECTIVE features into gr_logmodel etc.
      - Likelihood per reactor's rows.
    CROSS-REACTOR VALIDATION (Phase 7.2): fit theta_chem on DS1 only; FREEZE it;
    fit ONLY delta_r for DS3/DS4; check GR/Ge recovered within published R^2 band.
    """
    raise NotImplementedError("TODO(claude-code): hierarchical reactor transfer model")


# ============================================================================
# ===== MODULE: inverse_design ===============================================
# ============================================================================
def inverse_design(forward_log: Callable, theta_bar: dict, y_target_log: jnp.ndarray,
                   x0: jnp.ndarray, feasible_box: tuple[jnp.ndarray, jnp.ndarray],
                   uq_fn: Optional[Callable] = None, lam: float = 1.0,
                   steps: int = 500, lr: float = 1e-2) -> jnp.ndarray:
    """x* = argmin_x || forward_log(theta_bar, x) - y_target_log ||^2 + lam * U(x)
    subject to x in feasible_box. Differentiable; projected gradient descent.
    x here is the FEATURE-SPACE recipe (invT, ln-ratios); map back to flows after.
    (Phase 8.1)"""
    import optax
    lo, hi = feasible_box
    opt = optax.adam(lr)
    x = x0
    st = opt.init(x)

    def loss(x):
        pred = forward_log(theta_bar, x[None, :])[0]
        data_term = jnp.sum((pred - y_target_log) ** 2)
        uq_term = uq_fn(x) if uq_fn is not None else 0.0
        return data_term + lam * uq_term

    gval = jax.grad(loss)
    for _ in range(steps):
        g = gval(x)
        upd, st = opt.update(g, st)
        x = optax.apply_updates(x, upd)
        x = jnp.clip(x, lo, hi)   # projection onto feasible box
    return x
# TODO(claude-code): add a posterior-robust variant (optimize expected loss over
#   theta posterior samples), and a recipe<->feature mapping utility (flows <-> ratios).


# ============================================================================
# ===== MODULE: cfd (Layer B via CFD-ACE+)  (STUB) ===========================
# ============================================================================
# CFD-ACE+ computes the setpoint->surface map for AMAT's 3D reactor and consumes
# the calibrated mechanism. It does NOT reproduce Tomasini. (Phase 9)
@dataclass
class CFDCondition:
    T_set_K: float
    flows_sccm: dict           # {'DCS':.., 'GeH4':.., 'HCl':.., 'B2H6':..}
    P_tot_torr: float
    geometry_id: str


@dataclass
class CFDResult:
    condition: CFDCondition
    surface_p_i: dict          # local partial pressures at wafer (radial-averaged)
    surface_T_K: float
    GR_profile_r: np.ndarray   # GR(r) across wafer
    Ge_profile_r: np.ndarray


def export_mechanism_to_cfd(theta_posterior_mean: dict, path: str) -> None:
    """TODO(claude-code): write surface-reaction cards (S1..S9, G1..G3) with
    posterior-mean rate params to CFD-ACE+ format / UDF. Enforce site conservation.
    Consistency check: mechanism must reproduce the lumped power law in the
    surface-limited regime (Phase 9.3 note)."""
    raise NotImplementedError


def write_cfd_deck(cond: CFDCondition, path: str) -> None:
    """TODO(claude-code): serialize a CFD-ACE+ input deck for `cond`."""
    raise NotImplementedError


def parse_cfd_output(path: str) -> CFDResult:
    """TODO(claude-code): parse CFD-ACE+ output -> CFDResult."""
    raise NotImplementedError


def extract_transfer_priors(results: list[CFDResult]) -> dict:
    """TODO(claude-code): from CFD results, fit alpha_{i,r}=surface_p_i/inlet_p_i
    and dT_r=surface_T - T_set as functions of operating point. Return as PRIORS
    for reactor_transfer_model (Phase 9.4.3)."""
    raise NotImplementedError


# ============================================================================
# ===== MODULE: active_learning  (STUB) ======================================
# ============================================================================
# Minimize CFD-ACE+ runs by choosing the most informative conditions (Phase 10).
class ActiveLearner:
    """
    Loop: Sobol seed -> GP surrogate over CFD input space -> D-optimal/EIG
    acquisition (with cost) -> batch of k -> run CFD -> update -> repeat until
    delta_r posterior uncertainty < tol OR budget hit.
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.gp = None       # TODO(claude-code): GP over x_cfd -> {surface_p_i, dT} or delta_r
        self.done: list[CFDResult] = []

    def seed(self, n: int) -> list[CFDCondition]:
        """TODO(claude-code): Sobol/LHS space-filling seed of n conditions."""
        raise NotImplementedError

    def acquisition(self, candidates: list[CFDCondition]) -> np.ndarray:
        """TODO(claude-code): score candidates by expected information gain about
        the TARGET parameters (delta_r / sloppy chemistry dirs). Bayesian D-optimal:
        maximize det of expected posterior precision. Subtract a cost term."""
        raise NotImplementedError

    def select_batch(self, candidates: list[CFDCondition]) -> list[CFDCondition]:
        """TODO(claude-code): greedy top-k with diversity/repulsion penalty
        (reuse the role-quota batch selector pattern)."""
        raise NotImplementedError

    def step(self, candidate_pool: list[CFDCondition]) -> list[CFDCondition]:
        batch = self.select_batch(candidate_pool)
        # caller runs CFD on `batch`, appends CFDResults via self.ingest(), loops
        return batch

    def ingest(self, results: list[CFDResult]) -> None:
        self.done.extend(results)
        # TODO(claude-code): refit GP + delta_r posterior; check stop criterion.

    def budget_remaining(self) -> int:
        return self.cfg.cfd_run_budget - len(self.done)


# ============================================================================
# ===== MODULE: pipeline =====================================================
# ============================================================================
def run_tomasini_pilot(cfg: Config) -> dict:
    """
    Orchestrates Phases 1-8 (NO CFD). Returns the acceptance report.
    TODO(claude-code): fill ingest + prediction-in-real-units; this is the
    top-level 'reproduce Tomasini' entry point.
    """
    # 1. ingest -> Dataset (DS1..DS4)                      [TODO ingest_tomasini]
    # 2. registry + assembler; run anti-contamination test
    # 3. features (DS1)
    # 4. calibrate GR + Ge on DS1 (NUTS); calibrate B on DS2
    # 4.3 acceptance gates
    # 5. residual NN (optional for pilot pass)
    # 6. identifiability + sensitivity (reproduce Figs 4/5)
    # 7. reactor transfer: fit theta_chem on DS1, delta_r on DS3/DS4
    # 8. inverse design demo
    raise NotImplementedError("TODO(claude-code): wire the pilot end-to-end")


# ============================================================================
# ===== MODULE: cli / smoke test =============================================
# ============================================================================
def _smoke_test() -> None:
    """Minimal end-to-end sanity check on SYNTHETIC data so the core runs before
    real ingest exists. Generates data from known params, fits, checks recovery."""
    log.info("Running synthetic smoke test of GR physics core + calibration...")
    rng = np.random.default_rng(0)
    N = 120
    T = rng.uniform(873.0, 1038.0, N)                     # 600-765 C in K
    r_HCl = rng.uniform(0.1, 0.9, N)
    r_GeH4 = rng.uniform(0.02, 0.09, N)
    invT = 1.0 / T
    mu_inv, sd_inv = invT.mean(), invT.std()
    invT_s = (invT - mu_inv) / sd_inv
    # true params (kappa on standardized invT): pick kappa_true=-24507 -> std
    kappa_true_K = -24507.0
    kappa_std = kappa_true_K * sd_inv
    lnGR = (2.0 + kappa_std * invT_s
            + (-0.7) * np.log(r_HCl) + (1.3) * np.log(r_GeH4)
            + rng.normal(0, 0.05, N))
    X = jnp.asarray(np.stack([invT_s, np.log(r_HCl), np.log(r_GeH4), np.zeros(N)], axis=1))
    y_log = jnp.asarray(lnGR)

    cfg = Config(mcmc=MCMCConfig(num_warmup=500, num_samples=800, num_chains=2))
    mcmc = run_mcmc(gr_numpyro_model, X, y_log, cfg)
    s = mcmc.get_samples()
    kappa_rec_K = destandardize_kappa(float(np.mean(s["kappa_GR"])), (mu_inv, sd_inv))
    log.info("Recovered kappa_GR = %.0f K (true %.0f); gamma_HCl=%.2f (true -0.7); "
             "gamma_GeH4=%.2f (true 1.3)",
             kappa_rec_K, kappa_true_K,
             float(np.mean(s["gamma_HCl"])), float(np.mean(s["gamma_GeH4"])))
    assert abs(abs(kappa_rec_K) - 24507.0) / 24507.0 < 0.15, "kappa recovery off"
    log.info("Smoke test PASSED.")


if __name__ == "__main__":
    # Runs the synthetic recovery test so Claude Code can verify the core before
    # real data ingest is implemented. Replace with argparse CLI (calibrate/predict/
    # inverse/sensitivity/add-reactor/add-species/export-mechanism/active-learn).
    _smoke_test()
