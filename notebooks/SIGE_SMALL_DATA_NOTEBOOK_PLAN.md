# SiGe Small-Data Notebook Plan

This is a cell-by-cell notebook recipe for the current SiGe problem:

- Use only Tomasini DS1, the 70 undoped SiGe rows from Appendix 1.
- Use AMAT's small number of native SiGe conditions as the final calibration
  authority.
- Use AMAT spatial scans as real information, but hierarchically so one wafer
  with many scan points is not treated as many independent experiments.
- Reconcile precursor differences:
  - Tomasini: DCS + GeH4 + HCl.
  - AMAT: DCS + SiH4 + GeH4, no supplied HCl.
- Produce a provisional physics-kernel posterior that can later be exported to
  `surface_udf.c` after the mixed-Si-source terms are added to the package/UDF
  bridge.

The key modeling decision is:

$$
p_{\mathrm{Si,eff}}
=
p_{\mathrm{DCS}}
+\omega_{\mathrm{SiH4}}p_{\mathrm{SiH4}}
$$

and:

$$
p_{\mathrm{HCl,eff}}
=
p_{\mathrm{HCl,feed}}
+h_{\mathrm{floor}}p_{\mathrm{Si,eff}}
$$

For Tomasini:

$$
p_{\mathrm{SiH4}}=0,\qquad p_{\mathrm{HCl,feed}}>0
$$

For AMAT:

$$
p_{\mathrm{HCl,feed}}=0
$$

so \(h_{\mathrm{floor}}\) is not learned freely unless AMAT has an independent
measurement or CFD estimate of chlorine/HCl activity. Treat it as a fixed
sensitivity hyperparameter or a strongly regularized latent nuisance.

Do not set AMAT HCl to exactly zero and then feed
\(\log(10^{-30})\) into a Tomasini-fitted HCl power law. That would be a severe
out-of-domain extrapolation.

---

## Cell 1 - Notebook Title And Rules

```markdown
# SiGe Small-Data Bayesian Calibration

Goal:

Train a provisional SiGe chemistry model from:

1. Tomasini DS1 as a weak literature prior.
2. AMAT spatial scans as the final calibration target.
3. CFD-local fields later, once available, for transport deembedding.

Rules:

- Tomasini is not treated as AMAT data.
- Synthetic spatial data is not treated as experiment.
- Spatial points from one wafer are modeled hierarchically.
- AMAT no-HCl operation is treated as a separate domain with an effective
  HCl/chlorine floor, not as literal log(0).
- DCS and SiH4 are combined through an effective Si-source activity.
```

---

## Cell 2 - Imports

```python
from pathlib import Path

import arviz as az
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.infer import MCMC, NUTS, Predictive

from chem_ml.config import Config
from chem_ml.schema import ingest_tomasini
from chem_ml.spatial import wafer_average, wiwnu

numpyro.set_host_device_count(4)
```

---

## Cell 3 - Paths And Hyperparameters

```python
ROOT = Path.cwd()
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

# Replace these with the AMAT files once they are available.
AMAT_RECIPES_CSV = DATA_RAW / "amat_sige_recipes.csv"
AMAT_POINTS_CSV = DATA_RAW / "amat_sige_spatial_points.csv"

# Tomasini contributes prior information, not equal-weight production truth.
TOMASINI_PRIOR_WIDEN = 3.0

# AMAT has no supplied HCl. Use a fixed finite effective floor first, then
# run a sensitivity sweep over this value.
HCL_FLOOR_GRID = [1e-6, 1e-5, 1e-4, 1e-3]
HCL_FLOOR_DEFAULT = 1e-4

# SiH4 effective activity relative to DCS. Start broad. If AMAT has both DCS
# and SiH4 variation, this can be learned; otherwise fix/sweep it too.
OMEGA_SIH4_PRIOR_LOG_MEAN = np.log(1.0)
OMEGA_SIH4_PRIOR_LOG_SD = 1.0

# Weak pseudo-likelihood weight if you decide to include Tomasini again during
# AMAT fitting. Recommended first pass: False, because Tomasini already becomes
# the prior.
USE_TOMASINI_WEAK_LIKELIHOOD = False
LAMBDA_TOMASINI = 0.10

RNG_SEED = 7
```

---

## Cell 4 - Load Tomasini DS1 Only

```python
tomasini_all = ingest_tomasini(DATA_RAW)
tomasini_ds1 = tomasini_all.filter(source_dataset="DS1")

tom = tomasini_ds1.to_dataframe().copy()
tom["source"] = "Tomasini_DS1"
tom["T_C"] = tom["T_K"] - 273.15
tom["DCS_sccm"] = 1.0
tom["SiH4_sccm"] = 0.0
tom["GeH4_sccm"] = tom["p_GeH4"]
tom["HCl_sccm"] = tom["p_HCl"]
tom["GR_nm_min"] = tom["GR_nm_min"].astype(float)
tom["Ge_at_frac"] = tom["Ge_at_frac"].astype(float)
tom["run_id"] = ["tom_%03d" % i for i in range(len(tom))]

tom[["run_id", "T_C", "HCl_sccm", "GeH4_sccm", "GR_nm_min", "Ge_at_frac"]].head()
```

Expected result: 70 rows, no boron chemistry.

---

## Cell 5 - Define Expected AMAT CSV Schemas

Use two files.

`amat_sige_recipes.csv`:

```text
run_id,T_C,P_torr,DCS_sccm,SiH4_sccm,GeH4_sccm,HCl_sccm,H2_sccm,N2_sccm,growth_time_s
```

`HCl_sccm` may be omitted or set to zero.

`amat_sige_spatial_points.csv`:

```text
run_id,x_mm,y_mm,GR_nm_min_local,Ge_at_pct_local,measurement_source
```

If only thickness is available:

```text
run_id,x_mm,y_mm,thickness_A_local,Ge_at_pct_local,measurement_source
```

and use `growth_time_s` from the recipe table.

```python
if not AMAT_RECIPES_CSV.exists() or not AMAT_POINTS_CSV.exists():
    print("AMAT files not found yet.")
    print("Create:", AMAT_RECIPES_CSV)
    print("Create:", AMAT_POINTS_CSV)
```

---

## Cell 6 - Load AMAT Spatial Data

```python
def load_amat_spatial(recipes_csv: Path, points_csv: Path) -> pd.DataFrame:
    recipes = pd.read_csv(recipes_csv)
    points = pd.read_csv(points_csv)

    if "HCl_sccm" not in recipes.columns:
        recipes["HCl_sccm"] = 0.0
    for col in ["DCS_sccm", "SiH4_sccm", "GeH4_sccm", "HCl_sccm", "H2_sccm", "N2_sccm"]:
        if col not in recipes.columns:
            recipes[col] = 0.0

    df = points.merge(recipes, on="run_id", how="left", validate="many_to_one")
    if df["T_C"].isna().any():
        missing = df.loc[df["T_C"].isna(), "run_id"].unique()
        raise ValueError(f"spatial points reference missing recipe run_id(s): {missing}")

    if "GR_nm_min_local" not in df.columns or df["GR_nm_min_local"].isna().all():
        if "thickness_A_local" not in df.columns or "growth_time_s" not in df.columns:
            raise ValueError("Need GR_nm_min_local, or thickness_A_local plus growth_time_s")
        df["GR_nm_min_local"] = df["thickness_A_local"] * 6.0 / df["growth_time_s"]

    df["Ge_at_frac_local"] = df["Ge_at_pct_local"] / 100.0
    df["r_mm"] = np.sqrt(df["x_mm"].astype(float)**2 + df["y_mm"].astype(float)**2)
    df["T_K"] = df["T_C"].astype(float) + 273.15
    df["source"] = "AMAT_spatial"

    return df

if AMAT_RECIPES_CSV.exists() and AMAT_POINTS_CSV.exists():
    amat_pts = load_amat_spatial(AMAT_RECIPES_CSV, AMAT_POINTS_CSV)
    display(amat_pts.head())
    print(amat_pts.groupby("run_id").size())
```

---

## Cell 7 - Wafer-Level Summaries

These summaries are not a replacement for spatial fitting. They are sanity
checks and parity metrics.

```python
def summarize_wafer(df_one: pd.DataFrame) -> pd.Series:
    r = df_one["r_mm"].to_numpy(float)
    gr = df_one["GR_nm_min_local"].to_numpy(float)
    ge = df_one["Ge_at_frac_local"].to_numpy(float)
    return pd.Series({
        "n_points": len(df_one),
        "GR_area_mean": wafer_average(r, gr),
        "Ge_area_mean": wafer_average(r, ge),
        "GR_wiwnu": wiwnu(r, gr),
        "Ge_wiwnu": wiwnu(r, ge),
        "T_C": df_one["T_C"].iloc[0],
        "DCS_sccm": df_one["DCS_sccm"].iloc[0],
        "SiH4_sccm": df_one["SiH4_sccm"].iloc[0],
        "GeH4_sccm": df_one["GeH4_sccm"].iloc[0],
        "HCl_sccm": df_one["HCl_sccm"].iloc[0],
    })

if "amat_pts" in globals():
    amat_summary = amat_pts.groupby("run_id").apply(summarize_wafer)
    display(amat_summary)
```

---

## Cell 8 - Effective Feature Builder

This is the reconciliation layer between Tomasini and AMAT.

```python
EPS = 1e-30

def add_effective_features(
    df: pd.DataFrame,
    *,
    omega_sih4: float,
    hcl_floor: float,
    invT_mu: float | None = None,
    invT_sd: float | None = None,
) -> tuple[pd.DataFrame, tuple[float, float]]:
    out = df.copy()
    out["T_K"] = out["T_C"].astype(float) + 273.15

    dcs = out.get("DCS_sccm", 0.0).astype(float)
    sih4 = out.get("SiH4_sccm", 0.0).astype(float)
    geh4 = out.get("GeH4_sccm", 0.0).astype(float)
    hcl_feed = out.get("HCl_sccm", 0.0).astype(float)

    p_si_eff = dcs + omega_sih4 * sih4
    if (p_si_eff <= 0).any():
        raise ValueError("Every row needs positive DCS + omega_sih4*SiH4")

    p_hcl_eff = hcl_feed + hcl_floor * p_si_eff

    out["p_Si_eff"] = p_si_eff
    out["p_HCl_eff"] = p_hcl_eff
    out["ln_Ge_over_Si_eff"] = np.log((geh4 + EPS) / (p_si_eff + EPS))
    out["ln_HCl_eff_over_Si_eff"] = np.log((p_hcl_eff + EPS) / (p_si_eff + EPS))
    out["SiH4_fraction_raw"] = sih4 / (dcs + sih4 + EPS)

    invT = 1.0 / out["T_K"].to_numpy(float)
    if invT_mu is None:
        invT_mu = float(invT.mean())
    if invT_sd is None:
        invT_sd = float(invT.std() + 1e-12)
    out["invT_std"] = (invT - invT_mu) / invT_sd

    return out, (invT_mu, invT_sd)

tom_feat, invT_scaler = add_effective_features(
    tom,
    omega_sih4=1.0,
    hcl_floor=0.0,
)
invT_scaler
```

---

## Cell 9 - Plot Tomasini Feature Domain

```python
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].scatter(tom_feat["ln_HCl_eff_over_Si_eff"], np.log(tom_feat["GR_nm_min"]))
ax[0].set_xlabel("ln(HCl/Si_eff)")
ax[0].set_ylabel("ln(GR)")

ax[1].scatter(tom_feat["ln_Ge_over_Si_eff"], np.log(tom_feat["Ge_at_frac"] / (1 - tom_feat["Ge_at_frac"])))
ax[1].set_xlabel("ln(GeH4/Si_eff)")
ax[1].set_ylabel("logit(Ge)")

plt.tight_layout()
```

This plot makes the AMAT no-HCl issue visible later. AMAT will sit outside
the Tomasini HCl range unless we use an effective finite HCl/chlorine floor.

---

## Cell 10 - Matrix Builder

```python
FEATURE_COLS = [
    "invT_std",
    "ln_HCl_eff_over_Si_eff",
    "ln_Ge_over_Si_eff",
    "SiH4_fraction_raw",
]

def make_X(df: pd.DataFrame) -> jnp.ndarray:
    return jnp.asarray(df[FEATURE_COLS].to_numpy(float))

def logit(x):
    x = np.asarray(x, dtype=float)
    return np.log(x / (1.0 - x))

X_tom = make_X(tom_feat)
y_gr_tom = jnp.asarray(np.log(tom_feat["GR_nm_min"].to_numpy(float)))
y_ge_tom = jnp.asarray(logit(tom_feat["Ge_at_frac"].to_numpy(float)))
```

---

## Cell 11 - Tomasini Physics Model

Here \(X[:,3]\), the SiH4 fraction, is always zero. Tomasini cannot identify
the SiH4 correction.

```python
def tomasini_sige_model(X, y_gr=None, y_ge=None):
    invT, ln_hcl, ln_ge, sih4_frac = X[:, 0], X[:, 1], X[:, 2], X[:, 3]

    lnK_GR = numpyro.sample("lnK_GR", dist.Normal(0.0, 10.0))
    kappa_GR = numpyro.sample("kappa_GR", dist.Normal(-24507.0 * invT_scaler[1], 5000.0 * invT_scaler[1]))
    gamma_HCl_GR = numpyro.sample("gamma_HCl_GR", dist.Normal(-0.7, 0.4))
    gamma_Ge_GR = numpyro.sample("gamma_Ge_GR", dist.Normal(1.3, 0.4))

    lnK_Ge = numpyro.sample("lnK_Ge", dist.Normal(0.0, 10.0))
    kappa_Ge = numpyro.sample("kappa_Ge", dist.Normal(4319.0 * invT_scaler[1], 3000.0 * invT_scaler[1]))
    gamma_HCl_Ge = numpyro.sample("gamma_HCl_Ge", dist.Normal(0.1, 0.3))
    gamma_Ge_Ge = numpyro.sample("gamma_Ge_Ge", dist.Normal(0.5, 0.3))

    sigma_GR = numpyro.sample("sigma_GR", dist.HalfNormal(0.5))
    sigma_Ge = numpyro.sample("sigma_Ge", dist.HalfNormal(0.5))

    mu_gr = lnK_GR + kappa_GR * invT + gamma_HCl_GR * ln_hcl + gamma_Ge_GR * ln_ge
    mu_ge = lnK_Ge + kappa_Ge * invT + gamma_HCl_Ge * ln_hcl + gamma_Ge_Ge * ln_ge

    numpyro.sample("obs_GR", dist.Normal(mu_gr, sigma_GR), obs=y_gr)
    numpyro.sample("obs_Ge", dist.Normal(mu_ge, sigma_Ge), obs=y_ge)
```

---

## Cell 12 - Fit Tomasini DS1

```python
tom_kernel = NUTS(tomasini_sige_model, target_accept_prob=0.9)
tom_mcmc = MCMC(tom_kernel, num_warmup=1500, num_samples=2000, num_chains=4)
tom_mcmc.run(
    jax.random.PRNGKey(RNG_SEED),
    X=X_tom,
    y_gr=y_gr_tom,
    y_ge=y_ge_tom,
)
tom_mcmc.print_summary()

idata_tom = az.from_numpyro(tom_mcmc)
az.summary(idata_tom, var_names=[
    "lnK_GR", "kappa_GR", "gamma_HCl_GR", "gamma_Ge_GR",
    "lnK_Ge", "kappa_Ge", "gamma_HCl_Ge", "gamma_Ge_Ge",
    "sigma_GR", "sigma_Ge",
])
```

---

## Cell 13 - Convert Tomasini Posterior To Inflated Priors

```python
TOM_PARAM_NAMES = [
    "lnK_GR", "kappa_GR", "gamma_HCl_GR", "gamma_Ge_GR",
    "lnK_Ge", "kappa_Ge", "gamma_HCl_Ge", "gamma_Ge_Ge",
]

def posterior_normal_priors(mcmc, names, widen=3.0):
    samples = mcmc.get_samples()
    priors = {}
    for name in names:
        v = np.asarray(samples[name])
        priors[name] = (float(v.mean()), float(v.std() * widen + 1e-9))
    return priors

tom_priors = posterior_normal_priors(tom_mcmc, TOM_PARAM_NAMES, TOMASINI_PRIOR_WIDEN)
tom_priors
```

Interpretation:

- These are not AMAT-fitted parameters.
- They are broad literature-informed priors.
- The widening factor protects against reactor, precursor, pressure, and
  metrology differences.

---

## Cell 14 - Prepare AMAT Spatial Arrays

```python
if "amat_pts" in globals():
    amat_feat, _ = add_effective_features(
        amat_pts,
        omega_sih4=1.0,
        hcl_floor=HCL_FLOOR_DEFAULT,
        invT_mu=invT_scaler[0],
        invT_sd=invT_scaler[1],
    )

    run_ids = sorted(amat_feat["run_id"].unique())
    run_to_idx = {r: i for i, r in enumerate(run_ids)}
    amat_feat["cond_idx"] = amat_feat["run_id"].map(run_to_idx)
    n_by_cond = amat_feat.groupby("run_id").size().reindex(run_ids).to_numpy()
    amat_feat["n_points_this_condition"] = amat_feat["run_id"].map(dict(zip(run_ids, n_by_cond)))

    display(amat_feat[[
        "run_id", "T_C", "DCS_sccm", "SiH4_sccm", "GeH4_sccm", "HCl_sccm",
        "ln_HCl_eff_over_Si_eff", "ln_Ge_over_Si_eff", "SiH4_fraction_raw",
        "GR_nm_min_local", "Ge_at_frac_local",
    ]].head())
```

---

## Cell 15 - AMAT Hierarchical Physics Model

This is the main small-data model.

Important details:

- Tomasini posterior gives broad priors.
- AMAT spatial points are real observations.
- Each wafer has a random condition effect.
- Point likelihood is inflated by \(\sqrt{N_c}\) so one wafer with many points
  does not dominate as if it were many independent recipes.
- SiH4 activity is learned through \(\omega_{\mathrm{SiH4}}\), if AMAT varies
  DCS/SiH4 enough to identify it.
- HCl floor can be fixed first. Only sample it if there is independent
  evidence.

```python
def amat_sige_spatial_model(
    T_K,
    DCS,
    SiH4,
    GeH4,
    HCl_feed,
    cond_idx,
    n_points_this_condition,
    y_gr=None,
    y_ge=None,
    hcl_floor_fixed=HCL_FLOOR_DEFAULT,
    sample_hcl_floor=False,
):
    n_cond = int(jnp.max(cond_idx)) + 1

    # Literature-informed priors from Tomasini, widened.
    lnK_GR = numpyro.sample("lnK_GR", dist.Normal(*tom_priors["lnK_GR"]))
    kappa_GR = numpyro.sample("kappa_GR", dist.Normal(*tom_priors["kappa_GR"]))
    gamma_HCl_GR = numpyro.sample("gamma_HCl_GR", dist.Normal(*tom_priors["gamma_HCl_GR"]))
    gamma_Ge_GR = numpyro.sample("gamma_Ge_GR", dist.Normal(*tom_priors["gamma_Ge_GR"]))

    lnK_Ge = numpyro.sample("lnK_Ge", dist.Normal(*tom_priors["lnK_Ge"]))
    kappa_Ge = numpyro.sample("kappa_Ge", dist.Normal(*tom_priors["kappa_Ge"]))
    gamma_HCl_Ge = numpyro.sample("gamma_HCl_Ge", dist.Normal(*tom_priors["gamma_HCl_Ge"]))
    gamma_Ge_Ge = numpyro.sample("gamma_Ge_Ge", dist.Normal(*tom_priors["gamma_Ge_Ge"]))

    # AMAT-only mixed-Si-source correction.
    log_omega_sih4 = numpyro.sample(
        "log_omega_sih4",
        dist.Normal(OMEGA_SIH4_PRIOR_LOG_MEAN, OMEGA_SIH4_PRIOR_LOG_SD),
    )
    omega_sih4 = jnp.exp(log_omega_sih4)

    beta_sih4_GR = numpyro.sample("beta_sih4_GR", dist.Normal(0.0, 0.5))
    beta_sih4_Ge = numpyro.sample("beta_sih4_Ge", dist.Normal(0.0, 0.5))

    if sample_hcl_floor:
        log_hcl_floor = numpyro.sample("log_hcl_floor", dist.Normal(np.log(hcl_floor_fixed), 1.0))
        hcl_floor = jnp.exp(log_hcl_floor)
    else:
        hcl_floor = hcl_floor_fixed

    p_si_eff = DCS + omega_sih4 * SiH4
    p_hcl_eff = HCl_feed + hcl_floor * p_si_eff
    ln_ge = jnp.log((GeH4 + EPS) / (p_si_eff + EPS))
    ln_hcl = jnp.log((p_hcl_eff + EPS) / (p_si_eff + EPS))
    sih4_frac = SiH4 / (DCS + SiH4 + EPS)
    invT = (1.0 / T_K - invT_scaler[0]) / invT_scaler[1]

    sigma_cond_GR = numpyro.sample("sigma_cond_GR", dist.HalfNormal(0.15))
    sigma_cond_Ge = numpyro.sample("sigma_cond_Ge", dist.HalfNormal(0.15))
    b_GR = numpyro.sample("b_GR_condition", dist.Normal(0.0, sigma_cond_GR).expand([n_cond]))
    b_Ge = numpyro.sample("b_Ge_condition", dist.Normal(0.0, sigma_cond_Ge).expand([n_cond]))

    sigma_point_GR = numpyro.sample("sigma_point_GR", dist.HalfNormal(0.35))
    sigma_point_Ge = numpyro.sample("sigma_point_Ge", dist.HalfNormal(0.35))
    nu = numpyro.sample("student_t_nu", dist.Exponential(1 / 10.0)) + 2.0

    mu_gr = (
        lnK_GR
        + kappa_GR * invT
        + gamma_HCl_GR * ln_hcl
        + gamma_Ge_GR * ln_ge
        + beta_sih4_GR * sih4_frac
        + b_GR[cond_idx]
    )
    mu_ge = (
        lnK_Ge
        + kappa_Ge * invT
        + gamma_HCl_Ge * ln_hcl
        + gamma_Ge_Ge * ln_ge
        + beta_sih4_Ge * sih4_frac
        + b_Ge[cond_idx]
    )

    wafer_weight_scale = jnp.sqrt(n_points_this_condition)
    numpyro.sample(
        "obs_GR",
        dist.StudentT(nu, mu_gr, sigma_point_GR * wafer_weight_scale),
        obs=y_gr,
    )
    numpyro.sample(
        "obs_Ge",
        dist.StudentT(nu, mu_ge, sigma_point_Ge * wafer_weight_scale),
        obs=y_ge,
    )
```

---

## Cell 16 - Fit AMAT Model

```python
if "amat_feat" in globals():
    T_K = jnp.asarray(amat_feat["T_K"].to_numpy(float))
    DCS = jnp.asarray(amat_feat["DCS_sccm"].to_numpy(float))
    SiH4 = jnp.asarray(amat_feat["SiH4_sccm"].to_numpy(float))
    GeH4 = jnp.asarray(amat_feat["GeH4_sccm"].to_numpy(float))
    HCl_feed = jnp.asarray(amat_feat["HCl_sccm"].to_numpy(float))
    cond_idx = jnp.asarray(amat_feat["cond_idx"].to_numpy(int))
    n_points_this_condition = jnp.asarray(amat_feat["n_points_this_condition"].to_numpy(float))
    y_gr_amat = jnp.asarray(np.log(amat_feat["GR_nm_min_local"].to_numpy(float)))
    y_ge_amat = jnp.asarray(logit(amat_feat["Ge_at_frac_local"].to_numpy(float)))

    amat_kernel = NUTS(amat_sige_spatial_model, target_accept_prob=0.92)
    amat_mcmc = MCMC(amat_kernel, num_warmup=2000, num_samples=3000, num_chains=4)
    amat_mcmc.run(
        jax.random.PRNGKey(RNG_SEED + 1),
        T_K=T_K,
        DCS=DCS,
        SiH4=SiH4,
        GeH4=GeH4,
        HCl_feed=HCl_feed,
        cond_idx=cond_idx,
        n_points_this_condition=n_points_this_condition,
        y_gr=y_gr_amat,
        y_ge=y_ge_amat,
        hcl_floor_fixed=HCL_FLOOR_DEFAULT,
        sample_hcl_floor=False,
    )
    amat_mcmc.print_summary()
```

---

## Cell 17 - Diagnostics

```python
if "amat_mcmc" in globals():
    idata_amat = az.from_numpyro(amat_mcmc)
    display(az.summary(idata_amat, var_names=[
        "lnK_GR", "kappa_GR", "gamma_HCl_GR", "gamma_Ge_GR",
        "lnK_Ge", "kappa_Ge", "gamma_HCl_Ge", "gamma_Ge_Ge",
        "log_omega_sih4", "beta_sih4_GR", "beta_sih4_Ge",
        "sigma_cond_GR", "sigma_cond_Ge", "sigma_point_GR", "sigma_point_Ge",
    ]))

    az.plot_trace(idata_amat, var_names=[
        "lnK_GR", "kappa_GR", "gamma_Ge_GR",
        "lnK_Ge", "kappa_Ge", "gamma_Ge_Ge",
        "log_omega_sih4",
    ])
    plt.tight_layout()
```

Reject or revise the fit if:

- divergences are nonzero,
- \(\hat R > 1.01\),
- \(\omega_{\mathrm{SiH4}}\) is prior-dominated and AMAT varies SiH4 heavily,
- condition effects dominate the physics terms,
- posterior uncertainty collapses despite only 4-6 real conditions.

---

## Cell 18 - Posterior Prediction Helper

```python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def posterior_predict_amat(mcmc, **kwargs):
    pred = Predictive(amat_sige_spatial_model, posterior_samples=mcmc.get_samples())
    return pred(jax.random.PRNGKey(RNG_SEED + 2), **kwargs)

if "amat_mcmc" in globals():
    pp = posterior_predict_amat(
        amat_mcmc,
        T_K=T_K,
        DCS=DCS,
        SiH4=SiH4,
        GeH4=GeH4,
        HCl_feed=HCl_feed,
        cond_idx=cond_idx,
        n_points_this_condition=n_points_this_condition,
        y_gr=None,
        y_ge=None,
        hcl_floor_fixed=HCL_FLOOR_DEFAULT,
        sample_hcl_floor=False,
    )
    gr_pred = np.exp(np.asarray(pp["obs_GR"]))
    ge_pred = sigmoid(np.asarray(pp["obs_Ge"]))
```

---

## Cell 19 - Spatial Parity Plots

```python
if "gr_pred" in globals():
    amat_plot = amat_feat.copy()
    amat_plot["GR_pred_mean"] = gr_pred.mean(axis=0)
    amat_plot["Ge_pred_mean"] = ge_pred.mean(axis=0)

    for run_id, d in amat_plot.groupby("run_id"):
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        order = np.argsort(d["r_mm"].to_numpy(float))

        ax[0].plot(d["r_mm"].to_numpy(float)[order], d["GR_nm_min_local"].to_numpy(float)[order], "o", label="measured")
        ax[0].plot(d["r_mm"].to_numpy(float)[order], d["GR_pred_mean"].to_numpy(float)[order], "-", label="predicted")
        ax[0].set_title(f"{run_id}: GR")
        ax[0].set_xlabel("r_mm")
        ax[0].set_ylabel("nm/min")
        ax[0].legend()

        ax[1].plot(d["r_mm"].to_numpy(float)[order], d["Ge_at_frac_local"].to_numpy(float)[order] * 100, "o", label="measured")
        ax[1].plot(d["r_mm"].to_numpy(float)[order], d["Ge_pred_mean"].to_numpy(float)[order] * 100, "-", label="predicted")
        ax[1].set_title(f"{run_id}: Ge")
        ax[1].set_xlabel("r_mm")
        ax[1].set_ylabel("at.%")
        ax[1].legend()

        plt.tight_layout()
        plt.show()
```

---

## Cell 20 - Wafer-Average Validation Metrics

```python
if "amat_plot" in globals():
    rows = []
    for run_id, d in amat_plot.groupby("run_id"):
        r = d["r_mm"].to_numpy(float)
        rows.append({
            "run_id": run_id,
            "GR_meas_avg": wafer_average(r, d["GR_nm_min_local"].to_numpy(float)),
            "GR_pred_avg": wafer_average(r, d["GR_pred_mean"].to_numpy(float)),
            "Ge_meas_avg": wafer_average(r, d["Ge_at_frac_local"].to_numpy(float)),
            "Ge_pred_avg": wafer_average(r, d["Ge_pred_mean"].to_numpy(float)),
            "GR_meas_wiwnu": wiwnu(r, d["GR_nm_min_local"].to_numpy(float)),
            "GR_pred_wiwnu": wiwnu(r, d["GR_pred_mean"].to_numpy(float)),
            "Ge_meas_wiwnu": wiwnu(r, d["Ge_at_frac_local"].to_numpy(float)),
            "Ge_pred_wiwnu": wiwnu(r, d["Ge_pred_mean"].to_numpy(float)),
        })
    wafer_metrics = pd.DataFrame(rows)
    wafer_metrics["GR_avg_pct_err"] = 100 * (wafer_metrics["GR_pred_avg"] - wafer_metrics["GR_meas_avg"]) / wafer_metrics["GR_meas_avg"]
    wafer_metrics["Ge_avg_abs_err_atpct"] = 100 * (wafer_metrics["Ge_pred_avg"] - wafer_metrics["Ge_meas_avg"])
    display(wafer_metrics)
```

---

## Cell 21 - HCl Floor Sensitivity

Because AMAT supplied no HCl, this is mandatory. If predictions move wildly
across plausible \(h_{\mathrm{floor}}\), the model is not yet chemically
identified.

```python
def fit_one_hcl_floor(hcl_floor):
    kernel = NUTS(amat_sige_spatial_model, target_accept_prob=0.92)
    mcmc = MCMC(kernel, num_warmup=1200, num_samples=1200, num_chains=4, progress_bar=False)
    mcmc.run(
        jax.random.PRNGKey(RNG_SEED + int(abs(np.log10(hcl_floor))) + 10),
        T_K=T_K,
        DCS=DCS,
        SiH4=SiH4,
        GeH4=GeH4,
        HCl_feed=HCl_feed,
        cond_idx=cond_idx,
        n_points_this_condition=n_points_this_condition,
        y_gr=y_gr_amat,
        y_ge=y_ge_amat,
        hcl_floor_fixed=hcl_floor,
        sample_hcl_floor=False,
    )
    s = mcmc.get_samples()
    return {
        "hcl_floor": hcl_floor,
        "omega_sih4_mean": float(np.exp(np.asarray(s["log_omega_sih4"])).mean()),
        "gamma_HCl_GR_mean": float(np.asarray(s["gamma_HCl_GR"]).mean()),
        "gamma_HCl_Ge_mean": float(np.asarray(s["gamma_HCl_Ge"]).mean()),
        "sigma_point_GR_mean": float(np.asarray(s["sigma_point_GR"]).mean()),
        "sigma_point_Ge_mean": float(np.asarray(s["sigma_point_Ge"]).mean()),
    }

if "amat_mcmc" in globals():
    hcl_sweep = pd.DataFrame([fit_one_hcl_floor(v) for v in HCL_FLOOR_GRID])
    display(hcl_sweep)
```

Decision rule:

- If GR/Ge predictions are stable over \(10^{-6}\) to \(10^{-3}\), use fixed
  \(h_{\mathrm{floor}}\).
- If not stable, do not claim AMAT no-HCl chemistry is calibrated. Request
  either purge/HCl measurement, CFD chlorine/HCl estimate, or one deliberate
  low-HCl perturbation run.

---

## Cell 22 - Optional Tomasini Pseudo-Spatial Regularization

Use this only after AMAT spatial patterns are understood. It creates weak
pseudo-spatial profiles whose wafer average matches Tomasini scalar values.

This should never be mixed into the final dataset at equal weight.

```python
def centered_log_spatial_modes(amat_plot: pd.DataFrame, value_col: str, pred_col: str):
    modes = []
    for run_id, d in amat_plot.groupby("run_id"):
        r = d["r_mm"].to_numpy(float)
        observed = d[value_col].to_numpy(float)
        predicted = d[pred_col].to_numpy(float)
        eps = np.log(observed) - np.log(predicted)
        eps = eps - wafer_average(r, eps)
        modes.append(pd.DataFrame({"run_id": run_id, "r_mm": r, "eps": eps}))
    return modes

def make_pseudo_gr_profile(r_mm, eps, gr_scalar):
    correction = np.log(wafer_average(r_mm, np.exp(eps)))
    return gr_scalar * np.exp(eps - correction)

if "amat_plot" in globals():
    gr_modes = centered_log_spatial_modes(amat_plot, "GR_nm_min_local", "GR_pred_mean")
    mode0 = gr_modes[0]
    pseudo = make_pseudo_gr_profile(
        mode0["r_mm"].to_numpy(float),
        mode0["eps"].to_numpy(float),
        gr_scalar=float(tom_feat["GR_nm_min"].iloc[0]),
    )
    print("Pseudo profile average:", wafer_average(mode0["r_mm"].to_numpy(float), pseudo))
    print("Target Tomasini scalar:", float(tom_feat["GR_nm_min"].iloc[0]))
```

If used in fitting, assign large observation noise or small weight. The
purpose is to regularize spatial shape, not to invent new AMAT wafers.

---

## Cell 23 - Leave-One-Condition-Out Validation

With only 6 AMAT conditions, this is more informative than a random point
split.

```python
def fit_leave_one_condition_out(left_out_run_id: str, hcl_floor=HCL_FLOOR_DEFAULT):
    train = amat_feat[amat_feat["run_id"] != left_out_run_id].copy()
    test = amat_feat[amat_feat["run_id"] == left_out_run_id].copy()

    run_ids_train = sorted(train["run_id"].unique())
    run_to_idx_train = {r: i for i, r in enumerate(run_ids_train)}
    train["cond_idx"] = train["run_id"].map(run_to_idx_train)
    n_train = train.groupby("run_id").size().reindex(run_ids_train).to_numpy()
    train["n_points_this_condition"] = train["run_id"].map(dict(zip(run_ids_train, n_train)))

    kernel = NUTS(amat_sige_spatial_model, target_accept_prob=0.92)
    mcmc = MCMC(kernel, num_warmup=1200, num_samples=1200, num_chains=4, progress_bar=False)
    mcmc.run(
        jax.random.PRNGKey(RNG_SEED + 100 + list(sorted(amat_feat["run_id"].unique())).index(left_out_run_id)),
        T_K=jnp.asarray(train["T_K"].to_numpy(float)),
        DCS=jnp.asarray(train["DCS_sccm"].to_numpy(float)),
        SiH4=jnp.asarray(train["SiH4_sccm"].to_numpy(float)),
        GeH4=jnp.asarray(train["GeH4_sccm"].to_numpy(float)),
        HCl_feed=jnp.asarray(train["HCl_sccm"].to_numpy(float)),
        cond_idx=jnp.asarray(train["cond_idx"].to_numpy(int)),
        n_points_this_condition=jnp.asarray(train["n_points_this_condition"].to_numpy(float)),
        y_gr=jnp.asarray(np.log(train["GR_nm_min_local"].to_numpy(float))),
        y_ge=jnp.asarray(logit(train["Ge_at_frac_local"].to_numpy(float))),
        hcl_floor_fixed=hcl_floor,
        sample_hcl_floor=False,
    )
    return mcmc, test

# Run only when you are ready; this can take time.
# loo_results = {rid: fit_leave_one_condition_out(rid) for rid in sorted(amat_feat["run_id"].unique())}
```

LOO failure is not fatal, but it tells you which new experiment would be most
valuable.

---

## Cell 24 - Decide Whether Residual NN Is Justified

Do not fit a residual NN automatically with 4-6 conditions.

Fit residual correction only if all are true:

1. Physics-only posterior diagnostics are clean.
2. HCl-floor sensitivity is not dominating conclusions.
3. Leave-one-condition-out error has a repeated structure.
4. The residual improves both wafer-average and spatial-profile metrics.
5. The residual amplitude stays small in log space, for example:

$$
|g_{\mathrm{NN}}| \le 0.25
$$

In notebook terms, first plot residuals:

```python
if "amat_plot" in globals():
    amat_plot["resid_log_GR"] = np.log(amat_plot["GR_nm_min_local"]) - np.log(amat_plot["GR_pred_mean"])
    amat_plot["resid_logit_Ge"] = logit(amat_plot["Ge_at_frac_local"]) - logit(amat_plot["Ge_pred_mean"])

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].scatter(amat_plot["r_mm"], amat_plot["resid_log_GR"], c=amat_plot["cond_idx"])
    ax[0].axhline(0, color="k", lw=1)
    ax[0].set_title("GR residual")

    ax[1].scatter(amat_plot["r_mm"], amat_plot["resid_logit_Ge"], c=amat_plot["cond_idx"])
    ax[1].axhline(0, color="k", lw=1)
    ax[1].set_title("Ge residual")

    plt.tight_layout()
```

If residuals are just one condition being off, active learning is better than
a neural residual. If residuals have repeatable radial/local-state structure,
then a tiny bounded residual is justified.

---

## Cell 25 - Provisional Parameter Package

This extracts posterior means. It is not yet the final repo model package
because the current `surface_udf.c` exporter needs to be extended to evaluate:

$$
p_{\mathrm{Si,eff}}
=
p_{\mathrm{DCS}}
+\omega_{\mathrm{SiH4}}p_{\mathrm{SiH4}}
$$

inside the UDF.

```python
if "amat_mcmc" in globals():
    s = amat_mcmc.get_samples()
    posterior_mean = {k: float(np.asarray(v).mean()) for k, v in s.items() if np.asarray(v).ndim == 1}

    provisional_package = {
        "chem_class": "SiGe",
        "target_deposit": "SiGe",
        "species": ["dichlorosilane", "silane", "germane", "hydrogen"],
        "feature_model": {
            "p_Si_eff": "p_DCS + omega_sih4 * p_SiH4",
            "p_HCl_eff": "p_HCl_feed + hcl_floor * p_Si_eff",
            "ln_Ge": "log(p_GeH4 / p_Si_eff)",
            "ln_HCl": "log(p_HCl_eff / p_Si_eff)",
        },
        "invT_scaler": list(invT_scaler),
        "hcl_floor_fixed": HCL_FLOOR_DEFAULT,
        "theta_mean": posterior_mean,
        "training_source": "Tomasini_DS1_prior_plus_AMAT_spatial",
        "transport_deembedding": "not_started",
    }

    import json
    out_path = DATA_PROCESSED / "sige_small_data_provisional_package.json"
    out_path.write_text(json.dumps(provisional_package, indent=2))
    out_path
```

---

## Cell 26 - CFD Deembedding Iteration

Once CFD-ACE+ is available:

1. Export provisional UDF after adding mixed-Si-source support.
2. Run the 6 AMAT conditions in CFD-ACE+.
3. Export wall-adjacent local fields:

```text
condition_id,r_mm,T_wall_K,p_DCS,p_SiH4,p_GeH4,p_HCl,u_tau_or_u,mu,rho,D_GeH4,D_HCl,GR_pred,Ge_pred
```

4. Replace setpoint features in the AMAT model with CFD-local features:

$$
\mathbf q_{cj}
=
\left[
T_{cj}^{\mathrm{wall}},
p_{\mathrm{DCS},cj},
p_{\mathrm{SiH4},cj},
p_{\mathrm{GeH4},cj},
p_{\mathrm{HCl},cj}^{\mathrm{eff}},
\mathrm{Da}_{i,cj},
\delta_{i,cj}
\right]
$$

5. Refit.
6. Stop when parameters and wafer metrics stabilize.

Convergence means:

$$
\|\theta^{(n+1)}-\theta^{(n)}\|_{\Sigma^{-1}} < \epsilon_\theta
$$

and:

$$
\max_c
\left|
\overline{\mathrm{GR}}_c^{(n+1)}
-
\overline{\mathrm{GR}}_c^{(n)}
\right|
<
\epsilon_{\mathrm{GR}}
$$

and:

$$
\max_c
\left|
\bar{x}_{\mathrm{Ge},c}^{(n+1)}
-
\bar{x}_{\mathrm{Ge},c}^{(n)}
\right|
<
\epsilon_x
$$

and residuals no longer show systematic spatial or reactor structure.

---

## Cell 27 - Minimum Output Report

At the end of each notebook run, save:

```python
report = {
    "n_tomasini_ds1": int(len(tom_feat)),
    "n_amat_conditions": int(amat_feat["run_id"].nunique()) if "amat_feat" in globals() else 0,
    "n_amat_spatial_points": int(len(amat_feat)) if "amat_feat" in globals() else 0,
    "invT_scaler": list(invT_scaler),
    "hcl_floor_default": HCL_FLOOR_DEFAULT,
    "hcl_floor_grid": HCL_FLOOR_GRID,
}

if "wafer_metrics" in globals():
    report["wafer_metrics"] = wafer_metrics.to_dict(orient="records")

if "hcl_sweep" in globals():
    report["hcl_sweep"] = hcl_sweep.to_dict(orient="records")

import json
report_path = DATA_PROCESSED / "sige_small_data_notebook_report.json"
report_path.write_text(json.dumps(report, indent=2))
report_path
```

---

## Practical Decision Tree

### If the AMAT 6 conditions vary DCS/SiH4 ratio

Fit \(\omega_{\mathrm{SiH4}}\) with the model above.

### If DCS/SiH4 ratio is nearly constant

Do not claim \(\omega_{\mathrm{SiH4}}\) is identified. Fix it or sweep it.

### If AMAT has no HCl and no chlorine/HCl local estimate

Use fixed \(h_{\mathrm{floor}}\) and report sensitivity. Do not learn a free
HCl term from no-HCl data.

### If HCl sensitivity changes predictions strongly

The next experiment should be a small chlorine/HCl perturbation or a CFD/local
chemistry estimate, not more same-recipe repeats.

### If SiGe model works but SiGeC has only 4-5 conditions

Carry the Si/Ge posterior forward as a prior and add only a strongly
regularized carbon slot. Do not refit the whole SiGe kernel from 4-5 SiGeC
conditions.

---

## What This Notebook Proves

It can prove:

- Tomasini DS1 gives a reasonable SiGe literature prior.
- AMAT spatial scans constrain AMAT-specific SiGe behavior.
- Mixed DCS/SiH4 operation can be represented by an effective Si-source
  activity.
- No-HCl AMAT operation requires either a fixed effective HCl/chlorine floor,
  a sensitivity study, or an independent CFD/local estimate.

It cannot prove:

- that synthetic spatial Tomasini profiles are equivalent to experiments,
- that HCl effects are known when AMAT supplied no HCl,
- that SiH4 and DCS are interchangeable without an identified
  \(\omega_{\mathrm{SiH4}}\),
- that 4-6 conditions support an unrestricted residual neural network.
