# Build Plan — Physics-ML Chemistry Model, Tomasini Reproduction, and CFD-ACE+ Integration

**Read this first — the mental model:**
The system has **two physically separate layers** that must never be conflated:

- **Layer A — Intrinsic chemistry** (rate constants, reaction orders, activation energies). Reactor-independent. This is what Tomasini captures. Calibrated on data. **No CFD needed.**
- **Layer B — Reactor transport** (setpoint → local surface conditions: local `p_i`, local `T`). Reactor-*specific*. This is what changes between ASM Epsilon and AMAT's 3D reactor. **This is the only place CFD-ACE+ lives.**

Observable = `A( B(setpoint, geometry) )`. Tomasini folds `B` into constants because his data is one reactor. We keep `A` and `B` separate so the chemistry stays portable and `B` is either fit as a lumped offset (`δ_r`) or computed by CFD-ACE+.

The build proceeds A-first (Phases 1–8, no CFD), then bolts on B (Phases 9–11, CFD-ACE+). Do not start CFD work until Phase 8 passes.

---

## PHASE 0 — Repo scaffold & environment

**0.1** Create the repo layout (Claude Code will split the single scaffold `.py` into these):
```
chem_ml/
  config.py            # Config dataclass, paths, hyperparameters
  schema.py            # CanonicalRow, Dataset, validators
  registry.py          # SpeciesRegistry, Species
  assembler.py         # ReactionNetworkAssembler (present-species → active terms)
  features.py          # log-space feature builder
  physics_core.py      # JAX forward model (GR, Ge/Si, B)
  residual_nn.py       # small regularized residual network
  calibration.py       # NumPyro model + MCMC/SVI runners
  identifiability.py   # posterior covariance / Fisher eigenspectrum
  reactor_transfer.py  # δ_r hierarchical random effect (Layer B, data-only)
  inverse_design.py    # differentiable argmin with UQ penalty
  cfd/
    mechanism.py       # export calibrated mechanism → CFD-ACE+ surface-reaction cards
    io.py              # write CFD input decks, parse CFD output
    transfer.py        # extract setpoint→surface map from CFD runs
  active_learning.py   # acquisition loop to pick next CFD/experiment condition
  pipeline.py          # orchestration
  cli.py               # entry points
tests/
data/
  raw/                 # Tomasini appendices, AMAT exports
  processed/           # canonical-schema parquet
notebooks/
```
**0.2** Environment: Python 3.11, `jax`, `jaxlib`, `numpyro`, `flax` (or `equinox`) for the NN, `optax`, `arviz`, `pandas`, `polars`, `scipy`, `scikit-learn`, `pydantic` (schema validation), `pytest`. Pin versions. GPU optional for MCMC.
**0.3** Set `jax.config.update("jax_enable_x64", True)` globally — kinetics fits are ill-conditioned in float32.
**0.4** Central `Config` dataclass: RNG seeds, priors, MCMC settings (chains, warmup, samples), tolerances, file paths. Everything downstream reads from `Config`; nothing hard-codes numbers.

---

## PHASE 1 — Data layer (canonical schema)

**1.1** Define the canonical row (one physical condition):
`{reactor_id, mode(blanket|selective), pattern_density, T_K, p_DCS, p_GeH4, p_HCl, p_B2H6, p_carrier, growth_time_s, GR_nm_min, Ge_at_frac, B_conc, source_dataset}`.
**1.2** Ingest Tomasini appendices → canonical rows:
  - **DS1** (Appendix I, 70 rows, i-SiGe, 10 Torr, 600–765 °C) → primary calibration target.
  - **DS2** (Appendix I, SiGe:B, 15 GR + 11 [B] rows, 760 °C) → boron.
  - **DS3** (Appendix II, Hartmann, 35 rows, 20 Torr, 750 °C) → cross-pressure.
  - **DS4** (Appendix III, Tan, 18 rows, 5–10 Torr, 740–760 °C) → cross-reactor.
  - Tomasini gives **ratios** (HCl/DCS, GeH4/DCS); reconstruct individual `p_i` from ratios × `p_DCS`, with `p_DCS` inferred from total pressure and flow split where needed. Keep the ratios too — they are the model's native features.
  - **Temperature is in °C in the appendix → convert to K** (`T_K = T_C + 273.15`). Every kinetics bug starts here; unit-test it.
**1.3** Validation gates (fail loudly): no negative partial pressures; `GR > 0`; `0 < Ge_frac < 1`; ratios finite; T within a sane window. Reject/flag rows that violate.
**1.4** Split: DS1 → train/val (e.g., stratified by T). DS3/DS4 held out for **cross-reactor** test (they must NOT influence the chemistry fit). DS2 → boron module only.
**1.5** Persist processed data as parquet; log row counts per dataset.

---

## PHASE 2 — Species registry & reaction-network assembler

**2.1** Build the `SpeciesRegistry` (see architecture spec): each species has `role, formula, n_Si/n_Ge/n_C/n_Cl/n_H, family, produces_HCl, default_prior`. Seed it with DCS, GeH4, HCl, B2H6, H2 (carrier). Leave room for SiH4, Si2H6, Si3H8, PH3.
**2.2** Implement `ReactionNetworkAssembler`: given a declared class + species list, it returns the set of **active terms** (which power-law factors and which modules are on). Absent species contribute nothing — enforce structurally.
**2.3** **Correctness test (the anti-contamination guarantee):** predict GR for a SiGe recipe with B2H6 present vs. removed; with B2H6 partial pressure = 0 the B term must vanish and the GR/Ge predictions must be bit-identical to the no-boron network. Add this as a permanent unit test.

---

## PHASE 3 — Physics core (forward model, JAX)

**3.1** Implement the log-linear power-law core. **Parametrize the temperature term as the coefficient of `1/T` directly** to avoid the exp(-Ea/RT) sign ambiguity in Tomasini's table:

$$
\ln(\mathrm{GR}) = \ln K_{\mathrm{GR}} + \kappa_{\mathrm{GR}}\cdot\frac{1}{T} + \gamma_{\text{HCl}}\ln\!\tfrac{p_{\text{HCl}}}{p_{\text{DCS}}} + \gamma_{\text{GeH}_4}\ln\!\tfrac{p_{\text{GeH}_4}}{p_{\text{DCS}}}
$$

$$
\ln\!\frac{x_{\text{Ge}}}{1-x_{\text{Ge}}} = \ln K_{\text{Ge}} + \kappa_{\text{Ge}}\cdot\frac{1}{T} + 0.1\,\ln\!\tfrac{p_{\text{HCl}}}{p_{\text{DCS}}} + 0.51\,\ln\!\tfrac{p_{\text{GeH}_4}}{p_{\text{DCS}}}\ \ (\text{orders as free params})
$$

$$
\ln\!\frac{[\text{B}]}{[\text{Si}]} = \ln K_{\text{B}} + \beta_{\text{HCl}}\ln\!\tfrac{p_{\text{HCl}}}{p_{\text{DCS}}} + \beta_{\text{GeH}_4}\ln\!\tfrac{p_{\text{GeH}_4}}{p_{\text{DCS}}} + \beta_{\text{B}_2\text{H}_6}\ln\!\tfrac{p_{\text{B}_2\text{H}_6}}{p_{\text{DCS}}}
$$

  - **Sign expectations (from the data, verify):** GR *increases* with T ⟹ $\kappa_{\mathrm{GR}} < 0$, expect $\kappa_{\mathrm{GR}} \approx -24{,}507$ K (this equals Tomasini's tabulated "Ea/R"; the negative coefficient of $1/T$ is what makes GR rise with T). Ge fraction *decreases* with T ⟹ $\kappa_{\text{Ge}} > 0$, expect $\kappa_{\text{Ge}} \approx +4{,}319$ K — **note the opposite sign from $\kappa_{\mathrm{GR}}$**, since both observables share the identical $\ln y = \ln K + \kappa/T$ form and move in opposite directions with T. (An earlier draft of this doc had $\kappa_{\text{Ge}} \approx -4{,}319$ K by analogy with $\kappa_{\mathrm{GR}}$; that sign was checked against DS1 during implementation and is wrong — at matched GeH4/DCS≈0.045, Ge% is ≈33% at 605 °C vs ≈21% at 765 °C, i.e. Ge% falls as T rises, which requires $\kappa_{\text{Ge}}>0$ in this parametrization. Tomasini's own tabulated "ΔEa/R" for the Ge/Si ratio equation evidently uses a different internal sign convention than the plain "Ea/R" for GR — plausible since it's a ratio of two Arrhenius terms — which is exactly the ambiguity this κ-parametrization is meant to sidestep.) **Document this in the code**; it is the single most error-prone point.
**3.2** Build the feature matrix in `features.py`: columns `[1, 1/T, ln(pHCl/pDCS), ln(pGeH4/pDCS), ln(pB2H6/pDCS)]`. Standardize `1/T` for conditioning (store the scaler).
**3.3** The core is a pure JAX function `f_phys(params, X) -> y_pred` (vmapped over rows) so it is differentiable end-to-end (needed for inverse design and Fisher info).

---

## PHASE 4 — Bayesian calibration (reproduce Tomasini here)

**4.1** NumPyro model for each observable: priors on `{lnK, κ, orders}` from literature (γ_GeH4 ~ N(1.3, 0.3), γ_HCl ~ N(-0.7, 0.3), κ_GR ~ N(-24507, 5000)), `HalfNormal` on the log-space noise `σ`. Likelihood: `Normal(f_phys, σ)` in log space.
**4.2** Run **NUTS** (4 chains, 1–2k warmup, 2k draws). Diagnostics: R̂ < 1.01, sufficient ESS, no divergences. This is the "MCMC/VI against Tomasini appendix data" line on your slide.
**4.3** **Reproduction acceptance gates (the objective definition of "validated"):**

| Metric | Target |
|---|---|
| DS1 GR parity (posterior-mean vs measured) | R² ≥ 0.98 (paper 0.985), residuals random |
| DS1 Ge% parity | R² ≥ 0.98 (paper 0.988) |
| Temperature coefficient | \|κ_GR\| = 24,507 K within ±10% |
| Reaction orders | γ_HCl = −0.7 ± 0.1, γ_GeH4 = 1.3 ± 0.15 |
| Boron scaling (DS2) | [B] ∝ p_B2H6^0.8 |

**4.4** Save the posterior (`arviz` InferenceData). This posterior is the **prior for every later step** — this is the "yesterday's posterior = tomorrow's prior" mechanism.

---

## PHASE 5 — Residual NN + hybrid

**5.1** Implement `g_NN(x; φ)`: a *small* MLP (e.g., 2 layers × 16 units), input = the same features (+ raw ratios), output = per-observable log-residual. Strong L2/`weight_decay` (via `optax`) + a prior that shrinks it toward 0.
**5.2** Hybrid: `y_pred = f_phys + g_NN`. Fit φ (and optionally refit θ jointly) with the physics prior dominating. Confirm on DS1 that `g_NN` stays small (physics already gives R² ≥ 0.98) and only activates on the **Regime-I curvature** at low `pGeH4/pDCS` that the pure power law misses.
**5.3** Gate `g_NN` by declared class (hard gate) so it never contaminates across chemistries.

---

## PHASE 6 — Identifiability & sensitivity (reproduce Figs 4 & 5)

**6.1** From the posterior, compute the parameter covariance; eigendecompose → **stiff vs sloppy** directions. Rank parameters by data-constrained-ness. Deliverable: the identifiability table.
**6.2** Fisher information `I = Jᵀ Σ⁻¹ J` with `J = ∂f_phys/∂θ` (JAX `jacfwd`) as a cross-check.
**6.3** Sensitivity derivatives via autodiff: `∂GR/∂T`, `∂GR/∂p_i`, `∂x_Ge/∂p_GeH4`. **Reproduce Tomasini Fig. 4/5 targets**: `∂GR/∂T ~ 1–2 nm/min/K` and `∂x_Ge/∂p_GeH4 ~ 0.2 at.%` near 750 °C. These come free from the differentiable model.

---

## PHASE 7 — Reactor transfer block (Layer B, data-only version)

**7.1** Implement the hierarchical `δ_r = {ΔT_r, α_{i,r}, η_r}` random effect: shared `θ_chem`, per-reactor low-dim offsets, partially pooled via hyperpriors.
**7.2** **Cross-reactor validation:** fit `θ_chem` on DS1 (Epsilon); freeze it; fit only `δ_r` for **DS3** (Hartmann, 20 Torr) and **DS4** (Tan, 5–10 Torr). Acceptance: recover their GR/Ge within the published R² band using only the ~3–5 `δ_r` parameters. This proves portability **before** CFD and before AMAT data.

---

## PHASE 8 — Inverse design

**8.1** Implement `x* = argmin_x ‖ŷ(x;θ̄) − y*‖²_W + λ·U(x)` s.t. `x ∈ X_feasible`, using JAX gradients + a constrained optimizer (`optax` + projection, or `jaxopt`). `U(x)` = posterior predictive variance.
**8.2** Deliver the inverse-design notebook: target spec (GR, %Ge, [B]) → recipe + confidence flag. Refuse/flag targets in low-confidence regions.

> **STOP GATE.** Phases 1–8 are the full Tomasini deliverable and require **no CFD-ACE+**. Everything below is Layer B for the real AMAT reactor.

---

## PHASE 9 — CFD-ACE+ integration: where it actually falls

### 9.1 The exact role of CFD-ACE+

CFD-ACE+ solves the coupled **flow + heat transfer + species transport + surface chemistry** in a specific reactor geometry. In this project it does exactly two things:

1. **Computes the reactor transfer map (Layer B) for AMAT's 3D reactor.** Input: inlet flows, total pressure, susceptor/lamp temperature setpoints, 3D geometry/mesh. Output: **local partial pressures and temperature at the wafer surface**, plus their **radial profile across the wafer**. This is the physics behind `δ_r` — instead of fitting `δ_r` blindly from wafers, CFD *computes* it from geometry.
2. **Consumes the calibrated chemistry mechanism** (Phase 9.3) as its surface-reaction module, so its surface boundary condition is *your* chemistry.

That is the whole of it. CFD-ACE+ does **not** reproduce Tomasini, does **not** calibrate rate constants (your ML does that), and is **not** surrogated. It is the geometry-aware translator between setpoints and surface conditions, and the consumer of your mechanism.

### 9.2 What CFD-ACE+ gives your model (the payoff)

- **Setpoint → surface map** for AMAT's reactor → turns AMAT setpoints into the effective conditions the intrinsic chemistry sees, so `θ_chem` stays portable and doesn't have to be re-fit per reactor.
- **Physics-based priors on `δ_r`** (`ΔT_r` from the thermal solution, `α_{i,r}` from depletion along the flow) → fewer AMAT wafers needed to pin Layer B.
- **Within-wafer radial profiles** → Tomasini gives one scalar GR; CFD gives GR(r), enabling **uniformity / loading** modeling that Tomasini cannot.
- **Synthetic training points** in expensive corners of the 3D operating space where running wafers is prohibitive.

### 9.3 The chemistry mechanism CFD-ACE+ needs (surface-reaction module)

CFD-ACE+'s surface chemistry module wants **elementary reactions with rate expressions**, not the lumped power law. Below is the **canonical site-based SiGe:B mechanism** (DCS / GeH4 / HCl / B2H6) to seed the module. Rate form is Arrhenius `k = A·Tᵇ·exp(−Eₐ/RT)` for gas steps and sticking-coefficient / site-fraction form for surface steps. `s` = open surface site; `(s)` = adsorbed; `(b)` = bulk solid.

> **CRITICAL HONESTY NOTE:** the numeric `A, b, Eₐ` below are **representative literature-order values to seed the mechanism**, not final constants. Each must be verified against the cited primary source, and the whole point of your calibration pipeline is that these become the **prior** whose **posterior** is fit to data. Mark every value `status: seed/verify` in the table. Do not ship these to production CFD without the verification pass.

**Gas phase (minimal; RP-CVD @10–20 Torr is surface-limited, so keep sparse):**

| # | Reaction | Rate form | Seed params (verify) | Source class |
|---|---|---|---|---|
| G1 | SiH₂Cl₂ ⇌ SiCl₂ + H₂ | Arrhenius, reversible | Eₐ ≈ 55–75 kcal/mol | Coltrin/Kee chlorosilane; Ho & Melius |
| G2 | GeH₄ ⇌ GeH₂ + H₂ | Arrhenius, reversible | Eₐ ≈ 40–55 kcal/mol | germane pyrolysis lit. |
| G3 | SiCl₂ + H₂ ⇌ SiH₂Cl₂ (reverse of G1) | — | (couples to G1) | — |

**Surface (site-based; the growth-controlling set):**

| # | Reaction | Rate form | Seed params (verify) | Source class |
|---|---|---|---|---|
| S1 | SiH₂Cl₂(g) + 2s → SiCl₂(s) + 2H(s) | dissociative sticking γ | γ ≈ 0.1–1e-2 | Coltrin/Kee; Kleijn reactor models |
| S2 | SiCl₂(s) → Si(b) + 2Cl(s) | Arrhenius surface | Eₐ ≈ 40–60 kcal/mol | chlorosilane surface kinetics |
| S3 | 2H(s) → H₂(g) + 2s | Arrhenius (H₂ desorption, **often rate-limiting at low T**) | Eₐ ≈ 45–58 kcal/mol | Si(100) H₂ TPD lit. |
| S4 | 2Cl(s) → Cl₂(g) + 2s / or HCl route | Arrhenius (Cl/HCl desorption) | Eₐ ≈ 65–80 kcal/mol | Cl/Si desorption lit. |
| S5 | HCl(g) + Si(b) → SiHCl(s)/SiCl₂(g) (etch) | Arrhenius (drives selectivity, **negative GR order**) | Eₐ ≈ 20–40 kcal/mol | SEG selectivity lit. |
| S6 | GeH₄(g) + 2s → Ge(s) + ... + H(s) | dissociative sticking γ | γ ≈ 0.1–1 (Ge sticks readily) | Imai 2008; germane surface |
| S7 | Ge(s) → Ge(b) (incorporation, competes with Si for sites) | Arrhenius | Eₐ low (Ge favored at low T) | Imai 2008; Ge/Si competition |
| S8 | B₂H₆(g) + 2s → 2BH₃(s) → 2B(b) (dopant incorporation) | dissociative sticking + Arrhenius | γ, Eₐ ~ empirical | Tomasini DS2; B-doping lit. |
| S9 | H(s) + Cl(s) → HCl(g) + 2s (alt. Cl removal) | Arrhenius | Eₐ ≈ 50–70 kcal/mol | surface HCl formation |

**Notes for the CFD deck:**
- Enforce **site conservation** `θ_free + θ_H + θ_Cl + θ_Si + θ_Ge = 1` as a hard constraint (element/site balance is what keeps the mechanism physical; the combustion-PINN literature uses the same trick).
- The **selectivity/negative-HCl-order** behavior Tomasini sees emerges from S5 competing with deposition — this is the physical origin of `γ_HCl ≈ −0.7`.
- For the **pilot**, carbon (SiGeC) is **out of scope** → omit MMS/methylsilane steps. When SiGeC comes, add the carbon sub-mechanism (the 15/18-ready set you already audited against Imai 2008 / Danielsson 2002; the genuine unknowns are carbon partitioning pre-exponentials and surface incorporation).
- **Consistency check:** the lumped power-law (`θ_chem`) and this elementary mechanism must agree in the surface-limited regime. Fit the elementary rate constants so the mechanism reproduces the calibrated power-law GR/Ge over Tomasini's window. This is a mechanism-reduction / consistency step, not a second calibration.

### 9.4 CFD I/O interface (code)

**9.4.1** `cfd/mechanism.py`: write the calibrated mechanism (S1–S9, G1–G3 with posterior-mean params) to CFD-ACE+ surface-reaction card format (or the UDF the solver expects).
**9.4.2** `cfd/io.py`: programmatically write CFD input decks for a list of conditions `{T_set, flows, P_tot, geometry_id}`; launch (or stage for HPC); parse outputs → `{GR(r), Ge(r), local p_i(r), local T(r)}`.
**9.4.3** `cfd/transfer.py`: from a CFD run, extract the **setpoint→surface map** `B_r`: fit `α_{i,r}` (surface `p_i` / inlet `p_i`) and `ΔT_r` (surface T − setpoint T) as functions of operating point; feed these as **priors on `δ_r`** into `reactor_transfer.py`.

---

## PHASE 10 — Active Learning to minimize CFD-ACE+ runs

CFD-ACE+ 3D runs are the expensive resource (~15–25 runs budget). Active learning chooses **which** conditions to simulate so a handful of runs pin Layer B.

**10.1 Surrogate.** Build a cheap **GP surrogate** over the CFD input space `x_cfd = {T_set, flow ratios, P_tot}` → outputs of interest (surface `p_i`, `ΔT`, or the resulting `δ_r` parameters). Initialize with a small space-filling seed (Sobol/LHS, ~5–8 points).
**10.2 Fit.** Update the posterior over `δ_r` (and any chemistry params CFD is meant to refine) using the runs done so far.
**10.3 Acquisition.** Score every candidate condition by **expected information gain about the target parameters** — Bayesian **D-optimal** design (maximize `det` of the expected posterior precision) or maximize expected reduction in `δ_r` posterior variance. Add a **cost term** (3D runs are not equal cost) so cheap-but-informative runs are preferred.
**10.4 Batch select.** Pick the top-`k` conditions under a batch quota (greedy with a diversity/repulsion penalty so the batch isn't clustered). This is the same role-quota batch-selection pattern you built before — reuse it.
**10.5 Run & iterate.** Submit the batch to CFD-ACE+ (Phase 9.4), ingest results, GOTO 10.2. **Stop** when `δ_r` posterior uncertainty is below tolerance or the run budget is hit.
**10.6 Report.** Log runs-used vs. a naive grid sweep; target ~50% reduction. This is the "cuts CFD-ACE+ HPC runs by ~50%" claim, now with an auditable number.

**Logistics summary (one line):** *Sobol seed → GP over CFD inputs → D-optimal/EIG acquisition with cost → batch of `k` runs → update → repeat until `δ_r` uncertainty < tol.* Active learning targets **Layer B parameters**, because that's what CFD is there to inform — not the chemistry, which the cheap experimental data already constrains.

---

## PHASE 11 — Commissioning & handover

**11.1** CLI/API: `calibrate`, `predict`, `inverse`, `sensitivity`, `add-reactor`, `add-species`, `export-mechanism`, `active-learn`.
**11.2** Runbook: for each event (new reactor / drift / new precursor / new class) the exact command + how many wafers/CFD runs to feed (`N ≥ 10·d_x`).
**11.3** Tests: the anti-contamination test (2.3), unit-conversion tests (1.2), reproduction gates (4.3), cross-reactor gate (7.2) all in CI.
**11.4** Validation report auto-generated: reproduces the Phase-4/6/7 acceptance tables on every run.

---

## Dependency graph (what blocks what)

```
0 → 1 → 2 → 3 → 4 ──► 5 ──► 6
                4 ──► 7 ──► 8   (STOP GATE: full Tomasini deliverable, no CFD)
                7 ──► 9 (mechanism export needs calibrated θ; transfer needs δ_r block)
                9 ──► 10 (active learning needs CFD I/O + surrogate)
        4,7,9,10 ──► 11
```

**Do 1→8 completely before touching 9.** If the Tomasini reproduction and cross-reactor `δ_r` validation don't pass, no amount of CFD will save the project — and if they do pass, CFD becomes a clean bolt-on that only computes Layer B for the real reactor.
