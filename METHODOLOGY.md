# Methodology and Mathematics: Physics-ML Chemistry Calibration

This document walks through **every sub-model** built in Phases 0–8, in math, so you
can see exactly what kind of estimator each one is, what it structurally can and
cannot do, what data it saw, and what was actually validated versus what was
merely fit. The short answer to "is this black-box regression": no — the primary
model is a 4-parameter interpretable rate law fit by full Bayesian inference
(MCMC), and the one place a neural net appears, it is a small, heavily
regularized *correction* bolted onto that rate law, structurally biased toward
contributing nothing. No Gaussian Process is used anywhere in Phases 0–8 (a GP
surrogate is reserved for the *not-yet-built* Phase 10 active-learning loop over
CFD runs — flagging this since you asked specifically about GP/NN/MCMC).

---

## 1. The architecture in one line

$$
\text{Observable} = A\big(B(\text{setpoint}, \text{geometry})\big)
$$

- $A$ = **intrinsic chemistry** — reactor-independent rate law. Fit once, on
  Tomasini's data. This document is entirely about $A$.
- $B$ = **reactor transport** — setpoint $\to$ local surface conditions.
  Reactor-*specific*. Phases 1–8 approximate $B$ as a small fitted offset
  $\delta_r$ (Phase 7); Phase 9+ (not started) would replace that offset with a
  CFD-ACE+ solve.

Everything below is $A$, plus the offset mechanism used to test whether $A$
transfers to a reactor it never saw data from.

---

## 2. What came from Tomasini — and what that means for the AMAT ask

Tomasini's paper supplied four datasets, each playing a **different structural
role**. This mapping matters because it tells you exactly what shape of data to
request from the AMAT epitaxy team.

| Dataset | Reactor | N | T range | Pressure | Precursors | Measured | Role in this pipeline |
|---|---|---|---|---|---|---|---|
| **DS1** | ASM Epsilon | 70 | 605–765°C (wide, 9 levels) | 10 Torr | DCS, GeH₄, HCl | GR (nm/min), Ge (at.%) | **Fits $\theta_{\text{chem}}$** — the entire chemistry model comes from this one table |
| **DS2** | ASM Epsilon | 18 (GR) + 11 ([B]) | 760°C (fixed) | 10 Torr | + B₂H₆ | GR, Ge%, [B] | Fits the boron sub-model only |
| **DS3** | Hartmann (different reactor) | 35 | 750°C (fixed) | **20 Torr** | DCS, GeH₄, HCl | GR, Ge% | **Held out entirely from Phase 4.** Used only in Phase 7 to test transfer |
| **DS4** | Tan (different reactor) | 18 | 740–760°C | 5–10 Torr | + B₂H₆ (trace) | Ge%, thickness (**no growth time** $\Rightarrow$ no GR) | Held out; Ge/Si transfer test only |

**Why this shape of data matters:** DS1 alone is what makes the chemistry model
identifiable at all — it is the only table that varies **both** temperature
(9 levels) **and** both precursor ratios independently. DS3/DS4 each vary the
ratios but hold temperature essentially fixed, which is enough to test whether
$\theta_{\text{chem}}$ *transfers*, but would not have been enough to *fit* it in
the first place (Phase 7 shows exactly why: with T fixed, a temperature-offset
parameter becomes mathematically indistinguishable from a scale factor — see
§8).

### What to request from AMAT, in this language

1. **One "DS1-equivalent" sweep on a single, well-characterized reactor**: at
   least 6–9 distinct temperatures spanning your real operating window, crossed
   with a range of HCl/DCS and GeH₄/DCS ratios (ideally a few ratio levels at
   each temperature, not just one). This is what identifies $\theta_{\text{chem}}$
   — without it, nothing below is fittable.
2. **Every run must record growth time**, not just final thickness. DS4's
   absence of this is the one hard blocker in the whole reproduction — it
   demoted an entire dataset from "fits GR" to "Ge% only." This is the single
   cheapest thing to fix in a new data request.
3. **A smaller confirmatory set on the actual target (AMAT) reactor** — DS3/DS4
   scale (20–35 runs, a couple of ratio levels, doesn't need to span T as
   widely) is enough to fit the reactor-offset $\delta_r$ in §8, *provided*
   (1) already exists from a reference reactor.
4. If boron (or any other dopant) is a program interest: a DS2-shaped table
   (a modest factorial in HCl/GeH₄/dopant ratio at one fixed T) — note DS2's
   own boron table is only 11 points, which is already the thin edge of what's
   fittable (§12 flags this).
5. Partial pressures **or** raw flows + total pressure are equally usable — the
   pipeline only ever consumes the *ratio* $p_i/p_{\text{DCS}}$ (§3), so absolute
   calibration isn't required, just internal consistency per run.

---

## 3. Canonical features — a physically-motivated map, not a learned embedding

Every row (one growth condition) is reduced to four numbers. This transform is
**fixed by chemistry, not learned**:

$$
p_{\text{DCS}} := 1 \quad \text{(normalization; Tomasini's own convention, Eq. 6 of the paper)}
$$

$$
\mathbf{x} = \Big[\ \tfrac{1}{T},\ \ \ln\!\big(p_{\text{HCl}}/p_{\text{DCS}}\big),\ \ \ln\!\big(p_{\text{GeH}_4}/p_{\text{DCS}}\big),\ \ \ln\!\big(p_{\text{B}_2\text{H}_6}/p_{\text{DCS}}\big)\ \Big]
$$

The choice of $1/T$ and $\ln(\text{ratio})$ is not a feature-engineering
convenience — it is *forced* by the assumed rate law (§4): a power law in
pressures and an Arrhenius form in temperature become **linear** in exactly
these coordinates. That linearity is what makes the model interpretable and
identifiable with 70 data points; a generic ML model would need far more data
to discover this structure on its own.

$1/T$ is standardized (mean/std of the *training* reactor's own temperatures)
purely for MCMC numerical conditioning:

$$
\widehat{\tfrac1T} = \frac{1/T - \mu}{\sigma}, \qquad \mu, \sigma \text{ computed once from DS1}
$$

---

## 4. Sub-model 1 — Growth rate (GR): the core rate law

**Inductive bias:** growth rate is a power function of precursor partial
pressures with an Arrhenius temperature dependence — the standard form for
heterogeneous, surface-reaction-limited kinetics (this is literally what
Tomasini's Eq. 3, and Weller 1956 before it, assert; we did not invent this
functional form, we inherited and fit it).

$$
\mathrm{GR} = K_{\mathrm{GR}}\, e^{\kappa_{\mathrm{GR}}/T} \left(\frac{p_{\mathrm{HCl}}}{p_{\mathrm{DCS}}}\right)^{\gamma_{\mathrm{HCl}}} \left(\frac{p_{\mathrm{GeH_4}}}{p_{\mathrm{DCS}}}\right)^{\gamma_{\mathrm{GeH_4}}}
$$

which is log-linear in the features from §3:

$$
\ln(\mathrm{GR}) = \ln K_{\mathrm{GR}} + \kappa_{\mathrm{GR}}\cdot\widehat{\tfrac1T} + \gamma_{\mathrm{HCl}}\ln\!\left(\frac{p_{\mathrm{HCl}}}{p_{\mathrm{DCS}}}\right) + \gamma_{\mathrm{GeH_4}}\ln\!\left(\frac{p_{\mathrm{GeH_4}}}{p_{\mathrm{DCS}}}\right)
$$

**Four parameters, each physically named**: $\ln K_{\mathrm{GR}}$ (scale),
$\kappa_{\mathrm{GR}}$ (temperature sensitivity, $= -E_a/R$ in Arrhenius
language), $\gamma_{\mathrm{HCl}}, \gamma_{\mathrm{GeH_4}}$ (reaction orders).
Compare this to a black-box regressor, which would need hundreds of weights
with no such correspondence to activation energy or reaction order.

### Bayesian calibration (the actual "training")

This is **not** least-squares point-fitting. It is full Bayesian inference —
the ML method here is **MCMC via NUTS** (No-U-Turn Sampler, a self-tuning
variant of Hamiltonian Monte Carlo), producing a full posterior *distribution*
over the 4 parameters, not a point estimate.

**Priors** (weakly informative, centered on the paper's own reported values —
so the model can disagree with Tomasini if the data says so, but starts near
his answer):

$$
\ln K_{\mathrm{GR}} \sim \mathcal{N}(0, 10^2), \quad
\kappa_{\mathrm{GR}} \sim \mathcal{N}(-24507,\ 5000^2)
$$
$$
\gamma_{\mathrm{HCl}} \sim \mathcal{N}(-0.7,\ 0.3^2), \quad
\gamma_{\mathrm{GeH_4}} \sim \mathcal{N}(1.3,\ 0.3^2), \quad
\sigma \sim \text{HalfNormal}(0.5)
$$

**Likelihood** (Gaussian noise in log-space — i.e. the model assumes
multiplicative, not additive, measurement noise on GR, which is standard for a
positive quantity spanning two orders of magnitude):

$$
\ln(\mathrm{GR}_i^{\text{obs}}) \sim \mathcal{N}\!\big(\ln(\mathrm{GR})_i^{\text{model}},\ \sigma^2\big), \qquad i = 1,\dots,70
$$

**Posterior** (Bayes' rule — this is the actual target of inference):

$$
p(\theta \mid \mathcal{D}) \ \propto\ p(\mathcal{D}\mid\theta)\, p(\theta), \qquad \theta = (\ln K_{\mathrm{GR}}, \kappa_{\mathrm{GR}}, \gamma_{\mathrm{HCl}}, \gamma_{\mathrm{GeH_4}})
$$

**How NUTS actually samples this** (why it's not "just an optimizer"): HMC
augments $\theta$ with an auxiliary momentum $p \sim \mathcal{N}(0, M)$ and
simulates Hamiltonian dynamics

$$
H(\theta, p) = -\log p(\theta\mid\mathcal D) + \tfrac12 p^\top M^{-1} p
$$

via leapfrog integration (a symplectic, volume-preserving numerical scheme),
using the **exact gradient** $\nabla_\theta \log p(\theta\mid\mathcal D)$ —
computed by JAX autodiff through the log-linear model above, not by finite
differences. This lets each proposal move far through parameter space with a
high acceptance probability (unlike random-walk Metropolis). NUTS removes the
one manual knob HMC would otherwise need (trajectory length) by recursively
doubling the simulated path until it starts to double back on itself.

**Result** (4 chains $\times$ 2000 samples, R-hat $<1.001$, ESS $>3000$, zero
divergences — i.e. the sampler actually converged, not just ran):

| Parameter | Posterior mean | Paper | Match |
|---|---|---|---|
| $\kappa_{\mathrm{GR}}$ | $-24{,}483$ K | $-24{,}507$ K | within 0.1% |
| $\gamma_{\mathrm{HCl}}$ | $-0.710$ | $-0.7$ | |
| $\gamma_{\mathrm{GeH_4}}$ | $1.310$ | $1.3$ | |
| $R^2$ (real-unit parity) | 0.996 | 0.985 | |

**Data**: trained on **all 70 rows of DS1** — no train/test split within DS1
(this matches how Tomasini reports his own $R^2$: as a full-data fit). The
genuine out-of-sample test is Phase 7 (§8), on reactors this posterior never saw.

---

## 5. Sub-model 2 — Ge/Si ratio: same rate-law family, one sign inversion worth understanding

Identical structure, target is the Ge/Si atomic ratio $x/(1-x)$:

$$
\ln\!\left(\frac{x}{1-x}\right) = \ln K_{\mathrm{Ge}} + \kappa_{\mathrm{Ge}}\cdot\widehat{\tfrac1T} + \Delta\gamma_{\mathrm{HCl}}\ln\!\left(\frac{p_{\mathrm{HCl}}}{p_{\mathrm{DCS}}}\right) + \Delta\gamma_{\mathrm{GeH_4}}\ln\!\left(\frac{p_{\mathrm{GeH_4}}}{p_{\mathrm{DCS}}}\right)
$$

**Why $\kappa_{\mathrm{Ge}}$ has the *opposite* sign from $\kappa_{\mathrm{GR}}$,
mathematically:** for any $y = e^{\kappa/T}(\cdots)$,

$$
\frac{dy}{dT} = -\frac{\kappa}{T^2}\,y \quad\Longrightarrow\quad \mathrm{sign}\left(\frac{dy}{dT}\right) = \mathrm{sign}(-\kappa)
$$

GR *rises* with $T$ $\Rightarrow \kappa_{\mathrm{GR}}<0$. Ge fraction *falls*
with $T$ (verified directly on DS1: $\approx 33\%$ Ge at 605°C vs.
$\approx21\%$ at 765°C, same GeH₄/DCS) $\Rightarrow \kappa_{\mathrm{Ge}}$ **must
be positive** under this identical functional form — a same-signed prior,
copied from $\kappa_{\mathrm{GR}}$ by analogy, was the one real bug this build
caught (fixed to $\kappa_{\mathrm{Ge}} \sim \mathcal N(+4319, 3000^2)$). This is
exactly the kind of error that's invisible if you only look at $R^2$ and never
re-derive the sign from the functional form — which is the point of writing
the math out like this.

**Result**: $R^2 = 0.987$ (paper: 0.988). **Data**: same 70 DS1 rows as §4 (a
separate NUTS run — GR and Ge/Si are fit independently, not jointly).

---

## 6. Sub-model 3 — Boron incorporation: smallest dataset, honest caveat

$$
\ln\!\left(\frac{[\mathrm B]}{[\mathrm{Si}]}\right) = \ln K_{\mathrm B} + \beta_{\mathrm{HCl}}\ln\!\left(\frac{p_{\mathrm{HCl}}}{p_{\mathrm{DCS}}}\right) + \beta_{\mathrm{GeH_4}}\ln\!\left(\frac{p_{\mathrm{GeH_4}}}{p_{\mathrm{DCS}}}\right) + \beta_{\mathrm{B_2H_6}}\ln\!\left(\frac{p_{\mathrm{B_2H_6}}}{p_{\mathrm{DCS}}}\right)
$$

No temperature term (DS2's boron table is isothermal at 760°C, so $\kappa_B$
isn't identifiable — it's simply omitted, not fit-and-ignored).

**Result**: $\beta_{\mathrm{B_2H_6}} = 0.780$ (paper: $\sim 0.8$), $R^2=0.994$.

**Data**: **11 rows** — the same NUTS/MCMC machinery as §4–5, but this is the
thinnest dataset in the whole pipeline (4 free parameters $+ \sigma$ fit to 11
points). The posterior is wide; treat $\beta_{\mathrm{B_2H_6}}$ as indicative,
not tightly pinned. If boron control matters to the AMAT program, this is the
first sub-model worth re-fitting on more data (§2, point 4).

---

## 7. Diagnostic layer — identifiability (not a trained model; linear algebra on the posterior)

Two complementary checks on how well-constrained the 4 GR parameters are,
computed *from* the Phase 4 posterior, not by fitting anything new:

**Posterior covariance eigen-decomposition** (a "sloppy model" analysis):

$$
\Sigma = \mathrm{Cov}(\theta \mid \mathcal D) \text{ (from MCMC samples)}, \qquad \Sigma\, v_i = \lambda_i v_i
$$

Small $\lambda_i$ = *stiff* direction (tightly data-constrained); large
$\lambda_i$ = *sloppy* direction (weakly constrained, could shift a lot without
hurting fit). Result: eigenvalues span $2\times10^{-5}$ to $2.5\times10^{-2}$ —
over 1000$\times$ — with $\ln K_{\mathrm{GR}}$ stiffest (it's pinned by all 70
points at once) and $\gamma_{\mathrm{GeH_4}}$ sloppiest.

**Fisher information** (a local, likelihood-curvature cross-check on the same
question, via exact autodiff Jacobians, not the MCMC samples):

$$
I(\theta) = \frac{1}{\sigma^2}\, J^\top J, \qquad J = \frac{\partial \ln(\mathrm{GR})}{\partial \theta}\Big|_{X_{\text{DS1}}}
$$

**Sensitivity derivatives** (autodiff again, not finite differences —
$\partial \mathrm{GR}/\partial T$ etc. fall out of the same computational graph
used for MCMC gradients):

$$
\frac{\partial \mathrm{GR}}{\partial T}\bigg|_{750^\circ\mathrm C} = 2.22\ \text{nm/min/K} \qquad (\text{paper: } 1\text{–}2\ \text{nm/min/K})
$$

---

## 8. Sub-model 4 — Reactor transfer: frozen chemistry + a 3–4 parameter offset

**Inductive bias — this is the whole point of Phase 7**: $\theta_{\text{chem}}$
(all 4+4 GR/Ge parameters above) is **frozen** at its DS1 posterior mean and
*not* re-fit. Only a small offset $\delta_r$ is estimated per new reactor:

$$
\ln(\mathrm{GR})_r = \ln \eta_{\mathrm{GR},r} + f_{\mathrm{GR}}\big(\theta_{\text{chem}}^{\text{frozen}};\ X_{\text{eff}}\big), \qquad
X_{\text{eff}} = X + \big[0,\ \ln\alpha_{\mathrm{HCl},r},\ \ln\alpha_{\mathrm{GeH_4},r},\ 0\big]
$$

i.e. the reactor's own precursor-ratio delivery can be rescaled
($\alpha_{i,r}$) and its overall rate can be rescaled ($\eta_r$), but the
*reaction orders and activation energy are not allowed to change* between
reactors. This is a strong, falsifiable structural claim, and it's exactly
what Phase 7 tests.

**Why $\Delta T_r$ (a temperature offset) is dropped, not estimated** — a clean
identifiability argument, not a modeling shortcut: DS3 is measured at a
*single* fixed $T=750^\circ\mathrm C$. For one fixed $T$,

$$
\kappa_{\mathrm{GR}} \cdot \widehat{\tfrac{1}{T+\Delta T_r}} = \text{(some constant, the same for all 35 rows)}
$$

which is **perfectly collinear** with $\ln\eta_{\mathrm{GR},r}$ (also a
constant added to every row) — the two parameters cannot be told apart from
this data, at all, by any estimator. $\eta_r$ absorbs whatever true
temperature offset exists; this is a limitation of single-temperature
validation data, not of the model.

**ML method**: NUTS/MCMC again (same machinery as §4), but now with
$\theta_{\text{chem}}$ held as a **constant** in the log-likelihood rather than
a variable — only $(\ln\alpha_{\mathrm{HCl}}, \ln\alpha_{\mathrm{GeH_4}},
\ln\eta_{\mathrm{GR}}, \ln\eta_{\mathrm{Ge}})$ are sampled for DS3 (4 params);
$(\ln\alpha_{\mathrm{HCl}}, \ln\alpha_{\mathrm{GeH_4}}, \ln\eta_{\mathrm{Ge}})$
for DS4 (3 params, no GR channel — DS4 has no growth time, §2).

### Why this doesn't beat the paper's own per-reactor $R^2$ — and why it can't, by construction

Look again at how $\alpha_{i,r}$ enters the frozen model:

$$
\gamma_{\mathrm{HCl}}\big(\ln(\text{ratio}_{\mathrm{HCl}}) + \ln\alpha_{\mathrm{HCl},r}\big) = \gamma_{\mathrm{HCl}}\ln(\text{ratio}_{\mathrm{HCl}}) + \gamma_{\mathrm{HCl}}\ln\alpha_{\mathrm{HCl},r}
$$

Since $\gamma_{\mathrm{HCl}}$ is frozen, $\alpha_{i,r}$'s entire effect is a
**constant** added to every row's intercept — it can shift the fitted curve up
or down, but it structurally **cannot change its slope**. Tomasini's own DS3
fit (Eq. 11) uses different reaction orders ($\gamma_{\mathrm{HCl}}=-1$,
$\gamma_{\mathrm{GeH_4}}=1$) than DS1 ($-0.7$, $1.3$) — a real slope
difference. Our transfer model is a **nested restriction** of "fit a fresh
power law to DS3 with no constraint from DS1" (which is exactly what the
paper did): a restricted model's $R^2$ on the same data can never exceed the
unrestricted one's. The gap (0.839 vs. paper's 0.844 for DS3 GR; 0.960 vs.
0.994 for DS3 Ge/Si) is the price of testing the falsifiable claim "the
*same* exponents describe a new reactor" — it is not a fitting failure, and
chasing a higher number here would mean quietly re-allowing the exponents to
drift per reactor, which defeats the point of the test.

**The fix, if you want it, is not "tune harder" — it's "decide how much slope
flexibility the data justifies":**
- With only 2 reference reactors (current state): keep exponents frozen, as
  now. There isn't enough cross-reactor data to estimate how much they're
  allowed to vary without just overfitting reactor #3.
- With 3+ reactors: a proper hierarchical extension,
  $\gamma_{i,r} = \gamma_i + \Delta\gamma_{i,r}$, $\Delta\gamma_{i,r}\sim\mathcal N(0,\tau_i^2)$,
  with $\tau_i$ *itself* estimated from the observed spread across reactors —
  this was the original build doc's plan (`sigma_alpha` as a learned
  hyperprior); Phase 7 simplified it away specifically because a spread
  can't be estimated with confidence from only 2 data points.

### What $\alpha_{i,r}$ actually is, and its limits — for the epitaxy conversation

$\alpha_{i,r}$ is a multiplicative correction to the *delivered* ratio
$p_i/p_{\mathrm{DCS}}$ at the wafer, relative to what DS1's reactor would
produce from the same setpoint — a stand-in for gas-phase depletion along the
flow, injector/showerhead mixing, boundary-layer effects, or MFC calibration
differences between tools. It is **fit**, not measured or derived.

Refitting DS3 and inspecting the raw posterior directly shows the limit of
this: with priors $\ln\alpha_{\mathrm{HCl}}, \ln\alpha_{\mathrm{GeH_4}}, \ln\eta_{\mathrm{GR}}, \ln\eta_{\mathrm{Ge}} \sim \mathcal N(0,1)$,

$$
\mathrm{corr}\big(\ln\alpha_{\mathrm{GeH_4}},\ \ln\eta_{\mathrm{Ge}}\big) = -0.97, \qquad
\mathrm{corr}\big(\ln\alpha_{\mathrm{GeH_4}},\ \ln\eta_{\mathrm{GR}}\big) = -0.71
$$

i.e. nearly perfectly degenerate. This is expected, not a bug: each reactor
supplies exactly **2** pieces of intercept information (one net GR shift, one
net Ge/Si shift), but there are **4** raw parameters trying to explain them —
the system is under-determined by 2 degrees of freedom, resolved only by the
prior, not the data. **Practical consequence: don't read individual
$\alpha_{i,r}$ values as "your GeH₄ delivery is off by $X\%$"** — that
specific number is mostly prior, not measurement. What *is* robust is the
combined statement "the frozen chemistry needs this much overall correction
to match your reactor."

This is precisely the gap Phase 9 (CFD-ACE+) exists to close: the build doc
frames CFD as computing $\alpha_{i,r}$ (from species-transport/depletion) and
$\Delta T_r$ (from the thermal solve) **from geometry**, rather than
curve-fitting them blindly from wafer outcomes. If that's available, $\alpha_{i,r}$
stops being an unidentifiable fitted number and becomes an actual physical
prediction you can hand to the epitaxy team.

### Does a downstream user need to specify the reactor?

$\theta_{\text{chem}}$ never takes reactor identity as an input at all (no
one-hot/embedding) — it's genuinely reactor-agnostic once frozen. A user
predicting on the reactor it was fit on needs only $(T, \text{ratios})$.

For a *different* reactor, $\delta_r$ is not inferred zero-shot from "this is
a new reactor" — it must be **fit once**, from a DS3/DS4-scale calibration
run (18–35 conditions) on that specific tool, exactly as Phase 7 did. After
that one-time calibration, a downstream user just calls
`predict(T, ratios)`; reactor identity becomes a config/lookup choice made
once at setup (which $\delta_r$ to add), not something the chemistry math
reasons about per call.

**This is the genuine held-out test in the whole pipeline** — DS3/DS4 rows
never entered Phase 4's likelihood at all:

| Reactor | Observable | $R^2$ (this pipeline) | $R^2$ (paper, same reactor) | $\delta_r$ params |
|---|---|---|---|---|
| DS3 (Hartmann) | GR | 0.839 | 0.844 | 4 |
| DS3 (Hartmann) | Ge/Si | 0.960 | 0.994 | 4 |
| DS4 (Tan) | Ge/Si | 0.888 | $\sim$0.97 (2 sub-models blended) | 3 |

DS3's own paper $R^2$ for GR (0.844) is already far below DS1's (0.985) — that
gap **is** Tomasini's Fig. 1 Regime-I curvature, and reproducing 0.839 means
reproducing that specific limitation, not failing to fit.

---

## 9. Sub-model 5 — Residual neural network: the *only* black-box piece, and why it can't misbehave

This is the one place a generic function approximator is used — worth being
precise about exactly how contained it is.

$$
y_{\log} = \underbrace{f_{\text{phys}}(\theta_{\text{chem}}; x)}_{\text{§4, frozen at posterior mean}} + \underbrace{g_{\mathrm{NN}}(\phi; x_{\text{full}})}_{\text{small MLP, this section}}
$$

**Architecture** — 2 hidden layers, 16 units, $\tanh$ activation, 7-dimensional
input (the 4 log-features of §3 **plus** the 3 raw, non-log ratios — giving the
net a chance to represent curvature the log-linear core structurally cannot),
2-dimensional output (GR residual, Ge residual):

$$
g(x;\phi) = W_3 \tanh\big(W_2 \tanh(W_1 x + b_1) + b_2\big) + b_3
$$

$\approx 434$ trainable weights total — versus 4 physically-named parameters in
the model it's correcting. That ratio is exactly why the regularization below
is load-bearing, not optional.

**ML method**: plain point-estimate supervised learning — **not** Bayesian,
**not** MCMC. Gradient descent via AdamW (Adam + decoupled weight decay) on:

$$
\mathcal L(\phi) = \frac{1}{N}\sum_{i=1}^N \big\| g(x_i;\phi) - r_i \big\|^2 + \lambda\,\|\phi\|^2, \qquad r_i = y_i^{\log} - f_{\text{phys}}(\theta_{\text{chem}};x_i)
$$

The $\lambda\|\phi\|^2$ term (ridge/weight-decay) is the **inductive bias**,
made explicit: it pulls $g_{\mathrm{NN}} \to 0$ everywhere the physics core
already explains the data, so the net can only ever contribute where there's
real, systematic residual structure left over. $\lambda$ was swept
($0.01\to1.0$) against DS1: at $0.01$ the net's output RMS reached $\sim75\%$
of the physics residual's own RMS (i.e. it was fitting noise on 70 points with
434 weights); $\lambda=0.3$ keeps it under 50% while still reducing hybrid RMSE
below physics-only.

**Data**: the **same 70 DS1 rows** §4 was fit on — this is an **in-sample**
residual correction, and $\lambda$ was chosen by watching in-sample RMS, not by
a held-out validation fold (there isn't enough data in DS1 for a clean nested
CV split alongside a 4-chain NUTS fit; this is a known limitation, flagged
rather than hidden — see §12).

---

## 10. Sub-model 6 — Inverse design: constrained optimization, not learning at all

Given a target $(\mathrm{GR}^*, \%\mathrm{Ge}^*)$, find the recipe:

$$
x^\star = \arg\min_x \ \big\| f(\bar\theta; x) - y^\star_{\log} \big\|^2 \ +\ \lambda\, U(x), \qquad x \in [\,x_{\min}, x_{\max}\,]
$$

$$
U(x) = \sum_{d=1}^{2} \mathrm{Var}_{\theta \sim p(\theta\mid\mathcal D)}\Big[f_d(\theta; x)\Big] \quad \text{(per-output-dimension posterior-predictive variance, estimated from 60 posterior draws)}
$$

**ML method**: projected gradient descent (Adam), **not** a trained model at
all — a per-query numerical solve exploiting the physics core's
differentiability (JAX autodiff through $f$). $U(x)$ is what makes it
*uncertainty-aware*: candidate recipes where the posterior samples disagree
with each other are penalized, so the optimizer is discouraged from wandering
into regions the Phase 4 posterior is unsure about. A solution pinned to the
edge of DS1's observed feature range, or with $U(x)$ well above what's typical
for DS1's own points, is **refused** rather than returned.

**"Tested"**: two spot checks, not a systematic evaluation — a target matching
an actual DS1 row (recovered to $<0.1\%$ error, accepted) and a deliberately
extreme target outside DS1's range (refused). This sub-model is a decision
rule built on top of §4's posterior, not something with its own train/test
split.

---

## 11. Summary table

| Sub-model | Inputs (from experiment) | ML method | Free params | Data (N) | Train vs. held-out |
|---|---|---|---|---|---|
| GR rate law (§4) | $T$, $p_{\mathrm{HCl}}/p_{\mathrm{DCS}}$, $p_{\mathrm{GeH_4}}/p_{\mathrm{DCS}}$ | NUTS/MCMC (Bayesian) | 4 + $\sigma$ | 70 (DS1) | full-data fit, no internal split |
| Ge/Si rate law (§5) | same | NUTS/MCMC | 4 + $\sigma$ | 70 (DS1) | same |
| B/Si rate law (§6) | $+\,p_{\mathrm{B_2H_6}}/p_{\mathrm{DCS}}$ | NUTS/MCMC | 4 + $\sigma$ | **11** (DS2) | full-data fit; thin |
| Identifiability (§7) | (Phase 4 posterior) | eigendecomp. + Fisher info | n/a | 70 (DS1) | diagnostic only |
| Reactor transfer (§8) | same features, new reactor's rows | NUTS/MCMC, $\theta_{\text{chem}}$ frozen | 3–4 | 35 (DS3) / 18 (DS4) | **genuinely held out** |
| Residual NN (§9) | features + raw ratios | AdamW / SGD (point estimate) | $\approx$434 | 70 (DS1) | in-sample; $\lambda$ chosen in-sample |
| Inverse design (§10) | target $(\mathrm{GR}^*, \%\mathrm{Ge}^*)$ | projected gradient descent (Adam) | n/a (per-query) | n/a | 2 spot checks |

**No Gaussian Process appears anywhere above.** A GP surrogate is part of the
*unbuilt* Phase 10 (active learning over CFD-ACE+ runs) — if/when that phase
starts, that is where a GP would enter, as a cheap stand-in for expensive 3D
CFD solves, not as a replacement for anything in this table.

---

## 12. Honest accounting: what was genuinely validated vs. merely fit

Being precise about this, since it's the crux of your question:

- **Genuinely out-of-sample**: §8 (DS3, DS4) — these rows never entered any
  likelihood before Phase 7, and the *chemistry* parameters were frozen, not
  re-estimated, when fitting them. This is the strongest evidence in the whole
  pipeline that the model captures real chemistry rather than curve-fitting
  DS1's specific noise.
- **Independent check, same reactor**: §7's Fig. 4 reproduction — the
  $(T, \mathrm{Ge}\%{=}20\%)$ operating points weren't a fit target, but they're
  still on DS1's own reactor, so this is weaker evidence than §8.
- **Full-data fit, no split**: §4, §5, §6 — matches how Tomasini himself
  reports $R^2$ (also full-data), so it's an apples-to-apples reproduction, but
  it means DS1's own $R^2$ numbers shouldn't be read as generalization evidence
  on their own — §8 is what carries that weight.
- **In-sample, regularization chosen in-sample**: §9 (residual NN) — the
  weakest-validated piece, by construction (smallest, most flexible model, on
  the smallest amount of scrutiny). It's kept safe only by the weight-decay
  sweep showing it stays small, not by a held-out fold.
- **Diagnostic, not a claim**: §7 (identifiability) is a statement about
  *this* posterior's shape, not a prediction that could be right or wrong.
- **Ad hoc, not systematic**: §10 (inverse design) — 2 examples, not a test
  suite over a grid of targets.

If the AMAT data (§2) arrives, the single highest-leverage thing to redo with
it is **§8's exact procedure**: freeze $\theta_{\text{chem}}$ from a DS1-scale
sweep, fit only $\delta_r$ on a DS3/DS4-scale AMAT sample, and see if the same
$R^2$ band holds. That is the direct, falsifiable test of "does Tomasini's
chemistry actually describe this precursor system," independent of reactor.
