# Methodology: Physics-Kernel + Residual-NN Epitaxy Chemistry Calibration

## 1. Objective

The goal is to train one chamber-agnostic epitaxy chemistry model for a
declared chemistry such as `SiGe`, then export it as a deterministic
`surface_udf.c` for CFD-ACE+.

The model predicts local and wafer-level outputs such as:

$$
\mathbf y =
\left[
\mathrm{GR},
x_{\mathrm{Ge}},
x_{\mathrm{C}},
X/\mathrm{Si}
\right]
$$

Only the slots relevant to the declared chemistry are enabled. For `SiGe`,
the production output vector is:

$$
\mathbf y_{\mathrm{SiGe}}
=
\left[
\mathrm{GR},
x_{\mathrm{Ge}}
\right]
$$

The training paradigm is:

$$
\text{Applied wafer data}
\rightarrow
\text{physics-kernel fit}
\rightarrow
\text{bounded residual-NN fit}
\rightarrow
\text{provisional UDF}
\rightarrow
\text{CFD local-field deembedding}
\rightarrow
\text{refit chamber-agnostic chemistry}
$$

Tomasini is a benchmark for the rate-law family. It is not the production
training dataset.

## 2. Species And Flow Normalization

For inlet MFC flow \(F_i^{\mathrm{MFC}}\), correct any diluted cylinder:

$$
F_i^{\mathrm{eff}}
=
\chi_i^{\mathrm{cyl}}F_i^{\mathrm{MFC}}
$$

The inlet mole fraction is:

$$
y_i^{\mathrm{in}}
=
\frac{F_i^{\mathrm{eff}}}{\sum_k F_k^{\mathrm{eff}}}
$$

and the inlet partial pressure is:

$$
p_i^{\mathrm{in}}
=
y_i^{\mathrm{in}}P_{\mathrm{tot}}
$$

For a generic Si source, define the effective Si-source pressure:

$$
p_{\mathrm{Si}} =
\sum_{s \in \mathcal S_{\mathrm{Si}}}
\nu_{\mathrm{Si},s}p_s
$$

Examples:

$$
\nu_{\mathrm{Si,DCS}}=1,\quad
\nu_{\mathrm{Si,SiH_4}}=1,\quad
\nu_{\mathrm{Si,Si_2H_6}}=2,\quad
\nu_{\mathrm{Si,Si_3H_8}}=3
$$

All chemistry features are normalized ratios:

$$
r_i =
\frac{p_i}{p_{\mathrm{Si}}}
$$

For DCS-based SiGe:

$$
p_{\mathrm{Si}} = p_{\mathrm{DCS}}
$$

## 3. Inference-Time CFD State

At inference time, CFD-ACE+ gives local wall-adjacent state:

$$
\mathbf x_{\mathrm{cell}}
=
\left[
T_w,\,
p_i^{\mathrm{loc}},\,
\rho,\,
\mu,\,
u,\,
D_i,\,
\nabla c_i,\,
\mathrm{geometry}
\right]
$$

The UDF computes:

$$
r_i^{\mathrm{loc}}(\mathbf x_{\mathrm{cell}})
=
\frac{p_i^{\mathrm{loc}}(\mathbf x_{\mathrm{cell}})}
{p_{\mathrm{Si}}^{\mathrm{loc}}(\mathbf x_{\mathrm{cell}})}
$$

The model should ultimately use local wall fields, not inlet setpoints.

## 4. Transport Functionals

Useful hydrodynamic and transport descriptors are:

$$
Re(\mathbf x)
=
\frac{\rho(\mathbf x)u(\mathbf x)L_c}{\mu(\mathbf x)}
$$

$$
Sc_i(\mathbf x)
=
\frac{\mu(\mathbf x)}{\rho(\mathbf x)D_i(\mathbf x)}
$$

$$
Sh_i(\mathbf x)
=
\frac{k_{mt,i}(\mathbf x)L_c}{D_i(\mathbf x)}
$$

The mass-transfer coefficient can be estimated from CFD wall fluxes:

$$
k_{mt,i}(\mathbf x)
=
\frac{|J_i^{mt}(\mathbf x)|}
{|c_i^{bulk}(\mathbf x)-c_i^{wall}(\mathbf x)|+\epsilon}
$$

Local depletion is:

$$
\delta_i(\mathbf x)
=
1 -
\frac{r_i^{loc}(\mathbf x)}{r_i^{in}}
$$

The reaction-vs-transport indicator is:

$$
Da_i(\mathbf x)
=
\frac{k_{\mathrm{surf},i}(\mathbf x)}
{k_{mt,i}(\mathbf x)}
$$

where:

$$
k_{\mathrm{surf},i}(\mathbf x)
\propto
\left|
\frac{\partial \mathrm{GR}}
{\partial r_i^{loc}}
\right|
$$

These descriptors help identify whether observed wafer variation is intrinsic
chemistry or reactor transport.

## 5. Physics Kernel

Define standardized inverse temperature:

$$
\widehat{\tau}
=
\frac{1/T_w-\mu_{1/T}}{\sigma_{1/T}}
$$

For each observable \(j\):

$$
f_{\mathrm{phys},j}
=
\beta_{0,j}
+
\kappa_j\widehat{\tau}
+
\sum_i \gamma_{j,i}\log r_i^{loc}
$$

For growth rate:

$$
\log \mathrm{GR}_{phys}
=
\beta_{0,GR}
+
\kappa_{GR}\widehat{\tau}
+
\sum_i\gamma_{GR,i}\log r_i^{loc}
$$

For Ge fraction:

$$
\log\frac{x_{\mathrm{Ge}}}{1-x_{\mathrm{Ge}}}
=
\beta_{0,Ge}
+
\kappa_{Ge}\widehat{\tau}
+
\sum_i\gamma_{Ge,i}\log r_i^{loc}
$$

For carbon fraction:

$$
\log\frac{x_{\mathrm C}}{1-x_{\mathrm C}}
=
\beta_{0,C}
+
\kappa_C\widehat{\tau}
+
\sum_i\gamma_{C,i}\log r_i^{loc}
$$

For dopant:

$$
\log\frac{X}{\mathrm{Si}}
=
\beta_{0,X}
+
\kappa_X\widehat{\tau}
+
\sum_i\gamma_{X,i}\log r_i^{loc}
$$

## 6. Bounded Residual Neural Network

The residual NN is trained after the physics kernel on the remaining
structured error:

$$
e_j =
\log y_j^{meas} - f_{\mathrm{phys},j}
$$

It is bounded in log space:

$$
g_{\mathrm{NN},j}(\mathbf q;\phi_j)
=
b_j
\tanh
\left(
\frac{h_j(\mathbf q;\phi_j)}{b_j}
\right)
$$

so:

$$
|g_{\mathrm{NN},j}| \le b_j
$$

The NN input is:

$$
\mathbf q =
\left[
\widehat{\tau},
\log r_i^{loc},
r_i^{loc},
\delta_i,
Da_i,
Re,
Sc_i,
Sh_i,
\mathrm{carrier},
\mathrm{loading}
\right]
$$

For an \(L\)-layer MLP:

$$
\mathbf a^{(0)}=\mathbf q
$$

$$
\mathbf a^{(\ell)}
=
\tanh
\left(
W^{(\ell)}\mathbf a^{(\ell-1)}
+
\mathbf b^{(\ell)}
\right)
$$

$$
h_j =
W_j^{(L)}\mathbf a^{(L-1)}
+
b_j^{(L)}
$$

## 7. Final Inference-Time Model

For each enabled observable:

$$
\boxed{
\log y_j(\mathbf x_{\mathrm{cell}})
=
f_{\mathrm{phys},j}(\mathbf x_{\mathrm{cell}};\theta_j)
+
g_{\mathrm{NN},j}(\mathbf q(\mathbf x_{\mathrm{cell}});\phi_j)
}
$$

For growth:

$$
\mathrm{GR}
=
\exp(\log y_{GR})
$$

For Ge:

$$
y_{Ge}
=
\frac{x_{\mathrm{Ge}}}{1-x_{\mathrm{Ge}}}
$$

$$
x_{\mathrm{Ge}}
=
\frac{y_{Ge}}{1+y_{Ge}}
$$

For C:

$$
x_C
=
\frac{y_C}{1+y_C}
$$

For dopant:

$$
\frac{X}{\mathrm{Si}}
=
\exp(\log y_X)
$$

## 8. First Iteration Versus Finished Model

Before CFD local fields are trustworthy, a first-iteration model may still
use inlet ratios and reactor correction terms:

$$
r_i^{eff}
=
\alpha_i r_i^{in}
$$

$$
\log y_j^{(0)}
=
\log \eta_j
+
\beta_{0,j}
+
\kappa_j\widehat{\tau}_{set}
+
\sum_i\gamma_{j,i}
\log(\alpha_i r_i^{in})
+
g_{\mathrm{NN},j}^{(0)}
$$

This is provisional because \(\alpha_i\), \(\eta_j\), and \(g_{\mathrm{NN}}\)
can all absorb reactor geometry effects.

The finished model is:

$$
\boxed{
\log y_j^*
=
\beta_{0,j}
+
\kappa_j\widehat{\tau}^{loc}
+
\sum_i\gamma_{j,i}\log r_i^{loc}
+
g_{\mathrm{NN},j}^*(\mathbf q^{loc})
}
$$

At that point the chemistry sees the CFD-resolved local surface environment
directly.

## 9. Training One SiGe Model

Recommended practical sequence:

1. Collect one rich SiGe DOE on one AMAT reactor.
2. Register scalar wafer data.
3. Register representative spatial scans.
4. Fit physics kernels for GR and Ge.
5. Fit bounded residual NNs on held-out-validated residuals.
6. Export provisional `surface_udf.c`.
7. Run CFD-ACE+ on the same reactor geometry.
8. Ingest CFD local wall fields.
9. Refit chemistry against local surface state.
10. Repeat until parameter and wafer predictions stabilize.
11. Validate with frozen chemistry on another reactor geometry or operating
    configuration if possible.

For a useful SiGe first model, aim for:

$$
50\text{--}100
$$

scalar wafer conditions, plus spatial scans on:

$$
10\text{--}25
$$

representative wafers.

The DOE should vary:

- temperature,
- HCl/Si,
- GeH4/Si,
- pressure if relevant,
- carrier and loading conditions if those are expected to matter.

## 10. Convergence

Bayesian fit convergence:

$$
\hat R < 1.01
$$

$$
ESS > ESS_{\min}
$$

$$
N_{\mathrm{divergences}} = 0
$$

Residual NN convergence:

$$
\mathcal L_{\mathrm{val}}^{phys+NN}
<
\mathcal L_{\mathrm{val}}^{phys}
$$

and:

$$
|g_{\mathrm{NN},j}| \le b_j
$$

Wafer-average convergence:

$$
\bar y
=
\frac{2}{R_w^2}
\int_0^{R_w} r y(r)\,dr
$$

$$
|\bar y_{pred} - \bar y_{meas}| < \epsilon_{\bar y}
$$

WIWNU convergence:

$$
WIWNU(y)
=
\frac{\max_r y(r)-\min_r y(r)}
{2\bar y}
$$

$$
|WIWNU_{pred}-WIWNU_{meas}| < \epsilon_{WIWNU}
$$

Outer deembedding convergence:

$$
\frac{\|\theta^{(k+1)}-\theta^{(k)}\|}
{\|\theta^{(k)}\|+\epsilon}
<
\epsilon_\theta
$$

$$
\frac{\|\phi^{(k+1)}-\phi^{(k)}\|}
{\|\phi^{(k)}\|+\epsilon}
<
\epsilon_\phi
$$

and residuals should stop correlating with reactor/transport variables:

$$
\mathrm{corr}(e_j,\delta_i) \approx 0
$$

$$
\mathrm{corr}(e_j,r/R_w) \approx 0
$$

$$
\mathrm{corr}(e_j,\mathrm{reactor\ ID}) \approx 0
$$

## 11. Offline Versus Online CFD Coupling

Training should remain offline.

CFD-ACE+ calls the UDF online during the solve:

$$
\text{local CFD state}
\rightarrow
\text{surface UDF}
\rightarrow
\text{surface rates/composition}
$$

But parameter updates should not happen inside CFD-ACE+. MCMC and NN training
inside the solver would be expensive, difficult to validate, and harmful to
solver determinism.

The practical outer loop is:

$$
\theta^{(k)},\phi^{(k)}
\rightarrow
UDF^{(k)}
\rightarrow
CFD^{(k)}
\rightarrow
\text{local fields}^{(k)}
\rightarrow
\theta^{(k+1)},\phi^{(k+1)}
$$

Expected number of outer loops:

$$
2\text{--}4
$$

If more than:

$$
5\text{--}6
$$

outer loops are needed, that is a warning that chemistry and transport are
not being separated cleanly or the DOE is under-designed.

## 12. Inference-Time Cost

Let:

$$
J = \text{number of enabled output slots}
$$

$$
S = \text{number of active precursor features}
$$

$$
d = \text{NN input dimension}
$$

$$
H = \text{hidden width}
$$

$$
L = \text{hidden layers}
$$

Physics kernel cost:

$$
\mathcal O(JS)
$$

Residual NN cost:

$$
\mathcal O\left(J[dH+(L-1)H^2+H]\right)
$$

Total UDF cost per reacting wall cell per CFD iteration:

$$
\boxed{
\mathcal O
\left(
JS
+
J[dH+(L-1)H^2+H]
\right)
}
$$

For a small network:

$$
d \approx 13,\quad H=16,\quad L=2,\quad J=2
$$

the cost is only a few thousand floating-point operations per wall cell.
That is small compared with the CFD flow, heat-transfer, and species
transport solve.

## 13. What Is Already Implemented

Implemented in this repo:

- scalar data intake,
- spatial scan registration,
- species registry,
- shared model-package object,
- production data loading separate from Tomasini benchmark data,
- Bayesian physics-kernel calibration,
- bounded residual NN,
- model package JSON export,
- deterministic `surface_udf.c` export,
- CFD wall-profile contract parsing,
- CFD transfer-prior extraction,
- active-learning seed generation,
- benchmark validation tests.

Still needing real Applied data / CFD integration:

- real AMAT scalar DOE files,
- real AMAT spatial scans,
- real CFD-ACE+ wall-profile adapter,
- production deembedding loop over multiple CFD runs,
- frozen-chemistry validation on another reactor geometry.
