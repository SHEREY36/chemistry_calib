# Physics-ML Chemistry Calibration (Tomasini SiGe VPE reproduction + CFD-ACE+ integration)

A Bayesian, physics-first model of Si1-xGex vapor-phase epitaxy kinetics
(growth rate, Ge incorporation, boron doping), calibrated end-to-end against
Tomasini et al. (2010, *Thin Solid Films* 518, S12-S17), with a bolt-on path
to CFD-ACE+ for reactor-specific transport (XYZ's 3D reactor), an
active-learning loop to minimize how many CFD runs that needs, and a
radially-resolved spatial layer (Phase 12) that fits within-wafer
non-uniformity directly from a reactor's own contour/XRD scans, no CFD run
required.

**Start here if you're new to this repo:**
- `build_steps_and_cfd_integration.md` -- the original 11-phase design doc (what gets built, why, in what order)
- `METHODOLOGY.md` -- the math of every sub-model, in plain terms: what ML method each one is (Bayesian MCMC, a small NN, a GP, plain optimization), what data it saw, what's genuinely validated vs. just fit -- sec 14-16 answer the training/validation/usage paradigm questions directly, not just the math
- `VALIDATION_REPORT.md` -- the actual reproduction numbers against the paper's own Tables 1-2 / Figs 2-5, regenerate with `chem-ml report`
- `cfd_cases/sige_dcs/CASE.md` and `cfd_cases/sige_silane/CASE.md` -- two worked CFD-ACE+ input examples (one calibrated, one explicitly flagged as an uncalibrated literature-seed case)

This README is the **command reference** -- everything below assumes you're
in the repo root with the `epitaxy` conda env active.

---

## 1. Training, validation, and usage paradigms -- the short version

**Training: does new wafer data improve the base model, or spin up a
separate one?** Depends whose data it is -- there's no attention mechanism
or embedding table anywhere in this pipeline, so neither analogy is quite
right:
- **Same reactor, same chemistry** (more ASM_Epsilon SiGe wafers): genuinely
  improves the SAME shared posterior over the chemistry parameters
  (`calibrate --pooled` or `warm-start`, sec 2 below).
- **A different reactor** (e.g. XYZ), same precursor chemistry: the shared
  chemistry is frozen and reused as-is; only a small, reactor-specific
  correction is fit on top (`add-reactor`'s scalar offset, or
  `spatial-fit`'s radially-resolved one) -- closer to a frozen-backbone-plus-
  small-adapter pattern than either "improves the base" or "trains something
  independent."
- **A new chemistry/precursor**: intake is now class-aware for `Si`, `Si:X`,
  `SiGe`, `SiGe:X`, `SiGeC`, and `SiGeC:X`. Existing `SiGe:B` and `SiGe:P`
  labels remain accepted aliases. Model-wise, the legacy SiGe GR/Ge/B fits
  stay unchanged; SiGeC adds a separate carbon-incorporation slot that trains
  only when `C_at_pct` is measured.

**Validation: reactor-to-reactor? Any chemistry?** Reactor-to-reactor
transfer IS the core validated claim (freeze the chemistry, test whether a
small correction recovers a NEW reactor's data) -- it applies to any new
reactor running the **same precursor chemistry** (DCS + GeH4 + HCl, +
B2H6). It does **not** automatically cover "any chemistry based on the
precursor set" -- a genuinely different Si/Ge source (e.g. silane instead of
DCS, see `cfd_cases/sige_silane/CASE.md`'s explicit uncalibrated flag) or a
new dopant needs its own separately fit and validated sub-model first.

**Usage, for the epitaxy business unit:** predict/design recipes with a
calibrated confidence interval instead of a bare number; qualify a new
reactor tool on ~15-35 wafers instead of a full 70+ point sweep; diagnose
within-wafer non-uniformity from a single scan (Phase 12, new) without a CFD
run; hand a validated chemistry model to CFD-ACE+ for full 3D reactor design
with a minimized number of expensive runs.

Full mechanics, code paths, and the honest limits of each claim:
**METHODOLOGY.md sec 14-16**.

---

## 2. Environment

```bash
conda activate epitaxy          # Python 3.11.15, already has jax/numpyro/equinox/optax/
                                 # polars/pydantic/pytest/matplotlib installed (see requirements.txt)
cd /path/to/chemistry_calib
pip install -e .                # registers the `chem-ml` console script (pyproject.toml)
```
All commands below can be run either as `chem-ml <command>` or
`python -m chem_ml.cli <command>` -- identical, use whichever is on your PATH.

**Run every command from the repo root.** Paths (`data/raw`, `data/processed`)
are relative to `Config()`'s defaults, not the package location.

---

## 3. Preferred intent-based workflows

### "Does the base model still reproduce Tomasini?" (regression check)
```bash
chem-ml train --target chemistry --strategy pooled --base-only
chem-ml validate --suite all --write-report
```

### "What GR/Ge do I expect at this recipe?"
```bash
chem-ml predict --t-c 725 --hcl-ratio 0.5 --geh4-ratio 0.03
# -> {"GR_nm_min": {"mean":..., "p5":..., "p50":..., "p95":...}, "Ge_at_frac": {...}}
```
`p5`/`p95` are a 90% credible interval (parameter uncertainty only). For the
full posterior-predictive version (parameter uncertainty + observation
noise) at a specific query, with a plot:
```bash
python -c "
from chem_ml.config import Config
from chem_ml.pipeline import run_phase4_calibration
from chem_ml.inference_plots import plot_credible_interval_for_query
p4 = run_phase4_calibration(Config())
print(plot_credible_interval_for_query(p4, Config(), T_C=725, hcl_ratio=0.5, geh4_ratio=0.03, out='figures/my_query.png'))
"
```

### "What recipe gives me a target GR and Ge%?"
```bash
chem-ml inverse --target-gr 40 --target-ge 0.22
```
Returns a recipe AND a confidence flag (`"accepted": true/false`) -- targets
far outside the calibration envelope are refused, not silently extrapolated
(see METHODOLOGY.md sec 10).

### "Which knob matters most, and how sensitive is GR to temperature?"
```bash
chem-ml sensitivity
```
Prints the identifiability eigenspectrum (stiff vs. sloppy parameter
directions) and autodiff sensitivity derivatives (dGR/dT etc.), reproducing
Tomasini's Figs. 4-5.

### "The epitaxy team ran more wafers -- how do I add them?"
Data is **never** thrown away or retrained from scratch; it accumulates.
Put new data in the **standard intake CSV format** (see sec. 5 below), then:

```bash
# Option A: register now, fold in later with a full pooled refit (exact, slower)
chem-ml data add --kind scalar --csv new_wafers.csv --reactor ASM_Epsilon --chem-class SiGe --tag batch_2026_07
chem-ml train --target chemistry --strategy pooled --save-posteriors

# Option B: register AND fold in immediately (approximate, fast -- see METHODOLOGY.md
# / pipeline.run_phase4_warm_start docstring for exactly what "approximate" means here)
chem-ml train --target chemistry --strategy warm-start --csv new_wafers.csv --reactor ASM_Epsilon --chem-class SiGe --tag batch_2026_07
```
Re-run Option A periodically even if you've been using warm-start day to
day -- it's the ground-truth resync.

### "We have SiGeC or doped SiGeC data"
The same scalar intake command is used; the difference is the declared
chemistry class and the columns in the CSV. For undoped SiGeC:

```bash
chem-ml data add --kind scalar \
  --csv data/raw/sigec_reference.csv \
  --reactor XYZ_tool_1 \
  --chem-class SiGeC \
  --tag sigec_reference_2026_07

chem-ml train --target chemistry \
  --strategy pooled \
  --chem-class SiGeC \
  --reference-reactor XYZ_tool_1 \
  --save-posteriors
```

For generic doped SiGeC, use `SiGeC:X`; the actual dopant identity lives in
the row-level `dopant_species` column:

```bash
chem-ml data add --kind scalar \
  --csv data/raw/sigec_doped_reference.csv \
  --reactor XYZ_tool_1 \
  --chem-class SiGeC:X \
  --tag sigecx_reference_2026_07

chem-ml train --target chemistry \
  --strategy pooled \
  --chem-class SiGeC:X \
  --reference-reactor XYZ_tool_1
```

If `C_at_pct` is present, the carbon slot fits
`ln(x_C/(1-x_C))` from temperature, HCl/Si, GeH4/Si, and MMS/Si. If
`C_at_pct` is absent, the run returns a clear skip report instead of
pretending carbon was learned.

**Anti-contamination guarantee** (tested, see `tests/test_data_store.py`):
new data only pools into the fit it structurally matches. Data tagged a
different `--chem-class` (e.g. `SiGe:P`) or a different `--reactor` than the
reference reactor is silently excluded from this fit -- not blended in, not
"a little bit contaminated." A different reactor's data belongs in
`add-reactor` instead (below); a genuinely new chemistry class needs its own
sub-model (see `chem_ml/assembler.py`'s pattern) before any of its data does
anything at all.

### "We're bringing up a new reactor tool -- does the same chemistry apply?"
```bash
chem-ml train --target reactor-transfer --csv xyz_tool_1_wafers.csv --reactor XYZ_tool_1
```
Freezes the reference reactor's chemistry (`theta_chem`) and fits ONLY a
small per-reactor offset (`alpha_HCl`, `alpha_GeH4`, plus a rate scale) on
the new reactor's own data -- this is the actual "does Tomasini's chemistry
transfer" test (Phase 7), not a re-fit. Needs on the order of 15-35
conditions spanning a couple of ratio levels; see METHODOLOGY.md sec 8 for
what the recovered `alpha` values do and don't tell you on their own (short
version: individual alpha values are not uniquely identified from wafer
data alone -- CFD-ACE+, sec below, is what actually resolves that).

### "I have a contour/radial scan from a wafer -- can I use it?"
```bash
chem-ml data add --kind spatial-scan --runs-csv wafer_runs.csv --points-csv wafer_points.csv \
  --reactor XYZ_tool_1 --chem-class SiGe --tag xyz_wafer_042
chem-ml train --target spatial-transfer --tag xyz_wafer_042
```
Phase 12: freezes `theta_chem` exactly like `add-reactor` above, but fits a
**radially-resolved** offset $\delta_r(r)$ (linear in $r/R_w$) directly
against ONE wafer's own contour or XRD line scan, instead of one scalar
$\delta_r$ for the whole reactor. Reports predicted-vs-measured WIWNU
(within-wafer non-uniformity) alongside R², and works from a single scan --
no CFD-ACE+ run needed. See sec. 6 below for the CSV format and
METHODOLOGY.md sec 13 for the math. Real XYZ scan files aren't in hand yet
as of this writing; the ingestion/fit path is verified against synthetic
fixtures (`tests/test_spatial.py`) and a manual smoke test matching the
actual contour sampling pattern.

### "A new precursor/dopant shows up in our process" (e.g. phosphine, disilane)
```bash
chem-ml add-species --name phosphine --formula PH3 --role dopant --family dopant --n-h 3
```
This registers the species (persisted to `data/processed/custom_species.json`)
but it is **inert** until a sub-model is written to read it -- adding a
species never silently perturbs GR/Ge/B (see `chem_ml/registry.py` module
docstring). Most species relevant to this chemistry (silane, disilane,
trisilane, phosphine, HCl, B2H6...) are already pre-registered; check
`chem_ml/registry.py` before adding a duplicate.

### "Hand the calibrated chemistry to CFD-ACE+"
```bash
chem-ml export-mechanism --system dcs --out-dir cfd_export      # calibrated (Tomasini's system)
chem-ml export-mechanism --system silane --out-dir cfd_export_silane  # UNCALIBRATED seed, see cfd_cases/sige_silane/CASE.md
```
Writes `surface_bc_udf.c` (exact calibrated UDF, recommended) and
`elementary_mechanism.json` (fallback elementary form, necessarily lossy --
see the file's own `calibration_status` field for exactly what's data-
pinned vs. literature seed). Worked, filled-in examples with boundary
conditions and expected-response tables: `cfd_cases/sige_dcs/CASE.md` and
`cfd_cases/sige_silane/CASE.md`.

### "Minimize how many CFD-ACE+ runs it takes to pin the reactor-transfer offset"
```bash
chem-ml active-learn --mode seed --n 8 \
  --bounds 873 1053 0.1 0.9 0.01 0.09 5 20 \
  --geometry-id XYZ_3D_v1 --out-dir cfd_runs
```
`--bounds` is `T_lo T_hi HCl_ratio_lo HCl_ratio_hi GeH4_ratio_lo GeH4_ratio_hi P_lo P_hi`.
Writes CFD-ACE+ input specifications for a Sobol space-filling seed set. Run
CFD-ACE+ on each, produce the output CSV contract (`chem_ml/cfd/io.py`), then
continue the loop from a script:
```python
from chem_ml.active_learning import ActiveLearner
from chem_ml.cfd.io import parse_cfd_output
al = ActiveLearner(cfg, bounds, geometry_id="XYZ_3D_v1")
al.ingest([parse_cfd_output(csv_path, cond) for csv_path, cond in your_results])
next_batch = al.select_batch(candidate_pool, k=3)   # GP-variance-guided, cost-weighted, diversity-penalized
```
No CFD-ACE+ license is available in this environment -- the GP surrogate,
Sobol seeding, and batch selection are validated against a synthetic stand-
in (`tests/test_active_learning.py`), not real CFD output. Swapping in real
`CFDResult`s needs no code change.

### "Show me the reproduction plots / uncertainty diagnostics"
```bash
chem-ml plots              # Figs. 2-5 reproduction + posterior-predictive calibration -> figures/
chem-ml inference-plots    # Posterior diagnostics plus presentation plots:
                           # response envelope, extrapolation, process-window map -> figures/inference_*.png
```

---

## 4. Preferred command reference

| Command | Purpose |
|---|---|
| `data add --kind scalar --csv --reactor --chem-class --tag [--mode]` | Register a one-row-per-run scalar CSV for later chemistry pooling or warm-start |
| `data add --kind spatial-scan --runs-csv --points-csv --reactor --chem-class --tag` | Register a wafer scan without mixing spatial points into scalar chemistry data |
| `train --target chemistry --strategy pooled [--chem-class] [--reference-reactor] [--base-only] [--save-posteriors]` | Fit the chemistry model; default includes registered scalar additions, `--base-only` reproduces Tomasini only |
| `train --target chemistry --strategy warm-start --csv --reactor --chem-class --tag [--widen-factor]` | Register and fold in new matching chemistry data immediately |
| `train --target reactor-transfer --csv --reactor` | Fit a per-reactor transfer offset with frozen reference chemistry |
| `train --target spatial-transfer --tag` | Fit a radially-resolved reactor-transfer offset against a registered scan |
| `validate --suite reproduction\|transfer\|spatial\|cfd-contract\|all [--write-report]` | Run validation suites through the workflow facade |
| `add-species --name --formula --role --family [--n-si --n-ge --n-c --n-cl --n-h --produces-hcl]` | Register a new precursor/dopant/carrier (inert until a sub-model reads it) |
| `predict --t-c --hcl-ratio --geh4-ratio [--b2h6-ratio]` | Posterior-predictive GR/Ge at one recipe, with credible intervals |
| `inverse --target-gr --target-ge` | Phase 8: find a recipe for a target, with confidence gating |
| `sensitivity` | Phase 6: identifiability eigenspectrum + Fisher info + sensitivity derivatives |
| `export-mechanism [--system dcs\|silane] [--out-dir]` | Phase 9: export calibrated chemistry as a CFD-ACE+ UDF + elementary mechanism |
| `active-learn --mode seed --n --bounds ... [--geometry-id] [--out-dir]` | Phase 10: GP-guided CFD condition selection |
| `report` | Regenerate `VALIDATION_REPORT.md` + all figures end to end |
| `plots` | Regenerate the Figs. 2-5 reproduction + calibration plots only |
| `inference-plots` | Regenerate the posterior/credible-interval/extrapolation-comparison plots only |

### Legacy aliases

These still work and call the same workflow facade, but the grouped commands
above are the preferred public surface:

| Legacy command | Preferred equivalent |
|---|---|
| `calibrate --pooled` | `train --target chemistry --strategy pooled` |
| `calibrate` | `train --target chemistry --strategy pooled --base-only` |
| `add-data ...` | `data add --kind scalar ...` |
| `warm-start ...` | `train --target chemistry --strategy warm-start ...` |
| `add-reactor ...` | `train --target reactor-transfer ...` |
| `add-wafer-scan ...` | `data add --kind spatial-scan ...` |
| `spatial-fit ...` | `train --target spatial-transfer ...` |
| `report` | `validate --suite all --write-report` |

Every command that needs a fitted posterior (`predict`, `inverse`,
`sensitivity`, `export-mechanism`, `train --target reactor-transfer`,
`train --target spatial-transfer`) re-runs
Phase 4 calibration fresh rather than loading a cached model file -- NUTS on
this dataset size takes single-digit seconds, so there's no serialized
artifact to keep in sync with the data. If the accumulated dataset grows
enough that this becomes slow, that's the point to add caching, not before.

---

## 5. Standard data intake format

New scalar data (`data add --kind scalar` / `train --target chemistry
--strategy warm-start`) uses one stable CSV schema, independent of Tomasini's
own quirky per-appendix columns. Legacy Tomasini-shaped files still work:

```csv
T_C,HCl_over_DCS,GeH4_over_DCS,B2H6_over_DCS,GR_nm_min,Ge_at_pct,B_conc_at_cm3
705,0.45,0.032,0,26.0,20.5,
```
`B2H6_over_DCS`, `GR_nm_min`, `Ge_at_pct`, `B_conc_at_cm3` are optional
(leave blank or omit the column if not measured for that run). Partial
pressures are ratios to the Si-source precursor (`p_i/p_DCS`), matching
Tomasini's own normalization -- absolute pressure calibration is never
needed, only internal consistency per run.

Generalized Si/SiGe/SiGeC files may provide either ratios to the Si source
or raw flows. Supported chemistry classes are `Si`, `Si:X`, `SiGe`,
`SiGe:X`, `SiGeC`, and `SiGeC:X`; `SiGe:B` and `SiGe:P` are compatibility
aliases for old boron/phosphine data.

Recommended SiGeC raw-flow format:

```csv
run_id,T_C,Si_source,Si_source_flow_sccm,GeH4_flow_sccm,HCl_flow_sccm,MMS_flow_sccm,H2_flow_sccm,N2_flow_sccm,XT_flow_H2_minus_N2_sccm,dopant_species,dopant_flow_sccm,growth_time_s,thickness_nm,GR_nm_min,Ge_at_pct,C_at_pct,dopant_conc_at_cm3
```

Ratio-form alternative:

```csv
run_id,T_C,Si_source,HCl_over_Si,GeH4_over_Si,MMS_over_Si,dopant_species,dopant_over_Si,GR_nm_min,Ge_at_pct,C_at_pct,dopant_conc_at_cm3
```

Rules:
- `Si_source` can be `SiH4`, `DCS`, `trisilane`, or another registered Si
  precursor name used consistently by the team.
- If ratio columns are absent, `Si_source_flow_sccm` is required so
  `HCl_flow_sccm`, `GeH4_flow_sccm`, `MMS_flow_sccm`, and
  `dopant_flow_sccm` can be normalized to the Si source.
- `GR_nm_min` is preferred when present; otherwise it is derived from
  `thickness_nm + growth_time_s` or `thickness_A + growth_time_s`.
- `MMS_over_Si` / `MMS_flow_sccm` feeds the SiGeC carbon slot, but only
  `SiGeC` / `SiGeC:X` rows with measured `C_at_pct` train that slot.
- `XT_flow_H2_minus_N2_sccm`, `H2`, `N2`, carbon, and generic dopant fields
  are stored as appended process features. They do not perturb the legacy
  SiGe GR/Ge/B models unless a class-specific model explicitly reads them.

**Record raw growth time on every run, even though it's not a column
here.** GR_nm_min is expected pre-computed, but Tomasini's own DS4 (Appendix
III, no growth time given) is the one hard data gap the entire reproduction
hit -- it demoted an otherwise-usable dataset to Ge%-only. Don't repeat it.

---

## 6. Spatial wafer-scan intake format (Phase 12)

Unlike sec. 5's one-row-per-run format, a wafer scan is one recipe -> N
spatial measurement points, so it uses a **two-file** format
(`data add --kind spatial-scan --runs-csv/--points-csv`) rather than being folded into
`add-data`'s schema -- see `chem_ml/spatial.py`'s module docstring for why
this is a deliberately PARALLEL path, not a bolt-on field (pooling raw
spatial points through the scalar `add-data`/`add-reactor` path unmodified
would silently pseudo-replicate one wafer's own systematic pattern as if it
were N independent chemistry confirmations).

**runs.csv** -- one row per physical wafer run:
```csv
run_id,T_set_C,HCl_over_DCS,GeH4_over_DCS,B2H6_over_DCS,growth_time_s,Stick_1,Stick_2,probe_1
wafer_042,727.0,0.5,0.03,0,600,50.0,48.0,725.0
```
`B2H6_over_DCS`/`growth_time_s` are optional. Any column starting with
`Stick_` (nozzle flow, sccm) or `probe_` (temperature, deg C) is captured
generically as per-run instrumentation metadata -- the number of
nozzles/probes varies by reactor, none are hardcoded.

**points.csv** -- one row per measured location:
```csv
run_id,x_mm,y_mm,GR_nm_min_local,Ge_at_pct_local,thickness_A_local,measurement_source
wafer_042,0,0,41.0,22.0,,SE_contour
wafer_042,147,0,33.0,18.2,,SE_contour
wafer_042,142,-38,33.5,18.4,,SE_contour
```
Pass `y_mm=0` for a pure radial line scan (e.g. an XRD scan from `(0,0)` to
`(145,0)`) -- it's a subset of the general 2D contour case, not a separate
format. `GR_nm_min_local` is optional if `thickness_A_local` + the run's
`growth_time_s` are both present (GR is then derived, same
thickness-over-time convention as sec. 5's DS4 note).

**Nozzle/probe metadata is currently instrumentation-only** -- registered
and carried through, but `spatial-fit` doesn't yet use it to build LOCAL
per-point $(T, p_i)$ features (see METHODOLOGY.md sec 13); the radial trend
is inferred purely from the measured spatial outcome pattern. Feeding
per-nozzle/per-probe data into local features directly is a natural future
extension, not yet built.

---

## 7. Testing

```bash
pytest tests/ -q          # ~80 tests, ~55s (NUTS fits run for real, not mocked)
```
Covers: unit conversion, the anti-contamination guarantee (assembler +
data_store, both), the kappa sign convention, Phase 4's acceptance gates as
actual assertions, cross-reactor recovery, the CFD I/O contract, the GP
active-learning surrogate (against a synthetic stand-in), the plotting code
(checks files are written and non-empty / that the calibration isn't
overconfident, not pixel content), and the Phase 12 spatial layer
(`tests/test_spatial.py`: wafer-scan ingestion/registration, the
radial-profile/WIWNU utilities against hand-computed values, and a synthetic
planted-radial-trend recovery check for `spatial-fit`).

---

## 8. Project layout

```
chem_ml/
  config.py         schema.py         registry.py        assembler.py
  features.py       physics_core.py   residual_nn.py     calibration.py
  identifiability.py reactor_transfer.py inverse_design.py data_store.py
  active_learning.py plots.py         inference_plots.py report.py
  spatial.py         spatial_ingest.py
  pipeline.py        cli.py
  cfd/
    mechanism.py     io.py             transfer.py
tests/
data/
  raw/               # Tomasini appendices (transcribed CSVs)
  processed/         # posteriors/*.nc, additions_manifest.json, spatial_manifest.json, custom_species.json (all gitignored except raw)
cfd_cases/
  sige_dcs/          # calibrated CFD-ACE+ case
  sige_silane/        # uncalibrated-seed CFD-ACE+ case
figures/             # regenerate with `chem-ml plots` / `chem-ml inference-plots`
```
