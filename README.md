# Physics-ML Chemistry Calibration (Tomasini SiGe VPE reproduction + CFD-ACE+ integration)

A Bayesian, physics-first model of Si1-xGex vapor-phase epitaxy kinetics
(growth rate, Ge incorporation, boron doping), calibrated end-to-end against
Tomasini et al. (2010, *Thin Solid Films* 518, S12-S17), with a bolt-on path
to CFD-ACE+ for reactor-specific transport (AMAT's 3D reactor) and an
active-learning loop to minimize how many CFD runs that needs.

**Start here if you're new to this repo:**
- `build_steps_and_cfd_integration.md` -- the original 11-phase design doc (what gets built, why, in what order)
- `METHODOLOGY.md` -- the math of every sub-model, in plain terms: what ML method each one is (Bayesian MCMC, a small NN, a GP, plain optimization), what data it saw, what's genuinely validated vs. just fit
- `VALIDATION_REPORT.md` -- the actual reproduction numbers against the paper's own Tables 1-2 / Figs 2-5, regenerate with `chem-ml report`
- `cfd_cases/sige_dcs/CASE.md` and `cfd_cases/sige_silane/CASE.md` -- two worked CFD-ACE+ input examples (one calibrated, one explicitly flagged as an uncalibrated literature-seed case)

This README is the **command reference** -- everything below assumes you're
in the repo root with the `epitaxy` conda env active.

---

## 0. Environment

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

## 1. Typical workflows

### "Does the base model still reproduce Tomasini?" (regression check)
```bash
chem-ml calibrate                 # Phase 1-4: ingest + NUTS-fit, prints the acceptance report
chem-ml report                    # Phase 1-8 end to end -> VALIDATION_REPORT.md + figures/
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
Put new data in the **standard intake CSV format** (see sec. 3 below), then:

```bash
# Option A: register now, fold in later with a full pooled refit (exact, slower)
chem-ml add-data --csv new_wafers.csv --reactor ASM_Epsilon --chem-class SiGe --tag batch_2026_07
chem-ml calibrate --pooled --save-posteriors

# Option B: register AND fold in immediately (approximate, fast -- see METHODOLOGY.md
# / pipeline.run_phase4_warm_start docstring for exactly what "approximate" means here)
chem-ml warm-start --csv new_wafers.csv --reactor ASM_Epsilon --chem-class SiGe --tag batch_2026_07
```
Re-run Option A periodically even if you've been using warm-start day to
day -- it's the ground-truth resync.

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
chem-ml add-reactor --csv amat_tool_1_wafers.csv --reactor AMAT_tool_1
```
Freezes the reference reactor's chemistry (`theta_chem`) and fits ONLY a
small per-reactor offset (`alpha_HCl`, `alpha_GeH4`, plus a rate scale) on
the new reactor's own data -- this is the actual "does Tomasini's chemistry
transfer" test (Phase 7), not a re-fit. Needs on the order of 15-35
conditions spanning a couple of ratio levels; see METHODOLOGY.md sec 8 for
what the recovered `alpha` values do and don't tell you on their own (short
version: individual alpha values are not uniquely identified from wafer
data alone -- CFD-ACE+, sec below, is what actually resolves that).

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
  --geometry-id AMAT_3D_v1 --out-dir cfd_runs
```
`--bounds` is `T_lo T_hi HCl_ratio_lo HCl_ratio_hi GeH4_ratio_lo GeH4_ratio_hi P_lo P_hi`.
Writes CFD-ACE+ input specifications for a Sobol space-filling seed set. Run
CFD-ACE+ on each, produce the output CSV contract (`chem_ml/cfd/io.py`), then
continue the loop from a script:
```python
from chem_ml.active_learning import ActiveLearner
from chem_ml.cfd.io import parse_cfd_output
al = ActiveLearner(cfg, bounds, geometry_id="AMAT_3D_v1")
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
chem-ml inference-plots    # MCMC posterior pairplot/trace, single-query credible interval,
                           # physics-vs-black-box extrapolation comparison -> figures/inference_*.png
```

---

## 2. Full command reference

| Command | Purpose |
|---|---|
| `calibrate [--pooled] [--save-posteriors]` | Phase 1-4: ingest + NUTS calibration. `--pooled` includes all registered additions (data_store), not just Tomasini. `--save-posteriors` writes `data/processed/posteriors/*.nc` |
| `add-data --csv --reactor --chem-class --tag [--mode]` | Register a new CSV (standard intake format) for later pooling |
| `warm-start --csv --reactor --chem-class --tag [--widen-factor]` | Register AND fold in immediately (approximate, fast) |
| `add-reactor --csv --reactor` | Phase 7: fit a per-reactor offset for a new reactor, chemistry frozen |
| `add-species --name --formula --role --family [--n-si --n-ge --n-c --n-cl --n-h --produces-hcl]` | Register a new precursor/dopant/carrier (inert until a sub-model reads it) |
| `predict --t-c --hcl-ratio --geh4-ratio [--b2h6-ratio]` | Posterior-predictive GR/Ge at one recipe, with credible intervals |
| `inverse --target-gr --target-ge` | Phase 8: find a recipe for a target, with confidence gating |
| `sensitivity` | Phase 6: identifiability eigenspectrum + Fisher info + sensitivity derivatives |
| `export-mechanism [--system dcs\|silane] [--out-dir]` | Phase 9: export calibrated chemistry as a CFD-ACE+ UDF + elementary mechanism |
| `active-learn --mode seed --n --bounds ... [--geometry-id] [--out-dir]` | Phase 10: GP-guided CFD condition selection |
| `report` | Regenerate `VALIDATION_REPORT.md` + all figures end to end |
| `plots` | Regenerate the Figs. 2-5 reproduction + calibration plots only |
| `inference-plots` | Regenerate the posterior/credible-interval/extrapolation-comparison plots only |

Every command that needs a fitted posterior (`predict`, `inverse`,
`sensitivity`, `export-mechanism`, `add-reactor`) re-runs Phase 4
calibration fresh rather than loading a cached model file -- NUTS on this
dataset size takes single-digit seconds, so there's no serialized artifact
to keep in sync with the data. If the accumulated dataset grows enough that
this becomes slow, that's the point to add caching, not before.

---

## 3. Standard data intake format

New data (`add-data` / `warm-start`) uses one stable CSV schema, independent
of Tomasini's own quirky per-appendix columns:

```csv
T_C,HCl_over_DCS,GeH4_over_DCS,B2H6_over_DCS,GR_nm_min,Ge_at_pct,B_conc_at_cm3
705,0.45,0.032,0,26.0,20.5,
```
`B2H6_over_DCS`, `GR_nm_min`, `Ge_at_pct`, `B_conc_at_cm3` are optional
(leave blank or omit the column if not measured for that run). Partial
pressures are ratios to the Si-source precursor (`p_i/p_DCS`), matching
Tomasini's own normalization -- absolute pressure calibration is never
needed, only internal consistency per run.

**Record raw growth time on every run, even though it's not a column
here.** GR_nm_min is expected pre-computed, but Tomasini's own DS4 (Appendix
III, no growth time given) is the one hard data gap the entire reproduction
hit -- it demoted an otherwise-usable dataset to Ge%-only. Don't repeat it.

---

## 4. Testing

```bash
pytest tests/ -q          # ~65-70 tests, ~35-45s (NUTS fits run for real, not mocked)
```
Covers: unit conversion, the anti-contamination guarantee (assembler +
data_store, both), the kappa sign convention, Phase 4's acceptance gates as
actual assertions, cross-reactor recovery, the CFD I/O contract, the GP
active-learning surrogate (against a synthetic stand-in), and the plotting
code (checks files are written and non-empty / that the calibration isn't
overconfident, not pixel content).

---

## 5. Project layout

```
chem_ml/
  config.py         schema.py         registry.py        assembler.py
  features.py       physics_core.py   residual_nn.py     calibration.py
  identifiability.py reactor_transfer.py inverse_design.py data_store.py
  active_learning.py plots.py         inference_plots.py report.py
  pipeline.py        cli.py
  cfd/
    mechanism.py     io.py             transfer.py
tests/
data/
  raw/               # Tomasini appendices (transcribed CSVs)
  processed/         # posteriors/*.nc, additions_manifest.json, custom_species.json (all gitignored except raw)
cfd_cases/
  sige_dcs/          # calibrated CFD-ACE+ case
  sige_silane/        # uncalibrated-seed CFD-ACE+ case
figures/             # regenerate with `chem-ml plots` / `chem-ml inference-plots`
```
