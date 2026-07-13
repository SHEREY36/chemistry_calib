# Epitaxy Chemistry Calibrator

Train a chamber-agnostic Si/SiGe/SiGeC surface-chemistry model from Applied
epitaxy data, refine it with CFD-ACE+ local wall fields, and export the final
deterministic `surface_udf.c` used by CFD-ACE+.

This repo is no longer centered on Tomasini training. Tomasini is kept as a
benchmark proving that the borrowed rate-law family is sensible. Production
models are trained from registered Applied experimental data and CFD-local
field outputs.

## What The Model Is

One architecture covers:

- `Si`
- `Si:X`
- `SiGe`
- `SiGe:X`
- `SiGeC`
- `SiGeC:X`

The declared chemistry and precursor species enable output slots:

- `GR`
- `Ge`
- `C`
- `dopant`

Each slot uses:

```text
log(output ratio or rate) = physics kernel + bounded residual neural network
```

For SiGe, the output vector is:

```text
[GR_nm_min, Ge_at_frac]
```

The physics kernel carries Arrhenius temperature dependence and power-law
precursor-ratio dependence. The residual NN captures systematic deviation in
log space and is bounded so it cannot replace the physics model.

## Install On A Work Computer

```bash
git clone <repo-url>
cd chemistry_calib
conda create -n epitaxy python=3.11 -y
conda activate epitaxy
python -m pip install -e .
```

If the work machine already has the `epitaxy` conda environment:

```bash
conda activate epitaxy
python -m pip install -e .
```

Confirm the CLI is visible:

```bash
chem-ml --help
python -m chem_ml.cli --help
```

## Train One SiGe Model

Prepare a scalar CSV with one row per wafer condition:

```csv
run_id,T_C,Si_source,Si_source_flow_sccm,GeH4_flow_sccm,HCl_flow_sccm,H2_flow_sccm,growth_time_s,thickness_nm,GR_nm_min,Ge_at_pct
sige_001,700,DCS,50,1.5,25,10000,600,250,25.0,21.0
```

Register it:

```bash
chem-ml data add --kind scalar \
  --csv data/applied/sige_reference.csv \
  --reactor AMAT_tool_1 \
  --chem-class SiGe \
  --tag sige_reference_v1
```

Train the production model and write the portable model package:

```bash
chem-ml train --target chemistry \
  --strategy pooled \
  --chem-class SiGe \
  --reference-reactor AMAT_tool_1 \
  --species dichlorosilane germane hcl hydrogen \
  --target-deposit SiGe \
  --save-posteriors \
  --save-model-package \
  --model-package-path data/processed/sige_model_package.json
```

Export the CFD-ACE+ UDF:

```bash
chem-ml export-udf \
  --model-json data/processed/sige_model_package.json \
  --out-dir cfd_export/sige
```

The exported files are:

```text
cfd_export/sige/surface_udf.c
cfd_export/sige/model_manifest.json
```

`surface_udf.c` is deterministic. No MCMC, neural-network training, or
posterior sampling runs inside CFD-ACE+.

## Add Spatial Wafer Scans

Spatial scans are core calibration data because they expose transport and
within-wafer effects.

`runs.csv`:

```csv
run_id,T_set_C,HCl_over_DCS,GeH4_over_DCS,growth_time_s,Stick_1,probe_1
wafer_042,700,0.5,0.03,600,50,700
```

`points.csv`:

```csv
run_id,x_mm,y_mm,GR_nm_min_local,Ge_at_pct_local,thickness_A_local,measurement_source
wafer_042,0,0,42.0,21.0,,SE_contour
wafer_042,100,0,38.0,19.8,,SE_contour
```

Register:

```bash
chem-ml data add --kind spatial-scan \
  --runs-csv data/applied/sige_scan_runs.csv \
  --points-csv data/applied/sige_scan_points.csv \
  --reactor AMAT_tool_1 \
  --chem-class SiGe \
  --tag sige_spatial_v1
```

The current scalar training path stores spatial scans separately so a single
wafer scan is not accidentally counted as many independent wafers. Spatial
deembedding and CFD-local refinement use this data in the next stage.

## CFD-ACE+ Deembedding Loop

The practical training loop is:

1. Register Applied scalar wafer data.
2. Register representative spatial scans.
3. Train a first physics-kernel + bounded-residual model.
4. Export provisional `surface_udf.c`.
5. Run CFD-ACE+ on the AMAT reactor geometry with that UDF.
6. Export wall-adjacent local profiles from CFD-ACE+.
7. Parse CFD profiles and extract transport priors.
8. Refit the chemistry using CFD-local surface conditions.
9. Repeat until parameters and wafer predictions stabilize.

Parse one CFD wall-profile CSV:

```bash
chem-ml cfd-ingest \
  --csv cfd_results/run_001_wall_profile.csv \
  --t-set-k 973.15 \
  --p-tot-torr 10 \
  --geometry-id AMAT_3D_v1 \
  --condition-id run_001 \
  --flow DCS=50 \
  --flow HCl=25 \
  --flow GeH4=1.5 \
  --flow H2=10000
```

Expected CFD profile contract:

```csv
r_mm,surface_T_K,p_HCl_over_pDCS,p_GeH4_over_pDCS,p_B2H6_over_pDCS,GR_nm_min,Ge_frac
0,973.1,0.50,0.030,0,42.0,0.21
100,971.5,0.47,0.028,0,38.0,0.20
```

## Minimize CFD Runs

Use active learning to generate an initial CFD design:

```bash
chem-ml active-learn --mode seed --n 8 \
  --bounds 873 1053 0.1 0.9 0.01 0.09 5 20 \
  --geometry-id AMAT_3D_v1 \
  --out-dir cfd_runs
```

Bounds are:

```text
T_lo T_hi HCl_ratio_lo HCl_ratio_hi GeH4_ratio_lo GeH4_ratio_hi P_lo P_hi
```

Run CFD-ACE+ on those cases, ingest the resulting profiles, and use the
`ActiveLearner` Python API for subsequent batches once real CFD results are
available.

## Benchmark Tomasini Separately

Run the literature benchmark only when you explicitly want it:

```bash
chem-ml train --target chemistry --strategy pooled --benchmark-tomasini
chem-ml validate --suite reproduction
chem-ml validate --suite all --write-report
```

This benchmark should not be used as the production Applied chemistry model.

## Main CLI Commands

```bash
chem-ml data add          # register scalar or spatial Applied data
chem-ml train             # train production chemistry or transfer fits
chem-ml export-udf        # export model_package.json to surface_udf.c
chem-ml cfd-ingest        # parse CFD wall-profile CSV and transport priors
chem-ml active-learn      # choose CFD conditions to reduce transport uncertainty
chem-ml validate          # run benchmark, spatial, transfer, or CFD-contract checks
chem-ml add-species       # register a new precursor/dopant/carrier
```

Legacy/demo commands remain available:

```bash
chem-ml calibrate
chem-ml predict
chem-ml inverse
chem-ml sensitivity
chem-ml export-mechanism
chem-ml plots
chem-ml inference-plots
```

## Commit-Readiness Checklist

Before pushing:

```bash
conda activate epitaxy
python -m pytest tests/ -q
python -m chem_ml.cli --help
python -m chem_ml.cli train --help
python -m chem_ml.cli export-udf --help
python -m chem_ml.cli cfd-ingest --help
git status --short
```

The repo is ready for the work-computer clone workflow when tests pass and
the only untracked files are intentional.
