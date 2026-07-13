"""Command-line entry points for production chemistry calibration.

The primary workflow is:
  data add -> train --save-model-package -> export-udf -> CFD-ACE+.

Tomasini-oriented commands remain as explicit benchmarks/demos and are not
the production Applied-data training path.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from chem_ml.config import Config
from chem_ml.contracts import (
    DataKind,
    RegisterExperimentRequest,
    TrainRequest,
    TrainStrategy,
    TrainTarget,
    ValidateRequest,
    ValidationSuite,
)
from chem_ml.schema import ChemClass, Mode
from chem_ml.workflows import register_experiment, train, validate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("chem_ml.cli")

CHEM_CLASS_CHOICES = ["Si", "Si:X", "SiGe", "SiGe:X", "SiGe:B", "SiGe:P", "SiC", "SiGeC", "SiGeC:X"]


def _print_json(obj) -> None:
    def default(o):
        try:
            return float(o)
        except (TypeError, ValueError):
            return str(o)
    print(json.dumps(obj, indent=2, default=default))


def _public(obj):
    if isinstance(obj, dict):
        return {k: _public(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, list):
        return [_public(v) for v in obj]
    return obj


def _enum(value: str, enum_cls):
    return enum_cls(value.replace("-", "_"))


# ---------------------------------------------------------------------------
def cmd_calibrate(args: argparse.Namespace) -> None:
    cfg = Config()
    result = train(
        cfg,
        TrainRequest(
            target=TrainTarget.CHEMISTRY,
            strategy=TrainStrategy.POOLED,
            include_registered=args.pooled,
            save_posteriors=args.save_posteriors,
            use_benchmark_data=True,
        ),
    )
    _print_json(result["report"])
    if args.save_posteriors:
        log.info("Wrote posteriors to %s", result["posterior_dir"])


def cmd_add_data(args: argparse.Namespace) -> None:
    cfg = Config()
    register_experiment(
        cfg,
        RegisterExperimentRequest(
            kind=DataKind.SCALAR,
            csv_path=args.csv,
            reactor_id=args.reactor,
            chem_class=ChemClass(args.chem_class),
            tag=args.tag,
            mode=Mode(args.mode),
        ),
    )
    print(f"Registered '{args.tag}' ({args.csv}) for reactor={args.reactor}, "
          f"chem_class={args.chem_class}. Run 'calibrate --pooled' to fold it in, "
          f"or 'warm-start' for a fast approximate update.")


def cmd_warm_start(args: argparse.Namespace) -> None:
    cfg = Config()
    updated = train(
        cfg,
        TrainRequest(
            target=TrainTarget.CHEMISTRY,
            strategy=TrainStrategy.WARM_START,
            csv_path=args.csv,
            reactor_id=args.reactor,
            chem_class=ChemClass(args.chem_class),
            mode=Mode(args.mode),
            tag=args.tag,
            widen_factor=args.widen_factor,
        ),
    )
    _print_json(_public(updated))
    print("Warm-start applied. Recommend a full 'calibrate --pooled' periodically as the ground-truth resync.")


def cmd_add_reactor(args: argparse.Namespace) -> None:
    """Phase 7-style: freeze theta_chem from the reference reactor, fit
    ONLY delta_r for a brand-new reactor's data."""
    cfg = Config()
    result = train(
        cfg,
        TrainRequest(
            target=TrainTarget.REACTOR_TRANSFER,
            strategy=TrainStrategy.FROZEN_CHEMISTRY,
            csv_path=args.csv,
            reactor_id=args.reactor,
            chem_class=ChemClass.SIGE,
            mode=Mode.BLANKET,
        ),
    )
    _print_json(result["report"])


def cmd_add_wafer_scan(args: argparse.Namespace) -> None:
    cfg = Config()
    register_experiment(
        cfg,
        RegisterExperimentRequest(
            kind=DataKind.SPATIAL_SCAN,
            runs_csv=args.runs_csv,
            points_csv=args.points_csv,
            reactor_id=args.reactor,
            chem_class=ChemClass(args.chem_class),
            tag=args.tag,
        ),
    )
    print(f"Registered wafer scan '{args.tag}' ({args.runs_csv}, {args.points_csv}) for "
          f"reactor={args.reactor}. Run 'spatial-fit --tag {args.tag}' to fit the radially-"
          f"resolved reactor-transfer offset against it.")


def cmd_spatial_fit(args: argparse.Namespace) -> None:
    cfg = Config()
    result = train(
        cfg,
        TrainRequest(
            target=TrainTarget.SPATIAL_TRANSFER,
            strategy=TrainStrategy.FROZEN_CHEMISTRY,
            tag=args.tag,
        ),
    )
    reports = result["reports"]
    _print_json(reports if len(reports) > 1 else reports[0])


def cmd_data_add(args: argparse.Namespace) -> None:
    cfg = Config()
    kind = _enum(args.kind, DataKind)
    if kind == DataKind.SCALAR:
        result = register_experiment(
            cfg,
            RegisterExperimentRequest(
                kind=kind,
                csv_path=args.csv,
                reactor_id=args.reactor,
                chem_class=ChemClass(args.chem_class),
                mode=Mode(args.mode),
                tag=args.tag,
            ),
        )
    elif kind == DataKind.SPATIAL_SCAN:
        result = register_experiment(
            cfg,
            RegisterExperimentRequest(
                kind=kind,
                runs_csv=args.runs_csv,
                points_csv=args.points_csv,
                reactor_id=args.reactor,
                chem_class=ChemClass(args.chem_class),
                tag=args.tag,
            ),
        )
    else:
        raise ValueError("CLI data add currently supports scalar and spatial-scan inputs")
    _print_json(result)


def cmd_train(args: argparse.Namespace) -> None:
    cfg = Config()
    target = _enum(args.target, TrainTarget)
    if args.strategy:
        strategy = _enum(args.strategy, TrainStrategy)
    elif target == TrainTarget.CHEMISTRY:
        strategy = TrainStrategy.POOLED
    else:
        strategy = TrainStrategy.FROZEN_CHEMISTRY
    result = train(
        cfg,
        TrainRequest(
            target=target,
            strategy=strategy,
            csv_path=args.csv,
            reactor_id=args.reactor or "",
            reference_reactor=args.reference_reactor,
            chem_class=ChemClass(args.chem_class),
            mode=Mode(args.mode),
            tag=args.tag or "",
            widen_factor=args.widen_factor,
            include_registered=not args.base_only,
            save_posteriors=args.save_posteriors,
            use_benchmark_data=args.benchmark_tomasini,
            species_names=tuple(args.species or ()),
            target_deposit=args.target_deposit or "",
            save_model_package=args.save_model_package,
            model_package_path=args.model_package_path,
            fit_residual_nn=not args.no_residual_nn,
            residual_steps=args.residual_steps,
        ),
    )
    public = _public(result)
    if "reports" in public and len(public["reports"]) == 1:
        public["report"] = public.pop("reports")[0]
    _print_json(public)


def cmd_validate(args: argparse.Namespace) -> None:
    cfg = Config()
    result = validate(
        cfg,
        ValidateRequest(
            suite=_enum(args.suite, ValidationSuite),
            tag=args.tag or "",
            write_report=args.write_report,
            report_path=args.report_path,
        ),
    )
    _print_json(_public(result))


def cmd_add_species(args: argparse.Namespace) -> None:
    from chem_ml.registry import Role, Species, save_custom_species
    from pathlib import Path

    cfg = Config()
    path = Path(cfg.data_processed) / "custom_species.json"
    sp = Species(canonical_name=args.name, formula=args.formula, role=Role(args.role), family=args.family,
                n_Si=args.n_si, n_Ge=args.n_ge, n_C=args.n_c, n_Cl=args.n_cl, n_H=args.n_h,
                produces_HCl=args.produces_hcl)
    save_custom_species(path, [sp])
    print(f"Registered species '{args.name}' ({args.formula}) to {path}. "
          f"It is INERT until a sub-model reads it -- see registry.py module docstring.")


def cmd_predict(args: argparse.Namespace) -> None:
    from chem_ml.pipeline import run_phase4_calibration
    from chem_ml.calibration import mu_draws
    from chem_ml.physics_core import gr_logmodel, ge_logmodel
    import jax.numpy as jnp
    import numpy as np

    cfg = Config()
    result = run_phase4_calibration(cfg)
    mu, sd = result["features_ds1"].invT_scaler
    T_K = args.t_c + 273.15
    invT_std = (1.0 / T_K - mu) / sd
    b2h6 = args.b2h6_ratio or 1e-12
    X = jnp.array([[invT_std, jnp.log(args.hcl_ratio), jnp.log(args.geh4_ratio), jnp.log(b2h6)]])

    gr_draws = np.exp(np.asarray(mu_draws(gr_logmodel, result["mcmc_gr"], X,
                                         ["lnK_GR", "kappa_GR", "gamma_HCl", "gamma_GeH4"])))[:, 0]
    ge_ratio_draws = np.exp(np.asarray(mu_draws(ge_logmodel, result["mcmc_ge"], X,
                                               ["lnK_Ge", "kappa_Ge", "dgamma_HCl", "dgamma_GeH4"])))[:, 0]
    ge_draws = ge_ratio_draws / (1 + ge_ratio_draws)

    def summarize(draws):
        return {"mean": float(np.mean(draws)), "p5": float(np.percentile(draws, 5)),
               "p50": float(np.percentile(draws, 50)), "p95": float(np.percentile(draws, 95))}

    _print_json({
        "recipe": {"T_C": args.t_c, "HCl_over_DCS": args.hcl_ratio, "GeH4_over_DCS": args.geh4_ratio},
        "GR_nm_min": summarize(gr_draws), "Ge_at_frac": summarize(ge_draws),
        "note": "p5/p95 are the 90% posterior credible interval -- parameter uncertainty only "
               "(mu_draws, no observation noise); see chem_ml/plots.py for the full posterior-"
               "predictive version including observation noise.",
    })


def cmd_inverse(args: argparse.Namespace) -> None:
    from chem_ml.pipeline import run_phase4_calibration, run_phase8_inverse_design

    cfg = Config()
    result = run_phase4_calibration(cfg)
    out = run_phase8_inverse_design(cfg, result, target_gr_nm_min=args.target_gr, target_ge_frac=args.target_ge)
    _print_json(out)


def cmd_sensitivity(args: argparse.Namespace) -> None:
    from chem_ml.pipeline import run_phase4_calibration, run_phase6_identifiability

    cfg = Config()
    p4 = run_phase4_calibration(cfg)
    p6 = run_phase6_identifiability(cfg, p4)
    _print_json(p6["report"])


def cmd_export_mechanism(args: argparse.Namespace) -> None:
    from chem_ml.pipeline import run_phase4_calibration, _GR_PARAM_NAMES, _GE_PARAM_NAMES
    from chem_ml.calibration import posterior_mean_params
    from chem_ml.cfd.mechanism import export_mechanism_to_cfd

    cfg = Config()
    result = run_phase4_calibration(cfg)
    theta_gr = posterior_mean_params(result["mcmc_gr"], _GR_PARAM_NAMES)
    theta_ge = posterior_mean_params(result["mcmc_ge"], _GE_PARAM_NAMES)
    out = export_mechanism_to_cfd(theta_gr, theta_ge, result["features_ds1"].invT_scaler,
                                  args.out_dir, chem_system=args.system)
    _print_json(out["manifest"])
    print(f"UDF written to {out['udf_path']}, mechanism to {out['mechanism_path']}")


def cmd_export_udf(args: argparse.Namespace) -> None:
    from pathlib import Path

    from chem_ml.cfd.mechanism import export_calibrated_model_to_cfd
    from chem_ml.model_package import CalibratedChemistryModel

    payload = json.loads(Path(args.model_json).read_text())
    model = CalibratedChemistryModel.from_jsonable(payload)
    out = export_calibrated_model_to_cfd(model, args.out_dir)
    _print_json(out["manifest"])
    print(f"UDF written to {out['udf_path']}; manifest written to {out['manifest_path']}")


def _parse_flow_assignments(assignments: list[str]) -> dict[str, float]:
    flows = {}
    for item in assignments:
        if "=" not in item:
            raise ValueError(f"Flow must be SPECIES=SCCM, got {item!r}")
        species, value = item.split("=", 1)
        flows[species] = float(value)
    return flows


def cmd_cfd_ingest(args: argparse.Namespace) -> None:
    from chem_ml.cfd.io import CFDCondition, parse_cfd_output
    from chem_ml.cfd.transfer import extract_transfer_priors

    cond = CFDCondition(
        T_set_K=args.t_set_k,
        flows_sccm=_parse_flow_assignments(args.flow),
        P_tot_torr=args.p_tot_torr,
        geometry_id=args.geometry_id,
        condition_id=args.condition_id or "",
    )
    result = parse_cfd_output(args.csv, cond)
    priors = extract_transfer_priors([result])
    _print_json({
        "status": "parsed",
        "condition_id": cond.condition_id,
        "n_radial_points": int(len(result.r_mm)),
        "transfer_priors": priors,
    })


def cmd_active_learn(args: argparse.Namespace) -> None:
    import numpy as np
    from pathlib import Path
    from chem_ml.active_learning import ActiveLearner
    from chem_ml.cfd.io import write_cfd_deck, _default_condition_id

    cfg = Config()
    bounds = np.array(args.bounds).reshape(4, 2)
    al = ActiveLearner(cfg, bounds, geometry_id=args.geometry_id)

    out_dir = Path(args.out_dir)
    if args.mode == "seed":
        conds = al.seed(args.n)
    else:
        raise NotImplementedError(
            "mode=batch requires a persisted GP state across CLI calls -- not implemented "
            "for the CLI (re-run inside a script/notebook using ActiveLearner directly, "
            "ingesting your accumulated CFDResults, then calling select_batch)."
        )
    for cond in conds:
        write_cfd_deck(cond, out_dir / f"{cond.condition_id or _default_condition_id(cond)}.txt")
    print(f"Wrote {len(conds)} CFD-ACE+ input specifications to {out_dir}. "
         f"Run CFD-ACE+ on each, produce the output CSV contract (see cfd/io.py), "
         f"then use ActiveLearner.ingest() + select_batch() in a script for the next round.")


def cmd_report(args: argparse.Namespace) -> None:
    cfg = Config()
    result = validate(
        cfg,
        ValidateRequest(suite=ValidationSuite.ALL, write_report=True, report_path="VALIDATION_REPORT.md"),
    )
    print(f"Wrote {result['report_path']} (with figures/ regenerated).")


def cmd_plots(args: argparse.Namespace) -> None:
    from chem_ml.plots import main as generate_figures

    calib = generate_figures()
    print(f"Wrote figures/*.png. Calibration summary: {calib}")


def cmd_inference_plots(args: argparse.Namespace) -> None:
    from chem_ml.inference_plots import main as generate_inference_figures

    generate_inference_figures()
    print("Wrote figures/inference_*.png.")


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        prog="chem-ml",
        description="Train Applied epitaxy chemistry models and export deterministic CFD-ACE+ surface UDFs.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_data = sub.add_parser("data", help="Intent-based data intake")
    data_sub = p_data.add_subparsers(dest="data_command", required=True)
    p = data_sub.add_parser("add", help="Register scalar or spatial experimental data")
    p.add_argument("--kind", required=True, choices=["scalar", "spatial-scan"])
    p.add_argument("--csv", help="Scalar standard-intake CSV")
    p.add_argument("--runs-csv", help="Spatial scan runs CSV")
    p.add_argument("--points-csv", help="Spatial scan points CSV")
    p.add_argument("--reactor", required=True)
    p.add_argument("--chem-class", required=True, choices=CHEM_CLASS_CHOICES)
    p.add_argument("--tag", required=True, help="Unique source tag -- re-using one is a hard error")
    p.add_argument("--mode", default="blanket", choices=["blanket", "selective"])
    p.set_defaults(func=cmd_data_add)

    p = sub.add_parser("train", help="Intent-based training and transfer fitting")
    p.add_argument("--target", required=True, choices=["chemistry", "reactor-transfer", "spatial-transfer"])
    p.add_argument("--strategy", choices=["pooled", "warm-start", "frozen-chemistry"],
                   help="Defaults to pooled for chemistry and frozen-chemistry for transfer targets")
    p.add_argument("--csv", help="Scalar CSV for warm-start or reactor-transfer training")
    p.add_argument("--reactor", help="Reactor id for scalar CSV inputs")
    p.add_argument("--chem-class", default="SiGe", choices=CHEM_CLASS_CHOICES)
    p.add_argument("--reference-reactor", default="ASM_Epsilon",
                   help="Reference reactor for class-aware pooled chemistry training")
    p.add_argument("--tag", help="Unique source tag for warm-start data, or registered spatial scan tag")
    p.add_argument("--mode", default="blanket", choices=["blanket", "selective"])
    p.add_argument("--widen-factor", type=float, default=2.0)
    p.add_argument("--base-only", action="store_true", help="For chemistry/pooled, ignore registered additions")
    p.add_argument("--benchmark-tomasini", action="store_true",
                   help="Use Tomasini benchmark data. Production training defaults to registered Applied data only.")
    p.add_argument("--save-posteriors", action="store_true", help="Write data/processed/posteriors/*.nc")
    p.add_argument("--species", nargs="+",
                   help="Active precursor species, e.g. dichlorosilane germane hcl hydrogen")
    p.add_argument("--target-deposit", help="Human-readable target label, e.g. SiGe or SiGeC:B")
    p.add_argument("--save-model-package", action="store_true",
                   help="Write a JSON package consumable by export-udf")
    p.add_argument("--model-package-path", default="data/processed/model_package.json")
    p.add_argument("--no-residual-nn", action="store_true",
                   help="Disable bounded residual-NN fitting in the saved model package")
    p.add_argument("--residual-steps", type=int, default=2000,
                   help="Optimizer steps for bounded residual-NN package fit")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("validate", help="Intent-based validation suites")
    p.add_argument("--suite", required=True, choices=["reproduction", "transfer", "spatial", "cfd-contract", "all"])
    p.add_argument("--tag", help="Registered spatial scan tag for --suite spatial")
    p.add_argument("--write-report", action="store_true", help="For --suite all, also write the markdown report")
    p.add_argument("--report-path", default="VALIDATION_REPORT.md")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("calibrate", help="Run the Tomasini benchmark calibration")
    p.add_argument("--pooled", action="store_true", help="Include registered additions (data_store), not just Tomasini")
    p.add_argument("--save-posteriors", action="store_true", help="Write data/processed/posteriors/*.nc")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("add-data", help="Register a new CSV (standard intake format) for later pooling/warm-start")
    p.add_argument("--csv", required=True)
    p.add_argument("--reactor", required=True)
    p.add_argument("--chem-class", required=True, choices=CHEM_CLASS_CHOICES)
    p.add_argument("--tag", required=True, help="Unique source tag -- re-using one is a hard error")
    p.add_argument("--mode", default="blanket", choices=["blanket", "selective"])
    p.set_defaults(func=cmd_add_data)

    p = sub.add_parser("warm-start", help="Register new data AND fold it in fast (approximate, see docs)")
    p.add_argument("--csv", required=True)
    p.add_argument("--reactor", required=True)
    p.add_argument("--chem-class", required=True, choices=CHEM_CLASS_CHOICES)
    p.add_argument("--tag", required=True)
    p.add_argument("--mode", default="blanket", choices=["blanket", "selective"])
    p.add_argument("--widen-factor", type=float, default=2.0)
    p.set_defaults(func=cmd_warm_start)

    p = sub.add_parser("add-reactor", help="Fit a reactor-transfer adapter with chemistry frozen")
    p.add_argument("--csv", required=True)
    p.add_argument("--reactor", required=True)
    p.set_defaults(func=cmd_add_reactor)

    p = sub.add_parser("add-wafer-scan", help="Register a spatial wafer scan (contour/radial GR/Ge scan)")
    p.add_argument("--runs-csv", required=True, help="One row per wafer run: run_id, T_set_C, ratios, Stick_i/probe_i cols")
    p.add_argument("--points-csv", required=True, help="One row per measured point: run_id, x_mm, y_mm, GR/Ge/thickness")
    p.add_argument("--reactor", required=True)
    p.add_argument("--chem-class", required=True, choices=CHEM_CLASS_CHOICES)
    p.add_argument("--tag", required=True, help="Unique source tag -- re-using one is a hard error")
    p.set_defaults(func=cmd_add_wafer_scan)

    p = sub.add_parser("spatial-fit", help="Fit a radially-resolved transfer adapter against a registered wafer scan")
    p.add_argument("--tag", required=True, help="source_tag a wafer scan was registered under via add-wafer-scan")
    p.set_defaults(func=cmd_spatial_fit)

    p = sub.add_parser("add-species", help="Register a new precursor/dopant/carrier species")
    p.add_argument("--name", required=True)
    p.add_argument("--formula", required=True)
    p.add_argument("--role", required=True,
                   choices=["Si-source", "Ge-source", "C-source", "dopant", "selectivity-agent", "carrier", "byproduct"])
    p.add_argument("--family", required=True)
    p.add_argument("--n-si", type=int, default=0)
    p.add_argument("--n-ge", type=int, default=0)
    p.add_argument("--n-c", type=int, default=0)
    p.add_argument("--n-cl", type=int, default=0)
    p.add_argument("--n-h", type=int, default=0)
    p.add_argument("--produces-hcl", action="store_true")
    p.set_defaults(func=cmd_add_species)

    p = sub.add_parser("predict", help="Posterior-predictive GR/Ge at one recipe, with credible intervals")
    p.add_argument("--t-c", type=float, required=True)
    p.add_argument("--hcl-ratio", type=float, required=True, help="p_HCl / p_DCS")
    p.add_argument("--geh4-ratio", type=float, required=True, help="p_GeH4 / p_DCS")
    p.add_argument("--b2h6-ratio", type=float, default=None, help="p_B2H6 / p_DCS (optional)")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("inverse", help="Benchmark/demo inverse design for target GR/Ge")
    p.add_argument("--target-gr", type=float, required=True, help="nm/min")
    p.add_argument("--target-ge", type=float, required=True, help="fraction, e.g. 0.20 for 20%%")
    p.set_defaults(func=cmd_inverse)

    p = sub.add_parser("sensitivity", help="Benchmark identifiability + sensitivity derivatives")
    p.set_defaults(func=cmd_sensitivity)

    p = sub.add_parser("export-mechanism", help="Benchmark DCS/silane mechanism export; production uses export-udf")
    p.add_argument("--system", default="dcs", choices=["dcs", "silane"])
    p.add_argument("--out-dir", default="cfd_export")
    p.set_defaults(func=cmd_export_mechanism)

    p = sub.add_parser("export-udf", help="Export a production chemistry package JSON as surface_udf.c")
    p.add_argument("--model-json", required=True, help="CalibratedChemistryModel JSON package")
    p.add_argument("--out-dir", default="cfd_export")
    p.set_defaults(func=cmd_export_udf)

    p = sub.add_parser("cfd-ingest", help="Parse a CFD-ACE+ wall-profile CSV and report transport priors")
    p.add_argument("--csv", required=True, help="CFD wall-profile CSV matching chem_ml.cfd.io output contract")
    p.add_argument("--t-set-k", type=float, required=True)
    p.add_argument("--p-tot-torr", type=float, required=True)
    p.add_argument("--geometry-id", required=True)
    p.add_argument("--condition-id", default="")
    p.add_argument("--flow", action="append", required=True,
                   help="Inlet flow assignment SPECIES=SCCM; repeat, e.g. --flow DCS=50 --flow HCl=25 --flow GeH4=2")
    p.set_defaults(func=cmd_cfd_ingest)

    p = sub.add_parser("active-learn", help="GP-guided CFD condition selection")
    p.add_argument("--mode", default="seed", choices=["seed", "batch"])
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--bounds", type=float, nargs=8, required=True,
                  metavar=("T_LO", "T_HI", "HCL_LO", "HCL_HI", "GEH4_LO", "GEH4_HI", "P_LO", "P_HI"))
    p.add_argument("--geometry-id", default="XYZ_3D_v1")
    p.add_argument("--out-dir", default="cfd_runs")
    p.set_defaults(func=cmd_active_learn)

    p = sub.add_parser("report", help="Regenerate VALIDATION_REPORT.md + figures/ end to end")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("plots", help="Regenerate benchmark reproduction + calibration plots only")
    p.set_defaults(func=cmd_plots)

    p = sub.add_parser("inference-plots", help="Generate posterior/credible-interval/extrapolation-comparison plots")
    p.set_defaults(func=cmd_inference_plots)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
