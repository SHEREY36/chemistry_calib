"""
Command-line entry points for the whole pipeline (Phase 11 commissioning).
See README.md for the full command reference an HVM process engineer
would actually use day to day.

Design note: every command that needs a fitted posterior (predict,
inverse, sensitivity, export-mechanism, add-reactor) just RE-RUNS Phase 4
calibration fresh rather than loading a cached model file. This is a
deliberate simplicity choice, not an oversight: NUTS on this dataset size
(order ~100 rows, 3 small models) takes single-digit seconds, so there is
no serialized "model.pt"-style artifact to keep in sync -- the source of
truth is always data/raw + data/processed/additions_manifest.json, and
`calibrate` writes the resulting posteriors to data/processed/posteriors/
*.nc (arviz NetCDF) purely as a record, not as something other commands
read back from. If the accumulated dataset grows to the point this
becomes slow, that's the point to add caching -- not before.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from chem_ml.config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("chem_ml.cli")


def _print_json(obj) -> None:
    def default(o):
        try:
            return float(o)
        except (TypeError, ValueError):
            return str(o)
    print(json.dumps(obj, indent=2, default=default))


# ---------------------------------------------------------------------------
def cmd_calibrate(args: argparse.Namespace) -> None:
    from chem_ml.pipeline import run_phase4_calibration
    from chem_ml.data_store import load_accumulated_dataset

    cfg = Config()
    if args.pooled:
        ds = load_accumulated_dataset(cfg)
        log.info("Calibrating against the FULL accumulated dataset (Tomasini + all registered additions)")
    else:
        ds = None
        log.info("Calibrating against Tomasini only (pass --pooled to include registered additions)")

    result = run_phase4_calibration(cfg, ds=ds)
    _print_json(result["report"])

    if args.save_posteriors:
        import arviz as az
        from pathlib import Path
        out = Path(cfg.data_processed) / "posteriors"
        out.mkdir(parents=True, exist_ok=True)
        az.to_netcdf(result["idata_gr"], out / "gr.nc")
        az.to_netcdf(result["idata_ge"], out / "ge.nc")
        az.to_netcdf(result["idata_b"], out / "b.nc")
        log.info("Wrote posteriors to %s", out)


def cmd_add_data(args: argparse.Namespace) -> None:
    from chem_ml.data_store import register_new_data
    from chem_ml.schema import ChemClass, Mode

    cfg = Config()
    register_new_data(cfg, args.csv, reactor_id=args.reactor, chem_class=ChemClass(args.chem_class),
                      source_tag=args.tag, mode=Mode(args.mode))
    print(f"Registered '{args.tag}' ({args.csv}) for reactor={args.reactor}, "
          f"chem_class={args.chem_class}. Run 'calibrate --pooled' to fold it in, "
          f"or 'warm-start' for a fast approximate update.")


def cmd_warm_start(args: argparse.Namespace) -> None:
    from chem_ml.pipeline import run_phase4_calibration, run_phase4_warm_start
    from chem_ml.data_store import register_new_data, load_accumulated_dataset
    from chem_ml.schema import ChemClass, Mode, ingest_standard_csv

    cfg = Config()
    log.info("Fitting previous posterior on the CURRENT accumulated dataset (pre-registration)...")
    previous = run_phase4_calibration(cfg, ds=load_accumulated_dataset(cfg))

    register_new_data(cfg, args.csv, reactor_id=args.reactor, chem_class=ChemClass(args.chem_class),
                      source_tag=args.tag, mode=Mode(args.mode))
    new_ds = ingest_standard_csv(args.csv, reactor_id=args.reactor, chem_class=ChemClass(args.chem_class),
                                 mode=Mode(args.mode), source_tag=args.tag)

    updated = run_phase4_warm_start(cfg, previous, new_ds, widen_factor=args.widen_factor)
    _print_json({
        "n_new_sige_rows": updated["n_new_sige_rows"], "n_new_sigeb_rows": updated["n_new_sigeb_rows"],
        "warm_start_widen_factor": updated["warm_start_widen_factor"],
    })
    print("Warm-start applied. Recommend a full 'calibrate --pooled' periodically as the ground-truth resync.")


def cmd_add_reactor(args: argparse.Namespace) -> None:
    """Phase 7-style: freeze theta_chem from the reference reactor, fit
    ONLY delta_r for a brand-new reactor's data."""
    from chem_ml.pipeline import run_phase4_calibration, _GR_PARAM_NAMES, _GE_PARAM_NAMES, _run_reactor_mcmc
    from chem_ml.calibration import posterior_mean_params, r2_score
    from chem_ml.features import build_features
    from chem_ml.physics_core import gr_logmodel, ge_logmodel
    from chem_ml.reactor_transfer import reactor_transfer_model_ge_only, reactor_transfer_model_gr_ge
    from chem_ml.schema import ChemClass, Mode, ingest_standard_csv
    import numpy as np
    import jax.numpy as jnp

    cfg = Config()
    base = run_phase4_calibration(cfg)
    theta_gr = posterior_mean_params(base["mcmc_gr"], _GR_PARAM_NAMES)
    theta_ge = posterior_mean_params(base["mcmc_ge"], _GE_PARAM_NAMES)
    invT_scaler = base["features_ds1"].invT_scaler

    ds_new = ingest_standard_csv(args.csv, reactor_id=args.reactor, chem_class=ChemClass.SIGE,
                                 mode=Mode.BLANKET, source_tag=f"reactor:{args.reactor}")
    fb = build_features(ds_new, invT_scaler=invT_scaler)
    y_ge = np.array([r.Ge_at_frac for r in ds_new.rows])
    y_ge_log = jnp.asarray(np.log(y_ge / (1.0 - y_ge)))
    has_gr = all(r.GR_nm_min is not None for r in ds_new.rows)

    if has_gr:
        y_gr = np.array([r.GR_nm_min for r in ds_new.rows])
        y_gr_log = jnp.asarray(np.log(y_gr))
        mcmc = _run_reactor_mcmc(reactor_transfer_model_gr_ge, cfg, X=fb.X, y_gr_log=y_gr_log,
                                y_ge_log=y_ge_log, theta_gr=theta_gr, theta_ge=theta_ge)
        s = mcmc.get_samples()
        X_eff = fb.X.at[:, 1].add(float(jnp.mean(s["ln_alpha_HCl"]))).at[:, 2].add(float(jnp.mean(s["ln_alpha_GeH4"])))
        gr_pred = np.exp(float(jnp.mean(s["ln_eta_GR"])) + np.asarray(gr_logmodel(theta_gr, X_eff)))
        r2_gr = r2_score(y_gr, gr_pred)
    else:
        mcmc = _run_reactor_mcmc(reactor_transfer_model_ge_only, cfg, X=fb.X, y_ge_log=y_ge_log, theta_ge=theta_ge)
        s = mcmc.get_samples()
        X_eff = fb.X.at[:, 1].add(float(jnp.mean(s["ln_alpha_HCl"]))).at[:, 2].add(float(jnp.mean(s["ln_alpha_GeH4"])))
        r2_gr = None

    ge_pred = np.exp(float(jnp.mean(s["ln_eta_Ge"])) + np.asarray(ge_logmodel(theta_ge, X_eff)))
    r2_ge = r2_score(y_ge / (1 - y_ge), ge_pred)

    result = {
        "reactor_id": args.reactor, "n_rows": len(ds_new),
        "ln_alpha_HCl": float(jnp.mean(s["ln_alpha_HCl"])), "ln_alpha_GeH4": float(jnp.mean(s["ln_alpha_GeH4"])),
        "R2_GR": r2_gr, "R2_Ge": r2_ge,
        "note": "See METHODOLOGY.md sec 8 for why individual alpha values are not uniquely identified from wafer data alone.",
    }
    _print_json(result)


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
    from chem_ml.report import generate_validation_report

    report = generate_validation_report()
    with open("VALIDATION_REPORT.md", "w") as f:
        f.write(report)
    print("Wrote VALIDATION_REPORT.md (with figures/ regenerated).")


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
    parser = argparse.ArgumentParser(prog="chem-ml")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("calibrate", help="Run Phase 1-4 ingest + Bayesian calibration")
    p.add_argument("--pooled", action="store_true", help="Include registered additions (data_store), not just Tomasini")
    p.add_argument("--save-posteriors", action="store_true", help="Write data/processed/posteriors/*.nc")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("add-data", help="Register a new CSV (standard intake format) for later pooling/warm-start")
    p.add_argument("--csv", required=True)
    p.add_argument("--reactor", required=True)
    p.add_argument("--chem-class", required=True, choices=["Si", "SiGe", "SiGe:B", "SiGe:P", "SiC", "SiGeC"])
    p.add_argument("--tag", required=True, help="Unique source tag -- re-using one is a hard error")
    p.add_argument("--mode", default="blanket", choices=["blanket", "selective"])
    p.set_defaults(func=cmd_add_data)

    p = sub.add_parser("warm-start", help="Register new data AND fold it in fast (approximate, see docs)")
    p.add_argument("--csv", required=True)
    p.add_argument("--reactor", required=True)
    p.add_argument("--chem-class", required=True, choices=["Si", "SiGe", "SiGe:B", "SiGe:P", "SiC", "SiGeC"])
    p.add_argument("--tag", required=True)
    p.add_argument("--mode", default="blanket", choices=["blanket", "selective"])
    p.add_argument("--widen-factor", type=float, default=2.0)
    p.set_defaults(func=cmd_warm_start)

    p = sub.add_parser("add-reactor", help="Phase 7: fit delta_r for a brand-new reactor (chemistry frozen)")
    p.add_argument("--csv", required=True)
    p.add_argument("--reactor", required=True)
    p.set_defaults(func=cmd_add_reactor)

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

    p = sub.add_parser("inverse", help="Phase 8: find a recipe achieving a target GR/Ge, with confidence gating")
    p.add_argument("--target-gr", type=float, required=True, help="nm/min")
    p.add_argument("--target-ge", type=float, required=True, help="fraction, e.g. 0.20 for 20%%")
    p.set_defaults(func=cmd_inverse)

    p = sub.add_parser("sensitivity", help="Phase 6: identifiability + sensitivity derivatives")
    p.set_defaults(func=cmd_sensitivity)

    p = sub.add_parser("export-mechanism", help="Phase 9: export calibrated model as CFD-ACE+ UDF + mechanism deck")
    p.add_argument("--system", default="dcs", choices=["dcs", "silane"])
    p.add_argument("--out-dir", default="cfd_export")
    p.set_defaults(func=cmd_export_mechanism)

    p = sub.add_parser("active-learn", help="Phase 10: GP-guided CFD condition selection")
    p.add_argument("--mode", default="seed", choices=["seed", "batch"])
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--bounds", type=float, nargs=8, required=True,
                  metavar=("T_LO", "T_HI", "HCL_LO", "HCL_HI", "GEH4_LO", "GEH4_HI", "P_LO", "P_HI"))
    p.add_argument("--geometry-id", default="XYZ_3D_v1")
    p.add_argument("--out-dir", default="cfd_runs")
    p.set_defaults(func=cmd_active_learn)

    p = sub.add_parser("report", help="Regenerate VALIDATION_REPORT.md + figures/ end to end")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("plots", help="Regenerate the Fig 2-5 reproduction + calibration plots only")
    p.set_defaults(func=cmd_plots)

    p = sub.add_parser("inference-plots", help="Generate posterior/credible-interval/extrapolation-comparison plots")
    p.set_defaults(func=cmd_inference_plots)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
