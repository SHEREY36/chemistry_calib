"""Intent-based public workflows for data intake, training, and validation.

The lower-level modules still own the science. This facade gives callers one
route into the project without asking them to remember which historical phase
implemented each capability.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from chem_ml.calibration import diagnostics, posterior_mean_params, r2_score
from chem_ml.cfd.io import _OUTPUT_COLUMNS, parse_cfd_output
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
from chem_ml.data_store import (
    load_accumulated_dataset,
    load_production_dataset,
    load_registered_wafer_scans,
    register_new_data,
    register_wafer_scan,
)
from chem_ml.features import build_features
from chem_ml.model_package import (
    CalibratedChemistryModel,
    Observable,
    build_model_spec,
    default_species_for_chem_class,
)
from chem_ml.physics_core import ge_logmodel, gr_logmodel
from chem_ml.pipeline import (
    _B_PARAM_NAMES,
    _C_PARAM_NAMES,
    _GE_PARAM_NAMES,
    _GR_PARAM_NAMES,
    _X_PARAM_NAMES,
    _run_reactor_mcmc,
    load_all_datasets,
    run_class_calibration,
    run_core_chemistry_calibration,
    run_phase4_calibration,
    run_phase4_warm_start,
    run_phase7_cross_reactor,
    run_phase12_spatial_transfer,
)
from chem_ml.reactor_transfer import reactor_transfer_model_ge_only, reactor_transfer_model_gr_ge
from chem_ml.report import generate_validation_report
from chem_ml.residual_nn import ResidualNN, build_residual_input, export_bounded_residuals
from chem_ml.schema import ChemClass, Mode, canonical_chem_class, ingest_standard_csv

log = logging.getLogger("chem_ml.workflows")


def _require(value, name: str):
    if value is None or value == "":
        raise ValueError(f"{name} is required")
    return value


def _save_posteriors(cfg: Config, result: dict) -> Path:
    import arviz as az

    out = Path(cfg.data_processed) / "posteriors"
    out.mkdir(parents=True, exist_ok=True)
    for key, name in (
        ("idata_gr", "gr.nc"),
        ("idata_ge", "ge.nc"),
        ("idata_b", "b.nc"),
        ("idata_c", "c.nc"),
        ("idata_dopant", "dopant.nc"),
    ):
        if key in result:
            az.to_netcdf(result[key], out / name)
    return out


def _fit_package_residuals(request: TrainRequest, result: dict, chem_class: ChemClass) -> dict[Observable, object]:
    """Fit GR/Ge bounded residuals from the production fit residuals."""
    if "mcmc_gr" not in result or "mcmc_ge" not in result:
        return {}
    ds = result.get("selected_dataset")
    invT_scaler = result.get("invT_scaler")
    if ds is None or invT_scaler is None:
        return {}
    joint = ds.filter_where(lambda r: r.GR_nm_min is not None and r.Ge_at_frac is not None)
    if len(joint) < 8:
        log.info("Skipping residual NN package fit: need at least 8 joint GR/Ge rows, got %d", len(joint))
        return {}

    fb = build_features(joint, invT_scaler=invT_scaler)
    theta_gr = posterior_mean_params(result["mcmc_gr"], _GR_PARAM_NAMES)
    theta_ge = posterior_mean_params(result["mcmc_ge"], _GE_PARAM_NAMES)
    y_gr_log = jnp.asarray(np.log(np.array([r.GR_nm_min for r in joint.rows])))
    y_ge = np.array([r.Ge_at_frac for r in joint.rows])
    y_ge_log = jnp.asarray(np.log(y_ge / (1.0 - y_ge)))
    resid = jnp.stack([
        y_gr_log - gr_logmodel(theta_gr, fb.X),
        y_ge_log - ge_logmodel(theta_ge, fb.X),
    ], axis=1)

    X_resid = build_residual_input(joint, fb)
    net = ResidualNN(chem_class=chem_class, in_size=X_resid.shape[1], n_out=2)
    net.fit(X_resid, resid, l2=0.3, steps=request.residual_steps)
    return export_bounded_residuals(net, [Observable.GR, Observable.GE])


def _build_model_package(cfg: Config, request: TrainRequest, result: dict) -> CalibratedChemistryModel:
    target_class = canonical_chem_class(request.chem_class)
    species = request.species_names or default_species_for_chem_class(target_class)
    spec = build_model_spec(
        target_class,
        species,
        target_deposit=request.target_deposit or target_class.value,
        mode=request.mode,
    )
    theta: dict[Observable, dict[str, float]] = {}
    if "mcmc_gr" in result:
        theta[Observable.GR] = posterior_mean_params(result["mcmc_gr"], _GR_PARAM_NAMES)
    if "mcmc_ge" in result:
        theta[Observable.GE] = posterior_mean_params(result["mcmc_ge"], _GE_PARAM_NAMES)
    if "mcmc_c" in result:
        theta[Observable.C] = posterior_mean_params(result["mcmc_c"], _C_PARAM_NAMES)
    if "mcmc_b" in result:
        theta[Observable.DOPANT] = posterior_mean_params(result["mcmc_b"], _B_PARAM_NAMES)
    if "mcmc_x" in result:
        theta[Observable.DOPANT] = posterior_mean_params(result["mcmc_x"], _X_PARAM_NAMES)
    if not theta:
        raise ValueError("No fitted observable slots are available for model package export.")

    residuals = _fit_package_residuals(request, result, target_class) if request.fit_residual_nn else {}
    invT_scaler = result.get("invT_scaler")
    if invT_scaler is None:
        for key in ("features_gr", "features_ge", "features_c", "features_dopant", "features_ds1"):
            if key in result:
                invT_scaler = result[key].invT_scaler
                break
    if invT_scaler is None:
        raise ValueError("Cannot export model package without an inverse-temperature scaler.")

    return CalibratedChemistryModel(
        spec=spec,
        theta=theta,
        invT_scaler=invT_scaler,
        residuals=residuals,
        training_source="tomasini_benchmark" if request.use_benchmark_data else "registered_production_data",
        transport_deembedding="not_started",
    )


def _save_model_package(path: str | Path, model: CalibratedChemistryModel) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(model.to_jsonable(), indent=2))
    return p


def register_experiment(cfg: Config, request: RegisterExperimentRequest) -> dict:
    """Register or parse one experimental data source by declared kind."""
    if request.kind == DataKind.SCALAR:
        register_new_data(
            cfg,
            _require(request.csv_path, "csv_path"),
            reactor_id=_require(request.reactor_id, "reactor_id"),
            chem_class=request.chem_class,
            source_tag=_require(request.tag, "tag"),
            mode=request.mode,
        )
        return {
            "kind": request.kind.value,
            "tag": request.tag,
            "reactor_id": request.reactor_id,
            "chem_class": request.chem_class.value,
            "mode": request.mode.value,
        }

    if request.kind == DataKind.SPATIAL_SCAN:
        register_wafer_scan(
            cfg,
            _require(request.runs_csv, "runs_csv"),
            _require(request.points_csv, "points_csv"),
            reactor_id=_require(request.reactor_id, "reactor_id"),
            chem_class=request.chem_class,
            source_tag=_require(request.tag, "tag"),
        )
        return {
            "kind": request.kind.value,
            "tag": request.tag,
            "reactor_id": request.reactor_id,
            "chem_class": request.chem_class.value,
        }

    if request.kind == DataKind.CFD_PROFILE:
        condition = _require(request.cfd_condition, "cfd_condition")
        result = parse_cfd_output(_require(request.cfd_output_csv, "cfd_output_csv"), condition)
        return {
            "kind": request.kind.value,
            "condition_id": condition.condition_id,
            "n_radial_points": int(len(result.r_mm)),
            "_cfd_result": result,
        }

    raise ValueError(f"Unsupported data kind: {request.kind}")


def _fit_reactor_transfer(cfg: Config, request: TrainRequest) -> dict:
    """Fit a scalar reactor-transfer adapter with frozen reference chemistry."""
    base = run_phase4_calibration(cfg)
    theta_gr = posterior_mean_params(base["mcmc_gr"], _GR_PARAM_NAMES)
    theta_ge = posterior_mean_params(base["mcmc_ge"], _GE_PARAM_NAMES)
    invT_scaler = base["features_ds1"].invT_scaler

    ds_new = ingest_standard_csv(
        _require(request.csv_path, "csv_path"),
        reactor_id=_require(request.reactor_id, "reactor_id"),
        chem_class=request.chem_class,
        mode=request.mode,
        source_tag=request.tag or f"reactor:{request.reactor_id}",
    )
    fb = build_features(ds_new, invT_scaler=invT_scaler)
    y_ge = np.array([r.Ge_at_frac for r in ds_new.rows])
    if any(v is None for v in y_ge):
        raise ValueError("reactor-transfer training requires Ge_at_pct/Ge_at_frac on every row")
    y_ge_log = jnp.asarray(np.log(y_ge / (1.0 - y_ge)))
    has_gr = all(r.GR_nm_min is not None for r in ds_new.rows)

    if has_gr:
        y_gr = np.array([r.GR_nm_min for r in ds_new.rows])
        y_gr_log = jnp.asarray(np.log(y_gr))
        mcmc = _run_reactor_mcmc(
            reactor_transfer_model_gr_ge,
            cfg,
            X=fb.X,
            y_gr_log=y_gr_log,
            y_ge_log=y_ge_log,
            theta_gr=theta_gr,
            theta_ge=theta_ge,
        )
    else:
        y_gr = None
        mcmc = _run_reactor_mcmc(
            reactor_transfer_model_ge_only,
            cfg,
            X=fb.X,
            y_ge_log=y_ge_log,
            theta_ge=theta_ge,
        )

    diag = diagnostics(mcmc)
    s = mcmc.get_samples()
    X_eff = fb.X.at[:, 1].add(float(jnp.mean(s["ln_alpha_HCl"]))).at[:, 2].add(
        float(jnp.mean(s["ln_alpha_GeH4"]))
    )
    ge_ratio_pred = np.exp(float(jnp.mean(s["ln_eta_Ge"])) + np.asarray(ge_logmodel(theta_ge, X_eff)))

    report = {
        "reactor_id": request.reactor_id,
        "n_rows": len(ds_new),
        "ln_alpha_HCl": float(jnp.mean(s["ln_alpha_HCl"])),
        "ln_alpha_GeH4": float(jnp.mean(s["ln_alpha_GeH4"])),
        "R2_GR": None,
        "R2_Ge": r2_score(y_ge / (1 - y_ge), ge_ratio_pred),
        "note": "Individual alpha values are not uniquely identified from wafer data alone; CFD-informed priors can tighten them.",
    }
    if has_gr:
        gr_pred = np.exp(float(jnp.mean(s["ln_eta_GR"])) + np.asarray(gr_logmodel(theta_gr, X_eff)))
        report["R2_GR"] = r2_score(y_gr, gr_pred)

    return {
        "target": TrainTarget.REACTOR_TRANSFER.value,
        "strategy": TrainStrategy.FROZEN_CHEMISTRY.value,
        "report": report,
        "diagnostics": diag,
        "_mcmc": mcmc,
        "_base_result": base,
    }


def train(cfg: Config, request: TrainRequest) -> dict:
    """Train or fit the requested model/update path."""
    if request.target == TrainTarget.CHEMISTRY:
        if request.strategy == TrainStrategy.POOLED:
            if request.use_benchmark_data:
                ds = load_accumulated_dataset(cfg) if request.include_registered else None
            else:
                ds = load_production_dataset(cfg) if request.include_registered else None
            target_class = canonical_chem_class(request.chem_class)
            reference_reactor = request.reference_reactor or request.reactor_id or "ASM_Epsilon"
            if request.use_benchmark_data and target_class == ChemClass.SIGE and reference_reactor == "ASM_Epsilon":
                result = run_phase4_calibration(cfg, ds=ds)
            elif request.use_benchmark_data:
                result = run_class_calibration(
                    cfg,
                    ds=ds,
                    chem_class=request.chem_class,
                    reference_reactor=reference_reactor,
                )
            else:
                result = run_core_chemistry_calibration(
                    cfg,
                    ds=ds or load_production_dataset(cfg),
                    chem_class=request.chem_class,
                    reference_reactor=reference_reactor,
                )
            response = {
                "target": request.target.value,
                "strategy": request.strategy.value,
                "include_registered": request.include_registered,
                "use_benchmark_data": request.use_benchmark_data,
                "chem_class": target_class.value,
                "reference_reactor": reference_reactor,
                "report": result["report"],
                "_model_result": result,
            }
            if request.save_posteriors:
                response["posterior_dir"] = str(_save_posteriors(cfg, result))
            if request.save_model_package:
                model = _build_model_package(cfg, request, result)
                response["model_package_path"] = str(_save_model_package(request.model_package_path, model))
                response["model_package"] = model.to_jsonable()
            return response

        if request.strategy == TrainStrategy.WARM_START:
            if request.chem_class not in (ChemClass.SIGE, ChemClass.SIGE_B):
                raise ValueError("warm-start currently supports legacy SiGe / SiGe:B data only; "
                                 "use pooled class training for SiGeC or other chemistry classes")
            previous = run_phase4_calibration(cfg, ds=load_accumulated_dataset(cfg))
            register_experiment(
                cfg,
                RegisterExperimentRequest(
                    kind=DataKind.SCALAR,
                    csv_path=_require(request.csv_path, "csv_path"),
                    reactor_id=_require(request.reactor_id, "reactor_id"),
                    chem_class=request.chem_class,
                    mode=request.mode,
                    tag=_require(request.tag, "tag"),
                ),
            )
            new_ds = ingest_standard_csv(
                request.csv_path,
                reactor_id=request.reactor_id,
                chem_class=request.chem_class,
                mode=request.mode,
                source_tag=request.tag,
            )
            updated = run_phase4_warm_start(cfg, previous, new_ds, widen_factor=request.widen_factor)
            return {
                "target": request.target.value,
                "strategy": request.strategy.value,
                "n_new_sige_rows": updated["n_new_sige_rows"],
                "n_new_sigeb_rows": updated["n_new_sigeb_rows"],
                "warm_start_widen_factor": updated["warm_start_widen_factor"],
                "_model_result": updated,
            }

        raise ValueError(f"Unsupported chemistry training strategy: {request.strategy}")

    if request.target == TrainTarget.REACTOR_TRANSFER:
        if request.strategy != TrainStrategy.FROZEN_CHEMISTRY:
            raise ValueError("reactor-transfer training requires strategy=frozen_chemistry")
        return _fit_reactor_transfer(cfg, request)

    if request.target == TrainTarget.SPATIAL_TRANSFER:
        if request.strategy != TrainStrategy.FROZEN_CHEMISTRY:
            raise ValueError("spatial-transfer training requires strategy=frozen_chemistry")
        base = run_phase4_calibration(cfg)
        scans = load_registered_wafer_scans(cfg, _require(request.tag, "tag"))
        reports = [run_phase12_spatial_transfer(cfg, base, scan)["report"] for scan in scans]
        return {
            "target": request.target.value,
            "strategy": request.strategy.value,
            "tag": request.tag,
            "reports": reports,
            "_base_result": base,
        }

    raise ValueError(f"Unsupported training target: {request.target}")


def validate(cfg: Config, request: ValidateRequest) -> dict:
    """Run one validation suite through the facade."""
    if request.suite == ValidationSuite.REPRODUCTION:
        out = train(
            cfg,
            TrainRequest(
                target=TrainTarget.CHEMISTRY,
                strategy=TrainStrategy.POOLED,
                include_registered=False,
                use_benchmark_data=True,
            ),
        )
        return {"suite": request.suite.value, "report": out["report"], "_model_result": out["_model_result"]}

    if request.suite == ValidationSuite.TRANSFER:
        ds = load_all_datasets(cfg)
        p4 = run_phase4_calibration(cfg, ds)
        p7 = run_phase7_cross_reactor(cfg, p4, ds)
        return {"suite": request.suite.value, "report": p7["report"], "_phase4": p4, "_transfer": p7}

    if request.suite == ValidationSuite.SPATIAL:
        if not request.tag:
            return {
                "suite": request.suite.value,
                "status": "skipped",
                "reason": "No spatial source tag supplied; synthetic spatial recovery is covered by tests/test_spatial.py.",
            }
        out = train(
            cfg,
            TrainRequest(
                target=TrainTarget.SPATIAL_TRANSFER,
                strategy=TrainStrategy.FROZEN_CHEMISTRY,
                tag=request.tag,
            ),
        )
        return {"suite": request.suite.value, "reports": out["reports"], "_validation": out}

    if request.suite == ValidationSuite.CFD_CONTRACT:
        if not request.cfd_output_csv:
            return {
                "suite": request.suite.value,
                "status": "contract_available",
                "required_columns": list(_OUTPUT_COLUMNS),
            }
        result = parse_cfd_output(
            request.cfd_output_csv,
            _require(request.cfd_condition, "cfd_condition"),
        )
        return {
            "suite": request.suite.value,
            "status": "parsed",
            "n_radial_points": int(len(result.r_mm)),
            "_cfd_result": result,
        }

    if request.suite == ValidationSuite.ALL:
        response = {
            "suite": request.suite.value,
            "reproduction": validate(cfg, ValidateRequest(suite=ValidationSuite.REPRODUCTION))["report"],
            "transfer": validate(cfg, ValidateRequest(suite=ValidationSuite.TRANSFER))["report"],
            "spatial": validate(cfg, ValidateRequest(suite=ValidationSuite.SPATIAL, tag=request.tag)),
            "cfd_contract": validate(
                cfg,
                ValidateRequest(
                    suite=ValidationSuite.CFD_CONTRACT,
                    cfd_output_csv=request.cfd_output_csv,
                    cfd_condition=request.cfd_condition,
                ),
            ),
        }
        if request.write_report:
            report = generate_validation_report(cfg)
            Path(request.report_path).write_text(report)
            response["report_path"] = request.report_path
            response["_report_text"] = report
        return response

    raise ValueError(f"Unsupported validation suite: {request.suite}")
