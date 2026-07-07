"""Tests for the intent-based workflow facade."""
import pytest

from chem_ml.cfd.io import CFDCondition
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
from chem_ml.data_store import load_accumulated_dataset, load_registered_wafer_scans
from chem_ml.schema import ChemClass
from chem_ml import workflows


def _cfg(tmp_path):
    return Config(data_raw="data/raw", data_processed=str(tmp_path))


def test_register_experiment_scalar_routes_to_scalar_manifest(tmp_path):
    csv_path = tmp_path / "new_wafers.csv"
    csv_path.write_text(
        "T_C,HCl_over_DCS,GeH4_over_DCS,GR_nm_min,Ge_at_pct\n"
        "730,0.6,0.03,40.0,20.0\n"
    )
    cfg = _cfg(tmp_path)

    out = workflows.register_experiment(
        cfg,
        RegisterExperimentRequest(
            kind=DataKind.SCALAR,
            csv_path=str(csv_path),
            reactor_id="ASM_Epsilon",
            chem_class=ChemClass.SIGE,
            tag="scalar_batch",
        ),
    )

    assert out["kind"] == "scalar"
    assert len(load_accumulated_dataset(cfg).filter(source_dataset="scalar_batch")) == 1


def test_register_experiment_spatial_scan_routes_to_spatial_manifest(tmp_path):
    runs_csv = tmp_path / "runs.csv"
    runs_csv.write_text(
        "run_id,T_set_C,HCl_over_DCS,GeH4_over_DCS\n"
        "wafer_1,727.0,0.5,0.03\n"
    )
    points_csv = tmp_path / "points.csv"
    points_csv.write_text(
        "run_id,x_mm,y_mm,GR_nm_min_local,Ge_at_pct_local\n"
        "wafer_1,0,0,40.0,22.0\n"
        "wafer_1,100,0,36.0,20.0\n"
    )
    cfg = _cfg(tmp_path)

    out = workflows.register_experiment(
        cfg,
        RegisterExperimentRequest(
            kind=DataKind.SPATIAL_SCAN,
            runs_csv=str(runs_csv),
            points_csv=str(points_csv),
            reactor_id="XYZ_tool_1",
            chem_class=ChemClass.SIGE,
            tag="scan_batch",
        ),
    )

    assert out["kind"] == "spatial_scan"
    scans = load_registered_wafer_scans(cfg, "scan_batch")
    assert len(scans) == 1
    assert len(scans[0].points) == 2


def test_register_experiment_cfd_profile_parses_output_contract(tmp_path):
    csv_path = tmp_path / "cfd_profile.csv"
    csv_path.write_text(
        "r_mm,surface_T_K,p_HCl_over_pDCS,p_GeH4_over_pDCS,p_B2H6_over_pDCS,GR_nm_min,Ge_frac\n"
        "0,1020.0,0.55,0.032,0.0,60.0,0.21\n"
        "50,1021.5,0.53,0.031,0.0,58.0,0.205\n"
    )
    cond = CFDCondition(
        T_set_K=1023.15,
        flows_sccm={"DCS": 10.0, "GeH4": 0.3, "HCl": 5.0, "B2H6": 0.0},
        P_tot_torr=10.0,
        geometry_id="XYZ_3D_v1",
        condition_id="cfd_1",
    )

    out = workflows.register_experiment(
        _cfg(tmp_path),
        RegisterExperimentRequest(
            kind=DataKind.CFD_PROFILE,
            cfd_output_csv=str(csv_path),
            cfd_condition=cond,
        ),
    )

    assert out["kind"] == "cfd_profile"
    assert out["n_radial_points"] == 2
    assert out["_cfd_result"].condition.condition_id == "cfd_1"


def test_train_chemistry_pooled_dispatches_to_accumulated_dataset(monkeypatch, tmp_path):
    calls = {}

    def fake_load_accumulated_dataset(cfg):
        calls["loaded"] = True
        return "accumulated"

    def fake_run_phase4_calibration(cfg, ds=None):
        calls["ds"] = ds
        return {"report": {"PASS": True}}

    monkeypatch.setattr(workflows, "load_accumulated_dataset", fake_load_accumulated_dataset)
    monkeypatch.setattr(workflows, "run_phase4_calibration", fake_run_phase4_calibration)

    out = workflows.train(
        _cfg(tmp_path),
        TrainRequest(target=TrainTarget.CHEMISTRY, strategy=TrainStrategy.POOLED),
    )

    assert calls == {"loaded": True, "ds": "accumulated"}
    assert out["report"]["PASS"] is True


def test_validate_cfd_contract_without_file_reports_required_columns(tmp_path):
    out = workflows.validate(_cfg(tmp_path), ValidateRequest(suite=ValidationSuite.CFD_CONTRACT))

    assert out["status"] == "contract_available"
    assert "r_mm" in out["required_columns"]


def test_validate_cfd_contract_rejects_missing_condition_when_file_given(tmp_path):
    csv_path = tmp_path / "cfd_profile.csv"
    csv_path.write_text("r_mm,surface_T_K\n0,1020.0\n")

    with pytest.raises(ValueError, match="cfd_condition is required"):
        workflows.validate(
            _cfg(tmp_path),
            ValidateRequest(suite=ValidationSuite.CFD_CONTRACT, cfd_output_csv=str(csv_path)),
        )
