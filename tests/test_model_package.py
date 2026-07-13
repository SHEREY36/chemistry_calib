import json
import shutil
import subprocess

import numpy as np
import pytest

from chem_ml.cfd.mechanism import export_calibrated_model_to_cfd
from chem_ml.model_package import (
    BoundedResidualMLP,
    CalibratedChemistryModel,
    Observable,
    build_model_spec,
    default_species_for_chem_class,
)
from chem_ml.schema import ChemClass


def _theta_gr():
    return {"lnK_GR": 6.4, "kappa_GR": -1.5, "gamma_HCl": -0.71, "gamma_GeH4": 1.31}


def _theta_ge():
    return {"lnK_Ge": 0.66, "kappa_Ge": 0.26, "dgamma_HCl": 0.10, "dgamma_GeH4": 0.51}


def test_build_model_spec_enables_observable_slots_from_species():
    spec = build_model_spec(
        ChemClass.SIGEC_X,
        ["dichlorosilane", "germane", "methylsilane", "diborane", "hcl", "hydrogen"],
        target_deposit="SiGeC:B",
    )

    assert spec.target_deposit == "SiGeC:B"
    assert spec.si_source == "dichlorosilane"
    assert spec.ge_source == "germane"
    assert spec.c_source == "methylsilane"
    assert spec.dopant == "diborane"
    assert spec.enabled_observables == (
        Observable.GR,
        Observable.GE,
        Observable.C,
        Observable.DOPANT,
    )


def test_bounded_residual_mlp_is_log_space_limited():
    residual = BoundedResidualMLP(
        observable=Observable.GR,
        input_names=("invT_std",),
        weights=[np.array([[100.0]])],
        biases=[np.array([0.0])],
        max_abs_log_correction=0.25,
    )

    assert residual([1.0]) <= 0.25
    assert residual([1.0]) == pytest.approx(0.25, abs=1e-6)
    assert residual([-1.0]) == pytest.approx(-0.25, abs=1e-6)


def test_default_species_for_sigecx_is_clone_and_train_ready():
    species = default_species_for_chem_class(ChemClass.SIGEC_X)

    assert "dichlorosilane" in species
    assert "germane" in species
    assert "methylsilane" in species
    assert "diborane" in species
    assert "hcl" in species


def test_model_package_roundtrip_jsonable():
    spec = build_model_spec(ChemClass.SIGE, ["dichlorosilane", "germane", "hcl"])
    residual = BoundedResidualMLP(
        observable=Observable.GR,
        input_names=("invT_std",),
        weights=[np.array([[0.1]])],
        biases=[np.array([0.0])],
        max_abs_log_correction=0.4,
    )
    model = CalibratedChemistryModel(
        spec=spec,
        theta={Observable.GR: _theta_gr(), Observable.GE: _theta_ge()},
        invT_scaler=(9.1e-4, 5.2e-5),
        residuals={Observable.GR: residual},
    )

    restored = CalibratedChemistryModel.from_jsonable(model.to_jsonable())

    assert restored.spec.chem_class == ChemClass.SIGE
    assert restored.theta[Observable.GR]["gamma_HCl"] == pytest.approx(-0.71)
    assert restored.residuals[Observable.GR]([10.0]) <= 0.4


def test_export_calibrated_model_writes_residual_nn_into_udf(tmp_path):
    spec = build_model_spec(ChemClass.SIGE, ["dichlorosilane", "germane", "hcl"])
    residual = BoundedResidualMLP(
        observable=Observable.GR,
        input_names=("invT_std", "ln_HCl"),
        weights=[np.array([[0.1, -0.2], [0.3, 0.4]]), np.array([[0.5, -0.6]])],
        biases=[np.array([0.01, -0.02]), np.array([0.03])],
        max_abs_log_correction=0.4,
    )
    model = CalibratedChemistryModel(
        spec=spec,
        theta={Observable.GR: _theta_gr(), Observable.GE: _theta_ge()},
        invT_scaler=(9.1e-4, 5.2e-5),
        residuals={Observable.GR: residual},
        training_source="applied_experimental_data",
        transport_deembedding="cfd_local_fields_ingested",
    )

    out = export_calibrated_model_to_cfd(model, tmp_path)
    udf = (tmp_path / "surface_udf.c").read_text()
    manifest = json.loads((tmp_path / "model_manifest.json").read_text())

    assert "residual_GR_log" in udf
    assert "calibrated_GR_nm_min_full" in udf
    assert "+ residual_GR_log(all_x)" in udf
    assert manifest["training_source"] == "applied_experimental_data"
    assert manifest["transport_deembedding"] == "cfd_local_fields_ingested"
    assert out["udf_path"].endswith("surface_udf.c")


def test_exported_udf_compiles_as_c99(tmp_path):
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("No C compiler available for UDF compile check")

    spec = build_model_spec(ChemClass.SIGE, ["dichlorosilane", "germane", "hcl"])
    model = CalibratedChemistryModel(
        spec=spec,
        theta={Observable.GR: _theta_gr(), Observable.GE: _theta_ge()},
        invT_scaler=(9.1e-4, 5.2e-5),
    )
    out = export_calibrated_model_to_cfd(model, tmp_path)

    subprocess.run(
        [cc, "-std=c99", "-c", out["udf_path"], "-o", str(tmp_path / "surface_udf.o")],
        check=True,
    )
