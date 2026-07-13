import sys

from chem_ml import cli
from chem_ml.contracts import DataKind, TrainTarget
from chem_ml.schema import ChemClass


def test_cli_train_accepts_sigec_reference_reactor(monkeypatch, capsys):
    captured = {}

    def fake_train(cfg, request):
        captured["request"] = request
        return {"target": request.target.value, "strategy": request.strategy.value, "report": {"ok": True}}

    monkeypatch.setattr(cli, "train", fake_train)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "chem-ml",
            "train",
            "--target",
            "chemistry",
            "--strategy",
            "pooled",
            "--chem-class",
            "SiGeC",
            "--reference-reactor",
            "XYZ_tool_1",
        ],
    )

    cli.main()

    assert captured["request"].target == TrainTarget.CHEMISTRY
    assert captured["request"].chem_class == ChemClass.SIGEC
    assert captured["request"].reference_reactor == "XYZ_tool_1"
    assert '"ok": true' in capsys.readouterr().out


def test_cli_train_accepts_model_package_and_species_flags(monkeypatch):
    captured = {}

    def fake_train(cfg, request):
        captured["request"] = request
        return {"target": request.target.value, "strategy": request.strategy.value, "report": {"ok": True}}

    monkeypatch.setattr(cli, "train", fake_train)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "chem-ml",
            "train",
            "--target",
            "chemistry",
            "--chem-class",
            "SiGe",
            "--reference-reactor",
            "AMAT_tool_1",
            "--species",
            "dichlorosilane",
            "germane",
            "hcl",
            "hydrogen",
            "--target-deposit",
            "SiGe",
            "--save-model-package",
            "--model-package-path",
            "data/processed/sige_model_package.json",
            "--no-residual-nn",
        ],
    )

    cli.main()

    req = captured["request"]
    assert req.species_names == ("dichlorosilane", "germane", "hcl", "hydrogen")
    assert req.target_deposit == "SiGe"
    assert req.save_model_package is True
    assert req.model_package_path == "data/processed/sige_model_package.json"
    assert req.fit_residual_nn is False


def test_cli_data_add_accepts_generic_doped_sigec(monkeypatch):
    captured = {}

    def fake_register_experiment(cfg, request):
        captured["request"] = request
        return {"kind": request.kind.value, "chem_class": request.chem_class.value}

    monkeypatch.setattr(cli, "register_experiment", fake_register_experiment)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "chem-ml",
            "data",
            "add",
            "--kind",
            "scalar",
            "--csv",
            "sigecx.csv",
            "--reactor",
            "XYZ_tool_1",
            "--chem-class",
            "SiGeC:X",
            "--tag",
            "sigecx_batch",
        ],
    )

    cli.main()

    assert captured["request"].kind == DataKind.SCALAR
    assert captured["request"].chem_class == ChemClass.SIGEC_X
