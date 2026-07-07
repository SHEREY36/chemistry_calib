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
