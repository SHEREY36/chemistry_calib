"""Smoke tests for the grouped CLI workflow commands."""
import sys

from chem_ml.contracts import DataKind, TrainStrategy, TrainTarget, ValidationSuite
from chem_ml import cli


def test_cli_data_add_scalar_builds_register_request(monkeypatch, capsys):
    seen = {}

    def fake_register(cfg, request):
        seen["request"] = request
        return {"kind": request.kind.value, "tag": request.tag}

    monkeypatch.setattr(cli, "register_experiment", fake_register)
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
            "new.csv",
            "--reactor",
            "ASM_Epsilon",
            "--chem-class",
            "SiGe",
            "--tag",
            "batch_1",
        ],
    )

    cli.main()

    assert seen["request"].kind == DataKind.SCALAR
    assert seen["request"].csv_path == "new.csv"
    assert '"tag": "batch_1"' in capsys.readouterr().out


def test_cli_train_reactor_transfer_defaults_to_frozen_chemistry(monkeypatch, capsys):
    seen = {}

    def fake_train(cfg, request):
        seen["request"] = request
        return {"target": request.target.value, "strategy": request.strategy.value, "report": {"R2_Ge": 0.9}}

    monkeypatch.setattr(cli, "train", fake_train)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "chem-ml",
            "train",
            "--target",
            "reactor-transfer",
            "--csv",
            "reactor.csv",
            "--reactor",
            "XYZ_tool_1",
        ],
    )

    cli.main()

    assert seen["request"].target == TrainTarget.REACTOR_TRANSFER
    assert seen["request"].strategy == TrainStrategy.FROZEN_CHEMISTRY
    assert '"R2_Ge": 0.9' in capsys.readouterr().out


def test_cli_train_chemistry_warm_start_builds_request(monkeypatch):
    seen = {}

    def fake_train(cfg, request):
        seen["request"] = request
        return {"target": request.target.value, "strategy": request.strategy.value, "n_new_sige_rows": 1}

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
            "warm-start",
            "--csv",
            "new.csv",
            "--reactor",
            "ASM_Epsilon",
            "--chem-class",
            "SiGe",
            "--tag",
            "batch_2",
        ],
    )

    cli.main()

    assert seen["request"].target == TrainTarget.CHEMISTRY
    assert seen["request"].strategy == TrainStrategy.WARM_START
    assert seen["request"].tag == "batch_2"


def test_cli_validate_all_builds_request(monkeypatch):
    seen = {}

    def fake_validate(cfg, request):
        seen["request"] = request
        return {"suite": request.suite.value, "ok": True}

    monkeypatch.setattr(cli, "validate", fake_validate)
    monkeypatch.setattr(sys, "argv", ["chem-ml", "validate", "--suite", "all", "--write-report"])

    cli.main()

    assert seen["request"].suite == ValidationSuite.ALL
    assert seen["request"].write_report is True
