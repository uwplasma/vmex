from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from vmec_jax import cli
from vmec_jax.namelist import InData


def test_python_module_entrypoint_sets_warning_suppression_env(monkeypatch) -> None:
    for key in ("TF_CPP_MIN_LOG_LEVEL", "ABSL_MIN_LOG_LEVEL", "GLOG_minloglevel"):
        monkeypatch.delenv(key, raising=False)
    sys.modules.pop("vmec_jax.__main__", None)

    module = importlib.import_module("vmec_jax.__main__")

    assert os.environ["TF_CPP_MIN_LOG_LEVEL"] == "2"
    assert os.environ["ABSL_MIN_LOG_LEVEL"] == "2"
    assert os.environ["GLOG_minloglevel"] == "2"
    assert module.main is cli.main


def test_cli_case_and_wout_path_conventions(tmp_path: Path) -> None:
    assert cli._case_from_input(Path("input.circular_tokamak")) == "circular_tokamak"
    assert cli._case_from_input(Path("input_nfp2_QA")) == "nfp2_QA"
    assert cli._case_from_input(Path("custom_input.txt")) == "custom_input"

    input_path = tmp_path / "input.case"
    assert cli.resolve_wout_path(input_path=input_path, outdir=None, output=None) == tmp_path / "wout_case.nc"
    assert cli.resolve_wout_path(input_path=input_path, outdir=tmp_path / "out", output=None) == tmp_path / "out" / "wout_case.nc"
    assert cli.resolve_wout_path(input_path=input_path, outdir=None, output=tmp_path / "explicit.nc") == tmp_path / "explicit.nc"


def test_cli_jit_forces_parser_accepts_documented_values() -> None:
    assert cli._parse_jit_forces("auto") == "auto"
    assert cli._parse_jit_forces("") == "auto"
    assert cli._parse_jit_forces("yes") is True
    assert cli._parse_jit_forces("0") is False

    with pytest.raises(ValueError, match="Invalid --jit-forces"):
        cli._parse_jit_forces("maybe")


def test_cli_plot_mode_dispatches_without_solver(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[Path, Path]] = []
    wout = tmp_path / "wout_case.nc"
    wout.write_text("placeholder")

    def fake_plot_wout(path: Path, *, outdir: Path):
        calls.append((path, outdir))

    monkeypatch.setitem(sys.modules, "vmec_jax.plotting", SimpleNamespace(plot_wout=fake_plot_wout))

    assert cli.main(["--plot", str(wout), "--outdir", str(tmp_path / "plots")]) == 0
    assert calls == [(wout.resolve(), (tmp_path / "plots").resolve())]


def test_cli_errors_for_missing_plot_or_input(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as no_input:
        cli.main([])
    assert no_input.value.code == 2

    with pytest.raises(SystemExit) as missing_plot:
        cli.main(["--plot", str(tmp_path / "missing.nc")])
    assert missing_plot.value.code == 2


def test_cli_run_mode_wires_solver_kwargs_and_output(monkeypatch, tmp_path: Path) -> None:
    input_path = tmp_path / "input.case"
    input_path.write_text("&INDATA\n  NITER = 12\n/\n")
    outdir = tmp_path / "out"
    indata = InData(scalars={"NITER": 12}, indexed={})
    calls = {}

    monkeypatch.setattr(cli, "read_indata", lambda path: indata)
    monkeypatch.setattr(cli, "default_non_autodiff_solver_policy", lambda _: ("default", True))

    def fake_run_fixed_boundary(path: str, **kwargs):
        calls["path"] = path
        calls["kwargs"] = kwargs
        return SimpleNamespace(state=SimpleNamespace(Rcos=0.0))

    def fake_write_wout(path: Path, run, *, include_fsq: bool):
        calls["wout_path"] = path
        calls["include_fsq"] = include_fsq
        path.write_text("wout")

    monkeypatch.setattr(cli, "run_fixed_boundary", fake_run_fixed_boundary)
    monkeypatch.setattr(cli, "write_wout_from_fixed_boundary_run", fake_write_wout)

    rc = cli.main(
        [
            str(input_path),
            "--outdir",
            str(outdir),
            "--no-use-input-niter",
            "--quiet",
            "--jit-forces",
            "false",
        ]
    )

    assert rc == 0
    assert calls["path"] == str(input_path.resolve())
    assert calls["wout_path"] == outdir.resolve() / "wout_case.nc"
    assert calls["include_fsq"] is True
    kwargs = calls["kwargs"]
    assert kwargs["max_iter"] == 12
    assert kwargs["multigrid_use_input_niter"] is False
    assert kwargs["verbose"] is False
    assert kwargs["jit_forces"] is False
    assert kwargs["solver_mode"] == "default"
    assert kwargs["performance_mode"] is True
    assert kwargs["cli_fixed_boundary_mode"] is True
    assert (outdir / "wout_case.nc").read_text() == "wout"


def test_cli_rejects_conflicting_solver_flags(tmp_path: Path) -> None:
    input_path = tmp_path / "input.case"
    input_path.write_text("&INDATA\n/\n")

    with pytest.raises(SystemExit) as conflict:
        cli.main([str(input_path), "--parity", "--fast"])
    assert conflict.value.code == 2

    with pytest.raises(SystemExit) as mode_conflict:
        cli.main([str(input_path), "--solver-mode", "default", "--fast"])
    assert mode_conflict.value.code == 2
