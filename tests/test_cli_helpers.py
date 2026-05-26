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


def test_cli_wout_io_warmup_is_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("VMEC_JAX_WOUT_IO_WARMUP", raising=False)
    assert cli._start_wout_io_warmup() is None

    calls: list[str] = []
    real_import_module = importlib.import_module

    def fake_import_module(name: str):
        calls.append(name)
        return real_import_module("types")

    monkeypatch.setenv("VMEC_JAX_WOUT_IO_WARMUP", "1")
    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    thread = cli._start_wout_io_warmup()
    assert thread is not None
    thread.join(timeout=5)
    assert calls == ["netCDF4"]


def test_cli_wout_io_warmup_swallows_import_and_thread_start_failures(monkeypatch) -> None:
    monkeypatch.setenv("VMEC_JAX_WOUT_IO_WARMUP", "yes")

    calls: list[str] = []

    def fake_import_module(name: str):
        calls.append(name)
        raise RuntimeError("netcdf unavailable")

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    thread = cli._start_wout_io_warmup()
    assert thread is not None
    thread.join(timeout=5)
    assert calls == ["netCDF4"]

    class BrokenThread:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("threading unavailable")

    monkeypatch.setattr(importlib, "import_module", lambda name: object())
    monkeypatch.setattr("threading.Thread", BrokenThread)

    assert cli._start_wout_io_warmup() is None


def test_cli_plot_mode_dispatches_without_solver(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[Path, Path]] = []
    wout = tmp_path / "wout_case.nc"
    wout.write_text("placeholder")

    def fake_plot_wout(path: Path, *, outdir: Path):
        calls.append((path, outdir))

    monkeypatch.setitem(sys.modules, "vmec_jax.plotting", SimpleNamespace(plot_wout=fake_plot_wout))

    assert cli.main(["--plot", str(wout), "--outdir", str(tmp_path / "plots")]) == 0
    assert calls == [(wout.resolve(), (tmp_path / "plots").resolve())]


def test_cli_test_mode_copies_packaged_input_solves_and_plots(monkeypatch, tmp_path: Path, capsys) -> None:
    outdir = tmp_path / "demo"
    calls = {}
    indata = InData(scalars={"NITER": 1}, indexed={})

    monkeypatch.setattr(cli, "read_indata", lambda _path: indata)
    monkeypatch.setattr(cli, "default_non_autodiff_solver_policy", lambda _indata: ("default", True))
    monkeypatch.setattr(cli, "_default_use_scan_for_backend", lambda _indata, _backend, _mode: False)

    def fake_run_fixed_boundary(path: str, **kwargs):
        calls["run_path"] = Path(path)
        calls["run_kwargs"] = kwargs
        return SimpleNamespace(state=SimpleNamespace(Rcos=0.0))

    def fake_write_wout(path: Path, run, *, include_fsq: bool):
        calls["wout_path"] = path
        calls["include_fsq"] = include_fsq
        path.write_text("wout")

    def fake_plot_wout(path: Path, *, outdir: Path):
        calls["plot_path"] = path
        calls["plot_outdir"] = outdir

    monkeypatch.setattr(cli, "run_fixed_boundary", fake_run_fixed_boundary)
    monkeypatch.setattr(cli, "write_wout_from_fixed_boundary_run", fake_write_wout)
    monkeypatch.setitem(sys.modules, "vmec_jax.plotting", SimpleNamespace(plot_wout=fake_plot_wout))

    assert cli.main(["--test", "--outdir", str(outdir), "--quiet"]) == 0

    input_path = outdir / "input.nfp4_QH_warm_start"
    wout_path = outdir / "wout_nfp4_QH_warm_start.nc"
    assert input_path.exists()
    assert "&INDATA" in input_path.read_text()
    assert wout_path.read_text() == "wout"
    assert calls["run_path"] == input_path.resolve()
    assert calls["run_kwargs"]["verbose"] is False
    assert calls["wout_path"] == wout_path.resolve()
    assert calls["include_fsq"] is True
    assert calls["plot_path"] == wout_path.resolve()
    assert calls["plot_outdir"] == (outdir / "figures").resolve()
    output = capsys.readouterr().out
    assert "Equivalent manual command" in output
    assert "Equivalent manual plotting command" in output


def test_cli_errors_for_missing_plot_or_input(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as no_input:
        cli.main([])
    assert no_input.value.code == 2

    with pytest.raises(SystemExit) as missing_plot:
        cli.main(["--plot", str(tmp_path / "missing.nc")])
    assert missing_plot.value.code == 2

    with pytest.raises(SystemExit) as test_with_input:
        cli.main(["--test", str(tmp_path / "input.case")])
    assert test_with_input.value.code == 2


def test_cli_errors_for_invalid_jit_and_conflicting_solver_mode_flags(tmp_path: Path) -> None:
    input_path = tmp_path / "input.case"
    input_path.write_text("&INDATA\n/\n")

    with pytest.raises(SystemExit) as invalid_jit:
        cli.main([str(input_path), "--jit-forces", "not-a-bool"])
    assert invalid_jit.value.code == 2

    with pytest.raises(SystemExit) as parity_fast:
        cli.main([str(input_path), "--parity", "--fast"])
    assert parity_fast.value.code == 2

    with pytest.raises(SystemExit) as mode_alias_conflict:
        cli.main([str(input_path), "--solver-mode", "parity", "--fast"])
    assert mode_alias_conflict.value.code == 2


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
    assert kwargs["use_scan"] is False
    assert kwargs["cli_fixed_boundary_mode"] is True
    assert (outdir / "wout_case.nc").read_text() == "wout"


def test_cli_run_mode_uses_cpu_default_policy_and_explicit_output(monkeypatch, tmp_path: Path) -> None:
    input_path = tmp_path / "input.case"
    input_path.write_text("&INDATA\n  NITER = 12\n/\n")
    output = tmp_path / "custom_wout.nc"
    indata = InData(scalars={"NITER": 12}, indexed={})
    calls = {}

    monkeypatch.setattr(cli, "read_indata", lambda path: indata)
    monkeypatch.setattr(
        cli,
        "_default_non_autodiff_solver_policy_for_backend",
        lambda _indata, backend: ("default", False) if backend == "cpu" else ("accelerated", True),
    )

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
            "--solver-device",
            "cpu",
            "--output",
            str(output),
            "--max-iter",
            "3",
        ]
    )

    assert rc == 0
    assert calls["wout_path"] == output.resolve()
    assert calls["kwargs"]["solver_device"] == "cpu"
    assert calls["kwargs"]["solver_mode"] == "default"
    assert calls["kwargs"]["performance_mode"] is False
    assert calls["kwargs"]["use_scan"] is False
    assert calls["kwargs"]["max_iter"] == 3


def test_cli_cpu_default_keeps_host_loop_when_scan_policy_is_true(monkeypatch, tmp_path: Path) -> None:
    input_path = tmp_path / "input.case"
    input_path.write_text("&INDATA\n  NITER = 12\n/\n")
    indata = InData(scalars={"NITER": 12}, indexed={})
    calls = {}

    monkeypatch.setattr(cli, "read_indata", lambda path: indata)
    monkeypatch.setattr(
        cli,
        "_default_non_autodiff_solver_policy_for_backend",
        lambda _indata, backend: ("accelerated", True) if backend == "cpu" else ("default", False),
    )
    monkeypatch.setattr(cli, "_default_use_scan_for_backend", lambda _indata, _backend, _mode: True)

    def fake_run_fixed_boundary(path: str, **kwargs):
        calls["kwargs"] = kwargs
        return SimpleNamespace(state=SimpleNamespace(Rcos=0.0))

    monkeypatch.setattr(cli, "run_fixed_boundary", fake_run_fixed_boundary)
    monkeypatch.setattr(cli, "write_wout_from_fixed_boundary_run", lambda path, run, *, include_fsq: path.write_text("wout"))

    assert cli.main([str(input_path), "--solver-device", "cpu"]) == 0
    assert calls["kwargs"]["verbose"] is True
    assert calls["kwargs"]["use_scan"] is False

    assert cli.main([str(input_path), "--solver-device", "cpu", "--quiet"]) == 0
    assert calls["kwargs"]["verbose"] is False
    assert calls["kwargs"]["use_scan"] is False


def test_cli_joins_wout_warmup_best_effort(monkeypatch, tmp_path: Path) -> None:
    input_path = tmp_path / "input.case"
    input_path.write_text("&INDATA\n  NITER = 1\n/\n")
    indata = InData(scalars={"NITER": 1}, indexed={})
    joined = {"called": False}

    class BrokenJoinThread:
        def join(self):
            joined["called"] = True
            raise RuntimeError("join failed")

    monkeypatch.setattr(cli, "read_indata", lambda path: indata)
    monkeypatch.setattr(cli, "_start_wout_io_warmup", lambda: BrokenJoinThread())
    monkeypatch.setattr(cli, "default_non_autodiff_solver_policy", lambda _: ("default", True))
    monkeypatch.setattr(cli, "run_fixed_boundary", lambda path, **kwargs: SimpleNamespace(state=SimpleNamespace(Rcos=0.0)))
    monkeypatch.setattr(cli, "write_wout_from_fixed_boundary_run", lambda path, run, *, include_fsq: path.write_text("wout"))

    assert cli.main([str(input_path), "--quiet"]) == 0
    assert joined["called"] is True


def test_cli_run_mode_uses_gpu_default_policy(monkeypatch, tmp_path: Path) -> None:
    input_path = tmp_path / "input.case"
    input_path.write_text("&INDATA\n/\n")
    indata = InData(scalars={}, indexed={})
    calls = {}

    monkeypatch.setattr(cli, "read_indata", lambda path: indata)
    monkeypatch.setattr(
        cli,
        "_default_non_autodiff_solver_policy_for_backend",
        lambda _indata, backend: ("accelerated", True) if backend == "gpu" else ("default", False),
    )

    def fake_run_fixed_boundary(path: str, **kwargs):
        calls["kwargs"] = kwargs
        return SimpleNamespace(state=SimpleNamespace(Rcos=0.0))

    monkeypatch.setattr(cli, "run_fixed_boundary", fake_run_fixed_boundary)
    monkeypatch.setattr(cli, "write_wout_from_fixed_boundary_run", lambda path, run, *, include_fsq: path.write_text("wout"))

    assert cli.main([str(input_path), "--solver-device", "gpu"]) == 0
    assert calls["kwargs"]["solver_device"] == "gpu"
    assert calls["kwargs"]["solver_mode"] == "accelerated"
    assert calls["kwargs"]["performance_mode"] is True
    assert calls["kwargs"]["use_scan"] is True


def test_cli_run_mode_explicit_solver_flags_and_vmecpp_restart(monkeypatch, tmp_path: Path) -> None:
    input_path = tmp_path / "input.case"
    input_path.write_text("&INDATA\n/\n")
    indata = InData(scalars={}, indexed={})
    calls = {}

    monkeypatch.setattr(cli, "read_indata", lambda path: indata)

    def fake_run_fixed_boundary(path: str, **kwargs):
        calls["kwargs"] = kwargs
        return SimpleNamespace(state=SimpleNamespace(Rcos=0.0))

    monkeypatch.setattr(cli, "run_fixed_boundary", fake_run_fixed_boundary)
    monkeypatch.setattr(cli, "write_wout_from_fixed_boundary_run", lambda path, run, *, include_fsq: path.write_text("wout"))

    assert cli.main([str(input_path), "--parity", "--vmecpp-restart"]) == 0
    assert calls["kwargs"]["solver_mode"] == "parity"
    assert calls["kwargs"]["performance_mode"] is False
    assert calls["kwargs"]["vmecpp_restart"] is True

    assert cli.main([str(input_path), "--fast", "--no-vmecpp-restart"]) == 0
    assert calls["kwargs"]["solver_mode"] == "default"
    assert calls["kwargs"]["performance_mode"] is True
    assert calls["kwargs"]["vmecpp_restart"] is False


def test_cli_profile_hooks_are_best_effort(monkeypatch, tmp_path: Path) -> None:
    input_path = tmp_path / "input.case"
    input_path.write_text("&INDATA\n/\n")
    profile_dir = tmp_path / "profile"
    indata = InData(scalars={}, indexed={})
    calls: list[tuple[str, object]] = []

    class FakeProfiler:
        @staticmethod
        def start_server(port: int):
            calls.append(("start_server", port))
            return object()

        @staticmethod
        def stop_server():
            calls.append(("stop_server", None))

        @staticmethod
        def start_trace(path: str, *, create_perfetto_trace: bool):
            calls.append(("start_trace", (Path(path), create_perfetto_trace)))

        @staticmethod
        def stop_trace():
            calls.append(("stop_trace", None))

    fake_jax = SimpleNamespace(
        profiler=FakeProfiler,
        block_until_ready=lambda value: calls.append(("block_until_ready", value)),
    )

    monkeypatch.setitem(sys.modules, "jax", fake_jax)
    monkeypatch.setenv("VMEC_JAX_PROFILE_DIR", str(profile_dir))
    monkeypatch.setenv("VMEC_JAX_PROFILE_SERVER", "1")
    monkeypatch.setenv("VMEC_JAX_PROFILE_SERVER_PORT", "12345")
    monkeypatch.setenv("VMEC_JAX_PROFILE_PERFETTO", "0")
    monkeypatch.setattr(cli, "read_indata", lambda path: indata)
    monkeypatch.setattr(cli, "default_non_autodiff_solver_policy", lambda _: ("default", True))
    monkeypatch.setattr(
        cli,
        "run_fixed_boundary",
        lambda path, **kwargs: SimpleNamespace(state=SimpleNamespace(Rcos="ready")),
    )
    monkeypatch.setattr(cli, "write_wout_from_fixed_boundary_run", lambda path, run, *, include_fsq: path.write_text("wout"))

    assert cli.main([str(input_path)]) == 0

    assert ("start_server", 12345) in calls
    assert ("start_trace", (profile_dir.resolve(), False)) in calls
    assert ("block_until_ready", "ready") in calls
    assert ("stop_trace", None) in calls
    assert ("stop_server", None) in calls


def test_cli_profile_hook_failures_do_not_block_solver(monkeypatch, tmp_path: Path) -> None:
    input_path = tmp_path / "input.case"
    input_path.write_text("&INDATA\n/\n")
    indata = InData(scalars={}, indexed={})

    class RaisingProfiler:
        @staticmethod
        def start_server(port: int):
            raise RuntimeError(f"cannot start {port}")

        @staticmethod
        def start_trace(path: str, *, create_perfetto_trace: bool):
            raise RuntimeError(f"cannot trace {path}")

    fake_jax = SimpleNamespace(
        profiler=RaisingProfiler,
        default_backend=lambda: (_ for _ in ()).throw(RuntimeError("no backend")),
    )
    calls = {}

    monkeypatch.setitem(sys.modules, "jax", fake_jax)
    monkeypatch.setenv("VMEC_JAX_PROFILE_DIR", str(tmp_path / "profile"))
    monkeypatch.setenv("VMEC_JAX_PROFILE_SERVER", "1")
    monkeypatch.setattr(cli, "read_indata", lambda path: indata)
    monkeypatch.setattr(cli, "default_non_autodiff_solver_policy", lambda _: ("default", True))

    def fake_run_fixed_boundary(path: str, **kwargs):
        calls["kwargs"] = kwargs
        return SimpleNamespace(state=SimpleNamespace(Rcos="not-awaited"))

    monkeypatch.setattr(cli, "run_fixed_boundary", fake_run_fixed_boundary)
    monkeypatch.setattr(cli, "write_wout_from_fixed_boundary_run", lambda path, run, *, include_fsq: path.write_text("wout"))

    assert cli.main([str(input_path)]) == 0
    assert calls["kwargs"]["use_scan"] is False


def test_cli_profile_cleanup_failures_are_best_effort(monkeypatch, tmp_path: Path) -> None:
    input_path = tmp_path / "input.case"
    input_path.write_text("&INDATA\n/\n")
    indata = InData(scalars={}, indexed={})
    calls: list[str] = []

    class CleanupRaisingProfiler:
        @staticmethod
        def start_server(port: int):
            calls.append(f"start_server:{port}")
            return object()

        @staticmethod
        def stop_server():
            calls.append("stop_server")
            raise RuntimeError("stop server failed")

        @staticmethod
        def start_trace(path: str, *, create_perfetto_trace: bool):
            calls.append(f"start_trace:{create_perfetto_trace}")

        @staticmethod
        def stop_trace():
            calls.append("stop_trace")
            raise RuntimeError("stop trace failed")

    def raise_block_until_ready(value):
        calls.append(f"block:{value}")
        raise RuntimeError("device wait failed")

    fake_jax = SimpleNamespace(
        profiler=CleanupRaisingProfiler,
        block_until_ready=raise_block_until_ready,
    )

    monkeypatch.setitem(sys.modules, "jax", fake_jax)
    monkeypatch.setenv("VMEC_JAX_PROFILE_DIR", str(tmp_path / "profile"))
    monkeypatch.setenv("VMEC_JAX_PROFILE_SERVER", "1")
    monkeypatch.setattr(cli, "read_indata", lambda path: indata)
    monkeypatch.setattr(cli, "default_non_autodiff_solver_policy", lambda _: ("default", True))
    monkeypatch.setattr(cli, "run_fixed_boundary", lambda path, **kwargs: SimpleNamespace(state=SimpleNamespace(Rcos="ready")))
    monkeypatch.setattr(cli, "write_wout_from_fixed_boundary_run", lambda path, run, *, include_fsq: path.write_text("wout"))

    assert cli.main([str(input_path)]) == 0
    assert "block:ready" in calls
    assert "stop_trace" in calls
    assert "stop_server" in calls


def test_cli_errors_for_missing_input_invalid_jit_and_read_failure(monkeypatch, tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as missing_input:
        cli.main([str(tmp_path / "missing.input")])
    assert missing_input.value.code == 2

    input_path = tmp_path / "input.case"
    input_path.write_text("&INDATA\n/\n")

    with pytest.raises(SystemExit) as invalid_jit:
        cli.main([str(input_path), "--jit-forces", "maybe"])
    assert invalid_jit.value.code == 2

    monkeypatch.setattr(cli, "read_indata", lambda path: (_ for _ in ()).throw(RuntimeError("bad input")))
    with pytest.raises(SystemExit) as read_failure:
        cli.main([str(input_path)])
    assert read_failure.value.code == 2


def test_cli_rejects_conflicting_solver_flags(tmp_path: Path) -> None:
    input_path = tmp_path / "input.case"
    input_path.write_text("&INDATA\n/\n")

    with pytest.raises(SystemExit) as conflict:
        cli.main([str(input_path), "--parity", "--fast"])
    assert conflict.value.code == 2

    with pytest.raises(SystemExit) as mode_conflict:
        cli.main([str(input_path), "--solver-mode", "default", "--fast"])
    assert mode_conflict.value.code == 2
