import importlib.util
import json
import os
from pathlib import Path

import vmec_jax.discrete_adjoint as da


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools" / "diagnostics" / "gpu_cpu_performance_matrix.py"
FIXED_TOOL_PATH = REPO_ROOT / "tools" / "diagnostics" / "profile_fixed_boundary.py"
QI_TOOL_PATH = REPO_ROOT / "tools" / "diagnostics" / "profile_qi_boozer_gpu.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("gpu_cpu_performance_matrix", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_fixed_tool():
    spec = importlib.util.spec_from_file_location("profile_fixed_boundary", FIXED_TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_qi_tool():
    spec = importlib.util.spec_from_file_location("profile_qi_boozer_gpu", QI_TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_replay_column_chunk_override_is_nonfatal_for_bad_env():
    assert da._replay_column_chunk_override(None) == (False, None)
    assert da._replay_column_chunk_override("auto") == (False, None)
    assert da._replay_column_chunk_override("bad") == (False, None)
    assert da._replay_column_chunk_override("0") == (True, None)
    assert da._replay_column_chunk_override("off") == (True, None)
    assert da._replay_column_chunk_override("8") == (True, 8)


def test_performance_matrix_auto_backend_preserves_jax_selection():
    tool = _load_tool()
    args = tool._build_parser().parse_args(
        [
            "--backend",
            "auto",
            "--x64",
            "inherit",
            "--replay-column-chunk",
            "16",
            "--dynamic-replay-mode",
            "whole_scan",
        ]
    )
    env = tool.child_env(
        backend="auto",
        args=args,
        base_env={
            "JAX_PLATFORM_NAME": "gpu",
            "JAX_PLATFORMS": "cuda",
            "PYTHONPATH": "/existing",
        },
    )

    assert env["JAX_PLATFORM_NAME"] == "gpu"
    assert env["JAX_PLATFORMS"] == "cuda"
    assert env["PYTHONPATH"].split(os.pathsep)[0] == str(REPO_ROOT)
    assert env["VMEC_JAX_REPLAY_COLUMN_CHUNK"] == "16"
    assert env["VMEC_JAX_DYNAMIC_REPLAY_MODE"] == "whole_scan"
    assert "JAX_ENABLE_X64" not in env


def test_performance_matrix_explicit_backend_env_overrides():
    tool = _load_tool()
    args = tool._build_parser().parse_args(["--backend", "gpu"])

    gpu_env = tool.child_env(
        backend="gpu",
        args=args,
        base_env={"JAX_PLATFORMS": "cpu", "PYTHONPATH": ""},
    )
    cpu_env = tool.child_env(
        backend="cpu",
        args=args,
        base_env={"JAX_PLATFORM_NAME": "gpu", "PYTHONPATH": ""},
    )

    assert gpu_env["JAX_PLATFORM_NAME"] == "gpu"
    assert "JAX_PLATFORMS" not in gpu_env
    assert cpu_env["JAX_PLATFORMS"] == "cpu"
    assert "JAX_PLATFORM_NAME" not in cpu_env
    assert gpu_env["JAX_ENABLE_X64"] == "1"
    assert cpu_env["JAX_ENABLE_X64"] == "1"


def test_performance_matrix_fixed_command_uses_backend_solver_device(tmp_path):
    tool = _load_tool()
    args = tool._build_parser().parse_args(
        [
            "--mode",
            "fixed-boundary",
            "--backend",
            "gpu",
            "--iters",
            "7",
            "--no-warmup",
            "--vmec-timing-detail",
        ]
    )
    report = tmp_path / "report.json"
    trace = tmp_path / "trace"

    command = tool.build_child_command(
        args=args,
        backend="gpu",
        report_path=report,
        trace_dir=trace,
    )

    assert str(tool.PROFILE_FIXED_BOUNDARY) in command
    assert command[command.index("--solver-device") + 1] == "gpu"
    assert command[command.index("--iters") + 1] == "7"
    assert "--simple-profile" in command
    assert "--use-scan" in command
    assert "--no-auto-cli-policy" in command
    assert "--no-multigrid" in command
    assert "--vmec-timing-detail" in command


def test_performance_matrix_exact_command_can_request_memory_profile(tmp_path):
    tool = _load_tool()
    args = tool._build_parser().parse_args(
        [
            "--mode",
            "exact-callback",
            "--backend",
            "cpu",
            "--problem",
            "qa",
            "--max-mode",
            "3",
            "--callback",
            "gradient",
            "--method",
            "scipy_matrix_free",
            "--scipy-tr-solver",
            "exact",
            "--lsmr-maxiter",
            "5",
            "--trial-use-scan",
            "--trace",
            "--device-memory-profile",
            "--vmec-timing-detail",
        ]
    )
    report = tmp_path / "report.json"
    trace = tmp_path / "trace"
    memory = tmp_path / "mem.prof"

    command = tool.build_child_command(
        args=args,
        backend="cpu",
        report_path=report,
        trace_dir=trace,
        memory_profile_path=memory,
    )

    assert str(tool.PROFILE_EXACT_OPTIMIZER) in command
    assert command[command.index("--solver-device") + 1] == "cpu"
    assert command[command.index("--problem") + 1] == "qa"
    assert command[command.index("--max-mode") + 1] == "3"
    assert command[command.index("--callback") + 1] == "gradient"
    assert command[command.index("--method") + 1] == "scipy_matrix_free"
    assert command[command.index("--scipy-tr-solver") + 1] == "exact"
    assert command[command.index("--lsmr-maxiter") + 1] == "5"
    assert "--trial-use-scan" in command
    assert "--vmec-timing-detail" in command
    assert command[command.index("--trace-outdir") + 1] == str(trace)
    assert command[command.index("--device-memory-profile-out") + 1] == str(memory)


def test_fixed_boundary_profiler_compacts_timing_diagnostics():
    fixed_tool = _load_fixed_tool()
    compact = fixed_tool._compact_diagnostics(
        {
            "solver_mode": "accelerated",
            "timing": {
                "iterations": 4,
                "compute_forces_s": 1.25,
                "preconditioner_s": 0.5,
                "update_s": 0.125,
            },
            "ignored": "large-history",
        }
    )

    assert compact == {
        "solver_mode": "accelerated",
        "timing": {
            "iterations": 4,
            "compute_forces_s": 1.25,
            "preconditioner_s": 0.5,
            "update_s": 0.125,
        },
    }


def test_fixed_boundary_profiler_reports_effective_jit_default():
    fixed_tool = _load_fixed_tool()
    args = fixed_tool.argparse.Namespace(
        input="input.test",
        iters=2,
        solver_mode="accelerated",
        solver_device="cpu",
        multigrid=False,
        use_input_niter=False,
        use_scan=True,
        jit_forces=False,
        no_jit_forces=False,
        auto_cli_policy=False,
        dynamic_scan=False,
    )

    class Result:
        diagnostics = {}
        n_iter = 2
        w_history = [1.0, 0.5]
        fsqr2_history = [0.1, 0.05]
        fsqz2_history = [0.2, 0.1]
        fsql2_history = [0.3, 0.15]

    class Run:
        result = Result()

    class JaxModule:
        __version__ = "test"

        @staticmethod
        def devices():
            return ["cpu0"]

        @staticmethod
        def default_backend():
            return "cpu"

    summary = fixed_tool._summarize_run(args=args, run=Run(), wall_time=1.25, jax_module=JaxModule)

    assert fixed_tool._effective_jit_forces(args) is True
    assert summary["args"]["jit_forces"] is True

    args.no_jit_forces = True
    assert fixed_tool._effective_jit_forces(args) is False


def test_performance_matrix_qi_boozer_command_uses_qi_profiler(tmp_path):
    tool = _load_tool()
    args = tool._build_parser().parse_args(
        [
            "--mode",
            "qi-boozer",
            "--backend",
            "gpu",
            "--repeat",
            "3",
            "--jit-booz",
            "--mboz",
            "8",
            "--nphi",
            "17",
        ]
    )
    report = tmp_path / "qi.json"
    trace = tmp_path / "trace"

    command = tool.build_child_command(
        args=args,
        backend="gpu",
        report_path=report,
        trace_dir=trace,
    )

    assert str(tool.PROFILE_QI_BOOZER) in command
    assert command[command.index("--input") + 1].endswith("examples/data/input.nfp2_QI")
    assert command[command.index("--solver-device") + 1] == "gpu"
    assert command[command.index("--output") + 1] == str(report)
    assert command[command.index("--repeat") + 1] == "3"
    assert command[command.index("--mboz") + 1] == "8"
    assert command[command.index("--nphi") + 1] == "17"
    assert "--jit-booz" in command


def test_qi_profiler_contamination_warnings_identify_gpu_request_on_cpu():
    qi_tool = _load_qi_tool()
    runtime = {
        "default_backend": "cpu",
        "active_gpu": False,
        "env": {"JAX_PLATFORM_NAME": "gpu", "JAX_PLATFORMS": "cpu,gpu"},
        "device_details": [{"platform": "cpu", "device_kind": "cpu"}],
    }

    warnings = qi_tool._contamination_warnings(runtime, requested_solver_device="gpu")

    assert any("Both JAX_PLATFORM_NAME and JAX_PLATFORMS" in warning for warning in warnings)
    assert any("GPU solver was requested" in warning for warning in warnings)
    assert any("default_backend=cpu" in warning for warning in warnings)


def test_performance_matrix_dry_run_writes_summary(tmp_path):
    tool = _load_tool()
    json_out = tmp_path / "matrix.json"
    outdir = tmp_path / "profiles"

    rc = tool.main(
        [
            "--dry-run",
            "--backend",
            "auto",
            "--mode",
            "fixed-boundary",
            "--outdir",
            str(outdir),
            "--json-out",
            str(json_out),
        ]
    )

    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["report_kind"] == "gpu_cpu_performance_matrix"
    assert payload["runs"][0]["backend"] == "auto"
    assert payload["runs"][0]["dry_run"] is True
    assert payload["runs"][0]["exit_code"] is None
    assert payload["runs"][0]["stdout_path"].endswith(".stdout.log")
    assert payload["runs"][0]["stderr_path"].endswith(".stderr.log")
