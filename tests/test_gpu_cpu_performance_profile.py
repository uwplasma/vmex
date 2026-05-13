import importlib.util
import json
import os
from pathlib import Path

import vmec_jax.discrete_adjoint as da


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools" / "diagnostics" / "gpu_cpu_performance_matrix.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("gpu_cpu_performance_matrix", TOOL_PATH)
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
            "--trace",
            "--device-memory-profile",
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
    assert command[command.index("--trace-outdir") + 1] == str(trace)
    assert command[command.index("--device-memory-profile-out") + 1] == str(memory)


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
