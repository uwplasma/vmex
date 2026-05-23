import importlib.util
import json
import os
from pathlib import Path

import vmec_jax.discrete_adjoint as da


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools" / "diagnostics" / "gpu_cpu_performance_matrix.py"
FIXED_TOOL_PATH = REPO_ROOT / "tools" / "diagnostics" / "profile_fixed_boundary.py"
QI_TOOL_PATH = REPO_ROOT / "tools" / "diagnostics" / "profile_qi_boozer_gpu.py"
COMPARE_TOOL_PATH = REPO_ROOT / "tools" / "diagnostics" / "compare_profile_reports.py"
EXACT_TOOL_PATH = REPO_ROOT / "tools" / "diagnostics" / "profile_exact_optimizer.py"


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


def _load_compare_tool():
    spec = importlib.util.spec_from_file_location("compare_profile_reports", COMPARE_TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_exact_tool():
    spec = importlib.util.spec_from_file_location("profile_exact_optimizer", EXACT_TOOL_PATH)
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
            "--sync-replay-timing",
            "--jvp-only-exact-tape",
            "--jvp-only-basepoint-carries",
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
    assert env["VMEC_JAX_OPT_SYNC_REPLAY_TIMING"] == "1"
    assert env["VMEC_JAX_OPT_JVP_ONLY_EXACT_TAPE"] == "1"
    assert env["VMEC_JAX_JVP_ONLY_EXACT_TAPE_BASEPOINT_CARRIES"] == "1"
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


def test_performance_matrix_trial_scan_env_and_summary():
    tool = _load_tool()

    args_on = tool._build_parser().parse_args(["--backend", "auto", "--trial-scan", "on"])
    env_on = tool.child_env(backend="auto", args=args_on, base_env={"PYTHONPATH": ""})
    assert env_on["VMEC_JAX_OPT_TRIAL_SCAN"] == "1"
    assert tool.env_summary(env_on)["VMEC_JAX_OPT_TRIAL_SCAN"] == "1"

    args_off = tool._build_parser().parse_args(["--backend", "auto", "--trial-scan", "off"])
    env_off = tool.child_env(backend="auto", args=args_off, base_env={"PYTHONPATH": ""})
    assert env_off["VMEC_JAX_OPT_TRIAL_SCAN"] == "0"
    assert tool.env_summary(env_off)["VMEC_JAX_OPT_TRIAL_SCAN"] == "0"

    args_legacy = tool._build_parser().parse_args(
        ["--backend", "auto", "--trial-scan", "off", "--trial-use-scan"]
    )
    env_legacy = tool.child_env(backend="auto", args=args_legacy, base_env={"PYTHONPATH": ""})
    assert env_legacy["VMEC_JAX_OPT_TRIAL_SCAN"] == "1"


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
    assert "--use-scan" not in command
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
            "--trial-scan",
            "off",
            "--trace",
            "--device-memory-profile",
            "--vmec-timing-detail",
            "--sync-replay-timing",
            "--jvp-only-exact-tape",
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
    assert command[command.index("--trial-scan") + 1] == "off"
    assert "--vmec-timing-detail" in command
    assert "--sync-replay-timing" in command
    assert "--jvp-only-exact-tape" in command
    assert command[command.index("--trace-outdir") + 1] == str(trace)
    assert command[command.index("--device-memory-profile-out") + 1] == str(memory)


def test_exact_optimizer_profiler_trial_scan_flag_normalization(monkeypatch):
    exact_tool = _load_exact_tool()

    legacy = exact_tool._normalize_callback_args(
        exact_tool._parse_args(["--trial-use-scan", "--method", "auto"])
    )
    explicit_off = exact_tool._normalize_callback_args(exact_tool._parse_args(["--trial-scan", "off"]))

    assert legacy.trial_scan == "on"
    assert legacy.method == "auto"
    assert explicit_off.trial_scan == "off"


def test_exact_callback_summary_preserves_cold_tangent_replay_and_scan_trial_buckets():
    compare = _load_compare_tool()
    payload = {
        "report_kind": "exact_optimizer_callback_profile",
        "problem": "qh",
        "max_mode": 2,
        "callback": "jacobian",
        "solver_device_resolved": "gpu",
        "total_wall_time_s": 10.0,
        "samples": [{"repeat": 0, "wall_time_s": 10.0}],
        "profile": {
            "jacobian_total": {"count": 1, "wall_time_s": 9.5},
            "exact_solve_with_tape_total": {"count": 1, "wall_time_s": 4.0},
            "exact_solve_with_tape_jvp_only_total": {"count": 1, "wall_time_s": 3.5},
            "exact_tape_build": {"count": 1, "wall_time_s": 3.0},
            "exact_tape_build_jvp_only": {"count": 1, "wall_time_s": 2.8},
            "exact_tape_build_unattributed": {"count": 1, "wall_time_s": 0.7},
            "jacobian_initial_tangents": {"count": 1, "wall_time_s": 1.4},
            "jacobian_initial_tangents_linearize": {"count": 1, "wall_time_s": 0.3},
            "jacobian_initial_tangents_vmap_dispatch": {"count": 1, "wall_time_s": 0.2},
            "jacobian_initial_tangents_vmap_ready": {"count": 1, "wall_time_s": 0.9},
            "jacobian_tape_replay": {"count": 1, "wall_time_s": 2.1},
            "jacobian_tape_replay_dispatch": {"count": 1, "wall_time_s": 0.2},
            "jacobian_tape_replay_ready": {"count": 1, "wall_time_s": 1.9},
            "jacobian_residual_tangents": {"count": 1, "wall_time_s": 0.8},
            "exact_tape_solver_compute_forces": {"count": 1, "wall_time_s": 1.7},
            "exact_tape_solver_compute_forces_first": {"count": 1, "wall_time_s": 0.9},
            "exact_tape_solver_compute_forces_rest": {"count": 1, "wall_time_s": 0.8},
            "trial_solver_scan_total": {"count": 1, "wall_time_s": 1.2},
            "trial_solver_scan_setup": {"count": 1, "wall_time_s": 0.1},
            "trial_solver_scan_run_setup": {"count": 1, "wall_time_s": 0.15},
            "trial_solver_scan_device_run": {"count": 1, "wall_time_s": 0.75},
            "trial_solver_scan_device_dispatch": {"count": 1, "wall_time_s": 0.05},
            "trial_solver_scan_device_ready": {"count": 1, "wall_time_s": 0.7},
            "trial_solver_scan_host_materialize": {"count": 1, "wall_time_s": 0.1},
            "trial_solver_scan_postprocess": {"count": 1, "wall_time_s": 0.1},
        },
    }

    summary = compare.summarize_payload(payload, label="cold-gpu", top_profile=3)
    metrics = summary["metrics"]

    assert metrics["replay_time_s"] == 2.1
    assert metrics["exact_solve_with_tape_jvp_only_s"] == 3.5
    assert metrics["exact_tape_build_jvp_only_s"] == 2.8
    assert metrics["accepted_replay_dispatch_s"] == 0.2
    assert metrics["accepted_replay_ready_s"] == 1.9
    assert metrics["initial_tangents_s"] == 1.4
    assert metrics["initial_tangents_linearize_s"] == 0.3
    assert metrics["initial_tangents_vmap_dispatch_s"] == 0.2
    assert metrics["initial_tangents_vmap_ready_s"] == 0.9
    assert metrics["residual_tangents_s"] == 0.8
    assert metrics["exact_tape_solver_compute_forces_s"] == 1.7
    assert metrics["exact_tape_solver_compute_forces_first_s"] == 0.9
    assert metrics["exact_tape_solver_compute_forces_rest_s"] == 0.8
    assert metrics["trial_solver_scan_total_s"] == 1.2
    assert metrics["trial_solver_scan_run_setup_s"] == 0.15
    assert metrics["trial_solver_scan_device_run_s"] == 0.75
    assert metrics["trial_solver_scan_device_dispatch_s"] == 0.05
    assert metrics["trial_solver_scan_device_ready_s"] == 0.7
    assert summary["exact_optimizer_patch_target"]["name"] == "jacobian_tape_replay_ready"


def test_performance_matrix_loaded_summary_adds_scan_cache_and_jvp_sections(tmp_path):
    tool = _load_tool()
    report = tmp_path / "profile.json"
    report.write_text(
        json.dumps(
            {
                "report_kind": "exact_optimizer_callback_profile",
                "problem": "qh",
                "max_mode": 2,
                "callback": "jacobian",
                "solver_device_resolved": "cpu",
                "total_wall_time_s": 12.0,
                "jvp_only_exact_tape": True,
                "jvp_only_basepoint_carries": True,
                "samples": [{"repeat": 0, "wall_time_s": 12.0}],
                "replay_scan_cache_diagnostics": {
                    "replay_checkpoint_scan_cache_hit_count": 2,
                    "replay_checkpoint_scan_cache_miss_count": 1,
                    "replay_checkpoint_scan_cache_lookup_s": 0.01,
                    "replay_checkpoint_scan_cache_build_s": 0.2,
                    "replay_dynamic_basepoint_scan_cache_hit_count": 3,
                    "replay_dynamic_basepoint_scan_cache_miss_count": 4,
                    "replay_dynamic_basepoint_scan_cache_lookup_s": 0.02,
                    "replay_dynamic_basepoint_scan_cache_build_s": 0.3,
                },
                "profile": {
                    "exact_solve_with_tape_jvp_only_total": {"count": 1, "wall_time_s": 5.0},
                    "exact_tape_build_jvp_only": {"count": 1, "wall_time_s": 4.5},
                    "jacobian_initial_tangents_vmap_dispatch": {"count": 1, "wall_time_s": 0.4},
                    "jacobian_initial_tangents_vmap_ready": {"count": 1, "wall_time_s": 0.8},
                    "jacobian_projected_replay_total": {"count": 2, "wall_time_s": 3.0},
                    "jacobian_projected_tape_replay_dispatch": {"count": 2, "wall_time_s": 0.5},
                    "jacobian_projected_replay_residual_tangents": {"count": 2, "wall_time_s": 2.5},
                    "trial_solver_scan_total": {"count": 1, "wall_time_s": 1.2},
                    "trial_solver_scan_runner_cache_lookup": {"count": 1, "wall_time_s": 0.03},
                    "trial_solver_scan_runner_cache_build": {"count": 1, "wall_time_s": 0.2},
                    "trial_solver_scan_runner_cache_hit_count": {"count": 1, "wall_time_s": 2.0},
                    "trial_solver_scan_runner_cache_miss_count": {"count": 1, "wall_time_s": 1.0},
                    "trial_solver_scan_runner_cache_hit_ready": {"count": 1, "wall_time_s": 0.6},
                    "trial_solver_scan_runner_cache_miss_ready": {"count": 1, "wall_time_s": 0.9},
                },
            }
        ),
        encoding="utf-8",
    )

    summary = tool._load_profile_summary(report, label="cpu")
    assert summary is not None

    scan_cache = summary["matrix_scan_cache_summary"]
    projected_jvp = summary["matrix_projected_replay_jvp_summary"]

    assert scan_cache["trial"]["total_s"] == 1.2
    assert scan_cache["trial"]["cache"]["hit_count"] == 2
    assert scan_cache["trial"]["cache"]["miss_count"] == 1
    assert scan_cache["trial"]["cache"]["hit_ready_s"] == 0.6
    assert scan_cache["trial"]["cache"]["miss_ready_s"] == 0.9
    assert scan_cache["replay"]["hit_count"] == 5
    assert scan_cache["replay"]["miss_count"] == 5
    assert scan_cache["replay"]["lookup_s"] == 0.03
    assert scan_cache["replay"]["build_s"] == 0.5
    assert projected_jvp["jvp"]["exact_tape"] is True
    assert projected_jvp["jvp"]["basepoint_carries"] is True
    assert projected_jvp["jvp"]["exact_solve_with_tape_s"] == 5.0
    assert projected_jvp["jvp"]["tape_build_s"] == 4.5
    assert projected_jvp["projected_replay"]["total_s"] == 3.0
    assert projected_jvp["projected_replay"]["dispatch_s"] == 0.5
    assert projected_jvp["projected_replay"]["residual_tangents_s"] == 2.5


def test_exact_optimizer_point_cache_clear_preserves_initial_tangent_cache():
    exact_tool = _load_exact_tool()

    class Optimizer:
        def __init__(self):
            self._exact_cache = {"point": object()}
            self._exact_state_cache = {"point": object()}
            self._exact_state_key_by_id = {1: "point"}
            self._exact_residual_cache = {"point": object()}
            self._exact_jacobian_cache = {"point": object()}
            self._trial_residual_cache = {"point": object()}
            self._initial_state_cache = {"point": object()}
            self._initial_tangent_cache = {"branch": object()}
            self._last_jacobian_residual = object()

    opt = Optimizer()
    exact_tool._clear_optimizer_point_caches(opt)

    assert opt._exact_cache == {}
    assert opt._exact_state_cache == {}
    assert opt._exact_state_key_by_id == {}
    assert opt._exact_residual_cache == {}
    assert opt._exact_jacobian_cache == {}
    assert opt._trial_residual_cache == {}
    assert opt._initial_state_cache == {}
    assert set(opt._initial_tangent_cache) == {"branch"}
    assert opt._last_jacobian_residual is None


def test_exact_callback_summary_uses_split_replay_when_total_bucket_is_absent():
    compare = _load_compare_tool()
    payload = {
        "report_kind": "exact_optimizer_callback_profile",
        "total_wall_time_s": 3.0,
        "samples": [{"repeat": 0, "wall_time_s": 3.0}],
        "profile": {
            "jacobian_tape_replay_dispatch": {"count": 1, "wall_time_s": 0.4},
            "jacobian_tape_replay_ready": {"count": 1, "wall_time_s": 0.6},
        },
    }

    summary = compare.summarize_payload(payload, label="split-only", top_profile=2)

    assert summary["metrics"]["replay_time_s"] == 1.0
    assert summary["metrics"]["accepted_replay_dispatch_s"] == 0.4
    assert summary["metrics"]["accepted_replay_ready_s"] == 0.6
    assert summary["metrics"]["accepted_point_replay_count"] is None


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
