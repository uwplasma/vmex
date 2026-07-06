import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

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


def _fixed_profiler_args(fixed_tool, **overrides):
    values = {
        "input": "input.test",
        "iters": 2,
        "solver_mode": "accelerated",
        "solver_device": "cpu",
        "multigrid": False,
        "use_input_niter": False,
        "use_scan": None,
        "jit_forces": False,
        "no_jit_forces": False,
        "auto_cli_policy": False,
        "dynamic_scan": False,
        "scan_arg_summary": False,
        "require_scan": False,
        "require_no_scan": False,
        "phase_timing": {},
    }
    values.update(overrides)
    return fixed_tool.argparse.Namespace(**values)


def _fixed_profiler_run(diagnostics):
    class Result:
        n_iter = 2
        w_history = [1.0, 0.5]
        fsqr2_history = [0.1, 0.05]
        fsqz2_history = [0.2, 0.1]
        fsql2_history = [0.3, 0.15]

    Result.diagnostics = diagnostics

    class Run:
        result = Result()

    return Run()


class _FixedProfilerJaxModule:
    __version__ = "test"

    @staticmethod
    def devices():
        return ["cpu0"]

    @staticmethod
    def default_backend():
        return "cpu"


class _ExactProfilerJaxModule:
    __version__ = "test"

    @staticmethod
    def devices():
        return ["cpu0"]

    @staticmethod
    def default_backend():
        return "cpu"


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
            "--use-input-niter",
            "--vmec-timing-detail",
            "--scan-arg-summary",
        ]
    )
    report = tmp_path / "report.json"
    trace = tmp_path / "trace"

    assert tool.report_stem(args, "gpu") == "fixed_boundary_gpu_input_niter"

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
    assert "--use-input-niter" in command
    assert "--use-scan" not in command
    assert "--no-use-scan" not in command
    assert "--no-auto-cli-policy" in command
    assert "--no-multigrid" in command
    assert "--vmec-timing-detail" in command
    assert "--scan-arg-summary" in command


def test_performance_matrix_fixed_report_stem_uses_iter_budget_by_default():
    tool = _load_tool()
    args = tool._build_parser().parse_args(
        [
            "--mode",
            "fixed-boundary",
            "--backend",
            "cpu",
            "--iters",
            "9",
        ]
    )

    assert tool.report_stem(args, "cpu") == "fixed_boundary_cpu_iters9"


def test_performance_matrix_fixed_command_can_force_non_scan(tmp_path):
    tool = _load_tool()
    args = tool._build_parser().parse_args(
        [
            "--mode",
            "fixed-boundary",
            "--backend",
            "gpu",
            "--no-use-scan",
        ]
    )

    command = tool.build_child_command(
        args=args,
        backend="gpu",
        report_path=tmp_path / "report.json",
        trace_dir=tmp_path / "trace",
    )

    assert "--no-use-scan" in command
    assert "--use-scan" not in command


def test_fixed_boundary_profiler_scan_policy_is_tristate(monkeypatch):
    tool = _load_fixed_tool()

    monkeypatch.setattr(sys, "argv", ["profile_fixed_boundary.py", "--input", "input.test"])
    default = tool._parse_args()
    monkeypatch.setattr(sys, "argv", ["profile_fixed_boundary.py", "--input", "input.test", "--use-scan"])
    forced_scan = tool._parse_args()
    monkeypatch.setattr(sys, "argv", ["profile_fixed_boundary.py", "--input", "input.test", "--no-use-scan"])
    forced_nonscan = tool._parse_args()

    assert default.use_scan is None
    assert forced_scan.use_scan is True
    assert forced_nonscan.use_scan is False


def test_fixed_boundary_profiler_prefers_checkout_source():
    tool = _load_fixed_tool()

    assert str(REPO_ROOT) in sys.path
    assert tool._REPO_ROOT == REPO_ROOT


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
            "--scan-arg-summary",
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
    assert "--scan-arg-summary" in command
    assert "--sync-replay-timing" in command
    assert "--jvp-only-exact-tape" in command
    assert command[command.index("--trace-outdir") + 1] == str(trace)
    assert command[command.index("--device-memory-profile-out") + 1] == str(memory)


def test_performance_matrix_exact_command_supports_qp_problem(tmp_path):
    tool = _load_tool()
    args = tool._build_parser().parse_args(
        [
            "--mode",
            "exact-callback",
            "--backend",
            "gpu",
            "--problem",
            "qp",
            "--max-mode",
            "4",
            "--callback",
            "jacobian",
        ]
    )

    command = tool.build_child_command(
        args=args,
        backend="gpu",
        report_path=tmp_path / "qp_gpu.json",
        trace_dir=None,
        memory_profile_path=None,
    )

    assert command[command.index("--problem") + 1] == "qp"
    assert command[command.index("--solver-device") + 1] == "gpu"


def test_performance_matrix_exact_command_supports_auto_scalar_method(tmp_path):
    tool = _load_tool()
    args = tool._build_parser().parse_args(
        [
            "--mode",
            "exact-callback",
            "--backend",
            "gpu",
            "--problem",
            "qp",
            "--callback",
            "run",
            "--method",
            "auto_scalar",
        ]
    )

    command = tool.build_child_command(
        args=args,
        backend="gpu",
        report_path=tmp_path / "auto_scalar_gpu.json",
        trace_dir=None,
        memory_profile_path=None,
    )

    assert command[command.index("--method") + 1] == "auto_scalar"


def test_exact_optimizer_profiler_trial_scan_flag_normalization(monkeypatch):
    exact_tool = _load_exact_tool()

    legacy = exact_tool._normalize_callback_args(
        exact_tool._parse_args(["--trial-use-scan", "--method", "auto"])
    )
    explicit_off = exact_tool._normalize_callback_args(
        exact_tool._parse_args(["--trial-scan", "off", "--trial-state-only", "off"])
    )

    assert legacy.trial_scan == "on"
    assert legacy.method == "auto"
    assert explicit_off.trial_scan == "off"
    assert explicit_off.trial_state_only == "off"


def test_exact_optimizer_profiler_accepts_qp_problem():
    exact_tool = _load_exact_tool()

    args = exact_tool._parse_args(["--problem", "qp", "--callback", "jacobian"])

    assert args.problem == "qp"


def test_exact_profiler_runtime_info_records_device_timing():
    exact_tool = _load_exact_tool()
    phase_timing = {}

    runtime = exact_tool._jax_runtime_info(_ExactProfilerJaxModule, phase_timing=phase_timing)

    assert runtime["jax_version"] == "test"
    assert runtime["default_backend"] == "cpu"
    assert runtime["devices"] == ["cpu0"]
    assert "jax_devices_s" in runtime
    assert phase_timing["jax_devices_s"] == runtime["jax_devices_s"]


def test_exact_callback_payload_exposes_common_timing_aliases():
    exact_tool = _load_exact_tool()
    args = exact_tool._parse_args(["--callback", "jacobian", "--trial-scan", "off"])
    profile = {
        "jacobian_total": {"count": 1, "wall_time_s": 1.25, "mean_wall_time_s": 1.25},
        "trial_solver_scan_runner_cache_hit_count": {
            "count": 1,
            "wall_time_s": 2.0,
            "mean_wall_time_s": 2.0,
        },
    }
    phase_timing = {
        "process_to_main_s": 0.01,
        "vmec_jax_import_s": 0.02,
        "jax_devices_pre_run_s": 0.03,
        "jax_devices_s": 0.04,
    }

    payload = exact_tool._build_callback_payload(
        args=args,
        specs_count=3,
        solver_device_resolved="cpu",
        samples=[
            {
                "repeat": 0,
                "wall_time_s": 2.5,
                "exact_callback_metadata": {
                    "exact_replay_policy": {
                        "backend": "cpu",
                        "n_parameters": 3,
                    }
                },
            }
        ],
        profile=profile,
        cache_before={"total_entries": 0},
        cache_after={"total_entries": 0},
        rss_before_bytes=None,
        rss_after_bytes=None,
        total_wall_s=2.5,
        phase_timing=phase_timing,
        runtime={"default_backend": "cpu"},
    )

    assert payload["total_wall_time_s"] == 2.5
    assert payload["wall_time_s"] == 2.5
    assert payload["phase_timing"]["run_wall_s"] == 2.5
    assert payload["phase_timing"]["process_to_main_s"] == 0.01
    assert payload["phase_timing"]["vmec_jax_import_s"] == 0.02
    assert payload["phase_timing"]["jax_devices_pre_run_s"] == 0.03
    assert payload["phase_timing"]["jax_devices_s"] == 0.04
    assert payload["timing"]["jacobian_total"] == 1.25
    assert payload["timing"]["trial_solver_scan_runner_cache_hit_count"] == 2
    assert payload["exact_callback_metadata"]["exact_replay_policy"]["n_parameters"] == 3
    json.dumps(exact_tool._json_safe(payload))


def test_exact_run_history_payload_exposes_timing_aliases():
    exact_tool = _load_exact_tool()
    history = {
        "total_wall_time_s": 4.0,
        "profile": {
            "exact_tape_build": {"count": 1, "wall_time_s": 1.5, "mean_wall_time_s": 1.5},
            "jacobian_tape_replay": {"count": 2, "wall_time_s": 0.75, "mean_wall_time_s": 0.375},
        },
    }

    payload = exact_tool._history_payload_with_aliases(
        history,
        phase_timing={"process_to_main_s": 0.01, "vmec_jax_import_s": 0.02},
    )

    assert payload["total_wall_time_s"] == 4.0
    assert payload["wall_time_s"] == 4.0
    assert payload["phase_timing"]["run_wall_s"] == 4.0
    assert payload["phase_timing"]["vmec_jax_import_s"] == 0.02
    assert payload["timing"]["exact_tape_build"] == 1.5
    assert payload["timing"]["jacobian_tape_replay"] == 0.75
    json.dumps(exact_tool._json_safe(payload))


def test_exact_run_payload_attaches_replay_scan_cache_diagnostics():
    exact_tool = _load_exact_tool()
    payload = {"profile": {}}
    diagnostics = {
        "replay_dynamic_basepoint_vjp_scan_cache_hit_count": 2,
        "replay_dynamic_basepoint_vjp_scan_cache_build_s": 0.25,
    }

    returned = exact_tool._attach_replay_scan_cache_diagnostics(payload, diagnostics)

    assert returned is payload
    assert payload["replay_scan_cache_diagnostics"] == diagnostics
    json.dumps(exact_tool._json_safe(payload))


def test_exact_run_history_metadata_helper_records_problem_context():
    exact_tool = _load_exact_tool()
    args = exact_tool._parse_args(
        [
            "--problem",
            "qp",
            "--max-mode",
            "5",
            "--method",
            "scipy_matrix_free",
            "--solver-device",
            "gpu",
        ]
    )
    payload = exact_tool._attach_optimizer_run_metadata(
        {},
        args=args,
        specs_count=120,
        solver_device_resolved="default",
    )

    assert payload["problem"] == "qp"
    assert payload["max_mode"] == 5
    assert payload["dofs"] == 120
    assert payload["method"] == "scipy_matrix_free"
    assert payload["solver_device_requested"] == "gpu"
    assert payload["solver_device_resolved"] == "default"


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
            "exact_tape_build_solve_call": {"count": 1, "wall_time_s": 2.1},
            "exact_tape_build_unattributed": {"count": 1, "wall_time_s": 0.7},
            "exact_tape_solver_iteration_loop_unattributed": {"count": 1, "wall_time_s": 0.6},
            "jacobian_initial_tangents": {"count": 1, "wall_time_s": 1.4},
            "jacobian_initial_tangents_jacfwd": {"count": 1, "wall_time_s": 0.25},
            "jacobian_initial_tangents_linearize": {"count": 1, "wall_time_s": 0.3},
            "jacobian_initial_tangents_vmap_dispatch": {"count": 1, "wall_time_s": 0.2},
            "jacobian_initial_tangents_vmap_ready": {"count": 1, "wall_time_s": 0.9},
            "jacobian_tape_replay": {"count": 1, "wall_time_s": 2.1},
            "jacobian_tape_replay_dispatch": {"count": 1, "wall_time_s": 0.2},
            "jacobian_tape_replay_ready": {"count": 1, "wall_time_s": 1.9},
            "jacobian_projected_replay_total": {"count": 1, "wall_time_s": 1.5},
            "jacobian_projected_tape_replay_dispatch": {"count": 1, "wall_time_s": 0.4},
            "jacobian_projected_replay_residual_tangents": {"count": 1, "wall_time_s": 1.1},
            "jacobian_projected_replay_residual_tangents_dispatch": {
                "count": 1,
                "wall_time_s": 0.2,
            },
            "jacobian_projected_replay_residual_tangents_ready": {
                "count": 1,
                "wall_time_s": 0.9,
            },
            "jacobian_residual_tangents": {"count": 1, "wall_time_s": 0.8},
            "jacobian_residual_tangents_dispatch": {"count": 1, "wall_time_s": 0.1},
            "jacobian_residual_tangents_ready": {"count": 1, "wall_time_s": 0.7},
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

    assert metrics["replay_time_s"] == 3.6
    assert metrics["exact_solve_with_tape_jvp_only_s"] == 3.5
    assert metrics["exact_tape_build_jvp_only_s"] == 2.8
    assert metrics["exact_tape_build_solve_call_s"] == 2.1
    assert metrics["exact_tape_solver_iteration_loop_unattributed_s"] == 0.6
    assert metrics["projected_replay_total_s"] == 1.5
    assert metrics["projected_replay_dispatch_s"] == 0.4
    assert metrics["projected_residual_tangents_s"] == 1.1
    assert metrics["projected_residual_tangents_dispatch_s"] == 0.2
    assert metrics["projected_residual_tangents_ready_s"] == 0.9
    assert metrics["accepted_replay_dispatch_s"] == pytest.approx(0.6)
    assert metrics["accepted_replay_ready_s"] == 1.9
    assert metrics["initial_tangents_s"] == 1.4
    assert metrics["initial_tangents_jacfwd_s"] == 0.25
    assert metrics["initial_tangents_linearize_s"] == 0.3
    assert metrics["initial_tangents_vmap_dispatch_s"] == 0.2
    assert metrics["initial_tangents_vmap_ready_s"] == 0.9
    assert metrics["residual_tangents_s"] == 0.8
    assert metrics["residual_tangents_dispatch_s"] == 0.1
    assert metrics["residual_tangents_ready_s"] == 0.7
    assert metrics["exact_tape_solver_compute_forces_s"] == 1.7
    assert metrics["exact_tape_solver_compute_forces_first_s"] == 0.9
    assert metrics["exact_tape_solver_compute_forces_rest_s"] == 0.8
    assert metrics["trial_solver_scan_total_s"] == 1.2
    assert metrics["trial_solver_scan_run_setup_s"] == 0.15
    assert metrics["trial_solver_scan_device_run_s"] == 0.75
    assert metrics["trial_solver_scan_device_dispatch_s"] == 0.05
    assert metrics["trial_solver_scan_device_ready_s"] == 0.7
    projected = summary["projected_replay_summary"]
    assert projected["total_s"] == 1.5
    assert projected["dispatch_s"] == 0.4
    assert projected["residual_tangents_s"] == 1.1
    assert projected["residual_tangents_dispatch_s"] == 0.2
    assert projected["residual_tangents_ready_s"] == 0.9
    assert projected["count"] == 1
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
                    "jacobian_initial_tangents_jacfwd": {"count": 1, "wall_time_s": 0.25},
                    "jacobian_initial_tangents_vmap_dispatch": {"count": 1, "wall_time_s": 0.4},
                    "jacobian_initial_tangents_vmap_ready": {"count": 1, "wall_time_s": 0.8},
                    "jacobian_initial_tangents_cache_miss_lflip_false": {"count": 1, "wall_time_s": 0.0},
                    "jacobian_initial_tangents_cache_miss_lflip_true": {"count": 2, "wall_time_s": 0.0},
                    "jacobian_initial_tangents_cache_hit_lflip_false": {"count": 3, "wall_time_s": 0.0},
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
    assert projected_jvp["jvp"]["initial_tangents_jacfwd_s"] == 0.25
    assert projected_jvp["initial_tangent_cache"]["miss_lflip_false"] == 1
    assert projected_jvp["initial_tangent_cache"]["miss_lflip_true"] == 2
    assert projected_jvp["initial_tangent_cache"]["hit_lflip_false"] == 3
    assert projected_jvp["projected_replay"]["total_s"] == 3.0
    assert projected_jvp["projected_replay"]["dispatch_s"] == 0.5
    assert projected_jvp["projected_replay"]["residual_tangents_s"] == 2.5


def test_performance_matrix_loaded_summary_infers_effective_jvp_from_profile(tmp_path):
    tool = _load_tool()
    report = tmp_path / "profile.json"
    report.write_text(
        json.dumps(
            {
                "report_kind": "exact_optimizer_callback_profile",
                "callback": "jacobian",
                "solver_device_resolved": "default",
                "runtime": {"default_backend": "gpu"},
                "total_wall_time_s": 8.0,
                "jvp_only_exact_tape": False,
                "jvp_only_basepoint_carries": False,
                "samples": [{"repeat": 0, "wall_time_s": 8.0}],
                "profile": {
                    "exact_solve_with_tape_jvp_only_total": {"count": 1, "wall_time_s": 5.0},
                    "exact_tape_build_jvp_only": {"count": 1, "wall_time_s": 4.5},
                },
            }
        ),
        encoding="utf-8",
    )

    summary = tool._load_profile_summary(report, label="gpu")

    assert summary["matrix_projected_replay_jvp_summary"]["jvp"]["exact_tape"] is True
    assert summary["matrix_projected_replay_jvp_summary"]["jvp"]["basepoint_carries"] is True


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


def test_exact_callback_summary_counts_fused_projected_replay():
    compare = _load_compare_tool()
    payload = {
        "report_kind": "exact_optimizer_callback_profile",
        "total_wall_time_s": 5.0,
        "samples": [{"repeat": 0, "wall_time_s": 5.0}],
        "profile": {
            "jacobian_fused_projected_replay_total": {"count": 1, "wall_time_s": 2.25},
            "jacobian_initial_tangents": {"count": 1, "wall_time_s": 0.5},
        },
    }

    summary = compare.summarize_payload(payload, label="fused-projected", top_profile=2)

    assert summary["metrics"]["replay_time_s"] == 2.25
    assert summary["metrics"]["projected_replay_total_s"] == 2.25
    assert summary["metrics"]["accepted_point_replay_count"] == 1
    assert summary["projected_replay_summary"]["total_s"] == 2.25
    assert summary["projected_replay_summary"]["count"] == 1
    assert summary["exact_optimizer_patch_target"]["name"] == "jacobian_fused_projected_replay_total"


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
            "setup_axis_reset_applied": True,
            "setup_axis_reset_done": True,
            "setup_axis_force_probe_available": True,
            "setup_axis_force_probe_reused": True,
            "ignored": "large-history",
        }
    )

    assert compact == {
        "solver_mode": "accelerated",
        "setup_axis_reset_applied": True,
        "setup_axis_reset_done": True,
        "setup_axis_force_probe_available": True,
        "setup_axis_force_probe_reused": True,
        "timing": {
            "iterations": 4,
            "compute_forces_s": 1.25,
            "preconditioner_s": 0.5,
            "update_s": 0.125,
        },
    }


def test_fixed_boundary_profiler_summary_exposes_wall_and_host_update_timing():
    fixed_tool = _load_fixed_tool()
    args = _fixed_profiler_args(fixed_tool, use_scan=False)
    timing = {
        "iterations": 3,
        "solve_total_s": 2.0,
        "compute_forces_s": 0.8,
        "preconditioner_s": 0.4,
        "update_s": 0.3,
        "update_state_s": 0.25,
        "update_state_per_iter_s": 0.08333333333333333,
    }

    summary = fixed_tool._summarize_run(
        args=args,
        run=_fixed_profiler_run({"use_scan": False, "timing": timing}),
        wall_time=1.5,
        jax_module=_FixedProfilerJaxModule,
    )

    assert summary["wall_time_sec"] == 1.5
    assert summary["wall_time_s"] == 1.5
    assert summary["phase_timing"]["run_wall_s"] == 1.5
    assert "jax_devices_s" in summary["phase_timing"]
    assert summary["timing"]["update_state_s"] == 0.25
    assert summary["diagnostics"]["timing"]["update_state_s"] == 0.25
    assert summary["args"]["use_scan"] is False
    json.dumps(fixed_tool._json_safe(summary))


def test_fixed_boundary_profiler_summary_exposes_scan_timing_fields():
    fixed_tool = _load_fixed_tool()
    args = _fixed_profiler_args(fixed_tool, use_scan=True)
    timing = {
        "iterations": 4,
        "scan_total_s": 3.0,
        "scan_setup_s": 0.2,
        "scan_run_setup_s": 0.3,
        "scan_device_run_s": 1.5,
        "scan_device_dispatch_s": 0.2,
        "scan_device_ready_s": 1.3,
        "scan_host_materialize_s": 0.4,
        "scan_postprocess_s": 0.1,
        "scan_unattributed_s": 0.5,
        "scan_runner_cache_hit_count": 1,
        "scan_runner_cache_miss_count": 2,
        "scan_runner_cache_bypass_count": 0,
        "scan_runner_cache_miss_category_cold_empty_count": 1,
        "scan_runner_cache_miss_category_iteration_budget_count": 2,
        "scan_runner_arg_leaf_count": 15,
        "scan_runner_arg_array_nbytes": 700,
        "scan_runner_arg_category_velocity_leaf_count": 6,
        "scan_runner_arg_category_velocity_array_leaf_count": 6,
        "scan_runner_arg_category_velocity_array_nbytes": 480,
        "scan_runner_arg_category_preconditioner_leaf_count": 4,
        "scan_runner_arg_category_preconditioner_array_leaf_count": 3,
        "scan_runner_arg_category_preconditioner_array_nbytes": 160,
        "scan_runner_arg_category_controller_leaf_count": 5,
        "scan_runner_arg_category_controller_array_leaf_count": 1,
        "scan_runner_arg_category_controller_array_nbytes": 60,
        "scan_history_none": 0,
        "scan_history_leaf_count": 9,
        "scan_history_array_leaf_count": 9,
        "scan_history_scalar_leaf_count": 0,
        "scan_history_array_nbytes": 288,
    }

    summary = fixed_tool._summarize_run(
        args=args,
        run=_fixed_profiler_run({"use_scan": True, "vmec2000_scan": True, "timing": timing}),
        wall_time=3.25,
        jax_module=_FixedProfilerJaxModule,
    )

    assert summary["wall_time_s"] == 3.25
    assert summary["timing"]["scan_total_s"] == 3.0
    assert summary["timing"]["scan_device_dispatch_s"] == 0.2
    assert summary["timing"]["scan_device_ready_s"] == 1.3
    assert summary["timing"]["scan_host_materialize_s"] == 0.4
    assert summary["timing"]["scan_runner_cache_hit_count"] == 1
    assert summary["diagnostics"]["timing"]["scan_runner_cache_miss_count"] == 2
    assert summary["scan_arg_categories"]["total_array_nbytes"] == 700
    assert summary["scan_arg_categories"]["largest_category"] == "velocity"
    assert summary["scan_arg_categories"]["categories"]["velocity"]["array_nbytes"] == 480
    assert summary["scan_arg_categories"]["categories"]["preconditioner"]["array_leaf_count"] == 3
    assert summary["scan_cache_miss_categories"]["total_count"] == 3
    assert summary["scan_cache_miss_categories"]["largest_category"] == "iteration_budget"
    assert summary["scan_cache_miss_categories"]["categories"]["cold_empty"] == 1
    assert summary["scan_history"] == {
        "none": False,
        "leaf_count": 9,
        "array_leaf_count": 9,
        "scalar_leaf_count": 0,
        "array_nbytes": 288,
    }
    assert summary["args"]["use_scan"] is True
    json.dumps(fixed_tool._json_safe(summary))


def test_fixed_boundary_profiler_scan_arg_category_summary_handles_empty_timing():
    fixed_tool = _load_fixed_tool()

    assert fixed_tool._scan_arg_category_summary(None) == {}
    assert fixed_tool._scan_arg_category_summary({"iterations": 2}) == {}
    assert fixed_tool._scan_cache_miss_category_summary(None) == {}
    assert fixed_tool._scan_cache_miss_category_summary({"iterations": 2}) == {}
    assert fixed_tool._scan_history_summary(None) == {}
    assert fixed_tool._scan_history_summary({"iterations": 2}) == {}


def test_fixed_boundary_profiler_compile_summary_extracts_jax_rows():
    fixed_tool = _load_fixed_tool()

    class Stats:
        stats = {
            ("/pkg/jax/_src/interpreters/pxla.py", 2486, "compile"): (7, 7, 0.01, 0.70, {}),
            ("/pkg/jax/_src/compiler.py", 396, "compile_or_get_cached"): (5, 5, 0.02, 0.60, {}),
            ("/pkg/jax/_src/compiler.py", 736, "_compile_and_write_cache"): (5, 5, 0.03, 0.55, {}),
            ("/pkg/jax/_src/compiler.py", 312, "backend_compile_and_load"): (5, 5, 0.50, 0.50, {}),
            ("/usr/lib/python3.11/re.py", 10, "compile"): (99, 99, 9.0, 9.0, {}),
        }

    summary = fixed_tool._compile_summary_from_pstats(Stats(), source="synthetic.prof")

    assert summary["source"] == "synthetic.prof"
    assert summary["pxla_compile_call_count"] == 7
    assert summary["compile_or_get_cached_call_count"] == 5
    assert summary["compile_and_write_cache_call_count"] == 5
    assert summary["backend_compile_and_load_call_count"] == 5
    assert summary["backend_compile_and_load_cumulative_s"] == pytest.approx(0.5)
    assert summary["top_compile_call_sites"][0]["function"] == "compile"
    json.dumps(fixed_tool._json_safe(summary))


def test_fixed_boundary_profiler_prints_host_update_and_scan_timing(capsys):
    fixed_tool = _load_fixed_tool()
    args = _fixed_profiler_args(fixed_tool, use_scan=True)
    summary = fixed_tool._summarize_run(
        args=args,
        run=_fixed_profiler_run(
            {
                "use_scan": True,
                "timing": {
                    "iterations": 2,
                    "update_state_s": 0.25,
                    "scan_total_s": 2.0,
                    "scan_device_ready_s": 1.3,
                    "scan_runner_cache_hit_count": 1,
                },
            }
        ),
        wall_time=2.5,
        jax_module=_FixedProfilerJaxModule,
    )

    fixed_tool._print_run_summary(summary)

    output = capsys.readouterr().out
    assert "update_state_s=0.25" in output
    assert "scan_total_s=2" in output
    assert "scan_device_ready_s=1.3" in output
    assert "scan_runner_cache_hit_count=1" in output


def test_fixed_boundary_profiler_scan_requirement_detects_policy_mismatch():
    fixed_tool = _load_fixed_tool()
    require_scan = _fixed_profiler_args(fixed_tool, require_scan=True)
    require_no_scan = _fixed_profiler_args(fixed_tool, require_no_scan=True)

    assert (
        fixed_tool._scan_requirement_error(
            require_scan,
            {"diagnostics": {"use_scan": False, "vmec2000_scan": False}},
        )
        == "Required scan path, but run diagnostics did not report scan execution."
    )
    assert (
        fixed_tool._scan_requirement_error(
            require_no_scan,
            {"diagnostics": {"use_scan": True}},
        )
        == "Required non-scan path, but run diagnostics reported scan execution."
    )
    assert (
        fixed_tool._scan_requirement_error(
            _fixed_profiler_args(fixed_tool, require_scan=True, require_no_scan=True),
            {"diagnostics": {}},
        )
        == "Specify at most one of --require-scan or --require-no-scan."
    )
    assert fixed_tool._scan_requirement_error(require_scan, {"diagnostics": {"vmec2000_scan": True}}) is None
    assert fixed_tool._scan_requirement_error(require_no_scan, {"diagnostics": {"use_scan": False}}) is None


def test_fixed_boundary_profiler_reports_effective_jit_default():
    fixed_tool = _load_fixed_tool()
    args = _fixed_profiler_args(fixed_tool, use_scan=True)

    summary = fixed_tool._summarize_run(
        args=args,
        run=_fixed_profiler_run({}),
        wall_time=1.25,
        jax_module=_FixedProfilerJaxModule,
    )

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
