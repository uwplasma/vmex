import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools" / "diagnostics" / "performance" / "gpu_cpu_performance_matrix.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("gpu_cpu_performance_matrix_wave3", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_matrix_report_surfaces_cold_exact_callback_buckets(capsys):
    tool = _load_tool()
    payload = {
        "runs": [
            {
                "backend": "gpu",
                "dry_run": False,
                "exit_code": 0,
                "wall_time_s": 11.0,
                "summary": {
                    "metrics": {
                        "total_runtime_s": 10.0,
                        "vmec_solve_s": 1.0,
                        "vmec_compute_forces_s": 0.1,
                        "vmec_preconditioner_s": 0.2,
                        "vmec_update_s": 0.3,
                        "exact_solve_s": 4.0,
                        "exact_tape_build_s": 3.0,
                        "exact_tape_build_unattributed_s": 0.6,
                        "replay_time_s": 2.2,
                        "accepted_replay_dispatch_s": 0.2,
                        "accepted_replay_ready_s": 2.0,
                        "initial_tangents_s": 1.5,
                        "initial_tangents_jacfwd_s": 0.35,
                        "residual_tangents_s": 0.7,
                        "projected_replay_total_s": 1.1,
                        "projected_replay_dispatch_s": 0.2,
                        "projected_residual_tangents_s": 0.9,
                        "trial_solver_scan_total_s": 1.6,
                        "trial_solver_scan_setup_s": 0.1,
                        "trial_solver_scan_run_setup_s": 0.2,
                        "trial_solver_scan_preflight_s": 0.05,
                        "trial_solver_scan_device_run_s": 0.4,
                        "trial_solver_scan_device_dispatch_s": 0.06,
                        "trial_solver_scan_device_ready_s": 0.34,
                        "trial_solver_scan_host_materialize_s": 0.07,
                        "trial_solver_scan_postprocess_s": 0.08,
                        "trial_solver_scan_unattributed_s": 0.12,
                        "trial_solver_scan_runner_cache_lookup_s": 0.03,
                        "trial_solver_scan_runner_cache_build_s": 0.25,
                        "trial_solver_scan_runner_cache_hit_count": 2,
                        "trial_solver_scan_runner_cache_miss_count": 1,
                        "trial_solver_scan_runner_cache_bypass_count": 0,
                        "trial_solver_scan_runner_cache_hit_ready_s": 0.11,
                        "trial_solver_scan_runner_cache_miss_ready_s": 0.23,
                        "replay_scan_cache_hit_count": 4,
                        "replay_scan_cache_miss_count": 3,
                        "replay_scan_cache_lookup_s": 0.04,
                        "replay_scan_cache_build_s": 0.5,
                        "callback_count": 1,
                        "accepted_point_replay_count": 1,
                        "contamination_warning_count": 0,
                    },
                    "metadata": {
                        "jvp_only_exact_tape": True,
                        "jvp_only_basepoint_carries": True,
                    },
                    "projected_replay_summary": {
                        "total_s": 1.1,
                        "dispatch_s": 0.2,
                        "residual_tangents_s": 0.9,
                        "count": 1,
                        "share_of_total": 0.11,
                    },
                    "scan_payload_leaders": {
                        "total_array_nbytes": 4096,
                        "velocity_array_nbytes": 2048,
                        "velocity_R_array_nbytes": 1024,
                        "velocity_Z_array_nbytes": 768,
                        "velocity_lambda_array_nbytes": 256,
                        "preconditioner_array_nbytes": 1024,
                        "preconditioner_rz_apply_array_nbytes": 768,
                        "preconditioner_lambda_array_nbytes": 128,
                        "preconditioner_other_array_nbytes": 128,
                        "history_array_nbytes": 96,
                        "largest_category": "velocity",
                        "largest_subcategory": "preconditioner_rz_apply",
                    },
                    "scan_payload_next_target": {
                        "target": "velocity",
                        "target_array_nbytes": 2048,
                        "target_subcategory": "velocity_R",
                        "target_subcategory_array_nbytes": 1024,
                        "bucket_padding_active": True,
                        "requested_seq_len": 5,
                        "actual_seq_len": 8,
                        "padded_extra_iter_count": 3,
                    },
                    "budget_status": {
                        "ok": True,
                        "action": "warn",
                        "measurements": {
                            "scan_hlo_instructions": 512,
                        },
                        "limits": {
                            "scan_velocity_nbytes": 4096,
                            "scan_preconditioner_nbytes": 2048,
                            "scan_history_nbytes": 256,
                            "scan_hlo_instructions": 1024,
                        },
                    },
                    "repeat_runs": [
                        {
                            "wall_time_s": 3.0,
                            "execution_classification": "scan_cold_compile",
                            "timing": {"scan_runner_cache_miss_count": 1},
                        },
                        {
                            "wall_time_s": 0.5,
                            "execution_classification": "scan_cache_hit",
                            "timing": {"scan_runner_cache_miss_count": 0},
                        },
                    ],
                    "sample_profile_summaries": [
                        {
                            "index": 0,
                            "repeat": 0,
                            "wall_time_s": 6.0,
                            "metrics": {
                                "replay_scan_cache_miss_count": 2,
                                "trial_solver_scan_runner_cache_miss_count": 1,
                                "exact_tape_build_s": 0.3,
                                "state_cotangent_s": 2.5,
                                "accepted_replay_dispatch_s": 4.0,
                                "initial_tangents_s": 0.5,
                                "initial_projection_s": 0.6,
                            },
                            "exact_optimizer_patch_target": {"name": "exact_tape_build"},
                        },
                        {
                            "index": 1,
                            "repeat": 1,
                            "wall_time_s": 1.5,
                            "metrics": {
                                "replay_scan_cache_miss_count": 0,
                                "trial_solver_scan_runner_cache_miss_count": 0,
                                "exact_tape_build_s": 0.0,
                                "state_cotangent_s": 0.01,
                                "accepted_replay_dispatch_s": 0.02,
                                "initial_tangents_s": 0.0,
                                "initial_projection_s": 0.03,
                            },
                            "exact_optimizer_patch_target": {"name": "projected_replay"},
                        },
                    ],
                },
                "report_path": "/tmp/qh_m2_gpu_jacobian.json",
            }
        ]
    }

    tool.print_report(payload)
    output = capsys.readouterr().out

    assert "exact_s" in output
    assert "tape_build_s" in output
    assert "tape_unattr_s" in output
    assert "replay_dispatch_s" in output
    assert "replay_ready_s" in output
    assert "callbacks" in output
    assert "replays" in output
    assert "Scan timing:" in output
    assert "scan_s" in output
    assert "run_setup_s" in output
    assert "Scan cache details:" in output
    assert "trial_hits" in output
    assert "trial_misses" in output
    assert "replay_hits" in output
    assert "replay_misses" in output
    assert "Scan payload:" in output
    assert "velocity_B" in output
    assert "hlo_instr" in output
    assert "hlo_budget" in output
    assert "preconditioner_rz_apply" in output
    assert "next_target" in output
    assert "target_B" in output
    assert "target_subcat" in output
    assert "subcat_B" in output
    assert "velocity_R" in output
    assert "bucket_pad" in output
    assert "budget_ok" in output
    assert "Repeated fixed-boundary runs:" in output
    assert "speedup" in output
    assert "scan_cold_compile" in output
    assert "scan_cache_hit" in output
    assert "Exact callback repeats:" in output
    assert "first_replay_misses" in output
    assert "first_tape_s" in output
    assert "first_cot_s" in output
    assert "first_init_tangent_s" in output
    assert "first_init_proj_s" in output
    assert "exact_tape_build" in output
    assert "projected_replay" in output
    assert "Projected replay / JVP details:" in output
    assert "jvp_tape" in output
    assert "base_carries" in output
    assert "init_jacfwd_s" in output
    assert "init_miss_lflip_f" in output
    assert "init_miss_lflip_t" in output
    assert "proj_replay_s" in output
    assert "proj_resid_tangent_s" in output
    assert "qh_m2_gpu_jacobian.json" in output
    assert "3.000" in output
    assert "0.600" in output
    assert "True" in output
    assert "1.100" in output


def test_exact_callback_matrix_dry_run_records_replay_tuning_without_gpu_run(tmp_path):
    tool = _load_tool()
    json_out = tmp_path / "matrix.json"
    outdir = tmp_path / "profiles"

    rc = tool.main(
        [
            "--dry-run",
            "--mode",
            "exact-callback",
            "--backend",
            "cpu",
            "--backend",
            "gpu",
            "--keep-going",
            "--problem",
            "qh",
            "--max-mode",
            "2",
            "--callback",
            "jacobian",
            "--repeats",
            "1",
            "--perturb-scale",
            "1e-4",
            "--inner-max-iter",
            "20",
            "--trial-max-iter",
            "20",
            "--method",
            "scipy_matrix_free",
            "--replay-column-chunk",
            "8",
            "--dynamic-replay-bucket",
            "128",
            "--dynamic-replay-mode",
            "basepoint",
            "--sync-replay-timing",
            "--device-memory-profile",
            "--outdir",
            str(outdir),
            "--json-out",
            str(json_out),
        ]
    )

    payload = json.loads(json_out.read_text(encoding="utf-8"))
    runs = payload["runs"]

    assert rc == 0
    assert payload["mode"] == "exact-callback"
    assert payload["backends"] == ["cpu", "gpu"]
    assert [run["dry_run"] for run in runs] == [True, True]
    assert [run["exit_code"] for run in runs] == [None, None]

    for run in runs:
        command = run["command"]
        env = run["env"]
        assert str(tool.PROFILE_EXACT_OPTIMIZER) in command
        assert command[command.index("--callback") + 1] == "jacobian"
        assert command[command.index("--repeats") + 1] == "1"
        assert command[command.index("--solver-device") + 1] == run["backend"]
        assert "--sync-replay-timing" in command
        assert "--device-memory-profile-out" in command
        assert env["VMEC_JAX_REPLAY_COLUMN_CHUNK"] == "8"
        assert env["VMEC_JAX_DYNAMIC_REPLAY_BUCKET"] == "128"
        assert env["VMEC_JAX_DYNAMIC_REPLAY_MODE"] == "basepoint"
        assert env["VMEC_JAX_OPT_SYNC_REPLAY_TIMING"] == "1"

    assert runs[0]["env"]["JAX_PLATFORMS"] == "cpu"
    assert runs[1]["env"]["JAX_PLATFORM_NAME"] == "gpu"


def test_exact_callback_matrix_problem_all_dry_run_expands_qs_cases(tmp_path):
    tool = _load_tool()
    json_out = tmp_path / "matrix_all.json"
    outdir = tmp_path / "profiles_all"

    rc = tool.main(
        [
            "--dry-run",
            "--mode",
            "exact-callback",
            "--backend",
            "cpu",
            "--problem",
            "all",
            "--max-mode",
            "2",
            "--callback",
            "gradient",
            "--repeats",
            "2",
            "--outdir",
            str(outdir),
            "--json-out",
            str(json_out),
        ]
    )

    payload = json.loads(json_out.read_text(encoding="utf-8"))
    runs = payload["runs"]

    assert rc == 0
    assert payload["mode"] == "exact-callback"
    assert payload["problems"] == ["qa", "qh", "qp"]
    assert [run["problem"] for run in runs] == ["qa", "qh", "qp"]
    assert [run["dry_run"] for run in runs] == [True, True, True]
    assert [run["exit_code"] for run in runs] == [None, None, None]
    for problem, run in zip(("qa", "qh", "qp"), runs, strict=True):
        command = run["command"]
        assert command[command.index("--problem") + 1] == problem
        assert command[command.index("--callback") + 1] == "gradient"
        assert command[command.index("--repeats") + 1] == "2"
        assert f"exact_{problem}_m2_gradient" in run["report_path"]
