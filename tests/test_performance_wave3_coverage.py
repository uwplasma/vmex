import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools" / "diagnostics" / "gpu_cpu_performance_matrix.py"


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
                        "residual_tangents_s": 0.7,
                        "trial_solver_scan_device_run_s": 0.4,
                        "callback_count": 1,
                        "accepted_point_replay_count": 1,
                        "contamination_warning_count": 0,
                    }
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
    assert "qh_m2_gpu_jacobian.json" in output
    assert "3.000" in output
    assert "0.600" in output


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
