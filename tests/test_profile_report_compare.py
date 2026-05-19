from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.diagnostics import compare_profile_reports as compare_tool


MIB = 1024 * 1024


def _callback_report(
    *,
    total_wall_time_s: float,
    samples: int,
    rss_peak_mib: int,
    replay_wall_time_s: float,
    accepted_replays: int,
    solve_count: int,
    cache_entry_growth: int,
    solver_device: str,
) -> dict:
    return {
        "schema_version": 2,
        "report_kind": "exact_optimizer_callback_profile",
        "problem": "qh",
        "max_mode": 2,
        "callback": "jacobian",
        "solver_device_resolved": solver_device,
        "total_wall_time_s": total_wall_time_s,
        "rss_before_bytes": 120 * MIB,
        "rss_after_bytes": rss_peak_mib * MIB,
        "samples": [{"repeat": repeat, "wall_time_s": total_wall_time_s / samples} for repeat in range(samples)],
        "profile": {
            "xla_compile": {"count": 1, "wall_time_s": 1.25, "mean_wall_time_s": 1.25},
            "exact_tape_build": {"count": accepted_replays, "wall_time_s": 3.0, "mean_wall_time_s": 1.5},
            "exact_tape_build_unattributed": {"count": accepted_replays, "wall_time_s": 0.5},
            "jacobian_initial_tangents": {"count": accepted_replays, "wall_time_s": 0.75},
            "gradient_initial_vjp": {"count": accepted_replays, "wall_time_s": 0.4},
            "jacobian_residual_tangents": {"count": accepted_replays, "wall_time_s": 0.6},
            "jacobian_tape_replay": {
                "count": accepted_replays,
                "wall_time_s": replay_wall_time_s,
                "mean_wall_time_s": replay_wall_time_s / accepted_replays,
            },
            "exact_unpack_cache": {"count": accepted_replays, "wall_time_s": 0.4, "mean_wall_time_s": 0.2},
            "exact_solve_with_tape_total": {
                "count": accepted_replays,
                "wall_time_s": 3.5,
                "mean_wall_time_s": 1.75,
            },
            "solve_forward_trial": {
                "count": solve_count - accepted_replays,
                "wall_time_s": 1.0,
                "mean_wall_time_s": 1.0,
            },
        },
        "cache": {
            "growth": {
                "total_entries_after": 12 + cache_entry_growth,
                "total_entries_delta": cache_entry_growth,
            }
        },
    }


def test_callback_report_summary_extracts_bottleneck_metrics() -> None:
    summary = compare_tool.summarize_payload(
        _callback_report(
            total_wall_time_s=10.0,
            samples=2,
            rss_peak_mib=256,
            replay_wall_time_s=2.0,
            accepted_replays=2,
            solve_count=3,
            cache_entry_growth=4,
            solver_device="cpu",
        ),
        path=Path("cpu.json"),
        label="cpu",
    )

    metrics = summary["metrics"]
    assert metrics["total_runtime_s"] == 10.0
    assert metrics["compile_time_s"] == 1.25
    assert metrics["exact_tape_build_s"] == 3.0
    assert metrics["exact_tape_build_unattributed_s"] == 0.5
    assert metrics["initial_tangents_s"] == 0.75
    assert metrics["initial_projection_s"] == 0.4
    assert metrics["residual_tangents_s"] == 0.6
    assert metrics["trial_solve_s"] == 1.0
    assert metrics["exact_solve_s"] == 3.5
    assert metrics["replay_time_s"] == 2.0
    assert metrics["cache_time_s"] == 0.4
    assert metrics["callback_count"] == 2
    assert metrics["rss_peak_mib"] == 256.0
    assert metrics["solve_count"] == 3
    assert metrics["accepted_point_replay_count"] == 2
    assert metrics["cache_entry_growth"] == 4
    assert summary["bottleneck_hint"]["metric"] == "exact_solve_s"
    assert summary["exact_optimizer_patch_target"]["name"] == "jacobian_tape_replay"
    assert summary["exact_optimizer_patch_target"]["share_of_total"] == pytest.approx(0.2)
    assert summary["top_profile"][0]["name"] == "exact_solve_with_tape_total"


def test_profile_summary_prefers_total_containers_over_leaf_timers() -> None:
    report = _callback_report(
        total_wall_time_s=20.0,
        samples=2,
        rss_peak_mib=256,
        replay_wall_time_s=2.0,
        accepted_replays=2,
        solve_count=3,
        cache_entry_growth=4,
        solver_device="cpu",
    )
    report["profile"]["solve_forward_trial_total"] = {"count": 1, "wall_time_s": 3.0}
    report["profile"]["solve_forward_trial"] = {"count": 1, "wall_time_s": 1.0}
    report["profile"]["trial_solver_compute_forces"] = {"count": 1, "wall_time_s": 0.4}
    report["profile"]["trial_solver_preconditioner"] = {"count": 1, "wall_time_s": 0.5}
    report["profile"]["trial_solver_update"] = {"count": 1, "wall_time_s": 0.6}
    report["profile"]["solve_forward_trial_unattributed"] = {"count": 1, "wall_time_s": 1.5}
    report["profile"]["solve_forward_exact_total"] = {"count": 2, "wall_time_s": 2.5}
    report["profile"]["solve_forward_exact"] = {"count": 2, "wall_time_s": 1.5}
    report["profile"]["forward_exact_solver_compute_forces"] = {"count": 2, "wall_time_s": 0.7}
    report["profile"]["forward_exact_solver_preconditioner"] = {"count": 2, "wall_time_s": 0.8}
    report["profile"]["forward_exact_solver_update"] = {"count": 2, "wall_time_s": 0.9}
    report["profile"]["solve_forward_exact_unattributed"] = {"count": 2, "wall_time_s": 0.1}

    summary = compare_tool.summarize_payload(report, label="cpu")

    assert summary["metrics"]["trial_solve_s"] == 3.0
    assert summary["metrics"]["exact_solve_s"] == 3.5
    assert summary["metrics"]["trial_solver_compute_forces_s"] == 0.4
    assert summary["metrics"]["trial_solver_preconditioner_s"] == 0.5
    assert summary["metrics"]["trial_solver_update_s"] == 0.6
    assert summary["metrics"]["trial_solve_unattributed_s"] == 1.5
    assert summary["metrics"]["forward_exact_solver_compute_forces_s"] == 0.7
    assert summary["metrics"]["forward_exact_solver_preconditioner_s"] == 0.8
    assert summary["metrics"]["forward_exact_solver_update_s"] == 0.9
    assert summary["metrics"]["forward_exact_solve_unattributed_s"] == 0.1


def test_profile_summary_extracts_solver_subphase_buckets() -> None:
    report = _callback_report(
        total_wall_time_s=20.0,
        samples=2,
        rss_peak_mib=256,
        replay_wall_time_s=2.0,
        accepted_replays=2,
        solve_count=3,
        cache_entry_growth=4,
        solver_device="cpu",
    )
    report["profile"]["trial_solver_compute_forces"] = {"count": 1, "wall_time_s": 1.25}
    report["profile"]["trial_solver_preconditioner"] = {"count": 1, "wall_time_s": 0.75}
    report["profile"]["trial_solver_update"] = {"count": 1, "wall_time_s": 0.50}
    report["profile"]["solve_forward_trial_unattributed"] = {"count": 1, "wall_time_s": 0.25}
    report["profile"]["exact_tape_solver_compute_forces"] = {"count": 1, "wall_time_s": 2.25}
    report["profile"]["exact_tape_solver_preconditioner"] = {"count": 1, "wall_time_s": 1.75}
    report["profile"]["exact_tape_solver_update"] = {"count": 1, "wall_time_s": 1.50}

    summary = compare_tool.summarize_payload(report, label="cpu")

    assert summary["metrics"]["trial_solver_compute_forces_s"] == 1.25
    assert summary["metrics"]["trial_solver_preconditioner_s"] == 0.75
    assert summary["metrics"]["trial_solver_update_s"] == 0.50
    assert summary["metrics"]["trial_solve_unattributed_s"] == 0.25
    assert summary["metrics"]["exact_tape_solver_compute_forces_s"] == 2.25
    assert summary["metrics"]["exact_tape_solver_preconditioner_s"] == 1.75
    assert summary["metrics"]["exact_tape_solver_update_s"] == 1.50


def test_profile_summary_extracts_scan_solver_buckets() -> None:
    report = _callback_report(
        total_wall_time_s=20.0,
        samples=2,
        rss_peak_mib=256,
        replay_wall_time_s=2.0,
        accepted_replays=2,
        solve_count=3,
        cache_entry_growth=4,
        solver_device="cpu",
    )
    report["profile"]["trial_solver_scan_total"] = {"count": 1, "wall_time_s": 2.0}
    report["profile"]["trial_solver_scan_device_run"] = {"count": 1, "wall_time_s": 2.5}
    report["profile"]["trial_solver_scan_host_materialize"] = {"count": 1, "wall_time_s": 0.2}
    report["profile"]["trial_solver_scan_postprocess"] = {"count": 1, "wall_time_s": 0.3}

    summary = compare_tool.summarize_payload(report, label="cpu")

    assert summary["metrics"]["trial_solver_scan_total_s"] == 2.0
    assert summary["metrics"]["trial_solver_scan_device_run_s"] == 2.5
    assert summary["metrics"]["trial_solver_scan_host_materialize_s"] == 0.2
    assert summary["metrics"]["trial_solver_scan_postprocess_s"] == 0.3
    assert summary["exact_optimizer_patch_target"]["name"] == "trial_solver_scan_device_run"


def test_comparison_reports_ratios_against_baseline() -> None:
    cpu = compare_tool.summarize_payload(
        _callback_report(
            total_wall_time_s=10.0,
            samples=2,
            rss_peak_mib=256,
            replay_wall_time_s=2.0,
            accepted_replays=2,
            solve_count=3,
            cache_entry_growth=4,
            solver_device="cpu",
        ),
        label="cpu",
    )
    gpu = compare_tool.summarize_payload(
        _callback_report(
            total_wall_time_s=25.0,
            samples=3,
            rss_peak_mib=512,
            replay_wall_time_s=5.0,
            accepted_replays=3,
            solve_count=5,
            cache_entry_growth=6,
            solver_device="gpu",
        ),
        label="gpu",
    )

    comparison = compare_tool.build_comparison([cpu, gpu], baseline="cpu")
    ratios = comparison["comparisons"][0]["ratios"]

    assert ratios["total_runtime_s"] == 2.5
    assert ratios["replay_time_s"] == 2.5
    assert ratios["callback_count"] == 1.5
    assert ratios["rss_peak_mib"] == 2.0
    assert ratios["solve_count"] == pytest.approx(5.0 / 3.0)
    assert ratios["accepted_point_replay_count"] == 1.5

    text = compare_tool.format_text(comparison)
    assert "Profile report comparison" in text
    assert "gpu" in text
    assert "2.500x" in text
    assert "Bottleneck hints" in text
    assert "Exact optimizer patch targets" in text
    assert "jacobian_tape_replay" in text


def test_repeated_run_report_aggregates_runs() -> None:
    payload = {
        "problem": "qa",
        "max_mode": 1,
        "run_repeats": 2,
        "runs": [
            {
                "total_wall_time_s": 2.0,
                "nfev": 2,
                "njev": 1,
                "profile": {
                    "jacobian_tape_replay": {"count": 1, "wall_time_s": 0.75},
                    "exact_solve_with_tape_total": {"count": 1, "wall_time_s": 1.25},
                },
            },
            {
                "total_wall_time_s": 3.0,
                "nfev": 1,
                "njev": 1,
                "profile": {
                    "jacobian_tape_replay": {"count": 1, "wall_time_s": 1.25},
                    "solve_forward_trial": {"count": 2, "wall_time_s": 0.8},
                },
            },
        ],
    }

    summary = compare_tool.summarize_payload(payload, label="warm")

    assert summary["metadata"]["source_report_kind"] == "exact_optimizer_run_repeats"
    assert summary["metrics"]["total_runtime_s"] == 5.0
    assert summary["metrics"]["callback_count"] == 5
    assert summary["metrics"]["replay_time_s"] == 2.0
    assert summary["metrics"]["accepted_point_replay_count"] == 2
    assert summary["metrics"]["solve_count"] == 3
    assert summary["exact_optimizer_patch_target"]["name"] == "jacobian_tape_replay"


def test_exact_optimizer_patch_target_ignores_container_timers() -> None:
    payload = {
        "report_kind": "exact_optimizer_callback_profile",
        "total_wall_time_s": 20.0,
        "profile": {
            "jacobian_total": {"count": 1, "wall_time_s": 20.0},
            "exact_solve_with_tape_total": {"count": 1, "wall_time_s": 12.0},
            "exact_tape_build": {"count": 1, "wall_time_s": 9.0},
            "exact_tape_build_unattributed": {"count": 1, "wall_time_s": 4.0},
            "jacobian_tape_replay": {"count": 1, "wall_time_s": 5.0},
            "jacobian_residual_tangents": {"count": 1, "wall_time_s": 3.0},
        },
    }

    summary = compare_tool.summarize_payload(payload, label="profile")
    target = summary["exact_optimizer_patch_target"]

    assert target["name"] == "jacobian_tape_replay"
    assert target["wall_time_s"] == 5.0
    assert target["share_of_total"] == pytest.approx(0.25)


def test_exact_optimizer_patch_target_skips_broad_tape_build_when_unattributed_absent() -> None:
    payload = {
        "report_kind": "exact_optimizer_callback_profile",
        "total_wall_time_s": 20.0,
        "profile": {
            "jacobian_total": {"count": 1, "wall_time_s": 20.0},
            "exact_tape_build": {"count": 1, "wall_time_s": 9.0},
            "jacobian_tape_replay": {"count": 1, "wall_time_s": 5.0},
            "jacobian_residual_tangents": {"count": 1, "wall_time_s": 3.0},
        },
    }

    summary = compare_tool.summarize_payload(payload, label="profile")

    assert summary["exact_optimizer_patch_target"]["name"] == "jacobian_tape_replay"


def test_qi_boozer_report_summary_extracts_phase_metrics() -> None:
    payload = {
        "schema_version": 1,
        "report_kind": "qi_boozer_profile",
        "total_wall_time_s": 12.5,
        "solver_device": "gpu",
        "active_gpu": True,
        "runtime": {"default_backend": "gpu", "jax_version": "test"},
        "qi_resolution": {"jit_booz": True},
        "contamination_warnings": ["mixed visible devices"],
        "wall_time_s": {
            "vmec_solve": 4.0,
            "qi_first_call": 6.0,
            "qi_warm_min": 1.0,
            "qi_warm_mean": 1.25,
        },
        "qi_evaluations": [
            {"phase": "first_call", "wall_time_s": 6.0, "jit_booz": True},
            {"phase": "warm_call", "wall_time_s": 1.0, "jit_booz": True},
            {"phase": "warm_call", "wall_time_s": 1.5, "jit_booz": True},
        ],
    }

    summary = compare_tool.summarize_payload(payload, label="gpu")

    assert summary["metadata"]["source_report_kind"] == "qi_boozer_profile"
    assert summary["metadata"]["active_gpu"] is True
    assert summary["metadata"]["jit_booz"] is True
    assert summary["metrics"]["total_runtime_s"] == 12.5
    assert summary["metrics"]["vmec_solve_s"] == 4.0
    assert summary["metrics"]["qi_first_call_s"] == 6.0
    assert summary["metrics"]["qi_warm_min_s"] == 1.0
    assert summary["metrics"]["qi_warm_mean_s"] == 1.25
    assert summary["metrics"]["contamination_warning_count"] == 1


def test_fixed_boundary_report_summary_extracts_solver_phase_timing() -> None:
    payload = {
        "input": "examples/data/input.nfp2_QI",
        "requested_iters": 5,
        "wall_time_sec": 2.5,
        "jax_default_backend": "cpu",
        "args": {"solver_device": "cpu"},
        "diagnostics": {
            "timing": {
                "iterations": 5,
                "compute_forces_s": 1.5,
                "preconditioner_s": 0.6,
                "precond_refresh_s": 0.2,
                "precond_apply_s": 0.3,
                "precond_mode_scale_s": 0.1,
                "update_s": 0.25,
                "update_state_s": 0.2,
            }
        },
    }

    summary = compare_tool.summarize_payload(payload, label="cpu")
    metrics = summary["metrics"]

    assert summary["metadata"]["source_report_kind"] == "fixed_boundary_profile"
    assert metrics["total_runtime_s"] == 2.5
    assert metrics["vmec_compute_forces_s"] == 1.5
    assert metrics["vmec_preconditioner_s"] == 0.6
    assert metrics["vmec_precond_refresh_s"] == 0.2
    assert metrics["vmec_precond_apply_s"] == 0.3
    assert metrics["vmec_precond_mode_scale_s"] == 0.1
    assert metrics["vmec_update_s"] == 0.25
    assert metrics["vmec_update_state_s"] == 0.2
    assert summary["bottleneck_hint"]["metric"] == "vmec_compute_forces_s"


def test_summary_bottleneck_hint_uses_qi_phase_when_largest() -> None:
    payload = {
        "report_kind": "qi_boozer_profile",
        "total_wall_time_s": 12.0,
        "wall_time_s": {
            "vmec_solve": 3.0,
            "qi_first_call": 7.0,
            "qi_warm_mean": 1.0,
        },
    }

    summary = compare_tool.summarize_payload(payload, label="qi")
    hint = summary["bottleneck_hint"]

    assert hint["metric"] == "qi_first_call_s"
    assert hint["label"] == "QI/Boozer first call"
    assert hint["share_of_total"] == pytest.approx(7.0 / 12.0)


def test_cli_prints_text_and_writes_json(tmp_path: Path, capsys) -> None:
    cpu_path = tmp_path / "cpu.json"
    gpu_path = tmp_path / "gpu.json"
    out_path = tmp_path / "comparison.json"
    cpu_path.write_text(
        json.dumps(
            _callback_report(
                total_wall_time_s=10.0,
                samples=2,
                rss_peak_mib=256,
                replay_wall_time_s=2.0,
                accepted_replays=2,
                solve_count=3,
                cache_entry_growth=4,
                solver_device="cpu",
            )
        ),
        encoding="utf-8",
    )
    gpu_path.write_text(
        json.dumps(
            _callback_report(
                total_wall_time_s=25.0,
                samples=3,
                rss_peak_mib=512,
                replay_wall_time_s=5.0,
                accepted_replays=3,
                solve_count=5,
                cache_entry_growth=6,
                solver_device="gpu",
            )
        ),
        encoding="utf-8",
    )

    rc = compare_tool.main(
        [
            str(cpu_path),
            str(gpu_path),
            "--label",
            "cpu",
            "--label",
            "gpu",
            "--json-out",
            str(out_path),
        ]
    )

    assert rc == 0
    stdout = capsys.readouterr().out
    assert "Ratios vs baseline" in stdout
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written["report_kind"] == "profile_report_comparison"
    assert written["baseline_label"] == "cpu"
    assert written["comparisons"][0]["ratios"]["total_runtime_s"] == 2.5
