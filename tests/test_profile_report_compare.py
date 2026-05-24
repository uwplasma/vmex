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


def test_profile_summary_extracts_linear_operator_transpose_projection() -> None:
    report = _callback_report(
        total_wall_time_s=10.0,
        samples=2,
        rss_peak_mib=256,
        replay_wall_time_s=0.2,
        accepted_replays=2,
        solve_count=3,
        cache_entry_growth=4,
        solver_device="cpu",
    )
    del report["profile"]["gradient_initial_vjp"]
    report["profile"]["linear_operator_initial_transpose"] = {"count": 2, "wall_time_s": 1.4}

    summary = compare_tool.summarize_payload(report, label="cpu")

    assert summary["metrics"]["initial_projection_s"] == 1.4
    assert summary["exact_optimizer_patch_target"]["name"] == "linear_operator_initial_transpose"


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
    report["profile"]["trial_solver_compute_forces_first"] = {"count": 1, "wall_time_s": 0.45}
    report["profile"]["trial_solver_compute_forces_rest"] = {"count": 1, "wall_time_s": 0.80}
    report["profile"]["trial_solver_preconditioner"] = {"count": 1, "wall_time_s": 0.75}
    report["profile"]["trial_solver_precond_refresh"] = {"count": 1, "wall_time_s": 0.15}
    report["profile"]["trial_solver_preconditioner_apply"] = {"count": 1, "wall_time_s": 0.25}
    report["profile"]["trial_solver_preconditioner_mode_scale"] = {"count": 1, "wall_time_s": 0.10}
    report["profile"]["trial_solver_update"] = {"count": 1, "wall_time_s": 0.50}
    report["profile"]["trial_solver_update_state"] = {"count": 1, "wall_time_s": 0.40}
    report["profile"]["solve_forward_trial_unattributed"] = {"count": 1, "wall_time_s": 0.25}
    report["profile"]["exact_tape_solver_compute_forces"] = {"count": 1, "wall_time_s": 2.25}
    report["profile"]["exact_tape_solver_compute_forces_first"] = {"count": 1, "wall_time_s": 1.00}
    report["profile"]["exact_tape_solver_compute_forces_rest"] = {"count": 1, "wall_time_s": 1.25}
    report["profile"]["exact_tape_solver_preconditioner"] = {"count": 1, "wall_time_s": 1.75}
    report["profile"]["exact_tape_solver_precond_refresh"] = {"count": 1, "wall_time_s": 0.35}
    report["profile"]["exact_tape_solver_preconditioner_apply"] = {"count": 1, "wall_time_s": 0.95}
    report["profile"]["exact_tape_solver_preconditioner_mode_scale"] = {"count": 1, "wall_time_s": 0.20}
    report["profile"]["exact_tape_solver_update"] = {"count": 1, "wall_time_s": 1.50}
    report["profile"]["exact_tape_solver_update_state"] = {"count": 1, "wall_time_s": 1.20}
    report["profile"]["forward_exact_solver_compute_forces"] = {"count": 1, "wall_time_s": 1.35}
    report["profile"]["forward_exact_solver_compute_forces_first"] = {"count": 1, "wall_time_s": 0.55}
    report["profile"]["forward_exact_solver_compute_forces_rest"] = {"count": 1, "wall_time_s": 0.80}
    report["profile"]["forward_exact_solver_preconditioner"] = {"count": 1, "wall_time_s": 0.95}
    report["profile"]["forward_exact_solver_precond_refresh"] = {"count": 1, "wall_time_s": 0.25}
    report["profile"]["forward_exact_solver_preconditioner_apply"] = {"count": 1, "wall_time_s": 0.45}
    report["profile"]["forward_exact_solver_preconditioner_mode_scale"] = {"count": 1, "wall_time_s": 0.15}
    report["profile"]["forward_exact_solver_update"] = {"count": 1, "wall_time_s": 0.65}
    report["profile"]["forward_exact_solver_update_state"] = {"count": 1, "wall_time_s": 0.50}

    summary = compare_tool.summarize_payload(report, label="cpu")

    assert summary["metrics"]["trial_solver_compute_forces_s"] == 1.25
    assert summary["metrics"]["trial_solver_compute_forces_first_s"] == 0.45
    assert summary["metrics"]["trial_solver_compute_forces_rest_s"] == 0.80
    assert summary["metrics"]["trial_solver_preconditioner_s"] == 0.75
    assert summary["metrics"]["trial_solver_precond_refresh_s"] == 0.15
    assert summary["metrics"]["trial_solver_preconditioner_apply_s"] == 0.25
    assert summary["metrics"]["trial_solver_preconditioner_mode_scale_s"] == 0.10
    assert summary["metrics"]["trial_solver_update_s"] == 0.50
    assert summary["metrics"]["trial_solver_update_state_s"] == 0.40
    assert summary["metrics"]["trial_solve_unattributed_s"] == 0.25
    assert summary["metrics"]["exact_tape_solver_compute_forces_s"] == 2.25
    assert summary["metrics"]["exact_tape_solver_compute_forces_first_s"] == 1.00
    assert summary["metrics"]["exact_tape_solver_compute_forces_rest_s"] == 1.25
    assert summary["metrics"]["exact_tape_solver_preconditioner_s"] == 1.75
    assert summary["metrics"]["exact_tape_solver_precond_refresh_s"] == 0.35
    assert summary["metrics"]["exact_tape_solver_preconditioner_apply_s"] == 0.95
    assert summary["metrics"]["exact_tape_solver_preconditioner_mode_scale_s"] == 0.20
    assert summary["metrics"]["exact_tape_solver_update_s"] == 1.50
    assert summary["metrics"]["exact_tape_solver_update_state_s"] == 1.20
    assert summary["metrics"]["forward_exact_solver_compute_forces_s"] == 1.35
    assert summary["metrics"]["forward_exact_solver_compute_forces_first_s"] == 0.55
    assert summary["metrics"]["forward_exact_solver_compute_forces_rest_s"] == 0.80
    assert summary["metrics"]["forward_exact_solver_preconditioner_s"] == 0.95
    assert summary["metrics"]["forward_exact_solver_precond_refresh_s"] == 0.25
    assert summary["metrics"]["forward_exact_solver_preconditioner_apply_s"] == 0.45
    assert summary["metrics"]["forward_exact_solver_preconditioner_mode_scale_s"] == 0.15
    assert summary["metrics"]["forward_exact_solver_update_s"] == 0.65
    assert summary["metrics"]["forward_exact_solver_update_state_s"] == 0.50


def test_profile_summary_extracts_free_boundary_nestor_buckets() -> None:
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
    report["profile"]["trial_solver_freeb_nestor_sample"] = {"count": 1, "wall_time_s": 0.11}
    report["profile"]["trial_solver_freeb_nestor_solve"] = {"count": 1, "wall_time_s": 0.12}
    report["profile"]["trial_solver_freeb_nestor_trial_sample"] = {"count": 1, "wall_time_s": 0.13}
    report["profile"]["trial_solver_freeb_nestor_trial_solve"] = {"count": 1, "wall_time_s": 0.14}
    report["profile"]["trial_solver_freeb_nestor_full_update_count"] = {"count": 1, "wall_time_s": 2.0}
    report["profile"]["trial_solver_freeb_nestor_reused_count"] = {"count": 1, "wall_time_s": 3.0}
    report["profile"]["trial_solver_freeb_nestor_trial_reused_count"] = {"count": 1, "wall_time_s": 4.0}
    report["profile"]["trial_solver_freeb_nestor_trial_failed_count"] = {"count": 1, "wall_time_s": 5.0}
    report["profile"]["forward_exact_solver_freeb_nestor_sample"] = {"count": 1, "wall_time_s": 0.21}
    report["profile"]["forward_exact_solver_freeb_nestor_solve"] = {"count": 1, "wall_time_s": 0.22}
    report["profile"]["forward_exact_solver_freeb_nestor_trial_sample"] = {"count": 1, "wall_time_s": 0.23}
    report["profile"]["forward_exact_solver_freeb_nestor_trial_solve"] = {"count": 1, "wall_time_s": 0.24}
    report["profile"]["exact_tape_solver_freeb_nestor_sample"] = {"count": 1, "wall_time_s": 0.31}
    report["profile"]["exact_tape_solver_freeb_nestor_solve"] = {"count": 1, "wall_time_s": 0.32}
    report["profile"]["exact_tape_solver_freeb_nestor_trial_sample"] = {"count": 1, "wall_time_s": 0.33}
    report["profile"]["exact_tape_solver_freeb_nestor_trial_solve"] = {"count": 1, "wall_time_s": 0.34}

    summary = compare_tool.summarize_payload(report, label="cpu")

    assert summary["metrics"]["trial_solver_freeb_nestor_sample_s"] == 0.11
    assert summary["metrics"]["trial_solver_freeb_nestor_solve_s"] == 0.12
    assert summary["metrics"]["trial_solver_freeb_nestor_trial_sample_s"] == 0.13
    assert summary["metrics"]["trial_solver_freeb_nestor_trial_solve_s"] == 0.14
    assert summary["metrics"]["trial_solver_freeb_nestor_full_update_count"] == 2.0
    assert summary["metrics"]["trial_solver_freeb_nestor_reused_count"] == 3.0
    assert summary["metrics"]["trial_solver_freeb_nestor_trial_reused_count"] == 4.0
    assert summary["metrics"]["trial_solver_freeb_nestor_trial_failed_count"] == 5.0
    assert summary["metrics"]["forward_exact_solver_freeb_nestor_sample_s"] == 0.21
    assert summary["metrics"]["forward_exact_solver_freeb_nestor_solve_s"] == 0.22
    assert summary["metrics"]["forward_exact_solver_freeb_nestor_trial_sample_s"] == 0.23
    assert summary["metrics"]["forward_exact_solver_freeb_nestor_trial_solve_s"] == 0.24
    assert summary["metrics"]["exact_tape_solver_freeb_nestor_sample_s"] == 0.31
    assert summary["metrics"]["exact_tape_solver_freeb_nestor_solve_s"] == 0.32
    assert summary["metrics"]["exact_tape_solver_freeb_nestor_trial_sample_s"] == 0.33
    assert summary["metrics"]["exact_tape_solver_freeb_nestor_trial_solve_s"] == 0.34


def test_profile_summary_extracts_solve_call_internal_buckets() -> None:
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
    report["profile"]["exact_tape_build_solve_call"] = {"count": 1, "wall_time_s": 4.0}
    report["profile"]["exact_tape_solver_setup_total"] = {"count": 1, "wall_time_s": 0.8}
    report["profile"]["exact_tape_solver_setup_axis_reset"] = {"count": 1, "wall_time_s": 0.1}
    report["profile"]["exact_tape_solver_setup_unattributed"] = {"count": 1, "wall_time_s": 0.7}
    report["profile"]["exact_tape_solver_iteration_loop"] = {"count": 1, "wall_time_s": 2.8}
    report["profile"]["exact_tape_solver_iteration_prepare"] = {"count": 1, "wall_time_s": 0.2}
    report["profile"]["exact_tape_solver_compute_forces"] = {"count": 1, "wall_time_s": 0.5}
    report["profile"]["exact_tape_solver_iteration_residual_metrics"] = {"count": 1, "wall_time_s": 1.2}
    report["profile"]["exact_tape_solver_preconditioner"] = {"count": 1, "wall_time_s": 0.3}
    report["profile"]["exact_tape_solver_update"] = {"count": 1, "wall_time_s": 0.4}
    report["profile"]["exact_tape_solver_iteration_post_update"] = {"count": 1, "wall_time_s": 0.2}
    report["profile"]["exact_tape_solver_iteration_loop_unattributed"] = {"count": 1, "wall_time_s": 2.5}
    report["profile"]["exact_tape_solver_finalize"] = {"count": 1, "wall_time_s": 0.3}

    summary = compare_tool.summarize_payload(report, label="cpu")

    assert summary["metrics"]["exact_tape_build_solve_call_s"] == 4.0
    assert summary["metrics"]["exact_tape_solver_setup_total_s"] == 0.8
    assert summary["metrics"]["exact_tape_solver_setup_axis_reset_s"] == 0.1
    assert summary["metrics"]["exact_tape_solver_iteration_loop_s"] == 2.8
    assert summary["metrics"]["exact_tape_solver_iteration_residual_metrics_s"] == 1.2
    assert summary["metrics"]["exact_tape_solver_iteration_loop_unattributed_s"] == 2.5
    assert summary["metrics"]["exact_tape_solver_finalize_s"] == 0.3
    assert summary["exact_optimizer_patch_target"]["name"] == "exact_tape_solver_iteration_loop_unattributed"


def test_profile_summary_extracts_direct_tape_build_leaf_buckets() -> None:
    report = _callback_report(
        total_wall_time_s=20.0,
        samples=2,
        rss_peak_mib=256,
        replay_wall_time_s=1.0,
        accepted_replays=2,
        solve_count=3,
        cache_entry_growth=4,
        solver_device="cpu",
    )
    report["profile"]["exact_tape_build_solve_call"] = {"count": 2, "wall_time_s": 1.25}
    report["profile"]["exact_tape_build_final_state_pack"] = {"count": 2, "wall_time_s": 0.10}
    report["profile"]["exact_tape_build_step_trace_extract"] = {"count": 2, "wall_time_s": 0.20}
    report["profile"]["exact_tape_build_dynamic_payload"] = {"count": 2, "wall_time_s": 1.75}
    report["profile"]["exact_tape_build_trace_stack"] = {"count": 2, "wall_time_s": 0.30}
    report["profile"]["exact_tape_build_unattributed"] = {"count": 2, "wall_time_s": 0.05}

    summary = compare_tool.summarize_payload(report, label="cpu")

    assert summary["metrics"]["exact_tape_build_solve_call_s"] == 1.25
    assert summary["metrics"]["exact_tape_build_final_state_pack_s"] == 0.10
    assert summary["metrics"]["exact_tape_build_step_trace_extract_s"] == 0.20
    assert summary["metrics"]["exact_tape_build_dynamic_payload_s"] == 1.75
    assert summary["metrics"]["exact_tape_build_trace_stack_s"] == 0.30
    assert summary["metrics"]["exact_tape_build_unattributed_s"] == 0.05
    assert summary["exact_optimizer_patch_target"]["name"] == "exact_tape_build_dynamic_payload"


def test_profile_summary_extracts_scan_solver_buckets() -> None:
    report = _callback_report(
        total_wall_time_s=20.0,
        samples=2,
        rss_peak_mib=256,
        replay_wall_time_s=0.2,
        accepted_replays=2,
        solve_count=3,
        cache_entry_growth=4,
        solver_device="cpu",
    )
    report["profile"]["trial_solver_scan_total"] = {"count": 1, "wall_time_s": 2.0}
    report["profile"]["trial_solver_scan_setup"] = {"count": 1, "wall_time_s": 0.04}
    report["profile"]["trial_solver_scan_initial_compute_forces"] = {"count": 1, "wall_time_s": 0.05}
    report["profile"]["trial_solver_scan_axis_reset_compute_forces"] = {"count": 1, "wall_time_s": 0.06}
    report["profile"]["trial_solver_scan_run_setup"] = {"count": 1, "wall_time_s": 0.09}
    report["profile"]["trial_solver_scan_runner_cache_lookup"] = {"count": 1, "wall_time_s": 0.11}
    report["profile"]["trial_solver_scan_runner_cache_build"] = {"count": 1, "wall_time_s": 0.22}
    report["profile"]["trial_solver_scan_runner_cache_hit_count"] = {"count": 1, "wall_time_s": 3.0}
    report["profile"]["trial_solver_scan_runner_cache_miss_count"] = {"count": 1, "wall_time_s": 2.0}
    report["profile"]["trial_solver_scan_runner_cache_bypass_count"] = {"count": 1, "wall_time_s": 1.0}
    report["profile"]["trial_solver_scan_preflight"] = {"count": 1, "wall_time_s": 0.07}
    report["profile"]["trial_solver_scan_device_run"] = {"count": 1, "wall_time_s": 2.5}
    report["profile"]["trial_solver_scan_device_dispatch"] = {"count": 1, "wall_time_s": 0.4}
    report["profile"]["trial_solver_scan_device_ready"] = {"count": 1, "wall_time_s": 2.1}
    report["profile"]["trial_solver_scan_runner_cache_hit_device_run"] = {
        "count": 1,
        "wall_time_s": 0.6,
    }
    report["profile"]["trial_solver_scan_runner_cache_hit_dispatch"] = {
        "count": 1,
        "wall_time_s": 0.1,
    }
    report["profile"]["trial_solver_scan_runner_cache_hit_ready"] = {"count": 1, "wall_time_s": 0.5}
    report["profile"]["trial_solver_scan_runner_cache_miss_device_run"] = {
        "count": 1,
        "wall_time_s": 1.5,
    }
    report["profile"]["trial_solver_scan_runner_cache_miss_dispatch"] = {
        "count": 1,
        "wall_time_s": 0.2,
    }
    report["profile"]["trial_solver_scan_runner_cache_miss_ready"] = {
        "count": 1,
        "wall_time_s": 1.3,
    }
    report["profile"]["trial_solver_scan_runner_cache_bypass_device_run"] = {
        "count": 1,
        "wall_time_s": 0.4,
    }
    report["profile"]["trial_solver_scan_runner_cache_bypass_dispatch"] = {
        "count": 1,
        "wall_time_s": 0.1,
    }
    report["profile"]["trial_solver_scan_runner_cache_bypass_ready"] = {
        "count": 1,
        "wall_time_s": 0.3,
    }
    report["profile"]["trial_solver_scan_host_materialize"] = {"count": 1, "wall_time_s": 0.2}
    report["profile"]["trial_solver_scan_postprocess"] = {"count": 1, "wall_time_s": 0.3}
    report["profile"]["trial_solver_scan_unattributed"] = {"count": 1, "wall_time_s": 0.08}

    summary = compare_tool.summarize_payload(report, label="cpu")

    assert summary["metrics"]["trial_solver_scan_total_s"] == 2.0
    assert summary["metrics"]["trial_solver_scan_setup_s"] == 0.04
    assert summary["metrics"]["trial_solver_scan_initial_compute_forces_s"] == 0.05
    assert summary["metrics"]["trial_solver_scan_axis_reset_compute_forces_s"] == 0.06
    assert summary["metrics"]["trial_solver_scan_run_setup_s"] == 0.09
    assert summary["metrics"]["trial_solver_scan_runner_cache_lookup_s"] == 0.11
    assert summary["metrics"]["trial_solver_scan_runner_cache_build_s"] == 0.22
    assert summary["metrics"]["trial_solver_scan_runner_cache_hit_count"] == 3.0
    assert summary["metrics"]["trial_solver_scan_runner_cache_miss_count"] == 2.0
    assert summary["metrics"]["trial_solver_scan_runner_cache_bypass_count"] == 1.0
    assert summary["metrics"]["trial_solver_scan_preflight_s"] == 0.07
    assert summary["metrics"]["trial_solver_scan_device_run_s"] == 2.5
    assert summary["metrics"]["trial_solver_scan_device_dispatch_s"] == 0.4
    assert summary["metrics"]["trial_solver_scan_device_ready_s"] == 2.1
    assert summary["metrics"]["trial_solver_scan_runner_cache_hit_device_run_s"] == 0.6
    assert summary["metrics"]["trial_solver_scan_runner_cache_hit_dispatch_s"] == 0.1
    assert summary["metrics"]["trial_solver_scan_runner_cache_hit_ready_s"] == 0.5
    assert summary["metrics"]["trial_solver_scan_runner_cache_miss_device_run_s"] == 1.5
    assert summary["metrics"]["trial_solver_scan_runner_cache_miss_dispatch_s"] == 0.2
    assert summary["metrics"]["trial_solver_scan_runner_cache_miss_ready_s"] == 1.3
    assert summary["metrics"]["trial_solver_scan_runner_cache_bypass_device_run_s"] == 0.4
    assert summary["metrics"]["trial_solver_scan_runner_cache_bypass_dispatch_s"] == 0.1
    assert summary["metrics"]["trial_solver_scan_runner_cache_bypass_ready_s"] == 0.3
    assert summary["metrics"]["trial_solver_scan_host_materialize_s"] == 0.2
    assert summary["metrics"]["trial_solver_scan_postprocess_s"] == 0.3
    assert summary["metrics"]["trial_solver_scan_unattributed_s"] == 0.08
    assert summary["trial_scan_summary"]["cache_lookup_s"] == 0.11
    assert summary["trial_scan_summary"]["cache_build_s"] == 0.22
    assert summary["trial_scan_summary"]["cache_status"]["miss"]["count"] == 2
    assert summary["trial_scan_summary"]["cache_status"]["miss"]["device_run_s"] == 1.5
    assert summary["trial_scan_summary"]["cache_miss_fraction"] == pytest.approx(2.0 / 6.0)
    assert summary["trial_scan_summary"]["dominant_cache_status_by_device_run"] == "miss"
    assert summary["exact_optimizer_patch_target"]["name"] == "trial_solver_scan_runner_cache_miss_ready"
    assert "trial_solver_scan_runner_cache_miss_count" not in {
        entry["name"] for entry in summary["top_profile"]
    }
    comparison = compare_tool.build_comparison([summary, summary], baseline="cpu")
    text = compare_tool.format_text(comparison)
    assert "Trial scan timing/cache status" in text
    assert "miss_frac" in text
    assert "33.3%" in text


def test_profile_summary_prefers_split_replay_and_tangent_buckets() -> None:
    report = _callback_report(
        total_wall_time_s=20.0,
        samples=2,
        rss_peak_mib=256,
        replay_wall_time_s=3.0,
        accepted_replays=2,
        solve_count=3,
        cache_entry_growth=4,
        solver_device="cpu",
    )
    report["profile"]["jacobian_tape_replay_dispatch"] = {"count": 2, "wall_time_s": 0.4}
    report["profile"]["jacobian_tape_replay_ready"] = {"count": 2, "wall_time_s": 2.6}
    report["profile"]["jacobian_initial_tangents_linearize"] = {"count": 1, "wall_time_s": 1.4}
    report["profile"]["jacobian_initial_tangents_vmap"] = {"count": 1, "wall_time_s": 1.0}
    report["profile"]["jacobian_initial_tangents_vmap_dispatch"] = {"count": 1, "wall_time_s": 0.2}
    report["profile"]["jacobian_initial_tangents_vmap_ready"] = {"count": 1, "wall_time_s": 0.8}

    summary = compare_tool.summarize_payload(report, label="cpu")

    assert summary["metrics"]["accepted_replay_dispatch_s"] == 0.4
    assert summary["metrics"]["accepted_replay_ready_s"] == 2.6
    assert summary["metrics"]["replay_time_s"] == 3.0
    assert summary["metrics"]["initial_tangents_linearize_s"] == 1.4
    assert summary["metrics"]["initial_tangents_vmap_s"] == 1.0
    assert summary["metrics"]["initial_tangents_vmap_dispatch_s"] == 0.2
    assert summary["metrics"]["initial_tangents_vmap_ready_s"] == 0.8
    assert summary["exact_optimizer_patch_target"]["name"] == "jacobian_tape_replay_ready"


def test_profile_summary_accounts_projected_replay_buckets() -> None:
    report = _callback_report(
        total_wall_time_s=20.0,
        samples=2,
        rss_peak_mib=256,
        replay_wall_time_s=0.0,
        accepted_replays=2,
        solve_count=3,
        cache_entry_growth=4,
        solver_device="gpu",
    )
    report["profile"].pop("jacobian_tape_replay")
    report["profile"].pop("jacobian_residual_tangents")
    report["profile"]["jacobian_projected_replay_total"] = {
        "count": 2,
        "wall_time_s": 4.0,
        "mean_wall_time_s": 2.0,
    }
    report["profile"]["jacobian_projected_tape_replay_dispatch"] = {
        "count": 2,
        "wall_time_s": 0.7,
        "mean_wall_time_s": 0.35,
    }
    report["profile"]["jacobian_projected_replay_residual_tangents"] = {
        "count": 2,
        "wall_time_s": 3.3,
        "mean_wall_time_s": 1.65,
    }

    summary = compare_tool.summarize_payload(report, label="gpu")

    assert summary["metrics"]["replay_time_s"] == 4.0
    assert summary["metrics"]["projected_replay_total_s"] == 4.0
    assert summary["metrics"]["projected_replay_dispatch_s"] == 0.7
    assert summary["metrics"]["accepted_replay_dispatch_s"] == 0.7
    assert summary["metrics"]["projected_residual_tangents_s"] == 3.3
    assert summary["metrics"]["accepted_point_replay_count"] == 2
    assert summary["projected_replay_summary"]["total_s"] == 4.0
    assert summary["projected_replay_summary"]["dispatch_s"] == 0.7
    assert summary["projected_replay_summary"]["residual_tangents_s"] == 3.3
    assert summary["projected_replay_summary"]["count"] == 2
    assert summary["projected_replay_summary"]["share_of_total"] == pytest.approx(0.2)
    assert summary["projected_replay_summary"]["residual_tangent_share_of_projected"] == pytest.approx(
        3.3 / 4.0
    )
    assert summary["exact_optimizer_patch_target"]["name"] == "jacobian_projected_replay_residual_tangents"
    comparison = compare_tool.build_comparison([summary, summary], baseline="gpu")
    text = compare_tool.format_text(comparison)
    assert "Projected replay totals" in text
    assert "residual_tangent_s" in text


def test_profile_summary_extracts_replay_scan_cache_diagnostics() -> None:
    report = _callback_report(
        total_wall_time_s=20.0,
        samples=2,
        rss_peak_mib=256,
        replay_wall_time_s=3.0,
        accepted_replays=2,
        solve_count=3,
        cache_entry_growth=4,
        solver_device="cpu",
    )
    report["replay_scan_cache_diagnostics"] = {
        "replay_checkpoint_scan_cache_hit_count": 2,
        "replay_checkpoint_scan_cache_miss_count": 1,
        "replay_checkpoint_scan_cache_lookup_s": 0.01,
        "replay_checkpoint_scan_cache_build_s": 0.20,
        "replay_dynamic_basepoint_scan_cache_hit_count": 4,
        "replay_dynamic_basepoint_scan_cache_miss_count": 3,
        "replay_dynamic_basepoint_scan_cache_lookup_s": 0.03,
        "replay_dynamic_basepoint_scan_cache_build_s": 0.40,
    }

    summary = compare_tool.summarize_payload(report, label="cpu")

    assert summary["metrics"]["replay_scan_cache_hit_count"] == 6
    assert summary["metrics"]["replay_scan_cache_miss_count"] == 4
    assert summary["metrics"]["replay_scan_cache_lookup_s"] == pytest.approx(0.04)
    assert summary["metrics"]["replay_scan_cache_build_s"] == pytest.approx(0.60)

    comparison = compare_tool.build_comparison([summary, summary], baseline="cpu")
    text = compare_tool.format_text(comparison)
    assert "replay scan-cache misses" in text
    assert "replay scan-cache build" in text


def test_callback_report_summary_exposes_cold_sample_hotspot() -> None:
    report = _callback_report(
        total_wall_time_s=20.0,
        samples=2,
        rss_peak_mib=256,
        replay_wall_time_s=4.0,
        accepted_replays=2,
        solve_count=3,
        cache_entry_growth=4,
        solver_device="gpu",
    )
    report["samples"][0]["profile_delta"] = {
        "jacobian_total": {"count": 1, "wall_time_s": 16.0},
        "exact_tape_build": {"count": 1, "wall_time_s": 3.0},
        "jacobian_tape_replay": {"count": 1, "wall_time_s": 9.0},
        "jacobian_tape_replay_dispatch": {"count": 1, "wall_time_s": 8.5},
        "jacobian_tape_replay_ready": {"count": 1, "wall_time_s": 0.5},
        "jacobian_residual_tangents": {"count": 1, "wall_time_s": 2.0},
    }
    report["samples"][0]["replay_scan_cache_diagnostics"] = {
        "replay_checkpoint_scan_cache_miss_count": 1,
        "replay_checkpoint_scan_cache_build_s": 0.3,
    }
    report["samples"][1]["profile_delta"] = {
        "jacobian_total": {"count": 1, "wall_time_s": 4.0},
        "jacobian_tape_replay": {"count": 1, "wall_time_s": 1.0},
        "jacobian_tape_replay_dispatch": {"count": 1, "wall_time_s": 0.9},
        "jacobian_tape_replay_ready": {"count": 1, "wall_time_s": 0.1},
    }
    report["samples"][1]["replay_scan_cache_diagnostics"] = {
        "replay_checkpoint_scan_cache_hit_count": 1,
        "replay_checkpoint_scan_cache_lookup_s": 0.01,
    }

    summary = compare_tool.summarize_payload(report, label="gpu")

    samples = summary["sample_profile_summaries"]
    assert len(samples) == 2
    assert samples[0]["repeat"] == 0
    assert samples[0]["metrics"]["accepted_replay_dispatch_s"] == 8.5
    assert samples[0]["metrics"]["accepted_replay_ready_s"] == 0.5
    assert samples[0]["metrics"]["replay_time_s"] == 9.0
    assert samples[0]["metrics"]["replay_scan_cache_miss_count"] == 1
    assert samples[0]["metrics"]["replay_scan_cache_build_s"] == 0.3
    assert samples[0]["exact_optimizer_patch_target"]["name"] == "jacobian_tape_replay_dispatch"
    assert samples[0]["exact_optimizer_patch_target"]["share_of_total"] == pytest.approx(0.85)
    assert samples[1]["exact_optimizer_patch_target"]["name"] == "jacobian_tape_replay_dispatch"

    comparison = compare_tool.build_comparison([summary, summary], baseline="0")
    text = compare_tool.format_text(comparison)
    assert "Cold callback patch targets" in text
    assert "gpu repeat 0: jacobian_tape_replay_dispatch" in text


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


def test_exact_optimizer_patch_target_prefers_split_solver_leaves() -> None:
    payload = {
        "report_kind": "exact_optimizer_callback_profile",
        "total_wall_time_s": 10.0,
        "profile": {
            "trial_solver_compute_forces": {"count": 1, "wall_time_s": 4.0},
            "trial_solver_compute_forces_first": {"count": 1, "wall_time_s": 2.5},
            "trial_solver_compute_forces_rest": {"count": 1, "wall_time_s": 1.5},
            "exact_tape_solver_preconditioner": {"count": 1, "wall_time_s": 3.0},
            "exact_tape_solver_precond_refresh": {"count": 1, "wall_time_s": 0.8},
            "exact_tape_solver_preconditioner_apply": {"count": 1, "wall_time_s": 1.9},
            "exact_tape_solver_preconditioner_mode_scale": {"count": 1, "wall_time_s": 0.3},
        },
    }

    summary = compare_tool.summarize_payload(payload, label="profile")

    target = summary["exact_optimizer_patch_target"]
    assert target["name"] == "trial_solver_compute_forces_first"
    assert target["wall_time_s"] == 2.5
    assert target["share_of_total"] == pytest.approx(0.25)


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
