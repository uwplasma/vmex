"""Profiling helpers for fixed-boundary exact optimization.

The optimizer class owns the profile dictionary and compatibility method names.
This module keeps the timing-bucket logic out of the class so the optimizer
remains focused on solves, Jacobians, and adjoint replay.
"""

from __future__ import annotations

import os
import time

import numpy as np


_EXACT_TAPE_BUILD_TIMING_PROFILE_NAMES = (
    ("tape_solve_call_s", "exact_tape_build_solve_call"),
    ("tape_final_state_pack_s", "exact_tape_build_final_state_pack"),
    ("tape_step_trace_extract_s", "exact_tape_build_step_trace_extract"),
    ("tape_dynamic_payload_build_s", "exact_tape_build_dynamic_payload"),
    ("tape_trace_stack_s", "exact_tape_build_trace_stack"),
)


def profile_solver_free_boundary_timing(optimizer, diagnostics, *, profile_prefix: str) -> None:
    """Record optional free-boundary NESTOR timing/counter arrays from diagnostics."""

    if not isinstance(diagnostics, dict):
        return

    def _sum_time(key: str) -> float | None:
        if key not in diagnostics:
            return None
        try:
            arr = np.asarray(diagnostics.get(key), dtype=float).reshape(-1)
        except Exception:
            return None
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return None
        return float(np.sum(arr))

    def _count_nonzero(key: str) -> int | None:
        if key not in diagnostics:
            return None
        try:
            arr = np.asarray(diagnostics.get(key), dtype=int).reshape(-1)
        except Exception:
            return None
        if arr.size == 0:
            return None
        return int(np.count_nonzero(arr))

    for key, suffix in (
        ("freeb_nestor_sample_time_history", "freeb_nestor_sample"),
        ("freeb_nestor_solve_time_history", "freeb_nestor_solve"),
        ("freeb_nestor_trial_sample_time_history", "freeb_nestor_trial_sample"),
        ("freeb_nestor_trial_solve_time_history", "freeb_nestor_trial_solve"),
    ):
        value = _sum_time(key)
        if value is not None:
            optimizer._profile_add(f"{profile_prefix}_{suffix}", value)
    for key, suffix in (
        ("freeb_full_update_history", "freeb_nestor_full_update_count"),
        ("freeb_nestor_reused_history", "freeb_nestor_reused_count"),
        ("freeb_nestor_trial_reused_history", "freeb_nestor_trial_reused_count"),
        ("freeb_nestor_trial_failed_history", "freeb_nestor_trial_failed_count"),
    ):
        value = _count_nonzero(key)
        if value is not None:
            optimizer._profile_add_counter(f"{profile_prefix}_{suffix}", value)


def profile_solver_timing(
    optimizer,
    diagnostics,
    *,
    profile_prefix: str,
    phase_wall_s: float,
    unattributed_name: str | None,
) -> float:
    """Record VMEC solver timing buckets and return attributed solver wall time."""

    if not isinstance(diagnostics, dict):
        return 0.0
    timing = diagnostics.get("timing")
    if not isinstance(timing, dict):
        optimizer._profile_solver_free_boundary_timing(diagnostics, profile_prefix=profile_prefix)
        return 0.0
    solver_total = 0.0
    timing_keys = (
        ("solve_total_s", "solve_total"),
        ("setup_total_s", "setup_total"),
        ("setup_axis_reset_s", "setup_axis_reset"),
        ("setup_axis_reset_compute_forces_s", "setup_axis_reset_compute_forces"),
        ("setup_axis_reset_unattributed_s", "setup_axis_reset_unattributed"),
        ("setup_unattributed_s", "setup_unattributed"),
        ("iteration_loop_s", "iteration_loop"),
        ("iteration_prepare_s", "iteration_prepare"),
        ("compute_forces_s", "compute_forces"),
        ("compute_forces_first_s", "compute_forces_first"),
        ("compute_forces_rest_s", "compute_forces_rest"),
        ("iteration_residual_metrics_s", "iteration_residual_metrics"),
        ("preconditioner_s", "preconditioner"),
        ("iteration_control_s", "iteration_control"),
        ("iteration_control_fsq1_s", "iteration_control_fsq1"),
        ("iteration_control_fsq1_precond_norm_s", "iteration_control_fsq1_precond_norm"),
        ("iteration_control_fsq1_scalar_build_s", "iteration_control_fsq1_scalar_build"),
        ("iteration_control_fsq1_payload_get_s", "iteration_control_fsq1_payload_get"),
        ("iteration_control_fsq1_direct_get_s", "iteration_control_fsq1_direct_get"),
        ("iteration_control_fsq1_unattributed_s", "iteration_control_fsq1_unattributed"),
        ("iteration_control_badjac_s", "iteration_control_badjac"),
        ("iteration_control_badjac_ptau_get_s", "iteration_control_badjac_ptau_get"),
        ("iteration_control_badjac_state_jacobian_s", "iteration_control_badjac_state_jacobian"),
        ("iteration_control_badjac_unattributed_s", "iteration_control_badjac_unattributed"),
        ("iteration_control_vmec_time_s", "iteration_control_vmec_time"),
        ("iteration_control_restart_s", "iteration_control_restart"),
        ("iteration_control_evolve_s", "iteration_control_evolve"),
        ("iteration_control_unattributed_s", "iteration_control_unattributed"),
        ("precond_refresh_s", "precond_refresh"),
        ("precond_apply_s", "preconditioner_apply"),
        ("precond_mode_scale_s", "preconditioner_mode_scale"),
        ("update_s", "update"),
        ("update_state_s", "update_state"),
        ("update_trace_build_s", "update_trace_build"),
        ("update_trace_finalize_s", "update_trace_finalize"),
        ("iteration_post_update_s", "iteration_post_update"),
        ("iteration_loop_unattributed_s", "iteration_loop_unattributed"),
        ("finalize_s", "finalize"),
        ("scan_total_s", "scan_total"),
        ("scan_setup_s", "scan_setup"),
        ("scan_initial_compute_forces_s", "scan_initial_compute_forces"),
        ("scan_axis_reset_compute_forces_s", "scan_axis_reset_compute_forces"),
        ("scan_run_setup_s", "scan_run_setup"),
        ("scan_runner_cache_lookup_s", "scan_runner_cache_lookup"),
        ("scan_runner_cache_build_s", "scan_runner_cache_build"),
        ("scan_preflight_s", "scan_preflight"),
        ("scan_device_run_s", "scan_device_run"),
        ("scan_device_dispatch_s", "scan_device_dispatch"),
        ("scan_device_ready_s", "scan_device_ready"),
        ("scan_runner_cache_hit_device_run_s", "scan_runner_cache_hit_device_run"),
        ("scan_runner_cache_hit_dispatch_s", "scan_runner_cache_hit_dispatch"),
        ("scan_runner_cache_hit_ready_s", "scan_runner_cache_hit_ready"),
        ("scan_runner_cache_miss_device_run_s", "scan_runner_cache_miss_device_run"),
        ("scan_runner_cache_miss_dispatch_s", "scan_runner_cache_miss_dispatch"),
        ("scan_runner_cache_miss_ready_s", "scan_runner_cache_miss_ready"),
        ("scan_runner_cache_bypass_device_run_s", "scan_runner_cache_bypass_device_run"),
        ("scan_runner_cache_bypass_dispatch_s", "scan_runner_cache_bypass_dispatch"),
        ("scan_runner_cache_bypass_ready_s", "scan_runner_cache_bypass_ready"),
        ("scan_host_materialize_s", "scan_host_materialize"),
        ("scan_postprocess_s", "scan_postprocess"),
        ("scan_unattributed_s", "scan_unattributed"),
    )
    counter_keys = (
        ("scan_runner_cache_hit_count", "scan_runner_cache_hit_count"),
        ("scan_runner_cache_miss_count", "scan_runner_cache_miss_count"),
        ("scan_runner_cache_bypass_count", "scan_runner_cache_bypass_count"),
    )
    outer_solver_total_keys = {"setup_total_s", "iteration_loop_s", "finalize_s", "scan_total_s"}
    fallback_solver_total_keys = {"compute_forces_s", "preconditioner_s", "update_s", "scan_total_s"}
    has_outer_solver_total = any(key in timing for key in outer_solver_total_keys)
    for key, suffix in timing_keys:
        if key not in timing:
            continue
        try:
            value = float(timing.get(key, 0.0))
        except Exception:
            continue
        optimizer._profile_add(f"{profile_prefix}_{suffix}", value)
        if key in (outer_solver_total_keys if has_outer_solver_total else fallback_solver_total_keys):
            solver_total += max(0.0, value)
    for key, suffix in counter_keys:
        if key not in timing:
            continue
        try:
            value = int(timing.get(key, 0))
        except Exception:
            continue
        optimizer._profile_add_counter(f"{profile_prefix}_{suffix}", value)
    optimizer._profile_solver_free_boundary_timing(diagnostics, profile_prefix=profile_prefix)
    for key, value_raw in sorted(timing.items()):
        if not (str(key).startswith("scan_runner_cache_miss_category_") and str(key).endswith("_count")):
            continue
        try:
            value = int(value_raw)
        except Exception:
            continue
        optimizer._profile_add_counter(f"{profile_prefix}_{key}", value)
    if unattributed_name is not None:
        optimizer._profile_add(unattributed_name, max(0.0, float(phase_wall_s) - solver_total))
    return solver_total


def profile_exact_tape_solver_timing(optimizer, tape, tape_build_wall_s: float) -> None:
    """Record solver and tape-construction timing for an accepted exact callback."""

    diagnostics = getattr(tape, "diagnostics", None)
    solver_total = optimizer._profile_solver_timing(
        diagnostics,
        profile_prefix="exact_tape_solver",
        phase_wall_s=tape_build_wall_s,
        unattributed_name=None,
    )
    timing = diagnostics.get("timing") if isinstance(diagnostics, dict) else None
    build_leaf_total = 0.0
    has_solve_call_timer = False
    if isinstance(timing, dict):
        for key, profile_name in _EXACT_TAPE_BUILD_TIMING_PROFILE_NAMES:
            if key not in timing:
                continue
            try:
                value = float(timing.get(key, 0.0))
            except Exception:
                continue
            optimizer._profile_add(profile_name, value)
            build_leaf_total += max(0.0, value)
            if key == "tape_solve_call_s":
                has_solve_call_timer = True
    attributed = build_leaf_total if has_solve_call_timer else solver_total + build_leaf_total
    optimizer._profile_add("exact_tape_build_unattributed", max(0.0, float(tape_build_wall_s) - attributed))


def profile_dump(optimizer) -> dict[str, dict[str, float | int]]:
    """Return a sorted profile summary with counts, total time, and mean time."""

    out: dict[str, dict[str, float | int]] = {}
    for name, rec in sorted(optimizer._profile.items()):
        count = int(rec.get("count", 0))
        total = float(rec.get("wall_time_s", 0.0))
        out[name] = {
            "count": count,
            "wall_time_s": total,
            "mean_wall_time_s": total / count if count else 0.0,
        }
    return out


def sync_replay_timing_enabled() -> bool:
    """Return whether replay profiling should force device synchronization."""

    flag = os.getenv("VMEC_JAX_OPT_SYNC_REPLAY_TIMING", "").strip().lower()
    return flag not in ("", "0", "false", "no", "off")


def profile_async_phase(optimizer, name: str, start: float, value):
    """Record dispatch time, optionally synchronizing for device-ready timing."""

    dispatch_s = time.perf_counter() - float(start)
    optimizer._profile_add(f"{name}_dispatch", dispatch_s)
    total_s = dispatch_s
    if optimizer._sync_replay_timing_enabled():
        try:
            from ... import _compat

            t_ready = time.perf_counter()
            value = _compat.jax.block_until_ready(value)
            ready_s = time.perf_counter() - t_ready
        except Exception:
            ready_s = 0.0
        optimizer._profile_add(f"{name}_ready", ready_s)
        total_s += ready_s
    optimizer._profile_add(name, total_s)
    return value


def profile_blocking_phase(optimizer, name: str, start: float, value):
    """Record dispatch and mandatory device-ready timing for a blocking phase."""

    dispatch_s = time.perf_counter() - float(start)
    optimizer._profile_add(f"{name}_dispatch", dispatch_s)
    try:
        from ... import _compat

        t_ready = time.perf_counter()
        value = _compat.jax.block_until_ready(value)
        ready_s = time.perf_counter() - t_ready
    except Exception:
        ready_s = 0.0
    optimizer._profile_add(f"{name}_ready", ready_s)
    optimizer._profile_add(name, dispatch_s + ready_s)
    return value
