#!/usr/bin/env python3
"""Compare vmec_jax profiling JSON reports.

The exact optimizer emits a few JSON shapes: callback-only profiles, short run
histories, and same-process repeated runs.  This tool normalizes those shapes
into production bottleneck metrics that can be compared across CPU/GPU or
before/after runs without launching another solver.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import json
from pathlib import Path
from typing import Any


PROFILE_TIME_GROUPS = {
    "compile_time_s": ("compile", "compilation"),
    "replay_time_s": ("replay",),
    "cache_time_s": ("cache",),
}

DIRECT_TIME_FIELDS = {
    "compile_time_s": (
        "compile_time_s",
        "compilation_time_s",
        "xla_compile_time_s",
        "compile_wall_time_s",
    ),
    "replay_time_s": ("replay_time_s", "tape_replay_time_s"),
    "cache_time_s": ("cache_time_s", "cache_wall_time_s"),
}

SOLVE_PROFILE_NAMES = {
    "solve_forward_trial",
    "solve_forward_exact",
    "exact_solve_with_tape_total",
    "scan_exact_state_solve",
}

EXACT_PROFILE_METRIC_NAMES = {
    "exact_tape_build_s": ("exact_tape_build",),
    "exact_tape_build_jvp_only_s": ("exact_tape_build_jvp_only",),
    "exact_tape_build_solve_call_s": ("exact_tape_build_solve_call",),
    "exact_tape_build_final_state_pack_s": ("exact_tape_build_final_state_pack",),
    "exact_tape_build_step_trace_extract_s": ("exact_tape_build_step_trace_extract",),
    "exact_tape_build_dynamic_payload_s": ("exact_tape_build_dynamic_payload",),
    "exact_tape_build_trace_stack_s": ("exact_tape_build_trace_stack",),
    "exact_tape_build_unattributed_s": ("exact_tape_build_unattributed",),
    "initial_tangents_s": ("jacobian_initial_tangents",),
    "initial_tangents_linearize_s": ("jacobian_initial_tangents_linearize",),
    "initial_tangents_vmap_s": ("jacobian_initial_tangents_vmap",),
    "initial_tangents_vmap_dispatch_s": ("jacobian_initial_tangents_vmap_dispatch",),
    "initial_tangents_vmap_ready_s": ("jacobian_initial_tangents_vmap_ready",),
    "initial_projection_s": (
        "gradient_initial_projection",
        "gradient_initial_vjp",
        "linear_operator_initial_vjp",
        "linear_operator_initial_transpose",
    ),
    "residual_tangents_s": ("jacobian_residual_tangents",),
    "residual_tangents_dispatch_s": ("jacobian_residual_tangents_dispatch",),
    "residual_tangents_ready_s": ("jacobian_residual_tangents_ready",),
    "projected_residual_tangents_s": ("jacobian_projected_replay_residual_tangents",),
    "projected_residual_tangents_dispatch_s": (
        "jacobian_projected_replay_residual_tangents_dispatch",
    ),
    "projected_residual_tangents_ready_s": (
        "jacobian_projected_replay_residual_tangents_ready",
    ),
    "projected_replay_total_s": (
        "jacobian_projected_replay_total",
        "jacobian_fused_projected_replay_total",
    ),
    "projected_replay_dispatch_s": ("jacobian_projected_tape_replay_dispatch",),
    "accepted_replay_dispatch_s": (
        "jacobian_tape_replay_dispatch",
        "jacobian_projected_tape_replay_dispatch",
        "gradient_tape_replay_dispatch",
        "state_tangent_tape_replay_dispatch",
        "b_cartesian_tangent_tape_replay_dispatch",
        "linear_operator_tape_vjp_dispatch",
    ),
    "accepted_replay_ready_s": (
        "jacobian_tape_replay_ready",
        "gradient_tape_replay_ready",
        "state_tangent_tape_replay_ready",
        "b_cartesian_tangent_tape_replay_ready",
        "linear_operator_tape_vjp_ready",
    ),
    "trial_solve_s": ("solve_forward_trial", "solve_forward_trial_total"),
    "trial_solver_compute_forces_s": ("trial_solver_compute_forces",),
    "trial_solver_compute_forces_first_s": ("trial_solver_compute_forces_first",),
    "trial_solver_compute_forces_rest_s": ("trial_solver_compute_forces_rest",),
    "trial_solver_preconditioner_s": ("trial_solver_preconditioner",),
    "trial_solver_precond_refresh_s": ("trial_solver_precond_refresh",),
    "trial_solver_preconditioner_apply_s": ("trial_solver_preconditioner_apply",),
    "trial_solver_preconditioner_mode_scale_s": ("trial_solver_preconditioner_mode_scale",),
    "trial_solver_update_s": ("trial_solver_update",),
    "trial_solver_update_state_s": ("trial_solver_update_state",),
    "trial_solver_scan_total_s": ("trial_solver_scan_total",),
    "trial_solver_scan_setup_s": ("trial_solver_scan_setup",),
    "trial_solver_scan_initial_compute_forces_s": ("trial_solver_scan_initial_compute_forces",),
    "trial_solver_scan_axis_reset_compute_forces_s": ("trial_solver_scan_axis_reset_compute_forces",),
    "trial_solver_scan_run_setup_s": ("trial_solver_scan_run_setup",),
    "trial_solver_scan_runner_cache_lookup_s": ("trial_solver_scan_runner_cache_lookup",),
    "trial_solver_scan_runner_cache_build_s": ("trial_solver_scan_runner_cache_build",),
    "trial_solver_scan_runner_cache_hit_count": ("trial_solver_scan_runner_cache_hit_count",),
    "trial_solver_scan_runner_cache_miss_count": ("trial_solver_scan_runner_cache_miss_count",),
    "trial_solver_scan_runner_cache_bypass_count": ("trial_solver_scan_runner_cache_bypass_count",),
    "trial_solver_scan_preflight_s": ("trial_solver_scan_preflight",),
    "trial_solver_scan_device_run_s": ("trial_solver_scan_device_run",),
    "trial_solver_scan_device_dispatch_s": ("trial_solver_scan_device_dispatch",),
    "trial_solver_scan_device_ready_s": ("trial_solver_scan_device_ready",),
    "trial_solver_scan_runner_cache_hit_device_run_s": ("trial_solver_scan_runner_cache_hit_device_run",),
    "trial_solver_scan_runner_cache_hit_dispatch_s": ("trial_solver_scan_runner_cache_hit_dispatch",),
    "trial_solver_scan_runner_cache_hit_ready_s": ("trial_solver_scan_runner_cache_hit_ready",),
    "trial_solver_scan_runner_cache_miss_device_run_s": ("trial_solver_scan_runner_cache_miss_device_run",),
    "trial_solver_scan_runner_cache_miss_dispatch_s": ("trial_solver_scan_runner_cache_miss_dispatch",),
    "trial_solver_scan_runner_cache_miss_ready_s": ("trial_solver_scan_runner_cache_miss_ready",),
    "trial_solver_scan_runner_cache_bypass_device_run_s": ("trial_solver_scan_runner_cache_bypass_device_run",),
    "trial_solver_scan_runner_cache_bypass_dispatch_s": ("trial_solver_scan_runner_cache_bypass_dispatch",),
    "trial_solver_scan_runner_cache_bypass_ready_s": ("trial_solver_scan_runner_cache_bypass_ready",),
    "trial_solver_scan_host_materialize_s": ("trial_solver_scan_host_materialize",),
    "trial_solver_scan_postprocess_s": ("trial_solver_scan_postprocess",),
    "trial_solver_scan_unattributed_s": ("trial_solver_scan_unattributed",),
    "trial_solve_unattributed_s": ("solve_forward_trial_unattributed",),
    "exact_solve_s": ("solve_forward_exact", "solve_forward_exact_total", "exact_solve_with_tape_total"),
    "exact_solve_with_tape_jvp_only_s": ("exact_solve_with_tape_jvp_only_total",),
    "forward_exact_solver_compute_forces_s": ("forward_exact_solver_compute_forces",),
    "forward_exact_solver_compute_forces_first_s": ("forward_exact_solver_compute_forces_first",),
    "forward_exact_solver_compute_forces_rest_s": ("forward_exact_solver_compute_forces_rest",),
    "forward_exact_solver_preconditioner_s": ("forward_exact_solver_preconditioner",),
    "forward_exact_solver_precond_refresh_s": ("forward_exact_solver_precond_refresh",),
    "forward_exact_solver_preconditioner_apply_s": ("forward_exact_solver_preconditioner_apply",),
    "forward_exact_solver_preconditioner_mode_scale_s": ("forward_exact_solver_preconditioner_mode_scale",),
    "forward_exact_solver_update_s": ("forward_exact_solver_update",),
    "forward_exact_solver_update_state_s": ("forward_exact_solver_update_state",),
    "forward_exact_solver_scan_total_s": ("forward_exact_solver_scan_total",),
    "forward_exact_solver_scan_setup_s": ("forward_exact_solver_scan_setup",),
    "forward_exact_solver_scan_initial_compute_forces_s": ("forward_exact_solver_scan_initial_compute_forces",),
    "forward_exact_solver_scan_axis_reset_compute_forces_s": (
        "forward_exact_solver_scan_axis_reset_compute_forces",
    ),
    "forward_exact_solver_scan_run_setup_s": ("forward_exact_solver_scan_run_setup",),
    "forward_exact_solver_scan_runner_cache_hit_count": ("forward_exact_solver_scan_runner_cache_hit_count",),
    "forward_exact_solver_scan_runner_cache_miss_count": ("forward_exact_solver_scan_runner_cache_miss_count",),
    "forward_exact_solver_scan_runner_cache_bypass_count": ("forward_exact_solver_scan_runner_cache_bypass_count",),
    "forward_exact_solver_scan_preflight_s": ("forward_exact_solver_scan_preflight",),
    "forward_exact_solver_scan_device_run_s": ("forward_exact_solver_scan_device_run",),
    "forward_exact_solver_scan_device_dispatch_s": ("forward_exact_solver_scan_device_dispatch",),
    "forward_exact_solver_scan_device_ready_s": ("forward_exact_solver_scan_device_ready",),
    "forward_exact_solver_scan_host_materialize_s": ("forward_exact_solver_scan_host_materialize",),
    "forward_exact_solver_scan_postprocess_s": ("forward_exact_solver_scan_postprocess",),
    "forward_exact_solver_scan_unattributed_s": ("forward_exact_solver_scan_unattributed",),
    "forward_exact_solve_unattributed_s": ("solve_forward_exact_unattributed",),
    "exact_tape_solver_solve_total_s": ("exact_tape_solver_solve_total",),
    "exact_tape_solver_setup_total_s": ("exact_tape_solver_setup_total",),
    "exact_tape_solver_setup_axis_reset_s": ("exact_tape_solver_setup_axis_reset",),
    "exact_tape_solver_setup_unattributed_s": ("exact_tape_solver_setup_unattributed",),
    "exact_tape_solver_iteration_loop_s": ("exact_tape_solver_iteration_loop",),
    "exact_tape_solver_iteration_prepare_s": ("exact_tape_solver_iteration_prepare",),
    "exact_tape_solver_compute_forces_s": ("exact_tape_solver_compute_forces",),
    "exact_tape_solver_compute_forces_first_s": ("exact_tape_solver_compute_forces_first",),
    "exact_tape_solver_compute_forces_rest_s": ("exact_tape_solver_compute_forces_rest",),
    "exact_tape_solver_iteration_residual_metrics_s": ("exact_tape_solver_iteration_residual_metrics",),
    "exact_tape_solver_preconditioner_s": ("exact_tape_solver_preconditioner",),
    "exact_tape_solver_precond_refresh_s": ("exact_tape_solver_precond_refresh",),
    "exact_tape_solver_preconditioner_apply_s": ("exact_tape_solver_preconditioner_apply",),
    "exact_tape_solver_preconditioner_mode_scale_s": ("exact_tape_solver_preconditioner_mode_scale",),
    "exact_tape_solver_update_s": ("exact_tape_solver_update",),
    "exact_tape_solver_update_state_s": ("exact_tape_solver_update_state",),
    "exact_tape_solver_iteration_post_update_s": ("exact_tape_solver_iteration_post_update",),
    "exact_tape_solver_iteration_loop_unattributed_s": ("exact_tape_solver_iteration_loop_unattributed",),
    "exact_tape_solver_finalize_s": ("exact_tape_solver_finalize",),
    "exact_tape_solver_scan_runner_cache_hit_count": ("exact_tape_solver_scan_runner_cache_hit_count",),
    "exact_tape_solver_scan_runner_cache_miss_count": ("exact_tape_solver_scan_runner_cache_miss_count",),
    "exact_tape_solver_scan_runner_cache_bypass_count": ("exact_tape_solver_scan_runner_cache_bypass_count",),
}

EXACT_PROFILE_CONTAINER_PRIORITY = {
    "trial_solve_s": (
        ("solve_forward_trial_total",),
        ("solve_forward_trial",),
    ),
    "exact_solve_s": (
        ("exact_solve_with_tape_total",),
        ("solve_forward_exact_total",),
        ("solve_forward_exact",),
    ),
}

ACCEPTED_REPLAY_PROFILE_NAMES = {
    "jacobian_tape_replay",
    "jacobian_projected_replay_total",
    "jacobian_fused_projected_replay_total",
    "gradient_tape_replay",
    "state_tangent_tape_replay",
    "b_cartesian_tangent_tape_replay",
    "linear_operator_tape_vjp",
}
ACCEPTED_REPLAY_DISPATCH_PROFILE_NAMES = {
    f"{name}_dispatch" for name in ACCEPTED_REPLAY_PROFILE_NAMES
} | {
    "jacobian_projected_tape_replay_dispatch",
}
ACCEPTED_REPLAY_READY_PROFILE_NAMES = {f"{name}_ready" for name in ACCEPTED_REPLAY_PROFILE_NAMES}

INITIAL_TANGENT_DETAIL_NAMES = {
    "jacobian_initial_tangents_cache_key",
    "jacobian_initial_tangents_eye",
    "jacobian_initial_tangents_linearize",
    "jacobian_initial_tangents_vmap",
    "jacobian_initial_tangents_vmap_dispatch",
    "jacobian_initial_tangents_vmap_ready",
}

SCAN_DEVICE_RUN_PROFILE_NAMES = {
    "trial_solver_scan_device_run",
    "forward_exact_solver_scan_device_run",
}

METRIC_ORDER = (
    "total_runtime_s",
    "vmec_solve_s",
    "vmec_compute_forces_s",
    "vmec_preconditioner_s",
    "vmec_precond_refresh_s",
    "vmec_precond_apply_s",
    "vmec_precond_mode_scale_s",
    "vmec_update_s",
    "vmec_update_state_s",
    "qi_first_call_s",
    "qi_warm_min_s",
    "qi_warm_mean_s",
    "exact_tape_build_s",
    "exact_tape_build_jvp_only_s",
    "exact_tape_build_solve_call_s",
    "exact_tape_build_final_state_pack_s",
    "exact_tape_build_step_trace_extract_s",
    "exact_tape_build_dynamic_payload_s",
    "exact_tape_build_trace_stack_s",
    "exact_tape_build_unattributed_s",
    "initial_tangents_s",
    "initial_tangents_linearize_s",
    "initial_tangents_vmap_s",
    "initial_tangents_vmap_dispatch_s",
    "initial_tangents_vmap_ready_s",
    "initial_projection_s",
    "residual_tangents_s",
    "projected_residual_tangents_s",
    "projected_replay_total_s",
    "projected_replay_dispatch_s",
    "accepted_replay_dispatch_s",
    "accepted_replay_ready_s",
    "trial_solve_s",
    "trial_solver_compute_forces_s",
    "trial_solver_compute_forces_first_s",
    "trial_solver_compute_forces_rest_s",
    "trial_solver_preconditioner_s",
    "trial_solver_precond_refresh_s",
    "trial_solver_preconditioner_apply_s",
    "trial_solver_preconditioner_mode_scale_s",
    "trial_solver_update_s",
    "trial_solver_update_state_s",
    "trial_solver_scan_total_s",
    "trial_solver_scan_setup_s",
    "trial_solver_scan_initial_compute_forces_s",
    "trial_solver_scan_axis_reset_compute_forces_s",
    "trial_solver_scan_run_setup_s",
    "trial_solver_scan_runner_cache_lookup_s",
    "trial_solver_scan_runner_cache_build_s",
    "trial_solver_scan_runner_cache_hit_count",
    "trial_solver_scan_runner_cache_miss_count",
    "trial_solver_scan_runner_cache_bypass_count",
    "trial_solver_scan_preflight_s",
    "trial_solver_scan_device_run_s",
    "trial_solver_scan_device_dispatch_s",
    "trial_solver_scan_device_ready_s",
    "trial_solver_scan_runner_cache_hit_device_run_s",
    "trial_solver_scan_runner_cache_hit_dispatch_s",
    "trial_solver_scan_runner_cache_hit_ready_s",
    "trial_solver_scan_runner_cache_miss_device_run_s",
    "trial_solver_scan_runner_cache_miss_dispatch_s",
    "trial_solver_scan_runner_cache_miss_ready_s",
    "trial_solver_scan_runner_cache_bypass_device_run_s",
    "trial_solver_scan_runner_cache_bypass_dispatch_s",
    "trial_solver_scan_runner_cache_bypass_ready_s",
    "trial_solver_scan_host_materialize_s",
    "trial_solver_scan_postprocess_s",
    "trial_solver_scan_unattributed_s",
    "trial_solve_unattributed_s",
    "exact_solve_s",
    "exact_solve_with_tape_jvp_only_s",
    "forward_exact_solver_compute_forces_s",
    "forward_exact_solver_compute_forces_first_s",
    "forward_exact_solver_compute_forces_rest_s",
    "forward_exact_solver_preconditioner_s",
    "forward_exact_solver_precond_refresh_s",
    "forward_exact_solver_preconditioner_apply_s",
    "forward_exact_solver_preconditioner_mode_scale_s",
    "forward_exact_solver_update_s",
    "forward_exact_solver_update_state_s",
    "forward_exact_solver_scan_total_s",
    "forward_exact_solver_scan_setup_s",
    "forward_exact_solver_scan_initial_compute_forces_s",
    "forward_exact_solver_scan_axis_reset_compute_forces_s",
    "forward_exact_solver_scan_run_setup_s",
    "forward_exact_solver_scan_runner_cache_hit_count",
    "forward_exact_solver_scan_runner_cache_miss_count",
    "forward_exact_solver_scan_runner_cache_bypass_count",
    "forward_exact_solver_scan_preflight_s",
    "forward_exact_solver_scan_device_run_s",
    "forward_exact_solver_scan_device_dispatch_s",
    "forward_exact_solver_scan_device_ready_s",
    "forward_exact_solver_scan_host_materialize_s",
    "forward_exact_solver_scan_postprocess_s",
    "forward_exact_solver_scan_unattributed_s",
    "forward_exact_solve_unattributed_s",
    "exact_tape_solver_solve_total_s",
    "exact_tape_solver_setup_total_s",
    "exact_tape_solver_setup_axis_reset_s",
    "exact_tape_solver_setup_unattributed_s",
    "exact_tape_solver_iteration_loop_s",
    "exact_tape_solver_iteration_prepare_s",
    "exact_tape_solver_compute_forces_s",
    "exact_tape_solver_compute_forces_first_s",
    "exact_tape_solver_compute_forces_rest_s",
    "exact_tape_solver_iteration_residual_metrics_s",
    "exact_tape_solver_preconditioner_s",
    "exact_tape_solver_precond_refresh_s",
    "exact_tape_solver_preconditioner_apply_s",
    "exact_tape_solver_preconditioner_mode_scale_s",
    "exact_tape_solver_update_s",
    "exact_tape_solver_update_state_s",
    "exact_tape_solver_iteration_post_update_s",
    "exact_tape_solver_iteration_loop_unattributed_s",
    "exact_tape_solver_finalize_s",
    "exact_tape_solver_scan_runner_cache_hit_count",
    "exact_tape_solver_scan_runner_cache_miss_count",
    "exact_tape_solver_scan_runner_cache_bypass_count",
    "replay_scan_cache_hit_count",
    "replay_scan_cache_miss_count",
    "replay_scan_cache_lookup_s",
    "replay_scan_cache_build_s",
    "compile_time_s",
    "replay_time_s",
    "cache_time_s",
    "contamination_warning_count",
    "callback_count",
    "rss_peak_mib",
    "solve_count",
    "accepted_point_replay_count",
    "cache_entry_growth",
    "cache_entries_after",
)

METRIC_LABELS = {
    "total_runtime_s": "total runtime",
    "vmec_solve_s": "VMEC solve",
    "vmec_compute_forces_s": "VMEC compute_forces",
    "vmec_preconditioner_s": "VMEC preconditioner",
    "vmec_precond_refresh_s": "VMEC precond refresh",
    "vmec_precond_apply_s": "VMEC precond apply",
    "vmec_precond_mode_scale_s": "VMEC precond mode scale",
    "vmec_update_s": "VMEC update",
    "vmec_update_state_s": "VMEC update state",
    "qi_first_call_s": "QI first call",
    "qi_warm_min_s": "QI warm min",
    "qi_warm_mean_s": "QI warm mean",
    "exact_tape_build_s": "exact tape build",
    "exact_tape_build_jvp_only_s": "exact tape build JVP-only",
    "exact_tape_build_solve_call_s": "exact tape build solve call",
    "exact_tape_build_final_state_pack_s": "exact tape build final state pack",
    "exact_tape_build_step_trace_extract_s": "exact tape build step trace extract",
    "exact_tape_build_dynamic_payload_s": "exact tape build dynamic payload",
    "exact_tape_build_trace_stack_s": "exact tape build trace stack",
    "exact_tape_build_unattributed_s": "exact tape build unattributed",
    "initial_tangents_s": "initial tangents",
    "initial_tangents_linearize_s": "initial tangents linearize",
    "initial_tangents_vmap_s": "initial tangents vmap",
    "initial_tangents_vmap_dispatch_s": "initial tangents vmap dispatch",
    "initial_tangents_vmap_ready_s": "initial tangents vmap ready",
    "initial_projection_s": "initial VJP/projection",
    "residual_tangents_s": "residual tangents",
    "projected_residual_tangents_s": "projected residual tangents",
    "projected_replay_total_s": "projected replay total",
    "projected_replay_dispatch_s": "projected replay dispatch",
    "accepted_replay_dispatch_s": "accepted replay dispatch",
    "accepted_replay_ready_s": "accepted replay ready",
    "trial_solve_s": "trial solve",
    "trial_solver_compute_forces_s": "trial solver compute_forces",
    "trial_solver_compute_forces_first_s": "trial solver first compute_forces",
    "trial_solver_compute_forces_rest_s": "trial solver remaining compute_forces",
    "trial_solver_preconditioner_s": "trial solver preconditioner",
    "trial_solver_precond_refresh_s": "trial solver precond refresh",
    "trial_solver_preconditioner_apply_s": "trial solver preconditioner apply",
    "trial_solver_preconditioner_mode_scale_s": "trial solver preconditioner mode scale",
    "trial_solver_update_s": "trial solver update",
    "trial_solver_update_state_s": "trial solver update state",
    "trial_solver_scan_total_s": "trial solver scan total",
    "trial_solver_scan_setup_s": "trial solver scan setup",
    "trial_solver_scan_initial_compute_forces_s": "trial solver scan initial force assembly",
    "trial_solver_scan_axis_reset_compute_forces_s": "trial solver scan axis-reset force assembly",
    "trial_solver_scan_run_setup_s": "trial solver scan run setup",
    "trial_solver_scan_runner_cache_lookup_s": "trial scan runner cache lookup",
    "trial_solver_scan_runner_cache_build_s": "trial scan runner cache build",
    "trial_solver_scan_runner_cache_hit_count": "trial scan runner cache hits",
    "trial_solver_scan_runner_cache_miss_count": "trial scan runner cache misses",
    "trial_solver_scan_runner_cache_bypass_count": "trial scan runner cache bypasses",
    "trial_solver_scan_preflight_s": "trial solver scan preflight",
    "trial_solver_scan_device_run_s": "trial solver scan device run",
    "trial_solver_scan_device_dispatch_s": "trial solver scan device dispatch",
    "trial_solver_scan_device_ready_s": "trial solver scan device ready",
    "trial_solver_scan_runner_cache_hit_device_run_s": "trial scan cache-hit device run",
    "trial_solver_scan_runner_cache_hit_dispatch_s": "trial scan cache-hit dispatch",
    "trial_solver_scan_runner_cache_hit_ready_s": "trial scan cache-hit ready",
    "trial_solver_scan_runner_cache_miss_device_run_s": "trial scan cache-miss device run",
    "trial_solver_scan_runner_cache_miss_dispatch_s": "trial scan cache-miss dispatch",
    "trial_solver_scan_runner_cache_miss_ready_s": "trial scan cache-miss ready",
    "trial_solver_scan_runner_cache_bypass_device_run_s": "trial scan cache-bypass device run",
    "trial_solver_scan_runner_cache_bypass_dispatch_s": "trial scan cache-bypass dispatch",
    "trial_solver_scan_runner_cache_bypass_ready_s": "trial scan cache-bypass ready",
    "trial_solver_scan_host_materialize_s": "trial solver scan host materialize",
    "trial_solver_scan_postprocess_s": "trial solver scan postprocess",
    "trial_solver_scan_unattributed_s": "trial solver scan unattributed",
    "trial_solve_unattributed_s": "trial solve unattributed",
    "exact_solve_s": "exact solve",
    "exact_solve_with_tape_jvp_only_s": "exact solve with tape JVP-only",
    "forward_exact_solver_compute_forces_s": "forward exact solver compute_forces",
    "forward_exact_solver_compute_forces_first_s": "forward exact solver first compute_forces",
    "forward_exact_solver_compute_forces_rest_s": "forward exact solver remaining compute_forces",
    "forward_exact_solver_preconditioner_s": "forward exact solver preconditioner",
    "forward_exact_solver_precond_refresh_s": "forward exact solver precond refresh",
    "forward_exact_solver_preconditioner_apply_s": "forward exact solver preconditioner apply",
    "forward_exact_solver_preconditioner_mode_scale_s": "forward exact solver preconditioner mode scale",
    "forward_exact_solver_update_s": "forward exact solver update",
    "forward_exact_solver_update_state_s": "forward exact solver update state",
    "forward_exact_solver_scan_total_s": "forward exact solver scan total",
    "forward_exact_solver_scan_setup_s": "forward exact solver scan setup",
    "forward_exact_solver_scan_initial_compute_forces_s": "forward exact solver scan initial force assembly",
    "forward_exact_solver_scan_axis_reset_compute_forces_s": (
        "forward exact solver scan axis-reset force assembly"
    ),
    "forward_exact_solver_scan_run_setup_s": "forward exact solver scan run setup",
    "forward_exact_solver_scan_runner_cache_hit_count": "forward exact scan runner cache hits",
    "forward_exact_solver_scan_runner_cache_miss_count": "forward exact scan runner cache misses",
    "forward_exact_solver_scan_runner_cache_bypass_count": "forward exact scan runner cache bypasses",
    "forward_exact_solver_scan_preflight_s": "forward exact solver scan preflight",
    "forward_exact_solver_scan_device_run_s": "forward exact solver scan device run",
    "forward_exact_solver_scan_device_dispatch_s": "forward exact solver scan device dispatch",
    "forward_exact_solver_scan_device_ready_s": "forward exact solver scan device ready",
    "forward_exact_solver_scan_host_materialize_s": "forward exact solver scan host materialize",
    "forward_exact_solver_scan_postprocess_s": "forward exact solver scan postprocess",
    "forward_exact_solver_scan_unattributed_s": "forward exact solver scan unattributed",
    "forward_exact_solve_unattributed_s": "forward exact solve unattributed",
    "exact_tape_solver_solve_total_s": "exact tape solver total",
    "exact_tape_solver_setup_total_s": "exact tape solver setup total",
    "exact_tape_solver_setup_axis_reset_s": "exact tape solver setup axis reset",
    "exact_tape_solver_setup_unattributed_s": "exact tape solver setup unattributed",
    "exact_tape_solver_iteration_loop_s": "exact tape solver iteration loop",
    "exact_tape_solver_iteration_prepare_s": "exact tape solver iteration prepare",
    "exact_tape_solver_compute_forces_s": "exact tape solver compute_forces",
    "exact_tape_solver_compute_forces_first_s": "exact tape solver first compute_forces",
    "exact_tape_solver_compute_forces_rest_s": "exact tape solver remaining compute_forces",
    "exact_tape_solver_iteration_residual_metrics_s": "exact tape solver residual metrics",
    "exact_tape_solver_preconditioner_s": "exact tape solver preconditioner",
    "exact_tape_solver_precond_refresh_s": "exact tape solver precond refresh",
    "exact_tape_solver_preconditioner_apply_s": "exact tape solver preconditioner apply",
    "exact_tape_solver_preconditioner_mode_scale_s": "exact tape solver preconditioner mode scale",
    "exact_tape_solver_update_s": "exact tape solver update",
    "exact_tape_solver_update_state_s": "exact tape solver update state",
    "exact_tape_solver_iteration_post_update_s": "exact tape solver post-update",
    "exact_tape_solver_iteration_loop_unattributed_s": "exact tape solver loop unattributed",
    "exact_tape_solver_finalize_s": "exact tape solver finalize",
    "exact_tape_solver_scan_runner_cache_hit_count": "exact tape solver scan runner cache hits",
    "exact_tape_solver_scan_runner_cache_miss_count": "exact tape solver scan runner cache misses",
    "exact_tape_solver_scan_runner_cache_bypass_count": "exact tape solver scan runner cache bypasses",
    "replay_scan_cache_hit_count": "replay scan-cache hits",
    "replay_scan_cache_miss_count": "replay scan-cache misses",
    "replay_scan_cache_lookup_s": "replay scan-cache lookup",
    "replay_scan_cache_build_s": "replay scan-cache build",
    "compile_time_s": "compile time",
    "replay_time_s": "replay time",
    "cache_time_s": "cache time",
    "contamination_warning_count": "warnings",
    "callback_count": "callbacks",
    "rss_peak_mib": "RSS peak",
    "solve_count": "solves",
    "accepted_point_replay_count": "accepted replays",
    "cache_entry_growth": "cache entry growth",
    "cache_entries_after": "cache entries after",
}

BOTTLENECK_METRICS = (
    ("qi_first_call_s", "QI/Boozer first call"),
    ("vmec_compute_forces_s", "VMEC force assembly"),
    ("vmec_preconditioner_s", "VMEC preconditioner"),
    ("vmec_update_s", "VMEC state update"),
    ("exact_tape_build_s", "exact tape build"),
    ("exact_tape_build_jvp_only_s", "exact tape build JVP-only"),
    ("exact_tape_build_solve_call_s", "exact tape build solve call"),
    ("exact_tape_build_final_state_pack_s", "exact tape build final state pack"),
    ("exact_tape_build_step_trace_extract_s", "exact tape build step-trace extract"),
    ("exact_tape_build_dynamic_payload_s", "exact tape build dynamic payload"),
    ("exact_tape_build_trace_stack_s", "exact tape build trace stacking"),
    ("exact_tape_build_unattributed_s", "unattributed tape build"),
    ("initial_tangents_s", "initial tangent build"),
    ("initial_tangents_linearize_s", "initial tangent linearize"),
    ("initial_tangents_vmap_dispatch_s", "initial tangent vmap dispatch"),
    ("initial_tangents_vmap_ready_s", "initial tangent vmap ready"),
    ("initial_projection_s", "initial VJP/projection"),
    ("residual_tangents_s", "residual tangent projection"),
    ("projected_residual_tangents_s", "projected residual tangent projection"),
    ("projected_replay_total_s", "projected replay total"),
    ("projected_replay_dispatch_s", "projected replay dispatch"),
    ("accepted_replay_dispatch_s", "accepted-point replay dispatch"),
    ("accepted_replay_ready_s", "accepted-point replay ready"),
    ("trial_solve_s", "trial solve"),
    ("trial_solver_compute_forces_s", "trial solver force assembly"),
    ("trial_solver_compute_forces_first_s", "trial solver first force assembly"),
    ("trial_solver_compute_forces_rest_s", "trial solver remaining force assembly"),
    ("trial_solver_preconditioner_s", "trial solver preconditioner"),
    ("trial_solver_precond_refresh_s", "trial solver preconditioner refresh"),
    ("trial_solver_preconditioner_apply_s", "trial solver preconditioner apply"),
    ("trial_solver_preconditioner_mode_scale_s", "trial solver preconditioner mode scale"),
    ("trial_solver_update_state_s", "trial solver update state"),
    ("trial_solver_scan_setup_s", "trial scan setup"),
    ("trial_solver_scan_initial_compute_forces_s", "trial scan initial force assembly"),
    ("trial_solver_scan_axis_reset_compute_forces_s", "trial scan axis-reset force assembly"),
    ("trial_solver_scan_run_setup_s", "trial scan run setup"),
    ("trial_solver_scan_runner_cache_lookup_s", "trial scan cache lookup"),
    ("trial_solver_scan_runner_cache_build_s", "trial scan cache build"),
    ("trial_solver_scan_preflight_s", "trial scan preflight"),
    ("trial_solver_scan_device_run_s", "trial scan device run"),
    ("trial_solver_scan_device_dispatch_s", "trial scan device dispatch"),
    ("trial_solver_scan_device_ready_s", "trial scan device ready"),
    ("trial_solver_scan_runner_cache_hit_device_run_s", "trial scan cache-hit device run"),
    ("trial_solver_scan_runner_cache_hit_dispatch_s", "trial scan cache-hit dispatch"),
    ("trial_solver_scan_runner_cache_hit_ready_s", "trial scan cache-hit ready"),
    ("trial_solver_scan_runner_cache_miss_device_run_s", "trial scan cache-miss device run"),
    ("trial_solver_scan_runner_cache_miss_dispatch_s", "trial scan cache-miss dispatch"),
    ("trial_solver_scan_runner_cache_miss_ready_s", "trial scan cache-miss ready"),
    ("trial_solver_scan_runner_cache_bypass_device_run_s", "trial scan cache-bypass device run"),
    ("trial_solver_scan_runner_cache_bypass_dispatch_s", "trial scan cache-bypass dispatch"),
    ("trial_solver_scan_runner_cache_bypass_ready_s", "trial scan cache-bypass ready"),
    ("trial_solver_scan_host_materialize_s", "trial scan host materialize"),
    ("trial_solver_scan_postprocess_s", "trial scan postprocess"),
    ("trial_solver_scan_unattributed_s", "trial scan unattributed"),
    ("trial_solve_unattributed_s", "trial solver unattributed"),
    ("exact_solve_s", "accepted exact solve"),
    ("exact_solve_with_tape_jvp_only_s", "accepted exact JVP-only solve"),
    ("forward_exact_solver_compute_forces_s", "forward exact solver force assembly"),
    ("forward_exact_solver_compute_forces_first_s", "forward exact solver first force assembly"),
    ("forward_exact_solver_compute_forces_rest_s", "forward exact solver remaining force assembly"),
    ("forward_exact_solver_preconditioner_s", "forward exact solver preconditioner"),
    ("forward_exact_solver_precond_refresh_s", "forward exact solver preconditioner refresh"),
    ("forward_exact_solver_preconditioner_apply_s", "forward exact solver preconditioner apply"),
    ("forward_exact_solver_preconditioner_mode_scale_s", "forward exact solver preconditioner mode scale"),
    ("forward_exact_solver_update_state_s", "forward exact solver update state"),
    ("forward_exact_solver_scan_setup_s", "forward exact scan setup"),
    ("forward_exact_solver_scan_initial_compute_forces_s", "forward exact scan initial force assembly"),
    ("forward_exact_solver_scan_axis_reset_compute_forces_s", "forward exact scan axis-reset force assembly"),
    ("forward_exact_solver_scan_run_setup_s", "forward exact scan run setup"),
    ("forward_exact_solver_scan_preflight_s", "forward exact scan preflight"),
    ("forward_exact_solver_scan_device_run_s", "forward exact scan device run"),
    ("forward_exact_solver_scan_device_dispatch_s", "forward exact scan device dispatch"),
    ("forward_exact_solver_scan_device_ready_s", "forward exact scan device ready"),
    ("forward_exact_solver_scan_host_materialize_s", "forward exact scan host materialize"),
    ("forward_exact_solver_scan_postprocess_s", "forward exact scan postprocess"),
    ("forward_exact_solver_scan_unattributed_s", "forward exact scan unattributed"),
    ("forward_exact_solve_unattributed_s", "forward exact solver unattributed"),
    ("exact_tape_solver_setup_axis_reset_s", "exact tape solver setup axis reset"),
    ("exact_tape_solver_setup_unattributed_s", "exact tape solver setup unattributed"),
    ("exact_tape_solver_iteration_prepare_s", "exact tape solver iteration preparation"),
    ("exact_tape_solver_compute_forces_s", "exact tape solver force assembly"),
    ("exact_tape_solver_compute_forces_first_s", "exact tape solver first force assembly"),
    ("exact_tape_solver_compute_forces_rest_s", "exact tape solver remaining force assembly"),
    ("exact_tape_solver_iteration_residual_metrics_s", "exact tape solver residual metrics"),
    ("exact_tape_solver_preconditioner_s", "exact tape solver preconditioner"),
    ("exact_tape_solver_precond_refresh_s", "exact tape solver preconditioner refresh"),
    ("exact_tape_solver_preconditioner_apply_s", "exact tape solver preconditioner apply"),
    ("exact_tape_solver_preconditioner_mode_scale_s", "exact tape solver preconditioner mode scale"),
    ("exact_tape_solver_update_s", "exact tape solver update"),
    ("exact_tape_solver_update_state_s", "exact tape solver update state"),
    ("exact_tape_solver_iteration_post_update_s", "exact tape solver post-update"),
    ("exact_tape_solver_iteration_loop_unattributed_s", "exact tape solver loop unattributed"),
    ("exact_tape_solver_finalize_s", "exact tape solver finalization"),
    ("replay_scan_cache_build_s", "replay scan-cache build"),
    ("replay_time_s", "accepted-point replay"),
    ("compile_time_s", "compile/JIT"),
    ("cache_time_s", "cache bookkeeping"),
)

EXACT_OPTIMIZER_PATCH_TARGET_NAMES = {
    "exact_tape_build_unattributed",
    "exact_tape_build_solve_call",
    "exact_tape_build_final_state_pack",
    "exact_tape_build_step_trace_extract",
    "exact_tape_build_dynamic_payload",
    "exact_tape_build_trace_stack",
    "jacobian_tape_replay",
    "jacobian_projected_tape_replay_dispatch",
    "jacobian_projected_replay_residual_tangents",
    "jacobian_projected_replay_total",
    "jacobian_fused_projected_replay_total",
    "gradient_tape_replay",
    "state_tangent_tape_replay",
    "b_cartesian_tangent_tape_replay",
    "linear_operator_tape_vjp",
    "jacobian_tape_replay_dispatch",
    "jacobian_tape_replay_ready",
    "gradient_tape_replay_dispatch",
    "gradient_tape_replay_ready",
    "state_tangent_tape_replay_dispatch",
    "state_tangent_tape_replay_ready",
    "b_cartesian_tangent_tape_replay_dispatch",
    "b_cartesian_tangent_tape_replay_ready",
    "linear_operator_tape_vjp_dispatch",
    "linear_operator_tape_vjp_ready",
    "jacobian_initial_tangents",
    "jacobian_initial_tangents_eye",
    "jacobian_initial_tangents_linearize",
    "jacobian_initial_tangents_vmap",
    "jacobian_initial_tangents_vmap_dispatch",
    "jacobian_initial_tangents_vmap_ready",
    "gradient_initial_vjp",
    "linear_operator_initial_vjp",
    "linear_operator_initial_transpose",
    "jacobian_residual_tangents",
    "gradient_residual_vjp",
    "exact_unpack_cache",
    "initial_guess_trial",
    "initial_guess_forward",
    "initial_guess_exact",
    "trial_residual_exact_cache_hit",
    "residual_eval_trial",
    "residual_eval_exact",
    "scan_residual_eval_exact",
    "trial_solver_compute_forces",
    "trial_solver_compute_forces_first",
    "trial_solver_compute_forces_rest",
    "trial_solver_preconditioner",
    "trial_solver_precond_refresh",
    "trial_solver_preconditioner_apply",
    "trial_solver_preconditioner_mode_scale",
    "trial_solver_update",
    "trial_solver_update_state",
    "trial_solver_scan_total",
    "trial_solver_scan_setup",
    "trial_solver_scan_initial_compute_forces",
    "trial_solver_scan_axis_reset_compute_forces",
    "trial_solver_scan_run_setup",
    "trial_solver_scan_runner_cache_lookup",
    "trial_solver_scan_runner_cache_build",
    "trial_solver_scan_preflight",
    "trial_solver_scan_device_run",
    "trial_solver_scan_device_dispatch",
    "trial_solver_scan_device_ready",
    "trial_solver_scan_runner_cache_hit_device_run",
    "trial_solver_scan_runner_cache_hit_dispatch",
    "trial_solver_scan_runner_cache_hit_ready",
    "trial_solver_scan_runner_cache_miss_device_run",
    "trial_solver_scan_runner_cache_miss_dispatch",
    "trial_solver_scan_runner_cache_miss_ready",
    "trial_solver_scan_runner_cache_bypass_device_run",
    "trial_solver_scan_runner_cache_bypass_dispatch",
    "trial_solver_scan_runner_cache_bypass_ready",
    "trial_solver_scan_host_materialize",
    "trial_solver_scan_postprocess",
    "trial_solver_scan_unattributed",
    "solve_forward_trial_unattributed",
    "forward_exact_solver_compute_forces",
    "forward_exact_solver_compute_forces_first",
    "forward_exact_solver_compute_forces_rest",
    "forward_exact_solver_preconditioner",
    "forward_exact_solver_precond_refresh",
    "forward_exact_solver_preconditioner_apply",
    "forward_exact_solver_preconditioner_mode_scale",
    "forward_exact_solver_update",
    "forward_exact_solver_update_state",
    "forward_exact_solver_scan_total",
    "forward_exact_solver_scan_setup",
    "forward_exact_solver_scan_initial_compute_forces",
    "forward_exact_solver_scan_axis_reset_compute_forces",
    "forward_exact_solver_scan_run_setup",
    "forward_exact_solver_scan_preflight",
    "forward_exact_solver_scan_device_run",
    "forward_exact_solver_scan_device_dispatch",
    "forward_exact_solver_scan_device_ready",
    "forward_exact_solver_scan_host_materialize",
    "forward_exact_solver_scan_postprocess",
    "forward_exact_solver_scan_unattributed",
    "solve_forward_exact_unattributed",
    "exact_tape_solver_setup_axis_reset",
    "exact_tape_solver_setup_unattributed",
    "exact_tape_solver_iteration_prepare",
    "exact_tape_solver_compute_forces",
    "exact_tape_solver_compute_forces_first",
    "exact_tape_solver_compute_forces_rest",
    "exact_tape_solver_iteration_residual_metrics",
    "exact_tape_solver_preconditioner",
    "exact_tape_solver_precond_refresh",
    "exact_tape_solver_preconditioner_apply",
    "exact_tape_solver_preconditioner_mode_scale",
    "exact_tape_solver_update",
    "exact_tape_solver_update_state",
    "exact_tape_solver_iteration_post_update",
    "exact_tape_solver_iteration_loop_unattributed",
    "exact_tape_solver_finalize",
}

EXACT_TAPE_SOLVE_CALL_DETAIL_NAMES = {
    "exact_tape_solver_setup_axis_reset",
    "exact_tape_solver_setup_unattributed",
    "exact_tape_solver_iteration_prepare",
    "exact_tape_solver_compute_forces_first",
    "exact_tape_solver_compute_forces_rest",
    "exact_tape_solver_iteration_residual_metrics",
    "exact_tape_solver_precond_refresh",
    "exact_tape_solver_preconditioner_apply",
    "exact_tape_solver_preconditioner_mode_scale",
    "exact_tape_solver_update_state",
    "exact_tape_solver_iteration_post_update",
    "exact_tape_solver_iteration_loop_unattributed",
    "exact_tape_solver_finalize",
}

EXACT_OPTIMIZER_CONTAINER_PROFILE_NAMES = {
    "exact_solve_with_tape_total",
    "solve_forward_trial_total",
    "solve_forward_exact_total",
    "gradient_total",
    "jacobian_total",
    "linear_operator_total",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two or more vmec_jax profiling JSON reports and emit "
            "CPU/GPU or before/after bottleneck ratios."
        )
    )
    parser.add_argument("reports", nargs="+", type=Path, help="Profile JSON report paths.")
    parser.add_argument(
        "--label",
        action="append",
        default=None,
        help="Label for a report. Repeat once per input path.",
    )
    parser.add_argument(
        "--baseline",
        default="0",
        help="Baseline label or zero-based report index for ratios (default: 0).",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format printed to stdout.",
    )
    parser.add_argument("--json-out", type=Path, default=None, help="Optional path for machine-readable JSON.")
    parser.add_argument(
        "--top-profile",
        type=int,
        default=5,
        help="Number of largest profile terms to include per report.",
    )
    return parser


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _get_path(tree: Any, keys: Iterable[str]) -> Any:
    value = tree
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _sum_optional(values: Iterable[float | None]) -> float | None:
    total = 0.0
    found = False
    for value in values:
        if value is None:
            continue
        total += float(value)
        found = True
    return total if found else None


def _profile_from_payload(payload: dict[str, Any]) -> dict[str, dict[str, float | int]]:
    profile = payload.get("profile")
    if isinstance(profile, dict):
        return _normalize_profile(profile)

    runs = payload.get("runs")
    if isinstance(runs, list):
        merged: dict[str, dict[str, float | int]] = {}
        for run in runs:
            if not isinstance(run, dict):
                continue
            _merge_profile(merged, _profile_from_payload(run))
        return _finalize_profile(merged)

    return {}


def _normalize_profile(profile: dict[str, Any]) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    for name, rec in profile.items():
        if not isinstance(rec, dict):
            continue
        count = _as_int(rec.get("count")) or 0
        wall = _as_float(rec.get("wall_time_s")) or 0.0
        out[str(name)] = {
            "count": count,
            "wall_time_s": wall,
            "mean_wall_time_s": wall / count if count else 0.0,
        }
    return out


def _merge_profile(
    target: dict[str, dict[str, float | int]],
    source: dict[str, dict[str, float | int]],
) -> None:
    for name, rec in source.items():
        out = target.setdefault(name, {"count": 0, "wall_time_s": 0.0})
        out["count"] = int(out.get("count", 0)) + int(rec.get("count", 0))
        out["wall_time_s"] = float(out.get("wall_time_s", 0.0)) + float(rec.get("wall_time_s", 0.0))


def _finalize_profile(profile: dict[str, dict[str, float | int]]) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    for name, rec in sorted(profile.items()):
        count = int(rec.get("count", 0))
        wall = float(rec.get("wall_time_s", 0.0))
        out[name] = {
            "count": count,
            "wall_time_s": wall,
            "mean_wall_time_s": wall / count if count else 0.0,
        }
    return out


def _source_kind(payload: dict[str, Any]) -> str:
    kind = payload.get("report_kind")
    if isinstance(kind, str):
        return kind
    if isinstance(payload.get("runs"), list):
        return "exact_optimizer_run_repeats"
    if "wall_time_sec" in payload and "diagnostics" in payload:
        return "fixed_boundary_profile"
    if "qi_evaluations" in payload or "qi_resolution" in payload:
        return "qi_boozer_profile"
    if "history" in payload and "profile" in payload:
        return "exact_optimizer_run_history"
    if "profile" in payload:
        return "profile_report"
    return "unknown"


def _total_runtime(payload: dict[str, Any]) -> float | None:
    direct = next(
        (
            value
            for value in (
                _as_float(payload.get("total_wall_time_s")),
                _as_float(payload.get("wall_time_sec")),
                _as_float(payload.get("wall_time_s")),
                _as_float(payload.get("runtime_s")),
            )
            if value is not None
        ),
        None,
    )
    if direct is not None:
        return direct

    runs = payload.get("runs")
    if isinstance(runs, list):
        return _sum_optional(_total_runtime(run) for run in runs if isinstance(run, dict))
    return None


def _wall_time_metric(payload: dict[str, Any], key: str) -> float | None:
    value = _as_float(_get_path(payload, ("wall_time_s", key)))
    if value is not None:
        return value
    runs = payload.get("runs")
    if isinstance(runs, list):
        return _sum_optional(_wall_time_metric(run, key) for run in runs if isinstance(run, dict))
    return None


def _contamination_warning_count(payload: dict[str, Any]) -> int | None:
    warnings = payload.get("contamination_warnings")
    if isinstance(warnings, list):
        return len(warnings)
    runs = payload.get("runs")
    if isinstance(runs, list):
        values = [_contamination_warning_count(run) for run in runs if isinstance(run, dict)]
        if any(value is not None for value in values):
            return sum(int(value or 0) for value in values)
    return None


def _first_present(*values: float | None) -> float | None:
    for value in values:
        if value is not None:
            return value
    return None


def _direct_time(payload: dict[str, Any], field: str) -> float | None:
    candidates = list(DIRECT_TIME_FIELDS[field])
    current = next(
        (
            value
            for key in candidates
            for value in (
                _as_float(payload.get(key)),
                _as_float(_get_path(payload, ("timing", key))),
                _as_float(_get_path(payload, ("diagnostics", "timing", key))),
            )
            if value is not None
        ),
        None,
    )
    if current is not None:
        return current
    runs = payload.get("runs")
    if isinstance(runs, list):
        return _sum_optional(_direct_time(run, field) for run in runs if isinstance(run, dict))
    return None


def _vmec_timing_metric(payload: dict[str, Any], key: str) -> float | None:
    value = _as_float(_get_path(payload, ("diagnostics", "timing", key)))
    if value is not None:
        return value
    value = _as_float(_get_path(payload, ("timing", key)))
    if value is not None:
        return value
    runs = payload.get("runs")
    if isinstance(runs, list):
        return _sum_optional(_vmec_timing_metric(run, key) for run in runs if isinstance(run, dict))
    return None


def _profile_time(profile: dict[str, dict[str, float | int]], field: str) -> float | None:
    tokens = PROFILE_TIME_GROUPS[field]
    values = [
        float(rec.get("wall_time_s", 0.0))
        for name, rec in profile.items()
        if any(token in name.lower() for token in tokens)
    ]
    return sum(values) if values else None


def _profile_named_time(profile: dict[str, dict[str, float | int]], names: Iterable[str]) -> float | None:
    """Return summed wall time for exact profile names that are present."""

    name_set = {str(name) for name in names}
    values = [
        float(rec.get("wall_time_s", 0.0))
        for name, rec in profile.items()
        if str(name) in name_set
    ]
    return sum(values) if values else None


def _profile_named_count(profile: dict[str, dict[str, float | int]], names: Iterable[str]) -> int | None:
    """Return summed counts for exact profile names that are present."""

    name_set = {str(name) for name in names}
    values = [
        int(rec.get("count", 0))
        for name, rec in profile.items()
        if str(name) in name_set
    ]
    return sum(values) if values else None


def _profile_metric_time(
    profile: dict[str, dict[str, float | int]],
    metric: str,
    names: Iterable[str],
) -> float | None:
    """Return a phase time without double-counting enclosing total timers."""

    priority = EXACT_PROFILE_CONTAINER_PRIORITY.get(metric)
    if priority is not None:
        for group in priority:
            value = _profile_named_time(profile, group)
            if value is not None:
                return value
        return None
    return _profile_named_time(profile, names)


def _accepted_replay_profile_time(profile: dict[str, dict[str, float | int]]) -> float | None:
    """Return accepted replay time without double-counting split child timers."""

    total = _profile_named_time(profile, ACCEPTED_REPLAY_PROFILE_NAMES)
    if total is not None:
        return total
    return _profile_named_time(
        profile,
        ACCEPTED_REPLAY_DISPATCH_PROFILE_NAMES | ACCEPTED_REPLAY_READY_PROFILE_NAMES,
    )


def _callback_count(payload: dict[str, Any]) -> int | None:
    trace = payload.get("callback_trace")
    if isinstance(trace, dict):
        events = trace.get("events")
        if isinstance(events, list):
            return len(events)
        summary = trace.get("summary")
        if isinstance(summary, dict):
            total = 0
            found = False
            for rec in summary.values():
                if isinstance(rec, dict) and "count" in rec:
                    total += int(rec.get("count", 0))
                    found = True
            if found:
                return total

    samples = payload.get("samples")
    if isinstance(samples, list):
        return len(samples)

    runs = payload.get("runs")
    if isinstance(runs, list):
        values = [_callback_count(run) for run in runs if isinstance(run, dict)]
        if any(value is not None for value in values):
            return sum(int(value or 0) for value in values)

    nfev = _as_int(payload.get("nfev"))
    njev = _as_int(payload.get("njev"))
    if nfev is not None or njev is not None:
        return int(nfev or 0) + int(njev or 0)
    return None


def _accepted_replay_count(payload: dict[str, Any], profile: dict[str, dict[str, float | int]]) -> int | None:
    trace = payload.get("callback_trace")
    if isinstance(trace, dict):
        summary = trace.get("summary")
        if isinstance(summary, dict):
            total = 0
            found = False
            for key, rec in summary.items():
                if not isinstance(rec, dict):
                    continue
                key_l = str(key).lower()
                if "exact_tape_replay" in key_l or key_l.endswith(":tape_replay"):
                    total += int(rec.get("count", 0))
                    found = True
            if found:
                return total

    if any(name in profile for name in ACCEPTED_REPLAY_PROFILE_NAMES):
        return sum(int(profile[name].get("count", 0)) for name in ACCEPTED_REPLAY_PROFILE_NAMES if name in profile)

    runs = payload.get("runs")
    if isinstance(runs, list):
        values = [
            _accepted_replay_count(run, _profile_from_payload(run))
            for run in runs
            if isinstance(run, dict)
        ]
        if any(value is not None for value in values):
            return sum(int(value or 0) for value in values)
    return None


def _solve_count(payload: dict[str, Any], profile: dict[str, dict[str, float | int]]) -> int | None:
    direct = _as_int(payload.get("solve_count"))
    if direct is not None:
        return direct

    if any(name in profile for name in SOLVE_PROFILE_NAMES):
        return sum(int(profile[name].get("count", 0)) for name in SOLVE_PROFILE_NAMES if name in profile)

    trace = payload.get("callback_trace")
    if isinstance(trace, dict):
        summary = trace.get("summary")
        if isinstance(summary, dict):
            total = 0
            found = False
            for key, rec in summary.items():
                if not isinstance(rec, dict):
                    continue
                key_l = str(key).lower()
                if "trial_solve" in key_l or "exact_tape_replay" in key_l:
                    total += int(rec.get("count", 0))
                    found = True
            if found:
                return total

    runs = payload.get("runs")
    if isinstance(runs, list):
        values = [_solve_count(run, _profile_from_payload(run)) for run in runs if isinstance(run, dict)]
        if any(value is not None for value in values):
            return sum(int(value or 0) for value in values)
    return None


def _rss_peak_bytes(payload: dict[str, Any]) -> int | None:
    values: list[int] = []

    def walk(value: Any, key: str | None = None) -> None:
        if isinstance(value, dict):
            for child_key, child in value.items():
                walk(child, str(child_key))
            return
        if isinstance(value, list):
            for child in value:
                walk(child, key)
            return
        if key is None:
            return
        key_l = key.lower()
        numeric = _as_float(value)
        if numeric is None:
            return
        if key_l in {"rss_peak_bytes", "max_rss_bytes", "rss_before_bytes", "rss_after_bytes"}:
            values.append(int(numeric))
        elif key_l in {"rss_peak_mb", "rss_peak_mib", "max_rss_mb", "max_rss_mib"}:
            values.append(int(float(numeric) * 1024 * 1024))

    walk(payload)
    return max(values) if values else None


def _cache_entries_after(payload: dict[str, Any]) -> int | None:
    candidates = (
        _get_path(payload, ("cache", "growth", "total_entries_after")),
        _get_path(payload, ("cache_growth", "total_entries_after")),
        payload.get("cache_entries_after"),
    )
    values = [_as_int(candidate) for candidate in candidates]
    value = next((item for item in values if item is not None), None)
    if value is not None:
        return value
    runs = payload.get("runs")
    if isinstance(runs, list):
        run_values = [_cache_entries_after(run) for run in runs if isinstance(run, dict)]
        run_values = [item for item in run_values if item is not None]
        if run_values:
            return int(run_values[-1])
    return None


def _cache_entry_growth(payload: dict[str, Any]) -> int | None:
    candidates = (
        _get_path(payload, ("cache", "growth", "total_entries_delta")),
        _get_path(payload, ("cache_growth", "total_entries_delta")),
        payload.get("cache_entry_growth"),
    )
    values = [_as_int(candidate) for candidate in candidates]
    value = next((item for item in values if item is not None), None)
    if value is not None:
        return value
    runs = payload.get("runs")
    if isinstance(runs, list):
        run_values = [_cache_entry_growth(run) for run in runs if isinstance(run, dict)]
        if any(item is not None for item in run_values):
            return sum(int(item or 0) for item in run_values)
    return None


def _replay_scan_cache_metric(payload: dict[str, Any], suffix: str) -> float | int | None:
    diagnostics = payload.get("replay_scan_cache_diagnostics")
    if isinstance(diagnostics, dict):
        values = [
            diagnostics.get(key)
            for key in diagnostics
            if str(key).startswith("replay_") and str(key).endswith(f"_scan_cache_{suffix}")
        ]
        if values:
            if suffix.endswith("_count"):
                return int(sum(int(value or 0) for value in values))
            return float(sum(float(value or 0.0) for value in values))

    samples = payload.get("samples")
    if isinstance(samples, list):
        values = [
            _replay_scan_cache_metric(sample, suffix)
            for sample in samples
            if isinstance(sample, dict)
        ]
        if any(value is not None for value in values):
            if suffix.endswith("_count"):
                return int(sum(int(value or 0) for value in values))
            return float(sum(float(value or 0.0) for value in values))

    runs = payload.get("runs")
    if isinstance(runs, list):
        values = [
            _replay_scan_cache_metric(run, suffix)
            for run in runs
            if isinstance(run, dict)
        ]
        if any(value is not None for value in values):
            if suffix.endswith("_count"):
                return int(sum(int(value or 0) for value in values))
            return float(sum(float(value or 0.0) for value in values))
    return None


def _top_profile(
    profile: dict[str, dict[str, float | int]],
    *,
    limit: int,
) -> list[dict[str, float | int | str]]:
    rows = [
        {
            "name": name,
            "count": int(rec.get("count", 0)),
            "wall_time_s": float(rec.get("wall_time_s", 0.0)),
            "mean_wall_time_s": float(rec.get("mean_wall_time_s", 0.0)),
        }
        for name, rec in profile.items()
        if not str(name).endswith("_count")
    ]
    rows.sort(key=lambda row: float(row["wall_time_s"]), reverse=True)
    return rows[: max(0, int(limit))]


def _metric_float(metrics: dict[str, Any], key: str) -> float | None:
    return _as_float(metrics.get(key))


def _metric_int(metrics: dict[str, Any], key: str) -> int | None:
    value = _as_float(metrics.get(key))
    if value is None:
        return None
    return int(round(value))


def _scan_trial_summary(metrics: dict[str, Any]) -> dict[str, Any] | None:
    """Group trial-scan timing and cache-status metrics for actionability."""

    timing_keys = (
        "trial_solver_scan_total_s",
        "trial_solver_scan_setup_s",
        "trial_solver_scan_initial_compute_forces_s",
        "trial_solver_scan_axis_reset_compute_forces_s",
        "trial_solver_scan_run_setup_s",
        "trial_solver_scan_preflight_s",
        "trial_solver_scan_device_run_s",
        "trial_solver_scan_device_dispatch_s",
        "trial_solver_scan_device_ready_s",
        "trial_solver_scan_host_materialize_s",
        "trial_solver_scan_postprocess_s",
        "trial_solver_scan_unattributed_s",
        "trial_solver_scan_runner_cache_lookup_s",
        "trial_solver_scan_runner_cache_build_s",
    )
    count_keys = (
        "trial_solver_scan_runner_cache_hit_count",
        "trial_solver_scan_runner_cache_miss_count",
        "trial_solver_scan_runner_cache_bypass_count",
    )
    if not any(_metric_float(metrics, key) is not None for key in (*timing_keys, *count_keys)):
        return None

    status: dict[str, dict[str, float | int | None]] = {}
    for name in ("hit", "miss", "bypass"):
        status[name] = {
            "count": _metric_int(metrics, f"trial_solver_scan_runner_cache_{name}_count"),
            "device_run_s": _metric_float(metrics, f"trial_solver_scan_runner_cache_{name}_device_run_s"),
            "dispatch_s": _metric_float(metrics, f"trial_solver_scan_runner_cache_{name}_dispatch_s"),
            "ready_s": _metric_float(metrics, f"trial_solver_scan_runner_cache_{name}_ready_s"),
        }

    count_values = [
        status_name["count"]
        for status_name in status.values()
        if status_name["count"] is not None
    ]
    miss_count = status["miss"]["count"]
    total_count = sum(int(value) for value in count_values) if count_values else None
    dominant_status = max(
        (
            (name, _as_float(values.get("device_run_s")) or 0.0)
            for name, values in status.items()
        ),
        key=lambda item: item[1],
    )
    return {
        "total_s": _metric_float(metrics, "trial_solver_scan_total_s"),
        "setup_s": _metric_float(metrics, "trial_solver_scan_setup_s"),
        "initial_compute_forces_s": _metric_float(
            metrics, "trial_solver_scan_initial_compute_forces_s"
        ),
        "axis_reset_compute_forces_s": _metric_float(
            metrics, "trial_solver_scan_axis_reset_compute_forces_s"
        ),
        "run_setup_s": _metric_float(metrics, "trial_solver_scan_run_setup_s"),
        "preflight_s": _metric_float(metrics, "trial_solver_scan_preflight_s"),
        "device_run_s": _metric_float(metrics, "trial_solver_scan_device_run_s"),
        "device_dispatch_s": _metric_float(metrics, "trial_solver_scan_device_dispatch_s"),
        "device_ready_s": _metric_float(metrics, "trial_solver_scan_device_ready_s"),
        "host_materialize_s": _metric_float(metrics, "trial_solver_scan_host_materialize_s"),
        "postprocess_s": _metric_float(metrics, "trial_solver_scan_postprocess_s"),
        "unattributed_s": _metric_float(metrics, "trial_solver_scan_unattributed_s"),
        "cache_lookup_s": _metric_float(metrics, "trial_solver_scan_runner_cache_lookup_s"),
        "cache_build_s": _metric_float(metrics, "trial_solver_scan_runner_cache_build_s"),
        "cache_status": status,
        "cache_miss_fraction": (
            float(miss_count) / float(total_count)
            if miss_count is not None and total_count not in (None, 0)
            else None
        ),
        "dominant_cache_status_by_device_run": (
            dominant_status[0] if dominant_status[1] > 0.0 else None
        ),
    }


def _projected_replay_summary(
    metrics: dict[str, Any],
    *,
    profile: dict[str, dict[str, float | int]] | None = None,
    total_runtime_s: float | None = None,
) -> dict[str, Any] | None:
    """Group projected replay accounting separately from generic replay time."""

    total_s = _metric_float(metrics, "projected_replay_total_s")
    dispatch_s = _metric_float(metrics, "projected_replay_dispatch_s")
    residual_s = _metric_float(metrics, "projected_residual_tangents_s")
    if total_s is None and dispatch_s is None and residual_s is None:
        return None
    count = None
    if profile is not None:
        count = _profile_named_count(
            profile,
            (
                "jacobian_projected_replay_total",
                "jacobian_fused_projected_replay_total",
            ),
        )
    if count is None:
        count = _metric_int(metrics, "accepted_point_replay_count")
    return {
        "total_s": total_s,
        "dispatch_s": dispatch_s,
        "residual_tangents_s": residual_s,
        "count": count,
        "share_of_total": (
            float(total_s) / float(total_runtime_s)
            if total_s is not None and total_runtime_s is not None and total_runtime_s > 0.0
            else None
        ),
        "residual_tangent_share_of_projected": (
            float(residual_s) / float(total_s)
            if residual_s is not None and total_s is not None and total_s > 0.0
            else None
        ),
    }


SAMPLE_PROFILE_METRICS = {
    "exact_tape_build_s",
    "exact_tape_build_jvp_only_s",
    "exact_tape_build_solve_call_s",
    "exact_tape_build_dynamic_payload_s",
    "initial_tangents_s",
    "initial_tangents_linearize_s",
    "initial_tangents_vmap_dispatch_s",
    "initial_tangents_vmap_ready_s",
    "residual_tangents_s",
    "projected_residual_tangents_s",
    "projected_replay_total_s",
    "projected_replay_dispatch_s",
    "accepted_replay_dispatch_s",
    "accepted_replay_ready_s",
    "trial_solver_scan_total_s",
    "trial_solver_scan_runner_cache_lookup_s",
    "trial_solver_scan_runner_cache_build_s",
    "trial_solver_scan_device_run_s",
    "trial_solver_scan_device_dispatch_s",
    "trial_solver_scan_device_ready_s",
    "trial_solver_scan_runner_cache_hit_device_run_s",
    "trial_solver_scan_runner_cache_hit_dispatch_s",
    "trial_solver_scan_runner_cache_hit_ready_s",
    "trial_solver_scan_runner_cache_miss_device_run_s",
    "trial_solver_scan_runner_cache_miss_dispatch_s",
    "trial_solver_scan_runner_cache_miss_ready_s",
    "trial_solver_scan_runner_cache_bypass_device_run_s",
    "trial_solver_scan_runner_cache_bypass_dispatch_s",
    "trial_solver_scan_runner_cache_bypass_ready_s",
    "trial_solver_scan_runner_cache_miss_count",
    "forward_exact_solver_scan_runner_cache_miss_count",
    "exact_tape_solver_scan_runner_cache_miss_count",
}


def _sample_profile_summaries(payload: dict[str, Any], *, top_profile: int) -> list[dict[str, Any]]:
    """Summarize per-repeat callback profile deltas when a report includes them."""

    samples = payload.get("samples")
    if not isinstance(samples, list):
        return []

    out: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict) or not isinstance(sample.get("profile_delta"), dict):
            continue
        profile = _normalize_profile(sample["profile_delta"])
        wall_time_s = _as_float(sample.get("wall_time_s"))
        metrics = {
            metric: _profile_metric_time(profile, metric, names)
            for metric, names in EXACT_PROFILE_METRIC_NAMES.items()
            if metric in SAMPLE_PROFILE_METRICS
        }
        metrics["replay_time_s"] = _accepted_replay_profile_time(profile)
        metrics["replay_scan_cache_hit_count"] = _replay_scan_cache_metric(sample, "hit_count")
        metrics["replay_scan_cache_miss_count"] = _replay_scan_cache_metric(sample, "miss_count")
        metrics["replay_scan_cache_lookup_s"] = _replay_scan_cache_metric(sample, "lookup_s")
        metrics["replay_scan_cache_build_s"] = _replay_scan_cache_metric(sample, "build_s")
        out.append(
            {
                "index": index,
                "repeat": _as_int(sample.get("repeat")),
                "wall_time_s": wall_time_s,
                "param_step_norm": _as_float(sample.get("param_step_norm")),
                "metrics": metrics,
                "trial_scan_summary": _scan_trial_summary(metrics),
                "projected_replay_summary": _projected_replay_summary(
                    metrics,
                    profile=profile,
                    total_runtime_s=wall_time_s,
                ),
                "exact_optimizer_patch_target": _exact_optimizer_patch_target(
                    profile,
                    total_runtime_s=wall_time_s,
                ),
                "top_profile": _top_profile(profile, limit=top_profile),
            }
        )
    return out


def _exact_optimizer_patch_target(
    profile: dict[str, dict[str, float | int]],
    *,
    total_runtime_s: float | None,
) -> dict[str, Any] | None:
    """Pick the largest actionable exact-optimizer leaf timer.

    The optimizer profile contains enclosing timers such as
    ``exact_solve_with_tape_total`` and ``jacobian_total``.  Those are useful
    for accounting but too broad for selecting the next patch target.
    """

    candidates: list[dict[str, Any]] = []
    for name, rec in profile.items():
        name_s = str(name)
        if name_s in EXACT_OPTIMIZER_CONTAINER_PROFILE_NAMES or (
            name_s.endswith("_total") and name_s != "jacobian_fused_projected_replay_total"
        ):
            continue
        if name_s == "exact_tape_build":
            # ``exact_tape_build`` encloses named tape phases.  If the profiler
            # does not expose ``exact_tape_build_unattributed``, prefer the
            # largest available replay/tangent leaf instead of this broad timer.
            continue
        if name_s == "exact_tape_build_solve_call" and any(
            float(profile[detail].get("wall_time_s", 0.0)) > 0.0
            for detail in EXACT_TAPE_SOLVE_CALL_DETAIL_NAMES
            if detail in profile
        ):
            # Once solve-call internals are present, treat the broad external
            # solve timer as a container and choose a concrete child phase.
            continue
        if name_s in ACCEPTED_REPLAY_PROFILE_NAMES and any(
            float(profile[child].get("wall_time_s", 0.0)) > 0.0
            for child in (f"{name_s}_dispatch", f"{name_s}_ready")
            if child in profile
        ):
            # When dispatch/ready instrumentation is present, the broad replay
            # timer is a parent of those child buckets.
            continue
        if name_s.endswith("_compute_forces") and any(
            float(profile[child].get("wall_time_s", 0.0)) > 0.0
            for child in (f"{name_s}_first", f"{name_s}_rest")
            if child in profile
        ):
            # Prefer cold/warm force-assembly leaves when the optimizer exposed
            # them; the broad compute_forces timer is still useful for accounting.
            continue
        if name_s.endswith("_preconditioner") and any(
            float(profile[child].get("wall_time_s", 0.0)) > 0.0
            for child in (
                name_s.replace("_preconditioner", "_precond_refresh"),
                f"{name_s}_apply",
                f"{name_s}_mode_scale",
            )
            if child in profile
        ):
            # Detailed preconditioner leaves identify refresh/apply/mode-scaling
            # costs more directly than the enclosing preconditioner timer.
            continue
        if name_s == "jacobian_initial_tangents" and any(
            float(profile[detail].get("wall_time_s", 0.0)) > 0.0
            for detail in INITIAL_TANGENT_DETAIL_NAMES
            if detail in profile
        ):
            # Prefer the cold tangent setup subphase over the enclosing tangent
            # build timer once detailed buckets are present.
            continue
        if name_s in SCAN_DEVICE_RUN_PROFILE_NAMES and any(
            float(profile[child].get("wall_time_s", 0.0)) > 0.0
            for child in (name_s.replace("_run", "_dispatch"), name_s.replace("_run", "_ready"))
            if child in profile
        ):
            # The split dispatch/ready buckets identify whether this is a
            # compile/launch issue or actual device-body execution.
            continue
        if name_s in {
            "trial_solver_scan_device_run",
            "trial_solver_scan_device_dispatch",
            "trial_solver_scan_device_ready",
        } and any(
            float(profile[child].get("wall_time_s", 0.0)) > 0.0
            for status in ("hit", "miss", "bypass")
            for child in (
                f"trial_solver_scan_runner_cache_{status}_device_run",
                f"trial_solver_scan_runner_cache_{status}_dispatch",
                f"trial_solver_scan_runner_cache_{status}_ready",
            )
            if child in profile
        ):
            # When scan timing is split by cache status, those buckets explain
            # whether misses, hits, or bypasses are driving the broad trial scan.
            continue
        if name_s.endswith("_device_run") and "_scan_runner_cache_" in name_s and any(
            float(profile[child].get("wall_time_s", 0.0)) > 0.0
            for child in (name_s.replace("_device_run", "_dispatch"), name_s.replace("_device_run", "_ready"))
            if child in profile
        ):
            # Prefer cache-status dispatch/ready leaves over their enclosing
            # cache-status device-run bucket when split timing is available.
            continue
        if name_s not in EXACT_OPTIMIZER_PATCH_TARGET_NAMES:
            continue
        wall = float(rec.get("wall_time_s", 0.0))
        if wall <= 0.0:
            continue
        candidates.append(
            {
                "name": name_s,
                "count": int(rec.get("count", 0)),
                "wall_time_s": wall,
                "mean_wall_time_s": float(rec.get("mean_wall_time_s", 0.0)),
            }
        )

    if not candidates:
        return None

    target = max(candidates, key=lambda row: float(row["wall_time_s"]))
    total = _as_float(total_runtime_s)
    target["share_of_total"] = (
        float(target["wall_time_s"]) / total if total is not None and total > 0.0 else None
    )
    target["note"] = "largest non-container exact optimizer timer; use as next patch target"
    return target


def _bottleneck_hint(metrics: dict[str, Any]) -> dict[str, Any] | None:
    total = _as_float(metrics.get("total_runtime_s"))
    candidates: list[tuple[str, str, float]] = []
    for key, label in BOTTLENECK_METRICS:
        value = _as_float(metrics.get(key))
        if value is not None and value > 0.0:
            candidates.append((key, label, value))

    if not candidates:
        rss_mib = _as_float(metrics.get("rss_peak_mib"))
        if rss_mib is not None:
            return {
                "metric": "rss_peak_mib",
                "label": "RSS peak",
                "value": rss_mib,
                "share_of_total": None,
                "note": "no phase timings were present; inspect memory pressure and profiler source output",
            }
        return None

    key, label, value = max(candidates, key=lambda item: item[2])
    share = value / total if total is not None and total > 0.0 else None
    return {
        "metric": key,
        "label": label,
        "value": value,
        "share_of_total": share,
        "note": "largest normalized phase currently exposed by this report",
    }


def _profile_has_jvp_only_exact_tape(profile: dict[str, Any]) -> bool:
    for name in ("exact_solve_with_tape_jvp_only_total", "exact_tape_build_jvp_only"):
        rec = profile.get(name)
        if not isinstance(rec, dict):
            continue
        if int(rec.get("count", 0) or 0) > 0 or float(rec.get("wall_time_s", 0.0) or 0.0) > 0.0:
            return True
    return False


def _effective_jvp_only_exact_tape_metadata(
    payload: dict[str, Any],
    *,
    profile: dict[str, Any],
    runtime: dict[str, Any],
) -> Any:
    if "jvp_only_exact_tape_effective" in payload:
        return payload.get("jvp_only_exact_tape_effective")
    if _profile_has_jvp_only_exact_tape(profile):
        return True
    if "jvp_only_exact_tape" in payload:
        return payload.get("jvp_only_exact_tape")
    return runtime.get("vmec_jax_opt_jvp_only_exact_tape")


def _effective_jvp_only_basepoint_carries_metadata(
    payload: dict[str, Any],
    *,
    profile: dict[str, Any],
    runtime: dict[str, Any],
) -> Any:
    if "jvp_only_basepoint_carries_effective" in payload:
        return payload.get("jvp_only_basepoint_carries_effective")
    if payload.get("jvp_only_basepoint_carries") is True:
        return True
    env_value = runtime.get("vmec_jax_jvp_only_exact_tape_basepoint_carries")
    if env_value not in (None, ""):
        return env_value
    backend = str(runtime.get("default_backend") or payload.get("jax_default_backend") or "").strip().lower()
    if _profile_has_jvp_only_exact_tape(profile) and backend in ("gpu", "cuda", "rocm"):
        return True
    if "jvp_only_basepoint_carries" in payload:
        return payload.get("jvp_only_basepoint_carries")
    return env_value


def summarize_payload(
    payload: dict[str, Any],
    *,
    path: Path | None = None,
    label: str | None = None,
    top_profile: int = 5,
) -> dict[str, Any]:
    profile = _profile_from_payload(payload)
    compile_time = _direct_time(payload, "compile_time_s")
    replay_time = _direct_time(payload, "replay_time_s")
    cache_time = _direct_time(payload, "cache_time_s")
    if compile_time is None:
        compile_time = _profile_time(profile, "compile_time_s")
    if replay_time is None:
        replay_time = _accepted_replay_profile_time(profile)
    if replay_time is None:
        replay_time = _profile_time(profile, "replay_time_s")
    if cache_time is None:
        cache_time = _profile_time(profile, "cache_time_s")

    rss_peak = _rss_peak_bytes(payload)
    metrics = {
        "total_runtime_s": _total_runtime(payload),
        "vmec_solve_s": _wall_time_metric(payload, "vmec_solve"),
        "vmec_compute_forces_s": _vmec_timing_metric(payload, "compute_forces_s"),
        "vmec_preconditioner_s": _vmec_timing_metric(payload, "preconditioner_s"),
        "vmec_precond_refresh_s": _vmec_timing_metric(payload, "precond_refresh_s"),
        "vmec_precond_apply_s": _vmec_timing_metric(payload, "precond_apply_s"),
        "vmec_precond_mode_scale_s": _vmec_timing_metric(payload, "precond_mode_scale_s"),
        "vmec_update_s": _vmec_timing_metric(payload, "update_s"),
        "vmec_update_state_s": _vmec_timing_metric(payload, "update_state_s"),
        "qi_first_call_s": _first_present(
            _wall_time_metric(payload, "qi_first_call"),
            _wall_time_metric(payload, "qi_first"),
        ),
        "qi_warm_min_s": _wall_time_metric(payload, "qi_warm_min"),
        "qi_warm_mean_s": _wall_time_metric(payload, "qi_warm_mean"),
        **{
            metric: _profile_metric_time(profile, metric, names)
            for metric, names in EXACT_PROFILE_METRIC_NAMES.items()
        },
        "compile_time_s": compile_time,
        "replay_time_s": replay_time,
        "cache_time_s": cache_time,
        "contamination_warning_count": _contamination_warning_count(payload),
        "callback_count": _callback_count(payload),
        "rss_peak_bytes": rss_peak,
        "rss_peak_mib": None if rss_peak is None else rss_peak / (1024.0 * 1024.0),
        "solve_count": _solve_count(payload, profile),
        "accepted_point_replay_count": _accepted_replay_count(payload, profile),
        "cache_entries_after": _cache_entries_after(payload),
        "cache_entry_growth": _cache_entry_growth(payload),
        "replay_scan_cache_hit_count": _replay_scan_cache_metric(payload, "hit_count"),
        "replay_scan_cache_miss_count": _replay_scan_cache_metric(payload, "miss_count"),
        "replay_scan_cache_lookup_s": _replay_scan_cache_metric(payload, "lookup_s"),
        "replay_scan_cache_build_s": _replay_scan_cache_metric(payload, "build_s"),
    }

    runtime = payload.get("runtime") if isinstance(payload.get("runtime"), dict) else {}
    metadata = {
        "source_report_kind": _source_kind(payload),
        "problem": payload.get("problem"),
        "max_mode": payload.get("max_mode"),
        "callback": payload.get("callback"),
        "method": payload.get("method"),
        "solver_device": (
            payload.get("solver_device_resolved")
            or payload.get("solver_device")
            or _get_path(payload, ("args", "solver_device"))
        ),
        "jax_default_backend": payload.get("jax_default_backend") or runtime.get("default_backend"),
        "jax_version": payload.get("jax_version") or runtime.get("jax_version"),
        "active_gpu": payload.get("active_gpu") if "active_gpu" in payload else runtime.get("active_gpu"),
        "jvp_only_exact_tape": _effective_jvp_only_exact_tape_metadata(
            payload,
            profile=profile,
            runtime=runtime,
        ),
        "jvp_only_basepoint_carries": _effective_jvp_only_basepoint_carries_metadata(
            payload,
            profile=profile,
            runtime=runtime,
        ),
        "jit_booz": _get_path(payload, ("qi_resolution", "jit_booz")),
        "contamination_warnings": payload.get("contamination_warnings"),
        "run_repeats": payload.get("run_repeats"),
    }
    return {
        "label": label,
        "path": None if path is None else str(path),
        "metadata": metadata,
        "metrics": metrics,
        "trial_scan_summary": _scan_trial_summary(metrics),
        "projected_replay_summary": _projected_replay_summary(
            metrics,
            profile=profile,
            total_runtime_s=_as_float(metrics.get("total_runtime_s")),
        ),
        "bottleneck_hint": _bottleneck_hint(metrics),
        "exact_optimizer_patch_target": _exact_optimizer_patch_target(
            profile,
            total_runtime_s=_as_float(metrics.get("total_runtime_s")),
        ),
        "sample_profile_summaries": _sample_profile_summaries(payload, top_profile=top_profile),
        "top_profile": _top_profile(profile, limit=top_profile),
    }


def _ratio(value: Any, baseline: Any) -> float | None:
    value_f = _as_float(value)
    baseline_f = _as_float(baseline)
    if value_f is None or baseline_f is None or baseline_f == 0.0:
        return None
    return value_f / baseline_f


def build_comparison(
    summaries: list[dict[str, Any]],
    *,
    baseline: str = "0",
) -> dict[str, Any]:
    if len(summaries) < 2:
        raise ValueError("at least two reports are required for comparison")
    baseline_index = _resolve_baseline(summaries, baseline)
    base = summaries[baseline_index]
    base_metrics = base["metrics"]
    comparisons: list[dict[str, Any]] = []
    for index, summary in enumerate(summaries):
        if index == baseline_index:
            continue
        metrics = summary["metrics"]
        ratios = {
            key: _ratio(metrics.get(key), base_metrics.get(key))
            for key in METRIC_ORDER
            if key in metrics and key in base_metrics
        }
        deltas = {
            key: (
                None
                if _as_float(metrics.get(key)) is None or _as_float(base_metrics.get(key)) is None
                else float(metrics[key]) - float(base_metrics[key])
            )
            for key in METRIC_ORDER
            if key in metrics and key in base_metrics
        }
        comparisons.append(
            {
                "label": summary["label"],
                "baseline_label": base["label"],
                "ratios": ratios,
                "deltas": deltas,
            }
        )
    return {
        "schema_version": 1,
        "report_kind": "profile_report_comparison",
        "baseline_label": base["label"],
        "reports": summaries,
        "comparisons": comparisons,
    }


def _resolve_baseline(summaries: list[dict[str, Any]], baseline: str) -> int:
    try:
        index = int(str(baseline))
    except ValueError:
        index = -1
    if 0 <= index < len(summaries):
        return index
    for idx, summary in enumerate(summaries):
        if str(summary.get("label")) == str(baseline):
            return idx
    labels = ", ".join(str(summary.get("label")) for summary in summaries)
    raise ValueError(f"baseline {baseline!r} does not match index or label; available labels: {labels}")


def _format_value(value: Any, metric: str) -> str:
    if value is None:
        return "n/a"
    value_f = _as_float(value)
    if value_f is None:
        return str(value)
    if metric.endswith("_count") or metric in {
        "callback_count",
        "solve_count",
        "cache_entry_growth",
        "cache_entries_after",
    }:
        return str(int(round(value_f)))
    if metric.endswith("_mib"):
        return f"{value_f:.1f}"
    if metric.endswith("_s"):
        return f"{value_f:.3f}"
    return f"{value_f:.3f}"


def _format_ratio(value: Any) -> str:
    value_f = _as_float(value)
    if value_f is None:
        return "n/a"
    return f"{value_f:.3f}x"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [
        max(len(headers[col]), *(len(row[col]) for row in rows)) if rows else len(headers[col])
        for col in range(len(headers))
    ]
    lines = ["  ".join(headers[col].ljust(widths[col]) for col in range(len(headers)))]
    lines.append("  ".join("-" * widths[col] for col in range(len(headers))))
    for row in rows:
        lines.append("  ".join(row[col].ljust(widths[col]) for col in range(len(headers))))
    return "\n".join(lines)


def format_text(comparison: dict[str, Any]) -> str:
    reports = list(comparison["reports"])
    baseline_label = str(comparison["baseline_label"])
    lines = [
        "Profile report comparison",
        f"Baseline: {baseline_label}",
        "",
        "Reports:",
    ]
    report_rows = []
    for report in reports:
        metadata = report["metadata"]
        metrics = report["metrics"]
        report_rows.append(
            [
                str(report["label"]),
                str(metadata.get("source_report_kind") or "unknown"),
                str(metadata.get("solver_device") or metadata.get("jax_default_backend") or "unknown"),
                str(metadata.get("problem") or ""),
                str(metadata.get("callback") or metadata.get("method") or ""),
                _format_value(metrics.get("total_runtime_s"), "total_runtime_s"),
                _format_value(metrics.get("callback_count"), "callback_count"),
                _format_value(metrics.get("rss_peak_mib"), "rss_peak_mib"),
                _format_value(metrics.get("solve_count"), "solve_count"),
                _format_value(metrics.get("accepted_point_replay_count"), "accepted_point_replay_count"),
            ]
        )
    lines.append(
        _table(
            [
                "label",
                "kind",
                "device",
                "problem",
                "mode",
                "total_s",
                "callbacks",
                "rss_mib",
                "solves",
                "accepted_replays",
            ],
            report_rows,
        )
    )

    lines.extend(["", "Ratios vs baseline:"])
    ratio_headers = ["metric"] + [str(item["label"]) for item in comparison["comparisons"]]
    ratio_rows = []
    for metric in METRIC_ORDER:
        row = [METRIC_LABELS.get(metric, metric)]
        for item in comparison["comparisons"]:
            row.append(_format_ratio(item["ratios"].get(metric)))
        ratio_rows.append(row)
    lines.append(_table(ratio_headers, ratio_rows))

    lines.extend(["", "Top profile terms:"])
    for report in reports:
        entries = report.get("top_profile") or []
        if not entries:
            lines.append(f"  {report['label']}: n/a")
            continue
        total = _as_float(report["metrics"].get("total_runtime_s")) or 0.0
        formatted = []
        for entry in entries:
            wall = float(entry["wall_time_s"])
            share = "" if total <= 0.0 else f", {100.0 * wall / total:.1f}%"
            formatted.append(f"{entry['name']}={wall:.3f}s{share}")
        lines.append(f"  {report['label']}: " + "; ".join(formatted))

    projected_rows = []
    for report in reports:
        projected = report.get("projected_replay_summary")
        if not isinstance(projected, dict):
            continue
        projected_rows.append(
            [
                str(report["label"]),
                _format_value(projected.get("total_s"), "projected_replay_total_s"),
                _format_value(projected.get("dispatch_s"), "projected_replay_dispatch_s"),
                _format_value(projected.get("residual_tangents_s"), "projected_residual_tangents_s"),
                _format_value(projected.get("count"), "accepted_point_replay_count"),
                (
                    "n/a"
                    if _as_float(projected.get("share_of_total")) is None
                    else f"{100.0 * float(projected['share_of_total']):.1f}%"
                ),
            ]
        )
    if projected_rows:
        lines.extend(["", "Projected replay totals:"])
        lines.append(
            _table(
                ["label", "total_s", "dispatch_s", "residual_tangent_s", "count", "share"],
                projected_rows,
            )
        )

    scan_rows = []
    for report in reports:
        scan = report.get("trial_scan_summary")
        if not isinstance(scan, dict):
            continue
        cache_status = scan.get("cache_status") if isinstance(scan.get("cache_status"), dict) else {}
        hit = cache_status.get("hit") if isinstance(cache_status.get("hit"), dict) else {}
        miss = cache_status.get("miss") if isinstance(cache_status.get("miss"), dict) else {}
        bypass = cache_status.get("bypass") if isinstance(cache_status.get("bypass"), dict) else {}
        miss_fraction = _as_float(scan.get("cache_miss_fraction"))
        scan_rows.append(
            [
                str(report["label"]),
                _format_value(scan.get("total_s"), "trial_solver_scan_total_s"),
                _format_value(scan.get("device_dispatch_s"), "trial_solver_scan_device_dispatch_s"),
                _format_value(scan.get("device_ready_s"), "trial_solver_scan_device_ready_s"),
                _format_value(scan.get("cache_lookup_s"), "trial_solver_scan_runner_cache_lookup_s"),
                _format_value(scan.get("cache_build_s"), "trial_solver_scan_runner_cache_build_s"),
                _format_value(hit.get("count"), "trial_solver_scan_runner_cache_hit_count"),
                _format_value(miss.get("count"), "trial_solver_scan_runner_cache_miss_count"),
                _format_value(bypass.get("count"), "trial_solver_scan_runner_cache_bypass_count"),
                "n/a" if miss_fraction is None else f"{100.0 * miss_fraction:.1f}%",
                _format_value(
                    miss.get("device_run_s"),
                    "trial_solver_scan_runner_cache_miss_device_run_s",
                ),
                str(scan.get("dominant_cache_status_by_device_run") or "n/a"),
            ]
        )
    if scan_rows:
        lines.extend(["", "Trial scan timing/cache status:"])
        lines.append(
            _table(
                [
                    "label",
                    "total_s",
                    "dispatch_s",
                    "ready_s",
                    "lookup_s",
                    "build_s",
                    "hits",
                    "misses",
                    "bypasses",
                    "miss_frac",
                    "miss_device_s",
                    "dominant",
                ],
                scan_rows,
            )
        )
    hints = []
    for report in reports:
        hint = report.get("bottleneck_hint")
        if not isinstance(hint, dict):
            continue
        value = _as_float(hint.get("value"))
        share = _as_float(hint.get("share_of_total"))
        value_text = "n/a" if value is None else f"{value:.3f}"
        share_text = "" if share is None else f", {100.0 * share:.1f}% of total"
        hints.append(f"  {report['label']}: {hint.get('label')} ({value_text}{share_text})")
    if hints:
        lines.extend(["", "Bottleneck hints:"])
        lines.extend(hints)
    patch_targets = []
    for report in reports:
        target = report.get("exact_optimizer_patch_target")
        if not isinstance(target, dict):
            continue
        wall = _as_float(target.get("wall_time_s"))
        share = _as_float(target.get("share_of_total"))
        count = _as_int(target.get("count"))
        mean = _as_float(target.get("mean_wall_time_s"))
        wall_text = "n/a" if wall is None else f"{wall:.3f}s"
        share_text = "" if share is None else f", {100.0 * share:.1f}% of total"
        count_text = "" if count is None else f", count={count}"
        mean_text = "" if mean is None else f", mean={mean:.3f}s"
        patch_targets.append(
            f"  {report['label']}: {target.get('name')} ({wall_text}{share_text}{count_text}{mean_text})"
        )
    if patch_targets:
        lines.extend(["", "Exact optimizer patch targets:"])
        lines.extend(patch_targets)
    sample_targets = []
    for report in reports:
        samples = report.get("sample_profile_summaries")
        if not isinstance(samples, list) or not samples:
            continue
        cold = samples[0]
        target = cold.get("exact_optimizer_patch_target")
        if not isinstance(target, dict):
            continue
        wall = _as_float(target.get("wall_time_s"))
        share = _as_float(target.get("share_of_total"))
        repeat = _as_int(cold.get("repeat"))
        repeat_text = str(cold.get("index") if repeat is None else repeat)
        wall_text = "n/a" if wall is None else f"{wall:.3f}s"
        share_text = "" if share is None else f", {100.0 * share:.1f}% of repeat"
        sample_targets.append(
            f"  {report['label']} repeat {repeat_text}: {target.get('name')} ({wall_text}{share_text})"
        )
    if sample_targets:
        lines.extend(["", "Cold callback patch targets:"])
        lines.extend(sample_targets)
    return "\n".join(lines)


def _default_labels(paths: list[Path], labels: list[str] | None) -> list[str]:
    if labels is not None:
        if len(labels) != len(paths):
            raise ValueError("--label must be provided once per report when used")
        if len(set(labels)) != len(labels):
            raise ValueError("--label values must be unique")
        return labels

    counts: dict[str, int] = {}
    out: list[str] = []
    for path in paths:
        base = path.stem
        count = counts.get(base, 0)
        counts[base] = count + 1
        out.append(base if count == 0 else f"{base}_{count + 1}")
    return out


def load_summary(path: Path, *, label: str, top_profile: int = 5) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return summarize_payload(payload, path=path.resolve(), label=label, top_profile=top_profile)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        labels = _default_labels(args.reports, args.label)
        summaries = [
            load_summary(path.expanduser().resolve(), label=label, top_profile=int(args.top_profile))
            for path, label in zip(args.reports, labels, strict=True)
        ]
        comparison = build_comparison(summaries, baseline=str(args.baseline))
    except Exception as exc:
        raise SystemExit(f"compare_profile_reports: {exc}") from exc

    json_text = json.dumps(comparison, indent=2, sort_keys=True)
    if args.json_out is not None:
        json_path = args.json_out.expanduser().resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json_text + "\n", encoding="utf-8")

    if args.format == "json":
        print(json_text)
    else:
        print(format_text(comparison))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
