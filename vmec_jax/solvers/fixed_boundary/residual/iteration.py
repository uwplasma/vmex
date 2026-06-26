"""Fixed-boundary solvers.

The first solver milestone is a robust "inner solve" for the VMEC ``lambda`` field
with R/Z held fixed. This is useful for:

- validating the magnetic energy objective against VMEC2000 `wout` files,
- building toward a full fixed-boundary equilibrium solve.

Notes
-----
This module intentionally avoids optional dependencies (e.g. jaxopt). The current
implementation uses gradient descent with a simple backtracking line search.
"""

from __future__ import annotations

from collections import OrderedDict
from functools import partial
import time
import os
from pathlib import Path
from typing import Any

import numpy as np

from vmec_jax._compat import has_jax, jax, jnp, jit
from vmec_jax import _solve_runtime
from vmec_jax.solvers.fixed_boundary.residual import config as _residual_iter_config
from vmec_jax.solvers.fixed_boundary.residual import policy as _residual_iter_policy
from vmec_jax.solvers.fixed_boundary.residual.config import (
    HEAVY_DUMP_ENVS as _HEAVY_DUMP_ENVS,
    LIGHT_DUMP_ENVS as _LIGHT_DUMP_ENVS,
    resolve_axis_reset_config as _resolve_axis_reset_config,
    resolve_debug_print_config as _resolve_debug_print_config,
    resolve_host_profile_setup as _resolve_host_profile_setup,
    indata_has_profile_setup_work as _indata_has_profile_setup_work,
    resolve_nstep_screen as _resolve_nstep_screen,
    resolve_setup_host_enforce as _resolve_setup_host_enforce,
)
from vmec_jax.solvers.fixed_boundary.residual.policy import (
    host_restart_decision as _host_restart_decision,
    new_residual_iter_histories as _new_residual_iter_histories,
    numpy_preconditioner_apply_policy as _numpy_preconditioner_apply_policy,
    pop_residual_iter_rollback_histories as _pop_residual_iter_rollback_histories,
    resolve_residual_iter_startup_policy as _resolve_residual_iter_startup_policy,
    scan_fallback_decision as _scan_fallback_decision,
    scan_fallback_message as _scan_fallback_message,
    vmec2000_time_control_decision as _vmec2000_time_control_decision,
)
from vmec_jax.solvers.fixed_boundary.residual.runtime import (
    _attach_free_boundary_external_field_diag as _runtime_attach_free_boundary_external_field_diag,
    _converged_residuals_scan_fast as _runtime_converged_residuals_scan_fast,
    _device_get_floats,
    _freeb_trial_bsqvac_half as _runtime_freeb_trial_bsqvac_half,
    _initial_setup_phase_timings,
    initial_free_boundary_loop_state as _runtime_initial_free_boundary_loop_state,
    _maybe_dump_ptau as _runtime_maybe_dump_ptau,
    _maybe_print_nonscan_state_debug,
    _new_residual_iter_timing_stats as _runtime_new_residual_iter_timing_stats,
    _record_compute_force_timing as _runtime_record_compute_force_timing,
    _record_setup_timing as _runtime_record_setup_timing,
    _setup_timer_start as _runtime_setup_timer_start,
    _vmec_freeb_plascur_from_bcovar as _runtime_vmec_freeb_plascur_from_bcovar,
    dump_xc_with_velocity_blocks as _dump_xc_with_velocity_blocks,
    edge_bsqvac_from_nestor as _edge_bsqvac_from_nestor,
    record_elapsed_timing as _record_elapsed_timing,
    record_update_state_ready_timing as _record_update_state_ready_timing,
    record_update_total_timing as _record_update_total_timing,
    resume_free_boundary_loop_state as _runtime_resume_free_boundary_loop_state,
    resolve_free_boundary_coupling_runtime as _runtime_resolve_free_boundary_coupling_runtime,
    resolve_free_boundary_iteration_controls as _runtime_resolve_free_boundary_iteration_controls,
    resolve_residual_profile_window as _resolve_residual_profile_window,
    trial_residual_total_runtime as _runtime_trial_residual_total,
)
from vmec_jax.solvers.fixed_boundary.residual.accelerated_scan import (
    run_accelerated_residual_scan as _run_accelerated_residual_scan,
)
from vmec_jax.solvers.fixed_boundary.residual.setup import (
    build_residual_profile_setup as _build_residual_profile_setup,
    build_residual_ptau_bindings as _build_residual_ptau_bindings,
    build_residual_cache_keys as _build_residual_cache_keys,
    build_residual_static_grid_setup as _build_residual_static_grid_setup,
    free_boundary_pressure_edge_scale as _free_boundary_pressure_edge_scale,
    resolve_free_boundary_setup_policy as _resolve_free_boundary_setup_policy,
)
from vmec_jax.solvers.fixed_boundary.residual.state_setup import (
    build_residual_state_setup as _build_residual_state_setup,
)
from vmec_jax.solvers.fixed_boundary.residual.finalize import (
    finalize_residual_iter_from_namespace as _finalize_residual_iter_from_namespace,
    precompile_only_residual_iter_result as _precompile_only_residual_iter_result,
)
from vmec_jax.solvers.fixed_boundary.residual.force_cache import (
    compute_forces_jit_cache_key as _compute_forces_jit_cache_key,
    maybe_precompile_residual_force_kernels as _maybe_precompile_residual_force_kernels,
    prepare_numpy_force_fast_path as _prepare_numpy_force_fast_path,
    select_compute_forces_callable as _select_compute_forces_callable,
)
from vmec_jax.solvers.fixed_boundary.residual.force_payload import (
    build_strict_update_adjoint_trace_entry as _build_strict_update_adjoint_trace_entry,
    compute_forces_iter_runtime as _compute_forces_iter_runtime,
    compute_forces_without_iter_dump as _compute_forces_without_iter_dump,
    evaluate_residual_force_from_state as _evaluate_residual_force_from_state,  # noqa: F401 - compatibility alias for tests/internal users.
    finalize_strict_update_adjoint_trace_entry as _finalize_strict_update_adjoint_trace_entry,
    make_residual_force_evaluator as _make_residual_force_evaluator,
    residual_iter_dump_index as _residual_iter_dump_index,
    tomnsps_to_numpy_host as _tomnsps_to_numpy_host,
)
from vmec_jax.solvers.fixed_boundary.residual.update import (
    ResidualControllerState as _ResidualControllerState,
    apply_controller_state_update as _apply_controller_state_update,
    backtracking_momentum_search as _backtracking_momentum_search,
    candidate_state_from_deltas as _candidate_state_from_deltas_helper,
    candidate_state_from_delta_tuple as _candidate_state_from_delta_tuple_helper,
    controller_state_after_catastrophic_restart_update as _controller_state_after_catastrophic_restart_update,
    controller_state_after_free_boundary_turnon_restart_update as _controller_state_after_free_boundary_turnon_restart_update,
    controller_state_after_host_restart_decision_sample as _controller_state_after_host_restart_decision_sample,
    controller_state_after_initial_axis_setup_result as _controller_state_after_initial_axis_setup_result,
    controller_state_after_initial_axis_reset_update as _controller_after_axis_reset,
    controller_state_after_pre_restart_update as _controller_state_after_pre_restart_update,
    controller_state_after_vmec2000_time_control_sample as _controller_state_after_vmec2000_time_control_sample,
    controller_state_after_vmec2000_time_control_restart_update as _controller_state_after_vmec2000_time_control_restart_update,
    controller_state_from_runtime_scalars as _controller_state_from_runtime_scalars,
    controller_state_from_resume_state as _controller_state_from_resume_state,
    controller_state_legacy_values as _controller_state_legacy_values,
    delta_tuple_from_blocks as _delta_tuple_from_blocks_helper,
    direct_force_fallback_acceptance_decision as _direct_force_fallback_acceptance_decision,
    direct_force_fallback_trial as _direct_force_fallback_trial,
    host_catastrophic_restart_update as _host_catastrophic_restart_update,
    host_free_boundary_turnon_restart_update as _host_free_boundary_turnon_restart_update,
    host_initial_axis_reset_update as _host_axis_reset_update,
    host_pre_restart_trigger_branch_result as _host_pre_restart_trigger_branch_result,
    host_pre_restart_trigger_update as _host_pre_restart_trigger_update,
    host_vmec2000_time_control_restart_branch_result as _host_vmec2000_time_control_restart_branch_result,
    host_vmec2000_time_control_restart_update as _host_vmec2000_time_control_restart_update,
    initial_residual_controller_state as _initial_residual_controller_state,
    initial_residual_velocity_state as _initial_residual_velocity_state,
    jit_strict_momentum_update_proposal as _jit_strict_momentum_update_proposal,
    residual_evolve_coefficients as _residual_evolve_coefficients,
    strict_momentum_update_proposal as _strict_momentum_update_proposal,
    strict_step_branch_application as _strict_step_branch_application,
    strict_step_branch_result as _strict_step_branch_result,
    strict_step_branch_result_after_catastrophic_restart as _strict_step_branch_result_after_catastrophic_restart,
    strict_step_branch_result_after_direct_fallback as _strict_step_branch_result_after_direct_fallback,
    strict_step_acceptance_decision as _strict_step_acceptance_decision,
    strict_trial_evaluation as _strict_trial_evaluation,
    velocity_blocks_from_force_blocks as _velocity_blocks_from_force_blocks,
    velocity_blocks_from_resume_state as _velocity_blocks_from_resume_state,
    zero_all_velocity_blocks_like as _zero_all_velocity_blocks_like,
    zero_primary_velocity_blocks_like as _zero_primary_velocity_blocks_like,
    zero_velocity_blocks_like as _zero_velocity_blocks_like,
)
from vmec_jax.field import TWOPI
from vmec_jax.solvers.fixed_boundary.jit_cache import (
    jit_cache_get as _jit_cache_get,
    jit_cache_put as _jit_cache_put,
    record_scan_runner_cache_miss_categories as _record_scan_runner_cache_miss_categories,
)
from vmec_jax.solvers.fixed_boundary.diagnostics import hlo as _hlo_dump_helpers
from vmec_jax.solvers.fixed_boundary.optimization import residual_context as _residual_force_context_helpers
from vmec_jax.solvers.fixed_boundary.diagnostics.io import (
    _dump_freeb_axis_trace_record,
    _dump_freeb_control_trace_record,
    _dump_time_control_trace_record,
    _finite_float_or_zero,
    _format_checkpoint_log_row as _format_checkpoint_log_row,
    _format_evolve_trace_row as _format_evolve_trace_row,
    _format_freeb_control_trace_row as _format_freeb_control_trace_row,
    _format_time_control_log_row as _format_time_control_log_row,
    _legacy_dump_record_path as _legacy_dump_record_path,
    _legacy_single_dump_iter_selected as _legacy_single_dump_iter_selected,
    _materialize_adjoint_trace_array,
    _maybe_dump_checkpoint_record,
    _maybe_dump_evolve_trace_record,
    _maybe_dump_jacobian_terms_record,
    _maybe_dump_time_control_record,
    _normalize_adjoint_trace_mode,
    _normalize_resume_state_mode,
    _should_print_vmec2000_row,
    _vmec2000_cadence_selected,
)
from vmec_jax.solvers.fixed_boundary.options import (
    validate_residual_iteration_options,
)
from vmec_jax.solvers.fixed_boundary.profiles import (
    build_wout_like_profiles_from_indata as _build_wout_like_profiles_from_indata,
)
from vmec_jax.solvers.fixed_boundary.residual.geometry import (
    _m1_internal_to_physical_pair as _geometry_m1_internal_to_physical_pair,
    _mn_sin_to_signed_physical_batch as _geometry_mn_sin_to_signed_physical_batch,
    _rz_norm_np as _geometry_rz_norm_np,
)
from vmec_jax.solvers.fixed_boundary.residual.mode_transform import (
    build_mode_transform_context as _build_mode_transform_context,
)
from vmec_jax.solvers.fixed_boundary.residual.payload_blocks import (
    radial_preconditioner_output_blocks_jax as _radial_preconditioner_output_blocks_jax,
)
from vmec_jax.solvers.fixed_boundary.residual.force_norms import (
    force_blocks_from_update_order as _force_blocks_from_update_order,
    lambda_preconditioned_full_norm as _lambda_preconditioned_full_norm,
    mode_weight_force_blocks_jax as _mode_weight_force_blocks_jax,
    mode_weight_force_blocks_np as _mode_weight_force_blocks_np,
    residual_fsq_from_norms as _residual_fsq_from_norms,
    safe_dt_from_force_blocks as _safe_dt_from_force_blocks,
)
from vmec_jax.solvers.fixed_boundary.residual.host_diagnostics import (
    PreRestartTriggerCallbacks as _PreRestartTriggerCallbacks,
    Vmec2000TimeControlCallbacks as _Vmec2000TimeControlCallbacks,
    dump_residual_evolve_trace as _dump_residual_evolve_trace,
    print_compact_converged_status as _print_compact_converged_status,
    print_compact_physical_residual_status as _print_compact_physical_residual_status,
    print_compact_residual_iteration_update_status as _print_compact_residual_iteration_update_status,
    print_residual_iteration_update_status as _print_residual_iteration_update_status,
    residual_update_rms_for_print as _residual_update_rms_for_print,
    resolve_vmec2000_print_context as _resolve_vmec2000_print_context,
    run_pre_restart_trigger_runtime as _run_pre_restart_trigger_runtime,
    run_vmec2000_time_control_runtime as _run_vmec2000_time_control_runtime,
    sample_vmec_iteration_scalars as _sample_vmec_iteration_scalars,
)
from vmec_jax.solvers.fixed_boundary.residual.iteration_control import (
    constraint_preconditioner_channels as _constraint_preconditioner_channels,
    resolve_residual_iteration_control_sample as _resolve_residual_iteration_control_sample,
)
from vmec_jax.solvers.fixed_boundary.residual.iteration_metrics import (
    physical_residual_metric_channels as _physical_residual_metric_channels,
    select_residual_norms_for_iteration as _select_residual_norms_for_iteration,
)
from vmec_jax.solvers.fixed_boundary.residual.iteration_preconditioner import (
    apply_residual_iteration_preconditioner as _apply_residual_iteration_preconditioner,
    resolve_preconditioned_residual_scalars as _resolve_preconditioned_residual_scalars,
)
from vmec_jax.solvers.fixed_boundary.residual.scan_adapters import (
    ResidualScanPathHooks,
    dispatch_residual_scan_path,
)
from vmec_jax.solvers.fixed_boundary.residual import preconditioner_payload as _precond_payload_facade
from vmec_jax.solvers.fixed_boundary.residual.ptau import (
    accepted_control_ptau_arrays as _accepted_control_ptau_arrays_helper,
    accepted_control_ptau_host_from_payload as _accepted_control_ptau_host_from_payload,
    maybe_dump_jacobian_terms as _maybe_dump_jacobian_terms_helper,
    maybe_dump_ptau as _maybe_dump_ptau_helper,
    ptau_minmax as _ptau_minmax_helper,
    ptau_minmax_from_k_host as _ptau_minmax_from_k_host_helper,
    resolve_bad_jacobian_tau_selection as _resolve_bad_jacobian_tau_selection,
    state_tau_minmax_from_vmec_state as _state_tau_minmax_from_vmec_state,
)
from vmec_jax.solvers.fixed_boundary.optimization.constraints import (
    apply_vmec_lambda_axis_rules_to_state as _apply_vmec_lambda_axis_rules_to_state,
    enforce_fixed_boundary_and_axis as _enforce_fixed_boundary_and_axis,
    enforce_fixed_boundary_and_axis_np as _enforce_fixed_boundary_and_axis_np,
    mode00_index as _mode00_index,  # noqa: F401 - re-exported for existing internal tests/importers.
)
from vmec_jax.solvers.fixed_boundary.preconditioning.operators import (
    can_reassemble_precond_mats as _can_reassemble_precond_mats,
    PreconditionerCacheState as _PreconditionerCacheState,
    pshalf_from_s_jax as _pshalf_from_s_jax,
    pshalf_from_s_np as _pshalf_from_s_np,
    radial_tridi_smooth_dirichlet as _radial_tridi_smooth_dirichlet,
    resolve_preconditioner_tridi_policies as _resolve_preconditioner_tridi_policies,
    scale_m1_precond_rhs_from_mats as _scale_m1_precond_rhs_from_mats,
    update_preconditioner_cache as _update_preconditioner_cache,  # noqa: F401 - re-exported for existing internal tests/importers.
)
from vmec_jax.vmec_lforbal import plascur_edge_from_bcovar
from vmec_jax.solvers.fixed_boundary.results import (
    SolveVmecResidualResult,
)
from vmec_jax.solvers.fixed_boundary.diagnostics.axis_reset import (
    InitialAxisResetRuntimeCallbacks as _InitialAxisResetRuntimeCallbacks,
    reset_axis_from_boundary as _reset_axis_from_boundary_impl,
    run_initial_axis_reset_runtime as _run_initial_axis_reset_runtime,
    run_initial_axis_reset_setup as _run_initial_axis_reset_setup,
)
from vmec_jax.solvers.free_boundary.control import (
    free_boundary_iter_controls_vmec as _free_boundary_iter_controls_vmec,
    free_boundary_nestor_iteration_coupling as _free_boundary_nestor_iteration_coupling,
    free_boundary_prev_rz_fsq_next as _free_boundary_prev_rz_fsq_next,
    free_boundary_should_damp_constraint_baseline as _free_boundary_should_damp_constraint_baseline,
    free_boundary_turnon_resets_iter1_immediately as _free_boundary_turnon_resets_iter1_immediately,
)
from vmec_jax.solvers.free_boundary.diagnostics import sample_free_boundary_external_field as _sample_free_boundary_external_field
from vmec_jax.solvers.fixed_boundary.diagnostics.force import (
    maybe_dump_force_kernels as _maybe_dump_force_kernels,
    maybe_dump_gc as _maybe_dump_gc,
    maybe_dump_gcx2 as _maybe_dump_gcx2,
    maybe_dump_scalars as _maybe_dump_scalars,
    maybe_dump_tomnsps as _maybe_dump_tomnsps,
)
from vmec_jax.solvers.fixed_boundary.diagnostics.bsub import (
    maybe_dump_bsube as _maybe_dump_bsube,
    maybe_dump_bsube_terms as _maybe_dump_bsube_terms,
    maybe_dump_bsubh as _maybe_dump_bsubh,
    maybe_dump_bsubs as _maybe_dump_bsubs,
)
from vmec_jax.solvers.fixed_boundary.diagnostics.lambda_debug import (
    maybe_dump_lam_fsql1 as _maybe_dump_lam_fsql1,
    maybe_dump_lam_gcl as _maybe_dump_lam_gcl,
    maybe_dump_lam_prec as _maybe_dump_lam_prec,
    maybe_dump_lamcal as _maybe_dump_lamcal,
    maybe_dump_lulv as _maybe_dump_lulv,
    maybe_dump_precond_mats as _maybe_dump_precond_mats,
)
from vmec_jax.solvers.fixed_boundary.diagnostics.metric import (
    maybe_dump_gmetric as _maybe_dump_gmetric,
    maybe_dump_precond_inputs as _maybe_dump_precond_inputs,
    maybe_dump_xc as _maybe_dump_xc,
)
from vmec_jax.solvers.fixed_boundary.scan.math import (
    _kernel_arrays_from_k as _scan_math_kernel_arrays_from_k,
    _ptau_minmax_from_k_host as _scan_math_ptau_minmax_from_k_host,
    _ptau_minmax_from_k_jax as _scan_math_ptau_minmax_from_k_jax,
    build_ptau_minmax_context as _build_ptau_minmax_context,
)
from vmec_jax.solvers.fixed_boundary.scan.debug import (
    _emit_vmec2000_iter_row as _emit_scan_vmec2000_iter_row,
    _print_axis_guess as _print_scan_axis_guess,
    _print_vmec2000_row as _print_scan_vmec2000_row,
    _record_scan_device_ready,
)
from vmec_jax.solvers.fixed_boundary.scan.planning import (
    build_scan_timing_report as _build_scan_timing_report,
    default_vmec2000_controller_constants as _default_vmec2000_controller_constants,
    new_scan_timing_stats as _new_scan_timing_stats,
    scan_chunk_settings as _resolve_scan_chunk_settings,
    scan_timing_enabled as _scan_timing_enabled,
)
from vmec_jax.solvers.fixed_boundary.scan.controller import (
    Vmec2000ScanControllerContext,
    run_vmec2000_scan,
)
from vmec_jax.solvers.fixed_boundary.scan.runtime import (
    get_or_build_scan_runner as _get_or_build_scan_runner,
)
from vmec_jax.state import VMECState


_SCAN_RUNNER_CACHE: OrderedDict[tuple, Any] = OrderedDict()
_COMPUTE_FORCES_CACHE: OrderedDict[tuple, Any] = OrderedDict()

_HostRestartDecision = _residual_iter_policy.HostRestartDecision
_ResidualIterHistoryRecord = _residual_iter_policy.ResidualIterHistoryRecord
_Vmec2000ScanOptions = _residual_iter_policy.Vmec2000ScanOptions
_Vmec2000TimeControlDecision = _residual_iter_policy.Vmec2000TimeControlDecision

_m1_internal_to_physical_pair = _geometry_m1_internal_to_physical_pair
_mn_sin_to_signed_physical_batch = _geometry_mn_sin_to_signed_physical_batch
_rz_norm_np = _geometry_rz_norm_np

def _strict_update_step_jit(*args, **kwargs):
    kwargs.setdefault("has_jax_func", has_jax)
    return _precond_payload_facade._strict_update_step_jit(*args, **kwargs)


def _accepted_control_payload_jit(*args, **kwargs):
    kwargs.setdefault("has_jax_func", has_jax)
    return _precond_payload_facade._accepted_control_payload_jit(*args, **kwargs)


_cached_or_current_f_norm1_jax = _precond_payload_facade._cached_or_current_f_norm1_jax

_ptau_compute_jit = _precond_payload_facade._ptau_compute_jit

_hash_array_bytes = _solve_runtime._hash_array_bytes
_tree_has_tracer = _solve_runtime._tree_has_tracer
_scan_backend_name = _solve_runtime._scan_backend_name
_parse_iter_list = _solve_runtime._parse_iter_list
_dump_env_enabled = _solve_runtime._dump_env_enabled
_dump_iter_selected = _solve_runtime._dump_iter_selected
_runtime_env_enabled = _solve_runtime._runtime_env_enabled
_edge_signature_key = _solve_runtime._edge_signature_key
_edge_value_key = _solve_runtime._edge_value_key
_scan_fallback_policy = _solve_runtime._scan_fallback_policy
_residual_convergence_flags = _solve_runtime._residual_convergence_flags


def _scan_chunk_settings(
    *,
    max_iter_scan: int,
    nstep_screen: int,
    need_print: bool,
    lthreed: bool,
    spectral_mode_count: int | None = None,
) -> tuple[int, bool]:
    return _resolve_scan_chunk_settings(
        max_iter_scan=max_iter_scan,
        nstep_screen=nstep_screen,
        need_print=need_print,
        lthreed=lthreed,
        backend_name=_scan_backend_name(),
        chunk_size_env=os.getenv("VMEC_JAX_SCAN_CHUNK_SIZE", ""),
        spectral_mode_count=spectral_mode_count,
    )


def _resolve_startup_policy_for_residual_iter(
    *,
    max_iter: int,
    step_size: float,
    precompile_only: bool,
    signgs: int,
    lambda_update_scale: float,
    enforce_vmec_lambda_axis: bool,
    vmec2000_control: bool,
    reference_mode: bool,
    limit_dt_from_force: bool,
    limit_update_rms: bool,
    backtracking: bool,
    strict_update: bool,
    jit_precompile: bool,
    use_scan: bool,
    host_update_assembly: bool | None,
    state_has_tracer: bool,
    preconditioner_use_precomputed_tridi: bool | None,
    preconditioner_use_lax_tridi: bool | None,
    adjoint_trace: bool,
    adjoint_trace_mode: str,
    fsq_total_target: float | None,
    light_history: bool | None,
    resume_state_mode: str | None,
    use_restart_triggers: bool | None,
    use_direct_fallback: bool | None,
    vmecpp_restart: bool,
    verbose_vmec2000_table: bool,
    stage_prev_fsq: float | None,
    stage_transition_factor: float,
    stage_transition_scale: float,
    auto_flip_force: bool,
    jit_forces: bool,
) -> _residual_iter_policy.ResidualIterStartupPolicy:
    """Resolve host startup policy before entering the numerical loop."""

    return _resolve_residual_iter_startup_policy(
        max_iter=max_iter,
        step_size=step_size,
        precompile_only=precompile_only,
        signgs=signgs,
        lambda_update_scale=lambda_update_scale,
        enforce_vmec_lambda_axis=enforce_vmec_lambda_axis,
        vmec2000_control=vmec2000_control,
        reference_mode=reference_mode,
        limit_dt_from_force=limit_dt_from_force,
        limit_update_rms=limit_update_rms,
        backtracking=backtracking,
        strict_update=strict_update,
        jit_precompile=jit_precompile,
        use_scan=use_scan,
        host_update_assembly=host_update_assembly,
        backend_name=jax.default_backend(),
        scan_backend_name=_scan_backend_name(),
        state_has_tracer=state_has_tracer,
        env=os.environ,
        validate_options=validate_residual_iteration_options,
        resolve_tridi_policies=_resolve_preconditioner_tridi_policies,
        normalize_adjoint_trace_mode=_normalize_adjoint_trace_mode,
        normalize_resume_state_mode=_normalize_resume_state_mode,
        resolve_scan_fallback_policy=_scan_fallback_policy,
        preconditioner_use_precomputed_tridi=preconditioner_use_precomputed_tridi,
        preconditioner_use_lax_tridi=preconditioner_use_lax_tridi,
        adjoint_trace=adjoint_trace,
        adjoint_trace_mode=adjoint_trace_mode,
        fsq_total_target=fsq_total_target,
        light_history=light_history,
        resume_state_mode=resume_state_mode,
        use_restart_triggers=use_restart_triggers,
        use_direct_fallback=use_direct_fallback,
        vmecpp_restart=vmecpp_restart,
        verbose_vmec2000_table=verbose_vmec2000_table,
        stage_prev_fsq=stage_prev_fsq,
        stage_transition_factor=stage_transition_factor,
        stage_transition_scale=stage_transition_scale,
        auto_flip_force=auto_flip_force,
        jit_forces=jit_forces,
        heavy_dump_envs=_HEAVY_DUMP_ENVS,
        light_dump_envs=_LIGHT_DUMP_ENVS,
    )


def _prepare_residual_cache_keys_for_state(
    *,
    state0: VMECState,
    state0_has_tracer: bool,
    static: Any,
    wout_like: Any,
    indata: Any,
    include_constraint_force: bool,
) -> tuple[Any, Any, Any, Any, float | None, bool, Any, Any, Any, Any]:
    """Build static/profile/edge cache keys for residual force kernels."""

    if state0_has_tracer:
        edge_Rcos = jnp.asarray(state0.Rcos)[-1, :]
        edge_Rsin = jnp.asarray(state0.Rsin)[-1, :]
        edge_Zcos = jnp.asarray(state0.Zcos)[-1, :]
        edge_Zsin = jnp.asarray(state0.Zsin)[-1, :]
    else:
        edge_Rcos = np.asarray(state0.Rcos)[-1, :]
        edge_Rsin = np.asarray(state0.Rsin)[-1, :]
        edge_Zcos = np.asarray(state0.Zcos)[-1, :]
        edge_Zsin = np.asarray(state0.Zsin)[-1, :]

    constraint_tcon0: float | None = None
    if bool(include_constraint_force):
        constraint_tcon0 = float(indata.get_float("TCON0", 1.0))
    apply_lforbal = bool(indata.get_bool("LFORBAL", False)) if indata is not None else False

    cache_keys = _build_residual_cache_keys(
        static=static,
        wout_like=wout_like,
        edge_Rcos=edge_Rcos,
        edge_Rsin=edge_Rsin,
        edge_Zcos=edge_Zcos,
        edge_Zsin=edge_Zsin,
        constraint_tcon0=constraint_tcon0,
        hash_array_bytes_func=_hash_array_bytes,
        edge_signature_key_func=_edge_signature_key,
        edge_value_key_func=_edge_value_key,
    )
    return (
        edge_Rcos,
        edge_Rsin,
        edge_Zcos,
        edge_Zsin,
        constraint_tcon0,
        apply_lforbal,
        cache_keys.static_key,
        cache_keys.wout_key,
        cache_keys.edge_signature_key,
        cache_keys.edge_value_key,
    )


def _use_numpy_force_fast_path_policy(
    *,
    host_update_assembly: bool,
    max_iter: int,
    fast_path_env: str,
    max_iter_env: str,
) -> bool:
    """Resolve the optional NumPy force fast path for short CPU stages."""

    try:
        numpy_force_max_iter = max(0, int(max_iter_env))
    except Exception:
        numpy_force_max_iter = 600
    if fast_path_env in ("1", "true", "yes", "on"):
        return bool(host_update_assembly)
    if fast_path_env in ("0", "false", "no", "off"):
        return False
    short_stage_for_numpy_force = int(max_iter) <= int(numpy_force_max_iter)
    return bool(host_update_assembly) and bool(short_stage_for_numpy_force)


_default_scan_core = _solve_runtime._default_scan_core


def solve_fixed_boundary_residual_iter(
    state0: VMECState,
    static,
    *,
    indata,
    signgs: int,
    ftol: float | None = None,
    max_iter: int = 50,
    step_size: float = 1.0,
    initial_flip_sign: float = 1.0,
    include_constraint_force: bool = True,
    apply_m1_constraints: bool = True,
    precond_radial_alpha: float = 0.5,
    precond_lambda_alpha: float = 0.5,
    mode_diag_exponent: float = 0.0,
    auto_flip_force: bool = True,
    divide_by_scalxc_for_update: bool = False,
    lambda_update_scale: float = 1.0,
    enforce_vmec_lambda_axis: bool = False,
    vmec2000_control: bool = False,
    strict_update: bool = True,
    backtracking: bool = False,
    limit_dt_from_force: bool = False,
    limit_update_rms: bool = False,
    reference_mode: bool = False,
    use_restart_triggers: bool | None = None,
    vmecpp_restart: bool = False,
    stage_prev_fsq: float | None = None,
    stage_transition_factor: float = 50.0,
    stage_transition_scale: float = 0.5,
    use_direct_fallback: bool | None = None,
    verbose: bool = True,
    verbose_vmec2000_table: bool = True,
    jit_forces: bool = True,
    jit_warmup_iters: int = 0,
    jit_precompile: bool = False,
    use_scan: bool = False,
    precompile_only: bool = False,
    resume_state: dict | None = None,
    scan_minimal_default: bool | None = None,
    light_history: bool | None = None,
    resume_state_mode: str | None = None,
    fsq_total_target: float | None = None,
    host_update_assembly: bool | None = None,
    preconditioner_use_precomputed_tridi: bool | None = None,
    preconditioner_use_lax_tridi: bool | None = None,
    adjoint_trace: bool = False,
    adjoint_trace_mode: str = "full",
    external_field_provider_kind: str | None = None,
    external_field_provider_static: Any = None,
    external_field_provider_params: Any = None,
    free_boundary_activate_fsq: float | None = None,
    state_only: bool = False,
    return_final_force_payload: bool = False,
) -> SolveVmecResidualResult:
    """VMEC-style fixed-point update loop using preconditioned force residuals."""
    _solve_wall_start = time.perf_counter()
    if not has_jax():
        raise ImportError("solve_fixed_boundary_residual_iter requires JAX (jax + jaxlib)")

    timing_enabled = _runtime_env_enabled(os.getenv("VMEC_JAX_TIMING", ""))
    timing_detail_enabled = timing_enabled and _runtime_env_enabled(os.getenv("VMEC_JAX_TIMING_DETAIL", ""))
    _setup_phase_timings = _initial_setup_phase_timings()
    state0_has_tracer = _tree_has_tracer(state0)

    _setup_timer_start = partial(_runtime_setup_timer_start, timing_enabled=bool(timing_enabled), perf_counter=time.perf_counter)
    _record_setup_timing = partial(_runtime_record_setup_timing, _setup_phase_timings, perf_counter=time.perf_counter)

    startup_policy = _resolve_startup_policy_for_residual_iter(
        max_iter=max_iter,
        step_size=step_size,
        precompile_only=precompile_only,
        signgs=signgs,
        lambda_update_scale=lambda_update_scale,
        enforce_vmec_lambda_axis=enforce_vmec_lambda_axis,
        vmec2000_control=vmec2000_control,
        reference_mode=reference_mode,
        limit_dt_from_force=limit_dt_from_force,
        limit_update_rms=limit_update_rms,
        backtracking=backtracking,
        strict_update=strict_update,
        jit_precompile=jit_precompile,
        use_scan=use_scan,
        host_update_assembly=host_update_assembly,
        state_has_tracer=state0_has_tracer,
        preconditioner_use_precomputed_tridi=preconditioner_use_precomputed_tridi,
        preconditioner_use_lax_tridi=preconditioner_use_lax_tridi,
        adjoint_trace=adjoint_trace,
        adjoint_trace_mode=adjoint_trace_mode,
        fsq_total_target=fsq_total_target,
        light_history=light_history,
        resume_state_mode=resume_state_mode,
        use_restart_triggers=use_restart_triggers,
        use_direct_fallback=use_direct_fallback,
        vmecpp_restart=vmecpp_restart,
        verbose_vmec2000_table=verbose_vmec2000_table,
        stage_prev_fsq=stage_prev_fsq,
        stage_transition_factor=stage_transition_factor,
        stage_transition_scale=stage_transition_scale,
        auto_flip_force=auto_flip_force,
        jit_forces=jit_forces,
    )
    max_iter = startup_policy.max_iter
    step_size = startup_policy.step_size
    host_update_assembly = startup_policy.host_update_assembly
    adjoint_trace = startup_policy.adjoint_trace
    adjoint_trace_mode = startup_policy.adjoint_trace_mode
    preconditioner_use_precomputed_tridi_policy = startup_policy.preconditioner_use_precomputed_tridi_policy
    preconditioner_use_lax_tridi_policy = startup_policy.preconditioner_use_lax_tridi_policy

    signgs = startup_policy.signgs
    fsq_total_target = startup_policy.fsq_total_target
    lambda_update_scale = startup_policy.lambda_update_scale
    vmec2000_control = startup_policy.vmec2000_control
    badjac_use_state = startup_policy.badjac_use_state
    resume_state_mode = startup_policy.resume_state_mode
    ptau_tol = startup_policy.ptau_tol
    reference_mode = startup_policy.reference_mode
    use_direct_fallback = startup_policy.use_direct_fallback
    vmecpp_restart = startup_policy.vmecpp_restart
    verbose_vmec2000_table = startup_policy.verbose_vmec2000_table
    stage_transition_factor = startup_policy.stage_transition_factor
    stage_transition_scale = startup_policy.stage_transition_scale
    stage_prev_fsq = startup_policy.stage_prev_fsq
    jit_forces = startup_policy.jit_forces
    use_scan = startup_policy.use_scan
    limit_dt_from_force = startup_policy.limit_dt_from_force
    limit_update_rms = startup_policy.limit_update_rms
    backtracking = startup_policy.backtracking
    strict_update = startup_policy.strict_update
    if startup_policy.disabled_jit_for_dumps:
        if verbose:
            print("[solve_fixed_boundary_residual_iter] jit_forces disabled (debug dumps enabled)")
    track_history = startup_policy.track_history

    from vmec_jax.static import build_static
    from vmec_jax.boundary import boundary_from_indata
    from vmec_jax.init_guess import (
        _recompute_axis_from_boundary,
        _recompute_axis_from_state_vmec,
        _read_axis_coeffs,
        initial_guess_from_boundary,
    )
    from vmec_jax.vmec_residue import (
        vmec_gcx2_from_tomnsps,
        vmec_gcx2_from_tomnsps_np,
        vmec_scalxc_from_s,
    )
    from vmec_jax.vmec_jacobian import vmec_half_mesh_jacobian_from_state
    from vmec_jax.vmec_tomnsp import TomnspsRZL, vmec_angle_grid, vmec_trig_tables
    from vmec_jax.free_boundary import NestorRuntimeState, nestor_external_only_step

    # VMEC2000 evaluates force kernels on VMEC's internal angle grid. Rebuild
    # static data when the caller supplied a plotting/full-grid object.
    _t_setup_static_grid = _setup_timer_start()
    static, cfg = _build_residual_static_grid_setup(
        static=static,
        build_static_func=build_static,
        vmec_angle_grid_func=vmec_angle_grid,
    )
    _record_setup_timing("setup_static_grid_rebuild", _t_setup_static_grid)
    # Free-boundary control + coupling path:
    # VMEC-style ivac/ivacskip cadence with edge bsqvac coupling.
    _t_setup_freeb_policy = _setup_timer_start()
    _freeb_policy = _resolve_free_boundary_setup_policy(
        cfg,
        external_field_provider_kind=external_field_provider_kind,
        use_scan=use_scan,
        freeb_couple_env=os.getenv("VMEC_JAX_FREEB_COUPLE_EDGE", "1"),
        freeb_sample_env=os.getenv("VMEC_JAX_FREEB_SAMPLE_EXTERNAL", "1"),
        jit_strict_update_env=os.getenv("VMEC_JAX_JIT_STRICT_UPDATE", "auto"),
        backend_name=_scan_backend_name(),
        host_update_assembly=host_update_assembly,
        cpu_work_limit_env=os.getenv("VMEC_JAX_HOST_UPDATE_CPU_WORK_LIMIT", "1000"),
    )
    free_boundary_enabled = _freeb_policy.free_boundary_enabled
    direct_free_boundary_provider = _freeb_policy.direct_free_boundary_provider
    freeb_nvacskip = _freeb_policy.freeb_nvacskip
    freeb_nvskip0 = _freeb_policy.freeb_nvskip0
    freeb_couple_edge = _freeb_policy.freeb_couple_edge
    use_scan = _freeb_policy.use_scan
    freeb_sample_external = _freeb_policy.freeb_sample_external
    jit_strict_update_enabled = _freeb_policy.jit_strict_update_enabled
    _record_setup_timing("setup_freeb_policy", _t_setup_freeb_policy)

    _attach_freeb_diag = partial(
        _runtime_attach_free_boundary_external_field_diag,
        free_boundary_enabled=free_boundary_enabled,
        external_field_provider_kind=external_field_provider_kind,
        freeb_sample_external=freeb_sample_external,
        sample_external_field_func=_sample_free_boundary_external_field,
        static=static,
        result_type=SolveVmecResidualResult,
    )

    _t_setup_boundary_profiles = _setup_timer_start()
    idx00 = _mode00_index(static.modes)
    concrete_host_setup = bool(host_update_assembly) and not bool(state0_has_tracer) and not bool(use_scan)
    s = np.asarray(static.s) if concrete_host_setup else jnp.asarray(static.s)
    freeb_pres_scale = _free_boundary_pressure_edge_scale(
        free_boundary_enabled=bool(free_boundary_enabled),
        indata=indata,
        s=s,
    )
    dtype_state = np.asarray(state0.Rcos).dtype if concrete_host_setup else jnp.asarray(state0.Rcos).dtype
    zeros_like_radial = np.zeros if concrete_host_setup else jnp.zeros
    zero_precond_diag = (
        zeros_like_radial((int(s.shape[0]),), dtype=dtype_state),
        zeros_like_radial((int(s.shape[0]),), dtype=dtype_state),
    )
    zero_tcon = zeros_like_radial((int(s.shape[0]),), dtype=dtype_state)
    constraint_active_false = np.asarray(False) if concrete_host_setup else jnp.asarray(False)

    # Boundary coefficients for VMEC-style bad-Jacobian reset are needed only
    # when a reset is actually applied.  Build them lazily so ordinary cold
    # solves do not pay a duplicate boundary conversion during setup.
    boundary_for_axis = None
    axis_reset_done = bool(resume_state is not None)
    lmove_axis = True if indata is None else bool(indata.get_bool("LMOVE_AXIS", True))
    axis_reset_config = _resolve_axis_reset_config(
        force_axis_reset_env=os.getenv("VMEC_JAX_FORCE_AXIS_RESET_INIT", "0"),
        axis_reset_always_3d_env=os.getenv("VMEC_JAX_AXIS_RESET_ALWAYS_3D", "0"),
        axis_reset_fsq_min_env=os.getenv("VMEC_JAX_AXIS_RESET_FSQ_MIN", "1.0"),
    )
    force_axis_reset = axis_reset_config.force_axis_reset
    axis_reset_always_3d = axis_reset_config.axis_reset_always_3d
    axis_reset_fsq_min = axis_reset_config.axis_reset_fsq_min

    # VMEC applies the m=0 lambda axis-closure during real-space synthesis
    # without overwriting stored axis coefficients; only enforce the gauge here.
    _apply_vmec_lambda_axis_rules = partial(
        _apply_vmec_lambda_axis_rules_to_state,
        enforce_vmec_lambda_axis=startup_policy.enforce_vmec_lambda_axis,
        host_update_assembly=host_update_assembly,
        idx00=idx00,
    )

    axis_reset_coeffs = None

    def _reset_axis_from_boundary(
        st: VMECState,
        *,
        k_guess=None,
        full_reset: bool = False,
        refine_axis_guess: bool = True,
    ) -> VMECState:
        nonlocal axis_reset_coeffs, boundary_for_axis
        if boundary_for_axis is None and indata is not None:
            boundary_for_axis = boundary_from_indata(indata, static.modes, apply_m1_constraint=True)
        st_out, coeffs = _reset_axis_from_boundary_impl(
            st,
            boundary_for_axis=boundary_for_axis,
            static=static,
            indata=indata,
            signgs=int(signgs),
            trig=trig,
            k_guess=k_guess,
            full_reset=full_reset,
            refine_axis_guess=refine_axis_guess,
            zero_precond_diag=zero_precond_diag,
            zero_tcon=zero_tcon,
            constraint_active_false=constraint_active_false,
            compute_forces_iter_func=_compute_forces_iter,
            apply_vmec_lambda_axis_rules_func=_apply_vmec_lambda_axis_rules,
            initial_guess_from_boundary_func=initial_guess_from_boundary,
            read_axis_coeffs_func=_read_axis_coeffs,
            recompute_axis_from_state_vmec_func=_recompute_axis_from_state_vmec,
            recompute_axis_from_boundary_func=_recompute_axis_from_boundary,
            axis_dump_dir=os.environ.get("VMEC_JAX_DUMP_AXIS_DIR", "").strip(),
        )
        if coeffs is not None:
            axis_reset_coeffs = coeffs
        return st_out

    host_profile_setup = _resolve_host_profile_setup(
        backend_name=_scan_backend_name(),
        profile_setup_env=os.getenv("VMEC_JAX_HOST_PROFILE_SETUP", "auto"),
        profile_setup_has_work=_indata_has_profile_setup_work(indata),
    )
    try:
        wout_like, trig = _build_residual_profile_setup(
            indata=indata,
            static=static,
            s=s,
            signgs=signgs,
            idx00=idx00,
            state0=state0,
            state0_has_tracer=bool(state0_has_tracer),
            host_update_assembly=bool(host_update_assembly),
            host_profile_setup=bool(host_profile_setup) and has_jax(),
            build_wout_like_profiles_func=_build_wout_like_profiles_from_indata,
            resolve_residual_trig_func=_residual_force_context_helpers.resolve_residual_trig,
            vmec_trig_tables_func=vmec_trig_tables,
            tree_has_tracer_func=_tree_has_tracer,
            jnp_module=jnp,
            setup_phase_timings=_setup_phase_timings,
            timing_enabled=bool(timing_enabled),
            perf_counter_func=time.perf_counter,
        )
    except Exception:
        if bool(startup_policy.precompile_only):
            return _precompile_only_residual_iter_result(result_type=SolveVmecResidualResult, state=state0)
        raise
    _record_setup_timing("setup_boundary_profiles", _t_setup_boundary_profiles)
    idx00 = _mode00_index(static.modes)
    _state_dtype = jnp.asarray(state0.Rcos).dtype if state0_has_tracer else np.asarray(state0.Rcos).dtype
    lambda_update_scale_j = (
        jnp.asarray(lambda_update_scale, dtype=_state_dtype)
        if state0_has_tracer
        else np.asarray(lambda_update_scale, dtype=_state_dtype)
    )

    # VMEC stores Fourier coefficients in an internal (mscale/nscale) basis and
    # uses `scalxc` to represent odd-m modes in 1/sqrt(s) form. The force pipeline
    # applies `scalxc` after `tomnsps` (see `funct3d.f: gc = gc*scalxc`) so the
    # residual/preconditioner updates operate in the same internal coefficient
    # space as `VMECState`.

    _t_setup_cache_key_hash = _setup_timer_start()
    (
        edge_Rcos,
        edge_Rsin,
        edge_Zcos,
        edge_Zsin,
        constraint_tcon0,
        apply_lforbal,
        static_key,
        wout_key,
        edge_signature_key,
        edge_value_key,
    ) = _prepare_residual_cache_keys_for_state(
        state0=state0,
        state0_has_tracer=bool(state0_has_tracer),
        static=static,
        wout_like=wout_like,
        indata=indata,
        include_constraint_force=bool(include_constraint_force),
    )
    _record_setup_timing("setup_cache_key_hash", _t_setup_cache_key_hash)

    _t_setup_ptau_constants = _setup_timer_start()
    _ptau_bindings = _build_residual_ptau_bindings(
        s=s,
        has_jax_value=has_jax(),
        s_has_tracer=_tree_has_tracer(s),
        pshalf_from_s_np_func=_pshalf_from_s_np,
        pshalf_from_s_jax_func=_pshalf_from_s_jax,
        build_context_func=_build_ptau_minmax_context,
        compute_jit_func=_ptau_compute_jit,
        ptau_minmax_host_helper=_ptau_minmax_from_k_host_helper,
        ptau_minmax_helper=_ptau_minmax_helper,
        scan_ptau_minmax_host_func=_scan_math_ptau_minmax_from_k_host,
        scan_ptau_minmax_jax_func=_scan_math_ptau_minmax_from_k_jax,
        accepted_control_ptau_arrays_helper=_accepted_control_ptau_arrays_helper,
        scan_kernel_arrays_from_k_func=_scan_math_kernel_arrays_from_k,
        has_jax_func=has_jax,
        host_update_assembly=bool(host_update_assembly),
    )
    _ptau_context, _ptau_minmax_from_k_host, _ptau_minmax, _accepted_control_ptau_arrays = _ptau_bindings
    _record_setup_timing("setup_ptau_constants", _t_setup_ptau_constants)
    _maybe_dump_jacobian_terms = partial(
        _maybe_dump_jacobian_terms_helper,
        s=s,
        dump_func=_maybe_dump_jacobian_terms_record,
    )
    _maybe_dump_ptau = partial(
        _maybe_dump_ptau_helper,
        getenv=os.getenv,
        dump_func=_runtime_maybe_dump_ptau,
    )

    _numpy_precond_policy = _numpy_preconditioner_apply_policy(
        host_update_assembly=bool(host_update_assembly),
        max_iter=int(max_iter),
        mpol=int(getattr(cfg, "mpol", 0)),
        ntor=int(getattr(cfg, "ntor", 0)),
        max_iter_env=os.getenv("VMEC_JAX_NUMPY_PRECOND_MAX_ITER", "240"),
        min_mode_count_env=os.getenv("VMEC_JAX_NUMPY_PRECOND_MIN_MODES", "0"),
    )
    _use_numpy_preconditioner_apply = bool(_numpy_precond_policy.enabled)
    preconditioner_ops = _precond_payload_facade.residual_preconditioner_operators(
        trig=trig,
        s=s,
        cfg=cfg,
        use_numpy_preconditioner_apply=bool(_use_numpy_preconditioner_apply),
        tree_has_tracer_func=_tree_has_tracer,
        radial_tridi_smooth_dirichlet_func=_radial_tridi_smooth_dirichlet,
        jnp_module=jnp,
    )
    _apply_radial_tridi = preconditioner_ops.apply_radial_tridi
    _apply_radial_tridi_batched = preconditioner_ops.apply_radial_tridi_batched
    _lambda_preconditioner = preconditioner_ops.lambda_preconditioner
    _rz_preconditioner_matrices_local = preconditioner_ops.rz_preconditioner_matrices
    _rz_preconditioner_apply_local = preconditioner_ops.rz_preconditioner_apply
    _rz_preconditioner = preconditioner_ops.rz_preconditioner

    _compute_forces = _make_residual_force_evaluator(
        static=static,
        wout_like=wout_like,
        trig=trig,
        s=s,
        signgs=signgs,
        constraint_tcon0=constraint_tcon0,
        freeb_pres_scale=freeb_pres_scale,
        apply_lforbal=apply_lforbal,
        apply_m1_constraints=bool(apply_m1_constraints),
        runtime_env_enabled=_runtime_env_enabled,
        getenv=os.getenv,
        maybe_dump_hlo_kernel=_hlo_dump_helpers.maybe_dump_hlo_kernel,
        dump_hooks={
            "bsube": _maybe_dump_bsube,
            "bsube_terms": _maybe_dump_bsube_terms,
            "bsubh": _maybe_dump_bsubh,
            "bsubs": _maybe_dump_bsubs,
            "lulv": _maybe_dump_lulv,
            "jacobian_terms": _maybe_dump_jacobian_terms,
            "precond_inputs": _maybe_dump_precond_inputs,
            "gmetric": _maybe_dump_gmetric,
            "force_kernels": _maybe_dump_force_kernels,
            "tomnsps": _maybe_dump_tomnsps,
            "gc": _maybe_dump_gc,
            "gcx2": _maybe_dump_gcx2,
            "scalars": _maybe_dump_scalars,
        },
    )

    _hlo_dump_helpers.maybe_dump_initial_residual_hlo_kernels(
        state0=state0,
        static=static,
        wout_like=wout_like,
        trig=trig,
        constraint_tcon0=constraint_tcon0,
        apply_lforbal=bool(apply_lforbal),
        maybe_dump_kernel=_hlo_dump_helpers.maybe_dump_hlo_kernel,
    )

    _compute_forces_impl = _compute_forces

    use_numpy_force_fast_path = _use_numpy_force_fast_path_policy(
        host_update_assembly=bool(host_update_assembly),
        max_iter=int(max_iter),
        fast_path_env=os.getenv("VMEC_JAX_NUMPY_FORCE_FAST_PATH", "auto").strip().lower(),
        max_iter_env=os.getenv("VMEC_JAX_NUMPY_FORCE_MAX_ITER", "600"),
    )

    # NumPy hot-path: wrap _compute_forces_impl with pure-NumPy module patching.
    # Used for short host-update CPU stages to eliminate JAX dispatch overhead.
    numpy_force = _prepare_numpy_force_fast_path(
        host_update_assembly=bool(host_update_assembly),
        use_numpy_force_fast_path=bool(use_numpy_force_fast_path),
        has_jax_func=has_jax,
        compute_forces_impl=_compute_forces_impl,
        state0=state0,
        static=static,
        trig=trig,
        wout_like=wout_like,
    )
    static = numpy_force.static
    trig = numpy_force.trig
    wout_like = numpy_force.wout_like
    _compute_forces_np = numpy_force.compute_forces_np

    compute_cache_key = _compute_forces_jit_cache_key(
        static_key=static_key,
        wout_key=wout_key,
        signgs=int(signgs),
        apply_m1_constraints=bool(apply_m1_constraints),
    )
    if jit_forces:
        _compute_forces_nodump = partial(
            _compute_forces_without_iter_dump,
            compute_forces_impl=_compute_forces_impl,
        )

        _compute_forces = _select_compute_forces_callable(
            _compute_forces_nodump,
            differentiating_scan=bool(startup_policy.differentiating_scan),
            cache=_COMPUTE_FORCES_CACHE,
            cache_key=compute_cache_key,
            jit_func=jit,
            cache_get=_jit_cache_get,
            cache_put=_jit_cache_put,
            cache_env_name="VMEC_JAX_COMPUTE_FORCES_CACHE_SIZE",
            cache_default=32,
        )

    _maybe_precompile_residual_force_kernels(
        jit_forces=bool(jit_forces),
        jit_precompile=bool(startup_policy.jit_precompile),
        has_jax_func=has_jax,
        jax_module=jax,
        jnp_module=jnp,
        compute_forces_np=_compute_forces_np,
        compute_forces=_compute_forces,
        state0=state0,
        dtype_state=dtype_state,
        zero_precond_diag=zero_precond_diag,
        zero_tcon=zero_tcon,
        constraint_active_false=constraint_active_false,
        backtracking=bool(backtracking),
        reference_mode=bool(reference_mode),
        use_direct_fallback=bool(use_direct_fallback),
        strict_update=bool(strict_update),
        jit_strict_update_enabled=bool(jit_strict_update_enabled),
        host_update_assembly=bool(host_update_assembly),
        limit_dt_from_force=bool(limit_dt_from_force),
        limit_update_rms=bool(limit_update_rms),
        tree_has_tracer_func=_tree_has_tracer,
        track_history=bool(track_history),
        verbose=bool(verbose),
        adjoint_trace=bool(adjoint_trace),
        adjoint_trace_mode=adjoint_trace_mode,
        strict_update_step_jit_func=_strict_update_step_jit,
        static=static,
        divide_by_scalxc_for_update=bool(divide_by_scalxc_for_update),
        free_boundary_enabled=bool(free_boundary_enabled),
        step_size=float(step_size),
        initial_flip_sign=float(initial_flip_sign),
    )

    if startup_policy.precompile_only:
        return _precompile_only_residual_iter_result(result_type=SolveVmecResidualResult, state=state0)

    _iter_idx_for_dump = partial(_residual_iter_dump_index, jit_forces=bool(jit_forces))

    warmup_iters = int(jit_warmup_iters) if bool(jit_forces) else 0

    _compute_forces_iter = partial(
        _compute_forces_iter_runtime,
        compute_forces_impl=_compute_forces_impl,
        compute_forces=_compute_forces,
        compute_forces_np=_compute_forces_np,
        warmup_iters=warmup_iters,
    )

    _t_setup_index_constants = _setup_timer_start()
    mpol = int(static.cfg.mpol)
    ntor = int(static.cfg.ntor)
    nrange = ntor + 1
    ncoeff = int(jnp.asarray(state0.Rcos).shape[1])
    # On accelerator host-forward runs the initial row/gauge enforcement is
    # setup work.  Using the NumPy row-assignment path avoids several tiny eager
    # device dispatches without touching traced/differentiable solves.
    setup_host_enforce = _resolve_setup_host_enforce(
        setup_host_enforce_env=os.getenv("VMEC_JAX_HOST_SETUP_ENFORCE", "auto"),
        host_update_assembly=bool(host_update_assembly),
        use_scan=bool(use_scan),
        state_has_tracer=state0_has_tracer,
        backend_name=_scan_backend_name(),
    )

    _mode_context = _build_mode_transform_context(
        static=static,
        state0=state0,
        s=s,
        host_update_assembly=bool(host_update_assembly),
        setup_host_enforce=bool(setup_host_enforce),
        divide_by_scalxc_for_update=bool(divide_by_scalxc_for_update),
        mode_diag_exponent=mode_diag_exponent,
        tree_has_tracer=_tree_has_tracer,
        vmec_scalxc_from_s=vmec_scalxc_from_s,
    )
    m0_mask = _mode_context.m0_mask
    w_mode_mn = _mode_context.w_mode_mn
    w_mode_mn_np = _mode_context.w_mode_mn_np
    _state0_dtype = _mode_context.state0_dtype
    _record_setup_timing("setup_index_constants", _t_setup_index_constants)

    _mn_cos_to_signed = _mode_context.mn_cos_to_signed
    _mn_sin_to_signed = _mode_context.mn_sin_to_signed
    _mn_cos_to_signed_physical = _mode_context.mn_cos_to_signed_physical
    _mn_sin_to_signed_physical = _mode_context.mn_sin_to_signed_physical
    _mn_sin_to_signed_physical_lambda = _mode_context.mn_sin_to_signed_physical_lambda
    _mn_cos_to_signed_physical_lambda = _mode_context.mn_cos_to_signed_physical_lambda
    _physical_delta_transforms = (_mn_cos_to_signed_physical, _mn_sin_to_signed_physical, _mn_cos_to_signed_physical_lambda, _mn_sin_to_signed_physical_lambda)
    _internal_delta_transforms = (_mn_cos_to_signed, _mn_sin_to_signed, _mn_cos_to_signed, _mn_sin_to_signed)
    _rz_norm_np = _mode_context.rz_norm_np
    _rz_norm = _mode_context.rz_norm
    state_setup = _build_residual_state_setup(
        state0=state0,
        static=static,
        s=s,
        edge_Rcos=edge_Rcos,
        edge_Rsin=edge_Rsin,
        edge_Zcos=edge_Zcos,
        edge_Zsin=edge_Zsin,
        free_boundary_enabled=bool(free_boundary_enabled),
        host_update_assembly=bool(host_update_assembly),
        setup_host_enforce=bool(setup_host_enforce),
        idx00=idx00,
        mpol=mpol,
        nrange=nrange,
        state0_dtype=_state0_dtype,
        apply_lambda_axis_rules=_apply_vmec_lambda_axis_rules,
        tree_has_tracer=_tree_has_tracer,
        has_jax_func=has_jax,
    )
    state = state_setup.state
    _precomputed_axis_mask_np = state_setup.precomputed_axis_mask_np
    _jnp_state_dtype = state_setup.jnp_state_dtype
    _jnp_zero_m1_0 = state_setup.jnp_zero_m1_0
    _jnp_zero_m1_1 = state_setup.jnp_zero_m1_1
    _jnp_true_bool = state_setup.jnp_true_bool
    _jnp_false_bool = state_setup.jnp_false_bool
    _zeros_coeff_np = state_setup.zeros_coeff_np
    _zeros_dR_np = state_setup.zeros_dR_np
    delta_s = state_setup.delta_s

    ftol = float(indata.get_float("FTOL", 1e-13)) if ftol is None else float(ftol)
    gamma = float(indata.get_float("GAMMA", 0.0))
    if abs(gamma - 1.0) < 1e-14:
        raise ValueError("GAMMA=1 makes wp/(gamma-1) singular (VMEC objective undefined)")

    stage_prev_fsq_j = None
    if stage_prev_fsq is not None:
        try:
            stage_prev_fsq_j = jnp.asarray(float(stage_prev_fsq), dtype=dtype)
        except Exception:
            stage_prev_fsq_j = None
    _t_setup_update_constants = _setup_timer_start()
    _record_setup_timing("setup_update_constants", _t_setup_update_constants)


    if use_scan:
        scan_outcome = dispatch_residual_scan_path(
            namespace=locals(),
            state=state,
            hooks=ResidualScanPathHooks(
                run_vmec2000_scan=run_vmec2000_scan,
                scan_context_type=Vmec2000ScanControllerContext,
                scan_fallback_decision=_scan_fallback_decision,
                scan_fallback_message=_scan_fallback_message,
                run_accelerated_scan=_run_accelerated_residual_scan,
                converged_residuals_func=_runtime_converged_residuals_scan_fast,
                scan_device_ready_recorder=_record_scan_device_ready,
                get_or_build_scan_runner=_get_or_build_scan_runner,
                jit_cache_get=_jit_cache_get,
                jit_cache_put=_jit_cache_put,
                record_scan_runner_cache_miss_categories=_record_scan_runner_cache_miss_categories,
                scan_timing_enabled=_scan_timing_enabled,
                new_scan_timing_stats=_new_scan_timing_stats,
                build_scan_timing_report=_build_scan_timing_report,
                runtime_env_enabled=_runtime_env_enabled,
                scan_backend_name=_scan_backend_name,
                scan_chunk_settings=_scan_chunk_settings,
                tree_has_tracer=_tree_has_tracer,
                scan_runner_cache=_SCAN_RUNNER_CACHE,
                enforce_fixed_boundary_and_axis=_enforce_fixed_boundary_and_axis,
                jax_module=jax,
                jnp_module=jnp,
                jit_func=jit,
                perf_counter=time.perf_counter,
            ),
        )
        if scan_outcome.handled:
            return scan_outcome.result
        use_scan = scan_outcome.use_scan
        state = scan_outcome.state
        resume_state = scan_outcome.resume_state

    profile_window_config = _resolve_residual_profile_window(
        profile_window_env=os.getenv("VMEC_JAX_PROFILE_WINDOW", ""),
        profile_dir_env=os.getenv("VMEC_JAX_PROFILE_DIR", ""),
    )
    profile_started = profile_window_config.started
    profile_active = profile_window_config.active
    profile_start_iter = profile_window_config.start_iter
    profile_dir = profile_window_config.directory
    profile_perfetto = _runtime_env_enabled(os.getenv("VMEC_JAX_PROFILE_PERFETTO", "1"))

    timing_stats = _runtime_new_residual_iter_timing_stats(_setup_phase_timings)
    _record_timing = partial(
        _record_elapsed_timing,
        bool(timing_enabled),
        timing_stats,
        perf_counter=time.perf_counter,
    )
    _record_compute_force_timing = partial(
        _runtime_record_compute_force_timing,
        timing_enabled=bool(timing_enabled),
        timing_stats=timing_stats,
        perf_counter=time.perf_counter,
        block_until_ready=jax.block_until_ready if has_jax() else None,
    )

    history_lists = _new_residual_iter_histories()
    fsqz2_history = history_lists["fsqz2_history"]
    adjoint_step_trace_history = history_lists["adjoint_step_trace_history"]

    r00_last = float("nan")
    z00_last = float("nan")
    wb_last = float("nan")
    wp_last = float("nan")
    w_vmec_last = float("nan")

    # Conjugate-gradient-like time-stepping state.
    controller_constants = _default_vmec2000_controller_constants()
    k_ndamp = controller_constants.ndamp
    controller_state = _initial_residual_controller_state(
        step_size=float(step_size),
        k_ndamp=int(k_ndamp),
        initial_flip_sign=float(initial_flip_sign),
        state_checkpoint=state,
    )
    (
        time_step, inv_tau, fsq_prev, fsq0_prev, flip_sign, iter1, ijacob, bad_resets, res0, res1,
        prev_rz_fsq, bad_growth_streak, huge_force_restart_count, state_checkpoint,
    ) = _controller_state_legacy_values(controller_state)

    def _set_controller_state(next_state: _ResidualControllerState) -> None:
        nonlocal controller_state, time_step, inv_tau, fsq_prev, fsq0_prev, flip_sign, iter1, ijacob
        nonlocal bad_resets, res0, res1, prev_rz_fsq, bad_growth_streak, huge_force_restart_count
        nonlocal state_checkpoint
        controller_state = next_state
        (
            time_step, inv_tau, fsq_prev, fsq0_prev, flip_sign, iter1, ijacob, bad_resets, res0, res1,
            prev_rz_fsq, bad_growth_streak, huge_force_restart_count, state_checkpoint,
        ) = _controller_state_legacy_values(controller_state)

    def _current_controller_state() -> _ResidualControllerState:
        return _controller_state_from_runtime_scalars(
            time_step=float(time_step),
            inv_tau=list(inv_tau),
            fsq_prev=float(fsq_prev),
            fsq0_prev=float(fsq0_prev),
            flip_sign=float(flip_sign),
            iter1=int(iter1),
            ijacob=int(ijacob),
            bad_resets=int(bad_resets),
            res0=float(res0),
            res1=float(res1),
            prev_rz_fsq=float(prev_rz_fsq),
            bad_growth_streak=int(bad_growth_streak),
            huge_force_restart_count=int(huge_force_restart_count),
            state_checkpoint=state_checkpoint,
        )

    def _apply_controller_update(update_func, update) -> None:
        _set_controller_state(_apply_controller_state_update(_current_controller_state(), update_func, update))

    def _apply_controller_sample(sample_func, decision) -> None:
        _set_controller_state(sample_func(_current_controller_state(), decision, state_checkpoint=state))

    _initial_velocity = _initial_residual_velocity_state(
        state=state,
        mpol=mpol,
        nrange=nrange,
        host_update_assembly=bool(host_update_assembly),
        reference_mode=bool(reference_mode),
    )
    velocity_blocks = _initial_velocity.velocities
    max_coeff_delta_rms = _initial_velocity.max_coeff_delta_rms
    max_update_rms = _initial_velocity.max_update_rms
    # VMEC runvmec/funct3d cadence starts free-boundary control at ivac=0.
    # Starting at -1 delays vacuum turn-on by one accepted iteration.
    # VMEC initializes ivac=-1 (reset_params.f), then promotes to 0/1/...
    # once free-boundary activation criteria are met.
    freeb_loop_state = _runtime_initial_free_boundary_loop_state(
        nvacskip=int(freeb_nvacskip),
        nvskip0=int(freeb_nvskip0),
        wout_like=wout_like,
    )
    freeb_ivac = freeb_loop_state.ivac
    freeb_ivacskip = freeb_loop_state.ivacskip
    freeb_nvacskip = freeb_loop_state.nvacskip
    freeb_nvskip0 = freeb_loop_state.nvskip0
    freeb_nestor_runtime: NestorRuntimeState | None = freeb_loop_state.nestor_runtime
    freeb_bsqvac_half_current = freeb_loop_state.bsqvac_half_current
    freeb_nestor_trace_current = freeb_loop_state.nestor_trace_current
    freeb_last_model = freeb_loop_state.last_model
    freeb_last_diagnostics: dict[str, Any] = freeb_loop_state.last_diagnostics
    freeb_plascur = freeb_loop_state.plascur

    _vmec_freeb_plascur_from_bcovar = partial(
        _runtime_vmec_freeb_plascur_from_bcovar,
        plascur_edge_from_bcovar=plascur_edge_from_bcovar,
        trig=trig,
        wout=wout_like,
        s=s,
    )

    k_preconditioner_update_interval = controller_constants.preconditioner_update_interval
    # Restart trigger factors:
    # - bad_jacobian: time_step *= 0.9
    # - bad_progress: time_step /= 1.03
    restart_badjac_factor = controller_constants.restart_badjac_factor
    restart_badprog_factor = controller_constants.restart_badprog_factor
    vmec2000_fact = controller_constants.vmec2000_fact

    print_context = _resolve_vmec2000_print_context(
        cfg=cfg,
        indata=indata,
        verbose=bool(verbose),
        vmec2000_control=bool(vmec2000_control),
        verbose_vmec2000_table=bool(verbose_vmec2000_table),
        getenv=os.getenv,
        resolve_debug_print_config=_resolve_debug_print_config,
        resolve_nstep_screen=_resolve_nstep_screen,
        emit_iter_row=_emit_scan_vmec2000_iter_row,
        should_print_row=_should_print_vmec2000_row,
        print_row=_print_scan_vmec2000_row,
    )
    nstep_screen = print_context.nstep_screen
    _print_vmec2000_iter_row = print_context.print_iter_row
    _should_print_vmec2000 = print_context.should_print

    # VMEC2000 caches 1D preconditioner/norm/tcon updates every `ns4` iterations
    # (vmec_params.f: ns4=25), reusing the cached values between refreshes.
    # This materially affects the nonlinear iteration trace because the
    # Garabedian time-step control depends on ratios of the *preconditioned*
    # residual scalars.
    precond_cache = _PreconditionerCacheState()
    cache_constraint_rcon0 = None
    cache_constraint_zcon0 = None

    _rollback_history_lists = history_lists.rollback_lists()
    iter_offset = 0

    if resume_state is not None:
        iter_offset = int(resume_state.get("iter_offset", iter_offset))
        _set_controller_state(_controller_state_from_resume_state(resume_state, controller_state))

        if "vRcc" in resume_state:
            _as_velocity = np.asarray if bool(host_update_assembly) else jnp.asarray
            velocity_blocks = _velocity_blocks_from_resume_state(
                resume_state,
                velocity_blocks,
                as_velocity=_as_velocity,
            )

        precond_cache.update_from_resume_state(resume_state)
        cache_constraint_rcon0 = resume_state.get("cache_constraint_rcon0", cache_constraint_rcon0)
        cache_constraint_zcon0 = resume_state.get("cache_constraint_zcon0", cache_constraint_zcon0)
        freeb_loop_state = _runtime_resume_free_boundary_loop_state(
            freeb_loop_state,
            resume_state=resume_state,
            free_boundary_enabled=bool(free_boundary_enabled),
        )
        freeb_ivac = freeb_loop_state.ivac
        freeb_ivacskip = freeb_loop_state.ivacskip
        freeb_nvacskip = freeb_loop_state.nvacskip
        freeb_nvskip0 = freeb_loop_state.nvskip0
        freeb_last_model = freeb_loop_state.last_model

    _apply_vmec_scale_m1_precond_rhs = partial(
        _scale_m1_precond_rhs_from_mats,
        lconm1=getattr(cfg, "lconm1", True),
        mpol=int(cfg.mpol),
        host_update_assembly=host_update_assembly,
    )

    def _refresh_preconditioner_cache(k, *, iter2: int):
        return _precond_payload_facade.refresh_preconditioner_cache_state_runtime(
            k,
            cache=precond_cache,
            cfg=cfg,
            static=static,
            iter2=int(iter2),
            env_dump_lam=_env_dump_lam,
            env_dump_lamcal=_env_dump_lamcal,
            timing_enabled=bool(timing_enabled),
            timing_stats=timing_stats,
            perf_counter=time.perf_counter,
            block_until_ready=jax.block_until_ready if has_jax() else None,
            tree_has_tracer=_tree_has_tracer,
            update_preconditioner_cache_func=_update_preconditioner_cache,
            can_reassemble_func=_can_reassemble_precond_mats,
            lambda_preconditioner_func=_lambda_preconditioner,
            rz_preconditioner_matrices_func=_rz_preconditioner_matrices_local,
            maybe_dump_lam_prec=_maybe_dump_lam_prec,
            maybe_dump_precond_mats=_maybe_dump_precond_mats,
            maybe_dump_lamcal=_maybe_dump_lamcal,
            need_bcovar_update=bool(need_bcovar_update),
            precond_cache_seeded_from_bcovar_update=bool(precond_cache_seeded_from_bcovar_update),
            precond_expected_jmax=int(precond_expected_jmax),
            precond_jmax_override=precond_jmax_override,
            preconditioner_use_precomputed_tridi=preconditioner_use_precomputed_tridi_policy,
            preconditioner_use_lax_tridi=preconditioner_use_lax_tridi_policy,
        )

    def _pop_iteration_histories() -> None:
        _pop_residual_iter_rollback_histories(_rollback_history_lists)

    _dump_evolve_trace = partial(_maybe_dump_evolve_trace_record, static=static)

    def _zero_all_velocity_blocks() -> None:
        nonlocal velocity_blocks
        velocity_blocks = _zero_all_velocity_blocks_like(velocity_blocks)

    def _zero_primary_velocity_blocks() -> None:
        nonlocal velocity_blocks
        velocity_blocks = _zero_primary_velocity_blocks_like(velocity_blocks)

    axis_reset_runtime_callbacks = _InitialAxisResetRuntimeCallbacks(
        _reset_axis_from_boundary, _host_axis_reset_update, _apply_controller_update, _controller_after_axis_reset,
        _zero_primary_velocity_blocks, lambda: axis_reset_coeffs, _print_scan_axis_guess,
    )

    def _apply_strict_step_branch(branch_result, *, after_catastrophic_restart: bool = False):
        nonlocal state, step_status, restart_reason, huge_force_restart_count, restart_path, update_rms
        nonlocal max_coeff_delta_rms, max_update_rms, freeb_controls_cached

        branch_application = _strict_step_branch_application(
            branch_result,
            max_coeff_delta_rms=float(max_coeff_delta_rms),
            max_update_rms=float(max_update_rms),
            after_catastrophic_restart=bool(after_catastrophic_restart),
        )
        branch_runtime = branch_application.runtime
        state = branch_runtime.state
        step_status = branch_runtime.step_status
        restart_reason = branch_runtime.restart_reason
        huge_force_restart_count = branch_runtime.huge_force_restart_count
        restart_path = branch_runtime.restart_path
        update_rms = branch_runtime.update_rms
        max_coeff_delta_rms = branch_runtime.max_coeff_delta_rms
        max_update_rms = branch_runtime.max_update_rms

        side_effects = branch_application.side_effects
        if side_effects.zero_all_velocity_blocks:
            _zero_all_velocity_blocks()
        if side_effects.zero_primary_velocity_blocks:
            _zero_primary_velocity_blocks()
        if side_effects.clear_freeb_controls_cached:
            freeb_controls_cached = None
        if side_effects.clear_precond_cache:
            precond_cache.clear()
        return branch_application

    # VMEC `eqsolve`: if the initial Jacobian changes sign, improve the axis
    # guess *before* the first iteration (no extra iter1). This aligns the
    # zero_m1 gating and time-control history with VMEC2000.
    axis_setup = _run_initial_axis_reset_setup(
        state=state,
        axis_reset_done=bool(axis_reset_done),
        ijacob=int(ijacob),
        state_checkpoint=state_checkpoint,
        velocities=(
            velocity_blocks.rcc,
            velocity_blocks.rss,
            velocity_blocks.zsc,
            velocity_blocks.zcs,
            velocity_blocks.lsc,
            velocity_blocks.lcs,
        ),
        res0=float(res0),
        res1=float(res1),
        prev_rz_fsq=float(prev_rz_fsq),
        vmec2000_control=bool(vmec2000_control),
        lmove_axis=bool(lmove_axis),
        verbose=bool(verbose),
        verbose_vmec2000_table=bool(verbose_vmec2000_table),
        timing_enabled=bool(timing_enabled),
        timing_stats=timing_stats,
        force_axis_reset=bool(force_axis_reset),
        axis_reset_always_3d=bool(axis_reset_always_3d),
        axis_reset_fsq_min=float(axis_reset_fsq_min),
        badjac_use_state=bool(badjac_use_state),
        static=static,
        trig=trig,
        s=s,
        zero_precond_diag=zero_precond_diag,
        zero_tcon=zero_tcon,
        compute_forces_iter_func=_compute_forces_iter,
        reset_axis_from_boundary_func=_reset_axis_from_boundary,
        zero_velocity_blocks_like_func=_zero_velocity_blocks_like,
        ptau_minmax_from_k_host_func=_ptau_minmax_from_k_host,
        vmec_half_mesh_jacobian_from_state_func=vmec_half_mesh_jacobian_from_state,
        print_axis_guess_func=_print_scan_axis_guess,
        axis_reset_coeffs_func=lambda: axis_reset_coeffs,
        env_enabled_func=_runtime_env_enabled,
        getenv_func=os.getenv,
        perf_counter_func=time.perf_counter,
        has_jax_func=has_jax,
        block_until_ready_func=jax.block_until_ready if has_jax() else None,
        jnp_module=jnp,
    )
    state = axis_setup.state
    axis_reset_done = bool(axis_setup.axis_reset_done)
    velocity_blocks = velocity_blocks._replace(
        rcc=axis_setup.velocities[0],
        rss=axis_setup.velocities[1],
        zsc=axis_setup.velocities[2],
        zcs=axis_setup.velocities[3],
        lsc=axis_setup.velocities[4],
        lcs=axis_setup.velocities[5],
    )
    _set_controller_state(
        _controller_state_after_initial_axis_setup_result(_current_controller_state(), axis_setup)
    )
    setup_axis_reset_applied = bool(axis_setup.reset_applied)
    setup_axis_force_probe = axis_setup.force_probe
    setup_axis_force_probe_reused = False
    if setup_axis_reset_applied:
        precond_cache.clear()
        cache_constraint_rcon0 = None
        cache_constraint_zcon0 = None

    # Cache os.getenv calls that would otherwise be repeated every iteration
    # in the hot loop below (saves ~9 os.getenv calls × ~2144 iters = ~19k calls).
    _env_freeb_include_edge = _runtime_env_enabled(os.getenv("VMEC_JAX_FREEB_INCLUDE_EDGE", "0"))
    _env_force_edge_residual = os.getenv("VMEC_JAX_FORCE_EDGE_RESIDUAL", "").strip().lower()
    _env_freeb_raise = _runtime_env_enabled(os.getenv("VMEC_JAX_FREEB_RAISE", ""))
    _env_debug_iter = os.getenv("VMEC_JAX_DEBUG_ITER", "").strip()
    _env_dump_lam = os.getenv("VMEC_JAX_DUMP_LAM", "")
    _env_dump_lamcal = os.getenv("VMEC_JAX_DUMP_LAMCAL", "")
    _env_dump_badjac = os.getenv("VMEC_JAX_DUMP_BADJAC", "")
    _env_dump_dir = os.getenv("VMEC_JAX_DUMP_DIR", "")

    if timing_enabled:
        timing_stats["setup_total"] = time.perf_counter() - float(_solve_wall_start)
    t_iteration_loop_start = time.perf_counter() if timing_enabled else None
    last_iter2 = 0
    for it in range(max_iter):
        iter2 = it + 1 + int(iter_offset)
        last_iter2 = iter2
        converged = False
        skip_time_control = False
        force_bcovar_update = False
        time_step_report_hold: float | None = None
        freeb_turnon_applied = False
        freeb_controls_cached: tuple[int, int, int] | None = None
        while True:
            t_iteration_prepare_start = time.perf_counter() if timing_enabled else None
            fsq_prev_before = fsq_prev
            fsq0_prev_before = fsq0_prev
            pre_restart_reason = "none"
            if time_step_report_hold is None:
                time_step_report_hold = float(time_step)
            # VMEC vacuum.f promotes ivac=0 -> 1 inside the vacuum solve.
            # Keep both values: pre-vacuum (`freeb_ivac`) for cadence/calls,
            # and post-vacuum effective (`freeb_ivac_effective`) for force/
            # residue gating in this same iteration.
            freeb_control = _runtime_resolve_free_boundary_iteration_controls(
                free_boundary_enabled=bool(free_boundary_enabled),
                controls_cached=freeb_controls_cached,
                iter2=int(iter2),
                iter1=int(iter1),
                ivac=int(freeb_ivac),
                ivacskip=int(freeb_ivacskip),
                nvacskip=int(freeb_nvacskip),
                nvskip0=int(freeb_nvskip0),
                prev_rz_fsq=float(prev_rz_fsq),
                activate_fsq=free_boundary_activate_fsq,
                iter_controls_func=_free_boundary_iter_controls_vmec,
                dump_freeb_control_trace=_dump_freeb_control_trace_record,
            )
            freeb_ivac = freeb_control.ivac
            freeb_ivacskip = freeb_control.ivacskip
            freeb_nvacskip = freeb_control.nvacskip
            freeb_controls_cached = freeb_control.controls_cached
            freeb_turnon_iter = freeb_control.turnon_iter
            freeb_ivac_effective = freeb_control.ivac_effective
            control_sample = _resolve_residual_iteration_control_sample(
                iter2=int(iter2),
                iter1=int(iter1),
                vmec2000_control=bool(vmec2000_control),
                free_boundary_enabled=bool(free_boundary_enabled),
                freeb_ivac_effective=int(freeb_ivac_effective),
                prev_rz_fsq=float(prev_rz_fsq),
                fsqz2_history=fsqz2_history,
                env_freeb_include_edge=bool(_env_freeb_include_edge),
                env_force_edge_residual=_env_force_edge_residual,
                precond_cache_valid=bool(precond_cache.valid),
                force_bcovar_update=bool(force_bcovar_update),
                preconditioner_update_interval=int(k_preconditioner_update_interval),
                ns=int(s.shape[0]),
            )
            iter_since_restart = control_sample.iter_since_restart
            zero_m1_val = control_sample.zero_m1_value
            if host_update_assembly and _jnp_zero_m1_0 is not None:
                # Use pre-cached JAX scalars to avoid jnp.asarray dispatch + dtype
                # lookup every iteration (saves 2 apply_primitive calls per iter).
                zero_m1 = _jnp_zero_m1_1 if zero_m1_val > 0.5 else _jnp_zero_m1_0
            else:
                zero_m1 = jnp.asarray(zero_m1_val, dtype=jnp.asarray(state.Rcos).dtype)
            include_edge = control_sample.include_edge
            if track_history:
                history_lists["include_edge_history"].append(int(bool(include_edge)))
            include_edge_residual = control_sample.include_edge_residual
            precond_jmax_override = control_sample.precond_jmax_override
            precond_expected_jmax = control_sample.precond_expected_jmax
            # `zero_m1` originates from host control flow, so keep the history
            # without forcing an unnecessary device synchronization.
            if track_history:
                history_lists["zero_m1_history"].append(int(zero_m1_val > 0.5))

            need_bcovar_update = control_sample.need_bcovar_update
            precond_cache_seeded_from_bcovar_update = False
            precond_refresh_seed_time_in_residual_metrics = 0.0
            force_bcovar_update = False
            history_lists["bcovar_update_history"].append(int(bool(need_bcovar_update)))

            use_cached_precond = control_sample.use_cached_precond
            constraint_channels = _constraint_preconditioner_channels(
                use_cached_precond=bool(use_cached_precond),
                cached_precond_diag=precond_cache.precond_diag,
                cached_tcon=precond_cache.tcon,
                zero_precond_diag=zero_precond_diag,
                zero_tcon=zero_tcon,
                host_update_assembly=bool(host_update_assembly),
                jnp_true_bool=_jnp_true_bool,
                jnp_false_bool=_jnp_false_bool,
                jnp_module=jnp,
            )
            constraint_precond_diag = constraint_channels.precond_diag
            constraint_tcon_override = constraint_channels.tcon
            constraint_precond_active = constraint_channels.precond_active
            constraint_tcon_active = constraint_channels.tcon_active

            # Free-boundary WP2 scaffold: run/update the NESTOR-like external
            # vacuum solve and couple bsqvac on the edge slice into bcovar.
            freeb_plascur_for_bsqvac = float(freeb_plascur)
            freeb_coupling = _runtime_resolve_free_boundary_coupling_runtime(
                free_boundary_enabled=bool(free_boundary_enabled),
                freeb_couple_edge=bool(freeb_couple_edge),
                state=state,
                static=static,
                freeb_ivac=int(freeb_ivac),
                freeb_ivacskip=int(freeb_ivacskip),
                iter2=int(iter2),
                freeb_nestor_runtime=freeb_nestor_runtime,
                freeb_plascur=float(freeb_plascur),
                external_field_provider_kind=external_field_provider_kind,
                external_field_provider_static=external_field_provider_static,
                external_field_provider_params=external_field_provider_params,
                collect_trace_arrays=bool(adjoint_trace and adjoint_trace_mode in {"full", "branch"}),
                freeb_turnon_iter=bool(freeb_turnon_iter),
                freeb_ivac_effective=int(freeb_ivac_effective),
                freeb_nvacskip=int(freeb_nvacskip),
                controls_cached=freeb_controls_cached,
                last_model=freeb_last_model,
                last_diagnostics=freeb_last_diagnostics,
                env_freeb_raise=bool(_env_freeb_raise),
                nestor_external_only_step_func=nestor_external_only_step,
                edge_bsqvac_from_nestor_func=_edge_bsqvac_from_nestor,
                nestor_iteration_coupling_func=_free_boundary_nestor_iteration_coupling,
                trial_bsqvac_half_func=_runtime_freeb_trial_bsqvac_half,
                source_history_lists=history_lists.freeb_source_history_lists(),
                trial_history_lists=history_lists.freeb_trial_history_lists(),
            )
            freeb_bsqvac_half_current = freeb_coupling.bsqvac_half_current
            freeb_nestor_runtime = freeb_coupling.nestor_runtime
            freeb_nestor_trace_current = freeb_coupling.trace_arrays
            freeb_reused = freeb_coupling.reused
            freeb_solve_time = freeb_coupling.solve_time
            freeb_sample_time = freeb_coupling.sample_time
            freeb_last_model = freeb_coupling.last_model
            freeb_last_diagnostics = freeb_coupling.last_diagnostics
            freeb_ivac = freeb_coupling.ivac
            freeb_ivac_effective = freeb_coupling.ivac_effective
            freeb_controls_cached = freeb_coupling.controls_cached
            _freeb_bsqvac_half_for_trial_state = freeb_coupling.trial_bsqvac_half_for_state

            _trial_residual_total = partial(
                _runtime_trial_residual_total,
                compute_forces_iter_func=_compute_forces_iter,
                include_edge=bool(include_edge),
                constraint_precond_diag=constraint_precond_diag,
                constraint_tcon=constraint_tcon_override,
                constraint_precond_active=constraint_precond_active,
                constraint_tcon_active=constraint_tcon_active,
                iter2=int(iter2),
                timing_detail_enabled=bool(timing_detail_enabled),
                perf_counter=time.perf_counter,
                record_compute_force_timing=_record_compute_force_timing,
                residual_fsq_from_norms_func=_residual_fsq_from_norms,
                numpy_module=np,
            )

            _candidate_state_from_deltas = partial(
                _candidate_state_from_deltas_helper,
                state=state,
                static=static,
                edge_Rcos=edge_Rcos,
                edge_Rsin=edge_Rsin,
                edge_Zcos=edge_Zcos,
                edge_Zsin=edge_Zsin,
                free_boundary_enabled=bool(free_boundary_enabled),
                idx00=idx00,
                precomputed_axis_mask=_precomputed_axis_mask_np,
                enforce_fixed_boundary_and_axis=_enforce_fixed_boundary_and_axis,
                enforce_fixed_boundary_and_axis_np=_enforce_fixed_boundary_and_axis_np,
                apply_vmec_lambda_axis_rules=_apply_vmec_lambda_axis_rules,
            )
            _delta_tuple_from_blocks = partial(
                _delta_tuple_from_blocks_helper,
                lasym=bool(cfg.lasym),
                zeros_dR_np=_zeros_dR_np,
            )
            _candidate_state_from_delta_tuple = partial(
                _candidate_state_from_delta_tuple_helper,
                scale=1.0,
                use_numpy_arrays=False,
                use_numpy_enforce=False,
                candidate_from_deltas=_candidate_state_from_deltas,
            )

            constraint_rcon0_current = None
            constraint_zcon0_current = None
            if (
                bool(vmec2000_control)
                and bool(free_boundary_enabled)
                and (cache_constraint_rcon0 is not None)
                and (cache_constraint_zcon0 is not None)
            ):
                # VMEC keeps rcon0/zcon0 as persistent baselines; once free-
                # boundary control is active, damp them by 0.9 on reuse steps.
                # The first turn-on iteration keeps the pre-turn-on baseline.
                if _free_boundary_should_damp_constraint_baseline(
                    freeb_ivac=int(freeb_ivac),
                    freeb_turnon_iter=bool(freeb_turnon_iter),
                    lthreed=bool(cfg.lthreed),
                ):
                    cache_constraint_rcon0 = 0.9 * jnp.asarray(cache_constraint_rcon0)
                    cache_constraint_zcon0 = 0.9 * jnp.asarray(cache_constraint_zcon0)
                constraint_rcon0_current = cache_constraint_rcon0
                constraint_zcon0_current = cache_constraint_zcon0

            if (
                profile_active
                and (not profile_started)
                and (profile_start_iter is not None)
                and (iter2 == profile_start_iter)
            ):
                if has_jax():
                    try:
                        Path(profile_dir).mkdir(parents=True, exist_ok=True)
                        jax.profiler.start_trace(profile_dir, create_perfetto_trace=profile_perfetto)
                        profile_started = True
                    except Exception:
                        profile_active = False

            _record_timing("iteration_prepare", t_iteration_prepare_start)
            t_compute_start = time.perf_counter() if timing_enabled else None
            iter_dump_idx = _iter_idx_for_dump(iter2)
            reuse_setup_axis_force = (
                int(iter2) == 1
                and setup_axis_force_probe is not None
                and not bool(setup_axis_reset_applied)
                and not bool(free_boundary_enabled)
                and not bool(include_edge)
                and not bool(include_edge_residual)
                and freeb_bsqvac_half_current is None
                and constraint_rcon0_current is None
                and constraint_zcon0_current is None
                and not bool(use_cached_precond)
                and float(zero_m1_val) > 0.5
                and iter_dump_idx is None
            )
            if reuse_setup_axis_force:
                k, frzl, gcr2, gcz2, gcl2, rz_scale, l_scale, norms_current = setup_axis_force_probe
                setup_axis_force_probe_reused = True
                if timing_enabled:
                    timing_stats["compute_forces_main_reuse_count"] = (
                        int(timing_stats.get("compute_forces_main_reuse_count", 0)) + 1
                    )
            else:
                k, frzl, gcr2, gcz2, gcl2, rz_scale, l_scale, norms_current = _compute_forces_iter(
                    state,
                    include_edge=bool(include_edge),
                    include_edge_residual=bool(include_edge_residual),
                    zero_m1=zero_m1,
                    freeb_bsqvac_half=freeb_bsqvac_half_current,
                    constraint_rcon0=constraint_rcon0_current,
                    constraint_zcon0=constraint_zcon0_current,
                    constraint_precond_diag=constraint_precond_diag,
                    constraint_tcon=constraint_tcon_override,
                    constraint_precond_active=constraint_precond_active,
                    constraint_tcon_active=constraint_tcon_active,
                    iter_idx=iter_dump_idx,
                    iter2=iter2,
                )
            if bool(free_boundary_enabled):
                freeb_plascur = _vmec_freeb_plascur_from_bcovar(k.bc, freeb_plascur)
                try:
                    pr1_axis = np.asarray(k.pr1_even, dtype=float)
                    pz1_axis = np.asarray(k.pz1_even, dtype=float)
                    if pr1_axis.ndim >= 3 and pz1_axis.ndim >= 3:
                        _dump_freeb_axis_trace_record(
                            iter2=int(iter2),
                            axis_r=np.asarray(pr1_axis[0, 0, :], dtype=float).reshape(-1),
                            axis_z=np.asarray(pz1_axis[0, 0, :], dtype=float).reshape(-1),
                        )
                except Exception:
                    pass
                if getattr(k, "constraint_rcon0", None) is not None:
                    if cache_constraint_rcon0 is None or cache_constraint_zcon0 is None:
                        # Initialize persistent VMEC-style constraint baseline.
                        cache_constraint_rcon0 = jnp.asarray(k.constraint_rcon0)
                        cache_constraint_zcon0 = jnp.asarray(k.constraint_zcon0)
            if timing_enabled and not reuse_setup_axis_force:
                _record_compute_force_timing("main", t_compute_start, gcr2)
            t_residual_metrics_start = time.perf_counter() if timing_enabled else None
            norms_used = _select_residual_norms_for_iteration(
                vmec2000_control=bool(vmec2000_control),
                precond_cache_valid=bool(precond_cache.valid),
                need_bcovar_update=bool(need_bcovar_update),
                cached_norms=precond_cache.norms,
                current_norms=norms_current,
            )
            use_host_residual_metrics = (
                bool(startup_policy.host_residual_metrics_on_accelerator)
                and (not bool(host_update_assembly))
                and (jax.default_backend() != "cpu")
                and (not _tree_has_tracer((gcr2, gcz2, gcl2, norms_used)))
            )
            physical_metrics = _physical_residual_metric_channels(
                gcr2=gcr2,
                gcz2=gcz2,
                gcl2=gcl2,
                norms_used=norms_used,
                host_update_assembly=bool(host_update_assembly),
                use_host_residual_metrics=bool(use_host_residual_metrics),
                device_get_floats=_device_get_floats,
            )
            norms_used = physical_metrics.norms_used
            fsqr = physical_metrics.fsqr
            fsqz = physical_metrics.fsqz
            fsql = physical_metrics.fsql
            debug_iter_env = _env_debug_iter
            _maybe_print_nonscan_state_debug(
                debug_iter_env=debug_iter_env,
                iter2=int(iter2),
                state=state,
                state_checkpoint=state_checkpoint,
                gcr2=gcr2,
                gcz2=gcz2,
                gcl2=gcl2,
                norms_used=norms_used,
                print_fn=print,
            )
            if bool(vmec2000_control) and bool(precond_cache.valid) and (not bool(need_bcovar_update)):
                rz_scale = precond_cache.rz_scale
                l_scale = precond_cache.l_scale
            preconditioner_cache_update_trace = False
            if bool(vmec2000_control) and bool(need_bcovar_update):
                seed_result = _precond_payload_facade.seed_preconditioner_cache_from_bcovar_update(
                    cache=precond_cache,
                    k=k,
                    state=state,
                    trig=trig,
                    s=s,
                    cfg=cfg,
                    norms_used=norms_used,
                    rz_scale=rz_scale,
                    l_scale=l_scale,
                    constraint_tcon0=constraint_tcon0,
                    zero_tcon=zero_tcon,
                    host_update_assembly=bool(host_update_assembly),
                    timing_enabled=bool(timing_enabled),
                    timing_stats=timing_stats,
                    perf_counter=time.perf_counter,
                    tree_has_tracer=_tree_has_tracer,
                    rz_norm_np=_rz_norm_np,
                    rz_norm_func=_rz_norm,
                    lambda_preconditioner_func=_lambda_preconditioner,
                    rz_preconditioner_matrices_func=_rz_preconditioner_matrices_local,
                    precond_jmax_override=precond_jmax_override,
                    preconditioner_use_precomputed_tridi=preconditioner_use_precomputed_tridi_policy,
                    preconditioner_use_lax_tridi=preconditioner_use_lax_tridi_policy,
                    jnp_module=jnp,
                )
                preconditioner_cache_update_trace = seed_result.cache_update_trace
                precond_cache_seeded_from_bcovar_update = seed_result.seeded_from_bcovar_update
                precond_refresh_seed_time_in_residual_metrics += seed_result.seed_time_in_residual_metrics
            if host_update_assembly or use_host_residual_metrics:
                # fsqr/fsqz/fsql are already Python floats from the NumPy path above.
                fsqr_f, fsqz_f, fsql_f = fsqr, fsqz, fsql
            else:
                fsqr_f, fsqz_f, fsql_f = _device_get_floats(fsqr, fsqz, fsql)
            force_state_pre_current = state
            if bool(free_boundary_enabled) and bool(freeb_turnon_iter) and (not bool(freeb_turnon_applied)):
                # VMEC restarts funct3d immediately after the first
                # free-boundary turn-on solve, keeping the cached ns4 blocks
                # intact across the same-iteration retry.
                turnon_update = _host_free_boundary_turnon_restart_update(
                    state_checkpoint=state_checkpoint,
                    time_step=float(time_step),
                    iter2=int(iter2),
                    iter1=int(iter1),
                    ijacob=int(ijacob),
                    k_ndamp=int(k_ndamp),
                    reset_iter1=_free_boundary_turnon_resets_iter1_immediately(
                        lthreed=bool(cfg.lthreed),
                        lasym=bool(cfg.lasym),
                    ),
                )
                state = turnon_update.state
                _zero_all_velocity_blocks()
                time_step_report_hold = turnon_update.time_step_report_hold
                _apply_controller_update(_controller_state_after_free_boundary_turnon_restart_update, turnon_update)
                freeb_turnon_applied = True
            fsq0_curr = fsqr_f + fsqz_f + fsql_f
            prev_rz_fsq_before = prev_rz_fsq
            prev_rz_fsq = _free_boundary_prev_rz_fsq_next(
                prev_fsq_before=prev_rz_fsq_before,
                fsq_rz_curr=fsqr_f + fsqz_f,
                turnon_restart=bool(free_boundary_enabled) and bool(freeb_turnon_iter) and bool(freeb_turnon_applied),
                preserve_turnon_restart=bool(free_boundary_enabled) and bool(cfg.lthreed),
            )

            # VMEC printout uses r00 = r1(1,0): axis R at theta=0, zeta=0,
            # evaluated in real space after scalxc (see funct3d.f).
            # For parity diagnostics, sample these scalars on VMEC's screen cadence.
            sample_vmec = bool(vmec2000_control) and _vmec2000_cadence_selected(
                iter_idx=int(iter2),
                max_iter=int(max_iter),
                nstep_screen=nstep_screen,
            )
            need_scalar = bool(sample_vmec) or (bool(verbose) and (not bool(vmec2000_control)))
            vmec_scalars = _sample_vmec_iteration_scalars(
                need_scalar=bool(need_scalar),
                k=k,
                state=state,
                norms_current=norms_current,
                m0_mask=m0_mask,
                lasym=bool(cfg.lasym),
                host_update_assembly=bool(host_update_assembly),
                vmec2000_control=bool(vmec2000_control),
                gamma=float(gamma),
                twopi=float(TWOPI),
                previous_r00=float(r00_last),
                previous_z00=float(z00_last),
                previous_wb=float(wb_last),
                previous_wp=float(wp_last),
                tree_has_tracer=_tree_has_tracer,
                device_get_floats=_device_get_floats,
                jnp_module=jnp,
            )
            r00_last = vmec_scalars.r00
            z00_last = vmec_scalars.z00
            wb_last = vmec_scalars.wb
            wp_last = vmec_scalars.wp
            w_vmec_last = vmec_scalars.w_vmec
            history_lists.append_physical_sample(
                track_history=bool(track_history),
                fsq=(fsq0_curr, fsqr_f, fsqz_f, fsql_f),
                vmec_scalars=vmec_scalars,
            )

            _print_compact_physical_residual_status(
                verbose=bool(verbose),
                vmec2000_control=bool(vmec2000_control),
                verbose_vmec2000_table=bool(verbose_vmec2000_table),
                iter_idx=int(it),
                fsqr=fsqr_f,
                fsqz=fsqz_f,
                fsql=fsql_f,
                include_edge=bool(include_edge),
            )
            # Defer convergence exit until after preconditioned diagnostics are
            # computed for this iteration, so fsqr1/fsqz1/fsql1 histories and
            # VMEC-style tables remain length-aligned.
            converged_physical = _residual_convergence_flags(
                fsqr=fsqr_f,
                fsqz=fsqz_f,
                fsql=fsql_f,
                ftol=ftol,
                fsq_total_target=fsq_total_target,
            )[2]
            accepted_control_ptau_payload: tuple[Any, Any, Any] | None = None
            fuse_accepted_control_ptau = (
                bool(free_boundary_enabled)
                and bool(direct_free_boundary_provider)
                and (not bool(converged_physical))
                and (bool(reference_mode) or bool(vmec2000_control))
                and (not bool(host_update_assembly))
                and (not bool(badjac_use_state))
                and (not bool(startup_policy.dump_ptau_state))
                and os.getenv("VMEC_JAX_DUMP_PTAU", "") in ("", "0")
                and jax.default_backend() != "cpu"
            )
            accepted_control_ptau_arrays = _accepted_control_ptau_arrays(k) if fuse_accepted_control_ptau else None

            # Precondition forces.
            if timing_enabled and t_residual_metrics_start is not None:
                residual_metrics_dt = (
                    time.perf_counter()
                    - float(t_residual_metrics_start)
                    - float(precond_refresh_seed_time_in_residual_metrics)
                )
                timing_stats["iteration_residual_metrics"] += max(0.0, residual_metrics_dt)
            t_precond_start = time.perf_counter() if timing_enabled else None
            use_fused_precond_output_scaling = (not bool(host_update_assembly)) and jax.default_backend() != "cpu"
            preconditioner_result = _apply_residual_iteration_preconditioner(
                use_vmec2000_preconditioner=(bool(vmec2000_control) and bool(cfg.lthreed)) or (not bool(cfg.lthreed)),
                frzl=frzl,
                k=k,
                state=state,
                iter2=int(iter2),
                cfg=cfg,
                static=static,
                s=s,
                delta_s=delta_s,
                w_mode_mn=w_mode_mn,
                w_mode_mn_np=w_mode_mn_np,
                lambda_update_scale=float(lambda_update_scale),
                lambda_update_scale_j=lambda_update_scale_j,
                vmec2000_control=bool(vmec2000_control),
                precond_cache=precond_cache,
                need_bcovar_update=bool(need_bcovar_update),
                host_update_assembly=bool(host_update_assembly),
                use_fused_precond_output_scaling=bool(use_fused_precond_output_scaling),
                adjoint_trace=bool(adjoint_trace),
                adjoint_trace_mode=adjoint_trace_mode,
                accepted_control_ptau_arrays=accepted_control_ptau_arrays,
                ptau_pshalf_jax=_ptau_context.pshalf_jax,
                ptau_ohs_jax=_ptau_context.ohs_jax,
                preconditioner_use_precomputed_tridi_policy=preconditioner_use_precomputed_tridi_policy,
                preconditioner_use_lax_tridi_policy=preconditioner_use_lax_tridi_policy,
                timing_enabled=bool(timing_enabled),
                timing_detail_enabled=bool(timing_detail_enabled),
                timing_stats=timing_stats,
                t_precond_start=t_precond_start,
                perf_counter=time.perf_counter,
                record_timing=_record_timing,
                has_jax_func=has_jax,
                block_until_ready=jax.block_until_ready if has_jax() else None,
                tomnsps_type=TomnspsRZL,
                refresh_preconditioner_cache_func=_refresh_preconditioner_cache,
                scale_m1_precond_rhs_func=_apply_vmec_scale_m1_precond_rhs,
                rz_preconditioner_apply_func=_rz_preconditioner_apply_local,
                rz_norm_func=_rz_norm,
                apply_vmec2000_preconditioner_runtime_func=_precond_payload_facade.apply_vmec2000_preconditioner_runtime,
                radial_preconditioner_output_blocks_jax_func=_radial_preconditioner_output_blocks_jax,
                apply_radial_tridi_func=_apply_radial_tridi,
                mode_weight_force_blocks_np_func=_mode_weight_force_blocks_np,
                mode_weight_force_blocks_jax_func=_mode_weight_force_blocks_jax,
                zeros_coeff_np=_zeros_coeff_np,
                rz_scale=rz_scale,
                l_scale=l_scale,
                precond_radial_alpha=precond_radial_alpha,
                precond_lambda_alpha=precond_lambda_alpha,
            )
            lam_prec = preconditioner_result.lam_prec
            mats = preconditioner_result.mats
            jmax = preconditioner_result.jmax
            preconditioner_cache_update_trace = preconditioner_result.cache_update_trace
            preconditioned_blocks = preconditioner_result.preconditioned_blocks
            update_force_blocks = preconditioner_result.update_force_blocks
            frzl_pre = preconditioner_result.frzl_pre
            frzl_rz = preconditioner_result.frzl_rz
            frzl_lam_pre = preconditioner_result.frzl_lam_pre
            accepted_control_ptau_payload = preconditioner_result.accepted_control_ptau_payload
            preconditioner_outputs_scaled = preconditioner_result.outputs_scaled
            preconditioner_fsq1_ready = preconditioner_result.fsq1_ready
            if preconditioner_fsq1_ready:
                gcr2_p = preconditioner_result.gcr2_p
                gcz2_p = preconditioner_result.gcz2_p
                gcl2_p = preconditioner_result.gcl2_p
                fsqr1_safe = preconditioner_result.fsqr1_safe
                fsqz1_safe = preconditioner_result.fsqz1_safe
                fsql1_safe = preconditioner_result.fsql1_safe
                fsq1_safe = preconditioner_result.fsq1_safe
                fsqr1 = fsqr1_safe
                fsqz1 = fsqz1_safe
                fsql1 = fsql1_safe
            if frzl_lam_pre is not None:
                _maybe_dump_lam_gcl(
                    frzl_pre=frzl_lam_pre,
                    frzl_post=frzl_pre,
                    static=static,
                    iter_idx=int(iter2),
                    delta_s=delta_s,
                )
            _maybe_dump_gc(frzl=frzl_pre, static=static, iter_idx=int(iter2), label="precond")
            t_iteration_control_start = time.perf_counter() if timing_enabled else None
            t_iteration_control_fsq1_start = time.perf_counter() if timing_enabled else None

            if startup_policy.auto_flip_force and it == 0:
                # Choose force direction by a tiny trial step on the VMEC residual
                # (fsqr+fsqz+fsql), not magnetic energy. Energy monotonicity is not a
                # reliable proxy for VMEC's preconditioned convergence metrics.
                w_curr = float(fsqr_f + fsqz_f + fsql_f)
                # Use a probe step that is large enough to be numerically decisive,
                # but still small relative to typical pseudo-time updates.
                dt_probe = min(1e-2, 0.1 * float(time_step))
                dR_dir = dt_probe * _mn_cos_to_signed_physical(update_force_blocks.frcc, update_force_blocks.frss)
                dZ_dir = dt_probe * _mn_sin_to_signed_physical(update_force_blocks.fzsc, update_force_blocks.fzcs)
                dL_dir = dt_probe * _mn_sin_to_signed_physical_lambda(update_force_blocks.flsc, update_force_blocks.flcs)
                if bool(cfg.lasym):
                    dR_sin_dir = dt_probe * _mn_sin_to_signed_physical(update_force_blocks.frsc, update_force_blocks.frcs)
                    dZ_cos_dir = dt_probe * _mn_cos_to_signed_physical(update_force_blocks.fzcc, update_force_blocks.fzss)
                    dL_cos_dir = dt_probe * _mn_cos_to_signed_physical_lambda(update_force_blocks.flcc, update_force_blocks.flss)
                else:
                    dR_sin_dir = jnp.zeros_like(dR_dir)
                    dZ_cos_dir = jnp.zeros_like(dR_dir)
                    dL_cos_dir = jnp.zeros_like(dR_dir)

                def _trial(sign: float) -> float:
                    st_try = VMECState(
                        layout=state.layout,
                        Rcos=jnp.asarray(state.Rcos) + sign * dR_dir,
                        Rsin=jnp.asarray(state.Rsin) + sign * dR_sin_dir,
                        Zcos=jnp.asarray(state.Zcos) + sign * dZ_cos_dir,
                        Zsin=jnp.asarray(state.Zsin) + sign * dZ_dir,
                        Lcos=jnp.asarray(state.Lcos) + sign * dL_cos_dir,
                        Lsin=jnp.asarray(state.Lsin) + sign * dL_dir,
                    )
                    return _trial_residual_total(
                        st_try,
                        _freeb_bsqvac_half_for_trial_state(st_try),
                        zero_m1_value=zero_m1,
                        timing_label="auto_flip",
                    )

                w_pos = _trial(+1.0)
                w_neg = _trial(-1.0)
                if np.isfinite(w_neg) and np.isfinite(w_pos) and (w_neg < w_pos):
                    flip_sign = -1.0
                    if verbose and not (bool(vmec2000_control) and bool(verbose_vmec2000_table)):
                        print(
                            "[solve_fixed_boundary_residual_iter] flipping force sign "
                            f"(w_curr={w_curr:.3e} w_pos={w_pos:.3e} w_neg={w_neg:.3e})"
                        )

            # Damping for the fixed-point update.
            t_fsq1_precond_norm_start = time.perf_counter() if timing_enabled else None

            def _dump_lam_fsql1_if_requested(fsql1_post):
                gcr2_raw, gcz2_raw, gcl2_raw = vmec_gcx2_from_tomnsps(
                    frzl=frzl,
                    lconm1=bool(getattr(static.cfg, "lconm1", True)),
                    apply_m1_constraints=False,
                    include_edge=True,
                    apply_scalxc=False,
                    s=s,
                )
                fsql1_pre = gcl2_raw * delta_s
                _maybe_dump_lam_fsql1(
                    fsql1_pre=fsql1_pre,
                    fsql1_post=fsql1_post,
                    static=static,
                    iter_idx=int(iter2),
                )

            scalar_result = _resolve_preconditioned_residual_scalars(
                preconditioner_payload=preconditioner_result,
                frzl_pre=frzl_pre,
                state=state,
                static=static,
                k=k,
                s=s,
                delta_s=delta_s,
                host_update_assembly=bool(host_update_assembly),
                host_fsq1_norms_on_accelerator=bool(startup_policy.host_fsq1_norms_on_accelerator),
                backend_name=jax.default_backend(),
                vmec2000_control=bool(vmec2000_control),
                precond_cache=precond_cache,
                need_bcovar_update=bool(need_bcovar_update),
                converged_physical=bool(converged_physical),
                reference_mode=bool(reference_mode),
                badjac_use_state=bool(badjac_use_state),
                dump_ptau_state=bool(startup_policy.dump_ptau_state),
                dump_ptau_env=os.getenv("VMEC_JAX_DUMP_PTAU", ""),
                timing_enabled=bool(timing_enabled),
                timing_stats=timing_stats,
                t_fsq1_precond_norm_start=t_fsq1_precond_norm_start,
                t_iteration_control_fsq1_start=t_iteration_control_fsq1_start,
                perf_counter=time.perf_counter,
                record_timing=_record_timing,
                tree_has_tracer_func=_tree_has_tracer,
                tomnsps_to_numpy_host_func=_tomnsps_to_numpy_host,
                vmec_gcx2_from_tomnsps_np_func=vmec_gcx2_from_tomnsps_np,
                vmec_gcx2_from_tomnsps_func=vmec_gcx2_from_tomnsps,
                host_preconditioned_residual_scalar_channels_func=(
                    _precond_payload_facade.host_preconditioned_residual_scalar_channels
                ),
                jax_preconditioned_residual_scalar_channels_func=(
                    _precond_payload_facade.jax_preconditioned_residual_scalar_channels
                ),
                materialize_accepted_control_payload_func=_precond_payload_facade.materialize_accepted_control_payload,
                numpy_module=np,
                jnp_module=jnp,
                jax_module=jax,
                rz_norm_np_func=_rz_norm_np,
                rz_norm_func=_rz_norm,
                lambda_preconditioned_full_norm_func=_lambda_preconditioned_full_norm,
                finite_float_or_zero_func=_finite_float_or_zero,
                cached_or_current_f_norm1_jax_func=_cached_or_current_f_norm1_jax,
                dump_lam_fsql1_func=(
                    _dump_lam_fsql1_if_requested if _env_dump_lam not in ("", "0") and frzl_lam_pre is None else None
                ),
                device_get_floats_func=_device_get_floats,
                accepted_control_ptau_host_from_payload_func=_accepted_control_ptau_host_from_payload,
                scan_math_kernel_arrays_from_k_func=_scan_math_kernel_arrays_from_k,
                accepted_control_payload_jit_func=_accepted_control_payload_jit,
                ptau_pshalf_jax=_ptau_context.pshalf_jax,
                ptau_ohs_jax=_ptau_context.ohs_jax,
            )
            use_host_fsq1_norms = scalar_result.use_host_fsq1_norms
            gcr2_p = scalar_result.gcr2_p
            gcz2_p = scalar_result.gcz2_p
            gcl2_p = scalar_result.gcl2_p
            rz_norm = scalar_result.rz_norm
            f_norm1 = scalar_result.f_norm1
            fsqr1 = scalar_result.fsqr1
            fsqz1 = scalar_result.fsqz1
            fsql1 = scalar_result.fsql1
            fsqr1_safe = scalar_result.fsqr1_safe
            fsqz1_safe = scalar_result.fsqz1_safe
            fsql1_safe = scalar_result.fsql1_safe
            fsq1 = scalar_result.fsq1
            accepted_control_ptau_host = scalar_result.accepted_control_ptau_host
            control_payload_used = scalar_result.control_payload_used
            precond_diag_host: tuple[float, float, float] | None = None

            def _precond_diag_floats() -> tuple[float, float, float]:
                nonlocal precond_diag_host
                if precond_diag_host is None:
                    precond_diag_host = _device_get_floats(fsqr1_safe, fsqz1_safe, fsql1_safe)
                return precond_diag_host

            def _append_current_zero_update_history(
                *,
                restart_path: str,
                step_status: str,
                restart_reason: str,
                pre_restart_reason: str,
                time_step_value: float,
            ) -> bool:
                return history_lists.append_zero_update(
                    track_history=bool(track_history),
                    restart_path=restart_path,
                    step_status=step_status,
                    restart_reason=restart_reason,
                    pre_restart_reason=pre_restart_reason,
                    time_step_value=time_step_value,
                    fsqr=fsqr_f,
                    fsqz=fsqz_f,
                    fsql=fsql_f,
                    res0=res0,
                    res1=res1,
                    fsq_prev=fsq_prev,
                    bad_growth_streak=bad_growth_streak,
                    iter1=iter1,
                    iter2=iter2,
                    free_boundary_enabled=bool(free_boundary_enabled),
                    freeb_ivac=freeb_ivac,
                    freeb_ivacskip=freeb_ivacskip,
                )

            def _apply_restart_branch_result(branch, controller_update_func, *, time_step_value: float) -> None:
                nonlocal force_bcovar_update, freeb_controls_cached, prev_rz_fsq, restart_reason
                nonlocal skip_time_control, state, step_status

                state = branch.state
                _zero_all_velocity_blocks()
                _apply_controller_update(controller_update_func, branch.update)
                step_status = branch.step_status
                restart_reason = branch.restart_reason
                if branch.clear_freeb_controls:
                    freeb_controls_cached = None
                if branch.clear_preconditioner_cache:
                    precond_cache.clear()
                if branch.force_bcovar_update:
                    force_bcovar_update = True
                _append_current_zero_update_history(
                    restart_path=branch.restart_path,
                    step_status=step_status,
                    restart_reason=restart_reason,
                    pre_restart_reason=branch.pre_restart_reason,
                    time_step_value=time_step_value,
                )
                if branch.pop_iteration_history:
                    _pop_iteration_histories()
                prev_rz_fsq = branch.prev_rz_fsq
                skip_time_control = branch.skip_time_control

            history_lists.append_preconditioned(
                track_history=bool(track_history),
                rz_norm=rz_norm,
                f_norm1=f_norm1,
                gcr2_p=gcr2_p,
                gcz2_p=gcz2_p,
                gcl2_p=gcl2_p,
                fsq1=fsq1,
                fsqr1_safe=fsqr1_safe,
                fsqz1_safe=fsqz1_safe,
                fsql1_safe=fsql1_safe,
            )

            if converged_physical:
                # Keep per-iteration history channels length-aligned with
                # fsqr/fsqz/fsql when convergence happens before the update
                # block. VMEC's table still reports DELT on this row.
                _append_current_zero_update_history(
                    restart_path="converged",
                    step_status="converged",
                    restart_reason="none",
                    pre_restart_reason="none",
                    time_step_value=time_step,
                )
                _print_compact_converged_status(
                    verbose=bool(verbose),
                    vmec2000_control=bool(vmec2000_control),
                    verbose_vmec2000_table=bool(verbose_vmec2000_table),
                    fsqr=fsqr_f,
                    fsqz=fsqz_f,
                    fsql=fsql_f,
                    target=float(fsq_total_target) if fsq_total_target is not None else float(ftol),
                )
                if _record_timing("iteration_control", t_iteration_control_start):
                    t_iteration_control_start = None
                _print_residual_iteration_update_status(
                    verbose=bool(verbose),
                    vmec2000_control=bool(vmec2000_control),
                    verbose_vmec2000_table=bool(verbose_vmec2000_table),
                    should_print_vmec2000=_should_print_vmec2000,
                    print_vmec2000_iter_row=_print_vmec2000_iter_row,
                    precond_diag_floats=_precond_diag_floats,
                    iter_idx=int(iter2),
                    max_iter=int(max_iter),
                    compact_iter_idx=int(it),
                    fsqr=fsqr_f,
                    fsqz=fsqz_f,
                    fsql=fsql_f,
                    dt_eff=0.0,
                    update_rms=0.0,
                    time_step=float(time_step),
                    r00=float(r00_last),
                    z00=float(z00_last),
                    w_mhd=float(w_vmec_last),
                    step_status="converged",
                    force_vmec2000_row=True,
                    compact_status=False,
                )
                converged = True
                break

            # Jacobian sign-change check (VMEC jacobian.f sets irst=2).
            t_iteration_control_badjac_start = time.perf_counter() if timing_enabled else None
            bad_jacobian = False
            if bool(reference_mode) or bool(vmec2000_control):
                from vmec_jax.vmec_numpy_forces import _numpy_module_patch as _hot_numpy_patch

                badjac_selection = _resolve_bad_jacobian_tau_selection(
                    reference_mode=bool(reference_mode),
                    vmec2000_control=bool(vmec2000_control),
                    accepted_control_ptau_host=accepted_control_ptau_host,
                    k=k,
                    state=state,
                    iter_idx=int(iter2),
                    startup_policy=startup_policy,
                    badjac_use_state=bool(badjac_use_state),
                    ptau_tol=ptau_tol,
                    static=static,
                    trig=trig,
                    s=s,
                    host_update_assembly=bool(host_update_assembly),
                    timing_enabled=bool(timing_enabled),
                    perf_counter=time.perf_counter,
                    record_timing=_record_timing,
                    ptau_minmax_from_k_host_func=_ptau_minmax_from_k_host,
                    device_get_floats_func=_device_get_floats,
                    should_probe_bad_jacobian_state_func=_residual_iter_config.should_probe_bad_jacobian_state,
                    bad_jacobian_requires_state_jacobian_func=(
                        _residual_iter_policy.bad_jacobian_requires_state_jacobian
                    ),
                    bad_jacobian_tau_decision_func=_residual_iter_policy.bad_jacobian_tau_decision,
                    select_bad_jacobian_decision_func=_residual_iter_policy.select_bad_jacobian_decision,
                    state_tau_minmax_from_vmec_state_func=_state_tau_minmax_from_vmec_state,
                    tree_has_tracer_func=_tree_has_tracer,
                    jacobian_from_state_func=vmec_half_mesh_jacobian_from_state,
                    jnp_module=jnp,
                    numpy_patch_context=_hot_numpy_patch,
                )
                bad_jacobian = bool(badjac_selection.bad_jacobian)
                min_tau = float(badjac_selection.min_tau)
                max_tau = float(badjac_selection.max_tau)
                min_tau_ptau = badjac_selection.min_tau_ptau
                max_tau_ptau = badjac_selection.max_tau_ptau
                min_tau_state = badjac_selection.min_tau_state
                max_tau_state = badjac_selection.max_tau_state
                bad_jacobian_ptau = badjac_selection.bad_jacobian_ptau
                bad_jacobian_state = bool(badjac_selection.bad_jacobian_state)

                _maybe_dump_ptau(
                    iter_idx=int(iter2),
                    ptau_min=float(min_tau_ptau if min_tau_ptau is not None else float("nan")),
                    ptau_max=float(max_tau_ptau if max_tau_ptau is not None else float("nan")),
                    tau_min_state=min_tau_state if np.isfinite(min_tau_state) else None,
                    tau_max_state=max_tau_state if np.isfinite(max_tau_state) else None,
                    badjac_ptau=bad_jacobian_ptau,
                    badjac_state=bad_jacobian_state,
                    badjac_used=bool(bad_jacobian),
                    mode=startup_policy.badjac_mode,
                    label="iter",
                )

                if np.isfinite(min_tau) and np.isfinite(max_tau):
                    history_lists.append_bad_jacobian(track_history, min_tau, max_tau, bool(bad_jacobian))
                    if bad_jacobian and _env_dump_badjac not in ("", "0"):
                        dump_dir = _env_dump_dir
                        if dump_dir:
                            try:
                                path = Path(dump_dir) / "bad_jacobian.log"
                                with path.open("a", encoding="utf-8") as f:
                                    f.write(f"iter={iter2} min_tau={min_tau:.6e} max_tau={max_tau:.6e}\n")
                            except Exception:
                                pass
                else:
                    history_lists.append_bad_jacobian(track_history, float("nan"), float("nan"), False)
            else:
                history_lists.append_bad_jacobian(track_history, float("nan"), float("nan"), False)

            # VMEC eqsolve: after the first evolve step, if the Jacobian is bad
            # and ijacob==0, retry with an improved axis guess.
            if bool(vmec2000_control) and (not axis_reset_done) and bool(lmove_axis) and (iter2 == 1):
                axis_runtime_result = _run_initial_axis_reset_runtime(
                    state=state,
                    k=k,
                    iter_idx=int(iter2),
                    bad_jacobian=bool(bad_jacobian),
                    fsq_phys=fsqr_f + fsqz_f + fsql_f,
                    axis_reset_done=bool(axis_reset_done),
                    lmove_axis=bool(lmove_axis),
                    vmec2000_control=bool(vmec2000_control),
                    axis_reset_fsq_min=float(axis_reset_fsq_min),
                    force_axis_reset=bool(force_axis_reset),
                    axis_reset_always_3d=bool(axis_reset_always_3d),
                    lthreed=bool(getattr(cfg, "lthreed", True)),
                    time_step=float(time_step),
                    prev_rz_fsq_before=float(prev_rz_fsq_before),
                    k_ndamp=int(k_ndamp),
                    verbose=bool(verbose),
                    verbose_vmec2000_table=bool(verbose_vmec2000_table),
                    callbacks=axis_reset_runtime_callbacks,
                )
                bad_jacobian = bool(axis_runtime_result.bad_jacobian)
                if axis_runtime_result.reset:
                    state = axis_runtime_result.state
                    axis_reset_done = True
                    freeb_controls_cached = None
                    precond_cache.clear()
                    _pop_iteration_histories()
                    # VMEC restarts the iteration after axis reset without
                    # advancing the iteration counter. Emulate that by
                    # repeating iter2==1 on the next loop pass.
                    if axis_runtime_result.repeat_iteration:
                        iter_offset -= 1
                    _record_timing("iteration_control_badjac", t_iteration_control_badjac_start)
                    continue
            _record_timing("iteration_control_badjac", t_iteration_control_badjac_start)

            # VMEC-style time-step control: VMEC2000's `TimeStepControl` + `restart_iter`.
            t_iteration_control_vmec_time_start = time.perf_counter() if timing_enabled else None
            if bool(vmec2000_control) and (not skip_time_control):
                tc_runtime = _run_vmec2000_time_control_runtime(
                    vmec2000_control=bool(vmec2000_control),
                    skip_time_control=bool(skip_time_control),
                    iter2=int(iter2),
                    iter1=int(iter1),
                    fsq_prev=float(fsq_prev),
                    fsq0_curr=float(fsq0_curr),
                    fsq0_prev=float(fsq0_prev),
                    res0=float(res0),
                    res1=float(res1),
                    bad_jacobian=bool(bad_jacobian),
                    vmec2000_fact=float(vmec2000_fact),
                    time_step=float(time_step),
                    restart_badjac_factor=float(restart_badjac_factor),
                    restart_badprog_factor=float(restart_badprog_factor),
                    ijacob=int(ijacob),
                    bad_resets=int(bad_resets),
                    fsq_prev_before=float(fsq_prev_before),
                    fsq0_prev_before=float(fsq0_prev_before),
                    k_ndamp=int(k_ndamp),
                    state_checkpoint=state_checkpoint,
                    prev_rz_fsq_before=prev_rz_fsq_before,
                    callbacks=_Vmec2000TimeControlCallbacks(
                        time_control_decision=_vmec2000_time_control_decision,
                        dump_time_control_trace=_dump_time_control_trace_record,
                        maybe_dump_checkpoint=_maybe_dump_checkpoint_record,
                        maybe_dump_time_control=_maybe_dump_time_control_record,
                        apply_controller_sample=_apply_controller_sample,
                        controller_sample=_controller_state_after_vmec2000_time_control_sample,
                        host_restart_update=_host_vmec2000_time_control_restart_update,
                        host_restart_branch_result=_host_vmec2000_time_control_restart_branch_result,
                        apply_restart_branch_result=_apply_restart_branch_result,
                        controller_restart_update=_controller_state_after_vmec2000_time_control_restart_update,
                    ),
                )
                fsq = tc_runtime.fsq
                fsq0 = tc_runtime.fsq0
                if tc_runtime.restarted:
                    pre_restart_reason = tc_runtime.pre_restart_reason
                    _record_timing("iteration_control_vmec_time", t_iteration_control_vmec_time_start)
                    continue
            _record_timing("iteration_control_vmec_time", t_iteration_control_vmec_time_start)

            # --- time-step control trackers + optional restart triggers ---
            t_iteration_control_restart_start = time.perf_counter() if timing_enabled else None
            restart_decision = _host_restart_decision(
                iter2=int(iter2),
                iter1=int(iter1),
                fsqr=fsqr_f,
                fsqz=fsqz_f,
                fsql=fsql_f,
                fsq1=fsq1,
                fsq_prev=fsq_prev,
                res0=res0,
                bad_growth_streak=bad_growth_streak,
                pre_restart_reason=pre_restart_reason,
                reference_mode=reference_mode,
                vmec2000_control=vmec2000_control,
                bad_jacobian=bad_jacobian,
                stage_prev_fsq=stage_prev_fsq,
                stage_transition_factor=stage_transition_factor,
                lmove_axis=lmove_axis,
                vmecpp_restart=vmecpp_restart,
                k_preconditioner_update_interval=k_preconditioner_update_interval,
            )
            fsq = restart_decision.fsq
            _apply_controller_sample(_controller_state_after_host_restart_decision_sample, restart_decision)
            pre_restart_reason = restart_decision.pre_restart_reason
            huge_initial_forces = restart_decision.huge_initial_forces

            if startup_policy.use_restart_triggers and pre_restart_reason != "none":
                pre_restart_runtime = _run_pre_restart_trigger_runtime(
                    use_restart_triggers=True,
                    pre_restart_reason=pre_restart_reason,
                    huge_initial_forces=bool(huge_initial_forces),
                    huge_force_restart_count=int(huge_force_restart_count),
                    time_step=float(time_step),
                    restart_badjac_factor=float(restart_badjac_factor),
                    restart_badprog_factor=float(restart_badprog_factor),
                    stage_transition_scale=float(stage_transition_scale),
                    step_size=float(step_size),
                    ijacob=int(ijacob),
                    bad_resets=int(bad_resets),
                    iter2=int(iter2),
                    compact_iter_idx=int(it),
                    fsq_prev_before=float(fsq_prev_before),
                    fsq0_prev_before=float(fsq0_prev_before),
                    k_ndamp=int(k_ndamp),
                    state_checkpoint=state_checkpoint,
                    state_before_restart=state,
                    velocity_blocks_before=velocity_blocks,
                    static=static,
                    prev_rz_fsq_before=prev_rz_fsq_before,
                    vmec2000_control=bool(vmec2000_control),
                    verbose=bool(verbose),
                    verbose_vmec2000_table=bool(verbose_vmec2000_table),
                    callbacks=_PreRestartTriggerCallbacks(
                        host_update=_host_pre_restart_trigger_update,
                        host_branch_result=_host_pre_restart_trigger_branch_result,
                        apply_restart_branch_result=_apply_restart_branch_result,
                        controller_restart_update=_controller_state_after_pre_restart_update,
                        print_compact_update_status=_print_compact_residual_iteration_update_status,
                        preconditioner_diag_floats=_precond_diag_floats,
                        dump_xc_with_velocity_blocks=lambda *, state, velocities, static, iter_idx: (
                            _dump_xc_with_velocity_blocks(
                                dump_xc=_maybe_dump_xc,
                                state=state,
                                velocities=velocities,
                                static=static,
                                iter_idx=iter_idx,
                            )
                        ),
                    ),
                )
                pre_restart_reason = pre_restart_runtime.pre_restart_reason
                time_step_iter = pre_restart_runtime.time_step_iter
                step_status = pre_restart_runtime.step_status
                _record_timing("iteration_control_restart", t_iteration_control_restart_start)
                continue

            _record_timing("iteration_control_restart", t_iteration_control_restart_start)
            break
        if profile_started and (profile_start_iter is not None) and (iter2 == profile_start_iter):
            if has_jax():
                try:
                    jax.block_until_ready(state.Rcos)
                    jax.profiler.stop_trace()
                except Exception:
                    pass
            profile_started = False
            profile_active = False
        if converged:
            break
        t_iteration_control_evolve_start = time.perf_counter() if timing_enabled else None
        evolve = _residual_evolve_coefficients(
            iter2=int(iter2),
            iter1=int(iter1),
            inv_tau=inv_tau,
            time_step=float(time_step),
            fsq1=float(fsq1),
            fsq_prev=float(fsq_prev),
            fsq0_curr=float(fsq0_curr),
            k_ndamp=int(k_ndamp),
        )
        inv_tau = evolve.inv_tau
        fsq_prev = evolve.fsq_prev
        fsq0_prev = evolve.fsq0_prev
        dtau = evolve.dtau
        b1 = evolve.b1
        fac = evolve.fac
        force_blocks = _velocity_blocks_from_force_blocks(update_force_blocks)
        _dump_residual_evolve_trace(
            dump_evolve_trace=_dump_evolve_trace,
            iter2=int(iter2),
            iter1=int(iter1),
            stage="pre",
            fsq1=float(fsq1),
            fsq_prev=float(fsq_prev_before),
            time_step=float(time_step),
            dtau=float(dtau),
            b1=float(b1),
            fac=float(fac),
            state=state,
            velocities=velocity_blocks,
            forces=force_blocks,
        )

        _record_timing("iteration_control_evolve", t_iteration_control_evolve_start)
        if _record_timing("iteration_control", t_iteration_control_start):
            t_iteration_control_start = None
        t_update_start = time.perf_counter() if timing_enabled else None
        if bool(strict_update):
            # Strict update semantics: one preconditioned momentum update per
            # iteration in (m, n>=0) storage, no line-search accept/reject.
            w_curr = fsqr_f + fsqz_f + fsql_f
            state_backup = state
            t_trace_build_start = time.perf_counter() if timing_enabled and adjoint_trace else None
            if adjoint_trace:
                trace_entry = _build_strict_update_adjoint_trace_entry(
                    locals(),
                    materialize_func=_materialize_adjoint_trace_array,
                    adjoint_trace_mode=adjoint_trace_mode,
                )
            _record_timing("update_trace_build", t_trace_build_start)
            t_state_update_start = time.perf_counter() if timing_enabled else None
            dt_eff = float(time_step)
            if bool(limit_dt_from_force):
                dt_eff = _safe_dt_from_force_blocks(
                    dt_nominal=time_step,
                    max_coeff_delta_rms=max_coeff_delta_rms,
                    blocks=_force_blocks_from_update_order(force_blocks),
                )

            # Momentum semantics: v <- fac*(b1*v + dt*F), x <- x + dt*v.
            # Do not drop the dt factor in the force term; otherwise updates
            # scale like O(dt) instead of O(dt^2) and can immediately blow up.
            force_scale = float(dt_eff)

            need_update_rms = (
                bool(limit_update_rms)
                or bool(track_history)
                or bool(verbose)
                or bool(backtracking)
                or (bool(adjoint_trace) and adjoint_trace_mode == "full")
            )
            need_trial_eval = bool(backtracking) or bool(reference_mode) or bool(use_direct_fallback)
            use_jit_strict_update_step = (
                bool(jit_strict_update_enabled)
                and (not bool(host_update_assembly))
                and (not bool(limit_dt_from_force))
                and (not bool(limit_update_rms))
                and (not bool(need_trial_eval))
                and (not _tree_has_tracer(state))
            )
            if use_jit_strict_update_step:
                update_proposal = _jit_strict_momentum_update_proposal(
                    state=state,
                    static=static,
                    velocities=velocity_blocks,
                    forces=force_blocks,
                    dt_eff=float(dt_eff),
                    b1=float(b1),
                    fac=float(fac),
                    force_scale=float(force_scale),
                    flip_sign=float(flip_sign),
                    max_update_rms=float(max_update_rms),
                    need_update_rms=bool(need_update_rms),
                    divide_by_scalxc_for_update=bool(divide_by_scalxc_for_update),
                    free_boundary_enabled=bool(free_boundary_enabled),
                    strict_update_step_jit_func=_strict_update_step_jit,
                )
            else:
                update_proposal = _strict_momentum_update_proposal(
                    velocities=velocity_blocks,
                    forces=force_blocks,
                    host_update_assembly=bool(host_update_assembly),
                    need_update_rms=bool(need_update_rms),
                    materialize_update_rms=(
                        bool(limit_update_rms)
                        or bool(backtracking)
                        or (bool(adjoint_trace) and adjoint_trace_mode == "full")
                    ),
                    limit_update_rms=bool(limit_update_rms),
                    max_update_rms=float(max_update_rms),
                    b1=float(b1),
                    fac=float(fac),
                    force_scale=float(force_scale),
                    flip_sign=float(flip_sign),
                    dt_eff=float(dt_eff),
                    delta_transforms=_physical_delta_transforms,
                    delta_tuple_from_blocks=_delta_tuple_from_blocks,
                    candidate_state_from_delta_tuple=_candidate_state_from_delta_tuple,
                )
            velocity_blocks = update_proposal.velocities
            update_rms_j = update_proposal.update_rms_j
            update_rms = update_proposal.update_rms
            update_rms_preclip = update_proposal.update_rms_preclip
            scl = update_proposal.scale
            update_deltas = update_proposal.update_deltas
            state_try = update_proposal.state
            probe_bad_jacobian = False
            if need_trial_eval:
                trial_eval = _strict_trial_evaluation(
                    state_try=state_try,
                    velocities=velocity_blocks,
                    update_deltas=update_deltas,
                    update_rms=update_rms,
                    dt_eff=float(dt_eff),
                    w_curr=float(w_curr),
                    backtracking=bool(backtracking),
                    reference_mode=bool(reference_mode),
                    host_update_assembly=bool(host_update_assembly),
                    zero_m1_value=zero_m1,
                    zero_m1_host=float(np.asarray(zero_m1)),
                    zero_m1_probe_value=jnp.asarray(0.0, dtype=zero_m1.dtype),
                    candidate_state_from_delta_tuple=_candidate_state_from_delta_tuple,
                    freeb_bsqvac_half_for_trial_state=_freeb_bsqvac_half_for_trial_state,
                    trial_residual_total=_trial_residual_total,
                )
                state_try = trial_eval.state
                velocity_blocks = trial_eval.velocities
                dt_eff = trial_eval.dt_eff
                update_rms = trial_eval.update_rms
                w_try = trial_eval.w_try
                w_try_ratio = trial_eval.w_try_ratio
                probe_bad_jacobian = trial_eval.probe_bad_jacobian
            else:
                w_try = w_curr
                w_try_ratio = 1.0

            # Require (near) monotone improvement; otherwise fall back to the
            # restart/timestep control path.
            step_acceptance = _strict_step_acceptance_decision(
                w_try=float(w_try),
                w_curr=float(w_curr),
                backtracking=bool(backtracking),
            )
            branch_result = _strict_step_branch_result(
                acceptance=step_acceptance,
                state_try=state_try,
                state_backup=state_backup,
                update_rms=update_rms,
                vmec2000_control=bool(vmec2000_control),
                huge_force_restart_count=int(huge_force_restart_count),
            )
            if not branch_result.accepted:
                if use_direct_fallback:
                    # Try a small direct-force step (no momentum memory) before
                    # a full restart. This is an experimental parity path.
                    fallback_trial = _direct_force_fallback_trial(
                        forces=force_blocks,
                        dt_eff=float(dt_eff),
                        max_update_rms=float(max_update_rms),
                        flip_sign=float(flip_sign),
                        delta_transforms=_internal_delta_transforms,
                        delta_tuple_from_blocks=_delta_tuple_from_blocks,
                        candidate_state_from_delta_tuple=_candidate_state_from_delta_tuple,
                        freeb_bsqvac_half_for_trial_state=_freeb_bsqvac_half_for_trial_state,
                        trial_residual_total=lambda candidate_state, freeb_bsqvac_half_trial: _trial_residual_total(
                            candidate_state,
                            freeb_bsqvac_half_trial,
                            zero_m1_value=zero_m1,
                        ),
                    )
                    fallback_acceptance = _direct_force_fallback_acceptance_decision(
                        residual=float(fallback_trial.residual),
                        current_residual=float(w_curr),
                    )
                    branch_result = _strict_step_branch_result_after_direct_fallback(
                        branch=branch_result,
                        fallback_trial=fallback_trial,
                        acceptance=fallback_acceptance,
                        clear_cache_after_rejected=bool(vmec2000_control),
                    )
                    if adjoint_trace and branch_result.fallback_direct_dt is not None:
                        trace_entry["fallback_direct_dt"] = float(branch_result.fallback_direct_dt)
            branch_application = _apply_strict_step_branch(branch_result)
            if not branch_result.accepted:
                catastrophic_restart = branch_result.catastrophic_restart
                if catastrophic_restart:
                    restart_update = _host_catastrophic_restart_update(
                        probe_bad_jacobian=bool(probe_bad_jacobian),
                        w_try=float(w_try),
                        time_step=float(time_step),
                        restart_badjac_factor=float(restart_badjac_factor),
                        restart_badprog_factor=float(restart_badprog_factor),
                        step_size=float(step_size),
                        ijacob=int(ijacob),
                        bad_resets=int(bad_resets),
                        iter2=int(iter2),
                        fsq_prev_before=float(fsq_prev_before),
                        fsq0_prev_before=float(fsq0_prev_before),
                        k_ndamp=int(k_ndamp),
                        max_coeff_delta_rms=float(max_coeff_delta_rms),
                        max_update_rms=float(max_update_rms),
                    )
                    _apply_controller_update(_controller_state_after_catastrophic_restart_update, restart_update)
                    branch_result = _strict_step_branch_result_after_catastrophic_restart(
                        branch=branch_result,
                        restart_update=restart_update,
                        state_backup=state_backup,
                    )
                    branch_application = _apply_strict_step_branch(
                        branch_result,
                        after_catastrophic_restart=True,
                    )
            _record_update_state_ready_timing(
                timing_enabled=bool(timing_enabled),
                timing_stats=timing_stats,
                start=t_state_update_start,
                state=state,
                perf_counter=time.perf_counter,
                has_jax=has_jax,
                jax_module=jax,
            )
            t_trace_finalize_start = time.perf_counter() if timing_enabled and adjoint_trace else None
            if adjoint_trace:
                _finalize_strict_update_adjoint_trace_entry(
                    trace_entry,
                    locals(),
                    adjoint_trace_mode=adjoint_trace_mode,
                )
                adjoint_step_trace_history.append(trace_entry)
            _record_timing("update_trace_finalize", t_trace_finalize_start)
            _record_update_total_timing(
                timing_enabled=bool(timing_enabled),
                timing_stats=timing_stats,
                start=t_update_start,
                state=state,
                perf_counter=time.perf_counter,
                has_jax=has_jax,
                jax_module=jax,
            )
            timing_stats["iterations"] += 1
        else:
            w_curr = fsqr_f + fsqz_f + fsql_f
            non_strict_update = _backtracking_momentum_search(
                state=state,
                velocities=velocity_blocks,
                forces=force_blocks,
                time_step=float(time_step),
                step_size=float(step_size),
                b1=float(b1),
                fac=float(fac),
                flip_sign=float(flip_sign),
                w_curr=float(w_curr),
                delta_transforms=_internal_delta_transforms,
                delta_tuple_from_blocks=_delta_tuple_from_blocks,
                candidate_state_from_delta_tuple=_candidate_state_from_delta_tuple,
                freeb_bsqvac_half_for_trial_state=_freeb_bsqvac_half_for_trial_state,
                trial_residual_total=lambda candidate_state, freeb_bsqvac_half_trial: _trial_residual_total(
                    candidate_state,
                    freeb_bsqvac_half_trial,
                    zero_m1_value=zero_m1,
                    timing_label="backtracking",
                ),
            )
            state = non_strict_update.state
            velocity_blocks = non_strict_update.velocities
            dt_eff = non_strict_update.dt_eff
            update_rms = non_strict_update.update_rms
            step_status = non_strict_update.step_status
            timing_stats["iterations"] += 1
            restart_reason = "none"
            w_try = float("nan")
            w_try_ratio = float("nan")
            restart_path = "non_strict"
        update_rms_record = update_rms_j if bool(strict_update) else float(update_rms)
        history_lists.append_step_sample(
            track_history=bool(track_history),
            step=float(dt_eff),
            dt_eff=float(dt_eff),
            update_rms=update_rms_record,
            w_curr=float(w_curr),
            w_try=float(w_try),
            w_try_ratio=float(w_try_ratio),
            restart_path=str(restart_path),
        )
        t_iteration_post_update_start = time.perf_counter() if timing_enabled else None
        _dump_residual_evolve_trace(
            dump_evolve_trace=_dump_evolve_trace,
            iter2=int(iter2),
            iter1=int(iter1),
            stage="post",
            fsq1=float(fsq1),
            fsq_prev=float(fsq_prev_before),
            time_step=float(time_step),
            dtau=float(dtau),
            b1=float(b1),
            fac=float(fac),
            state=state,
            velocities=velocity_blocks,
            forces=force_blocks,
        )
        _dump_xc_with_velocity_blocks(
            dump_xc=_maybe_dump_xc,
            state=state,
            velocities=velocity_blocks,
            static=static,
            iter_idx=int(iter2),
        )
        update_rms_print = _residual_update_rms_for_print(
            verbose=bool(verbose),
            strict_update=bool(strict_update),
            update_rms_j=update_rms_j if bool(strict_update) else None,
            update_rms=update_rms,
        )
        _print_residual_iteration_update_status(
            verbose=bool(verbose),
            vmec2000_control=bool(vmec2000_control),
            verbose_vmec2000_table=bool(verbose_vmec2000_table),
            should_print_vmec2000=_should_print_vmec2000,
            print_vmec2000_iter_row=_print_vmec2000_iter_row,
            precond_diag_floats=_precond_diag_floats,
            iter_idx=int(iter2),
            max_iter=int(max_iter),
            compact_iter_idx=int(it),
            fsqr=fsqr_f,
            fsqz=fsqz_f,
            fsql=fsql_f,
            dt_eff=float(dt_eff),
            update_rms=update_rms_print,
            time_step=float(time_step),
            r00=float(r00_last),
            z00=float(z00_last),
            w_mhd=float(w_vmec_last),
            step_status=step_status,
        )
        history_lists.append_terminal(
            track_history=bool(track_history),
            step_status=step_status,
            restart_reason=restart_reason,
            pre_restart_reason=pre_restart_reason,
            time_step=float(time_step),
            res0=float(res0),
            res1=float(res1),
            fsq_prev=float(fsq_prev),
            bad_growth_streak=int(bad_growth_streak),
            iter1=int(iter1),
            iter2=int(iter2),
            fsqr=fsqr_f,
            fsqz=fsqz_f,
            fsql=fsql_f,
            free_boundary_enabled=bool(free_boundary_enabled),
            freeb_ivac=freeb_ivac,
            freeb_ivacskip=freeb_ivacskip,
            freeb_reused=freeb_reused,
            freeb_solve_time=freeb_solve_time,
            freeb_sample_time=freeb_sample_time,
        )
        # VMEC eqsolve behavior: when `ivac==1`, print turn-on and promote to
        # `ivac=2` for subsequent iterations.
        if free_boundary_enabled and int(freeb_ivac) == 1:
            if verbose and bool(verbose_vmec2000_table):
                print(f"\n  VACUUM PRESSURE TURNED ON AT {int(iter2):4d} ITERATIONS\n", flush=True)
            freeb_ivac = int(freeb_ivac) + 1
        skip_time_control = False
        _record_timing("iteration_post_update", t_iteration_post_update_start)

    return _finalize_residual_iter_from_namespace(
        locals(),
        result_type=SolveVmecResidualResult,
        nestor_external_only_step_func=nestor_external_only_step,
        residual_fsq_from_norms_func=_residual_fsq_from_norms,
        device_get_floats_func=_device_get_floats,
        residual_convergence_flags_func=_residual_convergence_flags,
        residual_iter_history_diagnostics_func=lambda _namespace: history_lists.diagnostics(),
        attach_free_boundary_diagnostics=_attach_freeb_diag,
        return_final_force_payload=bool(return_final_force_payload),
    )
