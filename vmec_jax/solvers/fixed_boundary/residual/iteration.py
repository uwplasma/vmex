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

from contextlib import nullcontext
from collections import OrderedDict
from functools import partial
import time
import os
from pathlib import Path
from typing import Any

import numpy as np

from vmec_jax._compat import has_jax, jax, jnp, jit
from vmec_jax import _solve_runtime
from vmec_jax.solvers.fixed_boundary.residual import policy as _residual_iter_policy
from vmec_jax.solvers.fixed_boundary.residual.config import (
    HEAVY_DUMP_ENVS as _HEAVY_DUMP_ENVS,
    LIGHT_DUMP_ENVS as _LIGHT_DUMP_ENVS,
    resolve_axis_reset_config as _resolve_axis_reset_config,
    resolve_debug_print_config as _resolve_debug_print_config,
    resolve_host_profile_setup as _resolve_host_profile_setup,
    resolve_nstep_screen as _resolve_nstep_screen,
    resolve_setup_host_enforce as _resolve_setup_host_enforce,
    should_probe_bad_jacobian_state as _should_probe_bad_jacobian_state,
)
from vmec_jax.solvers.fixed_boundary.residual.policy import (
    append_residual_iter_terminal_history as _append_residual_iter_terminal_history,
    append_preconditioned_residual_history as _append_preconditioned_residual_history,
    append_zero_update_history_record as _append_zero_update_history_record,
    bad_jacobian_requires_state_jacobian as _bad_jacobian_requires_state_jacobian,
    bad_jacobian_tau_decision as _bad_jacobian_tau_decision,
    host_restart_decision as _host_restart_decision,
    new_residual_iter_histories as _new_residual_iter_histories,
    numpy_preconditioner_apply_policy as _numpy_preconditioner_apply_policy,
    pop_residual_iter_rollback_histories as _pop_residual_iter_rollback_histories,
    resolve_residual_iter_startup_policy as _resolve_residual_iter_startup_policy,
    scan_fallback_decision as _scan_fallback_decision,
    scan_fallback_message as _scan_fallback_message,
    select_bad_jacobian_decision as _select_bad_jacobian_decision,
    vmec2000_time_control_decision as _vmec2000_time_control_decision,
)
from vmec_jax.solvers.fixed_boundary.residual.runtime import (
    _attach_free_boundary_external_field_diag as _runtime_attach_free_boundary_external_field_diag,
    _converged_residuals_scan_fast as _runtime_converged_residuals_scan_fast,
    _device_get_floats,
    _freeb_trial_bsqvac_half as _runtime_freeb_trial_bsqvac_half,
    _initial_setup_phase_timings,
    _maybe_dump_ptau as _runtime_maybe_dump_ptau,
    _maybe_print_nonscan_state_debug,
    _new_residual_iter_timing_stats as _runtime_new_residual_iter_timing_stats,
    _record_compute_force_timing as _runtime_record_compute_force_timing,
    _record_setup_timing as _runtime_record_setup_timing,
    _setup_timer_start as _runtime_setup_timer_start,
    _vmec_freeb_plascur_from_bcovar as _runtime_vmec_freeb_plascur_from_bcovar,
    resolve_residual_profile_window as _resolve_residual_profile_window,
)
from vmec_jax.solvers.fixed_boundary.residual.accelerated_scan import (
    run_accelerated_residual_scan as _run_accelerated_residual_scan,
)
from vmec_jax.solvers.fixed_boundary.residual.setup import (
    build_residual_cache_keys as _build_residual_cache_keys,
    free_boundary_pressure_edge_scale as _free_boundary_pressure_edge_scale,
    grid_matches_vmec_static_grid as _grid_matches_vmec_static_grid,
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
    evaluate_residual_force_from_state as _evaluate_residual_force_from_state,  # noqa: F401 - compatibility alias for tests/internal users.
    finalize_strict_update_adjoint_trace_entry as _finalize_strict_update_adjoint_trace_entry,
    make_residual_force_evaluator as _make_residual_force_evaluator,
)
from vmec_jax.solvers.fixed_boundary.residual.update import (
    ResidualVelocityBlocks as _ResidualVelocityBlocks,
    backtracking_momentum_search as _backtracking_momentum_search,
    candidate_state_from_deltas as _candidate_state_from_deltas_helper,
    candidate_state_from_delta_tuple as _candidate_state_from_delta_tuple_helper,
    delta_tuple_from_blocks as _delta_tuple_from_blocks_helper,
    force_update_rms as _force_update_rms,
    host_catastrophic_restart_update as _host_catastrophic_restart_update,
    host_force_update_rms as _host_force_update_rms,
    host_momentum_update_np as _host_momentum_update_np,
    initial_residual_velocity_state as _initial_residual_velocity_state,
    momentum_update_jax as _momentum_update_jax,
    scale_velocity_blocks as _scale_velocity_blocks,
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
    _pack_resume_state_record,
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
    ForceBlocks as _ForceBlocks,
    preconditioner_output_blocks_jax as _preconditioner_output_blocks_jax,
    preconditioner_output_blocks_np as _preconditioner_output_blocks_np,
    radial_preconditioner_output_blocks_jax as _radial_preconditioner_output_blocks_jax,
)
from vmec_jax.solvers.fixed_boundary.residual.force_norms import (
    lambda_preconditioned_full_norm as _lambda_preconditioned_full_norm,
    mode_weight_force_blocks_jax as _mode_weight_force_blocks_jax,
    mode_weight_force_blocks_np as _mode_weight_force_blocks_np,
    residual_fsq_from_norms as _residual_fsq_from_norms,
    safe_dt_from_force_blocks as _safe_dt_from_force_blocks,
)
from vmec_jax.solvers.fixed_boundary.residual.host_diagnostics import (
    print_compact_converged_status as _print_compact_converged_status,
    print_compact_physical_residual_status as _print_compact_physical_residual_status,
    print_compact_residual_iteration_update_status as _print_compact_residual_iteration_update_status,
    print_residual_iteration_update_status as _print_residual_iteration_update_status,
    resolve_vmec2000_print_context as _resolve_vmec2000_print_context,
    sample_vmec_iteration_scalars as _sample_vmec_iteration_scalars,
)
from vmec_jax.solvers.fixed_boundary.residual.scan_adapters import (
    ResidualScanPathHooks,
    ScanConvergencePredicate,
    ScanDeviceRuntime,
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
    ptau_minmax_from_k_jax as _ptau_minmax_from_k_jax_helper,
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
    empty_preconditioner_cache_snapshot as _empty_preconditioner_cache_snapshot,
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
    initial_axis_reset_runtime_decision as _initial_axis_reset_runtime_decision,
    reset_axis_from_boundary as _reset_axis_from_boundary_impl,
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
from vmec_jax.vmec_tomnsp import TomnspsRZL


_SCAN_RUNNER_CACHE: OrderedDict[tuple, Any] = OrderedDict()
_COMPUTE_FORCES_CACHE: OrderedDict[tuple, Any] = OrderedDict()

_HostRestartDecision = _residual_iter_policy.HostRestartDecision
_ResidualIterHistoryRecord = _residual_iter_policy.ResidualIterHistoryRecord
_Vmec2000ScanOptions = _residual_iter_policy.Vmec2000ScanOptions
_Vmec2000TimeControlDecision = _residual_iter_policy.Vmec2000TimeControlDecision

_m1_internal_to_physical_pair = _geometry_m1_internal_to_physical_pair
_mn_sin_to_signed_physical_batch = _geometry_mn_sin_to_signed_physical_batch
_rz_norm_np = _geometry_rz_norm_np

_strict_update_step_jit = partial(_precond_payload_facade._strict_update_step_jit, has_jax_func=has_jax)
_preconditioner_output_scaling_jit = partial(
    _precond_payload_facade._preconditioner_output_scaling_jit,
    has_jax_func=has_jax,
)
_preconditioner_output_payload_jit = partial(
    _precond_payload_facade._preconditioner_output_payload_jit,
    has_jax_func=has_jax,
)
_preconditioner_apply_payload_jit = partial(
    _precond_payload_facade._preconditioner_apply_payload_jit,
    has_jax_func=has_jax,
)
_accepted_control_payload_jit = partial(_precond_payload_facade._accepted_control_payload_jit, has_jax_func=has_jax)
_preconditioner_apply_payload_fused = _precond_payload_facade._preconditioner_apply_payload_fused
_cached_or_current_f_norm1_jax = _precond_payload_facade._cached_or_current_f_norm1_jax
_split_preconditioner_apply_payload = _precond_payload_facade._split_preconditioner_apply_payload


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


def _edge_bsqvac_from_nestor(nestor_result, static) -> np.ndarray:
    bsqvac_edge = np.asarray(nestor_result.vac_total.bsqvac, dtype=float)
    if bsqvac_edge.ndim == 2 and int(bsqvac_edge.shape[1]) == 1 and int(getattr(static.cfg, "nzeta", 1)) > 1:
        bsqvac_edge = np.repeat(bsqvac_edge, int(static.cfg.nzeta), axis=1)
    return bsqvac_edge


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

    startup_policy = _resolve_residual_iter_startup_policy(
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
        state_has_tracer=state0_has_tracer,
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
    max_iter = startup_policy.max_iter
    step_size = startup_policy.step_size
    precompile_only = startup_policy.precompile_only
    host_update_assembly = startup_policy.host_update_assembly
    adjoint_trace = startup_policy.adjoint_trace
    adjoint_trace_mode = startup_policy.adjoint_trace_mode
    preconditioner_use_precomputed_tridi_policy = startup_policy.preconditioner_use_precomputed_tridi_policy
    preconditioner_use_lax_tridi_policy = startup_policy.preconditioner_use_lax_tridi_policy

    signgs = startup_policy.signgs
    fsq_total_target = startup_policy.fsq_total_target
    lambda_update_scale = startup_policy.lambda_update_scale
    enforce_vmec_lambda_axis = startup_policy.enforce_vmec_lambda_axis
    vmec2000_control = startup_policy.vmec2000_control
    badjac_mode = startup_policy.badjac_mode
    badjac_use_state = startup_policy.badjac_use_state
    dump_ptau_state = startup_policy.dump_ptau_state
    light_history = startup_policy.light_history
    resume_state_mode = startup_policy.resume_state_mode
    badjac_state_probe = startup_policy.badjac_state_probe
    badjac_initial_state_probe_iters = startup_policy.badjac_initial_state_probe_iters
    ptau_tol = startup_policy.ptau_tol
    ptau_tol_rel = startup_policy.ptau_tol_rel
    reference_mode = startup_policy.reference_mode
    jit_precompile = startup_policy.jit_precompile
    use_restart_triggers = startup_policy.use_restart_triggers
    use_direct_fallback = startup_policy.use_direct_fallback
    vmecpp_restart = startup_policy.vmecpp_restart
    verbose_vmec2000_table = startup_policy.verbose_vmec2000_table
    scan_fallback_enabled = startup_policy.scan_fallback_enabled
    scan_fallback_iters = startup_policy.scan_fallback_iters
    scan_fallback_badjac_limit = startup_policy.scan_fallback_badjac_limit
    scan_fallback_fsq_abs = startup_policy.scan_fallback_fsq_abs
    scan_fallback_accept_frac = startup_policy.scan_fallback_accept_frac
    scan_fallback_fsq_factor = startup_policy.scan_fallback_fsq_factor
    stage_transition_factor = startup_policy.stage_transition_factor
    stage_transition_scale = startup_policy.stage_transition_scale
    stage_prev_fsq = startup_policy.stage_prev_fsq
    auto_flip_force = startup_policy.auto_flip_force
    jit_forces = startup_policy.jit_forces
    use_scan = startup_policy.use_scan
    differentiating_scan = startup_policy.differentiating_scan
    limit_dt_from_force = startup_policy.limit_dt_from_force
    limit_update_rms = startup_policy.limit_update_rms
    backtracking = startup_policy.backtracking
    strict_update = startup_policy.strict_update
    if startup_policy.disabled_jit_for_dumps:
        if verbose:
            print("[solve_fixed_boundary_residual_iter] jit_forces disabled (debug dumps enabled)")
    track_history = startup_policy.track_history

    _pack_resume_state = partial(_pack_resume_state_record, mode=resume_state_mode)

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

    # VMEC2000 evaluates the force kernels on VMEC's internal
    # angle grid. In particular, when `lasym=False`, VMEC uses a reduced theta
    # grid (stellarator symmetry) for the force pipeline. Rebuild `static`
    # using `vmec_angle_grid(...)` so the force terms do not mix full-grid and
    # VMEC-grid arrays (which triggers broadcasting errors and parity drift).
    _t_setup_static_grid = _setup_timer_start()
    cfg = static.cfg
    grid_vmec = vmec_angle_grid(
        ntheta=int(cfg.ntheta),
        nzeta=int(cfg.nzeta),
        nfp=int(cfg.nfp),
        lasym=bool(cfg.lasym),
    )
    reuse_static = _grid_matches_vmec_static_grid(static.grid, grid_vmec)
    if not reuse_static:
        static = build_static(
            cfg,
            grid=grid_vmec,
            mgrid_metadata=getattr(static, "mgrid_metadata", None),
            free_boundary_extcur=getattr(static, "free_boundary_extcur", None),
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
    s = jnp.asarray(static.s)
    freeb_pres_scale = _free_boundary_pressure_edge_scale(
        free_boundary_enabled=bool(free_boundary_enabled),
        indata=indata,
        s=s,
    )
    dtype_state = jnp.asarray(state0.Rcos).dtype
    zero_precond_diag = (
        jnp.zeros((int(s.shape[0]),), dtype=dtype_state),
        jnp.zeros((int(s.shape[0]),), dtype=dtype_state),
    )
    zero_tcon = jnp.zeros((int(s.shape[0]),), dtype=dtype_state)
    constraint_active_false = jnp.asarray(False)

    # Boundary + axis recompute helpers (for VMEC-style bad-Jacobian reset).
    boundary_for_axis = (
        boundary_from_indata(indata, static.modes, apply_m1_constraint=True) if indata is not None else None
    )
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
        enforce_vmec_lambda_axis=enforce_vmec_lambda_axis,
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
        nonlocal axis_reset_coeffs
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

    prefer_host_default_profiles = not state0_has_tracer

    _profile_numpy_patch = None
    host_profile_setup = _resolve_host_profile_setup(
        backend_name=_scan_backend_name(),
        profile_setup_env=os.getenv("VMEC_JAX_HOST_PROFILE_SETUP", "auto"),
    )
    if (
        (bool(host_update_assembly) or bool(host_profile_setup))
        and has_jax()
        and (not state0_has_tracer)
    ):
        try:
            from vmec_jax.vmec_numpy_forces import _numpy_module_patch as _profile_numpy_patch
        except Exception:
            _profile_numpy_patch = None
    try:
        with _profile_numpy_patch() if _profile_numpy_patch is not None else nullcontext():
            s_profile = s
            if _profile_numpy_patch is not None:
                from vmec_jax.vmec_numpy_forces import _wrap as _np_wrap

                s_profile = _np_wrap(np.asarray(s))
            profile_setup = _build_wout_like_profiles_from_indata(
                indata=indata,
                static=static,
                s_profile=s_profile,
                signgs=signgs,
                idx00=idx00,
                prefer_host_default_profiles=prefer_host_default_profiles,
                s_profile_has_tracer=_tree_has_tracer(s_profile),
        )
    except Exception:
        if bool(precompile_only):
            return _precompile_only_residual_iter_result(result_type=SolveVmecResidualResult, state=state0)
        raise
    wout_like = profile_setup.wout_like

    trig = _residual_force_context_helpers.resolve_residual_trig(
        state0=state0,
        static=static,
        wout_like=wout_like,
        vmec_trig_tables_func=vmec_trig_tables,
        jnp_module=jnp,
    )
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

    _t_setup_cache_key_hash = _setup_timer_start()
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
    static_key = cache_keys.static_key
    wout_key = cache_keys.wout_key
    edge_signature_key = cache_keys.edge_signature_key
    edge_value_key = cache_keys.edge_value_key
    _record_setup_timing("setup_cache_key_hash", _t_setup_cache_key_hash)

    def _apply_radial_tridi(a, alpha: float):
        return _radial_tridi_smooth_dirichlet(a, alpha=alpha, skip_nonpositive=True)

    def _apply_radial_tridi_batched(arrs, alpha: float):
        if alpha <= 0.0:
            return tuple(arrs)
        # Stack directly into (ns, B, ...) to avoid swapaxes.
        stack = jnp.stack(arrs, axis=1)
        smooth = _radial_tridi_smooth_dirichlet(stack, alpha=alpha)
        return tuple(smooth[:, i] for i in range(int(smooth.shape[1])))

    _t_setup_ptau_constants = _setup_timer_start()
    _ptau_context = _build_ptau_minmax_context(
        s,
        has_jax=has_jax(),
        s_has_tracer=_tree_has_tracer(s),
        pshalf_from_s_np=_pshalf_from_s_np,
        pshalf_from_s_jax=_pshalf_from_s_jax,
    )
    _record_setup_timing("setup_ptau_constants", _t_setup_ptau_constants)

    _ptau_minmax_from_k_host = partial(
        _ptau_minmax_from_k_host_helper,
        ptau_context=_ptau_context,
        compute_jit=_ptau_compute_jit,
        ptau_minmax_host_func=_scan_math_ptau_minmax_from_k_host,
    )
    _ptau_minmax_from_k_jax = partial(
        _ptau_minmax_from_k_jax_helper,
        ptau_context=_ptau_context,
        pshalf_from_s_jax=_pshalf_from_s_jax,
        ptau_minmax_jax_func=_scan_math_ptau_minmax_from_k_jax,
    )
    _ptau_minmax = partial(
        _ptau_minmax_helper,
        ptau_context=_ptau_context,
        has_jax_func=has_jax,
        compute_jit=_ptau_compute_jit,
        pshalf_from_s_jax=_pshalf_from_s_jax,
        ptau_minmax_host_func=_scan_math_ptau_minmax_from_k_host,
        ptau_minmax_jax_func=_scan_math_ptau_minmax_from_k_jax,
    )
    _accepted_control_ptau_arrays = partial(
        _accepted_control_ptau_arrays_helper,
        kernel_arrays_from_k=_scan_math_kernel_arrays_from_k,
    )
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

    def _lambda_preconditioner(bc, *, return_faclam: bool = False, return_debug: bool = False):
        lam_r0scale = float(getattr(trig, "r0scale", 1.0)) if trig is not None else 1.0
        from vmec_jax.preconditioner_1d_jax import lambda_preconditioner_cached

        return lambda_preconditioner_cached(
            bc=bc,
            trig=trig,
            s=s,
            cfg=cfg,
            return_faclam=return_faclam,
            return_debug=return_debug,
            r0scale=lam_r0scale,
        )

    def _rz_preconditioner_matrices_local(
        *,
        bc,
        k,
        jmax_override: int | None = None,
        use_precomputed: bool | None = None,
        use_lax_tridi: bool | None = None,
    ):
        from vmec_jax.preconditioner_1d_jax import rz_preconditioner_matrices

        return rz_preconditioner_matrices(
            bc=bc,
            k=k,
            trig=trig,
            s=s,
            cfg=cfg,
            jmax_override=jmax_override,
            use_precomputed=use_precomputed,
            use_lax_tridi=use_lax_tridi,
        )

    _numpy_precond_policy = _numpy_preconditioner_apply_policy(
        host_update_assembly=bool(host_update_assembly),
        max_iter=int(max_iter),
        mpol=int(getattr(cfg, "mpol", 0)),
        ntor=int(getattr(cfg, "ntor", 0)),
        max_iter_env=os.getenv("VMEC_JAX_NUMPY_PRECOND_MAX_ITER", "240"),
        min_mode_count_env=os.getenv("VMEC_JAX_NUMPY_PRECOND_MIN_MODES", "16"),
    )
    _use_numpy_preconditioner_apply = bool(_numpy_precond_policy.enabled)

    def _rz_preconditioner_apply_local(
        *,
        frzl_in,
        mats,
        jmax,
        use_precomputed: bool | None = None,
        use_lax_tridi: bool | None = None,
    ):
        if bool(_use_numpy_preconditioner_apply) and not _tree_has_tracer(frzl_in):
            from vmec_jax.preconditioner_1d_jax import rz_preconditioner_apply_numpy

            return rz_preconditioner_apply_numpy(
                frzl_in=frzl_in,
                mats=mats,
                jmax=jmax,
                cfg=cfg,
                use_precomputed=use_precomputed,
            )
        from vmec_jax.preconditioner_1d_jax import rz_preconditioner_apply_jit

        return rz_preconditioner_apply_jit(
            frzl_in=frzl_in,
            mats=mats,
            jmax=jmax,
            cfg=cfg,
            use_precomputed=use_precomputed,
            use_lax_tridi=use_lax_tridi,
        )

    def _rz_preconditioner(frzl_in: TomnspsRZL, bc, k):
        from vmec_jax.preconditioner_1d_jax import rz_preconditioner

        return rz_preconditioner(
            frzl_in=frzl_in,
            bc=bc,
            k=k,
            trig=trig,
            s=s,
            cfg=cfg,
        )

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

    # NumPy hot-path: wrap _compute_forces_impl with pure-NumPy module patching.
    # Used when host_update_assembly=True to eliminate all JAX dispatch overhead.
    numpy_force = _prepare_numpy_force_fast_path(
        host_update_assembly=bool(host_update_assembly),
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

        def _compute_forces_nodump(
            state: VMECState,
            *,
            include_edge: bool,
            include_edge_residual: bool | None = None,
            zero_m1: Any,
            freeb_bsqvac_half: Any | None = None,
            constraint_rcon0: Any | None = None,
            constraint_zcon0: Any | None = None,
            constraint_precond_diag: tuple[Any, Any] | None = None,
            constraint_tcon: Any | None = None,
            constraint_precond_active: Any | None = None,
            constraint_tcon_active: Any | None = None,
            iter_idx: int | None = None,
        ):
            return _compute_forces_impl(
                state,
                include_edge=include_edge,
                include_edge_residual=include_edge_residual,
                zero_m1=zero_m1,
                freeb_bsqvac_half=freeb_bsqvac_half,
                constraint_rcon0=constraint_rcon0,
                constraint_zcon0=constraint_zcon0,
                constraint_precond_diag=constraint_precond_diag,
                constraint_tcon=constraint_tcon,
                constraint_precond_active=constraint_precond_active,
                constraint_tcon_active=constraint_tcon_active,
                iter_idx=None,
            )

        _compute_forces = _select_compute_forces_callable(
            _compute_forces_nodump,
            differentiating_scan=bool(differentiating_scan),
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
        jit_precompile=bool(jit_precompile),
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

    if precompile_only:
        return _precompile_only_residual_iter_result(result_type=SolveVmecResidualResult, state=state0)

    def _iter_idx_for_dump(it: int | None) -> int | None:
        return None if jit_forces else it

    warmup_iters = int(jit_warmup_iters) if bool(jit_forces) else 0

    def _compute_forces_iter(
        state: VMECState,
        *,
        include_edge: bool,
        include_edge_residual: bool | None = None,
        zero_m1: Any,
        freeb_bsqvac_half: Any | None = None,
        constraint_rcon0: Any | None = None,
        constraint_zcon0: Any | None = None,
        constraint_precond_diag: tuple[Any, Any] | None = None,
        constraint_tcon: Any | None = None,
        constraint_precond_active: Any | None = None,
        constraint_tcon_active: Any | None = None,
        iter_idx: int | None = None,
        iter2: int | None = None,
    ):
        force_kwargs = {
            "include_edge": include_edge,
            "include_edge_residual": include_edge_residual,
            "zero_m1": zero_m1,
            "freeb_bsqvac_half": freeb_bsqvac_half,
            "constraint_rcon0": constraint_rcon0,
            "constraint_zcon0": constraint_zcon0,
            "constraint_precond_diag": constraint_precond_diag,
            "constraint_tcon": constraint_tcon,
            "constraint_precond_active": constraint_precond_active,
            "constraint_tcon_active": constraint_tcon_active,
            "iter_idx": iter_idx,
        }
        if warmup_iters > 0 and (iter2 is not None) and (int(iter2) <= warmup_iters):
            if has_jax():
                import jax

                with jax.disable_jit():
                    return _compute_forces_impl(state, **force_kwargs)
            return _compute_forces_impl(state, **force_kwargs)
        # NumPy fast path: use pure-NumPy force computation when available.
        # This eliminates all JAX dispatch overhead from the per-iteration loop.
        if _compute_forces_np is not None:
            return _compute_forces_np(state, **force_kwargs)
        if freeb_bsqvac_half is None:
            force_kwargs = {key: value for key, value in force_kwargs.items() if key != "freeb_bsqvac_half"}
        return _compute_forces(state, **force_kwargs)

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

    def _tomnsps_to_numpy_host(frzl: Any) -> TomnspsRZL:
        """Materialize a force block on the host for tiny scalar reductions."""

        def _host_array(value):
            if value is None:
                return None
            return np.asarray(jax.device_get(value))

        return TomnspsRZL(
            frcc=_host_array(frzl.frcc),
            frss=_host_array(frzl.frss),
            fzsc=_host_array(frzl.fzsc),
            fzcs=_host_array(frzl.fzcs),
            flsc=_host_array(frzl.flsc),
            flcs=_host_array(frzl.flcs),
            frsc=_host_array(getattr(frzl, "frsc", None)),
            frcs=_host_array(getattr(frzl, "frcs", None)),
            fzcc=_host_array(getattr(frzl, "fzcc", None)),
            fzss=_host_array(getattr(frzl, "fzss", None)),
            flcc=_host_array(getattr(frzl, "flcc", None)),
            flss=_host_array(getattr(frzl, "flss", None)),
        )

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
                scan_device_runtime_type=ScanDeviceRuntime,
                scan_convergence_predicate_type=ScanConvergencePredicate,
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

    _record_compute_force_timing = partial(
        _runtime_record_compute_force_timing,
        timing_enabled=bool(timing_enabled),
        timing_stats=timing_stats,
        perf_counter=time.perf_counter,
        block_until_ready=jax.block_until_ready if has_jax() else None,
    )

    history_lists = _new_residual_iter_histories()
    (
        (w_history, fsqr2_history, fsqz2_history, fsql2_history),
        (r00_history, z00_history, wb_history, wp_history, w_vmec_history),
        (fsqr1_history, fsqz1_history, fsql1_history, fsq1_history),
        (rz_norm_history, f_norm1_history, gcr2_p_history, gcz2_p_history, gcl2_p_history),
        (step_status_history, restart_reason_history, pre_restart_reason_history),
        (time_step_history, res0_history, res1_history, fsq_prev_history),
        (bad_growth_streak_history, iter1_history, iter2_history),
        (include_edge_history, zero_m1_history),
        (freeb_ivac_history, freeb_ivacskip_history, freeb_full_update_history),
        (
        freeb_nestor_reused_history,
        freeb_nestor_source_reused_history,
        freeb_nestor_provider_allows_source_reuse_history,
        freeb_nestor_bnormal_rms_history,
        freeb_nestor_gsource_rms_history,
        freeb_nestor_bsqvac_rms_history,
        freeb_nestor_solve_time_history,
        freeb_nestor_sample_time_history,
        freeb_nestor_trial_reused_history,
        freeb_nestor_trial_solve_time_history,
        freeb_nestor_trial_sample_time_history,
        freeb_nestor_trial_failed_history,
        ),
        (dt_eff_history, update_rms_history, w_curr_history, w_try_history, w_try_ratio_history),
        (restart_path_history, adjoint_step_trace_history),
        (min_tau_history, max_tau_history, bad_jacobian_history),
        (grad_rms_history, step_history),
    ) = history_lists.solver_alias_groups()

    def _append_badjac_history(min_tau_value: float, max_tau_value: float, bad_flag: bool) -> None:
        if track_history:
            min_tau_history.append(float(min_tau_value))
            max_tau_history.append(float(max_tau_value))
            bad_jacobian_history.append(int(bool(bad_flag)))

    _history_record_lists = history_lists.record_lists(
        free_boundary_enabled=bool(free_boundary_enabled),
    )
    _terminal_history_lists = history_lists.terminal_lists(
        free_boundary_enabled=bool(free_boundary_enabled),
    )

    r00_last = float("nan")
    z00_last = float("nan")
    wb_last = float("nan")
    wp_last = float("nan")
    w_vmec_last = float("nan")

    # Conjugate-gradient-like time-stepping state.
    controller_constants = _default_vmec2000_controller_constants()
    time_step = float(step_size)
    k_ndamp = controller_constants.ndamp
    inv_tau = [0.15 / time_step] * k_ndamp
    fsq_prev = 1.0
    fsq0_prev = 1.0
    _initial_velocity = _initial_residual_velocity_state(
        state=state,
        mpol=mpol,
        nrange=nrange,
        host_update_assembly=bool(host_update_assembly),
        reference_mode=bool(reference_mode),
    )
    (
        vRcc,
        vRss,
        vRsc,
        vRcs,
        vZsc,
        vZcs,
        vZcc,
        vZss,
        vLsc,
        vLcs,
        vLcc,
        vLss,
    ) = _initial_velocity.velocities
    flip_sign = float(initial_flip_sign)
    max_coeff_delta_rms = _initial_velocity.max_coeff_delta_rms
    max_update_rms = _initial_velocity.max_update_rms
    ijacob = 0
    bad_resets = 0
    iter1 = 1
    # VMEC runvmec/funct3d cadence starts free-boundary control at ivac=0.
    # Starting at -1 delays vacuum turn-on by one accepted iteration.
    # VMEC initializes ivac=-1 (reset_params.f), then promotes to 0/1/...
    # once free-boundary activation criteria are met.
    freeb_ivac = -1
    freeb_ivacskip = 0
    freeb_nestor_runtime: NestorRuntimeState | None = None
    freeb_bsqvac_half_current = None
    freeb_nestor_trace_current = None
    freeb_last_model = "none"
    freeb_last_diagnostics: dict[str, Any] = {}
    freeb_plascur = 0.0
    try:
        icurv_arr = np.asarray(getattr(wout_like, "icurv", np.asarray([0.0], dtype=float)), dtype=float)
        if icurv_arr.size > 0:
            freeb_plascur = float((2.0 * np.pi) * icurv_arr[-1])
    except Exception:
        freeb_plascur = 0.0

    _vmec_freeb_plascur_from_bcovar = partial(
        _runtime_vmec_freeb_plascur_from_bcovar,
        plascur_edge_from_bcovar=plascur_edge_from_bcovar,
        trig=trig,
        wout=wout_like,
        s=s,
    )

    res0 = -1.0
    k_preconditioner_update_interval = controller_constants.preconditioner_update_interval
    state_checkpoint = state
    bad_growth_streak = 0
    # Restart trigger factors:
    # - bad_jacobian: time_step *= 0.9
    # - bad_progress: time_step /= 1.03
    restart_badjac_factor = controller_constants.restart_badjac_factor
    restart_badprog_factor = controller_constants.restart_badprog_factor
    huge_force_restart_count = 0
    res1 = -1.0
    vmec2000_fact = controller_constants.vmec2000_fact

    # Edge-force gating uses the *previous* iteration's residual (the first
    # iteration initializes forces to 1.0). Track that explicitly.
    prev_rz_fsq = 2.0

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
    (
        vmec2000_cache_valid,
        cache_precond_diag,
        cache_tcon,
        cache_norms,
        cache_rz_scale,
        cache_l_scale,
        cache_rz_norm,
        cache_f_norm1,
        cache_prec_rz_mats,
        cache_prec_rz_jmax,
        cache_prec_lam_prec,
        cache_prec_faclam,
        cache_prec_lam_debug,
    ) = _empty_preconditioner_cache_snapshot()
    cache_constraint_rcon0 = None
    cache_constraint_zcon0 = None

    def _clear_preconditioner_cache_locals() -> None:
        nonlocal vmec2000_cache_valid
        nonlocal cache_precond_diag, cache_tcon, cache_norms, cache_rz_scale, cache_l_scale
        nonlocal cache_rz_norm, cache_f_norm1
        nonlocal cache_prec_rz_mats, cache_prec_rz_jmax, cache_prec_lam_prec
        nonlocal cache_prec_faclam, cache_prec_lam_debug

        (
            vmec2000_cache_valid,
            cache_precond_diag,
            cache_tcon,
            cache_norms,
            cache_rz_scale,
            cache_l_scale,
            cache_rz_norm,
            cache_f_norm1,
            cache_prec_rz_mats,
            cache_prec_rz_jmax,
            cache_prec_lam_prec,
            cache_prec_faclam,
            cache_prec_lam_debug,
        ) = _empty_preconditioner_cache_snapshot()

    bcovar_update_history = history_lists["bcovar_update_history"]
    _rollback_history_lists = history_lists.rollback_lists()
    iter_offset = 0

    if resume_state is not None:
        iter_offset = int(resume_state.get("iter_offset", iter_offset))
        time_step = float(resume_state.get("time_step", time_step))
        inv_tau = list(resume_state.get("inv_tau", inv_tau))
        fsq_prev = float(resume_state.get("fsq_prev", fsq_prev))
        fsq0_prev = float(resume_state.get("fsq0_prev", fsq0_prev))
        flip_sign = float(resume_state.get("flip_sign", flip_sign))
        iter1 = int(resume_state.get("iter1", iter1))
        ijacob = int(resume_state.get("ijacob", ijacob))
        bad_resets = int(resume_state.get("bad_resets", bad_resets))
        res0 = float(resume_state.get("res0", res0))
        res1 = float(resume_state.get("res1", res1))
        prev_rz_fsq = float(resume_state.get("prev_rz_fsq", prev_rz_fsq))
        bad_growth_streak = int(resume_state.get("bad_growth_streak", bad_growth_streak))
        huge_force_restart_count = int(resume_state.get("huge_force_restart_count", huge_force_restart_count))

        if "vRcc" in resume_state:
            _as_velocity = np.asarray if bool(host_update_assembly) else jnp.asarray
            vRcc = _as_velocity(resume_state["vRcc"])
            vRss = _as_velocity(resume_state.get("vRss", vRss))
            vZsc = _as_velocity(resume_state.get("vZsc", vZsc))
            vZcs = _as_velocity(resume_state.get("vZcs", vZcs))
            vLsc = _as_velocity(resume_state.get("vLsc", vLsc))
            vLcs = _as_velocity(resume_state.get("vLcs", vLcs))

        state_checkpoint = resume_state.get("state_checkpoint", state)
        vmec2000_cache_valid = bool(resume_state.get("vmec2000_cache_valid", vmec2000_cache_valid))
        cache_precond_diag = resume_state.get("cache_precond_diag", cache_precond_diag)
        cache_tcon = resume_state.get("cache_tcon", cache_tcon)
        cache_norms = resume_state.get("cache_norms", cache_norms)
        cache_rz_scale = resume_state.get("cache_rz_scale", cache_rz_scale)
        cache_l_scale = resume_state.get("cache_l_scale", cache_l_scale)
        cache_rz_norm = resume_state.get("cache_rz_norm", cache_rz_norm)
        cache_f_norm1 = resume_state.get("cache_f_norm1", cache_f_norm1)
        cache_prec_rz_mats = resume_state.get("cache_prec_rz_mats", cache_prec_rz_mats)
        cache_prec_rz_jmax = resume_state.get("cache_prec_rz_jmax", cache_prec_rz_jmax)
        cache_prec_lam_prec = resume_state.get("cache_prec_lam_prec", cache_prec_lam_prec)
        cache_prec_faclam = resume_state.get("cache_prec_faclam", cache_prec_faclam)
        cache_prec_lam_debug = resume_state.get("cache_prec_lam_debug", cache_prec_lam_debug)
        cache_constraint_rcon0 = resume_state.get("cache_constraint_rcon0", cache_constraint_rcon0)
        cache_constraint_zcon0 = resume_state.get("cache_constraint_zcon0", cache_constraint_zcon0)
        if free_boundary_enabled:
            freeb_ivac = int(resume_state.get("freeb_ivac", freeb_ivac))
            freeb_ivacskip = int(resume_state.get("freeb_ivacskip", freeb_ivacskip))
            freeb_nvacskip = max(1, int(resume_state.get("freeb_nvacskip", freeb_nvacskip)))
            freeb_nvskip0 = max(1, int(resume_state.get("freeb_nvskip0", freeb_nvskip0)))
            freeb_last_model = str(resume_state.get("freeb_model", freeb_last_model))

    _apply_vmec_scale_m1_precond_rhs = partial(
        _scale_m1_precond_rhs_from_mats,
        lconm1=getattr(cfg, "lconm1", True),
        mpol=int(cfg.mpol),
        host_update_assembly=host_update_assembly,
    )

    def _refresh_preconditioner_cache(k, *, iter2: int):
        nonlocal cache_prec_lam_prec, cache_prec_faclam, cache_prec_lam_debug
        nonlocal cache_prec_rz_mats, cache_prec_rz_jmax

        refresh = _precond_payload_facade.refresh_preconditioner_cache_runtime(
            k=k,
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
            vmec2000_cache_valid=bool(vmec2000_cache_valid),
            need_bcovar_update=bool(need_bcovar_update),
            precond_cache_seeded_from_bcovar_update=bool(precond_cache_seeded_from_bcovar_update),
            precond_expected_jmax=int(precond_expected_jmax),
            precond_jmax_override=precond_jmax_override,
            preconditioner_use_precomputed_tridi=preconditioner_use_precomputed_tridi_policy,
            preconditioner_use_lax_tridi=preconditioner_use_lax_tridi_policy,
            cache_prec_lam_prec=cache_prec_lam_prec,
            cache_prec_faclam=cache_prec_faclam,
            cache_prec_lam_debug=cache_prec_lam_debug,
            cache_prec_rz_mats=cache_prec_rz_mats,
            cache_prec_rz_jmax=cache_prec_rz_jmax,
        )
        cache_prec_lam_prec = refresh.cache_prec_lam_prec
        cache_prec_faclam = refresh.cache_prec_faclam
        cache_prec_lam_debug = refresh.cache_prec_lam_debug
        cache_prec_rz_mats = refresh.cache_prec_rz_mats
        cache_prec_rz_jmax = refresh.cache_prec_rz_jmax
        return (
            refresh.lam_prec,
            refresh.mats,
            refresh.jmax,
            refresh.need_lam_prec,
            refresh.need_lamcal,
            refresh.cache_update_trace,
        )

    def _pop_iteration_histories() -> None:
        _pop_residual_iter_rollback_histories(_rollback_history_lists)

    _maybe_dump_time_control = _maybe_dump_time_control_record
    _dump_time_control_trace = _dump_time_control_trace_record
    _maybe_dump_checkpoint = _maybe_dump_checkpoint_record
    _dump_freeb_control_trace = _dump_freeb_control_trace_record
    _dump_freeb_axis_trace = _dump_freeb_axis_trace_record
    _dump_evolve_trace = partial(_maybe_dump_evolve_trace_record, static=static)

    # VMEC `eqsolve`: if the initial Jacobian changes sign, improve the axis
    # guess *before* the first iteration (no extra iter1). This aligns the
    # zero_m1 gating and time-control history with VMEC2000.
    axis_setup = _run_initial_axis_reset_setup(
        state=state,
        axis_reset_done=bool(axis_reset_done),
        ijacob=int(ijacob),
        state_checkpoint=state_checkpoint,
        velocities=(vRcc, vRss, vZsc, vZcs, vLsc, vLcs),
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
    ijacob = int(axis_setup.ijacob)
    state_checkpoint = axis_setup.state_checkpoint
    vRcc, vRss, vZsc, vZcs, vLsc, vLcs = axis_setup.velocities
    res0 = float(axis_setup.res0)
    res1 = float(axis_setup.res1)
    prev_rz_fsq = float(axis_setup.prev_rz_fsq)
    if axis_setup.reset_applied:
        _clear_preconditioner_cache_locals()
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
            iter_since_restart = iter2 - iter1
            fsq_prev_before = fsq_prev
            fsq0_prev_before = fsq0_prev
            pre_restart_reason = "none"
            if time_step_report_hold is None:
                time_step_report_hold = float(time_step)
            if free_boundary_enabled:
                # Keep free-boundary cadence fixed for this `iter2` across
                # retry/restart passes in the inner while-loop.
                fsq_rz_prev = float(prev_rz_fsq) if np.isfinite(prev_rz_fsq) else 1.0
                controls_cached_before = freeb_controls_cached is not None
                if freeb_controls_cached is None:
                    freeb_ivac, freeb_ivacskip, freeb_nvacskip = _free_boundary_iter_controls_vmec(
                        iter2=int(iter2),
                        iter1=int(iter1),
                        ivac=int(freeb_ivac),
                        nvacskip=int(freeb_nvacskip),
                        nvskip0=int(freeb_nvskip0),
                        fsq_rz_prev=float(fsq_rz_prev),
                        activate_fsq=free_boundary_activate_fsq,
                    )
                    freeb_controls_cached = (
                        int(freeb_ivac),
                        int(freeb_ivacskip),
                        int(freeb_nvacskip),
                    )
                else:
                    freeb_ivac, freeb_ivacskip, freeb_nvacskip = freeb_controls_cached
                _dump_freeb_control_trace(
                    iter2=int(iter2),
                    iter1=int(iter1),
                    ivac=int(freeb_ivac),
                    ivacskip=int(freeb_ivacskip),
                    nvacskip=int(freeb_nvacskip),
                    fsq_rz_prev=float(fsq_rz_prev),
                    cached=bool(controls_cached_before),
                )
            # VMEC vacuum.f promotes ivac=0 -> 1 inside the vacuum solve.
            # Keep both values: pre-vacuum (`freeb_ivac`) for cadence/calls,
            # and post-vacuum effective (`freeb_ivac_effective`) for force/
            # residue gating in this same iteration.
            freeb_turnon_iter = bool(free_boundary_enabled) and (int(freeb_ivac) == 0) and (int(freeb_ivacskip) == 0)
            freeb_ivac_effective = int(freeb_ivac)
            if freeb_turnon_iter:
                freeb_ivac_effective = 1
            if vmec2000_control:
                # VMEC2000 `constrain_m1` logic (residue.f90):
                #   zero gcz(m=1) if (fsqz_prev < 1e-6) OR (iter2 < 2).
                fsqz_prev = float(fsqz2_history[-1]) if fsqz2_history else 1.0
                zero_m1_val = 1.0 if (iter2 < 2) or (fsqz_prev < 1.0e-6) else 0.0
            else:
                # A conservative heuristic early in a restart window.
                zero_m1_val = (
                    1.0 if (iter_since_restart < 2) or (len(fsqz2_history) and fsqz2_history[-1] < 1e-6) else 0.0
                )
            if host_update_assembly and _jnp_zero_m1_0 is not None:
                # Use pre-cached JAX scalars to avoid jnp.asarray dispatch + dtype
                # lookup every iteration (saves 2 apply_primitive calls per iter).
                zero_m1 = _jnp_zero_m1_1 if zero_m1_val > 0.5 else _jnp_zero_m1_0
            else:
                zero_m1 = jnp.asarray(zero_m1_val, dtype=jnp.asarray(state.Rcos).dtype)
            if vmec2000_control:
                # VMEC2000 keeps the core R/Z residual assembly on the
                # interior mesh; free-boundary coupling enters through the
                # dedicated edge `rbsq` terms in `forces.f`, not by enabling
                # generic edge residual rows.
                include_edge = _env_freeb_include_edge
            else:
                include_edge = bool(iter_since_restart < 50) and (float(prev_rz_fsq) < 1e-6)
            if track_history:
                include_edge_history.append(int(bool(include_edge)))
            # Residual transform edge handling:
            # VMEC tomnsp_mod uses jmax=ns once free-boundary vacuum is on
            # (ivac >= 1), independent of residue's `jedge` scalar gating.
            # Keep `include_edge` for scalar gating/diagnostics, but include
            # edge rows in the transform when vacuum coupling is active.
            include_edge_residual = bool(include_edge)
            if bool(free_boundary_enabled) and int(freeb_ivac_effective) >= 1:
                include_edge_residual = True
            if _env_force_edge_residual in ("1", "true", "yes"):
                include_edge_residual = True
            precond_jmax_override: int | None = None
            if bool(vmec2000_control) and bool(free_boundary_enabled) and (int(freeb_ivac_effective) >= 1):
                # VMEC scalfor: jmax=ns once free-boundary vacuum is active.
                precond_jmax_override = int(s.shape[0])
            precond_expected_jmax = (
                int(precond_jmax_override) if (precond_jmax_override is not None) else max(int(s.shape[0]) - 1, 1)
            )
            # `zero_m1` originates from host control flow, so keep the history
            # without forcing an unnecessary device synchronization.
            if track_history:
                zero_m1_history.append(int(zero_m1_val > 0.5))

            need_bcovar_update = bool(vmec2000_control) and (
                (not bool(vmec2000_cache_valid))
                or bool(force_bcovar_update)
                or ((iter2 - iter1) % k_preconditioner_update_interval == 0)
            )
            precond_cache_seeded_from_bcovar_update = False
            precond_refresh_seed_time_in_residual_metrics = 0.0
            force_bcovar_update = False
            bcovar_update_history.append(int(bool(need_bcovar_update)))

            use_cached_precond = (
                bool(vmec2000_control) and bool(vmec2000_cache_valid) and (not bool(need_bcovar_update))
            )
            constraint_precond_diag = (
                cache_precond_diag if (use_cached_precond and cache_precond_diag is not None) else zero_precond_diag
            )
            # VMEC updates tcon only when refreshing the 1D preconditioner
            # blocks; between refreshes it reuses the last tcon profile.
            constraint_tcon_override = cache_tcon if (use_cached_precond and cache_tcon is not None) else zero_tcon
            if host_update_assembly and _jnp_true_bool is not None:
                # Use pre-cached bool scalars — avoids 2 jnp.asarray dispatches/iter.
                constraint_precond_active = _jnp_true_bool if use_cached_precond else _jnp_false_bool
                constraint_tcon_active = _jnp_true_bool if use_cached_precond else _jnp_false_bool
            else:
                constraint_precond_active = jnp.asarray(use_cached_precond, dtype=bool)
                constraint_tcon_active = jnp.asarray(use_cached_precond, dtype=bool)

            # Free-boundary WP2 scaffold: run/update the NESTOR-like external
            # vacuum solve and couple bsqvac on the edge slice into bcovar.
            freeb_bsqvac_half_current = None
            freeb_nestor_trace_current = None
            freeb_reused = False
            freeb_solve_time = 0.0
            freeb_sample_time = 0.0
            freeb_plascur_for_bsqvac = float(freeb_plascur)
            freeb_coupling = _free_boundary_nestor_iteration_coupling(
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
                source_reused_history=freeb_nestor_source_reused_history,
                provider_allows_source_reuse_history=freeb_nestor_provider_allows_source_reuse_history,
                bnormal_rms_history=freeb_nestor_bnormal_rms_history,
                gsource_rms_history=freeb_nestor_gsource_rms_history,
                bsqvac_rms_history=freeb_nestor_bsqvac_rms_history,
            )
            freeb_bsqvac_half_current = freeb_coupling.bsqvac_half_current
            freeb_nestor_runtime = freeb_coupling.runtime
            freeb_nestor_trace_current = freeb_coupling.trace_arrays
            freeb_reused = freeb_coupling.reused
            freeb_solve_time = freeb_coupling.solve_time
            freeb_sample_time = freeb_coupling.sample_time
            freeb_last_model = freeb_coupling.last_model
            freeb_last_diagnostics = freeb_coupling.last_diagnostics
            freeb_ivac = freeb_coupling.ivac
            freeb_ivac_effective = freeb_coupling.ivac_effective
            freeb_controls_cached = freeb_coupling.controls_cached

            _freeb_bsqvac_half_for_trial_state = partial(
                _runtime_freeb_trial_bsqvac_half,
                free_boundary_enabled=bool(free_boundary_enabled),
                freeb_couple_edge=bool(freeb_couple_edge),
                freeb_bsqvac_half_current=freeb_bsqvac_half_current,
                external_field_provider_kind=external_field_provider_kind,
                external_field_provider_static=external_field_provider_static,
                external_field_provider_params=external_field_provider_params,
                freeb_ivac_effective=int(freeb_ivac_effective),
                freeb_nestor_runtime=freeb_nestor_runtime,
                static=static,
                iter2=int(iter2),
                freeb_plascur=float(freeb_plascur),
                env_freeb_raise=bool(_env_freeb_raise),
                nestor_external_only_step_func=nestor_external_only_step,
                edge_bsqvac_from_nestor_func=_edge_bsqvac_from_nestor,
                trial_reused_history=freeb_nestor_trial_reused_history,
                trial_solve_time_history=freeb_nestor_trial_solve_time_history,
                trial_sample_time_history=freeb_nestor_trial_sample_time_history,
                trial_failed_history=freeb_nestor_trial_failed_history,
            )

            def _trial_residual_total(
                candidate_state: VMECState,
                freeb_bsqvac_half_trial,
                *,
                zero_m1_value,
                timing_label: str | None = None,
            ) -> float:
                t_trial_force_start = time.perf_counter() if (timing_label and timing_detail_enabled) else None
                _, _, gcr2_t, gcz2_t, gcl2_t, _, _, norms_t = _compute_forces_iter(
                    candidate_state,
                    include_edge=include_edge,
                    zero_m1=zero_m1_value,
                    freeb_bsqvac_half=freeb_bsqvac_half_trial,
                    constraint_precond_diag=constraint_precond_diag,
                    constraint_tcon=constraint_tcon_override,
                    constraint_precond_active=constraint_precond_active,
                    constraint_tcon_active=constraint_tcon_active,
                    iter2=iter2,
                )
                if timing_label:
                    _record_compute_force_timing(timing_label, t_trial_force_start, gcr2_t)
                fsqr_t, fsqz_t, fsql_t = _residual_fsq_from_norms(
                    norms_t,
                    gcr2=gcr2_t,
                    gcz2=gcz2_t,
                    gcl2=gcl2_t,
                )
                return float(np.asarray(fsqr_t + fsqz_t + fsql_t))

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

            if timing_enabled and t_iteration_prepare_start is not None:
                timing_stats["iteration_prepare"] += time.perf_counter() - float(t_iteration_prepare_start)
            t_compute_start = time.perf_counter() if timing_enabled else None
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
                iter_idx=_iter_idx_for_dump(iter2),
                iter2=iter2,
            )
            if bool(free_boundary_enabled):
                freeb_plascur = _vmec_freeb_plascur_from_bcovar(k.bc, freeb_plascur)
                try:
                    pr1_axis = np.asarray(k.pr1_even, dtype=float)
                    pz1_axis = np.asarray(k.pz1_even, dtype=float)
                    if pr1_axis.ndim >= 3 and pz1_axis.ndim >= 3:
                        _dump_freeb_axis_trace(
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
            if timing_enabled:
                _record_compute_force_timing("main", t_compute_start, gcr2)
            t_residual_metrics_start = time.perf_counter() if timing_enabled else None
            norms_used = (
                cache_norms
                if (bool(vmec2000_control) and bool(vmec2000_cache_valid) and (not bool(need_bcovar_update)))
                else norms_current
            )
            use_host_residual_metrics = (
                bool(startup_policy.host_residual_metrics_on_accelerator)
                and (not bool(host_update_assembly))
                and (jax.default_backend() != "cpu")
                and (not _tree_has_tracer((gcr2, gcz2, gcl2, norms_used)))
            )
            if host_update_assembly:
                # NumPy path: gcr2/gcz2/gcl2 already synced by block_until_ready above.
                # float() on synced JAX scalars is fast (no blocking). Avoids 5 JAX dispatches.
                _gcr2_f = float(gcr2)
                _gcz2_f = float(gcz2)
                _gcl2_f = float(gcl2)
                _fnorm_f = float(norms_used.fnorm)
                _fnormL_f = float(norms_used.fnormL)
                _r1_f = float(norms_used.r1)
                fsqr = _r1_f * _fnorm_f * _gcr2_f
                fsqz = _r1_f * _fnorm_f * _gcz2_f
                fsql = _fnormL_f * _gcl2_f
            elif use_host_residual_metrics:
                (
                    _gcr2_f,
                    _gcz2_f,
                    _gcl2_f,
                    _fnorm_f,
                    _fnormL_f,
                    _r1_f,
                ) = _device_get_floats(
                    gcr2,
                    gcz2,
                    gcl2,
                    norms_used.fnorm,
                    norms_used.fnormL,
                    norms_used.r1,
                )
                fsqr = _r1_f * _fnorm_f * _gcr2_f
                fsqz = _r1_f * _fnorm_f * _gcz2_f
                fsql = _fnormL_f * _gcl2_f
            else:
                fsqr = norms_used.r1 * norms_used.fnorm * gcr2
                fsqz = norms_used.r1 * norms_used.fnorm * gcz2
                fsql = norms_used.fnormL * gcl2
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
            if bool(vmec2000_control) and bool(vmec2000_cache_valid) and (not bool(need_bcovar_update)):
                rz_scale = cache_rz_scale
                l_scale = cache_l_scale
            preconditioner_cache_update_trace = False
            if bool(vmec2000_control) and bool(need_bcovar_update):
                if constraint_tcon0 is None or float(constraint_tcon0) == 0.0:
                    cache_precond_diag = None
                    cache_tcon = jnp.zeros((int(s.shape[0]),), dtype=jnp.asarray(state.Rcos).dtype)
                else:
                    from vmec_jax.vmec_constraints import precondn_diag_axd1_from_bcovar

                    if host_update_assembly and (not _tree_has_tracer(k)) and (not _tree_has_tracer(s)):
                        from vmec_jax.vmec_numpy_forces import _numpy_module_patch as _hot_numpy_patch

                        with _hot_numpy_patch():
                            ard1, azd1 = precondn_diag_axd1_from_bcovar(
                                trig=trig,
                                s=s,
                                bsq=k.bc.bsq,
                                r12=k.bc.jac.r12,
                                sqrtg=k.bc.jac.sqrtg,
                                ru12=k.bc.jac.ru12,
                                zu12=k.bc.jac.zu12,
                            )
                    else:
                        ard1, azd1 = precondn_diag_axd1_from_bcovar(
                            trig=trig,
                            s=s,
                            bsq=k.bc.bsq,
                            r12=k.bc.jac.r12,
                            sqrtg=k.bc.jac.sqrtg,
                            ru12=k.bc.jac.ru12,
                            zu12=k.bc.jac.zu12,
                        )
                    cache_precond_diag = (ard1, azd1)
                    cache_tcon = np.asarray(k.tcon) if host_update_assembly else jnp.asarray(k.tcon)
                cache_norms = norms_used
                cache_rz_scale = rz_scale
                cache_l_scale = l_scale
                if host_update_assembly:
                    # NumPy path: avoids JAX dispatch + XLA blocking for fnorm1.
                    cache_rz_norm = _rz_norm_np(state)  # Python float
                    cache_f_norm1 = (1.0 / cache_rz_norm) if cache_rz_norm != 0.0 else float("inf")
                else:
                    cache_rz_norm = _rz_norm(state)
                    cache_f_norm1 = jnp.where(
                        jnp.asarray(cache_rz_norm) != 0.0,
                        1.0 / jnp.asarray(cache_rz_norm),
                        jnp.asarray(float("inf"), dtype=jnp.asarray(cache_rz_norm).dtype),
                    )
                if not bool(cfg.lasym):
                    t_precond_refresh_seed_start = time.perf_counter() if timing_enabled else None
                    cache_prec_lam_prec = _lambda_preconditioner(k.bc)
                    cache_prec_faclam = None
                    cache_prec_lam_debug = None
                    mats, _jmin, jmax = _rz_preconditioner_matrices_local(
                        bc=k.bc,
                        k=k,
                        jmax_override=precond_jmax_override,
                        use_precomputed=preconditioner_use_precomputed_tridi_policy,
                        use_lax_tridi=preconditioner_use_lax_tridi_policy,
                    )
                    cache_prec_rz_mats = mats
                    cache_prec_rz_jmax = None if _tree_has_tracer(k) else int(jmax)
                    precond_cache_seeded_from_bcovar_update = cache_prec_rz_jmax is not None
                    preconditioner_cache_update_trace = True
                    if timing_enabled and t_precond_refresh_seed_start is not None:
                        seed_dt = time.perf_counter() - float(t_precond_refresh_seed_start)
                        precond_refresh_seed_time_in_residual_metrics += seed_dt
                        timing_stats["precond_refresh_seed"] += seed_dt
                        timing_stats["precond_refresh"] += seed_dt
                        timing_stats["preconditioner"] += seed_dt
                        timing_stats["precond_refresh_calls"] = int(timing_stats["precond_refresh_calls"]) + 1
                vmec2000_cache_valid = True
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
                state = state_checkpoint
                (
                    vRcc,
                    vRss,
                    vZsc,
                    vZcs,
                    vLsc,
                    vLcs,
                    vRsc,
                    vRcs,
                    vZcc,
                    vZss,
                    vLcc,
                    vLss,
                ) = _zero_velocity_blocks_like(vRcc, vRss, vZsc, vZcs, vLsc, vLcs, vRsc, vRcs, vZcc, vZss, vLcc, vLss)
                time_step_report_hold = float(time_step)
                ijacob += 1
                if _free_boundary_turnon_resets_iter1_immediately(
                    lthreed=bool(cfg.lthreed),
                    lasym=bool(cfg.lasym),
                ):
                    iter1 = int(iter2)
                bad_growth_streak = 0
                inv_tau = [0.15 / max(float(time_step), 1e-12)] * k_ndamp
                freeb_turnon_applied = True
            fsq0_curr = fsqr_f + fsqz_f + fsql_f
            prev_rz_fsq_before = prev_rz_fsq
            prev_rz_fsq = _free_boundary_prev_rz_fsq_next(
                prev_fsq_before=prev_rz_fsq_before,
                fsq_rz_curr=fsqr_f + fsqz_f,
                turnon_restart=bool(free_boundary_enabled) and bool(freeb_turnon_iter) and bool(freeb_turnon_applied),
                preserve_turnon_restart=bool(free_boundary_enabled) and bool(cfg.lthreed),
            )

            w_history.append(fsq0_curr)
            fsqr2_history.append(fsqr_f)
            fsqz2_history.append(fsqz_f)
            fsql2_history.append(fsql_f)
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
            if track_history:
                r00_history.append(r00_last)
                z00_history.append(z00_last)
                wb_history.append(wb_last)
                wp_history.append(wp_last)
                w_vmec_history.append(w_vmec_last)

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
                and (not bool(dump_ptau_state))
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
            frzl_lam_pre = None
            preconditioner_outputs_scaled = False
            preconditioner_fsq1_ready = False
            use_fused_precond_output_scaling = (not bool(host_update_assembly)) and jax.default_backend() != "cpu"
            if bool(vmec2000_control) and bool(cfg.lthreed):
                lam_prec, mats, jmax, need_lam_prec, need_lamcal, preconditioner_cache_update_trace = (
                    _refresh_preconditioner_cache(k, iter2=int(iter2))
                )
                t_precond_apply_start = time.perf_counter() if timing_detail_enabled else None
                use_apply_payload_fusion = (
                    bool(use_fused_precond_output_scaling)
                    and need_lam_prec is False
                    and need_lamcal is False
                )
                frzl_rhs = _apply_vmec_scale_m1_precond_rhs(frzl, mats)
                _apply_rz_preconditioner_current = partial(
                    _rz_preconditioner_apply_local,
                    mats=mats,
                    jmax=jmax,
                    use_precomputed=preconditioner_use_precomputed_tridi_policy,
                    use_lax_tridi=preconditioner_use_lax_tridi_policy,
                )
                if use_apply_payload_fusion:
                    _, f_norm1 = _cached_or_current_f_norm1_jax(
                        vmec2000_control=bool(vmec2000_control),
                        vmec2000_cache_valid=bool(vmec2000_cache_valid),
                        need_bcovar_update=bool(need_bcovar_update),
                        cache_rz_norm=cache_rz_norm,
                        cache_f_norm1=cache_f_norm1,
                        state=state,
                        rz_norm_func=_rz_norm,
                    )
                    _precond_payload = _preconditioner_apply_payload_fused(
                        frzl_in=frzl_rhs,
                        mats=mats,
                        jmax=jmax,
                        cfg=cfg,
                        lam_prec=lam_prec,
                        w_mode_mn=w_mode_mn,
                        lambda_update_scale_j=lambda_update_scale_j,
                        f_norm1=f_norm1,
                        delta_s=delta_s,
                        s=s,
                        use_precomputed=preconditioner_use_precomputed_tridi_policy,
                        use_lax_tridi=preconditioner_use_lax_tridi_policy,
                        apply_lambda_update_scale=(lambda_update_scale != 1.0),
                        vmec2000_control=bool(vmec2000_control),
                        lconm1=bool(getattr(static.cfg, "lconm1", True)),
                        include_control_ptau=accepted_control_ptau_arrays is not None,
                        control_ptau_arrays=accepted_control_ptau_arrays,
                        control_ptau_pshalf=_ptau_context.pshalf_jax,
                        control_ptau_ohs=_ptau_context.ohs_jax,
                    )
                    (
                        _precond_pre_blocks,
                        _precond_update_blocks,
                        _precond_diag,
                        accepted_control_ptau_payload,
                    ) = _split_preconditioner_apply_payload(_precond_payload)
                    (
                        (frcc, frss, fzsc, fzcs, flsc, flcs, frsc, frcs, fzcc, fzss, flcc, flss),
                        (
                            frcc_u,
                            frss_u,
                            fzsc_u,
                            fzcs_u,
                            flsc_u,
                            flcs_u,
                            frsc_u,
                            frcs_u,
                            fzcc_u,
                            fzss_u,
                            flcc_u,
                            flss_u,
                        ),
                        (gcr2_p, gcz2_p, gcl2_p, fsqr1_safe, fsqz1_safe, fsql1_safe, fsq1_safe),
                    ) = (_precond_pre_blocks, _precond_update_blocks, _precond_diag)
                    fsqr1 = fsqr1_safe
                    fsqz1 = fsqz1_safe
                    fsql1 = fsql1_safe
                    preconditioner_outputs_scaled = True
                    preconditioner_fsq1_ready = True
                else:
                    frzl_rz = _apply_rz_preconditioner_current(frzl_in=frzl_rhs)
                    frzl_lam_pre = frzl_rz
                if use_apply_payload_fusion and adjoint_trace and adjoint_trace_mode == "full":
                    # The fused GPU-oriented path returns only scaled update
                    # payloads.  Full accepted-trace replay also needs the raw
                    # R/Z-preconditioned force, so materialize it only for that
                    # opt-in diagnostic/validation mode.
                    frzl_rz = _apply_rz_preconditioner_current(frzl_in=frzl_rhs)
                if (not use_apply_payload_fusion) and host_update_assembly:
                    # NumPy path: avoids ~15 JAX dispatches (jnp.asarray, zeros_like, mul).
                    # Asymmetric (lasym) components default to None — the downstream
                    # mode-diag scaling uses _z (pre-allocated zeros) for None entries,
                    # avoiding 6 np.zeros_like allocations per iteration (~0.5s saving).
                    (frcc, frss, fzsc, fzcs, flsc, flcs, frsc, frcs, fzcc, fzss, flcc, flss) = (
                        _preconditioner_output_blocks_np(frzl_rz=frzl_rz, lam_prec=lam_prec)
                    )
                elif (not use_apply_payload_fusion) and use_fused_precond_output_scaling:
                    rz_norm, f_norm1 = _cached_or_current_f_norm1_jax(
                        vmec2000_control=bool(vmec2000_control),
                        vmec2000_cache_valid=bool(vmec2000_cache_valid),
                        need_bcovar_update=bool(need_bcovar_update),
                        cache_rz_norm=cache_rz_norm,
                        cache_f_norm1=cache_f_norm1,
                        state=state,
                        rz_norm_func=_rz_norm,
                    )
                    payload_outputs = _preconditioner_output_payload_jit(
                        apply_lambda_update_scale=(lambda_update_scale != 1.0),
                        vmec2000_control=bool(vmec2000_control),
                        lconm1=bool(getattr(static.cfg, "lconm1", True)),
                    )
                    (
                        (frcc, frss, fzsc, fzcs, flsc, flcs, frsc, frcs, fzcc, fzss, flcc, flss),
                        (
                            frcc_u,
                            frss_u,
                            fzsc_u,
                            fzcs_u,
                            flsc_u,
                            flcs_u,
                            frsc_u,
                            frcs_u,
                            fzcc_u,
                            fzss_u,
                            flcc_u,
                            flss_u,
                        ),
                        (gcr2_p, gcz2_p, gcl2_p, fsqr1_safe, fsqz1_safe, fsql1_safe, fsq1_safe),
                    ) = payload_outputs(frzl_rz, lam_prec, w_mode_mn, lambda_update_scale_j, f_norm1, delta_s, s)
                    fsqr1 = fsqr1_safe
                    fsqz1 = fsqz1_safe
                    fsql1 = fsql1_safe
                    preconditioner_outputs_scaled = True
                    preconditioner_fsq1_ready = True
                elif not use_apply_payload_fusion:
                    (frcc, frss, fzsc, fzcs, flsc, flcs, frsc, frcs, fzcc, fzss, flcc, flss) = (
                        _preconditioner_output_blocks_jax(frzl_rz=frzl_rz, lam_prec=lam_prec)
                    )
                if timing_detail_enabled and t_precond_apply_start is not None:
                    try:
                        if has_jax():
                            jax.block_until_ready(flsc)
                    except Exception:
                        pass
                    timing_stats["precond_apply"] += time.perf_counter() - float(t_precond_apply_start)
            elif not bool(cfg.lthreed):
                lam_prec, mats, jmax, need_lam_prec, need_lamcal, preconditioner_cache_update_trace = (
                    _refresh_preconditioner_cache(k, iter2=int(iter2))
                )
                t_precond_apply_start = time.perf_counter() if timing_detail_enabled else None
                use_apply_payload_fusion = (
                    bool(use_fused_precond_output_scaling)
                    and need_lam_prec is False
                    and need_lamcal is False
                )
                frzl_rhs = _apply_vmec_scale_m1_precond_rhs(frzl, mats) if bool(getattr(cfg, "lasym", False)) else frzl
                _apply_rz_preconditioner_current = partial(
                    _rz_preconditioner_apply_local,
                    mats=mats,
                    jmax=jmax,
                    use_precomputed=preconditioner_use_precomputed_tridi_policy,
                    use_lax_tridi=preconditioner_use_lax_tridi_policy,
                )
                if use_apply_payload_fusion:
                    _, f_norm1 = _cached_or_current_f_norm1_jax(
                        vmec2000_control=bool(vmec2000_control),
                        vmec2000_cache_valid=bool(vmec2000_cache_valid),
                        need_bcovar_update=bool(need_bcovar_update),
                        cache_rz_norm=cache_rz_norm,
                        cache_f_norm1=cache_f_norm1,
                        state=state,
                        rz_norm_func=_rz_norm,
                    )
                    _precond_payload = _preconditioner_apply_payload_fused(
                        frzl_in=frzl_rhs,
                        mats=mats,
                        jmax=jmax,
                        cfg=cfg,
                        lam_prec=lam_prec,
                        w_mode_mn=w_mode_mn,
                        lambda_update_scale_j=lambda_update_scale_j,
                        f_norm1=f_norm1,
                        delta_s=delta_s,
                        s=s,
                        use_precomputed=preconditioner_use_precomputed_tridi_policy,
                        use_lax_tridi=preconditioner_use_lax_tridi_policy,
                        apply_lambda_update_scale=(lambda_update_scale != 1.0),
                        vmec2000_control=bool(vmec2000_control),
                        lconm1=bool(getattr(static.cfg, "lconm1", True)),
                        include_control_ptau=accepted_control_ptau_arrays is not None,
                        control_ptau_arrays=accepted_control_ptau_arrays,
                        control_ptau_pshalf=_ptau_context.pshalf_jax,
                        control_ptau_ohs=_ptau_context.ohs_jax,
                    )
                    (
                        _precond_pre_blocks,
                        _precond_update_blocks,
                        _precond_diag,
                        accepted_control_ptau_payload,
                    ) = _split_preconditioner_apply_payload(_precond_payload)
                    (
                        (frcc, frss, fzsc, fzcs, flsc, flcs, frsc, frcs, fzcc, fzss, flcc, flss),
                        (
                            frcc_u,
                            frss_u,
                            fzsc_u,
                            fzcs_u,
                            flsc_u,
                            flcs_u,
                            frsc_u,
                            frcs_u,
                            fzcc_u,
                            fzss_u,
                            flcc_u,
                            flss_u,
                        ),
                        (gcr2_p, gcz2_p, gcl2_p, fsqr1_safe, fsqz1_safe, fsql1_safe, fsq1_safe),
                    ) = (_precond_pre_blocks, _precond_update_blocks, _precond_diag)
                    fsqr1 = fsqr1_safe
                    fsqz1 = fsqz1_safe
                    fsql1 = fsql1_safe
                    preconditioner_outputs_scaled = True
                    preconditioner_fsq1_ready = True
                else:
                    frzl_rz = _apply_rz_preconditioner_current(frzl_in=frzl_rhs)
                    frzl_lam_pre = frzl_rz
                if use_apply_payload_fusion and adjoint_trace and adjoint_trace_mode == "full":
                    # The fused GPU-oriented path returns only scaled update
                    # payloads.  Full accepted-trace replay also needs the raw
                    # R/Z-preconditioned force, so materialize it only for that
                    # opt-in diagnostic/validation mode.
                    frzl_rz = _apply_rz_preconditioner_current(frzl_in=frzl_rhs)
                if (not use_apply_payload_fusion) and host_update_assembly:
                    # NumPy path: avoids ~15 JAX dispatches (jnp.asarray, zeros_like, mul).
                    # Asymmetric (lasym) components default to None — the downstream
                    # mode-diag scaling uses _z (pre-allocated zeros) for None entries,
                    # avoiding 6 np.zeros_like allocations per iteration (~0.5s saving).
                    (frcc, frss, fzsc, fzcs, flsc, flcs, frsc, frcs, fzcc, fzss, flcc, flss) = (
                        _preconditioner_output_blocks_np(frzl_rz=frzl_rz, lam_prec=lam_prec)
                    )
                elif (not use_apply_payload_fusion) and use_fused_precond_output_scaling:
                    rz_norm, f_norm1 = _cached_or_current_f_norm1_jax(
                        vmec2000_control=bool(vmec2000_control),
                        vmec2000_cache_valid=bool(vmec2000_cache_valid),
                        need_bcovar_update=bool(need_bcovar_update),
                        cache_rz_norm=cache_rz_norm,
                        cache_f_norm1=cache_f_norm1,
                        state=state,
                        rz_norm_func=_rz_norm,
                    )
                    payload_outputs = _preconditioner_output_payload_jit(
                        apply_lambda_update_scale=(lambda_update_scale != 1.0),
                        vmec2000_control=bool(vmec2000_control),
                        lconm1=bool(getattr(static.cfg, "lconm1", True)),
                    )
                    (
                        (frcc, frss, fzsc, fzcs, flsc, flcs, frsc, frcs, fzcc, fzss, flcc, flss),
                        (
                            frcc_u,
                            frss_u,
                            fzsc_u,
                            fzcs_u,
                            flsc_u,
                            flcs_u,
                            frsc_u,
                            frcs_u,
                            fzcc_u,
                            fzss_u,
                            flcc_u,
                            flss_u,
                        ),
                        (gcr2_p, gcz2_p, gcl2_p, fsqr1_safe, fsqz1_safe, fsql1_safe, fsq1_safe),
                    ) = payload_outputs(frzl_rz, lam_prec, w_mode_mn, lambda_update_scale_j, f_norm1, delta_s, s)
                    fsqr1 = fsqr1_safe
                    fsqz1 = fsqz1_safe
                    fsql1 = fsql1_safe
                    preconditioner_outputs_scaled = True
                    preconditioner_fsq1_ready = True
                elif not use_apply_payload_fusion:
                    frcc = jnp.asarray(frzl_rz.frcc)
                    frss = frzl_rz.frss
                    fzsc = jnp.asarray(frzl_rz.fzsc)
                    fzcs = frzl_rz.fzcs
                    flsc = jnp.asarray(frzl_rz.flsc) * jnp.asarray(lam_prec)
                    flcs = None if frzl_rz.flcs is None else (jnp.asarray(frzl_rz.flcs) * jnp.asarray(lam_prec))
                    frsc = jnp.zeros_like(frcc)
                    frcs = jnp.zeros_like(frcc)
                    fzcc = jnp.zeros_like(fzsc)
                    fzss = jnp.zeros_like(fzsc)
                    flcc = jnp.zeros_like(flsc)
                    flss = jnp.zeros_like(flsc)
                    if getattr(frzl_rz, "frsc", None) is not None:
                        frsc = jnp.asarray(frzl_rz.frsc)
                    if getattr(frzl_rz, "frcs", None) is not None:
                        frcs = jnp.asarray(frzl_rz.frcs)
                    if getattr(frzl_rz, "fzcc", None) is not None:
                        fzcc = jnp.asarray(frzl_rz.fzcc)
                    if getattr(frzl_rz, "fzss", None) is not None:
                        fzss = jnp.asarray(frzl_rz.fzss)
                    if getattr(frzl_rz, "flcc", None) is not None:
                        flcc = jnp.asarray(frzl_rz.flcc) * jnp.asarray(lam_prec)
                    if getattr(frzl_rz, "flss", None) is not None:
                        flss = jnp.asarray(frzl_rz.flss) * jnp.asarray(lam_prec)
                if timing_detail_enabled and t_precond_apply_start is not None:
                    try:
                        if has_jax():
                            jax.block_until_ready(flsc)
                    except Exception:
                        pass
                    timing_stats["precond_apply"] += time.perf_counter() - float(t_precond_apply_start)
            else:
                t_precond_apply_start = time.perf_counter() if timing_detail_enabled else None
                (frcc, frss, fzsc, fzcs, flsc, flcs, frsc, frcs, fzcc, fzss, flcc, flss) = (
                    _radial_preconditioner_output_blocks_jax(
                        frzl=frzl,
                        rz_scale=rz_scale,
                        l_scale=l_scale,
                        precond_radial_alpha=precond_radial_alpha,
                        precond_lambda_alpha=precond_lambda_alpha,
                        apply_radial_tridi_func=_apply_radial_tridi,
                    )
                )
                if timing_detail_enabled and t_precond_apply_start is not None:
                    try:
                        if has_jax():
                            jax.block_until_ready(flsc)
                    except Exception:
                        pass
                    timing_stats["precond_apply"] += time.perf_counter() - float(t_precond_apply_start)

            frzl_pre = TomnspsRZL(
                frcc=frcc,
                frss=frss,
                fzsc=fzsc,
                fzcs=fzcs,
                flsc=flsc,
                flcs=flcs,
                frsc=frsc,
                frcs=frcs,
                fzcc=fzcc,
                fzss=fzss,
                flcc=flcc,
                flss=flss,
            )
            if frzl_lam_pre is not None:
                _maybe_dump_lam_gcl(
                    frzl_pre=frzl_lam_pre,
                    frzl_post=frzl_pre,
                    static=static,
                    iter_idx=int(iter2),
                    delta_s=delta_s,
                )
            _maybe_dump_gc(frzl=frzl_pre, static=static, iter_idx=int(iter2), label="precond")

            # Mode-diagonal preconditioning in (m, n>=0) storage.
            t_precond_mode_start = time.perf_counter() if timing_detail_enabled else None
            if preconditioner_outputs_scaled:
                pass
            elif host_update_assembly:
                # NumPy path: avoids 36 JAX dispatches (expand_dims + broadcast + mul per array).
                # _zeros_coeff_np replaces np.zeros_like (pre-allocated, avoids 6+ allocs/iter).
                (frcc_u, frss_u, fzsc_u, fzcs_u, flsc_u, flcs_u, frsc_u, frcs_u, fzcc_u, fzss_u, flcc_u, flss_u) = (
                    _mode_weight_force_blocks_np(
                        _ForceBlocks(frcc, frss, fzsc, fzcs, flsc, flcs, frsc, frcs, fzcc, fzss, flcc, flss),
                        w_mode_mn=w_mode_mn_np,
                        zeros_coeff=_zeros_coeff_np,
                    )
                )
            else:
                (frcc_u, frss_u, fzsc_u, fzcs_u, flsc_u, flcs_u, frsc_u, frcs_u, fzcc_u, fzss_u, flcc_u, flss_u) = (
                    _mode_weight_force_blocks_jax(
                        _ForceBlocks(frcc, frss, fzsc, fzcs, flsc, flcs, frsc, frcs, fzcc, fzss, flcc, flss),
                        w_mode_mn=w_mode_mn,
                    )
                )
            if timing_detail_enabled and t_precond_mode_start is not None:
                try:
                    if has_jax():
                        jax.block_until_ready(flsc_u)
                except Exception:
                    pass
                timing_stats["precond_mode_scale"] += time.perf_counter() - float(t_precond_mode_start)
            if timing_enabled:
                try:
                    if has_jax() and not timing_detail_enabled:
                        jax.block_until_ready(flsc_u)
                except Exception:
                    pass
                timing_stats["preconditioner"] += time.perf_counter() - float(t_precond_start)
            t_iteration_control_start = time.perf_counter() if timing_enabled else None
            t_iteration_control_fsq1_start = time.perf_counter() if timing_enabled else None

            # VMEC's lambda coefficients can be expressed in multiple scaling
            # conventions (e.g. restart vs. `wout` vs. internal). Allow parity drivers
            # to apply a constant scale to the lambda residual channel before mapping
            # it into coefficient updates.
            if (lambda_update_scale != 1.0) and (not preconditioner_outputs_scaled):
                flsc_u = flsc_u * lambda_update_scale_j
                flcs_u = flcs_u * lambda_update_scale_j
                flcc_u = flcc_u * lambda_update_scale_j
                flss_u = flss_u * lambda_update_scale_j

            if auto_flip_force and it == 0:
                # Choose force direction by a tiny trial step on the VMEC residual
                # (fsqr+fsqz+fsql), not magnetic energy. Energy monotonicity is not a
                # reliable proxy for VMEC's preconditioned convergence metrics.
                w_curr = float(fsqr_f + fsqz_f + fsql_f)
                # Use a probe step that is large enough to be numerically decisive,
                # but still small relative to typical pseudo-time updates.
                dt_probe = min(1e-2, 0.1 * float(time_step))
                dR_dir = dt_probe * _mn_cos_to_signed_physical(frcc_u, frss_u)
                dZ_dir = dt_probe * _mn_sin_to_signed_physical(fzsc_u, fzcs_u)
                dL_dir = dt_probe * _mn_sin_to_signed_physical_lambda(flsc_u, flcs_u)
                if bool(cfg.lasym):
                    dR_sin_dir = dt_probe * _mn_sin_to_signed_physical(frsc_u, frcs_u)
                    dZ_cos_dir = dt_probe * _mn_cos_to_signed_physical(fzcc_u, fzss_u)
                    dL_cos_dir = dt_probe * _mn_cos_to_signed_physical_lambda(flcc_u, flss_u)
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
            accepted_control_ptau_host: tuple[float, float] | None = None
            use_host_fsq1_norms = (
                bool(startup_policy.host_fsq1_norms_on_accelerator)
                and (not bool(host_update_assembly))
                and (jax.default_backend() != "cpu")
                and (not _tree_has_tracer(state))
                and (not _tree_has_tracer(frzl_pre))
            )
            frzl_pre_host = None
            t_fsq1_precond_norm_start = time.perf_counter() if timing_enabled else None
            if preconditioner_fsq1_ready:
                pass
            elif host_update_assembly or use_host_fsq1_norms:
                # NumPy path: avoids 6+ JAX dispatches for sum-of-squares.
                frzl_pre_host = frzl_pre if host_update_assembly else _tomnsps_to_numpy_host(frzl_pre)
                gcr2_p, gcz2_p, gcl2_p = vmec_gcx2_from_tomnsps_np(
                    frzl=frzl_pre_host,
                    include_edge=True,
                )
            else:
                gcr2_p, gcz2_p, gcl2_p = vmec_gcx2_from_tomnsps(
                    frzl=frzl_pre,
                    lconm1=bool(getattr(static.cfg, "lconm1", True)),
                    apply_m1_constraints=False,
                    # VMEC residue.f90 calls getfsq(..., medge=m1) for fsq*1,
                    # i.e. it includes the edge row in preconditioned R/Z norms.
                    include_edge=True,
                    apply_scalxc=False,
                    s=s,
                )
            if host_update_assembly or use_host_fsq1_norms:
                host_channels = _precond_payload_facade.host_preconditioned_residual_scalar_channels(
                    gcr2_p=gcr2_p,
                    gcz2_p=gcz2_p,
                    gcl2_p=gcl2_p,
                    frzl_pre=frzl_pre,
                    frzl_pre_host=frzl_pre_host,
                    vmec2000_control=bool(vmec2000_control),
                    vmec2000_cache_valid=bool(vmec2000_cache_valid),
                    need_bcovar_update=bool(need_bcovar_update),
                    cache_rz_norm=cache_rz_norm,
                    cache_f_norm1=cache_f_norm1,
                    state=state,
                    delta_s=float(delta_s),
                    numpy_module=np,
                    rz_norm_np=_rz_norm_np,
                    lambda_preconditioned_full_norm=_lambda_preconditioned_full_norm,
                    finite_float_or_zero=_finite_float_or_zero,
                )
                (
                    rz_norm,
                    f_norm1,
                    fsqr1,
                    fsqz1,
                    fsql1,
                    fsqr1_safe,
                    fsqz1_safe,
                    fsql1_safe,
                    fsq1,
                ) = host_channels
            elif not preconditioner_fsq1_ready:
                jax_channels = _precond_payload_facade.jax_preconditioned_residual_scalar_channels(
                    gcr2_p=gcr2_p,
                    gcz2_p=gcz2_p,
                    gcl2_p=gcl2_p,
                    frzl_pre=frzl_pre,
                    vmec2000_control=bool(vmec2000_control),
                    vmec2000_cache_valid=bool(vmec2000_cache_valid),
                    need_bcovar_update=bool(need_bcovar_update),
                    cache_rz_norm=cache_rz_norm,
                    cache_f_norm1=cache_f_norm1,
                    state=state,
                    delta_s=delta_s,
                    jnp_module=jnp,
                    cached_or_current_f_norm1_jax=_cached_or_current_f_norm1_jax,
                    rz_norm_func=_rz_norm,
                    lambda_preconditioned_full_norm=_lambda_preconditioned_full_norm,
                )
                (
                    rz_norm,
                    f_norm1,
                    fsqr1,
                    fsqz1,
                    fsql1,
                    fsqr1_safe,
                    fsqz1_safe,
                    fsql1_safe,
                    fsq1,
                ) = jax_channels
            if timing_enabled and t_fsq1_precond_norm_start is not None:
                timing_stats["iteration_control_fsq1_precond_norm"] += time.perf_counter() - float(
                    t_fsq1_precond_norm_start
                )
            if _env_dump_lam not in ("", "0") and frzl_lam_pre is None:
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
                    fsql1_post=fsql1,
                    static=static,
                    iter_idx=int(iter2),
                )
            if not (host_update_assembly or use_host_fsq1_norms):
                t_fsq1_scalar_build_start = time.perf_counter() if timing_enabled else None
                # Extremely small late-iteration channels can occasionally surface
                # as NaN/Inf through mixed 0*Inf paths in XLA. VMEC treats these
                # as effectively zero for the preconditioned residual diagnostics.
                if preconditioner_fsq1_ready:
                    fsq1_j = fsq1_safe
                else:
                    fsq1_j = fsq1
                if timing_enabled and t_fsq1_scalar_build_start is not None:
                    timing_stats["iteration_control_fsq1_scalar_build"] += time.perf_counter() - float(
                        t_fsq1_scalar_build_start
                    )
                use_control_payload = (
                    (not bool(converged_physical))
                    and (bool(reference_mode) or bool(vmec2000_control))
                    and (not bool(badjac_use_state))
                    and (not bool(dump_ptau_state))
                    and os.getenv("VMEC_JAX_DUMP_PTAU", "") in ("", "0")
                    and jax.default_backend() != "cpu"
                )
                control_payload = _precond_payload_facade.materialize_accepted_control_payload(
                    accepted_control_ptau_payload=accepted_control_ptau_payload,
                    use_control_payload=bool(use_control_payload),
                    fsq1_j=fsq1_j,
                    k=k,
                    ptau_pshalf_jax=_ptau_context.pshalf_jax,
                    ptau_ohs_jax=_ptau_context.ohs_jax,
                    timing_enabled=bool(timing_enabled),
                    timing_stats=timing_stats,
                    perf_counter=time.perf_counter,
                    jax_module=jax,
                    device_get_floats=_device_get_floats,
                    accepted_control_ptau_host_from_payload=_accepted_control_ptau_host_from_payload,
                    scan_math_kernel_arrays_from_k=_scan_math_kernel_arrays_from_k,
                    accepted_control_payload_jit=_accepted_control_payload_jit,
                )
                fsq1 = control_payload.fsq1
                accepted_control_ptau_host = control_payload.accepted_control_ptau_host
                control_payload_used = control_payload.control_payload_used
            if timing_enabled and t_iteration_control_fsq1_start is not None:
                timing_stats["iteration_control_fsq1"] += time.perf_counter() - float(t_iteration_control_fsq1_start)
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
                return _append_zero_update_history_record(
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
                    history_record_lists=_history_record_lists,
                )

            _append_preconditioned_residual_history(
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
                rz_norm_history=rz_norm_history,
                f_norm1_history=f_norm1_history,
                gcr2_p_history=gcr2_p_history,
                gcz2_p_history=gcz2_p_history,
                gcl2_p_history=gcl2_p_history,
                fsq1_history=fsq1_history,
                fsqr1_history=fsqr1_history,
                fsqz1_history=fsqz1_history,
                fsql1_history=fsql1_history,
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
                if timing_enabled and t_iteration_control_start is not None:
                    timing_stats["iteration_control"] += time.perf_counter() - float(t_iteration_control_start)
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
                min_tau_ptau = max_tau_ptau = None
                bad_jacobian_ptau = None
                if accepted_control_ptau_host is not None:
                    min_tau_ptau, max_tau_ptau = accepted_control_ptau_host
                else:
                    ptau_min, ptau_max = _ptau_minmax_from_k_host(k)
                    if ptau_min is not None and ptau_max is not None:
                        t_badjac_ptau_get_start = time.perf_counter() if timing_enabled else None
                        min_tau_ptau, max_tau_ptau = _device_get_floats(ptau_min, ptau_max)
                        if timing_enabled and t_badjac_ptau_get_start is not None:
                            timing_stats["iteration_control_badjac_ptau_get"] += time.perf_counter() - float(
                                t_badjac_ptau_get_start
                            )
                ptau_decision = None
                if min_tau_ptau is not None and max_tau_ptau is not None:
                    ptau_decision = _bad_jacobian_tau_decision(
                        min_tau=min_tau_ptau,
                        max_tau=max_tau_ptau,
                        vmec2000_control=bool(vmec2000_control),
                        ptau_tol=ptau_tol,
                    )
                    bad_jacobian_ptau = bool(ptau_decision.bad_jacobian)

                state_probe = _should_probe_bad_jacobian_state(
                    state_probe=bool(badjac_state_probe),
                    initial_state_probe_iters=int(badjac_initial_state_probe_iters),
                    iter_idx=int(iter2),
                )
                need_state_jac = _bad_jacobian_requires_state_jacobian(
                    badjac_use_state=bool(badjac_use_state),
                    dump_ptau_state=bool(dump_ptau_state),
                    state_probe=bool(state_probe),
                    ptau_decision=ptau_decision,
                )
                if need_state_jac:
                    t_badjac_state_jacobian_start = time.perf_counter() if timing_enabled else None
                    from vmec_jax.vmec_numpy_forces import _numpy_module_patch as _hot_numpy_patch

                    min_tau_state, max_tau_state = _state_tau_minmax_from_vmec_state(
                        state=state,
                        modes=static.modes,
                        trig=trig,
                        s=s,
                        lconm1=bool(getattr(static.cfg, "lconm1", True)),
                        lthreed=bool(getattr(static.cfg, "lthreed", True)),
                        mask_even=getattr(static, "m_is_even", None),
                        mask_odd=getattr(static, "m_is_odd", None),
                        host_update_assembly=bool(host_update_assembly),
                        tree_has_tracer=_tree_has_tracer,
                        jacobian_from_state=vmec_half_mesh_jacobian_from_state,
                        device_get_floats=_device_get_floats,
                        jnp_module=jnp,
                        numpy_patch_context=_hot_numpy_patch,
                    )
                    if timing_enabled and t_badjac_state_jacobian_start is not None:
                        timing_stats["iteration_control_badjac_state_jacobian"] += time.perf_counter() - float(
                            t_badjac_state_jacobian_start
                        )
                    state_decision = _bad_jacobian_tau_decision(
                        min_tau=min_tau_state,
                        max_tau=max_tau_state,
                        vmec2000_control=bool(vmec2000_control),
                        ptau_tol=ptau_tol,
                    )
                    bad_jacobian_state = bool(state_decision.bad_jacobian)
                else:
                    min_tau_state = float("nan")
                    max_tau_state = float("nan")
                    bad_jacobian_state = False
                    state_decision = _bad_jacobian_tau_decision(
                        min_tau=min_tau_state,
                        max_tau=max_tau_state,
                        vmec2000_control=bool(vmec2000_control),
                        ptau_tol=ptau_tol,
                    )

                badjac_selection = _select_bad_jacobian_decision(
                    badjac_use_state=bool(badjac_use_state),
                    ptau_decision=ptau_decision,
                    state_decision=state_decision,
                )
                bad_jacobian = bool(badjac_selection.bad_jacobian)
                min_tau = float(badjac_selection.min_tau)
                max_tau = float(badjac_selection.max_tau)

                _maybe_dump_ptau(
                    iter_idx=int(iter2),
                    ptau_min=float(min_tau_ptau if min_tau_ptau is not None else float("nan")),
                    ptau_max=float(max_tau_ptau if max_tau_ptau is not None else float("nan")),
                    tau_min_state=min_tau_state if np.isfinite(min_tau_state) else None,
                    tau_max_state=max_tau_state if np.isfinite(max_tau_state) else None,
                    badjac_ptau=bad_jacobian_ptau,
                    badjac_state=bad_jacobian_state,
                    badjac_used=bool(bad_jacobian),
                    mode=badjac_mode,
                    label="iter",
                )

                if np.isfinite(min_tau) and np.isfinite(max_tau):
                    _append_badjac_history(min_tau, max_tau, bool(bad_jacobian))
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
                    _append_badjac_history(float("nan"), float("nan"), False)
            else:
                _append_badjac_history(float("nan"), float("nan"), False)

            # VMEC eqsolve: after the first evolve step, if the Jacobian is bad
            # and ijacob==0, retry with an improved axis guess.
            if bool(vmec2000_control) and (not axis_reset_done) and bool(lmove_axis) and (iter2 == 1):
                fsq_curr = fsqr_f + fsqz_f + fsql_f
                axis_runtime_decision = _initial_axis_reset_runtime_decision(
                    bad_jacobian=bool(bad_jacobian),
                    fsq_phys=fsq_curr,
                    axis_reset_fsq_min=float(axis_reset_fsq_min),
                    force_axis_reset=bool(force_axis_reset),
                    axis_reset_always_3d=bool(axis_reset_always_3d),
                    lthreed=bool(getattr(cfg, "lthreed", True)),
                    vmec2000_control=bool(vmec2000_control),
                    lmove_axis=bool(lmove_axis),
                )
                bad_jacobian = bool(axis_runtime_decision.bad_jacobian)
                huge_initial_forces = bool(axis_runtime_decision.huge_initial_forces)
                force_axis_reset_init = bool(axis_runtime_decision.force_reset)
                if axis_runtime_decision.reset:
                    if verbose and bool(vmec2000_control) and bool(verbose_vmec2000_table):
                        if bad_jacobian or force_axis_reset_init:
                            print(" INITIAL JACOBIAN CHANGED SIGN!", flush=True)
                        print(" TRYING TO IMPROVE INITIAL MAGNETIC AXIS GUESS", flush=True)
                    state = _reset_axis_from_boundary(state, k_guess=k, full_reset=False, refine_axis_guess=False)
                    if verbose and bool(vmec2000_control) and bool(verbose_vmec2000_table):
                        if axis_reset_coeffs is not None:
                            raxis_cc, _raxis_cs, _zaxis_cc, zaxis_cs = axis_reset_coeffs
                            _print_scan_axis_guess(raxis_cc, zaxis_cs)
                    state_checkpoint = state
                    vRcc, vRss, vZsc, vZcs, vLsc, vLcs = _zero_velocity_blocks_like(
                        vRcc, vRss, vZsc, vZcs, vLsc, vLcs
                    )
                    time_step = float(time_step)
                    ijacob = 1
                    axis_reset_done = True
                    iter1 = iter2
                    freeb_controls_cached = None
                    bad_growth_streak = 0
                    inv_tau = [0.15 / time_step] * k_ndamp
                    _clear_preconditioner_cache_locals()
                    _pop_iteration_histories()
                    prev_rz_fsq = prev_rz_fsq_before
                    # VMEC restarts the iteration after axis reset without
                    # advancing the iteration counter. Emulate that by
                    # repeating iter2==1 on the next loop pass.
                    if iter2 == 1:
                        iter_offset -= 1
                    if timing_enabled and t_iteration_control_badjac_start is not None:
                        timing_stats["iteration_control_badjac"] += time.perf_counter() - float(
                            t_iteration_control_badjac_start
                        )
                    continue
            if timing_enabled and t_iteration_control_badjac_start is not None:
                timing_stats["iteration_control_badjac"] += time.perf_counter() - float(
                    t_iteration_control_badjac_start
                )

            # VMEC-style time-step control: VMEC2000's `TimeStepControl` + `restart_iter`.
            t_iteration_control_vmec_time_start = time.perf_counter() if timing_enabled else None
            if bool(vmec2000_control) and (not skip_time_control):
                # VMEC's TimeStepControl uses the *previous* preconditioned
                # residual (fsq) which is updated at the end of evolve.f.
                tc = _vmec2000_time_control_decision(
                    iter2=int(iter2),
                    iter1=int(iter1),
                    fsq_prev=float(fsq_prev),
                    fsq0_curr=float(fsq0_curr),
                    fsq0_prev=float(fsq0_prev),
                    res0=float(res0),
                    res1=float(res1),
                    bad_jacobian=bool(bad_jacobian),
                    vmec2000_fact=float(vmec2000_fact),
                )
                fsq = tc.fsq
                fsq0 = tc.fsq0
                res0 = tc.res0
                res1 = tc.res1
                irst_tc = tc.irst
                irst_trace = tc.trace_irst
                if tc.initialized:
                    state_checkpoint = state
                    _dump_time_control_trace(
                        stage="init",
                        iter2=int(iter2),
                        iter1=int(iter1),
                        fsq=float(fsq),
                        fsq0=float(fsq0),
                        res0=float(res0),
                        res1=float(res1),
                        time_step=float(time_step),
                        irst=int(irst_trace),
                    )
                    _maybe_dump_checkpoint(
                        iter_idx=int(iter2), fsq=float(fsq), fsq0=float(fsq0), res0=float(res0), res1=float(res1)
                    )
                _dump_time_control_trace(
                    stage="pre",
                    iter2=int(iter2),
                    iter1=int(iter1),
                    fsq=float(fsq),
                    fsq0=float(fsq0),
                    res0=float(res0),
                    res1=float(res1),
                    time_step=float(time_step),
                    irst=int(irst_trace),
                )
                if tc.store_checkpoint:
                    _dump_time_control_trace(
                        stage="checkpoint",
                        iter2=int(iter2),
                        iter1=int(iter1),
                        fsq=float(fsq),
                        fsq0=float(fsq0),
                        res0=float(res0),
                        res1=float(res1),
                        time_step=float(time_step),
                        irst=int(irst_trace),
                    )
                    state_checkpoint = state
                    _maybe_dump_checkpoint(
                        iter_idx=int(iter2), fsq=float(fsq), fsq0=float(fsq0), res0=float(res0), res1=float(res1)
                    )
                if tc.restart:
                    _maybe_dump_time_control(
                        iter_idx=int(iter2),
                        fsq=float(fsq),
                        fsq0=float(fsq0),
                        res0=float(res0),
                        res1=float(res1),
                        time_step=float(time_step),
                    )
                    pre_restart_reason = tc.pre_restart_reason
                    state = state_checkpoint
                    (
                        vRcc,
                        vRss,
                        vZsc,
                        vZcs,
                        vLsc,
                        vLcs,
                        vRsc,
                        vRcs,
                        vZcc,
                        vZss,
                        vLcc,
                        vLss,
                    ) = _zero_velocity_blocks_like(
                        vRcc, vRss, vZsc, vZcs, vLsc, vLcs, vRsc, vRcs, vZcc, vZss, vLcc, vLss
                    )
                    iter1_prev = int(iter1)
                    time_step_prev = float(time_step)
                    _dump_time_control_trace(
                        stage="restart",
                        iter2=int(iter2),
                        iter1=iter1_prev,
                        fsq=float(fsq),
                        fsq0=float(fsq0),
                        res0=float(res0),
                        res1=float(res1),
                        time_step=time_step_prev,
                        irst=int(irst_tc),
                    )
                    # VMEC2000 `restart_iter`: irst=2 (bad-jac) -> dt*0.9,
                    # irst=3 (time-control) -> dt/1.03.
                    if irst_tc == 2:
                        time_step = max(restart_badjac_factor * time_step, 1e-12)
                        ijacob += 1
                        step_status = "restart_bad_jacobian"
                        restart_reason = "bad_jacobian"
                    else:
                        time_step = max(time_step / restart_badprog_factor, 1e-12)
                        step_status = "restart_time_control"
                        restart_reason = "time_control"
                    bad_resets += 1
                    iter1 = iter2
                    freeb_controls_cached = None
                    bad_growth_streak = 0
                    fsq_prev = fsq_prev_before
                    fsq0_prev = fsq0_prev_before
                    inv_tau = [0.15 / time_step] * k_ndamp
                    _clear_preconditioner_cache_locals()
                    force_bcovar_update = True
                    _append_current_zero_update_history(
                        restart_path="vmec2000_bad_jacobian" if irst_tc == 2 else "vmec2000_time_control",
                        step_status=step_status,
                        restart_reason=restart_reason,
                        pre_restart_reason=pre_restart_reason,
                        time_step_value=time_step,
                    )
                    _pop_iteration_histories()
                    prev_rz_fsq = prev_rz_fsq_before
                    skip_time_control = True
                    if timing_enabled and t_iteration_control_vmec_time_start is not None:
                        timing_stats["iteration_control_vmec_time"] += time.perf_counter() - float(
                            t_iteration_control_vmec_time_start
                        )
                    continue
            if timing_enabled and t_iteration_control_vmec_time_start is not None:
                timing_stats["iteration_control_vmec_time"] += time.perf_counter() - float(
                    t_iteration_control_vmec_time_start
                )

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
            res0 = restart_decision.res0
            bad_growth_streak = restart_decision.bad_growth_streak
            pre_restart_reason = restart_decision.pre_restart_reason
            huge_initial_forces = restart_decision.huge_initial_forces

            # Store a "good" checkpoint once residual has improved for many
            # iterations since the last restart marker.
            if restart_decision.store_checkpoint:
                state_checkpoint = state

            if use_restart_triggers and pre_restart_reason != "none":
                state_before_restart = state
                vRcc_before = vRcc
                vRss_before = vRss
                vZsc_before = vZsc
                vZcs_before = vZcs
                vLsc_before = vLsc
                vLcs_before = vLcs
                vRsc_before = vRsc
                vRcs_before = vRcs
                vZcc_before = vZcc
                vZss_before = vZss
                vLcc_before = vLcc
                vLss_before = vLss
                state = state_checkpoint
                (
                    vRcc,
                    vRss,
                    vZsc,
                    vZcs,
                    vLsc,
                    vLcs,
                    vRsc,
                    vRcs,
                    vZcc,
                    vZss,
                    vLcc,
                    vLss,
                ) = _zero_velocity_blocks_like(vRcc, vRss, vZsc, vZcs, vLsc, vLcs, vRsc, vRcs, vZcc, vZss, vLcc, vLss)
                if pre_restart_reason == "bad_jacobian":
                    time_step = max(restart_badjac_factor * time_step, 1e-12)
                    ijacob += 1
                    step_status = "restart_bad_jacobian"
                elif pre_restart_reason == "stage_transition":
                    time_step = max(time_step * stage_transition_scale, 1e-12)
                    step_status = "restart_stage_transition"
                else:
                    time_step = max(time_step / restart_badprog_factor, 1e-12)
                    step_status = "restart_bad_progress"
                if bool(huge_initial_forces) and (pre_restart_reason == "bad_jacobian"):
                    huge_force_restart_count += 1
                else:
                    huge_force_restart_count = 0
                if ijacob in (25, 50):
                    scale = 0.98 if ijacob < 50 else 0.96
                    time_step = max(scale * float(step_size), 1e-12)
                time_step_iter = float(time_step)
                bad_resets += 1
                iter1 = iter2
                freeb_controls_cached = None
                bad_growth_streak = 0
                fsq_prev = fsq_prev_before
                fsq0_prev = fsq0_prev_before
                inv_tau = [0.15 / time_step] * k_ndamp
                _clear_preconditioner_cache_locals()
                if bool(vmec2000_control):
                    force_bcovar_update = True
                _append_current_zero_update_history(
                    restart_path="pre_restart_trigger",
                    step_status=step_status,
                    restart_reason=pre_restart_reason,
                    pre_restart_reason=pre_restart_reason,
                    time_step_value=time_step_iter,
                )
                _print_compact_residual_iteration_update_status(
                    verbose=bool(verbose),
                    vmec2000_control=bool(vmec2000_control),
                    verbose_vmec2000_table=bool(verbose_vmec2000_table),
                    precond_diag_floats=_precond_diag_floats,
                    iter_idx=int(it),
                    dt_eff=0.0,
                    update_rms=0.0,
                    step_status=step_status,
                )
                _maybe_dump_xc(
                    state=state_before_restart,
                    vRcc=vRcc_before,
                    vRss=vRss_before,
                    vZsc=vZsc_before,
                    vZcs=vZcs_before,
                    vLsc=vLsc_before,
                    vLcs=vLcs_before,
                    vRsc=vRsc_before,
                    vRcs=vRcs_before,
                    vZcc=vZcc_before,
                    vZss=vZss_before,
                    vLcc=vLcc_before,
                    vLss=vLss_before,
                    static=static,
                    iter_idx=int(iter2),
                )
                _pop_iteration_histories()
                prev_rz_fsq = prev_rz_fsq_before
                skip_time_control = True
                if timing_enabled and t_iteration_control_restart_start is not None:
                    timing_stats["iteration_control_restart"] += time.perf_counter() - float(
                        t_iteration_control_restart_start
                    )
                continue

            if timing_enabled and t_iteration_control_restart_start is not None:
                timing_stats["iteration_control_restart"] += time.perf_counter() - float(
                    t_iteration_control_restart_start
                )
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
        if iter2 == iter1:
            inv_tau = [0.15 / time_step] * k_ndamp
        else:
            invtau_num = 0.0 if fsq1 == 0.0 else min(abs(np.log(fsq1 / fsq_prev)), 0.15)
            inv_tau = inv_tau[1:] + [invtau_num / time_step]
        fsq_prev = fsq1
        fsq0_prev = fsq0_curr

        otav = float(np.sum(inv_tau)) / float(k_ndamp)
        dtau = time_step * otav / 2.0
        b1 = 1.0 - dtau
        fac = 1.0 / (1.0 + dtau)
        _dump_evolve_trace(
            iter2=int(iter2),
            iter1=int(iter1),
            stage="pre",
            fsq1_val=float(fsq1),
            fsq_prev_val=float(fsq_prev_before),
            time_step_val=float(time_step),
            dtau_val=float(dtau),
            b1_val=float(b1),
            fac_val=float(fac),
            state_val=state,
            vRcc_val=vRcc,
            vRss_val=vRss,
            vZsc_val=vZsc,
            vZcs_val=vZcs,
            vLsc_val=vLsc,
            vLcs_val=vLcs,
            vRsc_val=vRsc,
            vRcs_val=vRcs,
            vZcc_val=vZcc,
            vZss_val=vZss,
            vLcc_val=vLcc,
            vLss_val=vLss,
            frcc_val=frcc_u,
            frss_val=frss_u,
            fzsc_val=fzsc_u,
            fzcs_val=fzcs_u,
            flsc_val=flsc_u,
            flcs_val=flcs_u,
            frsc_val=frsc_u,
            frcs_val=frcs_u,
            fzcc_val=fzcc_u,
            fzss_val=fzss_u,
            flcc_val=flcc_u,
            flss_val=flss_u,
        )

        if timing_enabled and t_iteration_control_evolve_start is not None:
            timing_stats["iteration_control_evolve"] += time.perf_counter() - float(t_iteration_control_evolve_start)
        if timing_enabled and t_iteration_control_start is not None:
            timing_stats["iteration_control"] += time.perf_counter() - float(t_iteration_control_start)
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
            if timing_enabled and t_trace_build_start is not None:
                timing_stats["update_trace_build"] += time.perf_counter() - float(t_trace_build_start)
            t_state_update_start = time.perf_counter() if timing_enabled else None
            dt_eff = float(time_step)
            if bool(limit_dt_from_force):
                dt_eff = _safe_dt_from_force_blocks(
                    dt_nominal=time_step,
                    max_coeff_delta_rms=max_coeff_delta_rms,
                    blocks=_ForceBlocks(
                        frcc_u,
                        frss_u,
                        fzsc_u,
                        fzcs_u,
                        flsc_u,
                        flcs_u,
                        frsc_u,
                        frcs_u,
                        fzcc_u,
                        fzss_u,
                        flcc_u,
                        flss_u,
                    ),
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
                step_fn = _strict_update_step_jit(
                    static,
                    limit_update_rms=False,
                    need_update_rms=need_update_rms,
                    divide_by_scalxc_for_update=bool(divide_by_scalxc_for_update),
                    enforce_edge=not bool(free_boundary_enabled),
                )
                step_out = step_fn(
                    state,
                    dt_eff,
                    b1,
                    fac,
                    force_scale,
                    flip_sign,
                    vRcc,
                    vRss,
                    vZsc,
                    vZcs,
                    vLsc,
                    vLcs,
                    vRsc,
                    vRcs,
                    vZcc,
                    vZss,
                    vLcc,
                    vLss,
                    frcc_u,
                    frss_u,
                    fzsc_u,
                    fzcs_u,
                    flsc_u,
                    flcs_u,
                    frsc_u,
                    frcs_u,
                    fzcc_u,
                    fzss_u,
                    flcc_u,
                    flss_u,
                    max_update_rms,
                )
                state_try = step_out["state_post"]
                vRcc = step_out["vRcc_after"]
                vRss = step_out["vRss_after"]
                vZsc = step_out["vZsc_after"]
                vZcs = step_out["vZcs_after"]
                vLsc = step_out["vLsc_after"]
                vLcs = step_out["vLcs_after"]
                vRsc = step_out["vRsc_after"]
                vRcs = step_out["vRcs_after"]
                vZcc = step_out["vZcc_after"]
                vZss = step_out["vZss_after"]
                vLcc = step_out["vLcc_after"]
                vLss = step_out["vLss_after"]
                update_rms_j = (
                    step_out["update_rms_postclip"]
                    if need_update_rms
                    else jnp.asarray(0.0, dtype=jnp.asarray(vRcc).dtype)
                )
                update_rms = None
                update_rms_preclip = None
                scl = 1.0
            else:
                velocity_blocks = _ResidualVelocityBlocks(
                    vRcc,
                    vRss,
                    vRsc,
                    vRcs,
                    vZsc,
                    vZcs,
                    vZcc,
                    vZss,
                    vLsc,
                    vLcs,
                    vLcc,
                    vLss,
                )
                force_blocks = _ResidualVelocityBlocks(
                    frcc_u,
                    frss_u,
                    frsc_u,
                    frcs_u,
                    fzsc_u,
                    fzcs_u,
                    fzcc_u,
                    fzss_u,
                    flsc_u,
                    flcs_u,
                    flcc_u,
                    flss_u,
                )
                if host_update_assembly:
                    update_result = _host_momentum_update_np(
                        velocities=velocity_blocks,
                        forces=force_blocks,
                        b1=b1,
                        fac=fac,
                        force_scale=force_scale,
                        flip_sign=flip_sign,
                        dt_eff=dt_eff,
                        compute_update_rms=need_update_rms,
                    )
                else:
                    update_result = _momentum_update_jax(
                        velocities=velocity_blocks,
                        forces=force_blocks,
                        b1=b1,
                        fac=fac,
                        force_scale=force_scale,
                        flip_sign=flip_sign,
                        dt_eff=dt_eff,
                        compute_update_rms=need_update_rms,
                    )
                (vRcc, vRss, vRsc, vRcs, vZsc, vZcs, vZcc, vZss, vLsc, vLcs, vLcc, vLss) = (
                    update_result.velocities
                )
                update_rms_j = (
                    update_result.update_rms if need_update_rms else jnp.asarray(0.0, dtype=jnp.asarray(vRcc).dtype)
                )

            if not use_jit_strict_update_step:
                update_rms_host: float | None = None

                def _update_rms_float() -> float:
                    nonlocal update_rms_host
                    if update_rms_host is None:
                        update_rms_host = float(np.asarray(update_rms_j))
                    return update_rms_host

                if (
                    bool(limit_update_rms)
                    or bool(backtracking)
                    or (bool(adjoint_trace) and adjoint_trace_mode == "full")
                ):
                    update_rms = _update_rms_float()
                else:
                    update_rms = None
                update_rms_preclip = update_rms
                if bool(limit_update_rms) and np.isfinite(update_rms) and (update_rms > max_update_rms):
                    scl = max_update_rms / max(update_rms, 1e-30)
                    (vRcc, vRss, vRsc, vRcs, vZsc, vZcs, vZcc, vZss, vLsc, vLcs, vLcc, vLss) = (
                        _scale_velocity_blocks(scl, vRcc, vRss, vRsc, vRcs, vZsc, vZcs, vZcc, vZss, vLsc, vLcs, vLcc, vLss)
                    )
                    update_rms_j = _force_update_rms(
                        dt_eff,
                        vRcc,
                        vRss,
                        vRsc,
                        vRcs,
                        vZsc,
                        vZcs,
                        vZcc,
                        vZss,
                        vLsc,
                        vLcs,
                        vLcc,
                        vLss,
                    )
                    update_rms_host = float(np.asarray(update_rms_j))
                    update_rms = update_rms_host
                else:
                    scl = 1.0

                update_deltas = _delta_tuple_from_blocks(
                    dt_eff,
                    _physical_delta_transforms,
                    vRcc,
                    vRss,
                    vRsc,
                    vRcs,
                    vZsc,
                    vZcs,
                    vZcc,
                    vZss,
                    vLsc,
                    vLcs,
                    vLcc,
                    vLss,
                    use_numpy_lasym_zeros=bool(host_update_assembly),
                )
                state_try = _candidate_state_from_delta_tuple(
                    update_deltas,
                    use_numpy_arrays=bool(host_update_assembly),
                    use_numpy_enforce=bool(host_update_assembly),
                )
            probe_bad_jacobian = False
            if need_trial_eval:
                freeb_bsqvac_half_trial = _freeb_bsqvac_half_for_trial_state(state_try)
                w_try = _trial_residual_total(
                    state_try,
                    freeb_bsqvac_half_trial,
                    zero_m1_value=zero_m1,
                    timing_label="trial",
                )
                w_try_ratio = w_try / max(w_curr, 1e-30) if np.isfinite(w_try) else float("inf")
                if bool(reference_mode) and (float(np.asarray(zero_m1)) > 0.5):
                    w_probe = _trial_residual_total(
                        state_try,
                        freeb_bsqvac_half_trial,
                        zero_m1_value=jnp.asarray(0.0, dtype=zero_m1.dtype),
                    )
                    if (not np.isfinite(w_probe)) or (w_probe > 1.0e2 * max(w_curr, 1e-30)):
                        probe_bad_jacobian = True
                        w_try = float("inf")
                        w_try_ratio = float("inf")
            else:
                w_try = w_curr
                w_try_ratio = 1.0

            # The reference iteration is typically stable under its restart
            # triggers, but our parity-path preconditioners are still evolving.
            # Add a small,
            # bounded backtracking on the position update (not the force
            # evaluation) to prevent systematic residual growth.
            alpha = 1.0
            accept_ratio = 1.001 if backtracking else float("inf")
            if np.isfinite(w_try) and (w_try > accept_ratio * max(w_curr, 1e-30)):
                for _ in range(8):
                    alpha *= 0.5
                    state_try = _candidate_state_from_delta_tuple(
                        update_deltas,
                        scale=alpha,
                        use_numpy_arrays=False,
                        use_numpy_enforce=bool(host_update_assembly),
                    )
                    freeb_bsqvac_half_trial = _freeb_bsqvac_half_for_trial_state(state_try)
                    w_try = _trial_residual_total(
                        state_try,
                        freeb_bsqvac_half_trial,
                        zero_m1_value=zero_m1,
                        timing_label="trial",
                    )
                    w_try_ratio = w_try / max(w_curr, 1e-30) if np.isfinite(w_try) else float("inf")
                    if np.isfinite(w_try) and (w_try <= accept_ratio * max(w_curr, 1e-30)):
                        # Keep momentum consistent with the smaller step.
                        vRcc, vRss, vZsc, vZcs, vLsc, vLcs = _scale_velocity_blocks(
                            alpha, vRcc, vRss, vZsc, vZcs, vLsc, vLcs
                        )
                        update_rms *= alpha
                        dt_eff *= alpha
                        break

            # Require (near) monotone improvement; otherwise fall back to the
            # restart/timestep control path.
            if np.isfinite(w_try) and (w_try <= accept_ratio * max(w_curr, 1e-30)):
                state = state_try
                step_status = "momentum"
                restart_reason = "none"
                huge_force_restart_count = 0
                restart_path = "momentum_accept"
            else:
                catastrophic_restart = True
                clear_cache_after_catastrophic = not bool(vmec2000_control)
                if use_direct_fallback:
                    # Try a small direct-force step (no momentum memory) before
                    # a full restart. This is an experimental parity path.
                    clear_cache_after_catastrophic = bool(vmec2000_control)
                    dt_direct = max(0.1 * dt_eff, 1e-12)
                    force_rms = _host_force_update_rms(
                        1.0,
                        frcc_u,
                        frss_u,
                        frsc_u,
                        frcs_u,
                        fzsc_u,
                        fzcs_u,
                        fzcc_u,
                        fzss_u,
                        flsc_u,
                        flcs_u,
                        flcc_u,
                        flss_u,
                    )
                    if np.isfinite(force_rms) and force_rms > 0.0:
                        dt_cap = max_update_rms / max(force_rms, 1e-30)
                        dt_direct = max(min(dt_direct, float(dt_cap)), 1e-12)
                    state_dir = _candidate_state_from_delta_tuple(
                        _delta_tuple_from_blocks(
                            dt_direct,
                            _internal_delta_transforms,
                            flip_sign * frcc_u,
                            flip_sign * frss_u,
                            flip_sign * frsc_u,
                            flip_sign * frcs_u,
                            flip_sign * fzsc_u,
                            flip_sign * fzcs_u,
                            flip_sign * fzcc_u,
                            flip_sign * fzss_u,
                            flip_sign * flsc_u,
                            flip_sign * flcs_u,
                            flip_sign * flcc_u,
                            flip_sign * flss_u,
                        ),
                        use_numpy_arrays=False,
                        use_numpy_enforce=False,
                    )
                    freeb_bsqvac_half_dir = _freeb_bsqvac_half_for_trial_state(state_dir)
                    w_dir = _trial_residual_total(
                        state_dir,
                        freeb_bsqvac_half_dir,
                        zero_m1_value=zero_m1,
                    )
                    if np.isfinite(w_dir) and (w_dir <= 1.5 * max(w_curr, 1e-30)):
                        state = state_dir
                        (
                            vRcc,
                            vRss,
                            vZsc,
                            vZcs,
                            vLsc,
                            vLcs,
                            vRsc,
                            vRcs,
                            vZcc,
                            vZss,
                            vLcc,
                            vLss,
                        ) = _zero_velocity_blocks_like(
                            vRcc,
                            vRss,
                            vZsc,
                            vZcs,
                            vLsc,
                            vLcs,
                            vRsc,
                            vRcs,
                            vZcc,
                            vZss,
                            vLcc,
                            vLss,
                        )
                        step_status = "fallback_direct"
                        restart_reason = "none"
                        huge_force_restart_count = 0
                        restart_path = "fallback_direct"
                        update_rms = _host_force_update_rms(
                            dt_direct,
                            frcc_u,
                            frss_u,
                            frsc_u,
                            frcs_u,
                            fzsc_u,
                            fzcs_u,
                            fzcc_u,
                            fzss_u,
                            flsc_u,
                            flcs_u,
                            flcc_u,
                            flss_u,
                        )
                        if adjoint_trace:
                            trace_entry["fallback_direct_dt"] = float(dt_direct)
                        catastrophic_restart = False
                if catastrophic_restart:
                    # Roll back state and zero velocity.
                    state = state_backup
                    vRcc, vRss, vZsc, vZcs, vLsc, vLcs = _zero_velocity_blocks_like(
                        vRcc, vRss, vZsc, vZcs, vLsc, vLcs
                    )
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
                    time_step = restart_update.time_step
                    ijacob = restart_update.ijacob
                    restart_reason = restart_update.restart_reason
                    step_status = restart_update.step_status
                    restart_path = restart_update.restart_path
                    max_coeff_delta_rms = restart_update.max_coeff_delta_rms
                    max_update_rms = restart_update.max_update_rms
                    bad_resets = restart_update.bad_resets
                    iter1 = restart_update.iter1
                    freeb_controls_cached = None
                    fsq_prev = restart_update.fsq_prev
                    fsq0_prev = restart_update.fsq0_prev
                    inv_tau = restart_update.inv_tau
                    update_rms = restart_update.update_rms
                    if bool(clear_cache_after_catastrophic):
                        _clear_preconditioner_cache_locals()
            if timing_enabled and t_state_update_start is not None:
                t_state_update_dispatch_done = time.perf_counter()
                try:
                    if has_jax():
                        jax.block_until_ready(state.Rcos)
                except Exception:
                    pass
                t_state_update_ready_done = time.perf_counter()
                timing_stats["update_state_ready"] += (
                    t_state_update_ready_done - float(t_state_update_dispatch_done)
                )
                timing_stats["update_state"] += t_state_update_ready_done - float(t_state_update_start)
            t_trace_finalize_start = time.perf_counter() if timing_enabled and adjoint_trace else None
            if adjoint_trace:
                _finalize_strict_update_adjoint_trace_entry(
                    trace_entry,
                    locals(),
                    adjoint_trace_mode=adjoint_trace_mode,
                )
                adjoint_step_trace_history.append(trace_entry)
            if timing_enabled and t_trace_finalize_start is not None:
                timing_stats["update_trace_finalize"] += time.perf_counter() - float(t_trace_finalize_start)
            if timing_enabled and t_update_start is not None:
                try:
                    if has_jax():
                        jax.block_until_ready(state.Rcos)
                except Exception:
                    pass
                timing_stats["update"] += time.perf_counter() - float(t_update_start)
            timing_stats["iterations"] += 1
            if track_history:
                step_history.append(float(dt_eff))
                w_curr_history.append(float(w_curr))
                w_try_history.append(float(w_try))
                w_try_ratio_history.append(float(w_try_ratio))
                restart_path_history.append(str(restart_path))
        else:
            w_curr = fsqr_f + fsqz_f + fsql_f
            non_strict_update = _backtracking_momentum_search(
                state=state,
                velocities=_ResidualVelocityBlocks(
                    vRcc, vRss, vRsc, vRcs, vZsc, vZcs, vZcc, vZss, vLsc, vLcs, vLcc, vLss
                ),
                forces=_ResidualVelocityBlocks(
                    frcc_u, frss_u, frsc_u, frcs_u, fzsc_u, fzcs_u, fzcc_u, fzss_u, flsc_u, flcs_u, flcc_u, flss_u
                ),
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
            (vRcc, vRss, vRsc, vRcs, vZsc, vZcs, vZcc, vZss, vLsc, vLcs, vLcc, vLss) = (
                non_strict_update.velocities
            )
            dt_eff = non_strict_update.dt_eff
            update_rms = non_strict_update.update_rms
            step_status = non_strict_update.step_status
            timing_stats["iterations"] += 1
            if track_history:
                restart_reason = "none"
                step_history.append(dt_eff)
                w_curr_history.append(float(w_curr))
                w_try_history.append(float("nan"))
                w_try_ratio_history.append(float("nan"))
                restart_path_history.append("non_strict")
        t_iteration_post_update_start = time.perf_counter() if timing_enabled else None
        _dump_evolve_trace(
            iter2=int(iter2),
            iter1=int(iter1),
            stage="post",
            fsq1_val=float(fsq1),
            fsq_prev_val=float(fsq_prev_before),
            time_step_val=float(time_step),
            dtau_val=float(dtau),
            b1_val=float(b1),
            fac_val=float(fac),
            state_val=state,
            vRcc_val=vRcc,
            vRss_val=vRss,
            vZsc_val=vZsc,
            vZcs_val=vZcs,
            vLsc_val=vLsc,
            vLcs_val=vLcs,
            vRsc_val=vRsc,
            vRcs_val=vRcs,
            vZcc_val=vZcc,
            vZss_val=vZss,
            vLcc_val=vLcc,
            vLss_val=vLss,
            frcc_val=frcc_u,
            frss_val=frss_u,
            fzsc_val=fzsc_u,
            fzcs_val=fzcs_u,
            flsc_val=flsc_u,
            flcs_val=flcs_u,
            frsc_val=frsc_u,
            frcs_val=frcs_u,
            fzcc_val=fzcc_u,
            fzss_val=fzss_u,
            flcc_val=flcc_u,
            flss_val=flss_u,
        )
        _maybe_dump_xc(
            state=state,
            vRcc=vRcc,
            vRss=vRss,
            vZsc=vZsc,
            vZcs=vZcs,
            vLsc=vLsc,
            vLcs=vLcs,
            vRsc=vRsc,
            vRcs=vRcs,
            vZcc=vZcc,
            vZss=vZss,
            vLcc=vLcc,
            vLss=vLss,
            static=static,
            iter_idx=int(iter2),
        )
        if track_history:
            dt_eff_history.append(float(dt_eff))
            update_rms_history.append(update_rms_j if bool(strict_update) else float(update_rms))
        if bool(verbose):
            update_rms_print = float(np.asarray(update_rms_j)) if bool(strict_update) else float(update_rms)
        else:
            update_rms_print = 0.0
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
        if track_history:
            _append_residual_iter_terminal_history(
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
                freeb_ivac=freeb_ivac,
                freeb_ivacskip=freeb_ivacskip,
                freeb_reused=freeb_reused,
                freeb_solve_time=freeb_solve_time,
                freeb_sample_time=freeb_sample_time,
                **_terminal_history_lists,
            )
        # VMEC eqsolve behavior: when `ivac==1`, print turn-on and promote to
        # `ivac=2` for subsequent iterations.
        if free_boundary_enabled and int(freeb_ivac) == 1:
            if verbose and bool(verbose_vmec2000_table):
                print(f"\n  VACUUM PRESSURE TURNED ON AT {int(iter2):4d} ITERATIONS\n", flush=True)
            freeb_ivac = int(freeb_ivac) + 1
        skip_time_control = False
        if timing_enabled and t_iteration_post_update_start is not None:
            timing_stats["iteration_post_update"] += time.perf_counter() - float(t_iteration_post_update_start)

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
