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

# ruff: noqa: F401
# This module still owns many private compatibility aliases that are re-exported
# by ``vmec_jax.solve``.  Some are consumed by fixed-boundary API wrappers or by
# downstream diagnostic monkeypatch workflows rather than by this file directly.

from __future__ import annotations

from contextlib import nullcontext
from collections import OrderedDict
from functools import partial
import time
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np

from vmec_jax._compat import has_jax, jax, jnp, jit
from vmec_jax import _solve_runtime
from vmec_jax.solvers.fixed_boundary.residual import policy as _residual_iter_policy
from vmec_jax.solvers.fixed_boundary.residual.config import (
    HEAVY_DUMP_ENVS as _HEAVY_DUMP_ENVS,
    LIGHT_DUMP_ENVS as _LIGHT_DUMP_ENVS,
    bad_jacobian_tau_tolerance as _bad_jacobian_tau_tolerance,
    normalize_debug_print_mode as _normalize_debug_print_mode,  # noqa: F401 - re-exported for internal helpers/tests.
    parse_bad_jacobian_config as _parse_bad_jacobian_config,  # noqa: F401 - re-exported for internal helpers/tests.
    resolve_axis_reset_config as _resolve_axis_reset_config,
    resolve_chunked_scan_config as _resolve_chunked_scan_config,  # noqa: F401 - re-exported for internal helpers/tests.
    resolve_debug_print_config as _resolve_debug_print_config,
    resolve_host_profile_setup as _resolve_host_profile_setup,
    resolve_nstep_screen as _resolve_nstep_screen,
    resolve_setup_host_enforce as _resolve_setup_host_enforce,
    should_probe_bad_jacobian_state as _should_probe_bad_jacobian_state,
)
from vmec_jax.solvers.fixed_boundary.residual.policy import (
    append_residual_iter_history_record as _append_residual_iter_history_record,
    append_residual_iter_terminal_history as _append_residual_iter_terminal_history,
    host_restart_decision as _host_restart_decision,
    numpy_preconditioner_apply_policy as _numpy_preconditioner_apply_policy,
    resolve_residual_iter_startup_policy as _resolve_residual_iter_startup_policy,
    residual_iter_history_record as _residual_iter_history_record,
    scan_fallback_decision as _scan_fallback_decision,
    scan_fallback_message as _scan_fallback_message,
    vmec2000_scan_options_from_env as _vmec2000_scan_options_from_env,
    vmec2000_time_control_decision as _vmec2000_time_control_decision,
)
from vmec_jax.solvers.fixed_boundary.residual.runtime import (
    _attach_free_boundary_external_field_diag as _runtime_attach_free_boundary_external_field_diag,
    _converged_residuals_scan_fast as _runtime_converged_residuals_scan_fast,
    _device_get_floats,
    _initial_setup_phase_timings,
    _maybe_dump_ptau as _runtime_maybe_dump_ptau,
    _maybe_print_nonscan_state_debug,
    _new_residual_iter_timing_stats as _runtime_new_residual_iter_timing_stats,
    _record_compute_force_timing as _runtime_record_compute_force_timing,
    _record_setup_timing as _runtime_record_setup_timing,
    _setup_timer_start as _runtime_setup_timer_start,
    _vmec_freeb_plascur_from_bcovar as _runtime_vmec_freeb_plascur_from_bcovar,
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
    attach_residual_iter_timing_diagnostics as _attach_residual_iter_timing_diagnostics,
    build_residual_iter_resume_state_payload as _build_residual_iter_resume_state_payload,
    finalize_residual_iter_result as _finalize_residual_iter_result,
    precompile_only_residual_iter_result as _precompile_only_residual_iter_result,
    vmec2000_state_only_scan_result as _vmec2000_state_only_scan_result,
    vmec2000_traced_scan_result as _vmec2000_traced_scan_result,
)
from vmec_jax.solvers.fixed_boundary.residual.force_cache import (
    compute_forces_jit_cache_key as _compute_forces_jit_cache_key,
    maybe_precompile_residual_force_kernels as _maybe_precompile_residual_force_kernels,
    prepare_numpy_force_fast_path as _prepare_numpy_force_fast_path,
    select_compute_forces_callable as _select_compute_forces_callable,
)
from vmec_jax.solvers.fixed_boundary.residual.force_payload import (
    evaluate_residual_force_from_state as _evaluate_residual_force_from_state,
    force_z_channel_square_sums as _force_z_channel_square_sums,  # noqa: F401 - compatibility alias for tests/internal users.
    maybe_debug_force_z_channel_square_sums as _maybe_debug_force_z_channel_square_sums,  # noqa: F401 - compatibility alias for tests/internal users.
    residual_force_payload_after_m1_scalxc_with_scan_debug as _residual_force_payload_after_m1_scalxc_with_scan_debug,  # noqa: F401 - compatibility alias for tests/internal users.
    residual_force_gcx2_after_edge_policy as _residual_force_gcx2_after_edge_policy,  # noqa: F401 - compatibility alias for tests/internal users.
    residual_force_payload_from_kernels as _residual_force_payload_from_kernels,
    resolve_residual_force_mask_pack as _resolve_residual_force_mask_pack,  # noqa: F401 - compatibility alias for tests/internal users.
)
from vmec_jax.solvers.fixed_boundary.residual.update import (
    ResidualVelocityBlocks as _ResidualVelocityBlocks,
    force_update_rms as _force_update_rms,
    host_catastrophic_restart_update as _host_catastrophic_restart_update,
    host_force_update_rms as _host_force_update_rms,
    host_momentum_update_np as _host_momentum_update_np,
    momentum_update_jax as _momentum_update_jax,
    scale_velocity_blocks as _scale_velocity_blocks,
    zero_velocity_blocks_like as _zero_velocity_blocks_like,
)
from vmec_jax.field import TWOPI, b2_from_bsup, bsup_from_geom, bsup_from_sqrtg_lambda
from vmec_jax.fourier import eval_fourier_dtheta, eval_fourier_dzeta_phys
from vmec_jax.geom import eval_geom
from vmec_jax.grids import angle_steps
from vmec_jax.solvers.fixed_boundary.jit_cache import (
    jit_cache_get as _jit_cache_get,
    jit_cache_limit as _jit_cache_limit,  # noqa: F401 - re-exported for existing internal tests/importers.
    jit_cache_put as _jit_cache_put,
    record_scan_runner_cache_miss_categories as _record_scan_runner_cache_miss_categories,
)
from vmec_jax.solvers.fixed_boundary.diagnostics import hlo as _hlo_dump_helpers
from vmec_jax.solvers.fixed_boundary.diagnostics import first_step as _first_step_diagnostics_helpers
from vmec_jax.solvers.fixed_boundary.optimization import energy as _fixed_boundary_energy_helpers
from vmec_jax.solvers.fixed_boundary.optimization import gd as _fixed_boundary_gd_helpers
from vmec_jax.solvers.fixed_boundary.optimization import lambda_gd as _lambda_optimizer_helpers
from vmec_jax.solvers.fixed_boundary.optimization import lbfgs as _fixed_boundary_lbfgs_helpers
from vmec_jax.solvers.fixed_boundary.optimization import residual_context as _residual_force_context_helpers
from vmec_jax.solvers.fixed_boundary.optimization import residual_gn as _residual_gn_helpers
from vmec_jax.solvers.fixed_boundary.optimization import residual_lbfgs as _residual_lbfgs_helpers
from vmec_jax.solvers.fixed_boundary.diagnostics.io import (
    _dump_freeb_axis_trace_record,
    _dump_freeb_control_trace_record,
    _dump_time_control_trace_record,
    _finite_float_or_zero,
    _format_axis_coeff,  # noqa: F401 - re-exported for existing internal tests/importers.
    _format_checkpoint_log_row as _format_checkpoint_log_row,
    _format_evolve_trace_row as _format_evolve_trace_row,
    _format_freeb_control_trace_row as _format_freeb_control_trace_row,
    _format_time_control_log_row as _format_time_control_log_row,
    _format_time_control_trace_row,  # noqa: F401 - re-exported for existing internal tests/importers.
    _format_vmec2000_iter_row,  # noqa: F401 - re-exported for existing internal tests/importers.
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
    validate_fixed_boundary_gd_options,
    validate_fixed_boundary_lbfgs_options,
    validate_lambda_gd_options,
    validate_pressure_shape,
    validate_residual_gn_options,
    validate_residual_iteration_options,
    validate_residual_lbfgs_options,
)
from vmec_jax.solvers.fixed_boundary.optimization.quasi_newton import (
    ensure_descent_direction as _ensure_descent_direction,
    lbfgs_curvature_tolerance as _resolve_lbfgs_curvature_tol,
    lbfgs_two_loop_direction as _lbfgs_two_loop_direction,
)
from vmec_jax.solvers.fixed_boundary.profiles import (
    _half_mesh_from_full_mesh,
    _icurv_full_mesh_from_indata,
    _mass_half_mesh_from_indata,
    _pressure_half_mesh_from_indata,
    _s_half_from_full_mesh_s,  # noqa: F401 - re-exported for existing internal tests/importers.
    _vmec_force_flux_profiles,
    build_wout_like_profiles_from_indata as _build_wout_like_profiles_from_indata,
)
from vmec_jax.solvers.fixed_boundary.residual.geometry import (
    _m1_internal_to_physical_pair as _geometry_m1_internal_to_physical_pair,
    _mn_sin_to_signed_physical_batch as _geometry_mn_sin_to_signed_physical_batch,
    _rz_norm_np as _geometry_rz_norm_np,
)
from vmec_jax.solvers.fixed_boundary.residual.mode_transform import (
    build_mode_transform_context as _build_mode_transform_context,
    build_mode_transform_host_projection as _build_mode_transform_host_projection,
    mn_cos_to_signed_host_projected as _mn_cos_to_signed_host_projected,
    mn_sin_to_signed_host_projected as _mn_sin_to_signed_host_projected,
    mode_diag_weights_mn as _mode_diag_weights_mn_helper,
    mode_diag_weights_mn_np as _mode_diag_weights_mn_np_helper,
    vmec_scalxc_from_s_np as _vmec_scalxc_from_s_np_helper,
)
from vmec_jax.solvers.fixed_boundary.residual.payload_blocks import (
    ForceBlocks as _ForceBlocks,
    normalize_force_blocks as _normalize_force_blocks,  # noqa: F401 - re-exported for internal tests/importers.
    preconditioner_output_blocks_jax as _preconditioner_output_blocks_jax,
    preconditioner_output_blocks_np as _preconditioner_output_blocks_np,
    radial_preconditioner_output_blocks_jax as _radial_preconditioner_output_blocks_jax,
    residual_force_payload_after_m1_scalxc as _residual_force_payload_after_m1_scalxc,  # noqa: F401 - compatibility alias for tests/internal users.
    residual_force_payload_m1_scalxc_stages as _residual_force_payload_m1_scalxc_stages,  # noqa: F401 - compatibility alias for tests/internal users.
    zero_edge_rz_force_block as _zero_edge_rz_force_block,  # noqa: F401 - re-exported for internal tests/importers.
    zero_edge_rz_force_blocks as _zero_edge_rz_force_blocks,
)
from vmec_jax.solvers.fixed_boundary.residual.force_norms import (
    lambda_preconditioned_full_norm as _lambda_preconditioned_full_norm,
    mode_weight_force_blocks_jax as _mode_weight_force_blocks_jax,
    mode_weight_force_blocks_np as _mode_weight_force_blocks_np,
    residual_fsq_from_norms as _residual_fsq_from_norms,
    safe_dt_from_force_blocks as _safe_dt_from_force_blocks,
)
from vmec_jax.solvers.fixed_boundary.residual.host_diagnostics import (
    resolve_vmec2000_print_context as _resolve_vmec2000_print_context,
)
from vmec_jax.solvers.fixed_boundary.residual.scan_adapters import (
    ScanConvergencePredicate,
    ScanDeviceRuntime,
    ScanTimeControlDumper,
    ScanVmec2000PrintContext,
    build_vmec2000_scan_runtime_setup as _build_vmec2000_scan_runtime_setup,
    scan_m1_preconditioner_rhs as _scan_m1_preconditioner_rhs,
)
from vmec_jax.solvers.fixed_boundary.residual import preconditioner_payload as _precond_payload_facade
from vmec_jax.solvers.fixed_boundary.residual.preconditioner_payload import (
    _ACCEPTED_CONTROL_PAYLOAD_JIT_CACHE,
    _PRECOND_APPLY_PAYLOAD_JIT_CACHE,
    _PRECOND_OUTPUT_PAYLOAD_JIT_CACHE,
    _PRECOND_OUTPUT_SCALE_JIT_CACHE,
    _STRICT_UPDATE_STEP_JIT_CACHE,
)
from vmec_jax.solvers.fixed_boundary.residual.ptau import (
    accepted_control_ptau_arrays as _accepted_control_ptau_arrays_helper,
    maybe_dump_jacobian_terms as _maybe_dump_jacobian_terms_helper,
    maybe_dump_ptau as _maybe_dump_ptau_helper,
    ptau_minmax as _ptau_minmax_helper,
    ptau_minmax_from_k_host as _ptau_minmax_from_k_host_helper,
    ptau_minmax_from_k_jax as _ptau_minmax_from_k_jax_helper,
)
from vmec_jax.solvers.fixed_boundary.optimization.tolerances import (
    dtype_eps as _dtype_eps,  # noqa: F401 - re-exported for existing internal tests/importers.
    dtype_tiny as _dtype_tiny,
    resolve_cg_tol as _resolve_cg_tol,
    resolve_grad_tol as _resolve_grad_tol,
    resolve_lm_damping as _resolve_lm_damping,
)
from vmec_jax.solvers.fixed_boundary.optimization.constraints import (
    apply_vmec_lambda_axis_rules_to_state as _apply_vmec_lambda_axis_rules_to_state,
    axis_m0_mask as _axis_m0_mask,  # noqa: F401 - re-exported for existing internal tests/importers.
    enforce_field_rows as _enforce_field_rows,  # noqa: F401 - re-exported for existing internal tests/importers.
    enforce_field_rows_np as _enforce_field_rows_np,  # noqa: F401 - re-exported for existing internal tests/importers.
    enforce_fixed_boundary_and_axis as _enforce_fixed_boundary_and_axis,
    enforce_fixed_boundary_and_axis_np as _enforce_fixed_boundary_and_axis_np,
    enforce_lambda_gauge as _enforce_lambda_gauge,
    grad_rms_state as _grad_rms_state,
    mode00_index as _mode00_index,
    replace_mode_slice as _replace_mode_slice,  # noqa: F401 - re-exported for existing internal tests/importers.
    replace_mode_slice_np as _replace_mode_slice_np,  # noqa: F401 - re-exported for existing internal tests/importers.
    scale_mode_slice as _scale_mode_slice,  # noqa: F401 - re-exported for existing internal tests/importers.
    scale_mode_slice_np as _scale_mode_slice_np,  # noqa: F401 - re-exported for existing internal tests/importers.
    zero_coeff_column as _zero_coeff_column,  # noqa: F401 - re-exported for existing internal tests/importers.
    zero_coeff_column_np as _zero_coeff_column_np,  # noqa: F401 - re-exported for existing internal tests/importers.
)
from vmec_jax.solvers.fixed_boundary.optimization.gradient import (
    mask_grad_for_constraints as _mask_grad_for_constraints,
    update_state_gd as _update_state_gd,
)
from vmec_jax.solvers.fixed_boundary.preconditioning.operators import (
    apply_preconditioner as _apply_preconditioner,
    can_reassemble_precond_mats as _can_reassemble_precond_mats,
    empty_preconditioner_cache_snapshot as _empty_preconditioner_cache_snapshot,
    lambda_preconditioner_outputs as _lambda_preconditioner_outputs,
    metric_surface_precond_from_bcovar_jax as _metric_surface_precond_from_bcovar_jax,
    metric_surface_precond_scales_jax as _metric_surface_precond_scales_jax,  # noqa: F401 - re-exported for existing internal tests/importers.
    metric_surface_precond_scales_np as _metric_surface_precond_scales_np,
    pshalf_from_s_jax as _pshalf_from_s_jax,
    pshalf_from_s_np as _pshalf_from_s_np,
    radial_tridi_smooth_dirichlet as _radial_tridi_smooth_dirichlet,
    resolve_preconditioner_cache_decision as _resolve_preconditioner_cache_decision,
    resolve_preconditioner_tridi_policies as _resolve_preconditioner_tridi_policies,
    scale_m1_precond_rhs_from_mats as _scale_m1_precond_rhs_from_mats,
    sm_sp_from_s_np as _sm_sp_from_s_np,  # noqa: F401 - re-exported for existing internal tests/importers.
    update_preconditioner_cache as _update_preconditioner_cache,
    vmec_scale_m1_factors_from_mats as _vmec_scale_m1_factors_from_mats,  # noqa: F401 - re-exported for existing internal tests/importers.
    vmec_scale_m1_factors_from_mats_np as _vmec_scale_m1_factors_from_mats_np,  # noqa: F401 - re-exported for existing internal tests/importers.
)
from vmec_jax.solvers.fixed_boundary.results import (
    ScanCarry as _ScanCarry,
    SolveFixedBoundaryResult,
    SolveLambdaResult,
    SolveVmecResidualResult,
    WoutLikeVmecForces as _WoutLikeVmecForces,
)
from vmec_jax.solvers.fixed_boundary.diagnostics.axis_reset import (
    InitialAxisResetDecision as _InitialAxisResetDecision,  # noqa: F401 - re-exported for existing internal tests/importers.
    bad_jacobian_from_tau_range as _axis_reset_bad_jacobian_from_tau_range,
    bad_jacobian_ptau_from_minmax as _axis_reset_bad_jacobian_ptau_from_minmax,
    initial_axis_reset_decision as _initial_axis_reset_decision,
    initial_force_physical_fsq as _axis_reset_initial_force_physical_fsq,
    merge_axis_reset_state as _merge_axis_reset_state,
    reset_axis_from_boundary as _reset_axis_from_boundary_impl,
    write_axis_reset_dump as _write_axis_reset_dump,
)
from vmec_jax.solvers.free_boundary.control import (
    free_boundary_iter_controls as _free_boundary_iter_controls,
    free_boundary_iter_controls_vmec as _free_boundary_iter_controls_vmec,
    free_boundary_prev_rz_fsq_next as _free_boundary_prev_rz_fsq_next,
    free_boundary_should_damp_constraint_baseline as _free_boundary_should_damp_constraint_baseline,
    free_boundary_turnon_resets_iter1_immediately as _free_boundary_turnon_resets_iter1_immediately,
)
from vmec_jax.solvers.free_boundary.diagnostics import sample_free_boundary_external_field as _sample_free_boundary_external_field
from vmec_jax.solvers.fixed_boundary.diagnostics.force import (
    dump_array as _dump_array,  # noqa: F401 - re-exported for internal tests/importers.
    gc_from_frzl as _gc_from_frzl,  # noqa: F401 - compatibility alias for internal tests/importers.
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
from vmec_jax.solvers.fixed_boundary.scan.resume import (
    ScanResumeInitialFields as _ScanResumeInitialFields,  # noqa: F401 - re-exported for existing internal tests/importers.
    build_initial_scan_carry as _build_initial_scan_carry,
    build_traced_scan_resume_state as _build_traced_scan_resume_state,
    initialize_scan_resume_state as _initialize_scan_resume_state,
)
from vmec_jax.solvers.fixed_boundary.optimization.residual_objective import (
    assemble_residual_objective_terms as _assemble_residual_objective_terms,
    residual_objective_vector as _residual_objective_vector,
)
from vmec_jax.solvers.fixed_boundary.scan.output import (
    finalize_vmec2000_scan_run,
    finalize_vmec2000_scan_step,
    postprocess_vmec2000_scan_result,
    unpack_vmec2000_scan_histories,
    vmec2000_scan_full_history_row,
    vmec2000_scan_light_history_row,
    vmec2000_scan_minimal_history_row,
    vmec2000_scan_residual_result,
    vmec2000_scan_step_result,
    vmec2000_state_only_scan_diagnostics,
    vmec2000_traced_scan_diagnostics,
)
from vmec_jax.solvers.fixed_boundary.scan.payload import (
    build_current_preconditioned_scan_payload as _build_current_preconditioned_scan_payload,
    build_initial_preconditioner_cache as _build_initial_preconditioner_cache,
    build_restart_preconditioned_scan_payload as _build_restart_preconditioned_scan_payload,
    build_scan_step_fields as _build_scan_step_fields,
    evaluate_scan_step_force as _evaluate_scan_step_force,
    mask_scan_restart_force_payload as _mask_scan_restart_force_payload,  # noqa: F401 - re-exported for internal tests/importers.
    select_payload_and_build_step_fields as _select_payload_and_build_step_fields,
    select_scan_force_payload as _select_scan_force_payload,
)
from vmec_jax.solvers.fixed_boundary.scan.math import (
    _hold_step as _scan_math_hold_step,
    _kernel_arrays_from_k as _scan_math_kernel_arrays_from_k,
    _no_restart_updates as _scan_math_no_restart_updates,
    _ptau_minmax_from_context_host as _scan_math_ptau_minmax_from_context_host,
    _ptau_minmax_from_context_jax as _scan_math_ptau_minmax_from_context_jax,
    _ptau_minmax_from_k_host as _scan_math_ptau_minmax_from_k_host,
    _ptau_minmax_from_k_jax as _scan_math_ptau_minmax_from_k_jax,
    _restart_updates as _scan_math_restart_updates,
    scan_bad_jacobian_decision as _scan_bad_jacobian_decision,
    _state_jacobian as _scan_math_state_jacobian,
    build_ptau_minmax_context as _build_ptau_minmax_context,
)
from vmec_jax.solvers.fixed_boundary.scan.debug import (
    dump_vmec2000_scan_ptau_rows as _dump_vmec2000_scan_ptau_rows,
    emit_live_scan_vmec2000_row as _emit_live_scan_vmec2000_row,
    emit_vmec2000_post_scan_rows as _emit_vmec2000_post_scan_rows,
    maybe_debug_scan_force_first_iter as _maybe_debug_scan_force_first_iter,
    maybe_debug_scan_state_iter as _maybe_debug_scan_state_iter,
    _emit_vmec2000_iter_row as _emit_scan_vmec2000_iter_row,
    _emit_scan_prints as _emit_scan_debug_prints,
    _print_vmec2000_row as _print_scan_vmec2000_row,
    _record_scan_device_ready,
)
from vmec_jax.solvers.fixed_boundary.scan.planning import (
    build_scan_timing_report as _build_scan_timing_report,
    build_vmec2000_scan_cache_key as _build_vmec2000_scan_cache_key,
    default_vmec2000_controller_constants as _default_vmec2000_controller_constants,
    new_scan_timing_stats as _new_scan_timing_stats,
    resolve_scan_iteration_plan as _resolve_scan_iteration_plan,
    resolve_scan_iteration_runtime_plan as _resolve_scan_iteration_runtime_plan,
    resolve_scan_preflight_iters as _resolve_scan_preflight_iters,
    resolve_vmec2000_scan_setup as _resolve_vmec2000_scan_setup,
    scan_chunk_settings as _resolve_scan_chunk_settings,
    scan_jit_forces_enabled as _scan_jit_forces_enabled,
    scan_jit_preflight_enabled as _scan_jit_preflight_enabled,
    scan_timing_enabled as _scan_timing_enabled,
    validate_vmec2000_scan_guards as _validate_vmec2000_scan_guards,
)
from vmec_jax.solvers.fixed_boundary.scan.runtime import (
    get_or_build_scan_runner as _get_or_build_scan_runner,
    resolve_scan_runtime_hooks_from_env as _resolve_scan_runtime_hooks_from_env,
    run_chunked_scan as _run_chunked_scan,
    run_nonchunked_scan as _run_nonchunked_scan,
    run_scan_preflight_step as _run_scan_preflight_step,
    scan_trace_context_or_null as _scan_trace_context_or_null,
)
from vmec_jax.solvers.fixed_boundary.scan.time_control import (
    evaluate_scan_time_control_restart,
    scan_restart_transition,
)
from vmec_jax.state import VMECState, pack_state, unpack_state
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


def _strict_update_step_jit(*args, **kwargs):
    return _precond_payload_facade._strict_update_step_jit(*args, has_jax_func=has_jax, **kwargs)


def _preconditioner_output_scaling_jit(*args, **kwargs):
    return _precond_payload_facade._preconditioner_output_scaling_jit(*args, has_jax_func=has_jax, **kwargs)


def _preconditioner_output_payload_jit(*args, **kwargs):
    return _precond_payload_facade._preconditioner_output_payload_jit(*args, has_jax_func=has_jax, **kwargs)


def _preconditioner_apply_payload_jit(*args, **kwargs):
    return _precond_payload_facade._preconditioner_apply_payload_jit(*args, has_jax_func=has_jax, **kwargs)


def _accepted_control_payload_jit():
    return _precond_payload_facade._accepted_control_payload_jit(has_jax_func=has_jax)


def _preconditioner_apply_payload_fused(*args, **kwargs):
    return _precond_payload_facade._preconditioner_apply_payload_fused(*args, **kwargs)


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
_scalar_history_array = _solve_runtime._scalar_history_array


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


_HLO_DUMPED_KEYS = _hlo_dump_helpers.HLO_DUMPED_KEYS


def _maybe_dump_hlo_kernel(
    *,
    label: str,
    fn,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    static: Any,
    wout_like: Any,
    force: bool = False,
) -> None:
    _hlo_dump_helpers.maybe_dump_hlo_kernel(
        label=label,
        fn=fn,
        args=args,
        kwargs=kwargs,
        static=static,
        wout_like=wout_like,
        force=force,
        has_jax_func=has_jax,
        path_cls=Path,
    )


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

    timing_env = os.getenv("VMEC_JAX_TIMING", "").strip().lower()
    timing_enabled = timing_env not in ("", "0", "false", "no")
    timing_detail_env = os.getenv("VMEC_JAX_TIMING_DETAIL", "").strip().lower()
    timing_detail_enabled = timing_enabled and timing_detail_env not in ("", "0", "false", "no")
    _setup_phase_timings = _initial_setup_phase_timings()
    state0_has_tracer = _tree_has_tracer(state0)

    def _setup_timer_start() -> float | None:
        return _runtime_setup_timer_start(timing_enabled=bool(timing_enabled), perf_counter=time.perf_counter)

    def _record_setup_timing(key: str, start: float | None) -> None:
        _runtime_record_setup_timing(_setup_phase_timings, key, start, perf_counter=time.perf_counter)

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

    def _adjoint_trace_array(value):
        return _materialize_adjoint_trace_array(value, mode=adjoint_trace_mode)

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

    def _pack_resume_state(base: dict[str, Any], heavy: dict[str, Any] | None = None):
        return _pack_resume_state_record(base=base, heavy=heavy, mode=resume_state_mode)

    from vmec_jax.static import build_static
    from vmec_jax.boundary import boundary_from_indata
    from vmec_jax.init_guess import (
        _recompute_axis_from_boundary,
        _recompute_axis_from_state_vmec,
        _read_axis_coeffs,
        initial_guess_from_boundary,
    )
    from vmec_jax.vmec_forces import vmec_forces_rz_from_wout, vmec_residual_internal_from_kernels
    from vmec_jax.vmec_residue import (
        vmec_force_norms_from_bcovar_dynamic,
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

    def _attach_freeb_diag(res: SolveVmecResidualResult) -> SolveVmecResidualResult:
        return _runtime_attach_free_boundary_external_field_diag(
            res,
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

    def _apply_vmec_lambda_axis_rules(st: VMECState) -> VMECState:
        """Enforce VMEC lambda gauge without mutating stored axis coefficients.

        VMEC applies the m=0 lambda axis-closure during real-space synthesis
        (totzsps) but does not overwrite the stored `xc` coefficients. Keep
        the state axis row intact and only enforce the (m,n)=(0,0) gauge here.
        """
        return _apply_vmec_lambda_axis_rules_to_state(
            st,
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

    def _compute_forces(
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
        scan_debug_force_enabled = os.getenv("VMEC_JAX_SCAN_DEBUG_FORCE", "") not in ("", "0")
        dump_hlo_force_tomnsps = os.getenv("VMEC_JAX_DUMP_HLO_FORCE_TOMNSPS", "").strip().lower() not in (
            "",
            "0",
            "false",
            "no",
        )

        def _dump_force_tomnsps_hlo(*, label, fn, args, kwargs) -> None:
            _maybe_dump_hlo_kernel(
                label=label,
                fn=fn,
                args=args,
                kwargs=kwargs,
                static=static,
                wout_like=wout_like,
                force=True,
            )

        force_eval = _evaluate_residual_force_from_state(
            state=state,
            static=static,
            wout_like=wout_like,
            trig=trig,
            s=s,
            signgs=signgs,
            constraint_tcon0=constraint_tcon0,
            freeb_pres_scale=freeb_pres_scale,
            apply_lforbal=apply_lforbal,
            apply_m1_constraints=bool(apply_m1_constraints),
            include_edge=bool(include_edge),
            include_edge_residual=include_edge_residual,
            zero_m1=zero_m1,
            freeb_bsqvac_half=freeb_bsqvac_half,
            constraint_rcon0=constraint_rcon0,
            constraint_zcon0=constraint_zcon0,
            constraint_precond_diag=constraint_precond_diag,
            constraint_tcon=constraint_tcon,
            constraint_precond_active=constraint_precond_active,
            constraint_tcon_active=constraint_tcon_active,
            iter_idx=iter_idx,
            scan_debug_force_enabled=bool(scan_debug_force_enabled),
            dump_hlo_force_tomnsps=bool(dump_hlo_force_tomnsps),
            hlo_dump_func=_dump_force_tomnsps_hlo,
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
        return (
            force_eval.kernels,
            force_eval.frzl_full,
            force_eval.gcr2,
            force_eval.gcz2,
            force_eval.gcl2,
            force_eval.rz_scale,
            force_eval.l_scale,
            force_eval.norms,
        )

    _hlo_dump_helpers.maybe_dump_initial_residual_hlo_kernels(
        state0=state0,
        static=static,
        wout_like=wout_like,
        trig=trig,
        constraint_tcon0=constraint_tcon0,
        apply_lforbal=bool(apply_lforbal),
        maybe_dump_kernel=_maybe_dump_hlo_kernel,
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
        if warmup_iters > 0 and (iter2 is not None) and (int(iter2) <= warmup_iters):
            if has_jax():
                import jax

                with jax.disable_jit():
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
                        iter_idx=iter_idx,
                    )
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
                iter_idx=iter_idx,
            )
        # NumPy fast path: use pure-NumPy force computation when available.
        # This eliminates all JAX dispatch overhead from the per-iteration loop.
        if _compute_forces_np is not None:
            return _compute_forces_np(
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
                iter_idx=iter_idx,
            )
        return _compute_forces(
            state,
            include_edge=include_edge,
            include_edge_residual=include_edge_residual,
            zero_m1=zero_m1,
            constraint_rcon0=constraint_rcon0,
            constraint_zcon0=constraint_zcon0,
            constraint_precond_diag=constraint_precond_diag,
            constraint_tcon=constraint_tcon,
            constraint_precond_active=constraint_precond_active,
            constraint_tcon_active=constraint_tcon_active,
            iter_idx=iter_idx,
            **({"freeb_bsqvac_half": freeb_bsqvac_half} if freeb_bsqvac_half is not None else {}),
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

    def _run_vmec2000_scan(state_init: VMECState) -> SolveVmecResidualResult:
        scan_runtime = _build_vmec2000_scan_runtime_setup(
            env=os.environ,
            state_init=state_init,
            indata=indata,
            cfg=cfg,
            mpol=mpol,
            nrange=nrange,
            resume_state=resume_state,
            state_only=bool(state_only),
            scan_fallback_enabled=bool(scan_fallback_enabled),
            force_chunked_scan=bool(startup_policy.force_chunked_scan),
            preconditioner_use_precomputed_tridi=preconditioner_use_precomputed_tridi,
            preconditioner_use_lax_tridi=preconditioner_use_lax_tridi,
            verbose=bool(verbose),
            vmec2000_control=bool(vmec2000_control),
            verbose_vmec2000_table=bool(verbose_vmec2000_table),
            light_history=bool(light_history),
            scan_minimal_default=scan_minimal_default,
            dump_any=bool(startup_policy.dump_any),
            fsq_total_target=fsq_total_target,
            axis_reset_done=bool(axis_reset_done),
            lmove_axis=bool(lmove_axis),
            step_size=float(step_size),
            initial_flip_sign=float(initial_flip_sign),
            ftol=float(ftol),
            jit_forces=bool(jit_forces),
            compute_forces=_compute_forces,
            compute_forces_impl=_compute_forces_impl,
            scan_timing_enabled_func=_scan_timing_enabled,
            new_scan_timing_stats_func=_new_scan_timing_stats,
            scan_backend_name_func=_scan_backend_name,
            tree_has_tracer_func=_tree_has_tracer,
            validate_vmec2000_scan_guards_func=_validate_vmec2000_scan_guards,
            resolve_vmec2000_scan_setup_func=_resolve_vmec2000_scan_setup,
            default_vmec2000_controller_constants_func=_default_vmec2000_controller_constants,
            resolve_scan_runtime_hooks_from_env_func=_resolve_scan_runtime_hooks_from_env,
            scan_jit_forces_enabled_func=_scan_jit_forces_enabled,
            scan_trace_context_or_null_func=_scan_trace_context_or_null,
            initialize_scan_resume_state_func=_initialize_scan_resume_state,
            scan_m1_preconditioner_rhs_func=_scan_m1_preconditioner_rhs,
            scale_m1_precond_rhs_from_mats_func=_scale_m1_precond_rhs_from_mats,
            converged_func=_runtime_converged_residuals_scan_fast,
            record_scan_device_ready_func=_record_scan_device_ready,
            has_jax_func=has_jax,
            jax_module=jax,
            jnp_module=jnp,
            time_module=time,
            backtracking=bool(backtracking),
            limit_dt_from_force=bool(limit_dt_from_force),
            limit_update_rms=bool(limit_update_rms),
            use_direct_fallback=bool(use_direct_fallback),
            reference_mode=bool(reference_mode),
            strict_update=bool(strict_update),
            auto_flip_force=bool(auto_flip_force),
        )

        scan_timing_enabled = scan_runtime.timing_enabled
        scan_timing_stats = scan_runtime.timing_stats
        scan_total_start = scan_runtime.total_start
        scan_device_runtime = scan_runtime.device_runtime
        scan_differentiated = scan_runtime.scan_differentiated
        state_only_scan = scan_runtime.state_only_scan
        scan_fallback_enabled_run = scan_runtime.scan_fallback_enabled_run
        controller_constants = scan_runtime.controller_constants
        k_preconditioner_update_interval = controller_constants.preconditioner_update_interval
        restart_badjac_factor = controller_constants.restart_badjac_factor
        restart_badprog_factor = controller_constants.restart_badprog_factor
        vmec2000_fact = controller_constants.vmec2000_fact
        iter_offset0 = scan_runtime.iter_offset0
        nstep_screen = scan_runtime.nstep_screen
        scan_options = scan_runtime.options
        scan_print_ordered = scan_options.scan_print_ordered
        scan_light = scan_options.scan_light
        scan_minimal = scan_options.scan_minimal
        scan_collect_scalars = scan_options.scan_collect_scalars
        scan_collect_print = scan_options.scan_collect_print
        scan_core = scan_options.scan_core
        abort_scan_on_badjac = scan_options.abort_scan_on_badjac
        scan_use_precomputed = scan_options.scan_use_precomputed
        scan_use_lax_tridi = scan_options.scan_use_lax_tridi
        # On GPU/TPU, lax.cond executes BOTH branches unconditionally. The
        # restart payload (_restart_payload) re-runs the full vmec_bcovar +
        # force computation for the checkpoint state, doubling per-iteration
        # cost even when restarts are rare. On CPU, lax.cond branches are
        # selected at Python level (Python loop), so this overhead is avoided.
        # Default: use restart payload on CPU only; skip it on GPU/TPU.
        scan_use_restart_payload = scan_options.scan_use_restart_payload
        dump_timecontrol_scan = scan_runtime.dump_timecontrol_scan
        chunked_print = scan_runtime.chunked_print
        print_in_scan = scan_runtime.print_in_scan
        scan_print_mode = scan_runtime.scan_print_mode

        def _maybe_trace(label: str):
            return scan_runtime.maybe_trace(label)

        axis_reset_enabled = scan_runtime.axis_reset_enabled
        axis_reset_repeat = scan_runtime.axis_reset_repeat
        scan_print_context = scan_runtime.print_context
        dtype = scan_runtime.dtype
        scan_timecontrol_dumper = scan_runtime.timecontrol_dumper
        flip_sign0 = scan_runtime.flip_sign0
        scan_converged = scan_runtime.converged
        k_ndamp = controller_constants.ndamp
        scan_resume0 = scan_runtime.resume_fields
        scale_m1_precond_rhs = scan_runtime.scale_m1_precond_rhs
        jit_forces_scan = scan_runtime.jit_forces_scan
        _compute_forces_scan = scan_runtime.compute_forces_scan
        t_scan_initial_force = time.perf_counter() if scan_timing_enabled else None
        with _maybe_trace("scan/compute_forces:init"):
            k0, frzl0, gcr2_0, gcz2_0, gcl2_0, rz_scale0, l_scale0, norms0 = _compute_forces_scan(
                state_init,
                include_edge=False,
                zero_m1=jnp.asarray(1.0, dtype=dtype),
                constraint_precond_diag=zero_precond_diag,
                constraint_tcon=zero_tcon,
                constraint_precond_active=constraint_active_false,
                constraint_tcon_active=constraint_active_false,
                iter_idx=None,
            )
        if scan_timing_enabled and t_scan_initial_force is not None:
            try:
                if has_jax():
                    scan_device_runtime.block_value((gcr2_0, gcz2_0, gcl2_0))
            except Exception:
                pass
            scan_timing_stats["scan_initial_compute_forces_s"] += time.perf_counter() - float(
                t_scan_initial_force
            )
        fsq_phys0_val = _axis_reset_initial_force_physical_fsq(
            norms=norms0,
            gcr2=gcr2_0,
            gcz2=gcz2_0,
            gcl2=gcl2_0,
        )
        bad_jacobian0 = False
        if axis_reset_enabled:
            axis_reset_debug = os.getenv("VMEC_JAX_AXIS_RESET_DEBUG", "").strip().lower() not in (
                "",
                "0",
                "false",
                "no",
            )
            try:
                ptau_min0, ptau_max0 = _ptau_minmax_from_k_host(k0)
            except Exception:
                ptau_min0, ptau_max0 = None, None
            bad_jacobian_ptau = _axis_reset_bad_jacobian_ptau_from_minmax(
                ptau_min=ptau_min0,
                ptau_max=ptau_max0,
                ptau_tol=ptau_tol,
                ptau_tol_rel=ptau_tol_rel,
            )
            bad_jacobian_state = False
            if badjac_use_state:
                try:
                    jac0 = vmec_half_mesh_jacobian_from_state(
                        state=state_init,
                        modes=static.modes,
                        trig=trig,
                        s=s,
                        lconm1=bool(getattr(static.cfg, "lconm1", True)),
                        lthreed=bool(getattr(static.cfg, "lthreed", True)),
                        mask_even=getattr(static, "m_is_even", None),
                        mask_odd=getattr(static, "m_is_odd", None),
                    )
                    tau0 = jnp.asarray(jac0.tau)
                    tau0_use = tau0[1:] if int(tau0.shape[0]) > 1 else tau0
                    min_tau_state0 = float(np.asarray(jnp.min(tau0_use)))
                    max_tau_state0 = float(np.asarray(jnp.max(tau0_use)))
                    tau_scale_state0 = max(abs(min_tau_state0), abs(max_tau_state0))
                    bad_jacobian_state = _axis_reset_bad_jacobian_from_tau_range(
                        min_tau=min_tau_state0,
                        max_tau=max_tau_state0,
                        abs_tol=max(1.0e-12, 1.0e-2 * tau_scale_state0),
                    )
                except Exception:
                    bad_jacobian_state = False

            axis_reset_decision = _initial_axis_reset_decision(
                bad_jacobian_ptau=bad_jacobian_ptau,
                bad_jacobian_state=bad_jacobian_state,
                badjac_use_state=badjac_use_state,
                fsq_phys=fsq_phys0_val,
                axis_reset_fsq_min=axis_reset_fsq_min,
                force_axis_reset=force_axis_reset,
                axis_reset_always_3d=axis_reset_always_3d,
                lthreed=bool(getattr(static.cfg, "lthreed", True)),
                vmec2000_control=vmec2000_control,
                lmove_axis=lmove_axis,
                axis_reset_enabled=axis_reset_enabled,
            )
            bad_jacobian0 = axis_reset_decision.bad_jacobian
            if axis_reset_debug:
                try:
                    fsq_debug_val = float("nan") if fsq_phys0_val is None else float(fsq_phys0_val)
                    print(
                        "[axis_reset] fsq0="
                        f"{fsq_debug_val:.6e} "
                        f"axis_reset_fsq_min={axis_reset_fsq_min:.3e} "
                        f"badjac_ptau={bad_jacobian_ptau} badjac_state={bad_jacobian_state} "
                        f"badjac_used={bad_jacobian0}",
                        flush=True,
                    )
                except Exception:
                    pass
        else:
            axis_reset_decision = _initial_axis_reset_decision(
                bad_jacobian_ptau=None,
                bad_jacobian_state=False,
                badjac_use_state=badjac_use_state,
                fsq_phys=fsq_phys0_val,
                axis_reset_fsq_min=axis_reset_fsq_min,
                force_axis_reset=force_axis_reset,
                axis_reset_always_3d=axis_reset_always_3d,
                lthreed=bool(getattr(static.cfg, "lthreed", True)),
                vmec2000_control=vmec2000_control,
                lmove_axis=lmove_axis,
                axis_reset_enabled=axis_reset_enabled,
            )
        force_axis_reset_init = axis_reset_decision.force_reset
        if axis_reset_decision.reset:
            if bool(verbose) and bool(vmec2000_control) and bool(verbose_vmec2000_table):
                if bad_jacobian0 or force_axis_reset_init:
                    print(" INITIAL JACOBIAN CHANGED SIGN!", flush=True)
                print(" TRYING TO IMPROVE INITIAL MAGNETIC AXIS GUESS", flush=True)
            state_init = _reset_axis_from_boundary(state_init, k_guess=k0, full_reset=False, refine_axis_guess=False)
            if bool(verbose) and bool(vmec2000_control) and bool(verbose_vmec2000_table):
                if axis_reset_coeffs is not None:
                    raxis_cc, _raxis_cs, _zaxis_cc, zaxis_cs = axis_reset_coeffs
                    scan_print_context.print_axis_guess(raxis_cc, zaxis_cs)
            scan_resume0 = scan_resume0._replace(
                ijacob=jnp.asarray(1, dtype=jnp.int32),
                state_checkpoint=state_init,
            )
            axis_reset_enabled = False
            axis_reset_repeat = True
            t_scan_axis_force = time.perf_counter() if scan_timing_enabled else None
            k0, frzl0, gcr2_0, gcz2_0, gcl2_0, rz_scale0, l_scale0, norms0 = _compute_forces_scan(
                state_init,
                include_edge=False,
                zero_m1=jnp.asarray(1.0, dtype=dtype),
                constraint_precond_diag=zero_precond_diag,
                constraint_tcon=zero_tcon,
                constraint_precond_active=constraint_active_false,
                constraint_tcon_active=constraint_active_false,
                iter_idx=None,
            )
            if scan_timing_enabled and t_scan_axis_force is not None:
                try:
                    if has_jax():
                        scan_device_runtime.block_value((gcr2_0, gcz2_0, gcl2_0))
                except Exception:
                    pass
                scan_timing_stats["scan_axis_reset_compute_forces_s"] += time.perf_counter() - float(
                    t_scan_axis_force
                )
        # Axis reset handled before scan; avoid per-iteration callbacks.
        axis_reset_enabled = False
        scan_run_setup_start = time.perf_counter() if scan_timing_enabled else None
        initial_cache = _build_initial_preconditioner_cache(
            state_init=state_init,
            k=k0,
            norms=norms0,
            rz_scale=rz_scale0,
            l_scale=l_scale0,
            constraint_tcon0=constraint_tcon0,
            zero_precond_diag=zero_precond_diag,
            zero_tcon=zero_tcon,
            trig=trig,
            s=s,
            cfg=cfg,
            dtype=dtype,
            scan_use_precomputed=bool(scan_use_precomputed),
            scan_use_lax_tridi=bool(scan_use_lax_tridi),
            lambda_preconditioner_func=_lambda_preconditioner,
            rz_norm_func=_rz_norm,
            resume_state=resume_state,
        )
        cache_precond_diag0 = initial_cache.precond_diag
        cache_tcon0 = initial_cache.tcon
        cache_norms0 = initial_cache.norms
        cache_rz_scale0 = initial_cache.rz_scale
        cache_l_scale0 = initial_cache.l_scale
        cache_rz_norm0 = initial_cache.rz_norm
        cache_f_norm1_0 = initial_cache.f_norm1
        cache_lam_prec0 = initial_cache.lam_prec
        cache_rz_mats0 = initial_cache.rz_mats
        jmax0 = initial_cache.jmax
        cache_valid0 = initial_cache.valid

        def _tree_select(cond, t_true, t_false):
            if t_true is None or t_false is None:
                return t_true if t_false is None else t_false
            if isinstance(t_true, tuple) and isinstance(t_false, tuple):
                return type(t_true)(_tree_select(cond, a, b) for a, b in zip(t_true, t_false, strict=True))
            if isinstance(t_true, list) and isinstance(t_false, list):
                return [_tree_select(cond, a, b) for a, b in zip(t_true, t_false, strict=True)]
            return jnp.where(cond, jnp.asarray(t_true), jnp.asarray(t_false))

        scan_fallback_iters_j = jnp.asarray(int(scan_fallback_iters), dtype=jnp.int32)
        scan_fallback_badjac_limit_j = jnp.asarray(int(scan_fallback_badjac_limit), dtype=jnp.int32)
        scan_fallback_accept_frac_j = jnp.asarray(float(scan_fallback_accept_frac), dtype=dtype)
        scan_fallback_fsq_factor_j = jnp.asarray(float(scan_fallback_fsq_factor), dtype=dtype)
        scan_fallback_fsq_abs_j = jnp.asarray(float(scan_fallback_fsq_abs), dtype=dtype)
        scan_fallback_improve_j = jnp.asarray(float(startup_policy.scan_fallback_improve), dtype=dtype)

        scan_debug_force = os.getenv("VMEC_JAX_SCAN_DEBUG_FORCE", "") not in ("", "0")
        debug_iter_env = os.getenv("VMEC_JAX_SCAN_DEBUG_ITER", "").strip()
        try:
            scan_debug_iter = int(debug_iter_env) if debug_iter_env else -1
        except Exception:
            scan_debug_iter = -1

        def _scan_step(carry: _ScanCarry, it):
            def _hold_step(carry_hold: _ScanCarry):
                return _scan_math_hold_step(
                    carry_hold,
                    dtype=dtype,
                    state_only_scan=state_only_scan,
                    scan_minimal=scan_minimal,
                    scan_light=scan_light,
                    scan_hist_min=vmec2000_scan_minimal_history_row,
                    scan_hist_light=vmec2000_scan_light_history_row,
                )

            def _advance_step(carry_adv: _ScanCarry):
                force_eval = _evaluate_scan_step_force(
                    carry_adv=carry_adv,
                    it=it,
                    dtype=dtype,
                    k_preconditioner_update_interval=k_preconditioner_update_interval,
                    zero_precond_diag=zero_precond_diag,
                    zero_tcon=zero_tcon,
                    compute_forces_scan=_compute_forces_scan,
                    scan_converged=scan_converged,
                    tree_select=_tree_select,
                    cond=jax.lax.cond,
                    trace_context=lambda: _maybe_trace("scan/compute_forces"),
                    scan_debug_force_enabled=bool(scan_debug_force),
                    scan_debug_iter=int(scan_debug_iter),
                    debug_force_first_iter=_maybe_debug_scan_force_first_iter,
                    debug_state_iter=_maybe_debug_scan_state_iter,
                    debug_print=scan_runtime.jax_debug_print,
                )
                iter2 = force_eval.iter2
                fsq_prev_before = force_eval.fsq_prev_before
                fsq0_prev_before = force_eval.fsq0_prev_before
                skip_timecontrol = force_eval.skip_timecontrol
                time_step_report = force_eval.time_step_report
                zero_m1 = force_eval.zero_m1
                include_edge = force_eval.include_edge
                need_bcovar_update = force_eval.need_bcovar_update
                k = force_eval.kernels
                frzl = force_eval.frzl
                rz_scale = force_eval.rz_scale
                l_scale = force_eval.l_scale
                norms_current = force_eval.norms_current
                norms_used = force_eval.norms_used
                fsqr = force_eval.fsqr
                fsqz = force_eval.fsqz
                fsql = force_eval.fsql
                conv_now = force_eval.conv_now
                # Scalars for VMEC-style screen output (sampled on NSTEP cadence + convergence).
                sample_vmec = (iter2 <= 1) | (iter2 >= int(max_iter)) | ((iter2 % nstep_screen) == 0) | conv_now
                sample_vmec = sample_vmec & jnp.asarray(scan_collect_scalars, dtype=bool)

                def _compute_scalars(_):
                    r00_j = jnp.asarray(k.pr1_even)[0, 0, 0]
                    if bool(cfg.lasym):
                        z00_j = jnp.asarray(k.pz1_even)[0, 0, 0]
                    else:
                        z00_j = jnp.asarray(0.0, dtype=r00_j.dtype)
                    # `norms_current` already reflects the current bcovar state.
                    wb_val = jnp.asarray(norms_current.wb)
                    wp_val = jnp.asarray(norms_current.wp)
                    w_mhd = (wb_val + wp_val / (gamma - 1.0)) * jnp.asarray(float(TWOPI * TWOPI), dtype=wb_val.dtype)
                    return r00_j, z00_j, w_mhd

                def _reuse_scalars(_):
                    return carry_adv.r00_prev, carry_adv.z00_prev, carry_adv.w_mhd_prev

                r00_j, z00_j, w_mhd = jax.lax.cond(sample_vmec, _compute_scalars, _reuse_scalars, operand=None)

                current_payload_pre = _build_current_preconditioned_scan_payload(
                    need_bcovar_update=need_bcovar_update,
                    carry_adv=carry_adv,
                    k=k,
                    frzl=frzl,
                    norms_used=norms_used,
                    rz_scale=rz_scale,
                    l_scale=l_scale,
                    constraint_tcon0=constraint_tcon0,
                    zero_precond_diag=zero_precond_diag,
                    zero_tcon=zero_tcon,
                    trig=trig,
                    s=s,
                    cfg=cfg,
                    dtype=dtype,
                    scan_use_precomputed=bool(scan_use_precomputed),
                    scan_use_lax_tridi=bool(scan_use_lax_tridi),
                    lambda_preconditioner_func=_lambda_preconditioner,
                    rz_norm_func=_rz_norm,
                    scale_m1_precond_rhs_func=scale_m1_precond_rhs,
                    w_mode_mn=w_mode_mn,
                    lambda_update_scale_j=lambda_update_scale_j,
                    apply_lambda_update_scale=(lambda_update_scale != 1.0),
                    fsqr=fsqr,
                    fsqz=fsqz,
                    fsql=fsql,
                    delta_s=delta_s,
                    jmax0=jmax0,
                    cond=jax.lax.cond,
                )
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
                ) = current_payload_pre.blocks
                fsqr1 = current_payload_pre.fsqr1
                fsqz1 = current_payload_pre.fsqz1
                fsql1 = current_payload_pre.fsql1
                fsq1 = fsqr1 + fsqz1 + fsql1

                fsq0 = fsqr + fsqz + fsql
                use_state_jac = os.getenv("VMEC_JAX_SCAN_JAC_FROM_STATE", "0").strip().lower() not in (
                    "",
                    "0",
                    "false",
                    "no",
                )
                use_apply_payload_fusion = False
                ptau_min, ptau_max = _ptau_minmax(k) if bool(vmec2000_control) else (None, None)

                def _state_tau():
                    jac_scan = vmec_half_mesh_jacobian_from_state(
                        state=carry_adv.state,
                        modes=static.modes,
                        trig=trig,
                        s=s,
                        lconm1=bool(getattr(static.cfg, "lconm1", True)),
                        lthreed=bool(getattr(static.cfg, "lthreed", True)),
                        mask_even=getattr(static, "m_is_even", None),
                        mask_odd=getattr(static, "m_is_odd", None),
                    )
                    return jnp.asarray(jac_scan.tau)

                def _state_jacobian_decision():
                    tau_decision = _scan_math_state_jacobian(
                        _state_tau(),
                        vmec2000_control=bool(vmec2000_control),
                        ptau_tol=ptau_tol,
                        relative_tol=1.0e-2 if bool(vmec2000_control) else None,
                    )
                    return tau_decision.bad_jacobian, tau_decision.min_tau, tau_decision.max_tau

                def _nonvmec_tau():
                    if use_state_jac:
                        return _state_tau()
                    return jnp.asarray(k.bc.jac.tau)

                tau_decision = _scan_bad_jacobian_decision(
                    vmec2000_control=bool(vmec2000_control),
                    use_apply_payload_fusion=bool(use_apply_payload_fusion),
                    badjac_use_state=bool(badjac_use_state),
                    dump_ptau_state=bool(dump_ptau_state),
                    badjac_state_probe=bool(badjac_state_probe),
                    badjac_initial_state_probe_iters=int(badjac_initial_state_probe_iters),
                    iter2=iter2,
                    ptau_min=ptau_min,
                    ptau_max=ptau_max,
                    state_tau_fn=_state_jacobian_decision,
                    nonvmec_tau_fn=_nonvmec_tau,
                    ptau_tol=ptau_tol,
                    dtype=dtype,
                    cond=jax.lax.cond,
                )
                bad_jacobian = tau_decision.bad_jacobian
                min_tau = tau_decision.min_tau
                max_tau = tau_decision.max_tau
                min_tau_ptau = tau_decision.min_tau_ptau
                max_tau_ptau = tau_decision.max_tau_ptau
                min_tau_state = tau_decision.min_tau_state
                max_tau_state = tau_decision.max_tau_state
                badjac_ptau = tau_decision.badjac_ptau
                badjac_state = tau_decision.badjac_state
                if os.getenv("VMEC_JAX_SCAN_IGNORE_BADJAC", "") not in ("", "0"):
                    bad_jacobian = jnp.asarray(False)
                # Axis reset handled before entering the scan loop.

                time_restart = evaluate_scan_time_control_restart(
                    carry_adv=carry_adv,
                    iter2=iter2,
                    fsqr=fsqr,
                    fsqz=fsqz,
                    fsql=fsql,
                    fsqr1=fsqr1,
                    fsqz1=fsqz1,
                    fsql1=fsql1,
                    fsq0=fsq0,
                    fsq1=fsq1,
                    fsq_prev_before=fsq_prev_before,
                    fsq0_prev_before=fsq0_prev_before,
                    bad_jacobian=bad_jacobian,
                    skip_timecontrol=skip_timecontrol,
                    vmec2000_control=bool(vmec2000_control),
                    reference_mode=bool(reference_mode),
                    use_apply_payload_fusion=bool(use_apply_payload_fusion),
                    dump_timecontrol_scan=bool(dump_timecontrol_scan),
                    scan_timecontrol_dumper=scan_timecontrol_dumper,
                    vmec2000_fact=vmec2000_fact,
                    use_restart_triggers=bool(use_restart_triggers),
                    vmecpp_restart=bool(vmecpp_restart),
                    k_preconditioner_update_interval=k_preconditioner_update_interval,
                    stage_prev_fsq=stage_prev_fsq_j,
                    stage_transition_factor=stage_transition_factor,
                    restart_badjac_factor=restart_badjac_factor,
                    restart_badprog_factor=restart_badprog_factor,
                    stage_transition_scale=stage_transition_scale,
                    step_size=step_size,
                    k_ndamp=k_ndamp,
                    dtype=dtype,
                    restart_updates_func=_scan_math_restart_updates,
                    no_restart_updates_func=_scan_math_no_restart_updates,
                    scan_restart_transition_func=scan_restart_transition,
                    cond_func=jax.lax.cond,
                )
                fsq_phys = time_restart.fsq_phys
                res0 = time_restart.res0
                res1 = time_restart.res1
                checkpoint_update = time_restart.checkpoint_update
                restart_decision = time_restart.restart_decision
                do_restart = restart_decision.do_restart
                restart_update = time_restart.restart_update
                state_post = restart_update.state
                time_step_post = restart_update.time_step
                inv_tau_post = restart_update.inv_tau
                fsq_prev_post = restart_update.fsq_prev
                (
                    vRcc_post,
                    vRss_post,
                    vZsc_post,
                    vZcs_post,
                    vLsc_post,
                    vLcs_post,
                    vRsc_post,
                    vRcs_post,
                    vZcc_post,
                    vZss_post,
                    vLcc_post,
                    vLss_post,
                ) = restart_update.velocity_blocks
                iter_offset_post = restart_update.iter_offset
                iter1_post = restart_update.iter1
                ijacob_post = restart_update.ijacob
                bad_resets_post = restart_update.bad_resets
                bad_growth_post = restart_update.bad_growth
                force_bcovar_post = restart_update.force_bcovar_update

                fsq0_prev_post = time_restart.fsq0_prev_post

                payload_step = _select_payload_and_build_step_fields(
                    do_restart=do_restart,
                    use_restart_payload=bool(scan_use_restart_payload),
                    current_payload=current_payload_pre,
                    state_post=state_post,
                    compute_forces_scan_func=_compute_forces_scan,
                    restart_trace_context=lambda: _maybe_trace("scan/compute_forces:restart"),
                    zero_m1=zero_m1,
                    zero_precond_diag=zero_precond_diag,
                    zero_tcon=zero_tcon,
                    constraint_active_false=constraint_active_false,
                    constraint_tcon0=constraint_tcon0,
                    trig=trig,
                    s=s,
                    cfg=cfg,
                    dtype=dtype,
                    scan_use_precomputed=bool(scan_use_precomputed),
                    scan_use_lax_tridi=bool(scan_use_lax_tridi),
                    lambda_preconditioner_func=_lambda_preconditioner,
                    rz_norm_func=_rz_norm,
                    scale_m1_precond_rhs_func=scale_m1_precond_rhs,
                    w_mode_mn=w_mode_mn,
                    lambda_update_scale_j=lambda_update_scale_j,
                    apply_lambda_update_scale=(lambda_update_scale != 1.0),
                    delta_s=delta_s,
                    jmax0=jmax0,
                    velocity_blocks_post=(
                        vRcc_post,
                        vRss_post,
                        vZsc_post,
                        vZcs_post,
                        vLsc_post,
                        vLcs_post,
                        vRsc_post,
                        vRcs_post,
                        vZcc_post,
                        vZss_post,
                        vLcc_post,
                        vLss_post,
                    ),
                    inv_tau_post=inv_tau_post,
                    fsq_prev_post=fsq_prev_post,
                    time_step_post=time_step_post,
                    iter2=iter2,
                    iter1_post=iter1_post,
                    k_ndamp=k_ndamp,
                    flip_sign=flip_sign0,
                    lasym=bool(cfg.lasym),
                    static=static,
                    edge_Rcos=carry.edge_Rcos,
                    edge_Rsin=carry.edge_Rsin,
                    edge_Zcos=carry.edge_Zcos,
                    edge_Zsin=carry.edge_Zsin,
                    free_boundary_enabled=bool(free_boundary_enabled),
                    idx00=idx00,
                    mn_cos_to_signed_physical=_mn_cos_to_signed_physical,
                    mn_sin_to_signed_physical=_mn_sin_to_signed_physical,
                    mn_sin_to_signed_physical_lambda=_mn_sin_to_signed_physical_lambda,
                    mn_cos_to_signed_physical_lambda=_mn_cos_to_signed_physical_lambda,
                    enforce_fixed_boundary_and_axis=_enforce_fixed_boundary_and_axis,
                    apply_vmec_lambda_axis_rules=_apply_vmec_lambda_axis_rules,
                    vmec2000_control=bool(vmec2000_control),
                    cond=jax.lax.cond,
                )
                payload_use = payload_step.payload
                step_fields = payload_step.step_fields
                fsqr = payload_use.fsqr
                fsqz = payload_use.fsqz
                fsql = payload_use.fsql
                fsq1 = payload_step.fsq1
                # VMEC prints the updated time-step (post TimeStepControl/restart),
                # so report the post-update value on this iteration.
                time_step_report = time_step_post
                _ = _emit_live_scan_vmec2000_row(
                    enabled=print_in_scan,
                    sample_vmec=sample_vmec,
                    iter_idx=iter2,
                    fsqr=fsqr,
                    fsqz=fsqz,
                    fsql=fsql,
                    delt0r=time_step_report,
                    r00=r00_j,
                    w_mhd=w_mhd,
                    scan_print_mode=scan_print_mode,
                    scan_print_ordered=bool(scan_print_ordered),
                    jax_debug=scan_runtime.jax_debug,
                    io_callback=scan_runtime.io_callback,
                    cond=jax.lax.cond,
                    print_row=_print_scan_vmec2000_row,
                )
                step_result = finalize_vmec2000_scan_step(
                    carry_adv=carry_adv,
                    step_fields=step_fields,
                    current_payload=current_payload_pre,
                    selected_payload=payload_use,
                    checkpoint_update=checkpoint_update,
                    scan_fallback_enabled_run=scan_fallback_enabled_run,
                    scan_core=bool(scan_core),
                    fsq_phys=fsq_phys,
                    fsq1=fsq1,
                    bad_jacobian=bad_jacobian,
                    abort_scan_on_badjac=abort_scan_on_badjac,
                    scan_fallback_iters=scan_fallback_iters_j,
                    scan_fallback_badjac_limit=scan_fallback_badjac_limit_j,
                    scan_fallback_accept_frac=scan_fallback_accept_frac_j,
                    scan_fallback_fsq_factor=scan_fallback_fsq_factor_j,
                    scan_fallback_fsq_abs=scan_fallback_fsq_abs_j,
                    scan_fallback_improve=scan_fallback_improve_j,
                    dtype=dtype,
                    vmec2000_control=bool(vmec2000_control),
                    do_restart=do_restart,
                    state_only_scan=bool(state_only_scan),
                    scan_minimal=bool(scan_minimal),
                    scan_light=bool(scan_light),
                    fsq0_prev_post=fsq0_prev_post,
                    force_bcovar_post=force_bcovar_post,
                    flip_sign=flip_sign0,
                    iter_offset_post=iter_offset_post,
                    iter1_post=iter1_post,
                    res0=res0,
                    res1=res1,
                    ijacob_post=ijacob_post,
                    bad_resets_post=bad_resets_post,
                    bad_growth_post=bad_growth_post,
                    r00=r00_j,
                    z00=z00_j,
                    w_mhd=w_mhd,
                    conv_now=conv_now,
                    time_step_report=time_step_report,
                    zero_m1=zero_m1,
                    include_edge=include_edge,
                    min_tau=min_tau,
                    max_tau=max_tau,
                    min_tau_ptau=min_tau_ptau,
                    max_tau_ptau=max_tau_ptau,
                    min_tau_state=min_tau_state,
                    max_tau_state=max_tau_state,
                    badjac_ptau=badjac_ptau,
                    badjac_state=badjac_state,
                )
                return step_result.carry, step_result.history_row

            iter2_hold = jnp.asarray(it + 1, dtype=jnp.int32) + jnp.asarray(carry.iter_offset, dtype=jnp.int32)
            hold_cond = carry.converged | carry.abort_scan | (iter2_hold > jnp.asarray(int(max_iter), dtype=jnp.int32))
            return jax.lax.cond(hold_cond, _hold_step, _advance_step, operand=carry)

        carry0 = _build_initial_scan_carry(
            state_init=state_init,
            resume_fields=scan_resume0,
            dtype=dtype,
            iter_offset0=iter_offset0,
            cache_valid=cache_valid0,
            cache_precond_diag=cache_precond_diag0,
            cache_tcon=cache_tcon0,
            cache_norms=cache_norms0,
            cache_rz_scale=cache_rz_scale0,
            cache_l_scale=cache_l_scale0,
            cache_rz_norm=cache_rz_norm0,
            cache_f_norm1=cache_f_norm1_0,
            cache_rz_mats=cache_rz_mats0,
            cache_lam_prec=cache_lam_prec0,
            edge_Rcos=edge_Rcos,
            edge_Rsin=edge_Rsin,
            edge_Zcos=edge_Zcos,
            edge_Zsin=edge_Zsin,
        )

        scan_runtime_plan = _resolve_scan_iteration_runtime_plan(
            env=os.environ,
            jit_forces_scan=bool(jit_forces_scan),
            vmec2000_control=bool(vmec2000_control),
            max_iter=int(max_iter),
            axis_reset_repeat=bool(axis_reset_repeat),
            iter_offset0=int(iter_offset0),
            static_key=static_key,
            wout_key=wout_key,
            edge_signature_key=edge_signature_key,
            step_size=float(step_size),
            initial_flip_sign=float(initial_flip_sign),
            lambda_update_scale=float(lambda_update_scale),
            ftol=float(ftol),
            nstep_screen=int(nstep_screen),
            use_restart_triggers=bool(use_restart_triggers),
            vmecpp_restart=bool(vmecpp_restart),
            scan_use_precomputed=bool(scan_use_precomputed),
            scan_use_lax_tridi=bool(scan_use_lax_tridi),
            scan_use_restart_payload=bool(scan_use_restart_payload),
            stage_prev_fsq=stage_prev_fsq,
            stage_transition_factor=float(stage_transition_factor),
            stage_transition_scale=float(stage_transition_scale),
            state_only_scan=bool(state_only_scan),
            scan_light=bool(scan_light),
            scan_minimal=bool(scan_minimal),
            scan_fallback_iters=int(scan_fallback_iters),
            scan_fallback_accept_frac=float(scan_fallback_accept_frac),
            scan_fallback_fsq_factor=float(scan_fallback_fsq_factor),
            scan_fallback_badjac_limit=int(scan_fallback_badjac_limit),
            scan_fallback_fsq_abs=float(scan_fallback_fsq_abs),
        )
        preflight_iters = scan_runtime_plan.preflight_iters
        max_iter_scan = scan_runtime_plan.max_iter_scan
        max_iter_tail = scan_runtime_plan.max_iter_tail
        iter_offset_preflight = scan_runtime_plan.iter_offset_preflight
        iter_offset0 = scan_runtime_plan.iter_offset0
        if scan_runtime_plan.axis_reset_repeated:
            carry0 = carry0._replace(iter_offset=jnp.asarray(iter_offset0, dtype=jnp.int32))

        scan_cache_key = scan_runtime_plan.scan_cache_key

        def _run_scan(carry_init, it_seq):
            return jax.lax.scan(_scan_step, carry_init, it_seq)

        def _get_scan_runner(seq_len: int):
            key = scan_cache_key + (int(seq_len),)
            return _get_or_build_scan_runner(
                _run_scan,
                cache=_SCAN_RUNNER_CACHE,
                key=key,
                differentiating_scan=bool(scan_differentiated),
                scan_timing_enabled=bool(scan_timing_enabled),
                scan_timing_stats=scan_timing_stats,
                jit_func=jit,
                cache_get=_jit_cache_get,
                cache_put=_jit_cache_put,
                record_miss_categories=_record_scan_runner_cache_miss_categories,
                perf_counter=time.perf_counter,
            )

        def _emit_scan_prints(
            *,
            hist_np,
            it_start: int,
            max_iter_local: int,
        ) -> bool:
            return _emit_scan_debug_prints(
                hist_np=hist_np,
                it_start=it_start,
                max_iter_local=max_iter_local,
                scan_minimal=bool(scan_minimal),
                scan_light=bool(scan_light),
                ftol=float(ftol),
                fsq_total_target=fsq_total_target,
                iter_offset0=int(iter_offset0),
                should_print=scan_print_context.should_print,
                print_row=scan_print_context.print_row,
            )

        if scan_timing_enabled and scan_run_setup_start is not None:
            scan_timing_stats["scan_run_setup_s"] += time.perf_counter() - float(scan_run_setup_start)
        carry_init = carry0._replace(state=state_init, state_checkpoint=state_init)
        if chunked_print:
            need_print = bool(scan_collect_print)
            chunked_result = _run_chunked_scan(
                carry_init,
                max_iter=int(max_iter),
                max_iter_scan=int(max_iter_scan),
                nstep_screen=int(nstep_screen),
                need_print=bool(need_print),
                lthreed=bool(cfg.lthreed),
                spectral_mode_count=int(ncoeff),
                scan_chunk_settings_func=_scan_chunk_settings,
                scan_jit_preflight_enabled_func=_scan_jit_preflight_enabled,
                scan_jit_preflight_env=os.getenv("VMEC_JAX_SCAN_JIT_PREFLIGHT"),
                backend_name=_scan_backend_name(),
                scan_differentiated=bool(scan_differentiated),
                preflight_iters=int(preflight_iters),
                iter_offset_preflight=int(iter_offset_preflight),
                axis_reset_repeat=bool(axis_reset_repeat),
                iter_offset0=int(iter_offset0),
                get_scan_runner=_get_scan_runner,
                scan_step=_scan_step,
                scan_timing_enabled=bool(scan_timing_enabled),
                scan_timing_stats=scan_timing_stats,
                scan_device_runtime=scan_device_runtime,
                perf_counter=time.perf_counter,
                state_only_scan=bool(state_only_scan),
                scan_fallback_enabled_run=bool(scan_fallback_enabled_run),
                scan_fallback_iters=int(scan_fallback_iters),
                scan_fallback_fsq_abs=float(scan_fallback_fsq_abs),
                dtype=dtype,
                emit_scan_prints=_emit_scan_prints,
                tree_has_tracer=_tree_has_tracer,
                jnp_module=jnp,
                jax_module=jax,
                np_module=np,
            )
            carry_final = chunked_result.carry_final
            hist = chunked_result.history
        else:
            nonchunked_result = _run_nonchunked_scan(
                carry_init,
                max_iter_scan=int(max_iter_scan),
                max_iter_tail=int(max_iter_tail),
                preflight_iters=int(preflight_iters),
                iter_offset_preflight=int(iter_offset_preflight),
                axis_reset_repeat=bool(axis_reset_repeat),
                iter_offset0=int(iter_offset0),
                get_scan_runner=_get_scan_runner,
                scan_step=_scan_step,
                scan_jit_preflight_enabled_func=_scan_jit_preflight_enabled,
                scan_jit_preflight_env=os.getenv("VMEC_JAX_SCAN_JIT_PREFLIGHT"),
                backend_name=_scan_backend_name(),
                scan_differentiated=bool(scan_differentiated),
                scan_collect_print=bool(scan_collect_print),
                scan_timing_enabled=bool(scan_timing_enabled),
                scan_timing_stats=scan_timing_stats,
                scan_device_runtime=scan_device_runtime,
                perf_counter=time.perf_counter,
                state_only_scan=bool(state_only_scan),
                scan_fallback_enabled_run=bool(scan_fallback_enabled_run),
                scan_fallback_iters=int(scan_fallback_iters),
                jnp_module=jnp,
                jax_module=jax,
            )
            carry_final = nonchunked_result.carry_final
            hist = nonchunked_result.history
        return finalize_vmec2000_scan_run(
            carry_final=carry_final,
            history=hist,
            state0=state0,
            result_type=SolveVmecResidualResult,
            state_only_scan=bool(state_only_scan),
            scan_minimal=bool(scan_minimal),
            scan_light=bool(scan_light),
            scan_use_precomputed=bool(scan_use_precomputed),
            scan_use_lax_tridi=bool(scan_use_lax_tridi),
            vmec2000_control=bool(vmec2000_control),
            ftol=float(ftol),
            fsq_total_target=fsq_total_target,
            max_iter=int(max_iter),
            resume_state_mode=str(resume_state_mode),
            pack_resume_state=_pack_resume_state,
            free_boundary_enabled=bool(free_boundary_enabled),
            freeb_nvacskip=int(freeb_nvacskip),
            freeb_nvskip0=int(freeb_nvskip0),
            iter_offset0=int(iter_offset0),
            free_boundary_iter_controls=_free_boundary_iter_controls,
            scan_timing_enabled=bool(scan_timing_enabled),
            scan_timing_stats=scan_timing_stats,
            scan_postprocess_start=time.perf_counter() if scan_timing_enabled else None,
            scan_total_start=scan_total_start,
            perf_counter=time.perf_counter,
            build_timing_report=_build_scan_timing_report,
            tree_has_tracer=_tree_has_tracer,
            build_traced_scan_resume_state=_build_traced_scan_resume_state,
            state_only_scan_result=_vmec2000_state_only_scan_result,
            traced_scan_result=_vmec2000_traced_scan_result,
            attach_free_boundary_diagnostics=_attach_freeb_diag,
            emit_post_scan_rows=_emit_vmec2000_post_scan_rows,
            post_scan_print_enabled=(
                (not bool(scan_minimal))
                and (not bool(print_in_scan))
                and (not bool(chunked_print))
                and bool(verbose)
                and bool(vmec2000_control)
                and bool(verbose_vmec2000_table)
            ),
            should_print=scan_print_context.should_print,
            print_row=scan_print_context.print_row,
            dump_ptau_rows=_dump_vmec2000_scan_ptau_rows,
            dump_ptau_enabled=(
                (not bool(scan_light))
                and (not bool(scan_minimal))
                and os.getenv("VMEC_JAX_DUMP_PTAU", "") not in ("", "0")
            ),
            badjac_mode=badjac_mode,
            dump_ptau=_maybe_dump_ptau,
            badjac_use_state=bool(badjac_use_state),
            badjac_state_probe=bool(badjac_state_probe),
            badjac_initial_state_probe_iters=int(badjac_initial_state_probe_iters),
        )

    if use_scan:
        if vmec2000_control:
            scan_result = _run_vmec2000_scan(state)
            if scan_fallback_enabled and (not bool(state_only)):
                fallback_decision = _scan_fallback_decision(
                    diagnostics=scan_result.diagnostics,
                    fsqr_history=scan_result.fsqr2_history,
                    fsqz_history=scan_result.fsqz2_history,
                    fsql_history=scan_result.fsql2_history,
                    max_iter=int(max_iter),
                    fallback_iters=int(scan_fallback_iters),
                    badjac_limit=int(scan_fallback_badjac_limit),
                    fsq_abs=float(scan_fallback_fsq_abs),
                    accept_frac=float(scan_fallback_accept_frac),
                    fsq_factor=float(scan_fallback_fsq_factor),
                )
                if fallback_decision.fallback:
                    if verbose:
                        print(_scan_fallback_message(fallback_decision), flush=True)
                    use_scan = False
                    resume_state = None
                    state = state0
                else:
                    return _attach_freeb_diag(scan_result)
            else:
                return _attach_freeb_diag(scan_result)

        if use_scan:
            if (
                backtracking
                or use_restart_triggers
                or auto_flip_force
                or limit_dt_from_force
                or limit_update_rms
                or strict_update
                or use_direct_fallback
                or reference_mode
            ):
                raise ValueError(
                    "use_scan requires vmec2000_control=False, backtracking=False, "
                    "use_restart_triggers=False, auto_flip_force=False, "
                    "limit_dt_from_force=False, limit_update_rms=False, strict_update=False, "
                    "use_direct_fallback=False, reference_mode=False."
                )

            return _run_accelerated_residual_scan(
                state=state,
                state0=state0,
                static=static,
                cfg=cfg,
                max_iter=int(max_iter),
                step_size=float(step_size),
                initial_flip_sign=float(initial_flip_sign),
                lambda_update_scale=float(lambda_update_scale),
                lambda_update_scale_j=lambda_update_scale_j,
                ftol=float(ftol),
                fsq_total_target=fsq_total_target,
                precond_radial_alpha=float(precond_radial_alpha),
                precond_lambda_alpha=float(precond_lambda_alpha),
                apply_m1_constraints=bool(apply_m1_constraints),
                jit_forces=bool(jit_forces),
                free_boundary_enabled=bool(free_boundary_enabled),
                static_key=static_key,
                wout_key=wout_key,
                edge_value_key=edge_value_key,
                edge_Rcos=edge_Rcos,
                edge_Rsin=edge_Rsin,
                edge_Zcos=edge_Zcos,
                edge_Zsin=edge_Zsin,
                idx00=int(idx00),
                w_mode_mn=w_mode_mn,
                mode_context=_mode_context,
                compute_forces=_compute_forces,
                compute_forces_impl=_compute_forces_impl,
                apply_radial_tridi_batched=_apply_radial_tridi_batched,
                mn_cos_to_signed_physical=_mn_cos_to_signed_physical,
                mn_sin_to_signed_physical=_mn_sin_to_signed_physical,
                mn_cos_to_signed_physical_lambda=_mn_cos_to_signed_physical_lambda,
                enforce_fixed_boundary_and_axis=_enforce_fixed_boundary_and_axis,
                apply_vmec_lambda_axis_rules=_apply_vmec_lambda_axis_rules,
                attach_freeb_diag=_attach_freeb_diag,
                scan_timing_env=os.getenv("VMEC_JAX_TIMING", ""),
                jax_module=jax,
                jnp_module=jnp,
                jit_func=jit,
                scan_timing_enabled_func=_scan_timing_enabled,
                new_scan_timing_stats_func=_new_scan_timing_stats,
                build_scan_timing_report_func=_build_scan_timing_report,
                scan_device_runtime_type=ScanDeviceRuntime,
                scan_convergence_predicate_type=ScanConvergencePredicate,
                converged_residuals_func=_runtime_converged_residuals_scan_fast,
                scan_device_ready_recorder=_record_scan_device_ready,
                get_or_build_scan_runner_func=_get_or_build_scan_runner,
                scan_runner_cache=_SCAN_RUNNER_CACHE,
                jit_cache_get_func=_jit_cache_get,
                jit_cache_put_func=_jit_cache_put,
                record_scan_runner_cache_miss_categories_func=_record_scan_runner_cache_miss_categories,
                perf_counter=time.perf_counter,
                differentiating_scan=bool(differentiating_scan),
            )

    profile_window = os.getenv("VMEC_JAX_PROFILE_WINDOW", "").strip().lower()
    profile_dir_env = os.getenv("VMEC_JAX_PROFILE_DIR", "").strip()
    profile_started = False
    profile_active = False
    profile_start_iter = None
    profile_dir = ""
    if profile_window and profile_dir_env:
        if profile_window in ("pre", "iter1", "1"):
            profile_start_iter = 1
        else:
            window_str = profile_window
            if window_str.startswith("iter"):
                window_str = window_str[4:]
            try:
                profile_start_iter = max(1, int(window_str))
            except Exception:
                profile_start_iter = None
        if profile_start_iter is not None:
            profile_dir = str(Path(profile_dir_env) / f"window_{profile_window}")
            profile_active = True
    perfetto_env = os.getenv("VMEC_JAX_PROFILE_PERFETTO", "1")
    profile_perfetto = perfetto_env.strip().lower() not in ("", "0", "false", "no")

    timing_stats = _runtime_new_residual_iter_timing_stats(_setup_phase_timings)

    _record_compute_force_timing = partial(
        _runtime_record_compute_force_timing,
        timing_enabled=bool(timing_enabled),
        timing_stats=timing_stats,
        perf_counter=time.perf_counter,
        block_until_ready=jax.block_until_ready if has_jax() else None,
    )

    w_history = []
    fsqr2_history = []
    fsqz2_history = []
    fsql2_history = []
    r00_history: list[float] = []
    z00_history: list[float] = []
    wb_history: list[float] = []
    wp_history: list[float] = []
    w_vmec_history: list[float] = []
    fsqr1_history = []
    fsqz1_history = []
    fsql1_history = []
    fsq1_history = []
    rz_norm_history: list[float] = []
    f_norm1_history: list[float] = []
    gcr2_p_history: list[float] = []
    gcz2_p_history: list[float] = []
    gcl2_p_history: list[float] = []
    step_status_history: list[str] = []
    restart_reason_history: list[str] = []
    pre_restart_reason_history: list[str] = []
    time_step_history: list[float] = []
    res0_history: list[float] = []
    res1_history: list[float] = []
    fsq_prev_history: list[float] = []
    bad_growth_streak_history: list[int] = []
    iter1_history: list[int] = []
    iter2_history: list[int] = []
    include_edge_history: list[int] = []
    zero_m1_history: list[int] = []
    freeb_ivac_history: list[int] = []
    freeb_ivacskip_history: list[int] = []
    freeb_full_update_history: list[int] = []
    freeb_nestor_reused_history: list[int] = []
    freeb_nestor_source_reused_history: list[int] = []
    freeb_nestor_provider_allows_source_reuse_history: list[int] = []
    freeb_nestor_bnormal_rms_history: list[float] = []
    freeb_nestor_gsource_rms_history: list[float] = []
    freeb_nestor_bsqvac_rms_history: list[float] = []
    freeb_nestor_solve_time_history: list[float] = []
    freeb_nestor_sample_time_history: list[float] = []
    freeb_nestor_trial_reused_history: list[int] = []
    freeb_nestor_trial_solve_time_history: list[float] = []
    freeb_nestor_trial_sample_time_history: list[float] = []
    freeb_nestor_trial_failed_history: list[int] = []
    dt_eff_history: list[float] = []
    update_rms_history: list[float] = []
    w_curr_history: list[float] = []
    w_try_history: list[float] = []
    w_try_ratio_history: list[float] = []
    restart_path_history: list[str] = []
    adjoint_step_trace_history: list[dict[str, Any]] = []
    min_tau_history: list[float] = []
    max_tau_history: list[float] = []
    bad_jacobian_history: list[int] = []
    grad_rms_history = []
    step_history = []

    _history_record_lists = {
        "step_history": step_history,
        "dt_eff_history": dt_eff_history,
        "update_rms_history": update_rms_history,
        "w_curr_history": w_curr_history,
        "w_try_history": w_try_history,
        "w_try_ratio_history": w_try_ratio_history,
        "restart_path_history": restart_path_history,
        "step_status_history": step_status_history,
        "restart_reason_history": restart_reason_history,
        "pre_restart_reason_history": pre_restart_reason_history,
        "time_step_history": time_step_history,
        "res0_history": res0_history,
        "res1_history": res1_history,
        "fsq_prev_history": fsq_prev_history,
        "bad_growth_streak_history": bad_growth_streak_history,
        "iter1_history": iter1_history,
        "iter2_history": iter2_history,
        "grad_rms_history": grad_rms_history,
        "free_boundary_enabled": free_boundary_enabled,
        "freeb_ivac_history": freeb_ivac_history,
        "freeb_ivacskip_history": freeb_ivacskip_history,
        "freeb_full_update_history": freeb_full_update_history,
    }
    _terminal_history_lists = {
        "step_status_history": step_status_history,
        "restart_reason_history": restart_reason_history,
        "pre_restart_reason_history": pre_restart_reason_history,
        "time_step_history": time_step_history,
        "res0_history": res0_history,
        "res1_history": res1_history,
        "fsq_prev_history": fsq_prev_history,
        "bad_growth_streak_history": bad_growth_streak_history,
        "iter1_history": iter1_history,
        "iter2_history": iter2_history,
        "grad_rms_history": grad_rms_history,
        "free_boundary_enabled": free_boundary_enabled,
        "freeb_ivac_history": freeb_ivac_history,
        "freeb_ivacskip_history": freeb_ivacskip_history,
        "freeb_full_update_history": freeb_full_update_history,
        "freeb_nestor_reused_history": freeb_nestor_reused_history,
        "freeb_nestor_solve_time_history": freeb_nestor_solve_time_history,
        "freeb_nestor_sample_time_history": freeb_nestor_sample_time_history,
    }

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
    velocity_shape = (int(state.Rcos.shape[0]), mpol, nrange)
    if bool(host_update_assembly) and (not _tree_has_tracer(state.Rcos)):
        vRcc = np.zeros(velocity_shape, dtype=np.asarray(state.Rcos).dtype)
    else:
        vRcc = jnp.zeros(velocity_shape, dtype=jnp.asarray(state.Rcos).dtype)
    (
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
    ) = _zero_velocity_blocks_like(vRcc, vRcc, vRcc, vRcc, vRcc, vRcc, vRcc, vRcc, vRcc, vRcc, vRcc)
    flip_sign = float(initial_flip_sign)
    max_coeff_delta_rms = 1e-5
    max_update_rms = 5e-3
    if bool(reference_mode):
        max_coeff_delta_rms = 5e-6
        max_update_rms = 1e-3
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

    def _vmec_freeb_plascur_from_bcovar(bc_obj, fallback: float) -> float:
        """VMEC `ctor` proxy used by NESTOR (`vacuum_par(..., ctor, ...)`)."""
        try:
            from vmec_jax.vmec_lforbal import plascur_edge_from_bcovar
            return _runtime_vmec_freeb_plascur_from_bcovar(
                bc_obj,
                fallback=fallback,
                plascur_edge_from_bcovar=plascur_edge_from_bcovar,
                trig=trig,
                wout=wout_like,
                s=s,
            )
        except Exception:
            return float(fallback)
        return float(fallback)

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

    bcovar_update_history: list[int] = []
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

    def _apply_vmec_scale_m1_precond_rhs(frzl_in: TomnspsRZL, mats: dict[str, Any]) -> TomnspsRZL:
        return _scale_m1_precond_rhs_from_mats(
            frzl_in,
            mats,
            lconm1=getattr(cfg, "lconm1", True),
            mpol=int(cfg.mpol),
            host_update_assembly=host_update_assembly,
        )

    def _pop_iteration_histories() -> None:
        def _pop(hist):
            if hist:
                hist.pop()

        for h in (
            include_edge_history,
            zero_m1_history,
            bcovar_update_history,
            w_history,
            fsqr2_history,
            fsqz2_history,
            fsql2_history,
            r00_history,
            z00_history,
            wb_history,
            wp_history,
            w_vmec_history,
            rz_norm_history,
            f_norm1_history,
            gcr2_p_history,
            gcz2_p_history,
            gcl2_p_history,
            fsq1_history,
            fsqr1_history,
            fsqz1_history,
            fsql1_history,
            min_tau_history,
            max_tau_history,
            bad_jacobian_history,
            step_history,
            dt_eff_history,
            update_rms_history,
            w_curr_history,
            w_try_history,
            w_try_ratio_history,
            restart_path_history,
            step_status_history,
            restart_reason_history,
            pre_restart_reason_history,
            time_step_history,
            res0_history,
            res1_history,
            fsq_prev_history,
            bad_growth_streak_history,
            iter1_history,
            iter2_history,
            freeb_ivac_history,
            freeb_ivacskip_history,
            freeb_full_update_history,
            grad_rms_history,
        ):
            _pop(h)

    _maybe_dump_time_control = _maybe_dump_time_control_record
    _dump_time_control_trace = _dump_time_control_trace_record
    _maybe_dump_checkpoint = _maybe_dump_checkpoint_record
    _dump_freeb_control_trace = _dump_freeb_control_trace_record
    _dump_freeb_axis_trace = _dump_freeb_axis_trace_record
    _dump_evolve_trace = partial(_maybe_dump_evolve_trace_record, static=static)

    # VMEC `eqsolve`: if the initial Jacobian changes sign, improve the axis
    # guess *before* the first iteration (no extra iter1). This aligns the
    # zero_m1 gating and time-control history with VMEC2000.
    t_setup_axis_reset_start = time.perf_counter() if timing_enabled else None
    if bool(vmec2000_control) and (not axis_reset_done) and bool(lmove_axis):
        try:
            t_setup_axis_force_start = time.perf_counter() if timing_enabled else None
            k0, _frzl0, _gcr2_0, _gcz2_0, _gcl2_0, _rz_scale0, _l_scale0, _norms0 = _compute_forces_iter(
                state,
                include_edge=False,
                zero_m1=jnp.asarray(1.0, dtype=jnp.asarray(state.Rcos).dtype),
                constraint_precond_diag=zero_precond_diag,
                constraint_tcon=zero_tcon,
                constraint_precond_active=jnp.asarray(False),
                constraint_tcon_active=jnp.asarray(False),
                iter_idx=None,
                iter2=1,
            )
            if timing_enabled and t_setup_axis_force_start is not None:
                try:
                    if has_jax():
                        jax.block_until_ready((_gcr2_0, _gcz2_0, _gcl2_0))
                except Exception:
                    pass
                timing_stats["setup_axis_reset_compute_forces"] += time.perf_counter() - float(t_setup_axis_force_start)
            ptau_min0, ptau_max0 = _ptau_minmax_from_k_host(k0)
            bad_jacobian_ptau = _axis_reset_bad_jacobian_ptau_from_minmax(
                ptau_min=ptau_min0,
                ptau_max=ptau_max0,
                ptau_tol=0.0,
                ptau_tol_rel=0.0,
            )

            bad_jacobian_state = False
            min_tau_state = float("nan")
            max_tau_state = float("nan")
            if badjac_use_state or (bad_jacobian_ptau is None):
                jac0 = vmec_half_mesh_jacobian_from_state(
                    state=state,
                    modes=static.modes,
                    trig=trig,
                    s=s,
                    lconm1=bool(getattr(static.cfg, "lconm1", True)),
                    lthreed=bool(getattr(static.cfg, "lthreed", True)),
                    mask_even=getattr(static, "m_is_even", None),
                    mask_odd=getattr(static, "m_is_odd", None),
                )
                tau0 = jnp.asarray(jac0.tau)
                tau0_use = tau0[1:] if int(tau0.shape[0]) > 1 else tau0
                min_tau_state = float(np.asarray(jnp.min(tau0_use)))
                max_tau_state = float(np.asarray(jnp.max(tau0_use)))
                bad_jacobian_state = _axis_reset_bad_jacobian_from_tau_range(
                    min_tau=min_tau_state,
                    max_tau=max_tau_state,
                )

            axis_reset_debug = os.getenv("VMEC_JAX_AXIS_RESET_DEBUG", "").strip().lower() not in (
                "",
                "0",
                "false",
                "no",
            )
            fsq_phys0_val = _axis_reset_initial_force_physical_fsq(
                norms=_norms0,
                gcr2=_gcr2_0,
                gcz2=_gcz2_0,
                gcl2=_gcl2_0,
            )

            axis_reset_decision = _initial_axis_reset_decision(
                bad_jacobian_ptau=bad_jacobian_ptau,
                bad_jacobian_state=bad_jacobian_state,
                badjac_use_state=badjac_use_state,
                fsq_phys=fsq_phys0_val,
                axis_reset_fsq_min=axis_reset_fsq_min,
                force_axis_reset=force_axis_reset,
                axis_reset_always_3d=axis_reset_always_3d,
                lthreed=bool(getattr(cfg, "lthreed", True)),
            )
            bad_jacobian0 = axis_reset_decision.bad_jacobian
            if axis_reset_debug:
                try:
                    fsq_debug_val = float("nan") if fsq_phys0_val is None else float(fsq_phys0_val)
                    print(
                        "[axis_reset] fsq0="
                        f"{fsq_debug_val:.6e} "
                        f"axis_reset_fsq_min={axis_reset_fsq_min:.3e} "
                        f"badjac_ptau={bad_jacobian_ptau} badjac_state={bad_jacobian_state} "
                        f"badjac_used={bad_jacobian0}",
                        flush=True,
                    )
                except Exception:
                    pass

            force_axis_reset_init = axis_reset_decision.force_reset
            if axis_reset_decision.reset:
                if verbose and bool(vmec2000_control) and bool(verbose_vmec2000_table):
                    if bad_jacobian0 or force_axis_reset_init:
                        print(" INITIAL JACOBIAN CHANGED SIGN!", flush=True)
                    print(" TRYING TO IMPROVE INITIAL MAGNETIC AXIS GUESS", flush=True)
                state = _reset_axis_from_boundary(state, k_guess=k0, full_reset=False, refine_axis_guess=False)
                if verbose and bool(vmec2000_control) and bool(verbose_vmec2000_table):
                    if axis_reset_coeffs is not None:
                        raxis_cc, _raxis_cs, _zaxis_cc, zaxis_cs = axis_reset_coeffs
                        _print_scan_axis_guess(raxis_cc, zaxis_cs)
                axis_reset_done = True
                ijacob = 1
                state_checkpoint = state
                vRcc, vRss, vZsc, vZcs, vLsc, vLcs = _zero_velocity_blocks_like(
                    vRcc, vRss, vZsc, vZcs, vLsc, vLcs
                )
                res0 = -1.0
                res1 = -1.0
                prev_rz_fsq = 2.0
                _clear_preconditioner_cache_locals()
                cache_constraint_rcon0 = None
                cache_constraint_zcon0 = None
        except Exception:
            pass
    if timing_enabled and t_setup_axis_reset_start is not None:
        timing_stats["setup_axis_reset"] += time.perf_counter() - float(t_setup_axis_reset_start)

    # Cache os.getenv calls that would otherwise be repeated every iteration
    # in the hot loop below (saves ~9 os.getenv calls × ~2144 iters = ~19k calls).
    _env_freeb_include_edge = os.getenv("VMEC_JAX_FREEB_INCLUDE_EDGE", "0").strip().lower()
    _env_force_edge_residual = os.getenv("VMEC_JAX_FORCE_EDGE_RESIDUAL", "").strip().lower()
    _env_freeb_raise = os.getenv("VMEC_JAX_FREEB_RAISE", "").strip().lower()
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
                include_edge = _env_freeb_include_edge not in ("", "0", "false", "no")
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
            if bool(free_boundary_enabled and freeb_couple_edge):
                try:
                    # VMEC free-boundary path in funct3d only enters NESTOR
                    # once control is active (`ivac >= 0`), with `vacuum.f`
                    # promoting ivac=0 -> 1 internally on first turn-on.
                    if int(freeb_ivac) >= 0:
                        nestor_res, freeb_nestor_runtime = nestor_external_only_step(
                            state=state,
                            static=static,
                            ivac=int(freeb_ivac),
                            ivacskip=int(freeb_ivacskip),
                            iter_idx=int(iter2),
                            runtime=freeb_nestor_runtime,
                            extcur=tuple(getattr(static, "free_boundary_extcur", ()) or ()),
                            plascur=float(freeb_plascur),
                            external_field_provider_kind=external_field_provider_kind,
                            external_field_provider_static=external_field_provider_static,
                            external_field_provider_params=external_field_provider_params,
                            collect_trace_arrays=bool(adjoint_trace and adjoint_trace_mode in {"full", "branch"}),
                        )
                        freeb_last_model = str(getattr(nestor_res, "model", "spectral_poisson_external_only"))
                        freeb_reused = bool(getattr(nestor_res, "reused", False))
                        freeb_solve_time = float(getattr(nestor_res, "solve_time_s", 0.0))
                        freeb_sample_time = float(getattr(nestor_res, "sample_time_s", 0.0))
                        diag_nestor = getattr(nestor_res, "diagnostics", None)
                        freeb_nestor_trace_current = getattr(nestor_res, "trace_arrays", None)
                        if isinstance(diag_nestor, dict):
                            freeb_last_diagnostics = dict(diag_nestor)
                            freeb_nestor_source_reused_history.append(
                                1 if bool(diag_nestor.get("source_reused", False)) else 0
                            )
                            freeb_nestor_provider_allows_source_reuse_history.append(
                                1 if bool(diag_nestor.get("provider_allows_source_reuse", False)) else 0
                            )
                            for _key, _hist in (
                                ("bnormal_rms", freeb_nestor_bnormal_rms_history),
                                ("gsource_rms", freeb_nestor_gsource_rms_history),
                                ("bsqvac_rms", freeb_nestor_bsqvac_rms_history),
                            ):
                                try:
                                    _hist.append(float(diag_nestor.get(_key, float("nan"))))
                                except Exception:
                                    _hist.append(float("nan"))
                        else:
                            freeb_nestor_source_reused_history.append(0)
                            freeb_nestor_provider_allows_source_reuse_history.append(0)
                            freeb_nestor_bnormal_rms_history.append(float("nan"))
                            freeb_nestor_gsource_rms_history.append(float("nan"))
                            freeb_nestor_bsqvac_rms_history.append(float("nan"))
                        bsqvac_edge = np.asarray(nestor_res.vac_total.bsqvac, dtype=float)
                        if (
                            bsqvac_edge.ndim == 2
                            and int(bsqvac_edge.shape[1]) == 1
                            and int(getattr(static.cfg, "nzeta", 1)) > 1
                        ):
                            bsqvac_edge = np.repeat(bsqvac_edge, int(static.cfg.nzeta), axis=1)
                        # Only the edge slice is consumed by the force kernels.
                        # Keep this as a 2D edge field so the GPU path does not
                        # re-transfer a mostly-zero `(ns, ntheta, nzeta)` array
                        # on every free-boundary iteration.
                        freeb_bsqvac_half_current = bsqvac_edge
                        if freeb_turnon_iter:
                            # VMEC promotes ivac=0 -> 1 inside vacuum.f before
                            # the same-iteration funct3d restart on turn-on.
                            freeb_ivac = 1
                            freeb_ivac_effective = 1
                            freeb_controls_cached = (
                                int(freeb_ivac),
                                int(freeb_ivacskip),
                                int(freeb_nvacskip),
                            )
                except Exception:
                    if _env_freeb_raise not in ("", "0", "false", "no"):
                        raise
                    freeb_bsqvac_half_current = None
                    freeb_nestor_trace_current = None
                    freeb_reused = False
                    freeb_solve_time = 0.0
                    freeb_sample_time = 0.0

            def _freeb_bsqvac_half_for_trial_state(candidate_state: VMECState):
                """Return a non-mutating direct-provider vacuum field for trials.

                Legacy mgrid runs keep VMEC's committed ivac/ivacskip cadence.
                Direct coil providers need candidate-state sampling during
                trial/backtracking scoring so the trial boundary is not scored
                against stale pre-update vacuum source data. The scratch
                runtime returned by NESTOR is intentionally discarded so
                rejected trials cannot mutate the accepted runtime state.
                """

                if not bool(free_boundary_enabled and freeb_couple_edge):
                    return freeb_bsqvac_half_current
                if freeb_bsqvac_half_current is None:
                    return None
                provider_kind_trial = (
                    "mgrid"
                    if external_field_provider_kind is None
                    else str(external_field_provider_kind).strip().lower()
                )
                if provider_kind_trial in ("", "mgrid", "legacy_mgrid"):
                    return freeb_bsqvac_half_current
                if isinstance(external_field_provider_static, dict) and not bool(
                    external_field_provider_static.get("resample_trial_bsqvac", True)
                ):
                    return freeb_bsqvac_half_current
                if int(freeb_ivac_effective) < 1:
                    return freeb_bsqvac_half_current
                try:
                    nestor_trial, _runtime_trial = nestor_external_only_step(
                        state=candidate_state,
                        static=static,
                        ivac=1,
                        ivacskip=0,
                        iter_idx=int(iter2),
                        runtime=freeb_nestor_runtime,
                        extcur=tuple(getattr(static, "free_boundary_extcur", ()) or ()),
                        plascur=float(freeb_plascur),
                        external_field_provider_kind=external_field_provider_kind,
                        external_field_provider_static=external_field_provider_static,
                        external_field_provider_params=external_field_provider_params,
                    )
                    freeb_nestor_trial_reused_history.append(1 if bool(getattr(nestor_trial, "reused", False)) else 0)
                    freeb_nestor_trial_solve_time_history.append(float(getattr(nestor_trial, "solve_time_s", 0.0)))
                    freeb_nestor_trial_sample_time_history.append(float(getattr(nestor_trial, "sample_time_s", 0.0)))
                    freeb_nestor_trial_failed_history.append(0)
                    bsqvac_edge_trial = np.asarray(nestor_trial.vac_total.bsqvac, dtype=float)
                    if (
                        bsqvac_edge_trial.ndim == 2
                        and int(bsqvac_edge_trial.shape[1]) == 1
                        and int(getattr(static.cfg, "nzeta", 1)) > 1
                    ):
                        bsqvac_edge_trial = np.repeat(bsqvac_edge_trial, int(static.cfg.nzeta), axis=1)
                    return bsqvac_edge_trial
                except Exception:
                    freeb_nestor_trial_reused_history.append(0)
                    freeb_nestor_trial_solve_time_history.append(0.0)
                    freeb_nestor_trial_sample_time_history.append(0.0)
                    freeb_nestor_trial_failed_history.append(1)
                    if _env_freeb_raise not in ("", "0", "false", "no"):
                        raise
                    return freeb_bsqvac_half_current

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
            if need_scalar:
                if host_update_assembly and (not _tree_has_tracer(k)):
                    try:
                        r00_val = float(np.asarray(k.pr1_even)[0, 0, 0])
                        z00_val = float(np.asarray(k.pz1_even)[0, 0, 0]) if bool(cfg.lasym) else 0.0
                    except Exception:
                        if not np.any(m0_mask):
                            r00_val = float("nan")
                            z00_val = float("nan")
                        else:
                            r00_val = float(np.sum(np.asarray(state.Rcos)[0, m0_mask]))
                            z00_val = float(np.sum(np.asarray(state.Zcos)[0, m0_mask])) if bool(cfg.lasym) else 0.0
                    wb_val = float(np.asarray(norms_current.wb))
                    wp_val = float(np.asarray(norms_current.wp))
                else:
                    try:
                        r00_j = jnp.asarray(k.pr1_even)[0, 0, 0]
                        if bool(cfg.lasym):
                            z00_j = jnp.asarray(k.pz1_even)[0, 0, 0]
                        else:
                            z00_j = jnp.asarray(0.0, dtype=jnp.asarray(r00_j).dtype)
                    except Exception:
                        if not np.any(m0_mask):
                            r00_j = jnp.asarray(float("nan"))
                            z00_j = jnp.asarray(float("nan"))
                        else:
                            r00_j = jnp.sum(jnp.asarray(state.Rcos)[0, m0_mask])
                            if bool(cfg.lasym):
                                z00_j = jnp.sum(jnp.asarray(state.Zcos)[0, m0_mask])
                            else:
                                z00_j = jnp.asarray(0.0, dtype=jnp.asarray(r00_j).dtype)
                    # `norms_used` may be cached (VMEC2000 `ns4=25` behavior), but
                    # `norms_current` already reflects the current bcovar state and
                    # therefore matches VMEC's printed wb/wp without recomputing.
                    wb_j = jnp.asarray(norms_current.wb)
                    wp_j = jnp.asarray(norms_current.wp)
                    r00_val, z00_val, wb_val, wp_val = _device_get_floats(r00_j, z00_j, wb_j, wp_j)
                if bool(vmec2000_control):
                    # Match VMEC's printed precision (E11.3) for parity checks.
                    r00_val = float(f"{float(r00_val):.3E}")
                    z00_val = float(f"{float(z00_val):.3E}")
            else:
                r00_val = r00_last
                z00_val = z00_last
                wb_val = wb_last
                wp_val = wp_last
            r00_last = float(r00_val)
            z00_last = float(z00_val)
            wb_last = float(wb_val)
            wp_last = float(wp_val)
            w_vmec_last = (wb_last + wp_last / (gamma - 1.0)) * float(TWOPI * TWOPI)
            if track_history:
                r00_history.append(r00_last)
                z00_history.append(z00_last)
                wb_history.append(wb_last)
                wp_history.append(wp_last)
                w_vmec_history.append(w_vmec_last)

            if verbose and (not (bool(vmec2000_control) and bool(verbose_vmec2000_table))):
                print(
                    f"[solve_fixed_boundary_residual_iter] iter={it:03d} fsqr={fsqr_f:.3e} fsqz={fsqz_f:.3e} "
                    f"fsql={fsql_f:.3e} include_edge={include_edge}",
                    flush=True,
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
                from vmec_jax.preconditioner_1d_jax import rz_preconditioner_matrices_reassemble

                precond_traced = _tree_has_tracer(k)
                need_lam_prec = _env_dump_lam not in ("", "0")
                need_lamcal = _env_dump_lamcal not in ("", "0")
                t_prec_refresh_start = time.perf_counter() if timing_enabled else None
                precond_cache_update = _update_preconditioner_cache(
                    bc=k.bc,
                    k=k,
                    cfg=cfg,
                    precond_traced=bool(precond_traced),
                    vmec2000_cache_valid=bool(vmec2000_cache_valid),
                    need_bcovar_update=bool(need_bcovar_update),
                    precond_cache_seeded_from_bcovar_update=bool(precond_cache_seeded_from_bcovar_update),
                    need_lam_prec=bool(need_lam_prec),
                    need_lamcal=bool(need_lamcal),
                    cache_prec_lam_prec=cache_prec_lam_prec,
                    cache_prec_faclam=cache_prec_faclam,
                    cache_prec_lam_debug=cache_prec_lam_debug,
                    cache_prec_rz_mats=cache_prec_rz_mats,
                    cache_prec_rz_jmax=cache_prec_rz_jmax,
                    precond_expected_jmax=int(precond_expected_jmax),
                    precond_jmax_override=precond_jmax_override,
                    preconditioner_use_precomputed_tridi=preconditioner_use_precomputed_tridi_policy,
                    preconditioner_use_lax_tridi=preconditioner_use_lax_tridi_policy,
                    lambda_preconditioner_func=_lambda_preconditioner,
                    rz_preconditioner_matrices_func=_rz_preconditioner_matrices_local,
                    rz_preconditioner_matrices_reassemble_func=rz_preconditioner_matrices_reassemble,
                    can_reassemble_func=_can_reassemble_precond_mats,
                )
                precond_cache_decision = precond_cache_update.decision
                need_prec_refresh = precond_cache_decision.need_prec_refresh
                if need_prec_refresh:
                    preconditioner_cache_update_trace = True
                    if timing_enabled:
                        timing_stats["precond_refresh_calls"] = int(timing_stats["precond_refresh_calls"]) + 1
                    if timing_enabled and t_prec_refresh_start is not None:
                        try:
                            if has_jax():
                                jax.block_until_ready(precond_cache_update.lam_prec)
                        except Exception:
                            pass
                        timing_stats["precond_refresh"] += time.perf_counter() - float(t_prec_refresh_start)
                else:
                    if timing_enabled:
                        timing_stats["precond_cache_hit_count"] = int(timing_stats["precond_cache_hit_count"]) + 1
                        if bool(precond_cache_decision.can_reuse_bcovar_seeded_precond) and bool(need_bcovar_update):
                            timing_stats["precond_refresh_seed_reuse_count"] = (
                                int(timing_stats["precond_refresh_seed_reuse_count"]) + 1
                            )
                    if bool(precond_cache_decision.need_prec_reassemble):
                        if timing_enabled:
                            timing_stats["precond_reassemble_calls"] = int(timing_stats["precond_reassemble_calls"]) + 1
                lam_prec = precond_cache_update.lam_prec
                faclam_dump = precond_cache_update.faclam_dump
                lam_debug = precond_cache_update.lam_debug
                mats = precond_cache_update.mats
                jmax = precond_cache_update.jmax
                cache_prec_lam_prec = precond_cache_update.cache_prec_lam_prec
                cache_prec_faclam = precond_cache_update.cache_prec_faclam
                cache_prec_lam_debug = precond_cache_update.cache_prec_lam_debug
                cache_prec_rz_mats = precond_cache_update.cache_prec_rz_mats
                cache_prec_rz_jmax = precond_cache_update.cache_prec_rz_jmax
                _maybe_dump_lam_prec(lam_prec=lam_prec, faclam=faclam_dump, static=static, iter_idx=int(iter2))
                if not precond_traced:
                    _maybe_dump_precond_mats(
                        mats=mats,
                        static=static,
                        iter_idx=int(iter2),
                        jmax=int(jmax),
                        used_cache=(not bool(need_prec_refresh)),
                    )
                if lam_debug is not None:
                    _maybe_dump_lamcal(lam_debug=lam_debug, static=static, iter_idx=int(iter2))
                t_precond_apply_start = time.perf_counter() if timing_detail_enabled else None
                use_apply_payload_fusion = (
                    bool(use_fused_precond_output_scaling)
                    and need_lam_prec is False
                    and need_lamcal is False
                )
                frzl_rhs = _apply_vmec_scale_m1_precond_rhs(frzl, mats)
                if use_apply_payload_fusion:
                    if (
                        bool(vmec2000_control)
                        and bool(vmec2000_cache_valid)
                        and (not bool(need_bcovar_update))
                        and (cache_rz_norm is not None)
                        and (cache_f_norm1 is not None)
                    ):
                        f_norm1 = jnp.asarray(cache_f_norm1)
                    else:
                        rz_norm = _rz_norm(state)
                        f_norm1 = jnp.where(
                            rz_norm != 0.0,
                            1.0 / rz_norm,
                            jnp.asarray(float("inf"), dtype=rz_norm.dtype),
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
                    if len(_precond_payload) == 4:
                        (
                            _precond_pre_blocks,
                            _precond_update_blocks,
                            _precond_diag,
                            accepted_control_ptau_payload,
                        ) = _precond_payload
                    else:
                        _precond_pre_blocks, _precond_update_blocks, _precond_diag = _precond_payload
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
                    frzl_rz = _rz_preconditioner_apply_local(
                        frzl_in=frzl_rhs,
                        mats=mats,
                        jmax=jmax,
                        use_precomputed=preconditioner_use_precomputed_tridi_policy,
                        use_lax_tridi=preconditioner_use_lax_tridi_policy,
                    )
                    frzl_lam_pre = frzl_rz
                if use_apply_payload_fusion and adjoint_trace and adjoint_trace_mode == "full":
                    # The fused GPU-oriented path returns only scaled update
                    # payloads.  Full accepted-trace replay also needs the raw
                    # R/Z-preconditioned force, so materialize it only for that
                    # opt-in diagnostic/validation mode.
                    frzl_rz = _rz_preconditioner_apply_local(
                        frzl_in=frzl_rhs,
                        mats=mats,
                        jmax=jmax,
                        use_precomputed=preconditioner_use_precomputed_tridi_policy,
                        use_lax_tridi=preconditioner_use_lax_tridi_policy,
                    )
                if (not use_apply_payload_fusion) and host_update_assembly:
                    # NumPy path: avoids ~15 JAX dispatches (jnp.asarray, zeros_like, mul).
                    # Asymmetric (lasym) components default to None — the downstream
                    # mode-diag scaling uses _z (pre-allocated zeros) for None entries,
                    # avoiding 6 np.zeros_like allocations per iteration (~0.5s saving).
                    (frcc, frss, fzsc, fzcs, flsc, flcs, frsc, frcs, fzcc, fzss, flcc, flss) = (
                        _preconditioner_output_blocks_np(frzl_rz=frzl_rz, lam_prec=lam_prec)
                    )
                elif (not use_apply_payload_fusion) and use_fused_precond_output_scaling:
                    if (
                        bool(vmec2000_control)
                        and bool(vmec2000_cache_valid)
                        and (not bool(need_bcovar_update))
                        and (cache_rz_norm is not None)
                        and (cache_f_norm1 is not None)
                    ):
                        rz_norm = jnp.asarray(cache_rz_norm)
                        f_norm1 = jnp.asarray(cache_f_norm1)
                    else:
                        rz_norm = _rz_norm(state)
                        f_norm1 = jnp.where(
                            rz_norm != 0.0,
                            1.0 / rz_norm,
                            jnp.asarray(float("inf"), dtype=rz_norm.dtype),
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
                from vmec_jax.preconditioner_1d_jax import rz_preconditioner_matrices_reassemble

                precond_traced = _tree_has_tracer(k)
                need_lam_prec = _env_dump_lam not in ("", "0")
                need_lamcal = _env_dump_lamcal not in ("", "0")
                t_prec_refresh_start = time.perf_counter() if timing_enabled else None
                precond_cache_update = _update_preconditioner_cache(
                    bc=k.bc,
                    k=k,
                    cfg=cfg,
                    precond_traced=bool(precond_traced),
                    vmec2000_cache_valid=bool(vmec2000_cache_valid),
                    need_bcovar_update=bool(need_bcovar_update),
                    precond_cache_seeded_from_bcovar_update=bool(precond_cache_seeded_from_bcovar_update),
                    need_lam_prec=bool(need_lam_prec),
                    need_lamcal=bool(need_lamcal),
                    cache_prec_lam_prec=cache_prec_lam_prec,
                    cache_prec_faclam=cache_prec_faclam,
                    cache_prec_lam_debug=cache_prec_lam_debug,
                    cache_prec_rz_mats=cache_prec_rz_mats,
                    cache_prec_rz_jmax=cache_prec_rz_jmax,
                    precond_expected_jmax=int(precond_expected_jmax),
                    precond_jmax_override=precond_jmax_override,
                    preconditioner_use_precomputed_tridi=preconditioner_use_precomputed_tridi_policy,
                    preconditioner_use_lax_tridi=preconditioner_use_lax_tridi_policy,
                    lambda_preconditioner_func=_lambda_preconditioner,
                    rz_preconditioner_matrices_func=_rz_preconditioner_matrices_local,
                    rz_preconditioner_matrices_reassemble_func=rz_preconditioner_matrices_reassemble,
                    can_reassemble_func=_can_reassemble_precond_mats,
                )
                precond_cache_decision = precond_cache_update.decision
                need_prec_refresh = precond_cache_decision.need_prec_refresh
                if need_prec_refresh:
                    preconditioner_cache_update_trace = True
                    if timing_enabled:
                        timing_stats["precond_refresh_calls"] = int(timing_stats["precond_refresh_calls"]) + 1
                    if timing_enabled and t_prec_refresh_start is not None:
                        try:
                            if has_jax():
                                jax.block_until_ready(precond_cache_update.lam_prec)
                        except Exception:
                            pass
                        timing_stats["precond_refresh"] += time.perf_counter() - float(t_prec_refresh_start)
                else:
                    if timing_enabled:
                        timing_stats["precond_cache_hit_count"] = int(timing_stats["precond_cache_hit_count"]) + 1
                        if bool(precond_cache_decision.can_reuse_bcovar_seeded_precond) and bool(need_bcovar_update):
                            timing_stats["precond_refresh_seed_reuse_count"] = (
                                int(timing_stats["precond_refresh_seed_reuse_count"]) + 1
                            )
                    if bool(precond_cache_decision.need_prec_reassemble):
                        if timing_enabled:
                            timing_stats["precond_reassemble_calls"] = int(timing_stats["precond_reassemble_calls"]) + 1
                lam_prec = precond_cache_update.lam_prec
                faclam_dump = precond_cache_update.faclam_dump
                lam_debug = precond_cache_update.lam_debug
                mats = precond_cache_update.mats
                jmax = precond_cache_update.jmax
                cache_prec_lam_prec = precond_cache_update.cache_prec_lam_prec
                cache_prec_faclam = precond_cache_update.cache_prec_faclam
                cache_prec_lam_debug = precond_cache_update.cache_prec_lam_debug
                cache_prec_rz_mats = precond_cache_update.cache_prec_rz_mats
                cache_prec_rz_jmax = precond_cache_update.cache_prec_rz_jmax
                _maybe_dump_lam_prec(lam_prec=lam_prec, faclam=faclam_dump, static=static, iter_idx=int(iter2))
                if not precond_traced:
                    _maybe_dump_precond_mats(
                        mats=mats,
                        static=static,
                        iter_idx=int(iter2),
                        jmax=int(jmax),
                        used_cache=(not bool(need_prec_refresh)),
                    )
                if lam_debug is not None:
                    _maybe_dump_lamcal(lam_debug=lam_debug, static=static, iter_idx=int(iter2))
                t_precond_apply_start = time.perf_counter() if timing_detail_enabled else None
                use_apply_payload_fusion = (
                    bool(use_fused_precond_output_scaling)
                    and need_lam_prec is False
                    and need_lamcal is False
                )
                frzl_rhs = _apply_vmec_scale_m1_precond_rhs(frzl, mats) if bool(getattr(cfg, "lasym", False)) else frzl
                if use_apply_payload_fusion:
                    if (
                        bool(vmec2000_control)
                        and bool(vmec2000_cache_valid)
                        and (not bool(need_bcovar_update))
                        and (cache_rz_norm is not None)
                        and (cache_f_norm1 is not None)
                    ):
                        f_norm1 = jnp.asarray(cache_f_norm1)
                    else:
                        rz_norm = _rz_norm(state)
                        f_norm1 = jnp.where(
                            rz_norm != 0.0,
                            1.0 / rz_norm,
                            jnp.asarray(float("inf"), dtype=rz_norm.dtype),
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
                    if len(_precond_payload) == 4:
                        (
                            _precond_pre_blocks,
                            _precond_update_blocks,
                            _precond_diag,
                            accepted_control_ptau_payload,
                        ) = _precond_payload
                    else:
                        _precond_pre_blocks, _precond_update_blocks, _precond_diag = _precond_payload
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
                    frzl_rz = _rz_preconditioner_apply_local(
                        frzl_in=frzl_rhs,
                        mats=mats,
                        jmax=jmax,
                        use_precomputed=preconditioner_use_precomputed_tridi_policy,
                        use_lax_tridi=preconditioner_use_lax_tridi_policy,
                    )
                    frzl_lam_pre = frzl_rz
                if use_apply_payload_fusion and adjoint_trace and adjoint_trace_mode == "full":
                    # The fused GPU-oriented path returns only scaled update
                    # payloads.  Full accepted-trace replay also needs the raw
                    # R/Z-preconditioned force, so materialize it only for that
                    # opt-in diagnostic/validation mode.
                    frzl_rz = _rz_preconditioner_apply_local(
                        frzl_in=frzl_rhs,
                        mats=mats,
                        jmax=jmax,
                        use_precomputed=preconditioner_use_precomputed_tridi_policy,
                        use_lax_tridi=preconditioner_use_lax_tridi_policy,
                    )
                if (not use_apply_payload_fusion) and host_update_assembly:
                    # NumPy path: avoids ~15 JAX dispatches (jnp.asarray, zeros_like, mul).
                    # Asymmetric (lasym) components default to None — the downstream
                    # mode-diag scaling uses _z (pre-allocated zeros) for None entries,
                    # avoiding 6 np.zeros_like allocations per iteration (~0.5s saving).
                    (frcc, frss, fzsc, fzcs, flsc, flcs, frsc, frcs, fzcc, fzss, flcc, flss) = (
                        _preconditioner_output_blocks_np(frzl_rz=frzl_rz, lam_prec=lam_prec)
                    )
                elif (not use_apply_payload_fusion) and use_fused_precond_output_scaling:
                    if (
                        bool(vmec2000_control)
                        and bool(vmec2000_cache_valid)
                        and (not bool(need_bcovar_update))
                        and (cache_rz_norm is not None)
                        and (cache_f_norm1 is not None)
                    ):
                        rz_norm = jnp.asarray(cache_rz_norm)
                        f_norm1 = jnp.asarray(cache_f_norm1)
                    else:
                        rz_norm = _rz_norm(state)
                        f_norm1 = jnp.where(
                            rz_norm != 0.0,
                            1.0 / rz_norm,
                            jnp.asarray(float("inf"), dtype=rz_norm.dtype),
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
                    t_auto_flip_force_start = time.perf_counter() if timing_detail_enabled else None
                    _, _, gcr2_t, gcz2_t, gcl2_t, _, _, norms_t = _compute_forces_iter(
                        st_try,
                        include_edge=True,
                        zero_m1=zero_m1,
                        freeb_bsqvac_half=_freeb_bsqvac_half_for_trial_state(st_try),
                        constraint_precond_diag=constraint_precond_diag,
                        constraint_tcon=constraint_tcon_override,
                        constraint_precond_active=constraint_precond_active,
                        constraint_tcon_active=constraint_tcon_active,
                        iter2=iter2,
                    )
                    _record_compute_force_timing("auto_flip", t_auto_flip_force_start, gcr2_t)
                    fsqr_t, fsqz_t, fsql_t = _residual_fsq_from_norms(
                        norms_t,
                        gcr2=gcr2_t,
                        gcz2=gcz2_t,
                        gcl2=gcl2_t,
                    )
                    return float(np.asarray(fsqr_t + fsqz_t + fsql_t))

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
                # Fast NumPy path: use cached Python-float fnorm1 directly — no JAX dispatch.
                if (
                    bool(vmec2000_control)
                    and bool(vmec2000_cache_valid)
                    and (not bool(need_bcovar_update))
                    and (cache_rz_norm is not None)
                    and (cache_f_norm1 is not None)
                ):
                    _f_norm1_np = float(cache_f_norm1)
                    rz_norm = cache_rz_norm  # Python float (for history list)
                else:
                    _rz_norm_val = _rz_norm_np(state)
                    _f_norm1_np = (1.0 / _rz_norm_val) if _rz_norm_val != 0.0 else float("inf")
                    rz_norm = _rz_norm_val
                f_norm1 = _f_norm1_np  # alias for history list (Python float)
                _finite = np.isfinite(_f_norm1_np)
                fsqr1 = float(gcr2_p) * _f_norm1_np if _finite else 0.0
                fsqz1 = float(gcz2_p) * _f_norm1_np if _finite else 0.0
                if bool(vmec2000_control):
                    # VMEC2000 `residue.f90`: fsql1 = hs * SUM( (faclam*gcl)**2 ) over all js.
                    frzl_for_gcl2_full = frzl_pre if frzl_pre_host is None else frzl_pre_host
                    _gcl2_full = _lambda_preconditioned_full_norm(
                        frzl_for_gcl2_full,
                        use_jax=False,
                    )
                    fsql1 = _gcl2_full * delta_s
                else:
                    fsql1 = float(gcl2_p) * delta_s
                # Safe values: NaN/Inf → 0 (same semantics as jnp.where below).
                fsqr1_safe = _finite_float_or_zero(fsqr1)
                fsqz1_safe = _finite_float_or_zero(fsqz1)
                fsql1_safe = _finite_float_or_zero(fsql1)
                fsq1 = fsqr1_safe + fsqz1_safe + fsql1_safe
                # host_update_assembly: keep as Python floats — downstream code (history
                # lists, _precond_diag_floats) handles both float and JAX scalar.
                fsqr1 = fsqr1_safe
                fsqz1 = fsqz1_safe
                fsql1 = fsql1_safe
            else:
                # JAX path: set rz_norm and f_norm1 from cache or recompute.
                if (
                    bool(vmec2000_control)
                    and bool(vmec2000_cache_valid)
                    and (not bool(need_bcovar_update))
                    and (cache_rz_norm is not None)
                    and (cache_f_norm1 is not None)
                ):
                    rz_norm = jnp.asarray(cache_rz_norm)
                    f_norm1 = jnp.asarray(cache_f_norm1)
                else:
                    rz_norm = _rz_norm(state)
                    f_norm1 = jnp.where(rz_norm != 0.0, 1.0 / rz_norm, jnp.asarray(float("inf"), dtype=rz_norm.dtype))
                # Avoid inf*0 -> NaN in late-converged iterations when rz_norm=0 and
                # gcx2 terms are exactly zero. VMEC treats these channels as zero.
                finite_fnorm1 = jnp.isfinite(f_norm1)
                fsqr1 = jnp.where(finite_fnorm1, gcr2_p * f_norm1, jnp.asarray(0.0, dtype=jnp.asarray(gcr2_p).dtype))
                fsqz1 = jnp.where(finite_fnorm1, gcz2_p * f_norm1, jnp.asarray(0.0, dtype=jnp.asarray(gcz2_p).dtype))
                if bool(vmec2000_control):
                    # VMEC2000 `residue.f90`: fsql1 = hs * SUM( (faclam*gcl)**2 ) over all js.
                    gcl2_full = _lambda_preconditioned_full_norm(frzl_pre, use_jax=True)
                    fsql1 = gcl2_full * delta_s
                else:
                    fsql1 = gcl2_p * delta_s
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
                if not preconditioner_fsq1_ready:
                    fsqr1_safe = jnp.where(
                        jnp.isfinite(fsqr1),
                        fsqr1,
                        jnp.asarray(0.0, dtype=jnp.asarray(fsqr1).dtype),
                    )
                    fsqz1_safe = jnp.where(
                        jnp.isfinite(fsqz1),
                        fsqz1,
                        jnp.asarray(0.0, dtype=jnp.asarray(fsqz1).dtype),
                    )
                    fsql1_safe = jnp.where(
                        jnp.isfinite(fsql1),
                        fsql1,
                        jnp.asarray(0.0, dtype=jnp.asarray(fsql1).dtype),
                    )
                if preconditioner_fsq1_ready:
                    fsq1_j = fsq1_safe
                else:
                    fsq1_j = fsqr1_safe + fsqz1_safe + fsql1_safe
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
                control_payload_used = False
                if accepted_control_ptau_payload is not None:
                    t_fsq1_payload_get_start = time.perf_counter() if timing_enabled else None
                    try:
                        fsq1_payload, ptau_min_payload, ptau_max_payload = accepted_control_ptau_payload
                        fsq1, min_tau_ptau_payload, max_tau_ptau_payload = _device_get_floats(
                            fsq1_payload,
                            ptau_min_payload,
                            ptau_max_payload,
                        )
                        accepted_control_ptau_host = (min_tau_ptau_payload, max_tau_ptau_payload)
                        control_payload_used = True
                    except Exception:
                        control_payload_used = False
                    finally:
                        if timing_enabled and t_fsq1_payload_get_start is not None:
                            timing_stats["iteration_control_fsq1_payload_get"] += time.perf_counter() - float(
                                t_fsq1_payload_get_start
                            )
                if (not control_payload_used) and use_control_payload:
                    ptau_arrays = _scan_math_kernel_arrays_from_k(k)
                    payload_fn = _accepted_control_payload_jit()
                    if ptau_arrays is not None and payload_fn is not None:
                        t_fsq1_payload_get_start = time.perf_counter() if timing_enabled else None
                        try:
                            fsq1_payload, ptau_min_payload, ptau_max_payload = payload_fn(
                                fsq1_j,
                                *ptau_arrays,
                                _ptau_context.pshalf_jax,
                                _ptau_context.ohs_jax,
                            )
                            fsq1, min_tau_ptau_payload, max_tau_ptau_payload = _device_get_floats(
                                fsq1_payload,
                                ptau_min_payload,
                                ptau_max_payload,
                            )
                            accepted_control_ptau_host = (min_tau_ptau_payload, max_tau_ptau_payload)
                            control_payload_used = True
                        except Exception:
                            control_payload_used = False
                        finally:
                            if timing_enabled and t_fsq1_payload_get_start is not None:
                                timing_stats["iteration_control_fsq1_payload_get"] += time.perf_counter() - float(
                                    t_fsq1_payload_get_start
                                )
                if not control_payload_used:
                    t_fsq1_direct_get_start = time.perf_counter() if timing_enabled else None
                    fsq1 = float(jax.device_get(fsq1_j))
                    if timing_enabled and t_fsq1_direct_get_start is not None:
                        timing_stats["iteration_control_fsq1_direct_get"] += time.perf_counter() - float(
                            t_fsq1_direct_get_start
                        )
            if timing_enabled and t_iteration_control_fsq1_start is not None:
                timing_stats["iteration_control_fsq1"] += time.perf_counter() - float(t_iteration_control_fsq1_start)
            precond_diag_host: tuple[float, float, float] | None = None

            def _precond_diag_floats() -> tuple[float, float, float]:
                nonlocal precond_diag_host
                if precond_diag_host is None:
                    precond_diag_host = _device_get_floats(fsqr1_safe, fsqz1_safe, fsql1_safe)
                return precond_diag_host

            if track_history:
                rz_norm_history.append(rz_norm)
                f_norm1_history.append(f_norm1)
                gcr2_p_history.append(gcr2_p)
                gcz2_p_history.append(gcz2_p)
                gcl2_p_history.append(gcl2_p)
                fsq1_history.append(fsq1)
                fsqr1_history.append(fsqr1_safe)
                fsqz1_history.append(fsqz1_safe)
                fsql1_history.append(fsql1_safe)

            if converged_physical:
                if track_history:
                    # Keep per-iteration history channels length-aligned with
                    # fsqr/fsqz/fsql when convergence happens before the update
                    # block. VMEC's table still reports DELT on this row.
                    rec = _residual_iter_history_record(
                        step=0.0,
                        dt_eff=0.0,
                        update_rms=0.0,
                        w_curr=fsqr_f + fsqz_f + fsql_f,
                        w_try=float("nan"),
                        w_try_ratio=float("nan"),
                        restart_path="converged",
                        step_status="converged",
                        restart_reason="none",
                        pre_restart_reason="none",
                        time_step=time_step,
                        res0=res0,
                        res1=res1,
                        fsq_prev=fsq_prev,
                        bad_growth_streak=bad_growth_streak,
                        iter1=iter1,
                        iter2=iter2,
                        fsqr=fsqr_f,
                        fsqz=fsqz_f,
                        fsql=fsql_f,
                        free_boundary_enabled=free_boundary_enabled,
                        freeb_ivac=freeb_ivac,
                        freeb_ivacskip=freeb_ivacskip,
                    )
                    _append_residual_iter_history_record(rec, **_history_record_lists)
                if verbose and not (bool(vmec2000_control) and bool(verbose_vmec2000_table)):
                    print(
                        f"[solve_fixed_boundary_residual_iter] converged: "
                        f"fsqr={fsqr_f:.3e} fsqz={fsqz_f:.3e} fsql={fsql_f:.3e} "
                        f"target={float(fsq_total_target) if fsq_total_target is not None else float(ftol):.3e}",
                        flush=True,
                    )
                if timing_enabled and t_iteration_control_start is not None:
                    timing_stats["iteration_control"] += time.perf_counter() - float(t_iteration_control_start)
                    t_iteration_control_start = None
                if bool(vmec2000_control) and bool(verbose_vmec2000_table):
                    fsqr1_f, fsqz1_f, fsql1_f = _precond_diag_floats()
                    _print_vmec2000_iter_row(
                        iter_idx=int(iter2),
                        fsqr=fsqr_f,
                        fsqz=fsqz_f,
                        fsql=fsql_f,
                        fsqr1=fsqr1_f,
                        fsqz1=fsqz1_f,
                        fsql1=fsql1_f,
                        delt0r=float(time_step),
                        r00=float(r00_last),
                        w_mhd=float(w_vmec_last),
                        z00=float(z00_last),
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
                if min_tau_ptau is not None and max_tau_ptau is not None:
                    if bool(vmec2000_control):
                        tau_tol = _bad_jacobian_tau_tolerance(
                            ptau_tol=ptau_tol,
                            ptau_tol_rel=0.0,
                            tau_scale=0.0,
                        )
                        bad_jacobian_ptau = (min_tau_ptau < -tau_tol) and (max_tau_ptau > tau_tol)
                    else:
                        tau_scale = max(abs(min_tau_ptau), abs(max_tau_ptau))
                        tau_tol = max(1.0e-12, 1.0e-3 * tau_scale)
                        bad_jacobian_ptau = (min_tau_ptau < -tau_tol) and (max_tau_ptau > tau_tol)

                state_probe = _should_probe_bad_jacobian_state(
                    state_probe=bool(badjac_state_probe),
                    initial_state_probe_iters=int(badjac_initial_state_probe_iters),
                    iter_idx=int(iter2),
                )
                need_state_jac = (
                    badjac_use_state
                    or dump_ptau_state
                    or state_probe
                    or (bad_jacobian_ptau is None)
                    or bool(bad_jacobian_ptau)
                )
                if need_state_jac:
                    t_badjac_state_jacobian_start = time.perf_counter() if timing_enabled else None
                    if host_update_assembly and (not _tree_has_tracer(state)) and (not _tree_has_tracer(s)):
                        from vmec_jax.vmec_numpy_forces import _numpy_module_patch as _hot_numpy_patch

                        with _hot_numpy_patch():
                            jac_state = vmec_half_mesh_jacobian_from_state(
                                state=state,
                                modes=static.modes,
                                trig=trig,
                                s=s,
                                lconm1=bool(getattr(static.cfg, "lconm1", True)),
                                lthreed=bool(getattr(static.cfg, "lthreed", True)),
                                mask_even=getattr(static, "m_is_even", None),
                                mask_odd=getattr(static, "m_is_odd", None),
                            )
                        tau = np.asarray(jac_state.tau)
                        if int(tau.size) > 0:
                            tau_use = tau[1:] if int(tau.shape[0]) > 1 else tau
                            min_tau_state = float(np.min(tau_use))
                            max_tau_state = float(np.max(tau_use))
                        else:
                            min_tau_state = float("nan")
                            max_tau_state = float("nan")
                    else:
                        jac_state = vmec_half_mesh_jacobian_from_state(
                            state=state,
                            modes=static.modes,
                            trig=trig,
                            s=s,
                            lconm1=bool(getattr(static.cfg, "lconm1", True)),
                            lthreed=bool(getattr(static.cfg, "lthreed", True)),
                            mask_even=getattr(static, "m_is_even", None),
                            mask_odd=getattr(static, "m_is_odd", None),
                        )
                        tau = jnp.asarray(jac_state.tau)
                        if int(tau.size) > 0:
                            tau_use = tau[1:] if int(tau.shape[0]) > 1 else tau
                            tau_min = jnp.min(tau_use)
                            tau_max = jnp.max(tau_use)
                            min_tau_state, max_tau_state = _device_get_floats(tau_min, tau_max)
                        else:
                            min_tau_state = float("nan")
                            max_tau_state = float("nan")
                    if timing_enabled and t_badjac_state_jacobian_start is not None:
                        timing_stats["iteration_control_badjac_state_jacobian"] += time.perf_counter() - float(
                            t_badjac_state_jacobian_start
                        )
                    if np.isfinite(min_tau_state) and np.isfinite(max_tau_state):
                        if bool(vmec2000_control):
                            tau_tol = _bad_jacobian_tau_tolerance(
                                ptau_tol=ptau_tol,
                                ptau_tol_rel=0.0,
                                tau_scale=0.0,
                            )
                            bad_jacobian_state = (min_tau_state < -tau_tol) and (max_tau_state > tau_tol)
                        else:
                            tau_scale = max(abs(min_tau_state), abs(max_tau_state))
                            tau_tol = max(1.0e-12, 1.0e-3 * tau_scale)
                            bad_jacobian_state = (min_tau_state < -tau_tol) and (max_tau_state > tau_tol)
                    else:
                        bad_jacobian_state = False
                else:
                    min_tau_state = float("nan")
                    max_tau_state = float("nan")
                    bad_jacobian_state = False

                if badjac_use_state:
                    bad_jacobian = bad_jacobian_state
                    min_tau = min_tau_state
                    max_tau = max_tau_state
                else:
                    bad_jacobian = bool(bad_jacobian_ptau) if bad_jacobian_ptau is not None else False
                    min_tau = min_tau_ptau if min_tau_ptau is not None else float("nan")
                    max_tau = max_tau_ptau if max_tau_ptau is not None else float("nan")

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
                    if track_history:
                        min_tau_history.append(min_tau)
                        max_tau_history.append(max_tau)
                        bad_jacobian_history.append(int(bool(bad_jacobian)))
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
                    if track_history:
                        min_tau_history.append(float("nan"))
                        max_tau_history.append(float("nan"))
                        bad_jacobian_history.append(0)
            else:
                if track_history:
                    min_tau_history.append(float("nan"))
                    max_tau_history.append(float("nan"))
                    bad_jacobian_history.append(0)

            # VMEC eqsolve: after the first evolve step, if the Jacobian is bad
            # and ijacob==0, retry with an improved axis guess.
            if bool(vmec2000_control) and (not axis_reset_done) and bool(lmove_axis) and (iter2 == 1):
                fsq_curr = fsqr_f + fsqz_f + fsql_f
                huge_initial_forces = (not np.isfinite(fsq_curr)) or (fsq_curr > 1.0e2)
                force_axis_reset_init = bool(force_axis_reset) or (
                    bool(getattr(cfg, "lthreed", True)) and axis_reset_always_3d
                )
                if (not force_axis_reset_init) and axis_reset_fsq_min > 0.0:
                    if np.isfinite(fsq_curr) and (fsq_curr < axis_reset_fsq_min):
                        bad_jacobian = False
                        huge_initial_forces = False
                if bad_jacobian or huge_initial_forces or force_axis_reset_init:
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
                    if track_history:
                        rec = _residual_iter_history_record(
                            step=0.0,
                            dt_eff=0.0,
                            update_rms=0.0,
                            w_curr=fsqr_f + fsqz_f + fsql_f,
                            w_try=float("nan"),
                            w_try_ratio=float("nan"),
                            restart_path="vmec2000_bad_jacobian" if irst_tc == 2 else "vmec2000_time_control",
                            step_status=step_status,
                            restart_reason=restart_reason,
                            pre_restart_reason=pre_restart_reason,
                            time_step=time_step,
                            res0=res0,
                            res1=res1,
                            fsq_prev=fsq_prev,
                            bad_growth_streak=bad_growth_streak,
                            iter1=iter1,
                            iter2=iter2,
                            fsqr=fsqr_f,
                            fsqz=fsqz_f,
                            fsql=fsql_f,
                            free_boundary_enabled=free_boundary_enabled,
                            freeb_ivac=freeb_ivac,
                            freeb_ivacskip=freeb_ivacskip,
                        )
                        _append_residual_iter_history_record(rec, **_history_record_lists)
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
                if track_history:
                    rec = _residual_iter_history_record(
                        step=0.0,
                        dt_eff=0.0,
                        update_rms=0.0,
                        w_curr=fsqr_f + fsqz_f + fsql_f,
                        w_try=float("nan"),
                        w_try_ratio=float("nan"),
                        restart_path="pre_restart_trigger",
                        step_status=step_status,
                        restart_reason=pre_restart_reason,
                        pre_restart_reason=pre_restart_reason,
                        time_step=time_step_iter,
                        res0=res0,
                        res1=res1,
                        fsq_prev=fsq_prev,
                        bad_growth_streak=bad_growth_streak,
                        iter1=iter1,
                        iter2=iter2,
                        fsqr=fsqr_f,
                        fsqz=fsqz_f,
                        fsql=fsql_f,
                        free_boundary_enabled=free_boundary_enabled,
                        freeb_ivac=freeb_ivac,
                        freeb_ivacskip=freeb_ivacskip,
                    )
                    _append_residual_iter_history_record(rec, **_history_record_lists)
                if verbose:
                    if bool(vmec2000_control) and bool(verbose_vmec2000_table):
                        # VMEC does not print rejected restart steps.
                        pass
                    else:
                        fsqr1_f, fsqz1_f, fsql1_f = _precond_diag_floats()
                        print(
                            f"[solve_fixed_boundary_residual_iter] iter={it:03d} "
                            f"dt_eff=0.000e+00 update_rms=0.000e+00 "
                            f"fsqr1={fsqr1_f:.3e} fsqz1={fsqz1_f:.3e} fsql1={fsql1_f:.3e} "
                            f"step_status={step_status}",
                            flush=True,
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
                constraint_precond_diag_trace = (
                    None
                    if constraint_precond_diag is None
                    else tuple(_adjoint_trace_array(x) for x in constraint_precond_diag)
                )
                trace_entry: dict[str, Any] = {
                    "branch": "strict_update",
                    "state_pre": state_backup,
                    "force_state_pre": force_state_pre_current,
                    "max_update_rms_pre": float(max_update_rms),
                    "max_coeff_delta_rms_pre": float(max_coeff_delta_rms),
                    "divide_by_scalxc_for_update": bool(divide_by_scalxc_for_update),
                    "lambda_update_scale": float(lambda_update_scale),
                    "apply_lforbal": bool(apply_lforbal),
                    "apply_m1_constraints": bool(apply_m1_constraints),
                    "include_edge_residual": bool(include_edge_residual),
                    "vmec2000_control": bool(vmec2000_control),
                    "limit_dt_from_force": bool(limit_dt_from_force),
                    "signgs": int(signgs),
                    "zero_m1": _adjoint_trace_array(zero_m1),
                    "wout_like": wout_like,
                    "trig": trig,
                    "w_mode_mn": _adjoint_trace_array(w_mode_mn),
                    "precond_jmax": int(jmax),
                    "preconditioner_use_precomputed_tridi": bool(preconditioner_use_precomputed_tridi_policy),
                    "preconditioner_use_lax_tridi": bool(preconditioner_use_lax_tridi_policy),
                    "inv_tau_before": _adjoint_trace_array(inv_tau),
                    "fsq_prev_before": float(fsq_prev_before),
                    "reset_inv_tau": bool(iter2 == iter1),
                    "constraint_cache_update": bool(need_bcovar_update),
                    "precond_cache_update": bool(preconditioner_cache_update_trace),
                    "vRcc_before": _adjoint_trace_array(vRcc),
                    "vRss_before": _adjoint_trace_array(vRss),
                    "vZsc_before": _adjoint_trace_array(vZsc),
                    "vZcs_before": _adjoint_trace_array(vZcs),
                    "vLsc_before": _adjoint_trace_array(vLsc),
                    "vLcs_before": _adjoint_trace_array(vLcs),
                    "vRsc_before": _adjoint_trace_array(vRsc),
                    "vRcs_before": _adjoint_trace_array(vRcs),
                    "vZcc_before": _adjoint_trace_array(vZcc),
                    "vZss_before": _adjoint_trace_array(vZss),
                    "vLcc_before": _adjoint_trace_array(vLcc),
                    "vLss_before": _adjoint_trace_array(vLss),
                    "freeb_bsqvac_half": (
                        None
                        if freeb_bsqvac_half_current is None
                        else _adjoint_trace_array(freeb_bsqvac_half_current)
                    ),
                    "freeb_pres_scale": None if freeb_pres_scale is None else float(freeb_pres_scale),
                    "freeb_plascur": float(freeb_plascur),
                    "freeb_plascur_for_bsqvac": float(freeb_plascur_for_bsqvac),
                    "freeb_nestor_trace": freeb_nestor_trace_current,
                    "constraint_rcon0": (
                        None if constraint_rcon0_current is None else _adjoint_trace_array(constraint_rcon0_current)
                    ),
                    "constraint_zcon0": (
                        None if constraint_zcon0_current is None else _adjoint_trace_array(constraint_zcon0_current)
                    ),
                    "constraint_tcon0": None if constraint_tcon0 is None else float(constraint_tcon0),
                    "constraint_precond_diag": constraint_precond_diag_trace,
                    "constraint_tcon": None if constraint_tcon_override is None else _adjoint_trace_array(
                        constraint_tcon_override
                    ),
                    "constraint_precond_active": _adjoint_trace_array(constraint_precond_active),
                    "constraint_tcon_active": _adjoint_trace_array(constraint_tcon_active),
                    "lam_prec": np.asarray(lam_prec),
                    "precond_mats": mats,
                }
                if adjoint_trace_mode in {"full", "branch"}:
                    trace_entry.update(
                        {
                            "lam_prec": np.asarray(lam_prec),
                            "precond_mats": mats,
                        }
                    )
                if adjoint_trace_mode == "full":
                    trace_entry.update(
                        {
                            "frzl_frcc": np.asarray(frzl.frcc),
                            "frzl_frss": None if frzl.frss is None else np.asarray(frzl.frss),
                            "frzl_fzsc": np.asarray(frzl.fzsc),
                            "frzl_fzcs": None if frzl.fzcs is None else np.asarray(frzl.fzcs),
                            "frzl_flsc": np.asarray(frzl.flsc),
                            "frzl_flcs": None if frzl.flcs is None else np.asarray(frzl.flcs),
                            "frzl_frsc": None if getattr(frzl, "frsc", None) is None else np.asarray(frzl.frsc),
                            "frzl_frcs": None if getattr(frzl, "frcs", None) is None else np.asarray(frzl.frcs),
                            "frzl_fzcc": None if getattr(frzl, "fzcc", None) is None else np.asarray(frzl.fzcc),
                            "frzl_fzss": None if getattr(frzl, "fzss", None) is None else np.asarray(frzl.fzss),
                            "frzl_flcc": None if getattr(frzl, "flcc", None) is None else np.asarray(frzl.flcc),
                            "frzl_flss": None if getattr(frzl, "flss", None) is None else np.asarray(frzl.flss),
                            "frzl_rz_frcc": np.asarray(frzl_rz.frcc),
                            "frzl_rz_frss": None if frzl_rz.frss is None else np.asarray(frzl_rz.frss),
                            "frzl_rz_fzsc": np.asarray(frzl_rz.fzsc),
                            "frzl_rz_fzcs": None if frzl_rz.fzcs is None else np.asarray(frzl_rz.fzcs),
                            "frzl_rz_flsc": np.asarray(frzl_rz.flsc),
                            "frzl_rz_flcs": None if frzl_rz.flcs is None else np.asarray(frzl_rz.flcs),
                            "frzl_rz_frsc": None
                            if getattr(frzl_rz, "frsc", None) is None
                            else np.asarray(frzl_rz.frsc),
                            "frzl_rz_frcs": None
                            if getattr(frzl_rz, "frcs", None) is None
                            else np.asarray(frzl_rz.frcs),
                            "frzl_rz_fzcc": None
                            if getattr(frzl_rz, "fzcc", None) is None
                            else np.asarray(frzl_rz.fzcc),
                            "frzl_rz_fzss": None
                            if getattr(frzl_rz, "fzss", None) is None
                            else np.asarray(frzl_rz.fzss),
                            "frzl_rz_flcc": None
                            if getattr(frzl_rz, "flcc", None) is None
                            else np.asarray(frzl_rz.flcc),
                            "frzl_rz_flss": None
                            if getattr(frzl_rz, "flss", None) is None
                            else np.asarray(frzl_rz.flss),
                            "frcc_u": np.asarray(frcc_u),
                            "frss_u": np.asarray(frss_u),
                            "fzsc_u": np.asarray(fzsc_u),
                            "fzcs_u": np.asarray(fzcs_u),
                            "flsc_u": np.asarray(flsc_u),
                            "flcs_u": np.asarray(flcs_u),
                            "frsc_u": np.asarray(frsc_u),
                            "frcs_u": np.asarray(frcs_u),
                            "fzcc_u": np.asarray(fzcc_u),
                            "fzss_u": np.asarray(fzss_u),
                            "flcc_u": np.asarray(flcc_u),
                            "flss_u": np.asarray(flss_u),
                        }
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

                dR = dt_eff * _mn_cos_to_signed_physical(vRcc, vRss)
                dZ = dt_eff * _mn_sin_to_signed_physical(vZsc, vZcs)
                dL = dt_eff * _mn_sin_to_signed_physical_lambda(vLsc, vLcs)
                if bool(cfg.lasym):
                    dR_sin = dt_eff * _mn_sin_to_signed_physical(vRsc, vRcs)
                    dZ_cos = dt_eff * _mn_cos_to_signed_physical(vZcc, vZss)
                    dL_cos = dt_eff * _mn_cos_to_signed_physical_lambda(vLcc, vLss)
                else:
                    if host_update_assembly:
                        # Use pre-allocated zero arrays (avoid 3 np.zeros_like allocs/iter).
                        dR_sin = _zeros_dR_np
                        dZ_cos = _zeros_dR_np
                        dL_cos = _zeros_dR_np
                    else:
                        dR_sin = jnp.zeros_like(dR)
                        dZ_cos = jnp.zeros_like(dR)
                        dL_cos = jnp.zeros_like(dR)
                if host_update_assembly:
                    # All dR/dZ/dL/dR_sin/dZ_cos/dL_cos are NumPy here;
                    # keep state arrays as NumPy — JAX JIT converts at call site.
                    state_try = VMECState(
                        layout=state.layout,
                        Rcos=np.asarray(state.Rcos) + np.asarray(dR),
                        Rsin=np.asarray(state.Rsin) + np.asarray(dR_sin),
                        Zcos=np.asarray(state.Zcos) + np.asarray(dZ_cos),
                        Zsin=np.asarray(state.Zsin) + np.asarray(dZ),
                        Lcos=np.asarray(state.Lcos) + np.asarray(dL_cos),
                        Lsin=np.asarray(state.Lsin) + np.asarray(dL),
                    )
                else:
                    state_try = VMECState(
                        layout=state.layout,
                        Rcos=jnp.asarray(state.Rcos) + dR,
                        Rsin=jnp.asarray(state.Rsin) + dR_sin,
                        Zcos=jnp.asarray(state.Zcos) + dZ_cos,
                        Zsin=jnp.asarray(state.Zsin) + dZ,
                        Lcos=jnp.asarray(state.Lcos) + dL_cos,
                        Lsin=jnp.asarray(state.Lsin) + dL,
                    )
                if host_update_assembly:
                    state_try = _enforce_fixed_boundary_and_axis_np(
                        state_try,
                        static,
                        edge_Rcos=edge_Rcos,
                        edge_Rsin=edge_Rsin,
                        edge_Zcos=edge_Zcos,
                        edge_Zsin=edge_Zsin,
                        enforce_edge=not bool(free_boundary_enabled),
                        enforce_lambda_axis=True,
                        idx00=idx00,
                        precomputed_axis_mask=_precomputed_axis_mask_np,
                    )
                else:
                    state_try = _enforce_fixed_boundary_and_axis(
                        state_try,
                        static,
                        edge_Rcos=edge_Rcos,
                        edge_Rsin=edge_Rsin,
                        edge_Zcos=edge_Zcos,
                        edge_Zsin=edge_Zsin,
                        enforce_edge=not bool(free_boundary_enabled),
                        enforce_lambda_axis=True,
                        idx00=idx00,
                    )
                state_try = _apply_vmec_lambda_axis_rules(state_try)
            probe_bad_jacobian = False
            if need_trial_eval:
                freeb_bsqvac_half_trial = _freeb_bsqvac_half_for_trial_state(state_try)
                t_trial_force_start = time.perf_counter() if timing_detail_enabled else None
                _, _, gcr2_t, gcz2_t, gcl2_t, _, _, norms_t = _compute_forces_iter(
                    state_try,
                    include_edge=include_edge,
                    zero_m1=zero_m1,
                    freeb_bsqvac_half=freeb_bsqvac_half_trial,
                    constraint_precond_diag=constraint_precond_diag,
                    constraint_tcon=constraint_tcon_override,
                    constraint_precond_active=constraint_precond_active,
                    constraint_tcon_active=constraint_tcon_active,
                    iter2=iter2,
                )
                _record_compute_force_timing("trial", t_trial_force_start, gcr2_t)
                fsqr_t, fsqz_t, fsql_t = _residual_fsq_from_norms(
                    norms_t,
                    gcr2=gcr2_t,
                    gcz2=gcz2_t,
                    gcl2=gcl2_t,
                )
                w_try = float(np.asarray(fsqr_t + fsqz_t + fsql_t))
                w_try_ratio = w_try / max(w_curr, 1e-30) if np.isfinite(w_try) else float("inf")
                if bool(reference_mode) and (float(np.asarray(zero_m1)) > 0.5):
                    _, _, gcr2_probe, gcz2_probe, gcl2_probe, _, _, norms_probe = _compute_forces_iter(
                        state_try,
                        include_edge=include_edge,
                        zero_m1=jnp.asarray(0.0, dtype=zero_m1.dtype),
                        freeb_bsqvac_half=freeb_bsqvac_half_trial,
                        constraint_precond_diag=constraint_precond_diag,
                        constraint_tcon=constraint_tcon_override,
                        constraint_precond_active=constraint_precond_active,
                        constraint_tcon_active=constraint_tcon_active,
                        iter2=iter2,
                    )
                    fsqr_probe, fsqz_probe, fsql_probe = _residual_fsq_from_norms(
                        norms_probe,
                        gcr2=gcr2_probe,
                        gcz2=gcz2_probe,
                        gcl2=gcl2_probe,
                    )
                    w_probe = float(np.asarray(fsqr_probe + fsqz_probe + fsql_probe))
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
                    state_try = VMECState(
                        layout=state.layout,
                        Rcos=jnp.asarray(state.Rcos) + alpha * dR,
                        Rsin=jnp.asarray(state.Rsin) + alpha * dR_sin,
                        Zcos=jnp.asarray(state.Zcos) + alpha * dZ_cos,
                        Zsin=jnp.asarray(state.Zsin) + alpha * dZ,
                        Lcos=jnp.asarray(state.Lcos) + alpha * dL_cos,
                        Lsin=jnp.asarray(state.Lsin) + alpha * dL,
                    )
                    if host_update_assembly:
                        state_try = _enforce_fixed_boundary_and_axis_np(
                            state_try,
                            static,
                            edge_Rcos=edge_Rcos,
                            edge_Rsin=edge_Rsin,
                            edge_Zcos=edge_Zcos,
                            edge_Zsin=edge_Zsin,
                            enforce_edge=not bool(free_boundary_enabled),
                            enforce_lambda_axis=True,
                            idx00=idx00,
                            precomputed_axis_mask=_precomputed_axis_mask_np,
                        )
                    else:
                        state_try = _enforce_fixed_boundary_and_axis(
                            state_try,
                            static,
                            edge_Rcos=edge_Rcos,
                            edge_Rsin=edge_Rsin,
                            edge_Zcos=edge_Zcos,
                            edge_Zsin=edge_Zsin,
                            enforce_edge=not bool(free_boundary_enabled),
                            enforce_lambda_axis=True,
                            idx00=idx00,
                    )
                    state_try = _apply_vmec_lambda_axis_rules(state_try)
                    freeb_bsqvac_half_trial = _freeb_bsqvac_half_for_trial_state(state_try)
                    t_trial_force_start = time.perf_counter() if timing_detail_enabled else None
                    _, _, gcr2_t, gcz2_t, gcl2_t, _, _, norms_t = _compute_forces_iter(
                        state_try,
                        include_edge=include_edge,
                        zero_m1=zero_m1,
                        freeb_bsqvac_half=freeb_bsqvac_half_trial,
                        constraint_precond_diag=constraint_precond_diag,
                        constraint_tcon=constraint_tcon_override,
                        constraint_precond_active=constraint_precond_active,
                        constraint_tcon_active=constraint_tcon_active,
                        iter2=iter2,
                    )
                    _record_compute_force_timing("trial", t_trial_force_start, gcr2_t)
                    fsqr_t, fsqz_t, fsql_t = _residual_fsq_from_norms(
                        norms_t,
                        gcr2=gcr2_t,
                        gcz2=gcz2_t,
                        gcl2=gcl2_t,
                    )
                    w_try = float(np.asarray(fsqr_t + fsqz_t + fsql_t))
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
                    dR_dir = dt_direct * _mn_cos_to_signed(flip_sign * frcc_u, flip_sign * frss_u)
                    dZ_dir = dt_direct * _mn_sin_to_signed(flip_sign * fzsc_u, flip_sign * fzcs_u)
                    dL_dir = dt_direct * _mn_sin_to_signed(flip_sign * flsc_u, flip_sign * flcs_u)
                    if bool(cfg.lasym):
                        dR_sin_dir = dt_direct * _mn_sin_to_signed(flip_sign * frsc_u, flip_sign * frcs_u)
                        dZ_cos_dir = dt_direct * _mn_cos_to_signed(flip_sign * fzcc_u, flip_sign * fzss_u)
                        dL_cos_dir = dt_direct * _mn_cos_to_signed(flip_sign * flcc_u, flip_sign * flss_u)
                    else:
                        dR_sin_dir = jnp.zeros_like(dR_dir)
                        dZ_cos_dir = jnp.zeros_like(dR_dir)
                        dL_cos_dir = jnp.zeros_like(dR_dir)
                    state_dir = VMECState(
                        layout=state.layout,
                        Rcos=jnp.asarray(state.Rcos) + dR_dir,
                        Rsin=jnp.asarray(state.Rsin) + dR_sin_dir,
                        Zcos=jnp.asarray(state.Zcos) + dZ_cos_dir,
                        Zsin=jnp.asarray(state.Zsin) + dZ_dir,
                        Lcos=jnp.asarray(state.Lcos) + dL_cos_dir,
                        Lsin=jnp.asarray(state.Lsin) + dL_dir,
                    )
                    state_dir = _enforce_fixed_boundary_and_axis(
                        state_dir,
                        static,
                        edge_Rcos=edge_Rcos,
                        edge_Rsin=edge_Rsin,
                        edge_Zcos=edge_Zcos,
                        edge_Zsin=edge_Zsin,
                        enforce_edge=not bool(free_boundary_enabled),
                        enforce_lambda_axis=True,
                        idx00=idx00,
                    )
                    state_dir = _apply_vmec_lambda_axis_rules(state_dir)
                    freeb_bsqvac_half_dir = _freeb_bsqvac_half_for_trial_state(state_dir)
                    _, _, gcr2_d, gcz2_d, gcl2_d, _, _, norms_d = _compute_forces_iter(
                        state_dir,
                        include_edge=include_edge,
                        zero_m1=zero_m1,
                        freeb_bsqvac_half=freeb_bsqvac_half_dir,
                        constraint_precond_diag=constraint_precond_diag,
                        constraint_tcon=constraint_tcon_override,
                        constraint_precond_active=constraint_precond_active,
                        constraint_tcon_active=constraint_tcon_active,
                        iter2=iter2,
                    )
                    fsqr_d, fsqz_d, fsql_d = _residual_fsq_from_norms(
                        norms_d,
                        gcr2=gcr2_d,
                        gcz2=gcz2_d,
                        gcl2=gcl2_d,
                    )
                    w_dir = float(np.asarray(fsqr_d + fsqz_d + fsql_d))
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
                trace_entry.update(
                    {
                        "step_status": str(step_status),
                        "restart_reason": str(restart_reason),
                        "restart_path": str(restart_path),
                        "time_step": float(time_step),
                        "flip_sign": float(flip_sign),
                        "limit_update_rms": bool(limit_update_rms),
                    }
                )
                if adjoint_trace_mode in {"full", "branch"}:
                    trace_entry.update(
                        {
                            "dt_eff": float(dt_eff),
                            "b1": float(b1),
                            "fac": float(fac),
                            "force_scale": float(force_scale),
                            "state_post": state,
                        }
                    )
                if adjoint_trace_mode == "full":
                    trace_entry.update(
                        {
                            "w_curr": float(w_curr),
                            "w_try": float(w_try),
                            "w_try_ratio": float(w_try_ratio),
                            "update_rms_preclip": None if update_rms_preclip is None else float(update_rms_preclip),
                            "update_rms_postclip": None if update_rms is None else float(update_rms),
                            "update_rms_scale": float(scl),
                            "vRcc_after": np.asarray(vRcc),
                            "vRss_after": np.asarray(vRss),
                            "vZsc_after": np.asarray(vZsc),
                            "vZcs_after": np.asarray(vZcs),
                            "vLsc_after": np.asarray(vLsc),
                            "vLcs_after": np.asarray(vLcs),
                            "vRsc_after": np.asarray(vRsc),
                            "vRcs_after": np.asarray(vRcs),
                            "vZcc_after": np.asarray(vZcc),
                            "vZss_after": np.asarray(vZss),
                            "vLcc_after": np.asarray(vLcc),
                            "vLss_after": np.asarray(vLss),
                        }
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
            accepted = False
            step_status = "rejected"
            step_factor = 1.0
            vRcc_best, vRss_best = vRcc, vRss
            vZsc_best, vZcs_best = vZsc, vZcs
            vLsc_best, vLcs_best = vLsc, vLcs
            vRsc_best, vRcs_best = vRsc, vRcs
            vZcc_best, vZss_best = vZcc, vZss
            vLcc_best, vLss_best = vLcc, vLss
            state_best = state
            dt_eff = float(time_step)
            update_rms = 0.0
            w_curr = fsqr_f + fsqz_f + fsql_f

            for _bt in range(6):
                dt_try = time_step * step_factor
                vRcc_try = fac * (b1 * vRcc + dt_try * (flip_sign * jnp.asarray(frcc_u)))
                vRss_try = fac * (b1 * vRss + dt_try * (flip_sign * jnp.asarray(frss_u)))
                vRsc_try = fac * (b1 * vRsc + dt_try * (flip_sign * jnp.asarray(frsc_u)))
                vRcs_try = fac * (b1 * vRcs + dt_try * (flip_sign * jnp.asarray(frcs_u)))
                vZsc_try = fac * (b1 * vZsc + dt_try * (flip_sign * jnp.asarray(fzsc_u)))
                vZcs_try = fac * (b1 * vZcs + dt_try * (flip_sign * jnp.asarray(fzcs_u)))
                vZcc_try = fac * (b1 * vZcc + dt_try * (flip_sign * jnp.asarray(fzcc_u)))
                vZss_try = fac * (b1 * vZss + dt_try * (flip_sign * jnp.asarray(fzss_u)))
                vLsc_try = fac * (b1 * vLsc + dt_try * (flip_sign * jnp.asarray(flsc_u)))
                vLcs_try = fac * (b1 * vLcs + dt_try * (flip_sign * jnp.asarray(flcs_u)))
                vLcc_try = fac * (b1 * vLcc + dt_try * (flip_sign * jnp.asarray(flcc_u)))
                vLss_try = fac * (b1 * vLss + dt_try * (flip_sign * jnp.asarray(flss_u)))

                dR_try = dt_try * _mn_cos_to_signed(vRcc_try, vRss_try)
                dZ_try = dt_try * _mn_sin_to_signed(vZsc_try, vZcs_try)
                dL_try = dt_try * _mn_sin_to_signed(vLsc_try, vLcs_try)
                if bool(cfg.lasym):
                    dR_sin_try = dt_try * _mn_sin_to_signed(vRsc_try, vRcs_try)
                    dZ_cos_try = dt_try * _mn_cos_to_signed(vZcc_try, vZss_try)
                    dL_cos_try = dt_try * _mn_cos_to_signed(vLcc_try, vLss_try)
                else:
                    dR_sin_try = jnp.zeros_like(dR_try)
                    dZ_cos_try = jnp.zeros_like(dR_try)
                    dL_cos_try = jnp.zeros_like(dR_try)

                state_try = VMECState(
                    layout=state.layout,
                    Rcos=jnp.asarray(state.Rcos) + dR_try,
                    Rsin=jnp.asarray(state.Rsin) + dR_sin_try,
                    Zcos=jnp.asarray(state.Zcos) + dZ_cos_try,
                    Zsin=jnp.asarray(state.Zsin) + dZ_try,
                    Lcos=jnp.asarray(state.Lcos) + dL_cos_try,
                    Lsin=jnp.asarray(state.Lsin) + dL_try,
                )
                state_try = _enforce_fixed_boundary_and_axis(
                    state_try,
                    static,
                    edge_Rcos=edge_Rcos,
                    edge_Rsin=edge_Rsin,
                    edge_Zcos=edge_Zcos,
                    edge_Zsin=edge_Zsin,
                    enforce_edge=not bool(free_boundary_enabled),
                    enforce_lambda_axis=True,
                    idx00=idx00,
                )
                state_try = _apply_vmec_lambda_axis_rules(state_try)
                freeb_bsqvac_half_trial = _freeb_bsqvac_half_for_trial_state(state_try)
                t_backtracking_force_start = time.perf_counter() if timing_detail_enabled else None
                _, _, gcr2_t, gcz2_t, gcl2_t, _, _, norms_t = _compute_forces_iter(
                    state_try,
                    include_edge=include_edge,
                    zero_m1=zero_m1,
                    freeb_bsqvac_half=freeb_bsqvac_half_trial,
                    constraint_precond_diag=constraint_precond_diag,
                    constraint_tcon=constraint_tcon_override,
                    constraint_precond_active=constraint_precond_active,
                    constraint_tcon_active=constraint_tcon_active,
                    iter2=iter2,
                )
                _record_compute_force_timing("backtracking", t_backtracking_force_start, gcr2_t)
                fsqr_t, fsqz_t, fsql_t = _residual_fsq_from_norms(
                    norms_t,
                    gcr2=gcr2_t,
                    gcz2=gcz2_t,
                    gcl2=gcl2_t,
                )
                w_try = float(np.asarray(fsqr_t + fsqz_t + fsql_t))
                if np.isfinite(w_try) and (w_try <= 1.05 * w_curr):
                    accepted = True
                    step_status = "momentum"
                    state_best = state_try
                    vRcc_best, vRss_best = vRcc_try, vRss_try
                    vZsc_best, vZcs_best = vZsc_try, vZcs_try
                    vLsc_best, vLcs_best = vLsc_try, vLcs_try
                    vRsc_best, vRcs_best = vRsc_try, vRcs_try
                    vZcc_best, vZss_best = vZcc_try, vZss_try
                    vLcc_best, vLss_best = vLcc_try, vLss_try
                    dt_eff = float(dt_try)
                    update_rms = _host_force_update_rms(
                        dt_try,
                        vRcc_try,
                        vRss_try,
                        vRsc_try,
                        vRcs_try,
                        vZsc_try,
                        vZcs_try,
                        vZcc_try,
                        vZss_try,
                        vLsc_try,
                        vLcs_try,
                        vLcc_try,
                        vLss_try,
                    )
                    break
                step_factor *= 0.5

            state = state_best
            vRcc, vRss = vRcc_best, vRss_best
            vZsc, vZcs = vZsc_best, vZcs_best
            vLsc, vLcs = vLsc_best, vLcs_best
            vRsc, vRcs = vRsc_best, vRcs_best
            vZcc, vZss = vZcc_best, vZss_best
            vLcc, vLss = vLcc_best, vLss_best
            if not accepted:
                # No acceptable update was found; damp velocity to avoid runaway.
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
                ) = _scale_velocity_blocks(0.5, vRcc, vRss, vRsc, vRcs, vZsc, vZcs, vZcc, vZss, vLsc, vLcs, vLcc, vLss)
                dt_eff = float(step_size * step_factor)
                update_rms = 0.0
                step_status = "rejected"
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
        if verbose:
            if bool(vmec2000_control) and bool(verbose_vmec2000_table):
                if _should_print_vmec2000(int(iter2), int(max_iter)):
                    fsqr1_f, fsqz1_f, fsql1_f = _precond_diag_floats()
                    _print_vmec2000_iter_row(
                        iter_idx=int(iter2),
                        fsqr=fsqr_f,
                        fsqz=fsqz_f,
                        fsql=fsql_f,
                        fsqr1=fsqr1_f,
                        fsqz1=fsqz1_f,
                        fsql1=fsql1_f,
                        delt0r=float(time_step),
                        r00=float(r00_last),
                        w_mhd=float(w_vmec_last),
                        z00=float(z00_last),
                    )
            else:
                fsqr1_f, fsqz1_f, fsql1_f = _precond_diag_floats()
                update_rms_print = _update_rms_float() if bool(strict_update) else float(update_rms)
                print(
                    f"[solve_fixed_boundary_residual_iter] iter={it:03d} "
                    f"dt_eff={dt_eff:.3e} update_rms={update_rms_print:.3e} "
                    f"fsqr1={fsqr1_f:.3e} fsqz1={fsqz1_f:.3e} fsql1={fsql1_f:.3e} "
                    f"step_status={step_status}",
                    flush=True,
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

    t_finalize_start = time.perf_counter() if timing_enabled else None
    final_fsqr_report = float(fsqr_f)
    final_fsqz_report = float(fsqz_f)
    final_fsql_report = float(fsql_f)
    final_residual_recomputed = False
    final_pre_update_fsqr = float(fsqr_f)
    final_pre_update_fsqz = float(fsqz_f)
    final_pre_update_fsql = float(fsql_f)
    final_nestor_model = str(freeb_last_model)
    final_nestor_diagnostics = dict(freeb_last_diagnostics)
    final_vacuum_stub = not bool(str(final_nestor_model).strip() and str(final_nestor_model) != "none")
    final_bsqvac_half_current = freeb_bsqvac_half_current
    final_nestor_recompute_attempted = False
    final_nestor_recompute_failed = False
    final_nestor_sample_time_s = 0.0
    final_nestor_solve_time_s = 0.0
    if bool(free_boundary_enabled and freeb_couple_edge) and not final_vacuum_stub:
        final_nestor_recompute_attempted = True
        t_finalize_nestor_recompute_start = time.perf_counter() if timing_enabled else None
        try:
            nestor_final, _freeb_nestor_runtime_final = nestor_external_only_step(
                state=state,
                static=static,
                ivac=1,
                ivacskip=0,
                iter_idx=None,
                runtime=freeb_nestor_runtime,
                extcur=tuple(getattr(static, "free_boundary_extcur", ()) or ()),
                plascur=float(freeb_plascur),
                external_field_provider_kind=external_field_provider_kind,
                external_field_provider_static=external_field_provider_static,
                external_field_provider_params=external_field_provider_params,
            )
            final_nestor_sample_time_s = float(getattr(nestor_final, "sample_time_s", 0.0))
            final_nestor_solve_time_s = float(getattr(nestor_final, "solve_time_s", 0.0))
            final_nestor_model = str(getattr(nestor_final, "model", final_nestor_model))
            diag_final = getattr(nestor_final, "diagnostics", None)
            if isinstance(diag_final, dict):
                final_nestor_diagnostics = dict(diag_final)
            bsqvac_edge_final = np.asarray(nestor_final.vac_total.bsqvac, dtype=float)
            if (
                bsqvac_edge_final.ndim == 2
                and int(bsqvac_edge_final.shape[1]) == 1
                and int(getattr(static.cfg, "nzeta", 1)) > 1
            ):
                bsqvac_edge_final = np.repeat(bsqvac_edge_final, int(static.cfg.nzeta), axis=1)
            final_bsqvac_half_current = bsqvac_edge_final
            final_vacuum_stub = False
        except Exception:
            final_nestor_recompute_failed = True
            final_bsqvac_half_current = freeb_bsqvac_half_current
        finally:
            if timing_enabled and t_finalize_nestor_recompute_start is not None:
                timing_stats["finalize_nestor_recompute"] += (
                    time.perf_counter() - float(t_finalize_nestor_recompute_start)
                )
    if bool(free_boundary_enabled) and final_bsqvac_half_current is not None:
        t_finalize_residual_recompute_start = time.perf_counter() if timing_enabled else None
        try:
            _, _, gcr2_final, gcz2_final, gcl2_final, _, _, norms_final = _compute_forces_iter(
                state,
                include_edge=bool(include_edge),
                include_edge_residual=True,
                zero_m1=zero_m1,
                freeb_bsqvac_half=final_bsqvac_half_current,
                constraint_precond_diag=constraint_precond_diag,
                constraint_tcon=constraint_tcon_override,
                constraint_precond_active=constraint_precond_active,
                constraint_tcon_active=constraint_tcon_active,
                iter2=last_iter2,
            )
            fsqr_final, fsqz_final, fsql_final = _residual_fsq_from_norms(
                norms_final,
                gcr2=gcr2_final,
                gcz2=gcz2_final,
                gcl2=gcl2_final,
            )
            t_finalize_residual_get_start = time.perf_counter() if timing_enabled else None
            final_fsqr_report, final_fsqz_report, final_fsql_report = _device_get_floats(
                fsqr_final,
                fsqz_final,
                fsql_final,
            )
            if timing_enabled and t_finalize_residual_get_start is not None:
                timing_stats["finalize_residual_device_get"] += (
                    time.perf_counter() - float(t_finalize_residual_get_start)
                )
            final_residual_recomputed = True
        except Exception:
            final_fsqr_report = float(fsqr_f)
            final_fsqz_report = float(fsqz_f)
            final_fsql_report = float(fsql_f)
        finally:
            if timing_enabled and t_finalize_residual_recompute_start is not None:
                timing_stats["finalize_residual_recompute"] += (
                    time.perf_counter() - float(t_finalize_residual_recompute_start)
                )
    converged_strict_final, converged_total_final, _ = _residual_convergence_flags(
        fsqr=final_fsqr_report,
        fsqz=final_fsqz_report,
        fsql=final_fsql_report,
        ftol=ftol,
        fsq_total_target=fsq_total_target,
    )
    t_finalize_diag_build_start = time.perf_counter() if timing_enabled else None
    diag: Dict[str, Any] = {
        "ftol": ftol,
        "requested_ftol": float(ftol),
        "gamma": gamma,
        "step_size": float(step_size),
        "precond_radial_alpha": float(precond_radial_alpha),
        "precond_lambda_alpha": float(precond_lambda_alpha),
        "strict_update": bool(strict_update),
        "reference_mode": bool(reference_mode),
        "use_restart_triggers": bool(use_restart_triggers),
        "use_direct_fallback": bool(use_direct_fallback),
        "max_update_rms": float(max_update_rms),
        "converged": bool(converged),
        "converged_strict": bool(converged_strict_final),
        "converged_by_total_fsq": bool(converged_total_final),
        "final_fsqr": float(final_fsqr_report),
        "final_fsqz": float(final_fsqz_report),
        "final_fsql": float(final_fsql_report),
        "pre_update_final_fsqr": float(final_pre_update_fsqr),
        "pre_update_final_fsqz": float(final_pre_update_fsqz),
        "pre_update_final_fsql": float(final_pre_update_fsql),
        "final_residual_recomputed_on_accepted_state": bool(final_residual_recomputed),
        "badjac_use_state": bool(badjac_use_state),
        "badjac_mode": badjac_mode,
        "badjac_state_probe": bool(badjac_state_probe),
        "badjac_initial_state_probe_iters": int(badjac_initial_state_probe_iters),
        "light_history": bool(light_history),
        "resume_state_mode": str(resume_state_mode),
        "fsq_total_target": fsq_total_target,
        "ijacob": int(ijacob),
        "bad_resets": int(bad_resets),
        "iter1_final": int(iter1),
        "res0": float(res0),
        "step_status_history": np.asarray(step_status_history, dtype=object),
        "restart_reason_history": np.asarray(restart_reason_history, dtype=object),
        "pre_restart_reason_history": np.asarray(pre_restart_reason_history, dtype=object),
        "time_step_history": np.asarray(time_step_history, dtype=float),
        "res0_history": np.asarray(res0_history, dtype=float),
        "res1_history": np.asarray(res1_history, dtype=float),
        "fsq_prev_history": np.asarray(fsq_prev_history, dtype=float),
        "bad_growth_streak_history": np.asarray(bad_growth_streak_history, dtype=int),
        "iter1_history": np.asarray(iter1_history, dtype=int),
        "iter2_history": np.asarray(iter2_history, dtype=int),
        "bcovar_update_history": np.asarray(bcovar_update_history, dtype=int),
        "include_edge_history": np.asarray(include_edge_history, dtype=int),
        "zero_m1_history": np.asarray(zero_m1_history, dtype=int),
        "dt_eff_history": np.asarray(dt_eff_history, dtype=float),
        "update_rms_history": _scalar_history_array(update_rms_history),
        "w_curr_history": np.asarray(w_curr_history, dtype=float),
        "w_try_history": np.asarray(w_try_history, dtype=float),
        "w_try_ratio_history": np.asarray(w_try_ratio_history, dtype=float),
        "restart_path_history": np.asarray(restart_path_history, dtype=object),
        "adjoint_step_trace": adjoint_step_trace_history,
        "min_tau_history": np.asarray(min_tau_history, dtype=float),
        "max_tau_history": np.asarray(max_tau_history, dtype=float),
        "bad_jacobian_history": np.asarray(bad_jacobian_history, dtype=int),
        "r00_history": np.asarray(r00_history, dtype=float),
        "z00_history": np.asarray(z00_history, dtype=float),
        "wb_history": np.asarray(wb_history, dtype=float),
        "wp_history": np.asarray(wp_history, dtype=float),
        "w_vmec_history": np.asarray(w_vmec_history, dtype=float),
        "fsq1_history": _scalar_history_array(fsq1_history),
        "fsqr1_history": _scalar_history_array(fsqr1_history),
        "fsqz1_history": _scalar_history_array(fsqz1_history),
        "fsql1_history": _scalar_history_array(fsql1_history),
        "rz_norm_history": _scalar_history_array(rz_norm_history),
        "f_norm1_history": _scalar_history_array(f_norm1_history),
        "gcr2_p_history": _scalar_history_array(gcr2_p_history),
        "gcz2_p_history": _scalar_history_array(gcz2_p_history),
        "gcl2_p_history": _scalar_history_array(gcl2_p_history),
        "free_boundary": {
            "enabled": bool(free_boundary_enabled),
            "nvacskip": int(freeb_nvacskip),
            "nvskip0": int(freeb_nvskip0),
            "ivac": int(freeb_ivac),
            "ivacskip": int(freeb_ivacskip),
            "couple_edge": bool(freeb_couple_edge),
            "nestor_model": str(final_nestor_model),
            "vacuum_stub": bool(final_vacuum_stub),
            "activate_fsq": None if free_boundary_activate_fsq is None else float(free_boundary_activate_fsq),
            "plascur": float(freeb_plascur),
            "last_nestor_diagnostics": dict(final_nestor_diagnostics),
            "final_nestor_recompute_attempted": bool(final_nestor_recompute_attempted),
            "final_nestor_recompute_failed": bool(final_nestor_recompute_failed),
            "final_nestor_sample_time_s": float(final_nestor_sample_time_s),
            "final_nestor_solve_time_s": float(final_nestor_solve_time_s),
        },
        "freeb_ivac_history": np.asarray(freeb_ivac_history, dtype=int),
        "freeb_ivacskip_history": np.asarray(freeb_ivacskip_history, dtype=int),
        "freeb_full_update_history": np.asarray(freeb_full_update_history, dtype=int),
        "freeb_nestor_reused_history": np.asarray(freeb_nestor_reused_history, dtype=int),
        "freeb_nestor_source_reused_history": np.asarray(freeb_nestor_source_reused_history, dtype=int),
        "freeb_nestor_provider_allows_source_reuse_history": np.asarray(
            freeb_nestor_provider_allows_source_reuse_history, dtype=int
        ),
        "freeb_nestor_bnormal_rms_history": np.asarray(freeb_nestor_bnormal_rms_history, dtype=float),
        "freeb_nestor_gsource_rms_history": np.asarray(freeb_nestor_gsource_rms_history, dtype=float),
        "freeb_nestor_bsqvac_rms_history": np.asarray(freeb_nestor_bsqvac_rms_history, dtype=float),
        "freeb_nestor_solve_time_history": np.asarray(freeb_nestor_solve_time_history, dtype=float),
        "freeb_nestor_sample_time_history": np.asarray(freeb_nestor_sample_time_history, dtype=float),
        "freeb_nestor_trial_reused_history": np.asarray(freeb_nestor_trial_reused_history, dtype=int),
        "freeb_nestor_trial_solve_time_history": np.asarray(freeb_nestor_trial_solve_time_history, dtype=float),
        "freeb_nestor_trial_sample_time_history": np.asarray(freeb_nestor_trial_sample_time_history, dtype=float),
        "freeb_nestor_trial_failed_history": np.asarray(freeb_nestor_trial_failed_history, dtype=int),
    }
    diag = _attach_residual_iter_timing_diagnostics(
        diag,
        timing_stats,
        timing_enabled=bool(timing_enabled),
        timing_detail_enabled=bool(timing_detail_enabled),
        finalize_diag_build_start=t_finalize_diag_build_start,
        iteration_loop_start=t_iteration_loop_start,
        finalize_start=t_finalize_start,
        solve_wall_start=float(_solve_wall_start),
    )
    resume_state_base_kwargs = {
        "time_step": time_step,
        "inv_tau": inv_tau,
        "fsq_prev": fsq_prev,
        "fsq0_prev": fsq0_prev,
        "flip_sign": flip_sign,
        "iter1": iter1,
        "last_iter2": last_iter2,
        "ijacob": ijacob,
        "bad_resets": bad_resets,
        "res0": res0,
        "res1": res1,
        "prev_rz_fsq": prev_rz_fsq,
        "bad_growth_streak": bad_growth_streak,
        "huge_force_restart_count": huge_force_restart_count,
        "vmec2000_cache_valid": vmec2000_cache_valid,
        "freeb_ivac": freeb_ivac,
        "freeb_ivacskip": freeb_ivacskip,
        "freeb_nvacskip": freeb_nvacskip,
        "freeb_nvskip0": freeb_nvskip0,
        "freeb_last_model": freeb_last_model,
        "freeb_nestor_runtime": freeb_nestor_runtime,
    }
    resume_state_heavy = None
    if resume_state_mode == "full":
        resume_state_heavy = {
            "vRcc": np.asarray(vRcc),
            "vRss": np.asarray(vRss),
            "vZsc": np.asarray(vZsc),
            "vZcs": np.asarray(vZcs),
            "vLsc": np.asarray(vLsc),
            "vLcs": np.asarray(vLcs),
            "vRsc": np.asarray(vRsc),
            "vRcs": np.asarray(vRcs),
            "vZcc": np.asarray(vZcc),
            "vZss": np.asarray(vZss),
            "vLcc": np.asarray(vLcc),
            "vLss": np.asarray(vLss),
            "state_checkpoint": state_checkpoint,
            "cache_precond_diag": cache_precond_diag,
            "cache_tcon": cache_tcon,
            "cache_norms": cache_norms,
            "cache_rz_scale": cache_rz_scale,
            "cache_l_scale": cache_l_scale,
            "cache_rz_norm": cache_rz_norm,
            "cache_f_norm1": cache_f_norm1,
            "cache_prec_rz_mats": cache_prec_rz_mats,
            "cache_prec_rz_jmax": cache_prec_rz_jmax,
            "cache_prec_lam_prec": cache_prec_lam_prec,
            "cache_prec_faclam": cache_prec_faclam,
            "cache_prec_lam_debug": cache_prec_lam_debug,
            "cache_constraint_rcon0": cache_constraint_rcon0,
            "cache_constraint_zcon0": cache_constraint_zcon0,
        }
    diag["resume_state"] = _build_residual_iter_resume_state_payload(
        resume_state_mode=str(resume_state_mode),
        base_kwargs=resume_state_base_kwargs,
        heavy_payload=resume_state_heavy,
    )
    return _finalize_residual_iter_result(
        result_type=SolveVmecResidualResult,
        state=state,
        w_history=w_history,
        fsqr2_history=fsqr2_history,
        fsqz2_history=fsqz2_history,
        fsql2_history=fsql2_history,
        grad_rms_history=grad_rms_history,
        step_history=step_history,
        diagnostics=diag,
        attach_free_boundary_diagnostics=_attach_freeb_diag,
        return_final_force_payload=bool(return_final_force_payload),
        converged=bool(converged),
        final_force_payload=k,
    )
