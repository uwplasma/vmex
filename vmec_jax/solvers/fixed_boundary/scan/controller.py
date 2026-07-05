"""VMEC2000-style residual scan controller for fixed-boundary solves."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import os
import time
from typing import Any, NamedTuple

import numpy as np

from vmec_jax._compat import has_jax, jax, jnp, jit
from vmec_jax._solve_runtime import _dataclass_from_namespace
from vmec_jax.field import TWOPI
from vmec_jax.solvers.fixed_boundary.diagnostics.axis_reset import evaluate_initial_axis_reset as _evaluate_initial_axis_reset
from vmec_jax.solvers.fixed_boundary.diagnostics.io import _pack_resume_state_record
from vmec_jax.solvers.fixed_boundary.jit_cache import (
    jit_cache_get as _jit_cache_get,
    jit_cache_put as _jit_cache_put,
    record_scan_runner_cache_miss_categories as _record_scan_runner_cache_miss_categories,
)
from vmec_jax.solvers.fixed_boundary.preconditioning.operators import (
    scale_m1_precond_rhs_from_mats as _scale_m1_precond_rhs_from_mats,
)
from vmec_jax.solvers.fixed_boundary.residual.finalize import (
    vmec2000_state_only_scan_result as _vmec2000_state_only_scan_result,
    vmec2000_traced_scan_result as _vmec2000_traced_scan_result,
)
from vmec_jax.solvers.fixed_boundary.residual.runtime import (
    _converged_residuals_scan_fast as _runtime_converged_residuals_scan_fast,
)
from vmec_jax.solvers.fixed_boundary.optimization.constraints import enforce_fixed_boundary_and_axis as _enforce_fixed_boundary_and_axis
from vmec_jax.solvers.fixed_boundary.results import ScanCarry as _ScanCarry, SolveVmecResidualResult
from vmec_jax.solvers.fixed_boundary.scan.debug import (
    dump_vmec2000_scan_ptau_rows as _dump_vmec2000_scan_ptau_rows,
    emit_live_scan_vmec2000_row as _emit_live_scan_vmec2000_row,
    emit_vmec2000_post_scan_rows as _emit_vmec2000_post_scan_rows,
    maybe_debug_scan_force_first_iter as _maybe_debug_scan_force_first_iter,
    maybe_debug_scan_state_iter as _maybe_debug_scan_state_iter,
    _emit_scan_prints as _emit_scan_debug_prints,
    _print_vmec2000_row as _print_scan_vmec2000_row,
    _record_scan_device_ready,
)
from vmec_jax.solvers.fixed_boundary.scan.math import (
    _hold_step as _scan_math_hold_step,
    _no_restart_updates as _scan_math_no_restart_updates,
    _restart_updates as _scan_math_restart_updates,
    sample_vmec2000_scan_scalars as _sample_vmec2000_scan_scalars,
    scan_bad_jacobian_decision_from_step as _scan_bad_jacobian_decision_from_step,
)
from vmec_jax.solvers.fixed_boundary.scan.output import (
    finalize_vmec2000_scan_run,
    finalize_vmec2000_scan_step,
    vmec2000_scan_light_history_row,
    vmec2000_scan_minimal_history_row,
)
from vmec_jax.solvers.fixed_boundary.scan.payload import (
    build_current_preconditioned_scan_payload as _build_current_preconditioned_scan_payload,
    build_initial_preconditioner_cache as _build_initial_preconditioner_cache,
    evaluate_scan_step_force as _evaluate_scan_step_force,
    select_payload_and_build_step_fields as _select_payload_and_build_step_fields,
)
from vmec_jax.solvers.fixed_boundary.scan.planning import (
    build_scan_timing_report as _build_scan_timing_report,
    default_vmec2000_controller_constants as _default_vmec2000_controller_constants,
    new_scan_timing_stats as _new_scan_timing_stats,
    resolve_scan_iteration_runtime_plan as _resolve_scan_iteration_runtime_plan,
    resolve_vmec2000_scan_setup as _resolve_vmec2000_scan_setup,
    scan_jit_forces_enabled as _scan_jit_forces_enabled,
    scan_jit_preflight_enabled as _scan_jit_preflight_enabled,
    scan_timing_enabled as _scan_timing_enabled,
    validate_vmec2000_scan_guards as _validate_vmec2000_scan_guards,
)
from vmec_jax.solvers.fixed_boundary.scan.resume import (
    build_initial_scan_carry as _build_initial_scan_carry,
    build_traced_scan_resume_state as _build_traced_scan_resume_state,
    initialize_scan_resume_state as _initialize_scan_resume_state,
)
from vmec_jax.solvers.fixed_boundary.scan.runtime import (
    get_or_build_scan_runner as _get_or_build_scan_runner,
    resolve_scan_runtime_hooks_from_env as _resolve_scan_runtime_hooks_from_env,
    run_vmec2000_scan_dispatch as _run_vmec2000_scan_dispatch,
    scan_trace_context_or_null as _scan_trace_context_or_null,
)
from vmec_jax.solvers.fixed_boundary.residual.scan_adapters import (
    ScanConvergenceControls,
    build_vmec2000_scan_runtime_setup as _build_vmec2000_scan_runtime_setup,
    scan_m1_preconditioner_rhs as _scan_m1_preconditioner_rhs,
)
from vmec_jax.solvers.fixed_boundary.scan.time_control import (
    evaluate_scan_time_control_restart,
    scan_restart_transition,
)
from vmec_jax.solvers.free_boundary.control import free_boundary_iter_controls as _free_boundary_iter_controls
from vmec_jax.state import VMECState


def _scan_tree_select(cond, t_true, t_false):
    if t_true is None or t_false is None:
        return t_true if t_false is None else t_false
    if isinstance(t_true, tuple) and isinstance(t_false, tuple):
        return type(t_true)(_scan_tree_select(cond, a, b) for a, b in zip(t_true, t_false, strict=True))
    if isinstance(t_true, list) and isinstance(t_false, list):
        return [_scan_tree_select(cond, a, b) for a, b in zip(t_true, t_false, strict=True)]
    return jnp.where(cond, jnp.asarray(t_true), jnp.asarray(t_false))


def _scan_iteration_input_sequence(
    start: int,
    stop: int,
    *,
    dtype: Any,
    convergence_controls: ScanConvergenceControls,
) -> Any:
    """Build legacy scan iteration inputs with dynamic convergence tolerances."""

    it_seq = jnp.arange(int(start), int(stop), dtype=jnp.int32)
    ftol_seq = jnp.full(it_seq.shape, convergence_controls.ftol, dtype=dtype)
    if convergence_controls.fsq_total_target is None:
        return it_seq, ftol_seq
    target_seq = jnp.full(it_seq.shape, convergence_controls.fsq_total_target, dtype=dtype)
    return it_seq, ftol_seq, target_seq


def _scan_iteration_sequence_only(start: int, stop: int) -> Any:
    """Build the compact VMEC scan iteration sequence used by cached runners."""

    return jnp.arange(int(start), int(stop), dtype=jnp.int32)


def _scan_runtime_controls_args(
    controls: ScanConvergenceControls,
    *,
    dtype: Any,
) -> tuple[Any, ...]:
    """Return scalar runtime controls for cache-stable scan-runner calls."""

    target = controls.fsq_total_target
    if target is None:
        target = jnp.asarray(jnp.nan, dtype=dtype)
    stage_prev_fsq = controls.stage_prev_fsq
    if stage_prev_fsq is None:
        stage_prev_fsq = jnp.asarray(jnp.nan, dtype=dtype)
    stage_transition_factor = controls.stage_transition_factor
    if stage_transition_factor is None:
        stage_transition_factor = jnp.asarray(50.0, dtype=dtype)
    stage_transition_scale = controls.stage_transition_scale
    if stage_transition_scale is None:
        stage_transition_scale = jnp.asarray(0.5, dtype=dtype)
    accept_frac = controls.scan_fallback_accept_frac
    if accept_frac is None:
        accept_frac = jnp.asarray(jnp.nan, dtype=dtype)
    fsq_factor = controls.scan_fallback_fsq_factor
    if fsq_factor is None:
        fsq_factor = jnp.asarray(jnp.nan, dtype=dtype)
    fsq_abs = controls.scan_fallback_fsq_abs
    if fsq_abs is None:
        fsq_abs = jnp.asarray(jnp.nan, dtype=dtype)
    improve = controls.scan_fallback_improve
    if improve is None:
        improve = jnp.asarray(jnp.nan, dtype=dtype)
    return (
        jnp.asarray(controls.ftol, dtype=dtype),
        jnp.asarray(target, dtype=dtype),
        jnp.asarray(stage_prev_fsq, dtype=dtype),
        jnp.asarray(stage_transition_factor, dtype=dtype),
        jnp.asarray(stage_transition_scale, dtype=dtype),
        jnp.asarray(accept_frac, dtype=dtype),
        jnp.asarray(fsq_factor, dtype=dtype),
        jnp.asarray(fsq_abs, dtype=dtype),
        jnp.asarray(improve, dtype=dtype),
    )


def _scan_controls_with_runtime_tolerances(
    base: ScanConvergenceControls,
    *,
    ftol: Any,
    fsq_total_target: Any | None,
) -> ScanConvergenceControls:
    """Return scan controls with updated tolerances and preserved runtime gates."""

    return ScanConvergenceControls(
        ftol=ftol,
        fsq_total_target=fsq_total_target,
        stage_prev_fsq=base.stage_prev_fsq,
        stage_transition_factor=base.stage_transition_factor,
        stage_transition_scale=base.stage_transition_scale,
        scan_fallback_accept_frac=base.scan_fallback_accept_frac,
        scan_fallback_fsq_factor=base.scan_fallback_fsq_factor,
        scan_fallback_fsq_abs=base.scan_fallback_fsq_abs,
        scan_fallback_improve=base.scan_fallback_improve,
    )


def _scan_step_iteration_and_controls(
    step_ctx: "ScanStepContext",
    it: Any,
) -> tuple[Any, ScanConvergenceControls]:
    """Unpack one scan input into the iteration index and convergence controls."""

    if isinstance(it, tuple):
        if len(it) == 2:
            iter_index, ftol = it
            return iter_index, _scan_controls_with_runtime_tolerances(
                step_ctx.convergence_controls,
                ftol=ftol,
                fsq_total_target=None,
            )
        if len(it) == 3:
            iter_index, ftol, fsq_total_target = it
            return iter_index, _scan_controls_with_runtime_tolerances(
                step_ctx.convergence_controls,
                ftol=ftol,
                fsq_total_target=fsq_total_target,
            )
    return it, step_ctx.convergence_controls


def _scan_step_with_runtime_controls(
    step_ctx: "ScanStepContext",
    carry: _ScanCarry,
    it: Any,
    ftol: Any,
    fsq_total_target_value: Any,
    stage_prev_fsq_value: Any,
    stage_transition_factor: Any,
    stage_transition_scale: Any,
    scan_fallback_accept_frac: Any,
    scan_fallback_fsq_factor: Any,
    scan_fallback_fsq_abs: Any,
    scan_fallback_improve: Any,
) -> Any:
    """Dispatch one scan step using scalar runtime convergence controls."""

    has_target = step_ctx.convergence_controls.fsq_total_target is not None
    has_stage_reset = step_ctx.convergence_controls.stage_prev_fsq is not None
    controls = ScanConvergenceControls(
        ftol=ftol,
        fsq_total_target=fsq_total_target_value if has_target else None,
        stage_prev_fsq=stage_prev_fsq_value if has_stage_reset else None,
        stage_transition_factor=stage_transition_factor,
        stage_transition_scale=stage_transition_scale,
        scan_fallback_accept_frac=scan_fallback_accept_frac,
        scan_fallback_fsq_factor=scan_fallback_fsq_factor,
        scan_fallback_fsq_abs=scan_fallback_fsq_abs,
        scan_fallback_improve=scan_fallback_improve,
    )
    iter2_hold = jnp.asarray(it + 1, dtype=jnp.int32) + jnp.asarray(carry.iter_offset, dtype=jnp.int32)
    hold_cond = carry.converged | carry.abort_scan | (iter2_hold > jnp.asarray(int(step_ctx.max_iter), dtype=jnp.int32))
    return jax.lax.cond(
        hold_cond,
        lambda c: _hold_vmec2000_scan_step(step_ctx, c),
        lambda c: _advance_vmec2000_scan_step(step_ctx, c, it, controls),
        operand=carry,
    )


def _select_initial_rz_norm_func(
    *,
    state_init: Any,
    rz_norm_func: Any,
    rz_norm_np_func: Any,
    tree_has_tracer: Any,
    dtype: Any,
):
    """Use host R/Z normalization for non-traced initial scan-cache setup.

    The initial VMEC2000 scan cache is built before the steady scan runner is
    launched. In normal CLI/profile solves the initial state is not traced, so
    the pure-NumPy normalization avoids several tiny JAX compilations. When the
    solve is traced for differentiation, the selector keeps the JAX path so the
    derivative graph remains valid.
    """

    if tree_has_tracer(state_init):
        return rz_norm_func

    def initial_rz_norm_func(state):
        """Return a JAX scalar after computing the non-traced norm on host."""

        try:
            return jnp.asarray(float(rz_norm_np_func(state)), dtype=dtype)
        except Exception:
            return rz_norm_func(state)

    return initial_rz_norm_func


@dataclass(slots=True)
class Vmec2000ScanControllerContext:
    """Closed-over solver state needed by the VMEC2000-style scan controller."""
    _SCAN_RUNNER_CACHE: Any
    _apply_vmec_lambda_axis_rules: Any
    _attach_freeb_diag: Any
    _compute_forces: Any
    _compute_forces_impl: Any
    _lambda_preconditioner: Any
    _maybe_dump_ptau: Any
    _mn_cos_to_signed_physical: Any
    _mn_cos_to_signed_physical_lambda: Any
    _mn_sin_to_signed_physical: Any
    _mn_sin_to_signed_physical_lambda: Any
    _ptau_minmax: Any
    _ptau_minmax_from_k_host: Any
    _reset_axis_from_boundary: Any
    _runtime_env_enabled: Any
    _rz_norm: Any
    _rz_norm_np: Any
    _scan_backend_name: Any
    _scan_chunk_settings: Any
    _tree_has_tracer: Any
    axis_reset_always_3d: Any
    axis_reset_coeffs: Any
    axis_reset_done: Any
    axis_reset_fsq_min: Any
    backtracking: Any
    badjac_use_state: Any
    cfg: Any
    constraint_active_false: Any
    constraint_tcon0: Any
    delta_s: Any
    edge_Rcos: Any
    edge_Rsin: Any
    edge_Zcos: Any
    edge_Zsin: Any
    edge_signature_key: Any
    force_axis_reset: Any
    free_boundary_enabled: Any
    freeb_nvacskip: Any
    freeb_nvskip0: Any
    fsq_total_target: Any
    ftol: Any
    gamma: Any
    idx00: Any
    indata: Any
    initial_flip_sign: Any
    jit_forces: Any
    lambda_update_scale: Any
    lambda_update_scale_j: Any
    limit_dt_from_force: Any
    limit_update_rms: Any
    lmove_axis: Any
    max_iter: Any
    mpol: Any
    ncoeff: Any
    nrange: Any
    preconditioner_use_lax_tridi: Any
    preconditioner_use_precomputed_tridi: Any
    ptau_tol: Any
    reference_mode: Any
    resume_state: Any
    resume_state_mode: Any
    s: Any
    scan_minimal_default: Any
    stage_prev_fsq: Any
    stage_prev_fsq_j: Any
    stage_transition_factor: Any
    stage_transition_scale: Any
    startup_policy: Any
    state0: Any
    state_only: Any
    static: Any
    static_key: Any
    step_size: Any
    strict_update: Any
    trig: Any
    use_direct_fallback: Any
    verbose: Any
    verbose_vmec2000_table: Any
    vmec2000_control: Any
    vmec_half_mesh_jacobian_from_state: Any
    vmecpp_restart: Any
    w_mode_mn: Any
    wout_key: Any
    zero_precond_diag: Any
    zero_tcon: Any

    @classmethod
    def from_namespace(cls, namespace: dict[str, Any], /, **overrides: Any) -> "Vmec2000ScanControllerContext":
        """Build the scan context from resolved residual-iteration locals."""

        return _dataclass_from_namespace(cls, namespace, label="VMEC2000 scan", overrides=overrides)


@dataclass(frozen=True)
class ScanInitialForceSetup:
    """Initial force payload and axis-reset state used to enter the scan loop."""

    state_init: VMECState
    scan_resume0: Any
    axis_reset_enabled: bool
    axis_reset_repeat: bool
    kernels: Any
    frzl: Any
    gcr2: Any
    gcz2: Any
    gcl2: Any
    rz_scale: Any
    l_scale: Any
    norms: Any


@dataclass(slots=True)
class ScanDispatchFinalizeInputs:
    """Explicit inputs for running and finalizing a VMEC2000 scan."""

    ctx: Vmec2000ScanControllerContext
    state_init: VMECState
    carry0: Any
    step_context: "ScanStepContext"
    scan_step: Any
    convergence_controls: ScanConvergenceControls
    scan_runtime_plan: Any
    scan_timing_enabled: bool
    scan_timing_stats: dict[str, float]
    scan_differentiated: bool
    scan_print_context: Any
    scan_minimal: bool
    scan_light: bool
    ftol: float
    fsq_total_target: Any
    scan_run_setup_start: float | None
    axis_reset_repeat: bool
    scan_backend_name: Any
    scan_device_runtime: Any
    state_only_scan: bool
    scan_fallback_enabled_run: bool
    scan_fallback_iters: int
    chunked_print: bool
    max_iter: int
    nstep_screen: int
    scan_collect_print: bool
    scan_fallback_fsq_abs: float
    dtype: Any
    tree_has_tracer: Any
    scan_use_precomputed: bool
    scan_use_lax_tridi: bool
    vmec2000_control: bool
    free_boundary_enabled: bool
    scan_total_start: float
    print_in_scan: bool
    verbose: bool
    verbose_vmec2000_table: bool
    badjac_use_state: bool
    badjac_state_probe: bool
    badjac_initial_state_probe_iters: int


@dataclass(frozen=True)
class ScanFallbackControlArrays:
    """Device arrays for scan fallback acceptance/rejection gates."""

    iters: Any
    badjac_limit: Any
    accept_frac: Any
    fsq_factor: Any
    fsq_abs: Any
    improve: Any


@dataclass(frozen=True)
class ScanDebugSelection:
    """Runtime debug-selection flags used by one VMEC2000 scan."""

    force_enabled: bool
    iter_index: int


@dataclass(frozen=True)
class ScanStepContext:
    """Closed-over constants for one VMEC2000 scan step function."""

    ctx: Vmec2000ScanControllerContext
    scan_runtime: Any
    scan_options: Any
    controller_constants: Any
    fallback_controls: ScanFallbackControlArrays
    scan_debug: ScanDebugSelection
    dtype: Any
    compute_forces_scan: Any
    scan_converged: Any
    convergence_controls: ScanConvergenceControls
    scale_m1_precond_rhs: Any
    jmax0: int
    max_iter: int
    nstep_screen: int
    scan_fallback_enabled_run: bool
    state_only_scan: bool
    scan_minimal: bool
    scan_light: bool
    print_in_scan: bool
    scan_print_mode: Any
    scan_timecontrol_dumper: Any
    flip_sign0: Any


class ScanStepPreparedPayload(NamedTuple):
    """Force, residual, and bad-Jacobian data prepared for one scan step."""

    force_eval: Any
    current_payload: Any
    sample_vmec: Any
    r00: Any
    z00: Any
    w_mhd: Any
    fsq0: Any
    fsq1: Any
    tau_decision: Any
    use_apply_payload_fusion: bool


def _build_vmec2000_scan_runtime(ctx: Vmec2000ScanControllerContext, state_init: VMECState) -> Any:
    """Resolve runtime/JIT/print settings for the VMEC2000-style scan."""

    return _build_vmec2000_scan_runtime_setup(
        env=os.environ,
        state_init=state_init,
        indata=ctx.indata,
        cfg=ctx.cfg,
        mpol=ctx.mpol,
        nrange=ctx.nrange,
        resume_state=ctx.resume_state,
        state_only=bool(ctx.state_only),
        scan_fallback_enabled=bool(ctx.startup_policy.scan_fallback_enabled),
        force_chunked_scan=bool(ctx.startup_policy.force_chunked_scan),
        preconditioner_use_precomputed_tridi=ctx.preconditioner_use_precomputed_tridi,
        preconditioner_use_lax_tridi=ctx.preconditioner_use_lax_tridi,
        verbose=bool(ctx.verbose),
        vmec2000_control=bool(ctx.vmec2000_control),
        verbose_vmec2000_table=bool(ctx.verbose_vmec2000_table),
        light_history=bool(ctx.startup_policy.light_history),
        scan_minimal_default=ctx.scan_minimal_default,
        dump_any=bool(ctx.startup_policy.dump_any),
        fsq_total_target=ctx.fsq_total_target,
        axis_reset_done=bool(ctx.axis_reset_done),
        lmove_axis=bool(ctx.lmove_axis),
        step_size=float(ctx.step_size),
        initial_flip_sign=float(ctx.initial_flip_sign),
        ftol=float(ctx.ftol),
        jit_forces=bool(ctx.jit_forces),
        compute_forces=ctx._compute_forces,
        compute_forces_impl=ctx._compute_forces_impl,
        scan_timing_enabled_func=_scan_timing_enabled,
        new_scan_timing_stats_func=_new_scan_timing_stats,
        scan_backend_name_func=ctx._scan_backend_name,
        tree_has_tracer_func=ctx._tree_has_tracer,
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
        backtracking=bool(ctx.backtracking),
        limit_dt_from_force=bool(ctx.limit_dt_from_force),
        limit_update_rms=bool(ctx.limit_update_rms),
        use_direct_fallback=bool(ctx.use_direct_fallback),
        reference_mode=bool(ctx.reference_mode),
        strict_update=bool(ctx.strict_update),
        auto_flip_force=bool(ctx.startup_policy.auto_flip_force),
    )


def _scan_debug_selection_from_env() -> ScanDebugSelection:
    debug_iter_env = os.getenv("VMEC_JAX_SCAN_DEBUG_ITER", "").strip()
    try:
        scan_debug_iter = int(debug_iter_env) if debug_iter_env else -1
    except Exception:
        scan_debug_iter = -1
    return ScanDebugSelection(
        force_enabled=os.getenv("VMEC_JAX_SCAN_DEBUG_FORCE", "") not in ("", "0"),
        iter_index=scan_debug_iter,
    )


def _scan_fallback_control_arrays(*, ctx: Vmec2000ScanControllerContext, dtype: Any) -> ScanFallbackControlArrays:
    startup_policy = ctx.startup_policy
    return ScanFallbackControlArrays(
        iters=jnp.asarray(int(ctx.startup_policy.scan_fallback_iters), dtype=jnp.int32),
        badjac_limit=jnp.asarray(int(ctx.startup_policy.scan_fallback_badjac_limit), dtype=jnp.int32),
        accept_frac=jnp.asarray(float(ctx.startup_policy.scan_fallback_accept_frac), dtype=dtype),
        fsq_factor=jnp.asarray(float(ctx.startup_policy.scan_fallback_fsq_factor), dtype=dtype),
        fsq_abs=jnp.asarray(float(ctx.startup_policy.scan_fallback_fsq_abs), dtype=dtype),
        improve=jnp.asarray(float(startup_policy.scan_fallback_improve), dtype=dtype),
    )


def _hold_vmec2000_scan_step(step_ctx: ScanStepContext, carry_hold: _ScanCarry) -> Any:
    """Emit a passive scan row after convergence/abort/max-iteration."""

    return _scan_math_hold_step(
        carry_hold,
        dtype=step_ctx.dtype,
        state_only_scan=step_ctx.state_only_scan,
        scan_minimal=step_ctx.scan_minimal,
        scan_light=step_ctx.scan_light,
        scan_hist_min=vmec2000_scan_minimal_history_row,
        scan_hist_light=vmec2000_scan_light_history_row,
    )


def _prepare_vmec2000_scan_step_payload(
    step_ctx: ScanStepContext,
    carry_adv: _ScanCarry,
    it: Any,
    convergence_controls: ScanConvergenceControls,
) -> ScanStepPreparedPayload:
    """Prepare force, residual, and bad-Jacobian inputs for one scan step."""

    ctx = step_ctx.ctx
    cfg = ctx.cfg
    dtype = step_ctx.dtype
    scan_runtime = step_ctx.scan_runtime
    scan_options = step_ctx.scan_options
    constants = step_ctx.controller_constants
    force_eval = _evaluate_scan_step_force(
        carry_adv=carry_adv,
        it=it,
        dtype=dtype,
        k_preconditioner_update_interval=constants.preconditioner_update_interval,
        zero_precond_diag=ctx.zero_precond_diag,
        zero_tcon=ctx.zero_tcon,
        compute_forces_scan=step_ctx.compute_forces_scan,
        scan_converged=step_ctx.scan_converged,
        convergence_controls=convergence_controls,
        tree_select=_scan_tree_select,
        cond=jax.lax.cond,
        trace_context=lambda: scan_runtime.maybe_trace("scan/compute_forces"),
        scan_debug_force_enabled=bool(step_ctx.scan_debug.force_enabled),
        scan_debug_iter=int(step_ctx.scan_debug.iter_index),
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
    sample_vmec, r00_j, z00_j, w_mhd = _sample_vmec2000_scan_scalars(
        carry_adv=carry_adv,
        iter2=iter2,
        max_iter=int(step_ctx.max_iter),
        nstep_screen=int(step_ctx.nstep_screen),
        scan_collect_scalars=bool(scan_options.scan_collect_scalars),
        force_sample=conv_now,
        kernels=k,
        norms_current=norms_current,
        gamma=float(ctx.gamma),
        twopi=float(TWOPI),
        lasym=bool(cfg.lasym),
        cond=jax.lax.cond,
    )

    current_payload_pre = _build_current_preconditioned_scan_payload(
        need_bcovar_update=need_bcovar_update,
        carry_adv=carry_adv,
        k=k,
        frzl=frzl,
        norms_used=norms_used,
        rz_scale=rz_scale,
        l_scale=l_scale,
        constraint_tcon0=ctx.constraint_tcon0,
        zero_precond_diag=ctx.zero_precond_diag,
        zero_tcon=ctx.zero_tcon,
        trig=ctx.trig,
        s=ctx.s,
        cfg=cfg,
        dtype=dtype,
        scan_use_precomputed=bool(scan_options.scan_use_precomputed),
        scan_use_lax_tridi=bool(scan_options.scan_use_lax_tridi),
        lambda_preconditioner_func=ctx._lambda_preconditioner,
        rz_norm_func=ctx._rz_norm,
        scale_m1_precond_rhs_func=step_ctx.scale_m1_precond_rhs,
        w_mode_mn=ctx.w_mode_mn,
        lambda_update_scale_j=ctx.lambda_update_scale_j,
        apply_lambda_update_scale=(ctx.lambda_update_scale != 1.0),
        fsqr=fsqr,
        fsqz=fsqz,
        fsql=fsql,
        delta_s=ctx.delta_s,
        jmax0=step_ctx.jmax0,
        cond=jax.lax.cond,
    )
    fsqr1 = current_payload_pre.fsqr1
    fsqz1 = current_payload_pre.fsqz1
    fsql1 = current_payload_pre.fsql1
    fsq1 = fsqr1 + fsqz1 + fsql1
    fsq0 = fsqr + fsqz + fsql
    use_state_jac = ctx._runtime_env_enabled(os.getenv("VMEC_JAX_SCAN_JAC_FROM_STATE", "0"))
    use_apply_payload_fusion = False
    ptau_min, ptau_max = ctx._ptau_minmax(k) if bool(ctx.vmec2000_control) else (None, None)
    tau_decision = _scan_bad_jacobian_decision_from_step(
        carry_adv=carry_adv,
        kernels=k,
        iter2=iter2,
        static=ctx.static,
        trig=ctx.trig,
        s=ctx.s,
        vmec2000_control=bool(ctx.vmec2000_control),
        use_apply_payload_fusion=bool(use_apply_payload_fusion),
        badjac_use_state=bool(ctx.badjac_use_state),
        dump_ptau_state=bool(ctx.startup_policy.dump_ptau_state),
        badjac_state_probe=bool(ctx.startup_policy.badjac_state_probe),
        badjac_initial_state_probe_iters=int(ctx.startup_policy.badjac_initial_state_probe_iters),
        ptau_min=ptau_min,
        ptau_max=ptau_max,
        ptau_tol=ctx.ptau_tol,
        dtype=dtype,
        use_state_jac=bool(use_state_jac),
        ignore_badjac=(os.getenv("VMEC_JAX_SCAN_IGNORE_BADJAC", "") not in ("", "0")),
        vmec_half_mesh_jacobian_from_state_func=ctx.vmec_half_mesh_jacobian_from_state,
        cond=jax.lax.cond,
    )

    return ScanStepPreparedPayload(
        force_eval=force_eval,
        current_payload=current_payload_pre,
        sample_vmec=sample_vmec,
        r00=r00_j,
        z00=z00_j,
        w_mhd=w_mhd,
        fsq0=fsq0,
        fsq1=fsq1,
        tau_decision=tau_decision,
        use_apply_payload_fusion=use_apply_payload_fusion,
    )


def _advance_vmec2000_scan_step(
    step_ctx: ScanStepContext,
    carry_adv: _ScanCarry,
    it: Any,
    convergence_controls: ScanConvergenceControls,
) -> Any:
    """Advance one active VMEC2000 scan step."""

    ctx = step_ctx.ctx
    cfg = ctx.cfg
    dtype = step_ctx.dtype
    scan_runtime = step_ctx.scan_runtime
    scan_options = step_ctx.scan_options
    constants = step_ctx.controller_constants
    fallback_controls = ScanFallbackControlArrays(
        iters=step_ctx.fallback_controls.iters,
        badjac_limit=step_ctx.fallback_controls.badjac_limit,
        accept_frac=convergence_controls.scan_fallback_accept_frac,
        fsq_factor=convergence_controls.scan_fallback_fsq_factor,
        fsq_abs=convergence_controls.scan_fallback_fsq_abs,
        improve=convergence_controls.scan_fallback_improve,
    )
    prepared = _prepare_vmec2000_scan_step_payload(step_ctx, carry_adv, it, convergence_controls)
    force_eval = prepared.force_eval
    current_payload_pre = prepared.current_payload
    tau_decision = prepared.tau_decision
    iter2 = force_eval.iter2
    fsq_prev_before = force_eval.fsq_prev_before
    fsq0_prev_before = force_eval.fsq0_prev_before
    skip_timecontrol = force_eval.skip_timecontrol
    zero_m1 = force_eval.zero_m1
    include_edge = force_eval.include_edge
    fsqr = force_eval.fsqr
    fsqz = force_eval.fsqz
    fsql = force_eval.fsql
    fsqr1 = current_payload_pre.fsqr1
    fsqz1 = current_payload_pre.fsqz1
    fsql1 = current_payload_pre.fsql1
    fsq0 = prepared.fsq0
    fsq1 = prepared.fsq1
    bad_jacobian = tau_decision.bad_jacobian
    min_tau = tau_decision.min_tau
    max_tau = tau_decision.max_tau
    min_tau_ptau = tau_decision.min_tau_ptau
    max_tau_ptau = tau_decision.max_tau_ptau
    min_tau_state = tau_decision.min_tau_state
    max_tau_state = tau_decision.max_tau_state
    badjac_ptau = tau_decision.badjac_ptau
    badjac_state = tau_decision.badjac_state

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
        vmec2000_control=bool(ctx.vmec2000_control),
        reference_mode=bool(ctx.reference_mode),
        use_apply_payload_fusion=bool(prepared.use_apply_payload_fusion),
        dump_timecontrol_scan=bool(step_ctx.scan_runtime.dump_timecontrol_scan),
        scan_timecontrol_dumper=step_ctx.scan_timecontrol_dumper,
        vmec2000_fact=constants.vmec2000_fact,
        use_restart_triggers=bool(ctx.startup_policy.use_restart_triggers),
        vmecpp_restart=bool(ctx.vmecpp_restart),
        k_preconditioner_update_interval=constants.preconditioner_update_interval,
        stage_prev_fsq=convergence_controls.stage_prev_fsq,
        stage_transition_factor=convergence_controls.stage_transition_factor,
        restart_badjac_factor=constants.restart_badjac_factor,
        restart_badprog_factor=constants.restart_badprog_factor,
        stage_transition_scale=convergence_controls.stage_transition_scale,
        step_size=ctx.step_size,
        k_ndamp=constants.ndamp,
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

    payload_step = _select_payload_and_build_step_fields(
        do_restart=do_restart,
        use_restart_payload=bool(scan_options.scan_use_restart_payload),
        current_payload=current_payload_pre,
        state_post=state_post,
        compute_forces_scan_func=step_ctx.compute_forces_scan,
        restart_trace_context=lambda: scan_runtime.maybe_trace("scan/compute_forces:restart"),
        zero_m1=zero_m1,
        zero_precond_diag=ctx.zero_precond_diag,
        zero_tcon=ctx.zero_tcon,
        constraint_active_false=ctx.constraint_active_false,
        constraint_tcon0=ctx.constraint_tcon0,
        trig=ctx.trig,
        s=ctx.s,
        cfg=cfg,
        dtype=dtype,
        scan_use_precomputed=bool(scan_options.scan_use_precomputed),
        scan_use_lax_tridi=bool(scan_options.scan_use_lax_tridi),
        lambda_preconditioner_func=ctx._lambda_preconditioner,
        rz_norm_func=ctx._rz_norm,
        scale_m1_precond_rhs_func=step_ctx.scale_m1_precond_rhs,
        w_mode_mn=ctx.w_mode_mn,
        lambda_update_scale_j=ctx.lambda_update_scale_j,
        apply_lambda_update_scale=(ctx.lambda_update_scale != 1.0),
        delta_s=ctx.delta_s,
        jmax0=step_ctx.jmax0,
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
        k_ndamp=constants.ndamp,
        flip_sign=step_ctx.flip_sign0,
        lasym=bool(cfg.lasym),
        static=ctx.static,
        edge_Rcos=carry_adv.edge_Rcos,
        edge_Rsin=carry_adv.edge_Rsin,
        edge_Zcos=carry_adv.edge_Zcos,
        edge_Zsin=carry_adv.edge_Zsin,
        free_boundary_enabled=bool(ctx.free_boundary_enabled),
        idx00=ctx.idx00,
        mn_cos_to_signed_physical=ctx._mn_cos_to_signed_physical,
        mn_sin_to_signed_physical=ctx._mn_sin_to_signed_physical,
        mn_sin_to_signed_physical_lambda=ctx._mn_sin_to_signed_physical_lambda,
        mn_cos_to_signed_physical_lambda=ctx._mn_cos_to_signed_physical_lambda,
        enforce_fixed_boundary_and_axis=_enforce_fixed_boundary_and_axis,
        apply_vmec_lambda_axis_rules=ctx._apply_vmec_lambda_axis_rules,
        vmec2000_control=bool(ctx.vmec2000_control),
        cond=jax.lax.cond,
    )
    payload_use = payload_step.payload
    step_fields = payload_step.step_fields
    fsqr = payload_use.fsqr
    fsqz = payload_use.fsqz
    fsql = payload_use.fsql
    fsq1 = payload_step.fsq1
    time_step_report = time_step_post
    _ = _emit_live_scan_vmec2000_row(
        enabled=step_ctx.print_in_scan,
        sample_vmec=prepared.sample_vmec,
        iter_idx=iter2,
        fsqr=fsqr,
        fsqz=fsqz,
        fsql=fsql,
        delt0r=time_step_report,
        r00=prepared.r00,
        w_mhd=prepared.w_mhd,
        scan_print_mode=step_ctx.scan_print_mode,
        scan_print_ordered=bool(scan_options.scan_print_ordered),
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
        scan_fallback_enabled_run=step_ctx.scan_fallback_enabled_run,
        scan_core=bool(scan_options.scan_core),
        fsq_phys=fsq_phys,
        fsq1=fsq1,
        bad_jacobian=bad_jacobian,
        abort_scan_on_badjac=scan_options.abort_scan_on_badjac,
        scan_fallback_iters=fallback_controls.iters,
        scan_fallback_badjac_limit=fallback_controls.badjac_limit,
        scan_fallback_accept_frac=fallback_controls.accept_frac,
        scan_fallback_fsq_factor=fallback_controls.fsq_factor,
        scan_fallback_fsq_abs=fallback_controls.fsq_abs,
        scan_fallback_improve=fallback_controls.improve,
        dtype=dtype,
        vmec2000_control=bool(ctx.vmec2000_control),
        do_restart=do_restart,
        state_only_scan=bool(step_ctx.state_only_scan),
        scan_minimal=bool(step_ctx.scan_minimal),
        scan_light=bool(step_ctx.scan_light),
        fsq0_prev_post=time_restart.fsq0_prev_post,
        force_bcovar_post=force_bcovar_post,
        flip_sign=step_ctx.flip_sign0,
        iter_offset_post=iter_offset_post,
        iter1_post=iter1_post,
        res0=res0,
        res1=res1,
        ijacob_post=ijacob_post,
        bad_resets_post=bad_resets_post,
        bad_growth_post=bad_growth_post,
        r00=prepared.r00,
        z00=prepared.z00,
        w_mhd=prepared.w_mhd,
        conv_now=force_eval.conv_now,
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


def _vmec2000_scan_step(step_ctx: ScanStepContext, carry: _ScanCarry, it: Any) -> Any:
    """Dispatch one scan iteration to hold or active advancement."""

    iter_index, convergence_controls = _scan_step_iteration_and_controls(step_ctx, it)
    iter2_hold = jnp.asarray(iter_index + 1, dtype=jnp.int32) + jnp.asarray(carry.iter_offset, dtype=jnp.int32)
    hold_cond = carry.converged | carry.abort_scan | (iter2_hold > jnp.asarray(int(step_ctx.max_iter), dtype=jnp.int32))
    return jax.lax.cond(
        hold_cond,
        lambda c: _hold_vmec2000_scan_step(step_ctx, c),
        lambda c: _advance_vmec2000_scan_step(step_ctx, c, iter_index, convergence_controls),
        operand=carry,
    )


def _prepare_scan_initial_force_and_axis_reset(
    *,
    ctx: Vmec2000ScanControllerContext,
    state_init: VMECState,
    scan_runtime,
    scan_resume0,
    axis_reset_enabled: bool,
    axis_reset_repeat: bool,
    dtype,
    static,
    trig,
    s,
    zero_precond_diag,
    zero_tcon,
    constraint_active_false,
    vmec2000_control: bool,
    lmove_axis: bool,
    verbose: bool,
    verbose_vmec2000_table: bool,
    badjac_use_state: bool,
    ptau_tol: float,
    compute_forces_scan,
    scan_print_context,
) -> ScanInitialForceSetup:
    """Compute initial scan forces and apply the VMEC initial-axis reset."""

    timing_enabled = bool(scan_runtime.timing_enabled)
    timing_stats = scan_runtime.timing_stats
    device_runtime = scan_runtime.device_runtime
    t_scan_initial_force = time.perf_counter() if timing_enabled else None
    with scan_runtime.maybe_trace("scan/compute_forces:init"):
        k0, frzl0, gcr2_0, gcz2_0, gcl2_0, rz_scale0, l_scale0, norms0 = compute_forces_scan(
            state_init,
            include_edge=False,
            zero_m1=jnp.asarray(1.0, dtype=dtype),
            constraint_precond_diag=zero_precond_diag,
            constraint_tcon=zero_tcon,
            constraint_precond_active=constraint_active_false,
            constraint_tcon_active=constraint_active_false,
            iter_idx=None,
        )
    if timing_enabled and t_scan_initial_force is not None:
        try:
            if has_jax():
                device_runtime.block_value((gcr2_0, gcz2_0, gcl2_0))
        except Exception:
            pass
        timing_stats["scan_initial_compute_forces_s"] += time.perf_counter() - float(t_scan_initial_force)

    axis_reset_eval = _evaluate_initial_axis_reset(
        axis_reset_enabled=bool(axis_reset_enabled),
        norms=norms0,
        gcr2=gcr2_0,
        gcz2=gcz2_0,
        gcl2=gcl2_0,
        k=k0,
        state=state_init,
        static=static,
        trig=trig,
        s=s,
        badjac_use_state=bool(badjac_use_state),
        ptau_tol=ptau_tol,
        ptau_tol_rel=ctx.startup_policy.ptau_tol_rel,
        axis_reset_fsq_min=ctx.axis_reset_fsq_min,
        force_axis_reset=bool(ctx.force_axis_reset),
        axis_reset_always_3d=bool(ctx.axis_reset_always_3d),
        vmec2000_control=bool(vmec2000_control),
        lmove_axis=bool(lmove_axis),
        debug_enabled=(
            bool(axis_reset_enabled)
            and ctx._runtime_env_enabled(os.getenv("VMEC_JAX_AXIS_RESET_DEBUG", ""))
        ),
        ptau_minmax_from_k_host=ctx._ptau_minmax_from_k_host,
        vmec_half_mesh_jacobian_from_state_func=ctx.vmec_half_mesh_jacobian_from_state,
    )
    axis_reset_decision = axis_reset_eval.decision
    bad_jacobian0 = bool(axis_reset_decision.bad_jacobian)
    force_axis_reset_init = axis_reset_decision.force_reset
    if axis_reset_decision.reset:
        if bool(verbose) and bool(vmec2000_control) and bool(verbose_vmec2000_table):
            if bad_jacobian0 or force_axis_reset_init:
                print(" INITIAL JACOBIAN CHANGED SIGN!", flush=True)
            print(" TRYING TO IMPROVE INITIAL MAGNETIC AXIS GUESS", flush=True)
        state_init = ctx._reset_axis_from_boundary(state_init, k_guess=k0, full_reset=False, refine_axis_guess=False)
        if bool(verbose) and bool(vmec2000_control) and bool(verbose_vmec2000_table):
            if ctx.axis_reset_coeffs is not None:
                raxis_cc, _raxis_cs, _zaxis_cc, zaxis_cs = ctx.axis_reset_coeffs
                scan_print_context.print_axis_guess(raxis_cc, zaxis_cs)
        scan_resume0 = scan_resume0._replace(
            ijacob=jnp.asarray(1, dtype=jnp.int32),
            state_checkpoint=state_init,
        )
        axis_reset_enabled = False
        axis_reset_repeat = True
        t_scan_axis_force = time.perf_counter() if timing_enabled else None
        k0, frzl0, gcr2_0, gcz2_0, gcl2_0, rz_scale0, l_scale0, norms0 = compute_forces_scan(
            state_init,
            include_edge=False,
            zero_m1=jnp.asarray(1.0, dtype=dtype),
            constraint_precond_diag=zero_precond_diag,
            constraint_tcon=zero_tcon,
            constraint_precond_active=constraint_active_false,
            constraint_tcon_active=constraint_active_false,
            iter_idx=None,
        )
        if timing_enabled and t_scan_axis_force is not None:
            try:
                if has_jax():
                    device_runtime.block_value((gcr2_0, gcz2_0, gcl2_0))
            except Exception:
                pass
            timing_stats["scan_axis_reset_compute_forces_s"] += time.perf_counter() - float(
                t_scan_axis_force
            )
    return ScanInitialForceSetup(
        state_init=state_init,
        scan_resume0=scan_resume0,
        axis_reset_enabled=False,
        axis_reset_repeat=bool(axis_reset_repeat),
        kernels=k0,
        frzl=frzl0,
        gcr2=gcr2_0,
        gcz2=gcz2_0,
        gcl2=gcl2_0,
        rz_scale=rz_scale0,
        l_scale=l_scale0,
        norms=norms0,
    )


def _run_scan_dispatch_and_finalize(inputs: ScanDispatchFinalizeInputs) -> SolveVmecResidualResult:
    """Run the scan body and convert scan outputs to the public result."""

    ctx = inputs.ctx
    scan_runtime_plan = inputs.scan_runtime_plan
    scan_cache_key = scan_runtime_plan.scan_cache_key
    iter_offset0 = scan_runtime_plan.iter_offset0
    scan_timing_enabled = bool(inputs.scan_timing_enabled)
    scan_timing_stats = inputs.scan_timing_stats

    def _run_scan(
        carry_init,
        it_seq,
        ftol_dyn,
        fsq_total_target_dyn,
        stage_prev_fsq_dyn,
        stage_transition_factor_dyn,
        stage_transition_scale_dyn,
        fallback_accept_frac_dyn,
        fallback_fsq_factor_dyn,
        fallback_fsq_abs_dyn,
        fallback_improve_dyn,
    ):
        def _step(carry, it):
            return _scan_step_with_runtime_controls(
                inputs.step_context,
                carry,
                it,
                ftol_dyn,
                fsq_total_target_dyn,
                stage_prev_fsq_dyn,
                stage_transition_factor_dyn,
                stage_transition_scale_dyn,
                fallback_accept_frac_dyn,
                fallback_fsq_factor_dyn,
                fallback_fsq_abs_dyn,
                fallback_improve_dyn,
            )

        return jax.lax.scan(_step, carry_init, it_seq)

    def _get_scan_runner(seq_len: int):
        key = scan_cache_key + (int(seq_len),)
        return _get_or_build_scan_runner(
            _run_scan,
            cache=ctx._SCAN_RUNNER_CACHE,
            key=key,
            differentiating_scan=bool(inputs.scan_differentiated),
            scan_timing_enabled=scan_timing_enabled,
            scan_timing_stats=scan_timing_stats,
            jit_func=jit,
            cache_get=_jit_cache_get,
            cache_put=_jit_cache_put,
            record_miss_categories=_record_scan_runner_cache_miss_categories,
            perf_counter=time.perf_counter,
        )

    scan_print_context = inputs.scan_print_context
    emit_scan_prints = partial(
        _emit_scan_debug_prints,
        scan_minimal=bool(inputs.scan_minimal),
        scan_light=bool(inputs.scan_light),
        ftol=float(inputs.ftol),
        fsq_total_target=inputs.fsq_total_target,
        iter_offset0=int(iter_offset0),
        should_print=scan_print_context.should_print,
        print_row=scan_print_context.print_row,
    )

    scan_run_setup_start = inputs.scan_run_setup_start
    if scan_timing_enabled and scan_run_setup_start is not None:
        scan_timing_stats["scan_run_setup_s"] += time.perf_counter() - float(scan_run_setup_start)
    carry_init = inputs.carry0._replace(state=inputs.state_init, state_checkpoint=inputs.state_init)
    scan_dispatch_common = {
        "scan_jit_preflight_enabled_func": _scan_jit_preflight_enabled,
        "scan_jit_preflight_env": os.getenv("VMEC_JAX_SCAN_JIT_PREFLIGHT"),
        "backend_name": inputs.scan_backend_name(),
        "scan_differentiated": bool(inputs.scan_differentiated),
        "preflight_iters": int(scan_runtime_plan.preflight_iters),
        "iter_offset_preflight": int(scan_runtime_plan.iter_offset_preflight),
        "axis_reset_repeat": bool(inputs.axis_reset_repeat),
        "iter_offset0": int(iter_offset0),
        "get_scan_runner": _get_scan_runner,
        "scan_step": inputs.scan_step,
        "build_scan_it_seq": _scan_iteration_sequence_only,
        "runtime_scan_args": _scan_runtime_controls_args(
            inputs.convergence_controls,
            dtype=inputs.dtype,
        ),
        "scan_timing_enabled": scan_timing_enabled,
        "scan_timing_stats": scan_timing_stats,
        "scan_device_runtime": inputs.scan_device_runtime,
        "perf_counter": time.perf_counter,
        "state_only_scan": bool(inputs.state_only_scan),
        "scan_fallback_enabled_run": bool(inputs.scan_fallback_enabled_run),
        "scan_fallback_iters": int(inputs.scan_fallback_iters),
        "jnp_module": jnp,
        "jax_module": jax,
    }
    cfg = ctx.cfg
    scan_dispatch = _run_vmec2000_scan_dispatch(
        carry_init,
        chunked_print=bool(inputs.chunked_print),
        chunked_kwargs={
            **scan_dispatch_common,
            "max_iter": int(inputs.max_iter),
            "max_iter_scan": int(scan_runtime_plan.max_iter_scan),
            "nstep_screen": int(inputs.nstep_screen),
            "need_print": bool(inputs.scan_collect_print),
            "lthreed": bool(cfg.lthreed),
            "spectral_mode_count": int(ctx.ncoeff),
            "scan_chunk_settings_func": ctx._scan_chunk_settings,
            "scan_fallback_fsq_abs": float(inputs.scan_fallback_fsq_abs),
            "dtype": inputs.dtype,
            "emit_scan_prints": emit_scan_prints,
            "tree_has_tracer": inputs.tree_has_tracer,
            "np_module": np,
        },
        nonchunked_kwargs={
            **scan_dispatch_common,
            "max_iter_scan": int(scan_runtime_plan.max_iter_scan),
            "max_iter_tail": int(scan_runtime_plan.max_iter_tail),
            "scan_collect_print": bool(inputs.scan_collect_print),
        },
    )
    return finalize_vmec2000_scan_run(
        carry_final=scan_dispatch.carry_final,
        history=scan_dispatch.history,
        state0=ctx.state0,
        result_type=SolveVmecResidualResult,
        state_only_scan=bool(inputs.state_only_scan),
        scan_minimal=bool(inputs.scan_minimal),
        scan_light=bool(inputs.scan_light),
        scan_use_precomputed=bool(inputs.scan_use_precomputed),
        scan_use_lax_tridi=bool(inputs.scan_use_lax_tridi),
        vmec2000_control=bool(inputs.vmec2000_control),
        ftol=float(inputs.ftol),
        fsq_total_target=inputs.fsq_total_target,
        max_iter=int(inputs.max_iter),
        resume_state_mode=str(ctx.resume_state_mode),
        pack_resume_state=partial(_pack_resume_state_record, mode=str(ctx.resume_state_mode)),
        free_boundary_enabled=bool(inputs.free_boundary_enabled),
        freeb_nvacskip=int(ctx.freeb_nvacskip),
        freeb_nvskip0=int(ctx.freeb_nvskip0),
        iter_offset0=int(iter_offset0),
        free_boundary_iter_controls=_free_boundary_iter_controls,
        scan_timing_enabled=scan_timing_enabled,
        scan_timing_stats=scan_timing_stats,
        scan_postprocess_start=time.perf_counter() if scan_timing_enabled else None,
        scan_total_start=inputs.scan_total_start,
        perf_counter=time.perf_counter,
        build_timing_report=_build_scan_timing_report,
        tree_has_tracer=inputs.tree_has_tracer,
        build_traced_scan_resume_state=_build_traced_scan_resume_state,
        state_only_scan_result=_vmec2000_state_only_scan_result,
        traced_scan_result=_vmec2000_traced_scan_result,
        attach_free_boundary_diagnostics=ctx._attach_freeb_diag,
        emit_post_scan_rows=_emit_vmec2000_post_scan_rows,
        post_scan_print_enabled=(
            (not bool(inputs.scan_minimal))
            and (not bool(inputs.print_in_scan))
            and (not bool(inputs.chunked_print))
            and bool(inputs.verbose)
            and bool(inputs.vmec2000_control)
            and bool(inputs.verbose_vmec2000_table)
        ),
        should_print=scan_print_context.should_print,
        print_row=scan_print_context.print_row,
        dump_ptau_rows=_dump_vmec2000_scan_ptau_rows,
        dump_ptau_enabled=(
            (not bool(inputs.scan_light))
            and (not bool(inputs.scan_minimal))
            and os.getenv("VMEC_JAX_DUMP_PTAU", "") not in ("", "0")
        ),
        badjac_mode=ctx.startup_policy.badjac_mode,
        dump_ptau=ctx._maybe_dump_ptau,
        badjac_use_state=bool(inputs.badjac_use_state),
        badjac_state_probe=bool(inputs.badjac_state_probe),
        badjac_initial_state_probe_iters=int(inputs.badjac_initial_state_probe_iters),
    )


def run_vmec2000_scan(ctx: Vmec2000ScanControllerContext, state_init: VMECState) -> SolveVmecResidualResult:
    """Run the VMEC2000-style residual scan controller."""
    _lambda_preconditioner = ctx._lambda_preconditioner
    _runtime_env_enabled = ctx._runtime_env_enabled
    _rz_norm = ctx._rz_norm
    _rz_norm_np = ctx._rz_norm_np
    _scan_backend_name = ctx._scan_backend_name
    _tree_has_tracer = ctx._tree_has_tracer
    axis_reset_coeffs = ctx.axis_reset_coeffs
    startup_policy = ctx.startup_policy
    badjac_initial_state_probe_iters = startup_policy.badjac_initial_state_probe_iters
    badjac_state_probe = startup_policy.badjac_state_probe
    badjac_use_state = ctx.badjac_use_state
    cfg = ctx.cfg
    constraint_active_false = ctx.constraint_active_false
    constraint_tcon0 = ctx.constraint_tcon0
    delta_s = ctx.delta_s
    free_boundary_enabled = ctx.free_boundary_enabled
    fsq_total_target = ctx.fsq_total_target
    ftol = ctx.ftol
    initial_flip_sign = ctx.initial_flip_sign
    lambda_update_scale = ctx.lambda_update_scale
    lambda_update_scale_j = ctx.lambda_update_scale_j
    lmove_axis = ctx.lmove_axis
    max_iter = ctx.max_iter
    ptau_tol = ctx.ptau_tol
    reference_mode = ctx.reference_mode
    resume_state = ctx.resume_state
    s = ctx.s
    scan_fallback_accept_frac = startup_policy.scan_fallback_accept_frac
    scan_fallback_badjac_limit = startup_policy.scan_fallback_badjac_limit
    scan_fallback_fsq_abs = startup_policy.scan_fallback_fsq_abs
    scan_fallback_fsq_factor = startup_policy.scan_fallback_fsq_factor
    scan_fallback_iters = startup_policy.scan_fallback_iters
    stage_transition_factor = ctx.stage_transition_factor
    stage_transition_scale = ctx.stage_transition_scale
    static = ctx.static
    step_size = ctx.step_size
    trig = ctx.trig
    use_restart_triggers = startup_policy.use_restart_triggers
    verbose = ctx.verbose
    verbose_vmec2000_table = ctx.verbose_vmec2000_table
    vmec2000_control = ctx.vmec2000_control
    vmec_half_mesh_jacobian_from_state = ctx.vmec_half_mesh_jacobian_from_state
    vmecpp_restart = ctx.vmecpp_restart
    w_mode_mn = ctx.w_mode_mn
    zero_precond_diag = ctx.zero_precond_diag
    zero_tcon = ctx.zero_tcon
    scan_runtime = _build_vmec2000_scan_runtime(ctx, state_init)

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
    initial_force = _prepare_scan_initial_force_and_axis_reset(
        ctx=ctx,
        state_init=state_init,
        scan_runtime=scan_runtime,
        scan_resume0=scan_resume0,
        axis_reset_enabled=bool(axis_reset_enabled),
        axis_reset_repeat=bool(axis_reset_repeat),
        dtype=dtype,
        static=static,
        trig=trig,
        s=s,
        zero_precond_diag=zero_precond_diag,
        zero_tcon=zero_tcon,
        constraint_active_false=constraint_active_false,
        vmec2000_control=bool(vmec2000_control),
        lmove_axis=bool(lmove_axis),
        verbose=bool(verbose),
        verbose_vmec2000_table=bool(verbose_vmec2000_table),
        badjac_use_state=bool(badjac_use_state),
        ptau_tol=ptau_tol,
        compute_forces_scan=_compute_forces_scan,
        scan_print_context=scan_print_context,
    )
    state_init = initial_force.state_init
    scan_resume0 = initial_force.scan_resume0
    axis_reset_enabled = initial_force.axis_reset_enabled
    axis_reset_repeat = initial_force.axis_reset_repeat
    k0 = initial_force.kernels
    frzl0 = initial_force.frzl
    gcr2_0 = initial_force.gcr2
    gcz2_0 = initial_force.gcz2
    gcl2_0 = initial_force.gcl2
    rz_scale0 = initial_force.rz_scale
    l_scale0 = initial_force.l_scale
    norms0 = initial_force.norms
    initial_rz_norm_func = _select_initial_rz_norm_func(
        state_init=state_init,
        rz_norm_func=_rz_norm,
        rz_norm_np_func=_rz_norm_np,
        tree_has_tracer=_tree_has_tracer,
        dtype=dtype,
    )
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
        rz_norm_func=initial_rz_norm_func,
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

    scan_fallback_controls = _scan_fallback_control_arrays(ctx=ctx, dtype=dtype)
    scan_debug = _scan_debug_selection_from_env()
    convergence_controls = ScanConvergenceControls(
        ftol=scan_converged.ftol,
        fsq_total_target=scan_converged.fsq_total_target,
        stage_prev_fsq=ctx.stage_prev_fsq_j,
        stage_transition_factor=jnp.asarray(float(stage_transition_factor), dtype=dtype),
        stage_transition_scale=jnp.asarray(float(stage_transition_scale), dtype=dtype),
        scan_fallback_accept_frac=scan_fallback_controls.accept_frac,
        scan_fallback_fsq_factor=scan_fallback_controls.fsq_factor,
        scan_fallback_fsq_abs=scan_fallback_controls.fsq_abs,
        scan_fallback_improve=scan_fallback_controls.improve,
    )

    step_context = ScanStepContext(
        ctx=ctx,
        scan_runtime=scan_runtime,
        scan_options=scan_options,
        controller_constants=controller_constants,
        fallback_controls=scan_fallback_controls,
        scan_debug=scan_debug,
        dtype=dtype,
        compute_forces_scan=_compute_forces_scan,
        scan_converged=scan_converged,
        convergence_controls=convergence_controls,
        scale_m1_precond_rhs=scale_m1_precond_rhs,
        jmax0=jmax0,
        max_iter=int(max_iter),
        nstep_screen=int(nstep_screen),
        scan_fallback_enabled_run=bool(scan_fallback_enabled_run),
        state_only_scan=bool(state_only_scan),
        scan_minimal=bool(scan_minimal),
        scan_light=bool(scan_light),
        print_in_scan=bool(print_in_scan),
        scan_print_mode=scan_print_mode,
        scan_timecontrol_dumper=scan_timecontrol_dumper,
        flip_sign0=flip_sign0,
    )
    scan_step = partial(_vmec2000_scan_step, step_context)

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
        edge_Rcos=ctx.edge_Rcos,
        edge_Rsin=ctx.edge_Rsin,
        edge_Zcos=ctx.edge_Zcos,
        edge_Zsin=ctx.edge_Zsin,
        use_numpy_defaults=not bool(scan_differentiated),
    )

    scan_runtime_plan = _resolve_scan_iteration_runtime_plan(
        env=os.environ,
        jit_forces_scan=bool(jit_forces_scan),
        vmec2000_control=bool(vmec2000_control),
        max_iter=int(max_iter),
        axis_reset_repeat=bool(axis_reset_repeat),
        iter_offset0=int(iter_offset0),
        static_key=ctx.static_key,
        wout_key=ctx.wout_key,
        edge_signature_key=ctx.edge_signature_key,
        step_size=float(step_size),
        initial_flip_sign=float(initial_flip_sign),
        lambda_update_scale=float(lambda_update_scale),
        ftol=float(ftol),
        fsq_total_target=None if fsq_total_target is None else float(fsq_total_target),
        nstep_screen=int(nstep_screen),
        use_restart_triggers=bool(use_restart_triggers),
        vmecpp_restart=bool(vmecpp_restart),
        scan_use_precomputed=bool(scan_use_precomputed),
        scan_use_lax_tridi=bool(scan_use_lax_tridi),
        scan_use_restart_payload=bool(scan_use_restart_payload),
        stage_prev_fsq=ctx.stage_prev_fsq,
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
    iter_offset0 = scan_runtime_plan.iter_offset0
    if scan_runtime_plan.axis_reset_repeated:
        carry0 = carry0._replace(iter_offset=jnp.asarray(iter_offset0, dtype=jnp.int32))

    return _run_scan_dispatch_and_finalize(
        ScanDispatchFinalizeInputs(
            ctx=ctx,
            state_init=state_init,
            carry0=carry0,
            step_context=step_context,
            scan_step=scan_step,
            convergence_controls=convergence_controls,
            scan_runtime_plan=scan_runtime_plan,
            scan_timing_enabled=bool(scan_timing_enabled),
            scan_timing_stats=scan_timing_stats,
            scan_differentiated=bool(scan_differentiated),
            scan_print_context=scan_print_context,
            scan_minimal=bool(scan_minimal),
            scan_light=bool(scan_light),
            ftol=float(ftol),
            fsq_total_target=fsq_total_target,
            scan_run_setup_start=scan_run_setup_start,
            axis_reset_repeat=bool(axis_reset_repeat),
            scan_backend_name=_scan_backend_name,
            scan_device_runtime=scan_device_runtime,
            state_only_scan=bool(state_only_scan),
            scan_fallback_enabled_run=bool(scan_fallback_enabled_run),
            scan_fallback_iters=int(scan_fallback_iters),
            chunked_print=bool(chunked_print),
            max_iter=int(max_iter),
            nstep_screen=int(nstep_screen),
            scan_collect_print=bool(scan_collect_print),
            scan_fallback_fsq_abs=float(scan_fallback_fsq_abs),
            dtype=dtype,
            tree_has_tracer=_tree_has_tracer,
            scan_use_precomputed=bool(scan_use_precomputed),
            scan_use_lax_tridi=bool(scan_use_lax_tridi),
            vmec2000_control=bool(vmec2000_control),
            free_boundary_enabled=bool(free_boundary_enabled),
            scan_total_start=scan_total_start,
            print_in_scan=bool(print_in_scan),
            verbose=bool(verbose),
            verbose_vmec2000_table=bool(verbose_vmec2000_table),
            badjac_use_state=bool(badjac_use_state),
            badjac_state_probe=bool(badjac_state_probe),
            badjac_initial_state_probe_iters=int(badjac_initial_state_probe_iters),
        )
    )
