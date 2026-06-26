"""Host-side VMEC2000 print/diagnostic contexts for residual iteration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from vmec_jax.solvers.fixed_boundary.diagnostics.io import (
    _format_residual_converged_message,
    _format_residual_iteration_update_message,
    _format_residual_physical_status_message,
)


@dataclass(frozen=True)
class Vmec2000PrintContext:
    """Bound host-side VMEC2000 row-print helpers for one residual solve."""

    nstep_screen: int
    print_iter_row: Callable[..., None]
    should_print: Callable[[int, int], bool]


@dataclass(frozen=True)
class VmecIterationScalars:
    """Screen/history scalars sampled from one residual iteration."""

    r00: float
    z00: float
    wb: float
    wp: float
    w_vmec: float


@dataclass(frozen=True)
class Vmec2000TimeControlCallbacks:
    """Callbacks required by one VMEC2000 time-control sample."""

    time_control_decision: Callable[..., Any]
    dump_time_control_trace: Callable[..., None]
    maybe_dump_checkpoint: Callable[..., None]
    maybe_dump_time_control: Callable[..., None]
    apply_controller_sample: Callable[..., Any]
    controller_sample: Callable[..., Any]
    host_restart_update: Callable[..., Any]
    host_restart_branch_result: Callable[..., Any]
    apply_restart_branch_result: Callable[..., Any]
    controller_restart_update: Callable[..., Any]


@dataclass(frozen=True)
class Vmec2000TimeControlRuntimeResult:
    """Scalar outputs from one VMEC2000 time-control runtime sample."""

    fsq: float
    fsq0: float
    pre_restart_reason: str
    restarted: bool


@dataclass(frozen=True)
class PreRestartTriggerCallbacks:
    """Callbacks required by one host restart-trigger branch."""

    host_update: Callable[..., Any]
    host_branch_result: Callable[..., Any]
    apply_restart_branch_result: Callable[..., Any]
    controller_restart_update: Callable[..., Any]
    print_compact_update_status: Callable[..., Any]
    preconditioner_diag_floats: Callable[..., Any]
    dump_xc_with_velocity_blocks: Callable[..., Any]


@dataclass(frozen=True)
class PreRestartTriggerRuntimeResult:
    """Scalar outputs from one optional pre-restart trigger branch."""

    applied: bool
    pre_restart_reason: str
    time_step_iter: float
    step_status: str


def resolve_vmec2000_print_context(
    *,
    cfg: Any,
    indata: Any,
    verbose: bool,
    vmec2000_control: bool,
    verbose_vmec2000_table: bool,
    getenv: Callable[[str, str], str],
    resolve_debug_print_config: Callable[..., Any],
    resolve_nstep_screen: Callable[..., int],
    emit_iter_row: Callable[..., None],
    should_print_row: Callable[..., bool],
    print_row: Callable[..., None],
) -> Vmec2000PrintContext:
    """Resolve row-printing policy and return bound print/cadence helpers."""

    debug_print_config = resolve_debug_print_config(
        print_env=getenv("VMEC_JAX_SCAN_PRINT", "1"),
        mode_env=getenv("VMEC_JAX_SCAN_PRINT_MODE", "debug_print"),
        ordered_env=getenv("VMEC_JAX_SCAN_PRINT_ORDERED", "0"),
    )
    scan_print_mode = debug_print_config.mode
    scan_print_ordered = debug_print_config.ordered
    print_live = debug_print_config.print_live
    jax_debug = None
    io_callback = None
    if print_live:
        try:
            from jax import debug as jax_debug  # type: ignore[assignment]
        except Exception:
            jax_debug = None
    if scan_print_mode == "io_callback":
        try:
            from jax.experimental import io_callback as io_callback  # type: ignore[assignment]
        except Exception:
            scan_print_mode = resolve_debug_print_config(
                print_env="1",
                mode_env=scan_print_mode,
                ordered_env="0",
                io_callback_available=False,
            ).mode
            io_callback = None

    nstep_screen = resolve_nstep_screen(
        indata_nstep=int(indata.get_int("NSTEP", 1)) if indata is not None else 1,
        override_env=getenv("VMEC_JAX_NSTEP_OVERRIDE", ""),
    )

    def print_iter_row(
        *,
        iter_idx: int,
        fsqr: float,
        fsqz: float,
        fsql: float,
        fsqr1: float,
        fsqz1: float,
        fsql1: float,
        delt0r: float,
        r00: float,
        w_mhd: float,
        z00: float | None = None,
    ) -> None:
        """Evaluate print iter row for fixed-boundary VMEC solve and implicit differentiation."""
        del fsqr1, fsqz1, fsql1
        emit_iter_row(
            iter_idx=iter_idx,
            fsqr=fsqr,
            fsqz=fsqz,
            fsql=fsql,
            delt0r=delt0r,
            r00=r00,
            w_mhd=w_mhd,
            lasym=bool(cfg.lasym),
            z00=z00,
            verbose=bool(verbose),
            vmec2000_control=bool(vmec2000_control),
            verbose_vmec2000_table=bool(verbose_vmec2000_table),
            print_live=bool(print_live),
            scan_print_mode=scan_print_mode,
            scan_print_ordered=bool(scan_print_ordered),
            jax_debug=jax_debug,
            io_callback=io_callback,
            print_row=print_row,
        )

    def should_print(iter_idx: int, max_iter: int) -> bool:
        """Evaluate should print for fixed-boundary VMEC solve and implicit differentiation."""
        return should_print_row(
            iter_idx=iter_idx,
            max_iter=max_iter,
            nstep_screen=nstep_screen,
            verbose=bool(verbose),
            vmec2000_control=bool(vmec2000_control),
            verbose_vmec2000_table=bool(verbose_vmec2000_table),
        )

    return Vmec2000PrintContext(
        nstep_screen=int(nstep_screen),
        print_iter_row=print_iter_row,
        should_print=should_print,
    )


def evaluate_vmec2000_time_control(
    *,
    iter2: int,
    iter1: int,
    fsq_prev: float,
    fsq0_curr: float,
    fsq0_prev: float,
    res0: float,
    res1: float,
    bad_jacobian: bool,
    vmec2000_fact: float,
    time_step: float,
    time_control_decision: Callable[..., Any],
    dump_time_control_trace: Callable[..., None],
    maybe_dump_checkpoint: Callable[..., None],
    maybe_dump_time_control: Callable[..., None],
) -> Any:
    """Evaluate VMEC2000 time control and emit legacy-compatible traces."""

    tc = time_control_decision(
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
    trace_args = dict(
        iter2=int(iter2),
        iter1=int(iter1),
        fsq=float(tc.fsq),
        fsq0=float(tc.fsq0),
        res0=float(tc.res0),
        res1=float(tc.res1),
        time_step=float(time_step),
    )
    checkpoint_args = dict(
        iter_idx=int(iter2),
        fsq=float(tc.fsq),
        fsq0=float(tc.fsq0),
        res0=float(tc.res0),
        res1=float(tc.res1),
    )
    if bool(tc.initialized):
        dump_time_control_trace(stage="init", irst=int(tc.trace_irst), **trace_args)
        maybe_dump_checkpoint(**checkpoint_args)
    dump_time_control_trace(stage="pre", irst=int(tc.trace_irst), **trace_args)
    if bool(tc.store_checkpoint):
        dump_time_control_trace(stage="checkpoint", irst=int(tc.trace_irst), **trace_args)
        maybe_dump_checkpoint(**checkpoint_args)
    if bool(tc.restart):
        maybe_dump_time_control(time_step=float(time_step), **checkpoint_args)
        dump_time_control_trace(stage="restart", irst=int(tc.irst), **trace_args)
    return tc


def run_vmec2000_time_control_runtime(
    *,
    vmec2000_control: bool,
    skip_time_control: bool,
    iter2: int,
    iter1: int,
    fsq_prev: float,
    fsq0_curr: float,
    fsq0_prev: float,
    res0: float,
    res1: float,
    bad_jacobian: bool,
    vmec2000_fact: float,
    time_step: float,
    restart_badjac_factor: float,
    restart_badprog_factor: float,
    ijacob: int,
    bad_resets: int,
    fsq_prev_before: float,
    fsq0_prev_before: float,
    k_ndamp: int,
    state_checkpoint: Any,
    prev_rz_fsq_before: float,
    callbacks: Vmec2000TimeControlCallbacks,
) -> Vmec2000TimeControlRuntimeResult:
    """Run VMEC2000 time control and apply restart side effects if needed."""

    if (not bool(vmec2000_control)) or bool(skip_time_control):
        return Vmec2000TimeControlRuntimeResult(float(fsq_prev), float(fsq0_curr), "none", False)

    tc = evaluate_vmec2000_time_control(
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
        time_control_decision=callbacks.time_control_decision,
        dump_time_control_trace=callbacks.dump_time_control_trace,
        maybe_dump_checkpoint=callbacks.maybe_dump_checkpoint,
        maybe_dump_time_control=callbacks.maybe_dump_time_control,
    )
    callbacks.apply_controller_sample(callbacks.controller_sample, tc)
    if not bool(tc.restart):
        return Vmec2000TimeControlRuntimeResult(float(tc.fsq), float(tc.fsq0), "none", False)

    restart_update = callbacks.host_restart_update(
        irst=int(tc.irst),
        time_step=float(time_step),
        restart_badjac_factor=float(restart_badjac_factor),
        restart_badprog_factor=float(restart_badprog_factor),
        ijacob=int(ijacob),
        bad_resets=int(bad_resets),
        iter2=int(iter2),
        fsq_prev_before=float(fsq_prev_before),
        fsq0_prev_before=float(fsq0_prev_before),
        k_ndamp=int(k_ndamp),
    )
    restart_branch = callbacks.host_restart_branch_result(
        state_checkpoint=state_checkpoint,
        restart_update=restart_update,
        pre_restart_reason=tc.pre_restart_reason,
        prev_rz_fsq_before=prev_rz_fsq_before,
    )
    callbacks.apply_restart_branch_result(
        restart_branch,
        callbacks.controller_restart_update,
        time_step_value=float(time_step),
    )
    return Vmec2000TimeControlRuntimeResult(float(tc.fsq), float(tc.fsq0), str(tc.pre_restart_reason), True)


def run_pre_restart_trigger_runtime(
    *,
    use_restart_triggers: bool,
    pre_restart_reason: str,
    huge_initial_forces: bool,
    huge_force_restart_count: int,
    time_step: float,
    restart_badjac_factor: float,
    restart_badprog_factor: float,
    stage_transition_scale: float,
    step_size: float,
    ijacob: int,
    bad_resets: int,
    iter2: int,
    compact_iter_idx: int,
    fsq_prev_before: float,
    fsq0_prev_before: float,
    k_ndamp: int,
    state_checkpoint: Any,
    state_before_restart: Any,
    velocity_blocks_before: Any,
    static: Any,
    prev_rz_fsq_before: float,
    vmec2000_control: bool,
    verbose: bool,
    verbose_vmec2000_table: bool,
    callbacks: PreRestartTriggerCallbacks,
) -> PreRestartTriggerRuntimeResult:
    """Apply VMEC-style optional restart triggers after residual diagnostics.

    The residual loop owns timing and loop control.  This helper owns the
    branch-local controller/update callbacks so the large loop body stays
    readable while preserving VMEC2000 restart semantics.
    """

    if (not bool(use_restart_triggers)) or str(pre_restart_reason) == "none":
        return PreRestartTriggerRuntimeResult(False, str(pre_restart_reason), float(time_step), "none")

    pre_restart_update = callbacks.host_update(
        pre_restart_reason=str(pre_restart_reason),
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
        fsq_prev_before=float(fsq_prev_before),
        fsq0_prev_before=float(fsq0_prev_before),
        k_ndamp=int(k_ndamp),
    )
    pre_restart_branch = callbacks.host_branch_result(
        state_checkpoint=state_checkpoint,
        pre_restart_update=pre_restart_update,
        pre_restart_reason=str(pre_restart_reason),
        prev_rz_fsq_before=prev_rz_fsq_before,
        vmec2000_control=bool(vmec2000_control),
    )
    callbacks.apply_restart_branch_result(
        pre_restart_branch,
        callbacks.controller_restart_update,
        time_step_value=pre_restart_branch.time_step_iter,
    )
    callbacks.print_compact_update_status(
        verbose=bool(verbose),
        vmec2000_control=bool(vmec2000_control),
        verbose_vmec2000_table=bool(verbose_vmec2000_table),
        precond_diag_floats=callbacks.preconditioner_diag_floats,
        iter_idx=int(compact_iter_idx),
        dt_eff=0.0,
        update_rms=0.0,
        step_status=pre_restart_branch.step_status,
    )
    callbacks.dump_xc_with_velocity_blocks(
        state=state_before_restart,
        velocities=velocity_blocks_before,
        static=static,
        iter_idx=int(iter2),
    )
    return PreRestartTriggerRuntimeResult(
        True,
        pre_restart_branch.pre_restart_reason,
        float(pre_restart_branch.time_step_iter),
        pre_restart_branch.step_status,
    )


def dump_residual_evolve_trace(
    *,
    dump_evolve_trace: Callable[..., None],
    iter2: int,
    iter1: int,
    stage: str,
    fsq1: float,
    fsq_prev: float,
    time_step: float,
    dtau: float,
    b1: float,
    fac: float,
    state: Any,
    velocities: Any,
    forces: Any | None = None,
) -> None:
    """Route residual evolve traces through compact velocity/force blocks."""

    kwargs = {
        "iter2": int(iter2),
        "iter1": int(iter1),
        "stage": str(stage),
        "fsq1_val": float(fsq1),
        "fsq_prev_val": float(fsq_prev),
        "time_step_val": float(time_step),
        "dtau_val": float(dtau),
        "b1_val": float(b1),
        "fac_val": float(fac),
        "state_val": state,
        "vRcc_val": velocities.rcc,
        "vRss_val": velocities.rss,
        "vRsc_val": velocities.rsc,
        "vRcs_val": velocities.rcs,
        "vZsc_val": velocities.zsc,
        "vZcs_val": velocities.zcs,
        "vZcc_val": velocities.zcc,
        "vZss_val": velocities.zss,
        "vLsc_val": velocities.lsc,
        "vLcs_val": velocities.lcs,
        "vLcc_val": velocities.lcc,
        "vLss_val": velocities.lss,
    }
    if forces is not None:
        kwargs.update(
            {
                "frcc_val": forces.rcc,
                "frss_val": forces.rss,
                "frsc_val": forces.rsc,
                "frcs_val": forces.rcs,
                "fzsc_val": forces.zsc,
                "fzcs_val": forces.zcs,
                "fzcc_val": forces.zcc,
                "fzss_val": forces.zss,
                "flsc_val": forces.lsc,
                "flcs_val": forces.lcs,
                "flcc_val": forces.lcc,
                "flss_val": forces.lss,
            }
        )
    dump_evolve_trace(**kwargs)


def sample_vmec_iteration_scalars(
    *,
    need_scalar: bool,
    k: Any,
    state: Any,
    norms_current: Any,
    m0_mask: Any,
    lasym: bool,
    host_update_assembly: bool,
    vmec2000_control: bool,
    gamma: float,
    twopi: float,
    previous_r00: float,
    previous_z00: float,
    previous_wb: float,
    previous_wp: float,
    tree_has_tracer: Callable[[Any], bool],
    device_get_floats: Callable[..., tuple[float, ...]],
    jnp_module: Any,
) -> VmecIterationScalars:
    """Sample VMEC screen/history scalars with host/device parity rules."""

    if not bool(need_scalar):
        r00_val = float(previous_r00)
        z00_val = float(previous_z00)
        wb_val = float(previous_wb)
        wp_val = float(previous_wp)
    elif bool(host_update_assembly) and (not tree_has_tracer(k)):
        try:
            r00_val = float(np.asarray(k.pr1_even)[0, 0, 0])
            z00_val = float(np.asarray(k.pz1_even)[0, 0, 0]) if bool(lasym) else 0.0
        except Exception:
            if not np.any(m0_mask):
                r00_val = float("nan")
                z00_val = float("nan")
            else:
                r00_val = float(np.sum(np.asarray(state.Rcos)[0, m0_mask]))
                z00_val = float(np.sum(np.asarray(state.Zcos)[0, m0_mask])) if bool(lasym) else 0.0
        wb_val = float(np.asarray(norms_current.wb))
        wp_val = float(np.asarray(norms_current.wp))
    else:
        try:
            r00_j = jnp_module.asarray(k.pr1_even)[0, 0, 0]
            if bool(lasym):
                z00_j = jnp_module.asarray(k.pz1_even)[0, 0, 0]
            else:
                z00_j = jnp_module.asarray(0.0, dtype=jnp_module.asarray(r00_j).dtype)
        except Exception:
            if not np.any(m0_mask):
                r00_j = jnp_module.asarray(float("nan"))
                z00_j = jnp_module.asarray(float("nan"))
            else:
                r00_j = jnp_module.sum(jnp_module.asarray(state.Rcos)[0, m0_mask])
                if bool(lasym):
                    z00_j = jnp_module.sum(jnp_module.asarray(state.Zcos)[0, m0_mask])
                else:
                    z00_j = jnp_module.asarray(0.0, dtype=jnp_module.asarray(r00_j).dtype)
        # `norms_current` reflects the current bcovar state and therefore
        # matches VMEC's printed wb/wp even when the preconditioner norm is cached.
        wb_j = jnp_module.asarray(norms_current.wb)
        wp_j = jnp_module.asarray(norms_current.wp)
        r00_val, z00_val, wb_val, wp_val = device_get_floats(r00_j, z00_j, wb_j, wp_j)

    if bool(vmec2000_control):
        # Match VMEC's printed precision (E11.3) for parity checks.
        r00_val = float(f"{float(r00_val):.3E}")
        z00_val = float(f"{float(z00_val):.3E}")
    w_vmec = (float(wb_val) + float(wp_val) / (float(gamma) - 1.0)) * float(twopi) * float(twopi)
    return VmecIterationScalars(
        r00=float(r00_val),
        z00=float(z00_val),
        wb=float(wb_val),
        wp=float(wp_val),
        w_vmec=float(w_vmec),
    )


def print_compact_physical_residual_status(
    *,
    verbose: bool,
    vmec2000_control: bool,
    verbose_vmec2000_table: bool,
    iter_idx: int,
    fsqr: float,
    fsqz: float,
    fsql: float,
    include_edge: bool,
    print_func: Callable[..., None] = print,
) -> bool:
    """Print the compact non-VMEC physical residual status line."""

    if not (bool(verbose) and not (bool(vmec2000_control) and bool(verbose_vmec2000_table))):
        return False
    print_func(
        _format_residual_physical_status_message(
            iter_idx=iter_idx,
            fsqr=fsqr,
            fsqz=fsqz,
            fsql=fsql,
            include_edge=include_edge,
        ),
        flush=True,
    )
    return True


def print_compact_converged_status(
    *,
    verbose: bool,
    vmec2000_control: bool,
    verbose_vmec2000_table: bool,
    fsqr: float,
    fsqz: float,
    fsql: float,
    target: float,
    print_func: Callable[..., None] = print,
) -> bool:
    """Print the compact non-VMEC convergence status line."""

    if not (bool(verbose) and not (bool(vmec2000_control) and bool(verbose_vmec2000_table))):
        return False
    print_func(
        _format_residual_converged_message(
            fsqr=fsqr,
            fsqz=fsqz,
            fsql=fsql,
            target=target,
        ),
        flush=True,
    )
    return True


def print_compact_residual_iteration_update_status(
    *,
    verbose: bool,
    vmec2000_control: bool,
    verbose_vmec2000_table: bool,
    precond_diag_floats: Callable[[], tuple[float, float, float]],
    iter_idx: int,
    dt_eff: float,
    update_rms: float,
    step_status: str,
    print_func: Callable[..., None] = print,
) -> bool:
    """Print compact update status only on the non-VMEC table path."""

    if not (bool(verbose) and not (bool(vmec2000_control) and bool(verbose_vmec2000_table))):
        return False
    fsqr1_f, fsqz1_f, fsql1_f = precond_diag_floats()
    print_func(
        _format_residual_iteration_update_message(
            iter_idx=int(iter_idx),
            dt_eff=dt_eff,
            update_rms=update_rms,
            fsqr1=fsqr1_f,
            fsqz1=fsqz1_f,
            fsql1=fsql1_f,
            step_status=step_status,
        ),
        flush=True,
    )
    return True


def residual_update_rms_for_print(
    *,
    verbose: bool,
    strict_update: bool,
    update_rms_j: Any,
    update_rms: Any,
) -> float:
    """Return the scalar update RMS shown in residual-iteration logs."""

    if not bool(verbose):
        return 0.0
    return float(np.asarray(update_rms_j)) if bool(strict_update) else float(update_rms)


def print_residual_iteration_update_status(
    *,
    verbose: bool,
    vmec2000_control: bool,
    verbose_vmec2000_table: bool,
    should_print_vmec2000: Callable[[int, int], bool],
    print_vmec2000_iter_row: Callable[..., None],
    precond_diag_floats: Callable[[], tuple[float, float, float]],
    iter_idx: int,
    max_iter: int,
    compact_iter_idx: int,
    fsqr: float,
    fsqz: float,
    fsql: float,
    dt_eff: float,
    update_rms: float,
    time_step: float,
    r00: float,
    z00: float,
    w_mhd: float,
    step_status: str,
    force_vmec2000_row: bool = False,
    compact_status: bool = True,
    print_func: Callable[..., None] = print,
) -> bool:
    """Print either a VMEC2000 row or a compact residual update line."""

    if not bool(verbose):
        return False
    if bool(vmec2000_control) and bool(verbose_vmec2000_table):
        if (not bool(force_vmec2000_row)) and (not should_print_vmec2000(int(iter_idx), int(max_iter))):
            return False
        fsqr1_f, fsqz1_f, fsql1_f = precond_diag_floats()
        print_vmec2000_iter_row(
            iter_idx=int(iter_idx),
            fsqr=fsqr,
            fsqz=fsqz,
            fsql=fsql,
            fsqr1=fsqr1_f,
            fsqz1=fsqz1_f,
            fsql1=fsql1_f,
            delt0r=float(time_step),
            r00=float(r00),
            w_mhd=float(w_mhd),
            z00=float(z00),
        )
        return True

    if not bool(compact_status):
        return False
    fsqr1_f, fsqz1_f, fsql1_f = precond_diag_floats()
    print_func(
        _format_residual_iteration_update_message(
            iter_idx=int(compact_iter_idx),
            dt_eff=dt_eff,
            update_rms=update_rms,
            fsqr1=fsqr1_f,
            fsqz1=fsqz1_f,
            fsql1=fsql1_f,
            step_status=step_status,
        ),
        flush=True,
    )
    return True


__all__ = [
    "Vmec2000TimeControlCallbacks",
    "Vmec2000TimeControlRuntimeResult",
    "Vmec2000PrintContext",
    "VmecIterationScalars",
    "dump_residual_evolve_trace",
    "evaluate_vmec2000_time_control",
    "print_compact_converged_status",
    "print_compact_physical_residual_status",
    "print_compact_residual_iteration_update_status",
    "print_residual_iteration_update_status",
    "residual_update_rms_for_print",
    "resolve_vmec2000_print_context",
    "run_vmec2000_time_control_runtime",
    "sample_vmec_iteration_scalars",
]
