"""Scalar time-control helpers for the VMEC2000 scan loop."""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

from ...._compat import jnp


RESTART_NONE = 0
RESTART_BADJAC = 1
RESTART_STAGE = 2
RESTART_BADPROG_VMEC = 3
RESTART_TIME = 4


def _restart_code(value: int) -> Any:
    return jnp.asarray(value, dtype=jnp.int32)


class ScanTimeControlScalars(NamedTuple):
    res0: Any
    res1: Any
    checkpoint_mask: Any


class ScanCheckpointResiduals(NamedTuple):
    """Residual diagnostics stored with a VMEC scan restart checkpoint."""

    fsqr: Any
    fsqz: Any
    fsql: Any
    fsqr1: Any
    fsqz1: Any
    fsql1: Any


class ScanCheckpointUpdate(NamedTuple):
    """State and residual values stored at the best accepted scan checkpoint."""

    state_checkpoint: Any
    residuals: ScanCheckpointResiduals


class ScanRestartDecision(NamedTuple):
    restart_time: Any
    vmecpp_bad_progress: Any
    stage_spike: Any
    do_restart: Any
    restart_reason: Any
    irst_restart: Any


class ScanRestartTransition(NamedTuple):
    time_step: Any
    damping_time_step: Any
    iter_offset: Any
    iter1: Any
    ijacob: Any
    bad_resets: Any
    bad_growth: Any
    force_bcovar_update: Any


class ScanStageSpikePostScalars(NamedTuple):
    time_step: Any
    apply_stage_reset: Any


class ScanStageSpikePostUpdate(NamedTuple):
    time_step: Any
    inv_tau: Any
    velocity_blocks: tuple[Any, ...]
    iter1: Any


class ScanPostRestartUpdate(NamedTuple):
    """Scan fields after restart/no-restart selection and stage-spike reset."""

    state: Any
    time_step: Any
    inv_tau: Any
    fsq_prev: Any
    velocity_blocks: tuple[Any, ...]
    iter_offset: Any
    iter1: Any
    ijacob: Any
    bad_resets: Any
    bad_growth: Any
    force_bcovar_update: Any


class ScanTimeRestartEvaluation(NamedTuple):
    """Time-control, checkpoint, and restart state for one scan step."""

    fsq_phys: Any
    fsq: Any
    res0: Any
    res1: Any
    checkpoint_update: ScanCheckpointUpdate
    restart_decision: ScanRestartDecision
    restart_update: ScanPostRestartUpdate
    fsq0_prev_post: Any


class ScanFallbackProbeUpdate(NamedTuple):
    probe_count: Any
    probe_bad_jac: Any
    probe_accept: Any
    probe_fsq_start: Any
    probe_fsq_min: Any
    probe_fsq_max: Any
    abort_scan: Any


def scan_time_control_scalars(
    *,
    skip_timecontrol: Any,
    init_mask: Any,
    fsq: Any,
    fsq_res: Any,
    fsq_phys: Any,
    fsq1: Any,
    fsq_prev_before: Any,
    res0_prev: Any,
    res1_prev: Any,
    bad_jacobian: Any,
    vmec2000_control: bool,
) -> ScanTimeControlScalars:
    """Update scalar residual trackers and checkpoint mask for scan time control."""
    res0_tc = jnp.where(init_mask, fsq_res, res0_prev)
    res1_tc = jnp.where(init_mask, fsq_phys, res1_prev)
    if bool(vmec2000_control):
        res0_tc = jnp.minimum(res0_tc, fsq)
    else:
        res0_tc = jnp.minimum(res0_tc, fsq_res)
    res1_tc = jnp.minimum(res1_tc, fsq_phys)
    checkpoint_mask_tc = (fsq <= res0_tc) & (fsq_phys <= res1_tc) & (~bad_jacobian)

    res0_skip = jnp.where(
        (fsq1 <= fsq_prev_before) & jnp.isfinite(fsq1),
        jnp.minimum(res0_prev, fsq1),
        res0_prev,
    )
    res0 = jnp.where(skip_timecontrol, res0_skip, res0_tc)
    res1 = jnp.where(skip_timecontrol, res1_prev, res1_tc)
    checkpoint_mask = jnp.where(skip_timecontrol, jnp.asarray(False), checkpoint_mask_tc)
    return ScanTimeControlScalars(res0=res0, res1=res1, checkpoint_mask=checkpoint_mask)


def scan_checkpoint_update(
    *,
    skip_timecontrol: Any,
    init_mask: Any,
    checkpoint_mask: Any,
    current_state: Any,
    previous_state_checkpoint: Any,
    current_residuals: ScanCheckpointResiduals,
    previous_residuals: ScanCheckpointResiduals,
    cond_func: Any,
) -> ScanCheckpointUpdate:
    """Materialize the VMEC scan checkpoint selected by time-control scalars.

    VMEC stores a restart checkpoint when the residual improves.  The state
    itself can be large, so this helper keeps the state switch as scalar
    conditionals and only uses elementwise selection for the small diagnostic
    residual scalars.
    """
    state_checkpoint_init = cond_func(
        (~skip_timecontrol) & init_mask,
        lambda _: current_state,
        lambda _: previous_state_checkpoint,
        operand=None,
    )
    state_checkpoint = cond_func(
        checkpoint_mask,
        lambda _: current_state,
        lambda _: state_checkpoint_init,
        operand=None,
    )
    return ScanCheckpointUpdate(
        state_checkpoint=state_checkpoint,
        residuals=ScanCheckpointResiduals(
            fsqr=jnp.where(checkpoint_mask, current_residuals.fsqr, previous_residuals.fsqr),
            fsqz=jnp.where(checkpoint_mask, current_residuals.fsqz, previous_residuals.fsqz),
            fsql=jnp.where(checkpoint_mask, current_residuals.fsql, previous_residuals.fsql),
            fsqr1=jnp.where(checkpoint_mask, current_residuals.fsqr1, previous_residuals.fsqr1),
            fsqz1=jnp.where(checkpoint_mask, current_residuals.fsqz1, previous_residuals.fsqz1),
            fsql1=jnp.where(checkpoint_mask, current_residuals.fsql1, previous_residuals.fsql1),
        ),
    )


def scan_restart_decision(
    *,
    skip_timecontrol: Any,
    iter2: Any,
    iter1: Any,
    fsq: Any,
    fsq_phys: Any,
    res0: Any,
    res1: Any,
    bad_jacobian: Any,
    fsqr: Any,
    fsqz: Any,
    vmec2000_fact: float,
    use_restart_triggers: bool,
    vmecpp_restart: bool,
    k_preconditioner_update_interval: int,
    stage_prev_fsq: Any | None,
    stage_transition_factor: float,
    vmec2000_control: bool,
) -> ScanRestartDecision:
    """Select scan restart reason codes from scalar residual state."""
    restart_time = (~bad_jacobian) & ((iter2 - iter1) > 10)
    restart_time = restart_time & (
        (fsq > vmec2000_fact * jnp.maximum(res0, 1.0e-30)) | (fsq_phys > vmec2000_fact * jnp.maximum(res1, 1.0e-30))
    )

    vmecpp_bad_progress = jnp.asarray(False)
    if bool(vmecpp_restart):
        vmecpp_bad_progress = (
            ((iter2 - iter1) > (int(k_preconditioner_update_interval) // 2))
            & (iter2 > 2 * int(k_preconditioner_update_interval))
            & ((fsqr + fsqz) > 1.0e-2)
        )

    stage_spike = jnp.asarray(False)
    if stage_prev_fsq is not None:
        stage_spike = (iter2 == 1) & (fsq_phys > (stage_prev_fsq * stage_transition_factor))

    restart_none = _restart_code(RESTART_NONE)
    restart_badjac = _restart_code(RESTART_BADJAC)
    restart_stage = _restart_code(RESTART_STAGE)
    restart_badprog = _restart_code(RESTART_BADPROG_VMEC)
    restart_time_code = _restart_code(RESTART_TIME)

    if bool(vmec2000_control):
        pre_reason = jnp.where(stage_spike, restart_stage, restart_none)
        pre_reason = jnp.where(bad_jacobian & (iter2 > iter1), restart_badjac, pre_reason)
        pre_reason = jnp.where((pre_reason == restart_none) & vmecpp_bad_progress, restart_badprog, pre_reason)
        do_restart = restart_time | (bool(use_restart_triggers) & (pre_reason != restart_none))
        restart_reason = jnp.where(restart_time, restart_time_code, pre_reason)
    else:
        restart_badjac_mask = bool(use_restart_triggers) & bad_jacobian & (iter2 > iter1)
        restart_vmecpp = bool(use_restart_triggers) & vmecpp_bad_progress
        do_restart = restart_time | restart_badjac_mask | restart_vmecpp
        restart_reason = jnp.where(
            restart_time, restart_time_code, jnp.where(restart_badjac_mask, restart_badjac, restart_badprog)
        )

    do_restart = jnp.where(skip_timecontrol, jnp.asarray(False), do_restart)
    restart_reason = jnp.where(skip_timecontrol, restart_none, restart_reason)
    restart_time = jnp.where(skip_timecontrol, jnp.asarray(False), restart_time)

    irst_restart = jnp.where(
        restart_reason == restart_badjac,
        jnp.asarray(2, dtype=jnp.int32),
        jnp.where(
            (restart_reason == restart_time_code) | (restart_reason == restart_badprog),
            jnp.asarray(3, dtype=jnp.int32),
            jnp.asarray(1, dtype=jnp.int32),
        ),
    )
    return ScanRestartDecision(
        restart_time=restart_time,
        vmecpp_bad_progress=vmecpp_bad_progress,
        stage_spike=stage_spike,
        do_restart=do_restart,
        restart_reason=restart_reason,
        irst_restart=irst_restart,
    )


def scan_restart_transition(
    *,
    time_step: Any,
    iter_offset: Any,
    ijacob: Any,
    bad_resets: Any,
    iter2: Any,
    restart_reason: Any,
    vmec2000_control: bool,
    restart_badjac_factor: float,
    restart_badprog_factor: float,
    stage_transition_scale: float,
    step_size: float,
) -> ScanRestartTransition:
    """Compute scalar updates for a taken scan restart."""
    dtype = getattr(time_step, "dtype", None)
    restart_badjac = _restart_code(RESTART_BADJAC)
    restart_stage = _restart_code(RESTART_STAGE)
    restart_badprog = _restart_code(RESTART_BADPROG_VMEC)
    restart_time_code = _restart_code(RESTART_TIME)

    if bool(vmec2000_control):
        next_time_step = jnp.where(restart_reason == restart_badjac, restart_badjac_factor * time_step, time_step)
        next_time_step = jnp.where(
            (restart_reason == restart_time_code) | (restart_reason == restart_badprog),
            next_time_step / restart_badprog_factor,
            next_time_step,
        )
        next_time_step = jnp.where(
            restart_reason == restart_stage,
            next_time_step * jnp.asarray(stage_transition_scale, dtype=dtype),
            next_time_step,
        )
        next_iter_offset = iter_offset
    else:
        next_time_step = jnp.where(
            restart_reason == restart_badjac,
            restart_badjac_factor * time_step,
            time_step / restart_badprog_factor,
        )
        next_iter_offset = iter_offset - jnp.asarray(1, dtype=jnp.int32)

    next_time_step = jnp.maximum(next_time_step, jnp.asarray(1.0e-12, dtype=dtype))
    damping_time_step = next_time_step
    if bool(vmec2000_control):
        next_ijacob = jnp.where(restart_reason == restart_badjac, ijacob + 1, ijacob)
        ijacob25 = next_ijacob == jnp.asarray(25, dtype=next_ijacob.dtype)
        ijacob50 = next_ijacob == jnp.asarray(50, dtype=next_ijacob.dtype)
        next_time_step = jnp.where(
            ijacob25 & (restart_reason == restart_badjac),
            jnp.asarray(0.98, dtype=dtype) * jnp.asarray(float(step_size), dtype=dtype),
            next_time_step,
        )
        next_time_step = jnp.where(
            ijacob50 & (restart_reason == restart_badjac),
            jnp.asarray(0.96, dtype=dtype) * jnp.asarray(float(step_size), dtype=dtype),
            next_time_step,
        )
    else:
        next_ijacob = jnp.where(restart_reason == restart_badjac, ijacob + 1, ijacob)

    return ScanRestartTransition(
        time_step=next_time_step,
        damping_time_step=damping_time_step,
        iter_offset=next_iter_offset,
        iter1=iter2,
        ijacob=next_ijacob,
        bad_resets=bad_resets + 1,
        bad_growth=jnp.asarray(0, dtype=jnp.int32),
        force_bcovar_update=jnp.asarray(True),
    )


def scan_stage_spike_post_scalars(
    *,
    time_step: Any,
    stage_spike: Any,
    stage_prev_fsq: Any | None,
    stage_transition_scale: float,
) -> ScanStageSpikePostScalars:
    """Apply the scan loop's post-restart stage-spike scalar time-step scaling."""
    dtype = getattr(time_step, "dtype", None)
    apply_stage_reset = jnp.asarray(False) if stage_prev_fsq is None else stage_spike
    next_time_step = jnp.where(
        apply_stage_reset,
        jnp.maximum(
            time_step * jnp.asarray(stage_transition_scale, dtype=dtype),
            jnp.asarray(1.0e-12, dtype=dtype),
        ),
        time_step,
    )
    return ScanStageSpikePostScalars(time_step=next_time_step, apply_stage_reset=apply_stage_reset)


def scan_stage_spike_post_update(
    *,
    time_step: Any,
    inv_tau: Any,
    velocity_blocks: tuple[Any, ...],
    iter1: Any,
    iter2: Any,
    stage_spike: Any,
    stage_prev_fsq: Any | None,
    stage_transition_scale: float,
    k_ndamp: int,
    dtype: Any,
) -> ScanStageSpikePostUpdate:
    """Apply VMEC stage-spike damping reset to scalar and velocity state."""

    scalars = scan_stage_spike_post_scalars(
        time_step=time_step,
        stage_spike=stage_spike,
        stage_prev_fsq=stage_prev_fsq,
        stage_transition_scale=stage_transition_scale,
    )
    if stage_prev_fsq is None:
        return ScanStageSpikePostUpdate(
            time_step=scalars.time_step,
            inv_tau=inv_tau,
            velocity_blocks=tuple(velocity_blocks),
            iter1=iter1,
        )

    reset = scalars.apply_stage_reset
    inv_tau_next = jnp.where(
        reset,
        jnp.full((int(k_ndamp),), jnp.asarray(0.15, dtype=dtype) / scalars.time_step),
        inv_tau,
    )
    velocity_next = tuple(jnp.where(reset, jnp.zeros_like(block), block) for block in velocity_blocks)
    iter1_next = jnp.where(reset, iter2, iter1)
    return ScanStageSpikePostUpdate(
        time_step=scalars.time_step,
        inv_tau=inv_tau_next,
        velocity_blocks=velocity_next,
        iter1=iter1_next,
    )


def scan_post_restart_update(
    *,
    do_restart: Any,
    restart_updates_fn: Any,
    no_restart_updates_fn: Any,
    cond_func: Any,
    iter2: Any,
    stage_spike: Any,
    stage_prev_fsq: Any | None,
    stage_transition_scale: float,
    k_ndamp: int,
    dtype: Any,
) -> ScanPostRestartUpdate:
    """Select restart/no-restart update fields and apply stage-spike reset."""

    selected = cond_func(do_restart, restart_updates_fn, no_restart_updates_fn, operand=None)
    (
        state_post,
        time_step_post,
        inv_tau_post,
        fsq_prev_post,
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
        iter_offset_post,
        iter1_post,
        ijacob_post,
        bad_resets_post,
        bad_growth_post,
        force_bcovar_post,
    ) = selected

    stage_update = scan_stage_spike_post_update(
        time_step=time_step_post,
        inv_tau=inv_tau_post,
        velocity_blocks=(
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
        iter1=iter1_post,
        iter2=iter2,
        stage_spike=stage_spike,
        stage_prev_fsq=stage_prev_fsq,
        stage_transition_scale=stage_transition_scale,
        k_ndamp=k_ndamp,
        dtype=dtype,
    )
    return ScanPostRestartUpdate(
        state=state_post,
        time_step=stage_update.time_step,
        inv_tau=stage_update.inv_tau,
        fsq_prev=fsq_prev_post,
        velocity_blocks=stage_update.velocity_blocks,
        iter_offset=iter_offset_post,
        iter1=stage_update.iter1,
        ijacob=ijacob_post,
        bad_resets=bad_resets_post,
        bad_growth=bad_growth_post,
        force_bcovar_update=force_bcovar_post,
    )


def evaluate_scan_time_control_restart(
    *,
    carry_adv: Any,
    iter2: Any,
    fsqr: Any,
    fsqz: Any,
    fsql: Any,
    fsqr1: Any,
    fsqz1: Any,
    fsql1: Any,
    fsq0: Any,
    fsq1: Any,
    fsq_prev_before: Any,
    fsq0_prev_before: Any,
    bad_jacobian: Any,
    skip_timecontrol: Any,
    vmec2000_control: bool,
    reference_mode: bool,
    use_apply_payload_fusion: bool,
    dump_timecontrol_scan: bool,
    scan_timecontrol_dumper: Callable[..., Any],
    vmec2000_fact: float,
    use_restart_triggers: bool,
    vmecpp_restart: bool,
    k_preconditioner_update_interval: int,
    stage_prev_fsq: Any | None,
    stage_transition_factor: float,
    restart_badjac_factor: float,
    restart_badprog_factor: float,
    stage_transition_scale: float,
    step_size: float,
    k_ndamp: int,
    dtype: Any,
    restart_updates_func: Callable[..., Any],
    no_restart_updates_func: Callable[..., Any],
    scan_restart_transition_func: Callable[..., Any],
    cond_func: Callable[..., Any],
) -> ScanTimeRestartEvaluation:
    """Apply VMEC2000 time-control, checkpoint, and restart selection."""

    fsq_phys = fsq0
    if bool(vmec2000_control):
        fsq_phys = jnp.where(
            bad_jacobian & (iter2 > carry_adv.iter1),
            fsq0_prev_before,
            fsq_phys,
        )
        # VMEC2000 TimeStepControl uses the previous preconditioned residual.
        fsq = carry_adv.fsq_prev
        fsq_res = fsq
    elif not bool(use_apply_payload_fusion):
        fsq_res = jnp.where(jnp.asarray(reference_mode), fsq_phys, fsq1)
        fsq = fsq_res
    else:
        fsq_res = fsq1
        fsq = fsq1

    init_mask = (iter2 == carry_adv.iter1) | (carry_adv.res0 < 0.0) | (carry_adv.res1 < 0.0)
    tc_scalars = scan_time_control_scalars(
        skip_timecontrol=skip_timecontrol,
        init_mask=init_mask,
        fsq=fsq,
        fsq_res=fsq_res,
        fsq_phys=fsq_phys,
        fsq1=fsq1,
        fsq_prev_before=fsq_prev_before,
        res0_prev=carry_adv.res0,
        res1_prev=carry_adv.res1,
        bad_jacobian=bad_jacobian,
        vmec2000_control=bool(vmec2000_control),
    )
    res0 = tc_scalars.res0
    res1 = tc_scalars.res1
    checkpoint_mask = tc_scalars.checkpoint_mask
    checkpoint_update = scan_checkpoint_update(
        skip_timecontrol=skip_timecontrol,
        init_mask=init_mask,
        checkpoint_mask=checkpoint_mask,
        current_state=carry_adv.state,
        previous_state_checkpoint=carry_adv.state_checkpoint,
        current_residuals=ScanCheckpointResiduals(fsqr, fsqz, fsql, fsqr1, fsqz1, fsql1),
        previous_residuals=ScanCheckpointResiduals(
            carry_adv.fsqr_checkpoint,
            carry_adv.fsqz_checkpoint,
            carry_adv.fsql_checkpoint,
            carry_adv.fsqr1_checkpoint,
            carry_adv.fsqz1_checkpoint,
            carry_adv.fsql1_checkpoint,
        ),
        cond_func=cond_func,
    )
    if bool(dump_timecontrol_scan):
        for cond, stage_id, irst in (
            ((~skip_timecontrol) & init_mask, 0, 1),
            (~skip_timecontrol, 1, 1),
            (checkpoint_mask, 2, 1),
        ):
            scan_timecontrol_dumper(
                cond=cond,
                stage_id=jnp.asarray(stage_id, dtype=jnp.int32),
                iter2=iter2,
                iter1=carry_adv.iter1,
                fsq=fsq,
                fsq0=fsq_phys,
                res0=res0,
                res1=res1,
                time_step=carry_adv.time_step,
                irst=jnp.asarray(irst, dtype=jnp.int32),
            )

    restart_decision = scan_restart_decision(
        skip_timecontrol=skip_timecontrol,
        iter2=iter2,
        iter1=carry_adv.iter1,
        fsq=fsq,
        fsq_phys=fsq_phys,
        res0=res0,
        res1=res1,
        bad_jacobian=bad_jacobian,
        fsqr=fsqr,
        fsqz=fsqz,
        vmec2000_fact=vmec2000_fact,
        use_restart_triggers=bool(use_restart_triggers),
        vmecpp_restart=bool(vmecpp_restart),
        k_preconditioner_update_interval=k_preconditioner_update_interval,
        stage_prev_fsq=stage_prev_fsq,
        stage_transition_factor=stage_transition_factor,
        vmec2000_control=bool(vmec2000_control),
    )
    if bool(dump_timecontrol_scan):
        scan_timecontrol_dumper(
            cond=restart_decision.do_restart,
            stage_id=jnp.asarray(3, dtype=jnp.int32),
            iter2=iter2,
            iter1=carry_adv.iter1,
            fsq=fsq,
            fsq0=fsq_phys,
            res0=res0,
            res1=res1,
            time_step=carry_adv.time_step,
            irst=restart_decision.irst_restart,
        )

    def _restart_updates(_):
        return restart_updates_func(
            carry_adv=carry_adv,
            state_checkpoint=checkpoint_update.state_checkpoint,
            fsq_prev_before=fsq_prev_before,
            iter2=iter2,
            restart_reason=restart_decision.restart_reason,
            vmec2000_control=bool(vmec2000_control),
            restart_badjac_factor=restart_badjac_factor,
            restart_badprog_factor=restart_badprog_factor,
            stage_transition_scale=stage_transition_scale,
            step_size=step_size,
            k_ndamp=k_ndamp,
            dtype=dtype,
            scan_restart_transition_fn=scan_restart_transition_func,
        )

    def _no_restart_updates(_):
        return no_restart_updates_func(carry_adv)

    restart_update = scan_post_restart_update(
        do_restart=restart_decision.do_restart,
        restart_updates_fn=_restart_updates,
        no_restart_updates_fn=_no_restart_updates,
        cond_func=cond_func,
        iter2=iter2,
        stage_spike=restart_decision.stage_spike,
        stage_prev_fsq=stage_prev_fsq,
        stage_transition_scale=stage_transition_scale,
        k_ndamp=k_ndamp,
        dtype=dtype,
    )
    return ScanTimeRestartEvaluation(
        fsq_phys=fsq_phys,
        fsq=fsq,
        res0=res0,
        res1=res1,
        checkpoint_update=checkpoint_update,
        restart_decision=restart_decision,
        restart_update=restart_update,
        fsq0_prev_post=jnp.where(restart_decision.do_restart, fsq0_prev_before, fsq_phys),
    )


def scan_fallback_probe_update(
    *,
    enabled: bool,
    scan_core: bool,
    probe_count: Any,
    probe_bad_jac: Any,
    probe_accept: Any,
    probe_fsq_start: Any,
    probe_fsq_min: Any,
    probe_fsq_max: Any,
    fallback_active: Any,
    abort_scan: Any,
    fsq_phys: Any,
    fsq1: Any,
    bad_jacobian: Any,
    accepted: Any,
    abort_scan_on_badjac: bool,
    fallback_iters: Any,
    badjac_limit: Any,
    accept_frac: Any,
    fsq_factor: Any,
    fsq_abs: Any,
    improve: Any,
    dtype: Any,
) -> ScanFallbackProbeUpdate:
    """Update early scan-fallback probe counters and abort decision."""

    nan_fsq = (~jnp.isfinite(fsq_phys)) | (~jnp.isfinite(fsq1))
    abort_base = abort_scan | nan_fsq | (bad_jacobian & jnp.asarray(abort_scan_on_badjac))
    if (not bool(enabled)) or bool(scan_core):
        return ScanFallbackProbeUpdate(
            probe_count=probe_count,
            probe_bad_jac=probe_bad_jac,
            probe_accept=probe_accept,
            probe_fsq_start=probe_fsq_start,
            probe_fsq_min=probe_fsq_min,
            probe_fsq_max=probe_fsq_max,
            abort_scan=abort_base,
        )

    one_i = jnp.asarray(1, dtype=jnp.int32)
    zero_i = jnp.asarray(0, dtype=jnp.int32)
    probe_active = (probe_count < fallback_iters) & fallback_active
    probe_inc = jnp.where(probe_active, one_i, zero_i)
    probe_count_new = probe_count + probe_inc
    probe_bad_jac_new = probe_bad_jac + jnp.where(probe_active & bad_jacobian, one_i, zero_i)
    probe_accept_new = probe_accept + jnp.where(probe_active & accepted, one_i, zero_i)
    probe_fsq_start_new = jnp.where(probe_active & (probe_count == 0), fsq_phys, probe_fsq_start)
    probe_fsq_min_new = jnp.where(probe_active, jnp.minimum(probe_fsq_min, fsq_phys), probe_fsq_min)
    probe_fsq_max_new = jnp.where(probe_active, jnp.maximum(probe_fsq_max, fsq_phys), probe_fsq_max)

    has_probe = (probe_count_new >= fallback_iters) & fallback_active
    accepted_frac = probe_accept_new.astype(dtype) / jnp.maximum(
        probe_count_new.astype(dtype),
        jnp.asarray(1.0, dtype=dtype),
    )
    probe_start = jnp.maximum(probe_fsq_start_new, jnp.asarray(1.0e-30, dtype=dtype))
    probe_ratio = probe_fsq_max_new / probe_start
    bad_progress = probe_ratio > fsq_factor
    probe_improve = probe_fsq_min_new <= (probe_fsq_start_new * improve)
    stagnation_trigger = (
        has_probe
        & (~probe_improve)
        & (probe_fsq_min_new > fsq_abs)
        & (bad_progress | (accepted_frac < accept_frac))
    )
    accepted_trigger = has_probe & (accepted_frac < accept_frac) & (probe_fsq_min_new > fsq_abs) & bad_progress
    bad_jac_trigger = (
        (probe_bad_jac_new > badjac_limit)
        & (probe_fsq_min_new > fsq_abs)
        & (probe_count_new > 0)
        & bad_progress
    )
    scan_fallback_abort = (bad_jac_trigger | accepted_trigger | stagnation_trigger) & fallback_active

    return ScanFallbackProbeUpdate(
        probe_count=probe_count_new,
        probe_bad_jac=probe_bad_jac_new,
        probe_accept=probe_accept_new,
        probe_fsq_start=probe_fsq_start_new,
        probe_fsq_min=probe_fsq_min_new,
        probe_fsq_max=probe_fsq_max_new,
        abort_scan=abort_base | scan_fallback_abort,
    )
