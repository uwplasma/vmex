"""Pure update helpers for the residual-iteration VMEC solve."""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from ...._compat import jax, jnp
from ...._solve_runtime import _tree_has_tracer


class ResidualVelocityBlocks(NamedTuple):
    """Velocity-memory channels used by the VMEC residual update."""

    rcc: Any
    rss: Any
    rsc: Any
    rcs: Any
    zsc: Any
    zcs: Any
    zcc: Any
    zss: Any
    lsc: Any
    lcs: Any
    lcc: Any
    lss: Any


class ResidualControllerState(NamedTuple):
    """Host-loop scalar state that survives residual-solve restarts."""

    time_step: float
    inv_tau: list[float]
    fsq_prev: float
    fsq0_prev: float
    flip_sign: float
    iter1: int
    ijacob: int
    bad_resets: int
    res0: float
    res1: float
    prev_rz_fsq: float
    bad_growth_streak: int
    huge_force_restart_count: int
    state_checkpoint: Any


VELOCITY_RESUME_KEYS = {
    "vRcc": "rcc",
    "vRss": "rss",
    "vZsc": "zsc",
    "vZcs": "zcs",
    "vLsc": "lsc",
    "vLcs": "lcs",
    "vRsc": "rsc",
    "vRcs": "rcs",
    "vZcc": "zcc",
    "vZss": "zss",
    "vLcc": "lcc",
    "vLss": "lss",
}

STRICT_STEP_INPUT_ATTRS = "rcc rss zsc zcs lsc lcs rsc rcs zcc zss lcc lss".split()
STRICT_STEP_OUTPUT_KEYS = (
    "vRcc_after vRss_after vRsc_after vRcs_after vZsc_after vZcs_after "
    "vZcc_after vZss_after vLsc_after vLcs_after vLcc_after vLss_after"
).split()


CONTROLLER_RESUME_KEYS = (
    "time_step",
    "inv_tau",
    "fsq_prev",
    "fsq0_prev",
    "flip_sign",
    "iter1",
    "ijacob",
    "bad_resets",
    "res0",
    "res1",
    "prev_rz_fsq",
    "bad_growth_streak",
    "huge_force_restart_count",
    "state_checkpoint",
)


def initial_residual_controller_state(
    *,
    step_size: float,
    k_ndamp: int,
    initial_flip_sign: float,
    state_checkpoint: Any,
) -> ResidualControllerState:
    """Create VMEC residual-loop controller scalars from input controls."""

    time_step = float(step_size)
    return ResidualControllerState(
        time_step=time_step,
        inv_tau=[0.15 / time_step] * int(k_ndamp),
        fsq_prev=1.0,
        fsq0_prev=1.0,
        flip_sign=float(initial_flip_sign),
        iter1=1,
        ijacob=0,
        bad_resets=0,
        res0=-1.0,
        res1=-1.0,
        prev_rz_fsq=2.0,
        bad_growth_streak=0,
        huge_force_restart_count=0,
        state_checkpoint=state_checkpoint,
    )


def controller_state_from_resume_state(
    resume_state: dict[str, Any],
    defaults: ResidualControllerState,
) -> ResidualControllerState:
    """Restore residual-loop controller scalars from a legacy resume payload."""

    return ResidualControllerState(
        time_step=float(resume_state.get("time_step", defaults.time_step)),
        inv_tau=list(resume_state.get("inv_tau", defaults.inv_tau)),
        fsq_prev=float(resume_state.get("fsq_prev", defaults.fsq_prev)),
        fsq0_prev=float(resume_state.get("fsq0_prev", defaults.fsq0_prev)),
        flip_sign=float(resume_state.get("flip_sign", defaults.flip_sign)),
        iter1=int(resume_state.get("iter1", defaults.iter1)),
        ijacob=int(resume_state.get("ijacob", defaults.ijacob)),
        bad_resets=int(resume_state.get("bad_resets", defaults.bad_resets)),
        res0=float(resume_state.get("res0", defaults.res0)),
        res1=float(resume_state.get("res1", defaults.res1)),
        prev_rz_fsq=float(resume_state.get("prev_rz_fsq", defaults.prev_rz_fsq)),
        bad_growth_streak=int(resume_state.get("bad_growth_streak", defaults.bad_growth_streak)),
        huge_force_restart_count=int(
            resume_state.get("huge_force_restart_count", defaults.huge_force_restart_count)
        ),
        state_checkpoint=resume_state.get("state_checkpoint", defaults.state_checkpoint),
    )


def controller_state_legacy_payload(state: ResidualControllerState) -> dict[str, Any]:
    """Return legacy controller keys for resume-state diagnostics."""

    return {key: getattr(state, key) for key in CONTROLLER_RESUME_KEYS}


def controller_state_legacy_values(state: ResidualControllerState) -> tuple[Any, ...]:
    """Return controller values in the explicit legacy scalar-slot order."""

    return tuple(getattr(state, key) for key in CONTROLLER_RESUME_KEYS)


def controller_state_from_runtime_scalars(
    *,
    time_step: float,
    inv_tau: list[float],
    fsq_prev: float,
    fsq0_prev: float,
    flip_sign: float,
    iter1: int,
    ijacob: int,
    bad_resets: int,
    res0: float,
    res1: float,
    prev_rz_fsq: float,
    bad_growth_streak: int,
    huge_force_restart_count: int,
    state_checkpoint: Any,
) -> ResidualControllerState:
    """Snapshot the residual loop's legacy scalar slots into controller state."""

    return ResidualControllerState(
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


def apply_controller_state_update(
    state: ResidualControllerState,
    update_func,
    update,
) -> ResidualControllerState:
    """Apply one pure controller update to a controller-state snapshot."""

    return update_func(state, update)


def velocity_blocks_from_resume_state(
    resume_state: dict[str, Any],
    defaults: ResidualVelocityBlocks,
    *,
    as_velocity,
) -> ResidualVelocityBlocks:
    """Restore residual-loop velocity memory from legacy resume-state keys."""

    values = {
        attr_name: as_velocity(resume_state.get(resume_key, getattr(defaults, attr_name)))
        for resume_key, attr_name in VELOCITY_RESUME_KEYS.items()
    }
    return ResidualVelocityBlocks(
        rcc=values["rcc"],
        rss=values["rss"],
        rsc=values["rsc"],
        rcs=values["rcs"],
        zsc=values["zsc"],
        zcs=values["zcs"],
        zcc=values["zcc"],
        zss=values["zss"],
        lsc=values["lsc"],
        lcs=values["lcs"],
        lcc=values["lcc"],
        lss=values["lss"],
    )


def velocity_blocks_legacy_payload(blocks: ResidualVelocityBlocks) -> dict[str, Any]:
    """Return legacy ``vRcc``-style fields for resume diagnostics and traces."""

    return {resume_key: getattr(blocks, attr_name) for resume_key, attr_name in VELOCITY_RESUME_KEYS.items()}


def velocity_blocks_from_force_blocks(blocks: Any) -> ResidualVelocityBlocks:
    """Map VMEC force-block channel order to residual update channel order."""

    return ResidualVelocityBlocks(
        blocks.frcc, blocks.frss, blocks.frsc, blocks.frcs,
        blocks.fzsc, blocks.fzcs, blocks.fzcc, blocks.fzss,
        blocks.flsc, blocks.flcs, blocks.flcc, blocks.flss,
    )


class HostMomentumUpdate(NamedTuple):
    """Updated velocity memory plus RMS step size for a host momentum step."""

    velocities: ResidualVelocityBlocks
    update_rms: Any


class HostCatastrophicRestartUpdate(NamedTuple):
    """Scalar state after a host-loop catastrophic trial restart."""

    time_step: float
    ijacob: int
    restart_reason: str
    step_status: str
    restart_path: str
    max_coeff_delta_rms: float
    max_update_rms: float
    bad_resets: int
    iter1: int
    fsq_prev: float
    fsq0_prev: float
    inv_tau: list[float]
    update_rms: float


class HostPreRestartTriggerUpdate(NamedTuple):
    """Scalar state after a host-loop pre-restart trigger."""

    time_step: float
    time_step_iter: float
    ijacob: int
    step_status: str
    bad_resets: int
    iter1: int
    fsq_prev: float
    fsq0_prev: float
    inv_tau: list[float]
    huge_force_restart_count: int


class HostPreRestartTriggerBranchResult(NamedTuple):
    """Loop-side effects for a host pre-restart trigger branch."""

    state: Any
    update: HostPreRestartTriggerUpdate
    step_status: str
    restart_path: str
    restart_reason: str
    pre_restart_reason: str
    time_step_iter: float
    prev_rz_fsq: float
    clear_freeb_controls: bool
    clear_preconditioner_cache: bool
    force_bcovar_update: bool
    pop_iteration_history: bool
    skip_time_control: bool


class HostVmec2000TimeControlRestartUpdate(NamedTuple):
    """Scalar state after a VMEC2000 time-control restart."""

    time_step: float
    ijacob: int
    step_status: str
    restart_reason: str
    restart_path: str
    bad_resets: int
    iter1: int
    fsq_prev: float
    fsq0_prev: float
    inv_tau: list[float]


class HostVmec2000TimeControlRestartBranchResult(NamedTuple):
    """Loop-side effects for a VMEC2000 time-control restart branch."""

    state: Any
    update: HostVmec2000TimeControlRestartUpdate
    step_status: str
    restart_reason: str
    restart_path: str
    pre_restart_reason: str
    prev_rz_fsq: float
    clear_freeb_controls: bool
    clear_preconditioner_cache: bool
    force_bcovar_update: bool
    pop_iteration_history: bool
    skip_time_control: bool


class HostInitialAxisResetUpdate(NamedTuple):
    """Controller scalars after VMEC's first-iteration axis reset."""

    time_step: float
    ijacob: int
    iter1: int
    inv_tau: list[float]
    prev_rz_fsq: float
    bad_growth_streak: int
    state_checkpoint: Any


class HostFreeBoundaryTurnonRestartUpdate(NamedTuple):
    """Controller scalars after VMEC's free-boundary vacuum turn-on retry."""

    state: Any
    time_step_report_hold: float
    ijacob: int
    iter1: int
    inv_tau: list[float]
    bad_growth_streak: int


def host_initial_axis_reset_update(
    state_checkpoint: Any,
    time_step: float,
    iter2: int,
    prev_rz_fsq_before: float,
    k_ndamp: int,
) -> HostInitialAxisResetUpdate:
    """Return controller scalars for VMEC's initial-axis retry branch."""

    dt = float(time_step)
    return HostInitialAxisResetUpdate(
        time_step=dt,
        ijacob=1,
        iter1=int(iter2),
        inv_tau=[0.15 / dt] * int(k_ndamp),
        prev_rz_fsq=float(prev_rz_fsq_before),
        bad_growth_streak=0,
        state_checkpoint=state_checkpoint,
    )


def host_free_boundary_turnon_restart_update(
    *,
    state_checkpoint: Any,
    time_step: float,
    iter2: int,
    iter1: int,
    ijacob: int,
    k_ndamp: int,
    reset_iter1: bool,
) -> HostFreeBoundaryTurnonRestartUpdate:
    """Return VMEC controller scalars for the first active vacuum retry.

    VMEC restarts ``funct3d`` immediately when the free-boundary vacuum
    pressure first turns on.  The physical state rolls back to the checkpoint,
    velocity memory is cleared by the caller, and the nonlinear controller
    resets its bad-growth history while keeping the current pseudo-time step.
    """

    dt = float(time_step)
    return HostFreeBoundaryTurnonRestartUpdate(
        state=state_checkpoint,
        time_step_report_hold=dt,
        ijacob=int(ijacob) + 1,
        iter1=int(iter2) if bool(reset_iter1) else int(iter1),
        inv_tau=[0.15 / max(dt, 1.0e-12)] * int(k_ndamp),
        bad_growth_streak=0,
    )


def controller_state_after_free_boundary_turnon_restart_update(
    state: ResidualControllerState,
    update: HostFreeBoundaryTurnonRestartUpdate,
) -> ResidualControllerState:
    """Apply free-boundary turn-on retry scalars to controller state."""

    return state._replace(
        inv_tau=list(update.inv_tau),
        iter1=int(update.iter1),
        ijacob=int(update.ijacob),
        bad_growth_streak=int(update.bad_growth_streak),
    )


def controller_state_after_initial_axis_reset_update(
    state: ResidualControllerState,
    update: HostInitialAxisResetUpdate,
) -> ResidualControllerState:
    """Apply VMEC initial-axis retry scalars to controller state."""

    return state._replace(
        time_step=float(update.time_step),
        inv_tau=list(update.inv_tau),
        iter1=int(update.iter1),
        ijacob=int(update.ijacob),
        prev_rz_fsq=float(update.prev_rz_fsq),
        bad_growth_streak=int(update.bad_growth_streak),
        state_checkpoint=update.state_checkpoint,
    )


def controller_state_after_initial_axis_setup_result(
    state: ResidualControllerState,
    axis_setup: Any,
) -> ResidualControllerState:
    """Apply setup-time axis-reset controller scalars to controller state."""

    return state._replace(
        ijacob=int(axis_setup.ijacob),
        res0=float(axis_setup.res0),
        res1=float(axis_setup.res1),
        prev_rz_fsq=float(axis_setup.prev_rz_fsq),
        state_checkpoint=axis_setup.state_checkpoint,
    )


def host_vmec2000_time_control_restart_update(
    *,
    irst: int,
    time_step: float,
    restart_badjac_factor: float,
    restart_badprog_factor: float,
    ijacob: int,
    bad_resets: int,
    iter2: int,
    fsq_prev_before: float,
    fsq0_prev_before: float,
    k_ndamp: int,
) -> HostVmec2000TimeControlRestartUpdate:
    """Return VMEC2000 ``restart_iter`` scalar updates for time control."""

    if int(irst) == 2:
        time_step_next = max(float(restart_badjac_factor) * float(time_step), 1.0e-12)
        ijacob_next = int(ijacob) + 1
        step_status = "restart_bad_jacobian"
        restart_reason = "bad_jacobian"
        restart_path = "vmec2000_bad_jacobian"
    else:
        time_step_next = max(float(time_step) / float(restart_badprog_factor), 1.0e-12)
        ijacob_next = int(ijacob)
        step_status = "restart_time_control"
        restart_reason = "time_control"
        restart_path = "vmec2000_time_control"

    return HostVmec2000TimeControlRestartUpdate(
        time_step=float(time_step_next),
        ijacob=int(ijacob_next),
        step_status=step_status,
        restart_reason=restart_reason,
        restart_path=restart_path,
        bad_resets=int(bad_resets) + 1,
        iter1=int(iter2),
        fsq_prev=float(fsq_prev_before),
        fsq0_prev=float(fsq0_prev_before),
        inv_tau=[0.15 / float(time_step_next)] * int(k_ndamp),
    )


def controller_state_after_vmec2000_time_control_restart_update(
    state: ResidualControllerState,
    update: HostVmec2000TimeControlRestartUpdate,
) -> ResidualControllerState:
    """Apply VMEC2000 time-control restart scalars to controller state."""

    return state._replace(
        time_step=float(update.time_step),
        inv_tau=list(update.inv_tau),
        fsq_prev=float(update.fsq_prev),
        fsq0_prev=float(update.fsq0_prev),
        iter1=int(update.iter1),
        ijacob=int(update.ijacob),
        bad_resets=int(update.bad_resets),
        bad_growth_streak=0,
    )


def controller_state_after_vmec2000_time_control_sample(
    state: ResidualControllerState,
    decision: Any,
    *,
    state_checkpoint: Any,
) -> ResidualControllerState:
    """Apply non-restart VMEC2000 time-control scalar samples."""

    next_state = state._replace(res0=float(decision.res0), res1=float(decision.res1))
    if bool(decision.initialized) or bool(decision.store_checkpoint):
        next_state = next_state._replace(state_checkpoint=state_checkpoint)
    return next_state


def host_vmec2000_time_control_restart_branch_result(
    *,
    state_checkpoint: Any,
    restart_update: HostVmec2000TimeControlRestartUpdate,
    pre_restart_reason: str,
    prev_rz_fsq_before: float,
) -> HostVmec2000TimeControlRestartBranchResult:
    """Package VMEC2000 restart-loop side effects into one explicit contract."""

    return HostVmec2000TimeControlRestartBranchResult(
        state=state_checkpoint,
        update=restart_update,
        step_status=str(restart_update.step_status),
        restart_reason=str(restart_update.restart_reason),
        restart_path=str(restart_update.restart_path),
        pre_restart_reason=str(pre_restart_reason),
        prev_rz_fsq=float(prev_rz_fsq_before),
        clear_freeb_controls=True,
        clear_preconditioner_cache=True,
        force_bcovar_update=True,
        pop_iteration_history=True,
        skip_time_control=True,
    )


def controller_state_after_pre_restart_update(
    state: ResidualControllerState,
    update: HostPreRestartTriggerUpdate,
) -> ResidualControllerState:
    """Apply pre-restart scalar updates to controller state."""

    return state._replace(
        time_step=float(update.time_step),
        inv_tau=list(update.inv_tau),
        fsq_prev=float(update.fsq_prev),
        fsq0_prev=float(update.fsq0_prev),
        iter1=int(update.iter1),
        ijacob=int(update.ijacob),
        bad_resets=int(update.bad_resets),
        bad_growth_streak=0,
        huge_force_restart_count=int(update.huge_force_restart_count),
    )


def controller_state_after_host_restart_decision_sample(
    state: ResidualControllerState,
    decision: Any,
    *,
    state_checkpoint: Any,
) -> ResidualControllerState:
    """Apply non-restart host restart-decision tracker scalars."""

    next_state = state._replace(
        res0=float(decision.res0),
        bad_growth_streak=int(decision.bad_growth_streak),
    )
    if bool(decision.store_checkpoint):
        next_state = next_state._replace(state_checkpoint=state_checkpoint)
    return next_state


def host_pre_restart_trigger_branch_result(
    *,
    state_checkpoint: Any,
    pre_restart_update: HostPreRestartTriggerUpdate,
    pre_restart_reason: str,
    prev_rz_fsq_before: float,
    vmec2000_control: bool,
) -> HostPreRestartTriggerBranchResult:
    """Package pre-restart trigger side effects into one explicit contract."""

    reason = str(pre_restart_reason)
    return HostPreRestartTriggerBranchResult(
        state=state_checkpoint,
        update=pre_restart_update,
        step_status=str(pre_restart_update.step_status),
        restart_path="pre_restart_trigger",
        restart_reason=reason,
        pre_restart_reason=reason,
        time_step_iter=float(pre_restart_update.time_step_iter),
        prev_rz_fsq=float(prev_rz_fsq_before),
        clear_freeb_controls=True,
        clear_preconditioner_cache=True,
        force_bcovar_update=bool(vmec2000_control),
        pop_iteration_history=True,
        skip_time_control=True,
    )


def controller_state_after_catastrophic_restart_update(
    state: ResidualControllerState,
    update: HostCatastrophicRestartUpdate,
) -> ResidualControllerState:
    """Apply catastrophic-restart scalar updates to controller state."""

    return state._replace(
        time_step=float(update.time_step),
        inv_tau=list(update.inv_tau),
        fsq_prev=float(update.fsq_prev),
        fsq0_prev=float(update.fsq0_prev),
        iter1=int(update.iter1),
        ijacob=int(update.ijacob),
        bad_resets=int(update.bad_resets),
    )


class BacktrackingMomentumSearchResult(NamedTuple):
    """Result of the non-strict host backtracking momentum search."""

    state: Any
    velocities: ResidualVelocityBlocks
    dt_eff: float
    update_rms: float
    step_status: str
    accepted: bool


class DirectForceFallbackTrial(NamedTuple):
    """One direct-force fallback proposal before a catastrophic restart."""

    state: Any
    dt_eff: float
    update_rms: float
    residual: float


class DirectForceFallbackAcceptanceDecision(NamedTuple):
    """Host decision for accepting a no-momentum direct-force fallback step."""

    accepted: bool
    accept_ratio: float


class StrictMomentumProposal(NamedTuple):
    """Non-JIT strict momentum proposal and its host bookkeeping scalars."""

    state: Any
    velocities: ResidualVelocityBlocks
    update_deltas: tuple[Any, ...] | None
    update_rms_j: Any
    update_rms: float | None
    update_rms_preclip: float | None
    update_delta_rms_j: Any
    update_delta_rms: float | None
    scale: float


class StrictTrialEvaluation(NamedTuple):
    """Trial residual/backtracking result for a strict momentum proposal."""

    state: Any
    velocities: ResidualVelocityBlocks
    dt_eff: float
    update_rms: float | None
    w_try: float
    w_try_ratio: float
    probe_bad_jacobian: bool
    alpha: float


class StrictStepAcceptanceDecision(NamedTuple):
    """Host decision for accepting or rejecting one strict trial update."""

    accepted: bool
    accept_ratio: float


class StrictStepBranchResult(NamedTuple):
    """State and status selected by the strict trial accept/reject branch."""

    state: Any
    accepted: bool
    catastrophic_restart: bool
    clear_cache_after_catastrophic: bool
    step_status: str
    restart_reason: str
    restart_path: str
    huge_force_restart_count: int
    update_rms: float | None
    fallback_direct_dt: float | None = None
    max_coeff_delta_rms: float | None = None
    max_update_rms: float | None = None


class StrictStepBranchFingerprint(NamedTuple):
    """Array-free strict-step branch identity for same-branch validation gates."""

    path: str
    accepted: bool
    catastrophic_restart: bool
    clear_cache_after_catastrophic: bool
    restart_reason: str
    step_status: str
    has_direct_fallback: bool


class StrictStepRuntimeFields(NamedTuple):
    """Runtime scalar slots selected by a strict-step branch decision."""

    state: Any
    step_status: str
    restart_reason: str
    huge_force_restart_count: int
    restart_path: str
    update_rms: float | None
    max_coeff_delta_rms: float
    max_update_rms: float


class StrictStepBranchSideEffects(NamedTuple):
    """Loop side effects required by a strict-step branch decision."""

    zero_all_velocity_blocks: bool
    zero_primary_velocity_blocks: bool
    clear_freeb_controls_cached: bool
    clear_precond_cache: bool


class StrictStepBranchApplication(NamedTuple):
    """Runtime fields and side effects selected by a strict-step branch."""

    runtime: StrictStepRuntimeFields
    side_effects: StrictStepBranchSideEffects


class InitialResidualVelocityState(NamedTuple):
    """Initial residual-loop velocity memory and conservative update caps."""

    velocities: ResidualVelocityBlocks
    max_coeff_delta_rms: float
    max_update_rms: float


def initial_residual_velocity_state(
    *,
    state: Any,
    mpol: int,
    nrange: int,
    host_update_assembly: bool,
    reference_mode: bool,
) -> InitialResidualVelocityState:
    """Create VMEC residual-loop velocity memory with host/JAX array parity."""

    velocity_shape = (int(state.Rcos.shape[0]), int(mpol), int(nrange))
    if bool(host_update_assembly) and (not _tree_has_tracer(state.Rcos)):
        v_rcc = np.zeros(velocity_shape, dtype=np.asarray(state.Rcos).dtype)
    else:
        v_rcc = jnp.zeros(velocity_shape, dtype=jnp.asarray(state.Rcos).dtype)
    (
        v_rss,
        v_zsc,
        v_zcs,
        v_lsc,
        v_lcs,
        v_rsc,
        v_rcs,
        v_zcc,
        v_zss,
        v_lcc,
        v_lss,
    ) = zero_velocity_blocks_like(
        v_rcc,
        v_rcc,
        v_rcc,
        v_rcc,
        v_rcc,
        v_rcc,
        v_rcc,
        v_rcc,
        v_rcc,
        v_rcc,
        v_rcc,
    )
    max_coeff_delta_rms = 5e-6 if bool(reference_mode) else 1e-5
    max_update_rms = 1e-3 if bool(reference_mode) else 5e-3
    return InitialResidualVelocityState(
        ResidualVelocityBlocks(
            v_rcc,
            v_rss,
            v_rsc,
            v_rcs,
            v_zsc,
            v_zcs,
            v_zcc,
            v_zss,
            v_lsc,
            v_lcs,
            v_lcc,
            v_lss,
        ),
        max_coeff_delta_rms,
        max_update_rms,
    )


def candidate_state_from_deltas(
    *,
    state: Any,
    static: Any,
    dR_value: Any,
    dR_sin_value: Any,
    dZ_cos_value: Any,
    dZ_value: Any,
    dL_cos_value: Any,
    dL_value: Any,
    use_numpy_arrays: bool,
    use_numpy_enforce: bool,
    edge_Rcos: Any,
    edge_Rsin: Any,
    edge_Zcos: Any,
    edge_Zsin: Any,
    free_boundary_enabled: bool,
    idx00: int,
    precomputed_axis_mask: Any,
    enforce_fixed_boundary_and_axis: Any,
    enforce_fixed_boundary_and_axis_np: Any,
    apply_vmec_lambda_axis_rules: Any,
):
    """Build a candidate VMEC state after one residual update proposal."""

    from ....state import VMECState

    array = np.asarray if use_numpy_arrays else jnp.asarray
    candidate = VMECState(
        layout=state.layout,
        Rcos=array(state.Rcos) + array(dR_value),
        Rsin=array(state.Rsin) + array(dR_sin_value),
        Zcos=array(state.Zcos) + array(dZ_cos_value),
        Zsin=array(state.Zsin) + array(dZ_value),
        Lcos=array(state.Lcos) + array(dL_cos_value),
        Lsin=array(state.Lsin) + array(dL_value),
    )
    if use_numpy_enforce:
        candidate = enforce_fixed_boundary_and_axis_np(
            candidate,
            static,
            edge_Rcos=edge_Rcos,
            edge_Rsin=edge_Rsin,
            edge_Zcos=edge_Zcos,
            edge_Zsin=edge_Zsin,
            enforce_edge=not bool(free_boundary_enabled),
            enforce_lambda_axis=True,
            idx00=idx00,
            precomputed_axis_mask=precomputed_axis_mask,
        )
    else:
        candidate = enforce_fixed_boundary_and_axis(
            candidate,
            static,
            edge_Rcos=edge_Rcos,
            edge_Rsin=edge_Rsin,
            edge_Zcos=edge_Zcos,
            edge_Zsin=edge_Zsin,
            enforce_edge=not bool(free_boundary_enabled),
            enforce_lambda_axis=True,
            idx00=idx00,
        )
    return apply_vmec_lambda_axis_rules(candidate)


def delta_tuple_from_blocks(
    dt,
    transforms,
    *blocks,
    lasym: bool,
    zeros_dR_np: Any | None = None,
    use_numpy_lasym_zeros: bool = False,
):
    """Transform velocity blocks into physical R/Z/lambda update arrays."""

    rcc, rss, rsc, rcs, zsc, zcs, zcc, zss, lsc, lcs, lcc, lss = blocks
    mn_cos_to_signed, mn_sin_to_signed, mn_cos_to_signed_lambda, mn_sin_to_signed_lambda = transforms
    dR = dt * mn_cos_to_signed(rcc, rss)
    dZ = dt * mn_sin_to_signed(zsc, zcs)
    dL = dt * mn_sin_to_signed_lambda(lsc, lcs)
    if bool(lasym):
        dR_sin = dt * mn_sin_to_signed(rsc, rcs)
        dZ_cos = dt * mn_cos_to_signed(zcc, zss)
        dL_cos = dt * mn_cos_to_signed_lambda(lcc, lss)
    elif use_numpy_lasym_zeros:
        dR_sin = zeros_dR_np
        dZ_cos = zeros_dR_np
        dL_cos = zeros_dR_np
    else:
        dR_sin = jnp.zeros_like(dR)
        dZ_cos = jnp.zeros_like(dR)
        dL_cos = jnp.zeros_like(dR)
    return (dR, dR_sin, dZ_cos, dZ, dL_cos, dL)


def candidate_state_from_delta_tuple(
    deltas,
    *,
    scale: float,
    use_numpy_arrays: bool,
    use_numpy_enforce: bool,
    candidate_from_deltas: Any,
):
    """Build a candidate state from an already transformed delta tuple."""

    if float(scale) != 1.0:
        deltas = tuple(float(scale) * value for value in deltas)
    dR, dR_sin, dZ_cos, dZ, dL_cos, dL = deltas
    return candidate_from_deltas(
        dR_value=dR,
        dR_sin_value=dR_sin,
        dZ_cos_value=dZ_cos,
        dZ_value=dZ,
        dL_cos_value=dL_cos,
        dL_value=dL,
        use_numpy_arrays=use_numpy_arrays,
        use_numpy_enforce=use_numpy_enforce,
    )


def zero_velocity_blocks_like(*blocks):
    """Return zeroed velocity blocks with each input block's shape and dtype."""

    out = []
    for block in blocks:
        if _tree_has_tracer(block):
            out.append(jnp.zeros_like(block))
            continue
        try:
            if jax is not None and isinstance(block, jax.Array):
                out.append(jnp.zeros_like(block))
                continue
        except Exception:
            pass
        out.append(np.zeros_like(np.asarray(block)))
    return tuple(out)


def zero_all_velocity_blocks_like(velocities: ResidualVelocityBlocks) -> ResidualVelocityBlocks:
    """Return a velocity-memory bundle with every channel zeroed."""

    return ResidualVelocityBlocks(*zero_velocity_blocks_like(*velocities))


def zero_primary_velocity_blocks_like(velocities: ResidualVelocityBlocks) -> ResidualVelocityBlocks:
    """Zero only VMEC's primary symmetric velocity-memory channels."""

    rcc, rss, zsc, zcs, lsc, lcs = zero_velocity_blocks_like(
        velocities.rcc,
        velocities.rss,
        velocities.zsc,
        velocities.zcs,
        velocities.lsc,
        velocities.lcs,
    )
    return velocities._replace(rcc=rcc, rss=rss, zsc=zsc, zcs=zcs, lsc=lsc, lcs=lcs)


def scale_velocity_blocks(scale: float, *blocks):
    """Scale velocity blocks uniformly while preserving JAX array semantics."""

    return tuple(float(scale) * block for block in blocks)


def force_update_rms(scale: float, *blocks):
    """Return the JAX-visible RMS coefficient update implied by scaled blocks."""

    if not blocks:
        return jnp.asarray(0.0)
    total = None
    scale_j = jnp.asarray(scale)
    for block in blocks:
        term = (scale_j * block) ** 2
        total = term if total is None else total + term
    return jnp.sqrt(jnp.mean(total))


def host_force_update_rms(scale: float, *blocks) -> float:
    """Return the host scalar RMS coefficient update implied by scaled blocks."""

    return float(np.asarray(force_update_rms(scale, *blocks)))


def delta_tuple_rms(*deltas):
    """Return the JAX-visible RMS of an already transformed update tuple."""

    if not deltas:
        return jnp.asarray(0.0)
    total = None
    for delta in deltas:
        term = jnp.asarray(delta) ** 2
        total = term if total is None else total + term
    return jnp.sqrt(jnp.mean(total))


class ResidualEvolveCoefficients(NamedTuple):
    """Damping coefficients for one residual-loop evolve step."""

    inv_tau: list[float]
    fsq_prev: float
    fsq0_prev: float
    otav: float
    dtau: float
    b1: float
    fac: float


def residual_evolve_coefficients(
    *,
    iter2: int,
    iter1: int,
    inv_tau: list[float],
    time_step: float,
    fsq1: float,
    fsq_prev: float,
    fsq0_curr: float,
    k_ndamp: int,
) -> ResidualEvolveCoefficients:
    """Advance VMEC's damping-history recurrence for the next state update."""

    dt = float(time_step)
    if int(iter2) == int(iter1):
        next_inv_tau = [0.15 / dt] * int(k_ndamp)
    else:
        invtau_num = 0.0 if float(fsq1) == 0.0 else min(abs(np.log(float(fsq1) / float(fsq_prev))), 0.15)
        next_inv_tau = list(inv_tau)[1:] + [invtau_num / dt]
    next_fsq_prev = float(fsq1)
    next_fsq0_prev = float(fsq0_curr)
    otav = float(np.sum(next_inv_tau)) / float(k_ndamp)
    dtau = dt * otav / 2.0
    b1 = 1.0 - dtau
    fac = 1.0 / (1.0 + dtau)
    return ResidualEvolveCoefficients(
        inv_tau=next_inv_tau,
        fsq_prev=next_fsq_prev,
        fsq0_prev=next_fsq0_prev,
        otav=float(otav),
        dtau=float(dtau),
        b1=float(b1),
        fac=float(fac),
    )


def momentum_update_jax(
    *,
    velocities: ResidualVelocityBlocks,
    forces: ResidualVelocityBlocks,
    b1: float,
    fac: float,
    force_scale: float,
    flip_sign: float,
    dt_eff: float,
    compute_update_rms: bool,
) -> HostMomentumUpdate:
    """Apply the JAX-visible strict momentum update to all velocity blocks."""

    updated = ResidualVelocityBlocks(
        *(
            fac * (b1 * velocity + force_scale * (flip_sign * jnp.asarray(force)))
            for velocity, force in zip(velocities, forces)
        )
    )
    if compute_update_rms:
        update_rms = force_update_rms(dt_eff, *updated)
    else:
        update_rms = jnp.asarray(0.0, dtype=jnp.asarray(updated.rcc).dtype)
    return HostMomentumUpdate(velocities=updated, update_rms=update_rms)


def host_momentum_update_np(
    *,
    velocities: ResidualVelocityBlocks,
    forces: ResidualVelocityBlocks,
    b1: float,
    fac: float,
    force_scale: float,
    flip_sign: float,
    dt_eff: float,
    compute_update_rms: bool,
) -> HostMomentumUpdate:
    """Apply the host strict momentum update to all velocity blocks."""
    velocity_stack = np.stack([np.asarray(block) for block in velocities])
    force_stack = np.stack([np.asarray(block) for block in forces])

    np.multiply(velocity_stack, float(fac) * float(b1), out=velocity_stack)
    np.multiply(force_stack, float(fac) * float(force_scale) * float(flip_sign), out=force_stack)
    np.add(velocity_stack, force_stack, out=velocity_stack)

    if compute_update_rms:
        flat = velocity_stack.ravel()
        update_rms = abs(float(dt_eff)) * np.sqrt(np.dot(flat, flat) / velocity_stack.size)
    else:
        update_rms = np.asarray(0.0, dtype=velocity_stack.dtype)

    return HostMomentumUpdate(
        velocities=ResidualVelocityBlocks(*velocity_stack),
        update_rms=update_rms,
    )


def strict_momentum_update_proposal(
    *,
    velocities: ResidualVelocityBlocks,
    forces: ResidualVelocityBlocks,
    host_update_assembly: bool,
    need_update_rms: bool,
    materialize_update_rms: bool,
    limit_update_rms: bool,
    max_update_rms: float,
    b1: float,
    fac: float,
    force_scale: float,
    flip_sign: float,
    dt_eff: float,
    delta_transforms: tuple,
    delta_tuple_from_blocks: Any,
    candidate_state_from_delta_tuple: Any,
    delta_tuple_projector: Any | None = None,
) -> StrictMomentumProposal:
    """Build the non-JIT strict momentum candidate state for one iteration."""

    if host_update_assembly:
        update_result = host_momentum_update_np(
            velocities=velocities,
            forces=forces,
            b1=b1,
            fac=fac,
            force_scale=force_scale,
            flip_sign=flip_sign,
            dt_eff=dt_eff,
            compute_update_rms=need_update_rms,
        )
    else:
        update_result = momentum_update_jax(
            velocities=velocities,
            forces=forces,
            b1=b1,
            fac=fac,
            force_scale=force_scale,
            flip_sign=flip_sign,
            dt_eff=dt_eff,
            compute_update_rms=need_update_rms,
        )

    updated_velocities = update_result.velocities
    update_rms_j = (
        update_result.update_rms
        if need_update_rms
        else jnp.asarray(0.0, dtype=jnp.asarray(updated_velocities.rcc).dtype)
    )
    update_rms = float(np.asarray(update_rms_j)) if materialize_update_rms else None
    update_rms_preclip = update_rms
    scale = 1.0

    if bool(limit_update_rms) and update_rms is not None and np.isfinite(update_rms) and update_rms > max_update_rms:
        scale = float(max_update_rms) / max(update_rms, 1.0e-30)
        updated_velocities = ResidualVelocityBlocks(*scale_velocity_blocks(scale, *updated_velocities))
        update_rms_j = force_update_rms(dt_eff, *updated_velocities)
        update_rms = float(np.asarray(update_rms_j))

    update_deltas = delta_tuple_from_blocks(
        dt_eff,
        delta_transforms,
        *updated_velocities,
        use_numpy_lasym_zeros=bool(host_update_assembly),
    )
    if delta_tuple_projector is not None:
        update_deltas = delta_tuple_projector(update_deltas, host_update=bool(host_update_assembly))
    update_delta_rms_j = (
        delta_tuple_rms(*update_deltas)
        if need_update_rms
        else jnp.asarray(0.0, dtype=jnp.asarray(updated_velocities.rcc).dtype)
    )
    update_delta_rms = float(np.asarray(update_delta_rms_j)) if materialize_update_rms else None
    state = candidate_state_from_delta_tuple(
        update_deltas,
        use_numpy_arrays=bool(host_update_assembly),
        use_numpy_enforce=bool(host_update_assembly),
    )
    return StrictMomentumProposal(
        state=state,
        velocities=updated_velocities,
        update_deltas=update_deltas,
        update_rms_j=update_rms_j,
        update_rms=update_rms,
        update_rms_preclip=update_rms_preclip,
        update_delta_rms_j=update_delta_rms_j,
        update_delta_rms=update_delta_rms,
        scale=float(scale),
    )


def jit_strict_momentum_update_proposal(
    *,
    state: Any, static: Any, velocities: ResidualVelocityBlocks, forces: ResidualVelocityBlocks,
    dt_eff: float, b1: float, fac: float, force_scale: float, flip_sign: float,
    max_update_rms: float, need_update_rms: bool, divide_by_scalxc_for_update: bool,
    free_boundary_enabled: bool, strict_update_step_jit_func: Any,
) -> StrictMomentumProposal:
    """Build the strict momentum proposal using the compiled update kernel."""

    step_fn = strict_update_step_jit_func(
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
        *(getattr(velocities, name) for name in STRICT_STEP_INPUT_ATTRS),
        *(getattr(forces, name) for name in STRICT_STEP_INPUT_ATTRS),
        max_update_rms,
    )
    updated_velocities = ResidualVelocityBlocks(*(step_out[key] for key in STRICT_STEP_OUTPUT_KEYS))
    update_rms_j = (
        step_out["update_rms_postclip"]
        if need_update_rms
        else jnp.asarray(0.0, dtype=jnp.asarray(updated_velocities.rcc).dtype)
    )
    return StrictMomentumProposal(
        state=step_out["state_post"], velocities=updated_velocities, update_deltas=None,
        update_rms_j=update_rms_j, update_rms=None, update_rms_preclip=None,
        update_delta_rms_j=jnp.asarray(0.0, dtype=jnp.asarray(updated_velocities.rcc).dtype),
        update_delta_rms=None,
        scale=1.0,
    )


def scale_primary_velocity_blocks(scale: float, velocities: ResidualVelocityBlocks) -> ResidualVelocityBlocks:
    """Scale VMEC's primary symmetric velocity memory used by backtracking."""

    return ResidualVelocityBlocks(
        float(scale) * velocities.rcc,
        float(scale) * velocities.rss,
        velocities.rsc,
        velocities.rcs,
        float(scale) * velocities.zsc,
        float(scale) * velocities.zcs,
        velocities.zcc,
        velocities.zss,
        float(scale) * velocities.lsc,
        float(scale) * velocities.lcs,
        velocities.lcc,
        velocities.lss,
    )


def strict_trial_evaluation(
    *,
    state_try: Any,
    velocities: ResidualVelocityBlocks,
    update_deltas: tuple[Any, ...],
    update_rms: float | None,
    dt_eff: float,
    w_curr: float,
    backtracking: bool,
    reference_mode: bool,
    host_update_assembly: bool,
    zero_m1_value: Any,
    zero_m1_host: float,
    zero_m1_probe_value: Any,
    candidate_state_from_delta_tuple: Any,
    freeb_bsqvac_half_for_trial_state: Any,
    trial_residual_total: Any,
    heartbeat: Any | None = None,
    max_backtracks: int = 8,
    probe_growth_factor: float = 1.0e2,
) -> StrictTrialEvaluation:
    """Evaluate and optionally backtrack one strict momentum trial state."""

    if heartbeat is not None:
        heartbeat("trial_bsqvac_start", alpha=1.0)
    freeb_bsqvac_half_trial = freeb_bsqvac_half_for_trial_state(state_try)
    if heartbeat is not None:
        heartbeat("trial_force_start", alpha=1.0)
    w_try = trial_residual_total(
        state_try,
        freeb_bsqvac_half_trial,
        zero_m1_value=zero_m1_value,
        timing_label="trial",
    )
    w_try_ratio = float(w_try) / max(float(w_curr), 1.0e-30) if np.isfinite(w_try) else float("inf")
    if heartbeat is not None:
        heartbeat("trial_force_done", alpha=1.0, w_try=float(w_try), w_try_ratio=float(w_try_ratio))
    probe_bad_jacobian = False
    if bool(reference_mode) and float(zero_m1_host) > 0.5:
        if heartbeat is not None:
            heartbeat("trial_probe_force_start", alpha=1.0)
        w_probe = trial_residual_total(
            state_try,
            freeb_bsqvac_half_trial,
            zero_m1_value=zero_m1_probe_value,
        )
        if heartbeat is not None:
            heartbeat("trial_probe_force_done", alpha=1.0, w_probe=float(w_probe))
        if (not np.isfinite(w_probe)) or (w_probe > float(probe_growth_factor) * max(float(w_curr), 1.0e-30)):
            probe_bad_jacobian = True
            w_try = float("inf")
            w_try_ratio = float("inf")

    alpha = 1.0
    accept_ratio = 1.001 if bool(backtracking) else float("inf")
    if np.isfinite(w_try) and (w_try > accept_ratio * max(float(w_curr), 1.0e-30)):
        for backtrack_index in range(int(max_backtracks)):
            alpha *= 0.5
            if heartbeat is not None:
                heartbeat("backtrack_state_start", alpha=float(alpha), backtrack_index=int(backtrack_index))
            state_try = candidate_state_from_delta_tuple(
                update_deltas,
                scale=alpha,
                use_numpy_arrays=False,
                use_numpy_enforce=bool(host_update_assembly),
            )
            if heartbeat is not None:
                heartbeat("backtrack_bsqvac_start", alpha=float(alpha), backtrack_index=int(backtrack_index))
            freeb_bsqvac_half_trial = freeb_bsqvac_half_for_trial_state(state_try)
            if heartbeat is not None:
                heartbeat("backtrack_force_start", alpha=float(alpha), backtrack_index=int(backtrack_index))
            w_try = trial_residual_total(
                state_try,
                freeb_bsqvac_half_trial,
                zero_m1_value=zero_m1_value,
                timing_label="trial",
            )
            w_try_ratio = float(w_try) / max(float(w_curr), 1.0e-30) if np.isfinite(w_try) else float("inf")
            if heartbeat is not None:
                heartbeat(
                    "backtrack_force_done",
                    alpha=float(alpha),
                    backtrack_index=int(backtrack_index),
                    w_try=float(w_try),
                    w_try_ratio=float(w_try_ratio),
                )
            if np.isfinite(w_try) and (w_try <= accept_ratio * max(float(w_curr), 1.0e-30)):
                velocities = scale_primary_velocity_blocks(alpha, velocities)
                if update_rms is not None:
                    update_rms = float(update_rms) * alpha
                dt_eff = float(dt_eff) * alpha
                break

    return StrictTrialEvaluation(
        state=state_try,
        velocities=velocities,
        dt_eff=float(dt_eff),
        update_rms=update_rms,
        w_try=float(w_try),
        w_try_ratio=float(w_try_ratio),
        probe_bad_jacobian=bool(probe_bad_jacobian),
        alpha=float(alpha),
    )


def host_catastrophic_restart_update(
    *,
    probe_bad_jacobian: bool,
    w_try: float,
    time_step: float,
    restart_badjac_factor: float,
    restart_badprog_factor: float,
    step_size: float,
    ijacob: int,
    bad_resets: int,
    iter2: int,
    fsq_prev_before: float,
    fsq0_prev_before: float,
    k_ndamp: int,
    max_coeff_delta_rms: float,
    max_update_rms: float,
) -> HostCatastrophicRestartUpdate:
    """Apply VMEC-style scalar updates after rejecting a catastrophic trial.

    The caller owns the large state rollback and velocity zeroing.  This helper
    keeps the scalar branch policy in one place so scan/replay fingerprints can
    reason about the same restart semantics.
    """

    max_coeff_delta_rms_next = max(0.5 * float(max_coeff_delta_rms), 1.0e-12)
    max_update_rms_next = max(0.8 * float(max_update_rms), 1.0e-6)

    if bool(probe_bad_jacobian) or (not np.isfinite(float(w_try))):
        time_step_next = max(float(restart_badjac_factor) * float(time_step), 1.0e-12)
        ijacob_next = int(ijacob) + 1
        restart_reason = "bad_jacobian"
        step_status = "restart_bad_jacobian"
        restart_path = "catastrophic_nonfinite"
    else:
        time_step_next = max(float(time_step) / float(restart_badprog_factor), 1.0e-12)
        ijacob_next = int(ijacob)
        restart_reason = "bad_progress"
        step_status = "restart_bad_progress"
        restart_path = "catastrophic_growth"

    if ijacob_next in (25, 50):
        scale = 0.98 if ijacob_next < 50 else 0.96
        time_step_next = max(scale * float(step_size), 1.0e-12)

    return HostCatastrophicRestartUpdate(
        time_step=float(time_step_next),
        ijacob=int(ijacob_next),
        restart_reason=restart_reason,
        step_status=step_status,
        restart_path=restart_path,
        max_coeff_delta_rms=float(max_coeff_delta_rms_next),
        max_update_rms=float(max_update_rms_next),
        bad_resets=int(bad_resets) + 1,
        iter1=int(iter2),
        fsq_prev=float(fsq_prev_before),
        fsq0_prev=float(fsq0_prev_before),
        inv_tau=[0.15 / float(time_step_next)] * int(k_ndamp),
        update_rms=0.0,
    )


def host_pre_restart_trigger_update(
    *,
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
    fsq_prev_before: float,
    fsq0_prev_before: float,
    k_ndamp: int,
) -> HostPreRestartTriggerUpdate:
    """Return scalar updates for the pre-state-update restart path."""

    reason = str(pre_restart_reason)
    ijacob_next = int(ijacob)
    if reason == "bad_jacobian":
        time_step_next = max(float(restart_badjac_factor) * float(time_step), 1.0e-12)
        ijacob_next += 1
        step_status = "restart_bad_jacobian"
    elif reason == "stage_transition":
        time_step_next = max(float(time_step) * float(stage_transition_scale), 1.0e-12)
        step_status = "restart_stage_transition"
    else:
        time_step_next = max(float(time_step) / float(restart_badprog_factor), 1.0e-12)
        step_status = "restart_bad_progress"

    if bool(huge_initial_forces) and reason == "bad_jacobian":
        huge_force_restart_count_next = int(huge_force_restart_count) + 1
    else:
        huge_force_restart_count_next = 0

    if ijacob_next in (25, 50):
        scale = 0.98 if ijacob_next < 50 else 0.96
        time_step_next = max(scale * float(step_size), 1.0e-12)

    return HostPreRestartTriggerUpdate(
        time_step=float(time_step_next),
        time_step_iter=float(time_step_next),
        ijacob=int(ijacob_next),
        step_status=step_status,
        bad_resets=int(bad_resets) + 1,
        iter1=int(iter2),
        fsq_prev=float(fsq_prev_before),
        fsq0_prev=float(fsq0_prev_before),
        inv_tau=[0.15 / float(time_step_next)] * int(k_ndamp),
        huge_force_restart_count=int(huge_force_restart_count_next),
    )


def strict_step_acceptance_decision(
    *,
    w_try: float,
    w_curr: float,
    backtracking: bool,
) -> StrictStepAcceptanceDecision:
    """Decide whether a strict trial residual step is acceptable.

    This is intentionally a host-side branch object: the production VMEC loop
    still uses Python control flow for accept/reject decisions, while tests and
    future fingerprint gates can reason about the branch decision explicitly.
    """

    accept_ratio = 1.001 if bool(backtracking) else float("inf")
    accepted = bool(np.isfinite(w_try) and (float(w_try) <= accept_ratio * max(float(w_curr), 1.0e-30)))
    return StrictStepAcceptanceDecision(accepted=accepted, accept_ratio=float(accept_ratio))


def strict_step_branch_result(
    *,
    acceptance: StrictStepAcceptanceDecision,
    state_try: Any,
    state_backup: Any,
    update_rms: float | None,
    vmec2000_control: bool,
    huge_force_restart_count: int,
) -> StrictStepBranchResult:
    """Package strict-step accept/reject state and host status bookkeeping."""

    if bool(acceptance.accepted):
        return StrictStepBranchResult(
            state=state_try,
            accepted=True,
            catastrophic_restart=False,
            clear_cache_after_catastrophic=False,
            step_status="momentum",
            restart_reason="none",
            restart_path="momentum_accept",
            huge_force_restart_count=0,
            update_rms=update_rms,
            fallback_direct_dt=None,
            max_coeff_delta_rms=None,
            max_update_rms=None,
        )
    return StrictStepBranchResult(
        state=state_backup,
        accepted=False,
        catastrophic_restart=True,
        clear_cache_after_catastrophic=not bool(vmec2000_control),
        step_status="restart_pending",
        restart_reason="trial_rejected",
        restart_path="trial_rejected",
        huge_force_restart_count=int(huge_force_restart_count),
        update_rms=update_rms,
        fallback_direct_dt=None,
        max_coeff_delta_rms=None,
        max_update_rms=None,
    )


def strict_step_branch_result_after_direct_fallback(
    *,
    branch: StrictStepBranchResult,
    fallback_trial: DirectForceFallbackTrial,
    acceptance: DirectForceFallbackAcceptanceDecision,
    clear_cache_after_rejected: bool,
) -> StrictStepBranchResult:
    """Update a rejected strict-step branch after a direct-force fallback trial."""

    if bool(acceptance.accepted):
        return branch._replace(
            state=fallback_trial.state,
            accepted=True,
            catastrophic_restart=False,
            clear_cache_after_catastrophic=False,
            step_status="fallback_direct",
            restart_reason="none",
            restart_path="fallback_direct",
            huge_force_restart_count=0,
            update_rms=fallback_trial.update_rms,
            fallback_direct_dt=float(fallback_trial.dt_eff),
        )
    return branch._replace(clear_cache_after_catastrophic=bool(clear_cache_after_rejected))


def strict_step_branch_result_after_catastrophic_restart(
    *,
    branch: StrictStepBranchResult,
    restart_update: HostCatastrophicRestartUpdate,
    state_backup: Any,
) -> StrictStepBranchResult:
    """Update a rejected strict-step branch after catastrophic restart policy."""

    return branch._replace(
        state=state_backup,
        accepted=False,
        catastrophic_restart=True,
        step_status=str(restart_update.step_status),
        restart_reason=str(restart_update.restart_reason),
        restart_path=str(restart_update.restart_path),
        update_rms=float(restart_update.update_rms),
        max_coeff_delta_rms=float(restart_update.max_coeff_delta_rms),
        max_update_rms=float(restart_update.max_update_rms),
    )


def strict_step_branch_fingerprint(branch: StrictStepBranchResult) -> StrictStepBranchFingerprint:
    """Return the array-free identity of a strict-step branch decision."""

    return StrictStepBranchFingerprint(
        path=str(branch.restart_path),
        accepted=bool(branch.accepted),
        catastrophic_restart=bool(branch.catastrophic_restart),
        clear_cache_after_catastrophic=bool(branch.clear_cache_after_catastrophic),
        restart_reason=str(branch.restart_reason),
        step_status=str(branch.step_status),
        has_direct_fallback=branch.fallback_direct_dt is not None,
    )


def strict_step_runtime_fields(
    branch: StrictStepBranchResult,
    *,
    max_coeff_delta_rms: float,
    max_update_rms: float,
) -> StrictStepRuntimeFields:
    """Return the legacy residual-loop scalar slots selected by ``branch``."""

    return StrictStepRuntimeFields(
        state=branch.state,
        step_status=str(branch.step_status),
        restart_reason=str(branch.restart_reason),
        huge_force_restart_count=int(branch.huge_force_restart_count),
        restart_path=str(branch.restart_path),
        update_rms=branch.update_rms,
        max_coeff_delta_rms=(
            float(max_coeff_delta_rms)
            if branch.max_coeff_delta_rms is None
            else float(branch.max_coeff_delta_rms)
        ),
        max_update_rms=float(max_update_rms) if branch.max_update_rms is None else float(branch.max_update_rms),
    )


def strict_step_branch_side_effects(
    branch: StrictStepBranchResult,
    *,
    after_catastrophic_restart: bool = False,
) -> StrictStepBranchSideEffects:
    """Return non-state side effects selected by a strict-step branch."""

    catastrophic = (not bool(branch.accepted)) and bool(branch.catastrophic_restart)
    after_restart = bool(after_catastrophic_restart)
    return StrictStepBranchSideEffects(
        zero_all_velocity_blocks=bool(branch.accepted) and branch.restart_path == "fallback_direct",
        zero_primary_velocity_blocks=catastrophic and not after_restart,
        clear_freeb_controls_cached=catastrophic and after_restart,
        clear_precond_cache=catastrophic and after_restart and bool(branch.clear_cache_after_catastrophic),
    )


def strict_step_branch_application(
    branch: StrictStepBranchResult,
    *,
    max_coeff_delta_rms: float,
    max_update_rms: float,
    after_catastrophic_restart: bool = False,
) -> StrictStepBranchApplication:
    """Return all runtime effects selected by one strict-step branch."""

    return StrictStepBranchApplication(
        runtime=strict_step_runtime_fields(
            branch,
            max_coeff_delta_rms=float(max_coeff_delta_rms),
            max_update_rms=float(max_update_rms),
        ),
        side_effects=strict_step_branch_side_effects(
            branch,
            after_catastrophic_restart=bool(after_catastrophic_restart),
        ),
    )


def backtracking_momentum_search(
    *,
    state: Any,
    velocities: ResidualVelocityBlocks,
    forces: ResidualVelocityBlocks,
    time_step: float,
    step_size: float,
    b1: float,
    fac: float,
    flip_sign: float,
    w_curr: float,
    delta_transforms: tuple,
    delta_tuple_from_blocks: Any,
    candidate_state_from_delta_tuple: Any,
    freeb_bsqvac_half_for_trial_state: Any,
    trial_residual_total: Any,
    delta_tuple_projector: Any | None = None,
    max_backtracks: int = 6,
    accept_ratio: float = 1.05,
) -> BacktrackingMomentumSearchResult:
    """Try non-strict momentum updates, halving the step until residual growth is acceptable."""

    accepted = False
    step_status = "rejected"
    step_factor = 1.0
    best_state = state
    best_velocities = velocities
    dt_eff = float(time_step)
    update_rms = 0.0

    for _ in range(int(max_backtracks)):
        dt_try = float(time_step) * step_factor
        trial_velocities = ResidualVelocityBlocks(
            *(
                fac * (b1 * velocity + dt_try * (flip_sign * jnp.asarray(force)))
                for velocity, force in zip(velocities, forces)
            )
        )
        trial_deltas = delta_tuple_from_blocks(dt_try, delta_transforms, *trial_velocities)
        if delta_tuple_projector is not None:
            trial_deltas = delta_tuple_projector(trial_deltas, host_update=False)
        state_try = candidate_state_from_delta_tuple(
            trial_deltas,
            use_numpy_arrays=False,
            use_numpy_enforce=False,
        )
        freeb_bsqvac_half_trial = freeb_bsqvac_half_for_trial_state(state_try)
        w_try = trial_residual_total(state_try, freeb_bsqvac_half_trial)
        if np.isfinite(w_try) and (w_try <= float(accept_ratio) * float(w_curr)):
            accepted = True
            step_status = "momentum"
            best_state = state_try
            best_velocities = trial_velocities
            dt_eff = float(dt_try)
            update_rms = host_force_update_rms(dt_try, *trial_velocities)
            break
        step_factor *= 0.5

    if not accepted:
        best_velocities = ResidualVelocityBlocks(*scale_velocity_blocks(0.5, *best_velocities))
        dt_eff = float(step_size) * step_factor
        update_rms = 0.0

    return BacktrackingMomentumSearchResult(
        state=best_state,
        velocities=best_velocities,
        dt_eff=float(dt_eff),
        update_rms=float(update_rms),
        step_status=step_status,
        accepted=bool(accepted),
    )


def direct_force_fallback_trial(
    *,
    forces: ResidualVelocityBlocks,
    dt_eff: float,
    max_update_rms: float,
    flip_sign: float,
    delta_transforms: tuple,
    delta_tuple_from_blocks: Any,
    candidate_state_from_delta_tuple: Any,
    freeb_bsqvac_half_for_trial_state: Any,
    trial_residual_total: Any,
) -> DirectForceFallbackTrial:
    """Build one small no-momentum force proposal for the strict restart path.

    The caller remains responsible for accepting/rejecting the proposal.  That
    keeps branch status and history bookkeeping local to the residual loop while
    making the pure proposal calculation testable.
    """

    dt_direct = max(0.1 * float(dt_eff), 1.0e-12)
    force_rms = host_force_update_rms(1.0, *forces)
    if np.isfinite(force_rms) and force_rms > 0.0:
        dt_cap = float(max_update_rms) / max(force_rms, 1.0e-30)
        dt_direct = max(min(dt_direct, float(dt_cap)), 1.0e-12)

    state = candidate_state_from_delta_tuple(
        delta_tuple_from_blocks(
            dt_direct,
            delta_transforms,
            *(float(flip_sign) * block for block in forces),
        ),
        use_numpy_arrays=False,
        use_numpy_enforce=False,
    )
    freeb_bsqvac_half = freeb_bsqvac_half_for_trial_state(state)
    residual = trial_residual_total(state, freeb_bsqvac_half)
    return DirectForceFallbackTrial(
        state=state,
        dt_eff=float(dt_direct),
        update_rms=host_force_update_rms(dt_direct, *forces),
        residual=float(residual),
    )


def direct_force_fallback_acceptance_decision(
    *,
    residual: float,
    current_residual: float,
    accept_ratio: float = 1.5,
) -> DirectForceFallbackAcceptanceDecision:
    """Decide whether the strict-step direct-force fallback is acceptable."""

    accepted = bool(
        np.isfinite(residual)
        and (float(residual) <= float(accept_ratio) * max(float(current_residual), 1.0e-30))
    )
    return DirectForceFallbackAcceptanceDecision(accepted=accepted, accept_ratio=float(accept_ratio))


_ResidualVelocityBlocks = ResidualVelocityBlocks
_HostCatastrophicRestartUpdate = HostCatastrophicRestartUpdate
_HostPreRestartTriggerUpdate = HostPreRestartTriggerUpdate
_DirectForceFallbackTrial = DirectForceFallbackTrial
_DirectForceFallbackAcceptanceDecision = DirectForceFallbackAcceptanceDecision
_StrictMomentumProposal = StrictMomentumProposal
_StrictTrialEvaluation = StrictTrialEvaluation
_zero_velocity_blocks_like = zero_velocity_blocks_like
_zero_all_velocity_blocks_like = zero_all_velocity_blocks_like
_zero_primary_velocity_blocks_like = zero_primary_velocity_blocks_like
_scale_velocity_blocks = scale_velocity_blocks
_scale_primary_velocity_blocks = scale_primary_velocity_blocks
_velocity_blocks_from_force_blocks = velocity_blocks_from_force_blocks
_host_force_update_rms = host_force_update_rms
_delta_tuple_rms = delta_tuple_rms
_momentum_update_jax = momentum_update_jax
_host_momentum_update_np = host_momentum_update_np
_strict_momentum_update_proposal = strict_momentum_update_proposal
_strict_trial_evaluation = strict_trial_evaluation
_strict_step_branch_fingerprint = strict_step_branch_fingerprint
_strict_step_branch_result_after_catastrophic_restart = strict_step_branch_result_after_catastrophic_restart
_strict_step_branch_result_after_direct_fallback = strict_step_branch_result_after_direct_fallback
_host_catastrophic_restart_update = host_catastrophic_restart_update
_host_pre_restart_trigger_update = host_pre_restart_trigger_update
_direct_force_fallback_trial = direct_force_fallback_trial
_direct_force_fallback_acceptance_decision = direct_force_fallback_acceptance_decision
