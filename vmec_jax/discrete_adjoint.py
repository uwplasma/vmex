"""Utilities for the discrete-adjoint recovery path.

The first step is a structured view of the existing fixed-boundary residual
iteration history. This keeps the initial refactor narrow: no solver behavior
changes, only a stable extraction layer over the primal trace data already
recorded in diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._compat import jnp
from .state import pack_state
from .vmec_tomnsp import TomnspsRZL


@dataclass(frozen=True)
class ResidualIterationTrace:
    """Structured view of one fixed-boundary residual solve history."""

    iter2: np.ndarray
    step_status: np.ndarray
    restart_reason: np.ndarray
    pre_restart_reason: np.ndarray
    time_step: np.ndarray
    dt_eff: np.ndarray
    update_rms: np.ndarray
    include_edge: np.ndarray
    zero_m1: np.ndarray
    fsq_curr: np.ndarray
    fsq_try: np.ndarray
    fsq_prev: np.ndarray
    r00: np.ndarray
    z00: np.ndarray
    wb: np.ndarray
    wp: np.ndarray
    w_vmec: np.ndarray
    state_advanced: np.ndarray


@dataclass(frozen=True)
class ResidualCheckpointTape:
    """Replay-friendly checkpoints from repeated one-step residual solves."""

    packed_states: np.ndarray
    trace: ResidualIterationTrace
    resume_states: tuple[dict[str, Any] | None, ...]
    step_traces: tuple[dict[str, Any], ...]


def _array_from_diag(diagnostics: dict[str, Any], key: str, *, dtype=None) -> np.ndarray:
    value = diagnostics.get(key, np.zeros((0,), dtype=float))
    arr = np.asarray(value)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def residual_iteration_trace_from_result(result) -> ResidualIterationTrace:
    """Extract a compact, typed residual-iteration trace from a solver result."""
    diagnostics = getattr(result, "diagnostics", None)
    if not isinstance(diagnostics, dict):
        raise TypeError("result.diagnostics must be a dict")

    iter2 = _array_from_diag(diagnostics, "iter2_history", dtype=int)
    step_status = _array_from_diag(diagnostics, "step_status_history", dtype=object)
    restart_reason = _array_from_diag(diagnostics, "restart_reason_history", dtype=object)
    pre_restart_reason = _array_from_diag(diagnostics, "pre_restart_reason_history", dtype=object)
    time_step = _array_from_diag(diagnostics, "time_step_history", dtype=float)
    dt_eff = _array_from_diag(diagnostics, "dt_eff_history", dtype=float)
    update_rms = _array_from_diag(diagnostics, "update_rms_history", dtype=float)
    include_edge = _array_from_diag(diagnostics, "include_edge_history", dtype=int)
    zero_m1 = _array_from_diag(diagnostics, "zero_m1_history", dtype=int)
    fsq_curr = _array_from_diag(diagnostics, "w_curr_history", dtype=float)
    fsq_try = _array_from_diag(diagnostics, "w_try_history", dtype=float)
    fsq_prev = _array_from_diag(diagnostics, "fsq_prev_history", dtype=float)
    r00 = _array_from_diag(diagnostics, "r00_history", dtype=float)
    z00 = _array_from_diag(diagnostics, "z00_history", dtype=float)
    wb = _array_from_diag(diagnostics, "wb_history", dtype=float)
    wp = _array_from_diag(diagnostics, "wp_history", dtype=float)
    w_vmec = _array_from_diag(diagnostics, "w_vmec_history", dtype=float)

    lengths = {
        int(arr.shape[0])
        for arr in (
            iter2,
            step_status,
            restart_reason,
            pre_restart_reason,
            time_step,
            dt_eff,
            update_rms,
            include_edge,
            zero_m1,
            fsq_curr,
            fsq_try,
            fsq_prev,
            r00,
            z00,
            wb,
            wp,
            w_vmec,
        )
        if arr.ndim >= 1 and arr.shape[0] > 0
    }
    if len(lengths) > 1:
        raise ValueError(f"inconsistent residual trace lengths: {sorted(lengths)}")

    rejected = np.isin(
        step_status,
        np.asarray(["rejected", "restart_bad_progress", "restart_bad_jacobian"], dtype=object),
    )
    state_advanced = ~rejected

    return ResidualIterationTrace(
        iter2=iter2,
        step_status=step_status,
        restart_reason=restart_reason,
        pre_restart_reason=pre_restart_reason,
        time_step=time_step,
        dt_eff=dt_eff,
        update_rms=update_rms,
        include_edge=include_edge,
        zero_m1=zero_m1,
        fsq_curr=fsq_curr,
        fsq_try=fsq_try,
        fsq_prev=fsq_prev,
        r00=r00,
        z00=z00,
        wb=wb,
        wp=wp,
        w_vmec=w_vmec,
        state_advanced=state_advanced,
    )


def concat_residual_iteration_traces(traces: list[ResidualIterationTrace]) -> ResidualIterationTrace:
    """Concatenate per-call residual traces into one longer trace."""
    if not traces:
        empty_i = np.zeros((0,), dtype=int)
        empty_f = np.zeros((0,), dtype=float)
        empty_o = np.zeros((0,), dtype=object)
        empty_b = np.zeros((0,), dtype=bool)
        return ResidualIterationTrace(
            iter2=empty_i,
            step_status=empty_o,
            restart_reason=empty_o,
            pre_restart_reason=empty_o,
            time_step=empty_f,
            dt_eff=empty_f,
            update_rms=empty_f,
            include_edge=empty_i,
            zero_m1=empty_i,
            fsq_curr=empty_f,
            fsq_try=empty_f,
            fsq_prev=empty_f,
            r00=empty_f,
            z00=empty_f,
            wb=empty_f,
            wp=empty_f,
            w_vmec=empty_f,
            state_advanced=empty_b,
        )

    def _cat(name: str) -> np.ndarray:
        parts = [np.asarray(getattr(trace, name)) for trace in traces]
        return np.concatenate(parts, axis=0)

    return ResidualIterationTrace(
        iter2=_cat("iter2").astype(int, copy=False),
        step_status=_cat("step_status").astype(object, copy=False),
        restart_reason=_cat("restart_reason").astype(object, copy=False),
        pre_restart_reason=_cat("pre_restart_reason").astype(object, copy=False),
        time_step=_cat("time_step").astype(float, copy=False),
        dt_eff=_cat("dt_eff").astype(float, copy=False),
        update_rms=_cat("update_rms").astype(float, copy=False),
        include_edge=_cat("include_edge").astype(int, copy=False),
        zero_m1=_cat("zero_m1").astype(int, copy=False),
        fsq_curr=_cat("fsq_curr").astype(float, copy=False),
        fsq_try=_cat("fsq_try").astype(float, copy=False),
        fsq_prev=_cat("fsq_prev").astype(float, copy=False),
        r00=_cat("r00").astype(float, copy=False),
        z00=_cat("z00").astype(float, copy=False),
        wb=_cat("wb").astype(float, copy=False),
        wp=_cat("wp").astype(float, copy=False),
        w_vmec=_cat("w_vmec").astype(float, copy=False),
        state_advanced=_cat("state_advanced").astype(bool, copy=False),
    )


def build_residual_checkpoint_tape(
    state0,
    static,
    *,
    indata,
    signgs: int,
    max_iter: int,
    ftol: float | None = None,
    step_size: float = 1.0,
    resume_state_mode: str = "minimal",
    light_history: bool = True,
    solver_kwargs: dict[str, Any] | None = None,
) -> ResidualCheckpointTape:
    """Replay the residual solver in one-step chunks and collect checkpoints."""
    from .solve import solve_fixed_boundary_residual_iter

    solver_kwargs = dict(solver_kwargs or {})
    solve_kwargs = dict(solver_kwargs)
    solve_kwargs.setdefault("indata", indata)
    solve_kwargs.setdefault("signgs", int(signgs))
    solve_kwargs.setdefault("ftol", ftol)
    solve_kwargs.setdefault("step_size", float(step_size))
    # The checkpoint tape needs per-iteration scalar histories even when the
    # caller wants minimal resume checkpoints, so force full history capture
    # here and keep compactness in the saved resume_state dictionaries instead.
    solve_kwargs["light_history"] = False
    # Multi-step replay currently needs the cached preconditioner/control state
    # carried in the full resume checkpoint. A later optimization pass can
    # shrink this once exact replay coverage is in place.
    solve_kwargs["resume_state_mode"] = "full"
    state = state0
    resume_state = None
    traces: list[ResidualIterationTrace] = []
    packed_states: list[np.ndarray] = []
    resume_states: list[dict[str, Any] | None] = []
    step_traces: list[dict[str, Any]] = []

    for _ in range(int(max_iter)):
        result = replay_residual_checkpoint_step(
            state,
            static,
            resume_state=resume_state,
            solve_kwargs=solve_kwargs,
        )
        state = result.state
        packed_states.append(np.asarray(pack_state(state), dtype=float))
        traces.append(residual_iteration_trace_from_result(result))
        resume_state = result.diagnostics.get("resume_state")
        resume_states.append(resume_state)
        step_traces.extend(list(result.diagnostics.get("adjoint_step_trace", [])))
        if bool(result.diagnostics.get("converged", False)):
            break

    if packed_states:
        packed_states_arr = np.stack(packed_states, axis=0)
    else:
        packed_states_arr = np.zeros((0, int(state0.layout.size)), dtype=float)

    return ResidualCheckpointTape(
        packed_states=packed_states_arr,
        trace=concat_residual_iteration_traces(traces),
        resume_states=tuple(resume_states),
        step_traces=tuple(step_traces),
    )


def replay_residual_checkpoint_step(
    state,
    static,
    *,
    resume_state: dict[str, Any] | None,
    solve_kwargs: dict[str, Any],
):
    """Replay exactly one residual-solver step from a stored checkpoint."""
    from .solve import solve_fixed_boundary_residual_iter

    return solve_fixed_boundary_residual_iter(
        state,
        static,
        max_iter=1,
        resume_state=resume_state,
        adjoint_trace=True,
        **solve_kwargs,
    )


def strict_update_velocity_block(
    *,
    b1,
    fac,
    force_scale,
    flip_sign,
    vRcc_before,
    vRss_before,
    vZsc_before,
    vZcs_before,
    vLsc_before,
    vLcs_before,
    frcc_u,
    frss_u,
    fzsc_u,
    fzcs_u,
    flsc_u,
    flcs_u,
    vRsc_before=None,
    vRcs_before=None,
    vZcc_before=None,
    vZss_before=None,
    vLcc_before=None,
    vLss_before=None,
    frsc_u=None,
    frcs_u=None,
    fzcc_u=None,
    fzss_u=None,
    flcc_u=None,
    flss_u=None,
):
    """Apply the strict-update velocity recurrence for one solver step."""
    b1 = jnp.asarray(b1, dtype=jnp.asarray(vRcc_before).dtype)
    fac = jnp.asarray(fac, dtype=jnp.asarray(vRcc_before).dtype)
    force_scale = jnp.asarray(force_scale, dtype=jnp.asarray(vRcc_before).dtype)
    flip_sign = jnp.asarray(flip_sign, dtype=jnp.asarray(vRcc_before).dtype)
    scale = fac * force_scale * flip_sign
    memory = fac * b1

    def _update(v_before, force):
        return memory * jnp.asarray(v_before) + scale * jnp.asarray(force)

    vRcc_after = _update(vRcc_before, frcc_u)
    vRss_after = _update(vRss_before, frss_u)
    vZsc_after = _update(vZsc_before, fzsc_u)
    vZcs_after = _update(vZcs_before, fzcs_u)
    vLsc_after = _update(vLsc_before, flsc_u)
    vLcs_after = _update(vLcs_before, flcs_u)
    if vRsc_before is None:
        vRsc_after = None
        vRcs_after = None
        vZcc_after = None
        vZss_after = None
        vLcc_after = None
        vLss_after = None
    else:
        vRsc_after = _update(vRsc_before, frsc_u)
        vRcs_after = _update(vRcs_before, frcs_u)
        vZcc_after = _update(vZcc_before, fzcc_u)
        vZss_after = _update(vZss_before, fzss_u)
        vLcc_after = _update(vLcc_before, flcc_u)
        vLss_after = _update(vLss_before, flss_u)
    return {
        "vRcc_after": vRcc_after,
        "vRss_after": vRss_after,
        "vZsc_after": vZsc_after,
        "vZcs_after": vZcs_after,
        "vLsc_after": vLsc_after,
        "vLcs_after": vLcs_after,
        "vRsc_after": vRsc_after,
        "vRcs_after": vRcs_after,
        "vZcc_after": vZcc_after,
        "vZss_after": vZss_after,
        "vLcc_after": vLcc_after,
        "vLss_after": vLss_after,
    }


def strict_update_velocity_limit(
    *,
    dt_eff,
    max_update_rms,
    limit_update_rms,
    vRcc,
    vRss,
    vZsc,
    vZcs,
    vLsc,
    vLcs,
    vRsc=None,
    vRcs=None,
    vZcc=None,
    vZss=None,
    vLcc=None,
    vLss=None,
):
    """Apply the strict-update velocity RMS limiter for one solver step."""
    dt_eff = jnp.asarray(dt_eff, dtype=jnp.asarray(vRcc).dtype)
    max_update_rms = jnp.asarray(max_update_rms, dtype=jnp.asarray(vRcc).dtype)
    limit_update_rms = bool(limit_update_rms)
    base = jnp.asarray(vRcc)
    zeros = jnp.zeros_like(base)
    pieces = [
        base,
        jnp.asarray(vRss),
        jnp.asarray(vRsc) if vRsc is not None else zeros,
        jnp.asarray(vRcs) if vRcs is not None else zeros,
        jnp.asarray(vZsc),
        jnp.asarray(vZcs),
        jnp.asarray(vZcc) if vZcc is not None else zeros,
        jnp.asarray(vZss) if vZss is not None else zeros,
        jnp.asarray(vLsc),
        jnp.asarray(vLcs),
        jnp.asarray(vLcc) if vLcc is not None else zeros,
        jnp.asarray(vLss) if vLss is not None else zeros,
    ]
    sq = sum((dt_eff * p) ** 2 for p in pieces)
    update_rms = jnp.sqrt(jnp.mean(sq))
    if limit_update_rms:
        scale = jnp.where(
            jnp.isfinite(update_rms) & (update_rms > max_update_rms),
            max_update_rms / jnp.maximum(update_rms, jnp.asarray(1.0e-30, dtype=update_rms.dtype)),
            jnp.asarray(1.0, dtype=update_rms.dtype),
        )
    else:
        scale = jnp.asarray(1.0, dtype=update_rms.dtype)

    def _scale(x):
        return None if x is None else scale * jnp.asarray(x)

    out = {
        "vRcc": _scale(vRcc),
        "vRss": _scale(vRss),
        "vZsc": _scale(vZsc),
        "vZcs": _scale(vZcs),
        "vLsc": _scale(vLsc),
        "vLcs": _scale(vLcs),
        "vRsc": _scale(vRsc),
        "vRcs": _scale(vRcs),
        "vZcc": _scale(vZcc),
        "vZss": _scale(vZss),
        "vLcc": _scale(vLcc),
        "vLss": _scale(vLss),
        "update_rms_preclip": update_rms,
        "update_rms_scale": scale,
        "update_rms_postclip": scale * update_rms,
    }
    return out


def preconditioned_force_channels_from_rz_output(
    *,
    frzl_rz,
    lam_prec,
    w_mode_mn,
    lambda_update_scale=1.0,
):
    """Map R/Z preconditioner output into solver force channels for one step."""
    frcc = jnp.asarray(frzl_rz.frcc)
    zeros_r = jnp.zeros_like(frcc)
    fzsc = jnp.asarray(frzl_rz.fzsc)
    zeros_z = jnp.zeros_like(fzsc)
    flsc = jnp.asarray(frzl_rz.flsc) * jnp.asarray(lam_prec)
    zeros_l = jnp.zeros_like(flsc)

    frss = None if frzl_rz.frss is None else jnp.asarray(frzl_rz.frss)
    fzcs = None if frzl_rz.fzcs is None else jnp.asarray(frzl_rz.fzcs)
    flcs = None if frzl_rz.flcs is None else (jnp.asarray(frzl_rz.flcs) * jnp.asarray(lam_prec))
    frsc = jnp.asarray(frzl_rz.frsc) if getattr(frzl_rz, "frsc", None) is not None else zeros_r
    frcs = jnp.asarray(frzl_rz.frcs) if getattr(frzl_rz, "frcs", None) is not None else zeros_r
    fzcc = jnp.asarray(frzl_rz.fzcc) if getattr(frzl_rz, "fzcc", None) is not None else zeros_z
    fzss = jnp.asarray(frzl_rz.fzss) if getattr(frzl_rz, "fzss", None) is not None else zeros_z
    flcc = (
        jnp.asarray(frzl_rz.flcc) * jnp.asarray(lam_prec)
        if getattr(frzl_rz, "flcc", None) is not None
        else zeros_l
    )
    flss = (
        jnp.asarray(frzl_rz.flss) * jnp.asarray(lam_prec)
        if getattr(frzl_rz, "flss", None) is not None
        else zeros_l
    )

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
    w = jnp.asarray(w_mode_mn)[None, :, :]
    frcc_u = frcc * w
    frss_u = (frss if frss is not None else zeros_r) * w
    fzsc_u = fzsc * w
    fzcs_u = (fzcs if fzcs is not None else zeros_z) * w
    flsc_u = flsc * w
    flcs_u = (flcs if flcs is not None else zeros_l) * w
    frsc_u = frsc * w
    frcs_u = frcs * w
    fzcc_u = fzcc * w
    fzss_u = fzss * w
    flcc_u = flcc * w
    flss_u = flss * w
    if float(lambda_update_scale) != 1.0:
        scale = jnp.asarray(lambda_update_scale, dtype=flsc_u.dtype)
        flsc_u = flsc_u * scale
        flcs_u = flcs_u * scale
        flcc_u = flcc_u * scale
        flss_u = flss_u * scale
    return {
        "frzl_pre": frzl_pre,
        "frcc_u": frcc_u,
        "frss_u": frss_u,
        "fzsc_u": fzsc_u,
        "fzcs_u": fzcs_u,
        "flsc_u": flsc_u,
        "flcs_u": flcs_u,
        "frsc_u": frsc_u,
        "frcs_u": frcs_u,
        "fzcc_u": fzcc_u,
        "fzss_u": fzss_u,
        "flcc_u": flcc_u,
        "flss_u": flss_u,
    }


def preconditioned_force_channels_from_raw_forces(
    *,
    frzl,
    mats,
    jmax,
    cfg,
    lam_prec,
    w_mode_mn,
    lambda_update_scale=1.0,
):
    """Apply the radial preconditioner, lambda scaling, and mode scaling."""
    from .preconditioner_1d_jax import rz_preconditioner_apply_jit

    frzl_rz = rz_preconditioner_apply_jit(
        frzl_in=frzl,
        mats=mats,
        jmax=int(jmax),
        cfg=cfg,
    )
    out = preconditioned_force_channels_from_rz_output(
        frzl_rz=frzl_rz,
        lam_prec=lam_prec,
        w_mode_mn=w_mode_mn,
        lambda_update_scale=lambda_update_scale,
    )
    return {"frzl_rz": frzl_rz, **out}


def strict_update_accepted_step(
    state_pre,
    static,
    *,
    dt_eff,
    b1,
    fac,
    force_scale,
    flip_sign,
    vRcc_before,
    vRss_before,
    vZsc_before,
    vZcs_before,
    vLsc_before,
    vLcs_before,
    frcc_u,
    frss_u,
    fzsc_u,
    fzcs_u,
    flsc_u,
    flcs_u,
    vRsc_before=None,
    vRcs_before=None,
    vZcc_before=None,
    vZss_before=None,
    vLcc_before=None,
    vLss_before=None,
    frsc_u=None,
    frcs_u=None,
    fzcc_u=None,
    fzss_u=None,
    flcc_u=None,
    flss_u=None,
    max_update_rms=5.0e-3,
    limit_update_rms: bool = True,
    divide_by_scalxc_for_update: bool = False,
):
    """Compose the accepted strict-update velocity and state-advance blocks."""
    velocity_raw = strict_update_velocity_block(
        b1=b1,
        fac=fac,
        force_scale=force_scale,
        flip_sign=flip_sign,
        vRcc_before=vRcc_before,
        vRss_before=vRss_before,
        vZsc_before=vZsc_before,
        vZcs_before=vZcs_before,
        vLsc_before=vLsc_before,
        vLcs_before=vLcs_before,
        frcc_u=frcc_u,
        frss_u=frss_u,
        fzsc_u=fzsc_u,
        fzcs_u=fzcs_u,
        flsc_u=flsc_u,
        flcs_u=flcs_u,
        vRsc_before=vRsc_before,
        vRcs_before=vRcs_before,
        vZcc_before=vZcc_before,
        vZss_before=vZss_before,
        vLcc_before=vLcc_before,
        vLss_before=vLss_before,
        frsc_u=frsc_u,
        frcs_u=frcs_u,
        fzcc_u=fzcc_u,
        fzss_u=fzss_u,
        flcc_u=flcc_u,
        flss_u=flss_u,
    )
    velocity_out = strict_update_velocity_limit(
        dt_eff=dt_eff,
        max_update_rms=max_update_rms,
        limit_update_rms=limit_update_rms,
        vRcc=velocity_raw["vRcc_after"],
        vRss=velocity_raw["vRss_after"],
        vZsc=velocity_raw["vZsc_after"],
        vZcs=velocity_raw["vZcs_after"],
        vLsc=velocity_raw["vLsc_after"],
        vLcs=velocity_raw["vLcs_after"],
        vRsc=velocity_raw["vRsc_after"],
        vRcs=velocity_raw["vRcs_after"],
        vZcc=velocity_raw["vZcc_after"],
        vZss=velocity_raw["vZss_after"],
        vLcc=velocity_raw["vLcc_after"],
        vLss=velocity_raw["vLss_after"],
    )
    state_post = strict_update_velocity_state_advance(
        state_pre,
        static,
        dt_eff=dt_eff,
        vRcc=velocity_out["vRcc"],
        vRss=velocity_out["vRss"],
        vZsc=velocity_out["vZsc"],
        vZcs=velocity_out["vZcs"],
        vLsc=velocity_out["vLsc"],
        vLcs=velocity_out["vLcs"],
        vRsc=velocity_out["vRsc"],
        vRcs=velocity_out["vRcs"],
        vZcc=velocity_out["vZcc"],
        vZss=velocity_out["vZss"],
        vLcc=velocity_out["vLcc"],
        vLss=velocity_out["vLss"],
        edge_Rcos=jnp.asarray(state_pre.Rcos)[-1, :],
        edge_Rsin=jnp.asarray(state_pre.Rsin)[-1, :],
        edge_Zcos=jnp.asarray(state_pre.Zcos)[-1, :],
        edge_Zsin=jnp.asarray(state_pre.Zsin)[-1, :],
        divide_by_scalxc_for_update=divide_by_scalxc_for_update,
    )
    return {
        "state_post": state_post,
        "vRcc_after": velocity_out["vRcc"],
        "vRss_after": velocity_out["vRss"],
        "vZsc_after": velocity_out["vZsc"],
        "vZcs_after": velocity_out["vZcs"],
        "vLsc_after": velocity_out["vLsc"],
        "vLcs_after": velocity_out["vLcs"],
        "vRsc_after": velocity_out["vRsc"],
        "vRcs_after": velocity_out["vRcs"],
        "vZcc_after": velocity_out["vZcc"],
        "vZss_after": velocity_out["vZss"],
        "vLcc_after": velocity_out["vLcc"],
        "vLss_after": velocity_out["vLss"],
        "update_rms_preclip": velocity_out["update_rms_preclip"],
        "update_rms_scale": velocity_out["update_rms_scale"],
        "update_rms_postclip": velocity_out["update_rms_postclip"],
    }


def strict_update_velocity_state_advance(
    state,
    static,
    *,
    dt_eff,
    vRcc,
    vRss,
    vZsc,
    vZcs,
    vLsc,
    vLcs,
    edge_Rcos,
    edge_Rsin,
    edge_Zcos,
    edge_Zsin,
    vRsc=None,
    vRcs=None,
    vZcc=None,
    vZss=None,
    vLcc=None,
    vLss=None,
    divide_by_scalxc_for_update: bool = False,
):
    """Apply the strict-update state-advance block from VMEC residual iteration.

    This is the accepted geometry/lambda update map after the velocity blocks
    have already been formed for the current step. It excludes force assembly
    and acceptance/restart logic and is intended as the first local reverse-mode
    target for the discrete-adjoint refactor.
    """
    from .solve import _enforce_fixed_boundary_and_axis, _enforce_lambda_gauge, _mode00_index
    from .vmec_parity import _mn_cos_to_signed_cached, _mn_sin_to_signed_cached, signed_maps_from_modes
    from .vmec_residue import vmec_scalxc_from_s

    dt_eff = jnp.asarray(dt_eff, dtype=jnp.asarray(state.Rcos).dtype)
    scalxc = vmec_scalxc_from_s(s=jnp.asarray(static.s), mpol=int(static.cfg.mpol)).astype(jnp.asarray(state.Rcos).dtype)
    scalxc = scalxc[:, :, None]
    if not bool(divide_by_scalxc_for_update):
        scalxc = jnp.ones_like(scalxc)
    maps = static.signed_maps if getattr(static, "signed_maps", None) is not None else signed_maps_from_modes(static.modes)
    ncoeff = int(static.modes.K)
    idx00 = _mode00_index(static.modes)

    def _cos_phys(cc, ss):
        cc = jnp.asarray(cc) / scalxc
        ss = jnp.asarray(ss) / scalxc if ss is not None else None
        return _mn_cos_to_signed_cached(cc, ss, maps=maps, ncoeff=ncoeff)

    def _sin_phys(sc, cs):
        sc = jnp.asarray(sc) / scalxc
        cs = jnp.asarray(cs) / scalxc if cs is not None else None
        return _mn_sin_to_signed_cached(sc, cs, maps=maps, ncoeff=ncoeff)

    dR = dt_eff * _cos_phys(vRcc, vRss)
    dZ = dt_eff * _sin_phys(vZsc, vZcs)
    dL = dt_eff * _sin_phys(vLsc, vLcs)
    if bool(static.cfg.lasym):
        dR_sin = dt_eff * _sin_phys(vRsc, vRcs)
        dZ_cos = dt_eff * _cos_phys(vZcc, vZss)
        dL_cos = dt_eff * _cos_phys(vLcc, vLss)
    else:
        dR_sin = jnp.zeros_like(dR)
        dZ_cos = jnp.zeros_like(dR)
        dL_cos = jnp.zeros_like(dR)

    state_try = type(state)(
        layout=state.layout,
        Rcos=jnp.asarray(state.Rcos) + dR,
        Rsin=jnp.asarray(state.Rsin) + dR_sin,
        Zcos=jnp.asarray(state.Zcos) + dZ_cos,
        Zsin=jnp.asarray(state.Zsin) + dZ,
        Lcos=jnp.asarray(state.Lcos) + dL_cos,
        Lsin=jnp.asarray(state.Lsin) + dL,
    )
    return _enforce_fixed_boundary_and_axis(
        state_try,
        static,
        edge_Rcos=edge_Rcos,
        edge_Rsin=edge_Rsin,
        edge_Zcos=edge_Zcos,
        edge_Zsin=edge_Zsin,
        enforce_edge=True,
        enforce_lambda_axis=True,
        idx00=idx00,
    )
    Lcos, Lsin = _enforce_lambda_gauge(
        jnp.asarray(state_try.Lcos),
        jnp.asarray(state_try.Lsin),
        idx00=idx00,
    )
    return type(state_try)(
        layout=state_try.layout,
        Rcos=state_try.Rcos,
        Rsin=state_try.Rsin,
        Zcos=state_try.Zcos,
        Zsin=state_try.Zsin,
        Lcos=Lcos,
        Lsin=Lsin,
    )


__all__ = [
    "ResidualIterationTrace",
    "ResidualCheckpointTape",
    "build_residual_checkpoint_tape",
    "concat_residual_iteration_traces",
    "preconditioned_force_channels_from_raw_forces",
    "preconditioned_force_channels_from_rz_output",
    "replay_residual_checkpoint_step",
    "strict_update_accepted_step",
    "strict_update_velocity_limit",
    "strict_update_velocity_block",
    "strict_update_velocity_state_advance",
    "residual_iteration_trace_from_result",
]
