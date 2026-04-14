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
    from .solve import _enforce_fixed_boundary_and_axis, _mode00_index
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


__all__ = [
    "ResidualIterationTrace",
    "ResidualCheckpointTape",
    "build_residual_checkpoint_tape",
    "concat_residual_iteration_traces",
    "replay_residual_checkpoint_step",
    "strict_update_velocity_state_advance",
    "residual_iteration_trace_from_result",
]
