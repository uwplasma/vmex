"""Pure math helpers for the residual-iteration scan loop."""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from ._compat import jnp


class ScanTauDecision(NamedTuple):
    bad_jacobian: Any
    min_tau: Any
    max_tau: Any


class ScanRestartUpdates(NamedTuple):
    state: Any
    time_step: Any
    inv_tau: Any
    fsq_prev: Any
    vRcc: Any
    vRss: Any
    vZsc: Any
    vZcs: Any
    vLsc: Any
    vLcs: Any
    vRsc: Any
    vRcs: Any
    vZcc: Any
    vZss: Any
    vLcc: Any
    vLss: Any
    iter_offset: Any
    iter1: Any
    ijacob: Any
    bad_resets: Any
    bad_growth: Any
    force_bcovar_update: Any


def _kernel_arrays_from_k(k: Any) -> tuple[Any, ...] | None:
    try:
        return (
            getattr(k, "pru_even"),
            getattr(k, "pru_odd"),
            getattr(k, "pzu_even"),
            getattr(k, "pzu_odd"),
            getattr(k, "pr1_even"),
            getattr(k, "pr1_odd"),
            getattr(k, "pz1_even"),
            getattr(k, "pz1_odd"),
        )
    except Exception:
        return None


def _ptau_values_np(
    *,
    pru_even: Any,
    pru_odd: Any,
    pzu_even: Any,
    pzu_odd: Any,
    pr1_even: Any,
    pr1_odd: Any,
    pz1_even: Any,
    pz1_odd: Any,
    pshalf: Any,
    ohs: float,
) -> np.ndarray | None:
    pru_even_np = np.asarray(pru_even)
    ns = int(pru_even_np.shape[0])
    if ns < 2:
        return None
    pru_odd_np = np.asarray(pru_odd)
    pzu_even_np = np.asarray(pzu_even)
    pzu_odd_np = np.asarray(pzu_odd)
    pr1_even_np = np.asarray(pr1_even)
    pr1_odd_np = np.asarray(pr1_odd)
    pz1_even_np = np.asarray(pz1_even)
    pz1_odd_np = np.asarray(pz1_odd)
    pshalf_np = np.asarray(pshalf)
    if int(pshalf_np.shape[0]) != ns:
        pshalf_np = np.resize(pshalf_np, (ns,))
    dphids = 0.25
    psh = pshalf_np[1:, None, None]
    psh_safe = np.where(psh != 0.0, psh, 1.0)
    ru12 = 0.5 * (pru_even_np[1:] + pru_even_np[:-1] + psh * (pru_odd_np[1:] + pru_odd_np[:-1]))
    pzs = ohs * ((pz1_even_np[1:] - pz1_even_np[:-1]) + psh * (pz1_odd_np[1:] - pz1_odd_np[:-1]))
    ptau = ru12 * pzs + dphids * (
        pru_odd_np[1:] * pz1_odd_np[1:]
        + pru_odd_np[:-1] * pz1_odd_np[:-1]
        + (pru_even_np[1:] * pz1_odd_np[1:] + pru_even_np[:-1] * pz1_odd_np[:-1]) / psh_safe
    )
    pzu12 = 0.5 * (pzu_even_np[1:] + pzu_even_np[:-1] + psh * (pzu_odd_np[1:] + pzu_odd_np[:-1]))
    prs = ohs * ((pr1_even_np[1:] - pr1_even_np[:-1]) + psh * (pr1_odd_np[1:] - pr1_odd_np[:-1]))
    return ptau - prs * pzu12 - dphids * (
        pzu_odd_np[1:] * pr1_odd_np[1:]
        + pzu_odd_np[:-1] * pr1_odd_np[:-1]
        + (pzu_even_np[1:] * pr1_odd_np[1:] + pzu_even_np[:-1] * pr1_odd_np[:-1]) / psh_safe
    )


def _ptau_minmax_from_k_host(
    k: Any,
    *,
    pshalf: Any,
    ohs: float,
    compute_jit: Any = None,
    pshalf_jax: Any = None,
    ohs_jax: Any = None,
) -> tuple[Any | None, Any | None]:
    """Compute VMEC ``ptau`` min/max from kernel arrays with host fallbacks."""
    arrays = _kernel_arrays_from_k(k)
    if arrays is None:
        return None, None
    try:
        if compute_jit is not None:
            ns = int(arrays[0].shape[0]) if hasattr(arrays[0], "shape") else 0
            if ns < 2:
                return None, None
            ptau_min_j, ptau_max_j = compute_jit(*arrays, pshalf_jax, ohs_jax)
            return float(ptau_min_j), float(ptau_max_j)

        ptau = _ptau_values_np(
            pru_even=arrays[0],
            pru_odd=arrays[1],
            pzu_even=arrays[2],
            pzu_odd=arrays[3],
            pr1_even=arrays[4],
            pr1_odd=arrays[5],
            pz1_even=arrays[6],
            pz1_odd=arrays[7],
            pshalf=pshalf,
            ohs=ohs,
        )
        if ptau is None:
            return None, None
        return float(np.min(ptau)), float(np.max(ptau))
    except Exception:
        return None, None


def _ptau_minmax_from_k_jax(k: Any, *, s: Any, pshalf_from_s_jax: Any):
    arrays = _kernel_arrays_from_k(k)
    if arrays is None:
        nan_val = jnp.asarray(jnp.nan)
        return nan_val, nan_val
    pru_even, pru_odd, pzu_even, pzu_odd, pr1_even, pr1_odd, pz1_even, pz1_odd = (
        jnp.asarray(arr) for arr in arrays
    )
    ns = jnp.asarray(pru_even).shape[0]
    nan_val = jnp.asarray(jnp.nan, dtype=pru_even.dtype)
    if int(ns) < 2:
        return nan_val, nan_val

    def _compute(_):
        pshalf = pshalf_from_s_jax(s, dtype=pru_even.dtype)
        pshalf_loc = jnp.resize(pshalf, (ns,)) if int(pshalf.shape[0]) != int(ns) else pshalf
        hs = jnp.asarray(s[1] - s[0]) if int(jnp.asarray(s).shape[0]) > 1 else jnp.asarray(1.0, dtype=pru_even.dtype)
        ohs = jnp.where(hs != 0.0, 1.0 / hs, jnp.asarray(0.0, dtype=pru_even.dtype))
        dphids = jnp.asarray(0.25, dtype=pru_even.dtype)
        psh = pshalf_loc[1:][:, None, None]
        psh_safe = jnp.where(psh != 0.0, psh, jnp.asarray(1.0, dtype=pru_even.dtype))
        ru12 = 0.5 * (pru_even[1:] + pru_even[:-1] + psh * (pru_odd[1:] + pru_odd[:-1]))
        pzs = ohs * ((pz1_even[1:] - pz1_even[:-1]) + psh * (pz1_odd[1:] - pz1_odd[:-1]))
        ptau = ru12 * pzs + dphids * (
            pru_odd[1:] * pz1_odd[1:]
            + pru_odd[:-1] * pz1_odd[:-1]
            + (pru_even[1:] * pz1_odd[1:] + pru_even[:-1] * pz1_odd[:-1]) / psh_safe
        )
        pzu12 = 0.5 * (pzu_even[1:] + pzu_even[:-1] + psh * (pzu_odd[1:] + pzu_odd[:-1]))
        prs = ohs * ((pr1_even[1:] - pr1_even[:-1]) + psh * (pr1_odd[1:] - pr1_odd[:-1]))
        ptau = ptau - prs * pzu12 - dphids * (
            pzu_odd[1:] * pr1_odd[1:]
            + pzu_odd[:-1] * pr1_odd[:-1]
            + (pzu_even[1:] * pr1_odd[1:] + pzu_even[:-1] * pr1_odd[:-1]) / psh_safe
        )
        return jnp.min(ptau), jnp.max(ptau)

    return _compute(None)


def _state_jacobian(
    tau: Any,
    *,
    vmec2000_control: bool,
    ptau_tol: float,
    relative_tol: float | None = None,
) -> ScanTauDecision:
    """Return bad-Jacobian flag and tau min/max for a half-mesh Jacobian array."""
    tau = jnp.asarray(tau)
    dtype = tau.dtype
    if int(tau.size) == 0:
        nan_val = jnp.asarray(jnp.nan, dtype=dtype)
        return ScanTauDecision(jnp.asarray(False), nan_val, nan_val)
    tau_use = tau[1:] if int(tau.shape[0]) > 1 else tau
    min_tau = jnp.min(tau_use)
    max_tau = jnp.max(tau_use)
    if relative_tol is not None:
        tau_scale = jnp.maximum(jnp.abs(min_tau), jnp.abs(max_tau))
        tau_tol = jnp.maximum(
            jnp.asarray(1.0e-12, dtype=dtype),
            jnp.asarray(float(relative_tol), dtype=dtype) * tau_scale,
        )
    elif bool(vmec2000_control):
        tau_tol = jnp.asarray(abs(ptau_tol), dtype=dtype)
    else:
        tau_scale = jnp.maximum(jnp.abs(min_tau), jnp.abs(max_tau))
        tau_tol = jnp.maximum(jnp.asarray(1.0e-12, dtype=dtype), jnp.asarray(1.0e-3, dtype=dtype) * tau_scale)
    finite = jnp.isfinite(min_tau) & jnp.isfinite(max_tau)
    bad_state = finite & (min_tau < -tau_tol) & (max_tau > tau_tol)
    return ScanTauDecision(bad_state, min_tau, max_tau)


def _hold_step(
    carry_hold: Any,
    *,
    dtype: Any,
    state_only_scan: bool,
    scan_minimal: bool,
    scan_light: bool,
    scan_hist_min: Any,
    scan_hist_light: Any,
):
    """Return the carry/history pair emitted when a scan iteration is held."""
    accepted_h = jnp.asarray(False)
    if state_only_scan:
        return carry_hold, ()
    if scan_minimal:
        return carry_hold, scan_hist_min(carry_hold.fsqr_prev_phys, carry_hold.fsqz_prev_phys, carry_hold.fsql_prev_phys)
    if scan_light:
        return carry_hold, scan_hist_light(
            carry_hold.fsqr_prev_phys,
            carry_hold.fsqz_prev_phys,
            carry_hold.fsql_prev_phys,
            accepted_h,
            carry_hold.r00_prev,
            carry_hold.z00_prev,
            carry_hold.w_mhd_prev,
            carry_hold.time_step,
            jnp.asarray(False),
        )
    return carry_hold, (
        carry_hold.fsqr_prev_phys,
        carry_hold.fsqz_prev_phys,
        carry_hold.fsql_prev_phys,
        carry_hold.fsqr1_prev,
        carry_hold.fsqz1_prev,
        carry_hold.fsql1_prev,
        accepted_h,
        carry_hold.r00_prev,
        carry_hold.z00_prev,
        carry_hold.w_mhd_prev,
        carry_hold.time_step,
        jnp.asarray(0.0, dtype=dtype),
        jnp.asarray(False),
        carry_hold.res0,
        carry_hold.res1,
        carry_hold.iter1,
        jnp.asarray(False),
        jnp.asarray(0.0, dtype=dtype),
        jnp.asarray(0.0, dtype=dtype),
        jnp.asarray(jnp.nan, dtype=dtype),
        jnp.asarray(jnp.nan, dtype=dtype),
        jnp.asarray(jnp.nan, dtype=dtype),
        jnp.asarray(jnp.nan, dtype=dtype),
        jnp.asarray(False),
        jnp.asarray(False),
    )


def _restart_updates(
    *,
    carry_adv: Any,
    state_checkpoint: Any,
    fsq_prev_before: Any,
    iter2: Any,
    restart_reason: Any,
    vmec2000_control: bool,
    restart_badjac_factor: float,
    restart_badprog_factor: float,
    stage_transition_scale: float,
    step_size: float,
    k_ndamp: int,
    dtype: Any,
    scan_restart_transition_fn: Any,
) -> ScanRestartUpdates:
    restart_scalars = scan_restart_transition_fn(
        time_step=carry_adv.time_step,
        iter_offset=carry_adv.iter_offset,
        ijacob=carry_adv.ijacob,
        bad_resets=carry_adv.bad_resets,
        iter2=iter2,
        restart_reason=restart_reason,
        vmec2000_control=bool(vmec2000_control),
        restart_badjac_factor=restart_badjac_factor,
        restart_badprog_factor=restart_badprog_factor,
        stage_transition_scale=stage_transition_scale,
        step_size=step_size,
    )
    zero_like_r = jnp.zeros_like(carry_adv.vRcc)
    return ScanRestartUpdates(
        state_checkpoint,
        restart_scalars.time_step,
        jnp.full((k_ndamp,), jnp.asarray(0.15, dtype=dtype) / restart_scalars.damping_time_step),
        fsq_prev_before,
        zero_like_r,
        jnp.zeros_like(carry_adv.vRss),
        jnp.zeros_like(carry_adv.vZsc),
        jnp.zeros_like(carry_adv.vZcs),
        jnp.zeros_like(carry_adv.vLsc),
        jnp.zeros_like(carry_adv.vLcs),
        zero_like_r,
        zero_like_r,
        zero_like_r,
        zero_like_r,
        zero_like_r,
        zero_like_r,
        restart_scalars.iter_offset,
        restart_scalars.iter1,
        restart_scalars.ijacob,
        restart_scalars.bad_resets,
        restart_scalars.bad_growth,
        restart_scalars.force_bcovar_update,
    )


def _no_restart_updates(carry_adv: Any) -> ScanRestartUpdates:
    return ScanRestartUpdates(
        carry_adv.state,
        carry_adv.time_step,
        carry_adv.inv_tau,
        carry_adv.fsq_prev,
        carry_adv.vRcc,
        carry_adv.vRss,
        carry_adv.vZsc,
        carry_adv.vZcs,
        carry_adv.vLsc,
        carry_adv.vLcs,
        carry_adv.vRsc,
        carry_adv.vRcs,
        carry_adv.vZcc,
        carry_adv.vZss,
        carry_adv.vLcc,
        carry_adv.vLss,
        carry_adv.iter_offset,
        carry_adv.iter1,
        carry_adv.ijacob,
        carry_adv.bad_resets,
        carry_adv.bad_growth,
        jnp.asarray(False),
    )
