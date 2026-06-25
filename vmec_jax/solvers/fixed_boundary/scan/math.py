"""Pure math helpers for the residual-iteration scan loop."""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from ...._compat import jnp


class ScanTauDecision(NamedTuple):
    """Bad-Jacobian tau extrema computed inside the scan loop."""

    bad_jacobian: Any
    min_tau: Any
    max_tau: Any


class ScanBadJacobianDecision(NamedTuple):
    """Bad-Jacobian decision plus the ptau/state diagnostics emitted to history."""

    bad_jacobian: Any
    min_tau: Any
    max_tau: Any
    min_tau_ptau: Any
    max_tau_ptau: Any
    min_tau_state: Any
    max_tau_state: Any
    badjac_ptau: Any
    badjac_state: Any


class ScanRestartUpdates(NamedTuple):
    """State and velocity fields after applying a scan-mode restart."""

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


class PtauMinmaxContext(NamedTuple):
    """Precomputed constants for VMEC ``ptau`` bad-Jacobian checks."""

    s: Any
    pshalf_np: Any | None
    ohs_scalar: float | None
    pshalf_jax: Any | None
    ohs_jax: Any | None


def build_ptau_minmax_context(
    s: Any,
    *,
    has_jax: bool,
    s_has_tracer: bool,
    pshalf_from_s_np: Any,
    pshalf_from_s_jax: Any,
) -> PtauMinmaxContext:
    """Precompute host/JAX ``ptau`` constants for a fixed radial mesh."""

    if bool(s_has_tracer) and bool(has_jax):
        s_jax = jnp.asarray(s, dtype=jnp.float64)
        hs_jax0 = s_jax[1] - s_jax[0] if int(s_jax.shape[0]) > 1 else jnp.asarray(1.0, dtype=jnp.float64)
        pshalf_np = None
        ohs_scalar = None
    else:
        s_np = np.asarray(s)
        hs = float(s_np[1] - s_np[0]) if int(s_np.shape[0]) > 1 else 1.0
        pshalf_np = pshalf_from_s_np(s)
        ohs_scalar = 0.0 if hs == 0.0 else 1.0 / hs
        s_jax = None
        hs_jax0 = None

    pshalf_jax = None
    ohs_jax = None
    if bool(has_jax):
        if bool(s_has_tracer):
            pshalf_jax = pshalf_from_s_jax(s_jax, jnp.float64)
            ohs_jax = jnp.where(
                hs_jax0 == 0.0,
                jnp.asarray(0.0, dtype=jnp.float64),
                jnp.asarray(1.0, dtype=jnp.float64) / hs_jax0,
            )
        else:
            pshalf_jax = jnp.asarray(pshalf_np, dtype=jnp.float64)
            ohs_jax = jnp.asarray(ohs_scalar, dtype=jnp.float64)

    return PtauMinmaxContext(
        s=s,
        pshalf_np=pshalf_np,
        ohs_scalar=ohs_scalar,
        pshalf_jax=pshalf_jax,
        ohs_jax=ohs_jax,
    )


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


def _ptau_minmax_from_context_host(
    k: Any,
    *,
    context: PtauMinmaxContext,
    host_update_assembly: bool,
    tree_has_tracer: Any,
    compute_jit: Any = None,
) -> tuple[Any | None, Any | None]:
    """Compute host-side ``ptau`` min/max using a precomputed context."""

    use_host_np_ptau = bool(host_update_assembly) and (not tree_has_tracer(k))
    return _ptau_minmax_from_k_host(
        k,
        pshalf=context.pshalf_np,
        ohs=context.ohs_scalar,
        compute_jit=None if use_host_np_ptau else compute_jit,
        pshalf_jax=None if use_host_np_ptau else context.pshalf_jax,
        ohs_jax=None if use_host_np_ptau else context.ohs_jax,
    )


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
        if int(ptau.size) == 0:
            return nan_val, nan_val
        return jnp.min(ptau), jnp.max(ptau)

    return _compute(None)


def _ptau_minmax_from_context_jax(k: Any, *, context: PtauMinmaxContext, pshalf_from_s_jax: Any):
    """Compute JAX-side ``ptau`` min/max using a precomputed context."""

    return _ptau_minmax_from_k_jax(k, s=context.s, pshalf_from_s_jax=pshalf_from_s_jax)


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


def scan_bad_jacobian_decision(
    *,
    vmec2000_control: bool,
    use_apply_payload_fusion: bool,
    badjac_use_state: bool,
    dump_ptau_state: bool,
    badjac_state_probe: bool,
    badjac_initial_state_probe_iters: int,
    iter2: Any,
    ptau_min: Any | None,
    ptau_max: Any | None,
    state_tau_fn: Any,
    nonvmec_tau_fn: Any,
    ptau_tol: float,
    dtype: Any,
    cond: Any,
) -> ScanBadJacobianDecision:
    """Resolve scan bad-Jacobian policy from ptau and optional state Jacobians."""

    nan = jnp.asarray(jnp.nan, dtype=dtype)
    false = jnp.asarray(False)

    if bool(vmec2000_control):
        if (ptau_min is None) or (ptau_max is None):
            min_tau_ptau = nan
            max_tau_ptau = nan
            badjac_ptau = false
        else:
            min_tau_ptau = jnp.asarray(ptau_min)
            max_tau_ptau = jnp.asarray(ptau_max)
            tau_tol_ptau = jnp.asarray(abs(ptau_tol), dtype=dtype)
            badjac_ptau = (min_tau_ptau < -tau_tol_ptau) & (max_tau_ptau > tau_tol_ptau)

        ptau_valid = jnp.isfinite(min_tau_ptau) & jnp.isfinite(max_tau_ptau)
        state_probe = (
            jnp.asarray(bool(badjac_state_probe))
            & (jnp.asarray(int(badjac_initial_state_probe_iters), dtype=jnp.int32) > 0)
            & (iter2 <= jnp.asarray(int(badjac_initial_state_probe_iters), dtype=jnp.int32))
        )
        need_state_jac = (
            jnp.asarray(bool(badjac_use_state))
            | jnp.asarray(bool(dump_ptau_state))
            | state_probe
            | (~ptau_valid)
            | badjac_ptau
        )

        def _state_jacobian_branch(_):
            return state_tau_fn()

        def _ptau_only(_):
            return false, nan, nan

        badjac_state, min_tau_state, max_tau_state = cond(
            need_state_jac, _state_jacobian_branch, _ptau_only, operand=None
        )
        min_tau = jnp.where(bool(badjac_use_state), min_tau_state, min_tau_ptau)
        max_tau = jnp.where(bool(badjac_use_state), max_tau_state, max_tau_ptau)
        bad_jacobian = jnp.where(bool(badjac_use_state), badjac_state, badjac_ptau)
        return ScanBadJacobianDecision(
            bad_jacobian=bad_jacobian,
            min_tau=min_tau,
            max_tau=max_tau,
            min_tau_ptau=min_tau_ptau,
            max_tau_ptau=max_tau_ptau,
            min_tau_state=min_tau_state,
            max_tau_state=max_tau_state,
            badjac_ptau=badjac_ptau,
            badjac_state=badjac_state,
        )

    if bool(use_apply_payload_fusion):
        return ScanBadJacobianDecision(false, nan, nan, nan, nan, nan, nan, false, false)

    tau_decision = _state_jacobian(nonvmec_tau_fn(), vmec2000_control=False, ptau_tol=ptau_tol)
    return ScanBadJacobianDecision(
        bad_jacobian=tau_decision.bad_jacobian,
        min_tau=tau_decision.min_tau,
        max_tau=tau_decision.max_tau,
        min_tau_ptau=tau_decision.min_tau,
        max_tau_ptau=tau_decision.max_tau,
        min_tau_state=tau_decision.min_tau,
        max_tau_state=tau_decision.max_tau,
        badjac_ptau=false,
        badjac_state=tau_decision.bad_jacobian,
    )


def sample_vmec2000_scan_scalars(
    *,
    carry_adv: Any,
    iter2: Any,
    max_iter: int,
    nstep_screen: int,
    scan_collect_scalars: bool,
    force_sample: Any,
    kernels: Any,
    norms_current: Any,
    gamma: float,
    twopi: float,
    lasym: bool,
    cond: Any,
) -> tuple[Any, Any, Any, Any]:
    """Sample scalar quantities used by VMEC2000 scan screen output."""

    sample_vmec = (iter2 <= 1) | (iter2 >= int(max_iter)) | ((iter2 % int(nstep_screen)) == 0) | force_sample
    sample_vmec = sample_vmec & jnp.asarray(bool(scan_collect_scalars), dtype=bool)

    def _compute_scalars(_):
        r00_j = jnp.asarray(kernels.pr1_even)[0, 0, 0]
        if bool(lasym):
            z00_j = jnp.asarray(kernels.pz1_even)[0, 0, 0]
        else:
            z00_j = jnp.asarray(0.0, dtype=r00_j.dtype)
        wb_val = jnp.asarray(norms_current.wb)
        wp_val = jnp.asarray(norms_current.wp)
        w_mhd = (wb_val + wp_val / (float(gamma) - 1.0)) * jnp.asarray(float(twopi * twopi), dtype=wb_val.dtype)
        return r00_j, z00_j, w_mhd

    def _reuse_scalars(_):
        return carry_adv.r00_prev, carry_adv.z00_prev, carry_adv.w_mhd_prev

    return (sample_vmec, *cond(sample_vmec, _compute_scalars, _reuse_scalars, operand=None))


def scan_bad_jacobian_decision_from_step(
    *,
    carry_adv: Any,
    kernels: Any,
    iter2: Any,
    static: Any,
    trig: Any,
    s: Any,
    vmec2000_control: bool,
    use_apply_payload_fusion: bool,
    badjac_use_state: bool,
    dump_ptau_state: bool,
    badjac_state_probe: bool,
    badjac_initial_state_probe_iters: int,
    ptau_min: Any | None,
    ptau_max: Any | None,
    ptau_tol: float,
    dtype: Any,
    use_state_jac: bool,
    ignore_badjac: bool,
    vmec_half_mesh_jacobian_from_state_func: Any,
    cond: Any,
) -> ScanBadJacobianDecision:
    """Resolve the scan bad-Jacobian branch for one controller step."""

    def _state_tau():
        jac_scan = vmec_half_mesh_jacobian_from_state_func(
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
        tau_decision = _state_jacobian(
            _state_tau(),
            vmec2000_control=bool(vmec2000_control),
            ptau_tol=ptau_tol,
            relative_tol=1.0e-2 if bool(vmec2000_control) else None,
        )
        return tau_decision.bad_jacobian, tau_decision.min_tau, tau_decision.max_tau

    def _nonvmec_tau():
        if bool(use_state_jac):
            return _state_tau()
        return jnp.asarray(kernels.bc.jac.tau)

    decision = scan_bad_jacobian_decision(
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
        cond=cond,
    )
    if bool(ignore_badjac):
        decision = decision._replace(bad_jacobian=jnp.asarray(False))
    return decision


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
