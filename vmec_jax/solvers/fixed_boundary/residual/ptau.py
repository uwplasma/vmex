"""Ptau/Jacobian bookkeeping for residual-iteration controllers."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Callable, NamedTuple

import numpy as np


class BadJacobianTauSelection(NamedTuple):
    """Resolved bad-Jacobian decision from ptau and optional state probes."""

    bad_jacobian: bool
    min_tau: float
    max_tau: float
    min_tau_ptau: float | None
    max_tau_ptau: float | None
    min_tau_state: float
    max_tau_state: float
    bad_jacobian_ptau: bool | None
    bad_jacobian_state: bool


def ptau_minmax_from_k_host(
    k: Any,
    *,
    ptau_context: Any,
    compute_jit: Callable[..., Any],
    ptau_minmax_host_func: Callable[..., tuple[Any | None, Any | None]],
) -> tuple[Any | None, Any | None]:
    """Compute VMEC ``ptau`` min/max on the host for controller decisions."""

    return ptau_minmax_host_func(
        k,
        pshalf=ptau_context.pshalf_np,
        ohs=float(ptau_context.ohs_scalar) if ptau_context.ohs_scalar is not None else 0.0,
        compute_jit=compute_jit,
        pshalf_jax=ptau_context.pshalf_jax,
        ohs_jax=ptau_context.ohs_jax,
    )


def ptau_minmax_from_k_jax(
    k: Any,
    *,
    ptau_context: Any,
    pshalf_from_s_jax: Callable[..., Any],
    ptau_minmax_jax_func: Callable[..., Any],
) -> Any:
    """Compute VMEC ``ptau`` min/max on the active JAX path."""

    return ptau_minmax_jax_func(
        k,
        s=ptau_context.s,
        pshalf_from_s_jax=pshalf_from_s_jax,
    )


def ptau_minmax(
    k: Any,
    *,
    ptau_context: Any,
    has_jax_func: Callable[[], bool],
    compute_jit: Callable[..., Any],
    pshalf_from_s_jax: Callable[..., Any],
    ptau_minmax_host_func: Callable[..., tuple[Any | None, Any | None]],
    ptau_minmax_jax_func: Callable[..., Any],
) -> Any:
    """Dispatch ptau min/max computation to the current host/JAX path."""

    if has_jax_func():
        return ptau_minmax_from_k_jax(
            k,
            ptau_context=ptau_context,
            pshalf_from_s_jax=pshalf_from_s_jax,
            ptau_minmax_jax_func=ptau_minmax_jax_func,
        )
    return ptau_minmax_from_k_host(
        k,
        ptau_context=ptau_context,
        compute_jit=compute_jit,
        ptau_minmax_host_func=ptau_minmax_host_func,
    )


def accepted_control_ptau_arrays(
    k: Any,
    *,
    kernel_arrays_from_k: Callable[[Any], tuple[Any, ...] | None],
) -> tuple[Any, ...] | None:
    """Return kernel arrays only when there are enough radial points for ptau."""

    arrays = kernel_arrays_from_k(k)
    if arrays is None:
        return None
    try:
        ns = int(getattr(arrays[0], "shape", (0,))[0])
    except Exception:
        return None
    return arrays if ns >= 2 else None


def accepted_control_ptau_host_from_payload(
    payload: tuple[Any, Any, Any] | None,
    *,
    device_get_floats: Callable[..., tuple[float, ...]],
) -> tuple[float | None, tuple[float, float] | None, bool]:
    """Materialize an accepted-controller payload on the host.

    The hot residual loop treats payload failures as cache misses and falls back
    to direct host transfer. Keeping that behavior here avoids duplicating the
    same defensive unpacking around pre-fused and lazily computed ptau payloads.
    """

    if payload is None:
        return None, None, False
    try:
        fsq1_payload, ptau_min_payload, ptau_max_payload = payload
        fsq1, ptau_min, ptau_max = device_get_floats(
            fsq1_payload,
            ptau_min_payload,
            ptau_max_payload,
        )
    except Exception:
        return None, None, False
    return float(fsq1), (float(ptau_min), float(ptau_max)), True


def state_tau_minmax_from_vmec_state(
    *,
    state: Any,
    modes: Any,
    trig: Any,
    s: Any,
    lconm1: bool,
    lthreed: bool,
    mask_even: Any,
    mask_odd: Any,
    host_update_assembly: bool,
    tree_has_tracer: Callable[[Any], bool],
    jacobian_from_state: Callable[..., Any],
    device_get_floats: Callable[..., tuple[float, ...]],
    jnp_module: Any,
    numpy_patch_context: Callable[[], Any] | None = None,
) -> tuple[float, float]:
    """Return min/max tau from a VMEC state-Jacobian diagnostic path."""

    host_path = bool(host_update_assembly) and (not tree_has_tracer(state)) and (not tree_has_tracer(s))
    if host_path:
        context_factory = numpy_patch_context if numpy_patch_context is not None else nullcontext
        with context_factory():
            jac_state = jacobian_from_state(
                state=state,
                modes=modes,
                trig=trig,
                s=s,
                lconm1=bool(lconm1),
                lthreed=bool(lthreed),
                mask_even=mask_even,
                mask_odd=mask_odd,
            )
        tau = np.asarray(jac_state.tau)
        if int(tau.size) <= 0:
            return float("nan"), float("nan")
        tau_use = tau[1:] if int(tau.shape[0]) > 1 else tau
        return float(np.min(tau_use)), float(np.max(tau_use))

    jac_state = jacobian_from_state(
        state=state,
        modes=modes,
        trig=trig,
        s=s,
        lconm1=bool(lconm1),
        lthreed=bool(lthreed),
        mask_even=mask_even,
        mask_odd=mask_odd,
    )
    tau = jnp_module.asarray(jac_state.tau)
    if int(tau.size) <= 0:
        return float("nan"), float("nan")
    tau_use = tau[1:] if int(tau.shape[0]) > 1 else tau
    tau_min = jnp_module.min(tau_use)
    tau_max = jnp_module.max(tau_use)
    tau_min_f, tau_max_f = device_get_floats(tau_min, tau_max)
    return float(tau_min_f), float(tau_max_f)


def maybe_dump_jacobian_terms(
    *,
    k: Any,
    s: Any,
    iter_idx: int,
    dump_func: Callable[..., None],
) -> None:
    """Best-effort Jacobian-term dump wrapper."""

    dump_func(k=k, s=s, iter_idx=iter_idx)


def maybe_dump_ptau(
    *,
    iter_idx: int,
    ptau_min: float,
    ptau_max: float,
    tau_min_state: float | None,
    tau_max_state: float | None,
    badjac_ptau: bool | None,
    badjac_state: bool | None,
    badjac_used: bool,
    mode: str,
    label: str,
    getenv: Callable[[str, str], str],
    dump_func: Callable[..., None],
    dump_ptau_env_name: str = "VMEC_JAX_DUMP_PTAU",
    dump_dir_env_name: str = "VMEC_JAX_DUMP_DIR",
) -> None:
    """Best-effort ptau dump wrapper with call-time environment values."""

    dump_func(
        iter_idx=iter_idx,
        ptau_min=ptau_min,
        ptau_max=ptau_max,
        tau_min_state=tau_min_state,
        tau_max_state=tau_max_state,
        badjac_ptau=badjac_ptau,
        badjac_state=badjac_state,
        badjac_used=badjac_used,
        mode=mode,
        label=label,
        dump_ptau_env=getenv(dump_ptau_env_name, ""),
        dump_dir=getenv(dump_dir_env_name, ""),
    )


def resolve_bad_jacobian_tau_selection(
    *,
    reference_mode: bool,
    vmec2000_control: bool,
    accepted_control_ptau_host: tuple[float, float] | None,
    k: Any,
    state: Any,
    iter_idx: int,
    startup_policy: Any,
    badjac_use_state: bool,
    ptau_tol: float,
    static: Any,
    trig: Any,
    s: Any,
    host_update_assembly: bool,
    timing_enabled: bool,
    perf_counter: Callable[[], float],
    record_timing: Callable[[str, float | None], Any],
    ptau_minmax_from_k_host_func: Callable[[Any], tuple[Any | None, Any | None]],
    device_get_floats_func: Callable[..., tuple[float, ...]],
    should_probe_bad_jacobian_state_func: Callable[..., bool],
    bad_jacobian_requires_state_jacobian_func: Callable[..., bool],
    bad_jacobian_tau_decision_func: Callable[..., Any],
    select_bad_jacobian_decision_func: Callable[..., Any],
    state_tau_minmax_from_vmec_state_func: Callable[..., tuple[float, float]],
    tree_has_tracer_func: Callable[[Any], bool],
    jacobian_from_state_func: Callable[..., Any],
    jnp_module: Any,
    numpy_patch_context: Callable[[], Any] | None = None,
) -> BadJacobianTauSelection:
    """Resolve the VMEC bad-Jacobian tau decision for one iteration.

    The hot solver loop still owns side effects such as history appends,
    debug-file writes, prints, and axis resets.  This helper keeps only the
    ptau/state tau selection logic in one place so the branch fingerprint is
    explicit and testable.
    """

    if not (bool(reference_mode) or bool(vmec2000_control)):
        return BadJacobianTauSelection(
            bad_jacobian=False,
            min_tau=float("nan"),
            max_tau=float("nan"),
            min_tau_ptau=None,
            max_tau_ptau=None,
            min_tau_state=float("nan"),
            max_tau_state=float("nan"),
            bad_jacobian_ptau=None,
            bad_jacobian_state=False,
        )

    min_tau_ptau = max_tau_ptau = None
    bad_jacobian_ptau = None
    if accepted_control_ptau_host is not None:
        min_tau_ptau, max_tau_ptau = accepted_control_ptau_host
    else:
        ptau_min, ptau_max = ptau_minmax_from_k_host_func(k)
        if ptau_min is not None and ptau_max is not None:
            t_badjac_ptau_get_start = perf_counter() if timing_enabled else None
            min_tau_ptau, max_tau_ptau = device_get_floats_func(ptau_min, ptau_max)
            record_timing("iteration_control_badjac_ptau_get", t_badjac_ptau_get_start)

    ptau_decision = None
    if min_tau_ptau is not None and max_tau_ptau is not None:
        ptau_decision = bad_jacobian_tau_decision_func(
            min_tau=min_tau_ptau,
            max_tau=max_tau_ptau,
            vmec2000_control=bool(vmec2000_control),
            ptau_tol=ptau_tol,
        )
        bad_jacobian_ptau = bool(ptau_decision.bad_jacobian)

    state_probe = should_probe_bad_jacobian_state_func(
        state_probe=bool(startup_policy.badjac_state_probe),
        initial_state_probe_iters=int(startup_policy.badjac_initial_state_probe_iters),
        iter_idx=int(iter_idx),
    )
    need_state_jac = bad_jacobian_requires_state_jacobian_func(
        badjac_use_state=bool(badjac_use_state),
        dump_ptau_state=bool(startup_policy.dump_ptau_state),
        state_probe=bool(state_probe),
        ptau_decision=ptau_decision,
    )
    if need_state_jac:
        t_badjac_state_jacobian_start = perf_counter() if timing_enabled else None
        min_tau_state, max_tau_state = state_tau_minmax_from_vmec_state_func(
            state=state,
            modes=static.modes,
            trig=trig,
            s=s,
            lconm1=bool(getattr(static.cfg, "lconm1", True)),
            lthreed=bool(getattr(static.cfg, "lthreed", True)),
            mask_even=getattr(static, "m_is_even", None),
            mask_odd=getattr(static, "m_is_odd", None),
            host_update_assembly=bool(host_update_assembly),
            tree_has_tracer=tree_has_tracer_func,
            jacobian_from_state=jacobian_from_state_func,
            device_get_floats=device_get_floats_func,
            jnp_module=jnp_module,
            numpy_patch_context=numpy_patch_context,
        )
        record_timing("iteration_control_badjac_state_jacobian", t_badjac_state_jacobian_start)
        state_decision = bad_jacobian_tau_decision_func(
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
        state_decision = bad_jacobian_tau_decision_func(
            min_tau=min_tau_state,
            max_tau=max_tau_state,
            vmec2000_control=bool(vmec2000_control),
            ptau_tol=ptau_tol,
        )

    badjac_selection = select_bad_jacobian_decision_func(
        badjac_use_state=bool(badjac_use_state),
        ptau_decision=ptau_decision,
        state_decision=state_decision,
    )
    return BadJacobianTauSelection(
        bad_jacobian=bool(badjac_selection.bad_jacobian),
        min_tau=float(badjac_selection.min_tau),
        max_tau=float(badjac_selection.max_tau),
        min_tau_ptau=min_tau_ptau,
        max_tau_ptau=max_tau_ptau,
        min_tau_state=float(min_tau_state),
        max_tau_state=float(max_tau_state),
        bad_jacobian_ptau=bad_jacobian_ptau,
        bad_jacobian_state=bool(bad_jacobian_state),
    )


__all__ = [
    "accepted_control_ptau_arrays",
    "accepted_control_ptau_host_from_payload",
    "BadJacobianTauSelection",
    "maybe_dump_jacobian_terms",
    "maybe_dump_ptau",
    "ptau_minmax",
    "ptau_minmax_from_k_host",
    "ptau_minmax_from_k_jax",
    "resolve_bad_jacobian_tau_selection",
    "state_tau_minmax_from_vmec_state",
]
