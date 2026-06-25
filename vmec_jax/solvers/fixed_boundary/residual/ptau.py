"""Ptau/Jacobian bookkeeping for residual-iteration controllers."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Callable

import numpy as np


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


__all__ = [
    "accepted_control_ptau_arrays",
    "accepted_control_ptau_host_from_payload",
    "maybe_dump_jacobian_terms",
    "maybe_dump_ptau",
    "ptau_minmax",
    "ptau_minmax_from_k_host",
    "ptau_minmax_from_k_jax",
    "state_tau_minmax_from_vmec_state",
]
