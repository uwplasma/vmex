"""JIT-cache seams for residual-iteration force evaluation."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable

__all__ = [
    "NumpyForceFastPath",
    "compute_forces_jit_cache_key",
    "prepare_numpy_force_fast_path",
    "select_compute_forces_callable",
]


@dataclass(frozen=True)
class NumpyForceFastPath:
    """Prepared NumPy fast-path state for residual force evaluation."""

    static: Any
    trig: Any
    wout_like: Any
    compute_forces_np: Callable[..., Any] | None


def compute_forces_jit_cache_key(
    *,
    static_key: tuple[Any, ...],
    wout_key: tuple[Any, ...],
    signgs: int,
    apply_m1_constraints: bool,
) -> tuple[Any, ...]:
    """Return the structural cache key for the residual force JIT closure."""

    return (
        "compute_forces_v1",
        tuple(static_key),
        tuple(wout_key),
        int(signgs),
        bool(apply_m1_constraints),
    )


def select_compute_forces_callable(
    compute_forces_nodump: Callable[..., Any],
    *,
    differentiating_scan: bool,
    cache: OrderedDict[tuple, Any],
    cache_key: tuple[Any, ...],
    jit_func: Callable[..., Any],
    cache_get: Callable[[OrderedDict[tuple, Any], tuple[Any, ...]], Any],
    cache_put: Callable[..., Any],
    cache_env_name: str = "VMEC_JAX_COMPUTE_FORCES_CACHE_SIZE",
    cache_default: int = 32,
) -> Any:
    """Return a JIT-wrapped force callable without changing cache ownership."""

    if bool(differentiating_scan):
        # Do not store a jitted closure created while tracing the scan solve:
        # it can retain traced closure constants and leak them out of the
        # transformation.  Primal solves still reuse the global cache.
        return jit_func(
            compute_forces_nodump,
            static_argnames=("include_edge", "include_edge_residual"),
        )

    cached = cache_get(cache, cache_key)
    if cached is not None:
        return cached
    cached = jit_func(
        compute_forces_nodump,
        static_argnames=("include_edge", "include_edge_residual"),
    )
    return cache_put(
        cache,
        cache_key,
        cached,
        env_name=cache_env_name,
        default=int(cache_default),
    )


def prepare_numpy_force_fast_path(
    *,
    host_update_assembly: bool,
    has_jax_func: Callable[[], bool],
    compute_forces_impl: Callable[..., Any],
    state0: Any,
    static: Any,
    trig: Any,
    wout_like: Any,
) -> NumpyForceFastPath:
    """Prepare the host NumPy force fast path and converted closure constants.

    The residual solver's force closure reads ``static``, ``trig``, and
    ``wout_like`` from its parent scope.  Returning converted replacements keeps
    that behavior explicit while avoiding repeated JAX device-to-host dispatch
    when ``host_update_assembly=True``.
    """

    compute_forces_np = None
    if bool(host_update_assembly) and has_jax_func():
        try:
            from vmec_jax.vmec_numpy_forces import compute_forces_numpy as _cfn_helper

            def compute_forces_np(
                state,
                *,
                include_edge: bool,
                include_edge_residual: bool | None = None,
                zero_m1: Any,
                freeb_bsqvac_half: Any | None = None,
                constraint_rcon0: Any | None = None,
                constraint_zcon0: Any | None = None,
                constraint_precond_diag: tuple[Any, Any] | None = None,
                constraint_tcon: Any | None = None,
                constraint_precond_active: Any | None = None,
                constraint_tcon_active: Any | None = None,
                iter_idx: int | None = None,
            ):
                return _cfn_helper(
                    compute_forces_impl,
                    state,
                    include_edge=include_edge,
                    include_edge_residual=include_edge_residual,
                    zero_m1=zero_m1,
                    freeb_bsqvac_half=freeb_bsqvac_half,
                    constraint_rcon0=constraint_rcon0,
                    constraint_zcon0=constraint_zcon0,
                    constraint_precond_diag=constraint_precond_diag,
                    constraint_tcon=constraint_tcon,
                    constraint_precond_active=constraint_precond_active,
                    constraint_tcon_active=constraint_tcon_active,
                    iter_idx=iter_idx,
                )
        except Exception:
            compute_forces_np = None

    if compute_forces_np is None:
        return NumpyForceFastPath(
            static=static,
            trig=trig,
            wout_like=wout_like,
            compute_forces_np=None,
        )

    try:
        import dataclasses as _dc
        import numpy as _np_host
        from vmec_jax.vmec_numpy_forces import _to_numpy_recursive as _tonp, _wrap as _np_wrap

        trig = _tonp(trig)
        try:
            if getattr(trig, "phase_stack", None) is not None:
                trig = _dc.replace(
                    trig,
                    phase_stack_m=static.modes.m,
                    phase_stack_n=static.modes.n,
                )
        except Exception:
            pass
        wout_like = _tonp(wout_like)
        replacements: dict[str, Any] = {}
        s_val = getattr(static, "s", None)
        if s_val is not None:
            try:
                replacements["s"] = _np_wrap(_np_host.asarray(s_val))
            except Exception:
                pass
        np_masks = getattr(static, "tomnsps_masks", None)
        np_masks_edge = getattr(static, "tomnsps_masks_edge", None)
        if np_masks is not None:
            replacements["tomnsps_masks"] = _tonp(np_masks)
            if np_masks_edge is not None:
                replacements["tomnsps_masks_edge"] = _tonp(np_masks_edge)
        try:
            state_dtype = _np_host.asarray(state0.Rcos).dtype
            for mask_field in ("m_is_even", "m_is_odd", "m_is_m1", "m_is_odd_rest"):
                mask_value = getattr(static, mask_field, None)
                if mask_value is not None:
                    replacements[mask_field] = _np_wrap(_np_host.asarray(mask_value, dtype=state_dtype))
        except Exception:
            pass
        if replacements:
            static = _dc.replace(static, **replacements)
    except Exception:
        pass

    return NumpyForceFastPath(
        static=static,
        trig=trig,
        wout_like=wout_like,
        compute_forces_np=compute_forces_np,
    )
