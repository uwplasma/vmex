"""JIT-cache seams for residual-iteration force evaluation."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable

__all__ = [
    "NumpyForceFastPath",
    "compute_forces_jit_cache_key",
    "maybe_precompile_residual_force_kernels",
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
    use_numpy_force_fast_path: bool | None = None,
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
    use_numpy_force = bool(host_update_assembly) if use_numpy_force_fast_path is None else bool(use_numpy_force_fast_path)
    if bool(use_numpy_force) and has_jax_func():
        try:
            from vmec_jax.kernels.numpy_forces import compute_forces_numpy as _cfn_helper

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
        from vmec_jax.kernels.numpy_forces import _to_numpy_recursive as _tonp, _wrap as _np_wrap

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


def maybe_precompile_residual_force_kernels(
    *,
    jit_forces: bool,
    jit_precompile: bool,
    has_jax_func: Callable[[], bool],
    jax_module: Any,
    jnp_module: Any,
    compute_forces_np: Callable[..., Any] | None,
    compute_forces: Callable[..., Any],
    state0: Any,
    dtype_state: Any,
    zero_precond_diag: Any,
    zero_tcon: Any,
    constraint_active_false: Any,
    backtracking: bool,
    reference_mode: bool,
    use_direct_fallback: bool,
    strict_update: bool,
    jit_strict_update_enabled: bool,
    host_update_assembly: bool,
    limit_dt_from_force: bool,
    limit_update_rms: bool,
    tree_has_tracer_func: Callable[[Any], bool],
    track_history: bool,
    verbose: bool,
    adjoint_trace: bool,
    adjoint_trace_mode: str,
    strict_update_step_jit_func: Callable[..., Any],
    static: Any,
    divide_by_scalxc_for_update: bool,
    free_boundary_enabled: bool,
    step_size: float,
    initial_flip_sign: float,
) -> None:
    """Optionally precompile residual force and strict-update kernels.

    All compile failures are intentionally swallowed to preserve the previous
    best-effort precompile behavior.
    """

    if not (bool(jit_forces) and bool(jit_precompile) and has_jax_func() and (jax_module is not None)):
        return
    if compute_forces_np is not None:
        return
    try:
        zero_m1_pre = jnp_module.asarray(1.0, dtype=dtype_state)
        for include_edge_flag in (False, True):
            compute_forces.lower(
                state0,
                include_edge=include_edge_flag,
                zero_m1=zero_m1_pre,
                constraint_precond_diag=zero_precond_diag,
                constraint_tcon=zero_tcon,
                constraint_precond_active=constraint_active_false,
                constraint_tcon_active=constraint_active_false,
                iter_idx=None,
            ).compile()
    except Exception:
        pass

    need_trial_eval_precompile = bool(backtracking) or bool(reference_mode) or bool(use_direct_fallback)
    use_strict_update_precompile = (
        bool(strict_update)
        and bool(jit_strict_update_enabled)
        and (not bool(host_update_assembly))
        and (not bool(limit_dt_from_force))
        and (not bool(limit_update_rms))
        and (not bool(need_trial_eval_precompile))
        and (not tree_has_tracer_func(state0))
    )
    if not use_strict_update_precompile:
        return
    try:
        velocity_shape_pre = (
            int(jnp_module.asarray(state0.Rcos).shape[0]),
            int(static.cfg.mpol),
            int(static.cfg.ntor) + 1,
        )
        zero_update_pre = jnp_module.zeros(velocity_shape_pre, dtype=dtype_state)
        need_update_rms_precompile = (
            bool(limit_update_rms)
            or bool(track_history)
            or bool(verbose)
            or bool(backtracking)
            or (bool(adjoint_trace) and adjoint_trace_mode == "full")
        )
        step_fn_pre = strict_update_step_jit_func(
            static,
            limit_update_rms=False,
            need_update_rms=need_update_rms_precompile,
            divide_by_scalxc_for_update=bool(divide_by_scalxc_for_update),
            enforce_edge=not bool(free_boundary_enabled),
        )
        scalar_pre = jnp_module.asarray(1.0, dtype=dtype_state)
        step_fn_pre.lower(
            state0,
            jnp_module.asarray(float(step_size), dtype=dtype_state),
            scalar_pre,
            scalar_pre,
            jnp_module.asarray(float(step_size), dtype=dtype_state),
            jnp_module.asarray(float(initial_flip_sign), dtype=dtype_state),
            zero_update_pre,
            zero_update_pre,
            zero_update_pre,
            zero_update_pre,
            zero_update_pre,
            zero_update_pre,
            zero_update_pre,
            zero_update_pre,
            zero_update_pre,
            zero_update_pre,
            zero_update_pre,
            zero_update_pre,
            zero_update_pre,
            zero_update_pre,
            zero_update_pre,
            zero_update_pre,
            zero_update_pre,
            zero_update_pre,
            zero_update_pre,
            zero_update_pre,
            zero_update_pre,
            zero_update_pre,
            zero_update_pre,
            zero_update_pre,
            jnp_module.asarray(1.0e-3 if bool(reference_mode) else 5.0e-3, dtype=dtype_state),
        ).compile()
    except Exception:
        pass
