"""Exact scan/replay helpers for fixed-boundary optimization."""

from __future__ import annotations

from inspect import Parameter
from inspect import signature
import os
import time


def scan_exact_helpers(optimizer, *, initial_guess_from_boundary_func):
    """Return JIT-compiled scan residual/Jacobian helpers for accelerator solves."""

    from ..._compat import jax
    from ..._compat import jnp as _jnp
    from ... import solve as solve_module

    cache_key = (
        int(len(optimizer._specs)),
        int(optimizer._layout.size),
        id(optimizer._residuals_fn),
        int(optimizer._inner_max_iter),
        float(optimizer._inner_ftol),
        optimizer._solver_device_name or "default",
    )
    helper_cache = optimizer._scan_exact_helper_cache.get(cache_key)
    if helper_cache is not None:
        return helper_cache

    scan_solver_kwargs = dict(optimizer._exact_solver_kwargs)
    scan_solver_kwargs.update(
        use_scan=True,
        state_only=True,
        light_history=True,
        resume_state_mode="none",
    )

    def _scan_state_from_params(p):
        boundary_now = optimizer._boundary_from_params(p)
        axis_override = getattr(optimizer, "_initial_axis_override", None)
        if axis_override is None:
            state0 = initial_guess_from_boundary_func(
                optimizer._static,
                boundary_now,
                optimizer._indata,
                vmec_project=True,
            )
        else:
            state0 = initial_guess_from_boundary_func(
                optimizer._static,
                boundary_now,
                optimizer._indata,
                vmec_project=True,
                axis_override=axis_override,
            )
        result = solve_module.solve_fixed_boundary_residual_iter(
            state0,
            optimizer._static,
            max_iter=optimizer._inner_max_iter,
            ftol=optimizer._inner_ftol,
            **scan_solver_kwargs,
        )
        return result.state

    def _scan_residuals_from_params(p):
        return _jnp.asarray(
            optimizer._residuals_fn(_scan_state_from_params(p)),
            dtype=_jnp.float64,
        )

    @jax.jit
    def _residual_impl(p):
        return _scan_residuals_from_params(p)

    @jax.jit
    def _residual_and_jacobian_impl(p):
        residuals, linear = jax.linearize(_scan_residuals_from_params, p)
        directions = _jnp.eye(int(p.size), dtype=p.dtype)
        columns = jax.vmap(linear)(directions)
        return residuals, columns.T

    helper_cache = {
        "state": _scan_state_from_params,
        "residual": _residual_impl,
        "residual_and_jacobian": _residual_and_jacobian_impl,
    }
    optimizer._scan_exact_helper_cache[cache_key] = helper_cache
    return helper_cache


def solve_scan_exact_state(optimizer, params):
    """Run the scan accepted-point solve and remember the final state."""

    from ..._compat import jnp as _jnp

    cache_key = optimizer._exact_cache_key(params)
    if cache_key in getattr(optimizer, "_exact_state_cache", {}):
        optimizer._profile_add("scan_exact_state_cache_hit", 0.0)
        return optimizer._exact_state_cache[cache_key]
    helpers = optimizer._scan_exact_helpers()
    t0 = time.perf_counter()
    state = helpers["state"](_jnp.asarray(params, dtype=_jnp.float64))
    optimizer._remember_exact_state(cache_key, state)
    optimizer._profile_add("scan_exact_state_solve", time.perf_counter() - t0)
    return state


def solve_exact_with_tape_for_jvp(optimizer, params):
    """Build an exact tape optimized for forward tangent-column replay."""

    solve = optimizer._solve_exact_with_tape
    if not optimizer._jvp_only_exact_tape_enabled():
        return solve(params, return_payload=True)
    try:
        parameters = signature(solve).parameters
        accepts_jvp_only = "jvp_only" in parameters or any(
            parameter.kind == Parameter.VAR_KEYWORD for parameter in parameters.values()
        )
    except (TypeError, ValueError):
        accepts_jvp_only = True
    if accepts_jvp_only:
        env_name = "VMEC_JAX_JVP_ONLY_EXACT_TAPE_BASEPOINT_CARRIES"
        previous = os.environ.get(env_name)
        use_basepoint_carries = optimizer._jvp_only_basepoint_carries_enabled()
        if previous is None and use_basepoint_carries:
            os.environ[env_name] = "1"
            optimizer._profile_add("exact_tape_jvp_only_basepoint_carries_auto", 0.0)
        try:
            return solve(params, return_payload=True, jvp_only=True)
        finally:
            if previous is None and use_basepoint_carries:
                os.environ.pop(env_name, None)
    return solve(params, return_payload=True)


def jvp_only_exact_tape_enabled(optimizer) -> bool:
    """Return whether the accepted tape should skip scalar-adjoint-only payloads."""

    forced = optimizer._env_bool_override("VMEC_JAX_OPT_JVP_ONLY_EXACT_TAPE")
    if forced is not None:
        return bool(forced)
    enabled = optimizer._gpu_like_exact_tape_backend()
    if enabled:
        optimizer._profile_add("exact_tape_jvp_only_auto_gpu", 0.0)
    return bool(enabled)


def jvp_only_basepoint_carries_enabled(optimizer) -> bool:
    """Return whether JVP-only tapes should use dynamic-basepoint carries."""

    forced = optimizer._env_bool_override("VMEC_JAX_JVP_ONLY_EXACT_TAPE_BASEPOINT_CARRIES")
    if forced is not None:
        return bool(forced)
    return optimizer._gpu_like_exact_tape_backend()
