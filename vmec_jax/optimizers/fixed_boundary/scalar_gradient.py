"""Scalar objective and reverse discrete-adjoint gradient replay."""

from __future__ import annotations

import time
from typing import Any

import numpy as np


def exact_objective_and_gradient(owner: Any, params) -> tuple[float, np.ndarray]:
    """Return ``0.5 * ||residuals||**2`` and its exact scalar-adjoint gradient."""

    if owner._solver_device_name is not None and not owner._inside_solver_device_context:
        return owner._run_in_solver_device_context(owner.objective_and_gradient_fun, params)
    from vmec_jax._compat import jax, jnp as _jnp
    from vmec_jax.discrete_adjoint import checkpoint_tape_state_vjp
    from vmec_jax.state import unpack_state

    t_total = time.perf_counter()
    params = _jnp.asarray(params, dtype=_jnp.float64)
    state, payload = owner._solve_exact_with_tape(params, return_payload=True)
    tape = payload["tape"]
    axis_override = payload["axis_override"]
    packed_final = owner._packed_final_from_exact_payload(state, payload)

    def _residuals_from_packed(packed):
        return owner._residuals_fn(unpack_state(packed, owner._layout))

    t_res_vjp = time.perf_counter()
    objective_cotangent_factory = getattr(
        owner._residuals_fn,
        "_state_objective_value_and_cotangent_from_packed",
        None,
    )
    if objective_cotangent_factory is not None:
        helper_key = (
            "objective_value_and_cotangent",
            int(owner._layout.size),
            id(owner._residuals_fn),
        )
        helper_cache = owner._discrete_jacobian_helper_cache.get(helper_key)
        if helper_cache is None:

            @jax.jit
            def _objective_value_and_cotangent_helper(packed_state_arg):
                return objective_cotangent_factory(packed_state_arg, owner._layout)

            helper_cache = {
                "objective_value_and_cotangent": _objective_value_and_cotangent_helper,
                "jitted": True,
            }
            owner._discrete_jacobian_helper_cache[helper_key] = helper_cache
        try:
            cost, final_cotangent = helper_cache["objective_value_and_cotangent"](packed_final)
        except Exception:
            if not bool(helper_cache.get("jitted", False)):
                raise
            helper_cache = {
                "objective_value_and_cotangent": lambda packed_state_arg: objective_cotangent_factory(
                    packed_state_arg,
                    owner._layout,
                ),
                "jitted": False,
            }
            owner._discrete_jacobian_helper_cache[helper_key] = helper_cache
            cost, final_cotangent = helper_cache["objective_value_and_cotangent"](packed_final)
    else:
        residuals = owner._residuals_fn(state)
        residuals = _jnp.asarray(residuals, dtype=_jnp.float64)
        cost = 0.5 * _jnp.vdot(residuals, residuals)
        state_cotangent_operator_factory = getattr(
            owner._residuals_fn,
            "_state_cotangent_operator_from_packed",
            None,
        )
        if state_cotangent_operator_factory is not None:
            final_cotangent = state_cotangent_operator_factory(packed_final, owner._layout)(residuals)
        else:
            _, residual_vjp = jax.vjp(_residuals_from_packed, packed_final)
            final_cotangent = residual_vjp(residuals)[0]
    final_cotangent = _jnp.nan_to_num(final_cotangent, nan=0.0, posinf=0.0, neginf=0.0)
    owner._profile_add("gradient_residual_vjp", time.perf_counter() - t_res_vjp)

    t_replay = time.perf_counter()
    initial_cotangent = checkpoint_tape_state_vjp(
        tape=tape,
        static=owner._static,
        final_cotangent=final_cotangent,
        rebuild_preconditioner=True,
    )
    initial_cotangent = owner._profile_async_phase(
        "gradient_tape_replay",
        t_replay,
        initial_cotangent,
    )
    initial_cotangent = _jnp.nan_to_num(initial_cotangent, nan=0.0, posinf=0.0, neginf=0.0)

    t_initial = time.perf_counter()
    cache_key = None
    initial_tangents = None
    try:
        cache_key = owner._initial_tangent_cache_key(params)
        initial_tangents = owner._initial_tangent_cache.get(cache_key) if cache_key is not None else None
    except Exception:
        cache_key = None
        initial_tangents = None
    if initial_tangents is not None:
        owner._profile_add("gradient_initial_tangents_cache_hit", 0.0)
        grad = _jnp.tensordot(
            _jnp.asarray(initial_tangents, dtype=_jnp.float64),
            _jnp.asarray(initial_cotangent, dtype=_jnp.float64),
            axes=([1], [0]),
        )
        owner._profile_add("gradient_initial_projection", time.perf_counter() - t_initial)
    elif owner._scalar_gradient_initial_tangents_enabled(int(params.size)):
        owner._profile_add("gradient_initial_tangents_precompute", 0.0)
        initial_tangents = owner._initial_tangent_columns(
            params,
            axis_override,
            profile_prefix="gradient",
        )
        grad = _jnp.tensordot(
            _jnp.asarray(initial_tangents, dtype=_jnp.float64),
            _jnp.asarray(initial_cotangent, dtype=_jnp.float64),
            axes=([1], [0]),
        )
        owner._profile_add("gradient_initial_projection", time.perf_counter() - t_initial)
    else:
        initial_vjp_key = None if cache_key is None else ("gradient_initial_vjp", cache_key)
        initial_vjp = (
            owner._discrete_jacobian_helper_cache.get(initial_vjp_key) if initial_vjp_key is not None else None
        )
        if initial_vjp is not None:
            owner._profile_add("gradient_initial_vjp_cache_hit", 0.0)
        else:
            axis_override = {key: _jnp.asarray(value, dtype=params.dtype) for key, value in axis_override.items()}
            _, initial_vjp = jax.vjp(
                lambda p: owner._solver_initial_state_packed_from_params(p, axis_override),
                params,
            )
            if initial_vjp_key is not None:
                owner._discrete_jacobian_helper_cache[initial_vjp_key] = initial_vjp
        grad = initial_vjp(_jnp.asarray(initial_cotangent, dtype=_jnp.float64))[0]
        owner._profile_add("gradient_initial_vjp", time.perf_counter() - t_initial)
    owner._profile_add("gradient_total", time.perf_counter() - t_total)
    return float(np.asarray(cost, dtype=float)), np.asarray(grad, dtype=float)

