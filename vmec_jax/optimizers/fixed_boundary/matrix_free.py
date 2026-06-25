"""Matrix-free residual-Jacobian products for exact fixed-boundary optimizers."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .linear_guards import finite_linear_operator_output
from .linear_guards import linear_operator_matrix_arg
from .linear_guards import linear_operator_vector_arg


def build_residual_linear_operator(owner: Any, params):
    """Return a matrix-free exact residual Jacobian at ``params``."""

    if owner._solver_device_name is not None and not owner._inside_solver_device_context:
        return owner._run_in_solver_device_context(owner.residual_linear_operator, params)
    try:
        from scipy.sparse.linalg import LinearOperator
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("residual_linear_operator requires scipy") from exc

    from vmec_jax._compat import jax, jnp as _jnp
    from vmec_jax.discrete_adjoint import (
        checkpoint_tape_state_jvp,
        checkpoint_tape_state_jvp_columns,
        checkpoint_tape_state_vjp,
    )
    from vmec_jax.state import unpack_state

    t_total = time.perf_counter()
    params = _jnp.asarray(params, dtype=_jnp.float64)
    state, payload = owner._solve_exact_with_tape(params, return_payload=True)
    tape = payload["tape"]
    axis_override = {key: _jnp.asarray(value, dtype=params.dtype) for key, value in payload["axis_override"].items()}
    packed_final = owner._packed_final_from_exact_payload(state, payload)

    def _residuals_from_packed(packed):
        return owner._residuals_fn(unpack_state(packed, owner._layout))

    t_setup = time.perf_counter()
    initial_tangent_cache_key = None
    initial_tangent_columns = None
    try:
        initial_tangent_cache_key = owner._initial_tangent_cache_key(params)
        initial_tangent_columns = (
            owner._initial_tangent_cache.get(initial_tangent_cache_key)
            if initial_tangent_cache_key is not None
            else None
        )
    except Exception:
        initial_tangent_cache_key = None
        initial_tangent_columns = None
    if initial_tangent_columns is not None:
        initial_tangent_columns = _jnp.asarray(initial_tangent_columns, dtype=_jnp.float64)
        initial_linear = None
        initial_transpose = None
        owner._profile_add("linear_operator_initial_tangents_cache_hit", 0.0)
    elif owner._precompute_linear_operator_initial_tangents_enabled(int(params.size)):
        owner._profile_add("linear_operator_initial_tangents_precompute", 0.0)
        initial_tangent_columns = _jnp.asarray(
            owner._initial_tangent_columns(
                params,
                axis_override,
                profile_prefix="linear_operator",
            ),
            dtype=_jnp.float64,
        )
        initial_linear = None
        initial_transpose = None
    else:
        _, initial_linear = jax.linearize(
            lambda p: owner._solver_initial_state_packed_from_params(p, axis_override),
            params,
        )
        initial_transpose = jax.linear_transpose(initial_linear, params)
        owner._profile_add("linear_operator_initial_tangents_cache_miss", 0.0)
    residuals, residual_linear = jax.linearize(_residuals_from_packed, packed_final)
    state_cotangent_from_packed = getattr(owner._residuals_fn, "_state_cotangent_from_packed", None)
    residual_cotangent_helper = None
    if state_cotangent_from_packed is not None:
        residual_cotangent_key = (
            "linear_operator_residual_cotangent",
            int(owner._layout.size),
            int(residuals.size),
            id(owner._residuals_fn),
        )
        helper_cache = owner._discrete_jacobian_helper_cache.get(residual_cotangent_key)
        if helper_cache is None:

            @jax.jit
            def _residual_cotangent_helper(packed_state_arg, cotangent_arg):
                return state_cotangent_from_packed(packed_state_arg, owner._layout, cotangent_arg)

            helper_cache = {"residual_cotangent": _residual_cotangent_helper}
            owner._discrete_jacobian_helper_cache[residual_cotangent_key] = helper_cache
        residual_cotangent_helper = helper_cache["residual_cotangent"]
    residual_vjp = None
    if state_cotangent_from_packed is None:
        _, residual_vjp = jax.vjp(_residuals_from_packed, packed_final)
    residuals_np = np.asarray(residuals, dtype=float)
    owner._remember_exact_residual(owner._exact_cache_key(params), residuals_np)
    owner._profile_add("linear_operator_setup", time.perf_counter() - t_setup)

    n_res = int(residuals_np.size)
    n_params = int(params.size)

    def _matvec(direction):
        t_mv = time.perf_counter()
        direction_j = _jnp.asarray(
            linear_operator_vector_arg(direction, size=n_params, name="matvec direction"),
            dtype=params.dtype,
        )
        if initial_tangent_columns is not None:
            initial_tangent = _jnp.tensordot(direction_j, initial_tangent_columns, axes=([0], [0]))
        else:
            initial_tangent = initial_linear(direction_j)
        final_tangent = checkpoint_tape_state_jvp(
            tape=tape,
            static=owner._static,
            initial_tangent=initial_tangent,
            rebuild_preconditioner=True,
        )
        out = residual_linear(final_tangent)
        owner._profile_add("linear_operator_matvec", time.perf_counter() - t_mv)
        return finite_linear_operator_output(
            out,
            profile_add=owner._profile_add,
            profile_name="linear_operator_nonfinite_matvec",
        )

    def _matmat(directions):
        t_mm = time.perf_counter()
        directions_arr = linear_operator_matrix_arg(
            directions,
            rows=n_params,
            name="matmat directions",
        )
        directions_j = _jnp.asarray(directions_arr.T, dtype=params.dtype)
        if initial_tangent_columns is not None:
            initial_tangents = _jnp.tensordot(directions_j, initial_tangent_columns, axes=([1], [0]))
        else:
            initial_tangents = jax.vmap(initial_linear)(directions_j)
        final_tangents = checkpoint_tape_state_jvp_columns(
            tape=tape,
            static=owner._static,
            initial_tangents=initial_tangents,
            rebuild_preconditioner=True,
            column_chunk=owner._lasym_replay_column_chunk(int(directions_j.shape[0])),
        )
        out_columns = jax.vmap(residual_linear)(final_tangents)
        owner._profile_add("linear_operator_matmat", time.perf_counter() - t_mm)
        return finite_linear_operator_output(
            np.asarray(out_columns, dtype=float).T,
            profile_add=owner._profile_add,
            profile_name="linear_operator_nonfinite_matmat",
        )

    def _rmatvec(cotangent):
        t_rmv = time.perf_counter()
        cotangent_j = _jnp.asarray(
            linear_operator_vector_arg(cotangent, size=n_res, name="rmatvec cotangent"),
            dtype=_jnp.float64,
        )
        t_res_cot = time.perf_counter()
        if residual_cotangent_helper is not None:
            final_cotangent = residual_cotangent_helper(packed_final, cotangent_j)
        else:
            final_cotangent = residual_vjp(cotangent_j)[0]
        owner._profile_add("linear_operator_residual_vjp", time.perf_counter() - t_res_cot)
        final_cotangent = _jnp.nan_to_num(final_cotangent, nan=0.0, posinf=0.0, neginf=0.0)
        t_tape_vjp = time.perf_counter()
        initial_cotangent = checkpoint_tape_state_vjp(
            tape=tape,
            static=owner._static,
            final_cotangent=final_cotangent,
            rebuild_preconditioner=True,
        )
        initial_cotangent = owner._profile_async_phase(
            "linear_operator_tape_vjp",
            t_tape_vjp,
            initial_cotangent,
        )
        initial_cotangent = _jnp.nan_to_num(initial_cotangent, nan=0.0, posinf=0.0, neginf=0.0)
        t_initial_transpose = time.perf_counter()
        if initial_tangent_columns is not None:
            grad = _jnp.tensordot(initial_tangent_columns, initial_cotangent, axes=([1], [0]))
            owner._profile_add(
                "linear_operator_initial_tangent_projection",
                time.perf_counter() - t_initial_transpose,
            )
        else:
            grad = initial_transpose(_jnp.asarray(initial_cotangent, dtype=_jnp.float64))[0]
            owner._profile_add("linear_operator_initial_transpose", time.perf_counter() - t_initial_transpose)
        owner._profile_add("linear_operator_rmatvec", time.perf_counter() - t_rmv)
        return finite_linear_operator_output(
            grad,
            profile_add=owner._profile_add,
            profile_name="linear_operator_nonfinite_rmatvec",
        )

    owner._profile_add("linear_operator_total", time.perf_counter() - t_total)
    return LinearOperator(
        shape=(n_res, n_params),
        matvec=_matvec,
        rmatvec=_rmatvec,
        matmat=_matmat,
        dtype=np.dtype(float),
    )

