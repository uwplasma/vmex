"""Dense validation-scale adjoint primitives for free-boundary operators.

The dense path is intentionally small and mathematically transparent.  It
checks the adjoint contract used by larger NESTOR/mode-space operators: solve
the primal linear system in the forward pass, then apply the transpose solve in
the reverse pass.  These helpers are validation kernels, not the fast production
free-boundary path.
"""

from __future__ import annotations

from typing import Any

from vmec_jax._compat import jax, jnp, tree_util


def dense_vacuum_solve_jax(A: Any, b: Any, *, symmetric: bool = False) -> Any:
    """Solve a dense vacuum linear system with an implicit transpose adjoint."""

    A_arr = jnp.asarray(A)
    b_arr = jnp.asarray(b)
    if A_arr.ndim != 2 or A_arr.shape[0] != A_arr.shape[1]:
        raise ValueError("A must be a square dense matrix")
    if b_arr.shape[0] != A_arr.shape[0]:
        raise ValueError(f"b leading dimension {b_arr.shape[0]} does not match A size {A_arr.shape[0]}")

    if jax is None:  # pragma: no cover - dependency fallback.
        return jnp.linalg.solve(A_arr, b_arr)

    def matvec(x):
        """Evaluate matvec for direct-coil free-boundary solve and branch-local adjoint validation."""
        return A_arr @ x

    def solve_fn(_matvec, rhs):
        """Solve solve fn for direct-coil free-boundary solve and branch-local adjoint validation."""
        return jnp.linalg.solve(A_arr, rhs)

    def transpose_solve_fn(_matvec, rhs):
        """Evaluate transpose solve fn for direct-coil free-boundary solve and branch-local adjoint validation."""
        matrix = A_arr if bool(symmetric) else A_arr.T
        return jnp.linalg.solve(matrix, rhs)

    return jax.lax.custom_linear_solve(
        matvec,
        b_arr,
        solve_fn,
        transpose_solve=transpose_solve_fn,
        symmetric=bool(symmetric),
    )


def dense_vacuum_residual(A: Any, x: Any, b: Any) -> Any:
    """Return ``A @ x - b`` for tests and diagnostics."""

    return jnp.asarray(A) @ jnp.asarray(x) - jnp.asarray(b)


def dense_nonlinear_solve_jax(
    residual_fn: Any,
    initial: Any,
    params: Any,
    *,
    max_iter: int = 10,
    damping: float = 1.0,
) -> Any:
    """Solve a small nonlinear residual with an implicit-root adjoint."""

    x0 = jnp.asarray(initial)
    if x0.ndim != 1:
        raise ValueError("initial must be a 1D state vector")
    max_iter_i = int(max_iter)
    if max_iter_i < 0:
        raise ValueError("max_iter must be non-negative")
    damping_f = float(damping)

    def _newton_solve(init, prm):
        def _step(_i, x):
            residual = jnp.asarray(residual_fn(x, prm))
            if residual.shape != x.shape:
                raise ValueError("residual_fn must return the same shape as initial")
            jac_x = jax.jacfwd(lambda y: jnp.asarray(residual_fn(y, prm)))(x)
            delta = jnp.linalg.solve(jac_x, residual)
            return x - damping_f * delta

        if jax is None:  # pragma: no cover - JAX-free import fallback.
            x = init
            for _ in range(max_iter_i):
                residual = jnp.asarray(residual_fn(x, prm))
                jac_x = finite_difference_jacobian(lambda y: residual_fn(y, prm), x)
                x = x - damping_f * jnp.linalg.solve(jac_x, residual)
            return x
        return jax.lax.fori_loop(0, max_iter_i, _step, init)

    if jax is None:  # pragma: no cover - dependency fallback.
        return _newton_solve(x0, params)

    @jax.custom_vjp
    def _solve(init, prm):
        return _newton_solve(init, prm)

    def _solve_fwd(init, prm):
        root = _newton_solve(init, prm)
        return root, (root, prm, jnp.zeros_like(init))

    def _solve_bwd(saved, root_bar):
        root, prm, init_zero = saved
        jac_x = jax.jacfwd(lambda y: jnp.asarray(residual_fn(y, prm)))(root)
        lam = jnp.linalg.solve(jac_x.T, jnp.asarray(root_bar))
        _, pullback_params = jax.vjp(lambda pp: jnp.asarray(residual_fn(root, pp)), prm)
        grad_params = pullback_params(lam)[0]
        grad_params = tree_util.tree_map(lambda value: -value, grad_params)
        return init_zero, grad_params

    _solve.defvjp(_solve_fwd, _solve_bwd)
    return _solve(x0, params)


def dense_fixed_point_solve_jax(
    update_fn: Any,
    initial: Any,
    params: Any,
    *,
    max_iter: int = 10,
    damping: float = 1.0,
) -> Any:
    """Solve ``x = update_fn(x, params)`` with the nonlinear implicit adjoint."""

    def residual(state, prm):
        """Evaluate residual for direct-coil free-boundary solve and branch-local adjoint validation."""
        state_arr = jnp.asarray(state)
        update = jnp.asarray(update_fn(state_arr, prm))
        if update.shape != state_arr.shape:
            raise ValueError("update_fn must return the same shape as initial")
        return state_arr - update

    return dense_nonlinear_solve_jax(
        residual,
        initial,
        params,
        max_iter=max_iter,
        damping=damping,
    )


def finite_difference_jacobian(fn: Any, x: Any, eps: float = 1.0e-6) -> Any:
    """Small NumPy/JAX-free fallback Jacobian for import-only environments."""

    x_arr = jnp.asarray(x)
    eye = jnp.eye(int(x_arr.size), dtype=x_arr.dtype)
    cols = []
    for k in range(int(x_arr.size)):
        step = eps * eye[k]
        cols.append((jnp.asarray(fn(x_arr + step)) - jnp.asarray(fn(x_arr - step))) / (2.0 * eps))
    return jnp.stack(cols, axis=1)


__all__ = [
    "dense_fixed_point_solve_jax",
    "dense_nonlinear_solve_jax",
    "dense_vacuum_residual",
    "dense_vacuum_solve_jax",
    "finite_difference_jacobian",
]
