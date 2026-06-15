"""Reusable AD-vs-FD checks for JAX-visible free-boundary controllers."""

from __future__ import annotations

from typing import Any

from vmec_jax._compat import jax, jnp, tree_util


def controller_directional_check_jax(
    controller_runner: Any,
    objective_from_run: Any,
    params: Any,
    direction: Any,
    *,
    eps: float = 1.0e-4,
    compute_fd: bool = True,
) -> dict[str, Any]:
    """Compare a controller objective's exact directional derivative to FD."""

    def objective(controller_params):
        return objective_from_run(controller_runner(controller_params))

    check = pytree_directional_derivative_check_jax(
        objective,
        params,
        direction,
        eps=eps,
        compute_fd=compute_fd,
    )
    return {**check, "run": controller_runner(params)}


def pytree_directional_derivative_check_jax(
    objective_fn: Any,
    params: Any,
    direction: Any,
    *,
    eps: float = 1.0e-4,
    compute_fd: bool = True,
) -> dict[str, Any]:
    """Compare an exact JAX directional derivative with central differences.

    Set ``compute_fd=False`` when an external finite-difference slope is
    already available and the caller only needs the exact JAX directional
    derivative.  This avoids two additional objective evaluations for expensive
    replay diagnostics while preserving the default AD-vs-FD contract.
    """

    if jax is None:  # pragma: no cover - JAX is required for exact gradients.
        raise RuntimeError("JAX is required for exact directional derivatives.")
    step = float(eps)
    if not step > 0.0:
        raise ValueError("eps must be positive.")

    def shifted(scale):
        return tree_util.tree_map(
            lambda value, delta: jnp.asarray(value) + float(scale) * jnp.asarray(delta),
            params,
            direction,
        )

    value, grad_params = jax.value_and_grad(objective_fn)(params)
    exact_directional = pytree_vdot_jax(grad_params, direction)
    if bool(compute_fd):
        fd_directional = (objective_fn(shifted(step)) - objective_fn(shifted(-step))) / (2.0 * step)
        abs_error = jnp.abs(exact_directional - fd_directional)
        rel_error = abs_error / jnp.maximum(jnp.asarray(1.0, dtype=abs_error.dtype), jnp.abs(fd_directional))
    else:
        fd_directional = jnp.asarray(jnp.nan, dtype=jnp.asarray(exact_directional).dtype)
        abs_error = fd_directional
        rel_error = fd_directional
    return {
        "value": value,
        "grad": grad_params,
        "exact_directional": exact_directional,
        "fd_directional": fd_directional,
        "abs_error": abs_error,
        "rel_error": rel_error,
    }


def pytree_vdot_jax(lhs: Any, rhs: Any) -> Any:
    """Return the sum of leafwise ``vdot`` values for matching pytrees."""

    products = tree_util.tree_leaves(
        tree_util.tree_map(
            lambda left, right: jnp.vdot(jnp.asarray(left), jnp.asarray(right)),
            lhs,
            rhs,
        )
    )
    if not products:
        return jnp.asarray(0.0)
    total = products[0]
    for product in products[1:]:
        total = total + product
    return total


__all__ = [
    "controller_directional_check_jax",
    "pytree_directional_derivative_check_jax",
    "pytree_vdot_jax",
]
