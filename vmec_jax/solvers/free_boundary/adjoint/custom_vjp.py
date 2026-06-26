"""Small custom-VJP wrappers for branch-local free-boundary replay helpers."""

from __future__ import annotations

from typing import Any

from vmec_jax._compat import jax, jnp, tree_util


def scalar_custom_vjp_value_jax(objective_fn: Any, params: Any) -> Any:
    """Evaluate a scalar objective with an explicit JAX custom-VJP wrapper."""

    if jax is None:  # pragma: no cover - callers normally guard this first.
        raise RuntimeError("JAX is required for custom-VJP replay wrappers.")

    @jax.custom_vjp
    def wrapped(coil_params):
        """Evaluate wrapped for direct-coil free-boundary solve and branch-local adjoint validation."""
        return objective_fn(coil_params)

    def wrapped_fwd(coil_params):
        """Evaluate wrapped fwd for direct-coil free-boundary solve and branch-local adjoint validation."""
        return objective_fn(coil_params), coil_params

    def wrapped_bwd(coil_params, cotangent):
        """Evaluate wrapped bwd for direct-coil free-boundary solve and branch-local adjoint validation."""
        grad_params = jax.grad(objective_fn)(coil_params)
        scaled_grad = tree_util.tree_map(
            lambda value: jnp.asarray(cotangent) * jnp.asarray(value),
            grad_params,
        )
        return (scaled_grad,)

    wrapped.defvjp(wrapped_fwd, wrapped_bwd)
    return wrapped(params)


def vector_custom_vjp_value_jax(objective_fn: Any, params: Any) -> Any:
    """Evaluate a vector objective with a cotangent-aware custom-VJP wrapper."""

    if jax is None:  # pragma: no cover - callers normally guard this first.
        raise RuntimeError("JAX is required for custom-VJP replay wrappers.")

    @jax.custom_vjp
    def wrapped(coil_params):
        """Evaluate wrapped for direct-coil free-boundary solve and branch-local adjoint validation."""
        return objective_fn(coil_params)

    def wrapped_fwd(coil_params):
        """Evaluate wrapped fwd for direct-coil free-boundary solve and branch-local adjoint validation."""
        return objective_fn(coil_params), coil_params

    def wrapped_bwd(coil_params, cotangent):
        """Evaluate wrapped bwd for direct-coil free-boundary solve and branch-local adjoint validation."""
        _, pullback = jax.vjp(objective_fn, coil_params)
        return pullback(jnp.asarray(cotangent))

    wrapped.defvjp(wrapped_fwd, wrapped_bwd)
    return wrapped(params)


__all__ = [
    "scalar_custom_vjp_value_jax",
    "vector_custom_vjp_value_jax",
]
