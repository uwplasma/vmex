"""Pytree linear-algebra helpers for free-boundary adjoint reports."""

from __future__ import annotations

from typing import Any

from vmec_jax._compat import jax, jnp, tree_util


def pytree_batched_directional_vdot_jax(jacobian_tree: Any, direction: Any, n_outputs: int) -> Any:
    """Contract a vector-output pytree Jacobian with one pytree direction."""

    leaves = tree_util.tree_leaves(
        tree_util.tree_map(
            lambda jac_leaf, direction_leaf: jnp.sum(
                jnp.reshape(jnp.asarray(jac_leaf), (int(n_outputs), -1))
                * jnp.reshape(jnp.asarray(direction_leaf), (1, -1)),
                axis=1,
            ),
            jacobian_tree,
            direction,
        )
    )
    if not leaves:
        return jnp.zeros((int(n_outputs),), dtype=float)
    total = leaves[0]
    for leaf in leaves[1:]:
        total = total + leaf
    return total


def pytree_pullback_basis_jax(pullback: Any, basis: Any) -> Any:
    """Apply a VJP pullback to all basis cotangents with one batched transform.

    ``jax.vjp`` returns a pullback that accepts one output cotangent.  Several
    free-boundary validation paths need one gradient per scalar output.  Calling
    the pullback in a Python loop is correct but introduces avoidable host
    overhead and can inflate dispatch timing.  ``vmap`` batches the cotangents
    while preserving a leading scalar-output axis on every gradient leaf.

    Some unusual pytrees/backend combinations may not be vmappable; in that
    case fall back to the previous loop so this remains a performance
    improvement, not a semantic requirement.
    """

    try:
        return jax.vmap(lambda cotangent: pullback(cotangent)[0])(basis)
    except Exception:  # pragma: no cover - defensive fallback for exotic pytrees.
        basis_gradients = tuple(pullback(basis[index])[0] for index in range(int(basis.shape[0])))
        return tree_util.tree_map(
            lambda *parts: jnp.stack([jnp.asarray(part) for part in parts], axis=0),
            *basis_gradients,
        )


def pytree_unstack_leading_axis_jax(pytree: Any, n_outputs: int) -> tuple[Any, ...]:
    """Return one pytree per leading output axis from a batched pytree."""

    return tuple(
        tree_util.tree_map(lambda leaf, index=index: jnp.asarray(leaf)[index], pytree)
        for index in range(int(n_outputs))
    )
