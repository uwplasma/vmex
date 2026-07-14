"""Interpolation kernels for parametric exterior panels."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

Array = Any


def cgl_interpolation_weights(targets: Array, nxi: int) -> Array:
    """Barycentric weights from increasing CGL nodes to ``targets``."""

    targets = jnp.asarray(targets)
    degree = nxi - 1
    nodes = jnp.cos(jnp.pi * jnp.arange(nxi, dtype=targets.dtype) / degree)[::-1]
    barycentric = (-1.0) ** jnp.arange(nxi, dtype=targets.dtype)
    barycentric = barycentric.at[jnp.asarray([0, nxi - 1])].multiply(0.5)
    return _normalized_barycentric_weights(nodes, barycentric, targets)


def periodic_interpolation_weights(targets: Array, ntheta: int) -> Array:
    """Real trigonometric interpolation weights on a uniform periodic grid."""

    targets = jnp.asarray(targets)
    nodes = 2.0 * jnp.pi * jnp.arange(ntheta, dtype=targets.dtype) / ntheta
    difference = targets[..., None] - nodes
    weights = jnp.ones_like(difference)
    for mode in range(1, (ntheta + 1) // 2):
        weights = weights + 2.0 * jnp.cos(mode * difference)
    if ntheta % 2 == 0:
        weights = weights + jnp.cos((ntheta // 2) * difference)
    return weights / ntheta


def _normalized_barycentric_weights(nodes: Array, barycentric: Array, targets: Array) -> Array:
    """Normalize barycentric weights and handle targets at source nodes."""

    targets = jnp.asarray(targets)
    difference = targets[..., None] - nodes
    exact = jnp.abs(difference) <= 8.0 * jnp.finfo(nodes.dtype).eps
    scaled = barycentric / jnp.where(exact, 1.0, difference)
    weights = scaled / jnp.sum(scaled, axis=-1, keepdims=True)
    return jnp.where(jnp.any(exact, axis=-1, keepdims=True), exact, weights)
