"""Interpolation kernels for parametric exterior panels."""

from __future__ import annotations

from typing import Any

import jax
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


def local_interpolation_weights(nodes: Array, targets: Array, *, width: int = 4) -> Array:
    """Local Lagrange weights robust to strongly clustered cap rings."""

    nodes = jnp.asarray(nodes)
    targets = jnp.asarray(targets)
    width = min(int(width), int(nodes.size))
    if width < 2:
        raise ValueError("at least two interpolation nodes are required")
    insertion = jnp.searchsorted(nodes, targets)
    start = jnp.clip(insertion - width // 2, 0, nodes.size - width)
    indices = start[..., None] + jnp.arange(width)
    local_nodes = nodes[indices]
    differences = local_nodes[..., :, None] - local_nodes[..., None, :]
    differences = differences + jnp.eye(width, dtype=nodes.dtype)
    denominators = jnp.prod(differences, axis=-1)
    numerator_factors = targets[..., None, None] - local_nodes[..., None, :]
    numerator_factors = jnp.where(
        jnp.eye(width, dtype=bool), 1.0, numerator_factors
    )
    numerators = jnp.prod(numerator_factors, axis=-1)
    local_weights = numerators / denominators
    return jnp.sum(
        jax.nn.one_hot(indices, nodes.size) * local_weights[..., None], axis=-2
    )


def _normalized_barycentric_weights(
    nodes: Array, barycentric: Array, targets: Array
) -> Array:
    """Normalize barycentric weights and handle targets at source nodes."""

    targets = jnp.asarray(targets)
    difference = targets[..., None] - nodes
    exact = jnp.abs(difference) <= 8.0 * jnp.finfo(nodes.dtype).eps
    scaled = barycentric / jnp.where(exact, 1.0, difference)
    weights = scaled / jnp.sum(scaled, axis=-1, keepdims=True)
    return jnp.where(jnp.any(exact, axis=-1, keepdims=True), exact, weights)


def spectral_cap_density_samples(
    source: Array,
    cap_values: Array,
    cap_xyz: Array,
) -> Array:
    """Interpolate cap data in periodic angle and the graded radial coordinate."""

    source = jnp.asarray(source)
    cap_values = jnp.asarray(cap_values)
    cap_xyz = jnp.asarray(cap_xyz)
    scalar_input = cap_values.ndim == 2
    ns, ntheta = cap_values.shape[-2:]
    cap_values = cap_values.reshape((-1, ns, ntheta))
    theta = jnp.mod(jnp.arctan2(source[..., 1], source[..., 0]), 2.0 * jnp.pi)
    angular_weights = periodic_interpolation_weights(theta, ntheta)
    boundary_radius = jnp.linalg.norm(cap_xyz[-1, ..., :2], axis=-1)
    sampled_boundary = jnp.einsum("...j,j->...", angular_weights, boundary_radius)
    normalized_radius = jnp.linalg.norm(source[..., :2], axis=-1) / sampled_boundary
    # Theta zero lies on +x. This ratio avoids differentiating the Euclidean
    # norm at the cap center, whose coordinate is identically zero.
    reference_radius = jnp.maximum(cap_xyz[-1, 0, 0], jnp.finfo(source.dtype).tiny)
    radial_nodes = cap_xyz[:, 0, 0] / reference_radius
    width = 2 if ns < 7 else 3 if ns < 11 else 4
    radial_weights = local_interpolation_weights(
        radial_nodes, normalized_radius, width=width
    )
    samples = jnp.einsum(
        "...j,arj,...r->a...", angular_weights, cap_values, radial_weights
    )
    return samples[0] if scalar_input else samples


def cap_nodal_values(
    values: Array,
    *,
    ntheta: int,
    nxi: int,
    ns: int,
    upper: bool,
) -> Array:
    """Restore one cap's center, interior rings, and shared lateral rim."""

    values = jnp.asarray(values)
    scalar_input = values.ndim == 1
    values = values.reshape((-1, values.shape[-1]))
    lateral_size = ntheta * nxi
    cap_interior_size = 1 + (ns - 2) * ntheta
    start = lateral_size + (cap_interior_size if upper else 0)
    center = jnp.broadcast_to(values[:, start, None, None], (values.shape[0], 1, ntheta))
    interior = values[:, start + 1 : start + cap_interior_size].reshape(
        values.shape[0], ns - 2, ntheta
    )
    endpoint = -1 if upper else 0
    rim = values[:, :lateral_size].reshape(values.shape[0], ntheta, nxi)[
        :, :, endpoint
    ][:, None, :]
    result = jnp.concatenate([center, interior, rim], axis=1)
    return result[0] if scalar_input else result
