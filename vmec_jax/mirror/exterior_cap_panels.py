"""Exact polar geometry for planar mirror end-cap panels."""

from __future__ import annotations

from functools import lru_cache, partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .exterior_interpolation import (
    cap_nodal_values,
    periodic_interpolation_weights,
    spectral_cap_density_samples,
)

Array = Any


@lru_cache(maxsize=None)
def _unit_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return 0.5 * (nodes + 1.0), 0.5 * weights


def _triangle_source_points(vertices: Array, *, order: int) -> Array:
    """Map Duffy quadrature nodes to linear triangle source points."""

    nodes, _ = _unit_rule(order)
    u = jnp.asarray(nodes, dtype=vertices.dtype)[None, :, None]
    v = jnp.asarray(nodes, dtype=vertices.dtype)[None, None, :]
    edge1 = vertices[:, 1] - vertices[:, 0]
    edge2 = vertices[:, 2] - vertices[:, 0]
    ray = (1.0 - v)[..., None] * edge1[:, None, None] + (
        v[..., None] * edge2[:, None, None]
    )
    return vertices[:, 0, None, None] + u[..., None] * ray


def spectral_cap_samples(
    vertices: Array,
    dirichlet: Array,
    neumann: Array,
    lower_cap_xyz: Array,
    upper_cap_xyz: Array,
    *,
    side_count: int,
    ntheta: int,
    nxi: int,
    order: int,
    lower_source: Array | None = None,
    upper_source: Array | None = None,
) -> Array:
    """Evaluate both boundary densities on lower and upper cap triangles."""

    ns = int(lower_cap_xyz.shape[0])
    cap_count = (vertices.shape[0] - side_count) // 2
    values = jnp.stack([dirichlet, neumann])
    lower_values = cap_nodal_values(
        values, ntheta=ntheta, nxi=nxi, ns=ns, upper=False
    )
    upper_values = cap_nodal_values(
        values, ntheta=ntheta, nxi=nxi, ns=ns, upper=True
    )
    if lower_source is None:
        lower_source = _triangle_source_points(
            vertices[side_count : side_count + cap_count], order=order
        )
    if upper_source is None:
        upper_source = _triangle_source_points(
            vertices[side_count + cap_count :], order=order
        )
    lower = spectral_cap_density_samples(lower_source, lower_values, lower_cap_xyz)
    upper = spectral_cap_density_samples(upper_source, upper_values, upper_cap_xyz)
    return jnp.concatenate([lower, upper], axis=1)


def _duffy_coordinates(
    triangle_indices: Array,
    cap_xyz: Array,
    *,
    nxi: int,
    upper: bool,
    order: int,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Map each oriented cap triangle to polar coordinates and derivatives."""

    indices = jnp.asarray(triangle_indices)
    cap_xyz = jnp.asarray(cap_xyz)
    ns, ntheta = cap_xyz.shape[:2]
    lateral_size = ntheta * nxi
    cap_size = 1 + (ns - 2) * ntheta
    center = lateral_size + (cap_size if upper else 0)
    interior_start = center + 1
    total_size = lateral_size + 2 * cap_size
    endpoint = nxi - 1 if upper else 0
    rim_indices = jnp.arange(ntheta) * nxi + endpoint
    interior_indices = interior_start + jnp.arange((ns - 2) * ntheta)
    radial_nodes = cap_xyz[:, 0, 0] / cap_xyz[-1, 0, 0]
    rho_lookup = jnp.zeros(total_size, dtype=cap_xyz.dtype)
    rho_lookup = rho_lookup.at[rim_indices].set(1.0)
    rho_lookup = rho_lookup.at[interior_indices].set(
        jnp.repeat(radial_nodes[1:-1], ntheta)
    )
    angle_nodes = 2.0 * jnp.pi * jnp.arange(ntheta, dtype=cap_xyz.dtype) / ntheta
    theta_lookup = jnp.zeros(total_size, dtype=cap_xyz.dtype)
    theta_lookup = theta_lookup.at[rim_indices].set(angle_nodes)
    theta_lookup = theta_lookup.at[interior_indices].set(
        jnp.tile(angle_nodes, ns - 2)
    )

    nodes, _ = _unit_rule(order)
    u = jnp.asarray(nodes, dtype=cap_xyz.dtype)[None, :, None]
    v = jnp.asarray(nodes, dtype=cap_xyz.dtype)[None, None, :]
    shape = (indices.shape[0], order, order)
    barycentric = jnp.stack(
        [
            jnp.broadcast_to(1.0 - u, shape),
            jnp.broadcast_to(u * (1.0 - v), shape),
            jnp.broadcast_to(u * v, shape),
        ],
        axis=-1,
    )
    derivative_u = jnp.stack(
        [
            -jnp.ones(shape, dtype=cap_xyz.dtype),
            jnp.broadcast_to(1.0 - v, shape),
            jnp.broadcast_to(v, shape),
        ],
        axis=-1,
    )
    derivative_v = jnp.stack(
        [
            jnp.zeros(shape, dtype=cap_xyz.dtype),
            -jnp.broadcast_to(u, shape),
            jnp.broadcast_to(u, shape),
        ],
        axis=-1,
    )
    triangle_rho = rho_lookup[indices]
    rho = jnp.sum(barycentric * triangle_rho[:, None, None, :], axis=-1)
    rho_u = jnp.sum(derivative_u * triangle_rho[:, None, None, :], axis=-1)
    rho_v = jnp.sum(derivative_v * triangle_rho[:, None, None, :], axis=-1)

    center_mask = indices == center
    anchor_position = jnp.argmax(~center_mask, axis=1)
    raw_angles = theta_lookup[indices]
    anchor = jnp.take_along_axis(raw_angles, anchor_position[:, None], axis=1)
    angles = anchor + (raw_angles - anchor + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
    angles = jnp.where(center_mask, 0.0, angles)
    affine_theta = jnp.sum(barycentric * angles[:, None, None, :], axis=-1)
    affine_theta_u = jnp.sum(derivative_u * angles[:, None, None, :], axis=-1)
    affine_theta_v = jnp.sum(derivative_v * angles[:, None, None, :], axis=-1)
    center_weight = jnp.sum(
        barycentric * center_mask[:, None, None, :], axis=-1
    )
    center_weight_u = jnp.sum(
        derivative_u * center_mask[:, None, None, :], axis=-1
    )
    center_weight_v = jnp.sum(
        derivative_v * center_mask[:, None, None, :], axis=-1
    )
    denominator = 1.0 - center_weight
    has_center = jnp.any(center_mask, axis=1)[:, None, None]
    theta = jnp.where(has_center, affine_theta / denominator, affine_theta)
    theta_u = jnp.where(
        has_center,
        (affine_theta_u * denominator + affine_theta * center_weight_u)
        / denominator**2,
        affine_theta_u,
    )
    theta_v = jnp.where(
        has_center,
        (affine_theta_v * denominator + affine_theta * center_weight_v)
        / denominator**2,
        affine_theta_v,
    )
    return rho, theta, rho_u, rho_v, theta_u, theta_v


def curved_cap_geometry(
    triangle_indices: Array,
    cap_xyz: Array,
    *,
    nxi: int,
    upper: bool,
    order: int,
) -> tuple[Array, Array]:
    """Evaluate exact star-shaped cap points and oriented area vectors."""

    cap_xyz = jnp.asarray(cap_xyz)
    _, ntheta = cap_xyz.shape[:2]
    rho, theta, rho_u, rho_v, theta_u, theta_v = _duffy_coordinates(
        triangle_indices, cap_xyz, nxi=nxi, upper=upper, order=order
    )
    rim_radius = jnp.linalg.norm(cap_xyz[-1, :, :2], axis=-1)
    modes = jnp.fft.fftfreq(ntheta, d=1.0 / ntheta)
    rim_theta = jnp.fft.ifft(1j * modes * jnp.fft.fft(rim_radius)).real
    angular_weights = periodic_interpolation_weights(theta, ntheta)
    radius = jnp.einsum("...j,j->...", angular_weights, rim_radius)
    radius_theta = jnp.einsum("...j,j->...", angular_weights, rim_theta)
    cosine, sine = jnp.cos(theta), jnp.sin(theta)
    source = jnp.stack(
        [
            rho * radius * cosine,
            rho * radius * sine,
            jnp.full_like(rho, cap_xyz[0, 0, 2]),
        ],
        axis=-1,
    )
    tangent_rho = jnp.stack(
        [radius * cosine, radius * sine, jnp.zeros_like(radius)], axis=-1
    )
    tangent_theta = jnp.stack(
        [
            rho * (radius_theta * cosine - radius * sine),
            rho * (radius_theta * sine + radius * cosine),
            jnp.zeros_like(radius),
        ],
        axis=-1,
    )
    tangent_u = tangent_rho * rho_u[..., None] + tangent_theta * theta_u[..., None]
    tangent_v = tangent_rho * rho_v[..., None] + tangent_theta * theta_v[..., None]
    return source, jnp.cross(tangent_u, tangent_v)


@partial(jax.jit, static_argnames=("nxi", "upper", "order"))
def curved_cap_layer_sum(
    target: Array,
    triangle_indices: Array,
    cap_xyz: Array,
    dirichlet: Array,
    neumann: Array,
    *,
    nxi: int,
    upper: bool,
    order: int,
) -> Array:
    """Sum Green layers over one exact polar cap."""

    source, area_vectors = curved_cap_geometry(
        triangle_indices, cap_xyz, nxi=nxi, upper=upper, order=order
    )
    displacement = target[None, None, None] - source
    inverse_radius = jax.lax.rsqrt(jnp.sum(displacement**2, axis=-1))
    area_scale = jnp.linalg.norm(area_vectors, axis=-1)
    normal_displacement_area = jnp.sum(area_vectors * displacement, axis=-1)
    _, weights = _unit_rule(order)
    weights_2d = jnp.asarray(weights)[None, :, None] * jnp.asarray(weights)[None, None, :]
    integrand = (
        neumann * area_scale * inverse_radius
        - dirichlet * normal_displacement_area * inverse_radius**3
    ) / (4.0 * jnp.pi)
    return jnp.sum(weights_2d * integrand)


@partial(jax.jit, static_argnames=("nxi", "upper", "order"))
def curved_cap_gradient_sum(
    target: Array,
    triangle_indices: Array,
    cap_xyz: Array,
    dirichlet: Array,
    neumann: Array,
    *,
    nxi: int,
    upper: bool,
    order: int,
) -> Array:
    """Evaluate the Green-layer gradient over one exact polar cap."""

    source, area_vectors = curved_cap_geometry(
        triangle_indices, cap_xyz, nxi=nxi, upper=upper, order=order
    )
    displacement = target[None, None, None] - source
    radius_squared = jnp.sum(displacement**2, axis=-1)
    inverse_radius3 = jax.lax.rsqrt(radius_squared) ** 3
    area_scale = jnp.linalg.norm(area_vectors, axis=-1)
    normal_displacement_area = jnp.sum(area_vectors * displacement, axis=-1)
    single = -neumann[..., None] * area_scale[..., None] * displacement * inverse_radius3[..., None]
    double = dirichlet[..., None] * (
        -area_vectors * inverse_radius3[..., None]
        + 3.0
        * normal_displacement_area[..., None]
        * displacement
        * (inverse_radius3 / radius_squared)[..., None]
    )
    _, weights = _unit_rule(order)
    weights_2d = (
        jnp.asarray(weights)[None, :, None, None]
        * jnp.asarray(weights)[None, None, :, None]
    )
    return jnp.sum(weights_2d * (single + double), axis=(0, 1, 2)) / (4.0 * jnp.pi)
