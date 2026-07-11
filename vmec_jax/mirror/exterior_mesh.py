"""Panel connectivity for the closed mirror boundary."""

from __future__ import annotations

from functools import lru_cache
from functools import partial
from typing import Any

import jax.numpy as jnp
import jax
import numpy as np

Array = Any


def closed_surface_triangles(
    lateral: np.ndarray, lower_cap: np.ndarray, upper_cap: np.ndarray
) -> np.ndarray:
    """Triangulate periodic side quads and polar caps with outward orientation."""

    lateral = np.asarray(lateral, dtype=int)
    lower_cap = np.asarray(lower_cap, dtype=int)
    upper_cap = np.asarray(upper_cap, dtype=int)
    ntheta, nxi = lateral.shape
    triangles: list[tuple[int, int, int]] = []

    for j in range(ntheta):
        jp = (j + 1) % ntheta
        for k in range(nxi - 1):
            triangles.append((lateral[j, k], lateral[jp, k], lateral[jp, k + 1]))
            triangles.append((lateral[j, k], lateral[jp, k + 1], lateral[j, k + 1]))

    def add_cap(mapping: np.ndarray, *, upper: bool) -> None:
        center = int(mapping[0, 0])
        rings = mapping[1:]
        for j in range(ntheta):
            jp = (j + 1) % ntheta
            triangle = (center, int(rings[0, j]), int(rings[0, jp]))
            triangles.append(triangle if upper else triangle[::-1])
        for inner, outer in zip(rings[:-1], rings[1:], strict=True):
            for j in range(ntheta):
                jp = (j + 1) % ntheta
                first = (int(inner[j]), int(outer[j]), int(outer[jp]))
                second = (int(inner[j]), int(outer[jp]), int(inner[jp]))
                triangles.extend((first, second) if upper else (first[::-1], second[::-1]))

    add_cap(lower_cap, upper=False)
    add_cap(upper_cap, upper=True)
    return np.asarray(triangles, dtype=int)


@lru_cache(maxsize=None)
def _unit_gauss_legendre(order: int) -> tuple[np.ndarray, np.ndarray]:
    order = int(order)
    if order < 1:
        raise ValueError("quadrature order must be positive")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return 0.5 * (nodes + 1.0), 0.5 * weights


def duffy_triangle_single_layer(
    vertices: Array, vertex_density: Array, *, order: int = 8
) -> Array:
    """Integrate ``density/(4*pi*r)`` with the target at vertex zero.

    The Duffy map ``y = v0 + u[(1-v)(v1-v0) + v(v2-v0)]`` contributes a
    Jacobian proportional to ``u`` that cancels the Laplace ``1/r``
    singularity. Density is interpolated linearly from the three vertices.
    """

    vertices = jnp.asarray(vertices)
    density = jnp.asarray(vertex_density)
    if vertices.shape != (3, 3):
        raise ValueError("vertices must have shape (3, 3)")
    if density.shape != (3,):
        raise ValueError("vertex_density must have shape (3,)")
    nodes, weights = _unit_gauss_legendre(order)
    u = jnp.asarray(nodes, dtype=vertices.dtype)[:, None]
    v = jnp.asarray(nodes, dtype=vertices.dtype)[None, :]
    quadrature_weights = (
        jnp.asarray(weights, dtype=vertices.dtype)[:, None]
        * jnp.asarray(weights, dtype=vertices.dtype)[None, :]
    )

    edge1 = vertices[1] - vertices[0]
    edge2 = vertices[2] - vertices[0]
    ray = (1.0 - v)[..., None] * edge1 + v[..., None] * edge2
    radius_per_u = jnp.linalg.norm(ray, axis=-1)
    area_scale = jnp.linalg.norm(jnp.cross(edge1, edge2))
    interpolated_density = (1.0 - u) * density[0] + u * (
        (1.0 - v) * density[1] + v * density[2]
    )
    regular_integrand = (
        area_scale * interpolated_density / (4.0 * jnp.pi * radius_per_u)
    )
    return jnp.sum(quadrature_weights * regular_integrand)


@partial(jax.jit, static_argnames=("order",))
def _triangle_layer_sum(
    target: Array,
    vertices: Array,
    dirichlet: Array,
    neumann: Array,
    *,
    order: int,
) -> Array:
    """Sum single and double layers over triangles anchored for one target."""

    nodes, weights = _unit_gauss_legendre(order)
    dtype = vertices.dtype
    u = jnp.asarray(nodes, dtype=dtype)[None, :, None]
    v = jnp.asarray(nodes, dtype=dtype)[None, None, :]
    weights_2d = (
        jnp.asarray(weights, dtype=dtype)[None, :, None]
        * jnp.asarray(weights, dtype=dtype)[None, None, :]
    )
    edge1 = vertices[:, 1] - vertices[:, 0]
    edge2 = vertices[:, 2] - vertices[:, 0]
    ray = (1.0 - v)[..., None] * edge1[:, None, None, :] + (
        v[..., None] * edge2[:, None, None, :]
    )
    source = vertices[:, 0, :][:, None, None, :] + u[..., None] * ray
    displacement = target[None, None, None, :] - source
    radius_squared = jnp.sum(displacement**2, axis=-1)
    inverse_radius = jax.lax.rsqrt(radius_squared)
    area_vectors = jnp.cross(edge1, edge2)
    area_scale = jnp.linalg.norm(area_vectors, axis=-1)
    normals = area_vectors / area_scale[:, None]
    jacobian = area_scale[:, None, None] * u

    def interpolate(values: Array) -> Array:
        return (1.0 - u) * values[:, 0, None, None] + u * (
            (1.0 - v) * values[:, 1, None, None]
            + v * values[:, 2, None, None]
        )

    single = interpolate(neumann) * jacobian * inverse_radius / (4.0 * jnp.pi)
    normal_displacement = jnp.einsum(
        "ti,tqri->tqr", normals, displacement
    )
    double = (
        -interpolate(dirichlet)
        * jacobian
        * normal_displacement
        * inverse_radius**3
        / (4.0 * jnp.pi)
    )
    return jnp.sum(weights_2d * (single + double))


def panel_green_boundary_residual(
    xyz: Array,
    triangles: Array,
    dirichlet: Array,
    neumann: Array,
    *,
    order: int = 8,
) -> Array:
    """Evaluate ``S(q) + K(u-u_target)`` at all mesh vertices.

    Incident triangles are reordered so the collocation vertex is Duffy's
    singular vertex. The subtraction makes constants an exact nullspace and
    avoids assuming a smooth-surface jump coefficient at cap rims.
    """

    xyz = jnp.asarray(xyz)
    triangles_np = np.asarray(triangles, dtype=int)
    dirichlet = jnp.asarray(dirichlet)
    neumann = jnp.asarray(neumann)
    nvertices = int(xyz.shape[0])
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape (n, 3)")
    if dirichlet.shape != (nvertices,) or neumann.shape != (nvertices,):
        raise ValueError("dirichlet and neumann must have one value per vertex")

    residual = []
    for target_index in range(nvertices):
        ordered = np.array(triangles_np, copy=True)
        rows, positions = np.nonzero(ordered == target_index)
        for row, position in zip(rows, positions, strict=True):
            ordered[row, [0, position]] = ordered[row, [position, 0]]
        triangle_indices = jnp.asarray(ordered)
        triangle_dirichlet = dirichlet[triangle_indices] - dirichlet[target_index]
        residual.append(
            _triangle_layer_sum(
                xyz[target_index],
                xyz[triangle_indices],
                triangle_dirichlet,
                neumann[triangle_indices],
                order=order,
            )
        )
    return jnp.stack(residual)
