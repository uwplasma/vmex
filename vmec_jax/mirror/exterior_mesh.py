"""Panel connectivity for the closed mirror boundary."""

from __future__ import annotations

from functools import lru_cache
from functools import partial
from typing import Any

import jax.numpy as jnp
import jax
import numpy as np
from .exterior_interpolation import (
    cgl_interpolation_weights as _cgl_interpolation_weights,
    periodic_interpolation_weights as _periodic_interpolation_weights,
)

Array = Any


def closed_surface_triangles(lateral: np.ndarray, lower_cap: np.ndarray, upper_cap: np.ndarray) -> np.ndarray:
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


def duffy_triangle_single_layer(vertices: Array, vertex_density: Array, *, order: int = 8) -> Array:
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
        jnp.asarray(weights, dtype=vertices.dtype)[:, None] * jnp.asarray(weights, dtype=vertices.dtype)[None, :]
    )

    edge1 = vertices[1] - vertices[0]
    edge2 = vertices[2] - vertices[0]
    ray = (1.0 - v)[..., None] * edge1 + v[..., None] * edge2
    radius_per_u = jnp.linalg.norm(ray, axis=-1)
    area_scale = jnp.linalg.norm(jnp.cross(edge1, edge2))
    interpolated_density = (1.0 - u) * density[0] + u * ((1.0 - v) * density[1] + v * density[2])
    regular_integrand = area_scale * interpolated_density / (4.0 * jnp.pi * radius_per_u)
    return jnp.sum(quadrature_weights * regular_integrand)


def _linear_density_samples(values: Array, *, order: int) -> Array:
    """Interpolate triangle-vertex values to Duffy quadrature nodes."""

    nodes, _ = _unit_gauss_legendre(order)
    u = jnp.asarray(nodes, dtype=values.dtype)[None, :, None]
    v = jnp.asarray(nodes, dtype=values.dtype)[None, None, :]
    return (1.0 - u) * values[:, 0, None, None] + u * (
        (1.0 - v) * values[:, 1, None, None] + v * values[:, 2, None, None]
    )


def _side_parameter_data(
    triangle_indices: Array,
    *,
    ntheta: int,
    nxi: int,
    order: int,
    dtype: Any,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Map Duffy nodes to side ``(theta, xi)`` coordinates and derivatives."""

    indices = jnp.asarray(triangle_indices)
    nodes, _ = _unit_gauss_legendre(order)
    u = jnp.asarray(nodes, dtype=dtype)[None, :, None]
    v = jnp.asarray(nodes, dtype=dtype)[None, None, :]
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
            -jnp.ones(shape, dtype=dtype),
            jnp.broadcast_to(1.0 - v, shape),
            jnp.broadcast_to(v, shape),
        ],
        axis=-1,
    )
    derivative_v = jnp.stack(
        [
            jnp.zeros(shape, dtype=dtype),
            -jnp.broadcast_to(u, shape),
            jnp.broadcast_to(u, shape),
        ],
        axis=-1,
    )
    axial_nodes = jnp.cos(jnp.pi * jnp.arange(nxi, dtype=dtype) / (nxi - 1))[::-1]
    triangle_axial = axial_nodes[indices % nxi]
    axial = jnp.sum(barycentric * triangle_axial[:, None, None, :], axis=-1)
    axial_u = jnp.sum(derivative_u * triangle_axial[:, None, None, :], axis=-1)
    axial_v = jnp.sum(derivative_v * triangle_axial[:, None, None, :], axis=-1)

    angular = 2.0 * jnp.pi * (indices // nxi) / ntheta
    anchor = angular[:, :1]
    angular = anchor + (angular - anchor + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
    theta = jnp.sum(barycentric * angular[:, None, None, :], axis=-1)
    theta_u = jnp.sum(derivative_u * angular[:, None, None, :], axis=-1)
    theta_v = jnp.sum(derivative_v * angular[:, None, None, :], axis=-1)
    return theta, axial, theta_u, theta_v, axial_u, axial_v


def _spectral_side_density_samples(
    triangle_indices: Array,
    lateral_values: Array,
    *,
    ntheta: int,
    nxi: int,
    order: int,
    axisymmetric: bool,
) -> Array:
    """Interpolate lateral nodal data spectrally within side triangles."""

    theta, axial, *_ = _side_parameter_data(
        triangle_indices,
        ntheta=ntheta,
        nxi=nxi,
        order=order,
        dtype=lateral_values.dtype,
    )
    axial_weights = _cgl_interpolation_weights(axial, nxi)
    scalar_input = lateral_values.ndim == 1
    values = lateral_values.reshape((-1, ntheta, nxi))
    if axisymmetric:
        samples = jnp.einsum("tqrk,ak->atqr", axial_weights, values[:, 0])
        return samples[0] if scalar_input else samples

    angular_weights = _periodic_interpolation_weights(theta, ntheta)
    samples = jnp.einsum("tqrj,ajk,tqrk->atqr", angular_weights, values, axial_weights)
    return samples[0] if scalar_input else samples


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
    weights_2d = jnp.asarray(weights, dtype=dtype)[None, :, None] * jnp.asarray(weights, dtype=dtype)[None, None, :]
    edge1 = vertices[:, 1] - vertices[:, 0]
    edge2 = vertices[:, 2] - vertices[:, 0]
    ray = (1.0 - v)[..., None] * edge1[:, None, None, :] + (v[..., None] * edge2[:, None, None, :])
    source = vertices[:, 0, :][:, None, None, :] + u[..., None] * ray
    displacement = target[None, None, None, :] - source
    radius_squared = jnp.sum(displacement**2, axis=-1)
    inverse_radius = jax.lax.rsqrt(radius_squared)
    area_vectors = jnp.cross(edge1, edge2)
    area_scale = jnp.linalg.norm(area_vectors, axis=-1)
    normals = area_vectors / area_scale[:, None]
    jacobian = area_scale[:, None, None] * u

    def interpolate(values: Array) -> Array:
        if values.ndim == 3:
            return values
        return (1.0 - u) * values[:, 0, None, None] + u * (
            (1.0 - v) * values[:, 1, None, None] + v * values[:, 2, None, None]
        )

    single = interpolate(neumann) * jacobian * inverse_radius / (4.0 * jnp.pi)
    normal_displacement = jnp.einsum("ti,tqri->tqr", normals, displacement)
    double = -interpolate(dirichlet) * jacobian * normal_displacement * inverse_radius**3 / (4.0 * jnp.pi)
    return jnp.sum(weights_2d * (single + double))


@partial(jax.jit, static_argnames=("order",))
def _triangle_gradient_sum(
    target: Array,
    vertices: Array,
    dirichlet: Array,
    neumann: Array,
    *,
    order: int,
) -> Array:
    """Gradient of panel Green layers at one off-surface target."""

    nodes, weights = _unit_gauss_legendre(order)
    dtype = vertices.dtype
    u = jnp.asarray(nodes, dtype=dtype)[None, :, None]
    v = jnp.asarray(nodes, dtype=dtype)[None, None, :]
    weights_2d = jnp.asarray(weights, dtype=dtype)[None, :, None] * jnp.asarray(weights, dtype=dtype)[None, None, :]
    edge1 = vertices[:, 1] - vertices[:, 0]
    edge2 = vertices[:, 2] - vertices[:, 0]
    ray = (1.0 - v)[..., None] * edge1[:, None, None, :] + (v[..., None] * edge2[:, None, None, :])
    source = vertices[:, 0, :][:, None, None, :] + u[..., None] * ray
    displacement = target[None, None, None, :] - source
    radius_squared = jnp.sum(displacement**2, axis=-1)
    inverse_radius = jax.lax.rsqrt(radius_squared)
    inverse_radius3 = inverse_radius**3
    area_vectors = jnp.cross(edge1, edge2)
    area_scale = jnp.linalg.norm(area_vectors, axis=-1)
    normals = area_vectors / area_scale[:, None]
    jacobian = area_scale[:, None, None] * u

    def interpolate(values: Array) -> Array:
        if values.ndim == 3:
            return values
        return (1.0 - u) * values[:, 0, None, None] + u * (
            (1.0 - v) * values[:, 1, None, None] + v * values[:, 2, None, None]
        )

    normal_displacement = jnp.einsum("ti,tqri->tqr", normals, displacement)
    single = -interpolate(neumann)[..., None] * jacobian[..., None] * displacement * inverse_radius3[..., None]
    double = (
        interpolate(dirichlet)[..., None]
        * jacobian[..., None]
        * (
            -normals[:, None, None, :] * inverse_radius3[..., None]
            + 3.0 * normal_displacement[..., None] * displacement * (inverse_radius3 / radius_squared)[..., None]
        )
    )
    return jnp.sum(weights_2d[..., None] * (single + double), axis=(0, 1, 2)) / (4.0 * jnp.pi)


def panel_green_gradient_off_surface(
    xyz: Array,
    triangles: Array,
    dirichlet: Array,
    neumann: Array,
    targets: Array,
    *,
    order: int = 8,
    lateral_shape: tuple[int, int] | None = None,
    spectral_side_density: bool = False,
    axisymmetric_side: bool = False,
) -> Array:
    """Evaluate Green-layer gradients using triangular panels."""

    xyz = jnp.asarray(xyz)
    triangles = jnp.asarray(triangles)
    dirichlet = jnp.asarray(dirichlet)
    neumann = jnp.asarray(neumann)
    targets = jnp.asarray(targets)
    if targets.ndim != 2 or targets.shape[1] != 3:
        raise ValueError("targets must have shape (n, 3)")
    vertices = xyz[triangles]
    triangle_dirichlet = dirichlet[triangles]
    triangle_neumann = neumann[triangles]
    if spectral_side_density:
        if lateral_shape is None:
            raise ValueError("lateral_shape is required for spectral side density")
        ntheta, nxi = lateral_shape
        lateral_size = ntheta * nxi
        side_count = 2 * ntheta * (nxi - 1)
        side_samples = _spectral_side_density_samples(
            triangles[:side_count],
            jnp.stack([dirichlet[:lateral_size], neumann[:lateral_size]]),
            ntheta=ntheta,
            nxi=nxi,
            order=order,
            axisymmetric=axisymmetric_side,
        )
        triangle_dirichlet = (
            _linear_density_samples(triangle_dirichlet, order=order).at[:side_count].set(side_samples[0])
        )
        triangle_neumann = _linear_density_samples(triangle_neumann, order=order).at[:side_count].set(side_samples[1])
    return jnp.stack(
        [
            _triangle_gradient_sum(
                target,
                vertices,
                triangle_dirichlet,
                triangle_neumann,
                order=order,
            )
            for target in targets
        ]
    )


def panel_green_boundary_residual(
    xyz: Array,
    triangles: Array,
    dirichlet: Array,
    neumann: Array,
    *,
    order: int = 8,
    target_indices: np.ndarray | None = None,
    lateral_shape: tuple[int, int] | None = None,
    spectral_side_density: bool = False,
    axisymmetric_side: bool = False,
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

    if target_indices is None:
        target_indices = np.arange(nvertices)
    else:
        target_indices = np.asarray(target_indices, dtype=int)
        if target_indices.ndim != 1 or np.any((target_indices < 0) | (target_indices >= nvertices)):
            raise ValueError("target_indices must select valid vertices")

    residual = []
    for target_index in target_indices:
        target_index = int(target_index)
        ordered = np.array(triangles_np, copy=True)
        rows, positions = np.nonzero(ordered == target_index)
        for row, position in zip(rows, positions, strict=True):
            ordered[row] = np.roll(ordered[row], -int(position))
        triangle_indices = jnp.asarray(ordered)
        triangle_dirichlet = dirichlet[triangle_indices] - dirichlet[target_index]
        triangle_neumann = neumann[triangle_indices]
        if spectral_side_density:
            if lateral_shape is None:
                raise ValueError("lateral_shape is required for spectral side density")
            ntheta, nxi = lateral_shape
            lateral_size = ntheta * nxi
            side_count = 2 * ntheta * (nxi - 1)
            side_samples = _spectral_side_density_samples(
                triangle_indices[:side_count],
                jnp.stack([dirichlet[:lateral_size], neumann[:lateral_size]]),
                ntheta=ntheta,
                nxi=nxi,
                order=order,
                axisymmetric=axisymmetric_side,
            )
            triangle_dirichlet = (
                _linear_density_samples(triangle_dirichlet, order=order)
                .at[:side_count]
                .set(side_samples[0] - dirichlet[target_index])
            )
            triangle_neumann = (
                _linear_density_samples(triangle_neumann, order=order).at[:side_count].set(side_samples[1])
            )
        residual.append(
            _triangle_layer_sum(
                xyz[target_index],
                xyz[triangle_indices],
                triangle_dirichlet,
                triangle_neumann,
                order=order,
            )
        )
    return jnp.stack(residual)
