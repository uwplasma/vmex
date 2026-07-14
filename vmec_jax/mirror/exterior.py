"""Closed-surface quadrature for the unbounded mirror vacuum.

The plasma side wall alone is not a closed boundary: the two axial cuts must
be filled by disks before a free-space boundary integral is well posed. The
cuts carry magnetic through-flux and are not plasma-vacuum interfaces; the
disks only close the Green integration surface and do not impose ``B.n=0`` or
pressure balance. This module builds that geometric adapter. It deliberately
contains no Laplace solver; the quadrature identities are validated first,
then the kernels from ``virtual_casing_jax`` can be attached without
duplicating geometry logic.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from .exterior_mesh import closed_surface_triangles

Array = Any


@dataclass(frozen=True)
class ClosedMirrorSurface:
    """Lateral wall and end-disk quadrature with outward weighted normals.

    ``weighted_normals`` are ``n dA`` including the tensor-product quadrature
    weights.  Keeping this quantity directly avoids unit-normal divisions at
    the disk centers and is the natural source measure for boundary integrals.
    """

    lateral_xyz: Array
    lateral_weighted_normals: Array
    lower_cap_xyz: Array
    lower_cap_weighted_normals: Array
    upper_cap_xyz: Array
    upper_cap_weighted_normals: Array
    collocation_xyz: Array
    collocation_normals: Array
    quadrature_to_collocation: Array
    collocation_to_reduced: Array
    triangles: Array
    n_reduced: int
    reduced_representatives: tuple[int, ...]
    triangle_connectivity: tuple[tuple[int, int, int], ...]

    @property
    def xyz(self) -> Array:
        """All quadrature nodes as a flat ``(n, 3)`` array."""

        return jnp.concatenate(
            [
                self.lateral_xyz.reshape(-1, 3),
                self.lower_cap_xyz.reshape(-1, 3),
                self.upper_cap_xyz.reshape(-1, 3),
            ],
            axis=0,
        )

    @property
    def weighted_normals(self) -> Array:
        """Outward ``n dA`` vectors in the same ordering as :attr:`xyz`."""

        return jnp.concatenate(
            [
                self.lateral_weighted_normals.reshape(-1, 3),
                self.lower_cap_weighted_normals.reshape(-1, 3),
                self.upper_cap_weighted_normals.reshape(-1, 3),
            ],
            axis=0,
        )

    @property
    def quadrature_weights(self) -> Array:
        """Scalar surface-area weights ``dA``."""

        return jnp.linalg.norm(self.weighted_normals, axis=-1)

    @property
    def normals(self) -> Array:
        """Outward unit normals at all quadrature nodes."""

        weights = self.quadrature_weights
        return self.weighted_normals / weights[:, None]

    @property
    def area(self) -> Array:
        """Total closed-surface area."""

        return jnp.sum(self.quadrature_weights)

    @property
    def volume(self) -> Array:
        """Enclosed volume from ``integral(x dot n) / 3``."""

        return jnp.sum(self.xyz * self.weighted_normals) / 3.0

    def expand_collocation_values(self, values: Array) -> Array:
        """Copy unique boundary values onto all quadrature nodes."""

        values = jnp.asarray(values)
        expected = (self.collocation_xyz.shape[0],)
        if values.shape != expected:
            raise ValueError(f"collocation values shape {values.shape} must be {expected}")
        return values[self.quadrature_to_collocation]

    @property
    def reduced_size(self) -> int:
        """Number of independent density values after symmetry reduction."""

        return self.n_reduced

    def expand_reduced_values(self, values: Array) -> Array:
        """Expand reduced density values onto unique collocation nodes."""

        values = jnp.asarray(values)
        expected = (self.reduced_size,)
        if values.shape != expected:
            raise ValueError(f"reduced values shape {values.shape} must be {expected}")
        return values[self.collocation_to_reduced]

    def reduce_collocation_values(self, values: Array) -> Array:
        """Average collocation values over each symmetry orbit."""

        values = jnp.asarray(values)
        expected = (self.collocation_xyz.shape[0],)
        if values.shape != expected:
            raise ValueError(f"collocation values shape {values.shape} must be {expected}")
        indices = jnp.asarray(self.collocation_to_reduced)
        totals = jnp.zeros(self.reduced_size, dtype=values.dtype).at[indices].add(values)
        counts = jnp.zeros(self.reduced_size, dtype=values.dtype).at[indices].add(1.0)
        return totals / counts

    @property
    def triangle_xyz(self) -> Array:
        """Linear-panel vertices with shape ``(ntriangle, 3, 3)``."""

        return self.collocation_xyz[self.triangles]

    @property
    def mesh_area(self) -> Array:
        """Area of the piecewise-linear closed panel mesh."""

        vertices = self.triangle_xyz
        cross = jnp.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
        return 0.5 * jnp.sum(jnp.linalg.norm(cross, axis=1))

    @property
    def mesh_volume(self) -> Array:
        """Signed volume of the outward-oriented panel mesh."""

        vertices = self.triangle_xyz
        return jnp.sum(
            jnp.einsum("ij,ij->i", vertices[:, 0], jnp.cross(vertices[:, 1], vertices[:, 2]))
        ) / 6.0


jax.tree_util.register_dataclass(
    ClosedMirrorSurface,
    data_fields=[
        field.name
        for field in fields(ClosedMirrorSurface)
        if field.name
        not in {"n_reduced", "reduced_representatives", "triangle_connectivity"}
    ],
    meta_fields=["n_reduced", "reduced_representatives", "triangle_connectivity"],
)


def build_closed_mirror_surface(
    boundary: "MirrorBoundary",
    grid: "MirrorGrid",
    *,
    axisymmetric_ntheta: int = 16,
    cap_rim_grade: float = 1.0,
) -> ClosedMirrorSurface:
    """Close a star-shaped mirror LCFS with disks on both axial cuts.

    The side wall uses ``(theta, xi)`` and each cap uses the regular disk
    coordinate ``r = rho(s) a(theta)``. ``cap_rim_grade > 1`` clusters panels
    at the sharp rim. The area coordinate remains ``rho^2``, giving the regular
    element ``a(theta)^2 d(rho^2) dtheta / 2`` at the center.
    """

    radius = jnp.asarray(boundary.radius_scale)
    expected = (grid.ntheta, grid.nxi)
    if radius.shape != expected:
        raise ValueError(f"boundary shape {radius.shape} does not match {expected}")

    axisymmetric = grid.ntheta == 1
    if axisymmetric:
        # The equilibrium stores no redundant angular samples in axisymmetry,
        # but Cartesian surface moments still require an angular quadrature.
        axisymmetric_ntheta = int(axisymmetric_ntheta)
        if axisymmetric_ntheta < 4:
            raise ValueError("axisymmetric_ntheta must be at least 4")
        theta = jnp.asarray(
            np.linspace(0.0, 2.0 * np.pi, axisymmetric_ntheta, endpoint=False)
        )
        theta_weights_1d = jnp.full(theta.shape, 2.0 * jnp.pi / theta.size)
        radius = jnp.broadcast_to(radius, (theta.size, grid.nxi))
        d_radius_dtheta = jnp.zeros_like(radius)
    else:
        theta = jnp.asarray(grid.theta)
        theta_weights_1d = jnp.asarray(grid.theta_basis.weights)
        d_radius_dtheta = grid.theta_basis.differentiate(radius, axis=0)
    cosine = jnp.cos(theta)
    sine = jnp.sin(theta)
    z = jnp.asarray(grid.z)
    dz_dxi = float(grid.dz_dxi)
    d_radius_dxi = grid.axial_basis.differentiate(radius, axis=1)

    lateral_xyz = jnp.stack(
        [
            radius * cosine[:, None],
            radius * sine[:, None],
            jnp.broadcast_to(z[None, :], radius.shape),
        ],
        axis=-1,
    )
    lateral_area_vectors = jnp.stack(
        [
            radius * dz_dxi * cosine[:, None],
            radius * dz_dxi * sine[:, None],
            -radius * d_radius_dxi,
        ],
        axis=-1,
    )
    lateral_area_vectors = lateral_area_vectors.at[..., 0].add(
        d_radius_dtheta * dz_dxi * sine[:, None]
    )
    lateral_area_vectors = lateral_area_vectors.at[..., 1].add(
        -d_radius_dtheta * dz_dxi * cosine[:, None]
    )
    lateral_weights = (
        theta_weights_1d[:, None]
        * jnp.asarray(grid.axial_basis.weights)[None, :]
    )
    lateral_weighted_normals = lateral_area_vectors * lateral_weights[..., None]

    cap_rim_grade = float(cap_rim_grade)
    if not np.isfinite(cap_rim_grade) or cap_rim_grade < 1.0:
        raise ValueError("cap_rim_grade must be finite and at least 1")
    base_radius_nodes = np.sqrt(np.asarray(grid.s))
    cap_radius_nodes = 1.0 - (1.0 - base_radius_nodes) ** cap_rim_grade
    cap_area_nodes = cap_radius_nodes**2
    cap_radial_weights = np.empty_like(cap_area_nodes)
    cap_radial_weights[0] = 0.5 * (cap_area_nodes[1] - cap_area_nodes[0])
    cap_radial_weights[-1] = 0.5 * (cap_area_nodes[-1] - cap_area_nodes[-2])
    cap_radial_weights[1:-1] = 0.5 * (
        cap_area_nodes[2:] - cap_area_nodes[:-2]
    )
    cap_radius_nodes = jnp.asarray(cap_radius_nodes)[:, None]
    radial_weights = jnp.asarray(cap_radial_weights)[:, None]
    theta_weights = theta_weights_1d[None, :]

    def cap(endpoint: int, orientation: float) -> tuple[Array, Array]:
        cap_radius = cap_radius_nodes * radius[:, endpoint][None, :]
        cap_xyz = jnp.stack(
            [
                cap_radius * cosine[None, :],
                cap_radius * sine[None, :],
                jnp.full_like(cap_radius, z[endpoint]),
            ],
            axis=-1,
        )
        area = 0.5 * radius[:, endpoint] ** 2
        weighted_z = orientation * radial_weights * theta_weights * area[None, :]
        cap_weighted_normals = jnp.stack(
            [jnp.zeros_like(weighted_z), jnp.zeros_like(weighted_z), weighted_z],
            axis=-1,
        )
        return cap_xyz, cap_weighted_normals

    lower_xyz, lower_normals = cap(0, -1.0)
    upper_xyz, upper_normals = cap(-1, 1.0)

    ntheta, nxi = radius.shape
    lateral_map = np.arange(ntheta * nxi).reshape(ntheta, nxi)
    next_index = lateral_map.size

    def cap_map(endpoint: int) -> tuple[np.ndarray, int]:
        nonlocal next_index
        mapping = np.empty((grid.ns, ntheta), dtype=int)
        mapping[0] = next_index
        next_index += 1
        interior_size = max(0, grid.ns - 2) * ntheta
        mapping[1:-1] = np.arange(next_index, next_index + interior_size).reshape(
            grid.ns - 2, ntheta
        )
        next_index += interior_size
        mapping[-1] = lateral_map[:, endpoint]
        return mapping, next_index

    lower_map, _ = cap_map(0)
    upper_map, _ = cap_map(-1)
    quadrature_to_collocation = jnp.asarray(
        np.concatenate([lateral_map.reshape(-1), lower_map.reshape(-1), upper_map.reshape(-1)])
    )

    lateral_normals = lateral_area_vectors / jnp.linalg.norm(
        lateral_area_vectors, axis=-1, keepdims=True
    )
    lower_collocation_xyz = jnp.concatenate(
        [lower_xyz[0, :1], lower_xyz[1:-1].reshape(-1, 3)], axis=0
    )
    upper_collocation_xyz = jnp.concatenate(
        [upper_xyz[0, :1], upper_xyz[1:-1].reshape(-1, 3)], axis=0
    )
    lower_collocation_normals = jnp.zeros_like(lower_collocation_xyz).at[:, 2].set(-1.0)
    upper_collocation_normals = jnp.zeros_like(upper_collocation_xyz).at[:, 2].set(1.0)
    collocation_xyz = jnp.concatenate(
        [lateral_xyz.reshape(-1, 3), lower_collocation_xyz, upper_collocation_xyz]
    )
    collocation_normals = jnp.concatenate(
        [
            lateral_normals.reshape(-1, 3),
            lower_collocation_normals,
            upper_collocation_normals,
        ]
    )
    if axisymmetric:
        lateral_reduced = np.tile(np.arange(nxi), ntheta)
        next_reduced = nxi

        def cap_reduced() -> np.ndarray:
            nonlocal next_reduced
            center = np.asarray([next_reduced])
            next_reduced += 1
            rings = np.repeat(
                np.arange(next_reduced, next_reduced + grid.ns - 2), ntheta
            )
            next_reduced += grid.ns - 2
            return np.concatenate([center, rings])

        collocation_to_reduced = np.concatenate(
            [lateral_reduced, cap_reduced(), cap_reduced()]
        )
    else:
        collocation_to_reduced = np.arange(collocation_xyz.shape[0])
    triangle_array = closed_surface_triangles(lateral_map, lower_map, upper_map)
    triangles = jnp.asarray(triangle_array)
    return ClosedMirrorSurface(
        lateral_xyz=lateral_xyz,
        lateral_weighted_normals=lateral_weighted_normals,
        lower_cap_xyz=lower_xyz,
        lower_cap_weighted_normals=lower_normals,
        upper_cap_xyz=upper_xyz,
        upper_cap_weighted_normals=upper_normals,
        collocation_xyz=collocation_xyz,
        collocation_normals=collocation_normals,
        quadrature_to_collocation=quadrature_to_collocation,
        collocation_to_reduced=jnp.asarray(collocation_to_reduced),
        triangles=triangles,
        n_reduced=int(np.max(collocation_to_reduced)) + 1,
        reduced_representatives=tuple(
            int(index)
            for index in np.unique(collocation_to_reduced, return_index=True)[1]
        ),
        triangle_connectivity=tuple(
            (int(triangle[0]), int(triangle[1]), int(triangle[2]))
            for triangle in triangle_array
        ),
    )


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .basis import MirrorGrid
    from .model import MirrorBoundary
