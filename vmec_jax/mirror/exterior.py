"""Closed-surface quadrature for the unbounded mirror vacuum.

The plasma side wall alone is not a closed boundary: the two axial cuts must
be filled by disks before a free-space boundary integral is well posed.  This
module builds that geometric adapter.  It deliberately contains no Laplace
solver; the quadrature identities are validated first, then the kernels from
``virtual_casing_jax`` can be attached without duplicating geometry logic.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from virtual_casing_jax import laplace_dx_u_eval, laplace_fx_u, laplace_fxd_u_eval

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
    triangles: Array

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
    data_fields=[field.name for field in fields(ClosedMirrorSurface)],
    meta_fields=[],
)


def build_closed_mirror_surface(
    boundary: "MirrorBoundary",
    grid: "MirrorGrid",
    *,
    axisymmetric_ntheta: int = 16,
) -> ClosedMirrorSurface:
    """Close a star-shaped mirror LCFS with disks on both axial cuts.

    The side wall uses ``(theta, xi)`` and each cap uses the regular disk
    coordinate ``r = sqrt(s) a(theta)``.  The latter gives the nonsingular area
    element ``a(theta)^2 ds dtheta / 2`` even at ``s=0``.
    """

    radius = jnp.asarray(boundary.radius_scale)
    expected = (grid.ntheta, grid.nxi)
    if radius.shape != expected:
        raise ValueError(f"boundary shape {radius.shape} does not match {expected}")

    if grid.ntheta == 1:
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

    sqrt_s = jnp.sqrt(jnp.asarray(grid.s))[:, None]
    radial_weights = jnp.asarray(grid.radial_weights)[:, None]
    theta_weights = theta_weights_1d[None, :]

    def cap(endpoint: int, orientation: float) -> tuple[Array, Array]:
        cap_radius = sqrt_s * radius[:, endpoint][None, :]
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
    triangles = jnp.asarray(
        closed_surface_triangles(lateral_map, lower_map, upper_map)
    )
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
        triangles=triangles,
    )


def laplace_double_layer_off_surface(
    surface: ClosedMirrorSurface,
    density: Array,
    targets: Array,
    *,
    chunk_size: int = 1024,
) -> Array:
    """Evaluate a Laplace double layer away from the source surface.

    This thin adapter supplies the open-mirror quadrature to the tested
    ``virtual_casing_jax`` kernel.  Targets must not lie on the surface;
    singular and near-singular evaluation is a separate M5 gate.
    """

    density, targets = _validate_off_surface_inputs(surface, density, targets)
    value = laplace_dx_u_eval(
        surface.xyz.T,
        surface.normals.T,
        targets.T,
        density,
        surface.quadrature_weights,
        chunk_size=chunk_size,
    )
    return value.reshape(-1)


def laplace_single_layer_gradient_off_surface(
    surface: ClosedMirrorSurface,
    density: Array,
    targets: Array,
    *,
    chunk_size: int = 1024,
) -> Array:
    """Evaluate ``grad integral G density dA`` away from the surface."""

    density, targets = _validate_off_surface_inputs(surface, density, targets)
    value = laplace_fxd_u_eval(
        surface.xyz.T,
        targets.T,
        density,
        surface.quadrature_weights,
        chunk_size=chunk_size,
    )
    return value.T


def laplace_single_layer_off_surface(
    surface: ClosedMirrorSurface, density: Array, targets: Array
) -> Array:
    """Evaluate ``integral G density dA`` away from the surface."""

    density, targets = _validate_off_surface_inputs(surface, density, targets)
    displacement = targets[:, None, :] - surface.xyz[None, :, :]
    weighted_density = density * surface.quadrature_weights
    return jnp.sum(laplace_fx_u(displacement, weighted_density[None, :]), axis=1)


def laplace_green_representation_off_surface(
    surface: ClosedMirrorSurface,
    dirichlet: Array,
    neumann: Array,
    targets: Array,
) -> Array:
    """Evaluate Green's representation from harmonic boundary data.

    For consistent Dirichlet and outward-normal Neumann data this returns the
    harmonic function inside the closed surface and zero outside.  It is a
    manufactured-solution validator for the mixed wall/cap quadrature.
    """

    return laplace_single_layer_off_surface(surface, neumann, targets) + (
        laplace_double_layer_off_surface(surface, dirichlet, targets)
    )


def _validate_off_surface_inputs(
    surface: ClosedMirrorSurface, density: Array, targets: Array
) -> tuple[Array, Array]:
    density = jnp.asarray(density)
    targets = jnp.asarray(targets)
    if density.shape != (surface.xyz.shape[0],):
        raise ValueError(
            f"density shape {density.shape} must be ({surface.xyz.shape[0]},)"
        )
    if targets.ndim != 2 or targets.shape[1] != 3:
        raise ValueError("targets must have shape (n, 3)")
    return density, targets


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .basis import MirrorGrid
    from .model import MirrorBoundary
