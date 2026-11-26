"""Closed-surface quadrature and Laplace solver for the unbounded mirror vacuum.

The plasma side wall alone is not a closed boundary: the two axial cuts must
be filled by disks before a free-space boundary integral is well posed. The
cuts carry magnetic through-flux and are not plasma-vacuum interfaces; the
disks only close the Green integration surface and do not impose ``B.n=0`` or
pressure balance. This module builds that geometric adapter together with the
reduced-Neumann Laplace boundary-integral solver that runs on it.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from functools import lru_cache, partial
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from .basis import MirrorGrid
    from .geometry import ContravariantField
    from .model import MirrorBoundary

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
            [self.lateral_xyz.reshape(-1, 3), self.lower_cap_xyz.reshape(-1, 3), self.upper_cap_xyz.reshape(-1, 3)],
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
        return self.weighted_normals / self.quadrature_weights[:, None]

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
        return jnp.sum(jnp.einsum("ij,ij->i", vertices[:, 0], jnp.cross(vertices[:, 1], vertices[:, 2]))) / 6.0


_SURFACE_META = ("n_reduced", "reduced_representatives", "triangle_connectivity")
jax.tree_util.register_dataclass(
    ClosedMirrorSurface,
    data_fields=[f.name for f in fields(ClosedMirrorSurface) if f.name not in _SURFACE_META],
    meta_fields=list(_SURFACE_META),
)


@lru_cache(maxsize=64)
def _closed_surface_topology(
    ntheta: int, nxi: int, ns: int, axisymmetric: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, tuple[int, ...], tuple[tuple[int, int, int], ...]]:
    lateral_map = np.arange(ntheta * nxi, dtype=np.int32).reshape(ntheta, nxi)
    next_index = lateral_map.size

    def cap_map(endpoint: int) -> np.ndarray:
        nonlocal next_index
        mapping = np.empty((ns, ntheta), dtype=np.int32)
        mapping[0] = next_index
        next_index += 1
        interior_size = max(0, ns - 2) * ntheta
        mapping[1:-1] = np.arange(next_index, next_index + interior_size, dtype=np.int32).reshape(ns - 2, ntheta)
        next_index += interior_size
        mapping[-1] = lateral_map[:, endpoint]
        return mapping

    lower_map, upper_map = cap_map(0), cap_map(-1)
    quad_to_coll = np.concatenate([lateral_map.reshape(-1), lower_map.reshape(-1), upper_map.reshape(-1)])
    if axisymmetric:
        ring_ids = np.concatenate([[0], np.repeat(np.arange(1, ns - 1, dtype=np.int32), ntheta)])
        coll_to_red = np.concatenate(
            [np.tile(np.arange(nxi, dtype=np.int32), ntheta), nxi + ring_ids, nxi + (ns - 1) + ring_ids]
        )
    else:
        coll_to_red = np.arange(next_index, dtype=np.int32)

    triangle_array = closed_surface_triangles(lateral_map, lower_map, upper_map).astype(np.int32)
    n_reduced = int(np.max(coll_to_red)) + 1
    representatives = tuple(int(idx) for idx in np.unique(coll_to_red, return_index=True)[1])
    connectivity = tuple((int(tri[0]), int(tri[1]), int(tri[2])) for tri in triangle_array)
    return quad_to_coll, coll_to_red, triangle_array, n_reduced, representatives, connectivity


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
        axisymmetric_ntheta = int(axisymmetric_ntheta)
        if axisymmetric_ntheta < 4:
            raise ValueError("axisymmetric_ntheta must be at least 4")
        theta = jnp.asarray(np.linspace(0.0, 2.0 * np.pi, axisymmetric_ntheta, endpoint=False))
        theta_weights_1d = jnp.full(theta.shape, 2.0 * jnp.pi / theta.size)
        radius = jnp.broadcast_to(radius, (theta.size, grid.nxi))
        d_radius_dtheta = jnp.zeros_like(radius)
    else:
        theta = jnp.asarray(grid.theta)
        theta_weights_1d = jnp.asarray(grid.theta_basis.weights)
        d_radius_dtheta = grid.theta_basis.differentiate(radius, axis=0)
    cosine, sine = jnp.cos(theta), jnp.sin(theta)
    z, dz_dxi = jnp.asarray(grid.z), float(grid.dz_dxi)
    d_radius_dxi = grid.axial_basis.differentiate(radius, axis=1)

    lateral_xyz = jnp.stack(
        [radius * cosine[:, None], radius * sine[:, None], jnp.broadcast_to(z[None, :], radius.shape)], axis=-1
    )
    lateral_area_vectors = jnp.stack(
        [
            radius * dz_dxi * cosine[:, None] + d_radius_dtheta * dz_dxi * sine[:, None],
            radius * dz_dxi * sine[:, None] - d_radius_dtheta * dz_dxi * cosine[:, None],
            -radius * d_radius_dxi,
        ],
        axis=-1,
    )
    lateral_weights = theta_weights_1d[:, None] * jnp.asarray(grid.axial_basis.weights)[None, :]
    lateral_weighted_normals = lateral_area_vectors * lateral_weights[..., None]

    cap_rim_grade = float(cap_rim_grade)
    if not np.isfinite(cap_rim_grade) or cap_rim_grade < 1.0:
        raise ValueError("cap_rim_grade must be finite and at least 1")
    cap_radius_nodes = 1.0 - (1.0 - np.sqrt(np.asarray(grid.s))) ** cap_rim_grade
    cap_area_nodes = cap_radius_nodes**2
    cap_radial_weights = np.empty_like(cap_area_nodes)
    cap_radial_weights[0] = 0.5 * (cap_area_nodes[1] - cap_area_nodes[0])
    cap_radial_weights[-1] = 0.5 * (cap_area_nodes[-1] - cap_area_nodes[-2])
    cap_radial_weights[1:-1] = 0.5 * (cap_area_nodes[2:] - cap_area_nodes[:-2])
    cap_radius_col = jnp.asarray(cap_radius_nodes)[:, None]
    radial_weights = jnp.asarray(cap_radial_weights)[:, None]
    theta_weights = theta_weights_1d[None, :]

    def cap(endpoint: int, orientation: float) -> tuple[Array, Array]:
        cap_radius = cap_radius_col * radius[:, endpoint][None, :]
        cap_xyz = jnp.stack(
            [cap_radius * cosine[None, :], cap_radius * sine[None, :], jnp.full_like(cap_radius, z[endpoint])], axis=-1
        )
        weighted_z = orientation * radial_weights * theta_weights * (0.5 * radius[:, endpoint] ** 2)[None, :]
        return cap_xyz, jnp.stack([jnp.zeros_like(weighted_z), jnp.zeros_like(weighted_z), weighted_z], axis=-1)

    lower_xyz, lower_normals = cap(0, -1.0)
    upper_xyz, upper_normals = cap(-1, 1.0)
    lateral_normals = lateral_area_vectors / jnp.linalg.norm(lateral_area_vectors, axis=-1, keepdims=True)
    lower_collocation_xyz = jnp.concatenate([lower_xyz[0, :1], lower_xyz[1:-1].reshape(-1, 3)], axis=0)
    upper_collocation_xyz = jnp.concatenate([upper_xyz[0, :1], upper_xyz[1:-1].reshape(-1, 3)], axis=0)
    collocation_xyz = jnp.concatenate([lateral_xyz.reshape(-1, 3), lower_collocation_xyz, upper_collocation_xyz])
    collocation_normals = jnp.concatenate(
        [
            lateral_normals.reshape(-1, 3),
            jnp.zeros_like(lower_collocation_xyz).at[:, 2].set(-1.0),
            jnp.zeros_like(upper_collocation_xyz).at[:, 2].set(1.0),
        ]
    )
    ntheta, nxi = radius.shape
    quad_np, red_np, tri_np, n_reduced, representatives, connectivity = _closed_surface_topology(
        ntheta, nxi, int(grid.ns), axisymmetric
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
        quadrature_to_collocation=jnp.asarray(quad_np),
        collocation_to_reduced=jnp.asarray(red_np),
        triangles=jnp.asarray(tri_np),
        n_reduced=n_reduced,
        reduced_representatives=representatives,
        triangle_connectivity=connectivity,
    )


def _normalized_interpolation_weights(nodes: Array, barycentric: Array, targets: Array) -> Array:
    targets = jnp.asarray(targets)
    difference = targets[..., None] - nodes
    exact = jnp.abs(difference) <= 8.0 * jnp.finfo(nodes.dtype).eps
    scaled = barycentric / jnp.where(exact, 1.0, difference)
    weights = scaled / jnp.sum(scaled, axis=-1, keepdims=True)
    return jnp.where(jnp.any(exact, axis=-1, keepdims=True), exact, weights)


def _cgl_interpolation_weights(targets: Array, nxi: int) -> Array:
    targets = jnp.asarray(targets)
    nodes = jnp.cos(jnp.pi * jnp.arange(nxi, dtype=targets.dtype) / (nxi - 1))[::-1]
    barycentric = ((-1.0) ** jnp.arange(nxi, dtype=targets.dtype)).at[jnp.asarray([0, nxi - 1])].multiply(0.5)
    return _normalized_interpolation_weights(nodes, barycentric, targets)


def _periodic_interpolation_weights(targets: Array, ntheta: int) -> Array:
    targets = jnp.asarray(targets)
    difference = targets[..., None] - 2.0 * jnp.pi * jnp.arange(ntheta, dtype=targets.dtype) / ntheta
    modes = jnp.arange(1, (ntheta + 1) // 2, dtype=targets.dtype)
    weights = 1.0 + 2.0 * jnp.sum(jnp.cos(difference[..., None] * modes), axis=-1)
    if ntheta % 2 == 0:
        weights = weights + jnp.cos((ntheta // 2) * difference)
    return weights / ntheta


def closed_surface_triangles(lateral: np.ndarray, lower_cap: np.ndarray, upper_cap: np.ndarray) -> np.ndarray:
    """Triangulate periodic side quads and polar caps with outward orientation."""
    lateral, lower_cap, upper_cap = np.asarray(lateral, int), np.asarray(lower_cap, int), np.asarray(upper_cap, int)
    lateral_next = np.roll(lateral, -1, axis=0)
    side = np.stack(
        [
            np.stack([lateral[:, :-1], lateral_next[:, :-1], lateral_next[:, 1:]], axis=-1),
            np.stack([lateral[:, :-1], lateral_next[:, 1:], lateral[:, 1:]], axis=-1),
        ],
        axis=2,
    ).reshape(-1, 3)

    def cap_triangles(mapping: np.ndarray, *, upper: bool) -> np.ndarray:
        rings = mapping[1:]
        rings_next = np.roll(rings, -1, axis=1)
        fan = np.stack([np.full(rings.shape[1], mapping[0, 0], dtype=int), rings[0], rings_next[0]], axis=-1)
        quads = np.stack(
            [
                np.stack([rings[:-1], rings[1:], rings_next[1:]], axis=-1),
                np.stack([rings[:-1], rings_next[1:], rings_next[:-1]], axis=-1),
            ],
            axis=2,
        ).reshape(-1, 3)
        tris = np.concatenate([fan, quads], axis=0)
        return tris if upper else tris[:, ::-1]

    return np.concatenate([side, cap_triangles(lower_cap, upper=False), cap_triangles(upper_cap, upper=True)], axis=0)


@lru_cache(maxsize=None)
def _unit_gauss_legendre(order: int) -> tuple[np.ndarray, np.ndarray]:
    order = int(order)
    if order < 1:
        raise ValueError("quadrature order must be positive")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return 0.5 * (nodes + 1.0), 0.5 * weights


def _side_basis_flat(
    triangle_indices: Array, *, ntheta: int, nxi: int, order: int, axisymmetric: bool, dtype: Any
) -> tuple[Array, Array]:
    indices = jnp.asarray(triangle_indices)
    nodes, _ = _unit_gauss_legendre(order)
    u = jnp.asarray(nodes, dtype=dtype)[None, :, None]
    v = jnp.asarray(nodes, dtype=dtype)[None, None, :]
    shape = (indices.shape[0], order, order)
    bary = jnp.stack(
        [jnp.broadcast_to(1.0 - u, shape), jnp.broadcast_to(u * (1.0 - v), shape), jnp.broadcast_to(u * v, shape)],
        axis=-1,
    )
    axial_nodes = jnp.cos(jnp.pi * jnp.arange(nxi, dtype=dtype) / (nxi - 1))[::-1]
    axial = jnp.sum(bary * axial_nodes[indices % nxi][:, None, None, :], axis=-1)
    q2 = order * order
    axial_w = _cgl_interpolation_weights(axial, nxi).reshape(-1, q2, nxi)
    if axisymmetric:
        return axial_w, jnp.arange(nxi, dtype=jnp.int32)
    angular = 2.0 * jnp.pi * (indices // nxi) / ntheta
    anchor = angular[:, :1]
    angular = anchor + (angular - anchor + jnp.pi) % (2.0 * np.pi) - jnp.pi
    theta = jnp.sum(bary * angular[:, None, None, :], axis=-1)
    angular_w = _periodic_interpolation_weights(theta, ntheta).reshape(-1, q2, ntheta)
    combined = (angular_w[:, :, :, None] * axial_w[:, :, None, :]).reshape(-1, q2, ntheta * nxi)
    return combined, jnp.arange(ntheta * nxi, dtype=jnp.int32)


def _spectral_side_density_samples(
    triangle_indices: Array, lateral_values: Array, *, ntheta: int, nxi: int, order: int, axisymmetric: bool
) -> Array:
    """Interpolate lateral nodal data spectrally within side triangles."""
    basis, _ = _side_basis_flat(
        triangle_indices, ntheta=ntheta, nxi=nxi, order=order, axisymmetric=axisymmetric, dtype=lateral_values.dtype
    )
    scalar_input = lateral_values.ndim == 1
    values = lateral_values.reshape((-1, ntheta, nxi))
    flat_values = values[:, 0] if axisymmetric else values.reshape((-1, ntheta * nxi))
    samples = jnp.einsum("tqk,ak->atq", basis, flat_values).reshape((-1, basis.shape[0], order, order))
    return samples[0] if scalar_input else samples


@lru_cache(maxsize=64)
def _connectivity_bytes(connectivity: tuple[tuple[int, int, int], ...]) -> tuple[bytes, int]:
    arr = np.asarray(connectivity, dtype=np.int32)
    return arr.tobytes(), int(arr.shape[0])


@lru_cache(maxsize=64)
def _incident_plan(
    triangles_bytes: bytes, ntriangles: int, targets: tuple[int, ...], side_count: int
) -> tuple[np.ndarray, ...]:
    triangles_np = np.frombuffer(triangles_bytes, dtype=np.int32).reshape(ntriangles, 3)
    targets_np = np.asarray(targets, dtype=np.int32)
    vertex_to_tpos: dict[int, list[int]] = {}
    for pos, vid in enumerate(targets_np.tolist()):
        vertex_to_tpos.setdefault(vid, []).append(pos)

    inc_tpos, inc_tri, inc_rolled = [], [], []
    incident_mask = np.zeros((targets_np.size, ntriangles), dtype=bool)
    for tri_idx in range(ntriangles):
        row = triangles_np[tri_idx]
        for local_pos in range(3):
            for tpos in vertex_to_tpos.get(int(row[local_pos]), ()):
                if not incident_mask[tpos, tri_idx]:
                    incident_mask[tpos, tri_idx] = True
                    inc_tpos.append(tpos)
                    inc_tri.append(tri_idx)
                    inc_rolled.append(np.roll(row, -local_pos))

    inc_tpos_arr, inc_tri_arr = np.asarray(inc_tpos, dtype=np.int32), np.asarray(inc_tri, dtype=np.int32)
    inc_rolled_arr = np.asarray(inc_rolled, dtype=np.int32).reshape(-1, 3)
    side_mask = inc_tri_arr < side_count
    side_pos, lin_pos = np.flatnonzero(side_mask).astype(np.int32), np.flatnonzero(~side_mask).astype(np.int32)
    return triangles_np, targets_np, incident_mask, inc_tpos_arr, inc_rolled_arr, side_pos, inc_rolled_arr[side_mask], lin_pos


def _panel_quadrature(vertices: Array, order: int) -> tuple[Array, Array, Array, Array]:
    nodes, weights = _unit_gauss_legendre(order)
    dtype = vertices.dtype
    u = jnp.asarray(nodes, dtype=dtype)[None, :, None]
    v = jnp.asarray(nodes, dtype=dtype)[None, None, :]
    w2d = jnp.asarray(weights, dtype=dtype)[None, :, None] * jnp.asarray(weights, dtype=dtype)[None, None, :]
    edge1, edge2 = vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]
    source = vertices[:, 0, :][:, None, None, :] + u[..., None] * (
        (1.0 - v)[..., None] * edge1[:, None, None, :] + v[..., None] * edge2[:, None, None, :]
    )
    area_vectors = jnp.cross(edge1, edge2)
    area_scale = jnp.linalg.norm(area_vectors, axis=-1)
    normals = area_vectors / area_scale[:, None]
    scale = w2d * (area_scale[:, None, None] * u) / (4.0 * jnp.pi)
    bary_flat = jnp.stack(
        [
            jnp.broadcast_to(1.0 - u, (1, order, order))[0],
            jnp.broadcast_to(u * (1.0 - v), (1, order, order))[0],
            jnp.broadcast_to(u * v, (1, order, order))[0],
        ],
        axis=-1,
    ).reshape(order * order, 3)
    return source.reshape(-1, order * order, 3), normals, scale.reshape(-1, order * order), bary_flat


_GREEN_STATIC = (
    "order", "lateral_shape", "spectral_side_density", "axisymmetric_side",
    "n_reduced", "ntargets", "side_count", "assemble_matrix",
)


@partial(jax.jit, static_argnames=_GREEN_STATIC)
def _green_operator_kernel(
    xyz: Array,
    dirichlet_full: Array,
    neumann_full: Array,
    collocation_to_reduced: Array,
    plan: tuple[Array, ...],
    *,
    order: int,
    lateral_shape: tuple[int, int] | None,
    spectral_side_density: bool,
    axisymmetric_side: bool,
    n_reduced: int,
    ntargets: int,
    side_count: int,
    assemble_matrix: bool,
) -> tuple[Array, Array | None]:
    triangles, target_idx, incident_mask, inc_tpos, inc_rolled, inc_side_pos, inc_side_rolled, inc_lin_pos = plan
    dtype = xyz.dtype
    source, normals, scale, bary_flat = _panel_quadrature(xyz[triangles], order)
    base_side = inc_side = None
    if spectral_side_density and lateral_shape is not None and side_count > 0:
        ntheta, nxi = lateral_shape
        base_side = _side_basis_flat(
            triangles[:side_count], ntheta=ntheta, nxi=nxi, order=order, axisymmetric=axisymmetric_side, dtype=dtype
        )
        inc_side = _side_basis_flat(
            inc_side_rolled, ntheta=ntheta, nxi=nxi, order=order, axisymmetric=axisymmetric_side, dtype=dtype
        )

    def interp_panels(
        vals: Array, tris: Array, side_data: tuple[Array, Array] | None, pos: Array | None = None
    ) -> Array:
        out = vals[tris] @ bary_flat.T
        if side_data is None:
            return out
        spec = jnp.einsum("tqk,k->tq", side_data[0], vals[side_data[1]])
        return out.at[:side_count].set(spec) if pos is None else out.at[pos].set(spec)

    w_single_n = scale * interp_panels(neumann_full, triangles, base_side)
    d_tri = jnp.zeros_like(w_single_n) if assemble_matrix else interp_panels(dirichlet_full, triangles, base_side)
    lin_tri_red = collocation_to_reduced[triangles[side_count:]]

    def per_target_far(args: tuple[Array, Array, Array]) -> Any:
        t_xyz, d_targ, inc_row = args
        disp = t_xyz[None, None, :] - source
        r2 = jnp.sum(disp * disp, axis=-1)
        inv_r = jnp.where(inc_row[:, None], 0.0, jax.lax.rsqrt(jnp.where(inc_row[:, None], 1.0, r2)))
        dbl = -scale * jnp.einsum("ti,tqi->tq", normals, disp) * (inv_r * inv_r * inv_r)
        if not assemble_matrix:
            return jnp.sum(w_single_n * inv_r + dbl * (d_tri - d_targ))
        lin_w = dbl[side_count:] @ bary_flat
        row_vec = jnp.zeros(n_reduced, dtype=dtype)
        for v_local in range(3):
            row_vec = row_vec.at[lin_tri_red[:, v_local]].add(lin_w[:, v_local])
        if base_side is not None:
            row_vec = row_vec.at[collocation_to_reduced[base_side[1]]].add(
                jnp.einsum("tq,tqk->k", dbl[:side_count], base_side[0])
            )
        return -jnp.sum(w_single_n * inv_r), jnp.sum(dbl), row_vec

    batch_sz = min(ntargets, max(1, 2097152 // max(1, triangles.shape[0] * order * order)))
    far_inputs = (xyz[target_idx], dirichlet_full[target_idx], incident_mask)
    far_out = (
        jax.vmap(per_target_far)(far_inputs)
        if batch_sz >= ntargets
        else jax.lax.map(per_target_far, far_inputs, batch_size=batch_sz)
    )

    inc_source, inc_normals, inc_scale, _ = _panel_quadrature(xyz[inc_rolled], order)
    inc_disp = xyz[target_idx[inc_tpos]][:, None, :] - inc_source
    inc_inv_r = jax.lax.rsqrt(jnp.sum(inc_disp * inc_disp, axis=-1))
    inc_sng = inc_scale * inc_inv_r
    inc_dbl = -inc_scale * jnp.einsum("ei,eqi->eq", inc_normals, inc_disp) * (inc_inv_r * inc_inv_r * inc_inv_r)
    inc_n = interp_panels(neumann_full, inc_rolled, inc_side, inc_side_pos)

    if not assemble_matrix:
        inc_d = interp_panels(dirichlet_full, inc_rolled, inc_side, inc_side_pos) - dirichlet_full[target_idx[inc_tpos], None]
        res = far_out + jax.ops.segment_sum(
            jnp.sum(inc_sng * inc_n + inc_dbl * inc_d, axis=1), inc_tpos, num_segments=ntargets
        )
        return res, None

    rhs_far, diag_far, matrix = far_out
    rhs = rhs_far - jax.ops.segment_sum(jnp.sum(inc_sng * inc_n, axis=1), inc_tpos, num_segments=ntargets)
    diag_inc = jax.ops.segment_sum(jnp.sum(inc_dbl, axis=1), inc_tpos, num_segments=ntargets)
    matrix = (matrix + jnp.eye(ntargets, n_reduced, dtype=dtype)).at[
        jnp.arange(ntargets), collocation_to_reduced[target_idx]
    ].add(-(diag_far + diag_inc))

    inc_lin_w = inc_dbl[inc_lin_pos] @ bary_flat
    inc_lin_rows, inc_lin_cols = inc_tpos[inc_lin_pos], collocation_to_reduced[inc_rolled[inc_lin_pos]]
    for v_local in range(3):
        matrix = matrix.at[inc_lin_rows, inc_lin_cols[:, v_local]].add(inc_lin_w[:, v_local])

    if inc_side is not None:
        inc_spec_w = jnp.einsum("eq,eqk->ek", inc_dbl[inc_side_pos], inc_side[0])
        inc_row_sum = jax.ops.segment_sum(inc_spec_w, inc_tpos[inc_side_pos], num_segments=ntargets)
        lat_red = collocation_to_reduced[inc_side[1]]
        matrix = matrix + jax.vmap(lambda row: jnp.zeros(n_reduced, dtype=dtype).at[lat_red].add(row))(inc_row_sum)

    return rhs, matrix


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
    xyz, dirichlet, neumann = jnp.asarray(xyz), jnp.asarray(dirichlet), jnp.asarray(neumann)
    triangles_np = np.ascontiguousarray(np.asarray(triangles, dtype=np.int32))
    nvertices = int(xyz.shape[0])
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape (n, 3)")
    if dirichlet.shape != (nvertices,) or neumann.shape != (nvertices,):
        raise ValueError("dirichlet and neumann must have one value per vertex")

    if target_indices is None:
        targets_tuple = tuple(range(nvertices))
    else:
        targets_arr = np.asarray(target_indices, dtype=int)
        if targets_arr.ndim != 1 or np.any((targets_arr < 0) | (targets_arr >= nvertices)):
            raise ValueError("target_indices must select valid vertices")
        targets_tuple = tuple(int(i) for i in targets_arr.tolist())

    side_count = 0
    if spectral_side_density:
        if lateral_shape is None:
            raise ValueError("lateral_shape is required for spectral side density")
        lateral_shape = (int(lateral_shape[0]), int(lateral_shape[1]))
        side_count = 2 * lateral_shape[0] * (lateral_shape[1] - 1)

    plan = tuple(
        jnp.asarray(a)
        for a in _incident_plan(triangles_np.tobytes(), int(triangles_np.shape[0]), targets_tuple, side_count)
    )
    residual, _ = _green_operator_kernel(
        xyz, dirichlet, neumann, jnp.arange(nvertices, dtype=jnp.int32), plan,
        order=int(order), lateral_shape=lateral_shape, spectral_side_density=bool(spectral_side_density),
        axisymmetric_side=bool(axisymmetric_side), n_reduced=nvertices, ntargets=len(targets_tuple),
        side_count=side_count, assemble_matrix=False,
    )
    return residual


def _external_field_xyz(source: Any, points_xyz: Array) -> Array:
    """Evaluate an MGRID field or vectorized ``xyz -> B`` callable."""
    if hasattr(source, "b_cyl"):
        x, y, z = jnp.moveaxis(points_xyz, -1, 0)
        off_axis = x**2 + y**2 > 0.0
        radius = jnp.where(off_axis, jnp.sqrt(jnp.where(off_axis, x**2 + y**2, 1.0)), 0.0)
        phi = jnp.arctan2(jnp.where(off_axis, y, 0.0), jnp.where(off_axis, x, 1.0))
        b_r, b_phi, b_z = source.b_cyl(radius, phi, z)
        cosine, sine = jnp.cos(phi), jnp.sin(phi)
        return jnp.stack((b_r * cosine - b_phi * sine, b_r * sine + b_phi * cosine, b_z), axis=-1)
    if callable(source):
        field = jnp.asarray(source(points_xyz))
        if field.shape != points_xyz.shape:
            raise ValueError(f"external field returned shape {field.shape}; expected {points_xyz.shape}")
        return field
    raise TypeError("external field must provide b_cyl or be a vectorized xyz -> B callable")


@dataclass(frozen=True)
class LaplaceNeumannResult:
    """Reduced boundary potential and diagnostics for a Neumann solve."""

    boundary_potential: Array
    residual: Array
    compatibility_error: Array
    raw_compatibility_error: Array
    condition_number: Array
    gauge_error: Array


@dataclass(frozen=True)
class ExteriorVacuum:
    """Solved free-space vacuum field on a mirror boundary."""

    surface: ClosedMirrorSurface
    neumann: Array
    neumann_result: LaplaceNeumannResult
    lateral_field_xyz: Array
    lateral_b_normal: Array


jax.tree_util.register_dataclass(
    LaplaceNeumannResult, data_fields=[f.name for f in fields(LaplaceNeumannResult)], meta_fields=[]
)
jax.tree_util.register_dataclass(ExteriorVacuum, data_fields=[f.name for f in fields(ExteriorVacuum)], meta_fields=[])


def axisymmetric_plasma_external_neumann(
    surface: ClosedMirrorSurface, plasma_field: "ContravariantField", plasma_grid: "MirrorGrid", external_field: Any
) -> Array:
    """Build axisymmetric closed-surface Neumann data without redundant theta."""
    if plasma_grid.ntheta != 1:
        raise ValueError("axisymmetric Neumann data requires ntheta=1")
    expected_size = plasma_grid.nxi + 2 * (plasma_grid.ns - 1)
    if surface.reduced_size != expected_size:
        raise ValueError(f"surface reduced size {surface.reduced_size} must be {expected_size}")
    external_normal = jnp.sum(
        _external_field_xyz(external_field, surface.collocation_xyz) * surface.collocation_normals, axis=1
    )
    neumann = -surface.reduce_collocation_values(external_normal)
    points = surface.collocation_xyz[jnp.asarray(surface.reduced_representatives)]

    nxi, cap_size = plasma_grid.nxi, plasma_grid.ns - 1
    lower, upper = slice(nxi, nxi + cap_size), slice(nxi + cap_size, nxi + 2 * cap_size)
    lower_s = jnp.sum(points[lower, :2] ** 2, axis=1) / jnp.sum(surface.lateral_xyz[0, 0, :2] ** 2)
    upper_s = jnp.sum(points[upper, :2] ** 2, axis=1) / jnp.sum(surface.lateral_xyz[0, -1, :2] ** 2)
    s_grid, dz_dxi = jnp.asarray(plasma_grid.s), float(plasma_grid.dz_dxi)
    lower_bz = jnp.interp(lower_s, s_grid, plasma_field.b_sup_xi[:, 0, 0] * dz_dxi)
    upper_bz = jnp.interp(upper_s, s_grid, plasma_field.b_sup_xi[:, 0, -1] * dz_dxi)
    return neumann.at[lower].add(-lower_bz).at[upper].add(upper_bz)


def axisymmetric_exterior_lateral_field(
    surface: ClosedMirrorSurface,
    boundary_potential: Array,
    neumann: Array,
    plasma_grid: "MirrorGrid",
    external_xyz: Array,
) -> Array:
    """Reconstruct total Cartesian field on the axisymmetric lateral boundary.

    The solved Neumann data supplies the correction normal component and the
    CGL derivative of boundary potential supplies its tangential component.
    Coordinates are returned at theta zero, one value per axial node.
    """
    boundary_potential = jnp.asarray(boundary_potential)
    neumann, external_xyz = jnp.asarray(neumann), jnp.asarray(external_xyz)
    expected = (surface.reduced_size,)
    if boundary_potential.shape != expected or neumann.shape != expected:
        raise ValueError(f"potential and neumann must have shape {expected}")
    if external_xyz.shape != (plasma_grid.nxi, 3):
        raise ValueError(f"external_xyz must have shape ({plasma_grid.nxi}, 3)")

    radius_xi = plasma_grid.axial_basis.differentiate(jnp.linalg.norm(surface.lateral_xyz[0, :, :2], axis=1))
    tangent = jnp.stack([radius_xi, jnp.zeros_like(radius_xi), jnp.full_like(radius_xi, plasma_grid.dz_dxi)], axis=1)
    arc_xi = jnp.linalg.norm(tangent, axis=1)
    tangent_hat = tangent / arc_xi[:, None]
    normal_hat = (
        jnp.stack([jnp.full_like(radius_xi, plasma_grid.dz_dxi), jnp.zeros_like(radius_xi), -radius_xi], axis=1)
        / arc_xi[:, None]
    )
    potential_xi = plasma_grid.axial_basis.differentiate(boundary_potential[: plasma_grid.nxi])
    return external_xyz + neumann[: plasma_grid.nxi, None] * normal_hat + (potential_xi / arc_xi)[:, None] * tangent_hat


def solve_axisymmetric_exterior_vacuum(
    boundary: "MirrorBoundary",
    plasma_field: "ContravariantField",
    plasma_grid: "MirrorGrid",
    external_field: Any,
    *,
    axisymmetric_ntheta: int = 40,
    cap_rim_grade: float = 3.5,
    order: int = 8,
    spectral_side_density: bool = False,
) -> ExteriorVacuum:
    """Solve the unbounded vacuum field and reconstruct its lateral trace.

    The two end cuts are closed geometrically by graded disks. Their Neumann
    data continue the plasma axial field into free space, while the lateral
    data cancel the supplied external normal field. The caps are not material
    interfaces. The returned trace is sampled at theta zero on the plasma
    grid's axial nodes.
    """
    surface = build_closed_mirror_surface(
        boundary, plasma_grid, axisymmetric_ntheta=axisymmetric_ntheta, cap_rim_grade=cap_rim_grade
    )
    neumann = axisymmetric_plasma_external_neumann(surface, plasma_field, plasma_grid, external_field)
    result = solve_reduced_exterior_laplace_neumann(
        surface, neumann, order=order, spectral_side_density=spectral_side_density
    )
    external = _external_field_xyz(external_field, surface.lateral_xyz[0])
    radius_xi = plasma_grid.axial_basis.differentiate(jnp.linalg.norm(surface.lateral_xyz[0, :, :2], axis=1))
    normal = jnp.stack([jnp.full_like(radius_xi, plasma_grid.dz_dxi), jnp.zeros_like(radius_xi), -radius_xi], axis=1)
    normal /= jnp.linalg.norm(normal, axis=1)[:, None]
    physical_neumann = neumann.at[: plasma_grid.nxi].set(-jnp.sum(external * normal, axis=1))
    lateral = axisymmetric_exterior_lateral_field(
        surface, result.boundary_potential, physical_neumann, plasma_grid, external
    )
    return ExteriorVacuum(
        surface=surface,
        neumann=neumann,
        neumann_result=result,
        lateral_field_xyz=lateral,
        lateral_b_normal=jnp.sum(lateral * normal, axis=1),
    )


def laplace_reduced_green_boundary_residual(
    surface: ClosedMirrorSurface, dirichlet: Array, neumann: Array, *, order: int = 8, spectral_side_density: bool = False
) -> Array:
    """Evaluate the boundary identity in the surface's symmetry basis."""
    return panel_green_boundary_residual(
        surface.collocation_xyz,
        np.asarray(surface.triangle_connectivity),
        surface.expand_reduced_values(dirichlet),
        surface.expand_reduced_values(neumann),
        order=order,
        target_indices=np.asarray(surface.reduced_representatives),
        lateral_shape=surface.lateral_xyz.shape[:2],
        spectral_side_density=spectral_side_density,
        axisymmetric_side=surface.reduced_size < surface.collocation_xyz.shape[0],
    )


_SOLVE_STATIC = (
    "order", "lateral_shape", "spectral_side_density", "axisymmetric_side",
    "n_reduced", "ntargets", "side_count", "reduced_lateral_size",
)


@partial(jax.jit, static_argnames=_SOLVE_STATIC)
def _solve_reduced_kernel(
    xyz: Array,
    q_weights: Array,
    q_to_coll: Array,
    coll_to_red: Array,
    raw_neumann: Array,
    plan: tuple[Array, ...],
    *,
    order: int,
    lateral_shape: tuple[int, int],
    spectral_side_density: bool,
    axisymmetric_side: bool,
    n_reduced: int,
    ntargets: int,
    side_count: int,
    reduced_lateral_size: int,
) -> LaplaceNeumannResult:
    dtype = xyz.dtype
    red_weights = jnp.zeros(n_reduced, dtype=dtype).at[coll_to_red[q_to_coll]].add(q_weights)
    area = jnp.sum(q_weights)

    def compat_error(vals: Array) -> Array:
        scale_val = area * jnp.maximum(jnp.sqrt(jnp.mean(vals * vals)), jnp.finfo(vals.dtype).tiny)
        return jnp.abs(jnp.sum(red_weights * vals)) / scale_val

    raw_compat = compat_error(raw_neumann)
    cap_weights = red_weights.at[:reduced_lateral_size].set(0.0)
    neumann = raw_neumann.at[reduced_lateral_size:].add(-jnp.sum(red_weights * raw_neumann) / jnp.sum(cap_weights))

    rhs, matrix = _green_operator_kernel(
        xyz, jnp.zeros(xyz.shape[0], dtype=dtype), neumann[coll_to_red], coll_to_red, plan,
        order=order, lateral_shape=lateral_shape, spectral_side_density=spectral_side_density,
        axisymmetric_side=axisymmetric_side, n_reduced=n_reduced, ntargets=ntargets,
        side_count=side_count, assemble_matrix=True,
    )
    assert matrix is not None
    potential = jnp.linalg.solve(matrix, rhs)
    return LaplaceNeumannResult(
        boundary_potential=potential,
        residual=matrix @ potential - rhs,
        compatibility_error=compat_error(neumann),
        raw_compatibility_error=raw_compat,
        condition_number=jnp.linalg.cond(matrix),
        gauge_error=jnp.asarray(0.0, dtype=matrix.dtype),
    )


def solve_reduced_exterior_laplace_neumann(
    surface: ClosedMirrorSurface, neumann: Array, *, order: int = 8, spectral_side_density: bool = False
) -> LaplaceNeumannResult:
    """Solve for the unique harmonic potential decaying in the exterior."""
    neumann = jnp.asarray(neumann)
    expected = (surface.reduced_size,)
    if neumann.shape != expected:
        raise ValueError(f"neumann shape {neumann.shape} must be {expected}")

    ntheta, nxi = int(surface.lateral_xyz.shape[0]), int(surface.lateral_xyz.shape[1])
    full_lateral_size = ntheta * nxi
    reduced_lateral_size = sum(1 for idx in surface.reduced_representatives if idx < full_lateral_size)
    axisymmetric_side = surface.reduced_size < int(surface.collocation_xyz.shape[0])
    side_count = 2 * ntheta * (nxi - 1) if spectral_side_density else 0
    tri_bytes, ntriangles = _connectivity_bytes(surface.triangle_connectivity)
    plan = tuple(
        jnp.asarray(a) for a in _incident_plan(tri_bytes, ntriangles, surface.reduced_representatives, side_count)
    )
    return _solve_reduced_kernel(
        surface.collocation_xyz, surface.quadrature_weights, surface.quadrature_to_collocation,
        surface.collocation_to_reduced, neumann, plan,
        order=int(order), lateral_shape=(ntheta, nxi), spectral_side_density=bool(spectral_side_density),
        axisymmetric_side=bool(axisymmetric_side), n_reduced=surface.reduced_size,
        ntargets=len(surface.reduced_representatives), side_count=side_count,
        reduced_lateral_size=reduced_lateral_size,
    )
