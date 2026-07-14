"""Laplace boundary integrals and reduced Neumann solves for mirrors."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

try:
    from virtual_casing_jax import laplace_dx_u_eval, laplace_fx_u, laplace_fxd_u_eval
except ModuleNotFoundError as _vcj_error:  # optional research acceleration
    if _vcj_error.name != "virtual_casing_jax":
        raise
    _VCJ_ERROR = _vcj_error

    def _missing_virtual_casing(*_args, **_kwargs):
        raise ModuleNotFoundError(
            "The free-space mirror boundary-integral backend requires "
            "virtual_casing_jax. Install its pinned optional dependency to use it."
        ) from _VCJ_ERROR

    laplace_dx_u_eval = _missing_virtual_casing
    laplace_fx_u = _missing_virtual_casing
    laplace_fxd_u_eval = _missing_virtual_casing

from .exterior import ClosedMirrorSurface, build_closed_mirror_surface
from .exterior_mesh import panel_green_boundary_residual, panel_green_gradient_off_surface
from .geometry import magnetic_field_xyz
from .vacuum import _external_field_xyz

Array = Any


@dataclass(frozen=True)
class LaplaceNeumannResult:
    """Reduced boundary potential and diagnostics for a Neumann solve."""

    boundary_potential: Array
    residual: Array
    compatibility_error: Array
    condition_number: Array
    gauge_error: Array


@dataclass(frozen=True)
class AxisymmetricExteriorVacuum:
    """Solved free-space vacuum field on an axisymmetric mirror boundary."""

    surface: ClosedMirrorSurface
    neumann: Array
    neumann_result: LaplaceNeumannResult
    lateral_field_xyz: Array
    lateral_b_normal: Array


@dataclass(frozen=True)
class NonaxisymmetricExteriorVacuum:
    """Solved free-space vacuum field on a theta-dependent mirror boundary."""

    surface: ClosedMirrorSurface
    neumann: Array
    neumann_result: LaplaceNeumannResult
    lateral_field_xyz: Array
    lateral_b_normal: Array


jax.tree_util.register_dataclass(
    LaplaceNeumannResult,
    data_fields=[field.name for field in fields(LaplaceNeumannResult)],
    meta_fields=[],
)
jax.tree_util.register_dataclass(
    NonaxisymmetricExteriorVacuum,
    data_fields=[field.name for field in fields(NonaxisymmetricExteriorVacuum)],
    meta_fields=[],
)
jax.tree_util.register_dataclass(
    AxisymmetricExteriorVacuum,
    data_fields=[field.name for field in fields(AxisymmetricExteriorVacuum)],
    meta_fields=[],
)


def plasma_external_neumann(
    surface: ClosedMirrorSurface,
    plasma_field: "ContravariantField",
    plasma_geometry: "MirrorGeometry",
    plasma_grid: "MirrorGrid",
    external_field: Any,
) -> Array:
    """Build ``(B_plasma-B_external) dot n`` on the Green surface.

    The lateral plasma trace is sampled directly. End-cut ``Bz`` is
    interpolated in ``s=r^2/a_end^2`` onto the graded cap rings. The cap values
    continue physical through-flux; they are not zero-normal plasma boundary
    conditions.
    """

    ntheta, nxi = surface.lateral_xyz.shape[:2]
    if plasma_grid.ntheta == 1:
        raise ValueError("use axisymmetric_plasma_external_neumann for ntheta=1")
    if ntheta != plasma_grid.ntheta or nxi != plasma_grid.nxi:
        raise ValueError("surface and plasma grid have incompatible lateral nodes")
    external_normal = jnp.sum(
        _external_field_xyz(external_field, surface.collocation_xyz)
        * surface.collocation_normals,
        axis=1,
    )
    field_xyz = magnetic_field_xyz(plasma_field, plasma_geometry)
    lateral_count = ntheta * nxi
    lateral_normal = jnp.sum(
        field_xyz[-1].reshape(-1, 3)
        * surface.collocation_normals[:lateral_count],
        axis=1,
    )

    def cap_normal(cap_xyz: Array, endpoint: int, orientation: float) -> Array:
        boundary_radius = jnp.linalg.norm(surface.lateral_xyz[:, endpoint, :2], axis=1)
        cap_s = jnp.sum(cap_xyz[..., :2] ** 2, axis=-1) / boundary_radius[None, :] ** 2
        source_bz = field_xyz[:, :, endpoint, 2]
        interpolated = jnp.stack(
            [
                jnp.interp(cap_s[:, index], jnp.asarray(plasma_grid.s), source_bz[:, index])
                for index in range(ntheta)
            ],
            axis=1,
        )
        return orientation * jnp.concatenate(
            [jnp.mean(interpolated[0])[None], interpolated[1:-1].reshape(-1)]
        )

    plasma_normal = jnp.concatenate(
        [
            lateral_normal,
            cap_normal(surface.lower_cap_xyz, 0, -1.0),
            cap_normal(surface.upper_cap_xyz, -1, 1.0),
        ]
    )
    return surface.reduce_collocation_values(plasma_normal - external_normal)


def axisymmetric_plasma_external_neumann(
    surface: ClosedMirrorSurface,
    plasma_field: "ContravariantField",
    plasma_grid: "MirrorGrid",
    external_field: Any,
) -> Array:
    """Build axisymmetric closed-surface Neumann data without redundant theta."""

    if plasma_grid.ntheta != 1:
        raise ValueError("axisymmetric Neumann data requires ntheta=1")
    expected_size = plasma_grid.nxi + 2 * (plasma_grid.ns - 1)
    if surface.reduced_size != expected_size:
        raise ValueError(
            f"surface reduced size {surface.reduced_size} must be {expected_size}"
        )
    external_normal = jnp.sum(
        _external_field_xyz(external_field, surface.collocation_xyz)
        * surface.collocation_normals,
        axis=1,
    )
    neumann = -surface.reduce_collocation_values(external_normal)
    points = surface.collocation_xyz[jnp.asarray(surface.reduced_representatives)]

    nxi = plasma_grid.nxi
    cap_size = plasma_grid.ns - 1
    lower = slice(nxi, nxi + cap_size)
    upper = slice(nxi + cap_size, nxi + 2 * cap_size)
    lower_s = jnp.sum(points[lower, :2] ** 2, axis=1) / jnp.sum(
        surface.lateral_xyz[0, 0, :2] ** 2
    )
    upper_s = jnp.sum(points[upper, :2] ** 2, axis=1) / jnp.sum(
        surface.lateral_xyz[0, -1, :2] ** 2
    )
    lower_bz = jnp.interp(
        lower_s,
        jnp.asarray(plasma_grid.s),
        plasma_field.b_sup_xi[:, 0, 0] * float(plasma_grid.dz_dxi),
    )
    upper_bz = jnp.interp(
        upper_s,
        jnp.asarray(plasma_grid.s),
        plasma_field.b_sup_xi[:, 0, -1] * float(plasma_grid.dz_dxi),
    )
    neumann = neumann.at[lower].add(-lower_bz)
    return neumann.at[upper].add(upper_bz)


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
    neumann = jnp.asarray(neumann)
    external_xyz = jnp.asarray(external_xyz)
    expected = (surface.reduced_size,)
    if boundary_potential.shape != expected or neumann.shape != expected:
        raise ValueError(f"potential and neumann must have shape {expected}")
    if external_xyz.shape != (plasma_grid.nxi, 3):
        raise ValueError(f"external_xyz must have shape ({plasma_grid.nxi}, 3)")

    radius = jnp.linalg.norm(surface.lateral_xyz[0, :, :2], axis=1)
    radius_xi = plasma_grid.axial_basis.differentiate(radius)
    tangent = jnp.stack(
        [radius_xi, jnp.zeros_like(radius_xi), jnp.full_like(radius_xi, plasma_grid.dz_dxi)],
        axis=1,
    )
    arc_xi = jnp.linalg.norm(tangent, axis=1)
    tangent_hat = tangent / arc_xi[:, None]
    normal_hat = jnp.stack(
        [
            jnp.full_like(radius_xi, plasma_grid.dz_dxi),
            jnp.zeros_like(radius_xi),
            -radius_xi,
        ],
        axis=1,
    ) / arc_xi[:, None]
    potential_xi = plasma_grid.axial_basis.differentiate(
        boundary_potential[: plasma_grid.nxi]
    )
    correction = (
        neumann[: plasma_grid.nxi, None] * normal_hat
        + (potential_xi / arc_xi)[:, None] * tangent_hat
    )
    return external_xyz + correction


def nonaxisymmetric_exterior_lateral_field(
    surface: ClosedMirrorSurface,
    boundary_potential: Array,
    neumann: Array,
    plasma_grid: "MirrorGrid",
    external_xyz: Array,
) -> Array:
    """Reconstruct total field on a theta-dependent lateral boundary."""

    if plasma_grid.ntheta == 1:
        raise ValueError("use axisymmetric_exterior_lateral_field for ntheta=1")
    ntheta, nxi = plasma_grid.ntheta, plasma_grid.nxi
    expected = (surface.reduced_size,)
    boundary_potential = jnp.asarray(boundary_potential)
    neumann = jnp.asarray(neumann)
    external_xyz = jnp.asarray(external_xyz)
    if boundary_potential.shape != expected or neumann.shape != expected:
        raise ValueError(f"potential and neumann must have shape {expected}")
    if external_xyz.shape != (ntheta, nxi, 3):
        raise ValueError(f"external_xyz must have shape ({ntheta}, {nxi}, 3)")

    lateral_size = ntheta * nxi
    potential = boundary_potential[:lateral_size].reshape(ntheta, nxi)
    potential_theta = plasma_grid.theta_basis.differentiate(potential, axis=0)
    potential_xi = plasma_grid.axial_basis.differentiate(potential, axis=1)
    radius = jnp.linalg.norm(surface.lateral_xyz[..., :2], axis=-1)
    radius_theta = plasma_grid.theta_basis.differentiate(radius, axis=0)
    radius_xi = plasma_grid.axial_basis.differentiate(radius, axis=1)
    theta = jnp.asarray(plasma_grid.theta)[:, None]
    cosine, sine = jnp.cos(theta), jnp.sin(theta)
    zeros = jnp.zeros_like(radius)
    e_theta = jnp.stack(
        [
            radius_theta * cosine - radius * sine,
            radius_theta * sine + radius * cosine,
            zeros,
        ],
        axis=-1,
    )
    e_xi = jnp.stack(
        [
            radius_xi * cosine,
            radius_xi * sine,
            jnp.full_like(radius, plasma_grid.dz_dxi),
        ],
        axis=-1,
    )
    gtt = jnp.sum(e_theta**2, axis=-1)
    gtx = jnp.sum(e_theta * e_xi, axis=-1)
    gxx = jnp.sum(e_xi**2, axis=-1)
    determinant = gtt * gxx - gtx**2
    coefficient_theta = (gxx * potential_theta - gtx * potential_xi) / determinant
    coefficient_xi = (gtt * potential_xi - gtx * potential_theta) / determinant
    tangential = (
        coefficient_theta[..., None] * e_theta
        + coefficient_xi[..., None] * e_xi
    )
    normals = surface.collocation_normals[:lateral_size].reshape(ntheta, nxi, 3)
    correction = (
        neumann[:lateral_size].reshape(ntheta, nxi)[..., None] * normals
        + tangential
    )
    return external_xyz + correction


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
    spectral_cap_density: bool = False,
    curved_side_geometry: bool = False,
) -> AxisymmetricExteriorVacuum:
    """Solve the unbounded vacuum field and reconstruct its lateral trace.

    The two end cuts are closed geometrically by graded disks. Their Neumann
    data continue the plasma axial field into free space, while the lateral
    data cancel the supplied external normal field. The caps are not material
    interfaces. The returned trace is sampled at theta zero on the plasma
    grid's axial nodes.
    """

    surface = build_closed_mirror_surface(
        boundary,
        plasma_grid,
        axisymmetric_ntheta=axisymmetric_ntheta,
        cap_rim_grade=cap_rim_grade,
    )
    neumann = axisymmetric_plasma_external_neumann(
        surface, plasma_field, plasma_grid, external_field
    )
    result = solve_reduced_exterior_laplace_neumann(
        surface,
        neumann,
        order=order,
        spectral_side_density=spectral_side_density,
        spectral_cap_density=spectral_cap_density,
        curved_side_geometry=curved_side_geometry,
    )
    external = _external_field_xyz(external_field, surface.lateral_xyz[0])
    lateral = axisymmetric_exterior_lateral_field(
        surface,
        result.boundary_potential,
        neumann,
        plasma_grid,
        external,
    )
    radius = jnp.linalg.norm(surface.lateral_xyz[0, :, :2], axis=1)
    radius_xi = plasma_grid.axial_basis.differentiate(radius)
    normal = jnp.stack(
        [
            jnp.full_like(radius_xi, plasma_grid.dz_dxi),
            jnp.zeros_like(radius_xi),
            -radius_xi,
        ],
        axis=1,
    )
    normal /= jnp.linalg.norm(normal, axis=1)[:, None]
    return AxisymmetricExteriorVacuum(
        surface=surface,
        neumann=neumann,
        neumann_result=result,
        lateral_field_xyz=lateral,
        lateral_b_normal=jnp.sum(lateral * normal, axis=1),
    )


def solve_nonaxisymmetric_exterior_vacuum(
    boundary: "MirrorBoundary",
    plasma_field: "ContravariantField",
    plasma_geometry: "MirrorGeometry",
    plasma_grid: "MirrorGrid",
    external_field: Any,
    *,
    cap_rim_grade: float = 3.5,
    order: int = 8,
    spectral_side_density: bool = False,
    spectral_cap_density: bool = False,
    curved_side_geometry: bool = False,
) -> NonaxisymmetricExteriorVacuum:
    """Solve and reconstruct the unbounded theta-dependent vacuum field."""

    if plasma_grid.ntheta == 1:
        raise ValueError("nonaxisymmetric exterior vacuum requires ntheta > 1")
    surface = build_closed_mirror_surface(
        boundary,
        plasma_grid,
        cap_rim_grade=cap_rim_grade,
    )
    neumann = plasma_external_neumann(
        surface, plasma_field, plasma_geometry, plasma_grid, external_field
    )
    result = solve_reduced_exterior_laplace_neumann(
        surface,
        neumann,
        order=order,
        spectral_side_density=spectral_side_density,
        spectral_cap_density=spectral_cap_density,
        curved_side_geometry=curved_side_geometry,
    )
    external = _external_field_xyz(external_field, surface.lateral_xyz)
    lateral = nonaxisymmetric_exterior_lateral_field(
        surface,
        result.boundary_potential,
        neumann,
        plasma_grid,
        external,
    )
    lateral_size = plasma_grid.ntheta * plasma_grid.nxi
    normals = surface.collocation_normals[:lateral_size].reshape(
        plasma_grid.ntheta, plasma_grid.nxi, 3
    )
    return NonaxisymmetricExteriorVacuum(
        surface=surface,
        neumann=neumann,
        neumann_result=result,
        lateral_field_xyz=lateral,
        lateral_b_normal=jnp.sum(lateral * normals, axis=-1),
    )


def laplace_double_layer_off_surface(
    surface: ClosedMirrorSurface,
    density: Array,
    targets: Array,
    *,
    chunk_size: int = 1024,
) -> Array:
    """Evaluate a Laplace double layer away from the source surface."""

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
    """Evaluate Green's representation inside and outside the surface."""

    return laplace_single_layer_off_surface(surface, neumann, targets) + (
        laplace_double_layer_off_surface(surface, dirichlet, targets)
    )


def laplace_green_gradient_off_surface(
    surface: ClosedMirrorSurface,
    dirichlet: Array,
    neumann: Array,
    targets: Array,
) -> Array:
    """Evaluate the analytic gradient of Green's representation."""

    dirichlet, targets = _validate_off_surface_inputs(surface, dirichlet, targets)
    neumann = jnp.asarray(neumann)
    if neumann.shape != dirichlet.shape:
        raise ValueError(f"neumann shape {neumann.shape} must be {dirichlet.shape}")
    single_gradient = laplace_single_layer_gradient_off_surface(
        surface, neumann, targets
    )
    displacement = targets[:, None, :] - surface.xyz[None, :, :]
    radius_squared = jnp.sum(displacement**2, axis=-1)
    inverse_radius = jax.lax.rsqrt(radius_squared)
    inverse_radius3 = inverse_radius**3
    normal_displacement = jnp.einsum(
        "si,tsi->ts", surface.normals, displacement
    )
    weighted_dirichlet = dirichlet * surface.quadrature_weights
    double_gradient = (
        -surface.normals[None, :, :] * inverse_radius3[..., None]
        + 3.0
        * normal_displacement[..., None]
        * displacement
        * (inverse_radius3 / radius_squared)[..., None]
    ) * weighted_dirichlet[None, :, None] / (4.0 * jnp.pi)
    return single_gradient + jnp.sum(double_gradient, axis=1)


def laplace_reduced_green_gradient_off_surface(
    surface: ClosedMirrorSurface,
    dirichlet: Array,
    neumann: Array,
    targets: Array,
    *,
    order: int = 8,
    spectral_side_density: bool = False,
    spectral_cap_density: bool = False,
    curved_side_geometry: bool = False,
) -> Array:
    """Evaluate a reduced solution with Duffy panel quadrature."""

    return panel_green_gradient_off_surface(
        surface.collocation_xyz,
        np.asarray(surface.triangle_connectivity),
        surface.expand_reduced_values(dirichlet),
        surface.expand_reduced_values(neumann),
        targets,
        order=order,
        lateral_shape=surface.lateral_xyz.shape[:2],
        lateral_xyz=surface.lateral_xyz,
        lower_cap_xyz=surface.lower_cap_xyz,
        upper_cap_xyz=surface.upper_cap_xyz,
        spectral_side_density=spectral_side_density,
        spectral_cap_density=spectral_cap_density,
        curved_side_geometry=curved_side_geometry,
        axisymmetric_side=surface.reduced_size < surface.collocation_xyz.shape[0],
    )


def laplace_green_boundary_residual(
    surface: ClosedMirrorSurface,
    dirichlet: Array,
    neumann: Array,
    *,
    order: int = 8,
) -> Array:
    """Evaluate the singular Green identity on unique boundary nodes."""

    return panel_green_boundary_residual(
        surface.collocation_xyz,
        surface.triangles,
        dirichlet,
        neumann,
        order=order,
    )


def laplace_reduced_green_boundary_residual(
    surface: ClosedMirrorSurface,
    dirichlet: Array,
    neumann: Array,
    *,
    order: int = 8,
    spectral_side_density: bool = False,
    spectral_cap_density: bool = False,
    curved_side_geometry: bool = False,
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
        lateral_xyz=surface.lateral_xyz,
        lower_cap_xyz=surface.lower_cap_xyz,
        upper_cap_xyz=surface.upper_cap_xyz,
        spectral_side_density=spectral_side_density,
        spectral_cap_density=spectral_cap_density,
        curved_side_geometry=curved_side_geometry,
        axisymmetric_side=surface.reduced_size < surface.collocation_xyz.shape[0],
    )


def laplace_reduced_exterior_boundary_residual(
    surface: ClosedMirrorSurface,
    dirichlet: Array,
    neumann: Array,
    *,
    order: int = 8,
    spectral_side_density: bool = False,
    spectral_cap_density: bool = False,
    curved_side_geometry: bool = False,
) -> Array:
    """Boundary residual for a harmonic potential decaying in the exterior."""

    dirichlet = jnp.asarray(dirichlet)
    return dirichlet + laplace_reduced_green_boundary_residual(
        surface,
        dirichlet,
        neumann,
        order=order,
        spectral_side_density=spectral_side_density,
        spectral_cap_density=spectral_cap_density,
        curved_side_geometry=curved_side_geometry,
    )


def laplace_reduced_exterior_gradient_off_surface(
    surface: ClosedMirrorSurface,
    dirichlet: Array,
    neumann: Array,
    targets: Array,
    *,
    order: int = 8,
    spectral_side_density: bool = False,
    spectral_cap_density: bool = False,
    curved_side_geometry: bool = False,
) -> Array:
    """Gradient of the decaying exterior representation."""

    return -laplace_reduced_green_gradient_off_surface(
        surface,
        dirichlet,
        neumann,
        targets,
        order=order,
        spectral_side_density=spectral_side_density,
        spectral_cap_density=spectral_cap_density,
        curved_side_geometry=curved_side_geometry,
    )


def solve_reduced_interior_laplace_neumann(
    surface: ClosedMirrorSurface,
    neumann: Array,
    *,
    order: int = 8,
    spectral_side_density: bool = False,
    spectral_cap_density: bool = False,
    curved_side_geometry: bool = False,
) -> LaplaceNeumannResult:
    """Solve the interior Neumann problem with a zero-mean gauge."""

    return _solve_reduced_laplace_neumann(
        surface,
        neumann,
        order=order,
        exterior=False,
        spectral_side_density=spectral_side_density,
        spectral_cap_density=spectral_cap_density,
        curved_side_geometry=curved_side_geometry,
    )


def solve_reduced_exterior_laplace_neumann(
    surface: ClosedMirrorSurface,
    neumann: Array,
    *,
    order: int = 8,
    spectral_side_density: bool = False,
    spectral_cap_density: bool = False,
    curved_side_geometry: bool = False,
) -> LaplaceNeumannResult:
    """Solve for the unique harmonic potential decaying in the exterior."""

    return _solve_reduced_laplace_neumann(
        surface,
        neumann,
        order=order,
        exterior=True,
        spectral_side_density=spectral_side_density,
        spectral_cap_density=spectral_cap_density,
        curved_side_geometry=curved_side_geometry,
    )


def _solve_reduced_laplace_neumann(
    surface: ClosedMirrorSurface,
    neumann: Array,
    *,
    order: int,
    exterior: bool,
    spectral_side_density: bool,
    spectral_cap_density: bool,
    curved_side_geometry: bool,
) -> LaplaceNeumannResult:
    """Shared dense differentiable solve for the two Calderon limits."""

    if curved_side_geometry and not spectral_side_density:
        raise ValueError("curved side geometry requires spectral side density")
    neumann = jnp.asarray(neumann)
    expected = (surface.reduced_size,)
    if neumann.shape != expected:
        raise ValueError(f"neumann shape {neumann.shape} must be {expected}")
    zero = jnp.zeros_like(neumann)

    def dirichlet_operator(values: Array) -> Array:
        if exterior:
            return laplace_reduced_exterior_boundary_residual(
                surface,
                values,
                zero,
                order=order,
                spectral_side_density=spectral_side_density,
                spectral_cap_density=spectral_cap_density,
                curved_side_geometry=curved_side_geometry,
            )
        return laplace_reduced_green_boundary_residual(
            surface,
            values,
            zero,
            order=order,
            spectral_side_density=spectral_side_density,
            spectral_cap_density=spectral_cap_density,
            curved_side_geometry=curved_side_geometry,
        )

    matrix = jax.jacfwd(dirichlet_operator)(zero)
    residual_function = (
        laplace_reduced_exterior_boundary_residual
        if exterior
        else laplace_reduced_green_boundary_residual
    )
    right_hand_side = -residual_function(
        surface,
        zero,
        neumann,
        order=order,
        spectral_side_density=spectral_side_density,
        spectral_cap_density=spectral_cap_density,
        curved_side_geometry=curved_side_geometry,
    )
    quadrature_to_reduced = surface.collocation_to_reduced[
        surface.quadrature_to_collocation
    ]
    reduced_weights = jnp.zeros(surface.reduced_size).at[
        quadrature_to_reduced
    ].add(surface.quadrature_weights)
    reduced_weights /= jnp.sum(reduced_weights)
    if exterior:
        solve_matrix = matrix
        potential = jnp.linalg.solve(matrix, right_hand_side)
        gauge_error = jnp.asarray(0.0, dtype=matrix.dtype)
    else:
        solve_matrix = jnp.block(
            [
                [matrix, reduced_weights[:, None]],
                [reduced_weights[None, :], jnp.zeros((1, 1), dtype=matrix.dtype)],
            ]
        )
        solution = jnp.linalg.solve(
            solve_matrix, jnp.concatenate([right_hand_side, jnp.zeros(1)])
        )
        potential = solution[:-1]
        gauge_error = jnp.abs(reduced_weights @ potential)
    residual = matrix @ potential - right_hand_side

    full_neumann = surface.expand_reduced_values(neumann)
    quadrature_neumann = surface.expand_collocation_values(full_neumann)
    net_flux = jnp.sum(quadrature_neumann * surface.quadrature_weights)
    flux_scale = surface.area * jnp.maximum(
        jnp.sqrt(jnp.mean(neumann**2)), jnp.finfo(neumann.dtype).tiny
    )
    return LaplaceNeumannResult(
        boundary_potential=potential,
        residual=residual,
        compatibility_error=jnp.abs(net_flux) / flux_scale,
        condition_number=jnp.linalg.cond(solve_matrix),
        gauge_error=gauge_error,
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
    from .geometry import ContravariantField, MirrorGeometry
    from .model import MirrorBoundary
