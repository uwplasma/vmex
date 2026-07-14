"""Closed mirror surface geometry required by the free-space exterior solve."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

requires_virtual_casing = pytest.mark.skipif(
    importlib.util.find_spec("virtual_casing_jax") is None,
    reason="off-surface Laplace kernels require virtual_casing_jax",
)

from vmec_jax.mirror.exterior_mesh import (  # noqa: E402
    _spectral_side_density_samples,
    _unit_gauss_legendre,
    duffy_triangle_single_layer,
)
from vmec_jax.mirror.exterior_interpolation import (  # noqa: E402
    spectral_cap_density_samples,
)
from vmec_jax.mirror.exterior_cap_panels import (  # noqa: E402
    _unit_rule,
    curved_cap_geometry,
)

from vmec_jax.mirror import (  # noqa: E402
    MirrorBoundary,
    MirrorConfig,
    MirrorResolution,
    MirrorState,
)
from vmec_jax.mirror.exterior import build_closed_mirror_surface  # noqa: E402
from vmec_jax.mirror.exterior_bie import (  # noqa: E402
    axisymmetric_plasma_external_neumann,
    axisymmetric_exterior_lateral_field,
    laplace_double_layer_off_surface,
    laplace_green_boundary_residual,
    laplace_green_gradient_off_surface,
    laplace_green_representation_off_surface,
    laplace_reduced_green_boundary_residual,
    laplace_reduced_exterior_gradient_off_surface,
    laplace_reduced_green_gradient_off_surface,
    laplace_single_layer_gradient_off_surface,
    plasma_external_neumann,
    solve_reduced_exterior_laplace_neumann,
    solve_reduced_interior_laplace_neumann,
    solve_axisymmetric_exterior_vacuum,
    solve_nonaxisymmetric_exterior_vacuum,
)
from vmec_jax.mirror.geometry import (  # noqa: E402
    contravariant_field,
    evaluate_geometry,
    magnetic_field_squared,
    magnetic_field_xyz,
)


def _grid(*, ns: int = 17, mpol: int = 0, ntheta: int = 1, nxi: int = 25):
    return MirrorConfig(
        resolution=MirrorResolution(ns=ns, mpol=mpol, ntheta=ntheta, nxi=nxi),
        z_min=-1.4,
        z_max=1.4,
    ).build_grid()


def _paraxial_field(points, *, center_field: float = 0.08, curvature: float = 0.02):
    """Curl-free, divergence-free external mirror field."""

    points = jnp.asarray(points)
    x, y, z = jnp.moveaxis(points, -1, 0)
    return jnp.stack(
        (
            -curvature * x * z,
            -curvature * y * z,
            center_field + curvature * (z**2 - 0.5 * (x**2 + y**2)),
        ),
        axis=-1,
    )


def _zero_field(points):
    return jnp.zeros_like(points)


def test_closed_cylinder_has_exact_area_volume_and_orientation() -> None:
    grid = _grid()
    radius = 0.37
    surface = build_closed_mirror_surface(MirrorBoundary.from_radius(radius, grid), grid)

    expected_area = 2.0 * np.pi * radius * 2.8 + 2.0 * np.pi * radius**2
    expected_volume = np.pi * radius**2 * 2.8
    np.testing.assert_allclose(surface.area, expected_area, rtol=3.0e-14)
    np.testing.assert_allclose(surface.volume, expected_volume, rtol=3.0e-14)
    np.testing.assert_allclose(jnp.sum(surface.weighted_normals, axis=0), 0.0, atol=3.0e-15)
    assert np.all(np.asarray(surface.lower_cap_weighted_normals[..., 2]) < 0.0)
    assert np.all(np.asarray(surface.upper_cap_weighted_normals[..., 2]) > 0.0)


def test_curved_cap_panels_integrate_exact_circular_disks() -> None:
    grid = _grid(ns=7, nxi=9)
    radius = 0.37
    surface = build_closed_mirror_surface(MirrorBoundary.from_radius(radius, grid), grid, axisymmetric_ntheta=12)
    side_count = 2 * 12 * (grid.nxi - 1)
    cap_count = (surface.triangles.shape[0] - side_count) // 2
    _, weights = _unit_rule(8)
    for upper, cap_xyz, triangles in (
        (False, surface.lower_cap_xyz, surface.triangles[side_count : side_count + cap_count]),
        (True, surface.upper_cap_xyz, surface.triangles[side_count + cap_count :]),
    ):
        _, area_vectors = curved_cap_geometry(triangles, cap_xyz, nxi=grid.nxi, upper=upper, order=8)
        area = jnp.sum(
            jnp.asarray(weights)[None, :, None]
            * jnp.asarray(weights)[None, None, :]
            * jnp.linalg.norm(area_vectors, axis=-1)
        )
        np.testing.assert_allclose(area, np.pi * radius**2, rtol=2.0e-14)
        assert np.all(np.sign(np.asarray(area_vectors[..., 2])) == (1 if upper else -1))


def test_shaped_surface_satisfies_divergence_theorem_moments() -> None:
    grid = _grid(ns=21, mpol=3, ntheta=9, nxi=33)
    theta = jnp.asarray(grid.theta)[:, None]
    xi = jnp.asarray(grid.xi)[None, :]
    radius = 0.31 * (1.0 + 0.08 * xi**2 + 0.04 * jnp.cos(2.0 * theta) * (1.0 - xi**2))
    surface = build_closed_mirror_surface(MirrorBoundary.from_radius(radius, grid), grid)

    net_normal = jnp.sum(surface.weighted_normals, axis=0)
    first_moment = jnp.einsum("ni,nj->ij", surface.xyz, surface.weighted_normals)
    np.testing.assert_allclose(net_normal, 0.0, atol=2.0e-13)
    np.testing.assert_allclose(first_moment, np.eye(3) * float(surface.volume), rtol=3.0e-12, atol=3.0e-13)


def test_caps_share_the_lateral_end_rings() -> None:
    grid = _grid(ns=13, mpol=2, ntheta=7, nxi=17)
    theta = jnp.asarray(grid.theta)[:, None]
    xi = jnp.asarray(grid.xi)[None, :]
    boundary = MirrorBoundary.from_radius(0.3 * (1.0 + 0.05 * jnp.cos(theta) * (1.0 + xi)), grid)
    surface = build_closed_mirror_surface(boundary, grid)

    np.testing.assert_allclose(surface.lower_cap_xyz[-1], surface.lateral_xyz[:, 0])
    np.testing.assert_allclose(surface.upper_cap_xyz[-1], surface.lateral_xyz[:, -1])


def test_collocation_map_identifies_cap_centers_and_rims() -> None:
    grid = _grid(ns=13, mpol=2, ntheta=7, nxi=17)
    theta = jnp.asarray(grid.theta)[:, None]
    xi = jnp.asarray(grid.xi)[None, :]
    surface = build_closed_mirror_surface(
        MirrorBoundary.from_radius(0.3 * (1.0 + 0.04 * jnp.cos(2.0 * theta) * (1.0 - xi**2)), grid),
        grid,
    )
    nquadrature = surface.xyz.shape[0]
    ncollocation = surface.collocation_xyz.shape[0]
    expected = grid.ntheta * grid.nxi + 2 * (1 + (grid.ns - 2) * grid.ntheta)
    assert ncollocation == expected
    assert ncollocation < nquadrature
    assert np.unique(np.asarray(surface.collocation_xyz), axis=0).shape[0] == ncollocation

    values = surface.collocation_xyz[:, 0] + 2.0 * surface.collocation_xyz[:, 1] - 0.5 * surface.collocation_xyz[:, 2]
    expanded = surface.expand_collocation_values(values)
    expected_values = surface.xyz[:, 0] + 2.0 * surface.xyz[:, 1] - 0.5 * surface.xyz[:, 2]
    np.testing.assert_allclose(expanded, expected_values, atol=2.0e-15)


def test_axisymmetric_reduction_preserves_ring_values() -> None:
    grid = _grid(ns=13, nxi=17)
    surface = build_closed_mirror_surface(MirrorBoundary.from_radius(0.3, grid), grid, axisymmetric_ntheta=12)
    assert surface.reduced_size == grid.nxi + 2 * (grid.ns - 1)
    reduced = jnp.linspace(-0.4, 0.7, surface.reduced_size)
    expanded = surface.expand_reduced_values(reduced)
    np.testing.assert_allclose(surface.reduce_collocation_values(expanded), reduced)

    residual = laplace_reduced_green_boundary_residual(
        surface, jnp.ones(surface.reduced_size), jnp.zeros(surface.reduced_size)
    )
    np.testing.assert_allclose(residual, 0.0, atol=2.0e-15)

    full_dirichlet = surface.collocation_xyz[:, 2]
    full_neumann = surface.collocation_normals[:, 2]
    full = laplace_green_boundary_residual(surface, full_dirichlet, full_neumann)
    reduced = laplace_reduced_green_boundary_residual(
        surface,
        surface.reduce_collocation_values(full_dirichlet),
        surface.reduce_collocation_values(full_neumann),
    )
    np.testing.assert_allclose(
        reduced,
        surface.reduce_collocation_values(full),
        rtol=2.0e-12,
        atol=2.0e-13,
    )


def test_cap_grading_clusters_rim_without_changing_cylinder_integrals() -> None:
    grid = _grid(ns=13, nxi=17)
    boundary = MirrorBoundary.from_radius(0.3, grid)
    uniform = build_closed_mirror_surface(boundary, grid, cap_rim_grade=1.0)
    graded = build_closed_mirror_surface(boundary, grid, cap_rim_grade=2.5)
    radii = np.linalg.norm(np.asarray(graded.lower_cap_xyz[:, 0, :2]), axis=1)
    assert radii[-1] - radii[-2] < 0.1 * (radii[1] - radii[0])
    np.testing.assert_allclose(graded.area, uniform.area, rtol=3.0e-14)
    np.testing.assert_allclose(graded.volume, uniform.volume, rtol=3.0e-14)


def test_reduced_neumann_solve_recovers_linear_harmonic() -> None:
    grid = _grid(ns=13, nxi=21)
    surface = build_closed_mirror_surface(
        MirrorBoundary.from_radius(0.37, grid),
        grid,
        axisymmetric_ntheta=24,
        cap_rim_grade=3.5,
    )
    exact = surface.reduce_collocation_values(surface.collocation_xyz[:, 2])
    neumann = surface.reduce_collocation_values(surface.collocation_normals[:, 2])
    result = solve_reduced_interior_laplace_neumann(surface, neumann)
    exact -= jnp.mean(exact)
    recovered = result.boundary_potential - jnp.mean(result.boundary_potential)
    relative_error = jnp.linalg.norm(recovered - exact) / jnp.linalg.norm(exact)

    assert float(relative_error) < 3.0e-4
    assert float(jnp.linalg.norm(result.residual)) < 2.0e-8
    assert float(result.compatibility_error) < 2.0e-14
    assert float(result.condition_number) < 25.0
    assert float(result.gauge_error) < 2.0e-14


def test_reduced_neumann_solve_is_forward_differentiable() -> None:
    grid = _grid(ns=7, nxi=9)
    surface = build_closed_mirror_surface(
        MirrorBoundary.from_radius(0.37, grid),
        grid,
        axisymmetric_ntheta=8,
        cap_rim_grade=3.0,
    )
    neumann = surface.reduce_collocation_values(surface.collocation_normals[:, 2])
    direction = jnp.sin(jnp.arange(surface.reduced_size, dtype=neumann.dtype))
    _, tangent = jax.jvp(
        lambda data: solve_reduced_interior_laplace_neumann(surface, data, order=6).boundary_potential,
        (neumann,),
        (direction,),
    )
    assert np.all(np.isfinite(np.asarray(tangent)))
    assert float(jnp.linalg.norm(tangent)) > 0.0


def test_reduced_exterior_neumann_solve_recovers_decaying_dipole() -> None:
    boundary_errors = []
    field_errors = []
    lateral_errors = []
    for ns, nxi, ntheta in (
        (9, 13, 16),
        (13, 21, 24),
        (17, 29, 32),
        (21, 37, 40),
    ):
        grid = _grid(ns=ns, nxi=nxi)
        surface = build_closed_mirror_surface(
            MirrorBoundary.from_radius(0.37, grid),
            grid,
            axisymmetric_ntheta=ntheta,
            cap_rim_grade=3.5,
        )
        xyz = surface.collocation_xyz
        radius_squared = jnp.sum(xyz**2, axis=1)
        exact_full = xyz[:, 2] / radius_squared**1.5
        gradient_full = jnp.stack(
            [
                -3.0 * xyz[:, 2] * xyz[:, 0] / radius_squared**2.5,
                -3.0 * xyz[:, 2] * xyz[:, 1] / radius_squared**2.5,
                1.0 / radius_squared**1.5 - 3.0 * xyz[:, 2] ** 2 / radius_squared**2.5,
            ],
            axis=1,
        )
        exact = surface.reduce_collocation_values(exact_full)
        neumann = surface.reduce_collocation_values(jnp.sum(gradient_full * surface.collocation_normals, axis=1))
        result = solve_reduced_exterior_laplace_neumann(surface, neumann)
        boundary_errors.append(float(jnp.linalg.norm(result.boundary_potential - exact) / jnp.linalg.norm(exact)))
        gradient = laplace_reduced_exterior_gradient_off_surface(
            surface,
            result.boundary_potential,
            neumann,
            jnp.asarray([[0.0, 0.0, 2.0]]),
        )
        field_errors.append(float(jnp.abs(gradient[0, 2] + 0.25) / 0.25))
        lateral = axisymmetric_exterior_lateral_field(
            surface,
            result.boundary_potential,
            neumann,
            grid,
            jnp.zeros((grid.nxi, 3)),
        )
        lateral_xyz = surface.lateral_xyz[0]
        lateral_radius_squared = jnp.sum(lateral_xyz**2, axis=1)
        exact_lateral = jnp.stack(
            [
                -3.0 * lateral_xyz[:, 2] * lateral_xyz[:, 0] / lateral_radius_squared**2.5,
                jnp.zeros(grid.nxi),
                1.0 / lateral_radius_squared**1.5 - 3.0 * lateral_xyz[:, 2] ** 2 / lateral_radius_squared**2.5,
            ],
            axis=1,
        )
        lateral_error = jnp.linalg.norm(lateral - exact_lateral) / jnp.linalg.norm(exact_lateral)
        lateral_errors.append(float(lateral_error))
        np.testing.assert_allclose(gradient[:, :2], 0.0, atol=2.0e-14)
        assert float(result.compatibility_error) < 2.0e-14
        assert float(result.condition_number) < 5.0
        assert float(jnp.linalg.norm(result.residual)) < 3.0e-14

    assert all(fine < coarse for coarse, fine in zip(boundary_errors[:-1], boundary_errors[1:], strict=True))
    assert boundary_errors[-1] < 3.5e-2
    assert all(fine < coarse for coarse, fine in zip(field_errors[:-1], field_errors[1:], strict=True))
    assert field_errors[-1] < 8.0e-3
    assert all(fine < coarse for coarse, fine in zip(lateral_errors[:-1], lateral_errors[1:], strict=True))
    assert lateral_errors[-1] < 4.0e-2


def test_spectral_side_density_improves_exterior_dipole() -> None:
    grid = _grid(ns=13, nxi=21)
    surface = build_closed_mirror_surface(
        MirrorBoundary.from_radius(0.37, grid),
        grid,
        axisymmetric_ntheta=24,
        cap_rim_grade=3.5,
    )
    xyz = surface.collocation_xyz
    radius_squared = jnp.sum(xyz**2, axis=1)
    exact = surface.reduce_collocation_values(xyz[:, 2] / radius_squared**1.5)
    gradient = jnp.stack(
        [
            -3.0 * xyz[:, 2] * xyz[:, 0] / radius_squared**2.5,
            -3.0 * xyz[:, 2] * xyz[:, 1] / radius_squared**2.5,
            1.0 / radius_squared**1.5 - 3.0 * xyz[:, 2] ** 2 / radius_squared**2.5,
        ],
        axis=1,
    )
    neumann = surface.reduce_collocation_values(jnp.sum(gradient * surface.collocation_normals, axis=1))
    with pytest.raises(ValueError, match="requires spectral side density"):
        solve_reduced_exterior_laplace_neumann(surface, neumann, curved_side_geometry=True)

    errors = []
    for spectral, cap, curved in (
        (False, False, False),
        (True, False, False),
        (True, False, True),
        (True, True, True),
    ):
        result = solve_reduced_exterior_laplace_neumann(
            surface,
            neumann,
            spectral_side_density=spectral,
            spectral_cap_density=cap,
            curved_side_geometry=curved,
        )
        boundary_error = jnp.linalg.norm(result.boundary_potential - exact) / jnp.linalg.norm(exact)
        recovered = laplace_reduced_exterior_gradient_off_surface(
            surface,
            result.boundary_potential,
            neumann,
            jnp.asarray([[0.0, 0.0, 2.0]]),
            spectral_side_density=spectral,
            spectral_cap_density=cap,
            curved_side_geometry=curved,
        )
        field_error = jnp.abs(recovered[0, 2] + 0.25) / 0.25
        errors.append((float(boundary_error), float(field_error)))
        assert float(result.condition_number) < 5.0
        assert float(jnp.linalg.norm(result.residual)) < 3.0e-14

    assert errors[1][0] < 0.3 * errors[0][0]
    assert errors[1][1] < 0.7 * errors[0][1]
    assert errors[2][0] < 0.7 * errors[1][0]
    assert errors[3][0] < errors[2][0]


def test_panel_mesh_is_watertight_oriented_and_convergent() -> None:
    errors = []
    for ns, nxi, ntheta in ((9, 13, 8), (17, 25, 32)):
        grid = _grid(ns=ns, nxi=nxi)
        radius = 0.37
        surface = build_closed_mirror_surface(
            MirrorBoundary.from_radius(radius, grid),
            grid,
            axisymmetric_ntheta=ntheta,
        )
        triangles = np.asarray(surface.triangles)
        edges = np.sort(
            np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]),
            axis=1,
        )
        _, edge_counts = np.unique(edges, axis=0, return_counts=True)
        assert np.all(edge_counts == 2)

        vertices = np.asarray(surface.triangle_xyz)
        doubled_area = np.linalg.norm(
            np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]),
            axis=1,
        )
        assert np.all(doubled_area > 0.0)
        assert float(surface.mesh_volume) > 0.0

        expected_area = 2.0 * np.pi * radius * 2.8 + 2.0 * np.pi * radius**2
        expected_volume = np.pi * radius**2 * 2.8
        errors.append(
            max(
                abs(float(surface.mesh_area) - expected_area) / expected_area,
                abs(float(surface.mesh_volume) - expected_volume) / expected_volume,
            )
        )

    assert errors[1] < 0.3 * errors[0]
    assert errors[1] < 7.0e-3


def test_duffy_rule_converges_for_constant_and_linear_density() -> None:
    vertices = jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    exact_constant = np.sqrt(2.0) * np.log1p(np.sqrt(2.0)) / (4.0 * np.pi)
    errors = []
    for order in (2, 4, 8, 16):
        constant = duffy_triangle_single_layer(vertices, jnp.ones(3), order=order)
        linear = duffy_triangle_single_layer(vertices, jnp.asarray([0.0, 1.0, 1.0]), order=order)
        errors.append(abs(float(constant) - exact_constant))
        np.testing.assert_allclose(linear, 0.5 * constant, rtol=3.0e-15)

    assert all(right < left for left, right in zip(errors[:-1], errors[1:], strict=True))
    assert errors[-1] < 2.0e-14


def test_duffy_rule_is_differentiable_in_geometry_and_density() -> None:
    vertices = jnp.asarray([[0.0, 0.0, 0.0], [0.8, 0.1, 0.0], [-0.1, 0.7, 0.2]])
    density = jnp.asarray([0.4, -0.2, 0.7])

    geometry_gradient, density_gradient = jax.grad(
        lambda xyz, values: duffy_triangle_single_layer(xyz, values, order=10),
        argnums=(0, 1),
    )(vertices, density)

    assert np.all(np.isfinite(np.asarray(geometry_gradient)))
    assert np.all(np.isfinite(np.asarray(density_gradient)))
    np.testing.assert_allclose(jnp.sum(geometry_gradient, axis=0), 0.0, atol=3.0e-15)


@pytest.mark.parametrize("ntheta", [5, 6])
def test_spectral_side_density_reproduces_fourier_chebyshev_data(ntheta: int) -> None:
    grid = _grid(ns=7, mpol=2, ntheta=ntheta, nxi=7)
    surface = build_closed_mirror_surface(MirrorBoundary.from_radius(0.3, grid), grid)
    nside = 2 * grid.ntheta * (grid.nxi - 1)
    triangles = np.asarray(surface.triangles[:nside])
    theta = np.asarray(grid.theta)[:, None]
    xi = np.asarray(grid.xi)[None, :]
    values = 1.0 + 0.2 * np.cos(theta) + 0.1 * np.sin(2.0 * theta) + 0.3 * xi**3
    order = 4
    samples = _spectral_side_density_samples(
        jnp.asarray(triangles),
        jnp.asarray(values).reshape(-1),
        ntheta=grid.ntheta,
        nxi=grid.nxi,
        order=order,
        axisymmetric=False,
    )

    nodes, _ = _unit_gauss_legendre(order)
    u, v = nodes[:, None], nodes[None, :]
    barycentric = np.stack(
        [
            np.broadcast_to(1.0 - u, (nside, order, order)),
            np.broadcast_to(u * (1.0 - v), (nside, order, order)),
            np.broadcast_to(u * v, (nside, order, order)),
        ],
        axis=-1,
    )
    theta_nodes = theta[:, 0][triangles // grid.nxi]
    anchor = theta_nodes[:, :1]
    theta_nodes = anchor + (theta_nodes - anchor + np.pi) % (2.0 * np.pi) - np.pi
    theta_targets = np.sum(barycentric * theta_nodes[:, None, None, :], axis=-1)
    triangle_xi = xi[0][triangles % grid.nxi]
    xi_targets = np.sum(barycentric * triangle_xi[:, None, None, :], axis=-1)
    expected = 1.0 + 0.2 * np.cos(theta_targets) + 0.1 * np.sin(2.0 * theta_targets) + 0.3 * xi_targets**3
    np.testing.assert_allclose(samples, expected, rtol=2.0e-13, atol=2.0e-13)
    derivative = jax.grad(
        lambda data: jnp.sum(
            _spectral_side_density_samples(
                jnp.asarray(triangles),
                data,
                ntheta=grid.ntheta,
                nxi=grid.nxi,
                order=order,
                axisymmetric=False,
            )
            ** 2
        )
    )(jnp.asarray(values).reshape(-1))
    assert np.all(np.isfinite(np.asarray(derivative)))


def test_spectral_cap_density_reproduces_radial_fourier_data() -> None:
    ns, ntheta = 11, 8
    radial_nodes = jnp.asarray(np.linspace(0.0, 1.0, ns) ** 2)
    theta_nodes = 2.0 * jnp.pi * jnp.arange(ntheta) / ntheta
    rho, theta = jnp.meshgrid(radial_nodes, theta_nodes, indexing="ij")
    cap_xyz = jnp.stack(
        [0.3 * rho * jnp.cos(theta), 0.3 * rho * jnp.sin(theta), jnp.ones_like(rho)],
        axis=-1,
    )
    values = jnp.stack(
        [
            1.0 + 0.2 * jnp.cos(theta) + 0.3 * rho**2,
            -0.4 + 0.1 * jnp.sin(2.0 * theta) - 0.2 * rho**3,
        ]
    )
    target_rho = jnp.asarray([[0.07, 0.28], [0.61, 0.93]])
    target_theta = jnp.asarray([[0.13, 1.2], [3.4, 5.7]])
    source = jnp.stack(
        [
            0.3 * target_rho * jnp.cos(target_theta),
            0.3 * target_rho * jnp.sin(target_theta),
            jnp.ones_like(target_rho),
        ],
        axis=-1,
    )
    samples = spectral_cap_density_samples(source, values, cap_xyz)
    expected = jnp.stack(
        [
            1.0 + 0.2 * jnp.cos(target_theta) + 0.3 * target_rho**2,
            -0.4 + 0.1 * jnp.sin(2.0 * target_theta) - 0.2 * target_rho**3,
        ]
    )
    np.testing.assert_allclose(samples, expected, rtol=2.0e-12, atol=2.0e-12)
    derivative = jax.grad(lambda data: jnp.sum(spectral_cap_density_samples(source, data, cap_xyz) ** 2))(values)
    assert np.all(np.isfinite(np.asarray(derivative)))


def test_closed_surface_volume_is_differentiable() -> None:
    grid = _grid(ns=15, nxi=21)

    def volume(radius):
        boundary = MirrorBoundary.from_radius(radius, grid)
        return build_closed_mirror_surface(boundary, grid).volume

    radius = 0.29
    derivative = jax.grad(volume)(radius)
    expected = 2.0 * np.pi * radius * 2.8
    np.testing.assert_allclose(derivative, expected, rtol=4.0e-13)
    jax.make_jaxpr(volume)(radius)


@requires_virtual_casing
def test_constant_double_layer_distinguishes_inside_and_outside() -> None:
    """The closed-surface solid angle is one inside and zero outside."""

    values = []
    for ns, nxi in ((17, 25), (33, 49)):
        grid = _grid(ns=ns, nxi=nxi)
        surface = build_closed_mirror_surface(MirrorBoundary.from_radius(0.37, grid), grid)
        density = jnp.ones(surface.xyz.shape[0])
        targets = jnp.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 4.0]])
        values.append(np.asarray(laplace_double_layer_off_surface(surface, density, targets)))

    assert abs(values[1][0] - 1.0) < abs(values[0][0] - 1.0)
    assert abs(values[1][1]) < abs(values[0][1])
    np.testing.assert_allclose(values[1], [1.0, 0.0], atol=2.0e-5)


@requires_virtual_casing
def test_single_layer_gradient_has_far_field_monopole_limit() -> None:
    grid = _grid(ns=25, nxi=41)
    surface = build_closed_mirror_surface(MirrorBoundary.from_radius(0.31, grid), grid)
    density = jnp.ones(surface.xyz.shape[0])
    targets = jnp.asarray([[0.0, 0.0, 10.0], [0.0, 0.0, 20.0]])
    field = laplace_single_layer_gradient_off_surface(surface, density, targets)
    charge = surface.area
    reference = -targets * charge / (4.0 * jnp.pi * jnp.linalg.norm(targets, axis=1)[:, None] ** 3)
    relative_error = jnp.linalg.norm(field - reference, axis=1) / jnp.linalg.norm(reference, axis=1)

    assert float(relative_error[1]) < 0.3 * float(relative_error[0])
    np.testing.assert_allclose(field[:, :2], 0.0, atol=2.0e-16)


@requires_virtual_casing
def test_green_gradient_matches_finite_difference_on_axis() -> None:
    grid = _grid(ns=13, nxi=21)
    surface = build_closed_mirror_surface(MirrorBoundary.from_radius(0.31, grid), grid, cap_rim_grade=2.0)
    dirichlet = surface.xyz[:, 2]
    neumann = surface.normals[:, 2]
    targets = jnp.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.3]])
    gradient = laplace_green_gradient_off_surface(surface, dirichlet, neumann, targets)
    epsilon = 1.0e-5
    offset = jnp.asarray([0.0, 0.0, epsilon])
    finite_difference = (
        laplace_green_representation_off_surface(surface, dirichlet, neumann, targets + offset)
        - laplace_green_representation_off_surface(surface, dirichlet, neumann, targets - offset)
    ) / (2.0 * epsilon)

    assert np.all(np.isfinite(np.asarray(gradient)))
    np.testing.assert_allclose(gradient[:, 2], finite_difference, rtol=2.0e-8)
    np.testing.assert_allclose(gradient[:, :2], 0.0, atol=3.0e-15)


def test_panel_green_gradient_recovers_linear_harmonic_near_cap() -> None:
    grid = _grid(ns=13, nxi=21)
    surface = build_closed_mirror_surface(
        MirrorBoundary.from_radius(0.31, grid),
        grid,
        axisymmetric_ntheta=24,
        cap_rim_grade=3.5,
    )
    dirichlet = surface.reduce_collocation_values(surface.collocation_xyz[:, 2])
    neumann = surface.reduce_collocation_values(surface.collocation_normals[:, 2])
    targets = jnp.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 1.2]])
    gradient = laplace_reduced_green_gradient_off_surface(surface, dirichlet, neumann, targets)

    np.testing.assert_allclose(gradient, [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], rtol=2.0e-4, atol=2.0e-4)


def test_external_field_neumann_reconstruction_converges_near_caps() -> None:
    target_bz = jnp.asarray(0.08)
    targets = jnp.asarray([[0.0, 0.0, -0.4], [0.0, 0.0, 0.0], [0.0, 0.0, 0.4]])
    errors = []
    for ns, nxi, ntheta in ((9, 13, 16), (13, 21, 24)):
        grid = MirrorConfig(
            resolution=MirrorResolution(ns=ns, mpol=0, ntheta=1, nxi=nxi),
            z_min=-0.6,
            z_max=0.6,
        ).build_grid()
        surface = build_closed_mirror_surface(
            MirrorBoundary.from_radius(0.3, grid),
            grid,
            axisymmetric_ntheta=ntheta,
            cap_rim_grade=3.5,
        )
        external = _paraxial_field(surface.collocation_xyz)
        target_field = jnp.asarray([0.0, 0.0, target_bz])
        neumann = surface.reduce_collocation_values(
            jnp.sum((target_field - external) * surface.collocation_normals, axis=1)
        )
        result = solve_reduced_interior_laplace_neumann(surface, neumann)
        total_field = _paraxial_field(targets) + (
            laplace_reduced_green_gradient_off_surface(surface, result.boundary_potential, neumann, targets)
        )
        errors.append(float(jnp.max(jnp.abs(total_field[:, 2] - target_bz)) / target_bz))
        np.testing.assert_allclose(total_field[:, :2], 0.0, atol=2.0e-14)
        assert float(result.compatibility_error) < 2.0e-14
        assert float(result.condition_number) < 10.0

    assert errors[1] < 0.7 * errors[0]
    assert errors[1] < 6.0e-3


def test_plasma_external_neumann_adapter_matches_uniform_field_data() -> None:
    grid = MirrorConfig(
        resolution=MirrorResolution(ns=13, mpol=0, ntheta=1, nxi=21),
        z_min=-0.6,
        z_max=0.6,
    ).build_grid()
    radius = 0.3
    boundary = MirrorBoundary.from_radius(radius, grid)
    state = MirrorState.from_boundary(boundary, grid)
    geometry = evaluate_geometry(state, grid)
    target_bz = 0.08
    plasma_field = contravariant_field(
        state,
        geometry,
        grid,
        axial_flux_derivative=0.5 * target_bz * radius**2,
    )
    surface = build_closed_mirror_surface(boundary, grid, axisymmetric_ntheta=24, cap_rim_grade=3.5)
    adapted = axisymmetric_plasma_external_neumann(
        surface, plasma_field, grid, _paraxial_field
    )
    mapping = np.asarray(surface.collocation_to_reduced)
    _, representatives = np.unique(mapping, return_index=True)
    points = surface.collocation_xyz[jnp.asarray(representatives)]
    normals = surface.collocation_normals[jnp.asarray(representatives)]
    expected = jnp.sum(
        (jnp.asarray([0.0, 0.0, target_bz]) - _paraxial_field(points)) * normals,
        axis=1,
    )

    np.testing.assert_allclose(adapted, expected, rtol=3.0e-13, atol=3.0e-14)

    vacuum = solve_axisymmetric_exterior_vacuum(
        boundary,
        plasma_field,
        grid,
        _paraxial_field,
        axisymmetric_ntheta=24,
    )
    np.testing.assert_allclose(vacuum.neumann, adapted, rtol=3.0e-13, atol=3.0e-14)
    assert vacuum.lateral_field_xyz.shape == (grid.nxi, 3)
    assert float(jnp.max(jnp.abs(vacuum.lateral_b_normal))) < 2.0e-12
    assert float(vacuum.neumann_result.compatibility_error) < 2.0e-12
    assert float(vacuum.neumann_result.condition_number) < 10.0


def test_nonaxisymmetric_plasma_neumann_data_preserves_closed_flux() -> None:
    grid = MirrorConfig(
        resolution=MirrorResolution(ns=5, mpol=1, ntheta=3, nxi=7),
        z_min=-0.6,
        z_max=0.6,
    ).build_grid()
    theta = jnp.asarray(grid.theta)[:, None]
    xi = jnp.asarray(grid.xi)[None, :]
    boundary = MirrorBoundary.from_radius(0.3 * (1.0 + 0.05 * jnp.cos(theta) * (1.0 - xi**2)), grid)
    base = MirrorState.from_boundary(boundary, grid)
    radial = jnp.asarray(grid.s)[:, None, None]
    state = MirrorState(
        base.radius_scale,
        2.0e-3 * radial * jnp.sin(theta)[None] * (1.0 - xi**2)[None],
    )
    geometry = evaluate_geometry(state, grid)
    field = contravariant_field(
        state,
        geometry,
        grid,
        axial_flux_derivative=0.003,
        current_derivative=1.0e-3 * jnp.asarray(grid.s),
    )
    field_xyz = magnetic_field_xyz(field, geometry)
    np.testing.assert_allclose(
        jnp.sum(field_xyz**2, axis=-1),
        magnetic_field_squared(field, geometry),
        rtol=5.0e-13,
        atol=5.0e-15,
    )

    surface = build_closed_mirror_surface(boundary, grid, cap_rim_grade=2.5)
    neumann = plasma_external_neumann(surface, field, geometry, grid, _zero_field)
    assert neumann.shape == (surface.reduced_size,)
    lateral_size = grid.ntheta * grid.nxi
    np.testing.assert_allclose(neumann[:lateral_size], 0.0, atol=2.0e-15)
    quadrature_neumann = surface.expand_collocation_values(surface.expand_reduced_values(neumann))
    net_flux = jnp.sum(quadrature_neumann * surface.quadrature_weights)
    flux_scale = surface.area * jnp.sqrt(jnp.mean(neumann**2))
    assert float(jnp.abs(net_flux) / flux_scale) < 2.0e-3

    vacuum = solve_nonaxisymmetric_exterior_vacuum(
        boundary,
        field,
        geometry,
        grid,
        _zero_field,
        cap_rim_grade=2.5,
        order=6,
    )
    assert vacuum.lateral_field_xyz.shape == (grid.ntheta, grid.nxi, 3)
    assert float(jnp.max(jnp.abs(vacuum.lateral_b_normal))) < 3.0e-15
    assert float(vacuum.neumann_result.condition_number) < 20.0
    assert float(jnp.linalg.norm(vacuum.neumann_result.residual)) < 2.0e-12


def test_axisymmetric_exterior_vacuum_is_shape_differentiable() -> None:
    grid = MirrorConfig(
        resolution=MirrorResolution(ns=5, mpol=0, ntheta=1, nxi=7),
        z_min=-0.6,
        z_max=0.6,
    ).build_grid()
    def lateral_field(radius):
        boundary = MirrorBoundary.from_radius(radius, grid)
        state = MirrorState.from_boundary(boundary, grid)
        geometry = evaluate_geometry(state, grid)
        plasma_field = contravariant_field(
            state,
            geometry,
            grid,
            axial_flux_derivative=0.0036,
        )
        return solve_axisymmetric_exterior_vacuum(
            boundary,
            plasma_field,
            grid,
            _paraxial_field,
            axisymmetric_ntheta=8,
            cap_rim_grade=3.0,
            order=6,
            spectral_side_density=True,
            spectral_cap_density=True,
            curved_side_geometry=True,
        ).lateral_field_xyz

    _, tangent = jax.jvp(
        lateral_field,
        (jnp.asarray(0.3),),
        (jnp.asarray(1.0),),
    )
    assert np.all(np.isfinite(np.asarray(tangent)))
    assert float(jnp.linalg.norm(tangent)) > 0.0


def test_nonaxisymmetric_exterior_vacuum_is_shape_differentiable() -> None:
    grid = MirrorConfig(
        resolution=MirrorResolution(ns=5, mpol=1, ntheta=3, nxi=5),
        z_min=-0.5,
        z_max=0.5,
    ).build_grid()
    theta = jnp.asarray(grid.theta)[:, None]
    xi = jnp.asarray(grid.xi)[None, :]
    def lateral_field(amplitude):
        boundary = MirrorBoundary.from_radius(
            0.3 * (1.0 + amplitude * jnp.cos(theta) * (1.0 - xi**2)),
            grid,
        )
        state = MirrorState.from_boundary(boundary, grid)
        geometry = evaluate_geometry(state, grid)
        field = contravariant_field(
            state,
            geometry,
            grid,
            axial_flux_derivative=0.003,
            current_derivative=1.0e-3 * jnp.asarray(grid.s),
        )
        return solve_nonaxisymmetric_exterior_vacuum(
            boundary,
            field,
            geometry,
            grid,
            _zero_field,
            cap_rim_grade=2.5,
            order=4,
            spectral_side_density=True,
            spectral_cap_density=True,
            curved_side_geometry=True,
        ).lateral_field_xyz

    _, tangent = jax.jvp(
        lateral_field,
        (jnp.asarray(0.04),),
        (jnp.asarray(1.0),),
    )
    assert np.all(np.isfinite(np.asarray(tangent)))
    assert float(jnp.linalg.norm(tangent)) > 0.0


@requires_virtual_casing
def test_green_representation_converges_for_harmonic_polynomials() -> None:
    targets = jnp.asarray([[0.1, 0.05, 0.2], [0.0, 0.0, 4.0]])
    errors = []
    for ns, nxi, ntheta in ((17, 25, 16), (33, 49, 32)):
        grid = _grid(ns=ns, nxi=nxi)
        surface = build_closed_mirror_surface(
            MirrorBoundary.from_radius(0.37, grid),
            grid,
            axisymmetric_ntheta=ntheta,
        )
        xyz = surface.xyz
        normal = surface.normals
        cases = (
            (jnp.ones(xyz.shape[0]), jnp.zeros(xyz.shape[0]), 1.0),
            (xyz[:, 0], normal[:, 0], targets[0, 0]),
            (xyz[:, 2], normal[:, 2], targets[0, 2]),
            (
                xyz[:, 0] ** 2 - xyz[:, 1] ** 2,
                2.0 * (xyz[:, 0] * normal[:, 0] - xyz[:, 1] * normal[:, 1]),
                targets[0, 0] ** 2 - targets[0, 1] ** 2,
            ),
        )
        case_errors = []
        for dirichlet, neumann, interior_value in cases:
            represented = laplace_green_representation_off_surface(surface, dirichlet, neumann, targets)
            case_errors.append(float(jnp.max(jnp.abs(represented - jnp.asarray([interior_value, 0.0])))))
        errors.append(max(case_errors))

    assert errors[1] < 0.4 * errors[0]
    assert errors[1] < 2.0e-5


def test_singular_boundary_green_identity_converges_for_linear_harmonics() -> None:
    errors = []
    for ns, nxi, ntheta, order in ((7, 9, 8, 6), (13, 21, 20, 8)):
        grid = _grid(ns=ns, nxi=nxi)
        surface = build_closed_mirror_surface(
            MirrorBoundary.from_radius(0.37, grid),
            grid,
            axisymmetric_ntheta=ntheta,
        )
        xyz = surface.collocation_xyz
        normal = surface.collocation_normals
        case_errors = []
        for dirichlet, neumann in (
            (xyz[:, 0], normal[:, 0]),
            (xyz[:, 2], normal[:, 2]),
        ):
            residual = laplace_green_boundary_residual(surface, dirichlet, neumann, order=order)
            scale = jnp.sqrt(jnp.mean(dirichlet**2) + surface.area * jnp.mean(neumann**2))
            case_errors.append(float(jnp.sqrt(jnp.mean(residual**2)) / scale))
        errors.append(max(case_errors))

    assert errors[1] < errors[0]
    assert errors[1] < 2.0e-3
