"""Closed mirror surface geometry required by the free-space exterior solve."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from vmec_jax.mirror import (  # noqa: E402
    MirrorBoundary,
    MirrorConfig,
    MirrorResolution,
    build_closed_mirror_surface,
    laplace_double_layer_off_surface,
    laplace_green_representation_off_surface,
    laplace_single_layer_gradient_off_surface,
)


def _grid(*, ns: int = 17, mpol: int = 0, ntheta: int = 1, nxi: int = 25):
    return MirrorConfig(
        resolution=MirrorResolution(ns=ns, mpol=mpol, ntheta=ntheta, nxi=nxi),
        z_min=-1.4,
        z_max=1.4,
    ).build_grid()


def test_closed_cylinder_has_exact_area_volume_and_orientation() -> None:
    grid = _grid()
    radius = 0.37
    surface = build_closed_mirror_surface(MirrorBoundary.from_radius(radius, grid), grid)

    expected_area = 2.0 * np.pi * radius * 2.8 + 2.0 * np.pi * radius**2
    expected_volume = np.pi * radius**2 * 2.8
    np.testing.assert_allclose(surface.area, expected_area, rtol=3.0e-14)
    np.testing.assert_allclose(surface.volume, expected_volume, rtol=3.0e-14)
    np.testing.assert_allclose(
        jnp.sum(surface.weighted_normals, axis=0), 0.0, atol=3.0e-15
    )
    assert np.all(np.asarray(surface.lower_cap_weighted_normals[..., 2]) < 0.0)
    assert np.all(np.asarray(surface.upper_cap_weighted_normals[..., 2]) > 0.0)


def test_shaped_surface_satisfies_divergence_theorem_moments() -> None:
    grid = _grid(ns=21, mpol=3, ntheta=9, nxi=33)
    theta = jnp.asarray(grid.theta)[:, None]
    xi = jnp.asarray(grid.xi)[None, :]
    radius = 0.31 * (
        1.0
        + 0.08 * xi**2
        + 0.04 * jnp.cos(2.0 * theta) * (1.0 - xi**2)
    )
    surface = build_closed_mirror_surface(MirrorBoundary.from_radius(radius, grid), grid)

    net_normal = jnp.sum(surface.weighted_normals, axis=0)
    first_moment = jnp.einsum(
        "ni,nj->ij", surface.xyz, surface.weighted_normals
    )
    np.testing.assert_allclose(net_normal, 0.0, atol=2.0e-13)
    np.testing.assert_allclose(
        first_moment, np.eye(3) * float(surface.volume), rtol=3.0e-12, atol=3.0e-13
    )


def test_caps_share_the_lateral_end_rings() -> None:
    grid = _grid(ns=13, mpol=2, ntheta=7, nxi=17)
    theta = jnp.asarray(grid.theta)[:, None]
    xi = jnp.asarray(grid.xi)[None, :]
    boundary = MirrorBoundary.from_radius(
        0.3 * (1.0 + 0.05 * jnp.cos(theta) * (1.0 + xi)), grid
    )
    surface = build_closed_mirror_surface(boundary, grid)

    np.testing.assert_allclose(surface.lower_cap_xyz[-1], surface.lateral_xyz[:, 0])
    np.testing.assert_allclose(surface.upper_cap_xyz[-1], surface.lateral_xyz[:, -1])


def test_collocation_map_identifies_cap_centers_and_rims() -> None:
    grid = _grid(ns=13, mpol=2, ntheta=7, nxi=17)
    theta = jnp.asarray(grid.theta)[:, None]
    xi = jnp.asarray(grid.xi)[None, :]
    surface = build_closed_mirror_surface(
        MirrorBoundary.from_radius(
            0.3 * (1.0 + 0.04 * jnp.cos(2.0 * theta) * (1.0 - xi**2)), grid
        ),
        grid,
    )
    nquadrature = surface.xyz.shape[0]
    ncollocation = surface.collocation_xyz.shape[0]
    expected = grid.ntheta * grid.nxi + 2 * (1 + (grid.ns - 2) * grid.ntheta)
    assert ncollocation == expected
    assert ncollocation < nquadrature
    assert np.unique(np.asarray(surface.collocation_xyz), axis=0).shape[0] == ncollocation

    values = (
        surface.collocation_xyz[:, 0]
        + 2.0 * surface.collocation_xyz[:, 1]
        - 0.5 * surface.collocation_xyz[:, 2]
    )
    expanded = surface.expand_collocation_values(values)
    expected_values = surface.xyz[:, 0] + 2.0 * surface.xyz[:, 1] - 0.5 * surface.xyz[:, 2]
    np.testing.assert_allclose(expanded, expected_values, atol=2.0e-15)


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
            np.concatenate(
                [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]
            ),
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


def test_constant_double_layer_distinguishes_inside_and_outside() -> None:
    """The closed-surface solid angle is one inside and zero outside."""

    values = []
    for ns, nxi in ((17, 25), (33, 49)):
        grid = _grid(ns=ns, nxi=nxi)
        surface = build_closed_mirror_surface(
            MirrorBoundary.from_radius(0.37, grid), grid
        )
        density = jnp.ones(surface.xyz.shape[0])
        targets = jnp.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 4.0]])
        values.append(
            np.asarray(
                laplace_double_layer_off_surface(surface, density, targets)
            )
        )

    assert abs(values[1][0] - 1.0) < abs(values[0][0] - 1.0)
    assert abs(values[1][1]) < abs(values[0][1])
    np.testing.assert_allclose(values[1], [1.0, 0.0], atol=2.0e-5)


def test_single_layer_gradient_has_far_field_monopole_limit() -> None:
    grid = _grid(ns=25, nxi=41)
    surface = build_closed_mirror_surface(MirrorBoundary.from_radius(0.31, grid), grid)
    density = jnp.ones(surface.xyz.shape[0])
    targets = jnp.asarray([[0.0, 0.0, 10.0], [0.0, 0.0, 20.0]])
    field = laplace_single_layer_gradient_off_surface(surface, density, targets)
    charge = surface.area
    reference = -targets * charge / (
        4.0 * jnp.pi * jnp.linalg.norm(targets, axis=1)[:, None] ** 3
    )
    relative_error = jnp.linalg.norm(field - reference, axis=1) / jnp.linalg.norm(
        reference, axis=1
    )

    assert float(relative_error[1]) < 0.3 * float(relative_error[0])
    np.testing.assert_allclose(field[:, :2], 0.0, atol=2.0e-16)


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
            represented = laplace_green_representation_off_surface(
                surface, dirichlet, neumann, targets
            )
            case_errors.append(
                float(jnp.max(jnp.abs(represented - jnp.asarray([interior_value, 0.0]))))
            )
        errors.append(max(case_errors))

    assert errors[1] < 0.4 * errors[0]
    assert errors[1] < 2.0e-5
