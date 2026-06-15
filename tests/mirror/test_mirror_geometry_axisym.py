from __future__ import annotations

import numpy as np
import pytest

from vmec_jax.mirror import MirrorConfig, MirrorResolution
from vmec_jax.mirror.core.boundary import MirrorBoundary
from vmec_jax.mirror.core.state import MirrorStateAxisym
from vmec_jax.mirror.kernels.constraints import lambda_surface_average_axisym, project_axisym_state
from vmec_jax.mirror.kernels.geometry import evaluate_axisym_geometry

pytestmark = pytest.mark.mirror


def _axisym_grid(*, ns=17, nxi=33, length=1.7):
    return MirrorConfig(
        resolution=MirrorResolution(ns=ns, ntheta=1, nxi=nxi, mpol=0),
        z_min=-length,
        z_max=length,
    ).build_grid()


def test_straight_cylinder_geometry_matches_analytic_metrics():
    radius = 0.37
    length = 1.8
    grid = _axisym_grid(length=length)
    boundary = MirrorBoundary.constant_radius(radius)
    state = MirrorStateAxisym.from_boundary(grid, boundary)

    geom = evaluate_axisym_geometry(state, grid)
    expected_sqrtg = 0.5 * radius**2 * length
    assert np.allclose(geom.r, grid.rho_full[:, None] * radius)
    assert np.allclose(geom.z, length * grid.xi)
    assert np.allclose(geom.sqrtg, expected_sqrtg)
    assert np.all(geom.sqrtg > 0.0)
    assert np.allclose(geom.g_thetatheta, radius**2 * grid.s_full[:, None])
    assert np.allclose(geom.g_sxi, 0.0, atol=2.0e-14)
    assert np.allclose(geom.g_xixi, length**2)

    nonaxis = grid.s_full > 0.0
    assert np.allclose(geom.g_ss[nonaxis], radius**2 / (4.0 * grid.s_full[nonaxis, None]))
    expected_volume = 2.0 * np.pi * length * radius**2
    assert np.isclose(geom.volume, expected_volume, rtol=2.0e-14, atol=2.0e-14)


def test_polynomial_flared_tube_geometry_matches_analytic_metrics():
    radius = 0.25
    epsilon = 0.18
    length = 1.4
    grid = _axisym_grid(ns=19, nxi=41, length=length)
    boundary = MirrorBoundary.polynomial_radius(r0=radius, a2=epsilon)
    state = MirrorStateAxisym.from_boundary(grid, boundary)

    geom = evaluate_axisym_geometry(state, grid)
    a_xi = 2.0 * radius * epsilon * grid.xi
    boundary_radius = radius * (1.0 + epsilon * grid.xi**2)
    expected_sqrtg = 0.5 * boundary_radius[None, :] ** 2 * length
    expected_g_sxi = 0.5 * boundary_radius * a_xi
    expected_volume = np.pi * length * np.dot(grid.w_xi, boundary_radius**2)

    assert np.allclose(geom.r[-1], boundary_radius)
    assert np.allclose(geom.sqrtg, expected_sqrtg, atol=2.0e-12, rtol=2.0e-12)
    assert np.allclose(geom.g_sxi, expected_g_sxi[None, :], atol=2.0e-12, rtol=2.0e-12)
    expected_g_xixi = grid.s_full[:, None] * a_xi[None, :] ** 2 + length**2
    assert np.allclose(geom.g_xixi, expected_g_xixi, atol=2.0e-12, rtol=2.0e-12)
    assert np.isclose(geom.volume, expected_volume, atol=2.0e-13, rtol=2.0e-13)


def test_axisymmetric_constraints_fix_side_boundary_ends_axis_and_lambda_gauge():
    grid = _axisym_grid(ns=7, nxi=17, length=1.0)
    boundary = MirrorBoundary.polynomial_radius(r0=0.32, a2=-0.1, a4=0.05)
    rng = np.random.default_rng(1729)
    state = MirrorStateAxisym(
        a=0.2 + 0.1 * rng.random((grid.ns, grid.nxi)),
        lam=1.0 + rng.normal(size=(grid.ns, grid.nxi)),
    )

    projected = project_axisym_state(state, grid, boundary)
    boundary_radius = boundary.radius_on_grid(grid)
    assert np.allclose(projected.a[-1], boundary_radius)
    assert np.allclose(projected.a[:, 0], boundary_radius[0])
    assert np.allclose(projected.a[:, -1], boundary_radius[-1])
    assert np.allclose(projected.a[0], projected.a[1])
    assert np.allclose(lambda_surface_average_axisym(projected.lam, grid), 0.0, atol=2.0e-15)


def test_tabulated_boundary_interpolates_and_preserves_requested_nodes():
    grid = _axisym_grid(nxi=25)
    coarse = _axisym_grid(nxi=9)
    radius_values = 0.27 * (1.0 + 0.12 * coarse.xi**2 + 0.03 * coarse.xi**4)
    boundary = MirrorBoundary.tabulated_radius(coarse.xi, radius_values)

    assert np.allclose(boundary.radius(coarse.xi), radius_values)
    state = MirrorStateAxisym.from_boundary(grid, boundary)
    geom = evaluate_axisym_geometry(state, grid)
    assert np.allclose(geom.r[-1], boundary.radius_on_grid(grid))
