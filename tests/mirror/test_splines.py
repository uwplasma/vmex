"""Cubic B-spline identities, refinement, and differentiation tests."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.interpolate import BSpline

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from vmec_jax.mirror.splines import CubicBSplineBasis  # noqa: E402
from vmec_jax.mirror import MirrorBoundary, MirrorConfig, MirrorResolution, MirrorState  # noqa: E402
from vmec_jax.mirror.forces import mirror_energy  # noqa: E402
from vmec_jax.mirror.geometry import evaluate_geometry  # noqa: E402
from vmec_jax.mirror.splines import (  # noqa: E402
    SplineMirrorDiscretization,
    SplineMirrorState,
)


def test_clamped_basis_matches_scipy_and_partitions_unity() -> None:
    basis = CubicBSplineBasis.clamped(np.linspace(-1.0, 1.0, 7))
    points = np.linspace(-1.0, 1.0, 101)
    actual = np.asarray(basis.basis_matrix(points))
    expected = np.column_stack(
        [BSpline(basis.knots, np.eye(basis.size)[index], 3)(points) for index in range(basis.size)]
    )
    np.testing.assert_allclose(actual, expected, rtol=2.0e-14, atol=2.0e-14)
    np.testing.assert_allclose(np.sum(actual, axis=1), 1.0, atol=2.0e-15)
    np.testing.assert_allclose(actual[0], np.eye(basis.size)[0], atol=0.0)
    np.testing.assert_allclose(actual[-1], np.eye(basis.size)[-1], atol=0.0)


def test_clamped_derivatives_and_cubic_reproduction() -> None:
    basis = CubicBSplineBasis.clamped(np.linspace(-1.0, 1.0, 8))

    def polynomial(x):
        return 1.2 - 0.4 * x + 0.7 * x**2 - 0.3 * x**3

    coefficients = basis.fit(polynomial(jnp.asarray(basis.collocation_nodes)))
    points = jnp.linspace(-1.0, 1.0, 83)
    np.testing.assert_allclose(basis.evaluate(coefficients, points), polynomial(points), atol=2.0e-14)
    np.testing.assert_allclose(
        basis.evaluate(coefficients, points, derivative=1), -0.4 + 1.4 * points - 0.9 * points**2, atol=8.0e-14
    )
    np.testing.assert_allclose(
        basis.evaluate(coefficients, points, derivative=2), 1.4 - 1.8 * points, atol=4.0e-13
    )
    np.testing.assert_allclose(basis.integrate(coefficients), 2.0 * 1.2 + 2.0 * 0.7 / 3.0, atol=2.0e-14)


def test_open_knot_insertion_preserves_curve_and_jax_derivatives() -> None:
    basis = CubicBSplineBasis.clamped(np.linspace(-1.0, 1.0, 6))
    coefficients = jnp.sin(1.7 * jnp.asarray(basis.collocation_nodes)) + 0.2 * basis.collocation_nodes**2
    refined, refined_coefficients = basis.insert_knot(coefficients, 0.13)
    points = jnp.linspace(-1.0, 1.0, 117)
    np.testing.assert_allclose(
        refined.evaluate(refined_coefficients, points), basis.evaluate(coefficients, points), rtol=3.0e-14, atol=3.0e-14
    )

    direction = jnp.linspace(-0.3, 0.4, basis.size)
    primal, tangent = jax.jvp(lambda values: basis.evaluate(values, points), (coefficients,), (direction,))
    np.testing.assert_allclose(primal, basis.evaluate(coefficients, points), atol=0.0)
    np.testing.assert_allclose(tangent, basis.evaluate(direction, points), atol=2.0e-15)
    cotangent = jnp.cos(points)
    reverse = jax.grad(lambda values: jnp.vdot(basis.evaluate(values, points), cotangent))(coefficients)
    np.testing.assert_allclose(reverse, basis.basis_matrix(points).T @ cotangent, atol=2.0e-14)


def test_periodic_basis_has_c2_endpoint_continuity_and_partition_unity() -> None:
    basis = CubicBSplineBasis.periodic_uniform(12)
    points = jnp.linspace(0.0, 2.0 * jnp.pi, 97)
    np.testing.assert_allclose(jnp.sum(basis.basis_matrix(points), axis=1), 1.0, atol=2.0e-15)
    for derivative in range(3):
        np.testing.assert_allclose(
            basis.basis_matrix(jnp.asarray([0.0]), derivative=derivative),
            basis.basis_matrix(jnp.asarray([2.0 * jnp.pi]), derivative=derivative),
            atol=2.0e-14,
        )


def test_periodic_fit_converges_and_is_differentiable() -> None:
    errors = []
    for size in (8, 16, 32):
        basis = CubicBSplineBasis.periodic_uniform(size)
        values = jnp.sin(basis.collocation_nodes) + 0.2 * jnp.cos(2.0 * basis.collocation_nodes)
        coefficients = basis.fit(values)
        points = jnp.linspace(0.0, 2.0 * jnp.pi, 257, endpoint=False)
        exact = jnp.sin(points) + 0.2 * jnp.cos(2.0 * points)
        errors.append(float(jnp.max(jnp.abs(basis.evaluate(coefficients, points) - exact))))
        derivative = jax.grad(lambda scale: jnp.sum(basis.fit(scale * values) ** 2))(1.0)
        assert np.isfinite(float(derivative))
    assert errors[1] < 0.08 * errors[0]
    assert errors[2] < 0.08 * errors[1]


def _spline_polynomial_state():
    config = MirrorConfig(
        resolution=MirrorResolution(ns=7, mpol=0, ntheta=1, nxi=41),
        z_min=-1.4,
        z_max=1.4,
    )
    chebyshev_grid = config.build_grid()
    radius = 0.27 * (1.0 + 0.16 * jnp.asarray(chebyshev_grid.xi) ** 2)
    boundary = MirrorBoundary.from_radius(radius, chebyshev_grid)
    state = MirrorState.from_boundary(boundary, chebyshev_grid)
    discretization = SplineMirrorDiscretization.build(config, elements=6)
    spline_boundary = discretization.fit_boundary(boundary, chebyshev_grid)
    spline_state = discretization.fit_state(state, chebyshev_grid)
    return config, chebyshev_grid, boundary, state, discretization, spline_boundary, spline_state


def test_coefficient_native_state_matches_chebyshev_polynomial_geometry_and_energy() -> None:
    _, chebyshev_grid, _, state, discretization, spline_boundary, spline_state = (
        _spline_polynomial_state()
    )
    projected = discretization.project_fixed_boundary(spline_state, spline_boundary)
    evaluated = discretization.evaluate_state(projected)
    spline_geometry = evaluate_geometry(evaluated, discretization.grid)
    chebyshev_geometry = evaluate_geometry(state, chebyshev_grid)
    np.testing.assert_allclose(spline_geometry.volume, chebyshev_geometry.volume, rtol=3.0e-14)
    actual_ends = evaluated.radius_scale[:, :, [0, -1]]
    boundary_ends = discretization.evaluate_boundary(spline_boundary).radius_scale[:, [0, -1]]
    np.testing.assert_allclose(
        actual_ends, jnp.broadcast_to(boundary_ends, actual_ends.shape), atol=3.0e-15
    )

    spline_energy = mirror_energy(evaluated, discretization.grid, axial_flux_derivative=0.1)
    chebyshev_energy = mirror_energy(state, chebyshev_grid, axial_flux_derivative=0.1)
    np.testing.assert_allclose(spline_energy.total, chebyshev_energy.total, rtol=2.0e-12)
    assert projected.radius_coefficients.shape[-1] == discretization.coefficient_count
    assert evaluated.radius_scale.shape[-1] > projected.radius_coefficients.shape[-1]


def test_coefficient_native_energy_gradient_matches_central_difference() -> None:
    _, _, _, _, discretization, spline_boundary, spline_state = _spline_polynomial_state()
    projected = discretization.project_fixed_boundary(spline_state, spline_boundary)
    direction = jnp.zeros_like(projected.radius_coefficients).at[2, 0, 2:-2].set(
        jnp.linspace(-0.2, 0.3, discretization.coefficient_count - 4)
    )

    def objective(radius_coefficients):
        candidate = SplineMirrorState(radius_coefficients, projected.lambda_coefficients)
        evaluated = discretization.evaluate_state(candidate)
        return mirror_energy(evaluated, discretization.grid, axial_flux_derivative=0.1).total

    derivative = jnp.vdot(jax.grad(objective)(projected.radius_coefficients), direction)
    step = 3.0e-6
    finite_difference = (
        objective(projected.radius_coefficients + step * direction)
        - objective(projected.radius_coefficients - step * direction)
    ) / (2.0 * step)
    np.testing.assert_allclose(derivative, finite_difference, rtol=1.0e-7, atol=2.0e-7)
