"""Cubic B-spline identities, refinement, and differentiation tests."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.interpolate import BSpline

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from vmec_jax.mirror.splines import CubicBSplineBasis  # noqa: E402
from vmec_jax.mirror import (  # noqa: E402
    MirrorBoundary,
    MirrorConfig,
    MirrorResolution,
    MirrorState,
    solve_fixed_boundary_cli,
)
from vmec_jax.mirror.forces import mirror_energy  # noqa: E402
from vmec_jax.mirror.geometry import (  # noqa: E402
    evaluate_closed_spline_axis,
    evaluate_geometry,
    racetrack_centerline_coefficients,
)
from vmec_jax.mirror.solver import SeparableMirrorPreconditioner  # noqa: E402
from vmec_jax.mirror.splines import (  # noqa: E402
    SplineMirrorDiscretization,
    SplineMirrorState,
    solve_spline_fixed_boundary_cli,
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
    np.testing.assert_allclose(basis.evaluate(coefficients, points, derivative=2), 1.4 - 1.8 * points, atol=4.0e-13)
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


def test_closed_circle_axis_has_periodic_rotation_minimizing_frame() -> None:
    radius = 1.7
    basis = CubicBSplineBasis.periodic_uniform(24)
    nodes = jnp.asarray(basis.collocation_nodes)
    samples = jnp.stack(
        (radius * jnp.cos(nodes), jnp.zeros_like(nodes), radius * jnp.sin(nodes)),
        axis=-1,
    )
    coefficients = basis.fit(samples, axis=0)
    points = np.linspace(*basis.domain, 193, endpoint=False)
    axis = evaluate_closed_spline_axis(
        coefficients,
        basis,
        points,
        initial_normal=jnp.asarray([0.0, 1.0, 0.0]),
    )

    np.testing.assert_allclose(np.linalg.norm(axis.tangent, axis=-1), 1.0, atol=2.0e-14)
    np.testing.assert_allclose(np.linalg.norm(axis.normal, axis=-1), 1.0, atol=2.0e-14)
    np.testing.assert_allclose(np.sum(axis.tangent * axis.normal, axis=-1), 0.0, atol=2.0e-14)
    np.testing.assert_allclose(axis.curvature, 1.0 / radius, rtol=6.0e-3)
    np.testing.assert_allclose(axis.arc_length, 2.0 * np.pi * radius, rtol=2.0e-5)
    assert float(axis.closure_error) < 2.0e-14
    assert float(axis.tangent_closure_error) < 2.0e-14
    assert float(axis.frame_closure_error) < 2.0e-14
    assert abs(float(axis.frame_holonomy)) < 2.0e-13

    gradient = jax.grad(
        lambda values: evaluate_closed_spline_axis(
            values,
            basis,
            points,
            initial_normal=jnp.asarray([0.0, 1.0, 0.0]),
        ).arc_length
    )(coefficients)
    assert np.all(np.isfinite(gradient))
    assert float(jnp.linalg.norm(gradient)) > 1.0


def test_racetrack_spline_has_long_straight_legs_and_c2_closure() -> None:
    basis = CubicBSplineBasis.periodic_uniform(32)
    coefficients = racetrack_centerline_coefficients(
        basis.size,
        straight_length=6.0,
        return_radius=1.0,
    )
    points = np.linspace(*basis.domain, 257, endpoint=False)
    axis = evaluate_closed_spline_axis(
        coefficients,
        basis,
        points,
        initial_normal=jnp.asarray([0.0, 1.0, 0.0]),
    )

    straight = np.asarray(axis.curvature) < 1.0e-10
    assert np.count_nonzero(straight) > 0.25 * points.size
    assert float(axis.closure_error) < 2.0e-14
    assert float(axis.tangent_closure_error) < 2.0e-14
    assert float(axis.frame_closure_error) < 2.0e-14
    assert float(jnp.min(axis.speed)) > 0.1


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
    _, chebyshev_grid, _, state, discretization, spline_boundary, spline_state = _spline_polynomial_state()
    projected = discretization.project_fixed_boundary(spline_state, spline_boundary)
    evaluated = discretization.evaluate_state(projected)
    spline_geometry = evaluate_geometry(evaluated, discretization.grid)
    chebyshev_geometry = evaluate_geometry(state, chebyshev_grid)
    np.testing.assert_allclose(spline_geometry.volume, chebyshev_geometry.volume, rtol=3.0e-14)
    actual_ends = evaluated.radius_scale[:, :, [0, -1]]
    boundary_ends = discretization.evaluate_boundary(spline_boundary).radius_scale[:, [0, -1]]
    np.testing.assert_allclose(actual_ends, jnp.broadcast_to(boundary_ends, actual_ends.shape), atol=3.0e-15)

    spline_energy = mirror_energy(evaluated, discretization.grid, axial_flux_derivative=0.1)
    chebyshev_energy = mirror_energy(state, chebyshev_grid, axial_flux_derivative=0.1)
    np.testing.assert_allclose(spline_energy.total, chebyshev_energy.total, rtol=2.0e-12)
    assert projected.radius_coefficients.shape[-1] == discretization.coefficient_count
    assert evaluated.radius_scale.shape[-1] > projected.radius_coefficients.shape[-1]


def test_boundary_transfer_preserves_nested_self_similarity() -> None:
    config = MirrorConfig(resolution=MirrorResolution(ns=5, mpol=1, ntheta=4, nxi=9))
    source_grid = config.build_grid()
    theta = jnp.asarray(source_grid.theta)[:, None]
    xi = jnp.asarray(source_grid.xi)[None, :]
    source_boundary = MirrorBoundary.from_radius(0.3, source_grid)
    target_boundary = MirrorBoundary.from_radius(
        0.24 * (1.0 + 0.08 * jnp.cos(2.0 * theta) * (1.0 - xi**2)),
        source_grid,
    )
    discretization = SplineMirrorDiscretization.build(config, elements=4)
    source = discretization.fit_boundary(source_boundary, source_grid)
    target = discretization.fit_boundary(target_boundary, source_grid)
    state = discretization.fit_state(MirrorState.from_boundary(source_boundary, source_grid), source_grid)
    transferred = discretization.transfer_boundary(state, source, target)
    evaluated = discretization.evaluate_state(transferred)
    evaluated_target = discretization.evaluate_boundary(target).radius_scale

    np.testing.assert_allclose(
        evaluated.radius_scale,
        jnp.broadcast_to(evaluated_target, evaluated.radius_scale.shape),
        rtol=3.0e-14,
        atol=3.0e-14,
    )
    assert not bool(evaluate_geometry(evaluated, discretization.grid).jacobian_sign_changed)


def test_coefficient_native_energy_gradient_matches_central_difference() -> None:
    _, _, _, _, discretization, spline_boundary, spline_state = _spline_polynomial_state()
    projected = discretization.project_fixed_boundary(spline_state, spline_boundary)
    direction = (
        jnp.zeros_like(projected.radius_coefficients)
        .at[2, 0, 2:-2]
        .set(jnp.linspace(-0.2, 0.3, discretization.coefficient_count - 4))
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
    np.testing.assert_allclose(derivative, finite_difference, rtol=2.0e-7, atol=2.0e-7)


def test_spline_fixed_boundary_solver_recovers_cylindrical_equilibrium() -> None:
    config = MirrorConfig(
        resolution=MirrorResolution(ns=7, mpol=0, ntheta=1, nxi=9),
        z_min=-1.2,
        z_max=1.2,
        ftol=1.0e-12,
        max_iterations=300,
    )
    source_grid = config.build_grid()
    boundary = MirrorBoundary.from_radius(0.3, source_grid)
    base = MirrorState.from_boundary(boundary, source_grid)
    s = jnp.asarray(source_grid.s)[:, None, None]
    xi = jnp.asarray(source_grid.xi)[None, None, :]
    initial = MirrorState(
        base.radius_scale + 0.03 * s * (1.0 - s) * (1.0 - xi**2),
        base.lambda_stream,
    )
    discretization = SplineMirrorDiscretization.build(config, elements=4)
    spline_boundary = discretization.fit_boundary(boundary, source_grid)
    spline_initial = discretization.fit_state(initial, source_grid)

    result = solve_spline_fixed_boundary_cli(
        spline_initial,
        spline_boundary,
        discretization,
        config,
        axial_flux_derivative=0.1,
        gradient_tolerance=1.0e-12,
        require_convergence=True,
    )

    assert result.evaluated.converged
    assert result.evaluated.iterations > 0
    assert float(result.evaluated.variational.maximum) <= config.ftol
    assert float(result.evaluated.staggered_weak_force.maximum) <= 1.2 * config.ftol
    np.testing.assert_allclose(result.evaluated.state.radius_scale, 0.3, atol=3.0e-13)
    assert result.coefficient_state.radius_coefficients.shape[-1] == 7


def test_spline_solver_converges_nonaxisymmetric_finite_current_state() -> None:
    config = MirrorConfig(
        resolution=MirrorResolution(ns=5, mpol=1, ntheta=3, nxi=7),
        z_min=-1.0,
        z_max=1.0,
        ftol=1.0e-12,
        max_iterations=1000,
    )
    source_grid = config.build_grid()
    theta = jnp.asarray(source_grid.theta)[:, None]
    xi = jnp.asarray(source_grid.xi)[None, :]
    boundary = MirrorBoundary.from_radius(0.3 * (1.0 + 0.02 * jnp.cos(theta) * (1.0 - xi**2)), source_grid)
    initial = MirrorState.from_boundary(boundary, source_grid)
    discretization = SplineMirrorDiscretization.build(config, elements=3)
    result = solve_spline_fixed_boundary_cli(
        discretization.fit_state(initial, source_grid),
        discretization.fit_boundary(boundary, source_grid),
        discretization,
        config,
        axial_flux_derivative=0.1,
        current_derivative=1.0e-3 * jnp.asarray(source_grid.s),
        solve_lambda=True,
        gradient_tolerance=1.0e-12,
        require_convergence=True,
    )
    evaluated = result.evaluated
    surface_integrals = np.einsum(
        "j,k,ijk->i",
        discretization.grid.theta_basis.weights,
        discretization.grid.axial_basis.weights,
        np.asarray(evaluated.state.lambda_stream),
    )

    assert evaluated.converged
    assert float(evaluated.variational.maximum) <= config.ftol
    assert float(evaluated.staggered_weak_force.maximum) <= 1.1 * config.ftol
    assert float(jnp.max(jnp.abs(evaluated.state.lambda_stream))) > 1.0e-3
    np.testing.assert_allclose(surface_integrals, 0.0, atol=3.0e-15)
    np.testing.assert_allclose(evaluated.state.lambda_stream[0], evaluated.state.lambda_stream[1])


def test_spline_coefficient_preconditioner_inverts_tensor_model() -> None:
    config = MirrorConfig(resolution=MirrorResolution(ns=9, mpol=2, ntheta=7, nxi=9))
    discretization = SplineMirrorDiscretization.build(config, elements=8)
    derivative = np.asarray(
        discretization.spline.basis_matrix(discretization.grid.axial_basis.nodes, derivative=1)
    ) / float(discretization.grid.dz_dxi)
    weights = np.asarray(discretization.grid.axial_basis.weights)
    interior = derivative[:, 1:-1]
    stiffness = interior.T @ (weights[:, None] * interior)
    preconditioner = SeparableMirrorPreconditioner.build_from_axial_stiffness(discretization.grid, stiffness)
    exact = np.random.default_rng(32).normal(size=preconditioner.size)

    np.testing.assert_allclose(
        preconditioner.apply(preconditioner.operator(exact)),
        exact,
        rtol=4.0e-12,
        atol=4.0e-12,
    )


@pytest.mark.full
def test_large_spline_solve_uses_matrix_free_coefficient_preconditioner() -> None:
    config = MirrorConfig(
        resolution=MirrorResolution(ns=29, mpol=1, ntheta=3, nxi=25),
        z_min=-1.2,
        z_max=1.2,
        ftol=1.0e-12,
        max_iterations=300,
    )
    source_grid = config.build_grid()
    boundary = MirrorBoundary.from_radius(0.3, source_grid)
    base = MirrorState.from_boundary(boundary, source_grid)
    s = jnp.asarray(source_grid.s)[:, None, None]
    xi = jnp.asarray(source_grid.xi)[None, None, :]
    initial = MirrorState(
        base.radius_scale + 0.03 * s * (1.0 - s) * (1.0 - xi**2),
        base.lambda_stream,
    )
    discretization = SplineMirrorDiscretization.build(config, elements=12)
    assert (source_grid.ns - 2) * source_grid.ntheta * (discretization.coefficient_count - 2) > 1024

    result = solve_spline_fixed_boundary_cli(
        discretization.fit_state(initial, source_grid),
        discretization.fit_boundary(boundary, source_grid),
        discretization,
        config,
        axial_flux_derivative=0.1,
        gradient_tolerance=1.0e-12,
        require_convergence=True,
    ).evaluated

    assert result.converged
    assert result.linear_iterations > 0
    assert result.final_linear_residual < 1.0e-5
    assert float(result.variational.maximum) <= config.ftol
    assert float(result.staggered_weak_force.maximum) <= 1.1 * config.ftol
    np.testing.assert_allclose(result.state.radius_scale, 0.3, atol=7.0e-15)


@pytest.mark.full
def test_knot_refined_rotating_ellipse_uses_matrix_free_rescue() -> None:
    from vmec_jax.mirror.analytic import RotatingEllipseParaxial

    config = MirrorConfig(
        resolution=MirrorResolution(ns=5, mpol=2, ntheta=12, nxi=17),
        ftol=1.0e-12,
        max_iterations=2000,
    )
    source_grid = config.build_grid()
    theta = jnp.asarray(source_grid.theta)[:, None]
    z = jnp.asarray(source_grid.z)[None, :]
    discretization = SplineMirrorDiscretization.build(config, elements=6)

    def boundary(stage):
        fixture = RotatingEllipseParaxial(
            mirror_strength=0.2 * stage,
            elongation=1.0 + 0.5 * stage,
            rotation=0.5 * jnp.pi * stage,
        )
        return MirrorBoundary.from_radius(
            fixture.boundary_radius(0.05, theta, z), source_grid
        )

    source = discretization.fit_boundary(boundary(0.0), source_grid)
    state = discretization.fit_state(
        MirrorState.from_boundary(boundary(0.0), source_grid), source_grid
    )
    result = None
    for stage in (0.0, 0.25, 0.5):
        target = discretization.fit_boundary(boundary(stage), source_grid)
        state = discretization.transfer_boundary(state, source, target)
        result = solve_spline_fixed_boundary_cli(
            state,
            target,
            discretization,
            config,
            axial_flux_derivative=0.01,
            solve_lambda=True,
            gradient_tolerance=1.0e-12,
            require_convergence=True,
        )
        state, source = result.coefficient_state, target

    assert result is not None
    assert result.evaluated.linear_iterations > 0
    assert float(result.evaluated.variational.maximum) <= config.ftol
    assert float(result.evaluated.staggered_weak_force.maximum) <= config.ftol


@pytest.mark.full
def test_finite_beta_spline_knot_refinement_converges_to_chebyshev() -> None:
    config = MirrorConfig(
        resolution=MirrorResolution(ns=5, mpol=0, ntheta=1, nxi=17),
        ftol=1.0e-12,
        max_iterations=1000,
    )
    source_grid = config.build_grid()
    s = jnp.asarray(source_grid.s)
    xi = jnp.asarray(source_grid.xi)
    boundary = MirrorBoundary.from_radius(0.3 * (1.0 + 0.12 * (1.0 - xi**2)), source_grid)
    initial = MirrorState.from_boundary(boundary, source_grid)
    solve_kwargs = {
        "axial_flux_derivative": 0.1,
        "mass_profile": 2.0e3 * (1.0 - s),
        "current_derivative": 3.0e-2 * s,
        "solve_lambda": True,
        "gradient_tolerance": 1.0e-12,
        "require_convergence": True,
    }
    reference = solve_fixed_boundary_cli(initial, boundary, source_grid, config, **solve_kwargs)
    energy_errors = []
    volume_errors = []
    for elements in (2, 4, 8):
        discretization = SplineMirrorDiscretization.build(config, elements=elements)
        result = solve_spline_fixed_boundary_cli(
            discretization.fit_state(initial, source_grid),
            discretization.fit_boundary(boundary, source_grid),
            discretization,
            config,
            **solve_kwargs,
        ).evaluated
        energy_errors.append(abs(float(result.energy.total / reference.energy.total) - 1.0))
        volume_errors.append(abs(float(result.energy.geometry.volume / reference.energy.geometry.volume) - 1.0))
        assert float(result.variational.maximum) < 1.0e-14
        assert float(result.staggered_weak_force.maximum) < 1.0e-14

    assert np.all(np.diff(energy_errors) < 0.0)
    assert np.all(np.diff(volume_errors) < 0.0)
    assert energy_errors[-1] < 1.0e-7
    assert volume_errors[-1] < 3.0e-6
