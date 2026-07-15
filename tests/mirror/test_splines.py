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
)
from vmec_jax.mirror.forces import (  # noqa: E402
    isotropic_force_residual,
    isotropic_staggered_energy_gradient,
    mirror_energy,
)
from vmec_jax.mirror.free_boundary import (  # noqa: E402
    _SplineFreeBoundaryVectorizer,
    _spline_boundary_work,
    _spline_boundary_work_residual,
)
from vmec_jax.mirror.geometry import (  # noqa: E402
    contravariant_field,
    divergence_b,
    evaluate_closed_spline_axis,
    evaluate_closed_geometry,
    evaluate_geometry,
    magnetic_field_squared,
    magnetic_field_xyz,
    racetrack_centerline_coefficients,
)
from vmec_jax.mirror.solver import MirrorConvergenceError  # noqa: E402
from vmec_jax.mirror.splines import (  # noqa: E402
    SplineMirrorBoundary,
    SplineMirrorDiscretization,
    SplineMirrorState,
    _SplineStateVectorizer,
    _packed_spline_preconditioner,
    initialize_from_cartesian_field,
    initialize_closed_vacuum_stream_function,
    solve_fixed_boundary_cli as solve_spline_fixed_boundary_cli,
    trace_closed_field_line,
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
        lambda values: (
            evaluate_closed_spline_axis(
                values,
                basis,
                points,
                initial_normal=jnp.asarray([0.0, 1.0, 0.0]),
            ).arc_length
        )
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


def test_closed_circular_surface_recovers_torus_volume_and_field_metric() -> None:
    major_radius = 2.0
    minor_radius = 0.23
    resolution = MirrorResolution(ns=9, mpol=0, nxi=4)
    discretization = SplineMirrorDiscretization.build_closed(
        resolution,
        coefficient_count=24,
        quadrature_order=4,
    )
    basis = discretization.spline
    nodes = jnp.asarray(basis.collocation_nodes)
    axis_samples = jnp.stack(
        (
            major_radius * jnp.cos(nodes),
            jnp.zeros_like(nodes),
            major_radius * jnp.sin(nodes),
        ),
        axis=-1,
    )
    axis_coefficients = basis.fit(axis_samples, axis=0)
    axis = evaluate_closed_spline_axis(
        axis_coefficients,
        basis,
        discretization.grid.z,
        initial_normal=jnp.asarray([0.0, 1.0, 0.0]),
    )
    boundary = SplineMirrorBoundary(jnp.full((1, basis.size), minor_radius))
    coefficient_state = SplineMirrorState(
        radius_coefficients=jnp.full(
            (resolution.ns, 1, basis.size),
            minor_radius,
        ),
        lambda_coefficients=jnp.zeros((resolution.ns, 1, basis.size)),
    )
    state = discretization.evaluate_state(discretization.project_fixed_boundary(coefficient_state, boundary))
    geometry = evaluate_closed_geometry(state, discretization.grid, axis)
    expected_volume = 2.0 * np.pi**2 * major_radius * minor_radius**2
    np.testing.assert_allclose(geometry.volume, expected_volume, rtol=2.0e-5)
    assert not bool(geometry.jacobian_sign_changed)

    field = contravariant_field(
        state,
        geometry,
        discretization.grid,
        axial_flux_derivative=0.03,
    )
    np.testing.assert_allclose(
        divergence_b(field, geometry, discretization.grid)[1:],
        0.0,
        atol=2.0e-14,
    )
    cartesian = magnetic_field_xyz(field, geometry)
    np.testing.assert_allclose(
        jnp.sum(cartesian**2, axis=-1),
        magnetic_field_squared(field, geometry),
        rtol=3.0e-14,
        atol=3.0e-14,
    )

    direction = jnp.reshape(jnp.linspace(-0.4, 0.6, axis_coefficients.size), axis_coefficients.shape)

    def total_energy(coefficients):
        trial_axis = evaluate_closed_spline_axis(
            coefficients,
            basis,
            discretization.grid.z,
            initial_normal=jnp.asarray([0.0, 1.0, 0.0]),
        )
        return mirror_energy(
            state,
            discretization.grid,
            axial_flux_derivative=0.03,
            axis=trial_axis,
        ).total

    automatic = jax.jvp(total_energy, (axis_coefficients,), (direction,))[1]
    step = 2.0e-5
    finite_difference = (
        total_energy(axis_coefficients + step * direction) - total_energy(axis_coefficients - step * direction)
    ) / (2.0 * step)
    np.testing.assert_allclose(automatic, finite_difference, rtol=2.0e-7)


def test_racetrack_ellipse_rotates_ninety_degrees_between_straight_legs() -> None:
    semi_major, semi_minor = 0.18, 0.12
    resolution = MirrorResolution(ns=7, mpol=6, nxi=4)
    discretization = SplineMirrorDiscretization.build_closed(
        resolution,
        coefficient_count=32,
        quadrature_order=4,
    )
    basis = discretization.spline
    axis_coefficients = racetrack_centerline_coefficients(
        basis.size,
        straight_length=6.0,
        return_radius=1.0,
    )
    axis = evaluate_closed_spline_axis(
        axis_coefficients,
        basis,
        discretization.grid.z,
        initial_normal=jnp.asarray([0.0, 1.0, 0.0]),
    )

    coefficient_nodes = jnp.asarray(basis.collocation_nodes)
    angle = 0.25 * jnp.pi * (1.0 - jnp.cos(coefficient_nodes))
    theta = jnp.asarray(discretization.grid.theta)[:, None]
    local_angle = theta - angle[None, :]
    radius_samples = (
        semi_major
        * semi_minor
        / jnp.sqrt((semi_minor * jnp.cos(local_angle)) ** 2 + (semi_major * jnp.sin(local_angle)) ** 2)
    )
    boundary = SplineMirrorBoundary(basis.fit(radius_samples, axis=-1))
    boundary_at_legs = basis.evaluate(
        boundary.radius_coefficients,
        jnp.asarray([0.0, jnp.pi]),
        axis=-1,
    )
    np.testing.assert_allclose(boundary_at_legs[0, 0], semi_major, rtol=2.0e-13)
    np.testing.assert_allclose(boundary_at_legs[0, 1], semi_minor, rtol=2.0e-13)

    coefficient_state = SplineMirrorState(
        radius_coefficients=jnp.broadcast_to(
            boundary.radius_coefficients[None],
            (resolution.ns,) + boundary.radius_coefficients.shape,
        ),
        lambda_coefficients=jnp.zeros((resolution.ns, resolution.ntheta, basis.size)),
    )
    state = discretization.evaluate_state(discretization.project_fixed_boundary(coefficient_state, boundary))
    geometry = evaluate_closed_geometry(state, discretization.grid, axis)
    np.testing.assert_allclose(
        geometry.volume,
        np.pi * semi_major * semi_minor * axis.arc_length,
        rtol=3.0e-4,
    )
    assert not bool(geometry.jacobian_sign_changed)

    volume_gradient = jax.grad(
        lambda coefficients: (
            evaluate_closed_geometry(
                state,
                discretization.grid,
                evaluate_closed_spline_axis(
                    coefficients,
                    basis,
                    discretization.grid.z,
                    initial_normal=jnp.asarray([0.0, 1.0, 0.0]),
                ),
            ).volume
        )
    )(axis_coefficients)
    assert np.all(np.isfinite(volume_gradient))
    assert float(jnp.linalg.norm(volume_gradient)) > 1.0e-3


def _closed_circular_torus(resolution, *, coefficient_count=8):
    discretization = SplineMirrorDiscretization.build_closed(
        resolution,
        coefficient_count=coefficient_count,
        quadrature_order=3,
    )
    basis = discretization.spline
    nodes = jnp.asarray(basis.collocation_nodes)
    major_radius = 2.5
    axis = evaluate_closed_spline_axis(
        basis.fit(
            jnp.stack(
                (
                    major_radius * jnp.cos(nodes),
                    jnp.zeros_like(nodes),
                    major_radius * jnp.sin(nodes),
                ),
                axis=-1,
            ),
            axis=0,
        ),
        basis,
        discretization.grid.z,
        initial_normal=jnp.asarray([0.0, 1.0, 0.0]),
    )
    minor_radius = 0.25
    boundary = SplineMirrorBoundary(jnp.full((resolution.ntheta, basis.size), minor_radius))
    radius = jnp.full(
        (resolution.ns, resolution.ntheta, basis.size),
        minor_radius,
    )
    return (
        discretization,
        axis,
        boundary,
        SplineMirrorState(
            radius,
            jnp.zeros_like(radius),
        ),
    )


def test_closed_field_line_recovers_constant_iota_and_derivative() -> None:
    resolution = MirrorResolution(ns=5, mpol=4, nxi=4)
    discretization, axis, _, state = _closed_circular_torus(resolution)
    evaluated = discretization.evaluate_state(state)
    geometry = evaluate_closed_geometry(evaluated, discretization.grid, axis)
    flux = 0.02

    def iota(current):
        field = contravariant_field(
            evaluated,
            geometry,
            discretization.grid,
            axial_flux_derivative=flux,
            current_derivative=current,
        )
        return trace_closed_field_line(
            field,
            discretization,
            radial_index=2,
            theta0=0.3,
            turns=3,
            steps_per_turn=64,
        ).iota

    current = 1.0e-3
    np.testing.assert_allclose(iota(current), current / flux, rtol=2.0e-13)
    np.testing.assert_allclose(jax.grad(iota)(current), 1.0 / flux, rtol=2.0e-13)


def test_closed_vacuum_initializer_recovers_one_over_r_field() -> None:
    resolution = MirrorResolution(ns=7, mpol=6, nxi=4)
    discretization, axis, _, zero = _closed_circular_torus(resolution)
    initialized = initialize_closed_vacuum_stream_function(
        zero,
        discretization,
        axis,
        axial_flux_derivative=0.03,
    )
    zero_energy = mirror_energy(
        discretization.evaluate_state(zero),
        discretization.grid,
        axis=axis,
        axial_flux_derivative=0.03,
    )
    energy = mirror_energy(
        discretization.evaluate_state(initialized),
        discretization.grid,
        axis=axis,
        axial_flux_derivative=0.03,
    )
    cylindrical_radius = jnp.sqrt(energy.geometry.xyz[..., 0] ** 2 + energy.geometry.xyz[..., 2] ** 2)
    invariant = jnp.sqrt(energy.b_squared) * cylindrical_radius
    relative_spread = (jnp.max(invariant[1:], axis=(1, 2)) - jnp.min(invariant[1:], axis=(1, 2))) / jnp.mean(
        invariant[1:], axis=(1, 2)
    )
    zero_invariant = jnp.sqrt(zero_energy.b_squared) * cylindrical_radius
    zero_spread = (jnp.max(zero_invariant[-1]) - jnp.min(zero_invariant[-1])) / jnp.mean(zero_invariant[-1])
    surface_integral = jnp.einsum(
        "j,k,ijk->i",
        discretization.grid.theta_basis.weights,
        discretization.grid.axial_basis.weights,
        discretization.evaluate_state(initialized).lambda_stream,
    )

    assert float(energy.total) < float(zero_energy.total)
    assert float(jnp.max(relative_spread)) < 2.0e-3
    assert float(zero_spread) > 0.15
    np.testing.assert_allclose(surface_integral, 0.0, atol=2.0e-17)
    assert float(jnp.max(jnp.abs(initialized.lambda_coefficients))) > 1.0e-3


def test_closed_vacuum_strong_force_converges_at_axis() -> None:
    """Refine the physical force for a current-free circular torus."""

    first_row_residuals = []
    for ns in (5, 9, 17):
        resolution = MirrorResolution(ns=ns, mpol=5, nxi=12)
        discretization, axis, _, zero = _closed_circular_torus(
            resolution,
            coefficient_count=12,
        )
        evaluated_zero = discretization.evaluate_state(zero)
        geometry = evaluate_closed_geometry(evaluated_zero, discretization.grid, axis)
        axial_weights = jnp.asarray(discretization.grid.axial_basis.weights)
        theta_weights = jnp.asarray(discretization.grid.theta_basis.weights)
        metric_weight = jnp.einsum(
            "ijk,k->ij",
            geometry.sqrt_g / geometry.g_xixi,
            axial_weights,
        ) / jnp.sum(axial_weights)
        surface_mean = jnp.einsum("ij,j->i", metric_weight, theta_weights) / jnp.sum(theta_weights)
        flux = 0.02 * surface_mean / surface_mean[0]
        initialized = initialize_closed_vacuum_stream_function(
            zero,
            discretization,
            axis,
            axial_flux_derivative=flux,
        )
        state = discretization.evaluate_state(initialized)
        energy = mirror_energy(
            state,
            discretization.grid,
            axis=axis,
            axial_flux_derivative=flux,
        )
        residual = isotropic_force_residual(
            energy,
            discretization.grid,
            state=state,
            axis=axis,
            closed=True,
            axial_flux_derivative=flux,
        )
        first_row_residuals.append(float(residual.first_row_normalized_rms))

    ratios = np.asarray(first_row_residuals[:-1]) / np.asarray(first_row_residuals[1:])
    np.testing.assert_allclose(ratios, 4.0, rtol=0.03)
    assert first_row_residuals[-1] < 6.0e-7


def test_closed_staggered_first_variation_matches_autodiff() -> None:
    resolution = MirrorResolution(ns=5, mpol=3, nxi=4)
    discretization, axis, _, _ = _closed_circular_torus(resolution)
    grid = discretization.grid
    s = jnp.asarray(grid.s)[:, None, None]
    theta = jnp.asarray(grid.theta)[None, :, None]
    xi = jnp.asarray(grid.xi)[None, None, :]
    state = MirrorState(
        0.25 * (1.0 + 0.03 * s * jnp.cos(2.0 * theta) + 0.01 * s * jnp.cos(xi)),
        0.004 * s * jnp.sin(theta) * jnp.cos(xi),
    )
    kwargs = {
        "axial_flux_derivative": jnp.linspace(0.02, 0.03, resolution.ns),
        "current_derivative": jnp.linspace(0.001, 0.002, resolution.ns),
        "mass_profile": 10.0 * (1.0 - jnp.asarray(grid.s)) ** 2,
        "axis": axis,
    }
    automatic = jax.grad(lambda trial: mirror_energy(trial, grid, **kwargs).total)(state)
    staggered = isotropic_staggered_energy_gradient(state, grid, **kwargs)

    np.testing.assert_allclose(
        staggered.radius_scale,
        automatic.radius_scale,
        rtol=3.0e-12,
        atol=1.0e-8,
    )
    np.testing.assert_allclose(
        staggered.lambda_stream,
        automatic.lambda_stream,
        rtol=3.0e-12,
        atol=1.0e-8,
    )


def test_closed_spline_fixed_boundary_torus_converges_to_ftol() -> None:
    resolution = MirrorResolution(ns=5, mpol=3, nxi=4)
    config = MirrorConfig(
        resolution=resolution,
        ftol=1.0e-12,
        max_iterations=1000,
    )
    discretization, axis, boundary, base = _closed_circular_torus(resolution)
    s = jnp.asarray(discretization.grid.s)[:, None, None]
    theta = jnp.asarray(discretization.grid.theta)[None, :, None]
    initial_radius = base.radius_coefficients
    initial_radius += 0.015 * s * (1.0 - s) * jnp.cos(theta)
    initial = initialize_closed_vacuum_stream_function(
        SplineMirrorState(initial_radius, base.lambda_coefficients),
        discretization,
        axis,
        axial_flux_derivative=0.03,
    )

    with jax.disable_jit(False):
        result = solve_spline_fixed_boundary_cli(
            initial,
            boundary,
            discretization,
            config,
            axial_flux_derivative=0.03,
            solve_lambda=True,
            axis=axis,
            require_convergence=True,
        ).evaluated

    assert result.converged
    assert result.iterations < 50
    assert float(result.variational.maximum) <= config.ftol
    assert float(result.staggered_weak_force.maximum) <= 1.1 * config.ftol
    assert float(result.normalized_divergence_rms) < 1.0e-12
    assert not bool(result.energy.geometry.jacobian_sign_changed)


@pytest.mark.full
def test_large_closed_torus_uses_cyclic_sparse_factor() -> None:
    resolution = MirrorResolution(ns=7, mpol=4, nxi=4)
    config = MirrorConfig(
        resolution=resolution,
        ftol=1.0e-12,
        max_iterations=1000,
    )
    discretization, axis, boundary, base = _closed_circular_torus(
        resolution,
        coefficient_count=12,
    )
    initial = initialize_closed_vacuum_stream_function(
        base,
        discretization,
        axis,
        axial_flux_derivative=0.03,
    )
    vectorizer = _SplineStateVectorizer.build(
        initial,
        boundary,
        discretization,
        axial_flux_derivative=0.03,
        solve_lambda=True,
    )
    assert vectorizer.pack().size > 1024

    with jax.disable_jit(False):
        result = solve_spline_fixed_boundary_cli(
            initial,
            boundary,
            discretization,
            config,
            axial_flux_derivative=0.03,
            solve_lambda=True,
            axis=axis,
            require_convergence=True,
        ).evaluated

    assert result.converged
    assert float(result.variational.maximum) <= config.ftol
    assert result.linear_iterations < 2000
    assert result.final_linear_residual < 1.0e-8


def test_closed_racetrack_finite_current_and_lambda_converge() -> None:
    resolution = MirrorResolution(ns=5, mpol=4, nxi=4)
    config = MirrorConfig(
        resolution=resolution,
        ftol=1.0e-12,
        max_iterations=1000,
    )
    discretization = SplineMirrorDiscretization.build_closed(
        resolution,
        coefficient_count=16,
        quadrature_order=3,
    )
    basis = discretization.spline
    axis = evaluate_closed_spline_axis(
        racetrack_centerline_coefficients(
            basis.size,
            straight_length=6.0,
            return_radius=1.0,
        ),
        basis,
        discretization.grid.z,
        initial_normal=jnp.asarray([0.0, 1.0, 0.0]),
    )
    nodes = jnp.asarray(basis.collocation_nodes)
    angle = 0.25 * jnp.pi * (1.0 - jnp.cos(nodes))
    theta = jnp.asarray(discretization.grid.theta)[:, None]
    local_angle = theta - angle[None]
    semi_major, semi_minor = 0.18, 0.12
    samples = (
        semi_major
        * semi_minor
        / jnp.sqrt((semi_minor * jnp.cos(local_angle)) ** 2 + (semi_major * jnp.sin(local_angle)) ** 2)
    )
    boundary = SplineMirrorBoundary(basis.fit(samples, axis=-1))
    radius = jnp.broadcast_to(
        boundary.radius_coefficients[None],
        (resolution.ns,) + boundary.radius_coefficients.shape,
    )
    initial = SplineMirrorState(radius, jnp.zeros_like(radius))

    result = solve_spline_fixed_boundary_cli(
        initial,
        boundary,
        discretization,
        config,
        axial_flux_derivative=0.02,
        current_derivative=1.0e-3 * jnp.asarray(discretization.grid.s),
        solve_lambda=True,
        axis=axis,
        require_convergence=True,
    ).evaluated

    assert result.converged
    assert result.iterations < 100
    assert float(result.variational.maximum) <= config.ftol
    assert float(result.staggered_weak_force.maximum) <= 1.1 * config.ftol
    # The Clebsch field is analytically solenoidal; two independently applied
    # mixed derivative matrices leave an x64 commutator floor near 1e-12.
    assert float(result.normalized_divergence_rms) < 2.0e-12
    assert float(jnp.max(jnp.abs(result.state.lambda_stream))) > 1.0e-3
    assert not bool(result.energy.geometry.jacobian_sign_changed)
    field_line = trace_closed_field_line(
        result.energy.field,
        discretization,
        radial_index=resolution.ns - 1,
        theta0=0.2,
        turns=3,
        steps_per_turn=64,
    )
    assert abs(float(field_line.iota)) > 1.0e-3
    assert np.all(np.isfinite(field_line.theta))


def _spline_polynomial_state():
    config = MirrorConfig(
        resolution=MirrorResolution(ns=7, mpol=0, nxi=41),
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


def test_clamped_projection_preserves_prescribed_nested_cut_profiles() -> None:
    _, _, _, _, discretization, boundary, state = _spline_polynomial_state()
    radial_profile = jnp.linspace(0.82, 1.0, discretization.grid.ns)
    radius = state.radius_coefficients.at[:, :, 0].multiply(radial_profile[:, None])
    radius = radius.at[:, :, -1].multiply((2.0 - radial_profile)[:, None])
    projected = discretization.project_fixed_boundary(
        SplineMirrorState(radius, state.lambda_coefficients),
        boundary,
    )

    np.testing.assert_allclose(projected.radius_coefficients[-1], boundary.radius_coefficients)
    np.testing.assert_allclose(projected.radius_coefficients[1:-1, :, 0], radius[1:-1, :, 0])
    np.testing.assert_allclose(projected.radius_coefficients[1:-1, :, -1], radius[1:-1, :, -1])
    np.testing.assert_allclose(projected.radius_coefficients[0], projected.radius_coefficients[1])


def test_boundary_transfer_preserves_nested_self_similarity() -> None:
    config = MirrorConfig(resolution=MirrorResolution(ns=5, mpol=1, nxi=9))
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


def _free_boundary_spline_fixture():
    config = MirrorConfig(
        resolution=MirrorResolution(ns=5, mpol=1, nxi=9),
        z_min=-1.4,
        z_max=1.4,
    )
    discretization = SplineMirrorDiscretization.build(config, elements=3, quadrature_order=5)
    theta = jnp.asarray(discretization.grid.theta)[:, None]
    nodes = jnp.asarray(discretization.spline.collocation_nodes)[None, :]
    coefficients = 0.28 * (1.0 + 0.08 * jnp.cos(theta) * (1.0 - nodes**2))
    boundary = SplineMirrorBoundary(coefficients)
    radius = jnp.broadcast_to(
        coefficients,
        (discretization.grid.ns,) + coefficients.shape,
    )
    state = SplineMirrorState(radius, jnp.zeros_like(radius))
    return discretization, boundary, state


def test_spline_boundary_work_is_the_discrete_shape_derivative() -> None:
    discretization, boundary, _ = _free_boundary_spline_fixture()
    theta = jnp.asarray(discretization.grid.theta)[:, None]
    xi = jnp.asarray(discretization.grid.xi)[None, :]
    jump = 1.7 + 0.2 * jnp.cos(theta) * (1.0 - xi**2)
    direction = jnp.sin(jnp.arange(boundary.radius_coefficients.size, dtype=float)).reshape(
        boundary.radius_coefficients.shape
    )
    direction = direction.at[:, [0, -1]].set(0.0)
    weights = (
        jnp.asarray(discretization.grid.theta_basis.weights)[:, None]
        * jnp.asarray(discretization.grid.axial_basis.weights)[None, :]
    )

    def pressure_potential(coefficients):
        radius = discretization.evaluate_boundary(SplineMirrorBoundary(coefficients)).radius_scale
        return 0.5 * abs(float(discretization.grid.dz_dxi)) * jnp.sum(jump * radius**2 * weights)

    work = _spline_boundary_work(boundary, jump, discretization)
    derivative = jnp.vdot(work, direction)
    automatic = jnp.vdot(jax.grad(pressure_potential)(boundary.radius_coefficients), direction)
    step = 2.0e-6
    finite_difference = (
        pressure_potential(boundary.radius_coefficients + step * direction)
        - pressure_potential(boundary.radius_coefficients - step * direction)
    ) / (2.0 * step)

    np.testing.assert_allclose(derivative, automatic, rtol=3.0e-14, atol=3.0e-14)
    np.testing.assert_allclose(derivative, finite_difference, rtol=2.0e-10, atol=2.0e-10)


def test_spline_boundary_work_normalization_preserves_constant_stress_ratio() -> None:
    discretization, boundary, _ = _free_boundary_spline_fixture()
    shape = (discretization.grid.ntheta, discretization.grid.nxi)
    residual = _spline_boundary_work_residual(
        boundary,
        2.5 * jnp.ones(shape),
        5.0 * jnp.ones(shape),
        discretization,
    )

    np.testing.assert_allclose(residual[:, 1:-1], 0.5, rtol=2.0e-15, atol=2.0e-15)


def test_spline_free_boundary_map_is_square_and_preserves_constraints() -> None:
    discretization, boundary, state = _free_boundary_spline_fixture()
    vectorizer = _SplineFreeBoundaryVectorizer.build(
        boundary,
        state,
        discretization,
        axial_flux_derivative=0.1,
        solve_lambda=True,
        calibrate_pressure=True,
        initial_mass_scale=1.2,
    )
    varied = vectorizer.pack()
    varied[: vectorizer.boundary_size] *= 1.03
    varied_boundary, varied_state, mass_scale = vectorizer.unpack(varied)
    evaluated_boundary = discretization.evaluate_boundary(varied_boundary).radius_scale
    evaluated_state = discretization.evaluate_state(varied_state)

    assert vectorizer.size == vectorizer.boundary_size + vectorizer.state_size + 1
    assert vectorizer.boundary_size == discretization.grid.ntheta * (discretization.coefficient_count - 2)
    np.testing.assert_allclose(
        varied_boundary.radius_coefficients[:, [0, -1]],
        boundary.radius_coefficients[:, [0, -1]],
    )
    np.testing.assert_allclose(evaluated_state.radius_scale[-1], evaluated_boundary, atol=3.0e-15)
    np.testing.assert_allclose(evaluated_state.radius_scale[0], evaluated_state.radius_scale[1], atol=3.0e-15)
    np.testing.assert_allclose(
        jnp.einsum(
            "j,k,ijk->i",
            discretization.grid.theta_basis.weights,
            discretization.grid.axial_basis.weights,
            evaluated_state.lambda_stream,
        ),
        0.0,
        atol=3.0e-15,
    )
    np.testing.assert_allclose(mass_scale, 1.2)


def _perturbed_cylinder(config, elements, amplitude):
    """Return coefficient inputs for fixed-boundary solver tests."""

    grid = config.build_grid()
    boundary = MirrorBoundary.from_radius(0.3, grid)
    base = MirrorState.from_boundary(boundary, grid)
    s = jnp.asarray(grid.s)[:, None, None]
    xi = jnp.asarray(grid.xi)[None, None, :]
    initial = MirrorState(
        base.radius_scale + amplitude * s * (1.0 - s) * (1.0 - xi**2),
        base.lambda_stream,
    )
    discretization = SplineMirrorDiscretization.build(config, elements=elements)
    return (
        grid,
        discretization,
        discretization.fit_boundary(boundary, grid),
        discretization.fit_state(initial, grid),
    )


def test_spline_fixed_boundary_solver_recovers_cylindrical_equilibrium() -> None:
    config = MirrorConfig(
        resolution=MirrorResolution(ns=7, mpol=0, nxi=9),
        z_min=-1.2,
        z_max=1.2,
        ftol=1.0e-12,
        max_iterations=300,
    )
    _, discretization, spline_boundary, spline_initial = _perturbed_cylinder(config, 4, 0.03)

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


def test_spline_solver_raises_when_convergence_is_required() -> None:
    config = MirrorConfig(
        resolution=MirrorResolution(ns=7, mpol=0, nxi=9),
        ftol=1.0e-14,
        max_iterations=1,
    )
    _, discretization, spline_boundary, spline_initial = _perturbed_cylinder(config, 4, 0.04)

    with pytest.raises(MirrorConvergenceError) as caught:
        solve_spline_fixed_boundary_cli(
            spline_initial,
            spline_boundary,
            discretization,
            config,
            axial_flux_derivative=0.1,
            require_convergence=True,
        )

    assert not caught.value.result.converged
    assert float(caught.value.result.variational.maximum) > config.ftol


def test_spline_solver_converges_nonaxisymmetric_finite_current_state() -> None:
    config = MirrorConfig(
        resolution=MirrorResolution(ns=5, mpol=1, nxi=7),
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
    axis_field = np.sqrt(np.asarray(evaluated.energy.b_squared)[0])
    np.testing.assert_allclose(
        axis_field,
        np.broadcast_to(np.mean(axis_field, axis=0), axis_field.shape),
        rtol=2.0e-13,
    )


def test_supplied_field_initializer_recovers_straight_field_line_mirror() -> None:
    from vmec_jax.mirror.analytic import StraightFieldLineMirror

    config = MirrorConfig(
        resolution=MirrorResolution(ns=9, mpol=8, nxi=17),
        z_min=-1.0,
        z_max=1.0,
    )
    source_grid = config.build_grid()
    discretization = SplineMirrorDiscretization.build(config, elements=8)
    fixture = StraightFieldLineMirror(center_field=1.0, axial_scale=2.5)
    theta = jnp.asarray(source_grid.theta)[:, None]
    z = jnp.asarray(source_grid.z)[None, :]
    boundary = MirrorBoundary.from_radius(
        fixture.boundary_radius(0.03, theta, z),
        source_grid,
    )
    spline_boundary = discretization.fit_boundary(boundary, source_grid)
    initial = discretization.fit_state(
        MirrorState.from_boundary(boundary, source_grid),
        source_grid,
    )
    initialized = initialize_from_cartesian_field(
        initial,
        spline_boundary,
        discretization,
        fixture.field,
    )
    state = discretization.evaluate_state(initialized.state)
    energy = mirror_energy(
        state,
        discretization.grid,
        axial_flux_derivative=initialized.axial_flux_derivative,
    )
    points = energy.geometry.xyz.reshape((-1, 3))
    supplied = jax.vmap(fixture.field)(points).reshape(energy.geometry.xyz.shape)
    reconstructed = magnetic_field_xyz(energy.field, energy.geometry)
    field_error = jnp.linalg.norm((reconstructed - supplied)[1:])
    field_error /= jnp.linalg.norm(supplied[1:])
    normal = jnp.cross(
        energy.geometry.e_theta_xyz,
        energy.geometry.e_xi_xyz,
    )
    normal /= jnp.linalg.norm(normal, axis=-1, keepdims=True)
    tangency = jnp.sum(supplied * normal, axis=-1)
    tangency /= jnp.linalg.norm(supplied, axis=-1)
    tangency_rms = jnp.sqrt(jnp.mean(tangency[1:] ** 2))
    force = isotropic_force_residual(
        energy,
        discretization.grid,
        state=state,
        axial_flux_derivative=initialized.axial_flux_derivative,
    )
    sampled = initialize_from_cartesian_field(
        initial,
        spline_boundary,
        discretization,
        supplied,
    )
    _, flux_tangent = jax.jvp(
        lambda scale: (
            initialize_from_cartesian_field(
                initial,
                spline_boundary,
                discretization,
                scale * supplied,
            ).axial_flux_derivative
        ),
        (jnp.asarray(1.0),),
        (jnp.asarray(1.0),),
    )

    assert not bool(energy.geometry.jacobian_sign_changed)
    assert float(tangency_rms) < 2.0e-4
    assert float(field_error) < 5.0e-4
    assert float(force.normalized_rms) < 6.0e-3
    assert float(jnp.max(jnp.abs(initialized.state.lambda_coefficients))) > 1.0e-6
    assert np.all(np.asarray(initialized.axial_flux_derivative) > 0.0)
    np.testing.assert_allclose(
        initialized.axial_flux_derivative,
        0.5 * fixture.center_field * 0.03**2,
        rtol=2.0e-4,
    )
    np.testing.assert_allclose(
        flux_tangent,
        initialized.axial_flux_derivative,
        rtol=2.0e-12,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        sampled.state.lambda_coefficients,
        initialized.state.lambda_coefficients,
        rtol=2.0e-12,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        sampled.axial_flux_derivative,
        initialized.axial_flux_derivative,
        rtol=2.0e-12,
        atol=2.0e-15,
    )


@pytest.mark.full
def test_equal_end_axisymmetric_mirror_is_independent_of_cut_location() -> None:
    from vmec_jax.mirror.analytic import AxisymmetricPolynomialMirror

    fixture = AxisymmetricPolynomialMirror(mirror_strength=0.5)
    center_radius = []
    center_axis_field = []
    for half_length, elements in ((0.6, 6), (0.8, 8), (1.0, 10)):
        config = MirrorConfig(
            resolution=MirrorResolution(ns=9, mpol=0, nxi=17),
            z_min=-half_length,
            z_max=half_length,
            ftol=1.0e-12,
            max_iterations=1000,
        )
        discretization = SplineMirrorDiscretization.build(config, elements=elements)
        boundary_samples = fixture.boundary_radius(
            0.12,
            half_length * jnp.asarray(discretization.spline.collocation_nodes),
        )[None]
        boundary = SplineMirrorBoundary(discretization.spline.fit(boundary_samples, axis=-1))
        radius = jnp.broadcast_to(
            boundary.radius_coefficients[None],
            (config.resolution.ns,) + boundary.radius_coefficients.shape,
        )
        initialized = initialize_from_cartesian_field(
            SplineMirrorState(radius, jnp.zeros_like(radius)),
            boundary,
            discretization,
            fixture.field,
        )
        result = solve_spline_fixed_boundary_cli(
            initialized.state,
            boundary,
            discretization,
            config,
            axial_flux_derivative=initialized.axial_flux_derivative,
            gradient_tolerance=config.ftol,
            require_convergence=True,
        ).evaluated
        center = int(np.argmin(np.abs(discretization.grid.z)))
        center_radius.append(float(result.state.radius_scale[-1, 0, center]))
        center_axis_field.append(float(jnp.sqrt(result.energy.b_squared[0, 0, center])))
        assert result.final_linear_residual < 2.0e-9
        assert float(result.variational.maximum) <= config.ftol
        assert float(result.staggered_weak_force.maximum) <= config.ftol
        assert float(result.force.normalized_rms) < 6.0e-3
        assert float(result.force.bulk_normalized_rms) < 2.0e-3

    np.testing.assert_allclose(center_radius, center_radius[0], rtol=3.0e-8)
    np.testing.assert_allclose(center_axis_field, center_axis_field[0], rtol=2.0e-5)


def test_local_spline_preconditioner_builds_from_bounded_hessian_chunks() -> None:
    config = MirrorConfig(resolution=MirrorResolution(ns=5, mpol=2, nxi=9))
    grid = config.build_grid()
    boundary = MirrorBoundary.from_radius(0.3, grid)
    discretization = SplineMirrorDiscretization.build(config, elements=4)
    spline_boundary = discretization.fit_boundary(boundary, grid)
    state = discretization.fit_state(MirrorState.from_boundary(boundary, grid), grid)
    vectorizer = _SplineStateVectorizer.build(
        state,
        spline_boundary,
        discretization,
        axial_flux_derivative=0.1,
        solve_lambda=True,
    )
    _, _, build_local = _packed_spline_preconditioner(discretization, vectorizer)
    size = vectorizer.pack().size
    diagonal = np.linspace(1.0, 3.0, size)
    matrix = np.diag(diagonal)
    stream_row = slice(
        vectorizer.radius_size,
        vectorizer.radius_size + vectorizer.lambda_free_indices.size,
    )
    gauge_coupling = np.linspace(0.01, 0.03, vectorizer.lambda_free_indices.size)
    matrix[stream_row, stream_row] += np.outer(gauge_coupling, gauge_coupling)
    batch_sizes = []

    def matrix_columns(directions):
        batch_sizes.append(directions.shape[0])
        return directions @ matrix.T

    assert build_local is not None
    apply = build_local(matrix_columns)
    exact = np.random.default_rng(7).normal(size=size)

    np.testing.assert_allclose(apply(matrix @ exact), exact, rtol=3.0e-14, atol=3.0e-14)
    assert max(batch_sizes) <= 32
    assert sum(batch_sizes) >= size


def test_periodic_spline_preconditioner_factors_cyclic_support() -> None:
    resolution = MirrorResolution(ns=5, mpol=1, nxi=4)
    discretization, _, boundary, state = _closed_circular_torus(resolution, coefficient_count=8)
    vectorizer = _SplineStateVectorizer.build(
        state,
        boundary,
        discretization,
        axial_flux_derivative=0.03,
        solve_lambda=True,
    )
    _, _, build_local = _packed_spline_preconditioner(discretization, vectorizer)
    size = vectorizer.pack().size
    matrix = 4.0 * np.eye(size)
    first = np.flatnonzero(vectorizer.radius_indices[2] == 0)
    last = np.flatnonzero(vectorizer.radius_indices[2] == discretization.coefficient_count - 1)
    for left, right in zip(first, last, strict=True):
        matrix[left, right] = matrix[right, left] = 0.2

    assert build_local is not None
    apply = build_local(lambda directions: directions @ matrix.T)
    exact = np.random.default_rng(91).normal(size=size)
    np.testing.assert_allclose(apply(matrix @ exact), exact, rtol=3.0e-14, atol=3.0e-14)


@pytest.mark.full
def test_large_spline_solve_uses_matrix_free_coefficient_preconditioner() -> None:
    config = MirrorConfig(
        resolution=MirrorResolution(ns=29, mpol=1, nxi=25),
        z_min=-1.2,
        z_max=1.2,
        ftol=1.0e-12,
        max_iterations=300,
    )
    source_grid, discretization, spline_boundary, spline_initial = _perturbed_cylinder(config, 12, 0.03)
    assert (source_grid.ns - 2) * source_grid.ntheta * (discretization.coefficient_count - 2) > 1024

    result = solve_spline_fixed_boundary_cli(
        spline_initial,
        spline_boundary,
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
        resolution=MirrorResolution(ns=5, mpol=8, nxi=17),
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
        return MirrorBoundary.from_radius(fixture.boundary_radius(0.05, theta, z), source_grid)

    source = discretization.fit_boundary(boundary(0.0), source_grid)
    state = discretization.fit_state(MirrorState.from_boundary(boundary(0.0), source_grid), source_grid)
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
    assert result.evaluated.final_linear_residual < 1.0e-8
    assert float(result.evaluated.variational.maximum) <= config.ftol
    assert float(result.evaluated.staggered_weak_force.maximum) <= config.ftol
