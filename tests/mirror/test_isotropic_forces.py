"""M2 isotropic energy, constraints, and physical-force diagnostics."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.sparse.linalg import LinearOperator, gmres

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from vmec_jax.mirror import (  # noqa: E402
    MirrorBoundary,
    MirrorConfig,
    MirrorResolution,
    MirrorState,
    solve_fixed_boundary_cli,
)
from vmec_jax.mirror.forces import (  # noqa: E402
    MU0,
    fixed_boundary_energy_gradient,
    fixed_boundary_variational_residual,
    isotropic_force_residual,
    isotropic_staggered_energy_gradient,
    isotropic_staggered_fixed_boundary_gradient,
    isotropic_staggered_weak_residual,
    mass_profile_from_pressure,
    mirror_energy,
    staggered_field_strength,
)
from vmec_jax.mirror.solver import (  # noqa: E402
    MirrorConvergenceError,
    SeparableMirrorPreconditioner,
    _valid_energy_objective,
)
from vmec_jax.mirror.model import project_fixed_boundary_state  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _enable_solver_jit():
    """Exercise nonlinear solver tests in their production execution mode."""

    previous = bool(jax.config.jax_disable_jit)
    jax.config.update("jax_disable_jit", False)
    yield
    jax.config.update("jax_disable_jit", previous)


def _cylinder(*, ns: int = 11, nxi: int = 21, radius: float = 0.3, half_length: float = 1.2):
    grid = MirrorConfig(
        resolution=MirrorResolution(ns=ns, mpol=0, ntheta=1, nxi=nxi),
        z_min=-half_length,
        z_max=half_length,
    ).build_grid()
    boundary = MirrorBoundary.from_radius(radius, grid)
    return grid, boundary, MirrorState.from_boundary(boundary, grid)


def test_fixed_boundary_projection_enforces_geometry_and_lambda_gauge() -> None:
    grid = MirrorConfig(resolution=MirrorResolution(ns=7, mpol=2, ntheta=7, nxi=13)).build_grid()
    theta = jnp.asarray(grid.theta)[:, None]
    xi = jnp.asarray(grid.xi)[None, :]
    boundary = MirrorBoundary.from_radius(0.3 * (1.0 + 0.05 * jnp.cos(2.0 * theta) * xi**2), grid)
    rng = np.random.default_rng(17)
    state = MirrorState(
        radius_scale=jnp.asarray(0.2 + 0.1 * rng.random(grid.shape)),
        lambda_stream=jnp.asarray(1.0 + rng.normal(size=grid.shape)),
    )
    projected = project_fixed_boundary_state(state, boundary, grid)
    np.testing.assert_allclose(projected.radius_scale[-1], boundary.radius_scale)
    np.testing.assert_allclose(
        projected.radius_scale[:, :, 0],
        np.broadcast_to(np.asarray(boundary.radius_scale[:, 0]), (grid.ns, grid.ntheta)),
    )
    np.testing.assert_allclose(projected.radius_scale[0], projected.radius_scale[1])
    np.testing.assert_allclose(projected.lambda_stream[0], projected.lambda_stream[1])
    surface_mean = np.einsum(
        "j,k,ijk->i",
        grid.theta_basis.weights,
        grid.axial_basis.weights,
        np.asarray(projected.lambda_stream),
    ) / (4.0 * np.pi)
    np.testing.assert_allclose(surface_mean, 0.0, atol=1.0e-15)


def test_vacuum_cylinder_has_exact_energy_and_negligible_physical_force() -> None:
    radius, half_length, psi_prime = 0.3, 1.2, 0.1
    grid, boundary, state = _cylinder(radius=radius, half_length=half_length)
    energy = mirror_energy(state, grid, axial_flux_derivative=psi_prime)
    expected_b = 2.0 * psi_prime / radius**2
    expected_volume = 2.0 * np.pi * half_length * radius**2
    expected_energy = expected_b**2 * expected_volume / (2.0 * MU0)
    np.testing.assert_allclose(energy.magnetic, expected_energy, rtol=3.0e-14, atol=3.0e-9)
    np.testing.assert_allclose(energy.pressure_energy, 0.0)

    residual = isotropic_force_residual(energy, grid)
    assert float(residual.normalized_rms) < 2.0e-13
    assert float(residual.bulk_normalized_rms) < 2.0e-13
    assert float(residual.axis_normalized_rms) < 2.0e-13
    np.testing.assert_allclose(residual.component_rms[1:], 0.0, atol=2.0e-14)

    gradient = fixed_boundary_energy_gradient(
        state,
        boundary,
        grid,
        axial_flux_derivative=psi_prime,
    )
    assert float(jnp.max(jnp.abs(gradient.radius_scale))) / expected_energy < 5.0e-14
    np.testing.assert_allclose(gradient.lambda_stream, 0.0, atol=2.0e-12)


def test_mass_profile_recovers_reference_isotropic_pressure() -> None:
    pressure = 2.5e4
    grid, _, state = _cylinder()
    vacuum = mirror_energy(state, grid, axial_flux_derivative=0.1)
    mass = mass_profile_from_pressure(pressure, vacuum.volume_derivative)
    finite_beta = mirror_energy(
        state,
        grid,
        axial_flux_derivative=0.1,
        mass_profile=mass,
    )
    np.testing.assert_allclose(finite_beta.pressure, pressure, rtol=2.0e-14, atol=2.0e-10)
    expected = pressure * float(vacuum.volume_derivative[0]) / (5.0 / 3.0 - 1.0)
    np.testing.assert_allclose(finite_beta.pressure_energy, expected, rtol=3.0e-14, atol=3.0e-9)
    assert float(isotropic_force_residual(finite_beta, grid).normalized_rms) < 2.0e-13


def test_optimizer_merit_rejects_crossed_flux_surfaces() -> None:
    config = MirrorConfig(resolution=MirrorResolution(ns=5, nxi=7))
    grid = config.build_grid()
    boundary = MirrorBoundary.from_radius(0.3, grid)
    valid = MirrorState.from_boundary(boundary, grid)
    valid_energy = mirror_energy(valid, grid, axial_flux_derivative=0.1)
    invalid = replace(valid, radius_scale=valid.radius_scale.at[1].set(1.2))
    invalid_energy = mirror_energy(invalid, grid, axial_flux_derivative=0.1)

    assert not bool(valid_energy.geometry.jacobian_sign_changed)
    assert bool(invalid_energy.geometry.jacobian_sign_changed)
    np.testing.assert_allclose(_valid_energy_objective(valid_energy, float(valid_energy.total)), 1.0)
    assert np.isinf(float(_valid_energy_objective(invalid_energy, 1.0)))


def test_staggered_field_strength_is_exact_for_uniform_cylinder() -> None:
    grid, _, state = _cylinder(ns=7, nxi=13)
    mod_b = staggered_field_strength(state, grid, axial_flux_derivative=0.1)

    np.testing.assert_allclose(mod_b, mod_b[0, 0, 0], rtol=3.0e-15, atol=3.0e-15)


def test_energy_gradient_matches_central_difference_for_interior_shape() -> None:
    grid, boundary, base = _cylinder(ns=9, nxi=17)
    s = jnp.asarray(grid.s)[:, None, None]
    xi = jnp.asarray(grid.xi)[None, None, :]
    perturbation = s * (1.0 - s) * (1.0 - xi**2) * (1.0 + 0.2 * xi)
    state = replace(base, radius_scale=base.radius_scale + 0.015 * perturbation)
    direction = MirrorState(radius_scale=perturbation, lambda_stream=jnp.zeros_like(perturbation))

    kwargs = {"axial_flux_derivative": 0.1}
    gradient = fixed_boundary_energy_gradient(state, boundary, grid, **kwargs)
    directional_ad = jnp.vdot(gradient.radius_scale, direction.radius_scale)
    directional_ad += jnp.vdot(gradient.lambda_stream, direction.lambda_stream)

    def objective(alpha):
        trial = MirrorState(
            radius_scale=state.radius_scale + alpha * direction.radius_scale,
            lambda_stream=state.lambda_stream,
        )
        projected = project_fixed_boundary_state(trial, boundary, grid)
        return mirror_energy(projected, grid, **kwargs).total

    epsilon = 2.0e-6
    directional_fd = (objective(epsilon) - objective(-epsilon)) / (2.0 * epsilon)
    np.testing.assert_allclose(directional_ad, directional_fd, rtol=2.0e-7, atol=2.0e-5)


def test_staggered_first_variation_matches_autodiff_for_3d_finite_beta() -> None:
    config = MirrorConfig(
        resolution=MirrorResolution(ns=7, mpol=2, ntheta=7, nxi=11),
        z_min=-1.3,
        z_max=1.1,
    )
    grid = config.build_grid()
    theta = jnp.asarray(grid.theta)[None, :, None]
    xi = jnp.asarray(grid.xi)[None, None, :]
    s = jnp.asarray(grid.s)[:, None, None]
    radius = 0.31 * (1.0 + 0.04 * s * jnp.cos(2.0 * theta) * (1.0 - xi**2) + 0.03 * s * xi)
    lam = 0.006 * s * jnp.sin(theta) * (1.0 - xi**2)
    state = MirrorState(radius, lam)
    kwargs = {
        "axial_flux_derivative": jnp.linspace(0.09, 0.12, grid.ns),
        "current_derivative": jnp.linspace(0.01, 0.025, grid.ns),
        "mass_profile": 1.2e3 * (1.0 - jnp.asarray(grid.s)) ** 2,
    }
    automatic = jax.grad(lambda trial: mirror_energy(trial, grid, **kwargs).total)(state)
    staggered = isotropic_staggered_energy_gradient(state, grid, **kwargs)
    np.testing.assert_allclose(
        staggered.radius_scale,
        automatic.radius_scale,
        rtol=2.0e-12,
        atol=2.0e-8,
    )
    np.testing.assert_allclose(
        staggered.lambda_stream,
        automatic.lambda_stream,
        rtol=2.0e-12,
        atol=2.0e-8,
    )


def test_staggered_weak_force_matches_fixed_boundary_projection() -> None:
    config = MirrorConfig(
        resolution=MirrorResolution(ns=7, mpol=2, ntheta=7, nxi=11),
        z_min=-1.3,
        z_max=1.1,
    )
    grid = config.build_grid()
    theta = jnp.asarray(grid.theta)[:, None]
    xi = jnp.asarray(grid.xi)[None, :]
    boundary = MirrorBoundary.from_radius(
        0.31 * (1.0 + 0.03 * jnp.cos(2.0 * theta) * (1.0 - xi**2)),
        grid,
    )
    state = MirrorState.from_boundary(boundary, grid)
    s = jnp.asarray(grid.s)[:, None, None]
    state = replace(
        state,
        radius_scale=state.radius_scale + 0.01 * s * (1.0 - s) * (1.0 - xi[None] ** 2),
        lambda_stream=0.004 * s * jnp.sin(jnp.asarray(grid.theta))[None, :, None] * (1.0 - xi[None] ** 2),
    )
    kwargs = {
        "axial_flux_derivative": jnp.linspace(0.09, 0.12, grid.ns),
        "current_derivative": jnp.linspace(0.01, 0.025, grid.ns),
        "mass_profile": 1.2e3 * (1.0 - jnp.asarray(grid.s)) ** 2,
    }
    automatic = fixed_boundary_energy_gradient(state, boundary, grid, **kwargs)
    staggered = isotropic_staggered_fixed_boundary_gradient(
        state,
        boundary,
        grid,
        **kwargs,
    )
    np.testing.assert_allclose(
        staggered.radius_scale,
        automatic.radius_scale,
        rtol=3.0e-12,
        atol=3.0e-8,
    )
    np.testing.assert_allclose(
        staggered.lambda_stream,
        automatic.lambda_stream,
        rtol=3.0e-12,
        atol=3.0e-8,
    )
    weak = isotropic_staggered_weak_residual(state, boundary, grid, **kwargs)
    variational = fixed_boundary_variational_residual(
        state,
        boundary,
        grid,
        **kwargs,
    )
    np.testing.assert_allclose(weak.radius_rms, variational.radius_rms, rtol=3.0e-12)
    np.testing.assert_allclose(weak.lambda_rms, variational.lambda_rms, rtol=3.0e-12)
    np.testing.assert_allclose(weak.maximum, variational.maximum, rtol=3.0e-12)


def test_radial_gauss_quadrature_controls_lambda_checkerboard_mode() -> None:
    config = MirrorConfig(resolution=MirrorResolution(ns=15, mpol=1, ntheta=3, nxi=15))
    grid = config.build_grid()
    boundary = MirrorBoundary.from_radius(0.3, grid)
    base = MirrorState.from_boundary(boundary, grid)
    s = jnp.asarray(grid.s)[:, None, None]
    theta = jnp.asarray(grid.theta)[None, :, None]
    xi = jnp.asarray(grid.xi)[None, None, :]
    mode = 0.01 * jnp.cos(theta) * (1.0 - xi**2)
    alternating = (-1.0) ** jnp.arange(grid.ns)[:, None, None] * mode
    smooth = jnp.sin(jnp.pi * s) * mode
    baseline = mirror_energy(base, grid, axial_flux_derivative=0.1).total

    def excess_energy(lam):
        state = project_fixed_boundary_state(MirrorState(base.radius_scale, lam), boundary, grid)
        return mirror_energy(state, grid, axial_flux_derivative=0.1).total - baseline

    alternating_energy = excess_energy(alternating)
    smooth_energy = excess_energy(smooth)
    assert float(alternating_energy) > 0.2 * float(smooth_energy)
    assert float(alternating_energy) < 2.0 * float(smooth_energy)


def test_flared_tube_manufactured_lorentz_force_converges_spectrally() -> None:
    """Compare the continuum force to a closed-form divergence-free field."""

    relative_errors = []
    for nxi in (9, 13, 17):
        half_length = 1.4
        grid, _, _ = _cylinder(ns=9, nxi=nxi, half_length=half_length)
        xi = jnp.asarray(grid.xi)
        radius = 0.3 * (1.0 + 0.12 * xi**2)
        boundary = MirrorBoundary.from_radius(radius, grid)
        state = MirrorState.from_boundary(boundary, grid)
        flux = 0.1
        residual = isotropic_force_residual(mirror_energy(state, grid, axial_flux_derivative=flux), grid)

        radius_z = 0.3 * 0.24 * xi / half_length
        radius_zz = jnp.full_like(xi, 0.3 * 0.24 / half_length**2)
        logarithmic_slope = radius_z / radius
        curvature = radius_zz / radius - 3.0 * logarithmic_slope**2
        axial_field = 2.0 * flux / radius**2
        exact_covariant_s = 0.5 * radius**2 * axial_field**2 * curvature / MU0
        numerical = np.asarray(residual.covariant_s)[1:, 0, :]
        error = np.max(np.abs(numerical - np.asarray(exact_covariant_s)))
        relative_errors.append(error / np.max(np.abs(np.asarray(exact_covariant_s))))
        np.testing.assert_allclose(residual.covariant_xi[1:], 0.0, atol=3.0e-11)

    assert relative_errors[1] < 2.0e-3 * relative_errors[0]
    assert relative_errors[2] < 2.0e-3 * relative_errors[1]
    assert relative_errors[-1] < 3.0e-11


def test_separable_preconditioner_is_exact_for_its_model_and_reduces_gmres_work() -> None:
    grid, _, _ = _cylinder(ns=17, nxi=33)
    preconditioner = SeparableMirrorPreconditioner.build(grid)
    rng = np.random.default_rng(42)
    exact = rng.normal(size=preconditioner.size)
    right_hand_side = preconditioner.operator(exact)
    np.testing.assert_allclose(preconditioner.apply(right_hand_side), exact, rtol=2.0e-12, atol=2.0e-12)

    operator = LinearOperator(
        (preconditioner.size, preconditioner.size),
        matvec=preconditioner.operator,
        dtype=float,
    )
    inverse = LinearOperator(
        (preconditioner.size, preconditioner.size),
        matvec=preconditioner.apply,
        dtype=float,
    )
    iterations = {"plain": 0, "preconditioned": 0}
    plain, plain_info = gmres(
        operator,
        right_hand_side,
        rtol=1.0e-11,
        atol=0.0,
        callback=lambda _: iterations.__setitem__("plain", iterations["plain"] + 1),
        callback_type="pr_norm",
    )
    accelerated, accelerated_info = gmres(
        operator,
        right_hand_side,
        M=inverse,
        rtol=1.0e-11,
        atol=0.0,
        callback=lambda _: iterations.__setitem__("preconditioned", iterations["preconditioned"] + 1),
        callback_type="pr_norm",
    )
    assert plain_info == accelerated_info == 0
    assert iterations["plain"] >= 10
    assert iterations["preconditioned"] <= 2
    np.testing.assert_allclose(plain, exact, rtol=2.0e-9, atol=2.0e-9)
    np.testing.assert_allclose(accelerated, exact, rtol=2.0e-12, atol=2.0e-12)


def test_reference_solver_polishes_perturbed_cylinder_to_physical_ftol() -> None:
    config = MirrorConfig(
        resolution=MirrorResolution(ns=7, mpol=0, ntheta=1, nxi=9),
        z_min=-1.2,
        z_max=1.2,
        ftol=1.0e-12,
        max_iterations=300,
    )
    grid = config.build_grid()
    boundary = MirrorBoundary.from_radius(0.3, grid)
    base = MirrorState.from_boundary(boundary, grid)
    s = jnp.asarray(grid.s)[:, None, None]
    xi = jnp.asarray(grid.xi)[None, None, :]
    perturbation = 0.03 * s * (1.0 - s) * (1.0 - xi**2)
    initial = replace(base, radius_scale=base.radius_scale + perturbation)
    initial_force = isotropic_force_residual(mirror_energy(initial, grid, axial_flux_derivative=0.1), grid)
    assert float(initial_force.normalized_rms) > 1.0e-2

    result = solve_fixed_boundary_cli(
        initial,
        boundary,
        grid,
        config,
        axial_flux_derivative=0.1,
        gradient_tolerance=1.0e-12,
        require_convergence=True,
    )
    assert result.converged
    assert result.optimizer_success
    assert result.iterations > 0
    assert float(result.variational.maximum) <= config.ftol
    assert result.staggered_weak_force is not None
    assert float(result.staggered_weak_force.maximum) <= 1.1 * config.ftol
    assert result.history.shape[1] == 6
    assert float(result.history[-1, 4]) <= config.ftol
    np.testing.assert_allclose(result.state.radius_scale, 0.3, atol=2.0e-13)


def test_host_reference_closes_medium_system_above_old_dense_limit() -> None:
    config = MirrorConfig(
        resolution=MirrorResolution(ns=17, mpol=0, ntheta=1, nxi=41),
        z_min=-1.2,
        z_max=1.2,
        ftol=1.0e-12,
        max_iterations=300,
    )
    grid = config.build_grid()
    boundary = MirrorBoundary.from_radius(0.3, grid)
    base = MirrorState.from_boundary(boundary, grid)
    s = jnp.asarray(grid.s)[:, None, None]
    xi = jnp.asarray(grid.xi)[None, None, :]
    initial = replace(
        base,
        radius_scale=base.radius_scale + 0.03 * s * (1.0 - s) * (1.0 - xi**2),
    )
    active_size = (grid.ns - 2) * (grid.nxi - 2)
    assert 512 < active_size < 1024

    result = solve_fixed_boundary_cli(
        initial,
        boundary,
        grid,
        config,
        axial_flux_derivative=0.1,
        gradient_tolerance=1.0e-12,
        require_convergence=True,
    )
    assert result.converged
    assert result.linear_iterations == 0
    assert result.final_linear_residual == 0.0
    assert float(result.variational.maximum) <= config.ftol
    assert float(result.force.normalized_rms) < 1.0e-11


def test_reference_solver_raises_instead_of_returning_best_unconverged_state() -> None:
    config = MirrorConfig(
        resolution=MirrorResolution(ns=7, mpol=0, ntheta=1, nxi=9),
        ftol=1.0e-14,
        max_iterations=1,
    )
    grid = config.build_grid()
    boundary = MirrorBoundary.from_radius(0.3, grid)
    base = MirrorState.from_boundary(boundary, grid)
    s = jnp.asarray(grid.s)[:, None, None]
    xi = jnp.asarray(grid.xi)[None, None, :]
    initial = replace(
        base,
        radius_scale=base.radius_scale + 0.04 * s * (1.0 - s) * (1.0 - xi**2),
    )
    with pytest.raises(MirrorConvergenceError) as caught:
        solve_fixed_boundary_cli(
            initial,
            boundary,
            grid,
            config,
            axial_flux_derivative=0.1,
            require_convergence=True,
        )
    assert not caught.value.result.converged
    assert float(caught.value.result.variational.maximum) > config.ftol
