"""M3 pressure-closure consistency and anisotropy validity tests."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from vmec_jax.mirror import (  # noqa: E402
    BiMaxwellianPressureClosure,
    IsotropicPressureClosure,
    MirrorBoundary,
    MirrorConfig,
    MirrorResolution,
    MirrorState,
    TabulatedPressureClosure,
)
from vmec_jax.mirror.forces import (  # noqa: E402
    MU0,
    anisotropic_fixed_boundary_energy_gradient,
    anisotropic_force_residual,
    anisotropic_mirror_energy,
    interface_residual,
)
from vmec_jax.mirror.model import (  # noqa: E402
    anisotropy_indicators,
    project_fixed_boundary_state,
)
from vmec_jax.mirror.solver import solve_anisotropic_fixed_boundary_cli  # noqa: E402


def test_isotropic_closure_is_exact_limit_and_has_positive_indicators() -> None:
    closure = IsotropicPressureClosure(jnp.asarray([2.0e4, -1.5e4]))
    s = jnp.linspace(0.0, 1.0, 9)[:, None]
    b = jnp.linspace(0.4, 1.2, 11)[None, :]
    moments = closure.moments(s, b)
    expected = 2.0e4 - 1.5e4 * s + jnp.zeros_like(b)
    np.testing.assert_allclose(moments.parallel, expected)
    np.testing.assert_allclose(moments.perpendicular, expected)
    indicators = anisotropy_indicators(closure, s, b)
    np.testing.assert_allclose(indicators.sigma, 1.0 / MU0, rtol=2.0e-14)
    np.testing.assert_allclose(indicators.mirror_ellipticity, 1.0 / MU0, rtol=2.0e-14)
    assert bool(indicators.valid)


def test_bimaxwellian_form_factor_matches_animec_passing_formula() -> None:
    closure = BiMaxwellianPressureClosure(
        mass_coefficients=jnp.asarray([1.0e4, -5.0e3]),
        hot_fraction_coefficients=jnp.asarray([0.4]),
        temperature_ratio=0.3,
        critical_field=0.5,
    )
    b = jnp.asarray([0.55, 0.8, 1.1])
    normalized = b / 0.5
    expected = normalized / (1.0 - 0.3 * (1.0 - normalized))
    np.testing.assert_allclose(closure.form_factor(b), expected, rtol=2.0e-14, atol=2.0e-14)

    below = closure.form_factor(jnp.asarray([0.5 - 1.0e-8]))
    at = closure.form_factor(jnp.asarray([0.5]))
    assert np.isfinite(float(below[0]))
    np.testing.assert_allclose(below, at, rtol=3.0e-8, atol=3.0e-8)


def test_bimaxwellian_isotropic_passing_limit_and_parallel_force_identity() -> None:
    isotropic_passing = BiMaxwellianPressureClosure(
        mass_coefficients=jnp.asarray([1.2e4, -2.0e3]),
        hot_fraction_coefficients=jnp.asarray([0.25, -0.1]),
        temperature_ratio=1.0,
        critical_field=0.2,
    )
    s = jnp.linspace(0.0, 1.0, 7)
    b = jnp.linspace(0.4, 1.0, 7)
    moments = isotropic_passing.moments(s, b)
    np.testing.assert_allclose(isotropic_passing.form_factor(b), 1.0, rtol=2.0e-14)
    np.testing.assert_allclose(moments.parallel, moments.perpendicular, rtol=3.0e-14, atol=3.0e-10)

    anisotropic = BiMaxwellianPressureClosure(
        mass_coefficients=jnp.asarray([1.2e4, -2.0e3]),
        hot_fraction_coefficients=jnp.asarray([0.4]),
        temperature_ratio=0.25,
        critical_field=0.6,
    )
    b = jnp.linspace(0.35, 1.1, 9)
    s = jnp.full_like(b, 0.3)
    moments = anisotropic.moments(s, b)
    dp_db = jax.grad(lambda field: jnp.sum(anisotropic.parallel_pressure(s, field)))(b)
    np.testing.assert_allclose(
        moments.perpendicular,
        moments.parallel - b * dp_db,
        rtol=3.0e-14,
        atol=3.0e-10,
    )
    assert np.all(np.isfinite(np.asarray(moments.perpendicular)))


def test_tabulated_parallel_pressure_derives_perpendicular_moment() -> None:
    s_nodes = jnp.asarray([0.0, 0.4, 1.0])
    b_nodes = jnp.asarray([0.3, 0.7, 1.2])
    # p_parallel = 10000*(1-s) + 2000*B, exactly bilinear.
    values = 1.0e4 * (1.0 - s_nodes[:, None]) + 2.0e3 * b_nodes[None, :]
    closure = TabulatedPressureClosure(s_nodes, b_nodes, values)
    s = jnp.asarray([0.2, 0.6, 0.9])
    b = jnp.asarray([0.5, 0.8, 1.0])
    moments = closure.moments(s, b)
    np.testing.assert_allclose(moments.parallel, 1.0e4 * (1.0 - s) + 2.0e3 * b)
    np.testing.assert_allclose(moments.perpendicular, 1.0e4 * (1.0 - s), atol=2.0e-10)


def test_closure_coefficients_are_differentiable_leaves() -> None:
    s = jnp.asarray([0.2, 0.7])
    b = jnp.asarray([0.5, 0.9])

    def total_pressure(coefficients):
        closure = BiMaxwellianPressureClosure(
            mass_coefficients=coefficients,
            hot_fraction_coefficients=jnp.asarray([0.3]),
            temperature_ratio=0.4,
            critical_field=0.6,
        )
        return jnp.sum(closure.moments(s, b).parallel)

    derivative = jax.grad(total_pressure)(jnp.asarray([1.0e4, -2.0e3]))
    assert derivative.shape == (2,)
    assert np.all(np.isfinite(np.asarray(derivative)))
    assert np.linalg.norm(np.asarray(derivative)) > 0.0


def test_animec_energy_recovers_isotropic_passing_cylinder_limit() -> None:
    radius, half_length, flux = 0.3, 1.2, 0.1
    grid = MirrorConfig(
        resolution=MirrorResolution(ns=9, mpol=0, ntheta=1, nxi=17),
        z_min=-half_length,
        z_max=half_length,
    ).build_grid()
    boundary = MirrorBoundary.from_radius(radius, grid)
    state = MirrorState.from_boundary(boundary, grid)
    closure = BiMaxwellianPressureClosure(
        mass_coefficients=jnp.asarray([2.0e4]),
        hot_fraction_coefficients=jnp.asarray([0.25]),
        temperature_ratio=1.0,
        critical_field=0.1,
        gamma=0.0,
    )
    energy = anisotropic_mirror_energy(
        state,
        grid,
        closure,
        axial_flux_derivative=flux,
    )
    volume = 2.0 * np.pi * half_length * radius**2
    pressure = 2.0e4 * 1.25
    expected_magnetic = (2.0 * flux / radius**2) ** 2 * volume / (2.0 * MU0)
    np.testing.assert_allclose(energy.magnetic, expected_magnetic, rtol=3.0e-14)
    np.testing.assert_allclose(energy.pressure_energy, -pressure * volume, rtol=3.0e-14)
    np.testing.assert_allclose(
        energy.moments_half.parallel,
        energy.moments_half.perpendicular,
        rtol=3.0e-14,
        atol=3.0e-10,
    )
    assert bool(energy.indicators_half.valid)
    residual = anisotropic_force_residual(state, energy, grid, closure)
    assert float(residual.normalized_rms) < 3.0e-13
    assert float(residual.parallel_pressure_rms) < 1.0e-9


def test_animec_energy_shape_gradient_matches_central_difference() -> None:
    grid = MirrorConfig(
        resolution=MirrorResolution(ns=7, mpol=0, ntheta=1, nxi=13)
    ).build_grid()
    boundary = MirrorBoundary.from_radius(0.3 * (1.0 + 0.05 * jnp.asarray(grid.xi) ** 2), grid)
    state = MirrorState.from_boundary(boundary, grid)
    s = jnp.asarray(grid.s)[:, None, None]
    xi = jnp.asarray(grid.xi)[None, None, :]
    direction = s * (1.0 - s) * (1.0 - xi**2)
    closure = BiMaxwellianPressureClosure(
        mass_coefficients=jnp.asarray([1.5e4, -1.0e4]),
        hot_fraction_coefficients=jnp.asarray([0.3]),
        temperature_ratio=0.35,
        critical_field=0.8,
    )
    gradient = anisotropic_fixed_boundary_energy_gradient(
        state,
        boundary,
        grid,
        closure,
        axial_flux_derivative=0.1,
    )
    directional_ad = jnp.vdot(gradient.radius_scale, direction)

    def objective(alpha):
        trial = MirrorState(
            radius_scale=state.radius_scale + alpha * direction,
            lambda_stream=state.lambda_stream,
        )
        return anisotropic_mirror_energy(
            project_fixed_boundary_state(trial, boundary, grid),
            grid,
            closure,
            axial_flux_derivative=0.1,
        ).total

    epsilon = 2.0e-6
    directional_fd = (objective(epsilon) - objective(-epsilon)) / (2.0 * epsilon)
    np.testing.assert_allclose(directional_ad, directional_fd, rtol=3.0e-7, atol=3.0e-5)


def test_animec_energy_gradient_converges_to_weak_tensor_force_projection() -> None:
    previous_jit_setting = bool(jax.config.jax_disable_jit)
    jax.config.update("jax_disable_jit", False)
    try:
        relative_errors = []
        correlations = []
        for ns, nxi in ((7, 13), (9, 17), (13, 25)):
            config = MirrorConfig(
                resolution=MirrorResolution(ns=ns, mpol=0, ntheta=1, nxi=nxi),
                z_min=-1.2,
                z_max=1.2,
            )
            grid = config.build_grid()
            boundary = MirrorBoundary.from_radius(0.3, grid)
            base = MirrorState.from_boundary(boundary, grid)
            s = jnp.asarray(grid.s)[:, None, None]
            xi = jnp.asarray(grid.xi)[None, None, :]
            state = replace(
                base,
                radius_scale=base.radius_scale
                + 0.01 * s * (1.0 - s) * (1.0 - xi**2),
            )
            closure = BiMaxwellianPressureClosure(
                mass_coefficients=jnp.asarray([2.0e4, -2.0e4]),
                hot_fraction_coefficients=jnp.asarray([0.3]),
                temperature_ratio=0.4,
                critical_field=2.5,
            )
            gradient = anisotropic_fixed_boundary_energy_gradient(
                state,
                boundary,
                grid,
                closure,
                axial_flux_derivative=0.1,
            )
            energy = anisotropic_mirror_energy(
                state, grid, closure, axial_flux_derivative=0.1
            )
            force = anisotropic_force_residual(state, energy, grid, closure)
            active = np.s_[1:-1, :, 1:-1]
            discrete = np.asarray(gradient.radius_scale[active]).ravel()
            continuum = np.asarray(
                force.radius_variation_projection[active]
            ).ravel()
            relative_errors.append(
                np.linalg.norm(discrete - continuum) / np.linalg.norm(discrete)
            )
            correlations.append(
                np.vdot(discrete, continuum)
                / (np.linalg.norm(discrete) * np.linalg.norm(continuum))
            )
    finally:
        jax.config.update("jax_disable_jit", previous_jit_setting)

    assert relative_errors[0] > relative_errors[1] > relative_errors[2]
    assert relative_errors[-1] < 3.0e-2
    assert correlations[0] > 0.999
    assert correlations[-1] > 0.9998


def test_finite_beta_bimaxwellian_cylinder_solves_both_force_routes() -> None:
    previous_jit_setting = bool(jax.config.jax_disable_jit)
    jax.config.update("jax_disable_jit", False)
    try:
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
        initial = replace(
            base,
            radius_scale=base.radius_scale + 0.02 * s * (1.0 - s) * (1.0 - xi**2),
        )
        closure = BiMaxwellianPressureClosure(
            mass_coefficients=jnp.asarray([2.0e4]),
            hot_fraction_coefficients=jnp.asarray([0.3]),
            temperature_ratio=0.4,
            critical_field=2.5,
        )
        result = solve_anisotropic_fixed_boundary_cli(
            initial,
            boundary,
            grid,
            config,
            closure,
            axial_flux_derivative=0.1,
            gradient_tolerance=1.0e-12,
            require_convergence=True,
        )
    finally:
        jax.config.update("jax_disable_jit", previous_jit_setting)

    beta_parallel = (
        2.0
        * MU0
        * result.energy.moments_half.parallel
        / result.energy.b_squared_half
    )
    assert result.converged
    assert float(result.variational.maximum) <= config.ftol
    assert float(result.force.normalized_rms) < 1.0e-11
    assert float(result.force.parallel_pressure_rms) < 1.0e-9
    assert bool(result.energy.indicators_half.valid)
    assert 5.0e-3 < float(jnp.mean(beta_parallel)) < 5.0e-2
    assert float(
        jnp.max(
            jnp.abs(
                result.energy.moments_half.parallel
                - result.energy.moments_half.perpendicular
            )
        )
    ) > 100.0
    np.testing.assert_allclose(result.state.radius_scale, 0.3, atol=2.0e-13)


def test_anisotropic_tensor_divergence_satisfies_parallel_force_balance() -> None:
    grid = MirrorConfig(
        resolution=MirrorResolution(ns=9, mpol=2, ntheta=7, nxi=21),
        z_min=-1.0,
        z_max=1.0,
    ).build_grid()
    theta = jnp.asarray(grid.theta)[:, None]
    xi = jnp.asarray(grid.xi)[None, :]
    boundary = MirrorBoundary.from_radius(
        0.3 * (1.0 + 0.05 * xi**2 + 0.03 * jnp.cos(2.0 * theta) * (1.0 - xi**2)),
        grid,
    )
    state = MirrorState.from_boundary(boundary, grid)
    closure = BiMaxwellianPressureClosure(
        mass_coefficients=jnp.asarray([2.0e4, -1.0e4]),
        hot_fraction_coefficients=jnp.asarray([0.3]),
        temperature_ratio=0.3,
        critical_field=0.8,
    )
    energy = anisotropic_mirror_energy(
        state,
        grid,
        closure,
        axial_flux_derivative=0.1,
    )
    residual = anisotropic_force_residual(state, energy, grid, closure)
    assert float(residual.parallel_pressure_rms) < 1.0e-7
    assert np.all(np.isfinite(np.asarray(residual.force_xyz)))
    # This prescribed shaped state is intentionally not an equilibrium; only
    # the closure's parallel force identity should vanish before solving.
    assert float(residual.normalized_rms) > 1.0e-3


def test_anisotropic_interface_residual_recognizes_exact_pressure_balance() -> None:
    ntheta, nxi = 7, 13
    pressure = jnp.full((ntheta, nxi), 2.0e4)
    plasma_b_squared = jnp.full((ntheta, nxi), 1.1**2)
    vacuum_b_squared = plasma_b_squared + 2.0 * MU0 * pressure
    zeros = jnp.zeros_like(pressure)
    residual = interface_residual(
        perpendicular_pressure=pressure,
        plasma_b_squared=plasma_b_squared,
        vacuum_b_squared=vacuum_b_squared,
        plasma_b_normal=zeros,
        vacuum_b_normal=zeros,
        theta_weights=jnp.full(ntheta, 2.0 * np.pi / ntheta),
        axial_weights=jnp.full(nxi, 2.0 / nxi),
    )
    np.testing.assert_allclose(residual.normal_stress_jump, 0.0, atol=2.0e-10)
    assert float(residual.normal_stress_rms) < 2.0e-15
    assert float(residual.plasma_b_normal_rms) == 0.0
    assert float(residual.vacuum_b_normal_rms) == 0.0

    perturbed = interface_residual(
        perpendicular_pressure=pressure,
        plasma_b_squared=plasma_b_squared,
        vacuum_b_squared=vacuum_b_squared * 1.01,
        plasma_b_normal=jnp.full_like(pressure, 1.0e-3),
        vacuum_b_normal=jnp.full_like(pressure, -2.0e-3),
        theta_weights=jnp.full(ntheta, 2.0 * np.pi / ntheta),
        axial_weights=jnp.full(nxi, 2.0 / nxi),
    )
    assert float(perturbed.normal_stress_rms) > 1.0e-4
    assert float(perturbed.plasma_b_normal_rms) > 0.0
    assert float(perturbed.vacuum_b_normal_rms) > 0.0
