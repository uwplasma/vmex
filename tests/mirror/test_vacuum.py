"""M5 open-annulus vacuum geometry and scalar-potential tests."""

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
    MirrorState,
    build_vacuum_grid,
    evaluate_vacuum_field,
    evaluate_vacuum_geometry,
    external_field_from_coils,
    mass_profile_from_pressure,
    mirror_energy,
    solve_axisymmetric_free_boundary_cli,
    solve_vacuum_potential,
    solve_axisymmetric_beta_scan_cli,
    summarize_axisymmetric_beta_scan,
    vacuum_energy_functional,
    vacuum_laplacian,
)
from vmec_jax.core.coils import CoilSet, two_coil_on_axis_bz  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _enable_solver_jit():
    previous = bool(jax.config.jax_disable_jit)
    jax.config.update("jax_disable_jit", False)
    yield
    jax.config.update("jax_disable_jit", previous)


def _grid(*, ns: int = 7, nxi: int = 7):
    config = MirrorConfig(
        resolution=MirrorResolution(ns=ns, mpol=1, ntheta=3, nxi=nxi),
        z_min=-1.2,
        z_max=1.2,
        ftol=1.0e-12,
        max_iterations=500,
    )
    return config, build_vacuum_grid(config.build_grid(), nrho=ns)


def _two_end_coils() -> CoilSet:
    dofs = np.zeros((2, 3, 3))
    dofs[:, 0, 2] = 0.9
    dofs[:, 1, 1] = 0.9
    dofs[:, 2, 0] = np.asarray([-1.0, 1.0])
    return CoilSet(
        base_curve_dofs=jnp.asarray(dofs),
        base_currents=jnp.asarray([2.0e5, 2.0e5]),
        n_segments=64,
    )


def test_cylindrical_annulus_has_exact_volume_metric_and_normal() -> None:
    config, grid = _grid(ns=9, nxi=11)
    inner, outer = 0.3, 0.7
    boundary = MirrorBoundary.from_radius(inner, grid)
    geometry = evaluate_vacuum_geometry(boundary, grid, outer_radius=outer)
    expected_volume = np.pi * (outer**2 - inner**2) * (
        config.z_max - config.z_min
    )
    np.testing.assert_allclose(geometry.volume, expected_volume, rtol=3.0e-14)
    np.testing.assert_allclose(
        geometry.sqrt_g,
        jnp.linalg.norm(geometry.xyz[..., :2], axis=-1)
        * (outer - inner)
        * grid.dz_dxi,
        rtol=3.0e-14,
        atol=3.0e-14,
    )
    theta = np.asarray(grid.theta)
    expected_normal = np.stack(
        [np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=-1
    )[:, None, :]
    expected_normal = np.broadcast_to(expected_normal, geometry.inner_normal_xyz.shape)
    np.testing.assert_allclose(geometry.inner_normal_xyz, expected_normal, atol=2.0e-14)
    assert bool(geometry.valid)


def test_linear_harmonic_potentials_have_zero_laplacian_and_exact_gradient() -> None:
    _, grid = _grid(ns=9, nxi=11)
    geometry = evaluate_vacuum_geometry(
        MirrorBoundary.from_radius(0.3, grid), grid, outer_radius=0.7
    )
    potential = geometry.xyz[..., 0]
    laplacian = vacuum_laplacian(potential, geometry, grid)
    np.testing.assert_allclose(laplacian[1:-1, :, 1:-1], 0.0, atol=3.0e-11)
    field = evaluate_vacuum_field(
        potential, geometry, grid, jnp.zeros_like(geometry.xyz)
    )
    expected = np.zeros(geometry.xyz.shape)
    expected[..., 0] = 1.0
    np.testing.assert_allclose(field.correction_xyz, expected, atol=3.0e-13)


def test_scalar_potential_solve_recovers_uniform_field_cancellation() -> None:
    config, grid = _grid()
    boundary = MirrorBoundary.from_radius(0.3, grid)
    geometry = evaluate_vacuum_geometry(boundary, grid, outer_radius=0.7)
    exact_potential = geometry.xyz[..., 0]
    external = jnp.zeros_like(geometry.xyz).at[..., 0].set(-1.0)
    result = solve_vacuum_potential(
        boundary,
        grid,
        config,
        external,
        exact_potential,
        outer_radius=0.7,
        initial_potential=0.0,
        boundary_condition="fixed_potential",
    )
    free = np.s_[:-1, :, 1:-1]
    assert result.converged
    assert float(result.variational_max) <= config.ftol
    assert float(result.linear_residual) < 1.0e-11
    assert float(result.laplacian_rms) < 1.0e-9
    assert float(result.b_normal_rms) < 1.0e-10
    np.testing.assert_allclose(result.potential[free], exact_potential[free], atol=2.0e-11)
    np.testing.assert_allclose(result.field.total_xyz, 0.0, atol=2.0e-11)


def test_direct_coil_field_on_annulus_is_jittable_and_current_differentiable() -> None:
    _, grid = _grid(ns=5, nxi=5)
    geometry = evaluate_vacuum_geometry(
        MirrorBoundary.from_radius(0.3, grid), grid, outer_radius=0.7
    )
    def field_norm(currents):
        coils = _two_end_coils().with_arrays(base_currents=currents)
        field = external_field_from_coils(coils, geometry)
        return jnp.mean(jnp.sum(field**2, axis=-1))

    currents = jnp.asarray([2.0e5, 2.0e5])
    value = jax.jit(field_norm)(currents)
    derivative = jax.grad(field_norm)(currents)
    assert float(value) > 0.0
    assert np.all(np.isfinite(np.asarray(derivative)))
    assert np.all(np.asarray(derivative) > 0.0)


def test_vacuum_operator_is_reciprocal_for_shaped_annulus() -> None:
    _, grid = _grid(ns=5, nxi=7)
    theta = jnp.asarray(grid.theta)[:, None]
    xi = jnp.asarray(grid.axial_basis.nodes)[None, :]
    boundary = MirrorBoundary.from_radius(
        0.3 * (1.0 + 0.04 * jnp.cos(theta) * (1.0 - xi**2)), grid
    )
    geometry = evaluate_vacuum_geometry(boundary, grid, outer_radius=0.7)
    external = external_field_from_coils(_two_end_coils(), geometry)
    rng = np.random.default_rng(91)
    point = jnp.asarray(rng.normal(size=grid.shape))
    left = jnp.asarray(rng.normal(size=grid.shape))
    right = jnp.asarray(rng.normal(size=grid.shape))

    def energy(potential):
        return vacuum_energy_functional(potential, geometry, grid, external)

    gradient = jax.grad(energy)
    h_left = jax.jvp(gradient, (point,), (left,))[1]
    h_right = jax.jvp(gradient, (point,), (right,))[1]
    np.testing.assert_allclose(
        jnp.vdot(left, h_right), jnp.vdot(right, h_left), rtol=2.0e-12
    )


def test_two_coil_vacuum_solve_reduces_plasma_normal_field_under_refinement() -> None:
    config = MirrorConfig(
        resolution=MirrorResolution(ns=7, mpol=0, ntheta=1, nxi=13),
        z_min=-0.8,
        z_max=0.8,
        ftol=1.0e-12,
        max_iterations=500,
    )
    grid = build_vacuum_grid(config.build_grid(), nrho=7)
    boundary = MirrorBoundary.from_radius(0.25, grid)
    geometry = evaluate_vacuum_geometry(boundary, grid, outer_radius=0.65)
    external = external_field_from_coils(_two_end_coils(), geometry)
    external_normal = jnp.sum(
        external[0] * geometry.inner_normal_xyz, axis=-1
    )[:, 1:-1]
    field_scale = jnp.sqrt(
        jnp.mean(jnp.sum(external[0, :, 1:-1] ** 2, axis=-1))
    )
    initial_normal_rms = jnp.sqrt(jnp.mean(external_normal**2))
    result = solve_vacuum_potential(
        boundary,
        grid,
        config,
        external,
        jnp.zeros(grid.shape),
        outer_radius=0.65,
    )

    assert result.converged
    assert float(result.variational_max) <= config.ftol
    assert float(result.b_normal_rms / field_scale) < 5.0e-3
    assert float(result.b_normal_rms / initial_normal_rms) < 5.0e-2
    assert float(
        jnp.sqrt(jnp.mean(result.field.correction_normal_outer[:, 1:-1] ** 2))
        / field_scale
    ) < 6.0e-3
    assert float(
        jnp.sqrt(
            jnp.mean(
                result.field.correction_normal_lower**2
                + result.field.correction_normal_upper**2
            )
        )
        / field_scale
    ) < 3.0e-3


def test_two_coil_free_boundary_beta_scan_uses_solved_expanding_surfaces() -> None:
    config = MirrorConfig(
        resolution=MirrorResolution(ns=5, mpol=0, ntheta=1, nxi=7),
        z_min=-0.8,
        z_max=0.8,
        ftol=1.0e-12,
        max_iterations=1000,
    )
    plasma_grid = config.build_grid()
    vacuum_grid = build_vacuum_grid(plasma_grid, nrho=5)
    on_axis_field = two_coil_on_axis_bz(
        jnp.asarray(plasma_grid.z),
        coil_radius=0.9,
        separation=2.0,
        current=2.0e5,
    )
    center = plasma_grid.nxi // 2
    flux = 0.5 * on_axis_field[center] * 0.25**2
    boundary = MirrorBoundary.from_axis_field(flux, on_axis_field, plasma_grid)
    results = solve_axisymmetric_beta_scan_cli(
        boundary,
        plasma_grid,
        vacuum_grid,
        config,
        _two_end_coils(),
        jnp.asarray([0.0, 0.01, 0.03, 0.10]),
        outer_radius=0.65,
        axial_flux_derivative=flux,
        reference_field=float(on_axis_field[center]),
    )
    center_radii = np.asarray(
        [result.boundary.radius_scale[0, center] for result in results]
    )
    diagnostics = summarize_axisymmetric_beta_scan(
        results,
        jnp.asarray([0.0, 0.01, 0.03, 0.10]),
        plasma_grid,
        reference_field=float(on_axis_field[center]),
    )

    assert len(results) == 4
    assert all(result.converged for result in results)
    assert all(float(result.variational_max) <= config.ftol for result in results)
    assert all(float(result.interface.normal_stress_rms) < 1.0e-12 for result in results)
    assert np.all(np.diff(center_radii) > 0.0)
    assert center_radii[-1] > 1.005 * center_radii[0]
    assert float(diagnostics[-1].achieved_reference_beta) > 0.09
    assert 0.03 < float(diagnostics[-1].volume_averaged_beta) < 0.04
    assert float(diagnostics[-1].diamagnetic_field_ratio) < 0.97
    assert abs(float(diagnostics[-1].paraxial_relative_error)) < 0.01
    assert all(result.vacuum_potential.shape == vacuum_grid.shape for result in results)
    assert all(float(result.vacuum_potential[-1, 0, 0]) == 0.0 for result in results)
    assert all(np.all(np.isfinite(np.asarray(result.vacuum_field.total_xyz))) for result in results)


@pytest.mark.py311_slow_coverage
def test_free_boundary_beta_observables_converge_with_resolution() -> None:
    summaries = []
    tangency = []
    for ns, nxi, nrho in ((5, 7, 5), (7, 13, 7), (9, 17, 9)):
        config = MirrorConfig(
            resolution=MirrorResolution(ns=ns, mpol=0, ntheta=1, nxi=nxi),
            z_min=-0.8,
            z_max=0.8,
            ftol=1.0e-12,
            max_iterations=2000,
        )
        plasma_grid = config.build_grid()
        vacuum_grid = build_vacuum_grid(plasma_grid, nrho=nrho)
        on_axis_field = two_coil_on_axis_bz(
            jnp.asarray(plasma_grid.z), coil_radius=0.9, separation=2.0, current=2.0e5
        )
        center = int(np.argmin(np.abs(plasma_grid.z)))
        flux = 0.5 * on_axis_field[center] * 0.25**2
        results = solve_axisymmetric_beta_scan_cli(
            MirrorBoundary.from_axis_field(flux, on_axis_field, plasma_grid),
            plasma_grid,
            vacuum_grid,
            config,
            _two_end_coils(),
            jnp.asarray([0.0, 0.10]),
            outer_radius=0.65,
            axial_flux_derivative=flux,
            reference_field=float(on_axis_field[center]),
        )
        summary = summarize_axisymmetric_beta_scan(
            results,
            jnp.asarray([0.0, 0.10]),
            plasma_grid,
            reference_field=float(on_axis_field[center]),
        )[-1]
        summaries.append(
            np.asarray([summary.center_radius, summary.center_axis_field, summary.achieved_reference_beta])
        )
        tangency.append(float(results[-1].interface.vacuum_b_normal_rms))
        assert all(float(result.variational_max) <= config.ftol for result in results)

    relative_change = np.abs((summaries[-1] - summaries[-2]) / summaries[-1])
    assert np.max(relative_change) < 1.0e-4
    assert np.all(np.diff(tangency) < 0.0)
    assert tangency[-1] < 0.002


@pytest.mark.py311_slow_coverage
def test_free_boundary_solution_is_independent_of_free_side_initial_radius() -> None:
    config = MirrorConfig(
        resolution=MirrorResolution(ns=7, mpol=0, ntheta=1, nxi=13),
        z_min=-0.8,
        z_max=0.8,
        ftol=1.0e-12,
        max_iterations=2000,
    )
    plasma_grid = config.build_grid()
    vacuum_grid = build_vacuum_grid(plasma_grid, nrho=7)
    on_axis_field = two_coil_on_axis_bz(
        jnp.asarray(plasma_grid.z), coil_radius=0.9, separation=2.0, current=2.0e5
    )
    center = int(np.argmin(np.abs(plasma_grid.z)))
    flux = 0.5 * on_axis_field[center] * 0.25**2
    reference_boundary = MirrorBoundary.from_axis_field(flux, on_axis_field, plasma_grid)
    reference_energy = mirror_energy(
        MirrorState.from_boundary(reference_boundary, plasma_grid),
        plasma_grid,
        axial_flux_derivative=flux,
    )
    central_pressure = 0.10 * float(on_axis_field[center]) ** 2 / (2.0 * 4.0e-7 * np.pi)
    mass = mass_profile_from_pressure(
        central_pressure * (1.0 - jnp.asarray(plasma_grid.s)),
        reference_energy.volume_derivative,
    )
    results = []
    for amplitude in (-0.10, 0.10):
        radius = np.asarray(reference_boundary.radius_scale) * (
            1.0 + amplitude * (1.0 - np.asarray(plasma_grid.xi)[None, :] ** 2)
        )
        results.append(
            solve_axisymmetric_free_boundary_cli(
                MirrorBoundary(jnp.asarray(radius)),
                plasma_grid,
                vacuum_grid,
                config,
                _two_end_coils(),
                outer_radius=0.65,
                axial_flux_derivative=flux,
                mass_profile=mass,
                require_convergence=True,
            )
        )

    np.testing.assert_allclose(results[0].boundary.radius_scale, results[1].boundary.radius_scale, atol=2.0e-12)
    np.testing.assert_allclose(results[0].plasma_energy.b_squared, results[1].plasma_energy.b_squared, rtol=2.0e-11)
    assert all(float(result.variational_max) <= config.ftol for result in results)
