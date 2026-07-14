"""M5 open-annulus vacuum geometry and scalar-potential tests."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from vmec_jax.mirror import (  # noqa: E402
    BiMaxwellianPressureClosure,
    MirrorBoundary,
    MirrorConfig,
    MirrorResolution,
    MirrorState,
    TabulatedPressureClosure,
    build_vacuum_grid,
    solve_beta_scan_cli,
)
from vmec_jax.mirror.continuation import solve_axisymmetric_beta_scan_cli  # noqa: E402
from vmec_jax.mirror.diagnostics import (  # noqa: E402
    boundary_fourier_amplitudes,
    boundary_fourier_norms,
    summarize_axisymmetric_beta_scan,
    summarize_nonaxisymmetric_beta_scan,
)
from vmec_jax.mirror.free_boundary import solve_axisymmetric_free_boundary_cli  # noqa: E402
from vmec_jax.mirror.restart import (  # noqa: E402
    FreeBoundaryRestart,
    load_free_boundary_restart,
    save_free_boundary_restart,
)
from vmec_jax.mirror.forces import MU0, mass_profile_from_pressure, mirror_energy  # noqa: E402
from vmec_jax.mirror.geometry import magnetic_field_squared  # noqa: E402
from vmec_jax.mirror.vacuum import (  # noqa: E402
    evaluate_vacuum_field,
    evaluate_vacuum_geometry,
    external_field_from_source,
    solve_vacuum_potential,
    vacuum_energy_functional,
    vacuum_laplacian,
)
from vmec_jax.core.mgrid import MgridData, MgridField  # noqa: E402


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


def test_free_boundary_restart_roundtrip_is_compact_and_grid_checked(tmp_path) -> None:
    config = MirrorConfig(resolution=MirrorResolution(ns=7, mpol=0, ntheta=1, nxi=9))
    plasma_grid = config.build_grid()
    vacuum_grid = build_vacuum_grid(plasma_grid, nrho=5)
    boundary = MirrorBoundary.from_radius(0.3, plasma_grid)
    state = MirrorState.from_boundary(boundary, plasma_grid)
    restart = FreeBoundaryRestart(
        boundary=boundary,
        plasma_state=state,
        vacuum_potential=jnp.zeros(vacuum_grid.shape),
        mass_scale=1.25,
    )

    path = save_free_boundary_restart(tmp_path / "beta_003", restart)
    loaded = load_free_boundary_restart(path, plasma_grid, vacuum_grid)

    assert path.suffix == ".npz"
    assert path.stat().st_size < 4096
    np.testing.assert_array_equal(loaded.boundary.radius_scale, boundary.radius_scale)
    np.testing.assert_array_equal(loaded.plasma_state.radius_scale, state.radius_scale)
    np.testing.assert_array_equal(loaded.plasma_state.lambda_stream, state.lambda_stream)
    np.testing.assert_array_equal(loaded.vacuum_potential, restart.vacuum_potential)
    assert loaded.mass_scale == restart.mass_scale

    mismatched = MirrorConfig(resolution=MirrorResolution(ns=9, mpol=0, ntheta=1, nxi=9)).build_grid()
    with pytest.raises(ValueError, match="plasma state"):
        load_free_boundary_restart(path, mismatched, build_vacuum_grid(mismatched, nrho=5))


def test_boundary_fourier_amplitudes_are_grid_independent() -> None:
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, 7, endpoint=False)
    axial = jnp.linspace(-1.0, 1.0, 5)
    radius = 0.2 + 0.01 * jnp.cos(theta)[:, None] * (1.0 - axial**2)[None, :] + 0.004 * jnp.sin(2.0 * theta)[:, None]
    amplitudes = boundary_fourier_amplitudes(MirrorBoundary(radius))

    np.testing.assert_allclose(amplitudes[0], 0.2, atol=3.0e-17)
    np.testing.assert_allclose(amplitudes[1], 0.01 * (1.0 - axial**2), atol=5.0e-17)
    np.testing.assert_allclose(amplitudes[2], 0.004, atol=5.0e-17)
    np.testing.assert_allclose(amplitudes[3], 0.0, atol=5.0e-17)


def test_boundary_fourier_norms_do_not_use_a_symmetry_zero() -> None:
    grid = MirrorConfig(
        resolution=MirrorResolution(ns=5, mpol=2, ntheta=7, nxi=9)
    ).build_grid()
    theta = jnp.asarray(grid.theta)[:, None]
    xi = jnp.asarray(grid.xi)[None, :]
    boundary = MirrorBoundary(0.2 + 0.03 * xi * jnp.cos(theta))

    l2, maximum = boundary_fourier_norms(boundary, grid)
    core_l2, core_maximum = boundary_fourier_norms(
        boundary, grid, central_fraction=0.75
    )

    np.testing.assert_allclose(l2[1], 0.03 / np.sqrt(3.0), rtol=2.0e-14)
    np.testing.assert_allclose(maximum[1], 0.03, rtol=2.0e-14)
    expected_interior = 0.03 * 0.75 / np.sqrt(3.0)
    np.testing.assert_allclose(core_l2[1], expected_interior, rtol=2.0e-14)
    np.testing.assert_allclose(core_maximum[1], 0.03 * 0.75, rtol=2.0e-14)
    np.testing.assert_allclose(boundary_fourier_amplitudes(boundary)[1, grid.nxi // 2], 0.0, atol=5e-17)


def _external_mirror_field(points):
    """Curl-free, divergence-free paraxial mirror field."""

    points = jnp.asarray(points)
    x, y, z = jnp.moveaxis(points, -1, 0)
    curvature = 0.02
    return jnp.stack(
        (
            -curvature * x * z,
            -curvature * y * z,
            0.08 + curvature * (z**2 - 0.5 * (x**2 + y**2)),
        ),
        axis=-1,
    )


def _on_axis_mirror_field(z, **_unused):
    return 0.08 + 0.02 * jnp.asarray(z) ** 2


def _nonaxisymmetric_mirror_field(points):
    field = _external_mirror_field(points)
    return field.at[..., 0].add(0.004)


def test_cylindrical_annulus_has_exact_volume_metric_and_normal() -> None:
    config, grid = _grid(ns=9, nxi=11)
    inner, outer = 0.3, 0.7
    boundary = MirrorBoundary.from_radius(inner, grid)
    geometry = evaluate_vacuum_geometry(boundary, grid, outer_radius=outer)
    expected_volume = np.pi * (outer**2 - inner**2) * (config.z_max - config.z_min)
    np.testing.assert_allclose(geometry.volume, expected_volume, rtol=3.0e-14)
    np.testing.assert_allclose(
        geometry.sqrt_g,
        jnp.linalg.norm(geometry.xyz[..., :2], axis=-1) * (outer - inner) * grid.dz_dxi,
        rtol=3.0e-14,
        atol=3.0e-14,
    )
    theta = np.asarray(grid.theta)
    expected_normal = np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=-1)[:, None, :]
    expected_normal = np.broadcast_to(expected_normal, geometry.inner_normal_xyz.shape)
    np.testing.assert_allclose(geometry.inner_normal_xyz, expected_normal, atol=2.0e-14)
    assert bool(geometry.valid)


def test_mgrid_external_field_is_converted_to_cartesian_on_annulus() -> None:
    _, grid = _grid(ns=5, nxi=7)
    geometry = evaluate_vacuum_geometry(MirrorBoundary.from_radius(0.3, grid), grid, outer_radius=0.7)
    shape = (1, 4, 5, 6)
    data = MgridData(
        rmin=0.0,
        rmax=1.0,
        zmin=-1.5,
        zmax=1.5,
        ir=6,
        jz=5,
        kp=4,
        nfp=1,
        nextcur=1,
        mgrid_mode="S",
        coil_groups=("uniform",),
        raw_coil_cur=(1.0,),
        br=np.full(shape, 0.2),
        bp=np.full(shape, 0.3),
        bz=np.full(shape, 0.4),
    )
    source = MgridField.from_mgrid_data(data, extcur=jnp.asarray([2.0]))
    actual = external_field_from_source(source, geometry)
    phi = jnp.arctan2(geometry.xyz[..., 1], geometry.xyz[..., 0])
    expected = jnp.stack(
        (
            0.4 * jnp.cos(phi) - 0.6 * jnp.sin(phi),
            0.4 * jnp.sin(phi) + 0.6 * jnp.cos(phi),
            jnp.full_like(phi, 0.8),
        ),
        axis=-1,
    )
    np.testing.assert_allclose(actual, expected, rtol=2.0e-14, atol=2.0e-14)


def test_linear_harmonic_potentials_have_zero_laplacian_and_exact_gradient() -> None:
    _, grid = _grid(ns=9, nxi=11)
    geometry = evaluate_vacuum_geometry(MirrorBoundary.from_radius(0.3, grid), grid, outer_radius=0.7)
    potential = geometry.xyz[..., 0]
    laplacian = vacuum_laplacian(potential, geometry, grid)
    np.testing.assert_allclose(laplacian[1:-1, :, 1:-1], 0.0, atol=3.0e-11)
    field = evaluate_vacuum_field(potential, geometry, grid, jnp.zeros_like(geometry.xyz))
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


def test_vacuum_operator_is_reciprocal_for_shaped_annulus() -> None:
    _, grid = _grid(ns=5, nxi=7)
    theta = jnp.asarray(grid.theta)[:, None]
    xi = jnp.asarray(grid.axial_basis.nodes)[None, :]
    boundary = MirrorBoundary.from_radius(0.3 * (1.0 + 0.04 * jnp.cos(theta) * (1.0 - xi**2)), grid)
    geometry = evaluate_vacuum_geometry(boundary, grid, outer_radius=0.7)
    external = external_field_from_source(_external_mirror_field, geometry)
    rng = np.random.default_rng(91)
    point = jnp.asarray(rng.normal(size=grid.shape))
    left = jnp.asarray(rng.normal(size=grid.shape))
    right = jnp.asarray(rng.normal(size=grid.shape))

    def energy(potential):
        return vacuum_energy_functional(potential, geometry, grid, external)

    gradient = jax.grad(energy)
    h_left = jax.jvp(gradient, (point,), (left,))[1]
    h_right = jax.jvp(gradient, (point,), (right,))[1]
    np.testing.assert_allclose(jnp.vdot(left, h_right), jnp.vdot(right, h_left), rtol=2.0e-12)


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
    external = external_field_from_source(_external_mirror_field, geometry)
    external_normal = jnp.sum(external[0] * geometry.inner_normal_xyz, axis=-1)[:, 1:-1]
    field_scale = jnp.sqrt(jnp.mean(jnp.sum(external[0, :, 1:-1] ** 2, axis=-1)))
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
    assert float(jnp.sqrt(jnp.mean(result.field.correction_normal_outer[:, 1:-1] ** 2)) / field_scale) < 6.0e-3
    assert (
        float(
            jnp.sqrt(jnp.mean(result.field.correction_normal_lower**2 + result.field.correction_normal_upper**2))
            / field_scale
        )
        < 3.0e-3
    )


def test_free_boundary_beta_scan_uses_solved_expanding_surfaces() -> None:
    config = MirrorConfig(
        resolution=MirrorResolution(ns=5, mpol=0, ntheta=1, nxi=7),
        z_min=-0.8,
        z_max=0.8,
        ftol=1.0e-12,
        max_iterations=1000,
    )
    plasma_grid = config.build_grid()
    vacuum_grid = build_vacuum_grid(plasma_grid, nrho=5)
    on_axis_field = _on_axis_mirror_field(
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
        _external_mirror_field,
        jnp.asarray([0.0, 0.01, 0.03, 0.10]),
        outer_radius=0.65,
        axial_flux_derivative=flux,
        reference_field=float(on_axis_field[center]),
    )
    center_radii = np.asarray([result.boundary.radius_scale[0, center] for result in results])
    diagnostics = summarize_axisymmetric_beta_scan(
        results,
        jnp.asarray([0.0, 0.01, 0.03, 0.10]),
        plasma_grid,
        reference_field=float(on_axis_field[center]),
    )

    assert len(results) == 4
    assert all(result.converged for result in results)
    assert all(float(result.variational_max) <= config.ftol for result in results)
    assert all(
        float(result.plasma_staggered_weak_force.maximum) <= 1.1 * config.ftol
        for result in results
    )
    assert all(
        np.isfinite(float(result.plasma_force.normalized_rms)) for result in results
    )
    assert all(
        float(result.normalized_divergence_rms) < 3.0e-13 for result in results
    )
    assert all(float(result.interface.normal_stress_rms) < 1.0e-12 for result in results)
    assert np.all(np.diff(center_radii) > 0.0)
    assert center_radii[-1] > 1.005 * center_radii[0]
    np.testing.assert_allclose(
        [item.achieved_reference_beta for item in diagnostics],
        [0.0, 0.01, 0.03, 0.10],
        rtol=2.0e-8,
        atol=1.0e-12,
    )
    assert 0.045 < float(diagnostics[-1].volume_averaged_beta) < 0.052
    assert float(diagnostics[-1].diamagnetic_field_ratio) < 0.97
    assert abs(float(diagnostics[-1].paraxial_relative_error)) < 0.01
    assert all(result.vacuum_potential.shape == vacuum_grid.shape for result in results)
    assert all(float(result.vacuum_potential[-1, 0, 0]) == 0.0 for result in results)
    assert all(np.all(np.isfinite(np.asarray(result.vacuum_field.total_xyz))) for result in results)


@pytest.mark.full
def test_unbounded_exterior_free_boundary_beta_scan_converges() -> None:
    config = MirrorConfig(
        resolution=MirrorResolution(ns=5, mpol=0, ntheta=1, nxi=7),
        z_min=-0.8,
        z_max=0.8,
        ftol=1.0e-12,
        max_iterations=200,
    )
    plasma_grid = config.build_grid()
    vacuum_grid = build_vacuum_grid(plasma_grid, nrho=5)
    on_axis = _on_axis_mirror_field(
        jnp.asarray(plasma_grid.z),
        coil_radius=0.9,
        separation=2.0,
        current=2.0e5,
    )
    center = plasma_grid.nxi // 2
    flux = 0.5 * on_axis[center] * 0.25**2
    betas = jnp.asarray([0.0, 0.10, 0.25, 0.50])
    results = solve_axisymmetric_beta_scan_cli(
        MirrorBoundary.from_axis_field(flux, on_axis, plasma_grid),
        plasma_grid,
        vacuum_grid,
        config,
        _external_mirror_field,
        betas,
        outer_radius=0.1,
        axial_flux_derivative=flux,
        reference_field=float(on_axis[center]),
        vacuum_backend="exterior",
        exterior_ntheta=8,
        exterior_order=6,
        exterior_spectral_side_density=True,
        exterior_high_order_cap_panels=True,
        exterior_curved_side_geometry=True,
    )

    assert all(result.converged for result in results)
    assert all(float(result.variational_max) <= config.ftol for result in results)
    assert all(float(result.interface.vacuum_b_normal_rms) < 1.0e-12 for result in results)
    assert all(float(result.interface.normal_stress_rms) < 1.0e-12 for result in results)
    assert all(result.vacuum_potential.shape == vacuum_grid.shape for result in results)
    assert all(np.all(np.asarray(result.vacuum_potential) == 0.0) for result in results)
    diagnostics = summarize_axisymmetric_beta_scan(
        results,
        betas,
        plasma_grid,
        reference_field=float(on_axis[center]),
    )
    np.testing.assert_allclose(
        [item.achieved_reference_beta for item in diagnostics],
        betas,
        rtol=2.0e-8,
        atol=1.0e-12,
    )
    center_radii = np.asarray([item.center_radius for item in diagnostics])
    field_ratios = np.asarray([item.diamagnetic_field_ratio for item in diagnostics])
    assert np.all(np.diff(center_radii) > 0.0)
    assert np.all(np.diff(field_ratios) < 0.0)
    assert center_radii[-1] > 1.07 * center_radii[0]
    assert field_ratios[-1] < 0.77
    assert all(np.isfinite(float(item.center_vacuum_side_field)) for item in diagnostics)


@pytest.mark.full
def test_nonaxisymmetric_exterior_free_boundary_equilibrium_converges() -> None:
    config = MirrorConfig(
        resolution=MirrorResolution(ns=5, mpol=1, ntheta=3, nxi=5),
        z_min=-0.8,
        z_max=0.8,
        ftol=1.0e-12,
        max_iterations=300,
    )
    grid = config.build_grid()
    vacuum_grid = build_vacuum_grid(grid, nrho=5)
    on_axis = _on_axis_mirror_field(
        jnp.asarray(grid.z),
        coil_radius=0.9,
        separation=2.0,
        current=2.0e5,
    )
    center = grid.nxi // 2
    flux = 0.5 * on_axis[center] * 0.2**2
    base = MirrorBoundary.from_axis_field(flux, on_axis, grid)
    boundary = MirrorBoundary(
        base.radius_scale + 0.03 * jnp.asarray(grid.xi)[None, :] * jnp.cos(jnp.asarray(grid.theta)[:, None])
    )
    betas = jnp.asarray([0.0, 0.50])
    results = solve_beta_scan_cli(
        boundary,
        grid,
        vacuum_grid,
        config,
        _nonaxisymmetric_mirror_field,
        betas,
        outer_radius=0.1,
        axial_flux_derivative=flux,
        reference_field=float(on_axis[center]),
        current_derivative=1.0e-3 * jnp.asarray(grid.s),
        vacuum_backend="exterior",
        exterior_order=6,
        exterior_spectral_side_density=True,
        exterior_high_order_cap_panels=False,
        exterior_curved_side_geometry=True,
    )

    assert all(result.converged for result in results)
    assert all(float(result.variational_max) <= config.ftol for result in results)
    assert all(
        float(result.plasma_staggered_weak_force.maximum) <= config.ftol
        for result in results
    )
    assert all(float(result.normalized_divergence_rms) < 1.0e-12 for result in results)
    assert all(
        float(jnp.max(jnp.abs(result.plasma_state.lambda_stream))) > 1.0e-5
        for result in results
    )
    assert all(float(result.interface.vacuum_b_normal_rms) < 1.0e-12 for result in results)
    assert all(float(result.interface.normal_stress_rms) < 1.0e-12 for result in results)
    assert all(float(result.vacuum_field.neumann_result.compatibility_error) < 2.0e-3 for result in results)
    assert all(float(result.vacuum_field.neumann_result.condition_number) < 5.0 for result in results)

    diagnostics = summarize_nonaxisymmetric_beta_scan(
        results,
        betas,
        grid,
        reference_field=float(on_axis[center]),
    )
    achieved_betas = np.asarray([item.achieved_reference_beta for item in diagnostics])
    np.testing.assert_allclose(achieved_betas, betas, rtol=2.0e-8, atol=1.0e-12)
    mean_radii = np.asarray([item.center_mean_radius for item in diagnostics])
    mean_fields = np.asarray([item.center_mean_field for item in diagnostics])
    mode_one = np.asarray([item.center_boundary_modes[1] for item in diagnostics])
    mode_one_l2 = np.asarray([item.boundary_mode_l2[1] for item in diagnostics])
    mode_one_max = np.asarray([item.boundary_mode_max[1] for item in diagnostics])
    mode_one_core_l2 = np.asarray(
        [item.boundary_mode_core_l2[1] for item in diagnostics]
    )
    assert np.all(np.diff(mean_radii) > 0.0)
    assert np.all(np.diff(mean_fields) < 0.0)
    assert np.all(mode_one > 1.0e-4)
    assert mode_one[-1] > 1.2 * mode_one[0]
    assert np.all(mode_one_l2 > 1.0e-2)
    assert np.all(mode_one_max > 2.5e-2)
    assert np.all(mode_one_core_l2 > 1.0e-3)
    assert np.all(mode_one / mode_one_max < 1.2e-1)
    assert all(float(item.plasma_volume) > 0.0 for item in diagnostics)
    assert all(float(item.plasma_energy) > 0.0 for item in diagnostics)


@pytest.mark.full
def test_unbounded_exterior_beta_observables_converge_with_resolution() -> None:
    observables = []
    compatibility = []
    betas = jnp.asarray([0.0, 0.10, 0.50])
    for ns, nxi, ntheta_panel in ((5, 7, 8), (7, 13, 12), (9, 17, 16)):
        config = MirrorConfig(
            resolution=MirrorResolution(ns=ns, mpol=0, ntheta=1, nxi=nxi),
            z_min=-0.8,
            z_max=0.8,
            ftol=1.0e-12,
            max_iterations=500,
        )
        plasma_grid = config.build_grid()
        vacuum_grid = build_vacuum_grid(plasma_grid, nrho=ns)
        on_axis = _on_axis_mirror_field(
            jnp.asarray(plasma_grid.z),
            coil_radius=0.9,
            separation=2.0,
            current=2.0e5,
        )
        center = plasma_grid.nxi // 2
        flux = 0.5 * on_axis[center] * 0.25**2
        results = solve_axisymmetric_beta_scan_cli(
            MirrorBoundary.from_axis_field(flux, on_axis, plasma_grid),
            plasma_grid,
            vacuum_grid,
            config,
            _external_mirror_field,
            betas,
            outer_radius=0.1,
            axial_flux_derivative=flux,
            reference_field=float(on_axis[center]),
            vacuum_backend="exterior",
            exterior_ntheta=ntheta_panel,
            exterior_order=8,
        )
        assert all(result.converged for result in results)
        assert all(float(result.variational_max) <= config.ftol for result in results)
        observables.append(
            np.asarray(
                [
                    [
                        float(result.boundary.radius_scale[0, center]),
                        float(jnp.sqrt(result.plasma_b_squared[0, 0, center])),
                    ]
                    for result in results
                ]
            )
        )
        compatibility.append(
            np.asarray([float(result.vacuum_field.neumann_result.compatibility_error) for result in results])
        )

    relative_change = np.abs((observables[-1] - observables[-2]) / observables[-1])
    assert np.max(relative_change[:2]) < 5.0e-4
    assert np.max(relative_change[2]) < 5.0e-3
    assert np.max(compatibility[-1]) < 3.0e-9
    assert np.all(compatibility[-1] < compatibility[0])
    assert float(results[-1].boundary.radius_scale[0, center]) > 1.07 * float(
        results[0].boundary.radius_scale[0, center]
    )


def test_two_coil_anisotropic_free_boundary_calibrates_perpendicular_beta() -> None:
    config = MirrorConfig(
        resolution=MirrorResolution(ns=5, mpol=0, ntheta=1, nxi=7),
        z_min=-0.8,
        z_max=0.8,
        ftol=1.0e-12,
        max_iterations=1000,
    )
    plasma_grid = config.build_grid()
    vacuum_grid = build_vacuum_grid(plasma_grid, nrho=5)
    on_axis = _on_axis_mirror_field(
        jnp.asarray(plasma_grid.z),
        coil_radius=0.9,
        separation=2.0,
        current=2.0e5,
    )
    center = plasma_grid.nxi // 2
    flux = 0.5 * on_axis[center] * 0.25**2
    boundary = MirrorBoundary.from_axis_field(flux, on_axis, plasma_grid)
    target_pressure = 0.01 * on_axis[center] ** 2 / (2.0 * MU0)
    closure = BiMaxwellianPressureClosure(
        mass_coefficients=jnp.asarray([1.0, -1.0]),
        hot_fraction_coefficients=jnp.asarray([0.2]),
        temperature_ratio=0.7,
        critical_field=float(on_axis[center]),
        gamma=0.0,
    )

    results = solve_axisymmetric_beta_scan_cli(
        boundary,
        plasma_grid,
        vacuum_grid,
        config,
        _external_mirror_field,
        jnp.asarray([0.0, 0.01]),
        outer_radius=0.65,
        axial_flux_derivative=flux,
        reference_field=float(on_axis[center]),
        pressure_closure=closure,
    )
    result = results[-1]
    energy = result.plasma_energy
    b_squared = magnetic_field_squared(energy.field, energy.geometry)
    central_b = jnp.sqrt(b_squared[0, 0, center])
    central_perpendicular = result.mass_scale * closure.moments(0.0, central_b).perpendicular
    anisotropy = jnp.max(jnp.abs(energy.moments_half.perpendicular - energy.moments_half.parallel)) / jnp.max(
        energy.moments_half.parallel
    )

    assert result.converged
    assert float(result.variational_max) <= config.ftol
    assert float(result.interface.normal_stress_rms) < 2.0e-12
    np.testing.assert_allclose(central_perpendicular, target_pressure, rtol=2.0e-12)
    assert float(anisotropy) > 0.04
    assert bool(jnp.all(energy.indicators_half.valid))
    assert float(result.boundary.radius_scale[0, center]) > float(results[0].boundary.radius_scale[0, center])


def test_tabulated_pressure_free_boundary_matches_sampled_bimaxwellian() -> None:
    config = MirrorConfig(
        resolution=MirrorResolution(ns=5, mpol=0, ntheta=1, nxi=7),
        z_min=-0.8,
        z_max=0.8,
        ftol=1.0e-12,
        max_iterations=1000,
    )
    grid = config.build_grid()
    vacuum_grid = build_vacuum_grid(grid, nrho=5)
    on_axis = _on_axis_mirror_field(jnp.asarray(grid.z))
    center = grid.nxi // 2
    flux = 0.5 * on_axis[center] * 0.25**2
    boundary = MirrorBoundary.from_axis_field(flux, on_axis, grid)
    reference = BiMaxwellianPressureClosure(
        mass_coefficients=jnp.asarray([1.0, -1.0]),
        hot_fraction_coefficients=jnp.asarray([0.2]),
        temperature_ratio=0.7,
        critical_field=float(on_axis[center]),
        gamma=0.0,
    )
    s_nodes = jnp.linspace(0.0, 1.0, 5)
    b_nodes = jnp.linspace(0.04, 0.18, 9)
    closure = TabulatedPressureClosure(
        s_nodes,
        b_nodes,
        reference.parallel_pressure(s_nodes[:, None], b_nodes[None, :]),
        gamma=0.0,
    )
    betas = jnp.asarray([0.0, 0.01])
    results = solve_axisymmetric_beta_scan_cli(
        boundary,
        grid,
        vacuum_grid,
        config,
        _external_mirror_field,
        betas,
        outer_radius=0.65,
        axial_flux_derivative=flux,
        reference_field=float(on_axis[center]),
        pressure_closure=closure,
    )
    reference_results = solve_axisymmetric_beta_scan_cli(
        boundary,
        grid,
        vacuum_grid,
        config,
        _external_mirror_field,
        betas,
        outer_radius=0.65,
        axial_flux_derivative=flux,
        reference_field=float(on_axis[center]),
        pressure_closure=reference,
    )
    result = results[-1]
    diagnostic = summarize_axisymmetric_beta_scan(results, betas, grid, reference_field=float(on_axis[center]))[-1]
    reference_diagnostic = summarize_axisymmetric_beta_scan(
        reference_results,
        betas,
        grid,
        reference_field=float(on_axis[center]),
    )[-1]

    assert result.converged
    assert float(result.variational_max) <= config.ftol
    assert bool(jnp.all(result.plasma_energy.indicators_half.valid))
    np.testing.assert_allclose(diagnostic.achieved_reference_beta, 0.01, rtol=2e-8)
    np.testing.assert_allclose(
        diagnostic.center_radius,
        reference_diagnostic.center_radius,
        rtol=5.0e-5,
    )
    np.testing.assert_allclose(
        diagnostic.diamagnetic_field_ratio,
        reference_diagnostic.diamagnetic_field_ratio,
        rtol=5.0e-5,
    )


@pytest.mark.full
def test_anisotropic_free_boundary_observables_converge_with_resolution() -> None:
    observables = []
    normal_fields = []
    for ns, nxi, nrho in ((5, 7, 5), (7, 13, 7), (9, 17, 9)):
        config = MirrorConfig(
            resolution=MirrorResolution(ns=ns, mpol=0, ntheta=1, nxi=nxi),
            z_min=-0.8,
            z_max=0.8,
            ftol=1.0e-12,
            max_iterations=2000,
        )
        grid = config.build_grid()
        vacuum_grid = build_vacuum_grid(grid, nrho=nrho)
        on_axis = _on_axis_mirror_field(
            jnp.asarray(grid.z),
            coil_radius=0.9,
            separation=2.0,
            current=2.0e5,
        )
        center = grid.nxi // 2
        flux = 0.5 * on_axis[center] * 0.25**2
        boundary = MirrorBoundary.from_axis_field(flux, on_axis, grid)
        closure = BiMaxwellianPressureClosure(
            mass_coefficients=jnp.asarray([1.0, -1.0]),
            hot_fraction_coefficients=jnp.asarray([0.2]),
            temperature_ratio=0.7,
            critical_field=float(on_axis[center]),
            gamma=0.0,
        )
        results = solve_axisymmetric_beta_scan_cli(
            boundary,
            grid,
            vacuum_grid,
            config,
            _external_mirror_field,
            jnp.asarray([0.0, 0.10]),
            outer_radius=0.65,
            axial_flux_derivative=flux,
            reference_field=float(on_axis[center]),
            pressure_closure=closure,
        )
        result = results[-1]
        diagnostic = summarize_axisymmetric_beta_scan(
            results,
            jnp.asarray([0.0, 0.10]),
            grid,
            reference_field=float(on_axis[center]),
        )[-1]
        anisotropy = jnp.max(
            jnp.abs(result.plasma_energy.moments_half.perpendicular - result.plasma_energy.moments_half.parallel)
        ) / jnp.max(result.plasma_energy.moments_half.parallel)
        observables.append(
            np.asarray(
                [
                    diagnostic.center_radius,
                    diagnostic.diamagnetic_field_ratio,
                    diagnostic.volume_averaged_beta,
                    anisotropy,
                ]
            )
        )
        normal_fields.append(float(result.interface.vacuum_b_normal_rms))
        assert result.converged
        assert float(result.variational_max) <= config.ftol
        assert bool(jnp.all(result.plasma_energy.indicators_half.valid))

    relative_change = np.abs(observables[-1] - observables[-2]) / np.abs(observables[-1])
    assert np.max(relative_change) < 1.0e-3
    assert normal_fields[2] < normal_fields[1] < normal_fields[0]


@pytest.mark.full
def test_anisotropic_high_beta_scan_remains_elliptic_and_diamagnetic() -> None:
    config = MirrorConfig(
        resolution=MirrorResolution(ns=7, mpol=0, ntheta=1, nxi=13),
        z_min=-0.8,
        z_max=0.8,
        ftol=1.0e-12,
        max_iterations=2000,
    )
    grid = config.build_grid()
    vacuum_grid = build_vacuum_grid(grid, nrho=7)
    on_axis = _on_axis_mirror_field(
        jnp.asarray(grid.z),
        coil_radius=0.9,
        separation=2.0,
        current=2.0e5,
    )
    center = grid.nxi // 2
    flux = 0.5 * on_axis[center] * 0.25**2
    boundary = MirrorBoundary.from_axis_field(flux, on_axis, grid)
    closure = BiMaxwellianPressureClosure(
        mass_coefficients=jnp.asarray([1.0, -1.0]),
        hot_fraction_coefficients=jnp.asarray([0.2]),
        temperature_ratio=0.7,
        critical_field=float(on_axis[center]),
        gamma=0.0,
    )
    betas = jnp.asarray([0.0, 0.10, 0.25, 0.50])
    results = solve_axisymmetric_beta_scan_cli(
        boundary,
        grid,
        vacuum_grid,
        config,
        _external_mirror_field,
        betas,
        outer_radius=0.65,
        axial_flux_derivative=flux,
        reference_field=float(on_axis[center]),
        pressure_closure=closure,
    )
    diagnostics = summarize_axisymmetric_beta_scan(results, betas, grid, reference_field=float(on_axis[center]))

    assert all(result.converged for result in results)
    assert all(float(result.variational_max) <= config.ftol for result in results)
    assert all(bool(jnp.all(result.plasma_energy.indicators_half.valid)) for result in results[1:])
    assert np.all(np.diff([item.center_radius for item in diagnostics]) > 0.0)
    assert np.all(np.diff([item.diamagnetic_field_ratio for item in diagnostics]) < 0.0)
    np.testing.assert_allclose(
        [item.achieved_reference_beta for item in diagnostics],
        betas,
        rtol=2.0e-8,
        atol=1.0e-12,
    )


@pytest.mark.full
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
        on_axis_field = _on_axis_mirror_field(jnp.asarray(plasma_grid.z))
        center = int(np.argmin(np.abs(plasma_grid.z)))
        flux = 0.5 * on_axis_field[center] * 0.25**2
        results = solve_axisymmetric_beta_scan_cli(
            MirrorBoundary.from_axis_field(flux, on_axis_field, plasma_grid),
            plasma_grid,
            vacuum_grid,
            config,
            _external_mirror_field,
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


@pytest.mark.full
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
    on_axis_field = _on_axis_mirror_field(jnp.asarray(plasma_grid.z))
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
                _external_mirror_field,
                outer_radius=0.65,
                axial_flux_derivative=flux,
                mass_profile=mass,
                require_convergence=True,
            )
        )

    np.testing.assert_allclose(results[0].boundary.radius_scale, results[1].boundary.radius_scale, atol=2.0e-12)
    np.testing.assert_allclose(results[0].plasma_energy.b_squared, results[1].plasma_energy.b_squared, rtol=2.0e-11)
    assert all(float(result.variational_max) <= config.ftol for result in results)


@pytest.mark.full
def test_outer_vacuum_dirichlet_neumann_gap_narrows_with_domain() -> None:
    neumann_center_field = []
    dirichlet_center_field = []
    for outer_radius, nrho in ((0.50, 7), (0.65, 11), (0.75, 13), (0.82, 15)):
        config = MirrorConfig(
            resolution=MirrorResolution(ns=7, mpol=0, ntheta=1, nxi=17),
            z_min=-0.8,
            z_max=0.8,
            ftol=1.0e-12,
            max_iterations=1000,
        )
        plasma_grid = config.build_grid()
        vacuum_grid = build_vacuum_grid(plasma_grid, nrho=nrho)
        boundary = MirrorBoundary.from_radius(0.25, plasma_grid)
        geometry = evaluate_vacuum_geometry(boundary, vacuum_grid, outer_radius=outer_radius)
        external = external_field_from_source(_external_mirror_field, geometry)
        center = plasma_grid.nxi // 2
        for condition, values in (
            ("fixed_external_flux", neumann_center_field),
            ("decaying_outer", dirichlet_center_field),
        ):
            result = solve_vacuum_potential(
                boundary,
                vacuum_grid,
                config,
                external,
                jnp.zeros(vacuum_grid.shape),
                outer_radius=outer_radius,
                boundary_condition=condition,
            )
            assert result.converged
            values.append(float(jnp.linalg.norm(result.field.total_xyz[0, 0, center])))
            if condition == "decaying_outer":
                np.testing.assert_allclose(result.potential[-1], 0.0, atol=0.0)

    gap = np.asarray(dirichlet_center_field) - np.asarray(neumann_center_field)
    assert np.all(np.diff(neumann_center_field) > 0.0)
    assert np.all(np.diff(dirichlet_center_field) < 0.0)
    assert np.all(gap > 0.0)
    assert np.all(np.diff(gap) < 0.0)
    assert gap[-1] < 0.2 * gap[0]
