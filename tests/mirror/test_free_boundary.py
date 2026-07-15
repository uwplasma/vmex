"""Free-space mirror free-boundary equilibrium and diagnostic tests."""

from __future__ import annotations

from types import SimpleNamespace

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
    solve_beta_scan_cli,
)
import vmec_jax.mirror.free_boundary as continuation  # noqa: E402
from vmec_jax.mirror.free_boundary import (  # noqa: E402
    interpolate_fixed_boundary_state,
    solve_axisymmetric_beta_scan_cli,
)
from vmec_jax.mirror.model import project_fixed_boundary_state  # noqa: E402
from vmec_jax.mirror.output import (  # noqa: E402
    boundary_fourier_amplitudes,
    boundary_fourier_norms,
    summarize_axisymmetric_beta_scan,
    summarize_nonaxisymmetric_beta_scan,
)
from vmec_jax.mirror.output import (  # noqa: E402
    FreeBoundaryRestart,
    load_free_boundary_restart,
    save_free_boundary_restart,
)


@pytest.fixture(scope="module", autouse=True)
def _enable_solver_jit():
    previous = bool(jax.config.jax_disable_jit)
    jax.config.update("jax_disable_jit", False)
    yield
    jax.config.update("jax_disable_jit", previous)


def test_free_boundary_restart_roundtrip_is_compact_and_grid_checked(tmp_path) -> None:
    config = MirrorConfig(resolution=MirrorResolution(ns=7, mpol=0, nxi=9))
    plasma_grid = config.build_grid()
    boundary = MirrorBoundary.from_radius(0.3, plasma_grid)
    state = MirrorState.from_boundary(boundary, plasma_grid)
    restart = FreeBoundaryRestart(
        boundary=boundary,
        plasma_state=state,
        mass_scale=1.25,
    )

    path = save_free_boundary_restart(tmp_path / "beta_003", restart)
    loaded = load_free_boundary_restart(path, plasma_grid)

    assert path.suffix == ".npz"
    assert path.stat().st_size < 4096
    np.testing.assert_array_equal(loaded.boundary.radius_scale, boundary.radius_scale)
    np.testing.assert_array_equal(loaded.plasma_state.radius_scale, state.radius_scale)
    np.testing.assert_array_equal(loaded.plasma_state.lambda_stream, state.lambda_stream)
    assert loaded.mass_scale == restart.mass_scale

    mismatched = MirrorConfig(resolution=MirrorResolution(ns=9, mpol=0, nxi=9)).build_grid()
    with pytest.raises(ValueError, match="plasma state"):
        load_free_boundary_restart(path, mismatched)


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
    grid = MirrorConfig(resolution=MirrorResolution(ns=5, mpol=3, nxi=9)).build_grid()
    theta = jnp.asarray(grid.theta)[:, None]
    xi = jnp.asarray(grid.xi)[None, :]
    boundary = MirrorBoundary(0.2 + 0.03 * xi * jnp.cos(theta))

    l2, maximum = boundary_fourier_norms(boundary, grid)
    core_l2, core_maximum = boundary_fourier_norms(boundary, grid, central_fraction=0.75)

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


@pytest.mark.full
def test_unbounded_exterior_free_boundary_beta_scan_converges() -> None:
    config = MirrorConfig(
        resolution=MirrorResolution(ns=5, mpol=0, nxi=7),
        z_min=-0.8,
        z_max=0.8,
        ftol=1.0e-12,
        max_iterations=200,
    )
    plasma_grid = config.build_grid()
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
        config,
        _external_mirror_field,
        betas,
        axial_flux_derivative=flux,
        reference_field=float(on_axis[center]),
        exterior_ntheta=8,
        exterior_order=6,
        exterior_spectral_side_density=True,
    )

    assert all(result.converged for result in results)
    assert all(float(result.variational_max) <= config.ftol for result in results)
    assert all(float(result.interface.vacuum_b_normal_rms) < 1.0e-12 for result in results)
    assert all(float(result.interface.normal_stress_rms) < 1.0e-12 for result in results)
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
        resolution=MirrorResolution(ns=5, mpol=1, nxi=5),
        z_min=-0.8,
        z_max=0.8,
        ftol=1.0e-12,
        max_iterations=300,
    )
    grid = config.build_grid()
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
        config,
        _nonaxisymmetric_mirror_field,
        betas,
        axial_flux_derivative=flux,
        reference_field=float(on_axis[center]),
        current_derivative=1.0e-3 * jnp.asarray(grid.s),
        exterior_order=6,
        exterior_spectral_side_density=True,
    )

    assert all(result.converged for result in results)
    assert all(float(result.variational_max) <= config.ftol for result in results)
    assert all(float(result.plasma_staggered_weak_force.maximum) <= config.ftol for result in results)
    assert all(float(result.normalized_divergence_rms) < 1.0e-12 for result in results)
    assert all(float(jnp.max(jnp.abs(result.plasma_state.lambda_stream))) > 1.0e-5 for result in results)
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
    mode_one_core_l2 = np.asarray([item.boundary_mode_core_l2[1] for item in diagnostics])
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
            resolution=MirrorResolution(ns=ns, mpol=0, nxi=nxi),
            z_min=-0.8,
            z_max=0.8,
            ftol=1.0e-12,
            max_iterations=500,
        )
        plasma_grid = config.build_grid()
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
            config,
            _external_mirror_field,
            betas,
            axial_flux_derivative=flux,
            reference_field=float(on_axis[center]),
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
            np.asarray([float(result.vacuum_field.neumann_result.raw_compatibility_error) for result in results])
        )

    relative_change = np.abs((observables[-1] - observables[-2]) / observables[-1])
    assert np.max(relative_change[:2]) < 5.0e-4
    assert np.max(relative_change[2]) < 5.0e-3
    assert np.max(compatibility[-1]) < 3.0e-9
    assert np.all(compatibility[-1] < compatibility[0])
    assert float(results[-1].boundary.radius_scale[0, center]) > 1.07 * float(
        results[0].boundary.radius_scale[0, center]
    )


def _grid(ns: int, nxi: int):
    return MirrorConfig(resolution=MirrorResolution(ns=ns, mpol=1, nxi=nxi)).build_grid()


def test_fixed_boundary_state_interpolation_roundtrips_and_preserves_constraints() -> None:
    coarse, fine = _grid(7, 9), _grid(11, 13)

    def fields(grid):
        s = jnp.asarray(grid.s)[:, None, None]
        theta = jnp.asarray(grid.theta)[None, :, None]
        xi = jnp.asarray(grid.xi)[None, None, :]
        radius = 0.2 + 0.1 * s + 0.01 * jnp.cos(theta) * (1.0 - xi**2)
        lam = s * jnp.cos(theta) * (1.0 - xi**2)
        boundary = MirrorBoundary.from_radius(radius[-1], grid)
        state = project_fixed_boundary_state(MirrorState(radius, lam), boundary, grid)
        return boundary, state

    coarse_boundary, coarse_state = fields(coarse)
    fine_boundary, _ = fields(fine)
    interpolated = interpolate_fixed_boundary_state(coarse_state, coarse, fine_boundary, fine)
    roundtrip = interpolate_fixed_boundary_state(interpolated, fine, coarse_boundary, coarse)

    # Axis closure deliberately flattens the first radial interval, so only
    # surfaces outside that interval are an exact coarse-fine round trip.
    np.testing.assert_allclose(roundtrip.radius_scale[2:], coarse_state.radius_scale[2:], atol=3.0e-14)
    np.testing.assert_allclose(roundtrip.lambda_stream[2:], coarse_state.lambda_stream[2:], atol=3.0e-14)
    np.testing.assert_allclose(interpolated.radius_scale[-1], fine_boundary.radius_scale)
    np.testing.assert_allclose(interpolated.radius_scale[0], interpolated.radius_scale[1])
    np.testing.assert_allclose(interpolated.lambda_stream[0], interpolated.lambda_stream[1])
    np.testing.assert_allclose(interpolated.lambda_stream[:, :, [0, -1]], 0.0, atol=2.0e-15)


def test_fixed_boundary_state_interpolation_is_differentiable() -> None:
    coarse, fine = _grid(5, 7), _grid(7, 9)
    boundary = MirrorBoundary.from_radius(0.3, coarse)
    state = MirrorState.from_boundary(boundary, coarse)
    fine_boundary = MirrorBoundary.from_radius(0.3, fine)

    tangent = jnp.ones(coarse.shape).at[:, :, [0, -1]].set(0.0)

    def interpolate_lambda(scale):
        varied = MirrorState(state.radius_scale, state.lambda_stream + scale * tangent)
        return interpolate_fixed_boundary_state(varied, coarse, fine_boundary, fine).lambda_stream

    derivative = jax.jacfwd(interpolate_lambda)(1.0)
    assert bool(jnp.all(jnp.isfinite(derivative)))
    assert float(jnp.linalg.norm(derivative)) > 0.0


def test_beta_scan_propagates_restart_mass_scale(monkeypatch) -> None:
    config = MirrorConfig(resolution=MirrorResolution(ns=5, mpol=0, nxi=5))
    grid = config.build_grid()
    reference = MirrorBoundary.from_radius(0.3, grid)
    restart_boundary = MirrorBoundary.from_radius(0.31, grid)
    restart_state = MirrorState.from_boundary(restart_boundary, grid)
    restart = FreeBoundaryRestart(restart_boundary, restart_state, 2.5)
    received = []

    def fake_solve(boundary, *_args, **kwargs):
        received.append(
            (
                boundary,
                kwargs["initial_state"],
                kwargs["initial_mass_scale"],
                kwargs["exterior_spectral_side_density"],
            )
        )
        return SimpleNamespace(
            boundary=boundary,
            plasma_state=kwargs["initial_state"],
            mass_scale=jnp.asarray(kwargs["initial_mass_scale"] + 0.5),
        )

    monkeypatch.setattr(continuation, "solve_axisymmetric_free_boundary_cli", fake_solve)
    solve_axisymmetric_beta_scan_cli(
        reference,
        grid,
        config,
        object(),
        jnp.asarray([0.0, 0.0]),
        axial_flux_derivative=0.1,
        reference_field=1.0,
        initial_restart=restart,
        exterior_spectral_side_density=True,
    )

    assert received[0][0] is restart_boundary
    assert received[0][1] is restart_state
    assert received[0][2] == 2.5
    assert received[1][2] == 3.0
    assert all(item[3] is True for item in received)
