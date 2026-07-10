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
    build_vacuum_grid,
    evaluate_vacuum_field,
    evaluate_vacuum_geometry,
    external_field_from_coils,
    solve_vacuum_potential,
    vacuum_laplacian,
)
from vmec_jax.core.coils import CoilSet  # noqa: E402


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
