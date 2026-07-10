"""M4 nonaxisymmetric fixed-boundary equilibrium tests."""

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
    SeparableMirrorPreconditioner,
    contravariant_field,
    solve_fixed_boundary_cli,
)


@pytest.fixture(scope="module", autouse=True)
def _enable_solver_jit():
    previous = bool(jax.config.jax_disable_jit)
    jax.config.update("jax_disable_jit", False)
    yield
    jax.config.update("jax_disable_jit", previous)


def test_helical_finite_current_solve_converges_with_gauge_free_lambda() -> None:
    config = MirrorConfig(
        resolution=MirrorResolution(ns=5, mpol=1, ntheta=3, nxi=5),
        z_min=-1.0,
        z_max=1.0,
        ftol=1.0e-12,
        max_iterations=1000,
    )
    grid = config.build_grid()
    theta = jnp.asarray(grid.theta)[:, None]
    xi = jnp.asarray(grid.xi)[None, :]
    boundary = MirrorBoundary.from_radius(
        0.3 * (1.0 + 0.02 * jnp.cos(theta) * (1.0 - xi**2)), grid
    )
    initial = MirrorState.from_boundary(boundary, grid)
    current_derivative = 1.0e-3 * jnp.asarray(grid.s)

    result = solve_fixed_boundary_cli(
        initial,
        boundary,
        grid,
        config,
        axial_flux_derivative=0.1,
        current_derivative=current_derivative,
        solve_lambda=True,
        gradient_tolerance=1.0e-12,
        require_convergence=True,
    )
    field = contravariant_field(
        result.state,
        result.energy.geometry,
        grid,
        axial_flux_derivative=0.1,
        current_derivative=current_derivative,
    )
    pitch = field.b_sup_theta[1:, :, 1:-1] / field.b_sup_xi[1:, :, 1:-1]
    surface_means = np.einsum(
        "j,k,ijk->i",
        grid.theta_basis.weights,
        grid.axial_basis.weights,
        np.asarray(result.state.lambda_stream),
    ) / (4.0 * np.pi)

    assert result.converged
    assert float(result.variational.maximum) <= config.ftol
    assert float(result.variational.lambda_rms) < config.ftol
    assert float(jnp.max(jnp.abs(result.state.lambda_stream))) > 1.0e-3
    assert float(jnp.max(jnp.abs(pitch))) > 1.0e-3
    np.testing.assert_allclose(surface_means, 0.0, atol=2.0e-15)
    np.testing.assert_allclose(result.state.lambda_stream[0], result.state.lambda_stream[1])
    np.testing.assert_allclose(result.state.lambda_stream[:, :, [0, -1]], 0.0)
    np.testing.assert_allclose(result.state.radius_scale[-1], boundary.radius_scale)


def test_mode_aware_preconditioner_preserves_lambda_gauge_subspace() -> None:
    config = MirrorConfig(
        resolution=MirrorResolution(ns=7, mpol=2, ntheta=7, nxi=9)
    )
    grid = config.build_grid()
    preconditioner = SeparableMirrorPreconditioner.build(
        grid, radial_nodes=grid.ns - 1
    )
    weights = (
        np.asarray(grid.theta_basis.weights)[:, None]
        * np.asarray(grid.axial_basis.weights)[None, 1:-1]
    ).reshape(-1)
    pivot = int(np.argmax(weights))
    free_indices = np.delete(np.arange(weights.size), pivot)
    rng = np.random.default_rng(21)
    left = rng.normal(size=(grid.ns - 1) * free_indices.size)
    right = rng.normal(size=left.size)

    applied_left = preconditioner.apply_gauge_free(
        left, free_indices=free_indices, pivot=pivot, weights=weights
    )
    applied_right = preconditioner.apply_gauge_free(
        right, free_indices=free_indices, pivot=pivot, weights=weights
    )

    np.testing.assert_allclose(
        np.dot(left, applied_right), np.dot(applied_left, right), rtol=2.0e-12
    )
    assert np.dot(left, applied_left) > 0.0


def test_mode_aware_tensor_operator_is_inverted_to_roundoff() -> None:
    config = MirrorConfig(
        resolution=MirrorResolution(ns=9, mpol=3, ntheta=9, nxi=13)
    )
    preconditioner = SeparableMirrorPreconditioner.build(config.build_grid())
    exact = np.random.default_rng(8).normal(size=preconditioner.size)

    np.testing.assert_allclose(
        preconditioner.apply(preconditioner.operator(exact)),
        exact,
        rtol=3.0e-12,
        atol=3.0e-12,
    )


@pytest.mark.full
def test_helical_equilibrium_radial_axial_refinement() -> None:
    energies = []
    for ns, nxi in ((5, 5), (7, 7), (9, 9)):
        config = MirrorConfig(
            resolution=MirrorResolution(ns=ns, mpol=1, ntheta=3, nxi=nxi),
            z_min=-1.0,
            z_max=1.0,
            ftol=1.0e-12,
            max_iterations=1000,
        )
        grid = config.build_grid()
        theta = jnp.asarray(grid.theta)[:, None]
        xi = jnp.asarray(grid.xi)[None, :]
        boundary = MirrorBoundary.from_radius(
            0.3 * (1.0 + 0.02 * jnp.cos(theta) * (1.0 - xi**2)), grid
        )
        current_derivative = 1.0e-3 * jnp.asarray(grid.s)
        result = solve_fixed_boundary_cli(
            MirrorState.from_boundary(boundary, grid),
            boundary,
            grid,
            config,
            axial_flux_derivative=0.1,
            current_derivative=current_derivative,
            solve_lambda=True,
            gradient_tolerance=1.0e-12,
            require_convergence=True,
        )
        energies.append(float(result.energy.total))
        assert result.converged
        assert float(result.variational.maximum) <= config.ftol

    increments = np.abs(np.diff(energies))
    assert increments[1] < increments[0]
    assert increments[1] / abs(energies[-1]) < 1.0e-6
