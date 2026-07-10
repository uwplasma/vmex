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
