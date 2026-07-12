"""Implicit differentiation of converged mirror equilibria."""

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
    fixed_boundary_adjoint,
    fixed_boundary_parameters,
    project_fixed_boundary_state,
    solve_fixed_boundary_cli,
)


def test_fixed_boundary_adjoint_matches_reconverged_central_difference() -> None:
    config = MirrorConfig(
        resolution=MirrorResolution(ns=3, nxi=5),
        ftol=1.0e-12,
        max_iterations=500,
    )
    grid = config.build_grid()
    xi, s = jnp.asarray(grid.xi), jnp.asarray(grid.s)
    boundary = MirrorBoundary.from_radius(0.3 * (1.0 + 0.12 * (1.0 - xi**2)), grid)
    mass = 2.0e-4 * (1.0 - s)
    current = 1.0e-3 * s
    result = solve_fixed_boundary_cli(
        MirrorState.from_boundary(boundary, grid),
        boundary,
        grid,
        config,
        axial_flux_derivative=0.1,
        mass_profile=mass,
        current_derivative=current,
        solve_lambda=True,
        require_convergence=True,
    )
    parameters = fixed_boundary_parameters(
        boundary,
        axial_flux_derivative=0.1,
        mass_profile=mass,
        current_derivative=current,
    )

    def quantity(state, _energy):
        return state.radius_scale[1, 0, grid.nxi // 2]

    adjoint = fixed_boundary_adjoint(
        result,
        parameters,
        grid,
        quantity,
        solve_lambda=True,
        rtol=1.0e-10,
    )
    boundary_direction = 0.02 * (1.0 - xi**2)[None, :]
    flux_direction = 0.003
    mass_direction = 1.0e-5 * (1.0 - s)
    current_direction = 2.0e-4 * s
    predicted = float(
        jnp.vdot(adjoint.gradient.boundary_radius, boundary_direction)
        + adjoint.gradient.axial_flux_derivative * flux_direction
        + jnp.vdot(adjoint.gradient.mass_profile, mass_direction)
        + jnp.vdot(adjoint.gradient.current_derivative, current_direction)
    )

    epsilon = 1.0e-4
    values = []
    for sign in (-1.0, 1.0):
        varied_boundary = MirrorBoundary(
            boundary.radius_scale + sign * epsilon * boundary_direction
        )
        varied = solve_fixed_boundary_cli(
            project_fixed_boundary_state(result.state, varied_boundary, grid),
            varied_boundary,
            grid,
            config,
            axial_flux_derivative=0.1 + sign * epsilon * flux_direction,
            mass_profile=mass + sign * epsilon * mass_direction,
            current_derivative=current + sign * epsilon * current_direction,
            solve_lambda=True,
            require_convergence=True,
        )
        values.append(float(quantity(varied.state, varied.energy)))
    finite_difference = (values[1] - values[0]) / (2.0 * epsilon)

    assert adjoint.converged
    assert adjoint.iterations > 0
    assert adjoint.relative_residual < 1.0e-10
    np.testing.assert_allclose(predicted, finite_difference, rtol=2.0e-8, atol=1.0e-11)


def test_fixed_boundary_adjoint_rejects_unconverged_state() -> None:
    class Result:
        converged = False

    config = MirrorConfig(resolution=MirrorResolution(ns=3, nxi=5))
    grid = config.build_grid()
    boundary = MirrorBoundary.from_radius(0.3, grid)
    parameters = fixed_boundary_parameters(boundary, axial_flux_derivative=0.1)

    with pytest.raises(ValueError, match="converged"):
        fixed_boundary_adjoint(
            Result(), parameters, grid, lambda state, energy: energy.total
        )
