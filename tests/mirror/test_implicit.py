"""Implicit differentiation of converged mirror equilibria."""

from __future__ import annotations

from dataclasses import replace
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
    IsotropicPressureClosure,
    BiMaxwellianPressureClosure,
    TabulatedPressureClosure,
    fixed_boundary_adjoint,
    solve_fixed_boundary_cli,
    solve_fixed_boundary_implicit,
    spline_fixed_boundary_adjoint,
)
from vmec_jax.mirror.implicit import (  # noqa: E402
    fixed_boundary_parameters,
    make_fixed_boundary_implicit_config,
    spline_fixed_boundary_parameters,
)
from vmec_jax.mirror.model import project_fixed_boundary_state  # noqa: E402
from vmec_jax.mirror.solver import solve_anisotropic_fixed_boundary_cli  # noqa: E402
from vmec_jax.mirror.splines import (  # noqa: E402
    SplineMirrorBoundary,
    SplineMirrorDiscretization,
    solve_spline_fixed_boundary_cli,
)


@pytest.fixture(autouse=True)
def _enable_solver_jit():
    previous = bool(jax.config.jax_disable_jit)
    jax.config.update("jax_disable_jit", False)
    yield
    jax.config.update("jax_disable_jit", previous)


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


def test_spline_adjoint_matches_reconverged_central_difference() -> None:
    config = MirrorConfig(
        resolution=MirrorResolution(ns=5, nxi=9),
        ftol=1.0e-12,
        max_iterations=1000,
    )
    source_grid = config.build_grid()
    s, xi = jnp.asarray(source_grid.s), jnp.asarray(source_grid.xi)
    boundary = MirrorBoundary.from_radius(
        0.3 * (1.0 + 0.1 * (1.0 - xi**2)), source_grid
    )
    discretization = SplineMirrorDiscretization.build(config, elements=3)
    spline_boundary = discretization.fit_boundary(boundary, source_grid)
    mass = 2.0e3 * (1.0 - s)
    current = 1.0e-2 * s
    result = solve_spline_fixed_boundary_cli(
        discretization.fit_state(
            MirrorState.from_boundary(boundary, source_grid), source_grid
        ),
        spline_boundary,
        discretization,
        config,
        axial_flux_derivative=0.1,
        mass_profile=mass,
        current_derivative=current,
        gradient_tolerance=1.0e-12,
        require_convergence=True,
    )
    parameters = spline_fixed_boundary_parameters(
        spline_boundary,
        axial_flux_derivative=0.1,
        mass_profile=mass,
        current_derivative=current,
    )

    def quantity(state, _energy):
        return state.radius_scale[2, 0, discretization.grid.nxi // 2]

    adjoint = spline_fixed_boundary_adjoint(
        result,
        parameters,
        discretization,
        quantity,
        rtol=1.0e-10,
    )
    nodes = jnp.asarray(discretization.spline.collocation_nodes)
    boundary_direction = 0.01 * (1.0 - nodes**2)[None, :]
    flux_direction = jnp.asarray(0.003)
    mass_direction = 100.0 * (1.0 - s)
    current_direction = 2.0e-3 * s
    predicted = float(
        jnp.vdot(adjoint.gradient.boundary_coefficients, boundary_direction)
        + adjoint.gradient.axial_flux_derivative * flux_direction
        + jnp.vdot(adjoint.gradient.mass_profile, mass_direction)
        + jnp.vdot(adjoint.gradient.current_derivative, current_direction)
    )

    values = []
    epsilon = 1.0e-4
    for sign in (-1.0, 1.0):
        varied_boundary = SplineMirrorBoundary(
            spline_boundary.radius_coefficients
            + sign * epsilon * boundary_direction
        )
        initial = discretization.transfer_boundary(
            result.coefficient_state, spline_boundary, varied_boundary
        )
        varied = solve_spline_fixed_boundary_cli(
            initial,
            varied_boundary,
            discretization,
            config,
            axial_flux_derivative=0.1 + sign * epsilon * flux_direction,
            mass_profile=mass + sign * epsilon * mass_direction,
            current_derivative=current + sign * epsilon * current_direction,
            gradient_tolerance=1.0e-12,
            require_convergence=True,
        )
        values.append(float(quantity(varied.evaluated.state, varied.evaluated.energy)))
    finite_difference = (values[1] - values[0]) / (2.0 * epsilon)

    assert adjoint.converged
    assert adjoint.iterations > 0
    assert adjoint.relative_residual < 1.0e-10
    np.testing.assert_allclose(predicted, finite_difference, rtol=2.0e-6, atol=1.0e-10)


def test_nonaxisymmetric_spline_adjoint_includes_stream_function() -> None:
    config = MirrorConfig(
        resolution=MirrorResolution(ns=5, mpol=1, ntheta=4, nxi=7),
        ftol=1.0e-12,
        max_iterations=1000,
    )
    source_grid = config.build_grid()
    s = jnp.asarray(source_grid.s)
    theta = jnp.asarray(source_grid.theta)[:, None]
    xi = jnp.asarray(source_grid.xi)[None, :]
    boundary = MirrorBoundary.from_radius(
        0.3 * (1.0 + 0.02 * jnp.cos(theta) * (1.0 - xi**2)), source_grid
    )
    discretization = SplineMirrorDiscretization.build(config, elements=3)
    spline_boundary = discretization.fit_boundary(boundary, source_grid)
    current = 1.0e-3 * s
    result = solve_spline_fixed_boundary_cli(
        discretization.fit_state(
            MirrorState.from_boundary(boundary, source_grid), source_grid
        ),
        spline_boundary,
        discretization,
        config,
        axial_flux_derivative=0.1,
        mass_profile=100.0 * (1.0 - s),
        current_derivative=current,
        solve_lambda=True,
        gradient_tolerance=1.0e-12,
        require_convergence=True,
    )
    parameters = spline_fixed_boundary_parameters(
        spline_boundary,
        axial_flux_derivative=0.1,
        mass_profile=100.0 * (1.0 - s),
        current_derivative=current,
    )

    adjoint = spline_fixed_boundary_adjoint(
        result,
        parameters,
        discretization,
        lambda _state, energy: energy.geometry.volume,
        solve_lambda=True,
        rtol=1.0e-9,
    )
    direction = jnp.zeros_like(spline_boundary.radius_coefficients).at[0, 2].set(0.01)
    predicted = float(jnp.vdot(adjoint.gradient.boundary_coefficients, direction))
    values = []
    epsilon = 2.0e-4
    for sign in (-1.0, 1.0):
        varied_boundary = SplineMirrorBoundary(
            spline_boundary.radius_coefficients + sign * epsilon * direction
        )
        varied = solve_spline_fixed_boundary_cli(
            discretization.transfer_boundary(
                result.coefficient_state, spline_boundary, varied_boundary
            ),
            varied_boundary,
            discretization,
            config,
            axial_flux_derivative=0.1,
            mass_profile=parameters.mass_profile,
            current_derivative=current,
            solve_lambda=True,
            gradient_tolerance=1.0e-12,
            require_convergence=True,
        )
        values.append(float(varied.evaluated.energy.geometry.volume))
    finite_difference = (values[1] - values[0]) / (2.0 * epsilon)

    assert adjoint.converged
    assert adjoint.relative_residual < 1.0e-8
    assert float(jnp.max(jnp.abs(result.evaluated.state.lambda_stream))) > 1.0e-4
    np.testing.assert_allclose(predicted, finite_difference, rtol=2.0e-5, atol=1.0e-9)


@pytest.mark.parametrize(
    "closure",
    [None, IsotropicPressureClosure(jnp.asarray([2.0e3, -2.0e3]))],
)
def test_custom_vjp_matches_explicit_fixed_boundary_adjoint(closure) -> None:
    config = MirrorConfig(
        resolution=MirrorResolution(ns=3, nxi=5), ftol=1.0e-12, max_iterations=500
    )
    grid = config.build_grid()
    xi, s = jnp.asarray(grid.xi), jnp.asarray(grid.s)
    boundary = MirrorBoundary.from_radius(0.3 * (1.0 + 0.1 * (1.0 - xi**2)), grid)
    parameters = fixed_boundary_parameters(
        boundary,
        axial_flux_derivative=0.1,
        mass_profile=2.0e-4 * (1.0 - s) if closure is None else 0.0,
        current_derivative=1.0e-3 * s,
        pressure_closure=closure,
    )
    initial = MirrorState.from_boundary(boundary, grid)
    common = dict(
        axial_flux_derivative=parameters.axial_flux_derivative,
        current_derivative=parameters.current_derivative,
        solve_lambda=True,
        require_convergence=True,
    )
    if closure is None:
        result = solve_fixed_boundary_cli(
            initial,
            boundary,
            grid,
            config,
            mass_profile=parameters.mass_profile,
            **common,
        )
    else:
        result = solve_anisotropic_fixed_boundary_cli(
            initial, boundary, grid, config, closure, **common
        )

    def quantity(state, _energy):
        return state.radius_scale[1, 0, grid.nxi // 2]

    reference = fixed_boundary_adjoint(
        result, parameters, grid, quantity, solve_lambda=True, rtol=1.0e-10
    )
    implicit_config = make_fixed_boundary_implicit_config(
        initial, grid, config, solve_lambda=True
    )
    gradient = jax.jit(
        jax.grad(
            lambda controls: solve_fixed_boundary_implicit(
                controls, implicit_config
            ).radius_scale[1, 0, grid.nxi // 2]
        )
    )(parameters)

    for actual, expected in zip(
        jax.tree.leaves(gradient), jax.tree.leaves(reference.gradient), strict=True
    ):
        np.testing.assert_allclose(actual, expected, rtol=2.0e-9, atol=2.0e-11)


@pytest.mark.parametrize(
    ("closure", "closure_direction"),
    [
        (
            IsotropicPressureClosure(jnp.asarray([2.0e3, -2.0e3])),
            IsotropicPressureClosure(jnp.asarray([100.0, -100.0])),
        ),
        (
            BiMaxwellianPressureClosure(
                jnp.asarray([2.0e3, -2.0e3]),
                jnp.asarray([0.2]),
                0.7,
                2.0,
                gamma=0.0,
            ),
            BiMaxwellianPressureClosure(
                jnp.asarray([100.0, -100.0]),
                jnp.asarray([0.05]),
                0.7,
                2.0,
                gamma=0.0,
            ),
        ),
    ],
)
def test_anisotropic_closure_adjoint_matches_central_difference(
    closure, closure_direction
) -> None:
    config = MirrorConfig(
        resolution=MirrorResolution(ns=3, nxi=5), ftol=1.0e-12, max_iterations=500
    )
    grid = config.build_grid()
    xi, s = jnp.asarray(grid.xi), jnp.asarray(grid.s)
    boundary = MirrorBoundary.from_radius(0.3 * (1.0 + 0.12 * (1.0 - xi**2)), grid)
    current = 1.0e-2 * s
    result = solve_anisotropic_fixed_boundary_cli(
        MirrorState.from_boundary(boundary, grid),
        boundary,
        grid,
        config,
        closure,
        axial_flux_derivative=0.1,
        current_derivative=current,
        solve_lambda=True,
        require_convergence=True,
    )
    parameters = fixed_boundary_parameters(
        boundary,
        axial_flux_derivative=0.1,
        current_derivative=current,
        pressure_closure=closure,
    )
    def quantity(state, _energy):
        return state.radius_scale[1, 0, grid.nxi // 2]
    adjoint = fixed_boundary_adjoint(
        result, parameters, grid, quantity, solve_lambda=True, rtol=1.0e-10
    )
    boundary_direction = 0.002 * (1.0 - xi**2)[None, :]
    flux_direction = 0.1
    current_direction = 0.05 * s
    closure_contribution = sum(
        jnp.vdot(gradient, direction)
        for gradient, direction in zip(
            jax.tree.leaves(adjoint.gradient.pressure_closure),
            jax.tree.leaves(closure_direction),
            strict=True,
        )
    )
    predicted = float(
        jnp.vdot(adjoint.gradient.boundary_radius, boundary_direction)
        + adjoint.gradient.axial_flux_derivative * flux_direction
        + jnp.vdot(adjoint.gradient.current_derivative, current_direction)
        + closure_contribution
    )
    epsilon = 1.0e-4
    values = []
    for sign in (-1.0, 1.0):
        varied_boundary = MirrorBoundary(
            boundary.radius_scale + sign * epsilon * boundary_direction
        )
        varied_closure = jax.tree.map(
            lambda value, direction: value + sign * epsilon * direction,
            closure,
            closure_direction,
        )
        varied = solve_anisotropic_fixed_boundary_cli(
            project_fixed_boundary_state(result.state, varied_boundary, grid),
            varied_boundary,
            grid,
            config,
            varied_closure,
            axial_flux_derivative=0.1 + sign * epsilon * flux_direction,
            current_derivative=current + sign * epsilon * current_direction,
            solve_lambda=True,
            require_convergence=True,
        )
        values.append(float(quantity(varied.state, varied.energy)))
    finite_difference = (values[1] - values[0]) / (2.0 * epsilon)

    assert adjoint.converged
    assert adjoint.relative_residual < 1.0e-10
    np.testing.assert_allclose(predicted, finite_difference, rtol=2.0e-7)

    class AnisotropicResult:
        converged = True
        energy = object()

    with pytest.raises(ValueError, match="energy model"):
        fixed_boundary_adjoint(
            AnisotropicResult(),
            parameters,
            grid,
            lambda state, energy: energy.total,
        )


def test_tabulated_closure_adjoint_matches_central_difference() -> None:
    config = MirrorConfig(
        resolution=MirrorResolution(ns=3, nxi=5), ftol=1.0e-12, max_iterations=500
    )
    grid = config.build_grid()
    xi, s = jnp.asarray(grid.xi), jnp.asarray(grid.s)
    boundary = MirrorBoundary.from_radius(0.3 * (1.0 + 0.12 * (1.0 - xi**2)), grid)
    s_nodes = jnp.asarray([0.0, 0.5, 1.0])
    b_nodes = jnp.asarray([0.1, 1.0, 3.0])
    values = 2.0e3 * (1.0 - s_nodes[:, None]) * (
        1.0 + 0.05 * (b_nodes[None, :] - 1.0)
    )
    closure = TabulatedPressureClosure(s_nodes, b_nodes, values)
    current = 1.0e-2 * s
    result = solve_anisotropic_fixed_boundary_cli(
        MirrorState.from_boundary(boundary, grid),
        boundary,
        grid,
        config,
        closure,
        axial_flux_derivative=0.1,
        current_derivative=current,
        solve_lambda=True,
        require_convergence=True,
    )
    parameters = fixed_boundary_parameters(
        boundary,
        axial_flux_derivative=0.1,
        current_derivative=current,
        pressure_closure=closure,
    )

    def quantity(state, _energy):
        return state.radius_scale[1, 0, grid.nxi // 2]

    adjoint = fixed_boundary_adjoint(
        result, parameters, grid, quantity, solve_lambda=True, rtol=1.0e-10
    )
    direction = 100.0 * (1.0 - s_nodes[:, None]) * jnp.ones_like(values)
    predicted = float(
        jnp.vdot(adjoint.gradient.pressure_closure.parallel_values, direction)
    )
    epsilon = 1.0e-3
    quantities = []
    for sign in (-1.0, 1.0):
        varied_closure = TabulatedPressureClosure(
            s_nodes, b_nodes, values + sign * epsilon * direction
        )
        varied = solve_anisotropic_fixed_boundary_cli(
            project_fixed_boundary_state(result.state, boundary, grid),
            boundary,
            grid,
            config,
            varied_closure,
            axial_flux_derivative=0.1,
            current_derivative=current,
            solve_lambda=True,
            require_convergence=True,
        )
        quantities.append(float(quantity(varied.state, varied.energy)))
    finite_difference = (quantities[1] - quantities[0]) / (2.0 * epsilon)

    assert adjoint.converged
    assert adjoint.relative_residual < 1.0e-10
    np.testing.assert_allclose(predicted, finite_difference, rtol=2.0e-7)


@pytest.mark.full
def test_fixed_boundary_adjoint_closes_above_dense_reference_limit() -> None:
    config = MirrorConfig(
        resolution=MirrorResolution(ns=17, nxi=41),
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
    result = solve_fixed_boundary_cli(
        initial,
        boundary,
        grid,
        config,
        axial_flux_derivative=0.1,
        require_convergence=True,
    )
    adjoint = fixed_boundary_adjoint(
        result,
        fixed_boundary_parameters(boundary, axial_flux_derivative=0.1),
        grid,
        lambda state, _energy: state.radius_scale[
            grid.ns // 2, 0, grid.nxi // 2
        ],
        rtol=1.0e-9,
    )
    block_adjoint = fixed_boundary_adjoint(
        result,
        fixed_boundary_parameters(boundary, axial_flux_derivative=0.1),
        grid,
        lambda state, _energy: state.radius_scale[
            grid.ns // 2, 0, grid.nxi // 2
        ],
        linear_solver="block",
        rtol=1.0e-9,
    )

    assert (grid.ns - 2) * (grid.nxi - 2) > 512
    assert adjoint.converged
    assert adjoint.iterations < 250
    assert adjoint.relative_residual < 1.0e-8
    assert block_adjoint.converged
    assert block_adjoint.iterations == 0
    assert block_adjoint.relative_residual < 1.0e-12
    np.testing.assert_allclose(
        block_adjoint.gradient.boundary_radius,
        adjoint.gradient.boundary_radius,
        rtol=1.0e-5,
        atol=5.0e-10,
    )
