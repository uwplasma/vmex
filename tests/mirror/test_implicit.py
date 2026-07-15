"""Implicit differentiation of converged mirror equilibria."""

from __future__ import annotations

from dataclasses import dataclass, replace
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
    free_boundary_adjoint,
    solve_fixed_boundary_cli,
    solve_fixed_boundary_implicit,
    solve_free_boundary_cli,
    spline_fixed_boundary_adjoint,
    spline_fixed_boundary_tangent,
)
from vmec_jax.mirror.implicit import (  # noqa: E402
    FreeBoundaryAdjointConfig,
    fixed_boundary_parameters,
    free_boundary_parameters,
    make_fixed_boundary_implicit_config,
    spline_fixed_boundary_parameters,
)
from vmec_jax.mirror.model import project_fixed_boundary_state  # noqa: E402
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
        varied_boundary = MirrorBoundary(boundary.radius_scale + sign * epsilon * boundary_direction)
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
        fixed_boundary_adjoint(Result(), parameters, grid, lambda state, energy: energy.total)


def test_spline_adjoint_matches_reconverged_central_difference() -> None:
    config = MirrorConfig(
        resolution=MirrorResolution(ns=5, nxi=9),
        ftol=1.0e-12,
        max_iterations=1000,
    )
    source_grid = config.build_grid()
    s, xi = jnp.asarray(source_grid.s), jnp.asarray(source_grid.xi)
    boundary = MirrorBoundary.from_radius(0.3 * (1.0 + 0.1 * (1.0 - xi**2)), source_grid)
    discretization = SplineMirrorDiscretization.build(config, elements=3)
    spline_boundary = discretization.fit_boundary(boundary, source_grid)
    mass = 2.0e3 * (1.0 - s)
    current = 1.0e-2 * s
    result = solve_spline_fixed_boundary_cli(
        discretization.fit_state(MirrorState.from_boundary(boundary, source_grid), source_grid),
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
            spline_boundary.radius_coefficients + sign * epsilon * boundary_direction
        )
        initial = discretization.transfer_boundary(result.coefficient_state, spline_boundary, varied_boundary)
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
    boundary = MirrorBoundary.from_radius(0.3 * (1.0 + 0.02 * jnp.cos(theta) * (1.0 - xi**2)), source_grid)
    discretization = SplineMirrorDiscretization.build(config, elements=3)
    spline_boundary = discretization.fit_boundary(boundary, source_grid)
    current = 1.0e-3 * s
    result = solve_spline_fixed_boundary_cli(
        discretization.fit_state(MirrorState.from_boundary(boundary, source_grid), source_grid),
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
    zero_tangent = jax.tree.map(jnp.zeros_like, parameters)
    tangent = spline_fixed_boundary_tangent(
        result,
        parameters,
        replace(zero_tangent, boundary_coefficients=direction),
        discretization,
        solve_lambda=True,
        rtol=1.0e-9,
    )
    predicted = float(jnp.vdot(adjoint.gradient.boundary_coefficients, direction))
    values = []
    states = []
    epsilon = 2.0e-4
    for sign in (-1.0, 1.0):
        varied_boundary = SplineMirrorBoundary(spline_boundary.radius_coefficients + sign * epsilon * direction)
        varied = solve_spline_fixed_boundary_cli(
            discretization.transfer_boundary(result.coefficient_state, spline_boundary, varied_boundary),
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
        states.append(varied.evaluated.state)
    finite_difference = (values[1] - values[0]) / (2.0 * epsilon)
    state_difference = jax.tree.map(
        lambda upper, lower: (upper - lower) / (2.0 * epsilon),
        states[1],
        states[0],
    )

    assert adjoint.converged
    assert adjoint.relative_residual < 1.0e-8
    assert tangent.converged
    assert tangent.relative_residual < 1.0e-8
    assert float(jnp.max(jnp.abs(result.evaluated.state.lambda_stream))) > 1.0e-4
    np.testing.assert_allclose(predicted, finite_difference, rtol=2.0e-5, atol=1.0e-9)
    for actual, expected in zip(
        jax.tree.leaves(tangent.tangent),
        jax.tree.leaves(state_difference),
        strict=True,
    ):
        relative_error = jnp.linalg.norm(actual - expected) / jnp.maximum(
            jnp.linalg.norm(expected), jnp.finfo(expected.dtype).tiny
        )
        assert float(relative_error) < 2.0e-4


def test_custom_vjp_matches_explicit_fixed_boundary_adjoint() -> None:
    config = MirrorConfig(resolution=MirrorResolution(ns=3, nxi=5), ftol=1.0e-12, max_iterations=500)
    grid = config.build_grid()
    xi, s = jnp.asarray(grid.xi), jnp.asarray(grid.s)
    boundary = MirrorBoundary.from_radius(0.3 * (1.0 + 0.1 * (1.0 - xi**2)), grid)
    parameters = fixed_boundary_parameters(
        boundary,
        axial_flux_derivative=0.1,
        mass_profile=2.0e-4 * (1.0 - s),
        current_derivative=1.0e-3 * s,
    )
    initial = MirrorState.from_boundary(boundary, grid)
    common = dict(
        axial_flux_derivative=parameters.axial_flux_derivative,
        current_derivative=parameters.current_derivative,
        solve_lambda=True,
        require_convergence=True,
    )
    result = solve_fixed_boundary_cli(
        initial,
        boundary,
        grid,
        config,
        mass_profile=parameters.mass_profile,
        **common,
    )

    def quantity(state, _energy):
        return state.radius_scale[1, 0, grid.nxi // 2]

    reference = fixed_boundary_adjoint(result, parameters, grid, quantity, solve_lambda=True, rtol=1.0e-10)
    implicit_config = make_fixed_boundary_implicit_config(initial, grid, config, solve_lambda=True)
    gradient = jax.jit(
        jax.grad(
            lambda controls: solve_fixed_boundary_implicit(controls, implicit_config).radius_scale[1, 0, grid.nxi // 2]
        )
    )(parameters)

    for actual, expected in zip(jax.tree.leaves(gradient), jax.tree.leaves(reference.gradient), strict=True):
        np.testing.assert_allclose(actual, expected, rtol=2.0e-9, atol=2.0e-11)


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
        lambda state, _energy: state.radius_scale[grid.ns // 2, 0, grid.nxi // 2],
        rtol=1.0e-9,
    )
    block_adjoint = fixed_boundary_adjoint(
        result,
        fixed_boundary_parameters(boundary, axial_flux_derivative=0.1),
        grid,
        lambda state, _energy: state.radius_scale[grid.ns // 2, 0, grid.nxi // 2],
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


@dataclass(frozen=True)
class ParaxialMirrorField:
    """Differentiable divergence-free field with mirror-like axial strength."""

    center_field: jax.Array
    curvature: jax.Array

    def __call__(self, points):
        points = jnp.asarray(points)
        x, y, z = jnp.moveaxis(points, -1, 0)
        return jnp.stack(
            (
                -self.curvature * x * z,
                -self.curvature * y * z,
                self.center_field + self.curvature * z**2,
            ),
            axis=-1,
        )


jax.tree_util.register_dataclass(
    ParaxialMirrorField,
    data_fields=["center_field", "curvature"],
    meta_fields=[],
)


def test_free_boundary_adjoint_rejects_unconverged_and_3d_results() -> None:
    axisymmetric_grid = MirrorConfig(resolution=MirrorResolution(ns=3, nxi=5)).build_grid()
    parameters = free_boundary_parameters(object(), axial_flux_derivative=0.1)
    with pytest.raises(ValueError, match="converged"):
        free_boundary_adjoint(
            type("Result", (), {"converged": False})(),
            parameters,
            axisymmetric_grid,
            lambda *_: 0.0,
        )

    nonaxisymmetric_grid = MirrorConfig(resolution=MirrorResolution(ns=3, mpol=1, ntheta=3, nxi=5)).build_grid()
    with pytest.raises(ValueError, match="axisymmetry"):
        free_boundary_adjoint(
            type("Result", (), {"converged": True})(),
            parameters,
            nonaxisymmetric_grid,
            lambda *_: 0.0,
        )


@pytest.mark.full
def test_free_boundary_field_adjoint_matches_central_difference() -> None:
    config = MirrorConfig(
        resolution=MirrorResolution(ns=5, nxi=7),
        z_min=-0.8,
        z_max=0.8,
        ftol=1.0e-12,
        max_iterations=300,
    )
    grid = config.build_grid()
    field = ParaxialMirrorField(jnp.asarray(0.08), jnp.asarray(0.02))
    on_axis = field.center_field + field.curvature * jnp.asarray(grid.z) ** 2
    center = grid.nxi // 2
    flux = 0.5 * on_axis[center] * 0.25**2
    initial_boundary = MirrorBoundary.from_axis_field(flux, on_axis, grid)
    solve_options = dict(
        axial_flux_derivative=flux,
        exterior_ntheta=8,
        exterior_order=6,
        exterior_spectral_side_density=True,
        require_convergence=True,
    )
    result = solve_free_boundary_cli(initial_boundary, grid, config, field, **solve_options)
    parameters = free_boundary_parameters(field, axial_flux_derivative=flux)

    def quantity(boundary, _state, _energy, _vacuum):
        return boundary.radius_scale[0, center]

    adjoint = free_boundary_adjoint(
        result,
        parameters,
        grid,
        quantity,
        config=FreeBoundaryAdjointConfig(
            axisymmetric_ntheta=8,
            exterior_order=6,
            spectral_side_density=True,
            rtol=1.0e-8,
        ),
    )
    direction = ParaxialMirrorField(jnp.asarray(0.01), jnp.asarray(-0.005))
    predicted = float(
        adjoint.gradient.external_field.center_field * direction.center_field
        + adjoint.gradient.external_field.curvature * direction.curvature
    )

    epsilon = 1.0e-4
    values = []
    for sign in (-1.0, 1.0):
        varied_field = ParaxialMirrorField(
            field.center_field + sign * epsilon * direction.center_field,
            field.curvature + sign * epsilon * direction.curvature,
        )
        varied = solve_free_boundary_cli(
            result.boundary,
            grid,
            config,
            varied_field,
            initial_state=result.plasma_state,
            **solve_options,
        )
        values.append(float(quantity(varied.boundary, None, None, None)))
    finite_difference = (values[1] - values[0]) / (2.0 * epsilon)
    assert result.converged
    assert float(result.variational_max) <= config.ftol
    assert adjoint.converged
    assert adjoint.relative_residual < 1.0e-8
    np.testing.assert_allclose(predicted, finite_difference, rtol=2.0e-4)
