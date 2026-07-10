"""Continuation drivers for free-boundary mirror equilibria."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .forces import MU0, mass_profile_from_pressure, mirror_energy
from .model import MirrorBoundary, MirrorConfig, MirrorState, project_fixed_boundary_state
from .vacuum import FreeBoundaryMirrorResult, VacuumGrid, solve_axisymmetric_free_boundary_cli

Array = Any


def interpolate_fixed_boundary_state(
    state: MirrorState,
    source_grid: "MirrorGrid",
    boundary: MirrorBoundary,
    target_grid: "MirrorGrid",
) -> MirrorState:
    """Interpolate a converged fixed-boundary state to a new ``ns/nxi`` grid.

    Axial values use the source CGL barycentric interpolant; radial values use
    piecewise-linear interpolation in normalized flux. Poloidal nodes must be
    unchanged so no Fourier information is silently added or discarded. The
    target boundary, axis closure, end cuts, and lambda gauge are projected
    after interpolation.
    """

    state.validate_shape(source_grid)
    expected_boundary_shape = (target_grid.ntheta, target_grid.nxi)
    if tuple(jnp.shape(boundary.radius_scale)) != expected_boundary_shape:
        raise ValueError(
            f"boundary shape {jnp.shape(boundary.radius_scale)} must be "
            f"{expected_boundary_shape}"
        )
    if source_grid.ntheta != target_grid.ntheta or not np.allclose(
        source_grid.theta, target_grid.theta, rtol=0.0, atol=2.0e-14
    ):
        raise ValueError("state interpolation requires identical theta grids")

    def interpolate(values: Array) -> Array:
        axial = source_grid.axial_basis.interpolate(
            values, target_grid.xi, axis=2
        )
        columns = axial.reshape(source_grid.ns, -1).T
        radial = jax.vmap(
            lambda column: jnp.interp(target_grid.s, source_grid.s, column)
        )(columns)
        return radial.T.reshape(target_grid.shape)

    candidate = MirrorState(
        radius_scale=interpolate(state.radius_scale),
        lambda_stream=interpolate(state.lambda_stream),
    )
    return project_fixed_boundary_state(candidate, boundary, target_grid)


def solve_axisymmetric_beta_scan_cli(
    initial_boundary: MirrorBoundary,
    plasma_grid: "MirrorGrid",
    vacuum_grid: VacuumGrid,
    config: MirrorConfig,
    coilset: Any,
    beta_values: Array,
    *,
    outer_radius: float,
    axial_flux_derivative: Array,
    reference_field: float,
    gamma: float = 5.0 / 3.0,
    beta_rtol: float = 1.0e-8,
) -> tuple[FreeBoundaryMirrorResult, ...]:
    """Solve a fully hot-started axisymmetric free-boundary beta scan.

    Each accepted point supplies the boundary, plasma interior, and vacuum
    potential for the next pressure value. The coupled nonlinear system adds
    one mass-amplitude unknown and one central-pressure equation so requested
    beta is achieved without an outer sequence of full equilibrium solves.
    """

    beta_values = np.asarray(beta_values, dtype=float)
    if beta_values.ndim != 1 or beta_values.size < 1:
        raise ValueError("beta_values must be a nonempty one-dimensional array")
    if np.any(beta_values < 0.0) or np.any(np.diff(beta_values) < 0.0):
        raise ValueError("beta_values must be nonnegative and increasing")
    if beta_rtol <= 0.0:
        raise ValueError("beta_rtol must be positive")
    reference_state = MirrorState.from_boundary(initial_boundary, plasma_grid)
    reference_energy = mirror_energy(
        reference_state,
        plasma_grid,
        axial_flux_derivative=axial_flux_derivative,
    )
    pressure_shape = 1.0 - jnp.asarray(plasma_grid.s)
    boundary = initial_boundary
    state = None
    potential = None
    results = []
    for beta in beta_values:
        central_pressure = float(beta) * float(reference_field) ** 2 / (2.0 * MU0)
        mass = mass_profile_from_pressure(
            central_pressure * pressure_shape,
            reference_energy.volume_derivative,
            gamma=gamma,
        )
        result = solve_axisymmetric_free_boundary_cli(
            boundary,
            plasma_grid,
            vacuum_grid,
            config,
            coilset,
            outer_radius=outer_radius,
            axial_flux_derivative=axial_flux_derivative,
            mass_profile=mass,
            gamma=gamma,
            initial_state=state,
            initial_potential=potential,
            target_central_pressure=None if beta == 0.0 else central_pressure,
            require_convergence=True,
        )
        if beta > 0.0:
            achieved_beta = 2.0 * MU0 * float(result.plasma_energy.pressure[0]) / float(reference_field) ** 2
            if abs(achieved_beta - float(beta)) / float(beta) > beta_rtol:
                raise RuntimeError(f"central beta did not reach rtol={beta_rtol:.3e}")
        results.append(result)
        boundary = result.boundary
        state = result.plasma_state
        potential = result.vacuum_potential
    return tuple(results)


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .basis import MirrorGrid
