"""Continuation drivers for free-boundary mirror equilibria."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .forces import MU0, mass_profile_from_pressure, mirror_energy
from .model import MirrorBoundary, MirrorConfig, MirrorState, PressureClosure, project_fixed_boundary_state
from .free_boundary import FreeBoundaryMirrorResult, solve_axisymmetric_free_boundary_cli
from .output import FreeBoundaryRestart
from .vacuum import VacuumGrid

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
        raise ValueError(f"boundary shape {jnp.shape(boundary.radius_scale)} must be {expected_boundary_shape}")
    if source_grid.ntheta != target_grid.ntheta or not np.allclose(
        source_grid.theta, target_grid.theta, rtol=0.0, atol=2.0e-14
    ):
        raise ValueError("state interpolation requires identical theta grids")

    def interpolate(values: Array) -> Array:
        axial = source_grid.axial_basis.interpolate(values, target_grid.xi, axis=2)
        columns = axial.reshape(source_grid.ns, -1).T
        radial = jax.vmap(lambda column: jnp.interp(target_grid.s, source_grid.s, column))(columns)
        return radial.T.reshape(target_grid.shape)

    candidate = MirrorState(
        radius_scale=interpolate(state.radius_scale),
        lambda_stream=interpolate(state.lambda_stream),
    )
    return project_fixed_boundary_state(candidate, boundary, target_grid)


def solve_beta_scan_cli(
    initial_boundary: MirrorBoundary,
    plasma_grid: "MirrorGrid",
    vacuum_grid: VacuumGrid,
    config: MirrorConfig,
    external_field: Any,
    beta_values: Array,
    *,
    outer_radius: float,
    axial_flux_derivative: Array,
    reference_field: float,
    current_derivative: Array = 0.0,
    gamma: float = 5.0 / 3.0,
    beta_rtol: float = 1.0e-8,
    initial_restart: FreeBoundaryRestart | None = None,
    pressure_closure: PressureClosure | None = None,
    vacuum_backend: str = "annulus",
    exterior_ntheta: int = 40,
    exterior_order: int = 8,
    exterior_spectral_side_density: bool = False,
    exterior_high_order_cap_panels: bool = False,
    exterior_curved_side_geometry: bool = False,
    exterior_jacobian_chunk_size: int = 6,
) -> tuple[FreeBoundaryMirrorResult, ...]:
    """Solve a fully hot-started free-boundary mirror beta scan.

    Each accepted point supplies the boundary, plasma interior, and vacuum
    potential for the next pressure value. The coupled nonlinear system adds
    one mass-amplitude unknown and one central-pressure equation so requested
    beta is achieved without an outer sequence of full equilibrium solves.
    ``initial_restart`` resumes a suffix of the scan while ``initial_boundary``
    remains the beta-zero reference used to construct the pressure profile.
    The opt-in ``vacuum_backend="exterior"`` hot-starts only plasma and
    boundary variables because its boundary-integral potential is eliminated
    exactly at every nonlinear evaluation.
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
        current_derivative=current_derivative,
    )
    pressure_shape = 1.0 - jnp.asarray(plasma_grid.s)
    boundary = initial_boundary if initial_restart is None else initial_restart.boundary
    state = None if initial_restart is None else initial_restart.plasma_state
    potential = None if initial_restart is None else initial_restart.vacuum_potential
    mass_scale = 1.0 if initial_restart is None else initial_restart.mass_scale
    using_closure = initial_restart is not None and pressure_closure is not None
    results = []
    for beta in beta_values:
        central_pressure = float(beta) * float(reference_field) ** 2 / (2.0 * MU0)
        active_closure = pressure_closure if beta > 0.0 else None
        mass = (
            mass_profile_from_pressure(
                central_pressure * pressure_shape,
                reference_energy.volume_derivative,
                gamma=gamma,
            )
            if active_closure is None
            else 0.0
        )
        if active_closure is not None and not using_closure:
            base_pressure = float(active_closure.moments(0.0, float(reference_field)).perpendicular)
            if base_pressure <= 0.0:
                raise ValueError("pressure_closure must have positive central p_perp")
            mass_scale = central_pressure / base_pressure
        result = solve_axisymmetric_free_boundary_cli(
            boundary,
            plasma_grid,
            vacuum_grid,
            config,
            external_field,
            outer_radius=outer_radius,
            axial_flux_derivative=axial_flux_derivative,
            current_derivative=current_derivative,
            mass_profile=mass,
            gamma=gamma,
            initial_state=state,
            initial_potential=potential,
            initial_mass_scale=mass_scale,
            pressure_closure=active_closure,
            vacuum_backend=vacuum_backend,
            exterior_ntheta=exterior_ntheta,
            exterior_order=exterior_order,
            exterior_spectral_side_density=exterior_spectral_side_density,
            exterior_high_order_cap_panels=exterior_high_order_cap_panels,
            exterior_curved_side_geometry=exterior_curved_side_geometry,
            exterior_jacobian_chunk_size=exterior_jacobian_chunk_size,
            target_central_pressure=None if beta == 0.0 else central_pressure,
            require_convergence=True,
        )
        if beta > 0.0:
            center = int(np.argmin(np.abs(plasma_grid.z)))
            achieved_beta = 2.0 * MU0 * float(result.perpendicular_pressure[0, 0, center]) / float(reference_field) ** 2
            if abs(achieved_beta - float(beta)) / float(beta) > beta_rtol:
                raise RuntimeError(f"central beta did not reach rtol={beta_rtol:.3e}")
        results.append(result)
        boundary = result.boundary
        state = result.plasma_state
        potential = result.vacuum_potential
        mass_scale = float(result.mass_scale)
        using_closure = active_closure is not None
    return tuple(results)


solve_axisymmetric_beta_scan_cli = solve_beta_scan_cli


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .basis import MirrorGrid
