"""Continuation drivers for free-boundary mirror equilibria."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from .forces import MU0, mass_profile_from_pressure, mirror_energy
from .model import MirrorBoundary, MirrorConfig, MirrorState
from .vacuum import FreeBoundaryMirrorResult, VacuumGrid, solve_axisymmetric_free_boundary_cli

Array = Any


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
) -> tuple[FreeBoundaryMirrorResult, ...]:
    """Solve a fully hot-started axisymmetric free-boundary beta scan.

    Each accepted point supplies the boundary, plasma interior, and vacuum
    potential for the next pressure value. The conserved mass profile is
    always defined from the original reference state, so continuation does
    not change the physical pressure input.
    """

    beta_values = np.asarray(beta_values, dtype=float)
    if beta_values.ndim != 1 or beta_values.size < 1:
        raise ValueError("beta_values must be a nonempty one-dimensional array")
    if np.any(beta_values < 0.0) or np.any(np.diff(beta_values) < 0.0):
        raise ValueError("beta_values must be nonnegative and increasing")
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
            require_convergence=True,
        )
        results.append(result)
        boundary = result.boundary
        state = result.plasma_state
        potential = result.vacuum_potential
    return tuple(results)


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .basis import MirrorGrid
