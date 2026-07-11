"""Coupled plasma-boundary-vacuum solves for straight-axis mirrors."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

from .forces import (
    AnisotropicMirrorEnergy,
    InterfaceResidual,
    MirrorEnergy,
    anisotropic_mirror_energy,
    interface_residual,
    mirror_energy,
)
from .geometry import magnetic_field_squared
from .exterior_bie import (
    AxisymmetricExteriorVacuum,
    NonaxisymmetricExteriorVacuum,
    solve_axisymmetric_exterior_vacuum,
    solve_nonaxisymmetric_exterior_vacuum,
)
from .model import MirrorBoundary, MirrorConfig, MirrorState, PressureClosure, PressureMoments
from .vacuum import (
    MU0,
    VacuumField,
    VacuumGeometry,
    VacuumGrid,
    evaluate_vacuum_field,
    evaluate_vacuum_geometry,
    external_field_from_source,
    vacuum_energy_functional,
)

Array = Any
_MONOLITHIC_JACOBIAN_MAX_SIZE = 80


@dataclass(frozen=True)
class _ScaledPressureClosure:
    """Multiply a consistent ANIMEC closure by a solved positive amplitude."""

    closure: PressureClosure
    scale: Array

    def parallel_pressure(self, s: Array, magnetic_field_strength: Array) -> Array:
        return self.scale * self.closure.parallel_pressure(s, magnetic_field_strength)

    def moments(self, s: Array, magnetic_field_strength: Array) -> PressureMoments:
        moments = self.closure.moments(s, magnetic_field_strength)
        return PressureMoments(
            parallel=self.scale * moments.parallel,
            perpendicular=self.scale * moments.perpendicular,
            energy_density=self.scale * moments.energy_density,
        )


@dataclass(frozen=True)
class FreeBoundaryMirrorResult:
    """Joint plasma-boundary-vacuum equilibrium result."""

    boundary: MirrorBoundary
    plasma_state: MirrorState
    plasma_energy: MirrorEnergy | AnisotropicMirrorEnergy
    plasma_b_squared: Array
    perpendicular_pressure: Array
    vacuum_geometry: VacuumGeometry | "ClosedMirrorSurface"
    vacuum_field: VacuumField | AxisymmetricExteriorVacuum | NonaxisymmetricExteriorVacuum
    vacuum_potential: Array
    mass_scale: Array
    interface: InterfaceResidual
    history: Array
    variational_max: Array
    iterations: int
    converged: bool
    optimizer_success: bool
    message: str


jax.tree_util.register_dataclass(
    FreeBoundaryMirrorResult,
    data_fields=[
        field.name
        for field in fields(FreeBoundaryMirrorResult)
        if field.name not in {"iterations", "converged", "message"}
    ],
    meta_fields=[
        field.name for field in fields(FreeBoundaryMirrorResult) if field.name in {"iterations", "converged", "message"}
    ],
)


def solve_free_boundary_cli(
    initial_boundary: MirrorBoundary,
    plasma_grid: "MirrorGrid",
    vacuum_grid: VacuumGrid,
    config: MirrorConfig,
    coilset: Any,
    *,
    outer_radius: float,
    axial_flux_derivative: Array,
    mass_profile: Array = 0.0,
    current_derivative: Array = 0.0,
    gamma: float = 5.0 / 3.0,
    initial_state: MirrorState | None = None,
    initial_potential: Array | None = None,
    target_central_pressure: float | None = None,
    initial_mass_scale: float = 1.0,
    pressure_closure: PressureClosure | None = None,
    vacuum_backend: str = "annulus",
    exterior_ntheta: int = 40,
    exterior_order: int = 8,
    exterior_jacobian_chunk_size: int = 6,
    require_convergence: bool = False,
) -> FreeBoundaryMirrorResult:
    """Jointly solve a mirror plasma boundary and its open vacuum.

    The default ``vacuum_backend="annulus"`` retains the finite outer-cylinder
    potential discretization. ``vacuum_backend="exterior"`` instead solves the
    closed-surface free-space Neumann problem at every nonlinear evaluation;
    its active vector contains no vacuum degrees of freedom.
    """

    if plasma_grid.ntheta != vacuum_grid.ntheta:
        raise ValueError("plasma and vacuum grids must share theta nodes")
    if vacuum_backend not in {"annulus", "exterior"}:
        raise ValueError("vacuum_backend must be 'annulus' or 'exterior'")
    use_exterior = vacuum_backend == "exterior"
    if plasma_grid.ntheta != 1 and not use_exterior:
        raise ValueError("nonaxisymmetric free boundary requires vacuum_backend='exterior'")
    exterior_jacobian_chunk_size = int(exterior_jacobian_chunk_size)
    if exterior_jacobian_chunk_size < 1:
        raise ValueError("exterior_jacobian_chunk_size must be positive")
    if plasma_grid.nxi != vacuum_grid.nxi:
        raise ValueError("plasma and vacuum grids must share axial nodes")
    if target_central_pressure is not None and target_central_pressure <= 0.0:
        raise ValueError("target_central_pressure must be positive")
    if initial_mass_scale <= 0.0:
        raise ValueError("initial_mass_scale must be positive")
    if pressure_closure is not None and np.any(np.asarray(mass_profile) != 0.0):
        raise ValueError("mass_profile and pressure_closure are mutually exclusive")
    initial_boundary_radius = np.asarray(initial_boundary.radius_scale, dtype=float)
    if initial_boundary_radius.shape != (plasma_grid.ntheta, plasma_grid.nxi):
        raise ValueError("initial boundary does not match the plasma grid")
    boundary_scale = float(np.mean(initial_boundary_radius))
    flux = np.asarray(axial_flux_derivative, dtype=float)
    flux_scale = max(float(np.max(np.abs(flux))), np.finfo(float).tiny)
    potential_scale = max(
        2.0 * flux_scale / boundary_scale**2 * float(plasma_grid.dz_dxi),
        np.finfo(float).tiny,
    )

    boundary_mask = np.zeros(initial_boundary_radius.shape, dtype=bool)
    boundary_mask[:, 1:-1] = True
    boundary_indices = tuple(
        np.asarray(index) for index in np.nonzero(boundary_mask)
    )
    plasma_mask = np.zeros(plasma_grid.shape, dtype=bool)
    plasma_mask[1:-1, :, 1:-1] = True
    plasma_indices = tuple(np.asarray(index) for index in np.nonzero(plasma_mask))
    vacuum_mask = (
        np.zeros(vacuum_grid.shape, dtype=bool)
        if use_exterior
        else np.ones(vacuum_grid.shape, dtype=bool)
    )
    if not use_exterior:
        vacuum_mask[-1] = False
    vacuum_indices = tuple(np.asarray(index) for index in np.nonzero(vacuum_mask))
    nb = boundary_indices[0].size
    np_state = plasma_indices[0].size
    nv = vacuum_indices[0].size

    base_state = MirrorState.from_boundary(initial_boundary, plasma_grid) if initial_state is None else initial_state
    base_state.validate_shape(plasma_grid)
    if not np.allclose(np.asarray(base_state.radius_scale[-1]), initial_boundary_radius):
        raise ValueError("initial_state boundary must match initial_boundary")
    potential_seed = (
        np.zeros(vacuum_grid.shape)
        if initial_potential is None or use_exterior
        else np.asarray(initial_potential)
    )
    if potential_seed.shape != vacuum_grid.shape:
        raise ValueError(f"initial_potential shape {potential_seed.shape} must be {vacuum_grid.shape}")
    calibrate_pressure = target_central_pressure is not None
    mass_scale_index = nb + np_state + nv
    x0_parts = [
        initial_boundary_radius[boundary_indices] / boundary_scale,
        np.asarray(base_state.radius_scale)[plasma_indices] / boundary_scale,
        potential_seed[vacuum_indices] / potential_scale,
    ]
    if calibrate_pressure:
        x0_parts.append(np.asarray([initial_mass_scale]))
    x0 = np.concatenate(x0_parts)
    geometric_upper = (
        np.inf if use_exterior else 0.98 * float(outer_radius) / boundary_scale
    )
    if np.isfinite(geometric_upper) and np.max(x0[:nb]) >= geometric_upper:
        raise ValueError("initial plasma boundary must lie inside the outer vacuum cylinder")
    lower_parts = [np.full(nb + np_state, 0.2), np.full(nv, -np.inf)]
    upper_parts = [np.full(nb + np_state, geometric_upper), np.full(nv, np.inf)]
    if calibrate_pressure:
        lower_parts.append(np.asarray([np.finfo(float).tiny]))
        upper_parts.append(np.asarray([np.inf]))
    lower, upper = np.concatenate(lower_parts), np.concatenate(upper_parts)

    def unpack(vector: Array) -> tuple[MirrorBoundary, MirrorState, Array, Array]:
        vector = jnp.asarray(vector)
        boundary_radius = jnp.asarray(initial_boundary_radius).at[
            tuple(jnp.asarray(index) for index in boundary_indices)
        ].set(vector[:nb] * boundary_scale)
        boundary = MirrorBoundary(boundary_radius)
        radius = base_state.radius_scale.at[plasma_indices].set(vector[nb : nb + np_state] * boundary_scale)
        radius = radius.at[-1].set(boundary_radius)
        radius = radius.at[:, :, 0].set(boundary_radius[:, 0])
        radius = radius.at[:, :, -1].set(boundary_radius[:, -1])
        radius = radius.at[0].set(radius[1])
        state = MirrorState(radius, base_state.lambda_stream)
        potential = (
            jnp.zeros(vacuum_grid.shape)
            .at[vacuum_indices]
            .set(vector[nb + np_state : nb + np_state + nv] * potential_scale)
        )
        if calibrate_pressure:
            mass_scale = vector[mass_scale_index]
        else:
            mass_scale = jnp.asarray(1.0, dtype=vector.dtype)
        return boundary, state, potential, mass_scale

    center = int(np.argmin(np.abs(plasma_grid.z)))

    def components(vector: Array):
        boundary, state, potential, mass_scale = unpack(vector)
        if pressure_closure is None:
            plasma = mirror_energy(
                state,
                plasma_grid,
                axial_flux_derivative=axial_flux_derivative,
                mass_profile=jnp.asarray(mass_profile) * mass_scale,
                current_derivative=current_derivative,
                gamma=gamma,
            )
            plasma_b_squared = plasma.b_squared
            perpendicular_pressure = jnp.broadcast_to(plasma.pressure[:, None, None], plasma_b_squared.shape)
            central_pressure = plasma.pressure[0]
            anisotropy_valid = jnp.asarray(True)
        else:
            scaled_closure = _ScaledPressureClosure(pressure_closure, mass_scale)
            plasma = anisotropic_mirror_energy(
                state,
                plasma_grid,
                scaled_closure,
                axial_flux_derivative=axial_flux_derivative,
                current_derivative=current_derivative,
            )
            plasma_b_squared = magnetic_field_squared(plasma.field, plasma.geometry)
            full_moments = scaled_closure.moments(
                jnp.asarray(plasma_grid.s)[:, None, None],
                jnp.sqrt(jnp.maximum(plasma_b_squared, 0.0)),
            )
            perpendicular_pressure = full_moments.perpendicular
            central_pressure = perpendicular_pressure[0, 0, center]
            anisotropy_valid = jnp.all(plasma.indicators_half.valid)
        if use_exterior:
            if plasma_grid.ntheta == 1:
                vacuum_field = solve_axisymmetric_exterior_vacuum(
                    boundary,
                    plasma.field,
                    plasma_grid,
                    coilset,
                    axisymmetric_ntheta=exterior_ntheta,
                    order=exterior_order,
                )
            else:
                vacuum_field = solve_nonaxisymmetric_exterior_vacuum(
                    boundary,
                    plasma.field,
                    plasma.geometry,
                    plasma_grid,
                    coilset,
                    order=exterior_order,
                )
            vacuum_geometry = vacuum_field.surface
            vacuum_functional = jnp.asarray(0.0, dtype=state.radius_scale.dtype)
        else:
            vacuum_geometry = evaluate_vacuum_geometry(boundary, vacuum_grid, outer_radius=outer_radius)
            external = external_field_from_source(coilset, vacuum_geometry)
            vacuum_functional = vacuum_energy_functional(
                potential,
                vacuum_geometry,
                vacuum_grid,
                external,
                boundary_condition="decaying_outer",
            )
            vacuum_field = evaluate_vacuum_field(potential, vacuum_geometry, vacuum_grid, external)
        return (
            plasma,
            plasma_b_squared,
            perpendicular_pressure,
            central_pressure,
            anisotropy_valid,
            vacuum_geometry,
            vacuum_field,
            vacuum_functional,
        )

    initial_components = components(jnp.asarray(x0))
    plasma_scale = max(abs(float(initial_components[0].total)), 1.0)
    vacuum_scale = max(abs(float(initial_components[7])), 1.0)

    def plasma_objective(vector: Array) -> Array:
        return components(vector)[0].total / plasma_scale

    def vacuum_objective(vector: Array) -> Array:
        return components(vector)[7] / vacuum_scale

    def residual_function(vector: Array) -> Array:
        (
            plasma,
            plasma_b_squared,
            perpendicular_pressure,
            central_pressure,
            _,
            _,
            vacuum_field,
            _,
        ) = components(vector)
        plasma_gradient = jax.grad(plasma_objective)(vector)[nb : nb + np_state]
        vacuum_gradient = (
            jnp.empty((0,), dtype=vector.dtype)
            if use_exterior
            else jax.grad(vacuum_objective)(vector)[
                nb + np_state : nb + np_state + nv
            ]
        )
        plasma_b_squared = plasma_b_squared[-1, :, 1:-1].reshape(-1)
        if use_exterior:
            vacuum_xyz = vacuum_field.lateral_field_xyz
            if plasma_grid.ntheta == 1:
                vacuum_xyz = vacuum_xyz[None]
            vacuum_xyz = vacuum_xyz[:, 1:-1].reshape(-1, 3)
        else:
            vacuum_xyz = vacuum_field.total_xyz[0, :, 1:-1].reshape(-1, 3)
        vacuum_b_squared = jnp.sum(vacuum_xyz**2, axis=-1)
        pressure = perpendicular_pressure[-1, :, 1:-1].reshape(-1)
        jump = pressure + plasma_b_squared / (2.0 * MU0) - vacuum_b_squared / (2.0 * MU0)
        stress_scale = jnp.abs(pressure) + plasma_b_squared / (2.0 * MU0) + vacuum_b_squared / (2.0 * MU0)
        stress = jump / jnp.maximum(stress_scale, jnp.finfo(stress_scale.dtype).tiny)
        residuals = [stress, plasma_gradient, vacuum_gradient]
        if calibrate_pressure:
            target = float(target_central_pressure)
            residuals.append(jnp.asarray([(central_pressure - target) / target]))
        return jnp.concatenate(residuals)

    residual_jit = jax.jit(residual_function)
    jacobian_jit = jax.jit(jax.jacfwd(residual_function))
    jvp_batch_jit = jax.jit(
        jax.vmap(
            lambda primal, tangent: jax.jvp(
                residual_function, (primal,), (tangent,)
            )[1],
            in_axes=(None, 0),
        )
    )

    history: list[tuple[float, float, float, float, float]] = []
    last_recorded: np.ndarray | None = None

    def residual_host(vector: np.ndarray) -> np.ndarray:
        nonlocal last_recorded
        residual = np.asarray(residual_jit(jnp.asarray(vector)), dtype=float)
        if last_recorded is None or not np.array_equal(vector, last_recorded):
            history.append(
                (
                    float(len(history)),
                    float(np.sqrt(np.mean(residual[:nb] ** 2))),
                    float(np.sqrt(np.mean(residual[nb : nb + np_state] ** 2))),
                    float(np.sqrt(np.mean(residual[nb + np_state : nb + np_state + nv] ** 2))) if nv else 0.0,
                    float(np.max(np.abs(residual))),
                )
            )
            last_recorded = np.array(vector, copy=True)
        return residual

    def jacobian_host(vector: np.ndarray) -> np.ndarray:
        if not use_exterior or vector.size <= _MONOLITHIC_JACOBIAN_MAX_SIZE:
            return np.asarray(jacobian_jit(jnp.asarray(vector)), dtype=float)
        size = vector.size
        columns = []
        identity = np.eye(size)
        for start in range(0, size, exterior_jacobian_chunk_size):
            stop = min(start + exterior_jacobian_chunk_size, size)
            columns.append(
                np.asarray(
                    jvp_batch_jit(
                        jnp.asarray(vector), jnp.asarray(identity[start:stop])
                    ),
                    dtype=float,
                )
            )
        return np.concatenate(columns, axis=0).T

    solve = least_squares(
        fun=residual_host,
        x0=x0,
        jac=jacobian_host,
        bounds=(lower, upper),
        method="trf",
        ftol=1.0e-14,
        xtol=1.0e-14,
        gtol=1.0e-14,
        x_scale="jac",
        max_nfev=config.max_iterations,
    )
    solution = np.asarray(solve.x)

    boundary, state, potential, mass_scale = unpack(jnp.asarray(solution))
    (
        plasma,
        plasma_b_squared_full,
        perpendicular_pressure,
        _,
        anisotropy_valid,
        vacuum_geometry,
        vacuum_field,
        _,
    ) = components(jnp.asarray(solution))
    plasma_b_squared = plasma_b_squared_full[-1]
    if use_exterior:
        if plasma_grid.ntheta == 1:
            vacuum_b_squared = jnp.sum(vacuum_field.lateral_field_xyz**2, axis=-1)[None, :]
            vacuum_b_normal = vacuum_field.lateral_b_normal[None, :]
        else:
            vacuum_b_squared = jnp.sum(vacuum_field.lateral_field_xyz**2, axis=-1)
            vacuum_b_normal = vacuum_field.lateral_b_normal
        compatibility_limit = 1.0e-6 if plasma_grid.ntheta == 1 else 2.0e-3
        vacuum_valid = (
            vacuum_field.neumann_result.compatibility_error <= compatibility_limit
        ) & (vacuum_field.neumann_result.condition_number <= 1.0e8)
    else:
        vacuum_b_squared = jnp.sum(vacuum_field.total_xyz[0] ** 2, axis=-1)
        vacuum_b_normal = vacuum_field.b_normal_inner
        vacuum_valid = vacuum_geometry.valid
    active_axial_weights = (
        jnp.asarray(plasma_grid.axial_basis.weights).at[jnp.asarray([0, plasma_grid.nxi - 1])].set(0.0)
    )
    interface = interface_residual(
        perpendicular_pressure=perpendicular_pressure[-1],
        plasma_b_squared=plasma_b_squared,
        vacuum_b_squared=vacuum_b_squared,
        plasma_b_normal=jnp.zeros_like(plasma_b_squared),
        vacuum_b_normal=vacuum_b_normal,
        theta_weights=jnp.asarray(plasma_grid.theta_basis.weights),
        axial_weights=active_axial_weights,
    )
    final_residual = np.asarray(residual_jit(jnp.asarray(solution)), dtype=float)
    variational_max = float(np.max(np.abs(final_residual)))
    converged = bool(
        variational_max <= config.ftol
        and not bool(plasma.geometry.jacobian_sign_changed)
        and bool(vacuum_valid)
        and bool(anisotropy_valid)
    )
    message = str(solve.message)
    if not converged:
        message += f"; variational force={variational_max:.3e}"
    result = FreeBoundaryMirrorResult(
        boundary=boundary,
        plasma_state=state,
        plasma_energy=plasma,
        plasma_b_squared=plasma_b_squared_full,
        perpendicular_pressure=perpendicular_pressure,
        vacuum_geometry=vacuum_geometry,
        vacuum_field=vacuum_field,
        vacuum_potential=potential,
        mass_scale=mass_scale,
        interface=interface,
        history=jnp.asarray(history),
        variational_max=jnp.asarray(variational_max),
        iterations=int(solve.nfev),
        converged=converged,
        optimizer_success=bool(solve.success),
        message=message,
    )
    if require_convergence and not converged:
        raise RuntimeError(message)
    return result


solve_axisymmetric_free_boundary_cli = solve_free_boundary_cli


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .basis import MirrorGrid
    from .exterior import ClosedMirrorSurface
