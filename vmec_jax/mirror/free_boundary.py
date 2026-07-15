"""Coupled plasma-boundary-vacuum solves for straight-axis mirrors."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

from .forces import (
    MU0,
    InterfaceResidual,
    IsotropicForceResidual,
    MirrorEnergy,
    VariationalResidual,
    interface_residual,
    isotropic_force_residual,
    isotropic_staggered_fixed_boundary_gradient,
    isotropic_staggered_weak_residual,
    mass_profile_from_pressure,
    mirror_energy,
)
from .geometry import normalized_divergence_rms
from .exterior_bie import (
    AxisymmetricExteriorVacuum,
    NonaxisymmetricExteriorVacuum,
    solve_axisymmetric_exterior_vacuum,
    solve_nonaxisymmetric_exterior_vacuum,
)
from .model import MirrorBoundary, MirrorConfig, MirrorState, project_fixed_boundary_state
from .output import FreeBoundaryRestart
from .solver import _MirrorStateVectorizer
from .splines import (
    SplineMirrorBoundary,
    SplineMirrorDiscretization,
    SplineMirrorState,
    _SplineStateVectorizer,
)

Array = Any
_MONOLITHIC_JACOBIAN_MAX_SIZE = 80


@dataclass(frozen=True, eq=False)
class _SplineFreeBoundaryVectorizer:
    """Compose free boundary coefficients with the shared spline state map."""

    base_boundary: SplineMirrorBoundary
    state_vectorizer: _SplineStateVectorizer
    discretization: SplineMirrorDiscretization
    boundary_indices: tuple[np.ndarray, np.ndarray]
    boundary_scale: float
    calibrate_pressure: bool
    initial_mass_scale: float

    @classmethod
    def build(
        cls,
        boundary: SplineMirrorBoundary,
        state: SplineMirrorState,
        discretization: SplineMirrorDiscretization,
        *,
        axial_flux_derivative: Array,
        solve_lambda: bool,
        calibrate_pressure: bool = False,
        initial_mass_scale: float = 1.0,
    ) -> "_SplineFreeBoundaryVectorizer":
        """Build the square free-equilibrium coefficient layout."""

        if discretization.closed:
            raise ValueError("open free boundary requires a clamped spline discretization")
        coefficients = np.asarray(boundary.radius_coefficients, dtype=float)
        expected = (discretization.grid.ntheta, discretization.coefficient_count)
        if coefficients.shape != expected:
            raise ValueError(f"boundary coefficient shape {coefficients.shape} must be {expected}")
        evaluated = np.asarray(discretization.evaluate_boundary(boundary).radius_scale)
        boundary_scale = float(np.mean(evaluated))
        if not np.isfinite(boundary_scale) or boundary_scale <= 0.0:
            raise ValueError("mean boundary radius must be positive and finite")
        initial_mass_scale = float(initial_mass_scale)
        if not np.isfinite(initial_mass_scale) or initial_mass_scale <= 0.0:
            raise ValueError("initial_mass_scale must be positive and finite")

        mask = np.zeros(expected, dtype=bool)
        mask[:, 1:-1] = True
        state_vectorizer = _SplineStateVectorizer.build(
            state,
            boundary,
            discretization,
            axial_flux_derivative=axial_flux_derivative,
            solve_lambda=solve_lambda,
        )
        return cls(
            base_boundary=boundary,
            state_vectorizer=state_vectorizer,
            discretization=discretization,
            boundary_indices=tuple(np.asarray(index) for index in np.nonzero(mask)),
            boundary_scale=boundary_scale,
            calibrate_pressure=bool(calibrate_pressure),
            initial_mass_scale=initial_mass_scale,
        )

    @property
    def boundary_size(self) -> int:
        """Number of free lateral-boundary coefficients."""

        return int(self.boundary_indices[0].size)

    @property
    def state_size(self) -> int:
        """Number of active interior geometry and stream coefficients."""

        return self.state_vectorizer.radius_size + self.state_vectorizer.lambda_size

    @property
    def size(self) -> int:
        """Total nonlinear unknown count, including optional pressure scale."""

        return self.boundary_size + self.state_size + int(self.calibrate_pressure)

    def pack(self) -> np.ndarray:
        """Pack boundary, plasma, and optional mass scale."""

        boundary = np.asarray(self.base_boundary.radius_coefficients)[self.boundary_indices]
        parts = [boundary / self.boundary_scale, self.state_vectorizer.pack()]
        if self.calibrate_pressure:
            parts.append(np.asarray([self.initial_mass_scale]))
        return np.concatenate(parts)

    def unpack(self, vector: Array) -> tuple[SplineMirrorBoundary, SplineMirrorState, Array]:
        """Reconstruct a constrained coefficient boundary and plasma state."""

        vector = jnp.asarray(vector)
        if vector.ndim != 1 or vector.shape[0] != self.size:
            raise ValueError(f"free-boundary vector shape {vector.shape} must be ({self.size},)")
        indices = tuple(jnp.asarray(index) for index in self.boundary_indices)
        boundary_coefficients = jnp.asarray(self.base_boundary.radius_coefficients).at[indices].set(
            vector[: self.boundary_size] * self.boundary_scale
        )
        boundary = SplineMirrorBoundary(boundary_coefficients)
        start, stop = self.boundary_size, self.boundary_size + self.state_size
        state = self.state_vectorizer.unpack(vector[start:stop])
        state = self.discretization.project_fixed_boundary(state, boundary)
        mass_scale = vector[-1] if self.calibrate_pressure else jnp.asarray(1.0, dtype=vector.dtype)
        return boundary, state, mass_scale

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return normalized positivity and stream-function bounds."""

        state_lower, state_upper = self.state_vectorizer.bounds()
        lower = [np.full(self.boundary_size, 0.2), state_lower]
        upper = [np.full(self.boundary_size, np.inf), state_upper]
        if self.calibrate_pressure:
            lower.append(np.asarray([np.finfo(float).tiny]))
            upper.append(np.asarray([np.inf]))
        return np.concatenate(lower), np.concatenate(upper)

    def boundary_work_residual(
        self,
        boundary: SplineMirrorBoundary,
        jump: Array,
        stress_scale: Array,
    ) -> Array:
        """Pull normalized lateral virtual work to free boundary coefficients."""

        normalized = _spline_boundary_work_residual(
            boundary,
            jump,
            stress_scale,
            self.discretization,
        )
        return normalized[self.boundary_indices]


def _spline_boundary_work(
    boundary: SplineMirrorBoundary,
    jump: Array,
    discretization: SplineMirrorDiscretization,
) -> Array:
    """Return physical interface work for every boundary coefficient.

    A radial coefficient variation moves the side wall along ``e_r``. For
    ``x=(a cos(theta), a sin(theta), z)``, its virtual-work measure is
    ``e_r dot (n dA) = a * dz/dxi * dtheta * dxi``. The endpoint coefficients
    are returned as well; the free map removes them from the solve.
    """

    evaluated = discretization.evaluate_boundary(boundary).radius_scale
    jump = jnp.asarray(jump)
    expected = (discretization.grid.ntheta, discretization.grid.nxi)
    if jump.shape != expected:
        raise ValueError(f"interface jump shape {jump.shape} must be {expected}")
    weights = (
        jnp.asarray(discretization.grid.theta_basis.weights)[:, None]
        * jnp.asarray(discretization.grid.axial_basis.weights)[None, :]
    )
    radial_measure = evaluated * abs(float(discretization.grid.dz_dxi)) * weights
    return jnp.einsum(
        "jk,kc->jc",
        jump * radial_measure,
        jnp.asarray(discretization.evaluation_matrix),
    )


def _spline_boundary_work_residual(
    boundary: SplineMirrorBoundary,
    jump: Array,
    stress_scale: Array,
    discretization: SplineMirrorDiscretization,
) -> Array:
    """Normalize coefficient boundary work without changing its zeros."""

    jump = jnp.asarray(jump)
    stress_scale = jnp.broadcast_to(jnp.asarray(stress_scale, dtype=jump.dtype), jump.shape)
    physical = _spline_boundary_work(boundary, jump, discretization)
    scale = _spline_boundary_work(boundary, jnp.abs(stress_scale), discretization)
    tiny = jnp.finfo(physical.dtype).tiny
    return physical / jnp.maximum(scale, tiny)


@dataclass(frozen=True)
class FreeBoundaryMirrorResult:
    """Joint plasma-boundary-vacuum equilibrium result."""

    boundary: MirrorBoundary
    plasma_state: MirrorState
    plasma_energy: MirrorEnergy
    plasma_force: IsotropicForceResidual
    plasma_staggered_weak_force: VariationalResidual | None
    normalized_divergence_rms: Array
    plasma_b_squared: Array
    pressure: Array
    vacuum_geometry: "ClosedMirrorSurface"
    vacuum_field: AxisymmetricExteriorVacuum | NonaxisymmetricExteriorVacuum
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
    config: MirrorConfig,
    external_field: Any,
    *,
    axial_flux_derivative: Array,
    mass_profile: Array = 0.0,
    current_derivative: Array = 0.0,
    solve_lambda: bool | None = None,
    gamma: float = 5.0 / 3.0,
    initial_state: MirrorState | None = None,
    target_central_pressure: float | None = None,
    initial_mass_scale: float = 1.0,
    exterior_ntheta: int = 40,
    exterior_order: int = 8,
    exterior_spectral_side_density: bool = False,
    exterior_jacobian_chunk_size: int = 6,
    require_convergence: bool = False,
) -> FreeBoundaryMirrorResult:
    """Jointly solve a mirror plasma boundary and its free-space vacuum.

    The vacuum field is the unbounded closed-surface Neumann solution and adds
    no vacuum degrees of freedom to the nonlinear vector. Axisymmetric solves
    keep the stream function fixed by default. Nonaxisymmetric solves include
    its gauge-free coefficients unless ``solve_lambda=False`` is requested.
    """

    if solve_lambda is None:
        solve_lambda = plasma_grid.ntheta != 1
    exterior_jacobian_chunk_size = int(exterior_jacobian_chunk_size)
    if exterior_jacobian_chunk_size < 1:
        raise ValueError("exterior_jacobian_chunk_size must be positive")
    if target_central_pressure is not None and target_central_pressure <= 0.0:
        raise ValueError("target_central_pressure must be positive")
    if initial_mass_scale <= 0.0:
        raise ValueError("initial_mass_scale must be positive")
    initial_boundary_radius = np.asarray(initial_boundary.radius_scale, dtype=float)
    if initial_boundary_radius.shape != (plasma_grid.ntheta, plasma_grid.nxi):
        raise ValueError("initial boundary does not match the plasma grid")
    boundary_scale = float(np.mean(initial_boundary_radius))
    boundary_mask = np.zeros(initial_boundary_radius.shape, dtype=bool)
    boundary_mask[:, 1:-1] = True
    boundary_indices = tuple(np.asarray(index) for index in np.nonzero(boundary_mask))
    plasma_mask = np.zeros(plasma_grid.shape, dtype=bool)
    plasma_mask[1:-1, :, 1:-1] = True
    plasma_indices = tuple(np.asarray(index) for index in np.nonzero(plasma_mask))
    nb = boundary_indices[0].size

    base_state = MirrorState.from_boundary(initial_boundary, plasma_grid) if initial_state is None else initial_state
    base_state.validate_shape(plasma_grid)
    if not np.allclose(np.asarray(base_state.radius_scale[-1]), initial_boundary_radius):
        raise ValueError("initial_state boundary must match initial_boundary")
    plasma_vectorizer = _MirrorStateVectorizer.build(
        base_state,
        initial_boundary,
        plasma_grid,
        axial_flux_derivative=axial_flux_derivative,
        solve_lambda=solve_lambda,
    )
    if plasma_vectorizer.radius_size != plasma_indices[0].size:
        raise ValueError("free-boundary plasma packing does not match the interior radius")
    np_state = plasma_vectorizer.size
    calibrate_pressure = target_central_pressure is not None
    mass_scale_index = nb + np_state
    x0_parts = [
        initial_boundary_radius[boundary_indices] / boundary_scale,
        plasma_vectorizer.pack(),
    ]
    if calibrate_pressure:
        x0_parts.append(np.asarray([initial_mass_scale]))
    x0 = np.concatenate(x0_parts)
    geometric_upper = np.inf
    plasma_lower, plasma_upper = plasma_vectorizer.bounds()
    plasma_upper[: plasma_vectorizer.radius_size] = geometric_upper
    lower_parts = [np.full(nb, 0.2), plasma_lower]
    upper_parts = [np.full(nb, geometric_upper), plasma_upper]
    if calibrate_pressure:
        lower_parts.append(np.asarray([np.finfo(float).tiny]))
        upper_parts.append(np.asarray([np.inf]))
    lower, upper = np.concatenate(lower_parts), np.concatenate(upper_parts)

    def unpack(vector: Array) -> tuple[MirrorBoundary, MirrorState, Array]:
        vector = jnp.asarray(vector)
        boundary_radius = (
            jnp.asarray(initial_boundary_radius)
            .at[tuple(jnp.asarray(index) for index in boundary_indices)]
            .set(vector[:nb] * boundary_scale)
        )
        boundary = MirrorBoundary(boundary_radius)
        packed_state = plasma_vectorizer.unpack(vector[nb : nb + np_state])
        radius = packed_state.radius_scale
        radius = radius.at[-1].set(boundary_radius)
        radius = radius.at[:, :, 0].set(boundary_radius[:, 0])
        radius = radius.at[:, :, -1].set(boundary_radius[:, -1])
        radius = radius.at[0].set(radius[1])
        state = MirrorState(radius, packed_state.lambda_stream)
        if calibrate_pressure:
            mass_scale = vector[mass_scale_index]
        else:
            mass_scale = jnp.asarray(1.0, dtype=vector.dtype)
        return boundary, state, mass_scale

    def components(vector: Array):
        boundary, state, mass_scale = unpack(vector)
        plasma = mirror_energy(
            state,
            plasma_grid,
            axial_flux_derivative=axial_flux_derivative,
            mass_profile=jnp.asarray(mass_profile) * mass_scale,
            current_derivative=current_derivative,
            gamma=gamma,
        )
        plasma_b_squared = plasma.b_squared
        pressure = jnp.broadcast_to(plasma.pressure[:, None, None], plasma_b_squared.shape)
        central_pressure = plasma.pressure[0]
        if plasma_grid.ntheta == 1:
            vacuum_field = solve_axisymmetric_exterior_vacuum(
                boundary,
                plasma.field,
                plasma_grid,
                external_field,
                axisymmetric_ntheta=exterior_ntheta,
                order=exterior_order,
                spectral_side_density=exterior_spectral_side_density,
            )
        else:
            vacuum_field = solve_nonaxisymmetric_exterior_vacuum(
                boundary,
                plasma.field,
                plasma.geometry,
                plasma_grid,
                external_field,
                order=exterior_order,
                spectral_side_density=exterior_spectral_side_density,
            )
        return (
            plasma,
            plasma_b_squared,
            pressure,
            central_pressure,
            vacuum_field.surface,
            vacuum_field,
        )

    initial_components = components(jnp.asarray(x0))
    plasma_scale = max(abs(float(initial_components[0].total)), 1.0)

    def plasma_objective(vector: Array) -> Array:
        return components(vector)[0].total / plasma_scale

    def residual_function(vector: Array) -> Array:
        (
            plasma,
            plasma_b_squared,
            pressure,
            central_pressure,
            _,
            vacuum_field,
        ) = components(vector)
        plasma_gradient = jax.grad(plasma_objective)(vector)[nb : nb + np_state]
        plasma_b_squared = plasma_b_squared[-1, :, 1:-1].reshape(-1)
        vacuum_xyz = vacuum_field.lateral_field_xyz
        if plasma_grid.ntheta == 1:
            vacuum_xyz = vacuum_xyz[None]
        vacuum_xyz = vacuum_xyz[:, 1:-1].reshape(-1, 3)
        vacuum_b_squared = jnp.sum(vacuum_xyz**2, axis=-1)
        pressure = pressure[-1, :, 1:-1].reshape(-1)
        jump = pressure + plasma_b_squared / (2.0 * MU0) - vacuum_b_squared / (2.0 * MU0)
        stress_scale = jnp.abs(pressure) + plasma_b_squared / (2.0 * MU0) + vacuum_b_squared / (2.0 * MU0)
        stress = jump / jnp.maximum(stress_scale, jnp.finfo(stress_scale.dtype).tiny)
        residuals = [stress, plasma_gradient]
        if calibrate_pressure:
            target = float(target_central_pressure)
            residuals.append(jnp.asarray([(central_pressure - target) / target]))
        return jnp.concatenate(residuals)

    residual_jit = jax.jit(residual_function)
    jacobian_jit = jax.jit(jax.jacfwd(residual_function))
    jvp_batch_jit = jax.jit(
        jax.vmap(
            lambda primal, tangent: jax.jvp(residual_function, (primal,), (tangent,))[1],
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
                    0.0,
                    float(np.max(np.abs(residual))),
                )
            )
            last_recorded = np.array(vector, copy=True)
        return residual

    def jacobian_host(vector: np.ndarray) -> np.ndarray:
        if vector.size <= _MONOLITHIC_JACOBIAN_MAX_SIZE:
            return np.asarray(jacobian_jit(jnp.asarray(vector)), dtype=float)
        size = vector.size
        columns = []
        identity = np.eye(size)
        for start in range(0, size, exterior_jacobian_chunk_size):
            stop = min(start + exterior_jacobian_chunk_size, size)
            columns.append(
                np.asarray(
                    jvp_batch_jit(jnp.asarray(vector), jnp.asarray(identity[start:stop])),
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

    boundary, state, mass_scale = unpack(jnp.asarray(solution))
    (
        plasma,
        plasma_b_squared_full,
        pressure,
        _,
        vacuum_geometry,
        vacuum_field,
    ) = components(jnp.asarray(solution))
    plasma_b_squared = plasma_b_squared_full[-1]
    if plasma_grid.ntheta == 1:
        vacuum_b_squared = jnp.sum(vacuum_field.lateral_field_xyz**2, axis=-1)[None, :]
        vacuum_b_normal = vacuum_field.lateral_b_normal[None, :]
    else:
        vacuum_b_squared = jnp.sum(vacuum_field.lateral_field_xyz**2, axis=-1)
        vacuum_b_normal = vacuum_field.lateral_b_normal
    compatibility_limit = 1.0e-6 if plasma_grid.ntheta == 1 else 2.0e-3
    vacuum_valid = (vacuum_field.neumann_result.compatibility_error <= compatibility_limit) & (
        vacuum_field.neumann_result.condition_number <= 1.0e8
    )
    active_axial_weights = (
        jnp.asarray(plasma_grid.axial_basis.weights).at[jnp.asarray([0, plasma_grid.nxi - 1])].set(0.0)
    )
    interface = interface_residual(
        pressure=pressure[-1],
        plasma_b_squared=plasma_b_squared,
        vacuum_b_squared=vacuum_b_squared,
        plasma_b_normal=jnp.zeros_like(plasma_b_squared),
        vacuum_b_normal=vacuum_b_normal,
        theta_weights=jnp.asarray(plasma_grid.theta_basis.weights),
        axial_weights=active_axial_weights,
    )
    final_residual = np.asarray(residual_jit(jnp.asarray(solution)), dtype=float)
    variational_max = float(np.max(np.abs(final_residual)))
    energy_kwargs = {
        "axial_flux_derivative": axial_flux_derivative,
        "mass_profile": jnp.asarray(mass_profile) * mass_scale,
        "current_derivative": current_derivative,
        "gamma": gamma,
    }
    plasma_force = isotropic_force_residual(
        plasma,
        plasma_grid,
        state=state,
        **energy_kwargs,
    )
    full_weak_force = isotropic_staggered_weak_residual(
        state,
        boundary,
        plasma_grid,
        **energy_kwargs,
    )
    weak_gradient = isotropic_staggered_fixed_boundary_gradient(
        state,
        boundary,
        plasma_grid,
        **energy_kwargs,
    )
    active_weak = plasma_vectorizer.pullback_gradient(weak_gradient) / plasma_scale
    radius_weak = active_weak[: plasma_vectorizer.radius_size]
    lambda_weak = active_weak[plasma_vectorizer.radius_size :]
    plasma_staggered_weak_force = VariationalResidual(
        radius_gradient=full_weak_force.radius_gradient,
        lambda_gradient=full_weak_force.lambda_gradient,
        radius_rms=jnp.asarray(np.sqrt(np.mean(radius_weak**2))),
        lambda_rms=jnp.asarray(np.sqrt(np.mean(lambda_weak**2)) if lambda_weak.size else 0.0),
        maximum=jnp.asarray(np.max(np.abs(active_weak))),
    )
    divergence_rms = normalized_divergence_rms(plasma.field, plasma.geometry, plasma_grid)
    converged = bool(
        variational_max <= config.ftol and not bool(plasma.geometry.jacobian_sign_changed) and bool(vacuum_valid)
    )
    message = str(solve.message)
    if not converged:
        message += (
            f"; variational force={variational_max:.3e}"
            f"; crossed surfaces={bool(plasma.geometry.jacobian_sign_changed)}"
            f"; exterior compatibility="
            f"{float(vacuum_field.neumann_result.compatibility_error):.3e}"
            f"; raw compatibility="
            f"{float(vacuum_field.neumann_result.raw_compatibility_error):.3e}"
            f"; exterior condition="
            f"{float(vacuum_field.neumann_result.condition_number):.3e}"
        )
    result = FreeBoundaryMirrorResult(
        boundary=boundary,
        plasma_state=state,
        plasma_energy=plasma,
        plasma_force=plasma_force,
        plasma_staggered_weak_force=plasma_staggered_weak_force,
        normalized_divergence_rms=divergence_rms,
        plasma_b_squared=plasma_b_squared_full,
        pressure=pressure,
        vacuum_geometry=vacuum_geometry,
        vacuum_field=vacuum_field,
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


def interpolate_fixed_boundary_state(
    state: MirrorState,
    source_grid: "MirrorGrid",
    boundary: MirrorBoundary,
    target_grid: "MirrorGrid",
) -> MirrorState:
    """Interpolate a solved state to a new radial and axial grid."""

    state.validate_shape(source_grid)
    expected = (target_grid.ntheta, target_grid.nxi)
    if tuple(jnp.shape(boundary.radius_scale)) != expected:
        raise ValueError(f"boundary shape {jnp.shape(boundary.radius_scale)} must be {expected}")
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
    config: MirrorConfig,
    external_field: Any,
    beta_values: Array,
    *,
    axial_flux_derivative: Array,
    reference_field: float,
    current_derivative: Array = 0.0,
    gamma: float = 5.0 / 3.0,
    beta_rtol: float = 1.0e-8,
    initial_restart: FreeBoundaryRestart | None = None,
    exterior_ntheta: int = 40,
    exterior_order: int = 8,
    exterior_spectral_side_density: bool = False,
    exterior_jacobian_chunk_size: int = 6,
) -> tuple[FreeBoundaryMirrorResult, ...]:
    """Solve an increasing, fully hot-started free-boundary beta scan."""

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
    mass_scale = 1.0 if initial_restart is None else initial_restart.mass_scale
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
            config,
            external_field,
            axial_flux_derivative=axial_flux_derivative,
            current_derivative=current_derivative,
            mass_profile=mass,
            gamma=gamma,
            initial_state=state,
            initial_mass_scale=mass_scale,
            exterior_ntheta=exterior_ntheta,
            exterior_order=exterior_order,
            exterior_spectral_side_density=exterior_spectral_side_density,
            exterior_jacobian_chunk_size=exterior_jacobian_chunk_size,
            target_central_pressure=None if beta == 0.0 else central_pressure,
            require_convergence=True,
        )
        if beta > 0.0:
            center = int(np.argmin(np.abs(plasma_grid.z)))
            achieved_beta = 2.0 * MU0 * float(result.pressure[0, 0, center]) / float(reference_field) ** 2
            if abs(achieved_beta - float(beta)) / float(beta) > beta_rtol:
                raise RuntimeError(f"central beta did not reach rtol={beta_rtol:.3e}")
        results.append(result)
        boundary = result.boundary
        state = result.plasma_state
        mass_scale = float(result.mass_scale)
    return tuple(results)


solve_axisymmetric_free_boundary_cli = solve_free_boundary_cli
solve_axisymmetric_beta_scan_cli = solve_beta_scan_cli


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .basis import MirrorGrid
    from .exterior import ClosedMirrorSurface
