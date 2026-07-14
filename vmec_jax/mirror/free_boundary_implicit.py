"""Implicit adjoints for converged axisymmetric free-boundary mirrors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from .exterior_bie import (
    AxisymmetricExteriorVacuum,
    solve_axisymmetric_exterior_vacuum,
)
from .forces import (
    AnisotropicMirrorEnergy,
    MirrorEnergy,
    anisotropic_mirror_energy,
    mirror_energy,
)
from .geometry import magnetic_field_squared
from .model import MirrorBoundary, MirrorState
from .solver import _MirrorStateVectorizer, _packed_preconditioner
from .vacuum import MU0

Array = Any
FreeBoundaryQuantity = Callable[
    [MirrorBoundary, MirrorState, MirrorEnergy | AnisotropicMirrorEnergy, Any],
    Array,
]


@dataclass(frozen=True)
class FreeBoundaryParameters:
    """Differentiable controls for an exterior free-boundary equilibrium."""

    external_field: Any
    axial_flux_derivative: Array
    mass_profile: Array
    current_derivative: Array
    pressure_closure: Any


@dataclass(frozen=True, eq=False)
class FreeBoundaryAdjointConfig:
    """Static exterior quadrature and linear-solver controls."""

    axisymmetric_ntheta: int = 40
    exterior_order: int = 8
    spectral_side_density: bool = False
    gamma: float = 5.0 / 3.0
    rtol: float = 1.0e-9
    max_restarts: int = 30


@dataclass(frozen=True)
class FreeBoundaryAdjointResult:
    """Scalar value, total control gradient, and transpose-solve diagnostics."""

    value: Array
    gradient: FreeBoundaryParameters
    iterations: int
    relative_residual: float
    converged: bool


jax.tree_util.register_dataclass(
    FreeBoundaryParameters,
    data_fields=[
        "external_field",
        "axial_flux_derivative",
        "mass_profile",
        "current_derivative",
        "pressure_closure",
    ],
    meta_fields=[],
)
jax.tree_util.register_dataclass(
    FreeBoundaryAdjointResult,
    data_fields=["value", "gradient"],
    meta_fields=["iterations", "relative_residual", "converged"],
)


def free_boundary_parameters(
    external_field: Any,
    *,
    axial_flux_derivative: Array,
    mass_profile: Array = 0.0,
    current_derivative: Array = 0.0,
    pressure_closure: Any = None,
) -> FreeBoundaryParameters:
    """Collect free-boundary controls in a differentiable pytree."""

    return FreeBoundaryParameters(
        external_field=external_field,
        axial_flux_derivative=jnp.asarray(axial_flux_derivative),
        mass_profile=jnp.asarray(mass_profile),
        current_derivative=jnp.asarray(current_derivative),
        pressure_closure=pressure_closure,
    )


def free_boundary_adjoint(
    result: Any,
    parameters: FreeBoundaryParameters,
    plasma_grid: Any,
    quantity: FreeBoundaryQuantity,
    *,
    config: FreeBoundaryAdjointConfig = FreeBoundaryAdjointConfig(),
) -> FreeBoundaryAdjointResult:
    """Differentiate a scalar through a converged axisymmetric exterior solve.

    End-cut radii and the pressure profile are held fixed. The active
    equilibrium variables are the lateral boundary and plasma-interior radii.
    The exterior Neumann solve eliminates vacuum degrees of freedom, so its
    field-parameter and shape derivatives enter the interface-stress rows directly.
    """

    if not bool(result.converged):
        raise ValueError("free-boundary differentiation requires a converged result")
    if plasma_grid.ntheta != 1:
        raise ValueError("free-boundary adjoint currently supports axisymmetry only")
    if not isinstance(result.vacuum_field, AxisymmetricExteriorVacuum):
        raise ValueError("free-boundary adjoint requires the exterior vacuum backend")
    if config.rtol <= 0.0 or config.max_restarts < 1:
        raise ValueError("adjoint tolerances and iteration limits must be positive")
    anisotropic = parameters.pressure_closure is not None
    if anisotropic != isinstance(result.plasma_energy, AnisotropicMirrorEnergy):
        raise ValueError("pressure_closure must match the converged energy model")

    fixed_boundary = jnp.asarray(result.boundary.radius_scale)
    base_state = result.plasma_state
    boundary_indices = (
        np.zeros(plasma_grid.nxi - 2, dtype=int),
        np.arange(1, plasma_grid.nxi - 1),
    )
    plasma_mask = np.zeros(plasma_grid.shape, dtype=bool)
    plasma_mask[1:-1, :, 1:-1] = True
    plasma_indices = tuple(np.asarray(index) for index in np.nonzero(plasma_mask))
    boundary_size = plasma_grid.nxi - 2
    plasma_size = plasma_indices[0].size
    x_star = jnp.concatenate(
        [
            fixed_boundary[boundary_indices],
            jnp.asarray(base_state.radius_scale)[plasma_indices],
        ]
    )
    energy_scale = max(abs(float(result.plasma_energy.total)), 1.0)

    def unpack(vector: Array) -> tuple[MirrorBoundary, MirrorState]:
        boundary_radius = fixed_boundary.at[boundary_indices].set(
            vector[:boundary_size]
        )
        boundary = MirrorBoundary(boundary_radius)
        radius = base_state.radius_scale.at[plasma_indices].set(
            vector[boundary_size:]
        )
        radius = radius.at[-1].set(boundary_radius)
        radius = radius.at[:, :, 0].set(boundary_radius[:, 0])
        radius = radius.at[:, :, -1].set(boundary_radius[:, -1])
        radius = radius.at[0].set(radius[1])
        return boundary, MirrorState(radius, base_state.lambda_stream)

    def plasma_components(vector: Array, controls: FreeBoundaryParameters):
        boundary, state = unpack(vector)
        if controls.pressure_closure is None:
            plasma = mirror_energy(
                state,
                plasma_grid,
                axial_flux_derivative=controls.axial_flux_derivative,
                mass_profile=controls.mass_profile,
                current_derivative=controls.current_derivative,
                gamma=config.gamma,
            )
            pressure = jnp.broadcast_to(
                plasma.pressure[:, None, None], plasma.b_squared.shape
            )
            plasma_b_squared = magnetic_field_squared(plasma.field, plasma.geometry)
        else:
            plasma = anisotropic_mirror_energy(
                state,
                plasma_grid,
                controls.pressure_closure,
                axial_flux_derivative=controls.axial_flux_derivative,
                current_derivative=controls.current_derivative,
            )
            plasma_b_squared = magnetic_field_squared(plasma.field, plasma.geometry)
            pressure = controls.pressure_closure.moments(
                jnp.asarray(plasma_grid.s)[:, None, None],
                jnp.sqrt(jnp.maximum(plasma_b_squared, 0.0)),
            ).perpendicular
        return boundary, state, plasma, pressure, plasma_b_squared

    def components(vector: Array, controls: FreeBoundaryParameters):
        boundary, state, plasma, pressure, plasma_b_squared = plasma_components(
            vector, controls
        )
        vacuum = solve_axisymmetric_exterior_vacuum(
            boundary,
            plasma.field,
            plasma_grid,
            controls.external_field,
            axisymmetric_ntheta=config.axisymmetric_ntheta,
            order=config.exterior_order,
            spectral_side_density=config.spectral_side_density,
        )
        return boundary, state, plasma, pressure, plasma_b_squared, vacuum

    def normalized_plasma_energy(
        vector: Array, controls: FreeBoundaryParameters
    ) -> Array:
        return plasma_components(vector, controls)[2].total / energy_scale

    def residual(vector: Array, controls: FreeBoundaryParameters) -> Array:
        _, _, _, pressure, plasma_b_squared, vacuum = components(vector, controls)
        plasma_gradient = jax.grad(normalized_plasma_energy, argnums=0)(
            vector, controls
        )[boundary_size:]
        plasma_side = plasma_b_squared[-1, 0, 1:-1]
        vacuum_side = jnp.sum(vacuum.lateral_field_xyz[1:-1] ** 2, axis=-1)
        pressure_side = pressure[-1, 0, 1:-1]
        jump = pressure_side + (plasma_side - vacuum_side) / (2.0 * MU0)
        stress_scale = (
            jnp.abs(pressure_side)
            + plasma_side / (2.0 * MU0)
            + vacuum_side / (2.0 * MU0)
        )
        stress = jump / jnp.maximum(
            stress_scale, jnp.finfo(stress_scale.dtype).tiny
        )
        return jnp.concatenate([stress, plasma_gradient])

    def evaluate_quantity(vector: Array, controls: FreeBoundaryParameters) -> Array:
        boundary, state, plasma, _, _, vacuum = components(vector, controls)
        value = jnp.asarray(quantity(boundary, state, plasma, vacuum))
        if value.ndim != 0:
            raise ValueError("free-boundary adjoint quantity must return a scalar")
        return value

    equilibrium_residual = np.asarray(residual(x_star, parameters), dtype=float)
    if np.max(np.abs(equilibrium_residual)) > 10.0 * max(
        float(result.variational_max), 1.0e-12
    ):
        raise ValueError("reconstructed free-boundary residual does not match result")
    value, (quantity_x, quantity_parameters) = jax.value_and_grad(
        evaluate_quantity, argnums=(0, 1)
    )(x_star, parameters)
    _, transpose = jax.vjp(lambda vector: residual(vector, parameters), x_star)
    transpose_action = jax.jit(lambda vector: transpose(vector)[0])

    def matrix_vector(vector: np.ndarray) -> np.ndarray:
        return np.asarray(transpose_action(jnp.asarray(vector)), dtype=float)

    vectorizer = _MirrorStateVectorizer.build(
        base_state,
        result.boundary,
        plasma_grid,
        axial_flux_derivative=parameters.axial_flux_derivative,
        solve_lambda=False,
    )
    if vectorizer.radius_size != plasma_size:
        raise ValueError("free-boundary plasma packing does not match primal packing")
    plasma_preconditioner, plasma_scales = _packed_preconditioner(
        plasma_grid, vectorizer
    )
    boundary_scale = np.ones(1)

    def apply_preconditioner(vector: np.ndarray) -> np.ndarray:
        output = np.array(vector, dtype=float, copy=True)
        output[:boundary_size] *= boundary_scale[0]
        output[boundary_size:] = plasma_preconditioner(vector[boundary_size:])
        return output

    probe = np.random.default_rng(0).choice((-1.0, 1.0), size=x_star.size)
    for block, active in enumerate(
        (slice(0, boundary_size), slice(boundary_size, None))
    ):
        direction = np.zeros_like(probe)
        direction[active] = probe[active]
        response = apply_preconditioner(matrix_vector(direction))
        denominator = abs(float(np.dot(direction, response)))
        if denominator > np.finfo(float).tiny:
            scale = np.clip(
                np.dot(direction, direction) / denominator, 1.0e-8, 1.0e8
            )
            if block == 0:
                boundary_scale[0] = scale
            else:
                plasma_scales[0] = scale

    operator = LinearOperator(
        (x_star.size, x_star.size), matvec=matrix_vector, dtype=float
    )
    inverse = LinearOperator(
        (x_star.size, x_star.size), matvec=apply_preconditioner, dtype=float
    )
    iterations = 0

    def count_iteration(_residual: float) -> None:
        nonlocal iterations
        iterations += 1

    right_hand_side = np.asarray(quantity_x, dtype=float)
    adjoint, info = gmres(
        operator,
        right_hand_side,
        M=inverse,
        restart=min(50, x_star.size),
        maxiter=config.max_restarts,
        rtol=config.rtol,
        atol=0.0,
        callback=count_iteration,
        callback_type="pr_norm",
    )
    linear_error = matrix_vector(adjoint) - right_hand_side
    relative_residual = float(
        np.linalg.norm(linear_error)
        / max(np.linalg.norm(right_hand_side), np.finfo(float).tiny)
    )
    _, parameter_pullback = jax.vjp(
        lambda controls: residual(x_star, controls), parameters
    )
    residual_parameter_gradient = parameter_pullback(jnp.asarray(adjoint))[0]
    total_gradient = jax.tree.map(
        lambda direct, implicit: direct - implicit,
        quantity_parameters,
        residual_parameter_gradient,
    )
    return FreeBoundaryAdjointResult(
        value=value,
        gradient=total_gradient,
        iterations=iterations,
        relative_residual=relative_residual,
        converged=bool(
            info == 0 and relative_residual <= max(10.0 * config.rtol, 1.0e-12)
        ),
    )


__all__ = [
    "FreeBoundaryAdjointConfig",
    "FreeBoundaryAdjointResult",
    "FreeBoundaryParameters",
    "free_boundary_adjoint",
    "free_boundary_parameters",
]
