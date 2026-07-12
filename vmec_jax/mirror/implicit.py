"""Implicit adjoints for converged fixed-boundary mirror equilibria.

The derivative is taken through the equilibrium equation, not through the
host-controlled optimization history. This gives memory use independent of
the number of nonlinear iterations and one adjoint solve per scalar quantity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from .forces import MirrorEnergy, mirror_energy
from .model import MirrorBoundary, MirrorState, project_fixed_boundary_state
from .solver import SeparableMirrorPreconditioner, _MirrorStateVectorizer

Array = Any
MirrorQuantity = Callable[[MirrorState, MirrorEnergy], Array]


@dataclass(frozen=True)
class FixedBoundaryParameters:
    """Differentiable physical inputs to an isotropic mirror equilibrium."""

    boundary_radius: Array
    axial_flux_derivative: Array
    mass_profile: Array
    current_derivative: Array


@dataclass(frozen=True)
class MirrorAdjointResult:
    """Scalar value, total parameter gradient, and linear-solve diagnostics."""

    value: Array
    gradient: FixedBoundaryParameters
    iterations: int
    relative_residual: float
    converged: bool


jax.tree_util.register_dataclass(
    FixedBoundaryParameters,
    data_fields=[
        "boundary_radius",
        "axial_flux_derivative",
        "mass_profile",
        "current_derivative",
    ],
    meta_fields=[],
)
jax.tree_util.register_dataclass(
    MirrorAdjointResult,
    data_fields=["value", "gradient"],
    meta_fields=["iterations", "relative_residual", "converged"],
)


def fixed_boundary_parameters(
    boundary: MirrorBoundary,
    *,
    axial_flux_derivative: Array,
    mass_profile: Array = 0.0,
    current_derivative: Array = 0.0,
) -> FixedBoundaryParameters:
    """Collect fixed-boundary controls in a differentiable pytree."""

    return FixedBoundaryParameters(
        boundary_radius=jnp.asarray(boundary.radius_scale),
        axial_flux_derivative=jnp.asarray(axial_flux_derivative),
        mass_profile=jnp.asarray(mass_profile),
        current_derivative=jnp.asarray(current_derivative),
    )


def _packed_preconditioner(grid, vectorizer):
    geometry = SeparableMirrorPreconditioner.build(grid)
    stream = None
    if vectorizer.lambda_size:
        stream = SeparableMirrorPreconditioner.build(grid, radial_nodes=grid.ns - 1)
    scales = np.ones(2)

    def apply(vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=float)
        result = np.array(vector, copy=True)
        result[: vectorizer.radius_size] = geometry.apply(
            vector[: vectorizer.radius_size]
        ) * scales[0]
        if stream is not None:
            result[vectorizer.radius_size :] = stream.apply_gauge_free(
                vector[vectorizer.radius_size :],
                free_indices=vectorizer.lambda_free_indices,
                pivot=vectorizer.lambda_pivot,
                weights=vectorizer.lambda_interior_weights,
            ) * scales[1]
        return result

    return apply, scales


def fixed_boundary_adjoint(
    result: Any,
    parameters: FixedBoundaryParameters,
    grid: Any,
    quantity: MirrorQuantity,
    *,
    gamma: float = 5.0 / 3.0,
    solve_lambda: bool = False,
    rtol: float = 1.0e-10,
    max_restarts: int = 20,
) -> MirrorAdjointResult:
    """Differentiate one scalar quantity through a converged equilibrium.

    The packed equilibrium equation is the gradient of the normalized MHD
    energy with all fixed side/end geometry and the stream-function gauge
    already eliminated. The transpose system is solved matrix-free with exact
    JAX products and the primal separable preconditioner.
    """

    if not bool(result.converged):
        raise ValueError("implicit differentiation requires a converged mirror result")
    if rtol <= 0.0 or max_restarts < 1:
        raise ValueError("rtol and max_restarts must be positive")
    boundary = MirrorBoundary(jnp.asarray(parameters.boundary_radius))
    vectorizer = _MirrorStateVectorizer.build(
        result.state,
        boundary,
        grid,
        axial_flux_derivative=parameters.axial_flux_derivative,
        solve_lambda=solve_lambda,
    )
    x_star = jnp.asarray(vectorizer.pack())
    energy_scale = max(abs(float(result.energy.total)), np.finfo(float).tiny)

    def state_at(x: Array, controls: FixedBoundaryParameters) -> MirrorState:
        return project_fixed_boundary_state(
            vectorizer.unpack(x), MirrorBoundary(controls.boundary_radius), grid
        )

    def energy_at(x: Array, controls: FixedBoundaryParameters) -> MirrorEnergy:
        return mirror_energy(
            state_at(x, controls),
            grid,
            axial_flux_derivative=controls.axial_flux_derivative,
            mass_profile=controls.mass_profile,
            current_derivative=controls.current_derivative,
            gamma=gamma,
        )

    def normalized_energy(x: Array, controls: FixedBoundaryParameters) -> Array:
        return energy_at(x, controls).total / energy_scale

    def residual(x: Array, controls: FixedBoundaryParameters) -> Array:
        return jax.grad(normalized_energy, argnums=0)(x, controls)

    def evaluate_quantity(x: Array, controls: FixedBoundaryParameters) -> Array:
        state = state_at(x, controls)
        value = jnp.asarray(quantity(state, energy_at(x, controls)))
        if value.ndim != 0:
            raise ValueError("mirror adjoint quantity must return a scalar")
        return value

    value, (quantity_x, quantity_parameters) = jax.value_and_grad(
        evaluate_quantity, argnums=(0, 1)
    )(x_star, parameters)
    _, transpose = jax.vjp(lambda x: residual(x, parameters), x_star)
    transpose_action = jax.jit(lambda vector: transpose(vector)[0])

    def matrix_vector(vector: np.ndarray) -> np.ndarray:
        return np.asarray(transpose_action(jnp.asarray(vector)), dtype=float)

    apply_preconditioner, scales = _packed_preconditioner(grid, vectorizer)
    probe = np.random.default_rng(0).choice((-1.0, 1.0), size=x_star.size)
    split = (slice(0, vectorizer.radius_size), slice(vectorizer.radius_size, None))
    for block, active in enumerate(split[: 2 if vectorizer.lambda_size else 1]):
        direction = np.zeros_like(probe)
        direction[active] = probe[active]
        response = apply_preconditioner(matrix_vector(direction))
        denominator = abs(float(np.dot(direction, response)))
        if denominator > np.finfo(float).tiny:
            scales[block] = np.clip(
                np.dot(direction, direction) / denominator, 1.0e-8, 1.0e8
            )

    operator = LinearOperator((x_star.size, x_star.size), matvec=matrix_vector, dtype=float)
    inverse = LinearOperator(
        (x_star.size, x_star.size), matvec=apply_preconditioner, dtype=float
    )
    iterations = 0

    def count_iteration(_residual: float) -> None:
        nonlocal iterations
        iterations += 1

    adjoint, info = gmres(
        operator,
        np.asarray(quantity_x, dtype=float),
        M=inverse,
        restart=min(50, x_star.size),
        maxiter=int(max_restarts),
        rtol=float(rtol),
        atol=0.0,
        callback=count_iteration,
        callback_type="pr_norm",
    )
    linear_error = matrix_vector(adjoint) - np.asarray(quantity_x, dtype=float)
    relative_residual = float(
        np.linalg.norm(linear_error)
        / max(np.linalg.norm(np.asarray(quantity_x)), np.finfo(float).tiny)
    )
    _, parameter_pullback = jax.vjp(lambda p: residual(x_star, p), parameters)
    residual_parameter_gradient = parameter_pullback(jnp.asarray(adjoint))[0]
    total_gradient = jax.tree.map(
        lambda direct, implicit: direct - implicit,
        quantity_parameters,
        residual_parameter_gradient,
    )
    return MirrorAdjointResult(
        value=value,
        gradient=total_gradient,
        iterations=iterations,
        relative_residual=relative_residual,
        converged=bool(info == 0 and relative_residual <= max(10.0 * rtol, 1.0e-12)),
    )


__all__ = [
    "FixedBoundaryParameters",
    "MirrorAdjointResult",
    "fixed_boundary_adjoint",
    "fixed_boundary_parameters",
]
