"""Implicit fixed, spline, and free-boundary mirror derivatives.

Derivatives use converged equilibrium equations, not host iteration histories.
"""

from __future__ import annotations

from dataclasses import dataclass
import functools
from types import SimpleNamespace
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres
from solvax import block_thomas_factor, block_thomas_solve

from .exterior_bie import AxisymmetricExteriorVacuum, solve_axisymmetric_exterior_vacuum
from .forces import MU0, MirrorEnergy, mirror_energy
from .geometry import magnetic_field_squared, regularize_axis_stream_function
from .model import MirrorBoundary, MirrorState, project_fixed_boundary_state
from .solver import (
    _MirrorStateVectorizer,
    _packed_preconditioner,
    solve_fixed_boundary_cli,
)

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
class SplineFixedBoundaryParameters:
    """Differentiable controls for a coefficient-native spline equilibrium."""

    boundary_coefficients: Array
    axial_flux_derivative: Array
    mass_profile: Array
    current_derivative: Array


@dataclass(frozen=True)
class MirrorAdjointResult:
    """Scalar value, total parameter gradient, and linear-solve diagnostics."""

    value: Array
    gradient: Any
    iterations: int
    relative_residual: float
    converged: bool
    linear_solver: str


@dataclass(frozen=True)
class MirrorTangentResult:
    """Equilibrium state tangent and linear-solve diagnostics."""

    tangent: MirrorState
    iterations: int
    relative_residual: float
    converged: bool


@dataclass(frozen=True, eq=False)
class FixedBoundaryImplicitConfig:
    """Static numerical context for a differentiable fixed-boundary solve."""

    initial_state: MirrorState
    grid: Any
    config: Any
    gamma: float = 5.0 / 3.0
    solve_lambda: bool = False
    gradient_tolerance: float = 1.0e-11
    adjoint_rtol: float = 1.0e-10
    adjoint_max_restarts: int = 20


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
    SplineFixedBoundaryParameters,
    data_fields=[
        "boundary_coefficients",
        "axial_flux_derivative",
        "mass_profile",
        "current_derivative",
    ],
    meta_fields=[],
)
jax.tree_util.register_dataclass(
    MirrorAdjointResult,
    data_fields=["value", "gradient"],
    meta_fields=["iterations", "relative_residual", "converged", "linear_solver"],
)
jax.tree_util.register_dataclass(
    MirrorTangentResult,
    data_fields=["tangent"],
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


def spline_fixed_boundary_parameters(
    boundary: Any,
    *,
    axial_flux_derivative: Array,
    mass_profile: Array = 0.0,
    current_derivative: Array = 0.0,
) -> SplineFixedBoundaryParameters:
    """Collect native spline controls in a differentiable pytree."""

    return SplineFixedBoundaryParameters(
        boundary_coefficients=jnp.asarray(boundary.radius_coefficients),
        axial_flux_derivative=jnp.asarray(axial_flux_derivative),
        mass_profile=jnp.asarray(mass_profile),
        current_derivative=jnp.asarray(current_derivative),
    )


def make_fixed_boundary_implicit_config(
    initial_state: MirrorState,
    grid: Any,
    config: Any,
    *,
    gamma: float = 5.0 / 3.0,
    solve_lambda: bool = False,
    gradient_tolerance: float = 1.0e-11,
    adjoint_rtol: float = 1.0e-10,
    adjoint_max_restarts: int = 20,
) -> FixedBoundaryImplicitConfig:
    """Build the static context used by :func:`solve_fixed_boundary_implicit`."""

    return FixedBoundaryImplicitConfig(
        initial_state=initial_state,
        grid=grid,
        config=config,
        gamma=gamma,
        solve_lambda=solve_lambda,
        gradient_tolerance=gradient_tolerance,
        adjoint_rtol=adjoint_rtol,
        adjoint_max_restarts=adjoint_max_restarts,
    )


def _solve_implicit_system(
    matrix_vector: Callable[[np.ndarray], np.ndarray],
    right_hand_side: np.ndarray,
    apply_preconditioner: Callable[[np.ndarray], np.ndarray],
    scales: np.ndarray,
    block_slices: tuple[slice, ...],
    *,
    rtol: float,
    max_restarts: int,
    initial: np.ndarray | None = None,
) -> tuple[np.ndarray, int, float, bool]:
    """Solve one preconditioned implicit linear system on the host."""

    size = right_hand_side.size
    probe = np.random.default_rng(0).choice((-1.0, 1.0), size=size)
    for block, active in enumerate(block_slices):
        direction = np.zeros_like(probe)
        direction[active] = probe[active]
        response = apply_preconditioner(matrix_vector(direction))
        denominator = abs(float(np.dot(direction, response)))
        if denominator > np.finfo(float).tiny:
            scales[block] = np.clip(np.dot(direction, direction) / denominator, 1.0e-8, 1.0e8)

    operator = LinearOperator((size, size), matvec=matrix_vector, dtype=float)
    inverse = LinearOperator((size, size), matvec=apply_preconditioner, dtype=float)
    if initial is None:
        initial_relative_residual = np.inf
    else:
        initial_error = matrix_vector(initial) - right_hand_side
        initial_relative_residual = float(
            np.linalg.norm(initial_error) / max(np.linalg.norm(right_hand_side), np.finfo(float).tiny)
        )
    iterations = 0

    def count_iteration(_residual: float) -> None:
        nonlocal iterations
        iterations += 1

    if initial_relative_residual <= rtol:
        solution, info = initial, 0
    else:
        solution, info = gmres(
            operator,
            right_hand_side,
            x0=initial,
            M=inverse,
            restart=min(50, size),
            maxiter=min(3, max_restarts) if initial is not None else max_restarts,
            rtol=rtol,
            atol=0.0,
            callback=count_iteration,
            callback_type="pr_norm",
        )
    linear_error = matrix_vector(solution) - right_hand_side
    relative_residual = float(np.linalg.norm(linear_error) / max(np.linalg.norm(right_hand_side), np.finfo(float).tiny))
    converged = bool(info == 0 and relative_residual <= max(10.0 * rtol, 1.0e-12))
    return solution, iterations, relative_residual, converged


def _implicit_adjoint(
    x_star: Array,
    parameters: Any,
    residual: Callable[[Array, Any], Array],
    evaluate_quantity: Callable[[Array, Any], Array],
    apply_preconditioner: Callable[[np.ndarray], np.ndarray],
    scales: np.ndarray,
    block_slices: tuple[slice, ...],
    *,
    rtol: float,
    max_restarts: int,
    initializer: Callable[[Callable[[Array], Array], np.ndarray], tuple[np.ndarray, str]] | None = None,
) -> MirrorAdjointResult:
    """Solve one implicit transpose system shared by nodal and spline states."""

    value, (quantity_x, quantity_parameters) = jax.value_and_grad(evaluate_quantity, argnums=(0, 1))(x_star, parameters)
    _, transpose = jax.vjp(lambda x: residual(x, parameters), x_star)
    transpose_action = jax.jit(lambda vector: transpose(vector)[0])

    def matrix_vector(vector: np.ndarray) -> np.ndarray:
        return np.asarray(transpose_action(jnp.asarray(vector)), dtype=float)

    right_hand_side = np.asarray(quantity_x, dtype=float)
    initial_adjoint, solver_used = (
        (None, "gmres") if initializer is None else initializer(transpose_action, right_hand_side)
    )
    adjoint, iterations, relative_residual, converged = _solve_implicit_system(
        matrix_vector,
        right_hand_side,
        apply_preconditioner,
        scales,
        block_slices,
        rtol=rtol,
        max_restarts=max_restarts,
        initial=initial_adjoint,
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
        converged=converged,
        linear_solver=solver_used,
    )


def fixed_boundary_adjoint(
    result: Any,
    parameters: FixedBoundaryParameters,
    grid: Any,
    quantity: MirrorQuantity,
    *,
    gamma: float = 5.0 / 3.0,
    solve_lambda: bool = False,
    linear_solver: str = "gmres",
    rtol: float = 1.0e-10,
    max_restarts: int = 20,
) -> MirrorAdjointResult:
    """Differentiate one scalar quantity through a converged equilibrium.

    The residual is the normalized MHD energy gradient after fixed geometry
    and gauge elimination. ``"gmres"`` is the one-RHS default; ``"block"``
    verifies the nearest-radial Hessian with SOLVAX block Thomas.
    """

    if not bool(result.converged):
        raise ValueError("implicit differentiation requires a converged mirror result")
    if not isinstance(result.energy, MirrorEnergy):
        raise ValueError("unsupported converged mirror energy model")
    if rtol <= 0.0 or max_restarts < 1:
        raise ValueError("rtol and max_restarts must be positive")
    if linear_solver not in {"block", "gmres"}:
        raise ValueError("linear_solver must be 'block' or 'gmres'")
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
        state = project_fixed_boundary_state(
            vectorizer.unpack(x),
            MirrorBoundary(controls.boundary_radius),
            grid,
        )
        return regularize_axis_stream_function(
            state,
            grid,
            controls.axial_flux_derivative,
        )

    def energy_at(x: Array, controls: FixedBoundaryParameters) -> MirrorEnergy:
        state = state_at(x, controls)
        return mirror_energy(
            state,
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

    apply_preconditioner, scales = _packed_preconditioner(grid, vectorizer)
    split = (slice(0, vectorizer.radius_size), slice(vectorizer.radius_size, None))

    def block_initializer(transpose_action, right_hand_side):
        radial_blocks = grid.ns - 2
        block_size = grid.ntheta * (grid.nxi - 2)
        if radial_blocks * block_size != x_star.size:
            raise ValueError("packed geometry does not match radial block structure")
        colors = jnp.repeat(jnp.arange(3), block_size)
        columns = jnp.tile(jnp.arange(block_size), 3)
        active_rows = jnp.arange(radial_blocks)[None, :] % 3 == colors[:, None]
        probes = (
            active_rows[:, :, None] * jax.nn.one_hot(columns, block_size, dtype=x_star.dtype)[:, None, :]
        ).reshape(3 * block_size, x_star.size)
        responses = jax.jit(jax.vmap(transpose_action))(probes).reshape(3, block_size, radial_blocks, block_size)
        radial_index = jnp.arange(radial_blocks)

        def band(offset: int) -> Array:
            values = responses[(radial_index + offset) % 3, :, radial_index, :]
            return jnp.swapaxes(values, 1, 2)

        factors = block_thomas_factor(band(-1), band(0), band(1))
        initial_adjoint = np.asarray(
            block_thomas_solve(factors, jnp.asarray(right_hand_side).reshape(radial_blocks, block_size))
        ).reshape(-1)
        return initial_adjoint, "block+gmres"

    return _implicit_adjoint(
        x_star,
        parameters,
        residual,
        evaluate_quantity,
        apply_preconditioner,
        scales,
        split[: 2 if vectorizer.lambda_size else 1],
        rtol=rtol,
        max_restarts=max_restarts,
        initializer=(block_initializer if linear_solver == "block" and not vectorizer.lambda_size else None),
    )


def _spline_implicit_problem(
    result: Any,
    parameters: SplineFixedBoundaryParameters,
    discretization: Any,
    *,
    gamma: float,
    solve_lambda: bool,
):
    """Build the converged spline residual and its packed linear model."""

    from .splines import (
        SplineMirrorBoundary,
        _SplineStateVectorizer,
        _packed_spline_preconditioner,
    )

    evaluated = result.evaluated
    if not bool(evaluated.converged):
        raise ValueError("implicit differentiation requires a converged mirror result")
    if not isinstance(evaluated.energy, MirrorEnergy):
        raise ValueError("spline implicit derivatives require the isotropic energy")
    boundary = SplineMirrorBoundary(jnp.asarray(parameters.boundary_coefficients))
    vectorizer = _SplineStateVectorizer.build(
        result.coefficient_state,
        boundary,
        discretization,
        axial_flux_derivative=parameters.axial_flux_derivative,
        solve_lambda=solve_lambda,
    )
    x_star = jnp.asarray(vectorizer.pack())
    energy_scale = max(abs(float(evaluated.energy.total)), np.finfo(float).tiny)

    def state_at(x: Array, controls: SplineFixedBoundaryParameters) -> MirrorState:
        coefficients = discretization.project_fixed_boundary(
            vectorizer.unpack(x),
            SplineMirrorBoundary(controls.boundary_coefficients),
        )
        return regularize_axis_stream_function(
            discretization.evaluate_state(coefficients),
            discretization.grid,
            controls.axial_flux_derivative,
        )

    def energy_at(x: Array, controls: SplineFixedBoundaryParameters) -> MirrorEnergy:
        return mirror_energy(
            state_at(x, controls),
            discretization.grid,
            axial_flux_derivative=controls.axial_flux_derivative,
            mass_profile=controls.mass_profile,
            current_derivative=controls.current_derivative,
            gamma=gamma,
        )

    def residual(x: Array, controls: SplineFixedBoundaryParameters) -> Array:
        return jax.grad(lambda vector: energy_at(vector, controls).total / energy_scale)(x)

    apply_preconditioner, scales = _packed_spline_preconditioner(discretization, vectorizer)
    split = (slice(0, vectorizer.radius_size), slice(vectorizer.radius_size, None))
    return (
        x_star,
        state_at,
        energy_at,
        residual,
        apply_preconditioner,
        scales,
        split[: 2 if vectorizer.lambda_size else 1],
    )


def spline_fixed_boundary_adjoint(
    result: Any,
    parameters: SplineFixedBoundaryParameters,
    discretization: Any,
    quantity: MirrorQuantity,
    *,
    gamma: float = 5.0 / 3.0,
    solve_lambda: bool = False,
    rtol: float = 1.0e-10,
    max_restarts: int = 20,
) -> MirrorAdjointResult:
    """Differentiate a scalar through a converged spline equilibrium."""

    if rtol <= 0.0 or max_restarts < 1:
        raise ValueError("rtol and max_restarts must be positive")
    (
        x_star,
        state_at,
        energy_at,
        residual,
        apply_preconditioner,
        scales,
        split,
    ) = _spline_implicit_problem(
        result,
        parameters,
        discretization,
        gamma=gamma,
        solve_lambda=solve_lambda,
    )

    def evaluate_quantity(x: Array, controls: SplineFixedBoundaryParameters) -> Array:
        state = state_at(x, controls)
        value = jnp.asarray(quantity(state, energy_at(x, controls)))
        if value.ndim != 0:
            raise ValueError("mirror adjoint quantity must return a scalar")
        return value

    return _implicit_adjoint(
        x_star,
        parameters,
        residual,
        evaluate_quantity,
        apply_preconditioner,
        scales,
        split,
        rtol=rtol,
        max_restarts=max_restarts,
    )


def spline_fixed_boundary_tangent(
    result: Any,
    parameters: SplineFixedBoundaryParameters,
    parameter_tangent: SplineFixedBoundaryParameters,
    discretization: Any,
    *,
    gamma: float = 5.0 / 3.0,
    solve_lambda: bool = False,
    rtol: float = 1.0e-10,
    max_restarts: int = 20,
) -> MirrorTangentResult:
    """Differentiate a converged spline state in one control direction."""

    if rtol <= 0.0 or max_restarts < 1:
        raise ValueError("rtol and max_restarts must be positive")
    (
        x_star,
        state_at,
        _,
        residual,
        apply_preconditioner,
        scales,
        split,
    ) = _spline_implicit_problem(
        result,
        parameters,
        discretization,
        gamma=gamma,
        solve_lambda=solve_lambda,
    )
    parameter_tangent = jax.tree.map(jnp.asarray, parameter_tangent)
    residual_tangent = jax.jvp(
        lambda controls: residual(x_star, controls),
        (parameters,),
        (parameter_tangent,),
    )[1]
    tangent_action = jax.jit(
        lambda direction: jax.jvp(
            lambda x: residual(x, parameters),
            (x_star,),
            (direction,),
        )[1]
    )

    def matrix_vector(vector: np.ndarray) -> np.ndarray:
        return np.asarray(tangent_action(jnp.asarray(vector)), dtype=float)

    packed_tangent, iterations, relative_residual, converged = _solve_implicit_system(
        matrix_vector,
        -np.asarray(residual_tangent, dtype=float),
        apply_preconditioner,
        scales,
        split,
        rtol=rtol,
        max_restarts=max_restarts,
    )
    state_tangent = jax.jvp(
        state_at,
        (x_star, parameters),
        (jnp.asarray(packed_tangent), parameter_tangent),
    )[1]
    return MirrorTangentResult(
        tangent=state_tangent,
        iterations=iterations,
        relative_residual=relative_residual,
        converged=converged,
    )


def _state_shape(config: FixedBoundaryImplicitConfig) -> MirrorState:
    shape = config.grid.shape
    field = jax.ShapeDtypeStruct(shape, jnp.float64)
    return MirrorState(radius_scale=field, lambda_stream=field)


def _parameter_shape(parameters: FixedBoundaryParameters) -> FixedBoundaryParameters:
    return jax.tree.map(lambda value: jax.ShapeDtypeStruct(value.shape, value.dtype), parameters)


def _host_fixed_boundary_solve(config: FixedBoundaryImplicitConfig, parameters: FixedBoundaryParameters) -> MirrorState:
    parameters = jax.tree.map(jnp.asarray, parameters)
    boundary = MirrorBoundary(parameters.boundary_radius)
    initial = project_fixed_boundary_state(config.initial_state, boundary, config.grid)
    common = dict(
        axial_flux_derivative=parameters.axial_flux_derivative,
        current_derivative=parameters.current_derivative,
        solve_lambda=config.solve_lambda,
        gradient_tolerance=config.gradient_tolerance,
        require_convergence=True,
    )
    result = solve_fixed_boundary_cli(
        initial,
        boundary,
        config.grid,
        config.config,
        mass_profile=parameters.mass_profile,
        gamma=config.gamma,
        **common,
    )
    return jax.tree.map(lambda value: np.asarray(value, dtype=np.float64), result.state)


@functools.partial(jax.custom_vjp, nondiff_argnums=(1,))
def solve_fixed_boundary_implicit(
    parameters: FixedBoundaryParameters, config: FixedBoundaryImplicitConfig
) -> MirrorState:
    """Return a converged state with an implicit reverse derivative.

    Host iterations stay outside the AD tape through :func:`jax.pure_callback`;
    reverse memory is therefore independent of the primal iteration count.
    """

    return jax.pure_callback(
        functools.partial(_host_fixed_boundary_solve, config),
        _state_shape(config),
        parameters,
    )


def _solve_fixed_boundary_implicit_fwd(parameters, config):
    state = jax.pure_callback(
        functools.partial(_host_fixed_boundary_solve, config),
        _state_shape(config),
        parameters,
    )
    return state, (parameters, state)


def _host_fixed_boundary_pullback(config, parameters, state, cotangent):
    parameters = jax.tree.map(jnp.asarray, parameters)
    state = jax.tree.map(jnp.asarray, state)
    cotangent = jax.tree.map(jnp.asarray, cotangent)
    energy = mirror_energy(
        state,
        config.grid,
        axial_flux_derivative=parameters.axial_flux_derivative,
        mass_profile=parameters.mass_profile,
        current_derivative=parameters.current_derivative,
        gamma=config.gamma,
    )

    def cotangent_quantity(candidate, _energy):
        return sum(
            jnp.vdot(value, weight)
            for value, weight in zip(jax.tree.leaves(candidate), jax.tree.leaves(cotangent), strict=True)
        )

    result = SimpleNamespace(converged=True, state=state, energy=energy)
    adjoint = fixed_boundary_adjoint(
        result,
        parameters,
        config.grid,
        cotangent_quantity,
        gamma=config.gamma,
        solve_lambda=config.solve_lambda,
        rtol=config.adjoint_rtol,
        max_restarts=config.adjoint_max_restarts,
    )
    if not adjoint.converged:
        raise RuntimeError(f"fixed-boundary adjoint failed at residual {adjoint.relative_residual:.3e}")
    return jax.tree.map(lambda value: np.asarray(value, dtype=np.float64), adjoint.gradient)


def _solve_fixed_boundary_implicit_bwd(config, residual, cotangent):
    parameters, state = residual
    gradient = jax.pure_callback(
        functools.partial(_host_fixed_boundary_pullback, config),
        _parameter_shape(parameters),
        parameters,
        state,
        cotangent,
    )
    return (gradient,)


solve_fixed_boundary_implicit.defvjp(_solve_fixed_boundary_implicit_fwd, _solve_fixed_boundary_implicit_bwd)


FreeBoundaryQuantity = Callable[[MirrorBoundary, MirrorState, MirrorEnergy, Any], Array]


@dataclass(frozen=True)
class FreeBoundaryParameters:
    """Differentiable controls for an exterior free-boundary equilibrium."""

    external_field: Any
    axial_flux_derivative: Array
    mass_profile: Array
    current_derivative: Array


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


for _cls, _data, _meta in (
    (FreeBoundaryParameters, ["external_field", "axial_flux_derivative", "mass_profile", "current_derivative"], []),
    (FreeBoundaryAdjointResult, ["value", "gradient"], ["iterations", "relative_residual", "converged"]),
):
    jax.tree_util.register_dataclass(_cls, data_fields=_data, meta_fields=_meta)


def free_boundary_parameters(
    external_field: Any,
    *,
    axial_flux_derivative: Array,
    mass_profile: Array = 0.0,
    current_derivative: Array = 0.0,
) -> FreeBoundaryParameters:
    """Collect free-boundary controls in a differentiable pytree."""

    return FreeBoundaryParameters(
        external_field=external_field,
        axial_flux_derivative=jnp.asarray(axial_flux_derivative),
        mass_profile=jnp.asarray(mass_profile),
        current_derivative=jnp.asarray(current_derivative),
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

    End-cut radii and pressure are fixed. The eliminated exterior Neumann solve
    contributes field and shape derivatives directly to interface-stress rows.
    """

    if not bool(result.converged):
        raise ValueError("free-boundary differentiation requires a converged result")
    if plasma_grid.ntheta != 1:
        raise ValueError("free-boundary adjoint currently supports axisymmetry only")
    if not isinstance(result.vacuum_field, AxisymmetricExteriorVacuum):
        raise ValueError("free-boundary adjoint requires the exterior vacuum backend")
    if config.rtol <= 0.0 or config.max_restarts < 1:
        raise ValueError("adjoint tolerances and iteration limits must be positive")
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
        boundary_radius = fixed_boundary.at[boundary_indices].set(vector[:boundary_size])
        boundary = MirrorBoundary(boundary_radius)
        radius = base_state.radius_scale.at[plasma_indices].set(vector[boundary_size:])
        radius = radius.at[-1].set(boundary_radius)
        radius = radius.at[:, :, 0].set(boundary_radius[:, 0])
        radius = radius.at[:, :, -1].set(boundary_radius[:, -1])
        radius = radius.at[0].set(radius[1])
        return boundary, MirrorState(radius, base_state.lambda_stream)

    def plasma_components(vector: Array, controls: FreeBoundaryParameters):
        boundary, state = unpack(vector)
        plasma = mirror_energy(
            state,
            plasma_grid,
            axial_flux_derivative=controls.axial_flux_derivative,
            mass_profile=controls.mass_profile,
            current_derivative=controls.current_derivative,
            gamma=config.gamma,
        )
        pressure = jnp.broadcast_to(plasma.pressure[:, None, None], plasma.b_squared.shape)
        plasma_b_squared = magnetic_field_squared(plasma.field, plasma.geometry)
        return boundary, state, plasma, pressure, plasma_b_squared

    def components(vector: Array, controls: FreeBoundaryParameters):
        boundary, state, plasma, pressure, plasma_b_squared = plasma_components(vector, controls)
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

    def normalized_plasma_energy(vector: Array, controls: FreeBoundaryParameters) -> Array:
        return plasma_components(vector, controls)[2].total / energy_scale

    def residual(vector: Array, controls: FreeBoundaryParameters) -> Array:
        _, _, _, pressure, plasma_b_squared, vacuum = components(vector, controls)
        plasma_gradient = jax.grad(normalized_plasma_energy, argnums=0)(vector, controls)[boundary_size:]
        plasma_side = plasma_b_squared[-1, 0, 1:-1]
        vacuum_side = jnp.sum(vacuum.lateral_field_xyz[1:-1] ** 2, axis=-1)
        pressure_side = pressure[-1, 0, 1:-1]
        jump = pressure_side + (plasma_side - vacuum_side) / (2.0 * MU0)
        stress_scale = jnp.abs(pressure_side) + plasma_side / (2.0 * MU0) + vacuum_side / (2.0 * MU0)
        stress = jump / jnp.maximum(stress_scale, jnp.finfo(stress_scale.dtype).tiny)
        return jnp.concatenate([stress, plasma_gradient])

    def evaluate_quantity(vector: Array, controls: FreeBoundaryParameters) -> Array:
        boundary, state, plasma, _, _, vacuum = components(vector, controls)
        value = jnp.asarray(quantity(boundary, state, plasma, vacuum))
        if value.ndim != 0:
            raise ValueError("free-boundary adjoint quantity must return a scalar")
        return value

    equilibrium_residual = np.asarray(residual(x_star, parameters), dtype=float)
    if np.max(np.abs(equilibrium_residual)) > 10.0 * max(float(result.variational_max), 1.0e-12):
        raise ValueError("reconstructed free-boundary residual does not match result")
    value, (quantity_x, quantity_parameters) = jax.value_and_grad(evaluate_quantity, argnums=(0, 1))(x_star, parameters)
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
    plasma_preconditioner, _ = _packed_preconditioner(plasma_grid, vectorizer)
    scales = np.ones(2)

    def apply_preconditioner(vector: np.ndarray) -> np.ndarray:
        output = np.array(vector, dtype=float, copy=True)
        output[:boundary_size] *= scales[0]
        output[boundary_size:] = scales[1] * plasma_preconditioner(vector[boundary_size:])
        return output

    right_hand_side = np.asarray(quantity_x, dtype=float)
    adjoint, iterations, relative_residual, converged = _solve_implicit_system(
        matrix_vector,
        right_hand_side,
        apply_preconditioner,
        scales,
        (slice(0, boundary_size), slice(boundary_size, None)),
        rtol=config.rtol,
        max_restarts=config.max_restarts,
    )
    _, parameter_pullback = jax.vjp(lambda controls: residual(x_star, controls), parameters)
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
        converged=converged,
    )


__all__ = [
    "FreeBoundaryAdjointConfig",
    "FreeBoundaryAdjointResult",
    "FreeBoundaryParameters",
    "free_boundary_adjoint",
    "free_boundary_parameters",
]

__all__ = [
    "FreeBoundaryAdjointConfig",
    "FreeBoundaryAdjointResult",
    "FreeBoundaryParameters",
    "FixedBoundaryImplicitConfig",
    "FixedBoundaryParameters",
    "MirrorAdjointResult",
    "MirrorTangentResult",
    "SplineFixedBoundaryParameters",
    "fixed_boundary_adjoint",
    "fixed_boundary_parameters",
    "free_boundary_adjoint",
    "free_boundary_parameters",
    "make_fixed_boundary_implicit_config",
    "spline_fixed_boundary_adjoint",
    "spline_fixed_boundary_tangent",
    "spline_fixed_boundary_parameters",
    "solve_fixed_boundary_implicit",
]
