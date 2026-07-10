"""Host-controlled fixed-boundary mirror reference solver.

This M2 solver is the non-differentiable CLI/reference lane.  It minimizes the
same JAX energy used by the future traced lane, but lets SciPy control L-BFGS
line searches and early exits.  Crucially, SciPy success is not equilibrium
success: the returned state is converged only when the independently computed
physical tensor-force residual meets ``MirrorConfig.ftol``.

Large M2 systems are polished with separably preconditioned Newton-GMRES;
small systems retain an exact dense Newton reference.  M9 adds implicit
differentiation without changing the energy or force APIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import least_squares, minimize
from scipy.sparse.linalg import LinearOperator, gmres

from .forces import (
    AnisotropicForceResidual,
    AnisotropicMirrorEnergy,
    IsotropicForceResidual,
    MirrorEnergy,
    VariationalResidual,
    anisotropic_fixed_boundary_variational_residual,
    anisotropic_force_residual,
    anisotropic_mirror_energy,
    fixed_boundary_variational_residual,
    isotropic_force_residual,
    mirror_energy,
)
from .model import (
    MirrorBoundary,
    MirrorConfig,
    MirrorState,
    PressureClosure,
    project_fixed_boundary_state,
)

Array = Any


@dataclass(frozen=True)
class MirrorSolveResult:
    """Solved state, diagnostics, and dense iteration history.

    History columns are ``iteration, total_energy, radius_variational_rms,
    lambda_variational_rms, variational_max, continuum_tensor_rms``.
    ``optimizer_success`` is recorded separately and never substitutes for
    ``converged``.
    """

    state: MirrorState
    energy: MirrorEnergy | AnisotropicMirrorEnergy
    variational: VariationalResidual
    force: IsotropicForceResidual | AnisotropicForceResidual
    history: Array
    iterations: int
    converged: bool
    optimizer_success: bool
    linear_iterations: int
    final_linear_residual: float
    message: str


jax.tree_util.register_dataclass(
    MirrorSolveResult,
    data_fields=["state", "energy", "variational", "force", "history"],
    meta_fields=[
        "iterations",
        "converged",
        "optimizer_success",
        "linear_iterations",
        "final_linear_residual",
        "message",
    ],
)


class MirrorConvergenceError(RuntimeError):
    """Raised when a caller requires a physically converged mirror solve."""

    def __init__(self, result: MirrorSolveResult):
        self.result = result
        super().__init__(
            f"mirror solve did not reach ftol: variational force "
            f"{float(result.variational.maximum):.3e} after {result.iterations} iterations"
        )


def _free_radius_mask(grid: "MirrorGrid") -> np.ndarray:
    """Geometry dofs excluding axis, side boundary, and both end cuts."""

    mask = np.zeros(grid.shape, dtype=bool)
    mask[1:-1, :, 1:-1] = True
    return mask


@dataclass(frozen=True)
class _MirrorStateVectorizer:
    """Pack constrained geometry and gauge-free lambda into solver vectors."""

    base: MirrorState
    radius_indices: tuple[np.ndarray, np.ndarray, np.ndarray]
    radius_scale: float
    flux_scale: float
    lambda_free_indices: np.ndarray
    lambda_pivot: int
    lambda_interior_weights: np.ndarray
    lambda_fixed_weighted_sum: np.ndarray
    solve_lambda: bool

    @classmethod
    def build(
        cls,
        state: MirrorState,
        boundary: MirrorBoundary,
        grid: "MirrorGrid",
        *,
        axial_flux_derivative: Array,
        solve_lambda: bool,
    ) -> "_MirrorStateVectorizer":
        base = project_fixed_boundary_state(state, boundary, grid)
        radius_scale = float(np.mean(np.asarray(boundary.radius_scale)))
        if not np.isfinite(radius_scale) or radius_scale <= 0.0:
            raise ValueError("mean boundary radius must be positive and finite")
        flux = np.asarray(axial_flux_derivative, dtype=float)
        flux_scale = max(float(np.max(np.abs(flux))), np.finfo(float).tiny)
        interior_weights = (
            np.asarray(grid.theta_basis.weights)[:, None]
            * np.asarray(grid.axial_basis.weights)[None, 1:-1]
        ).reshape(-1)
        if solve_lambda and interior_weights.size < 2:
            raise ValueError("lambda solve requires at least two interior theta-xi nodes")
        pivot = int(np.argmax(interior_weights)) if interior_weights.size else 0
        free_indices = np.delete(np.arange(interior_weights.size), pivot)
        full_weights = (
            np.asarray(grid.theta_basis.weights)[:, None]
            * np.asarray(grid.axial_basis.weights)[None, :]
        )
        endpoint_weights = np.zeros_like(full_weights)
        endpoint_weights[:, [0, -1]] = full_weights[:, [0, -1]]
        fixed_sum = np.einsum(
            "jk,ijk->i", endpoint_weights, np.asarray(base.lambda_stream)[1:]
        )
        return cls(
            base=base,
            radius_indices=tuple(np.asarray(index) for index in np.nonzero(_free_radius_mask(grid))),
            radius_scale=radius_scale,
            flux_scale=flux_scale,
            lambda_free_indices=free_indices,
            lambda_pivot=pivot,
            lambda_interior_weights=interior_weights,
            lambda_fixed_weighted_sum=fixed_sum,
            solve_lambda=bool(solve_lambda),
        )

    @property
    def radius_size(self) -> int:
        return int(self.radius_indices[0].size)

    @property
    def lambda_size(self) -> int:
        if not self.solve_lambda:
            return 0
        return int((self.base.radius_scale.shape[0] - 1) * self.lambda_free_indices.size)

    @property
    def size(self) -> int:
        return self.radius_size + self.lambda_size

    def pack(self) -> np.ndarray:
        radius = np.asarray(self.base.radius_scale)[self.radius_indices] / self.radius_scale
        if not self.solve_lambda:
            return radius
        interior = np.asarray(self.base.lambda_stream)[1:, :, 1:-1].reshape(
            self.base.radius_scale.shape[0] - 1, -1
        )
        lam = interior[:, self.lambda_free_indices].reshape(-1) / self.flux_scale
        return np.concatenate([radius, lam])

    def unpack(self, vector: Array) -> MirrorState:
        vector = jnp.asarray(vector)
        radius = self.base.radius_scale.at[self.radius_indices].set(
            vector[: self.radius_size] * self.radius_scale
        )
        radius = radius.at[0].set(radius[1])
        if not self.solve_lambda:
            return MirrorState(radius, self.base.lambda_stream)
        shape = self.base.radius_scale.shape
        free = vector[self.radius_size :].reshape(
            shape[0] - 1, self.lambda_free_indices.size
        ) * self.flux_scale
        interior = self.base.lambda_stream[1:, :, 1:-1].reshape(shape[0] - 1, -1)
        interior = interior.at[:, jnp.asarray(self.lambda_free_indices)].set(free)
        weighted_free = jnp.sum(
            free * jnp.asarray(self.lambda_interior_weights[self.lambda_free_indices])[None, :],
            axis=1,
        )
        pivot_value = -(
            jnp.asarray(self.lambda_fixed_weighted_sum) + weighted_free
        ) / float(self.lambda_interior_weights[self.lambda_pivot])
        interior = interior.at[:, self.lambda_pivot].set(pivot_value)
        lam = self.base.lambda_stream.at[1:, :, 1:-1].set(
            interior.reshape(shape[0] - 1, shape[1], shape[2] - 2)
        )
        lam = lam.at[0].set(lam[1])
        return MirrorState(radius, lam)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lower = np.concatenate(
            [np.full(self.radius_size, 0.2), np.full(self.lambda_size, -np.inf)]
        )
        upper = np.concatenate(
            [np.full(self.radius_size, 5.0), np.full(self.lambda_size, np.inf)]
        )
        return lower, upper


@dataclass(frozen=True)
class SeparableMirrorPreconditioner:
    """Tensor-product inverse for the free radial/axial geometry block.

    The model operator is a shifted sum of a second-order radial Dirichlet
    stiffness matrix and a CGL weak-form axial stiffness matrix.  Its inverse
    is applied with two small symmetric eigendecompositions rather than a
    dense factorization of the full ``(s, theta, xi)`` block.  Poloidal rows
    are independent in M2; Fourier-mode coupling is added with M4.
    """

    radial_vectors: np.ndarray
    axial_vectors: np.ndarray
    denominator: np.ndarray
    active_shape: tuple[int, int, int]

    @classmethod
    def build(
        cls,
        grid: "MirrorGrid",
        *,
        shift: float = 1.0e-3,
        radial_strength: float = 1.0,
        axial_strength: float = 1.0,
    ) -> "SeparableMirrorPreconditioner":
        """Build the normalized separable stiffness inverse for ``grid``."""

        if shift <= 0.0 or radial_strength < 0.0 or axial_strength < 0.0:
            raise ValueError("shift must be positive and stiffness strengths nonnegative")
        nr, nx = grid.ns - 2, grid.nxi - 2
        if nr < 1 or nx < 1:
            raise ValueError("preconditioning requires interior radial and axial nodes")

        ds = float(grid.s[1] - grid.s[0])
        radial = np.diag(np.full(nr, 2.0 / ds**2))
        if nr > 1:
            off_diagonal = np.full(nr - 1, -1.0 / ds**2)
            radial += np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)

        derivative = np.asarray(grid.axial_basis.derivative_matrix, dtype=float)
        weights = np.asarray(grid.axial_basis.weights, dtype=float)
        interior_derivative = derivative[:, 1:-1] / float(grid.dz_dxi)
        axial = interior_derivative.T @ (weights[:, None] * interior_derivative)

        radial_values, radial_vectors = eigh(radial, check_finite=True)
        axial_values, axial_vectors = eigh(axial, check_finite=True)
        radial_values /= max(float(radial_values[-1]), np.finfo(float).tiny)
        axial_values /= max(float(axial_values[-1]), np.finfo(float).tiny)
        denominator = (
            float(shift)
            + float(radial_strength) * radial_values[:, None]
            + float(axial_strength) * axial_values[None, :]
        )
        return cls(
            radial_vectors=radial_vectors,
            axial_vectors=axial_vectors,
            denominator=denominator,
            active_shape=(nr, grid.ntheta, nx),
        )

    @property
    def size(self) -> int:
        """Number of active geometry unknowns."""

        return int(np.prod(self.active_shape))

    def _transform(self, vector: Array, *, inverse: bool) -> np.ndarray:
        values = np.asarray(vector, dtype=float)
        if values.size != self.size:
            raise ValueError(f"vector has {values.size} values; expected {self.size}")
        blocks = values.reshape(self.active_shape)
        result = np.empty_like(blocks)
        multiplier = 1.0 / self.denominator if inverse else self.denominator
        for poloidal_index in range(self.active_shape[1]):
            coefficients = (
                self.radial_vectors.T
                @ blocks[:, poloidal_index, :]
                @ self.axial_vectors
            )
            result[:, poloidal_index, :] = (
                self.radial_vectors
                @ (multiplier * coefficients)
                @ self.axial_vectors.T
            )
        return result.reshape(values.shape)

    def apply(self, vector: Array) -> np.ndarray:
        """Apply the inverse model operator to a flattened residual."""

        return self._transform(vector, inverse=True)

    def operator(self, vector: Array) -> np.ndarray:
        """Apply the model operator, primarily for verification tests."""

        return self._transform(vector, inverse=False)


def _matrix_free_newton_polish(
    x0: np.ndarray,
    gradient_function: Any,
    objective_function: Any,
    grid: "MirrorGrid",
    *,
    ftol: float,
    max_steps: int,
    record_step: Any,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> tuple[np.ndarray, int, int, float, bool, str]:
    """Damped Newton-GMRES polish using exact JAX Hessian products."""

    x = np.asarray(x0, dtype=float)
    preconditioner = SeparableMirrorPreconditioner.build(grid)

    def apply_preconditioner(vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=float)
        result = np.array(vector, copy=True)
        result[: preconditioner.size] = preconditioner.apply(
            vector[: preconditioner.size]
        )
        return result

    hessian_vector = jax.jit(
        lambda point, direction: jax.jvp(
            gradient_function, (point,), (direction,)
        )[1]
    )
    linear_iterations = 0
    final_linear_residual = np.inf
    damping = 1.0e-8

    for step_index in range(max(0, int(max_steps))):
        gradient = np.asarray(gradient_function(jnp.asarray(x)), dtype=float)
        gradient_max = float(np.max(np.abs(gradient)))
        if gradient_max <= float(ftol):
            return x, step_index, linear_iterations, final_linear_residual, True, "Newton-GMRES converged"

        def matrix_vector(direction: np.ndarray) -> np.ndarray:
            product = np.asarray(
                hessian_vector(jnp.asarray(x), jnp.asarray(direction)), dtype=float
            )
            return product + damping * np.asarray(direction)

        operator = LinearOperator((x.size, x.size), matvec=matrix_vector, dtype=float)
        inverse = LinearOperator(
            (x.size, x.size), matvec=apply_preconditioner, dtype=float
        )
        iteration_counter = 0

        def count_iteration(_residual: float) -> None:
            nonlocal iteration_counter
            iteration_counter += 1

        direction, info = gmres(
            operator,
            -gradient,
            M=inverse,
            restart=min(50, x.size),
            maxiter=max(20, min(200, x.size)),
            rtol=min(1.0e-3, max(1.0e-10, 0.1 * gradient_max)),
            atol=0.0,
            callback=count_iteration,
            callback_type="pr_norm",
        )
        linear_iterations += iteration_counter
        linear_error = matrix_vector(direction) + gradient
        final_linear_residual = float(
            np.linalg.norm(linear_error) / max(np.linalg.norm(gradient), np.finfo(float).tiny)
        )
        if info < 0 or not np.all(np.isfinite(direction)):
            return x, step_index, linear_iterations, final_linear_residual, False, "GMRES breakdown"
        if float(np.dot(gradient, direction)) >= 0.0:
            direction = -apply_preconditioner(gradient)

        value = float(objective_function(jnp.asarray(x)))
        slope = float(np.dot(gradient, direction))
        accepted = False
        step_length = 1.0
        for _ in range(24):
            candidate = np.clip(
                x + step_length * direction, lower_bounds, upper_bounds
            )
            candidate_value = float(objective_function(jnp.asarray(candidate)))
            if np.isfinite(candidate_value) and candidate_value <= value + 1.0e-4 * step_length * slope:
                x = candidate
                record_step(x)
                accepted = True
                damping = max(1.0e-12, 0.3 * damping)
                break
            step_length *= 0.5
        if not accepted:
            damping *= 10.0
            if damping > 1.0:
                return x, step_index, linear_iterations, final_linear_residual, False, "Newton line search stalled"

    gradient_max = float(
        np.max(np.abs(np.asarray(gradient_function(jnp.asarray(x)), dtype=float)))
    )
    converged = gradient_max <= float(ftol)
    return (
        x,
        int(max_steps),
        linear_iterations,
        final_linear_residual,
        converged,
        "Newton-GMRES converged" if converged else "Newton-GMRES iteration limit",
    )


def solve_fixed_boundary_cli(
    initial_state: MirrorState,
    boundary: MirrorBoundary,
    grid: "MirrorGrid",
    config: MirrorConfig,
    *,
    axial_flux_derivative: Array,
    mass_profile: Array = 0.0,
    current_derivative: Array = 0.0,
    gamma: float = 5.0 / 3.0,
    pressure_closure: PressureClosure | None = None,
    solve_lambda: bool = False,
    gradient_tolerance: float = 1.0e-11,
    require_convergence: bool = False,
) -> MirrorSolveResult:
    """Solve a fixed-boundary mirror with host L-BFGS/Newton control.

    Supplying ``solve_lambda=True`` enables the gauge-free M4 stream-function
    variables while preserving their fixed end-cut values. A
    ``pressure_closure`` selects the consistent ANIMEC functional; otherwise
    the mass-conserving isotropic functional is used.
    """

    initial_state.validate_shape(grid)
    if grid.shape != (
        config.resolution.ns,
        config.resolution.ntheta,
        config.resolution.nxi,
    ):
        raise ValueError("grid resolution does not match MirrorConfig")
    if gradient_tolerance <= 0.0:
        raise ValueError("gradient_tolerance must be positive")

    vectorizer = _MirrorStateVectorizer.build(
        initial_state,
        boundary,
        grid,
        axial_flux_derivative=axial_flux_derivative,
        solve_lambda=solve_lambda,
    )
    projected_initial = vectorizer.base
    x0 = vectorizer.pack()
    lower_bounds, upper_bounds = vectorizer.bounds()

    if pressure_closure is None:
        energy_kwargs = {
            "axial_flux_derivative": axial_flux_derivative,
            "mass_profile": mass_profile,
            "current_derivative": current_derivative,
            "gamma": gamma,
        }

        def evaluate_energy(state: MirrorState) -> MirrorEnergy | AnisotropicMirrorEnergy:
            return mirror_energy(state, grid, **energy_kwargs)

        def evaluate_variational(state: MirrorState) -> VariationalResidual:
            return fixed_boundary_variational_residual(
                state, boundary, grid, **energy_kwargs
            )

        def evaluate_force(
            state: MirrorState, energy: MirrorEnergy | AnisotropicMirrorEnergy
        ) -> IsotropicForceResidual | AnisotropicForceResidual:
            del state
            return isotropic_force_residual(energy, grid)

    else:
        energy_kwargs = {
            "axial_flux_derivative": axial_flux_derivative,
            "current_derivative": current_derivative,
        }

        def evaluate_energy(state: MirrorState) -> MirrorEnergy | AnisotropicMirrorEnergy:
            return anisotropic_mirror_energy(
                state, grid, pressure_closure, **energy_kwargs
            )

        def evaluate_variational(state: MirrorState) -> VariationalResidual:
            return anisotropic_fixed_boundary_variational_residual(
                state, boundary, grid, pressure_closure, **energy_kwargs
            )

        def evaluate_force(
            state: MirrorState, energy: MirrorEnergy | AnisotropicMirrorEnergy
        ) -> IsotropicForceResidual | AnisotropicForceResidual:
            return anisotropic_force_residual(
                state, energy, grid, pressure_closure
            )

    initial_energy = evaluate_energy(projected_initial)
    energy_scale = max(abs(float(initial_energy.total)), np.finfo(float).tiny)

    def unpack(x: Array) -> MirrorState:
        return vectorizer.unpack(x)

    def objective(x: Array) -> Array:
        return evaluate_energy(unpack(x)).total / energy_scale

    value_and_gradient = jax.jit(jax.value_and_grad(objective))
    cache_x: np.ndarray | None = None
    cache_value = 0.0
    cache_gradient = np.empty_like(x0)

    def evaluate(x: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal cache_x, cache_value, cache_gradient
        if cache_x is None or not np.array_equal(x, cache_x):
            value, gradient = value_and_gradient(jnp.asarray(x))
            cache_x = np.array(x, copy=True)
            cache_value = float(value)
            cache_gradient = np.asarray(gradient, dtype=float)
        return cache_value, cache_gradient

    def packed_variational(x: Array, state: MirrorState) -> VariationalResidual:
        variational = evaluate_variational(state)
        if not solve_lambda:
            return variational
        gradient = evaluate(np.asarray(x, dtype=float))[1]
        radius_values = gradient[: vectorizer.radius_size]
        lambda_values = gradient[vectorizer.radius_size :]
        radius_rms = float(np.sqrt(np.mean(radius_values**2)))
        lambda_rms = float(np.sqrt(np.mean(lambda_values**2)))
        maximum = float(
            max(np.max(np.abs(radius_values)), np.max(np.abs(lambda_values)))
        )
        return VariationalResidual(
            radius_gradient=variational.radius_gradient,
            lambda_gradient=variational.lambda_gradient,
            radius_rms=jnp.asarray(radius_rms),
            lambda_rms=jnp.asarray(lambda_rms),
            maximum=jnp.asarray(maximum),
        )

    history: list[tuple[float, float, float, float, float, float]] = []

    def record(iteration: int, x: np.ndarray) -> None:
        state = unpack(jnp.asarray(x))
        energy = evaluate_energy(state)
        variational = packed_variational(x, state)
        force = evaluate_force(state, energy)
        history.append(
            (
                float(iteration),
                float(energy.total),
                float(variational.radius_rms),
                float(variational.lambda_rms),
                float(variational.maximum),
                float(force.normalized_rms),
            )
        )

    record(0, x0)
    if history[-1][4] <= config.ftol:
        initial_variational = packed_variational(x0, projected_initial)
        result = MirrorSolveResult(
            state=projected_initial,
            energy=initial_energy,
            variational=initial_variational,
            force=evaluate_force(projected_initial, initial_energy),
            history=jnp.asarray(history),
            iterations=0,
            converged=True,
            optimizer_success=True,
            linear_iterations=0,
            final_linear_residual=0.0,
            message="initial state satisfies physical ftol",
        )
        return result

    callback_iterations = 0

    def callback(x: np.ndarray) -> None:
        nonlocal callback_iterations
        callback_iterations += 1
        record(callback_iterations, x)

    lbfgs_budget = int(config.max_iterations)
    if x0.size > 512:
        lbfgs_budget = max(10, lbfgs_budget // 2)
    optimization = minimize(
        fun=lambda x: evaluate(x)[0],
        x0=x0,
        jac=lambda x: evaluate(x)[1],
        method="L-BFGS-B",
        bounds=list(zip(lower_bounds, upper_bounds, strict=True)),
        callback=callback,
        options={
            "maxiter": lbfgs_budget,
            "gtol": float(gradient_tolerance),
            "ftol": np.finfo(float).eps,
            "maxls": 50,
            "maxcor": 20,
        },
    )
    final_x = np.asarray(optimization.x)
    optimizer_success = bool(optimization.success)
    optimizer_message = str(optimization.message)
    polish_evaluations = 0
    newton_steps = 0
    linear_iterations = 0
    final_linear_residual = 0.0

    # L-BFGS commonly reaches machine-level relative energy change before the
    # physical force is small.  An exact dense residual-Newton polish is a
    # reliable M2 reference for modest systems; M4 replaces this size-limited
    # path with matrix-free Newton-GMRES and the separable preconditioner.
    candidate_variational = packed_variational(
        final_x, unpack(jnp.asarray(final_x))
    )
    gradient_function = jax.jit(jax.grad(objective))
    if float(candidate_variational.maximum) > config.ftol and final_x.size > 512:
        remaining = max(1, int(config.max_iterations) - int(optimization.nit))

        def record_newton(x: np.ndarray) -> None:
            nonlocal newton_steps
            newton_steps += 1
            record(callback_iterations + newton_steps, x)

        (
            final_x,
            _attempted_newton_steps,
            linear_iterations,
            final_linear_residual,
            newton_success,
            newton_message,
        ) = _matrix_free_newton_polish(
            final_x,
            gradient_function,
            objective,
            grid,
            ftol=config.ftol,
            max_steps=min(30, remaining),
            record_step=record_newton,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )
        optimizer_success = bool(newton_success)
        optimizer_message += f"; {newton_message}"
        candidate_variational = packed_variational(
            final_x, unpack(jnp.asarray(final_x))
        )

    if float(candidate_variational.maximum) > config.ftol and final_x.size <= 512:
        hessian_function = jax.jit(jax.jacfwd(jax.grad(objective)))
        remaining = max(1, int(config.max_iterations) - int(optimization.nit))
        polish = least_squares(
            fun=lambda x: np.asarray(gradient_function(jnp.asarray(x)), dtype=float),
            x0=final_x,
            jac=lambda x: np.asarray(hessian_function(jnp.asarray(x)), dtype=float),
            bounds=(lower_bounds, upper_bounds),
            method="trf",
            ftol=1.0e-14,
            xtol=1.0e-14,
            gtol=1.0e-14,
            x_scale="jac",
            max_nfev=remaining,
        )
        final_x = np.asarray(polish.x)
        polish_evaluations = int(polish.nfev)
        optimizer_success = bool(polish.success)
        optimizer_message += f"; residual-Newton: {polish.message}"

    final_state = unpack(jnp.asarray(final_x))
    final_energy = evaluate_energy(final_state)
    final_variational = packed_variational(final_x, final_state)
    final_force = evaluate_force(final_state, final_energy)
    record(callback_iterations + newton_steps + polish_evaluations, final_x)
    converged = bool(
        float(final_variational.maximum) <= config.ftol
        and not bool(final_energy.geometry.jacobian_sign_changed)
    )
    message = optimizer_message
    if not converged:
        message += f"; variational force={float(final_variational.maximum):.3e}"
    result = MirrorSolveResult(
        state=final_state,
        energy=final_energy,
        variational=final_variational,
        force=final_force,
        history=jnp.asarray(history),
        iterations=int(optimization.nit) + newton_steps + polish_evaluations,
        converged=converged,
        optimizer_success=optimizer_success,
        linear_iterations=linear_iterations,
        final_linear_residual=final_linear_residual,
        message=message,
    )
    if require_convergence and not converged:
        raise MirrorConvergenceError(result)
    return result


def solve_anisotropic_fixed_boundary_cli(
    initial_state: MirrorState,
    boundary: MirrorBoundary,
    grid: "MirrorGrid",
    config: MirrorConfig,
    closure: PressureClosure,
    *,
    axial_flux_derivative: Array,
    current_derivative: Array = 0.0,
    solve_lambda: bool = False,
    gradient_tolerance: float = 1.0e-11,
    require_convergence: bool = False,
) -> MirrorSolveResult:
    """Solve a consistent anisotropic fixed-boundary mirror equilibrium."""

    return solve_fixed_boundary_cli(
        initial_state,
        boundary,
        grid,
        config,
        axial_flux_derivative=axial_flux_derivative,
        current_derivative=current_derivative,
        pressure_closure=closure,
        solve_lambda=solve_lambda,
        gradient_tolerance=gradient_tolerance,
        require_convergence=require_convergence,
    )


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .basis import MirrorGrid
