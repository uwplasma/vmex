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
    IsotropicForceResidual,
    MirrorEnergy,
    VariationalResidual,
    fixed_boundary_variational_residual,
    isotropic_force_residual,
    mirror_energy,
)
from .model import MirrorBoundary, MirrorConfig, MirrorState, project_fixed_boundary_state

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
    energy: MirrorEnergy
    variational: VariationalResidual
    force: IsotropicForceResidual
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
) -> tuple[np.ndarray, int, int, float, bool, str]:
    """Damped Newton-GMRES polish using exact JAX Hessian products."""

    x = np.asarray(x0, dtype=float)
    preconditioner = SeparableMirrorPreconditioner.build(grid)
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
            (x.size, x.size), matvec=preconditioner.apply, dtype=float
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
            direction = -preconditioner.apply(gradient)

        value = float(objective_function(jnp.asarray(x)))
        slope = float(np.dot(gradient, direction))
        accepted = False
        step_length = 1.0
        for _ in range(24):
            candidate = np.clip(x + step_length * direction, 0.2, 5.0)
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
    gradient_tolerance: float = 1.0e-11,
    require_convergence: bool = False,
) -> MirrorSolveResult:
    """Solve an isotropic fixed-boundary mirror with host L-BFGS control.

    M2 evolves geometry only.  ``lambda_stream`` remains in the physical state
    and energy, but its nonlinear solve is enabled with finite-current M4 after
    the lambda preconditioner and end-flux data are implemented.
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

    projected_initial = project_fixed_boundary_state(initial_state, boundary, grid)
    mask_numpy = _free_radius_mask(grid)
    mask = jnp.asarray(mask_numpy)
    boundary_scale = float(np.mean(np.asarray(boundary.radius_scale)))
    if not np.isfinite(boundary_scale) or boundary_scale <= 0.0:
        raise ValueError("mean boundary radius must be positive and finite")
    x0 = np.asarray(projected_initial.radius_scale)[mask_numpy] / boundary_scale

    energy_kwargs = {
        "axial_flux_derivative": axial_flux_derivative,
        "mass_profile": mass_profile,
        "current_derivative": current_derivative,
        "gamma": gamma,
    }
    initial_energy = mirror_energy(projected_initial, grid, **energy_kwargs)
    energy_scale = max(abs(float(initial_energy.total)), np.finfo(float).tiny)

    def unpack(x: Array) -> MirrorState:
        radius_scale = projected_initial.radius_scale.at[mask].set(
            jnp.asarray(x) * boundary_scale
        )
        return project_fixed_boundary_state(
            MirrorState(radius_scale, projected_initial.lambda_stream),
            boundary,
            grid,
        )

    def objective(x: Array) -> Array:
        return mirror_energy(unpack(x), grid, **energy_kwargs).total / energy_scale

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

    history: list[tuple[float, float, float, float, float, float]] = []

    def record(iteration: int, x: np.ndarray) -> None:
        state = unpack(jnp.asarray(x))
        energy = mirror_energy(state, grid, **energy_kwargs)
        variational = fixed_boundary_variational_residual(
            state, boundary, grid, **energy_kwargs
        )
        force = isotropic_force_residual(energy, grid)
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
        initial_variational = fixed_boundary_variational_residual(
            projected_initial, boundary, grid, **energy_kwargs
        )
        result = MirrorSolveResult(
            state=projected_initial,
            energy=initial_energy,
            variational=initial_variational,
            force=isotropic_force_residual(initial_energy, grid),
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
        bounds=[(0.2, 5.0)] * x0.size,
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
    candidate_variational = fixed_boundary_variational_residual(
        unpack(jnp.asarray(final_x)), boundary, grid, **energy_kwargs
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
        )
        optimizer_success = bool(newton_success)
        optimizer_message += f"; {newton_message}"
        candidate_variational = fixed_boundary_variational_residual(
            unpack(jnp.asarray(final_x)), boundary, grid, **energy_kwargs
        )

    if float(candidate_variational.maximum) > config.ftol and final_x.size <= 512:
        hessian_function = jax.jit(jax.jacfwd(jax.grad(objective)))
        remaining = max(1, int(config.max_iterations) - int(optimization.nit))
        polish = least_squares(
            fun=lambda x: np.asarray(gradient_function(jnp.asarray(x)), dtype=float),
            x0=final_x,
            jac=lambda x: np.asarray(hessian_function(jnp.asarray(x)), dtype=float),
            bounds=(np.full(final_x.size, 0.2), np.full(final_x.size, 5.0)),
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
    final_energy = mirror_energy(final_state, grid, **energy_kwargs)
    final_variational = fixed_boundary_variational_residual(
        final_state, boundary, grid, **energy_kwargs
    )
    final_force = isotropic_force_residual(final_energy, grid)
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


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .basis import MirrorGrid
