"""Shared host optimization and preconditioning for mirror equilibria.

The coefficient fixed-boundary solver and the nodal free-boundary solver use
the same convergence contract: optimizer status never substitutes for a
normalized variational residual below ``MirrorConfig.ftol``. Open systems use
matrix-free Newton-GMRES at every size, with a bounded dense rescue for small
stalled systems. Closed periodic spline systems retain an exact dense Newton
reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

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
)
from .model import MirrorConfig, MirrorState

Array = Any


def _valid_energy_objective(energy: MirrorEnergy, energy_scale: float) -> Array:
    """Normalize energy and reject states with crossed flux surfaces."""

    value = energy.total / float(energy_scale)
    return jnp.where(energy.geometry.jacobian_sign_changed, jnp.inf, value)


@dataclass(frozen=True)
class MirrorSolveResult:
    """Solved state, diagnostics, and dense iteration history.

    History columns are ``iteration, total_energy, radius_variational_rms,
    lambda_variational_rms, variational_max, pointwise_force_rms``.
    ``optimizer_success`` is recorded separately and never substitutes for
    ``converged``.
    """

    state: MirrorState
    energy: MirrorEnergy
    variational: VariationalResidual
    force: IsotropicForceResidual
    staggered_weak_force: VariationalResidual | None
    normalized_divergence_rms: Array
    history: Array
    iterations: int
    converged: bool
    optimizer_success: bool
    linear_iterations: int
    final_linear_residual: float
    message: str


jax.tree_util.register_dataclass(
    MirrorSolveResult,
    data_fields=[
        "state",
        "energy",
        "variational",
        "force",
        "staggered_weak_force",
        "normalized_divergence_rms",
        "history",
    ],
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


@dataclass(frozen=True)
class SeparableMirrorPreconditioner:
    """Tensor-product inverse for radial, poloidal, and axial stiffness.

    The model is a shifted radial/Fourier/CGL stiffness sum. Its inverse uses
    two small eigendecompositions and FFTs, not a dense 3D factorization.
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
        radial_nodes: int | None = None,
        shift: float = 1.0e-3,
        radial_strength: float = 1.0,
        poloidal_strength: float = 1.0,
        axial_strength: float = 1.0,
    ) -> "SeparableMirrorPreconditioner":
        """Build the normalized separable stiffness inverse for ``grid``."""

        derivative = np.asarray(grid.axial_basis.derivative_matrix, dtype=float)
        weights = np.asarray(grid.axial_basis.weights, dtype=float)
        interior_derivative = derivative[:, 1:-1] / float(grid.dz_dxi)
        axial = interior_derivative.T @ (weights[:, None] * interior_derivative)
        return cls.build_from_axial_stiffness(
            grid,
            axial,
            radial_nodes=radial_nodes,
            shift=shift,
            radial_strength=radial_strength,
            poloidal_strength=poloidal_strength,
            axial_strength=axial_strength,
        )

    @classmethod
    def build_from_axial_stiffness(
        cls,
        grid: "MirrorGrid",
        axial_stiffness: Array,
        *,
        radial_nodes: int | None = None,
        shift: float = 1.0e-3,
        radial_strength: float = 1.0,
        poloidal_strength: float = 1.0,
        axial_strength: float = 1.0,
    ) -> "SeparableMirrorPreconditioner":
        """Build the tensor inverse from a representation-specific axial block."""

        strengths = (radial_strength, poloidal_strength, axial_strength)
        if shift <= 0.0 or min(strengths) < 0.0:
            raise ValueError("shift must be positive and stiffness strengths nonnegative")
        nr = grid.ns - 2 if radial_nodes is None else int(radial_nodes)
        axial = np.asarray(axial_stiffness, dtype=float)
        if axial.ndim != 2 or axial.shape[0] != axial.shape[1]:
            raise ValueError("axial stiffness must be a square matrix")
        nx = int(axial.shape[0])
        if nr < 1 or nx < 1:
            raise ValueError("preconditioning requires interior radial and axial values")

        ds = float(grid.s[1] - grid.s[0])
        radial = np.diag(np.full(nr, 2.0 / ds**2))
        if nr > 1:
            off_diagonal = np.full(nr - 1, -1.0 / ds**2)
            radial += np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)

        radial_values, radial_vectors = eigh(radial, check_finite=True)
        axial_values, axial_vectors = eigh(axial, check_finite=True)
        radial_values /= max(float(radial_values[-1]), np.finfo(float).tiny)
        axial_values /= max(float(axial_values[-1]), np.finfo(float).tiny)
        poloidal_values = np.fft.fftfreq(grid.ntheta, d=1.0 / grid.ntheta) ** 2
        if np.max(poloidal_values) > 0.0:
            poloidal_values /= np.max(poloidal_values)
        denominator = (
            float(shift)
            + float(radial_strength) * radial_values[:, None, None]
            + float(poloidal_strength) * poloidal_values[None, :, None]
            + float(axial_strength) * axial_values[None, None, :]
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
        multiplier = 1.0 / self.denominator if inverse else self.denominator
        coefficients = np.einsum("ri,rtx,xj->itj", self.radial_vectors, blocks, self.axial_vectors)
        coefficients = np.fft.fft(coefficients, axis=1, norm="ortho")
        coefficients = np.fft.ifft(multiplier * coefficients, axis=1, norm="ortho").real
        result = np.einsum("ri,itj,xj->rtx", self.radial_vectors, coefficients, self.axial_vectors)
        return result.reshape(values.shape)

    def apply(self, vector: Array) -> np.ndarray:
        """Apply the inverse model operator to a flattened residual."""
        return self._transform(vector, inverse=True)

    def operator(self, vector: Array) -> np.ndarray:
        """Apply the model operator, primarily for verification tests."""
        return self._transform(vector, inverse=False)

    def apply_gauge_free(
        self,
        vector: Array,
        *,
        free_indices: np.ndarray,
        pivot: int,
        weights: np.ndarray,
    ) -> np.ndarray:
        """Apply the inverse after eliminating one weighted-mean gauge node.

        The lift and orthogonal projection keep the reduced operator symmetric.
        """

        free_indices = np.asarray(free_indices, dtype=int)
        weights = np.asarray(weights, dtype=float)
        reduced = np.asarray(vector, dtype=float).reshape(self.active_shape[0], free_indices.size)
        ratio = weights[free_indices] / weights[int(pivot)]
        gram = 1.0 + ratio @ ratio
        lifted_free = reduced - ((reduced @ ratio) / gram)[:, None] * ratio[None, :]
        lifted = np.zeros((self.active_shape[0], weights.size))
        lifted[:, free_indices] = lifted_free
        lifted[:, int(pivot)] = -(lifted_free @ ratio)

        solved = self.apply(lifted.reshape(-1)).reshape(lifted.shape)
        projected = solved[:, free_indices] - solved[:, int(pivot), None] * ratio[None, :]
        projected -= ((projected @ ratio) / gram)[:, None] * ratio[None, :]
        return projected.reshape(np.asarray(vector).shape)


def _matrix_free_newton_polish(
    x0: np.ndarray,
    gradient_function: Any,
    objective_function: Any,
    vectorizer: Any,
    preconditioner: tuple[Any, np.ndarray, Any],
    *,
    ftol: float,
    max_steps: int,
    record_step: Any,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> tuple[np.ndarray, int, int, float, bool, str]:
    """Damped Newton-GMRES polish using exact JAX Hessian products."""

    x = np.asarray(x0, dtype=float)
    apply_preconditioner, block_scales, build_local = preconditioner

    hessian_vector = jax.jit(lambda point, direction: jax.jvp(gradient_function, (point,), (direction,))[1])
    hessian_columns = jax.jit(
        lambda point, directions: jax.vmap(lambda direction: jax.jvp(gradient_function, (point,), (direction,))[1])(
            directions
        )
    )
    linear_iterations = 0
    final_linear_residual = np.inf
    damping = 1.0e-8
    local_preconditioner = None

    for step_index in range(max(0, int(max_steps))):
        gradient = np.asarray(gradient_function(jnp.asarray(x)), dtype=float)
        gradient_max = float(np.max(np.abs(gradient)))
        if gradient_max <= float(ftol):
            return x, step_index, linear_iterations, final_linear_residual, True, "Newton-GMRES converged"

        def matrix_vector(direction: np.ndarray) -> np.ndarray:
            product = np.asarray(hessian_vector(jnp.asarray(x), jnp.asarray(direction)), dtype=float)
            return product + damping * np.asarray(direction)

        if local_preconditioner is None and build_local is not None:

            def matrix_columns(directions: np.ndarray) -> np.ndarray:
                directions = np.asarray(directions, dtype=float)
                products = hessian_columns(jnp.asarray(x), jnp.asarray(directions))
                return np.asarray(products, dtype=float) + damping * directions

            try:
                local_preconditioner = build_local(matrix_columns)
            except RuntimeError:
                local_preconditioner = False
        elif build_local is None:
            local_preconditioner = False

        if local_preconditioner is False:
            # Match the fallback tensor blocks to the local Hessian scale.
            block_scales[:] = 1.0
            probe = np.random.default_rng(0).choice((-1.0, 1.0), size=x.size)
            split = (slice(0, vectorizer.radius_size), slice(vectorizer.radius_size, None))
            for block, active in enumerate(split[: 2 if vectorizer.lambda_size else 1]):
                direction = np.zeros_like(probe)
                direction[active] = probe[active]
                response = apply_preconditioner(matrix_vector(direction))
                denominator = abs(float(np.dot(direction, response)))
                if denominator > np.finfo(float).tiny:
                    block_scales[block] = np.clip(np.dot(direction, direction) / denominator, 1.0e-8, 1.0e8)
            active_preconditioner = apply_preconditioner
        else:
            active_preconditioner = local_preconditioner

        operator = LinearOperator((x.size, x.size), matvec=matrix_vector, dtype=float)
        inverse = LinearOperator((x.size, x.size), matvec=active_preconditioner, dtype=float)
        iteration_counter = 0

        def count_iteration(_residual: float) -> None:
            nonlocal iteration_counter
            iteration_counter += 1

        direction, info = gmres(
            operator,
            -gradient,
            M=inverse,
            restart=min(200, x.size),
            maxiter=10,
            rtol=min(1.0e-8, max(1.0e-10, 0.1 * gradient_max)),
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
            direction = -active_preconditioner(gradient)

        value = float(objective_function(jnp.asarray(x)))
        slope = float(np.dot(gradient, direction))
        accepted = False
        step_length = 1.0
        for _ in range(24):
            candidate = np.clip(x + step_length * direction, lower_bounds, upper_bounds)
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

    gradient_max = float(np.max(np.abs(np.asarray(gradient_function(jnp.asarray(x)), dtype=float))))
    converged = gradient_max <= float(ftol)
    return (
        x,
        int(max_steps),
        linear_iterations,
        final_linear_residual,
        converged,
        "Newton-GMRES converged" if converged else "Newton-GMRES iteration limit",
    )


@dataclass(frozen=True)
class _OptimizationOutcome:
    """Representation-independent result from the host nonlinear driver."""

    vector: np.ndarray
    iterations: int
    optimizer_success: bool
    linear_iterations: int
    final_linear_residual: float
    message: str


def _optimize_fixed_boundary(
    x0: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    *,
    objective: Any,
    evaluate: Any,
    packed_variational: Any,
    unpack: Any,
    record: Any,
    config: MirrorConfig,
    gradient_tolerance: float,
    matrix_free_context: tuple[Any, tuple[Any, np.ndarray, Any]] | None,
    start_with_residual_newton: bool = False,
) -> _OptimizationOutcome:
    """Run the common host L-BFGS and residual-Newton solve policy.

    Small closed spline systems can start with residual Newton when their
    geometry initializer is already in a valid local basin. Open systems use
    their matrix-free preconditioner at every size; the dense residual solve is
    only a bounded rescue when matrix-free Newton does not reach ``ftol``.
    """

    callback_iterations = 0
    use_matrix_free = matrix_free_context is not None
    history_stride = 10 if use_matrix_free else 1

    def callback(x: np.ndarray) -> None:
        nonlocal callback_iterations
        callback_iterations += 1
        if callback_iterations == 1 or callback_iterations % history_stride == 0:
            record(callback_iterations, x)

    # Reserve iterations for Newton: L-BFGS can stall on relative energy while
    # physical forces are still large.
    polish_cap = 100 if x0.size > 2048 else 50
    if not use_matrix_free:
        polish_cap = 200
    polish_reserve = min(polish_cap, max(1, int(config.max_iterations) // 4))
    available = int(config.max_iterations) - polish_reserve
    lbfgs_budget = max(10, available // 2) if use_matrix_free else max(1, available)
    if start_with_residual_newton and not use_matrix_free:
        final_x = np.asarray(x0)
        optimizer_success = False
        optimizer_message = "started with residual-Newton"
        lbfgs_iterations = 0
    else:
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
        lbfgs_iterations = int(optimization.nit)
    polish_evaluations = 0
    newton_steps = 0
    linear_iterations = 0
    final_linear_residual = 0.0

    candidate_variational = packed_variational(final_x, unpack(jnp.asarray(final_x)))
    gradient_function = jax.jit(jax.grad(objective))

    def run_matrix_free_polish() -> None:
        nonlocal final_x, optimizer_success, optimizer_message
        nonlocal newton_steps, linear_iterations, final_linear_residual
        remaining = max(
            1,
            int(config.max_iterations) - lbfgs_iterations - polish_evaluations - newton_steps,
        )

        def record_newton(x: np.ndarray) -> None:
            nonlocal newton_steps
            newton_steps += 1
            record(callback_iterations + polish_evaluations + newton_steps, x)

        vectorizer, preconditioner = matrix_free_context
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
            vectorizer,
            preconditioner,
            ftol=config.ftol,
            max_steps=min(polish_reserve, remaining),
            record_step=record_newton,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )
        optimizer_success = bool(newton_success)
        optimizer_message += f"; {newton_message}"

    if float(candidate_variational.maximum) > config.ftol and use_matrix_free and matrix_free_context is not None:
        run_matrix_free_polish()
        candidate_variational = packed_variational(final_x, unpack(jnp.asarray(final_x)))

    # A bounded dense fallback is the robust reference lane up to 2048 dofs.
    if float(candidate_variational.maximum) > config.ftol and final_x.size <= 2048:
        hessian_function = jax.jit(jax.jacfwd(jax.grad(objective)))
        remaining = max(1, int(config.max_iterations) - lbfgs_iterations - newton_steps)
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
        candidate_variational = packed_variational(final_x, unpack(jnp.asarray(final_x)))

    return _OptimizationOutcome(
        vector=final_x,
        iterations=lbfgs_iterations + newton_steps + polish_evaluations,
        optimizer_success=optimizer_success,
        linear_iterations=linear_iterations,
        final_linear_residual=final_linear_residual,
        message=optimizer_message,
    )


if TYPE_CHECKING:
    from .basis import MirrorGrid
