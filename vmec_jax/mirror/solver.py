"""Host-controlled fixed-boundary mirror reference solver.

This M2 solver is the non-differentiable CLI/reference lane.  It minimizes the
same JAX energy used by the future traced lane, but lets SciPy control L-BFGS
line searches and early exits.  Crucially, SciPy success is not equilibrium
success: the returned state is converged only when the normalized variational
force meets ``MirrorConfig.ftol``. The independently differenced tensor force
and ``div(B)`` remain discretization-verification diagnostics.

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

from ..core.device import resolve_device
from .forces import (
    IsotropicForceResidual,
    MirrorEnergy,
    VariationalResidual,
    fixed_boundary_variational_residual,
    isotropic_force_residual,
    isotropic_staggered_fixed_boundary_gradient,
    isotropic_staggered_weak_residual,
    mirror_energy,
)
from .geometry import normalized_divergence_rms, regularize_axis_stream_function
from .model import MirrorBoundary, MirrorConfig, MirrorState, project_fixed_boundary_state

Array = Any
_DEVICE_ACTIVE = object()
_HOST_REFERENCE_MAX_SIZE = 1024


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
        """Build the constrained state-to-solver-vector mapping."""
        base = project_fixed_boundary_state(state, boundary, grid)
        radius_scale = float(np.mean(np.asarray(boundary.radius_scale)))
        if not np.isfinite(radius_scale) or radius_scale <= 0.0:
            raise ValueError("mean boundary radius must be positive and finite")
        flux = np.asarray(axial_flux_derivative, dtype=float)
        flux_scale = max(float(np.max(np.abs(flux))), np.finfo(float).tiny)
        interior_weights = (
            np.asarray(grid.theta_basis.weights)[:, None] * np.asarray(grid.axial_basis.weights)[None, 1:-1]
        ).reshape(-1)
        if solve_lambda and interior_weights.size < 2:
            raise ValueError("lambda solve requires at least two interior theta-xi nodes")
        pivot = int(np.argmax(interior_weights)) if interior_weights.size else 0
        free_indices = np.delete(np.arange(interior_weights.size), pivot)
        full_weights = np.asarray(grid.theta_basis.weights)[:, None] * np.asarray(grid.axial_basis.weights)[None, :]
        endpoint_weights = np.zeros_like(full_weights)
        endpoint_weights[:, [0, -1]] = full_weights[:, [0, -1]]
        fixed_sum = np.einsum("jk,ijk->i", endpoint_weights, np.asarray(base.lambda_stream)[1:])
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
        """Return the number of independently solved radius values."""
        return int(self.radius_indices[0].size)

    @property
    def lambda_size(self) -> int:
        """Return the number of gauge-free stream-function values."""
        if not self.solve_lambda:
            return 0
        return int((self.base.radius_scale.shape[0] - 1) * self.lambda_free_indices.size)

    @property
    def size(self) -> int:
        """Return the total solver-vector length."""
        return self.radius_size + self.lambda_size

    def pack(self) -> np.ndarray:
        """Pack the constrained mirror state into normalized solver variables."""
        radius = np.asarray(self.base.radius_scale)[self.radius_indices] / self.radius_scale
        if not self.solve_lambda:
            return radius
        interior = np.asarray(self.base.lambda_stream)[1:, :, 1:-1].reshape(self.base.radius_scale.shape[0] - 1, -1)
        lam = interior[:, self.lambda_free_indices].reshape(-1) / self.flux_scale
        return np.concatenate([radius, lam])

    def unpack(self, vector: Array) -> MirrorState:
        """Reconstruct a constrained mirror state from solver variables."""
        vector = jnp.asarray(vector)
        radius = self.base.radius_scale.at[self.radius_indices].set(vector[: self.radius_size] * self.radius_scale)
        radius = radius.at[0].set(radius[1])
        if not self.solve_lambda:
            return MirrorState(radius, self.base.lambda_stream)
        shape = self.base.radius_scale.shape
        free = vector[self.radius_size :].reshape(shape[0] - 1, self.lambda_free_indices.size) * self.flux_scale
        interior = self.base.lambda_stream[1:, :, 1:-1].reshape(shape[0] - 1, -1)
        interior = interior.at[:, jnp.asarray(self.lambda_free_indices)].set(free)
        weighted_free = jnp.sum(
            free * jnp.asarray(self.lambda_interior_weights[self.lambda_free_indices])[None, :],
            axis=1,
        )
        pivot_value = -(jnp.asarray(self.lambda_fixed_weighted_sum) + weighted_free) / float(
            self.lambda_interior_weights[self.lambda_pivot]
        )
        interior = interior.at[:, self.lambda_pivot].set(pivot_value)
        lam = self.base.lambda_stream.at[1:, :, 1:-1].set(interior.reshape(shape[0] - 1, shape[1], shape[2] - 2))
        lam = lam.at[0].set(lam[1])
        return MirrorState(radius, lam)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return conservative bounds for normalized solver variables."""
        lower = np.concatenate([np.full(self.radius_size, 0.2), np.full(self.lambda_size, -np.inf)])
        upper = np.concatenate([np.full(self.radius_size, 5.0), np.full(self.lambda_size, np.inf)])
        return lower, upper

    def pullback_gradient(self, gradient: MirrorState) -> np.ndarray:
        """Map a physical state gradient to normalized solver variables."""

        radius = np.asarray(gradient.radius_scale)[self.radius_indices] * self.radius_scale
        if not self.solve_lambda:
            return radius
        shape = self.base.radius_scale.shape
        interior = np.asarray(gradient.lambda_stream)[1:, :, 1:-1].reshape(shape[0] - 1, -1)
        pivot_gradient = interior[:, self.lambda_pivot]
        free = interior[:, self.lambda_free_indices] - (
            pivot_gradient[:, None]
            * self.lambda_interior_weights[self.lambda_free_indices][None, :]
            / self.lambda_interior_weights[self.lambda_pivot]
        )
        return np.concatenate([radius, (free * self.flux_scale).reshape(-1)])


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


def _packed_preconditioner(grid: "MirrorGrid", vectorizer: _MirrorStateVectorizer) -> tuple[Any, np.ndarray]:
    """Build the shared geometry/lambda inverse and mutable block scales."""

    geometry = SeparableMirrorPreconditioner.build(grid)
    stream = None
    if vectorizer.lambda_size:
        stream = SeparableMirrorPreconditioner.build(grid, radial_nodes=grid.ns - 1)
    scales = np.ones(2)

    def apply(vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=float)
        result = np.array(vector, copy=True)
        result[: vectorizer.radius_size] = geometry.apply(vector[: vectorizer.radius_size]) * scales[0]
        if stream is not None:
            result[vectorizer.radius_size :] = (
                stream.apply_gauge_free(
                    vector[vectorizer.radius_size :],
                    free_indices=vectorizer.lambda_free_indices,
                    pivot=vectorizer.lambda_pivot,
                    weights=vectorizer.lambda_interior_weights,
                )
                * scales[1]
            )
        return result

    return apply, scales


def _matrix_free_newton_polish(
    x0: np.ndarray,
    gradient_function: Any,
    objective_function: Any,
    vectorizer: Any,
    preconditioner: tuple[Any, np.ndarray],
    *,
    ftol: float,
    max_steps: int,
    record_step: Any,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> tuple[np.ndarray, int, int, float, bool, str]:
    """Damped Newton-GMRES polish using exact JAX Hessian products."""

    x = np.asarray(x0, dtype=float)
    apply_preconditioner, block_scales = preconditioner

    hessian_vector = jax.jit(lambda point, direction: jax.jvp(gradient_function, (point,), (direction,))[1])
    linear_iterations = 0
    final_linear_residual = np.inf
    damping = 1.0e-8

    for step_index in range(max(0, int(max_steps))):
        gradient = np.asarray(gradient_function(jnp.asarray(x)), dtype=float)
        gradient_max = float(np.max(np.abs(gradient)))
        if gradient_max <= float(ftol):
            return x, step_index, linear_iterations, final_linear_residual, True, "Newton-GMRES converged"

        def matrix_vector(direction: np.ndarray) -> np.ndarray:
            product = np.asarray(hessian_vector(jnp.asarray(x), jnp.asarray(direction)), dtype=float)
            return product + damping * np.asarray(direction)

        # Match each model block to the exact local Hessian scale. This keeps
        # geometry and lambda units from dominating one another in GMRES.
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

        operator = LinearOperator((x.size, x.size), matvec=matrix_vector, dtype=float)
        inverse = LinearOperator((x.size, x.size), matvec=apply_preconditioner, dtype=float)
        iteration_counter = 0

        def count_iteration(_residual: float) -> None:
            nonlocal iteration_counter
            iteration_counter += 1

        direction, info = gmres(
            operator,
            -gradient,
            M=inverse,
            restart=min(50, x.size),
            maxiter=10,
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
    matrix_free_context: tuple[Any, tuple[Any, np.ndarray]] | None,
    start_with_residual_newton: bool = False,
) -> _OptimizationOutcome:
    """Run the common host L-BFGS and residual-Newton solve policy.

    Small closed spline systems can start with residual Newton when their
    geometry initializer is already in a valid local basin. This avoids
    hundreds of relative-energy L-BFGS steps before the same Newton polish;
    larger systems retain the matrix-free policy.
    """

    callback_iterations = 0
    use_matrix_free = x0.size > _HOST_REFERENCE_MAX_SIZE
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

    # Medium systems normally finish on the fast host reference. If that
    # polish stalls just above physical ftol, use the scalable Newton path as
    # a rescue rather than accepting a tolerance-dependent size cliff.
    if float(candidate_variational.maximum) > config.ftol and not use_matrix_free and matrix_free_context is not None:
        run_matrix_free_polish()

    return _OptimizationOutcome(
        vector=final_x,
        iterations=lbfgs_iterations + newton_steps + polish_evaluations,
        optimizer_success=optimizer_success,
        linear_iterations=linear_iterations,
        final_linear_residual=final_linear_residual,
        message=optimizer_message,
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
    solve_lambda: bool = False,
    gradient_tolerance: float = 1.0e-11,
    require_convergence: bool = False,
    device: Any = None,
) -> MirrorSolveResult:
    """Solve a fixed-boundary mirror with host L-BFGS/Newton control.

    Supplying ``solve_lambda=True`` enables the gauge-free stream-function
    variables while preserving their fixed end-cut values. ``device=None``
    applies the measured core device policy; an explicit device or JAX
    platform pin is always honored.
    """

    if device is not _DEVICE_ACTIVE:
        resolved = resolve_device(device, config.resolution)
        if resolved is not None:
            with jax.default_device(resolved):
                return solve_fixed_boundary_cli(
                    initial_state,
                    boundary,
                    grid,
                    config,
                    axial_flux_derivative=axial_flux_derivative,
                    mass_profile=mass_profile,
                    current_derivative=current_derivative,
                    gamma=gamma,
                    solve_lambda=solve_lambda,
                    gradient_tolerance=gradient_tolerance,
                    require_convergence=require_convergence,
                    device=_DEVICE_ACTIVE,
                )

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

    energy_kwargs = {
        "axial_flux_derivative": axial_flux_derivative,
        "mass_profile": mass_profile,
        "current_derivative": current_derivative,
        "gamma": gamma,
    }

    def evaluate_energy(state: MirrorState) -> MirrorEnergy:
        return mirror_energy(state, grid, **energy_kwargs)

    def evaluate_variational(state: MirrorState) -> VariationalResidual:
        return fixed_boundary_variational_residual(state, boundary, grid, **energy_kwargs)

    def evaluate_force(state: MirrorState, energy: MirrorEnergy) -> IsotropicForceResidual:
        del state
        return isotropic_force_residual(energy, grid)

    initial_energy = evaluate_energy(projected_initial)
    energy_scale = max(abs(float(initial_energy.total)), np.finfo(float).tiny)

    def unpack(x: Array) -> MirrorState:
        return vectorizer.unpack(x)

    def objective(x: Array) -> Array:
        return _valid_energy_objective(evaluate_energy(unpack(x)), energy_scale)

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
        maximum = float(max(np.max(np.abs(radius_values)), np.max(np.abs(lambda_values))))
        return VariationalResidual(
            radius_gradient=variational.radius_gradient,
            lambda_gradient=variational.lambda_gradient,
            radius_rms=jnp.asarray(radius_rms),
            lambda_rms=jnp.asarray(lambda_rms),
            maximum=jnp.asarray(maximum),
        )

    def packed_staggered_weak(state: MirrorState) -> VariationalResidual | None:
        weak = isotropic_staggered_weak_residual(
            state,
            boundary,
            grid,
            **energy_kwargs,
        )
        if not solve_lambda:
            return weak
        gradient = isotropic_staggered_fixed_boundary_gradient(
            state,
            boundary,
            grid,
            **energy_kwargs,
        )
        packed = vectorizer.pullback_gradient(gradient) / energy_scale
        radius_values = packed[: vectorizer.radius_size]
        lambda_values = packed[vectorizer.radius_size :]
        return VariationalResidual(
            radius_gradient=weak.radius_gradient,
            lambda_gradient=weak.lambda_gradient,
            radius_rms=jnp.asarray(np.sqrt(np.mean(radius_values**2))),
            lambda_rms=jnp.asarray(np.sqrt(np.mean(lambda_values**2))),
            maximum=jnp.asarray(np.max(np.abs(packed))),
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
    if history[-1][4] <= config.ftol and not bool(initial_energy.geometry.jacobian_sign_changed):
        initial_variational = packed_variational(x0, projected_initial)
        initial_weak_force = packed_staggered_weak(projected_initial)
        projected_initial = regularize_axis_stream_function(
            projected_initial,
            grid,
            axial_flux_derivative,
        )
        result = MirrorSolveResult(
            state=projected_initial,
            energy=initial_energy,
            variational=initial_variational,
            force=evaluate_force(projected_initial, initial_energy),
            staggered_weak_force=initial_weak_force,
            normalized_divergence_rms=normalized_divergence_rms(initial_energy.field, initial_energy.geometry, grid),
            history=jnp.asarray(history),
            iterations=0,
            converged=True,
            optimizer_success=True,
            linear_iterations=0,
            final_linear_residual=0.0,
            message="initial state satisfies physical ftol",
        )
        return result

    optimization = _optimize_fixed_boundary(
        x0,
        lower_bounds,
        upper_bounds,
        objective=objective,
        evaluate=evaluate,
        packed_variational=packed_variational,
        unpack=unpack,
        record=record,
        config=config,
        gradient_tolerance=gradient_tolerance,
        matrix_free_context=(vectorizer, _packed_preconditioner(grid, vectorizer)),
    )
    final_x = optimization.vector

    final_state = regularize_axis_stream_function(
        unpack(jnp.asarray(final_x)),
        grid,
        axial_flux_derivative,
    )
    final_energy = evaluate_energy(final_state)
    final_variational = packed_variational(final_x, final_state)
    final_force = evaluate_force(final_state, final_energy)
    final_weak_force = packed_staggered_weak(final_state)
    record(optimization.iterations, final_x)
    converged = bool(
        float(final_variational.maximum) <= config.ftol and not bool(final_energy.geometry.jacobian_sign_changed)
    )
    message = optimization.message
    if not converged:
        message += f"; variational force={float(final_variational.maximum):.3e}"
    result = MirrorSolveResult(
        state=final_state,
        energy=final_energy,
        variational=final_variational,
        force=final_force,
        staggered_weak_force=final_weak_force,
        normalized_divergence_rms=normalized_divergence_rms(final_energy.field, final_energy.geometry, grid),
        history=jnp.asarray(history),
        iterations=optimization.iterations,
        converged=converged,
        optimizer_success=optimization.optimizer_success,
        linear_iterations=optimization.linear_iterations,
        final_linear_residual=optimization.final_linear_residual,
        message=message,
    )
    if require_convergence and not converged:
        raise MirrorConvergenceError(result)
    return result


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .basis import MirrorGrid
