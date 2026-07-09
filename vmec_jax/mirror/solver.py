"""Host-controlled fixed-boundary mirror reference solver.

This M2 solver is the non-differentiable CLI/reference lane.  It minimizes the
same JAX energy used by the future traced lane, but lets SciPy control L-BFGS
line searches and early exits.  Crucially, SciPy success is not equilibrium
success: the returned state is converged only when the independently computed
physical tensor-force residual meets ``MirrorConfig.ftol``.

The M4/M9 lanes will add the separably preconditioned Newton-Krylov and
implicit-differentiation wrappers without changing the energy or force APIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares, minimize

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
    message: str


jax.tree_util.register_dataclass(
    MirrorSolveResult,
    data_fields=["state", "energy", "variational", "force", "history"],
    meta_fields=["iterations", "converged", "optimizer_success", "message"],
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
            message="initial state satisfies physical ftol",
        )
        return result

    callback_iterations = 0

    def callback(x: np.ndarray) -> None:
        nonlocal callback_iterations
        callback_iterations += 1
        record(callback_iterations, x)

    optimization = minimize(
        fun=lambda x: evaluate(x)[0],
        x0=x0,
        jac=lambda x: evaluate(x)[1],
        method="L-BFGS-B",
        bounds=[(0.2, 5.0)] * x0.size,
        callback=callback,
        options={
            "maxiter": int(config.max_iterations),
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

    # L-BFGS commonly reaches machine-level relative energy change before the
    # physical force is small.  An exact dense residual-Newton polish is a
    # reliable M2 reference for modest systems; M4 replaces this size-limited
    # path with matrix-free Newton-GMRES and the separable preconditioner.
    candidate_energy = mirror_energy(unpack(jnp.asarray(final_x)), grid, **energy_kwargs)
    candidate_variational = fixed_boundary_variational_residual(
        unpack(jnp.asarray(final_x)), boundary, grid, **energy_kwargs
    )
    if float(candidate_variational.maximum) > config.ftol and final_x.size <= 512:
        gradient_function = jax.jit(jax.grad(objective))
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
    record(callback_iterations + polish_evaluations, final_x)
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
        iterations=int(optimization.nit) + polish_evaluations,
        converged=converged,
        optimizer_success=optimizer_success,
        message=message,
    )
    if require_convergence and not converged:
        raise MirrorConvergenceError(result)
    return result


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .basis import MirrorGrid
