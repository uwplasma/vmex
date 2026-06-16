"""Small projected optimizers for fixed-boundary mirror solves."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...core.boundary import MirrorBoundary
from ...core.grids import MirrorGrid
from ...core.profiles import IPrimeProfile, PressureProfile, PsiPrimeProfile
from ...core.state import MirrorStateAxisym
from ...kernels.constraints import project_axisym_state
from ...kernels.forces import axisym_energy_value_and_gradient, axisym_projected_energy_residual


@dataclass(frozen=True)
class OptimizerOptions:
    """Numerical options for fixed-boundary optimizer stages."""

    optimizer: str = "gradient_descent"
    maxiter: int = 50
    tolerance: float = 1.0e-8
    step_size: float = 1.0e-3
    min_step_size: float = 1.0e-12
    line_search_steps: int = 16
    mu0: float = 4.0e-7 * np.pi


@dataclass(frozen=True)
class OptimizerStep:
    """Accepted optimizer step payload."""

    state: MirrorStateAxisym
    energy: float
    residual_norm: float
    step_size: float
    accepted: bool


@dataclass(frozen=True)
class OptimizerRun:
    """Multi-step optimizer payload."""

    state: MirrorStateAxisym
    steps: tuple[OptimizerStep, ...]


def _positive_radius(state: MirrorStateAxisym, floor: float = 1.0e-10) -> bool:
    return bool(np.all(np.asarray(state.a) > floor))


def axisym_reduced_a_mask(grid: MirrorGrid) -> np.ndarray:
    """Return the independent ``a`` nodes for fixed-boundary axisymmetric solves."""
    mask = np.zeros((grid.ns, grid.nxi), dtype=bool)
    if grid.ns > 2 and grid.nxi > 2:
        mask[1:-1, 1:-1] = True
    return mask


def pack_axisym_reduced_state(state: MirrorStateAxisym, grid: MirrorGrid, boundary: MirrorBoundary) -> np.ndarray:
    """Pack independent ``a`` nodes and gauge-fixed ``lambda`` nodes."""
    projected = project_axisym_state(state, grid, boundary)
    a_values = projected.a[axisym_reduced_a_mask(grid)]
    lam_values = np.asarray(projected.lam[:, :-1], dtype=float).ravel()
    return np.concatenate([a_values, lam_values])


def unpack_axisym_reduced_state(vector, grid: MirrorGrid, boundary: MirrorBoundary) -> MirrorStateAxisym:
    """Reconstruct a projected axisymmetric state from reduced coordinates."""
    vector = np.asarray(vector, dtype=float)
    mask = axisym_reduced_a_mask(grid)
    num_a = int(np.count_nonzero(mask))
    expected = num_a + grid.ns * (grid.nxi - 1)
    if vector.size != expected:
        raise ValueError(f"reduced vector has size {vector.size}, expected {expected}")

    boundary_radius = boundary.radius_on_grid(grid)
    a = np.broadcast_to(boundary_radius[None, :], (grid.ns, grid.nxi)).copy()
    a[mask] = vector[:num_a]

    lam = np.zeros((grid.ns, grid.nxi), dtype=float)
    lam[:, :-1] = vector[num_a:].reshape(grid.ns, grid.nxi - 1)
    lam[:, -1] = -np.einsum("j,ij->i", grid.w_xi[:-1], lam[:, :-1]) / float(grid.w_xi[-1])
    return project_axisym_state(MirrorStateAxisym(a=a, lam=lam), grid, boundary)


def _pack_axisym_reduced_gradient(gradient, grid: MirrorGrid) -> np.ndarray:
    mask = axisym_reduced_a_mask(grid)
    grad_a = np.asarray(gradient.grad_a, dtype=float).copy()
    if grid.ns > 2:
        grad_a[1, :] += grad_a[0, :]
    a_values = grad_a[mask]

    grad_lam = np.asarray(gradient.grad_lam, dtype=float)
    lam_values = grad_lam[:, :-1] - (grid.w_xi[:-1] / grid.w_xi[-1])[None, :] * grad_lam[:, -1:]
    return np.concatenate([a_values, lam_values.ravel()])


def reduced_axisym_energy_and_gradient(
    vector,
    grid: MirrorGrid,
    boundary: MirrorBoundary,
    *,
    psi_prime: PsiPrimeProfile,
    i_prime: IPrimeProfile,
    pressure: PressureProfile,
    mu0: float = 4.0e-7 * np.pi,
) -> tuple[float, np.ndarray]:
    """Return energy and exact reduced-coordinate gradient."""
    state = unpack_axisym_reduced_state(vector, grid, boundary)
    gradient = axisym_energy_value_and_gradient(
        state,
        grid,
        psi_prime=psi_prime,
        i_prime=i_prime,
        pressure=pressure,
        mu0=mu0,
    )
    return gradient.energy, _pack_axisym_reduced_gradient(gradient, grid)


def projected_gradient_step(
    state: MirrorStateAxisym,
    grid: MirrorGrid,
    boundary: MirrorBoundary,
    *,
    psi_prime: PsiPrimeProfile,
    i_prime: IPrimeProfile,
    pressure: PressureProfile,
    options: OptimizerOptions,
) -> OptimizerStep:
    """Take one projected gradient step with backtracking line search."""
    residual = axisym_projected_energy_residual(
        state,
        grid,
        psi_prime=psi_prime,
        i_prime=i_prime,
        pressure=pressure,
        mu0=options.mu0,
    )
    if residual.norm <= options.tolerance:
        return OptimizerStep(
            state=state,
            energy=residual.energy,
            residual_norm=residual.norm,
            step_size=0.0,
            accepted=True,
        )

    step = float(options.step_size)
    for _ in range(int(options.line_search_steps)):
        trial = MirrorStateAxisym(
            a=state.a - step * residual.projected_a,
            lam=state.lam - step * residual.projected_lam,
        )
        trial = project_axisym_state(trial, grid, boundary)
        if _positive_radius(trial):
            trial_residual = axisym_projected_energy_residual(
                trial,
                grid,
                psi_prime=psi_prime,
                i_prime=i_prime,
                pressure=pressure,
                mu0=options.mu0,
            )
            if np.isfinite(trial_residual.energy) and trial_residual.energy <= residual.energy:
                return OptimizerStep(
                    state=trial,
                    energy=trial_residual.energy,
                    residual_norm=trial_residual.norm,
                    step_size=step,
                    accepted=True,
                )
        step *= 0.5
        if step < options.min_step_size:
            break

    return OptimizerStep(
        state=state,
        energy=residual.energy,
        residual_norm=residual.norm,
        step_size=0.0,
        accepted=False,
    )


def _reduced_step_payload(
    vector,
    previous_vector,
    grid: MirrorGrid,
    boundary: MirrorBoundary,
    *,
    psi_prime: PsiPrimeProfile,
    i_prime: IPrimeProfile,
    pressure: PressureProfile,
    options: OptimizerOptions,
    accepted: bool,
) -> OptimizerStep:
    state = unpack_axisym_reduced_state(vector, grid, boundary)
    residual = axisym_projected_energy_residual(
        state,
        grid,
        psi_prime=psi_prime,
        i_prime=i_prime,
        pressure=pressure,
        mu0=options.mu0,
    )
    step_size = float(np.linalg.norm(np.asarray(vector, dtype=float) - np.asarray(previous_vector, dtype=float)))
    return OptimizerStep(
        state=state,
        energy=residual.energy,
        residual_norm=residual.norm,
        step_size=step_size,
        accepted=accepted,
    )


def _rejected_lbfgs_step(initial_state: MirrorStateAxisym, initial_residual) -> OptimizerStep:
    return OptimizerStep(
        state=initial_state,
        energy=initial_residual.energy,
        residual_norm=initial_residual.norm,
        step_size=0.0,
        accepted=False,
    )


def _lbfgs_options(options: OptimizerOptions) -> dict[str, float | int]:
    return {
        "maxiter": int(options.maxiter),
        "gtol": float(options.tolerance),
        "maxls": int(options.line_search_steps),
        "ftol": float(max(options.min_step_size, np.finfo(float).eps)),
    }


def projected_lbfgs_solve(
    state: MirrorStateAxisym,
    grid: MirrorGrid,
    boundary: MirrorBoundary,
    *,
    psi_prime: PsiPrimeProfile,
    i_prime: IPrimeProfile,
    pressure: PressureProfile,
    options: OptimizerOptions,
) -> OptimizerRun:
    """Run a reduced-coordinate L-BFGS-B fixed-boundary solve."""
    try:
        from scipy.optimize import minimize
    except Exception as exc:  # pragma: no cover
        raise ImportError("mirror optimizer='lbfgs' requires scipy.optimize.minimize") from exc

    initial_state = project_axisym_state(state, grid, boundary)
    x0 = pack_axisym_reduced_state(initial_state, grid, boundary)
    initial_residual = axisym_projected_energy_residual(
        initial_state,
        grid,
        psi_prime=psi_prime,
        i_prime=i_prime,
        pressure=pressure,
        mu0=options.mu0,
    )
    if initial_residual.norm <= options.tolerance:
        return OptimizerRun(state=initial_state, steps=())

    steps: list[OptimizerStep] = []
    previous_x = x0.copy()

    def objective(vector):
        value, gradient = reduced_axisym_energy_and_gradient(
            vector,
            grid,
            boundary,
            psi_prime=psi_prime,
            i_prime=i_prime,
            pressure=pressure,
            mu0=options.mu0,
        )
        return value, gradient

    def record_step(vector, *, accepted: bool = True) -> OptimizerStep:
        nonlocal previous_x
        step = _reduced_step_payload(
            vector,
            previous_x,
            grid,
            boundary,
            psi_prime=psi_prime,
            i_prime=i_prime,
            pressure=pressure,
            options=options,
            accepted=accepted,
        )
        previous_x = np.asarray(vector, dtype=float).copy()
        return step

    def callback(vector):
        steps.append(record_step(vector))

    result = minimize(
        objective,
        x0,
        jac=True,
        method="L-BFGS-B",
        callback=callback,
        options=_lbfgs_options(options),
    )
    final_step = record_step(np.asarray(result.x, dtype=float), accepted=bool(np.isfinite(result.fun)))
    if not steps or final_step.step_size > 0.0 or abs(final_step.energy - steps[-1].energy) > 1.0e-14:
        steps.append(final_step)

    final = steps[-1]
    improved = np.isfinite(final.energy) and final.energy <= initial_residual.energy
    if not improved or not _positive_radius(final.state):
        return OptimizerRun(state=initial_state, steps=(_rejected_lbfgs_step(initial_state, initial_residual),))
    return OptimizerRun(state=final.state, steps=tuple(steps))
