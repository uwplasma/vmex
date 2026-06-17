"""Small projected optimizers for fixed-boundary mirror solves."""

from __future__ import annotations

import numpy as np

from vmec_jax._compat import jax, jnp

from ...core.boundary import MirrorBoundary
from ...core.grids import MirrorGrid
from ...core.profiles import IPrimeProfile, PressureProfile, PsiPrimeProfile
from ...core.state import MirrorState3D, MirrorStateAxisym
from ...kernels.constraints import project_axisym_state, project_state_3d
from ...kernels.forces import (
    axisym_projected_energy_residual,
    projected_energy_residual_3d,
)
from ...kernels.geometry import evaluate_axisym_geometry, evaluate_geometry_3d
from .preconditioners import (
    _residual_linear_maxiter_policy_key,
    _residual_linear_solver_key,
    _residual_preconditioner_key,
    _validate_smoothing_alpha,
    axisym_reduced_residual_preconditioner,
    axisym_residual_linear_maxiter,
)
from .reduced import (
    _axisym_reduced_energy_jax,
    _sanitize_scale,
    _scaling_key,
    _scaled_bounds,
    axisym_reduced_a_mask,
    axisym_reduced_bounds,
    axisym_reduced_coordinate_scale,
    pack_axisym_reduced_state,
    pack_axisym_reduced_gradient_components,
    pack_reduced_state_3d,
    reduced_3d_energy_and_gradient,
    reduced_a_mask_3d,
    reduced_axisym_energy_and_gradient,
    reduced_bounds_3d,
    reduced_coordinate_scale_3d,
    scale_reduced_bounds,
    unpack_axisym_reduced_state,
    unpack_reduced_state_3d,
)
from .types import OptimizerOptions, OptimizerRun, OptimizerStep, _CandidateDiagnostics

__all__ = [
    "OptimizerOptions",
    "OptimizerRun",
    "OptimizerStep",
    "_axisym_reduced_energy_jax",
    "_lbfgs_options",
    "_residual_linear_maxiter_policy_key",
    "_residual_linear_solver_key",
    "_residual_preconditioner_key",
    "_sanitize_scale",
    "_scaled_bounds",
    "_scaling_key",
    "_validate_smoothing_alpha",
    "axisym_reduced_a_mask",
    "axisym_reduced_bounds",
    "axisym_reduced_coordinate_scale",
    "axisym_reduced_residual_preconditioner",
    "axisym_residual_linear_maxiter",
    "pack_axisym_reduced_gradient_components",
    "pack_axisym_reduced_state",
    "pack_reduced_state_3d",
    "projected_gradient_step",
    "projected_gradient_step_3d",
    "projected_lbfgs_solve",
    "projected_lbfgs_solve_3d",
    "projected_residual_newton_solve",
    "reduced_3d_energy_and_gradient",
    "reduced_a_mask_3d",
    "reduced_axisym_energy_and_gradient",
    "reduced_bounds_3d",
    "reduced_coordinate_scale_3d",
    "scale_reduced_bounds",
    "unpack_axisym_reduced_state",
    "unpack_reduced_state_3d",
]


def _positive_radius(state: MirrorStateAxisym | MirrorState3D, floor: float = 1.0e-10) -> bool:
    return bool(np.all(np.asarray(state.a) > floor))


def _positive_jacobian(state: MirrorStateAxisym | MirrorState3D, grid: MirrorGrid, floor: float = 1.0e-10) -> bool:
    geometry = (
        evaluate_geometry_3d(state, grid) if np.asarray(state.a).ndim == 3 else evaluate_axisym_geometry(state, grid)
    )
    return bool(np.all(np.asarray(geometry.sqrtg) > floor))


def _admissible_state(state: MirrorStateAxisym | MirrorState3D, grid: MirrorGrid) -> bool:
    return _positive_radius(state) and _positive_jacobian(state, grid)


def _candidate_diagnostics(
    step: OptimizerStep,
    grid: MirrorGrid,
    *,
    initial_energy: float,
    floor: float = 1.0e-10,
) -> _CandidateDiagnostics:
    energy = float(step.energy)
    finite_energy = bool(np.isfinite(energy))
    energy_improved = bool(finite_energy and energy <= float(initial_energy))
    min_a = float(np.min(np.asarray(step.state.a, dtype=float)))
    geometry = (
        evaluate_geometry_3d(step.state, grid)
        if np.asarray(step.state.a).ndim == 3
        else evaluate_axisym_geometry(step.state, grid)
    )
    min_sqrtg = float(np.min(np.asarray(geometry.sqrtg, dtype=float)))
    positive_radius = bool(np.isfinite(min_a) and min_a > floor)
    positive_jacobian = bool(np.isfinite(min_sqrtg) and min_sqrtg > floor)

    failures: list[str] = []
    if not finite_energy:
        failures.append("nonfinite_energy")
    elif not energy_improved:
        failures.append("energy_increase")
    if not positive_radius:
        failures.append("nonpositive_radius")
    if not positive_jacobian:
        failures.append("nonpositive_jacobian")
    return _CandidateDiagnostics(
        accepted=not failures,
        reason="accepted" if not failures else ",".join(failures),
        min_a=min_a,
        min_sqrtg=min_sqrtg,
        energy_improved=energy_improved,
        positive_radius=positive_radius,
        positive_jacobian=positive_jacobian,
    )


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
        if _admissible_state(trial, grid):
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


def projected_gradient_step_3d(
    state: MirrorState3D,
    grid: MirrorGrid,
    boundary: MirrorBoundary,
    *,
    psi_prime: PsiPrimeProfile,
    i_prime: IPrimeProfile,
    pressure: PressureProfile,
    options: OptimizerOptions,
) -> OptimizerStep:
    """Take one projected 3D gradient step with backtracking line search."""
    residual = projected_energy_residual_3d(
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
        trial = MirrorState3D(
            a=state.a - step * residual.projected_a,
            lam=state.lam - step * residual.projected_lam,
        )
        trial = project_state_3d(trial, grid, boundary)
        if _admissible_state(trial, grid):
            trial_residual = projected_energy_residual_3d(
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


def _reduced_step_payload_3d(
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
    state = unpack_reduced_state_3d(vector, grid, boundary)
    residual = projected_energy_residual_3d(
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


def _rejected_lbfgs_step_3d(initial_state: MirrorState3D, initial_residual) -> OptimizerStep:
    return OptimizerStep(
        state=initial_state,
        energy=initial_residual.energy,
        residual_norm=initial_residual.norm,
        step_size=0.0,
        accepted=False,
    )


def _lbfgs_options(options: OptimizerOptions) -> dict[str, float | int]:
    ftol = float(options.ftol) if options.ftol is not None else float(max(options.min_step_size, np.finfo(float).eps))
    return {
        "maxiter": int(options.maxiter),
        "gtol": float(options.tolerance),
        "maxls": int(options.line_search_steps),
        "ftol": ftol,
    }


def _optimizer_run_from_result(
    *,
    state: MirrorStateAxisym | MirrorState3D,
    steps: tuple[OptimizerStep, ...],
    result,
    accepted: bool = True,
    candidate_step: OptimizerStep | None = None,
    candidate_diagnostics: _CandidateDiagnostics | None = None,
) -> OptimizerRun:
    return OptimizerRun(
        state=state,
        steps=steps,
        success=bool(getattr(result, "success", False)),
        status=int(getattr(result, "status", 0)),
        message=str(getattr(result, "message", "")),
        nit=int(getattr(result, "nit", len(steps))),
        nfev=int(getattr(result, "nfev", 0)),
        njev=int(getattr(result, "njev", 0)),
        accepted=bool(accepted),
        rejection_reason="" if candidate_diagnostics is None else str(candidate_diagnostics.reason),
        candidate_energy_total=None if candidate_step is None else float(candidate_step.energy),
        candidate_residual_norm=None if candidate_step is None else float(candidate_step.residual_norm),
        candidate_min_a=None if candidate_diagnostics is None else float(candidate_diagnostics.min_a),
        candidate_min_sqrtg=None if candidate_diagnostics is None else float(candidate_diagnostics.min_sqrtg),
        candidate_energy_improved=None
        if candidate_diagnostics is None
        else bool(candidate_diagnostics.energy_improved),
        candidate_positive_radius=None
        if candidate_diagnostics is None
        else bool(candidate_diagnostics.positive_radius),
        candidate_positive_jacobian=None
        if candidate_diagnostics is None
        else bool(candidate_diagnostics.positive_jacobian),
        residual_linear_maxiter_policy=str(getattr(result, "residual_linear_maxiter_policy", "")),
        residual_linear_solver=str(getattr(result, "residual_linear_solver", "")),
        residual_linear_maxiter_effective_max=getattr(result, "residual_linear_maxiter_effective_max", None),
        residual_linear_maxiter_effective_last=getattr(result, "residual_linear_maxiter_effective_last", None),
    )


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
        return OptimizerRun(
            state=initial_state,
            steps=(),
            success=True,
            status=0,
            message="initial projected residual is below tolerance",
            nit=0,
            nfev=0,
            njev=0,
            accepted=True,
        )

    steps: list[OptimizerStep] = []
    previous_x = x0.copy()
    x_scale = axisym_reduced_coordinate_scale(
        initial_state,
        grid,
        boundary,
        mode=options.reduced_coordinate_scaling,
    )
    y0 = x0 / x_scale

    def _x_from_y(vector_y) -> np.ndarray:
        return np.asarray(vector_y, dtype=float) * x_scale

    def objective(vector_y):
        vector = _x_from_y(vector_y)
        value, gradient = reduced_axisym_energy_and_gradient(
            vector,
            grid,
            boundary,
            psi_prime=psi_prime,
            i_prime=i_prime,
            pressure=pressure,
            mu0=options.mu0,
        )
        return value, np.asarray(gradient, dtype=float) * x_scale

    def record_step(vector_y, *, accepted: bool = True) -> OptimizerStep:
        nonlocal previous_x
        vector = _x_from_y(vector_y)
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

    def callback(vector_y):
        steps.append(record_step(vector_y))

    result = minimize(
        objective,
        y0,
        jac=True,
        method="L-BFGS-B",
        bounds=_scaled_bounds(axisym_reduced_bounds(grid), x_scale),
        callback=callback,
        options=_lbfgs_options(options),
    )
    final_step = record_step(np.asarray(result.x, dtype=float), accepted=bool(np.isfinite(result.fun)))
    if not steps or final_step.step_size > 0.0 or abs(final_step.energy - steps[-1].energy) > 1.0e-14:
        steps.append(final_step)

    final = steps[-1]
    candidate_diagnostics = _candidate_diagnostics(final, grid, initial_energy=initial_residual.energy)
    if not candidate_diagnostics.accepted:
        rejected_steps = (_rejected_lbfgs_step(initial_state, initial_residual),)
        return _optimizer_run_from_result(
            state=initial_state,
            steps=rejected_steps,
            result=result,
            accepted=False,
            candidate_step=final,
            candidate_diagnostics=candidate_diagnostics,
        )
    return _optimizer_run_from_result(
        state=final.state,
        steps=tuple(steps),
        result=result,
        accepted=True,
        candidate_step=final,
        candidate_diagnostics=candidate_diagnostics,
    )


def projected_residual_newton_solve(
    state: MirrorStateAxisym,
    grid: MirrorGrid,
    boundary: MirrorBoundary,
    *,
    psi_prime: PsiPrimeProfile,
    i_prime: IPrimeProfile,
    pressure: PressureProfile,
    options: OptimizerOptions,
) -> OptimizerRun:
    """Run a matrix-free damped Newton solve on reduced fixed-boundary residuals."""
    if jax is None:
        raise RuntimeError("optimizer='residual_newton' requires JAX")
    try:
        from scipy.optimize import OptimizeResult
        from scipy.sparse.linalg import LinearOperator, lsmr
    except Exception as exc:  # pragma: no cover
        raise ImportError("optimizer='residual_newton' requires scipy.sparse.linalg.lsmr") from exc

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
        return OptimizerRun(
            state=initial_state,
            steps=(),
            success=True,
            status=0,
            message="initial projected residual is below tolerance",
            nit=0,
            nfev=0,
            njev=0,
            accepted=True,
        )

    x_scale = axisym_reduced_coordinate_scale(
        initial_state,
        grid,
        boundary,
        mode=options.reduced_coordinate_scaling,
    )
    y = x0 / x_scale
    scale_jax = jnp.asarray(x_scale, dtype=jnp.asarray(x0).dtype)
    preconditioner_kind = _residual_preconditioner_key(options.residual_preconditioner)
    linear_solver_kind = _residual_linear_solver_key(options.residual_linear_solver)

    def objective_x(vector):
        return _axisym_reduced_energy_jax(
            vector,
            grid,
            boundary,
            psi_prime=psi_prime,
            i_prime=i_prime,
            pressure=pressure,
            mu0=options.mu0,
        )

    grad_fun = jax.grad(objective_x)
    hessian_fun = jax.jacfwd(grad_fun) if linear_solver_kind == "dense_lstsq" else None

    def x_from_y(vector_y) -> np.ndarray:
        return np.asarray(vector_y, dtype=float) * x_scale

    def reduced_gradient_x(vector: np.ndarray) -> np.ndarray:
        return np.asarray(grad_fun(jnp.asarray(vector, dtype=scale_jax.dtype)), dtype=float)

    def step_payload(vector: np.ndarray, previous_vector: np.ndarray, *, accepted: bool = True) -> OptimizerStep:
        return _reduced_step_payload(
            vector,
            previous_vector,
            grid,
            boundary,
            psi_prime=psi_prime,
            i_prime=i_prime,
            pressure=pressure,
            options=options,
            accepted=accepted,
        )

    steps: list[OptimizerStep] = []
    current_x = x0.copy()
    current_step = step_payload(current_x, current_x)
    nfev = 0
    njev = 0
    success = False
    status = 0
    message = "maximum iterations reached"
    ftol = float(options.ftol) if options.ftol is not None else float(options.min_step_size)
    linear_maxiter_history: list[int] = []

    for _iteration in range(1, int(options.maxiter) + 1):
        grad_x = reduced_gradient_x(current_x)
        nfev += 1
        reduced_norm = float(np.linalg.norm(grad_x))
        if current_step.residual_norm <= options.tolerance or reduced_norm <= options.tolerance:
            success = True
            status = 1
            message = "`gtol` termination condition is satisfied."
            break

        x_jax = jnp.asarray(current_x, dtype=scale_jax.dtype)

        def matvec_y(vector_y):
            direction_x = scale_jax * jnp.asarray(vector_y, dtype=scale_jax.dtype)
            _, hvp = jax.jvp(grad_fun, (x_jax,), (direction_x,))
            return np.asarray(hvp * scale_jax, dtype=float)

        def precondition_y(vector_y):
            return axisym_reduced_residual_preconditioner(
                vector_y,
                grid,
                kind=preconditioner_kind,
                radial_alpha=options.residual_radial_alpha,
                lambda_alpha=options.residual_lambda_alpha,
                xi_alpha=options.residual_xi_alpha,
            )

        rhs = -grad_x * x_scale
        size = int(y.size)
        if linear_solver_kind == "dense_lstsq":
            assert hessian_fun is not None
            hessian_x = np.asarray(hessian_fun(x_jax), dtype=float)
            njev += 1
            jacobian_y = hessian_x * x_scale[:, None] * x_scale[None, :]
            if preconditioner_kind != "none":
                basis = np.eye(size)
                preconditioner_matrix = np.column_stack([precondition_y(basis[:, index]) for index in range(size)])
                jacobian_y = jacobian_y @ preconditioner_matrix
            try:
                step_y_raw, *_ = np.linalg.lstsq(jacobian_y, rhs, rcond=1.0e-12)
            except np.linalg.LinAlgError:
                message = "dense reduced-Hessian solve failed"
                break
            step_y = np.asarray(step_y_raw, dtype=float)
            if preconditioner_kind != "none":
                step_y = np.asarray(preconditioner_matrix @ step_y, dtype=float)
        else:
            if preconditioner_kind == "none":
                operator = LinearOperator((size, size), matvec=matvec_y, rmatvec=matvec_y, dtype=float)
            else:

                def matvec_preconditioned(vector_z):
                    return matvec_y(precondition_y(vector_z))

                def rmatvec_preconditioned(vector_y):
                    return precondition_y(matvec_y(vector_y))

                operator = LinearOperator(
                    (size, size),
                    matvec=matvec_preconditioned,
                    rmatvec=rmatvec_preconditioned,
                    dtype=float,
                )
            linear_maxiter = axisym_residual_linear_maxiter(
                options,
                grid,
                vector_size=size,
                residual_norm=current_step.residual_norm,
            )
            linear_maxiter_history.append(linear_maxiter)
            linear_result = lsmr(
                operator,
                rhs,
                atol=min(1.0e-10, max(options.tolerance, np.finfo(float).eps)),
                btol=min(1.0e-10, max(options.tolerance, np.finfo(float).eps)),
                maxiter=linear_maxiter,
            )
            njev += int(linear_result[2])
            step_y_raw = np.asarray(linear_result[0], dtype=float)
            step_y = step_y_raw if preconditioner_kind == "none" else precondition_y(step_y_raw)
        if not np.all(np.isfinite(step_y)):
            message = "non-finite reduced-Newton step"
            break

        step_x_norm = float(np.linalg.norm(step_y * x_scale))
        if step_x_norm <= ftol * max(1.0, float(np.linalg.norm(current_x))):
            success = True
            status = 3
            message = "`xtol` termination condition is satisfied."
            break

        accepted = False
        alpha = 1.0
        for _ in range(int(options.line_search_steps)):
            trial_y = y + alpha * step_y
            trial_x = x_from_y(trial_y)
            trial_state = unpack_axisym_reduced_state(trial_x, grid, boundary)
            if _admissible_state(trial_state, grid):
                trial_step = step_payload(trial_x, current_x)
                energy_ok = np.isfinite(trial_step.energy) and trial_step.energy <= current_step.energy + 1.0e-12
                residual_ok = np.isfinite(trial_step.residual_norm) and (
                    trial_step.residual_norm < current_step.residual_norm
                )
                if energy_ok and residual_ok:
                    y = trial_y
                    current_x = trial_x
                    current_step = trial_step
                    steps.append(trial_step)
                    accepted = True
                    if trial_step.residual_norm <= options.tolerance:
                        success = True
                        status = 1
                        message = "`gtol` termination condition is satisfied."
                    break
            alpha *= 0.5
        if success:
            break
        if not accepted:
            message = "line search failed to reduce the projected residual"
            rejected = OptimizerStep(
                state=current_step.state,
                energy=current_step.energy,
                residual_norm=current_step.residual_norm,
                step_size=0.0,
                accepted=False,
            )
            steps.append(rejected)
            break

    final_step = current_step if not steps else steps[-1]
    final_state = final_step.state if final_step.accepted else current_step.state
    result = OptimizeResult(
        success=success,
        status=status,
        message=message,
        nit=sum(1 for step in steps if step.accepted),
        nfev=nfev,
        njev=njev,
    )
    result.residual_linear_maxiter_policy = _residual_linear_maxiter_policy_key(options.residual_linear_maxiter_policy)
    result.residual_linear_solver = linear_solver_kind
    result.residual_linear_maxiter_effective_max = int(max(linear_maxiter_history)) if linear_maxiter_history else None
    result.residual_linear_maxiter_effective_last = int(linear_maxiter_history[-1]) if linear_maxiter_history else None
    candidate_diagnostics = _candidate_diagnostics(current_step, grid, initial_energy=initial_residual.energy)
    accepted_run = bool(steps and steps[-1].accepted and candidate_diagnostics.accepted)
    if not accepted_run:
        rejected_steps = tuple(steps) if steps else (_rejected_lbfgs_step(initial_state, initial_residual),)
        return _optimizer_run_from_result(
            state=final_state,
            steps=rejected_steps,
            result=result,
            accepted=False,
            candidate_step=current_step,
            candidate_diagnostics=candidate_diagnostics,
        )
    return _optimizer_run_from_result(
        state=current_step.state,
        steps=tuple(steps),
        result=result,
        accepted=True,
        candidate_step=current_step,
        candidate_diagnostics=candidate_diagnostics,
    )


def projected_lbfgs_solve_3d(
    state: MirrorState3D,
    grid: MirrorGrid,
    boundary: MirrorBoundary,
    *,
    psi_prime: PsiPrimeProfile,
    i_prime: IPrimeProfile,
    pressure: PressureProfile,
    options: OptimizerOptions,
) -> OptimizerRun:
    """Run a reduced-coordinate L-BFGS-B fixed-boundary solve for 3D states."""
    try:
        from scipy.optimize import minimize
    except Exception as exc:  # pragma: no cover
        raise ImportError("mirror optimizer='lbfgs' requires scipy.optimize.minimize") from exc

    initial_state = project_state_3d(state, grid, boundary)
    x0 = pack_reduced_state_3d(initial_state, grid, boundary)
    initial_residual = projected_energy_residual_3d(
        initial_state,
        grid,
        psi_prime=psi_prime,
        i_prime=i_prime,
        pressure=pressure,
        mu0=options.mu0,
    )
    if initial_residual.norm <= options.tolerance:
        return OptimizerRun(
            state=initial_state,
            steps=(),
            success=True,
            status=0,
            message="initial projected residual is below tolerance",
            nit=0,
            nfev=0,
            njev=0,
            accepted=True,
        )

    steps: list[OptimizerStep] = []
    previous_x = x0.copy()
    x_scale = reduced_coordinate_scale_3d(
        initial_state,
        grid,
        boundary,
        mode=options.reduced_coordinate_scaling,
    )
    y0 = x0 / x_scale

    def _x_from_y(vector_y) -> np.ndarray:
        return np.asarray(vector_y, dtype=float) * x_scale

    def objective(vector_y):
        vector = _x_from_y(vector_y)
        value, gradient = reduced_3d_energy_and_gradient(
            vector,
            grid,
            boundary,
            psi_prime=psi_prime,
            i_prime=i_prime,
            pressure=pressure,
            mu0=options.mu0,
        )
        return value, np.asarray(gradient, dtype=float) * x_scale

    def record_step(vector_y, *, accepted: bool = True) -> OptimizerStep:
        nonlocal previous_x
        vector = _x_from_y(vector_y)
        step = _reduced_step_payload_3d(
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

    def callback(vector_y):
        steps.append(record_step(vector_y))

    result = minimize(
        objective,
        y0,
        jac=True,
        method="L-BFGS-B",
        bounds=_scaled_bounds(reduced_bounds_3d(grid), x_scale),
        callback=callback,
        options=_lbfgs_options(options),
    )
    final_step = record_step(np.asarray(result.x, dtype=float), accepted=bool(np.isfinite(result.fun)))
    if not steps or final_step.step_size > 0.0 or abs(final_step.energy - steps[-1].energy) > 1.0e-14:
        steps.append(final_step)

    final = steps[-1]
    candidate_diagnostics = _candidate_diagnostics(final, grid, initial_energy=initial_residual.energy)
    if not candidate_diagnostics.accepted:
        rejected_steps = (_rejected_lbfgs_step_3d(initial_state, initial_residual),)
        return _optimizer_run_from_result(
            state=initial_state,
            steps=rejected_steps,
            result=result,
            accepted=False,
            candidate_step=final,
            candidate_diagnostics=candidate_diagnostics,
        )
    return _optimizer_run_from_result(
        state=final.state,
        steps=tuple(steps),
        result=result,
        accepted=True,
        candidate_step=final,
        candidate_diagnostics=candidate_diagnostics,
    )
