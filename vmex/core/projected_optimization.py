"""Projected objective descent with target restoration and bounded backtracking.

Geometry is supplied through a motion-bound callable. This module knows no
coil count, Fourier order, equilibrium implementation, or output directory.
"""

from dataclasses import dataclass
import numpy as np
from scipy.optimize import OptimizeResult

from .errors import TrialRejected as TrialRejected


@dataclass(frozen=True)
class ProjectedOptions:
    """Numerical acceptance, convergence, and physical motion limits."""

    constraint_relative_tolerance: float = 0.01
    maximum_coil_step_m: float = 0.001
    maximum_current_fraction_step: float = 0.01
    max_trials: int = 6
    backtrack_factor: float = 0.5
    armijo_fraction: float = 1e-4
    minimum_qa_decrease: float = 1e-10
    violation_reduction_fraction: float = 1e-4
    projected_gradient_atol: float = 1e-6
    projected_gradient_rtol: float = 1e-4
    restoration_fraction_interior: float = 0.001
    restoration_fraction_boundary: float = 0.02
    restoration_boundary_start: float = 0.8
    restoration_budget_fraction: float = 0.2
    restoration_descent_fraction: float = 0.5

    def __post_init__(self):
        for name in (
            "constraint_relative_tolerance",
            "maximum_coil_step_m",
            "maximum_current_fraction_step",
            "minimum_qa_decrease",
            "projected_gradient_atol",
            "projected_gradient_rtol",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError("invalid positive policy value: " + name)
        if not 1 <= self.max_trials <= 8 or int(self.max_trials) != self.max_trials:
            raise ValueError("max_trials must be an integer in 1..8")
        for name in ("backtrack_factor", "armijo_fraction", "violation_reduction_fraction"):
            if not 0 < getattr(self, name) < 1:
                raise ValueError("invalid fraction: " + name)
        for name in (
            "restoration_fraction_interior",
            "restoration_fraction_boundary",
            "restoration_boundary_start",
            "restoration_budget_fraction",
            "restoration_descent_fraction",
        ):
            if not np.isfinite(getattr(self, name)) or not 0 < getattr(self, name) < 1:
                raise ValueError("invalid restoring fraction: " + name)
        if self.restoration_fraction_interior > self.restoration_fraction_boundary:
            raise ValueError("boundary restoring weight must not be smaller")


@dataclass(frozen=True)
class Direction:
    """A projected/restoring direction with its prediction diagnostics."""

    delta: np.ndarray
    mode: str
    projected_gradient_norm: float
    objective_directional_derivative: float
    constraint_info: dict
    restoring_weight: float = 0.0
    combined_cap_factor: float = 1.0
    predicted_constraint_change: tuple = ()


@dataclass(frozen=True)
class SearchResult:
    """A bounded line-search result; a missing candidate means rejection."""

    candidate: object
    values: np.ndarray | None
    delta: np.ndarray | None
    trials: tuple


def limit_direction(delta, policy, *, expand, motion_bounds):
    """Scale a direction to the supplied physical motion bounds."""
    motion, current = motion_bounds(delta)
    ratios = [
        policy.maximum_coil_step_m / motion if motion else np.inf,
        policy.maximum_current_fraction_step / current if current else np.inf,
    ]
    factor = min(ratios)
    if not expand:
        factor = min(1.0, factor)
    if not np.isfinite(factor):
        return np.zeros_like(delta, dtype=float)
    return np.asarray(delta) * factor


def constraint_state(values, targets, constraint_scales, policy, tolerances=None):
    """Report errors relative to physical target bands."""
    values, targets, scales = map(np.asarray, (values, targets, constraint_scales))
    if (
        targets.ndim != 1
        or values.shape != (targets.size + 1,)
        or scales.shape != targets.shape
        or not all(np.all(np.isfinite(x)) for x in (values, targets, scales))
        or np.any(scales <= 0)
    ):
        raise ValueError("invalid physical constraints")
    tolerances = np.asarray(
        policy.constraint_relative_tolerance * np.abs(targets) if tolerances is None else tolerances
    )
    if tolerances.shape != targets.shape or np.any(tolerances <= 0) or not np.all(np.isfinite(tolerances)):
        raise ValueError("positive finite physical tolerances required")
    errors = values[1:] * scales
    ratios = np.abs(errors) / tolerances
    return dict(
        feasible=bool(np.all(ratios <= 1.0)),
        errors=errors.tolist(),
        tolerances=tolerances.tolist(),
        ratios=ratios.tolist(),
        violation=float(max(0.0, np.max(ratios, initial=0.0) - 1.0)),
    )


def proposal(values, jacobian, scales, targets, constraint_scales, policy, *, motion_bounds, tolerances=None):
    """Independently sized QA descent plus bounded target restoration.

    QA sizing depends on geometry/current caps, never the equality-error norm.
    The normal component corrects a fraction of the error to the original targets.
    Its weight grows near a tolerance boundary but is limited by separate motion
    and current budgets and by the requirement to preserve predicted QA descent.
    Projection uses the target equality Jacobian rows in scaled coordinates.
    Thus convergence means equality-tangent stationarity, not a full inequality
    KKT certificate for the tolerance band.
    """

    values, scales, jacobian = map(np.asarray, (values, scales, jacobian))
    info = constraint_state(values, targets, constraint_scales, policy, tolerances)
    if (
        scales.ndim != 1
        or np.any(scales <= 0)
        or not np.all(np.isfinite(scales))
        or jacobian.shape != (values.size, scales.size)
        or not np.all(np.isfinite(jacobian))
    ):
        raise ValueError("invalid scaled Jacobian")
    jac = jacobian * scales[None, :]
    u, singular, vt = np.linalg.svd(jac[1:], full_matrices=False)
    if len(targets) and (len(singular) < len(targets) or singular[-1] <= 1e-12 * max(1.0, singular[0])):
        raise ValueError("equality Jacobian is rank deficient")
    gradient = float(values[0]) * jac[0]
    # Remove the gradient component normal to the constraint surfaces.
    projected = gradient - vt.T @ (vt @ gradient)
    norm = float(np.linalg.norm(projected))
    normal = -(vt.T @ ((u.T @ values[1:]) / singular)) * scales
    weight = 0.0
    cap_factor = 1.0
    if info["feasible"]:
        qa = limit_direction(-projected * scales, policy, expand=True, motion_bounds=motion_bounds)
        gradient_physical = float(values[0]) * jacobian[0]
        qa_slope = float(gradient_physical @ qa)
        normal_slope = float(gradient_physical @ normal)
        proximity = np.clip(
            (max(info["ratios"], default=0.0) - policy.restoration_boundary_start)
            / (1.0 - policy.restoration_boundary_start),
            0.0,
            1.0,
        )
        weight = policy.restoration_fraction_interior + proximity**2 * (
            policy.restoration_fraction_boundary - policy.restoration_fraction_interior
        )
        motion, current = motion_bounds(normal)
        if motion:
            weight = min(weight, policy.restoration_budget_fraction * policy.maximum_coil_step_m / motion)
        if current:
            weight = min(weight, policy.restoration_budget_fraction * policy.maximum_current_fraction_step / current)
        if qa_slope >= 0.0:
            # Do not turn lack of QA descent into an uphill mixed proposal.
            weight = 0.0
        elif normal_slope > 0.0:
            weight = min(weight, policy.restoration_descent_fraction * (-qa_slope) / normal_slope)
        direction = qa + weight * normal
        motion, current = motion_bounds(direction)
        cap_factor = min(
            1.0,
            policy.maximum_coil_step_m / motion if motion else 1.0,
            policy.maximum_current_fraction_step / current if current else 1.0,
        )
        mode = "qa_restoring"
    else:
        direction = normal
        mode = "restoration"
    delta = limit_direction(direction, policy, expand=False, motion_bounds=motion_bounds)
    derivative = float((float(values[0]) * jacobian[0]) @ delta)
    return Direction(
        delta, mode, norm, derivative, info, weight, cap_factor, tuple(float(x) for x in jacobian[1:] @ delta)
    )


def converged(direction, initial_projected_gradient_norm, policy):
    """Test feasible equality-tangent stationarity against the original reference."""
    threshold = max(policy.projected_gradient_atol, policy.projected_gradient_rtol * initial_projected_gradient_norm)
    return bool(direction.constraint_info["feasible"] and direction.projected_gradient_norm <= threshold), threshold


def acceptance(before, after, direction, alpha, targets, constraint_scales, policy, tolerances=None):
    """Require objective descent within bands, or reduce existing infeasibility."""

    old = direction.constraint_info
    try:
        new = constraint_state(after, targets, constraint_scales, policy, tolerances)
    except ValueError:
        return False, dict(reason="nonfinite_candidate_metrics")
    qa_before, qa_after = 0.5 * float(before[0]) ** 2, 0.5 * float(after[0]) ** 2
    if not np.isfinite(qa_after):
        return False, dict(reason="nonfinite_candidate_objective")
    if old["feasible"]:
        required = max(
            policy.minimum_qa_decrease,
            policy.armijo_fraction * alpha * max(0.0, -direction.objective_directional_derivative),
        )
        passed = new["feasible"] and qa_before - qa_after >= required
        reason = (
            "qa_decrease_and_feasible"
            if passed
            else ("constraint_violation" if not new["feasible"] else "insufficient_qa_decrease")
        )
    else:
        required = policy.violation_reduction_fraction * alpha * old["violation"]
        passed = new["feasible"] or (old["violation"] - new["violation"] >= max(1e-12, required))
        reason = "violation_reduced" if passed else "insufficient_violation_reduction"
    return bool(passed), dict(
        reason=reason,
        before=old,
        after=new,
        qa_before=qa_before,
        qa_after=qa_after,
        required_decrease=required,
    )


def backtrack(
    before, direction, targets, constraint_scales, policy, evaluate, record, *, motion_bounds, tolerances=None
):
    """Evaluate each reduced proposal from the same accepted base state.

    evaluate(delta, trial_index) must return (candidate, actual_rows) without
    mutating the base state. Only a returned accepted candidate may be promoted.
    """
    import numpy as np

    trials = []
    for index in range(1, policy.max_trials + 1):
        alpha = policy.backtrack_factor ** (index - 1)
        delta = alpha * direction.delta
        motion, current = motion_bounds(delta)
        if motion > policy.maximum_coil_step_m * (1 + 1e-12) or current > policy.maximum_current_fraction_step * (
            1 + 1e-12
        ):
            raise ValueError("proposal exceeds the geometry/current budget")
        entry = dict(trial=index, alpha=alpha, maximum_coil_bound_m=motion, maximum_current_fraction=current)
        if not np.any(delta):
            entry.update(accepted=False, reason="zero_proposal")
            trials.append(entry)
            record(entry)
            break
        try:
            candidate, values = evaluate(delta, index)
            passed, details = acceptance(
                before, values, direction, alpha, targets, constraint_scales, policy, tolerances
            )
            entry.update(accepted=passed, **details)
        except TrialRejected as exc:
            candidate = values = None
            entry.update(accepted=False, reason="numerical_rejection", error=str(exc))
        trials.append(entry)
        record(entry)
        if entry["accepted"]:
            return SearchResult(candidate, np.asarray(values), delta, tuple(trials))
    return SearchResult(None, None, None, tuple(trials))


def minimize_projected(problem, *, maxiter, options=None, callback=None, initial_gradient_norm=None, event=None):
    """Minimize with projected descent/restoration; maxiter counts new accepted steps.

    The callback receives a SciPy OptimizeResult after each accepted step.
    Evaluations never promote a state. A budget stop has success=False;
    convergence denotes equality-tangent stationarity, not an inequality KKT certificate.
    """
    if isinstance(maxiter, bool) or int(maxiter) != maxiter or maxiter < 0:
        raise ValueError("maxiter must be a nonnegative integer")
    policy = options if isinstance(options, ProjectedOptions) else ProjectedOptions(**(options or {}))
    if initial_gradient_norm is not None and (not np.isfinite(initial_gradient_norm) or initial_gradient_norm <= 0):
        raise ValueError("invalid initial projected-gradient reference")
    emit = event or (lambda *args, **kwargs: None)
    status, nit, nfev, njev = "step_budget_reached", 0, 0, 0
    last_gradient = gradient_step = None

    def evaluate_trial(delta, index):
        nonlocal nfev
        nfev += 1
        return problem.evaluate_trial(delta, index)

    for _ in range(int(maxiter)):
        values, jacobian = problem.linearize()
        njev += 1
        direction = proposal(
            values,
            jacobian,
            problem.scales,
            problem.targets,
            problem.constraint_scales,
            policy,
            motion_bounds=problem.parameterization.motion_bounds,
            tolerances=problem.constraint_tolerances,
        )
        if initial_gradient_norm is None:
            initial_gradient_norm = direction.projected_gradient_norm
        problem.initial_gradient_norm = initial_gradient_norm
        last_gradient, gradient_step = direction.projected_gradient_norm, nit
        done, threshold = converged(direction, initial_gradient_norm, policy)
        emit(
            "direction",
            direction=direction,
            initial_gradient_norm=initial_gradient_norm,
            threshold=threshold,
            iteration=nit,
        )
        if done:
            status = "converged"
            break
        trial = backtrack(
            values,
            direction,
            problem.targets,
            problem.constraint_scales,
            policy,
            evaluate_trial,
            lambda record: emit("trial", record=record),
            motion_bounds=problem.parameterization.motion_bounds,
            tolerances=problem.constraint_tolerances,
        )
        if trial.candidate is None:
            status = "stagnated"
            emit("stagnated", trials=trial.trials)
            break
        problem.accept(trial.candidate)
        nit += 1
        intermediate = OptimizeResult(
            x=problem.accepted.parameters.copy(),
            fun=0.5 * float(trial.values[0]) ** 2,
            nit=nit,
            accepted=problem.accepted,
            values=trial.values.copy(),
            trial=trial,
            initial_gradient_norm=initial_gradient_norm,
        )
        emit("accepted", result=intermediate)
        if callback is not None:
            try:
                callback(intermediate)
            except StopIteration:
                status = "callback_stopped"
                break
    values = problem.optimizer_rows(problem.accepted)
    return OptimizeResult(
        x=problem.accepted.parameters.copy(),
        fun=0.5 * float(values[0]) ** 2,
        nit=nit,
        nfev=nfev,
        njev=njev,
        status=status,
        success=status == "converged",
        message={
            "converged": "Feasible equality-tangent stationarity reached.",
            "stagnated": "No acceptable step within the trial budget.",
            "step_budget_reached": "Accepted-step budget reached.",
            "callback_stopped": "Stopped by callback.",
        }[status],
        accepted=problem.accepted,
        constraints=problem.constraint_values(problem.accepted.parameters),
        feasible=constraint_state(
            values, problem.targets, problem.constraint_scales, policy, problem.constraint_tolerances
        )["feasible"],
        initial_gradient_norm=initial_gradient_norm,
        projected_gradient_norm=last_gradient,
        gradient_evaluated_at_iteration=gradient_step,
    )
