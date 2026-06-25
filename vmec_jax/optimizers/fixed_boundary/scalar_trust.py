"""Scalar-adjoint trust-region optimizer for fixed-boundary VMEC runs."""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np


def scalar_cost_only_trials_enabled(owner: Any, requested: bool | None) -> bool:
    """Resolve the cost-only trial policy for the scalar-trust optimizer."""

    if requested is not None:
        return bool(requested)
    env_flag = os.getenv("VMEC_JAX_OPT_SCALAR_COST_ONLY_TRIALS")
    if env_flag is None:
        return bool(getattr(owner, "_scalar_trust_cost_only_trials", False))
    return env_flag.strip().lower() in ("1", "true", "yes", "on")


def _fallback_direction(
    owner: Any,
    grad: np.ndarray,
    lbfgs_pairs: list[tuple[np.ndarray, np.ndarray, float]],
) -> np.ndarray:
    owner._profile_add("scalar_trust_gradient_direction", 0.0)
    lbfgs_pairs.clear()
    return -grad


def scalar_trust_direction(
    owner: Any,
    grad: np.ndarray,
    lbfgs_pairs: list[tuple[np.ndarray, np.ndarray, float]],
) -> np.ndarray:
    """Return a safeguarded L-BFGS direction in scaled parameter space."""

    grad = np.asarray(grad, dtype=float)
    if not lbfgs_pairs:
        owner._profile_add("scalar_trust_gradient_direction", 0.0)
        return -grad

    q = grad.copy()
    alphas: list[float] = []
    for s_vec, y_vec, rho in reversed(lbfgs_pairs):
        alpha = float(rho * np.dot(s_vec, q))
        alphas.append(alpha)
        q = q - alpha * y_vec

    s_last, y_last, _rho_last = lbfgs_pairs[-1]
    yy_last = float(np.dot(y_last, y_last))
    sy_last = float(np.dot(s_last, y_last))
    h0 = sy_last / yy_last if yy_last > 0.0 and sy_last > 0.0 else 1.0
    h0 = min(1.0e6, max(1.0e-12, h0))
    r = h0 * q
    for (s_vec, y_vec, rho), alpha in zip(lbfgs_pairs, reversed(alphas)):
        beta = float(rho * np.dot(y_vec, r))
        r = r + s_vec * (alpha - beta)

    direction = -r
    if (
        not np.all(np.isfinite(direction))
        or float(np.dot(direction, grad)) >= -1.0e-14 * max(1.0, float(np.linalg.norm(grad) ** 2))
    ):
        return _fallback_direction(owner, grad, lbfgs_pairs)
    owner._profile_add("scalar_trust_lbfgs_direction", 0.0)
    return direction


def run_scalar_trust_exact_optimizer(
    owner: Any,
    params0_arr: np.ndarray,
    *,
    x_scale,
    max_nfev: int,
    ftol: float,
    gtol: float,
    scalar_step_bound: float | None,
    scalar_cost_only_trials: bool | None,
) -> tuple[dict, bool]:
    """Run the safeguarded scalar-adjoint optimizer branch.

    ``owner`` is the :class:`FixedBoundaryExactOptimizer` instance.  The
    optimizer owns the exact solve, residual, and accepted-point caches; this
    helper owns only the scalar trust-region step selection.
    """

    scale = np.ones_like(params0_arr) if x_scale is None else np.asarray(x_scale, dtype=float)
    scale[scale == 0.0] = 1.0
    base_params = owner._base_params_vector()
    y_current = (params0_arr + base_params) / scale
    x_current = params0_arr.copy()
    last_history_key = [owner._exact_cache_key(params0_arr)]
    max_scalar_evals = max(1, int(max_nfev))
    initial_radius = 1.0 if scalar_step_bound is None or float(scalar_step_bound) <= 0.0 else float(scalar_step_bound)
    radius = initial_radius
    min_radius = max(1.0e-12, initial_radius * 1.0e-8)
    eval_count = 0
    accepted_steps = 0
    best_eval: dict[str, object] = {
        "cost": float("inf"),
        "x": x_current.copy(),
        "y": y_current.copy(),
        "state": None,
        "grad_x": np.zeros_like(params0_arr),
        "grad_y": np.zeros_like(y_current),
    }

    def _record_history_from_cached_state(x, cost):
        owner._record_cached_exact_history_entry(x, last_history_key=last_history_key, cost=float(cost))

    def _evaluate_y(y):
        nonlocal eval_count
        eval_count += 1
        x = np.asarray(y, dtype=float) * scale - base_params
        cost, grad_x = owner.objective_and_gradient_fun(x)
        grad_y = np.asarray(grad_x, dtype=float) * scale
        if float(cost) < float(best_eval["cost"]):
            best_state = owner._cached_exact_state(x)
            best_eval.update(
                {
                    "cost": float(cost),
                    "x": np.asarray(x, dtype=float).copy(),
                    "y": np.asarray(y, dtype=float).copy(),
                    "state": best_state,
                    "grad_x": np.asarray(grad_x, dtype=float).copy(),
                    "grad_y": grad_y.copy(),
                }
            )
        return float(cost), np.asarray(x, dtype=float), grad_y

    def _trial_cost_y(y):
        x = np.asarray(y, dtype=float) * scale - base_params
        t0 = time.perf_counter()
        residual = np.asarray(owner.forward_residual_fun(x), dtype=float).reshape(-1)
        cost = 0.5 * float(np.dot(residual, residual))
        owner._profile_add("scalar_trust_cost_only_trial", time.perf_counter() - t0)
        return cost, x

    cost_current, x_current, grad_y = _evaluate_y(y_current)
    grad_norm = float(np.linalg.norm(grad_y, ord=np.inf))
    success_result = bool(grad_norm <= float(gtol))
    status_result = 1 if success_result else 0
    message_result = (
        "`gtol` termination condition is satisfied."
        if success_result
        else "maximum number of scalar objective evaluations is exceeded"
    )
    lbfgs_pairs: list[tuple[np.ndarray, np.ndarray, float]] = []
    max_lbfgs_pairs = 8
    armijo_c1 = 1.0e-4
    backtrack_factor = 0.1
    cost_only_trials = scalar_cost_only_trials_enabled(owner, scalar_cost_only_trials)

    while not success_result and eval_count < max_scalar_evals:
        grad_norm_2 = float(np.linalg.norm(grad_y))
        if not np.isfinite(grad_norm_2) or grad_norm_2 <= 0.0:
            message_result = "zero or non-finite scalar-adjoint gradient"
            break

        direction_y = scalar_trust_direction(owner, grad_y, lbfgs_pairs)
        direction_norm = float(np.linalg.norm(direction_y))
        if not np.isfinite(direction_norm) or direction_norm <= 0.0:
            message_result = "zero or non-finite scalar-adjoint search direction"
            break
        base_step_y = direction_y * min(1.0, radius / direction_norm)
        directional_decrease = -float(np.dot(grad_y, base_step_y))
        if directional_decrease <= 0.0 or not np.isfinite(directional_decrease):
            base_step_y = -grad_y * min(1.0, radius / grad_norm_2)
            directional_decrease = -float(np.dot(grad_y, base_step_y))
            lbfgs_pairs.clear()

        accepted = False
        shrink = 1.0
        while eval_count < max_scalar_evals:
            step_y = shrink * base_step_y
            if float(np.linalg.norm(step_y)) < min_radius:
                break
            y_trial = y_current + step_y
            armijo_limit = cost_current - armijo_c1 * shrink * max(0.0, directional_decrease)
            if cost_only_trials:
                cost_trial_estimate, x_trial = _trial_cost_y(y_trial)
                passes_trial_filter = np.isfinite(cost_trial_estimate) and (
                    cost_trial_estimate <= armijo_limit or cost_trial_estimate < cost_current
                )
                if not passes_trial_filter:
                    owner._profile_add("scalar_trust_rejected_step", 0.0)
                    shrink *= backtrack_factor
                    continue
                cost_trial, x_trial, grad_trial = _evaluate_y(y_trial)
                if not (np.isfinite(cost_trial) and (cost_trial <= armijo_limit or cost_trial < cost_current)):
                    owner._profile_add("scalar_trust_exact_validation_rejected_step", 0.0)
                    shrink *= backtrack_factor
                    continue
            else:
                cost_trial, x_trial, grad_trial = _evaluate_y(y_trial)
            if np.isfinite(cost_trial) and (cost_trial <= armijo_limit or cost_trial < cost_current):
                y_current = y_trial
                x_current = x_trial
                cost_previous = cost_current
                cost_current = cost_trial
                step_accepted = np.asarray(step_y, dtype=float)
                grad_delta = np.asarray(grad_trial, dtype=float) - np.asarray(grad_y, dtype=float)
                grad_y = grad_trial
                grad_norm = float(np.linalg.norm(grad_y, ord=np.inf))
                accepted_steps += 1
                accepted = True
                sy = float(np.dot(step_accepted, grad_delta))
                curvature_floor = 1.0e-12 * max(
                    1.0,
                    float(np.linalg.norm(step_accepted)) * float(np.linalg.norm(grad_delta)),
                )
                if sy > curvature_floor and np.all(np.isfinite(grad_delta)):
                    lbfgs_pairs.append((step_accepted, grad_delta, 1.0 / sy))
                    if len(lbfgs_pairs) > max_lbfgs_pairs:
                        del lbfgs_pairs[0]
                _record_history_from_cached_state(x_current, cost_current)
                step_norm = float(np.linalg.norm(step_accepted))
                if shrink < 1.0:
                    radius = min(initial_radius, max(2.0 * step_norm, step_norm / backtrack_factor))
                    owner._profile_add("scalar_trust_backtracked_accept", 0.0)
                elif step_norm >= 0.8 * radius:
                    radius = min(initial_radius, max(radius * 1.5, radius))
                else:
                    radius = min(initial_radius, max(radius, 2.0 * step_norm))
                if abs(cost_previous - cost_current) <= float(ftol) * max(1.0, abs(cost_current)):
                    success_result = True
                    status_result = 2
                    message_result = "`ftol` termination condition is satisfied."
                elif grad_norm <= float(gtol):
                    success_result = True
                    status_result = 1
                    message_result = "`gtol` termination condition is satisfied."
                break
            owner._profile_add("scalar_trust_rejected_step", 0.0)
            shrink *= backtrack_factor

        if success_result:
            break
        if not accepted:
            message_result = "scalar trust-region radius became too small"
            break

    if not success_result and eval_count >= max_scalar_evals:
        message_result = "maximum number of scalar objective evaluations is exceeded"

    x_result = np.asarray(best_eval["x"], dtype=float)
    best_state = best_eval.get("state")
    if best_state is not None:
        owner._remember_exact_state(owner._exact_cache_key(x_result), best_state)
    cost_result = float(best_eval["cost"])
    return (
        {
            "x": x_result,
            "cost": cost_result,
            "objective": 2.0 * cost_result,
            "nfev": int(eval_count),
            "njev": int(eval_count),
            "nit": int(accepted_steps),
            "success": success_result,
            "status": status_result,
            "message": message_result,
            "step_norm": float(np.linalg.norm(x_result - params0_arr)),
            "x_prev": None,
            "cost_prev": None,
        },
        bool(cost_only_trials),
    )

