"""Concrete Gauss-Newton least-squares solver used by fixed-boundary examples."""

from __future__ import annotations

import math
import os
from typing import Sequence

import numpy as np


def _skip_exhausted_gauss_newton_jacobian() -> bool:
    flag = os.getenv("VMEC_JAX_OPT_SKIP_EXHAUSTED_GN_JACOBIAN", "").strip().lower()
    return flag in ("1", "true", "yes", "on")


def gauss_newton_least_squares(
    residual_fun,
    jacobian_fun,
    x0,
    *,
    max_nfev: int = 10,
    ftol: float = 1e-4,
    gtol: float = 1e-4,
    xtol: float = 1e-4,
    x_scale=None,
    forward_residual_fun=None,
    post_jacobian_callback=None,
    exact_residual_after_jacobian_fun=None,
    damping_factors: Sequence[float] | None = None,
    verbose: int = 1,
):
    """Solve a nonlinear least-squares problem with a concrete Gauss-Newton loop."""

    x = np.asarray(x0, dtype=float).copy()
    scale = np.ones_like(x) if x_scale is None else np.asarray(x_scale, dtype=float).copy()
    scale[scale == 0.0] = 1.0
    trial_residual_fun = residual_fun if forward_residual_fun is None else forward_residual_fun
    damping_schedule = (
        (1e-6, 1e-4, 1e-2, 1.0, 100.0) if damping_factors is None else tuple(float(value) for value in damping_factors)
    )

    nfev = 0
    njev = 0
    alpha_prev = 1.0
    x_prev = None
    cost_prev = None
    accepted_residual = None
    accepted_cost = None
    accepted_step_norm = None
    success = False
    message = "maximum function evaluations exceeded"

    if verbose:
        print("   Iteration     Total nfev        Cost      Cost reduction    Step norm     Optimality")

    iteration = 0
    while nfev < int(max_nfev):
        if accepted_residual is None:
            residual = np.asarray(residual_fun(x), dtype=float).reshape(-1)
            nfev += 1
        else:
            residual = accepted_residual
            accepted_residual = None
        cost = 0.5 * float(np.dot(residual, residual))
        if cost == 0.0:
            success = True
            message = "`gtol` termination condition is satisfied."
            accepted_cost = cost
            accepted_step_norm = 0.0
            if verbose:
                print(f"{iteration:12d}{nfev:16d}{cost:13.4e}{0.0:18.2e}{0.0:16.2e}{0.0:16.2e}")
            break
        if nfev >= int(max_nfev) and _skip_exhausted_gauss_newton_jacobian():
            accepted_cost = cost
            accepted_step_norm = 0.0
            if verbose:
                print(f"{iteration:12d}{nfev:16d}{cost:13.4e}{0.0:18.2e}{0.0:16.2e}{float('nan'):16.2e}")
            break

        jacobian = np.asarray(jacobian_fun(x), dtype=float)
        njev += 1
        if exact_residual_after_jacobian_fun is not None:
            _exact_res = exact_residual_after_jacobian_fun()
            if _exact_res is not None:
                residual = np.asarray(_exact_res, dtype=float).reshape(-1)
                cost = 0.5 * float(np.dot(residual, residual))
        if post_jacobian_callback is not None:
            post_jacobian_callback()
        gradient = jacobian.T @ residual
        optimality = float(np.linalg.norm(gradient, ord=np.inf))
        if not np.isfinite(optimality):
            message = "non-finite optimality encountered"
            break
        if optimality <= float(gtol):
            success = True
            message = "`gtol` termination condition is satisfied."
            accepted_cost = cost
            accepted_step_norm = 0.0
            if verbose:
                print(f"{iteration:12d}{nfev:16d}{cost:13.4e}{0.0:18.2e}{0.0:16.2e}{optimality:16.2e}")
            break

        jacobian_scaled = jacobian * scale[None, :]
        normal = jacobian_scaled.T @ jacobian_scaled
        rhs = -(jacobian_scaled.T @ residual)
        try:
            step_y, *_ = np.linalg.lstsq(jacobian_scaled, -residual, rcond=None)
        except np.linalg.LinAlgError:
            message = "linear least-squares solve failed"
            break
        step = scale * np.asarray(step_y, dtype=float)
        step_norm = float(np.linalg.norm(step))
        if not np.all(np.isfinite(step)):
            message = "non-finite Gauss-Newton step encountered"
            break
        if step_norm <= float(xtol):
            success = True
            message = "`xtol` termination condition is satisfied."
            accepted_cost = cost
            accepted_step_norm = step_norm
            if verbose:
                print(f"{iteration:12d}{nfev:16d}{cost:13.4e}{0.0:18.2e}{step_norm:16.2e}{optimality:16.2e}")
            break

        accepted = False
        cost_trial = math.inf
        residual_trial = None
        x_trial = None
        alpha_accepted = 0.0
        step_accepted = None

        def _try_step(candidate_step, initial_alpha):
            nonlocal nfev, accepted, cost_trial, residual_trial, x_trial
            nonlocal alpha_accepted, step_accepted
            alpha = min(max(float(initial_alpha), 1.0 / 128.0), 1.0)
            for _ in range(8):
                x_candidate = x + alpha * candidate_step
                residual_candidate = np.asarray(trial_residual_fun(x_candidate), dtype=float).reshape(-1)
                nfev += 1
                cost_candidate = 0.5 * float(np.dot(residual_candidate, residual_candidate))
                if np.isfinite(cost_candidate) and cost_candidate < cost:
                    x_trial = x_candidate
                    residual_trial = residual_candidate
                    cost_trial = cost_candidate
                    alpha_accepted = alpha
                    step_accepted = candidate_step
                    accepted = True
                    break
                alpha *= 0.5
                if nfev >= int(max_nfev):
                    break
            return accepted

        _try_step(step, alpha_prev)
        if (not accepted) and nfev < int(max_nfev):
            diag = np.maximum(np.diag(normal), 1.0)
            for damping in damping_schedule:
                if damping <= 0.0:
                    continue
                try:
                    damped_y = np.linalg.solve(
                        normal + float(damping) * np.diag(diag),
                        rhs,
                    )
                except np.linalg.LinAlgError:
                    continue
                damped_step = scale * np.asarray(damped_y, dtype=float)
                if not np.all(np.isfinite(damped_step)):
                    continue
                _try_step(damped_step, 1.0)
                if accepted or nfev >= int(max_nfev):
                    break

        if not accepted:
            message = "line search failed to reduce the objective"
            accepted_cost = cost
            accepted_step_norm = 0.0
            if verbose:
                print(f"{iteration:12d}{nfev:16d}{cost:13.4e}{0.0:18.2e}{0.0:16.2e}{optimality:16.2e}")
            break

        cost_reduction = cost - cost_trial
        step_norm_trial = float(np.linalg.norm(alpha_accepted * step_accepted))
        if verbose:
            print(
                f"{iteration:12d}{nfev:16d}{cost_trial:13.4e}{cost_reduction:18.2e}"
                f"{step_norm_trial:16.2e}{optimality:16.2e}"
            )

        x_prev = x
        cost_prev = cost
        x = x_trial
        accepted_residual = residual_trial
        accepted_cost = cost_trial
        accepted_step_norm = step_norm_trial
        alpha_prev = alpha_accepted
        iteration += 1

        if cost_prev is not None and cost_prev > 0.0 and cost_reduction <= float(ftol) * cost_prev:
            success = True
            message = "`ftol` termination condition is satisfied."
            break

    if accepted_cost is None:
        residual_final = np.asarray(residual_fun(x), dtype=float).reshape(-1)
        nfev += 1
        accepted_cost = 0.5 * float(np.dot(residual_final, residual_final))
        accepted_step_norm = 0.0

    return {
        "x": x,
        "cost": float(accepted_cost),
        "objective": float(2.0 * accepted_cost),
        "nfev": int(nfev),
        "njev": int(njev),
        "nit": int(iteration),
        "success": bool(success),
        "status": 1 if success else 0,
        "message": str(message),
        "step_norm": float(accepted_step_norm if accepted_step_norm is not None else 0.0),
        "x_prev": None if x_prev is None else np.asarray(x_prev, dtype=float),
        "cost_prev": None if cost_prev is None else float(cost_prev),
    }
