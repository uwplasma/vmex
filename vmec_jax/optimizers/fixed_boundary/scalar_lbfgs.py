"""L-BFGS scalar-adjoint optimizer branch for fixed-boundary VMEC runs."""

from __future__ import annotations

from typing import Any

import numpy as np


class LBFGSBudgetExceeded(RuntimeError):
    """Raised internally when the exact scalar-adjoint budget is exhausted."""


def run_lbfgs_adjoint_exact_optimizer(
    owner: Any,
    params0_arr: np.ndarray,
    *,
    x_scale,
    max_nfev: int,
    ftol: float,
    gtol: float,
    verbose: int,
    lbfgs_step_bound: float | None,
) -> dict:
    """Run the L-BFGS-B scalar-adjoint optimizer branch."""

    try:
        from scipy.optimize import minimize as _scipy_minimize
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("method='lbfgs_adjoint' requires scipy.optimize.minimize") from exc

    scale = np.ones_like(params0_arr) if x_scale is None else np.asarray(x_scale, dtype=float)
    scale[scale == 0.0] = 1.0
    base_params = owner._base_params_vector()
    y0 = (params0_arr + base_params) / scale
    last_history_key = [owner._exact_cache_key(params0_arr)]
    max_scalar_evals = max(1, int(max_nfev))
    eval_count = [0]
    best_eval: dict[str, object] = {
        "cost": float("inf"),
        "x": params0_arr.copy(),
        "y": y0.copy(),
        "grad_x": np.zeros_like(params0_arr),
        "grad_y": np.zeros_like(y0),
    }

    def _record_history_from_cached_state(x, cost):
        owner._record_cached_exact_history_entry(x, last_history_key=last_history_key, cost=float(cost))

    def _objective_and_gradient_y(y):
        if eval_count[0] >= max_scalar_evals:
            raise LBFGSBudgetExceeded
        eval_count[0] += 1
        x = np.asarray(y, dtype=float) * scale - base_params
        cost, grad_x = owner.objective_and_gradient_fun(x)
        grad_y = np.asarray(grad_x, dtype=float) * scale
        if float(cost) < float(best_eval["cost"]):
            best_eval.update(
                {
                    "cost": float(cost),
                    "x": np.asarray(x, dtype=float).copy(),
                    "y": np.asarray(y, dtype=float).copy(),
                    "grad_x": np.asarray(grad_x, dtype=float).copy(),
                    "grad_y": grad_y.copy(),
                }
            )
        _record_history_from_cached_state(x, cost)
        return float(cost), grad_y

    try:
        lbfgs_bounds = None
        if lbfgs_step_bound is not None and float(lbfgs_step_bound) > 0.0:
            bound = float(lbfgs_step_bound)
            lbfgs_bounds = [(float(center) - bound, float(center) + bound) for center in np.asarray(y0, dtype=float)]
        minimize_result = _scipy_minimize(
            _objective_and_gradient_y,
            y0,
            jac=True,
            method="L-BFGS-B",
            bounds=lbfgs_bounds,
            options={
                "maxiter": int(max_nfev),
                "maxfun": int(max_nfev),
                "ftol": float(ftol),
                "gtol": float(gtol),
                "disp": bool(int(verbose) > 0),
            },
        )
        x_result = np.asarray(minimize_result.x, dtype=float) * scale - base_params
        cost_result = float(minimize_result.fun)
        success_result = bool(minimize_result.success)
        status_result = int(minimize_result.status)
        message_result = str(minimize_result.message)
        nit_result = int(getattr(minimize_result, "nit", 0))
    except LBFGSBudgetExceeded:
        x_result = np.asarray(best_eval["x"], dtype=float)
        cost_result = float(best_eval["cost"])
        success_result = False
        status_result = 0
        message_result = "maximum number of scalar objective evaluations is exceeded"
        nit_result = 0

    return {
        "x": x_result,
        "cost": cost_result,
        "objective": 2.0 * cost_result,
        "nfev": int(eval_count[0]),
        "njev": int(eval_count[0]),
        "nit": nit_result,
        "success": success_result,
        "status": status_result,
        "message": message_result,
        "step_norm": 0.0,
        "x_prev": None,
        "cost_prev": None,
    }

