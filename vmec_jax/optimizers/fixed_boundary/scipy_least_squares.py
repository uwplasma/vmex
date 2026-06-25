"""SciPy least-squares optimizer branches for exact fixed-boundary runs."""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from .linear_guards import finite_linear_operator_output
from .linear_guards import finite_residual_vector
from .linear_guards import linear_operator_matrix_arg
from .linear_guards import linear_operator_vector_arg


def _best_exact_failure_result(owner: Any, params0_arr: np.ndarray, *, message: str, exc: Exception) -> dict:
    best_exact_params = getattr(owner, "_best_exact_params", None)
    best_exact_cost = float(getattr(owner, "_best_exact_cost", math.inf))
    if best_exact_params is None or not np.isfinite(best_exact_cost):
        raise exc
    x_result = np.asarray(best_exact_params, dtype=float).copy()
    return {
        "x": x_result,
        "cost": best_exact_cost,
        "objective": 2.0 * best_exact_cost,
        "nfev": max(1, len(getattr(owner, "_history", []))),
        "njev": max(0, len(getattr(owner, "_history", [])) - 1),
        "nit": 0,
        "success": False,
        "status": -1,
        "message": f"{message}; returning best exact accepted point: {exc}",
        "step_norm": float(np.linalg.norm(x_result - params0_arr)),
        "x_prev": None,
        "cost_prev": None,
        "_selected_best_exact_point": True,
        "_optimizer_exception": repr(exc),
    }


def _scipy_result_dict(scipy_result, *, scale: np.ndarray, base_params: np.ndarray) -> dict:
    x_result = np.asarray(scipy_result.x, dtype=float) * scale - base_params
    return {
        "x": x_result,
        "cost": float(scipy_result.cost),
        "objective": float(2.0 * scipy_result.cost),
        "nfev": int(scipy_result.nfev),
        "njev": 0 if scipy_result.njev is None else int(scipy_result.njev),
        "nit": 0,
        "success": bool(scipy_result.success),
        "status": int(scipy_result.status),
        "message": str(scipy_result.message),
        "step_norm": 0.0,
        "x_prev": None,
        "cost_prev": None,
    }


def scaled_optimizer_space(owner: Any, params0_arr: np.ndarray, x_scale) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(scale, base_params, y0)`` for scaled SciPy callbacks."""

    scale = np.ones_like(params0_arr) if x_scale is None else np.asarray(x_scale, dtype=float)
    scale[scale == 0.0] = 1.0
    base_params = owner._base_params_vector()
    y0 = (params0_arr + base_params) / scale
    return scale, base_params, y0


def run_scipy_matrix_free_exact_optimizer(
    owner: Any,
    params0_arr: np.ndarray,
    *,
    x_scale,
    max_nfev: int,
    ftol: float,
    gtol: float,
    xtol: float,
    verbose: int,
    scipy_lsmr_maxiter: int | None,
) -> tuple[dict, int | None]:
    """Run SciPy TRF with a matrix-free exact linear operator."""

    try:
        from scipy.optimize import least_squares as _scipy_least_squares
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("method='scipy_matrix_free' requires scipy.optimize.least_squares") from exc
    if scipy_lsmr_maxiter is None:
        scipy_lsmr_maxiter = 4

    scale, base_params, y0 = scaled_optimizer_space(owner, params0_arr, x_scale)
    last_history_key = [owner._exact_cache_key(params0_arr)]
    matrix_free_residual_size = [None]

    def _record_history_from_cached_state(x):
        owner._record_cached_exact_history_entry(x, last_history_key=last_history_key)

    def _residuals_y(y):
        x = np.asarray(y, dtype=float) * scale - base_params
        cached_residual = owner._cached_exact_residual(x)
        if cached_residual is not None:
            raw_residual = cached_residual
        else:
            cached_state = owner._cached_exact_state(x)
            raw_residual = (
                owner._evaluate_residuals_from_state(cached_state)
                if cached_state is not None
                else owner.forward_residual_fun(x)
            )
        residual = finite_residual_vector(
            raw_residual,
            profile_add=owner._profile_add,
            profile_name="matrix_free_nonfinite_residual",
            expected_size=matrix_free_residual_size[0],
        )
        if matrix_free_residual_size[0] is None:
            matrix_free_residual_size[0] = int(residual.size)
        return residual

    def _jacobian_y(y):
        x = np.asarray(y, dtype=float) * scale - base_params
        op_x = owner.residual_linear_operator(x)
        _record_history_from_cached_state(x)

        def _matvec(v):
            v_arr = linear_operator_vector_arg(v, size=int(scale.size), name="scaled matvec direction")
            return finite_linear_operator_output(
                op_x.matvec(v_arr * scale),
                profile_add=owner._profile_add,
                profile_name="matrix_free_nonfinite_matvec",
            )

        def _matmat(v):
            v_arr = linear_operator_matrix_arg(v, rows=int(scale.size), name="scaled matmat directions")
            return finite_linear_operator_output(
                op_x.matmat(v_arr * scale[:, None]),
                profile_add=owner._profile_add,
                profile_name="matrix_free_nonfinite_matmat",
            )

        def _rmatvec(w):
            w_arr = linear_operator_vector_arg(w, size=int(op_x.shape[0]), name="scaled rmatvec cotangent")
            return finite_linear_operator_output(
                op_x.rmatvec(w_arr) * scale,
                profile_add=owner._profile_add,
                profile_name="matrix_free_nonfinite_rmatvec",
            )

        try:
            from scipy.sparse.linalg import LinearOperator
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("method='scipy_matrix_free' requires scipy") from exc

        return LinearOperator(
            shape=op_x.shape,
            matvec=_matvec,
            matmat=_matmat,
            rmatvec=_rmatvec,
            dtype=np.dtype(float),
        )

    try:
        scipy_result = _scipy_least_squares(
            _residuals_y,
            y0,
            jac=_jacobian_y,
            method="trf",
            tr_solver="lsmr",
            tr_options=({"maxiter": int(scipy_lsmr_maxiter)} if scipy_lsmr_maxiter is not None else None),
            max_nfev=max_nfev,
            ftol=ftol,
            gtol=gtol,
            xtol=xtol,
            verbose=2 if int(verbose) > 0 else 0,
        )
        return _scipy_result_dict(scipy_result, scale=scale, base_params=base_params), scipy_lsmr_maxiter
    except Exception as exc:
        return (
            _best_exact_failure_result(
                owner,
                params0_arr,
                message="scipy matrix-free least_squares failed",
                exc=exc,
            ),
            scipy_lsmr_maxiter,
        )


def run_scipy_dense_exact_optimizer(
    owner: Any,
    params0_arr: np.ndarray,
    *,
    x_scale,
    max_nfev: int,
    ftol: float,
    gtol: float,
    xtol: float,
    verbose: int,
    scipy_tr_solver: str | None,
    scipy_lsmr_maxiter: int | None,
) -> dict:
    """Run SciPy TRF with dense exact Jacobian callbacks."""

    try:
        from scipy.optimize import least_squares as _scipy_least_squares
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("method='scipy' requires scipy.optimize.least_squares") from exc

    scale, base_params, y0 = scaled_optimizer_space(owner, params0_arr, x_scale)

    def _residuals_y(y):
        x = np.asarray(y, dtype=float) * scale - base_params
        t_cb = time.perf_counter()
        cache_key = owner._exact_cache_key(x)
        cached_residual = owner._cached_exact_residual(cache_key=cache_key)
        if cached_residual is not None:
            owner._trace_callback_event(
                "residual",
                x,
                source="exact_residual_cache",
                wall_time_s=time.perf_counter() - t_cb,
            )
            return cached_residual
        cached_state = owner._cached_exact_state(x)
        if cached_state is not None:
            out = owner._evaluate_residuals_from_state(cached_state)
            owner._trace_callback_event(
                "residual",
                x,
                source="exact_state_cache",
                wall_time_s=time.perf_counter() - t_cb,
            )
            return out
        cached_trial = owner._cached_trial_residual(x)
        if cached_trial is not None:
            owner._trace_callback_event(
                "residual",
                x,
                source="trial_residual_cache",
                wall_time_s=time.perf_counter() - t_cb,
            )
            return cached_trial
        out = owner.forward_residual_fun(x)
        owner._trace_callback_event(
            "residual",
            x,
            source="trial_solve",
            wall_time_s=time.perf_counter() - t_cb,
        )
        return out

    def _jacobian_y(y):
        x = np.asarray(y, dtype=float) * scale - base_params
        t_cb = time.perf_counter()
        owner._last_jacobian_source = "exact_tape_replay"
        jac = np.asarray(owner._jacobian_fun_tracked(x), dtype=float) * scale[None, :]
        owner._trace_callback_event(
            "jacobian",
            x,
            source=getattr(owner, "_last_jacobian_source", "exact_tape_replay"),
            wall_time_s=time.perf_counter() - t_cb,
        )
        owner._exact_cache.clear()
        return jac

    try:
        scipy_result = _scipy_least_squares(
            _residuals_y,
            y0,
            jac=_jacobian_y,
            method="trf",
            tr_solver=scipy_tr_solver,
            tr_options=(
                {"maxiter": int(scipy_lsmr_maxiter)}
                if scipy_lsmr_maxiter is not None and scipy_tr_solver == "lsmr"
                else None
            ),
            max_nfev=max_nfev,
            ftol=ftol,
            gtol=gtol,
            xtol=xtol,
            verbose=2 if int(verbose) > 0 else 0,
        )
        return _scipy_result_dict(scipy_result, scale=scale, base_params=base_params)
    except Exception as exc:
        return _best_exact_failure_result(
            owner,
            params0_arr,
            message="scipy least_squares failed",
            exc=exc,
        )

