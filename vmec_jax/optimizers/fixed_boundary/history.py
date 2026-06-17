"""History helpers for exact fixed-boundary optimization.

The exact optimizer records history from several paths: residual callbacks,
Jacobian callbacks, matrix-free replay, and final accepted-point replay.  These
small helpers keep the metric reconstruction policy in one place so optimizer
algorithms can focus on step selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ResidualHistoryPolicy:
    """Metadata required to reconstruct history rows directly from residuals."""

    aspect_target: float | None
    aspect_weight: float
    n_non_qs: int = 1
    n_qs: int | None = None
    has_residual_block_metadata: bool | None = None
    has_iota_callback: bool = False

    def has_qs_residual_block_metadata(self) -> bool:
        if self.has_residual_block_metadata is not None:
            return bool(self.has_residual_block_metadata)
        return self.n_qs is not None or self.n_non_qs is not None

    def can_build_qs_from_residuals(self) -> bool:
        return self.has_qs_residual_block_metadata()

    def can_build_aspect_from_residuals(self) -> bool:
        if self.aspect_target is None:
            return False
        return bool(np.isfinite(float(self.aspect_weight)) and float(self.aspect_weight) != 0.0)

    def can_build_history_from_residuals(self) -> bool:
        return (
            not bool(self.has_iota_callback)
            and self.can_build_aspect_from_residuals()
            and self.can_build_qs_from_residuals()
        )


def qs_objective_from_residuals(residuals: Any, policy: ResidualHistoryPolicy) -> float:
    """Return the QS/objective residual sum of squares from a residual vector."""

    res = np.asarray(residuals, dtype=float).reshape(-1)
    if policy.n_qs is not None:
        n_qs = max(0, min(int(policy.n_qs), int(res.shape[0])))
        if n_qs == 0:
            return 0.0
        block = res[-n_qs:]
    else:
        start = max(0, min(int(policy.n_non_qs), int(res.shape[0])))
        block = res[start:]
    return float(np.dot(block, block))


def history_entry_from_residuals(
    residuals: Any,
    *,
    wall_time_s: float,
    policy: ResidualHistoryPolicy,
) -> dict:
    """Build a history row from residual metadata without solving diagnostics."""

    if not policy.can_build_aspect_from_residuals():
        raise ValueError("Residual history requires a finite nonzero aspect residual weight.")
    res = np.asarray(residuals, dtype=float).reshape(-1)
    aspect = float(policy.aspect_target) + float(res[0]) / float(policy.aspect_weight)
    cost = float(0.5 * np.dot(res, res))
    return {
        "wall_time_s": float(wall_time_s),
        "cost": cost,
        "objective": 2.0 * cost,
        "qs_objective": qs_objective_from_residuals(res, policy),
        "aspect": aspect,
    }


def monotone_final_wall_time(*, now_s: float, history: list[dict]) -> float:
    """Return a final history timestamp that never goes backwards."""

    final_wall_time_s = float(now_s)
    if history:
        final_wall_time_s = max(final_wall_time_s, float(history[-1].get("wall_time_s", 0.0)))
    return final_wall_time_s


def scipy_tr_solver_for_history(method_key: str, scipy_tr_solver: str | None) -> str | None:
    """Return the SciPy trust-region solver name serialized in history."""

    if method_key == "scipy":
        return scipy_tr_solver
    if method_key in ("scipy_matrix_free", "matrix_free", "scipy_mf"):
        return "lsmr"
    return None


def build_run_history_dump(
    *,
    label: str = "Optimisation",
    max_nfev: int,
    ftol: float,
    gtol: float,
    xtol: float,
    method_key: str,
    method_requested: str,
    method_auto_reason: str | None,
    exact_path: str,
    scipy_tr_solver: str | None,
    scipy_lsmr_maxiter: int | None,
    lbfgs_step_bound: float | None,
    scalar_step_bound: float | None,
    scalar_cost_only_trials_used: bool | None,
    solver_device: str,
    inner_max_iter: int,
    inner_ftol: float,
    trial_max_iter: int,
    trial_ftol: float,
    final_wall_time_s: float,
    result: dict,
    cost0: float,
    cost_final: float,
    qs_total0: float,
    qs_total_final: float,
    aspect0: float,
    aspect_final: float,
    history: list[dict],
    profile: dict,
    selected_best_exact: bool,
    rejected_trial_exact_history_count: int,
    optimizer_exception: object | None,
    iota_fn_present: bool,
    entry0: dict,
    entry_final: dict,
    target_iota: float | None,
    target_aspect: float | None,
    callback_trace: list[dict] | None = None,
) -> dict:
    """Assemble the serializable exact-optimizer history payload."""

    history_dump: dict = {
        "label": label,
        "max_nfev": int(max_nfev),
        "ftol": float(ftol),
        "gtol": float(gtol),
        "xtol": float(xtol),
        "method": method_key,
        "method_requested": method_requested,
        "method_auto_reason": method_auto_reason,
        "exact_path": exact_path,
        "scipy_tr_solver": scipy_tr_solver_for_history(method_key, scipy_tr_solver),
        "scipy_lsmr_maxiter": (None if scipy_lsmr_maxiter is None else int(scipy_lsmr_maxiter)),
        "lbfgs_step_bound": (None if lbfgs_step_bound is None else float(lbfgs_step_bound)),
        "scalar_step_bound": (None if scalar_step_bound is None else float(scalar_step_bound)),
        "scalar_cost_only_trials": scalar_cost_only_trials_used,
        "solver_device": solver_device,
        "inner_max_iter": int(inner_max_iter),
        "inner_ftol": float(inner_ftol),
        "trial_max_iter": int(trial_max_iter),
        "trial_ftol": float(trial_ftol),
        "total_wall_time_s": float(final_wall_time_s),
        "nfev": result["nfev"],
        "njev": result["njev"],
        "success": result["success"],
        "message": result["message"],
        "objective_initial": 2.0 * float(cost0),
        "objective_final": 2.0 * float(cost_final),
        "qs_initial": float(qs_total0),
        "qs_final": float(qs_total_final),
        "aspect_initial": float(aspect0),
        "aspect_final": float(aspect_final),
        "history": history,
        "profile": profile,
        "selected_best_exact_point": bool(selected_best_exact),
        "rejected_trial_exact_history_count": int(rejected_trial_exact_history_count),
    }
    if optimizer_exception is not None:
        history_dump["optimizer_exception"] = str(optimizer_exception)
    if iota_fn_present:
        history_dump["iota_initial"] = float(entry0["iota"])
        history_dump["iota_final"] = float(entry_final["iota"])
    if target_iota is not None:
        history_dump["target_iota"] = float(target_iota)
    if target_aspect is not None:
        history_dump["target_aspect"] = float(target_aspect)
    if callback_trace is not None:
        history_dump["callback_trace"] = callback_trace
    return history_dump

