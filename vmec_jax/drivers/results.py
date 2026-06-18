"""Result assembly helpers for :mod:`vmec_jax.driver`.

These helpers stitch chunked/staged solver diagnostics into a single
``SolveVmecResidualResult``.  Keeping this mechanical bookkeeping outside the
driver keeps the driver focused on workflow control and makes result-merging
behavior independently testable.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..solve import SolveVmecResidualResult


STAGE_CHUNK_DIAG_KEYS = (
    "step_status_history",
    "restart_reason_history",
    "pre_restart_reason_history",
    "time_step_history",
    "res0_history",
    "res1_history",
    "fsq_prev_history",
    "bad_growth_streak_history",
    "iter1_history",
    "bcovar_update_history",
    "include_edge_history",
    "zero_m1_history",
    "dt_eff_history",
    "update_rms_history",
    "w_curr_history",
    "w_try_history",
    "w_try_ratio_history",
    "restart_path_history",
    "min_tau_history",
    "max_tau_history",
    "bad_jacobian_history",
    "fsq1_history",
    "fsqr1_history",
    "fsqz1_history",
    "fsql1_history",
    "r00_history",
    "z00_history",
    "wb_history",
    "wp_history",
    "w_vmec_history",
    "rz_norm_history",
    "f_norm1_history",
    "gcr2_p_history",
    "gcz2_p_history",
    "gcl2_p_history",
)


def copy_final_force_payload(result_i: SolveVmecResidualResult, source_i) -> SolveVmecResidualResult:
    payload = getattr(source_i, "_final_force_payload", None)
    if payload is not None:
        try:
            object.__setattr__(result_i, "_final_force_payload", payload)
        except Exception:
            pass
    return result_i


def result_with_diag(result_i: SolveVmecResidualResult, **updates) -> SolveVmecResidualResult:
    diag = dict(result_i.diagnostics)
    diag.update(updates)
    out = SolveVmecResidualResult(
        state=result_i.state,
        n_iter=int(result_i.n_iter),
        w_history=np.asarray(result_i.w_history),
        fsqr2_history=np.asarray(result_i.fsqr2_history),
        fsqz2_history=np.asarray(result_i.fsqz2_history),
        fsql2_history=np.asarray(result_i.fsql2_history),
        grad_rms_history=np.asarray(result_i.grad_rms_history),
        step_history=np.asarray(result_i.step_history),
        diagnostics=diag,
    )
    return copy_final_force_payload(out, result_i)


def cat_result_history(results_i: list[object], attr: str) -> np.ndarray:
    """Concatenate an optional history array across VMEC stage/chunk results."""

    parts = [
        np.asarray(getattr(result_i, attr))
        for result_i in results_i
        if getattr(result_i, attr, None) is not None
    ]
    return np.concatenate(parts, axis=0) if parts else np.zeros((0,), dtype=float)


def timing_solve_total_s(timing_i: dict) -> float:
    """Return the generic solve wall time from non-scan or scan timing blocks."""

    if not isinstance(timing_i, dict):
        return float("nan")
    for key in ("solve_total_s", "scan_total_s"):
        try:
            value = float(timing_i.get(key, np.nan))
        except Exception:
            value = float("nan")
        if np.isfinite(value):
            return value
    return float("nan")


def aggregate_stage_chunk_timing(results_i: list[SolveVmecResidualResult]) -> dict:
    """Combine per-chunk timing dictionaries for one logical VMEC stage."""

    timings: list[dict] = []
    for result_i in results_i:
        try:
            timing_i = result_i.diagnostics.get("timing", {})
        except Exception:
            timing_i = {}
        if isinstance(timing_i, dict) and timing_i:
            timings.append(timing_i)
    if not timings:
        return {}

    aggregate: dict[str, object] = dict(timings[-1])
    sum_keys: set[str] = set()
    for timing_i in timings:
        for key, value in timing_i.items():
            if key.endswith("_per_iter_s") or key.endswith("_first_s"):
                continue
            if key.endswith("_s") or key.endswith("_calls") or key == "iterations":
                try:
                    float(value)
                except Exception:
                    continue
                sum_keys.add(str(key))

    for key in sum_keys:
        vals: list[float] = []
        for timing_i in timings:
            try:
                vals.append(float(timing_i.get(key, 0.0)))
            except Exception:
                vals.append(0.0)
        total = float(np.sum(vals))
        if key.endswith("_calls") or key == "iterations":
            aggregate[key] = int(round(total))
        else:
            aggregate[key] = total

    iterations = max(int(aggregate.get("iterations", 0)), 1)
    for key, value in list(aggregate.items()):
        if key.endswith("_s") and not key.endswith("_per_iter_s") and not key.endswith("_first_s"):
            aggregate[f"{key[:-2]}_per_iter_s"] = float(value) / float(iterations)
    chunk_solve_total = np.asarray([timing_solve_total_s(t) for t in timings], dtype=float)
    valid = chunk_solve_total[np.isfinite(chunk_solve_total)]
    if valid.size:
        aggregate["solve_total_s"] = float(np.sum(valid))
        iterations = max(int(aggregate.get("iterations", 0)), 1)
        aggregate["solve_total_per_iter_s"] = float(aggregate["solve_total_s"]) / float(iterations)
    aggregate["chunk_count"] = int(len(timings))
    aggregate["chunk_solve_total_s"] = chunk_solve_total
    return aggregate


def merge_stage_chunk_results(
    results_i: list[SolveVmecResidualResult],
    *,
    mode_i: str,
) -> SolveVmecResidualResult:
    if len(results_i) == 1:
        return result_with_diag(
            results_i[0],
            accelerated_stage_chunked=False,
            accelerated_stage_effective_mode=str(mode_i),
        )

    last = results_i[-1]
    diag = dict(last.diagnostics)
    for key in STAGE_CHUNK_DIAG_KEYS:
        if any(key in r.diagnostics for r in results_i):
            diag[key] = np.concatenate(
                [np.asarray(r.diagnostics.get(key, np.zeros((0,), dtype=float))) for r in results_i]
            )
    diag["accelerated_stage_chunked"] = True
    diag["accelerated_stage_effective_mode"] = str(mode_i)
    diag["accelerated_stage_chunk_count"] = int(len(results_i))
    diag["accelerated_stage_chunk_iters"] = np.asarray(
        [int(r.n_iter) + 1 for r in results_i],
        dtype=int,
    )
    timing = aggregate_stage_chunk_timing(results_i)
    if timing:
        diag["timing"] = timing
    out = SolveVmecResidualResult(
        state=last.state,
        n_iter=int(sum(int(r.n_iter) + 1 for r in results_i) - 1),
        w_history=cat_result_history(results_i, "w_history"),
        fsqr2_history=cat_result_history(results_i, "fsqr2_history"),
        fsqz2_history=cat_result_history(results_i, "fsqz2_history"),
        fsql2_history=cat_result_history(results_i, "fsql2_history"),
        grad_rms_history=cat_result_history(results_i, "grad_rms_history"),
        step_history=cat_result_history(results_i, "step_history"),
        diagnostics=diag,
    )
    return copy_final_force_payload(out, last)


def assemble_multigrid_stage_result(
    *,
    stage_results: list[SolveVmecResidualResult],
    state: object,
    solver_mode: str,
    accelerated_mode: bool,
    multigrid_user_provided: bool,
    accelerated_single_grid_default: bool,
    ns_stages: list[int],
    niter_stages: list[int],
    ftol_stages: list[float],
    stage_offsets: list[int],
    stage_mode_history: list[str],
    stage_wall_s: list[float],
    stage_solve_total_s: list[float],
    niter_stages_input: object,
) -> SolveVmecResidualResult:
    """Assemble the public result for a completed fixed-boundary multigrid run."""

    last = stage_results[-1]
    diag = dict(last.diagnostics)
    diag["solver_mode"] = str(solver_mode)
    diag["accelerated_mode"] = bool(accelerated_mode)
    diag["accelerated_scan"] = bool(accelerated_mode) and bool(diag.get("use_scan", False))
    diag["multigrid_user_provided"] = bool(multigrid_user_provided)
    diag["accelerated_single_grid_default"] = bool(accelerated_single_grid_default)
    diag["multigrid_ns_stages"] = np.asarray(ns_stages, dtype=int)
    diag["multigrid_niter_stages"] = np.asarray(niter_stages, dtype=int)
    diag["multigrid_ftol_stages"] = np.asarray(ftol_stages, dtype=float)
    diag["multigrid_stage_offsets"] = np.asarray(stage_offsets, dtype=int)
    diag["multigrid_stage_modes"] = np.asarray(stage_mode_history, dtype=object)
    diag["multigrid_stage_wall_s"] = np.asarray(stage_wall_s, dtype=float)
    diag["multigrid_stage_solve_total_s"] = np.asarray(stage_solve_total_s, dtype=float)
    try:
        final_stage_niter = int(last.n_iter)
        final_stage_budget = int(niter_stages[-1]) if niter_stages else 0
        if niter_stages_input is not None:
            exhausted = bool(final_stage_niter + 1 >= final_stage_budget)
        else:
            exhausted = bool(final_stage_niter >= final_stage_budget)
        diag["multigrid_final_stage_niter_exhausted"] = exhausted
    except Exception:
        diag["multigrid_final_stage_niter_exhausted"] = False

    for key in STAGE_CHUNK_DIAG_KEYS:
        if any(key in r.diagnostics for r in stage_results):
            diag[key] = np.concatenate(
                [np.asarray(r.diagnostics.get(key, np.zeros((0,), dtype=float))) for r in stage_results]
            )

    out = SolveVmecResidualResult(
        state=state,
        n_iter=int(sum(int(r.n_iter) + 1 for r in stage_results) - 1),
        w_history=cat_result_history(stage_results, "w_history"),
        fsqr2_history=cat_result_history(stage_results, "fsqr2_history"),
        fsqz2_history=cat_result_history(stage_results, "fsqz2_history"),
        fsql2_history=cat_result_history(stage_results, "fsql2_history"),
        grad_rms_history=cat_result_history(stage_results, "grad_rms_history"),
        step_history=cat_result_history(stage_results, "step_history"),
        diagnostics=diag,
    )
    return copy_final_force_payload(out, last)


def finalize_fixed_boundary_convergence_result(
    result: SolveVmecResidualResult,
    *,
    requested_ftol: float,
    fsq_total_target: float,
    accelerated_mode: bool,
    result_final_residuals: Callable[..., object],
    result_meets_requested_ftol: Callable[..., bool],
    result_hits_total_target: Callable[..., bool],
) -> SolveVmecResidualResult:
    """Attach final convergence diagnostics to a fixed-boundary result."""

    final_residuals = result_final_residuals(result)
    final_diag = dict(result.diagnostics)
    final_diag["requested_ftol"] = float(requested_ftol)
    final_diag["fsq_total_target"] = (
        float(fsq_total_target)
        if (final_diag.get("fsq_total_target", None) is not None or bool(accelerated_mode))
        else None
    )
    if final_residuals is not None:
        final_diag["final_fsqr"] = float(final_residuals[0])
        final_diag["final_fsqz"] = float(final_residuals[1])
        final_diag["final_fsql"] = float(final_residuals[2])
    final_diag["converged_strict"] = bool(result_meets_requested_ftol(result, ftol=float(requested_ftol)))
    final_diag["converged_by_total_fsq"] = bool(
        result_hits_total_target(result, fsq_total_target=float(fsq_total_target))
    )
    final_diag["converged"] = bool(final_diag["converged_strict"])
    out = SolveVmecResidualResult(
        state=result.state,
        n_iter=int(result.n_iter),
        w_history=np.asarray(result.w_history),
        fsqr2_history=np.asarray(result.fsqr2_history),
        fsqz2_history=np.asarray(result.fsqz2_history),
        fsql2_history=np.asarray(result.fsql2_history),
        grad_rms_history=np.asarray(result.grad_rms_history),
        step_history=np.asarray(result.step_history),
        diagnostics=final_diag,
    )
    return copy_final_force_payload(out, result)


def stage_switch_reason_from_progress(
    *,
    start_total_fsq: float,
    best_total_fsq: float,
    target_total_fsq: float,
    chunk_iters: int,
    remaining_budget: int,
) -> str | None:
    if remaining_budget <= 0:
        return None
    if (not np.isfinite(best_total_fsq)) or (not np.isfinite(start_total_fsq)):
        return "nonfinite_total_fsq"
    if best_total_fsq <= max(0.0, float(target_total_fsq)):
        return None
    if best_total_fsq >= start_total_fsq:
        return "nondecreasing_total_fsq"
    if best_total_fsq <= 0.0 or start_total_fsq <= 0.0:
        return None
    rate = (np.log(float(start_total_fsq)) - np.log(float(best_total_fsq))) / max(1, int(chunk_iters))
    if (not np.isfinite(rate)) or rate <= 0.0:
        return "nonpositive_decay_rate"
    projected_iters = np.log(float(best_total_fsq) / max(float(target_total_fsq), 1.0e-300)) / rate
    if (not np.isfinite(projected_iters)) or projected_iters > float(remaining_budget):
        return (
            "projected_budget_miss:"
            f" projected_iters={float(projected_iters):.1f}"
            f" remaining_budget={int(remaining_budget)}"
        )
    return None


def vmec_history_relerr(lhs_hist, rhs_hist) -> float:
    lhs_hist = np.asarray(lhs_hist)
    rhs_hist = np.asarray(rhs_hist)
    if lhs_hist.shape != rhs_hist.shape:
        return float("inf")
    diff = np.max(np.abs(lhs_hist - rhs_hist))
    scale = max(float(np.max(np.abs(rhs_hist))), 1e-30)
    return float(diff / scale)


def vmec_histories_match(lhs, rhs, *, rtol: float, atol: float) -> bool:
    keys = ("w_history", "fsqr2_history", "fsqz2_history", "fsql2_history")
    for key in keys:
        lhs_hist = np.asarray(getattr(lhs, key))
        rhs_hist = np.asarray(getattr(rhs, key))
        if lhs_hist.shape != rhs_hist.shape:
            return False
        if not np.allclose(lhs_hist, rhs_hist, rtol=float(rtol), atol=float(atol)):
            return False
    return True
