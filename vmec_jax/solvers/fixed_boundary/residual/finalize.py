"""Final result assembly helpers for residual-iteration VMEC solves."""

from __future__ import annotations

from typing import Any, Callable, Mapping
import time

import numpy as np

from ....solve_diagnostics_io import _pack_resume_state_record
from .runtime import (
    _build_residual_iter_timing_report,
    _build_resume_state_base,
    _format_residual_iter_timing_message,
)

__all__ = [
    "attach_residual_iter_timing_diagnostics",
    "build_residual_iter_resume_state_payload",
    "finalize_residual_iter_result",
]


def attach_residual_iter_timing_diagnostics(
    diagnostics: dict[str, Any],
    timing_stats: dict[str, float],
    *,
    timing_enabled: bool,
    timing_detail_enabled: bool,
    finalize_diag_build_start: float | None,
    iteration_loop_start: float | None,
    finalize_start: float | None,
    solve_wall_start: float,
    print_timing: bool = True,
) -> dict[str, Any]:
    """Attach residual-iteration timing diagnostics to ``diagnostics``."""

    if not bool(timing_enabled):
        return diagnostics
    if finalize_diag_build_start is not None:
        timing_stats["finalize_diag_build"] += time.perf_counter() - float(finalize_diag_build_start)
    if iteration_loop_start is not None and finalize_start is not None:
        timing_stats["iteration_loop"] = float(finalize_start) - float(iteration_loop_start)
    if finalize_start is not None:
        timing_stats["finalize"] = time.perf_counter() - float(finalize_start)
    timing_report = _build_residual_iter_timing_report(
        timing_stats,
        solve_total_s=float(time.perf_counter() - float(solve_wall_start)),
        timing_detail_enabled=bool(timing_detail_enabled),
    )
    timing_stats["iteration_loop_unattributed"] = float(timing_report["iteration_loop_unattributed_s"])
    diagnostics["timing"] = timing_report
    if bool(print_timing):
        try:
            print(
                _format_residual_iter_timing_message(
                    timing_report,
                    timing_detail_enabled=bool(timing_detail_enabled),
                ),
                flush=True,
            )
        except Exception:
            pass
    return diagnostics


def build_residual_iter_resume_state_payload(
    *,
    resume_state_mode: str,
    base_kwargs: Mapping[str, Any],
    heavy_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Build packed resume-state payload for residual iteration."""

    mode = str(resume_state_mode)
    if mode == "none":
        return None
    base = _build_resume_state_base(**dict(base_kwargs))
    heavy = dict(heavy_payload) if mode == "full" and heavy_payload is not None else None
    return _pack_resume_state_record(base=base, heavy=heavy, mode=mode)


def finalize_residual_iter_result(
    *,
    result_type: type,
    state: Any,
    w_history: Any,
    fsqr2_history: Any,
    fsqz2_history: Any,
    fsql2_history: Any,
    grad_rms_history: Any,
    step_history: Any,
    diagnostics: dict[str, Any],
    attach_free_boundary_diagnostics: Callable[[Any], Any],
    return_final_force_payload: bool,
    converged: bool,
    final_force_payload: Any,
) -> Any:
    """Construct the final residual-iteration result object."""

    result = attach_free_boundary_diagnostics(
        result_type(
            state=state,
            n_iter=len(w_history) - 1,
            w_history=np.asarray(w_history, dtype=float),
            fsqr2_history=np.asarray(fsqr2_history, dtype=float),
            fsqz2_history=np.asarray(fsqz2_history, dtype=float),
            fsql2_history=np.asarray(fsql2_history, dtype=float),
            grad_rms_history=np.asarray(grad_rms_history, dtype=float),
            step_history=np.asarray(step_history, dtype=float),
            diagnostics=diagnostics,
        )
    )
    if bool(return_final_force_payload) and bool(converged):
        try:
            object.__setattr__(result, "_final_force_payload", final_force_payload)
        except Exception:
            pass
    return result
