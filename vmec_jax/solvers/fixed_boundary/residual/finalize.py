"""Final result assembly helpers for residual-iteration VMEC solves."""

from __future__ import annotations

from typing import Any, Callable, Mapping
import time

import numpy as np

from ..diagnostics.io import _pack_resume_state_record
from .runtime import (
    _build_residual_iter_timing_report,
    _build_resume_state_base,
    _format_residual_iter_timing_message,
)

_EMPTY_HISTORY_KEYS = ("w_history", "fsqr2_history", "fsqz2_history", "fsql2_history", "grad_rms_history", "step_history")
_RESUME_BASE_KEYS = (
    "time_step inv_tau fsq_prev fsq0_prev flip_sign iter1 last_iter2 ijacob "
    "bad_resets res0 res1 prev_rz_fsq bad_growth_streak huge_force_restart_count "
    "vmec2000_cache_valid freeb_ivac freeb_ivacskip freeb_nvacskip freeb_nvskip0 "
    "freeb_last_model freeb_nestor_runtime"
).split()
_RESUME_HEAVY_ARRAY_KEYS = "vRcc vRss vZsc vZcs vLsc vLcs vRsc vRcs vZcc vZss vLcc vLss".split()
_RESUME_HEAVY_OBJECT_KEYS = (
    "state_checkpoint cache_precond_diag cache_tcon cache_norms cache_rz_scale cache_l_scale "
    "cache_rz_norm cache_f_norm1 cache_prec_rz_mats cache_prec_rz_jmax cache_prec_lam_prec "
    "cache_prec_faclam cache_prec_lam_debug cache_constraint_rcon0 cache_constraint_zcon0"
).split()

__all__ = [
    "attach_residual_iter_timing_diagnostics",
    "build_residual_iter_resume_state_payload",
    "build_residual_iter_resume_state_from_namespace",
    "finalize_residual_iter_result",
    "precompile_only_residual_iter_result",
    "vmec2000_state_only_scan_result",
    "vmec2000_traced_scan_result",
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


def build_residual_iter_resume_state_from_namespace(
    namespace: Mapping[str, Any],
    *,
    resume_state_mode: str,
) -> dict[str, Any] | None:
    """Build a residual-iteration resume payload from selected local values."""

    mode = str(resume_state_mode)
    if mode == "none":
        return None
    base_kwargs = {key: namespace[key] for key in _RESUME_BASE_KEYS}
    heavy = None
    if mode == "full":
        heavy = {key: np.asarray(namespace[key]) for key in _RESUME_HEAVY_ARRAY_KEYS}
        heavy.update({key: namespace[key] for key in _RESUME_HEAVY_OBJECT_KEYS})
    return build_residual_iter_resume_state_payload(
        resume_state_mode=mode,
        base_kwargs=base_kwargs,
        heavy_payload=heavy,
    )


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


def _empty_history_result(
    *,
    result_type: type,
    state: Any,
    n_iter: int,
    diagnostics: dict[str, Any],
    empty_history: Any | None = None,
) -> Any:
    empty = np.zeros((0,), dtype=float) if empty_history is None else empty_history
    return result_type(
        state=state,
        n_iter=int(n_iter),
        diagnostics=diagnostics,
        **dict.fromkeys(_EMPTY_HISTORY_KEYS, empty),
    )


def precompile_only_residual_iter_result(*, result_type: type, state: Any) -> Any:
    return _empty_history_result(result_type=result_type, state=state, n_iter=0, diagnostics={"precompile_only": True})


def vmec2000_state_only_scan_result(
    *,
    result_type: type,
    carry_final: Any,
    empty_history: Any,
    max_iter: int,
    diagnostics: dict[str, Any],
    attach_free_boundary_diagnostics: Callable[[Any], Any],
) -> Any:
    """Construct a VMEC2000 state-only scan result with empty histories."""

    return attach_free_boundary_diagnostics(
        _empty_history_result(result_type=result_type, state=carry_final.state, n_iter=max_iter, diagnostics=diagnostics, empty_history=empty_history)
    )


def vmec2000_traced_scan_result(
    *,
    result_type: type,
    carry_final: Any,
    empty_history: Any,
    max_iter: int,
    resume_state: dict[str, Any],
    scan_use_precomputed: bool,
    scan_use_lax_tridi: bool,
    attach_free_boundary_diagnostics: Callable[[Any], Any],
    traced_diagnostics_func: Callable[..., dict[str, Any]],
) -> Any:
    """Construct a traced VMEC2000 scan result with resume diagnostics."""

    diagnostics = traced_diagnostics_func(
        resume_state=resume_state,
        scan_use_precomputed=bool(scan_use_precomputed),
        scan_use_lax_tridi=bool(scan_use_lax_tridi),
    )
    return attach_free_boundary_diagnostics(
        _empty_history_result(result_type=result_type, state=carry_final.state, n_iter=max_iter, diagnostics=diagnostics, empty_history=empty_history)
    )
