"""Final result assembly helpers for residual-iteration VMEC solves."""

from __future__ import annotations

from typing import Any, Callable, Mapping
import time

import numpy as np

from ..diagnostics.io import _pack_resume_state_record
from ...free_boundary.control import (
    _freeb_edge_control_delta_tuple_projection_metrics,
    _freeb_edge_control_state_residual_metrics,
)
from .runtime import (
    _build_residual_iter_timing_report,
    _build_resume_state_base,
    _format_residual_iter_timing_message,
)
from .update import (
    delta_tuple_from_blocks,
    velocity_blocks_from_force_blocks,
    velocity_blocks_legacy_payload,
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
    "finalize_residual_iter_from_namespace",
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


def _edge_control_state_residual_payload(ns: Mapping[str, Any]) -> dict[str, Any]:
    """Return reduced-edge projection residuals without risking finalization."""

    try:
        projection = ns.get("freeb_edge_control_projection", {"enabled": False})
        return _freeb_edge_control_state_residual_metrics(ns["state"], projection)
    except Exception as exc:
        return {
            "enabled": bool(ns.get("freeb_edge_control_projection_enabled", False)),
            "status": "failed",
            "error": repr(exc),
        }


def _edge_control_update_direction_payload(ns: Mapping[str, Any]) -> dict[str, Any]:
    """Return reduced-edge projection residuals for the final update direction."""

    try:
        projection = ns.get("freeb_edge_control_projection", {"enabled": False})
        if not bool(projection.get("enabled", False)):
            return {"enabled": False, "status": "disabled"}
        update_force_blocks = ns.get("update_force_blocks")
        transforms = ns.get("_physical_delta_transforms")
        if update_force_blocks is None or transforms is None:
            return {"enabled": True, "status": "unavailable"}
        force_blocks = velocity_blocks_from_force_blocks(update_force_blocks)
        deltas = delta_tuple_from_blocks(
            1.0,
            transforms,
            *force_blocks,
            lasym=bool(getattr(ns.get("cfg"), "lasym", False)),
        )
        return _freeb_edge_control_delta_tuple_projection_metrics(deltas, projection)
    except Exception as exc:
        return {
            "enabled": bool(ns.get("freeb_edge_control_projection_enabled", False)),
            "status": "failed",
            "error": repr(exc),
        }


def _optional_float(value: Any) -> float | None:
    """Materialize an optional scalar diagnostic without failing finalization."""

    if value is None:
        return None
    try:
        result = float(np.asarray(value))
    except Exception:
        return None
    return result if np.isfinite(result) else None


def _namespace_with_best_scored_state(namespace: Mapping[str, Any]) -> tuple[Mapping[str, Any], bool]:
    """Return a finalization namespace that restores the best scored state."""

    best = namespace.get("best_scored")
    if not isinstance(best, Mapping) or best.get("state") is None:
        return namespace, False
    if not bool(namespace.get("return_best_scored_state", best.get("enabled", False))):
        return namespace, False
    fsqr = _optional_float(best.get("fsqr"))
    fsqz = _optional_float(best.get("fsqz"))
    fsql = _optional_float(best.get("fsql"))
    if fsqr is None or fsqz is None or fsql is None:
        return namespace, False
    ns = dict(namespace)
    ns.update(
        {
            "state": best["state"],
            "fsqr_f": fsqr,
            "fsqz_f": fsqz,
            "fsql_f": fsql,
            "prev_rz_fsq": fsqr + fsqz,
        }
    )
    for key in (
        "freeb_bsqvac_half_current",
        "freeb_nestor_runtime",
        "freeb_last_model",
        "freeb_last_diagnostics",
        "freeb_ivac",
        "freeb_ivacskip",
        "freeb_nvacskip",
        "freeb_nvskip0",
        "freeb_plascur",
    ):
        if best.get(key) is not None:
            ns[key] = best[key]
    return ns, True


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
    if "precond_cache" in namespace:
        namespace = {
            **namespace["precond_cache"].legacy_resume_payload(),
            **dict(namespace),
        }
    if "velocity_blocks" in namespace:
        namespace = {**velocity_blocks_legacy_payload(namespace["velocity_blocks"]), **dict(namespace)}
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


def final_free_boundary_residual_reports_from_namespace(
    namespace: Mapping[str, Any],
    *,
    nestor_external_only_step_func: Callable[..., Any],
    residual_fsq_from_norms_func: Callable[..., Any],
    device_get_floats_func: Callable[..., tuple[float, ...]],
) -> dict[str, Any]:
    """Return final accepted-state free-boundary residual report fields."""

    ns = namespace
    clock = time.perf_counter
    timing_stats, timing_enabled = ns["timing_stats"], bool(ns["timing_enabled"])
    final_bsqvac_half_current = ns["freeb_bsqvac_half_current"]
    report = {
        "final_fsqr_report": float(ns["fsqr_f"]),
        "final_fsqz_report": float(ns["fsqz_f"]),
        "final_fsql_report": float(ns["fsql_f"]),
        "final_residual_recomputed": False,
        "final_nestor_model": str(ns["freeb_last_model"]),
        "final_nestor_diagnostics": dict(ns["freeb_last_diagnostics"]),
        "final_nestor_recompute_attempted": False,
        "final_nestor_recompute_failed": False,
        "final_nestor_sample_time_s": 0.0,
        "final_nestor_solve_time_s": 0.0,
    }
    report["final_vacuum_stub"] = not bool(
        str(report["final_nestor_model"]).strip() and str(report["final_nestor_model"]) != "none"
    )
    if bool(ns["free_boundary_enabled"] and ns["freeb_couple_edge"]) and not report["final_vacuum_stub"]:
        report["final_nestor_recompute_attempted"] = True
        start = clock() if timing_enabled else None
        try:
            nestor_final, _runtime = nestor_external_only_step_func(
                state=ns["state"], static=ns["static"], ivac=1, ivacskip=0, iter_idx=None,
                runtime=ns["freeb_nestor_runtime"], extcur=tuple(getattr(ns["static"], "free_boundary_extcur", ()) or ()),
                plascur=float(ns["freeb_plascur"]),
                external_field_provider_kind=ns["external_field_provider_kind"],
                external_field_provider_static=ns["external_field_provider_static"],
                external_field_provider_params=ns["external_field_provider_params"],
            )
            report["final_nestor_sample_time_s"] = float(getattr(nestor_final, "sample_time_s", 0.0))
            report["final_nestor_solve_time_s"] = float(getattr(nestor_final, "solve_time_s", 0.0))
            report["final_nestor_model"] = str(getattr(nestor_final, "model", report["final_nestor_model"]))
            diag_final = getattr(nestor_final, "diagnostics", None)
            if isinstance(diag_final, dict):
                report["final_nestor_diagnostics"] = dict(diag_final)
            bsqvac_edge_final = np.asarray(nestor_final.vac_total.bsqvac, dtype=float)
            if (
                bsqvac_edge_final.ndim == 2 and int(bsqvac_edge_final.shape[1]) == 1 and int(getattr(ns["static"].cfg, "nzeta", 1)) > 1
            ):
                bsqvac_edge_final = np.repeat(bsqvac_edge_final, int(ns["static"].cfg.nzeta), axis=1)
            final_bsqvac_half_current = bsqvac_edge_final
            report["final_vacuum_stub"] = False
        except Exception:
            report["final_nestor_recompute_failed"] = True
            final_bsqvac_half_current = ns["freeb_bsqvac_half_current"]
        finally:
            if timing_enabled and start is not None:
                timing_stats["finalize_nestor_recompute"] += clock() - float(start)
    if bool(ns["free_boundary_enabled"]) and final_bsqvac_half_current is not None:
        start = clock() if timing_enabled else None
        try:
            _, _, gcr2_final, gcz2_final, gcl2_final, _, _, norms_final = ns["_compute_forces_iter"](
                ns["state"], include_edge=bool(ns["include_edge"]), include_edge_residual=True,
                zero_m1=ns["zero_m1"], freeb_bsqvac_half=final_bsqvac_half_current,
                constraint_precond_diag=ns["constraint_precond_diag"], constraint_tcon=ns["constraint_tcon_override"],
                constraint_precond_active=ns["constraint_precond_active"],
                constraint_tcon_active=ns["constraint_tcon_active"],
                iter2=ns["last_iter2"],
            )
            fsqr_final, fsqz_final, fsql_final = residual_fsq_from_norms_func(
                norms_final, gcr2=gcr2_final, gcz2=gcz2_final, gcl2=gcl2_final,
            )
            get_start = clock() if timing_enabled else None
            report["final_fsqr_report"], report["final_fsqz_report"], report["final_fsql_report"] = device_get_floats_func(fsqr_final, fsqz_final, fsql_final)
            if timing_enabled and get_start is not None:
                timing_stats["finalize_residual_device_get"] += clock() - float(get_start)
            report["final_residual_recomputed"] = True
        except Exception:
            report["final_fsqr_report"] = float(ns["fsqr_f"])
            report["final_fsqz_report"] = float(ns["fsqz_f"])
            report["final_fsql_report"] = float(ns["fsql_f"])
        finally:
            if timing_enabled and start is not None:
                timing_stats["finalize_residual_recompute"] += clock() - float(start)
    return report


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


def finalize_residual_iter_from_namespace(
    namespace: Mapping[str, Any],
    *,
    result_type: type,
    nestor_external_only_step_func: Callable[..., Any],
    residual_fsq_from_norms_func: Callable[..., Any],
    device_get_floats_func: Callable[..., tuple[float, ...]],
    residual_convergence_flags_func: Callable[..., tuple[bool, bool, float]],
    residual_iter_history_diagnostics_func: Callable[[Mapping[str, Any]], dict[str, Any]],
    attach_free_boundary_diagnostics: Callable[[Any], Any],
    return_final_force_payload: bool,
) -> Any:
    """Assemble the final residual-iteration result from host-loop locals.

    This function is intentionally namespace-based for now: it preserves the
    existing VMEC host-controller state names while giving the large non-scan
    loop a single, explicit finalization seam.
    """

    ns, returned_best_scored_state = _namespace_with_best_scored_state(namespace)
    timing_enabled = bool(ns["timing_enabled"])
    t_finalize_start = time.perf_counter() if timing_enabled else None
    final_freeb = final_free_boundary_residual_reports_from_namespace(
        ns,
        nestor_external_only_step_func=nestor_external_only_step_func,
        residual_fsq_from_norms_func=residual_fsq_from_norms_func,
        device_get_floats_func=device_get_floats_func,
    )
    converged_strict_final, converged_total_final, _ = residual_convergence_flags_func(
        fsqr=final_freeb["final_fsqr_report"],
        fsqz=final_freeb["final_fsqz_report"],
        fsql=final_freeb["final_fsql_report"],
        ftol=ns["ftol"],
        fsq_total_target=ns["fsq_total_target"],
    )
    startup_policy = ns.get("startup_policy")
    policy_defaults = {
        "host_update_assembly": False,
        "host_fsq1_norms_on_accelerator": False,
        "host_residual_metrics_on_accelerator": False,
        "use_restart_triggers": True,
        "badjac_mode": "ptau",
        "badjac_state_probe": False,
        "badjac_initial_state_probe_iters": 0,
        "light_history": False,
    }

    def _policy_attr(name: str) -> Any:
        if startup_policy is not None:
            return getattr(startup_policy, name)
        return ns.get(name, policy_defaults[name])

    t_finalize_diag_build_start = time.perf_counter() if timing_enabled else None
    freeb_policy = ns.get("_freeb_policy")
    numpy_precond_policy = ns.get("_numpy_precond_policy")
    best_scored = ns.get("best_scored")
    best_scored = best_scored if isinstance(best_scored, Mapping) else {}
    return_best_scored_state = bool(ns.get("return_best_scored_state", best_scored.get("enabled", False)))
    update_delta_rms_final = _optional_float(ns.get("update_delta_rms"))
    if update_delta_rms_final is None:
        update_delta_rms_final = _optional_float(ns.get("update_delta_rms_j"))
    update_rms_final = _optional_float(ns.get("update_rms"))
    if update_rms_final is None:
        update_rms_final = _optional_float(ns.get("update_rms_j"))
    diag: dict[str, Any] = {
        "ftol": ns["ftol"],
        "requested_ftol": float(ns["ftol"]),
        "gamma": ns["gamma"],
        "step_size": float(ns["step_size"]),
        "precond_radial_alpha": float(ns["precond_radial_alpha"]),
        "precond_lambda_alpha": float(ns["precond_lambda_alpha"]),
        "strict_update": bool(ns["strict_update"]),
        "host_update_assembly": bool(_policy_attr("host_update_assembly")),
        "host_fsq1_norms_on_accelerator": bool(_policy_attr("host_fsq1_norms_on_accelerator")),
        "host_residual_metrics_on_accelerator": bool(_policy_attr("host_residual_metrics_on_accelerator")),
        "jit_strict_update_enabled": (
            bool(getattr(freeb_policy, "jit_strict_update_enabled"))
            if freeb_policy is not None
            else bool(ns.get("jit_strict_update_enabled", False))
        ),
        "jit_strict_update_work": (
            int(getattr(freeb_policy, "update_work")) if freeb_policy is not None else int(ns.get("update_work", 0))
        ),
        "jit_strict_update_cpu_work_limit": (
            int(getattr(freeb_policy, "cpu_work_limit"))
            if freeb_policy is not None
            else int(ns.get("cpu_work_limit", 0))
        ),
        "numpy_preconditioner_apply": bool(ns.get("_use_numpy_preconditioner_apply", False)),
        "numpy_preconditioner_apply_mode_count": (
            int(getattr(numpy_precond_policy, "mode_count")) if numpy_precond_policy is not None else 0
        ),
        "numpy_preconditioner_apply_max_iter_cutoff": (
            int(getattr(numpy_precond_policy, "max_iter_cutoff")) if numpy_precond_policy is not None else 0
        ),
        "numpy_preconditioner_apply_min_mode_count": (
            int(getattr(numpy_precond_policy, "min_mode_count")) if numpy_precond_policy is not None else 0
        ),
        "numpy_force_fast_path": bool(ns.get("use_numpy_force_fast_path", False)),
        "numpy_force_fast_path_active": ns.get("_compute_forces_np") is not None,
        "numpy_force_fast_path_max_iter": int(ns.get("numpy_force_max_iter", 0)),
        "reference_mode": bool(ns["reference_mode"]),
        "use_restart_triggers": bool(_policy_attr("use_restart_triggers")),
        "use_direct_fallback": bool(ns["use_direct_fallback"]),
        "max_update_rms": float(ns["max_update_rms"]),
        "return_best_scored_state": return_best_scored_state,
        "returned_best_scored_state": bool(returned_best_scored_state),
        "best_scored_iter": best_scored.get("iter"),
        "best_scored_fsq": _optional_float(best_scored.get("fsq")),
        "best_scored_fsqr": _optional_float(best_scored.get("fsqr")),
        "best_scored_fsqz": _optional_float(best_scored.get("fsqz")),
        "best_scored_fsql": _optional_float(best_scored.get("fsql")),
        "best_scored_component_max": _optional_float(best_scored.get("component_max")),
        "best_scored_full_boundary_count": int(best_scored.get("full_boundary_count", 0)),
        "best_scored_fresh_boundary_count": int(best_scored.get("fresh_boundary_count", 0)),
        "update_delta_rms": update_delta_rms_final,
        "update_delta_to_velocity_rms_ratio": (
            None
            if update_delta_rms_final is None
            or update_rms_final is None
            or abs(float(update_rms_final)) <= 0.0
            else float(update_delta_rms_final) / abs(float(update_rms_final))
        ),
        "converged": bool(ns["converged"]),
        "converged_strict": bool(converged_strict_final),
        "converged_by_total_fsq": bool(converged_total_final),
        "final_fsqr": float(final_freeb["final_fsqr_report"]),
        "final_fsqz": float(final_freeb["final_fsqz_report"]),
        "final_fsql": float(final_freeb["final_fsql_report"]),
        "pre_update_final_fsqr": float(ns["fsqr_f"]),
        "pre_update_final_fsqz": float(ns["fsqz_f"]),
        "pre_update_final_fsql": float(ns["fsql_f"]),
        "final_residual_recomputed_on_accepted_state": bool(final_freeb["final_residual_recomputed"]),
        "badjac_use_state": bool(ns["badjac_use_state"]),
        "badjac_mode": _policy_attr("badjac_mode"),
        "badjac_state_probe": bool(_policy_attr("badjac_state_probe")),
        "badjac_initial_state_probe_iters": int(_policy_attr("badjac_initial_state_probe_iters")),
        "light_history": bool(_policy_attr("light_history")),
        "resume_state_mode": str(ns["resume_state_mode"]),
        "fsq_total_target": ns["fsq_total_target"],
        "ijacob": int(ns["ijacob"]),
        "bad_resets": int(ns["bad_resets"]),
        "setup_axis_reset_applied": bool(ns.get("setup_axis_reset_applied", False)),
        "setup_axis_reset_done": bool(ns.get("axis_reset_done", False)),
        "setup_axis_force_probe_available": ns.get("setup_axis_force_probe") is not None,
        "setup_axis_force_probe_reused": bool(ns.get("setup_axis_force_probe_reused", False)),
        "iter1_final": int(ns["iter1"]),
        "res0": float(ns["res0"]),
        **residual_iter_history_diagnostics_func(ns),
        "free_boundary": {
            "enabled": bool(ns["free_boundary_enabled"]),
            "nvacskip": int(ns["freeb_nvacskip"]),
            "nvskip0": int(ns["freeb_nvskip0"]),
            "ivac": int(ns["freeb_ivac"]),
            "ivacskip": int(ns["freeb_ivacskip"]),
            "couple_edge": bool(ns["freeb_couple_edge"]),
            "nestor_model": str(final_freeb["final_nestor_model"]),
            "vacuum_stub": bool(final_freeb["final_vacuum_stub"]),
            "activate_fsq": (
                None
                if ns["free_boundary_activate_fsq"] is None
                else float(ns["free_boundary_activate_fsq"])
            ),
            "plascur": float(ns["freeb_plascur"]),
            "last_nestor_diagnostics": dict(final_freeb["final_nestor_diagnostics"]),
            "final_nestor_recompute_attempted": bool(final_freeb["final_nestor_recompute_attempted"]),
            "final_nestor_recompute_failed": bool(final_freeb["final_nestor_recompute_failed"]),
            "final_nestor_sample_time_s": float(final_freeb["final_nestor_sample_time_s"]),
            "final_nestor_solve_time_s": float(final_freeb["final_nestor_solve_time_s"]),
            "edge_control_projection": {
                **dict(ns.get("freeb_edge_control_projection_info", {"enabled": False, "reason": "not_requested"})),
                "apply_count": int(ns.get("freeb_edge_control_projection_apply_count", 0)),
                "delta_projection_count": int(
                    ns.get(
                        "freeb_edge_control_projection_delta_projection_count",
                        getattr(ns.get("freeb_edge_control_projector"), "delta_projection_count", 0),
                    )
                ),
                "zero_velocity_count": int(ns.get("freeb_edge_control_projection_zero_velocity_count", 0)),
                "state_residual": _edge_control_state_residual_payload(ns),
                "update_direction": _edge_control_update_direction_payload(ns),
            },
        },
    }
    diag = attach_residual_iter_timing_diagnostics(
        diag,
        ns["timing_stats"],
        timing_enabled=timing_enabled,
        timing_detail_enabled=bool(ns["timing_detail_enabled"]),
        finalize_diag_build_start=t_finalize_diag_build_start,
        iteration_loop_start=ns["t_iteration_loop_start"],
        finalize_start=t_finalize_start,
        solve_wall_start=float(ns["_solve_wall_start"]),
    )
    diag["resume_state"] = build_residual_iter_resume_state_from_namespace(
        ns,
        resume_state_mode=str(ns["resume_state_mode"]),
    )
    history = ns.get("history_lists", ns).__getitem__
    return finalize_residual_iter_result(
        result_type=result_type,
        state=ns["state"],
        w_history=history("w_history"),
        fsqr2_history=history("fsqr2_history"),
        fsqz2_history=history("fsqz2_history"),
        fsql2_history=history("fsql2_history"),
        grad_rms_history=history("grad_rms_history"),
        step_history=history("step_history"),
        diagnostics=diag,
        attach_free_boundary_diagnostics=attach_free_boundary_diagnostics,
        return_final_force_payload=bool(return_final_force_payload),
        converged=bool(ns["converged"]),
        final_force_payload=ns["k"],
    )


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
