"""Runtime helper seams for ``solve_fixed_boundary_residual_iter``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np


def _ptau_dump_enabled(*, dump_ptau_env: str, dump_dir: str) -> bool:
    return str(dump_ptau_env).strip() not in ("", "0") and bool(str(dump_dir).strip())


def _format_ptau_dump_row(
    *,
    iter_idx: int,
    ptau_min: float,
    ptau_max: float,
    tau_min_state: float | None,
    tau_max_state: float | None,
    badjac_ptau: bool | None,
    badjac_state: bool | None,
    badjac_used: bool,
    mode: str,
    label: str,
) -> str:
    return (
        f"{int(iter_idx)} {label} {mode} "
        f"{float(ptau_min):.16e} {float(ptau_max):.16e} "
        f"{float(tau_min_state if tau_min_state is not None else float('nan')):.16e} "
        f"{float(tau_max_state if tau_max_state is not None else float('nan')):.16e} "
        f"{int(badjac_ptau) if badjac_ptau is not None else -1} "
        f"{int(badjac_state) if badjac_state is not None else -1} "
        f"{int(bool(badjac_used))}\n"
    )


def _maybe_dump_ptau(
    *,
    iter_idx: int,
    ptau_min: float,
    ptau_max: float,
    tau_min_state: float | None,
    tau_max_state: float | None,
    badjac_ptau: bool | None,
    badjac_state: bool | None,
    badjac_used: bool,
    mode: str,
    label: str,
    dump_ptau_env: str,
    dump_dir: str,
) -> bool:
    """Append a ptau diagnostic row, preserving solve.py's best-effort behavior."""

    if not _ptau_dump_enabled(dump_ptau_env=dump_ptau_env, dump_dir=dump_dir):
        return False
    try:
        path = Path(str(dump_dir).strip()) / "ptau_minmax.log"
        if not path.exists():
            with path.open("w", encoding="utf-8") as f:
                f.write("iter label mode ptau_min ptau_max state_min state_max bad_ptau bad_state bad_used\n")
        with path.open("a", encoding="utf-8") as f:
            f.write(
                _format_ptau_dump_row(
                    iter_idx=iter_idx,
                    ptau_min=ptau_min,
                    ptau_max=ptau_max,
                    tau_min_state=tau_min_state,
                    tau_max_state=tau_max_state,
                    badjac_ptau=badjac_ptau,
                    badjac_state=badjac_state,
                    badjac_used=badjac_used,
                    mode=mode,
                    label=label,
                )
            )
    except Exception:
        return False
    return True


def _scan_block_until_ready(
    value: Any,
    *,
    block_until_ready: Callable[[Any], Any],
    tree_map: Callable[[Callable[[Any], Any], Any], Any],
) -> Any:
    """Synchronize a JAX value, falling back to per-leaf ``block_until_ready``."""

    try:
        return block_until_ready(value)
    except Exception:
        return tree_map(
            lambda a: a.block_until_ready() if hasattr(a, "block_until_ready") else a,
            value,
        )


def _scan_device_run_ready(
    *,
    start: float | None,
    value: Any,
    scan_timing_enabled: bool,
    perf_counter: Callable[[], float],
    block_until_ready: Callable[[Any], Any],
    tree_map: Callable[[Callable[[Any], Any], Any], Any],
    record_ready: Callable[..., bool],
    stats: dict[str, float | int],
    cache_status: str | None = None,
) -> Any:
    """Block for scan completion and record dispatch/ready timing when enabled."""

    if not bool(scan_timing_enabled) or start is None:
        return value
    dispatch_done = perf_counter()
    value = _scan_block_until_ready(
        value,
        block_until_ready=block_until_ready,
        tree_map=tree_map,
    )
    ready_done = perf_counter()
    record_ready(
        start=start,
        dispatch_done=dispatch_done,
        ready_done=ready_done,
        stats=stats,
        cache_status=cache_status,
    )
    return value


def _converged_residuals_scan_fast(
    fsqr: Any,
    fsqz: Any,
    fsql: Any,
    *,
    ftol: Any,
    fsq_total_target: Any | None,
) -> Any:
    strict = (fsqr <= ftol) & (fsqz <= ftol) & (fsql <= ftol)
    if fsq_total_target is None:
        return strict
    return strict | ((fsqr + fsqz + fsql) <= fsq_total_target)


def _vmec_freeb_plascur_from_bcovar(
    bc_obj: Any,
    fallback: float,
    *,
    plascur_edge_from_bcovar: Callable[..., Any],
    trig: Any,
    wout: Any,
    s: Any,
) -> float:
    """Best-effort VMEC ``ctor`` proxy used by NESTOR."""

    try:
        ctor = plascur_edge_from_bcovar(
            bc=bc_obj,
            trig=trig,
            wout=wout,
            s=s,
        )
        ctor_f = float(np.asarray(ctor))
        if np.isfinite(ctor_f):
            return float(ctor_f)
    except Exception:
        pass
    return float(fallback)


def _scan_print_uses_debug_print(*, scan_print_mode: str, debug_print_fn: Any) -> bool:
    return str(scan_print_mode) == "debug_print" and debug_print_fn is not None


def _scan_print_uses_debug_callback(*, scan_print_mode: str, debug_module: Any) -> bool:
    return str(scan_print_mode) == "debug_callback" and debug_module is not None


def _scan_print_uses_io_callback(*, scan_print_mode: str, io_callback_fn: Any) -> bool:
    return str(scan_print_mode) == "io_callback" and io_callback_fn is not None


def _nonscan_state_debug_payload(
    *,
    state: Any,
    state_checkpoint: Any,
    gcr2: Any,
    gcz2: Any,
    gcl2: Any,
    norms_used: Any,
) -> dict[str, float]:
    """Return host scalars for the optional non-scan state debug row."""

    gcr2_val = float(np.asarray(gcr2))
    gcz2_val = float(np.asarray(gcz2))
    gcl2_val = float(np.asarray(gcl2))
    fn_val = float(np.asarray(norms_used.fnorm))
    r1_val = float(np.asarray(norms_used.r1))
    rcos_sum = float(np.sum(np.asarray(state.Rcos)))
    zsin_sum = float(np.sum(np.asarray(state.Zsin)))
    lsin_sum = float(np.sum(np.asarray(state.Lsin)))
    rcos_ck = float(np.sum(np.asarray(state_checkpoint.Rcos)))
    zsin_ck = float(np.sum(np.asarray(state_checkpoint.Zsin)))
    lsin_ck = float(np.sum(np.asarray(state_checkpoint.Lsin)))
    fsqr_dbg = float(np.asarray(norms_used.r1 * norms_used.fnorm * gcr2))
    fsqz_dbg = float(np.asarray(norms_used.r1 * norms_used.fnorm * gcz2))
    fsql_dbg = float(np.asarray(norms_used.fnormL * gcl2))
    return {
        "gcr2": gcr2_val,
        "gcz2": gcz2_val,
        "gcl2": gcl2_val,
        "fnorm": fn_val,
        "r1": r1_val,
        "rcos_sum": rcos_sum,
        "zsin_sum": zsin_sum,
        "lsin_sum": lsin_sum,
        "rcos_ck": rcos_ck,
        "zsin_ck": zsin_ck,
        "lsin_ck": lsin_ck,
        "fsqr": fsqr_dbg,
        "fsqz": fsqz_dbg,
        "fsql": fsql_dbg,
    }


def _format_nonscan_state_debug_row(*, iter2: int, payload: dict[str, float]) -> str:
    return (
        f"[nonscan-state] iter={int(iter2)} rcos_sum={payload['rcos_sum']:.6e} "
        f"zsin_sum={payload['zsin_sum']:.6e} lsin_sum={payload['lsin_sum']:.6e} "
        f"rcos_ck={payload['rcos_ck']:.6e} zsin_ck={payload['zsin_ck']:.6e} "
        f"lsin_ck={payload['lsin_ck']:.6e} "
        f"gcr2={payload['gcr2']:.6e} gcz2={payload['gcz2']:.6e} gcl2={payload['gcl2']:.6e} "
        f"fnorm={payload['fnorm']:.6e} r1={payload['r1']:.6e} "
        f"fsqr={payload['fsqr']:.6e} fsqz={payload['fsqz']:.6e} fsql={payload['fsql']:.6e}"
    )


def _maybe_print_nonscan_state_debug(
    *,
    debug_iter_env: str,
    iter2: int,
    state: Any,
    state_checkpoint: Any,
    gcr2: Any,
    gcz2: Any,
    gcl2: Any,
    norms_used: Any,
    print_fn: Callable[..., Any],
) -> bool:
    """Emit the optional non-scan debug row and swallow diagnostics failures."""

    if not debug_iter_env:
        return False
    try:
        debug_iter = int(debug_iter_env)
    except Exception:
        debug_iter = -1
    if debug_iter <= 0 or int(iter2) != debug_iter:
        return False
    try:
        payload = _nonscan_state_debug_payload(
            state=state,
            state_checkpoint=state_checkpoint,
            gcr2=gcr2,
            gcz2=gcz2,
            gcl2=gcl2,
            norms_used=norms_used,
        )
        print_fn(_format_nonscan_state_debug_row(iter2=int(iter2), payload=payload), flush=True)
    except Exception:
        return False
    return True


_SETUP_PHASE_KEYS = (
    "setup_static_grid_rebuild",
    "setup_freeb_policy",
    "setup_boundary_profiles",
    "setup_cache_key_hash",
    "setup_ptau_constants",
    "setup_index_constants",
    "setup_update_constants",
)


def _residual_iter_timing_setup_scalars(timing_stats: dict[str, float]) -> tuple[float, float, float]:
    setup_phase_total = sum(float(timing_stats.get(key, 0.0)) for key in _SETUP_PHASE_KEYS)
    setup_unattributed = max(
        0.0,
        float(timing_stats["setup_total"]) - float(timing_stats["setup_axis_reset"]) - setup_phase_total,
    )
    setup_axis_reset_unattributed = max(
        0.0,
        float(timing_stats["setup_axis_reset"]) - float(timing_stats["setup_axis_reset_compute_forces"]),
    )
    loop_leaf_total = (
        float(timing_stats["iteration_prepare"])
        + float(timing_stats["compute_forces"])
        + float(timing_stats["iteration_residual_metrics"])
        + float(timing_stats["preconditioner"])
        + float(timing_stats.get("iteration_control", 0.0))
        + float(timing_stats["update"])
        + float(timing_stats["iteration_post_update"])
    )
    iteration_loop_unattributed = max(
        0.0,
        float(timing_stats["iteration_loop"]) - loop_leaf_total,
    )
    return setup_unattributed, setup_axis_reset_unattributed, iteration_loop_unattributed


def _build_residual_iter_timing_report(
    timing_stats: dict[str, float],
    *,
    solve_total_s: float,
    timing_detail_enabled: bool,
) -> dict[str, float | int]:
    setup_unattributed, setup_axis_reset_unattributed, iteration_loop_unattributed = (
        _residual_iter_timing_setup_scalars(timing_stats)
    )
    iters = max(int(timing_stats["iterations"]), 1)
    iteration_control_subtotal = (
        float(timing_stats.get("iteration_control_fsq1", 0.0))
        + float(timing_stats.get("iteration_control_badjac", 0.0))
        + float(timing_stats.get("iteration_control_vmec_time", 0.0))
        + float(timing_stats.get("iteration_control_restart", 0.0))
        + float(timing_stats.get("iteration_control_evolve", 0.0))
    )
    iteration_control_unattributed = max(
        0.0,
        float(timing_stats.get("iteration_control", 0.0)) - iteration_control_subtotal,
    )
    timing_report: dict[str, float | int] = {
        "iterations": int(timing_stats["iterations"]),
        "solve_total_s": float(solve_total_s),
        "setup_total_s": float(timing_stats["setup_total"]),
        "setup_static_grid_rebuild_s": float(timing_stats.get("setup_static_grid_rebuild", 0.0)),
        "setup_freeb_policy_s": float(timing_stats.get("setup_freeb_policy", 0.0)),
        "setup_boundary_profiles_s": float(timing_stats.get("setup_boundary_profiles", 0.0)),
        "setup_cache_key_hash_s": float(timing_stats.get("setup_cache_key_hash", 0.0)),
        "setup_ptau_constants_s": float(timing_stats.get("setup_ptau_constants", 0.0)),
        "setup_index_constants_s": float(timing_stats.get("setup_index_constants", 0.0)),
        "setup_update_constants_s": float(timing_stats.get("setup_update_constants", 0.0)),
        "setup_axis_reset_s": float(timing_stats["setup_axis_reset"]),
        "setup_axis_reset_compute_forces_s": float(timing_stats["setup_axis_reset_compute_forces"]),
        "setup_axis_reset_unattributed_s": float(setup_axis_reset_unattributed),
        "setup_unattributed_s": float(setup_unattributed),
        "iteration_loop_s": float(timing_stats["iteration_loop"]),
        "iteration_prepare_s": float(timing_stats["iteration_prepare"]),
        "compute_forces_s": float(timing_stats["compute_forces"]),
        "compute_forces_first_s": float(timing_stats["compute_forces_first"]),
        "compute_forces_rest_s": float(timing_stats["compute_forces_rest"]),
        "compute_forces_calls": int(timing_stats["compute_forces_calls"]),
        "force_eval_s": float(timing_stats["compute_forces"]),
        "force_eval_first_s": float(timing_stats["compute_forces_first"]),
        "force_eval_rest_s": float(timing_stats["compute_forces_rest"]),
        "force_eval_calls": int(timing_stats["compute_forces_calls"]),
        "iteration_residual_metrics_s": float(timing_stats["iteration_residual_metrics"]),
        "preconditioner_s": float(timing_stats["preconditioner"]),
        "iteration_control_s": float(timing_stats.get("iteration_control", 0.0)),
        "iteration_control_fsq1_s": float(timing_stats.get("iteration_control_fsq1", 0.0)),
        "iteration_control_badjac_s": float(timing_stats.get("iteration_control_badjac", 0.0)),
        "iteration_control_vmec_time_s": float(timing_stats.get("iteration_control_vmec_time", 0.0)),
        "iteration_control_restart_s": float(timing_stats.get("iteration_control_restart", 0.0)),
        "iteration_control_evolve_s": float(timing_stats.get("iteration_control_evolve", 0.0)),
        "iteration_control_unattributed_s": float(iteration_control_unattributed),
        "precond_refresh_s": float(timing_stats["precond_refresh"]),
        "update_s": float(timing_stats["update"]),
        "update_state_s": float(timing_stats["update_state"]),
        "update_trace_build_s": float(timing_stats["update_trace_build"]),
        "update_trace_finalize_s": float(timing_stats["update_trace_finalize"]),
        "iteration_post_update_s": float(timing_stats["iteration_post_update"]),
        "iteration_loop_unattributed_s": float(iteration_loop_unattributed),
        "finalize_s": float(timing_stats["finalize"]),
        "setup_per_iter_s": float(timing_stats["setup_total"]) / iters,
        "setup_static_grid_rebuild_per_iter_s": float(timing_stats.get("setup_static_grid_rebuild", 0.0)) / iters,
        "setup_freeb_policy_per_iter_s": float(timing_stats.get("setup_freeb_policy", 0.0)) / iters,
        "setup_boundary_profiles_per_iter_s": float(timing_stats.get("setup_boundary_profiles", 0.0)) / iters,
        "setup_cache_key_hash_per_iter_s": float(timing_stats.get("setup_cache_key_hash", 0.0)) / iters,
        "setup_ptau_constants_per_iter_s": float(timing_stats.get("setup_ptau_constants", 0.0)) / iters,
        "setup_index_constants_per_iter_s": float(timing_stats.get("setup_index_constants", 0.0)) / iters,
        "setup_update_constants_per_iter_s": float(timing_stats.get("setup_update_constants", 0.0)) / iters,
        "iteration_prepare_per_iter_s": float(timing_stats["iteration_prepare"]) / iters,
        "compute_forces_per_iter_s": float(timing_stats["compute_forces"]) / iters,
        "force_eval_per_iter_s": float(timing_stats["compute_forces"]) / iters,
        "iteration_residual_metrics_per_iter_s": float(timing_stats["iteration_residual_metrics"]) / iters,
        "preconditioner_per_iter_s": float(timing_stats["preconditioner"]) / iters,
        "iteration_control_per_iter_s": float(timing_stats.get("iteration_control", 0.0)) / iters,
        "iteration_control_fsq1_per_iter_s": float(timing_stats.get("iteration_control_fsq1", 0.0)) / iters,
        "iteration_control_badjac_per_iter_s": float(timing_stats.get("iteration_control_badjac", 0.0)) / iters,
        "iteration_control_vmec_time_per_iter_s": float(timing_stats.get("iteration_control_vmec_time", 0.0))
        / iters,
        "iteration_control_restart_per_iter_s": float(timing_stats.get("iteration_control_restart", 0.0)) / iters,
        "iteration_control_evolve_per_iter_s": float(timing_stats.get("iteration_control_evolve", 0.0)) / iters,
        "iteration_control_unattributed_per_iter_s": float(iteration_control_unattributed) / iters,
        "update_per_iter_s": float(timing_stats["update"]) / iters,
        "update_state_per_iter_s": float(timing_stats["update_state"]) / iters,
        "update_trace_build_per_iter_s": float(timing_stats["update_trace_build"]) / iters,
        "update_trace_finalize_per_iter_s": float(timing_stats["update_trace_finalize"]) / iters,
        "iteration_post_update_per_iter_s": float(timing_stats["iteration_post_update"]) / iters,
        "iteration_loop_unattributed_per_iter_s": float(iteration_loop_unattributed) / iters,
    }
    if timing_detail_enabled:
        main_force = float(timing_stats.get("compute_forces_main", timing_stats["compute_forces"]))
        auto_flip_force = float(timing_stats.get("compute_forces_auto_flip", 0.0))
        trial_force = float(timing_stats.get("compute_forces_trial", 0.0))
        backtracking_force = float(timing_stats.get("compute_forces_backtracking", 0.0))
        all_force = main_force + auto_flip_force + trial_force + backtracking_force
        timing_report.update(
            {
                "compute_forces_main_s": main_force,
                "compute_forces_main_calls": int(timing_stats.get("compute_forces_main_calls", 0)),
                "compute_forces_auto_flip_s": auto_flip_force,
                "compute_forces_auto_flip_calls": int(timing_stats.get("compute_forces_auto_flip_calls", 0)),
                "compute_forces_trial_s": trial_force,
                "compute_forces_trial_calls": int(timing_stats.get("compute_forces_trial_calls", 0)),
                "compute_forces_backtracking_s": backtracking_force,
                "compute_forces_backtracking_calls": int(timing_stats.get("compute_forces_backtracking_calls", 0)),
                "force_eval_all_s": all_force,
                "force_eval_all_calls": int(
                    timing_stats.get("compute_forces_main_calls", 0)
                    + timing_stats.get("compute_forces_auto_flip_calls", 0)
                    + timing_stats.get("compute_forces_trial_calls", 0)
                    + timing_stats.get("compute_forces_backtracking_calls", 0)
                ),
                "force_eval_extra_s": all_force - main_force,
                "precond_apply_s": float(timing_stats["precond_apply"]),
                "precond_mode_scale_s": float(timing_stats["precond_mode_scale"]),
                "compute_forces_main_per_iter_s": main_force / iters,
                "force_eval_all_per_iter_s": all_force / iters,
                "force_eval_extra_per_iter_s": (all_force - main_force) / iters,
                "precond_apply_per_iter_s": float(timing_stats["precond_apply"]) / iters,
                "precond_mode_scale_per_iter_s": float(timing_stats["precond_mode_scale"]) / iters,
            }
        )
    return timing_report


def _format_residual_iter_timing_message(
    timing_report: dict[str, float | int],
    *,
    timing_detail_enabled: bool,
) -> str:
    detail_text = ""
    if timing_detail_enabled:
        detail_text = (
            f"force_main={float(timing_report['compute_forces_main_s']):.3e}s "
            f"force_extra={float(timing_report['force_eval_extra_s']):.3e}s "
            f"precond_apply={float(timing_report['precond_apply_s']):.3e}s "
            f"precond_mode={float(timing_report['precond_mode_scale_s']):.3e}s "
            f"control={float(timing_report['iteration_control_s']):.3e}s "
            f"control_fsq1={float(timing_report['iteration_control_fsq1_s']):.3e}s "
            f"control_badjac={float(timing_report['iteration_control_badjac_s']):.3e}s "
            f"control_vmec={float(timing_report['iteration_control_vmec_time_s']):.3e}s "
            f"control_restart={float(timing_report['iteration_control_restart_s']):.3e}s "
            f"control_evolve={float(timing_report['iteration_control_evolve_s']):.3e}s "
        )
    return (
        "[vmec_jax timing] "
        f"iters={int(timing_report['iterations'])} "
        f"compute_forces={float(timing_report['compute_forces_s']):.3e}s "
        f"precond={float(timing_report['preconditioner_s']):.3e}s "
        f"precond_refresh={float(timing_report['precond_refresh_s']):.3e}s "
        f"{detail_text}"
        f"update={float(timing_report['update_s']):.3e}s "
        f"update_state={float(timing_report['update_state_s']):.3e}s "
        f"trace_build={float(timing_report['update_trace_build_s']):.3e}s "
        f"trace_final={float(timing_report['update_trace_finalize_s']):.3e}s "
        f"(per-iter: {float(timing_report['compute_forces_per_iter_s']):.3e}, "
        f"{float(timing_report['preconditioner_per_iter_s']):.3e}, "
        f"{float(timing_report['update_per_iter_s']):.3e})"
    )


def _build_resume_state_base(
    *,
    time_step: float,
    inv_tau: list[float],
    fsq_prev: float,
    fsq0_prev: float,
    flip_sign: float,
    iter1: int,
    last_iter2: int,
    ijacob: int,
    bad_resets: int,
    res0: float,
    res1: float,
    prev_rz_fsq: float,
    bad_growth_streak: int,
    huge_force_restart_count: int,
    vmec2000_cache_valid: bool,
    freeb_ivac: int,
    freeb_ivacskip: int,
    freeb_nvacskip: int,
    freeb_nvskip0: int,
    freeb_last_model: str,
    freeb_nestor_runtime: Any,
) -> dict[str, Any]:
    return {
        "time_step": float(time_step),
        "inv_tau": list(inv_tau),
        "fsq_prev": float(fsq_prev),
        "fsq0_prev": float(fsq0_prev),
        "flip_sign": float(flip_sign),
        "iter1": int(iter1),
        "iter_offset": int(last_iter2),
        "ijacob": int(ijacob),
        "bad_resets": int(bad_resets),
        "res0": float(res0),
        "res1": float(res1),
        "prev_rz_fsq": float(prev_rz_fsq),
        "bad_growth_streak": int(bad_growth_streak),
        "huge_force_restart_count": int(huge_force_restart_count),
        "vmec2000_cache_valid": bool(vmec2000_cache_valid),
        "freeb_ivac": int(freeb_ivac),
        "freeb_ivacskip": int(freeb_ivacskip),
        "freeb_nvacskip": int(freeb_nvacskip),
        "freeb_nvskip0": int(freeb_nvskip0),
        "freeb_model": str(freeb_last_model),
        "freeb_nestor_update_count": (
            0 if freeb_nestor_runtime is None else int(getattr(freeb_nestor_runtime, "update_count", 0))
        ),
        "freeb_nestor_reuse_count": (
            0 if freeb_nestor_runtime is None else int(getattr(freeb_nestor_runtime, "reuse_count", 0))
        ),
    }
