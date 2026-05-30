"""Pure planning helpers for the VMEC2000 scan solve path."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, NamedTuple

from .solve_residual_iter_policy import Vmec2000ScanOptions


SCAN_TIMING_KEYS: tuple[str, ...] = (
    "scan_setup_s",
    "scan_initial_compute_forces_s",
    "scan_axis_reset_compute_forces_s",
    "scan_run_setup_s",
    "scan_runner_cache_lookup_s",
    "scan_runner_cache_build_s",
    "scan_preflight_s",
    "scan_device_run_s",
    "scan_device_dispatch_s",
    "scan_device_ready_s",
    "scan_runner_cache_hit_device_run_s",
    "scan_runner_cache_hit_dispatch_s",
    "scan_runner_cache_hit_ready_s",
    "scan_runner_cache_miss_device_run_s",
    "scan_runner_cache_miss_dispatch_s",
    "scan_runner_cache_miss_ready_s",
    "scan_runner_cache_bypass_device_run_s",
    "scan_runner_cache_bypass_dispatch_s",
    "scan_runner_cache_bypass_ready_s",
    "scan_host_materialize_s",
    "scan_postprocess_s",
)

SCAN_TIMING_COUNT_KEYS: tuple[str, ...] = (
    "scan_runner_cache_hit_count",
    "scan_runner_cache_miss_count",
    "scan_runner_cache_bypass_count",
)


class ScanRunFlags(NamedTuple):
    state_only_scan: bool
    scan_fallback_enabled_run: bool
    force_chunked_scan_run: bool


class ScanPreflightPlan(NamedTuple):
    preflight_iters: int
    preflight_default: str


class ScanIterationPlan(NamedTuple):
    extra_iters: int
    extra_iters_default: str
    max_iter_scan: int
    preflight_iters: int
    max_iter_tail: int


def scan_timing_enabled(env_value: str) -> bool:
    """Return whether VMEC scan timing diagnostics should be collected."""
    return str(env_value).strip().lower() not in ("", "0", "false", "no")


def new_scan_timing_stats() -> dict[str, float | int]:
    """Return a fresh timing accumulator with the solver's stable key set."""
    return {
        **{key: 0.0 for key in SCAN_TIMING_KEYS},
        **{key: 0 for key in SCAN_TIMING_COUNT_KEYS},
    }


def build_scan_timing_report(
    *,
    iterations: int,
    stats: dict[str, float | int],
    scan_total_s: float,
) -> dict[str, float | int]:
    """Build the public timing diagnostic report from timing accumulators."""
    normalized = {key: float(stats.get(key, 0.0)) for key in SCAN_TIMING_KEYS}
    counts = {key: int(stats.get(key, 0)) for key in SCAN_TIMING_COUNT_KEYS}
    dynamic_counts = {
        key: int(value)
        for key, value in stats.items()
        if str(key).startswith("scan_runner_cache_miss_category_") and str(key).endswith("_count")
    }
    scan_leaf_total_s = sum(
        value
        for key, value in normalized.items()
        if key not in ("scan_device_dispatch_s", "scan_device_ready_s")
    )
    total = float(scan_total_s)
    return {
        "iterations": int(iterations),
        "scan_total_s": total,
        **normalized,
        **counts,
        **dynamic_counts,
        "scan_cold_cache_miss_s": float(normalized["scan_runner_cache_miss_device_run_s"]),
        "scan_cold_cache_miss_ready_s": float(normalized["scan_runner_cache_miss_ready_s"]),
        "scan_cache_build_wrapper_s": float(normalized["scan_runner_cache_build_s"]),
        "scan_unattributed_s": max(0.0, total - float(scan_leaf_total_s)),
    }


def validate_vmec2000_scan_guards(
    *,
    backtracking: bool,
    limit_dt_from_force: bool,
    limit_update_rms: bool,
    use_direct_fallback: bool,
    reference_mode: bool,
    strict_update: bool,
    auto_flip_force: bool,
) -> None:
    """Validate options that the VMEC2000 scan implementation cannot support."""
    if backtracking or limit_dt_from_force or limit_update_rms or use_direct_fallback or reference_mode:
        raise ValueError(
            "vmec2000 scan requires backtracking=False, limit_dt_from_force=False, "
            "limit_update_rms=False, use_direct_fallback=False, reference_mode=False."
        )
    if not bool(strict_update):
        raise ValueError("vmec2000 scan requires strict_update=True.")
    if bool(auto_flip_force):
        raise ValueError("vmec2000 scan does not yet support auto_flip_force=True.")


def resolve_scan_run_flags(
    *,
    state_only: bool,
    scan_differentiated: bool,
    scan_fallback_enabled: bool,
    force_chunked_scan: bool,
) -> ScanRunFlags:
    """Resolve host-only run flags that depend on tracing and state-only mode."""
    state_only_scan = bool(state_only)
    differentiated = bool(scan_differentiated)
    return ScanRunFlags(
        state_only_scan=state_only_scan,
        scan_fallback_enabled_run=bool(scan_fallback_enabled) and (not differentiated) and (not state_only_scan),
        force_chunked_scan_run=bool(force_chunked_scan) and (not differentiated),
    )


def apply_state_only_scan_options(
    options: Vmec2000ScanOptions,
    *,
    state_only_scan: bool,
) -> Vmec2000ScanOptions:
    """Apply the state-only scan history/printing overrides."""
    if not bool(state_only_scan):
        return options
    return replace(
        options,
        scan_light=False,
        scan_minimal=True,
        scan_collect_scalars=False,
        scan_collect_print=False,
        print_in_scan=False,
        chunked_print=False,
    )


def normalize_scan_print_mode(*, scan_print_mode: str, io_callback_available: bool) -> str:
    """Normalize the scan print mode after probing optional callback support."""
    mode = str(scan_print_mode).strip().lower()
    if mode == "io_callback" and not bool(io_callback_available):
        return "debug_print"
    if mode not in ("debug_print", "debug_callback", "io_callback"):
        return "debug_print"
    return mode


def scan_jit_forces_enabled(*, env_value: str | None, jit_forces: bool) -> bool:
    """Resolve VMEC_JAX_SCAN_JIT_FORCES while preserving the legacy default."""
    if env_value is None:
        return bool(jit_forces)
    return str(env_value).strip().lower() not in ("", "0", "false", "no")


def scan_jit_preflight_enabled(
    *,
    env_value: str | None,
    backend_name: str,
    scan_differentiated: bool,
) -> bool:
    """Resolve whether scan preflight should use a cached one-step JIT runner."""

    if env_value is not None:
        return str(env_value).strip().lower() not in ("", "0", "false", "no")
    backend = str(backend_name).strip().lower()
    return backend not in ("", "cpu") and (not bool(scan_differentiated))


def resolve_scan_preflight_iters(
    *,
    jit_forces_scan: bool,
    vmec2000_control: bool,
    max_iter: int,
    axis_reset_repeat: bool,
    preflight_env: str | None,
) -> ScanPreflightPlan:
    """Resolve the single-step scan preflight plan from env/control flags."""
    preflight_default = "0" if bool(jit_forces_scan) else "1"
    env_value = preflight_default if preflight_env is None else str(preflight_env).strip()
    enabled = str(env_value).strip().lower() not in ("", "0", "false", "no")
    if enabled and bool(vmec2000_control) and int(max_iter) > 0:
        try:
            preflight_iters = max(1, int(env_value))
        except Exception:
            preflight_iters = 1
    else:
        preflight_iters = 0
    if bool(axis_reset_repeat) and int(max_iter) > 0:
        preflight_iters = max(1, int(preflight_iters))
    return ScanPreflightPlan(preflight_iters=int(preflight_iters), preflight_default=preflight_default)


def resolve_scan_iteration_plan(
    *,
    max_iter: int,
    preflight_iters: int,
    vmec2000_control: bool,
    extra_iters_env: str | None,
) -> ScanIterationPlan:
    """Resolve extra scan iterations and clamp preflight to the scan length."""
    extra_iters_default = "0" if bool(vmec2000_control) else "10"
    env_value = extra_iters_default if extra_iters_env is None else str(extra_iters_env).strip()
    try:
        extra_iters = max(0, int(env_value))
    except Exception:
        extra_iters = 0
    max_iter_scan = int(max_iter) + int(extra_iters)
    preflight_clamped = int(preflight_iters)
    if max_iter_scan <= 0:
        preflight_clamped = 0
    elif preflight_clamped > max_iter_scan:
        preflight_clamped = int(max_iter_scan)
    max_iter_tail = int(max_iter_scan) - int(preflight_clamped)
    return ScanIterationPlan(
        extra_iters=int(extra_iters),
        extra_iters_default=extra_iters_default,
        max_iter_scan=int(max_iter_scan),
        preflight_iters=int(preflight_clamped),
        max_iter_tail=int(max_iter_tail),
    )


def scan_chunk_settings(
    *,
    max_iter_scan: int,
    nstep_screen: int,
    need_print: bool,
    lthreed: bool,
    backend_name: str,
    chunk_size_env: str,
    spectral_mode_count: int | None = None,
) -> tuple[int, bool]:
    """Resolve scan chunk size without reading process environment."""
    chunk_size_env = str(chunk_size_env).strip()
    backend = str(backend_name).strip().lower()
    low_mode_accelerator = (
        backend not in ("", "cpu")
        and not bool(need_print)
        and spectral_mode_count is not None
        and int(spectral_mode_count) <= 16
        and int(max_iter_scan) > 512
    )
    if chunk_size_env:
        try:
            chunk_size = max(1, int(chunk_size_env))
        except Exception:
            chunk_size = max(1, int(nstep_screen))
    elif (backend == "cpu") and (not bool(need_print)):
        chunk_size = max(1, int(max_iter_scan))
    elif low_mode_accelerator:
        # Fresh-process GPU profiles of low-mode QH warm starts are dominated
        # by the one large scan executable.  A fixed 256-iteration chunk keeps
        # the compiled body smaller and reusable inside the same solve.  Higher
        # mode-count cases keep the full chunk because launch overhead dominates
        # there.
        chunk_size = min(max(1, int(max_iter_scan)), 256)
    elif (backend != "cpu") and (not bool(need_print)):
        chunk_size = max(1, int(max_iter_scan))
    else:
        chunk_size = max(1, int(nstep_screen))
    cap_to_remaining = (not bool(need_print)) and (not low_mode_accelerator)
    return chunk_size, cap_to_remaining


def build_vmec2000_scan_cache_key(
    *,
    static_key: Any,
    wout_key: Any,
    edge_signature_key: Any,
    tomnsps_policy_key: Any = None,
    max_iter_tail: int,
    preflight_iters: int,
    iter_offset0: int,
    step_size: float,
    initial_flip_sign: float,
    lambda_update_scale: float,
    ftol: float,
    nstep_screen: int,
    use_restart_triggers: bool,
    vmecpp_restart: bool,
    scan_use_precomputed: bool = False,
    scan_use_lax_tridi: bool = False,
    scan_use_restart_payload: bool,
    stage_prev_fsq: float | None,
    stage_transition_factor: float,
    stage_transition_scale: float,
    jit_forces_scan: bool,
    state_only_scan: bool,
    scan_light: bool,
    scan_minimal: bool,
    scan_fallback_iters: int,
    scan_fallback_accept_frac: float,
    scan_fallback_fsq_factor: float,
    scan_fallback_badjac_limit: int,
    scan_fallback_fsq_abs: float,
) -> tuple[Any, ...]:
    """Construct the JIT-cache key for the VMEC2000 scan runner."""
    return (
        "vmec2000_scan_v5",
        static_key,
        wout_key,
        edge_signature_key,
        tomnsps_policy_key,
        int(max_iter_tail),
        int(preflight_iters),
        int(iter_offset0),
        float(step_size),
        float(initial_flip_sign),
        float(lambda_update_scale),
        float(ftol),
        int(nstep_screen),
        bool(use_restart_triggers),
        bool(vmecpp_restart),
        bool(scan_use_precomputed),
        bool(scan_use_lax_tridi),
        bool(scan_use_restart_payload),
        None if stage_prev_fsq is None else float(stage_prev_fsq),
        float(stage_transition_factor),
        float(stage_transition_scale),
        bool(jit_forces_scan),
        bool(state_only_scan),
        bool(scan_light),
        bool(scan_minimal),
        int(scan_fallback_iters),
        float(scan_fallback_accept_frac),
        float(scan_fallback_fsq_factor),
        int(scan_fallback_badjac_limit),
        float(scan_fallback_fsq_abs),
    )
