"""Pure planning helpers for the VMEC2000 scan solve path."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, NamedTuple

from .... import _solve_runtime
from ..residual.config import resolve_nstep_screen
from ..residual.policy import Vmec2000ScanOptions, vmec2000_scan_options_from_env


SCAN_TIMING_KEYS: tuple[str, ...] = (
    "scan_setup_s",
    "scan_initial_compute_forces_s",
    "scan_axis_reset_compute_forces_s",
    "scan_run_setup_s",
    "scan_runner_cache_lookup_s",
    "scan_runner_cache_build_s",
    "scan_runner_explicit_lower_s",
    "scan_runner_explicit_compile_s",
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
    "scan_runner_explicit_compile_count",
    "scan_runner_explicit_compile_failure_count",
    "scan_runner_explicit_hlo_line_count",
    "scan_runner_explicit_hlo_instruction_count",
    "scan_runner_explicit_hlo_failure_count",
    "scan_runner_arg_leaf_count",
    "scan_runner_arg_array_leaf_count",
    "scan_runner_arg_scalar_leaf_count",
    "scan_runner_arg_array_nbytes",
    "scan_runner_arg_preconditioner_rz_mats_key_count",
    "scan_runner_arg_preconditioner_rz_mats_unexpected_key_count",
    "scan_runner_arg_preconditioner_rz_mats_missing_mandatory_key_count",
    "scan_runner_arg_preconditioner_rz_mats_compact_ok_count",
    "scan_history_none",
    "scan_history_leaf_count",
    "scan_history_array_leaf_count",
    "scan_history_scalar_leaf_count",
    "scan_history_array_nbytes",
)


class ScanRunFlags(NamedTuple):
    """Boolean flags that select the scan execution variant."""

    state_only_scan: bool
    scan_fallback_enabled_run: bool
    force_chunked_scan_run: bool


class ScanPreflightPlan(NamedTuple):
    """Preflight iteration budget before a chunked scan run."""

    preflight_iters: int
    preflight_default: str


class ScanIterationPlan(NamedTuple):
    """Iteration counts for the preflight, scan body, and tail phases."""

    extra_iters: int
    extra_iters_default: str
    max_iter_scan: int
    preflight_iters: int
    max_iter_tail: int


class ScanIterationRuntimePlan(NamedTuple):
    """Resolved scan iteration counts, offsets, and runner cache key."""

    preflight_iters: int
    max_iter_scan: int
    max_iter_tail: int
    iter_offset0: int
    iter_offset_preflight: int
    axis_reset_repeated: bool
    scan_cache_key: tuple[Any, ...]


class Vmec2000ScanSetup(NamedTuple):
    """Host-side scan setup resolved before entering the numerical controller."""

    state_only_scan: bool
    scan_fallback_enabled_run: bool
    force_chunked_scan_run: bool
    nstep_screen: int
    options: Vmec2000ScanOptions


class Vmec2000ControllerConstants(NamedTuple):
    """VMEC2000-style residual-controller constants used by scan and host paths."""

    preconditioner_update_interval: int
    restart_badjac_factor: float
    restart_badprog_factor: float
    vmec2000_fact: float
    ndamp: int


def _env_value(env: Mapping[str, str | None], name: str, default: str = "") -> str:
    value = env.get(name, default)
    return default if value is None else str(value)


def default_vmec2000_controller_constants() -> Vmec2000ControllerConstants:
    """Return VMEC2000 controller constants used historically by the solver."""

    return Vmec2000ControllerConstants(
        preconditioner_update_interval=25,
        restart_badjac_factor=0.9,
        restart_badprog_factor=1.03,
        vmec2000_fact=1.0e4,
        ndamp=10,
    )


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
    dynamic_counts.update(
        {
            key: int(value)
            for key, value in stats.items()
            if (
                str(key).startswith("scan_runner_explicit_compile_")
                or str(key).startswith("scan_runner_explicit_hlo_op_")
                or str(key).startswith("scan_runner_arg_path_")
                or str(key).startswith("scan_runner_arg_category_")
            )
            and (
                str(key).endswith("_count")
                or str(key).endswith("_array_nbytes")
            )
            and key not in SCAN_TIMING_COUNT_KEYS
        }
    )
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


def resolve_vmec2000_scan_setup(
    *,
    env: Mapping[str, str | None],
    state_only: bool,
    scan_differentiated: bool,
    scan_fallback_enabled: bool,
    force_chunked_scan: bool,
    indata_nstep: int,
    preconditioner_use_precomputed_tridi: bool | None,
    preconditioner_use_lax_tridi: bool | None,
    verbose: bool,
    vmec2000_control: bool,
    verbose_vmec2000_table: bool,
    light_history: bool,
    scan_minimal_default: bool | None,
    dump_any: bool,
    fsq_total_target: float | None,
    backend_name: str,
) -> Vmec2000ScanSetup:
    """Resolve scan options and state-only overrides in one pure setup step."""

    run_flags = resolve_scan_run_flags(
        state_only=bool(state_only),
        scan_differentiated=bool(scan_differentiated),
        scan_fallback_enabled=bool(scan_fallback_enabled),
        force_chunked_scan=bool(force_chunked_scan),
    )
    nstep_screen = resolve_nstep_screen(indata_nstep=int(indata_nstep), override_env="")

    tridi_precompute_env = _env_value(env, "VMEC_JAX_TRIDI_PRECOMPUTE", "")
    if preconditioner_use_precomputed_tridi is not None:
        tridi_precompute_env = "1" if bool(preconditioner_use_precomputed_tridi) else "0"
    tridi_solve_env = _env_value(env, "VMEC_JAX_TRIDI_SOLVE", "")
    if preconditioner_use_lax_tridi is not None:
        tridi_solve_env = "force" if bool(preconditioner_use_lax_tridi) else "0"

    options = vmec2000_scan_options_from_env(
        verbose=bool(verbose),
        vmec2000_control=bool(vmec2000_control),
        verbose_vmec2000_table=bool(verbose_vmec2000_table),
        light_history=bool(light_history),
        scan_minimal_default=scan_minimal_default,
        dump_any=bool(dump_any),
        fsq_total_target=fsq_total_target,
        backend_name=str(backend_name),
        force_chunked_scan_run=bool(run_flags.force_chunked_scan_run),
        scan_print_env=_env_value(env, "VMEC_JAX_SCAN_PRINT", "1"),
        scan_print_mode_env=_env_value(env, "VMEC_JAX_SCAN_PRINT_MODE", "debug_callback"),
        scan_print_ordered_env=_env_value(env, "VMEC_JAX_SCAN_PRINT_ORDERED", "0"),
        scan_print_chunked_env=_env_value(env, "VMEC_JAX_SCAN_PRINT_CHUNKED", "1"),
        scan_light_env=_env_value(env, "VMEC_JAX_SCAN_LIGHT", "0"),
        scan_minimal_env=_env_value(env, "VMEC_JAX_SCAN_MINIMAL", ""),
        scan_core_env=_env_value(env, "VMEC_JAX_SCAN_CORE", ""),
        scan_trace_env=_env_value(env, "VMEC_JAX_SCAN_TRACE", "0"),
        abort_scan_env=_env_value(env, "VMEC_JAX_SCAN_ABORT_ON_BADJAC", "0"),
        scan_precompute_env=_env_value(env, "VMEC_JAX_SCAN_PRECOND_PRECOMPUTE", ""),
        tridi_precompute_env=tridi_precompute_env,
        scan_lax_env=_env_value(env, "VMEC_JAX_SCAN_PRECOND_LAXTRIDI", ""),
        tridi_solve_env=tridi_solve_env,
        scan_restart_payload_env=_env_value(env, "VMEC_JAX_SCAN_RESTART_PAYLOAD", ""),
    )
    options = apply_state_only_scan_options(options, state_only_scan=bool(run_flags.state_only_scan))

    return Vmec2000ScanSetup(
        state_only_scan=bool(run_flags.state_only_scan),
        scan_fallback_enabled_run=bool(run_flags.scan_fallback_enabled_run),
        force_chunked_scan_run=bool(run_flags.force_chunked_scan_run),
        nstep_screen=int(nstep_screen),
        options=options,
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
    # Preflight is only executed for special controller states, such as an
    # initial-axis reset repeat.  Running that step eagerly on CPU can dominate
    # cold solves for high-mode LASYM cases because it evaluates the full scan
    # step outside the compiled runner.  Use the same cached one-step runner on
    # all ordinary backends, but keep traced/differentiated scans out of this
    # host cache path.
    return not bool(scan_differentiated)


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
    return _solve_runtime._scan_chunk_settings(
        max_iter_scan=max_iter_scan,
        nstep_screen=nstep_screen,
        need_print=need_print,
        lthreed=lthreed,
        backend_name=backend_name,
        chunk_size_env=chunk_size_env,
        spectral_mode_count=spectral_mode_count,
    )


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
    fsq_total_target: float | None,
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
    # State-only scan runners are used for residual-only trial solves.  The
    # fallback probe is disabled for that path by ``resolve_scan_run_flags``, so
    # the fallback integer gates must not fragment the cache.  Normal scan
    # runners keep these fields structural because they affect abort/fallback
    # semantics and VMEC2000-parity diagnostics.
    state_only_key = bool(state_only_scan)
    fallback_iters_key = 0 if state_only_key else int(scan_fallback_iters)
    fallback_badjac_key = (
        0
        if state_only_key or int(fallback_iters_key) <= 0
        else int(scan_fallback_badjac_limit)
    )
    # State-only scans return only the final state.  They explicitly disable
    # screen-output scalar sampling and history materialization in
    # ``apply_state_only_scan_options``, so display/history settings must not
    # split the compiled runner cache for optimizer trial solves.
    nstep_screen_key = 0 if state_only_key else int(nstep_screen)
    scan_light_key = False if state_only_key else bool(scan_light)
    scan_minimal_key = True if state_only_key else bool(scan_minimal)
    return (
        "vmec2000_scan_v10",
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
        fsq_total_target is not None,
        int(nstep_screen_key),
        bool(use_restart_triggers),
        bool(vmecpp_restart),
        bool(scan_use_precomputed),
        bool(scan_use_lax_tridi),
        bool(scan_use_restart_payload),
        stage_prev_fsq is not None,
        bool(jit_forces_scan),
        state_only_key,
        scan_light_key,
        scan_minimal_key,
        int(fallback_iters_key),
        int(fallback_badjac_key),
    )


def resolve_scan_iteration_runtime_plan(
    *,
    env: Mapping[str, str | None],
    jit_forces_scan: bool,
    vmec2000_control: bool,
    max_iter: int,
    axis_reset_repeat: bool,
    iter_offset0: int,
    static_key: Any,
    wout_key: Any,
    edge_signature_key: Any,
    step_size: float,
    initial_flip_sign: float,
    lambda_update_scale: float,
    ftol: float,
    fsq_total_target: float | None,
    nstep_screen: int,
    use_restart_triggers: bool,
    vmecpp_restart: bool,
    scan_use_precomputed: bool,
    scan_use_lax_tridi: bool,
    scan_use_restart_payload: bool,
    stage_prev_fsq: float | None,
    stage_transition_factor: float,
    stage_transition_scale: float,
    state_only_scan: bool,
    scan_light: bool,
    scan_minimal: bool,
    scan_fallback_iters: int,
    scan_fallback_accept_frac: float,
    scan_fallback_fsq_factor: float,
    scan_fallback_badjac_limit: int,
    scan_fallback_fsq_abs: float,
) -> ScanIterationRuntimePlan:
    """Resolve scan iteration counts, axis-reset offset, and runner cache key."""

    preflight_plan = resolve_scan_preflight_iters(
        jit_forces_scan=bool(jit_forces_scan),
        vmec2000_control=bool(vmec2000_control),
        max_iter=int(max_iter),
        axis_reset_repeat=bool(axis_reset_repeat),
        preflight_env=env.get("VMEC_JAX_SCAN_PREFLIGHT"),
    )
    iteration_plan = resolve_scan_iteration_plan(
        max_iter=int(max_iter),
        preflight_iters=int(preflight_plan.preflight_iters),
        vmec2000_control=bool(vmec2000_control),
        extra_iters_env=env.get("VMEC_JAX_SCAN_EXTRA_ITERS"),
    )
    iter_offset_preflight = int(iter_offset0)
    iter_offset_runtime = int(iter_offset0)
    if bool(axis_reset_repeat):
        iter_offset_preflight = 0
        iter_offset_runtime = -1
    scan_cache_key = build_vmec2000_scan_cache_key(
        static_key=static_key,
        wout_key=wout_key,
        edge_signature_key=edge_signature_key,
        tomnsps_policy_key=(
            _env_value(env, "VMEC_JAX_TOMNSPS_FFT", "").strip().lower(),
            _env_value(env, "VMEC_JAX_TOMNSPS_FFT_FUSED", "1").strip().lower(),
            _env_value(env, "VMEC_JAX_TOMNSPS_THETA_FUSED", "1").strip().lower(),
            _env_value(env, "VMEC_JAX_TOMNSPS_ZETA_FUSED", "1").strip().lower(),
        ),
        max_iter_tail=int(iteration_plan.max_iter_tail),
        preflight_iters=int(iteration_plan.preflight_iters),
        iter_offset0=int(iter_offset_runtime),
        step_size=float(step_size),
        initial_flip_sign=float(initial_flip_sign),
        lambda_update_scale=float(lambda_update_scale),
        ftol=float(ftol),
        fsq_total_target=fsq_total_target,
        nstep_screen=int(nstep_screen),
        use_restart_triggers=bool(use_restart_triggers),
        vmecpp_restart=bool(vmecpp_restart),
        scan_use_precomputed=bool(scan_use_precomputed),
        scan_use_lax_tridi=bool(scan_use_lax_tridi),
        scan_use_restart_payload=bool(scan_use_restart_payload),
        stage_prev_fsq=stage_prev_fsq,
        stage_transition_factor=float(stage_transition_factor),
        stage_transition_scale=float(stage_transition_scale),
        jit_forces_scan=bool(jit_forces_scan),
        state_only_scan=bool(state_only_scan),
        scan_light=bool(scan_light),
        scan_minimal=bool(scan_minimal),
        scan_fallback_iters=int(scan_fallback_iters),
        scan_fallback_accept_frac=float(scan_fallback_accept_frac),
        scan_fallback_fsq_factor=float(scan_fallback_fsq_factor),
        scan_fallback_badjac_limit=int(scan_fallback_badjac_limit),
        scan_fallback_fsq_abs=float(scan_fallback_fsq_abs),
    )
    return ScanIterationRuntimePlan(
        preflight_iters=int(iteration_plan.preflight_iters),
        max_iter_scan=int(iteration_plan.max_iter_scan),
        max_iter_tail=int(iteration_plan.max_iter_tail),
        iter_offset0=int(iter_offset_runtime),
        iter_offset_preflight=int(iter_offset_preflight),
        axis_reset_repeated=bool(axis_reset_repeat),
        scan_cache_key=scan_cache_key,
    )
