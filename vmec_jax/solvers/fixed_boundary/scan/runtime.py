"""Runtime hook resolution for VMEC2000-style scan solves.

The numerical scan body should stay focused on force evaluation and VMEC
time-step logic.  Optional host callbacks, tracing hooks, and diagnostic dump
paths are resolved here so the large solver routine does not own that plumbing.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Mapping, NamedTuple

import numpy as np

from .planning import normalize_scan_print_mode


class ScanRuntimeHooks(NamedTuple):
    """Optional runtime hooks used by the VMEC2000 scan path."""

    dump_timecontrol_scan: bool
    timecontrol_callback: Any | None
    timecontrol_path: Path | None
    io_callback: Any | None
    print_in_scan: bool
    scan_print_mode: str
    jax_debug: Any | None
    jax_debug_print: Callable[..., Any] | None
    scan_trace: bool
    scan_trace_context: Callable[[str], Any] | None


class ScanPreflightStepResult(NamedTuple):
    """Result of a one-step scan preflight."""

    carry: Any
    history_row: Any


class ChunkedScanRunResult(NamedTuple):
    """Materialized result from a chunked VMEC scan run."""

    carry_final: Any
    history: Any


class NonChunkedScanRunResult(NamedTuple):
    """Materialized result from a single-runner VMEC scan run."""

    carry_final: Any
    history: Any


class Vmec2000ScanDispatchResult(NamedTuple):
    """Materialized carry/history from the selected VMEC2000 scan runner."""

    carry_final: Any
    history: Any


def _env_value(env: Mapping[str, str | None], name: str, default: str = "") -> str:
    value = env.get(name, default)
    return default if value is None else str(value)


def resolve_scan_runtime_hooks(
    *,
    dump_timecontrol_env: str,
    dump_dir_env: str,
    print_in_scan: bool,
    scan_print_mode: str,
    scan_trace: bool,
) -> ScanRuntimeHooks:
    """Resolve optional callbacks and trace hooks without touching solver state."""

    dump_timecontrol_scan = str(dump_timecontrol_env).strip().lower() not in ("", "0", "false", "no")
    timecontrol_callback = None
    if dump_timecontrol_scan:
        try:
            from jax.experimental import io_callback as _io_callback

            timecontrol_callback = _io_callback
        except Exception:
            dump_timecontrol_scan = False
            timecontrol_callback = None

    jax_debug = None
    jax_debug_print = None
    if print_in_scan:
        try:
            from jax import debug as jax_debug

            jax_debug_print = jax_debug.print
        except Exception:
            print_in_scan = False

    if scan_print_mode == "io_callback":
        try:
            from jax.experimental import io_callback as _io_callback  # noqa: F401

            io_callback = _io_callback
            scan_print_mode = normalize_scan_print_mode(
                scan_print_mode=scan_print_mode,
                io_callback_available=True,
            )
        except Exception:
            io_callback = None
            scan_print_mode = normalize_scan_print_mode(
                scan_print_mode=scan_print_mode,
                io_callback_available=False,
            )
    else:
        io_callback = None
        scan_print_mode = normalize_scan_print_mode(
            scan_print_mode=scan_print_mode,
            io_callback_available=False,
        )

    scan_trace_context = None
    if scan_trace:
        try:
            from jax import profiler as _jax_profiler

            def scan_trace_context(label: str):  # type: ignore[misc]
                """Evaluate scan trace context for fixed-boundary VMEC solve and implicit differentiation."""
                return _jax_profiler.TraceAnnotation(label)

        except Exception:
            scan_trace = False
            scan_trace_context = None

    timecontrol_path = None
    if dump_timecontrol_scan:
        dump_dir = str(dump_dir_env).strip()
        if dump_dir:
            try:
                timecontrol_path = Path(dump_dir) / "time_control_trace.log"
            except Exception:
                timecontrol_path = None
        if timecontrol_path is None:
            dump_timecontrol_scan = False
            timecontrol_callback = None

    return ScanRuntimeHooks(
        dump_timecontrol_scan=bool(dump_timecontrol_scan),
        timecontrol_callback=timecontrol_callback,
        timecontrol_path=timecontrol_path,
        io_callback=io_callback,
        print_in_scan=bool(print_in_scan),
        scan_print_mode=str(scan_print_mode),
        jax_debug=jax_debug,
        jax_debug_print=jax_debug_print,
        scan_trace=bool(scan_trace),
        scan_trace_context=scan_trace_context,
    )


def resolve_scan_runtime_hooks_from_env(
    env: Mapping[str, str | None],
    *,
    print_in_scan: bool,
    scan_print_mode: str,
    scan_trace: bool,
) -> ScanRuntimeHooks:
    """Resolve scan runtime hooks from process-environment-like values."""

    return resolve_scan_runtime_hooks(
        dump_timecontrol_env=_env_value(env, "VMEC_JAX_DUMP_TIMECONTROL", ""),
        dump_dir_env=_env_value(env, "VMEC_JAX_DUMP_DIR", ""),
        print_in_scan=bool(print_in_scan),
        scan_print_mode=str(scan_print_mode),
        scan_trace=bool(scan_trace),
    )


def scan_trace_context_or_null(hooks: ScanRuntimeHooks, label: str):
    """Return a scan trace annotation context, or a no-op context."""

    if hooks.scan_trace and hooks.scan_trace_context is not None:
        return hooks.scan_trace_context(label)
    return nullcontext()


def get_or_build_scan_runner(
    run_scan_func,
    *,
    cache,
    key: tuple,
    differentiating_scan: bool,
    scan_timing_enabled: bool,
    scan_timing_stats: dict[str, Any],
    jit_func: Callable[[Any], Any],
    cache_get: Callable[[Any, tuple], Any],
    cache_put: Callable[..., Any],
    record_miss_categories: Callable[..., Any],
    perf_counter: Callable[[], float],
    cache_env_name: str = "VMEC_JAX_SCAN_RUNNER_CACHE_SIZE",
    cache_default: int = 32,
):
    """Return a JIT scan runner and cache status for VMEC scan controllers.

    This helper owns only cache/timing bookkeeping.  The numerical scan body is
    still built at the call site so branch-specific closed-over constants remain
    explicit and fingerprintable.
    """

    if bool(differentiating_scan):
        if bool(scan_timing_enabled):
            scan_timing_stats["scan_runner_cache_bypass_count"] = (
                int(scan_timing_stats.get("scan_runner_cache_bypass_count", 0)) + 1
            )
        return jit_func(run_scan_func), "bypass"

    lookup_start = perf_counter() if bool(scan_timing_enabled) else None
    cached_run = cache_get(cache, key)
    if bool(scan_timing_enabled) and lookup_start is not None:
        scan_timing_stats["scan_runner_cache_lookup_s"] = float(
            scan_timing_stats.get("scan_runner_cache_lookup_s", 0.0)
        ) + (perf_counter() - float(lookup_start))

    if cached_run is not None:
        if bool(scan_timing_enabled):
            scan_timing_stats["scan_runner_cache_hit_count"] = (
                int(scan_timing_stats.get("scan_runner_cache_hit_count", 0)) + 1
            )
        return cached_run, "hit"

    if bool(scan_timing_enabled):
        scan_timing_stats["scan_runner_cache_miss_count"] = (
            int(scan_timing_stats.get("scan_runner_cache_miss_count", 0)) + 1
        )
        record_miss_categories(
            scan_timing_stats,
            requested_key=key,
            existing_keys=tuple(cache.keys()),
        )

    build_start = perf_counter() if bool(scan_timing_enabled) else None
    runner = jit_func(run_scan_func)
    cached_runner = cache_put(
        cache,
        key,
        runner,
        env_name=str(cache_env_name),
        default=int(cache_default),
    )
    if bool(scan_timing_enabled) and build_start is not None:
        scan_timing_stats["scan_runner_cache_build_s"] = float(
            scan_timing_stats.get("scan_runner_cache_build_s", 0.0)
        ) + (perf_counter() - float(build_start))
    return cached_runner, "miss"


def run_scan_preflight_step(
    carry,
    *,
    iter_offset_preflight: int | None,
    jit_preflight: bool,
    get_scan_runner: Callable[[int], tuple[Any, str]],
    scan_step: Callable[[Any, Any], tuple[Any, Any]],
    scan_timing_enabled: bool,
    scan_timing_stats: dict[str, Any],
    block_scan_value: Callable[[Any], Any],
    perf_counter: Callable[[], float],
    jnp_module,
    jax_module,
) -> ScanPreflightStepResult:
    """Run the VMEC2000 scan's one-step preflight with shared timing logic."""

    preflight_start = perf_counter() if bool(scan_timing_enabled) else None
    if iter_offset_preflight is not None:
        carry = carry._replace(iter_offset=jnp_module.asarray(iter_offset_preflight, dtype=jnp_module.int32))
    if bool(jit_preflight):
        preflight_runner, _preflight_cache_status = get_scan_runner(1)
        carry, hist_pre_seq = preflight_runner(carry, jnp_module.asarray([0], dtype=jnp_module.int32))
        hist_pre = jax_module.tree_util.tree_map(lambda a: a[0], hist_pre_seq)
    else:
        try:
            with jax_module.disable_jit():
                carry, hist_pre = scan_step(carry, jnp_module.asarray(0, dtype=jnp_module.int32))
        except Exception:
            carry, hist_pre = scan_step(carry, jnp_module.asarray(0, dtype=jnp_module.int32))
    if bool(scan_timing_enabled) and preflight_start is not None:
        carry, hist_pre = block_scan_value((carry, hist_pre))
        scan_timing_stats["scan_preflight_s"] = float(scan_timing_stats.get("scan_preflight_s", 0.0)) + (
            perf_counter() - float(preflight_start)
        )
    return ScanPreflightStepResult(carry=carry, history_row=hist_pre)


def run_chunked_scan(
    carry_init,
    *,
    max_iter: int,
    max_iter_scan: int,
    nstep_screen: int,
    need_print: bool,
    lthreed: bool,
    spectral_mode_count: int,
    scan_chunk_settings_func: Callable[..., tuple[int, bool]],
    scan_jit_preflight_enabled_func: Callable[..., bool],
    scan_jit_preflight_env: str | None,
    backend_name: str,
    scan_differentiated: bool,
    preflight_iters: int,
    iter_offset_preflight: int,
    axis_reset_repeat: bool,
    iter_offset0: int,
    get_scan_runner: Callable[[int], tuple[Any, str]],
    scan_step: Callable[[Any, Any], tuple[Any, Any]],
    scan_timing_enabled: bool,
    scan_timing_stats: dict[str, Any],
    scan_device_runtime: Any,
    perf_counter: Callable[[], float],
    state_only_scan: bool,
    scan_fallback_enabled_run: bool,
    scan_fallback_iters: int,
    scan_fallback_fsq_abs: float,
    dtype: Any,
    emit_scan_prints: Callable[..., bool],
    tree_has_tracer: Callable[[Any], bool],
    jnp_module,
    jax_module,
    np_module=np,
) -> ChunkedScanRunResult:
    """Run a VMEC scan in fixed-size chunks with optional print materialization."""

    hist_parts = []
    start_idx = 0
    carry = carry_init
    abort_scan_host = False
    fsq_min_global_j = jnp_module.asarray(jnp_module.inf, dtype=dtype)
    chunk_size, chunk_cap_remaining = scan_chunk_settings_func(
        max_iter_scan=int(max_iter_scan),
        nstep_screen=int(nstep_screen),
        need_print=bool(need_print),
        lthreed=bool(lthreed),
        spectral_mode_count=int(spectral_mode_count),
    )
    jit_preflight = scan_jit_preflight_enabled_func(
        env_value=scan_jit_preflight_env,
        backend_name=str(backend_name),
        scan_differentiated=bool(scan_differentiated),
    ) and (not bool(need_print))
    if int(preflight_iters) > 0:
        preflight = run_scan_preflight_step(
            carry,
            iter_offset_preflight=int(iter_offset_preflight),
            jit_preflight=bool(jit_preflight),
            get_scan_runner=get_scan_runner,
            scan_step=scan_step,
            scan_timing_enabled=bool(scan_timing_enabled),
            scan_timing_stats=scan_timing_stats,
            block_scan_value=scan_device_runtime.block_value,
            perf_counter=perf_counter,
            jnp_module=jnp_module,
            jax_module=jax_module,
        )
        carry = preflight.carry
        hist_pre = preflight.history_row
        if not bool(state_only_scan):
            fsq_min_global_j = jnp_module.minimum(
                fsq_min_global_j,
                jnp_module.min(hist_pre[0] + hist_pre[1] + hist_pre[2]),
            )
        if bool(need_print):
            hist_pre_np = jax_module.tree_util.tree_map(lambda a: np_module.asarray(a)[None], hist_pre)
            hist_parts.append(hist_pre_np)
            _ = emit_scan_prints(hist_np=hist_pre_np, it_start=0, max_iter_local=int(max_iter))
        elif not bool(state_only_scan):
            hist_parts.append(jax_module.tree_util.tree_map(lambda a: a[None], hist_pre))
        start_idx = int(preflight_iters)
        if bool(axis_reset_repeat):
            carry = carry._replace(iter_offset=jnp_module.asarray(iter_offset0, dtype=jnp_module.int32))

    while start_idx < int(max_iter_scan):
        remaining = int(max_iter_scan) - int(start_idx)
        if remaining <= 0:
            break
        chunk_len = min(int(chunk_size), int(remaining)) if chunk_cap_remaining else int(chunk_size)
        it_seq = jnp_module.arange(start_idx, start_idx + int(chunk_len), dtype=jnp_module.int32)
        runner, cache_status = get_scan_runner(int(chunk_len))
        t_device = perf_counter() if bool(scan_timing_enabled) else None
        carry, hist_chunk = runner(carry, it_seq)
        if bool(scan_timing_enabled) and t_device is not None:
            carry, hist_chunk = scan_device_runtime.ready(
                t_device,
                (carry, hist_chunk),
                cache_status=cache_status,
            )
        if not bool(state_only_scan):
            fsq_min_global_j = jnp_module.minimum(
                fsq_min_global_j,
                jnp_module.min(hist_chunk[0] + hist_chunk[1] + hist_chunk[2]),
            )
        if bool(need_print):
            hist_chunk_np = jax_module.tree_util.tree_map(lambda a: np_module.asarray(a), hist_chunk)
            hist_parts.append(hist_chunk_np)
            converged_now = emit_scan_prints(
                hist_np=hist_chunk_np,
                it_start=int(start_idx),
                max_iter_local=int(max_iter),
            )
        else:
            if not bool(state_only_scan):
                hist_parts.append(hist_chunk)
            converged_now = False
        start_idx = int(start_idx + int(chunk_len))
        if (
            bool(scan_fallback_enabled_run)
            and int(scan_fallback_iters) > 0
            and start_idx >= int(scan_fallback_iters)
            and bool(np_module.asarray(carry.fallback_active))
        ):
            carry = carry._replace(fallback_active=jnp_module.asarray(False))
        if converged_now:
            break
        if bool(np_module.asarray(carry.converged)) or bool(np_module.asarray(carry.abort_scan)):
            break

    if bool(scan_fallback_enabled_run) and start_idx >= int(scan_fallback_iters):
        try:
            fsq_min_global = float(jax_module.device_get(fsq_min_global_j))
        except Exception:
            fsq_min_global = None
        if fsq_min_global is not None and fsq_min_global > float(scan_fallback_fsq_abs):
            abort_scan_host = True

    if bool(state_only_scan) and not bool(need_print):
        hist = None
    elif bool(need_print):
        hist = jax_module.tree_util.tree_map(lambda *parts: np_module.concatenate(parts, axis=0), *hist_parts)
    else:
        t_materialize = perf_counter() if bool(scan_timing_enabled) else None
        hist = jax_module.tree_util.tree_map(lambda *parts: jnp_module.concatenate(parts, axis=0), *hist_parts)
        if not tree_has_tracer(hist):
            hist = jax_module.tree_util.tree_map(lambda a: np_module.asarray(a), hist)
        if bool(scan_timing_enabled) and t_materialize is not None:
            scan_timing_stats["scan_host_materialize_s"] = float(
                scan_timing_stats.get("scan_host_materialize_s", 0.0)
            ) + (perf_counter() - float(t_materialize))
    carry_final = carry
    if abort_scan_host:
        carry_final = carry_final._replace(abort_scan=jnp_module.asarray(True))
    return ChunkedScanRunResult(carry_final=carry_final, history=hist)


def run_nonchunked_scan(
    carry_init,
    *,
    max_iter_scan: int,
    max_iter_tail: int,
    preflight_iters: int,
    iter_offset_preflight: int,
    axis_reset_repeat: bool,
    iter_offset0: int,
    get_scan_runner: Callable[[int], tuple[Any, str]],
    scan_step: Callable[[Any, Any], tuple[Any, Any]],
    scan_jit_preflight_enabled_func: Callable[..., bool],
    scan_jit_preflight_env: str | None,
    backend_name: str,
    scan_differentiated: bool,
    scan_collect_print: bool,
    scan_timing_enabled: bool,
    scan_timing_stats: dict[str, Any],
    scan_device_runtime: Any,
    perf_counter: Callable[[], float],
    state_only_scan: bool,
    scan_fallback_enabled_run: bool,
    scan_fallback_iters: int,
    jnp_module,
    jax_module,
) -> NonChunkedScanRunResult:
    """Run a VMEC scan with one cached runner and optional one-step preflight."""

    runner, cache_status = get_scan_runner(int(max_iter_tail) if int(max_iter_tail) > 0 else int(max_iter_scan))
    if int(preflight_iters) > 0:
        carry_pre = carry_init
        jit_preflight = scan_jit_preflight_enabled_func(
            env_value=scan_jit_preflight_env,
            backend_name=str(backend_name),
            scan_differentiated=bool(scan_differentiated),
        ) and (not bool(scan_collect_print))
        preflight = run_scan_preflight_step(
            carry_pre,
            iter_offset_preflight=int(iter_offset_preflight),
            jit_preflight=bool(jit_preflight),
            get_scan_runner=get_scan_runner,
            scan_step=scan_step,
            scan_timing_enabled=bool(scan_timing_enabled),
            scan_timing_stats=scan_timing_stats,
            block_scan_value=scan_device_runtime.block_value,
            perf_counter=perf_counter,
            jnp_module=jnp_module,
            jax_module=jax_module,
        )
        carry_pre = preflight.carry
        hist_pre = preflight.history_row
        if (
            bool(scan_fallback_enabled_run)
            and int(scan_fallback_iters) > 0
            and int(preflight_iters) >= int(scan_fallback_iters)
        ):
            carry_pre = carry_pre._replace(fallback_active=jnp_module.asarray(False))
        if int(max_iter_tail) > 0:
            it_seq = jnp_module.arange(preflight_iters, int(max_iter_scan), dtype=jnp_module.int32)
            if bool(axis_reset_repeat):
                carry_pre = carry_pre._replace(iter_offset=jnp_module.asarray(iter_offset0, dtype=jnp_module.int32))
            t_device = perf_counter() if bool(scan_timing_enabled) else None
            carry_final, hist_tail = runner(carry_pre, it_seq)
            if bool(scan_timing_enabled) and t_device is not None:
                carry_final, hist_tail = scan_device_runtime.ready(
                    t_device,
                    (carry_final, hist_tail),
                    cache_status=cache_status,
                )
            if bool(state_only_scan):
                hist = None
            else:
                hist = jax_module.tree_util.tree_map(
                    lambda a, b: jnp_module.concatenate([a[None], b], axis=0),
                    hist_pre,
                    hist_tail,
                )
        else:
            carry_final = carry_pre
            hist = None if bool(state_only_scan) else jax_module.tree_util.tree_map(lambda a: a[None], hist_pre)
    else:
        it_seq = jnp_module.arange(int(max_iter_scan), dtype=jnp_module.int32)
        t_device = perf_counter() if bool(scan_timing_enabled) else None
        carry_final, hist = runner(carry_init, it_seq)
        if bool(scan_timing_enabled) and t_device is not None:
            carry_final, hist = scan_device_runtime.ready(
                t_device,
                (carry_final, hist),
                cache_status=cache_status,
            )
        if bool(state_only_scan):
            hist = None
    return NonChunkedScanRunResult(carry_final=carry_final, history=hist)


def run_vmec2000_scan_dispatch(
    carry_init,
    *,
    chunked_print: bool,
    chunked_kwargs: Mapping[str, Any],
    nonchunked_kwargs: Mapping[str, Any],
) -> Vmec2000ScanDispatchResult:
    """Run the VMEC2000 scan using the configured chunking/print policy."""

    if bool(chunked_print):
        chunked_result = run_chunked_scan(carry_init, **dict(chunked_kwargs))
        return Vmec2000ScanDispatchResult(
            carry_final=chunked_result.carry_final,
            history=chunked_result.history,
        )

    nonchunked_result = run_nonchunked_scan(carry_init, **dict(nonchunked_kwargs))
    return Vmec2000ScanDispatchResult(
        carry_final=nonchunked_result.carry_final,
        history=nonchunked_result.history,
    )
