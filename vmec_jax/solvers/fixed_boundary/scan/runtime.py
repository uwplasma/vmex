"""Runtime hook resolution for VMEC2000-style scan solves.

The numerical scan body should stay focused on force evaluation and VMEC
time-step logic.  Optional host callbacks, tracing hooks, and diagnostic dump
paths are resolved here so the large solver routine does not own that plumbing.
"""

from __future__ import annotations

from contextlib import nullcontext
import os
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


def scan_explicit_compile_enabled(env_value: str | None = None) -> bool:
    """Return whether scan-runner explicit compile attribution is enabled."""

    value = os.getenv("VMEC_JAX_SCAN_EXPLICIT_COMPILE", "") if env_value is None else str(env_value)
    return value.strip().lower() not in ("", "0", "false", "no", "off")


def scan_hlo_summary_enabled(env_value: str | None = None) -> bool:
    """Return whether explicit scan-runner HLO size attribution is enabled."""

    value = os.getenv("VMEC_JAX_SCAN_HLO_SUMMARY", "") if env_value is None else str(env_value)
    return value.strip().lower() not in ("", "0", "false", "no", "off")


def scan_arg_summary_enabled(env_value: str | None = None) -> bool:
    """Return whether scan-runner input-tree size diagnostics are enabled."""

    value = os.getenv("VMEC_JAX_SCAN_ARG_SUMMARY", "") if env_value is None else str(env_value)
    return value.strip().lower() not in ("", "0", "false", "no", "off")


def _hlo_text_from_lowered(lowered) -> str | None:
    """Return HLO text from a lowered JAX object when available."""

    try:
        hlo_ir = lowered.compiler_ir(dialect="hlo")
    except Exception:
        return None
    try:
        if hasattr(hlo_ir, "as_hlo_text"):
            return str(hlo_ir.as_hlo_text())
        if hasattr(hlo_ir, "as_text"):
            return str(hlo_ir.as_text())
    except Exception:
        return None
    try:
        return str(hlo_ir)
    except Exception:
        return None


def summarize_scan_runner_hlo_text(hlo_text: str) -> dict[str, int]:
    """Return stable line/instruction/op counts for lowered scan HLO text."""

    lines = [line.strip() for line in str(hlo_text).splitlines() if line.strip()]
    instruction_lines = [line for line in lines if "=" in line and not line.startswith("HloModule")]
    summary: dict[str, int] = {
        "scan_runner_explicit_hlo_line_count": int(len(lines)),
        "scan_runner_explicit_hlo_instruction_count": int(len(instruction_lines)),
    }
    for line in instruction_lines:
        op_name = _hlo_instruction_op_name(line)
        if op_name is None:
            continue
        key = f"scan_runner_explicit_hlo_op_{op_name}_count"
        summary[key] = int(summary.get(key, 0)) + 1
    return summary


def _hlo_instruction_op_name(line: str) -> str | None:
    """Extract an HLO opcode from one assignment line without regex fragility."""

    if "=" not in line or "(" not in line:
        return None
    rhs = line.split("=", 1)[1].strip()
    before_args = rhs.split("(", 1)[0].strip()
    if not before_args:
        return None
    op = before_args.split()[-1].strip()
    if not op or not op[0].isalpha():
        return None
    return op.lower().replace("-", "_")


def maybe_record_scan_runner_hlo_summary(
    lowered,
    *,
    scan_timing_enabled: bool,
    scan_timing_stats: dict[str, Any],
    env_value: str | None = None,
) -> None:
    """Optionally record lowered scan HLO size metrics into timing stats."""

    if not bool(scan_timing_enabled) or not scan_hlo_summary_enabled(env_value):
        return
    hlo_text = _hlo_text_from_lowered(lowered)
    if hlo_text is None:
        scan_timing_stats["scan_runner_explicit_hlo_failure_count"] = (
            int(scan_timing_stats.get("scan_runner_explicit_hlo_failure_count", 0)) + 1
        )
        return
    for key, value in summarize_scan_runner_hlo_text(hlo_text).items():
        scan_timing_stats[key] = int(scan_timing_stats.get(key, 0)) + int(value)


def _scan_runner_arg_path_leaves(value, path: tuple[str, ...] = ()):
    """Yield ``(path, leaf)`` pairs from scan-runner pytrees."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _scan_runner_arg_path_leaves(item, (*path, str(key)))
        return
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        for field, item in zip(value._fields, value, strict=True):
            yield from _scan_runner_arg_path_leaves(item, (*path, str(field)))
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _scan_runner_arg_path_leaves(item, (*path, str(index)))
        return
    yield path, value


def _scan_arg_group_key(path: tuple[str, ...], *, depth: int = 2) -> str:
    """Return a stable key for grouping scan-runner argument leaves."""

    parts = path[: max(1, int(depth))]
    text = "_".join(parts) if parts else "root"
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_") or "root"


_SCAN_CARRY_VELOCITY_FIELDS = frozenset(
    {
        "vRcc",
        "vRss",
        "vRsc",
        "vRcs",
        "vZsc",
        "vZcs",
        "vZcc",
        "vZss",
        "vLsc",
        "vLcs",
        "vLcc",
        "vLss",
    }
)
_SCAN_CARRY_PRECONDITIONER_FIELDS = frozenset(
    {
        "cache_precond_diag",
        "cache_tcon",
        "cache_norms",
        "cache_rz_scale",
        "cache_l_scale",
        "cache_rz_norm",
        "cache_f_norm1",
        "cache_prec_rz_mats",
        "cache_prec_lam_prec",
    }
)
_SCAN_CARRY_BOUNDARY_FIELDS = frozenset({"edge_Rcos", "edge_Rsin", "edge_Zcos", "edge_Zsin"})
_SCAN_CARRY_RESIDUAL_FIELDS = frozenset(
    {
        "fsq_prev",
        "fsq0_prev",
        "fsqz_prev",
        "fsqr_prev_phys",
        "fsqz_prev_phys",
        "fsql_prev_phys",
        "fsqr1_prev",
        "fsqz1_prev",
        "fsql1_prev",
        "fsqr_checkpoint",
        "fsqz_checkpoint",
        "fsql_checkpoint",
        "fsqr1_checkpoint",
        "fsqz1_checkpoint",
        "fsql1_checkpoint",
        "res0",
        "res1",
        "w_mhd_prev",
    }
)
_SCAN_CARRY_CONTROLLER_FIELDS = frozenset(
    {
        "time_step",
        "inv_tau",
        "accepted_count",
        "probe_count",
        "probe_bad_jac",
        "probe_accept",
        "probe_fsq_min",
        "probe_fsq_max",
        "probe_fsq_start",
        "fallback_active",
        "abort_scan",
        "skip_timecontrol",
        "flip_sign",
        "iter_offset",
        "iter1",
        "cache_valid",
        "ijacob",
        "bad_resets",
        "bad_growth",
        "converged",
        "r00_prev",
        "z00_prev",
    }
)
_SCAN_RZ_CARRY_APPLY_KEYS = frozenset(
    {
        "ar",
        "br",
        "dr",
        "az",
        "bz",
        "dz",
        "cr",
        "ir",
        "cz",
        "iz",
        "dlr_t",
        "dr_t",
        "dur_t",
        "dlz_t",
        "dz_t",
        "duz_t",
    }
)
_SCAN_RZ_CARRY_DERIVED_KEYS = frozenset({"m1_fac_r", "m1_fac_z"})
_SCAN_RZ_CARRY_ALLOWED_KEYS = _SCAN_RZ_CARRY_APPLY_KEYS | _SCAN_RZ_CARRY_DERIVED_KEYS
_SCAN_RZ_CARRY_MANDATORY_KEYS = frozenset({"ar", "br", "dr", "az", "bz", "dz", "m1_fac_r", "m1_fac_z"})


def _scan_arg_category_key(path: tuple[str, ...]) -> str:
    """Return a physics-aware category for scan-runner argument leaves."""

    if not path:
        return "root"
    if path[0] != "arg0":
        return "iteration_input" if path[0] == "arg1" else "runtime_input"
    field = path[1] if len(path) > 1 else ""
    if field in ("state", "state_checkpoint"):
        return "state"
    if field in _SCAN_CARRY_VELOCITY_FIELDS:
        return "velocity"
    if field in _SCAN_CARRY_PRECONDITIONER_FIELDS:
        return "preconditioner"
    if field in _SCAN_CARRY_BOUNDARY_FIELDS:
        return "boundary"
    if field in _SCAN_CARRY_RESIDUAL_FIELDS:
        return "residual"
    if field in _SCAN_CARRY_CONTROLLER_FIELDS:
        return "controller"
    return "other"


def record_scan_runner_arg_summary(
    args: tuple[Any, ...],
    *,
    scan_timing_enabled: bool,
    scan_timing_stats: dict[str, Any],
) -> None:
    """Record scan-runner argument breadth for compile-time diagnostics."""

    if not bool(scan_timing_enabled):
        return
    leaf_count = 0
    array_leaf_count = 0
    scalar_leaf_count = 0
    array_nbytes = 0
    group_counts: dict[str, int] = {}
    group_array_counts: dict[str, int] = {}
    group_nbytes: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    category_array_counts: dict[str, int] = {}
    category_nbytes: dict[str, int] = {}
    rz_mat_keys: set[str] = set()
    for index, arg in enumerate(args):
        for path, leaf in _scan_runner_arg_path_leaves(arg, (f"arg{index}",)):
            group_key = _scan_arg_group_key(path)
            category_key = _scan_arg_category_key(path)
            if len(path) > 2 and path[0] == "arg0" and path[1] == "cache_prec_rz_mats":
                rz_mat_keys.add(str(path[2]))
            group_counts[group_key] = int(group_counts.get(group_key, 0)) + 1
            category_counts[category_key] = int(category_counts.get(category_key, 0)) + 1
            leaf_count += 1
            shape = getattr(leaf, "shape", None)
            nbytes = getattr(leaf, "nbytes", None)
            if shape is not None:
                array_leaf_count += 1
                group_array_counts[group_key] = int(group_array_counts.get(group_key, 0)) + 1
                category_array_counts[category_key] = int(category_array_counts.get(category_key, 0)) + 1
                try:
                    nbytes_int = int(nbytes)
                except Exception:
                    nbytes_int = 0
                array_nbytes += nbytes_int
                group_nbytes[group_key] = int(group_nbytes.get(group_key, 0)) + nbytes_int
                category_nbytes[category_key] = int(category_nbytes.get(category_key, 0)) + nbytes_int
            else:
                scalar_leaf_count += 1
    scan_timing_stats["scan_runner_arg_leaf_count"] = int(leaf_count)
    scan_timing_stats["scan_runner_arg_array_leaf_count"] = int(array_leaf_count)
    scan_timing_stats["scan_runner_arg_scalar_leaf_count"] = int(scalar_leaf_count)
    scan_timing_stats["scan_runner_arg_array_nbytes"] = int(array_nbytes)
    for group_key, count in group_counts.items():
        prefix = f"scan_runner_arg_path_{group_key}"
        scan_timing_stats[f"{prefix}_leaf_count"] = int(count)
        scan_timing_stats[f"{prefix}_array_leaf_count"] = int(group_array_counts.get(group_key, 0))
        scan_timing_stats[f"{prefix}_array_nbytes"] = int(group_nbytes.get(group_key, 0))
    for category_key, count in category_counts.items():
        prefix = f"scan_runner_arg_category_{category_key}"
        scan_timing_stats[f"{prefix}_leaf_count"] = int(count)
        scan_timing_stats[f"{prefix}_array_leaf_count"] = int(category_array_counts.get(category_key, 0))
        scan_timing_stats[f"{prefix}_array_nbytes"] = int(category_nbytes.get(category_key, 0))
    if rz_mat_keys:
        unexpected_keys = rz_mat_keys - _SCAN_RZ_CARRY_ALLOWED_KEYS
        missing_mandatory_keys = _SCAN_RZ_CARRY_MANDATORY_KEYS - rz_mat_keys
        scan_timing_stats["scan_runner_arg_preconditioner_rz_mats_key_count"] = int(len(rz_mat_keys))
        scan_timing_stats["scan_runner_arg_preconditioner_rz_mats_unexpected_key_count"] = int(
            len(unexpected_keys)
        )
        scan_timing_stats["scan_runner_arg_preconditioner_rz_mats_missing_mandatory_key_count"] = int(
            len(missing_mandatory_keys)
        )
        scan_timing_stats["scan_runner_arg_preconditioner_rz_mats_compact_ok_count"] = int(
            (not unexpected_keys) and (not missing_mandatory_keys)
        )


def record_scan_history_summary(
    history: Any,
    *,
    scan_timing_enabled: bool,
    scan_timing_stats: dict[str, Any],
) -> None:
    """Record scan-output history breadth for compile and memory diagnostics.

    The VMEC2000-compatible scan path has two independent graph-width costs:
    the carry passed into the scan body and the per-iteration history row
    emitted by the scan.  Argument summaries classify the former; this helper
    classifies the latter so performance probes can distinguish a wide
    momentum/preconditioner carry from expensive history materialization.
    """

    if not bool(scan_timing_enabled):
        return
    prefix = "scan_history"
    if history is None:
        scan_timing_stats[f"{prefix}_none"] = 1
        scan_timing_stats[f"{prefix}_leaf_count"] = 0
        scan_timing_stats[f"{prefix}_array_leaf_count"] = 0
        scan_timing_stats[f"{prefix}_scalar_leaf_count"] = 0
        scan_timing_stats[f"{prefix}_array_nbytes"] = 0
        return

    leaf_count = 0
    array_leaf_count = 0
    scalar_leaf_count = 0
    array_nbytes = 0
    for _path, leaf in _scan_runner_arg_path_leaves(history, ("history",)):
        leaf_count += 1
        shape = getattr(leaf, "shape", None)
        nbytes = getattr(leaf, "nbytes", None)
        if shape is not None:
            array_leaf_count += 1
            try:
                array_nbytes += int(nbytes)
            except Exception:
                pass
        else:
            scalar_leaf_count += 1
    scan_timing_stats[f"{prefix}_none"] = 0
    scan_timing_stats[f"{prefix}_leaf_count"] = int(leaf_count)
    scan_timing_stats[f"{prefix}_array_leaf_count"] = int(array_leaf_count)
    scan_timing_stats[f"{prefix}_scalar_leaf_count"] = int(scalar_leaf_count)
    scan_timing_stats[f"{prefix}_array_nbytes"] = int(array_nbytes)


def maybe_record_scan_runner_arg_summary(
    args: tuple[Any, ...],
    *,
    scan_timing_enabled: bool,
    scan_timing_stats: dict[str, Any],
    env_value: str | None = None,
) -> None:
    """Optionally record scan-runner argument breadth without forcing compile.

    This is a cheap profiling seam for classifying scan graph breadth.  It is
    deliberately separate from explicit lowering/compile attribution so routine
    benchmark runs can inspect the runner input tree without changing JAX's
    lazy compilation behavior.
    """

    if not scan_arg_summary_enabled(env_value):
        return
    record_scan_runner_arg_summary(
        args,
        scan_timing_enabled=bool(scan_timing_enabled),
        scan_timing_stats=scan_timing_stats,
    )


def maybe_explicit_compile_scan_runner(
    runner,
    args: tuple[Any, ...],
    *,
    cache_status: str | None,
    scan_timing_enabled: bool,
    scan_timing_stats: dict[str, Any],
    perf_counter: Callable[[], float],
    env_value: str | None = None,
    arg_summary_env_value: str | None = None,
):
    """Optionally lower/compile a scan runner before dispatch for attribution.

    Normal solves leave JAX compilation lazy.  When
    ``VMEC_JAX_SCAN_EXPLICIT_COMPILE`` is enabled, this helper moves lowering
    and compilation into explicit timing buckets and returns the compiled
    executable for the immediate call.  It is diagnostic-only and intentionally
    leaves cache keys and default production behavior unchanged.
    """

    explicit_compile = scan_explicit_compile_enabled(env_value)
    if explicit_compile or scan_arg_summary_enabled(arg_summary_env_value):
        record_scan_runner_arg_summary(
            args,
            scan_timing_enabled=bool(scan_timing_enabled),
            scan_timing_stats=scan_timing_stats,
        )

    if not explicit_compile:
        return runner
    lowered = None
    try:
        t_lower = perf_counter() if bool(scan_timing_enabled) else None
        lowered = runner.lower(*args)
        if bool(scan_timing_enabled) and t_lower is not None:
            scan_timing_stats["scan_runner_explicit_lower_s"] = float(
                scan_timing_stats.get("scan_runner_explicit_lower_s", 0.0)
            ) + (perf_counter() - float(t_lower))
        maybe_record_scan_runner_hlo_summary(
            lowered,
            scan_timing_enabled=bool(scan_timing_enabled),
            scan_timing_stats=scan_timing_stats,
        )

        t_compile = perf_counter() if bool(scan_timing_enabled) else None
        compiled = lowered.compile()
        if bool(scan_timing_enabled):
            if t_compile is not None:
                scan_timing_stats["scan_runner_explicit_compile_s"] = float(
                    scan_timing_stats.get("scan_runner_explicit_compile_s", 0.0)
                ) + (perf_counter() - float(t_compile))
            scan_timing_stats["scan_runner_explicit_compile_count"] = (
                int(scan_timing_stats.get("scan_runner_explicit_compile_count", 0)) + 1
            )
            status = str(cache_status or "unknown").strip().lower() or "unknown"
            status_key = f"scan_runner_explicit_compile_{status}_count"
            scan_timing_stats[status_key] = int(scan_timing_stats.get(status_key, 0)) + 1
        return compiled
    except Exception:
        if bool(scan_timing_enabled):
            scan_timing_stats["scan_runner_explicit_compile_failure_count"] = (
                int(scan_timing_stats.get("scan_runner_explicit_compile_failure_count", 0)) + 1
            )
        return runner


def run_scan_preflight_step(
    carry,
    *,
    iter_offset_preflight: int | None,
    jit_preflight: bool,
    get_scan_runner: Callable[[int], tuple[Any, str]],
    scan_step: Callable[[Any, Any], tuple[Any, Any]],
    build_scan_it_seq: Callable[[int, int], Any] | None = None,
    runtime_scan_args: tuple[Any, ...] = (),
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
        it_seq = _scan_iteration_sequence(0, 1, jnp_module=jnp_module, build_scan_it_seq=build_scan_it_seq)
        carry, hist_pre_seq = preflight_runner(carry, it_seq, *runtime_scan_args)
        hist_pre = jax_module.tree_util.tree_map(lambda a: a[0], hist_pre_seq)
    else:
        it0 = _scan_iteration_item(0, jnp_module=jnp_module, jax_module=jax_module, build_scan_it_seq=build_scan_it_seq)
        try:
            with jax_module.disable_jit():
                carry, hist_pre = scan_step(carry, it0)
        except Exception:
            carry, hist_pre = scan_step(carry, it0)
    if bool(scan_timing_enabled) and preflight_start is not None:
        carry, hist_pre = block_scan_value((carry, hist_pre))
        scan_timing_stats["scan_preflight_s"] = float(scan_timing_stats.get("scan_preflight_s", 0.0)) + (
            perf_counter() - float(preflight_start)
        )
    return ScanPreflightStepResult(carry=carry, history_row=hist_pre)


def _scan_iteration_sequence(
    start: int,
    stop: int,
    *,
    jnp_module,
    build_scan_it_seq: Callable[[int, int], Any] | None,
) -> Any:
    """Return scan iteration inputs, optionally enriched with runtime controls."""

    if build_scan_it_seq is not None:
        return build_scan_it_seq(int(start), int(stop))
    if not hasattr(jnp_module, "arange"):
        return jnp_module.asarray(list(range(int(start), int(stop))), dtype=jnp_module.int32)
    return jnp_module.arange(int(start), int(stop), dtype=jnp_module.int32)


def _scan_iteration_item(
    index: int,
    *,
    jnp_module,
    jax_module,
    build_scan_it_seq: Callable[[int, int], Any] | None,
) -> Any:
    """Return one scan-step input matching the full scan-input structure."""

    if build_scan_it_seq is None:
        return jnp_module.asarray(int(index), dtype=jnp_module.int32)
    seq = _scan_iteration_sequence(index, index + 1, jnp_module=jnp_module, build_scan_it_seq=build_scan_it_seq)
    return jax_module.tree_util.tree_map(lambda value: value[0], seq)


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
    build_scan_it_seq: Callable[[int, int], Any] | None = None,
    runtime_scan_args: tuple[Any, ...] = (),
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
            build_scan_it_seq=build_scan_it_seq,
            runtime_scan_args=runtime_scan_args,
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
        it_seq = _scan_iteration_sequence(
            start_idx,
            start_idx + int(chunk_len),
            jnp_module=jnp_module,
            build_scan_it_seq=build_scan_it_seq,
        )
        runner, cache_status = get_scan_runner(int(chunk_len))
        runner = maybe_explicit_compile_scan_runner(
            runner,
            (carry, it_seq, *runtime_scan_args),
            cache_status=cache_status,
            scan_timing_enabled=bool(scan_timing_enabled),
            scan_timing_stats=scan_timing_stats,
            perf_counter=perf_counter,
        )
        t_device = perf_counter() if bool(scan_timing_enabled) else None
        carry, hist_chunk = runner(carry, it_seq, *runtime_scan_args)
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
    record_scan_history_summary(
        hist,
        scan_timing_enabled=bool(scan_timing_enabled),
        scan_timing_stats=scan_timing_stats,
    )
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
    build_scan_it_seq: Callable[[int, int], Any] | None = None,
    runtime_scan_args: tuple[Any, ...] = (),
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
            build_scan_it_seq=build_scan_it_seq,
            runtime_scan_args=runtime_scan_args,
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
            it_seq = _scan_iteration_sequence(
                preflight_iters,
                int(max_iter_scan),
                jnp_module=jnp_module,
                build_scan_it_seq=build_scan_it_seq,
            )
            if bool(axis_reset_repeat):
                carry_pre = carry_pre._replace(iter_offset=jnp_module.asarray(iter_offset0, dtype=jnp_module.int32))
            runner_call = maybe_explicit_compile_scan_runner(
                runner,
                (carry_pre, it_seq, *runtime_scan_args),
                cache_status=cache_status,
                scan_timing_enabled=bool(scan_timing_enabled),
                scan_timing_stats=scan_timing_stats,
                perf_counter=perf_counter,
            )
            t_device = perf_counter() if bool(scan_timing_enabled) else None
            carry_final, hist_tail = runner_call(carry_pre, it_seq, *runtime_scan_args)
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
        it_seq = _scan_iteration_sequence(
            0,
            int(max_iter_scan),
            jnp_module=jnp_module,
            build_scan_it_seq=build_scan_it_seq,
        )
        runner_call = maybe_explicit_compile_scan_runner(
            runner,
            (carry_init, it_seq, *runtime_scan_args),
            cache_status=cache_status,
            scan_timing_enabled=bool(scan_timing_enabled),
            scan_timing_stats=scan_timing_stats,
            perf_counter=perf_counter,
        )
        t_device = perf_counter() if bool(scan_timing_enabled) else None
        carry_final, hist = runner_call(carry_init, it_seq, *runtime_scan_args)
        if bool(scan_timing_enabled) and t_device is not None:
            carry_final, hist = scan_device_runtime.ready(
                t_device,
                (carry_final, hist),
                cache_status=cache_status,
            )
        if bool(state_only_scan):
            hist = None
    record_scan_history_summary(
        hist,
        scan_timing_enabled=bool(scan_timing_enabled),
        scan_timing_stats=scan_timing_stats,
    )
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
