"""Runtime hook resolution for VMEC2000-style scan solves.

The numerical scan body should stay focused on force evaluation and VMEC
time-step logic.  Optional host callbacks, tracing hooks, and diagnostic dump
paths are resolved here so the large solver routine does not own that plumbing.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Mapping, NamedTuple

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
