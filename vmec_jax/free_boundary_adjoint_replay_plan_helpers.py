"""Replay-plan utility helpers for free-boundary adjoint traces."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vmec_jax._compat import jnp, tree_util


def slice_replay_controls(controls: Mapping[str, Any], *, start: int, stop: int) -> dict[str, Any]:
    """Slice stacked replay controls without rebuilding them from traces."""

    return tree_util.tree_map(
        lambda value, start=start, stop=stop: jnp.asarray(value)[start:stop],
        controls,
    )


def extract_adjoint_step_trace(source: Any) -> tuple[Any, ...]:
    """Extract an ``adjoint_step_trace`` tuple from common solver/report containers."""

    if isinstance(source, Mapping):
        if "adjoint_step_trace" in source:
            return tuple(source["adjoint_step_trace"])
        if "diagnostics" in source and isinstance(source["diagnostics"], Mapping):
            diagnostics = source["diagnostics"]
            if "adjoint_step_trace" in diagnostics:
                return tuple(diagnostics["adjoint_step_trace"])
    diagnostics = getattr(source, "diagnostics", None)
    if isinstance(diagnostics, Mapping) and "adjoint_step_trace" in diagnostics:
        return tuple(diagnostics["adjoint_step_trace"])
    result = getattr(source, "result", None)
    result_diagnostics = getattr(result, "diagnostics", None)
    if isinstance(result_diagnostics, Mapping) and "adjoint_step_trace" in result_diagnostics:
        return tuple(result_diagnostics["adjoint_step_trace"])
    if isinstance(source, (str, bytes)):
        raise RuntimeError(
            "No adjoint_step_trace found. Run the residual solver with "
            "adjoint_trace=True and adjoint_trace_mode='full'."
        )
    try:
        traces = tuple(source)
    except TypeError as exc:
        raise RuntimeError(
            "No adjoint_step_trace found. Run the residual solver with "
            "adjoint_trace=True and adjoint_trace_mode='full'."
        ) from exc
    if traces and all(isinstance(trace, Mapping) for trace in traces):
        return traces
    raise RuntimeError(
        "No adjoint_step_trace found. Run the residual solver with "
        "adjoint_trace=True and adjoint_trace_mode='full'."
    )


def stackability_probe(name: str, fn: Any, traces: tuple[Any, ...]) -> tuple[bool, str | None]:
    """Return whether a trace-stacking helper accepts the supplied traces."""

    try:
        fn(traces)
    except Exception as exc:
        return False, f"{name}: {exc}"
    return True, None
