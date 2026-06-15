"""Replay-plan utility helpers for free-boundary adjoint traces."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from vmec_jax._compat import jnp, tree_util

from .free_boundary_adjoint_trace_stack import direct_coil_accepted_trace_step_policy_segments


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


def accepted_step_policy_signature_for_complete_payload(payload: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return branch-sensitive step-policy segment signatures for a complete payload."""

    traces = tuple(payload.get("traces", ()))
    if not traces:
        return ()
    return tuple(
        (
            int(segment["start"]),
            int(segment["stop"]),
            int(segment["n_steps"]),
            segment["signature"],
        )
        for segment in direct_coil_accepted_trace_step_policy_segments(traces)
    )


def accepted_step_policy_layout_for_complete_payload(payload: Mapping[str, Any]) -> tuple[tuple[int, int, int], ...]:
    """Return accepted step-policy segment boundaries for complete-solve branch checks.

    The strict segment helper still owns one-trace segmentation.  Complete-solve
    AD-vs-FD compatibility only needs to know whether base/plus/minus reused
    the same controller slots and segment boundaries; continuous payload values
    may legitimately differ under a finite perturbation and are checked by the
    physical scalar gate.
    """

    traces = tuple(payload.get("traces", ()))
    if not traces:
        return ()
    return tuple(
        (
            int(segment["start"]),
            int(segment["stop"]),
            int(segment["n_steps"]),
        )
        for segment in direct_coil_accepted_trace_step_policy_segments(traces)
    )


def accepted_step_policy_summary_for_complete_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a compact JSON-safe step-policy segment summary for diagnostics."""

    traces = tuple(payload.get("traces", ()))
    if not traces:
        return {"n_segments": 0, "segments": ()}
    segments = direct_coil_accepted_trace_step_policy_segments(traces)
    return {
        "n_segments": len(segments),
        "segments": tuple(
            {
                "start": int(segment["start"]),
                "stop": int(segment["stop"]),
                "n_steps": int(segment["n_steps"]),
            }
            for segment in segments
        ),
    }


def complete_solve_objective_values(value: Any) -> dict[str, float]:
    """Normalize one scalar or a mapping of scalar diagnostics."""

    if isinstance(value, Mapping):
        if not value:
            raise ValueError("objective_fn returned an empty mapping")
        values: dict[str, float] = {}
        for key, item in value.items():
            arr = np.asarray(item, dtype=float)
            if arr.size != 1:
                raise ValueError(f"objective_fn mapping entry {key!r} must be scalar")
            values[str(key)] = float(arr.reshape(-1)[0])
        return values

    arr = np.asarray(value, dtype=float)
    if arr.size != 1:
        raise ValueError("objective_fn must return a scalar or a mapping of scalars")
    return {"objective": float(arr.reshape(-1)[0])}
