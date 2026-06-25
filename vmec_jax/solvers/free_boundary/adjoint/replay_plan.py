"""Replay-plan utility helpers for free-boundary adjoint traces."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from vmec_jax._compat import jnp, tree_util

from .replay_context import direct_coil_boundary_replay_context_for_shape, direct_coil_trace_boundary_shape
from .trace_controls import (
    accepted_trace_effective_controller_masks,
    accepted_trace_segment_is_unconditionally_accepted,
    direct_coil_accepted_trace_controller_controls_jax,
    direct_coil_accepted_trace_status_masks,
)
from .trace_stack import (
    direct_coil_accepted_trace_array_controls_jax,
    direct_coil_accepted_trace_preconditioner_controls_jax,
    direct_coil_accepted_trace_preconditioner_policy_segment_summary,
    direct_coil_accepted_trace_preconditioner_policy_segments,
    direct_coil_accepted_trace_scalar_controls_jax,
    direct_coil_accepted_trace_step_controls_jax,
    direct_coil_accepted_trace_step_policy_segment_summary,
    direct_coil_accepted_trace_step_policy_segments,
)


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


def direct_coil_boundary_replay_contexts_by_shape(static: Any, trace_seq: tuple[Any, ...]) -> dict[tuple[int, int], Any]:
    """Precompute fixed NESTOR replay contexts keyed by active boundary shape."""

    contexts: dict[tuple[int, int], Any] = {}
    for trace in trace_seq:
        shape = direct_coil_trace_boundary_shape(trace)
        if shape is None or shape in contexts:
            continue
        contexts[shape] = direct_coil_boundary_replay_context_for_shape(
            static,
            ntheta=shape[0],
            nzeta=shape[1],
        )
    return contexts


def direct_coil_accepted_trace_controller_replay_plan(
    traces: Any,
    *,
    static: Any,
    accept_mask: Any | None = None,
    done_mask: Any | None = None,
    max_steps: int | None = None,
    use_preconditioner_policy_segments: bool = False,
    use_segment_preconditioner_controls: bool = False,
    use_stacked_step_controls: bool = False,
    use_accepted_only_fast_path: bool = True,
    boundary_replay_contexts_by_shape: Mapping[tuple[int, int], Any] | None = None,
) -> dict[str, Any]:
    """Build fixed accepted-branch replay controls outside AD transforms."""

    trace_seq = tuple(traces)
    if max_steps is not None:
        trace_seq = trace_seq[: int(max_steps)]
    if not trace_seq:
        raise ValueError("at least one accepted trace is required")

    status_masks = direct_coil_accepted_trace_status_masks(trace_seq)
    controller_controls = direct_coil_accepted_trace_controller_controls_jax(
        trace_seq,
        accept_mask=accept_mask,
        done_mask=done_mask,
    )
    effective_masks = accepted_trace_effective_controller_masks(controller_controls)
    scalar_controls = direct_coil_accepted_trace_scalar_controls_jax(trace_seq)
    array_controls = direct_coil_accepted_trace_array_controls_jax(trace_seq)
    step_controls = direct_coil_accepted_trace_step_controls_jax(trace_seq) if bool(use_stacked_step_controls) else None
    step_scalar_controls = {
        key: value
        for key, value in scalar_controls.items()
        if key
        not in (
            "preconditioner_use_precomputed_tridi",
            "preconditioner_use_lax_tridi",
        )
    }
    controls = {**controller_controls, "step_scalars": step_scalar_controls, "step_arrays": array_controls}
    preconditioner_controls = None
    preconditioner_controls_stacked = True
    try:
        preconditioner_controls = direct_coil_accepted_trace_preconditioner_controls_jax(trace_seq)
    except (KeyError, ValueError):
        preconditioner_controls_stacked = False
    else:
        controls = {**controls, "step_preconditioner": preconditioner_controls}
    if step_controls is not None:
        controls = {**controls, "step_controls": step_controls}

    preconditioner_policy_segments = direct_coil_accepted_trace_preconditioner_policy_segments(trace_seq)
    preconditioner_policy_segment_summary = direct_coil_accepted_trace_preconditioner_policy_segment_summary(
        trace_seq,
        accept_mask=accept_mask,
        done_mask=done_mask,
    )
    step_policy_segments = direct_coil_accepted_trace_step_policy_segments(trace_seq)
    step_policy_segment_summary = direct_coil_accepted_trace_step_policy_segment_summary(
        trace_seq,
        accept_mask=accept_mask,
        done_mask=done_mask,
    )

    control_segments: tuple[dict[str, Any], ...] | None = None
    segment_preconditioner_controls_stacked: tuple[bool, ...] = ()
    accepted_only_fast_path_segments: tuple[bool, ...] = ()
    segment_source = "none"
    if bool(use_stacked_step_controls):
        segment_source = "step_policy"
        control_segments_list = []
        segment_preconditioner_controls_stacked_list = []
        accepted_only_fast_path_segments_list = []
        for segment in step_policy_segments:
            start = int(segment["start"])
            stop = int(segment["stop"])
            segment_controls = slice_replay_controls(controls, start=start, stop=stop)
            if preconditioner_controls is None:
                try:
                    segment_preconditioner_controls = direct_coil_accepted_trace_preconditioner_controls_jax(
                        trace_seq[start:stop]
                    )
                except (KeyError, ValueError) as exc:
                    raise ValueError("stacked step replay requires stackable preconditioner controls per segment") from exc
                segment_controls = {**segment_controls, "step_preconditioner": segment_preconditioner_controls}
            segment_preconditioner_controls_stacked_list.append(True)
            accepted_only_fast_path_segments_list.append(
                bool(use_accepted_only_fast_path)
                and accepted_trace_segment_is_unconditionally_accepted(effective_masks, start=start, stop=stop)
            )
            control_segments_list.append(segment_controls)
        control_segments = tuple(control_segments_list)
        segment_preconditioner_controls_stacked = tuple(segment_preconditioner_controls_stacked_list)
        accepted_only_fast_path_segments = tuple(accepted_only_fast_path_segments_list)
    elif bool(use_preconditioner_policy_segments):
        segment_source = "preconditioner_policy"
        control_segments_list = []
        segment_preconditioner_controls_stacked_list = []
        accepted_only_fast_path_segments_list = []
        for segment in preconditioner_policy_segments:
            start = int(segment["start"])
            stop = int(segment["stop"])
            segment_controls = slice_replay_controls(controls, start=start, stop=stop)
            if preconditioner_controls is None and bool(use_segment_preconditioner_controls):
                try:
                    segment_preconditioner_controls = direct_coil_accepted_trace_preconditioner_controls_jax(
                        trace_seq[start:stop]
                    )
                except (KeyError, ValueError):
                    segment_preconditioner_controls_stacked_list.append(False)
                else:
                    segment_controls = {**segment_controls, "step_preconditioner": segment_preconditioner_controls}
                    segment_preconditioner_controls_stacked_list.append(True)
            elif preconditioner_controls is None:
                segment_preconditioner_controls_stacked_list.append(False)
            else:
                segment_preconditioner_controls_stacked_list.append(True)
            accepted_only_fast_path_segments_list.append(
                bool(use_accepted_only_fast_path)
                and accepted_trace_segment_is_unconditionally_accepted(effective_masks, start=start, stop=stop)
            )
            control_segments_list.append(segment_controls)
        control_segments = tuple(control_segments_list)
        segment_preconditioner_controls_stacked = tuple(segment_preconditioner_controls_stacked_list)
        accepted_only_fast_path_segments = tuple(accepted_only_fast_path_segments_list)
    else:
        accepted_only_fast_path_segments = (
            bool(use_accepted_only_fast_path)
            and accepted_trace_segment_is_unconditionally_accepted(effective_masks, start=0, stop=len(trace_seq)),
        )

    context_cache = dict(boundary_replay_contexts_by_shape or {})
    for trace in trace_seq:
        shape = direct_coil_trace_boundary_shape(trace)
        if shape is None or shape in context_cache:
            continue
        context_cache[shape] = direct_coil_boundary_replay_context_for_shape(
            static,
            ntheta=shape[0],
            nzeta=shape[1],
        )

    return {
        "contract": "fixed accepted-branch controller replay plan",
        "differentiates_adaptive_controller": False,
        "traces": trace_seq,
        "status_masks": status_masks,
        "controls": controls,
        "effective_masks": effective_masks,
        "scalar_controls": scalar_controls,
        "array_controls": array_controls,
        "step_controls": step_controls,
        "preconditioner_controls": preconditioner_controls,
        "preconditioner_controls_stacked": bool(preconditioner_controls_stacked),
        "preconditioner_policy_segments": preconditioner_policy_segments,
        "preconditioner_policy_segment_summary": preconditioner_policy_segment_summary,
        "step_policy_segments": step_policy_segments,
        "step_policy_segment_summary": step_policy_segment_summary,
        "control_segments": control_segments,
        "segment_source": segment_source,
        "preconditioner_controls_segment_stacked": segment_preconditioner_controls_stacked,
        "accepted_only_fast_path_segments": accepted_only_fast_path_segments,
        "boundary_replay_contexts_by_shape": context_cache,
        "options": {
            "max_steps": None if max_steps is None else int(max_steps),
            "use_preconditioner_policy_segments": bool(use_preconditioner_policy_segments),
            "use_segment_preconditioner_controls": bool(use_segment_preconditioner_controls),
            "use_stacked_step_controls": bool(use_stacked_step_controls),
            "use_accepted_only_fast_path": bool(use_accepted_only_fast_path),
        },
    }
