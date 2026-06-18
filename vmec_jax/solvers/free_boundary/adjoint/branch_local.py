"""Branch-local production-report helpers for direct-coil free-boundary adjoints.

These helpers assemble the non-numerical parts shared by scalar and vector
branch-local reports: complete-solve payload validation, production scalar
evaluation, fixed accepted-branch replay options, and compact report flags.
They deliberately do not claim differentiation through arbitrary adaptive host
branch changes.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any, Callable, NamedTuple

from .branch_metadata import (
    direct_coil_accepted_trace_branch_metadata,
    direct_coil_accepted_trace_replay_graph_metadata,
)
from .replay_plan import (
    complete_solve_objective_values,
    direct_coil_accepted_trace_controller_replay_plan,
)
from .trace_metadata import direct_coil_accepted_trace_controller_slot_summary


class BranchLocalPayload(NamedTuple):
    """Validated complete-solve payload and timing data."""

    payload: dict[str, Any]
    params: Any
    traces: tuple[Any, ...]
    init: Any
    timings: dict[str, float]


class BranchLocalReplaySetup(NamedTuple):
    """Replay configuration shared by branch-local scalar/vector reports."""

    replay_options: dict[str, Any]
    replay_traces: tuple[Any, ...]
    replay_payload: Any
    replay_payload_source: str
    replay_plan: Any
    replay_branch_metadata: dict[str, Any]
    controller_slot_summary: dict[str, Any]
    graph_metadata: dict[str, Any]


def prepare_branch_local_payload(
    *,
    input_path: Any | None,
    params: Any | None,
    complete_payload: Mapping[str, Any] | None,
    init_kwargs: dict[str, Any] | None,
    solve_kwargs: dict[str, Any] | None,
    require_active_trace: bool,
    complete_solve_trace_func: Callable[..., Mapping[str, Any]],
    perf_counter: Callable[[], float] = time.perf_counter,
) -> BranchLocalPayload:
    """Return a validated direct-coil complete-solve payload."""

    timings: dict[str, float] = {}
    if complete_payload is None:
        if input_path is None or params is None:
            raise ValueError("input_path and params are required when complete_payload is not supplied")
        t0 = perf_counter()
        payload = dict(
            complete_solve_trace_func(
                input_path,
                params,
                init_kwargs=init_kwargs,
                solve_kwargs=solve_kwargs,
                require_active_trace=require_active_trace,
            )
        )
        timings["complete_solve_trace_wall_s"] = float(perf_counter() - t0)
    else:
        t0 = perf_counter()
        payload = dict(complete_payload)
        if params is None:
            params = payload.get("params")
        if params is None:
            raise ValueError("params must be supplied when complete_payload does not contain params")
        timings["payload_copy_wall_s"] = float(perf_counter() - t0)

    traces = tuple(payload.get("traces", ()))
    if not traces:
        raise ValueError("complete payload contains no accepted traces")
    active_trace = any(trace.get("freeb_bsqvac_half") is not None for trace in traces)
    if bool(require_active_trace) and not active_trace:
        raise RuntimeError("complete payload contains no active free-boundary trace")
    init = payload.get("init")
    if init is None:
        raise ValueError("complete payload is missing the initialization result")
    return BranchLocalPayload(payload=payload, params=params, traces=traces, init=init, timings=timings)


def evaluate_branch_local_production_values(
    *,
    payload: Mapping[str, Any],
    scalar_fn: Callable[[Mapping[str, Any]], Any],
    production_values: Mapping[str, Any] | None,
    timings: dict[str, float],
    perf_counter: Callable[[], float] = time.perf_counter,
) -> tuple[dict[str, float], str]:
    """Evaluate or normalize production scalar values from a complete solve."""

    t0 = perf_counter()
    values = complete_solve_objective_values(
        scalar_fn(payload) if production_values is None else production_values
    )
    timings["production_scalar_eval_wall_s"] = float(perf_counter() - t0)
    return values, ("scalar_fn" if production_values is None else "precomputed")


def select_branch_local_scalar_key(values: Mapping[str, Any], scalar_key: str | None) -> str:
    """Select a scalar key from production values using project conventions."""

    key = str(scalar_key or ("objective" if "objective" in values else next(iter(values))))
    if key not in values:
        raise KeyError(f"scalar_key {key!r} not returned by scalar_fn")
    return key


def select_branch_local_scalar_keys(
    *,
    all_values: Mapping[str, Any],
    replay_scalar_fns: Mapping[str, Any],
    scalar_keys: tuple[str, ...] | list[str] | None,
) -> tuple[str, ...]:
    """Validate and return vector-report scalar keys."""

    keys = tuple(str(key) for key in (scalar_keys if scalar_keys is not None else tuple(replay_scalar_fns)))
    if not keys:
        raise ValueError("scalar_keys must contain at least one scalar")
    for key in keys:
        if key not in all_values:
            raise KeyError(f"scalar_key {key!r} not returned by scalar_fn")
        if key not in replay_scalar_fns:
            raise KeyError(f"scalar_key {key!r} not present in replay_scalar_fns")
    return keys


def prepare_branch_local_replay_setup(
    *,
    init: Any,
    traces: tuple[Any, ...],
    replay_kwargs: dict[str, Any] | None,
    replay_payload: Mapping[str, Any] | None,
    payload: Mapping[str, Any],
    replay_plan: Any,
    use_replay_plan: bool,
    include_replay_graph_metadata: bool,
    timings: dict[str, float],
    perf_counter: Callable[[], float] = time.perf_counter,
) -> BranchLocalReplaySetup:
    """Assemble fixed accepted-branch replay options and metadata."""

    replay_options: dict[str, Any] = {
        "static": init.static,
        "traces": traces,
        "signgs": int(init.signgs),
        "state_weight": 0.0,
        "bsqvac_weight": 0.0,
        "force_weight": 0.0,
        "enforce_edge": False,
        "use_preconditioner_policy_segments": True,
        "use_stacked_step_controls": True,
        "include_replay_aux": False,
        "unroll_accepted_only_segments_below": 8,
    }
    if replay_kwargs:
        replay_options.update(replay_kwargs)

    replay_traces = tuple(replay_options.get("traces", traces))
    replay_branch_metadata = direct_coil_accepted_trace_branch_metadata(
        replay_traces,
        accept_mask=replay_options.get("accept_mask"),
        done_mask=replay_options.get("done_mask"),
        max_steps=replay_options.get("max_steps"),
        json_safe=True,
    )
    controller_slot_summary = direct_coil_accepted_trace_controller_slot_summary(replay_branch_metadata)
    replay_payload_for_scalars = payload if replay_payload is None else replay_payload
    replay_payload_source = "complete_payload" if replay_payload is None else "user"

    replay_plan_for_scalars = replay_plan
    if replay_plan_for_scalars is None and bool(use_replay_plan):
        t0 = perf_counter()
        replay_plan_for_scalars = direct_coil_accepted_trace_controller_replay_plan(
            replay_traces,
            static=init.static,
            accept_mask=replay_options.get("accept_mask"),
            done_mask=replay_options.get("done_mask"),
            max_steps=replay_options.get("max_steps"),
            use_preconditioner_policy_segments=bool(
                replay_options.get("use_preconditioner_policy_segments", False)
            ),
            use_segment_preconditioner_controls=bool(
                replay_options.get("use_segment_preconditioner_controls", False)
            ),
            use_stacked_step_controls=bool(replay_options.get("use_stacked_step_controls", False)),
            use_accepted_only_fast_path=bool(replay_options.get("use_accepted_only_fast_path", True)),
        )
        timings["replay_plan_build_wall_s"] = float(perf_counter() - t0)
    else:
        timings["replay_plan_build_wall_s"] = 0.0

    t0 = perf_counter()
    if bool(include_replay_graph_metadata):
        graph_metadata = direct_coil_accepted_trace_replay_graph_metadata(
            replay_traces,
            static=init.static,
            accept_mask=replay_options.get("accept_mask"),
            done_mask=replay_options.get("done_mask"),
            max_steps=replay_options.get("max_steps"),
            sample_nzeta=replay_options.get("sample_nzeta"),
            include_analytic=bool(replay_options.get("include_analytic", True)),
            use_stacked_step_controls=bool(replay_options.get("use_stacked_step_controls", False)),
            use_accepted_only_fast_path=bool(replay_options.get("use_accepted_only_fast_path", True)),
            json_safe=True,
        )
    else:
        graph_metadata = {
            "contract": "fixed accepted-branch replay graph metadata",
            "omitted": True,
            "reason": "include_replay_graph_metadata=False",
            "differentiates_adaptive_controller": False,
        }
    timings["replay_graph_metadata_wall_s"] = float(perf_counter() - t0)

    return BranchLocalReplaySetup(
        replay_options=replay_options,
        replay_traces=replay_traces,
        replay_payload=replay_payload_for_scalars,
        replay_payload_source=replay_payload_source,
        replay_plan=replay_plan_for_scalars,
        replay_branch_metadata=replay_branch_metadata,
        controller_slot_summary=controller_slot_summary,
        graph_metadata=graph_metadata,
    )


def branch_local_replay_option_flags(
    replay_options: Mapping[str, Any],
    *,
    replay_plan: Any,
    ad_mode: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return compact replay option flags for user-facing reports."""

    flags: dict[str, Any] = {
        "use_preconditioner_policy_segments": bool(
            replay_options.get("use_preconditioner_policy_segments", False)
        ),
        "use_stacked_step_controls": bool(replay_options.get("use_stacked_step_controls", False)),
        "use_accepted_only_fast_path": bool(replay_options.get("use_accepted_only_fast_path", True)),
        "use_replay_plan": bool(replay_plan is not None),
        "include_replay_aux": bool(replay_options.get("include_replay_aux", True)),
        "include_analytic": bool(replay_options.get("include_analytic", True)),
        "include_mode_diagnostics": bool(replay_options.get("include_mode_diagnostics", False)),
        "nestor_solve_mode": str(replay_options.get("nestor_solve_mode", "dense")),
        "nestor_operator_solver": str(replay_options.get("nestor_operator_solver", "gmres")),
        "nestor_operator_tol": float(replay_options.get("nestor_operator_tol", 1.0e-11)),
        "nestor_operator_atol": float(replay_options.get("nestor_operator_atol", 1.0e-13)),
        "nestor_operator_maxiter": (
            None
            if replay_options.get("nestor_operator_maxiter") is None
            else int(replay_options.get("nestor_operator_maxiter"))
        ),
        "nestor_operator_restart": (
            None
            if replay_options.get("nestor_operator_restart") is None
            else int(replay_options.get("nestor_operator_restart"))
        ),
        "freeze_vacuum_field": bool(replay_options.get("freeze_vacuum_field", False)),
        "freeze_freeb_bsqvac": bool(replay_options.get("freeze_freeb_bsqvac", False)),
        "state_only_replay": bool(replay_options.get("state_only_replay", False)),
        "jit_preconditioner_apply": bool(replay_options.get("jit_preconditioner_apply", True)),
        "unroll_accepted_only_segments_below": int(
            replay_options.get("unroll_accepted_only_segments_below", 0)
        ),
        "replay_ad_mode": ad_mode,
    }
    if extra:
        flags.update(dict(extra))
    return flags
