"""Helpers for single-stage direct-coil free-boundary optimization.

These helpers intentionally do not run VMEC and do not decide whether a coil
step is accepted.  They only turn a validated same-branch derivative report
into bounded optimizer-coordinate trial points.  A normal complete free-boundary
solve must still evaluate every proposal before it is trusted.
"""

from __future__ import annotations

from copy import deepcopy
import time
from typing import Any, Sequence

import numpy as np

from vmec_jax.external_fields import CoilFieldParams, build_coil_field_geometry

__all__ = [
    "nestor_profile_policy_from_results",
    "same_branch_current_only_coil_geometry_cache",
    "same_branch_derivative_gate_evidence",
    "same_branch_derivative_proposal_from_report",
    "same_branch_derivative_proposals_from_report",
    "same_branch_rejected_slot_gate_from_vector_replay",
    "same_branch_replay_plan_cache",
    "same_branch_report_mode_count",
]


def same_branch_report_mode_count(report: dict[str, Any]) -> int:
    """Return the VMEC Fourier mode count for report-size policy decisions."""

    try:
        static = report["base"]["init"].static
        return int(np.asarray(static.modes.m).size)
    except Exception:
        return 0


def same_branch_replay_plan_cache(
    report: dict[str, Any],
    replay_kwargs: dict[str, Any],
    *,
    timing_key: str,
    scope: str,
) -> tuple[dict[str, Any] | None, dict[str, Any], float | None]:
    """Build an accepted-trace replay plan for repeated same-branch reports."""

    from vmec_jax.free_boundary_adjoint import direct_coil_accepted_trace_controller_replay_plan

    try:
        t0 = time.perf_counter()
        replay_plan = direct_coil_accepted_trace_controller_replay_plan(
            tuple(report["base"]["traces"]),
            static=report["base"]["init"].static,
            use_preconditioner_policy_segments=bool(
                replay_kwargs.get("use_preconditioner_policy_segments", False)
            ),
            use_segment_preconditioner_controls=bool(
                replay_kwargs.get("use_segment_preconditioner_controls", False)
            ),
            use_stacked_step_controls=bool(replay_kwargs.get("use_stacked_step_controls", False)),
            use_accepted_only_fast_path=bool(replay_kwargs.get("use_accepted_only_fast_path", True)),
        )
        return replay_plan, {"available": True, "timing_key": timing_key, "scope": scope}, float(
            time.perf_counter() - t0
        )
    except Exception as exc:  # pragma: no cover - synthetic tests may omit stackable trace controls.
        return None, {"available": False, "reason": f"{type(exc).__name__}: {exc}", "scope": scope}, None


def same_branch_current_only_coil_geometry_cache(
    params: CoilFieldParams,
    direction_params: CoilFieldParams,
) -> tuple[tuple[Any, Any] | None, dict[str, Any], float | None]:
    """Cache fixed coil geometry when same-branch reports vary currents only."""

    try:
        direction_dofs = np.asarray(direction_params.base_curve_dofs, dtype=float)
        if np.any(direction_dofs):
            return None, {"available": False, "reason": "direction includes coil-shape dofs"}, None
        t0 = time.perf_counter()
        gamma, gamma_dash, _currents = build_coil_field_geometry(params)
        return (
            (gamma, gamma_dash),
            {
                "available": True,
                "scope": "current-only branch-local vector/profile replays",
                "timing_key": "branch_local_current_only_coil_geometry_build_wall_s",
            },
            float(time.perf_counter() - t0),
        )
    except Exception as exc:  # pragma: no cover - defensive; report artifacts should not abort examples.
        return None, {"available": False, "reason": f"{type(exc).__name__}: {exc}"}, None


def nestor_profile_policy_from_results(
    results: list[dict[str, Any]],
    *,
    mode_count: int,
    min_mode_count: int,
    min_speedup: float,
) -> dict[str, Any]:
    """Decide whether matrix-free NESTOR should be promoted for this report."""

    dense = [item for item in results if item.get("nestor_solve_mode") == "dense" and item.get("available")]
    matrix_free = [
        item
        for item in results
        if item.get("nestor_solve_mode") == "matrix_free" and item.get("available")
    ]
    if not dense:
        return {
            "promote_matrix_free": False,
            "reason": "dense baseline timing is unavailable",
            "mode_count": int(mode_count),
        }
    if not matrix_free:
        return {
            "promote_matrix_free": False,
            "reason": "matrix-free timing is unavailable",
            "mode_count": int(mode_count),
        }
    dense_best_entry = min(dense, key=lambda item: float(item["wall_s"]))
    dense_best = float(dense_best_entry["wall_s"])
    mf_best_entry = min(matrix_free, key=lambda item: float(item["wall_s"]))
    mf_best = float(mf_best_entry["wall_s"])
    speedup = dense_best / mf_best if mf_best > 0.0 else np.inf
    if int(mode_count) < int(min_mode_count):
        reason = f"mode_count {int(mode_count)} below threshold {int(min_mode_count)}"
        promote = False
    elif speedup < float(min_speedup):
        reason = f"matrix-free speedup {speedup:.3g} below threshold {float(min_speedup):.3g}"
        promote = False
    else:
        reason = "matrix-free is faster beyond the configured mode-count and speedup thresholds"
        promote = True
    return {
        "promote_matrix_free": bool(promote),
        "reason": reason,
        "mode_count": int(mode_count),
        "min_mode_count": int(min_mode_count),
        "min_speedup": float(min_speedup),
        "dense_best_wall_s": dense_best,
        "matrix_free_best_wall_s": mf_best,
        "matrix_free_best_solver": str(mf_best_entry.get("nestor_operator_solver", "unknown")),
        "speedup_dense_over_matrix_free": float(speedup),
        "recommended_report_options": {
            "same_branch_report_nestor_solve_mode": "matrix_free" if promote else "dense",
            "same_branch_report_nestor_operator_solver": str(
                mf_best_entry.get("nestor_operator_solver", "gmres")
            )
            if promote
            else str(dense_best_entry.get("nestor_operator_solver", "gmres")),
            "reason": "use promoted matrix-free replay settings" if promote else "keep dense replay settings",
        },
    }


def same_branch_rejected_slot_gate_from_vector_replay(
    *,
    requested: bool,
    same_branch: bool,
    replay_mode_count_guard_triggered: bool,
    replay_mode_count_guard_reason: str,
    mode: str,
    report: dict[str, Any],
    missing_vector_keys: tuple[str, ...],
    vector_keys: tuple[str, ...],
    replay_kwargs: dict[str, Any],
    vector_uses_state_only_replay: bool,
    run_branch_local_vector: Any,
    summarize_vector_result: Any,
) -> tuple[dict[str, Any], float | None]:
    """Return the fixed accepted/rejected controller-slot gate artifact.

    This is a branch-local replay gate: it checks whether a fixed rejected
    controller slot can be replayed under the same fingerprint.  It does not
    claim derivatives through arbitrary host-side adaptive branch selection.
    """

    gate: dict[str, Any] = {
        "available": False,
        "requested": bool(requested),
        "passed": False,
        "reason": "not requested",
        "differentiates_adaptive_controller": False,
        "differentiates_run_free_boundary": False,
        "same_stacked_step_policy_branch": False,
    }
    if not requested:
        return gate, None
    if replay_mode_count_guard_triggered:
        gate["reason"] = replay_mode_count_guard_reason
        return gate, None
    if not (same_branch and mode == "vector" and "base" in report and not missing_vector_keys):
        gate["reason"] = "requires same-branch vector report with all requested scalar keys"
        return gate, None
    base_traces = tuple(report["base"].get("traces", ()))
    if not base_traces:
        gate["reason"] = "base complete-solve payload has no traces"
        return gate, None

    rejected_trace = deepcopy(base_traces[-1])
    rejected_trace["step_status"] = "rejected"
    padded_traces = base_traces + (rejected_trace,)
    t0 = time.perf_counter()
    rejected_vector = run_branch_local_vector(
        vector_keys,
        {
            **replay_kwargs,
            "state_only_replay": vector_uses_state_only_replay,
            "traces": padded_traces,
            "use_accepted_only_fast_path": False,
        },
        include_replay_graph_metadata=False,
    )
    wall_s = float(time.perf_counter() - t0)
    rejected_summary = summarize_vector_result(rejected_vector, vector_keys)
    rejected_metadata = rejected_summary.get("replay_branch_metadata", {})
    rejected_controller_slot_summary = rejected_summary.get("controller_slot_summary", {})
    rejected_mask = np.asarray(rejected_metadata.get("rejected_mask", []), dtype=bool)
    passed = bool(
        same_branch
        and rejected_summary["replay_option_flags"].get("use_stacked_step_controls", False)
        and not rejected_summary["replay_option_flags"].get("use_accepted_only_fast_path", True)
        and np.any(rejected_mask)
        and np.isfinite(float(rejected_summary["max_base_abs_delta"]))
        and float(rejected_summary["max_base_abs_delta"]) <= 2.0e-3
        and not bool(rejected_summary.get("differentiates_adaptive_controller", True))
        and not bool(rejected_summary.get("differentiates_run_free_boundary", True))
        and bool(rejected_summary.get("differentiates_fixed_accepted_branch", False))
    )
    return {
        "available": True,
        "requested": True,
        "passed": passed,
        "scope": (
            "fixed accepted/rejected controller-slot replay; "
            "does not differentiate adaptive host branch selection"
        ),
        "differentiates_adaptive_controller": False,
        "differentiates_run_free_boundary": False,
        "same_branch": same_branch,
        "same_stacked_step_policy_branch": bool(
            rejected_summary["replay_option_flags"].get("use_stacked_step_controls", False)
        ),
        "scalar_keys": list(vector_keys),
        "fixed_rejected_controller_slot_present": bool(np.any(rejected_mask)),
        "fixed_rejected_controller_slots": int(np.count_nonzero(rejected_mask)),
        "directional_jvp_fast_path": str(rejected_summary.get("directional_jvp_fast_path", "none")),
        "directional_uses_fixed_coil_geometry": bool(
            rejected_summary.get("directional_uses_fixed_coil_geometry", False)
        ),
        "controller_slot_summary": rejected_controller_slot_summary,
        "replay_option_flags": rejected_summary["replay_option_flags"],
        "replay_branch_metadata": rejected_metadata,
        "max_base_abs_delta": float(rejected_summary["max_base_abs_delta"]),
        "scalars": rejected_summary["scalars"],
        "wall_s": wall_s,
    }, wall_s


def same_branch_derivative_proposal_from_report(
    report: dict[str, Any],
    objective_model: dict[str, Any],
    best: dict[str, Any] | None,
    *,
    step_size: float,
    max_base_abs_delta: float = 2.0e-3,
) -> dict[str, Any]:
    """Return one conservative derivative-assisted proposal from a report."""

    proposals = same_branch_derivative_proposals_from_report(
        report,
        objective_model,
        best,
        step_sizes=(float(step_size),),
        max_base_abs_delta=float(max_base_abs_delta),
        max_trials=1,
    )
    if proposals and proposals[0].get("available", False):
        return proposals[0]
    if proposals:
        return proposals[0]
    return {"available": False, "reason": "no same-branch derivative proposal was generated"}


def same_branch_derivative_gate_evidence(report: dict[str, Any]) -> dict[str, Any]:
    """Return compact gate evidence attached to derivative-assisted proposals."""

    vector = report.get("branch_local_vector_jacobian", {})
    replay_flags = vector.get("replay_option_flags", {}) if isinstance(vector, dict) else {}
    current_only_cache = report.get("current_only_coil_geometry_cache", {})
    vector_gate = report.get("branch_local_vector_gate", {})
    physical_gate = vector_gate.get("physical_scalar_gate", {}) if isinstance(vector_gate, dict) else {}
    rejected_slot_gate = report.get("accepted_rejected_controller_slot_gate", {})
    rejected_slot_requested = isinstance(rejected_slot_gate, dict) and bool(rejected_slot_gate.get("requested", False))
    return {
        "directional_jvp_fast_path": str(
            vector.get("directional_jvp_fast_path", replay_flags.get("directional_jvp_fast_path", "none"))
            if isinstance(vector, dict)
            else "none"
        ),
        "directional_uses_fixed_coil_geometry": bool(
            vector.get(
                "directional_uses_fixed_coil_geometry",
                replay_flags.get("directional_uses_fixed_coil_geometry", False),
            )
            if isinstance(vector, dict)
            else False
        ),
        "current_only_coil_geometry_cache_available": bool(
            isinstance(current_only_cache, dict) and current_only_cache.get("available", False)
        ),
        "current_only_coil_geometry_cache_reason": str(
            current_only_cache.get("reason", "") if isinstance(current_only_cache, dict) else ""
        ),
        "current_only_coil_geometry_source": str(
            replay_flags.get("current_only_coil_geometry_source", "")
            if isinstance(replay_flags, dict)
            else ""
        ),
        "branch_local_vector_gate_available": bool(
            isinstance(vector_gate, dict) and vector_gate.get("available", False)
        ),
        "branch_local_vector_gate_passed": bool(
            isinstance(vector_gate, dict) and vector_gate.get("passed", False)
        ),
        "physical_scalar_gate_passed": bool(
            isinstance(physical_gate, dict) and physical_gate.get("passed", False)
        ),
        "accepted_rejected_controller_slot_gate_requested": bool(rejected_slot_requested),
        "accepted_rejected_controller_slot_gate_available": bool(
            isinstance(rejected_slot_gate, dict) and rejected_slot_gate.get("available", False)
        ),
        "accepted_rejected_controller_slot_gate_passed": bool(
            isinstance(rejected_slot_gate, dict) and rejected_slot_gate.get("passed", False)
        ),
        "accepted_rejected_controller_slot_scope": str(
            rejected_slot_gate.get("scope", "") if isinstance(rejected_slot_gate, dict) else ""
        ),
        "same_stacked_step_policy_branch": bool(
            isinstance(rejected_slot_gate, dict) and rejected_slot_gate.get("same_stacked_step_policy_branch", False)
        ),
        "fixed_rejected_controller_slots": int(
            rejected_slot_gate.get("fixed_rejected_controller_slots", 0)
            if isinstance(rejected_slot_gate, dict)
            else 0
        ),
        "controller_slot_summary": (
            dict(rejected_slot_gate.get("controller_slot_summary", {}))
            if isinstance(rejected_slot_gate, dict)
            and isinstance(rejected_slot_gate.get("controller_slot_summary", {}), dict)
            else {}
        ),
    }


def same_branch_derivative_proposals_from_report(
    report: dict[str, Any],
    objective_model: dict[str, Any],
    best: dict[str, Any] | None,
    *,
    step_sizes: Sequence[float],
    max_base_abs_delta: float = 2.0e-3,
    max_trials: int | None = None,
) -> list[dict[str, Any]]:
    """Return bounded derivative-assisted proposals from one same-branch report.

    Each proposal uses the same validated fixed-accepted-branch directional JVP
    and differs only by optimizer-coordinate step length.  Every returned
    ``trial_x`` is still a suggestion; the production complete solve remains
    the sole acceptance authority.
    """

    if best is None or "x" not in best:
        return [{"available": False, "reason": "no best point is available"}]
    raw_step_sizes = [float(step) for step in step_sizes]
    step_sizes = [step for step in raw_step_sizes if np.isfinite(step) and step > 0.0]
    if not step_sizes:
        return [{"available": False, "reason": "no positive finite proposal step sizes were requested"}]
    if max_trials is not None and int(max_trials) > 0:
        step_sizes = step_sizes[: int(max_trials)]
    vector = report.get("branch_local_vector_jacobian", {})
    if not bool(vector.get("available", False)):
        return [{"available": False, "reason": str(vector.get("reason", "branch-local vector report unavailable"))}]
    same_branch = bool(report.get("branch_compatibility", {}).get("same_branch", vector.get("same_branch", False)))
    if not same_branch:
        return [{"available": False, "reason": "complete-solve finite-difference branch fingerprint is not unchanged"}]
    if not bool(vector.get("uses_production_forward", False)):
        return [{"available": False, "reason": "branch-local vector report did not use production-forward scalar values"}]
    if bool(vector.get("differentiates_adaptive_controller", True)):
        return [{"available": False, "reason": "branch-local vector report claims adaptive-controller differentiation"}]
    if bool(vector.get("differentiates_run_free_boundary", True)):
        return [{"available": False, "reason": "branch-local vector report claims run_free_boundary differentiation"}]
    if not bool(vector.get("differentiates_fixed_accepted_branch", False)):
        return [{"available": False, "reason": "branch-local vector report does not differentiate a fixed accepted branch"}]
    replay_ad_mode = str(vector.get("replay_ad_mode", "")).strip().lower()
    if replay_ad_mode != "direct":
        return [{"available": False, "reason": "branch-local proposal requires direct JVP replay_ad_mode"}]
    derivative_mode = str(vector.get("derivative_mode", "")).strip().lower()
    if derivative_mode != "directional_jvp":
        return [{"available": False, "reason": "branch-local proposal requires directional_jvp derivative_mode"}]
    report_base_delta = float(vector.get("max_base_abs_delta", np.inf))
    if not np.isfinite(report_base_delta):
        return [{"available": False, "reason": "branch-local vector report has non-finite replay base delta"}]
    if report_base_delta > float(max_base_abs_delta):
        return [
            {
                "available": False,
                "reason": (
                    f"branch-local replay base delta {report_base_delta:.3e} exceeds proposal cap "
                    f"{float(max_base_abs_delta):.3e}"
                ),
            }
        ]
    vector_gate = report.get("branch_local_vector_gate")
    if isinstance(vector_gate, dict) and bool(vector_gate.get("available", False)):
        if not bool(vector_gate.get("passed", False)):
            return [{"available": False, "reason": "branch-local vector gate did not pass"}]
        physical_gate = vector_gate.get("physical_scalar_gate", {})
        if isinstance(physical_gate, dict) and not bool(physical_gate.get("passed", False)):
            return [{"available": False, "reason": "branch-local physical-scalar gate did not pass"}]
    rejected_slot_gate = report.get("accepted_rejected_controller_slot_gate")
    if isinstance(rejected_slot_gate, dict) and bool(rejected_slot_gate.get("requested", False)):
        if not bool(rejected_slot_gate.get("available", False)):
            return [
                {
                    "available": False,
                    "reason": str(
                        rejected_slot_gate.get(
                            "reason",
                            "requested accepted/rejected controller-slot gate is unavailable",
                        )
                    ),
                }
            ]
        if not bool(rejected_slot_gate.get("passed", False)):
            return [{"available": False, "reason": "accepted/rejected controller-slot gate did not pass"}]

    scalars = vector.get("scalars", {})
    contributions: dict[str, dict[str, float]] = {}
    omitted_terms: dict[str, dict[str, Any]] = {}
    directional = 0.0

    def _validated_scalar(key: str, weight: float) -> dict[str, Any] | None:
        if float(weight) == 0.0:
            return None
        scalar = scalars.get(key)
        if scalar is None:
            omitted_terms[key] = {
                "weight": float(weight),
                "reason": "not included in branch-local vector/JVP report",
            }
            return None
        value = float(scalar.get("value", np.nan))
        deriv = float(scalar.get("exact_directional", np.nan))
        base_delta = float(scalar.get("base_abs_delta", 0.0))
        if not (np.isfinite(value) and np.isfinite(deriv) and np.isfinite(base_delta)):
            raise ValueError(f"non-finite branch-local scalar evidence for {key}")
        if base_delta > float(max_base_abs_delta):
            raise ValueError(
                f"branch-local scalar {key} base delta {base_delta:.3e} exceeds proposal cap "
                f"{float(max_base_abs_delta):.3e}"
            )
        return {"value": value, "exact_directional": deriv, "base_abs_delta": base_delta}

    if float(objective_model.get("residual_weight", 0.0)) != 0.0:
        omitted_terms["residual_proxy"] = {
            "weight": float(objective_model.get("residual_weight", 0.0)),
            "reason": (
                "not included in branch-local vector/JVP report; the complete "
                "free-boundary solve remains acceptance authority"
            ),
        }

    try:
        qs_scalar = _validated_scalar("qs_total", float(objective_model.get("qs_weight", 0.0)))
        aspect_scalar = _validated_scalar("aspect", float(objective_model.get("aspect_weight", 0.0)))
        iota_scalar = _validated_scalar("mean_iota", float(objective_model.get("iota_weight", 0.0)))
    except ValueError as exc:
        return [{"available": False, "reason": str(exc)}]

    if qs_scalar is not None:
        deriv = float(qs_scalar["exact_directional"])
        contribution = float(objective_model.get("qs_weight", 0.0)) * deriv
        contributions["qs_total"] = {
            "exact_directional": deriv,
            "base_abs_delta": float(qs_scalar["base_abs_delta"]),
            "contribution": contribution,
        }
        directional += contribution

    if aspect_scalar is not None:
        value = float(aspect_scalar["value"])
        deriv = float(aspect_scalar["exact_directional"])
        target = float(objective_model.get("target_aspect", value))
        contribution = 2.0 * float(objective_model.get("aspect_weight", 0.0)) * (value - target) * deriv
        contributions["aspect"] = {
            "value": value,
            "target": target,
            "exact_directional": deriv,
            "base_abs_delta": float(aspect_scalar["base_abs_delta"]),
            "contribution": contribution,
        }
        directional += contribution

    if iota_scalar is not None:
        value = float(iota_scalar["value"])
        deriv = float(iota_scalar["exact_directional"])
        target = float(objective_model.get("target_iota", value))
        contribution = 2.0 * float(objective_model.get("iota_weight", 0.0)) * (value - target) * deriv
        contributions["mean_iota"] = {
            "value": value,
            "target": target,
            "exact_directional": deriv,
            "base_abs_delta": float(iota_scalar["base_abs_delta"]),
            "contribution": contribution,
        }
        directional += contribution

    if not contributions:
        return [{"available": False, "reason": "no report scalars map to the objective terms"}]
    if not np.isfinite(directional):
        return [{"available": False, "reason": "non-finite directional derivative"}]
    if directional == 0.0:
        return [{"available": False, "reason": "zero directional derivative"}]

    direction_x = np.asarray(report.get("direction_x", []), dtype=float)
    x_best = np.asarray(best["x"], dtype=float)
    if direction_x.shape != x_best.shape:
        return [
            {
                "available": False,
                "reason": f"direction_x shape {direction_x.shape} does not match best x shape {x_best.shape}",
            }
        ]

    gate_evidence = same_branch_derivative_gate_evidence(report)
    proposals = []
    for trial_index, step_size in enumerate(step_sizes):
        alpha = -float(step_size) * float(np.sign(directional))
        trial_x = x_best + alpha * direction_x
        proposals.append(
            {
                "available": True,
                "scope": "fixed accepted-branch directional proposal; complete solve decides acceptance",
                "same_branch": True,
                "uses_production_forward": True,
                "replay_ad_mode": replay_ad_mode,
                "derivative_mode": derivative_mode,
                "differentiates_adaptive_controller": False,
                "differentiates_run_free_boundary": False,
                "differentiates_fixed_accepted_branch": True,
                "complete_solve_acceptance_authority": True,
                "max_base_abs_delta": report_base_delta,
                "max_base_abs_delta_allowed": float(max_base_abs_delta),
                "directional_derivative": float(directional),
                "contributions": contributions,
                "gate_evidence": gate_evidence,
                "objective_terms_used": sorted(contributions),
                "objective_terms_omitted": omitted_terms,
                "alpha": float(alpha),
                "step_size": float(step_size),
                "trial_index": int(trial_index),
                "n_requested_trials": int(len(step_sizes)),
                "direction_x": direction_x.tolist(),
                "base_x": x_best.tolist(),
                "trial_x": trial_x.tolist(),
            }
        )
    return proposals
