"""Reviewer-facing free-boundary adjoint gate reports.

These helpers assemble conservative promotion reports for branch-local
direct-coil adjoints.  They intentionally keep ``differentiates_adaptive_controller``
false: passing these gates validates a fixed accepted/rejected branch against
complete-solve finite differences, not arbitrary host adaptive branch changes.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .replay_plan import (
    accepted_step_policy_layout_for_complete_payload,
    accepted_step_policy_summary_for_complete_payload,
)
from .trace_metadata import (
    _fingerprint_has_rejected_controller_slot,
    _json_safe_fingerprint_value,
    direct_coil_accepted_trace_controller_slot_fingerprint,
    direct_coil_accepted_trace_controller_slot_summary,
)


def direct_coil_same_branch_replay_gate_report(
    complete_report: Mapping[str, Any],
    *,
    require_active_free_boundary: bool = True,
    require_scalar_controls_stackable: bool = True,
    require_array_controls_stackable: bool = True,
    json_safe: bool = False,
) -> dict[str, Any]:
    """Return the branch gate for promoting a fixed-trace replay derivative."""

    errors: list[str] = []
    branch = complete_report.get("branch_compatibility", {})
    same_branch = bool(branch.get("same_branch", False))
    if not same_branch:
        errors.append("branch_compatibility.same_branch is false")

    trace_diags = complete_report.get("trace_replay_diagnostics", {})
    expected_labels = ("base", "plus", "minus")
    if set(trace_diags) != set(expected_labels):
        errors.append("trace_replay_diagnostics must contain base, plus, and minus")

    for label in expected_labels:
        diag = trace_diags.get(label)
        fingerprint = branch.get(f"{label}_fingerprint")
        if not isinstance(diag, Mapping):
            errors.append(f"{label}: missing replay diagnostics")
            continue
        if not isinstance(fingerprint, Mapping):
            errors.append(f"{label}: missing branch fingerprint")
            continue
        if bool(diag.get("differentiates_adaptive_controller", True)):
            errors.append(f"{label}: diagnostics unexpectedly claim adaptive-controller differentiation")
        diag_fingerprint = diag.get("branch_fingerprint", {})
        if int(diag.get("n_steps", -1)) != int(fingerprint.get("n_steps", -2)):
            errors.append(f"{label}: n_steps mismatch")
        if int(diag_fingerprint.get("n_steps", -1)) != int(fingerprint.get("n_steps", -2)):
            errors.append(f"{label}: fingerprint n_steps mismatch")
        if int(diag_fingerprint.get("n_freeb_steps", -1)) != int(fingerprint.get("n_freeb_steps", -2)):
            errors.append(f"{label}: fingerprint n_freeb_steps mismatch")
        try:
            if not np.array_equal(
                np.asarray(diag_fingerprint.get("freeb_sizes")),
                np.asarray(fingerprint.get("freeb_sizes")),
            ):
                errors.append(f"{label}: freeb_sizes mismatch")
        except Exception:
            errors.append(f"{label}: freeb_sizes comparison failed")

        masks = diag.get("masks", {})
        n_steps = int(fingerprint.get("n_steps", -1))
        for mask_key in ("active", "accepted", "rejected", "done", "has_active_freeb_replay"):
            mask = np.asarray(masks.get(mask_key, []), dtype=bool)
            if mask.shape != (n_steps,):
                errors.append(f"{label}: mask {mask_key!r} has shape {mask.shape}, expected {(n_steps,)}")
        if require_active_free_boundary:
            if int(fingerprint.get("n_freeb_steps", 0)) <= 0:
                errors.append(f"{label}: no active free-boundary replay steps in fingerprint")
            active_freeb = np.logical_and(
                np.asarray(masks.get("accepted", []), dtype=bool),
                np.asarray(masks.get("has_active_freeb_replay", []), dtype=bool),
            )
            if not bool(np.any(active_freeb)):
                errors.append(f"{label}: no accepted active free-boundary replay slots")

        replay_diag = diag.get("replay_diagnostics", {})
        if require_scalar_controls_stackable and not bool(replay_diag.get("scalar_controls_stackable", False)):
            errors.append(f"{label}: scalar controls are not stackable")
        if require_array_controls_stackable and not bool(replay_diag.get("array_controls_stackable", False)):
            errors.append(f"{label}: array controls are not stackable")
        if int(replay_diag.get("preconditioner_policy_n_segments", 0)) < 1:
            errors.append(f"{label}: no preconditioner policy segments")

    gate = {
        "contract": "same-branch accepted-trace replay gate",
        "passed": len(errors) == 0,
        "differentiates_adaptive_controller": False,
        "same_branch": same_branch,
        "errors": tuple(errors),
    }
    if json_safe:
        return _json_safe_fingerprint_value(gate)
    return gate


def direct_coil_same_branch_physical_scalar_gate_report(
    complete_report: Mapping[str, Any],
    scalars_report: Mapping[str, Any],
    *,
    scalar_keys: tuple[str, ...] | list[str] | None = None,
    json_safe: bool = False,
) -> dict[str, Any]:
    """Return a same-branch physical-scalar promotion gate."""

    replay_gate = direct_coil_same_branch_replay_gate_report(complete_report)
    scalar_reports = scalars_report.get("scalar_reports", {})
    if scalar_keys is None:
        scalar_keys = tuple(str(key) for key in scalars_report.get("scalar_keys", tuple(scalar_reports)))
    else:
        scalar_keys = tuple(str(key) for key in scalar_keys)

    errors: list[str] = []
    if not bool(replay_gate.get("passed", False)):
        errors.append("same-branch replay gate failed")
    if bool(replay_gate.get("differentiates_adaptive_controller", True)):
        errors.append("replay gate unexpectedly claims adaptive-controller differentiation")
    if not bool(scalars_report.get("passed", False)):
        errors.append("branch-local scalar report did not pass")
    if bool(scalars_report.get("differentiates_adaptive_controller", False)):
        errors.append("branch-local scalar report unexpectedly claims adaptive-controller differentiation")
    if bool(scalars_report.get("differentiates_run_free_boundary", False)):
        errors.append("branch-local scalar report unexpectedly claims run_free_boundary differentiation")
    if "differentiates_fixed_accepted_branch" in scalars_report and not bool(
        scalars_report.get("differentiates_fixed_accepted_branch", False)
    ):
        errors.append("branch-local scalar report does not differentiate the fixed accepted branch")
    if not bool(scalars_report.get("same_branch", False)):
        errors.append("scalar report is not same-branch")

    objective_values = complete_report.get("objective_values", {})
    branch = complete_report.get("branch_compatibility", {})
    replay_branch_metadata = scalars_report.get("replay_branch_metadata", {})
    controller_slot_summary = (
        direct_coil_accepted_trace_controller_slot_summary(replay_branch_metadata)
        if isinstance(replay_branch_metadata, Mapping)
        else {}
    )
    controller_slot_fingerprint = (
        direct_coil_accepted_trace_controller_slot_fingerprint(replay_branch_metadata)
        if isinstance(replay_branch_metadata, Mapping)
        else {}
    )
    same_accepted_trace_branch = bool(branch.get("same_accepted_trace_branch", branch.get("same_branch", False)))
    same_residual_branch = bool(branch.get("same_residual_branch", branch.get("same_branch", False)))
    if not same_accepted_trace_branch:
        errors.append("accepted-trace branch fingerprint changed")
    if not same_residual_branch:
        errors.append("residual-controller branch fingerprint changed")

    scalar_summaries: dict[str, dict[str, float | bool]] = {}
    for key in scalar_keys:
        scalar_report = scalar_reports.get(key)
        if not isinstance(scalar_report, Mapping):
            errors.append(f"{key}: missing scalar report")
            continue
        if key not in objective_values:
            errors.append(f"{key}: missing complete-solve objective values")
            continue
        if not bool(scalar_report.get("passed", False)):
            errors.append(f"{key}: scalar AD-vs-FD report failed")
        if not bool(scalar_report.get("same_branch", False)):
            errors.append(f"{key}: scalar report is not same-branch")
        complete_fd = float(objective_values[key]["central_fd_directional"])
        exact = float(np.asarray(scalar_report.get("exact_directional"), dtype=float))
        base_abs_delta = float(scalar_report.get("base_abs_delta", np.nan))
        if not np.isfinite(complete_fd):
            errors.append(f"{key}: non-finite complete-solve FD slope")
        if not np.isfinite(exact):
            errors.append(f"{key}: non-finite custom-VJP slope")
        scalar_summaries[key] = {
            "passed": bool(scalar_report.get("passed", False)),
            "complete_fd_directional": complete_fd,
            "exact_directional": exact,
            "abs_error": float(scalar_report.get("abs_error", np.nan)),
            "rel_error": float(scalar_report.get("rel_error", np.nan)),
            "base_abs_delta": base_abs_delta,
        }

    result = {
        "contract": "same-branch complete-solve physical-scalar AD-vs-FD gate",
        "passed": len(errors) == 0,
        "same_branch": bool(branch.get("same_branch", False)),
        "same_accepted_trace_branch": same_accepted_trace_branch,
        "same_residual_branch": same_residual_branch,
        "differentiates_adaptive_controller": False,
        "scalar_keys": scalar_keys,
        "controller_slot_summary": controller_slot_summary,
        "controller_slot_fingerprint": controller_slot_fingerprint,
        "replay_gate": replay_gate,
        "errors": tuple(errors),
        "scalars": scalar_summaries,
    }
    if json_safe:
        return _json_safe_fingerprint_value(result)
    return result


def direct_coil_adaptive_full_loop_same_branch_gate_report(
    complete_report: Mapping[str, Any],
    scalars_report: Mapping[str, Any],
    *,
    scalar_keys: tuple[str, ...] | list[str] | None = None,
    require_stacked_step_controls: bool = True,
    require_complete_loop_rejected_controller_slot: bool = False,
    require_fixed_rejected_controller_slot: bool = False,
    require_status_derived_rejected_controller_slot: bool = False,
    json_safe: bool = False,
) -> dict[str, Any]:
    """Report whether complete-loop FD is compatible with stacked replay AD."""

    physical_gate = direct_coil_same_branch_physical_scalar_gate_report(
        complete_report,
        scalars_report,
        scalar_keys=scalar_keys,
        json_safe=False,
    )
    branch = complete_report.get("branch_compatibility", {})
    branch_fingerprints = {
        "base": branch.get("base_fingerprint", {}),
        "plus": branch.get("plus_fingerprint", {}),
        "minus": branch.get("minus_fingerprint", {}),
    }
    residual_branch_fingerprints = {
        "base": branch.get("base_residual_fingerprint", {}),
        "plus": branch.get("plus_residual_fingerprint", {}),
        "minus": branch.get("minus_residual_fingerprint", {}),
    }
    same_full_loop_branch_fingerprint = bool(branch.get("same_branch", False)) and all(
        isinstance(branch_fingerprints[label], Mapping) and bool(branch_fingerprints[label])
        for label in ("base", "plus", "minus")
    )
    same_residual_branch_fingerprint = bool(branch.get("same_residual_branch", False))
    errors = [f"physical scalar gate: {error}" for error in physical_gate.get("errors", ())]
    if not same_full_loop_branch_fingerprint:
        errors.append("complete-loop branch fingerprints are missing or changed")
    replay_option_flags = scalars_report.get("replay_option_flags", {})
    used_stacked_step_controls = bool(replay_option_flags.get("use_stacked_step_controls", False))
    if bool(require_stacked_step_controls) and not used_stacked_step_controls:
        errors.append("stacked step-control replay was not used")
    replay_branch_metadata = scalars_report.get("replay_branch_metadata", {})
    controller_slot_summary = (
        direct_coil_accepted_trace_controller_slot_summary(replay_branch_metadata)
        if isinstance(replay_branch_metadata, Mapping)
        else {}
    )
    controller_slot_fingerprint = (
        direct_coil_accepted_trace_controller_slot_fingerprint(replay_branch_metadata)
        if isinstance(replay_branch_metadata, Mapping)
        else {}
    )
    fixed_rejected_controller_slots = int(controller_slot_summary.get("rejected_slots", 0))
    fixed_rejected_controller_slot_present = fixed_rejected_controller_slots > 0
    status_masks = replay_branch_metadata.get("status_masks", {}) if isinstance(replay_branch_metadata, Mapping) else {}
    status_acceptance_source = (
        replay_branch_metadata.get("status_acceptance_source")
        if isinstance(replay_branch_metadata, Mapping)
        else None
    )
    status_derived_rejected_controller_slot_present = bool(
        status_acceptance_source == "trace_step_status"
        and isinstance(status_masks, Mapping)
        and np.any(np.logical_not(np.asarray(status_masks.get("accept_mask", ()), dtype=bool)))
    )
    if bool(require_fixed_rejected_controller_slot):
        if not fixed_rejected_controller_slot_present:
            errors.append("fixed rejected controller slot was not replayed")
        if bool(replay_option_flags.get("use_accepted_only_fast_path", True)):
            errors.append("accepted-only fast path was used for a rejected-slot replay gate")
    if bool(require_status_derived_rejected_controller_slot):
        if not status_derived_rejected_controller_slot_present:
            errors.append("rejected controller slot was not derived from trace step_status")

    complete_loop_rejected_slot_present = all(
        _fingerprint_has_rejected_controller_slot(branch_fingerprints[label])
        for label in ("base", "plus", "minus")
    )
    if bool(require_complete_loop_rejected_controller_slot) and not complete_loop_rejected_slot_present:
        errors.append("complete-loop branch fingerprints do not contain a native rejected/restart controller slot")

    labels = ("base", "plus", "minus")
    step_policy_signatures: dict[str, tuple[tuple[int, int, int], ...]] = {}
    step_policy_summaries: dict[str, dict[str, Any]] = {}
    for label in labels:
        payload = complete_report.get(label)
        if not isinstance(payload, Mapping):
            errors.append(f"{label}: missing complete-solve payload")
            step_policy_signatures[label] = ()
            step_policy_summaries[label] = {"n_segments": 0, "segments": ()}
            continue
        signature = accepted_step_policy_layout_for_complete_payload(payload)
        if not signature:
            errors.append(f"{label}: no accepted step-policy segments")
        step_policy_signatures[label] = signature
        step_policy_summaries[label] = accepted_step_policy_summary_for_complete_payload(payload)

    same_stacked_step_policy_branch = (
        bool(step_policy_signatures.get("base"))
        and step_policy_signatures.get("base") == step_policy_signatures.get("plus")
        and step_policy_signatures.get("base") == step_policy_signatures.get("minus")
    )
    if not same_stacked_step_policy_branch:
        errors.append("stacked step-policy branch changed")

    result = {
        "contract": "same-branch adaptive full-loop seam report",
        "passed": len(errors) == 0,
        "ad_vs_fd_gate": "complete-loop central FD vs branch-local stacked replay custom VJP",
        "adaptive_loop_scope": "fingerprint-gated branch-local accepted/rejected replay slots",
        "unclaimed_adaptive_controller_reason": (
            "host adaptive branch selection remains outside the custom VJP; "
            "same-branch finite differences validate the fixed accepted/rejected controller slots"
        ),
        "differentiates_adaptive_controller": False,
        "differentiates_run_free_boundary": False,
        "fingerprint_gated": True,
        "same_branch": bool(branch.get("same_branch", False)),
        "same_accepted_trace_branch": bool(branch.get("same_accepted_trace_branch", branch.get("same_branch", False))),
        "same_residual_branch": bool(branch.get("same_residual_branch", branch.get("same_branch", False))),
        "same_full_loop_branch_fingerprint": bool(same_full_loop_branch_fingerprint),
        "same_residual_branch_fingerprint": bool(same_residual_branch_fingerprint),
        "branch_fingerprints": branch_fingerprints,
        "residual_branch_fingerprints": residual_branch_fingerprints,
        "same_stacked_step_policy_branch": bool(same_stacked_step_policy_branch),
        "requires_stacked_step_controls": bool(require_stacked_step_controls),
        "used_stacked_step_controls": used_stacked_step_controls,
        "requires_complete_loop_rejected_controller_slot": bool(require_complete_loop_rejected_controller_slot),
        "complete_loop_rejected_controller_slot_present": bool(complete_loop_rejected_slot_present),
        "requires_fixed_rejected_controller_slot": bool(require_fixed_rejected_controller_slot),
        "requires_status_derived_rejected_controller_slot": bool(require_status_derived_rejected_controller_slot),
        "fixed_rejected_controller_slot_present": bool(fixed_rejected_controller_slot_present),
        "fixed_rejected_controller_slots": int(fixed_rejected_controller_slots),
        "status_derived_rejected_controller_slot_present": bool(status_derived_rejected_controller_slot_present),
        "status_acceptance_source": status_acceptance_source,
        "controller_slot_summary": controller_slot_summary,
        "controller_slot_fingerprint": controller_slot_fingerprint,
        "replay_option_flags": replay_option_flags,
        "replay_branch_metadata": replay_branch_metadata,
        "scalar_keys": physical_gate.get("scalar_keys", ()),
        "physical_scalar_gate": physical_gate,
        "step_policy_segments": step_policy_summaries,
        "errors": tuple(errors),
    }
    if json_safe:
        return _json_safe_fingerprint_value(result)
    return result


def direct_coil_branch_local_scalars_report_from_complete_fd(
    complete_report: Mapping[str, Any],
    branch_local_scalars: Mapping[str, Any],
    *,
    scalar_keys: tuple[str, ...] | list[str] | None = None,
    rtol: float | Mapping[str, float] = 5.0e-3,
    atol: float | Mapping[str, float] = 1.0e-8,
    base_value_atol: float | Mapping[str, float] = 2.0e-3,
    json_safe: bool = False,
) -> dict[str, Any]:
    """Normalize production-forward branch-local scalar/JVP evidence."""

    if scalar_keys is None:
        scalar_keys = tuple(str(key) for key in branch_local_scalars.get("scalar_keys", ()))
    else:
        scalar_keys = tuple(str(key) for key in scalar_keys)
    if not scalar_keys:
        raise ValueError("scalar_keys must contain at least one scalar")

    def _option_for(option: float | Mapping[str, float], key: str) -> float:
        if isinstance(option, Mapping):
            return float(option[key])
        return float(option)

    replay_gate = direct_coil_same_branch_replay_gate_report(complete_report)
    branch = complete_report.get("branch_compatibility", {})
    same_branch = bool(branch.get("same_branch", False))
    objective_values = complete_report.get("objective_values", {})
    values = branch_local_scalars.get("values", {})
    replay_values = branch_local_scalars.get("replay_value_map", {})
    directionals = branch_local_scalars.get("directional_derivatives", {})
    base_abs_deltas = branch_local_scalars.get("base_abs_delta", {})
    scalar_reports: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    passed_values: list[bool] = []

    if directionals is None:
        directionals = {}
        errors.append("branch-local report does not contain directional derivatives")
    if not bool(branch_local_scalars.get("uses_production_forward", False)):
        errors.append("branch-local report did not use production forward values")
    if bool(branch_local_scalars.get("differentiates_adaptive_controller", True)):
        errors.append("branch-local report unexpectedly claims adaptive-controller differentiation")
    if bool(branch_local_scalars.get("differentiates_run_free_boundary", True)):
        errors.append("branch-local report unexpectedly claims run_free_boundary differentiation")
    if not bool(branch_local_scalars.get("differentiates_fixed_accepted_branch", False)):
        errors.append("branch-local report does not differentiate the fixed accepted branch")

    for key in scalar_keys:
        if key not in objective_values:
            errors.append(f"{key}: missing complete-solve objective values")
            passed_values.append(False)
            continue
        if not isinstance(directionals, Mapping) or key not in directionals:
            errors.append(f"{key}: missing branch-local directional derivative")
            passed_values.append(False)
            continue
        if not isinstance(values, Mapping) or key not in values:
            errors.append(f"{key}: missing production scalar value")
            passed_values.append(False)
            continue

        complete_values = objective_values[key]
        complete_base = float(complete_values["base"])
        complete_fd = float(complete_values["central_fd_directional"])
        value = float(np.asarray(values[key], dtype=float))
        exact = float(np.asarray(directionals[key], dtype=float))
        if isinstance(base_abs_deltas, Mapping) and key in base_abs_deltas:
            base_abs_delta = float(base_abs_deltas[key])
        elif isinstance(replay_values, Mapping) and key in replay_values:
            base_abs_delta = abs(float(np.asarray(replay_values[key], dtype=float)) - value)
        else:
            base_abs_delta = abs(value - complete_base)
        abs_error = abs(exact - complete_fd)
        rel_error = abs_error / max(1.0, abs(complete_fd))
        key_passed = bool(
            replay_gate["passed"]
            and same_branch
            and np.isfinite(value)
            and np.isfinite(exact)
            and np.isfinite(complete_fd)
            and abs_error <= _option_for(atol, key) + _option_for(rtol, key) * abs(complete_fd)
            and base_abs_delta <= _option_for(base_value_atol, key)
        )
        passed_values.append(key_passed)
        scalar_reports[key] = {
            "scalar_key": key,
            "passed": key_passed,
            "same_branch": same_branch,
            "replay_gate": replay_gate,
            "value": values[key],
            "exact_directional": directionals[key],
            "frozen_trace_fd_directional": np.nan,
            "complete_fd_directional": complete_fd,
            "abs_error": abs_error,
            "rel_error": rel_error,
            "base_value": value,
            "complete_base_value": complete_base,
            "base_abs_delta": base_abs_delta,
            "complete_values": complete_values,
        }

    result = {
        "contract": "production-forward branch-local vector/JVP physical-scalar report",
        "scalar_keys": scalar_keys,
        "passed": bool(passed_values and all(passed_values) and not errors),
        "same_branch": same_branch,
        "uses_production_forward": bool(branch_local_scalars.get("uses_production_forward", False)),
        "differentiates_adaptive_controller": False,
        "differentiates_run_free_boundary": False,
        "differentiates_fixed_accepted_branch": bool(
            branch_local_scalars.get("differentiates_fixed_accepted_branch", False)
        ),
        "derivative_mode": branch_local_scalars.get("derivative_mode"),
        "replay_ad_mode": branch_local_scalars.get("replay_ad_mode"),
        "replay_gate": replay_gate,
        "replay_option_flags": dict(branch_local_scalars.get("replay_option_flags", {})),
        "replay_branch_metadata": branch_local_scalars.get("replay_branch_metadata", {}),
        "controller_slot_summary": branch_local_scalars.get("controller_slot_summary", {}),
        "values": values,
        "exact_directionals": directionals,
        "scalar_reports": scalar_reports,
        "errors": tuple(errors),
    }
    if json_safe:
        return _json_safe_fingerprint_value(result)
    return result
