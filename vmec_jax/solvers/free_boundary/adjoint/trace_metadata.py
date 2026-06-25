"""Small metadata helpers for free-boundary accepted-trace reports."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

__all__ = [
    "compact_segment_summaries",
    "direct_coil_accepted_trace_controller_slot_fingerprint",
    "direct_coil_accepted_trace_controller_slot_summary",
    "fingerprint_has_rejected_controller_slot",
    "json_safe_fingerprint_value",
    "unique_shape_list",
]


def unique_shape_list(shapes: list[tuple[int, ...]]) -> list[list[int]]:
    """Return unique shapes in first-seen order using JSON-friendly lists."""

    seen: set[tuple[int, ...]] = set()
    unique: list[list[int]] = []
    for shape in shapes:
        if shape in seen:
            continue
        seen.add(shape)
        unique.append([int(value) for value in shape])
    return unique


def compact_segment_summaries(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop exact static signatures from replay-graph timing metadata."""

    return [{key: value for key, value in summary.items() if key != "signature_repr"} for summary in summaries]


def json_safe_fingerprint_value(value: Any) -> Any:
    """Convert accepted-trace fingerprint diagnostics to strict JSON values."""

    if isinstance(value, np.ndarray):
        return json_safe_fingerprint_value(value.tolist())
    if isinstance(value, np.generic):
        return json_safe_fingerprint_value(value.item())
    if isinstance(value, dict):
        return {str(key): json_safe_fingerprint_value(val) for key, val in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe_fingerprint_value(item) for item in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return json_safe_fingerprint_value(value.tolist())
        except Exception:
            pass
    return value


def direct_coil_accepted_trace_controller_slot_summary(metadata: Mapping[str, Any]) -> dict[str, int | bool]:
    """Return compact accepted/rejected/done slot counts from branch metadata.

    Promotion reports should not force callers to inspect nested JAX masks to
    answer basic questions like whether a fixed rejected-controller slot was
    replayed.  This helper accepts both raw and JSON-safe branch metadata and
    returns plain Python scalars suitable for CI artifacts and optimization
    reports.
    """

    masks = metadata.get("masks", {}) if isinstance(metadata, Mapping) else {}

    def _mask(name: str, fallback: str | None = None) -> np.ndarray:
        value = None
        if isinstance(masks, Mapping):
            value = masks.get(name)
        if value is None and fallback is not None and isinstance(metadata, Mapping):
            value = metadata.get(fallback)
        if value is None:
            return np.asarray([], dtype=bool)
        return np.asarray(value, dtype=bool).reshape(-1)

    accepted = _mask("accepted", "accepted_mask")
    rejected = _mask("rejected", "rejected_mask")
    done = _mask("done", "done_mask")
    active = _mask("active")
    active_freeb = _mask("has_active_freeb_replay", "has_active_freeb_replay")
    accepted_freeb = _mask("active_free_boundary", "active_free_boundary_mask")
    if accepted_freeb.size == 0 and accepted.size and active_freeb.size == accepted.size:
        accepted_freeb = np.logical_and(accepted, active_freeb)

    n_steps = int(metadata.get("n_steps", max(accepted.size, rejected.size, done.size, active.size)))
    n_freeb = int(metadata.get("n_free_boundary_replay_steps", np.count_nonzero(accepted_freeb)))
    rejected_slots = int(np.count_nonzero(rejected))
    return {
        "n_steps": n_steps,
        "active_slots": int(np.count_nonzero(active)) if active.size else n_steps,
        "accepted_slots": int(np.count_nonzero(accepted)),
        "rejected_slots": rejected_slots,
        "done_markers": int(np.count_nonzero(done)),
        "active_free_boundary_slots": int(np.count_nonzero(active_freeb)),
        "accepted_free_boundary_slots": int(n_freeb),
        "fixed_rejected_controller_slot_present": bool(rejected_slots > 0),
    }


def direct_coil_accepted_trace_controller_slot_fingerprint(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Return a stable JSON-safe accepted/rejected controller-slot fingerprint.

    The summary above answers "how many slots were replayed"; this fingerprint
    records the masks and status source that make a same-branch claim
    auditable without digging through the full trace payload.
    """

    masks = metadata.get("masks", {}) if isinstance(metadata, Mapping) else {}
    status_masks = metadata.get("status_masks", {}) if isinstance(metadata, Mapping) else {}

    def _mask(name: str, fallback: str | None = None) -> np.ndarray:
        value = None
        if isinstance(masks, Mapping):
            value = masks.get(name)
        if value is None and fallback is not None and isinstance(metadata, Mapping):
            value = metadata.get(fallback)
        if value is None and name == "accepted" and isinstance(status_masks, Mapping):
            value = status_masks.get("accept_mask")
        if value is None:
            return np.asarray([], dtype=bool)
        return np.asarray(value, dtype=bool).reshape(-1)

    accepted = _mask("accepted", "accepted_mask")
    rejected = _mask("rejected", "rejected_mask")
    done = _mask("done", "done_mask")
    active_freeb = _mask("has_active_freeb_replay", "has_active_freeb_replay")
    accepted_freeb = _mask("active_free_boundary", "active_free_boundary_mask")
    if accepted_freeb.size == 0 and accepted.size and active_freeb.size == accepted.size:
        accepted_freeb = np.logical_and(accepted, active_freeb)
    if rejected.size == 0 and accepted.size:
        rejected = np.logical_not(accepted)

    step_status: list[str] = []
    status_acceptance_source = None
    if isinstance(status_masks, Mapping):
        step_status = [str(item) for item in status_masks.get("step_status", ())]
        status_acceptance_source = status_masks.get("status_acceptance_source")
    if isinstance(metadata, Mapping):
        status_acceptance_source = metadata.get("status_acceptance_source", status_acceptance_source)
    n_steps = int(metadata.get("n_steps", max(accepted.size, rejected.size, done.size))) if isinstance(
        metadata, Mapping
    ) else max(accepted.size, rejected.size, done.size)
    return json_safe_fingerprint_value(
        {
            "n_steps": n_steps,
            "accepted_mask": accepted,
            "rejected_mask": rejected,
            "done_mask": done,
            "active_free_boundary_mask": accepted_freeb,
            "has_active_freeb_replay": active_freeb,
            "status_acceptance_source": status_acceptance_source,
            "step_status": step_status,
            "summary": direct_coil_accepted_trace_controller_slot_summary(metadata),
        }
    )


def fingerprint_has_rejected_controller_slot(fingerprint: Mapping[str, Any]) -> bool:
    """Return whether a complete-loop fingerprint includes a rejected slot."""

    if not isinstance(fingerprint, Mapping):
        return False
    accept_mask = np.asarray(fingerprint.get("accept_mask", ()), dtype=int)
    if accept_mask.size and np.any(accept_mask == 0):
        return True
    step_status = tuple(str(status) for status in fingerprint.get("step_status", ()))
    return any(status.startswith("restart_") or status == "rejected" for status in step_status)


_unique_shape_list = unique_shape_list
_compact_segment_summaries = compact_segment_summaries
_json_safe_fingerprint_value = json_safe_fingerprint_value
_fingerprint_has_rejected_controller_slot = fingerprint_has_rejected_controller_slot
_direct_coil_accepted_trace_controller_slot_fingerprint = direct_coil_accepted_trace_controller_slot_fingerprint
