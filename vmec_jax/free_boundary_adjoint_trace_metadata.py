"""Small metadata helpers for free-boundary accepted-trace reports."""

from __future__ import annotations

from typing import Any

import numpy as np


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


_unique_shape_list = unique_shape_list
_compact_segment_summaries = compact_segment_summaries
_json_safe_fingerprint_value = json_safe_fingerprint_value

