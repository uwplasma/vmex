"""Pure helpers for performance hotspot instrumentation tests.

These helpers keep cache-key and timing-bucket behavior testable without
building VMEC tapes or launching accelerator work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


SCAN_CACHE_KEY_FIELDS: tuple[str, ...] = (
    "schema",
    "static_key",
    "wout_key",
    "edge_signature_key",
    "tomnsps_policy_key",
    "max_iter_tail",
    "preflight_iters",
    "iter_offset0",
    "step_size",
    "initial_flip_sign",
    "lambda_update_scale",
    "ftol",
    "nstep_screen",
    "use_restart_triggers",
    "vmecpp_restart",
    "scan_use_precomputed",
    "scan_use_lax_tridi",
    "scan_use_restart_payload",
    "stage_prev_fsq",
    "stage_transition_factor",
    "stage_transition_scale",
    "jit_forces_scan",
    "state_only_scan",
    "scan_light",
    "scan_minimal",
    "scan_fallback_iters",
    "scan_fallback_accept_frac",
    "scan_fallback_fsq_factor",
    "scan_fallback_badjac_limit",
    "scan_fallback_fsq_abs",
)


@dataclass(frozen=True)
class CacheKeyDelta:
    """One labeled difference between two tuple cache keys."""

    index: int
    field: str
    before: Any
    after: Any


def exact_parameter_cache_key(params: Any) -> bytes:
    """Return the exact-optimizer parameter cache key for a parameter vector."""

    return np.asarray(params, dtype=np.float64).reshape(-1).tobytes()


def exact_parameter_cache_key_fingerprint(params: Any) -> dict[str, Any]:
    """Return deterministic, non-secret metadata for an exact parameter key."""

    arr = np.asarray(params, dtype=np.float64).reshape(-1)
    return {
        "dtype": str(arr.dtype),
        "n_params": int(arr.size),
        "byte_length": int(arr.nbytes),
        "cache_key": arr.tobytes(),
    }


def explain_scan_cache_key_delta(
    before_key: tuple[Any, ...],
    after_key: tuple[Any, ...],
    *,
    field_names: tuple[str, ...] = SCAN_CACHE_KEY_FIELDS,
) -> tuple[CacheKeyDelta, ...]:
    """Label the fields that changed between two scan cache-key tuples."""

    n_compare = max(len(before_key), len(after_key))
    deltas: list[CacheKeyDelta] = []
    missing = object()
    for index in range(n_compare):
        before = before_key[index] if index < len(before_key) else missing
        after = after_key[index] if index < len(after_key) else missing
        if before == after:
            continue
        field = field_names[index] if index < len(field_names) else f"field_{index}"
        deltas.append(
            CacheKeyDelta(
                index=index,
                field=field,
                before="<missing>" if before is missing else before,
                after="<missing>" if after is missing else after,
            )
        )
    return tuple(deltas)


def replay_timing_breakdown(
    profile: Mapping[str, Mapping[str, Any]],
    *,
    prefix: str,
) -> dict[str, float | int | None]:
    """Summarize exact replay timing buckets for ``<prefix>_tape_replay``."""

    total_name = f"{prefix}_tape_replay"
    dispatch_name = f"{total_name}_dispatch"
    ready_name = f"{total_name}_ready"
    total = _profile_wall_time(profile, total_name)
    dispatch = _profile_wall_time(profile, dispatch_name)
    ready = _profile_wall_time(profile, ready_name)
    split_total = dispatch + ready
    if total == 0.0 and split_total > 0.0:
        total = split_total
    return {
        "total_s": total,
        "dispatch_s": dispatch,
        "ready_s": ready,
        "split_total_s": split_total,
        "count": _profile_count(profile, total_name, dispatch_name, ready_name),
    }


def accumulate_scan_device_ready_timing(
    stats: dict[str, float],
    *,
    start: float | None,
    dispatch_done: float,
    ready_done: float,
) -> bool:
    """Accumulate scan dispatch/ready timing, returning whether data was recorded."""

    if start is None:
        return False
    start_f = float(start)
    dispatch_f = float(dispatch_done)
    ready_f = float(ready_done)
    stats["scan_device_dispatch_s"] = float(stats.get("scan_device_dispatch_s", 0.0)) + dispatch_f - start_f
    stats["scan_device_ready_s"] = float(stats.get("scan_device_ready_s", 0.0)) + ready_f - dispatch_f
    stats["scan_device_run_s"] = float(stats.get("scan_device_run_s", 0.0)) + ready_f - start_f
    return True


def _profile_wall_time(profile: Mapping[str, Mapping[str, Any]], name: str) -> float:
    rec = profile.get(name, {})
    try:
        return float(rec.get("wall_time_s", 0.0))
    except Exception:
        return 0.0


def _profile_count(profile: Mapping[str, Mapping[str, Any]], *names: str) -> int | None:
    counts: list[int] = []
    for name in names:
        rec = profile.get(name)
        if rec is None:
            continue
        try:
            counts.append(int(rec.get("count", 0)))
        except Exception:
            continue
    return max(counts) if counts else None


__all__ = [
    "CacheKeyDelta",
    "SCAN_CACHE_KEY_FIELDS",
    "accumulate_scan_device_ready_timing",
    "exact_parameter_cache_key",
    "exact_parameter_cache_key_fingerprint",
    "explain_scan_cache_key_delta",
    "replay_timing_breakdown",
]
