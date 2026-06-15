"""Accepted-trace branch fingerprints for free-boundary adjoint gates."""

from __future__ import annotations

from typing import Any

import numpy as np

from vmec_jax._compat import tree_util

from vmec_jax.state import pack_state

from .trace_controls import direct_coil_accepted_trace_status_masks
from .trace_metadata import json_safe_fingerprint_value

__all__ = [
    "direct_coil_accepted_trace_fingerprint",
    "direct_coil_accepted_trace_fingerprint_delta",
    "direct_coil_accepted_trace_fingerprint_delta_summary",
    "trace_array_size",
    "trace_bool",
    "trace_pack_size",
    "trace_pytree_shape_signature",
    "trace_scalar",
]


def trace_scalar(trace: dict[str, Any], key: str, *, default: float = np.nan) -> float:
    """Return one scalar branch-control value from an accepted trace."""

    value = trace.get(key, default)
    if value is None:
        return float(default)
    arr = np.asarray(value)
    if arr.size == 0:
        return float(default)
    return float(arr.reshape(-1)[0])


def trace_bool(trace: dict[str, Any], key: str) -> int:
    """Return one boolean branch-control value from an accepted trace."""

    value = trace.get(key, False)
    if value is None:
        return 0
    arr = np.asarray(value)
    if arr.size == 0:
        return 0
    return int(bool(arr.reshape(-1)[0]))


def trace_pack_size(value: Any) -> int:
    """Return packed VMEC state size, falling back to raw array size."""

    if value is None:
        return 0
    try:
        return int(np.asarray(pack_state(value)).size)
    except Exception:
        return int(np.asarray(value).size)


def _pack_or_array(value: Any) -> np.ndarray:
    """Pack a VMEC state or coerce an array-like trace state to 1D float."""

    try:
        return np.asarray(pack_state(value), dtype=float)
    except Exception:
        return np.asarray(value, dtype=float).reshape(-1)


def trace_array_size(value: Any) -> int:
    """Return raw array size for optional trace payloads."""

    if value is None:
        return 0
    return int(np.asarray(value).size)


def trace_pytree_shape_signature(value: Any) -> tuple[tuple[int, ...], ...]:
    """Return leaf shape signatures for a trace pytree payload."""

    if value is None:
        return ()
    try:
        leaves = tree_util.tree_leaves(value)
    except Exception:
        leaves = [value]
    return tuple(tuple(np.asarray(leaf).shape) for leaf in leaves)


def _state_reset_flags(trace_seq: list[dict[str, Any]]) -> np.ndarray:
    """Return flags for discontinuous accepted-trace state chaining."""

    reset_flags: list[int] = []
    for prev_trace, trace in zip(trace_seq[:-1], trace_seq[1:], strict=False):
        try:
            prev_packed = _pack_or_array(prev_trace.get("state_post"))
            next_packed = _pack_or_array(trace.get("state_pre"))
            reset_flags.append(
                int(
                    prev_packed.shape != next_packed.shape
                    or (not np.allclose(prev_packed, next_packed, rtol=1.0e-13, atol=1.0e-13))
                )
            )
        except Exception:
            reset_flags.append(0)
    return np.asarray(reset_flags, dtype=int)


def direct_coil_accepted_trace_fingerprint(
    traces: Any,
    *,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Return a branch-control fingerprint for accepted free-boundary traces.

    The fixed-trace direct-coil adjoint differentiates a frozen local model:
    accepted controller choices, time-step scalars, limiter policy, and NESTOR
    trace structure are fixed while coil fields are resampled.  This
    fingerprint captures those *discrete/control* choices so a complete-solve
    finite-difference check can reject perturbations that moved onto a
    different adaptive branch before comparing derivatives.

    Differentiable values that should vary with coil parameters, such as the
    actual ``freeb_bsqvac_half`` entries, are intentionally not included except
    for presence/size metadata.
    """

    trace_seq = list(traces)
    if max_steps is not None:
        trace_seq = trace_seq[: int(max_steps)]

    scalar_keys = (
        "dt_eff",
        "b1",
        "fac",
        "force_scale",
        "max_update_rms_pre",
        "limit_update_rms",
    )
    bool_keys = (
        "flip_sign",
        "divide_by_scalxc_for_update",
        "preconditioner_use_precomputed_tridi",
        "preconditioner_use_lax_tridi",
    )
    scalars = {
        key: np.asarray([trace_scalar(trace, key) for trace in trace_seq], dtype=float)
        for key in scalar_keys
    }
    flags = {
        key: np.asarray([trace_bool(trace, key) for trace in trace_seq], dtype=int)
        for key in bool_keys
    }
    freeb_sizes = np.asarray(
        [trace_array_size(trace.get("freeb_bsqvac_half")) for trace in trace_seq],
        dtype=int,
    )
    nestor_sizes = np.asarray(
        [
            len(trace.get("freeb_nestor_trace", {}) or {})
            if isinstance(trace.get("freeb_nestor_trace", {}), dict)
            else 0
            for trace in trace_seq
        ],
        dtype=int,
    )
    state_pre_sizes = np.asarray(
        [trace_pack_size(trace.get("state_pre")) for trace in trace_seq],
        dtype=int,
    )
    state_post_sizes = np.asarray(
        [trace_pack_size(trace.get("state_post")) for trace in trace_seq],
        dtype=int,
    )
    precond_jmax = np.asarray([int(trace.get("precond_jmax", -1)) for trace in trace_seq], dtype=int)
    precond_mats_shapes = tuple(trace_pytree_shape_signature(trace.get("precond_mats")) for trace in trace_seq)
    lam_prec_shapes = tuple(tuple(np.asarray(trace.get("lam_prec", [])).shape) for trace in trace_seq)
    w_mode_shapes = tuple(tuple(np.asarray(trace.get("w_mode_mn", [])).shape) for trace in trace_seq)
    if trace_seq:
        status_masks = direct_coil_accepted_trace_status_masks(trace_seq)
        step_status = tuple(status_masks["step_status"])
        accept_mask = np.asarray(status_masks["accept_mask"], dtype=int)
        done_mask = np.asarray(status_masks["done_mask"], dtype=int)
    else:
        step_status = ()
        accept_mask = np.asarray((), dtype=int)
        done_mask = np.asarray((), dtype=int)
    return {
        "n_steps": int(len(trace_seq)),
        "n_freeb_steps": int(np.count_nonzero(freeb_sizes)),
        "scalars": scalars,
        "flags": flags,
        "freeb_sizes": freeb_sizes,
        "nestor_trace_key_counts": nestor_sizes,
        "state_pre_sizes": state_pre_sizes,
        "state_post_sizes": state_post_sizes,
        "precond_jmax": precond_jmax,
        "precond_mats_shapes": precond_mats_shapes,
        "lam_prec_shapes": lam_prec_shapes,
        "w_mode_mn_shapes": w_mode_shapes,
        "step_status": step_status,
        "accept_mask": accept_mask,
        "done_mask": done_mask,
        "state_reset_flags": _state_reset_flags(trace_seq),
    }


def direct_coil_accepted_trace_fingerprint_delta(
    reference: Any,
    candidate: Any,
    *,
    rtol: float = 1.0e-10,
    atol: float = 1.0e-12,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Compare two accepted-trace fingerprints."""

    ref = direct_coil_accepted_trace_fingerprint(reference, max_steps=max_steps)
    cand = direct_coil_accepted_trace_fingerprint(candidate, max_steps=max_steps)
    changed: list[str] = []
    max_abs = 0.0
    max_rel = 0.0

    for key in ("n_steps", "n_freeb_steps"):
        if int(ref[key]) != int(cand[key]):
            changed.append(key)

    for group in ("flags",):
        for key, ref_values in ref[group].items():
            cand_values = cand[group].get(key, np.asarray([], dtype=ref_values.dtype))
            if ref_values.shape != cand_values.shape or not np.array_equal(ref_values, cand_values):
                changed.append(f"{group}.{key}")

    for key in (
        "freeb_sizes",
        "nestor_trace_key_counts",
        "state_pre_sizes",
        "state_post_sizes",
        "precond_jmax",
        "accept_mask",
        "done_mask",
        "state_reset_flags",
    ):
        ref_values = np.asarray(ref[key])
        cand_values = np.asarray(cand[key])
        if ref_values.shape != cand_values.shape or not np.array_equal(ref_values, cand_values):
            changed.append(key)

    for key in ("precond_mats_shapes", "lam_prec_shapes", "w_mode_mn_shapes", "step_status"):
        if ref[key] != cand[key]:
            changed.append(key)

    for key, ref_values in ref["scalars"].items():
        cand_values = cand["scalars"].get(key, np.asarray([], dtype=float))
        if ref_values.shape != cand_values.shape:
            changed.append(f"scalars.{key}")
            continue
        abs_delta = np.abs(cand_values - ref_values)
        finite = np.isfinite(abs_delta)
        if np.any(finite):
            max_abs = max(max_abs, float(np.max(abs_delta[finite])))
            denom = np.maximum(np.abs(ref_values[finite]), float(atol))
            max_rel = max(max_rel, float(np.max(abs_delta[finite] / denom)))
        if not np.allclose(cand_values, ref_values, rtol=float(rtol), atol=float(atol), equal_nan=True):
            changed.append(f"scalars.{key}")

    return {
        "compatible": len(changed) == 0,
        "changed_fields": tuple(changed),
        "max_abs_scalar_delta": float(max_abs),
        "max_rel_scalar_delta": float(max_rel),
        "reference": ref,
        "candidate": cand,
    }


def direct_coil_accepted_trace_fingerprint_delta_summary(
    reference: Any,
    candidate: Any,
    *,
    rtol: float = 1.0e-10,
    atol: float = 1.0e-12,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Return a strict-JSON-safe accepted-trace fingerprint delta summary."""

    delta = direct_coil_accepted_trace_fingerprint_delta(
        reference,
        candidate,
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
    )
    return json_safe_fingerprint_value(delta)


_trace_scalar = trace_scalar
_trace_bool = trace_bool
_trace_pack_size = trace_pack_size
_trace_array_size = trace_array_size
_trace_pytree_shape_signature = trace_pytree_shape_signature
