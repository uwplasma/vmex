"""Accepted-trace stacking and static-signature helpers."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from vmec_jax._compat import jnp, tree_util

from .trace_controls import direct_coil_accepted_trace_controller_controls_jax
from .trace_fingerprint import trace_pytree_shape_signature


def stack_trace_control_field(trace_seq: tuple[dict[str, Any], ...], key: str, *, dtype: Any | None = None) -> Any:
    if not trace_seq:
        raise ValueError("at least one accepted trace is required")
    arrays = []
    for index, trace in enumerate(trace_seq):
        if key not in trace:
            raise KeyError(f"accepted trace {index} is missing control field {key!r}")
        arrays.append(jnp.asarray(trace[key], dtype=dtype))
    shapes = {tuple(arr.shape) for arr in arrays}
    if len(shapes) != 1:
        raise ValueError(f"accepted trace control field {key!r} must have consistent shape")
    return jnp.stack(arrays, axis=0)


def stack_trace_pytree_field(trace_seq: tuple[dict[str, Any], ...], key: str) -> Any:
    if not trace_seq:
        raise ValueError("at least one accepted trace is required")
    values = []
    for index, trace in enumerate(trace_seq):
        if key not in trace:
            raise KeyError(f"accepted trace {index} is missing control field {key!r}")
        values.append(trace[key])
    treedef = tree_util.tree_structure(values[0])
    for index, value in enumerate(values[1:], start=1):
        if tree_util.tree_structure(value) != treedef:
            raise ValueError(f"accepted trace pytree field {key!r} has inconsistent structure at step {index}")

    def _stack_leaf(*leaves):
        arrays = [jnp.asarray(leaf) for leaf in leaves]
        shapes = {tuple(arr.shape) for arr in arrays}
        if len(shapes) != 1:
            raise ValueError(f"accepted trace pytree field {key!r} must have consistent leaf shapes")
        return jnp.stack(arrays, axis=0)

    return tree_util.tree_map(_stack_leaf, *values)


def stack_optional_trace_pytree_field(trace_seq: tuple[dict[str, Any], ...], key: str) -> Any | None:
    values = [trace.get(key) for trace in trace_seq]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        return None

    def _stack_leaf(*leaves):
        arrays = [jnp.asarray(leaf) for leaf in leaves]
        shapes = {tuple(arr.shape) for arr in arrays}
        if len(shapes) != 1:
            raise ValueError(f"accepted trace optional field {key!r} must have consistent leaf shapes")
        return jnp.stack(arrays, axis=0)

    return tree_util.tree_map(_stack_leaf, *values)


def trace_preconditioner_policy_value(trace: dict[str, Any], key: str) -> int:
    value = trace.get(key, None)
    if value is None:
        return -1
    arr = np.asarray(value)
    if arr.size == 0:
        return -1
    return 1 if bool(arr.reshape(-1)[0]) else 0


def trace_preconditioner_static_signature(trace: dict[str, Any]) -> tuple[Any, ...]:
    """Return the static preconditioner branch signature for one trace."""

    return (
        trace_preconditioner_policy_value(trace, "preconditioner_use_precomputed_tridi"),
        trace_preconditioner_policy_value(trace, "preconditioner_use_lax_tridi"),
        int(trace.get("precond_jmax", -1)),
        trace_pytree_shape_signature(trace.get("precond_mats")),
        tuple(np.asarray(trace.get("lam_prec", [])).shape),
        tuple(np.asarray(trace.get("w_mode_mn", [])).shape),
    )


def trace_static_value_shape_signature(value: Any) -> tuple[Any, ...]:
    """Return a compact signature for static trace payload structure/value."""

    if value is None:
        return ()
    try:
        leaves = tree_util.tree_leaves(value)
    except Exception:
        leaves = [value]
    signature = []
    for leaf in leaves:
        arr = np.asarray(leaf)
        if arr.dtype == object:
            digest = hashlib.sha256(repr(arr.tolist()).encode("utf-8")).hexdigest()
        else:
            digest = hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()
        signature.append((tuple(arr.shape), str(arr.dtype), digest))
    return tuple(signature)


def trace_optional_presence_signature(trace: dict[str, Any], keys: tuple[str, ...]) -> tuple[int, ...]:
    return tuple(0 if trace.get(key) is None else 1 for key in keys)


_ACCEPTED_TRACE_NESTOR_AXIS_KEYS = ("br_axis", "bp_axis", "bz_axis")


def stack_trace_nestor_axis_controls(trace_seq: tuple[dict[str, Any], ...]) -> dict[str, Any] | None:
    active_nestor = [
        trace.get("freeb_nestor_trace")
        if trace.get("freeb_bsqvac_half") is not None and isinstance(trace.get("freeb_nestor_trace"), dict)
        else None
        for trace in trace_seq
    ]
    if all(nestor_trace is None for nestor_trace in active_nestor):
        return None
    payload = {}
    for key in _ACCEPTED_TRACE_NESTOR_AXIS_KEYS:
        template = None
        for nestor_trace in active_nestor:
            if nestor_trace is not None:
                if key not in nestor_trace:
                    raise KeyError(f"active NESTOR trace is missing axis field {key!r}")
                template = jnp.asarray(nestor_trace[key])
                break
        if template is None:
            continue
        values = []
        for nestor_trace in active_nestor:
            if nestor_trace is None:
                values.append(jnp.zeros_like(template))
                continue
            value = jnp.asarray(nestor_trace[key])
            if tuple(value.shape) != tuple(template.shape):
                raise ValueError(f"NESTOR axis field {key!r} must have consistent shape for stacked replay")
            values.append(value)
        payload[key] = jnp.stack(values, axis=0)
    return payload


ACCEPTED_TRACE_NUMERIC_CONTROL_KEYS = (
    "dt_eff",
    "b1",
    "fac",
    "force_scale",
    "max_update_rms_pre",
    "lambda_update_scale",
)

ACCEPTED_TRACE_BOOL_CONTROL_KEYS = (
    "flip_sign",
    "limit_update_rms",
    "divide_by_scalxc_for_update",
    "preconditioner_use_precomputed_tridi",
    "preconditioner_use_lax_tridi",
)

ACCEPTED_TRACE_REQUIRED_ARRAY_CONTROL_KEYS = (
    "vRcc_before",
    "vRss_before",
    "vZsc_before",
    "vZcs_before",
    "vLsc_before",
    "vLcs_before",
)

ACCEPTED_TRACE_OPTIONAL_ARRAY_CONTROL_KEYS = (
    "vRsc_before",
    "vRcs_before",
    "vZcc_before",
    "vZss_before",
    "vLcc_before",
    "vLss_before",
)


def direct_coil_accepted_trace_scalar_controls_jax(traces: Any) -> dict[str, Any]:
    """Return stacked scalar/update controls consumed by accepted trace replay."""

    trace_seq = tuple(traces)
    if not trace_seq:
        raise ValueError("at least one accepted trace is required")
    payload: dict[str, Any] = {}
    for key in ACCEPTED_TRACE_NUMERIC_CONTROL_KEYS:
        payload[key] = stack_trace_control_field(trace_seq, key)
    for key in ACCEPTED_TRACE_BOOL_CONTROL_KEYS:
        payload[key] = stack_trace_control_field(trace_seq, key, dtype=bool)
    return payload


def direct_coil_accepted_trace_preconditioner_controls_jax(traces: Any) -> dict[str, Any]:
    """Return stacked preconditioner/mode payloads for accepted replay."""

    trace_seq = tuple(traces)
    if not trace_seq:
        raise ValueError("at least one accepted trace is required")
    return {
        "precond_mats": stack_trace_pytree_field(trace_seq, "precond_mats"),
        "lam_prec": stack_trace_control_field(trace_seq, "lam_prec"),
        "w_mode_mn": stack_trace_control_field(trace_seq, "w_mode_mn"),
    }


def direct_coil_accepted_trace_array_controls_jax(traces: Any) -> dict[str, Any]:
    """Return stacked array-valued update controls for accepted trace replay."""

    trace_seq = tuple(traces)
    if not trace_seq:
        raise ValueError("at least one accepted trace is required")
    payload: dict[str, Any] = {}
    for key in ACCEPTED_TRACE_REQUIRED_ARRAY_CONTROL_KEYS:
        payload[key] = stack_trace_control_field(trace_seq, key)
    for key in ACCEPTED_TRACE_OPTIONAL_ARRAY_CONTROL_KEYS:
        values = [trace.get(key) for trace in trace_seq]
        if all(value is None for value in values):
            continue
        if any(value is None for value in values):
            raise ValueError(f"accepted trace optional array field {key!r} must be present for every step or none")
        payload[key] = stack_trace_control_field(trace_seq, key)
    return payload


_ACCEPTED_TRACE_OPTIONAL_STEP_PYTREE_CONTROL_KEYS = (
    "force_state_pre",
    "freeb_pres_scale",
    "constraint_rcon0",
    "constraint_zcon0",
    "constraint_tcon0",
    "constraint_precond_diag",
    "constraint_tcon",
    "constraint_precond_active",
    "constraint_tcon_active",
)


def direct_coil_accepted_trace_step_controls_jax(
    traces: Any,
    *,
    include_state_pre: bool = True,
    include_force_state_pre: bool = True,
    include_nestor_axes: bool = True,
    include_constraints: bool = True,
) -> dict[str, Any]:
    """Return stacked state/constraint controls for direct accepted replay."""

    trace_seq = tuple(traces)
    if not trace_seq:
        raise ValueError("at least one accepted trace is required")
    payload: dict[str, Any] = {}
    if bool(include_state_pre):
        payload["state_pre"] = stack_trace_pytree_field(trace_seq, "state_pre")
    if bool(include_force_state_pre):
        force_state_pre = stack_optional_trace_pytree_field(trace_seq, "force_state_pre")
        if force_state_pre is not None:
            payload["force_state_pre"] = force_state_pre
    if bool(include_nestor_axes):
        nestor_axes = stack_trace_nestor_axis_controls(trace_seq)
        if nestor_axes is not None:
            payload["freeb_nestor_axes"] = nestor_axes
    freeb_pres_scale = stack_optional_trace_pytree_field(trace_seq, "freeb_pres_scale")
    if freeb_pres_scale is not None:
        payload["freeb_pres_scale"] = freeb_pres_scale
    if bool(include_constraints):
        for key in _ACCEPTED_TRACE_OPTIONAL_STEP_PYTREE_CONTROL_KEYS:
            if key in ("force_state_pre", "freeb_pres_scale"):
                continue
            value = stack_optional_trace_pytree_field(trace_seq, key)
            if value is not None:
                payload[key] = value
    return payload


def trace_step_policy_static_signature(trace: dict[str, Any]) -> tuple[Any, ...]:
    """Return the static-dispatch signature for one accepted replay step."""

    has_active_freeb_replay = trace.get("freeb_bsqvac_half") is not None and trace.get("freeb_nestor_trace") is not None
    return (
        trace_preconditioner_static_signature(trace),
        int(bool(trace.get("apply_lforbal", False))),
        int(bool(trace.get("include_edge_residual", False))),
        int(bool(trace.get("apply_m1_constraints", False))),
        int(bool(has_active_freeb_replay)),
        trace_static_value_shape_signature(trace.get("wout_like")),
        trace_static_value_shape_signature(trace.get("trig")),
        trace_static_value_shape_signature(trace.get("zero_m1")),
        trace_optional_presence_signature(trace, _ACCEPTED_TRACE_OPTIONAL_STEP_PYTREE_CONTROL_KEYS),
        tuple(
            trace_static_value_shape_signature(trace.get(key))
            for key in _ACCEPTED_TRACE_OPTIONAL_STEP_PYTREE_CONTROL_KEYS
            if trace.get(key) is not None
        ),
    )


def direct_coil_accepted_trace_step_policy_segments(
    traces: Any,
    *,
    max_steps: int | None = None,
) -> list[dict[str, Any]]:
    """Return consecutive static step-policy segments for stacked replay."""

    trace_seq = tuple(traces)
    if max_steps is not None:
        trace_seq = trace_seq[: int(max_steps)]
    if not trace_seq:
        raise ValueError("at least one accepted trace is required")

    segments: list[dict[str, Any]] = []
    start = 0
    current_signature = trace_step_policy_static_signature(trace_seq[0])
    for index, trace in enumerate(trace_seq[1:], start=1):
        signature = trace_step_policy_static_signature(trace)
        if signature == current_signature:
            continue
        segments.append(
            {
                "start": start,
                "stop": index,
                "n_steps": index - start,
                "signature": current_signature,
            }
        )
        start = index
        current_signature = signature
    segments.append(
        {
            "start": start,
            "stop": len(trace_seq),
            "n_steps": len(trace_seq) - start,
            "signature": current_signature,
        }
    )
    return segments


def direct_coil_accepted_trace_step_policy_segment_summary(
    traces: Any,
    *,
    accept_mask: Any | None = None,
    done_mask: Any | None = None,
    max_steps: int | None = None,
) -> list[dict[str, Any]]:
    """Return JSON-safe diagnostics for stacked step-policy segments."""

    trace_seq = tuple(traces)
    if max_steps is not None:
        trace_seq = trace_seq[: int(max_steps)]
    if not trace_seq:
        raise ValueError("at least one accepted trace is required")

    n_steps = len(trace_seq)
    if accept_mask is not None:
        accept_mask = np.asarray(accept_mask, dtype=bool)[:n_steps]
    if done_mask is not None:
        done_mask = np.asarray(done_mask, dtype=bool)[:n_steps]
    controls = direct_coil_accepted_trace_controller_controls_jax(
        trace_seq,
        accept_mask=accept_mask,
        done_mask=done_mask,
    )
    accepted = np.asarray(controls["accept"], dtype=bool)
    done = np.asarray(controls["done"], dtype=bool)
    reset = np.asarray(controls["reset_to_trace_pre"], dtype=bool)
    freeb = np.asarray(controls["has_active_freeb_replay"], dtype=bool)

    summaries: list[dict[str, Any]] = []
    for index, segment in enumerate(direct_coil_accepted_trace_step_policy_segments(trace_seq)):
        start = int(segment["start"])
        stop = int(segment["stop"])
        segment_accept = accepted[start:stop]
        segment_done = done[start:stop]
        segment_reset = reset[start:stop]
        segment_freeb = freeb[start:stop]
        summaries.append(
            {
                "index": int(index),
                "start": start,
                "stop": stop,
                "n_steps": int(stop - start),
                "accepted_steps": int(np.count_nonzero(segment_accept)),
                "rejected_steps": int(segment_accept.size - np.count_nonzero(segment_accept)),
                "done_markers": int(np.count_nonzero(segment_done)),
                "state_resets": int(np.count_nonzero(segment_reset)),
                "free_boundary_replay_steps": int(np.count_nonzero(segment_freeb)),
                "signature_repr": repr(segment["signature"]),
            }
        )
    return summaries


def direct_coil_accepted_trace_preconditioner_policy_segments(
    traces: Any,
    *,
    max_steps: int | None = None,
) -> list[dict[str, Any]]:
    """Return consecutive static-preconditioner-policy segments."""

    trace_seq = tuple(traces)
    if max_steps is not None:
        trace_seq = trace_seq[: int(max_steps)]
    if not trace_seq:
        raise ValueError("at least one accepted trace is required")

    segments: list[dict[str, Any]] = []
    start = 0
    current_signature = trace_preconditioner_static_signature(trace_seq[0])
    for index, trace in enumerate(trace_seq[1:], start=1):
        signature = trace_preconditioner_static_signature(trace)
        if signature == current_signature:
            continue
        segments.append(
            {
                "start": start,
                "stop": index,
                "n_steps": index - start,
                "signature": current_signature,
            }
        )
        start = index
        current_signature = signature
    segments.append(
        {
            "start": start,
            "stop": len(trace_seq),
            "n_steps": len(trace_seq) - start,
            "signature": current_signature,
        }
    )
    return segments


def direct_coil_accepted_trace_preconditioner_policy_segment_summary(
    traces: Any,
    *,
    accept_mask: Any | None = None,
    done_mask: Any | None = None,
    max_steps: int | None = None,
) -> list[dict[str, Any]]:
    """Return JSON-safe preconditioner-policy segment diagnostics."""

    trace_seq = tuple(traces)
    if max_steps is not None:
        trace_seq = trace_seq[: int(max_steps)]
    if not trace_seq:
        raise ValueError("at least one accepted trace is required")

    n_steps = len(trace_seq)
    if accept_mask is not None:
        accept_mask = np.asarray(accept_mask, dtype=bool)[:n_steps]
    if done_mask is not None:
        done_mask = np.asarray(done_mask, dtype=bool)[:n_steps]
    controls = direct_coil_accepted_trace_controller_controls_jax(
        trace_seq,
        accept_mask=accept_mask,
        done_mask=done_mask,
    )
    accepted = np.asarray(controls["accept"], dtype=bool)
    done = np.asarray(controls["done"], dtype=bool)
    reset = np.asarray(controls["reset_to_trace_pre"], dtype=bool)
    freeb = np.asarray(controls["has_active_freeb_replay"], dtype=bool)

    summaries: list[dict[str, Any]] = []
    for index, segment in enumerate(direct_coil_accepted_trace_preconditioner_policy_segments(trace_seq)):
        start = int(segment["start"])
        stop = int(segment["stop"])
        signature = segment["signature"]
        segment_accept = accepted[start:stop]
        segment_done = done[start:stop]
        segment_reset = reset[start:stop]
        segment_freeb = freeb[start:stop]
        summaries.append(
            {
                "index": int(index),
                "start": start,
                "stop": stop,
                "n_steps": int(stop - start),
                "accepted_steps": int(np.count_nonzero(segment_accept)),
                "rejected_steps": int(segment_accept.size - np.count_nonzero(segment_accept)),
                "done_markers": int(np.count_nonzero(segment_done)),
                "state_resets": int(np.count_nonzero(segment_reset)),
                "free_boundary_replay_steps": int(np.count_nonzero(segment_freeb)),
                "preconditioner_use_precomputed_tridi": int(signature[0]),
                "preconditioner_use_lax_tridi": int(signature[1]),
                "precond_jmax": int(signature[2]),
                "signature_repr": repr(signature),
            }
        )
    return summaries
