"""Accepted-trace stacking and static-signature helpers."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from vmec_jax._compat import jnp, tree_util

from .free_boundary_adjoint_trace_fingerprint import trace_pytree_shape_signature


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
