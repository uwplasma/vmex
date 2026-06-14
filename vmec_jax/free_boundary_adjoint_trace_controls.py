"""Accepted-trace controller masks for free-boundary adjoint replay."""

from __future__ import annotations

from typing import Any

import numpy as np

from vmec_jax._compat import jnp


def _accepted_trace_state_reset_between(prev_trace: dict[str, Any], trace: dict[str, Any]) -> bool:
    from .state import pack_state

    prev_post = prev_trace.get("state_post")
    next_pre = trace.get("state_pre")
    if prev_post is None or next_pre is None:
        return False
    try:
        prev_packed = np.asarray(pack_state(prev_post), dtype=float)
        next_packed = np.asarray(pack_state(next_pre), dtype=float)
    except Exception:
        return False
    if prev_packed.shape != next_packed.shape:
        return True
    return not np.allclose(prev_packed, next_packed, rtol=1.0e-13, atol=1.0e-13)


def _accepted_trace_reset_flags(trace_seq: Any) -> tuple[bool, ...]:
    traces_tuple = tuple(trace_seq)
    if not traces_tuple:
        return ()
    return (False,) + tuple(
        _accepted_trace_state_reset_between(prev_trace, trace)
        for prev_trace, trace in zip(traces_tuple[:-1], traces_tuple[1:], strict=False)
    )


_REJECTED_TRACE_STEP_STATUSES = {
    "rejected",
    "restart_bad_jacobian",
    "restart_bad_progress",
    "restart_stage_transition",
    "restart_time_control",
}

_DONE_TRACE_STEP_STATUSES = {
    "converged",
}


def _normalise_trace_step_status(status: Any) -> str:
    text = str(status).strip().lower()
    return text if text else "accepted"


def _trace_step_status_is_rejected(status: str) -> bool:
    return status in _REJECTED_TRACE_STEP_STATUSES or status.startswith("restart_")


def direct_coil_accepted_trace_status_masks(traces: Any) -> dict[str, Any]:
    """Derive replay accept/done masks from production trace ``step_status``."""

    trace_seq = tuple(traces)
    if not trace_seq:
        raise ValueError("at least one accepted trace is required")
    statuses = tuple(_normalise_trace_step_status(trace.get("step_status", "accepted")) for trace in trace_seq)
    accept_mask = np.asarray([not _trace_step_status_is_rejected(status) for status in statuses], dtype=bool)
    done_mask = np.asarray([status in _DONE_TRACE_STEP_STATUSES for status in statuses], dtype=bool)
    if not np.any(done_mask):
        done_mask[-1] = True
    return {
        "step_status": statuses,
        "accept_mask": accept_mask,
        "done_mask": done_mask,
        "status_acceptance_source": "trace_step_status",
    }


def direct_coil_accepted_trace_controller_controls_jax(
    traces: Any,
    *,
    accept_mask: Any | None = None,
    done_mask: Any | None = None,
) -> dict[str, Any]:
    """Return stacked JAX-visible controls for fixed accepted trace replay."""

    trace_seq = tuple(traces)
    if not trace_seq:
        raise ValueError("at least one accepted trace is required")
    step_count = len(trace_seq)
    status_masks = direct_coil_accepted_trace_status_masks(trace_seq)
    if accept_mask is None:
        accept_arr = jnp.asarray(status_masks["accept_mask"], dtype=bool)
    else:
        if np.shape(accept_mask) != (step_count,):
            raise ValueError("accept_mask must have shape (n_steps,)")
        accept_arr = jnp.asarray(accept_mask, dtype=bool)
    if done_mask is None:
        done_arr = jnp.asarray(status_masks["done_mask"], dtype=bool)
    else:
        if np.shape(done_mask) != (step_count,):
            raise ValueError("done_mask must have shape (n_steps,)")
        done_arr = jnp.asarray(done_mask, dtype=bool)
    return {
        "step_index": jnp.arange(step_count, dtype=jnp.int32),
        "accept": accept_arr,
        "done": done_arr,
        "reset_to_trace_pre": jnp.asarray(_accepted_trace_reset_flags(trace_seq), dtype=bool),
        "has_active_freeb_replay": jnp.asarray(
            [
                trace.get("freeb_bsqvac_half") is not None and trace.get("freeb_nestor_trace") is not None
                for trace in trace_seq
            ],
            dtype=bool,
        ),
    }
