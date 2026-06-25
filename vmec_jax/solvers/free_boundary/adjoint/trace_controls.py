"""Accepted-trace controller masks for free-boundary adjoint replay."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from vmec_jax._compat import jnp
from vmec_jax.state import pack_state


def accepted_trace_effective_state_pre(trace: Mapping[str, Any]) -> Any:
    """Return the state actually used by the recorded accepted update.

    Momentum/restart traces can keep ``state_pre`` as a controller checkpoint
    while computing the accepted update from ``force_state_pre``.  Replays must
    use ``force_state_pre`` when it is present to reconstruct the production
    accepted state exactly.
    """

    force_state_pre = trace.get("force_state_pre")
    return force_state_pre if force_state_pre is not None else trace.get("state_pre")


def _accepted_trace_state_reset_between(prev_trace: dict[str, Any], trace: dict[str, Any]) -> bool:
    prev_post = prev_trace.get("state_post")
    next_pre = accepted_trace_effective_state_pre(trace)
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


def accepted_trace_effective_controller_masks(controls: Mapping[str, Any]) -> dict[str, Any]:
    """Return effective accepted/rejected/done masks for controller controls."""

    accept_control = np.asarray(controls["accept"], dtype=bool)
    done_control = np.asarray(controls["done"], dtype=bool)
    active_values = []
    accepted_values = []
    rejected_values = []
    done_values = []
    done = False
    for accept_i, done_i in zip(accept_control, done_control, strict=True):
        active = not done
        accepted = bool(active and accept_i)
        rejected = bool(active and not accept_i)
        done = bool(done or (accepted and done_i))
        active_values.append(active)
        accepted_values.append(accepted)
        rejected_values.append(rejected)
        done_values.append(done)
    return {
        "accept_control": jnp.asarray(accept_control, dtype=bool),
        "done_control": jnp.asarray(done_control, dtype=bool),
        "active": jnp.asarray(active_values, dtype=bool),
        "accepted": jnp.asarray(accepted_values, dtype=bool),
        "rejected": jnp.asarray(rejected_values, dtype=bool),
        "done": jnp.asarray(done_values, dtype=bool),
        "reset_to_trace_pre": jnp.asarray(controls["reset_to_trace_pre"], dtype=bool),
        "has_active_freeb_replay": jnp.asarray(controls["has_active_freeb_replay"], dtype=bool),
    }


def accepted_trace_segment_is_unconditionally_accepted(
    masks: Mapping[str, Any],
    *,
    start: int,
    stop: int,
) -> bool:
    """Return whether a controller segment can skip accept/reject conditionals."""

    active = np.asarray(masks["active"], dtype=bool)[int(start) : int(stop)]
    accepted = np.asarray(masks["accepted"], dtype=bool)[int(start) : int(stop)]
    rejected = np.asarray(masks["rejected"], dtype=bool)[int(start) : int(stop)]
    done = np.asarray(masks["done"], dtype=bool)[int(start) : int(stop)]
    if active.size == 0:
        return False
    if not bool(np.all(active)):
        return False
    if not bool(np.all(accepted)):
        return False
    if bool(np.any(rejected)):
        return False
    # A final done marker is allowed. Any earlier done marker would make later
    # scan entries inactive in the ordinary controller, so it must not fast-path.
    if done.size > 1 and bool(np.any(done[:-1])):
        return False
    return True
