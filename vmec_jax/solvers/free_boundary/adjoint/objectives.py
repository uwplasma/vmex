"""Objective algebra helpers for free-boundary adjoint validation."""

from __future__ import annotations

from typing import Any

import numpy as np

from vmec_jax._compat import jnp, tree_util
from vmec_jax.state import pack_state


def weighted_half_norm(value: Any, weight: Any) -> Any:
    """Return ``0.5 * sum(weight * value**2)`` with scalar/array weights."""

    arr = jnp.asarray(value)
    w = jnp.asarray(weight, dtype=arr.dtype)
    return 0.5 * jnp.sum(w * arr * arr)


def static_weight_is_zero(weight: Any) -> bool:
    """Return true only for host-known scalar/array weights that are exactly zero."""

    try:
        arr = np.asarray(weight)
    except Exception:
        return False
    if arr.size == 0:
        return False
    try:
        return bool(np.all(arr == 0.0))
    except Exception:
        return False


def tree_weighted_half_norm(values: Any, weight: Any) -> Any:
    """Return the sum of weighted half-norms over numeric pytree leaves."""

    leaves = tree_util.tree_leaves(values)
    if not leaves:
        return jnp.asarray(0.0)
    total = jnp.asarray(0.0)
    for leaf in leaves:
        if leaf is None:
            continue
        try:
            total = total + weighted_half_norm(leaf, weight)
        except TypeError:
            continue
    return total


def accepted_controller_replay_result(
    *,
    run: dict[str, Any],
    controls: dict[str, Any],
    scalar_controls: Any,
    array_controls: Any,
    step_controls: Any,
    preconditioner_controls: Any,
    preconditioner_controls_stacked: bool,
    preconditioner_policy_segments: Any,
    preconditioner_policy_segment_summary: Any,
    step_policy_segments: Any,
    step_policy_segment_summary: Any,
    segment_preconditioner_controls_stacked: Any,
    use_preconditioner_policy_segments: bool,
    use_stacked_step_controls: bool,
    accepted_only_fast_path_segments: Any,
    state_weight: Any,
    include_replay_aux: bool,
    state_only_replay: bool,
) -> dict[str, Any]:
    """Package a controller replay run into objective and auxiliary fields."""

    state_objective = (
        jnp.asarray(0.0)
        if static_weight_is_zero(state_weight)
        else weighted_half_norm(pack_state(run["state"]), state_weight)
    )
    if bool(state_only_replay):
        objective_components = {
            "state": state_objective,
            "force": jnp.asarray(0.0),
            "bsqvac": jnp.asarray(0.0),
        }
    else:
        accepted = jnp.asarray(run["history"]["accepted"], dtype=jnp.asarray(state_objective).dtype)
        objective_components = {
            "state": state_objective,
            "force": jnp.sum(accepted * jnp.asarray(run["history"]["force"])),
            "bsqvac": jnp.sum(accepted * jnp.asarray(run["history"]["bsqvac"])),
        }
    objective = sum(objective_components.values())
    result = {
        "objective": objective,
        "objective_components": objective_components,
        "state": run["state"],
        "history": run["history"],
        "used_state_only_replay": bool(state_only_replay),
    }
    if not bool(include_replay_aux):
        return {
            **result,
            "controls": {
                "has_active_freeb_replay": controls["has_active_freeb_replay"],
            },
        }
    return {
        **result,
        "controls": controls,
        "scalar_controls": scalar_controls,
        "array_controls": array_controls,
        "step_controls": step_controls,
        "preconditioner_controls": preconditioner_controls,
        "preconditioner_controls_stacked": bool(preconditioner_controls_stacked),
        "preconditioner_policy_segments": preconditioner_policy_segments,
        "preconditioner_policy_n_segments": len(preconditioner_policy_segments),
        "preconditioner_policy_segment_summary": preconditioner_policy_segment_summary,
        "step_policy_segments": step_policy_segments,
        "step_policy_n_segments": len(step_policy_segments),
        "step_policy_segment_summary": step_policy_segment_summary,
        "preconditioner_controls_segment_stacked": segment_preconditioner_controls_stacked,
        "used_preconditioner_policy_segments": bool(use_preconditioner_policy_segments),
        "used_stacked_step_controls": bool(use_stacked_step_controls),
        "used_accepted_only_fast_path": bool(any(accepted_only_fast_path_segments)),
        "accepted_only_fast_path_segments": accepted_only_fast_path_segments,
        "state_reset_flags": tuple(bool(flag) for flag in np.asarray(controls["reset_to_trace_pre"], dtype=bool)),
    }
