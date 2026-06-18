"""JAX-visible nonlinear-controller primitives for free-boundary adjoints."""

from __future__ import annotations

from typing import Any

from vmec_jax._compat import jax, jnp, tree_util
from vmec_jax.solvers.free_boundary.adjoint.controller_checks import (
    controller_directional_check_jax as _controller_directional_check_jax,
    pytree_directional_derivative_check_jax,
    pytree_vdot_jax,
)

_pytree_vdot_jax = pytree_vdot_jax


def _normalize_step_output(out: Any) -> tuple[Any, Any]:
    if isinstance(out, tuple) and len(out) == 2:
        return out
    return out, {}


def _step_eval_from(step_fn: Any, params: Any, checkpoint_steps: bool) -> Any:
    def normalize_step(state, control):
        return _normalize_step_output(step_fn(state, params, control))

    return jax.checkpoint(normalize_step) if bool(checkpoint_steps) else normalize_step


def _select_state(flag: Any, old_state: Any, new_state: Any) -> Any:
    return tree_util.tree_map(
        lambda old, new: jnp.where(flag, jnp.asarray(new), jnp.asarray(old)),
        old_state,
        new_state,
    )


def jax_visible_nonlinear_controller_jax(
    step_fn: Any,
    initial_state: Any,
    params: Any,
    controls: Any,
    *,
    checkpoint_steps: bool = False,
) -> dict[str, Any]:
    """Run a nonlinear controller loop entirely through JAX ``lax.scan``."""

    if jax is None:  # pragma: no cover - JAX is required for scan controllers.
        raise RuntimeError("JAX is required for JAX-visible nonlinear controllers.")

    scan_step = _step_eval_from(step_fn, params, checkpoint_steps)
    final_state, history = jax.lax.scan(scan_step, initial_state, controls)
    return {"state": final_state, "history": history}


def jax_visible_masked_nonlinear_controller_jax(
    step_fn: Any,
    converged_fn: Any,
    initial_state: Any,
    params: Any,
    controls: Any,
    *,
    checkpoint_steps: bool = False,
) -> dict[str, Any]:
    """Run a fixed-length JAX controller with JAX-visible convergence masking."""

    if jax is None:  # pragma: no cover - JAX is required for scan controllers.
        raise RuntimeError("JAX is required for JAX-visible nonlinear controllers.")

    step_eval = _step_eval_from(step_fn, params, checkpoint_steps)

    def _scan_step(carry, control):
        state, done = carry
        proposed_state, aux = step_eval(state, control)
        active = jnp.logical_not(done)
        state_out = _select_state(active, state, proposed_state)
        proposed_done = jnp.asarray(converged_fn(proposed_state, params, control, aux), dtype=bool)
        done_out = jnp.logical_or(done, proposed_done)
        aux_out = dict(aux) if isinstance(aux, dict) else {"aux": aux}
        aux_out["active"] = active
        aux_out["done"] = done_out
        return (state_out, done_out), aux_out

    (final_state, final_done), history = jax.lax.scan(
        _scan_step,
        (initial_state, jnp.asarray(False)),
        controls,
    )
    return {"state": final_state, "done": final_done, "history": history}


def jax_visible_accepted_nonlinear_controller_jax(
    step_fn: Any,
    accept_fn: Any,
    converged_fn: Any,
    initial_state: Any,
    params: Any,
    controls: Any,
    *,
    checkpoint_steps: bool = False,
    initial_done: Any = False,
) -> dict[str, Any]:
    """Run a JAX-visible fixed-length controller with accept/reject masks."""

    if jax is None:  # pragma: no cover - JAX is required for scan controllers.
        raise RuntimeError("JAX is required for JAX-visible nonlinear controllers.")

    step_eval = _step_eval_from(step_fn, params, checkpoint_steps)

    def _scan_step(carry, control):
        state, done = carry
        proposed_state, aux = step_eval(state, control)
        active = jnp.logical_not(done)
        accepted_proposal = jnp.asarray(accept_fn(state, proposed_state, params, control, aux), dtype=bool)
        accepted = jnp.logical_and(active, accepted_proposal)
        rejected = jnp.logical_and(active, jnp.logical_not(accepted_proposal))
        state_after_accept = _select_state(accepted, state, proposed_state)
        accepted_done = jnp.asarray(converged_fn(state_after_accept, params, control, aux), dtype=bool)
        done_out = jnp.logical_or(done, jnp.logical_and(accepted, accepted_done))
        aux_out = dict(aux) if isinstance(aux, dict) else {"aux": aux}
        aux_out["active"] = active
        aux_out["accepted"] = accepted
        aux_out["rejected"] = rejected
        aux_out["done"] = done_out
        return (state_after_accept, done_out), aux_out

    (final_state, final_done), history = jax.lax.scan(
        _scan_step,
        (initial_state, jnp.asarray(initial_done, dtype=bool)),
        controls,
    )
    return {"state": final_state, "done": final_done, "history": history}


def jax_visible_accepted_only_nonlinear_controller_jax(
    step_fn: Any,
    converged_fn: Any,
    initial_state: Any,
    params: Any,
    controls: Any,
    *,
    checkpoint_steps: bool = False,
    initial_done: Any = False,
) -> dict[str, Any]:
    """Run a bounded JAX-visible segment whose proposals are all accepted.

    The caller must bound ``controls`` to a segment that is active for every
    slot, with at most a final convergence marker.  This keeps the accepted-only
    replay free of accept/reject state selection while still emitting the
    controller history masks consumed by replay objectives.
    """

    if jax is None:  # pragma: no cover - JAX is required for scan controllers.
        raise RuntimeError("JAX is required for JAX-visible nonlinear controllers.")

    step_eval = _step_eval_from(step_fn, params, checkpoint_steps)

    def _scan_step(carry, control):
        state, done = carry
        proposed_state, aux = step_eval(state, control)
        active = jnp.logical_not(done)
        state_out = _select_state(active, state, proposed_state)
        accepted_done = jnp.asarray(converged_fn(proposed_state, params, control, aux), dtype=bool)
        done_out = jnp.logical_or(done, jnp.logical_and(active, accepted_done))
        aux_out = dict(aux) if isinstance(aux, dict) else {"aux": aux}
        aux_out["active"] = active
        aux_out["accepted"] = active
        aux_out["rejected"] = jnp.zeros_like(active, dtype=bool)
        aux_out["done"] = done_out
        return (state_out, done_out), aux_out

    (final_state, final_done), history = jax.lax.scan(
        _scan_step,
        (initial_state, jnp.asarray(initial_done, dtype=bool)),
        controls,
    )
    return {"state": final_state, "done": final_done, "history": history}


def jax_visible_state_only_accepted_nonlinear_controller_jax(
    step_fn: Any,
    accept_fn: Any,
    converged_fn: Any,
    initial_state: Any,
    params: Any,
    controls: Any,
    *,
    checkpoint_steps: bool = False,
    initial_done: Any = False,
) -> dict[str, Any]:
    """Run an accepted/rejected controller without materializing history.

    This is for final-state-only replay reports.  It still applies the same
    accept/reject and convergence masks, but the scan returns only the final
    state and final done flag so cold JVP reports do not carry unused
    per-step diagnostics through the graph.
    """

    if jax is None:  # pragma: no cover - JAX is required for controllers.
        raise RuntimeError("JAX is required for JAX-visible nonlinear controllers.")

    step_eval = _step_eval_from(step_fn, params, checkpoint_steps)

    def _scan_step(carry, control):
        state, done = carry
        proposed_state, aux = step_eval(state, control)
        active = jnp.logical_not(done)
        accepted_proposal = jnp.asarray(accept_fn(state, proposed_state, params, control, aux), dtype=bool)
        accepted = jnp.logical_and(active, accepted_proposal)
        state_after_accept = _select_state(accepted, state, proposed_state)
        accepted_done = jnp.asarray(converged_fn(state_after_accept, params, control, aux), dtype=bool)
        done_out = jnp.logical_or(done, jnp.logical_and(accepted, accepted_done))
        return (state_after_accept, done_out), None

    (final_state, final_done), _history = jax.lax.scan(
        _scan_step,
        (initial_state, jnp.asarray(initial_done, dtype=bool)),
        controls,
    )
    return {"state": final_state, "done": final_done, "history": {}}


def jax_visible_state_only_accepted_only_nonlinear_controller_jax(
    step_fn: Any,
    converged_fn: Any,
    initial_state: Any,
    params: Any,
    controls: Any,
    *,
    checkpoint_steps: bool = False,
    initial_done: Any = False,
) -> dict[str, Any]:
    """Run an all-accepted controller segment without history materialization."""

    if jax is None:  # pragma: no cover - JAX is required for controllers.
        raise RuntimeError("JAX is required for JAX-visible nonlinear controllers.")

    step_eval = _step_eval_from(step_fn, params, checkpoint_steps)

    def _scan_step(carry, control):
        state, done = carry
        proposed_state, aux = step_eval(state, control)
        active = jnp.logical_not(done)
        state_out = _select_state(active, state, proposed_state)
        accepted_done = jnp.asarray(converged_fn(proposed_state, params, control, aux), dtype=bool)
        done_out = jnp.logical_or(done, jnp.logical_and(active, accepted_done))
        return (state_out, done_out), None

    (final_state, final_done), _history = jax.lax.scan(
        _scan_step,
        (initial_state, jnp.asarray(initial_done, dtype=bool)),
        controls,
    )
    return {"state": final_state, "done": final_done, "history": {}}


def _control_leading_size(controls: Any) -> int:
    leaves = tree_util.tree_leaves(controls)
    if not leaves:
        raise ValueError("controls must contain at least one array leaf")
    return int(jnp.asarray(leaves[0]).shape[0])


def _slice_control(controls: Any, index: int) -> Any:
    return tree_util.tree_map(lambda value, index=index: jnp.asarray(value)[index], controls)


def jax_visible_unrolled_accepted_only_nonlinear_controller_jax(
    step_fn: Any,
    converged_fn: Any,
    initial_state: Any,
    params: Any,
    controls: Any,
    *,
    checkpoint_steps: bool = False,
    initial_done: Any = False,
) -> dict[str, Any]:
    """Run a short accepted-only segment with a Python-unrolled JAX graph."""

    if jax is None:  # pragma: no cover - JAX is required for controllers.
        raise RuntimeError("JAX is required for JAX-visible nonlinear controllers.")

    step_eval = _step_eval_from(step_fn, params, checkpoint_steps)

    state = initial_state
    done = jnp.asarray(initial_done, dtype=bool)
    history_items = []
    for index in range(_control_leading_size(controls)):
        control = _slice_control(controls, index)
        proposed_state, aux = step_eval(state, control)
        active = jnp.logical_not(done)
        state = _select_state(active, state, proposed_state)
        accepted_done = jnp.asarray(converged_fn(proposed_state, params, control, aux), dtype=bool)
        done = jnp.logical_or(done, jnp.logical_and(active, accepted_done))
        aux_out = dict(aux) if isinstance(aux, dict) else {"aux": aux}
        aux_out["active"] = active
        aux_out["accepted"] = active
        aux_out["rejected"] = jnp.zeros_like(active, dtype=bool)
        aux_out["done"] = done
        history_items.append(aux_out)
    if not history_items:
        raise ValueError("controls must contain at least one step")
    history = {
        key: jnp.stack([jnp.asarray(item[key]) for item in history_items], axis=0)
        for key in history_items[0]
    }
    return {"state": state, "done": done, "history": history}


def jax_visible_unrolled_state_only_accepted_only_nonlinear_controller_jax(
    step_fn: Any,
    converged_fn: Any,
    initial_state: Any,
    params: Any,
    controls: Any,
    *,
    checkpoint_steps: bool = False,
    initial_done: Any = False,
) -> dict[str, Any]:
    """Run a short all-accepted state-only segment with a Python-unrolled graph."""

    if jax is None:  # pragma: no cover - JAX is required for controllers.
        raise RuntimeError("JAX is required for JAX-visible nonlinear controllers.")

    step_eval = _step_eval_from(step_fn, params, checkpoint_steps)

    state = initial_state
    done = jnp.asarray(initial_done, dtype=bool)
    for index in range(_control_leading_size(controls)):
        control = _slice_control(controls, index)
        proposed_state, aux = step_eval(state, control)
        active = jnp.logical_not(done)
        state = _select_state(active, state, proposed_state)
        accepted_done = jnp.asarray(converged_fn(proposed_state, params, control, aux), dtype=bool)
        done = jnp.logical_or(done, jnp.logical_and(active, accepted_done))
    return {"state": state, "done": done, "history": {}}


def jax_visible_segmented_accepted_nonlinear_controller_jax(
    step_fns: Any,
    accept_fn: Any,
    converged_fn: Any,
    initial_state: Any,
    params: Any,
    control_segments: Any,
    *,
    checkpoint_steps: bool = False,
    initial_done: Any = False,
    accepted_only_segments: Any | None = None,
    unroll_accepted_only_segments_below: int = 0,
) -> dict[str, Any]:
    """Run accepted/rejected JAX-visible controllers over static segments."""

    if jax is None:  # pragma: no cover - JAX is required for scan controllers.
        raise RuntimeError("JAX is required for JAX-visible nonlinear controllers.")

    segments = tuple(control_segments)
    if not segments:
        raise ValueError("control_segments must contain at least one segment")

    if callable(step_fns):
        step_fn_seq = (step_fns,) * len(segments)
    else:
        step_fn_seq = tuple(step_fns)
        if len(step_fn_seq) != len(segments):
            raise ValueError(
                f"step_fns length {len(step_fn_seq)} does not match control_segments length {len(segments)}"
            )

    if accepted_only_segments is None:
        accepted_only_segment_seq = (False,) * len(segments)
    else:
        accepted_only_segment_seq = tuple(bool(flag) for flag in accepted_only_segments)
        if len(accepted_only_segment_seq) != len(segments):
            raise ValueError(
                "accepted_only_segments length "
                f"{len(accepted_only_segment_seq)} does not match control_segments length {len(segments)}"
            )

    state = initial_state
    done = jnp.asarray(initial_done, dtype=bool)
    histories = []
    for step_fn, controls, accepted_only in zip(step_fn_seq, segments, accepted_only_segment_seq, strict=True):
        if bool(accepted_only):
            segment_len = _control_leading_size(controls)
            use_unrolled = int(unroll_accepted_only_segments_below) > 0 and segment_len <= int(
                unroll_accepted_only_segments_below
            )
            runner = (
                jax_visible_unrolled_accepted_only_nonlinear_controller_jax
                if use_unrolled
                else jax_visible_accepted_only_nonlinear_controller_jax
            )
            run = runner(
                step_fn,
                converged_fn,
                state,
                params,
                controls,
                checkpoint_steps=checkpoint_steps,
                initial_done=done,
            )
        else:
            run = jax_visible_accepted_nonlinear_controller_jax(
                step_fn,
                accept_fn,
                converged_fn,
                state,
                params,
                controls,
                checkpoint_steps=checkpoint_steps,
                initial_done=done,
            )
        state = run["state"]
        done = run["done"]
        histories.append(run["history"])

    if len(histories) == 1:
        history = histories[0]
    else:
        history = tree_util.tree_map(
            lambda *parts: jnp.concatenate([jnp.asarray(part) for part in parts], axis=0),
            *histories,
        )
    return {"state": state, "done": done, "history": history, "n_segments": len(segments)}


def jax_visible_segmented_state_only_accepted_nonlinear_controller_jax(
    step_fns: Any,
    accept_fn: Any,
    converged_fn: Any,
    initial_state: Any,
    params: Any,
    control_segments: Any,
    *,
    checkpoint_steps: bool = False,
    initial_done: Any = False,
    accepted_only_segments: Any | None = None,
    unroll_accepted_only_segments_below: int = 0,
) -> dict[str, Any]:
    """Run accepted/rejected controller segments and keep only final state."""

    if jax is None:  # pragma: no cover - JAX is required for controllers.
        raise RuntimeError("JAX is required for JAX-visible nonlinear controllers.")

    segments = tuple(control_segments)
    if not segments:
        raise ValueError("control_segments must contain at least one segment")

    if callable(step_fns):
        step_fn_seq = (step_fns,) * len(segments)
    else:
        step_fn_seq = tuple(step_fns)
        if len(step_fn_seq) != len(segments):
            raise ValueError(
                f"step_fns length {len(step_fn_seq)} does not match control_segments length {len(segments)}"
            )

    if accepted_only_segments is None:
        accepted_only_segment_seq = (False,) * len(segments)
    else:
        accepted_only_segment_seq = tuple(bool(flag) for flag in accepted_only_segments)
        if len(accepted_only_segment_seq) != len(segments):
            raise ValueError(
                "accepted_only_segments length "
                f"{len(accepted_only_segment_seq)} does not match control_segments length {len(segments)}"
            )

    state = initial_state
    done = jnp.asarray(initial_done, dtype=bool)
    for step_fn, controls, accepted_only in zip(step_fn_seq, segments, accepted_only_segment_seq, strict=True):
        if bool(accepted_only):
            segment_len = _control_leading_size(controls)
            use_unrolled = int(unroll_accepted_only_segments_below) > 0 and segment_len <= int(
                unroll_accepted_only_segments_below
            )
            runner = (
                jax_visible_unrolled_state_only_accepted_only_nonlinear_controller_jax
                if use_unrolled
                else jax_visible_state_only_accepted_only_nonlinear_controller_jax
            )
            run = runner(
                step_fn,
                converged_fn,
                state,
                params,
                controls,
                checkpoint_steps=checkpoint_steps,
                initial_done=done,
            )
        else:
            run = jax_visible_state_only_accepted_nonlinear_controller_jax(
                step_fn,
                accept_fn,
                converged_fn,
                state,
                params,
                controls,
                checkpoint_steps=checkpoint_steps,
                initial_done=done,
            )
        state = run["state"]
        done = run["done"]
    return {"state": state, "done": done, "history": {}, "n_segments": len(segments)}


def jax_visible_nonlinear_controller_directional_check_jax(
    step_fn: Any,
    objective_from_run: Any,
    params: Any,
    direction: Any,
    initial_state: Any,
    controls: Any,
    *,
    eps: float = 1.0e-4,
    checkpoint_steps: bool = False,
) -> dict[str, Any]:
    """AD-vs-FD check for a fully JAX-visible nonlinear controller."""

    def run_controller(controller_params):
        return jax_visible_nonlinear_controller_jax(
            step_fn,
            initial_state,
            controller_params,
            controls,
            checkpoint_steps=checkpoint_steps,
        )

    return _controller_directional_check_jax(run_controller, objective_from_run, params, direction, eps=eps)


def jax_visible_masked_nonlinear_controller_directional_check_jax(
    step_fn: Any,
    converged_fn: Any,
    objective_from_run: Any,
    params: Any,
    direction: Any,
    initial_state: Any,
    controls: Any,
    *,
    eps: float = 1.0e-4,
    checkpoint_steps: bool = False,
) -> dict[str, Any]:
    """AD-vs-FD check for a JAX-visible masked nonlinear controller."""

    def run_controller(controller_params):
        return jax_visible_masked_nonlinear_controller_jax(
            step_fn,
            converged_fn,
            initial_state,
            controller_params,
            controls,
            checkpoint_steps=checkpoint_steps,
        )

    return _controller_directional_check_jax(run_controller, objective_from_run, params, direction, eps=eps)


def jax_visible_accepted_nonlinear_controller_directional_check_jax(
    step_fn: Any,
    accept_fn: Any,
    converged_fn: Any,
    objective_from_run: Any,
    params: Any,
    direction: Any,
    initial_state: Any,
    controls: Any,
    *,
    eps: float = 1.0e-4,
    checkpoint_steps: bool = False,
) -> dict[str, Any]:
    """AD-vs-FD check for accepted/rejected JAX-visible controllers."""

    def run_controller(controller_params):
        return jax_visible_accepted_nonlinear_controller_jax(
            step_fn,
            accept_fn,
            converged_fn,
            initial_state,
            controller_params,
            controls,
            checkpoint_steps=checkpoint_steps,
        )

    return _controller_directional_check_jax(run_controller, objective_from_run, params, direction, eps=eps)


__all__ = [
    "jax_visible_accepted_only_nonlinear_controller_jax",
    "jax_visible_unrolled_accepted_only_nonlinear_controller_jax",
    "jax_visible_accepted_nonlinear_controller_directional_check_jax",
    "jax_visible_accepted_nonlinear_controller_jax",
    "jax_visible_masked_nonlinear_controller_directional_check_jax",
    "jax_visible_masked_nonlinear_controller_jax",
    "jax_visible_nonlinear_controller_directional_check_jax",
    "jax_visible_nonlinear_controller_jax",
    "jax_visible_segmented_accepted_nonlinear_controller_jax",
    "pytree_directional_derivative_check_jax",
]
