"""Accepted-branch metadata reports for free-boundary adjoint replay."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from vmec_jax._compat import jnp

from .trace_controls import (
    accepted_trace_effective_controller_masks,
    direct_coil_accepted_trace_controller_controls_jax,
    direct_coil_accepted_trace_status_masks,
)
from .trace_fingerprint import direct_coil_accepted_trace_fingerprint
from .trace_metadata import (
    compact_segment_summaries,
    json_safe_fingerprint_value,
    unique_shape_list,
)
from .trace_stack import (
    direct_coil_accepted_trace_preconditioner_policy_segment_summary,
    direct_coil_accepted_trace_preconditioner_policy_segments,
    direct_coil_accepted_trace_step_policy_segment_summary,
    direct_coil_accepted_trace_step_policy_segments,
)


def direct_coil_accepted_trace_branch_metadata(
    traces: Any,
    *,
    accept_mask: Any | None = None,
    done_mask: Any | None = None,
    max_steps: int | None = None,
    json_safe: bool = False,
) -> dict[str, Any]:
    """Return branch metadata for a fixed accepted free-boundary trace."""

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

    status_masks = direct_coil_accepted_trace_status_masks(trace_seq)
    controls = direct_coil_accepted_trace_controller_controls_jax(
        trace_seq,
        accept_mask=accept_mask,
        done_mask=done_mask,
    )
    masks = accepted_trace_effective_controller_masks(controls)
    freeb = jnp.asarray(controls["has_active_freeb_replay"], dtype=bool)
    active_freeb = jnp.logical_and(jnp.asarray(masks["accepted"], dtype=bool), freeb)
    metadata = {
        "n_steps": int(n_steps),
        "n_free_boundary_replay_steps": int(np.count_nonzero(np.asarray(active_freeb, dtype=bool))),
        "status_masks": status_masks,
        "step_status": status_masks["step_status"],
        "status_acceptance_source": status_masks["status_acceptance_source"],
        "fingerprint": direct_coil_accepted_trace_fingerprint(trace_seq),
        "controller_controls": controls,
        "masks": masks,
        "accepted_mask": jnp.asarray(masks["accepted"], dtype=bool),
        "rejected_mask": jnp.asarray(masks["rejected"], dtype=bool),
        "done_mask": jnp.asarray(masks["done"], dtype=bool),
        "reset_to_trace_pre": jnp.asarray(masks["reset_to_trace_pre"], dtype=bool),
        "has_active_freeb_replay": freeb,
        "active_free_boundary_mask": active_freeb,
        "preconditioner_policy_segments": direct_coil_accepted_trace_preconditioner_policy_segments(trace_seq),
        "preconditioner_policy_segment_summary": direct_coil_accepted_trace_preconditioner_policy_segment_summary(
            trace_seq,
            accept_mask=accept_mask,
            done_mask=done_mask,
        ),
    }
    if json_safe:
        return json_safe_fingerprint_value(metadata)
    return metadata


def direct_coil_accepted_trace_replay_graph_metadata(
    traces: Any,
    *,
    static: Any | None = None,
    accept_mask: Any | None = None,
    done_mask: Any | None = None,
    max_steps: int | None = None,
    sample_nzeta: int | None = None,
    include_analytic: bool = True,
    use_stacked_step_controls: bool = True,
    use_accepted_only_fast_path: bool = True,
    json_safe: bool = False,
) -> dict[str, Any]:
    """Return profiling metadata for the fixed accepted-branch replay graph."""

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
    masks = accepted_trace_effective_controller_masks(controls)
    accepted = np.asarray(masks["accepted"], dtype=bool)
    rejected = np.asarray(masks["rejected"], dtype=bool)
    done = np.asarray(masks["done"], dtype=bool)
    reset = np.asarray(masks["reset_to_trace_pre"], dtype=bool)
    freeb = np.asarray(controls["has_active_freeb_replay"], dtype=bool)
    active_freeb = np.logical_and(accepted, freeb)

    boundary_shapes: list[tuple[int, ...]] = []
    bsqvac_half_shapes: list[tuple[int, ...]] = []
    nestor_axis_shapes: list[tuple[int, ...]] = []
    for trace in trace_seq:
        if trace.get("freeb_bsqvac_half") is not None:
            bsqvac_half_shapes.append(tuple(int(v) for v in np.shape(trace["freeb_bsqvac_half"])))
        nestor_trace = trace.get("freeb_nestor_trace")
        if isinstance(nestor_trace, Mapping):
            for key in ("br_axis", "bp_axis", "bz_axis"):
                if nestor_trace.get(key) is None:
                    continue
                shape = tuple(int(v) for v in np.shape(nestor_trace[key]))
                nestor_axis_shapes.append(shape)
                if len(shape) == 2:
                    boundary_shapes.append(shape)
                break

    inferred_boundary_shape = boundary_shapes[0] if boundary_shapes else None
    static_cfg = getattr(static, "cfg", None)
    nfp = None if static_cfg is None else int(static_cfg.nfp)
    mpol = None if static_cfg is None else int(static_cfg.mpol)
    ntor = None if static_cfg is None else int(static_cfg.ntor)
    lasym = None if static_cfg is None else bool(static_cfg.lasym)
    nvper = None
    if inferred_boundary_shape is not None and nfp is not None:
        nzeta = int(inferred_boundary_shape[1])
        nvper = 64 if nzeta == 1 else max(1, int(nfp))

    metadata = {
        "contract": "fixed accepted-branch replay graph metadata",
        "differentiates_adaptive_controller": False,
        "n_steps": int(n_steps),
        "accepted_steps": int(np.count_nonzero(accepted)),
        "rejected_steps": int(np.count_nonzero(rejected)),
        "done_markers": int(np.count_nonzero(done)),
        "state_resets": int(np.count_nonzero(reset)),
        "free_boundary_trace_steps": int(np.count_nonzero(freeb)),
        "active_free_boundary_replay_steps": int(np.count_nonzero(active_freeb)),
        "step_policy_n_segments": int(len(direct_coil_accepted_trace_step_policy_segments(trace_seq))),
        "step_policy_segment_summary": compact_segment_summaries(
            direct_coil_accepted_trace_step_policy_segment_summary(
                trace_seq,
                accept_mask=accept_mask,
                done_mask=done_mask,
            )
        ),
        "preconditioner_policy_n_segments": int(len(direct_coil_accepted_trace_preconditioner_policy_segments(trace_seq))),
        "preconditioner_policy_segment_summary": compact_segment_summaries(
            direct_coil_accepted_trace_preconditioner_policy_segment_summary(
                trace_seq,
                accept_mask=accept_mask,
                done_mask=done_mask,
            )
        ),
        "boundary_shapes": unique_shape_list(boundary_shapes),
        "bsqvac_half_shapes": unique_shape_list(bsqvac_half_shapes),
        "nestor_axis_shapes": unique_shape_list(nestor_axis_shapes),
        "inferred_boundary_shape": None
        if inferred_boundary_shape is None
        else [int(value) for value in inferred_boundary_shape],
        "sample_nzeta": None if sample_nzeta is None else int(sample_nzeta),
        "nfp": nfp,
        "mpol": mpol,
        "ntor": ntor,
        "lasym": lasym,
        "nvper": nvper,
        "include_analytic": bool(include_analytic),
        "use_stacked_step_controls": bool(use_stacked_step_controls),
        "use_accepted_only_fast_path": bool(use_accepted_only_fast_path),
    }
    if json_safe:
        return json_safe_fingerprint_value(metadata)
    return metadata


__all__ = [
    "direct_coil_accepted_trace_branch_metadata",
    "direct_coil_accepted_trace_replay_graph_metadata",
]
