# ruff: noqa: F401
"""Adjoint scaffolding for free-boundary vacuum solves.

Phase 1 intentionally keeps this module small and explicit.  It validates the
linear-solve differentiation contract that the production NESTOR replacement
will need: solve the primal system in the forward pass and use transpose solves
in the backward pass rather than differentiating through an iterative solver.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Mapping
from contextlib import nullcontext
import types
from typing import Any, NamedTuple

import numpy as np

from vmec_jax._compat import jax, jnp, tree_util

from vmec_jax.solvers.free_boundary.adjoint.controller import (
    jax_visible_accepted_only_nonlinear_controller_jax,
    jax_visible_accepted_nonlinear_controller_directional_check_jax,
    jax_visible_accepted_nonlinear_controller_jax,
    jax_visible_masked_nonlinear_controller_directional_check_jax,
    jax_visible_masked_nonlinear_controller_jax,
    jax_visible_nonlinear_controller_directional_check_jax,
    jax_visible_nonlinear_controller_jax,
    jax_visible_segmented_accepted_nonlinear_controller_jax,
    jax_visible_segmented_state_only_accepted_nonlinear_controller_jax,
    jax_visible_state_only_accepted_nonlinear_controller_jax,
    jax_visible_state_only_accepted_only_nonlinear_controller_jax,
    jax_visible_unrolled_state_only_accepted_only_nonlinear_controller_jax,
    jax_visible_unrolled_accepted_only_nonlinear_controller_jax,
    pytree_directional_derivative_check_jax,
)
from vmec_jax.solvers.free_boundary.adjoint.objectives import (
    accepted_controller_replay_result as _accepted_controller_replay_result,  # noqa: F401 - compatibility alias.
    static_weight_is_zero as _static_weight_is_zero,  # noqa: F401 - compatibility alias for tests/internal users.
    tree_weighted_half_norm as _tree_weighted_half_norm,  # noqa: F401 - compatibility alias.
    weighted_half_norm as _weighted_half_norm,  # noqa: F401 - compatibility alias.
)
from vmec_jax.solvers.free_boundary.adjoint.pytrees import (
    pytree_batched_directional_vdot_jax as _pytree_batched_directional_vdot_jax,
    pytree_pullback_basis_jax as _pytree_pullback_basis_jax,
    pytree_unstack_leading_axis_jax as _pytree_unstack_leading_axis_jax,
)
from vmec_jax.solvers.free_boundary.adjoint.trace_controls import (
    _accepted_trace_reset_flags,  # noqa: F401 - compatibility alias for tests/internal users.
    accepted_trace_effective_controller_masks as _accepted_trace_effective_controller_masks,  # noqa: F401 - compatibility alias.
    accepted_trace_effective_state_pre,
    accepted_trace_segment_is_unconditionally_accepted as _accepted_trace_segment_is_unconditionally_accepted,  # noqa: F401 - compatibility alias.
    direct_coil_accepted_trace_controller_controls_jax,  # noqa: F401 - compatibility alias.
    direct_coil_accepted_trace_status_masks,
)
from vmec_jax.solvers.free_boundary.adjoint.trace_metadata import (
    _compact_segment_summaries,  # noqa: F401 - compatibility alias for tests/internal users.
    _fingerprint_has_rejected_controller_slot,  # noqa: F401 - compatibility alias for tests/internal users.
    _json_safe_fingerprint_value,  # noqa: F401 - compatibility alias for tests/internal users.
    _unique_shape_list,  # noqa: F401 - compatibility alias for tests/internal users.
    direct_coil_accepted_trace_controller_slot_fingerprint,
    direct_coil_accepted_trace_controller_slot_summary,
)
from vmec_jax.solvers.free_boundary.adjoint.gate_reports import (
    direct_coil_branch_local_scalars_report_from_complete_fd,
    direct_coil_adaptive_full_loop_same_branch_gate_report,
    direct_coil_same_branch_physical_scalar_gate_report,
    direct_coil_same_branch_replay_gate_report,
)
from vmec_jax.solvers.free_boundary.adjoint.branch_metadata import (
    direct_coil_accepted_trace_branch_metadata,
    direct_coil_accepted_trace_replay_graph_metadata,
)
from vmec_jax.solvers.free_boundary.adjoint.branch_local import (
    branch_local_replay_option_flags as _branch_local_replay_option_flags,
    evaluate_branch_local_production_values as _evaluate_branch_local_production_values,
    prepare_branch_local_payload as _prepare_branch_local_payload,
    prepare_branch_local_replay_setup as _prepare_branch_local_replay_setup,
    select_branch_local_scalar_key as _select_branch_local_scalar_key,
    select_branch_local_scalar_keys as _select_branch_local_scalar_keys,
)
from vmec_jax.solvers.free_boundary.adjoint.complete_solve_reports import (
    direct_coil_accepted_trace_replay_objective_jax,
    direct_coil_complete_solve_trace,
    direct_coil_same_branch_complete_solve_fd_report,
    free_boundary_adjoint_trace_replay_diagnostics,
)
from vmec_jax.solvers.free_boundary.adjoint.controller_replay import (
    direct_coil_accepted_trace_controller_replay_objective_jax,
)
from vmec_jax.solvers.free_boundary.adjoint.custom_vjp import (
    scalar_custom_vjp_value_jax as _scalar_custom_vjp_value_jax,
    vector_custom_vjp_value_jax as _vector_custom_vjp_value_jax,
)
from vmec_jax.solvers.free_boundary.adjoint.dense import (
    dense_fixed_point_solve_jax,
    dense_nonlinear_solve_jax,
    dense_vacuum_residual,
    dense_vacuum_solve_jax,
    finite_difference_jacobian as _finite_difference_jacobian,
)
from vmec_jax.solvers.free_boundary.adjoint.mode_operator import (
    mode_matrix_from_grpmn_jax,
    mode_matrix_matvec_from_grpmn_jax,
    mode_operator_vacuum_solve_jax,
    mode_rhs_from_gsource_jax,
    vmec_source_from_gsource_jax,
)
from vmec_jax.solvers.free_boundary.adjoint.mode_solve import dense_mode_vacuum_solve_jax
from vmec_jax.solvers.free_boundary.adjoint.vmec_nestor import (
    dense_vmec_nestor_mode_solve_jax,
    vmec_analytic_terms_from_geometry_jax,
    vmec_nonsingular_terms_from_bexni_jax,
)
from vmec_jax.solvers.free_boundary.adjoint.projected_modes import (
    direct_coil_projected_mode_fixed_point_directional_check_jax as _direct_coil_projected_mode_fixed_point_directional_check_jax_impl,
    direct_coil_projected_mode_fixed_point_jax,
    direct_coil_projected_mode_fixed_point_objective_jax,
)
from vmec_jax.solvers.free_boundary.adjoint.boundary_replay import (
    direct_coil_boundary_bnormal_rms_jax,
    free_boundary_boundary_geometry_jax,
    vacuum_boundary_fields_from_cylindrical_jax,
    vacuum_boundary_fields_from_mode_coeffs_jax,
)
from vmec_jax.solvers.free_boundary.adjoint.direct_coil_replay import (
    direct_coil_boundary_bsqvac_from_trace_jax,
    direct_coil_boundary_bsqvac_jax,
)
from vmec_jax.solvers.free_boundary.adjoint.replay_context import (
    direct_coil_boundary_replay_context,
    direct_coil_boundary_replay_context_for_shape,
    direct_coil_trace_boundary_shape as _direct_coil_trace_boundary_shape,  # noqa: F401 - compatibility alias.
    direct_coil_trace_vacuum_field_override as _direct_coil_trace_vacuum_field_override,  # noqa: F401 - compatibility alias.
    with_jax_nonsingular_replay_tables as _with_jax_nonsingular_replay_tables,  # noqa: F401 - compatibility alias.
)
from vmec_jax.solvers.free_boundary.adjoint.runtime import (
    block_until_ready_for_timing as _runtime_block_until_ready_for_timing,
    jax_named_scope as _runtime_jax_named_scope,
)
from vmec_jax.solvers.free_boundary.adjoint.replay_plan import (
    accepted_step_policy_layout_for_complete_payload as _accepted_step_policy_layout_for_complete_payload,  # noqa: F401 - compatibility alias for tests/internal users.
    accepted_step_policy_signature_for_complete_payload as _accepted_step_policy_signature_for_complete_payload,  # noqa: F401 - compatibility alias for tests/internal users.
    accepted_step_policy_summary_for_complete_payload as _accepted_step_policy_summary_for_complete_payload,  # noqa: F401 - compatibility alias for tests/internal users.
    complete_solve_objective_values as _complete_solve_objective_values,  # noqa: F401 - compatibility alias.
    direct_coil_accepted_trace_controller_replay_plan,
    direct_coil_boundary_replay_contexts_by_shape as _direct_coil_boundary_replay_contexts_by_shape,  # noqa: F401 - compatibility alias.
    extract_adjoint_step_trace as _extract_adjoint_step_trace,  # noqa: F401 - compatibility alias.
    slice_replay_controls as _slice_replay_controls,  # noqa: F401 - compatibility alias.
    stackability_probe as _stackability_probe,  # noqa: F401 - compatibility alias.
)
from vmec_jax.solvers.free_boundary.adjoint.trace_stack import (
    ACCEPTED_TRACE_BOOL_CONTROL_KEYS as _ACCEPTED_TRACE_BOOL_CONTROL_KEYS,  # noqa: F401 - compatibility alias for tests/internal users.
    ACCEPTED_TRACE_NUMERIC_CONTROL_KEYS as _ACCEPTED_TRACE_NUMERIC_CONTROL_KEYS,  # noqa: F401 - compatibility alias for tests/internal users.
    ACCEPTED_TRACE_OPTIONAL_ARRAY_CONTROL_KEYS as _ACCEPTED_TRACE_OPTIONAL_ARRAY_CONTROL_KEYS,  # noqa: F401 - compatibility alias for tests/internal users.
    ACCEPTED_TRACE_REQUIRED_ARRAY_CONTROL_KEYS as _ACCEPTED_TRACE_REQUIRED_ARRAY_CONTROL_KEYS,  # noqa: F401 - compatibility alias for tests/internal users.
    direct_coil_accepted_trace_array_controls_jax,
    direct_coil_accepted_trace_preconditioner_controls_jax,
    direct_coil_accepted_trace_preconditioner_policy_segment_summary,  # noqa: F401 - compatibility alias.
    direct_coil_accepted_trace_preconditioner_policy_segments,  # noqa: F401 - compatibility alias.
    direct_coil_accepted_trace_scalar_controls_jax,
    direct_coil_accepted_trace_step_controls_jax,
    direct_coil_accepted_trace_step_policy_segment_summary,  # noqa: F401 - compatibility alias.
    direct_coil_accepted_trace_step_policy_segments,
    stack_optional_trace_pytree_field as _stack_optional_trace_pytree_field,  # noqa: F401 - compatibility alias for tests/internal users.
    stack_trace_control_field as _stack_trace_control_field,  # noqa: F401 - compatibility alias for tests/internal users.
    stack_trace_nestor_axis_controls as _stack_trace_nestor_axis_controls,  # noqa: F401 - compatibility alias for tests/internal users.
    stack_trace_pytree_field as _stack_trace_pytree_field,  # noqa: F401 - compatibility alias for tests/internal users.
    trace_optional_presence_signature as _trace_optional_presence_signature,  # noqa: F401 - compatibility alias for tests/internal users.
    trace_preconditioner_policy_value as _trace_preconditioner_policy_value,  # noqa: F401 - compatibility alias for tests/internal users.
    trace_preconditioner_static_signature as _trace_preconditioner_static_signature,  # noqa: F401 - compatibility alias for tests/internal users.
    trace_static_value_shape_signature as _trace_static_value_shape_signature,  # noqa: F401 - compatibility alias for tests/internal users.
    trace_step_policy_static_signature as _trace_step_policy_static_signature,  # noqa: F401 - compatibility alias for tests/internal users.
)
from vmec_jax.solvers.free_boundary.adjoint import trace_fingerprint as _trace_fingerprint
from vmec_jax.solvers.free_boundary.adjoint import boundary_replay as _boundary_replay_module
from vmec_jax.solvers.free_boundary.adjoint import direct_coil_replay as _direct_coil_replay_module
from vmec_jax.solvers.free_boundary.adjoint import vmec_nestor as _vmec_nestor_module


class _CurrentOnlyDirectionalJVPConfig(NamedTuple):
    active: bool
    base_leaf: Any
    direction_leaf: Any
    fixed_gamma: Any
    fixed_gamma_dash: Any
    geometry_source: str


class _BranchLocalScalarDerivativeResult(NamedTuple):
    replay_values: Any
    jacobian: Any
    gradients: dict[str, Any]
    directional_values: Any
    derivative_mode: str
    directional_fast_path: str
    directional_uses_fixed_coil_geometry: bool
    current_only_geometry_source: str


__all__ = """
_finite_difference_jacobian
accepted_trace_effective_state_pre
dense_fixed_point_solve_jax dense_mode_vacuum_solve_jax dense_nonlinear_solve_jax
dense_vacuum_residual dense_vacuum_solve_jax dense_vmec_nestor_mode_solve_jax
direct_coil_accepted_trace_array_controls_jax
direct_coil_accepted_trace_branch_metadata
direct_coil_accepted_trace_controller_custom_vjp_scalars_jax
direct_coil_accepted_trace_controller_replay_plan
direct_coil_accepted_trace_controller_slot_fingerprint
direct_coil_accepted_trace_controller_slot_summary
direct_coil_accepted_trace_fingerprint direct_coil_accepted_trace_fingerprint_delta
direct_coil_accepted_trace_fingerprint_delta_summary
direct_coil_accepted_trace_preconditioner_controls_jax
direct_coil_accepted_trace_replay_graph_metadata
direct_coil_accepted_trace_replay_objective_jax
direct_coil_accepted_trace_scalar_controls_jax direct_coil_accepted_trace_status_masks
direct_coil_accepted_trace_step_controls_jax
direct_coil_accepted_trace_step_policy_segments
direct_coil_adaptive_full_loop_same_branch_gate_report
direct_coil_boundary_bnormal_rms_jax direct_coil_boundary_bsqvac_from_trace_jax
direct_coil_boundary_bsqvac_jax direct_coil_boundary_replay_context
direct_coil_boundary_replay_context_for_shape
direct_coil_branch_local_scalars_report_from_complete_fd direct_coil_complete_solve_trace
direct_coil_projected_mode_fixed_point_directional_check_jax
direct_coil_projected_mode_fixed_point_jax
direct_coil_projected_mode_fixed_point_objective_jax
direct_coil_run_free_boundary_branch_local_scalar_value_and_grad_jax
direct_coil_run_free_boundary_branch_local_scalars_value_and_jacobian_jax
direct_coil_same_branch_complete_solve_fd_report
direct_coil_same_branch_controller_scalars_custom_vjp_report
direct_coil_same_branch_physical_scalar_gate_report direct_coil_same_branch_replay_gate_report
free_boundary_adjoint_trace_replay_diagnostics free_boundary_boundary_geometry_jax
jax_visible_accepted_nonlinear_controller_directional_check_jax
jax_visible_accepted_nonlinear_controller_jax
jax_visible_accepted_only_nonlinear_controller_jax
jax_visible_masked_nonlinear_controller_directional_check_jax
jax_visible_masked_nonlinear_controller_jax
jax_visible_nonlinear_controller_directional_check_jax jax_visible_nonlinear_controller_jax
jax_visible_segmented_accepted_nonlinear_controller_jax
jax_visible_segmented_state_only_accepted_nonlinear_controller_jax
jax_visible_state_only_accepted_nonlinear_controller_jax
jax_visible_state_only_accepted_only_nonlinear_controller_jax
jax_visible_unrolled_accepted_only_nonlinear_controller_jax
jax_visible_unrolled_state_only_accepted_only_nonlinear_controller_jax
mode_matrix_from_grpmn_jax mode_matrix_matvec_from_grpmn_jax
mode_operator_vacuum_solve_jax mode_rhs_from_gsource_jax
pytree_directional_derivative_check_jax vacuum_boundary_fields_from_cylindrical_jax
vacuum_boundary_fields_from_mode_coeffs_jax vmec_analytic_terms_from_geometry_jax
vmec_nonsingular_terms_from_bexni_jax vmec_source_from_gsource_jax
""".split()

_trace_scalar = _trace_fingerprint.trace_scalar
_trace_bool = _trace_fingerprint.trace_bool
_trace_pack_size = _trace_fingerprint.trace_pack_size
_trace_array_size = _trace_fingerprint.trace_array_size
_trace_pytree_shape_signature = _trace_fingerprint.trace_pytree_shape_signature
direct_coil_accepted_trace_fingerprint = _trace_fingerprint.direct_coil_accepted_trace_fingerprint
direct_coil_accepted_trace_fingerprint_delta = _trace_fingerprint.direct_coil_accepted_trace_fingerprint_delta
direct_coil_accepted_trace_fingerprint_delta_summary = (
    _trace_fingerprint.direct_coil_accepted_trace_fingerprint_delta_summary
)


def _block_until_ready_for_timing(value: Any) -> Any:
    return _runtime_block_until_ready_for_timing(value, jax_module=jax, tree_util_module=tree_util)


def _jax_named_scope(name: str) -> Any:
    return _runtime_jax_named_scope(name, jax_module=jax, nullcontext_factory=nullcontext)


def direct_coil_same_branch_controller_scalar_custom_vjp_report(
    complete_report: dict[str, Any],
    base_params: Any,
    direction: Any,
    *,
    replay_scalar_fn: Any,
    scalar_key: str | None = None,
    eps: float = 1.0e-4,
    replay_kwargs: dict[str, Any] | None = None,
    rtol: float = 5.0e-3,
    atol: float = 1.0e-8,
    base_value_atol: float = 2.0e-3,
    compute_frozen_fd: bool = True,
) -> dict[str, Any]:
    """Compare a branch-local scalar custom VJP with complete-solve FD.

    ``complete_report`` must be returned by
    :func:`direct_coil_same_branch_complete_solve_fd_report`.  ``scalar_key``
    selects one scalar from its ``objective_values`` block; by default the
    report's primary scalar is used.  ``replay_scalar_fn(replay, base_payload)``
    receives the JAX-visible accepted-controller replay and the base complete
    solve payload, and must return the same scalar in replay coordinates.

    This is still a same-branch validation helper.  It proves that the frozen
    accepted-controller custom VJP agrees with complete-solve central
    differences when the accepted-trace fingerprint is unchanged.  It does not
    differentiate through an arbitrary adaptive host-controller branch change.
    Set ``compute_frozen_fd=False`` when the caller only needs the exact
    branch-local custom-VJP slope versus the complete-solve FD slope and wants
    to avoid two additional frozen replay evaluations.
    """

    if jax is None:  # pragma: no cover - JAX is required for custom VJP.
        raise RuntimeError("JAX is required for same-branch custom-VJP reports.")

    key = str(scalar_key or complete_report.get("primary_objective") or "objective")
    objective_values = complete_report.get("objective_values", {})
    if key not in objective_values:
        raise KeyError(f"scalar_key {key!r} not present in complete_report['objective_values']")

    replay_gate = direct_coil_same_branch_replay_gate_report(complete_report)
    branch = complete_report.get("branch_compatibility", {})
    same_branch = bool(branch.get("same_branch", False))
    base = complete_report["base"]
    traces = tuple(base["traces"])
    if not traces:
        raise ValueError("complete_report base payload contains no accepted traces")
    replay_options: dict[str, Any] = {
        "static": base["init"].static,
        "traces": traces,
        "signgs": int(base["init"].signgs),
        "state_weight": 0.0,
        "bsqvac_weight": 0.0,
        "force_weight": 0.0,
        "enforce_edge": False,
        "use_preconditioner_policy_segments": True,
    }
    if replay_kwargs:
        replay_options.update(replay_kwargs)

    def _controller_scalar(coil_params):
        return direct_coil_accepted_trace_controller_custom_vjp_scalar_jax(
            coil_params,
            traces[0]["state_pre"],
            scalar_fn=lambda replay: replay_scalar_fn(replay, base),
            **replay_options,
        )

    check = pytree_directional_derivative_check_jax(
        _controller_scalar,
        base_params,
        direction,
        eps=float(eps),
        compute_fd=bool(compute_frozen_fd),
    )
    value = float(np.asarray(check["value"], dtype=float))
    exact = float(np.asarray(check["exact_directional"], dtype=float))
    complete_values = objective_values[key]
    complete_base = float(complete_values["base"])
    complete_fd = float(complete_values["central_fd_directional"])
    abs_error = abs(exact - complete_fd)
    rel_error = abs_error / max(1.0, abs(complete_fd))
    base_abs_delta = abs(value - complete_base)
    passed = bool(
        replay_gate["passed"]
        and np.isfinite(exact)
        and np.isfinite(complete_fd)
        and abs_error <= float(atol) + float(rtol) * abs(complete_fd)
        and base_abs_delta <= float(base_value_atol)
    )
    return {
        "scalar_key": key,
        "passed": passed,
        "same_branch": same_branch,
        "replay_gate": replay_gate,
        "value": check["value"],
        "grad": check["grad"],
        "exact_directional": check["exact_directional"],
        "frozen_trace_fd_directional": check["fd_directional"],
        "complete_fd_directional": complete_fd,
        "abs_error": abs_error,
        "rel_error": rel_error,
        "base_value": value,
        "complete_base_value": complete_base,
        "base_abs_delta": base_abs_delta,
        "complete_values": complete_values,
    }


def direct_coil_same_branch_controller_scalars_custom_vjp_report(
    complete_report: dict[str, Any],
    base_params: Any,
    direction: Any,
    *,
    replay_scalar_fns: Mapping[str, Any],
    eps: float = 1.0e-4,
    replay_kwargs: dict[str, Any] | None = None,
    rtol: float | Mapping[str, float] = 5.0e-3,
    atol: float | Mapping[str, float] = 1.0e-8,
    base_value_atol: float | Mapping[str, float] = 2.0e-3,
    compute_frozen_fd: bool = False,
) -> dict[str, Any]:
    """Batch same-branch custom-VJP reports for several replay scalars.

    This helper preserves the same branch-local contract as
    :func:`direct_coil_same_branch_controller_scalar_custom_vjp_report`, but
    groups multiple scalar pullbacks through one vector-valued custom-VJP seam.
    It is intended for expensive promotion tests that compare several physical
    outputs against the same complete-solve finite-difference report.
    """

    if jax is None:  # pragma: no cover - JAX is required for custom VJP.
        raise RuntimeError("JAX is required for same-branch custom-VJP reports.")

    keys = tuple(str(key) for key in replay_scalar_fns)
    if not keys:
        raise ValueError("replay_scalar_fns must contain at least one scalar")
    objective_values = complete_report.get("objective_values", {})
    for key in keys:
        if key not in objective_values:
            raise KeyError(f"scalar_key {key!r} not present in complete_report['objective_values']")

    def _option_for(option: float | Mapping[str, float], key: str) -> float:
        if isinstance(option, Mapping):
            return float(option[key])
        return float(option)

    replay_gate = direct_coil_same_branch_replay_gate_report(complete_report)
    branch = complete_report.get("branch_compatibility", {})
    same_branch = bool(branch.get("same_branch", False))
    base = complete_report["base"]
    traces = tuple(base["traces"])
    if not traces:
        raise ValueError("complete_report base payload contains no accepted traces")
    replay_options: dict[str, Any] = {
        "static": base["init"].static,
        "traces": traces,
        "signgs": int(base["init"].signgs),
        "state_weight": 0.0,
        "bsqvac_weight": 0.0,
        "force_weight": 0.0,
        "enforce_edge": False,
        "use_preconditioner_policy_segments": True,
    }
    if replay_kwargs:
        replay_options.update(replay_kwargs)
    replay_traces = tuple(replay_options.get("traces", traces))
    if not replay_traces:
        raise ValueError("replay traces must contain at least one accepted trace")
    replay_branch_metadata = direct_coil_accepted_trace_branch_metadata(
        replay_traces,
        accept_mask=replay_options.get("accept_mask"),
        done_mask=replay_options.get("done_mask"),
        max_steps=replay_options.get("max_steps"),
        json_safe=False,
    )
    controller_slot_summary = direct_coil_accepted_trace_controller_slot_summary(replay_branch_metadata)

    scalar_fns = tuple(
        (lambda replay, fn=fn: fn(replay, base))
        for fn in replay_scalar_fns.values()
    )

    def _controller_scalars(coil_params):
        return direct_coil_accepted_trace_controller_custom_vjp_scalars_jax(
            coil_params,
            replay_traces[0]["state_pre"],
            scalar_fns=scalar_fns,
            **replay_options,
        )

    def _shifted(scale):
        return tree_util.tree_map(
            lambda value, delta: jnp.asarray(value) + float(scale) * jnp.asarray(delta),
            base_params,
            direction,
        )

    values, pullback = jax.vjp(_controller_scalars, base_params)
    basis = jnp.eye(len(keys), dtype=jnp.asarray(values).dtype)
    jacobian = _pytree_pullback_basis_jax(pullback, basis)
    exact_directionals = _pytree_batched_directional_vdot_jax(jacobian, direction, len(keys))
    if bool(compute_frozen_fd):
        step = float(eps)
        if not step > 0.0:
            raise ValueError("eps must be positive.")
        frozen_fd_directionals = (
            _controller_scalars(_shifted(step)) - _controller_scalars(_shifted(-step))
        ) / (2.0 * step)
    else:
        frozen_fd_directionals = jnp.full_like(exact_directionals, jnp.nan)

    scalar_reports: dict[str, dict[str, Any]] = {}
    passed_values: list[bool] = []
    for index, key in enumerate(keys):
        value = float(np.asarray(values[index], dtype=float))
        exact = float(np.asarray(exact_directionals[index], dtype=float))
        complete_values = objective_values[key]
        complete_base = float(complete_values["base"])
        complete_fd = float(complete_values["central_fd_directional"])
        abs_error = abs(exact - complete_fd)
        rel_error = abs_error / max(1.0, abs(complete_fd))
        base_abs_delta = abs(value - complete_base)
        key_passed = bool(
            replay_gate["passed"]
            and np.isfinite(exact)
            and np.isfinite(complete_fd)
            and abs_error <= _option_for(atol, key) + _option_for(rtol, key) * abs(complete_fd)
            and base_abs_delta <= _option_for(base_value_atol, key)
        )
        passed_values.append(key_passed)
        scalar_reports[key] = {
            "scalar_key": key,
            "passed": key_passed,
            "same_branch": same_branch,
            "replay_gate": replay_gate,
            "value": values[index],
            "exact_directional": exact_directionals[index],
            "frozen_trace_fd_directional": frozen_fd_directionals[index],
            "complete_fd_directional": complete_fd,
            "abs_error": abs_error,
            "rel_error": rel_error,
            "base_value": value,
            "complete_base_value": complete_base,
            "base_abs_delta": base_abs_delta,
            "complete_values": complete_values,
        }
    return {
        "scalar_keys": keys,
        "passed": bool(all(passed_values)),
        "same_branch": same_branch,
        "replay_gate": replay_gate,
        "replay_option_flags": {
            "use_preconditioner_policy_segments": bool(replay_options.get("use_preconditioner_policy_segments", False)),
            "use_stacked_step_controls": bool(replay_options.get("use_stacked_step_controls", False)),
            "use_accepted_only_fast_path": bool(replay_options.get("use_accepted_only_fast_path", True)),
            "include_analytic": bool(replay_options.get("include_analytic", True)),
            "include_mode_diagnostics": bool(replay_options.get("include_mode_diagnostics", False)),
            "freeze_vacuum_field": bool(replay_options.get("freeze_vacuum_field", False)),
            "freeze_freeb_bsqvac": bool(replay_options.get("freeze_freeb_bsqvac", False)),
        },
        "replay_branch_metadata": replay_branch_metadata,
        "controller_slot_summary": controller_slot_summary,
        "values": values,
        "jacobian": jacobian,
        "exact_directionals": exact_directionals,
        "frozen_trace_fd_directionals": frozen_fd_directionals,
        "scalar_reports": scalar_reports,
    }


def direct_coil_run_free_boundary_branch_local_scalar_value_and_grad_jax(
    input_path: Any | None = None,
    params: Any | None = None,
    *,
    scalar_fn: Any,
    replay_scalar_fn: Any,
    scalar_key: str | None = None,
    production_values: Mapping[str, Any] | None = None,
    replay_payload: Mapping[str, Any] | None = None,
    replay_plan: Mapping[str, Any] | None = None,
    complete_payload: Mapping[str, Any] | None = None,
    init_kwargs: dict[str, Any] | None = None,
    solve_kwargs: dict[str, Any] | None = None,
    replay_kwargs: dict[str, Any] | None = None,
    replay_ad_mode: str = "direct",
    include_trace_replay_diagnostics: bool = True,
    include_payload: bool = True,
    include_replay_graph_metadata: bool = True,
    use_replay_plan: bool = True,
    require_active_trace: bool = True,
) -> dict[str, Any]:
    """Return a production-forward branch-local scalar value and gradient.

    The forward value is evaluated from an actual direct-coil free-boundary
    solve payload, either supplied as ``complete_payload`` or obtained by
    calling :func:`direct_coil_complete_solve_trace`.  The gradient is computed
    by replaying the saved accepted branch through the stacked accepted
    controller custom-VJP path.  This is the narrow production seam currently
    validated by complete-loop finite differences: it differentiates direct
    coils through a *fixed accepted branch*, not arbitrary adaptive host
    controller branch changes.

    ``scalar_fn(payload)`` must return the production scalar from the complete
    solve payload.  Callers that already evaluated the production scalar, for
    example through :func:`direct_coil_same_branch_complete_solve_fd_report`,
    can pass ``production_values`` to avoid recomputing it.  The
    ``replay_scalar_fn(replay, payload)`` must return the same scalar from the
    JAX-visible replay dictionary.  ``replay_payload`` can be supplied to pass a
    slim context into that function, avoiding closure capture of a full complete
    solve payload during cold replay/JVP graph construction.  Set
    ``include_payload=False`` for production reports that only need scalar
    values/derivatives and should not retain the full complete-solve payload.
    Set ``include_replay_graph_metadata=False`` when a compact production
    report does not need structural replay metadata.
    """

    if jax is None:  # pragma: no cover - JAX is required for this helper.
        raise RuntimeError("JAX is required for branch-local scalar gradients.")

    ad_mode = str(replay_ad_mode).strip().lower()
    if ad_mode not in {"direct", "custom_vjp"}:
        raise ValueError("replay_ad_mode must be 'direct' or 'custom_vjp'")

    total_start = time.perf_counter()
    branch_payload = _prepare_branch_local_payload(
        input_path=input_path,
        params=params,
        complete_payload=complete_payload,
        init_kwargs=init_kwargs,
        solve_kwargs=solve_kwargs,
        require_active_trace=require_active_trace,
        complete_solve_trace_func=direct_coil_complete_solve_trace,
    )
    payload = branch_payload.payload
    params = branch_payload.params
    traces = branch_payload.traces
    init = branch_payload.init
    timings = branch_payload.timings

    values, production_values_source = _evaluate_branch_local_production_values(
        payload=payload,
        scalar_fn=scalar_fn,
        production_values=production_values,
        timings=timings,
    )
    key = _select_branch_local_scalar_key(values, scalar_key)

    replay_setup = _prepare_branch_local_replay_setup(
        init=init,
        traces=traces,
        replay_kwargs=replay_kwargs,
        replay_payload=replay_payload,
        payload=payload,
        replay_plan=replay_plan,
        use_replay_plan=bool(use_replay_plan),
        include_replay_graph_metadata=bool(include_replay_graph_metadata),
        timings=timings,
    )
    replay_options = replay_setup.replay_options
    replay_traces_for_scalars = replay_setup.replay_traces
    replay_payload_for_scalars = replay_setup.replay_payload
    replay_payload_source = replay_setup.replay_payload_source
    replay_plan_for_scalars = replay_setup.replay_plan
    replay_branch_metadata = replay_setup.replay_branch_metadata
    controller_slot_summary = replay_setup.controller_slot_summary
    graph_metadata = replay_setup.graph_metadata

    def _replay_scalar_direct(coil_params):
        replay = direct_coil_accepted_trace_controller_replay_objective_jax(
            coil_params,
            replay_traces_for_scalars[0]["state_pre"],
            replay_plan=replay_plan_for_scalars,
            **replay_options,
        )
        return replay_scalar_fn(replay, replay_payload_for_scalars)

    def _replay_scalar_custom_vjp(coil_params):
        return direct_coil_accepted_trace_controller_custom_vjp_scalar_jax(
            coil_params,
            replay_traces_for_scalars[0]["state_pre"],
            scalar_fn=lambda replay: replay_scalar_fn(replay, replay_payload_for_scalars),
            replay_plan=replay_plan_for_scalars,
            **replay_options,
        )

    _replay_scalar = _replay_scalar_direct if ad_mode == "direct" else _replay_scalar_custom_vjp

    t0 = time.perf_counter()
    replay_value, grad = jax.value_and_grad(_replay_scalar)(params)
    timings["replay_value_and_grad_dispatch_s"] = float(time.perf_counter() - t0)
    t0 = time.perf_counter()
    replay_value, grad = _block_until_ready_for_timing((replay_value, grad))
    timings["replay_value_and_grad_ready_s"] = float(time.perf_counter() - t0)
    timings["replay_value_and_grad_wall_s"] = (
        timings["replay_value_and_grad_dispatch_s"] + timings["replay_value_and_grad_ready_s"]
    )
    diagnostics = _branch_local_trace_replay_diagnostics(
        traces,
        include_trace_replay_diagnostics=bool(include_trace_replay_diagnostics),
        timings=timings,
    )
    timings["total_wall_s"] = float(time.perf_counter() - total_start)
    return {
        "contract": "production-forward branch-local run_free_boundary scalar value/gradient",
        "uses_production_forward": True,
        "differentiates_adaptive_controller": False,
        "differentiates_run_free_boundary": False,
        "differentiates_fixed_accepted_branch": True,
        "replay_ad_mode": ad_mode,
        "scalar_key": key,
        "value": float(values[key]),
        "all_values": values,
        "production_values_source": production_values_source,
        "replay_payload_source": replay_payload_source,
        "replay_value": replay_value,
        "base_abs_delta": abs(float(np.asarray(replay_value, dtype=float)) - float(values[key])),
        "grad": grad,
        "payload": payload if bool(include_payload) else None,
        "includes_payload": bool(include_payload),
        "includes_replay_graph_metadata": bool(include_replay_graph_metadata),
        "timings": timings,
        "trace_replay_diagnostics": diagnostics,
        "replay_graph_metadata": graph_metadata,
        "replay_branch_metadata": replay_branch_metadata,
        "controller_slot_summary": controller_slot_summary,
        "replay_option_flags": _branch_local_replay_option_flags(
            replay_options,
            replay_plan=replay_plan_for_scalars,
            ad_mode=ad_mode,
        ),
    }


def _current_only_directional_jvp_config(
    params: Any,
    direction_params: Any,
    *,
    current_only_coil_geometry: Any | None,
    timings: dict[str, float],
) -> _CurrentOnlyDirectionalJVPConfig:
    """Return cached-geometry current-only JVP inputs when applicable."""

    inactive = _CurrentOnlyDirectionalJVPConfig(False, None, None, None, None, "none")
    try:
        from vmec_jax.external_fields.coils_jax import CoilFieldParams, build_coil_field_geometry
    except Exception:
        return inactive
    if not isinstance(params, CoilFieldParams) or not isinstance(direction_params, CoilFieldParams):
        return inactive
    direction_dofs = np.asarray(direction_params.base_curve_dofs, dtype=float)
    if np.any(direction_dofs):
        return inactive

    if current_only_coil_geometry is None:
        t0 = time.perf_counter()
        fixed_gamma, fixed_gamma_dash, _fixed_currents = build_coil_field_geometry(params)
        timings["current_only_coil_geometry_build_wall_s"] = float(time.perf_counter() - t0)
        geometry_source = "built"
    else:
        fixed_gamma, fixed_gamma_dash = current_only_coil_geometry[:2]
        timings["current_only_coil_geometry_build_wall_s"] = 0.0
        geometry_source = "cached"
    return _CurrentOnlyDirectionalJVPConfig(
        True,
        jnp.asarray(params.base_currents),
        jnp.asarray(direction_params.base_currents),
        fixed_gamma,
        fixed_gamma_dash,
        geometry_source,
    )


def _branch_local_trace_replay_diagnostics(
    traces: tuple[Any, ...],
    *,
    include_trace_replay_diagnostics: bool,
    timings: dict[str, float],
) -> dict[str, Any]:
    """Return trace replay diagnostics and record their wall time."""

    t0 = time.perf_counter()
    if bool(include_trace_replay_diagnostics):
        diagnostics = free_boundary_adjoint_trace_replay_diagnostics(traces)
    else:
        diagnostics = {
            "contract": "fixed accepted-trace replay diagnostics only",
            "omitted": True,
            "reason": "include_trace_replay_diagnostics=False",
            "differentiates_adaptive_controller": False,
        }
    timings["trace_replay_diagnostics_wall_s"] = float(time.perf_counter() - t0)
    return diagnostics


def _branch_local_scalar_derivatives(
    *,
    params: Any,
    direction_params: Any | None,
    replay_traces: tuple[Any, ...],
    replay_plan: Mapping[str, Any] | None,
    replay_options: Mapping[str, Any],
    scalar_fn_seq: tuple[Any, ...],
    keys: tuple[str, ...],
    ad_mode: str,
    replay_scalars_fn: Any,
    current_only_coil_geometry: Any | None,
    timings: dict[str, float],
) -> _BranchLocalScalarDerivativeResult:
    """Evaluate branch-local replay scalars and their selected derivative mode."""

    jacobian = None
    gradients: dict[str, Any] = {}
    directional_values = None
    derivative_mode = "full_jacobian_vjp"
    directional_fast_path = "none"
    directional_uses_fixed_coil_geometry = False
    current_only_geometry_source = "none"

    if direction_params is not None:
        derivative_mode = "directional_jvp"
        current_jvp = _current_only_directional_jvp_config(
            params,
            direction_params,
            current_only_coil_geometry=current_only_coil_geometry,
            timings=timings,
        )
        if current_jvp.active:
            directional_fast_path = "current_only"
            directional_uses_fixed_coil_geometry = True
            current_only_geometry_source = str(current_jvp.geometry_source)
            from vmec_jax.external_fields.coils_jax import apply_stellarator_symmetry_to_currents

            def _fixed_geometry_for_currents(base_currents):
                expanded_currents = params.current_scale * apply_stellarator_symmetry_to_currents(
                    base_currents,
                    nfp=params.nfp,
                    stellsym=params.stellsym,
                )
                return current_jvp.fixed_gamma, current_jvp.fixed_gamma_dash, expanded_currents

            def _replay_scalars_current_only(base_currents):
                replay = direct_coil_accepted_trace_controller_replay_objective_jax(
                    params.with_arrays(base_currents=base_currents),
                    replay_traces[0]["state_pre"],
                    replay_plan=replay_plan,
                    coil_geometry=_fixed_geometry_for_currents(base_currents),
                    **replay_options,
                )
                return jnp.asarray([fn(replay) for fn in scalar_fn_seq])

            jvp_primal = (current_jvp.base_leaf,)
            jvp_tangent = (current_jvp.direction_leaf,)
            jvp_fn = _replay_scalars_current_only
        else:
            jvp_primal = (params,)
            jvp_tangent = (direction_params,)
            jvp_fn = replay_scalars_fn

        t0 = time.perf_counter()
        replay_values, directional_values = jax.jvp(jvp_fn, jvp_primal, jvp_tangent)
        timings["replay_jvp_dispatch_s"] = float(time.perf_counter() - t0)
        t0 = time.perf_counter()
        replay_values, directional_values = _block_until_ready_for_timing((replay_values, directional_values))
        timings["replay_jvp_ready_s"] = float(time.perf_counter() - t0)
        timings["replay_jvp_wall_s"] = timings["replay_jvp_dispatch_s"] + timings["replay_jvp_ready_s"]
        # Compatibility timing keys: no full VJP/Jacobian was built.
        timings["replay_vjp_wall_s"] = 0.0
        timings["replay_pullbacks_wall_s"] = 0.0
        timings["jacobian_stack_ready_s"] = 0.0
    else:
        t0 = time.perf_counter()
        replay_values, pullback = jax.vjp(replay_scalars_fn, params)
        timings["replay_vjp_dispatch_s"] = float(time.perf_counter() - t0)
        t0 = time.perf_counter()
        replay_values = _block_until_ready_for_timing(replay_values)
        timings["replay_vjp_ready_s"] = float(time.perf_counter() - t0)
        basis = jnp.eye(len(keys), dtype=jnp.asarray(replay_values).dtype)
        t0 = time.perf_counter()
        jacobian = _pytree_pullback_basis_jax(pullback, basis)
        timings["replay_pullbacks_dispatch_s"] = float(time.perf_counter() - t0)
        t0 = time.perf_counter()
        jacobian = _block_until_ready_for_timing(jacobian)
        timings["replay_pullbacks_ready_s"] = float(time.perf_counter() - t0)
        timings["replay_vjp_wall_s"] = timings["replay_vjp_dispatch_s"] + timings["replay_vjp_ready_s"]
        timings["replay_pullbacks_wall_s"] = (
            timings["replay_pullbacks_dispatch_s"] + timings["replay_pullbacks_ready_s"]
        )
        # Pullback readiness already materialized the full Jacobian pytree.
        # Keep the timing key for report compatibility without re-walking it.
        timings["jacobian_stack_ready_s"] = 0.0
        basis_gradients = _pytree_unstack_leading_axis_jax(jacobian, len(keys))
        gradients = {key: basis_gradients[index] for index, key in enumerate(keys)}

    return _BranchLocalScalarDerivativeResult(
        replay_values=replay_values,
        jacobian=jacobian,
        gradients=gradients,
        directional_values=directional_values,
        derivative_mode=derivative_mode,
        directional_fast_path=directional_fast_path,
        directional_uses_fixed_coil_geometry=directional_uses_fixed_coil_geometry,
        current_only_geometry_source=current_only_geometry_source,
    )


def _branch_local_scalar_report(
    *,
    derivative_result: _BranchLocalScalarDerivativeResult,
    keys: tuple[str, ...],
    all_values: Mapping[str, Any],
    production_values_source: str,
    replay_payload_source: str,
    payload: Mapping[str, Any],
    include_payload: bool,
    include_replay_graph_metadata: bool,
    traces: tuple[Any, ...],
    include_trace_replay_diagnostics: bool,
    timings: dict[str, float],
    total_start: float,
    ad_mode: str,
    replay_options: Mapping[str, Any],
    replay_plan: Mapping[str, Any] | None,
    graph_metadata: Mapping[str, Any],
    replay_branch_metadata: Mapping[str, Any],
    controller_slot_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the public branch-local scalar/Jacobian report."""

    replay_values = derivative_result.replay_values
    values = {key: float(all_values[key]) for key in keys}
    replay_value_map = {key: replay_values[index] for index, key in enumerate(keys)}
    base_abs_delta = {
        key: abs(float(np.asarray(replay_values[index], dtype=float)) - float(values[key]))
        for index, key in enumerate(keys)
    }
    directional_derivatives = (
        None
        if derivative_result.directional_values is None
        else {key: derivative_result.directional_values[index] for index, key in enumerate(keys)}
    )
    diagnostics = _branch_local_trace_replay_diagnostics(
        traces,
        include_trace_replay_diagnostics=bool(include_trace_replay_diagnostics),
        timings=timings,
    )
    timings["total_wall_s"] = float(time.perf_counter() - total_start)
    return {
        "contract": "production-forward branch-local run_free_boundary scalar values/Jacobian",
        "uses_production_forward": True,
        "differentiates_adaptive_controller": False,
        "differentiates_run_free_boundary": False,
        "differentiates_fixed_accepted_branch": True,
        "replay_ad_mode": ad_mode,
        "derivative_mode": derivative_result.derivative_mode,
        "scalar_keys": keys,
        "values": values,
        "all_values": all_values,
        "production_values_source": production_values_source,
        "replay_payload_source": replay_payload_source,
        "replay_values": replay_values,
        "replay_value_map": replay_value_map,
        "base_abs_delta": base_abs_delta,
        "max_base_abs_delta": max(base_abs_delta.values()) if base_abs_delta else 0.0,
        "jacobian": derivative_result.jacobian,
        "grads": derivative_result.gradients,
        "directional_derivatives": directional_derivatives,
        "payload": payload if bool(include_payload) else None,
        "includes_payload": bool(include_payload),
        "includes_replay_graph_metadata": bool(include_replay_graph_metadata),
        "timings": timings,
        "trace_replay_diagnostics": diagnostics,
        "replay_graph_metadata": graph_metadata,
        "replay_branch_metadata": replay_branch_metadata,
        "controller_slot_summary": controller_slot_summary,
        "replay_option_flags": _branch_local_replay_option_flags(
            replay_options,
            replay_plan=replay_plan,
            ad_mode=ad_mode,
            extra={
                "directional_jvp_fast_path": derivative_result.directional_fast_path,
                "directional_uses_fixed_coil_geometry": derivative_result.directional_uses_fixed_coil_geometry,
                "current_only_coil_geometry_source": derivative_result.current_only_geometry_source,
            },
        ),
    }


def direct_coil_run_free_boundary_branch_local_scalars_value_and_jacobian_jax(
    input_path: Any | None = None,
    params: Any | None = None,
    *,
    direction_params: Any | None = None,
    scalar_fn: Any,
    replay_scalar_fns: Mapping[str, Any],
    scalar_keys: tuple[str, ...] | list[str] | None = None,
    production_values: Mapping[str, Any] | None = None,
    replay_payload: Mapping[str, Any] | None = None,
    replay_plan: Mapping[str, Any] | None = None,
    complete_payload: Mapping[str, Any] | None = None,
    init_kwargs: dict[str, Any] | None = None,
    solve_kwargs: dict[str, Any] | None = None,
    replay_kwargs: dict[str, Any] | None = None,
    replay_ad_mode: str = "direct",
    include_trace_replay_diagnostics: bool = True,
    include_payload: bool = True,
    include_replay_graph_metadata: bool = True,
    use_replay_plan: bool = True,
    require_active_trace: bool = True,
    current_only_coil_geometry: Any | None = None,
) -> dict[str, Any]:
    """Return production values plus branch-local replay derivatives.

    The production values come from a complete direct-coil solve; derivatives
    come from the fixed accepted-branch replay, so adaptive host branch changes
    remain outside the differentiated contract.  Supplying ``direction_params``
    switches from full Jacobian VJP to directional JVP.
    """

    if jax is None:  # pragma: no cover - JAX is required for this helper.
        raise RuntimeError("JAX is required for branch-local scalar gradients.")
    if not replay_scalar_fns:
        raise ValueError("replay_scalar_fns must contain at least one scalar")
    ad_mode = str(replay_ad_mode).strip().lower()
    if ad_mode not in {"direct", "custom_vjp"}:
        raise ValueError("replay_ad_mode must be 'direct' or 'custom_vjp'")
    if direction_params is not None and ad_mode != "direct":
        raise ValueError("direction_params directional mode requires replay_ad_mode='direct'")

    total_start = time.perf_counter()
    branch_payload = _prepare_branch_local_payload(
        input_path=input_path,
        params=params,
        complete_payload=complete_payload,
        init_kwargs=init_kwargs,
        solve_kwargs=solve_kwargs,
        require_active_trace=require_active_trace,
        complete_solve_trace_func=direct_coil_complete_solve_trace,
    )
    payload = branch_payload.payload
    params = branch_payload.params
    traces = branch_payload.traces
    init = branch_payload.init
    timings = branch_payload.timings

    all_values, production_values_source = _evaluate_branch_local_production_values(
        payload=payload,
        scalar_fn=scalar_fn,
        production_values=production_values,
        timings=timings,
    )
    keys = _select_branch_local_scalar_keys(
        all_values=all_values,
        replay_scalar_fns=replay_scalar_fns,
        scalar_keys=scalar_keys,
    )

    replay_setup = _prepare_branch_local_replay_setup(
        init=init,
        traces=traces,
        replay_kwargs=replay_kwargs,
        replay_payload=replay_payload,
        payload=payload,
        replay_plan=replay_plan,
        use_replay_plan=bool(use_replay_plan),
        include_replay_graph_metadata=bool(include_replay_graph_metadata),
        timings=timings,
    )
    replay_options = replay_setup.replay_options
    replay_traces_for_scalars = replay_setup.replay_traces
    replay_payload_for_scalars = replay_setup.replay_payload
    replay_payload_source = replay_setup.replay_payload_source
    replay_plan_for_scalars = replay_setup.replay_plan
    replay_branch_metadata = replay_setup.replay_branch_metadata
    controller_slot_summary = replay_setup.controller_slot_summary
    graph_metadata = replay_setup.graph_metadata

    scalar_fn_seq = tuple(
        (lambda replay, key=key: replay_scalar_fns[key](replay, replay_payload_for_scalars)) for key in keys
    )

    def _replay_scalars_direct(coil_params):
        replay = direct_coil_accepted_trace_controller_replay_objective_jax(
            coil_params,
            replay_traces_for_scalars[0]["state_pre"],
            replay_plan=replay_plan_for_scalars,
            **replay_options,
        )
        return jnp.asarray([fn(replay) for fn in scalar_fn_seq])

    def _replay_scalars_custom_vjp(coil_params):
        return direct_coil_accepted_trace_controller_custom_vjp_scalars_jax(
            coil_params,
            replay_traces_for_scalars[0]["state_pre"],
            scalar_fns=scalar_fn_seq,
            replay_plan=replay_plan_for_scalars,
            **replay_options,
        )

    _replay_scalars = _replay_scalars_direct if ad_mode == "direct" else _replay_scalars_custom_vjp

    derivative_result = _branch_local_scalar_derivatives(
        params=params,
        direction_params=direction_params,
        replay_traces=replay_traces_for_scalars,
        replay_plan=replay_plan_for_scalars,
        replay_options=replay_options,
        scalar_fn_seq=scalar_fn_seq,
        keys=keys,
        ad_mode=ad_mode,
        replay_scalars_fn=_replay_scalars,
        current_only_coil_geometry=current_only_coil_geometry,
        timings=timings,
    )
    return _branch_local_scalar_report(
        derivative_result=derivative_result,
        keys=keys,
        all_values=all_values,
        production_values_source=production_values_source,
        replay_payload_source=replay_payload_source,
        payload=payload,
        include_payload=bool(include_payload),
        include_replay_graph_metadata=bool(include_replay_graph_metadata),
        traces=traces,
        include_trace_replay_diagnostics=bool(include_trace_replay_diagnostics),
        timings=timings,
        total_start=total_start,
        ad_mode=ad_mode,
        replay_options=replay_options,
        replay_plan=replay_plan_for_scalars,
        graph_metadata=graph_metadata,
        replay_branch_metadata=replay_branch_metadata,
        controller_slot_summary=controller_slot_summary,
    )


def direct_coil_fixed_trace_custom_vjp_objective_jax(
    params: Any,
    initial_state: Any,
    **replay_kwargs: Any,
) -> Any:
    """Return a scalar fixed-trace objective with an explicit custom VJP seam.

    This is the production-adjacent phase-2 bridge for direct-coil
    free-boundary adjoints.  The forward objective is the same fixed accepted
    trace replay used by :func:`direct_coil_accepted_trace_replay_objective_jax`.
    The custom backward rule differentiates only that frozen trace replay with
    respect to ``params``.  It deliberately does not differentiate through the
    adaptive host controller that chose accepted/rejected steps, activation
    cadence, limiters, or preconditioner policy.

    The helper is useful for call sites that need a scalar custom-VJP primitive
    while the full production ``run_free_boundary`` nonlinear controller is
    being refactored into a JAX-visible loop.  Use finite-difference trace
    fingerprint checks before promoting gradients from complete solves.
    """

    if jax is None:  # pragma: no cover - JAX is required for custom VJP.
        return direct_coil_accepted_trace_replay_objective_jax(
            params,
            initial_state,
            **replay_kwargs,
        )["objective"]

    def objective(coil_params):
        replay = direct_coil_accepted_trace_replay_objective_jax(
            coil_params,
            initial_state,
            **replay_kwargs,
        )
        return replay["objective"]

    return _scalar_custom_vjp_value_jax(objective, params)


def direct_coil_accepted_trace_controller_custom_vjp_objective_jax(
    params: Any,
    initial_state: Any,
    **replay_kwargs: Any,
) -> Any:
    """Return a scalar stacked-controller replay objective with custom VJP.

    This is the preferred phase-2 production-adjacent seam after the accepted
    trace controls have been lifted into a JAX-visible scan.  The forward path
    is :func:`direct_coil_accepted_trace_controller_replay_objective_jax`; the
    backward rule differentiates the same frozen accepted-controller replay
    with respect to coil parameters.  As with the older fixed-trace wrapper,
    adaptive host-control choices must be fingerprint-gated before complete
    solve finite differences are promoted.
    """

    if jax is None:  # pragma: no cover - JAX is required for custom VJP.
        return direct_coil_accepted_trace_controller_replay_objective_jax(
            params,
            initial_state,
            **replay_kwargs,
        )["objective"]

    def objective(coil_params):
        replay = direct_coil_accepted_trace_controller_replay_objective_jax(
            coil_params,
            initial_state,
            **replay_kwargs,
        )
        return replay["objective"]

    return _scalar_custom_vjp_value_jax(objective, params)


def direct_coil_accepted_trace_controller_custom_vjp_scalar_jax(
    params: Any,
    initial_state: Any,
    *,
    scalar_fn: Any,
    **replay_kwargs: Any,
) -> Any:
    """Return a scalar of accepted-controller replay with a custom VJP seam.

    ``scalar_fn`` is called with the replay dictionary returned by
    :func:`direct_coil_accepted_trace_controller_replay_objective_jax`; it can
    extract the replayed final state, objective history, or vacuum terms and
    return any scalar JAX expression.  The backward rule differentiates the
    same frozen accepted-controller replay with respect to coil parameters.

    This is a branch-local production-adjacent helper.  It deliberately does
    not differentiate the host policy that selected accepted/rejected steps,
    reset points, limiters, activation cadence, or preconditioner dispatch.
    Complete-solve promotion must therefore be guarded by accepted-trace
    fingerprints before comparing against finite differences.
    """

    if jax is None:  # pragma: no cover - JAX is required for custom VJP.
        replay = direct_coil_accepted_trace_controller_replay_objective_jax(
            params,
            initial_state,
            **replay_kwargs,
        )
        return scalar_fn(replay)

    def objective(coil_params):
        replay = direct_coil_accepted_trace_controller_replay_objective_jax(
            coil_params,
            initial_state,
            **replay_kwargs,
        )
        return scalar_fn(replay)

    return _scalar_custom_vjp_value_jax(objective, params)


def direct_coil_accepted_trace_controller_custom_vjp_scalars_jax(
    params: Any,
    initial_state: Any,
    *,
    scalar_fns: Any,
    **replay_kwargs: Any,
) -> Any:
    """Return several accepted-controller replay scalars with one custom VJP.

    The output is a one-dimensional JAX array whose entries are the scalars
    returned by ``scalar_fns``.  The backward rule differentiates the same
    frozen accepted-controller replay and supports vector cotangents, so tests
    can validate several physical scalar pullbacks against one complete-solve
    finite-difference branch report.
    """

    scalar_fn_seq = tuple(scalar_fns)
    if not scalar_fn_seq:
        raise ValueError("scalar_fns must contain at least one scalar function")
    if jax is None:  # pragma: no cover - JAX is required for custom VJP.
        replay = direct_coil_accepted_trace_controller_replay_objective_jax(
            params,
            initial_state,
            **replay_kwargs,
        )
        return jnp.asarray([fn(replay) for fn in scalar_fn_seq])

    def objective(coil_params):
        replay = direct_coil_accepted_trace_controller_replay_objective_jax(
            coil_params,
            initial_state,
            **replay_kwargs,
        )
        return jnp.asarray([fn(replay) for fn in scalar_fn_seq])

    return _vector_custom_vjp_value_jax(objective, params)


def direct_coil_accepted_trace_directional_check_jax(
    params: Any,
    direction: Any,
    initial_state: Any,
    *,
    eps: float = 1.0e-4,
    compute_fd: bool = True,
    **replay_kwargs: Any,
) -> dict[str, Any]:
    """Validate accepted-trace replay coil gradients by central FD.

    This wraps :func:`direct_coil_accepted_trace_replay_objective_jax` with the
    common AD-vs-central-FD contract used throughout the phase-2 free-boundary
    adjoint ladder.  The differentiated path includes direct-coil sampling,
    accepted-boundary geometry resampling, JAX NESTOR replay, and strict VMEC
    accepted updates under fixed production trace controls.

    The helper is production-adjacent but still intentionally scoped: the
    adaptive host controller that created the accepted traces is fixed data, so
    this is not yet a full custom VJP for :func:`vmec_jax.driver.run_free_boundary`.
    """

    def objective(coil_params):
        replay = direct_coil_accepted_trace_replay_objective_jax(
            coil_params,
            initial_state,
            **replay_kwargs,
        )
        return replay["objective"]

    check = pytree_directional_derivative_check_jax(
        objective,
        params,
        direction,
        eps=eps,
        compute_fd=compute_fd,
    )
    replay = direct_coil_accepted_trace_replay_objective_jax(
        params,
        initial_state,
        **replay_kwargs,
    )
    return {
        **check,
        "replay": replay,
        "objective_components": replay["objective_components"],
    }


def direct_coil_accepted_trace_controller_directional_check_jax(
    params: Any,
    direction: Any,
    initial_state: Any,
    *,
    eps: float = 1.0e-4,
    compute_fd: bool = True,
    **replay_kwargs: Any,
) -> dict[str, Any]:
    """Validate stacked accepted-controller replay gradients by central FD.

    This is the scan-controller counterpart to
    :func:`direct_coil_accepted_trace_directional_check_jax`.  It validates the
    differentiated path that carries accepted/rejected masks plus stacked
    scalar, velocity-history, and preconditioner controls through
    :func:`jax_visible_accepted_nonlinear_controller_jax`.  Passing
    ``use_preconditioner_policy_segments=True`` in ``replay_kwargs`` validates
    the segmented static-policy controller path used as the next staging/fusion
    rung for longer accepted traces.
    """

    def objective(coil_params):
        replay = direct_coil_accepted_trace_controller_replay_objective_jax(
            coil_params,
            initial_state,
            **replay_kwargs,
        )
        return replay["objective"]

    check = pytree_directional_derivative_check_jax(
        objective,
        params,
        direction,
        eps=eps,
        compute_fd=compute_fd,
    )
    replay = direct_coil_accepted_trace_controller_replay_objective_jax(
        params,
        initial_state,
        **replay_kwargs,
    )
    return {
        **check,
        "replay": replay,
        "objective_components": replay["objective_components"],
    }


def direct_coil_projected_mode_fixed_point_directional_check_jax(
    params: Any,
    direction: Any,
    initial_state: Any,
    *,
    eps: float = 1.0e-4,
    **objective_kwargs: Any,
) -> dict[str, Any]:
    """Validate projected-mode fixed-point coil gradients by central FD."""

    return _direct_coil_projected_mode_fixed_point_directional_check_jax_impl(
        params,
        direction,
        initial_state,
        directional_check_func=pytree_directional_derivative_check_jax,
        eps=eps,
        **objective_kwargs,
    )


_COMPAT_FORWARD_MODULES = (
    _boundary_replay_module,
    _direct_coil_replay_module,
    _vmec_nestor_module,
)
_FACADE_MODULE = sys.modules[__name__]


class _FreeBoundaryAdjointFacadeModule(types.ModuleType):
    """Forward root-facade monkeypatches to moved adjoint implementations."""

    def __setattr__(self, name, value):
        if not (name.startswith("__") and name.endswith("__")):
            for module in (_FACADE_MODULE, *_COMPAT_FORWARD_MODULES):
                if module is self:
                    continue
                if hasattr(module, name):
                    setattr(module, name, value)
        super().__setattr__(name, value)


def install_compat_facade_module(module_name: str) -> None:
    """Install compatibility monkeypatch forwarding for an imported facade."""

    sys.modules[module_name].__class__ = _FreeBoundaryAdjointFacadeModule


install_compat_facade_module(__name__)
