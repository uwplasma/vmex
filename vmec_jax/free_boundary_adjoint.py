"""Adjoint scaffolding for free-boundary vacuum solves.

Phase 1 intentionally keeps this module small and explicit.  It validates the
linear-solve differentiation contract that the production NESTOR replacement
will need: solve the primal system in the forward pass and use transpose solves
in the backward pass rather than differentiating through an iterative solver.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from contextlib import nullcontext
from typing import Any

import numpy as np

from vmec_jax._compat import jax, jnp, tree_util

from .free_boundary_adjoint_controller import (
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
from .solvers.free_boundary.adjoint.objectives import (
    accepted_controller_replay_result as _accepted_controller_replay_result,
    static_weight_is_zero as _static_weight_is_zero,  # noqa: F401 - compatibility alias for tests/internal users.
    tree_weighted_half_norm as _tree_weighted_half_norm,
    weighted_half_norm as _weighted_half_norm,
)
from .solvers.free_boundary.adjoint.pytrees import (
    pytree_batched_directional_vdot_jax as _pytree_batched_directional_vdot_jax,
    pytree_pullback_basis_jax as _pytree_pullback_basis_jax,
    pytree_unstack_leading_axis_jax as _pytree_unstack_leading_axis_jax,
)
from .solvers.free_boundary.adjoint.trace_controls import (
    _accepted_trace_reset_flags,
    accepted_trace_effective_controller_masks as _accepted_trace_effective_controller_masks,  # noqa: F401 - compatibility alias.
    accepted_trace_segment_is_unconditionally_accepted as _accepted_trace_segment_is_unconditionally_accepted,  # noqa: F401 - compatibility alias.
    direct_coil_accepted_trace_controller_controls_jax,  # noqa: F401 - compatibility alias.
    direct_coil_accepted_trace_status_masks,
)
from .solvers.free_boundary.adjoint.trace_metadata import (
    _compact_segment_summaries,  # noqa: F401 - compatibility alias for tests/internal users.
    _fingerprint_has_rejected_controller_slot,  # noqa: F401 - compatibility alias for tests/internal users.
    _json_safe_fingerprint_value,
    _unique_shape_list,  # noqa: F401 - compatibility alias for tests/internal users.
    direct_coil_accepted_trace_controller_slot_summary,
)
from .solvers.free_boundary.adjoint.gate_reports import (
    direct_coil_branch_local_scalars_report_from_complete_fd,
    direct_coil_adaptive_full_loop_same_branch_gate_report,
    direct_coil_same_branch_physical_scalar_gate_report,
    direct_coil_same_branch_replay_gate_report,
)
from .solvers.free_boundary.adjoint.branch_metadata import (
    direct_coil_accepted_trace_branch_metadata,
    direct_coil_accepted_trace_replay_graph_metadata,
)
from .solvers.free_boundary.adjoint.custom_vjp import (
    scalar_custom_vjp_value_jax as _scalar_custom_vjp_value_jax,
    vector_custom_vjp_value_jax as _vector_custom_vjp_value_jax,
)
from .solvers.free_boundary.adjoint.dense import (
    dense_fixed_point_solve_jax,
    dense_nonlinear_solve_jax,
    dense_vacuum_residual,
    dense_vacuum_solve_jax,
    finite_difference_jacobian as _finite_difference_jacobian,
)
from .solvers.free_boundary.adjoint.mode_operator import (
    mode_matrix_from_grpmn_jax,
    mode_matrix_matvec_from_grpmn_jax,
    mode_operator_vacuum_solve_jax,
    mode_rhs_from_gsource_jax,
    vmec_source_from_gsource_jax,
)
from .solvers.free_boundary.adjoint.boundary_replay import (
    direct_coil_boundary_bnormal_rms_jax,
    free_boundary_boundary_geometry_jax,
    vacuum_boundary_fields_from_cylindrical_jax,
    vacuum_boundary_fields_from_mode_coeffs_jax,
)
from .solvers.free_boundary.adjoint.replay_context import (
    direct_coil_boundary_replay_context,
    direct_coil_boundary_replay_context_for_shape,
    direct_coil_trace_boundary_shape as _direct_coil_trace_boundary_shape,
    direct_coil_trace_vacuum_field_override as _direct_coil_trace_vacuum_field_override,
    with_jax_nonsingular_replay_tables as _with_jax_nonsingular_replay_tables,  # noqa: F401 - compatibility alias.
)
from .solvers.free_boundary.adjoint.runtime import (
    block_until_ready_for_timing as _runtime_block_until_ready_for_timing,
    jax_named_scope as _runtime_jax_named_scope,
)
from .solvers.free_boundary.adjoint.replay_plan import (
    accepted_step_policy_layout_for_complete_payload as _accepted_step_policy_layout_for_complete_payload,  # noqa: F401 - compatibility alias for tests/internal users.
    accepted_step_policy_signature_for_complete_payload as _accepted_step_policy_signature_for_complete_payload,  # noqa: F401 - compatibility alias for tests/internal users.
    accepted_step_policy_summary_for_complete_payload as _accepted_step_policy_summary_for_complete_payload,  # noqa: F401 - compatibility alias for tests/internal users.
    complete_solve_objective_values as _complete_solve_objective_values,
    direct_coil_accepted_trace_controller_replay_plan,
    direct_coil_boundary_replay_contexts_by_shape as _direct_coil_boundary_replay_contexts_by_shape,  # noqa: F401 - compatibility alias.
    extract_adjoint_step_trace as _extract_adjoint_step_trace,
    slice_replay_controls as _slice_replay_controls,  # noqa: F401 - compatibility alias.
    stackability_probe as _stackability_probe,
)
from .solvers.free_boundary.adjoint.trace_stack import (
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
from .solvers.free_boundary.adjoint import trace_fingerprint as _trace_fingerprint

__all__ = [
    "_finite_difference_jacobian",
    "dense_fixed_point_solve_jax",
    "dense_nonlinear_solve_jax",
    "dense_vacuum_residual",
    "dense_vacuum_solve_jax",
    "mode_matrix_from_grpmn_jax",
    "mode_matrix_matvec_from_grpmn_jax",
    "mode_operator_vacuum_solve_jax",
    "mode_rhs_from_gsource_jax",
    "vmec_source_from_gsource_jax",
    "direct_coil_boundary_bnormal_rms_jax",
    "direct_coil_boundary_replay_context",
    "direct_coil_boundary_replay_context_for_shape",
    "free_boundary_boundary_geometry_jax",
    "vacuum_boundary_fields_from_cylindrical_jax",
    "vacuum_boundary_fields_from_mode_coeffs_jax",
    "direct_coil_accepted_trace_branch_metadata",
    "direct_coil_accepted_trace_controller_slot_summary",
    "direct_coil_accepted_trace_controller_custom_vjp_scalars_jax",
    "direct_coil_accepted_trace_controller_replay_plan",
    "direct_coil_accepted_trace_fingerprint",
    "direct_coil_accepted_trace_fingerprint_delta",
    "direct_coil_accepted_trace_fingerprint_delta_summary",
    "direct_coil_accepted_trace_replay_graph_metadata",
    "direct_coil_accepted_trace_status_masks",
    "direct_coil_accepted_trace_step_controls_jax",
    "direct_coil_accepted_trace_step_policy_segments",
    "direct_coil_boundary_replay_context_for_shape",
    "direct_coil_adaptive_full_loop_same_branch_gate_report",
    "direct_coil_run_free_boundary_branch_local_scalar_value_and_grad_jax",
    "direct_coil_run_free_boundary_branch_local_scalars_value_and_jacobian_jax",
    "direct_coil_branch_local_scalars_report_from_complete_fd",
    "direct_coil_same_branch_physical_scalar_gate_report",
    "direct_coil_same_branch_replay_gate_report",
    "direct_coil_same_branch_controller_scalars_custom_vjp_report",
    "free_boundary_adjoint_trace_replay_diagnostics",
    "jax_visible_accepted_only_nonlinear_controller_jax",
    "jax_visible_accepted_nonlinear_controller_directional_check_jax",
    "jax_visible_accepted_nonlinear_controller_jax",
    "jax_visible_masked_nonlinear_controller_directional_check_jax",
    "jax_visible_masked_nonlinear_controller_jax",
    "jax_visible_nonlinear_controller_directional_check_jax",
    "jax_visible_nonlinear_controller_jax",
    "jax_visible_segmented_accepted_nonlinear_controller_jax",
    "jax_visible_segmented_state_only_accepted_nonlinear_controller_jax",
    "jax_visible_state_only_accepted_nonlinear_controller_jax",
    "jax_visible_state_only_accepted_only_nonlinear_controller_jax",
    "jax_visible_unrolled_state_only_accepted_only_nonlinear_controller_jax",
    "jax_visible_unrolled_accepted_only_nonlinear_controller_jax",
    "pytree_directional_derivative_check_jax",
]

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


def vmec_nonsingular_terms_from_bexni_jax(
    *,
    R: Any,
    Z: Any,
    Ru: Any,
    Zu: Any,
    Rv: Any,
    Zv: Any,
    ruu: Any,
    ruv: Any,
    rvv: Any,
    zuu: Any,
    zuv: Any,
    zvv: Any,
    bexni: Any,
    basis: dict[str, Any],
    tables: dict[str, Any],
    signgs: int,
    nvper: int,
) -> tuple[Any, Any]:
    """JAX VMEC/NESTOR nonsingular Green-function source/matrix assembly.

    This mirrors the core low-resolution algebra in
    ``free_boundary._vmec_nonsingular_terms_from_bexni`` once the boundary
    geometry and second derivatives are already sampled on the full VMEC
    angular grid.  The trigonometric and tangent tables are treated as static
    constants, while the boundary geometry and external normal-field source are
    differentiable JAX inputs.

    The helper is intentionally explicit rather than performance-tuned.  It is
    the validation bridge between the phase-1 dense mode-space adjoint scaffold
    and the future production NESTOR operator: tests can now differentiate
    through Green-kernel source assembly, mode projection, matrix assembly, and
    the implicit dense solve without crossing to NumPy.
    """

    R2 = jnp.asarray(R)
    Z2 = jnp.asarray(Z)
    Ru2 = jnp.asarray(Ru)
    Zu2 = jnp.asarray(Zu)
    Rv2 = jnp.asarray(Rv)
    Zv2 = jnp.asarray(Zv)
    ruu2 = jnp.asarray(ruu)
    ruv2 = jnp.asarray(ruv)
    rvv2 = jnp.asarray(rvv)
    zuu2 = jnp.asarray(zuu)
    zuv2 = jnp.asarray(zuv)
    zvv2 = jnp.asarray(zvv)
    if R2.ndim != 2:
        raise ValueError("R must be a 2D full-grid array")
    for name, arr in (
        ("Z", Z2),
        ("Ru", Ru2),
        ("Zu", Zu2),
        ("Rv", Rv2),
        ("Zv", Zv2),
        ("ruu", ruu2),
        ("ruv", ruv2),
        ("rvv", rvv2),
        ("zuu", zuu2),
        ("zuv", zuv2),
        ("zvv", zvv2),
    ):
        if arr.shape != R2.shape:
            raise ValueError(f"{name} must match R shape")

    nu = int(R2.shape[0])
    nv = int(R2.shape[1])
    nuv_full = int(nu * nv)
    nuv3 = int(basis["nuv3"])
    mf = int(basis["mf"])
    nf = int(basis["nf"])
    mnpd = int(basis["mnpd"])
    lasym = bool(basis["lasym"])
    mnpd2 = int(basis["mnpd2"])
    onp = float(basis["onp"])
    sign = float(int(signgs))
    nvper = max(1, int(nvper))
    if int(basis.get("nu_full", nu)) != nu:
        raise ValueError("R grid must use basis['nu_full'] rows")

    Rf = jnp.reshape(R2, (-1,))
    Zf = jnp.reshape(Z2, (-1,))
    R_uf = jnp.reshape(Ru2, (-1,))
    Z_uf = jnp.reshape(Zu2, (-1,))
    R_vf = jnp.reshape(Rv2, (-1,))
    Z_vf = jnp.reshape(Zv2, (-1,))
    ruuf = jnp.reshape(ruu2, (-1,))
    ruvf = jnp.reshape(ruv2, (-1,))
    rvvf = jnp.reshape(rvv2, (-1,))
    zuuf = jnp.reshape(zuu2, (-1,))
    zuvf = jnp.reshape(zuv2, (-1,))
    zvvf = jnp.reshape(zvv2, (-1,))

    snr = sign * Rf * Z_uf
    snv = sign * (R_uf * Z_vf - R_vf * Z_uf)
    snz = -sign * Rf * R_uf
    drv = -(Rf * snr + Zf * snz)
    guu_b = R_uf * R_uf + Z_uf * Z_uf
    guv_b = (R_uf * R_vf + Z_uf * Z_vf) * onp * 2.0
    gvv_b = (R_vf * R_vf + Z_vf * Z_vf + Rf * Rf) * (onp * onp)
    auu = 0.5 * (snr * ruuf + snz * zuuf)
    auv = (snr * ruvf + snv * R_uf + snz * zuvf) * onp
    avv = (snv * R_vf + 0.5 * (snr * (rvvf - Rf) + snz * zvvf)) * (onp * onp)
    rzb2 = Rf * Rf + Zf * Zf

    idx_all = jnp.asarray(tables["idx_all"], dtype=jnp.int32)
    tanu = jnp.asarray(tables["tanu"])
    tanv = jnp.asarray(tables["tanv"])
    cosuv = jnp.asarray(tables["cosuv"])
    sinuv = jnp.asarray(tables["sinuv"])
    cosper = jnp.asarray(tables["cosper"])
    sinper = jnp.asarray(tables["sinper"])
    cosv_tab = jnp.asarray(tables["cosv_tab"])
    sinv_tab = jnp.asarray(tables["sinv_tab"])
    cosui = jnp.asarray(tables["cosui"])
    sinui = jnp.asarray(tables["sinui"])
    nu_fourp = int(cosui.shape[1])
    if nu_fourp <= 0:
        raise ValueError("invalid nonsingular table shape")

    rcosuv = Rf * cosuv
    rsinuv = Rf * sinuv
    bex = jnp.reshape(jnp.asarray(bexni), (-1,))
    if int(bex.shape[0]) < nuv3:
        raise ValueError("bexni must contain at least basis['nuv3'] entries")
    bex = bex[:nuv3]

    if "iuv_grid" in tables:
        iuv_grid = jnp.asarray(tables["iuv_grid"], dtype=jnp.int32)
        iref_grid = jnp.asarray(tables["iref_grid"], dtype=jnp.int32)
        cosv_modes = jnp.asarray(tables["cosv_modes"])
        sinv_modes = jnp.asarray(tables["sinv_modes"])
        idx_p_flat = jnp.asarray(tables["idx_p_flat"], dtype=jnp.int32)
        idx_m_negative = jnp.asarray(tables["idx_m_negative"], dtype=jnp.int32)
        negative_positions_arr = jnp.asarray(tables["negative_positions"], dtype=jnp.int32)
        sinm_sym = jnp.asarray(tables["sinm_sym"])
        cosm_sym = jnp.asarray(tables["cosm_sym"])
        sinm_asym = jnp.asarray(tables["sinm_asym"])
        cosm_asym = jnp.asarray(tables["cosm_asym"])
    else:
        imirr_full = jnp.asarray(basis["imirr_full"], dtype=jnp.int32)
        idx_u = jnp.arange(nu_fourp, dtype=jnp.int32)
        idx_v = jnp.arange(nv, dtype=jnp.int32)
        iuv_grid = idx_u[:, None] * int(nv) + idx_v[None, :]
        iref_grid = imirr_full[iuv_grid]
        cosv_modes = 0.5 * onp * cosv_tab[: nf + 1, :]
        sinv_modes = 0.5 * onp * sinv_tab[: nf + 1, :]
        mf1 = int(mf + 1)
        idx_p_rows: list[int] = []
        idx_m_rows: list[int] = []
        negative_positions: list[int] = []
        flat_pos = 0
        for m in range(mf + 1):
            for n in range(nf + 1):
                idx_p_rows.append(int(m + (n + nf) * mf1))
                if n != 0 and m != 0:
                    idx_m_rows.append(int(m + ((-n) + nf) * mf1))
                    negative_positions.append(int(flat_pos))
                flat_pos += 1
        idx_p_flat = jnp.asarray(idx_p_rows, dtype=jnp.int32)
        idx_m_negative = jnp.asarray(idx_m_rows, dtype=jnp.int32)
        negative_positions_arr = jnp.asarray(negative_positions, dtype=jnp.int32)
        sinm_sym = sinui[: mf + 1, :]
        cosm_sym = -cosui[: mf + 1, :]
        sinm_asym = cosui[: mf + 1, :]
        cosm_asym = sinui[: mf + 1, :]

    gstore = jnp.zeros((nuv_full,), dtype=Rf.dtype)
    grpmn = jnp.zeros((mnpd2, nuv3), dtype=Rf.dtype)

    def _ip_body(carry: tuple[Any, Any], ip: Any) -> tuple[tuple[Any, Any], None]:
        gstore_acc, grpmn_acc = carry
        ip = jnp.asarray(ip, dtype=jnp.int32)
        xip = rcosuv[ip]
        yip = rsinuv[ip]
        ivoff = jnp.asarray(nuv_full, dtype=jnp.int32) - ip
        iskip = ip // jnp.asarray(max(1, nv), dtype=jnp.int32)
        iuoff = jnp.asarray(nuv_full, dtype=jnp.int32) - jnp.asarray(nv, dtype=jnp.int32) * iskip
        gsave = rzb2[ip] + rzb2 - 2.0 * Zf[ip] * Zf
        dsave = drv[ip] + Zf * snz[ip]
        delgr = jnp.zeros((nuv_full,), dtype=Rf.dtype)
        delgrp = jnp.zeros((nuv_full,), dtype=Rf.dtype)

        for kp in range(nvper):
            xper = xip * cosper[kp] - yip * sinper[kp]
            yper = yip * cosper[kp] + xip * sinper[kp]
            sxsave = (snr[ip] * xper - snv[ip] * yper) / Rf[ip]
            sysave = (snr[ip] * yper + snv[ip] * xper) / Rf[ip]
            base = gsave - 2.0 * (xper * rcosuv + yper * rsinuv)
            deriv_num = rcosuv * sxsave + rsinuv * sysave + dsave

            if kp == 0 or nv == 1:
                tidx_u = idx_all + iuoff
                ivoff_k = ivoff + jnp.asarray(2 * nu * kp if nv == 1 else 0, dtype=jnp.int32)
                tidx_v = idx_all + ivoff_k
                tanu_use = tanu[tidx_u]
                tanv_use = tanv[tidx_v]
                ga1 = tanu_use * (guu_b[ip] * tanu_use + guv_b[ip] * tanv_use) + gvv_b[ip] * tanv_use * tanv_use
                ga2 = tanu_use * (auu[ip] * tanu_use + auv[ip] * tanv_use) + avv[ip] * tanv_use * tanv_use
                ga2 = ga2 / ga1
                ga1s = 1.0 / jnp.sqrt(ga1)
                mask = idx_all != ip if kp == 0 else jnp.ones((nuv_full,), dtype=bool)
                safe_base = jnp.where(mask, base, 1.0)
                ftemp = 1.0 / safe_base
                htemp = jnp.sqrt(ftemp)
                deriv = ftemp * htemp * deriv_num
                delgr = delgr + jnp.where(mask, htemp - ga1s, 0.0)
                delgrp = delgrp + jnp.where(mask, deriv - ga2 * ga1s, 0.0)
            else:
                ftemp = 1.0 / base
                htemp = jnp.sqrt(ftemp)
                delgr = delgr + htemp
                delgrp = delgrp + ftemp * htemp * deriv_num

        if nv == 1 and nvper > 1:
            scale = 1.0 / float(nvper)
            delgr = delgr * scale
            delgrp = delgrp * scale

        gstore_next = gstore_acc + bex[ip] * delgr
        del_iuv = delgrp[iuv_grid]
        del_ref = delgrp[iref_grid]
        ka_grid = del_iuv - del_ref
        g1_sym = jnp.einsum("uv,fv->uf", ka_grid, cosv_modes)
        g2_sym = jnp.einsum("uv,fv->uf", ka_grid, sinv_modes)

        gcos = jnp.einsum("mu,uf->mf", sinm_sym, g1_sym)
        gsin = jnp.einsum("mu,uf->mf", cosm_sym, g2_sym)
        total_plus = jnp.reshape(gcos + gsin, (-1,))
        total_minus = jnp.reshape(gcos - gsin, (-1,))
        cols_p = jnp.full_like(idx_p_flat, ip)
        cols_m = jnp.full_like(idx_m_negative, ip)
        grpmn_next = grpmn_acc.at[(idx_p_flat, cols_p)].add(total_plus)
        grpmn_next = grpmn_next.at[(idx_m_negative, cols_m)].add(total_minus[negative_positions_arr])

        if lasym:
            ks_grid = del_iuv + del_ref
            g1_asym = jnp.einsum("uv,fv->uf", ks_grid, cosv_modes)
            g2_asym = jnp.einsum("uv,fv->uf", ks_grid, sinv_modes)
            gcos_asym = jnp.einsum("mu,uf->mf", sinm_asym, g1_asym)
            gsin_asym = jnp.einsum("mu,uf->mf", cosm_asym, g2_asym)
            total_plus_asym = jnp.reshape(gcos_asym + gsin_asym, (-1,))
            total_minus_asym = jnp.reshape(gcos_asym - gsin_asym, (-1,))
            row_off = int(mnpd)
            grpmn_next = grpmn_next.at[(row_off + idx_p_flat, cols_p)].add(total_plus_asym)
            grpmn_next = grpmn_next.at[(row_off + idx_m_negative, cols_m)].add(
                total_minus_asym[negative_positions_arr]
            )

        return (gstore_next, grpmn_next), None

    if bool(tables.get("use_ip_scan", True)):
        (gstore, grpmn), _ = jax.lax.scan(
            _ip_body,
            (gstore, grpmn),
            jnp.arange(nuv3, dtype=jnp.int32),
        )
    else:
        for ip in range(nuv3):
            (gstore, grpmn), _ = _ip_body((gstore, grpmn), jnp.asarray(ip, dtype=jnp.int32))

    return gstore, grpmn


def vmec_analytic_terms_from_geometry_jax(
    *,
    R: Any,
    Ru: Any,
    Rv: Any,
    Zu: Any,
    Zv: Any,
    ruu: Any,
    ruv: Any,
    rvv: Any,
    zuu: Any,
    zuv: Any,
    zvv: Any,
    bexni: Any,
    basis: dict[str, Any],
    signgs: int,
) -> tuple[Any, Any]:
    """JAX VMEC/NESTOR analytic singular-source terms from ``analyt.f``.

    This helper ports the analytic/singular mode-source contribution used by
    the VMEC-like free-boundary bridge when first and second boundary
    derivatives are already available on the active VMEC angular grid.  The
    recurrence coefficients and mode tables are static, while the boundary
    metric/curvature channels and external source are differentiable.
    """

    R_arr = jnp.asarray(R)
    Ru_arr = jnp.asarray(Ru)
    Rv_arr = jnp.asarray(Rv)
    Zu_arr = jnp.asarray(Zu)
    Zv_arr = jnp.asarray(Zv)
    ruu_arr = jnp.asarray(ruu)
    ruv_arr = jnp.asarray(ruv)
    rvv_arr = jnp.asarray(rvv)
    zuu_arr = jnp.asarray(zuu)
    zuv_arr = jnp.asarray(zuv)
    zvv_arr = jnp.asarray(zvv)
    if R_arr.ndim != 2:
        raise ValueError("R must be a 2D active-grid array")
    for name, arr in (
        ("Ru", Ru_arr),
        ("Rv", Rv_arr),
        ("Zu", Zu_arr),
        ("Zv", Zv_arr),
        ("ruu", ruu_arr),
        ("ruv", ruv_arr),
        ("rvv", rvv_arr),
        ("zuu", zuu_arr),
        ("zuv", zuv_arr),
        ("zvv", zvv_arr),
    ):
        if arr.shape != R_arr.shape:
            raise ValueError(f"{name} must match R shape")

    mnpd = int(basis["mnpd"])
    lasym = bool(basis["lasym"])
    mf = int(basis["mf"])
    nf = int(basis["nf"])
    onp = float(basis["onp"])
    sign = float(int(signgs))
    npts = int(jnp.size(R_arr))
    theta = jnp.asarray(basis["theta"])
    zeta = jnp.asarray(basis["zeta"])
    if int(theta.size) != npts or int(zeta.size) != npts:
        raise ValueError("basis theta/zeta size must match active grid")
    bex = jnp.reshape(jnp.asarray(bexni), (-1,))
    if int(bex.shape[0]) < npts:
        raise ValueError("bexni must contain at least one active-grid value per point")
    bex = bex[:npts]

    Rf = jnp.reshape(R_arr, (-1,))
    Ruf = jnp.reshape(Ru_arr, (-1,))
    Rvf = jnp.reshape(Rv_arr, (-1,))
    Zuf = jnp.reshape(Zu_arr, (-1,))
    Zvf = jnp.reshape(Zv_arr, (-1,))
    ruuf = jnp.reshape(ruu_arr, (-1,))
    ruvf = jnp.reshape(ruv_arr, (-1,))
    rvvf = jnp.reshape(rvv_arr, (-1,))
    zuuf = jnp.reshape(zuu_arr, (-1,))
    zuvf = jnp.reshape(zuv_arr, (-1,))
    zvvf = jnp.reshape(zvv_arr, (-1,))

    guu_b = Ruf * Ruf + Zuf * Zuf
    guv_b = (Ruf * Rvf + Zuf * Zvf) * (2.0 * onp)
    gvv_b = (Rvf * Rvf + Zvf * Zvf + Rf * Rf) * (onp * onp)
    adp = guu_b + guv_b + gvv_b
    adm = guu_b - guv_b + gvv_b
    cma = gvv_b - guu_b
    sqrtc = 2.0 * jnp.sqrt(gvv_b)
    sqrta = 2.0 * jnp.sqrt(guu_b)
    sqad1 = jnp.sqrt(adp)
    sqad2 = jnp.sqrt(adm)
    tlp = (1.0 / sqad1) * jnp.log((sqad1 * sqrtc + adp + cma) / (sqad1 * sqrta - adp + cma))
    tlm = (1.0 / sqad2) * jnp.log((sqad2 * sqrtc + adm + cma) / (sqad2 * sqrta - adm + cma))
    tlp_prev = jnp.zeros_like(tlp)
    tlm_prev = jnp.zeros_like(tlm)
    tlpm = tlp + tlm

    snr = sign * Rf * Zuf
    snv = sign * (Ruf * Zvf - Rvf * Zuf)
    snz = -sign * Rf * Ruf
    auu = 0.5 * (snr * ruuf + snz * zuuf)
    auv = (snr * ruvf + snv * Ruf + snz * zuvf) * onp
    avv = (snv * Rvf + 0.5 * (snr * (rvvf - Rf) + snz * zvvf)) * (onp * onp)
    azp1u = auu + auv + avv
    azm1u = auu - auv + avv
    cma11u = avv - auu
    delt1u = adp * adm - cma * cma
    r1p = (azp1u * (delt1u - cma * cma) / adp - azm1u * adp + 2.0 * cma11u * cma) / delt1u
    r1m = (azm1u * (delt1u - cma * cma) / adm - azp1u * adm + 2.0 * cma11u * cma) / delt1u
    r0p = (-azp1u * adm * cma / adp - azm1u * cma + 2.0 * cma11u * adm) / delt1u
    r0m = (-azm1u * adp * cma / adm - azp1u * cma + 2.0 * cma11u * adp) / delt1u
    ra1p = azp1u / adp
    ra1m = azm1u / adm

    bsin = jnp.zeros((mf + 1, 2 * nf + 1), dtype=Rf.dtype)
    bcos = jnp.zeros((mf + 1, 2 * nf + 1), dtype=Rf.dtype)
    gsin = jnp.zeros((mf + 1, 2 * nf + 1, npts), dtype=Rf.dtype)
    gcos = jnp.zeros((mf + 1, 2 * nf + 1, npts), dtype=Rf.dtype)
    # ``cmns`` is a static VMEC analytic-integral coefficient table, not a
    # differentiable variable.  Keep it as a host constant so the compiled
    # closure can skip exact-zero coefficients without tracer booleans.
    cmns = np.asarray(basis["cmns"])

    sign1 = 1.0
    fl1 = 0.0
    for l in range(0, mf + nf + 1):
        fl = fl1
        slp = (r1p * fl + ra1p) * tlp + r0p * fl * tlp_prev - (r1p + r0p) / sqrtc + sign1 * (r0p - r1p) / sqrta
        slm = (r1m * fl + ra1m) * tlm + r0m * fl * tlm_prev - (r1m + r0m) / sqrtc + sign1 * (r0m - r1m) / sqrta
        slpm = slp + slm
        for nabs in range(0, nf + 1):
            zv = float(nabs) * zeta
            cosv = jnp.cos(zv)
            sinv = jnp.sin(zv)
            for m in range(0, mf + 1):
                cm = float(cmns[l, m, nabs])
                if cm == 0.0:
                    continue
                mu = float(m) * theta
                sinu = jnp.sin(mu)
                cosu = jnp.cos(mu)
                col_p = int(nabs + nf)
                col_m = int((-nabs) + nf)
                if nabs == 0 or m == 0:
                    sinp = (sinu * cosv - sinv * cosu) * cm
                    bsin = bsin.at[m, col_p].add(jnp.sum(tlpm * bex * sinp))
                    gsin = gsin.at[m, col_p, :].add(slpm * sinp)
                    if lasym:
                        cosp = (cosu * cosv + sinv * sinu) * cm
                        bcos = bcos.at[m, col_p].add(jnp.sum(tlpm * bex * cosp))
                        gcos = gcos.at[m, col_p, :].add(slpm * cosp)
                else:
                    sinp0 = sinu * cosv * cm
                    temp = -cosu * sinv * cm
                    sinm = sinp0 - temp
                    sinp = sinp0 + temp
                    bsin = bsin.at[m, col_p].add(jnp.sum(tlm * bex * sinp))
                    bsin = bsin.at[m, col_m].add(jnp.sum(tlp * bex * sinm))
                    gsin = gsin.at[m, col_p, :].add(slm * sinp)
                    gsin = gsin.at[m, col_m, :].add(slp * sinm)
                    if lasym:
                        cosp0 = cosu * cosv * cm
                        temp2 = sinu * sinv * cm
                        cosm = cosp0 - temp2
                        cosp = cosp0 + temp2
                        bcos = bcos.at[m, col_p].add(jnp.sum(tlm * bex * cosp))
                        bcos = bcos.at[m, col_m].add(jnp.sum(tlp * bex * cosm))
                        gcos = gcos.at[m, col_p, :].add(slm * cosp)
                        gcos = gcos.at[m, col_m, :].add(slp * cosm)

        fl1 = fl1 + 1.0
        fl2 = 2.0 * fl1 - 1.0
        sign1 = -sign1
        tlp_next = ((sqrtc + sign1 * sqrta) - fl2 * cma * tlp - fl * adm * tlp_prev) / (adp * fl1)
        tlm_next = ((sqrtc + sign1 * sqrta) - fl2 * cma * tlm - fl * adp * tlm_prev) / (adm * fl1)
        tlp_prev = tlp
        tlm_prev = tlm
        tlp = tlp_next
        tlm = tlm_next
        tlpm = tlp + tlm

    xmpot = np.asarray(basis["xmpot"], dtype=np.int32)
    n_raw = np.asarray(basis["n_raw"], dtype=np.int32)
    out_s = jnp.zeros((mnpd,), dtype=Rf.dtype)
    out_c = jnp.zeros((mnpd,), dtype=Rf.dtype)
    gr_s = jnp.zeros((mnpd, npts), dtype=Rf.dtype)
    gr_c = jnp.zeros((mnpd, npts), dtype=Rf.dtype)
    for j in range(mnpd):
        m = int(xmpot[j])
        n = int(n_raw[j])
        col = int(n + nf)
        out_s = out_s.at[j].set(bsin[m, col])
        gr_s = gr_s.at[j, :].set(gsin[m, col, :])
        if lasym:
            out_c = out_c.at[j].set(bcos[m, col])
            gr_c = gr_c.at[j, :].set(gcos[m, col, :])

    if lasym:
        return jnp.concatenate([out_s, out_c], axis=0), jnp.concatenate([gr_s, gr_c], axis=0)
    return out_s, gr_s


def dense_mode_vacuum_solve_jax(
    mode_matrix: Any,
    rhs_mode: Any,
    sin_basis: Any,
    cos_basis: Any | None = None,
    *,
    symmetric: bool = False,
    include_phi_flat: bool = True,
    include_residual: bool = True,
) -> dict[str, Any]:
    """Solve a dense mode-space vacuum system and reconstruct a grid potential.

    This is the next scaffold between the dense toy solve and the production
    NESTOR path.  The current VMEC-like NESTOR implementation eventually builds
    a dense mode-space matrix and right-hand side before reconstructing a
    scalar potential on the boundary grid.  This helper makes that contract
    JAX-transformable and differentiable while the full source/operator assembly
    remains in the host implementation.

    Parameters
    ----------
    mode_matrix:
        Dense mode-space matrix ``A``.
    rhs_mode:
        Right-hand side vector ``b``.
    sin_basis, cos_basis:
        Flattened boundary-grid basis arrays with shape ``(npoints, nmodes)``.
        For stellarator-symmetric mode vectors pass only ``sin_basis``.  For
        LASYM-style doubled vectors pass both basis blocks; the first block of
        ``mode_coeffs`` multiplies ``sin_basis`` and the second multiplies
        ``cos_basis``.
    symmetric:
        Forwarded to :func:`dense_vacuum_solve_jax`.
    include_phi_flat:
        Reconstruct the scalar potential on the boundary grid. Compact
        accepted-state replay paths can disable this because field
        reconstruction uses the mode coefficients directly.
    include_residual:
        Include the dense linear residual in the returned diagnostics. Compact
        accepted-state replay paths can disable this since only ``bsqvac`` is
        needed for the strict VMEC update.
    """

    A = jnp.asarray(mode_matrix)
    rhs = jnp.asarray(rhs_mode)
    sin = jnp.asarray(sin_basis)
    if sin.ndim != 2:
        raise ValueError("sin_basis must be a 2D array")
    coeffs = dense_vacuum_solve_jax(A, rhs, symmetric=bool(symmetric))

    if cos_basis is None:
        if coeffs.shape[0] != sin.shape[1]:
            raise ValueError("rhs/mode_matrix size must match sin_basis columns")
        phi_flat = sin @ coeffs if bool(include_phi_flat) else None
    else:
        cos = jnp.asarray(cos_basis)
        if cos.shape != sin.shape:
            raise ValueError("cos_basis must match sin_basis shape")
        nmodes = int(sin.shape[1])
        if coeffs.shape[0] != 2 * nmodes:
            raise ValueError("doubled rhs/mode_matrix size must be 2 * sin_basis columns")
        phi_flat = sin @ coeffs[:nmodes] + cos @ coeffs[nmodes:] if bool(include_phi_flat) else None

    out = {"mode_coeffs": coeffs}
    if bool(include_phi_flat):
        out["phi_flat"] = phi_flat
    if bool(include_residual):
        out["residual"] = dense_vacuum_residual(A, coeffs, rhs)
    return out


def _nonsingular_full_grid_from_active_jax(
    *,
    R: Any,
    Z: Any,
    Ru: Any,
    Zu: Any,
    Rv: Any,
    Zv: Any,
    ruu: Any,
    ruv: Any,
    rvv: Any,
    zuu: Any,
    zuv: Any,
    zvv: Any,
    basis: dict[str, Any],
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any]:
    """Return the full grid expected by VMEC's nonsingular Green block.

    The host bridge expands stellarator-symmetric active-grid geometry before
    calling the nonsingular Green-function assembly, but keeps the analytic
    singular terms on the active grid. This helper mirrors that convention in
    JAX for the combined low-resolution operator.
    """

    R2 = jnp.asarray(R)
    Z2 = jnp.asarray(Z)
    Ru2 = jnp.asarray(Ru)
    Zu2 = jnp.asarray(Zu)
    Rv2 = jnp.asarray(Rv)
    Zv2 = jnp.asarray(Zv)
    ruu2 = jnp.asarray(ruu)
    ruv2 = jnp.asarray(ruv)
    rvv2 = jnp.asarray(rvv)
    zuu2 = jnp.asarray(zuu)
    zuv2 = jnp.asarray(zuv)
    zvv2 = jnp.asarray(zvv)
    nu_full = int(basis["nu_full"])
    ntheta3, nv = int(R2.shape[0]), int(R2.shape[1])
    if bool(basis["lasym"]) or nu_full == ntheta3:
        return R2, Z2, Ru2, Zu2, Rv2, Zv2, ruu2, ruv2, rvv2, zuu2, zuv2, zvv2

    shape_full = (nu_full, nv)
    zeros = jnp.zeros(shape_full, dtype=R2.dtype)
    Rf = zeros.at[:ntheta3, :].set(R2)
    Zf = zeros.at[:ntheta3, :].set(Z2)
    Ruf = zeros.at[:ntheta3, :].set(Ru2)
    Zuf = zeros.at[:ntheta3, :].set(Zu2)
    Rvf = zeros.at[:ntheta3, :].set(Rv2)
    Zvf = zeros.at[:ntheta3, :].set(Zv2)
    ruuf = zeros.at[:ntheta3, :].set(ruu2)
    ruvf = zeros.at[:ntheta3, :].set(ruv2)
    rvvf = zeros.at[:ntheta3, :].set(rvv2)
    zuuf = zeros.at[:ntheta3, :].set(zuu2)
    zuvf = zeros.at[:ntheta3, :].set(zuv2)
    zvvf = zeros.at[:ntheta3, :].set(zvv2)

    kv_m = (nv - jnp.arange(nv, dtype=jnp.int32)) % max(1, nv)
    for ku in range(1, max(1, ntheta3 - 1)):
        km = (nu_full - ku) % max(1, nu_full)
        if km < ntheta3:
            continue
        Rf = Rf.at[km, :].set(R2[ku, kv_m])
        Zf = Zf.at[km, :].set(-Z2[ku, kv_m])
        Ruf = Ruf.at[km, :].set(-Ru2[ku, kv_m])
        Zuf = Zuf.at[km, :].set(Zu2[ku, kv_m])
        Rvf = Rvf.at[km, :].set(-Rv2[ku, kv_m])
        Zvf = Zvf.at[km, :].set(Zv2[ku, kv_m])

    return Rf, Zf, Ruf, Zuf, Rvf, Zvf, ruuf, ruvf, rvvf, zuuf, zuvf, zvvf


def dense_vmec_nestor_mode_solve_jax(
    *,
    R: Any,
    Z: Any,
    Ru: Any,
    Zu: Any,
    Rv: Any,
    Zv: Any,
    ruu: Any,
    ruv: Any,
    rvv: Any,
    zuu: Any,
    zuv: Any,
    zvv: Any,
    bexni: Any,
    basis: dict[str, Any],
    tables: dict[str, Any],
    signgs: int,
    nvper: int,
    include_analytic: bool = True,
    symmetric: bool = False,
    include_phi_flat: bool = True,
    include_residual: bool = True,
    solve_mode: str = "dense",
    operator_solver: str = "gmres",
    operator_tol: float = 1.0e-11,
    operator_atol: float = 1.0e-13,
    operator_maxiter: int | None = None,
    operator_restart: int | None = None,
) -> dict[str, Any]:
    """Assemble and solve the dense JAX VMEC/NESTOR mode operator.

    This is the first cohesive JAX-native operator API for the free-boundary
    adjoint lane.  It combines the nonsingular Green-function contribution, the
    analytic/singular ``analyt.f`` contribution, VMEC mode projection, and the
    implicit dense mode-space solve.  It is meant for low-resolution validation
    and finite-difference gates before replacing the production host NESTOR
    bridge with a matrix-free/custom-transpose implementation.
    """

    full_grid = _nonsingular_full_grid_from_active_jax(
        R=R,
        Z=Z,
        Ru=Ru,
        Zu=Zu,
        Rv=Rv,
        Zv=Zv,
        ruu=ruu,
        ruv=ruv,
        rvv=rvv,
        zuu=zuu,
        zuv=zuv,
        zvv=zvv,
        basis=basis,
    )
    gsource_nonsing, grpmn_nonsing = vmec_nonsingular_terms_from_bexni_jax(
        R=full_grid[0],
        Z=full_grid[1],
        Ru=full_grid[2],
        Zu=full_grid[3],
        Rv=full_grid[4],
        Zv=full_grid[5],
        ruu=full_grid[6],
        ruv=full_grid[7],
        rvv=full_grid[8],
        zuu=full_grid[9],
        zuv=full_grid[10],
        zvv=full_grid[11],
        bexni=bexni,
        basis=basis,
        tables=tables,
        signgs=signgs,
        nvper=nvper,
    )
    rhs = mode_rhs_from_gsource_jax(
        gsource_nonsing,
        sin_basis=basis["sinmni"],
        cos_basis=basis["cosmni"],
        xmpot=basis["xmpot"],
        n_raw=basis["n_raw"],
        onp=float(basis["onp"]),
        lasym=bool(basis["lasym"]),
        nuv3=int(basis["nuv3"]),
        nuv_full=int(basis["nuv_full"]),
        imirr=basis["imirr"],
        imirr_full=basis["imirr_full"],
    )
    grpmn = grpmn_nonsing
    if bool(include_analytic):
        bvec_analytic, grpmn_analytic = vmec_analytic_terms_from_geometry_jax(
            R=R,
            Ru=Ru,
            Rv=Rv,
            Zu=Zu,
            Zv=Zv,
            ruu=ruu,
            ruv=ruv,
            rvv=rvv,
            zuu=zuu,
            zuv=zuv,
            zvv=zvv,
            bexni=bexni,
            basis=basis,
            signgs=signgs,
        )
        rhs = rhs + bvec_analytic
        grpmn = grpmn + grpmn_analytic

    solve_mode_name = str(solve_mode).strip().lower()
    if solve_mode_name in ("dense", "matrix", "mode_matrix"):
        mode_matrix = mode_matrix_from_grpmn_jax(
            grpmn,
            sin_basis=basis["sinmni"],
            cos_basis=basis["cosmni"],
            xmpot=basis["xmpot"],
            n_raw=basis["n_raw"],
            lasym=bool(basis["lasym"]),
            mn0=int(basis["mn0"]),
        )
        solved = dense_mode_vacuum_solve_jax(
            mode_matrix,
            rhs,
            basis["sinmni"],
            basis["cosmni"] if bool(basis["lasym"]) else None,
            symmetric=symmetric,
            include_phi_flat=bool(include_phi_flat),
            include_residual=bool(include_residual),
        )
        solved["solve_mode"] = "dense"
        solved["mode_matrix_materialized"] = True
    elif solve_mode_name in ("matrix_free", "operator", "operator_gmres", "gmres", "bicgstab"):
        solver_name = "bicgstab" if solve_mode_name == "bicgstab" else str(operator_solver).strip().lower()
        mode_matrix = None
        solved = mode_operator_vacuum_solve_jax(
            grpmn,
            rhs,
            sin_basis=basis["sinmni"],
            cos_basis=basis["cosmni"] if bool(basis["lasym"]) else None,
            xmpot=basis["xmpot"],
            n_raw=basis["n_raw"],
            lasym=bool(basis["lasym"]),
            mn0=int(basis["mn0"]),
            include_phi_flat=bool(include_phi_flat),
            include_residual=bool(include_residual),
            solver=solver_name,
            tol=float(operator_tol),
            atol=float(operator_atol),
            maxiter=operator_maxiter,
            restart=operator_restart,
        )
    else:
        raise ValueError("solve_mode must be 'dense' or 'matrix_free'")
    return {
        **solved,
        "rhs_mode": rhs,
        "mode_matrix": mode_matrix,
        "gsource_nonsing": gsource_nonsing,
        "grpmn": grpmn,
    }


def direct_coil_boundary_bsqvac_jax(
    params: Any,
    *,
    R: Any,
    Z: Any,
    phi: Any,
    Ru: Any,
    Zu: Any,
    Rv: Any,
    Zv: Any,
    ruu: Any,
    ruv: Any,
    rvv: Any,
    zuu: Any,
    zuv: Any,
    zvv: Any,
    basis: dict[str, Any],
    tables: dict[str, Any],
    signgs: int,
    nvper: int,
    br_add: Any = 0.0,
    bp_add: Any = 0.0,
    bz_add: Any = 0.0,
    wint: Any | None = None,
    include_analytic: bool = True,
    include_diagnostics: bool = True,
    include_mode_diagnostics: bool = True,
    vac_override: Mapping[str, Any] | None = None,
    coil_geometry: Any | None = None,
    nestor_solve_mode: str = "dense",
    nestor_operator_solver: str = "gmres",
    nestor_operator_tol: float = 1.0e-11,
    nestor_operator_atol: float = 1.0e-13,
    nestor_operator_maxiter: int | None = None,
    nestor_operator_restart: int | None = None,
) -> dict[str, Any]:
    """Replay accepted-boundary direct-coil ``bsqvac`` through JAX NESTOR.

    This is the reusable phase-2 validation primitive for the production
    accepted-output ladder.  It holds a VMEC plasma boundary fixed, samples the
    differentiable direct-coil Biot-Savart field on that boundary, projects the
    normal field into VMEC/NESTOR source space, solves the dense JAX mode-space
    vacuum response, and reconstructs ``bsqvac`` on the boundary.

    The helper validates and exposes the differentiable accepted-boundary
    replay contract.  It intentionally does **not** differentiate through the
    outer host-controlled nonlinear VMEC iteration loop.
    """

    from .external_fields import sample_coil_field_cylindrical, sample_coil_field_cylindrical_from_geometry

    R_j = jnp.asarray(R)
    if vac_override is None:
        with _jax_named_scope("vmec_jax.free_boundary.direct_coil_sample"):
            if coil_geometry is None:
                br, bp, bz = sample_coil_field_cylindrical(
                    params,
                    R_j,
                    jnp.asarray(Z),
                    jnp.asarray(phi),
                )
            else:
                br, bp, bz = sample_coil_field_cylindrical_from_geometry(
                    coil_geometry,
                    R_j,
                    jnp.asarray(Z),
                    jnp.asarray(phi),
                    regularization_epsilon=float(getattr(params, "regularization_epsilon", 0.0)),
                    chunk_size=getattr(params, "chunk_size", None),
                )
            br = br + jnp.asarray(br_add, dtype=br.dtype)
            bp = bp + jnp.asarray(bp_add, dtype=bp.dtype)
            bz = bz + jnp.asarray(bz_add, dtype=bz.dtype)
        with _jax_named_scope("vmec_jax.free_boundary.vacuum_boundary_projection"):
            vac = vacuum_boundary_fields_from_cylindrical_jax(
                br=br,
                bp=bp,
                bz=bz,
                R=R_j,
                Ru=Ru,
                Zu=Zu,
                Rv=Rv,
                Zv=Zv,
                include_bnormal_unit=False,
                include_contravariant=False,
            )
    else:
        vac = {
            "bu": jnp.asarray(vac_override["bu"]),
            "bv": jnp.asarray(vac_override["bv"]),
            "bnormal": jnp.asarray(vac_override["bnormal"]),
            "g_uu": jnp.asarray(vac_override["g_uu"]),
            "g_uv": jnp.asarray(vac_override["g_uv"]),
            "g_vv": jnp.asarray(vac_override["g_vv"]),
        }
    if wint is None:
        wint_j = jnp.ones_like(R_j)
    else:
        wint_j = jnp.asarray(wint, dtype=jnp.asarray(vac["bnormal"]).dtype)
    bexni = -jnp.asarray(vac["bnormal"]) * wint_j * ((2.0 * jnp.pi) ** 2)
    with _jax_named_scope("vmec_jax.free_boundary.dense_nestor_mode_solve"):
        mode_solution = dense_vmec_nestor_mode_solve_jax(
            R=R_j,
            Z=Z,
            Ru=Ru,
            Zu=Zu,
            Rv=Rv,
            Zv=Zv,
            ruu=ruu,
            ruv=ruv,
            rvv=rvv,
            zuu=zuu,
            zuv=zuv,
            zvv=zvv,
            bexni=jnp.ravel(bexni),
            basis=basis,
            tables=tables,
            signgs=int(signgs),
            nvper=int(nvper),
            include_analytic=bool(include_analytic),
            include_phi_flat=bool(include_mode_diagnostics),
            include_residual=bool(include_mode_diagnostics),
            solve_mode=str(nestor_solve_mode),
            operator_solver=str(nestor_operator_solver),
            operator_tol=float(nestor_operator_tol),
            operator_atol=float(nestor_operator_atol),
            operator_maxiter=nestor_operator_maxiter,
            operator_restart=nestor_operator_restart,
        )
    with _jax_named_scope("vmec_jax.free_boundary.mode_field_reconstruction"):
        channels = vacuum_boundary_fields_from_mode_coeffs_jax(
            mode_solution["mode_coeffs"],
            basis=basis,
            bu_ext=vac["bu"],
            bv_ext=vac["bv"],
            g_uu=vac["g_uu"],
            g_uv=vac["g_uv"],
            g_vv=vac["g_vv"],
        )
    out = {"bsqvac": channels["bsqvac"]}
    if bool(include_diagnostics):
        out.update(
            {
                "channels": channels,
                "mode_solution": mode_solution,
                "vac": vac,
                "bexni": bexni,
            }
        )
    return out


def direct_coil_boundary_bsqvac_from_trace_jax(
    params: Any,
    geometry: dict[str, Any],
    trace: dict[str, Any],
    *,
    basis: dict[str, Any],
    tables: dict[str, Any],
    signgs: int,
    nvper: int,
    wint: Any,
    include_analytic: bool = True,
    include_diagnostics: bool = True,
    include_mode_diagnostics: bool = True,
    freeze_vacuum_field: bool = False,
    coil_geometry: Any | None = None,
    nestor_solve_mode: str = "dense",
    nestor_operator_solver: str = "gmres",
    nestor_operator_tol: float = 1.0e-11,
    nestor_operator_atol: float = 1.0e-13,
    nestor_operator_maxiter: int | None = None,
    nestor_operator_restart: int | None = None,
) -> dict[str, Any]:
    """Replay direct-coil ``bsqvac`` on accepted geometry using trace metadata.

    ``trace`` may be either a full residual-step trace containing
    ``freeb_nestor_trace`` or the nested NESTOR trace itself.  This keeps the
    production validation ladder from duplicating trace-to-replay plumbing in
    every test while keeping the differentiated path explicit: accepted
    geometry and direct-coil parameters remain JAX-visible, while basis/tables
    and axis-additive fields are captured trace data.
    """

    nestor_trace = trace.get("freeb_nestor_trace", trace)
    if not isinstance(nestor_trace, dict):
        raise ValueError("trace must be a NESTOR trace or contain 'freeb_nestor_trace'")

    vac_override = _direct_coil_trace_vacuum_field_override(trace) if bool(freeze_vacuum_field) else None
    return direct_coil_boundary_bsqvac_jax(
        params,
        R=geometry["R"],
        Z=geometry["Z"],
        phi=geometry["phi"],
        Ru=geometry["Ru"],
        Zu=geometry["Zu"],
        Rv=geometry["Rv"],
        Zv=geometry["Zv"],
        ruu=geometry["ruu"],
        ruv=geometry["ruv"],
        rvv=geometry["rvv"],
        zuu=geometry["zuu"],
        zuv=geometry["zuv"],
        zvv=geometry["zvv"],
        basis=basis,
        tables=tables,
        signgs=int(signgs),
        nvper=int(nvper),
        br_add=jnp.asarray(nestor_trace["br_axis"]),
        bp_add=jnp.asarray(nestor_trace["bp_axis"]),
        bz_add=jnp.asarray(nestor_trace["bz_axis"]),
        wint=jnp.asarray(wint),
        include_analytic=bool(include_analytic),
        include_diagnostics=bool(include_diagnostics),
        include_mode_diagnostics=bool(include_mode_diagnostics),
        vac_override=vac_override,
        coil_geometry=coil_geometry,
        nestor_solve_mode=nestor_solve_mode,
        nestor_operator_solver=nestor_operator_solver,
        nestor_operator_tol=nestor_operator_tol,
        nestor_operator_atol=nestor_operator_atol,
        nestor_operator_maxiter=nestor_operator_maxiter,
        nestor_operator_restart=nestor_operator_restart,
    )


def direct_coil_accepted_trace_replay_objective_jax(
    params: Any,
    initial_state: Any,
    *,
    static: Any,
    traces: Any,
    signgs: int,
    max_steps: int | None = None,
    sample_nzeta: int | None = None,
    include_analytic: bool = True,
    enforce_edge: bool = False,
    state_weight: Any = 1.0,
    force_weight: Any = 0.0,
    bsqvac_weight: Any = 0.0,
    coil_geometry: Any | None = None,
) -> dict[str, Any]:
    """Replay fixed accepted free-boundary traces with differentiable coils.

    This helper is the reusable bridge between accepted-boundary replay and a
    future full nonlinear ``run_free_boundary`` custom adjoint.  A production
    solve supplies accepted trace metadata: step controls, preconditioner
    matrices, axis-additive fields, and NESTOR replay context.  This function
    keeps those controls fixed, while recomputing at every replayed step

    ``state -> boundary geometry -> direct-coil Biot-Savart -> JAX NESTOR
    bsqvac -> strict VMEC update``.

    The result is a small differentiable fixed-control nonlinear replay.  It is
    appropriate for AD-vs-central-FD validation of accepted-output
    sensitivities, but it intentionally does not claim gradients through the
    adaptive host controller that selected the accepted production traces.
    """

    from .discrete_adjoint import strict_update_one_step_from_trace
    from .state import pack_state

    trace_seq = list(traces)
    if max_steps is not None:
        trace_seq = trace_seq[: int(max_steps)]
    if not trace_seq:
        raise ValueError("at least one accepted trace is required")
    reset_flags = _accepted_trace_reset_flags(trace_seq)

    state = initial_state
    objective_components: dict[str, Any] = {
        "state": jnp.asarray(0.0),
        "force": jnp.asarray(0.0),
        "bsqvac": jnp.asarray(0.0),
    }
    context_cache: dict[tuple[int, int], dict[str, Any]] = {}

    def _precomputed_context_for_trace(trace: Mapping[str, Any]) -> dict[str, Any] | None:
        shape = _direct_coil_trace_boundary_shape(trace)
        if shape is None:
            return None
        if shape not in context_cache:
            context_cache[shape] = direct_coil_boundary_replay_context_for_shape(
                static,
                ntheta=shape[0],
                nzeta=shape[1],
            )
        return context_cache[shape]

    steps: list[dict[str, Any]] = []
    bsqvac_values: list[Any] = []
    for trace, reset_to_trace_pre in zip(trace_seq, reset_flags, strict=True):
        if reset_to_trace_pre:
            # VMEC free-boundary turn-on/restart control can reset the working
            # state between accepted trace entries. Preserve that fixed host
            # control transition instead of incorrectly chaining state_post.
            state = trace["state_pre"]
        has_active_freeb_replay = trace.get("freeb_bsqvac_half") is not None and trace.get("freeb_nestor_trace") is not None
        if has_active_freeb_replay:
            with _jax_named_scope("vmec_jax.free_boundary.boundary_geometry"):
                geometry = free_boundary_boundary_geometry_jax(
                    state,
                    static,
                    sample_nzeta=sample_nzeta,
                )
            context = _precomputed_context_for_trace(trace)
            if context is None or tuple(int(v) for v in geometry["R"].shape) != (
                int(context["ntheta"]),
                int(context["nzeta"]),
            ):
                with _jax_named_scope("vmec_jax.free_boundary.replay_context"):
                    context = direct_coil_boundary_replay_context(static, geometry)
            with _jax_named_scope("vmec_jax.free_boundary.direct_coil_bsqvac_replay"):
                replay = direct_coil_boundary_bsqvac_from_trace_jax(
                    params,
                    geometry,
                    trace,
                    basis=context["basis"],
                    tables=context["tables"],
                    signgs=int(signgs),
                    nvper=int(context["nvper"]),
                    wint=jnp.asarray(context["wint"]),
                    include_analytic=bool(include_analytic),
                    coil_geometry=coil_geometry,
                )
            freeb_bsqvac_half = replay["bsqvac"]
        else:
            # Full accepted-trace replay must preserve non-vacuum/setup steps.
            # These steps do not have enough NESTOR metadata to resample coils,
            # so replay the original trace payload and keep coil derivatives
            # zero for that step.
            replay = None
            freeb_bsqvac_half = trace.get("freeb_bsqvac_half", None)
        with _jax_named_scope("vmec_jax.free_boundary.strict_update_one_step_from_trace"):
            step = strict_update_one_step_from_trace(
                state,
                static,
                trace,
                freeb_bsqvac_half=freeb_bsqvac_half,
                enforce_edge=bool(enforce_edge),
            )
        state = step["step"]["state_post"]
        steps.append(step)
        bsqvac_values.append(freeb_bsqvac_half)
        objective_components["force"] = objective_components["force"] + _tree_weighted_half_norm(
            step["force"],
            force_weight,
        )
        if replay is not None:
            objective_components["bsqvac"] = objective_components["bsqvac"] + _weighted_half_norm(
                replay["bsqvac"],
                bsqvac_weight,
            )

    objective_components["state"] = _weighted_half_norm(
        pack_state(state),
        state_weight,
    )
    objective = sum(objective_components.values())
    return {
        "objective": objective,
        "objective_components": objective_components,
        "state": state,
        "steps": steps,
        "bsqvac": bsqvac_values,
        "state_reset_flags": tuple(reset_flags),
    }


def free_boundary_adjoint_trace_replay_diagnostics(
    source: Any,
    *,
    accept_mask: Any | None = None,
    done_mask: Any | None = None,
    max_steps: int | None = None,
    json_safe: bool = False,
) -> dict[str, Any]:
    """Return diagnostics for fixed accepted-trace free-boundary replay.

    The returned contract is intentionally conservative: it describes a fixed
    accepted-branch replay payload and explicitly does *not* claim that the
    adaptive host controller is differentiated.  Callers should use it to gate
    complete-solve finite-difference comparisons before invoking any
    branch-local custom VJP.
    """

    traces = _extract_adjoint_step_trace(source)
    if max_steps is not None:
        traces = traces[: int(max_steps)]
    if not traces:
        raise RuntimeError(
            "adjoint_step_trace is empty. Run the residual solver with "
            "adjoint_trace=True and adjoint_trace_mode='full'."
        )
    metadata = direct_coil_accepted_trace_branch_metadata(
        traces,
        accept_mask=accept_mask,
        done_mask=done_mask,
        max_steps=max_steps,
        json_safe=False,
    )
    scalar_ok, scalar_error = _stackability_probe(
        "scalar_controls",
        direct_coil_accepted_trace_scalar_controls_jax,
        traces,
    )
    array_ok, array_error = _stackability_probe(
        "array_controls",
        direct_coil_accepted_trace_array_controls_jax,
        traces,
    )
    preconditioner_ok, preconditioner_error = _stackability_probe(
        "preconditioner_controls",
        direct_coil_accepted_trace_preconditioner_controls_jax,
        traces,
    )
    errors = {
        key: value
        for key, value in {
            "scalar_controls": scalar_error,
            "array_controls": array_error,
            "preconditioner_controls": preconditioner_error,
        }.items()
        if value is not None
    }
    diagnostics = {
        "contract": "fixed accepted-trace replay diagnostics only",
        "differentiates_adaptive_controller": False,
        "n_steps": metadata["n_steps"],
        "branch_fingerprint": metadata["fingerprint"],
        "masks": metadata["masks"],
        "replay_diagnostics": {
            "preconditioner_policy_n_segments": len(metadata["preconditioner_policy_segments"]),
            "preconditioner_policy_segment_summary": metadata["preconditioner_policy_segment_summary"],
            "scalar_controls_stackable": bool(scalar_ok),
            "array_controls_stackable": bool(array_ok),
            "preconditioner_controls_stackable": bool(preconditioner_ok),
            "errors": errors,
        },
    }
    if json_safe:
        return _json_safe_fingerprint_value(diagnostics)
    return diagnostics


def direct_coil_accepted_trace_controller_replay_objective_jax(
    params: Any,
    initial_state: Any,
    *,
    static: Any,
    traces: Any,
    signgs: int,
    max_steps: int | None = None,
    sample_nzeta: int | None = None,
    include_analytic: bool = True,
    enforce_edge: bool = False,
    state_weight: Any = 1.0,
    force_weight: Any = 0.0,
    bsqvac_weight: Any = 0.0,
    checkpoint_steps: bool = False,
    accept_mask: Any | None = None,
    done_mask: Any | None = None,
    use_preconditioner_policy_segments: bool = False,
    use_segment_preconditioner_controls: bool = False,
    use_stacked_step_controls: bool = False,
    use_accepted_only_fast_path: bool = True,
    replay_plan: Mapping[str, Any] | None = None,
    include_replay_aux: bool = True,
    state_only_replay: bool = False,
    freeze_vacuum_field: bool = False,
    freeze_freeb_bsqvac: bool = False,
    include_mode_diagnostics: bool = False,
    nestor_solve_mode: str = "dense",
    nestor_operator_solver: str = "gmres",
    nestor_operator_tol: float = 1.0e-11,
    nestor_operator_atol: float = 1.0e-13,
    nestor_operator_maxiter: int | None = None,
    nestor_operator_restart: int | None = None,
    jit_preconditioner_apply: bool = True,
    unroll_accepted_only_segments_below: int = 0,
    coil_geometry: Any | None = None,
) -> dict[str, Any]:
    """Replay fixed production traces through a JAX-visible accept controller.

    This is the bridge between the legacy Python-loop
    :func:`direct_coil_accepted_trace_replay_objective_jax` and a future full
    nonlinear free-boundary controller.  The production traces remain fixed
    data, but the replayed state, per-step accepted masks, and objective
    history are carried through :func:`jax_visible_accepted_nonlinear_controller_jax`.
    If ``use_preconditioner_policy_segments`` is true, the same controls are
    split into consecutive static-preconditioner-policy segments and run
    through :func:`jax_visible_segmented_accepted_nonlinear_controller_jax`.
    The segmented path is behavior-preserving and opt-in while production
    preconditioner dispatch remains partially branch-local.
    ``use_segment_preconditioner_controls`` is a narrower performance
    diagnostic: when the full trace cannot stack preconditioner controls, it
    tries stacking them independently inside each static segment.  It is kept
    opt-in because current tiny production traces show parity but not a speed
    win.
    ``use_stacked_step_controls`` is the next rung: it segments by the full
    static step-policy signature and calls ``strict_update_one_step_from_state``
    directly with stacked state/update/constraint controls.
    ``use_accepted_only_fast_path`` removes the per-step accept/reject proposal
    conditional only for segments whose effective controller masks prove that
    every slot is active and accepted. Rejected, inactive, or post-convergence
    padded slots automatically use the ordinary controller path.
    ``state_only_replay`` is a narrower production-report fast path: it still
    replays the direct-coil vacuum field and VMEC state update, but it omits
    per-step force/vacuum objective history needed only by history-dependent
    scalars such as accepted Bnormal/Bsqvac RMS.
    ``freeze_freeb_bsqvac`` is a diagnostic-only cost split: it reuses the
    accepted trace's ``bsqvac`` array instead of differentiably recomputing the
    direct-coil/NESTOR vacuum response.  This keeps the strict VMEC accepted
    update in the graph while intentionally removing coil sensitivity through
    the external-field replay, so it must not be used as a promoted derivative
    or optimization path.
    ``freeze_vacuum_field`` is an intermediate diagnostic split: it reuses the
    accepted trace's normal/tangential vacuum-field projection arrays but still
    runs the JAX dense NESTOR mode solve and field reconstruction.  This
    separates Biot-Savart/projection graph cost from NESTOR/source assembly
    graph cost, and is also not a promoted derivative path.
    ``include_mode_diagnostics`` controls dense-mode diagnostic outputs such as
    ``phi_flat`` and residual vectors.  Accepted-controller replay only needs
    ``bsqvac`` and optionally boundary RMS diagnostics, so branch-local reports
    default this to false to avoid building unused dense-solve outputs.
    ``nestor_solve_mode`` and the ``nestor_operator_*`` options expose the
    opt-in matrix-free NESTOR/source response inside fixed accepted-branch
    replay.  Dense remains the default; matrix-free replay is a validated
    performance/research seam until size-triggered promotion is justified.

    The helper intentionally keeps every trace accepted.  It does not
    differentiate through the host policy that selected the traces; it validates
    that a production accepted-trace replay can be represented as a static
    JAX-visible accepted-control scan.
    """

    from .discrete_adjoint import strict_update_one_step_from_state, strict_update_one_step_from_trace
    if replay_plan is None:
        trace_seq = tuple(traces)
        if max_steps is not None:
            trace_seq = trace_seq[: int(max_steps)]
    else:
        trace_seq = tuple(replay_plan["traces"])
    if not trace_seq:
        raise ValueError("at least one accepted trace is required")
    if jax is None:  # pragma: no cover - dependency fallback.
        raise RuntimeError("JAX is required for controller replay.")

    if replay_plan is None:
        replay_plan = direct_coil_accepted_trace_controller_replay_plan(
            trace_seq,
            static=static,
            accept_mask=accept_mask,
            done_mask=done_mask,
            max_steps=None,
            use_preconditioner_policy_segments=bool(use_preconditioner_policy_segments),
            use_segment_preconditioner_controls=bool(use_segment_preconditioner_controls),
            use_stacked_step_controls=bool(use_stacked_step_controls),
            use_accepted_only_fast_path=bool(use_accepted_only_fast_path),
        )

    controls = replay_plan["controls"]
    effective_masks = replay_plan["effective_masks"]
    preconditioner_policy_segments = replay_plan["preconditioner_policy_segments"]
    preconditioner_policy_segment_summary = replay_plan["preconditioner_policy_segment_summary"]
    scalar_controls = replay_plan["scalar_controls"]
    array_controls = replay_plan["array_controls"]
    step_controls = replay_plan["step_controls"]
    step_policy_segments = replay_plan["step_policy_segments"]
    step_policy_segment_summary = replay_plan["step_policy_segment_summary"]
    preconditioner_controls = replay_plan["preconditioner_controls"]
    preconditioner_controls_stacked = bool(replay_plan["preconditioner_controls_stacked"])
    plan_options = replay_plan.get("options", {})

    context_cache: dict[tuple[int, int], dict[str, Any]] = dict(replay_plan.get("boundary_replay_contexts_by_shape", {}))

    def _precomputed_context_for_trace(trace: Mapping[str, Any]) -> dict[str, Any] | None:
        shape = _direct_coil_trace_boundary_shape(trace)
        if shape is None:
            return None
        if shape not in context_cache:
            context_cache[shape] = direct_coil_boundary_replay_context_for_shape(
                static,
                ntheta=shape[0],
                nzeta=shape[1],
            )
        return context_cache[shape]

    def _step_control(control: Mapping[str, Any], key: str) -> Any:
        return control["step_controls"][key] if key in control.get("step_controls", {}) else None

    def _branch_for_trace(
        trace: dict[str, Any],
        state: Any,
        coil_params: Any,
        control: dict[str, Any],
        replay_context: dict[str, Any] | None,
    ):
        reset_to_trace_pre = jnp.asarray(control["reset_to_trace_pre"], dtype=bool)
        state_in = jax.lax.cond(
            reset_to_trace_pre,
            lambda _: trace["state_pre"],
            lambda _: state,
            operand=None,
        )
        has_active_freeb_replay = trace.get("freeb_bsqvac_half") is not None and trace.get("freeb_nestor_trace") is not None
        if has_active_freeb_replay:
            if bool(freeze_freeb_bsqvac):
                freeb_bsqvac_half = jnp.asarray(trace["freeb_bsqvac_half"])
            else:
                with _jax_named_scope("vmec_jax.free_boundary.boundary_geometry"):
                    geometry = free_boundary_boundary_geometry_jax(
                        state_in,
                        static,
                        sample_nzeta=sample_nzeta,
                    )
                context = replay_context
                if context is None or tuple(int(v) for v in geometry["R"].shape) != (
                    int(context["ntheta"]),
                    int(context["nzeta"]),
                ):
                    with _jax_named_scope("vmec_jax.free_boundary.replay_context"):
                        context = direct_coil_boundary_replay_context(static, geometry)
                nestor_axes = _step_control(control, "freeb_nestor_axes")
                if nestor_axes is None:
                    with _jax_named_scope("vmec_jax.free_boundary.direct_coil_bsqvac_replay"):
                        replay = direct_coil_boundary_bsqvac_from_trace_jax(
                            coil_params,
                            geometry,
                            trace,
                            basis=context["basis"],
                            tables=context["tables"],
                            signgs=int(signgs),
                            nvper=int(context["nvper"]),
                            wint=jnp.asarray(context["wint"]),
                            include_analytic=bool(include_analytic),
                            include_diagnostics=not bool(state_only_replay),
                            include_mode_diagnostics=bool(include_mode_diagnostics),
                            freeze_vacuum_field=bool(freeze_vacuum_field),
                            nestor_solve_mode=str(nestor_solve_mode),
                            nestor_operator_solver=str(nestor_operator_solver),
                            nestor_operator_tol=float(nestor_operator_tol),
                            nestor_operator_atol=float(nestor_operator_atol),
                            nestor_operator_maxiter=nestor_operator_maxiter,
                            nestor_operator_restart=nestor_operator_restart,
                            coil_geometry=coil_geometry,
                        )
                else:
                    with _jax_named_scope("vmec_jax.free_boundary.direct_coil_bsqvac_replay"):
                        replay = direct_coil_boundary_bsqvac_jax(
                            coil_params,
                            R=geometry["R"],
                            Z=geometry["Z"],
                            phi=geometry["phi"],
                            Ru=geometry["Ru"],
                            Zu=geometry["Zu"],
                            Rv=geometry["Rv"],
                            Zv=geometry["Zv"],
                            ruu=geometry["ruu"],
                            ruv=geometry["ruv"],
                            rvv=geometry["rvv"],
                            zuu=geometry["zuu"],
                            zuv=geometry["zuv"],
                            zvv=geometry["zvv"],
                            basis=context["basis"],
                            tables=context["tables"],
                            signgs=int(signgs),
                            nvper=int(context["nvper"]),
                            br_add=jnp.asarray(nestor_axes["br_axis"]),
                            bp_add=jnp.asarray(nestor_axes["bp_axis"]),
                            bz_add=jnp.asarray(nestor_axes["bz_axis"]),
                            wint=jnp.asarray(context["wint"]),
                            include_analytic=bool(include_analytic),
                            include_diagnostics=not bool(state_only_replay),
                            include_mode_diagnostics=bool(include_mode_diagnostics),
                            vac_override=(
                                _direct_coil_trace_vacuum_field_override(trace)
                                if bool(freeze_vacuum_field)
                                else None
                            ),
                            coil_geometry=coil_geometry,
                            nestor_solve_mode=str(nestor_solve_mode),
                            nestor_operator_solver=str(nestor_operator_solver),
                            nestor_operator_tol=float(nestor_operator_tol),
                            nestor_operator_atol=float(nestor_operator_atol),
                            nestor_operator_maxiter=nestor_operator_maxiter,
                            nestor_operator_restart=nestor_operator_restart,
                        )
                freeb_bsqvac_half = replay["bsqvac"]
            if bool(state_only_replay):
                bsqvac_objective = jnp.asarray(0.0)
                bsqvac_rms = jnp.asarray(0.0)
                bnormal_rms = jnp.asarray(0.0)
            elif bool(freeze_freeb_bsqvac):
                bsqvac_objective = _weighted_half_norm(freeb_bsqvac_half, bsqvac_weight)
                bsqvac_rms = jnp.sqrt(jnp.mean(jnp.square(jnp.asarray(freeb_bsqvac_half))))
                bnormal_rms = jnp.asarray(0.0)
            else:
                bsqvac_objective = _weighted_half_norm(replay["bsqvac"], bsqvac_weight)
                bsqvac_rms = jnp.sqrt(jnp.mean(jnp.square(jnp.asarray(replay["bsqvac"]))))
                bnormal_rms = jnp.sqrt(jnp.mean(jnp.square(jnp.asarray(replay["vac"]["bnormal"]))))
        else:
            freeb_bsqvac_half = trace.get("freeb_bsqvac_half", None)
            bsqvac_objective = jnp.asarray(0.0)
            bsqvac_rms = jnp.asarray(0.0)
            bnormal_rms = jnp.asarray(0.0)
        with _jax_named_scope("vmec_jax.free_boundary.strict_update_one_step_from_trace"):
            step = strict_update_one_step_from_trace(
                state_in,
                static,
                trace,
                scalar_controls=control["step_scalars"],
                array_controls=control["step_arrays"],
                preconditioner_controls=control["step_preconditioner"] if "step_preconditioner" in control else None,
                freeb_bsqvac_half=freeb_bsqvac_half,
                enforce_edge=bool(enforce_edge),
                jit_preconditioner_apply=bool(jit_preconditioner_apply),
            )
        if bool(state_only_replay):
            return step["step"]["state_post"], {
                "state_reset": reset_to_trace_pre,
            }
        return step["step"]["state_post"], {
            "force": _tree_weighted_half_norm(step["force"], force_weight),
            "bsqvac": bsqvac_objective,
            "bsqvac_rms": bsqvac_rms,
            "bnormal_rms": bnormal_rms,
            "state_reset": reset_to_trace_pre,
        }

    def _branch_from_stacked_controls(
        trace: dict[str, Any],
        state: Any,
        coil_params: Any,
        control: dict[str, Any],
        replay_context: dict[str, Any] | None,
    ):
        if "step_preconditioner" not in control:
            raise ValueError("stacked step replay requires stackable preconditioner controls")
        reset_to_trace_pre = jnp.asarray(control["reset_to_trace_pre"], dtype=bool)
        stacked_state_pre = _step_control(control, "state_pre")
        if stacked_state_pre is None:
            raise ValueError("stacked step replay requires state_pre controls")
        state_in = jax.lax.cond(
            reset_to_trace_pre,
            lambda _: stacked_state_pre,
            lambda _: state,
            operand=None,
        )
        has_active_freeb_replay = trace.get("freeb_bsqvac_half") is not None and trace.get("freeb_nestor_trace") is not None
        if has_active_freeb_replay:
            if bool(freeze_freeb_bsqvac):
                freeb_bsqvac_half = jnp.asarray(trace["freeb_bsqvac_half"])
            else:
                with _jax_named_scope("vmec_jax.free_boundary.boundary_geometry"):
                    geometry = free_boundary_boundary_geometry_jax(
                        state_in,
                        static,
                        sample_nzeta=sample_nzeta,
                    )
                context = replay_context
                if context is None or tuple(int(v) for v in geometry["R"].shape) != (
                    int(context["ntheta"]),
                    int(context["nzeta"]),
                ):
                    with _jax_named_scope("vmec_jax.free_boundary.replay_context"):
                        context = direct_coil_boundary_replay_context(static, geometry)
                nestor_axes = _step_control(control, "freeb_nestor_axes")
                if nestor_axes is None:
                    with _jax_named_scope("vmec_jax.free_boundary.direct_coil_bsqvac_replay"):
                        replay = direct_coil_boundary_bsqvac_from_trace_jax(
                            coil_params,
                            geometry,
                            trace,
                            basis=context["basis"],
                            tables=context["tables"],
                            signgs=int(signgs),
                            nvper=int(context["nvper"]),
                            wint=jnp.asarray(context["wint"]),
                            include_analytic=bool(include_analytic),
                            include_diagnostics=not bool(state_only_replay),
                            include_mode_diagnostics=bool(include_mode_diagnostics),
                            freeze_vacuum_field=bool(freeze_vacuum_field),
                            nestor_solve_mode=str(nestor_solve_mode),
                            nestor_operator_solver=str(nestor_operator_solver),
                            nestor_operator_tol=float(nestor_operator_tol),
                            nestor_operator_atol=float(nestor_operator_atol),
                            nestor_operator_maxiter=nestor_operator_maxiter,
                            nestor_operator_restart=nestor_operator_restart,
                            coil_geometry=coil_geometry,
                        )
                else:
                    with _jax_named_scope("vmec_jax.free_boundary.direct_coil_bsqvac_replay"):
                        replay = direct_coil_boundary_bsqvac_jax(
                            coil_params,
                            R=geometry["R"],
                            Z=geometry["Z"],
                            phi=geometry["phi"],
                            Ru=geometry["Ru"],
                            Zu=geometry["Zu"],
                            Rv=geometry["Rv"],
                            Zv=geometry["Zv"],
                            ruu=geometry["ruu"],
                            ruv=geometry["ruv"],
                            rvv=geometry["rvv"],
                            zuu=geometry["zuu"],
                            zuv=geometry["zuv"],
                            zvv=geometry["zvv"],
                            basis=context["basis"],
                            tables=context["tables"],
                            signgs=int(signgs),
                            nvper=int(context["nvper"]),
                            br_add=jnp.asarray(nestor_axes["br_axis"]),
                            bp_add=jnp.asarray(nestor_axes["bp_axis"]),
                            bz_add=jnp.asarray(nestor_axes["bz_axis"]),
                            wint=jnp.asarray(context["wint"]),
                            include_analytic=bool(include_analytic),
                            include_diagnostics=not bool(state_only_replay),
                            include_mode_diagnostics=bool(include_mode_diagnostics),
                            vac_override=(
                                _direct_coil_trace_vacuum_field_override(trace)
                                if bool(freeze_vacuum_field)
                                else None
                            ),
                            coil_geometry=coil_geometry,
                            nestor_solve_mode=str(nestor_solve_mode),
                            nestor_operator_solver=str(nestor_operator_solver),
                            nestor_operator_tol=float(nestor_operator_tol),
                            nestor_operator_atol=float(nestor_operator_atol),
                            nestor_operator_maxiter=nestor_operator_maxiter,
                            nestor_operator_restart=nestor_operator_restart,
                        )
                freeb_bsqvac_half = replay["bsqvac"]
            if bool(state_only_replay):
                bsqvac_objective = jnp.asarray(0.0)
                bsqvac_rms = jnp.asarray(0.0)
                bnormal_rms = jnp.asarray(0.0)
            elif bool(freeze_freeb_bsqvac):
                bsqvac_objective = _weighted_half_norm(freeb_bsqvac_half, bsqvac_weight)
                bsqvac_rms = jnp.sqrt(jnp.mean(jnp.square(jnp.asarray(freeb_bsqvac_half))))
                bnormal_rms = jnp.asarray(0.0)
            else:
                bsqvac_objective = _weighted_half_norm(replay["bsqvac"], bsqvac_weight)
                bsqvac_rms = jnp.sqrt(jnp.mean(jnp.square(jnp.asarray(replay["bsqvac"]))))
                bnormal_rms = jnp.sqrt(jnp.mean(jnp.square(jnp.asarray(replay["vac"]["bnormal"]))))
        else:
            freeb_bsqvac_half = trace.get("freeb_bsqvac_half", None)
            bsqvac_objective = jnp.asarray(0.0)
            bsqvac_rms = jnp.asarray(0.0)
            bnormal_rms = jnp.asarray(0.0)
        preconditioner_use_precomputed_tridi = trace.get("preconditioner_use_precomputed_tridi")
        preconditioner_use_lax_tridi = trace.get("preconditioner_use_lax_tridi")
        step_step_controls = control["step_controls"]
        with _jax_named_scope("vmec_jax.free_boundary.strict_update_one_step_from_state"):
            step = strict_update_one_step_from_state(
                state_in,
                static,
                force_state_pre=step_step_controls.get("force_state_pre"),
                wout_like=trace["wout_like"],
                trig=trace["trig"],
                apply_lforbal=bool(trace["apply_lforbal"]),
                include_edge_residual=bool(trace["include_edge_residual"]),
                apply_m1_constraints=bool(trace["apply_m1_constraints"]),
                zero_m1=trace["zero_m1"],
                mats=control["step_preconditioner"]["precond_mats"],
                jmax=int(trace["precond_jmax"]),
                lam_prec=control["step_preconditioner"]["lam_prec"],
                w_mode_mn=control["step_preconditioner"]["w_mode_mn"],
                lambda_update_scale=control["step_scalars"]["lambda_update_scale"],
                dt_eff=control["step_scalars"]["dt_eff"],
                b1=control["step_scalars"]["b1"],
                fac=control["step_scalars"]["fac"],
                force_scale=control["step_scalars"]["force_scale"],
                flip_sign=control["step_scalars"]["flip_sign"],
                vRcc_before=control["step_arrays"]["vRcc_before"],
                vRss_before=control["step_arrays"]["vRss_before"],
                vZsc_before=control["step_arrays"]["vZsc_before"],
                vZcs_before=control["step_arrays"]["vZcs_before"],
                vLsc_before=control["step_arrays"]["vLsc_before"],
                vLcs_before=control["step_arrays"]["vLcs_before"],
                vRsc_before=control["step_arrays"].get("vRsc_before"),
                vRcs_before=control["step_arrays"].get("vRcs_before"),
                vZcc_before=control["step_arrays"].get("vZcc_before"),
                vZss_before=control["step_arrays"].get("vZss_before"),
                vLcc_before=control["step_arrays"].get("vLcc_before"),
                vLss_before=control["step_arrays"].get("vLss_before"),
                max_update_rms=control["step_scalars"]["max_update_rms_pre"],
                limit_update_rms=control["step_scalars"]["limit_update_rms"],
                divide_by_scalxc_for_update=control["step_scalars"]["divide_by_scalxc_for_update"],
                preconditioner_use_precomputed_tridi=(
                    None if preconditioner_use_precomputed_tridi is None else bool(preconditioner_use_precomputed_tridi)
                ),
                preconditioner_use_lax_tridi=(
                    None if preconditioner_use_lax_tridi is None else bool(preconditioner_use_lax_tridi)
                ),
                freeb_bsqvac_half=freeb_bsqvac_half,
                freeb_pres_scale=step_step_controls.get("freeb_pres_scale", trace.get("freeb_pres_scale", None)),
                constraint_rcon0=step_step_controls.get("constraint_rcon0", trace.get("constraint_rcon0")),
                constraint_zcon0=step_step_controls.get("constraint_zcon0", trace.get("constraint_zcon0")),
                constraint_tcon0=step_step_controls.get("constraint_tcon0", trace.get("constraint_tcon0")),
                constraint_precond_diag=step_step_controls.get(
                    "constraint_precond_diag",
                    trace.get("constraint_precond_diag"),
                ),
                constraint_tcon=step_step_controls.get("constraint_tcon", trace.get("constraint_tcon")),
                constraint_precond_active=step_step_controls.get(
                    "constraint_precond_active",
                    trace.get("constraint_precond_active"),
                ),
                constraint_tcon_active=step_step_controls.get(
                    "constraint_tcon_active",
                    trace.get("constraint_tcon_active"),
                ),
                enforce_edge=bool(enforce_edge),
                jit_preconditioner_apply=bool(jit_preconditioner_apply),
            )
        if bool(state_only_replay):
            return step["step"]["state_post"], {
                "state_reset": reset_to_trace_pre,
            }
        return step["step"]["state_post"], {
            "force": _tree_weighted_half_norm(step["force"], force_weight),
            "bsqvac": bsqvac_objective,
            "bsqvac_rms": bsqvac_rms,
            "bnormal_rms": bnormal_rms,
            "state_reset": reset_to_trace_pre,
        }

    def _make_step_fn(
        segment_traces: tuple[dict[str, Any], ...],
        *,
        index_offset: int = 0,
        stacked_step_controls: bool = False,
        accepted_only: bool = False,
    ):
        if bool(stacked_step_controls):
            representative_trace = segment_traces[0]
            representative_context = _precomputed_context_for_trace(representative_trace)

            def _step_fn(state, coil_params, control):
                if bool(accepted_only):
                    return _branch_from_stacked_controls(
                        representative_trace,
                        state,
                        coil_params,
                        control,
                        representative_context,
                    )
                do_propose = jnp.asarray(control["accept"], dtype=bool)

                def _propose(_unused):
                    return _branch_from_stacked_controls(
                        representative_trace,
                        state,
                        coil_params,
                        control,
                        representative_context,
                    )

                def _skip(_unused):
                    if bool(state_only_replay):
                        return state, {"state_reset": jnp.asarray(False, dtype=bool)}
                    return (
                        state,
                        {
                            "force": jnp.asarray(0.0),
                            "bsqvac": jnp.asarray(0.0),
                            "bsqvac_rms": jnp.asarray(0.0),
                            "bnormal_rms": jnp.asarray(0.0),
                            "state_reset": jnp.asarray(False, dtype=bool),
                        },
                    )

                return jax.lax.cond(do_propose, _propose, _skip, operand=None)

            return _step_fn

        branches = tuple(
            (
                lambda operand, trace=trace, replay_context=_precomputed_context_for_trace(trace): _branch_for_trace(
                    trace,
                    operand[0],
                    operand[1],
                    operand[2],
                    replay_context,
                )
            )
            for trace in segment_traces
        )

        def _step_fn(state, coil_params, control):
            step_index = jnp.asarray(control["step_index"], dtype=jnp.int32) - jnp.asarray(index_offset, dtype=jnp.int32)
            if bool(accepted_only):
                return jax.lax.switch(step_index, branches, (state, coil_params, control))
            do_propose = jnp.asarray(control["accept"], dtype=bool)

            def _propose(_unused):
                return jax.lax.switch(step_index, branches, (state, coil_params, control))

            def _skip(_unused):
                if bool(state_only_replay):
                    return state, {"state_reset": jnp.asarray(False, dtype=bool)}
                return (
                    state,
                    {
                        "force": jnp.asarray(0.0),
                        "bsqvac": jnp.asarray(0.0),
                        "bsqvac_rms": jnp.asarray(0.0),
                        "bnormal_rms": jnp.asarray(0.0),
                        "state_reset": jnp.asarray(False, dtype=bool),
                    },
                )

            return jax.lax.cond(do_propose, _propose, _skip, operand=None)

        return _step_fn

    def accept_fn(_state, _proposed_state, _params, control, _aux):
        return control["accept"]

    def converged_fn(_accepted_state, _params, control, _aux):
        return control["done"]

    segment_preconditioner_controls_stacked: tuple[bool, ...] = ()
    accepted_only_fast_path_segments: tuple[bool, ...] = ()
    if use_stacked_step_controls:
        if replay_plan.get("segment_source") != "step_policy":
            replay_plan = direct_coil_accepted_trace_controller_replay_plan(
                trace_seq,
                static=static,
                accept_mask=accept_mask,
                done_mask=done_mask,
                max_steps=None,
                use_stacked_step_controls=True,
                use_accepted_only_fast_path=bool(use_accepted_only_fast_path),
            )
            controls = replay_plan["controls"]
            preconditioner_controls = replay_plan["preconditioner_controls"]
            step_policy_segments = replay_plan["step_policy_segments"]
            context_cache = dict(replay_plan.get("boundary_replay_contexts_by_shape", {}))
        control_segments = tuple(replay_plan["control_segments"])
        segment_preconditioner_controls_stacked = tuple(replay_plan["preconditioner_controls_segment_stacked"])
        accepted_only_fast_path_segments = tuple(replay_plan["accepted_only_fast_path_segments"])
        step_fns = tuple(
            _make_step_fn(
                trace_seq[int(segment["start"]) : int(segment["stop"])],
                index_offset=int(segment["start"]),
                stacked_step_controls=True,
                accepted_only=bool(accepted_only_fast_path_segments[index]),
            )
            for index, segment in enumerate(step_policy_segments)
        )
        segmented_runner = (
            jax_visible_segmented_state_only_accepted_nonlinear_controller_jax
            if bool(state_only_replay)
            else jax_visible_segmented_accepted_nonlinear_controller_jax
        )
        run = segmented_runner(
            step_fns,
            accept_fn,
            converged_fn,
            initial_state,
            params,
            control_segments,
            checkpoint_steps=checkpoint_steps,
            accepted_only_segments=accepted_only_fast_path_segments,
            unroll_accepted_only_segments_below=int(unroll_accepted_only_segments_below),
        )
    elif use_preconditioner_policy_segments:
        if replay_plan.get("segment_source") != "preconditioner_policy" or bool(
            plan_options.get("use_segment_preconditioner_controls", False)
        ) != bool(use_segment_preconditioner_controls):
            replay_plan = direct_coil_accepted_trace_controller_replay_plan(
                trace_seq,
                static=static,
                accept_mask=accept_mask,
                done_mask=done_mask,
                max_steps=None,
                use_preconditioner_policy_segments=True,
                use_segment_preconditioner_controls=bool(use_segment_preconditioner_controls),
                use_accepted_only_fast_path=bool(use_accepted_only_fast_path),
            )
            controls = replay_plan["controls"]
            preconditioner_controls = replay_plan["preconditioner_controls"]
            preconditioner_policy_segments = replay_plan["preconditioner_policy_segments"]
            context_cache = dict(replay_plan.get("boundary_replay_contexts_by_shape", {}))
        control_segments = tuple(replay_plan["control_segments"])
        segment_preconditioner_controls_stacked = tuple(replay_plan["preconditioner_controls_segment_stacked"])
        accepted_only_fast_path_segments = tuple(replay_plan["accepted_only_fast_path_segments"])
        step_fns = tuple(
            _make_step_fn(
                trace_seq[int(segment["start"]) : int(segment["stop"])],
                index_offset=int(segment["start"]),
                accepted_only=bool(accepted_only_fast_path_segments[index]),
            )
            for index, segment in enumerate(preconditioner_policy_segments)
        )
        segmented_runner = (
            jax_visible_segmented_state_only_accepted_nonlinear_controller_jax
            if bool(state_only_replay)
            else jax_visible_segmented_accepted_nonlinear_controller_jax
        )
        run = segmented_runner(
            step_fns,
            accept_fn,
            converged_fn,
            initial_state,
            params,
            control_segments,
            checkpoint_steps=checkpoint_steps,
            accepted_only_segments=accepted_only_fast_path_segments,
            unroll_accepted_only_segments_below=int(unroll_accepted_only_segments_below),
        )
    else:
        accepted_only_fast_path_segments = (
            bool(use_accepted_only_fast_path)
            and _accepted_trace_segment_is_unconditionally_accepted(effective_masks, start=0, stop=len(trace_seq)),
        )
        step_fn = _make_step_fn(trace_seq, accepted_only=accepted_only_fast_path_segments[0])
        if accepted_only_fast_path_segments[0]:
            use_unrolled = int(unroll_accepted_only_segments_below) > 0 and len(trace_seq) <= int(
                unroll_accepted_only_segments_below
            )
            if bool(state_only_replay):
                accepted_only_runner = (
                    jax_visible_unrolled_state_only_accepted_only_nonlinear_controller_jax
                    if use_unrolled
                    else jax_visible_state_only_accepted_only_nonlinear_controller_jax
                )
            else:
                accepted_only_runner = (
                    jax_visible_unrolled_accepted_only_nonlinear_controller_jax
                    if use_unrolled
                    else jax_visible_accepted_only_nonlinear_controller_jax
                )
            run = accepted_only_runner(
                step_fn,
                converged_fn,
                initial_state,
                params,
                controls,
                checkpoint_steps=checkpoint_steps,
            )
        else:
            accepted_runner = (
                jax_visible_state_only_accepted_nonlinear_controller_jax
                if bool(state_only_replay)
                else jax_visible_accepted_nonlinear_controller_jax
            )
            run = accepted_runner(
                step_fn,
                accept_fn,
                converged_fn,
                initial_state,
                params,
                controls,
                checkpoint_steps=checkpoint_steps,
            )
    return _accepted_controller_replay_result(
        run=run,
        controls=controls,
        scalar_controls=scalar_controls,
        array_controls=array_controls,
        step_controls=step_controls,
        preconditioner_controls=preconditioner_controls,
        preconditioner_controls_stacked=bool(preconditioner_controls_stacked),
        preconditioner_policy_segments=preconditioner_policy_segments,
        preconditioner_policy_segment_summary=preconditioner_policy_segment_summary,
        step_policy_segments=step_policy_segments,
        step_policy_segment_summary=step_policy_segment_summary,
        segment_preconditioner_controls_stacked=segment_preconditioner_controls_stacked,
        use_preconditioner_policy_segments=bool(use_preconditioner_policy_segments),
        use_stacked_step_controls=bool(use_stacked_step_controls),
        accepted_only_fast_path_segments=accepted_only_fast_path_segments,
        state_weight=state_weight,
        include_replay_aux=bool(include_replay_aux),
        state_only_replay=bool(state_only_replay),
    )


def direct_coil_complete_solve_trace(
    input_path: Any,
    params: Any,
    *,
    init_kwargs: dict[str, Any] | None = None,
    solve_kwargs: dict[str, Any] | None = None,
    require_active_trace: bool = True,
) -> dict[str, Any]:
    """Run a direct-coil free-boundary solve and return accepted traces.

    This is a validation helper for phase-2 same-branch adjoint promotion.  It
    runs the same direct-coil initialization plus accepted residual iteration
    used by the complete-solve finite-difference gates and returns the
    initialization result, final solve result, and recorded adjoint traces.

    The helper intentionally does not decide whether perturbations are on the
    same adaptive branch.  Use
    :func:`direct_coil_same_branch_complete_solve_fd_report` or
    :func:`direct_coil_accepted_trace_fingerprint_delta` for that gate.
    """

    from .driver import run_free_boundary
    from .solve import solve_fixed_boundary_residual_iter

    init_options: dict[str, Any] = {
        "use_initial_guess": True,
        "verbose": False,
        "external_field_provider_kind": "direct_coils",
        "external_field_provider_params": params,
    }
    if init_kwargs:
        init_options.update(init_kwargs)
    init = run_free_boundary(input_path, **init_options)

    solve_options: dict[str, Any] = {
        "max_iter": 2,
        "ftol": 1.0e-8,
        "vmec2000_control": True,
        "auto_flip_force": False,
        "use_direct_fallback": True,
        "verbose": False,
        "verbose_vmec2000_table": False,
        "jit_forces": False,
        "use_scan": False,
        "host_update_assembly": False,
        "adjoint_trace": True,
        "adjoint_trace_mode": "full",
        "external_field_provider_kind": "direct_coils",
        "external_field_provider_params": params,
        "free_boundary_activate_fsq": 1.0e99,
    }
    if solve_kwargs:
        solve_options.update(solve_kwargs)
    solve_options["external_field_provider_params"] = params
    result = solve_fixed_boundary_residual_iter(
        init.state,
        init.static,
        indata=init.indata,
        signgs=init.signgs,
        **solve_options,
    )
    traces = list(result.diagnostics.get("adjoint_step_trace", []))
    if not traces:
        raise RuntimeError("direct-coil solve did not record adjoint_step_trace")
    active_trace = any(trace.get("freeb_bsqvac_half") is not None for trace in traces)
    if bool(require_active_trace) and not active_trace:
        raise RuntimeError("direct-coil solve did not record an active free-boundary trace")
    return {
        "init": init,
        "result": result,
        "traces": traces,
        "params": params,
        "active_trace": bool(active_trace),
    }


def direct_coil_same_branch_complete_solve_fd_report(
    input_path: Any,
    base_params: Any,
    *,
    params_for: Any,
    objective_fn: Any,
    eps: float = 1.0e-4,
    init_kwargs: dict[str, Any] | None = None,
    solve_kwargs: dict[str, Any] | None = None,
    fingerprint_rtol: float = 1.0e-6,
    fingerprint_atol: float = 1.0e-9,
    require_active_trace: bool = True,
) -> dict[str, Any]:
    """Return same-branch complete-solve finite-difference diagnostics.

    ``params_for(scale)`` must return the coil parameters for ``base + scale *
    direction``.  ``objective_fn(payload)`` receives each payload returned by
    :func:`direct_coil_complete_solve_trace` and returns either one scalar or a
    mapping of scalar diagnostics.  The result contains raw base/plus/minus
    payloads, branch fingerprint deltas, scalar values, and central
    finite-difference slopes.  For backward compatibility, ``values`` reports
    the primary scalar.  ``objective_values`` reports every scalar returned by
    ``objective_fn``.

    This helper is deliberately a validation seam rather than a production
    adjoint: it rejects branch changes using accepted-trace and residual
    controller fingerprints and leaves the differentiated frozen-branch replay
    to the caller.
    """

    from .discrete_adjoint import residual_branch_fingerprint

    eps_f = float(eps)
    if eps_f == 0.0:
        raise ValueError("eps must be nonzero")
    base = direct_coil_complete_solve_trace(
        input_path,
        base_params,
        init_kwargs=init_kwargs,
        solve_kwargs=solve_kwargs,
        require_active_trace=require_active_trace,
    )
    plus = direct_coil_complete_solve_trace(
        input_path,
        params_for(eps_f),
        init_kwargs=init_kwargs,
        solve_kwargs=solve_kwargs,
        require_active_trace=require_active_trace,
    )
    minus = direct_coil_complete_solve_trace(
        input_path,
        params_for(-eps_f),
        init_kwargs=init_kwargs,
        solve_kwargs=solve_kwargs,
        require_active_trace=require_active_trace,
    )
    plus_branch = direct_coil_accepted_trace_fingerprint_delta(
        base["traces"],
        plus["traces"],
        rtol=float(fingerprint_rtol),
        atol=float(fingerprint_atol),
    )
    minus_branch = direct_coil_accepted_trace_fingerprint_delta(
        base["traces"],
        minus["traces"],
        rtol=float(fingerprint_rtol),
        atol=float(fingerprint_atol),
    )
    base_fingerprint = direct_coil_accepted_trace_fingerprint(base["traces"])
    plus_fingerprint = direct_coil_accepted_trace_fingerprint(plus["traces"])
    minus_fingerprint = direct_coil_accepted_trace_fingerprint(minus["traces"])
    base_residual_fingerprint = residual_branch_fingerprint(base["result"])
    plus_residual_fingerprint = residual_branch_fingerprint(plus["result"])
    minus_residual_fingerprint = residual_branch_fingerprint(minus["result"])
    same_residual_branch = bool(
        base_residual_fingerprint == plus_residual_fingerprint
        and base_residual_fingerprint == minus_residual_fingerprint
    )
    trace_replay_diagnostics = {
        "base": free_boundary_adjoint_trace_replay_diagnostics(base["traces"]),
        "plus": free_boundary_adjoint_trace_replay_diagnostics(plus["traces"]),
        "minus": free_boundary_adjoint_trace_replay_diagnostics(minus["traces"]),
    }
    base_values = _complete_solve_objective_values(objective_fn(base))
    plus_values = _complete_solve_objective_values(objective_fn(plus))
    minus_values = _complete_solve_objective_values(objective_fn(minus))
    if base_values.keys() != plus_values.keys() or base_values.keys() != minus_values.keys():
        raise ValueError("objective_fn returned different scalar keys for base/plus/minus solves")
    primary_key = "objective" if "objective" in base_values else next(iter(base_values))
    objective_values = {
        key: {
            "base": float(base_values[key]),
            "plus": float(plus_values[key]),
            "minus": float(minus_values[key]),
            "central_fd_directional": float((plus_values[key] - minus_values[key]) / (2.0 * eps_f)),
        }
        for key in base_values
    }
    return {
        "base": base,
        "plus": plus,
        "minus": minus,
        "branch_compatibility": {
            "same_branch": bool(plus_branch["compatible"] and minus_branch["compatible"] and same_residual_branch),
            "same_accepted_trace_branch": bool(plus_branch["compatible"] and minus_branch["compatible"]),
            "same_residual_branch": same_residual_branch,
            "plus": plus_branch,
            "minus": minus_branch,
            "base_fingerprint": base_fingerprint,
            "plus_fingerprint": plus_fingerprint,
            "minus_fingerprint": minus_fingerprint,
            "base_residual_fingerprint": base_residual_fingerprint,
            "plus_residual_fingerprint": plus_residual_fingerprint,
            "minus_residual_fingerprint": minus_residual_fingerprint,
        },
        "trace_replay_diagnostics": trace_replay_diagnostics,
        "primary_objective": primary_key,
        "values": objective_values[primary_key],
        "objective_values": objective_values,
    }


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
    frozen_fd = float(np.asarray(check["fd_directional"], dtype=float))
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
        frozen_fd = float(np.asarray(frozen_fd_directionals[index], dtype=float))
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

    timings: dict[str, float] = {}
    total_start = time.perf_counter()
    if complete_payload is None:
        if input_path is None or params is None:
            raise ValueError("input_path and params are required when complete_payload is not supplied")
        t0 = time.perf_counter()
        payload = direct_coil_complete_solve_trace(
            input_path,
            params,
            init_kwargs=init_kwargs,
            solve_kwargs=solve_kwargs,
            require_active_trace=require_active_trace,
        )
        timings["complete_solve_trace_wall_s"] = float(time.perf_counter() - t0)
    else:
        t0 = time.perf_counter()
        payload = dict(complete_payload)
        if params is None:
            params = payload.get("params")
        if params is None:
            raise ValueError("params must be supplied when complete_payload does not contain params")
        timings["payload_copy_wall_s"] = float(time.perf_counter() - t0)

    traces = tuple(payload.get("traces", ()))
    if not traces:
        raise ValueError("complete payload contains no accepted traces")
    active_trace = any(trace.get("freeb_bsqvac_half") is not None for trace in traces)
    if bool(require_active_trace) and not active_trace:
        raise RuntimeError("complete payload contains no active free-boundary trace")
    init = payload.get("init")
    if init is None:
        raise ValueError("complete payload is missing the initialization result")

    t0 = time.perf_counter()
    values = _complete_solve_objective_values(
        scalar_fn(payload) if production_values is None else production_values
    )
    timings["production_scalar_eval_wall_s"] = float(time.perf_counter() - t0)
    production_values_source = "scalar_fn" if production_values is None else "precomputed"
    key = str(scalar_key or ("objective" if "objective" in values else next(iter(values))))
    if key not in values:
        raise KeyError(f"scalar_key {key!r} not returned by scalar_fn")

    replay_options: dict[str, Any] = {
        "static": init.static,
        "traces": traces,
        "signgs": int(init.signgs),
        "state_weight": 0.0,
        "bsqvac_weight": 0.0,
        "force_weight": 0.0,
        "enforce_edge": False,
        "use_preconditioner_policy_segments": True,
        "use_stacked_step_controls": True,
        "include_replay_aux": False,
        "unroll_accepted_only_segments_below": 8,
    }
    if replay_kwargs:
        replay_options.update(replay_kwargs)
    replay_traces_for_scalars = tuple(replay_options.get("traces", traces))
    replay_branch_metadata = direct_coil_accepted_trace_branch_metadata(
        replay_traces_for_scalars,
        accept_mask=replay_options.get("accept_mask"),
        done_mask=replay_options.get("done_mask"),
        max_steps=replay_options.get("max_steps"),
        json_safe=True,
    )
    controller_slot_summary = direct_coil_accepted_trace_controller_slot_summary(replay_branch_metadata)
    replay_payload_for_scalars = payload if replay_payload is None else replay_payload
    replay_payload_source = "complete_payload" if replay_payload is None else "user"
    replay_plan_for_scalars = replay_plan
    if replay_plan_for_scalars is None and bool(use_replay_plan):
        t0 = time.perf_counter()
        replay_plan_for_scalars = direct_coil_accepted_trace_controller_replay_plan(
            replay_traces_for_scalars,
            static=init.static,
            accept_mask=replay_options.get("accept_mask"),
            done_mask=replay_options.get("done_mask"),
            max_steps=replay_options.get("max_steps"),
            use_preconditioner_policy_segments=bool(
                replay_options.get("use_preconditioner_policy_segments", False)
            ),
            use_segment_preconditioner_controls=bool(
                replay_options.get("use_segment_preconditioner_controls", False)
            ),
            use_stacked_step_controls=bool(replay_options.get("use_stacked_step_controls", False)),
            use_accepted_only_fast_path=bool(replay_options.get("use_accepted_only_fast_path", True)),
        )
        timings["replay_plan_build_wall_s"] = float(time.perf_counter() - t0)
    else:
        timings["replay_plan_build_wall_s"] = 0.0

    t0 = time.perf_counter()
    if bool(include_replay_graph_metadata):
        graph_metadata = direct_coil_accepted_trace_replay_graph_metadata(
            replay_traces_for_scalars,
            static=init.static,
            accept_mask=replay_options.get("accept_mask"),
            done_mask=replay_options.get("done_mask"),
            max_steps=replay_options.get("max_steps"),
            sample_nzeta=replay_options.get("sample_nzeta"),
            include_analytic=bool(replay_options.get("include_analytic", True)),
            use_stacked_step_controls=bool(replay_options.get("use_stacked_step_controls", False)),
            use_accepted_only_fast_path=bool(replay_options.get("use_accepted_only_fast_path", True)),
            json_safe=True,
        )
    else:
        graph_metadata = {
            "contract": "fixed accepted-branch replay graph metadata",
            "omitted": True,
            "reason": "include_replay_graph_metadata=False",
            "differentiates_adaptive_controller": False,
        }
    timings["replay_graph_metadata_wall_s"] = float(time.perf_counter() - t0)

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
        "replay_option_flags": {
            "use_preconditioner_policy_segments": bool(
                replay_options.get("use_preconditioner_policy_segments", False)
            ),
            "use_stacked_step_controls": bool(replay_options.get("use_stacked_step_controls", False)),
            "use_accepted_only_fast_path": bool(replay_options.get("use_accepted_only_fast_path", True)),
            "use_replay_plan": bool(replay_plan_for_scalars is not None),
            "include_replay_aux": bool(replay_options.get("include_replay_aux", True)),
            "include_analytic": bool(replay_options.get("include_analytic", True)),
            "include_mode_diagnostics": bool(replay_options.get("include_mode_diagnostics", False)),
            "nestor_solve_mode": str(replay_options.get("nestor_solve_mode", "dense")),
            "nestor_operator_solver": str(replay_options.get("nestor_operator_solver", "gmres")),
            "nestor_operator_tol": float(replay_options.get("nestor_operator_tol", 1.0e-11)),
            "nestor_operator_atol": float(replay_options.get("nestor_operator_atol", 1.0e-13)),
            "nestor_operator_maxiter": (
                None
                if replay_options.get("nestor_operator_maxiter") is None
                else int(replay_options.get("nestor_operator_maxiter"))
            ),
            "nestor_operator_restart": (
                None
                if replay_options.get("nestor_operator_restart") is None
                else int(replay_options.get("nestor_operator_restart"))
            ),
            "freeze_vacuum_field": bool(replay_options.get("freeze_vacuum_field", False)),
            "freeze_freeb_bsqvac": bool(replay_options.get("freeze_freeb_bsqvac", False)),
            "state_only_replay": bool(replay_options.get("state_only_replay", False)),
            "jit_preconditioner_apply": bool(replay_options.get("jit_preconditioner_apply", True)),
            "unroll_accepted_only_segments_below": int(
                replay_options.get("unroll_accepted_only_segments_below", 0)
            ),
            "replay_ad_mode": ad_mode,
        },
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
    """Return production-forward branch-local values and a scalar Jacobian.

    This is the vector-valued counterpart of
    :func:`direct_coil_run_free_boundary_branch_local_scalar_value_and_grad_jax`.
    The values are evaluated from a real direct-coil complete solve payload,
    while the Jacobian is computed by replaying the fixed accepted branch with
    a vector-output custom-VJP seam.  The contract is intentionally narrow: it
    differentiates direct-coil parameters through the saved accepted branch and
    does not differentiate adaptive host-controller branch changes.

    If ``direction_params`` is supplied, the helper computes only the
    directional derivatives ``J @ direction_params`` using ``jax.jvp`` instead
    of materializing the full Jacobian.  This is the fast path for production
    validation reports that compare against one complete-solve central
    finite-difference direction.

    ``scalar_fn(payload)`` must return a mapping of production scalar values.
    Callers that already have the production base values can pass
    ``production_values`` to avoid recomputing them from ``scalar_fn``.
    ``replay_scalar_fns`` maps the same scalar keys to callables of the form
    ``fn(replay, payload)`` that evaluate those scalars from the JAX-visible
    accepted-controller replay.  ``replay_payload`` can be supplied to pass a
    slim context into those functions, avoiding closure capture of a full
    complete-solve payload during cold replay/JVP graph construction.  Set
    ``include_payload=False`` for production reports that only need scalar
    values/derivatives and should not retain the full complete-solve payload.
    Set ``include_replay_graph_metadata=False`` when a compact production
    report does not need structural replay metadata.
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

    timings: dict[str, float] = {}
    total_start = time.perf_counter()
    if complete_payload is None:
        if input_path is None or params is None:
            raise ValueError("input_path and params are required when complete_payload is not supplied")
        t0 = time.perf_counter()
        payload = direct_coil_complete_solve_trace(
            input_path,
            params,
            init_kwargs=init_kwargs,
            solve_kwargs=solve_kwargs,
            require_active_trace=require_active_trace,
        )
        timings["complete_solve_trace_wall_s"] = float(time.perf_counter() - t0)
    else:
        t0 = time.perf_counter()
        payload = dict(complete_payload)
        if params is None:
            params = payload.get("params")
        if params is None:
            raise ValueError("params must be supplied when complete_payload does not contain params")
        timings["payload_copy_wall_s"] = float(time.perf_counter() - t0)

    traces = tuple(payload.get("traces", ()))
    if not traces:
        raise ValueError("complete payload contains no accepted traces")
    active_trace = any(trace.get("freeb_bsqvac_half") is not None for trace in traces)
    if bool(require_active_trace) and not active_trace:
        raise RuntimeError("complete payload contains no active free-boundary trace")
    init = payload.get("init")
    if init is None:
        raise ValueError("complete payload is missing the initialization result")

    t0 = time.perf_counter()
    all_values = _complete_solve_objective_values(
        scalar_fn(payload) if production_values is None else production_values
    )
    timings["production_scalar_eval_wall_s"] = float(time.perf_counter() - t0)
    production_values_source = "scalar_fn" if production_values is None else "precomputed"
    keys = tuple(str(key) for key in (scalar_keys if scalar_keys is not None else tuple(replay_scalar_fns)))
    if not keys:
        raise ValueError("scalar_keys must contain at least one scalar")
    for key in keys:
        if key not in all_values:
            raise KeyError(f"scalar_key {key!r} not returned by scalar_fn")
        if key not in replay_scalar_fns:
            raise KeyError(f"scalar_key {key!r} not present in replay_scalar_fns")

    replay_options: dict[str, Any] = {
        "static": init.static,
        "traces": traces,
        "signgs": int(init.signgs),
        "state_weight": 0.0,
        "bsqvac_weight": 0.0,
        "force_weight": 0.0,
        "enforce_edge": False,
        "use_preconditioner_policy_segments": True,
        "use_stacked_step_controls": True,
        "include_replay_aux": False,
        "unroll_accepted_only_segments_below": 8,
    }
    if replay_kwargs:
        replay_options.update(replay_kwargs)
    replay_traces_for_scalars = tuple(replay_options.get("traces", traces))
    replay_branch_metadata = direct_coil_accepted_trace_branch_metadata(
        replay_traces_for_scalars,
        accept_mask=replay_options.get("accept_mask"),
        done_mask=replay_options.get("done_mask"),
        max_steps=replay_options.get("max_steps"),
        json_safe=True,
    )
    controller_slot_summary = direct_coil_accepted_trace_controller_slot_summary(replay_branch_metadata)
    replay_payload_for_scalars = payload if replay_payload is None else replay_payload
    replay_payload_source = "complete_payload" if replay_payload is None else "user"
    replay_plan_for_scalars = replay_plan
    if replay_plan_for_scalars is None and bool(use_replay_plan):
        t0 = time.perf_counter()
        replay_plan_for_scalars = direct_coil_accepted_trace_controller_replay_plan(
            replay_traces_for_scalars,
            static=init.static,
            accept_mask=replay_options.get("accept_mask"),
            done_mask=replay_options.get("done_mask"),
            max_steps=replay_options.get("max_steps"),
            use_preconditioner_policy_segments=bool(
                replay_options.get("use_preconditioner_policy_segments", False)
            ),
            use_segment_preconditioner_controls=bool(
                replay_options.get("use_segment_preconditioner_controls", False)
            ),
            use_stacked_step_controls=bool(replay_options.get("use_stacked_step_controls", False)),
            use_accepted_only_fast_path=bool(replay_options.get("use_accepted_only_fast_path", True)),
        )
        timings["replay_plan_build_wall_s"] = float(time.perf_counter() - t0)
    else:
        timings["replay_plan_build_wall_s"] = 0.0

    t0 = time.perf_counter()
    if bool(include_replay_graph_metadata):
        graph_metadata = direct_coil_accepted_trace_replay_graph_metadata(
            replay_traces_for_scalars,
            static=init.static,
            accept_mask=replay_options.get("accept_mask"),
            done_mask=replay_options.get("done_mask"),
            max_steps=replay_options.get("max_steps"),
            sample_nzeta=replay_options.get("sample_nzeta"),
            include_analytic=bool(replay_options.get("include_analytic", True)),
            use_stacked_step_controls=bool(replay_options.get("use_stacked_step_controls", False)),
            use_accepted_only_fast_path=bool(replay_options.get("use_accepted_only_fast_path", True)),
            json_safe=True,
        )
    else:
        graph_metadata = {
            "contract": "fixed accepted-branch replay graph metadata",
            "omitted": True,
            "reason": "include_replay_graph_metadata=False",
            "differentiates_adaptive_controller": False,
        }
    timings["replay_graph_metadata_wall_s"] = float(time.perf_counter() - t0)

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

    derivative_mode = "full_jacobian_vjp"
    jacobian = None
    gradients: dict[str, Any] = {}
    directional_values = None
    directional_fast_path = "none"
    directional_uses_fixed_coil_geometry = False
    current_only_geometry_source = "none"
    if direction_params is not None:
        derivative_mode = "directional_jvp"
        current_only_direction = False
        current_direction_leaf = None
        current_base_leaf = None
        try:
            from .external_fields import CoilFieldParams

            if isinstance(params, CoilFieldParams) and isinstance(direction_params, CoilFieldParams):
                direction_dofs = np.asarray(direction_params.base_curve_dofs, dtype=float)
                current_only_direction = not np.any(direction_dofs)
                if current_only_direction:
                    current_base_leaf = jnp.asarray(params.base_currents)
                    current_direction_leaf = jnp.asarray(direction_params.base_currents)
        except Exception:
            current_only_direction = False
            current_direction_leaf = None
            current_base_leaf = None

        if current_only_direction and current_base_leaf is not None and current_direction_leaf is not None:
            directional_fast_path = "current_only"
            directional_uses_fixed_coil_geometry = True
            from .external_fields import build_coil_field_geometry, apply_stellarator_symmetry_to_currents

            if current_only_coil_geometry is None:
                t0 = time.perf_counter()
                fixed_gamma, fixed_gamma_dash, _fixed_currents = build_coil_field_geometry(params)
                timings["current_only_coil_geometry_build_wall_s"] = float(time.perf_counter() - t0)
                current_only_geometry_source = "built"
            else:
                fixed_gamma, fixed_gamma_dash = current_only_coil_geometry[:2]
                timings["current_only_coil_geometry_build_wall_s"] = 0.0
                current_only_geometry_source = "cached"

            def _fixed_geometry_for_currents(base_currents):
                expanded_currents = params.current_scale * apply_stellarator_symmetry_to_currents(
                    base_currents,
                    nfp=params.nfp,
                    stellsym=params.stellsym,
                )
                return fixed_gamma, fixed_gamma_dash, expanded_currents

            def _replay_scalars_current_only(base_currents):
                replay = direct_coil_accepted_trace_controller_replay_objective_jax(
                    params.with_arrays(base_currents=base_currents),
                    replay_traces_for_scalars[0]["state_pre"],
                    replay_plan=replay_plan_for_scalars,
                    coil_geometry=_fixed_geometry_for_currents(base_currents),
                    **replay_options,
                )
                return jnp.asarray([fn(replay) for fn in scalar_fn_seq])

            jvp_primal = (current_base_leaf,)
            jvp_tangent = (current_direction_leaf,)
            jvp_fn = _replay_scalars_current_only
        else:
            jvp_primal = (params,)
            jvp_tangent = (direction_params,)
            jvp_fn = _replay_scalars

        t0 = time.perf_counter()
        replay_values, directional_values = jax.jvp(
            jvp_fn,
            jvp_primal,
            jvp_tangent,
        )
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
        replay_values, pullback = jax.vjp(_replay_scalars, params)
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
    values = {key: float(all_values[key]) for key in keys}
    replay_value_map = {key: replay_values[index] for index, key in enumerate(keys)}
    base_abs_delta = {
        key: abs(float(np.asarray(replay_values[index], dtype=float)) - float(values[key]))
        for index, key in enumerate(keys)
    }
    directional_derivatives = (
        None
        if directional_values is None
        else {key: directional_values[index] for index, key in enumerate(keys)}
    )
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
    timings["total_wall_s"] = float(time.perf_counter() - total_start)
    return {
        "contract": "production-forward branch-local run_free_boundary scalar values/Jacobian",
        "uses_production_forward": True,
        "differentiates_adaptive_controller": False,
        "differentiates_run_free_boundary": False,
        "differentiates_fixed_accepted_branch": True,
        "replay_ad_mode": ad_mode,
        "derivative_mode": derivative_mode,
        "scalar_keys": keys,
        "values": values,
        "all_values": all_values,
        "production_values_source": production_values_source,
        "replay_payload_source": replay_payload_source,
        "replay_values": replay_values,
        "replay_value_map": replay_value_map,
        "base_abs_delta": base_abs_delta,
        "max_base_abs_delta": max(base_abs_delta.values()) if base_abs_delta else 0.0,
        "jacobian": jacobian,
        "grads": gradients,
        "directional_derivatives": directional_derivatives,
        "payload": payload if bool(include_payload) else None,
        "includes_payload": bool(include_payload),
        "includes_replay_graph_metadata": bool(include_replay_graph_metadata),
        "timings": timings,
        "trace_replay_diagnostics": diagnostics,
        "replay_graph_metadata": graph_metadata,
        "replay_branch_metadata": replay_branch_metadata,
        "controller_slot_summary": controller_slot_summary,
        "replay_option_flags": {
            "use_preconditioner_policy_segments": bool(
                replay_options.get("use_preconditioner_policy_segments", False)
            ),
            "use_stacked_step_controls": bool(replay_options.get("use_stacked_step_controls", False)),
            "use_accepted_only_fast_path": bool(replay_options.get("use_accepted_only_fast_path", True)),
            "use_replay_plan": bool(replay_plan_for_scalars is not None),
            "include_replay_aux": bool(replay_options.get("include_replay_aux", True)),
            "include_analytic": bool(replay_options.get("include_analytic", True)),
            "include_mode_diagnostics": bool(replay_options.get("include_mode_diagnostics", False)),
            "nestor_solve_mode": str(replay_options.get("nestor_solve_mode", "dense")),
            "nestor_operator_solver": str(replay_options.get("nestor_operator_solver", "gmres")),
            "nestor_operator_tol": float(replay_options.get("nestor_operator_tol", 1.0e-11)),
            "nestor_operator_atol": float(replay_options.get("nestor_operator_atol", 1.0e-13)),
            "nestor_operator_maxiter": (
                None
                if replay_options.get("nestor_operator_maxiter") is None
                else int(replay_options.get("nestor_operator_maxiter"))
            ),
            "nestor_operator_restart": (
                None
                if replay_options.get("nestor_operator_restart") is None
                else int(replay_options.get("nestor_operator_restart"))
            ),
            "freeze_vacuum_field": bool(replay_options.get("freeze_vacuum_field", False)),
            "freeze_freeb_bsqvac": bool(replay_options.get("freeze_freeb_bsqvac", False)),
            "state_only_replay": bool(replay_options.get("state_only_replay", False)),
            "jit_preconditioner_apply": bool(replay_options.get("jit_preconditioner_apply", True)),
            "unroll_accepted_only_segments_below": int(
                replay_options.get("unroll_accepted_only_segments_below", 0)
            ),
            "replay_ad_mode": ad_mode,
            "directional_jvp_fast_path": directional_fast_path,
            "directional_uses_fixed_coil_geometry": directional_uses_fixed_coil_geometry,
            "current_only_coil_geometry_source": current_only_geometry_source,
        },
    }


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


def direct_coil_projected_mode_fixed_point_jax(
    params: Any,
    initial_state: Any,
    *,
    boundary_from_state: Any,
    update_from_response: Any,
    mode_matrix: Any,
    sin_basis: Any,
    xmpot: Any,
    n_raw: Any,
    imirr: Any | None = None,
    imirr_full: Any | None = None,
    cos_basis: Any | None = None,
    onp: float = 1.0,
    lasym: bool = False,
    nuv3: int | None = None,
    nuv_full: int | None = None,
    max_iter: int = 10,
    damping: float = 1.0,
    symmetric: bool = False,
) -> dict[str, Any]:
    """Solve a small direct-coil free-boundary fixed-point validation loop.

    ``boundary_from_state(state)`` must return a mapping with ``R``, ``Z``,
    ``phi``, ``Ru``, ``Zu``, ``Rv``, and ``Zv`` arrays.  At each fixed-point
    step this helper samples the direct Biot-Savart field on that moving
    boundary, projects it into VMEC boundary channels, projects the normal
    source into mode space, solves the dense vacuum mode system, and passes the
    response to ``update_from_response(state, response, vac, boundary, params)``.

    This is a production-adjacent phase-2 validation primitive.  It exercises
    the same dependency graph as a free-boundary coil solve at tiny dense scale,
    while keeping the true production ``run_free_boundary`` loop out of scope
    until that loop is made JAX-visible or receives its own custom VJP.
    """

    from .external_fields import sample_coil_field_cylindrical

    required = ("R", "Z", "phi", "Ru", "Zu", "Rv", "Zv")

    def _mode_response_for_state(state, coil_params):
        boundary = boundary_from_state(state)
        missing = [name for name in required if name not in boundary]
        if missing:
            raise ValueError(f"boundary_from_state missing keys: {missing}")
        br, bp, bz = sample_coil_field_cylindrical(
            coil_params,
            jnp.asarray(boundary["R"]),
            jnp.asarray(boundary["Z"]),
            jnp.asarray(boundary["phi"]),
        )
        vac = vacuum_boundary_fields_from_cylindrical_jax(
            br=br,
            bp=bp,
            bz=bz,
            R=boundary["R"],
            Ru=boundary["Ru"],
            Zu=boundary["Zu"],
            Rv=boundary["Rv"],
            Zv=boundary["Zv"],
        )
        rhs_mode = mode_rhs_from_gsource_jax(
            vac["bnormal"],
            sin_basis=sin_basis,
            cos_basis=cos_basis,
            xmpot=xmpot,
            n_raw=n_raw,
            onp=float(onp),
            lasym=bool(lasym),
            imirr=imirr,
            imirr_full=imirr_full,
            nuv3=nuv3,
            nuv_full=nuv_full,
        )
        response = dense_mode_vacuum_solve_jax(
            mode_matrix,
            rhs_mode,
            sin_basis,
            cos_basis,
            symmetric=bool(symmetric),
        )
        response = {**response, "rhs_mode": rhs_mode}
        return boundary, vac, response

    def _update(state, coil_params):
        boundary, vac, response = _mode_response_for_state(state, coil_params)
        return update_from_response(state, response, vac, boundary, coil_params)

    root = dense_fixed_point_solve_jax(
        _update,
        initial_state,
        params,
        max_iter=max_iter,
        damping=damping,
    )
    boundary, vac, response = _mode_response_for_state(root, params)
    fixed_update = _update(root, params)
    return {
        "state": root,
        "fixed_point_residual": root - fixed_update,
        "update": fixed_update,
        "boundary": boundary,
        "vac": vac,
        "response": response,
    }


def direct_coil_projected_mode_fixed_point_objective_jax(
    params: Any,
    initial_state: Any,
    *,
    boundary_from_state: Any,
    update_from_response: Any,
    mode_matrix: Any,
    sin_basis: Any,
    xmpot: Any,
    n_raw: Any,
    imirr: Any | None = None,
    imirr_full: Any | None = None,
    cos_basis: Any | None = None,
    onp: float = 1.0,
    lasym: bool = False,
    nuv3: int | None = None,
    nuv_full: int | None = None,
    max_iter: int = 10,
    damping: float = 1.0,
    symmetric: bool = False,
    state_weights: Any = 1.0,
    update_weights: Any = 0.0,
    mode_weights: Any = 0.0,
    rhs_mode_weights: Any = 0.0,
    bnormal_weight: float = 0.0,
    fixed_point_residual_weight: float = 1.0,
) -> dict[str, Any]:
    """Return a scalar objective for the projected-mode fixed-point helper.

    This wraps :func:`direct_coil_projected_mode_fixed_point_jax` with the
    quadratic objective shape used by the phase-2 AD-vs-FD gates.  It is useful
    for optimizer-facing validation because it exposes the differentiable
    contract as a scalar objective while still returning the solved state and
    component values for diagnostics.

    The default objective is a weighted half-norm of the solved fixed-point
    state plus a small residual guard.  Additional weights can include the
    fixed-point update, vacuum mode coefficients, mode RHS, and boundary normal
    field.  All weights may be scalars or arrays broadcastable to the
    corresponding component.
    """

    solved = direct_coil_projected_mode_fixed_point_jax(
        params,
        initial_state,
        boundary_from_state=boundary_from_state,
        update_from_response=update_from_response,
        mode_matrix=mode_matrix,
        sin_basis=sin_basis,
        xmpot=xmpot,
        n_raw=n_raw,
        imirr=imirr,
        imirr_full=imirr_full,
        cos_basis=cos_basis,
        onp=onp,
        lasym=lasym,
        nuv3=nuv3,
        nuv_full=nuv_full,
        max_iter=max_iter,
        damping=damping,
        symmetric=symmetric,
    )
    components = {
        "state": _weighted_half_norm(solved["state"], state_weights),
        "update": _weighted_half_norm(solved["update"], update_weights),
        "mode": _weighted_half_norm(solved["response"]["mode_coeffs"], mode_weights),
        "rhs_mode": _weighted_half_norm(solved["response"]["rhs_mode"], rhs_mode_weights),
        "bnormal": _weighted_half_norm(solved["vac"]["bnormal"], bnormal_weight),
        "fixed_point_residual": _weighted_half_norm(
            solved["fixed_point_residual"],
            fixed_point_residual_weight,
        ),
    }
    objective = sum(components.values())
    return {
        **solved,
        "objective": objective,
        "objective_components": components,
    }


def direct_coil_projected_mode_fixed_point_directional_check_jax(
    params: Any,
    direction: Any,
    initial_state: Any,
    *,
    eps: float = 1.0e-4,
    **objective_kwargs: Any,
) -> dict[str, Any]:
    """Validate projected-mode fixed-point coil gradients by central FD.

    This is the reusable phase-2/phase-3 validation rung for the direct-coil
    free-boundary adjoint path.  It wraps
    :func:`direct_coil_projected_mode_fixed_point_objective_jax`, computes the
    exact JAX directional derivative with respect to the coil-parameter pytree,
    and compares it with a central finite difference along ``direction``.

    The helper intentionally targets the tiny JAX-visible projected-mode
    fixed-point surrogate.  It exercises the important dependency chain

    ``coil parameters -> Biot-Savart field -> moving boundary projection ->
    dense vacuum solve -> fixed-point state -> scalar objective``

    without overclaiming a production custom VJP for the full
    :func:`vmec_jax.driver.run_free_boundary` control loop.  The returned
    ``solved`` dictionary contains the same state, vacuum, response, and
    objective-component diagnostics as
    :func:`direct_coil_projected_mode_fixed_point_objective_jax`.
    """

    def objective(coil_params):
        solved = direct_coil_projected_mode_fixed_point_objective_jax(
            coil_params,
            initial_state,
            **objective_kwargs,
        )
        return solved["objective"]

    check = pytree_directional_derivative_check_jax(
        objective,
        params,
        direction,
        eps=eps,
    )
    solved = direct_coil_projected_mode_fixed_point_objective_jax(
        params,
        initial_state,
        **objective_kwargs,
    )
    return {
        **check,
        "solved": solved,
        "objective_components": solved["objective_components"],
    }
