"""Native spline-vector residual microprofile for square-coil decks."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from examples.toroidal_stellarator_mirror_hybrid_square_coils_free_boundary import (
    ExampleConfig,
    build_square_coils,
    make_free_boundary_indata,
)
from vmec_jax.boundary import boundary_from_indata
from vmec_jax.config import config_from_indata
from vmec_jax.init_guess import initial_guess_from_boundary
from vmec_jax.solvers.free_boundary import (
    FreeBoundaryNativeSplineResidualProblem,
    free_boundary_native_spline_force_blocks_to_state_residual,
    free_boundary_native_spline_matrix_free_normal_step_jax,
    free_boundary_native_spline_unknown_vector_from_vmec_state,
    free_boundary_native_spline_vector_edge_step,
    free_boundary_native_spline_vector_projected_residual_jax,
    free_boundary_native_spline_vector_to_vmec_state_jax,
)
from vmec_jax.solvers.free_boundary.control import _prepare_freeb_edge_control_projection
from vmec_jax.state import VMECState
from vmec_jax.static import build_static


def _block_until_ready(value: Any) -> Any:
    """Block JAX arrays produced by the microprofile."""

    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
        return value
    if isinstance(value, VMECState):
        for part in (value.Rcos, value.Rsin, value.Zcos, value.Zsin, value.Lcos, value.Lsin):
            _block_until_ready(part)
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            _block_until_ready(item)
    return value


def _time_call(func: Any) -> tuple[Any, float]:
    """Return ``func()`` and elapsed wall time, blocking JAX outputs first."""

    start = time.perf_counter()
    result = func()
    _block_until_ready(result)
    return result, float(time.perf_counter() - start)


def _state_delta(state: VMECState) -> VMECState:
    """Build a deterministic same-shape residual surrogate for profiling."""

    return VMECState(
        layout=state.layout,
        Rcos=0.25 * state.Rcos,
        Rsin=0.20 * state.Rsin,
        Zcos=0.15 * state.Zcos,
        Zsin=0.10 * state.Zsin,
        Lcos=0.05 * state.Lcos,
        Lsin=0.04 * state.Lsin,
    )


def _norms(values: Any) -> dict[str, float]:
    """Return compact finite-vector norms."""

    arr = np.asarray(values, dtype=float).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"l2": 0.0, "linf": 0.0}
    return {
        "l2": float(np.linalg.norm(finite)),
        "linf": float(np.max(np.abs(finite))),
    }


def _pack_state_payload(template: VMECState, value: Any) -> np.ndarray:
    """Pack a VMECState or six-array delta tuple into one host vector."""

    if isinstance(value, VMECState):
        parts = (value.Rcos, value.Rsin, value.Zcos, value.Zsin, value.Lcos, value.Lsin)
    elif isinstance(value, (tuple, list)) and len(value) == 6:
        parts = tuple(value)
    else:
        raise TypeError("value must be a VMECState or six-array tuple")
    return np.asarray(template.layout.pack(*parts), dtype=float).reshape(-1)


def _cosine(a: Any, b: Any) -> float | None:
    """Return the cosine between two finite vectors, or ``None`` for zero vectors."""

    left = np.asarray(a, dtype=float).reshape(-1)
    right = np.asarray(b, dtype=float).reshape(-1)
    if left.size != right.size or left.size == 0:
        return None
    mask = np.isfinite(left) & np.isfinite(right)
    if not np.any(mask):
        return None
    left = left[mask]
    right = right[mask]
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= np.finfo(float).tiny:
        return None
    return float(np.dot(left, right) / denom)


def _json_scalar(value: Any) -> float | int | str | None:
    """Return a compact JSON-safe scalar from an optional JAX/NumPy value."""

    if value is None:
        return None
    try:
        arr = np.asarray(value)
    except Exception:
        return str(value)
    if arr.shape == ():
        item = arr.item()
        if isinstance(item, (bool, int, np.integer)):
            return int(item)
        if isinstance(item, (float, np.floating)):
            return float(item) if np.isfinite(float(item)) else None
        return str(item)
    return str(value)


def _scaled_edge_update(
    direction: Any,
    *,
    target_norm: float,
) -> tuple[np.ndarray, float | None]:
    """Return ``direction`` scaled to ``target_norm`` and the applied scale."""

    values = np.asarray(direction, dtype=float).reshape(-1)
    norm = float(np.linalg.norm(values))
    if norm <= np.finfo(float).tiny:
        return np.zeros_like(values), None
    scale = float(target_norm) / norm
    return values * scale, scale


def _edge_only_update_residual_payload(
    *,
    problem: Any,
    vector: Any,
    edge_count: int,
    edge_update: Any,
    before_l2: float,
) -> dict[str, Any]:
    """Evaluate the projected residual after an edge-only native update."""

    update = np.asarray(edge_update, dtype=float).reshape(-1)
    if update.size != int(edge_count):
        raise ValueError("edge_update must match edge_count")
    candidate = np.array(np.asarray(vector, dtype=float), dtype=float, copy=True).reshape(-1)
    candidate[-int(edge_count) :] += update
    residual, wall_s = _time_call(lambda: problem.residual(candidate))
    norm = _norms(np.asarray(residual, dtype=float))
    reduction = None if float(before_l2) <= np.finfo(float).tiny else float(norm["l2"] / float(before_l2))
    return {
        "status": "completed",
        "edge_update_l2": _norms(update)["l2"],
        "edge_update_linf": _norms(update)["linf"],
        "post_step_residual_wall_s": float(wall_s),
        "projected_residual_l2_after_step": norm["l2"],
        "projected_residual_linf_after_step": norm["linf"],
        "projected_residual_reduction_factor": reduction,
    }


def _initial_free_boundary_vacuum_pressure(
    *,
    config: ExampleConfig,
    state: VMECState,
    static: Any,
    indata: Any,
) -> tuple[dict[str, Any], np.ndarray | None, float | None]:
    """Sample initial edge vacuum pressure for native free-boundary diagnostics."""

    try:
        from vmec_jax.external_fields import build_coil_field_geometry
        from vmec_jax.free_boundary import nestor_external_only_step
        from vmec_jax.solvers.fixed_boundary.residual.runtime import edge_bsqvac_from_nestor
        from vmec_jax.solvers.fixed_boundary.residual.setup import free_boundary_pressure_edge_scale
    except Exception as exc:  # pragma: no cover - dependencies are expected in this repo.
        return {
            "status": "blocked",
            "reason": "free_boundary_vacuum_dependencies_unavailable",
            "error": repr(exc),
        }, None, None

    try:
        coils = build_square_coils(config)
        provider_static = {
            "coil_geometry": build_coil_field_geometry(coils.params),
            "regularization_epsilon": float(getattr(coils.params, "regularization_epsilon", 0.0)),
            "chunk_size": getattr(coils.params, "chunk_size", None),
            "cache_scope": "native_spline_actual_force_step_profile",
            "jit_sampler": False,
            "resample_trial_bsqvac": True,
        }
        (nestor_result, _runtime), wall_s = _time_call(
            lambda: nestor_external_only_step(
                state=state,
                static=static,
                ivac=1,
                ivacskip=0,
                iter_idx=0,
                runtime=None,
                extcur=tuple(getattr(static, "free_boundary_extcur", ()) or ()),
                plascur=0.0,
                external_field_provider_kind="direct_coils",
                external_field_provider_static=provider_static,
                external_field_provider_params=coils.params,
                collect_trace_arrays=False,
            )
        )
        bsqvac = np.asarray(edge_bsqvac_from_nestor(nestor_result, static), dtype=float)
        pres_scale = free_boundary_pressure_edge_scale(
            free_boundary_enabled=True,
            indata=indata,
            s=np.asarray(static.s, dtype=float),
        )
        bsq_norms = _norms(bsqvac)
        diagnostics = getattr(nestor_result, "diagnostics", None)
        diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
        return {
            "status": "included",
            "mode": "direct_coils_nestor_external_only_frozen_initial_edge",
            "provider": "direct_coils",
            "coil_count": int(np.asarray(coils.centers).shape[0]),
            "coil_segments": int(config.coil_segments),
            "trial_vacuum_pressure_resampled": False,
            "differentiable_vacuum_pressure": False,
            "wall_s": float(wall_s),
            "bsqvac_shape": [int(value) for value in bsqvac.shape],
            "bsqvac_l2": bsq_norms["l2"],
            "bsqvac_linf": bsq_norms["linf"],
            "pressure_edge_scale": None if pres_scale is None else float(pres_scale),
            "nestor_reused": bool(getattr(nestor_result, "reused", False)),
            "nestor_model": str(getattr(nestor_result, "model", "")),
            "nestor_solve_time_s": float(getattr(nestor_result, "solve_time_s", 0.0)),
            "nestor_sample_time_s": float(getattr(nestor_result, "sample_time_s", 0.0)),
            "nestor_bnormal_rms": _json_scalar(diagnostics.get("bnormal_rms")),
            "nestor_bsqvac_rms": _json_scalar(diagnostics.get("bsqvac_rms")),
            "next_action": "replace_frozen_bsqvac_with_differentiable_jax_nestor_or_adjoint_replay",
        }, bsqvac, None if pres_scale is None else float(pres_scale)
    except Exception as exc:
        return {
            "status": "failed",
            "mode": "direct_coils_nestor_external_only_frozen_initial_edge",
            "error": repr(exc),
            "next_action": "repair_direct_coil_nestor_sampling_for_native_residual",
        }, None, None


def _mode00_index(modes: Any) -> int:
    """Return the signed-mode index for ``m=0,n=0``."""

    idx = np.where((np.asarray(modes.m) == 0) & (np.asarray(modes.n) == 0))[0]
    return int(idx[0]) if idx.size else 0


def _native_matrix_free_profile(
    *,
    vector: Any,
    direction: Any,
    template_state: VMECState,
    projection: dict[str, Any],
) -> dict[str, Any]:
    """Time JVP/VJP/CG pieces for a native spline-vector normal step."""

    import jax

    residual_fn = lambda values: free_boundary_native_spline_vector_projected_residual_jax(
        values,
        template_state,
        projection,
        _state_delta,
        edge_metric="pullback",
    )

    (vjp_residual, vjp_pullback), vjp_s = _time_call(
        lambda: (
            lambda residual_value, vjp_fun: (residual_value, vjp_fun(residual_value)[0])
        )(*jax.vjp(residual_fn, vector))
    )

    def apply_normal_matvec() -> Any:
        _residual_value, jvp_fun = jax.linearize(residual_fn, vector)
        _residual_for_vjp, vjp_fun = jax.vjp(residual_fn, vector)
        return vjp_fun(jvp_fun(direction))[0]

    normal_matvec, normal_matvec_s = _time_call(apply_normal_matvec)

    problem = FreeBoundaryNativeSplineResidualProblem(
        template_state=template_state,
        projection=projection,
        residual_fn=_state_delta,
        edge_metric="pullback",
    )
    damping = 1.0e-10
    linear_tol = 1.0e-10
    linear_maxiter = 8
    step, cg_step_s = _time_call(
        lambda: free_boundary_native_spline_matrix_free_normal_step_jax(
            problem,
            vector,
            damping=damping,
            tol=linear_tol,
            maxiter=linear_maxiter,
        )
    )
    next_residual, next_residual_s = _time_call(lambda: problem.residual(step.next_vector))
    residual_l2 = float(step.residual_l2)
    next_residual_norm = _norms(np.asarray(next_residual, dtype=float))
    reduction = (
        None
        if residual_l2 <= np.finfo(float).tiny
        else float(next_residual_norm["l2"] / residual_l2)
    )

    return {
        "status": "completed",
        "method": "jax.linearize_vjp_cg_damped_normal_equations",
        "strict_target_ftol": 1.0e-12,
        "dense_jacobian_formed": False,
        "damping": damping,
        "linear_tol": linear_tol,
        "linear_maxiter": linear_maxiter,
        "vjp_wall_s": float(vjp_s),
        "normal_matvec_wall_s": float(normal_matvec_s),
        "cg_step_wall_s": float(cg_step_s),
        "post_step_residual_wall_s": float(next_residual_s),
        "vjp_residual_l2": _norms(np.asarray(vjp_residual, dtype=float))["l2"],
        "vjp_pullback_l2": _norms(np.asarray(vjp_pullback, dtype=float))["l2"],
        "normal_matvec_l2": _norms(np.asarray(normal_matvec, dtype=float))["l2"],
        "normal_matvec_linf": _norms(np.asarray(normal_matvec, dtype=float))["linf"],
        "step_l2": float(step.step_l2),
        "residual_l2_before_step": residual_l2,
        "residual_l2_after_step": next_residual_norm["l2"],
        "residual_linf_after_step": next_residual_norm["linf"],
        "residual_reduction_factor": reduction,
        "cg_info": _json_scalar(step.cg_info),
        "recommended_solver_lane": "matrix_free_native_spline_normal_or_adjoint_solve",
        "next_action": "feed_real_vmec_force_residual_into_native_matrix_free_loop",
    }


def native_spline_actual_force_step_profile_payload(
    *,
    config: ExampleConfig,
    beta_percent: float,
    edge_control_projection_payload: dict[str, Any] | None,
    edge_control_requested: str,
) -> dict[str, Any]:
    """Profile one native matrix-free step using VMEC force blocks.

    This is still a preflight diagnostic: it evaluates the internal VMEC force
    residual on the deck-shaped initial state, converts ``TomnspsRZL`` blocks
    into the native state residual basis, and takes one matrix-free normal
    step. It does not run the nonlinear equilibrium loop and does not include
    NESTOR/free-boundary vacuum-pressure coupling yet.
    """

    if edge_control_projection_payload is None:
        return {
            "status": "blocked",
            "reason": "edge_control_projection_not_enabled",
            "equilibrium_solve_performed": False,
            "next_action": "rerun_with_freeb_edge_control_projection",
        }

    try:
        import jax.numpy as jnp
        from vmec_jax.field import signgs_from_sqrtg
        from vmec_jax.geom import eval_geom
        from vmec_jax.kernels.residue import vmec_scalxc_from_s
        from vmec_jax.kernels.tomnsp import vmec_trig_tables
        from vmec_jax.solvers.fixed_boundary.profiles import (
            build_wout_like_profiles_from_indata,
        )
        from vmec_jax.solvers.fixed_boundary.residual.force_payload import (
            evaluate_residual_force_from_state,
        )
        from vmec_jax.solvers.fixed_boundary.residual.mode_transform import (
            build_mode_transform_context,
        )
        from vmec_jax.solvers.fixed_boundary.residual.update import delta_tuple_from_blocks
    except Exception as exc:  # pragma: no cover - dependencies are expected in this repo.
        return {
            "status": "blocked",
            "reason": "actual_force_dependencies_unavailable",
            "error": repr(exc),
            "equilibrium_solve_performed": False,
        }

    indata = make_free_boundary_indata(config, beta_percent=float(beta_percent))
    vmec_config = config_from_indata(indata)
    static = build_static(vmec_config)
    boundary = boundary_from_indata(indata, static.modes)
    state = initial_guess_from_boundary(static, boundary, indata, vmec_project=True)
    projection = _prepare_freeb_edge_control_projection(
        edge_control_projection_payload,
        indata=indata,
        static=static,
        state0=state,
        free_boundary_enabled=True,
    )
    info = dict(projection.get("info", {}))
    if not bool(projection.get("enabled", False)):
        return {
            "status": "blocked",
            "reason": info.get("reason", "projection_disabled"),
            "edge_control_projection": info,
            "equilibrium_solve_performed": False,
            "next_action": "repair_edge_control_projection_payload",
        }

    geom = eval_geom(state, static)
    signgs = int(signgs_from_sqrtg(np.asarray(geom.sqrtg), axis_index=min(1, int(config.ns) - 1)))
    idx00 = _mode00_index(static.modes)
    s = jnp.asarray(static.s)
    profile_setup = build_wout_like_profiles_from_indata(
        indata=indata,
        static=static,
        s_profile=s,
        signgs=signgs,
        idx00=idx00,
        prefer_host_default_profiles=True,
        s_profile_has_tracer=False,
    )
    trig = vmec_trig_tables(
        ntheta=int(vmec_config.ntheta),
        nzeta=int(vmec_config.nzeta),
        nfp=int(vmec_config.nfp),
        mmax=int(vmec_config.mpol) - 1,
        nmax=int(vmec_config.ntor),
        lasym=bool(vmec_config.lasym),
    )
    mode_context = build_mode_transform_context(
        static=static,
        state0=state,
        s=s,
        host_update_assembly=False,
        setup_host_enforce=False,
        divide_by_scalxc_for_update=False,
        mode_diag_exponent=0.0,
        tree_has_tracer=lambda _value: False,
        vmec_scalxc_from_s=vmec_scalxc_from_s,
    )
    vacuum_pressure, freeb_bsqvac_half, freeb_pres_scale = _initial_free_boundary_vacuum_pressure(
        config=config,
        state=state,
        static=static,
        indata=indata,
    )
    vacuum_pressure_included = freeb_bsqvac_half is not None
    unknowns, encode_s = _time_call(
        lambda: free_boundary_native_spline_unknown_vector_from_vmec_state(state, projection)
    )
    vector = jnp.asarray(unknowns.vector)

    def force_eval(decoded_state: VMECState):
        return evaluate_residual_force_from_state(
            state=decoded_state,
            static=static,
            wout_like=profile_setup.wout_like,
            trig=trig,
            s=s,
            signgs=signgs,
            constraint_tcon0=None,
            freeb_pres_scale=freeb_pres_scale,
            apply_lforbal=False,
            apply_m1_constraints=True,
            include_edge=True,
            include_edge_residual=True,
            zero_m1=True,
            freeb_bsqvac_half=freeb_bsqvac_half,
            iter_idx=None,
            scan_debug_force_enabled=False,
            dump_hlo_force_tomnsps=False,
            dump_hooks={},
        )

    def residual_fn(decoded_state: VMECState) -> VMECState:
        force_result = force_eval(decoded_state)
        return free_boundary_native_spline_force_blocks_to_state_residual(
            template_state=decoded_state,
            force_blocks=force_result.frzl_full,
            mode_context=mode_context,
            lambda_update_scale=1.0,
        )

    force_result, force_s = _time_call(lambda: force_eval(state))
    residual_state, residual_state_s = _time_call(lambda: residual_fn(state))
    physical_delta_transforms = (
        mode_context.mn_cos_to_signed_physical,
        mode_context.mn_sin_to_signed_physical,
        mode_context.mn_cos_to_signed_physical_lambda,
        mode_context.mn_sin_to_signed_physical_lambda,
    )
    force_blocks = force_result.frzl_full
    delta_tuple_reference = delta_tuple_from_blocks(
        1.0,
        physical_delta_transforms,
        force_blocks.frcc,
        getattr(force_blocks, "frss", None),
        getattr(force_blocks, "frsc", None),
        getattr(force_blocks, "frcs", None),
        force_blocks.fzsc,
        getattr(force_blocks, "fzcs", None),
        getattr(force_blocks, "fzcc", None),
        getattr(force_blocks, "fzss", None),
        force_blocks.flsc,
        getattr(force_blocks, "flcs", None),
        getattr(force_blocks, "flcc", None),
        getattr(force_blocks, "flss", None),
        lasym=bool(vmec_config.lasym),
        zeros_dR_np=np.zeros_like(np.asarray(state.Rcos, dtype=float)),
        use_numpy_lasym_zeros=True,
    )
    residual_state_vector = _pack_state_payload(state, residual_state)
    delta_tuple_reference_vector = _pack_state_payload(state, delta_tuple_reference)
    mapping_delta = residual_state_vector - delta_tuple_reference_vector
    reference_norm = _norms(delta_tuple_reference_vector)
    mapping_delta_norm = _norms(mapping_delta)
    mapping_delta_rel = (
        None
        if reference_norm["l2"] <= np.finfo(float).tiny
        else float(mapping_delta_norm["l2"] / reference_norm["l2"])
    )
    mapping_delta_linf = mapping_delta_norm["linf"]
    mapping_status = (
        "state_residual_matches_delta_tuple_from_blocks"
        if mapping_delta_linf <= 1.0e-12 * max(1.0, reference_norm["linf"])
        else "state_residual_differs_from_delta_tuple_from_blocks"
    )
    problem = FreeBoundaryNativeSplineResidualProblem(
        template_state=state,
        projection=projection,
        residual_fn=residual_fn,
        edge_metric="pullback",
    )
    projected_residual, projected_s = _time_call(lambda: problem.residual(vector))
    damping = 1.0e-8
    linear_tol = 1.0e-10
    linear_maxiter = 8
    step, step_s = _time_call(
        lambda: free_boundary_native_spline_matrix_free_normal_step_jax(
            problem,
            vector,
            damping=damping,
            tol=linear_tol,
            maxiter=linear_maxiter,
        )
    )
    next_residual, next_residual_s = _time_call(lambda: problem.residual(step.next_vector))
    before_norm = _norms(np.asarray(projected_residual, dtype=float))
    after_norm = _norms(np.asarray(next_residual, dtype=float))
    reduction = (
        None
        if before_norm["l2"] <= np.finfo(float).tiny
        else float(after_norm["l2"] / before_norm["l2"])
    )
    bridge_dt_eff = 1.0 if config.delt is None else float(config.delt)
    bridge_b1 = 0.0
    bridge_fac = 1.0
    bridge_force_scale = 1.0
    bridge_flip_sign = 1.0
    bridge_step, bridge_s = _time_call(
        lambda: free_boundary_native_spline_vector_edge_step(
            state_current=state,
            state_candidate=state,
            update_deltas=residual_state,
            force_deltas=residual_state,
            projection=projection,
            unknowns=unknowns,
            control_velocity=None,
            dt_eff=bridge_dt_eff,
            b1=bridge_b1,
            fac=bridge_fac,
            force_scale=bridge_force_scale,
            flip_sign=bridge_flip_sign,
        )
    )
    bridge_residual, bridge_residual_s = _time_call(
        lambda: problem.residual(jnp.asarray(bridge_step.unknowns.vector))
    )
    bridge_norm = _norms(np.asarray(bridge_residual, dtype=float))
    bridge_reduction = (
        None
        if before_norm["l2"] <= np.finfo(float).tiny
        else float(bridge_norm["l2"] / before_norm["l2"])
    )
    edge_count = int(unknowns.edge_control_size)
    matrix_free_edge_update = np.asarray(step.step, dtype=float).reshape(-1)[-edge_count:]
    bridge_edge_update = np.asarray(bridge_step.control_update, dtype=float).reshape(-1)
    opposite_bridge_flip_sign = -bridge_flip_sign
    opposite_bridge_step, opposite_bridge_s = _time_call(
        lambda: free_boundary_native_spline_vector_edge_step(
            state_current=state,
            state_candidate=state,
            update_deltas=residual_state,
            force_deltas=residual_state,
            projection=projection,
            unknowns=unknowns,
            control_velocity=None,
            dt_eff=bridge_dt_eff,
            b1=bridge_b1,
            fac=bridge_fac,
            force_scale=bridge_force_scale,
            flip_sign=opposite_bridge_flip_sign,
        )
    )
    opposite_bridge_residual, opposite_bridge_residual_s = _time_call(
        lambda: problem.residual(jnp.asarray(opposite_bridge_step.unknowns.vector))
    )
    opposite_bridge_norm = _norms(np.asarray(opposite_bridge_residual, dtype=float))
    opposite_bridge_reduction = (
        None
        if before_norm["l2"] <= np.finfo(float).tiny
        else float(opposite_bridge_norm["l2"] / before_norm["l2"])
    )
    opposite_bridge_edge_update = np.asarray(opposite_bridge_step.control_update, dtype=float).reshape(-1)
    bridge_cosine = _cosine(bridge_edge_update, matrix_free_edge_update)
    opposite_bridge_cosine = _cosine(opposite_bridge_edge_update, matrix_free_edge_update)
    matrix_free_edge_update_norm = _norms(matrix_free_edge_update)
    bridge_edge_update_norm = _norms(bridge_edge_update)
    opposite_bridge_edge_update_norm = _norms(opposite_bridge_edge_update)
    bridge_norm_matched_update, bridge_norm_match_scale = _scaled_edge_update(
        bridge_edge_update,
        target_norm=matrix_free_edge_update_norm["l2"],
    )
    bridge_norm_matched = _edge_only_update_residual_payload(
        problem=problem,
        vector=vector,
        edge_count=edge_count,
        edge_update=bridge_norm_matched_update,
        before_l2=before_norm["l2"],
    )
    opposite_norm_matched_update, opposite_norm_match_scale = _scaled_edge_update(
        opposite_bridge_edge_update,
        target_norm=matrix_free_edge_update_norm["l2"],
    )
    opposite_norm_matched = _edge_only_update_residual_payload(
        problem=problem,
        vector=vector,
        edge_count=edge_count,
        edge_update=opposite_norm_matched_update,
        before_l2=before_norm["l2"],
    )
    norm_matched_candidates = (
        ("vmec_flip_sign", bridge_norm_matched),
        ("opposite_flip_sign", opposite_norm_matched),
    )
    norm_matched_best_label, norm_matched_best_payload = min(
        norm_matched_candidates,
        key=lambda item: float(item[1].get("projected_residual_l2_after_step", np.inf)),
    )
    norm_matched_best_reduction = norm_matched_best_payload.get("projected_residual_reduction_factor")
    norm_matched_status = (
        "edge_direction_reduces_projected_residual"
        if norm_matched_best_reduction is not None and float(norm_matched_best_reduction) < 1.0
        else "edge_direction_does_not_reduce_projected_residual"
    )
    if bridge_cosine is None and opposite_bridge_cosine is None:
        sign_alignment = "undetermined_zero_edge_update"
    elif opposite_bridge_cosine is not None and (bridge_cosine is None or opposite_bridge_cosine > bridge_cosine):
        sign_alignment = "opposite_flip_sign_better_matches_matrix_free_edge_update"
    else:
        sign_alignment = "vmec_flip_sign_better_matches_matrix_free_edge_update"
    force_norms = {
        "gcr2": float(np.asarray(force_result.gcr2)),
        "gcz2": float(np.asarray(force_result.gcz2)),
        "gcl2": float(np.asarray(force_result.gcl2)),
    }

    return {
        "status": "completed",
        "equilibrium_solve_performed": False,
        "force_scope": (
            "vmec_force_blocks_with_frozen_initial_free_boundary_vacuum_pressure"
            if vacuum_pressure_included
            else "internal_vmec_force_blocks_only"
        ),
        "free_boundary_vacuum_pressure_included": bool(vacuum_pressure_included),
        "free_boundary_vacuum_pressure": vacuum_pressure,
        "edge_control_projection_requested": str(edge_control_requested),
        "edge_control_projection": info,
        "native_state_schema": "FreeBoundaryNativeSplineUnknownVector.v1",
        "native_unknown_size": int(unknowns.native_unknown_size),
        "full_vmec_size": int(unknowns.full_vmec_size),
        "removed_fourier_edge_dofs": int(unknowns.removed_fourier_edge_dofs),
        "unknown_reduction_fraction": float(unknowns.native_unknown_size / unknowns.full_vmec_size),
        "signgs": signgs,
        "encode_wall_s": float(encode_s),
        "force_eval_wall_s": float(force_s),
        "force_blocks_to_state_wall_s": float(residual_state_s),
        "projected_residual_wall_s": float(projected_s),
        "matrix_free_step_wall_s": float(step_s),
        "post_step_residual_wall_s": float(next_residual_s),
        "force_gcr2": force_norms["gcr2"],
        "force_gcz2": force_norms["gcz2"],
        "force_gcl2": force_norms["gcl2"],
        "state_residual_l2": _norms(residual_state_vector)["l2"],
        "force_block_mapping_audit": {
            "status": mapping_status,
            "reference_path": "delta_tuple_from_blocks_with_physical_delta_transforms",
            "candidate_path": "free_boundary_native_spline_force_blocks_to_state_residual",
            "delta_l2": mapping_delta_norm["l2"],
            "delta_linf": mapping_delta_linf,
            "delta_rel": mapping_delta_rel,
            "cosine": _cosine(residual_state_vector, delta_tuple_reference_vector),
        },
        "projected_residual_l2_before_step": before_norm["l2"],
        "projected_residual_linf_before_step": before_norm["linf"],
        "projected_residual_l2_after_step": after_norm["l2"],
        "projected_residual_linf_after_step": after_norm["linf"],
        "projected_residual_reduction_factor": reduction,
        "matrix_free_damping": damping,
        "matrix_free_linear_tol": linear_tol,
        "matrix_free_linear_maxiter": linear_maxiter,
        "matrix_free_step_l2": float(step.step_l2),
        "matrix_free_cg_info": _json_scalar(step.cg_info),
        "edge_bridge_comparison": {
            "status": "completed",
            "method": "free_boundary_native_spline_vector_edge_step",
            "interpretation": "edge_only_vmec_momentum_style_update_not_newton_step",
            "force_metric": str(bridge_step.force_metric),
            "dt_eff": bridge_dt_eff,
            "b1": bridge_b1,
            "fac": bridge_fac,
            "force_scale": bridge_force_scale,
            "flip_sign": bridge_flip_sign,
            "wall_s": float(bridge_s),
            "post_step_residual_wall_s": float(bridge_residual_s),
            "target_l2": float(bridge_step.target_l2),
            "control_force_l2": float(bridge_step.control_force_l2),
            "control_velocity_l2": float(bridge_step.control_velocity_l2),
            "control_update_l2": float(bridge_step.control_update_l2),
            "trust_scale": float(bridge_step.trust_scale),
            "native_update_vector_l2": _norms(np.asarray(bridge_step.native_update_vector, dtype=float))["l2"],
            "matrix_free_edge_update_l2": matrix_free_edge_update_norm["l2"],
            "bridge_edge_update_l2": bridge_edge_update_norm["l2"],
            "bridge_edge_to_matrix_free_l2_ratio": (
                None
                if matrix_free_edge_update_norm["l2"] <= np.finfo(float).tiny
                else float(bridge_edge_update_norm["l2"] / matrix_free_edge_update_norm["l2"])
            ),
            "edge_update_cosine_to_matrix_free": bridge_cosine,
            "projected_residual_l2_after_step": bridge_norm["l2"],
            "projected_residual_linf_after_step": bridge_norm["linf"],
            "projected_residual_reduction_factor": bridge_reduction,
            "opposite_flip_sign_comparison": {
                "flip_sign": opposite_bridge_flip_sign,
                "wall_s": float(opposite_bridge_s),
                "post_step_residual_wall_s": float(opposite_bridge_residual_s),
                "control_update_l2": float(opposite_bridge_step.control_update_l2),
                "opposite_edge_to_matrix_free_l2_ratio": (
                    None
                    if matrix_free_edge_update_norm["l2"] <= np.finfo(float).tiny
                    else float(opposite_bridge_edge_update_norm["l2"] / matrix_free_edge_update_norm["l2"])
                ),
                "edge_update_cosine_to_matrix_free": opposite_bridge_cosine,
                "projected_residual_l2_after_step": opposite_bridge_norm["l2"],
                "projected_residual_linf_after_step": opposite_bridge_norm["linf"],
                "projected_residual_reduction_factor": opposite_bridge_reduction,
            },
            "norm_matched_edge_only_comparison": {
                "status": norm_matched_status,
                "target_l2": matrix_free_edge_update_norm["l2"],
                "best_direction": norm_matched_best_label,
                "best_projected_residual_l2_after_step": norm_matched_best_payload.get(
                    "projected_residual_l2_after_step"
                ),
                "best_projected_residual_reduction_factor": norm_matched_best_reduction,
                "vmec_flip_sign": {
                    "scale": bridge_norm_match_scale,
                    **bridge_norm_matched,
                },
                "opposite_flip_sign": {
                    "scale": opposite_norm_match_scale,
                    **opposite_norm_matched,
                },
            },
            "sign_alignment_status": sign_alignment,
        },
        "next_action": "promote_matrix_free_native_spline_residual_with_vacuum_pressure_and_line_search",
    }


def native_spline_vector_residual_profile_payload(
    *,
    config: ExampleConfig,
    beta_percent: float,
    edge_control_projection_payload: dict[str, Any] | None,
    edge_control_requested: str,
) -> dict[str, Any]:
    """Measure native-vector residual mechanics on a real deck-shaped state.

    This is a no-solve diagnostic. It uses the selected square-coil deck,
    boundary projection, and initial VMEC state to time native encode, decode,
    projected residual packing, and a forward-mode JVP through that projected
    residual. The residual itself is a deterministic same-shape surrogate so
    this function stays fast and does not claim equilibrium convergence.
    """

    if edge_control_projection_payload is None:
        return {
            "status": "blocked",
            "reason": "edge_control_projection_not_enabled",
            "equilibrium_solve_performed": False,
            "next_action": "rerun_with_freeb_edge_control_projection",
        }

    try:
        import jax
        import jax.numpy as jnp
    except Exception as exc:  # pragma: no cover - JAX is expected in this repo.
        return {
            "status": "blocked",
            "reason": "jax_unavailable",
            "error": repr(exc),
            "equilibrium_solve_performed": False,
        }

    indata = make_free_boundary_indata(config, beta_percent=float(beta_percent))
    vmec_config = config_from_indata(indata)
    static = build_static(vmec_config)
    boundary = boundary_from_indata(indata, static.modes)
    state = initial_guess_from_boundary(static, boundary, indata, vmec_project=True)
    projection = _prepare_freeb_edge_control_projection(
        edge_control_projection_payload,
        indata=indata,
        static=static,
        state0=state,
        free_boundary_enabled=True,
    )
    info = dict(projection.get("info", {}))
    if not bool(projection.get("enabled", False)):
        return {
            "status": "blocked",
            "reason": info.get("reason", "projection_disabled"),
            "edge_control_projection": info,
            "equilibrium_solve_performed": False,
            "next_action": "repair_edge_control_projection_payload",
        }

    unknowns, encode_s = _time_call(
        lambda: free_boundary_native_spline_unknown_vector_from_vmec_state(state, projection)
    )
    vector = jnp.asarray(unknowns.vector)
    direction = jnp.linspace(
        1.0,
        2.0,
        int(unknowns.native_unknown_size),
        dtype=vector.dtype,
    )

    decoded, decode_s = _time_call(
        lambda: free_boundary_native_spline_vector_to_vmec_state_jax(
            vector,
            state,
            projection,
        )
    )
    decoded_host = unknowns.to_vmec_state()
    decode_delta = np.asarray(
        decoded_host.layout.pack(
            np.asarray(decoded.Rcos) - np.asarray(decoded_host.Rcos),
            np.asarray(decoded.Rsin) - np.asarray(decoded_host.Rsin),
            np.asarray(decoded.Zcos) - np.asarray(decoded_host.Zcos),
            np.asarray(decoded.Zsin) - np.asarray(decoded_host.Zsin),
            np.asarray(decoded.Lcos) - np.asarray(decoded_host.Lcos),
            np.asarray(decoded.Lsin) - np.asarray(decoded_host.Lsin),
        )
    )
    residual_host = _state_delta(decoded_host)
    host_pullback = unknowns.vector_from_delta_tuple(residual_host, edge_metric="pullback")

    projected, projected_s = _time_call(
        lambda: free_boundary_native_spline_vector_projected_residual_jax(
            vector,
            state,
            projection,
            _state_delta,
            edge_metric="pullback",
        )
    )
    projected_np = np.asarray(projected, dtype=float)
    residual_delta = projected_np - np.asarray(host_pullback, dtype=float)
    residual_norm = _norms(projected_np)
    residual_delta_norm = _norms(residual_delta)
    residual_rel = (
        None
        if residual_norm["l2"] <= np.finfo(float).tiny
        else float(residual_delta_norm["l2"] / residual_norm["l2"])
    )

    (_jvp_value, jvp), jvp_s = _time_call(
        lambda: jax.jvp(
            lambda values: free_boundary_native_spline_vector_projected_residual_jax(
                values,
                state,
                projection,
                _state_delta,
                edge_metric="pullback",
            ),
            (vector,),
            (direction,),
        )
    )
    jvp_norm = _norms(np.asarray(jvp, dtype=float))
    matrix_free_profile = _native_matrix_free_profile(
        vector=vector,
        direction=direction,
        template_state=state,
        projection=projection,
    )
    full_vmec_size = int(unknowns.full_vmec_size)
    native_unknown_size = int(unknowns.native_unknown_size)
    bytes_per_float = int(np.asarray(unknowns.vector).dtype.itemsize)

    return {
        "status": "completed",
        "equilibrium_solve_performed": False,
        "residual_surrogate": "linear_same_shape_state_delta",
        "edge_control_projection_requested": str(edge_control_requested),
        "edge_control_projection": info,
        "native_state_schema": "FreeBoundaryNativeSplineUnknownVector.v1",
        "native_unknown_size": native_unknown_size,
        "full_vmec_size": full_vmec_size,
        "interior_unknown_size": int(unknowns.interior_size),
        "edge_control_size": int(unknowns.edge_control_size),
        "removed_fourier_edge_dofs": int(unknowns.removed_fourier_edge_dofs),
        "unknown_reduction_fraction": float(native_unknown_size / full_vmec_size),
        "native_vector_bytes": int(native_unknown_size * bytes_per_float),
        "full_vmec_vector_bytes": int(full_vmec_size * bytes_per_float),
        "encode_wall_s": float(encode_s),
        "decode_wall_s": float(decode_s),
        "projected_residual_wall_s": float(projected_s),
        "projected_residual_jvp_wall_s": float(jvp_s),
        "decode_parity_linf": _norms(decode_delta)["linf"],
        "projected_residual_l2": residual_norm["l2"],
        "projected_residual_linf": residual_norm["linf"],
        "projected_residual_host_parity_l2": residual_delta_norm["l2"],
        "projected_residual_host_parity_linf": residual_delta_norm["linf"],
        "projected_residual_host_parity_rel": residual_rel,
        "jvp_l2": jvp_norm["l2"],
        "jvp_linf": jvp_norm["linf"],
        "autodiff_method_profiled": "jax.jvp_forward_mode",
        "matrix_free_normal_step_profile": matrix_free_profile,
        "next_action": "promote_packed_native_vector_into_opt_in_solver_loop",
    }
