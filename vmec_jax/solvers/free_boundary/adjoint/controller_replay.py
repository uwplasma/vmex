"""JAX-visible accepted-controller replay for direct-coil free-boundary traces."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

from vmec_jax._compat import jax, jnp
from vmec_jax.solvers.free_boundary.adjoint.controller import (
    jax_visible_accepted_nonlinear_controller_jax,
    jax_visible_accepted_only_nonlinear_controller_jax,
    jax_visible_segmented_accepted_nonlinear_controller_jax,
    jax_visible_segmented_state_only_accepted_nonlinear_controller_jax,
    jax_visible_state_only_accepted_nonlinear_controller_jax,
    jax_visible_state_only_accepted_only_nonlinear_controller_jax,
    jax_visible_unrolled_accepted_only_nonlinear_controller_jax,
    jax_visible_unrolled_state_only_accepted_only_nonlinear_controller_jax,
)
from vmec_jax.solvers.free_boundary.adjoint.boundary_replay import free_boundary_boundary_geometry_jax
from vmec_jax.solvers.free_boundary.adjoint.direct_coil_replay import (
    direct_coil_boundary_bsqvac_from_trace_jax,
    direct_coil_boundary_bsqvac_jax,
)
from vmec_jax.solvers.free_boundary.adjoint.objectives import (
    accepted_controller_replay_result as _accepted_controller_replay_result,
    tree_weighted_half_norm as _tree_weighted_half_norm,
    weighted_half_norm as _weighted_half_norm,
)
from vmec_jax.solvers.free_boundary.adjoint.replay_context import (
    direct_coil_boundary_replay_context,
    direct_coil_boundary_replay_context_for_shape,
    direct_coil_trace_boundary_shape as _direct_coil_trace_boundary_shape,
    direct_coil_trace_vacuum_field_override as _direct_coil_trace_vacuum_field_override,
)
from vmec_jax.solvers.free_boundary.adjoint.replay_plan import (
    direct_coil_accepted_trace_controller_replay_plan,
)
from vmec_jax.solvers.free_boundary.adjoint.runtime import jax_named_scope as _runtime_jax_named_scope
from vmec_jax.solvers.free_boundary.adjoint.trace_controls import (
    accepted_trace_segment_is_unconditionally_accepted as _accepted_trace_segment_is_unconditionally_accepted,
)


def _jax_named_scope(name: str) -> Any:
    return _runtime_jax_named_scope(name, jax_module=jax, nullcontext_factory=nullcontext)


def _zero_aux() -> dict[str, Any]:
    return {
        "force": jnp.asarray(0.0),
        "bsqvac": jnp.asarray(0.0),
        "bsqvac_rms": jnp.asarray(0.0),
        "bnormal_rms": jnp.asarray(0.0),
        "state_reset": jnp.asarray(False, dtype=bool),
    }


@dataclass(frozen=True)
class _ControllerReplayOptions:
    """Static replay options shared by the controller replay helpers."""

    signgs: int
    sample_nzeta: int | None
    include_analytic: bool
    enforce_edge: bool
    force_weight: Any
    bsqvac_weight: Any
    checkpoint_steps: bool
    state_only_replay: bool
    freeze_vacuum_field: bool
    freeze_freeb_bsqvac: bool
    include_mode_diagnostics: bool
    nestor_solve_mode: str
    nestor_operator_solver: str
    nestor_operator_tol: float
    nestor_operator_atol: float
    nestor_operator_maxiter: int | None
    nestor_operator_restart: int | None
    jit_preconditioner_apply: bool
    unroll_accepted_only_segments_below: int
    coil_geometry: Any | None


def _precomputed_context_for_trace(
    trace: Mapping[str, Any],
    *,
    static: Any,
    context_cache: dict[tuple[int, int], dict[str, Any]],
) -> dict[str, Any] | None:
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


def _freeb_bsqvac_replay_terms(
    trace: dict[str, Any],
    state_in: Any,
    coil_params: Any,
    control: dict[str, Any],
    replay_context: dict[str, Any] | None,
    *,
    static: Any,
    options: _ControllerReplayOptions,
):
    has_active_freeb_replay = trace.get("freeb_bsqvac_half") is not None and trace.get("freeb_nestor_trace") is not None
    if has_active_freeb_replay:
        if bool(options.freeze_freeb_bsqvac):
            freeb_bsqvac_half = jnp.asarray(trace["freeb_bsqvac_half"])
        else:
            with _jax_named_scope("vmec_jax.free_boundary.boundary_geometry"):
                geometry = free_boundary_boundary_geometry_jax(
                    state_in,
                    static,
                    sample_nzeta=options.sample_nzeta,
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
                        signgs=int(options.signgs),
                        nvper=int(context["nvper"]),
                        wint=jnp.asarray(context["wint"]),
                        include_analytic=bool(options.include_analytic),
                        include_diagnostics=not bool(options.state_only_replay),
                        include_mode_diagnostics=bool(options.include_mode_diagnostics),
                        freeze_vacuum_field=bool(options.freeze_vacuum_field),
                        nestor_solve_mode=str(options.nestor_solve_mode),
                        nestor_operator_solver=str(options.nestor_operator_solver),
                        nestor_operator_tol=float(options.nestor_operator_tol),
                        nestor_operator_atol=float(options.nestor_operator_atol),
                        nestor_operator_maxiter=options.nestor_operator_maxiter,
                        nestor_operator_restart=options.nestor_operator_restart,
                        coil_geometry=options.coil_geometry,
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
                        signgs=int(options.signgs),
                        nvper=int(context["nvper"]),
                        br_add=jnp.asarray(nestor_axes["br_axis"]),
                        bp_add=jnp.asarray(nestor_axes["bp_axis"]),
                        bz_add=jnp.asarray(nestor_axes["bz_axis"]),
                        wint=jnp.asarray(context["wint"]),
                        include_analytic=bool(options.include_analytic),
                        include_diagnostics=not bool(options.state_only_replay),
                        include_mode_diagnostics=bool(options.include_mode_diagnostics),
                        vac_override=(
                            _direct_coil_trace_vacuum_field_override(trace)
                            if bool(options.freeze_vacuum_field)
                            else None
                        ),
                        coil_geometry=options.coil_geometry,
                        nestor_solve_mode=str(options.nestor_solve_mode),
                        nestor_operator_solver=str(options.nestor_operator_solver),
                        nestor_operator_tol=float(options.nestor_operator_tol),
                        nestor_operator_atol=float(options.nestor_operator_atol),
                        nestor_operator_maxiter=options.nestor_operator_maxiter,
                        nestor_operator_restart=options.nestor_operator_restart,
                    )
            freeb_bsqvac_half = replay["bsqvac"]
        if bool(options.state_only_replay):
            bsqvac_objective = jnp.asarray(0.0)
            bsqvac_rms = jnp.asarray(0.0)
            bnormal_rms = jnp.asarray(0.0)
        elif bool(options.freeze_freeb_bsqvac):
            bsqvac_objective = _weighted_half_norm(freeb_bsqvac_half, options.bsqvac_weight)
            bsqvac_rms = jnp.sqrt(jnp.mean(jnp.square(jnp.asarray(freeb_bsqvac_half))))
            bnormal_rms = jnp.asarray(0.0)
        else:
            bsqvac_objective = _weighted_half_norm(replay["bsqvac"], options.bsqvac_weight)
            bsqvac_rms = jnp.sqrt(jnp.mean(jnp.square(jnp.asarray(replay["bsqvac"]))))
            bnormal_rms = jnp.sqrt(jnp.mean(jnp.square(jnp.asarray(replay["vac"]["bnormal"]))))
    else:
        freeb_bsqvac_half = trace.get("freeb_bsqvac_half", None)
        bsqvac_objective = jnp.asarray(0.0)
        bsqvac_rms = jnp.asarray(0.0)
        bnormal_rms = jnp.asarray(0.0)
    return freeb_bsqvac_half, bsqvac_objective, bsqvac_rms, bnormal_rms


def _branch_for_trace(
    trace: dict[str, Any],
    state: Any,
    coil_params: Any,
    control: dict[str, Any],
    replay_context: dict[str, Any] | None,
    *,
    static: Any,
    options: _ControllerReplayOptions,
):
    from vmec_jax.discrete_adjoint import strict_update_one_step_from_trace

    reset_to_trace_pre = jnp.asarray(control["reset_to_trace_pre"], dtype=bool)
    state_in = jax.lax.cond(
        reset_to_trace_pre,
        lambda _: trace["state_pre"],
        lambda _: state,
        operand=None,
    )
    freeb_bsqvac_half, bsqvac_objective, bsqvac_rms, bnormal_rms = _freeb_bsqvac_replay_terms(
        trace,
        state_in,
        coil_params,
        control,
        replay_context,
        static=static,
        options=options,
    )
    with _jax_named_scope("vmec_jax.free_boundary.strict_update_one_step_from_trace"):
        step = strict_update_one_step_from_trace(
            state_in,
            static,
            trace,
            scalar_controls=control["step_scalars"],
            array_controls=control["step_arrays"],
            preconditioner_controls=control["step_preconditioner"] if "step_preconditioner" in control else None,
            freeb_bsqvac_half=freeb_bsqvac_half,
            enforce_edge=options.enforce_edge,
            jit_preconditioner_apply=options.jit_preconditioner_apply,
        )
    if bool(options.state_only_replay):
        return step["step"]["state_post"], {
            "state_reset": reset_to_trace_pre,
        }
    return step["step"]["state_post"], {
        "force": _tree_weighted_half_norm(step["force"], options.force_weight),
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
    *,
    static: Any,
    options: _ControllerReplayOptions,
):
    from vmec_jax.discrete_adjoint import strict_update_one_step_from_state

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
    freeb_bsqvac_half, bsqvac_objective, bsqvac_rms, bnormal_rms = _freeb_bsqvac_replay_terms(
        trace,
        state_in,
        coil_params,
        control,
        replay_context,
        static=static,
        options=options,
    )
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
            enforce_edge=options.enforce_edge,
            jit_preconditioner_apply=options.jit_preconditioner_apply,
        )
    if bool(options.state_only_replay):
        return step["step"]["state_post"], {
            "state_reset": reset_to_trace_pre,
        }
    return step["step"]["state_post"], {
        "force": _tree_weighted_half_norm(step["force"], options.force_weight),
        "bsqvac": bsqvac_objective,
        "bsqvac_rms": bsqvac_rms,
        "bnormal_rms": bnormal_rms,
        "state_reset": reset_to_trace_pre,
    }


def _make_controller_step_fn(
    segment_traces: tuple[dict[str, Any], ...],
    *,
    static: Any,
    context_cache: dict[tuple[int, int], dict[str, Any]],
    options: _ControllerReplayOptions,
    index_offset: int = 0,
    stacked_step_controls: bool = False,
    accepted_only: bool = False,
):
    if bool(stacked_step_controls):
        representative_trace = segment_traces[0]
        representative_context = _precomputed_context_for_trace(
            representative_trace,
            static=static,
            context_cache=context_cache,
        )

        def _step_fn(state, coil_params, control):
            if bool(accepted_only):
                return _branch_from_stacked_controls(
                    representative_trace,
                    state,
                    coil_params,
                    control,
                    representative_context,
                    static=static,
                    options=options,
                )
            do_propose = jnp.asarray(control["accept"], dtype=bool)

            def _propose(_unused):
                return _branch_from_stacked_controls(
                    representative_trace,
                    state,
                    coil_params,
                    control,
                    representative_context,
                    static=static,
                    options=options,
                )

            def _skip(_unused):
                if bool(options.state_only_replay):
                    return state, {"state_reset": jnp.asarray(False, dtype=bool)}
                return state, _zero_aux()

            return jax.lax.cond(do_propose, _propose, _skip, operand=None)

        return _step_fn

    branches = tuple(
        (
            lambda operand, trace=trace, replay_context=_precomputed_context_for_trace(
                trace,
                static=static,
                context_cache=context_cache,
            ): _branch_for_trace(
                trace,
                operand[0],
                operand[1],
                operand[2],
                replay_context,
                static=static,
                options=options,
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
            if bool(options.state_only_replay):
                return state, {"state_reset": jnp.asarray(False, dtype=bool)}
            return state, _zero_aux()

        return jax.lax.cond(do_propose, _propose, _skip, operand=None)

    return _step_fn


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
    """Replay fixed production traces through a JAX-visible accept controller."""

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
    options = _ControllerReplayOptions(
        signgs=int(signgs),
        sample_nzeta=sample_nzeta,
        include_analytic=bool(include_analytic),
        enforce_edge=bool(enforce_edge),
        force_weight=force_weight,
        bsqvac_weight=bsqvac_weight,
        checkpoint_steps=bool(checkpoint_steps),
        state_only_replay=bool(state_only_replay),
        freeze_vacuum_field=bool(freeze_vacuum_field),
        freeze_freeb_bsqvac=bool(freeze_freeb_bsqvac),
        include_mode_diagnostics=bool(include_mode_diagnostics),
        nestor_solve_mode=str(nestor_solve_mode),
        nestor_operator_solver=str(nestor_operator_solver),
        nestor_operator_tol=float(nestor_operator_tol),
        nestor_operator_atol=float(nestor_operator_atol),
        nestor_operator_maxiter=nestor_operator_maxiter,
        nestor_operator_restart=nestor_operator_restart,
        jit_preconditioner_apply=bool(jit_preconditioner_apply),
        unroll_accepted_only_segments_below=int(unroll_accepted_only_segments_below),
        coil_geometry=coil_geometry,
    )

    context_cache: dict[tuple[int, int], dict[str, Any]] = dict(replay_plan.get("boundary_replay_contexts_by_shape", {}))

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
            _make_controller_step_fn(
                trace_seq[int(segment["start"]) : int(segment["stop"])],
                static=static,
                context_cache=context_cache,
                options=options,
                index_offset=int(segment["start"]),
                stacked_step_controls=True,
                accepted_only=bool(accepted_only_fast_path_segments[index]),
            )
            for index, segment in enumerate(step_policy_segments)
        )
        segmented_runner = (
            jax_visible_segmented_state_only_accepted_nonlinear_controller_jax
            if bool(options.state_only_replay)
            else jax_visible_segmented_accepted_nonlinear_controller_jax
        )
        run = segmented_runner(
            step_fns,
            accept_fn,
            converged_fn,
            initial_state,
            params,
            control_segments,
            checkpoint_steps=options.checkpoint_steps,
            accepted_only_segments=accepted_only_fast_path_segments,
            unroll_accepted_only_segments_below=options.unroll_accepted_only_segments_below,
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
            _make_controller_step_fn(
                trace_seq[int(segment["start"]) : int(segment["stop"])],
                static=static,
                context_cache=context_cache,
                options=options,
                index_offset=int(segment["start"]),
                accepted_only=bool(accepted_only_fast_path_segments[index]),
            )
            for index, segment in enumerate(preconditioner_policy_segments)
        )
        segmented_runner = (
            jax_visible_segmented_state_only_accepted_nonlinear_controller_jax
            if bool(options.state_only_replay)
            else jax_visible_segmented_accepted_nonlinear_controller_jax
        )
        run = segmented_runner(
            step_fns,
            accept_fn,
            converged_fn,
            initial_state,
            params,
            control_segments,
            checkpoint_steps=options.checkpoint_steps,
            accepted_only_segments=accepted_only_fast_path_segments,
            unroll_accepted_only_segments_below=options.unroll_accepted_only_segments_below,
        )
    else:
        accepted_only_fast_path_segments = (
            bool(use_accepted_only_fast_path)
            and _accepted_trace_segment_is_unconditionally_accepted(effective_masks, start=0, stop=len(trace_seq)),
        )
        step_fn = _make_controller_step_fn(
            trace_seq,
            static=static,
            context_cache=context_cache,
            options=options,
            accepted_only=accepted_only_fast_path_segments[0],
        )
        if accepted_only_fast_path_segments[0]:
            use_unrolled = (
                options.unroll_accepted_only_segments_below > 0
                and len(trace_seq) <= options.unroll_accepted_only_segments_below
            )
            if bool(options.state_only_replay):
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
                checkpoint_steps=options.checkpoint_steps,
            )
        else:
            accepted_runner = (
                jax_visible_state_only_accepted_nonlinear_controller_jax
                if bool(options.state_only_replay)
                else jax_visible_accepted_nonlinear_controller_jax
            )
            run = accepted_runner(
                step_fn,
                accept_fn,
                converged_fn,
                initial_state,
                params,
                controls,
                checkpoint_steps=options.checkpoint_steps,
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
        state_only_replay=bool(options.state_only_replay),
    )
