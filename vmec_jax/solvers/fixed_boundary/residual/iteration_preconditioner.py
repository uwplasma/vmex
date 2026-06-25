"""Preconditioner application for one residual-iteration step."""

from __future__ import annotations

from typing import Any, Callable, NamedTuple


class ResidualIterationPreconditionerResult(NamedTuple):
    """Payloads produced by one preconditioner application."""

    lam_prec: Any
    mats: dict[str, Any] | None
    jmax: int | None
    cache_update_trace: bool
    preconditioned_blocks: Any
    update_force_blocks: Any
    frzl_pre: Any
    frzl_rz: Any
    frzl_lam_pre: Any
    outputs_scaled: bool
    fsq1_ready: bool
    gcr2_p: Any
    gcz2_p: Any
    gcl2_p: Any
    fsqr1_safe: Any
    fsqz1_safe: Any
    fsql1_safe: Any
    fsq1_safe: Any
    accepted_control_ptau_payload: Any


def _empty_preconditioner_payload() -> ResidualIterationPreconditionerResult:
    return ResidualIterationPreconditionerResult(
        lam_prec=None,
        mats=None,
        jmax=None,
        cache_update_trace=False,
        preconditioned_blocks=None,
        update_force_blocks=None,
        frzl_pre=None,
        frzl_rz=None,
        frzl_lam_pre=None,
        outputs_scaled=False,
        fsq1_ready=False,
        gcr2_p=None,
        gcz2_p=None,
        gcl2_p=None,
        fsqr1_safe=None,
        fsqz1_safe=None,
        fsql1_safe=None,
        fsq1_safe=None,
        accepted_control_ptau_payload=None,
    )


def _apply_vmec2000_preconditioner_branch(
    *,
    frzl: Any,
    k: Any,
    state: Any,
    iter2: int,
    cfg: Any,
    static: Any,
    s: Any,
    delta_s: Any,
    w_mode_mn: Any,
    lambda_update_scale: float,
    lambda_update_scale_j: Any,
    vmec2000_control: bool,
    precond_cache: Any,
    need_bcovar_update: bool,
    host_update_assembly: bool,
    use_fused_precond_output_scaling: bool,
    adjoint_trace: bool,
    adjoint_trace_mode: str,
    accepted_control_ptau_arrays: Any,
    ptau_pshalf_jax: Any,
    ptau_ohs_jax: Any,
    preconditioner_use_precomputed_tridi_policy: bool | None,
    preconditioner_use_lax_tridi_policy: bool | None,
    timing_detail_enabled: bool,
    timing_stats: dict[str, Any],
    perf_counter: Callable[[], float],
    block_until_ready: Callable[[Any], Any] | None,
    refresh_preconditioner_cache_func: Callable[..., Any],
    scale_m1_precond_rhs_func: Callable[..., Any],
    rz_preconditioner_apply_func: Callable[..., Any],
    rz_norm_func: Callable[[Any], Any],
    apply_vmec2000_preconditioner_runtime_func: Callable[..., Any],
) -> ResidualIterationPreconditionerResult:
    precond_apply = apply_vmec2000_preconditioner_runtime_func(
        frzl=frzl,
        k=k,
        state=state,
        iter2=int(iter2),
        cfg=cfg,
        s=s,
        delta_s=delta_s,
        w_mode_mn=w_mode_mn,
        lambda_update_scale=float(lambda_update_scale),
        lambda_update_scale_j=lambda_update_scale_j,
        lconm1=bool(getattr(static.cfg, "lconm1", True)),
        vmec2000_control=bool(vmec2000_control),
        vmec2000_cache_valid=bool(precond_cache.valid),
        need_bcovar_update=bool(need_bcovar_update),
        cache_rz_norm=precond_cache.rz_norm,
        cache_f_norm1=precond_cache.f_norm1,
        host_update_assembly=bool(host_update_assembly),
        use_fused_precond_output_scaling=bool(use_fused_precond_output_scaling),
        scale_m1_rhs=bool(cfg.lthreed) or bool(getattr(cfg, "lasym", False)),
        adjoint_trace=bool(adjoint_trace),
        adjoint_trace_mode=adjoint_trace_mode,
        accepted_control_ptau_arrays=accepted_control_ptau_arrays,
        ptau_pshalf_jax=ptau_pshalf_jax,
        ptau_ohs_jax=ptau_ohs_jax,
        preconditioner_use_precomputed_tridi=preconditioner_use_precomputed_tridi_policy,
        preconditioner_use_lax_tridi=preconditioner_use_lax_tridi_policy,
        timing_detail_enabled=bool(timing_detail_enabled),
        timing_stats=timing_stats,
        perf_counter=perf_counter,
        block_until_ready=block_until_ready,
        refresh_preconditioner_cache_func=refresh_preconditioner_cache_func,
        scale_m1_precond_rhs_func=scale_m1_precond_rhs_func,
        rz_preconditioner_apply_func=rz_preconditioner_apply_func,
        rz_norm_func=rz_norm_func,
    )

    return _empty_preconditioner_payload()._replace(
        lam_prec=precond_apply.lam_prec,
        mats=precond_apply.mats,
        jmax=precond_apply.jmax,
        cache_update_trace=precond_apply.cache_update_trace,
        preconditioned_blocks=precond_apply.blocks,
        update_force_blocks=precond_apply.update_blocks,
        frzl_rz=precond_apply.frzl_rz,
        frzl_lam_pre=precond_apply.frzl_lam_pre,
        outputs_scaled=precond_apply.outputs_scaled,
        fsq1_ready=precond_apply.fsq1_ready,
        gcr2_p=precond_apply.gcr2_p,
        gcz2_p=precond_apply.gcz2_p,
        gcl2_p=precond_apply.gcl2_p,
        fsqr1_safe=precond_apply.fsqr1_safe,
        fsqz1_safe=precond_apply.fsqz1_safe,
        fsql1_safe=precond_apply.fsql1_safe,
        fsq1_safe=precond_apply.fsq1_safe,
        accepted_control_ptau_payload=precond_apply.accepted_control_ptau_payload,
    )


def _apply_radial_preconditioner_branch(
    *,
    frzl: Any,
    rz_scale: float,
    l_scale: float,
    precond_radial_alpha: float,
    precond_lambda_alpha: float,
    timing_detail_enabled: bool,
    perf_counter: Callable[[], float],
    record_timing: Callable[[str, float | None], None],
    has_jax_func: Callable[[], bool],
    block_until_ready: Callable[[Any], Any] | None,
    radial_preconditioner_output_blocks_jax_func: Callable[..., Any],
    apply_radial_tridi_func: Callable[..., Any],
) -> ResidualIterationPreconditionerResult:
    t_precond_apply_start = perf_counter() if timing_detail_enabled else None
    preconditioned_blocks = radial_preconditioner_output_blocks_jax_func(
        frzl=frzl,
        rz_scale=rz_scale,
        l_scale=l_scale,
        precond_radial_alpha=precond_radial_alpha,
        precond_lambda_alpha=precond_lambda_alpha,
        apply_radial_tridi_func=apply_radial_tridi_func,
    )
    if timing_detail_enabled and t_precond_apply_start is not None:
        try:
            if has_jax_func() and block_until_ready is not None:
                block_until_ready(preconditioned_blocks.flsc)
        except Exception:
            pass
        record_timing("precond_apply", t_precond_apply_start)
    return _empty_preconditioner_payload()._replace(preconditioned_blocks=preconditioned_blocks)


def _scale_preconditioned_update_blocks(
    payload: ResidualIterationPreconditionerResult,
    *,
    w_mode_mn: Any,
    w_mode_mn_np: Any,
    lambda_update_scale: float,
    lambda_update_scale_j: Any,
    host_update_assembly: bool,
    timing_enabled: bool,
    timing_detail_enabled: bool,
    t_precond_start: float | None,
    perf_counter: Callable[[], float],
    record_timing: Callable[[str, float | None], None],
    has_jax_func: Callable[[], bool],
    block_until_ready: Callable[[Any], Any] | None,
    tomnsps_type: Callable[..., Any],
    mode_weight_force_blocks_np_func: Callable[..., Any],
    mode_weight_force_blocks_jax_func: Callable[..., Any],
    zeros_coeff_np: Any,
) -> ResidualIterationPreconditionerResult:
    frzl_pre = tomnsps_type(**payload.preconditioned_blocks._asdict())
    update_force_blocks = payload.update_force_blocks

    t_precond_mode_start = perf_counter() if timing_detail_enabled else None
    if payload.outputs_scaled:
        pass
    elif host_update_assembly:
        update_force_blocks = mode_weight_force_blocks_np_func(
            payload.preconditioned_blocks,
            w_mode_mn=w_mode_mn_np,
            zeros_coeff=zeros_coeff_np,
        )
    else:
        update_force_blocks = mode_weight_force_blocks_jax_func(
            payload.preconditioned_blocks,
            w_mode_mn=w_mode_mn,
        )
    if timing_detail_enabled and t_precond_mode_start is not None:
        try:
            if has_jax_func() and block_until_ready is not None:
                block_until_ready(update_force_blocks.flsc)
        except Exception:
            pass
        record_timing("precond_mode_scale", t_precond_mode_start)
    if timing_enabled:
        try:
            if has_jax_func() and (not timing_detail_enabled) and block_until_ready is not None:
                block_until_ready(update_force_blocks.flsc)
        except Exception:
            pass
        record_timing("preconditioner", t_precond_start)

    if (lambda_update_scale != 1.0) and (not payload.outputs_scaled):
        update_force_blocks = update_force_blocks._replace(
            flsc=update_force_blocks.flsc * lambda_update_scale_j,
            flcs=update_force_blocks.flcs * lambda_update_scale_j,
            flcc=update_force_blocks.flcc * lambda_update_scale_j,
            flss=update_force_blocks.flss * lambda_update_scale_j,
        )
    return payload._replace(frzl_pre=frzl_pre, update_force_blocks=update_force_blocks)


def apply_residual_iteration_preconditioner(
    *,
    use_vmec2000_preconditioner: bool,
    frzl: Any,
    k: Any,
    state: Any,
    iter2: int,
    cfg: Any,
    static: Any,
    s: Any,
    delta_s: Any,
    w_mode_mn: Any,
    w_mode_mn_np: Any,
    lambda_update_scale: float,
    lambda_update_scale_j: Any,
    vmec2000_control: bool,
    precond_cache: Any,
    need_bcovar_update: bool,
    host_update_assembly: bool,
    use_fused_precond_output_scaling: bool,
    adjoint_trace: bool,
    adjoint_trace_mode: str,
    accepted_control_ptau_arrays: Any,
    ptau_pshalf_jax: Any,
    ptau_ohs_jax: Any,
    preconditioner_use_precomputed_tridi_policy: bool | None,
    preconditioner_use_lax_tridi_policy: bool | None,
    timing_enabled: bool,
    timing_detail_enabled: bool,
    timing_stats: dict[str, Any],
    t_precond_start: float | None,
    perf_counter: Callable[[], float],
    record_timing: Callable[[str, float | None], None],
    has_jax_func: Callable[[], bool],
    block_until_ready: Callable[[Any], Any] | None,
    tomnsps_type: Callable[..., Any],
    refresh_preconditioner_cache_func: Callable[..., Any],
    scale_m1_precond_rhs_func: Callable[..., Any],
    rz_preconditioner_apply_func: Callable[..., Any],
    rz_norm_func: Callable[[Any], Any],
    apply_vmec2000_preconditioner_runtime_func: Callable[..., Any],
    radial_preconditioner_output_blocks_jax_func: Callable[..., Any],
    apply_radial_tridi_func: Callable[..., Any],
    mode_weight_force_blocks_np_func: Callable[..., Any],
    mode_weight_force_blocks_jax_func: Callable[..., Any],
    zeros_coeff_np: Any,
    rz_scale: float,
    l_scale: float,
    precond_radial_alpha: float,
    precond_lambda_alpha: float,
) -> ResidualIterationPreconditionerResult:
    """Apply the selected residual preconditioner and build update blocks."""

    if use_vmec2000_preconditioner:
        payload = _apply_vmec2000_preconditioner_branch(
            frzl=frzl,
            k=k,
            state=state,
            iter2=iter2,
            cfg=cfg,
            static=static,
            s=s,
            delta_s=delta_s,
            w_mode_mn=w_mode_mn,
            lambda_update_scale=lambda_update_scale,
            lambda_update_scale_j=lambda_update_scale_j,
            vmec2000_control=vmec2000_control,
            precond_cache=precond_cache,
            need_bcovar_update=need_bcovar_update,
            host_update_assembly=host_update_assembly,
            use_fused_precond_output_scaling=use_fused_precond_output_scaling,
            adjoint_trace=adjoint_trace,
            adjoint_trace_mode=adjoint_trace_mode,
            accepted_control_ptau_arrays=accepted_control_ptau_arrays,
            ptau_pshalf_jax=ptau_pshalf_jax,
            ptau_ohs_jax=ptau_ohs_jax,
            preconditioner_use_precomputed_tridi_policy=preconditioner_use_precomputed_tridi_policy,
            preconditioner_use_lax_tridi_policy=preconditioner_use_lax_tridi_policy,
            timing_detail_enabled=timing_detail_enabled,
            timing_stats=timing_stats,
            perf_counter=perf_counter,
            block_until_ready=block_until_ready,
            refresh_preconditioner_cache_func=refresh_preconditioner_cache_func,
            scale_m1_precond_rhs_func=scale_m1_precond_rhs_func,
            rz_preconditioner_apply_func=rz_preconditioner_apply_func,
            rz_norm_func=rz_norm_func,
            apply_vmec2000_preconditioner_runtime_func=apply_vmec2000_preconditioner_runtime_func,
        )
    else:
        payload = _apply_radial_preconditioner_branch(
            frzl=frzl,
            rz_scale=rz_scale,
            l_scale=l_scale,
            precond_radial_alpha=precond_radial_alpha,
            precond_lambda_alpha=precond_lambda_alpha,
            timing_detail_enabled=timing_detail_enabled,
            perf_counter=perf_counter,
            record_timing=record_timing,
            has_jax_func=has_jax_func,
            block_until_ready=block_until_ready,
            radial_preconditioner_output_blocks_jax_func=radial_preconditioner_output_blocks_jax_func,
            apply_radial_tridi_func=apply_radial_tridi_func,
        )

    return _scale_preconditioned_update_blocks(
        payload,
        w_mode_mn=w_mode_mn,
        w_mode_mn_np=w_mode_mn_np,
        lambda_update_scale=lambda_update_scale,
        lambda_update_scale_j=lambda_update_scale_j,
        host_update_assembly=host_update_assembly,
        timing_enabled=timing_enabled,
        timing_detail_enabled=timing_detail_enabled,
        t_precond_start=t_precond_start,
        perf_counter=perf_counter,
        record_timing=record_timing,
        has_jax_func=has_jax_func,
        block_until_ready=block_until_ready,
        tomnsps_type=tomnsps_type,
        mode_weight_force_blocks_np_func=mode_weight_force_blocks_np_func,
        mode_weight_force_blocks_jax_func=mode_weight_force_blocks_jax_func,
        zeros_coeff_np=zeros_coeff_np,
    )
