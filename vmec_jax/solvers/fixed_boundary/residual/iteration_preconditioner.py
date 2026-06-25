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


class PreconditionedResidualScalarResult(NamedTuple):
    """Scalar diagnostics produced from preconditioned residual blocks."""

    use_host_fsq1_norms: bool
    frzl_pre_host: Any
    gcr2_p: Any
    gcz2_p: Any
    gcl2_p: Any
    rz_norm: Any
    f_norm1: Any
    fsqr1: Any
    fsqz1: Any
    fsql1: Any
    fsqr1_safe: Any
    fsqz1_safe: Any
    fsql1_safe: Any
    fsq1: Any
    accepted_control_ptau_host: tuple[float, float] | None
    control_payload_used: bool


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


def _preconditioned_gcx2_channels(
    *,
    preconditioner_payload: ResidualIterationPreconditionerResult,
    frzl_pre: Any,
    state: Any,
    static: Any,
    s: Any,
    host_update_assembly: bool,
    host_fsq1_norms_on_accelerator: bool,
    backend_name: str,
    tree_has_tracer_func: Callable[[Any], bool],
    tomnsps_to_numpy_host_func: Callable[[Any], Any],
    vmec_gcx2_from_tomnsps_np_func: Callable[..., Any],
    vmec_gcx2_from_tomnsps_func: Callable[..., Any],
) -> tuple[bool, Any, Any, Any, Any]:
    use_host_fsq1_norms = (
        bool(host_fsq1_norms_on_accelerator)
        and (not bool(host_update_assembly))
        and backend_name != "cpu"
        and (not tree_has_tracer_func(state))
        and (not tree_has_tracer_func(frzl_pre))
    )
    if preconditioner_payload.fsq1_ready:
        return (
            use_host_fsq1_norms,
            None,
            preconditioner_payload.gcr2_p,
            preconditioner_payload.gcz2_p,
            preconditioner_payload.gcl2_p,
        )
    if host_update_assembly or use_host_fsq1_norms:
        frzl_pre_host = frzl_pre if host_update_assembly else tomnsps_to_numpy_host_func(frzl_pre)
        gcr2_p, gcz2_p, gcl2_p = vmec_gcx2_from_tomnsps_np_func(
            frzl=frzl_pre_host,
            include_edge=True,
        )
        return use_host_fsq1_norms, frzl_pre_host, gcr2_p, gcz2_p, gcl2_p
    gcr2_p, gcz2_p, gcl2_p = vmec_gcx2_from_tomnsps_func(
        frzl=frzl_pre,
        lconm1=bool(getattr(static.cfg, "lconm1", True)),
        apply_m1_constraints=False,
        include_edge=True,
        apply_scalxc=False,
        s=s,
    )
    return use_host_fsq1_norms, None, gcr2_p, gcz2_p, gcl2_p


def _preconditioned_scalar_channels(
    *,
    preconditioner_payload: ResidualIterationPreconditionerResult,
    gcr2_p: Any,
    gcz2_p: Any,
    gcl2_p: Any,
    frzl_pre: Any,
    frzl_pre_host: Any,
    use_host_fsq1_norms: bool,
    host_update_assembly: bool,
    vmec2000_control: bool,
    precond_cache: Any,
    need_bcovar_update: bool,
    state: Any,
    delta_s: Any,
    numpy_module: Any,
    jnp_module: Any,
    rz_norm_np_func: Callable[[Any], Any],
    rz_norm_func: Callable[[Any], Any],
    lambda_preconditioned_full_norm_func: Callable[..., Any],
    finite_float_or_zero_func: Callable[[Any], float],
    cached_or_current_f_norm1_jax_func: Callable[..., Any],
    host_preconditioned_residual_scalar_channels_func: Callable[..., Any],
    jax_preconditioned_residual_scalar_channels_func: Callable[..., Any],
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any]:
    if preconditioner_payload.fsq1_ready:
        fsqr1 = preconditioner_payload.fsqr1_safe
        fsqz1 = preconditioner_payload.fsqz1_safe
        fsql1 = preconditioner_payload.fsql1_safe
        return (None, None, fsqr1, fsqz1, fsql1, fsqr1, fsqz1, fsql1, preconditioner_payload.fsq1_safe)
    if host_update_assembly or use_host_fsq1_norms:
        return host_preconditioned_residual_scalar_channels_func(
            gcr2_p=gcr2_p,
            gcz2_p=gcz2_p,
            gcl2_p=gcl2_p,
            frzl_pre=frzl_pre,
            frzl_pre_host=frzl_pre_host,
            vmec2000_control=bool(vmec2000_control),
            vmec2000_cache_valid=bool(precond_cache.valid),
            need_bcovar_update=bool(need_bcovar_update),
            cache_rz_norm=precond_cache.rz_norm,
            cache_f_norm1=precond_cache.f_norm1,
            state=state,
            delta_s=float(delta_s),
            numpy_module=numpy_module,
            rz_norm_np=rz_norm_np_func,
            lambda_preconditioned_full_norm=lambda_preconditioned_full_norm_func,
            finite_float_or_zero=finite_float_or_zero_func,
        )
    return jax_preconditioned_residual_scalar_channels_func(
        gcr2_p=gcr2_p,
        gcz2_p=gcz2_p,
        gcl2_p=gcl2_p,
        frzl_pre=frzl_pre,
        vmec2000_control=bool(vmec2000_control),
        vmec2000_cache_valid=bool(precond_cache.valid),
        need_bcovar_update=bool(need_bcovar_update),
        cache_rz_norm=precond_cache.rz_norm,
        cache_f_norm1=precond_cache.f_norm1,
        state=state,
        delta_s=delta_s,
        jnp_module=jnp_module,
        cached_or_current_f_norm1_jax=cached_or_current_f_norm1_jax_func,
        rz_norm_func=rz_norm_func,
        lambda_preconditioned_full_norm=lambda_preconditioned_full_norm_func,
    )


def resolve_preconditioned_residual_scalars(
    *,
    preconditioner_payload: ResidualIterationPreconditionerResult,
    frzl_pre: Any,
    state: Any,
    static: Any,
    k: Any,
    s: Any,
    delta_s: Any,
    host_update_assembly: bool,
    host_fsq1_norms_on_accelerator: bool,
    backend_name: str,
    vmec2000_control: bool,
    precond_cache: Any,
    need_bcovar_update: bool,
    converged_physical: bool,
    reference_mode: bool,
    badjac_use_state: bool,
    dump_ptau_state: bool,
    dump_ptau_env: str,
    timing_enabled: bool,
    timing_stats: dict[str, Any],
    t_fsq1_precond_norm_start: float | None,
    t_iteration_control_fsq1_start: float | None,
    perf_counter: Callable[[], float],
    record_timing: Callable[[str, float | None], None],
    tree_has_tracer_func: Callable[[Any], bool],
    tomnsps_to_numpy_host_func: Callable[[Any], Any],
    vmec_gcx2_from_tomnsps_np_func: Callable[..., Any],
    vmec_gcx2_from_tomnsps_func: Callable[..., Any],
    host_preconditioned_residual_scalar_channels_func: Callable[..., Any],
    jax_preconditioned_residual_scalar_channels_func: Callable[..., Any],
    materialize_accepted_control_payload_func: Callable[..., Any],
    numpy_module: Any,
    jnp_module: Any,
    jax_module: Any,
    rz_norm_np_func: Callable[[Any], Any],
    rz_norm_func: Callable[[Any], Any],
    lambda_preconditioned_full_norm_func: Callable[..., Any],
    finite_float_or_zero_func: Callable[[Any], float],
    cached_or_current_f_norm1_jax_func: Callable[..., Any],
    dump_lam_fsql1_func: Callable[[Any], None] | None,
    device_get_floats_func: Callable[..., tuple[float, ...]],
    accepted_control_ptau_host_from_payload_func: Callable[..., Any],
    scan_math_kernel_arrays_from_k_func: Callable[..., Any],
    accepted_control_payload_jit_func: Callable[..., Any],
    ptau_pshalf_jax: Any,
    ptau_ohs_jax: Any,
) -> PreconditionedResidualScalarResult:
    """Build preconditioned residual scalar channels and optional control payload."""

    use_host_fsq1_norms, frzl_pre_host, gcr2_p, gcz2_p, gcl2_p = _preconditioned_gcx2_channels(
        preconditioner_payload=preconditioner_payload,
        frzl_pre=frzl_pre,
        state=state,
        static=static,
        s=s,
        host_update_assembly=host_update_assembly,
        host_fsq1_norms_on_accelerator=host_fsq1_norms_on_accelerator,
        backend_name=backend_name,
        tree_has_tracer_func=tree_has_tracer_func,
        tomnsps_to_numpy_host_func=tomnsps_to_numpy_host_func,
        vmec_gcx2_from_tomnsps_np_func=vmec_gcx2_from_tomnsps_np_func,
        vmec_gcx2_from_tomnsps_func=vmec_gcx2_from_tomnsps_func,
    )
    scalars = _preconditioned_scalar_channels(
        preconditioner_payload=preconditioner_payload,
        gcr2_p=gcr2_p,
        gcz2_p=gcz2_p,
        gcl2_p=gcl2_p,
        frzl_pre=frzl_pre,
        frzl_pre_host=frzl_pre_host,
        use_host_fsq1_norms=use_host_fsq1_norms,
        host_update_assembly=host_update_assembly,
        vmec2000_control=vmec2000_control,
        precond_cache=precond_cache,
        need_bcovar_update=need_bcovar_update,
        state=state,
        delta_s=delta_s,
        numpy_module=numpy_module,
        jnp_module=jnp_module,
        rz_norm_np_func=rz_norm_np_func,
        rz_norm_func=rz_norm_func,
        lambda_preconditioned_full_norm_func=lambda_preconditioned_full_norm_func,
        finite_float_or_zero_func=finite_float_or_zero_func,
        cached_or_current_f_norm1_jax_func=cached_or_current_f_norm1_jax_func,
        host_preconditioned_residual_scalar_channels_func=host_preconditioned_residual_scalar_channels_func,
        jax_preconditioned_residual_scalar_channels_func=jax_preconditioned_residual_scalar_channels_func,
    )
    rz_norm, f_norm1, fsqr1, fsqz1, fsql1, fsqr1_safe, fsqz1_safe, fsql1_safe, fsq1 = scalars
    record_timing("iteration_control_fsq1_precond_norm", t_fsq1_precond_norm_start)
    if dump_lam_fsql1_func is not None:
        dump_lam_fsql1_func(fsql1)

    accepted_control_ptau_host: tuple[float, float] | None = None
    control_payload_used = False
    if not (host_update_assembly or use_host_fsq1_norms):
        t_fsq1_scalar_build_start = perf_counter() if timing_enabled else None
        fsq1_j = fsq1_safe if preconditioner_payload.fsq1_ready else fsq1
        record_timing("iteration_control_fsq1_scalar_build", t_fsq1_scalar_build_start)
        use_control_payload = (
            (not bool(converged_physical))
            and (bool(reference_mode) or bool(vmec2000_control))
            and (not bool(badjac_use_state))
            and (not bool(dump_ptau_state))
            and dump_ptau_env in ("", "0")
            and backend_name != "cpu"
        )
        control_payload = materialize_accepted_control_payload_func(
            accepted_control_ptau_payload=preconditioner_payload.accepted_control_ptau_payload,
            use_control_payload=bool(use_control_payload),
            fsq1_j=fsq1_j,
            k=k,
            ptau_pshalf_jax=ptau_pshalf_jax,
            ptau_ohs_jax=ptau_ohs_jax,
            timing_enabled=bool(timing_enabled),
            timing_stats=timing_stats,
            perf_counter=perf_counter,
            jax_module=jax_module,
            device_get_floats=device_get_floats_func,
            accepted_control_ptau_host_from_payload=accepted_control_ptau_host_from_payload_func,
            scan_math_kernel_arrays_from_k=scan_math_kernel_arrays_from_k_func,
            accepted_control_payload_jit=accepted_control_payload_jit_func,
        )
        fsq1 = control_payload.fsq1
        accepted_control_ptau_host = control_payload.accepted_control_ptau_host
        control_payload_used = control_payload.control_payload_used
    record_timing("iteration_control_fsq1", t_iteration_control_fsq1_start)

    return PreconditionedResidualScalarResult(
        use_host_fsq1_norms=bool(use_host_fsq1_norms),
        frzl_pre_host=frzl_pre_host,
        gcr2_p=gcr2_p,
        gcz2_p=gcz2_p,
        gcl2_p=gcl2_p,
        rz_norm=rz_norm,
        f_norm1=f_norm1,
        fsqr1=fsqr1,
        fsqz1=fsqz1,
        fsql1=fsql1,
        fsqr1_safe=fsqr1_safe,
        fsqz1_safe=fsqz1_safe,
        fsql1_safe=fsql1_safe,
        fsq1=fsq1,
        accepted_control_ptau_host=accepted_control_ptau_host,
        control_payload_used=bool(control_payload_used),
    )


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
