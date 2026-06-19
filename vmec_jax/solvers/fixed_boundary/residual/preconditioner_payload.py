"""JIT payload facade for residual-iteration preconditioned updates."""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

from vmec_jax._compat import has_jax
from vmec_jax.solvers.fixed_boundary.preconditioning import payload as _payload
from vmec_jax.vmec_tomnsp import TomnspsRZL


def _jax_available(has_jax_func=None) -> bool:
    """Evaluate the caller-provided JAX availability hook when supplied."""

    return bool((has_jax if has_jax_func is None else has_jax_func)())


_STRICT_UPDATE_STEP_JIT_CACHE = _payload.STRICT_UPDATE_STEP_JIT_CACHE
_PRECOND_OUTPUT_SCALE_JIT_CACHE = _payload.PRECOND_OUTPUT_SCALE_JIT_CACHE
_PRECOND_OUTPUT_PAYLOAD_JIT_CACHE = _payload.PRECOND_OUTPUT_PAYLOAD_JIT_CACHE
_PRECOND_APPLY_PAYLOAD_JIT_CACHE = _payload.PRECOND_APPLY_PAYLOAD_JIT_CACHE
_ACCEPTED_CONTROL_PAYLOAD_JIT_CACHE = _payload.ACCEPTED_CONTROL_PAYLOAD_JIT_CACHE


class PreconditionerRefreshRuntimeResult(NamedTuple):
    """Updated preconditioner cache state and payload for one refresh point."""

    lam_prec: Any
    mats: dict[str, Any]
    jmax: int
    need_lam_prec: bool
    need_lamcal: bool
    cache_update_trace: bool
    cache_prec_lam_prec: Any
    cache_prec_faclam: Any
    cache_prec_lam_debug: Any
    cache_prec_rz_mats: Any
    cache_prec_rz_jmax: Any


def _strict_update_step_jit(
    static,
    *,
    limit_update_rms: bool,
    need_update_rms: bool,
    divide_by_scalxc_for_update: bool,
    enforce_edge: bool = True,
    has_jax_func=None,
):
    """Return the cached strict-update JIT, or ``None`` when JAX is unavailable."""

    if not _jax_available(has_jax_func):
        return None
    return _payload.strict_update_step_jit(
        static,
        limit_update_rms=limit_update_rms,
        need_update_rms=need_update_rms,
        divide_by_scalxc_for_update=divide_by_scalxc_for_update,
        enforce_edge=enforce_edge,
    )


def _preconditioner_output_scaling_jit(*, apply_lambda_update_scale: bool, has_jax_func=None):
    """Return the cached output-scaling JIT, or ``None`` when JAX is unavailable."""

    if not _jax_available(has_jax_func):
        return None
    return _payload.preconditioner_output_scaling_jit(apply_lambda_update_scale=apply_lambda_update_scale)


def _preconditioner_output_payload_jit(
    *,
    apply_lambda_update_scale: bool,
    vmec2000_control: bool,
    lconm1: bool,
    has_jax_func=None,
):
    """Return the cached residual-output payload JIT."""

    if not _jax_available(has_jax_func):
        return None
    return _payload.preconditioner_output_payload_jit(
        apply_lambda_update_scale=apply_lambda_update_scale,
        vmec2000_control=vmec2000_control,
        lconm1=lconm1,
        scaling_func=lambda **kwargs: _preconditioner_output_scaling_jit(
            has_jax_func=has_jax_func,
            **kwargs,
        ),
    )


def _preconditioner_apply_payload_jit(
    *,
    jmax: int,
    lthreed: bool,
    lasym: bool,
    use_precomputed: bool,
    use_lax_tridi: bool,
    has_lax_t: bool,
    has_frss: bool,
    has_fzcs: bool,
    has_frsc: bool,
    has_frcs: bool,
    has_fzcc: bool,
    has_fzss: bool,
    has_flcs: bool,
    has_flcc: bool,
    has_flss: bool,
    apply_lambda_update_scale: bool,
    vmec2000_control: bool,
    lconm1: bool,
    include_control_ptau: bool,
    has_jax_func=None,
):
    """Return the cached preconditioner-apply payload JIT."""

    if not _jax_available(has_jax_func):
        return None
    return _payload.preconditioner_apply_payload_jit(
        jmax=jmax,
        lthreed=lthreed,
        lasym=lasym,
        use_precomputed=use_precomputed,
        use_lax_tridi=use_lax_tridi,
        has_lax_t=has_lax_t,
        has_frss=has_frss,
        has_fzcs=has_fzcs,
        has_frsc=has_frsc,
        has_frcs=has_frcs,
        has_fzcc=has_fzcc,
        has_fzss=has_fzss,
        has_flcs=has_flcs,
        has_flcc=has_flcc,
        has_flss=has_flss,
        apply_lambda_update_scale=apply_lambda_update_scale,
        vmec2000_control=vmec2000_control,
        lconm1=lconm1,
        include_control_ptau=include_control_ptau,
    )


def _accepted_control_payload_jit(*, has_jax_func=None):
    """Return the cached accepted-controller payload JIT."""

    if not _jax_available(has_jax_func):
        return None
    return _payload.accepted_control_payload_jit()


def _preconditioner_apply_payload_fused(
    *,
    frzl_in: TomnspsRZL,
    mats: dict[str, Any],
    jmax: int,
    cfg,
    lam_prec,
    w_mode_mn,
    lambda_update_scale_j,
    f_norm1,
    delta_s,
    s,
    use_precomputed: bool | None,
    use_lax_tridi: bool | None,
    apply_lambda_update_scale: bool,
    vmec2000_control: bool,
    lconm1: bool,
    include_control_ptau: bool = False,
    control_ptau_arrays: tuple[Any, ...] | None = None,
    control_ptau_pshalf: Any = None,
    control_ptau_ohs: Any = None,
):
    """Apply the fused preconditioner payload using the shared JIT factory."""

    return _payload.preconditioner_apply_payload_fused(
        frzl_in=frzl_in,
        mats=mats,
        jmax=jmax,
        cfg=cfg,
        lam_prec=lam_prec,
        w_mode_mn=w_mode_mn,
        lambda_update_scale_j=lambda_update_scale_j,
        f_norm1=f_norm1,
        delta_s=delta_s,
        s=s,
        use_precomputed=use_precomputed,
        use_lax_tridi=use_lax_tridi,
        apply_lambda_update_scale=apply_lambda_update_scale,
        vmec2000_control=vmec2000_control,
        lconm1=lconm1,
        include_control_ptau=include_control_ptau,
        control_ptau_arrays=control_ptau_arrays,
        control_ptau_pshalf=control_ptau_pshalf,
        control_ptau_ohs=control_ptau_ohs,
        apply_payload_jit_func=_preconditioner_apply_payload_jit,
    )


def _split_preconditioner_apply_payload(payload):
    if len(payload) == 4:
        return payload
    pre_blocks, update_blocks, diag = payload
    return pre_blocks, update_blocks, diag, None


def refresh_preconditioner_cache_runtime(
    *,
    k: Any,
    iter2: int,
    cfg: Any,
    static: Any,
    env_dump_lam: str,
    env_dump_lamcal: str,
    timing_enabled: bool,
    timing_stats: dict[str, Any],
    perf_counter: Callable[[], float],
    block_until_ready: Callable[[Any], Any] | None,
    tree_has_tracer: Callable[[Any], bool],
    update_preconditioner_cache_func: Callable[..., Any],
    can_reassemble_func: Callable[..., bool],
    lambda_preconditioner_func: Callable[..., Any],
    rz_preconditioner_matrices_func: Callable[..., Any],
    maybe_dump_lam_prec: Callable[..., None],
    maybe_dump_precond_mats: Callable[..., None],
    maybe_dump_lamcal: Callable[..., None],
    vmec2000_cache_valid: bool,
    need_bcovar_update: bool,
    precond_cache_seeded_from_bcovar_update: bool,
    precond_expected_jmax: int,
    precond_jmax_override: int | None,
    preconditioner_use_precomputed_tridi: bool | None,
    preconditioner_use_lax_tridi: bool | None,
    cache_prec_lam_prec: Any,
    cache_prec_faclam: Any,
    cache_prec_lam_debug: Any,
    cache_prec_rz_mats: Any,
    cache_prec_rz_jmax: Any,
) -> PreconditionerRefreshRuntimeResult:
    """Refresh or reuse the VMEC2000-style radial preconditioner cache."""

    from vmec_jax.preconditioner_1d_jax import rz_preconditioner_matrices_reassemble

    precond_traced = tree_has_tracer(k)
    need_lam_prec = env_dump_lam not in ("", "0")
    need_lamcal = env_dump_lamcal not in ("", "0")
    t_prec_refresh_start = perf_counter() if timing_enabled else None
    precond_cache_update = update_preconditioner_cache_func(
        bc=k.bc,
        k=k,
        cfg=cfg,
        precond_traced=bool(precond_traced),
        vmec2000_cache_valid=bool(vmec2000_cache_valid),
        need_bcovar_update=bool(need_bcovar_update),
        precond_cache_seeded_from_bcovar_update=bool(precond_cache_seeded_from_bcovar_update),
        need_lam_prec=bool(need_lam_prec),
        need_lamcal=bool(need_lamcal),
        cache_prec_lam_prec=cache_prec_lam_prec,
        cache_prec_faclam=cache_prec_faclam,
        cache_prec_lam_debug=cache_prec_lam_debug,
        cache_prec_rz_mats=cache_prec_rz_mats,
        cache_prec_rz_jmax=cache_prec_rz_jmax,
        precond_expected_jmax=int(precond_expected_jmax),
        precond_jmax_override=precond_jmax_override,
        preconditioner_use_precomputed_tridi=preconditioner_use_precomputed_tridi,
        preconditioner_use_lax_tridi=preconditioner_use_lax_tridi,
        lambda_preconditioner_func=lambda_preconditioner_func,
        rz_preconditioner_matrices_func=rz_preconditioner_matrices_func,
        rz_preconditioner_matrices_reassemble_func=rz_preconditioner_matrices_reassemble,
        can_reassemble_func=can_reassemble_func,
    )
    precond_cache_decision = precond_cache_update.decision
    need_prec_refresh = precond_cache_decision.need_prec_refresh
    cache_update_trace = bool(need_prec_refresh)
    if need_prec_refresh:
        if timing_enabled:
            timing_stats["precond_refresh_calls"] = int(timing_stats["precond_refresh_calls"]) + 1
        if timing_enabled and t_prec_refresh_start is not None:
            try:
                if block_until_ready is not None:
                    block_until_ready(precond_cache_update.lam_prec)
            except Exception:
                pass
            timing_stats["precond_refresh"] += perf_counter() - float(t_prec_refresh_start)
    else:
        if timing_enabled:
            timing_stats["precond_cache_hit_count"] = int(timing_stats["precond_cache_hit_count"]) + 1
            if bool(precond_cache_decision.can_reuse_bcovar_seeded_precond) and bool(need_bcovar_update):
                timing_stats["precond_refresh_seed_reuse_count"] = (
                    int(timing_stats["precond_refresh_seed_reuse_count"]) + 1
                )
        if bool(precond_cache_decision.need_prec_reassemble) and timing_enabled:
            timing_stats["precond_reassemble_calls"] = int(timing_stats["precond_reassemble_calls"]) + 1

    maybe_dump_lam_prec(
        lam_prec=precond_cache_update.lam_prec,
        faclam=precond_cache_update.faclam_dump,
        static=static,
        iter_idx=int(iter2),
    )
    if not precond_traced:
        maybe_dump_precond_mats(
            mats=precond_cache_update.mats,
            static=static,
            iter_idx=int(iter2),
            jmax=int(precond_cache_update.jmax),
            used_cache=(not bool(need_prec_refresh)),
        )
    if precond_cache_update.lam_debug is not None:
        maybe_dump_lamcal(lam_debug=precond_cache_update.lam_debug, static=static, iter_idx=int(iter2))

    return PreconditionerRefreshRuntimeResult(
        lam_prec=precond_cache_update.lam_prec,
        mats=precond_cache_update.mats,
        jmax=precond_cache_update.jmax,
        need_lam_prec=need_lam_prec,
        need_lamcal=need_lamcal,
        cache_update_trace=cache_update_trace,
        cache_prec_lam_prec=precond_cache_update.cache_prec_lam_prec,
        cache_prec_faclam=precond_cache_update.cache_prec_faclam,
        cache_prec_lam_debug=precond_cache_update.cache_prec_lam_debug,
        cache_prec_rz_mats=precond_cache_update.cache_prec_rz_mats,
        cache_prec_rz_jmax=precond_cache_update.cache_prec_rz_jmax,
    )


_ptau_compute_jit = _payload.ptau_compute_jit


__all__ = [
    "_ACCEPTED_CONTROL_PAYLOAD_JIT_CACHE",
    "_PRECOND_APPLY_PAYLOAD_JIT_CACHE",
    "_PRECOND_OUTPUT_PAYLOAD_JIT_CACHE",
    "_PRECOND_OUTPUT_SCALE_JIT_CACHE",
    "_STRICT_UPDATE_STEP_JIT_CACHE",
    "_accepted_control_payload_jit",
    "_preconditioner_apply_payload_fused",
    "_preconditioner_apply_payload_jit",
    "_preconditioner_output_payload_jit",
    "_preconditioner_output_scaling_jit",
    "_ptau_compute_jit",
    "_strict_update_step_jit",
    "PreconditionerRefreshRuntimeResult",
    "refresh_preconditioner_cache_runtime",
]
