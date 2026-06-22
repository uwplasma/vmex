"""JIT payload facade for residual-iteration preconditioned updates."""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import numpy as np

from vmec_jax._compat import has_jax, jnp
from vmec_jax.solvers.fixed_boundary.preconditioning import payload as _payload
from vmec_jax.solvers.fixed_boundary.residual.payload_blocks import (
    ForceBlocks,
    preconditioner_output_blocks_jax,
    preconditioner_output_blocks_np,
)
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


class PreconditionerBcovarSeedResult(NamedTuple):
    """Result of seeding the residual preconditioner cache from bcovar fields."""

    cache_update_trace: bool
    seeded_from_bcovar_update: bool
    seed_time_in_residual_metrics: float


class AcceptedControlPayloadMaterialization(NamedTuple):
    """Host scalars needed by VMEC time-control after preconditioned fsq1."""

    fsq1: float
    accepted_control_ptau_host: tuple[float, float] | None
    control_payload_used: bool


class PreconditionedResidualScalarChannels(NamedTuple):
    """Preconditioned residual scalar channels used by VMEC time-control."""

    rz_norm: Any
    f_norm1: Any
    fsqr1: Any
    fsqz1: Any
    fsql1: Any
    fsqr1_safe: Any
    fsqz1_safe: Any
    fsql1_safe: Any
    fsq1: Any


class Vmec2000PreconditionerApplyResult(NamedTuple):
    """VMEC2000-style preconditioner payloads for one residual iteration."""

    lam_prec: Any
    mats: dict[str, Any]
    jmax: int
    cache_update_trace: bool
    blocks: ForceBlocks
    update_blocks: ForceBlocks | None
    gcr2_p: Any
    gcz2_p: Any
    gcl2_p: Any
    fsqr1_safe: Any
    fsqz1_safe: Any
    fsql1_safe: Any
    fsq1_safe: Any
    frzl_rz: Any | None
    frzl_lam_pre: Any | None
    outputs_scaled: bool
    fsq1_ready: bool
    accepted_control_ptau_payload: Any | None


class ResidualPreconditionerOperators(NamedTuple):
    """Callable preconditioner operators bound to one residual-iteration setup."""

    apply_radial_tridi: Callable[[Any, float], Any]
    apply_radial_tridi_batched: Callable[[Any, float], tuple[Any, ...]]
    lambda_preconditioner: Callable[..., Any]
    rz_preconditioner_matrices: Callable[..., Any]
    rz_preconditioner_apply: Callable[..., Any]
    rz_preconditioner: Callable[..., Any]


def residual_preconditioner_operators(
    *,
    trig: Any,
    s: Any,
    cfg: Any,
    use_numpy_preconditioner_apply: bool,
    tree_has_tracer_func: Callable[[Any], bool],
    radial_tridi_smooth_dirichlet_func: Callable[..., Any],
    jnp_module: Any,
) -> ResidualPreconditionerOperators:
    """Bind the preconditioner operator set used by one residual solve.

    The residual iteration needs the same small collection of operators in
    several update and cache-refresh paths. Keeping the bindings together makes
    the main loop read as policy orchestration instead of local function
    construction.
    """

    def apply_radial_tridi(a, alpha: float):
        return radial_tridi_smooth_dirichlet_func(a, alpha=alpha, skip_nonpositive=True)

    def apply_radial_tridi_batched(arrs, alpha: float):
        if alpha <= 0.0:
            return tuple(arrs)
        stack = jnp_module.stack(arrs, axis=1)
        smooth = radial_tridi_smooth_dirichlet_func(stack, alpha=alpha)
        return tuple(smooth[:, i] for i in range(int(smooth.shape[1])))

    def lambda_preconditioner(bc, *, return_faclam: bool = False, return_debug: bool = False):
        lam_r0scale = float(getattr(trig, "r0scale", 1.0)) if trig is not None else 1.0
        from vmec_jax.preconditioner_1d_jax import lambda_preconditioner_cached

        return lambda_preconditioner_cached(
            bc=bc,
            trig=trig,
            s=s,
            cfg=cfg,
            return_faclam=return_faclam,
            return_debug=return_debug,
            r0scale=lam_r0scale,
        )

    def rz_preconditioner_matrices(
        *,
        bc,
        k,
        jmax_override: int | None = None,
        use_precomputed: bool | None = None,
        use_lax_tridi: bool | None = None,
    ):
        if (
            bool(use_numpy_preconditioner_apply)
            and (not bool(getattr(cfg, "lasym", False)))
            and (not bool(getattr(cfg, "lthreed", False)))
            and (not tree_has_tracer_func((bc, k)))
        ):
            # The NumPy R/Z matrix representation is output-equivalent to the
            # JAX path for axisymmetric host updates, but not for the 3D
            # independent-toroidal-block representation.
            from vmec_jax.preconditioner_1d import rz_preconditioner_matrices as build_rz_matrices_np

            return build_rz_matrices_np(
                bc=bc,
                k=k,
                trig=trig,
                s=s,
                cfg=cfg,
                jmax_override=jmax_override,
            )
        from vmec_jax.preconditioner_1d_jax import rz_preconditioner_matrices as build_rz_matrices

        return build_rz_matrices(
            bc=bc,
            k=k,
            trig=trig,
            s=s,
            cfg=cfg,
            jmax_override=jmax_override,
            use_precomputed=use_precomputed,
            use_lax_tridi=use_lax_tridi,
        )

    def rz_preconditioner_apply(
        *,
        frzl_in,
        mats,
        jmax,
        use_precomputed: bool | None = None,
        use_lax_tridi: bool | None = None,
    ):
        if bool(use_numpy_preconditioner_apply) and not tree_has_tracer_func(frzl_in):
            from vmec_jax.preconditioner_1d_jax import rz_preconditioner_apply_numpy

            return rz_preconditioner_apply_numpy(
                frzl_in=frzl_in,
                mats=mats,
                jmax=jmax,
                cfg=cfg,
                use_precomputed=use_precomputed,
            )
        from vmec_jax.preconditioner_1d_jax import rz_preconditioner_apply_jit

        return rz_preconditioner_apply_jit(
            frzl_in=frzl_in,
            mats=mats,
            jmax=jmax,
            cfg=cfg,
            use_precomputed=use_precomputed,
            use_lax_tridi=use_lax_tridi,
        )

    def rz_preconditioner(frzl_in: TomnspsRZL, bc, k):
        from vmec_jax.preconditioner_1d_jax import rz_preconditioner as apply_rz_preconditioner

        return apply_rz_preconditioner(
            frzl_in=frzl_in,
            bc=bc,
            k=k,
            trig=trig,
            s=s,
            cfg=cfg,
        )

    return ResidualPreconditionerOperators(
        apply_radial_tridi,
        apply_radial_tridi_batched,
        lambda_preconditioner,
        rz_preconditioner_matrices,
        rz_preconditioner_apply,
        rz_preconditioner,
    )


def host_preconditioned_residual_scalar_channels(
    *,
    gcr2_p: Any,
    gcz2_p: Any,
    gcl2_p: Any,
    frzl_pre: Any,
    frzl_pre_host: Any,
    vmec2000_control: bool,
    vmec2000_cache_valid: bool,
    need_bcovar_update: bool,
    cache_rz_norm: Any,
    cache_f_norm1: Any,
    state: Any,
    delta_s: float,
    numpy_module: Any,
    rz_norm_np: Callable[[Any], float],
    lambda_preconditioned_full_norm: Callable[..., Any],
    finite_float_or_zero: Callable[[Any], float],
) -> PreconditionedResidualScalarChannels:
    """Build preconditioned fsq1 channels on the NumPy/host path."""

    if (
        bool(vmec2000_control)
        and bool(vmec2000_cache_valid)
        and (not bool(need_bcovar_update))
        and (cache_rz_norm is not None)
        and (cache_f_norm1 is not None)
    ):
        f_norm1_np = float(cache_f_norm1)
        rz_norm = cache_rz_norm
    else:
        rz_norm = rz_norm_np(state)
        f_norm1_np = (1.0 / rz_norm) if rz_norm != 0.0 else float("inf")
    f_norm1 = f_norm1_np
    finite = numpy_module.isfinite(f_norm1_np)
    fsqr1 = float(gcr2_p) * f_norm1_np if finite else 0.0
    fsqz1 = float(gcz2_p) * f_norm1_np if finite else 0.0
    if bool(vmec2000_control):
        frzl_for_gcl2_full = frzl_pre if frzl_pre_host is None else frzl_pre_host
        fsql1 = lambda_preconditioned_full_norm(frzl_for_gcl2_full, use_jax=False) * delta_s
    else:
        fsql1 = float(gcl2_p) * delta_s
    fsqr1_safe = finite_float_or_zero(fsqr1)
    fsqz1_safe = finite_float_or_zero(fsqz1)
    fsql1_safe = finite_float_or_zero(fsql1)
    fsq1 = fsqr1_safe + fsqz1_safe + fsql1_safe
    return PreconditionedResidualScalarChannels(
        rz_norm,
        f_norm1,
        fsqr1_safe,
        fsqz1_safe,
        fsql1_safe,
        fsqr1_safe,
        fsqz1_safe,
        fsql1_safe,
        fsq1,
    )


def jax_preconditioned_residual_scalar_channels(
    *,
    gcr2_p: Any,
    gcz2_p: Any,
    gcl2_p: Any,
    frzl_pre: Any,
    vmec2000_control: bool,
    vmec2000_cache_valid: bool,
    need_bcovar_update: bool,
    cache_rz_norm: Any,
    cache_f_norm1: Any,
    state: Any,
    delta_s: Any,
    jnp_module: Any,
    cached_or_current_f_norm1_jax: Callable[..., tuple[Any, Any]],
    rz_norm_func: Callable[[Any], Any],
    lambda_preconditioned_full_norm: Callable[..., Any],
) -> PreconditionedResidualScalarChannels:
    """Build preconditioned fsq1 channels on the JAX/device path."""

    rz_norm, f_norm1 = cached_or_current_f_norm1_jax(
        vmec2000_control=bool(vmec2000_control),
        vmec2000_cache_valid=bool(vmec2000_cache_valid),
        need_bcovar_update=bool(need_bcovar_update),
        cache_rz_norm=cache_rz_norm,
        cache_f_norm1=cache_f_norm1,
        state=state,
        rz_norm_func=rz_norm_func,
    )
    finite_fnorm1 = jnp_module.isfinite(f_norm1)
    fsqr1 = jnp_module.where(
        finite_fnorm1,
        gcr2_p * f_norm1,
        jnp_module.asarray(0.0, dtype=jnp_module.asarray(gcr2_p).dtype),
    )
    fsqz1 = jnp_module.where(
        finite_fnorm1,
        gcz2_p * f_norm1,
        jnp_module.asarray(0.0, dtype=jnp_module.asarray(gcz2_p).dtype),
    )
    if bool(vmec2000_control):
        fsql1 = lambda_preconditioned_full_norm(frzl_pre, use_jax=True) * delta_s
    else:
        fsql1 = gcl2_p * delta_s
    fsqr1_safe = jnp_module.where(
        jnp_module.isfinite(fsqr1),
        fsqr1,
        jnp_module.asarray(0.0, dtype=jnp_module.asarray(fsqr1).dtype),
    )
    fsqz1_safe = jnp_module.where(
        jnp_module.isfinite(fsqz1),
        fsqz1,
        jnp_module.asarray(0.0, dtype=jnp_module.asarray(fsqz1).dtype),
    )
    fsql1_safe = jnp_module.where(
        jnp_module.isfinite(fsql1),
        fsql1,
        jnp_module.asarray(0.0, dtype=jnp_module.asarray(fsql1).dtype),
    )
    return PreconditionedResidualScalarChannels(
        rz_norm,
        f_norm1,
        fsqr1,
        fsqz1,
        fsql1,
        fsqr1_safe,
        fsqz1_safe,
        fsql1_safe,
        fsqr1_safe + fsqz1_safe + fsql1_safe,
    )


def materialize_accepted_control_payload(
    *,
    accepted_control_ptau_payload: Any | None,
    use_control_payload: bool,
    fsq1_j: Any,
    k: Any,
    ptau_pshalf_jax: Any,
    ptau_ohs_jax: Any,
    timing_enabled: bool,
    timing_stats: dict[str, Any],
    perf_counter: Callable[[], float],
    jax_module: Any,
    device_get_floats: Callable[..., tuple[float, ...]],
    accepted_control_ptau_host_from_payload: Callable[..., tuple[float, tuple[float, float] | None, bool]],
    scan_math_kernel_arrays_from_k: Callable[[Any], tuple[Any, ...] | None],
    accepted_control_payload_jit: Callable[[], Callable[..., Any] | None],
) -> AcceptedControlPayloadMaterialization:
    """Materialize accepted-controller scalars with optional fused payloads."""

    accepted_control_ptau_host: tuple[float, float] | None = None
    control_payload_used = False
    if accepted_control_ptau_payload is not None:
        t_payload_start = perf_counter() if timing_enabled else None
        fsq1_payload_host, accepted_control_ptau_host, control_payload_used = (
            accepted_control_ptau_host_from_payload(
                accepted_control_ptau_payload,
                device_get_floats=device_get_floats,
            )
        )
        if control_payload_used:
            fsq1 = float(fsq1_payload_host)
        if timing_enabled and t_payload_start is not None:
            timing_stats["iteration_control_fsq1_payload_get"] += perf_counter() - float(t_payload_start)
    if (not control_payload_used) and use_control_payload:
        ptau_arrays = scan_math_kernel_arrays_from_k(k)
        payload_fn = accepted_control_payload_jit()
        if ptau_arrays is not None and payload_fn is not None:
            t_payload_start = perf_counter() if timing_enabled else None
            try:
                payload = payload_fn(
                    fsq1_j,
                    *ptau_arrays,
                    ptau_pshalf_jax,
                    ptau_ohs_jax,
                )
                fsq1_payload_host, accepted_control_ptau_host, control_payload_used = (
                    accepted_control_ptau_host_from_payload(
                        payload,
                        device_get_floats=device_get_floats,
                    )
                )
                if control_payload_used:
                    fsq1 = float(fsq1_payload_host)
            except Exception:
                control_payload_used = False
            finally:
                if timing_enabled and t_payload_start is not None:
                    timing_stats["iteration_control_fsq1_payload_get"] += perf_counter() - float(t_payload_start)
    if control_payload_used:
        return AcceptedControlPayloadMaterialization(
            fsq1,
            accepted_control_ptau_host,
            True,
        )

    t_direct_start = perf_counter() if timing_enabled else None
    fsq1 = float(jax_module.device_get(fsq1_j))
    if timing_enabled and t_direct_start is not None:
        timing_stats["iteration_control_fsq1_direct_get"] += perf_counter() - float(t_direct_start)
    return AcceptedControlPayloadMaterialization(
        fsq1,
        accepted_control_ptau_host,
        False,
    )


def seed_preconditioner_cache_from_bcovar_update(
    *,
    cache: Any,
    k: Any,
    state: Any,
    trig: Any,
    s: Any,
    cfg: Any,
    norms_used: Any,
    rz_scale: Any,
    l_scale: Any,
    constraint_tcon0: float | None,
    zero_tcon: Any,
    host_update_assembly: bool,
    timing_enabled: bool,
    timing_stats: dict[str, Any],
    perf_counter: Callable[[], float],
    tree_has_tracer: Callable[[Any], bool],
    rz_norm_np: Callable[[Any], float],
    rz_norm_func: Callable[[Any], Any],
    lambda_preconditioner_func: Callable[[Any], Any],
    rz_preconditioner_matrices_func: Callable[..., Any],
    precond_jmax_override: int | None,
    preconditioner_use_precomputed_tridi: bool | None,
    preconditioner_use_lax_tridi: bool | None,
    jnp_module: Any = jnp,
) -> PreconditionerBcovarSeedResult:
    """Seed VMEC2000 preconditioner cache entries after a bcovar refresh.

    VMEC updates the constraint diagonal, tcon profile, norms, and 1D
    preconditioner payloads together when the bcovar fields are refreshed. This
    helper keeps that mutable cache update in the preconditioner domain instead
    of spelling it out in the residual host loop.
    """

    if constraint_tcon0 is None or float(constraint_tcon0) == 0.0:
        cache.precond_diag = None
        cache.tcon = zero_tcon
    else:
        from vmec_jax.vmec_constraints import precondn_diag_axd1_from_bcovar

        use_numpy_patch = (
            bool(host_update_assembly)
            and (not tree_has_tracer(k))
            and (not tree_has_tracer(s))
        )
        if use_numpy_patch:
            from vmec_jax.vmec_numpy_forces import _numpy_module_patch

            with _numpy_module_patch():
                ard1, azd1 = precondn_diag_axd1_from_bcovar(
                    trig=trig,
                    s=s,
                    bsq=k.bc.bsq,
                    r12=k.bc.jac.r12,
                    sqrtg=k.bc.jac.sqrtg,
                    ru12=k.bc.jac.ru12,
                    zu12=k.bc.jac.zu12,
                )
        else:
            ard1, azd1 = precondn_diag_axd1_from_bcovar(
                trig=trig,
                s=s,
                bsq=k.bc.bsq,
                r12=k.bc.jac.r12,
                sqrtg=k.bc.jac.sqrtg,
                ru12=k.bc.jac.ru12,
                zu12=k.bc.jac.zu12,
            )
        cache.precond_diag = (ard1, azd1)
        cache.tcon = np.asarray(k.tcon) if host_update_assembly else jnp_module.asarray(k.tcon)

    cache.norms = norms_used
    cache.rz_scale = rz_scale
    cache.l_scale = l_scale
    if host_update_assembly:
        cache.rz_norm = rz_norm_np(state)
        cache.f_norm1 = (1.0 / cache.rz_norm) if cache.rz_norm != 0.0 else float("inf")
    else:
        cache.rz_norm = rz_norm_func(state)
        cache.f_norm1 = jnp_module.where(
            jnp_module.asarray(cache.rz_norm) != 0.0,
            1.0 / jnp_module.asarray(cache.rz_norm),
            jnp_module.asarray(float("inf"), dtype=jnp_module.asarray(cache.rz_norm).dtype),
        )

    cache_update_trace = False
    seeded_from_bcovar_update = False
    seed_time_in_residual_metrics = 0.0
    if not bool(cfg.lasym):
        t_precond_refresh_seed_start = perf_counter() if timing_enabled else None
        cache.prec_lam_prec = lambda_preconditioner_func(k.bc)
        cache.prec_faclam = None
        cache.prec_lam_debug = None
        mats, _jmin, jmax = rz_preconditioner_matrices_func(
            bc=k.bc,
            k=k,
            jmax_override=precond_jmax_override,
            use_precomputed=preconditioner_use_precomputed_tridi,
            use_lax_tridi=preconditioner_use_lax_tridi,
        )
        cache.prec_rz_mats = mats
        cache.prec_rz_jmax = None if tree_has_tracer(k) else int(jmax)
        seeded_from_bcovar_update = cache.prec_rz_jmax is not None
        cache_update_trace = True
        if timing_enabled and t_precond_refresh_seed_start is not None:
            seed_dt = perf_counter() - float(t_precond_refresh_seed_start)
            seed_time_in_residual_metrics += seed_dt
            timing_stats["precond_refresh_seed"] += seed_dt
            timing_stats["precond_refresh"] += seed_dt
            timing_stats["preconditioner"] += seed_dt
            timing_stats["precond_refresh_calls"] = int(timing_stats["precond_refresh_calls"]) + 1
    cache.valid = True

    return PreconditionerBcovarSeedResult(
        cache_update_trace=cache_update_trace,
        seeded_from_bcovar_update=seeded_from_bcovar_update,
        seed_time_in_residual_metrics=seed_time_in_residual_metrics,
    )


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


def _cached_or_current_f_norm1_jax(
    *,
    vmec2000_control: bool,
    vmec2000_cache_valid: bool,
    need_bcovar_update: bool,
    cache_rz_norm: Any,
    cache_f_norm1: Any,
    state: Any,
    rz_norm_func: Callable[[Any], Any],
):
    if (
        bool(vmec2000_control)
        and bool(vmec2000_cache_valid)
        and (not bool(need_bcovar_update))
        and (cache_rz_norm is not None)
        and (cache_f_norm1 is not None)
    ):
        return jnp.asarray(cache_rz_norm), jnp.asarray(cache_f_norm1)
    rz_norm = rz_norm_func(state)
    return rz_norm, jnp.where(rz_norm != 0.0, 1.0 / rz_norm, jnp.asarray(float("inf"), dtype=rz_norm.dtype))


def _force_blocks_from_sequence(blocks: tuple[Any, ...] | ForceBlocks) -> ForceBlocks:
    """Normalize payload tuples to the named force-block convention."""

    if isinstance(blocks, ForceBlocks):
        return blocks
    return ForceBlocks(*blocks)


def apply_vmec2000_preconditioner_runtime(
    *,
    frzl: TomnspsRZL,
    k: Any,
    state: Any,
    iter2: int,
    cfg: Any,
    s: Any,
    delta_s: Any,
    w_mode_mn: Any,
    lambda_update_scale: float,
    lambda_update_scale_j: Any,
    lconm1: bool,
    vmec2000_control: bool,
    vmec2000_cache_valid: bool,
    need_bcovar_update: bool,
    cache_rz_norm: Any,
    cache_f_norm1: Any,
    host_update_assembly: bool,
    use_fused_precond_output_scaling: bool,
    scale_m1_rhs: bool,
    adjoint_trace: bool,
    adjoint_trace_mode: str,
    accepted_control_ptau_arrays: tuple[Any, ...] | None,
    ptau_pshalf_jax: Any,
    ptau_ohs_jax: Any,
    preconditioner_use_precomputed_tridi: bool | None,
    preconditioner_use_lax_tridi: bool | None,
    timing_detail_enabled: bool,
    timing_stats: dict[str, Any],
    perf_counter: Callable[[], float],
    block_until_ready: Callable[[Any], Any] | None,
    refresh_preconditioner_cache_func: Callable[..., tuple[Any, dict[str, Any], int, bool, bool, bool]],
    scale_m1_precond_rhs_func: Callable[[TomnspsRZL, dict[str, Any]], TomnspsRZL],
    rz_preconditioner_apply_func: Callable[..., TomnspsRZL],
    rz_norm_func: Callable[[Any], Any],
) -> Vmec2000PreconditionerApplyResult:
    """Apply the VMEC2000-style R/Z/lambda preconditioner and payload scaling."""

    lam_prec, mats, jmax, need_lam_prec, need_lamcal, cache_update_trace = refresh_preconditioner_cache_func(
        k,
        iter2=int(iter2),
    )
    t_precond_apply_start = perf_counter() if timing_detail_enabled else None
    use_apply_payload_fusion = (
        bool(use_fused_precond_output_scaling)
        and need_lam_prec is False
        and need_lamcal is False
    )
    frzl_rhs = scale_m1_precond_rhs_func(frzl, mats) if bool(scale_m1_rhs) else frzl
    frzl_rz = None
    frzl_lam_pre = None
    update_blocks = None
    gcr2_p = gcz2_p = gcl2_p = None
    fsqr1_safe = fsqz1_safe = fsql1_safe = fsq1_safe = None
    accepted_control_ptau_payload = None
    blocks = None
    precond_kwargs = {
        "mats": mats,
        "jmax": jmax,
        "use_precomputed": preconditioner_use_precomputed_tridi,
        "use_lax_tridi": preconditioner_use_lax_tridi,
    }

    if use_apply_payload_fusion:
        _, f_norm1 = _cached_or_current_f_norm1_jax(
            vmec2000_control=bool(vmec2000_control),
            vmec2000_cache_valid=bool(vmec2000_cache_valid),
            need_bcovar_update=bool(need_bcovar_update),
            cache_rz_norm=cache_rz_norm,
            cache_f_norm1=cache_f_norm1,
            state=state,
            rz_norm_func=rz_norm_func,
        )
        precond_payload = _preconditioner_apply_payload_fused(
            frzl_in=frzl_rhs,
            mats=mats,
            jmax=jmax,
            cfg=cfg,
            lam_prec=lam_prec,
            w_mode_mn=w_mode_mn,
            lambda_update_scale_j=lambda_update_scale_j,
            f_norm1=f_norm1,
            delta_s=delta_s,
            s=s,
            use_precomputed=preconditioner_use_precomputed_tridi,
            use_lax_tridi=preconditioner_use_lax_tridi,
            apply_lambda_update_scale=(lambda_update_scale != 1.0),
            vmec2000_control=bool(vmec2000_control),
            lconm1=bool(lconm1),
            include_control_ptau=accepted_control_ptau_arrays is not None,
            control_ptau_arrays=accepted_control_ptau_arrays,
            control_ptau_pshalf=ptau_pshalf_jax,
            control_ptau_ohs=ptau_ohs_jax,
        )
        pre_blocks, update_blocks_raw, diag, accepted_control_ptau_payload = _split_preconditioner_apply_payload(
            precond_payload
        )
        blocks = _force_blocks_from_sequence(pre_blocks)
        update_blocks = _force_blocks_from_sequence(update_blocks_raw)
        (gcr2_p, gcz2_p, gcl2_p, fsqr1_safe, fsqz1_safe, fsql1_safe, fsq1_safe) = diag
    else:
        frzl_rz = rz_preconditioner_apply_func(frzl_in=frzl_rhs, **precond_kwargs)
        frzl_lam_pre = frzl_rz

    if use_apply_payload_fusion and adjoint_trace and adjoint_trace_mode == "full":
        # Fused payloads avoid materializing the raw R/Z-preconditioned force.
        # Full accepted-trace replay needs it, so only the opt-in replay path
        # pays this extra apply.
        frzl_rz = rz_preconditioner_apply_func(frzl_in=frzl_rhs, **precond_kwargs)

    if (not use_apply_payload_fusion) and bool(host_update_assembly):
        blocks = preconditioner_output_blocks_np(frzl_rz=frzl_rz, lam_prec=lam_prec)
    elif (not use_apply_payload_fusion) and bool(use_fused_precond_output_scaling):
        _, f_norm1 = _cached_or_current_f_norm1_jax(
            vmec2000_control=bool(vmec2000_control),
            vmec2000_cache_valid=bool(vmec2000_cache_valid),
            need_bcovar_update=bool(need_bcovar_update),
            cache_rz_norm=cache_rz_norm,
            cache_f_norm1=cache_f_norm1,
            state=state,
            rz_norm_func=rz_norm_func,
        )
        payload_outputs = _preconditioner_output_payload_jit(
            apply_lambda_update_scale=(lambda_update_scale != 1.0),
            vmec2000_control=bool(vmec2000_control),
            lconm1=bool(lconm1),
        )
        pre_blocks, update_blocks_raw, diag = payload_outputs(
            frzl_rz,
            lam_prec,
            w_mode_mn,
            lambda_update_scale_j,
            f_norm1,
            delta_s,
            s,
        )
        blocks = _force_blocks_from_sequence(pre_blocks)
        update_blocks = _force_blocks_from_sequence(update_blocks_raw)
        (gcr2_p, gcz2_p, gcl2_p, fsqr1_safe, fsqz1_safe, fsql1_safe, fsq1_safe) = diag
    elif not use_apply_payload_fusion:
        blocks = preconditioner_output_blocks_jax(frzl_rz=frzl_rz, lam_prec=lam_prec)

    if timing_detail_enabled and t_precond_apply_start is not None:
        try:
            if block_until_ready is not None and blocks is not None:
                block_until_ready(blocks.flsc)
        except Exception:
            pass
        timing_stats["precond_apply"] += perf_counter() - float(t_precond_apply_start)

    outputs_scaled = bool(use_apply_payload_fusion or update_blocks is not None)
    return Vmec2000PreconditionerApplyResult(
        lam_prec=lam_prec,
        mats=mats,
        jmax=jmax,
        cache_update_trace=bool(cache_update_trace),
        blocks=blocks,
        update_blocks=update_blocks,
        gcr2_p=gcr2_p,
        gcz2_p=gcz2_p,
        gcl2_p=gcl2_p,
        fsqr1_safe=fsqr1_safe,
        fsqz1_safe=fsqz1_safe,
        fsql1_safe=fsql1_safe,
        fsq1_safe=fsq1_safe,
        frzl_rz=frzl_rz,
        frzl_lam_pre=frzl_lam_pre,
        outputs_scaled=outputs_scaled,
        fsq1_ready=outputs_scaled,
        accepted_control_ptau_payload=accepted_control_ptau_payload,
    )


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


def refresh_preconditioner_cache_state_runtime(
    k: Any,
    *,
    cache: Any,
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
    need_bcovar_update: bool,
    precond_cache_seeded_from_bcovar_update: bool,
    precond_expected_jmax: int,
    precond_jmax_override: int | None,
    preconditioner_use_precomputed_tridi: bool | None,
    preconditioner_use_lax_tridi: bool | None,
) -> tuple[Any, dict[str, Any], int, bool, bool, bool]:
    """Refresh/reuse cache and write the mutable cache fields in one domain seam."""

    refresh = refresh_preconditioner_cache_runtime(
        k=k,
        cfg=cfg,
        static=static,
        iter2=int(iter2),
        env_dump_lam=env_dump_lam,
        env_dump_lamcal=env_dump_lamcal,
        timing_enabled=bool(timing_enabled),
        timing_stats=timing_stats,
        perf_counter=perf_counter,
        block_until_ready=block_until_ready,
        tree_has_tracer=tree_has_tracer,
        update_preconditioner_cache_func=update_preconditioner_cache_func,
        can_reassemble_func=can_reassemble_func,
        lambda_preconditioner_func=lambda_preconditioner_func,
        rz_preconditioner_matrices_func=rz_preconditioner_matrices_func,
        maybe_dump_lam_prec=maybe_dump_lam_prec,
        maybe_dump_precond_mats=maybe_dump_precond_mats,
        maybe_dump_lamcal=maybe_dump_lamcal,
        vmec2000_cache_valid=bool(cache.valid),
        need_bcovar_update=bool(need_bcovar_update),
        precond_cache_seeded_from_bcovar_update=bool(precond_cache_seeded_from_bcovar_update),
        precond_expected_jmax=int(precond_expected_jmax),
        precond_jmax_override=precond_jmax_override,
        preconditioner_use_precomputed_tridi=preconditioner_use_precomputed_tridi,
        preconditioner_use_lax_tridi=preconditioner_use_lax_tridi,
        cache_prec_lam_prec=cache.prec_lam_prec,
        cache_prec_faclam=cache.prec_faclam,
        cache_prec_lam_debug=cache.prec_lam_debug,
        cache_prec_rz_mats=cache.prec_rz_mats,
        cache_prec_rz_jmax=cache.prec_rz_jmax,
    )
    cache.prec_lam_prec = refresh.cache_prec_lam_prec
    cache.prec_faclam = refresh.cache_prec_faclam
    cache.prec_lam_debug = refresh.cache_prec_lam_debug
    cache.prec_rz_mats = refresh.cache_prec_rz_mats
    cache.prec_rz_jmax = refresh.cache_prec_rz_jmax
    return (
        refresh.lam_prec,
        refresh.mats,
        refresh.jmax,
        refresh.need_lam_prec,
        refresh.need_lamcal,
        refresh.cache_update_trace,
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
    "apply_vmec2000_preconditioner_runtime",
    "seed_preconditioner_cache_from_bcovar_update",
    "PreconditionerBcovarSeedResult",
    "PreconditionerRefreshRuntimeResult",
    "Vmec2000PreconditionerApplyResult",
    "refresh_preconditioner_cache_runtime",
    "refresh_preconditioner_cache_state_runtime",
]
