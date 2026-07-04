"""Pure helpers for VMEC2000 scan force/restart payloads."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Callable, NamedTuple

from ...._compat import jax, jnp
from ....kernels.residue import vmec_gcx2_from_tomnsps
from ....kernels.tomnsp import TomnspsRZL


class ScanForceBlocks(NamedTuple):
    """VMEC force channels carried by one scan step."""

    frcc: Any
    frss: Any
    fzsc: Any
    fzcs: Any
    flsc: Any
    flcs: Any
    frsc: Any
    frcs: Any
    fzcc: Any
    fzss: Any
    flcc: Any
    flss: Any


class ScanForcePayload(NamedTuple):
    """Force blocks, residual scalars, and preconditioner cache fields."""

    blocks: ScanForceBlocks
    fsqr: Any
    fsqz: Any
    fsql: Any
    fsqr1: Any
    fsqz1: Any
    fsql1: Any
    cache_precond_diag: Any
    cache_tcon: Any
    cache_norms: Any
    cache_rz_scale: Any
    cache_l_scale: Any
    cache_rz_norm: Any
    cache_f_norm1: Any
    cache_rz_mats: Any
    cache_lam_prec: Any
    cache_valid: Any


class ScanInitialCache(NamedTuple):
    """Initial VMEC2000 scan preconditioner cache fields."""

    precond_diag: Any
    tcon: Any
    norms: Any
    rz_scale: Any
    l_scale: Any
    rz_norm: Any
    f_norm1: Any
    lam_prec: Any
    rz_mats: Any
    jmax: int
    valid: Any


class ScanStepForceEvaluation(NamedTuple):
    """First force/residual payload for one VMEC2000 scan step."""

    iter2: Any
    fsq_prev_before: Any
    fsq0_prev_before: Any
    skip_timecontrol: Any
    time_step_report: Any
    zero_m1: Any
    include_edge: Any
    need_bcovar_update: Any
    use_cached_precond: Any
    kernels: Any
    frzl: Any
    gcr2: Any
    gcz2: Any
    gcl2: Any
    rz_scale: Any
    l_scale: Any
    norms_current: Any
    norms_used: Any
    fsqr: Any
    fsqz: Any
    fsql: Any
    conv_now: Any


class ScanStepFields(NamedTuple):
    """State, velocity memory, and residual controller scalars for a scan step."""

    state: Any
    vRcc: Any
    vRss: Any
    vZsc: Any
    vZcs: Any
    vLsc: Any
    vLcs: Any
    vRsc: Any
    vRcs: Any
    vZcc: Any
    vZss: Any
    vLcc: Any
    vLss: Any
    inv_tau: Any
    fsq_prev: Any


class ScanSelectedPayloadStep(NamedTuple):
    """Selected scan payload and state-update fields for one scan step."""

    payload: ScanForcePayload
    step_fields: ScanStepFields
    fsqr: Any
    fsqz: Any
    fsql: Any
    fsq1: Any


_SCAN_RZ_APPLY_MAT_KEYS = (
    "ar",
    "br",
    "dr",
    "az",
    "bz",
    "dz",
    "cr",
    "ir",
    "cz",
    "iz",
    "dlr_t",
    "dr_t",
    "dur_t",
    "dlz_t",
    "dz_t",
    "duz_t",
)


def compact_scan_rz_mats_for_carry(mats: Any) -> Any:
    """Keep only R/Z preconditioner data needed inside a fixed-grid scan carry.

    The full matrix dictionary also contains parity/reassembly coefficients.
    Those are useful to host-side resume and non-scan preconditioner paths, but
    fixed-grid scan iterations only apply the current tridiagonal matrices and
    VMEC's m=1 scale factors.  Storing the derived m=1 factors explicitly lets
    the scan carry drop the parity coefficients without changing residual
    algebra.
    """

    if not isinstance(mats, dict):
        return mats
    compact = {key: mats[key] for key in _SCAN_RZ_APPLY_MAT_KEYS if key in mats}
    if "m1_fac_r" in mats and "m1_fac_z" in mats:
        compact["m1_fac_r"] = mats["m1_fac_r"]
        compact["m1_fac_z"] = mats["m1_fac_z"]
        return compact

    from vmec_jax.solvers.fixed_boundary.preconditioning.operators import vmec_scale_m1_factors_from_mats

    fac_r, fac_z = vmec_scale_m1_factors_from_mats(mats)
    compact["m1_fac_r"] = fac_r
    compact["m1_fac_z"] = fac_z
    return compact


def build_initial_preconditioner_cache(
    *,
    state_init: Any,
    k: Any,
    norms: Any,
    rz_scale: Any,
    l_scale: Any,
    constraint_tcon0: Any,
    zero_precond_diag: Any,
    zero_tcon: Any,
    trig: Any,
    s: Any,
    cfg: Any,
    dtype: Any,
    scan_use_precomputed: bool,
    scan_use_lax_tridi: bool,
    lambda_preconditioner_func: Callable[[Any], Any],
    rz_norm_func: Callable[[Any], Any],
    resume_state: dict[str, Any] | None = None,
) -> ScanInitialCache:
    """Build the initial scan cache, then overlay any resume-cache payload."""

    if constraint_tcon0 is None or float(constraint_tcon0) == 0.0:
        cache_precond_diag = zero_precond_diag
        cache_tcon = zero_tcon
    else:
        from vmec_jax.kernels.constraints import precondn_diag_axd1_from_bcovar

        ard1, azd1 = precondn_diag_axd1_from_bcovar(
            trig=trig,
            s=s,
            bsq=k.bc.bsq,
            r12=k.bc.jac.r12,
            sqrtg=k.bc.jac.sqrtg,
            ru12=k.bc.jac.ru12,
            zu12=k.bc.jac.zu12,
        )
        cache_precond_diag = (ard1, azd1)
        cache_tcon = jnp.asarray(k.tcon)

    cache_norms = norms
    cache_rz_scale = rz_scale
    cache_l_scale = l_scale
    cache_rz_norm = rz_norm_func(state_init)
    cache_f_norm1 = jnp.where(
        cache_rz_norm != 0.0,
        1.0 / cache_rz_norm,
        jnp.asarray(float("inf"), dtype=dtype),
    )

    from vmec_jax.preconditioner_1d_jax import rz_preconditioner_matrices

    cache_lam_prec = lambda_preconditioner_func(k.bc)
    cache_rz_mats, _jmin, _jmax = rz_preconditioner_matrices(
        bc=k.bc,
        k=k,
        trig=trig,
        s=s,
        cfg=cfg,
        use_precomputed=bool(scan_use_precomputed),
        use_lax_tridi=bool(scan_use_lax_tridi),
    )
    # Fixed-grid scans use a shape-derived jmax; this avoids carrying a JIT
    # tracer from the matrix helper into host-side setup.
    jmax = max(int(jnp.asarray(s).shape[0]) - 1, 1)
    cache_valid = jnp.asarray(True)

    if resume_state is not None:
        try:
            cache_valid = jnp.asarray(
                bool(resume_state.get("vmec2000_cache_valid", bool(cache_valid))), dtype=bool
            )
        except Exception:
            cache_valid = jnp.asarray(cache_valid, dtype=bool)
        if "cache_precond_diag" in resume_state:
            cache_precond_diag = resume_state.get("cache_precond_diag", cache_precond_diag)
        if "cache_tcon" in resume_state:
            cache_tcon = resume_state.get("cache_tcon", cache_tcon)
        if "cache_norms" in resume_state:
            cache_norms = resume_state.get("cache_norms", cache_norms)
        if "cache_rz_scale" in resume_state:
            cache_rz_scale = resume_state.get("cache_rz_scale", cache_rz_scale)
        if "cache_l_scale" in resume_state:
            cache_l_scale = resume_state.get("cache_l_scale", cache_l_scale)
        if "cache_rz_norm" in resume_state:
            try:
                cache_rz_norm = jnp.asarray(resume_state.get("cache_rz_norm", cache_rz_norm), dtype=dtype)
            except Exception:
                pass
        if "cache_f_norm1" in resume_state:
            try:
                cache_f_norm1 = jnp.asarray(resume_state.get("cache_f_norm1", cache_f_norm1), dtype=dtype)
            except Exception:
                pass
        if "cache_prec_rz_mats" in resume_state:
            cache_rz_mats = resume_state.get("cache_prec_rz_mats", cache_rz_mats)
        if "cache_prec_lam_prec" in resume_state:
            cache_lam_prec = resume_state.get("cache_prec_lam_prec", cache_lam_prec)
    cache_rz_mats = compact_scan_rz_mats_for_carry(cache_rz_mats)

    return ScanInitialCache(
        precond_diag=cache_precond_diag,
        tcon=cache_tcon,
        norms=cache_norms,
        rz_scale=cache_rz_scale,
        l_scale=cache_l_scale,
        rz_norm=cache_rz_norm,
        f_norm1=cache_f_norm1,
        lam_prec=cache_lam_prec,
        rz_mats=cache_rz_mats,
        jmax=jmax,
        valid=cache_valid,
    )


def mask_scan_restart_force_payload(
    *, force_blocks: tuple[Any, ...], cache_valid: Any, do_restart: Any
) -> tuple[tuple[Any, ...], Any]:
    """Zero current-state scan forces on restart when checkpoint forces are skipped."""

    no_restart = jnp.logical_not(do_restart)
    masked_blocks = tuple(jnp.where(no_restart, block, jnp.zeros_like(block)) for block in force_blocks)
    cache_valid_masked = jnp.where(no_restart, cache_valid, jnp.asarray(False))
    return masked_blocks, cache_valid_masked


def _preconditioned_blocks(*, frzl_rz: TomnspsRZL, cache_lam_prec: Any) -> tuple[ScanForceBlocks, TomnspsRZL]:
    lam_prec = jnp.asarray(cache_lam_prec)
    frcc = jnp.asarray(frzl_rz.frcc)
    frss = frzl_rz.frss if frzl_rz.frss is not None else jnp.zeros_like(frcc)
    fzsc = jnp.asarray(frzl_rz.fzsc)
    fzcs = frzl_rz.fzcs if frzl_rz.fzcs is not None else jnp.zeros_like(fzsc)
    flsc = jnp.asarray(frzl_rz.flsc) * lam_prec
    flcs = None if frzl_rz.flcs is None else (jnp.asarray(frzl_rz.flcs) * lam_prec)
    frsc = jnp.zeros_like(frcc)
    frcs = jnp.zeros_like(frcc)
    fzcc = jnp.zeros_like(fzsc)
    fzss = jnp.zeros_like(fzsc)
    flcc = jnp.zeros_like(flsc)
    flss = jnp.zeros_like(flsc)

    if getattr(frzl_rz, "frsc", None) is not None:
        frsc = jnp.asarray(frzl_rz.frsc)
    if getattr(frzl_rz, "frcs", None) is not None:
        frcs = jnp.asarray(frzl_rz.frcs)
    if getattr(frzl_rz, "fzcc", None) is not None:
        fzcc = jnp.asarray(frzl_rz.fzcc)
    if getattr(frzl_rz, "fzss", None) is not None:
        fzss = jnp.asarray(frzl_rz.fzss)
    if getattr(frzl_rz, "flcc", None) is not None:
        flcc = jnp.asarray(frzl_rz.flcc) * lam_prec
    if getattr(frzl_rz, "flss", None) is not None:
        flss = jnp.asarray(frzl_rz.flss) * lam_prec

    blocks = ScanForceBlocks(
        frcc=frcc,
        frss=frss,
        fzsc=fzsc,
        fzcs=fzcs,
        flsc=flsc,
        flcs=flcs if flcs is not None else jnp.zeros_like(flsc),
        frsc=frsc,
        frcs=frcs,
        fzcc=fzcc,
        fzss=fzss,
        flcc=flcc,
        flss=flss,
    )
    frzl_pre = TomnspsRZL(
        frcc=blocks.frcc,
        frss=blocks.frss,
        fzsc=blocks.fzsc,
        fzcs=blocks.fzcs,
        flsc=blocks.flsc,
        flcs=flcs,
        frsc=blocks.frsc,
        frcs=blocks.frcs,
        fzcc=blocks.fzcc,
        fzss=blocks.fzss,
        flcc=blocks.flcc,
        flss=blocks.flss,
    )
    return blocks, frzl_pre


def _lambda_fsq1_from_blocks(
    *, frzl_pre: TomnspsRZL, delta_s: Any, optional_source: TomnspsRZL | None = None
) -> Any:
    gcl2_full = jnp.sum(jnp.asarray(frzl_pre.flsc)[1:] * jnp.asarray(frzl_pre.flsc)[1:])
    if frzl_pre.flcs is not None:
        flcs = jnp.asarray(frzl_pre.flcs)
        gcl2_full = gcl2_full + jnp.sum(flcs[1:] * flcs[1:])
    optional_blocks = optional_source if optional_source is not None else frzl_pre
    if getattr(optional_blocks, "flcc", None) is not None:
        flcc = jnp.asarray(optional_blocks.flcc)
        gcl2_full = gcl2_full + jnp.sum(flcc[1:] * flcc[1:])
    if getattr(optional_blocks, "flss", None) is not None:
        flss = jnp.asarray(optional_blocks.flss)
        gcl2_full = gcl2_full + jnp.sum(flss[1:] * flss[1:])
    return gcl2_full * delta_s


def _weighted_blocks(
    *,
    blocks: ScanForceBlocks,
    w_mode_mn: Any,
    lambda_update_scale_j: Any,
    apply_lambda_update_scale: bool,
) -> ScanForceBlocks:
    weights = jnp.asarray(w_mode_mn)[None, :, :]
    flsc = blocks.flsc * weights
    flcs = blocks.flcs * weights
    flcc = blocks.flcc * weights
    flss = blocks.flss * weights
    if bool(apply_lambda_update_scale):
        scale = jnp.asarray(lambda_update_scale_j)
        flsc = flsc * scale
        flcs = flcs * scale
        flcc = flcc * scale
        flss = flss * scale

    return ScanForceBlocks(
        frcc=blocks.frcc * weights,
        frss=blocks.frss * weights,
        fzsc=blocks.fzsc * weights,
        fzcs=blocks.fzcs * weights,
        flsc=flsc,
        flcs=flcs,
        frsc=blocks.frsc * weights,
        frcs=blocks.frcs * weights,
        fzcc=blocks.fzcc * weights,
        fzss=blocks.fzss * weights,
        flcc=flcc,
        flss=flss,
    )


def build_scan_force_payload(
    *,
    frzl_rz: TomnspsRZL,
    cache_lam_prec: Any,
    w_mode_mn: Any,
    lambda_update_scale_j: Any,
    apply_lambda_update_scale: bool,
    fsqr: Any,
    fsqz: Any,
    fsql: Any,
    f_norm1: Any,
    delta_s: Any,
    s: Any,
    lconm1: bool,
    cache_precond_diag: Any,
    cache_tcon: Any,
    cache_norms: Any,
    cache_rz_scale: Any,
    cache_l_scale: Any,
    cache_rz_norm: Any,
    cache_f_norm1: Any,
    cache_rz_mats: Any,
    cache_valid: Any,
    lambda_fsq1_optional_source: TomnspsRZL | None = None,
) -> ScanForcePayload:
    """Build the scan payload from preconditioned force blocks and cache fields."""

    pre_blocks, frzl_pre = _preconditioned_blocks(frzl_rz=frzl_rz, cache_lam_prec=cache_lam_prec)
    gcr2_p, gcz2_p, _gcl2_p = vmec_gcx2_from_tomnsps(
        frzl=frzl_pre,
        lconm1=bool(lconm1),
        apply_m1_constraints=False,
        include_edge=True,
        apply_scalxc=False,
        s=s,
    )
    fsqr1 = gcr2_p * f_norm1
    fsqz1 = gcz2_p * f_norm1
    fsql1 = _lambda_fsq1_from_blocks(
        frzl_pre=frzl_pre,
        delta_s=delta_s,
        optional_source=lambda_fsq1_optional_source,
    )
    weighted = _weighted_blocks(
        blocks=pre_blocks,
        w_mode_mn=w_mode_mn,
        lambda_update_scale_j=lambda_update_scale_j,
        apply_lambda_update_scale=apply_lambda_update_scale,
    )

    return ScanForcePayload(
        blocks=weighted,
        fsqr=fsqr,
        fsqz=fsqz,
        fsql=fsql,
        fsqr1=fsqr1,
        fsqz1=fsqz1,
        fsql1=fsql1,
        cache_precond_diag=cache_precond_diag,
        cache_tcon=cache_tcon,
        cache_norms=cache_norms,
        cache_rz_scale=cache_rz_scale,
        cache_l_scale=cache_l_scale,
        cache_rz_norm=cache_rz_norm,
        cache_f_norm1=cache_f_norm1,
        cache_rz_mats=cache_rz_mats,
        cache_lam_prec=cache_lam_prec,
        cache_valid=cache_valid,
    )


def evaluate_scan_step_force(
    *,
    carry_adv: Any,
    it: Any,
    dtype: Any,
    k_preconditioner_update_interval: int,
    zero_precond_diag: Any,
    zero_tcon: Any,
    compute_forces_scan: Callable[..., Any],
    scan_converged: Callable[..., Any],
    tree_select: Callable[[Any, Any, Any], Any],
    cond: Callable[..., Any],
    convergence_controls: Any | None = None,
    trace_context: Callable[[], Any] | None = None,
    scan_debug_force_enabled: bool = False,
    scan_debug_iter: int = -1,
    debug_force_first_iter: Callable[..., Any] | None = None,
    debug_state_iter: Callable[..., Any] | None = None,
    debug_print: Callable[..., Any] | None = None,
) -> ScanStepForceEvaluation:
    """Evaluate the force and scalar residuals at the start of a scan step."""

    iter2 = jnp.asarray(it + 1, dtype=jnp.int32) + jnp.asarray(carry_adv.iter_offset, dtype=jnp.int32)
    fsq_prev_before = carry_adv.fsq_prev
    fsq0_prev_before = carry_adv.fsq0_prev
    skip_timecontrol = carry_adv.skip_timecontrol
    iter_since_restart = iter2 - carry_adv.iter1
    time_step_report = carry_adv.time_step
    # VMEC `constrain_m1`: zero gcz(m=1) on the first global iteration, and
    # again when the previous fsqz drops below the tolerance.
    zero_m1 = jnp.where(
        (iter2 < 2) | (carry_adv.fsqz_prev < 1.0e-6),
        jnp.asarray(1.0, dtype=dtype),
        jnp.asarray(0.0, dtype=dtype),
    )
    prev_rz_fsq = carry_adv.fsqr_prev_phys + carry_adv.fsqz_prev_phys
    include_edge = (iter_since_restart < 50) & (prev_rz_fsq < jnp.asarray(1.0e-6, dtype=prev_rz_fsq.dtype))

    precond_age = iter2 - carry_adv.iter1
    need_periodic_precond_update = (precond_age > 0) & (
        (precond_age % int(k_preconditioner_update_interval)) == 0
    )
    need_bcovar_update = (~carry_adv.cache_valid) | carry_adv.force_bcovar_update | need_periodic_precond_update
    use_cached_precond = carry_adv.cache_valid & (~need_bcovar_update)
    constraint_precond_diag = tree_select(use_cached_precond, carry_adv.cache_precond_diag, zero_precond_diag)
    constraint_tcon_override = jnp.where(use_cached_precond, carry_adv.cache_tcon, zero_tcon)

    context = nullcontext if trace_context is None else trace_context
    with context():
        kernels, frzl, gcr2, gcz2, gcl2, rz_scale, l_scale, norms_current = compute_forces_scan(
            carry_adv.state,
            include_edge=False,
            zero_m1=zero_m1,
            constraint_precond_diag=constraint_precond_diag,
            constraint_tcon=constraint_tcon_override,
            constraint_precond_active=use_cached_precond,
            constraint_tcon_active=use_cached_precond,
            iter_idx=None,
        )
    norms_used = cond(
        use_cached_precond,
        lambda _: carry_adv.cache_norms,
        lambda _: norms_current,
        operand=None,
    )
    fsqr = norms_used.r1 * norms_used.fnorm * gcr2
    fsqz = norms_used.r1 * norms_used.fnorm * gcz2
    fsql = norms_used.fnormL * gcl2
    if debug_force_first_iter is not None:
        debug_force_first_iter(
            enabled=bool(scan_debug_force_enabled) and debug_print is not None,
            iter2=iter2,
            frzl=frzl,
            carry_state=carry_adv.state,
            use_cached_precond=use_cached_precond,
            need_bcovar_update=need_bcovar_update,
            norms_used=norms_used,
            gcr2=gcr2,
            gcz2=gcz2,
            fsqr=fsqr,
            fsqz=fsqz,
            jnp_module=jnp,
            cond=cond,
            debug_print=debug_print,
        )
    if debug_state_iter is not None:
        debug_state_iter(
            scan_debug_iter=int(scan_debug_iter),
            iter2=iter2,
            carry_adv=carry_adv,
            use_cached_precond=use_cached_precond,
            need_bcovar_update=need_bcovar_update,
            norms_used=norms_used,
            gcr2=gcr2,
            gcz2=gcz2,
            gcl2=gcl2,
            jnp_module=jnp,
            cond=cond,
        )
    if convergence_controls is None:
        conv_now = scan_converged(fsqr, fsqz, fsql)
    else:
        conv_now = scan_converged(fsqr, fsqz, fsql, convergence_controls)
    return ScanStepForceEvaluation(
        iter2=iter2,
        fsq_prev_before=fsq_prev_before,
        fsq0_prev_before=fsq0_prev_before,
        skip_timecontrol=skip_timecontrol,
        time_step_report=time_step_report,
        zero_m1=zero_m1,
        include_edge=include_edge,
        need_bcovar_update=need_bcovar_update,
        use_cached_precond=use_cached_precond,
        kernels=kernels,
        frzl=frzl,
        gcr2=gcr2,
        gcz2=gcz2,
        gcl2=gcl2,
        rz_scale=rz_scale,
        l_scale=l_scale,
        norms_current=norms_current,
        norms_used=norms_used,
        fsqr=fsqr,
        fsqz=fsqz,
        fsql=fsql,
        conv_now=conv_now,
    )


def build_current_preconditioned_scan_payload(
    *,
    need_bcovar_update: Any,
    carry_adv: Any,
    k: Any,
    frzl: TomnspsRZL,
    norms_used: Any,
    rz_scale: Any,
    l_scale: Any,
    constraint_tcon0: Any,
    zero_precond_diag: Any,
    zero_tcon: Any,
    trig: Any,
    s: Any,
    cfg: Any,
    dtype: Any,
    scan_use_precomputed: bool,
    scan_use_lax_tridi: bool,
    lambda_preconditioner_func: Callable[[Any], Any],
    rz_norm_func: Callable[[Any], Any],
    scale_m1_precond_rhs_func: Callable[[TomnspsRZL, Any], TomnspsRZL],
    w_mode_mn: Any,
    lambda_update_scale_j: Any,
    apply_lambda_update_scale: bool,
    fsqr: Any,
    fsqz: Any,
    fsql: Any,
    delta_s: Any,
    jmax0: Any,
    cond: Callable[..., Any],
) -> ScanForcePayload:
    """Build the current scan force payload and its refreshed cache fields."""

    def _refresh_cache(_):
        if constraint_tcon0 is None or float(constraint_tcon0) == 0.0:
            cache_precond_diag = zero_precond_diag
            cache_tcon = zero_tcon
        else:
            from vmec_jax.kernels.constraints import precondn_diag_axd1_from_bcovar

            ard1, azd1 = precondn_diag_axd1_from_bcovar(
                trig=trig,
                s=s,
                bsq=k.bc.bsq,
                r12=k.bc.jac.r12,
                sqrtg=k.bc.jac.sqrtg,
                ru12=k.bc.jac.ru12,
                zu12=k.bc.jac.zu12,
            )
            cache_precond_diag = (ard1, azd1)
            cache_tcon = jnp.asarray(k.tcon)
        cache_norms = norms_used
        cache_rz_scale = rz_scale
        cache_l_scale = l_scale
        cache_rz_norm = rz_norm_func(carry_adv.state)
        cache_f_norm1 = jnp.where(
            cache_rz_norm != 0.0,
            1.0 / cache_rz_norm,
            jnp.asarray(float("inf"), dtype=dtype),
        )
        from vmec_jax.preconditioner_1d_jax import rz_preconditioner_matrices

        cache_lam_prec = lambda_preconditioner_func(k.bc)
        mats, _jmin, _jmax = rz_preconditioner_matrices(
            bc=k.bc,
            k=k,
            trig=trig,
            s=s,
            cfg=cfg,
            use_precomputed=bool(scan_use_precomputed),
            use_lax_tridi=bool(scan_use_lax_tridi),
        )
        mats = compact_scan_rz_mats_for_carry(mats)
        return (
            cache_precond_diag,
            cache_tcon,
            cache_norms,
            cache_rz_scale,
            cache_l_scale,
            cache_rz_norm,
            cache_f_norm1,
            cache_lam_prec,
            mats,
            jnp.asarray(True),
        )

    def _keep_cache(_):
        return (
            carry_adv.cache_precond_diag,
            carry_adv.cache_tcon,
            carry_adv.cache_norms,
            carry_adv.cache_rz_scale,
            carry_adv.cache_l_scale,
            carry_adv.cache_rz_norm,
            carry_adv.cache_f_norm1,
            carry_adv.cache_prec_lam_prec,
            carry_adv.cache_prec_rz_mats,
            carry_adv.cache_valid,
        )

    (
        cache_precond_diag,
        cache_tcon,
        cache_norms,
        cache_rz_scale,
        cache_l_scale,
        cache_rz_norm,
        cache_f_norm1,
        cache_lam_prec,
        cache_rz_mats,
        cache_valid,
    ) = cond(need_bcovar_update, _refresh_cache, _keep_cache, operand=None)

    frzl_rhs = scale_m1_precond_rhs_func(frzl, cache_rz_mats)
    from vmec_jax.preconditioner_1d_jax import rz_preconditioner_apply

    frzl_rz = rz_preconditioner_apply(
        frzl_in=frzl_rhs,
        mats=cache_rz_mats,
        jmax=jmax0,
        cfg=cfg,
        use_precomputed=bool(scan_use_precomputed),
        use_lax_tridi=bool(scan_use_lax_tridi),
    )
    rz_norm = jnp.where(cache_valid, cache_rz_norm, rz_norm_func(carry_adv.state))
    f_norm1 = jnp.where(
        cache_valid,
        cache_f_norm1,
        jnp.where(rz_norm != 0.0, 1.0 / rz_norm, jnp.asarray(float("inf"), dtype=dtype)),
    )
    return current_scan_payload(
        frzl_rz=frzl_rz,
        cache_lam_prec=cache_lam_prec,
        w_mode_mn=w_mode_mn,
        lambda_update_scale_j=lambda_update_scale_j,
        apply_lambda_update_scale=bool(apply_lambda_update_scale),
        fsqr=fsqr,
        fsqz=fsqz,
        fsql=fsql,
        f_norm1=f_norm1,
        delta_s=delta_s,
        s=s,
        lconm1=bool(getattr(cfg, "lconm1", True)),
        cache_precond_diag=cache_precond_diag,
        cache_tcon=cache_tcon,
        cache_norms=cache_norms,
        cache_rz_scale=cache_rz_scale,
        cache_l_scale=cache_l_scale,
        cache_rz_norm=cache_rz_norm,
        cache_f_norm1=cache_f_norm1,
        cache_rz_mats=cache_rz_mats,
        cache_valid=cache_valid,
        lambda_fsq1_optional_source=frzl,
    )


def build_restart_preconditioned_scan_payload(
    *,
    state_post: Any,
    compute_forces_scan_func: Callable[..., tuple[Any, TomnspsRZL, Any, Any, Any, Any, Any, Any]],
    trace_context: Callable[[], Any],
    zero_m1: Any,
    zero_precond_diag: Any,
    zero_tcon: Any,
    constraint_active_false: Any,
    constraint_tcon0: Any,
    trig: Any,
    s: Any,
    cfg: Any,
    dtype: Any,
    scan_use_precomputed: bool,
    scan_use_lax_tridi: bool,
    lambda_preconditioner_func: Callable[[Any], Any],
    rz_norm_func: Callable[[Any], Any],
    scale_m1_precond_rhs_func: Callable[[TomnspsRZL, Any], TomnspsRZL],
    w_mode_mn: Any,
    lambda_update_scale_j: Any,
    apply_lambda_update_scale: bool,
    delta_s: Any,
    jmax0: Any,
) -> ScanForcePayload:
    """Recompute restart forces and return a fresh restart scan payload."""

    with trace_context():
        k_r, frzl_r, gcr2_r, gcz2_r, gcl2_r, rz_scale_r, l_scale_r, norms_used_r = compute_forces_scan_func(
            state_post,
            include_edge=False,
            zero_m1=zero_m1,
            constraint_precond_diag=zero_precond_diag,
            constraint_tcon=zero_tcon,
            constraint_precond_active=constraint_active_false,
            constraint_tcon_active=constraint_active_false,
            iter_idx=None,
        )
    fsqr_r = norms_used_r.r1 * norms_used_r.fnorm * gcr2_r
    fsqz_r = norms_used_r.r1 * norms_used_r.fnorm * gcz2_r
    fsql_r = norms_used_r.fnormL * gcl2_r

    rz_norm_r = rz_norm_func(state_post)
    f_norm1_r = jnp.where(rz_norm_r != 0.0, 1.0 / rz_norm_r, jnp.asarray(float("inf"), dtype=dtype))

    if constraint_tcon0 is None or float(constraint_tcon0) == 0.0:
        cache_precond_diag_r = zero_precond_diag
        cache_tcon_r = zero_tcon
    else:
        from vmec_jax.kernels.constraints import precondn_diag_axd1_from_bcovar

        ard1_r, azd1_r = precondn_diag_axd1_from_bcovar(
            trig=trig,
            s=s,
            bsq=k_r.bc.bsq,
            r12=k_r.bc.jac.r12,
            sqrtg=k_r.bc.jac.sqrtg,
            ru12=k_r.bc.jac.ru12,
            zu12=k_r.bc.jac.zu12,
        )
        cache_precond_diag_r = (ard1_r, azd1_r)
        cache_tcon_r = jnp.asarray(k_r.tcon)
    cache_lam_prec_r = lambda_preconditioner_func(k_r.bc)
    from vmec_jax.preconditioner_1d_jax import rz_preconditioner_matrices

    mats_r, _jmin, _jmax = rz_preconditioner_matrices(
        bc=k_r.bc,
        k=k_r,
        trig=trig,
        s=s,
        cfg=cfg,
        use_precomputed=bool(scan_use_precomputed),
        use_lax_tridi=bool(scan_use_lax_tridi),
    )
    mats_r = compact_scan_rz_mats_for_carry(mats_r)
    frzl_rhs_r = scale_m1_precond_rhs_func(frzl_r, mats_r)
    from vmec_jax.preconditioner_1d_jax import rz_preconditioner_apply

    frzl_rz_r = rz_preconditioner_apply(
        frzl_in=frzl_rhs_r,
        mats=mats_r,
        jmax=jmax0,
        cfg=cfg,
        use_precomputed=bool(scan_use_precomputed),
        use_lax_tridi=bool(scan_use_lax_tridi),
    )
    return restart_scan_payload(
        frzl_rz=frzl_rz_r,
        cache_lam_prec=cache_lam_prec_r,
        w_mode_mn=w_mode_mn,
        lambda_update_scale_j=lambda_update_scale_j,
        apply_lambda_update_scale=bool(apply_lambda_update_scale),
        fsqr=fsqr_r,
        fsqz=fsqz_r,
        fsql=fsql_r,
        f_norm1=f_norm1_r,
        delta_s=delta_s,
        s=s,
        lconm1=bool(getattr(cfg, "lconm1", True)),
        cache_precond_diag=cache_precond_diag_r,
        cache_tcon=cache_tcon_r,
        cache_norms=norms_used_r,
        cache_rz_scale=rz_scale_r,
        cache_l_scale=l_scale_r,
        cache_rz_norm=rz_norm_r,
        cache_f_norm1=f_norm1_r,
        cache_rz_mats=mats_r,
        cache_valid=jnp.asarray(True),
    )


def current_scan_payload(**kwargs: Any) -> ScanForcePayload:
    """Evaluate current scan payload for fixed-boundary VMEC solve and implicit differentiation."""
    return build_scan_force_payload(**kwargs)


def restart_scan_payload(**kwargs: Any) -> ScanForcePayload:
    """Evaluate restart scan payload for fixed-boundary VMEC solve and implicit differentiation."""
    return build_scan_force_payload(**kwargs)


def mask_scan_restart_payload(*, payload: ScanForcePayload, do_restart: Any) -> ScanForcePayload:
    """Evaluate mask scan restart payload for fixed-boundary VMEC solve and implicit differentiation."""
    masked_blocks, cache_valid = mask_scan_restart_force_payload(
        force_blocks=tuple(payload.blocks),
        cache_valid=payload.cache_valid,
        do_restart=do_restart,
    )
    return payload._replace(blocks=ScanForceBlocks(*masked_blocks), cache_valid=cache_valid)


def select_scan_force_payload(
    *,
    do_restart: Any,
    use_restart_payload: bool,
    restart_payload_fn: Callable[[Any], ScanForcePayload],
    current_payload_fn: Callable[[Any], ScanForcePayload],
    cond: Callable[..., ScanForcePayload] | None = None,
) -> ScanForcePayload:
    """Select restart/current payloads while preserving the no-restart fast path."""

    if bool(use_restart_payload):
        cond_fn = cond if cond is not None else jax.lax.cond
        return cond_fn(do_restart, restart_payload_fn, current_payload_fn, operand=None)
    return mask_scan_restart_payload(payload=current_payload_fn(None), do_restart=do_restart)


def build_scan_step_fields(
    *,
    payload: ScanForcePayload,
    state_post: Any,
    velocity_blocks_post: tuple[Any, ...],
    inv_tau_post: Any,
    fsq_prev_post: Any,
    fsq1: Any,
    time_step_post: Any,
    iter2: Any,
    iter1_post: Any,
    k_ndamp: int,
    dtype: Any,
    flip_sign: Any,
    lasym: bool,
    static: Any,
    edge_Rcos: Any,
    edge_Rsin: Any,
    edge_Zcos: Any,
    edge_Zsin: Any,
    free_boundary_enabled: bool,
    idx00: Any,
    mn_cos_to_signed_physical: Callable[[Any, Any], Any],
    mn_sin_to_signed_physical: Callable[[Any, Any], Any],
    mn_sin_to_signed_physical_lambda: Callable[[Any, Any], Any],
    mn_cos_to_signed_physical_lambda: Callable[[Any, Any], Any],
    enforce_fixed_boundary_and_axis: Callable[..., Any],
    apply_vmec_lambda_axis_rules: Callable[[Any], Any],
    vmec2000_control: bool,
    do_restart: Any,
    cond: Callable[..., ScanStepFields] | None = None,
) -> ScanStepFields:
    """Apply VMEC scan accept/reject semantics to state and velocity fields.

    The caller still owns branch policy.  This helper owns only the numerical
    update once a preconditioned force payload and post-restart fields exist.
    """

    (
        vRcc_post,
        vRss_post,
        vZsc_post,
        vZcs_post,
        vLsc_post,
        vLcs_post,
        vRsc_post,
        vRcs_post,
        vZcc_post,
        vZss_post,
        vLcc_post,
        vLss_post,
    ) = velocity_blocks_post
    blocks = payload.blocks

    def _accept_step(_):
        inv_tau_reset = jnp.full((int(k_ndamp),), jnp.asarray(0.15, dtype=dtype) / time_step_post)
        invtau_num = jnp.where(
            fsq1 == 0.0,
            0.0,
            jnp.minimum(jnp.abs(jnp.log(fsq1 / jnp.maximum(fsq_prev_post, 1.0e-30))), 0.15),
        )
        inv_tau = jnp.where(
            iter2 == iter1_post,
            inv_tau_reset,
            jnp.concatenate([inv_tau_post[1:], invtau_num[None] / time_step_post], axis=0),
        )
        fsq_prev = fsq1
        otav = jnp.sum(inv_tau) / float(k_ndamp)
        dtau = time_step_post * otav / 2.0
        b1 = 1.0 - dtau
        fac = 1.0 / (1.0 + dtau)
        force_scale = time_step_post
        vRcc = fac * (b1 * vRcc_post + force_scale * (flip_sign * blocks.frcc))
        vRss = fac * (b1 * vRss_post + force_scale * (flip_sign * blocks.frss))
        vZsc = fac * (b1 * vZsc_post + force_scale * (flip_sign * blocks.fzsc))
        vZcs = fac * (b1 * vZcs_post + force_scale * (flip_sign * blocks.fzcs))
        vLsc = fac * (b1 * vLsc_post + force_scale * (flip_sign * blocks.flsc))
        vLcs = fac * (b1 * vLcs_post + force_scale * (flip_sign * blocks.flcs))
        dR = time_step_post * mn_cos_to_signed_physical(vRcc, vRss)
        dZ = time_step_post * mn_sin_to_signed_physical(vZsc, vZcs)
        dL = time_step_post * mn_sin_to_signed_physical_lambda(vLsc, vLcs)
        if bool(lasym):
            vRsc = fac * (b1 * vRsc_post + force_scale * (flip_sign * blocks.frsc))
            vRcs = fac * (b1 * vRcs_post + force_scale * (flip_sign * blocks.frcs))
            vZcc = fac * (b1 * vZcc_post + force_scale * (flip_sign * blocks.fzcc))
            vZss = fac * (b1 * vZss_post + force_scale * (flip_sign * blocks.fzss))
            vLcc = fac * (b1 * vLcc_post + force_scale * (flip_sign * blocks.flcc))
            vLss = fac * (b1 * vLss_post + force_scale * (flip_sign * blocks.flss))
            dR_sin = time_step_post * mn_sin_to_signed_physical(vRsc, vRcs)
            dZ_cos = time_step_post * mn_cos_to_signed_physical(vZcc, vZss)
            dL_cos = time_step_post * mn_cos_to_signed_physical_lambda(vLcc, vLss)
        else:
            vRsc = vRsc_post
            vRcs = vRcs_post
            vZcc = vZcc_post
            vZss = vZss_post
            vLcc = vLcc_post
            vLss = vLss_post
            dR_sin = jnp.zeros_like(dR)
            dZ_cos = jnp.zeros_like(dR)
            dL_cos = jnp.zeros_like(dR)
        state_new = type(state_post)(
            layout=state_post.layout,
            Rcos=jnp.asarray(state_post.Rcos) + dR,
            Rsin=jnp.asarray(state_post.Rsin) + dR_sin,
            Zcos=jnp.asarray(state_post.Zcos) + dZ_cos,
            Zsin=jnp.asarray(state_post.Zsin) + dZ,
            Lcos=jnp.asarray(state_post.Lcos) + dL_cos,
            Lsin=jnp.asarray(state_post.Lsin) + dL,
        )
        state_new = enforce_fixed_boundary_and_axis(
            state_new,
            static,
            edge_Rcos=edge_Rcos,
            edge_Rsin=edge_Rsin,
            edge_Zcos=edge_Zcos,
            edge_Zsin=edge_Zsin,
            enforce_edge=not bool(free_boundary_enabled),
            enforce_lambda_axis=True,
            idx00=idx00,
        )
        state_new = apply_vmec_lambda_axis_rules(state_new)
        return ScanStepFields(
            state=state_new,
            vRcc=vRcc,
            vRss=vRss,
            vZsc=vZsc,
            vLsc=vLsc,
            vLcs=vLcs,
            vRsc=vRsc,
            vRcs=vRcs,
            vZcc=vZcc,
            vZss=vZss,
            vLcc=vLcc,
            vLss=vLss,
            inv_tau=inv_tau,
            fsq_prev=fsq_prev,
            vZcs=vZcs,
        )

    def _reject_step(_):
        return ScanStepFields(
            state=state_post,
            vRcc=vRcc_post,
            vRss=vRss_post,
            vZsc=vZsc_post,
            vZcs=vZcs_post,
            vLsc=vLsc_post,
            vLcs=vLcs_post,
            vRsc=vRsc_post,
            vRcs=vRcs_post,
            vZcc=vZcc_post,
            vZss=vZss_post,
            vLcc=vLcc_post,
            vLss=vLss_post,
            inv_tau=inv_tau_post,
            fsq_prev=fsq_prev_post,
        )

    return select_scan_step_fields(
        vmec2000_control=bool(vmec2000_control),
        do_restart=do_restart,
        accept_step_fn=_accept_step,
        reject_step_fn=_reject_step,
        cond=cond,
    )


def select_payload_and_build_step_fields(
    *,
    do_restart: Any,
    use_restart_payload: bool,
    current_payload: ScanForcePayload,
    state_post: Any,
    compute_forces_scan_func: Callable[..., tuple[Any, TomnspsRZL, Any, Any, Any, Any, Any, Any]],
    restart_trace_context: Callable[[], Any],
    zero_m1: Any,
    zero_precond_diag: Any,
    zero_tcon: Any,
    constraint_active_false: Any,
    constraint_tcon0: Any,
    trig: Any,
    s: Any,
    cfg: Any,
    dtype: Any,
    scan_use_precomputed: bool,
    scan_use_lax_tridi: bool,
    lambda_preconditioner_func: Callable[[Any], Any],
    rz_norm_func: Callable[[Any], Any],
    scale_m1_precond_rhs_func: Callable[[TomnspsRZL, Any], TomnspsRZL],
    w_mode_mn: Any,
    lambda_update_scale_j: Any,
    apply_lambda_update_scale: bool,
    delta_s: Any,
    jmax0: Any,
    velocity_blocks_post: tuple[Any, ...],
    inv_tau_post: Any,
    fsq_prev_post: Any,
    time_step_post: Any,
    iter2: Any,
    iter1_post: Any,
    k_ndamp: int,
    flip_sign: Any,
    lasym: bool,
    static: Any,
    edge_Rcos: Any,
    edge_Rsin: Any,
    edge_Zcos: Any,
    edge_Zsin: Any,
    free_boundary_enabled: bool,
    idx00: Any,
    mn_cos_to_signed_physical: Callable[[Any, Any], Any],
    mn_sin_to_signed_physical: Callable[[Any, Any], Any],
    mn_sin_to_signed_physical_lambda: Callable[[Any, Any], Any],
    mn_cos_to_signed_physical_lambda: Callable[[Any, Any], Any],
    enforce_fixed_boundary_and_axis: Callable[..., Any],
    apply_vmec_lambda_axis_rules: Callable[[Any], Any],
    vmec2000_control: bool,
    cond: Callable[..., Any],
) -> ScanSelectedPayloadStep:
    """Select restart/current payloads and build accepted-step fields."""

    def _restart_payload(_):
        return build_restart_preconditioned_scan_payload(
            state_post=state_post,
            compute_forces_scan_func=compute_forces_scan_func,
            trace_context=restart_trace_context,
            zero_m1=zero_m1,
            zero_precond_diag=zero_precond_diag,
            zero_tcon=zero_tcon,
            constraint_active_false=constraint_active_false,
            constraint_tcon0=constraint_tcon0,
            trig=trig,
            s=s,
            cfg=cfg,
            dtype=dtype,
            scan_use_precomputed=bool(scan_use_precomputed),
            scan_use_lax_tridi=bool(scan_use_lax_tridi),
            lambda_preconditioner_func=lambda_preconditioner_func,
            rz_norm_func=rz_norm_func,
            scale_m1_precond_rhs_func=scale_m1_precond_rhs_func,
            w_mode_mn=w_mode_mn,
            lambda_update_scale_j=lambda_update_scale_j,
            apply_lambda_update_scale=bool(apply_lambda_update_scale),
            delta_s=delta_s,
            jmax0=jmax0,
        )

    payload = select_scan_force_payload(
        do_restart=do_restart,
        use_restart_payload=bool(use_restart_payload),
        restart_payload_fn=_restart_payload,
        current_payload_fn=lambda _: current_payload,
        cond=cond,
    )
    fsq1 = payload.fsqr1 + payload.fsqz1 + payload.fsql1
    step_fields = build_scan_step_fields(
        payload=payload,
        state_post=state_post,
        velocity_blocks_post=velocity_blocks_post,
        inv_tau_post=inv_tau_post,
        fsq_prev_post=fsq_prev_post,
        fsq1=fsq1,
        time_step_post=time_step_post,
        iter2=iter2,
        iter1_post=iter1_post,
        k_ndamp=k_ndamp,
        dtype=dtype,
        flip_sign=flip_sign,
        lasym=bool(lasym),
        static=static,
        edge_Rcos=edge_Rcos,
        edge_Rsin=edge_Rsin,
        edge_Zcos=edge_Zcos,
        edge_Zsin=edge_Zsin,
        free_boundary_enabled=bool(free_boundary_enabled),
        idx00=idx00,
        mn_cos_to_signed_physical=mn_cos_to_signed_physical,
        mn_sin_to_signed_physical=mn_sin_to_signed_physical,
        mn_sin_to_signed_physical_lambda=mn_sin_to_signed_physical_lambda,
        mn_cos_to_signed_physical_lambda=mn_cos_to_signed_physical_lambda,
        enforce_fixed_boundary_and_axis=enforce_fixed_boundary_and_axis,
        apply_vmec_lambda_axis_rules=apply_vmec_lambda_axis_rules,
        vmec2000_control=bool(vmec2000_control),
        do_restart=do_restart,
        cond=cond,
    )
    return ScanSelectedPayloadStep(
        payload=payload,
        step_fields=step_fields,
        fsqr=payload.fsqr,
        fsqz=payload.fsqz,
        fsql=payload.fsql,
        fsq1=fsq1,
    )


def select_scan_step_fields(
    *,
    vmec2000_control: bool,
    do_restart: Any,
    accept_step_fn: Callable[[Any], ScanStepFields],
    reject_step_fn: Callable[[Any], ScanStepFields],
    cond: Callable[..., ScanStepFields] | None = None,
) -> ScanStepFields:
    """Select accepted/rejected scan step fields with VMEC2000 retry semantics."""

    if bool(vmec2000_control):
        return accept_step_fn(None)
    cond_fn = cond if cond is not None else jax.lax.cond
    return cond_fn(do_restart, reject_step_fn, accept_step_fn, operand=None)


_current_payload = current_scan_payload
_restart_payload = restart_scan_payload
