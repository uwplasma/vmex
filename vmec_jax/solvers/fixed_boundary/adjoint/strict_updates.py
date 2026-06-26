"""Strict accepted-step primitives for fixed-boundary discrete adjoints."""

from __future__ import annotations

import numpy as np

from vmec_jax._compat import jnp
from vmec_jax.kernels.tomnsp import TomnspsRZL

def strict_update_velocity_block(
    *,
    b1,
    fac,
    force_scale,
    flip_sign,
    vRcc_before,
    vRss_before,
    vZsc_before,
    vZcs_before,
    vLsc_before,
    vLcs_before,
    frcc_u,
    frss_u,
    fzsc_u,
    fzcs_u,
    flsc_u,
    flcs_u,
    vRsc_before=None,
    vRcs_before=None,
    vZcc_before=None,
    vZss_before=None,
    vLcc_before=None,
    vLss_before=None,
    frsc_u=None,
    frcs_u=None,
    fzcc_u=None,
    fzss_u=None,
    flcc_u=None,
    flss_u=None,
):
    """Apply the strict-update velocity recurrence for one solver step."""
    b1 = jnp.asarray(b1, dtype=jnp.asarray(vRcc_before).dtype)
    fac = jnp.asarray(fac, dtype=jnp.asarray(vRcc_before).dtype)
    force_scale = jnp.asarray(force_scale, dtype=jnp.asarray(vRcc_before).dtype)
    flip_sign = jnp.asarray(flip_sign, dtype=jnp.asarray(vRcc_before).dtype)
    scale = fac * force_scale * flip_sign
    memory = fac * b1

    def _update(v_before, force):
        return memory * jnp.asarray(v_before) + scale * jnp.asarray(force)

    vRcc_after = _update(vRcc_before, frcc_u)
    vRss_after = _update(vRss_before, frss_u)
    vZsc_after = _update(vZsc_before, fzsc_u)
    vZcs_after = _update(vZcs_before, fzcs_u)
    vLsc_after = _update(vLsc_before, flsc_u)
    vLcs_after = _update(vLcs_before, flcs_u)
    if vRsc_before is None:
        vRsc_after = None
        vRcs_after = None
        vZcc_after = None
        vZss_after = None
        vLcc_after = None
        vLss_after = None
    else:
        vRsc_after = _update(vRsc_before, frsc_u)
        vRcs_after = _update(vRcs_before, frcs_u)
        vZcc_after = _update(vZcc_before, fzcc_u)
        vZss_after = _update(vZss_before, fzss_u)
        vLcc_after = _update(vLcc_before, flcc_u)
        vLss_after = _update(vLss_before, flss_u)
    return {
        "vRcc_after": vRcc_after,
        "vRss_after": vRss_after,
        "vZsc_after": vZsc_after,
        "vZcs_after": vZcs_after,
        "vLsc_after": vLsc_after,
        "vLcs_after": vLcs_after,
        "vRsc_after": vRsc_after,
        "vRcs_after": vRcs_after,
        "vZcc_after": vZcc_after,
        "vZss_after": vZss_after,
        "vLcc_after": vLcc_after,
        "vLss_after": vLss_after,
    }


def strict_update_velocity_limit(
    *,
    dt_eff,
    max_update_rms,
    limit_update_rms,
    need_update_rms: bool = True,
    vRcc,
    vRss,
    vZsc,
    vZcs,
    vLsc,
    vLcs,
    vRsc=None,
    vRcs=None,
    vZcc=None,
    vZss=None,
    vLcc=None,
    vLss=None,
):
    """Apply the strict-update velocity RMS limiter for one solver step."""
    dt_eff = jnp.asarray(dt_eff, dtype=jnp.asarray(vRcc).dtype)
    max_update_rms = jnp.asarray(max_update_rms, dtype=jnp.asarray(vRcc).dtype)
    limit_update_rms = jnp.asarray(limit_update_rms, dtype=bool)
    need_update_rms = jnp.asarray(need_update_rms, dtype=bool)
    base = jnp.asarray(vRcc)
    zeros = jnp.zeros_like(base)
    pieces = [
        base,
        jnp.asarray(vRss),
        jnp.asarray(vRsc) if vRsc is not None else zeros,
        jnp.asarray(vRcs) if vRcs is not None else zeros,
        jnp.asarray(vZsc),
        jnp.asarray(vZcs),
        jnp.asarray(vZcc) if vZcc is not None else zeros,
        jnp.asarray(vZss) if vZss is not None else zeros,
        jnp.asarray(vLsc),
        jnp.asarray(vLcs),
        jnp.asarray(vLcc) if vLcc is not None else zeros,
        jnp.asarray(vLss) if vLss is not None else zeros,
    ]
    sq = sum((dt_eff * p) ** 2 for p in pieces)
    raw_update_rms = jnp.sqrt(jnp.mean(sq))
    report_update_rms = jnp.where(
        jnp.logical_or(limit_update_rms, need_update_rms),
        raw_update_rms,
        jnp.asarray(0.0, dtype=raw_update_rms.dtype),
    )
    clipped_scale = jnp.where(
        jnp.isfinite(raw_update_rms) & (raw_update_rms > max_update_rms),
        max_update_rms / jnp.maximum(raw_update_rms, jnp.asarray(1.0e-30, dtype=raw_update_rms.dtype)),
        jnp.asarray(1.0, dtype=raw_update_rms.dtype),
    )
    scale = jnp.where(limit_update_rms, clipped_scale, jnp.asarray(1.0, dtype=raw_update_rms.dtype))

    def _scale(x):
        return None if x is None else scale * jnp.asarray(x)

    out = {
        "vRcc": _scale(vRcc),
        "vRss": _scale(vRss),
        "vZsc": _scale(vZsc),
        "vZcs": _scale(vZcs),
        "vLsc": _scale(vLsc),
        "vLcs": _scale(vLcs),
        "vRsc": _scale(vRsc),
        "vRcs": _scale(vRcs),
        "vZcc": _scale(vZcc),
        "vZss": _scale(vZss),
        "vLcc": _scale(vLcc),
        "vLss": _scale(vLss),
        "update_rms_preclip": report_update_rms,
        "update_rms_scale": scale,
        "update_rms_postclip": scale * report_update_rms,
    }
    return out


def preconditioned_force_channels_from_rz_output(
    *,
    frzl_rz,
    lam_prec,
    w_mode_mn,
    lambda_update_scale=1.0,
):
    """Map R/Z preconditioner output into solver force channels for one step."""
    frcc = jnp.asarray(frzl_rz.frcc)
    zeros_r = jnp.zeros_like(frcc)
    fzsc = jnp.asarray(frzl_rz.fzsc)
    zeros_z = jnp.zeros_like(fzsc)
    flsc = jnp.asarray(frzl_rz.flsc) * jnp.asarray(lam_prec)
    zeros_l = jnp.zeros_like(flsc)

    frss = None if frzl_rz.frss is None else jnp.asarray(frzl_rz.frss)
    fzcs = None if frzl_rz.fzcs is None else jnp.asarray(frzl_rz.fzcs)
    flcs = None if frzl_rz.flcs is None else (jnp.asarray(frzl_rz.flcs) * jnp.asarray(lam_prec))
    frsc = jnp.asarray(frzl_rz.frsc) if getattr(frzl_rz, "frsc", None) is not None else zeros_r
    frcs = jnp.asarray(frzl_rz.frcs) if getattr(frzl_rz, "frcs", None) is not None else zeros_r
    fzcc = jnp.asarray(frzl_rz.fzcc) if getattr(frzl_rz, "fzcc", None) is not None else zeros_z
    fzss = jnp.asarray(frzl_rz.fzss) if getattr(frzl_rz, "fzss", None) is not None else zeros_z
    flcc = (
        jnp.asarray(frzl_rz.flcc) * jnp.asarray(lam_prec)
        if getattr(frzl_rz, "flcc", None) is not None
        else zeros_l
    )
    flss = (
        jnp.asarray(frzl_rz.flss) * jnp.asarray(lam_prec)
        if getattr(frzl_rz, "flss", None) is not None
        else zeros_l
    )

    frzl_pre = TomnspsRZL(
        frcc=frcc,
        frss=frss,
        fzsc=fzsc,
        fzcs=fzcs,
        flsc=flsc,
        flcs=flcs,
        frsc=frsc,
        frcs=frcs,
        fzcc=fzcc,
        fzss=fzss,
        flcc=flcc,
        flss=flss,
    )
    w = jnp.asarray(w_mode_mn)[None, :, :]
    frcc_u = frcc * w
    frss_u = (frss if frss is not None else zeros_r) * w
    fzsc_u = fzsc * w
    fzcs_u = (fzcs if fzcs is not None else zeros_z) * w
    flsc_u = flsc * w
    flcs_u = (flcs if flcs is not None else zeros_l) * w
    frsc_u = frsc * w
    frcs_u = frcs * w
    fzcc_u = fzcc * w
    fzss_u = fzss * w
    flcc_u = flcc * w
    flss_u = flss * w
    scale = jnp.asarray(lambda_update_scale, dtype=flsc_u.dtype)
    flsc_u = flsc_u * scale
    flcs_u = flcs_u * scale
    flcc_u = flcc_u * scale
    flss_u = flss_u * scale
    return {
        "frzl_pre": frzl_pre,
        "frcc_u": frcc_u,
        "frss_u": frss_u,
        "fzsc_u": fzsc_u,
        "fzcs_u": fzcs_u,
        "flsc_u": flsc_u,
        "flcs_u": flcs_u,
        "frsc_u": frsc_u,
        "frcs_u": frcs_u,
        "fzcc_u": fzcc_u,
        "fzss_u": fzss_u,
        "flcc_u": flcc_u,
        "flss_u": flss_u,
    }


def preconditioned_force_channels_from_raw_forces(
    *,
    frzl,
    mats,
    jmax,
    cfg,
    lam_prec,
    w_mode_mn,
    lambda_update_scale=1.0,
    use_precomputed: bool | None = None,
    use_lax_tridi: bool | None = None,
    jit_preconditioner_apply: bool = True,
):
    """Apply the radial preconditioner, lambda scaling, and mode scaling."""
    from vmec_jax.preconditioner_1d_jax import rz_preconditioner_apply, rz_preconditioner_apply_jit
    from vmec_jax.solve import _scale_mode_slice, _vmec_scale_m1_factors_from_mats

    frzl_rhs = frzl
    if bool(getattr(cfg, "lconm1", True)) and int(cfg.mpol) > 1:
        fac_r, fac_z = _vmec_scale_m1_factors_from_mats(mats)
        if int(jnp.asarray(fac_r).size) > 0:
            fac_r = jnp.asarray(fac_r, dtype=jnp.asarray(frzl.frcc).dtype)
            fac_z = jnp.asarray(fac_z, dtype=jnp.asarray(frzl.fzsc).dtype)
            ns_full = int(jnp.asarray(frzl.frcc).shape[0])
            nsolve = min(ns_full, int(fac_r.shape[0]))
            ones_r = jnp.ones((max(ns_full - nsolve, 0),), dtype=jnp.asarray(frzl.frcc).dtype)
            ones_z = jnp.ones((max(ns_full - nsolve, 0),), dtype=jnp.asarray(frzl.fzsc).dtype)
            fac_r_full = fac_r[:nsolve] if nsolve == ns_full else jnp.concatenate([fac_r[:nsolve], ones_r], axis=0)
            fac_z_full = fac_z[:nsolve] if nsolve == ns_full else jnp.concatenate([fac_z[:nsolve], ones_z], axis=0)
            frzl_rhs = TomnspsRZL(
                frcc=frzl.frcc,
                frss=_scale_mode_slice(frzl.frss, mode_idx=1, scale=fac_r_full),
                fzsc=frzl.fzsc,
                fzcs=_scale_mode_slice(frzl.fzcs, mode_idx=1, scale=fac_z_full),
                flsc=frzl.flsc,
                flcs=frzl.flcs,
                frsc=_scale_mode_slice(getattr(frzl, "frsc", None), mode_idx=1, scale=fac_r_full),
                frcs=getattr(frzl, "frcs", None),
                fzcc=_scale_mode_slice(getattr(frzl, "fzcc", None), mode_idx=1, scale=fac_z_full),
                fzss=getattr(frzl, "fzss", None),
                flcc=getattr(frzl, "flcc", None),
                flss=getattr(frzl, "flss", None),
            )

    apply_preconditioner = rz_preconditioner_apply_jit if bool(jit_preconditioner_apply) else rz_preconditioner_apply
    frzl_rz = apply_preconditioner(
        frzl_in=frzl_rhs,
        mats=mats,
        jmax=int(jmax),
        cfg=cfg,
        use_precomputed=use_precomputed,
        use_lax_tridi=use_lax_tridi,
    )
    out = preconditioned_force_channels_from_rz_output(
        frzl_rz=frzl_rz,
        lam_prec=lam_prec,
        w_mode_mn=w_mode_mn,
        lambda_update_scale=lambda_update_scale,
    )
    return {"frzl_rhs": frzl_rhs, "frzl_rz": frzl_rz, **out}


def raw_force_residual_from_state(
    state,
    static,
    *,
    wout_like,
    trig,
    apply_lforbal: bool,
    include_edge_residual: bool,
    apply_m1_constraints: bool,
    zero_m1,
    constraint_tcon0=None,
    constraint_tcon=None,
    constraint_precond_diag=None,
    constraint_precond_active=None,
    constraint_tcon_active=None,
    constraint_rcon0=None,
    constraint_zcon0=None,
    freeb_bsqvac_half=None,
    freeb_pres_scale=None,
):
    """Rebuild the solver's raw VMEC residual blocks for one fixed-boundary step."""
    from dataclasses import replace as dc_replace

    from vmec_jax.kernels.forces import vmec_forces_rz_from_wout, vmec_residual_internal_from_kernels
    from vmec_jax.kernels.residue import vmec_apply_m1_constraints, vmec_apply_scalxc_to_tomnsps, vmec_zero_m1_zforce

    k = vmec_forces_rz_from_wout(
        state=state,
        static=static,
        wout=wout_like,
        indata=None,
        constraint_tcon0=constraint_tcon0,
        constraint_tcon=constraint_tcon,
        constraint_precond_diag=constraint_precond_diag,
        constraint_precond_active=constraint_precond_active,
        constraint_tcon_active=constraint_tcon_active,
        constraint_rcon0=constraint_rcon0,
        constraint_zcon0=constraint_zcon0,
        freeb_bsqvac_half=freeb_bsqvac_half,
        freeb_pres_scale=freeb_pres_scale,
        use_vmec_synthesis=True,
        trig=trig,
        iter_idx=None,
    )
    mask_pack = None
    if getattr(static, "tomnsps_masks", None) is not None:
        mask_pack = static.tomnsps_masks_edge if bool(include_edge_residual) else static.tomnsps_masks
    frzl = vmec_residual_internal_from_kernels(
        k,
        cfg_ntheta=int(static.cfg.ntheta),
        cfg_nzeta=int(static.cfg.nzeta),
        wout=wout_like,
        trig=trig,
        apply_lforbal=bool(apply_lforbal),
        include_edge=bool(include_edge_residual),
        masks=mask_pack,
    )
    if bool(apply_m1_constraints):
        frzl = vmec_apply_m1_constraints(frzl=frzl, lconm1=bool(getattr(static.cfg, "lconm1", True)))
    frzl = vmec_zero_m1_zforce(frzl=frzl, enabled=zero_m1)
    frzl = vmec_apply_scalxc_to_tomnsps(frzl=frzl, s=jnp.asarray(static.s))

    def _nan_guard(x):
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            return x
        x = jnp.asarray(x)
        return jnp.where(jnp.isnan(x), x, x)

    frzl = dc_replace(
        frzl,
        frcc=_nan_guard(frzl.frcc),
        frss=_nan_guard(frzl.frss),
        fzsc=_nan_guard(frzl.fzsc),
        fzcs=_nan_guard(frzl.fzcs),
        flsc=_nan_guard(frzl.flsc),
        flcs=_nan_guard(frzl.flcs),
        frsc=_nan_guard(getattr(frzl, "frsc", None)),
        frcs=_nan_guard(getattr(frzl, "frcs", None)),
        fzcc=_nan_guard(getattr(frzl, "fzcc", None)),
        fzss=_nan_guard(getattr(frzl, "fzss", None)),
        flcc=_nan_guard(getattr(frzl, "flcc", None)),
        flss=_nan_guard(getattr(frzl, "flss", None)),
    )
    return {"k": k, "frzl": frzl}


def state_dependent_preconditioner_from_forces(
    *,
    k,
    static,
    trig,
    dtype=None,
    jmax_override: int | None = None,
    w_mode_mn=None,
    mode_diag_exponent: float = 0.0,
    use_precomputed: bool | None = None,
    use_lax_tridi: bool | None = None,
):
    """Rebuild the solver's state-dependent preconditioner objects."""
    from vmec_jax.preconditioner_1d_jax import lambda_preconditioner, rz_preconditioner_matrices

    cfg = static.cfg
    s = jnp.asarray(static.s)
    lam_prec = lambda_preconditioner(
        bc=k.bc,
        trig=trig,
        s=s,
        cfg=cfg,
        return_faclam=False,
    )
    mats, _jmin, jmax = rz_preconditioner_matrices(
        bc=k.bc,
        k=k,
        trig=trig,
        s=s,
        cfg=cfg,
        jmax_override=jmax_override,
        use_precomputed=use_precomputed,
        use_lax_tridi=use_lax_tridi,
    )
    if dtype is None:
        dtype = jnp.asarray(lam_prec).dtype
    if w_mode_mn is None:
        m = jnp.arange(int(cfg.mpol), dtype=jnp.float64)
        n = jnp.arange(int(cfg.ntor) + 1, dtype=jnp.float64) * float(cfg.nfp)
        k2 = (m[:, None] * m[:, None]) + (n[None, :] * n[None, :])
        w_mode_mn = (1.0 + k2).astype(jnp.float64) ** (-float(mode_diag_exponent))
        w_mode_mn = w_mode_mn.astype(dtype)
    else:
        w_mode_mn = jnp.asarray(w_mode_mn, dtype=dtype)
    return {
        "lam_prec": lam_prec,
        "mats": mats,
        "jmax": int(jmax_override) if (jmax_override is not None) else int(jmax),
        "w_mode_mn": w_mode_mn,
    }

def strict_update_accepted_step(
    state_pre,
    static,
    *,
    dt_eff,
    b1,
    fac,
    force_scale,
    flip_sign,
    vRcc_before,
    vRss_before,
    vZsc_before,
    vZcs_before,
    vLsc_before,
    vLcs_before,
    frcc_u,
    frss_u,
    fzsc_u,
    fzcs_u,
    flsc_u,
    flcs_u,
    vRsc_before=None,
    vRcs_before=None,
    vZcc_before=None,
    vZss_before=None,
    vLcc_before=None,
    vLss_before=None,
    frsc_u=None,
    frcs_u=None,
    fzcc_u=None,
    fzss_u=None,
    flcc_u=None,
    flss_u=None,
    max_update_rms=5.0e-3,
    limit_update_rms: bool = True,
    need_update_rms: bool = True,
    divide_by_scalxc_for_update: bool = False,
    enforce_edge: bool = True,
):
    """Compose the accepted strict-update velocity and state-advance blocks.

    ``enforce_edge`` preserves the historical fixed-boundary default.  Free
    boundary callers pass ``False`` so the fused update does not pin the LCFS.
    """
    velocity_raw = strict_update_velocity_block(
        b1=b1,
        fac=fac,
        force_scale=force_scale,
        flip_sign=flip_sign,
        vRcc_before=vRcc_before,
        vRss_before=vRss_before,
        vZsc_before=vZsc_before,
        vZcs_before=vZcs_before,
        vLsc_before=vLsc_before,
        vLcs_before=vLcs_before,
        frcc_u=frcc_u,
        frss_u=frss_u,
        fzsc_u=fzsc_u,
        fzcs_u=fzcs_u,
        flsc_u=flsc_u,
        flcs_u=flcs_u,
        vRsc_before=vRsc_before,
        vRcs_before=vRcs_before,
        vZcc_before=vZcc_before,
        vZss_before=vZss_before,
        vLcc_before=vLcc_before,
        vLss_before=vLss_before,
        frsc_u=frsc_u,
        frcs_u=frcs_u,
        fzcc_u=fzcc_u,
        fzss_u=fzss_u,
        flcc_u=flcc_u,
        flss_u=flss_u,
    )
    velocity_out = strict_update_velocity_limit(
        dt_eff=dt_eff,
        max_update_rms=max_update_rms,
        limit_update_rms=limit_update_rms,
        need_update_rms=need_update_rms,
        vRcc=velocity_raw["vRcc_after"],
        vRss=velocity_raw["vRss_after"],
        vZsc=velocity_raw["vZsc_after"],
        vZcs=velocity_raw["vZcs_after"],
        vLsc=velocity_raw["vLsc_after"],
        vLcs=velocity_raw["vLcs_after"],
        vRsc=velocity_raw["vRsc_after"],
        vRcs=velocity_raw["vRcs_after"],
        vZcc=velocity_raw["vZcc_after"],
        vZss=velocity_raw["vZss_after"],
        vLcc=velocity_raw["vLcc_after"],
        vLss=velocity_raw["vLss_after"],
    )
    state_post = strict_update_velocity_state_advance(
        state_pre,
        static,
        dt_eff=dt_eff,
        vRcc=velocity_out["vRcc"],
        vRss=velocity_out["vRss"],
        vZsc=velocity_out["vZsc"],
        vZcs=velocity_out["vZcs"],
        vLsc=velocity_out["vLsc"],
        vLcs=velocity_out["vLcs"],
        vRsc=velocity_out["vRsc"],
        vRcs=velocity_out["vRcs"],
        vZcc=velocity_out["vZcc"],
        vZss=velocity_out["vZss"],
        vLcc=velocity_out["vLcc"],
        vLss=velocity_out["vLss"],
        edge_Rcos=jnp.asarray(state_pre.Rcos)[-1, :],
        edge_Rsin=jnp.asarray(state_pre.Rsin)[-1, :],
        edge_Zcos=jnp.asarray(state_pre.Zcos)[-1, :],
        edge_Zsin=jnp.asarray(state_pre.Zsin)[-1, :],
        divide_by_scalxc_for_update=divide_by_scalxc_for_update,
        enforce_edge=enforce_edge,
    )
    return {
        "state_post": state_post,
        "vRcc_after": velocity_out["vRcc"],
        "vRss_after": velocity_out["vRss"],
        "vZsc_after": velocity_out["vZsc"],
        "vZcs_after": velocity_out["vZcs"],
        "vLsc_after": velocity_out["vLsc"],
        "vLcs_after": velocity_out["vLcs"],
        "vRsc_after": velocity_out["vRsc"],
        "vRcs_after": velocity_out["vRcs"],
        "vZcc_after": velocity_out["vZcc"],
        "vZss_after": velocity_out["vZss"],
        "vLcc_after": velocity_out["vLcc"],
        "vLss_after": velocity_out["vLss"],
        "update_rms_preclip": velocity_out["update_rms_preclip"],
        "update_rms_scale": velocity_out["update_rms_scale"],
        "update_rms_postclip": velocity_out["update_rms_postclip"],
    }


def strict_update_velocity_state_advance(
    state,
    static,
    *,
    dt_eff,
    vRcc,
    vRss,
    vZsc,
    vZcs,
    vLsc,
    vLcs,
    edge_Rcos,
    edge_Rsin,
    edge_Zcos,
    edge_Zsin,
    vRsc=None,
    vRcs=None,
    vZcc=None,
    vZss=None,
    vLcc=None,
    vLss=None,
    divide_by_scalxc_for_update: bool = False,
    enforce_edge: bool = True,
):
    """Apply the strict-update state-advance block from VMEC residual iteration.

    This is the accepted geometry/lambda update map after the velocity blocks
    have already been formed for the current step. It excludes force assembly
    and acceptance/restart logic and is intended as the first local reverse-mode
    target for the discrete-adjoint refactor.
    """
    from vmec_jax.solve import _apply_vmec_lambda_axis_rules_to_state, _enforce_fixed_boundary_and_axis, _mode00_index
    from vmec_jax.kernels.parity import _mn_cos_to_signed_cached, _mn_sin_to_signed_cached, signed_maps_from_modes
    from vmec_jax.kernels.residue import vmec_scalxc_from_s

    dt_eff = jnp.asarray(dt_eff, dtype=jnp.asarray(state.Rcos).dtype)
    scalxc = vmec_scalxc_from_s(s=jnp.asarray(static.s), mpol=int(static.cfg.mpol)).astype(jnp.asarray(state.Rcos).dtype)
    scalxc = scalxc[:, :, None]
    scalxc = jnp.where(jnp.asarray(divide_by_scalxc_for_update, dtype=bool), scalxc, jnp.ones_like(scalxc))
    maps = static.signed_maps if getattr(static, "signed_maps", None) is not None else signed_maps_from_modes(static.modes)
    ncoeff = int(static.modes.K)
    idx00 = _mode00_index(static.modes)

    def _cos_phys(cc, ss):
        cc = jnp.asarray(cc) / scalxc
        ss = jnp.asarray(ss) / scalxc if ss is not None else None
        return _mn_cos_to_signed_cached(cc, ss, maps=maps, ncoeff=ncoeff)

    def _sin_phys(sc, cs):
        sc = jnp.asarray(sc) / scalxc
        cs = jnp.asarray(cs) / scalxc if cs is not None else None
        return _mn_sin_to_signed_cached(sc, cs, maps=maps, ncoeff=ncoeff)

    dR = dt_eff * _cos_phys(vRcc, vRss)
    dZ = dt_eff * _sin_phys(vZsc, vZcs)
    dL = dt_eff * _sin_phys(vLsc, vLcs)
    if bool(static.cfg.lasym):
        dR_sin = dt_eff * _sin_phys(vRsc, vRcs)
        dZ_cos = dt_eff * _cos_phys(vZcc, vZss)
        dL_cos = dt_eff * _cos_phys(vLcc, vLss)
    else:
        dR_sin = jnp.zeros_like(dR)
        dZ_cos = jnp.zeros_like(dR)
        dL_cos = jnp.zeros_like(dR)

    state_try = type(state)(
        layout=state.layout,
        Rcos=jnp.asarray(state.Rcos) + dR,
        Rsin=jnp.asarray(state.Rsin) + dR_sin,
        Zcos=jnp.asarray(state.Zcos) + dZ_cos,
        Zsin=jnp.asarray(state.Zsin) + dZ,
        Lcos=jnp.asarray(state.Lcos) + dL_cos,
        Lsin=jnp.asarray(state.Lsin) + dL,
    )
    state_out = _enforce_fixed_boundary_and_axis(
        state_try,
        static,
        edge_Rcos=edge_Rcos,
        edge_Rsin=edge_Rsin,
        edge_Zcos=edge_Zcos,
        edge_Zsin=edge_Zsin,
        enforce_edge=enforce_edge,
        enforce_lambda_axis=True,
        idx00=idx00,
    )
    return _apply_vmec_lambda_axis_rules_to_state(
        state_out,
        enforce_vmec_lambda_axis=True,
        host_update_assembly=False,
        idx00=idx00,
    )
