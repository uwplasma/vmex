"""JAX ports of VMEC2000 1D (radial) preconditioner operators.

This module mirrors the reference NumPy implementation in
:mod:`vmec_jax.preconditioner_1d`, but uses JAX arrays/ops so the fixed-point
update loop stays JIT-able and differentiable.
"""

from __future__ import annotations

from typing import Any

import jax

from ._compat import jnp
from .vmec_tomnsp import TomnspsRZL


def _sqrt_profiles_from_ns(ns: int, *, dtype) -> tuple[Any, Any]:
    ns = int(ns)
    if ns <= 0:
        z = jnp.zeros((0,), dtype=dtype)
        return z, z
    if ns == 1:
        return jnp.zeros((1,), dtype=dtype), jnp.zeros((0,), dtype=dtype)
    denom = jnp.asarray(float(ns - 1), dtype=dtype)
    full_pos = jnp.linspace(0.0, 1.0, ns, dtype=dtype)
    sqrt_sf = jnp.sqrt(jnp.maximum(full_pos, 0.0))
    half_pos = (jnp.arange(ns - 1, dtype=dtype) + 0.5) / denom
    sqrt_sh = jnp.sqrt(jnp.maximum(half_pos, 0.0))
    return sqrt_sf, sqrt_sh


def _sm_sp_from_profiles(sqrt_sf, sqrt_sh) -> tuple[Any, Any]:
    sqrt_sf = jnp.asarray(sqrt_sf)
    sqrt_sh = jnp.asarray(sqrt_sh)
    ns = int(sqrt_sf.shape[0])
    if ns < 2:
        z = jnp.zeros((0,), dtype=sqrt_sf.dtype)
        return z, z
    denom_outer = jnp.where(sqrt_sf[1:] != 0.0, sqrt_sf[1:], 1.0)
    sm = sqrt_sh / denom_outer
    denom_inner = jnp.where(sqrt_sf[:-1] != 0.0, sqrt_sf[:-1], 1.0)
    sp = sqrt_sh / denom_inner
    sp = sp.at[0].set(sm[0])
    return sm, sp


def _wint_from_config(*, cfg, dtype) -> Any:
    ntheta = int(cfg.ntheta)
    nzeta = int(cfg.nzeta)
    lasym = bool(cfg.lasym)
    ntheta_even = 2 * (ntheta // 2)
    ntheta_reduced = ntheta_even // 2 + 1
    ntheta_eff = ntheta_even if lasym else ntheta_reduced
    if ntheta_eff <= 0:
        return jnp.zeros((0,), dtype=dtype)
    if lasym:
        dnorm3 = 1.0 / (float(nzeta) * float(ntheta_even))
    else:
        # Reduced theta grid integrates the full even-theta domain via doubled interior weights.
        dnorm3 = 1.0 / (float(nzeta) * float(ntheta_reduced - 1))
    w_int = jnp.full((ntheta_eff,), dnorm3, dtype=dtype)
    if (not lasym) and ntheta_eff > 0:
        w_int = w_int.at[0].multiply(0.5)
        w_int = w_int.at[-1].multiply(0.5)
    return w_int


def lambda_preconditioner(
    *,
    bc,
    trig,
    s,
    cfg,
    damping_factor: float = 2.0,
    return_faclam: bool = False,
    return_debug: bool = False,
) -> Any:
    """Compute the VMEC lambda preconditioner (n>=0 storage) in JAX.

    Parameters
    ----------
    bc:
        :class:`~vmec_jax.vmec_bcovar.VmecHalfMeshBcovar` (or compatible) providing
        ``guu/guv/gvv`` and ``jac.sqrtg`` on the *half mesh* and scalar ``lamscale``.
    trig:
        Unused placeholder to keep parity-call signatures consistent with the
        NumPy helper module.
    s:
        Full-mesh radial coordinate array.
    cfg:
        VMEC configuration (mpol/ntor/nfp/ntheta/nzeta/lasym/lthreed).
    damping_factor:
        Damping used in the diagonal lambda preconditioner.
    """
    r0scale = float(getattr(trig, "r0scale", 1.0)) if trig is not None else 1.0
    s = jnp.asarray(s)
    dtype = s.dtype

    guu = jnp.asarray(bc.guu, dtype=dtype)
    guv = jnp.asarray(bc.guv, dtype=dtype)
    gvv = jnp.asarray(bc.gvv, dtype=dtype)
    gsqrt = jnp.asarray(bc.jac.sqrtg, dtype=dtype)

    ns_full = int(guu.shape[0])
    mpol = int(cfg.mpol)
    nrange = int(cfg.ntor) + 1
    if ns_full < 2:
        out = jnp.zeros((ns_full, mpol, nrange), dtype=dtype)
        if return_debug:
            debug = {
                "blam_pre": jnp.zeros((ns_full,), dtype=dtype),
                "clam_pre": jnp.zeros((ns_full,), dtype=dtype),
                "dlam_pre": jnp.zeros((ns_full,), dtype=dtype),
                "blam_post": jnp.zeros((ns_full,), dtype=dtype),
                "clam_post": jnp.zeros((ns_full,), dtype=dtype),
                "dlam_post": jnp.zeros((ns_full,), dtype=dtype),
            }
            if return_faclam:
                return out, jnp.zeros_like(out), debug
            return out, debug
        return (out, jnp.zeros_like(out)) if return_faclam else out

    w_int = _wint_from_config(cfg=cfg, dtype=dtype)

    w3 = w_int[None, :, None]
    gsqrt_safe = jnp.where(gsqrt != 0.0, gsqrt, 1.0)
    b_pre = jnp.sum(guu / gsqrt_safe * w3, axis=(1, 2))
    c_pre = jnp.sum(gvv / gsqrt_safe * w3, axis=(1, 2))
    if bool(cfg.lthreed):
        d_pre = jnp.sum(guv / gsqrt_safe * w3, axis=(1, 2))
    else:
        d_pre = jnp.zeros_like(b_pre)

    if ns_full >= 2:
        b_pre = b_pre.at[0].set(b_pre[1])
        c_pre = c_pre.at[0].set(c_pre[1])
        d_pre = d_pre.at[0].set(d_pre[1])

    b_post = b_pre
    c_post = c_pre
    d_post = d_pre
    if ns_full >= 2:
        b_next = jnp.concatenate([b_pre[2:], jnp.zeros((1,), dtype=dtype)])
        c_next = jnp.concatenate([c_pre[2:], jnp.zeros((1,), dtype=dtype)])
        d_next = jnp.concatenate([d_pre[2:], jnp.zeros((1,), dtype=dtype)])
        b_post = b_post.at[1:].set(0.5 * (b_pre[1:] + b_next))
        c_post = c_post.at[1:].set(0.5 * (c_pre[1:] + c_next))
        d_post = d_post.at[1:].set(0.5 * (d_pre[1:] + d_next))

    blam_pre = b_pre
    clam_pre = c_pre
    dlam_pre = d_pre
    blam_post = b_post
    clam_post = c_post
    dlam_post = d_post

    lamscale = jnp.asarray(bc.lamscale, dtype=dtype)
    p_factor = jnp.asarray(float(damping_factor), dtype=dtype) / (
        4.0 * (float(r0scale) ** 2) * lamscale * lamscale
    )

    sqrt_sf, _sqrt_sh = _sqrt_profiles_from_ns(ns_full, dtype=dtype)
    sqrt_sf = sqrt_sf.at[-1].set(1.0)

    m = jnp.arange(mpol, dtype=dtype)
    n = jnp.arange(nrange, dtype=dtype)
    nfp = float(cfg.nfp)
    tnn = (n * nfp) ** 2
    tmm = m * m
    tmn = 2.0 * m[:, None] * n[None, :] * nfp
    pwr = jnp.minimum(tmm / (16.0 * 16.0), 8.0)

    bF = b_post[:, None, None]
    cF = c_post[:, None, None]
    dF = d_post[:, None, None]
    faclam_raw = (tnn[None, None, :] * bF) + (tmn[None, :, :] * jnp.copysign(dF, bF)) + (tmm[None, :, None] * cF)
    faclam_raw = jnp.where(faclam_raw == 0.0, -1.0e-10, faclam_raw)

    sqrt_pow = (sqrt_sf[:, None, None]) ** (pwr[None, :, None])
    lam_prec = p_factor * sqrt_pow / faclam_raw

    # VMEC special-case m=n=0 preconditioner (chip/iota channel).
    b_safe = jnp.where(b_post != 0.0, b_post, jnp.asarray(-1.0e-10, dtype=dtype))
    p_factor00 = p_factor * (lamscale * lamscale)
    lam_prec = lam_prec.at[:, 0, 0].set(p_factor00 / b_safe)

    # VMEC jlam(m)=2 => js=1 (0-based index 0) is zero for all m,n except (0,0).
    if ns_full > 0:
        axis_mask = jnp.zeros((mpol, nrange), dtype=dtype).at[0, 0].set(1.0)
        lam_prec = lam_prec.at[0].set(lam_prec[0] * axis_mask)

    if return_debug:
        debug = {
            "blam_pre": blam_pre,
            "clam_pre": clam_pre,
            "dlam_pre": dlam_pre,
            "blam_post": blam_post,
            "clam_post": clam_post,
            "dlam_post": dlam_post,
        }
        if return_faclam:
            # VMEC dumps faclam as the preconditioner (not the raw denominator).
            return lam_prec, lam_prec, debug
        return lam_prec, debug
    if return_faclam:
        # VMEC dumps faclam as the preconditioner (not the raw denominator).
        return lam_prec, lam_prec
    return lam_prec


def _compute_preconditioning_matrix(
    *,
    xs,
    xu12,
    xu_e,
    xu_o,
    x1_o,
    r12,
    total_pressure,
    tau,
    bsupv,
    sqrtg,
    w_int,
    sqrt_sh,
    sm,
    sp,
    delta_s: Any,
    ns_full: int | None = None,
) -> tuple[Any, Any, Any, Any, Any]:
    """JAX port of VMEC `precondn` matrix assembly (axisymmetric subset)."""
    xs = jnp.asarray(xs)
    xu12 = jnp.asarray(xu12)
    xu_e = jnp.asarray(xu_e)
    xu_o = jnp.asarray(xu_o)
    x1_o = jnp.asarray(x1_o)
    r12 = jnp.asarray(r12)
    total_pressure = jnp.asarray(total_pressure)
    tau = jnp.asarray(tau)
    bsupv = jnp.asarray(bsupv)
    sqrtg = jnp.asarray(sqrtg)
    w_int = jnp.asarray(w_int)
    sqrt_sh = jnp.asarray(sqrt_sh)
    sm = jnp.asarray(sm)
    sp = jnp.asarray(sp)

    ns_half = int(xs.shape[0])
    if ns_half <= 0:
        z2 = jnp.zeros((0, 2), dtype=xs.dtype)
        z1 = jnp.zeros((0,), dtype=xs.dtype)
        return z2, z2, z2, z2, z1
    ns_full = int(ns_full) if ns_full is not None else ns_half

    pfactor = jnp.asarray(-4.0, dtype=xs.dtype)
    delta_s = jnp.asarray(delta_s, dtype=xs.dtype)
    tau_safe = jnp.where(tau != 0.0, tau, 1.0)
    sqrt_sh_safe = jnp.where(sqrt_sh != 0.0, sqrt_sh, 1.0)

    # Broadcast helpers.
    sh = sqrt_sh_safe[:, None, None]
    w3 = w_int[None, :, None]

    p_tau = pfactor * r12 * total_pressure / tau_safe * w3
    t1a = xu12 / delta_s
    xu_e_o = xu_e[1 : ns_half + 1]
    xu_e_i = xu_e[:ns_half]
    xu_o_o = xu_o[1 : ns_half + 1]
    xu_o_i = xu_o[:ns_half]
    t2a = 0.25 * (xu_e_o / sh + xu_o_o) / sh
    t3a = 0.25 * (xu_e_i / sh + xu_o_i) / sh

    ax0 = jnp.sum(p_tau * (t1a * t1a), axis=(1, 2))
    ax1 = jnp.sum(p_tau * (t1a + t2a) * (-t1a + t3a), axis=(1, 2))
    ax2 = jnp.sum(p_tau * (t1a + t2a) * (t1a + t2a), axis=(1, 2))
    ax3 = jnp.sum(p_tau * (-t1a + t3a) * (-t1a + t3a), axis=(1, 2))
    ax = jnp.stack([ax0, ax1, ax2, ax3], axis=1)

    x1_o_o = x1_o[1 : ns_half + 1]
    x1_o_i = x1_o[:ns_half]
    t1b = 0.5 * (xs + 0.5 / sh * x1_o_o)
    t2b = 0.5 * (xs + 0.5 / sh * x1_o_i)
    bx0 = jnp.sum(p_tau * t1b * t2b, axis=(1, 2))
    bx1 = jnp.sum(p_tau * t1b * t1b, axis=(1, 2))
    bx2 = jnp.sum(p_tau * t2b * t2b, axis=(1, 2))
    bx = jnp.stack([bx0, bx1, bx2], axis=1)

    cx = jnp.sum(0.25 * pfactor * (bsupv * bsupv) * sqrtg * w3, axis=(1, 2))

    axm0 = -ax[:, 0]
    axm1 = ax[:, 1] * sm * sp
    bxm0 = bx[:, 0]
    bxm1 = bx[:, 0] * sm * sp
    axm = jnp.stack([axm0, axm1], axis=1)
    bxm = jnp.stack([bxm0, bxm1], axis=1)

    # Full-grid accumulation.
    z = jnp.zeros((1,), dtype=xs.dtype)
    ax0_inner = jnp.concatenate([z, ax[:-1, 0]], axis=0)[:ns_full]
    ax1_inner = jnp.concatenate([z, ax[:-1, 2] * (sm[:-1] * sm[:-1])], axis=0)[:ns_full]
    bx0_inner = jnp.concatenate([z, bx[:-1, 1]], axis=0)[:ns_full]
    bx1_inner = jnp.concatenate([z, bx[:-1, 1] * (sm[:-1] * sm[:-1])], axis=0)[:ns_full]
    cx_inner = jnp.concatenate([z, cx[:-1]], axis=0)[:ns_full]

    pad_len = max(ns_full - ns_half, 0)
    zp = jnp.zeros((pad_len,), dtype=xs.dtype)
    ax0_outer = jnp.concatenate([ax[:, 0], zp], axis=0)[:ns_full]
    ax1_outer = jnp.concatenate([ax[:, 3] * (sp * sp), zp], axis=0)[:ns_full]
    bx0_outer = jnp.concatenate([bx[:, 2], zp], axis=0)[:ns_full]
    bx1_outer = jnp.concatenate([bx[:, 2] * (sp * sp), zp], axis=0)[:ns_full]
    cx_outer = jnp.concatenate([cx, zp], axis=0)[:ns_full]

    axd0 = ax0_inner + ax0_outer
    axd1 = ax1_inner + ax1_outer
    bxd0 = bx0_inner + bx0_outer
    bxd1 = bx1_inner + bx1_outer
    cxd = cx_inner + cx_outer
    axd = jnp.stack([axd0, axd1], axis=1)
    bxd = jnp.stack([bxd0, bxd1], axis=1)
    return axm, axd, bxm, bxd, cxd


def rz_preconditioner_matrices(
    *,
    bc,
    k,
    trig,
    s,
    cfg,
) -> tuple[dict[str, Any], Any, int]:
    """Return VMEC R/Z radial preconditioner matrices (JAX, fixed-boundary)."""
    del trig
    if bool(cfg.lasym):
        raise ValueError("rz_preconditioner_matrices does not yet support lasym.")
    s = jnp.asarray(s)
    ns = int(s.shape[0])
    ns_f = max(ns - 1, 1)
    dtype = jnp.asarray(bc.guu).dtype
    w_int = _wint_from_config(cfg=cfg, dtype=dtype)

    r12 = jnp.asarray(bc.jac.r12, dtype=dtype)[1:]
    tau = jnp.asarray(bc.jac.tau, dtype=dtype)[1:]
    total_pressure = jnp.asarray(bc.bsq, dtype=dtype)[1:]
    bsupv = jnp.asarray(bc.bsupv, dtype=dtype)[1:]
    sqrtg = jnp.asarray(bc.jac.sqrtg, dtype=dtype)[1:]

    sqrt_sf, sqrt_sh = _sqrt_profiles_from_ns(ns, dtype=dtype)
    sm, sp = _sm_sp_from_profiles(sqrt_sf, sqrt_sh)
    delta_s = jnp.where(ns >= 2, s[1] - s[0], jnp.asarray(1.0, dtype=dtype))

    arm, ard, brm, brd, cxd = _compute_preconditioning_matrix(
        xs=jnp.asarray(bc.jac.zs, dtype=dtype)[1:],
        xu12=jnp.asarray(bc.jac.zu12, dtype=dtype)[1:],
        xu_e=jnp.asarray(k.pzu_even, dtype=dtype),
        xu_o=jnp.asarray(k.pzu_odd, dtype=dtype),
        x1_o=jnp.asarray(k.pz1_odd, dtype=dtype),
        r12=r12,
        total_pressure=total_pressure,
        tau=tau,
        bsupv=bsupv,
        sqrtg=sqrtg,
        w_int=w_int,
        sqrt_sh=sqrt_sh,
        sm=sm,
        sp=sp,
        delta_s=delta_s,
        ns_full=ns_f,
    )
    azm, azd, bzm, bzd, _cxd_z = _compute_preconditioning_matrix(
        xs=jnp.asarray(bc.jac.rs, dtype=dtype)[1:],
        xu12=jnp.asarray(bc.jac.ru12, dtype=dtype)[1:],
        xu_e=jnp.asarray(k.pru_even, dtype=dtype),
        xu_o=jnp.asarray(k.pru_odd, dtype=dtype),
        x1_o=jnp.asarray(k.pr1_odd, dtype=dtype),
        r12=r12,
        total_pressure=total_pressure,
        tau=tau,
        bsupv=bsupv,
        sqrtg=sqrtg,
        w_int=w_int,
        sqrt_sh=sqrt_sh,
        sm=sm,
        sp=sp,
        delta_s=delta_s,
        ns_full=ns_f,
    )

    mpol = int(cfg.mpol)
    nrange = int(cfg.ntor) + 1
    nfp = float(cfg.nfp)
    m = jnp.arange(mpol, dtype=dtype)
    n = jnp.arange(nrange, dtype=dtype)
    m2 = (m * m)[None, :, None]
    n2 = ((n * nfp) ** 2)[None, None, :]
    m_par = (jnp.arange(mpol) % 2).astype(jnp.int32)
    arm_m = arm[:, m_par]
    brm_m = brm[:, m_par]
    ard_m = ard[:, m_par]
    brd_m = brd[:, m_par]
    azm_m = azm[:, m_par]
    bzm_m = bzm[:, m_par]
    azd_m = azd[:, m_par]
    bzd_m = bzd[:, m_par]

    ar = -(arm_m[:, :, None] + brm_m[:, :, None] * m2)
    az = -(azm_m[:, :, None] + bzm_m[:, :, None] * m2)
    dr = -(ard_m[:, :, None] + brd_m[:, :, None] * m2 + cxd[:, None, None] * n2)
    dz = -(azd_m[:, :, None] + bzd_m[:, :, None] * m2 + cxd[:, None, None] * n2)

    br = jnp.zeros_like(ar)
    bz = jnp.zeros_like(az)
    if ns_f > 1:
        br = br.at[1:].set(-(arm_m[:-1, :, None] + brm_m[:-1, :, None] * m2))
        bz = bz.at[1:].set(-(azm_m[:-1, :, None] + bzm_m[:-1, :, None] * m2))

    # Set matrices to 0 for jf < jmin(m,n).
    if ns_f > 0 and mpol > 1:
        ar = ar.at[0, 1:, :].set(0.0)
        az = az.at[0, 1:, :].set(0.0)
        dr = dr.at[0, 1:, :].set(0.0)
        dz = dz.at[0, 1:, :].set(0.0)

    if ns_f > 1 and mpol > 1:
        dr = dr.at[1, 1, :].add(br[1, 1, :])
        dz = dz.at[1, 1, :].add(bz[1, 1, :])

    jmin_m = jnp.where(jnp.arange(mpol) > 0, 1, 0).astype(jnp.int32)
    jmin = jmin_m[:, None] * jnp.ones((mpol, nrange), dtype=jnp.int32)

    mats = {"ar": ar, "br": br, "dr": dr, "az": az, "bz": bz, "dz": dz}
    return mats, jmin, int(ns_f)


def _tridi_solve_batched_jmin0(a, d, b, rhs) -> Any:
    """Batched Thomas solve for a/d/b with jmin=0 (axisymmetric)."""
    a = jnp.asarray(a)
    d = jnp.asarray(d)
    b = jnp.asarray(b)
    rhs = jnp.asarray(rhs)
    n = int(rhs.shape[0])
    if n == 0:
        return rhs
    eps = jnp.asarray(1.0e-12, dtype=rhs.dtype)
    d0 = jnp.where(d[0] != 0.0, d[0], eps)
    a0 = a[0] / d0
    x0 = rhs[0] / d0

    def fwd(carry, inp):
        a_prev, x_prev = carry
        aj, dj, bj, rj = inp
        denom = dj - a_prev * bj
        denom = jnp.where(denom != 0.0, denom, eps)
        a_new = aj / denom
        x_new = (rj - x_prev * bj) / denom
        return (a_new, x_new), (a_new, x_new)

    if n == 1:
        a_norm = a0[None, ...]
        x = x0[None, ...]
    else:
        inp = (a[1:], d[1:], b[1:], rhs[1:])
        (_, _), (a_rest, x_rest) = jax.lax.scan(fwd, (a0, x0), inp)
        a_norm = jnp.concatenate([a0[None, ...], a_rest], axis=0)
        x = jnp.concatenate([x0[None, ...], x_rest], axis=0)

    def bwd(carry, inp):
        x_next = carry
        a_j, x_j = inp
        x_new = x_j - a_j * x_next
        return x_new, x_new

    if n <= 1:
        return x
    # Backward substitution for indices n-2..0 (a_norm[n-1] unused).
    x_last = x[-1]
    inp_b = (a_norm[:-1], x[:-1])
    _, x_rev = jax.lax.scan(bwd, x_last, inp_b, reverse=True)
    x_out = jnp.concatenate([x_rev, x_last[None, ...]], axis=0)
    return x_out


def rz_preconditioner_apply(
    *,
    frzl_in: TomnspsRZL,
    mats: dict[str, Any],
    jmax: int,
    cfg,
) -> TomnspsRZL:
    """Apply cached VMEC R/Z preconditioner matrices.

    This is the matrix-application half of :func:`rz_preconditioner`,
    split out so callers (e.g. VMEC2000-style cached preconditioners) can
    reuse matrices across iterations without recomputing them.
    """
    if bool(cfg.lasym):
        return frzl_in

    ar = mats["ar"]
    br = mats["br"]
    dr = mats["dr"]
    az = mats["az"]
    bz = mats["bz"]
    dz = mats["dz"]

    mpol = int(cfg.mpol)
    nrange = int(cfg.ntor) + 1

    frcc = jnp.asarray(frzl_in.frcc)
    frss = None if frzl_in.frss is None else jnp.asarray(frzl_in.frss)
    fzsc = jnp.asarray(frzl_in.fzsc)
    fzcs = None if frzl_in.fzcs is None else jnp.asarray(frzl_in.fzcs)
    frsc = None if getattr(frzl_in, "frsc", None) is None else jnp.asarray(getattr(frzl_in, "frsc"))
    fzcc = None if getattr(frzl_in, "fzcc", None) is None else jnp.asarray(getattr(frzl_in, "fzcc"))
    frcc_u = frcc
    frss_u = frss
    fzsc_u = fzsc
    fzcs_u = fzcs
    frsc_u = frsc
    fzcc_u = fzcc
    jmax = int(jmax)
    if jmax > 0:
        rhs_r = frcc[:jmax]
        rhs_z = fzsc[:jmax]
        rhs_rs = frss[:jmax] if frss is not None else None
        rhs_zc = fzcs[:jmax] if fzcs is not None else None
        rhs_rsc = frsc[:jmax] if frsc is not None else None
        rhs_zcc = fzcc[:jmax] if fzcc is not None else None

        # m=0 group (jmin=0)
        sol_r0 = _tridi_solve_batched_jmin0(ar[:, 0, :], dr[:, 0, :], br[:, 0, :], rhs_r[:, 0, :])
        sol_z0 = _tridi_solve_batched_jmin0(az[:, 0, :], dz[:, 0, :], bz[:, 0, :], rhs_z[:, 0, :])
        frcc_u = frcc_u.at[:jmax, 0, :].set(sol_r0)
        fzsc_u = fzsc_u.at[:jmax, 0, :].set(sol_z0)
        if rhs_rs is not None and frss_u is not None:
            sol_rs0 = _tridi_solve_batched_jmin0(ar[:, 0, :], dr[:, 0, :], br[:, 0, :], rhs_rs[:, 0, :])
            frss_u = frss_u.at[:jmax, 0, :].set(sol_rs0)
        if rhs_zc is not None and fzcs_u is not None:
            sol_zc0 = _tridi_solve_batched_jmin0(az[:, 0, :], dz[:, 0, :], bz[:, 0, :], rhs_zc[:, 0, :])
            fzcs_u = fzcs_u.at[:jmax, 0, :].set(sol_zc0)
        if rhs_rsc is not None and frsc_u is not None:
            sol_rsc0 = _tridi_solve_batched_jmin0(ar[:, 0, :], dr[:, 0, :], br[:, 0, :], rhs_rsc[:, 0, :])
            frsc_u = frsc_u.at[:jmax, 0, :].set(sol_rsc0)
        if rhs_zcc is not None and fzcc_u is not None:
            sol_zcc0 = _tridi_solve_batched_jmin0(az[:, 0, :], dz[:, 0, :], bz[:, 0, :], rhs_zcc[:, 0, :])
            fzcc_u = fzcc_u.at[:jmax, 0, :].set(sol_zcc0)

        # m>0 group (jmin=1)
        if mpol > 1 and jmax > 1:
            for m in range(1, mpol):
                sol_rm = _tridi_solve_batched_jmin0(
                    ar[1:, m, :],
                    dr[1:, m, :],
                    br[1:, m, :],
                    rhs_r[1:, m, :],
                )
                sol_zm = _tridi_solve_batched_jmin0(
                    az[1:, m, :],
                    dz[1:, m, :],
                    bz[1:, m, :],
                    rhs_z[1:, m, :],
                )
                frcc_u = frcc_u.at[:jmax, m, :].set(
                    jnp.concatenate([jnp.zeros((1, nrange), dtype=frcc.dtype), sol_rm], axis=0)
                )
                fzsc_u = fzsc_u.at[:jmax, m, :].set(
                    jnp.concatenate([jnp.zeros((1, nrange), dtype=fzsc.dtype), sol_zm], axis=0)
                )
                if rhs_rs is not None and frss_u is not None:
                    sol_rsm = _tridi_solve_batched_jmin0(
                        ar[1:, m, :],
                        dr[1:, m, :],
                        br[1:, m, :],
                        rhs_rs[1:, m, :],
                    )
                    frss_u = frss_u.at[:jmax, m, :].set(
                        jnp.concatenate([jnp.zeros((1, nrange), dtype=frss.dtype), sol_rsm], axis=0)
                    )
                if rhs_zc is not None and fzcs_u is not None:
                    sol_zcm = _tridi_solve_batched_jmin0(
                        az[1:, m, :],
                        dz[1:, m, :],
                        bz[1:, m, :],
                        rhs_zc[1:, m, :],
                    )
                    fzcs_u = fzcs_u.at[:jmax, m, :].set(
                        jnp.concatenate([jnp.zeros((1, nrange), dtype=fzcs.dtype), sol_zcm], axis=0)
                    )
                if rhs_rsc is not None and frsc_u is not None:
                    sol_rscm = _tridi_solve_batched_jmin0(
                        ar[1:, m, :],
                        dr[1:, m, :],
                        br[1:, m, :],
                        rhs_rsc[1:, m, :],
                    )
                    frsc_u = frsc_u.at[:jmax, m, :].set(
                        jnp.concatenate([jnp.zeros((1, nrange), dtype=frsc.dtype), sol_rscm], axis=0)
                    )
                if rhs_zcc is not None and fzcc_u is not None:
                    sol_zccm = _tridi_solve_batched_jmin0(
                        az[1:, m, :],
                        dz[1:, m, :],
                        bz[1:, m, :],
                        rhs_zcc[1:, m, :],
                    )
                    fzcc_u = fzcc_u.at[:jmax, m, :].set(
                        jnp.concatenate([jnp.zeros((1, nrange), dtype=fzcc.dtype), sol_zccm], axis=0)
                    )

    return TomnspsRZL(
        frcc=frcc_u,
        frss=frss_u,
        fzsc=fzsc_u,
        fzcs=fzcs_u,
        flsc=frzl_in.flsc,
        flcs=frzl_in.flcs,
        frsc=frsc_u,
        frcs=getattr(frzl_in, "frcs", None),
        fzcc=fzcc_u,
        fzss=getattr(frzl_in, "fzss", None),
        flcc=getattr(frzl_in, "flcc", None),
        flss=getattr(frzl_in, "flss", None),
    )


def rz_preconditioner(
    *,
    frzl_in: TomnspsRZL,
    bc,
    k,
    trig,
    s,
    cfg,
) -> TomnspsRZL:
    """Apply the VMEC R/Z radial preconditioner in JAX."""
    if bool(cfg.lasym):
        return frzl_in
    mats, _jmin, jmax = rz_preconditioner_matrices(bc=bc, k=k, trig=trig, s=s, cfg=cfg)
    return rz_preconditioner_apply(frzl_in=frzl_in, mats=mats, jmax=jmax, cfg=cfg)
