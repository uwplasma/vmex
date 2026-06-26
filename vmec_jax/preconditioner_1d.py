"""VMEC2000 1D (radial) preconditioner helpers (axisymmetric path)."""

from __future__ import annotations


import numpy as np

from .kernels.tomnsp import TomnspsRZL


def _sqrt_profiles_from_ns(ns: int) -> tuple[np.ndarray, np.ndarray]:
    ns = int(ns)
    if ns <= 0:
        return np.zeros((0,), dtype=float), np.zeros((0,), dtype=float)
    if ns == 1:
        return np.zeros((1,), dtype=float), np.zeros((0,), dtype=float)
    denom = float(ns - 1)
    full_pos = np.linspace(0.0, 1.0, ns, dtype=float)
    sqrt_sf = np.sqrt(np.maximum(full_pos, 0.0))
    half_pos = (np.arange(ns - 1, dtype=float) + 0.5) / denom
    sqrt_sh = np.sqrt(np.maximum(half_pos, 0.0))
    return sqrt_sf, sqrt_sh


def _sqrt_profiles_from_s(s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(s, dtype=float)
    return _sqrt_profiles_from_ns(int(s.shape[0]))


def _sm_sp_from_profiles(
    sqrt_sf: np.ndarray, sqrt_sh: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    sqrt_sf = np.asarray(sqrt_sf, dtype=float)
    sqrt_sh = np.asarray(sqrt_sh, dtype=float)
    ns = int(sqrt_sf.shape[0])
    if ns < 2:
        return np.zeros((0,), dtype=float), np.zeros((0,), dtype=float)
    sm = np.zeros((ns - 1,), dtype=float)
    sp = np.zeros((ns - 1,), dtype=float)
    for jh in range(ns - 1):
        jfi = jh
        jfo = jh + 1
        denom_outer = sqrt_sf[jfo] if sqrt_sf[jfo] != 0.0 else 1.0
        sm[jh] = sqrt_sh[jh] / denom_outer
        if jh > 0:
            denom_inner = sqrt_sf[jfi] if sqrt_sf[jfi] != 0.0 else 1.0
            sp[jh] = sqrt_sh[jh] / denom_inner
    if ns > 1:
        sp[0] = sm[0]
    return sm, sp


def _sm_sp_from_s(s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(s, dtype=float)
    ns = int(s.shape[0])
    if ns < 2:
        return np.zeros((0,), dtype=float), np.zeros((0,), dtype=float)
    sqrt_sf, sqrt_sh = _sqrt_profiles_from_ns(ns)
    return _sm_sp_from_profiles(sqrt_sf, sqrt_sh)


def wint_from_config(*, cfg) -> np.ndarray:
    """Evaluate wint from config for VMEC-JAX numerical workflow."""
    ntheta = int(cfg.ntheta)
    nzeta = int(cfg.nzeta)
    lasym = bool(cfg.lasym)
    ntheta_even = 2 * (ntheta // 2)
    ntheta_reduced = ntheta_even // 2 + 1
    ntheta_eff = ntheta_even if lasym else ntheta_reduced
    if lasym:
        dnorm3 = 1.0 / (nzeta * ntheta_even)
    else:
        dnorm3 = 1.0 / (nzeta * (ntheta_reduced - 1))
    w_int = np.full((ntheta_eff,), dnorm3, dtype=float)
    if not lasym and ntheta_eff > 0:
        w_int[0] *= 0.5
        w_int[-1] *= 0.5
    return w_int


def lambda_preconditioner(
    *,
    bc,
    trig,
    s: np.ndarray,
    cfg,
    damping_factor: float = 2.0,
    return_faclam: bool = False,
    return_debug: bool = False,
    r0scale: float | None = None,
) -> np.ndarray:
    """Compute VMEC lambda preconditioner (n>=0 storage)."""
    if r0scale is None:
        r0scale = float(getattr(trig, "r0scale", 1.0)) if trig is not None else 1.0
    r0scale = float(r0scale)
    guu = np.asarray(bc.guu, dtype=float)
    guv = np.asarray(bc.guv, dtype=float)
    gvv = np.asarray(bc.gvv, dtype=float)
    gsqrt = np.asarray(bc.jac.sqrtg, dtype=float)
    ns_full = int(guu.shape[0])
    if ns_full < 2:
        out = np.zeros((ns_full, int(cfg.mpol), int(cfg.ntor) + 1), dtype=float)
        if return_debug:
            debug = {
                "blam_pre": np.zeros((ns_full,), dtype=float),
                "clam_pre": np.zeros((ns_full,), dtype=float),
                "dlam_pre": np.zeros((ns_full,), dtype=float),
                "blam_post": np.zeros((ns_full,), dtype=float),
                "clam_post": np.zeros((ns_full,), dtype=float),
                "dlam_post": np.zeros((ns_full,), dtype=float),
            }
            if return_faclam:
                return out, np.zeros_like(out), debug
            return out, debug
        return (out, np.zeros_like(out)) if return_faclam else out
    w_int = wint_from_config(cfg=cfg)

    gsqrt_safe = np.where(gsqrt != 0.0, gsqrt, 1.0)
    b_pre = np.sum(guu / gsqrt_safe * w_int[None, :, None], axis=(1, 2))
    c_pre = np.sum(gvv / gsqrt_safe * w_int[None, :, None], axis=(1, 2))
    if bool(cfg.lthreed):
        d_pre = np.sum(guv / gsqrt_safe * w_int[None, :, None], axis=(1, 2))
    else:
        d_pre = np.zeros_like(b_pre)

    if ns_full >= 2:
        b_pre[0] = b_pre[1]
        c_pre[0] = c_pre[1]
        d_pre[0] = d_pre[1]

    b_post = b_pre.copy()
    c_post = c_pre.copy()
    d_post = d_pre.copy()
    if ns_full >= 2:
        b_next = np.concatenate([b_pre[2:], np.zeros((1,), dtype=float)])
        c_next = np.concatenate([c_pre[2:], np.zeros((1,), dtype=float)])
        d_next = np.concatenate([d_pre[2:], np.zeros((1,), dtype=float)])
        b_post[1:] = 0.5 * (b_pre[1:] + b_next)
        c_post[1:] = 0.5 * (c_pre[1:] + c_next)
        d_post[1:] = 0.5 * (d_pre[1:] + d_next)

    blam_pre = b_pre.copy()
    clam_pre = c_pre.copy()
    dlam_pre = d_pre.copy()
    blam_post = b_post.copy()
    clam_post = c_post.copy()
    dlam_post = d_post.copy()

    mpol = int(cfg.mpol)
    nrange = int(cfg.ntor) + 1
    p_factor = float(damping_factor) / (
        4.0 * (float(r0scale) ** 2) * float(bc.lamscale) * float(bc.lamscale)
    )
    sqrt_sf, _ = _sqrt_profiles_from_s(s)
    if sqrt_sf.size > 0:
        sqrt_sf[-1] = 1.0

    lam_prec = np.zeros((ns_full, mpol, nrange), dtype=float)
    faclam_out = np.zeros_like(lam_prec) if return_faclam else None
    for jf in range(1, ns_full):
        for n in range(nrange):
            tnn = (n * cfg.nfp) ** 2
            for m in range(mpol):
                tmm = m * m
                pwr = min(tmm / (16.0 * 16.0), 8.0)
                tmn = 2.0 * m * n * cfg.nfp
                faclam = tnn * b_post[jf] + tmn * np.copysign(d_post[jf], b_post[jf]) + tmm * c_post[jf]
                if faclam == 0.0:
                    faclam = -1.0e-10
                lam_prec[jf, m, n] = p_factor / faclam * (sqrt_sf[jf] ** pwr)
    # VMEC special-case m=n=0 preconditioner (chip/iota channel).
    b_safe = np.where(b_post != 0.0, b_post, -1.0e-10)
    p_factor00 = p_factor * (float(bc.lamscale) ** 2)
    lam_prec[:, 0, 0] = p_factor00 / b_safe

    # VMEC jlam(m)=2 => js=1 (0-based index 0) is zero for all m,n except (0,0).
    if ns_full > 0:
        axis_mask = np.zeros((mpol, nrange), dtype=float)
        axis_mask[0, 0] = 1.0
        lam_prec[0] = lam_prec[0] * axis_mask

    if faclam_out is not None:
        faclam_out[:] = lam_prec
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
            return lam_prec, faclam_out, debug
        return lam_prec, debug
    if return_faclam:
        return lam_prec, faclam_out
    return lam_prec


def _compute_preconditioning_matrix(
    *,
    xs: np.ndarray,
    xu12: np.ndarray,
    xu_e: np.ndarray,
    xu_o: np.ndarray,
    x1_o: np.ndarray,
    r12: np.ndarray,
    total_pressure: np.ndarray,
    tau: np.ndarray,
    bsupv: np.ndarray,
    sqrtg: np.ndarray,
    w_int: np.ndarray,
    sqrt_sh: np.ndarray,
    sm: np.ndarray,
    sp: np.ndarray,
    delta_s: float,
    ns_full: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ns_half = int(xs.shape[0])
    ntheta = int(xs.shape[1])
    nzeta = int(xs.shape[2])
    ns_full = int(ns_full) if ns_full is not None else ns_half
    if ns_half <= 0:
        return (
            np.zeros((0, 2), dtype=float),
            np.zeros((0, 2), dtype=float),
            np.zeros((0, 2), dtype=float),
            np.zeros((0, 2), dtype=float),
            np.zeros((0,), dtype=float),
        )
    ns_full_expected = ns_half + 1
    if xu_e.shape[0] < ns_full_expected:
        raise ValueError("xu_e must have ns_half+1 entries")
    if xu_o.shape[0] < ns_full_expected:
        raise ValueError("xu_o must have ns_half+1 entries")
    if x1_o.shape[0] < ns_full_expected:
        raise ValueError("x1_o must have ns_half+1 entries")
    if sqrt_sh.shape[0] < ns_half:
        raise ValueError("sqrt_sh must have ns_half entries")
    if sm.shape[0] < ns_half or sp.shape[0] < ns_half:
        raise ValueError("sm/sp must have ns_half entries")
    ax = np.zeros((ns_half, 4), dtype=float)
    bx = np.zeros((ns_half, 3), dtype=float)
    cx = np.zeros((ns_half,), dtype=float)
    pfactor = -4.0
    gsqrt_safe = np.where(sqrtg != 0.0, sqrtg, 1.0)
    sqrt_sh_safe = np.where(sqrt_sh != 0.0, sqrt_sh, 1.0)
    for jh in range(ns_half):
        for kl in range(ntheta * nzeta):
            l = kl % ntheta
            k = kl // ntheta
            p_tau = (
                pfactor
                * r12[jh, l, k]
                * r12[jh, l, k]
                * total_pressure[jh, l, k]
                / gsqrt_safe[jh, l, k]
                * w_int[l]
            )
            t1a = xu12[jh, l, k] / delta_s
            t2a = 0.25 * (xu_e[jh + 1, l, k] / sqrt_sh_safe[jh] + xu_o[jh + 1, l, k]) / sqrt_sh_safe[jh]
            t3a = 0.25 * (xu_e[jh, l, k] / sqrt_sh_safe[jh] + xu_o[jh, l, k]) / sqrt_sh_safe[jh]
            ax[jh, 0] += p_tau * t1a * t1a
            ax[jh, 1] += p_tau * (t1a + t2a) * (-t1a + t3a)
            ax[jh, 2] += p_tau * (t1a + t2a) * (t1a + t2a)
            ax[jh, 3] += p_tau * (-t1a + t3a) * (-t1a + t3a)
            t1b = 0.5 * (xs[jh, l, k] + 0.5 / sqrt_sh_safe[jh] * x1_o[jh + 1, l, k])
            t2b = 0.5 * (xs[jh, l, k] + 0.5 / sqrt_sh_safe[jh] * x1_o[jh, l, k])
            bx[jh, 0] += p_tau * t1b * t2b
            bx[jh, 1] += p_tau * t1b * t1b
            bx[jh, 2] += p_tau * t2b * t2b
            cx[jh] += 0.25 * pfactor * (bsupv[jh, l, k] ** 2) * sqrtg[jh, l, k] * w_int[l]
    axm = np.zeros((ns_half, 2), dtype=float)
    bxm = np.zeros((ns_half, 2), dtype=float)
    axm[:, 0] = -ax[:, 0]
    axm[:, 1] = ax[:, 1] * sm * sp
    bxm[:, 0] = bx[:, 0]
    bxm[:, 1] = bx[:, 0] * sm * sp
    axd = np.zeros((ns_full, 2), dtype=float)
    bxd = np.zeros((ns_full, 2), dtype=float)
    cxd = np.zeros((ns_full,), dtype=float)
    for jf in range(ns_full):
        jhi = jf - 1
        jho = jf
        if jf > 0:
            axd[jf, 0] += ax[jhi, 0]
            axd[jf, 1] += ax[jhi, 2] * sm[jhi] * sm[jhi]
            bxd[jf, 0] += bx[jhi, 1]
            bxd[jf, 1] += bx[jhi, 1] * sm[jhi] * sm[jhi]
            cxd[jf] += cx[jhi]
        if jf < ns_half:
            axd[jf, 0] += ax[jho, 0]
            axd[jf, 1] += ax[jho, 3] * sp[jho] * sp[jho]
            bxd[jf, 0] += bx[jho, 2]
            bxd[jf, 1] += bx[jho, 2] * sp[jho] * sp[jho]
            cxd[jf] += cx[jho]
    return axm, axd, bxm, bxd, cxd


def _tridiagonal_solve(a: np.ndarray, d: np.ndarray, b: np.ndarray, rhs: np.ndarray, jmin: int, jmax: int) -> np.ndarray:
    """Thomas solve with jmin/jmax bounds."""
    out = rhs.copy()
    n = int(rhs.shape[0])
    if jmax <= jmin or n == 0:
        return out
    jmin = max(0, int(jmin))
    jmax = min(int(jmax), n)
    a = a.copy()
    d = d.copy()
    b = b.copy()
    # zero out before jmin
    for j in range(0, jmin):
        a[j] = 0.0
        d[j] = 1.0
        b[j] = 0.0
        out[j] = 0.0
    if d[jmin] == 0.0:
        d[jmin] = 1.0e-12
    a[jmin] /= d[jmin]
    for j in range(jmin + 1, jmax - 1):
        denom = d[j] - a[j - 1] * b[j]
        if denom == 0.0:
            denom = 1.0e-12
        a[j] /= denom
    out[jmin] /= d[jmin]
    for j in range(jmin + 1, jmax):
        denom = d[j] - a[j - 1] * b[j]
        if denom == 0.0:
            denom = 1.0e-12
        out[j] = (out[j] - out[j - 1] * b[j]) / denom
    for j in range(jmax - 2, jmin - 1, -1):
        out[j] = out[j] - a[j] * out[j + 1]
    return out


def rz_preconditioner_matrices(
    *,
    bc,
    k,
    trig,
    s: np.ndarray,
    cfg,
    jmax_override: int | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, int]:
    """Return R/Z preconditioner matrices and jmin."""
    s_arr = np.asarray(s, dtype=float)
    ns = int(s_arr.shape[0])
    ns_f_default = max(ns - 1, 1)
    if jmax_override is None:
        ns_f = ns_f_default
    else:
        ns_f = int(max(1, min(int(jmax_override), ns)))
    w_int = wint_from_config(cfg=cfg)
    r12 = np.asarray(bc.jac.r12, dtype=float)[1:]
    tau = np.asarray(bc.jac.tau, dtype=float)[1:]
    total_pressure = np.asarray(bc.bsq, dtype=float)[1:]
    bsupv = np.asarray(bc.bsupv, dtype=float)[1:]
    sqrt_sf, sqrt_sh = _sqrt_profiles_from_s(s_arr)
    sm, sp = _sm_sp_from_profiles(sqrt_sf, sqrt_sh)
    delta_s = float(s_arr[1] - s_arr[0]) if ns >= 2 else 1.0

    arm, ard, brm, brd, cxd = _compute_preconditioning_matrix(
        xs=np.asarray(bc.jac.zs, dtype=float)[1:],
        xu12=np.asarray(bc.jac.zu12, dtype=float)[1:],
        xu_e=np.asarray(k.pzu_even, dtype=float),
        xu_o=np.asarray(k.pzu_odd, dtype=float),
        x1_o=np.asarray(k.pz1_odd, dtype=float),
        r12=r12,
        total_pressure=total_pressure,
        tau=tau,
        bsupv=bsupv,
        sqrtg=np.asarray(bc.jac.sqrtg, dtype=float)[1:],
        w_int=w_int,
        sqrt_sh=sqrt_sh,
        sm=sm,
        sp=sp,
        delta_s=delta_s,
        ns_full=ns_f,
    )
    azm, azd, bzm, bzd, _ = _compute_preconditioning_matrix(
        xs=np.asarray(bc.jac.rs, dtype=float)[1:],
        xu12=np.asarray(bc.jac.ru12, dtype=float)[1:],
        xu_e=np.asarray(k.pru_even, dtype=float),
        xu_o=np.asarray(k.pru_odd, dtype=float),
        x1_o=np.asarray(k.pr1_odd, dtype=float),
        r12=r12,
        total_pressure=total_pressure,
        tau=tau,
        bsupv=bsupv,
        sqrtg=np.asarray(bc.jac.sqrtg, dtype=float)[1:],
        w_int=w_int,
        sqrt_sh=sqrt_sh,
        sm=sm,
        sp=sp,
        delta_s=delta_s,
        ns_full=ns_f,
    )

    mpol = int(cfg.mpol)
    nrange = int(cfg.ntor) + 1
    ar = np.zeros((ns_f, mpol, nrange), dtype=float)
    br = np.zeros_like(ar)
    dr = np.zeros_like(ar)
    az = np.zeros_like(ar)
    bz = np.zeros_like(ar)
    dz = np.zeros_like(ar)
    if arm.shape[0] < ns_f:
        pad = ns_f - arm.shape[0]
        arm = np.concatenate([arm, np.zeros((pad, arm.shape[1]), dtype=arm.dtype)], axis=0)
        brm = np.concatenate([brm, np.zeros((pad, brm.shape[1]), dtype=brm.dtype)], axis=0)
        azm = np.concatenate([azm, np.zeros((pad, azm.shape[1]), dtype=azm.dtype)], axis=0)
        bzm = np.concatenate([bzm, np.zeros((pad, bzm.shape[1]), dtype=bzm.dtype)], axis=0)
    jmin = np.zeros((mpol, nrange), dtype=int)
    for m in range(mpol):
        jmin[m, :] = 1 if m > 0 else 0
        m_par = m % 2
        for n in range(nrange):
            for jf in range(ns_f):
                if jf < jmin[m, n]:
                    continue
                ar[jf, m, n] = -(arm[jf, m_par] + brm[jf, m_par] * m * m)
                az[jf, m, n] = -(azm[jf, m_par] + bzm[jf, m_par] * m * m)
                dr[jf, m, n] = -(ard[jf, m_par] + brd[jf, m_par] * m * m + cxd[jf] * (n * cfg.nfp) ** 2)
                dz[jf, m, n] = -(azd[jf, m_par] + bzd[jf, m_par] * m * m + cxd[jf] * (n * cfg.nfp) ** 2)
                if jf > 0:
                    br[jf, m, n] = -(arm[jf - 1, m_par] + brm[jf - 1, m_par] * m * m)
                    bz[jf, m, n] = -(azm[jf - 1, m_par] + bzm[jf - 1, m_par] * m * m)
                if jf == 1 and m == 1:
                    dr[jf, m, n] += br[jf, m, n]
                    dz[jf, m, n] += bz[jf, m, n]

    # VMEC scalfor free-boundary edge conditioning (scalfor.f):
    # when jmax==ns (vacuum active), strengthen edge diagonals and apply
    # the iflag=1 stabilization term to the Z-system m=n=0 mode.
    if ns_f >= ns and ns > 0:
        edge_idx = ns - 1
        edge_pedestal = 0.05
        fac = 0.25
        hs = float(delta_s)
        mult_fac = min(fac, fac * hs * 15.0)
        dr[edge_idx, 0:2, :] *= (1.0 + edge_pedestal)
        dz[edge_idx, 0:2, :] *= (1.0 + edge_pedestal)
        if mpol > 2:
            dr[edge_idx, 2:, :] *= (1.0 + 2.0 * edge_pedestal)
            dz[edge_idx, 2:, :] *= (1.0 + 2.0 * edge_pedestal)
        dz[edge_idx, 0, 0] *= (1.0 - mult_fac) / (1.0 + edge_pedestal)

    mats = {"ar": ar, "br": br, "dr": dr, "az": az, "bz": bz, "dz": dz}
    return mats, jmin, ns_f


def rz_preconditioner(
    *,
    frzl_in: TomnspsRZL,
    bc,
    k,
    trig,
    s: np.ndarray,
    cfg,
) -> TomnspsRZL:
    """Apply R/Z radial preconditioner (axisymmetric only)."""
    if bool(cfg.lthreed) or bool(cfg.lasym):
        return frzl_in
    mats, jmin, jmax = rz_preconditioner_matrices(
        bc=bc,
        k=k,
        trig=trig,
        s=s,
        cfg=cfg,
    )
    ar = mats["ar"]
    br = mats["br"]
    dr = mats["dr"]
    az = mats["az"]
    bz = mats["bz"]
    dz = mats["dz"]
    mpol = int(cfg.mpol)
    nrange = int(cfg.ntor) + 1

    frcc_u = np.array(frzl_in.frcc, dtype=float, copy=True)
    fzsc_u = np.array(frzl_in.fzsc, dtype=float, copy=True)
    for m in range(mpol):
        for n in range(nrange):
            jmin_mn = int(jmin[m, n])
            frcc_u[:jmax, m, n] = _tridiagonal_solve(
                ar[:, m, n], dr[:, m, n], br[:, m, n], frcc_u[:jmax, m, n], jmin_mn, jmax
            )
            fzsc_u[:jmax, m, n] = _tridiagonal_solve(
                az[:, m, n], dz[:, m, n], bz[:, m, n], fzsc_u[:jmax, m, n], jmin_mn, jmax
            )

    return TomnspsRZL(
        frcc=frcc_u,
        frss=frzl_in.frss,
        fzsc=fzsc_u,
        fzcs=frzl_in.fzcs,
        flsc=frzl_in.flsc,
        flcs=frzl_in.flcs,
        frsc=getattr(frzl_in, "frsc", None),
        frcs=getattr(frzl_in, "frcs", None),
        fzcc=getattr(frzl_in, "fzcc", None),
        fzss=getattr(frzl_in, "fzss", None),
        flcc=getattr(frzl_in, "flcc", None),
        flss=getattr(frzl_in, "flss", None),
    )


def rz_preconditioner_apply(
    *,
    frzl_in: TomnspsRZL,
    mats: dict[str, np.ndarray],
    jmax: int,
    cfg,
) -> TomnspsRZL:
    """Apply cached R/Z radial preconditioner matrices (NumPy, axisymmetric)."""
    if bool(cfg.lthreed) or bool(cfg.lasym):
        return frzl_in

    ar = np.asarray(mats["ar"], dtype=float)
    br = np.asarray(mats["br"], dtype=float)
    dr = np.asarray(mats["dr"], dtype=float)
    az = np.asarray(mats["az"], dtype=float)
    bz = np.asarray(mats["bz"], dtype=float)
    dz = np.asarray(mats["dz"], dtype=float)

    mpol = int(cfg.mpol)
    nrange = int(cfg.ntor) + 1
    jmax = min(max(0, int(jmax)), int(ar.shape[0]))

    frcc_u = np.array(frzl_in.frcc, dtype=float, copy=True)
    fzsc_u = np.array(frzl_in.fzsc, dtype=float, copy=True)

    for m in range(mpol):
        jmin_m = 1 if m > 0 else 0
        for n in range(nrange):
            frcc_u[:jmax, m, n] = _tridiagonal_solve(
                ar[:, m, n], dr[:, m, n], br[:, m, n], frcc_u[:jmax, m, n], jmin_m, jmax
            )
            fzsc_u[:jmax, m, n] = _tridiagonal_solve(
                az[:, m, n], dz[:, m, n], bz[:, m, n], fzsc_u[:jmax, m, n], jmin_m, jmax
            )

    return TomnspsRZL(
        frcc=frcc_u,
        frss=frzl_in.frss,
        fzsc=fzsc_u,
        fzcs=frzl_in.fzcs,
        flsc=frzl_in.flsc,
        flcs=frzl_in.flcs,
        frsc=getattr(frzl_in, "frsc", None),
        frcs=getattr(frzl_in, "frcs", None),
        fzcc=getattr(frzl_in, "fzcc", None),
        fzss=getattr(frzl_in, "fzss", None),
        flcc=getattr(frzl_in, "flcc", None),
        flss=getattr(frzl_in, "flss", None),
    )
