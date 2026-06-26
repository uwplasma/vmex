"""VMEC constraint pipeline pieces (`alias` / `gcon`) for parity work.

VMEC uses a "constraint force" to maintain polar regularity and enforce internal
relationships near the magnetic axis. In the VMEC2000 fixed-boundary pipeline:

1. `totzsps` produces real-space constraint-like arrays `rcon`, `zcon`
2. `funct3d` forms a scalar field

   ztemp = (rcon - rcon0)*ru0 + (zcon - zcon0)*zu0

3. `alias` computes `gcon` from `ztemp` via a de-aliased spectral operator
4. `forces` uses `gcon` to add constraint force kernels and to form the
   `arcon/azcon` arrays passed to `tomnsps`.

This module ports the *core* of VMEC's `alias.f` needed for parity diagnostics.

Notes
-----
The overall constraint strength depends on `tcon(js)`, which VMEC computes in
`bcovar.f` from preconditioner-related quantities. For parity work we provide a
`tcon_from_tcon0_heuristic` that matches VMEC's scaling structure but does not
yet reproduce the full `bcovar` computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .._compat import jnp, einsum
from .tomnsp import VmecTrigTables


@dataclass(frozen=True)
class VmecConstraintTables:
    faccon: Any  # (mpol,)  m index is 0..mpol-1 (mpol1 in VMEC)
    tcon: Any  # (ns,)


def faccon_from_signgs(*, mpol: int, signgs: int, dtype=jnp.float64) -> Any:
    """Compute VMEC's `faccon(m)` array (fixaray.f).

    VMEC (fixaray.f):
      faccon(0) = 0
      faccon(mpol1) = 0
      faccon(1:mpol1-1) = -0.25*signgs / xmpq(2:mpol1,1)^2

    where xmpq(m,1) = m*(m-1). Note the index shift by one.
    """
    mpol = int(mpol)
    if mpol <= 0:
        raise ValueError("mpol must be positive")
    signgs = int(signgs)

    m = np.arange(mpol, dtype=float)
    fac = np.zeros((mpol,), dtype=float)
    if mpol >= 3:
        denom = ((m[1:-1] + 1.0) * (m[1:-1])) ** 2  # xmpq(m+1,1)^2 for m=1..mpol-2
        fac[1:-1] = (-0.25 * float(signgs)) / denom
    return jnp.asarray(fac, dtype=dtype)


def tcon_from_tcon0_heuristic(*, tcon0: float, s, trig: VmecTrigTables, lasym: bool) -> Any:
    """Heuristic `tcon(js)` profile (placeholder for VMEC's bcovar-derived `tcon`).

    VMEC's actual computation depends on preconditioner quantities (`ard/azd`)
    and flux-surface norms of `ru0/zu0`. Until those kernels are ported, we use
    a conservative, VMEC-shaped scaling with a constant radial factor.

    Note: VMEC's `bcovar.f` previously halved `tcon` for `lasym`, but the
    current STELLOPT/VMEC2000 source leaves that line commented out. We mirror
    the *current* behavior and do not apply a lasym-specific halving here.
    """
    s = jnp.asarray(s)
    ns = int(s.shape[0])
    if ns < 2:
        return jnp.zeros_like(s, dtype=jnp.asarray(trig.cosmu).dtype)

    tcon0 = jnp.minimum(jnp.abs(jnp.asarray(tcon0, dtype=jnp.asarray(trig.cosmu).dtype)), 1.0)
    hs = jnp.asarray(s[1] - s[0], dtype=jnp.asarray(trig.cosmu).dtype)
    r0scale = float(trig.r0scale)

    # bcovar.f:
    #   tcon_mul = tcon0*(1 + ns*(1/60 + ns/(200*120)))
    #   tcon_mul = tcon_mul / (4*r0scale^2)^2
    #   tcon(js) = (...) * tcon_mul * (32*hs)^2
    tcon_mul = tcon0 * (1.0 + float(ns) * (1.0 / 60.0 + float(ns) / (200.0 * 120.0)))
    tcon_mul = tcon_mul / ((4.0 * (r0scale**2)) ** 2)
    tcon_val = jnp.asarray(tcon_mul, dtype=hs.dtype) * (32.0 * hs) ** 2

    tcon = jnp.full((ns,), tcon_val, dtype=hs.dtype)
    tcon = tcon.at[0].set(0.0)
    if ns >= 3:
        tcon = tcon.at[-1].set(0.5 * tcon[-2])
    return tcon.astype(jnp.asarray(trig.cosmu).dtype)


def tcon_from_bcovar_precondn_diag(
    *,
    tcon0: float,
    trig: VmecTrigTables,
    s,
    signgs: int,
    lasym: bool,
    bsq,
    r12,
    sqrtg,
    ru12,
    zu12,
    ru0,
    zu0,
) -> Any:
    """Compute VMEC-like `tcon(js)` using the diagonal `precondn` pieces from `bcovar.f`.

    This implements the `tcon(js)` block in `bcovar.f` for the common fixed-boundary
    path (no HBANGLE), but in a reduced form that computes only the quantities
    needed for the constraint scaling:

      tcon(js) = min(|ard(js,1)|/arnorm(js), |azd(js,1)|/aznorm(js)) * tcon_mul * (32*hs)^2

    where `ard(js,1)` and `azd(js,1)` are the (1,1) diagonal elements produced by
    `precondn` for the Z-like and R-like calls, and:

      arnorm(js) = sum(wint * ru0^2),  aznorm(js) = sum(wint * zu0^2).

    Notes
    -----
    - This does **not** attempt to reproduce the full 1D/2D preconditioner; it
      only reproduces the diagonal element used by `tcon`.
    - The expressions match `precondn.f` for the `axd(js,1)` contribution:
        ax(js,1) = sum(ptau * (xu12*ohs)^2)
        axd(js,1) = ax(js,1) + ax(js+1,1)
      with ptau = pfactor * r12^2 * bsq * wint / gsqrt.
    - VMEC's lasym-specific `tcon` halving is commented out in current
      STELLOPT/VMEC2000 sources, so we do not apply it here.
    """
    s = jnp.asarray(s)
    ns = int(s.shape[0])
    if ns < 2:
        return jnp.zeros((ns,), dtype=jnp.asarray(trig.cosmu).dtype)

    ard1, azd1 = precondn_diag_axd1_from_bcovar(
        trig=trig,
        s=s,
        bsq=bsq,
        r12=r12,
        sqrtg=sqrtg,
        ru12=ru12,
        zu12=zu12,
    )
    return tcon_from_cached_precondn_diag(
        tcon0=tcon0,
        trig=trig,
        s=s,
        lasym=lasym,
        ard1=ard1,
        azd1=azd1,
        ru0=ru0,
        zu0=zu0,
    )


def precondn_diag_axd1_from_bcovar(
    *,
    trig: VmecTrigTables,
    s,
    bsq,
    r12,
    sqrtg,
    ru12,
    zu12,
) -> tuple[Any, Any]:
    """Compute VMEC `precondn` diagonal outputs axd(:,1) for R/Z-like calls.

    Returns
    -------
    ard1, azd1:
        1D arrays (ns,) corresponding to VMEC's `ard(js,1)` and `azd(js,1)`.

    Notes
    -----
    In VMEC, `precondn(_par)` is only recomputed every `ns4` iterations
    (`vmec_params.f: ns4=25`). Between refreshes, `ard/azd` remain constant even
    though `bcovar` updates `tcon(js)` each iteration using the cached `ard/azd`
    and the *current* `ru0/zu0` norms. Parity drivers should cache the returned
    arrays accordingly.
    """
    s = jnp.asarray(s)
    ns = int(s.shape[0])
    if ns < 2:
        z = jnp.zeros((ns,), dtype=jnp.asarray(trig.cosmu).dtype)
        return z, z

    hs = jnp.asarray(s[1] - s[0], dtype=jnp.asarray(trig.cosmu).dtype)
    ohs = jnp.where(hs != 0, 1.0 / hs, jnp.asarray(0.0, dtype=hs.dtype))

    # VMEC precondn: pfactor = -4*r0scale^2 (v8.51+).
    pfactor = -4.0 * float(trig.r0scale) ** 2

    # Angular integration weights (wint) on the VMEC internal grid.
    wint3 = getattr(trig, "wint3_precond", None)
    if wint3 is None:
        w_theta = jnp.asarray(trig.cosmui3[:, 0]) / jnp.asarray(trig.mscale[0])
        wint = w_theta[:, None] * jnp.ones((int(trig.cosnv.shape[0]),), dtype=w_theta.dtype)[None, :]
        wint3 = wint[None, :, :]
    else:
        wint3 = jnp.asarray(wint3, dtype=jnp.asarray(trig.cosmu).dtype)

    bsq = jnp.asarray(bsq)
    r12 = jnp.asarray(r12)
    sqrtg = jnp.asarray(sqrtg)
    ru12 = jnp.asarray(ru12)
    zu12 = jnp.asarray(zu12)
    if wint3.shape[1:] != bsq.shape[1:]:
        dnorm3 = jnp.asarray(getattr(trig, "dnorm3", 0.0), dtype=bsq.dtype)
        wint3 = jnp.broadcast_to(dnorm3, (1,) + bsq.shape[1:])

    # Avoid division by zero in ptau.
    gs = jnp.where(sqrtg != 0, sqrtg, jnp.ones_like(sqrtg))
    ptau = (pfactor * (r12 * r12) * bsq * wint3) / gs

    ax_r = jnp.sum(ptau * ((zu12 * ohs) ** 2), axis=(1, 2))  # Z-like call => `ard(js,1)`
    ax_z = jnp.sum(ptau * ((ru12 * ohs) ** 2), axis=(1, 2))  # R-like call => `azd(js,1)`
    ax_r = ax_r.at[0].set(0.0)
    ax_z = ax_z.at[0].set(0.0)

    ard1 = ax_r + jnp.concatenate([ax_r[1:], jnp.zeros((1,), dtype=ax_r.dtype)], axis=0)
    azd1 = ax_z + jnp.concatenate([ax_z[1:], jnp.zeros((1,), dtype=ax_z.dtype)], axis=0)
    return ard1, azd1


def tcon_from_cached_precondn_diag(
    *,
    tcon0: float,
    trig: VmecTrigTables,
    s,
    lasym: bool,
    ard1,
    azd1,
    ru0,
    zu0,
) -> Any:
    """Compute `tcon(js)` from cached `precondn` diagonal outputs and current `ru0/zu0` norms."""
    s = jnp.asarray(s)
    ns = int(s.shape[0])
    if ns < 2:
        return jnp.zeros((ns,), dtype=jnp.asarray(trig.cosmu).dtype)

    hs = jnp.asarray(s[1] - s[0], dtype=jnp.asarray(trig.cosmu).dtype)

    w_theta = jnp.asarray(trig.cosmui3[:, 0]) / jnp.asarray(trig.mscale[0])
    wint = w_theta[:, None] * jnp.ones((int(trig.cosnv.shape[0]),), dtype=w_theta.dtype)[None, :]
    wint3 = wint[None, :, :]

    ard1 = jnp.asarray(ard1)
    azd1 = jnp.asarray(azd1)
    ru0 = jnp.asarray(ru0)
    zu0 = jnp.asarray(zu0)
    if wint3.shape[1:] != ru0.shape[1:]:
        dnorm3 = jnp.asarray(getattr(trig, "dnorm3", 0.0), dtype=ru0.dtype)
        wint3 = jnp.broadcast_to(dnorm3, (1,) + ru0.shape[1:])

    arnorm = jnp.sum((ru0 * ru0) * wint3, axis=(1, 2))
    aznorm = jnp.sum((zu0 * zu0) * wint3, axis=(1, 2))
    arnorm = jnp.where(arnorm != 0, arnorm, jnp.ones_like(arnorm))
    aznorm = jnp.where(aznorm != 0, aznorm, jnp.ones_like(aznorm))

    tcon0 = jnp.minimum(jnp.abs(jnp.asarray(tcon0, dtype=jnp.asarray(trig.cosmu).dtype)), 1.0)
    ns_f = float(ns)
    tcon_mul = tcon0 * (1.0 + ns_f * (1.0 / 60.0 + ns_f / (200.0 * 120.0)))
    tcon_mul = tcon_mul / ((4.0 * (float(trig.r0scale) ** 2)) ** 2)

    # VMEC sets `tcon(:) = tcon0` before overwriting interior surfaces with the
    # preconditioner-based scale. The axis value is not used by the constraint
    # operator, but matching it is helpful for parity dumps.
    tcon0_clamped = jnp.minimum(jnp.abs(jnp.asarray(tcon0, dtype=jnp.asarray(trig.cosmu).dtype)), 1.0)
    tcon = jnp.zeros((ns,), dtype=jnp.asarray(trig.cosmu).dtype)
    tcon = tcon.at[0].set(jnp.asarray(tcon0_clamped, dtype=tcon.dtype))
    if ns >= 3:
        js = jnp.arange(ns, dtype=jnp.int32) + 1
        mask = (js >= 2) & (js <= (ns - 1))
        ratio_r = jnp.abs(ard1) / arnorm
        ratio_z = jnp.abs(azd1) / aznorm
        core = jnp.minimum(ratio_r, ratio_z) * (jnp.asarray(tcon_mul, dtype=hs.dtype) * (32.0 * hs) ** 2)
        tcon = jnp.where(mask.astype(core.dtype), core, tcon)
        tcon = tcon.at[-1].set(0.5 * tcon[-2])

    # VMEC's lasym-specific halving is commented out in current STELLOPT sources.
    _ = lasym
    return tcon


def tcon_from_precondn_axisym(
    *,
    tcon0: float,
    bc,
    k,
    cfg,
    s,
    trig: VmecTrigTables,
    ru0,
    zu0,
) -> Any:
    """Compute `tcon(js)` from a VMEC-style axisymmetric preconditioner diagonal.

    This uses the same `precondn` diagonal elements (`ard/azd`) that VMEC
    feeds into the `tcon` scaling in `bcovar.f`. It is restricted to the
    axisymmetric, stellarator-symmetric path.
    """
    if bool(cfg.lthreed) or bool(cfg.lasym):
        raise ValueError("tcon_from_precondn_axisym only supports axisym.")

    s = jnp.asarray(s)
    ns = int(s.shape[0])
    if ns < 2:
        return jnp.zeros((ns,), dtype=jnp.asarray(trig.cosmu).dtype)

    dtype = jnp.asarray(trig.cosmu).dtype
    hs = jnp.asarray(s[1] - s[0], dtype=dtype)

    # Import preconditioner helpers lazily to avoid circular imports.
    from .. import preconditioner_1d_jax as _p1d

    w_int = _p1d._wint_from_config(cfg=cfg, dtype=dtype)
    sqrt_sf, sqrt_sh = _p1d._sqrt_profiles_from_ns(ns, dtype=dtype)
    sm, sp = _p1d._sm_sp_from_profiles(sqrt_sf, sqrt_sh)
    delta_s = jnp.where(ns >= 2, s[1] - s[0], jnp.asarray(1.0, dtype=dtype))

    r12 = jnp.asarray(bc.jac.r12, dtype=dtype)[1:]
    tau = jnp.asarray(bc.jac.tau, dtype=dtype)[1:]
    total_pressure = jnp.asarray(bc.bsq, dtype=dtype)[1:]
    bsupv = jnp.asarray(bc.bsupv, dtype=dtype)[1:]
    sqrtg = jnp.asarray(bc.jac.sqrtg, dtype=dtype)[1:]

    # R-like preconditioner (uses Z-derivatives).
    _axm_r, axd_r, _bxm_r, _bxd_r, _cxd_r = _p1d._compute_preconditioning_matrix(
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
        ns_full=ns,
    )
    # Z-like preconditioner (uses R-derivatives).
    _axm_z, axd_z, _bxm_z, _bxd_z, _cxd_z = _p1d._compute_preconditioning_matrix(
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
        ns_full=ns,
    )

    # Use the m-parity=0 diagonal (Fortran index 1).
    ard1 = jnp.asarray(axd_r[:, 0], dtype=dtype)
    azd1 = jnp.asarray(axd_z[:, 0], dtype=dtype)

    # Flux-surface norms of ru0/zu0 (bcovar.f).
    w_theta = jnp.asarray(trig.cosmui3[:, 0], dtype=dtype) / jnp.asarray(trig.mscale[0], dtype=dtype)
    wint = w_theta[:, None] * jnp.ones((int(trig.cosnv.shape[0]),), dtype=dtype)[None, :]
    wint3 = wint[None, :, :]
    ru0 = jnp.asarray(ru0, dtype=dtype)
    zu0 = jnp.asarray(zu0, dtype=dtype)
    arnorm = jnp.sum((ru0 * ru0) * wint3, axis=(1, 2))
    aznorm = jnp.sum((zu0 * zu0) * wint3, axis=(1, 2))
    arnorm = jnp.where(arnorm != 0.0, arnorm, jnp.ones_like(arnorm))
    aznorm = jnp.where(aznorm != 0.0, aznorm, jnp.ones_like(aznorm))

    # bcovar.f scaling for tcon_mul (with clamped tcon0).
    tcon0 = jnp.minimum(jnp.abs(jnp.asarray(tcon0, dtype=dtype)), 1.0)
    ns_f = float(ns)
    tcon_mul = tcon0 * (1.0 + ns_f * (1.0 / 60.0 + ns_f / (200.0 * 120.0)))
    tcon_mul = tcon_mul / ((4.0 * (float(trig.r0scale) ** 2)) ** 2)

    tcon = jnp.zeros((ns,), dtype=dtype)
    if ns >= 3:
        js = jnp.arange(ns, dtype=jnp.int32)
        mask = (js >= 1) & (js <= (ns - 2))
        ratio_r = jnp.abs(ard1) / arnorm
        ratio_z = jnp.abs(azd1) / aznorm
        core = jnp.minimum(ratio_r, ratio_z) * (jnp.asarray(tcon_mul, dtype=dtype) * (32.0 * hs) ** 2)
        tcon = jnp.where(mask.astype(dtype), core, tcon)
        tcon = tcon.at[-1].set(0.5 * tcon[-2])
    return tcon


def alias_gcon(
    *,
    ztemp: Any,
    trig: VmecTrigTables,
    ntor: int,
    mpol: int,
    signgs: int,
    tcon: Any,
    lasym: bool,
) -> Any:
    """Compute VMEC's `gcon` field from `ztemp` (alias.f).

    Parameters
    ----------
    ztemp:
        Shape (ns, ntheta3, nzeta), defined on the VMEC internal theta grid.
    trig:
        Trig tables from :func:`vmec_jax.kernels.tomnsp.vmec_trig_tables`.
    ntor, mpol:
        Spectral truncation. Uses n=0..ntor and m=0..mpol-1.
    signgs:
        VMEC orientation sign (+1/-1), used in `faccon`.
    tcon:
        Per-surface constraint scale factor (ns,).
    lasym:
        If True, uses the full lasym de-aliasing path in `alias.f`.
    """
    ztemp = jnp.asarray(ztemp)
    ns, ntheta3, nzeta = ztemp.shape
    if int(ntheta3) != int(trig.ntheta3):
        raise ValueError("ztemp theta size must match trig.ntheta3")
    if int(nzeta) != int(trig.cosnv.shape[0]):
        raise ValueError("ztemp nzeta must match trig.cosnv")

    mpol = int(mpol)
    ntor = int(ntor)
    faccon = faccon_from_signgs(mpol=mpol, signgs=signgs, dtype=jnp.asarray(trig.cosmu).dtype)  # (mpol,)
    tcon = jnp.asarray(tcon, dtype=jnp.asarray(trig.cosmu).dtype)  # (ns,)

    nt2 = int(trig.ntheta2)
    zhalf = ztemp[:, :nt2, :]  # (ns,nt2,nzeta)

    cosmui = trig.cosmui[:nt2, :mpol]  # (nt2,mpol)
    sinmui = trig.sinmui[:nt2, :mpol]
    cosmu = trig.cosmu[:nt2, :mpol]
    sinmu = trig.sinmu[:nt2, :mpol]

    cosnv = trig.cosnv[:, : (ntor + 1)]  # (nzeta,ntor+1)
    sinnv = trig.sinnv[:, : (ntor + 1)]

    # Theta integration on half interval.
    w1 = einsum("sik,im->smk", zhalf, cosmui)
    w2 = einsum("sik,im->smk", zhalf, sinmui)

    if not lasym:
        # Forward zeta transform (de-aliased):
        gcs = (tcon[:, None, None] * einsum("smk,kn->smn", w1, sinnv))
        gsc = (tcon[:, None, None] * einsum("smk,kn->smn", w2, cosnv))

        # Inverse zeta transform:
        work3 = einsum("smn,kn->smk", gcs, sinnv)
        work4 = einsum("smn,kn->smk", gsc, cosnv)

        cosmu_fac = cosmu * faccon[None, :]
        sinmu_fac = sinmu * faccon[None, :]
        gcon_half = einsum("smk,im->sik", work3, cosmu_fac) + einsum("smk,im->sik", work4, sinmu_fac)

        gcon = jnp.zeros((ns, ntheta3, nzeta), dtype=jnp.asarray(trig.cosmu).dtype)
        gcon = gcon.at[:, :nt2, :].set(gcon_half)
        return gcon

    # lasym=True path (alias.f: alias_par)
    ntheta1 = int(trig.ntheta1)
    if ntheta3 != ntheta1:
        raise ValueError("lasym=True requires trig.ntheta3 == trig.ntheta1")

    # Build reflected ztemp(kk,ir) restricted to i=1..ntheta2 (weights use original i).
    i0 = np.arange(nt2, dtype=int)
    ir = np.where(i0 == 0, 0, ntheta1 - i0)  # (nt2,)
    kk = (nzeta - np.arange(nzeta, dtype=int)) % nzeta  # (nzeta,)
    zref = ztemp[:, ir, :][:, :, kk]  # (ns, nt2, nzeta)

    w3 = einsum("sik,im->smk", zref, cosmui)
    w4 = einsum("sik,im->smk", zref, sinmui)

    half = 0.5 * tcon[:, None, None]
    gcs = half * einsum("smk,kn->smn", (w1 - w3), sinnv)
    gsc = half * einsum("smk,kn->smn", (w2 - w4), cosnv)
    gss = half * einsum("smk,kn->smn", (w2 + w4), sinnv)
    gcc = half * einsum("smk,kn->smn", (w1 + w3), cosnv)

    work3 = einsum("smn,kn->smk", gcs, sinnv)
    work4 = einsum("smn,kn->smk", gsc, cosnv)
    work1 = einsum("smn,kn->smk", gcc, cosnv)
    work2 = einsum("smn,kn->smk", gss, sinnv)

    cosmu_fac = cosmu * faccon[None, :]
    sinmu_fac = sinmu * faccon[None, :]
    gcons_half = einsum("smk,im->sik", work3, cosmu_fac) + einsum("smk,im->sik", work4, sinmu_fac)
    gcona_half = einsum("smk,im->sik", work1, cosmu_fac) + einsum("smk,im->sik", work2, sinmu_fac)

    gcon = jnp.zeros((ns, ntheta3, nzeta), dtype=jnp.asarray(trig.cosmu).dtype)
    gcon = gcon.at[:, :nt2, :].set(gcons_half + gcona_half)

    # Extend theta into [pi,2pi) using VMEC's reflection + zeta reversal.
    n_second = ntheta1 - nt2
    if n_second > 0:
        i2 = np.arange(nt2, ntheta1, dtype=int)
        ir2 = (ntheta1 - i2).astype(int)  # maps into [1..nt2-1]
        gcons_ref = gcons_half[:, ir2, :][:, :, kk]
        gcona_ref = gcona_half[:, ir2, :][:, :, kk]
        gcon = gcon.at[:, nt2:, :].set(-gcons_ref + gcona_ref)

    return gcon
