"""Differentiable Mercier and JXBFORCE algebra kernels.

These helpers are the source-level JAX counterparts of VMEC's Mercier and
JXBFORCE profile reductions.  They intentionally avoid VMEC-state orchestration
so they can be tested against the algebra directly and reused by finite-beta
objectives.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._compat import jnp


def glasser_resistive_interchange_from_mercier_terms(
    *,
    DMerc,
    shear,
    H=None,
    Dcurr=None,
    tjb=None,
    tbb=None,
    jdotb=None,
    bdotb=None,
    shear_epsilon: float = 0.0,
) -> dict[str, Any]:
    """Return the Glasser resistive-interchange criterion from Mercier terms.

    Landreman & Jorge rewrite the Glasser-Greene-Johnson necessary condition
    for resistive interchange stability as

    ``D_R = -DMerc + 4*pi^2/iota_prime^2 * (H - iota_prime^2/(8*pi^2))^2``.

    The Mercier implementation in this module uses the VMEC/Ichiguchi
    normalization in which ``shear = d iota / d Phi`` and
    ``Dshear = shear**2 / 4``.  Since ``Phi = 2*pi*psi``, the equivalent
    normalized expression is

    ``D_R = -DMerc + (H - shear**2/2)**2 / shear**2``.

    The strict resistive-interchange necessary condition is ``D_R <= 0`` on
    surfaces with nonzero magnetic shear.  ``glasser_shear_valid`` marks the
    surfaces on which this division is physically meaningful.  A positive
    ``shear_epsilon`` regularizes the denominator for smooth optimization
    penalties while preserving the validity mask.

    If ``H`` is not supplied, it is reconstructed from VMEC profile data as
    ``shear * (tjb - (jdotb / bdotb) * tbb)`` when those profiles are
    available.  If only the VMEC Mercier current term is available, the helper
    falls back to ``H = -Dcurr``; this is exact when the surface-averaged
    parallel-current ratio equals the toroidal-current derivative.
    """
    DMerc = jnp.asarray(DMerc, dtype=jnp.float64)
    shear = jnp.asarray(shear, dtype=jnp.float64)

    if H is None:
        if tjb is not None and tbb is not None and jdotb is not None and bdotb is not None:
            tjb = jnp.asarray(tjb, dtype=jnp.float64)
            tbb = jnp.asarray(tbb, dtype=jnp.float64)
            jdotb = jnp.asarray(jdotb, dtype=jnp.float64)
            bdotb = jnp.asarray(bdotb, dtype=jnp.float64)
            ratio = jnp.where(bdotb != 0.0, jdotb / bdotb, 0.0)
            H = shear * (tjb - ratio * tbb)
        elif Dcurr is not None:
            H = -jnp.asarray(Dcurr, dtype=jnp.float64)
        else:
            raise ValueError("Either H, Dcurr, or (tjb, tbb, jdotb, bdotb) must be supplied.")
    H = jnp.asarray(H, dtype=jnp.float64)

    shear2 = shear * shear
    eps = jnp.asarray(float(shear_epsilon), dtype=jnp.float64)
    eps2 = eps * eps
    denom = shear2 + eps2
    valid = shear2 > eps2
    denom_safe = jnp.where(denom != 0.0, denom, 1.0)
    correction_raw = (H - 0.5 * shear2) ** 2 / denom_safe
    correction = jnp.where(valid | (float(shear_epsilon) > 0.0), correction_raw, 0.0)
    D_R = jnp.where(valid | (float(shear_epsilon) > 0.0), -DMerc + correction_raw, 0.0)
    return {
        "D_R": D_R,
        "H": H,
        "glasser_correction": correction,
        "glasser_shear_valid": valid,
    }


def mercier_terms_from_profile_integrals(
    *,
    s,
    phips,
    iotas,
    vp,
    pres,
    torcur,
    tpp,
    tbb,
    tjb,
    tjj,
    jdotb=None,
    bdotb=None,
    shear_epsilon: float = 0.0,
    signgs: int = 1,
) -> dict[str, Any]:
    """Return JAX-differentiable Mercier terms from 1D VMEC profile integrals.

    This is the algebraic core of VMEC's ``mercier.f`` calculation after the
    geometric surface averages have been assembled:

    ``DMerc = DShear + DCurr + DWell + DGeod``.

    The inputs ``tpp``, ``tbb``, ``tjb``, and ``tjj`` are the per-surface
    geometry/current integrals in the same normalization used by the Mercier
    formula, i.e. after the ``(2*pi)^2`` factor applied in ``wout._compute_mercier``.
    This helper is intentionally small and differentiable; the geometric
    surface-integral assembly is handled separately.
    """
    s = jnp.asarray(s, dtype=jnp.float64)
    phips = jnp.asarray(phips, dtype=jnp.float64)
    iotas = jnp.asarray(iotas, dtype=jnp.float64)
    vp = jnp.asarray(vp, dtype=jnp.float64)
    pres = jnp.asarray(pres, dtype=jnp.float64)
    torcur = jnp.asarray(torcur, dtype=jnp.float64)
    tpp = jnp.asarray(tpp, dtype=jnp.float64)
    tbb = jnp.asarray(tbb, dtype=jnp.float64)
    tjb = jnp.asarray(tjb, dtype=jnp.float64)
    tjj = jnp.asarray(tjj, dtype=jnp.float64)

    ns = int(s.shape[0])
    zeros = jnp.zeros_like(s, dtype=jnp.float64)
    if ns < 3:
        valid = jnp.zeros_like(s, dtype=bool)
        return {
            "DMerc": zeros,
            "Dshear": zeros,
            "Dcurr": zeros,
            "Dwell": zeros,
            "Dgeod": zeros,
            "D_R": zeros,
            "H": zeros,
            "glasser_correction": zeros,
            "glasser_shear_valid": valid,
            "shear": zeros,
            "vpp": zeros,
            "presp": zeros,
            "ip": zeros,
        }

    sign_jac = jnp.asarray(1.0 if int(signgs) >= 0 else -1.0, dtype=jnp.float64)
    twopi = jnp.asarray(2.0 * np.pi, dtype=jnp.float64)
    hs = jnp.asarray(1.0 / float(ns - 1), dtype=jnp.float64)
    phip_real = twopi * phips * sign_jac
    vp_real = jnp.where(phip_real != 0.0, sign_jac * twopi * twopi * vp / phip_real, 0.0)
    vp_real = vp_real.at[0].set(0.0)

    phip_full = 0.5 * (phip_real[2:] + phip_real[1:-1])
    denom = jnp.where(phip_full != 0.0, 1.0 / (hs * phip_full), 0.0)
    shear_inner = (iotas[2:] - iotas[1:-1]) * denom
    vpp_inner = (vp_real[2:] - vp_real[1:-1]) * denom
    presp_inner = (pres[2:] - pres[1:-1]) * denom
    ip_inner = (torcur[2:] - torcur[1:-1]) * denom

    dshear_inner = 0.25 * shear_inner * shear_inner
    dcurr_inner = -shear_inner * (tjb[1:-1] - ip_inner * tbb[1:-1])
    dwell_inner = presp_inner * (vpp_inner - presp_inner * tpp[1:-1]) * tbb[1:-1]
    dgeod_inner = tjb[1:-1] * tjb[1:-1] - tbb[1:-1] * tjj[1:-1]

    Dshear = zeros.at[1:-1].set(dshear_inner)
    Dcurr = zeros.at[1:-1].set(dcurr_inner)
    Dwell = zeros.at[1:-1].set(dwell_inner)
    Dgeod = zeros.at[1:-1].set(dgeod_inner)
    shear = zeros.at[1:-1].set(shear_inner)
    vpp = zeros.at[1:-1].set(vpp_inner)
    presp = zeros.at[1:-1].set(presp_inner)
    ip = zeros.at[1:-1].set(ip_inner)
    DMerc = Dshear + Dcurr + Dwell + Dgeod
    glasser = glasser_resistive_interchange_from_mercier_terms(
        DMerc=DMerc,
        shear=shear,
        Dcurr=Dcurr,
        tjb=tjb,
        tbb=tbb,
        jdotb=jdotb,
        bdotb=bdotb,
        shear_epsilon=shear_epsilon,
    )
    return {
        "DMerc": DMerc,
        "Dshear": Dshear,
        "Dcurr": Dcurr,
        "Dwell": Dwell,
        "Dgeod": Dgeod,
        **glasser,
        "shear": shear,
        "vpp": vpp,
        "presp": presp,
        "ip": ip,
    }


def mercier_surface_integrals_from_realspace(
    *,
    phips,
    sqrtg,
    b2,
    gpp,
    bdotk_merc,
    wint,
    signgs: int = 1,
) -> dict[str, Any]:
    """Return JAX-differentiable Mercier surface integrals.

    Inputs are real-space arrays on the full radial mesh.  ``b2`` is the
    pressure-subtracted field strength used by VMEC's Mercier formula
    (``2 * (bsq - pressure)`` in the wout path), ``gpp`` is the contravariant
    metric component on the half-mesh surface, ``bdotk_merc`` is VMEC's
    ``mu0 * sqrt(g) J.B`` channel, and ``wint`` are the VMEC quadrature weights
    over ``(theta, zeta)``.

    The returned ``tpp``, ``tbb``, ``tjb``, and ``tjj`` arrays feed directly into
    :func:`mercier_terms_from_profile_integrals`.
    """
    phips = jnp.asarray(phips, dtype=jnp.float64)
    sqrtg = jnp.asarray(sqrtg, dtype=jnp.float64)
    b2 = jnp.asarray(b2, dtype=jnp.float64)
    gpp = jnp.asarray(gpp, dtype=jnp.float64)
    bdotk_merc = jnp.asarray(bdotk_merc, dtype=jnp.float64)
    wint = jnp.asarray(wint, dtype=jnp.float64)
    ns = int(phips.shape[0])
    zeros = jnp.zeros_like(phips, dtype=jnp.float64)
    if ns < 3:
        return {"tpp": zeros, "tbb": zeros, "tjb": zeros, "tjj": zeros}

    sign_jac = jnp.asarray(1.0 if int(signgs) >= 0 else -1.0, dtype=jnp.float64)
    twopi = jnp.asarray(2.0 * np.pi, dtype=jnp.float64)
    phip_real = twopi * phips * sign_jac
    phip_full = 0.5 * (phip_real[2:] + phip_real[1:-1])
    gsqrt_raw = 0.5 * (sqrtg[2:] + sqrtg[1:-1])
    phip_full = phip_full[:, None, None]
    phip_safe = jnp.where(phip_full != 0.0, phip_full, 1.0)
    gsqrt_full = jnp.where(phip_full != 0.0, gsqrt_raw / phip_safe, 0.0)
    b2i = 0.5 * (b2[2:] + b2[1:-1])
    b2_safe = jnp.where(b2i != 0.0, b2i, jnp.asarray(1.0, dtype=jnp.float64))
    norm = twopi * twopi

    weighted_sum = lambda arr: jnp.sum(arr * wint[None, :, :], axis=(1, 2))
    tpp_inner = weighted_sum(gsqrt_full / b2_safe) * norm
    tbb_inner = weighted_sum(b2i * gsqrt_full * gpp[1:-1]) * norm
    bdotj_norm = jnp.where(gsqrt_raw != 0.0, bdotk_merc[1:-1] / gsqrt_raw, 0.0)
    jdotb = bdotj_norm * gpp[1:-1] * gsqrt_full
    tjb_inner = weighted_sum(jdotb) * norm
    tjj_inner = weighted_sum(jdotb * bdotj_norm / b2_safe) * norm

    return {
        "tpp": zeros.at[1:-1].set(tpp_inner),
        "tbb": zeros.at[1:-1].set(tbb_inner),
        "tjb": zeros.at[1:-1].set(tjb_inner),
        "tjj": zeros.at[1:-1].set(tjj_inner),
    }


def jxbforce_profiles_from_realspace(
    *,
    phips,
    sqrtg,
    bsq,
    pres,
    vp,
    bdotk,
    wint,
    sigma_an=None,
    signgs: int = 1,
) -> dict[str, Any]:
    """Return JAX-differentiable JXBFORCE 1D field/current profiles.

    This is the state-level counterpart of VMEC's ``jxbforce`` reductions for
    ``jdotb``, ``bdotb`` and ``bdotgradv``.  Inputs are full-radial-mesh
    real-space arrays in the same normalization used by VMEC's Mercier path.
    The returned arrays live on the full radial mesh, with VMEC's endpoint
    extrapolation convention.
    """
    phips = jnp.asarray(phips, dtype=jnp.float64)
    sqrtg = jnp.asarray(sqrtg, dtype=jnp.float64)
    bsq = jnp.asarray(bsq, dtype=jnp.float64)
    pres = jnp.asarray(pres, dtype=jnp.float64)
    vp = jnp.asarray(vp, dtype=jnp.float64)
    bdotk = jnp.asarray(bdotk, dtype=jnp.float64)
    wint = jnp.asarray(wint, dtype=jnp.float64)
    sigma = jnp.ones_like(sqrtg, dtype=jnp.float64) if sigma_an is None else jnp.asarray(sigma_an, dtype=jnp.float64)

    ns = int(phips.shape[0])
    zeros = jnp.zeros_like(phips, dtype=jnp.float64)
    if ns < 3:
        return {"jdotb": zeros, "bdotb": zeros, "bdotgradv": zeros}

    dnorm1 = jnp.asarray((2.0 * np.pi) ** 2, dtype=jnp.float64)
    sign_jac = jnp.asarray(1.0 if int(signgs) >= 0 else -1.0, dtype=jnp.float64)
    denom = vp[2:] + vp[1:-1]
    ovp = jnp.where(denom != 0.0, 2.0 / denom / dnorm1, 0.0)
    tjnorm = ovp * sign_jac
    weighted_sum = lambda arr: jnp.sum(arr * wint[None, :, :], axis=(1, 2))

    sqgb2 = sqrtg[2:] * (bsq[2:] - pres[2:, None, None]) + sqrtg[1:-1] * (
        bsq[1:-1] - pres[1:-1, None, None]
    )
    sigma_inner = sigma[1:-1]
    jdotb_inner = dnorm1 * tjnorm * weighted_sum(bdotk[1:-1] / sigma_inner)
    bdotb_inner = dnorm1 * tjnorm * weighted_sum(sqgb2 / sigma_inner)
    bdotgradv_inner = 0.5 * dnorm1 * tjnorm * (phips[1:-1] + phips[2:])

    jdotb = zeros.at[1:-1].set(jdotb_inner)
    bdotb = zeros.at[1:-1].set(bdotb_inner)
    bdotgradv = zeros.at[1:-1].set(bdotgradv_inner)

    jdotb = jdotb.at[0].set(2.0 * jdotb[1] - jdotb[2])
    jdotb = jdotb.at[-1].set(2.0 * jdotb[-2] - jdotb[-3])
    bdotb = bdotb.at[0].set(2.0 * bdotb[2] - bdotb[1])
    bdotb = bdotb.at[-1].set(2.0 * bdotb[-2] - bdotb[-3])
    bdotgradv = bdotgradv.at[0].set(2.0 * bdotgradv[1] - bdotgradv[2])
    bdotgradv = bdotgradv.at[-1].set(2.0 * bdotgradv[-2] - bdotgradv[-3])

    return {"jdotb": jdotb, "bdotb": bdotb, "bdotgradv": bdotgradv}


__all__ = [
    "glasser_resistive_interchange_from_mercier_terms",
    "jxbforce_profiles_from_realspace",
    "mercier_surface_integrals_from_realspace",
    "mercier_terms_from_profile_integrals",
]
