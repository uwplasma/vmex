"""Finite-beta optimization diagnostics and residual helpers.

These helpers are intentionally VMEC-state based and JAX differentiable.  They
cover the global stage-one finite-beta quantities that are cheap and stable to
differentiate through the fixed-boundary discrete-adjoint path: aspect ratio,
iota bounds, volume-averaged field proxy, and total beta.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

from ._compat import jnp
from .energy import flux_profiles_from_indata
from .profiles import eval_profiles
from .solve import _half_mesh_from_full_mesh, _icurv_full_mesh_from_indata, _mass_half_mesh_from_indata
from .vmec_bcovar import vmec_bcovar_half_mesh_from_wout
from .vmec_residue import vmec_force_norms_from_bcovar_dynamic
from .wout import _chipf_from_chips, equilibrium_aspect_ratio_from_state, equilibrium_iota_profiles_from_state

MU0 = 4e-7 * np.pi


@dataclass(frozen=True)
class FiniteBetaTargets:
    """Targets and weights for stage-one finite-beta fixed-boundary objectives."""

    aspect_ratio: float
    min_iota: float
    min_average_iota: float
    max_iota: float
    volavgB: float
    beta_total: float
    aspect_weight: float = 1.0
    iota_weight: float = 1.0
    max_iota_weight: float = 1.0
    volavgB_weight: float = 1.0
    beta_weight: float = 1.0


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
    signgs: int = 1,
) -> dict[str, Any]:
    """Return JAX-differentiable Mercier terms from 1D VMEC profile integrals.

    This is the algebraic core of VMEC's ``mercier.f`` calculation after the
    geometric surface averages have been assembled:

    ``DMerc = DShear + DCurr + DWell + DGeod``.

    The inputs ``tpp``, ``tbb``, ``tjb``, and ``tjj`` are the per-surface
    geometry/current integrals in the same normalization used by the Mercier
    formula, i.e. after the ``(2*pi)^2`` factor applied in ``wout._compute_mercier``.
    This helper is intentionally small and differentiable; the next porting
    step is to replace the remaining NumPy surface-integral assembly with a JAX
    path that feeds this function.
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
        return {
            "DMerc": zeros,
            "Dshear": zeros,
            "Dcurr": zeros,
            "Dwell": zeros,
            "Dgeod": zeros,
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
    return {
        "DMerc": Dshear + Dcurr + Dwell + Dgeod,
        "Dshear": Dshear,
        "Dcurr": Dcurr,
        "Dwell": Dwell,
        "Dgeod": Dgeod,
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


def mercier_gpp_from_realspace_geometry(
    *,
    s,
    phips,
    sqrtg,
    R_even,
    R_odd,
    Ru_even,
    Ru_odd,
    Zu_even,
    Zu_odd,
    Rv_even,
    Rv_odd,
    Zv_even,
    Zv_odd,
    signgs: int = 1,
) -> Any:
    """Return VMEC Mercier contravariant ``gpp`` from real-space geometry.

    The inputs are the even/odd VMEC real-space channels used in the Mercier
    path after the internal parity conversion:
    ``X(s,theta,zeta) = X_even + sqrt(s) * X_odd``.  The returned array has the
    same ``(ns, ntheta, nzeta)`` shape as the input geometry and is populated on
    interior full-mesh surfaces.  Endpoints are zero because Mercier terms are
    only defined on ``1 <= js <= ns-2`` in VMEC's convention.
    """
    s = jnp.asarray(s, dtype=jnp.float64)
    phips = jnp.asarray(phips, dtype=jnp.float64)
    sqrtg = jnp.asarray(sqrtg, dtype=jnp.float64)
    R_even = jnp.asarray(R_even, dtype=jnp.float64)
    R_odd = jnp.asarray(R_odd, dtype=jnp.float64)
    Ru_even = jnp.asarray(Ru_even, dtype=jnp.float64)
    Ru_odd = jnp.asarray(Ru_odd, dtype=jnp.float64)
    Zu_even = jnp.asarray(Zu_even, dtype=jnp.float64)
    Zu_odd = jnp.asarray(Zu_odd, dtype=jnp.float64)
    Rv_even = jnp.asarray(Rv_even, dtype=jnp.float64)
    Rv_odd = jnp.asarray(Rv_odd, dtype=jnp.float64)
    Zv_even = jnp.asarray(Zv_even, dtype=jnp.float64)
    Zv_odd = jnp.asarray(Zv_odd, dtype=jnp.float64)

    ns = int(s.shape[0])
    zeros = jnp.zeros_like(sqrtg, dtype=jnp.float64)
    if ns < 3:
        return zeros

    sign_jac = jnp.asarray(1.0 if int(signgs) >= 0 else -1.0, dtype=jnp.float64)
    twopi = jnp.asarray(2.0 * np.pi, dtype=jnp.float64)
    phip_real = twopi * phips * sign_jac
    phip_full = 0.5 * (phip_real[2:] + phip_real[1:-1])
    gsqrt_raw = 0.5 * (sqrtg[2:] + sqrtg[1:-1])
    phip_full = phip_full[:, None, None]
    phip_safe = jnp.where(phip_full != 0.0, phip_full, 1.0)
    gsqrt_full = jnp.where(phip_full != 0.0, gsqrt_raw / phip_safe, 0.0)

    sqs = jnp.sqrt(jnp.maximum(s[1:-1], 0.0))[:, None, None]
    r1f = R_even[1:-1] + sqs * R_odd[1:-1]
    rtf = Ru_even[1:-1] + sqs * Ru_odd[1:-1]
    ztf = Zu_even[1:-1] + sqs * Zu_odd[1:-1]
    rzf = Rv_even[1:-1] + sqs * Rv_odd[1:-1]
    zzf = Zv_even[1:-1] + sqs * Zv_odd[1:-1]
    gtt = rtf * rtf + ztf * ztf
    denom = gtt * r1f * r1f + (rtf * zzf - rzf * ztf) ** 2
    denom_safe = jnp.where(denom != 0.0, denom, 1.0)
    gpp_inner = jnp.where(denom != 0.0, (gsqrt_full * gsqrt_full) / denom_safe, 0.0)
    return zeros.at[1:-1].set(gpp_inner)


def mercier_realspace_geometry_channels_from_state(
    *,
    state,
    modes,
    trig,
    s,
    lconm1: bool = True,
    lthreed: bool = True,
    lasym: bool = False,
    apply_scalxc: bool = True,
) -> dict[str, Any]:
    """Return VMEC even/odd real-space R/Z geometry channels for Mercier.

    This mirrors the geometry synthesis used by the NumPy ``wout`` Mercier
    parity path: VMEC internal Fourier coefficients are split by even/odd
    poloidal mode number, axis rules are applied, and base/theta/zeta
    derivatives are synthesized on the VMEC angular grid.  The odd channels are
    VMEC-internal channels; physical fields are recovered as
    ``X_even + sqrt(s) * X_odd``.
    """
    from .vmec_jacobian import _apply_vmec_axis_rules
    from .vmec_parity import vmec_m1_internal_to_physical_signed
    from .vmec_realspace import vmec_realspace_synthesis_multi

    s = jnp.asarray(s, dtype=jnp.float64)
    m_np = np.asarray(modes.m, dtype=int)
    Rcos = jnp.asarray(state.Rcos, dtype=jnp.float64)
    Rsin = jnp.asarray(state.Rsin, dtype=jnp.float64)
    Zcos = jnp.asarray(state.Zcos, dtype=jnp.float64)
    Zsin = jnp.asarray(state.Zsin, dtype=jnp.float64)

    if bool(lconm1):
        Rcos, Zsin, Rsin, Zcos = vmec_m1_internal_to_physical_signed(
            Rcos=Rcos,
            Zsin=Zsin,
            Rsin=Rsin,
            Zcos=Zcos,
            modes=modes,
            lthreed=bool(lthreed),
            lasym=bool(lasym),
            lconm1=bool(lconm1),
        )

    Rcos = _apply_vmec_axis_rules(Rcos, m_np)
    Rsin = _apply_vmec_axis_rules(Rsin, m_np)
    Zcos = _apply_vmec_axis_rules(Zcos, m_np)
    Zsin = _apply_vmec_axis_rules(Zsin, m_np)

    dtype = Rcos.dtype
    mask_even = jnp.asarray((m_np % 2) == 0, dtype=dtype)
    mask_odd = 1.0 - mask_even
    coeff_cos_stack = jnp.stack([Rcos, Zcos], axis=0)
    coeff_sin_stack = jnp.stack([Rsin, Zsin], axis=0)
    mask_stack = jnp.stack([mask_even, mask_odd], axis=0)
    coeff_cos = coeff_cos_stack[None, ...] * mask_stack[:, None, None, :]
    coeff_sin = coeff_sin_stack[None, ...] * mask_stack[:, None, None, :]

    stack, stack_t, stack_p = vmec_realspace_synthesis_multi(
        coeff_cos=coeff_cos,
        coeff_sin=coeff_sin,
        modes=modes,
        trig=trig,
        coeffs_internal=True,
        apply_scalxc=bool(apply_scalxc),
        s=s,
        derivs=("base", "dtheta", "dzeta"),
    )
    even = stack[0]
    odd = stack[1]
    even_t = stack_t[0]
    odd_t = stack_t[1]
    even_p = stack_p[0]
    odd_p = stack_p[1]
    return {
        "R_even": even[0],
        "R_odd": odd[0],
        "Z_even": even[1],
        "Z_odd": odd[1],
        "Ru_even": even_t[0],
        "Ru_odd": odd_t[0],
        "Zu_even": even_t[1],
        "Zu_odd": odd_t[1],
        "Rv_even": even_p[0],
        "Rv_odd": odd_p[0],
        "Zv_even": even_p[1],
        "Zv_odd": odd_p[1],
    }


def mercier_bsubs_derivatives_lasym_false(
    *,
    bsubs,
    trig,
    mmax_force: int,
    nmax_force: int,
) -> dict[str, Any]:
    """Return VMEC jxbforce ``bsubsu``/``bsubsv`` for stellarator symmetry.

    ``bsubs`` must be the VMEC full-mesh covariant radial field channel after
    the jxbforce radial averaging/filtering convention.  This helper ports the
    vectorized stellarator-symmetric branch in :mod:`vmec_jax.wout` to JAX so
    the Mercier ``bdotk`` path can be assembled without NumPy postprocessing.
    """
    bsubs = jnp.asarray(bsubs, dtype=jnp.float64)
    ns, ntheta, nzeta = bsubs.shape
    nt2 = int(trig.ntheta2)
    if int(ntheta) < nt2:
        raise ValueError("bsubs grid smaller than trig.ntheta2")

    mmax = int(mmax_force)
    nmax = int(nmax_force)
    if mmax < 0 or nmax < 0:
        zeros = jnp.zeros((int(ns), nt2, int(nzeta)), dtype=jnp.float64)
        return {"bsubsu": zeros, "bsubsv": zeros}

    cosmui = jnp.asarray(trig.cosmui, dtype=jnp.float64)[:nt2, : mmax + 1]
    sinmui = jnp.asarray(trig.sinmui, dtype=jnp.float64)[:nt2, : mmax + 1]
    cosmu = jnp.asarray(trig.cosmu, dtype=jnp.float64)[:nt2, : mmax + 1]
    sinmu = jnp.asarray(trig.sinmu, dtype=jnp.float64)[:nt2, : mmax + 1]
    cosmum = jnp.asarray(trig.cosmum, dtype=jnp.float64)[:nt2, : mmax + 1]
    sinmum = jnp.asarray(trig.sinmum, dtype=jnp.float64)[:nt2, : mmax + 1]
    cosnv = jnp.asarray(trig.cosnv, dtype=jnp.float64)[:, : nmax + 1]
    sinnv = jnp.asarray(trig.sinnv, dtype=jnp.float64)[:, : nmax + 1]
    cosnvn = jnp.asarray(trig.cosnvn, dtype=jnp.float64)[:, : nmax + 1]
    sinnvn = jnp.asarray(trig.sinnvn, dtype=jnp.float64)[:, : nmax + 1]

    r0scale = float(getattr(trig, "r0scale", 1.0))
    dnorm = jnp.asarray(1.0 / (r0scale**2), dtype=jnp.float64)
    dmult = jnp.full((mmax + 1, nmax + 1), dnorm, dtype=jnp.float64)
    mnyq = int(np.asarray(trig.cosmui).shape[1] - 1)
    nnyq = int(np.asarray(trig.cosnv).shape[1] - 1)
    if mnyq > 0 and mnyq <= mmax:
        dmult = dmult.at[mnyq, :].multiply(0.5)
    if nnyq > 0 and nnyq <= nmax:
        dmult = dmult.at[:, nnyq].multiply(0.5)

    bsubs_nt2 = bsubs[:, :nt2, :]
    f_theta_sin = jnp.einsum("sik,im->smk", bsubs_nt2, sinmui, optimize=True)
    f_theta_cos = jnp.einsum("sik,im->smk", bsubs_nt2, cosmui, optimize=True)
    bsubsmn1 = jnp.einsum("smk,kn->smn", f_theta_sin, cosnv, optimize=True) * dmult[None, :, :]
    bsubsmn2 = jnp.einsum("smk,kn->smn", f_theta_cos, sinnv, optimize=True) * dmult[None, :, :]

    tmp_su_1 = jnp.einsum("smn,im->sin", bsubsmn1, cosmum, optimize=True)
    tmp_su_2 = jnp.einsum("smn,im->sin", bsubsmn2, sinmum, optimize=True)
    bsubsu = jnp.einsum("sin,kn->sik", tmp_su_1, cosnv, optimize=True) + jnp.einsum(
        "sin,kn->sik", tmp_su_2, sinnv, optimize=True
    )

    tmp_sv_1 = jnp.einsum("smn,im->sin", bsubsmn1, sinmu, optimize=True)
    tmp_sv_2 = jnp.einsum("smn,im->sin", bsubsmn2, cosmu, optimize=True)
    bsubsv = jnp.einsum("sin,kn->sik", tmp_sv_1, sinnvn, optimize=True) + jnp.einsum(
        "sin,kn->sik", tmp_sv_2, cosnvn, optimize=True
    )
    return {"bsubsu": bsubsu, "bsubsv": bsubsv}


def mercier_bsubs_half_mesh_from_geometry(
    *,
    bsupu,
    bsupv,
    rs12,
    zs12,
    ru12,
    zu12,
    rv12,
    zv12,
) -> dict[str, Any]:
    """Return VMEC half-mesh ``bsubs`` from geometry and contravariant B.

    This is the differentiable core of VMEC's ``bss.f`` radial covariant field
    assembly once the half-mesh geometric channels have been synthesized:

    ``B_s = B^u (R_s R_u + Z_s Z_u) + B^v (R_s R_v + Z_s Z_v)``.
    """
    bsupu = jnp.asarray(bsupu, dtype=jnp.float64)
    bsupv = jnp.asarray(bsupv, dtype=jnp.float64)
    rs12 = jnp.asarray(rs12, dtype=jnp.float64)
    zs12 = jnp.asarray(zs12, dtype=jnp.float64)
    ru12 = jnp.asarray(ru12, dtype=jnp.float64)
    zu12 = jnp.asarray(zu12, dtype=jnp.float64)
    rv12 = jnp.asarray(rv12, dtype=jnp.float64)
    zv12 = jnp.asarray(zv12, dtype=jnp.float64)

    g_su = rs12 * ru12 + zs12 * zu12
    g_sv = rs12 * rv12 + zs12 * zv12
    bsubs = bsupu * g_su + bsupv * g_sv
    return {"bsubs": bsubs, "g_su": g_su, "g_sv": g_sv}


def mercier_zeta_half_mesh_from_realspace_geometry(
    *,
    s,
    Rv_even,
    Rv_odd,
    Zv_even,
    Zv_odd,
) -> dict[str, Any]:
    """Return VMEC half-mesh ``rv12``/``zv12`` from parity geometry channels."""
    s = jnp.asarray(s, dtype=jnp.float64)
    Rv_even = jnp.asarray(Rv_even, dtype=jnp.float64)
    Rv_odd = jnp.asarray(Rv_odd, dtype=jnp.float64)
    Zv_even = jnp.asarray(Zv_even, dtype=jnp.float64)
    Zv_odd = jnp.asarray(Zv_odd, dtype=jnp.float64)

    zeros = jnp.zeros_like(Rv_even, dtype=jnp.float64)
    ns = int(s.shape[0])
    if ns < 2:
        return {"rv12": zeros, "zv12": zeros}

    sh = jnp.sqrt(jnp.maximum(0.5 * (s[1:] + s[:-1]), 0.0))[:, None, None]
    rv_inner = 0.5 * (Rv_even[1:] + Rv_even[:-1] + sh * (Rv_odd[1:] + Rv_odd[:-1]))
    zv_inner = 0.5 * (Zv_even[1:] + Zv_even[:-1] + sh * (Zv_odd[1:] + Zv_odd[:-1]))
    rv12 = zeros.at[1:].set(rv_inner)
    zv12 = zeros.at[1:].set(zv_inner)
    rv12 = rv12.at[0].set(rv_inner[0])
    zv12 = zv12.at[0].set(zv_inner[0])
    return {"rv12": rv12, "zv12": zv12}


def mercier_bsubs_full_mesh_from_half_mesh(*, bsubs_half) -> Any:
    """Average half-mesh ``bsubs`` to VMEC's jxbforce full-mesh convention."""
    bsubs_half = jnp.asarray(bsubs_half, dtype=jnp.float64)
    ns = int(bsubs_half.shape[0])
    bsubs_full = jnp.array(bsubs_half)
    if ns > 2:
        bsubs_full = bsubs_full.at[1:-1].set(0.5 * (bsubs_half[1:-1] + bsubs_half[2:]))
    if ns > 0:
        bsubs_full = bsubs_full.at[0].set(jnp.zeros_like(bsubs_full[0]))
    return bsubs_full


def mercier_bdotk_from_covariant_derivatives(
    *,
    bsubu,
    bsubv,
    bsubsu,
    bsubsv,
    s,
) -> dict[str, Any]:
    """Return VMEC Mercier ``bdotk`` channels from covariant field derivatives.

    This is the JAX equivalent of the small jxbforce block that forms
    ``itheta``, ``izeta``, ``bdotk``, and ``bdotk_merc`` once the filtered
    covariant fields and their angular derivatives are available.
    """
    bsubu = jnp.asarray(bsubu, dtype=jnp.float64)
    bsubv = jnp.asarray(bsubv, dtype=jnp.float64)
    bsubsu = jnp.asarray(bsubsu, dtype=jnp.float64)
    bsubsv = jnp.asarray(bsubsv, dtype=jnp.float64)
    s = jnp.asarray(s, dtype=jnp.float64)
    ns = int(s.shape[0])
    zeros = jnp.zeros_like(bsubu, dtype=jnp.float64)
    if ns < 3:
        return {
            "itheta": zeros,
            "izeta": zeros,
            "bdotk": zeros,
            "bdotk_merc": zeros,
        }

    hs = jnp.asarray(1.0 / float(ns - 1), dtype=jnp.float64)
    ohs = 1.0 / hs
    itheta_inner = bsubsv[1:-1] - ohs * (bsubv[2:] - bsubv[1:-1])
    izeta_inner = -bsubsu[1:-1] + ohs * (bsubu[2:] - bsubu[1:-1])
    itheta = zeros.at[1:-1].set(itheta_inner)
    izeta = zeros.at[1:-1].set(izeta_inner)
    izeta = izeta.at[0].set(2.0 * izeta[1] - izeta[2])
    izeta = izeta.at[-1].set(2.0 * izeta[-2] - izeta[-3])

    itheta = itheta / jnp.asarray(MU0, dtype=jnp.float64)
    izeta = izeta / jnp.asarray(MU0, dtype=jnp.float64)
    bsubu1 = 0.5 * (bsubu[2:] + bsubu[1:-1])
    bsubv1 = 0.5 * (bsubv[2:] + bsubv[1:-1])
    bdotk_inner = itheta[1:-1] * bsubu1 + izeta[1:-1] * bsubv1
    bdotk = zeros.at[1:-1].set(bdotk_inner)
    bdotk_merc = jnp.asarray(MU0, dtype=jnp.float64) * bdotk
    return {
        "itheta": itheta,
        "izeta": izeta,
        "bdotk": bdotk,
        "bdotk_merc": bdotk_merc,
    }


def _s_half_from_static(static):
    s = jnp.asarray(static.s)
    if int(s.shape[0]) < 2:
        return s
    return jnp.concatenate([s[:1], 0.5 * (s[1:] + s[:-1])], axis=0)


def _wout_like_for_state(*, state, static, indata, signgs: int):
    s = jnp.asarray(static.s)
    flux = flux_profiles_from_indata(indata, s, signgs=int(signgs))
    phips = jnp.asarray(flux.phips)
    if int(phips.shape[0]) > 0:
        phips = phips.at[0].set(0.0)

    s_half = _s_half_from_static(static)
    prof = eval_profiles(indata, s_half)
    pres = jnp.asarray(prof.get("pressure", jnp.zeros_like(s_half)))
    if int(pres.shape[0]) > 0:
        pres = pres.at[0].set(0.0)

    chips, iotas, iotaf = equilibrium_iota_profiles_from_state(
        state=state,
        static=static,
        indata=indata,
        signgs=int(signgs),
    )
    chipf = _chipf_from_chips(chips)

    from .boundary import boundary_from_indata

    boundary = boundary_from_indata(indata, static.modes)
    mode_m = np.asarray(static.modes.m)
    mode_n = np.asarray(static.modes.n)
    idx00 = np.where((mode_m == 0) & (mode_n == 0))[0]
    r00 = float(np.asarray(boundary.R_cos)[int(idx00[0])]) if idx00.size else float(np.asarray(boundary.R_cos)[0])

    gamma = float(indata.get_float("GAMMA", 0.0))
    lrfp = bool(indata.get_bool("LRFP", False))
    chips_half = _half_mesh_from_full_mesh(jnp.asarray(flux.chipf)) if lrfp else None
    mass = _mass_half_mesh_from_indata(
        indata=indata,
        s_full=s,
        phips=phips,
        r00=r00,
        gamma=gamma,
        lrfp=lrfp,
        chips=chips_half,
    )
    icurv = _icurv_full_mesh_from_indata(indata=indata, s_full=s, signgs=int(signgs))

    wout_like = SimpleNamespace(
        phipf=jnp.asarray(flux.phipf),
        phips=phips,
        chipf=jnp.asarray(chipf),
        iotaf=jnp.asarray(iotaf),
        iotas=jnp.asarray(iotas),
        signgs=int(signgs),
        nfp=int(static.cfg.nfp),
        mpol=int(static.cfg.mpol),
        ntor=int(static.cfg.ntor),
        lasym=bool(static.cfg.lasym),
        flux_is_internal=True,
        ncurr=int(indata.get_int("NCURR", 0)),
        lcurrent=bool(indata.get_int("NCURR", 0) == 1),
        icurv=jnp.asarray(icurv),
        mass=jnp.asarray(mass),
        gamma=gamma,
    )
    return wout_like, pres


def finite_beta_scalars_from_state(*, state, static, indata, signgs: int) -> dict[str, Any]:
    """Return JAX-differentiable finite-beta scalar diagnostics from a VMEC state."""
    aspect = equilibrium_aspect_ratio_from_state(state=state, static=static)
    _chips, iotas, iotaf = equilibrium_iota_profiles_from_state(
        state=state,
        static=static,
        indata=indata,
        signgs=int(signgs),
    )
    iotaf = jnp.asarray(iotaf, dtype=jnp.float64)

    wout_like, pres = _wout_like_for_state(state=state, static=static, indata=indata, signgs=int(signgs))
    bc = vmec_bcovar_half_mesh_from_wout(
        state=state,
        static=static,
        wout=wout_like,
        pres=pres,
        use_wout_bsup=False,
        use_wout_bsub_for_lambda=False,
        use_wout_bmag_for_bsq=False,
        use_vmec_synthesis=True,
        trig=None,
    )
    norms = vmec_force_norms_from_bcovar_dynamic(
        bc=bc,
        trig=static.trig_vmec,
        s=jnp.asarray(static.s),
        signgs=int(signgs),
    )
    beta_total = jnp.where(norms.wb != 0.0, norms.wp / norms.wb, jnp.asarray(0.0, dtype=norms.wb.dtype))
    volavgB = jnp.sqrt(jnp.maximum(2.0 * norms.wb / jnp.maximum(norms.volume, 1e-300), 0.0))
    return {
        "aspect": aspect,
        "iotas": jnp.asarray(iotas, dtype=jnp.float64),
        "iotaf": iotaf,
        "mean_iota": jnp.mean(jnp.abs(iotaf[1:])) if int(iotaf.shape[0]) > 1 else jnp.asarray(0.0),
        "min_iota": jnp.min(jnp.abs(iotaf[1:])) if int(iotaf.shape[0]) > 1 else jnp.asarray(0.0),
        "max_iota": jnp.max(jnp.abs(iotaf[1:])) if int(iotaf.shape[0]) > 1 else jnp.asarray(0.0),
        "volavgB": volavgB,
        "betatotal": beta_total,
        "wb": norms.wb,
        "wp": norms.wp,
        "vp": getattr(norms, "vp", jnp.zeros_like(jnp.asarray(static.s))),
        "volume": norms.volume,
    }


def finite_beta_global_residuals_from_state(
    *,
    state,
    static,
    indata,
    signgs: int,
    targets: FiniteBetaTargets,
) -> jnp.ndarray:
    """Build global finite-beta residuals for stage-one surface optimization."""
    scalars = finite_beta_scalars_from_state(state=state, static=static, indata=indata, signgs=int(signgs))
    aspect_res = jnp.maximum(scalars["aspect"] - float(targets.aspect_ratio), 0.0)
    min_iota_res = jnp.minimum(scalars["min_iota"] - float(targets.min_iota), 0.0)
    mean_iota_res = jnp.minimum(scalars["mean_iota"] - float(targets.min_average_iota), 0.0)
    max_iota_res = jnp.maximum(scalars["max_iota"] - float(targets.max_iota), 0.0)
    return jnp.asarray(
        [
            float(targets.aspect_weight) * aspect_res,
            float(targets.iota_weight) * min_iota_res,
            float(targets.iota_weight) * mean_iota_res,
            float(targets.max_iota_weight) * max_iota_res,
            float(targets.volavgB_weight) * (scalars["volavgB"] - float(targets.volavgB)),
            float(targets.beta_weight) * (scalars["betatotal"] - float(targets.beta_total)),
        ],
        dtype=jnp.float64,
    )
