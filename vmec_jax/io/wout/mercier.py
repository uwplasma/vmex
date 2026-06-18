"""Mercier and JXBFORCE-style WOUT diagnostic reducers."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Callable

import numpy as np

from vmec_jax.state import VMECState
from vmec_jax.vmec_jacobian import _apply_vmec_axis_rules
from vmec_jax.vmec_parity import vmec_m1_internal_to_physical_signed
from vmec_jax.vmec_realspace import (
    vmec_realspace_synthesis,
    vmec_realspace_synthesis_dtheta,
    vmec_realspace_synthesis_dzeta_phys,
)

MU0 = 4e-7 * np.pi


@dataclass(frozen=True)
class _MercierWeightedSumContext:
    """Surface-quadrature policy used by Mercier/JXBFORCE reducers."""

    wint: np.ndarray
    ntheta: int
    nzeta: int
    exact_sum: bool
    sum_w: Callable[[np.ndarray], float]
    wint_f: np.ndarray | None


def _require_dependency(name: str, value):
    if value is None:
        raise TypeError(f"compute_mercier requires dependency {name}")
    return value


def _mercier_radial_stability_terms(
    *,
    ns: int,
    hs: float,
    s: np.ndarray,
    iotas: np.ndarray,
    pres: np.ndarray,
    phip_real: np.ndarray,
    vp_real: np.ndarray,
    bsubu: np.ndarray,
    bsq: np.ndarray,
    sqrtg: np.ndarray,
    geom_channels: dict[str, np.ndarray],
    bdotk_merc: np.ndarray,
    sign_jac: float,
    sum_context: _MercierWeightedSumContext,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute VMEC Mercier radial terms without changing formula order."""

    R_even, R_odd = geom_channels["R_even"], geom_channels["R_odd"]
    Ru_even, Ru_odd = geom_channels["Ru_even"], geom_channels["Ru_odd"]
    Zu_even, Zu_odd = geom_channels["Zu_even"], geom_channels["Zu_odd"]
    Rv_even, Rv_odd = geom_channels["Rv_even"], geom_channels["Rv_odd"]
    Zv_even, Zv_odd = geom_channels["Zv_even"], geom_channels["Zv_odd"]
    exact_sum, sum_w, wint_f = sum_context.exact_sum, sum_context.sum_w, sum_context.wint_f
    DMerc = np.zeros((ns,), dtype=float)
    Dshear = np.zeros((ns,), dtype=float)
    Dcurr = np.zeros((ns,), dtype=float)
    Dwell = np.zeros((ns,), dtype=float)
    Dgeod = np.zeros((ns,), dtype=float)
    shear = np.zeros((ns,), dtype=float)
    vpp = np.zeros((ns,), dtype=float)
    presp = np.zeros((ns,), dtype=float)
    ip = np.zeros((ns,), dtype=float)
    torcur = np.zeros((ns,), dtype=float)
    if ns > 1:
        if exact_sum:
            for i in range(1, ns):
                torcur[i] = sign_jac * (2.0 * np.pi) * sum_w(bsubu[i])
        else:
            if wint_f is None:
                raise ValueError("wint_f is required for vectorized Mercier summation")
            torcur[1:] = sign_jac * (2.0 * np.pi) * np.einsum("sij,ij->s", bsubu[1:], wint_f, optimize=True)
    if ns > 2:
        phip_full = 0.5 * (phip_real[2:] + phip_real[1:-1])
        denom = 1.0 / (hs * phip_full)
        shear[1:-1] = (iotas[2:] - iotas[1:-1]) * denom
        vpp[1:-1] = (vp_real[2:] - vp_real[1:-1]) * denom
        presp[1:-1] = (pres[2:] - pres[1:-1]) * denom
        ip[1:-1] = (torcur[2:] - torcur[1:-1]) * denom

    b2 = 2.0 * (bsq - pres[:, None, None])
    for i in range(1, ns - 1):
        phip_full = 0.5 * (phip_real[i + 1] + phip_real[i])
        gsqrt_raw = 0.5 * (sqrtg[i] + sqrtg[i + 1])
        gsqrt_full = gsqrt_raw / phip_full
        sqs = float(np.sqrt(s[i]))
        r1f = R_even[i] + sqs * R_odd[i]
        rtf = Ru_even[i] + sqs * Ru_odd[i]
        ztf = Zu_even[i] + sqs * Zu_odd[i]
        rzf = Rv_even[i] + sqs * Rv_odd[i]
        zzf = Zv_even[i] + sqs * Zv_odd[i]
        gtt = rtf * rtf + ztf * ztf
        gpp = (gsqrt_full * gsqrt_full) / (gtt * r1f * r1f + (rtf * zzf - rzf * ztf) ** 2)
        b2i = 0.5 * (b2[i + 1] + b2[i])
        ob2 = gsqrt_full / b2i
        tpp = sum_w(ob2)
        ob2 = b2i * gsqrt_full * gpp
        tbb = sum_w(ob2)
        # VMEC divides bdotj by the raw Jacobian (before phip scaling),
        # then uses the flux-normalized Jacobian in jdotb.
        bdotj_norm = np.where(gsqrt_raw != 0.0, bdotk_merc[i] / gsqrt_raw, 0.0)
        jdotb = bdotj_norm * gpp * gsqrt_full
        tjb = sum_w(jdotb)
        jdotb2 = jdotb * bdotj_norm / b2i
        tjj = sum_w(jdotb2)

        tpp *= (2.0 * np.pi) ** 2
        tjb *= (2.0 * np.pi) ** 2
        tbb *= (2.0 * np.pi) ** 2
        tjj *= (2.0 * np.pi) ** 2
        dshear = 0.25 * shear[i] * shear[i]
        dcurr = -shear[i] * (tjb - ip[i] * tbb)
        dwell = presp[i] * (vpp[i] - presp[i] * tpp) * tbb
        dgeod = tjb * tjb - tbb * tjj
        Dshear[i] = dshear
        Dcurr[i] = dcurr
        Dwell[i] = dwell
        Dgeod[i] = dgeod
        DMerc[i] = dshear + dcurr + dwell + dgeod

    return DMerc, Dshear, Dcurr, Dwell, Dgeod


def _jxbforce_1d_current_diagnostics(
    *,
    ns: int,
    vp: np.ndarray,
    phips: np.ndarray,
    sqrtg: np.ndarray,
    bsq: np.ndarray,
    pres: np.ndarray,
    bdotk: np.ndarray,
    sigma_an: np.ndarray,
    sign_jac: float,
    sum_w: Callable[[np.ndarray], float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return VMEC/JXBFORCE 1D current and field-line diagnostics."""

    jdotb = np.zeros((ns,), dtype=float)
    bdotb = np.zeros((ns,), dtype=float)
    bdotgradv = np.zeros((ns,), dtype=float)
    dnorm1 = float((2.0 * np.pi) ** 2)
    vp = np.asarray(vp, dtype=float)
    phips = np.asarray(phips, dtype=float)
    for js in range(1, ns - 1):
        denom = vp[js + 1] + vp[js]
        if denom == 0.0:
            continue
        ovp = 2.0 / denom / dnorm1
        tjnorm = ovp * float(sign_jac)
        sqgb2 = sqrtg[js + 1] * (bsq[js + 1] - pres[js + 1]) + sqrtg[js] * (bsq[js] - pres[js])
        sigma = sigma_an[js]
        jdotb[js] = dnorm1 * tjnorm * sum_w(bdotk[js] / sigma)
        bdotb[js] = dnorm1 * tjnorm * sum_w(sqgb2 / sigma)
        bdotgradv[js] = 0.5 * dnorm1 * tjnorm * (phips[js] + phips[js + 1])
    if ns > 2:
        jdotb[0] = 2.0 * jdotb[1] - jdotb[2]
        jdotb[-1] = 2.0 * jdotb[-2] - jdotb[-3]
        bdotb[0] = 2.0 * bdotb[2] - bdotb[1]
        bdotb[-1] = 2.0 * bdotb[-2] - bdotb[-3]
        bdotgradv[0] = 2.0 * bdotgradv[1] - bdotgradv[2]
        bdotgradv[-1] = 2.0 * bdotgradv[-2] - bdotgradv[-3]
    return jdotb, bdotb, bdotgradv


def _mercier_weighted_sum_context(*, trig, vmec_wint_from_trig) -> _MercierWeightedSumContext:
    """Return the VMEC-compatible weighted surface-sum implementation."""
    wint = vmec_wint_from_trig(trig)
    nzeta = int(wint.shape[1])
    ntheta = int(wint.shape[0])
    exact_sum = os.getenv("VMEC_JAX_MERCIER_EXACT_SUM", "").strip().lower() not in ("", "0", "false", "no")
    if exact_sum:
        # Match VMEC2000 Fortran summation order on cancellation-sensitive
        # surface averages (slow; intended for parity debugging).
        def _sum_w(arr: np.ndarray) -> float:
            acc = 0.0
            for j in range(ntheta):
                wrow = wint[j]
                arow = arr[j]
                for k in range(nzeta):
                    acc += float(arow[k]) * float(wrow[k])
            return acc

        wint_f = None
    else:
        wint_f = np.asarray(wint, dtype=float)

        def _sum_w(arr: np.ndarray) -> float:
            # Fast path: vectorized weighted sum on the reduced (theta,zeta) grid.
            return float(np.einsum("ij,ij->", np.asarray(arr, dtype=float), wint_f, optimize=True))

    return _MercierWeightedSumContext(
        wint=np.asarray(wint, dtype=float),
        ntheta=int(ntheta),
        nzeta=int(nzeta),
        exact_sum=bool(exact_sum),
        sum_w=_sum_w,
        wint_f=None if wint_f is None else np.asarray(wint_f, dtype=float),
    )


def _mercier_parity_geometry_fields(
    *,
    state: VMECState,
    geom_modes,
    trig,
    s: np.ndarray,
    lconm1: bool,
    lthreed: bool,
    lasym: bool,
) -> dict[str, np.ndarray]:
    """Synthesize even/odd geometry channels used by VMEC Mercier terms."""
    m = np.asarray(geom_modes.m)
    mask_even = (m % 2) == 0
    mask_odd = np.logical_not(mask_even)
    Rcos = np.asarray(state.Rcos)
    Rsin = np.asarray(state.Rsin)
    Zcos = np.asarray(state.Zcos)
    Zsin = np.asarray(state.Zsin)
    if bool(lconm1):
        Rcos, Zsin, Rsin, Zcos = vmec_m1_internal_to_physical_signed(
            Rcos=Rcos,
            Zsin=Zsin,
            Rsin=Rsin,
            Zcos=Zcos,
            modes=geom_modes,
            lthreed=bool(lthreed),
            lasym=bool(lasym),
            lconm1=bool(lconm1),
        )
    Rcos = _apply_vmec_axis_rules(Rcos, m)
    Rsin = _apply_vmec_axis_rules(Rsin, m)
    Zcos = _apply_vmec_axis_rules(Zcos, m)
    Zsin = _apply_vmec_axis_rules(Zsin, m)

    coeff_cos_stack = np.stack([Rcos, Zcos], axis=0)
    coeff_sin_stack = np.stack([Rsin, Zsin], axis=0)
    mask_stack = np.stack([mask_even.astype(float), mask_odd.astype(float)], axis=0)
    coeff_cos = coeff_cos_stack[None, ...] * mask_stack[:, None, None, :]
    coeff_sin = coeff_sin_stack[None, ...] * mask_stack[:, None, None, :]

    stack = vmec_realspace_synthesis(
        coeff_cos=coeff_cos,
        coeff_sin=coeff_sin,
        modes=geom_modes,
        trig=trig,
        coeffs_internal=True,
        apply_scalxc=True,
        s=s,
    )
    stack_t = vmec_realspace_synthesis_dtheta(
        coeff_cos=coeff_cos,
        coeff_sin=coeff_sin,
        modes=geom_modes,
        trig=trig,
        coeffs_internal=True,
        apply_scalxc=True,
        s=s,
    )
    stack_p = vmec_realspace_synthesis_dzeta_phys(
        coeff_cos=coeff_cos,
        coeff_sin=coeff_sin,
        modes=geom_modes,
        trig=trig,
        coeffs_internal=True,
        apply_scalxc=True,
        s=s,
    )
    even = np.asarray(stack[0])
    odd = np.asarray(stack[1])
    even_t = np.asarray(stack_t[0])
    odd_t = np.asarray(stack_t[1])
    even_p = np.asarray(stack_p[0])
    odd_p = np.asarray(stack_p[1])

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


def _mercier_preprocess_bsubuv(
    *,
    bsubu: np.ndarray,
    bsubv: np.ndarray,
    lasym: bool,
    trig,
    s: np.ndarray,
    mmax_force: int,
    nmax_force: int,
    bsubu_parity_even: np.ndarray | None,
    bsubu_parity_odd: np.ndarray | None,
    bsubv_parity_even: np.ndarray | None,
    bsubv_parity_odd: np.ndarray | None,
    symoutput_expand,
    filter_bsubuv_jxbforce_lasym_loop,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare bsubu/bsubv on the grid expected by VMEC Mercier/JXBFORCE."""

    bsubu_full = np.asarray(bsubu, dtype=float)
    bsubv_full = np.asarray(bsubv, dtype=float)
    nt2 = int(trig.ntheta2)
    nzeta = int(np.asarray(trig.cosnv).shape[0])
    bsubu = np.asarray(bsubu, dtype=float)
    bsubv = np.asarray(bsubv, dtype=float)
    if bool(lasym):
        nt3 = int(getattr(trig, "ntheta3", nt2))
        if bsubu.shape[1] == nt2 and nt3 > nt2:
            bsubu = symoutput_expand(sym=bsubu, asym=None, trig=trig)
        if bsubv.shape[1] == nt2 and nt3 > nt2:
            bsubv = symoutput_expand(sym=bsubv, asym=None, trig=trig)
        if os.getenv("VMEC_JAX_MERCIER_LASYM_FILTER", "1") not in ("", "0"):
            use_parity_channels = os.getenv("VMEC_JAX_LASYM_FILTER_USE_PARITY_CHANNELS", "0") not in (
                "",
                "0",
                "false",
                "no",
            )
            bsubu, bsubv = filter_bsubuv_jxbforce_lasym_loop(
                bsubu=bsubu,
                bsubv=bsubv,
                trig=trig,
                mmax_force=max(int(mmax_force), 0),
                nmax_force=max(int(nmax_force), 0),
                s=np.asarray(s, dtype=float),
                bsubu_even=None
                if (not use_parity_channels) or (bsubu_parity_even is None)
                else np.asarray(bsubu_parity_even, dtype=float),
                bsubu_odd=None
                if (not use_parity_channels) or (bsubu_parity_odd is None)
                else np.asarray(bsubu_parity_odd, dtype=float),
                bsubv_even=None
                if (not use_parity_channels) or (bsubv_parity_even is None)
                else np.asarray(bsubv_parity_even, dtype=float),
                bsubv_odd=None
                if (not use_parity_channels) or (bsubv_parity_odd is None)
                else np.asarray(bsubv_parity_odd, dtype=float),
            )
        return bsubu, bsubv, bsubu_full, bsubv_full

    if nt2 <= 0:
        return bsubu, bsubv, bsubu_full, bsubv_full
    if bsubu.shape[1] > nt2:
        nt1 = int(trig.ntheta1)
        i0 = np.arange(nt2, dtype=int)
        ir0 = np.where(i0 == 0, 0, nt1 - i0)
        kk = (nzeta - np.arange(nzeta, dtype=int)) % nzeta

        def _sym_half(arr: np.ndarray) -> np.ndarray:
            a_half = arr[:, :nt2, :]
            a_ref = arr[:, ir0, :][:, :, kk]
            return 0.5 * (a_half + a_ref)

        return _sym_half(bsubu), _sym_half(bsubv), bsubu_full, bsubv_full
    return bsubu[:, :nt2, :], bsubv[:, :nt2, :], bsubu_full, bsubv_full


def _jxbforce_bsubs_derivatives(
    *,
    bsubs: np.ndarray,
    bsubu: np.ndarray,
    bsubv: np.ndarray,
    ns: int,
    s: np.ndarray,
    lasym: bool,
    trig,
    mmax_force: int,
    nmax_force: int,
    bsubu_parity_even: np.ndarray | None,
    bsubu_parity_odd: np.ndarray | None,
    bsubv_parity_even: np.ndarray | None,
    bsubv_parity_odd: np.ndarray | None,
    pshalf_from_s,
    jxbforce_filter_with_bsubs_derivs_loop,
    jxbforce_bsubsu_bsubsv_loop,
    symoutput_split,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct VMEC/JXBFORCE bsubs derivatives on the Mercier grid."""

    mmax, nmax = int(mmax_force), int(nmax_force)
    nt2 = int(trig.ntheta2)
    bsubs_use = np.asarray(bsubs, dtype=float).copy()
    if int(ns) > 2:
        bsubs_use[1:-1] = 0.5 * (bsubs_use[1:-1] + bsubs_use[2:])
    bsubs_use[0] = 0.0

    use_coupled = (not bool(lasym)) and os.getenv("VMEC_JAX_JXBFORCE_COUPLED_BSUB", "0") not in (
        "",
        "0",
        "false",
        "no",
    )
    if use_coupled:
        psh = pshalf_from_s(np.asarray(s, dtype=float))[:, None, None]
        if psh.shape[0] > 1:
            psh[0] = psh[1]
        use_bc_parity = os.getenv("VMEC_JAX_JXBFORCE_COUPLED_USE_BC_PARITY", "0") not in ("", "0", "false", "no")
        bu_even = np.asarray(bsubu, dtype=float)
        bv_even = np.asarray(bsubv, dtype=float)
        bu_odd = psh * bu_even
        bv_odd = psh * bv_even
        if use_bc_parity:
            if bsubu_parity_even is not None and np.asarray(bsubu_parity_even).shape == bu_even.shape:
                bu_even = np.asarray(bsubu_parity_even, dtype=float)
            if bsubv_parity_even is not None and np.asarray(bsubv_parity_even).shape == bv_even.shape:
                bv_even = np.asarray(bsubv_parity_even, dtype=float)
            if bsubu_parity_odd is not None and np.asarray(bsubu_parity_odd).shape == bu_even.shape:
                bu_odd = np.asarray(bsubu_parity_odd, dtype=float)
            if bsubv_parity_odd is not None and np.asarray(bsubv_parity_odd).shape == bv_even.shape:
                bv_odd = np.asarray(bsubv_parity_odd, dtype=float)
        bsubu, bsubv, bsubsu, bsubsv = jxbforce_filter_with_bsubs_derivs_loop(
            bsubs=bsubs_use,
            bsubu_even=bu_even,
            bsubu_odd=bu_odd,
            bsubv_even=bv_even,
            bsubv_odd=bv_odd,
            trig=trig,
            mmax_force=mmax,
            nmax_force=nmax,
            s=np.asarray(s, dtype=float),
        )
        return bsubs_use, bsubu, bsubv, bsubsu, bsubsv

    if (not bool(lasym)) and os.getenv("VMEC_JAX_JXBFORCE_LOOP", "0") not in ("", "0"):
        bsubsu, bsubsv = jxbforce_bsubsu_bsubsv_loop(bsubs=bsubs_use, trig=trig, mmax_force=mmax, nmax_force=nmax)
        return bsubs_use, np.asarray(bsubu, dtype=float), np.asarray(bsubv, dtype=float), bsubsu, bsubsv

    cosmui = np.asarray(trig.cosmui, dtype=float)[:nt2, : mmax + 1]
    sinmui = np.asarray(trig.sinmui, dtype=float)[:nt2, : mmax + 1]
    cosmu = np.asarray(trig.cosmu, dtype=float)[:nt2, : mmax + 1]
    sinmu = np.asarray(trig.sinmu, dtype=float)[:nt2, : mmax + 1]
    cosmum = np.asarray(trig.cosmum, dtype=float)[:nt2, : mmax + 1]
    sinmum = np.asarray(trig.sinmum, dtype=float)[:nt2, : mmax + 1]
    cosnv = np.asarray(trig.cosnv, dtype=float)[:, : nmax + 1]
    sinnv = np.asarray(trig.sinnv, dtype=float)[:, : nmax + 1]
    cosnvn = np.asarray(trig.cosnvn, dtype=float)[:, : nmax + 1]
    sinnvn = np.asarray(trig.sinnvn, dtype=float)[:, : nmax + 1]
    dmult = np.full((mmax + 1, nmax + 1), 1.0 / float(getattr(trig, "r0scale", 1.0)) ** 2, dtype=float)
    mnyq = np.asarray(trig.cosmui, dtype=float).shape[1] - 1
    nnyq = np.asarray(trig.cosnv, dtype=float).shape[1] - 1
    if mnyq > 0 and mnyq <= mmax:
        dmult[mnyq, :] *= 0.5
    if nnyq > 0 and nnyq <= nmax:
        dmult[:, nnyq] *= 0.5

    if bool(lasym):
        bsubs_sym, bsubs_asym = symoutput_split(f=bsubs_use, trig=trig, reversed_sym=True)
        f_theta_sin = np.einsum("sik,im->smk", bsubs_sym[:, :nt2, :], sinmui, optimize=True)
        f_theta_cos = np.einsum("sik,im->smk", bsubs_sym[:, :nt2, :], cosmui, optimize=True)
        bsubsmn1 = np.einsum("smk,kn->smn", f_theta_sin, cosnv, optimize=True) * dmult[None, :, :]
        bsubsmn2 = np.einsum("smk,kn->smn", f_theta_cos, sinnv, optimize=True) * dmult[None, :, :]
        bsubsu_s = np.einsum("sin,kn->sik", np.einsum("smn,im->sin", bsubsmn1, cosmum, optimize=True),
                             cosnv, optimize=True) + np.einsum(
            "sin,kn->sik", np.einsum("smn,im->sin", bsubsmn2, sinmum, optimize=True), sinnv, optimize=True
        )
        bsubsv_s = np.einsum("sin,kn->sik", np.einsum("smn,im->sin", bsubsmn1, sinmu, optimize=True),
                             sinnvn, optimize=True) + np.einsum(
            "sin,kn->sik", np.einsum("smn,im->sin", bsubsmn2, cosmu, optimize=True), cosnvn, optimize=True
        )
        f_theta_cos_a = np.einsum("sik,im->smk", bsubs_asym[:, :nt2, :], cosmui, optimize=True)
        f_theta_sin_a = np.einsum("sik,im->smk", bsubs_asym[:, :nt2, :], sinmui, optimize=True)
        bsubsmn3 = np.einsum("smk,kn->smn", f_theta_cos_a, cosnv, optimize=True) * dmult[None, :, :]
        bsubsmn4 = np.einsum("smk,kn->smn", f_theta_sin_a, sinnv, optimize=True) * dmult[None, :, :]
        bsubsu_a = np.einsum("sin,kn->sik", np.einsum("smn,im->sin", bsubsmn3, sinmum, optimize=True),
                             cosnv, optimize=True) + np.einsum(
            "sin,kn->sik", np.einsum("smn,im->sin", bsubsmn4, cosmum, optimize=True), sinnv, optimize=True
        )
        bsubsv_a = np.einsum("sin,kn->sik", np.einsum("smn,im->sin", bsubsmn3, cosmu, optimize=True),
                             sinnvn, optimize=True) + np.einsum(
            "sin,kn->sik", np.einsum("smn,im->sin", bsubsmn4, sinmu, optimize=True), cosnvn, optimize=True
        )
        bsubsu = _extend_mercier_parity_to_full(bsubsu_s, bsubsu_a, trig=trig)
        bsubsv = _extend_mercier_parity_to_full(bsubsv_s, bsubsv_a, trig=trig)
        return bsubs_use, np.asarray(bsubu, dtype=float), np.asarray(bsubv, dtype=float), bsubsu, bsubsv

    f_theta_sin = np.einsum("sik,im->smk", bsubs_use[:, :nt2, :], sinmui, optimize=True)
    f_theta_cos = np.einsum("sik,im->smk", bsubs_use[:, :nt2, :], cosmui, optimize=True)
    bsubsmn1 = np.einsum("smk,kn->smn", f_theta_sin, cosnv, optimize=True) * dmult[None, :, :]
    bsubsmn2 = np.einsum("smk,kn->smn", f_theta_cos, sinnv, optimize=True) * dmult[None, :, :]
    bsubsu = np.einsum("sin,kn->sik", np.einsum("smn,im->sin", bsubsmn1, cosmum, optimize=True),
                       cosnv, optimize=True) + np.einsum(
        "sin,kn->sik", np.einsum("smn,im->sin", bsubsmn2, sinmum, optimize=True), sinnv, optimize=True
    )
    bsubsv = np.einsum("sin,kn->sik", np.einsum("smn,im->sin", bsubsmn1, sinmu, optimize=True),
                       sinnvn, optimize=True) + np.einsum(
        "sin,kn->sik", np.einsum("smn,im->sin", bsubsmn2, cosmu, optimize=True), cosnvn, optimize=True
    )
    return bsubs_use, np.asarray(bsubu, dtype=float), np.asarray(bsubv, dtype=float), bsubsu, bsubsv


def _extend_mercier_parity_to_full(par0: np.ndarray, par1: np.ndarray, *, trig) -> np.ndarray:
    nt1 = int(trig.ntheta1)
    nt2 = int(trig.ntheta2)
    nt3 = int(getattr(trig, "ntheta3", nt2))
    nzeta = int(np.asarray(par0).shape[2])
    full = np.zeros((par0.shape[0], nt3, nzeta), dtype=float)
    full[:, :nt2, :] = par0 + par1
    if nt3 == nt2:
        return full
    i0 = np.arange(nt2, dtype=int)
    ir0 = np.where(i0 == 0, 0, nt1 - i0)
    kk = (nzeta - np.arange(nzeta, dtype=int)) % nzeta
    mask = ir0 >= nt2
    if np.any(mask):
        full[:, ir0[mask], :] = par0[:, mask, :][:, :, kk] - par1[:, mask, :][:, :, kk]
    return full


def compute_mercier(
    *,
    state: VMECState,
    geom_modes,
    s: np.ndarray,
    lconm1: bool,
    lthreed: bool,
    lasym: bool,
    nfp: int,
    lbsubs: bool,
    mmax_force: int,
    nmax_force: int,
    pres: np.ndarray,
    vp: np.ndarray,
    phips: np.ndarray,
    iotas: np.ndarray,
    bsq: np.ndarray,
    sqrtg: np.ndarray,
    bsubu: np.ndarray,
    bsubv: np.ndarray,
    bsubu_parity_even: np.ndarray | None = None,
    bsubu_parity_odd: np.ndarray | None = None,
    bsubv_parity_even: np.ndarray | None = None,
    bsubv_parity_odd: np.ndarray | None = None,
    bsupu: np.ndarray,
    bsupv: np.ndarray,
    trig,
    geom: dict[str, Any],
    jac_half: Any | None = None,
    force_rs: np.ndarray | None = None,
    force_zs: np.ndarray | None = None,
    force_ru12: np.ndarray | None = None,
    force_zu12: np.ndarray | None = None,
    sigma_an: np.ndarray | None = None,
    bsubu_raw: np.ndarray | None = None,
    bsubv_raw: np.ndarray | None = None,
    signgs: int | None = None,
    compute_bsubs_half_mesh=None,
    symoutput_expand=None,
    filter_bsubuv_jxbforce_lasym_loop=None,
    pshalf_from_s=None,
    jxbforce_filter_with_bsubs_derivs_loop=None,
    jxbforce_bsubsu_bsubsv_loop=None,
    symoutput_split=None,
    vmec_wint_from_trig=None,
    jxbforce_apply_bsubs_correction_lasym_true=None,
    jxbforce_apply_bsubs_correction_lasym_false=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Port of VMEC mercier.f to compute Mercier and jxbforce-style scalars."""
    compute_bsubs_half_mesh = _require_dependency("compute_bsubs_half_mesh", compute_bsubs_half_mesh)
    symoutput_expand = _require_dependency("symoutput_expand", symoutput_expand)
    filter_bsubuv_jxbforce_lasym_loop = _require_dependency(
        "filter_bsubuv_jxbforce_lasym_loop", filter_bsubuv_jxbforce_lasym_loop
    )
    pshalf_from_s = _require_dependency("pshalf_from_s", pshalf_from_s)
    jxbforce_filter_with_bsubs_derivs_loop = _require_dependency(
        "jxbforce_filter_with_bsubs_derivs_loop", jxbforce_filter_with_bsubs_derivs_loop
    )
    jxbforce_bsubsu_bsubsv_loop = _require_dependency("jxbforce_bsubsu_bsubsv_loop", jxbforce_bsubsu_bsubsv_loop)
    symoutput_split = _require_dependency("symoutput_split", symoutput_split)
    vmec_wint_from_trig = _require_dependency("vmec_wint_from_trig", vmec_wint_from_trig)
    jxbforce_apply_bsubs_correction_lasym_true = _require_dependency(
        "jxbforce_apply_bsubs_correction_lasym_true", jxbforce_apply_bsubs_correction_lasym_true
    )
    jxbforce_apply_bsubs_correction_lasym_false = _require_dependency(
        "jxbforce_apply_bsubs_correction_lasym_false", jxbforce_apply_bsubs_correction_lasym_false
    )
    ns = int(pres.shape[0])
    if ns < 3:
        zero = np.zeros((ns,), dtype=float)
        return tuple(zero.copy() for _ in range(8))
    hs = 1.0 / float(ns - 1)

    sqrtg = np.asarray(sqrtg, dtype=float)
    if sigma_an is None:
        sigma_an = np.ones_like(sqrtg)
    else:
        sigma_an = np.asarray(sigma_an, dtype=float)
    if signgs is not None and int(signgs) != 0:
        sign_jac = float(np.sign(float(signgs)))
    else:
        sign_jac = float(np.sign(sqrtg[-1, 0, 0])) if float(sqrtg[-1, 0, 0]) != 0.0 else 1.0
    phip_real = (2.0 * np.pi) * np.asarray(phips, dtype=float) * sign_jac
    vp_real = np.zeros_like(phip_real)
    vp_real[1:] = sign_jac * (2.0 * np.pi) ** 2 * np.asarray(vp[1:], dtype=float) / phip_real[1:]

    sum_context = _mercier_weighted_sum_context(
        trig=trig,
        vmec_wint_from_trig=vmec_wint_from_trig,
    )
    wint = sum_context.wint
    nzeta = sum_context.nzeta
    exact_sum = sum_context.exact_sum
    _sum_w = sum_context.sum_w

    bsubs = compute_bsubs_half_mesh(
        state=state,
        geom_modes=geom_modes,
        s=s,
        lconm1=bool(lconm1),
        lthreed=bool(lthreed),
        lasym=bool(lasym),
        bsupu=bsupu,
        bsupv=bsupv,
        trig=trig,
        geom=geom,
        jac_half=jac_half,
        force_rs=force_rs,
        force_zs=force_zs,
        force_ru12=force_ru12,
        force_zu12=force_zu12,
        apply_scalxc=os.getenv("VMEC_JAX_BSS_APPLY_SCALXC", "1") not in ("", "0"),
    )

    bsubu, bsubv, bsubu_full, bsubv_full = _mercier_preprocess_bsubuv(
        bsubu=bsubu,
        bsubv=bsubv,
        lasym=bool(lasym),
        trig=trig,
        s=np.asarray(s, dtype=float),
        mmax_force=int(mmax_force),
        nmax_force=int(nmax_force),
        bsubu_parity_even=bsubu_parity_even,
        bsubu_parity_odd=bsubu_parity_odd,
        bsubv_parity_even=bsubv_parity_even,
        bsubv_parity_odd=bsubv_parity_odd,
        symoutput_expand=symoutput_expand,
        filter_bsubuv_jxbforce_lasym_loop=filter_bsubuv_jxbforce_lasym_loop,
    )

    geom_channels = _mercier_parity_geometry_fields(
        state=state,
        geom_modes=geom_modes,
        trig=trig,
        s=s,
        lconm1=bool(lconm1),
        lthreed=bool(lthreed),
        lasym=bool(lasym),
    )

    bsubs_use, bsubu, bsubv, bsubsu, bsubsv = _jxbforce_bsubs_derivatives(
        bsubs=bsubs,
        bsubu=bsubu,
        bsubv=bsubv,
        ns=int(ns),
        s=np.asarray(s, dtype=float),
        lasym=bool(lasym),
        trig=trig,
        mmax_force=int(mmax_force),
        nmax_force=int(nmax_force),
        bsubu_parity_even=bsubu_parity_even,
        bsubu_parity_odd=bsubu_parity_odd,
        bsubv_parity_even=bsubv_parity_even,
        bsubv_parity_odd=bsubv_parity_odd,
        pshalf_from_s=pshalf_from_s,
        jxbforce_filter_with_bsubs_derivs_loop=jxbforce_filter_with_bsubs_derivs_loop,
        jxbforce_bsubsu_bsubsv_loop=jxbforce_bsubsu_bsubsv_loop,
        symoutput_split=symoutput_split,
    )

    bsubu = np.asarray(bsubu, dtype=float)
    bsubv = np.asarray(bsubv, dtype=float)
    if bool(lasym):
        if lbsubs:
            bsubs_use, bsubsu, bsubsv = jxbforce_apply_bsubs_correction_lasym_true(
                bsubu=bsubu,
                bsubv=bsubv,
                bsubs=bsubs_use,
                bsubsu=bsubsu,
                bsubsv=bsubsv,
                bsupu=np.asarray(bsupu, dtype=float),
                bsupv=np.asarray(bsupv, dtype=float),
                sqrtg=np.asarray(sqrtg, dtype=float),
                pres=np.asarray(pres, dtype=float),
                vp=np.asarray(vp, dtype=float),
                hs=float(hs),
                signgs=float(sign_jac),
                trig=trig,
                nfp=int(nfp),
                sum_w=_sum_w,
            )
    elif lbsubs:
        bsubs_use, bsubsu, bsubsv = jxbforce_apply_bsubs_correction_lasym_false(
            bsubu=bsubu,
            bsubv=bsubv,
            bsubs=bsubs_use,
            bsubsu=bsubsu,
            bsubsv=bsubsv,
            bsupu=np.asarray(bsupu, dtype=float),
            bsupv=np.asarray(bsupv, dtype=float),
            sqrtg=np.asarray(sqrtg, dtype=float),
            pres=np.asarray(pres, dtype=float),
            vp=np.asarray(vp, dtype=float),
            hs=float(hs),
            signgs=float(sign_jac),
            trig=trig,
            nfp=int(nfp),
            sum_w=_sum_w,
        )

    if os.getenv("VMEC_JAX_DUMP_JXB_CHANNELS", "") not in ("", "0"):
        from pathlib import Path

        tag = os.getenv("VMEC_JAX_DUMP_TAG", "").strip()
        dump_dir = Path(os.getenv("VMEC_JAX_DUMP_DIR", ".")).expanduser().resolve()
        dump_dir.mkdir(parents=True, exist_ok=True)
        dump_js = os.getenv("VMEC_JAX_DUMP_JS", "").strip()
        js_sel = int(dump_js) if dump_js not in ("",) else -1
        js_list = range(ns) if js_sel < 0 else [js_sel]
        out = {}
        for js in js_list:
            if js < 0 or js >= ns:
                continue
            key = f"js{js}"
            out[key] = dict(
                bsubs=np.asarray(bsubs_use[js], dtype=float),
                bsubsu0=np.asarray(bsubsu[js], dtype=float),
                bsubsv0=np.asarray(bsubsv[js], dtype=float),
            )
        np.savez(
            dump_dir / f"jxb_channels_wout{('_' + tag) if tag else ''}.npz",
            ntheta2=int(trig.ntheta2),
            ntheta3=int(getattr(trig, "ntheta3", trig.ntheta2)),
            nzeta=int(trig.cosnv.shape[0]),
            **out,
        )
    if os.getenv("VMEC_JAX_DUMP_JXBOUT", "") not in ("", "0"):
        from pathlib import Path

        tag = os.getenv("VMEC_JAX_DUMP_TAG", "").strip()
        dump_dir = Path(os.getenv("VMEC_JAX_DUMP_DIR", ".")).expanduser().resolve()
        dump_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            dump_dir / f"jxbout_jax{('_' + tag) if tag else ''}.npz",
            bsubs=np.asarray(bsubs_use, dtype=float),
            bsubu=np.asarray(bsubu, dtype=float),
            bsubv=np.asarray(bsubv, dtype=float),
            bsubu_full=np.asarray(bsubu_full, dtype=float),
            bsubv_full=np.asarray(bsubv_full, dtype=float),
            ntheta3=int(getattr(trig, "ntheta3", trig.ntheta2)),
            nzeta=int(trig.cosnv.shape[0]),
            ns=int(ns),
        )
    itheta = np.zeros_like(bsubs)
    izeta = np.zeros_like(bsubs)
    ohs = 1.0 / hs
    if ns > 2:
        itheta[1:-1] = bsubsv[1:-1] - ohs * (bsubv[2:] - bsubv[1:-1])
        izeta[1:-1] = -bsubsu[1:-1] + ohs * (bsubu[2:] - bsubu[1:-1])
        izeta[0] = 2.0 * izeta[1] - izeta[2]
        izeta[-1] = 2.0 * izeta[-2] - izeta[-3]
    # Match jxbforce: convert to MKS units (1/mu0 factor).
    mu0 = 4e-7 * np.pi
    itheta = itheta / mu0
    izeta = izeta / mu0
    bdotk = np.zeros_like(bsubs)
    if ns > 2:
        bsubu1 = 0.5 * (bsubu[2:] + bsubu[1:-1])
        bsubv1 = 0.5 * (bsubv[2:] + bsubv[1:-1])
        bdotk[1:-1] = itheta[1:-1] * bsubu1 + izeta[1:-1] * bsubv1

    # VMEC multiplies bdotk by mu0 before feeding Mercier (bdotj = sqrt(g)*J·B).
    bdotk_merc = MU0 * bdotk

    # Optional debug dump for parity work.
    if os.getenv("VMEC_JAX_DUMP_JDOTB", "") not in ("", "0"):
        from pathlib import Path

        tag = os.getenv("VMEC_JAX_DUMP_TAG", "").strip()
        name = "jdotb_debug" + (f"_{tag}" if tag else "") + ".npz"
        outdir = Path(os.getenv("VMEC_JAX_DUMP_DIR", ".")).expanduser().resolve()
        outdir.mkdir(parents=True, exist_ok=True)
        np.savez(
            outdir / name,
            bdotk=np.asarray(bdotk, dtype=float),
            itheta=np.asarray(itheta, dtype=float),
            izeta=np.asarray(izeta, dtype=float),
            bsubsu=np.asarray(bsubsu, dtype=float),
            bsubsv=np.asarray(bsubsv, dtype=float),
            bsubu=np.asarray(bsubu, dtype=float),
            bsubv=np.asarray(bsubv, dtype=float),
            bsubu_raw=np.asarray(bsubu_raw, dtype=float) if bsubu_raw is not None else None,
            bsubv_raw=np.asarray(bsubv_raw, dtype=float) if bsubv_raw is not None else None,
            bsupu=np.asarray(bsupu, dtype=float),
            bsupv=np.asarray(bsupv, dtype=float),
            bsubs=np.asarray(bsubs_use, dtype=float),
            bsubs_raw=np.asarray(bsubs, dtype=float),
            sqrtg=np.asarray(sqrtg, dtype=float),
            bsq=np.asarray(bsq, dtype=float),
            vp=np.asarray(vp, dtype=float),
            phips=np.asarray(phips, dtype=float),
            wint=np.asarray(wint, dtype=float),
        )

    DMerc, Dshear, Dcurr, Dwell, Dgeod = _mercier_radial_stability_terms(
        ns=int(ns),
        hs=float(hs),
        s=np.asarray(s, dtype=float),
        iotas=np.asarray(iotas, dtype=float),
        pres=np.asarray(pres, dtype=float),
        phip_real=np.asarray(phip_real, dtype=float),
        vp_real=np.asarray(vp_real, dtype=float),
        bsubu=np.asarray(bsubu, dtype=float),
        bsq=np.asarray(bsq, dtype=float),
        sqrtg=np.asarray(sqrtg, dtype=float),
        geom_channels=geom_channels,
        bdotk_merc=np.asarray(bdotk_merc, dtype=float),
        sign_jac=float(sign_jac),
        sum_context=sum_context,
    )

    jdotb, bdotb, bdotgradv = _jxbforce_1d_current_diagnostics(
        ns=int(ns),
        vp=np.asarray(vp, dtype=float),
        phips=np.asarray(phips, dtype=float),
        sqrtg=np.asarray(sqrtg, dtype=float),
        bsq=np.asarray(bsq, dtype=float),
        pres=np.asarray(pres, dtype=float),
        bdotk=np.asarray(bdotk, dtype=float),
        sigma_an=np.asarray(sigma_an, dtype=float),
        sign_jac=float(sign_jac),
        sum_w=_sum_w,
    )

    return (
        np.asarray(DMerc, dtype=float),
        np.asarray(Dshear, dtype=float),
        np.asarray(Dcurr, dtype=float),
        np.asarray(Dwell, dtype=float),
        np.asarray(Dgeod, dtype=float),
        np.asarray(jdotb, dtype=float),
        np.asarray(bdotb, dtype=float),
        np.asarray(bdotgradv, dtype=float),
    )
