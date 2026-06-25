"""Pure VMEC JXBFORCE Bsub filtering helpers."""

from __future__ import annotations

import numpy as np

from .diagnostics import pshalf_from_s as _pshalf_from_s
from .nyquist import vmec_symoutput_expand as _vmec_symoutput_expand


def _jxbforce_nyquist_limits(trig) -> tuple[int, int]:
    """Return VMEC jxbforce Nyquist cutoffs from grid sizes.

    In VMEC2000, ``mnyq`` / ``nnyq`` are geometric Nyquist limits from
    ``fixaray`` (based on ``ntheta2`` / ``nzeta``), not simply the maximum
    retained Fourier mode in a truncated transform loop.
    """
    ntheta2 = int(getattr(trig, "ntheta2", 0))
    cosnv = np.asarray(getattr(trig, "cosnv"))
    nzeta = int(cosnv.shape[0]) if cosnv.ndim >= 1 else 0
    mnyq = max(ntheta2 - 1, 0)
    nnyq = max(nzeta // 2, 0)
    return mnyq, nnyq


def _solve_jxbforce_collocation(A: np.ndarray, rhs: np.ndarray) -> np.ndarray | None:
    try:
        return np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        try:
            sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
            return sol
        except np.linalg.LinAlgError:
            return None


def _jxbforce_getbsubs_coeffs_lasym_false(
    *,
    frho: np.ndarray,
    bsupu: np.ndarray,
    bsupv: np.ndarray,
    trig,
    nfp: int,
) -> np.ndarray | None:
    """Solve VMEC getbsubs collocation system for the stellarator-symmetric branch."""

    frho = np.asarray(frho, dtype=float)
    bsupu = np.asarray(bsupu, dtype=float)
    bsupv = np.asarray(bsupv, dtype=float)

    nt2 = int(trig.ntheta2)
    nzeta = int(np.asarray(trig.cosnv).shape[0])
    mmax = max(nt2 - 1, 0)
    nmax = max(nzeta // 2, 0)
    if frho.shape != (nt2, nzeta):
        return None
    if bsupu.shape != (nt2, nzeta) or bsupv.shape != (nt2, nzeta):
        return None

    nmax1 = max(0, nmax - 1)
    itotal = nt2 * nzeta - 2 * nmax1
    if itotal <= 0:
        return None

    cosmu = np.asarray(trig.cosmu, dtype=float)[:nt2, : mmax + 1]
    sinmu = np.asarray(trig.sinmu, dtype=float)[:nt2, : mmax + 1]
    cosnv = np.asarray(trig.cosnv, dtype=float)[:, : nmax + 1]
    sinnv = np.asarray(trig.sinnv, dtype=float)[:, : nmax + 1]

    A = np.zeros((itotal, itotal), dtype=float)
    rhs = np.zeros((itotal,), dtype=float)
    row = 0
    z_skip_start = (nzeta // 2) + 1

    for i in range(nt2):
        for k in range(nzeta):
            if (i == 0 or i == nt2 - 1) and (k >= z_skip_start):
                continue
            if row >= itotal:
                return None
            rhs[row] = frho[i, k]
            col = 0
            bu = bsupu[i, k]
            bv = bsupv[i, k]
            for m in range(mmax + 1):
                dm = float(m) * bu
                for n in range(nmax + 1):
                    ccmn = cosmu[i, m] * cosnv[k, n]
                    ssmn = sinmu[i, m] * sinnv[k, n]
                    dn = float(n * nfp) * bv
                    termsc = dm * ccmn - dn * ssmn
                    termcs = -dm * ssmn + dn * ccmn
                    if n == 0 or n == nmax:
                        if m > 0:
                            A[row, col] = termsc
                            col += 1
                        elif n == 0:
                            A[row, col] = bv
                            col += 1
                        else:
                            A[row, col] = termcs
                            col += 1
                    elif m == 0 or m == mmax:
                        A[row, col] = termcs
                        col += 1
                    else:
                        A[row, col] = termsc
                        col += 1
                        A[row, col] = termcs
                        col += 1
            if col != itotal:
                return None
            row += 1

    if row != itotal:
        return None

    sol = _solve_jxbforce_collocation(A, rhs)
    if sol is None:
        return None

    coeff = np.zeros((mmax + 1, 2 * nmax + 1), dtype=float)
    off = nmax
    idx = 0
    for m in range(mmax + 1):
        for n in range(nmax + 1):
            if n == 0 or n == nmax:
                if m > 0:
                    coeff[m, off + n] = sol[idx]
                    idx += 1
                elif n == 0:
                    coeff[m, off + n] = sol[idx]
                    idx += 1
                else:
                    coeff[m, off - n] = sol[idx]
                    idx += 1
            elif m == 0 or m == mmax:
                coeff[m, off - n] = sol[idx]
                idx += 1
            else:
                coeff[m, off + n] = sol[idx]
                idx += 1
                coeff[m, off - n] = sol[idx]
                idx += 1
    if idx != itotal:
        return None
    return coeff


def _jxbforce_getbsubs_coeffs_lasym_true(
    *,
    frho: np.ndarray,
    bsupu: np.ndarray,
    bsupv: np.ndarray,
    trig,
    nfp: int,
) -> np.ndarray | None:
    """Solve VMEC getbsubs collocation system for the LASYM branch."""

    frho = np.asarray(frho, dtype=float)
    bsupu = np.asarray(bsupu, dtype=float)
    bsupv = np.asarray(bsupv, dtype=float)

    ntheta3 = int(getattr(trig, "ntheta3", 0))
    ntheta2 = int(getattr(trig, "ntheta2", 0))
    nzeta = int(np.asarray(trig.cosnv).shape[0])
    mmax = max(ntheta2 - 1, 0)
    nmax = max(nzeta // 2, 0)
    if frho.shape != (ntheta3, nzeta):
        return None
    if bsupu.shape != (ntheta3, nzeta) or bsupv.shape != (ntheta3, nzeta):
        return None

    itotal = ntheta3 * nzeta
    if itotal <= 0:
        return None

    cosmu = np.asarray(trig.cosmu, dtype=float)[:ntheta3, : mmax + 1]
    sinmu = np.asarray(trig.sinmu, dtype=float)[:ntheta3, : mmax + 1]
    cosnv = np.asarray(trig.cosnv, dtype=float)[:, : nmax + 1]
    sinnv = np.asarray(trig.sinnv, dtype=float)[:, : nmax + 1]

    A = np.zeros((itotal, itotal), dtype=float)
    rhs = np.zeros((itotal,), dtype=float)
    row = 0

    for i in range(ntheta3):
        for j in range(nzeta):
            rhs[row] = frho[i, j]
            col = 0
            bu = bsupu[i, j]
            bv = bsupv[i, j]
            for m in range(mmax + 1):
                dm = float(m) * bu
                for n in range(nmax + 1):
                    if (m == 0) and (n == 0):
                        continue
                    if col >= itotal:
                        return None
                    ccmn = cosmu[i, m] * cosnv[j, n]
                    ssmn = sinmu[i, m] * sinnv[j, n]
                    dn = float(n * nfp) * bv
                    termsc = dm * ccmn - dn * ssmn
                    termcs = -dm * ssmn + dn * ccmn
                    if n == 0 or n == nmax:
                        if m > 0:
                            A[row, col] = termsc
                            col += 1
                        elif n == 0:
                            A[row, col] = bv
                            col += 1
                        else:
                            A[row, col] = termcs
                            col += 1
                    elif m == 0 or m == mmax:
                        A[row, col] = termcs
                        col += 1
                    else:
                        A[row, col] = termsc
                        col += 1
                        A[row, col] = termcs
                        col += 1

                    if (m == 0) and (n == 0 or n == nmax):
                        continue
                    if col >= itotal:
                        return None
                    csmn = cosmu[i, m] * sinnv[j, n]
                    scmn = sinmu[i, m] * cosnv[j, n]
                    termcc = -dm * scmn - dn * csmn
                    termss = dm * csmn + dn * scmn
                    if (n == 0 or n == nmax) or (m == 0 or m == mmax):
                        A[row, col] = termcc
                        col += 1
                    else:
                        A[row, col] = termcc
                        col += 1
                        A[row, col] = termss
                        col += 1
            if col != itotal:
                return None
            row += 1

    if row != itotal:
        return None

    sol = _solve_jxbforce_collocation(A, rhs)
    if sol is None:
        return None

    coeff = np.zeros((mmax + 1, 2 * nmax + 1, 2), dtype=float)
    off = nmax
    idx = 0
    for m in range(mmax + 1):
        for n in range(nmax + 1):
            if (m == 0) and (n == 0):
                continue
            if idx >= itotal:
                break
            if n == 0 or n == nmax:
                if m > 0:
                    coeff[m, off + n, 0] = sol[idx]
                    idx += 1
                elif n == 0:
                    coeff[m, off + n, 0] = sol[idx]
                    idx += 1
                else:
                    coeff[m, off - n, 0] = sol[idx]
                    idx += 1
            elif m == 0 or m == mmax:
                coeff[m, off - n, 0] = sol[idx]
                idx += 1
            else:
                coeff[m, off + n, 0] = sol[idx]
                idx += 1
                coeff[m, off - n, 0] = sol[idx]
                idx += 1

            if (m == 0) and (n == 0 or n == nmax):
                continue
            if idx >= itotal:
                break
            if (n == 0 or n == nmax) or (m == 0 or m == mmax):
                coeff[m, off + n, 1] = sol[idx]
                idx += 1
            else:
                coeff[m, off + n, 1] = sol[idx]
                idx += 1
                coeff[m, off - n, 1] = sol[idx]
                idx += 1
    return coeff


def _jxbforce_float_arrays(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    return tuple(np.asarray(arr, dtype=float) for arr in arrays)


def _jxbforce_trig_slices(trig, *, nt2: int, mnyq: int, nnyq: int) -> tuple[np.ndarray, ...]:
    return (
        np.asarray(trig.cosmu, dtype=float)[:nt2, : mnyq + 1],
        np.asarray(trig.sinmu, dtype=float)[:nt2, : mnyq + 1],
        np.asarray(trig.cosmum, dtype=float)[:nt2, : mnyq + 1],
        np.asarray(trig.sinmum, dtype=float)[:nt2, : mnyq + 1],
        np.asarray(trig.cosnv, dtype=float)[:, : nnyq + 1],
        np.asarray(trig.sinnv, dtype=float)[:, : nnyq + 1],
        np.asarray(trig.cosnvn, dtype=float)[:, : nnyq + 1],
        np.asarray(trig.sinnvn, dtype=float)[:, : nnyq + 1],
    )


def _jxbforce_apply_bsubs_correction_lasym_false(
    *,
    bsubu: np.ndarray,
    bsubv: np.ndarray,
    bsubs: np.ndarray,
    bsubsu: np.ndarray,
    bsubsv: np.ndarray,
    bsupu: np.ndarray,
    bsupv: np.ndarray,
    sqrtg: np.ndarray,
    pres: np.ndarray,
    vp: np.ndarray,
    hs: float,
    signgs: float,
    trig,
    nfp: int,
    sum_w,
    getbsubs_coeffs_func=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirror VMEC jxbforce corrected-bsubs pass for lasym=False."""

    bsubu, bsubv, bsubs, bsubsu, bsubsv, bsupu, bsupv, sqrtg, pres, vp = _jxbforce_float_arrays(
        bsubu, bsubv, bsubs, bsubsu, bsubsv, bsupu, bsupv, sqrtg, pres, vp
    )
    getbsubs_coeffs = getbsubs_coeffs_func or _jxbforce_getbsubs_coeffs_lasym_false

    ns, nt2, nzeta = bsubu.shape
    if ns < 3 or hs == 0.0:
        return bsubs, bsubsu, bsubsv

    ohs = 1.0 / hs
    mnyq, nnyq = _jxbforce_nyquist_limits(trig)
    cosmu, sinmu, cosmum, sinmum, cosnv, sinnv, cosnvn, sinnvn = _jxbforce_trig_slices(
        trig, nt2=nt2, mnyq=mnyq, nnyq=nnyq
    )

    for js in range(1, ns - 1):
        jxb = 0.5 * (sqrtg[js] + sqrtg[js + 1])
        bsupu1 = 0.5 * (bsupu[js] * sqrtg[js] + bsupu[js + 1] * sqrtg[js + 1])
        bsupv1 = 0.5 * (bsupv[js] * sqrtg[js] + bsupv[js + 1] * sqrtg[js + 1])

        brho = ohs * (bsupu1 * (bsubu[js + 1] - bsubu[js]) + bsupv1 * (bsubv[js + 1] - bsubv[js]))
        brho = brho + (pres[js + 1] - pres[js]) * ohs * jxb

        brho00 = float(sum_w(brho))
        vden = 0.5 * (vp[js] + vp[js + 1])
        if vden != 0.0:
            brho = brho - signgs * jxb * (brho00 / vden)

        coeff = getbsubs_coeffs(frho=brho, bsupu=bsupu1, bsupv=bsupv1, trig=trig, nfp=int(nfp))
        if coeff is None:
            continue

        bsubs_s = np.zeros((nt2, nzeta), dtype=float)
        bsubsu_s = np.zeros((nt2, nzeta), dtype=float)
        bsubsv_s = np.zeros((nt2, nzeta), dtype=float)
        off = nnyq

        for m in range(mnyq + 1):
            for n in range(nnyq + 1):
                c1 = coeff[m, off + n]
                c2 = 0.0 if n == 0 else coeff[m, off - n]
                for k in range(nzeta):
                    for j in range(nt2):
                        tsin1 = sinmu[j, m] * cosnv[k, n]
                        tsin2 = cosmu[j, m] * sinnv[k, n]
                        bsubs_s[j, k] += tsin1 * c1 + tsin2 * c2

                        tcosm1 = cosmum[j, m] * cosnv[k, n]
                        tcosm2 = sinmum[j, m] * sinnv[k, n]
                        bsubsu_s[j, k] += tcosm1 * c1 + tcosm2 * c2

                        tcosn1 = sinmu[j, m] * sinnvn[k, n]
                        tcosn2 = cosmu[j, m] * cosnvn[k, n]
                        bsubsv_s[j, k] += tcosn1 * c1 + tcosn2 * c2

        bsubs[js] = bsubs_s
        bsubsu[js] = bsubsu_s
        bsubsv[js] = bsubsv_s

    if ns > 2:
        bsubs[0] = 2.0 * bsubs[1] - bsubs[2]
        bsubs[-1] = 2.0 * bsubs[-1] - bsubs[-2]
    return bsubs, bsubsu, bsubsv


def _jxbforce_apply_bsubs_correction_lasym_true(
    *,
    bsubu: np.ndarray,
    bsubv: np.ndarray,
    bsubs: np.ndarray,
    bsubsu: np.ndarray,
    bsubsv: np.ndarray,
    bsupu: np.ndarray,
    bsupv: np.ndarray,
    sqrtg: np.ndarray,
    pres: np.ndarray,
    vp: np.ndarray,
    hs: float,
    signgs: float,
    trig,
    nfp: int,
    sum_w,
    getbsubs_coeffs_func=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirror VMEC jxbforce corrected-bsubs pass for lasym=True."""
    bsubu, bsubv, bsubs, bsubsu, bsubsv, bsupu, bsupv, sqrtg, pres, vp = _jxbforce_float_arrays(
        bsubu, bsubv, bsubs, bsubsu, bsubsv, bsupu, bsupv, sqrtg, pres, vp
    )
    getbsubs_coeffs = getbsubs_coeffs_func or _jxbforce_getbsubs_coeffs_lasym_true

    ns, nt2, nzeta = bsubu.shape
    nt3 = int(getattr(trig, "ntheta3", nt2))
    if nt3 < nt2:
        nt3 = nt2
    if ns < 3 or hs == 0.0:
        return bsubs, bsubsu, bsubsv

    ohs = 1.0 / hs
    mnyq, nnyq = _jxbforce_nyquist_limits(trig)
    cosmu, sinmu, cosmum, sinmum, cosnv, sinnv, cosnvn, sinnvn = _jxbforce_trig_slices(
        trig, nt2=nt2, mnyq=mnyq, nnyq=nnyq
    )

    if bsubu.shape[1] == nt3:
        bsubu_full = bsubu
        bsubv_full = bsubv
    else:
        bsubu_full = _vmec_symoutput_expand(sym=bsubu, asym=None, trig=trig)
        bsubv_full = _vmec_symoutput_expand(sym=bsubv, asym=None, trig=trig)

    if bsupu.shape[1] == nt3:
        bsupu_full = bsupu
        bsupv_full = bsupv
    else:
        bsupu_full = _vmec_symoutput_expand(sym=bsupu[:, :nt2, :], asym=None, trig=trig)
        bsupv_full = _vmec_symoutput_expand(sym=bsupv[:, :nt2, :], asym=None, trig=trig)

    if sqrtg.shape[1] == nt3:
        sqrtg_full = sqrtg
    else:
        sqrtg_full = _vmec_symoutput_expand(sym=sqrtg[:, :nt2, :], asym=None, trig=trig)

    if bsubs.shape[1] == nt3:
        bsubs_out = bsubs.copy()
    else:
        bsubs_out = np.zeros((ns, nt3, nzeta), dtype=float)
    bsubsu_out = np.zeros((ns, nt3, nzeta), dtype=float)
    bsubsv_out = np.zeros((ns, nt3, nzeta), dtype=float)

    for js in range(1, ns - 1):
        jxb = 0.5 * (sqrtg_full[js] + sqrtg_full[js + 1])
        bsupu1 = 0.5 * (bsupu_full[js] * sqrtg_full[js] + bsupu_full[js + 1] * sqrtg_full[js + 1])
        bsupv1 = 0.5 * (bsupv_full[js] * sqrtg_full[js] + bsupv_full[js + 1] * sqrtg_full[js + 1])

        brho_full = ohs * (
            bsupu1 * (bsubu_full[js + 1] - bsubu_full[js]) + bsupv1 * (bsubv_full[js + 1] - bsubv_full[js])
        )
        brho_full = brho_full + (pres[js + 1] - pres[js]) * ohs * jxb
        brho00 = float(sum_w(brho_full))
        vden = 0.5 * (vp[js] + vp[js + 1])
        if vden != 0.0:
            brho_full = brho_full - signgs * jxb * (brho00 / vden)

        coeff = getbsubs_coeffs(frho=brho_full, bsupu=bsupu1, bsupv=bsupv1, trig=trig, nfp=int(nfp))
        if coeff is None:
            continue

        bsubs_s = np.zeros((nt2, nzeta), dtype=float)
        bsubsu_s = np.zeros((nt2, nzeta), dtype=float)
        bsubsv_s = np.zeros((nt2, nzeta), dtype=float)
        bsubs_a = np.zeros((nt2, nzeta), dtype=float)
        bsubsu_a = np.zeros((nt2, nzeta), dtype=float)
        bsubsv_a = np.zeros((nt2, nzeta), dtype=float)
        off = nnyq

        for m in range(mnyq + 1):
            for n in range(nnyq + 1):
                c1 = coeff[m, off + n, 0]
                c2 = 0.0 if n == 0 else coeff[m, off - n, 0]
                c3 = coeff[m, off + n, 1]
                c4 = 0.0 if n == 0 else coeff[m, off - n, 1]
                for k in range(nzeta):
                    for j in range(nt2):
                        tsin1 = sinmu[j, m] * cosnv[k, n]
                        tsin2 = cosmu[j, m] * sinnv[k, n]
                        bsubs_s[j, k] += tsin1 * c1 + tsin2 * c2
                        tcosm1 = cosmum[j, m] * cosnv[k, n]
                        tcosm2 = sinmum[j, m] * sinnv[k, n]
                        bsubsu_s[j, k] += tcosm1 * c1 + tcosm2 * c2
                        tcosn1 = sinmu[j, m] * sinnvn[k, n]
                        tcosn2 = cosmu[j, m] * cosnvn[k, n]
                        bsubsv_s[j, k] += tcosn1 * c1 + tcosn2 * c2

                        tcos1 = cosmu[j, m] * cosnv[k, n]
                        tcos2 = sinmu[j, m] * sinnv[k, n]
                        bsubs_a[j, k] += tcos1 * c3 + tcos2 * c4
                        tsinm1 = sinmum[j, m] * cosnv[k, n]
                        tsinm2 = cosmum[j, m] * sinnv[k, n]
                        bsubsu_a[j, k] += tsinm1 * c3 + tsinm2 * c4
                        tsinn1 = cosmu[j, m] * sinnvn[k, n]
                        tsinn2 = sinmu[j, m] * cosnvn[k, n]
                        bsubsv_a[j, k] += tsinn1 * c3 + tsinn2 * c4

        bsubs_full = _vmec_symoutput_expand(sym=bsubs_a[None, ...], asym=bsubs_s[None, ...], trig=trig)[0]
        bsubs_out[js] = bsubs_full
        bsubsu_out[js] = _vmec_symoutput_expand(sym=bsubsu_s[None, ...], asym=bsubsu_a[None, ...], trig=trig)[0]
        bsubsv_out[js] = _vmec_symoutput_expand(sym=bsubsv_s[None, ...], asym=bsubsv_a[None, ...], trig=trig)[0]

    if ns > 2:
        bsubs_out[0] = 2.0 * bsubs_out[1] - bsubs_out[2]
        bsubs_out[-1] = 2.0 * bsubs_out[-1] - bsubs_out[-2]
    return bsubs_out, bsubsu_out, bsubsv_out



def _filter_bsubuv_jxbforce(
    *,
    bsubu: np.ndarray,
    bsubv: np.ndarray,
    trig,
    nfp: int,
    mmax_force: int,
    nmax_force: int,
    s: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """JXBFORCE-style low-pass filter for bsubu/bsubv (lasym=False)."""
    # For parity-critical diagnostics we prefer the explicit VMEC loop
    # ordering, which matches the Fortran summation order more closely.
    import os

    # Default to the vectorized path for performance; the loop-based path is
    # retained for parity debugging.
    use_loop = os.getenv("VMEC_JAX_BSUB_FILTER_LOOP", "0") not in ("", "0")
    if use_loop:
        return _filter_bsubuv_jxbforce_loop(
            bsubu=bsubu,
            bsubv=bsubv,
            trig=trig,
            mmax_force=mmax_force,
            nmax_force=nmax_force,
            s=s,
        )
    bsubu = np.asarray(bsubu, dtype=float)
    bsubv = np.asarray(bsubv, dtype=float)
    ns, ntheta, nzeta = bsubu.shape

    nt2 = int(trig.ntheta2)
    if ntheta < nt2:
        raise ValueError("bsubu grid smaller than ntheta2")

    mmax = int(mmax_force)
    nmax = int(nmax_force)
    if mmax < 0 or nmax < 0:
        return bsubu[:, :nt2, :].copy(), bsubv[:, :nt2, :].copy()

    # Implement the jxbforce low-pass filter explicitly to match VMEC's
    # parity-normalized Fourier transforms.
    bsubu_red = np.asarray(bsubu[:, :nt2, :], dtype=float)
    bsubv_red = np.asarray(bsubv[:, :nt2, :], dtype=float)

    cosmui = np.asarray(trig.cosmui, dtype=float)[:nt2, : mmax + 1]
    sinmui = np.asarray(trig.sinmui, dtype=float)[:nt2, : mmax + 1]
    cosmu = np.asarray(trig.cosmu, dtype=float)[:nt2, : mmax + 1]
    sinmu = np.asarray(trig.sinmu, dtype=float)[:nt2, : mmax + 1]
    cosnv = np.asarray(trig.cosnv, dtype=float)[:, : nmax + 1]
    sinnv = np.asarray(trig.sinnv, dtype=float)[:, : nmax + 1]

    r0scale = float(getattr(trig, "r0scale", 1.0))
    dnorm1 = 1.0 / (r0scale**2)
    dmult = np.full((mmax + 1, nmax + 1), dnorm1, dtype=float)
    mnyq, nnyq = _jxbforce_nyquist_limits(trig)
    if mnyq > 0 and mnyq <= mmax:
        dmult[mnyq, :] *= 0.5
    if nnyq > 0 and nnyq <= nmax:
        dmult[:, nnyq] *= 0.5

    def _pshalf_from_s(s_full: np.ndarray) -> np.ndarray:
        if s_full.shape[0] < 2:
            return np.sqrt(np.maximum(s_full, 0.0))
        sh = 0.5 * (s_full[1:] + s_full[:-1])
        p = np.concatenate([sh[:1], sh], axis=0)
        return np.sqrt(np.maximum(p, 0.0))

    pshalf = None
    if s is not None:
        s_full = np.asarray(s, dtype=float)
        pshalf = _pshalf_from_s(s_full)
        # Avoid divide-by-zero on-axis; VMEC sets shalf(1)=shalf(2).
        if pshalf.shape[0] > 1:
            pshalf[0] = pshalf[1]

    # When the filter limits match the available basis, the transform should
    # be identity (avoid numerical drift by returning the original fields).
    full_mmax, full_nmax = _jxbforce_nyquist_limits(trig)
    if mmax >= full_mmax and nmax >= full_nmax:
        return bsubu[:, :nt2, :].copy(), bsubv[:, :nt2, :].copy()

    def _filter_one(f: np.ndarray) -> np.ndarray:
        # Forward transform: cos(mu)cos(nv) + sin(mu)sin(nv) (jxbforce).
        f_theta_cos = np.einsum("sik,im->smk", f, cosmui, optimize=True)
        f_theta_sin = np.einsum("sik,im->smk", f, sinmui, optimize=True)
        coeff1 = np.einsum("smk,kn->smn", f_theta_cos, cosnv, optimize=True)
        coeff2 = np.einsum("smk,kn->smn", f_theta_sin, sinnv, optimize=True)
        coeff1 = coeff1 * dmult[None, :, :]
        coeff2 = coeff2 * dmult[None, :, :]
        # VMEC stores odd-m fields with an extra sqrt(s) factor. Undo that
        # scaling for odd m before the inverse transform (jxbforce does this
        # via bsubu(js,:,1)/shalf(js)).
        if pshalf is not None and mmax >= 1:
            odd = (np.arange(mmax + 1) % 2) == 1
            if np.any(odd):
                scale = np.ones((coeff1.shape[0], mmax + 1, 1), dtype=float)
                scale[:, odd, 0] = 1.0 / pshalf[:, None]
                coeff1 = coeff1 * scale
                coeff2 = coeff2 * scale

        # Inverse transform back to real space on the reduced grid.
        tmp_cos = np.einsum("smn,im->sin", coeff1, cosmu, optimize=True)
        tmp_sin = np.einsum("smn,im->sin", coeff2, sinmu, optimize=True)
        return np.einsum("sin,kn->sik", tmp_cos, cosnv, optimize=True) + np.einsum(
            "sin,kn->sik", tmp_sin, sinnv, optimize=True
        )

    return _filter_one(bsubu_red), _filter_one(bsubv_red)


def _filter_bsubuv_jxbforce_parity(
    *,
    bsubu_even: np.ndarray,
    bsubu_odd: np.ndarray,
    bsubv_even: np.ndarray,
    bsubv_odd: np.ndarray,
    trig,
    mmax_force: int,
    nmax_force: int,
    s: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """JXBFORCE-style low-pass filter using parity-separated bsubu/bsubv (lasym=False).

    This is a vectorized equivalent of :func:`_filter_bsubuv_jxbforce_parity_loop`.
    We keep the loop-based implementation for parity debugging, but default to the
    vectorized path for performance.
    """
    import os

    use_loop = os.getenv("VMEC_JAX_BSUB_FILTER_LOOP", "0") not in ("", "0")
    if use_loop:
        return _filter_bsubuv_jxbforce_parity_loop(
            bsubu_even=bsubu_even,
            bsubu_odd=bsubu_odd,
            bsubv_even=bsubv_even,
            bsubv_odd=bsubv_odd,
            trig=trig,
            mmax_force=mmax_force,
            nmax_force=nmax_force,
            s=s,
        )

    bsubu_even = np.asarray(bsubu_even, dtype=float)
    bsubu_odd = np.asarray(bsubu_odd, dtype=float)
    bsubv_even = np.asarray(bsubv_even, dtype=float)
    bsubv_odd = np.asarray(bsubv_odd, dtype=float)
    if bsubu_even.shape != bsubu_odd.shape or bsubu_even.shape != bsubv_even.shape:
        raise ValueError("Parity bsubu/bsubv shape mismatch")

    ns, ntheta, nzeta = bsubu_even.shape
    nt2 = int(trig.ntheta2)
    if ntheta < nt2:
        raise ValueError("bsubu grid smaller than ntheta2")

    mmax = int(mmax_force)
    nmax = int(nmax_force)
    if mmax < 0 or nmax < 0:
        return bsubu_even[:, :nt2, :].copy(), bsubv_even[:, :nt2, :].copy()

    bsubu_even_red = bsubu_even[:, :nt2, :]
    bsubu_odd_red = bsubu_odd[:, :nt2, :]
    bsubv_even_red = bsubv_even[:, :nt2, :]
    bsubv_odd_red = bsubv_odd[:, :nt2, :]

    cosmui = np.asarray(trig.cosmui, dtype=float)[:nt2, : mmax + 1]
    sinmui = np.asarray(trig.sinmui, dtype=float)[:nt2, : mmax + 1]
    cosmu = np.asarray(trig.cosmu, dtype=float)[:nt2, : mmax + 1]
    sinmu = np.asarray(trig.sinmu, dtype=float)[:nt2, : mmax + 1]
    cosnv = np.asarray(trig.cosnv, dtype=float)[:, : nmax + 1]
    sinnv = np.asarray(trig.sinnv, dtype=float)[:, : nmax + 1]

    r0scale = float(getattr(trig, "r0scale", 1.0))
    dnorm1 = 1.0 / (r0scale**2)
    dmult = np.full((mmax + 1, nmax + 1), dnorm1, dtype=float)
    mnyq, nnyq = _jxbforce_nyquist_limits(trig)
    if mnyq > 0 and mnyq <= mmax:
        dmult[mnyq, :] *= 0.5
    if nnyq > 0 and nnyq <= nmax:
        dmult[:, nnyq] *= 0.5

    pshalf = None
    if s is not None:
        s_full = np.asarray(s, dtype=float)
        pshalf = _pshalf_from_s(s_full)
        if pshalf.shape[0] > 1:
            pshalf[0] = pshalf[1]

    odd_m = (np.arange(mmax + 1) % 2) == 1

    def _forward(f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        f_theta_cos = np.einsum("sik,im->smk", f, cosmui, optimize=True)
        f_theta_sin = np.einsum("sik,im->smk", f, sinmui, optimize=True)
        coeff1 = np.einsum("smk,kn->smn", f_theta_cos, cosnv, optimize=True)
        coeff2 = np.einsum("smk,kn->smn", f_theta_sin, sinnv, optimize=True)
        coeff1 = coeff1 * dmult[None, :, :]
        coeff2 = coeff2 * dmult[None, :, :]
        return coeff1, coeff2

    def _inverse(coeff1: np.ndarray, coeff2: np.ndarray) -> np.ndarray:
        tmp_cos = np.einsum("smn,im->sin", coeff1, cosmu, optimize=True)
        tmp_sin = np.einsum("smn,im->sin", coeff2, sinmu, optimize=True)
        return np.einsum("sin,kn->sik", tmp_cos, cosnv, optimize=True) + np.einsum(
            "sin,kn->sik", tmp_sin, sinnv, optimize=True
        )

    def _filter_field(f_even: np.ndarray, f_odd: np.ndarray) -> np.ndarray:
        c1e, c2e = _forward(f_even)
        c1o, c2o = _forward(f_odd)
        if pshalf is not None and mmax >= 1 and np.any(odd_m):
            scale = pshalf[:, None, None]
            c1o[:, odd_m, :] = c1o[:, odd_m, :] / scale
            c2o[:, odd_m, :] = c2o[:, odd_m, :] / scale
        if np.any(odd_m):
            c1e = c1e.copy()
            c2e = c2e.copy()
            c1e[:, odd_m, :] = c1o[:, odd_m, :]
            c2e[:, odd_m, :] = c2o[:, odd_m, :]
        return _inverse(c1e, c2e)

    bsubu_out = _filter_field(bsubu_even_red, bsubu_odd_red)
    bsubv_out = _filter_field(bsubv_even_red, bsubv_odd_red)
    return np.asarray(bsubu_out, dtype=float), np.asarray(bsubv_out, dtype=float)


def _filter_bsubuv_jxbforce_parity_loop(
    *,
    bsubu_even: np.ndarray,
    bsubu_odd: np.ndarray,
    bsubv_even: np.ndarray,
    bsubv_odd: np.ndarray,
    trig,
    mmax_force: int,
    nmax_force: int,
    s: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Loop-based jxbforce low-pass filter using parity-separated bsubu/bsubv."""
    # Cancellation in the low-pass Fourier sums can be severe near the edge for
    # high-shear equilibria. Accumulate in long double, cast back to float.
    acc_dtype = np.longdouble
    bsubu_even = np.asarray(bsubu_even, dtype=acc_dtype)
    bsubu_odd = np.asarray(bsubu_odd, dtype=acc_dtype)
    bsubv_even = np.asarray(bsubv_even, dtype=acc_dtype)
    bsubv_odd = np.asarray(bsubv_odd, dtype=acc_dtype)
    if bsubu_even.shape != bsubu_odd.shape or bsubu_even.shape != bsubv_even.shape:
        raise ValueError("Parity bsubu/bsubv shape mismatch")

    ns, ntheta, nzeta = bsubu_even.shape
    nt2 = int(trig.ntheta2)
    if ntheta < nt2:
        raise ValueError("bsubu grid smaller than ntheta2")

    mmax = int(mmax_force)
    nmax = int(nmax_force)
    if mmax < 0 or nmax < 0:
        return bsubu_even[:, :nt2, :].copy(), bsubv_even[:, :nt2, :].copy()

    cosmui = np.asarray(trig.cosmui, dtype=acc_dtype)[:nt2, : mmax + 1]
    sinmui = np.asarray(trig.sinmui, dtype=acc_dtype)[:nt2, : mmax + 1]
    cosmu = np.asarray(trig.cosmu, dtype=acc_dtype)[:nt2, : mmax + 1]
    sinmu = np.asarray(trig.sinmu, dtype=acc_dtype)[:nt2, : mmax + 1]
    cosnv = np.asarray(trig.cosnv, dtype=acc_dtype)[:, : nmax + 1]
    sinnv = np.asarray(trig.sinnv, dtype=acc_dtype)[:, : nmax + 1]

    r0scale = float(getattr(trig, "r0scale", 1.0))
    base_dnorm = acc_dtype(1.0) / acc_dtype(r0scale**2)
    mnyq, nnyq = _jxbforce_nyquist_limits(trig)

    def _pshalf_from_s(s_full: np.ndarray) -> np.ndarray:
        if s_full.shape[0] < 2:
            return np.sqrt(np.maximum(s_full, 0.0))
        sh = 0.5 * (s_full[1:] + s_full[:-1])
        p = np.concatenate([sh[:1], sh], axis=0)
        return np.sqrt(np.maximum(p, 0.0))

    pshalf = None
    if s is not None:
        s_full = np.asarray(s, dtype=acc_dtype)
        pshalf = _pshalf_from_s(s_full)
        if pshalf.shape[0] > 1:
            pshalf[0] = pshalf[1]

    bsubu_out = np.zeros((ns, nt2, nzeta), dtype=acc_dtype)
    bsubv_out = np.zeros((ns, nt2, nzeta), dtype=acc_dtype)

    for js in range(ns):
        bsubu_even_s = bsubu_even[js, :nt2, :]
        bsubu_odd_s = bsubu_odd[js, :nt2, :]
        bsubv_even_s = bsubv_even[js, :nt2, :]
        bsubv_odd_s = bsubv_odd[js, :nt2, :]
        for m in range(mmax + 1):
            use_odd = (m % 2) == 1
            bsubu_in = bsubu_odd_s if use_odd else bsubu_even_s
            bsubv_in = bsubv_odd_s if use_odd else bsubv_even_s
            for n in range(nmax + 1):
                dnorm1 = base_dnorm
                if mnyq > 0 and m == mnyq:
                    dnorm1 *= 0.5
                if nnyq > 0 and n == nnyq and n != 0:
                    dnorm1 *= 0.5
                if use_odd and (pshalf is not None):
                    dnorm1 = dnorm1 / float(pshalf[js])

                bsubumn1 = acc_dtype(0.0)
                bsubumn2 = acc_dtype(0.0)
                bsubvmn1 = acc_dtype(0.0)
                bsubvmn2 = acc_dtype(0.0)

                for k in range(nzeta):
                    for j in range(nt2):
                        tcosi1 = cosmui[j, m] * cosnv[k, n] * dnorm1
                        tcosi2 = sinmui[j, m] * sinnv[k, n] * dnorm1
                        val_u = bsubu_in[j, k]
                        val_v = bsubv_in[j, k]
                        bsubumn1 += tcosi1 * val_u
                        bsubumn2 += tcosi2 * val_u
                        bsubvmn1 += tcosi1 * val_v
                        bsubvmn2 += tcosi2 * val_v

                for k in range(nzeta):
                    for j in range(nt2):
                        tcos1 = cosmu[j, m] * cosnv[k, n]
                        tcos2 = sinmu[j, m] * sinnv[k, n]
                        bsubu_out[js, j, k] += tcos1 * bsubumn1 + tcos2 * bsubumn2
                        bsubv_out[js, j, k] += tcos1 * bsubvmn1 + tcos2 * bsubvmn2

    return np.asarray(bsubu_out, dtype=float), np.asarray(bsubv_out, dtype=float)


def _jxbforce_filter_with_bsubs_derivs_loop(
    *,
    bsubs: np.ndarray,
    bsubu_even: np.ndarray,
    bsubu_odd: np.ndarray,
    bsubv_even: np.ndarray,
    bsubv_odd: np.ndarray,
    trig,
    mmax_force: int,
    nmax_force: int,
    s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """VMEC jxbforce low-pass + bsubsu/bsubsv in one loop (lasym=False).

    This mirrors the coupled transform block in ``jxbforce.f`` where filtered
    ``bsubu/bsubv`` and derivatives ``bsubsu/bsubsv`` are reconstructed from the
    same Fourier accumulators. Keeping these coupled reduces cancellation drift
    in downstream ``itheta/izeta/bdotk`` diagnostics.
    """
    acc_dtype = np.longdouble
    bsubs = np.asarray(bsubs, dtype=acc_dtype)
    bsubu_even = np.asarray(bsubu_even, dtype=acc_dtype)
    bsubu_odd = np.asarray(bsubu_odd, dtype=acc_dtype)
    bsubv_even = np.asarray(bsubv_even, dtype=acc_dtype)
    bsubv_odd = np.asarray(bsubv_odd, dtype=acc_dtype)

    if bsubu_even.shape != bsubu_odd.shape or bsubu_even.shape != bsubv_even.shape:
        raise ValueError("Parity bsubu/bsubv shape mismatch")
    if bsubs.shape[:2] != bsubu_even.shape[:2]:
        raise ValueError("bsubs and parity bsub shapes mismatch")

    ns, ntheta, nzeta = bsubu_even.shape
    nt2 = int(trig.ntheta2)
    if ntheta < nt2:
        raise ValueError("bsubu grid smaller than ntheta2")

    mmax = int(mmax_force)
    nmax = int(nmax_force)
    if mmax < 0 or nmax < 0:
        z = np.zeros((ns, nt2, nzeta), dtype=float)
        return z.copy(), z.copy(), z.copy(), z.copy()

    cosmui = np.asarray(trig.cosmui, dtype=acc_dtype)[:nt2, : mmax + 1]
    sinmui = np.asarray(trig.sinmui, dtype=acc_dtype)[:nt2, : mmax + 1]
    cosmu = np.asarray(trig.cosmu, dtype=acc_dtype)[:nt2, : mmax + 1]
    sinmu = np.asarray(trig.sinmu, dtype=acc_dtype)[:nt2, : mmax + 1]
    cosmum = np.asarray(trig.cosmum, dtype=acc_dtype)[:nt2, : mmax + 1]
    sinmum = np.asarray(trig.sinmum, dtype=acc_dtype)[:nt2, : mmax + 1]
    cosnv = np.asarray(trig.cosnv, dtype=acc_dtype)[:, : nmax + 1]
    sinnv = np.asarray(trig.sinnv, dtype=acc_dtype)[:, : nmax + 1]
    cosnvn = np.asarray(trig.cosnvn, dtype=acc_dtype)[:, : nmax + 1]
    sinnvn = np.asarray(trig.sinnvn, dtype=acc_dtype)[:, : nmax + 1]

    r0scale = float(getattr(trig, "r0scale", 1.0))
    base_dnorm = acc_dtype(1.0 / (r0scale**2))
    mnyq, nnyq = _jxbforce_nyquist_limits(trig)

    s_full = np.asarray(s, dtype=acc_dtype)
    if s_full.shape[0] < 2:
        pshalf = np.sqrt(np.maximum(s_full, 0.0))
    else:
        sh = 0.5 * (s_full[1:] + s_full[:-1])
        pshalf = np.concatenate([sh[:1], sh], axis=0)
        pshalf = np.sqrt(np.maximum(pshalf, 0.0))
    if pshalf.shape[0] > 1:
        pshalf[0] = pshalf[1]

    bsubu_out = np.zeros((ns, nt2, nzeta), dtype=acc_dtype)
    bsubv_out = np.zeros((ns, nt2, nzeta), dtype=acc_dtype)
    bsubsu = np.zeros((ns, nt2, nzeta), dtype=acc_dtype)
    bsubsv = np.zeros((ns, nt2, nzeta), dtype=acc_dtype)

    for js in range(ns):
        bsubs_s = bsubs[js, :nt2, :]
        bu_even = bsubu_even[js, :nt2, :]
        bu_odd = bsubu_odd[js, :nt2, :]
        bv_even = bsubv_even[js, :nt2, :]
        bv_odd = bsubv_odd[js, :nt2, :]
        for m in range(mmax + 1):
            use_odd = (m % 2) == 1
            bu_in = bu_odd if use_odd else bu_even
            bv_in = bv_odd if use_odd else bv_even
            for n in range(nmax + 1):
                dnorm1 = base_dnorm
                if mnyq > 0 and m == mnyq:
                    dnorm1 *= 0.5
                if nnyq > 0 and n == nnyq and n != 0:
                    dnorm1 *= 0.5
                if use_odd:
                    dnorm1 = dnorm1 / pshalf[js]

                bsubsmn1 = acc_dtype(0.0)
                bsubsmn2 = acc_dtype(0.0)
                bsubumn1 = acc_dtype(0.0)
                bsubumn2 = acc_dtype(0.0)
                bsubvmn1 = acc_dtype(0.0)
                bsubvmn2 = acc_dtype(0.0)

                for k in range(nzeta):
                    for j in range(nt2):
                        tsini1 = sinmui[j, m] * cosnv[k, n] * dnorm1
                        tsini2 = cosmui[j, m] * sinnv[k, n] * dnorm1
                        tcosi1 = cosmui[j, m] * cosnv[k, n] * dnorm1
                        tcosi2 = sinmui[j, m] * sinnv[k, n] * dnorm1
                        vbs = bsubs_s[j, k]
                        vu = bu_in[j, k]
                        vv = bv_in[j, k]
                        bsubsmn1 += tsini1 * vbs
                        bsubsmn2 += tsini2 * vbs
                        bsubumn1 += tcosi1 * vu
                        bsubumn2 += tcosi2 * vu
                        bsubvmn1 += tcosi1 * vv
                        bsubvmn2 += tcosi2 * vv

                for k in range(nzeta):
                    for j in range(nt2):
                        tcos1 = cosmu[j, m] * cosnv[k, n]
                        tcos2 = sinmu[j, m] * sinnv[k, n]
                        bsubu_out[js, j, k] += tcos1 * bsubumn1 + tcos2 * bsubumn2
                        bsubv_out[js, j, k] += tcos1 * bsubvmn1 + tcos2 * bsubvmn2

                        tcosm1 = cosmum[j, m] * cosnv[k, n]
                        tcosm2 = sinmum[j, m] * sinnv[k, n]
                        bsubsu[js, j, k] += tcosm1 * bsubsmn1 + tcosm2 * bsubsmn2
                        tcosn1 = sinmu[j, m] * sinnvn[k, n]
                        tcosn2 = cosmu[j, m] * cosnvn[k, n]
                        bsubsv[js, j, k] += tcosn1 * bsubsmn1 + tcosn2 * bsubsmn2

    return (
        np.asarray(bsubu_out, dtype=float),
        np.asarray(bsubv_out, dtype=float),
        np.asarray(bsubsu, dtype=float),
        np.asarray(bsubsv, dtype=float),
    )


def _filter_bsubuv_jxbforce_loop(
    *,
    bsubu: np.ndarray,
    bsubv: np.ndarray,
    trig,
    mmax_force: int,
    nmax_force: int,
    s: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Loop-based jxbforce low-pass filter (matches VMEC summation order)."""
    # Accumulate in long double for cancellation-sensitive mode sums.
    acc_dtype = np.longdouble
    bsubu = np.asarray(bsubu, dtype=acc_dtype)
    bsubv = np.asarray(bsubv, dtype=acc_dtype)
    ns, ntheta, nzeta = bsubu.shape

    nt2 = int(trig.ntheta2)
    if ntheta < nt2:
        raise ValueError("bsubu grid smaller than ntheta2")

    mmax = int(mmax_force)
    nmax = int(nmax_force)
    if mmax < 0 or nmax < 0:
        return bsubu[:, :nt2, :].copy(), bsubv[:, :nt2, :].copy()

    cosmui = np.asarray(trig.cosmui, dtype=acc_dtype)[:nt2, : mmax + 1]
    sinmui = np.asarray(trig.sinmui, dtype=acc_dtype)[:nt2, : mmax + 1]
    cosmu = np.asarray(trig.cosmu, dtype=acc_dtype)[:nt2, : mmax + 1]
    sinmu = np.asarray(trig.sinmu, dtype=acc_dtype)[:nt2, : mmax + 1]
    cosnv = np.asarray(trig.cosnv, dtype=acc_dtype)[:, : nmax + 1]
    sinnv = np.asarray(trig.sinnv, dtype=acc_dtype)[:, : nmax + 1]

    r0scale = float(getattr(trig, "r0scale", 1.0))
    base_dnorm = acc_dtype(1.0) / acc_dtype(r0scale**2)
    mnyq, nnyq = _jxbforce_nyquist_limits(trig)

    def _pshalf_from_s(s_full: np.ndarray) -> np.ndarray:
        if s_full.shape[0] < 2:
            return np.sqrt(np.maximum(s_full, 0.0))
        sh = 0.5 * (s_full[1:] + s_full[:-1])
        p = np.concatenate([sh[:1], sh], axis=0)
        return np.sqrt(np.maximum(p, 0.0))

    pshalf = None
    if s is not None:
        s_full = np.asarray(s, dtype=acc_dtype)
        pshalf = _pshalf_from_s(s_full)
        if pshalf.shape[0] > 1:
            pshalf[0] = pshalf[1]

    # If the requested filter spans the full available basis, return the
    # original fields to avoid introducing numerical drift.
    full_mmax, full_nmax = _jxbforce_nyquist_limits(trig)
    if mmax >= full_mmax and nmax >= full_nmax:
        return bsubu[:, :nt2, :].copy(), bsubv[:, :nt2, :].copy()

    bsubu_out = np.zeros((ns, nt2, nzeta), dtype=acc_dtype)
    bsubv_out = np.zeros((ns, nt2, nzeta), dtype=acc_dtype)

    for js in range(ns):
        bsubu_in = bsubu[js, :nt2, :]
        bsubv_in = bsubv[js, :nt2, :]
        for m in range(mmax + 1):
            for n in range(nmax + 1):
                dnorm1 = base_dnorm
                if mnyq > 0 and m == mnyq:
                    dnorm1 *= 0.5
                if nnyq > 0 and n == nnyq and n != 0:
                    dnorm1 *= 0.5
                # Undo odd-m sqrt(s) scaling (VMEC shalf factor).
                if (m % 2 == 1) and (pshalf is not None):
                    dnorm1 = dnorm1 / float(pshalf[js])

                bsubumn1 = acc_dtype(0.0)
                bsubumn2 = acc_dtype(0.0)
                bsubvmn1 = acc_dtype(0.0)
                bsubvmn2 = acc_dtype(0.0)

                for k in range(nzeta):
                    for j in range(nt2):
                        tcosi1 = cosmui[j, m] * cosnv[k, n] * dnorm1
                        tcosi2 = sinmui[j, m] * sinnv[k, n] * dnorm1
                        val_u = bsubu_in[j, k]
                        val_v = bsubv_in[j, k]
                        bsubumn1 += tcosi1 * val_u
                        bsubumn2 += tcosi2 * val_u
                        bsubvmn1 += tcosi1 * val_v
                        bsubvmn2 += tcosi2 * val_v

                for k in range(nzeta):
                    for j in range(nt2):
                        tcos1 = cosmu[j, m] * cosnv[k, n]
                        tcos2 = sinmu[j, m] * sinnv[k, n]
                        bsubu_out[js, j, k] += tcos1 * bsubumn1 + tcos2 * bsubumn2
                        bsubv_out[js, j, k] += tcos1 * bsubvmn1 + tcos2 * bsubvmn2

    return np.asarray(bsubu_out, dtype=float), np.asarray(bsubv_out, dtype=float)


def _filter_bsubuv_jxbforce_lasym_loop(
    *,
    bsubu: np.ndarray,
    bsubv: np.ndarray,
    trig,
    mmax_force: int,
    nmax_force: int,
    s: np.ndarray | None = None,
    bsubu_even: np.ndarray | None = None,
    bsubu_odd: np.ndarray | None = None,
    bsubv_even: np.ndarray | None = None,
    bsubv_odd: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Loop-accurate LASYM low-pass filter for bsubu/bsubv (jxbforce + fext_fft).

    Mirrors jxbforce.f for LASYM runs:
    1) contract full-grid fields to reduced-grid symmetric/antisymmetric channels
       (fsym_fft parity split),
    2) low-pass Fourier transform/inverse on the reduced grid,
    3) extend filtered channels back to full theta grid (fext_fft).
    """
    acc_dtype = np.longdouble
    bsubu = np.asarray(bsubu, dtype=acc_dtype)
    bsubv = np.asarray(bsubv, dtype=acc_dtype)
    if bsubu.shape != bsubv.shape:
        raise ValueError("bsubu/bsubv shape mismatch")
    have_parity_channels = (
        (bsubu_even is not None) and (bsubu_odd is not None) and (bsubv_even is not None) and (bsubv_odd is not None)
    )
    if have_parity_channels:
        bsubu_even = np.asarray(bsubu_even, dtype=acc_dtype)
        bsubu_odd = np.asarray(bsubu_odd, dtype=acc_dtype)
        bsubv_even = np.asarray(bsubv_even, dtype=acc_dtype)
        bsubv_odd = np.asarray(bsubv_odd, dtype=acc_dtype)
        if (
            bsubu_even.shape != bsubu.shape
            or bsubu_odd.shape != bsubu.shape
            or bsubv_even.shape != bsubu.shape
            or bsubv_odd.shape != bsubu.shape
        ):
            raise ValueError("LASYM bsub parity channel shape mismatch")

    ns, ntheta, nzeta = bsubu.shape
    nt2 = int(trig.ntheta2)
    nt1 = int(trig.ntheta1)
    nt3 = int(getattr(trig, "ntheta3", nt2))
    if ntheta < nt3:
        raise ValueError("LASYM bsubu grid smaller than ntheta3")

    mmax = int(mmax_force)
    nmax = int(nmax_force)
    if mmax < 0 or nmax < 0:
        return bsubu[:, :nt3, :].astype(float), bsubv[:, :nt3, :].astype(float)

    cosmui = np.asarray(trig.cosmui, dtype=acc_dtype)[:nt2, : mmax + 1]
    sinmui = np.asarray(trig.sinmui, dtype=acc_dtype)[:nt2, : mmax + 1]
    cosmu = np.asarray(trig.cosmu, dtype=acc_dtype)[:nt2, : mmax + 1]
    sinmu = np.asarray(trig.sinmu, dtype=acc_dtype)[:nt2, : mmax + 1]
    cosnv = np.asarray(trig.cosnv, dtype=acc_dtype)[:, : nmax + 1]
    sinnv = np.asarray(trig.sinnv, dtype=acc_dtype)[:, : nmax + 1]

    r0scale = float(getattr(trig, "r0scale", 1.0))
    base_dnorm = acc_dtype(1.0) / acc_dtype(r0scale**2)
    mnyq, nnyq = _jxbforce_nyquist_limits(trig)

    def _pshalf_from_s(s_full: np.ndarray) -> np.ndarray:
        if s_full.shape[0] < 2:
            return np.sqrt(np.maximum(s_full, 0.0))
        sh = 0.5 * (s_full[1:] + s_full[:-1])
        p = np.concatenate([sh[:1], sh], axis=0)
        return np.sqrt(np.maximum(p, 0.0))

    pshalf = None
    if s is not None:
        s_full = np.asarray(s, dtype=acc_dtype)
        pshalf = _pshalf_from_s(s_full)
        if pshalf.shape[0] > 1:
            pshalf[0] = pshalf[1]

    bsubu_out = np.zeros((ns, nt3, nzeta), dtype=acc_dtype)
    bsubv_out = np.zeros((ns, nt3, nzeta), dtype=acc_dtype)

    for js in range(ns):
        # Fortran fext/fsym paths use (zeta, theta) ordering.
        bu = np.asarray(bsubu[js, :nt3, :], dtype=acc_dtype).T  # (nzeta, ntheta3)
        bv = np.asarray(bsubv[js, :nt3, :], dtype=acc_dtype).T

        if have_parity_channels:
            bu0 = np.asarray(bsubu_even[js, :nt3, :], dtype=acc_dtype).T
            bv0 = np.asarray(bsubv_even[js, :nt3, :], dtype=acc_dtype).T
            if pshalf is not None:
                sh = acc_dtype(pshalf[js]) if pshalf[js] != 0.0 else acc_dtype(1.0)
            else:
                sh = acc_dtype(1.0)
            # VMEC stores odd channel as shalf*bsub*_odd before the immediate
            # in-loop divide by shalf in jxbforce.
            bu1 = np.asarray(bsubu_odd[js, :nt3, :], dtype=acc_dtype).T * sh
            bv1 = np.asarray(bsubv_odd[js, :nt3, :], dtype=acc_dtype).T * sh
            bu_ch = np.stack([bu0, bu1], axis=-1)
            bv_ch = np.stack([bv0, bv1], axis=-1)
        else:
            # Fallback path when only full bsub fields are available.
            bu_ch = np.stack([bu, bu], axis=-1)  # (nzeta, ntheta3, 2)
            bv_ch = np.stack([bv, bv], axis=-1)

        bu_s = np.zeros((nzeta, nt2, 2), dtype=acc_dtype)
        bu_a = np.zeros((nzeta, nt2, 2), dtype=acc_dtype)
        bv_s = np.zeros((nzeta, nt2, 2), dtype=acc_dtype)
        bv_a = np.zeros((nzeta, nt2, 2), dtype=acc_dtype)

        # fsym_fft contraction.
        for i in range(nt2):
            ir = 0 if i == 0 else (nt1 - i)
            for kz in range(nzeta):
                kzr = 0 if kz == 0 else (nzeta - kz)
                bu_a[kz, i, :] = 0.5 * (bu_ch[kz, i, :] - bu_ch[kzr, ir, :])
                bu_s[kz, i, :] = 0.5 * (bu_ch[kz, i, :] + bu_ch[kzr, ir, :])
                bv_a[kz, i, :] = 0.5 * (bv_ch[kz, i, :] - bv_ch[kzr, ir, :])
                bv_s[kz, i, :] = 0.5 * (bv_ch[kz, i, :] + bv_ch[kzr, ir, :])

        bsubua = np.zeros((nzeta, nt2, 2), dtype=acc_dtype)
        bsubva = np.zeros((nzeta, nt2, 2), dtype=acc_dtype)

        for m in range(mmax + 1):
            mparity = m & 1
            for n in range(nmax + 1):
                dnorm1 = base_dnorm
                if mnyq > 0 and m == mnyq:
                    dnorm1 *= 0.5
                if nnyq > 0 and n == nnyq and n != 0:
                    dnorm1 *= 0.5

                bsubumn1 = acc_dtype(0.0)
                bsubumn2 = acc_dtype(0.0)
                bsubvmn1 = acc_dtype(0.0)
                bsubvmn2 = acc_dtype(0.0)
                bsubumn3 = acc_dtype(0.0)
                bsubumn4 = acc_dtype(0.0)
                bsubvmn3 = acc_dtype(0.0)
                bsubvmn4 = acc_dtype(0.0)

                for k in range(nzeta):
                    for j in range(nt2):
                        tsini1 = sinmui[j, m] * cosnv[k, n] * dnorm1
                        tsini2 = cosmui[j, m] * sinnv[k, n] * dnorm1
                        tcosi1 = cosmui[j, m] * cosnv[k, n] * dnorm1
                        tcosi2 = sinmui[j, m] * sinnv[k, n] * dnorm1

                        bsubumn1 += tcosi1 * bu_s[k, j, mparity]
                        bsubumn2 += tcosi2 * bu_s[k, j, mparity]
                        bsubvmn1 += tcosi1 * bv_s[k, j, mparity]
                        bsubvmn2 += tcosi2 * bv_s[k, j, mparity]

                        bsubumn3 += tsini1 * bu_a[k, j, mparity]
                        bsubumn4 += tsini2 * bu_a[k, j, mparity]
                        bsubvmn3 += tsini1 * bv_a[k, j, mparity]
                        bsubvmn4 += tsini2 * bv_a[k, j, mparity]

                for k in range(nzeta):
                    for j in range(nt2):
                        tcos1 = cosmu[j, m] * cosnv[k, n]
                        tcos2 = sinmu[j, m] * sinnv[k, n]
                        bsubua[k, j, 0] += tcos1 * bsubumn1 + tcos2 * bsubumn2
                        bsubva[k, j, 0] += tcos1 * bsubvmn1 + tcos2 * bsubvmn2

                        tsin1 = sinmu[j, m] * cosnv[k, n]
                        tsin2 = cosmu[j, m] * sinnv[k, n]
                        bsubua[k, j, 1] += tsin1 * bsubumn3 + tsin2 * bsubumn4
                        bsubva[k, j, 1] += tsin1 * bsubvmn3 + tsin2 * bsubvmn4

        # fext_fft extension to full theta grid.
        bu_full = np.zeros((nzeta, nt3), dtype=acc_dtype)
        bv_full = np.zeros((nzeta, nt3), dtype=acc_dtype)
        bu_full[:, :nt2] = bsubua[:, :, 0] + bsubua[:, :, 1]
        bv_full[:, :nt2] = bsubva[:, :, 0] + bsubva[:, :, 1]
        for i in range(nt2, nt3):
            ir = nt1 - i
            for kz in range(nzeta):
                kzr = 0 if kz == 0 else (nzeta - kz)
                bu_full[kz, i] = bsubua[kzr, ir, 0] - bsubua[kzr, ir, 1]
                bv_full[kz, i] = bsubva[kzr, ir, 0] - bsubva[kzr, ir, 1]

        bsubu_out[js, :, :] = bu_full.T
        bsubv_out[js, :, :] = bv_full.T

    return np.asarray(bsubu_out, dtype=float), np.asarray(bsubv_out, dtype=float)


def _jxbforce_bsubsu_bsubsv_loop(
    *,
    bsubs: np.ndarray,
    trig,
    mmax_force: int,
    nmax_force: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Loop-based bsubsu/bsubsv reconstruction (jxbforce)."""
    # Cancellation in jxbforce transforms is severe for some equilibria
    # (e.g. QI_nfp2 near edge). Accumulate in long double, then cast back.
    acc_dtype = np.longdouble
    bsubs = np.asarray(bsubs, dtype=acc_dtype)
    ns, ntheta, nzeta = bsubs.shape
    nt2 = int(trig.ntheta2)
    if ntheta < nt2:
        raise ValueError("bsubs grid smaller than ntheta2")

    mmax = int(mmax_force)
    nmax = int(nmax_force)
    if mmax < 0 or nmax < 0:
        return np.zeros((ns, nt2, nzeta), dtype=float), np.zeros((ns, nt2, nzeta), dtype=float)

    cosmui = np.asarray(trig.cosmui, dtype=acc_dtype)[:nt2, : mmax + 1]
    sinmui = np.asarray(trig.sinmui, dtype=acc_dtype)[:nt2, : mmax + 1]
    cosmu = np.asarray(trig.cosmu, dtype=acc_dtype)[:nt2, : mmax + 1]
    sinmu = np.asarray(trig.sinmu, dtype=acc_dtype)[:nt2, : mmax + 1]
    cosmum = np.asarray(trig.cosmum, dtype=acc_dtype)[:nt2, : mmax + 1]
    sinmum = np.asarray(trig.sinmum, dtype=acc_dtype)[:nt2, : mmax + 1]
    cosnv = np.asarray(trig.cosnv, dtype=acc_dtype)[:, : nmax + 1]
    sinnv = np.asarray(trig.sinnv, dtype=acc_dtype)[:, : nmax + 1]
    cosnvn = np.asarray(trig.cosnvn, dtype=acc_dtype)[:, : nmax + 1]
    sinnvn = np.asarray(trig.sinnvn, dtype=acc_dtype)[:, : nmax + 1]

    r0scale = float(getattr(trig, "r0scale", 1.0))
    base_dnorm = acc_dtype(1.0 / (r0scale**2))
    mnyq, nnyq = _jxbforce_nyquist_limits(trig)

    bsubsu = np.zeros((ns, nt2, nzeta), dtype=acc_dtype)
    bsubsv = np.zeros((ns, nt2, nzeta), dtype=acc_dtype)

    for js in range(ns):
        bsubs_s = bsubs[js, :nt2, :]
        for m in range(mmax + 1):
            for n in range(nmax + 1):
                dnorm1 = base_dnorm
                if mnyq > 0 and m == mnyq:
                    dnorm1 *= 0.5
                if nnyq > 0 and n == nnyq and n != 0:
                    dnorm1 *= 0.5

                bsubsmn1 = acc_dtype(0.0)
                bsubsmn2 = acc_dtype(0.0)
                for k in range(nzeta):
                    for j in range(nt2):
                        tsini1 = sinmui[j, m] * cosnv[k, n] * dnorm1
                        tsini2 = cosmui[j, m] * sinnv[k, n] * dnorm1
                        val = bsubs_s[j, k]
                        bsubsmn1 += tsini1 * val
                        bsubsmn2 += tsini2 * val

                for k in range(nzeta):
                    for j in range(nt2):
                        tcosm1 = cosmum[j, m] * cosnv[k, n]
                        tcosm2 = sinmum[j, m] * sinnv[k, n]
                        bsubsu[js, j, k] += tcosm1 * bsubsmn1 + tcosm2 * bsubsmn2
                        tcosn1 = sinmu[j, m] * sinnvn[k, n]
                        tcosn2 = cosmu[j, m] * cosnvn[k, n]
                        bsubsv[js, j, k] += tcosn1 * bsubsmn1 + tcosn2 * bsubsmn2

    return np.asarray(bsubsu, dtype=float), np.asarray(bsubsv, dtype=float)
