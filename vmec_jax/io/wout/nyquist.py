"""VMEC ``wrout`` Nyquist and symmetry helpers.

These helpers are pure NumPy translations of small VMEC ``wrout.f`` and
``symforce.f`` transform kernels.  They live outside the top-level WOUT writer
so the large NetCDF/output path can be tested and refactored independently from
the Fourier parity utilities.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from vmec_jax.modes import ModeTable


class SymmetricNyquistCoefficientPayload(NamedTuple):
    """Nyquist coefficients for a stellarator-symmetric WOUT output block."""

    gmnc: np.ndarray
    gmns: np.ndarray
    bsupumnc: np.ndarray
    bsupumns: np.ndarray
    bsupvmnc: np.ndarray
    bsupvmns: np.ndarray
    bsubumnc: np.ndarray
    bsubumns: np.ndarray
    bsubvmnc: np.ndarray
    bsubvmns: np.ndarray
    bsubsmns: np.ndarray
    bsubsmnc: np.ndarray
    bmnc: np.ndarray
    bmns: np.ndarray


def apply_nyquist_half_weight(
    *,
    coeff_cos: np.ndarray,
    coeff_sin: np.ndarray,
    modes,
    trig,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply VMEC Nyquist normalization for edge modes."""
    coeff_cos = np.asarray(coeff_cos, dtype=float)
    coeff_sin = np.asarray(coeff_sin, dtype=float)
    if coeff_cos.ndim != 2 or coeff_sin.ndim != 2:
        raise ValueError("Expected coeff arrays with shape (ns, K)")

    m = np.asarray(modes.m, dtype=int)
    n = np.asarray(modes.n, dtype=int)
    if m.size == 0:
        return coeff_cos, coeff_sin

    m_nyq = int(np.max(m)) if m.size else -1
    n_nyq = int(np.max(np.abs(n))) if n.size else -1

    mask = np.zeros_like(m, dtype=bool)
    if m_nyq > 0:
        mask |= m == m_nyq
    if n_nyq > 0:
        mask |= np.abs(n) == n_nyq
    if not np.any(mask):
        return coeff_cos, coeff_sin

    coeff_cos = coeff_cos.copy()
    coeff_sin = coeff_sin.copy()
    coeff_cos[:, mask] *= 0.5
    coeff_sin[:, mask] *= 0.5
    return coeff_cos, coeff_sin


def vmec_wrout_nyquist_cos_coeffs(
    *,
    f: np.ndarray,
    modes: ModeTable,
    trig,
) -> np.ndarray:
    """VMEC ``wrout``-style Nyquist analysis for cosine coefficients."""
    f = np.asarray(f, dtype=float)
    if f.ndim != 3:
        raise ValueError("Expected f with shape (ns, ntheta, nzeta)")
    m = np.asarray(modes.m, dtype=int)
    n = np.asarray(modes.n, dtype=int)
    if m.size == 0:
        return np.zeros((f.shape[0], 0), dtype=float)

    nt2 = int(trig.ntheta2)
    if int(f.shape[1]) < nt2:
        raise ValueError("Input theta grid is smaller than VMEC ntheta2")
    f = f[:, :nt2, :]

    cosmui = np.asarray(trig.cosmui, dtype=float)[:nt2, :]
    sinmui = np.asarray(trig.sinmui, dtype=float)[:nt2, :]
    cosnv = np.asarray(trig.cosnv, dtype=float)
    sinnv = np.asarray(trig.sinnv, dtype=float)

    mmax = int(np.max(m))
    nmax = int(np.max(np.abs(n)))
    if cosmui.shape[1] <= mmax or cosnv.shape[1] <= nmax:
        raise ValueError("Trig tables do not cover Nyquist mode limits")

    # VMEC halves cosmui(:,mnyq) and cosnv(:,nnyq) during wrout when lnyquist.
    mnyq = cosmui.shape[1] - 1
    if mnyq > 0:
        cosmui = cosmui.copy()
        cosmui[:, mnyq] *= 0.5
    nnyq = cosnv.shape[1] - 1
    if nnyq > 0:
        cosnv = cosnv.copy()
        cosnv[:, nnyq] *= 0.5

    f_theta_cos = np.einsum("sik,im->smk", f, cosmui, optimize=True)
    f_theta_sin = np.einsum("sik,im->smk", f, sinmui, optimize=True)
    cos_zeta = np.einsum("smk,kn->smn", f_theta_cos, cosnv, optimize=True)
    sin_zeta = np.einsum("smk,kn->smn", f_theta_sin, sinnv, optimize=True)

    n_abs = np.abs(n)
    sgn = np.where(n < 0, -1.0, 1.0)
    cos_part = cos_zeta[:, m, n_abs]
    sin_part = sin_zeta[:, m, n_abs]
    coeff = cos_part + sgn[None, :] * sin_part

    mscale = np.asarray(trig.mscale, dtype=float)
    nscale = np.asarray(trig.nscale, dtype=float)
    tmult = 0.5 / float(getattr(trig, "r0scale", 1.0)) ** 2
    if int(getattr(trig, "ntheta3", nt2)) > nt2:
        # VMEC wrout doubles tmult for LASYM after switching to full-grid dnorm.
        tmult *= 2.0
    dmult = mscale[m] * nscale[n_abs] * tmult
    dmult = np.where((m == 0) | (n == 0), 2.0 * dmult, dmult)
    coeff = coeff * dmult[None, :]
    return np.asarray(coeff, dtype=float)


def minimal_wout_symmetric_nyquist_coefficients(
    *,
    bc: Any,
    bsubu_out: np.ndarray,
    bsubv_out: np.ndarray,
    bsubs_full: np.ndarray,
    pres: np.ndarray,
    ns: int,
    modes: ModeTable,
    trig: Any,
    use_loop: bool,
) -> SymmetricNyquistCoefficientPayload:
    """Return VMEC ``wrout`` Nyquist coefficients for ``LASYM = F``."""

    z2 = np.zeros((int(ns), int(modes.K)), dtype=float)
    gmnc = vmec_wrout_nyquist_cos_coeffs(f=np.asarray(bc.jac.sqrtg), modes=modes, trig=trig)
    bsupu_out = np.asarray(bc.bsupu)
    bsupv_out = np.asarray(bc.bsupv)
    bsupumnc = vmec_wrout_nyquist_cos_coeffs(f=bsupu_out, modes=modes, trig=trig)
    bsupvmnc = vmec_wrout_nyquist_cos_coeffs(f=bsupv_out, modes=modes, trig=trig)
    bsubumnc = vmec_wrout_nyquist_cos_coeffs(f=bsubu_out, modes=modes, trig=trig)
    bsubvmnc = vmec_wrout_nyquist_cos_coeffs(f=bsubv_out, modes=modes, trig=trig)
    if use_loop:
        bsubsmns = vmec_wrout_nyquist_sin_coeffs_loop(f=bsubs_full, modes=modes, trig=trig)
    else:
        bsubsmns = vmec_wrout_nyquist_sin_coeffs(f=bsubs_full, modes=modes, trig=trig)

    pres_h = np.asarray(pres, dtype=float)[:, None, None]
    bmag = np.sqrt(2.0 * np.abs(np.asarray(bc.bsq) - pres_h))
    bmnc = vmec_wrout_nyquist_cos_coeffs(f=bmag, modes=modes, trig=trig)

    if gmnc.shape[0] > 0:
        gmnc[0, :] = 0.0
        bsupumnc[0, :] = 0.0
        bsupvmnc[0, :] = 0.0
        bsubumnc[0, :] = 0.0
        bsubvmnc[0, :] = 0.0
        bmnc[0, :] = 0.0

    gmns = z2.copy()
    bsupumns = z2.copy()
    bsupvmns = z2.copy()
    bsubumns = z2.copy()
    bsubvmns = z2.copy()
    bsubsmnc = z2.copy()
    bmns = z2.copy()
    if ns > 2:
        bsubsmns[0, :] = 2.0 * bsubsmns[1, :] - bsubsmns[2, :]

    return SymmetricNyquistCoefficientPayload(
        gmnc=gmnc,
        gmns=gmns,
        bsupumnc=bsupumnc,
        bsupumns=bsupumns,
        bsupvmnc=bsupvmnc,
        bsupvmns=bsupvmns,
        bsubumnc=bsubumnc,
        bsubumns=bsubumns,
        bsubvmnc=bsubvmnc,
        bsubvmns=bsubvmns,
        bsubsmns=bsubsmns,
        bsubsmnc=bsubsmnc,
        bmnc=bmnc,
        bmns=bmns,
    )


def minimal_wout_lasym_nyquist_coefficients(
    *,
    bc: Any,
    bsubu_out: np.ndarray,
    bsubv_out: np.ndarray,
    bsupu_out: np.ndarray,
    bsupv_out: np.ndarray,
    bsubs_full: np.ndarray,
    bsubv_asym_source: np.ndarray | None,
    pres: np.ndarray,
    ns: int,
    mpol: int,
    ntor: int,
    modes: ModeTable,
    trig: Any,
    use_loop: bool,
) -> SymmetricNyquistCoefficientPayload:
    """Return VMEC ``wrout`` Nyquist coefficients for ``LASYM = T``."""

    pres_h = np.asarray(pres, dtype=float)[:, None, None]
    bmag = np.sqrt(2.0 * np.abs(np.asarray(bc.bsq) - pres_h))
    sqrtg = np.asarray(bc.jac.sqrtg)
    bsubu_sym, bsubu_asym = vmec_symoutput_split(f=bsubu_out, trig=trig)
    bsubv_sym, bsubv_asym = vmec_symoutput_split(f=bsubv_out, trig=trig)
    if bsubv_asym_source is not None:
        _, bsubv_asym = vmec_symoutput_split(f=bsubv_asym_source, trig=trig)
    bsupu_sym, bsupu_asym = vmec_symoutput_split(f=bsupu_out, trig=trig)
    bsupv_sym, bsupv_asym = vmec_symoutput_split(f=bsupv_out, trig=trig)
    bsubs_sym, bsubs_asym = vmec_symoutput_split(f=bsubs_full, trig=trig, reversed_sym=True)
    bmag_sym, bmag_asym = vmec_symoutput_split(f=bmag, trig=trig)
    sqrtg_sym, sqrtg_asym = vmec_symoutput_split(f=sqrtg, trig=trig)
    if use_loop:
        sym_coeffs = vmec_wrout_nyquist_lasym_loop(
            bsq=bmag_sym,
            gsqrt=sqrtg_sym,
            bsubu=bsubu_sym,
            bsubv=bsubv_sym,
            bsubs=bsubs_sym,
            bsupu=bsupu_sym,
            bsupv=bsupv_sym,
            modes=modes,
            trig=trig,
        )
        asym_coeffs = vmec_wrout_nyquist_lasym_loop(
            bsq=bmag_asym,
            gsqrt=sqrtg_asym,
            bsubu=bsubu_asym,
            bsubv=bsubv_asym,
            bsubs=bsubs_asym,
            bsupu=bsupu_asym,
            bsupv=bsupv_asym,
            modes=modes,
            trig=trig,
        )
        gmnc, bmnc, bsubumnc, bsubvmnc, bsubsmns, bsupumnc, bsupvmnc = (
            sym_coeffs[key]
            for key in ("gmnc", "bmnc", "bsubumnc", "bsubvmnc", "bsubsmns", "bsupumnc", "bsupvmnc")
        )
        gmns, bmns, bsubumns, bsubvmns, bsubsmnc, bsupumns, bsupvmns = (
            asym_coeffs[key]
            for key in ("gmns", "bmns", "bsubumns", "bsubvmns", "bsubsmnc", "bsupumns", "bsupvmns")
        )
    else:
        gmnc = vmec_wrout_nyquist_cos_coeffs(f=sqrtg_sym, modes=modes, trig=trig)
        bmnc = vmec_wrout_nyquist_cos_coeffs(f=bmag_sym, modes=modes, trig=trig)
        bsubumnc = vmec_wrout_nyquist_cos_coeffs(f=bsubu_sym, modes=modes, trig=trig)
        bsubvmnc = vmec_wrout_nyquist_cos_coeffs(f=bsubv_sym, modes=modes, trig=trig)
        bsupumnc = vmec_wrout_nyquist_cos_coeffs(f=bsupu_sym, modes=modes, trig=trig)
        bsupvmnc = vmec_wrout_nyquist_cos_coeffs(f=bsupv_sym, modes=modes, trig=trig)
        bsubsmns = vmec_wrout_nyquist_sin_coeffs(f=bsubs_sym, modes=modes, trig=trig)
        gmns = vmec_wrout_nyquist_sin_coeffs(f=sqrtg_asym, modes=modes, trig=trig)
        bmns = vmec_wrout_nyquist_sin_coeffs(f=bmag_asym, modes=modes, trig=trig)
        bsubumns = vmec_wrout_nyquist_sin_coeffs(f=bsubu_asym, modes=modes, trig=trig)
        bsubvmns = vmec_wrout_nyquist_sin_coeffs(f=bsubv_asym, modes=modes, trig=trig)
        bsupumns = vmec_wrout_nyquist_sin_coeffs(f=bsupu_asym, modes=modes, trig=trig)
        bsupvmns = vmec_wrout_nyquist_sin_coeffs(f=bsupv_asym, modes=modes, trig=trig)
        bsubsmnc = vmec_wrout_nyquist_cos_coeffs(f=bsubs_asym, modes=modes, trig=trig)

    m_mask = np.asarray(modes.m, dtype=int)
    n_mask = np.asarray(modes.n, dtype=int)
    mask_bsub = (m_mask >= int(mpol)) | (np.abs(n_mask) > int(ntor))
    if np.any(mask_bsub):
        bsubumnc[:, mask_bsub] = 0.0
        bsubumns[:, mask_bsub] = 0.0
        bsubvmnc[:, mask_bsub] = 0.0
        bsubvmns[:, mask_bsub] = 0.0
    bsubumnc, bsubvmnc, bsubumns, bsubvmns = vmec_wrout_lasym_bsubuv_output_scale(
        bsubumnc=bsubumnc,
        bsubvmnc=bsubvmnc,
        bsubumns=bsubumns,
        bsubvmns=bsubvmns,
    )
    for arr in (gmnc, bmnc, bsubumnc, bsubvmnc, bsupumnc, bsupvmnc, gmns, bmns, bsubumns, bsubvmns, bsupumns, bsupvmns):
        if arr.shape[0] > 0:
            arr[0, :] = 0.0
    if (not use_loop) and (int(ns) > 2):
        bsubsmns[0, :] = 2.0 * bsubsmns[1, :] - bsubsmns[2, :]
        bsubsmnc[0, :] = 2.0 * bsubsmnc[1, :] - bsubsmnc[2, :]
    return SymmetricNyquistCoefficientPayload(
        gmnc=gmnc,
        gmns=gmns,
        bsupumnc=bsupumnc,
        bsupumns=bsupumns,
        bsupvmnc=bsupvmnc,
        bsupvmns=bsupvmns,
        bsubumnc=bsubumnc,
        bsubumns=bsubumns,
        bsubvmnc=bsubvmnc,
        bsubvmns=bsubvmns,
        bsubsmns=bsubsmns,
        bsubsmnc=bsubsmnc,
        bmnc=bmnc,
        bmns=bmns,
    )


def vmec_symoutput_split(
    *,
    f: np.ndarray,
    trig,
    reversed_sym: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """VMEC ``symoutput`` split into symmetric/antisymmetric parts on ``[0, pi]``."""
    f = np.asarray(f, dtype=float)
    if f.ndim != 3:
        raise ValueError("Expected f with shape (ns, ntheta, nzeta)")
    nt2 = int(trig.ntheta2)
    nt1 = int(trig.ntheta1)
    if int(f.shape[1]) < nt2:
        raise ValueError("Input theta grid is smaller than VMEC ntheta2")
    nzeta = int(f.shape[2])
    if nzeta <= 0:
        return f[:, :nt2, :].copy(), f[:, :nt2, :].copy()

    i0 = np.arange(nt2, dtype=int)
    ir0 = np.where(i0 == 0, 0, nt1 - i0)
    kk = (nzeta - np.arange(nzeta, dtype=int)) % nzeta
    f_half = f[:, :nt2, :]
    f_ref = f[:, ir0, :][:, :, kk]
    if reversed_sym:
        sym = 0.5 * (f_half - f_ref)
        asym = 0.5 * (f_half + f_ref)
    else:
        sym = 0.5 * (f_half + f_ref)
        asym = 0.5 * (f_half - f_ref)
    return sym, asym


def vmec_symforce_apply(
    *,
    f: np.ndarray,
    trig,
    kind: str,
) -> np.ndarray:
    """Apply VMEC ``symforce.f`` to a full-grid real-space field."""
    f = np.asarray(f, dtype=float)
    if f.ndim != 3:
        raise ValueError("Expected f with shape (ns, ntheta, nzeta)")
    nt2 = int(trig.ntheta2)
    nt1 = int(trig.ntheta1)
    if int(f.shape[1]) < nt2:
        raise ValueError("Input theta grid is smaller than VMEC ntheta2")
    nzeta = int(f.shape[2])
    if nzeta <= 0:
        return f.copy()

    i0 = np.arange(nt2, dtype=int)
    ir0 = np.where(i0 == 0, 0, nt1 - i0)
    kk = (nzeta - np.arange(nzeta, dtype=int)) % nzeta
    f_half = f[:, :nt2, :]
    f_ref = f[:, ir0, :][:, :, kk]

    if kind in ("ars", "bzs", "bls", "rcs", "czs", "cls"):
        f_new = 0.5 * (f_half + f_ref)
    elif kind in ("brs", "azs", "zcs", "crs"):
        f_new = 0.5 * (f_half - f_ref)
    else:  # pragma: no cover
        raise ValueError(f"symforce: unknown kind {kind!r}")

    f_sym = np.array(f, copy=True)
    f_sym[:, :nt2, :] = f_new
    return f_sym


def vmec_symforce_antisym(
    *,
    f: np.ndarray,
    trig,
    kind: str,
    base: np.ndarray | None = None,
) -> np.ndarray:
    """Apply VMEC ``symforce.f`` antisymmetric output (ara/bra/etc)."""
    f = np.asarray(f, dtype=float)
    if f.ndim != 3:
        raise ValueError("Expected f with shape (ns, ntheta, nzeta)")
    nt2 = int(trig.ntheta2)
    nt1 = int(trig.ntheta1)
    if int(f.shape[1]) < nt2:
        raise ValueError("Input theta grid is smaller than VMEC ntheta2")
    nzeta = int(f.shape[2])
    if nzeta <= 0:
        return np.asarray(base, dtype=float).copy() if base is not None else f.copy()

    if base is None:
        out = np.zeros_like(f)
    else:
        base = np.asarray(base, dtype=float)
        if base.shape != f.shape:
            raise ValueError("symforce antisym base shape mismatch")
        out = np.array(base, copy=True)

    i0 = np.arange(nt2, dtype=int)
    ir0 = np.where(i0 == 0, 0, nt1 - i0)
    kk = (nzeta - np.arange(nzeta, dtype=int)) % nzeta
    f_half = f[:, :nt2, :]
    f_ref = f[:, ir0, :][:, :, kk]

    if kind in ("ars", "bzs", "bls", "rcs", "czs", "cls"):
        f_new = 0.5 * (f_half - f_ref)
    elif kind in ("brs", "azs", "zcs", "crs"):
        f_new = 0.5 * (f_half + f_ref)
    else:  # pragma: no cover
        raise ValueError(f"symforce: unknown kind {kind!r}")

    out[:, :nt2, :] = f_new
    return out


def vmec_symoutput_expand(
    *,
    sym: np.ndarray,
    asym: np.ndarray | None,
    trig,
) -> np.ndarray:
    """Expand VMEC ``symoutput`` parts back to the full ``[0, 2*pi)`` theta grid."""
    sym = np.asarray(sym, dtype=float)
    if sym.ndim != 3:
        raise ValueError("Expected sym with shape (ns, ntheta2, nzeta)")
    if asym is None:
        asym = np.zeros_like(sym)
    else:
        asym = np.asarray(asym, dtype=float)
        if asym.shape != sym.shape:
            raise ValueError("sym/asym shape mismatch")

    ns, nt2, nzeta = sym.shape
    nt1 = int(getattr(trig, "ntheta1", nt2))
    nt3 = int(getattr(trig, "ntheta3", nt2))
    if nt3 < nt2:
        nt3 = nt2

    full = np.zeros((ns, nt3, nzeta), dtype=float)
    full[:, :nt2, :] = sym + asym
    if nt3 == nt2:
        return full

    i0 = np.arange(nt2, dtype=int)
    ir0 = np.where(i0 == 0, 0, nt1 - i0)
    kk = (nzeta - np.arange(nzeta, dtype=int)) % nzeta
    mask = ir0 >= nt2
    if np.any(mask):
        ir = ir0[mask]
        sym_ref = sym[:, mask, :][:, :, kk]
        asym_ref = asym[:, mask, :][:, :, kk]
        full[:, ir, :] = sym_ref - asym_ref
    return full


def vmec_wrout_nyquist_sin_coeffs(
    *,
    f: np.ndarray,
    modes: ModeTable,
    trig,
) -> np.ndarray:
    """VMEC ``wrout``-style Nyquist analysis for sine coefficients."""
    f = np.asarray(f, dtype=float)
    if f.ndim != 3:
        raise ValueError("Expected f with shape (ns, ntheta, nzeta)")
    m = np.asarray(modes.m, dtype=int)
    n = np.asarray(modes.n, dtype=int)
    if m.size == 0:
        return np.zeros((f.shape[0], 0), dtype=float)

    nt2 = int(trig.ntheta2)
    if int(f.shape[1]) < nt2:
        raise ValueError("Input theta grid is smaller than VMEC ntheta2")
    f = f[:, :nt2, :]

    cosmui = np.asarray(trig.cosmui, dtype=float)[:nt2, :]
    sinmui = np.asarray(trig.sinmui, dtype=float)[:nt2, :]
    cosnv = np.asarray(trig.cosnv, dtype=float)
    sinnv = np.asarray(trig.sinnv, dtype=float)

    mmax = int(np.max(m))
    nmax = int(np.max(np.abs(n)))
    if cosmui.shape[1] <= mmax or cosnv.shape[1] <= nmax:
        raise ValueError("Trig tables do not cover Nyquist mode limits")

    # VMEC halves cosmui(:,mnyq) and cosnv(:,nnyq) during wrout when lnyquist.
    mnyq = cosmui.shape[1] - 1
    if mnyq > 0:
        cosmui = cosmui.copy()
        cosmui[:, mnyq] *= 0.5
    nnyq = cosnv.shape[1] - 1
    if nnyq > 0:
        cosnv = cosnv.copy()
        cosnv[:, nnyq] *= 0.5

    f_theta_sin = np.einsum("sik,im->smk", f, sinmui, optimize=True)
    f_theta_cos = np.einsum("sik,im->smk", f, cosmui, optimize=True)
    cos_zeta = np.einsum("smk,kn->smn", f_theta_sin, cosnv, optimize=True)
    sin_zeta = np.einsum("smk,kn->smn", f_theta_cos, sinnv, optimize=True)

    n_abs = np.abs(n)
    sgn = np.where(n < 0, -1.0, 1.0)
    cos_part = cos_zeta[:, m, n_abs]
    sin_part = sin_zeta[:, m, n_abs]
    coeff = cos_part - sgn[None, :] * sin_part

    mscale = np.asarray(trig.mscale, dtype=float)
    nscale = np.asarray(trig.nscale, dtype=float)
    tmult = 0.5 / float(getattr(trig, "r0scale", 1.0)) ** 2
    if int(getattr(trig, "ntheta3", nt2)) > nt2:
        # VMEC wrout doubles tmult for LASYM after switching to full-grid dnorm.
        tmult *= 2.0
    dmult = mscale[m] * nscale[n_abs] * tmult
    dmult = np.where((m == 0) | (n == 0), 2.0 * dmult, dmult)
    coeff = coeff * dmult[None, :]
    return np.asarray(coeff, dtype=float)


def vmec_wrout_nyquist_lasym_loop(
    *,
    bsq: np.ndarray,
    gsqrt: np.ndarray,
    bsubu: np.ndarray,
    bsubv: np.ndarray,
    bsubs: np.ndarray,
    bsupu: np.ndarray,
    bsupv: np.ndarray,
    modes: ModeTable,
    trig,
) -> dict[str, np.ndarray]:
    """VMEC ``wrout.f`` LASYM loop for Nyquist coefficients."""
    bsq = np.asarray(bsq, dtype=float)
    gsqrt = np.asarray(gsqrt, dtype=float)
    bsubu = np.asarray(bsubu, dtype=float)
    bsubv = np.asarray(bsubv, dtype=float)
    bsubs = np.asarray(bsubs, dtype=float)
    bsupu = np.asarray(bsupu, dtype=float)
    bsupv = np.asarray(bsupv, dtype=float)

    if bsq.ndim != 3:
        raise ValueError("Expected bsq with shape (ns, ntheta, nzeta)")
    ns, nt2, nzeta = bsq.shape
    if int(trig.ntheta2) != nt2:
        raise ValueError("lasym wrout expects reduced theta grid (ntheta2)")

    m = np.asarray(modes.m, dtype=int)
    n = np.asarray(modes.n, dtype=int)
    if m.size == 0:
        z = np.zeros((ns, 0), dtype=float)
        return {
            "gmnc": z.copy(),
            "bmnc": z.copy(),
            "bsubumnc": z.copy(),
            "bsubvmnc": z.copy(),
            "bsubsmns": z.copy(),
            "bsupumnc": z.copy(),
            "bsupvmnc": z.copy(),
            "gmns": z.copy(),
            "bmns": z.copy(),
            "bsubumns": z.copy(),
            "bsubvmns": z.copy(),
            "bsubsmnc": z.copy(),
            "bsupumns": z.copy(),
            "bsupvmns": z.copy(),
        }

    cosmui = np.asarray(trig.cosmui, dtype=float)[:nt2, :]
    sinmui = np.asarray(trig.sinmui, dtype=float)[:nt2, :]
    cosnv = np.asarray(trig.cosnv, dtype=float)
    sinnv = np.asarray(trig.sinnv, dtype=float)
    mscale = np.asarray(trig.mscale, dtype=float)
    nscale = np.asarray(trig.nscale, dtype=float)
    tmult = 0.5 / float(getattr(trig, "r0scale", 1.0)) ** 2
    if int(getattr(trig, "ntheta3", nt2)) > nt2:
        # VMEC wrout doubles tmult for LASYM after switching to full-grid dnorm.
        tmult *= 2.0

    mnyq = cosmui.shape[1] - 1
    if mnyq > 0:
        cosmui = cosmui.copy()
        cosmui[:, mnyq] *= 0.5
    nnyq = cosnv.shape[1] - 1
    if nnyq > 0:
        cosnv = cosnv.copy()
        cosnv[:, nnyq] *= 0.5

    K = int(m.size)
    gmnc = np.zeros((ns, K), dtype=float)
    bmnc = np.zeros((ns, K), dtype=float)
    bsubumnc = np.zeros((ns, K), dtype=float)
    bsubvmnc = np.zeros((ns, K), dtype=float)
    bsubsmns = np.zeros((ns, K), dtype=float)
    bsupumnc = np.zeros((ns, K), dtype=float)
    bsupvmnc = np.zeros((ns, K), dtype=float)
    gmns = np.zeros((ns, K), dtype=float)
    bmns = np.zeros((ns, K), dtype=float)
    bsubumns = np.zeros((ns, K), dtype=float)
    bsubvmns = np.zeros((ns, K), dtype=float)
    bsubsmnc = np.zeros((ns, K), dtype=float)
    bsupumns = np.zeros((ns, K), dtype=float)
    bsupvmns = np.zeros((ns, K), dtype=float)

    for js in range(1, ns):
        bsq_s = bsq[js]
        gsqrt_s = gsqrt[js]
        bsubu_s = bsubu[js]
        bsubv_s = bsubv[js]
        bsubs_s = bsubs[js]
        bsupu_s = bsupu[js]
        bsupv_s = bsupv[js]
        for mn in range(K):
            mval = int(m[mn])
            nval = int(n[mn])
            n1 = abs(nval)
            dmult = mscale[mval] * nscale[n1] * tmult
            if mval == 0 or nval == 0:
                dmult *= 2.0
            sgn = 1.0 if nval >= 0 else -1.0

            gmn = bmn = bsubumn = bsubvmn = bsubsmn = bsupumn = bsupvmn = 0.0
            gmn_a = bmn_a = bsubumn_a = bsubvmn_a = bsubsmn_a = bsupumn_a = bsupvmn_a = 0.0

            for j in range(nt2):
                cosmu_j = cosmui[j, mval]
                sinmu_j = sinmui[j, mval]
                for k in range(nzeta):
                    tcosi = dmult * (cosmu_j * cosnv[k, n1] + sgn * sinmu_j * sinnv[k, n1])
                    tsini = dmult * (sinmu_j * cosnv[k, n1] - sgn * cosmu_j * sinnv[k, n1])
                    gmn += tcosi * gsqrt_s[j, k]
                    bmn += tcosi * bsq_s[j, k]
                    bsubumn += tcosi * bsubu_s[j, k]
                    bsubvmn += tcosi * bsubv_s[j, k]
                    bsubsmn += tsini * bsubs_s[j, k]
                    bsupumn += tcosi * bsupu_s[j, k]
                    bsupvmn += tcosi * bsupv_s[j, k]
                    gmn_a += tsini * gsqrt_s[j, k]
                    bmn_a += tsini * bsq_s[j, k]
                    bsubumn_a += tsini * bsubu_s[j, k]
                    bsubvmn_a += tsini * bsubv_s[j, k]
                    bsubsmn_a += tcosi * bsubs_s[j, k]
                    bsupumn_a += tsini * bsupu_s[j, k]
                    bsupvmn_a += tsini * bsupv_s[j, k]

            gmnc[js, mn] = gmn
            bmnc[js, mn] = bmn
            bsubumnc[js, mn] = bsubumn
            bsubvmnc[js, mn] = bsubvmn
            bsubsmns[js, mn] = bsubsmn
            bsupumnc[js, mn] = bsupumn
            bsupvmnc[js, mn] = bsupvmn
            gmns[js, mn] = gmn_a
            bmns[js, mn] = bmn_a
            bsubumns[js, mn] = bsubumn_a
            bsubvmns[js, mn] = bsubvmn_a
            bsubsmnc[js, mn] = bsubsmn_a
            bsupumns[js, mn] = bsupumn_a
            bsupvmns[js, mn] = bsupvmn_a

    if ns > 0:
        gmnc[0, :] = 0.0
        bmnc[0, :] = 0.0
        bsubumnc[0, :] = 0.0
        bsubvmnc[0, :] = 0.0
        bsupumnc[0, :] = 0.0
        bsupvmnc[0, :] = 0.0
        gmns[0, :] = 0.0
        bmns[0, :] = 0.0
        bsubumns[0, :] = 0.0
        bsubvmns[0, :] = 0.0
        bsupumns[0, :] = 0.0
        bsupvmns[0, :] = 0.0
    if ns > 2:
        bsubsmns[0, :] = 2.0 * bsubsmns[1, :] - bsubsmns[2, :]
        bsubsmnc[0, :] = 2.0 * bsubsmnc[1, :] - bsubsmnc[2, :]

    return dict(
        gmnc=gmnc,
        bmnc=bmnc,
        bsubumnc=bsubumnc,
        bsubvmnc=bsubvmnc,
        bsubsmns=bsubsmns,
        bsupumnc=bsupumnc,
        bsupvmnc=bsupvmnc,
        gmns=gmns,
        bmns=bmns,
        bsubumns=bsubumns,
        bsubvmns=bsubvmns,
        bsubsmnc=bsubsmnc,
        bsupumns=bsupumns,
        bsupvmns=bsupvmns,
    )


def vmec_wrout_lasym_bsubuv_output_scale(
    *,
    bsubumnc: np.ndarray,
    bsubvmnc: np.ndarray,
    bsubumns: np.ndarray,
    bsubvmns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply VMEC's LASYM ``wrout`` scaling for covariant bsubu/bsubv coefficients."""
    return (
        2.0 * np.asarray(bsubumnc, dtype=float),
        2.0 * np.asarray(bsubvmnc, dtype=float),
        2.0 * np.asarray(bsubumns, dtype=float),
        2.0 * np.asarray(bsubvmns, dtype=float),
    )


def vmec_wrout_nyquist_synthesis(
    *,
    coeff_c: np.ndarray,
    coeff_s: np.ndarray,
    modes: ModeTable,
    trig,
) -> np.ndarray:
    """Synthesize real-space field from ``wrout``-style Nyquist coefficients."""
    coeff_c = np.asarray(coeff_c, dtype=float)
    coeff_s = np.asarray(coeff_s, dtype=float)
    if coeff_c.ndim != 2 or coeff_s.ndim != 2:
        raise ValueError("Expected coeff arrays with shape (ns, K)")
    m = np.asarray(modes.m, dtype=int)
    n = np.asarray(modes.n, dtype=int)
    if m.size == 0:
        return np.zeros((coeff_c.shape[0], 0, 0), dtype=float)

    nt2 = int(trig.ntheta2)
    cosmu = np.asarray(trig.cosmu, dtype=float)[:nt2, :]
    sinmu = np.asarray(trig.sinmu, dtype=float)[:nt2, :]
    cosnv = np.asarray(trig.cosnv, dtype=float)
    sinnv = np.asarray(trig.sinnv, dtype=float)

    mmax = int(np.max(m))
    nmax = int(np.max(np.abs(n)))
    if cosmu.shape[1] <= mmax or cosnv.shape[1] <= nmax:
        raise ValueError("Trig tables do not cover Nyquist mode limits")

    n_abs = np.abs(n)
    sgn = np.where(n < 0, -1.0, 1.0)

    mscale = np.asarray(trig.mscale, dtype=float)
    nscale = np.asarray(trig.nscale, dtype=float)
    tmult = 0.5 / float(getattr(trig, "r0scale", 1.0)) ** 2
    dmult = mscale[m] * nscale[n_abs] * tmult
    dmult = np.where((m == 0) | (n == 0), 2.0 * dmult, dmult)
    dmult = np.where(dmult == 0.0, 1.0, dmult)

    raw_c = coeff_c / dmult[None, :]
    raw_s = coeff_s / dmult[None, :]

    cosmu_m = cosmu[:, m]
    sinmu_m = sinmu[:, m]
    cosnv_n = cosnv[:, n_abs]
    sinnv_n = sinnv[:, n_abs] * sgn[None, :]

    term_c = cosmu_m[:, None, :] * cosnv_n[None, :, :] + sinmu_m[:, None, :] * sinnv_n[None, :, :]
    term_s = sinmu_m[:, None, :] * cosnv_n[None, :, :] - cosmu_m[:, None, :] * sinnv_n[None, :, :]

    f = np.einsum("sk,ijk->sij", raw_c, term_c, optimize=True) + np.einsum("sk,ijk->sij", raw_s, term_s, optimize=True)
    return np.asarray(f, dtype=float)


def vmec_wrout_nyquist_sin_coeffs_loop(
    *,
    f: np.ndarray,
    modes: ModeTable,
    trig,
) -> np.ndarray:
    """Loop-based ``wrout`` Nyquist sine coefficients with VMEC summation order."""
    f = np.asarray(f, dtype=float)
    if f.ndim != 3:
        raise ValueError("Expected f with shape (ns, ntheta, nzeta)")
    m_arr = np.asarray(modes.m, dtype=int)
    n_arr = np.asarray(modes.n, dtype=int)
    ns = int(f.shape[0])
    if m_arr.size == 0:
        return np.zeros((ns, 0), dtype=float)

    nt2 = int(trig.ntheta2)
    if int(f.shape[1]) < nt2:
        raise ValueError("Input theta grid is smaller than VMEC ntheta2")
    f = f[:, :nt2, :]

    cosmui = np.asarray(trig.cosmui, dtype=float)[:nt2, :]
    sinmui = np.asarray(trig.sinmui, dtype=float)[:nt2, :]
    cosnv = np.asarray(trig.cosnv, dtype=float)
    sinnv = np.asarray(trig.sinnv, dtype=float)

    mmax = int(np.max(m_arr))
    nmax = int(np.max(np.abs(n_arr)))
    if cosmui.shape[1] <= mmax or cosnv.shape[1] <= nmax:
        raise ValueError("Trig tables do not cover Nyquist mode limits")

    mnyq = cosmui.shape[1] - 1
    if mnyq > 0:
        cosmui = cosmui.copy()
        cosmui[:, mnyq] *= 0.5
    nnyq = cosnv.shape[1] - 1
    if nnyq > 0:
        cosnv = cosnv.copy()
        cosnv[:, nnyq] *= 0.5

    mscale = np.asarray(trig.mscale, dtype=float)
    nscale = np.asarray(trig.nscale, dtype=float)
    tmult = 0.5 / float(getattr(trig, "r0scale", 1.0)) ** 2

    coeff = np.zeros((ns, m_arr.size), dtype=float)
    nzeta = int(f.shape[2])
    for js in range(ns):
        f_js = f[js]
        for idx, (m, n) in enumerate(zip(m_arr, n_arr)):
            n1 = abs(int(n))
            sgn = -1.0 if n < 0 else 1.0
            dmult = mscale[m] * nscale[n1] * tmult
            if m == 0 or n == 0:
                dmult *= 2.0
            acc = 0.0
            for k in range(nzeta):
                for j in range(nt2):
                    tsini = dmult * (sinmui[j, m] * cosnv[k, n1] - sgn * cosmui[j, m] * sinnv[k, n1])
                    acc += tsini * f_js[j, k]
            coeff[js, idx] = acc

    return coeff


def vmec_jxbforce_cos_coeffs(
    *,
    f: np.ndarray,
    modes: ModeTable,
    trig,
) -> np.ndarray:
    """VMEC ``jxbforce``-style cosine coefficients for the low-pass filter."""
    f = np.asarray(f, dtype=float)
    if f.ndim != 3:
        raise ValueError("Expected f with shape (ns, ntheta, nzeta)")
    m = np.asarray(modes.m, dtype=int)
    n = np.asarray(modes.n, dtype=int)
    if m.size == 0:
        return np.zeros((f.shape[0], 0), dtype=float)

    nt2 = int(trig.ntheta2)
    if int(f.shape[1]) < nt2:
        raise ValueError("Input theta grid is smaller than VMEC ntheta2")
    f = f[:, :nt2, :]

    cosmui = np.asarray(trig.cosmui, dtype=float)[:nt2, :]
    sinmui = np.asarray(trig.sinmui, dtype=float)[:nt2, :]
    cosnv = np.asarray(trig.cosnv, dtype=float)
    sinnv = np.asarray(trig.sinnv, dtype=float)

    mmax = int(np.max(m))
    nmax = int(np.max(np.abs(n)))
    if cosmui.shape[1] <= mmax or cosnv.shape[1] <= nmax:
        raise ValueError("Trig tables do not cover Nyquist mode limits")

    f_theta_cos = np.einsum("sik,im->smk", f, cosmui, optimize=True)
    f_theta_sin = np.einsum("sik,im->smk", f, sinmui, optimize=True)
    cos_zeta = np.einsum("smk,kn->smn", f_theta_cos, cosnv, optimize=True)
    sin_zeta = np.einsum("smk,kn->smn", f_theta_sin, sinnv, optimize=True)

    n_abs = np.abs(n)
    sgn = np.where(n < 0, -1.0, 1.0)
    cos_part = cos_zeta[:, m, n_abs]
    sin_part = sin_zeta[:, m, n_abs]
    coeff = cos_part + sgn[None, :] * sin_part

    r0scale = float(getattr(trig, "r0scale", 1.0))
    dmult = np.full_like(m, 1.0 / (r0scale**2), dtype=float)
    mnyq = cosmui.shape[1] - 1
    nnyq = cosnv.shape[1] - 1
    if mnyq > 0:
        dmult = np.where(m == mnyq, 0.5 * dmult, dmult)
    if nnyq > 0:
        dmult = np.where((n_abs == nnyq) & (n_abs != 0), 0.5 * dmult, dmult)
    coeff = coeff * dmult[None, :]
    return np.asarray(coeff, dtype=float)


def vmec_jxbforce_sin_coeffs(
    *,
    f: np.ndarray,
    modes: ModeTable,
    trig,
) -> np.ndarray:
    """VMEC ``jxbforce``-style sine coefficients for the low-pass filter."""
    f = np.asarray(f, dtype=float)
    if f.ndim != 3:
        raise ValueError("Expected f with shape (ns, ntheta, nzeta)")
    m = np.asarray(modes.m, dtype=int)
    n = np.asarray(modes.n, dtype=int)
    if m.size == 0:
        return np.zeros((f.shape[0], 0), dtype=float)

    nt2 = int(trig.ntheta2)
    if int(f.shape[1]) < nt2:
        raise ValueError("Input theta grid is smaller than VMEC ntheta2")
    f = f[:, :nt2, :]

    cosmui = np.asarray(trig.cosmui, dtype=float)[:nt2, :]
    sinmui = np.asarray(trig.sinmui, dtype=float)[:nt2, :]
    cosnv = np.asarray(trig.cosnv, dtype=float)
    sinnv = np.asarray(trig.sinnv, dtype=float)

    mmax = int(np.max(m))
    nmax = int(np.max(np.abs(n)))
    if cosmui.shape[1] <= mmax or cosnv.shape[1] <= nmax:
        raise ValueError("Trig tables do not cover Nyquist mode limits")

    f_theta_sin = np.einsum("sik,im->smk", f, sinmui, optimize=True)
    f_theta_cos = np.einsum("sik,im->smk", f, cosmui, optimize=True)
    cos_zeta = np.einsum("smk,kn->smn", f_theta_sin, cosnv, optimize=True)
    sin_zeta = np.einsum("smk,kn->smn", f_theta_cos, sinnv, optimize=True)

    n_abs = np.abs(n)
    sgn = np.where(n < 0, -1.0, 1.0)
    cos_part = cos_zeta[:, m, n_abs]
    sin_part = sin_zeta[:, m, n_abs]
    coeff = cos_part - sgn[None, :] * sin_part

    r0scale = float(getattr(trig, "r0scale", 1.0))
    dmult = np.full_like(m, 1.0 / (r0scale**2), dtype=float)
    mnyq = cosmui.shape[1] - 1
    nnyq = cosnv.shape[1] - 1
    if mnyq > 0:
        dmult = np.where(m == mnyq, 0.5 * dmult, dmult)
    if nnyq > 0:
        dmult = np.where((n_abs == nnyq) & (n_abs != 0), 0.5 * dmult, dmult)
    coeff = coeff * dmult[None, :]
    return np.asarray(coeff, dtype=float)
