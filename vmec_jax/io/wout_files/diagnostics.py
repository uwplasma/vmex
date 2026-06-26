"""WOUT diagnostic reconstruction helpers.

These helpers operate on persisted VMEC ``wout`` profile arrays.  They are kept
separate from the large reader/writer module so stability-diagnostic algebra can
be tested and reused without importing the full WOUT synthesis path.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

_MU0 = 4e-7 * np.pi


def pshalf_from_s(s_full: np.ndarray) -> np.ndarray:
    """Return VMEC half-mesh ``sqrt(s)`` values used in parity formulas."""

    s_arr = np.asarray(s_full, dtype=float)
    if s_arr.shape[0] < 2:
        return np.sqrt(np.maximum(s_arr, 0.0))
    sh = 0.5 * (s_arr[1:] + s_arr[:-1])
    p = np.concatenate([sh[:1], sh], axis=0)
    return np.sqrt(np.maximum(p, 0.0))


def lambda_half_mesh_weights(s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return VMEC Fortran-style ``sm``/``sp`` weights for lambda half-mesh maps."""

    s_arr = np.asarray(s, dtype=float).reshape(-1)
    ns = int(s_arr.shape[0])
    if ns < 2:
        return np.zeros((ns + 1,), dtype=float), np.zeros((ns + 1,), dtype=float)

    hs = float(s_arr[1] - s_arr[0])
    sqrts_f = np.zeros((ns + 1,), dtype=float)
    shalf_f = np.zeros((ns + 1,), dtype=float)
    for i in range(1, ns + 1):
        sqrts_f[i] = np.sqrt(max(hs * float(i - 1), 0.0))
        shalf_f[i] = np.sqrt(hs * abs(float(i) - 1.5))
    sqrts_f[ns] = 1.0

    sm_f = np.zeros((ns + 1,), dtype=float)
    sp_f = np.zeros((ns + 1,), dtype=float)
    for i in range(2, ns + 1):
        sm_f[i] = shalf_f[i] / sqrts_f[i] if sqrts_f[i] != 0.0 else 0.0
        if i < ns:
            sp_f[i] = shalf_f[i + 1] / sqrts_f[i] if sqrts_f[i] != 0.0 else 0.0
        else:
            sp_f[i] = 1.0 / sqrts_f[i] if sqrts_f[i] != 0.0 else 0.0
    sm_f[1] = 0.0
    sp_f[0] = 0.0
    sp_f[1] = sm_f[2] if ns >= 2 else 0.0
    return sm_f, sp_f


def safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Divide with VMEC's zero-denominator convention for diagnostic scalars."""

    den_safe = np.where(np.abs(den) > 0.0, den, 1.0)
    return num / den_safe


def compute_eqfor_beta(
    *,
    pres: np.ndarray,
    vp: np.ndarray,
    bsq: np.ndarray,
    r12: np.ndarray,
    bsupv: np.ndarray,
    sqrtg: np.ndarray,
    wint: np.ndarray,
    signgs: int,
) -> tuple[float, float, float, float]:
    """Compute VMEC ``eqfor`` betapol/betator/betatot/betaxis diagnostics."""

    ns = int(pres.shape[0])
    if ns < 3:
        return 0.0, 0.0, 0.0, 0.0
    hs = 1.0 / float(ns - 1)
    vnorm = (2.0 * np.pi) ** 2 * hs
    tau = float(signgs) * wint * np.asarray(sqrtg, dtype=float)
    tau = np.asarray(tau, dtype=float)
    tau[0] = 0.0

    sump = vnorm * float(np.sum(np.asarray(vp[1:], dtype=float) * np.asarray(pres[1:], dtype=float)))
    bsq = np.asarray(bsq, dtype=float)
    r12 = np.asarray(r12, dtype=float)
    bsupv = np.asarray(bsupv, dtype=float)
    sum_bsq_tau = float(np.sum(bsq[1:] * tau[1:]))
    sumbtot = 2.0 * (vnorm * sum_bsq_tau - sump)
    sumbtor = vnorm * float(np.sum(tau[1:] * (r12[1:] * bsupv[1:]) ** 2))
    sumbpol = sumbtot - sumbtor

    betapol = float(safe_divide(2.0 * sump, sumbpol))
    betator = float(safe_divide(2.0 * sump, sumbtor))
    betatot = float(safe_divide(2.0 * sump, sumbtot))

    beta_vol = np.zeros((ns,), dtype=float)
    for i in range(1, ns):
        s2 = float(np.sum(bsq[i] * tau[i])) / float(vp[i]) - float(pres[i])
        beta_vol[i] = float(safe_divide(float(pres[i]), s2))
    betaxis = float(1.5 * beta_vol[1] - 0.5 * beta_vol[2])
    return betapol, betator, betatot, betaxis


def compute_eqfor_betaxis(
    *,
    pres: np.ndarray,
    vp: np.ndarray,
    bsq: np.ndarray,
    sqrtg: np.ndarray,
    wint: np.ndarray,
    signgs: int,
) -> float:
    """Compute VMEC ``eqfor`` betaxis independently of convergence status."""

    ns = int(pres.shape[0])
    if ns < 3:
        return 0.0
    tau = float(signgs) * np.asarray(wint, dtype=float) * np.asarray(sqrtg, dtype=float)
    tau[0] = 0.0
    beta_vol = np.zeros((ns,), dtype=float)
    for i in range(1, ns):
        denom = float(np.sum(bsq[i] * tau[i])) / float(vp[i]) - float(pres[i])
        if denom != 0.0:
            beta_vol[i] = float(pres[i]) / denom
    return float(1.5 * beta_vol[1] - 0.5 * beta_vol[2])


def _vmec_wint_from_trig(trig) -> np.ndarray:
    """Return VMEC-style angular weights on the internal grid."""

    cosmui3 = np.asarray(trig.cosmui3)
    mscale = np.asarray(trig.mscale)
    if cosmui3.ndim != 2:
        raise ValueError("Expected trig.cosmui3 with shape (ntheta3, mmax+1)")
    if mscale.size == 0:
        raise ValueError("Expected non-empty trig.mscale")
    w_theta = cosmui3[:, 0] / float(mscale[0])
    nzeta = int(np.asarray(trig.cosnv).shape[0])
    wint = w_theta[:, None] * np.ones((nzeta,), dtype=w_theta.dtype)[None, :]
    return np.asarray(wint, dtype=float)


def compute_equif_wout(
    *,
    bsubu: np.ndarray,
    bsubv: np.ndarray,
    pres: np.ndarray,
    vp: np.ndarray,
    phipf: np.ndarray,
    chipf: np.ndarray,
    signgs: int,
    trig,
    s: np.ndarray,
    mu0: float = _MU0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute buco/bvco/jcuru/jcurv/equif with VMEC eqfor normalization."""

    s = np.asarray(s, dtype=float)
    ns = int(s.shape[0])
    if ns < 3:
        z = np.zeros((ns,), dtype=float)
        return z.copy(), z.copy(), z.copy(), z.copy(), z.copy()

    hs = float(s[1] - s[0])
    ohs = 1.0 / hs if hs != 0.0 else 0.0
    wint = _vmec_wint_from_trig(trig)
    bsubu = np.asarray(bsubu, dtype=float)
    bsubv = np.asarray(bsubv, dtype=float)
    pres = np.asarray(pres, dtype=float)
    vp = np.asarray(vp, dtype=float)
    phipf = np.asarray(phipf, dtype=float)
    chipf = np.asarray(chipf, dtype=float)

    buco = np.zeros((ns,), dtype=float)
    bvco = np.zeros((ns,), dtype=float)
    ntheta = int(wint.shape[0])
    nzeta = int(wint.shape[1])
    # Match VMEC's summation order (theta, then zeta) to reduce parity drift.
    for js in range(1, ns):
        acc_u = 0.0
        acc_v = 0.0
        for j in range(ntheta):
            wrow = wint[j]
            bu_row = bsubu[js, j]
            bv_row = bsubv[js, j]
            for k in range(nzeta):
                w = float(wrow[k])
                acc_u += float(bu_row[k]) * w
                acc_v += float(bv_row[k]) * w
        buco[js] = acc_u
        bvco[js] = acc_v

    jcuru = np.zeros((ns,), dtype=float)
    jcurv = np.zeros((ns,), dtype=float)
    vpphi = np.zeros((ns,), dtype=float)
    presgrad = np.zeros((ns,), dtype=float)
    for js in range(1, ns - 1):
        jcurv[js] = float(signgs) * ohs * (buco[js + 1] - buco[js])
        jcuru[js] = -float(signgs) * ohs * (bvco[js + 1] - bvco[js])
        vpphi[js] = 0.5 * (vp[js + 1] + vp[js])
        presgrad[js] = (pres[js + 1] - pres[js]) * ohs

    equif = np.zeros((ns,), dtype=float)
    for js in range(1, ns - 1):
        denom = abs(jcurv[js] * chipf[js]) + abs(jcuru[js] * phipf[js]) + abs(presgrad[js] * vpphi[js])
        if denom != 0.0 and vpphi[js] != 0.0:
            raw = ((-phipf[js] * jcuru[js] + chipf[js] * jcurv[js]) / vpphi[js]) + presgrad[js]
            equif[js] = raw * vpphi[js] / denom

    # Extrapolate endpoints (eqfor.f).
    for arr in (equif, jcuru, jcurv, presgrad, vpphi):
        arr[0] = 2.0 * arr[1] - arr[2]
        arr[-1] = 2.0 * arr[-2] - arr[-3]

    return buco, bvco, jcuru / float(mu0), jcurv / float(mu0), equif


def apply_bsubv_equif_correction(
    *,
    bsubv: np.ndarray,
    bsubv_e: np.ndarray,
    trig,
    vmec_pwint_from_trig_func=None,
) -> np.ndarray:
    """Apply VMEC ``bcovar`` IEQUI=1 surface-average correction to ``bsubv``."""

    if vmec_pwint_from_trig_func is None:
        from ...kernels.residue import vmec_pwint_from_trig as vmec_pwint_from_trig_func

    bsubv = np.asarray(bsubv, dtype=float)
    bsubv_e = np.asarray(bsubv_e, dtype=float)
    ns = int(bsubv.shape[0])
    if ns < 3:
        return bsubv

    nzeta = int(bsubv.shape[2])
    pwint = np.asarray(vmec_pwint_from_trig_func(trig, ns=ns, nzeta=nzeta), dtype=float)
    if pwint.shape != bsubv.shape:
        raise ValueError("pwint shape mismatch in bsubv correction")

    fpsi = np.zeros((ns,), dtype=float)
    for js in range(1, ns):
        fpsi[js] = float(np.sum(bsubv[js] * pwint[js]))

    bsubv_h = np.array(bsubv, dtype=float, copy=True)
    for js in range(ns - 2, 0, -1):
        bsubv_h[js] = 2.0 * bsubv_e[js] - bsubv_h[js + 1]

    for js in range(1, ns):
        curpol = fpsi[js] - float(np.sum(bsubv_h[js] * pwint[js]))
        bsubv_h[js] = bsubv_h[js] + curpol

    return bsubv_h


def compute_aspectratio(
    *,
    R: np.ndarray,
    Zu: np.ndarray,
    wint: np.ndarray,
) -> tuple[float, float, float, float, float]:
    """Compute VMEC aspect-ratio geometry scalars from edge R, Zu arrays."""

    if R.ndim != 3 or Zu.ndim != 3:
        raise ValueError("Expected R/Zu with shape (ns, ntheta, nzeta)")
    rb = np.asarray(R[-1], dtype=float)
    zub = np.asarray(Zu[-1], dtype=float)
    wint = np.asarray(wint, dtype=float)
    if wint.shape != rb.shape:
        raise ValueError("wint shape mismatch for aspectratio")
    t1 = rb * zub * wint
    volume_p = float(2.0 * np.pi * np.pi * abs(np.sum(rb * t1)))
    cross_area_p = float(2.0 * np.pi * abs(np.sum(t1)))
    if cross_area_p == 0.0:
        return 0.0, 0.0, 0.0, volume_p, cross_area_p
    Rmajor_p = float(volume_p / (2.0 * np.pi * cross_area_p))
    Aminor_p = float(np.sqrt(cross_area_p / np.pi))
    aspect = float(Rmajor_p / Aminor_p) if Aminor_p != 0.0 else 0.0
    return Aminor_p, Rmajor_p, aspect, volume_p, cross_area_p


def compute_ctor_from_buco(*, buco: np.ndarray, signgs: int, indata, mu0: float = _MU0) -> float:
    """Compute VMEC ``ctor`` from half-mesh ``buco`` using wrout conventions."""

    ns = int(buco.shape[0])
    if ns < 2:
        return 0.0
    lfreeb = bool(indata.get_bool("LFREEB", False))
    ictrl_prec2d = int(indata.get_int("ICTRL_PREC2D", 0))
    lhess_exact = bool(indata.get_bool("LHESS_EXACT", False))
    if lhess_exact:
        lctor = lfreeb and (ictrl_prec2d != 0)
    else:
        lctor = lfreeb and (ictrl_prec2d > 1)
    if lctor:
        ctor_prec2d = 0.0
        ctor = float(signgs) * (2.0 * np.pi) * (float(buco[-1]) + ctor_prec2d)
    else:
        ctor = float(signgs) * (2.0 * np.pi) * (1.5 * float(buco[-1]) - 0.5 * float(buco[-2]))
    return float(ctor / float(mu0))


def glasser_from_wout_mercier_terms(
    *,
    DMerc: np.ndarray,
    Dshear: np.ndarray,
    Dcurr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return Glasser profiles from persisted VMEC Mercier components.

    Wout files do not store the full Mercier surface integrals needed to
    reconstruct the preferred state-level ``H`` expression.  For persistence
    and old-file fallback we use the equivalent current-term reconstruction
    ``H = -Dcurr`` and ``S^2 = 4*Dshear``.
    """

    dmerc = np.asarray(DMerc, dtype=float)
    dshear = np.asarray(Dshear, dtype=float)
    dcurr = np.asarray(Dcurr, dtype=float)
    shear2 = np.maximum(4.0 * dshear, 0.0)
    h_term = -dcurr
    valid = shear2 > 0.0
    denom = np.where(valid, shear2, 1.0)
    correction = np.where(valid, (h_term - 0.5 * shear2) ** 2 / denom, 0.0)
    d_r = np.where(valid, -dmerc + correction, 0.0)
    return (
        np.asarray(d_r, dtype=float),
        np.asarray(h_term, dtype=float),
        np.asarray(correction, dtype=float),
        np.asarray(valid, dtype=bool),
    )


class GlasserProfileArrays(NamedTuple):
    """Persisted or fallback Glasser profiles read from a WOUT variable map."""

    D_R: np.ndarray
    H: np.ndarray
    correction: np.ndarray
    shear_valid: np.ndarray


def _read_profile_variable(variables: dict[str, Any], name: str, fallback: np.ndarray, *, dtype=float) -> np.ndarray:
    value = variables.get(name)
    if value is None:
        return np.asarray(fallback, dtype=dtype)
    return np.asarray(value[:], dtype=dtype)


def glasser_profiles_from_wout_variables(
    variables: dict[str, Any],
    *,
    DMerc: np.ndarray,
    Dshear: np.ndarray,
    Dcurr: np.ndarray,
) -> GlasserProfileArrays:
    """Read Glasser profiles from WOUT variables, falling back to Mercier terms.

    New vmec_jax WOUT files persist ``D_R``, ``HGlasser``,
    ``GlasserCorrection`` and ``GlasserShearValid``.  Older VMEC/VMEC++
    files do not, so this helper reconstructs the fallback from the persisted
    Mercier components and then lets any explicit variables override it.
    """

    fallback_D_R, fallback_H, fallback_correction, fallback_valid = glasser_from_wout_mercier_terms(
        DMerc=DMerc,
        Dshear=Dshear,
        Dcurr=Dcurr,
    )
    h_variable = variables.get("HGlasser", variables.get("H"))
    h_profile = np.asarray(fallback_H if h_variable is None else h_variable[:], dtype=float)
    return GlasserProfileArrays(
        D_R=_read_profile_variable(variables, "D_R", fallback_D_R),
        H=h_profile,
        correction=_read_profile_variable(variables, "GlasserCorrection", fallback_correction),
        shear_valid=_read_profile_variable(variables, "GlasserShearValid", fallback_valid, dtype=bool),
    )


def glasser_profiles_from_wout_data(wout: Any, ns: int) -> GlasserProfileArrays:
    """Return Glasser profiles ready for WOUT persistence.

    ``Wout`` objects produced by older code paths may not yet carry the newer
    Glasser arrays.  Centralizing the defaults here keeps the writer focused on
    NetCDF materialization and ensures missing profiles are persisted with
    predictable zero-valued arrays of the correct length.
    """

    shape = (int(ns),)
    return GlasserProfileArrays(
        D_R=np.asarray(getattr(wout, "D_R", np.zeros(shape, dtype=float)), dtype=float),
        H=np.asarray(getattr(wout, "H", np.zeros(shape, dtype=float)), dtype=float),
        correction=np.asarray(getattr(wout, "glasser_correction", np.zeros(shape, dtype=float)), dtype=float),
        shear_valid=np.asarray(getattr(wout, "glasser_shear_valid", np.zeros(shape, dtype=bool)), dtype=bool),
    )


__all__ = [
    "GlasserProfileArrays",
    "apply_bsubv_equif_correction",
    "compute_aspectratio",
    "compute_ctor_from_buco",
    "compute_eqfor_beta",
    "compute_eqfor_betaxis",
    "compute_equif_wout",
    "glasser_profiles_from_wout_data",
    "glasser_profiles_from_wout_variables",
    "glasser_from_wout_mercier_terms",
    "lambda_half_mesh_weights",
    "pshalf_from_s",
    "safe_divide",
]
