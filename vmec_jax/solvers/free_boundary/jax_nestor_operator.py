"""Opt-in JAX NESTOR operator cache used by free-boundary validation paths."""

from __future__ import annotations

import hashlib
import os
from typing import Any

import numpy as np

from .types import ExternalBoundarySample, NestorPoissonCache, NestorVmecLikeCache

try:  # pragma: no cover - optional dependency
    from scipy.linalg import lu_factor as _SCIPY_LU_FACTOR  # type: ignore
    from scipy.linalg import lu_solve as _SCIPY_LU_SOLVE  # type: ignore
except Exception:  # pragma: no cover - SciPy is optional at runtime
    _SCIPY_LU_FACTOR = None
    _SCIPY_LU_SOLVE = None


JAX_NESTOR_BASIS_KEYS = (
    "lasym",
    "mf",
    "nf",
    "mn0",
    "mnpd",
    "mnpd2",
    "nu_full",
    "nuv3",
    "nuv_full",
    "onp",
    "cmns",
    "cos_phase",
    "cosmni",
    "imirr",
    "imirr_full",
    "n_raw",
    "sin_phase",
    "sinmni",
    "theta",
    "wint",
    "xmpot",
    "zeta",
)

FREEB_JAX_NESTOR_OPERATOR_FN_CACHE: dict[tuple[Any, ...], Any] = {}


def dense_lu_factor(matrix: np.ndarray) -> Any | None:
    """Evaluate dense lu factor for direct-coil free-boundary solve and branch-local adjoint validation."""
    if _SCIPY_LU_FACTOR is None:
        return None
    try:
        return _SCIPY_LU_FACTOR(np.asarray(matrix, dtype=float))
    except Exception:
        return None


def dense_lu_solve(lu_fac: Any | None, matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Evaluate dense lu solve for direct-coil free-boundary solve and branch-local adjoint validation."""
    rhs_arr = np.asarray(rhs, dtype=float)
    if lu_fac is not None and _SCIPY_LU_SOLVE is not None:
        try:
            return np.asarray(_SCIPY_LU_SOLVE(lu_fac, rhs_arr), dtype=float)
        except Exception:
            pass
    return np.asarray(np.linalg.solve(np.asarray(matrix, dtype=float), rhs_arr), dtype=float)


def build_vmec_cmns(*, mf: int, nf: int, onp: float) -> np.ndarray:
    """VMEC precal.f cmns(l,m,n) coefficients for the n>=0 block."""

    mf = max(0, int(mf))
    nf = max(0, int(nf))
    lmax = mf + nf
    cmn = np.zeros((lmax + 1, mf + 1, nf + 1), dtype=float)
    for m in range(mf + 1):
        for n in range(nf + 1):
            jmn = m + n
            imn = m - n
            kmn = abs(imn)
            smn = (jmn + kmn) // 2
            f1 = 1.0
            f2 = 1.0
            f3 = 1.0
            for i in range(1, kmn + 1):
                f1 *= float(smn + 1 - i)
                f2 *= float(i)
            for l in range(kmn, jmn + 1, 2):
                cmn[l, m, n] = (f1 / (f2 * f3)) * ((-1.0) ** ((l - imn) // 2))
                f1 = f1 * 0.25 * float((jmn + l + 2) * (jmn - l))
                f2 = f2 * 0.5 * float(l + 2 + kmn)
                f3 = f3 * 0.5 * float(l + 2 - kmn)

    alp = 2.0 * np.pi * float(onp)
    cmns = np.zeros_like(cmn)
    if mf >= 1 and nf >= 1:
        cmns[:, 1 : mf + 1, 1 : nf + 1] = (
            0.5
            * alp
            * (
                cmn[:, 1 : mf + 1, 1 : nf + 1]
                + cmn[:, :mf, 1 : nf + 1]
                + cmn[:, 1 : mf + 1, :nf]
                + cmn[:, :mf, :nf]
            )
        )
    if mf >= 1:
        cmns[:, 1 : mf + 1, 0] = 0.5 * alp * (cmn[:, 1 : mf + 1, 0] + cmn[:, :mf, 0])
    if nf >= 1:
        cmns[:, 0, 1 : nf + 1] = 0.5 * alp * (cmn[:, 0, 1 : nf + 1] + cmn[:, 0, :nf])
    cmns[:, 0, 0] = 0.5 * alp * (cmn[:, 0, 0] + cmn[:, 0, 0])
    return cmns


def build_vmec_mode_basis(
    *,
    ntheta: int,
    nzeta: int,
    nfp: int,
    mf: int,
    nf: int,
    lasym: bool,
    wint: np.ndarray,
) -> dict[str, Any]:
    """Build VMEC-like mode tables and weighted sin/cos basis arrays."""

    ntheta = int(ntheta)
    nzeta = int(nzeta)
    nfp = max(1, int(nfp))
    mf = max(0, int(mf))
    nf = max(0, int(nf))
    lasym = bool(lasym)

    pi2 = 2.0 * np.pi
    if lasym:
        nu_full = int(ntheta)
    else:
        nu_full = max(int(ntheta), 2 * (int(ntheta) - 1))
    theta = (pi2 / float(max(1, nu_full))) * np.arange(ntheta, dtype=float)
    zeta = (pi2 / float(max(1, nzeta))) * np.arange(nzeta, dtype=float)
    th_grid = np.broadcast_to(theta[:, None], (ntheta, nzeta))
    ze_grid = np.broadcast_to(zeta[None, :], (ntheta, nzeta))
    th = th_grid.reshape(-1)
    ze = ze_grid.reshape(-1)

    w = np.asarray(wint, dtype=float).reshape(-1)
    if w.size != th.size:
        w = np.full((th.size,), 1.0 / float(max(1, th.size)), dtype=float)

    mvals: list[int] = []
    nvals: list[int] = []
    for n in range(-nf, nf + 1):
        for m in range(0, mf + 1):
            mvals.append(int(m))
            nvals.append(int(n))
    xmpot = np.asarray(mvals, dtype=np.int64)
    n_raw = np.asarray(nvals, dtype=np.int64)
    xnpot = np.asarray(n_raw * nfp, dtype=np.int64)
    mnpd = int(xmpot.size)
    mnpd2 = int(mnpd * (2 if lasym else 1))

    phase = (xmpot[None, :] * th[:, None]) - (n_raw[None, :] * ze[:, None])
    sin_phase = np.sin(phase)
    cos_phase = np.cos(phase)
    weight = ((pi2 * pi2) * w)[:, None]
    sinmni = weight * sin_phase
    cosmni = weight * cos_phase

    idx = np.arange(th.size, dtype=np.int64)
    lt = idx // max(1, nzeta)
    lz = idx % max(1, nzeta)
    if lasym or (nu_full == ntheta):
        lt_m = (ntheta - lt) % max(1, ntheta)
    else:
        lt_m_full = (nu_full - lt) % max(1, nu_full)
        lt_m = np.minimum(lt_m_full, (nu_full - lt_m_full) % max(1, nu_full))
    lz_m = (nzeta - lz) % max(1, nzeta)
    imirr = (lt_m * nzeta + lz_m).astype(np.int64)
    nuv_full = int(max(1, nu_full) * max(1, nzeta))
    idx_full = np.arange(nuv_full, dtype=np.int64)
    ku_full = idx_full // max(1, nzeta)
    kv_full = idx_full % max(1, nzeta)
    ku_m_full = (nu_full - ku_full) % max(1, nu_full)
    kv_m_full = (nzeta - kv_full) % max(1, nzeta)
    imirr_full = (ku_m_full * nzeta + kv_m_full).astype(np.int64)

    mn0 = 0
    for j in range(mnpd):
        if int(xmpot[j]) == 0 and int(n_raw[j]) == 0:
            mn0 = int(j)
            break

    return {
        "xmpot": xmpot,
        "xnpot": xnpot,
        "n_raw": n_raw,
        "sin_phase": sin_phase,
        "cos_phase": cos_phase,
        "sinmni": sinmni,
        "cosmni": cosmni,
        "wint": w,
        "imirr": imirr,
        "imirr_full": imirr_full,
        "mnpd": mnpd,
        "mnpd2": mnpd2,
        "nuv3": int(th.size),
        "nuv_full": nuv_full,
        "mn0": mn0,
        "onp": 1.0 / float(nfp),
        "nfp": nfp,
        "mf": mf,
        "nf": nf,
        "nu_full": int(nu_full),
        "lasym": lasym,
        "theta": th,
        "zeta": ze,
        "cmns": build_vmec_cmns(mf=mf, nf=nf, onp=1.0 / float(nfp)),
    }


def build_poisson_cache(*, ntheta: int, nzeta: int) -> NestorPoissonCache:
    """Build spectral Laplacian eigenvalues on a periodic ``(theta,zeta)`` grid."""

    ntheta = int(ntheta)
    nzeta = int(nzeta)
    ku = 2.0 * np.pi * np.fft.fftfreq(ntheta)
    kv = 2.0 * np.pi * np.fft.fftfreq(nzeta)
    ku2 = ku[:, None] * ku[:, None]
    kv2 = kv[None, :] * kv[None, :]
    lam = ku2 + kv2
    lam[0, 0] = 1.0
    return NestorPoissonCache(ntheta=ntheta, nzeta=nzeta, lam=lam)


def build_vmec_like_cache(
    sample: ExternalBoundarySample,
    *,
    alpha: float,
    dist_eps: float,
    rhs_floor: float,
    diag_coeff: float,
    row_sum_zero: bool,
    singular_diag_scale: float,
    nfp: int,
    mf: int,
    nf: int,
    lasym: bool,
    wint_vmec: np.ndarray | None = None,
    factor_physical_matrix: bool = True,
) -> NestorVmecLikeCache:
    """Build a dense boundary-integral-like operator on the VMEC angular grid."""

    R = np.asarray(sample.R, dtype=float)
    Z = np.asarray(sample.Z, dtype=float)
    ntheta, nzeta = R.shape
    npts = int(ntheta * nzeta)
    phi_grid = np.asarray(sample.phi, dtype=float)
    if phi_grid.shape != R.shape:
        phi_grid = np.broadcast_to(phi_grid, R.shape)
    x = R * np.cos(phi_grid)
    y = R * np.sin(phi_grid)
    coords = np.stack([x, y, Z], axis=-1).reshape(npts, 3)
    det = np.asarray(sample.vac_ext.det_guv, dtype=float)
    w = np.sqrt(np.maximum(np.abs(det), 0.0)).reshape(npts)
    w_sum = float(np.sum(w))
    if not np.isfinite(w_sum) or w_sum <= rhs_floor:
        w = np.full((npts,), 1.0 / float(max(1, npts)), dtype=float)
    else:
        w = w / w_sum

    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1) + float(dist_eps) ** 2)
    invdist = np.where(dist > 0.0, 1.0 / dist, 0.0)
    np.fill_diagonal(invdist, 0.0)

    kernel = (invdist * w[None, :]) / (4.0 * np.pi)
    if bool(row_sum_zero):
        row_sum = np.sum(kernel, axis=1)
        kernel[np.arange(npts), np.arange(npts)] -= row_sum

    diag_extra = np.zeros((npts,), dtype=float)
    if float(singular_diag_scale) != 0.0:
        dist_nodiag = np.asarray(dist, dtype=float).copy()
        np.fill_diagonal(dist_nodiag, np.inf)
        h = np.minimum(np.min(dist_nodiag, axis=1), 1.0 / float(max(1, npts)))
        h = np.maximum(h, float(dist_eps))
        diag_extra = (float(singular_diag_scale) / (4.0 * np.pi)) * (w / h)

    matrix = float(alpha) * kernel
    matrix[np.arange(npts), np.arange(npts)] += float(diag_coeff) + diag_extra
    rhs_scale = np.where(w > rhs_floor, w, rhs_floor)

    wint_use = np.asarray(wint_vmec, dtype=float) if wint_vmec is not None else np.asarray(w, dtype=float).reshape(ntheta, nzeta)
    mode_basis = build_vmec_mode_basis(
        ntheta=ntheta,
        nzeta=nzeta,
        nfp=int(nfp),
        mf=int(mf),
        nf=int(nf),
        lasym=bool(lasym),
        wint=np.asarray(wint_use, dtype=float),
    )
    sinmni = np.asarray(mode_basis["sinmni"], dtype=float)
    cosmni = np.asarray(mode_basis["cosmni"], dtype=float)
    B = np.concatenate([sinmni, cosmni], axis=1) if bool(lasym) else sinmni
    mode_matrix = B.T @ (matrix @ B)
    mnpd = int(mode_basis["mnpd"])
    if mnpd > 0:
        pi3 = float(4.0 * (np.pi**3))
        mode_matrix[:mnpd, :mnpd][np.diag_indices(mnpd)] += pi3
        if bool(lasym):
            mode_matrix[mnpd:, mnpd:][np.diag_indices(mnpd)] += pi3
            mn0 = int(mode_basis["mn0"])
            if 0 <= mn0 < mnpd:
                mode_matrix[mnpd + mn0, mnpd + mn0] += pi3

    return NestorVmecLikeCache(
        ntheta=ntheta,
        nzeta=nzeta,
        matrix=matrix,
        rhs_scale=rhs_scale,
        mode_basis=mode_basis,
        mode_matrix=mode_matrix,
        matrix_lu=dense_lu_factor(matrix) if bool(factor_physical_matrix) else None,
        mode_matrix_lu=dense_lu_factor(mode_matrix),
    )


def solve_vmec_like_dense(rhs: np.ndarray, cache: NestorVmecLikeCache) -> np.ndarray:
    """Solve solve vmec like dense for direct-coil free-boundary solve and branch-local adjoint validation."""
    rhs_flat = np.asarray(rhs, dtype=float).reshape(-1) * np.asarray(cache.rhs_scale, dtype=float)
    phi_flat = dense_lu_solve(cache.matrix_lu, np.asarray(cache.matrix, dtype=float), rhs_flat)
    phi = phi_flat.reshape(int(cache.ntheta), int(cache.nzeta))
    phi = phi - float(np.mean(phi))
    return phi


def vmec_source_from_gsource(*, gsource: np.ndarray, basis: dict[str, Any]) -> np.ndarray:
    """VMEC fouri.f source symmetrization from gsource."""

    gsrc = np.asarray(gsource, dtype=float).reshape(-1)
    onp = float(basis["onp"])
    nuv3 = int(basis.get("nuv3", gsrc.size))
    nuv_full = int(basis.get("nuv_full", nuv3))
    if bool(basis["lasym"]):
        src = onp * gsrc[:nuv3]
    elif gsrc.size >= nuv_full and "imirr_full" in basis:
        imirr_full = np.asarray(basis["imirr_full"], dtype=np.int64)
        src = 0.5 * onp * (gsrc[:nuv3] - gsrc[imirr_full[:nuv3]])
    else:
        imirr = np.asarray(basis["imirr"], dtype=np.int64)
        src = 0.5 * onp * (gsrc[:nuv3] - gsrc[imirr[:nuv3]])
    return np.asarray(src, dtype=float)



def spectral_second_derivatives_2d(field: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Periodic spectral second derivatives on a uniform `(u,v)` grid."""

    f = np.asarray(field, dtype=float)
    nu, nv = f.shape
    ku = np.fft.fftfreq(nu, d=1.0 / float(max(1, nu)))
    kv = np.fft.fftfreq(nv, d=1.0 / float(max(1, nv)))
    fh = np.fft.fftn(f)
    duu = np.fft.ifftn((-(ku[:, None] ** 2)) * fh).real
    dvv = np.fft.ifftn((-(kv[None, :] ** 2)) * fh).real
    duv = np.fft.ifftn((-(ku[:, None] * kv[None, :])) * fh).real
    return np.asarray(duu, dtype=float), np.asarray(duv, dtype=float), np.asarray(dvv, dtype=float)


def vmec_precal_tan_tables(*, nu: int, nv: int, nvper: int) -> tuple[np.ndarray, np.ndarray]:
    """VMEC ``precal.f`` tan tables used by ``greenf.f``."""

    nu = max(1, int(nu))
    nv = max(1, int(nv))
    nvper = max(1, int(nvper))
    kp_count = int(nvper) if int(nv) == 1 else 1
    nuv_tan = int(2 * nu * nv * kp_count)
    tanu = np.zeros((nuv_tan,), dtype=float)
    tanv = np.zeros((nuv_tan,), dtype=float)
    alu = 2.0 * np.pi / float(nu)
    alv = 2.0 * np.pi / float(nv)
    alp_per = 2.0 * np.pi / float(nvper)
    epstan = np.finfo(float).eps
    bigno = 1.0e50
    i = 0
    for kp in range(1, kp_count + 1):
        argp = 0.5 * alp_per * float(kp - 1)
        for ku in range(1, 2 * nu + 1):
            argu = 0.5 * alu * float(ku - 1)
            near_qpi = abs(argu - 0.25 * 2.0 * np.pi) < epstan
            near_3qpi = abs(argu - 0.75 * 2.0 * np.pi) < epstan
            for kv in range(1, nv + 1):
                argv = 0.5 * alv * float(kv - 1) + argp
                if near_qpi or near_3qpi:
                    tanu[i] = bigno
                else:
                    tanu[i] = 2.0 * np.tan(argu)
                if abs(argv - 0.25 * 2.0 * np.pi) < epstan:
                    tanv[i] = bigno
                else:
                    tanv[i] = 2.0 * np.tan(argv)
                i += 1
    return np.asarray(tanu, dtype=float), np.asarray(tanv, dtype=float)


def ensure_vmec_nonsingular_kernel_tables(*, basis: dict[str, Any], nv: int, nvper: int) -> dict[str, np.ndarray]:
    """Cache VMEC nonsingular Green-function helper tables on the mode basis."""

    nv = max(1, int(nv))
    nvper = max(1, int(nvper))
    cache = basis.get("_nonsingular_kernel_tables")
    if (
        isinstance(cache, dict)
        and int(cache.get("nv", -1)) == nv
        and int(cache.get("nvper", -1)) == nvper
    ):
        return cache

    nu = int(basis["nu_full"])
    mf = int(basis["mf"])
    nf = int(basis["nf"])
    onp = float(basis["onp"])
    nuv_full = int(basis["nuv_full"])

    tanu, tanv = vmec_precal_tan_tables(nu=nu, nv=nv, nvper=nvper)

    alv = 2.0 * np.pi / float(max(1, nv))
    alvp = onp * alv
    kv = np.arange(nv, dtype=np.int64)
    cos_v = np.cos(alvp * kv)
    sin_v = np.sin(alvp * kv)
    cosuv = np.broadcast_to(cos_v[None, :], (nu, nv)).reshape(-1)
    sinuv = np.broadcast_to(sin_v[None, :], (nu, nv)).reshape(-1)

    alp_per = 2.0 * np.pi / float(max(1, nvper))
    cosper = np.cos(alp_per * np.arange(nvper, dtype=float))
    sinper = np.sin(alp_per * np.arange(nvper, dtype=float))

    cosv_tab = np.zeros((nf + 1, nv), dtype=float)
    sinv_tab = np.zeros((nf + 1, nv), dtype=float)
    kv_idx = np.arange(nv, dtype=float)
    for n in range(0, nf + 1):
        dn1 = alv * float(n)
        cosv_tab[n, :] = np.cos(dn1 * kv_idx)
        sinv_tab[n, :] = np.sin(dn1 * kv_idx)

    alu = 2.0 * np.pi / float(max(1, nu))
    nu_fourp = int(nu // 2 + 1)
    cosui = np.zeros((mf + 1, nu_fourp), dtype=float)
    sinui = np.zeros((mf + 1, nu_fourp), dtype=float)
    ku_idx = np.arange(nu_fourp, dtype=float)
    for m in range(0, mf + 1):
        c = np.cos(alu * float(m) * ku_idx)
        s = np.sin(alu * float(m) * ku_idx)
        cosui[m, :] = c * alu * alv * 2.0
        sinui[m, :] = s * alu * alv * 2.0
        cosui[m, 0] *= 0.5
        cosui[m, -1] *= 0.5

    cache = {
        "nv": np.asarray(nv, dtype=np.int64),
        "nvper": np.asarray(nvper, dtype=np.int64),
        "idx_all": np.arange(nuv_full, dtype=np.int64),
        "tanu": np.asarray(tanu, dtype=float),
        "tanv": np.asarray(tanv, dtype=float),
        "cosuv": np.asarray(cosuv, dtype=float),
        "sinuv": np.asarray(sinuv, dtype=float),
        "cosper": np.asarray(cosper, dtype=float),
        "sinper": np.asarray(sinper, dtype=float),
        "cosv_tab": np.asarray(cosv_tab, dtype=float),
        "sinv_tab": np.asarray(sinv_tab, dtype=float),
        "cosui": np.asarray(cosui, dtype=float),
        "sinui": np.asarray(sinui, dtype=float),
    }
    basis["_nonsingular_kernel_tables"] = cache
    return cache


def vmec_nonsingular_gsource_from_bexni(
    *,
    sample: ExternalBoundarySample,
    basis: dict[str, Any],
    bexni: np.ndarray,
    signgs: int,
    nvper: int,
) -> np.ndarray:
    """Approximate VMEC greenf+gstore source assembly on one boundary period.

    This ports the key numerics of ``greenf.f``/``scalpot.f`` source accumulation:
    ``gstore(i) = sum_ip bexni(ip) * delgr(i;ip)``, where ``delgr`` is the
    non-singular Green-function remainder over field periods.
    """

    ntheta3, nzeta = sample.R.shape
    nu = int(basis.get("nu_full", ntheta3))
    nv = int(nzeta)
    nuv_full = int(nu * nv)
    nuv3 = int(ntheta3 * nv)
    if nuv_full <= 0:
        return np.zeros((0,), dtype=float)

    onp = float(basis["onp"])
    onp2 = onp * onp
    signgs = int(signgs)
    nvper = max(1, int(nvper))

    R_red = np.asarray(sample.R, dtype=float)
    Z_red = np.asarray(sample.Z, dtype=float)
    Ru_red = np.asarray(sample.Ru, dtype=float)
    Zu_red = np.asarray(sample.Zu, dtype=float)
    Rv_red = np.asarray(sample.Rv, dtype=float)
    Zv_red = np.asarray(sample.Zv, dtype=float)

    if (nu == ntheta3) or bool(basis.get("lasym", False)):
        R2 = np.asarray(R_red, dtype=float)
        Z2 = np.asarray(Z_red, dtype=float)
        Ru2 = np.asarray(Ru_red, dtype=float)
        Zu2 = np.asarray(Zu_red, dtype=float)
        Rv2 = np.asarray(Rv_red, dtype=float)
        Zv2 = np.asarray(Zv_red, dtype=float)
    else:
        # Rebuild full `nu` surface arrays from stellarator-symmetric half grid.
        R2 = np.zeros((nu, nv), dtype=float)
        Z2 = np.zeros((nu, nv), dtype=float)
        Ru2 = np.zeros((nu, nv), dtype=float)
        Zu2 = np.zeros((nu, nv), dtype=float)
        Rv2 = np.zeros((nu, nv), dtype=float)
        Zv2 = np.zeros((nu, nv), dtype=float)
        R2[:ntheta3, :] = R_red
        Z2[:ntheta3, :] = Z_red
        Ru2[:ntheta3, :] = Ru_red
        Zu2[:ntheta3, :] = Zu_red
        Rv2[:ntheta3, :] = Rv_red
        Zv2[:ntheta3, :] = Zv_red
        kv_m = (nv - np.arange(nv, dtype=np.int64)) % max(1, nv)
        for ku in range(1, max(1, ntheta3 - 1)):
            km = (nu - ku) % max(1, nu)
            if km < ntheta3:
                continue
            # Stellarator symmetry for missing half-grid rows:
            # (u,v) -> (-u,+v) maps to source rows sampled at (+u,-v).
            R2[km, :] = R_red[ku, kv_m]
            Z2[km, :] = -Z_red[ku, kv_m]
            Ru2[km, :] = -Ru_red[ku, kv_m]
            Zu2[km, :] = Zu_red[ku, kv_m]
            Rv2[km, :] = -Rv_red[ku, kv_m]
            Zv2[km, :] = Zv_red[ku, kv_m]

    # Prefer exact modal second derivatives from surface sampling (VMEC surface.f).
    # For stellarator-symmetric runs, source derivatives are only needed on the
    # reduced `ntheta3` rows (primed mesh), so we embed them into full arrays.
    have_second = (
        sample.ruu is not None
        and sample.ruv is not None
        and sample.rvv is not None
        and sample.zuu is not None
        and sample.zuv is not None
        and sample.zvv is not None
    )
    if have_second:
        ruu_s = np.asarray(sample.ruu, dtype=float)
        ruv_s = np.asarray(sample.ruv, dtype=float)
        rvv_s = np.asarray(sample.rvv, dtype=float)
        zuu_s = np.asarray(sample.zuu, dtype=float)
        zuv_s = np.asarray(sample.zuv, dtype=float)
        zvv_s = np.asarray(sample.zvv, dtype=float)
        if ruu_s.shape == R2.shape:
            ruu, ruv, rvv = ruu_s, ruv_s, rvv_s
            zuu, zuv, zvv = zuu_s, zuv_s, zvv_s
        elif ruu_s.shape == (ntheta3, nv):
            ruu = np.zeros_like(R2)
            ruv = np.zeros_like(R2)
            rvv = np.zeros_like(R2)
            zuu = np.zeros_like(R2)
            zuv = np.zeros_like(R2)
            zvv = np.zeros_like(R2)
            ruu[:ntheta3, :] = ruu_s
            ruv[:ntheta3, :] = ruv_s
            rvv[:ntheta3, :] = rvv_s
            zuu[:ntheta3, :] = zuu_s
            zuv[:ntheta3, :] = zuv_s
            zvv[:ntheta3, :] = zvv_s
        else:
            ruu, ruv, rvv = spectral_second_derivatives_2d(R2)
            zuu, zuv, zvv = spectral_second_derivatives_2d(Z2)
    else:
        ruu, ruv, rvv = spectral_second_derivatives_2d(R2)
        zuu, zuv, zvv = spectral_second_derivatives_2d(Z2)
    R = R2.reshape(-1)
    Z = Z2.reshape(-1)
    Ru = Ru2.reshape(-1)
    Zu = Zu2.reshape(-1)
    Rv = Rv2.reshape(-1)
    Zv = Zv2.reshape(-1)
    ruu = ruu.reshape(-1)
    rvv = rvv.reshape(-1)
    ruv = ruv.reshape(-1)
    zuu = zuu.reshape(-1)
    zvv = zvv.reshape(-1)
    zuv = zuv.reshape(-1)

    snr = float(signgs) * R * Zu
    snv = float(signgs) * (Ru * Zv - Rv * Zu)
    snz = -float(signgs) * R * Ru
    guu_b = Ru * Ru + Zu * Zu
    guv_b = (Ru * Rv + Zu * Zv) * onp * 2.0
    gvv_b = (Rv * Rv + Zv * Zv + R * R) * onp2
    auu = 0.5 * (snr * ruu + snz * zuu)
    auv = (snr * ruv + snv * Ru + snz * zuv) * onp
    avv = (snv * Rv + 0.5 * (snr * (rvv - R) + snz * zvv)) * onp2
    rzb2 = R * R + Z * Z

    tables = ensure_vmec_nonsingular_kernel_tables(basis=basis, nv=nv, nvper=nvper)
    idx_all = np.asarray(tables["idx_all"], dtype=np.int64)
    tanu = np.asarray(tables["tanu"], dtype=float)
    tanv = np.asarray(tables["tanv"], dtype=float)
    cosuv = np.asarray(tables["cosuv"], dtype=float)
    sinuv = np.asarray(tables["sinuv"], dtype=float)
    cosper = np.asarray(tables["cosper"], dtype=float)
    sinper = np.asarray(tables["sinper"], dtype=float)
    rcosuv = R * cosuv
    rsinuv = R * sinuv

    bex = np.asarray(bexni, dtype=float).reshape(-1)
    if bex.size < nuv3:
        bex = np.resize(bex, (nuv3,))
    else:
        bex = bex[:nuv3]

    gstore = np.zeros((nuv_full,), dtype=float)
    for ip in range(nuv3):
        xip = rcosuv[ip]
        yip = rsinuv[ip]
        ivoff = nuv_full - ip
        iskip = ip // nv
        iuoff = nuv_full - nv * iskip

        gsave = rzb2[ip] + rzb2 - 2.0 * Z[ip] * Z
        delgr = np.zeros((nuv_full,), dtype=float)
        for kp in range(nvper):
            xper = xip * cosper[kp] - yip * sinper[kp]
            yper = yip * cosper[kp] + xip * sinper[kp]
            base = gsave - 2.0 * (xper * rcosuv + yper * rsinuv)
            if kp == 0 or nv == 1:
                tidx_u = idx_all + iuoff
                ivoff_k = ivoff + (2 * nu * kp if nv == 1 else 0)
                tidx_v = idx_all + ivoff_k
                ga1 = tanu[tidx_u] * (guu_b[ip] * tanu[tidx_u] + guv_b[ip] * tanv[tidx_v]) + gvv_b[ip] * tanv[tidx_v] * tanv[tidx_v]
                ga2 = tanu[tidx_u] * (auu[ip] * tanu[tidx_u] + auv[ip] * tanv[tidx_v]) + avv[ip] * tanv[tidx_v] * tanv[tidx_v]
                ga2 = ga2 / ga1
                ga1s = 1.0 / np.sqrt(ga1)
                mask = (idx_all != ip) if kp == 0 else np.ones_like(idx_all, dtype=bool)
                if np.any(mask):
                    base_m = base[mask]
                    htemp_m = np.sqrt(1.0 / base_m)
                    delgr[mask] += htemp_m - ga1s[mask]
            else:
                htemp = np.sqrt(1.0 / base)
                delgr += htemp
        # VMEC greenf.f: when nv==1, normalize the field-period sum by nvper.
        if nv == 1 and nvper > 1:
            delgr /= float(nvper)
        gstore += bex[ip] * delgr

    return np.asarray(gstore, dtype=float)


def vmec_mode_matrix_from_grpmn(
    *,
    grpmn: np.ndarray,
    basis: dict[str, Any],
) -> np.ndarray:
    """Build VMEC mode-space matrix from `grpmn` using ``fouri.f`` formulas."""

    g = np.asarray(grpmn, dtype=float)
    mnpd = int(basis["mnpd"])
    lasym = bool(basis["lasym"])
    sinmni = np.asarray(basis["sinmni"], dtype=float)
    cosmni = np.asarray(basis["cosmni"], dtype=float)
    # VMEC/NESTOR `pi3` from precal.f: p5*pi2**3 = 4*pi**3.
    pi3 = float(4.0 * (np.pi**3))
    mn0 = int(basis.get("mn0", 0))

    if g.ndim != 2 or g.shape[0] < mnpd:
        raise ValueError("invalid_grpmn_shape")
    xmpot = np.asarray(basis["xmpot"], dtype=np.int64)
    n_raw = np.asarray(basis["n_raw"], dtype=np.int64)
    skip_col = np.logical_and(xmpot == 0, n_raw < 0)
    gsin = g[:mnpd, :]
    a11 = gsin @ sinmni
    a11 = np.asarray(a11, dtype=float)
    if np.any(skip_col):
        # fouri.f skips m=0,n<0 in the primed-mesh loop: these are column-only skips.
        a11[:, skip_col] = 0.0
    a11[np.diag_indices(mnpd)] += pi3

    if not lasym:
        return a11

    if g.shape[0] < 2 * mnpd:
        raise ValueError("invalid_grpmn_shape_lasym")
    gcos = g[mnpd : 2 * mnpd, :]
    a12 = gsin @ cosmni
    a21 = gcos @ sinmni
    a22 = gcos @ cosmni
    if np.any(skip_col):
        a12 = np.asarray(a12, dtype=float)
        a21 = np.asarray(a21, dtype=float)
        a22 = np.asarray(a22, dtype=float)
        a12[:, skip_col] = 0.0
        a21[:, skip_col] = 0.0
        a22[:, skip_col] = 0.0
    a22 = np.asarray(a22, dtype=float)
    a22[np.diag_indices(mnpd)] += pi3
    if 0 <= mn0 < mnpd:
        a22[mn0, mn0] += pi3

    out = np.zeros((2 * mnpd, 2 * mnpd), dtype=float)
    out[:mnpd, :mnpd] = a11
    out[:mnpd, mnpd:] = a12
    out[mnpd:, :mnpd] = a21
    out[mnpd:, mnpd:] = a22
    return out


def vmec_nonsingular_terms_from_bexni(
    *,
    sample: ExternalBoundarySample,
    basis: dict[str, Any],
    bexni: np.ndarray,
    signgs: int,
    nvper: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute VMEC-like non-singular source and matrix kernel terms.

    Returns:
      - `gstore` (`gsource_full`) on the full `nu*nv` grid.
      - `grpmn_nonsing` Fourier-kernel contribution in mode space (`mnpd2,nuv3`).
    """

    ntheta3, nzeta = sample.R.shape
    nu = int(basis.get("nu_full", ntheta3))
    nv = int(nzeta)
    nuv_full = int(nu * nv)
    nuv3 = int(ntheta3 * nv)
    if nuv_full <= 0 or nuv3 <= 0:
        return np.zeros((0,), dtype=float), np.zeros((0, 0), dtype=float)

    mf = int(basis["mf"])
    nf = int(basis["nf"])
    mnpd = int(basis["mnpd"])
    lasym = bool(basis["lasym"])
    mnpd2 = int(basis["mnpd2"])
    onp = float(basis["onp"])
    signgs = int(signgs)
    nvper = max(1, int(nvper))

    R_red = np.asarray(sample.R, dtype=float)
    Z_red = np.asarray(sample.Z, dtype=float)
    Ru_red = np.asarray(sample.Ru, dtype=float)
    Zu_red = np.asarray(sample.Zu, dtype=float)
    Rv_red = np.asarray(sample.Rv, dtype=float)
    Zv_red = np.asarray(sample.Zv, dtype=float)

    if (nu == ntheta3) or lasym:
        R2 = np.asarray(R_red, dtype=float)
        Z2 = np.asarray(Z_red, dtype=float)
        Ru2 = np.asarray(Ru_red, dtype=float)
        Zu2 = np.asarray(Zu_red, dtype=float)
        Rv2 = np.asarray(Rv_red, dtype=float)
        Zv2 = np.asarray(Zv_red, dtype=float)
    else:
        R2 = np.zeros((nu, nv), dtype=float)
        Z2 = np.zeros((nu, nv), dtype=float)
        Ru2 = np.zeros((nu, nv), dtype=float)
        Zu2 = np.zeros((nu, nv), dtype=float)
        Rv2 = np.zeros((nu, nv), dtype=float)
        Zv2 = np.zeros((nu, nv), dtype=float)
        R2[:ntheta3, :] = R_red
        Z2[:ntheta3, :] = Z_red
        Ru2[:ntheta3, :] = Ru_red
        Zu2[:ntheta3, :] = Zu_red
        Rv2[:ntheta3, :] = Rv_red
        Zv2[:ntheta3, :] = Zv_red
        kv_m = (nv - np.arange(nv, dtype=np.int64)) % max(1, nv)
        for ku in range(1, max(1, ntheta3 - 1)):
            km = (nu - ku) % max(1, nu)
            if km < ntheta3:
                continue
            R2[km, :] = R_red[ku, kv_m]
            Z2[km, :] = -Z_red[ku, kv_m]
            Ru2[km, :] = -Ru_red[ku, kv_m]
            Zu2[km, :] = Zu_red[ku, kv_m]
            Rv2[km, :] = -Rv_red[ku, kv_m]
            Zv2[km, :] = Zv_red[ku, kv_m]

    have_second = (
        sample.ruu is not None
        and sample.ruv is not None
        and sample.rvv is not None
        and sample.zuu is not None
        and sample.zuv is not None
        and sample.zvv is not None
    )
    if have_second:
        ruu_s = np.asarray(sample.ruu, dtype=float)
        ruv_s = np.asarray(sample.ruv, dtype=float)
        rvv_s = np.asarray(sample.rvv, dtype=float)
        zuu_s = np.asarray(sample.zuu, dtype=float)
        zuv_s = np.asarray(sample.zuv, dtype=float)
        zvv_s = np.asarray(sample.zvv, dtype=float)
        if ruu_s.shape == R2.shape:
            ruu, ruv, rvv = ruu_s, ruv_s, rvv_s
            zuu, zuv, zvv = zuu_s, zuv_s, zvv_s
        elif ruu_s.shape == (ntheta3, nv):
            ruu = np.zeros_like(R2)
            ruv = np.zeros_like(R2)
            rvv = np.zeros_like(R2)
            zuu = np.zeros_like(R2)
            zuv = np.zeros_like(R2)
            zvv = np.zeros_like(R2)
            ruu[:ntheta3, :] = ruu_s
            ruv[:ntheta3, :] = ruv_s
            rvv[:ntheta3, :] = rvv_s
            zuu[:ntheta3, :] = zuu_s
            zuv[:ntheta3, :] = zuv_s
            zvv[:ntheta3, :] = zvv_s
        else:
            ruu, ruv, rvv = spectral_second_derivatives_2d(R2)
            zuu, zuv, zvv = spectral_second_derivatives_2d(Z2)
    else:
        ruu, ruv, rvv = spectral_second_derivatives_2d(R2)
        zuu, zuv, zvv = spectral_second_derivatives_2d(Z2)
    R = R2.reshape(-1)
    Z = Z2.reshape(-1)
    Ru = Ru2.reshape(-1)
    Zu = Zu2.reshape(-1)
    Rv = Rv2.reshape(-1)
    Zv = Zv2.reshape(-1)
    ruu = ruu.reshape(-1)
    rvv = rvv.reshape(-1)
    ruv = ruv.reshape(-1)
    zuu = zuu.reshape(-1)
    zvv = zvv.reshape(-1)
    zuv = zuv.reshape(-1)

    snr = float(signgs) * R * Zu
    snv = float(signgs) * (Ru * Zv - Rv * Zu)
    snz = -float(signgs) * R * Ru
    drv = -(R * snr + Z * snz)
    guu_b = Ru * Ru + Zu * Zu
    guv_b = (Ru * Rv + Zu * Zv) * onp * 2.0
    gvv_b = (Rv * Rv + Zv * Zv + R * R) * (onp * onp)
    auu = 0.5 * (snr * ruu + snz * zuu)
    auv = (snr * ruv + snv * Ru + snz * zuv) * onp
    avv = (snv * Rv + 0.5 * (snr * (rvv - R) + snz * zvv)) * (onp * onp)
    rzb2 = R * R + Z * Z

    tables = ensure_vmec_nonsingular_kernel_tables(basis=basis, nv=nv, nvper=nvper)
    idx_all = np.asarray(tables["idx_all"], dtype=np.int64)
    tanu = np.asarray(tables["tanu"], dtype=float)
    tanv = np.asarray(tables["tanv"], dtype=float)
    cosuv = np.asarray(tables["cosuv"], dtype=float)
    sinuv = np.asarray(tables["sinuv"], dtype=float)
    cosper = np.asarray(tables["cosper"], dtype=float)
    sinper = np.asarray(tables["sinper"], dtype=float)
    cosv_tab = np.asarray(tables["cosv_tab"], dtype=float)
    sinv_tab = np.asarray(tables["sinv_tab"], dtype=float)
    cosui = np.asarray(tables["cosui"], dtype=float)
    sinui = np.asarray(tables["sinui"], dtype=float)
    nu_fourp = int(cosui.shape[1])
    rcosuv = R * cosuv
    rsinuv = R * sinuv

    bex = np.asarray(bexni, dtype=float).reshape(-1)
    if bex.size < nuv3:
        bex = np.resize(bex, (nuv3,))
    else:
        bex = bex[:nuv3]

    imirr_full = np.asarray(basis["imirr_full"], dtype=np.int64)
    grpmn_nonsing = np.zeros((mnpd2, nuv3), dtype=float)
    mf1 = mf + 1
    ndim = 2 if lasym else 1
    iuv_grid = (np.arange(int(nu_fourp), dtype=np.int64)[:, None] * int(nv)) + np.arange(int(nv), dtype=np.int64)[
        None, :
    ]
    iuv_grid = np.asarray(iuv_grid, dtype=np.int64)
    iref_grid = np.asarray(imirr_full[iuv_grid], dtype=np.int64)
    cosv_modes = 0.5 * onp * np.asarray(cosv_tab[: nf + 1, :], dtype=float)
    sinv_modes = 0.5 * onp * np.asarray(sinv_tab[: nf + 1, :], dtype=float)
    m_idx = np.arange(mf + 1, dtype=np.int64)
    n_idx = np.arange(nf + 1, dtype=np.int64)
    idx_p_grid = m_idx[:, None] + (n_idx[None, :] + nf) * mf1
    idx_m_grid = m_idx[:, None] + ((-n_idx[None, :]) + nf) * mf1
    add_negative_n = (n_idx[None, :] != 0) & (m_idx[:, None] != 0)
    idx_p_flat = idx_p_grid.reshape(-1)
    idx_m_flat = idx_m_grid.reshape(-1)
    negative_n_flat = np.asarray(add_negative_n.reshape(-1), dtype=bool)
    sinm_sym = np.asarray(sinui[: mf + 1, :], dtype=float)
    cosm_sym = -np.asarray(cosui[: mf + 1, :], dtype=float)
    sinm_asym = np.asarray(cosui[: mf + 1, :], dtype=float) if lasym else None
    cosm_asym = np.asarray(sinui[: mf + 1, :], dtype=float) if lasym else None

    try:
        ip_chunk = int(os.getenv("VMEC_JAX_FREEB_NONSINGULAR_IP_CHUNK", "64"))
    except Exception:
        ip_chunk = 64
    ip_chunk = max(1, min(int(ip_chunk), int(nuv3)))

    gstore = np.zeros((nuv_full,), dtype=float)
    idx_all_b = idx_all[None, :]
    rcosuv_b = rcosuv[None, :]
    rsinuv_b = rsinuv[None, :]
    z_b = Z[None, :]
    for ip0 in range(0, nuv3, ip_chunk):
        ip1 = min(nuv3, ip0 + ip_chunk)
        ip_idx = np.arange(ip0, ip1, dtype=np.int64)
        n_chunk = int(ip_idx.size)

        xip = rcosuv[ip_idx]
        yip = rsinuv[ip_idx]
        ivoff = nuv_full - ip_idx
        iskip = ip_idx // nv
        iuoff = nuv_full - nv * iskip
        gsave = rzb2[ip_idx, None] + rzb2[None, :] - 2.0 * Z[ip_idx, None] * z_b
        dsave = drv[ip_idx, None] + z_b * snz[ip_idx, None]
        delgr = np.zeros((n_chunk, nuv_full), dtype=float)
        delgrp = np.zeros((n_chunk, nuv_full), dtype=float)

        for kp in range(nvper):
            xper = xip * cosper[kp] - yip * sinper[kp]
            yper = yip * cosper[kp] + xip * sinper[kp]
            sxsave = (snr[ip_idx] * xper - snv[ip_idx] * yper) / R[ip_idx]
            sysave = (snr[ip_idx] * yper + snv[ip_idx] * xper) / R[ip_idx]
            base = gsave - 2.0 * (xper[:, None] * rcosuv_b + yper[:, None] * rsinuv_b)
            deriv_num = rcosuv_b * sxsave[:, None] + rsinuv_b * sysave[:, None] + dsave

            if kp == 0 or nv == 1:
                tidx_u = idx_all_b + iuoff[:, None]
                ivoff_k = ivoff + (2 * nu * kp if nv == 1 else 0)
                tidx_v = idx_all_b + ivoff_k[:, None]
                tanu_use = tanu[tidx_u]
                tanv_use = tanv[tidx_v]
                ga1 = tanu_use * (
                    guu_b[ip_idx, None] * tanu_use + guv_b[ip_idx, None] * tanv_use
                ) + gvv_b[ip_idx, None] * tanv_use * tanv_use
                ga2 = tanu_use * (
                    auu[ip_idx, None] * tanu_use + auv[ip_idx, None] * tanv_use
                ) + avv[ip_idx, None] * tanv_use * tanv_use
                ga2 = ga2 / ga1
                ga1s = 1.0 / np.sqrt(ga1)
                if kp == 0:
                    mask = np.ones((n_chunk, nuv_full), dtype=bool)
                    mask[np.arange(n_chunk, dtype=np.int64), ip_idx] = False
                else:
                    mask = np.ones((n_chunk, nuv_full), dtype=bool)
                safe_base = np.where(mask, base, 1.0)
                ftemp = 1.0 / safe_base
                htemp = np.sqrt(ftemp)
                deriv = ftemp * htemp * deriv_num
                delgr += np.where(mask, htemp - ga1s, 0.0)
                delgrp += np.where(mask, deriv - ga2 * ga1s, 0.0)
            else:
                ftemp = 1.0 / base
                htemp = np.sqrt(ftemp)
                deriv = ftemp * htemp * deriv_num
                delgr += htemp
                delgrp += deriv

        # VMEC greenf.f: when nv==1, normalize both non-singular sums by nvper.
        if nv == 1 and nvper > 1:
            scale = 1.0 / float(nvper)
            delgr *= scale
            delgrp *= scale

        # Keep the gstore accumulation order explicit for close parity with the
        # scalar Fortran-style formulation while still vectorizing the expensive
        # kernel construction above.
        for loc, ip in enumerate(ip_idx):
            gstore += bex[int(ip)] * delgr[loc]

        del_iuv = delgrp[:, iuv_grid]
        del_ref = delgrp[:, iref_grid]
        ka_grid = del_iuv - del_ref
        g1_sym = np.einsum("cuv,fv->cuf", ka_grid, cosv_modes, optimize=True)
        g2_sym = np.einsum("cuv,fv->cuf", ka_grid, sinv_modes, optimize=True)

        for isym in range(ndim):
            if isym == 0:
                g1_use = g1_sym
                g2_use = g2_sym
                sinm_table = sinm_sym
                cosm_table = cosm_sym
                row_off = 0
            else:
                ks_grid = del_iuv + del_ref
                g1_use = np.einsum("cuv,fv->cuf", ks_grid, cosv_modes, optimize=True)
                g2_use = np.einsum("cuv,fv->cuf", ks_grid, sinv_modes, optimize=True)
                sinm_table = sinm_asym
                cosm_table = cosm_asym
                row_off = mnpd

            gcos = np.einsum("mu,cuf->cmf", sinm_table, g1_use, optimize=True)
            gsin = np.einsum("mu,cuf->cmf", cosm_table, g2_use, optimize=True)
            total_plus = (gcos + gsin).reshape(n_chunk, -1)
            total_minus = (gcos - gsin).reshape(n_chunk, -1)
            rows_plus = row_off + idx_p_flat
            rows_minus = row_off + idx_m_flat[negative_n_flat]
            grpmn_nonsing[np.ix_(rows_plus, ip_idx)] += total_plus.T
            grpmn_nonsing[np.ix_(rows_minus, ip_idx)] += total_minus[:, negative_n_flat].T

    # Keep raw fourp accumulation scale; any legacy scale experiments are
    # handled upstream in diagnostics, not in the core assembly path.

    return np.asarray(gstore, dtype=float), np.asarray(grpmn_nonsing, dtype=float)


def vmec_bvec_from_gsource(*, gsource: np.ndarray, basis: dict[str, Any]) -> np.ndarray:
    """Evaluate vmec bvec from gsource for direct-coil free-boundary solve and branch-local adjoint validation."""
    src = vmec_source_from_gsource(gsource=gsource, basis=basis)
    sinmni = np.asarray(basis["sinmni"], dtype=float)
    bsin = sinmni.T @ src
    xmpot = np.asarray(basis["xmpot"], dtype=np.int64)
    n_raw = np.asarray(basis["n_raw"], dtype=np.int64)
    skip_mask = np.logical_and(xmpot == 0, n_raw < 0)
    if np.any(skip_mask):
        bsin = np.asarray(bsin, dtype=float)
        bsin[skip_mask] = 0.0
    if bool(basis["lasym"]):
        cosmni = np.asarray(basis["cosmni"], dtype=float)
        bcos = cosmni.T @ src
        if np.any(skip_mask):
            bcos = np.asarray(bcos, dtype=float)
            bcos[skip_mask] = 0.0
        return np.concatenate([bsin, bcos], axis=0)
    return bsin


def vmec_analytic_terms_from_geometry(
    *,
    sample: ExternalBoundarySample,
    basis: dict[str, Any],
    bexni: np.ndarray,
    signgs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Analytic VMEC terms from ``analyt.f``: `(bvec_analytic, grpmn_analytic)`."""

    mnpd = int(basis["mnpd"])
    lasym = bool(basis["lasym"])
    mf = int(basis["mf"])
    nf = int(basis["nf"])
    onp = float(basis["onp"])
    signgs = int(signgs)
    cmns = np.asarray(basis["cmns"], dtype=float)
    theta = np.asarray(basis["theta"], dtype=float).reshape(-1)
    zeta = np.asarray(basis["zeta"], dtype=float).reshape(-1)
    npts = int(theta.size)
    bex = np.asarray(bexni, dtype=float).reshape(-1)
    if bex.size < npts:
        bex = np.resize(bex, (npts,))
    else:
        bex = bex[:npts]

    R = np.asarray(sample.R, dtype=float).reshape(-1)[:npts]
    Ru = np.asarray(sample.Ru, dtype=float).reshape(-1)[:npts]
    Rv = np.asarray(sample.Rv, dtype=float).reshape(-1)[:npts]
    Zu = np.asarray(sample.Zu, dtype=float).reshape(-1)[:npts]
    Zv = np.asarray(sample.Zv, dtype=float).reshape(-1)[:npts]

    guu_b = Ru * Ru + Zu * Zu
    guv_b = (Ru * Rv + Zu * Zv) * (2.0 * onp)
    gvv_b = (Rv * Rv + Zv * Zv + R * R) * (onp * onp)

    adp = guu_b + guv_b + gvv_b
    adm = guu_b - guv_b + gvv_b
    cma = gvv_b - guu_b
    sqrtc = 2.0 * np.sqrt(gvv_b)
    sqrta = 2.0 * np.sqrt(guu_b)
    sqad1 = np.sqrt(adp)
    sqad2 = np.sqrt(adm)

    tlp = (1.0 / sqad1) * np.log((sqad1 * sqrtc + adp + cma) / (sqad1 * sqrta - adp + cma))
    tlm = (1.0 / sqad2) * np.log((sqad2 * sqrtc + adm + cma) / (sqad2 * sqrta - adm + cma))
    tlp_prev = np.zeros_like(tlp)
    tlm_prev = np.zeros_like(tlm)
    tlpm = tlp + tlm

    bsin = np.zeros((mf + 1, 2 * nf + 1), dtype=float)
    bcos = np.zeros((mf + 1, 2 * nf + 1), dtype=float) if lasym else None
    gsin = np.zeros((mf + 1, 2 * nf + 1, npts), dtype=float)
    gcos = np.zeros((mf + 1, 2 * nf + 1, npts), dtype=float) if lasym else None

    delt1u = adp * adm - cma * cma
    azp1u = np.zeros_like(adp)
    azm1u = np.zeros_like(adm)
    cma11u = np.zeros_like(cma)
    r1p = np.zeros_like(adp)
    r1m = np.zeros_like(adm)
    r0p = np.zeros_like(adp)
    r0m = np.zeros_like(adm)
    ra1p = np.zeros_like(adp)
    ra1m = np.zeros_like(adm)
    azp1u[:] = 0.0
    azm1u[:] = 0.0
    cma11u[:] = 0.0

    # Second-derivative geometry terms (surface.f).
    ntheta3, nzeta = sample.R.shape
    nu_full = int(basis.get("nu_full", ntheta3))
    if ntheta3 * nzeta == npts and ntheta3 > 0 and nzeta > 0:
        R_red = np.asarray(sample.R, dtype=float)
        Z_red = np.asarray(sample.Z, dtype=float)
        Ru_red = np.asarray(sample.Ru, dtype=float)
        Zu_red = np.asarray(sample.Zu, dtype=float)
        Rv_red = np.asarray(sample.Rv, dtype=float)
        Zv_red = np.asarray(sample.Zv, dtype=float)
        nv = int(nzeta)
        have_second = (
            sample.ruu is not None
            and sample.ruv is not None
            and sample.rvv is not None
            and sample.zuu is not None
            and sample.zuv is not None
            and sample.zvv is not None
        )
        if have_second and np.asarray(sample.ruu).shape == (ntheta3, nv):
            # Preferred VMEC-equivalent path: second derivatives synthesized
            # directly from modal coefficients on the reduced surface grid.
            R_eval = np.asarray(R_red, dtype=float)
            Ru_eval = np.asarray(Ru_red, dtype=float)
            Rv_eval = np.asarray(Rv_red, dtype=float)
            Zu_eval = np.asarray(Zu_red, dtype=float)
            Zv_eval = np.asarray(Zv_red, dtype=float)
            ruu = np.asarray(sample.ruu, dtype=float)
            ruv = np.asarray(sample.ruv, dtype=float)
            rvv = np.asarray(sample.rvv, dtype=float)
            zuu = np.asarray(sample.zuu, dtype=float)
            zuv = np.asarray(sample.zuv, dtype=float)
            zvv = np.asarray(sample.zvv, dtype=float)
        else:
            if (nu_full == ntheta3) or lasym:
                R2 = np.asarray(R_red, dtype=float)
                Z2 = np.asarray(Z_red, dtype=float)
                Ru2 = np.asarray(Ru_red, dtype=float)
                Zu2 = np.asarray(Zu_red, dtype=float)
                Rv2 = np.asarray(Rv_red, dtype=float)
                Zv2 = np.asarray(Zv_red, dtype=float)
            else:
                R2 = np.zeros((nu_full, nv), dtype=float)
                Z2 = np.zeros((nu_full, nv), dtype=float)
                Ru2 = np.zeros((nu_full, nv), dtype=float)
                Zu2 = np.zeros((nu_full, nv), dtype=float)
                Rv2 = np.zeros((nu_full, nv), dtype=float)
                Zv2 = np.zeros((nu_full, nv), dtype=float)
                R2[:ntheta3, :] = R_red
                Z2[:ntheta3, :] = Z_red
                Ru2[:ntheta3, :] = Ru_red
                Zu2[:ntheta3, :] = Zu_red
                Rv2[:ntheta3, :] = Rv_red
                Zv2[:ntheta3, :] = Zv_red
                kv_m = (nv - np.arange(nv, dtype=np.int64)) % max(1, nv)
                for ku in range(1, max(1, ntheta3 - 1)):
                    km = (nu_full - ku) % max(1, nu_full)
                    if km < ntheta3:
                        continue
                    R2[km, :] = R_red[ku, kv_m]
                    Z2[km, :] = -Z_red[ku, kv_m]
                    Ru2[km, :] = -Ru_red[ku, kv_m]
                    Zu2[km, :] = Zu_red[ku, kv_m]
                    Rv2[km, :] = -Rv_red[ku, kv_m]
                    Zv2[km, :] = Zv_red[ku, kv_m]

            ruu, ruv, rvv = spectral_second_derivatives_2d(R2)
            zuu, zuv, zvv = spectral_second_derivatives_2d(Z2)
            if (nu_full != ntheta3) and (not lasym):
                sl = slice(0, ntheta3)
                R_eval = R2[sl, :]
                Ru_eval = Ru2[sl, :]
                Rv_eval = Rv2[sl, :]
                Zu_eval = Zu2[sl, :]
                Zv_eval = Zv2[sl, :]
                ruu = ruu[sl, :]
                rvv = rvv[sl, :]
                ruv = ruv[sl, :]
                zuu = zuu[sl, :]
                zvv = zvv[sl, :]
                zuv = zuv[sl, :]
            else:
                R_eval = R2
                Ru_eval = Ru2
                Rv_eval = Rv2
                Zu_eval = Zu2
                Zv_eval = Zv2

        sgn = float(signgs)
        snr = sgn * R_eval * Zu_eval
        snv = sgn * (Ru_eval * Zv_eval - Rv_eval * Zu_eval)
        snz = -sgn * R_eval * Ru_eval
        auu = 0.5 * (snr * ruu + snz * zuu)
        auv = (snr * ruv + snv * Ru_eval + snz * zuv) * onp
        avv = (snv * Rv_eval + 0.5 * (snr * (rvv - R_eval) + snz * zvv)) * (onp * onp)
        auu = auu.reshape(-1)
        auv = auv.reshape(-1)
        avv = avv.reshape(-1)
        azp1u = auu + auv + avv
        azm1u = auu - auv + avv
        cma11u = avv - auu
        r1p = (azp1u * (delt1u - cma * cma) / adp - azm1u * adp + 2.0 * cma11u * cma) / delt1u
        r1m = (azm1u * (delt1u - cma * cma) / adm - azp1u * adm + 2.0 * cma11u * cma) / delt1u
        r0p = (-azp1u * adm * cma / adp - azm1u * cma + 2.0 * cma11u * adm) / delt1u
        r0m = (-azm1u * adp * cma / adm - azp1u * cma + 2.0 * cma11u * adp) / delt1u
        ra1p = azp1u / adp
        ra1m = azm1u / adm

    sign1 = 1.0
    fl1 = 0.0
    for l in range(0, mf + nf + 1):
        fl = fl1
        slp = (r1p * fl + ra1p) * tlp + r0p * fl * tlp_prev - (r1p + r0p) / sqrtc + sign1 * (r0p - r1p) / sqrta
        slm = (r1m * fl + ra1m) * tlm + r0m * fl * tlm_prev - (r1m + r0m) / sqrtc + sign1 * (r0m - r1m) / sqrta
        slpm = slp + slm
        for nabs in range(0, nf + 1):
            zv = float(nabs) * zeta
            cosv = np.cos(zv)
            sinv = np.sin(zv)
            for m in range(0, mf + 1):
                cm = float(cmns[l, m, nabs])
                if cm == 0.0:
                    continue
                mu = float(m) * theta
                sinu = np.sin(mu)
                cosu = np.cos(mu)
                col_p = nabs + nf
                col_m = (-nabs) + nf
                if nabs == 0 or m == 0:
                    sinp = (sinu * cosv - sinv * cosu) * cm
                    bsin[m, col_p] += np.sum(tlpm * bex * sinp)
                    gsin[m, col_p, :] += slpm * sinp
                    if lasym and bcos is not None:
                        cosp = (cosu * cosv + sinv * sinu) * cm
                        bcos[m, col_p] += np.sum(tlpm * bex * cosp)
                        if gcos is not None:
                            gcos[m, col_p, :] += slpm * cosp
                else:
                    sinp0 = sinu * cosv * cm
                    temp = -cosu * sinv * cm
                    sinm = sinp0 - temp
                    sinp = sinp0 + temp
                    # VMEC analyt.f calls analysesum2 with swapped argument
                    # order: (slm, tlm, slp, tlp). Preserve this Fortran quirk
                    # for exact matrix-side parity.
                    bsin[m, col_p] += np.sum(tlm * bex * sinp)
                    bsin[m, col_m] += np.sum(tlp * bex * sinm)
                    gsin[m, col_p, :] += slm * sinp
                    gsin[m, col_m, :] += slp * sinm
                    if lasym and bcos is not None:
                        cosp0 = cosu * cosv * cm
                        temp2 = sinu * sinv * cm
                        cosm = cosp0 - temp2
                        cosp = cosp0 + temp2
                        bcos[m, col_p] += np.sum(tlm * bex * cosp)
                        bcos[m, col_m] += np.sum(tlp * bex * cosm)
                        if gcos is not None:
                            gcos[m, col_p, :] += slm * cosp
                            gcos[m, col_m, :] += slp * cosm

        fl1 = fl1 + 1.0
        fl2 = 2.0 * fl1 - 1.0
        sign1 = -sign1

        tlp_next = ((sqrtc + sign1 * sqrta) - fl2 * cma * tlp - fl * adm * tlp_prev) / (adp * fl1)
        tlm_next = ((sqrtc + sign1 * sqrta) - fl2 * cma * tlm - fl * adp * tlm_prev) / (adm * fl1)
        tlp_prev = tlp
        tlm_prev = tlm
        tlp = tlp_next
        tlm = tlm_next
        tlpm = tlp + tlm

    out_s = np.zeros((mnpd,), dtype=float)
    out_c = np.zeros((mnpd,), dtype=float) if lasym else None
    gr_s = np.zeros((mnpd, npts), dtype=float)
    gr_c = np.zeros((mnpd, npts), dtype=float) if lasym else None
    xmpot = np.asarray(basis["xmpot"], dtype=np.int64)
    n_raw = np.asarray(basis["n_raw"], dtype=np.int64)
    for j in range(mnpd):
        m = int(xmpot[j])
        n = int(n_raw[j])
        out_s[j] = bsin[m, n + nf]
        gr_s[j, :] = gsin[m, n + nf, :]
        if lasym and out_c is not None:
            out_c[j] = bcos[m, n + nf]
            if gr_c is not None and gcos is not None:
                gr_c[j, :] = gcos[m, n + nf, :]
    if lasym and out_c is not None and gr_c is not None:
        return np.concatenate([out_s, out_c], axis=0), np.concatenate([gr_s, gr_c], axis=0)
    return out_s, gr_s


def vmec_analytic_bvec_from_geometry(
    *,
    sample: ExternalBoundarySample,
    basis: dict[str, Any],
    bexni: np.ndarray,
    signgs: int,
) -> np.ndarray:
    """Analytic-source bvec term from VMEC ``analyt.f`` (bvec branch)."""

    bvec, _ = vmec_analytic_terms_from_geometry(sample=sample, basis=basis, bexni=bexni, signgs=signgs)
    return bvec


def solve_vmec_like_mode_from_gsource(
    *,
    cache: NestorVmecLikeCache,
    gsource: np.ndarray,
    rhs_mode: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve VMEC-like dense integral in mode space and return (phi, potvac)."""

    basis = cache.mode_basis
    amod = cache.mode_matrix
    if basis is None or amod is None:
        raise ValueError("missing_mode_cache")

    rhs_eff = np.asarray(rhs_mode, dtype=float) if rhs_mode is not None else vmec_bvec_from_gsource(gsource=gsource, basis=basis)
    potvac = dense_lu_solve(cache.mode_matrix_lu, np.asarray(amod, dtype=float), np.asarray(rhs_eff, dtype=float))

    sin_phase = np.asarray(basis["sin_phase"], dtype=float)
    cos_phase = np.asarray(basis["cos_phase"], dtype=float)
    mnpd = int(basis["mnpd"])
    if bool(basis["lasym"]):
        potsin = np.asarray(potvac[:mnpd], dtype=float)
        potcos = np.asarray(potvac[mnpd : 2 * mnpd], dtype=float)
        phi_flat = sin_phase @ potsin + cos_phase @ potcos
    else:
        potsin = np.asarray(potvac[:mnpd], dtype=float)
        phi_flat = sin_phase @ potsin
    phi = phi_flat.reshape(int(cache.ntheta), int(cache.nzeta))
    phi = phi - float(np.mean(phi))
    return np.asarray(phi, dtype=float), np.asarray(potvac, dtype=float), np.asarray(rhs_eff, dtype=float)

def env_truthy(name: str, default: bool = False) -> bool:
    """Evaluate env truthy for direct-coil free-boundary solve and branch-local adjoint validation."""
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() not in ("", "0", "false", "no")


def digest_array_for_cache(value: Any) -> tuple[tuple[int, ...], str, str]:
    """Evaluate digest array for cache for direct-coil free-boundary solve and branch-local adjoint validation."""
    arr = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.blake2b(arr.view(np.uint8), digest_size=16).hexdigest()
    return tuple(int(i) for i in arr.shape), str(arr.dtype), digest


def mapping_cache_signature(mapping: dict[str, Any], keys: tuple[str, ...] | None = None) -> tuple[Any, ...]:
    """Evaluate mapping cache signature for direct-coil free-boundary solve and branch-local adjoint validation."""
    selected = tuple(sorted(mapping)) if keys is None else tuple(key for key in keys if key in mapping)
    signature: list[Any] = []
    for key in selected:
        value = mapping[key]
        if isinstance(value, dict):
            continue
        signature.append((key, digest_array_for_cache(value)))
    return tuple(signature)


def compact_jax_nestor_basis(basis: dict[str, Any]) -> dict[str, Any]:
    """Evaluate compact jax nestor basis for direct-coil free-boundary solve and branch-local adjoint validation."""
    return {key: basis[key] for key in JAX_NESTOR_BASIS_KEYS if key in basis}


def jax_nestor_operator_cache_key(
    *,
    basis: dict[str, Any],
    tables: dict[str, Any],
    signgs: int,
    nvper: int,
    include_analytic: bool,
    symmetric: bool,
    input_signature: tuple[Any, ...] = (),
) -> tuple[Any, ...]:
    """Evaluate jax nestor operator cache key for direct-coil free-boundary solve and branch-local adjoint validation."""
    return (
        int(signgs),
        int(nvper),
        bool(include_analytic),
        bool(symmetric),
        tuple(input_signature),
        mapping_cache_signature(basis, JAX_NESTOR_BASIS_KEYS),
        mapping_cache_signature(tables),
    )


def jax_nestor_input_signature(args: tuple[Any, ...]) -> tuple[Any, ...]:
    """Evaluate jax nestor input signature for direct-coil free-boundary solve and branch-local adjoint validation."""
    return tuple((tuple(int(i) for i in np.asarray(arg).shape), str(np.asarray(arg).dtype)) for arg in args)


def jitted_jax_nestor_operator(
    *,
    basis: dict[str, Any],
    tables: dict[str, Any],
    signgs: int,
    nvper: int,
    include_analytic: bool,
    symmetric: bool = False,
    example_args: tuple[Any, ...] = (),
) -> tuple[Any | None, bool]:
    """Return a cached compiled dense JAX NESTOR operator closure.

    The closure bakes mode-basis and kernel-table arrays as static constants so
    the active free-boundary update does not execute the JAX operator as many
    small eager dispatches. This cache is intentionally used only by the opt-in
    research path selected with ``VMEC_JAX_FREEB_JAX_NESTOR_OPERATOR=1``.
    """

    try:
        from ..._compat import jax as _jax
        from .adjoint.facade import dense_vmec_nestor_mode_solve_jax
    except Exception:
        return None, False
    if _jax is None:
        return None, False
    if bool(getattr(_jax.config, "jax_disable_jit", False)):
        return None, False

    key = jax_nestor_operator_cache_key(
        basis=basis,
        tables=tables,
        signgs=int(signgs),
        nvper=int(nvper),
        include_analytic=bool(include_analytic),
        symmetric=bool(symmetric),
        input_signature=jax_nestor_input_signature(tuple(example_args)),
    )
    cached = FREEB_JAX_NESTOR_OPERATOR_FN_CACHE.get(key)
    if cached is not None:
        return cached, True

    if len(FREEB_JAX_NESTOR_OPERATOR_FN_CACHE) >= 32:
        FREEB_JAX_NESTOR_OPERATOR_FN_CACHE.clear()

    basis_static = compact_jax_nestor_basis(basis)
    tables_static = {key: tables[key] for key in sorted(tables)}

    def _compiled(
        R: Any,
        Z: Any,
        Ru: Any,
        Zu: Any,
        Rv: Any,
        Zv: Any,
        ruu: Any,
        ruv: Any,
        rvv: Any,
        zuu: Any,
        zuv: Any,
        zvv: Any,
        bexni: Any,
    ) -> dict[str, Any]:
        return dense_vmec_nestor_mode_solve_jax(
            R=R,
            Z=Z,
            Ru=Ru,
            Zu=Zu,
            Rv=Rv,
            Zv=Zv,
            ruu=ruu,
            ruv=ruv,
            rvv=rvv,
            zuu=zuu,
            zuv=zuv,
            zvv=zvv,
            bexni=bexni,
            basis=basis_static,
            tables=tables_static,
            signgs=int(signgs),
            nvper=int(nvper),
            include_analytic=bool(include_analytic),
            symmetric=bool(symmetric),
        )

    jitted = _jax.jit(_compiled)
    compiled = jitted.lower(*example_args).compile() if example_args else jitted
    FREEB_JAX_NESTOR_OPERATOR_FN_CACHE[key] = compiled
    return compiled, False


def jax_nestor_operator_guard(
    *,
    sample: Any,
    basis: dict[str, Any] | None,
) -> tuple[bool, str]:
    """Return whether the experimental JAX VMEC/NESTOR operator can run safely."""

    if basis is None:
        return False, "missing_mode_basis"
    try:
        from ..._compat import has_jax, x64_enabled

        if not has_jax():
            return False, "jax_unavailable"
        if not x64_enabled():
            return False, "jax_x64_disabled"
    except Exception:
        return False, "jax_unavailable"
    if sample.R.ndim != 2:
        return False, "sample_R_not_2d"
    if int(sample.R.size) != int(basis.get("nuv3", sample.R.size)):
        return False, "requires_active_vmec_grid_points"
    if bool(basis.get("lasym", False)) and int(sample.R.size) != int(basis.get("nuv_full", sample.R.size)):
        return False, "requires_lasym_full_vmec_grid_points"
    if int(sample.R.shape[0]) > int(basis.get("nu_full", sample.R.shape[0])):
        return False, "active_grid_exceeds_full_grid"
    for name in ("Z", "Ru", "Zu", "Rv", "Zv"):
        arr = np.asarray(getattr(sample, name), dtype=float)
        if arr.shape != sample.R.shape:
            return False, f"{name}_shape_mismatch"
    for name in ("ruu", "ruv", "rvv", "zuu", "zuv", "zvv"):
        arr = getattr(sample, name)
        if arr is None:
            return False, f"missing_{name}"
        if np.asarray(arr).shape != sample.R.shape:
            return False, f"{name}_shape_mismatch"
    return True, "enabled"


def solve_vmec_like_mode_with_jax_nestor_operator(
    *,
    sample: Any,
    basis: dict[str, Any],
    tables: dict[str, Any],
    bexni: np.ndarray,
    signgs: int,
    nvper: int,
    include_analytic: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, bool]:
    """Run the experimental dense JAX VMEC/NESTOR mode operator."""

    from .adjoint.facade import dense_vmec_nestor_mode_solve_jax

    R = np.asarray(sample.R, dtype=float)
    Z = np.asarray(sample.Z, dtype=float)
    Ru = np.asarray(sample.Ru, dtype=float)
    Zu = np.asarray(sample.Zu, dtype=float)
    Rv = np.asarray(sample.Rv, dtype=float)
    Zv = np.asarray(sample.Zv, dtype=float)
    ruu = np.asarray(sample.ruu, dtype=float)
    ruv = np.asarray(sample.ruv, dtype=float)
    rvv = np.asarray(sample.rvv, dtype=float)
    zuu = np.asarray(sample.zuu, dtype=float)
    zuv = np.asarray(sample.zuv, dtype=float)
    zvv = np.asarray(sample.zvv, dtype=float)
    bexni_arr = np.asarray(bexni, dtype=float)
    operator_args = (R, Z, Ru, Zu, Rv, Zv, ruu, ruv, rvv, zuu, zuv, zvv, bexni_arr)
    compiled = None
    cache_hit = False
    if env_truthy("VMEC_JAX_FREEB_JAX_NESTOR_JIT_OPERATOR", True):
        compiled, cache_hit = jitted_jax_nestor_operator(
            basis=basis,
            tables=tables,
            signgs=int(signgs),
            nvper=max(1, int(nvper)),
            include_analytic=bool(include_analytic),
            example_args=operator_args,
        )
    if compiled is None:
        out = dense_vmec_nestor_mode_solve_jax(
            R=R,
            Z=Z,
            Ru=Ru,
            Zu=Zu,
            Rv=Rv,
            Zv=Zv,
            ruu=ruu,
            ruv=ruv,
            rvv=rvv,
            zuu=zuu,
            zuv=zuv,
            zvv=zvv,
            bexni=bexni_arr,
            basis=basis,
            tables=tables,
            signgs=int(signgs),
            nvper=max(1, int(nvper)),
            include_analytic=bool(include_analytic),
        )
        jit_used = False
    else:
        out = compiled(*operator_args)
        jit_used = True
    potvac = np.asarray(out["mode_coeffs"], dtype=float)
    rhs_mode = np.asarray(out["rhs_mode"], dtype=float)
    mode_matrix = np.asarray(out["mode_matrix"], dtype=float)
    grpmn = np.asarray(out["grpmn"], dtype=float)
    gsource_nonsing = np.asarray(out["gsource_nonsing"], dtype=float)
    mnpd2 = int(basis["mnpd2"])
    if mode_matrix.shape != (mnpd2, mnpd2):
        raise ValueError("jax_nestor_mode_matrix_shape")
    if rhs_mode.shape != (mnpd2,) or potvac.shape != (mnpd2,):
        raise ValueError("jax_nestor_mode_vector_shape")
    for name, arr in (
        ("rhs_mode", rhs_mode),
        ("mode_matrix", mode_matrix),
        ("mode_coeffs", potvac),
        ("grpmn", grpmn),
        ("gsource_nonsing", gsource_nonsing),
    ):
        if not np.isfinite(arr).all():
            raise ValueError(f"jax_nestor_nonfinite_{name}")
    residual = mode_matrix @ potvac - rhs_mode
    residual_tol = 1.0e-8 * (1.0 + float(np.linalg.norm(rhs_mode)))
    if float(np.linalg.norm(residual)) > residual_tol:
        raise ValueError("jax_nestor_linear_residual")
    mnpd = int(basis["mnpd"])
    sin_phase = np.asarray(basis["sin_phase"], dtype=float)
    cos_phase = np.asarray(basis["cos_phase"], dtype=float)
    if bool(basis["lasym"]) and potvac.size >= 2 * mnpd:
        phi_flat = sin_phase @ potvac[:mnpd] + cos_phase @ potvac[mnpd : 2 * mnpd]
    else:
        phi_flat = sin_phase @ potvac[:mnpd]
    phi = np.asarray(phi_flat, dtype=float).reshape(np.asarray(sample.R).shape)
    phi = phi - float(np.mean(phi))
    return (
        phi,
        potvac,
        rhs_mode,
        mode_matrix,
        grpmn,
        gsource_nonsing,
        jit_used,
        cache_hit,
    )


__all__ = [
    "FREEB_JAX_NESTOR_OPERATOR_FN_CACHE",
    "JAX_NESTOR_BASIS_KEYS",
    "build_poisson_cache",
    "build_vmec_cmns",
    "build_vmec_like_cache",
    "build_vmec_mode_basis",
    "compact_jax_nestor_basis",
    "dense_lu_factor",
    "dense_lu_solve",
    "digest_array_for_cache",
    "ensure_vmec_nonsingular_kernel_tables",
    "env_truthy",
    "jax_nestor_input_signature",
    "jax_nestor_operator_cache_key",
    "jax_nestor_operator_guard",
    "jitted_jax_nestor_operator",
    "mapping_cache_signature",
    "solve_vmec_like_dense",
    "solve_vmec_like_mode_from_gsource",
    "solve_vmec_like_mode_with_jax_nestor_operator",
    "spectral_second_derivatives_2d",
    "vmec_analytic_bvec_from_geometry",
    "vmec_analytic_terms_from_geometry",
    "vmec_bvec_from_gsource",
    "vmec_mode_matrix_from_grpmn",
    "vmec_nonsingular_gsource_from_bexni",
    "vmec_nonsingular_terms_from_bexni",
    "vmec_precal_tan_tables",
    "vmec_source_from_gsource",
]
