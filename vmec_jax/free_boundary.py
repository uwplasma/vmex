"""Free-boundary typed config/state and mgrid loader skeleton.

WP0 scope:
- typed runtime state container for free-boundary iteration control,
- deterministic mgrid metadata/field loader,
- lightweight validation hooks that mirror VMEC2000 readin/read_indata behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from .config import VMECConfig

_MGRID_FIELD_CACHE: dict[str, MGridData] = {}


@dataclass(frozen=True)
class FreeBoundaryRuntimeState:
    """Runtime controls used by the free-boundary branch.

    These map directly to VMEC2000 control variables and are intentionally
    scalar-only for clean JAX/static integration later.
    """

    ivac: int
    ivacskip: int
    nvacskip: int
    nvskip0: int


@dataclass(frozen=True)
class MGridMetadata:
    path: str
    ir: int
    jz: int
    kp: int
    nfp: int
    nextcur: int
    rmin: float
    rmax: float
    zmin: float
    zmax: float
    mgrid_mode: str
    coil_groups: tuple[str, ...]
    raw_coil_cur: tuple[float, ...]


@dataclass(frozen=True)
class MGridData:
    metadata: MGridMetadata
    br: np.ndarray
    bp: np.ndarray
    bz: np.ndarray


@dataclass(frozen=True)
class VacuumBoundaryFields:
    """Boundary vacuum field channels on the VMEC angular grid.

    Arrays are defined on a single boundary surface with shape `(ntheta, nzeta)`.
    """

    bu: np.ndarray
    bv: np.ndarray
    bsupu: np.ndarray
    bsupv: np.ndarray
    bsqvac: np.ndarray
    bnormal: np.ndarray
    bnormal_unit: np.ndarray
    g_uu: np.ndarray
    g_uv: np.ndarray
    g_vv: np.ndarray
    det_guv: np.ndarray


@dataclass(frozen=True)
class ExternalBoundarySample:
    """External-field sample on the plasma boundary."""

    mgrid_path: str
    R: np.ndarray
    Z: np.ndarray
    Ru: np.ndarray
    Zu: np.ndarray
    Rv: np.ndarray
    Zv: np.ndarray
    phi: np.ndarray
    br: np.ndarray
    bp: np.ndarray
    bz: np.ndarray
    vac_ext: VacuumBoundaryFields


@dataclass(frozen=True)
class NestorPoissonCache:
    """Stage-static spectral Poisson operator cache on the (theta,zeta) torus."""

    ntheta: int
    nzeta: int
    lam: np.ndarray


@dataclass(frozen=True)
class NestorVmecLikeCache:
    """VMEC2000-like dense boundary-integral operator cache."""

    ntheta: int
    nzeta: int
    matrix: np.ndarray
    rhs_scale: np.ndarray


@dataclass(frozen=True)
class NestorRuntimeState:
    """Runtime cache for the NESTOR solve path."""

    operator_cache: Any
    phi: np.ndarray
    bsqvac: np.ndarray
    mode: str
    update_count: int
    reuse_count: int


@dataclass(frozen=True)
class NestorSolveResult:
    """Output of one NESTOR-like update/reuse step."""

    vac_total: VacuumBoundaryFields
    phi: np.ndarray
    reused: bool
    solve_time_s: float
    sample_time_s: float
    model: str = "spectral_poisson_external_only"


@dataclass(frozen=True)
class PreparedMGrid:
    """Validated mgrid metadata plus normalized external-current vector."""

    metadata: MGridMetadata
    extcur: tuple[float, ...]


def initial_free_boundary_state(cfg: VMECConfig) -> FreeBoundaryRuntimeState:
    """Initialize free-boundary control state for a VMEC stage."""

    nv = int(cfg.nvacskip)
    return FreeBoundaryRuntimeState(
        ivac=0,
        ivacskip=0,
        nvacskip=nv,
        nvskip0=max(1, nv),
    )


def validate_free_boundary_config(cfg: VMECConfig, *, strict: bool = False) -> None:
    """Validate parsed free-boundary inputs.

    In strict mode, raise if `LFREEB=T` but no usable mgrid path is given.
    """

    if not bool(cfg.lfreeb):
        return
    mg = str(cfg.mgrid_file).strip()
    if (not mg) or mg.upper() == "NONE":
        if strict:
            raise ValueError("LFREEB=T requires MGRID_FILE (not NONE)")
        return
    if int(cfg.nvacskip) <= 0:
        raise ValueError(f"nvacskip must be >=1 after normalization, got {cfg.nvacskip}")


def _normalize_extcur(extcur_in: tuple[float, ...], nextcur: int) -> tuple[float, ...]:
    if nextcur <= 0:
        return tuple()
    vals = list(float(v) for v in extcur_in)
    if len(vals) < nextcur:
        vals.extend([0.0] * (nextcur - len(vals)))
    elif len(vals) > nextcur:
        vals = vals[:nextcur]
    return tuple(vals)


def _broadcast_xyz(r: Any, z: Any, phi: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rr, zz, pp = np.broadcast_arrays(np.asarray(r, dtype=float), np.asarray(z, dtype=float), np.asarray(phi, dtype=float))
    return rr, zz, pp


def interpolate_mgrid_bfield(
    data: MGridData,
    *,
    r: Any,
    z: Any,
    phi: Any,
    extcur: tuple[float, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trilinear interpolation of mgrid BR/BP/BZ with periodic toroidal angle.

    Parameters
    ----------
    data:
        Loaded mgrid tensor data (`load_mgrid(..., load_fields=True)`).
    r, z, phi:
        Query coordinates (broadcastable arrays).
    extcur:
        External current weights per coil group. If omitted, uses `raw_coil_cur`
        when available; otherwise unit weights.
    """

    meta = data.metadata
    rr, zz, pp = _broadcast_xyz(r, z, phi)
    out_shape = rr.shape
    n = int(rr.size)

    ir = int(meta.ir)
    jz = int(meta.jz)
    kp = int(meta.kp)
    if ir < 2 or jz < 2 or kp < 2:
        raise ValueError(f"mgrid dimensions too small for interpolation: ir={ir} jz={jz} kp={kp}")

    # Non-periodic axes (R, Z): clamp to domain.
    rmin = float(meta.rmin)
    rmax = float(meta.rmax)
    zmin = float(meta.zmin)
    zmax = float(meta.zmax)
    if rmax <= rmin or zmax <= zmin:
        raise ValueError("Invalid mgrid bounds: require rmax>rmin and zmax>zmin")

    r_flat = np.clip(rr.reshape(-1), rmin, rmax)
    z_flat = np.clip(zz.reshape(-1), zmin, zmax)

    fr = (r_flat - rmin) * ((ir - 1) / (rmax - rmin))
    fz = (z_flat - zmin) * ((jz - 1) / (zmax - zmin))

    i0 = np.floor(fr).astype(np.int64)
    j0 = np.floor(fz).astype(np.int64)
    i0 = np.clip(i0, 0, ir - 2)
    j0 = np.clip(j0, 0, jz - 2)
    i1 = i0 + 1
    j1 = j0 + 1
    wr = fr - i0
    wz = fz - j0

    # Periodic toroidal axis: one field period [0, 2*pi/nfp).
    nfp = max(1, int(meta.nfp))
    period = (2.0 * np.pi) / float(nfp)
    phi_flat = np.mod(pp.reshape(-1), period)
    fk = phi_flat * (kp / period)
    k0 = np.floor(fk).astype(np.int64) % kp
    k1 = (k0 + 1) % kp
    wk = fk - np.floor(fk)

    w0r = 1.0 - wr
    w0z = 1.0 - wz
    w0k = 1.0 - wk

    cur = _normalize_extcur(
        tuple(extcur) if extcur is not None else tuple(meta.raw_coil_cur),
        int(meta.nextcur),
    )
    if len(cur) == 0:
        cur = tuple(1.0 for _ in range(int(meta.nextcur)))
    cur_vec = np.asarray(cur, dtype=float).reshape((-1, 1))

    def _interp(field: np.ndarray) -> np.ndarray:
        f = np.asarray(field, dtype=float)
        if f.shape != (int(meta.nextcur), kp, jz, ir):
            raise ValueError(
                f"field shape {f.shape} != expected {(int(meta.nextcur), kp, jz, ir)}"
            )
        v000 = f[:, k0, j0, i0]
        v001 = f[:, k0, j0, i1]
        v010 = f[:, k0, j1, i0]
        v011 = f[:, k0, j1, i1]
        v100 = f[:, k1, j0, i0]
        v101 = f[:, k1, j0, i1]
        v110 = f[:, k1, j1, i0]
        v111 = f[:, k1, j1, i1]

        c00 = v000 * w0r + v001 * wr
        c01 = v010 * w0r + v011 * wr
        c10 = v100 * w0r + v101 * wr
        c11 = v110 * w0r + v111 * wr
        c0 = c00 * w0z + c01 * wz
        c1 = c10 * w0z + c11 * wz
        c = c0 * w0k + c1 * wk
        return np.sum(cur_vec * c, axis=0).reshape(out_shape)

    br = _interp(data.br)
    bp = _interp(data.bp)
    bz = _interp(data.bz)
    return br, bp, bz


def boundary_metric_from_rz(
    *,
    R: Any,
    Ru: Any,
    Zu: Any,
    Rv: Any,
    Zv: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute 2D boundary metric terms for `(u=theta, v=zeta_phys)` coordinates."""

    R = np.asarray(R, dtype=float)
    Ru = np.asarray(Ru, dtype=float)
    Zu = np.asarray(Zu, dtype=float)
    Rv = np.asarray(Rv, dtype=float)
    Zv = np.asarray(Zv, dtype=float)
    g_uu = Ru * Ru + Zu * Zu
    g_uv = Ru * Rv + Zu * Zv
    g_vv = R * R + Rv * Rv + Zv * Zv
    det = g_uu * g_vv - g_uv * g_uv
    return g_uu, g_uv, g_vv, det


def covariant_boundary_field_from_cylindrical(
    *,
    br: Any,
    bp: Any,
    bz: Any,
    R: Any,
    Ru: Any,
    Zu: Any,
    Rv: Any,
    Zv: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Project cylindrical field `(Br,Bphi,Bz)` to boundary covariant components.

    Uses VMEC boundary tangent vectors:
    - ``x_u = (Ru, 0, Zu)``
    - ``x_v = (Rv, R, Zv)``
    in cylindrical orthonormal basis `(e_R, e_phi, e_Z)`.
    """

    br = np.asarray(br, dtype=float)
    bp = np.asarray(bp, dtype=float)
    bz = np.asarray(bz, dtype=float)
    R = np.asarray(R, dtype=float)
    Ru = np.asarray(Ru, dtype=float)
    Zu = np.asarray(Zu, dtype=float)
    Rv = np.asarray(Rv, dtype=float)
    Zv = np.asarray(Zv, dtype=float)
    bu = br * Ru + bz * Zu
    bv = br * Rv + bp * R + bz * Zv
    return bu, bv


def contravariant_boundary_field_from_covariant(
    *,
    bu: Any,
    bv: Any,
    g_uu: Any,
    g_uv: Any,
    g_vv: Any,
    det_floor: float = 1.0e-30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute `(B^u,B^v)` by inverting the 2x2 surface metric.

    A signed determinant floor is used to avoid non-finite divisions in
    degenerate cells.
    """

    bu = np.asarray(bu, dtype=float)
    bv = np.asarray(bv, dtype=float)
    g_uu = np.asarray(g_uu, dtype=float)
    g_uv = np.asarray(g_uv, dtype=float)
    g_vv = np.asarray(g_vv, dtype=float)
    det = g_uu * g_vv - g_uv * g_uv
    det_safe = np.where(np.abs(det) >= float(det_floor), det, np.sign(det + 1.0e-300) * float(det_floor))
    bsupu = (g_vv * bu - g_uv * bv) / det_safe
    bsupv = (g_uu * bv - g_uv * bu) / det_safe
    return bsupu, bsupv, det


def vacuum_boundary_fields_from_cylindrical(
    *,
    br: Any,
    bp: Any,
    bz: Any,
    R: Any,
    Ru: Any,
    Zu: Any,
    Rv: Any,
    Zv: Any,
    det_floor: float = 1.0e-30,
) -> VacuumBoundaryFields:
    """Compute VMEC-like vacuum boundary channels from cylindrical field input."""

    R = np.asarray(R, dtype=float)
    Ru = np.asarray(Ru, dtype=float)
    Zu = np.asarray(Zu, dtype=float)
    Rv = np.asarray(Rv, dtype=float)
    Zv = np.asarray(Zv, dtype=float)
    br = np.asarray(br, dtype=float)
    bp = np.asarray(bp, dtype=float)
    bz = np.asarray(bz, dtype=float)

    g_uu, g_uv, g_vv, _ = boundary_metric_from_rz(R=R, Ru=Ru, Zu=Zu, Rv=Rv, Zv=Zv)
    bu, bv = covariant_boundary_field_from_cylindrical(
        br=br,
        bp=bp,
        bz=bz,
        R=R,
        Ru=Ru,
        Zu=Zu,
        Rv=Rv,
        Zv=Zv,
    )
    bsupu, bsupv, det = contravariant_boundary_field_from_covariant(
        bu=bu,
        bv=bv,
        g_uu=g_uu,
        g_uv=g_uv,
        g_vv=g_vv,
        det_floor=det_floor,
    )
    bsqvac = bu * bsupu + bv * bsupv

    # Non-unit boundary normal from x_u x x_v in cylindrical components.
    n_r = -R * Zu
    n_phi = Zu * Rv - Ru * Zv
    n_z = R * Ru
    bnormal = br * n_r + bp * n_phi + bz * n_z
    n_norm = np.sqrt(n_r * n_r + n_phi * n_phi + n_z * n_z)
    n_norm_safe = np.where(n_norm > 0.0, n_norm, 1.0)
    bnormal_unit = bnormal / n_norm_safe

    return VacuumBoundaryFields(
        bu=bu,
        bv=bv,
        bsupu=bsupu,
        bsupv=bsupv,
        bsqvac=bsqvac,
        bnormal=bnormal,
        bnormal_unit=bnormal_unit,
        g_uu=g_uu,
        g_uv=g_uv,
        g_vv=g_vv,
        det_guv=det,
    )


def _sample_external_boundary_arrays(
    *,
    state: Any,
    static: Any,
    extcur: tuple[float, ...] | None = None,
) -> ExternalBoundarySample:
    """Return full boundary arrays for external mgrid field sampling."""

    from .vmec_realspace import (
        vmec_realspace_synthesis,
        vmec_realspace_synthesis_dtheta,
        vmec_realspace_synthesis_dzeta_phys,
    )
    from .vmec_tomnsp import vmec_trig_tables

    meta = getattr(static, "mgrid_metadata", None)
    if meta is None:
        raise ValueError("missing_mgrid_metadata")
    mgrid_path = str(getattr(meta, "path", "")).strip()
    if not mgrid_path:
        raise ValueError("missing_mgrid_path")

    mgrid = _MGRID_FIELD_CACHE.get(mgrid_path)
    if mgrid is None:
        loaded = load_mgrid(mgrid_path, load_fields=True)
        if not isinstance(loaded, MGridData):  # pragma: no cover
            raise TypeError("load_mgrid(load_fields=True) must return MGridData")
        mgrid = loaded
        _MGRID_FIELD_CACHE[mgrid_path] = mgrid
    extcur_eff = tuple(extcur) if extcur is not None else tuple(getattr(static, "free_boundary_extcur", ()) or ())

    trig = getattr(static, "trig_vmec", None)
    if trig is None:
        trig = vmec_trig_tables(
            ntheta=int(static.cfg.ntheta),
            nzeta=int(static.cfg.nzeta),
            nfp=int(static.cfg.nfp),
            mmax=int(static.cfg.mpol) - 1,
            nmax=int(static.cfg.ntor),
            lasym=bool(static.cfg.lasym),
        )

    R = np.asarray(
        vmec_realspace_synthesis(
            coeff_cos=np.asarray(state.Rcos)[-1:, :],
            coeff_sin=np.asarray(state.Rsin)[-1:, :],
            modes=static.modes,
            trig=trig,
            coeffs_internal=False,
        )[0]
    )
    Z = np.asarray(
        vmec_realspace_synthesis(
            coeff_cos=np.asarray(state.Zcos)[-1:, :],
            coeff_sin=np.asarray(state.Zsin)[-1:, :],
            modes=static.modes,
            trig=trig,
            coeffs_internal=False,
        )[0]
    )
    Ru = np.asarray(
        vmec_realspace_synthesis_dtheta(
            coeff_cos=np.asarray(state.Rcos)[-1:, :],
            coeff_sin=np.asarray(state.Rsin)[-1:, :],
            modes=static.modes,
            trig=trig,
            coeffs_internal=False,
        )[0]
    )
    Zu = np.asarray(
        vmec_realspace_synthesis_dtheta(
            coeff_cos=np.asarray(state.Zcos)[-1:, :],
            coeff_sin=np.asarray(state.Zsin)[-1:, :],
            modes=static.modes,
            trig=trig,
            coeffs_internal=False,
        )[0]
    )
    Rv = np.asarray(
        vmec_realspace_synthesis_dzeta_phys(
            coeff_cos=np.asarray(state.Rcos)[-1:, :],
            coeff_sin=np.asarray(state.Rsin)[-1:, :],
            modes=static.modes,
            trig=trig,
            coeffs_internal=False,
        )[0]
    )
    Zv = np.asarray(
        vmec_realspace_synthesis_dzeta_phys(
            coeff_cos=np.asarray(state.Zcos)[-1:, :],
            coeff_sin=np.asarray(state.Zsin)[-1:, :],
            modes=static.modes,
            trig=trig,
            coeffs_internal=False,
        )[0]
    )

    nzeta = int(R.shape[1])
    zeta = (2.0 * np.pi / max(1, nzeta)) * np.arange(nzeta, dtype=float)
    phi = zeta / max(1, int(static.cfg.nfp))
    phi_grid = np.broadcast_to(phi[None, :], R.shape)

    br, bp, bz = interpolate_mgrid_bfield(
        mgrid,
        r=R,
        z=Z,
        phi=phi_grid,
        extcur=extcur_eff,
    )
    vac = vacuum_boundary_fields_from_cylindrical(
        br=br,
        bp=bp,
        bz=bz,
        R=R,
        Ru=Ru,
        Zu=Zu,
        Rv=Rv,
        Zv=Zv,
    )
    return ExternalBoundarySample(
        mgrid_path=mgrid_path,
        R=R,
        Z=Z,
        Ru=Ru,
        Zu=Zu,
        Rv=Rv,
        Zv=Zv,
        phi=phi_grid,
        br=br,
        bp=bp,
        bz=bz,
        vac_ext=vac,
    )


def _build_poisson_cache(*, ntheta: int, nzeta: int) -> NestorPoissonCache:
    """Build spectral Laplacian eigenvalues on periodic `(theta,zeta)` grid."""

    ntheta = int(ntheta)
    nzeta = int(nzeta)
    ku = 2.0 * np.pi * np.fft.fftfreq(ntheta)
    kv = 2.0 * np.pi * np.fft.fftfreq(nzeta)
    ku2 = ku[:, None] * ku[:, None]
    kv2 = kv[None, :] * kv[None, :]
    lam = ku2 + kv2
    lam[0, 0] = 1.0
    return NestorPoissonCache(ntheta=ntheta, nzeta=nzeta, lam=lam)


def _build_vmec_like_cache(
    sample: ExternalBoundarySample,
    *,
    alpha: float,
    dist_eps: float,
    rhs_floor: float,
    diag_coeff: float,
    row_sum_zero: bool,
    singular_diag_scale: float,
) -> NestorVmecLikeCache:
    """Build a dense boundary-integral-like operator on the VMEC angular grid.

    This mirrors NESTOR's matrix-assembly style (dense Green-function operator
    over boundary samples), while remaining a compact NumPy implementation.
    """

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
    # Surface-area-like quadrature weights; normalization keeps conditioning sane.
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
        # Principal-value-inspired stabilization: remove each row mean contribution
        # from the diagonal so constant offsets do not spuriously dominate.
        row_sum = np.sum(kernel, axis=1)
        kernel[np.arange(npts), np.arange(npts)] -= row_sum

    diag_extra = np.zeros((npts,), dtype=float)
    if float(singular_diag_scale) != 0.0:
        # Local nearest-neighbor scale as a compact proxy for the singular
        # self-panel contribution in VMEC/NESTOR diagonal treatment.
        dist_nodiag = np.asarray(dist, dtype=float).copy()
        np.fill_diagonal(dist_nodiag, np.inf)
        h = np.minimum(np.min(dist_nodiag, axis=1), 1.0 / float(max(1, npts)))
        h = np.maximum(h, float(dist_eps))
        diag_extra = (float(singular_diag_scale) / (4.0 * np.pi)) * (w / h)

    matrix = float(alpha) * kernel
    matrix[np.arange(npts), np.arange(npts)] += float(diag_coeff) + diag_extra
    rhs_scale = np.where(w > rhs_floor, w, rhs_floor)
    return NestorVmecLikeCache(
        ntheta=ntheta,
        nzeta=nzeta,
        matrix=matrix,
        rhs_scale=rhs_scale,
    )


def _solve_vmec_like_dense(rhs: np.ndarray, cache: NestorVmecLikeCache) -> np.ndarray:
    rhs_flat = np.asarray(rhs, dtype=float).reshape(-1) * np.asarray(cache.rhs_scale, dtype=float)
    phi_flat = np.linalg.solve(np.asarray(cache.matrix, dtype=float), rhs_flat)
    phi = phi_flat.reshape(int(cache.ntheta), int(cache.nzeta))
    phi = phi - float(np.mean(phi))
    return phi


def _base_nestor_mode(mode: str) -> str:
    return str(mode).split("_fallback:", 1)[0]


def _is_dense_mode(mode: str) -> bool:
    return _base_nestor_mode(mode).startswith("vmec2000_like_dense_integral")


def _is_spectral_mode(mode: str) -> bool:
    return _base_nestor_mode(mode).startswith("spectral_poisson_external_only")


def _solve_periodic_poisson_fft(rhs: np.ndarray, cache: NestorPoissonCache) -> np.ndarray:
    rhs_hat = np.fft.fftn(rhs)
    phi_hat = -rhs_hat / cache.lam
    phi_hat[0, 0] = 0.0
    phi = np.fft.ifftn(phi_hat).real
    return phi


def _spectral_grad(phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ntheta, nzeta = phi.shape
    ku = 2.0 * np.pi * np.fft.fftfreq(ntheta)
    kv = 2.0 * np.pi * np.fft.fftfreq(nzeta)
    ph = np.fft.fftn(phi)
    du = np.fft.ifftn((1j * ku[:, None]) * ph).real
    dv = np.fft.ifftn((1j * kv[None, :]) * ph).real
    return du, dv


def _as_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if raw == "":
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _as_float_env(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if raw == "":
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _parse_iter_list_env(name: str) -> set[int]:
    raw = os.getenv(name, "").strip()
    if raw == "":
        return set()
    out: set[int] = set()
    for tok in raw.replace(";", ",").split(","):
        t = tok.strip()
        if not t:
            continue
        try:
            out.add(int(t))
        except Exception:
            continue
    return out


def _select_nestor_mode(*, ntheta: int, nzeta: int) -> tuple[str, str]:
    """Pick free-boundary NESTOR model mode with fallback-friendly semantics."""

    mode_raw = os.getenv("VMEC_JAX_FREEB_NESTOR_MODE", "auto").strip().lower()
    npts = int(ntheta * nzeta)
    max_pts = max(1, _as_int_env("VMEC_JAX_FREEB_VMEC_LIKE_MAX_POINTS", 4096))

    if mode_raw in ("spectral", "fast", "surrogate", "poisson"):
        return "spectral_poisson_external_only", "forced_fast"
    if mode_raw in ("vmec2000_like", "vmec_like", "vmec2000", "dense", "integral"):
        if npts > max_pts:
            return "spectral_poisson_external_only", f"fallback_max_points:{npts}>{max_pts}"
        return "vmec2000_like_dense_integral", "forced_vmec_like"
    # auto
    if npts <= max_pts:
        return "vmec2000_like_dense_integral", "auto_vmec_like"
    return "spectral_poisson_external_only", f"auto_fast_max_points:{npts}>{max_pts}"


def _vacuum_channels_from_sample_phi(sample: ExternalBoundarySample, phi: np.ndarray) -> VacuumBoundaryFields:
    dphi_u, dphi_v = _spectral_grad(phi)
    bu = np.asarray(sample.vac_ext.bu) + dphi_u
    bv = np.asarray(sample.vac_ext.bv) + dphi_v
    bsupu, bsupv, det = contravariant_boundary_field_from_covariant(
        bu=bu,
        bv=bv,
        g_uu=sample.vac_ext.g_uu,
        g_uv=sample.vac_ext.g_uv,
        g_vv=sample.vac_ext.g_vv,
    )
    bsqvac = bu * bsupu + bv * bsupv
    return VacuumBoundaryFields(
        bu=bu,
        bv=bv,
        bsupu=bsupu,
        bsupv=bsupv,
        bsqvac=bsqvac,
        bnormal=sample.vac_ext.bnormal,
        bnormal_unit=sample.vac_ext.bnormal_unit,
        g_uu=sample.vac_ext.g_uu,
        g_uv=sample.vac_ext.g_uv,
        g_vv=sample.vac_ext.g_vv,
        det_guv=det,
    )


def _maybe_dump_scalpot_jax(
    *,
    iter_idx: int | None,
    ivac: int,
    reused: bool,
    mode: str,
    rhs: np.ndarray,
    phi: np.ndarray,
    vac: VacuumBoundaryFields,
    cache: Any,
    sample: ExternalBoundarySample,
    mf: int,
    nf: int,
    nfp: int,
    lasym: bool,
) -> None:
    env = os.getenv("VMEC_JAX_DUMP_SCALPOT", "").strip().lower()
    if env in ("", "0", "false", "no"):
        return
    if iter_idx is None:
        return
    iters = _parse_iter_list_env("VMEC_JAX_DUMP_ITER")
    if iters and int(iter_idx) not in iters:
        return
    outdir = Path(os.getenv("VMEC_JAX_DUMP_DIR", ".")).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    out = {
        "iter": np.asarray(int(iter_idx), dtype=np.int64),
        "ivac": np.asarray(int(ivac), dtype=np.int64),
        "reused": np.asarray(1 if bool(reused) else 0, dtype=np.int64),
        "mode": np.asarray(str(mode)),
        "rhs": np.asarray(rhs, dtype=float),
        "phi": np.asarray(phi, dtype=float),
        "bu": np.asarray(vac.bu, dtype=float),
        "bv": np.asarray(vac.bv, dtype=float),
        "bsqvac": np.asarray(vac.bsqvac, dtype=float),
        "bnormal_unit": np.asarray(vac.bnormal_unit, dtype=float),
    }
    if isinstance(cache, NestorVmecLikeCache):
        out["cache_kind"] = np.asarray("dense")
        out["matrix"] = np.asarray(cache.matrix, dtype=float)
        out["rhs_scale"] = np.asarray(cache.rhs_scale, dtype=float)
        try:
            ntheta, nzeta = rhs.shape
            theta = (2.0 * np.pi / float(max(1, ntheta))) * np.arange(ntheta, dtype=float)
            zeta = np.asarray(sample.phi, dtype=float) * float(max(1, int(nfp)))
            if zeta.shape != rhs.shape:
                zeta = np.broadcast_to(zeta, rhs.shape)
            th = np.broadcast_to(theta[:, None], rhs.shape).reshape(-1)
            zz = zeta.reshape(-1)
            w = np.asarray(cache.rhs_scale, dtype=float).reshape(-1)
            rhs_f = np.asarray(rhs, dtype=float).reshape(-1)
            nmode = (int(mf) + 1) * (2 * int(nf) + 1)
            bsin = np.zeros((nmode,), dtype=float)
            bcos = np.zeros((nmode,), dtype=float)
            basis_s = np.zeros((rhs_f.size, nmode), dtype=float)
            basis_c = np.zeros((rhs_f.size, nmode), dtype=float)
            mn = 0
            for n in range(-int(nf), int(nf) + 1):
                for m in range(0, int(mf) + 1):
                    ph = (float(m) * th) - (float(n) * zz)
                    s = np.sin(ph)
                    c = np.cos(ph)
                    basis_s[:, mn] = s
                    basis_c[:, mn] = c
                    bsin[mn] = np.sum(w * rhs_f * s)
                    bcos[mn] = np.sum(w * rhs_f * c)
                    mn += 1
            if bool(lasym):
                B = np.concatenate([basis_s, basis_c], axis=1)
                out["bvec_mode_sin"] = bsin
                out["bvec_mode_cos"] = bcos
            else:
                B = basis_s
                out["bvec_mode_sin"] = bsin
            W = w[:, None]
            A = np.asarray(cache.matrix, dtype=float)
            amod = (B.T @ (W * (A @ B))) / float(max(1, rhs_f.size))
            out["amatrix_mode"] = amod
        except Exception:
            pass
    elif isinstance(cache, NestorPoissonCache):
        out["cache_kind"] = np.asarray("spectral")
        out["lam"] = np.asarray(cache.lam, dtype=float)
    else:
        out["cache_kind"] = np.asarray("unknown")
    fpath = outdir / f"scalpot_jax_iter{int(iter_idx)}.npz"
    np.savez_compressed(fpath, **out)


def nestor_external_only_step(
    *,
    state: Any,
    static: Any,
    ivac: int,
    iter_idx: int | None = None,
    runtime: NestorRuntimeState | None = None,
    extcur: tuple[float, ...] | None = None,
) -> tuple[NestorSolveResult, NestorRuntimeState]:
    """Simplified NESTOR-style update/reuse with ivacskip-compatible behavior.

    - `ivac==1`: full update (sample + spectral Poisson solve)
    - `ivac!=1`: reuse previous solution if available
    """

    runtime_cache = None if runtime is None else getattr(runtime, "operator_cache", None)
    runtime_mode = "spectral_poisson_external_only" if runtime is None else str(
        getattr(runtime, "mode", "spectral_poisson_external_only")
    )
    if runtime_cache is None and runtime is not None and hasattr(runtime, "poisson"):
        # Backward compatibility with older runtime state shape.
        runtime_cache = getattr(runtime, "poisson")
        if runtime_mode == "spectral_poisson_external_only":
            runtime_mode = "spectral_poisson_external_only"

    force_rhs_reuse = os.getenv("VMEC_JAX_FREEB_REUSE_RHS_UPDATE", "1").strip().lower() not in (
        "",
        "0",
        "false",
        "no",
    )

    if int(ivac) != 1 and runtime is not None and not force_rhs_reuse:
        # Legacy fast reuse mode: hold previous potential/bsqvac unchanged.
        bsqvac = np.asarray(runtime.bsqvac)
        z = np.zeros_like(bsqvac)
        vac_total = VacuumBoundaryFields(
            bu=z,
            bv=z,
            bsupu=z,
            bsupv=z,
            bsqvac=bsqvac,
            bnormal=z,
            bnormal_unit=z,
            g_uu=z,
            g_uv=z,
            g_vv=z,
            det_guv=z,
        )
        res = NestorSolveResult(
            vac_total=vac_total,
            phi=np.asarray(runtime.phi),
            reused=True,
            solve_time_s=0.0,
            sample_time_s=0.0,
            model=runtime_mode,
        )
        runtime_next = NestorRuntimeState(
            operator_cache=runtime_cache,
            phi=np.asarray(runtime.phi),
            bsqvac=np.asarray(bsqvac),
            mode=runtime_mode,
            update_count=int(runtime.update_count),
            reuse_count=int(runtime.reuse_count) + 1,
        )
        return res, runtime_next

    t0 = time.perf_counter()
    sample = _sample_external_boundary_arrays(state=state, static=static, extcur=extcur)
    sample_time = max(0.0, time.perf_counter() - t0)
    ntheta, nzeta = sample.R.shape
    selected_mode, mode_reason = _select_nestor_mode(ntheta=ntheta, nzeta=nzeta)

    rhs = -np.asarray(sample.vac_ext.bnormal_unit, dtype=float)
    ts = time.perf_counter()
    used_mode = selected_mode
    if mode_reason not in ("forced_fast", "forced_vmec_like", "auto_vmec_like"):
        used_mode = f"{selected_mode}_fallback:{mode_reason}"
    cache: Any = runtime_cache
    reuse_step = (int(ivac) != 1 and runtime is not None)

    # On ivacskip reuse, emulate VMEC2000 scalpot behavior by reusing the cached
    # operator while refreshing the source term / solve.
    mode_for_step = runtime_mode if reuse_step else used_mode
    if reuse_step and runtime_cache is None:
        mode_for_step = used_mode

    if _is_dense_mode(mode_for_step):
        alpha = _as_float_env("VMEC_JAX_FREEB_VMEC_LIKE_ALPHA", 1.0)
        dist_eps = _as_float_env("VMEC_JAX_FREEB_VMEC_LIKE_DIST_EPS", 1.0e-8)
        rhs_floor = _as_float_env("VMEC_JAX_FREEB_VMEC_LIKE_RHS_FLOOR", 1.0e-14)
        diag_coeff = _as_float_env("VMEC_JAX_FREEB_VMEC_LIKE_DIAG_COEFF", 0.5)
        row_sum_zero = _as_int_env("VMEC_JAX_FREEB_VMEC_LIKE_ROW_SUM_ZERO", 1) != 0
        singular_diag_scale = _as_float_env("VMEC_JAX_FREEB_VMEC_LIKE_SINGULAR_DIAG_SCALE", 1.0)
        try:
            if (
                not isinstance(cache, NestorVmecLikeCache)
                or int(cache.ntheta) != int(ntheta)
                or int(cache.nzeta) != int(nzeta)
                or not reuse_step
            ):
                cache = _build_vmec_like_cache(
                    sample,
                    alpha=alpha,
                    dist_eps=dist_eps,
                    rhs_floor=rhs_floor,
                    diag_coeff=diag_coeff,
                    row_sum_zero=row_sum_zero,
                    singular_diag_scale=singular_diag_scale,
                )
            phi = _solve_vmec_like_dense(rhs, cache)
            used_mode = mode_for_step
        except Exception:
            cache = _build_poisson_cache(ntheta=ntheta, nzeta=nzeta)
            phi = _solve_periodic_poisson_fft(rhs, cache)
            used_mode = "spectral_poisson_external_only_fallback:dense_failed"
    else:
        if (
            not isinstance(cache, NestorPoissonCache)
            or int(cache.ntheta) != int(ntheta)
            or int(cache.nzeta) != int(nzeta)
        ):
            cache = _build_poisson_cache(ntheta=ntheta, nzeta=nzeta)
        phi = _solve_periodic_poisson_fft(rhs, cache)
        used_mode = mode_for_step

    vac_total = _vacuum_channels_from_sample_phi(sample, phi)
    bsqvac = np.asarray(vac_total.bsqvac)
    solve_time = max(0.0, time.perf_counter() - ts)

    res = NestorSolveResult(
        vac_total=vac_total,
        phi=phi,
        reused=bool(reuse_step),
        solve_time_s=solve_time,
        sample_time_s=sample_time,
        model=used_mode,
    )
    runtime_next = NestorRuntimeState(
        operator_cache=cache,
        phi=np.asarray(phi),
        bsqvac=np.asarray(bsqvac),
        mode=used_mode,
        update_count=(0 if runtime is None else int(runtime.update_count)) + (0 if reuse_step else 1),
        reuse_count=(0 if runtime is None else int(runtime.reuse_count)) + (1 if reuse_step else 0),
    )
    _maybe_dump_scalpot_jax(
        iter_idx=iter_idx,
        ivac=int(ivac),
        reused=bool(reuse_step),
        mode=used_mode,
        rhs=np.asarray(rhs, dtype=float),
        phi=np.asarray(phi, dtype=float),
        vac=vac_total,
        cache=cache,
        sample=sample,
        mf=max(0, int(getattr(static.cfg, "mpol", 1)) + 1),
        nf=max(0, int(getattr(static.cfg, "ntor", 0))),
        nfp=max(1, int(getattr(static.cfg, "nfp", 1))),
        lasym=bool(getattr(static.cfg, "lasym", False)),
    )
    return res, runtime_next


def sample_external_vacuum_diagnostics(
    *,
    state: Any,
    static: Any,
    extcur: tuple[float, ...] | None = None,
) -> dict[str, Any]:
    """Sample external mgrid field on plasma boundary and derive vacuum channels.

    This is a WP2 diagnostic scaffold: it computes boundary geometry and
    VMEC-style surface field projections (`Bu/Bv/B^u/B^v/bsqvac`) using the
    external field only. NESTOR scalar-potential coupling is still pending.
    """

    out: dict[str, Any] = {
        "enabled": False,
        "available": False,
        "vacuum_stub": True,
    }

    t0 = time.perf_counter()
    try:
        sample = _sample_external_boundary_arrays(state=state, static=static, extcur=extcur)
        vac = sample.vac_ext
        br = sample.br
        bp = sample.bp
        bz = sample.bz
        bmag = np.sqrt(br * br + bp * bp + bz * bz)
        out.update(
            {
                "enabled": True,
                "available": True,
                "mgrid_path": sample.mgrid_path,
                "n_samples": int(br.size),
                "br_rms": float(np.sqrt(np.mean(br * br))),
                "bp_rms": float(np.sqrt(np.mean(bp * bp))),
                "bz_rms": float(np.sqrt(np.mean(bz * bz))),
                "bmag_mean": float(np.mean(bmag)),
                "bmag_max": float(np.max(bmag)),
                "bu_rms": float(np.sqrt(np.mean(vac.bu * vac.bu))),
                "bv_rms": float(np.sqrt(np.mean(vac.bv * vac.bv))),
                "bsupu_rms": float(np.sqrt(np.mean(vac.bsupu * vac.bsupu))),
                "bsupv_rms": float(np.sqrt(np.mean(vac.bsupv * vac.bsupv))),
                "bsqvac_mean": float(np.mean(vac.bsqvac)),
                "bsqvac_max": float(np.max(vac.bsqvac)),
                "bnormal_rms": float(np.sqrt(np.mean(vac.bnormal * vac.bnormal))),
                "bnormal_unit_rms": float(np.sqrt(np.mean(vac.bnormal_unit * vac.bnormal_unit))),
                "det_guv_min": float(np.min(vac.det_guv)),
                "det_guv_max": float(np.max(vac.det_guv)),
            }
        )
    except Exception as exc:
        out["reason"] = "sample_failed"
        out["error"] = str(exc)
    out["sample_time_s"] = float(max(0.0, time.perf_counter() - t0))
    return out


def _decode_char_scalar(x: Any) -> str:
    arr = np.asarray(x)
    if arr.dtype.kind == "S":
        if arr.ndim == 0:
            return arr.tobytes().decode(errors="ignore").strip()
        if arr.ndim == 1:
            return b"".join(arr.tolist()).decode(errors="ignore").strip()
    if arr.dtype.kind == "U":
        if arr.ndim == 0:
            return str(arr.item()).strip()
        if arr.ndim == 1:
            return "".join(arr.tolist()).strip()
    return str(arr).strip()


def _decode_char_rows(x: Any) -> tuple[str, ...]:
    arr = np.asarray(x)
    if arr.ndim == 1:
        return (_decode_char_scalar(arr),)
    if arr.ndim != 2:
        return tuple()
    out: list[str] = []
    for i in range(arr.shape[0]):
        out.append(_decode_char_scalar(arr[i, :]))
    return tuple(out)


_FIELD_RE = re.compile(r"^(br|bp|bz)_(\d{3})$")


def load_mgrid(path: str | Path, *, load_fields: bool = True) -> MGridMetadata | MGridData:
    """Load mgrid metadata and, optionally, coil field tables.

    Returns
    -------
    MGridMetadata or MGridData
        Metadata-only by default if `load_fields=False`, otherwise metadata + BR/BP/BZ
        arrays shaped `(nextcur, kp, jz, ir)`.
    """

    try:
        import netCDF4  # type: ignore
    except Exception as exc:
        raise RuntimeError("netCDF4 is required for mgrid loading") from exc

    p = Path(path).expanduser().resolve()
    with netCDF4.Dataset(str(p)) as ds:
        def _scalar(name: str, cast):
            if name not in ds.variables:
                raise KeyError(f"Missing mgrid variable: {name}")
            return cast(np.asarray(ds.variables[name][()]).item())

        ir = _scalar("ir", int)
        jz = _scalar("jz", int)
        kp = _scalar("kp", int)
        nfp = _scalar("nfp", int)
        nextcur = _scalar("nextcur", int)
        rmin = _scalar("rmin", float)
        rmax = _scalar("rmax", float)
        zmin = _scalar("zmin", float)
        zmax = _scalar("zmax", float)

        mode_var = ds.variables.get("mgrid_mode")
        mode = "S"
        if mode_var is not None:
            mode = _decode_char_scalar(mode_var[:])

        cg_var = ds.variables.get("coil_group")
        coil_groups = tuple()
        if cg_var is not None:
            coil_groups = _decode_char_rows(cg_var[:])

        cur_var = ds.variables.get("raw_coil_cur")
        raw_cur = tuple()
        if cur_var is not None:
            raw_cur = tuple(float(v) for v in np.asarray(cur_var[:]).reshape(-1))

        meta = MGridMetadata(
            path=str(p),
            ir=ir,
            jz=jz,
            kp=kp,
            nfp=nfp,
            nextcur=nextcur,
            rmin=rmin,
            rmax=rmax,
            zmin=zmin,
            zmax=zmax,
            mgrid_mode=mode,
            coil_groups=coil_groups,
            raw_coil_cur=raw_cur,
        )
        if not load_fields:
            return meta

        br = np.zeros((nextcur, kp, jz, ir), dtype=np.float64)
        bp = np.zeros((nextcur, kp, jz, ir), dtype=np.float64)
        bz = np.zeros((nextcur, kp, jz, ir), dtype=np.float64)

        fields: dict[tuple[str, int], np.ndarray] = {}
        for name, var in ds.variables.items():
            m = _FIELD_RE.match(name)
            if not m:
                continue
            kind = m.group(1)
            idx = int(m.group(2)) - 1
            fields[(kind, idx)] = np.asarray(var[:], dtype=np.float64)

        # VMEC mgrid netCDF convention is (phi, zee, rad).
        for i in range(nextcur):
            for kind, out in (("br", br), ("bp", bp), ("bz", bz)):
                v = fields.get((kind, i))
                if v is None:
                    continue
                if v.shape != (kp, jz, ir):
                    raise ValueError(
                        f"{kind}_{i+1:03d} shape {v.shape} != expected {(kp, jz, ir)}"
                    )
                out[i, :, :, :] = v

        return MGridData(metadata=meta, br=br, bp=bp, bz=bz)


def prepare_mgrid_for_config(
    cfg: VMECConfig,
    *,
    load_fields: bool = False,
    strict: bool = True,
) -> PreparedMGrid | MGridData | None:
    """Load and validate mgrid against VMECConfig.

    Validation mirrors VMEC2000's read_mgrid checks:
    - `nfp` must match mgrid `nfper0`,
    - `kp` must be divisible by `nzeta`.
    """

    if not bool(cfg.lfreeb):
        return None
    validate_free_boundary_config(cfg, strict=strict)
    loaded = load_mgrid(cfg.mgrid_file, load_fields=load_fields)
    meta = loaded.metadata if isinstance(loaded, MGridData) else loaded

    if int(meta.nfp) != int(cfg.nfp):
        raise ValueError(f"MGRID nfp={meta.nfp} does not match input nfp={cfg.nfp}")
    if int(cfg.nzeta) <= 0 or (int(meta.kp) % int(cfg.nzeta) != 0):
        raise ValueError(
            f"MGRID kp={meta.kp} must be divisible by nzeta={cfg.nzeta}"
        )

    extcur = _normalize_extcur(tuple(cfg.extcur), int(meta.nextcur))
    if isinstance(loaded, MGridData):
        return MGridData(metadata=meta, br=loaded.br, bp=loaded.bp, bz=loaded.bz)
    return PreparedMGrid(metadata=meta, extcur=extcur)
