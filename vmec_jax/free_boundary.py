"""Free-boundary typed config/state and mgrid loader skeleton.

WP0 scope:
- typed runtime state container for free-boundary iteration control,
- deterministic mgrid metadata/field loader,
- lightweight validation hooks that mirror VMEC2000 readin/read_indata behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from .config import VMECConfig


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

    from .vmec_realspace import (
        vmec_realspace_synthesis,
        vmec_realspace_synthesis_dtheta,
        vmec_realspace_synthesis_dzeta_phys,
    )
    from .vmec_tomnsp import vmec_trig_tables

    out: dict[str, Any] = {
        "enabled": False,
        "available": False,
        "vacuum_stub": True,
    }

    meta = getattr(static, "mgrid_metadata", None)
    if meta is None:
        out["reason"] = "missing_mgrid_metadata"
        return out
    mgrid_path = str(getattr(meta, "path", "")).strip()
    if not mgrid_path:
        out["reason"] = "missing_mgrid_path"
        return out

    t0 = time.perf_counter()
    try:
        mgrid = load_mgrid(mgrid_path, load_fields=True)
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

        # Boundary surface geometry (last radial surface).
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
        bmag = np.sqrt(br * br + bp * bp + bz * bz)
        out.update(
            {
                "enabled": True,
                "available": True,
                "mgrid_path": mgrid_path,
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
