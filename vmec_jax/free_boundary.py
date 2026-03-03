"""Free-boundary typed config/state and mgrid loader skeleton.

WP0 scope:
- typed runtime state container for free-boundary iteration control,
- deterministic mgrid metadata/field loader,
- lightweight validation hooks that mirror VMEC2000 readin/read_indata behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
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
