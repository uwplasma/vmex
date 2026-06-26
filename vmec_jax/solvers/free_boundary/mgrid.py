"""VMEC mgrid loading, validation, and interpolation helpers.

These routines preserve the VMEC2000-compatible mgrid path used by
free-boundary solves.  They are deliberately separate from the NESTOR solver
body so that mgrid IO/validation remains testable without importing the full
free-boundary algorithm.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re
from typing import Any, Callable

import numpy as np

from ...config import VMECConfig
from .types import MGridData, MGridMetadata, PreparedMGrid

MGRID_FIELD_CACHE: dict[str, MGridData] = {}


def validate_free_boundary_config(cfg: VMECConfig, *, strict: bool = False) -> None:
    """Validate parsed free-boundary inputs.

    In strict mode, raise if ``LFREEB=T`` but no usable mgrid path is given.
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


def normalize_extcur(extcur_in: tuple[float, ...], nextcur: int) -> tuple[float, ...]:
    """Normalize normalize extcur for direct-coil free-boundary solve and branch-local adjoint validation."""
    if nextcur <= 0:
        return tuple()
    vals = list(float(v) for v in extcur_in)
    if len(vals) < nextcur:
        vals.extend([0.0] * (nextcur - len(vals)))
    elif len(vals) > nextcur:
        vals = vals[:nextcur]
    return tuple(vals)


def broadcast_xyz(r: Any, z: Any, phi: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate broadcast xyz for direct-coil free-boundary solve and branch-local adjoint validation."""
    return np.broadcast_arrays(
        np.asarray(r, dtype=float),
        np.asarray(z, dtype=float),
        np.asarray(phi, dtype=float),
    )


def interpolate_mgrid_bfield(
    data: MGridData,
    *,
    r: Any,
    z: Any,
    phi: Any,
    extcur: tuple[float, ...] | None = None,
    use_vmec_kv: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trilinear interpolation of mgrid BR/BP/BZ with periodic toroidal angle.

    Parameters
    ----------
    data:
        Loaded mgrid tensor data (``load_mgrid(..., load_fields=True)``).
    r, z, phi:
        Query coordinates (broadcastable arrays).
    extcur:
        External current weights per coil group. If omitted, uses
        ``raw_coil_cur`` when available; otherwise unit weights.
    """

    meta = data.metadata
    rr, zz, pp = broadcast_xyz(r, z, phi)
    out_shape = rr.shape

    ir = int(meta.ir)
    jz = int(meta.jz)
    kp = int(meta.kp)
    if ir < 2 or jz < 2 or kp < 1:
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

    # Toroidal index selection:
    # - VMEC becoil path uses a zeta-grid index (no toroidal interpolation),
    # - generic path uses periodic toroidal interpolation in physical angle.
    if bool(use_vmec_kv):
        if rr.ndim == 0:
            raise ValueError("use_vmec_kv=True requires array inputs with an explicit zeta axis")
        nzeta = int(rr.shape[-1]) if int(rr.shape[-1]) > 0 else kp
        if kp == 1:
            k_idx = np.zeros(nzeta, dtype=np.int64)
        else:
            if nzeta < 1:
                raise ValueError("use_vmec_kv=True requires at least one zeta plane")
            if kp % nzeta != 0:
                raise ValueError(
                    "use_vmec_kv=True requires the number of mgrid zeta planes "
                    "to be divisible by the VMEC zeta axis length; kp must be divisible by nzeta"
                )
            # VMEC becoil samples the mgrid planes corresponding to the VMEC
            # zeta grid without toroidal interpolation.
            k_idx = np.arange(nzeta, dtype=np.int64) * int(kp // nzeta)
        k0 = np.broadcast_to(k_idx.reshape((1,) * (rr.ndim - 1) + (nzeta,)), rr.shape).reshape(-1)
        k1 = k0
        wk = np.zeros_like(fr)
    else:
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

    raw_cur = tuple(meta.raw_coil_cur)
    if extcur is None and len(raw_cur) == 0:
        cur = tuple(1.0 for _ in range(int(meta.nextcur)))
    else:
        cur = normalize_extcur(
            tuple(extcur) if extcur is not None else raw_cur,
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


def decode_char_scalar(x: Any) -> str:
    """Evaluate decode char scalar for direct-coil free-boundary solve and branch-local adjoint validation."""
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


def decode_char_rows(x: Any) -> tuple[str, ...]:
    """Evaluate decode char rows for direct-coil free-boundary solve and branch-local adjoint validation."""
    arr = np.asarray(x)
    if arr.ndim == 1:
        return (decode_char_scalar(arr),)
    if arr.ndim != 2:
        return tuple()
    out: list[str] = []
    for i in range(arr.shape[0]):
        out.append(decode_char_scalar(arr[i, :]))
    return tuple(out)


_FIELD_RE = re.compile(r"^(br|bp|bz)_(\d{3})$")


def load_mgrid(path: str | Path, *, load_fields: bool = True) -> MGridMetadata | MGridData:
    """Load mgrid metadata and, optionally, coil field tables.

    Returns
    -------
    MGridMetadata or MGridData
        Metadata-only by default if ``load_fields=False``, otherwise metadata +
        BR/BP/BZ arrays shaped ``(nextcur, kp, jz, ir)``.
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
            mode = decode_char_scalar(mode_var[:])

        cg_var = ds.variables.get("coil_group")
        coil_groups = tuple()
        if cg_var is not None:
            coil_groups = decode_char_rows(cg_var[:])

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
    load_mgrid_fn: Callable[..., MGridMetadata | MGridData] | None = None,
    validate_config_fn: Callable[..., None] | None = None,
) -> PreparedMGrid | MGridData | None:
    """Load and validate mgrid against ``VMECConfig``.

    Validation mirrors VMEC2000's read_mgrid checks:
    - ``nfp`` must match mgrid ``nfper0``,
    - ``kp`` must be divisible by ``nzeta``.
    """

    if not bool(cfg.lfreeb):
        return None
    validate = validate_free_boundary_config if validate_config_fn is None else validate_config_fn
    validate(cfg, strict=strict)
    loader = load_mgrid if load_mgrid_fn is None else load_mgrid_fn
    loaded = loader(str(cfg.mgrid_file), load_fields=load_fields)
    meta = loaded.metadata if isinstance(loaded, MGridData) else loaded

    if int(meta.nfp) != int(cfg.nfp):
        raise ValueError(f"MGRID nfp={meta.nfp} does not match input nfp={cfg.nfp}")
    if int(cfg.nzeta) <= 0 or (int(meta.kp) % int(cfg.nzeta) != 0):
        raise ValueError(f"MGRID kp={meta.kp} must be divisible by nzeta={cfg.nzeta}")

    extcur = normalize_extcur(tuple(cfg.extcur), int(meta.nextcur))
    kp_eff = min(int(meta.kp), max(1, 2 * int(cfg.nzeta))) if int(cfg.nzeta) > 0 else int(meta.kp)
    meta_eff = replace(meta, kp=kp_eff)
    if isinstance(loaded, MGridData):
        return MGridData(metadata=meta_eff, br=loaded.br, bp=loaded.bp, bz=loaded.bz)
    return PreparedMGrid(metadata=meta_eff, extcur=extcur)


__all__ = [
    "MGRID_FIELD_CACHE",
    "broadcast_xyz",
    "decode_char_rows",
    "decode_char_scalar",
    "interpolate_mgrid_bfield",
    "load_mgrid",
    "normalize_extcur",
    "prepare_mgrid_for_config",
    "validate_free_boundary_config",
]
