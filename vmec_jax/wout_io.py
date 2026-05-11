"""Low-level netCDF helpers for VMEC ``wout_*.nc`` I/O."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .wout_schema import _nc_scalar


def read_mode_table(variables: Any, name: str, *, path: Path) -> np.ndarray:
    """Read a required VMEC mode table and reject fully masked metadata."""
    raw = variables[name][:]
    if np.ma.isMaskedArray(raw):
        mask = np.asarray(raw.mask)
        if mask.size > 0 and bool(np.all(mask)):
            raise ValueError(f"Incomplete or masked wout mode metadata ({name}) in {path}")
    return np.asarray(np.ma.filled(raw, 0.0), dtype=int)


def read_optional_int_scalar(variables: Any, name: str, default: int | float) -> int:
    """Read an optional integer scalar from a netCDF variable mapping."""
    if name not in variables:
        return int(default)
    return int(_nc_scalar(variables[name][:], default, as_int=True))


def read_type_field(variables: Any, name: str) -> str:
    """Read a VMEC fixed-width string field from netCDF character storage."""
    if name not in variables:
        return ""
    raw = np.asarray(variables[name][:])
    if raw.dtype.kind in ("S", "U"):
        if raw.ndim == 0:
            out = str(raw)
        else:
            out = b"".join(raw.astype("S1").ravel()).decode("utf-8", "ignore")
    else:
        try:
            out = "".join(raw.tolist())
        except Exception:
            out = str(raw)
    return out.rstrip()


def read_nyquist_fourier_fields(variables: Any) -> dict[str, np.ndarray]:
    """Read Nyquist Fourier field groups from a VMEC wout variable mapping."""
    gmnc = np.asarray(variables["gmnc"][:])
    bsupumnc = np.asarray(variables["bsupumnc"][:])
    bsupvmnc = np.asarray(variables["bsupvmnc"][:])

    return {
        "gmnc": gmnc,
        "gmns": np.asarray(variables.get("gmns", np.zeros_like(gmnc))[:]),
        "bsupumnc": bsupumnc,
        "bsupumns": np.asarray(variables.get("bsupumns", np.zeros_like(bsupumnc))[:]),
        "bsupvmnc": bsupvmnc,
        "bsupvmns": np.asarray(variables.get("bsupvmns", np.zeros_like(bsupvmnc))[:]),
        "bsubumnc": np.asarray(variables.get("bsubumnc", np.zeros_like(bsupumnc))[:]),
        "bsubumns": np.asarray(variables.get("bsubumns", np.zeros_like(bsupumnc))[:]),
        "bsubvmnc": np.asarray(variables.get("bsubvmnc", np.zeros_like(bsupvmnc))[:]),
        "bsubvmns": np.asarray(variables.get("bsubvmns", np.zeros_like(bsupvmnc))[:]),
        "bsubsmns": np.asarray(variables.get("bsubsmns", np.zeros_like(bsupvmnc))[:]),
        "bsubsmnc": np.asarray(variables.get("bsubsmnc", np.zeros_like(bsupvmnc))[:]),
        "bmnc": np.asarray(variables.get("bmnc", np.zeros_like(gmnc))[:]),
        "bmns": np.asarray(variables.get("bmns", np.zeros_like(gmnc))[:]),
    }


def write_int_variable(ds: Any, name: str, dims: tuple[str, ...], data: Any) -> None:
    """Create and write an int32 netCDF variable."""
    var = ds.createVariable(name, "i4", dims)
    var[:] = np.asarray(data, dtype=np.int32)


def write_float_variable(ds: Any, name: str, dims: tuple[str, ...], data: Any) -> None:
    """Create and write a float64 netCDF variable."""
    var = ds.createVariable(name, "f8", dims)
    var[:] = np.asarray(data, dtype=np.float64)


def write_fixed_width_string_variable(
    ds: Any,
    name: str,
    value: Any,
    *,
    dim: str = "dim_00020",
    width: int = 20,
) -> None:
    """Create and write a fixed-width VMEC string variable."""
    text = (str(value or "")[:width]).ljust(width)
    var = ds.createVariable(name, "S1", (dim,))
    var[:] = np.asarray(list(text), dtype="S1")


def write_nyquist_fourier_fields(ds: Any, wout: Any) -> None:
    """Write Nyquist Fourier field groups from a WoutData-like object."""
    dims = ("radius", "mn_mode_nyq")
    for name in (
        "gmnc",
        "gmns",
        "bsupumnc",
        "bsupumns",
        "bsupvmnc",
        "bsupvmns",
        "bsubumnc",
        "bsubumns",
        "bsubvmnc",
        "bsubvmns",
        "bsubsmns",
        "bsubsmnc",
        "bmnc",
        "bmns",
    ):
        write_float_variable(ds, name, dims, np.asarray(getattr(wout, name)))
