"""Low-level netCDF helpers for VMEC ``wout_*.nc`` I/O."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .wout_schema import _bool_from_nc, _nc_scalar


# VMEC Nyquist Fourier fields are stored with (radius, mn_mode_nyq) dimensions.
NYQUIST_FOURIER_FIELD_NAMES: tuple[str, ...] = (
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
)

_REQUIRED_NYQUIST_FOURIER_FIELDS = ("gmnc", "bsupumnc", "bsupvmnc")
_NYQUIST_DEFAULT_TEMPLATES = {
    "gmns": "gmnc",
    "bsupumns": "bsupumnc",
    "bsupvmns": "bsupvmnc",
    "bsubumnc": "bsupumnc",
    "bsubumns": "bsupumnc",
    "bsubvmnc": "bsupvmnc",
    "bsubvmns": "bsupvmnc",
    "bsubsmns": "bsupvmnc",
    "bsubsmnc": "bsupvmnc",
    "bmnc": "gmnc",
    "bmns": "gmnc",
}


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


def read_wout_scalar_metadata(variables: Any, *, path: Path) -> tuple[int, int, int, int, bool, int]:
    """Extract and validate scalar metadata required before reading WOUT arrays."""

    ns = int(_nc_scalar(variables["ns"][:], 0.0, as_int=True))
    mpol = int(_nc_scalar(variables["mpol"][:], 0.0, as_int=True))
    ntor = int(_nc_scalar(variables["ntor"][:], 0.0, as_int=True))
    nfp = int(_nc_scalar(variables["nfp"][:], 0.0, as_int=True))

    lasym_var = variables.get("lasym__logical__")
    lasym = _bool_from_nc(lasym_var[:] if lasym_var is not None else 0)
    signgs_var = variables.get("signgs")
    signgs = int(_nc_scalar(signgs_var[:] if signgs_var is not None else 1.0, 1.0, as_int=True))
    if ns <= 0 or mpol <= 0 or ntor < 0 or nfp <= 0:
        raise ValueError(f"Incomplete or masked wout scalar metadata in {path}")
    return ns, mpol, ntor, nfp, lasym, signgs


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
    """Read Nyquist Fourier field groups from a VMEC wout variable mapping.

    VMEC omits sine/asymmetric channels from some stellarator-symmetric output
    files.  Missing optional channels are returned as zeros shaped like the
    matching cosine field group so downstream code can treat every field as
    present.
    """

    required_fields = {name: np.asarray(variables[name][:]) for name in _REQUIRED_NYQUIST_FOURIER_FIELDS}
    fields: dict[str, np.ndarray] = {}
    for name in NYQUIST_FOURIER_FIELD_NAMES:
        if name in required_fields:
            fields[name] = required_fields[name]
            continue
        if name in variables:
            fields[name] = np.asarray(variables[name][:])
            continue
        fields[name] = np.zeros_like(fields[_NYQUIST_DEFAULT_TEMPLATES[name]])
    return fields


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
    for name in NYQUIST_FOURIER_FIELD_NAMES:
        write_float_variable(ds, name, dims, np.asarray(getattr(wout, name)))
