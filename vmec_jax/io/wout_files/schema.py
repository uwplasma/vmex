"""Schema and low-level helpers for VMEC ``wout_*.nc`` data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ...modes import vmec_mode_table


@dataclass(frozen=True)
class WoutData:
    """In-memory representation of the VMEC ``wout_*.nc`` schema.

    The fields mirror VMEC2000 naming and mesh conventions as closely as
    possible so parity tests can compare arrays directly.  Optional or newer
    diagnostics are still represented with concrete arrays/scalars so plotting,
    validation, and optimization code can use one uniform object.
    """

    path: Path
    ns: int
    mpol: int
    ntor: int
    nfp: int
    lasym: bool
    signgs: int
    mnmax: int
    mpol_nyq: int
    ntor_nyq: int
    mnmax_nyq: int

    # main mode table
    xm: np.ndarray
    xn: np.ndarray

    # nyquist mode table
    xm_nyq: np.ndarray
    xn_nyq: np.ndarray

    # geometry coefficients (full mesh)
    rmnc: np.ndarray
    rmns: np.ndarray
    zmnc: np.ndarray
    zmns: np.ndarray
    lmnc: np.ndarray
    lmns: np.ndarray

    # flux functions / profiles
    phipf: np.ndarray
    chipf: np.ndarray
    phips: np.ndarray
    iotaf: np.ndarray  # (ns,) iota on full mesh (VMEC convention)
    iotas: np.ndarray  # (ns,) iota on half mesh (VMEC convention)

    # nyquist Fourier coefficients for derived fields
    gmnc: np.ndarray
    gmns: np.ndarray
    bsupumnc: np.ndarray
    bsupumns: np.ndarray
    bsupvmnc: np.ndarray
    bsupvmns: np.ndarray

    # nyquist Fourier coefficients for covariant field components (for parity checks)
    bsubumnc: np.ndarray
    bsubumns: np.ndarray
    bsubvmnc: np.ndarray
    bsubvmns: np.ndarray
    bsubsmns: np.ndarray
    bsubsmnc: np.ndarray

    # nyquist Fourier coefficients for |B|
    bmnc: np.ndarray
    bmns: np.ndarray

    wb: float
    volume_p: float

    # pressure / energy scalars (VMEC internal units)
    gamma: float
    wp: float
    vp: np.ndarray  # (ns,) volume derivative on half mesh, normalized by (2*pi)^2
    pres: np.ndarray  # (ns,) pressure on half mesh in mu0*Pa (B^2 units)
    presf: np.ndarray  # (ns,) pressure on full mesh in mu0*Pa (B^2 units)

    # force residual diagnostics (VMEC scalars)
    fsqr: float  # radial force residual
    fsqz: float  # vertical force residual
    fsql: float  # lambda/constraint residual
    fsqt: np.ndarray  # force trace vs iteration (if present)
    equif: np.ndarray  # (ns,) flux-surface-averaged force balance (if present)

    # additional wout fields used by vmecPlot2 and diagnostics
    phi: np.ndarray  # (ns,) toroidal flux
    buco: np.ndarray  # (ns,)
    bvco: np.ndarray  # (ns,)
    jcuru: np.ndarray  # (ns,)
    jcurv: np.ndarray  # (ns,)
    raxis_cc: np.ndarray  # (ntor+1,)
    zaxis_cs: np.ndarray  # (ntor+1,)
    raxis_cs: np.ndarray  # (ntor+1,)
    zaxis_cc: np.ndarray  # (ntor+1,)
    Aminor_p: float
    Rmajor_p: float
    aspect: float
    betatotal: float
    betapol: float
    betator: float
    betaxis: float
    ctor: float
    DMerc: np.ndarray  # (ns,)
    Dshear: np.ndarray  # (ns,)
    Dwell: np.ndarray  # (ns,)
    Dcurr: np.ndarray  # (ns,)
    Dgeod: np.ndarray  # (ns,)
    jdotb: np.ndarray  # (ns,)
    bdotb: np.ndarray  # (ns,)
    bdotgradv: np.ndarray  # (ns,)
    ac: np.ndarray  # (nac,)
    ac_aux_s: np.ndarray  # (ndfmax,)
    ac_aux_f: np.ndarray  # (ndfmax,)
    pcurr_type: str
    piota_type: str

    # vmec_jax solver status metadata.  VMEC2000 often leaves partially failed
    # outputs ambiguous; vmec_jax keeps computed fields and marks status.
    ier_flag: int = 0
    vmec_jax_converged: bool = True
    vmec_jax_status: str = "converged"
    D_R: np.ndarray | None = None  # (ns,) Glasser resistive-interchange diagnostic
    H: np.ndarray | None = None  # (ns,) Glasser H term in vmec_jax normalization
    glasser_correction: np.ndarray | None = None  # (ns,) positive correction added to -DMerc
    glasser_shear_valid: np.ndarray | None = None  # (ns,) nonzero-shear validity mask

    def __post_init__(self) -> None:
        zeros = np.zeros((int(self.ns),), dtype=float)
        false = np.zeros((int(self.ns),), dtype=bool)
        for name, default in (
            ("D_R", zeros),
            ("H", zeros),
            ("glasser_correction", zeros),
            ("glasser_shear_valid", false),
        ):
            if getattr(self, name) is None:
                object.__setattr__(self, name, default.copy())


def _bool_from_nc(x: Any) -> bool:
    # VMEC stores *_logical__ as 0/1 integers in netcdf.
    try:
        arr = np.asarray(np.ma.filled(x, 0))
        return bool(int(np.ravel(arr)[0]))
    except Exception:
        return bool(x)


def _nc_scalar(x: Any, default: float = 0.0, *, as_int: bool = False) -> int | float:
    """Robust scalar extraction from netCDF values, including masked scalars."""
    try:
        arr = np.asarray(np.ma.filled(x, default))
        val = np.ravel(arr)[0]
    except Exception:
        val = default
    if as_int:
        try:
            return int(val)
        except Exception:
            return int(default)
    try:
        return float(val)
    except Exception:
        return float(default)


def assert_main_modes_match_wout(*, wout: WoutData) -> None:
    """Ensure vmec_jax mode ordering matches the `wout` file (important for parity)."""
    modes = vmec_mode_table(wout.mpol, wout.ntor)
    if modes.K != int(wout.xm.size):
        raise ValueError(f"Mode count mismatch: vmec_jax K={modes.K} wout mnmax={wout.xm.size}")
    if not np.array_equal(modes.m, wout.xm.astype(int)):
        raise ValueError("wout xm ordering does not match vmec_jax vmec_mode_table")
    if not np.array_equal(modes.n, (wout.xn // wout.nfp).astype(int)):
        raise ValueError("wout xn ordering does not match vmec_jax (expected xn = n*nfp)")
