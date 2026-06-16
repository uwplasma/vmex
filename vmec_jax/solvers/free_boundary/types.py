"""Free-boundary data containers.

This module intentionally holds passive containers only.  Keeping these types
separate from the free-boundary solver body makes the public data contracts
easy to inspect and test without importing the full NESTOR/controller logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


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

    Arrays are defined on a single boundary surface with shape ``(ntheta, nzeta)``.
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
    br_mgrid: np.ndarray
    bp_mgrid: np.ndarray
    bz_mgrid: np.ndarray
    br_axis: np.ndarray
    bp_axis: np.ndarray
    bz_axis: np.ndarray
    axis_r: np.ndarray
    axis_z: np.ndarray
    vac_ext: VacuumBoundaryFields
    axis_r_full: np.ndarray | None = None
    axis_z_full: np.ndarray | None = None
    axis_r_parity: np.ndarray | None = None
    axis_z_parity: np.ndarray | None = None
    # Optional second-derivative channels on the same ``(ntheta, nzeta)`` grid.
    # When present these should match VMEC surface.f modal derivatives.
    ruu: np.ndarray | None = None
    ruv: np.ndarray | None = None
    rvv: np.ndarray | None = None
    zuu: np.ndarray | None = None
    zuv: np.ndarray | None = None
    zvv: np.ndarray | None = None
    timing: dict[str, float] | None = None


@dataclass(frozen=True)
class FreeBoundarySampleSetup:
    trig: Any
    second_facs: np.ndarray
    phi_grid: np.ndarray
    even_m_mask: np.ndarray
    wint_vmec: np.ndarray


@dataclass(frozen=True)
class NestorPoissonCache:
    """Stage-static spectral Poisson operator cache on the ``(theta,zeta)`` torus."""

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
    mode_basis: Any | None = None
    mode_matrix: np.ndarray | None = None
    matrix_lu: Any | None = None
    mode_matrix_lu: Any | None = None


@dataclass(frozen=True)
class NestorRuntimeState:
    """Runtime cache for the NESTOR solve path."""

    operator_cache: Any
    phi: np.ndarray
    bsqvac: np.ndarray
    mode: str
    update_count: int
    reuse_count: int
    # VMEC scalpot/fouri reuse state: on ivacskip>0, fouri is skipped and the
    # non-singular source transform from the last full update is reused.
    source_cache_iter: int = -1
    gsource_cached: np.ndarray | None = None
    source_sym_cached: np.ndarray | None = None
    bvec_nonsing_cached: np.ndarray | None = None


@dataclass(frozen=True)
class NestorSolveResult:
    """Output of one NESTOR-like update/reuse step."""

    vac_total: VacuumBoundaryFields
    phi: np.ndarray
    reused: bool
    solve_time_s: float
    sample_time_s: float
    model: str = "spectral_poisson_external_only"
    diagnostics: dict[str, float | str | bool] | None = None
    trace_arrays: dict[str, Any] | None = None


@dataclass(frozen=True)
class PreparedMGrid:
    """Validated mgrid metadata plus normalized external-current vector."""

    metadata: MGridMetadata
    extcur: tuple[float, ...]


__all__ = [
    "ExternalBoundarySample",
    "FreeBoundaryRuntimeState",
    "FreeBoundarySampleSetup",
    "MGridData",
    "MGridMetadata",
    "NestorPoissonCache",
    "NestorRuntimeState",
    "NestorSolveResult",
    "NestorVmecLikeCache",
    "PreparedMGrid",
    "VacuumBoundaryFields",
]
