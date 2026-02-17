"""Boundary coefficient handling.

VMEC input files specify boundary Fourier coefficients in arrays:

- rbc(n,m), rbs(n,m)
- zbc(n,m), zbs(n,m)

where the boundary is expanded in helical harmonics

    R(θ,ζ) = Σ_{m,n} [ rbc(n,m) cos(mθ - nζ) + rbs(n,m) sin(mθ - nζ) ]
    Z(θ,ζ) = Σ_{m,n} [ zbc(n,m) cos(mθ - nζ) + zbs(n,m) sin(mθ - nζ) ]

This module provides helpers to map the sparse namelist assignments to dense coefficient
vectors aligned with a ModeTable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .namelist import InData
from .modes import ModeTable


@dataclass(frozen=True)
class BoundaryCoeffs:
    # Each is shape (K,), aligned with ModeTable
    R_cos: np.ndarray
    R_sin: np.ndarray
    Z_cos: np.ndarray
    Z_sin: np.ndarray


@dataclass(frozen=True)
class BoundaryInternalCoeffs:
    """VMEC internal boundary coefficients (cos/sin of u and v)."""

    rbcc: np.ndarray
    rbss: np.ndarray
    rbcs: np.ndarray
    rbsc: np.ndarray
    zbcc: np.ndarray
    zbss: np.ndarray
    zbcs: np.ndarray
    zbsc: np.ndarray


def _boundary_internal_flip_theta(
    internal: BoundaryInternalCoeffs,
    *,
    lthreed: bool,
    lasym: bool,
) -> BoundaryInternalCoeffs:
    """Apply VMEC's flip_theta to internal boundary arrays (PI - theta)."""
    rbcc = internal.rbcc.copy()
    rbss = internal.rbss.copy()
    rbcs = internal.rbcs.copy()
    rbsc = internal.rbsc.copy()
    zbcc = internal.zbcc.copy()
    zbss = internal.zbss.copy()
    zbcs = internal.zbcs.copy()
    zbsc = internal.zbsc.copy()

    mpol = rbcc.shape[1] - 1
    mul1 = -1.0
    for m in range(1, mpol + 1):
        rbcc[:, m] = mul1 * rbcc[:, m]
        zbsc[:, m] = -mul1 * zbsc[:, m]
        if lthreed:
            rbss[:, m] = -mul1 * rbss[:, m]
            zbcs[:, m] = mul1 * zbcs[:, m]
        if lasym:
            rbsc[:, m] = -mul1 * rbsc[:, m]
            zbcc[:, m] = mul1 * zbcc[:, m]
            if lthreed:
                rbcs[:, m] = mul1 * rbcs[:, m]
                zbss[:, m] = -mul1 * zbss[:, m]
        mul1 = -mul1

    return BoundaryInternalCoeffs(
        rbcc=rbcc,
        rbss=rbss,
        rbcs=rbcs,
        rbsc=rbsc,
        zbcc=zbcc,
        zbss=zbss,
        zbcs=zbcs,
        zbsc=zbsc,
    )

def _get_indexed(indata: InData, name: str) -> Dict[Tuple[int, ...], float]:
    d = indata.indexed.get(name.upper(), {})
    out: Dict[Tuple[int, ...], float] = {}
    for k, v in d.items():
        if isinstance(v, bool) or isinstance(v, str):
            continue
        out[tuple(k)] = float(v)
    return out


def _infer_mpol_ntor(modes: ModeTable) -> tuple[int, int]:
    m_arr = np.asarray(modes.m, dtype=int)
    n_arr = np.asarray(modes.n, dtype=int)
    mpol = int(m_arr.max()) if m_arr.size else 0
    ntor = int(np.abs(n_arr).max()) if n_arr.size else 0
    return mpol, ntor


def _boundary_internal_from_helical(
    boundary: BoundaryCoeffs,
    modes: ModeTable,
    *,
    lthreed: bool,
    lasym: bool,
) -> BoundaryInternalCoeffs:
    """Convert helical boundary coefficients to VMEC internal arrays."""
    mpol, ntor = _infer_mpol_ntor(modes)
    shape = (ntor + 1, mpol + 1)
    rbcc = np.zeros(shape, dtype=float)
    rbss = np.zeros(shape, dtype=float)
    rbcs = np.zeros(shape, dtype=float)
    rbsc = np.zeros(shape, dtype=float)
    zbcc = np.zeros(shape, dtype=float)
    zbss = np.zeros(shape, dtype=float)
    zbcs = np.zeros(shape, dtype=float)
    zbsc = np.zeros(shape, dtype=float)

    m_arr = np.asarray(modes.m, dtype=int)
    n_arr = np.asarray(modes.n, dtype=int)
    for k, (m_i, n_i) in enumerate(zip(m_arr, n_arr)):
        m_i = int(m_i)
        n_i = int(n_i)
        if m_i < 0:
            continue
        ni = abs(n_i)
        if n_i == 0:
            isgn = 0
        elif n_i > 0:
            isgn = 1
        else:
            isgn = -1

        rbc = float(boundary.R_cos[k])
        rbs = float(boundary.R_sin[k])
        zbc = float(boundary.Z_cos[k])
        zbs = float(boundary.Z_sin[k])

        rbcc[ni, m_i] += rbc
        if m_i > 0:
            zbsc[ni, m_i] += zbs

        if lthreed:
            if m_i > 0:
                rbss[ni, m_i] += isgn * rbc
            zbcs[ni, m_i] += -isgn * zbs

        if lasym:
            if m_i > 0:
                rbsc[ni, m_i] += rbs
            zbcc[ni, m_i] += zbc
            if lthreed:
                rbcs[ni, m_i] += -isgn * rbs
                if m_i > 0:
                    zbss[ni, m_i] += isgn * zbc

    return BoundaryInternalCoeffs(
        rbcc=rbcc,
        rbss=rbss,
        rbcs=rbcs,
        rbsc=rbsc,
        zbcc=zbcc,
        zbss=zbss,
        zbcs=zbcs,
        zbsc=zbsc,
    )


def _boundary_helical_from_internal(
    internal: BoundaryInternalCoeffs,
    modes: ModeTable,
    *,
    lthreed: bool,
    lasym: bool,
) -> BoundaryCoeffs:
    """Convert VMEC internal boundary arrays back to helical coefficients."""
    m_arr = np.asarray(modes.m, dtype=int)
    n_arr = np.asarray(modes.n, dtype=int)
    K = m_arr.size
    R_cos = np.zeros((K,), dtype=float)
    R_sin = np.zeros((K,), dtype=float)
    Z_cos = np.zeros((K,), dtype=float)
    Z_sin = np.zeros((K,), dtype=float)

    rbcc = internal.rbcc
    rbss = internal.rbss
    rbcs = internal.rbcs
    rbsc = internal.rbsc
    zbcc = internal.zbcc
    zbss = internal.zbss
    zbcs = internal.zbcs
    zbsc = internal.zbsc

    for k, (m_i, n_i) in enumerate(zip(m_arr, n_arr)):
        m_i = int(m_i)
        n_i = int(n_i)
        if m_i < 0:
            continue
        ni = abs(n_i)
        if m_i == 0 and n_i != 0:
            isgn = 1 if n_i > 0 else -1
            R_cos[k] = rbcc[ni, m_i]
            if lasym:
                R_sin[k] = -isgn * rbcs[ni, m_i]
                Z_cos[k] = zbcc[ni, m_i]
            else:
                R_sin[k] = 0.0
                Z_cos[k] = 0.0
            Z_sin[k] = -isgn * zbcs[ni, m_i]
            continue
        if n_i == 0:
            R_cos[k] = rbcc[ni, m_i]
            R_sin[k] = rbsc[ni, m_i] if lasym else 0.0
            Z_cos[k] = zbcc[ni, m_i] if lasym else 0.0
            Z_sin[k] = zbsc[ni, m_i]
        elif n_i > 0:
            R_cos[k] = 0.5 * (rbcc[ni, m_i] + rbss[ni, m_i])
            R_sin[k] = 0.5 * (rbsc[ni, m_i] - rbcs[ni, m_i])
            Z_cos[k] = 0.5 * (zbcc[ni, m_i] + zbss[ni, m_i])
            Z_sin[k] = 0.5 * (zbsc[ni, m_i] - zbcs[ni, m_i])
        else:
            R_cos[k] = 0.5 * (rbcc[ni, m_i] - rbss[ni, m_i])
            R_sin[k] = 0.5 * (rbsc[ni, m_i] + rbcs[ni, m_i])
            Z_cos[k] = 0.5 * (zbcc[ni, m_i] - zbss[ni, m_i])
            Z_sin[k] = 0.5 * (zbsc[ni, m_i] + zbcs[ni, m_i])

    return BoundaryCoeffs(R_cos=R_cos, R_sin=R_sin, Z_cos=Z_cos, Z_sin=Z_sin)


def boundary_apply_vmec_m1_constraint(
    boundary: BoundaryCoeffs,
    modes: ModeTable,
    *,
    lthreed: bool,
    lasym: bool,
) -> BoundaryCoeffs:
    """Apply VMEC's internal m=1 boundary constraint and return helical coeffs."""
    internal = _boundary_internal_from_helical(boundary, modes, lthreed=lthreed, lasym=lasym)
    if internal.rbcc.shape[1] > 1:
        if lthreed:
            temp = internal.rbss[:, 1].copy()
            internal.rbss[:, 1] = 0.5 * (temp + internal.zbcs[:, 1])
            internal.zbcs[:, 1] = 0.5 * (temp - internal.zbcs[:, 1])
        if lasym:
            temp = internal.rbsc[:, 1].copy()
            internal.rbsc[:, 1] = 0.5 * (temp + internal.zbcc[:, 1])
            internal.zbcc[:, 1] = 0.5 * (temp - internal.zbcc[:, 1])
    return _boundary_helical_from_internal(internal, modes, lthreed=lthreed, lasym=lasym)


def boundary_undo_vmec_m1_constraint(
    boundary: BoundaryCoeffs,
    modes: ModeTable,
    *,
    lthreed: bool,
    lasym: bool,
) -> BoundaryCoeffs:
    """Undo VMEC's internal m=1 boundary constraint and return helical coeffs."""
    internal = _boundary_internal_from_helical(boundary, modes, lthreed=lthreed, lasym=lasym)
    if internal.rbcc.shape[1] > 1:
        if lthreed:
            temp = internal.rbss[:, 1].copy()
            internal.rbss[:, 1] = temp + internal.zbcs[:, 1]
            internal.zbcs[:, 1] = temp - internal.zbcs[:, 1]
        if lasym:
            temp = internal.rbsc[:, 1].copy()
            internal.rbsc[:, 1] = temp + internal.zbcc[:, 1]
            internal.zbcc[:, 1] = temp - internal.zbcc[:, 1]
    return _boundary_helical_from_internal(internal, modes, lthreed=lthreed, lasym=lasym)


def boundary_from_indata(
    indata: InData,
    modes: ModeTable,
    *,
    apply_m1_constraint: bool = False,
) -> BoundaryCoeffs:
    """Build dense boundary coefficient vectors aligned with `modes`.

    Notes:
    - Input indices are (n,m) in the namelist and in VMEC's vmec_input module.
    - Our ModeTable stores (m,n), so we swap.
    """
    rbc = _get_indexed(indata, "RBC")
    rbs = _get_indexed(indata, "RBS")
    zbc = _get_indexed(indata, "ZBC")
    zbs = _get_indexed(indata, "ZBS")

    K = modes.K
    R_cos = np.zeros((K,), dtype=float)
    R_sin = np.zeros((K,), dtype=float)
    Z_cos = np.zeros((K,), dtype=float)
    Z_sin = np.zeros((K,), dtype=float)

    # Build a quick lookup from (m,n) to k
    key_to_k = {(int(m), int(n)): k for k, (m, n) in enumerate(zip(modes.m, modes.n))}

    def assign_from(src: Dict[Tuple[int, ...], float], dest: np.ndarray, kind: str):
        for (n, m), val in src.items():
            k = key_to_k.get((int(m), int(n)))
            if k is None:
                continue
            dest[k] = val

    assign_from(rbc, R_cos, "rbc")
    assign_from(rbs, R_sin, "rbs")
    assign_from(zbc, Z_cos, "zbc")
    assign_from(zbs, Z_sin, "zbs")

    boundary = BoundaryCoeffs(R_cos=R_cos, R_sin=R_sin, Z_cos=Z_cos, Z_sin=Z_sin)

    lasym = bool(indata.get_bool("LASYM", False))
    mpol, ntor = _infer_mpol_ntor(modes)
    lthreed = ntor > 0

    # Convert through VMEC internal representation to match readin.f sign handling.
    internal = _boundary_internal_from_helical(boundary, modes, lthreed=lthreed, lasym=lasym)

    # VMEC readin.f: check sign of Jacobian using m=1 internal boundary modes
    # and optionally flip theta (PI - theta). This enforces signgs=-1 convention.
    if internal.rbcc.shape[1] > 1 and internal.rbcc.shape[0] > 1:
        rtest = float(np.sum(internal.rbcc[1:, 1]))
        ztest = float(np.sum(internal.zbsc[1:, 1]))
        if (rtest * ztest) < 0.0:
            internal = _boundary_internal_flip_theta(internal, lthreed=lthreed, lasym=lasym)
    if apply_m1_constraint and bool(indata.get_bool("LCONM1", True)) and (lthreed or lasym):
        if internal.rbcc.shape[1] > 1:
            if lthreed:
                temp = internal.rbss[:, 1].copy()
                internal.rbss[:, 1] = 0.5 * (temp + internal.zbcs[:, 1])
                internal.zbcs[:, 1] = 0.5 * (temp - internal.zbcs[:, 1])
            if lasym:
                temp = internal.rbsc[:, 1].copy()
                internal.rbsc[:, 1] = 0.5 * (temp + internal.zbcc[:, 1])
                internal.zbcc[:, 1] = 0.5 * (temp - internal.zbcc[:, 1])

    return _boundary_helical_from_internal(internal, modes, lthreed=lthreed, lasym=lasym)
