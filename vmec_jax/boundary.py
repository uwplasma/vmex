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
from collections import OrderedDict
import os

import numpy as np
import jax
import jax.numpy as jnp

from .namelist import InData
from .modes import ModeTable
from .fourier import build_helical_basis, eval_fourier


def boundary_aspect_ratio(boundary: BoundaryCoeffs, basis) -> jax.Array:
    """Compute aspect ratio from boundary coefficients using a precomputed basis."""
    Rb = eval_fourier(jnp.asarray(boundary.R_cos), jnp.asarray(boundary.R_sin), basis)
    Zb = eval_fourier(jnp.asarray(boundary.Z_cos), jnp.asarray(boundary.Z_sin), basis)
    dA = Rb * jnp.roll(Zb, -1, axis=0) - jnp.roll(Rb, -1, axis=0) * Zb
    area = 0.5 * jnp.sum(dA, axis=0)
    minor = jnp.sqrt(jnp.abs(area) / jnp.pi)
    Rmax = jnp.max(Rb, axis=0)
    Rmin = jnp.min(Rb, axis=0)
    Rmajor = jnp.mean(0.5 * (Rmax + Rmin))
    Aminor = jnp.mean(minor)
    return Rmajor / Aminor


def _is_jax_array(value: object) -> bool:
    return isinstance(value, (jax.Array, jax.core.Tracer))


def boundary_aspect_ratio_from_static(boundary: BoundaryCoeffs, static) -> jax.Array:
    """Compute aspect ratio from boundary coefficients using a VMEC static config."""
    basis = build_helical_basis(static.modes, static.grid)
    return boundary_aspect_ratio(boundary, basis)


@dataclass(frozen=True)
class BoundaryCoeffs:
    """Physical boundary Fourier coefficients aligned with a ``ModeTable``.

    The arrays store the input-facing ``R/Z`` cosine and sine channels for the
    same ``(m, n)`` mode ordering.  This is the object optimization examples
    manipulate when boundary degrees of freedom are exposed to autodiff.
    """

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


_BOUNDARY_CACHE: "OrderedDict[tuple, BoundaryCoeffs]" = OrderedDict()
_BOUNDARY_CACHE_MAX = 32


def _boundary_internal_flip_theta(
    internal: BoundaryInternalCoeffs,
    *,
    lthreed: bool,
    lasym: bool,
) -> BoundaryInternalCoeffs:
    """Apply VMEC's flip_theta to internal boundary arrays (PI - theta).

    Under theta → pi - theta the (m, n) harmonic gains a sign factor (-1)^m.
    The eight internal arrays are signed as follows:
      rbcc[:, m] *= (-1)^m       zbsc[:, m] *= (-1)^(m+1)
      rbss[:, m] *= (-1)^(m+1)  zbcs[:, m] *= (-1)^m        (lthreed)
      rbsc[:, m] *= (-1)^(m+1)  zbcc[:, m] *= (-1)^m        (lasym)
      rbcs[:, m] *= (-1)^m      zbss[:, m] *= (-1)^(m+1)    (lasym+lthreed)
    """
    if _is_jax_array(internal.rbcc):
        # Build a vectorised sign array: signs[m] = (-1)^m for m=0..mpol.
        # Avoid a Python loop over m (each .at[] on a JAX array outside JIT triggers
        # an eager XLA compilation).  A single broadcast-multiply is one XLA op.
        mpol = int(internal.rbcc.shape[1] - 1)
        signs = jnp.array([(-1.0) ** m for m in range(mpol + 1)])  # (mpol+1,)
        # signs starts at m=0 → +1, m=1 → -1, ...
        # Broadcast over (ntor+1, mpol+1) layout: multiply along the m axis (axis=1).
        rbcc = jnp.asarray(internal.rbcc) * signs[None, :]
        zbsc = jnp.asarray(internal.zbsc) * (-signs[None, :])
        rbss = jnp.asarray(internal.rbss) * (-signs[None, :]) if lthreed else jnp.asarray(internal.rbss)
        zbcs = jnp.asarray(internal.zbcs) * signs[None, :] if lthreed else jnp.asarray(internal.zbcs)
        rbsc = jnp.asarray(internal.rbsc) * (-signs[None, :]) if lasym else jnp.asarray(internal.rbsc)
        zbcc = jnp.asarray(internal.zbcc) * signs[None, :] if lasym else jnp.asarray(internal.zbcc)
        zbss = jnp.asarray(internal.zbss) * (-signs[None, :]) if (lasym and lthreed) else jnp.asarray(internal.zbss)
        rbcs = jnp.asarray(internal.rbcs) * signs[None, :] if (lasym and lthreed) else jnp.asarray(internal.rbcs)
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


def _rotate_lasym_input_maps(
    *,
    indata: InData,
    rbc: Dict[Tuple[int, ...], float],
    rbs: Dict[Tuple[int, ...], float],
    zbc: Dict[Tuple[int, ...], float],
    zbs: Dict[Tuple[int, ...], float],
) -> tuple[
    Dict[Tuple[int, ...], float],
    Dict[Tuple[int, ...], float],
    Dict[Tuple[int, ...], float],
    Dict[Tuple[int, ...], float],
]:
    """Apply VMEC readin.f LASYM boundary rotation to sparse input maps."""
    if not bool(indata.get_bool("LASYM", False)):
        return rbc, rbs, zbc, zbs

    rbc01 = float(rbc.get((0, 1), 0.0))
    rbs01 = float(rbs.get((0, 1), 0.0))
    zbc01 = float(zbc.get((0, 1), 0.0))
    zbs01 = float(zbs.get((0, 1), 0.0))
    denom = abs(rbc01) + abs(zbs01)
    delta = 0.0 if denom == 0.0 else float(np.arctan((rbs01 - zbc01) / denom))
    if delta == 0.0:
        return rbc, rbs, zbc, zbs

    def _rotate_pair(cos_map: Dict[Tuple[int, ...], float], sin_map: Dict[Tuple[int, ...], float]):
        out_cos: Dict[Tuple[int, ...], float] = {}
        out_sin: Dict[Tuple[int, ...], float] = {}
        keys = set(cos_map.keys()) | set(sin_map.keys())
        for key in keys:
            n_i, m_i = int(key[0]), int(key[1])
            val_cos = float(cos_map.get((n_i, m_i), 0.0))
            val_sin = float(sin_map.get((n_i, m_i), 0.0))
            ang = float(m_i) * delta
            c = float(np.cos(ang))
            s = float(np.sin(ang))
            out_cos[(n_i, m_i)] = val_cos * c + val_sin * s
            out_sin[(n_i, m_i)] = val_sin * c - val_cos * s
        return out_cos, out_sin

    return (*_rotate_pair(rbc, rbs), *_rotate_pair(zbc, zbs))


def boundary_input_from_indata(indata: InData, modes: ModeTable) -> BoundaryCoeffs:
    """Build dense boundary coefficients in the raw input convention.

    This matches the Fourier coefficient convention used by VMEC namelists and
    by SurfaceRZFourier-facing wrappers before VMEC's internal sign handling and
    optional theta flip are applied.
    """
    rbc = _get_indexed(indata, "RBC")
    rbs = _get_indexed(indata, "RBS")
    zbc = _get_indexed(indata, "ZBC")
    zbs = _get_indexed(indata, "ZBS")
    rbc, rbs, zbc, zbs = _rotate_lasym_input_maps(indata=indata, rbc=rbc, rbs=rbs, zbc=zbc, zbs=zbs)

    K = modes.K
    R_cos = np.zeros((K,), dtype=float)
    R_sin = np.zeros((K,), dtype=float)
    Z_cos = np.zeros((K,), dtype=float)
    Z_sin = np.zeros((K,), dtype=float)

    key_to_k = {(int(m), int(n)): k for k, (m, n) in enumerate(zip(modes.m, modes.n))}

    def assign_from(src: Dict[Tuple[int, ...], float], dest: np.ndarray):
        for (n, m), val in src.items():
            k = key_to_k.get((int(m), int(n)))
            if k is not None:
                dest[k] = val

    assign_from(rbc, R_cos)
    assign_from(rbs, R_sin)
    assign_from(zbc, Z_cos)
    assign_from(zbs, Z_sin)
    return BoundaryCoeffs(R_cos=R_cos, R_sin=R_sin, Z_cos=Z_cos, Z_sin=Z_sin)


def _boundary_cache_key(
    indata: InData,
    modes: ModeTable,
    *,
    apply_m1_constraint: bool,
    rbc: Dict[Tuple[int, ...], float],
    rbs: Dict[Tuple[int, ...], float],
    zbc: Dict[Tuple[int, ...], float],
    zbs: Dict[Tuple[int, ...], float],
) -> tuple:
    """Build a cache key for boundary decomposition across runs."""
    lasym = bool(indata.get_bool("LASYM", False))
    lconm1 = bool(indata.get_bool("LCONM1", True))
    mpol, ntor = _infer_mpol_ntor(modes)
    path = getattr(indata, "source_path", None)
    path_key = None
    if path:
        try:
            st = os.stat(path)
            path_key = (str(path), int(st.st_mtime_ns), int(st.st_size))
        except OSError:
            path_key = None

    coeff_key = None
    if path_key is None:
        def _sorted_items(d: Dict[Tuple[int, ...], float]):
            return tuple(sorted((int(k[0]), int(k[1]), float(v)) for k, v in d.items()))

        coeff_key = (
            _sorted_items(rbc),
            _sorted_items(rbs),
            _sorted_items(zbc),
            _sorted_items(zbs),
        )

    return (path_key, coeff_key, int(mpol), int(ntor), lasym, lconm1, bool(apply_m1_constraint))


def _boundary_cache_get(key: tuple) -> BoundaryCoeffs | None:
    cached = _BOUNDARY_CACHE.get(key)
    if cached is None:
        return None
    _BOUNDARY_CACHE.move_to_end(key)
    return cached


def _boundary_cache_put(key: tuple, boundary: BoundaryCoeffs) -> None:
    _BOUNDARY_CACHE[key] = boundary
    _BOUNDARY_CACHE.move_to_end(key)
    while len(_BOUNDARY_CACHE) > _BOUNDARY_CACHE_MAX:
        _BOUNDARY_CACHE.popitem(last=False)


def _infer_mpol_ntor(modes: ModeTable) -> tuple[int, int]:
    m_arr = np.asarray(modes.m, dtype=int)
    n_arr = np.asarray(modes.n, dtype=int)
    mpol = int(m_arr.max()) if m_arr.size else 0
    ntor = int(np.abs(n_arr).max()) if n_arr.size else 0
    return mpol, ntor


def boundary_from_input_convention(
    boundary: BoundaryCoeffs,
    modes: ModeTable,
    *,
    lasym: bool,
    apply_m1_constraint: bool = False,
) -> BoundaryCoeffs:
    """Convert raw input-convention boundary coefficients to solver convention."""
    mpol, ntor = _infer_mpol_ntor(modes)
    lthreed = ntor > 0
    internal = _boundary_internal_from_helical(boundary, modes, lthreed=lthreed, lasym=lasym)

    if internal.rbcc.shape[1] > 1 and internal.rbcc.shape[0] > 1:
        if _is_jax_array(internal.rbcc) or _is_jax_array(internal.zbsc):
            rtest = jnp.sum(jnp.asarray(internal.rbcc)[1:, 1])
            ztest = jnp.sum(jnp.asarray(internal.zbsc)[1:, 1])
            flip = (rtest * ztest) < 0.0
            flipped = _boundary_internal_flip_theta(internal, lthreed=lthreed, lasym=lasym)

            def _select(new_arr, old_arr):
                return jnp.where(flip, jnp.asarray(new_arr), jnp.asarray(old_arr))

            internal = BoundaryInternalCoeffs(
                rbcc=_select(flipped.rbcc, internal.rbcc),
                rbss=_select(flipped.rbss, internal.rbss),
                rbcs=_select(flipped.rbcs, internal.rbcs),
                rbsc=_select(flipped.rbsc, internal.rbsc),
                zbcc=_select(flipped.zbcc, internal.zbcc),
                zbss=_select(flipped.zbss, internal.zbss),
                zbcs=_select(flipped.zbcs, internal.zbcs),
                zbsc=_select(flipped.zbsc, internal.zbsc),
            )
        else:
            rtest = float(np.sum(internal.rbcc[1:, 1]))
            ztest = float(np.sum(internal.zbsc[1:, 1]))
            if (rtest * ztest) < 0.0:
                internal = _boundary_internal_flip_theta(internal, lthreed=lthreed, lasym=lasym)
    if apply_m1_constraint and (lthreed or lasym) and internal.rbcc.shape[1] > 1:
        if lthreed:
            temp = internal.rbss[:, 1].copy()
            internal.rbss[:, 1] = 0.5 * (temp + internal.zbcs[:, 1])
            internal.zbcs[:, 1] = 0.5 * (temp - internal.zbcs[:, 1])
        if lasym:
            temp = internal.rbsc[:, 1].copy()
            internal.rbsc[:, 1] = 0.5 * (temp + internal.zbcc[:, 1])
            internal.zbcc[:, 1] = 0.5 * (temp - internal.zbcc[:, 1])

    return _boundary_helical_from_internal(internal, modes, lthreed=lthreed, lasym=lasym)


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

    # JAX-friendly path for traced arrays.
    if _is_jax_array(boundary.R_cos) or _is_jax_array(boundary.Z_sin):
        m_arr = jnp.asarray(modes.m, dtype=jnp.int32)
        n_arr = jnp.asarray(modes.n, dtype=jnp.int32)
        m_nonneg = m_arr >= 0
        m_pos = m_arr > 0
        ni = jnp.abs(n_arr)
        isgn = jnp.sign(n_arr).astype(jnp.float64)

        m_safe = jnp.where(m_nonneg, m_arr, 0)
        ni_safe = jnp.where(m_nonneg, ni, 0)
        mask = m_nonneg.astype(jnp.float64)
        mask_pos = m_pos.astype(jnp.float64)

        rbc = jnp.asarray(boundary.R_cos)
        rbs = jnp.asarray(boundary.R_sin)
        zbc = jnp.asarray(boundary.Z_cos)
        zbs = jnp.asarray(boundary.Z_sin)

        rbcc = jnp.zeros(shape, dtype=rbc.dtype).at[ni_safe, m_safe].add(rbc * mask)
        rbss = jnp.zeros(shape, dtype=rbc.dtype)
        rbcs = jnp.zeros(shape, dtype=rbc.dtype)
        rbsc = jnp.zeros(shape, dtype=rbc.dtype)
        zbcc = jnp.zeros(shape, dtype=rbc.dtype)
        zbss = jnp.zeros(shape, dtype=rbc.dtype)
        zbcs = jnp.zeros(shape, dtype=rbc.dtype)
        zbsc = jnp.zeros(shape, dtype=rbc.dtype).at[ni_safe, m_safe].add(zbs * mask_pos)

        if lthreed:
            rbss = rbss.at[ni_safe, m_safe].add(isgn * rbc * mask_pos)
            zbcs = zbcs.at[ni_safe, m_safe].add(-isgn * zbs * mask)

        if lasym:
            rbsc = rbsc.at[ni_safe, m_safe].add(rbs * mask_pos)
            zbcc = zbcc.at[ni_safe, m_safe].add(zbc * mask)
            if lthreed:
                rbcs = rbcs.at[ni_safe, m_safe].add(-isgn * rbs * mask)
                zbss = zbss.at[ni_safe, m_safe].add(isgn * zbc * mask_pos)

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

    # NumPy path for host-side preprocessing.
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
    if _is_jax_array(internal.rbcc):
        m_arr = jnp.asarray(modes.m, dtype=jnp.int32)
        n_arr = jnp.asarray(modes.n, dtype=jnp.int32)
        K = m_arr.size
        R_cos = jnp.zeros((K,), dtype=internal.rbcc.dtype)
        R_sin = jnp.zeros((K,), dtype=internal.rbcc.dtype)
        Z_cos = jnp.zeros((K,), dtype=internal.rbcc.dtype)
        Z_sin = jnp.zeros((K,), dtype=internal.rbcc.dtype)

        rbcc = internal.rbcc
        rbss = internal.rbss if lthreed else jnp.zeros_like(internal.rbcc)
        rbcs = internal.rbcs if (lthreed and lasym) else jnp.zeros_like(internal.rbcc)
        rbsc = internal.rbsc if lasym else jnp.zeros_like(internal.rbcc)
        zbcc = internal.zbcc if lasym else jnp.zeros_like(internal.rbcc)
        zbss = internal.zbss if (lthreed and lasym) else jnp.zeros_like(internal.rbcc)
        zbcs = internal.zbcs if lthreed else jnp.zeros_like(internal.rbcc)
        zbsc = internal.zbsc

        m_nonneg = m_arr >= 0
        n_pos = n_arr > 0
        n_zero = n_arr == 0
        n_nonzero = n_arr != 0
        ni = jnp.abs(n_arr)

        rbcc_k = rbcc[ni, m_arr]
        rbss_k = rbss[ni, m_arr]
        rbcs_k = rbcs[ni, m_arr]
        rbsc_k = rbsc[ni, m_arr]
        zbcc_k = zbcc[ni, m_arr]
        zbss_k = zbss[ni, m_arr]
        zbcs_k = zbcs[ni, m_arr]
        zbsc_k = zbsc[ni, m_arr]

        isgn = jnp.sign(n_arr).astype(rbcc_k.dtype)

        # Case m=0 and n!=0
        mask_m0_nnz = (m_arr == 0) & n_nonzero & m_nonneg
        R_cos = jnp.where(mask_m0_nnz, rbcc_k, R_cos)
        Z_sin = jnp.where(mask_m0_nnz, -isgn * zbcs_k, Z_sin)
        if lasym:
            R_sin = jnp.where(mask_m0_nnz, -isgn * rbcs_k, R_sin)
            Z_cos = jnp.where(mask_m0_nnz, zbcc_k, Z_cos)

        # n==0 (m>=0)
        mask_n0 = n_zero & m_nonneg
        R_cos = jnp.where(mask_n0, rbcc_k, R_cos)
        Z_sin = jnp.where(mask_n0, zbsc_k, Z_sin)
        if lasym:
            R_sin = jnp.where(mask_n0, rbsc_k, R_sin)
            Z_cos = jnp.where(mask_n0, zbcc_k, Z_cos)

        # n>0 (exclude m=0 special-case so we don't overwrite it)
        mask_np = n_pos & m_nonneg & ~mask_n0 & ~mask_m0_nnz
        R_cos = jnp.where(mask_np, 0.5 * (rbcc_k + rbss_k), R_cos)
        R_sin = jnp.where(mask_np, 0.5 * (rbsc_k - rbcs_k), R_sin)
        Z_cos = jnp.where(mask_np, 0.5 * (zbcc_k + zbss_k), Z_cos)
        Z_sin = jnp.where(mask_np, 0.5 * (zbsc_k - zbcs_k), Z_sin)

        # n<0 (exclude m=0 special-case so we don't overwrite it)
        mask_nn = (n_arr < 0) & m_nonneg & ~mask_n0 & ~mask_m0_nnz
        R_cos = jnp.where(mask_nn, 0.5 * (rbcc_k - rbss_k), R_cos)
        R_sin = jnp.where(mask_nn, 0.5 * (rbsc_k + rbcs_k), R_sin)
        Z_cos = jnp.where(mask_nn, 0.5 * (zbcc_k - zbss_k), Z_cos)
        Z_sin = jnp.where(mask_nn, 0.5 * (zbsc_k + zbcs_k), Z_sin)

        # Zero out m<0
        R_cos = jnp.where(m_nonneg, R_cos, 0.0)
        R_sin = jnp.where(m_nonneg, R_sin, 0.0)
        Z_cos = jnp.where(m_nonneg, Z_cos, 0.0)
        Z_sin = jnp.where(m_nonneg, Z_sin, 0.0)

        return BoundaryCoeffs(R_cos=R_cos, R_sin=R_sin, Z_cos=Z_cos, Z_sin=Z_sin)

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
    if _is_jax_array(internal.rbcc):
        rbcc = internal.rbcc
        rbss = internal.rbss
        rbcs = internal.rbcs
        rbsc = internal.rbsc
        zbcc = internal.zbcc
        zbss = internal.zbss
        zbcs = internal.zbcs
        zbsc = internal.zbsc
        if internal.rbcc.shape[1] > 1:
            if lthreed:
                temp = rbss[:, 1]
                rbss = rbss.at[:, 1].set(0.5 * (temp + zbcs[:, 1]))
                zbcs = zbcs.at[:, 1].set(0.5 * (temp - zbcs[:, 1]))
            if lasym:
                temp = rbsc[:, 1]
                rbsc = rbsc.at[:, 1].set(0.5 * (temp + zbcc[:, 1]))
                zbcc = zbcc.at[:, 1].set(0.5 * (temp - zbcc[:, 1]))
        internal = BoundaryInternalCoeffs(
            rbcc=rbcc,
            rbss=rbss,
            rbcs=rbcs,
            rbsc=rbsc,
            zbcc=zbcc,
            zbss=zbss,
            zbcs=zbcs,
            zbsc=zbsc,
        )
        return _boundary_helical_from_internal(internal, modes, lthreed=lthreed, lasym=lasym)
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
    if _is_jax_array(internal.rbcc):
        rbcc = internal.rbcc
        rbss = internal.rbss
        rbcs = internal.rbcs
        rbsc = internal.rbsc
        zbcc = internal.zbcc
        zbss = internal.zbss
        zbcs = internal.zbcs
        zbsc = internal.zbsc
        if internal.rbcc.shape[1] > 1:
            if lthreed:
                temp = rbss[:, 1]
                rbss = rbss.at[:, 1].set(temp + zbcs[:, 1])
                zbcs = zbcs.at[:, 1].set(temp - zbcs[:, 1])
            if lasym:
                temp = rbsc[:, 1]
                rbsc = rbsc.at[:, 1].set(temp + zbcc[:, 1])
                zbcc = zbcc.at[:, 1].set(temp - zbcc[:, 1])
        internal = BoundaryInternalCoeffs(
            rbcc=rbcc,
            rbss=rbss,
            rbcs=rbcs,
            rbsc=rbsc,
            zbcc=zbcc,
            zbss=zbss,
            zbcs=zbcs,
            zbsc=zbsc,
        )
        return _boundary_helical_from_internal(internal, modes, lthreed=lthreed, lasym=lasym)
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
    rbc, rbs, zbc, zbs = _rotate_lasym_input_maps(indata=indata, rbc=rbc, rbs=rbs, zbc=zbc, zbs=zbs)

    lasym = bool(indata.get_bool("LASYM", False))

    cache_key = _boundary_cache_key(
        indata,
        modes,
        apply_m1_constraint=apply_m1_constraint,
        rbc=rbc,
        rbs=rbs,
        zbc=zbc,
        zbs=zbs,
    )
    cached = _boundary_cache_get(cache_key)
    if cached is not None:
        return cached

    boundary_input = boundary_input_from_indata(indata, modes)
    boundary_out = boundary_from_input_convention(
        boundary_input,
        modes,
        lasym=lasym,
        apply_m1_constraint=bool(apply_m1_constraint and indata.get_bool("LCONM1", True)),
    )
    _boundary_cache_put(cache_key, boundary_out)
    return boundary_out
