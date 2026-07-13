"""Boundary and current decision-variable parameterization.

The routines here define the canonical Fourier ordering, reversible boundary
packing, scaled AC/CURTOR controls, and Exponential Spectral Scaling used by
both finite-difference and implicit least-squares drivers.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from .input import VmecInput

__all__ = [
    "boundary_dof_names",
    "pack_boundary",
    "unpack_boundary",
]

def _dof_modes(inp: VmecInput, max_mode: int) -> list[tuple[int, int]]:
    """Canonical (m, n) list for the boundary dofs at ``max_mode``.

    ``m = 0`` keeps only ``n >= 1`` (negative-``n`` m=0 cosine modes are
    redundant, the m=0 sine modes are their sign flips, and ``RBC(0, 0)`` —
    the major radius — is held fixed to remove the trivial scale direction,
    exactly like the simsopt QS examples fix the major radius).
    """
    m_max = min(int(max_mode), int(inp.mpol) - 1)
    n_max = min(int(max_mode), int(inp.ntor))
    out: list[tuple[int, int]] = []
    for m in range(0, m_max + 1):
        for n in range(-n_max, n_max + 1):
            if m == 0 and n <= 0:
                continue
            out.append((m, n))
    return out


def boundary_dof_names(inp: VmecInput, max_mode: int) -> list[str]:
    """Human-readable labels ("RBC(n,m)" / "ZBS(n,m)", INDATA index order)."""
    modes = _dof_modes(inp, max_mode)
    return ([f"RBC({n},{m})" for (m, n) in modes]
            + [f"ZBS({n},{m})" for (m, n) in modes])


def pack_boundary(inp: VmecInput, max_mode: int) -> np.ndarray:
    """Flat boundary-dof vector ``[rbc..., zbs...]`` (see :func:`_dof_modes`).

    Inverse of :func:`unpack_boundary`; ``RBC(0,0)`` is excluded (fixed major
    radius).  Only the stellarator-symmetric ``rbc``/``zbs`` families are
    packed — lasym boundary optimization is out of scope here.
    """
    modes = _dof_modes(inp, max_mode)
    ntor = int(inp.ntor)
    rbc = np.asarray(inp.rbc, dtype=float)
    zbs = np.asarray(inp.zbs, dtype=float)
    return np.asarray([rbc[n + ntor, m] for (m, n) in modes]
                      + [zbs[n + ntor, m] for (m, n) in modes], dtype=float)


def unpack_boundary(inp: VmecInput, x, max_mode: int) -> VmecInput:
    """New :class:`VmecInput` with the boundary dofs ``x`` applied."""
    modes = _dof_modes(inp, max_mode)
    x = np.asarray(x, dtype=float).ravel()
    if x.size != 2 * len(modes):
        raise ValueError(f"expected {2 * len(modes)} dofs, got {x.size}")
    ntor = int(inp.ntor)
    rbc = np.array(inp.rbc, dtype=float, copy=True)
    zbs = np.array(inp.zbs, dtype=float, copy=True)
    for k, (m, n) in enumerate(modes):
        rbc[n + ntor, m] = x[k]
        zbs[n + ntor, m] = x[len(modes) + k]
    return dataclasses.replace(inp, rbc=rbc, zbs=zbs)


#: curtor dof storage scale (dof = CURTOR/1e6, i.e. MA) — keeps the trust
#: region O(1) alongside the boundary dofs (spec notes_r26g section 6.4).
_CURTOR_SCALE = 1.0e6


def _current_dof_setup(inp: VmecInput, current_dofs: int | None) -> tuple[int, float]:
    """Validate the optional AC/CURTOR dof block of :func:`least_squares`.

    Returns ``(k, ac_scale)``: ``k`` leading ``AC`` power-series coefficients
    are freed (0 disables the block); the dof vector then gains ``k + 1``
    trailing entries ``[ac_0/ac_scale, ..., ac_{k-1}/ac_scale,
    curtor/1e6]``.  ``ac_scale = max|AC|`` frozen from the seed input (VMEC
    normalizes the AC profile by its own edge integral, so the coefficient
    magnitude — ampere-scale for the Zenodo/self_consistent_bootstrap decks,
    O(1) for shape-normalized decks — is the right trust-region unit; the
    spec's ``|curtor|`` is the fallback when the seed AC block is all zero).
    """
    if not current_dofs:
        return 0, 1.0
    k = int(current_dofs)
    if k <= 0:
        raise ValueError(f"current_dofs must be a positive int, got {current_dofs!r}")
    if int(inp.ncurr) != 1:
        raise ValueError("current_dofs requires ncurr = 1 (prescribed current)")
    kind = str(inp.pcurr_type).strip().lower()
    if "spline" in kind or "line_segment" in kind:
        raise ValueError(
            "current_dofs requires an AC-coefficient pcurr_type (e.g. "
            f"'power_series'), got {inp.pcurr_type!r}; re-parameterize the "
            "deck (e.g. with vmec_jax.core.bootstrap.self_consistent_bootstrap, "
            "whose refit emits a power_series AC) first")
    if k > int(np.asarray(inp.ac).size):
        raise ValueError(f"current_dofs = {k} exceeds the dense AC length "
                         f"{int(np.asarray(inp.ac).size)}")
    ac_scale = float(np.max(np.abs(np.asarray(inp.ac, dtype=float))))
    if ac_scale == 0.0:
        ac_scale = max(abs(float(inp.curtor)), 1.0)
    return k, ac_scale


def _pack_current(inp: VmecInput, k: int, ac_scale: float) -> np.ndarray:
    """Scaled ``[ac_0..ac_{k-1}, curtor]`` dof block (see :func:`_current_dof_setup`)."""
    return np.concatenate([np.asarray(inp.ac, dtype=float)[:k] / ac_scale,
                           [float(inp.curtor) / _CURTOR_SCALE]])


def _apply_current(inp: VmecInput, xc, k: int, ac_scale: float) -> VmecInput:
    """New :class:`VmecInput` with the scaled current dof block ``xc`` applied."""
    xc = np.asarray(xc, dtype=float).ravel()
    if xc.size != k + 1:
        raise ValueError(f"expected {k + 1} current dofs, got {xc.size}")
    ac = np.array(inp.ac, dtype=float, copy=True)
    ac[:k] = xc[:k] * ac_scale
    return dataclasses.replace(inp, ac=ac, curtor=float(xc[k]) * _CURTOR_SCALE)


def _ess_scale(inp: VmecInput, max_mode: int, alpha: float) -> np.ndarray:
    """Exponential Spectral Scaling (ESS) trust-region weights per dof.

    ``x_scale[i] = exp(-alpha * max(|m_i|, |n_i|)) / exp(-alpha)`` — higher
    (m, n) boundary harmonics get proportionally smaller trust-region steps,
    which stabilizes staged ``max_mode`` continuation from crude seeds.
    Ported from legacy ``optimizers.fixed_boundary.parameterization.
    create_x_scale`` (the ``use_ess``/``ess_alpha`` option of the legacy
    ``least_squares_solve``); passed to scipy as ``x_scale``.
    """
    modes = _dof_modes(inp, max_mode)
    levels = np.asarray([max(abs(m), abs(n)) for (m, n) in modes] * 2, dtype=float)
    if alpha <= 0.0:
        return np.ones_like(levels)
    return np.exp(-alpha * levels) / np.exp(-alpha)


