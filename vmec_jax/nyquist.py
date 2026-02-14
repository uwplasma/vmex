"""Nyquist mode/basis helpers with caching.

These utilities avoid repeatedly constructing Nyquist mode tables and
helical bases when evaluating wout-stored Nyquist fields.
"""

from __future__ import annotations

import numpy as np

from ._compat import has_jax

from .fourier import HelicalBasis, build_helical_basis
from .grids import AngleGrid
from .modes import ModeTable

_NYQ_MODE_CACHE: dict[tuple[bytes, bytes], ModeTable] = {}
_NYQ_BASIS_CACHE: dict[tuple[bytes, bytes, bytes, bytes, int], HelicalBasis] = {}


def _cache_allowed() -> bool:
    if not has_jax():
        return True
    try:
        from jax import core

        return bool(core.trace_ctx.is_top_level())
    except Exception:
        return False


def nyquist_mode_table(*, xm_nyq, xn_nyq, nfp: int) -> ModeTable:
    """Return cached Nyquist ModeTable from wout xm_nyq/xn_nyq arrays."""
    m = np.asarray(xm_nyq, dtype=int)
    n = np.asarray(xn_nyq, dtype=int)
    nfp = int(nfp)
    if nfp != 0:
        n = (n // nfp).astype(int, copy=False)
    key = (m.tobytes(), n.tobytes())
    if _cache_allowed():
        cached = _NYQ_MODE_CACHE.get(key)
        if cached is not None:
            return cached
    modes = ModeTable(m=m, n=n)
    if _cache_allowed():
        _NYQ_MODE_CACHE[key] = modes
    return modes


def nyquist_basis_from_wout(*, wout, grid: AngleGrid) -> HelicalBasis:
    """Return cached Nyquist HelicalBasis for a given wout and grid."""
    modes = nyquist_mode_table(xm_nyq=wout.xm_nyq, xn_nyq=wout.xn_nyq, nfp=wout.nfp)
    theta = np.asarray(grid.theta)
    zeta = np.asarray(grid.zeta)
    key = (modes.m.tobytes(), modes.n.tobytes(), theta.tobytes(), zeta.tobytes(), int(grid.nfp))
    if _cache_allowed():
        cached = _NYQ_BASIS_CACHE.get(key)
        if cached is not None:
            return cached
    basis = build_helical_basis(modes, grid, cache=True)
    if _cache_allowed():
        _NYQ_BASIS_CACHE[key] = basis
    return basis
