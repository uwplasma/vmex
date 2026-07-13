"""Nyquist-grid limits and VMEC output-normalization tables.

These functions implement the grid bookkeeping from VMEC2000 ``fixaray.f``.
They are shared by WOUT transforms, current diagnostics, and Mercier output,
but contain no field or stability physics themselves.
"""

from __future__ import annotations

import numpy as np

from .fourier import ModeTable, TrigTables, mode_table

__all__ = ["nyquist_limits", "nyquist_mode_table_from_grid"]


def nyquist_limits(trig: TrigTables) -> tuple[int, int]:
    """Return geometric grid cutoffs ``(mnyq, nnyq)`` (``fixaray.f``)."""
    ntheta2 = int(trig.ntheta2)
    nzeta = int(np.asarray(trig.cosnv).shape[0])
    return max(ntheta2 - 1, 0), max(nzeta // 2, 0)


def nyquist_mode_table_from_grid(
    *, mpol: int, ntor: int, ntheta: int, nzeta: int
) -> ModeTable:
    """Build the Nyquist ``(m, n)`` table from angular grid sizes."""
    ntheta1 = 2 * (int(ntheta) // 2)
    mnyq = max(ntheta1 // 2, max(int(mpol) - 1, 0))
    nnyq = max(int(nzeta) // 2, max(int(ntor), 0))
    return mode_table(mnyq + 1, nnyq)


def _analysis_theta_tables(trig: TrigTables) -> tuple[np.ndarray, np.ndarray]:
    """Return theta analysis tables in VMEC's output normalization.

    ``fixaray.f`` uses the full theta grid for asymmetric output and the
    endpoint-half-weighted reduced grid for stellarator-symmetric output.
    """
    ntheta2 = int(trig.ntheta2)
    ntheta3 = int(trig.ntheta3)
    nzeta = int(np.asarray(trig.cosnv).shape[0])
    if ntheta3 > ntheta2:
        dnorm = 1.0 / (nzeta * ntheta3)
    else:
        dnorm = 1.0 / (nzeta * (ntheta2 - 1))
    cosmui = dnorm * np.asarray(trig.cosmu, dtype=float)[:ntheta2, :].copy()
    sinmui = dnorm * np.asarray(trig.sinmu, dtype=float)[:ntheta2, :].copy()
    cosmui[0, :] *= 0.5
    cosmui[ntheta2 - 1, :] *= 0.5
    return cosmui, sinmui


def _pshalf_from_s(s_full: np.ndarray) -> np.ndarray:
    """Return half-mesh ``sqrt(s)`` with the first interior value on axis."""
    s_array = np.asarray(s_full, dtype=float)
    if s_array.shape[0] < 2:
        return np.sqrt(np.maximum(s_array, 0.0))
    s_half = 0.5 * (s_array[1:] + s_array[:-1])
    return np.sqrt(np.maximum(np.concatenate([s_half[:1], s_half]), 0.0))
