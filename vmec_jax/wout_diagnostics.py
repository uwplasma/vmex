"""WOUT diagnostic reconstruction helpers.

These helpers operate on persisted VMEC ``wout`` profile arrays.  They are kept
separate from the large reader/writer module so stability-diagnostic algebra can
be tested and reused without importing the full WOUT synthesis path.
"""

from __future__ import annotations

import numpy as np


def pshalf_from_s(s_full: np.ndarray) -> np.ndarray:
    """Return VMEC half-mesh ``sqrt(s)`` values used in parity formulas."""

    s_arr = np.asarray(s_full, dtype=float)
    if s_arr.shape[0] < 2:
        return np.sqrt(np.maximum(s_arr, 0.0))
    sh = 0.5 * (s_arr[1:] + s_arr[:-1])
    p = np.concatenate([sh[:1], sh], axis=0)
    return np.sqrt(np.maximum(p, 0.0))


def lambda_half_mesh_weights(s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return VMEC Fortran-style ``sm``/``sp`` weights for lambda half-mesh maps."""

    s_arr = np.asarray(s, dtype=float).reshape(-1)
    ns = int(s_arr.shape[0])
    if ns < 2:
        return np.zeros((ns + 1,), dtype=float), np.zeros((ns + 1,), dtype=float)

    hs = float(s_arr[1] - s_arr[0])
    sqrts_f = np.zeros((ns + 1,), dtype=float)
    shalf_f = np.zeros((ns + 1,), dtype=float)
    for i in range(1, ns + 1):
        sqrts_f[i] = np.sqrt(max(hs * float(i - 1), 0.0))
        shalf_f[i] = np.sqrt(hs * abs(float(i) - 1.5))
    sqrts_f[ns] = 1.0

    sm_f = np.zeros((ns + 1,), dtype=float)
    sp_f = np.zeros((ns + 1,), dtype=float)
    for i in range(2, ns + 1):
        sm_f[i] = shalf_f[i] / sqrts_f[i] if sqrts_f[i] != 0.0 else 0.0
        if i < ns:
            sp_f[i] = shalf_f[i + 1] / sqrts_f[i] if sqrts_f[i] != 0.0 else 0.0
        else:
            sp_f[i] = 1.0 / sqrts_f[i] if sqrts_f[i] != 0.0 else 0.0
    sm_f[1] = 0.0
    sp_f[0] = 0.0
    sp_f[1] = sm_f[2] if ns >= 2 else 0.0
    return sm_f, sp_f


def safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Divide with VMEC's zero-denominator convention for diagnostic scalars."""

    den_safe = np.where(np.abs(den) > 0.0, den, 1.0)
    return num / den_safe


def glasser_from_wout_mercier_terms(
    *,
    DMerc: np.ndarray,
    Dshear: np.ndarray,
    Dcurr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return Glasser profiles from persisted VMEC Mercier components.

    Wout files do not store the full Mercier surface integrals needed to
    reconstruct the preferred state-level ``H`` expression.  For persistence
    and old-file fallback we use the equivalent current-term reconstruction
    ``H = -Dcurr`` and ``S^2 = 4*Dshear``.
    """

    dmerc = np.asarray(DMerc, dtype=float)
    dshear = np.asarray(Dshear, dtype=float)
    dcurr = np.asarray(Dcurr, dtype=float)
    shear2 = np.maximum(4.0 * dshear, 0.0)
    h_term = -dcurr
    valid = shear2 > 0.0
    denom = np.where(valid, shear2, 1.0)
    correction = np.where(valid, (h_term - 0.5 * shear2) ** 2 / denom, 0.0)
    d_r = np.where(valid, -dmerc + correction, 0.0)
    return (
        np.asarray(d_r, dtype=float),
        np.asarray(h_term, dtype=float),
        np.asarray(correction, dtype=float),
        np.asarray(valid, dtype=bool),
    )


__all__ = [
    "glasser_from_wout_mercier_terms",
    "lambda_half_mesh_weights",
    "pshalf_from_s",
    "safe_divide",
]
