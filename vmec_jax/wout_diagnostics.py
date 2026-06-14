"""WOUT diagnostic reconstruction helpers.

These helpers operate on persisted VMEC ``wout`` profile arrays.  They are kept
separate from the large reader/writer module so stability-diagnostic algebra can
be tested and reused without importing the full WOUT synthesis path.
"""

from __future__ import annotations

import numpy as np


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


__all__ = ["glasser_from_wout_mercier_terms"]
