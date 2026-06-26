"""WOUT parity compatibility helpers."""

from __future__ import annotations

import os

import numpy as np


def bss_should_undo_scalxc() -> bool:
    """Whether BSS parity geometry should undo VMEC's odd-m scalxc scaling."""

    return os.getenv("VMEC_JAX_BSS_UNDO_SCALXC", "0") not in ("", "0")


def bss_scalxc_undo_factor(s: np.ndarray) -> np.ndarray:
    """Return the VMEC inverse scalxc factor for BSS odd-parity geometry."""

    s_arr = np.asarray(s, dtype=float)
    sqrts = np.sqrt(np.maximum(s_arr, 0.0))
    if s_arr.shape[0] >= 1:
        sqrts = sqrts.copy()
        sqrts[-1] = 1.0
    sq2 = sqrts[1] if s_arr.shape[0] >= 2 else 1.0
    return np.maximum(sqrts, sq2)[:, None, None]


def undo_bss_scalxc_if_enabled(s: np.ndarray, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """Undo odd-m scalxc for BSS parity arrays when the compatibility flag is set."""

    if not bss_should_undo_scalxc():
        return tuple(arrays)
    factor = bss_scalxc_undo_factor(s)
    return tuple(np.asarray(arr, dtype=float) * factor for arr in arrays)


__all__ = [
    "bss_scalxc_undo_factor",
    "bss_should_undo_scalxc",
    "undo_bss_scalxc_if_enabled",
]
