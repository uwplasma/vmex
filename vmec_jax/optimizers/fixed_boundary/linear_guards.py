"""Validation and finite-output guards for optimizer linear operators."""

from __future__ import annotations

from typing import Callable

import numpy as np


MATRIX_FREE_NONFINITE_RESIDUAL_PENALTY = 1.0e12


def linear_operator_vector_arg(value, *, size: int, name: str) -> np.ndarray:
    """Evaluate linear operator vector arg for fixed-boundary VMEC solve and implicit differentiation."""
    arr = np.asarray(value, dtype=float).reshape(-1)
    if int(arr.size) != int(size):
        raise ValueError(f"{name} expected {int(size)} entries, got {int(arr.size)}.")
    return arr


def linear_operator_matrix_arg(value, *, rows: int, name: str) -> np.ndarray:
    """Evaluate linear operator matrix arg for fixed-boundary VMEC solve and implicit differentiation."""
    arr = np.asarray(value, dtype=float)
    rows = int(rows)
    if arr.ndim != 2:
        if rows <= 0:
            if arr.size != 0:
                raise ValueError(f"{name} expected 0 rows, got {int(arr.size)} entries.")
            return arr.reshape((0, 0))
        if int(arr.size) % rows != 0:
            raise ValueError(f"{name} with {int(arr.size)} entries cannot be reshaped to {rows} rows.")
        arr = arr.reshape((rows, -1))
    if int(arr.shape[0]) != rows:
        raise ValueError(f"{name} expected {rows} rows, got {int(arr.shape[0])}.")
    return arr


def finite_linear_operator_output(
    value,
    *,
    profile_add: Callable[[str, float], None] | None = None,
    profile_name: str,
) -> np.ndarray:
    """Return finite linear-operator output, zeroing invalid tangent products."""

    arr = np.asarray(value, dtype=float)
    if np.all(np.isfinite(arr)):
        return arr
    if profile_add is not None:
        profile_add(profile_name, 0.0)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def finite_residual_vector(
    value,
    *,
    profile_add: Callable[[str, float], None] | None = None,
    profile_name: str,
    expected_size: int | None = None,
    penalty: float = MATRIX_FREE_NONFINITE_RESIDUAL_PENALTY,
) -> np.ndarray:
    """Return a finite residual vector, replacing invalid entries by a penalty."""

    arr = np.asarray(value, dtype=float).reshape(-1)
    if expected_size is not None and int(arr.size) != int(expected_size):
        if profile_add is not None:
            profile_add(profile_name, 0.0)
        return np.full(int(expected_size), float(penalty), dtype=float)
    if np.all(np.isfinite(arr)):
        return arr
    if profile_add is not None:
        profile_add(profile_name, 0.0)
    return np.nan_to_num(arr, nan=float(penalty), posinf=float(penalty), neginf=-float(penalty))
