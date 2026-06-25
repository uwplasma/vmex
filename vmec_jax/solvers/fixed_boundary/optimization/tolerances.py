"""Scalar tolerance and damping policies for iterative VMEC solvers."""

from __future__ import annotations

from typing import Any

import numpy as np


def dtype_eps(dtype: Any) -> float:
    """Return machine epsilon for a NumPy/JAX dtype."""

    return float(np.finfo(np.dtype(dtype)).eps)


def dtype_tiny(dtype: Any) -> float:
    """Return the smallest positive normal value for a NumPy/JAX dtype."""

    return float(np.finfo(np.dtype(dtype)).tiny)


def resolve_grad_tol(
    grad_tol: float | None,
    *,
    grad_rms0: float,
    dtype: Any,
) -> float:
    """Resolve a user gradient tolerance or derive one from initial scale."""

    if grad_tol is not None:
        grad_tol = float(grad_tol)
        if grad_tol < 0.0:
            raise ValueError("grad_tol must be >= 0")
        return grad_tol
    scale = max(abs(float(grad_rms0)), dtype_tiny(dtype))
    return float(np.sqrt(dtype_eps(dtype)) * scale)


def resolve_cg_tol(
    cg_tol: float | None,
    *,
    current_obj: float,
    initial_obj: float,
    target_obj: float,
    dtype: Any,
) -> float:
    """Resolve conjugate-gradient tolerance for Newton/Gauss-Newton steps."""

    if cg_tol is not None:
        cg_tol = float(cg_tol)
        if cg_tol <= 0.0:
            raise ValueError("cg_tol must be > 0")
        return cg_tol
    tiny = dtype_tiny(dtype)
    denom = max(abs(float(initial_obj)), abs(float(target_obj)), tiny)
    ratio = max(abs(float(current_obj)), tiny) / denom
    eta = ratio / (1.0 + ratio)
    return float(max(eta, np.sqrt(dtype_eps(dtype))))


def resolve_lm_damping(
    damping: float | None,
    *,
    curvature_scale: float,
    dtype: Any,
) -> float:
    """Resolve Levenberg-Marquardt damping from curvature scale."""

    if damping is not None:
        damping = float(damping)
        if damping < 0.0:
            raise ValueError("damping must be nonnegative")
        return damping
    return float(np.sqrt(dtype_eps(dtype)) * max(abs(float(curvature_scale)), dtype_tiny(dtype)))
