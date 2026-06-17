"""Residual preconditioners for fixed-boundary mirror solves."""

from __future__ import annotations

import numpy as np

from ...core.grids import MirrorGrid
from .reduced import axisym_reduced_a_mask
from .types import OptimizerOptions


def _residual_preconditioner_key(value: str) -> str:
    key = str(value).strip().lower().replace("-", "_")
    if key in {"none", "identity", "off", "false"}:
        return "none"
    if key in {"radial_tridi", "radial", "vmec", "vmec_like"}:
        return "radial_tridi"
    if key in {"radial_xi_tridi", "open_xi_tridi", "radial_tridi_xi"}:
        return "radial_xi_tridi"
    raise ValueError(f"unsupported mirror residual preconditioner {value!r}")


def _residual_linear_maxiter_policy_key(value: str) -> str:
    key = str(value).strip().lower().replace("-", "_")
    if key in {"fixed", "constant", "manual", "off", "false"}:
        return "fixed"
    if key in {"adaptive", "auto", "automatic", "resolution"}:
        return "adaptive"
    raise ValueError(f"unsupported mirror residual linear maxiter policy {value!r}")


def _validate_smoothing_alpha(alpha: float, *, name: str) -> float:
    alpha = float(alpha)
    if not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return alpha


def _tridiagonal_smooth_zero_dirichlet(values: np.ndarray, *, alpha: float, axis: int = 0) -> np.ndarray:
    """Apply a symmetric zero-Dirichlet tridiagonal smoother along one axis."""
    alpha = _validate_smoothing_alpha(alpha, name="alpha")
    values = np.asarray(values, dtype=float)
    if alpha == 0.0 or values.size == 0:
        return values.copy()
    moved = np.moveaxis(values, axis, 0)
    original_shape = moved.shape
    n = int(original_shape[0])
    if n == 0:
        return values.copy()
    rhs = moved.reshape(n, -1).copy()
    if n == 1:
        solved = rhs / (1.0 + 2.0 * alpha)
        return np.moveaxis(solved.reshape(original_shape), 0, axis)

    lower = np.full(n - 1, -alpha, dtype=float)
    diag = np.full(n, 1.0 + 2.0 * alpha, dtype=float)
    upper = np.full(n - 1, -alpha, dtype=float)

    # Thomas elimination for many right-hand sides sharing one SPD tridiagonal matrix.
    c_prime = np.empty_like(upper)
    d_prime = np.empty_like(rhs)
    c_prime[0] = upper[0] / diag[0]
    d_prime[0] = rhs[0] / diag[0]
    for row in range(1, n):
        denom = diag[row] - lower[row - 1] * c_prime[row - 1]
        if row < n - 1:
            c_prime[row] = upper[row] / denom
        d_prime[row] = (rhs[row] - lower[row - 1] * d_prime[row - 1]) / denom

    solved = np.empty_like(rhs)
    solved[-1] = d_prime[-1]
    for row in range(n - 2, -1, -1):
        solved[row] = d_prime[row] - c_prime[row] * solved[row + 1]
    return np.moveaxis(solved.reshape(original_shape), 0, axis)


def axisym_reduced_residual_preconditioner(
    vector: np.ndarray,
    grid: MirrorGrid,
    *,
    kind: str = "radial_xi_tridi",
    radial_alpha: float = 0.5,
    lambda_alpha: float = 0.5,
    xi_alpha: float = 0.2,
) -> np.ndarray:
    """Apply a VMEC-like reduced-coordinate residual preconditioner.

    The mirror reduced vector contains interior radius ``a`` nodes followed by
    gauge-fixed ``lambda`` nodes.  The tridiagonal smoothers approximate the
    regular VMEC idea of radial preconditioning while respecting open mirror
    end caps.
    """
    key = _residual_preconditioner_key(kind)
    vector = np.asarray(vector, dtype=float).reshape(-1)
    if key == "none":
        return vector.copy()
    radial_alpha = _validate_smoothing_alpha(radial_alpha, name="radial_alpha")
    lambda_alpha = _validate_smoothing_alpha(lambda_alpha, name="lambda_alpha")
    xi_alpha = _validate_smoothing_alpha(xi_alpha, name="xi_alpha")

    num_a = int(np.count_nonzero(axisym_reduced_a_mask(grid)))
    expected_size = num_a + grid.ns * (grid.nxi - 1)
    if vector.size != expected_size:
        raise ValueError(f"preconditioner vector has size {vector.size}, expected {expected_size}")

    a_values = vector[:num_a]
    lam_values = vector[num_a:].reshape(grid.ns, grid.nxi - 1)
    if num_a:
        a_values = a_values.reshape(grid.ns - 2, grid.nxi - 2)
        if radial_alpha > 0.0:
            a_values = _tridiagonal_smooth_zero_dirichlet(a_values, alpha=radial_alpha, axis=0)
        if key == "radial_xi_tridi" and xi_alpha > 0.0:
            a_values = _tridiagonal_smooth_zero_dirichlet(a_values, alpha=xi_alpha, axis=1)
        a_values = a_values.ravel()
    if lambda_alpha > 0.0:
        lam_values = _tridiagonal_smooth_zero_dirichlet(lam_values, alpha=lambda_alpha, axis=0)
    return np.concatenate([a_values, lam_values.ravel()])


def axisym_residual_linear_maxiter(
    options: OptimizerOptions,
    grid: MirrorGrid,
    *,
    vector_size: int,
    residual_norm: float,
) -> int:
    """Return the effective LSMR iteration budget for one residual-Newton step."""
    base = max(1, int(options.residual_linear_maxiter))
    policy = _residual_linear_maxiter_policy_key(options.residual_linear_maxiter_policy)
    if policy == "fixed":
        return base

    factor = float(options.residual_linear_adaptive_factor)
    if not np.isfinite(factor) or factor <= 0.0:
        raise ValueError("residual_linear_adaptive_factor must be a finite positive number")
    target = max(float(options.tolerance), np.finfo(float).eps)
    residual_norm = float(residual_norm)
    if not np.isfinite(residual_norm):
        return min(max(base, int(np.ceil(factor * max(int(grid.ns), int(grid.nxi))))), max(1, int(vector_size)))
    if residual_norm <= target:
        return base
    resolution_budget = int(np.ceil(factor * max(int(grid.ns), int(grid.nxi))))
    return min(max(base, resolution_budget), max(1, int(vector_size)))
