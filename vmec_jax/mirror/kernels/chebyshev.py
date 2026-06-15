"""Chebyshev-Gauss-Lobatto nodes, differentiation, and quadrature."""

from __future__ import annotations

from typing import Any

import numpy as np


def _num_nodes(num_nodes: int) -> int:
    num_nodes = int(num_nodes)
    if num_nodes < 2:
        raise ValueError("Chebyshev-Gauss-Lobatto grids require at least 2 nodes")
    return num_nodes


def cosine_to_increasing_permutation(num_nodes: int) -> np.ndarray:
    """Return indices mapping cosine order ``[1, ..., -1]`` to increasing order."""
    return np.arange(_num_nodes(num_nodes) - 1, -1, -1, dtype=int)


def chebyshev_lobatto_nodes(num_nodes: int, *, dtype: Any = float, increasing: bool = True) -> np.ndarray:
    """Return Chebyshev-Gauss-Lobatto nodes on ``[-1, 1]``.

    The public default is increasing physical order, so endpoints are exactly
    ``[-1, +1]``.  This avoids the ordering ambiguity that has caused
    interpolation bugs in other Fourier-Chebyshev code paths.
    """
    num_nodes = _num_nodes(num_nodes)
    degree = num_nodes - 1
    nodes = np.cos(np.pi * np.arange(num_nodes, dtype=dtype) / dtype(degree))
    if increasing:
        nodes = nodes[cosine_to_increasing_permutation(num_nodes)]
        nodes[0] = dtype(-1.0)
        nodes[-1] = dtype(1.0)
    else:
        nodes[0] = dtype(1.0)
        nodes[-1] = dtype(-1.0)
    return nodes


def chebyshev_lobatto_derivative_matrix(
    num_nodes: int, *, dtype: Any = float, increasing: bool = True
) -> np.ndarray:
    """Return the first-derivative matrix on CGL nodes."""
    num_nodes = _num_nodes(num_nodes)
    degree = num_nodes - 1
    k = np.arange(num_nodes)
    x = np.cos(np.pi * k / degree).astype(dtype)
    c = np.ones(num_nodes, dtype=dtype)
    c[0] = dtype(2.0)
    c[-1] = dtype(2.0)
    c = c * ((-1.0) ** k)
    dx = x[:, None] - x[None, :]
    derivative = (c[:, None] / c[None, :]) / (dx + np.eye(num_nodes, dtype=dtype))
    derivative = derivative - np.diag(np.sum(derivative, axis=1))
    if increasing:
        permutation = cosine_to_increasing_permutation(num_nodes)
        derivative = derivative[np.ix_(permutation, permutation)]
    return np.asarray(derivative, dtype=dtype)


def clenshaw_curtis_weights(num_nodes: int, *, dtype: Any = float, increasing: bool = True) -> np.ndarray:
    """Return Clenshaw-Curtis quadrature weights on CGL nodes."""
    num_nodes = _num_nodes(num_nodes)
    degree = num_nodes - 1
    theta = np.pi * np.arange(num_nodes, dtype=dtype) / dtype(degree)
    weights = np.zeros(num_nodes, dtype=dtype)
    interior = np.arange(1, degree)
    if degree == 1:
        weights[:] = dtype(1.0)
    else:
        values = np.ones(degree - 1, dtype=dtype)
        if degree % 2 == 0:
            weights[0] = dtype(1.0) / dtype(degree * degree - 1)
            weights[-1] = weights[0]
            for mode in range(1, degree // 2):
                values -= dtype(2.0) * np.cos(dtype(2 * mode) * theta[interior]) / dtype(4 * mode * mode - 1)
            values -= np.cos(dtype(degree) * theta[interior]) / dtype(degree * degree - 1)
        else:
            weights[0] = dtype(1.0) / dtype(degree * degree)
            weights[-1] = weights[0]
            for mode in range(1, (degree + 1) // 2):
                values -= dtype(2.0) * np.cos(dtype(2 * mode) * theta[interior]) / dtype(4 * mode * mode - 1)
        weights[interior] = dtype(2.0) * values / dtype(degree)
    if increasing:
        weights = weights[cosine_to_increasing_permutation(num_nodes)]
    return np.asarray(weights, dtype=dtype)


def chebyshev_vandermonde(nodes, degree: int | None = None) -> np.ndarray:
    """Return a Chebyshev Vandermonde matrix through ``degree``."""
    nodes = np.asarray(nodes)
    if degree is None:
        degree = int(nodes.size) - 1
    return np.polynomial.chebyshev.chebvander(nodes, int(degree))


def chebyshev_values_to_coefficients(values, *, nodes=None, axis: int = -1) -> np.ndarray:
    """Convert nodal values to Chebyshev coefficients by interpolation."""
    values = np.asarray(values)
    axis = axis % values.ndim
    if nodes is None:
        nodes = chebyshev_lobatto_nodes(values.shape[axis], dtype=values.dtype)
    nodes = np.asarray(nodes, dtype=values.dtype)
    if nodes.size != values.shape[axis]:
        raise ValueError("nodes length must match the interpolation axis")
    vandermonde = chebyshev_vandermonde(nodes)
    moved = np.moveaxis(values, axis, -1)
    rhs = moved.reshape(-1, nodes.size).T
    coeffs = np.linalg.solve(vandermonde, rhs).T.reshape(moved.shape)
    return np.moveaxis(coeffs, -1, axis)


def chebyshev_coefficients_to_values(coefficients, *, nodes=None, axis: int = -1) -> np.ndarray:
    """Evaluate Chebyshev coefficients at ``nodes``."""
    coefficients = np.asarray(coefficients)
    axis = axis % coefficients.ndim
    if nodes is None:
        nodes = chebyshev_lobatto_nodes(coefficients.shape[axis], dtype=coefficients.dtype)
    nodes = np.asarray(nodes, dtype=coefficients.dtype)
    vandermonde = chebyshev_vandermonde(nodes, coefficients.shape[axis] - 1)
    moved = np.moveaxis(coefficients, axis, -1)
    values = moved.reshape(-1, moved.shape[-1]) @ vandermonde.T
    return np.moveaxis(values.reshape(moved.shape[:-1] + (nodes.size,)), -1, axis)


def chebyshev_interpolation_matrix(source_nodes, target_nodes, *, atol: float = 1.0e-14) -> np.ndarray:
    """Return a barycentric interpolation matrix between two CGL-like grids."""
    source_nodes = np.asarray(source_nodes, dtype=float)
    target_nodes = np.asarray(target_nodes, dtype=float)
    if source_nodes.ndim != 1 or target_nodes.ndim != 1:
        raise ValueError("source_nodes and target_nodes must be one-dimensional")
    if source_nodes.size < 2:
        raise ValueError("at least two source nodes are required")

    bary_weights = np.ones(source_nodes.size, dtype=float)
    for idx, node in enumerate(source_nodes):
        bary_weights[idx] = 1.0 / np.prod(node - np.delete(source_nodes, idx))

    matrix = np.empty((target_nodes.size, source_nodes.size), dtype=float)
    for row, target in enumerate(target_nodes):
        matches = np.isclose(target, source_nodes, rtol=0.0, atol=atol)
        if np.any(matches):
            matrix[row] = 0.0
            matrix[row, int(np.argmax(matches))] = 1.0
            continue
        scaled = bary_weights / (target - source_nodes)
        matrix[row] = scaled / np.sum(scaled)
    return matrix


def interpolate_chebyshev_values(values, source_nodes, target_nodes, *, axis: int = -1) -> np.ndarray:
    """Interpolate values from ``source_nodes`` to ``target_nodes`` along ``axis``."""
    values = np.asarray(values)
    matrix = chebyshev_interpolation_matrix(source_nodes, target_nodes).astype(values.dtype, copy=False)
    axis = axis % values.ndim
    moved = np.moveaxis(values, axis, 0)
    interpolated = matrix @ moved.reshape(moved.shape[0], -1)
    interpolated = interpolated.reshape((matrix.shape[0],) + moved.shape[1:])
    return np.moveaxis(interpolated, 0, axis)


def exponential_chebyshev_filter(num_modes: int, *, alpha: float = 36.0, order: int = 8, cutoff: int = 0) -> np.ndarray:
    """Return modal exponential filter factors for Chebyshev coefficients."""
    num_modes = int(num_modes)
    if num_modes < 1:
        raise ValueError("num_modes must be >= 1")
    if order <= 0:
        raise ValueError("order must be positive")
    modes = np.arange(num_modes, dtype=float)
    scale = modes / max(1, num_modes - 1)
    sigma = np.exp(-float(alpha) * scale**int(order))
    sigma[: int(cutoff) + 1] = 1.0
    return sigma


def apply_chebyshev_filter(
    values,
    *,
    nodes=None,
    alpha: float = 36.0,
    order: int = 8,
    cutoff: int = 0,
    axis: int = -1,
) -> np.ndarray:
    """Apply an exponential modal filter to CGL nodal values."""
    coefficients = chebyshev_values_to_coefficients(values, nodes=nodes, axis=axis)
    sigma = exponential_chebyshev_filter(coefficients.shape[axis], alpha=alpha, order=order, cutoff=cutoff)
    shape = [1] * coefficients.ndim
    shape[axis % coefficients.ndim] = sigma.size
    filtered = coefficients * sigma.reshape(shape)
    return chebyshev_coefficients_to_values(filtered, nodes=nodes, axis=axis)
