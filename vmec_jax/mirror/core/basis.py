"""Basis containers for mirror coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..kernels.chebyshev import (
    apply_chebyshev_filter,
    chebyshev_interpolation_matrix,
    chebyshev_lobatto_derivative_matrix,
    chebyshev_lobatto_nodes,
    clenshaw_curtis_weights,
    interpolate_chebyshev_values,
)
from ..kernels.fourier import (
    evaluate_real_fourier,
    evaluate_real_fourier_derivative,
    fourier_derivative,
    real_fourier_modes,
    theta_nodes,
    theta_weights,
)


@dataclass(frozen=True)
class ChebyshevLobattoBasis:
    """Chebyshev-Gauss-Lobatto axial basis in public increasing order."""

    nodes: np.ndarray
    weights: np.ndarray
    derivative_matrix: np.ndarray
    second_derivative_matrix: np.ndarray

    @classmethod
    def from_num_nodes(cls, num_nodes: int, *, dtype: Any = float) -> "ChebyshevLobattoBasis":
        """Construct CGL nodes, first/second derivative matrices, and weights."""
        nodes = chebyshev_lobatto_nodes(num_nodes, dtype=dtype)
        derivative_matrix = chebyshev_lobatto_derivative_matrix(num_nodes, dtype=dtype)
        weights = clenshaw_curtis_weights(num_nodes, dtype=dtype)
        return cls(
            nodes=nodes,
            weights=weights,
            derivative_matrix=derivative_matrix,
            second_derivative_matrix=derivative_matrix @ derivative_matrix,
        )

    @property
    def num_nodes(self) -> int:
        return int(self.nodes.size)

    def differentiate(self, values, *, axis: int = -1):
        """Differentiate nodal values along the axial CGL axis."""
        moved = np.moveaxis(np.asarray(values), axis, 0)
        derivative = np.tensordot(self.derivative_matrix, moved, axes=([1], [0]))
        return np.moveaxis(derivative, 0, axis)

    def integrate(self, values, *, axis: int = -1):
        """Integrate nodal values over ``xi in [-1, 1]``."""
        moved = np.moveaxis(np.asarray(values), axis, -1)
        return np.tensordot(moved, self.weights, axes=([-1], [0]))

    def interpolation_matrix(self, target_nodes) -> np.ndarray:
        """Return the barycentric matrix from this basis to ``target_nodes``."""
        return chebyshev_interpolation_matrix(self.nodes, target_nodes)

    def interpolate(self, values, target_nodes, *, axis: int = -1):
        """Interpolate nodal values to target axial nodes."""
        return interpolate_chebyshev_values(values, self.nodes, target_nodes, axis=axis)

    def filter(self, values, *, alpha: float = 36.0, order: int = 8, cutoff: int = 0, axis: int = -1):
        """Apply an exponential Chebyshev modal filter to nodal values."""
        return apply_chebyshev_filter(values, nodes=self.nodes, alpha=alpha, order=order, cutoff=cutoff, axis=axis)


@dataclass(frozen=True)
class ThetaFourierBasis:
    """Uniform theta grid and real Fourier modes for mirror azimuthal dependence."""

    theta: np.ndarray
    weights: np.ndarray
    modes: np.ndarray

    @classmethod
    def from_resolution(cls, ntheta: int, mpol: int = 0, *, dtype: Any = float) -> "ThetaFourierBasis":
        """Construct a theta basis with enough grid points for the requested modes."""
        ntheta = int(ntheta)
        mpol = int(mpol)
        if ntheta < 1:
            raise ValueError("ntheta must be >= 1")
        if mpol < 0:
            raise ValueError("mpol must be >= 0")
        min_points = max(1, 2 * mpol + 1)
        if ntheta < min_points:
            raise ValueError(f"ntheta={ntheta} cannot resolve mpol={mpol}; use at least {min_points} points")
        return cls(
            theta=theta_nodes(ntheta, dtype=dtype),
            weights=theta_weights(ntheta, dtype=dtype),
            modes=real_fourier_modes(mpol),
        )

    @property
    def ntheta(self) -> int:
        return int(self.theta.size)

    @property
    def mpol(self) -> int:
        return int(self.modes[-1]) if self.modes.size else 0

    def evaluate(self, cos_coeffs, sin_coeffs=None):
        """Evaluate a real Fourier series on this basis' theta grid."""
        return evaluate_real_fourier(self.theta, cos_coeffs, sin_coeffs=sin_coeffs)

    def evaluate_derivative(self, cos_coeffs, sin_coeffs=None):
        """Evaluate the theta derivative of a real Fourier series."""
        return evaluate_real_fourier_derivative(self.theta, cos_coeffs, sin_coeffs=sin_coeffs)

    def differentiate(self, values, *, axis: int = -1):
        """Differentiate periodic nodal values along theta."""
        return fourier_derivative(values, axis=axis)
