"""Fixed side-boundary parameterizations for mirror geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..kernels.chebyshev import interpolate_chebyshev_values


@dataclass(frozen=True)
class MirrorBoundary:
    """Axisymmetric fixed side boundary ``r_b(xi)``."""

    kind: str
    r0: float | None = None
    a2: float = 0.0
    a4: float = 0.0
    xi: np.ndarray | None = None
    radius_values: np.ndarray | None = None

    @classmethod
    def constant_radius(cls, radius: float) -> "MirrorBoundary":
        """Return a cylindrical side boundary."""
        radius = float(radius)
        if radius <= 0.0:
            raise ValueError("boundary radius must be positive")
        return cls(kind="polynomial_radius", r0=radius)

    @classmethod
    def polynomial_radius(cls, *, r0: float, a2: float = 0.0, a4: float = 0.0) -> "MirrorBoundary":
        """Return ``r_b(xi) = r0 * (1 + a2*xi**2 + a4*xi**4)``."""
        r0 = float(r0)
        if r0 <= 0.0:
            raise ValueError("r0 must be positive")
        return cls(kind="polynomial_radius", r0=r0, a2=float(a2), a4=float(a4))

    @classmethod
    def tabulated_radius(cls, xi, radius_values) -> "MirrorBoundary":
        """Return a boundary interpolated from nodal radius values."""
        xi = np.asarray(xi, dtype=float)
        radius_values = np.asarray(radius_values, dtype=float)
        if xi.ndim != 1 or radius_values.ndim != 1:
            raise ValueError("xi and radius_values must be one-dimensional")
        if xi.size != radius_values.size:
            raise ValueError("xi and radius_values must have the same length")
        if xi.size < 2:
            raise ValueError("at least two boundary nodes are required")
        if not np.all(np.diff(xi) > 0.0):
            raise ValueError("xi nodes must be strictly increasing")
        if np.any(radius_values <= 0.0):
            raise ValueError("boundary radius values must be positive")
        return cls(kind="tabulated_radius", xi=xi, radius_values=radius_values)

    def radius(self, xi, *, dtype: Any | None = None) -> np.ndarray:
        """Evaluate the boundary radius on axial nodes."""
        xi = np.asarray(xi, dtype=dtype or float)
        if self.kind == "polynomial_radius":
            radius = float(self.r0) * (1.0 + self.a2 * xi**2 + self.a4 * xi**4)
        elif self.kind == "tabulated_radius":
            radius = interpolate_chebyshev_values(self.radius_values, self.xi, xi)
        else:
            raise ValueError(f"unsupported mirror boundary kind {self.kind!r}")
        radius = np.asarray(radius, dtype=dtype or float)
        if np.any(radius <= 0.0):
            raise ValueError("boundary radius must be positive on the requested xi grid")
        return radius

    def radius_on_grid(self, grid) -> np.ndarray:
        """Evaluate the boundary radius on a ``MirrorGrid`` axial grid."""
        return self.radius(grid.xi, dtype=grid.xi.dtype)
