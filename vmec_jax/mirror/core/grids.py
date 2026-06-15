"""Static grids for mirror coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .basis import ChebyshevLobattoBasis, ThetaFourierBasis


def _trapezoid_weights(num_nodes: int, *, dtype: Any = float) -> np.ndarray:
    num_nodes = int(num_nodes)
    if num_nodes < 2:
        raise ValueError("radial grid requires ns >= 2")
    spacing = 1.0 / float(num_nodes - 1)
    weights = np.full(num_nodes, spacing, dtype=dtype)
    weights[0] *= 0.5
    weights[-1] *= 0.5
    return weights


@dataclass(frozen=True)
class MirrorGrid:
    """Static collocation and quadrature data for ``(s, theta, xi)``."""

    s_full: np.ndarray
    s_half: np.ndarray
    rho_full: np.ndarray
    w_s: np.ndarray
    theta_basis: ThetaFourierBasis
    axial_basis: ChebyshevLobattoBasis
    z: np.ndarray
    z_xi: float

    @property
    def ns(self) -> int:
        return int(self.s_full.size)

    @property
    def ntheta(self) -> int:
        return self.theta_basis.ntheta

    @property
    def nxi(self) -> int:
        return self.axial_basis.num_nodes

    @property
    def theta(self) -> np.ndarray:
        return self.theta_basis.theta

    @property
    def xi(self) -> np.ndarray:
        return self.axial_basis.nodes

    @property
    def w_theta(self) -> np.ndarray:
        return self.theta_basis.weights

    @property
    def w_xi(self) -> np.ndarray:
        return self.axial_basis.weights

    @property
    def quadrature_shape(self) -> tuple[int, int, int]:
        return (self.ns, self.ntheta, self.nxi)


def make_mirror_grid(
    *,
    ns: int,
    ntheta: int = 1,
    nxi: int = 33,
    mpol: int = 0,
    z_min: float = -1.0,
    z_max: float = 1.0,
    dtype: Any = float,
) -> MirrorGrid:
    """Build the first mirror grid: finite radial, Fourier theta, CGL axial."""
    ns = int(ns)
    nxi = int(nxi)
    z_min = float(z_min)
    z_max = float(z_max)
    if ns < 2:
        raise ValueError("ns must be >= 2")
    if nxi < 2:
        raise ValueError("nxi must be >= 2")
    if not z_max > z_min:
        raise ValueError("z_max must be greater than z_min")

    s_full = np.linspace(0.0, 1.0, ns, dtype=dtype)
    axial_basis = ChebyshevLobattoBasis.from_num_nodes(nxi, dtype=dtype)
    z_mid = 0.5 * (z_min + z_max)
    z_xi = 0.5 * (z_max - z_min)
    return MirrorGrid(
        s_full=s_full,
        s_half=0.5 * (s_full[:-1] + s_full[1:]),
        rho_full=np.sqrt(s_full),
        w_s=_trapezoid_weights(ns, dtype=dtype),
        theta_basis=ThetaFourierBasis.from_resolution(ntheta, mpol=mpol, dtype=dtype),
        axial_basis=axial_basis,
        z=np.asarray(z_mid + z_xi * axial_basis.nodes, dtype=dtype),
        z_xi=float(z_xi),
    )
