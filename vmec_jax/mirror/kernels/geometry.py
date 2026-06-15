"""Axisymmetric mirror geometry kernels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AxisymMirrorGeometry:
    """Axisymmetric embedding, metrics, and quadrature diagnostics."""

    r: np.ndarray
    z: np.ndarray
    r_xi: np.ndarray
    r_r_s: np.ndarray
    sqrtg: np.ndarray
    g_ss: np.ndarray
    g_stheta: np.ndarray
    g_sxi: np.ndarray
    g_thetatheta: np.ndarray
    g_thetaxi: np.ndarray
    g_xixi: np.ndarray
    volume: float


def _radial_derivative(values, s_full):
    edge_order = 2 if len(s_full) > 2 else 1
    return np.gradient(values, s_full, axis=0, edge_order=edge_order)


def evaluate_axisym_geometry(state, grid) -> AxisymMirrorGeometry:
    """Evaluate straight-axis axisymmetric mirror geometry.

    The coordinate ``s = rho**2`` is singular at the axis for ``g_ss``.  The
    Jacobian and mixed metric term are computed through regular products
    ``r*r_s`` and ``r_s*r_xi``; the axis value of ``g_ss`` is regularized from
    the first off-axis node for finite diagnostics.
    """
    if state.a.shape != (grid.ns, grid.nxi):
        raise ValueError(f"state shape {state.a.shape} does not match grid {(grid.ns, grid.nxi)}")

    a = np.asarray(state.a, dtype=float)
    rho = grid.rho_full[:, None]
    r = rho * a
    z = grid.z
    a_xi = grid.axial_basis.differentiate(a, axis=1)
    r_xi = rho * a_xi

    r_squared = r**2
    d_r_squared_ds = _radial_derivative(r_squared, grid.s_full)
    r_r_s = 0.5 * d_r_squared_ds
    sqrtg = r_r_s * grid.z_xi

    with np.errstate(divide="ignore", invalid="ignore"):
        g_ss = np.divide(r_r_s**2, r_squared, out=np.zeros_like(r_squared), where=r_squared > 0.0)
        a_xi_over_a = np.divide(a_xi, a, out=np.zeros_like(a_xi), where=a != 0.0)
    if grid.ns > 1:
        g_ss[0, :] = g_ss[1, :]

    g_stheta = np.zeros_like(r)
    g_sxi = r_r_s * a_xi_over_a
    g_thetatheta = r_squared
    g_thetaxi = np.zeros_like(r)
    g_xixi = r_xi**2 + grid.z_xi**2
    volume = float(np.einsum("i,j,k,ik->", grid.w_s, grid.w_theta, grid.w_xi, sqrtg))

    return AxisymMirrorGeometry(
        r=r,
        z=z,
        r_xi=r_xi,
        r_r_s=r_r_s,
        sqrtg=sqrtg,
        g_ss=g_ss,
        g_stheta=g_stheta,
        g_sxi=g_sxi,
        g_thetatheta=g_thetatheta,
        g_thetaxi=g_thetaxi,
        g_xixi=g_xixi,
        volume=volume,
    )
