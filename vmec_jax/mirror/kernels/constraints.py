"""Constraint projections for mirror states."""

from __future__ import annotations

import numpy as np

from ..core.state import MirrorStateAxisym


def lambda_surface_average_axisym(lam, grid) -> np.ndarray:
    """Return the CGL-weighted lambda average on each radial surface."""
    lam = np.asarray(lam)
    return np.tensordot(lam, grid.w_xi, axes=([-1], [0])) / np.sum(grid.w_xi)


def project_lambda_gauge_axisym(lam, grid) -> np.ndarray:
    """Remove the axisymmetric ``lambda -> lambda + c(s)`` gauge freedom."""
    lam = np.asarray(lam).copy()
    average = lambda_surface_average_axisym(lam, grid)
    return lam - average[:, None]


def project_axisym_state(
    state: MirrorStateAxisym,
    grid,
    boundary,
    *,
    fix_end_surfaces: bool = True,
    fix_axis_a_from_inner_surface: bool = True,
) -> MirrorStateAxisym:
    """Project side boundary, optional fixed ends, axis regularity, and lambda gauge."""
    if state.a.shape != (grid.ns, grid.nxi):
        raise ValueError(f"state shape {state.a.shape} does not match grid {(grid.ns, grid.nxi)}")
    boundary_radius = boundary.radius_on_grid(grid)
    a = np.asarray(state.a, dtype=boundary_radius.dtype).copy()
    lam = project_lambda_gauge_axisym(state.lam, grid)

    a[-1, :] = boundary_radius
    if fix_end_surfaces:
        a[:, 0] = boundary_radius[0]
        a[:, -1] = boundary_radius[-1]
    if fix_axis_a_from_inner_surface and grid.ns > 1:
        a[0, :] = a[1, :]
    if fix_end_surfaces:
        a[0, 0] = boundary_radius[0]
        a[0, -1] = boundary_radius[-1]
    return MirrorStateAxisym(a=a, lam=lam)
