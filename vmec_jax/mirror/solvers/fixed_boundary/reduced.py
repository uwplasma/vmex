"""Reduced fixed-boundary coordinates for mirror solves."""

from __future__ import annotations

import numpy as np

from vmec_jax._compat import jnp

from ...core.boundary import MirrorBoundary
from ...core.grids import MirrorGrid
from ...core.profiles import IPrimeProfile, PressureProfile, PsiPrimeProfile
from ...core.state import MirrorState3D, MirrorStateAxisym
from ...kernels.constraints import project_axisym_state, project_state_3d
from ...kernels.forces import (
    axisym_energy_value_and_gradient,
    axisym_total_energy_jax,
    energy_value_and_gradient_3d,
)


def _scaling_key(value: str) -> str:
    key = str(value).strip().lower().replace("-", "_")
    if key in {"none", "identity", "off", "false"}:
        return "none"
    if key in {"geometry", "vmec", "vmec_like", "diagonal"}:
        return "geometry"
    raise ValueError(f"unsupported mirror reduced-coordinate scaling {value!r}")


def _sanitize_scale(scale, *, expected_size: int) -> np.ndarray:
    scale = np.asarray(scale, dtype=float).reshape(-1)
    if scale.size != int(expected_size):
        raise ValueError(f"scale vector has size {scale.size}, expected {int(expected_size)}")
    scale = np.abs(scale)
    scale[(~np.isfinite(scale)) | (scale <= np.finfo(float).tiny)] = 1.0
    return scale


def axisym_reduced_a_mask(grid: MirrorGrid) -> np.ndarray:
    """Return the independent ``a`` nodes for fixed-boundary axisymmetric solves."""
    mask = np.zeros((grid.ns, grid.nxi), dtype=bool)
    if grid.ns > 2 and grid.nxi > 2:
        mask[1:-1, 1:-1] = True
    return mask


def axisym_reduced_coordinate_scale(
    state: MirrorStateAxisym,
    grid: MirrorGrid,
    boundary: MirrorBoundary,
    *,
    mode: str = "geometry",
) -> np.ndarray:
    """Return the diagonal scale for axisymmetric reduced coordinates.

    The reduced vector contains the interior radius nodes followed by one
    gauge-fixed lambda block per radial surface.  Scaling keeps radius and
    lambda coordinates near comparable sizes before SciPy sees them.
    """
    vector_size = pack_axisym_reduced_state(state, grid, boundary).size
    if _scaling_key(mode) == "none":
        return np.ones(vector_size, dtype=float)

    boundary_radius = np.asarray(boundary.radius_on_grid(grid), dtype=float)
    radius_scale = _sanitize_scale(boundary_radius, expected_size=grid.nxi)
    a_scale = np.broadcast_to(radius_scale[None, :], (grid.ns, grid.nxi))[axisym_reduced_a_mask(grid)]
    lambda_scale = np.full(grid.ns * (grid.nxi - 1), float(np.median(radius_scale)), dtype=float)
    return _sanitize_scale(np.concatenate([a_scale, lambda_scale]), expected_size=vector_size)


def reduced_a_mask_3d(grid: MirrorGrid) -> np.ndarray:
    """Return the independent ``a`` nodes for fixed-boundary 3D solves."""
    mask = np.zeros((grid.ns, grid.ntheta, grid.nxi), dtype=bool)
    if grid.ns > 2 and grid.nxi > 2:
        mask[1:-1, :, 1:-1] = True
    return mask


def reduced_coordinate_scale_3d(
    state: MirrorState3D,
    grid: MirrorGrid,
    boundary: MirrorBoundary,
    *,
    mode: str = "geometry",
) -> np.ndarray:
    """Return the diagonal reduced-coordinate scale used by 3D mirror L-BFGS-B."""
    vector_size = pack_reduced_state_3d(state, grid, boundary).size
    if _scaling_key(mode) == "none":
        return np.ones(vector_size, dtype=float)

    boundary_radius = np.asarray(boundary.radius_on_grid_3d(grid), dtype=float)
    radius_scale = _sanitize_scale(boundary_radius, expected_size=grid.ntheta * grid.nxi).reshape(
        grid.ntheta,
        grid.nxi,
    )
    a_scale = np.broadcast_to(radius_scale[None, :, :], (grid.ns, grid.ntheta, grid.nxi))[reduced_a_mask_3d(grid)]
    lambda_scale = np.full(grid.ns * (grid.ntheta * grid.nxi - 1), float(np.median(radius_scale)), dtype=float)
    return _sanitize_scale(np.concatenate([a_scale, lambda_scale]), expected_size=vector_size)


def pack_axisym_reduced_state(state: MirrorStateAxisym, grid: MirrorGrid, boundary: MirrorBoundary) -> np.ndarray:
    """Pack independent ``a`` nodes and gauge-fixed ``lambda`` nodes."""
    projected = project_axisym_state(state, grid, boundary)
    a_values = projected.a[axisym_reduced_a_mask(grid)]
    lam_values = np.asarray(projected.lam[:, :-1], dtype=float).ravel()
    return np.concatenate([a_values, lam_values])


def axisym_reduced_bounds(grid: MirrorGrid, *, a_floor: float = 1.0e-10) -> list[tuple[float | None, float | None]]:
    """Return L-BFGS-B bounds for axisymmetric reduced coordinates."""
    num_a = int(np.count_nonzero(axisym_reduced_a_mask(grid)))
    num_lam = grid.ns * (grid.nxi - 1)
    return [(float(a_floor), None)] * num_a + [(None, None)] * num_lam


def scale_reduced_bounds(
    bounds: list[tuple[float | None, float | None]], scale: np.ndarray
) -> list[tuple[float | None, float | None]]:
    """Convert reduced-coordinate bounds into scaled optimizer coordinates."""
    scale = _sanitize_scale(scale, expected_size=len(bounds))
    scaled: list[tuple[float | None, float | None]] = []
    for (lower, upper), item_scale in zip(bounds, scale, strict=True):
        scaled.append(
            (
                None if lower is None else float(lower) / float(item_scale),
                None if upper is None else float(upper) / float(item_scale),
            )
        )
    return scaled


def _scaled_bounds(
    bounds: list[tuple[float | None, float | None]], scale: np.ndarray
) -> list[tuple[float | None, float | None]]:
    return scale_reduced_bounds(bounds, scale)


def unpack_axisym_reduced_state(vector, grid: MirrorGrid, boundary: MirrorBoundary) -> MirrorStateAxisym:
    """Reconstruct a projected axisymmetric state from reduced coordinates."""
    vector = np.asarray(vector, dtype=float)
    mask = axisym_reduced_a_mask(grid)
    num_a = int(np.count_nonzero(mask))
    expected = num_a + grid.ns * (grid.nxi - 1)
    if vector.size != expected:
        raise ValueError(f"reduced vector has size {vector.size}, expected {expected}")

    boundary_radius = boundary.radius_on_grid(grid)
    a = np.broadcast_to(boundary_radius[None, :], (grid.ns, grid.nxi)).copy()
    a[mask] = vector[:num_a]

    lam = np.zeros((grid.ns, grid.nxi), dtype=float)
    lam[:, :-1] = vector[num_a:].reshape(grid.ns, grid.nxi - 1)
    lam[:, -1] = -np.einsum("j,ij->i", grid.w_xi[:-1], lam[:, :-1]) / float(grid.w_xi[-1])
    return project_axisym_state(MirrorStateAxisym(a=a, lam=lam), grid, boundary)


def _unpack_axisym_reduced_state_jax(vector, grid: MirrorGrid, boundary: MirrorBoundary):
    boundary_radius = jnp.asarray(boundary.radius_on_grid(grid), dtype=jnp.asarray(vector).dtype)
    mask_i, mask_j = np.nonzero(axisym_reduced_a_mask(grid))
    num_a = int(mask_i.size)

    a = jnp.broadcast_to(boundary_radius[None, :], (grid.ns, grid.nxi))
    a = a.at[(mask_i, mask_j)].set(vector[:num_a])
    a = a.at[0, :].set(a[1, :])
    a = a.at[0, 0].set(boundary_radius[0])
    a = a.at[0, -1].set(boundary_radius[-1])

    lam_inner = jnp.reshape(vector[num_a:], (grid.ns, grid.nxi - 1))
    w_xi = jnp.asarray(grid.w_xi, dtype=jnp.asarray(vector).dtype)
    lam_last = -jnp.einsum("j,ij->i", w_xi[:-1], lam_inner) / w_xi[-1]
    lam = jnp.concatenate([lam_inner, lam_last[:, None]], axis=1)
    return a, lam


def _axisym_reduced_energy_jax(
    vector,
    grid: MirrorGrid,
    boundary: MirrorBoundary,
    *,
    psi_prime: PsiPrimeProfile,
    i_prime: IPrimeProfile,
    pressure: PressureProfile,
    mu0: float,
):
    a, lam = _unpack_axisym_reduced_state_jax(vector, grid, boundary)
    return axisym_total_energy_jax(
        a,
        lam,
        grid,
        psi_prime=psi_prime,
        i_prime=i_prime,
        pressure=pressure,
        mu0=mu0,
    )


def pack_reduced_state_3d(state: MirrorState3D, grid: MirrorGrid, boundary: MirrorBoundary) -> np.ndarray:
    """Pack independent 3D ``a`` nodes and gauge-fixed ``lambda`` nodes."""
    projected = project_state_3d(state, grid, boundary)
    a_values = projected.a[reduced_a_mask_3d(grid)]
    lam_values = np.asarray(projected.lam[:, :, :], dtype=float).reshape(grid.ns, -1)[:, :-1].ravel()
    return np.concatenate([a_values, lam_values])


def reduced_bounds_3d(grid: MirrorGrid, *, a_floor: float = 1.0e-10) -> list[tuple[float | None, float | None]]:
    """Return L-BFGS-B bounds for 3D reduced coordinates."""
    num_a = int(np.count_nonzero(reduced_a_mask_3d(grid)))
    num_lam = grid.ns * (grid.ntheta * grid.nxi - 1)
    return [(float(a_floor), None)] * num_a + [(None, None)] * num_lam


def unpack_reduced_state_3d(vector, grid: MirrorGrid, boundary: MirrorBoundary) -> MirrorState3D:
    """Reconstruct a projected 3D state from reduced coordinates."""
    vector = np.asarray(vector, dtype=float)
    mask = reduced_a_mask_3d(grid)
    num_a = int(np.count_nonzero(mask))
    num_lam_surface = grid.ntheta * grid.nxi - 1
    expected = num_a + grid.ns * num_lam_surface
    if vector.size != expected:
        raise ValueError(f"reduced vector has size {vector.size}, expected {expected}")

    boundary_radius = boundary.radius_on_grid_3d(grid)
    a = np.broadcast_to(boundary_radius[None, :, :], (grid.ns, grid.ntheta, grid.nxi)).copy()
    a[mask] = vector[:num_a]

    lam = np.zeros((grid.ns, grid.ntheta * grid.nxi), dtype=float)
    lam[:, :-1] = vector[num_a:].reshape(grid.ns, num_lam_surface)
    flat_weights = (grid.w_theta[:, None] * grid.w_xi[None, :]).ravel()
    lam[:, -1] = -np.einsum("j,ij->i", flat_weights[:-1], lam[:, :-1]) / float(flat_weights[-1])
    lam = lam.reshape(grid.ns, grid.ntheta, grid.nxi)
    return project_state_3d(MirrorState3D(a=a, lam=lam), grid, boundary)


def pack_axisym_reduced_gradient_components(grad_a, grad_lam, grid: MirrorGrid) -> np.ndarray:
    """Pack full-state axisymmetric gradients into reduced fixed-boundary coordinates."""
    mask = axisym_reduced_a_mask(grid)
    grad_a = np.asarray(grad_a, dtype=float).copy()
    if grid.ns > 2:
        grad_a[1, :] += grad_a[0, :]
    a_values = grad_a[mask]

    grad_lam = np.asarray(grad_lam, dtype=float)
    lam_values = grad_lam[:, :-1] - (grid.w_xi[:-1] / grid.w_xi[-1])[None, :] * grad_lam[:, -1:]
    return np.concatenate([a_values, lam_values.ravel()])


def _pack_axisym_reduced_gradient(gradient, grid: MirrorGrid) -> np.ndarray:
    return pack_axisym_reduced_gradient_components(gradient.grad_a, gradient.grad_lam, grid)


def _pack_reduced_gradient_3d(gradient, grid: MirrorGrid) -> np.ndarray:
    mask = reduced_a_mask_3d(grid)
    grad_a = np.asarray(gradient.grad_a, dtype=float).copy()
    if grid.ns > 2:
        grad_a[1, :, :] += grad_a[0, :, :]
    a_values = grad_a[mask]

    grad_lam = np.asarray(gradient.grad_lam, dtype=float).reshape(grid.ns, -1)
    flat_weights = (grid.w_theta[:, None] * grid.w_xi[None, :]).ravel()
    lam_values = grad_lam[:, :-1] - (flat_weights[:-1] / flat_weights[-1])[None, :] * grad_lam[:, -1:]
    return np.concatenate([a_values, lam_values.ravel()])


def reduced_axisym_energy_and_gradient(
    vector,
    grid: MirrorGrid,
    boundary: MirrorBoundary,
    *,
    psi_prime: PsiPrimeProfile,
    i_prime: IPrimeProfile,
    pressure: PressureProfile,
    source_a=None,
    source_lam=None,
    mu0: float = 4.0e-7 * np.pi,
) -> tuple[float, np.ndarray]:
    """Return energy and exact reduced-coordinate gradient."""
    state = unpack_axisym_reduced_state(vector, grid, boundary)
    gradient = axisym_energy_value_and_gradient(
        state,
        grid,
        psi_prime=psi_prime,
        i_prime=i_prime,
        pressure=pressure,
        mu0=mu0,
    )
    energy = float(gradient.energy)
    grad_a = np.asarray(gradient.grad_a, dtype=float)
    grad_lam = np.asarray(gradient.grad_lam, dtype=float)
    if source_a is not None:
        source_a = np.asarray(source_a, dtype=float)
        if source_a.shape != state.a.shape:
            raise ValueError(f"source_a shape {source_a.shape} does not match state shape {state.a.shape}")
        energy -= float(np.sum(source_a * state.a))
        grad_a = grad_a - source_a
    if source_lam is not None:
        source_lam = np.asarray(source_lam, dtype=float)
        if source_lam.shape != state.lam.shape:
            raise ValueError(f"source_lam shape {source_lam.shape} does not match state shape {state.lam.shape}")
        energy -= float(np.sum(source_lam * state.lam))
        grad_lam = grad_lam - source_lam
    return energy, pack_axisym_reduced_gradient_components(grad_a, grad_lam, grid)


def reduced_3d_energy_and_gradient(
    vector,
    grid: MirrorGrid,
    boundary: MirrorBoundary,
    *,
    psi_prime: PsiPrimeProfile,
    i_prime: IPrimeProfile,
    pressure: PressureProfile,
    mu0: float = 4.0e-7 * np.pi,
) -> tuple[float, np.ndarray]:
    """Return 3D energy and exact reduced-coordinate gradient."""
    state = unpack_reduced_state_3d(vector, grid, boundary)
    gradient = energy_value_and_gradient_3d(
        state,
        grid,
        psi_prime=psi_prime,
        i_prime=i_prime,
        pressure=pressure,
        mu0=mu0,
    )
    return gradient.energy, _pack_reduced_gradient_3d(gradient, grid)
