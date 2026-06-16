"""Analytic circular-coil validation helpers for mirror geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.boundary import MirrorBoundary

MU0 = 4.0e-7 * np.pi


@dataclass(frozen=True)
class AxisymmetricFieldRZ:
    """Axisymmetric vacuum field components in cylindrical coordinates."""

    br: np.ndarray
    bz: np.ndarray
    bmag: np.ndarray


def circular_loop_on_axis_bz(z_m, *, loop_radius_m: float, current_a: float, loop_z_m: float = 0.0) -> np.ndarray:
    """Return the analytic on-axis ``B_z`` of one circular current loop."""
    z = np.asarray(z_m, dtype=float) - float(loop_z_m)
    radius = float(loop_radius_m)
    current = float(current_a)
    if radius <= 0.0:
        raise ValueError("loop_radius_m must be positive")
    return MU0 * current * radius**2 / (2.0 * (radius**2 + z**2) ** 1.5)


def circular_loop_field_rz(radius_m, z_rel_m, *, loop_radius_m: float, current_a: float) -> AxisymmetricFieldRZ:
    """Evaluate one circular current loop field at cylindrical ``(r, z_rel)`` points."""
    from scipy.special import ellipe, ellipk

    r = np.asarray(radius_m, dtype=float)
    z = np.asarray(z_rel_m, dtype=float)
    r, z = np.broadcast_arrays(r, z)
    loop_radius = float(loop_radius_m)
    current = float(current_a)
    if loop_radius <= 0.0:
        raise ValueError("loop_radius_m must be positive")
    br = np.zeros_like(r, dtype=float)
    bz = np.zeros_like(r, dtype=float)

    on_axis = np.abs(r) < 1.0e-14
    bz[on_axis] = circular_loop_on_axis_bz(
        z[on_axis],
        loop_radius_m=loop_radius,
        current_a=current,
    )

    off_axis = ~on_axis
    if np.any(off_axis):
        rr = r[off_axis]
        zz = z[off_axis]
        alpha2 = (loop_radius + rr) ** 2 + zz**2
        beta2 = (loop_radius - rr) ** 2 + zz**2
        k2 = np.clip(4.0 * loop_radius * rr / alpha2, 0.0, 1.0 - 1.0e-15)
        ellip_k = ellipk(k2)
        ellip_e = ellipe(k2)
        common = MU0 * current / (2.0 * np.pi * np.sqrt(alpha2))
        br[off_axis] = common * zz / rr * (-ellip_k + (loop_radius**2 + rr**2 + zz**2) * ellip_e / beta2)
        bz[off_axis] = common * (ellip_k + (loop_radius**2 - rr**2 - zz**2) * ellip_e / beta2)
    return AxisymmetricFieldRZ(br=br, bz=bz, bmag=np.sqrt(br**2 + bz**2))


def two_coil_on_axis_bz(
    z_m,
    *,
    coil_radius_m: float,
    separation_m: float,
    current_a: float,
    center_z_m: float = 0.0,
) -> np.ndarray:
    """Return the summed on-axis ``B_z`` from two equal circular coils."""
    separation = float(separation_m)
    if separation <= 0.0:
        raise ValueError("separation_m must be positive")
    center = float(center_z_m)
    half_separation = 0.5 * separation
    return circular_loop_on_axis_bz(
        z_m,
        loop_radius_m=coil_radius_m,
        current_a=current_a,
        loop_z_m=center - half_separation,
    ) + circular_loop_on_axis_bz(
        z_m,
        loop_radius_m=coil_radius_m,
        current_a=current_a,
        loop_z_m=center + half_separation,
    )


def two_coil_field_rz(
    radius_m,
    z_m,
    *,
    coil_radius_m: float,
    separation_m: float,
    current_a: float,
    center_z_m: float = 0.0,
) -> AxisymmetricFieldRZ:
    """Return the summed cylindrical field from two equal circular coils."""
    separation = float(separation_m)
    if separation <= 0.0:
        raise ValueError("separation_m must be positive")
    r = np.asarray(radius_m, dtype=float)
    z = np.asarray(z_m, dtype=float)
    r, z = np.broadcast_arrays(r, z)
    center = float(center_z_m)
    half_separation = 0.5 * separation
    left = circular_loop_field_rz(
        r,
        z - (center - half_separation),
        loop_radius_m=coil_radius_m,
        current_a=current_a,
    )
    right = circular_loop_field_rz(
        r,
        z - (center + half_separation),
        loop_radius_m=coil_radius_m,
        current_a=current_a,
    )
    br = left.br + right.br
    bz = left.bz + right.bz
    return AxisymmetricFieldRZ(br=br, bz=bz, bmag=np.sqrt(br**2 + bz**2))


def on_axis_mirror_ratio(bz_axis) -> float:
    """Return ``max(abs(B_z)) / min(abs(B_z))`` for on-axis field samples."""
    bmag = np.abs(np.asarray(bz_axis, dtype=float))
    if bmag.ndim != 1 or bmag.size < 2:
        raise ValueError("bz_axis must be a one-dimensional array with at least two samples")
    if np.any(bmag <= 0.0):
        raise ValueError("bz_axis must be nonzero at every sample")
    return float(np.max(bmag) / np.min(bmag))


def two_coil_on_axis_mirror_ratio(
    *,
    coil_radius_m: float,
    separation_m: float,
    current_a: float,
    center_z_m: float = 0.0,
    num_points: int = 257,
) -> float:
    """Return the two-coil on-axis mirror ratio between coil centers."""
    center = float(center_z_m)
    half_separation = 0.5 * float(separation_m)
    z = np.linspace(center - half_separation, center + half_separation, int(num_points))
    return on_axis_mirror_ratio(
        two_coil_on_axis_bz(
            z,
            coil_radius_m=coil_radius_m,
            separation_m=separation_m,
            current_a=current_a,
            center_z_m=center_z_m,
        )
    )


def mirror_boundary_from_on_axis_bz(
    psi_value: float,
    z_grid,
    bz_axis,
    *,
    radius_floor: float = 1.0e-4,
) -> MirrorBoundary:
    """Build an axisymmetric fixed boundary from a near-axis flux-tube model."""
    z = np.asarray(z_grid, dtype=float)
    bz = np.asarray(bz_axis, dtype=float)
    if z.ndim != 1 or z.size < 2:
        raise ValueError("z_grid must be a one-dimensional grid with at least two nodes")
    if bz.shape != z.shape:
        raise ValueError("bz_axis must have the same shape as z_grid")
    if not np.all(np.diff(z) > 0.0):
        raise ValueError("z_grid must be strictly increasing")
    if psi_value <= 0.0:
        raise ValueError("psi_value must be positive")
    bmag = np.maximum(np.abs(bz), np.finfo(float).tiny)
    radius = np.maximum(np.sqrt(2.0 * float(psi_value) / bmag), float(radius_floor))
    xi = 2.0 * (z - z[0]) / (z[-1] - z[0]) - 1.0
    return MirrorBoundary.tabulated_radius(xi, radius)


def mirror_boundary_from_two_coil_flux_tube(
    psi_value: float,
    z_grid,
    *,
    coil_radius_m: float,
    separation_m: float,
    current_a: float,
    center_z_m: float = 0.0,
    radius_floor: float = 1.0e-4,
) -> MirrorBoundary:
    """Build a fixed boundary from the analytic on-axis two-coil vacuum field."""
    bz = two_coil_on_axis_bz(
        z_grid,
        coil_radius_m=coil_radius_m,
        separation_m=separation_m,
        current_a=current_a,
        center_z_m=center_z_m,
    )
    return mirror_boundary_from_on_axis_bz(psi_value, z_grid, bz, radius_floor=radius_floor)
