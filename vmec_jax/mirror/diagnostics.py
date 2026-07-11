"""Physics diagnostics for solved mirror equilibria.

The free-boundary beta input scales a conserved mass profile, as in VMEC.
Geometry changes during the solve, so the requested beta and the achieved
pressure beta are related but not identical.  These helpers report both and
compare the solved on-axis field depression with paraxial pressure balance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np

MU0 = 4.0e-7 * np.pi
Array = Any


def boundary_fourier_amplitudes(boundary: "MirrorBoundary") -> Array:
    """Return real-signal theta-mode amplitudes along the mirror boundary.

    The result has shape ``(ntheta // 2 + 1, nxi)``. Mode zero is the theta
    mean; positive modes use peak-amplitude normalization. The Nyquist mode on
    an even grid is not doubled.
    """

    radius = jnp.asarray(boundary.radius_scale)
    if radius.ndim != 2:
        raise ValueError("boundary radius must have shape (ntheta, nxi)")
    ntheta = radius.shape[0]
    coefficients = jnp.fft.rfft(radius, axis=0) / float(ntheta)
    scale = jnp.full(coefficients.shape[0], 2.0, dtype=radius.dtype).at[0].set(1.0)
    if ntheta % 2 == 0 and ntheta > 1:
        scale = scale.at[-1].set(1.0)
    return jnp.abs(coefficients) * scale[:, None]


@dataclass(frozen=True)
class AxisymmetricBetaDiagnostics:
    """Scalar checks for one axisymmetric free-boundary beta point."""

    requested_beta: Array
    achieved_reference_beta: Array
    volume_averaged_beta: Array
    local_axis_beta: Array
    center_radius: Array
    center_axis_field: Array
    center_vacuum_side_field: Array
    diamagnetic_field_ratio: Array
    paraxial_field_ratio: Array
    paraxial_relative_error: Array


@dataclass(frozen=True)
class NonaxisymmetricBetaDiagnostics:
    """Global and modal checks for one theta-dependent beta point."""

    requested_beta: Array
    achieved_reference_beta: Array
    volume_averaged_beta: Array
    center_mean_radius: Array
    center_mean_field: Array
    center_boundary_modes: Array
    plasma_volume: Array
    plasma_energy: Array


def _volume_average(values: Array, result: "FreeBoundaryMirrorResult", grid: "MirrorGrid") -> Array:
    geometry = result.plasma_energy.geometry
    weights = (
        jnp.asarray(grid.radial_weights)[:, None, None]
        * jnp.asarray(grid.theta_basis.weights)[None, :, None]
        * jnp.asarray(grid.axial_basis.weights)[None, None, :]
    )
    measure = weights * geometry.sqrt_g
    return jnp.sum(jnp.asarray(values) * measure) / jnp.sum(measure)


def _plasma_volume(result: "FreeBoundaryMirrorResult", grid: "MirrorGrid") -> Array:
    geometry = result.plasma_energy.geometry
    weights = (
        jnp.asarray(grid.radial_weights)[:, None, None]
        * jnp.asarray(grid.theta_basis.weights)[None, :, None]
        * jnp.asarray(grid.axial_basis.weights)[None, None, :]
    )
    return jnp.sum(weights * geometry.sqrt_g)


def summarize_axisymmetric_beta_scan(
    results: tuple["FreeBoundaryMirrorResult", ...],
    requested_betas: Array,
    grid: "MirrorGrid",
    *,
    reference_field: float,
) -> tuple[AxisymmetricBetaDiagnostics, ...]:
    """Summarize solved beta points against the beta-zero equilibrium.

    ``achieved_reference_beta`` uses the supplied vacuum reference field,
    while ``local_axis_beta`` uses the finite-beta plasma field.  The
    paraxial comparison is ``B/B_vac = sqrt(1-beta)`` and is meaningful for
    a long, approximately cylindrical mirror away from beta one.
    """

    betas = jnp.asarray(requested_betas)
    if betas.ndim != 1 or betas.size != len(results):
        raise ValueError("requested_betas must have one value per result")
    if not results:
        raise ValueError("beta diagnostics require at least one result")
    if grid.ntheta != 1:
        raise ValueError("axisymmetric beta diagnostics require ntheta=1")
    center = int(np.argmin(np.abs(np.asarray(grid.z))))
    baseline_field = jnp.sqrt(results[0].plasma_b_squared[0, 0, center])
    reference_field_squared = float(reference_field) ** 2
    summaries = []
    for requested_beta, result in zip(betas, results, strict=True):
        pressure = result.perpendicular_pressure
        axis_field = jnp.sqrt(result.plasma_b_squared[0, 0, center])
        if hasattr(result.vacuum_field, "lateral_field_xyz"):
            vacuum_xyz = result.vacuum_field.lateral_field_xyz[center]
        else:
            vacuum_xyz = result.vacuum_field.total_xyz[0, 0, center]
        vacuum_side_field = jnp.linalg.norm(vacuum_xyz)
        achieved_beta = 2.0 * MU0 * pressure[0, 0, center] / reference_field_squared
        local_beta = 2.0 * MU0 * pressure[0, 0, center] / axis_field**2
        average_pressure = _volume_average(pressure, result, grid)
        average_b_squared = _volume_average(result.plasma_b_squared, result, grid)
        average_beta = 2.0 * MU0 * average_pressure / average_b_squared
        diamagnetic_ratio = axis_field / baseline_field
        paraxial_ratio = jnp.sqrt(jnp.maximum(1.0 - achieved_beta, 0.0))
        summaries.append(
            AxisymmetricBetaDiagnostics(
                requested_beta=requested_beta,
                achieved_reference_beta=achieved_beta,
                volume_averaged_beta=average_beta,
                local_axis_beta=local_beta,
                center_radius=result.boundary.radius_scale[0, center],
                center_axis_field=axis_field,
                center_vacuum_side_field=vacuum_side_field,
                diamagnetic_field_ratio=diamagnetic_ratio,
                paraxial_field_ratio=paraxial_ratio,
                paraxial_relative_error=(diamagnetic_ratio - paraxial_ratio)
                / jnp.maximum(paraxial_ratio, jnp.finfo(axis_field.dtype).tiny),
            )
        )
    return tuple(summaries)


def summarize_nonaxisymmetric_beta_scan(
    results: tuple["FreeBoundaryMirrorResult", ...],
    requested_betas: Array,
    grid: "MirrorGrid",
    *,
    reference_field: float,
) -> tuple[NonaxisymmetricBetaDiagnostics, ...]:
    """Summarize solved 3D beta points with global and Fourier observables."""

    betas = jnp.asarray(requested_betas)
    if betas.ndim != 1 or betas.size != len(results):
        raise ValueError("requested_betas must have one value per result")
    if not results:
        raise ValueError("beta diagnostics require at least one result")
    if grid.ntheta <= 1:
        raise ValueError("nonaxisymmetric beta diagnostics require ntheta > 1")
    center = int(np.argmin(np.abs(np.asarray(grid.z))))
    reference_field_squared = float(reference_field) ** 2
    summaries = []
    for requested_beta, result in zip(betas, results, strict=True):
        pressure = result.perpendicular_pressure
        center_field = jnp.sqrt(result.plasma_b_squared[0, :, center])
        summaries.append(
            NonaxisymmetricBetaDiagnostics(
                requested_beta=requested_beta,
                achieved_reference_beta=(
                    2.0 * MU0 * jnp.mean(pressure[0, :, center]) / reference_field_squared
                ),
                volume_averaged_beta=(
                    2.0
                    * MU0
                    * _volume_average(pressure, result, grid)
                    / _volume_average(result.plasma_b_squared, result, grid)
                ),
                center_mean_radius=jnp.mean(result.boundary.radius_scale[:, center]),
                center_mean_field=jnp.mean(center_field),
                center_boundary_modes=boundary_fourier_amplitudes(result.boundary)[:, center],
                plasma_volume=_plasma_volume(result, grid),
                plasma_energy=result.plasma_energy.total,
            )
        )
    return tuple(summaries)


if TYPE_CHECKING:
    from .basis import MirrorGrid
    from .free_boundary import FreeBoundaryMirrorResult
    from .model import MirrorBoundary
