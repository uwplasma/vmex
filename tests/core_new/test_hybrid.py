"""M8 toroidal stellarator-mirror hybrid geometry and VMEC projection."""

from __future__ import annotations

import numpy as np

from vmec_jax.core.hybrid import (
    coil_informed_toroidal_flux,
    hybrid_projection_error,
    sample_stellarator_mirror_hybrid,
    stellarator_mirror_hybrid_input,
    trace_square_coil_vacuum_axis,
)
from vmec_jax.core.coils import square_mirror_coils
from vmec_jax.core.hybrid_free_boundary import _corrector_config


def test_square_axis_has_straight_sides_and_four_localized_corners() -> None:
    samples = sample_stellarator_mirror_hybrid(ntheta=32, nzeta=512)
    x = samples.axis_radius * np.cos(samples.zeta)
    y = samples.axis_radius * np.sin(samples.zeta)
    side = samples.side_weight[0] > 0.98
    corner = samples.corner_weight[0] > 0.98

    side_distance = np.minimum(np.abs(np.abs(x) - 1.5), np.abs(np.abs(y) - 1.5))
    assert np.max(side_distance[side]) < 2.0e-3
    assert np.count_nonzero(np.diff(np.r_[corner, corner[0]].astype(int)) == 1) == 4
    assert np.ptp(samples.axis_radius[corner]) < 1.0e-2
    assert np.mean(samples.axis_radius[corner]) > np.mean(samples.axis_radius[side])

    circular = sample_stellarator_mirror_hybrid(ntheta=16, nzeta=64, axis_square_fraction=0.0)
    np.testing.assert_allclose(circular.axis_radius, 1.5, atol=2.0e-15)
    circular_power = sample_stellarator_mirror_hybrid(ntheta=16, nzeta=64, axis_square_power=2.0)
    np.testing.assert_allclose(circular_power.axis_radius, 1.5, atol=2.0e-15)
    intermediate = sample_stellarator_mirror_hybrid(ntheta=16, nzeta=64, axis_square_power=2.5)
    assert np.ptp(intermediate.axis_radius) > 0.0
    assert np.ptp(intermediate.axis_radius) < np.ptp(samples.axis_radius)


def test_corner_ellipse_rotates_while_side_sections_remain_aligned() -> None:
    samples = sample_stellarator_mirror_hybrid(ntheta=128, nzeta=256)

    def orientation(index: int) -> float:
        radial = samples.radius[:, index] - samples.axis_radius[index]
        height = samples.height[:, index]
        covariance = np.cov(np.stack([radial, height]), bias=True)
        return 0.5 * np.arctan2(2.0 * covariance[0, 1], covariance[0, 0] - covariance[1, 1])

    side_indices = np.flatnonzero(samples.side_weight[0] > 0.999)
    corner_indices = np.flatnonzero(samples.corner_weight[0] > 0.999)
    side_angles = np.asarray([orientation(i) for i in side_indices])
    corner_angles = np.asarray([orientation(i) for i in corner_indices])
    assert np.max(np.abs(np.abs(side_angles) - 0.5 * np.pi)) < 2.0e-10
    assert np.ptp(corner_angles) > 0.2


def test_fourier_projection_converges_and_builds_standard_vmec_input() -> None:
    low = hybrid_projection_error(mpol=4, ntor=8, ntheta=48, nzeta=256)
    high = hybrid_projection_error(mpol=6, ntor=20, ntheta=48, nzeta=256)
    assert high["maximum"] < 0.35 * low["maximum"]
    assert high["maximum"] < 2.0e-3

    inp = stellarator_mirror_hybrid_input(mpol=6, ntor=20, ntheta=48, nzeta=256, curtor=3.0e3)
    assert inp.nfp == 1 and not inp.lfreeb and not inp.lasym
    assert inp.ncurr == 1 and inp.curtor == 3.0e3
    np.testing.assert_allclose(inp.ac[0], 1.0)
    np.testing.assert_allclose(inp.ac[1:], 0.0)
    assert inp.rbc.shape == (41, 6)
    assert inp.zbs.shape == inp.rbc.shape
    assert np.count_nonzero(inp.rbc) > 10
    assert np.count_nonzero(inp.zbs) > 10


def test_square_coil_vacuum_axis_is_closed_planar_and_fourier_resolved() -> None:
    coils = square_mirror_coils(n_segments=24, regularization_epsilon=5.0e-7)
    axis = trace_square_coil_vacuum_axis(coils, n_steps=512, nzeta=128)
    assert axis.closure_error < 2.0e-5
    assert axis.planarity_error < 1.0e-12
    assert 1.49 < np.min(axis.radius) < 1.51
    assert 1.85 < np.max(axis.radius) < 1.87
    assert np.max(axis.field_strength) > 1.5 * np.min(axis.field_strength)
    assert np.all(axis.toroidal_field < 0.0)
    np.testing.assert_allclose(
        axis.field_strength * axis.flux_tube_scale**2,
        np.exp(np.mean(np.log(axis.field_strength))),
        rtol=2.0e-14,
    )
    np.testing.assert_allclose(
        np.abs(axis.toroidal_field) * axis.toroidal_flux_scale**2,
        np.exp(np.mean(np.log(np.abs(axis.toroidal_field)))),
        rtol=2.0e-14,
    )
    flux = coil_informed_toroidal_flux(axis, 0.1)
    assert -0.05 < flux < -0.04
    np.testing.assert_allclose(
        np.pi * (0.1 * axis.toroidal_flux_scale) ** 2 * axis.toroidal_field,
        flux,
        rtol=2.0e-14,
    )
    coefficients = np.fft.rfft(axis.radius)
    truncated = np.zeros_like(coefficients)
    truncated[:17] = coefficients[:17]
    reconstructed = np.fft.irfft(truncated, n=axis.radius.size)
    assert np.max(np.abs(reconstructed - axis.radius)) < 5.0e-4

    samples = sample_stellarator_mirror_hybrid(
        ntheta=16,
        nzeta=axis.zeta.size,
        axis_radius_samples=axis.radius,
        minor_radius_samples=axis.flux_tube_scale,
    )
    np.testing.assert_allclose(samples.axis_radius, axis.radius)


def test_hybrid_corrector_expands_krylov_space_only_above_validated_endpoint() -> None:
    assert _corrector_config(0.006978125).gmres_restart == 80
    assert _corrector_config(0.007040625).gmres_restart == 120
    assert _corrector_config(0.0072).gmres_restart == 160
