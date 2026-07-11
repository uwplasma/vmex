"""M8 toroidal stellarator-mirror hybrid geometry and VMEC projection."""

from __future__ import annotations

import numpy as np

from vmec_jax.core.hybrid import (
    hybrid_projection_error,
    sample_stellarator_mirror_hybrid,
    stellarator_mirror_hybrid_input,
)


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

    inp = stellarator_mirror_hybrid_input(
        mpol=6, ntor=20, ntheta=48, nzeta=256
    )
    assert inp.nfp == 1 and not inp.lfreeb and not inp.lasym
    assert inp.rbc.shape == (41, 6)
    assert inp.zbs.shape == inp.rbc.shape
    assert np.count_nonzero(inp.rbc) > 10
    assert np.count_nonzero(inp.zbs) > 10
