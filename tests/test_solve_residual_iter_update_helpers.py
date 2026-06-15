from __future__ import annotations

import numpy as np
import pytest

from vmec_jax.solvers.fixed_boundary.residual.update import (
    ResidualVelocityBlocks,
    host_momentum_update_np,
    scale_velocity_blocks,
    zero_velocity_blocks_like,
)


def _blocks(*, offset: float, scale: float = 1.0) -> ResidualVelocityBlocks:
    base = np.arange(6.0, dtype=float).reshape(2, 3)
    return ResidualVelocityBlocks(*(scale * (base + offset + float(idx)) for idx in range(12)))


def test_host_momentum_update_np_matches_strict_update_formula_and_rms() -> None:
    velocities = _blocks(offset=1.0, scale=0.5)
    forces = _blocks(offset=20.0, scale=0.125)
    velocity_inputs = tuple(block.copy() for block in velocities)
    force_inputs = tuple(block.copy() for block in forces)

    b1 = 0.75
    fac = 0.8
    force_scale = 0.2
    flip_sign = -1.0
    dt_eff = 0.05

    got = host_momentum_update_np(
        velocities=velocities,
        forces=forces,
        b1=b1,
        fac=fac,
        force_scale=force_scale,
        flip_sign=flip_sign,
        dt_eff=dt_eff,
        compute_update_rms=True,
    )

    expected = fac * (b1 * np.stack(velocity_inputs) + force_scale * flip_sign * np.stack(force_inputs))
    for block, expected_block in zip(got.velocities, expected):
        np.testing.assert_allclose(block, expected_block)

    expected_rms = abs(dt_eff) * np.sqrt(np.dot(expected.ravel(), expected.ravel()) / expected.size)
    assert got.update_rms == pytest.approx(expected_rms)
    for original, current in zip(velocity_inputs, velocities):
        np.testing.assert_allclose(current, original)
    for original, current in zip(force_inputs, forces):
        np.testing.assert_allclose(current, original)


def test_host_momentum_update_np_can_skip_rms_without_changing_blocks() -> None:
    velocities = _blocks(offset=-2.0, scale=0.25)
    forces = _blocks(offset=3.0, scale=2.0)
    kwargs = dict(
        velocities=velocities,
        forces=forces,
        b1=1.1,
        fac=0.25,
        force_scale=0.05,
        flip_sign=1.0,
        dt_eff=0.4,
    )

    with_rms = host_momentum_update_np(**kwargs, compute_update_rms=True)
    without_rms = host_momentum_update_np(**kwargs, compute_update_rms=False)

    for with_block, without_block in zip(with_rms.velocities, without_rms.velocities):
        np.testing.assert_allclose(without_block, with_block)
    assert without_rms.update_rms == pytest.approx(0.0)


def test_velocity_block_helpers_preserve_shape_dtype_and_scale() -> None:
    a = np.arange(6.0, dtype=np.float64).reshape(2, 3)
    b = np.arange(6, dtype=np.int32).reshape(2, 3)

    za, zb = zero_velocity_blocks_like(a, b)
    assert np.asarray(za).shape == a.shape
    assert np.asarray(zb).shape == b.shape
    assert np.asarray(za).dtype == a.dtype
    assert np.asarray(zb).dtype == b.dtype
    np.testing.assert_allclose(np.asarray(za), 0.0)
    np.testing.assert_allclose(np.asarray(zb), 0.0)

    sa, sb = scale_velocity_blocks(0.5, a, b)
    np.testing.assert_allclose(np.asarray(sa), 0.5 * a)
    np.testing.assert_allclose(np.asarray(sb), 0.5 * b)


def test_free_boundary_control_module_reexports_velocity_helpers() -> None:
    import vmec_jax.solve_free_boundary_control_helpers as freeb_control

    assert freeb_control.zero_velocity_blocks_like is zero_velocity_blocks_like
    assert freeb_control.scale_velocity_blocks is scale_velocity_blocks
