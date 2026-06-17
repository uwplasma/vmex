from __future__ import annotations

import numpy as np
import pytest

from vmec_jax.solvers.fixed_boundary.residual.update import (
    ResidualVelocityBlocks,
    force_update_rms,
    host_catastrophic_restart_update,
    host_force_update_rms,
    host_momentum_update_np,
    momentum_update_jax,
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


def test_momentum_update_jax_matches_host_momentum_update_np() -> None:
    velocities = _blocks(offset=0.25, scale=0.2)
    forces = _blocks(offset=-3.0, scale=0.7)
    kwargs = dict(
        velocities=velocities,
        forces=forces,
        b1=0.6,
        fac=1.2,
        force_scale=0.09,
        flip_sign=-1.0,
        dt_eff=0.03,
        compute_update_rms=True,
    )

    host_update = host_momentum_update_np(**kwargs)
    jax_update = momentum_update_jax(**kwargs)

    for host_block, jax_block in zip(host_update.velocities, jax_update.velocities):
        np.testing.assert_allclose(np.asarray(jax_block), host_block)
    expected_rms = np.sqrt(np.mean(sum((kwargs["dt_eff"] * block) ** 2 for block in host_update.velocities)))
    assert float(np.asarray(jax_update.update_rms)) == pytest.approx(expected_rms)


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


def test_host_force_update_rms_matches_inline_force_formula() -> None:
    blocks = tuple(np.full((2, 3), idx + 1.0) for idx in range(12))
    scale = 0.125

    expected = np.sqrt(np.mean(sum((scale * block) ** 2 for block in blocks)))

    assert host_force_update_rms(scale, *blocks) == pytest.approx(expected)
    assert host_force_update_rms(scale) == pytest.approx(0.0)


def test_force_update_rms_is_jax_visible_and_matches_host_wrapper() -> None:
    blocks = tuple(np.arange(6.0, dtype=float).reshape(2, 3) + idx for idx in range(12))
    scale = 0.0375

    got = force_update_rms(scale, *blocks)

    assert np.asarray(got).shape == ()
    assert float(np.asarray(got)) == pytest.approx(host_force_update_rms(scale, *blocks))


def test_free_boundary_control_module_reexports_velocity_helpers() -> None:
    import vmec_jax.solvers.free_boundary.control as freeb_control

    assert freeb_control.zero_velocity_blocks_like is zero_velocity_blocks_like
    assert freeb_control.scale_velocity_blocks is scale_velocity_blocks


def test_host_catastrophic_restart_update_handles_bad_progress_branch() -> None:
    update = host_catastrophic_restart_update(
        probe_bad_jacobian=False,
        w_try=4.0,
        time_step=0.103,
        restart_badjac_factor=0.9,
        restart_badprog_factor=1.03,
        step_size=0.2,
        ijacob=3,
        bad_resets=7,
        iter2=19,
        fsq_prev_before=1.25,
        fsq0_prev_before=2.5,
        k_ndamp=4,
        max_coeff_delta_rms=8.0e-2,
        max_update_rms=2.0e-3,
    )

    assert update.restart_reason == "bad_progress"
    assert update.step_status == "restart_bad_progress"
    assert update.restart_path == "catastrophic_growth"
    assert update.ijacob == 3
    assert update.bad_resets == 8
    assert update.iter1 == 19
    np.testing.assert_allclose(update.time_step, 0.1)
    np.testing.assert_allclose(update.max_coeff_delta_rms, 4.0e-2)
    np.testing.assert_allclose(update.max_update_rms, 1.6e-3)
    np.testing.assert_allclose(update.fsq_prev, 1.25)
    np.testing.assert_allclose(update.fsq0_prev, 2.5)
    np.testing.assert_allclose(update.inv_tau, [1.5] * 4)
    assert update.update_rms == pytest.approx(0.0)


def test_host_catastrophic_restart_update_handles_nonfinite_bad_jacobian_branch() -> None:
    update = host_catastrophic_restart_update(
        probe_bad_jacobian=False,
        w_try=np.inf,
        time_step=0.2,
        restart_badjac_factor=0.9,
        restart_badprog_factor=1.03,
        step_size=0.2,
        ijacob=4,
        bad_resets=0,
        iter2=5,
        fsq_prev_before=3.0,
        fsq0_prev_before=4.0,
        k_ndamp=2,
        max_coeff_delta_rms=1.0e-13,
        max_update_rms=1.0e-8,
    )

    assert update.restart_reason == "bad_jacobian"
    assert update.step_status == "restart_bad_jacobian"
    assert update.restart_path == "catastrophic_nonfinite"
    assert update.ijacob == 5
    assert update.bad_resets == 1
    np.testing.assert_allclose(update.time_step, 0.18)
    np.testing.assert_allclose(update.max_coeff_delta_rms, 1.0e-12)
    np.testing.assert_allclose(update.max_update_rms, 1.0e-6)
    np.testing.assert_allclose(update.inv_tau, [0.15 / 0.18] * 2)


def test_host_catastrophic_restart_update_applies_vmec_reset_milestone_scale() -> None:
    update = host_catastrophic_restart_update(
        probe_bad_jacobian=True,
        w_try=1.0,
        time_step=0.2,
        restart_badjac_factor=0.9,
        restart_badprog_factor=1.03,
        step_size=0.05,
        ijacob=24,
        bad_resets=0,
        iter2=25,
        fsq_prev_before=1.0,
        fsq0_prev_before=1.0,
        k_ndamp=1,
        max_coeff_delta_rms=1.0,
        max_update_rms=1.0,
    )

    assert update.ijacob == 25
    np.testing.assert_allclose(update.time_step, 0.98 * 0.05)
    np.testing.assert_allclose(update.inv_tau, [0.15 / (0.98 * 0.05)])
