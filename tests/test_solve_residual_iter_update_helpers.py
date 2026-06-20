from __future__ import annotations

import numpy as np
import pytest

from vmec_jax.solvers.fixed_boundary.residual.update import (
    ResidualControllerState,
    ResidualVelocityBlocks,
    backtracking_momentum_search,
    controller_state_after_catastrophic_restart_update,
    controller_state_after_pre_restart_update,
    controller_state_after_vmec2000_time_control_restart_update,
    controller_state_from_namespace,
    controller_state_from_resume_state,
    controller_state_legacy_payload,
    controller_state_values,
    direct_force_fallback_trial,
    force_update_rms,
    host_catastrophic_restart_update,
    host_force_update_rms,
    host_momentum_update_np,
    host_pre_restart_trigger_update,
    host_vmec2000_time_control_restart_update,
    initial_residual_controller_state,
    initial_residual_velocity_state,
    momentum_update_jax,
    residual_evolve_coefficients,
    scale_velocity_blocks,
    strict_momentum_update_proposal,
    strict_trial_evaluation,
    velocity_blocks_from_resume_state,
    velocity_blocks_legacy_payload,
    zero_velocity_blocks_like,
)


def _blocks(*, offset: float, scale: float = 1.0) -> ResidualVelocityBlocks:
    base = np.arange(6.0, dtype=float).reshape(2, 3)
    return ResidualVelocityBlocks(*(scale * (base + offset + float(idx)) for idx in range(12)))


def test_initial_residual_velocity_state_sets_caps_and_block_shapes() -> None:
    class State:
        Rcos = np.zeros((4, 2, 3), dtype=np.float64)

    init = initial_residual_velocity_state(
        state=State(),
        mpol=5,
        nrange=7,
        host_update_assembly=True,
        reference_mode=True,
    )

    assert init.max_coeff_delta_rms == pytest.approx(5.0e-6)
    assert init.max_update_rms == pytest.approx(1.0e-3)
    assert len(init.velocities) == 12
    for block in init.velocities:
        assert isinstance(block, np.ndarray)
        assert block.shape == (4, 5, 7)
        assert block.dtype == np.float64
        np.testing.assert_allclose(block, 0.0)


def test_velocity_blocks_resume_round_trip_preserves_named_channels() -> None:
    defaults = ResidualVelocityBlocks(*(f"default-{idx}" for idx in range(12)))
    resume_state = {
        "vRcc": "resume-rcc",
        "vRss": "resume-rss",
        "vZsc": "resume-zsc",
        "vZcs": "resume-zcs",
        "vLsc": "resume-lsc",
        "vLcs": "resume-lcs",
        "vRsc": "resume-rsc",
        "vRcs": "resume-rcs",
        "vZcc": "resume-zcc",
        "vZss": "resume-zss",
        "vLcc": "resume-lcc",
        "vLss": "resume-lss",
    }

    blocks = velocity_blocks_from_resume_state(
        resume_state,
        defaults,
        as_velocity=lambda value: value,
    )

    assert blocks == ResidualVelocityBlocks(
        rcc="resume-rcc",
        rss="resume-rss",
        rsc="resume-rsc",
        rcs="resume-rcs",
        zsc="resume-zsc",
        zcs="resume-zcs",
        zcc="resume-zcc",
        zss="resume-zss",
        lsc="resume-lsc",
        lcs="resume-lcs",
        lcc="resume-lcc",
        lss="resume-lss",
    )
    assert velocity_blocks_legacy_payload(blocks) == resume_state


def test_controller_state_resume_round_trip_preserves_legacy_scalars() -> None:
    checkpoint = object()
    defaults = ResidualControllerState(
        time_step=0.1,
        inv_tau=[1.0, 2.0],
        fsq_prev=3.0,
        fsq0_prev=4.0,
        flip_sign=1.0,
        iter1=2,
        ijacob=0,
        bad_resets=0,
        res0=-1.0,
        res1=-1.0,
        prev_rz_fsq=2.0,
        bad_growth_streak=0,
        huge_force_restart_count=0,
        state_checkpoint="default-checkpoint",
    )
    resume_state = {
        "time_step": "0.25",
        "inv_tau": (0.3, 0.4),
        "fsq_prev": "5.0",
        "fsq0_prev": "6.0",
        "flip_sign": "-1.0",
        "iter1": "7",
        "ijacob": "8",
        "bad_resets": "9",
        "res0": "0.5",
        "res1": "0.25",
        "prev_rz_fsq": "0.125",
        "bad_growth_streak": "3",
        "huge_force_restart_count": "4",
        "state_checkpoint": checkpoint,
    }

    state = controller_state_from_resume_state(resume_state, defaults)

    assert state.time_step == pytest.approx(0.25)
    assert state.inv_tau == [0.3, 0.4]
    assert state.fsq_prev == pytest.approx(5.0)
    assert state.fsq0_prev == pytest.approx(6.0)
    assert state.flip_sign == pytest.approx(-1.0)
    assert state.iter1 == 7
    assert state.ijacob == 8
    assert state.bad_resets == 9
    assert state.res0 == pytest.approx(0.5)
    assert state.res1 == pytest.approx(0.25)
    assert state.prev_rz_fsq == pytest.approx(0.125)
    assert state.bad_growth_streak == 3
    assert state.huge_force_restart_count == 4
    assert state.state_checkpoint is checkpoint
    assert controller_state_legacy_payload(state)["state_checkpoint"] is checkpoint


def test_initial_residual_controller_state_matches_vmec_defaults() -> None:
    checkpoint = object()

    state = initial_residual_controller_state(
        step_size=0.25,
        k_ndamp=4,
        initial_flip_sign=-1.0,
        state_checkpoint=checkpoint,
    )

    assert state.time_step == pytest.approx(0.25)
    assert state.inv_tau == [0.6, 0.6, 0.6, 0.6]
    assert state.fsq_prev == pytest.approx(1.0)
    assert state.fsq0_prev == pytest.approx(1.0)
    assert state.flip_sign == pytest.approx(-1.0)
    assert state.iter1 == 1
    assert state.ijacob == 0
    assert state.bad_resets == 0
    assert state.res0 == pytest.approx(-1.0)
    assert state.res1 == pytest.approx(-1.0)
    assert state.prev_rz_fsq == pytest.approx(2.0)
    assert state.bad_growth_streak == 0
    assert state.huge_force_restart_count == 0
    assert state.state_checkpoint is checkpoint


def test_controller_state_from_namespace_matches_legacy_local_order() -> None:
    namespace = {
        "time_step": 0.25,
        "inv_tau": (0.6, 0.7),
        "fsq_prev": 1.0,
        "fsq0_prev": 2.0,
        "flip_sign": -1.0,
        "iter1": 3,
        "ijacob": 4,
        "bad_resets": 5,
        "res0": 0.1,
        "res1": 0.2,
        "prev_rz_fsq": 0.3,
        "bad_growth_streak": 6,
        "huge_force_restart_count": 7,
        "state_checkpoint": "checkpoint",
    }

    state = controller_state_from_namespace(namespace)

    assert state.inv_tau == [0.6, 0.7]
    assert controller_state_values(state) == tuple(state)
    assert controller_state_legacy_payload(state)["iter1"] == 3


def test_residual_evolve_coefficients_match_vmec_damping_recurrence() -> None:
    first = residual_evolve_coefficients(
        iter2=4,
        iter1=4,
        inv_tau=[1.0, 2.0, 3.0],
        time_step=0.5,
        fsq1=2.0,
        fsq_prev=4.0,
        fsq0_curr=6.0,
        k_ndamp=3,
    )

    assert first.inv_tau == [0.3, 0.3, 0.3]
    assert first.fsq_prev == pytest.approx(2.0)
    assert first.fsq0_prev == pytest.approx(6.0)
    assert first.dtau == pytest.approx(0.075)
    assert first.b1 == pytest.approx(0.925)
    assert first.fac == pytest.approx(1.0 / 1.075)

    later = residual_evolve_coefficients(
        iter2=5,
        iter1=4,
        inv_tau=[0.1, 0.2, 0.3],
        time_step=0.5,
        fsq1=1.0,
        fsq_prev=2.0,
        fsq0_curr=3.0,
        k_ndamp=3,
    )

    assert later.inv_tau == [0.2, 0.3, 0.3]
    assert later.dtau == pytest.approx(0.5 * (0.2 + 0.3 + 0.3) / 3.0 / 2.0)


def test_controller_state_applies_restart_update_payloads() -> None:
    state = ResidualControllerState(
        time_step=0.5,
        inv_tau=[1.0, 1.0],
        fsq_prev=9.0,
        fsq0_prev=8.0,
        flip_sign=-1.0,
        iter1=2,
        ijacob=3,
        bad_resets=4,
        res0=0.3,
        res1=0.2,
        prev_rz_fsq=0.1,
        bad_growth_streak=7,
        huge_force_restart_count=8,
        state_checkpoint="checkpoint",
    )

    pre_restart = host_pre_restart_trigger_update(
        pre_restart_reason="bad_jacobian",
        huge_initial_forces=True,
        huge_force_restart_count=8,
        time_step=state.time_step,
        restart_badjac_factor=0.9,
        restart_badprog_factor=1.03,
        stage_transition_scale=0.5,
        step_size=0.1,
        ijacob=state.ijacob,
        bad_resets=state.bad_resets,
        iter2=11,
        fsq_prev_before=6.0,
        fsq0_prev_before=5.0,
        k_ndamp=2,
    )
    after_pre = controller_state_after_pre_restart_update(state, pre_restart)

    assert after_pre.time_step == pytest.approx(pre_restart.time_step)
    assert after_pre.inv_tau == pre_restart.inv_tau
    assert after_pre.fsq_prev == pytest.approx(6.0)
    assert after_pre.fsq0_prev == pytest.approx(5.0)
    assert after_pre.iter1 == 11
    assert after_pre.ijacob == 4
    assert after_pre.bad_resets == 5
    assert after_pre.bad_growth_streak == 0
    assert after_pre.huge_force_restart_count == 9
    assert after_pre.flip_sign == state.flip_sign
    assert after_pre.state_checkpoint == "checkpoint"

    catastrophic = host_catastrophic_restart_update(
        probe_bad_jacobian=False,
        w_try=4.0,
        time_step=state.time_step,
        restart_badjac_factor=0.9,
        restart_badprog_factor=1.03,
        step_size=0.1,
        ijacob=state.ijacob,
        bad_resets=state.bad_resets,
        iter2=12,
        fsq_prev_before=4.0,
        fsq0_prev_before=3.0,
        k_ndamp=2,
        max_coeff_delta_rms=1.0,
        max_update_rms=1.0,
    )
    after_catastrophic = controller_state_after_catastrophic_restart_update(state, catastrophic)

    assert after_catastrophic.time_step == pytest.approx(catastrophic.time_step)
    assert after_catastrophic.inv_tau == catastrophic.inv_tau
    assert after_catastrophic.fsq_prev == pytest.approx(4.0)
    assert after_catastrophic.fsq0_prev == pytest.approx(3.0)
    assert after_catastrophic.iter1 == 12
    assert after_catastrophic.ijacob == state.ijacob
    assert after_catastrophic.bad_resets == 5
    assert after_catastrophic.bad_growth_streak == state.bad_growth_streak
    assert after_catastrophic.huge_force_restart_count == state.huge_force_restart_count


def test_vmec2000_time_control_restart_update_preserves_restart_iter_semantics() -> None:
    bad_jac = host_vmec2000_time_control_restart_update(
        irst=2,
        time_step=0.2,
        restart_badjac_factor=0.9,
        restart_badprog_factor=1.03,
        ijacob=4,
        bad_resets=5,
        iter2=9,
        fsq_prev_before=1.25,
        fsq0_prev_before=2.5,
        k_ndamp=3,
    )

    assert bad_jac.time_step == pytest.approx(0.18)
    assert bad_jac.ijacob == 5
    assert bad_jac.step_status == "restart_bad_jacobian"
    assert bad_jac.restart_reason == "bad_jacobian"
    assert bad_jac.restart_path == "vmec2000_bad_jacobian"
    assert bad_jac.bad_resets == 6
    assert bad_jac.iter1 == 9
    assert bad_jac.fsq_prev == pytest.approx(1.25)
    assert bad_jac.fsq0_prev == pytest.approx(2.5)
    np.testing.assert_allclose(bad_jac.inv_tau, [0.15 / 0.18] * 3)

    bad_progress = host_vmec2000_time_control_restart_update(
        irst=3,
        time_step=0.206,
        restart_badjac_factor=0.9,
        restart_badprog_factor=1.03,
        ijacob=4,
        bad_resets=5,
        iter2=9,
        fsq_prev_before=1.25,
        fsq0_prev_before=2.5,
        k_ndamp=2,
    )

    assert bad_progress.time_step == pytest.approx(0.2)
    assert bad_progress.ijacob == 4
    assert bad_progress.step_status == "restart_time_control"
    assert bad_progress.restart_reason == "time_control"
    assert bad_progress.restart_path == "vmec2000_time_control"
    np.testing.assert_allclose(bad_progress.inv_tau, [0.15 / 0.2] * 2)


def test_controller_state_applies_vmec2000_time_control_restart_update() -> None:
    state = initial_residual_controller_state(
        step_size=0.2,
        k_ndamp=2,
        initial_flip_sign=-1.0,
        state_checkpoint="checkpoint",
    )._replace(res0=0.1, res1=0.2, prev_rz_fsq=0.3, bad_growth_streak=4)
    update = host_vmec2000_time_control_restart_update(
        irst=2,
        time_step=0.2,
        restart_badjac_factor=0.9,
        restart_badprog_factor=1.03,
        ijacob=3,
        bad_resets=4,
        iter2=5,
        fsq_prev_before=6.0,
        fsq0_prev_before=7.0,
        k_ndamp=2,
    )

    got = controller_state_after_vmec2000_time_control_restart_update(state, update)

    assert got.time_step == pytest.approx(0.18)
    assert got.inv_tau == update.inv_tau
    assert got.fsq_prev == pytest.approx(6.0)
    assert got.fsq0_prev == pytest.approx(7.0)
    assert got.iter1 == 5
    assert got.ijacob == 4
    assert got.bad_resets == 5
    assert got.bad_growth_streak == 0
    assert got.flip_sign == pytest.approx(-1.0)
    assert got.res0 == pytest.approx(0.1)
    assert got.res1 == pytest.approx(0.2)
    assert got.prev_rz_fsq == pytest.approx(0.3)


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


def test_strict_momentum_update_proposal_builds_candidate_and_reports_rms() -> None:
    velocities = ResidualVelocityBlocks(*(np.zeros((2, 3)) for _ in range(12)))
    forces = ResidualVelocityBlocks(*(np.ones((2, 3)) for _ in range(12)))

    def delta_tuple_from_blocks(dt, transforms, *blocks, **_kwargs):
        return tuple(float(dt) * np.asarray(block) for block in blocks)

    def candidate_state_from_delta_tuple(deltas, **_kwargs):
        return float(np.mean(deltas[0]))

    result = strict_momentum_update_proposal(
        velocities=velocities,
        forces=forces,
        host_update_assembly=True,
        need_update_rms=True,
        materialize_update_rms=True,
        limit_update_rms=False,
        max_update_rms=1.0,
        b1=0.0,
        fac=1.0,
        force_scale=0.2,
        flip_sign=1.0,
        dt_eff=0.1,
        delta_transforms=(),
        delta_tuple_from_blocks=delta_tuple_from_blocks,
        candidate_state_from_delta_tuple=candidate_state_from_delta_tuple,
    )

    assert result.scale == pytest.approx(1.0)
    assert result.state == pytest.approx(0.02)
    for block in result.velocities:
        np.testing.assert_allclose(block, 0.2)
    assert result.update_rms == pytest.approx(0.02)
    assert float(np.asarray(result.update_rms_j)) == pytest.approx(result.update_rms)


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


def test_backtracking_momentum_search_accepts_first_good_trial() -> None:
    velocities = _blocks(offset=0.0, scale=0.0)
    forces = ResidualVelocityBlocks(*(np.ones((2, 3)) for _ in range(12)))

    def delta_tuple_from_blocks(dt, transforms, *blocks):
        return tuple(float(dt) * np.asarray(block) for block in blocks)

    def candidate_state_from_delta_tuple(deltas, **_kwargs):
        return float(np.mean(deltas[0]))

    result = backtracking_momentum_search(
        state=0.0,
        velocities=velocities,
        forces=forces,
        time_step=0.2,
        step_size=0.2,
        b1=0.0,
        fac=1.0,
        flip_sign=1.0,
        w_curr=1.0,
        delta_transforms=(),
        delta_tuple_from_blocks=delta_tuple_from_blocks,
        candidate_state_from_delta_tuple=candidate_state_from_delta_tuple,
        freeb_bsqvac_half_for_trial_state=lambda state: None,
        trial_residual_total=lambda _state, _bsqvac: 1.0,
    )

    assert result.accepted
    assert result.step_status == "momentum"
    assert result.dt_eff == pytest.approx(0.2)
    assert result.state == pytest.approx(0.04)
    for block in result.velocities:
        np.testing.assert_allclose(block, 0.2)
    assert result.update_rms == pytest.approx(host_force_update_rms(0.2, *result.velocities))


def test_backtracking_momentum_search_rejects_and_damps_velocity() -> None:
    velocities = ResidualVelocityBlocks(*(2.0 * np.ones((2, 3)) for _ in range(12)))
    forces = ResidualVelocityBlocks(*(np.ones((2, 3)) for _ in range(12)))

    result = backtracking_momentum_search(
        state="old-state",
        velocities=velocities,
        forces=forces,
        time_step=0.4,
        step_size=0.4,
        b1=1.0,
        fac=1.0,
        flip_sign=1.0,
        w_curr=1.0,
        delta_transforms=(),
        delta_tuple_from_blocks=lambda dt, transforms, *blocks: blocks,
        candidate_state_from_delta_tuple=lambda deltas, **_kwargs: "trial-state",
        freeb_bsqvac_half_for_trial_state=lambda state: None,
        trial_residual_total=lambda _state, _bsqvac: float("inf"),
        max_backtracks=2,
    )

    assert not result.accepted
    assert result.step_status == "rejected"
    assert result.state == "old-state"
    assert result.dt_eff == pytest.approx(0.1)
    assert result.update_rms == pytest.approx(0.0)
    for block in result.velocities:
        np.testing.assert_allclose(block, 1.0)


def test_direct_force_fallback_trial_caps_step_and_reports_residual() -> None:
    forces = ResidualVelocityBlocks(*(np.full((2, 3), idx + 1.0) for idx in range(12)))
    force_rms = host_force_update_rms(1.0, *forces)
    expected_dt = max(min(0.1, 0.05 / force_rms), 1.0e-12)

    def delta_tuple_from_blocks(dt, transforms, *blocks):
        return tuple(float(dt) * np.asarray(block) for block in blocks)

    def candidate_state_from_delta_tuple(deltas, **_kwargs):
        return float(np.mean(deltas[0]))

    result = direct_force_fallback_trial(
        forces=forces,
        dt_eff=1.0,
        max_update_rms=0.05,
        flip_sign=-1.0,
        delta_transforms=(),
        delta_tuple_from_blocks=delta_tuple_from_blocks,
        candidate_state_from_delta_tuple=candidate_state_from_delta_tuple,
        freeb_bsqvac_half_for_trial_state=lambda state: ("bsq", state),
        trial_residual_total=lambda state, bsq: state + bsq[1] + 2.0,
    )

    assert result.dt_eff == pytest.approx(expected_dt)
    assert result.state == pytest.approx(-expected_dt)
    assert result.residual == pytest.approx(2.0 - 2.0 * expected_dt)
    assert result.update_rms == pytest.approx(host_force_update_rms(expected_dt, *forces))


def test_strict_trial_evaluation_backtracks_and_scales_primary_velocities() -> None:
    velocities = ResidualVelocityBlocks(*(np.ones((2, 3)) for _ in range(12)))

    def candidate_state_from_delta_tuple(_deltas, *, scale, **_kwargs):
        return float(scale)

    def trial_residual_total(state, _bsqvac, **_kwargs):
        return 2.0 if float(state) > 0.75 else 0.9

    result = strict_trial_evaluation(
        state_try=1.0,
        velocities=velocities,
        update_deltas=tuple(np.ones((2, 3)) for _ in range(6)),
        update_rms=0.4,
        dt_eff=0.2,
        w_curr=1.0,
        backtracking=True,
        reference_mode=False,
        host_update_assembly=False,
        zero_m1_value=1.0,
        zero_m1_host=1.0,
        zero_m1_probe_value=0.0,
        candidate_state_from_delta_tuple=candidate_state_from_delta_tuple,
        freeb_bsqvac_half_for_trial_state=lambda state: None,
        trial_residual_total=trial_residual_total,
    )

    assert result.alpha == pytest.approx(0.5)
    assert result.state == pytest.approx(0.5)
    assert result.dt_eff == pytest.approx(0.1)
    assert result.update_rms == pytest.approx(0.2)
    assert result.w_try == pytest.approx(0.9)
    assert not result.probe_bad_jacobian
    for block_name in ("rcc", "rss", "zsc", "zcs", "lsc", "lcs"):
        np.testing.assert_allclose(getattr(result.velocities, block_name), 0.5)
    for block_name in ("rsc", "rcs", "zcc", "zss", "lcc", "lss"):
        np.testing.assert_allclose(getattr(result.velocities, block_name), 1.0)


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


def test_host_pre_restart_trigger_update_handles_bad_jacobian_and_huge_force_streak() -> None:
    update = host_pre_restart_trigger_update(
        pre_restart_reason="bad_jacobian",
        huge_initial_forces=True,
        huge_force_restart_count=2,
        time_step=0.2,
        restart_badjac_factor=0.9,
        restart_badprog_factor=1.03,
        stage_transition_scale=0.5,
        step_size=0.1,
        ijacob=3,
        bad_resets=4,
        iter2=12,
        fsq_prev_before=1.25,
        fsq0_prev_before=2.5,
        k_ndamp=3,
    )

    assert update.step_status == "restart_bad_jacobian"
    assert update.ijacob == 4
    assert update.bad_resets == 5
    assert update.iter1 == 12
    assert update.huge_force_restart_count == 3
    np.testing.assert_allclose(update.time_step, 0.18)
    np.testing.assert_allclose(update.time_step_iter, 0.18)
    np.testing.assert_allclose(update.fsq_prev, 1.25)
    np.testing.assert_allclose(update.fsq0_prev, 2.5)
    np.testing.assert_allclose(update.inv_tau, [0.15 / 0.18] * 3)


def test_host_pre_restart_trigger_update_handles_stage_transition_and_bad_progress() -> None:
    stage = host_pre_restart_trigger_update(
        pre_restart_reason="stage_transition",
        huge_initial_forces=True,
        huge_force_restart_count=9,
        time_step=0.2,
        restart_badjac_factor=0.9,
        restart_badprog_factor=1.03,
        stage_transition_scale=0.5,
        step_size=0.1,
        ijacob=3,
        bad_resets=4,
        iter2=12,
        fsq_prev_before=1.25,
        fsq0_prev_before=2.5,
        k_ndamp=2,
    )
    assert stage.step_status == "restart_stage_transition"
    assert stage.ijacob == 3
    assert stage.huge_force_restart_count == 0
    np.testing.assert_allclose(stage.time_step, 0.1)
    np.testing.assert_allclose(stage.inv_tau, [1.5] * 2)

    bad_progress = host_pre_restart_trigger_update(
        pre_restart_reason="bad_progress",
        huge_initial_forces=False,
        huge_force_restart_count=9,
        time_step=0.103,
        restart_badjac_factor=0.9,
        restart_badprog_factor=1.03,
        stage_transition_scale=0.5,
        step_size=0.1,
        ijacob=3,
        bad_resets=4,
        iter2=12,
        fsq_prev_before=1.25,
        fsq0_prev_before=2.5,
        k_ndamp=2,
    )
    assert bad_progress.step_status == "restart_bad_progress"
    np.testing.assert_allclose(bad_progress.time_step, 0.1)
    np.testing.assert_allclose(bad_progress.inv_tau, [1.5] * 2)


def test_host_pre_restart_trigger_update_applies_vmec_milestone_scale() -> None:
    update = host_pre_restart_trigger_update(
        pre_restart_reason="bad_jacobian",
        huge_initial_forces=False,
        huge_force_restart_count=0,
        time_step=0.2,
        restart_badjac_factor=0.9,
        restart_badprog_factor=1.03,
        stage_transition_scale=0.5,
        step_size=0.05,
        ijacob=24,
        bad_resets=0,
        iter2=25,
        fsq_prev_before=1.0,
        fsq0_prev_before=1.0,
        k_ndamp=1,
    )

    assert update.ijacob == 25
    assert update.step_status == "restart_bad_jacobian"
    np.testing.assert_allclose(update.time_step, 0.98 * 0.05)
    np.testing.assert_allclose(update.inv_tau, [0.15 / (0.98 * 0.05)])
