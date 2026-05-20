from __future__ import annotations

import numpy as np

from vmec_jax._compat import jnp
from vmec_jax.solve_scan_time_control import (
    RESTART_BADJAC,
    RESTART_BADPROG_VMEC,
    RESTART_NONE,
    RESTART_STAGE,
    RESTART_TIME,
    scan_restart_decision,
    scan_restart_transition,
    scan_stage_spike_post_scalars,
    scan_time_control_scalars,
)


def _scalar(value):
    return np.asarray(value).item()


def test_bad_jacobian_blocks_checkpoint_and_restarts_with_ijacob_increment():
    tc = scan_time_control_scalars(
        skip_timecontrol=jnp.asarray(False),
        init_mask=jnp.asarray(True),
        fsq=jnp.asarray(0.25),
        fsq_res=jnp.asarray(0.25),
        fsq_phys=jnp.asarray(0.5),
        fsq1=jnp.asarray(0.2),
        fsq_prev_before=jnp.asarray(0.3),
        res0_prev=jnp.asarray(-1.0),
        res1_prev=jnp.asarray(-1.0),
        bad_jacobian=jnp.asarray(True),
        vmec2000_control=True,
    )
    assert not _scalar(tc.checkpoint_mask)
    assert _scalar(tc.res0) == 0.25
    assert _scalar(tc.res1) == 0.5

    decision = scan_restart_decision(
        skip_timecontrol=jnp.asarray(False),
        iter2=jnp.asarray(7, dtype=jnp.int32),
        iter1=jnp.asarray(5, dtype=jnp.int32),
        fsq=jnp.asarray(0.25),
        fsq_phys=jnp.asarray(0.5),
        res0=tc.res0,
        res1=tc.res1,
        bad_jacobian=jnp.asarray(True),
        fsqr=jnp.asarray(0.0),
        fsqz=jnp.asarray(0.0),
        vmec2000_fact=1.0,
        use_restart_triggers=True,
        vmecpp_restart=False,
        k_preconditioner_update_interval=10,
        stage_prev_fsq=None,
        stage_transition_factor=50.0,
        vmec2000_control=True,
    )
    assert _scalar(decision.do_restart)
    assert _scalar(decision.restart_reason) == RESTART_BADJAC
    assert _scalar(decision.irst_restart) == 2

    transition = scan_restart_transition(
        time_step=jnp.asarray(0.01),
        iter_offset=jnp.asarray(3, dtype=jnp.int32),
        ijacob=jnp.asarray(2, dtype=jnp.int32),
        bad_resets=jnp.asarray(4, dtype=jnp.int32),
        iter2=jnp.asarray(7, dtype=jnp.int32),
        restart_reason=decision.restart_reason,
        vmec2000_control=True,
        restart_badjac_factor=0.9,
        restart_badprog_factor=1.03,
        stage_transition_scale=0.5,
        step_size=0.01,
    )
    np.testing.assert_allclose(_scalar(transition.time_step), 0.009)
    assert _scalar(transition.damping_time_step) == _scalar(transition.time_step)
    assert _scalar(transition.ijacob) == 3
    assert _scalar(transition.iter1) == 7
    assert _scalar(transition.bad_resets) == 5
    assert _scalar(transition.force_bcovar_update)


def test_stage_spike_reason_and_post_scale_match_scan_double_scale_path():
    decision = scan_restart_decision(
        skip_timecontrol=jnp.asarray(False),
        iter2=jnp.asarray(1, dtype=jnp.int32),
        iter1=jnp.asarray(0, dtype=jnp.int32),
        fsq=jnp.asarray(1.0),
        fsq_phys=jnp.asarray(51.0),
        res0=jnp.asarray(1.0),
        res1=jnp.asarray(1.0),
        bad_jacobian=jnp.asarray(False),
        fsqr=jnp.asarray(0.0),
        fsqz=jnp.asarray(0.0),
        vmec2000_fact=2.0,
        use_restart_triggers=True,
        vmecpp_restart=False,
        k_preconditioner_update_interval=10,
        stage_prev_fsq=jnp.asarray(1.0),
        stage_transition_factor=50.0,
        vmec2000_control=True,
    )
    assert _scalar(decision.stage_spike)
    assert _scalar(decision.do_restart)
    assert _scalar(decision.restart_reason) == RESTART_STAGE

    transition = scan_restart_transition(
        time_step=jnp.asarray(0.02),
        iter_offset=jnp.asarray(0, dtype=jnp.int32),
        ijacob=jnp.asarray(0, dtype=jnp.int32),
        bad_resets=jnp.asarray(0, dtype=jnp.int32),
        iter2=jnp.asarray(1, dtype=jnp.int32),
        restart_reason=decision.restart_reason,
        vmec2000_control=True,
        restart_badjac_factor=0.9,
        restart_badprog_factor=1.03,
        stage_transition_scale=0.5,
        step_size=0.02,
    )
    np.testing.assert_allclose(_scalar(transition.time_step), 0.01)

    post = scan_stage_spike_post_scalars(
        time_step=transition.time_step,
        stage_spike=decision.stage_spike,
        stage_prev_fsq=jnp.asarray(1.0),
        stage_transition_scale=0.5,
    )
    assert _scalar(post.apply_stage_reset)
    np.testing.assert_allclose(_scalar(post.time_step), 0.005)


def test_vmecpp_bad_progress_uses_bad_progress_reason_and_time_scaling():
    decision = scan_restart_decision(
        skip_timecontrol=jnp.asarray(False),
        iter2=jnp.asarray(31, dtype=jnp.int32),
        iter1=jnp.asarray(20, dtype=jnp.int32),
        fsq=jnp.asarray(0.5),
        fsq_phys=jnp.asarray(0.5),
        res0=jnp.asarray(0.5),
        res1=jnp.asarray(0.5),
        bad_jacobian=jnp.asarray(False),
        fsqr=jnp.asarray(0.006),
        fsqz=jnp.asarray(0.005),
        vmec2000_fact=2.0,
        use_restart_triggers=True,
        vmecpp_restart=True,
        k_preconditioner_update_interval=10,
        stage_prev_fsq=None,
        stage_transition_factor=50.0,
        vmec2000_control=True,
    )
    assert _scalar(decision.vmecpp_bad_progress)
    assert _scalar(decision.do_restart)
    assert _scalar(decision.restart_reason) == RESTART_BADPROG_VMEC
    assert _scalar(decision.irst_restart) == 3

    transition = scan_restart_transition(
        time_step=jnp.asarray(1.03),
        iter_offset=jnp.asarray(0, dtype=jnp.int32),
        ijacob=jnp.asarray(4, dtype=jnp.int32),
        bad_resets=jnp.asarray(0, dtype=jnp.int32),
        iter2=jnp.asarray(31, dtype=jnp.int32),
        restart_reason=decision.restart_reason,
        vmec2000_control=True,
        restart_badjac_factor=0.9,
        restart_badprog_factor=1.03,
        stage_transition_scale=0.5,
        step_size=1.03,
    )
    np.testing.assert_allclose(_scalar(transition.time_step), 1.0)
    assert _scalar(transition.ijacob) == 4


def test_skip_timecontrol_tightens_res0_but_suppresses_checkpoint_and_restart():
    tc = scan_time_control_scalars(
        skip_timecontrol=jnp.asarray(True),
        init_mask=jnp.asarray(True),
        fsq=jnp.asarray(100.0),
        fsq_res=jnp.asarray(100.0),
        fsq_phys=jnp.asarray(100.0),
        fsq1=jnp.asarray(0.5),
        fsq_prev_before=jnp.asarray(1.0),
        res0_prev=jnp.asarray(0.75),
        res1_prev=jnp.asarray(0.25),
        bad_jacobian=jnp.asarray(False),
        vmec2000_control=True,
    )
    assert _scalar(tc.res0) == 0.5
    assert _scalar(tc.res1) == 0.25
    assert not _scalar(tc.checkpoint_mask)

    decision = scan_restart_decision(
        skip_timecontrol=jnp.asarray(True),
        iter2=jnp.asarray(20, dtype=jnp.int32),
        iter1=jnp.asarray(0, dtype=jnp.int32),
        fsq=jnp.asarray(100.0),
        fsq_phys=jnp.asarray(100.0),
        res0=tc.res0,
        res1=tc.res1,
        bad_jacobian=jnp.asarray(True),
        fsqr=jnp.asarray(1.0),
        fsqz=jnp.asarray(1.0),
        vmec2000_fact=1.0,
        use_restart_triggers=True,
        vmecpp_restart=True,
        k_preconditioner_update_interval=1,
        stage_prev_fsq=jnp.asarray(1.0),
        stage_transition_factor=1.0,
        vmec2000_control=True,
    )
    assert not _scalar(decision.restart_time)
    assert not _scalar(decision.do_restart)
    assert _scalar(decision.restart_reason) == RESTART_NONE


def test_time_control_restart_and_non_vmec_iter_offset_branch():
    decision = scan_restart_decision(
        skip_timecontrol=jnp.asarray(False),
        iter2=jnp.asarray(12, dtype=jnp.int32),
        iter1=jnp.asarray(0, dtype=jnp.int32),
        fsq=jnp.asarray(2.1),
        fsq_phys=jnp.asarray(1.0),
        res0=jnp.asarray(1.0),
        res1=jnp.asarray(1.0),
        bad_jacobian=jnp.asarray(False),
        fsqr=jnp.asarray(0.0),
        fsqz=jnp.asarray(0.0),
        vmec2000_fact=2.0,
        use_restart_triggers=False,
        vmecpp_restart=False,
        k_preconditioner_update_interval=10,
        stage_prev_fsq=None,
        stage_transition_factor=50.0,
        vmec2000_control=False,
    )
    assert _scalar(decision.restart_time)
    assert _scalar(decision.do_restart)
    assert _scalar(decision.restart_reason) == RESTART_TIME

    transition = scan_restart_transition(
        time_step=jnp.asarray(1.03),
        iter_offset=jnp.asarray(9, dtype=jnp.int32),
        ijacob=jnp.asarray(0, dtype=jnp.int32),
        bad_resets=jnp.asarray(0, dtype=jnp.int32),
        iter2=jnp.asarray(12, dtype=jnp.int32),
        restart_reason=decision.restart_reason,
        vmec2000_control=False,
        restart_badjac_factor=0.9,
        restart_badprog_factor=1.03,
        stage_transition_scale=0.5,
        step_size=1.03,
    )
    np.testing.assert_allclose(_scalar(transition.time_step), 1.0)
    assert _scalar(transition.iter_offset) == 8


def test_ijacob_25_and_50_checkpoint_time_step_branches_preserve_damping_step():
    hit_25 = scan_restart_transition(
        time_step=jnp.asarray(10.0),
        iter_offset=jnp.asarray(0, dtype=jnp.int32),
        ijacob=jnp.asarray(24, dtype=jnp.int32),
        bad_resets=jnp.asarray(0, dtype=jnp.int32),
        iter2=jnp.asarray(1, dtype=jnp.int32),
        restart_reason=jnp.asarray(RESTART_BADJAC, dtype=jnp.int32),
        vmec2000_control=True,
        restart_badjac_factor=0.9,
        restart_badprog_factor=1.03,
        stage_transition_scale=0.5,
        step_size=2.0,
    )
    np.testing.assert_allclose(_scalar(hit_25.time_step), 1.96)
    np.testing.assert_allclose(_scalar(hit_25.damping_time_step), 9.0)
    assert _scalar(hit_25.ijacob) == 25

    hit_50 = scan_restart_transition(
        time_step=jnp.asarray(10.0),
        iter_offset=jnp.asarray(0, dtype=jnp.int32),
        ijacob=jnp.asarray(49, dtype=jnp.int32),
        bad_resets=jnp.asarray(0, dtype=jnp.int32),
        iter2=jnp.asarray(1, dtype=jnp.int32),
        restart_reason=jnp.asarray(RESTART_BADJAC, dtype=jnp.int32),
        vmec2000_control=True,
        restart_badjac_factor=0.9,
        restart_badprog_factor=1.03,
        stage_transition_scale=0.5,
        step_size=2.0,
    )
    np.testing.assert_allclose(_scalar(hit_50.time_step), 1.92)
    np.testing.assert_allclose(_scalar(hit_50.damping_time_step), 9.0)
    assert _scalar(hit_50.ijacob) == 50
