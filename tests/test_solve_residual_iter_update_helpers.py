from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax.solvers.fixed_boundary.residual.iteration_control import (
    constraint_preconditioner_channels,
    resolve_residual_iteration_control_sample,
)
from vmec_jax.solvers.fixed_boundary.residual.iteration_metrics import (
    physical_residual_metric_channels,
    select_residual_norms_for_iteration,
)
from vmec_jax.solvers.fixed_boundary.residual.iteration import (
    _FreeBoundaryEdgeControlProjector,
    _free_boundary_best_state_drift_decision,
    _free_boundary_best_state_drift_restart,
    _new_best_scored_state_tracker,
    _record_best_scored_state,
)
from vmec_jax.solvers.fixed_boundary.residual.update import (
    ResidualControllerState,
    ResidualVelocityBlocks,
    apply_controller_state_update,
    backtracking_momentum_search,
    controller_state_after_catastrophic_restart_update,
    controller_state_after_free_boundary_turnon_restart_update,
    controller_state_after_host_restart_decision_sample,
    controller_state_after_initial_axis_setup_result,
    controller_state_after_initial_axis_reset_update,
    controller_state_after_pre_restart_update,
    controller_state_after_vmec2000_time_control_sample,
    controller_state_after_vmec2000_time_control_restart_update,
    controller_state_from_resume_state,
    controller_state_from_runtime_scalars,
    controller_state_legacy_payload,
    controller_state_legacy_values,
    direct_force_fallback_acceptance_decision,
    direct_force_fallback_trial,
    force_update_rms,
    host_catastrophic_restart_update,
    host_free_boundary_turnon_restart_update,
    host_force_update_rms,
    host_initial_axis_reset_update,
    host_momentum_update_np,
    host_pre_restart_trigger_branch_result,
    host_pre_restart_trigger_update,
    host_vmec2000_time_control_restart_branch_result,
    host_vmec2000_time_control_restart_update,
    initial_residual_controller_state,
    initial_residual_velocity_state,
    jit_strict_momentum_update_proposal,
    momentum_update_jax,
    residual_evolve_coefficients,
    scale_velocity_blocks,
    DirectForceFallbackTrial,
    strict_step_branch_application,
    strict_step_branch_side_effects,
    strict_step_branch_fingerprint,
    strict_step_branch_result,
    strict_step_branch_result_after_catastrophic_restart,
    strict_step_branch_result_after_direct_fallback,
    strict_step_runtime_fields,
    strict_step_acceptance_decision,
    strict_momentum_update_proposal,
    strict_trial_evaluation,
    velocity_blocks_from_force_blocks,
    velocity_blocks_from_resume_state,
    velocity_blocks_legacy_payload,
    zero_all_velocity_blocks_like,
    zero_primary_velocity_blocks_like,
    zero_velocity_blocks_like,
)
from vmec_jax.solvers.fixed_boundary.residual.payload_blocks import ForceBlocks


def _blocks(*, offset: float, scale: float = 1.0) -> ResidualVelocityBlocks:
    base = np.arange(6.0, dtype=float).reshape(2, 3)
    return ResidualVelocityBlocks(*(scale * (base + offset + float(idx)) for idx in range(12)))


def test_best_scored_state_tracker_prefers_strict_component_max_and_counts_fresh_updates() -> None:
    tracker = _new_best_scored_state_tracker(True)

    _record_best_scored_state(
        tracker,
        state="sum-better-but-component-worse",
        iter2=1,
        fsq=(1.0e-9, 1.0e-9, 8.0e-9),
        free_boundary_enabled=True,
        freeb_ivacskip=0,
        freeb_reused=False,
    )
    _record_best_scored_state(
        tracker,
        state="strict-better",
        iter2=2,
        fsq=(3.0e-9, 3.0e-9, 3.0e-9),
        free_boundary_enabled=True,
        freeb_ivacskip=1,
        freeb_reused=True,
        freeb_bsqvac_half_current="best-bsqvac",
        freeb_nestor_runtime={"runtime": "best"},
        freeb_last_model="jax_nestor",
        freeb_last_diagnostics={"bnormal_rms": 1.0e-6},
        freeb_ivac=2,
        freeb_nvacskip=5,
        freeb_nvskip0=7,
        freeb_plascur=0.125,
    )
    _record_best_scored_state(
        tracker,
        state="turnon-rollback",
        iter2=3,
        fsq=(1.0e-12, 1.0e-12, 1.0e-12),
        free_boundary_enabled=True,
        freeb_ivacskip=0,
        freeb_reused=False,
        skip=True,
    )

    assert tracker["state"] == "strict-better"
    assert tracker["iter"] == 2
    assert tracker["component_max"] == pytest.approx(3.0e-9)
    assert tracker["fsq"] == pytest.approx(9.0e-9)
    assert tracker["full_boundary_count"] == 1
    assert tracker["fresh_boundary_count"] == 1
    assert tracker["freeb_bsqvac_half_current"] == "best-bsqvac"
    assert tracker["freeb_nestor_runtime"] == {"runtime": "best"}
    assert tracker["freeb_last_model"] == "jax_nestor"
    assert tracker["freeb_last_diagnostics"] == {"bnormal_rms": 1.0e-6}
    assert tracker["freeb_ivac"] == 2
    assert tracker["freeb_ivacskip"] == 1
    assert tracker["freeb_nvacskip"] == 5
    assert tracker["freeb_nvskip0"] == 7
    assert tracker["freeb_plascur"] == pytest.approx(0.125)


def test_free_boundary_best_state_drift_decision_requires_tail_streak_and_caps_restarts() -> None:
    tracker = _new_best_scored_state_tracker(True)
    _record_best_scored_state(
        tracker,
        state="best",
        iter2=10,
        fsq=(2.0e-10, 1.0e-10, 1.0e-11),
        free_boundary_enabled=True,
        freeb_ivacskip=0,
        freeb_reused=False,
    )

    watching = _free_boundary_best_state_drift_decision(
        tracker,
        enabled=True,
        iter2=11,
        current_fsq=(1.0e-9, 1.0e-10, 1.0e-11),
        factor=3.0,
        min_iter_since_best=5,
        streak_window=2,
        max_restarts=1,
    )
    assert not watching.restart
    assert watching.reason == "watching"
    assert watching.streak == 0

    first_tail = _free_boundary_best_state_drift_decision(
        tracker,
        enabled=True,
        iter2=15,
        current_fsq=(7.0e-10, 1.0e-10, 1.0e-11),
        factor=3.0,
        min_iter_since_best=5,
        streak_window=2,
        max_restarts=1,
    )
    assert not first_tail.restart
    assert first_tail.streak == 1
    assert first_tail.ratio == pytest.approx(3.5)

    restart = _free_boundary_best_state_drift_decision(
        tracker,
        enabled=True,
        iter2=16,
        current_fsq=(8.0e-10, 1.0e-10, 1.0e-11),
        factor=3.0,
        min_iter_since_best=5,
        streak_window=2,
        max_restarts=1,
    )
    assert restart.restart
    assert restart.reason == "freeb_best_state_drift"
    assert restart.restart_count == 1
    assert tracker["drift_last_restart_iter"] == 16

    capped = _free_boundary_best_state_drift_decision(
        tracker,
        enabled=True,
        iter2=30,
        current_fsq=(8.0e-10, 1.0e-10, 1.0e-11),
        factor=3.0,
        min_iter_since_best=5,
        streak_window=1,
        max_restarts=1,
    )
    assert not capped.restart
    assert capped.reason == "max_restarts"


def test_free_boundary_best_state_drift_restart_restores_vacuum_payload_and_damping() -> None:
    tracker = _new_best_scored_state_tracker(True)
    _record_best_scored_state(
        tracker,
        state="best-state",
        iter2=7,
        fsq=(2.0e-9, 3.0e-9, 4.0e-12),
        free_boundary_enabled=True,
        freeb_ivacskip=2,
        freeb_reused=True,
        freeb_bsqvac_half_current="bsqvac-best",
        freeb_nestor_runtime={"model": "direct"},
        freeb_last_model="jax_direct",
        freeb_last_diagnostics={"bnormal_rms": 1.0e-8},
        freeb_ivac=5,
        freeb_nvacskip=11,
        freeb_nvskip0=13,
        freeb_plascur=0.42,
    )
    decision = _free_boundary_best_state_drift_decision(
        tracker,
        enabled=True,
        iter2=10,
        current_fsq=(9.0e-9, 1.0e-9, 1.0e-12),
        factor=2.0,
        min_iter_since_best=1,
        streak_window=1,
        max_restarts=3,
    )

    restart = _free_boundary_best_state_drift_restart(
        tracker,
        decision,
        freeb_ivac=99,
        freeb_ivacskip=99,
        freeb_nvacskip=99,
        freeb_nvskip0=99,
        freeb_plascur=99.0,
        time_step=0.02,
        restart_badprog_factor=0.5,
        k_ndamp=3,
        iter2=10,
        ijacob=4,
        bad_resets=6,
        fsq_prev=1.0,
        res0=1.0e-8,
        res1=-1.0,
    )

    assert restart is not None
    assert restart.state == "best-state"
    assert restart.freeb_bsqvac_half_current == "bsqvac-best"
    assert restart.freeb_nestor_runtime == {"model": "direct"}
    assert restart.freeb_last_model == "jax_direct"
    assert restart.freeb_last_diagnostics == {"bnormal_rms": 1.0e-8}
    assert restart.freeb_ivac == 5
    assert restart.freeb_ivacskip == 2
    assert restart.freeb_nvacskip == 11
    assert restart.freeb_nvskip0 == 13
    assert restart.freeb_plascur == pytest.approx(0.42)
    assert restart.time_step == pytest.approx(0.01)
    assert restart.inv_tau == pytest.approx([15.0, 15.0, 15.0])
    assert restart.iter1 == 10
    assert restart.ijacob == 5
    assert restart.bad_resets == 7
    assert restart.fsq_prev == pytest.approx(5.004e-9)
    assert restart.fsq0_prev == pytest.approx(restart.fsq_prev)
    assert restart.prev_rz_fsq == pytest.approx(5.0e-9)
    assert restart.res0 == pytest.approx(5.004e-9)
    assert restart.res1 == pytest.approx(5.004e-9)
    assert restart.state_checkpoint == "best-state"
    assert restart.step_status == "restart_freeb_drift"
    assert restart.restart_reason == "freeb_best_state_drift"
    assert restart.pre_restart_reason == "freeb_best_state_drift"


def test_free_boundary_edge_coordinate_mode_applies_reduced_update_once(monkeypatch) -> None:
    def fake_prepare(payload, **_kwargs):
        return {
            "enabled": True,
            "info": {"enabled": True, "basis_symmetry": "square"},
            "mode_count": 1,
            "mode_scale_np": np.ones(1),
            "pinv_np": np.ones((1, 4)),
        }

    monkeypatch.setattr(
        "vmec_jax.solvers.fixed_boundary.residual.iteration._prepare_freeb_edge_control_projection",
        fake_prepare,
    )

    coordinate = _FreeBoundaryEdgeControlProjector(
        {"update_mode": "coordinate"},
        indata=object(),
        static=object(),
        state0=object(),
        free_boundary_enabled=True,
        use_scan=False,
        jit_strict_update_enabled=True,
    )
    native = _FreeBoundaryEdgeControlProjector(
        {"update_mode": "native_coordinate"},
        indata=object(),
        static=object(),
        state0=object(),
        free_boundary_enabled=True,
        use_scan=False,
        jit_strict_update_enabled=True,
    )
    projected = _FreeBoundaryEdgeControlProjector(
        {"update_mode": "projected_delta"},
        indata=object(),
        static=object(),
        state0=object(),
        free_boundary_enabled=True,
        use_scan=False,
        jit_strict_update_enabled=True,
    )

    assert coordinate.delta_tuple_projector() is None
    assert native.delta_tuple_projector() is None
    assert native.update_mode == "native_coordinate"
    assert native.info["solver_native_spline_controls"] is True
    assert callable(projected.delta_tuple_projector())


def test_residual_iteration_control_sample_matches_vmec2000_edge_and_precond_rules() -> None:
    initial = resolve_residual_iteration_control_sample(
        iter2=1,
        iter1=0,
        vmec2000_control=True,
        free_boundary_enabled=False,
        freeb_ivac_effective=0,
        prev_rz_fsq=1.0,
        fsqz2_history=[],
        env_freeb_include_edge=False,
        env_force_edge_residual="",
        precond_cache_valid=False,
        force_bcovar_update=False,
        preconditioner_update_interval=25,
        ns=16,
    )
    assert initial.iter_since_restart == 1
    assert initial.zero_m1_value == pytest.approx(1.0)
    assert not initial.include_edge
    assert not initial.include_edge_residual
    assert initial.precond_jmax_override is None
    assert initial.precond_expected_jmax == 15
    assert initial.need_bcovar_update
    assert not initial.use_cached_precond

    active_free_boundary = resolve_residual_iteration_control_sample(
        iter2=3,
        iter1=1,
        vmec2000_control=True,
        free_boundary_enabled=True,
        freeb_ivac_effective=1,
        prev_rz_fsq=1.0,
        fsqz2_history=[1.0e-4],
        env_freeb_include_edge=False,
        env_force_edge_residual="",
        precond_cache_valid=True,
        force_bcovar_update=False,
        preconditioner_update_interval=25,
        ns=16,
    )
    assert active_free_boundary.zero_m1_value == pytest.approx(0.0)
    assert not active_free_boundary.include_edge
    assert active_free_boundary.include_edge_residual
    assert active_free_boundary.precond_jmax_override == 16
    assert active_free_boundary.precond_expected_jmax == 16
    assert not active_free_boundary.need_bcovar_update
    assert active_free_boundary.use_cached_precond


def test_residual_iteration_control_sample_non_vmec_restart_heuristics() -> None:
    sample = resolve_residual_iteration_control_sample(
        iter2=7,
        iter1=6,
        vmec2000_control=False,
        free_boundary_enabled=False,
        freeb_ivac_effective=0,
        prev_rz_fsq=5.0e-8,
        fsqz2_history=[2.0],
        env_freeb_include_edge=False,
        env_force_edge_residual="true",
        precond_cache_valid=True,
        force_bcovar_update=True,
        preconditioner_update_interval=25,
        ns=8,
    )
    assert sample.iter_since_restart == 1
    assert sample.zero_m1_value == pytest.approx(1.0)
    assert sample.include_edge
    assert sample.include_edge_residual
    assert sample.precond_expected_jmax == 7
    assert not sample.need_bcovar_update
    assert not sample.use_cached_precond


def test_constraint_preconditioner_channels_select_cached_or_zero_payloads() -> None:
    cached_diag = (np.array([1.0]), np.array([2.0]))
    cached_tcon = np.array([3.0])
    zero_diag = (np.array([0.0]), np.array([0.0]))
    zero_tcon = np.array([0.0])

    cached = constraint_preconditioner_channels(
        use_cached_precond=True,
        cached_precond_diag=cached_diag,
        cached_tcon=cached_tcon,
        zero_precond_diag=zero_diag,
        zero_tcon=zero_tcon,
        host_update_assembly=True,
        jnp_true_bool="true-sentinel",
        jnp_false_bool="false-sentinel",
        jnp_module=np,
    )
    assert cached.precond_diag is cached_diag
    assert cached.tcon is cached_tcon
    assert cached.precond_active == "true-sentinel"
    assert cached.tcon_active == "true-sentinel"

    zero = constraint_preconditioner_channels(
        use_cached_precond=False,
        cached_precond_diag=cached_diag,
        cached_tcon=cached_tcon,
        zero_precond_diag=zero_diag,
        zero_tcon=zero_tcon,
        host_update_assembly=False,
        jnp_true_bool=None,
        jnp_false_bool=None,
        jnp_module=np,
    )
    assert zero.precond_diag is zero_diag
    assert zero.tcon is zero_tcon
    assert np.asarray(zero.precond_active).shape == ()
    assert bool(zero.precond_active) is False
    assert bool(zero.tcon_active) is False


def test_residual_metric_channels_use_cached_norms_and_preserve_device_path() -> None:
    current = SimpleNamespace(r1=2.0, fnorm=3.0, fnormL=5.0)
    cached = SimpleNamespace(r1=7.0, fnorm=11.0, fnormL=13.0)

    assert (
        select_residual_norms_for_iteration(
            vmec2000_control=True,
            precond_cache_valid=True,
            need_bcovar_update=False,
            cached_norms=cached,
            current_norms=current,
        )
        is cached
    )
    assert (
        select_residual_norms_for_iteration(
            vmec2000_control=True,
            precond_cache_valid=True,
            need_bcovar_update=True,
            cached_norms=cached,
            current_norms=current,
        )
        is current
    )

    device_metrics = physical_residual_metric_channels(
        gcr2=np.array(2.0),
        gcz2=np.array(3.0),
        gcl2=np.array(4.0),
        norms_used=current,
        host_update_assembly=False,
        use_host_residual_metrics=False,
        device_get_floats=lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected host sync")),
    )
    assert device_metrics.norms_used is current
    np.testing.assert_allclose(device_metrics.fsqr, 12.0)
    np.testing.assert_allclose(device_metrics.fsqz, 18.0)
    np.testing.assert_allclose(device_metrics.fsql, 20.0)


def test_residual_metric_channels_host_sync_path_pulls_expected_scalars() -> None:
    norms = SimpleNamespace(r1=np.array(2.0), fnorm=np.array(3.0), fnormL=np.array(5.0))
    calls = []

    def fake_device_get_floats(*vals):
        calls.append(vals)
        return tuple(float(np.asarray(value)) for value in vals)

    metrics = physical_residual_metric_channels(
        gcr2=np.array(2.0),
        gcz2=np.array(3.0),
        gcl2=np.array(4.0),
        norms_used=norms,
        host_update_assembly=False,
        use_host_residual_metrics=True,
        device_get_floats=fake_device_get_floats,
    )

    assert len(calls) == 1
    assert len(calls[0]) == 6
    assert metrics.fsqr == pytest.approx(12.0)
    assert metrics.fsqz == pytest.approx(18.0)
    assert metrics.fsql == pytest.approx(20.0)


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


def test_velocity_blocks_from_force_blocks_preserves_vmec_channel_mapping() -> None:
    blocks = ForceBlocks(*(f"force-{name}" for name in ForceBlocks._fields))

    got = velocity_blocks_from_force_blocks(blocks)

    assert got == ResidualVelocityBlocks(
        rcc="force-frcc",
        rss="force-frss",
        rsc="force-frsc",
        rcs="force-frcs",
        zsc="force-fzsc",
        zcs="force-fzcs",
        zcc="force-fzcc",
        zss="force-fzss",
        lsc="force-flsc",
        lcs="force-flcs",
        lcc="force-flcc",
        lss="force-flss",
    )


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


def test_controller_state_from_runtime_scalars_normalizes_legacy_slots() -> None:
    checkpoint = object()

    state = controller_state_from_runtime_scalars(
        time_step="0.125",
        inv_tau=(1.0, 2.0, 3.0),
        fsq_prev="4.0",
        fsq0_prev=np.asarray(5.0),
        flip_sign="-1.0",
        iter1="6",
        ijacob=np.asarray(7),
        bad_resets="8",
        res0=np.asarray(0.25),
        res1="0.125",
        prev_rz_fsq="0.0625",
        bad_growth_streak=np.asarray(9),
        huge_force_restart_count="10",
        state_checkpoint=checkpoint,
    )

    assert state.time_step == pytest.approx(0.125)
    assert state.inv_tau == [1.0, 2.0, 3.0]
    assert state.fsq_prev == pytest.approx(4.0)
    assert state.fsq0_prev == pytest.approx(5.0)
    assert state.flip_sign == pytest.approx(-1.0)
    assert state.iter1 == 6
    assert state.ijacob == 7
    assert state.bad_resets == 8
    assert state.res0 == pytest.approx(0.25)
    assert state.res1 == pytest.approx(0.125)
    assert state.prev_rz_fsq == pytest.approx(0.0625)
    assert state.bad_growth_streak == 9
    assert state.huge_force_restart_count == 10
    assert state.state_checkpoint is checkpoint


def test_controller_state_legacy_values_follow_explicit_resume_key_order() -> None:
    checkpoint = object()
    state = ResidualControllerState(
        time_step=0.1,
        inv_tau=[1.0, 2.0],
        fsq_prev=3.0,
        fsq0_prev=4.0,
        flip_sign=-1.0,
        iter1=5,
        ijacob=6,
        bad_resets=7,
        res0=8.0,
        res1=9.0,
        prev_rz_fsq=10.0,
        bad_growth_streak=11,
        huge_force_restart_count=12,
        state_checkpoint=checkpoint,
    )

    values = controller_state_legacy_values(state)

    assert values == (
        state.time_step,
        state.inv_tau,
        state.fsq_prev,
        state.fsq0_prev,
        state.flip_sign,
        state.iter1,
        state.ijacob,
        state.bad_resets,
        state.res0,
        state.res1,
        state.prev_rz_fsq,
        state.bad_growth_streak,
        state.huge_force_restart_count,
        checkpoint,
    )
    assert dict(zip(controller_state_legacy_payload(state), values)) == controller_state_legacy_payload(state)


def test_apply_controller_state_update_delegates_to_pure_update() -> None:
    state = initial_residual_controller_state(
        step_size=0.2,
        k_ndamp=2,
        initial_flip_sign=-1.0,
        state_checkpoint="checkpoint",
    )
    update = host_initial_axis_reset_update(
        state_checkpoint="new-checkpoint",
        time_step=state.time_step,
        iter2=3,
        prev_rz_fsq_before=0.5,
        k_ndamp=2,
    )

    got = apply_controller_state_update(state, controller_state_after_initial_axis_reset_update, update)

    assert got.iter1 == 3
    assert got.ijacob == 1
    assert got.prev_rz_fsq == pytest.approx(0.5)
    assert got.state_checkpoint == "new-checkpoint"


def test_free_boundary_turnon_restart_update_matches_vmec_retry_semantics() -> None:
    base = initial_residual_controller_state(
        step_size=0.3,
        k_ndamp=3,
        initial_flip_sign=-1.0,
        state_checkpoint="old-checkpoint",
    )._replace(iter1=4, ijacob=2, bad_growth_streak=6)

    update = host_free_boundary_turnon_restart_update(
        state_checkpoint="turnon-checkpoint",
        time_step=base.time_step,
        iter2=9,
        iter1=base.iter1,
        ijacob=base.ijacob,
        k_ndamp=3,
        reset_iter1=True,
    )
    got = controller_state_after_free_boundary_turnon_restart_update(base, update)

    assert update.state == "turnon-checkpoint"
    assert update.time_step_report_hold == pytest.approx(0.3)
    assert update.ijacob == 3
    assert update.iter1 == 9
    np.testing.assert_allclose(update.inv_tau, [0.5, 0.5, 0.5])
    assert got.iter1 == 9
    assert got.ijacob == 3
    assert got.bad_growth_streak == 0
    assert got.fsq_prev == pytest.approx(base.fsq_prev)
    assert got.state_checkpoint == "old-checkpoint"


def test_free_boundary_turnon_restart_update_can_preserve_restart_marker() -> None:
    update = host_free_boundary_turnon_restart_update(
        state_checkpoint="checkpoint",
        time_step=0.0,
        iter2=9,
        iter1=4,
        ijacob=2,
        k_ndamp=2,
        reset_iter1=False,
    )

    assert update.iter1 == 4
    assert update.ijacob == 3
    np.testing.assert_allclose(update.inv_tau, [0.15 / 1.0e-12] * 2)


def test_strict_step_acceptance_decision_covers_accept_reject_and_nonfinite_paths() -> None:
    accepted = strict_step_acceptance_decision(w_try=1.0005, w_curr=1.0, backtracking=True)
    rejected = strict_step_acceptance_decision(w_try=1.01, w_curr=1.0, backtracking=True)
    nonfinite = strict_step_acceptance_decision(w_try=np.nan, w_curr=1.0, backtracking=False)
    no_backtracking = strict_step_acceptance_decision(w_try=10.0, w_curr=1.0, backtracking=False)

    assert accepted.accepted
    assert accepted.accept_ratio == pytest.approx(1.001)
    assert not rejected.accepted
    assert not nonfinite.accepted
    assert no_backtracking.accepted
    assert np.isinf(no_backtracking.accept_ratio)


def test_strict_step_branch_result_packages_accepted_momentum_status() -> None:
    acceptance = strict_step_acceptance_decision(w_try=0.5, w_curr=1.0, backtracking=True)

    result = strict_step_branch_result(
        acceptance=acceptance,
        state_try="trial-state",
        state_backup="backup-state",
        update_rms=0.25,
        vmec2000_control=False,
        huge_force_restart_count=7,
    )

    assert result.state == "trial-state"
    assert result.accepted
    assert not result.catastrophic_restart
    assert not result.clear_cache_after_catastrophic
    assert result.step_status == "momentum"
    assert result.restart_reason == "none"
    assert result.restart_path == "momentum_accept"
    assert result.huge_force_restart_count == 0
    assert result.update_rms == pytest.approx(0.25)


def test_strict_step_branch_result_packages_rejected_restart_status() -> None:
    acceptance = strict_step_acceptance_decision(w_try=np.inf, w_curr=1.0, backtracking=True)

    non_vmec = strict_step_branch_result(
        acceptance=acceptance,
        state_try="trial-state",
        state_backup="backup-state",
        update_rms=None,
        vmec2000_control=False,
        huge_force_restart_count=7,
    )
    vmec = strict_step_branch_result(
        acceptance=acceptance,
        state_try="trial-state",
        state_backup="backup-state",
        update_rms=None,
        vmec2000_control=True,
        huge_force_restart_count=7,
    )

    assert non_vmec.state == "backup-state"
    assert not non_vmec.accepted
    assert non_vmec.catastrophic_restart
    assert non_vmec.clear_cache_after_catastrophic
    assert non_vmec.step_status == "restart_pending"
    assert non_vmec.restart_reason == "trial_rejected"
    assert non_vmec.restart_path == "trial_rejected"
    assert non_vmec.huge_force_restart_count == 7
    assert non_vmec.update_rms is None
    assert not vmec.clear_cache_after_catastrophic


def test_direct_force_fallback_acceptance_decision_uses_vmec_trial_threshold() -> None:
    accepted = direct_force_fallback_acceptance_decision(residual=1.49, current_residual=1.0)
    rejected = direct_force_fallback_acceptance_decision(residual=1.51, current_residual=1.0)
    nonfinite = direct_force_fallback_acceptance_decision(residual=np.inf, current_residual=1.0)
    custom = direct_force_fallback_acceptance_decision(residual=1.9, current_residual=1.0, accept_ratio=2.0)

    assert accepted.accepted
    assert accepted.accept_ratio == pytest.approx(1.5)
    assert not rejected.accepted
    assert not nonfinite.accepted
    assert custom.accepted
    assert custom.accept_ratio == pytest.approx(2.0)


def test_strict_step_branch_result_after_direct_fallback_accepts_trial_state() -> None:
    branch = strict_step_branch_result(
        acceptance=strict_step_acceptance_decision(w_try=np.inf, w_curr=1.0, backtracking=True),
        state_try="trial-state",
        state_backup="backup-state",
        update_rms=None,
        vmec2000_control=False,
        huge_force_restart_count=3,
    )
    fallback = DirectForceFallbackTrial(
        state="fallback-state",
        dt_eff=0.0125,
        update_rms=0.125,
        residual=1.0,
    )

    result = strict_step_branch_result_after_direct_fallback(
        branch=branch,
        fallback_trial=fallback,
        acceptance=direct_force_fallback_acceptance_decision(residual=1.0, current_residual=1.0),
        clear_cache_after_rejected=True,
    )

    assert result.state == "fallback-state"
    assert result.accepted
    assert not result.catastrophic_restart
    assert not result.clear_cache_after_catastrophic
    assert result.step_status == "fallback_direct"
    assert result.restart_reason == "none"
    assert result.restart_path == "fallback_direct"
    assert result.huge_force_restart_count == 0
    assert result.update_rms == pytest.approx(0.125)
    assert result.fallback_direct_dt == pytest.approx(0.0125)


def test_strict_step_branch_result_after_direct_fallback_preserves_rejected_branch() -> None:
    branch = strict_step_branch_result(
        acceptance=strict_step_acceptance_decision(w_try=np.inf, w_curr=1.0, backtracking=True),
        state_try="trial-state",
        state_backup="backup-state",
        update_rms=0.5,
        vmec2000_control=False,
        huge_force_restart_count=3,
    )
    fallback = DirectForceFallbackTrial(
        state="fallback-state",
        dt_eff=0.0125,
        update_rms=0.125,
        residual=3.0,
    )

    result = strict_step_branch_result_after_direct_fallback(
        branch=branch,
        fallback_trial=fallback,
        acceptance=direct_force_fallback_acceptance_decision(residual=3.0, current_residual=1.0),
        clear_cache_after_rejected=False,
    )

    assert result.state == "backup-state"
    assert not result.accepted
    assert result.catastrophic_restart
    assert not result.clear_cache_after_catastrophic
    assert result.step_status == "restart_pending"
    assert result.restart_path == "trial_rejected"
    assert result.update_rms == pytest.approx(0.5)
    assert result.fallback_direct_dt is None


def test_strict_step_branch_result_after_catastrophic_restart_carries_policy_outputs() -> None:
    branch = strict_step_branch_result(
        acceptance=strict_step_acceptance_decision(w_try=np.inf, w_curr=1.0, backtracking=True),
        state_try="trial-state",
        state_backup="backup-state",
        update_rms=0.5,
        vmec2000_control=False,
        huge_force_restart_count=3,
    )
    restart = host_catastrophic_restart_update(
        probe_bad_jacobian=True,
        w_try=np.inf,
        time_step=0.2,
        restart_badjac_factor=0.9,
        restart_badprog_factor=1.03,
        step_size=0.1,
        ijacob=4,
        bad_resets=5,
        iter2=9,
        fsq_prev_before=1.25,
        fsq0_prev_before=2.5,
        k_ndamp=2,
        max_coeff_delta_rms=1.0e-5,
        max_update_rms=5.0e-3,
    )

    result = strict_step_branch_result_after_catastrophic_restart(
        branch=branch,
        restart_update=restart,
        state_backup="rollback-state",
    )

    assert result.state == "rollback-state"
    assert not result.accepted
    assert result.catastrophic_restart
    assert result.clear_cache_after_catastrophic == branch.clear_cache_after_catastrophic
    assert result.step_status == restart.step_status
    assert result.restart_reason == restart.restart_reason
    assert result.restart_path == restart.restart_path
    assert result.update_rms == pytest.approx(restart.update_rms)
    assert result.max_coeff_delta_rms == pytest.approx(restart.max_coeff_delta_rms)
    assert result.max_update_rms == pytest.approx(restart.max_update_rms)


def test_strict_step_branch_fingerprint_is_array_free_and_path_specific() -> None:
    accepted = strict_step_branch_result(
        acceptance=strict_step_acceptance_decision(w_try=0.5, w_curr=1.0, backtracking=True),
        state_try=np.asarray([1.0]),
        state_backup=np.asarray([0.0]),
        update_rms=0.25,
        vmec2000_control=False,
        huge_force_restart_count=3,
    )
    rejected = strict_step_branch_result(
        acceptance=strict_step_acceptance_decision(w_try=np.inf, w_curr=1.0, backtracking=True),
        state_try=np.asarray([1.0]),
        state_backup=np.asarray([0.0]),
        update_rms=None,
        vmec2000_control=False,
        huge_force_restart_count=3,
    )
    fallback = strict_step_branch_result_after_direct_fallback(
        branch=rejected,
        fallback_trial=DirectForceFallbackTrial(
            state=np.asarray([2.0]),
            dt_eff=0.01,
            update_rms=0.1,
            residual=1.0,
        ),
        acceptance=direct_force_fallback_acceptance_decision(residual=1.0, current_residual=1.0),
        clear_cache_after_rejected=True,
    )
    restart = host_catastrophic_restart_update(
        probe_bad_jacobian=True,
        w_try=np.inf,
        time_step=0.2,
        restart_badjac_factor=0.9,
        restart_badprog_factor=1.03,
        step_size=0.1,
        ijacob=4,
        bad_resets=5,
        iter2=9,
        fsq_prev_before=1.25,
        fsq0_prev_before=2.5,
        k_ndamp=2,
        max_coeff_delta_rms=1.0e-5,
        max_update_rms=5.0e-3,
    )
    catastrophic = strict_step_branch_result_after_catastrophic_restart(
        branch=rejected,
        restart_update=restart,
        state_backup=np.asarray([0.0]),
    )

    assert strict_step_branch_fingerprint(accepted) == (
        "momentum_accept",
        True,
        False,
        False,
        "none",
        "momentum",
        False,
    )
    assert strict_step_branch_fingerprint(fallback) == (
        "fallback_direct",
        True,
        False,
        False,
        "none",
        "fallback_direct",
        True,
    )
    assert strict_step_branch_fingerprint(catastrophic) == (
        restart.restart_path,
        False,
        True,
        True,
        restart.restart_reason,
        restart.step_status,
        False,
    )


def test_strict_step_runtime_fields_preserve_current_caps_for_accepted_branch() -> None:
    state_try = np.asarray([1.0, 2.0])
    branch = strict_step_branch_result(
        acceptance=strict_step_acceptance_decision(w_try=0.5, w_curr=1.0, backtracking=True),
        state_try=state_try,
        state_backup=np.asarray([0.0, 0.0]),
        update_rms=0.25,
        vmec2000_control=False,
        huge_force_restart_count=3,
    )

    fields = strict_step_runtime_fields(
        branch,
        max_coeff_delta_rms=1.0e-5,
        max_update_rms=2.0e-3,
    )

    assert fields.state is state_try
    assert fields.step_status == "momentum"
    assert fields.restart_reason == "none"
    assert fields.restart_path == "momentum_accept"
    assert fields.huge_force_restart_count == 0
    assert fields.update_rms == pytest.approx(0.25)
    assert fields.max_coeff_delta_rms == pytest.approx(1.0e-5)
    assert fields.max_update_rms == pytest.approx(2.0e-3)


def test_strict_step_runtime_fields_use_catastrophic_branch_caps() -> None:
    rejected = strict_step_branch_result(
        acceptance=strict_step_acceptance_decision(w_try=np.inf, w_curr=1.0, backtracking=True),
        state_try=np.asarray([1.0]),
        state_backup=np.asarray([0.0]),
        update_rms=None,
        vmec2000_control=False,
        huge_force_restart_count=3,
    )
    restart = host_catastrophic_restart_update(
        probe_bad_jacobian=True,
        w_try=np.inf,
        time_step=0.2,
        restart_badjac_factor=0.9,
        restart_badprog_factor=1.03,
        step_size=0.1,
        ijacob=4,
        bad_resets=5,
        iter2=9,
        fsq_prev_before=1.25,
        fsq0_prev_before=2.5,
        k_ndamp=2,
        max_coeff_delta_rms=1.0e-5,
        max_update_rms=5.0e-3,
    )
    catastrophic = strict_step_branch_result_after_catastrophic_restart(
        branch=rejected,
        restart_update=restart,
        state_backup=np.asarray([0.0]),
    )

    fields = strict_step_runtime_fields(
        catastrophic,
        max_coeff_delta_rms=9.0e-5,
        max_update_rms=9.0e-3,
    )

    assert fields.step_status == restart.step_status
    assert fields.restart_reason == restart.restart_reason
    assert fields.restart_path == restart.restart_path
    assert fields.huge_force_restart_count == catastrophic.huge_force_restart_count
    assert fields.update_rms == pytest.approx(restart.update_rms)
    assert fields.max_coeff_delta_rms == pytest.approx(restart.max_coeff_delta_rms)
    assert fields.max_update_rms == pytest.approx(restart.max_update_rms)


def test_strict_step_branch_side_effects_capture_velocity_and_cache_policy() -> None:
    accepted = strict_step_branch_result(
        acceptance=strict_step_acceptance_decision(w_try=0.5, w_curr=1.0, backtracking=True),
        state_try=np.asarray([1.0]),
        state_backup=np.asarray([0.0]),
        update_rms=0.25,
        vmec2000_control=False,
        huge_force_restart_count=3,
    )
    rejected = strict_step_branch_result(
        acceptance=strict_step_acceptance_decision(w_try=np.inf, w_curr=1.0, backtracking=True),
        state_try=np.asarray([1.0]),
        state_backup=np.asarray([0.0]),
        update_rms=None,
        vmec2000_control=False,
        huge_force_restart_count=3,
    )
    fallback = strict_step_branch_result_after_direct_fallback(
        branch=rejected,
        fallback_trial=DirectForceFallbackTrial(
            state=np.asarray([2.0]),
            dt_eff=0.01,
            update_rms=0.1,
            residual=1.0,
        ),
        acceptance=direct_force_fallback_acceptance_decision(residual=1.0, current_residual=1.0),
        clear_cache_after_rejected=True,
    )

    assert strict_step_branch_side_effects(accepted) == (False, False, False, False)
    assert strict_step_branch_side_effects(fallback) == (True, False, False, False)
    assert strict_step_branch_side_effects(rejected) == (False, True, False, False)
    assert strict_step_branch_side_effects(rejected, after_catastrophic_restart=True) == (
        False,
        False,
        True,
        True,
    )


def test_strict_step_branch_application_couples_runtime_and_side_effects() -> None:
    rejected = strict_step_branch_result(
        acceptance=strict_step_acceptance_decision(w_try=np.inf, w_curr=1.0, backtracking=True),
        state_try=np.asarray([1.0]),
        state_backup=np.asarray([0.0]),
        update_rms=None,
        vmec2000_control=False,
        huge_force_restart_count=3,
    )
    restart = host_catastrophic_restart_update(
        probe_bad_jacobian=True,
        w_try=np.inf,
        time_step=0.2,
        restart_badjac_factor=0.9,
        restart_badprog_factor=1.03,
        step_size=0.1,
        ijacob=4,
        bad_resets=5,
        iter2=9,
        fsq_prev_before=1.25,
        fsq0_prev_before=2.5,
        k_ndamp=2,
        max_coeff_delta_rms=1.0e-5,
        max_update_rms=5.0e-3,
    )
    catastrophic = strict_step_branch_result_after_catastrophic_restart(
        branch=rejected,
        restart_update=restart,
        state_backup=np.asarray([0.0]),
    )

    application = strict_step_branch_application(
        catastrophic,
        max_coeff_delta_rms=9.0e-5,
        max_update_rms=9.0e-3,
        after_catastrophic_restart=True,
    )

    assert application.runtime.restart_path == restart.restart_path
    assert application.runtime.max_update_rms == pytest.approx(restart.max_update_rms)
    assert application.side_effects == (False, False, True, True)


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


def test_controller_state_applies_initial_axis_setup_result() -> None:
    state = initial_residual_controller_state(
        step_size=0.25,
        k_ndamp=4,
        initial_flip_sign=-1.0,
        state_checkpoint="old-checkpoint",
    )._replace(iter1=7, bad_resets=2, bad_growth_streak=3)

    checkpoint = object()
    result = controller_state_after_initial_axis_setup_result(
        state,
        SimpleNamespace(
            ijacob=np.asarray(2),
            res0="0.75",
            res1=np.asarray(0.5),
            prev_rz_fsq="0.25",
            state_checkpoint=checkpoint,
        ),
    )

    assert result.ijacob == 2
    assert result.res0 == pytest.approx(0.75)
    assert result.res1 == pytest.approx(0.5)
    assert result.prev_rz_fsq == pytest.approx(0.25)
    assert result.state_checkpoint is checkpoint
    assert result.iter1 == state.iter1
    assert result.bad_resets == state.bad_resets
    assert result.bad_growth_streak == state.bad_growth_streak


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


def test_initial_axis_reset_update_preserves_vmec_retry_scalars() -> None:
    base = initial_residual_controller_state(
        step_size=0.3,
        k_ndamp=2,
        initial_flip_sign=-1.0,
        state_checkpoint="old-checkpoint",
    )._replace(fsq_prev=8.0, fsq0_prev=7.0, res0=0.5, res1=0.25)
    update = host_initial_axis_reset_update(
        state_checkpoint="new-checkpoint",
        time_step=base.time_step,
        iter2=1,
        prev_rz_fsq_before=0.125,
        k_ndamp=2,
    )

    got = controller_state_after_initial_axis_reset_update(base, update)

    assert got.time_step == pytest.approx(0.3)
    assert got.inv_tau == [0.5, 0.5]
    assert got.iter1 == 1
    assert got.ijacob == 1
    assert got.prev_rz_fsq == pytest.approx(0.125)
    assert got.bad_growth_streak == 0
    assert got.state_checkpoint == "new-checkpoint"
    assert got.fsq_prev == pytest.approx(8.0)
    assert got.fsq0_prev == pytest.approx(7.0)
    assert got.res0 == pytest.approx(0.5)
    assert got.res1 == pytest.approx(0.25)


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


def test_controller_state_applies_vmec2000_time_control_samples() -> None:
    state = initial_residual_controller_state(
        step_size=0.2,
        k_ndamp=2,
        initial_flip_sign=-1.0,
        state_checkpoint="old-checkpoint",
    )._replace(res0=0.1, res1=0.2)

    sample = controller_state_after_vmec2000_time_control_sample(
        state,
        SimpleNamespace(res0=1.5, res1=2.5, initialized=False, store_checkpoint=False),
        state_checkpoint="new-checkpoint",
    )

    assert sample.res0 == pytest.approx(1.5)
    assert sample.res1 == pytest.approx(2.5)
    assert sample.state_checkpoint == "old-checkpoint"
    assert sample.time_step == pytest.approx(state.time_step)
    assert sample.inv_tau == state.inv_tau

    checkpointed = controller_state_after_vmec2000_time_control_sample(
        sample,
        SimpleNamespace(res0=3.5, res1=4.5, initialized=True, store_checkpoint=False),
        state_checkpoint="new-checkpoint",
    )

    assert checkpointed.res0 == pytest.approx(3.5)
    assert checkpointed.res1 == pytest.approx(4.5)
    assert checkpointed.state_checkpoint == "new-checkpoint"


def test_controller_state_applies_host_restart_decision_samples() -> None:
    state = initial_residual_controller_state(
        step_size=0.2,
        k_ndamp=2,
        initial_flip_sign=-1.0,
        state_checkpoint="old-checkpoint",
    )._replace(res0=0.1, bad_growth_streak=2)

    sample = controller_state_after_host_restart_decision_sample(
        state,
        SimpleNamespace(res0=0.25, bad_growth_streak=3, store_checkpoint=False),
        state_checkpoint="new-checkpoint",
    )

    assert sample.res0 == pytest.approx(0.25)
    assert sample.bad_growth_streak == 3
    assert sample.state_checkpoint == "old-checkpoint"
    assert sample.res1 == pytest.approx(state.res1)

    checkpointed = controller_state_after_host_restart_decision_sample(
        sample,
        SimpleNamespace(res0=0.5, bad_growth_streak=0, store_checkpoint=True),
        state_checkpoint="new-checkpoint",
    )

    assert checkpointed.res0 == pytest.approx(0.5)
    assert checkpointed.bad_growth_streak == 0
    assert checkpointed.state_checkpoint == "new-checkpoint"


def test_vmec2000_time_control_restart_branch_result_packages_side_effects() -> None:
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

    got = host_vmec2000_time_control_restart_branch_result(
        state_checkpoint="checkpoint-state",
        restart_update=update,
        pre_restart_reason="bad_jacobian",
        prev_rz_fsq_before=0.125,
    )

    assert got.state == "checkpoint-state"
    assert got.update is update
    assert got.step_status == "restart_bad_jacobian"
    assert got.restart_reason == "bad_jacobian"
    assert got.restart_path == "vmec2000_bad_jacobian"
    assert got.pre_restart_reason == "bad_jacobian"
    assert got.prev_rz_fsq == pytest.approx(0.125)
    assert got.clear_freeb_controls is True
    assert got.clear_preconditioner_cache is True
    assert got.force_bcovar_update is True
    assert got.pop_iteration_history is True
    assert got.skip_time_control is True


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


def test_jit_strict_momentum_update_proposal_preserves_vmec_channel_order() -> None:
    velocities = _blocks(offset=0.0)
    forces = _blocks(offset=20.0)
    captured = {}
    output_keys = (
        "vRcc_after vRss_after vRsc_after vRcs_after vZsc_after vZcs_after "
        "vZcc_after vZss_after vLsc_after vLcs_after vLcc_after vLss_after"
    ).split()

    def strict_update_step_jit_func(static, **kwargs):
        captured["static"] = static
        captured["kwargs"] = kwargs

        def step_fn(*args):
            captured["args"] = args
            return {"state_post": "updated-state", "update_rms_postclip": np.asarray(0.125)} | {
                key: np.asarray([float(idx)]) for idx, key in enumerate(output_keys, start=1)
            }

        return step_fn

    result = jit_strict_momentum_update_proposal(
        state="state",
        static="static",
        velocities=velocities,
        forces=forces,
        dt_eff=0.2,
        b1=0.3,
        fac=0.4,
        force_scale=0.5,
        flip_sign=-1.0,
        max_update_rms=0.6,
        need_update_rms=True,
        divide_by_scalxc_for_update=True,
        free_boundary_enabled=False,
        strict_update_step_jit_func=strict_update_step_jit_func,
    )

    input_attrs = "rcc rss zsc zcs lsc lcs rsc rcs zcc zss lcc lss".split()
    expected_channels = [getattr(velocities, name) for name in input_attrs]
    expected_channels += [getattr(forces, name) for name in input_attrs]
    assert captured == {
        "static": "static",
        "kwargs": {
            "limit_update_rms": False,
            "need_update_rms": True,
            "divide_by_scalxc_for_update": True,
            "enforce_edge": True,
        },
        "args": captured["args"],
    }
    assert captured["args"][:6] == ("state", 0.2, 0.3, 0.4, 0.5, -1.0)
    for got, expected in zip(captured["args"][6:-1], expected_channels):
        np.testing.assert_allclose(got, expected)
    assert captured["args"][-1] == pytest.approx(0.6)
    assert result.state == "updated-state"
    assert result.update_deltas is None
    assert result.update_rms is None
    assert float(np.asarray(result.update_rms_j)) == pytest.approx(0.125)
    np.testing.assert_allclose(result.velocities.lss, [12.0])


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


def test_velocity_block_bundle_zero_helpers_preserve_restart_semantics() -> None:
    blocks = _blocks(offset=1.0)

    zero_all = zero_all_velocity_blocks_like(blocks)
    for got, reference in zip(zero_all, blocks, strict=True):
        np.testing.assert_allclose(got, np.zeros_like(reference))

    zero_primary = zero_primary_velocity_blocks_like(blocks)
    for name in ("rcc", "rss", "zsc", "zcs", "lsc", "lcs"):
        np.testing.assert_allclose(getattr(zero_primary, name), 0.0)
    for name in ("rsc", "rcs", "zcc", "zss", "lcc", "lss"):
        np.testing.assert_allclose(getattr(zero_primary, name), getattr(blocks, name))


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


def test_host_pre_restart_trigger_branch_result_packages_side_effects() -> None:
    update = host_pre_restart_trigger_update(
        pre_restart_reason="bad_progress",
        huge_initial_forces=False,
        huge_force_restart_count=0,
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

    got = host_pre_restart_trigger_branch_result(
        state_checkpoint="checkpoint-state",
        pre_restart_update=update,
        pre_restart_reason="bad_progress",
        prev_rz_fsq_before=0.375,
        vmec2000_control=True,
    )

    assert got.state == "checkpoint-state"
    assert got.update is update
    assert got.step_status == "restart_bad_progress"
    assert got.restart_path == "pre_restart_trigger"
    assert got.restart_reason == "bad_progress"
    assert got.pre_restart_reason == "bad_progress"
    assert got.time_step_iter == pytest.approx(update.time_step_iter)
    assert got.prev_rz_fsq == pytest.approx(0.375)
    assert got.clear_freeb_controls is True
    assert got.clear_preconditioner_cache is True
    assert got.force_bcovar_update is True
    assert got.pop_iteration_history is True
    assert got.skip_time_control is True

    no_vmec2000 = host_pre_restart_trigger_branch_result(
        state_checkpoint="checkpoint-state",
        pre_restart_update=update,
        pre_restart_reason="bad_progress",
        prev_rz_fsq_before=0.375,
        vmec2000_control=False,
    )
    assert no_vmec2000.force_bcovar_update is False
