from __future__ import annotations

import numpy as np
import pytest

from vmec_jax.solvers.fixed_boundary.residual.policy import (
    host_restart_decision,
    host_update_assembly_policy,
    numpy_preconditioner_apply_policy,
    resolve_light_history,
    resolve_restart_flags,
    scan_fallback_decision,
    scan_fallback_message,
    stage_transition_restart_reason,
)


def test_host_update_assembly_policy_matches_cpu_non_scan_defaults():
    auto = host_update_assembly_policy(
        requested=None,
        use_scan=False,
        backend_name="cpu",
        state_has_tracer=False,
    )
    assert auto.enabled
    assert auto.auto_enabled

    explicit_with_tracer = host_update_assembly_policy(
        requested=True,
        use_scan=False,
        backend_name="cpu",
        state_has_tracer=True,
    )
    assert explicit_with_tracer.enabled
    assert not explicit_with_tracer.auto_enabled

    scan_disabled = host_update_assembly_policy(
        requested=True,
        use_scan=True,
        backend_name="cpu",
        state_has_tracer=False,
    )
    assert not scan_disabled.enabled

    accelerator_disabled = host_update_assembly_policy(
        requested=True,
        use_scan=False,
        backend_name="gpu",
        state_has_tracer=False,
    )
    assert not accelerator_disabled.enabled

    accelerator_explicit = host_update_assembly_policy(
        requested=True,
        use_scan=False,
        backend_name="gpu",
        state_has_tracer=False,
        allow_accelerator=True,
    )
    assert accelerator_explicit.enabled
    assert not accelerator_explicit.auto_enabled


def test_numpy_preconditioner_apply_policy_uses_short_or_spectral_cpu_host_path():
    short_small = numpy_preconditioner_apply_policy(
        host_update_assembly=True,
        max_iter=120,
        mpol=2,
        ntor=2,
        max_iter_env="240",
        min_mode_count_env="16",
    )
    assert short_small.enabled
    assert short_small.mode_count == 6

    long_small = numpy_preconditioner_apply_policy(
        host_update_assembly=True,
        max_iter=1500,
        mpol=2,
        ntor=2,
        max_iter_env="240",
        min_mode_count_env="16",
    )
    assert not long_small.enabled

    long_spectral = numpy_preconditioner_apply_policy(
        host_update_assembly=True,
        max_iter=3000,
        mpol=5,
        ntor=5,
        max_iter_env="240",
        min_mode_count_env="16",
    )
    assert long_spectral.enabled
    assert long_spectral.mode_count == 30

    disabled_without_host_update = numpy_preconditioner_apply_policy(
        host_update_assembly=False,
        max_iter=120,
        mpol=5,
        ntor=5,
        max_iter_env="240",
        min_mode_count_env="16",
    )
    assert not disabled_without_host_update.enabled

    disabled_by_env = numpy_preconditioner_apply_policy(
        host_update_assembly=True,
        max_iter=3000,
        mpol=5,
        ntor=5,
        max_iter_env="0",
        min_mode_count_env="0",
    )
    assert not disabled_by_env.enabled

    parsed_defaults = numpy_preconditioner_apply_policy(
        host_update_assembly=True,
        max_iter=3000,
        mpol=5,
        ntor=5,
        max_iter_env="bad",
        min_mode_count_env="bad",
    )
    assert parsed_defaults.enabled
    assert parsed_defaults.max_iter_cutoff == 240
    assert parsed_defaults.min_mode_count == 16

    invalid_mode_shape = numpy_preconditioner_apply_policy(
        host_update_assembly=True,
        max_iter=3000,
        mpol=object(),
        ntor=object(),
        max_iter_env="240",
        min_mode_count_env="16",
    )
    assert not invalid_mode_shape.enabled
    assert invalid_mode_shape.mode_count == 0


@pytest.mark.parametrize(
    ("value", "env_value", "expected"),
    [
        (None, "1", True),
        (None, " TRUE ", True),
        (None, "false", False),
        (None, "", False),
        (True, "0", True),
        (False, "1", False),
    ],
)
def test_resolve_light_history_preserves_env_and_explicit_precedence(value, env_value, expected):
    assert resolve_light_history(value, env_value=env_value) is expected


def test_resolve_restart_flags_applies_legacy_defaults():
    defaults = resolve_restart_flags(
        use_restart_triggers=None,
        use_direct_fallback=None,
        vmecpp_restart=1,
    )
    assert defaults.use_restart_triggers
    assert not defaults.use_direct_fallback
    assert defaults.vmecpp_restart

    explicit = resolve_restart_flags(
        use_restart_triggers=False,
        use_direct_fallback=True,
        vmecpp_restart=False,
    )
    assert not explicit.use_restart_triggers
    assert explicit.use_direct_fallback
    assert not explicit.vmecpp_restart


def test_stage_transition_policy_only_restarts_valid_first_step_growth():
    assert (
        stage_transition_restart_reason(
            iter2=1,
            fsq=60.0,
            pre_restart_reason="none",
            stage_prev_fsq=1.0,
            stage_transition_factor=50.0,
        )
        == "stage_transition"
    )
    assert (
        stage_transition_restart_reason(
            iter2=2,
            fsq=60.0,
            pre_restart_reason="none",
            stage_prev_fsq=1.0,
            stage_transition_factor=50.0,
        )
        == "none"
    )
    assert (
        stage_transition_restart_reason(
            iter2=1,
            fsq=60.0,
            pre_restart_reason="bad_jacobian",
            stage_prev_fsq=1.0,
            stage_transition_factor=50.0,
        )
        == "bad_jacobian"
    )
    assert (
        stage_transition_restart_reason(
            iter2=1,
            fsq=60.0,
            pre_restart_reason="none",
            stage_prev_fsq=object(),
            stage_transition_factor=50.0,
        )
        == "none"
    )


def test_host_restart_decision_uses_stage_transition_policy():
    decision = host_restart_decision(
        iter2=1,
        iter1=1,
        fsqr=20.0,
        fsqz=20.0,
        fsql=20.0,
        fsq1=1.0,
        fsq_prev=1.0,
        res0=1.0,
        bad_growth_streak=0,
        pre_restart_reason="none",
        reference_mode=False,
        vmec2000_control=False,
        bad_jacobian=False,
        stage_prev_fsq=1.0,
        stage_transition_factor=50.0,
        lmove_axis=True,
        vmecpp_restart=False,
        k_preconditioner_update_interval=25,
    )
    assert decision.fsq == pytest.approx(60.0)
    assert decision.pre_restart_reason == "stage_transition"
    assert not decision.huge_initial_forces


def test_scan_fallback_decision_uses_full_diagnostics_and_probe_message():
    decision = scan_fallback_decision(
        diagnostics={
            "abort_scan": True,
            "bad_jacobian_full": np.asarray([1, 1, 1, 0]),
            "accepted_mask": np.asarray([1, 0, 0, 0]),
            "fsqr_full": np.asarray([0.1, 10.0, 20.0, 30.0]),
            "fsqz_full": np.zeros(4),
            "fsql_full": np.zeros(4),
            "probe_count": 4,
            "probe_accept_frac": 0.25,
            "probe_ratio": 300.0,
            "probe_fsq_min": 0.1,
        },
        fsqr_history=np.asarray([1.0]),
        fsqz_history=np.asarray([1.0]),
        fsql_history=np.asarray([1.0]),
        max_iter=4,
        fallback_iters=4,
        badjac_limit=2,
        fsq_abs=1.0e-2,
        accept_frac=0.5,
        fsq_factor=50.0,
    )

    assert decision.fallback
    assert decision.reasons == (
        "abort_scan",
        "bad_jac_count=3 > 2",
        "accepted_frac=0.25 < 0.50",
    )
    assert decision.reason_text == "abort_scan, bad_jac_count=3 > 2, accepted_frac=0.25 < 0.50"
    assert decision.probe_message == " (probe_count=4 probe_accept_frac=0.25 probe_ratio=300.00 probe_fsq_min=1.000e-01)"
    assert decision.bad_jac_count == 3
    assert decision.accepted_frac == pytest.approx(0.25)
    assert decision.fsq_min_full == pytest.approx(0.1)
    assert decision.fsq_max_full == pytest.approx(30.0)
    assert decision.fsq_all_finite


def test_scan_fallback_message_preserves_reason_and_probe_payload():
    decision = scan_fallback_decision(
        diagnostics={
            "abort_scan": True,
            "fsqr_full": np.asarray([0.25, 50.0]),
            "fsqz_full": np.zeros(2),
            "fsql_full": np.zeros(2),
            "probe_count": 2,
            "probe_accept_frac": 0.0,
            "probe_ratio": 200.0,
            "probe_fsq_min": 0.25,
        },
        fsqr_history=np.asarray([1.0]),
        fsqz_history=np.asarray([0.0]),
        fsql_history=np.asarray([0.0]),
        max_iter=2,
        fallback_iters=2,
        badjac_limit=10,
        fsq_abs=1.0e-2,
        accept_frac=0.5,
        fsq_factor=50.0,
    )

    assert scan_fallback_message(decision) == (
        "[solve_fixed_boundary_residual_iter] "
        "scan fallback -> non-scan (abort_scan) "
        "(probe_count=2 probe_accept_frac=0.00 probe_ratio=200.00 probe_fsq_min=2.500e-01)"
    )


def test_scan_fallback_decision_suppresses_weak_failure_signals():
    low_residual = scan_fallback_decision(
        diagnostics={
            "bad_jacobian_full": np.asarray([1, 1, 1]),
            "accepted_mask": np.asarray([0, 0, 0]),
            "fsqr_full": np.asarray([1.0e-4, 2.0e-4, 3.0e-4]),
            "fsqz_full": np.zeros(3),
            "fsql_full": np.zeros(3),
        },
        fsqr_history=np.asarray([1.0]),
        fsqz_history=np.asarray([1.0]),
        fsql_history=np.asarray([1.0]),
        max_iter=3,
        fallback_iters=3,
        badjac_limit=1,
        fsq_abs=1.0e-2,
        accept_frac=0.5,
        fsq_factor=50.0,
    )
    assert not low_residual.fallback
    assert low_residual.reasons == ()

    flat_residual = scan_fallback_decision(
        diagnostics={
            "bad_jacobian_full": np.asarray([1, 1, 1]),
            "accepted_mask": np.asarray([0, 0, 0]),
            "fsqr_full": np.asarray([1.0, 2.0, 3.0]),
            "fsqz_full": np.zeros(3),
            "fsql_full": np.zeros(3),
        },
        fsqr_history=np.asarray([1.0]),
        fsqz_history=np.asarray([1.0]),
        fsql_history=np.asarray([1.0]),
        max_iter=3,
        fallback_iters=3,
        badjac_limit=1,
        fsq_abs=1.0e-2,
        accept_frac=0.5,
        fsq_factor=50.0,
    )
    assert not flat_residual.fallback
    assert flat_residual.reasons == ()


def test_scan_fallback_decision_falls_back_to_public_histories_when_full_arrays_empty():
    decision = scan_fallback_decision(
        diagnostics={
            "accepted_mask": np.asarray([0, 0, 1]),
            "fsqr_full": np.asarray([]),
            "fsqz_full": np.asarray([]),
            "fsql_full": np.asarray([]),
        },
        fsqr_history=np.asarray([0.1, 20.0, 40.0]),
        fsqz_history=np.zeros(3),
        fsql_history=np.zeros(3),
        max_iter=3,
        fallback_iters=3,
        badjac_limit=10,
        fsq_abs=1.0e-2,
        accept_frac=0.8,
        fsq_factor=50.0,
    )

    assert decision.fallback
    assert decision.reasons == ("accepted_frac=0.33 < 0.80",)
    assert decision.fsq_min_full == pytest.approx(0.1)
    assert decision.fsq_max_full == pytest.approx(40.0)
