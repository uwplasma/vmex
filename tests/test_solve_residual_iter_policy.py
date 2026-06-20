from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax.solvers.fixed_boundary.options import validate_residual_iteration_options
from vmec_jax.solvers.fixed_boundary.residual.policy import (
    append_preconditioned_residual_history,
    append_zero_update_history_record,
    host_restart_decision,
    host_update_assembly_policy,
    numpy_preconditioner_apply_policy,
    resolve_light_history,
    resolve_residual_iter_startup_policy,
    resolve_restart_flags,
    scan_fallback_decision,
    scan_fallback_message,
    stage_transition_restart_reason,
)


def test_append_preconditioned_residual_history_keeps_channels_aligned():
    histories = {name: [] for name in "rz f gcr gcz gcl fsq fsqr fsqz fsql".split()}

    appended = append_preconditioned_residual_history(
        track_history=True,
        rz_norm=1.0,
        f_norm1=2.0,
        gcr2_p=3.0,
        gcz2_p=4.0,
        gcl2_p=5.0,
        fsq1=6.0,
        fsqr1_safe=7.0,
        fsqz1_safe=8.0,
        fsql1_safe=9.0,
        rz_norm_history=histories["rz"],
        f_norm1_history=histories["f"],
        gcr2_p_history=histories["gcr"],
        gcz2_p_history=histories["gcz"],
        gcl2_p_history=histories["gcl"],
        fsq1_history=histories["fsq"],
        fsqr1_history=histories["fsqr"],
        fsqz1_history=histories["fsqz"],
        fsql1_history=histories["fsql"],
    )

    assert appended is True
    assert histories == {
        "rz": [1.0],
        "f": [2.0],
        "gcr": [3.0],
        "gcz": [4.0],
        "gcl": [5.0],
        "fsq": [6.0],
        "fsqr": [7.0],
        "fsqz": [8.0],
        "fsql": [9.0],
    }
    assert not append_preconditioned_residual_history(
        track_history=False,
        rz_norm=0.0,
        f_norm1=0.0,
        gcr2_p=0.0,
        gcz2_p=0.0,
        gcl2_p=0.0,
        fsq1=0.0,
        fsqr1_safe=0.0,
        fsqz1_safe=0.0,
        fsql1_safe=0.0,
        rz_norm_history=histories["rz"],
        f_norm1_history=histories["f"],
        gcr2_p_history=histories["gcr"],
        gcz2_p_history=histories["gcz"],
        gcl2_p_history=histories["gcl"],
        fsq1_history=histories["fsq"],
        fsqr1_history=histories["fsqr"],
        fsqz1_history=histories["fsqz"],
        fsql1_history=histories["fsql"],
    )
    assert histories["rz"] == [1.0]


def test_append_zero_update_history_record_builds_aligned_converged_row():
    record_lists = {
        "step_history": [],
        "dt_eff_history": [],
        "update_rms_history": [],
        "w_curr_history": [],
        "w_try_history": [],
        "w_try_ratio_history": [],
        "restart_path_history": [],
        "step_status_history": [],
        "restart_reason_history": [],
        "pre_restart_reason_history": [],
        "time_step_history": [],
        "res0_history": [],
        "res1_history": [],
        "fsq_prev_history": [],
        "bad_growth_streak_history": [],
        "iter1_history": [],
        "iter2_history": [],
        "grad_rms_history": [],
        "freeb_ivac_history": [],
        "freeb_ivacskip_history": [],
        "freeb_full_update_history": [],
        "free_boundary_enabled": True,
    }

    appended = append_zero_update_history_record(
        track_history=True,
        restart_path="converged",
        step_status="converged",
        restart_reason="none",
        pre_restart_reason="none",
        time_step_value=0.9,
        fsqr=1.0,
        fsqz=2.0,
        fsql=3.0,
        res0=4.0,
        res1=5.0,
        fsq_prev=6.0,
        bad_growth_streak=7,
        iter1=8,
        iter2=9,
        free_boundary_enabled=True,
        freeb_ivac=1,
        freeb_ivacskip=0,
        history_record_lists=record_lists,
    )

    assert appended is True
    assert record_lists["step_history"] == [0.0]
    assert record_lists["w_curr_history"] == [6.0]
    assert record_lists["step_status_history"] == ["converged"]
    assert record_lists["iter2_history"] == [9]
    assert record_lists["freeb_ivac_history"] == [1]


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


def _startup_policy_for_test(**overrides):
    kwargs = dict(
        max_iter=100,
        step_size=0.9,
        precompile_only=False,
        signgs=1,
        lambda_update_scale=1.0,
        enforce_vmec_lambda_axis=True,
        vmec2000_control=False,
        reference_mode=False,
        limit_dt_from_force=False,
        limit_update_rms=False,
        backtracking=False,
        strict_update=False,
        jit_precompile=False,
        use_scan=False,
        host_update_assembly=None,
        backend_name="cpu",
        scan_backend_name="cpu",
        state_has_tracer=False,
        env={},
        validate_options=validate_residual_iteration_options,
        resolve_tridi_policies=lambda *, use_precomputed, use_lax_tridi: (
            False if use_precomputed is None else bool(use_precomputed),
            False if use_lax_tridi is None else bool(use_lax_tridi),
        ),
        normalize_adjoint_trace_mode=lambda mode: str(mode).strip().lower(),
        normalize_resume_state_mode=lambda mode: "compact" if mode is None else str(mode).strip().lower(),
        resolve_scan_fallback_policy=lambda **_: SimpleNamespace(
            enabled=True,
            iters=50,
            badjac_limit=10,
            fsq_abs=1.0e-2,
            accept_frac=0.5,
            fsq_factor=50.0,
            improve=0.1,
        ),
        preconditioner_use_precomputed_tridi=None,
        preconditioner_use_lax_tridi=None,
        adjoint_trace=False,
        adjoint_trace_mode="full",
        fsq_total_target=None,
        light_history=None,
        resume_state_mode=None,
        use_restart_triggers=None,
        use_direct_fallback=None,
        vmecpp_restart=False,
        verbose_vmec2000_table=True,
        stage_prev_fsq=None,
        stage_transition_factor=50.0,
        stage_transition_scale=0.5,
        auto_flip_force=True,
        jit_forces=True,
    )
    kwargs.update(overrides)
    return resolve_residual_iter_startup_policy(**kwargs)


def test_residual_iter_startup_policy_resolves_env_driven_host_and_dump_controls():
    policy = _startup_policy_for_test(
        backend_name="gpu",
        host_update_assembly=True,
        env={
            "VMEC_JAX_HOST_UPDATE_ON_ACCELERATOR": "1",
            "VMEC_JAX_HOST_FSQ1_NORMS": "off",
            "VMEC_JAX_HOST_RESIDUAL_METRICS": "on",
            "VMEC_JAX_LIGHT_HISTORY": "1",
            "VMEC_JAX_DUMP_XC": "1",
        },
    )
    assert policy.host_update_assembly
    assert not policy.host_fsq1_norms_on_accelerator
    assert policy.host_residual_metrics_on_accelerator
    assert policy.dumps_enabled
    assert policy.dump_any
    assert policy.disabled_jit_for_dumps
    assert not policy.jit_forces
    assert not policy.light_history
    assert policy.track_history


def test_residual_iter_startup_policy_preserves_scan_branch_safety_rules():
    policy = _startup_policy_for_test(
        use_scan=True,
        vmec2000_control=True,
        state_has_tracer=True,
        env={"VMEC_JAX_VMEC2000_CHUNKED": "1"},
        stage_prev_fsq=2.0,
        stage_transition_factor=0.0,
        stage_transition_scale=0.5,
    )
    assert policy.use_scan
    assert policy.vmec2000_control
    assert not policy.auto_flip_force
    assert not policy.force_chunked_scan
    assert policy.differentiating_scan
    assert not policy.scan_fallback_enabled
    assert policy.stage_prev_fsq is None


def test_residual_iter_startup_policy_normalizes_objective_and_restart_defaults():
    policy = _startup_policy_for_test(
        fsq_total_target=-1.0,
        use_restart_triggers=None,
        use_direct_fallback=None,
        vmecpp_restart=True,
        adjoint_trace=True,
        adjoint_trace_mode=" COMPACT ",
        resume_state_mode=" FULL ",
        preconditioner_use_precomputed_tridi=True,
    )
    assert policy.fsq_total_target == 0.0
    assert policy.use_restart_triggers
    assert not policy.use_direct_fallback
    assert policy.vmecpp_restart
    assert policy.adjoint_trace
    assert policy.adjoint_trace_mode == "compact"
    assert policy.resume_state_mode == "full"
    assert policy.preconditioner_use_precomputed_tridi_policy


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
