from __future__ import annotations

import numpy as np

from vmec_jax.solvers.fixed_boundary.residual.policy import host_restart_decision, scan_fallback_decision


class RaisingDiagnostics:
    def get(self, key, default=None):
        raise RuntimeError(key)


class BadArray:
    def __array__(self, dtype=None):
        raise TypeError(dtype)


class ProbeRaisingDiagnostics(dict):
    def get(self, key, default=None):
        if key == "probe_count":
            raise RuntimeError("probe metadata unavailable")
        return super().get(key, default)


def test_scan_fallback_decision_handles_diagnostics_get_failures():
    decision = scan_fallback_decision(
        diagnostics=RaisingDiagnostics(),
        fsqr_history=np.asarray([1.0]),
        fsqz_history=np.asarray([0.0]),
        fsql_history=np.asarray([0.0]),
        max_iter=1,
        fallback_iters=1,
        badjac_limit=0,
        fsq_abs=1.0e-4,
        accept_frac=0.5,
        fsq_factor=10.0,
    )

    assert not decision.fallback
    assert decision.bad_jac_count == 0
    assert decision.accepted_frac is None
    assert decision.fsq_min_full is None
    assert decision.fsq_max_full is None
    assert not decision.fsq_all_finite


def test_scan_fallback_decision_handles_bad_array_casts():
    decision = scan_fallback_decision(
        diagnostics={
            "bad_jacobian_full": BadArray(),
            "accepted_mask": BadArray(),
            "fsqr_full": np.asarray([1.0, 20.0]),
            "fsqz_full": np.zeros(2),
            "fsql_full": np.zeros(2),
        },
        fsqr_history=np.asarray([99.0]),
        fsqz_history=np.asarray([0.0]),
        fsql_history=np.asarray([0.0]),
        max_iter=2,
        fallback_iters=2,
        badjac_limit=0,
        fsq_abs=1.0e-4,
        accept_frac=0.5,
        fsq_factor=10.0,
    )

    assert not decision.fallback
    assert decision.bad_jac_count == 0
    assert decision.accepted_frac is None
    assert decision.fsq_min_full == 1.0
    assert decision.fsq_max_full == 20.0
    assert decision.fsq_all_finite


def test_scan_fallback_decision_suppresses_probe_message_when_probe_metadata_fails():
    decision = scan_fallback_decision(
        diagnostics=ProbeRaisingDiagnostics(
            abort_scan=True,
            fsqr_full=np.asarray([1.0, 20.0]),
            fsqz_full=np.zeros(2),
            fsql_full=np.zeros(2),
        ),
        fsqr_history=np.asarray([99.0]),
        fsqz_history=np.asarray([0.0]),
        fsql_history=np.asarray([0.0]),
        max_iter=2,
        fallback_iters=2,
        badjac_limit=10,
        fsq_abs=1.0e-4,
        accept_frac=0.5,
        fsq_factor=10.0,
    )

    assert decision.fallback
    assert decision.reasons == ("abort_scan",)
    assert decision.probe_message == ""


def test_host_restart_reference_mode_flags_residual_growth_as_bad_jacobian():
    decision = host_restart_decision(
        iter2=4,
        iter1=1,
        fsqr=200.0,
        fsqz=0.0,
        fsql=0.0,
        fsq1=0.5,
        fsq_prev=1.0,
        res0=1.0,
        bad_growth_streak=2,
        pre_restart_reason="none",
        reference_mode=True,
        vmec2000_control=False,
        bad_jacobian=False,
        stage_prev_fsq=None,
        stage_transition_factor=50.0,
        lmove_axis=False,
        vmecpp_restart=False,
        k_preconditioner_update_interval=4,
    )

    assert decision.pre_restart_reason == "bad_jacobian"
    assert decision.bad_growth_streak == 3


def test_host_restart_reference_mode_flags_slow_progress_after_update_window():
    decision = host_restart_decision(
        iter2=20,
        iter1=1,
        fsqr=0.02,
        fsqz=0.0,
        fsql=0.0,
        fsq1=0.5,
        fsq_prev=1.0,
        res0=1.0,
        bad_growth_streak=2,
        pre_restart_reason="none",
        reference_mode=True,
        vmec2000_control=False,
        bad_jacobian=False,
        stage_prev_fsq=None,
        stage_transition_factor=50.0,
        lmove_axis=False,
        vmecpp_restart=False,
        k_preconditioner_update_interval=4,
    )

    assert decision.pre_restart_reason == "bad_progress"
    assert decision.bad_growth_streak == 0


def test_host_restart_vmec2000_control_uses_vmecpp_bad_progress_reason():
    decision = host_restart_decision(
        iter2=20,
        iter1=1,
        fsqr=0.02,
        fsqz=0.0,
        fsql=0.0,
        fsq1=1.0,
        fsq_prev=2.0,
        res0=5.0,
        bad_growth_streak=0,
        pre_restart_reason="none",
        reference_mode=False,
        vmec2000_control=True,
        bad_jacobian=False,
        stage_prev_fsq=None,
        stage_transition_factor=50.0,
        lmove_axis=False,
        vmecpp_restart=True,
        k_preconditioner_update_interval=4,
    )

    assert decision.pre_restart_reason == "bad_progress_vmecpp"
    assert decision.vmecpp_bad_progress
