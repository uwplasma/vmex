from __future__ import annotations

from collections import OrderedDict
import os
from types import SimpleNamespace

import numpy as np
import pytest

import vmec_jax.optimization as opt_module
from vmec_jax.optimization import FixedBoundaryExactOptimizer
from tools.diagnostics import profile_exact_optimizer as exact_profile_tool


def _bare_optimizer() -> FixedBoundaryExactOptimizer:
    opt = FixedBoundaryExactOptimizer.__new__(FixedBoundaryExactOptimizer)
    opt._profile = {}
    opt._trial_residual_cache = OrderedDict()
    opt._trial_residual_cache_max = 2
    opt._callback_trace_enabled = True
    opt._callback_trace = []
    opt._callback_point_ids = {}
    opt._callback_previous_key = None
    return opt


def test_trial_residual_cache_is_small_lru() -> None:
    opt = _bare_optimizer()

    p0 = np.array([0.0, 1.0])
    p1 = np.array([1.0, 2.0])
    p2 = np.array([2.0, 3.0])

    opt._remember_trial_residual(p0, np.array([1.0, 2.0]))
    opt._remember_trial_residual(p1, np.array([3.0, 4.0]))
    np.testing.assert_allclose(opt._cached_trial_residual(p0), [1.0, 2.0])

    opt._remember_trial_residual(p2, np.array([5.0, 6.0]))

    assert opt._cached_trial_residual(p1) is None
    np.testing.assert_allclose(opt._cached_trial_residual(p0), [1.0, 2.0])
    np.testing.assert_allclose(opt._cached_trial_residual(p2), [5.0, 6.0])
    assert opt._profile["trial_residual_cache_hit"]["count"] == 3


def test_callback_trace_records_repeat_points_and_summary() -> None:
    opt = _bare_optimizer()

    p0 = np.array([0.0, 1.0])
    p1 = np.array([0.5, 1.0])

    opt._trace_callback_event("residual", p0, source="exact_state_cache", wall_time_s=0.1)
    opt._trace_callback_event("jacobian", p0, source="exact_tape_replay", wall_time_s=0.2)
    opt._trace_callback_event("residual", p1, source="trial_solve", wall_time_s=0.3)

    dump = opt._callback_trace_dump()

    assert dump["enabled"] is True
    assert [event["point_id"] for event in dump["events"]] == [0, 0, 1]
    assert [event["same_as_previous"] for event in dump["events"]] == [False, True, False]
    assert dump["summary"]["residual:exact_state_cache"]["count"] == 1
    assert dump["summary"]["jacobian:exact_tape_replay"]["wall_time_s"] == 0.2


def test_run_does_not_force_dynamic_replay_bucket(monkeypatch) -> None:
    monkeypatch.delenv("VMEC_JAX_DYNAMIC_REPLAY_BUCKET", raising=False)
    monkeypatch.setattr(
        "vmec_jax.wout.equilibrium_aspect_ratio_from_state",
        lambda *, state, static: 7.0,
    )

    def fake_gauss_newton(residual_fun, jacobian_fun, params0, **kwargs):
        del residual_fun, jacobian_fun, kwargs
        return {
            "x": np.asarray(params0, dtype=float),
            "cost": 0.5,
            "objective": 1.0,
            "nfev": 1,
            "njev": 1,
            "nit": 0,
            "success": True,
            "status": 1,
            "message": "stub",
            "step_norm": 0.0,
            "x_prev": None,
            "cost_prev": None,
        }

    monkeypatch.setattr(opt_module, "gauss_newton_least_squares", fake_gauss_newton)

    opt = FixedBoundaryExactOptimizer.__new__(FixedBoundaryExactOptimizer)
    opt._scan_exact_path = "tape"
    opt._static = object()
    opt._profile = {}
    opt._trial_residual_cache = OrderedDict()
    opt._exact_cache = {}
    opt._exact_state_cache = {}
    opt._initial_tangent_cache = {}
    opt._solver_device_name = "cpu"
    opt._inner_max_iter = 0
    opt._inner_ftol = 0.0
    opt._trial_max_iter = 0
    opt._trial_ftol = 0.0
    opt._callback_trace_enabled = False
    opt._callback_trace = []
    opt._callback_point_ids = {}
    opt._callback_previous_key = None
    opt.residual_fun = lambda params: np.asarray([1.0])
    opt.forward_residual_fun = lambda params: np.asarray([1.0])
    opt._jacobian_fun_tracked = lambda params: np.asarray([[1.0]])
    opt._solve_exact_with_tape = (
        lambda params, return_payload=False: ("state", {"tape": object()})
        if return_payload
        else "state"
    )
    opt._cached_exact_state = lambda params: "state"
    opt._residuals_fn = lambda state: np.asarray([1.0])
    opt._qs_total_from_state = lambda state, residuals: float(np.dot(residuals, residuals))

    result = opt.run(np.asarray([0.0]), method="gauss_newton", max_nfev=1, verbose=0)

    assert result["_history_dump"]["success"] is True
    assert "VMEC_JAX_DYNAMIC_REPLAY_BUCKET" not in os.environ


def test_exact_optimizer_profile_parser_accepts_cache_budget_args() -> None:
    args = exact_profile_tool._parse_args(
        [
            "--callback",
            "accepted",
            "--repeats",
            "3",
            "--perturb-scale",
            "1e-4",
            "--budget-total-wall-s",
            "30",
            "--budget-repeat-wall-s",
            "12",
            "--budget-rss-growth-mb",
            "256",
            "--budget-cache-entries",
            "40",
            "--budget-cache-entry-growth",
            "8",
            "--budget-tape-build-wall-s",
            "5",
            "--budget-replay-wall-s",
            "4",
            "--budget-residual-tangent-wall-s",
            "3",
            "--budget-accepted-replays",
            "2",
            "--budget-action",
            "warn",
            "--vmec-timing-detail",
            "--initial-metrics",
        ]
    )

    assert args.callback == "accepted"
    assert args.repeats == 3
    assert args.perturb_scale == 1.0e-4
    assert args.budget_total_wall_s == 30.0
    assert args.budget_repeat_wall_s == 12.0
    assert args.budget_rss_growth_mb == 256.0
    assert args.budget_cache_entries == 40
    assert args.budget_cache_entry_growth == 8
    assert args.budget_tape_build_wall_s == 5.0
    assert args.budget_replay_wall_s == 4.0
    assert args.budget_residual_tangent_wall_s == 3.0
    assert args.budget_accepted_replays == 2
    assert args.budget_action == "warn"
    assert args.vmec_timing_detail is True
    assert args.initial_metrics is True


def test_exact_optimizer_profile_skips_initial_metrics_by_default() -> None:
    args = exact_profile_tool._parse_args(["--callback", "jacobian"])

    assert args.initial_metrics is False


def test_exact_optimizer_profile_gradient_alias_preserves_check_gradient() -> None:
    gradient_only = exact_profile_tool._normalize_callback_args(
        exact_profile_tool._parse_args(["--gradient-only"])
    )
    assert gradient_only.callback == "gradient"

    checked = exact_profile_tool._normalize_callback_args(
        exact_profile_tool._parse_args(["--gradient-only", "--check-gradient"])
    )
    assert checked.callback == "run"


def test_exact_optimizer_profile_cache_snapshot_and_delta_schema() -> None:
    opt = FixedBoundaryExactOptimizer.__new__(FixedBoundaryExactOptimizer)
    opt._exact_cache = {"a": object()}
    opt._exact_state_cache = {}
    opt._exact_residual_cache = {"a": np.asarray([1.0])}
    opt._exact_jacobian_cache = {"a": np.asarray([[1.0]])}
    opt._trial_residual_cache = OrderedDict([("b", np.asarray([2.0]))])
    opt._initial_tangent_cache = {}
    opt._discrete_jacobian_helper_cache = {"j": object()}
    opt._scan_exact_helper_cache = {}

    snapshot = exact_profile_tool._cache_snapshot(opt, include_global=False)
    assert snapshot["optimizer"]["exact_cache"] == 1
    assert snapshot["optimizer"]["exact_jacobian_cache"] == 1
    assert snapshot["optimizer"]["trial_residual_cache"] == 1
    assert snapshot["total_entries"] == 5

    delta = exact_profile_tool._profile_delta(
        {"exact_tape_build": {"count": 1, "wall_time_s": 2.0}},
        {
            "exact_tape_build": {"count": 3, "wall_time_s": 7.0},
            "jacobian_tape_replay": {"count": 1, "wall_time_s": 4.0},
        },
    )
    assert delta["exact_tape_build"]["count"] == 2
    assert delta["exact_tape_build"]["wall_time_s"] == 5.0
    assert delta["jacobian_tape_replay"]["mean_wall_time_s"] == 4.0


def test_exact_optimizer_profile_timing_includes_preconditioner_subphases() -> None:
    opt = FixedBoundaryExactOptimizer.__new__(FixedBoundaryExactOptimizer)
    opt._profile = {}
    tape = SimpleNamespace(
        diagnostics={
            "timing": {
                "compute_forces_s": 0.10,
                "preconditioner_s": 0.40,
                "precond_refresh_s": 0.05,
                "precond_apply_s": 0.25,
                "precond_mode_scale_s": 0.10,
                "update_s": 0.20,
            }
        }
    )

    opt._profile_exact_tape_solver_timing(tape, tape_build_wall_s=1.0)
    profile = opt._profile_dump()

    assert profile["exact_tape_solver_preconditioner"]["wall_time_s"] == 0.40
    assert profile["exact_tape_solver_preconditioner_apply"]["wall_time_s"] == 0.25
    assert profile["exact_tape_solver_preconditioner_mode_scale"]["wall_time_s"] == 0.10
    assert profile["exact_tape_solver_precond_refresh"]["wall_time_s"] == 0.05
    assert profile["exact_tape_build_unattributed"]["wall_time_s"] == pytest.approx(0.30)


def test_exact_optimizer_profiles_trial_solver_timing_buckets() -> None:
    opt = FixedBoundaryExactOptimizer.__new__(FixedBoundaryExactOptimizer)
    opt._profile = {}

    opt._profile_solver_timing(
        {
            "timing": {
                "compute_forces_s": 0.20,
                "preconditioner_s": 0.30,
                "update_s": 0.10,
                "update_state_s": 0.04,
            }
        },
        profile_prefix="trial_solver",
        phase_wall_s=0.75,
        unattributed_name="solve_forward_trial_unattributed",
    )
    profile = opt._profile_dump()

    assert profile["trial_solver_compute_forces"]["wall_time_s"] == 0.20
    assert profile["trial_solver_preconditioner"]["wall_time_s"] == 0.30
    assert profile["trial_solver_update"]["wall_time_s"] == 0.10
    assert profile["trial_solver_update_state"]["wall_time_s"] == 0.04
    assert profile["solve_forward_trial_unattributed"]["wall_time_s"] == pytest.approx(0.15)


def test_exact_optimizer_profiles_scan_solver_timing_buckets() -> None:
    opt = FixedBoundaryExactOptimizer.__new__(FixedBoundaryExactOptimizer)
    opt._profile = {}

    opt._profile_solver_timing(
        {
            "timing": {
                "iterations": 4,
                "scan_total_s": 0.55,
                "scan_preflight_s": 0.05,
                "scan_device_run_s": 0.40,
                "scan_host_materialize_s": 0.03,
                "scan_postprocess_s": 0.07,
            }
        },
        profile_prefix="trial_solver",
        phase_wall_s=0.65,
        unattributed_name="solve_forward_trial_unattributed",
    )
    profile = opt._profile_dump()

    assert profile["trial_solver_scan_total"]["wall_time_s"] == 0.55
    assert profile["trial_solver_scan_preflight"]["wall_time_s"] == 0.05
    assert profile["trial_solver_scan_device_run"]["wall_time_s"] == 0.40
    assert profile["trial_solver_scan_host_materialize"]["wall_time_s"] == 0.03
    assert profile["trial_solver_scan_postprocess"]["wall_time_s"] == 0.07
    assert profile["solve_forward_trial_unattributed"]["wall_time_s"] == pytest.approx(0.10)


def test_exact_optimizer_callback_report_schema_and_budget_status() -> None:
    args = exact_profile_tool._parse_args(
        [
            "--problem",
            "qh",
            "--max-mode",
            "2",
            "--callback",
            "accepted",
            "--perturb-scale",
            "1e-4",
            "--budget-total-wall-s",
            "1.0",
            "--budget-repeat-wall-s",
            "0.5",
            "--budget-rss-growth-mb",
            "1.0",
            "--budget-cache-entry-growth",
            "0",
            "--budget-tape-build-wall-s",
            "0.3",
            "--budget-replay-wall-s",
            "0.2",
            "--budget-residual-tangent-wall-s",
            "0.1",
            "--budget-accepted-replays",
            "0",
        ]
    )
    cache_before = {"optimizer": {"exact_cache": 0}, "total_entries": 0}
    cache_after = {"optimizer": {"exact_cache": 2}, "total_entries": 2}

    report = exact_profile_tool._build_callback_payload(
        args=args,
        specs_count=24,
        solver_device_resolved="cpu",
        samples=[
            {
                "repeat": 0,
                "wall_time_s": 0.75,
                "metric_norm": 1.0,
                "param_step_norm": 1.0e-4,
                "shape": [3],
                "profile_delta": {},
                "cache_growth": {"total_entries_delta": 2},
            }
        ],
        profile={
            "exact_tape_build": {"count": 1, "wall_time_s": 0.4, "mean_wall_time_s": 0.4},
            "jacobian_tape_replay": {"count": 1, "wall_time_s": 0.25, "mean_wall_time_s": 0.25},
            "jacobian_residual_tangents": {"count": 1, "wall_time_s": 0.15, "mean_wall_time_s": 0.15},
        },
        cache_before=cache_before,
        cache_after=cache_after,
        rss_before_bytes=100 * 1024 * 1024,
        rss_after_bytes=103 * 1024 * 1024,
        total_wall_s=1.25,
        runtime={"default_backend": "cpu"},
    )

    assert report["schema_version"] == 2
    assert report["report_kind"] == "exact_optimizer_callback_profile"
    assert report["callback"] == "exact"
    assert report["cache"]["growth"]["total_entries_delta"] == 2
    assert report["budget_status"]["ok"] is False
    assert {
        item["name"] for item in report["budget_status"]["exceeded"]
    } == {
        "total_wall_s",
        "repeat_wall_s",
        "rss_growth_mb",
        "cache_entry_growth",
        "tape_build_wall_s",
        "replay_wall_s",
        "residual_tangent_wall_s",
        "accepted_replays",
    }
    assert report["budget_status"]["measurements"]["accepted_replays"] == 1
