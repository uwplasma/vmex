from __future__ import annotations

from collections import OrderedDict
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import vmec_jax.optimization as opt_module
from vmec_jax.optimization import FixedBoundaryExactOptimizer, gauss_newton_least_squares
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


def test_gauss_newton_skips_jacobian_for_exact_zero_residual() -> None:
    jacobian_calls = []

    def residual(_x):
        return np.asarray([0.0, 0.0], dtype=float)

    def jacobian(_x):
        jacobian_calls.append(True)
        return np.eye(2)

    result = gauss_newton_least_squares(
        residual,
        jacobian,
        np.asarray([1.0, 2.0], dtype=float),
        max_nfev=3,
        verbose=0,
    )

    assert result["success"] is True
    assert result["nfev"] == 1
    assert result["njev"] == 0
    assert jacobian_calls == []


def test_gauss_newton_can_skip_exhausted_budget_jacobian(monkeypatch) -> None:
    monkeypatch.setenv("VMEC_JAX_OPT_SKIP_EXHAUSTED_GN_JACOBIAN", "1")

    def residual(_x):
        return np.asarray([1.0], dtype=float)

    def jacobian(_x):
        raise AssertionError("Jacobian replay should not run after residual budget is exhausted")

    result = gauss_newton_least_squares(
        residual,
        jacobian,
        np.asarray([0.0], dtype=float),
        max_nfev=1,
        verbose=0,
    )

    assert result["success"] is False
    assert result["message"] == "maximum function evaluations exceeded"
    assert result["nfev"] == 1
    assert result["njev"] == 0
    assert result["cost"] == pytest.approx(0.5)


def test_run_final_output_reuses_best_exact_residual_when_jacobian_is_skipped(monkeypatch) -> None:
    monkeypatch.setenv("VMEC_JAX_OPT_SKIP_EXHAUSTED_GN_JACOBIAN", "1")
    residual = np.asarray([0.5, 3.0, 4.0], dtype=float)
    state = object()

    opt = FixedBoundaryExactOptimizer.__new__(FixedBoundaryExactOptimizer)
    opt._history = []
    opt._profile = {}
    opt._trial_residual_cache = OrderedDict()
    opt._exact_cache = {b"accepted": (state, {})}
    opt._exact_state_cache = {b"accepted": state}
    opt._exact_state_key_by_id = {id(state): b"accepted"}
    opt._exact_residual_cache = {}
    opt._exact_jacobian_cache = {}
    opt._initial_tangent_cache = {}
    opt._last_jacobian_key = [None]
    opt._last_jacobian_residual = None
    opt._last_jacobian_source = "exact_tape_replay"
    opt._static = object()
    opt._inner_max_iter = 0
    opt._inner_ftol = 0.0
    opt._trial_max_iter = 0
    opt._trial_ftol = 0.0
    opt._solver_device_name = None
    opt._scan_exact_path = "tape"
    opt._exact_cache_key = lambda _params: b"accepted"
    opt._aspect_target = 7.0
    opt._aspect_weight = 2.0
    opt._n_non_qs = 1
    opt._n_qs = None
    opt._has_residual_block_metadata = True
    opt._callback_trace_enabled = False
    opt._callback_trace = []
    opt._callback_point_ids = {}
    opt._callback_previous_key = None
    opt._post_jacobian_clear = lambda *args, **kwargs: None
    opt._solve_exact_with_tape = lambda _params, return_payload=False: (state, {}) if return_payload else state
    opt._cached_exact_state = lambda _params: state
    opt.residual_fun = lambda _params: residual.copy()
    opt.forward_residual_fun = lambda _params: (_ for _ in ()).throw(
        AssertionError("line-search trial residual should not run")
    )
    opt.jacobian_fun = lambda _params: (_ for _ in ()).throw(
        AssertionError("accepted-point Jacobian replay should be skipped")
    )
    opt._evaluate_residuals_from_state = lambda _state: (_ for _ in ()).throw(
        AssertionError("final residual should come from best exact residual")
    )
    opt._qs_total_from_state_fn = lambda _state: (_ for _ in ()).throw(
        AssertionError("QS state callback should not run with residual metadata")
    )

    result = opt.run(np.asarray([0.0]), method="gauss_newton", max_nfev=1, verbose=0)

    assert result["success"] is False
    assert result["njev"] == 0
    assert result["_history_dump"]["objective_final"] == pytest.approx(float(np.dot(residual, residual)))
    assert result["_history_dump"]["qs_final"] == pytest.approx(25.0)
    assert result["_history_dump"]["aspect_final"] == pytest.approx(7.25)


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
            "--budget-replay-dispatch-wall-s",
            "3.5",
            "--budget-residual-tangent-wall-s",
            "3",
            "--budget-accepted-replays",
            "2",
            "--budget-action",
            "warn",
            "--vmec-timing-detail",
            "--scan-arg-summary",
            "--sync-replay-timing",
            "--jvp-only-exact-tape",
            "--jvp-only-basepoint-carries",
            "--initial-metrics",
            "--exact-jit-forces",
            "off",
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
    assert args.budget_replay_dispatch_wall_s == 3.5
    assert args.budget_residual_tangent_wall_s == 3.0
    assert args.budget_accepted_replays == 2
    assert args.budget_action == "warn"
    assert args.vmec_timing_detail is True
    assert args.scan_arg_summary is True
    assert args.sync_replay_timing is True
    assert args.jvp_only_exact_tape is True
    assert args.jvp_only_basepoint_carries is True
    assert args.initial_metrics is True
    assert args.exact_jit_forces == "off"


def test_exact_optimizer_profile_skips_initial_metrics_by_default() -> None:
    args = exact_profile_tool._parse_args(["--callback", "jacobian"])

    assert args.initial_metrics is False
    assert args.jvp_only_exact_tape is None
    assert args.jvp_only_basepoint_carries is None


def test_exact_optimizer_callback_profiler_helpers_record_trace_and_memory(tmp_path) -> None:
    class FakeProfiler:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str | None]] = []

        def start_trace(self, path: str) -> None:
            self.calls.append(("start", path))

        def stop_trace(self) -> None:
            self.calls.append(("stop", None))

        def save_device_memory_profile(self, path: str) -> None:
            self.calls.append(("memory", path))
            Path(path).write_bytes(b"fake-profile")

    fake_jax = SimpleNamespace(profiler=FakeProfiler())
    trace_dir = tmp_path / "trace"
    memory_file = tmp_path / "memory.prof"

    resolved_trace = exact_profile_tool._start_profiler_trace(fake_jax, str(trace_dir))
    exact_profile_tool._stop_profiler_trace(fake_jax, resolved_trace)
    resolved_memory = exact_profile_tool._save_device_memory_profile(fake_jax, str(memory_file))

    assert resolved_trace == str(trace_dir.resolve())
    assert resolved_memory == str(memory_file.resolve())
    assert trace_dir.is_dir()
    assert memory_file.read_bytes() == b"fake-profile"
    assert fake_jax.profiler.calls == [
        ("start", str(trace_dir.resolve())),
        ("stop", None),
        ("memory", str(memory_file.resolve())),
    ]
    assert exact_profile_tool._start_profiler_trace(fake_jax, None) is None
    exact_profile_tool._stop_profiler_trace(fake_jax, None)
    assert exact_profile_tool._save_device_memory_profile(fake_jax, None) is None


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
    opt._initial_state_cache = OrderedDict([("c", object())])
    opt._exact_state_key_by_id = {1: "a"}
    opt._initial_tangent_cache = {}
    opt._discrete_jacobian_helper_cache = {"j": object()}
    opt._scan_exact_helper_cache = {}

    snapshot = exact_profile_tool._cache_snapshot(opt, include_global=False)
    assert snapshot["optimizer"]["exact_cache"] == 1
    assert snapshot["optimizer"]["exact_jacobian_cache"] == 1
    assert snapshot["optimizer"]["trial_residual_cache"] == 1
    assert snapshot["optimizer"]["initial_state_cache"] == 1
    assert snapshot["optimizer"]["exact_state_key_by_id"] == 1
    assert snapshot["total_entries"] == 7

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


def test_exact_optimizer_profile_timing_splits_direct_tape_build_leaves() -> None:
    opt = FixedBoundaryExactOptimizer.__new__(FixedBoundaryExactOptimizer)
    opt._profile = {}
    tape = SimpleNamespace(
        diagnostics={
            "timing": {
                "compute_forces_s": 0.10,
                "preconditioner_s": 0.40,
                "update_s": 0.20,
                "tape_solve_call_s": 0.85,
                "tape_final_state_pack_s": 0.03,
                "tape_step_trace_extract_s": 0.04,
                "tape_dynamic_payload_build_s": 0.05,
                "tape_trace_stack_s": 0.02,
            }
        }
    )

    opt._profile_exact_tape_solver_timing(tape, tape_build_wall_s=1.20)
    profile = opt._profile_dump()

    assert profile["exact_tape_build_solve_call"]["wall_time_s"] == 0.85
    assert profile["exact_tape_build_final_state_pack"]["wall_time_s"] == 0.03
    assert profile["exact_tape_build_step_trace_extract"]["wall_time_s"] == 0.04
    assert profile["exact_tape_build_dynamic_payload"]["wall_time_s"] == 0.05
    assert profile["exact_tape_build_trace_stack"]["wall_time_s"] == 0.02
    assert profile["exact_tape_solver_compute_forces"]["wall_time_s"] == 0.10
    assert profile["exact_tape_build_unattributed"]["wall_time_s"] == pytest.approx(0.21)


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
                "iteration_control_s": 0.07,
                "iteration_control_fsq1_s": 0.05,
                "iteration_control_fsq1_precond_norm_s": 0.011,
                "iteration_control_fsq1_scalar_build_s": 0.012,
                "iteration_control_fsq1_payload_get_s": 0.013,
                "iteration_control_fsq1_direct_get_s": 0.014,
                "iteration_control_fsq1_unattributed_s": 0.015,
                "iteration_control_badjac_s": 0.04,
                "iteration_control_badjac_ptau_get_s": 0.016,
                "iteration_control_badjac_state_jacobian_s": 0.017,
                "iteration_control_badjac_unattributed_s": 0.018,
                "iteration_control_vmec_time_s": 0.021,
                "iteration_control_restart_s": 0.022,
                "iteration_control_evolve_s": 0.023,
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
    assert profile["trial_solver_iteration_control"]["wall_time_s"] == 0.07
    assert profile["trial_solver_iteration_control_fsq1"]["wall_time_s"] == 0.05
    assert profile["trial_solver_iteration_control_fsq1_precond_norm"]["wall_time_s"] == 0.011
    assert profile["trial_solver_iteration_control_fsq1_scalar_build"]["wall_time_s"] == 0.012
    assert profile["trial_solver_iteration_control_fsq1_payload_get"]["wall_time_s"] == 0.013
    assert profile["trial_solver_iteration_control_fsq1_direct_get"]["wall_time_s"] == 0.014
    assert profile["trial_solver_iteration_control_fsq1_unattributed"]["wall_time_s"] == 0.015
    assert profile["trial_solver_iteration_control_badjac"]["wall_time_s"] == 0.04
    assert profile["trial_solver_iteration_control_badjac_ptau_get"]["wall_time_s"] == 0.016
    assert profile["trial_solver_iteration_control_badjac_state_jacobian"]["wall_time_s"] == 0.017
    assert profile["trial_solver_iteration_control_badjac_unattributed"]["wall_time_s"] == 0.018
    assert profile["trial_solver_iteration_control_vmec_time"]["wall_time_s"] == 0.021
    assert profile["trial_solver_iteration_control_restart"]["wall_time_s"] == 0.022
    assert profile["trial_solver_iteration_control_evolve"]["wall_time_s"] == 0.023
    assert profile["solve_forward_trial_unattributed"]["wall_time_s"] == pytest.approx(0.15)


def test_exact_optimizer_profiles_solver_outer_timing_without_double_counting() -> None:
    opt = FixedBoundaryExactOptimizer.__new__(FixedBoundaryExactOptimizer)
    opt._profile = {}

    opt._profile_solver_timing(
        {
            "timing": {
                "setup_total_s": 0.20,
                "setup_axis_reset_s": 0.03,
                "setup_unattributed_s": 0.17,
                "iteration_loop_s": 0.70,
                "iteration_prepare_s": 0.08,
                "compute_forces_s": 0.20,
                "iteration_residual_metrics_s": 0.09,
                "preconditioner_s": 0.15,
                "update_s": 0.10,
                "iteration_post_update_s": 0.04,
                "iteration_loop_unattributed_s": 0.04,
                "finalize_s": 0.05,
            }
        },
        profile_prefix="exact_tape_solver",
        phase_wall_s=1.00,
        unattributed_name="solve_forward_exact_unattributed",
    )
    profile = opt._profile_dump()

    assert profile["exact_tape_solver_setup_total"]["wall_time_s"] == 0.20
    assert profile["exact_tape_solver_setup_axis_reset"]["wall_time_s"] == 0.03
    assert profile["exact_tape_solver_iteration_loop"]["wall_time_s"] == 0.70
    assert profile["exact_tape_solver_iteration_residual_metrics"]["wall_time_s"] == 0.09
    assert profile["exact_tape_solver_iteration_loop_unattributed"]["wall_time_s"] == 0.04
    assert profile["exact_tape_solver_finalize"]["wall_time_s"] == 0.05
    assert profile["solve_forward_exact_unattributed"]["wall_time_s"] == pytest.approx(0.05)


def test_exact_optimizer_profiles_scan_solver_timing_buckets() -> None:
    opt = FixedBoundaryExactOptimizer.__new__(FixedBoundaryExactOptimizer)
    opt._profile = {}

    opt._profile_solver_timing(
        {
            "timing": {
                "iterations": 4,
                "scan_total_s": 0.55,
                "scan_setup_s": 0.01,
                "scan_preflight_s": 0.05,
                "scan_run_setup_s": 0.02,
                "scan_device_run_s": 0.40,
                "scan_device_dispatch_s": 0.04,
                "scan_device_ready_s": 0.36,
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
    assert profile["trial_solver_scan_setup"]["wall_time_s"] == 0.01
    assert profile["trial_solver_scan_preflight"]["wall_time_s"] == 0.05
    assert profile["trial_solver_scan_run_setup"]["wall_time_s"] == 0.02
    assert profile["trial_solver_scan_device_run"]["wall_time_s"] == 0.40
    assert profile["trial_solver_scan_device_dispatch"]["wall_time_s"] == 0.04
    assert profile["trial_solver_scan_device_ready"]["wall_time_s"] == 0.36
    assert profile["trial_solver_scan_host_materialize"]["wall_time_s"] == 0.03
    assert profile["trial_solver_scan_postprocess"]["wall_time_s"] == 0.07
    assert profile["solve_forward_trial_unattributed"]["wall_time_s"] == pytest.approx(0.10)


def test_profile_exact_supplements_scan_cache_status_timing_on_older_optimizer() -> None:
    class OlderOptimizer:
        def __init__(self) -> None:
            self._profile = {}

        def _profile_add(self, name: str, wall_time_s: float) -> None:
            rec = self._profile.setdefault(name, {"count": 0, "wall_time_s": 0.0})
            rec["count"] += 1
            rec["wall_time_s"] += float(wall_time_s)
            rec["mean_wall_time_s"] = rec["wall_time_s"] / rec["count"]

        def _profile_add_counter(self, name: str, value: int) -> None:
            rec = self._profile.setdefault(name, {"count": 0, "wall_time_s": 0.0})
            rec["count"] += 1
            rec["wall_time_s"] += int(value)
            rec["mean_wall_time_s"] = rec["wall_time_s"] / rec["count"]

        def _profile_solver_timing(
            self,
            diagnostics,
            *,
            profile_prefix: str,
            phase_wall_s: float,
            unattributed_name: str | None,
        ) -> float:
            del unattributed_name
            self._profile_add(f"{profile_prefix}_scan_total", phase_wall_s)
            return phase_wall_s

    opt = OlderOptimizer()
    exact_profile_tool._install_profile_timing_supplements(opt)
    opt._profile_solver_timing(
        {
            "timing": {
                "scan_runner_cache_lookup_s": 0.02,
                "scan_runner_cache_build_s": 0.30,
                "scan_runner_cache_hit_count": 2,
                "scan_runner_cache_miss_count": 1,
                "scan_runner_cache_bypass_count": 0,
                "scan_runner_cache_miss_category_iteration_budget_count": 1,
                "scan_runner_cache_hit_device_run_s": 0.40,
                "scan_runner_cache_hit_dispatch_s": 0.05,
                "scan_runner_cache_hit_ready_s": 0.35,
                "scan_runner_cache_miss_device_run_s": 1.20,
                "scan_runner_cache_miss_dispatch_s": 0.10,
                "scan_runner_cache_miss_ready_s": 1.10,
                "scan_runner_arg_leaf_count": 15,
                "scan_runner_arg_array_leaf_count": 12,
                "scan_runner_arg_scalar_leaf_count": 3,
                "scan_runner_arg_array_nbytes": 2048,
                "scan_runner_arg_category_state_leaf_count": 6,
                "scan_runner_arg_category_state_array_leaf_count": 6,
                "scan_runner_arg_category_state_array_nbytes": 1536,
                "scan_runner_arg_path_arg0_state_leaf_count": 6,
                "scan_runner_arg_path_arg0_state_array_leaf_count": 6,
                "scan_runner_arg_path_arg0_state_array_nbytes": 1536,
                "scan_runner_arg_preconditioner_rz_mats_key_count": 8,
                "scan_runner_arg_preconditioner_rz_mats_unexpected_key_count": 0,
                "scan_runner_arg_preconditioner_rz_mats_missing_mandatory_key_count": 0,
                "scan_runner_arg_preconditioner_rz_mats_compact_ok_count": 1,
                "scan_history_none": 0,
                "scan_history_leaf_count": 9,
                "scan_history_array_leaf_count": 9,
                "scan_history_scalar_leaf_count": 0,
                "scan_history_array_nbytes": 288,
                "iteration_control_fsq1_payload_get_s": 0.06,
                "iteration_control_badjac_state_jacobian_s": 0.07,
            }
        },
        profile_prefix="trial_solver",
        phase_wall_s=1.8,
        unattributed_name=None,
    )

    assert opt._profile["trial_solver_scan_runner_cache_lookup"]["wall_time_s"] == 0.02
    assert opt._profile["trial_solver_scan_runner_cache_build"]["wall_time_s"] == 0.30
    assert opt._profile["trial_solver_scan_runner_cache_hit_count"]["wall_time_s"] == 2
    assert opt._profile["trial_solver_scan_runner_cache_miss_count"]["wall_time_s"] == 1
    assert opt._profile["trial_solver_scan_runner_cache_miss_category_iteration_budget_count"]["wall_time_s"] == 1
    assert opt._profile["trial_solver_scan_runner_cache_hit_ready"]["wall_time_s"] == 0.35
    assert opt._profile["trial_solver_scan_runner_cache_miss_ready"]["wall_time_s"] == 1.10
    assert opt._profile["trial_solver_scan_runner_arg_leaf_count"]["wall_time_s"] == 15
    assert opt._profile["trial_solver_scan_runner_arg_array_nbytes"]["wall_time_s"] == 2048
    assert opt._profile["trial_solver_scan_runner_arg_category_state_array_nbytes"]["wall_time_s"] == 1536
    assert opt._profile["trial_solver_scan_runner_arg_path_arg0_state_array_nbytes"]["wall_time_s"] == 1536
    assert opt._profile["trial_solver_scan_runner_arg_preconditioner_rz_mats_key_count"]["wall_time_s"] == 8
    assert opt._profile["trial_solver_scan_runner_arg_preconditioner_rz_mats_compact_ok_count"]["wall_time_s"] == 1
    assert opt._profile["trial_solver_scan_history_none"]["wall_time_s"] == 0
    assert opt._profile["trial_solver_scan_history_leaf_count"]["wall_time_s"] == 9
    assert opt._profile["trial_solver_scan_history_array_leaf_count"]["wall_time_s"] == 9
    assert opt._profile["trial_solver_scan_history_array_nbytes"]["wall_time_s"] == 288
    assert opt._profile["trial_solver_iteration_control_fsq1_payload_get"]["wall_time_s"] == 0.06
    assert opt._profile["trial_solver_iteration_control_badjac_state_jacobian"]["wall_time_s"] == 0.07


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
            "--budget-replay-dispatch-wall-s",
            "0.15",
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
                "replay_scan_cache_diagnostics": {
                    "replay_checkpoint_scan_cache_hit_count": 1,
                    "replay_checkpoint_scan_cache_miss_count": 2,
                    "replay_checkpoint_scan_cache_lookup_s": 0.01,
                    "replay_checkpoint_scan_cache_build_s": 0.05,
                    "replay_dynamic_scan_cache_miss_count": 1,
                    "replay_dynamic_scan_cache_build_s": 0.07,
                },
                "cache_growth": {"total_entries_delta": 2},
            }
        ],
        profile={
            "exact_tape_build": {"count": 1, "wall_time_s": 0.4, "mean_wall_time_s": 0.4},
            "jacobian_tape_replay": {"count": 1, "wall_time_s": 0.25, "mean_wall_time_s": 0.25},
            "jacobian_tape_replay_dispatch": {"count": 1, "wall_time_s": 0.18, "mean_wall_time_s": 0.18},
            "jacobian_residual_tangents": {"count": 1, "wall_time_s": 0.15, "mean_wall_time_s": 0.15},
        },
        cache_before=cache_before,
        cache_after=cache_after,
        rss_before_bytes=100 * 1024 * 1024,
        rss_after_bytes=103 * 1024 * 1024,
        total_wall_s=1.25,
        runtime={"default_backend": "cpu"},
        trace_outdir="/tmp/vmec_jax_trace",
        device_memory_profile_out="/tmp/vmec_jax_memory.prof",
    )

    assert report["schema_version"] == 2
    assert report["report_kind"] == "exact_optimizer_callback_profile"
    assert report["callback"] == "exact"
    assert report["trace_outdir"] == "/tmp/vmec_jax_trace"
    assert report["device_memory_profile_out"] == "/tmp/vmec_jax_memory.prof"
    assert report["jvp_only_exact_tape_requested"] is False
    assert report["jvp_only_basepoint_carries_requested"] is False
    assert report["jvp_only_exact_tape"] is False
    assert report["replay_scan_cache_diagnostics"]["replay_checkpoint_scan_cache_miss_count"] == 2
    assert report["replay_scan_cache_diagnostics"]["replay_dynamic_scan_cache_build_s"] == 0.07
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
        "replay_dispatch_wall_s",
        "residual_tangent_wall_s",
        "accepted_replays",
    }
    assert report["budget_status"]["measurements"]["replay_dispatch_wall_s"] == 0.18
    assert report["budget_status"]["measurements"]["accepted_replays"] == 1


def test_exact_optimizer_callback_report_marks_effective_gpu_jvp_defaults() -> None:
    args = exact_profile_tool._parse_args(
        [
            "--problem",
            "qh",
            "--max-mode",
            "2",
            "--callback",
            "jacobian",
        ]
    )
    report = exact_profile_tool._build_callback_payload(
        args=args,
        specs_count=24,
        solver_device_resolved="default",
        samples=[{"repeat": 0, "wall_time_s": 1.0, "replay_scan_cache_diagnostics": {}}],
        profile={
            "exact_solve_with_tape_jvp_only_total": {
                "count": 1,
                "wall_time_s": 4.0,
                "mean_wall_time_s": 4.0,
            },
            "exact_tape_build_jvp_only": {
                "count": 1,
                "wall_time_s": 3.0,
                "mean_wall_time_s": 3.0,
            },
        },
        cache_before={"total_entries": 0},
        cache_after={"total_entries": 0},
        rss_before_bytes=None,
        rss_after_bytes=None,
        total_wall_s=4.5,
        runtime={"default_backend": "gpu"},
    )

    assert report["jvp_only_exact_tape_requested"] is False
    assert report["jvp_only_basepoint_carries_requested"] is False
    assert report["jvp_only_exact_tape"] is True
    assert report["jvp_only_basepoint_carries"] is True


def test_exact_optimizer_callback_budget_status_counts_projected_replay() -> None:
    args = exact_profile_tool._parse_args(
        [
            "--callback",
            "jacobian",
            "--budget-replay-wall-s",
            "0.2",
            "--budget-replay-dispatch-wall-s",
            "0.2",
            "--budget-residual-tangent-wall-s",
            "0.1",
            "--budget-accepted-replays",
            "0",
        ]
    )

    report = exact_profile_tool._build_callback_payload(
        args=args,
        specs_count=8,
        solver_device_resolved="gpu",
        samples=[{"repeat": 0, "wall_time_s": 0.5, "profile_delta": {}}],
        profile={
            "jacobian_projected_replay_total": {"count": 1, "wall_time_s": 0.25, "mean_wall_time_s": 0.25},
            "jacobian_projected_tape_replay_dispatch": {
                "count": 1,
                "wall_time_s": 0.22,
                "mean_wall_time_s": 0.22,
            },
            "jacobian_projected_replay_residual_tangents": {
                "count": 1,
                "wall_time_s": 0.15,
                "mean_wall_time_s": 0.15,
            },
        },
        cache_before={"optimizer": {}, "total_entries": 0},
        cache_after={"optimizer": {}, "total_entries": 0},
        rss_before_bytes=None,
        rss_after_bytes=None,
        total_wall_s=0.5,
        runtime={"default_backend": "gpu"},
    )

    assert report["budget_status"]["ok"] is False
    assert {
        item["name"] for item in report["budget_status"]["exceeded"]
    } == {
        "replay_wall_s",
        "replay_dispatch_wall_s",
        "residual_tangent_wall_s",
        "accepted_replays",
    }
    assert report["budget_status"]["measurements"]["replay_wall_s"] == 0.25
    assert report["budget_status"]["measurements"]["replay_dispatch_wall_s"] == 0.22
    assert report["budget_status"]["measurements"]["residual_tangent_wall_s"] == 0.15
    assert report["budget_status"]["measurements"]["accepted_replays"] == 1


def test_exact_optimizer_callback_budget_status_counts_chunked_projection_replay() -> None:
    args = exact_profile_tool._parse_args(
        [
            "--callback",
            "jacobian",
            "--budget-replay-wall-s",
            "0.2",
            "--budget-accepted-replays",
            "0",
        ]
    )

    report = exact_profile_tool._build_callback_payload(
        args=args,
        specs_count=80,
        solver_device_resolved="gpu",
        samples=[{"repeat": 0, "wall_time_s": 0.5, "profile_delta": {}}],
        profile={
            "jacobian_chunked_projected_replay_projection_total": {
                "count": 2,
                "wall_time_s": 0.25,
                "mean_wall_time_s": 0.125,
            },
        },
        cache_before={"optimizer": {}, "total_entries": 0},
        cache_after={"optimizer": {}, "total_entries": 0},
        rss_before_bytes=None,
        rss_after_bytes=None,
        total_wall_s=0.5,
        runtime={"default_backend": "gpu"},
    )

    assert report["budget_status"]["ok"] is False
    assert {
        item["name"] for item in report["budget_status"]["exceeded"]
    } == {
        "replay_wall_s",
        "accepted_replays",
    }
    assert report["budget_status"]["measurements"]["replay_wall_s"] == 0.25
    assert report["budget_status"]["measurements"]["accepted_replays"] == 2
