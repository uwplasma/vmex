from __future__ import annotations

from collections import OrderedDict
import os

import numpy as np

import vmec_jax.optimization as opt_module
from vmec_jax.optimization import FixedBoundaryExactOptimizer


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
