from __future__ import annotations

from collections import OrderedDict

import numpy as np

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
