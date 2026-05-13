from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import vmec_jax.driver as driver


class _Input:
    def __init__(self, **values):
        self.values = dict(values)

    def get(self, key, default=None):
        return self.values.get(key, default)

    def get_bool(self, key, default=False):
        return bool(self.values.get(key, default))

    def get_float(self, key, default=0.0):
        return float(self.values.get(key, default))

    def get_int(self, key, default=0):
        return int(self.values.get(key, default))


@pytest.mark.parametrize(
    ("solver_mode", "performance_mode", "expected"),
    [
        (None, True, "default"),
        (None, False, "parity"),
        (" DEFAULT ", False, "default"),
        ("Fast", False, "default"),
        ("SAFE", True, "parity"),
        (" reference ", True, "parity"),
        (" PERF ", False, "accelerated"),
        ("accelerated", True, "accelerated"),
    ],
)
def test_normalize_solver_mode_handles_aliases_case_and_defaults(solver_mode, performance_mode, expected):
    assert driver._normalize_solver_mode(solver_mode=solver_mode, performance_mode=performance_mode) == expected


def test_normalize_solver_mode_reports_valid_modes():
    with pytest.raises(ValueError, match="Expected one of: accelerated, default, parity"):
        driver._normalize_solver_mode(solver_mode="not-a-mode", performance_mode=False)


@pytest.mark.parametrize(
    ("backend", "indata", "expected"),
    [
        ("gpu", _Input(LFREEB=True, NS_ARRAY=[5, 9]), ("default", True)),
        ("gpu", _Input(NS_ARRAY=[5, 9]), ("parity", False)),
        ("gpu", _Input(NS_ARRAY=[9]), ("accelerated", True)),
        ("cpu", _Input(LASYM=True), ("accelerated", True)),
        ("cpu", _Input(NCURR=1, NS_ARRAY=[5, 9], NITER_ARRAY=[10, 20]), ("accelerated", True)),
        ("cpu", _Input(NCURR=1, NS_ARRAY=[5, 9]), ("parity", False)),
        ("cpu", _Input(), ("default", True)),
    ],
)
def test_default_non_autodiff_solver_policy_for_backend_uses_input_structure(backend, indata, expected):
    assert driver._default_non_autodiff_solver_policy_for_backend(indata, backend) == expected


def test_requested_final_ftol_prefers_last_valid_list_value_and_clamps():
    fallback = _Input(FTOL=4.0e-7)
    negative_fallback = _Input(FTOL=-1.0)

    assert driver._requested_final_ftol(indata=fallback, ftol_list_input=("1e-4", 2.0e-5)) == 2.0e-5
    assert driver._requested_final_ftol(indata=fallback, ftol_list_input=[1.0e-4, -2.0]) == 0.0
    assert driver._requested_final_ftol(indata=fallback, ftol_list_input=object()) == 4.0e-7
    assert driver._requested_final_ftol(indata=negative_fallback, ftol_list_input=[]) == 0.0


def test_accelerated_total_target_clamps_negative_ftol():
    assert driver._accelerated_fsq_total_target_from_ftol(-1.0e-8) == 0.0
    assert driver._accelerated_fsq_total_target_from_ftol(2.0e-8) == pytest.approx(6.0e-8)


def test_allocate_integer_budget_clamps_inputs_and_distributes_remainder():
    assert driver._allocate_integer_budget(total=-3, weights=[1, 2, 3]) == [0, 0, 0]
    assert driver._allocate_integer_budget(total=6, weights=[-5, 1, 2]) == [0, 2, 4]
    assert driver._allocate_integer_budget(total=5, weights=[1, 2]) == [2, 3]
    assert driver._allocate_integer_budget(total=5, weights=[0, 0, 0]) == [0, 0, 5]


def test_accelerated_cli_budget_helpers_scale_total_and_weight_stages():
    assert driver._accelerated_cli_budgeted_total_iters(total_budget=100, ns_stages=[9, 36]) == 50
    assert driver._accelerated_cli_budgeted_total_iters(total_budget=0, ns_stages=[4]) == 1
    assert driver._accelerated_cli_budgeted_stage_iters(total_budget=10, ns_stages=[4, 7, 10]) == [5, 3, 2]
    assert driver._accelerated_cli_budgeted_stage_iters(total_budget=0, ns_stages=[]) == [1]


def test_result_final_residuals_prefers_explicit_diagnostics_over_histories():
    result = SimpleNamespace(
        diagnostics={"final_fsqr": "1.0", "final_fsqz": 2, "final_fsql": np.asarray(3.0)},
        fsqr2_history=np.asarray([10.0]),
        fsqz2_history=np.asarray([20.0]),
        fsql2_history=np.asarray([30.0]),
    )

    assert driver._result_final_residuals(result) == (1.0, 2.0, 3.0)


def test_result_final_residuals_falls_back_when_explicit_diagnostics_are_incomplete():
    result = SimpleNamespace(
        diagnostics={"final_fsqr": 1.0, "final_fsqz": None, "final_fsql": 3.0},
        fsqr2_history=np.asarray([[1.0, 4.0]]),
        fsqz2_history=np.asarray([[2.0, 5.0]]),
        fsql2_history=np.asarray([[3.0, 6.0]]),
    )

    assert driver._result_final_residuals(result) == (4.0, 5.0, 6.0)


def test_result_final_residuals_uses_flattened_diagnostic_histories():
    result = SimpleNamespace(
        diagnostics={
            "fsqr_full": np.asarray([[1.0, 2.0]]),
            "fsqz_full": np.asarray([[3.0, 4.0]]),
            "fsql_full": np.asarray([[5.0, 6.0]]),
        },
    )

    assert driver._result_final_residuals(result) == (2.0, 4.0, 6.0)


def test_result_final_fsq_prefers_weight_history_then_residual_sum():
    weighted = SimpleNamespace(
        diagnostics={"final_fsqr": 1.0, "final_fsqz": 2.0, "final_fsql": 3.0},
        w_history=np.asarray([99.0, 0.25]),
    )
    residual_only = SimpleNamespace(diagnostics={"final_fsqr": 1.0, "final_fsqz": 2.0, "final_fsql": 3.0})

    assert driver._result_final_fsq(weighted) == 0.25
    assert driver._result_final_fsq(residual_only) == 6.0
    assert np.isinf(driver._result_final_fsq(None))


def test_result_meets_requested_ftol_respects_strict_diagnostic_flag():
    result = SimpleNamespace(
        diagnostics={"converged_strict": False, "final_fsqr": 0.0, "final_fsqz": 0.0, "final_fsql": 0.0},
    )

    assert driver._result_meets_requested_ftol(result, ftol=1.0) is False


def test_result_meets_requested_ftol_uses_legacy_converged_only_without_ftol_metadata():
    result = SimpleNamespace(
        diagnostics={"converged": False, "final_fsqr": 0.0, "final_fsqz": 0.0, "final_fsql": 0.0},
    )

    assert driver._result_meets_requested_ftol(result, ftol=1.0) is False


def test_result_meets_requested_ftol_clamps_negative_requested_target():
    zero = SimpleNamespace(
        diagnostics={"requested_ftol": -1.0, "final_fsqr": 0.0, "final_fsqz": 0.0, "final_fsql": 0.0},
    )
    nonzero = SimpleNamespace(
        diagnostics={"requested_ftol": -1.0, "final_fsqr": 1.0e-16, "final_fsqz": 0.0, "final_fsql": 0.0},
    )

    assert driver._result_meets_requested_ftol(zero, ftol=-1.0) is True
    assert driver._result_meets_requested_ftol(nonzero, ftol=-1.0) is False


def test_result_hits_total_target_clamps_negative_target_and_handles_none():
    zero = SimpleNamespace(w_history=np.asarray([0.0]), diagnostics={})
    nonzero = SimpleNamespace(w_history=np.asarray([1.0e-12]), diagnostics={})

    assert driver._result_hits_total_target(zero, fsq_total_target=-1.0) is True
    assert driver._result_hits_total_target(nonzero, fsq_total_target=-1.0) is False
    assert driver._result_hits_total_target(zero, fsq_total_target=None) is False
