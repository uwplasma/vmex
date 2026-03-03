from __future__ import annotations

from tools.diagnostics.parity_sweep_manifest import _evaluate_freeb_thresholds


def test_evaluate_freeb_thresholds_global_pass() -> None:
    case = {"metric_thresholds_rel_scaled": {"source_sym": 1.0e-2, "amatrix": 2.0e-1}}
    runs = [
        {"iter": 53, "metrics_full": {"source_sym": {"rel_scaled": 5.0e-3}, "amatrix": {"rel_scaled": 1.0e-1}}},
        {"iter": 60, "metrics_full": {"amatrix": {"rel_scaled": 1.5e-1}}},
    ]
    ok, report = _evaluate_freeb_thresholds(case, runs)
    assert ok
    assert report["global"]["source_sym"]["pass"]
    assert report["global"]["amatrix"]["pass"]


def test_evaluate_freeb_thresholds_global_fail_on_limit() -> None:
    case = {"metric_thresholds_rel_scaled": {"source_sym": 1.0e-2}}
    runs = [{"iter": 53, "metrics_full": {"source_sym": {"rel_scaled": 2.0e-2}}}]
    ok, report = _evaluate_freeb_thresholds(case, runs)
    assert not ok
    assert not report["global"]["source_sym"]["pass"]


def test_evaluate_freeb_thresholds_by_iter_and_missing_metric_fail() -> None:
    case = {
        "metric_thresholds_rel_scaled_by_iter": {
            "53": {"source_sym": 1.0e-2},
            "54": {"potvac": 5.0e-1},
        }
    }
    runs = [{"iter": 53, "metrics_full": {"source_sym": {"rel_scaled": 5.0e-3}}}]
    ok, report = _evaluate_freeb_thresholds(case, runs)
    assert not ok
    assert report["by_iter"]["53"]["source_sym"]["pass"]
    assert not report["by_iter"]["54"]["potvac"]["pass"]


def test_evaluate_freeb_thresholds_bad_iter_key_fails() -> None:
    case = {"metric_thresholds_rel_scaled_by_iter": {"iter53": {"source_sym": 1.0e-2}}}
    runs = [{"iter": 53, "metrics_full": {"source_sym": {"rel_scaled": 5.0e-3}}}]
    ok, report = _evaluate_freeb_thresholds(case, runs)
    assert not ok
    assert report["by_iter"]["iter53"]["pass"] is False
    assert "error" in report["by_iter"]["iter53"]
