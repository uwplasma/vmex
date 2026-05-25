from __future__ import annotations

import numpy as np
import pytest

import vmec_jax.solve as solve
from vmec_jax.solve_residual_iter_runtime_helpers import _build_residual_iter_timing_report


def test_residual_iter_timing_report_exposes_force_eval_aliases() -> None:
    timing_stats = {
        "setup_total": 0.25,
        "setup_axis_reset": 0.05,
        "setup_axis_reset_compute_forces": 0.02,
        "iteration_loop": 1.0,
        "iteration_prepare": 0.1,
        "iteration_residual_metrics": 0.2,
        "iteration_post_update": 0.05,
        "finalize": 0.03,
        "compute_forces": 0.4,
        "compute_forces_first": 0.15,
        "compute_forces_rest": 0.25,
        "compute_forces_calls": 3,
        "compute_forces_main": 0.4,
        "compute_forces_main_calls": 3,
        "compute_forces_auto_flip": 0.03,
        "compute_forces_auto_flip_calls": 2,
        "compute_forces_trial": 0.05,
        "compute_forces_trial_calls": 1,
        "compute_forces_backtracking": 0.07,
        "compute_forces_backtracking_calls": 1,
        "preconditioner": 0.12,
        "precond_apply": 0.08,
        "precond_mode_scale": 0.01,
        "update": 0.09,
        "update_state": 0.07,
        "update_trace_build": 0.0,
        "update_trace_finalize": 0.0,
        "precond_refresh": 0.04,
        "iterations": 2,
    }

    report = _build_residual_iter_timing_report(
        timing_stats,
        solve_total_s=1.5,
        timing_detail_enabled=True,
    )

    assert report["force_eval_s"] == pytest.approx(report["compute_forces_s"])
    assert report["force_eval_first_s"] == pytest.approx(report["compute_forces_first_s"])
    assert report["force_eval_rest_s"] == pytest.approx(report["compute_forces_rest_s"])
    assert report["force_eval_calls"] == report["compute_forces_calls"]
    assert report["force_eval_per_iter_s"] == pytest.approx(0.2)
    assert report["compute_forces_main_s"] == pytest.approx(0.4)
    assert report["force_eval_extra_s"] == pytest.approx(0.15)
    assert report["force_eval_all_s"] == pytest.approx(0.55)
    assert report["force_eval_all_calls"] == 7


def test_accelerated_scan_timing_is_opt_in_and_path_labeled(
    load_case_circular_tokamak,
    monkeypatch,
    capsys,
) -> None:
    pytest.importorskip("jax")

    _cfg, indata, static, _boundary, state0 = load_case_circular_tokamak
    monkeypatch.setenv("VMEC_JAX_SCAN_PRINT", "0")
    monkeypatch.setenv("VMEC_JAX_SCAN_LIGHT", "0")
    monkeypatch.setenv("VMEC_JAX_SCAN_MINIMAL", "0")
    monkeypatch.setenv("VMEC_JAX_SCAN_FALLBACK", "0")
    monkeypatch.setenv("VMEC_JAX_TIMING", "1")

    result = solve.solve_fixed_boundary_residual_iter(
        state0,
        static,
        indata=indata,
        signgs=1,
        max_iter=1,
        step_size=float(indata.get_float("DELT", 1.0)),
        vmec2000_control=False,
        strict_update=False,
        backtracking=False,
        auto_flip_force=False,
        use_restart_triggers=False,
        use_direct_fallback=False,
        precond_radial_alpha=0.0,
        precond_lambda_alpha=0.0,
        jit_forces=False,
        use_scan=True,
        scan_minimal_default=False,
        verbose=False,
        verbose_vmec2000_table=False,
    )

    captured = capsys.readouterr()
    assert "[vmec_jax timing]" not in captured.out
    assert result.diagnostics["use_scan"] is True
    assert result.diagnostics["accelerated_scan"] is True
    assert result.diagnostics["scan_path"] == "accelerated"
    timing = result.diagnostics["timing"]
    assert timing["scan_total_s"] >= 0.0
    assert timing["scan_device_run_s"] >= 0.0
    assert timing["scan_host_materialize_s"] >= 0.0
    assert (
        timing["scan_runner_cache_hit_count"]
        + timing["scan_runner_cache_miss_count"]
        + timing["scan_runner_cache_bypass_count"]
    ) >= 1
    assert np.isfinite(result.w_history[-1])
