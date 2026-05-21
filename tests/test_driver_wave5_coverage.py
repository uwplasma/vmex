from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import vmec_jax.driver as driver
from vmec_jax.solve import SolveVmecResidualResult


def _result(*, n: int = 3, bad_fsql: bool = False):
    fsql = np.asarray([0.3, 0.2, 0.1], dtype=object if bad_fsql else float)
    if bad_fsql:
        fsql[-1] = object()
    return SolveVmecResidualResult(
        state=object(),
        n_iter=n - 1,
        w_history=np.linspace(1.0, 0.1, n),
        fsqr2_history=np.linspace(0.4, 0.2, n),
        fsqz2_history=np.linspace(0.3, 0.1, n),
        fsql2_history=fsql,
        grad_rms_history=np.zeros((0,), dtype=float),
        step_history=np.zeros((0,), dtype=float),
        diagnostics={"converged": True},
    )


def test_list_coercion_and_resume_sanitizers_cover_numpy_scalar_and_bad_step_edges():
    assert driver._as_list_like(np.asarray([1, 2, 3])) == [1, 2, 3]
    assert driver._as_list_like(np.float64(2.5)) == [np.float64(2.5)]
    assert driver._as_list_like((v for v in [4, 5])) == [4, 5]

    resume_state = {"time_step": "0.25"}
    cross_grid = driver._sanitize_resume_state_for_grid_change(resume_state, step_size=object())
    same_grid = driver._sanitize_resume_state_for_same_grid(resume_state, step_size=object())

    assert cross_grid["time_step"] == pytest.approx(0.25)
    assert same_grid["time_step"] == pytest.approx(0.25)
    assert same_grid["inv_tau"] == [pytest.approx(0.6)] * 10
    assert driver._sanitize_resume_state_for_grid_change(None, step_size=1.0) is None
    assert driver._sanitize_resume_state_for_same_grid({}, step_size=1.0) is None


def test_stage_switch_reason_covers_zero_and_nonpositive_decay_edges():
    assert (
        driver._stage_switch_reason_from_progress(
            start_total_fsq=0.0,
            best_total_fsq=-1.0,
            target_total_fsq=-2.0,
            chunk_iters=4,
            remaining_budget=3,
        )
        is None
    )
    assert (
        driver._stage_switch_reason_from_progress(
            start_total_fsq=1.0,
            best_total_fsq=0.0,
            target_total_fsq=-1.0,
            chunk_iters=4,
            remaining_budget=3,
        )
        is None
    )


def test_result_residual_convergence_and_budget_helpers_cover_fallbacks():
    assert driver._result_final_residuals(None) is None
    explicit = SimpleNamespace(diagnostics={"final_fsqr": "1.0", "final_fsqz": 2.0, "final_fsql": np.float64(3.0)})
    assert driver._result_final_residuals(explicit) == (1.0, 2.0, 3.0)

    history = SimpleNamespace(
        diagnostics={"final_fsqr": object(), "final_fsqz": 2.0, "final_fsql": 3.0, "requested_ftol": 10.0},
        fsqr2_history=np.asarray([4.0, 5.0]),
        fsqz2_history=np.asarray([6.0, 7.0]),
        fsql2_history=np.asarray([8.0, 9.0]),
    )
    assert driver._result_final_residuals(history) == (5.0, 7.0, 9.0)

    diag_history = SimpleNamespace(
        diagnostics={
            "fsqr_full": np.asarray([1.25]),
            "fsqz_full": np.asarray([2.5]),
            "fsql_full": np.asarray([3.75]),
        },
        fsqr2_history=np.asarray([object()], dtype=object),
        fsqz2_history=np.asarray([1.0]),
        fsql2_history=np.asarray([1.0]),
    )
    assert driver._result_final_residuals(diag_history) == (1.25, 2.5, 3.75)

    no_residuals = SimpleNamespace(
        diagnostics={"fsqr_full": np.asarray([object()], dtype=object), "fsqz_full": [1.0], "fsql_full": [1.0]},
        fsqr2_history=np.asarray([object()], dtype=object),
        fsqz2_history=np.asarray([1.0]),
        fsql2_history=np.asarray([1.0]),
    )
    assert driver._result_final_residuals(no_residuals) is None
    assert driver._result_final_fsq(None) == np.inf
    assert driver._result_final_fsq(no_residuals) == np.inf

    assert driver._result_meets_requested_ftol(SimpleNamespace(diagnostics={"converged_strict": True}), ftol=0.0)
    assert not driver._result_meets_requested_ftol(SimpleNamespace(diagnostics={"converged_strict": False}), ftol=1.0)
    assert driver._result_meets_requested_ftol(SimpleNamespace(diagnostics={"converged": True}), ftol=0.0)
    assert not driver._result_meets_requested_ftol(no_residuals, ftol=1.0)
    assert driver._result_meets_requested_ftol(history, ftol=10.0)
    assert not driver._result_meets_requested_ftol(history, ftol=1.0)
    assert not driver._result_hits_total_target(None, fsq_total_target=1.0)
    assert not driver._result_hits_total_target(history, fsq_total_target=None)
    assert driver._result_hits_total_target(history, fsq_total_target=100.0)

    assert driver._allocate_integer_budget(total=0, weights=[1, 2]) == [0, 0]
    assert driver._allocate_integer_budget(total=5, weights=[]) == []
    assert driver._allocate_integer_budget(total=5, weights=[0, -1, 0]) == [0, 0, 5]
    assert sum(driver._allocate_integer_budget(total=7, weights=[1, 2, 3])) == 7
    assert driver._accelerated_cli_budgeted_total_iters(total_budget=10, ns_stages=[]) == 10
    assert driver._accelerated_cli_budgeted_total_iters(total_budget=100, ns_stages=[25, 100]) == 50
    assert driver._accelerated_cli_budgeted_stage_iters(total_budget=8, ns_stages=[]) == [8]
    assert driver._accelerated_cli_budgeted_stage_iters(total_budget=1, ns_stages=[8, 9])[-1] == 1
    assert driver._distribute_stage_iters(iters=0, nstep=4) == [0]
    assert driver._distribute_stage_iters(iters=3, nstep=1) == [3]
    assert driver._distribute_stage_iters(iters=2, nstep=5) == [2]
    assert driver._distribute_stage_iters(iters=7, nstep=3) == [3, 2, 2]


def test_stage_chunk_result_merging_preserves_histories_and_terminal_metadata():
    first = _result(n=2)
    first = replace(
        first,
        diagnostics={
            "step_status_history": np.asarray(["accepted"], dtype=object),
            "fsq_prev_history": np.asarray([2.0]),
        },
    )
    second = replace(
        _result(n=3),
        diagnostics={
            "step_status_history": np.asarray(["restart", "accepted"], dtype=object),
            "time_step_history": np.asarray([0.5, 0.25]),
        },
    )

    single = driver._merge_stage_chunk_results([first], mode_i="accelerated")
    assert single.diagnostics["accelerated_stage_chunked"] is False
    assert single.diagnostics["accelerated_stage_effective_mode"] == "accelerated"

    merged = driver._merge_stage_chunk_results([first, second], mode_i="parity")

    assert merged.state is second.state
    assert merged.n_iter == first.n_iter + second.n_iter + 1
    np.testing.assert_allclose(merged.w_history, np.concatenate([first.w_history, second.w_history]))
    np.testing.assert_array_equal(
        merged.diagnostics["step_status_history"],
        np.asarray(["accepted", "restart", "accepted"], dtype=object),
    )
    np.testing.assert_allclose(merged.diagnostics["fsq_prev_history"], [2.0])
    np.testing.assert_allclose(merged.diagnostics["time_step_history"], [0.5, 0.25])
    assert merged.diagnostics["accelerated_stage_chunked"] is True
    assert merged.diagnostics["accelerated_stage_effective_mode"] == "parity"
    np.testing.assert_array_equal(merged.diagnostics["accelerated_stage_chunk_iters"], [2, 3])


def test_wout_from_fixed_boundary_run_uses_sparse_history_sampling_and_restores_env(monkeypatch, tmp_path):
    captured = []

    def fake_wout_minimal_from_fixed_boundary(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(kind="wout", kwargs=kwargs)

    monkeypatch.setattr(driver, "residual_scalars_from_state", lambda **_kwargs: (9.0, 8.0, 7.0))
    monkeypatch.setattr("vmec_jax.wout.wout_minimal_from_fixed_boundary", fake_wout_minimal_from_fixed_boundary)
    monkeypatch.setenv("VMEC_JAX_WOUT_FAST_BCOVAR", "previous")

    run = driver.FixedBoundaryRun(
        cfg=object(),
        indata=object(),
        static=object(),
        state=object(),
        result=_result(n=150),
        flux=object(),
        profiles={},
        signgs=-1,
    )

    out = driver.wout_from_fixed_boundary_run(run, path=tmp_path / "wout.nc", fast_bcovar=False)

    assert out.kind == "wout"
    assert captured[-1]["signgs"] == -1
    assert captured[-1]["converged"] is True
    assert captured[-1]["fsqr"] == pytest.approx(0.2)
    assert captured[-1]["fsqz"] == pytest.approx(0.1)
    assert captured[-1]["fsql"] == pytest.approx(0.1)
    assert captured[-1]["fsqt"].shape == (100,)
    assert np.count_nonzero(captured[-1]["fsqt"]) == 75
    assert captured[-1]["fsqt"][0] == pytest.approx(run.result.fsqr2_history[1] + run.result.fsqz2_history[1])
    assert captured[-1]["fsqt"][74] == pytest.approx(run.result.fsqr2_history[149] + run.result.fsqz2_history[149])
    assert captured[-1]["fsqt"][75] == 0.0
    assert driver.os.environ["VMEC_JAX_WOUT_FAST_BCOVAR"] == "previous"


def test_wout_from_fixed_boundary_run_falls_back_for_bad_histories_and_include_fsq_false(monkeypatch, tmp_path):
    captured = []

    def fake_wout_minimal_from_fixed_boundary(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(kwargs=kwargs)

    monkeypatch.setattr(driver, "residual_scalars_from_state", lambda **_kwargs: (9.0, 8.0, 7.0))
    monkeypatch.setattr("vmec_jax.wout.wout_minimal_from_fixed_boundary", fake_wout_minimal_from_fixed_boundary)
    monkeypatch.delenv("VMEC_JAX_WOUT_FAST_BCOVAR", raising=False)

    run = driver.FixedBoundaryRun(
        cfg=object(),
        indata=object(),
        static=object(),
        state=object(),
        result=_result(bad_fsql=True),
        flux=object(),
        profiles={},
        signgs=1,
    )

    driver.wout_from_fixed_boundary_run(run, path=tmp_path / "fallback.nc", fast_bcovar=True)
    assert captured[-1]["fsqr"] == pytest.approx(9.0)
    assert captured[-1]["fsqz"] == pytest.approx(8.0)
    assert captured[-1]["fsql"] == pytest.approx(7.0)
    assert "VMEC_JAX_WOUT_FAST_BCOVAR" not in driver.os.environ

    driver.wout_from_fixed_boundary_run(run, path=tmp_path / "zeros.nc", include_fsq=False)
    assert captured[-1]["fsqr"] == 0.0
    assert captured[-1]["fsqz"] == 0.0
    assert captured[-1]["fsql"] == 0.0
    assert captured[-1]["fsqt"] is None
