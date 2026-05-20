from __future__ import annotations

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
