from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import vmec_jax.driver as driver


def _fixed_boundary_run(*, result=None, signgs: int = 1) -> driver.FixedBoundaryRun:
    return driver.FixedBoundaryRun(
        cfg=object(),
        indata=object(),
        static=object(),
        state=object(),
        result=result,
        flux=object(),
        profiles={},
        signgs=signgs,
    )


def test_wout_from_fixed_boundary_run_uses_residual_scalars_when_result_missing(monkeypatch, tmp_path: Path) -> None:
    captured = []

    def fake_wout_minimal_from_fixed_boundary(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(kind="wout", kwargs=kwargs)

    monkeypatch.setattr(driver, "residual_scalars_from_state", lambda **_kwargs: (1.25, 2.5, 3.75))
    monkeypatch.setattr("vmec_jax.wout.wout_minimal_from_fixed_boundary", fake_wout_minimal_from_fixed_boundary)

    out = driver.wout_from_fixed_boundary_run(
        _fixed_boundary_run(result=None, signgs=-1),
        path=tmp_path / "wout_missing_result.nc",
    )

    assert out.kind == "wout"
    assert captured[-1]["signgs"] == -1
    assert captured[-1]["converged"] is None
    assert captured[-1]["fsqr"] == pytest.approx(1.25)
    assert captured[-1]["fsqz"] == pytest.approx(2.5)
    assert captured[-1]["fsql"] == pytest.approx(3.75)
    assert captured[-1]["fsqt"] is None


def test_write_wout_from_fixed_boundary_run_creates_parent_and_overwrites(monkeypatch, tmp_path: Path) -> None:
    calls = []
    fake_wout = SimpleNamespace(kind="synthetic")
    out_path = tmp_path / "nested" / "wout_case.nc"

    def fake_wout_from_fixed_boundary_run(run, *, include_fsq, path, fast_bcovar):
        calls.append(("build", run, include_fsq, Path(path), fast_bcovar))
        return fake_wout

    def fake_write_wout(path, wout, *, overwrite):
        calls.append(("write", Path(path), wout, overwrite, Path(path).parent.exists()))

    monkeypatch.setattr(driver, "wout_from_fixed_boundary_run", fake_wout_from_fixed_boundary_run)
    monkeypatch.setattr("vmec_jax.wout.write_wout", fake_write_wout)

    returned = driver.write_wout_from_fixed_boundary_run(
        out_path,
        _fixed_boundary_run(),
        include_fsq=False,
        fast_bcovar=True,
    )

    assert returned is fake_wout
    assert calls[0][0] == "build"
    assert calls[0][2:] == (False, out_path, True)
    assert calls[1] == ("write", out_path, fake_wout, True, True)


def test_result_final_fsq_uses_residual_sum_when_history_is_unusable() -> None:
    result = SimpleNamespace(
        diagnostics={"final_fsqr": 0.5, "final_fsqz": "1.25", "final_fsql": np.float64(2.0)},
        w_history=np.asarray([object()], dtype=object),
    )

    assert driver._result_final_residuals(result) == (0.5, 1.25, 2.0)
    assert driver._result_final_fsq(result) == pytest.approx(3.75)


def test_stage_budget_helpers_keep_final_stage_nonzero_after_weight_rounding() -> None:
    budgets = driver._accelerated_cli_budgeted_stage_iters(total_budget=2, ns_stages=[20, 20, 21])

    assert budgets[-1] == 1
    assert len(budgets) == 3
    assert driver._accelerated_cli_budgeted_total_iters(total_budget=81, ns_stages=[9, 81]) == 27
