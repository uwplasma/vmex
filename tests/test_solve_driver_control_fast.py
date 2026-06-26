from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import vmec_jax.driver as driver
from vmec_jax.energy import FluxProfiles
from vmec_jax.solve import (
    SolveVmecResidualResult,
    _host_restart_decision,
    _resolve_cg_tol,
    _resolve_lm_damping,
)
from vmec_jax.kernels.tomnsp import vmec_angle_grid


def _small_grid():
    return vmec_angle_grid(ntheta=4, nzeta=1, nfp=1, lasym=False)


def _fake_state(ns: int):
    arr = np.zeros((int(ns), 1), dtype=float)
    return SimpleNamespace(
        layout=SimpleNamespace(ns=int(ns)),
        Rcos=arr,
        Rsin=arr,
        Zcos=arr,
        Zsin=arr,
        Lcos=arr,
        Lsin=arr,
    )


def _write_minimal_input(tmp_path: Path, *, signgs: int = -1) -> Path:
    path = tmp_path / "input.fast_control"
    path.write_text(
        "&INDATA\n"
        "  LFREEB = F\n"
        "  NFP = 1\n"
        "  MPOL = 2\n"
        "  NTOR = 0\n"
        "  NS = 3\n"
        "  PHIEDGE = 1.0\n"
        f"  SIGNGS = {int(signgs)}\n"
        "  RBC(0,0) = 1.0\n"
        "  ZBS(1,0) = 0.1\n"
        "/\n"
    )
    return path


def _patch_lightweight_driver_core(monkeypatch) -> None:
    def fake_static(cfg, **_kwargs):
        return SimpleNamespace(
            cfg=cfg,
            modes=SimpleNamespace(m=np.asarray([0]), n=np.asarray([0]), K=1),
            grid=SimpleNamespace(theta=np.asarray([0.0]), zeta=np.asarray([0.0]), ntheta=1, nzeta=1),
            s=np.linspace(0.0, 1.0, int(cfg.ns)),
            trig_vmec=None,
        )

    monkeypatch.setattr(driver, "validate_free_boundary_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(driver, "prepare_mgrid_for_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(driver, "build_static", fake_static)
    monkeypatch.setattr(driver, "boundary_from_indata", lambda *args, **kwargs: object())
    monkeypatch.setattr(driver, "initial_guess_from_boundary", lambda static, *_args, **_kwargs: _fake_state(static.cfg.ns))
    monkeypatch.setattr(
        driver,
        "flux_profiles_from_indata",
        lambda _indata, s, *, signgs: FluxProfiles(
            phipf=np.ones_like(np.asarray(s, dtype=float)),
            chipf=np.zeros_like(np.asarray(s, dtype=float)),
            phips=np.ones_like(np.asarray(s, dtype=float)),
            signgs=int(signgs),
            lamscale=np.asarray(1.0),
        ),
    )
    monkeypatch.setattr(
        driver,
        "eval_profiles",
        lambda _indata, s: {"pressure": np.zeros_like(np.asarray(s, dtype=float))},
    )


def _result(state, *, w_history=(1.0,), diagnostics=None):
    diag = {"converged": False, "resume_state": {}}
    if diagnostics:
        diag.update(diagnostics)
    return SolveVmecResidualResult(
        state=state,
        n_iter=max(0, len(w_history) - 1),
        w_history=np.asarray(w_history, dtype=float),
        fsqr2_history=np.asarray([1.0e-8], dtype=float),
        fsqz2_history=np.asarray([2.0e-8], dtype=float),
        fsql2_history=np.asarray([3.0e-8], dtype=float),
        grad_rms_history=np.asarray([], dtype=float),
        step_history=np.asarray([], dtype=float),
        diagnostics=diag,
    )


def test_solve_restart_and_tolerance_helpers_cover_reference_progress_branch():
    decision = _host_restart_decision(
        iter2=60,
        iter1=1,
        fsqr=2.0e-2,
        fsqz=0.0,
        fsql=0.0,
        fsq1=2.0e-2,
        fsq_prev=3.0e-2,
        res0=1.0,
        bad_growth_streak=0,
        pre_restart_reason="none",
        reference_mode=True,
        vmec2000_control=False,
        bad_jacobian=False,
        stage_prev_fsq=None,
        stage_transition_factor=50.0,
        lmove_axis=False,
        vmecpp_restart=False,
        k_preconditioner_update_interval=25,
    )

    assert decision.pre_restart_reason == "bad_progress"
    assert decision.vmecpp_bad_progress is False
    assert _resolve_cg_tol(1.0e-4, current_obj=1.0, initial_obj=2.0, target_obj=0.0, dtype=np.float64) == 1.0e-4
    assert _resolve_lm_damping(0.0, curvature_scale=10.0, dtype=np.float64) == 0.0


def test_result_target_helpers_use_histories_and_ignore_malformed_diagnostic_histories():
    passing = SimpleNamespace(
        diagnostics={"requested_ftol": 1.0e-6},
        fsqr2_history=np.asarray([9.0e-7]),
        fsqz2_history=np.asarray([8.0e-7]),
        fsql2_history=np.asarray([7.0e-7]),
    )
    failing = SimpleNamespace(
        diagnostics={"requested_ftol": 1.0e-6},
        fsqr2_history=np.asarray([9.0e-7]),
        fsqz2_history=np.asarray([1.1e-6]),
        fsql2_history=np.asarray([7.0e-7]),
    )

    class BadArray:
        def __array__(self, dtype=None, copy=None):
            raise TypeError("not array-like")

    malformed = SimpleNamespace(diagnostics={"fsqr_full": BadArray(), "fsqz_full": [0.0], "fsql_full": [0.0]})

    assert driver._result_meets_requested_ftol(passing, ftol=1.0e-6) is True
    assert driver._result_meets_requested_ftol(failing, ftol=1.0e-6) is False
    assert driver._result_final_residuals(malformed) is None


def test_resume_sanitizers_handle_missing_payloads_and_nonnumeric_step_cap():
    resume = {"time_step": 0.25, "inv_tau": [1.0], "iter_offset": 4}

    assert driver._sanitize_resume_state_for_grid_change(None, step_size=0.1) is None
    assert driver._sanitize_resume_state_for_grid_change({}, step_size=0.1) is None
    assert driver._sanitize_resume_state_for_same_grid({}, step_size=0.1) is None

    cross_grid = driver._sanitize_resume_state_for_grid_change(resume, step_size="not-a-float")
    same_grid = driver._sanitize_resume_state_for_same_grid(resume, step_size="not-a-float")

    assert cross_grid["time_step"] == pytest.approx(0.25)
    assert cross_grid["iter_offset"] == 0
    assert same_grid["time_step"] == pytest.approx(0.25)
    assert same_grid["iter_offset"] == 4


def test_wout_packing_downsamples_fsqt_and_recomputes_when_final_fsql_history_is_bad(monkeypatch, tmp_path):
    import vmec_jax.wout as wout_module

    captured = {}

    class BadArray:
        def __array__(self, dtype=None, copy=None):
            raise TypeError("bad fsql history")

    def fake_wout_minimal_from_fixed_boundary(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(path=kwargs["path"])

    monkeypatch.setattr(wout_module, "wout_minimal_from_fixed_boundary", fake_wout_minimal_from_fixed_boundary)
    monkeypatch.setattr(driver, "residual_scalars_from_state", lambda **_kwargs: (7.0, 8.0, 9.0))

    fsqr = np.arange(250.0)
    fsqz = 1000.0 + np.arange(250.0)
    result = SimpleNamespace(
        diagnostics={"converged": False},
        fsqr2_history=fsqr,
        fsqz2_history=fsqz,
        fsql2_history=BadArray(),
    )
    run = driver.FixedBoundaryRun(
        cfg=object(),
        indata=object(),
        static=object(),
        state=object(),
        result=result,
        flux=None,
        profiles={},
        signgs=-1,
    )

    out = driver.wout_from_fixed_boundary_run(run, include_fsq=True, path=tmp_path / "wout_fast.nc")

    assert out.path == tmp_path / "wout_fast.nc"
    assert captured["fsqr"] == 7.0
    assert captured["fsqz"] == 8.0
    assert captured["fsql"] == 9.0
    assert captured["converged"] is False
    assert captured["fsqt"][0] == pytest.approx(fsqr[2] + fsqz[2])
    assert captured["fsqt"][82] == pytest.approx(fsqr[248] + fsqz[248])
    np.testing.assert_allclose(captured["fsqt"][83:], 0.0)


def test_run_fixed_boundary_gd_uses_api_defaults_and_sanitizes_invalid_signgs(monkeypatch, tmp_path):
    calls = []

    def fake_solver(state, static, **kwargs):
        calls.append(dict(kwargs))
        return _result(state, w_history=(1.0, 0.5), diagnostics={"converged": True})

    monkeypatch.setenv("VMEC_JAX_DISABLE_JIT_INIT", "1")
    _patch_lightweight_driver_core(monkeypatch)
    monkeypatch.setattr(driver, "solve_fixed_boundary_gd", fake_solver)

    run = driver.run_fixed_boundary(
        _write_minimal_input(tmp_path, signgs=5),
        solver="gd",
        verbose=False,
        grid=_small_grid(),
        performance_mode=False,
    )

    assert run.result is not None
    assert len(calls) == 1
    assert calls[0]["max_iter"] == 10
    assert calls[0]["step_size"] == pytest.approx(5.0e-3)
    assert calls[0]["signgs"] == -1
    assert run.signgs == -1
