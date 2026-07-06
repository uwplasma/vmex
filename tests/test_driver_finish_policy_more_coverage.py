from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import vmec_jax.driver as driver
from vmec_jax.config import FreeBoundaryConfig, VMECConfig
from vmec_jax.energy import FluxProfiles
from vmec_jax.namelist import InData
from vmec_jax.solve import SolveVmecResidualResult


def _cfg(*, ns: int = 5, lasym: bool = False, lthreed: bool = False) -> VMECConfig:
    return VMECConfig(
        mpol=2,
        ntor=1 if lthreed else 0,
        ns=int(ns),
        nfp=1,
        lasym=bool(lasym),
        lthreed=bool(lthreed),
        lconm1=True,
        ntheta=4,
        nzeta=2 if (lasym or lthreed) else 1,
        free_boundary=FreeBoundaryConfig(
            enabled=False,
            mgrid_file="NONE",
            extcur=(),
            nvacskip=1,
        ),
    )


def _indata(**values) -> InData:
    scalars = {
        "LFREEB": False,
        "NITER": 2,
        "FTOL": 1.0e-8,
        "DELT": 0.25,
        "PHIEDGE": 1.0,
        "SIGNGS": -1,
    }
    scalars.update(values)
    return InData(scalars=scalars, indexed={})


def _state(ns: int, label: str = "state") -> SimpleNamespace:
    arr = np.zeros((int(ns), 1), dtype=float)
    return SimpleNamespace(
        label=label,
        layout=SimpleNamespace(ns=int(ns)),
        Rcos=arr,
        Rsin=arr,
        Zcos=arr,
        Zsin=arr,
        Lcos=arr,
        Lsin=arr,
    )


def _result(
    state,
    *,
    max_iter: int,
    fsq: float,
    converged: bool,
    use_scan: bool = False,
    diagnostics: dict | None = None,
) -> SolveVmecResidualResult:
    fsq_hist = np.full((max(1, int(max_iter)),), float(fsq), dtype=float)
    diag = {
        "converged": bool(converged),
        "ftol": 1.0e-8,
        "final_fsqr": float(fsq),
        "final_fsqz": 0.0,
        "final_fsql": 0.0,
        "use_scan": bool(use_scan),
        "resume_state": {"time_step": 0.25, "inv_tau": [1.0], "iter_offset": 1},
    }
    if diagnostics:
        diag.update(diagnostics)
    return SolveVmecResidualResult(
        state=state,
        n_iter=max(0, int(max_iter) - 1),
        w_history=fsq_hist,
        fsqr2_history=fsq_hist,
        fsqz2_history=np.zeros_like(fsq_hist),
        fsql2_history=np.zeros_like(fsq_hist),
        grad_rms_history=np.asarray([], dtype=float),
        step_history=np.asarray([], dtype=float),
        diagnostics=diag,
    )


def _install_light_driver(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cfg: VMECConfig,
    indata: InData,
    solver,
) -> None:
    monkeypatch.setenv("VMEC_JAX_COMPILATION_CACHE", "0")
    monkeypatch.setenv("VMEC_JAX_DISABLE_JIT_INIT", "1")
    monkeypatch.setattr(driver, "load_config", lambda _path: (cfg, indata))
    monkeypatch.setattr(driver, "_default_backend_name", lambda: "cpu")
    monkeypatch.setattr(driver, "validate_free_boundary_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(driver, "prepare_mgrid_for_config", lambda *_args, **_kwargs: None)

    def fake_static(cfg_in, **_kwargs):
        return SimpleNamespace(
            cfg=cfg_in,
            modes=SimpleNamespace(m=np.asarray([0]), n=np.asarray([0]), K=1),
            s=np.linspace(0.0, 1.0, int(cfg_in.ns)),
            grid=SimpleNamespace(theta=np.asarray([0.0]), zeta=np.asarray([0.0])),
            trig_vmec=None,
        )

    monkeypatch.setattr(driver, "build_static", fake_static)
    monkeypatch.setattr(driver, "boundary_from_indata", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(driver, "initial_guess_from_boundary", lambda static, *_args, **_kwargs: _state(static.cfg.ns))
    monkeypatch.setattr(driver, "interp_vmec_state", lambda *_args, ns_new, **_kwargs: _state(ns_new, f"interp-{ns_new}"))
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
    monkeypatch.setattr(driver, "eval_profiles", lambda _indata, s: {"pressure": np.zeros_like(np.asarray(s))})
    monkeypatch.setattr(driver, "_final_flux_profiles_from_state", lambda **kwargs: (kwargs["flux_local"], kwargs["prof_local"]))
    monkeypatch.setattr(driver, "solve_fixed_boundary_residual_iter", solver)


def test_cli_explicit_multigrid_niter_exhaustion_skips_finish_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[dict] = []

    def fake_solver(state, static, **kwargs):
        calls.append(
            {
                "ns": int(static.cfg.ns),
                "max_iter": int(kwargs["max_iter"]),
                "use_scan": bool(kwargs["use_scan"]),
            }
        )
        return _result(
            _state(static.cfg.ns, f"stage-{len(calls)}"),
            max_iter=int(kwargs["max_iter"]),
            fsq=1.0e-3,
            converged=False,
            use_scan=bool(kwargs["use_scan"]),
        )

    _install_light_driver(
        monkeypatch,
        cfg=_cfg(ns=5),
        indata=_indata(NS_ARRAY=[3, 5], NITER_ARRAY=[1, 1], FTOL_ARRAY=[1.0e-8, 1.0e-8]),
        solver=fake_solver,
    )

    run = driver.run_fixed_boundary(
        tmp_path / "input.niter_exhausted",
        solver="vmec2000_iter",
        solver_mode="parity",
        verbose=False,
        jit_forces=False,
        grid=object(),
        cli_fixed_boundary_mode=True,
    )

    assert calls == [
        {"ns": 3, "max_iter": 1, "use_scan": False},
        {"ns": 5, "max_iter": 1, "use_scan": False},
    ]
    diag = run.result.diagnostics
    assert diag["cli_fixed_boundary_initial_policy"] == "multigrid"
    assert diag["multigrid_final_stage_niter_exhausted"] is True
    assert np.asarray(diag["cli_fixed_boundary_finish_budgets"]).tolist() == []
    assert np.asarray(diag["cli_fixed_boundary_finish_modes"]).tolist() == []
    assert diag["cli_fixed_boundary_full_parity_fallback"] is False
    assert diag["converged"] is False


def test_cli_single_grid_finish_attempt_promotes_strict_accelerated_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[dict] = []

    def fake_solver(state, static, **kwargs):
        calls.append(
            {
                "max_iter": int(kwargs["max_iter"]),
                "use_scan": bool(kwargs["use_scan"]),
                "fsq_total_target": kwargs.get("fsq_total_target"),
                "resume_state_mode": kwargs.get("resume_state_mode"),
            }
        )
        if len(calls) == 1:
            return _result(
                _state(static.cfg.ns, "initial"),
                max_iter=int(kwargs["max_iter"]),
                fsq=1.0e-3,
                converged=False,
                use_scan=bool(kwargs["use_scan"]),
            )
        return _result(
            _state(static.cfg.ns, "finish"),
            max_iter=int(kwargs["max_iter"]),
            fsq=1.0e-10,
            converged=False,
            use_scan=bool(kwargs["use_scan"]),
        )

    _install_light_driver(monkeypatch, cfg=_cfg(ns=5), indata=_indata(NITER=4, FTOL=1.0e-8), solver=fake_solver)

    run = driver.run_fixed_boundary(
        tmp_path / "input.finish_promotes",
        solver="vmec2000_iter",
        solver_mode="accelerated",
        max_iter=4,
        verbose=False,
        multigrid=False,
        use_scan=False,
        jit_forces=False,
        grid=object(),
        cli_fixed_boundary_mode=True,
        finish_policy="converge",
        _auto_cli_fixed_boundary_mode=False,
    )

    assert calls == [
        {"max_iter": 4, "use_scan": False, "fsq_total_target": None, "resume_state_mode": "minimal"},
        {
            "max_iter": 4,
            "use_scan": False,
            "fsq_total_target": pytest.approx(driver._accelerated_fsq_total_target_from_ftol(1.0e-8)),
            "resume_state_mode": "minimal",
        },
    ]
    assert run.state.label == "finish"
    diag = run.result.diagnostics
    assert diag["cli_fixed_boundary_initial_policy"] == "single_grid"
    assert np.asarray(diag["cli_fixed_boundary_finish_budgets"]).tolist() == [4]
    assert np.asarray(diag["cli_fixed_boundary_finish_modes"]).tolist() == ["accelerated"]
    assert np.asarray(diag["cli_fixed_boundary_finish_converged"]).tolist() == [True]
    assert diag["cli_fixed_boundary_full_parity_fallback"] is False
    assert diag["converged"] is True
    assert diag["converged_strict"] is True


def test_finish_policy_none_suppresses_cli_finish_attempt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[dict] = []

    def fake_solver(state, static, **kwargs):
        calls.append(
            {
                "max_iter": int(kwargs["max_iter"]),
                "use_scan": bool(kwargs["use_scan"]),
                "resume_state_mode": kwargs.get("resume_state_mode"),
            }
        )
        return _result(
            _state(static.cfg.ns, "bounded"),
            max_iter=int(kwargs["max_iter"]),
            fsq=1.0e-3,
            converged=False,
            use_scan=bool(kwargs["use_scan"]),
        )

    _install_light_driver(monkeypatch, cfg=_cfg(ns=5), indata=_indata(NITER=4, FTOL=1.0e-8), solver=fake_solver)

    run = driver.run_fixed_boundary(
        tmp_path / "input.finish_none",
        solver="vmec2000_iter",
        solver_mode="accelerated",
        max_iter=4,
        verbose=False,
        multigrid=False,
        use_scan=False,
        jit_forces=False,
        grid=object(),
        cli_fixed_boundary_mode=True,
        finish_policy="none",
        _auto_cli_fixed_boundary_mode=False,
    )

    assert calls == [{"max_iter": 4, "use_scan": False, "resume_state_mode": "minimal"}]
    assert run.state.label == "bounded"
    diag = run.result.diagnostics
    assert diag["fixed_boundary_finish_policy"] == "none"
    assert diag["cli_fixed_boundary_finish_enabled"] is False
    assert "cli_fixed_boundary_finish_budgets" not in diag
    assert diag["converged"] is False


def test_scan_parity_guard_keeps_scan_when_probe_histories_match(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[dict] = []

    def fake_solver(state, static, **kwargs):
        calls.append({"max_iter": int(kwargs["max_iter"]), "use_scan": bool(kwargs["use_scan"])})
        return _result(
            state or _state(static.cfg.ns, "probe"),
            max_iter=int(kwargs["max_iter"]),
            fsq=1.0e-10,
            converged=True,
            use_scan=bool(kwargs["use_scan"]),
            diagnostics={"vmec2000_scan": bool(kwargs["use_scan"])},
        )

    monkeypatch.setenv("VMEC_JAX_LASYM_USE_SCAN", "1")
    monkeypatch.setenv("VMEC_JAX_SCAN_PARITY_GUARD", "1")
    monkeypatch.setenv("VMEC_JAX_DYNAMIC_SCAN", "0")
    _install_light_driver(monkeypatch, cfg=_cfg(ns=4, lasym=True), indata=_indata(LASYM=True, NITER=4), solver=fake_solver)

    run = driver.run_fixed_boundary(
        tmp_path / "input.scan_guard_pass",
        solver="vmec2000_iter",
        solver_mode="default",
        max_iter=4,
        verbose=False,
        multigrid=False,
        use_scan=True,
        jit_forces=False,
        grid=object(),
        _auto_cli_fixed_boundary_mode=False,
    )

    assert calls == [
        {"max_iter": 2, "use_scan": True},
        {"max_iter": 2, "use_scan": False},
        {"max_iter": 4, "use_scan": True},
    ]
    assert run.result.diagnostics["use_scan"] is True


def test_scan_parity_guard_probe_exception_falls_back_to_non_scan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[dict] = []

    def fake_solver(state, static, **kwargs):
        calls.append({"max_iter": int(kwargs["max_iter"]), "use_scan": bool(kwargs["use_scan"])})
        if len(calls) == 1:
            raise RuntimeError("synthetic guard probe failure")
        return _result(
            state or _state(static.cfg.ns, "after-probe"),
            max_iter=int(kwargs["max_iter"]),
            fsq=1.0e-10,
            converged=True,
            use_scan=bool(kwargs["use_scan"]),
            diagnostics={"vmec2000_scan": bool(kwargs["use_scan"])},
        )

    monkeypatch.setenv("VMEC_JAX_LASYM_USE_SCAN", "1")
    monkeypatch.setenv("VMEC_JAX_SCAN_PARITY_GUARD", "1")
    monkeypatch.setenv("VMEC_JAX_DYNAMIC_SCAN", "0")
    _install_light_driver(monkeypatch, cfg=_cfg(ns=4, lasym=True), indata=_indata(LASYM=True, NITER=3), solver=fake_solver)

    run = driver.run_fixed_boundary(
        tmp_path / "input.scan_guard_error",
        solver="vmec2000_iter",
        solver_mode="default",
        max_iter=3,
        verbose=True,
        multigrid=False,
        use_scan=True,
        jit_forces=False,
        grid=object(),
        _auto_cli_fixed_boundary_mode=False,
    )

    assert calls == [
        {"max_iter": 1, "use_scan": True},
        {"max_iter": 3, "use_scan": False},
    ]
    assert "scan parity guard probe failed (RuntimeError); using non-scan" in capsys.readouterr().out
    assert run.result.diagnostics["use_scan"] is False


def test_scan_wout_corrector_failure_leaves_original_scan_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[dict] = []

    def fake_solver(state, static, **kwargs):
        calls.append({"max_iter": int(kwargs["max_iter"]), "use_scan": bool(kwargs["use_scan"])})
        if len(calls) == 2:
            raise RuntimeError("synthetic corrector failure")
        return _result(
            _state(static.cfg.ns, "scan-result"),
            max_iter=int(kwargs["max_iter"]),
            fsq=1.0e-10,
            converged=True,
            use_scan=bool(kwargs["use_scan"]),
            diagnostics={"vmec2000_scan": bool(kwargs["use_scan"])},
        )

    _install_light_driver(monkeypatch, cfg=_cfg(ns=5), indata=_indata(NITER=2), solver=fake_solver)

    run = driver.run_fixed_boundary(
        tmp_path / "input.wout_corrector_failure",
        solver="vmec2000_iter",
        solver_mode="accelerated",
        max_iter=2,
        verbose=False,
        multigrid=False,
        scan_wout_corrector=True,
        grid=object(),
    )

    assert calls == [
        {"max_iter": 2, "use_scan": True},
        {"max_iter": 1, "use_scan": False},
    ]
    assert run.state.label == "scan-result"
    assert "scan_wout_corrector" not in run.result.diagnostics
    assert run.result.diagnostics["use_scan"] is True
