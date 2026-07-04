from __future__ import annotations

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
    raw_diagnostics=None,
    n_iter: int | None = None,
    w_history=None,
) -> SolveVmecResidualResult:
    if raw_diagnostics is None:
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
    else:
        diag = raw_diagnostics

    hist = np.full((max(1, int(max_iter)),), float(fsq), dtype=float) if w_history is None else w_history
    return SolveVmecResidualResult(
        state=state,
        n_iter=max(0, int(max_iter) - 1) if n_iter is None else int(n_iter),
        w_history=hist,
        fsqr2_history=np.full((max(1, int(max_iter)),), float(fsq), dtype=float),
        fsqz2_history=np.zeros((max(1, int(max_iter)),), dtype=float),
        fsql2_history=np.zeros((max(1, int(max_iter)),), dtype=float),
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


class _AlwaysRaisingGetDict(dict):
    def get(self, *_args, **_kwargs):
        raise RuntimeError("synthetic diagnostics get failure")


class _OneShotRaisingGetDict(dict):
    def __init__(self, *args, raise_key: str, **kwargs):
        super().__init__(*args, **kwargs)
        self._raise_key = str(raise_key)
        self._remaining = 1

    def get(self, key, default=None):
        if key == self._raise_key and self._remaining:
            self._remaining -= 1
            raise RuntimeError("synthetic one-shot diagnostics get failure")
        return super().get(key, default)


class _OneShotArray:
    def __init__(self, values):
        self._values = np.asarray(values, dtype=float)
        self._remaining = 1

    def __array__(self, dtype=None, copy=None):
        del copy
        if self._remaining:
            self._remaining -= 1
            raise RuntimeError("synthetic one-shot array failure")
        return np.asarray(self._values, dtype=dtype)


def test_cli_finish_marks_strict_residual_run_converged_without_retry(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    calls: list[dict] = []

    def fake_solver(state, static, **kwargs):
        calls.append({"max_iter": int(kwargs["max_iter"]), "use_scan": bool(kwargs["use_scan"])})
        return _result(
            _state(static.cfg.ns, "strict-with-false-flag"),
            max_iter=int(kwargs["max_iter"]),
            fsq=1.0e-10,
            converged=False,
            use_scan=bool(kwargs["use_scan"]),
            diagnostics={"ftol": float(kwargs["ftol"])},
        )

    _install_light_driver(monkeypatch, cfg=_cfg(ns=5), indata=_indata(NITER=3, FTOL=1.0e-8), solver=fake_solver)

    run = driver.run_fixed_boundary(
        tmp_path / "input.strict_false_flag",
        solver="vmec2000_iter",
        solver_mode="parity",
        max_iter=3,
        verbose=False,
        multigrid=False,
        cli_fixed_boundary_mode=True,
        jit_forces=True,
        grid=object(),
    )

    assert calls == [{"max_iter": 3, "use_scan": False}]
    diag = run.result.diagnostics
    assert diag["converged"] is True
    assert diag["converged_strict"] is True
    assert np.asarray(diag["cli_fixed_boundary_finish_budgets"]).tolist() == []
    assert np.asarray(diag["cli_fixed_boundary_finish_modes"]).tolist() == []
    assert diag["cli_fixed_boundary_full_parity_fallback"] is False


def test_accelerated_multigrid_partial_restart_exception_allows_full_fallback_replacement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    calls: list[dict] = []

    def fake_solver(state, static, **kwargs):
        idx = len(calls)
        calls.append({"ns": int(static.cfg.ns), "max_iter": int(kwargs["max_iter"]), "use_scan": bool(kwargs["use_scan"])})
        fsq = 1.0e-10 if idx == 5 else 1.0e-3
        converged = idx == 5
        diagnostics = {
            "converged": bool(converged),
            "ftol": float(kwargs["ftol"]),
            "final_fsqr": float(fsq),
            "final_fsqz": 0.0,
            "final_fsql": 0.0,
            "use_scan": bool(kwargs["use_scan"]),
            "resume_state": {"time_step": 0.25, "iter_offset": idx},
        }
        raw_diagnostics = _AlwaysRaisingGetDict(diagnostics) if idx == 1 else diagnostics
        return _result(
            _state(static.cfg.ns, f"solve-{idx}"),
            max_iter=int(kwargs["max_iter"]),
            fsq=fsq,
            converged=converged,
            raw_diagnostics=raw_diagnostics,
        )

    _install_light_driver(
        monkeypatch,
        cfg=_cfg(ns=9, lthreed=True),
        indata=_indata(
            NITER=3,
            FTOL=1.0e-8,
            NS_ARRAY=[3, 5, 9],
            NITER_ARRAY=[1, 1, 1],
            FTOL_ARRAY=[1.0e-8, 1.0e-8, 1.0e-8],
        ),
        solver=fake_solver,
    )

    run = driver.run_fixed_boundary(
        tmp_path / "input.partial_restart_failure",
        solver="vmec2000_iter",
        solver_mode="accelerated",
        verbose=False,
        cli_fixed_boundary_mode=True,
        jit_forces=True,
        grid=object(),
        finish_policy="converge",
    )

    assert calls == [
        {"ns": 3, "max_iter": 1, "use_scan": True},
        {"ns": 5, "max_iter": 1, "use_scan": True},
        {"ns": 9, "max_iter": 1, "use_scan": True},
        {"ns": 3, "max_iter": 1, "use_scan": False},
        {"ns": 5, "max_iter": 1, "use_scan": False},
        {"ns": 9, "max_iter": 1, "use_scan": False},
    ]
    assert run.state.label == "solve-5"
    diag = run.result.diagnostics
    assert diag["cli_fixed_boundary_partial_parity_fallback"] is False
    assert diag["cli_fixed_boundary_full_parity_fallback"] is True
    assert diag["converged"] is True


@pytest.mark.parametrize(
    ("solver_mode", "force_scan_env", "scan_jit_env", "expected_jit"),
    [
        ("parity", "1", None, False),
        ("default", None, "0", False),
        ("default", None, "1", True),
    ],
)
def test_scan_stage_jit_policy_env_branches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    solver_mode: str,
    force_scan_env: str | None,
    scan_jit_env: str | None,
    expected_jit: bool,
) -> None:
    calls: list[dict] = []

    def fake_solver(state, static, **kwargs):
        calls.append(
            {
                "jit_forces": bool(kwargs["jit_forces"]),
                "jit_precompile": bool(kwargs["jit_precompile"]),
                "jit_warmup_iters": int(kwargs["jit_warmup_iters"]),
                "use_scan": bool(kwargs["use_scan"]),
            }
        )
        return _result(
            state,
            max_iter=int(kwargs["max_iter"]),
            fsq=1.0e-10,
            converged=True,
            use_scan=bool(kwargs["use_scan"]),
            diagnostics={"ftol": float(kwargs["ftol"])},
        )

    if force_scan_env is not None:
        monkeypatch.setenv("VMEC_JAX_USE_SCAN", force_scan_env)
    if scan_jit_env is not None:
        monkeypatch.setenv("VMEC_JAX_SCAN_JIT_FORCES", scan_jit_env)
    monkeypatch.setenv("VMEC_JAX_LASYM_USE_SCAN", "1")
    monkeypatch.setenv("VMEC_JAX_DYNAMIC_SCAN", "0")
    _install_light_driver(monkeypatch, cfg=_cfg(ns=5, lasym=True), indata=_indata(NITER=2, LASYM=True), solver=fake_solver)

    driver.run_fixed_boundary(
        tmp_path / f"input.scan_jit_{solver_mode}_{scan_jit_env}",
        solver="vmec2000_iter",
        solver_mode=solver_mode,
        max_iter=2,
        verbose=False,
        multigrid=False,
        use_scan=True,
        jit_forces=True,
        grid=object(),
        _auto_cli_fixed_boundary_mode=False,
    )

    assert calls == [
        {
            "jit_forces": expected_jit,
            "jit_precompile": False,
            "jit_warmup_iters": 0,
            "use_scan": True,
        }
    ]


def test_non_scan_jit_invalid_warmup_env_defaults_to_two(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    calls: list[dict] = []

    def fake_solver(state, static, **kwargs):
        del static
        calls.append(
            {
                "jit_forces": bool(kwargs["jit_forces"]),
                "jit_precompile": bool(kwargs["jit_precompile"]),
                "jit_warmup_iters": int(kwargs["jit_warmup_iters"]),
                "use_scan": bool(kwargs["use_scan"]),
            }
        )
        return _result(
            state,
            max_iter=int(kwargs["max_iter"]),
            fsq=1.0e-10,
            converged=True,
            diagnostics={"ftol": float(kwargs["ftol"])},
        )

    monkeypatch.setenv("VMEC_JAX_JIT_WARMUP_ITERS", "not-an-int")
    _install_light_driver(monkeypatch, cfg=_cfg(ns=5), indata=_indata(NITER=2), solver=fake_solver)

    driver.run_fixed_boundary(
        tmp_path / "input.invalid_warmup_env",
        solver="vmec2000_iter",
        solver_mode="default",
        max_iter=2,
        verbose=False,
        multigrid=False,
        use_scan=False,
        jit_forces=True,
        grid=object(),
        _auto_cli_fixed_boundary_mode=False,
    )

    assert calls == [
        {
            "jit_forces": True,
            "jit_precompile": True,
            "jit_warmup_iters": 2,
            "use_scan": False,
        }
    ]


def test_dynamic_scan_timed_probe_warms_and_times_both_paths(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    calls: list[dict] = []

    def fake_solver(state, static, **kwargs):
        del static
        calls.append({"max_iter": int(kwargs["max_iter"]), "use_scan": bool(kwargs["use_scan"])})
        return _result(
            state,
            max_iter=int(kwargs["max_iter"]),
            fsq=1.0e-10,
            converged=True,
            use_scan=bool(kwargs["use_scan"]),
            diagnostics={"ftol": float(kwargs["ftol"]), "vmec2000_scan": bool(kwargs["use_scan"])},
        )

    times = [0.0, 10.0, 12.0, 20.0, 20.5]

    def fake_perf_counter():
        return times.pop(0) if times else 21.0

    monkeypatch.setenv("VMEC_JAX_DYNAMIC_SCAN", "1")
    monkeypatch.setenv("VMEC_JAX_DYNAMIC_SCAN_TIMED", "1")
    monkeypatch.setenv("VMEC_JAX_DYNAMIC_SCAN_ITERS", "2")
    monkeypatch.setenv("VMEC_JAX_LASYM_USE_SCAN", "1")
    monkeypatch.setattr(driver.time, "perf_counter", fake_perf_counter)
    _install_light_driver(monkeypatch, cfg=_cfg(ns=5, lasym=True), indata=_indata(NITER=4, LASYM=True), solver=fake_solver)

    run = driver.run_fixed_boundary(
        tmp_path / "input.dynamic_timed_probe",
        solver="vmec2000_iter",
        solver_mode="default",
        performance_mode=True,
        max_iter=4,
        verbose=False,
        multigrid=False,
        use_scan=True,
        jit_forces=True,
        grid=object(),
        _auto_cli_fixed_boundary_mode=False,
    )

    assert calls[0] == {"max_iter": 4, "use_scan": True}
    assert calls.count({"max_iter": 2, "use_scan": False}) == 2
    assert calls.count({"max_iter": 2, "use_scan": True}) >= 2
    assert calls[-1] == {"max_iter": 4, "use_scan": True}
    assert run.result.diagnostics["use_scan"] is True


def test_explicit_stage_monitor_runs_tail_when_chunk_can_still_meet_budget(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    calls: list[dict] = []

    def fake_solver(state, static, **kwargs):
        idx = len(calls)
        calls.append({"ns": int(static.cfg.ns), "max_iter": int(kwargs["max_iter"]), "use_scan": bool(kwargs["use_scan"])})
        if idx == 1:
            return SolveVmecResidualResult(
                state=_state(static.cfg.ns, "chunk"),
                n_iter=199,
                w_history=np.asarray([1.0e-12, 2.0e-14], dtype=float),
                fsqr2_history=np.asarray([1.0e-12, 2.0e-14], dtype=float),
                fsqz2_history=np.asarray([0.0, 0.0], dtype=float),
                fsql2_history=np.asarray([0.0, 0.0], dtype=float),
                grad_rms_history=np.asarray([], dtype=float),
                step_history=np.asarray([], dtype=float),
                diagnostics={
                    "converged": False,
                    "ftol": float(kwargs["ftol"]),
                    "final_fsqr": 2.0e-14,
                    "final_fsqz": 0.0,
                    "final_fsql": 0.0,
                    "resume_state": {"time_step": 0.2, "iter_offset": 200},
                },
            )
        fsq = 1.0e-16 if idx == 2 else 1.0e-3
        return _result(
            _state(static.cfg.ns, f"stage-{idx}"),
            max_iter=int(kwargs["max_iter"]),
            fsq=fsq,
            converged=idx == 2,
            use_scan=bool(kwargs["use_scan"]),
            diagnostics={"ftol": float(kwargs["ftol"])},
        )

    _install_light_driver(
        monkeypatch,
        cfg=_cfg(ns=9),
            indata=_indata(
                NITER=251,
                FTOL=1.0e-14,
                NSTEP=200,
                NS_ARRAY=[5, 9],
                NITER_ARRAY=[1, 250],
                FTOL_ARRAY=[1.0e-14, 1.0e-14],
            ),
        solver=fake_solver,
    )

    run = driver.run_fixed_boundary(
        tmp_path / "input.stage_monitor_tail",
        solver="vmec2000_iter",
        solver_mode="accelerated",
        verbose=False,
        cli_fixed_boundary_mode=True,
        jit_forces=True,
        grid=object(),
        use_scan=False,
        stage_transition_heuristic=True,
        finish_policy="converge",
    )

    assert calls == [
        {"ns": 5, "max_iter": 1, "use_scan": False},
        {"ns": 9, "max_iter": 200, "use_scan": False},
        {"ns": 9, "max_iter": 50, "use_scan": False},
    ]
    diag = run.result.diagnostics
    assert diag["accelerated_stage_chunked"] is True
    assert diag["accelerated_stage_effective_mode"] == "accelerated"
    np.testing.assert_array_equal(diag["accelerated_stage_chunk_iters"], [200, 50])
    assert diag["converged"] is True


def test_scan_abort_guard_exceptions_and_resume_sanitization_are_nonfatal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    calls: list[dict] = []

    def fake_solver(state, static, **kwargs):
        idx = len(calls)
        calls.append(
            {
                "ns": int(static.cfg.ns),
                "resume_state": kwargs.get("resume_state"),
                "use_scan": bool(kwargs["use_scan"]),
            }
        )
        if idx == 0:
            diagnostics = _OneShotRaisingGetDict(
                {
                    "converged": False,
                    "ftol": float(kwargs["ftol"]),
                    "final_fsqr": 1.0e-3,
                    "final_fsqz": 0.0,
                    "final_fsql": 0.0,
                    "use_scan": True,
                    "vmec2000_scan": True,
                    "abort_scan": True,
                    "resume_state": {"time_step": 0.5, "flip_sign": -1.0, "iter_offset": 7},
                },
                raise_key="vmec2000_scan",
            )
            return _result(
                _state(static.cfg.ns, "first"),
                max_iter=int(kwargs["max_iter"]),
                fsq=1.0e-3,
                converged=False,
                raw_diagnostics=diagnostics,
                w_history=_OneShotArray([1.0e-3]),
            )
        return _result(
            _state(static.cfg.ns, "second"),
            max_iter=int(kwargs["max_iter"]),
            fsq=1.0e-10,
            converged=True,
            use_scan=bool(kwargs["use_scan"]),
            diagnostics={"ftol": float(kwargs["ftol"])},
        )

    monkeypatch.setenv("VMEC_JAX_MULTIGRID_RESUME", "1")
    monkeypatch.setenv("VMEC_JAX_LASYM_USE_SCAN", "1")
    _install_light_driver(
        monkeypatch,
        cfg=_cfg(ns=5, lasym=True),
        indata=_indata(NITER=2, FTOL=1.0e-8, LASYM=True, NS_ARRAY=[3, 5], NITER_ARRAY=[1, 1]),
        solver=fake_solver,
    )

    run = driver.run_fixed_boundary(
        tmp_path / "input.scan_abort_exception",
        solver="vmec2000_iter",
        solver_mode="default",
        verbose=False,
        multigrid=True,
        use_scan=True,
        jit_forces=True,
        grid=object(),
        _auto_cli_fixed_boundary_mode=False,
    )

    assert calls[0] == {"ns": 3, "resume_state": None, "use_scan": True}
    assert calls[1]["ns"] == 5
    assert calls[1]["use_scan"] is True
    assert calls[1]["resume_state"]["time_step"] == pytest.approx(0.25)
    assert calls[1]["resume_state"]["iter_offset"] == 0
    assert run.result.diagnostics["converged"] is True
