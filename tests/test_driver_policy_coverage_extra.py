from __future__ import annotations

import sys
from contextlib import contextmanager
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import vmec_jax.cli as cli
import vmec_jax.driver as driver
from vmec_jax.config import FreeBoundaryConfig, VMECConfig
from vmec_jax.energy import FluxProfiles
from vmec_jax.namelist import InData
from vmec_jax.solve import SolveVmecResidualResult


def _cfg(
    *,
    ns: int = 5,
    lfreeb: bool = False,
    lthreed: bool = False,
    lasym: bool = False,
    ntheta: int = 4,
    nzeta: int | None = None,
) -> VMECConfig:
    return VMECConfig(
        mpol=2,
        ntor=1 if lthreed else 0,
        ns=int(ns),
        nfp=1,
        lasym=bool(lasym),
        lthreed=bool(lthreed),
        lconm1=True,
        ntheta=int(ntheta),
        nzeta=int(nzeta if nzeta is not None else (2 if lthreed or lasym else 1)),
        free_boundary=FreeBoundaryConfig(
            enabled=bool(lfreeb),
            mgrid_file="mgrid.synthetic.nc" if lfreeb else "NONE",
            extcur=(1.0,) if lfreeb else (),
            nvacskip=1,
        ),
    )


def _indata(**values) -> InData:
    scalars = {
        "LFREEB": False,
        "NITER": 2,
        "FTOL": 1.0e-10,
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
    fsq: float,
    converged: bool = False,
    max_iter: int = 1,
    diagnostics: dict | None = None,
) -> SolveVmecResidualResult:
    diag = {
        "converged": bool(converged),
        "ftol": 1.0e-10,
        "final_fsqr": float(fsq),
        "final_fsqz": 0.0,
        "final_fsql": 0.0,
        "resume_state": {"time_step": 0.25, "inv_tau": [1.0], "iter_offset": 1},
    }
    if diagnostics:
        diag.update(diagnostics)
    return SolveVmecResidualResult(
        state=state,
        n_iter=max(0, int(max_iter) - 1),
        w_history=np.asarray([float(fsq)], dtype=float),
        fsqr2_history=np.asarray([float(fsq)], dtype=float),
        fsqz2_history=np.asarray([0.0], dtype=float),
        fsql2_history=np.asarray([0.0], dtype=float),
        grad_rms_history=np.asarray([], dtype=float),
        step_history=np.asarray([], dtype=float),
        diagnostics=diag,
    )


def _install_driver_fakes(monkeypatch: pytest.MonkeyPatch, *, cfg: VMECConfig, indata: InData) -> dict[str, list]:
    calls: dict[str, list] = {"initial_guess": [], "interp": [], "build_static": []}
    monkeypatch.setenv("VMEC_JAX_COMPILATION_CACHE", "0")
    monkeypatch.setattr(driver, "load_config", lambda _path: (cfg, indata))
    monkeypatch.setattr(driver, "_default_backend_name", lambda: "cpu")
    monkeypatch.setattr(driver, "validate_free_boundary_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(driver, "prepare_mgrid_for_config", lambda *_args, **_kwargs: None)

    def fake_static(cfg_in, **_kwargs):
        calls["build_static"].append(cfg_in)
        return SimpleNamespace(
            cfg=cfg_in,
            modes=SimpleNamespace(
                m=np.zeros(1, dtype=int),
                n=np.zeros(1, dtype=int),
                K=1,
            ),
            s=np.linspace(0.0, 1.0, int(cfg_in.ns)),
            grid=SimpleNamespace(theta=np.asarray([0.0]), zeta=np.asarray([0.0])),
            trig_vmec=None,
        )

    def fake_initial_guess(static, *_args, **kwargs):
        calls["initial_guess"].append(kwargs)
        return _state(static.cfg.ns, "initial")

    def fake_interp(state, **kwargs):
        calls["interp"].append((state.layout.ns, int(kwargs["ns_new"])))
        return _state(int(kwargs["ns_new"]), f"interp-{kwargs['ns_new']}")

    monkeypatch.setattr(driver, "build_static", fake_static)
    monkeypatch.setattr(driver, "boundary_from_indata", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(driver, "initial_guess_from_boundary", fake_initial_guess)
    monkeypatch.setattr(driver, "interp_vmec_state", fake_interp)
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
    monkeypatch.setattr(
        driver,
        "_final_flux_profiles_from_state",
        lambda **kwargs: (kwargs["flux_local"], kwargs["prof_local"]),
    )
    return calls


def test_solver_policy_helper_edges_and_stage_switch_reasons() -> None:
    assert driver._resolve_fixed_boundary_solver_device_name(
        solver_device=" none ",
        backend="gpu",
        cfg=_cfg(),
        indata=_indata(),
        solver_lower="vmec2000_iter",
        cli_fixed_boundary_mode=True,
        accelerated_mode=True,
        ns_list_input=None,
        niter_list_input=None,
        restart_state_present=False,
        restart_solver_state_present=False,
    ) is None
    assert driver._resolve_fixed_boundary_solver_device_name(
        solver_device="tpu",
        backend="cpu",
        cfg=_cfg(),
        indata=_indata(),
        solver_lower="vmec2000_iter",
        cli_fixed_boundary_mode=False,
        accelerated_mode=False,
        ns_list_input=None,
        niter_list_input=None,
        restart_state_present=True,
        restart_solver_state_present=True,
    ) == "tpu"
    assert driver._default_non_autodiff_solver_policy_for_backend(_indata(LFREEB=True), "cpu") == ("default", True)
    assert driver._default_non_autodiff_solver_policy_for_backend(_indata(NS_ARRAY=[3, 5]), "cpu") == (
        "parity",
        False,
    )
    assert driver._default_non_autodiff_solver_policy_for_backend(_indata(LASYM=True), "cpu") == (
        "accelerated",
        True,
    )
    assert driver._default_non_autodiff_solver_policy_for_backend(
        _indata(NCURR=1, NS_ARRAY=[3, 5], NITER_ARRAY=[1, 1]), "cpu"
    ) == ("accelerated", True)
    assert driver._default_non_autodiff_solver_policy_for_backend(_indata(), "gpu") == ("accelerated", True)
    assert driver._stage_switch_reason_from_progress(
        start_total_fsq=1.0,
        best_total_fsq=float("nan"),
        target_total_fsq=1.0e-3,
        chunk_iters=1,
        remaining_budget=5,
    ) == "nonfinite_total_fsq"
    assert driver._stage_switch_reason_from_progress(
        start_total_fsq=1.0,
        best_total_fsq=1.0,
        target_total_fsq=1.0e-3,
        chunk_iters=1,
        remaining_budget=5,
    ) == "nondecreasing_total_fsq"
    assert driver._stage_switch_reason_from_progress(
        start_total_fsq=1.0,
        best_total_fsq=0.0,
        target_total_fsq=1.0e-3,
        chunk_iters=1,
        remaining_budget=5,
    ) is None


def test_cli_preflight_errors_before_driver_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    missing = tmp_path / "input.missing"
    with pytest.raises(SystemExit):
        cli.main([str(missing)])

    bad = tmp_path / "input.bad"
    bad.write_text("&INDATA\n/\n")
    monkeypatch.setattr(cli, "read_indata", lambda _path: (_ for _ in ()).throw(RuntimeError("synthetic read failure")))
    with pytest.raises(SystemExit):
        cli.main([str(bad)])

    with pytest.raises(SystemExit):
        cli.main(["--plot", str(tmp_path / "wout_missing.nc")])


def test_run_fixed_boundary_nojit_initial_guess_falls_back_when_jax_context_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _cfg(ns=3)
    indata = _indata(NS=3)
    calls = _install_driver_fakes(monkeypatch, cfg=cfg, indata=indata)
    monkeypatch.setenv("VMEC_JAX_DISABLE_JIT_INIT", "1")

    class FakeJax:
        @staticmethod
        def disable_jit():
            raise RuntimeError("disable_jit unavailable")

    monkeypatch.setitem(sys.modules, "jax", FakeJax)

    run = driver.run_fixed_boundary(
        tmp_path / "input.nojit_init",
        use_initial_guess=True,
        verbose=False,
        grid=object(),
        _auto_cli_fixed_boundary_mode=False,
    )

    assert run.result is None
    assert run.state.label == "initial"
    assert calls["initial_guess"] == [
        {"vmec_project": True, "infer_axis_if_missing": True},
    ]


def test_vmec2000_jit_overrides_and_precompile_stage_call(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = _cfg(ns=3)
    indata = _indata(NITER=2)
    _install_driver_fakes(monkeypatch, cfg=cfg, indata=indata)
    monkeypatch.setenv("VMEC_JAX_DISABLE_JIT_INIT", "1")
    monkeypatch.setenv("VMEC_JAX_VMEC2000_FORCE_JIT", "1")
    monkeypatch.setenv("VMEC_JAX_PRECOMPILE_STAGES", "1")
    calls: list[dict] = []

    def fake_solver(state, static, **kwargs):
        calls.append(
            {
                "precompile_only": bool(kwargs.get("precompile_only", False)),
                "jit_forces": bool(kwargs["jit_forces"]),
                "jit_precompile": bool(kwargs["jit_precompile"]),
                "max_iter": int(kwargs["max_iter"]),
            }
        )
        if kwargs.get("precompile_only", False):
            raise RuntimeError("synthetic precompile failure")
        return _result(state, fsq=1.0e-12, converged=True, max_iter=int(kwargs["max_iter"]))

    monkeypatch.setattr(driver, "solve_fixed_boundary_residual_iter", fake_solver)

    run = driver.run_fixed_boundary(
        tmp_path / "input.precompile",
        solver="vmec2000_iter",
        solver_mode="parity",
        max_iter=2,
        verbose=False,
        multigrid=False,
        jit_forces=False,
        grid=object(),
        _auto_cli_fixed_boundary_mode=False,
    )

    assert run.result.diagnostics["converged"] is True
    assert calls == [
        {"precompile_only": True, "jit_forces": True, "jit_precompile": True, "max_iter": 1},
        {"precompile_only": False, "jit_forces": True, "jit_precompile": True, "max_iter": 2},
    ]


def test_scan_guard_mismatch_and_abort_fallback_use_non_scan_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = _cfg(ns=4, lasym=True)
    indata = _indata(NITER=3, LASYM=True)
    _install_driver_fakes(monkeypatch, cfg=cfg, indata=indata)
    monkeypatch.setenv("VMEC_JAX_DISABLE_JIT_INIT", "1")
    monkeypatch.setenv("VMEC_JAX_LASYM_USE_SCAN", "1")
    monkeypatch.setenv("VMEC_JAX_SCAN_PARITY_GUARD", "1")
    calls: list[dict] = []

    def fake_solver(state, static, **kwargs):
        del static
        use_scan = bool(kwargs["use_scan"])
        calls.append({"max_iter": int(kwargs["max_iter"]), "use_scan": use_scan})
        fsq = 0.9 if use_scan and len(calls) == 1 else 0.1
        return _result(
            state,
            fsq=fsq,
            converged=True,
            max_iter=int(kwargs["max_iter"]),
            diagnostics={"use_scan": use_scan, "vmec2000_scan": use_scan},
        )

    monkeypatch.setattr(driver, "solve_fixed_boundary_residual_iter", fake_solver)

    run = driver.run_fixed_boundary(
        tmp_path / "input.scan_guard",
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
        {"max_iter": 3, "use_scan": True},
        {"max_iter": 3, "use_scan": False},
        {"max_iter": 3, "use_scan": False},
    ]
    assert "scan parity guard: disabling scan" in capsys.readouterr().out
    assert run.result.diagnostics["use_scan"] is False

    calls.clear()

    def fake_abort_solver(state, static, **kwargs):
        del static
        use_scan = bool(kwargs["use_scan"])
        calls.append({"max_iter": int(kwargs["max_iter"]), "use_scan": use_scan})
        return _result(
            state,
            fsq=1.0e-12,
            converged=True,
            max_iter=int(kwargs["max_iter"]),
            diagnostics={
                "use_scan": use_scan,
                "vmec2000_scan": use_scan,
                "abort_scan": use_scan,
            },
        )

    monkeypatch.setenv("VMEC_JAX_SCAN_PARITY_GUARD", "0")
    monkeypatch.setenv("VMEC_JAX_DYNAMIC_SCAN", "0")
    monkeypatch.setattr(driver, "solve_fixed_boundary_residual_iter", fake_abort_solver)

    run = driver.run_fixed_boundary(
        tmp_path / "input.scan_abort",
        solver="vmec2000_iter",
        solver_mode="default",
        max_iter=2,
        verbose=True,
        multigrid=False,
        use_scan=True,
        jit_forces=False,
        grid=object(),
        _auto_cli_fixed_boundary_mode=False,
    )

    assert calls == [
        {"max_iter": 2, "use_scan": True},
        {"max_iter": 2, "use_scan": False},
    ]
    assert "scan abort detected; rerunning stage in parity mode" in capsys.readouterr().out
    assert run.result.diagnostics["use_scan"] is False


def test_cli_finish_full_parity_fallback_can_replace_best_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _cfg(ns=5)
    indata = _indata(NITER=1, NS_ARRAY=[3, 5])
    _install_driver_fakes(monkeypatch, cfg=cfg, indata=indata)
    monkeypatch.setenv("VMEC_JAX_DISABLE_JIT_INIT", "1")
    calls: list[dict] = []

    def fake_solver(state, static, **kwargs):
        idx = len(calls)
        calls.append(
            {
                "ns": int(static.cfg.ns),
                "max_iter": int(kwargs["max_iter"]),
                "use_scan": bool(kwargs["use_scan"]),
            }
        )
        if idx == 3:
            return _result(
                _state(static.cfg.ns, "full-fallback"),
                fsq=1.0e-12,
                converged=True,
                max_iter=int(kwargs["max_iter"]),
                diagnostics={"ftol": float(kwargs["ftol"])},
            )
        return _result(
            _state(static.cfg.ns, f"miss-{idx}"),
            fsq=1.0e-3,
            converged=False,
            max_iter=int(kwargs["max_iter"]),
            diagnostics={"ftol": float(kwargs["ftol"])},
        )

    monkeypatch.setattr(driver, "solve_fixed_boundary_residual_iter", fake_solver)

    run = driver.run_fixed_boundary(
        tmp_path / "input.full_fallback",
        solver="vmec2000_iter",
        solver_mode="accelerated",
        max_iter=1,
        verbose=False,
        multigrid=False,
        cli_fixed_boundary_mode=True,
        jit_forces=False,
        grid=object(),
    )

    assert calls == [
        {"ns": 5, "max_iter": 1, "use_scan": True},
        {"ns": 5, "max_iter": 1, "use_scan": True},
        {"ns": 5, "max_iter": 1, "use_scan": False},
        {"ns": 5, "max_iter": 2, "use_scan": False},
    ]
    assert run.state.label == "full-fallback"
    diag = run.result.diagnostics
    assert diag["cli_fixed_boundary_full_parity_fallback"] is True
    assert diag["converged"] is True


def test_force_nojit_uses_disable_jit_context_for_stage_solve(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _cfg(ns=3)
    indata = _indata(NITER=1)
    _install_driver_fakes(monkeypatch, cfg=cfg, indata=indata)
    monkeypatch.setenv("VMEC_JAX_DISABLE_JIT_INIT", "1")
    monkeypatch.setenv("VMEC_JAX_VMEC2000_FORCE_NOJIT", "1")
    contexts: list[str] = []
    calls: list[bool] = []

    class FakeJax:
        @staticmethod
        @contextmanager
        def disable_jit():
            contexts.append("disable_jit")
            yield

    monkeypatch.setitem(sys.modules, "jax", FakeJax)

    def fake_solver(state, static, **kwargs):
        del static
        calls.append(bool(kwargs["jit_forces"]))
        return _result(state, fsq=1.0e-12, converged=True, max_iter=int(kwargs["max_iter"]))

    monkeypatch.setattr(driver, "solve_fixed_boundary_residual_iter", fake_solver)

    run = driver.run_fixed_boundary(
        tmp_path / "input.nojit_stage",
        solver="vmec2000_iter",
        solver_mode="default",
        max_iter=1,
        verbose=False,
        multigrid=False,
        jit_forces=True,
        use_scan=False,
        grid=object(),
        _auto_cli_fixed_boundary_mode=False,
    )

    assert run.result.diagnostics["converged"] is True
    assert contexts == ["disable_jit"]
    assert calls == [False]


def test_compilation_cache_primary_and_tmp_mkdir_failures_are_ignored(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _cfg(ns=3)
    indata = _indata(NS=3)
    _install_driver_fakes(monkeypatch, cfg=cfg, indata=indata)
    monkeypatch.setenv("VMEC_JAX_COMPILATION_CACHE", "1")
    monkeypatch.delenv("VMEC_JAX_DISABLE_COMPILATION_CACHE", raising=False)
    monkeypatch.setattr("vmec_jax._compat._default_compilation_cache_dir", lambda: tmp_path / "cache")

    mkdir_calls: list[str] = []
    cache_dirs: list[str] = []
    updates: list[tuple] = []

    class FailingPath:
        def __init__(self, value):
            self.value = str(value)

        def mkdir(self, **_kwargs):
            mkdir_calls.append(self.value)
            raise OSError("synthetic unwritable cache")

    jax_module = type(sys)("jax")
    jax_module.config = SimpleNamespace(update=lambda *args: updates.append(args))
    experimental_module = type(sys)("jax.experimental")
    cache_module = type(sys)("jax.experimental.compilation_cache")
    cache_module.compilation_cache = SimpleNamespace(set_cache_dir=lambda path: cache_dirs.append(path))
    monkeypatch.setitem(sys.modules, "jax", jax_module)
    monkeypatch.setitem(sys.modules, "jax.experimental", experimental_module)
    monkeypatch.setitem(sys.modules, "jax.experimental.compilation_cache", cache_module)
    monkeypatch.setattr(driver, "Path", FailingPath)

    run = driver.run_fixed_boundary(
        tmp_path / "input.cache_fallback_failure",
        use_initial_guess=True,
        verbose=False,
        grid=object(),
        _auto_cli_fixed_boundary_mode=False,
    )

    assert run.result is None
    assert mkdir_calls == [str(tmp_path / "cache"), "/tmp/vmec_jax/jax_compilation_cache"]
    assert cache_dirs == []
    assert updates == []


def test_gpu_solver_device_enables_default_compilation_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _cfg(ns=3)
    indata = _indata(NS=3)
    _install_driver_fakes(monkeypatch, cfg=cfg, indata=indata)
    monkeypatch.delenv("VMEC_JAX_COMPILATION_CACHE", raising=False)
    monkeypatch.delenv("VMEC_JAX_COMPILATION_CACHE_DIR", raising=False)
    monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR", raising=False)
    monkeypatch.delenv("VMEC_JAX_DISABLE_COMPILATION_CACHE", raising=False)

    cache_calls: list[str | None] = []

    def fake_default_cache_dir():
        cache_calls.append(os.environ.get("VMEC_JAX_COMPILATION_CACHE"))
        if os.environ.get("VMEC_JAX_COMPILATION_CACHE") == "1":
            return tmp_path / "gpu-cache"
        return None

    monkeypatch.setattr("vmec_jax._compat._default_compilation_cache_dir", fake_default_cache_dir)

    cache_dirs: list[str] = []
    updates: list[tuple] = []

    class FakeJaxModule:
        config = SimpleNamespace(update=lambda *args: updates.append(args))

        @staticmethod
        def devices(kind=None):
            return [f"{kind or 'gpu'}:0"]

        @staticmethod
        @contextmanager
        def default_device(_device):
            yield

    experimental_module = type(sys)("jax.experimental")
    cache_module = type(sys)("jax.experimental.compilation_cache")
    cache_module.compilation_cache = SimpleNamespace(set_cache_dir=lambda path: cache_dirs.append(path))
    monkeypatch.setitem(sys.modules, "jax", FakeJaxModule)
    monkeypatch.setitem(sys.modules, "jax.experimental", experimental_module)
    monkeypatch.setitem(sys.modules, "jax.experimental.compilation_cache", cache_module)

    monkeypatch.setattr(
        driver,
        "solve_fixed_boundary_residual_iter",
        lambda state, static, **_kwargs: _result(state, fsq=0.0, converged=True),
    )

    driver.run_fixed_boundary(
        tmp_path / "input.gpu_cache",
        use_initial_guess=True,
        verbose=False,
        grid=object(),
        solver_device="gpu",
        _auto_cli_fixed_boundary_mode=False,
    )

    assert cache_calls == [None, "1"]
    assert os.environ.get("VMEC_JAX_COMPILATION_CACHE") is None
    assert cache_dirs == [str(tmp_path / "gpu-cache")]
    assert ("jax_enable_compilation_cache", True) in updates
    assert ("jax_compilation_cache_dir", str(tmp_path / "gpu-cache")) in updates
    assert ("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir") in updates


def test_dynamic_scan_invalid_env_values_fall_back_and_keep_scan_when_histories_match(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = _cfg(ns=4, lasym=True)
    indata = _indata(NITER=4, LASYM=True)
    _install_driver_fakes(monkeypatch, cfg=cfg, indata=indata)
    monkeypatch.setenv("VMEC_JAX_DISABLE_JIT_INIT", "1")
    monkeypatch.setenv("VMEC_JAX_LASYM_USE_SCAN", "1")
    monkeypatch.setenv("VMEC_JAX_SCAN_PARITY_GUARD", "0")
    monkeypatch.setenv("VMEC_JAX_DYNAMIC_SCAN", "1")
    monkeypatch.setenv("VMEC_JAX_DYNAMIC_SCAN_TIMED", "0")
    monkeypatch.setenv("VMEC_JAX_DYNAMIC_SCAN_ITERS", "bad-int")
    monkeypatch.setenv("VMEC_JAX_DYNAMIC_SCAN_FSQ_RTOL", "bad-float")
    monkeypatch.setenv("VMEC_JAX_DYNAMIC_SCAN_ATOL", "bad-float")
    calls: list[dict] = []

    def fake_solver(state, static, **kwargs):
        del static
        use_scan = bool(kwargs["use_scan"])
        calls.append({"max_iter": int(kwargs["max_iter"]), "use_scan": use_scan})
        return _result(
            state,
            fsq=1.0e-12,
            converged=True,
            max_iter=int(kwargs["max_iter"]),
            diagnostics={"use_scan": use_scan, "vmec2000_scan": use_scan},
        )

    monkeypatch.setattr(driver, "solve_fixed_boundary_residual_iter", fake_solver)

    run = driver.run_fixed_boundary(
        tmp_path / "input.dynamic_scan_invalid_env",
        solver="vmec2000_iter",
        solver_mode="default",
        max_iter=4,
        verbose=True,
        multigrid=False,
        use_scan=True,
        jit_forces=False,
        grid=object(),
        _auto_cli_fixed_boundary_mode=False,
    )

    assert calls == [
        {"max_iter": 3, "use_scan": True},
        {"max_iter": 3, "use_scan": False},
        {"max_iter": 4, "use_scan": True},
    ]
    assert "dynamic scan parity probe: backend=cpu iters=3 fsq_ok=True -> use_scan=True" in capsys.readouterr().out
    assert run.result.diagnostics["use_scan"] is True


def test_cli_finisher_short_circuits_when_initial_run_is_already_strict_converged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _cfg(ns=3)
    indata = _indata(NITER=2, FTOL=1.0e-8)
    _install_driver_fakes(monkeypatch, cfg=cfg, indata=indata)
    monkeypatch.setenv("VMEC_JAX_DISABLE_JIT_INIT", "1")
    calls: list[dict] = []

    def fake_solver(state, static, **kwargs):
        del static
        calls.append({"max_iter": int(kwargs["max_iter"]), "use_scan": bool(kwargs["use_scan"])})
        return _result(
            state,
            fsq=1.0e-12,
            converged=True,
            max_iter=int(kwargs["max_iter"]),
            diagnostics={"ftol": float(kwargs["ftol"])},
        )

    monkeypatch.setattr(driver, "solve_fixed_boundary_residual_iter", fake_solver)

    run = driver.run_fixed_boundary(
        tmp_path / "input.strict_converged_short_circuit",
        solver="vmec2000_iter",
        solver_mode="accelerated",
        max_iter=2,
        verbose=False,
        multigrid=False,
        cli_fixed_boundary_mode=True,
        use_scan=False,
        jit_forces=False,
        grid=object(),
    )

    assert calls == [{"max_iter": 2, "use_scan": False}]
    diag = run.result.diagnostics
    assert diag["converged"] is True
    assert diag["converged_strict"] is True
    assert np.asarray(diag["cli_fixed_boundary_finish_budgets"]).tolist() == []
    assert np.asarray(diag["cli_fixed_boundary_finish_modes"]).tolist() == []
    assert diag["cli_fixed_boundary_full_parity_fallback"] is False
