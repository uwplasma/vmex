from __future__ import annotations

import sys
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import vmec_jax.cli as cli
import vmec_jax.driver as driver
from vmec_jax.config import FreeBoundaryConfig, VMECConfig
from vmec_jax.free_boundary import MGridMetadata, PreparedMGrid
from vmec_jax.namelist import InData
from vmec_jax.solve import SolveVmecResidualResult


def _cfg(
    *,
    ns: int = 3,
    lfreeb: bool = False,
    lthreed: bool = False,
    lasym: bool = False,
) -> VMECConfig:
    return VMECConfig(
        mpol=2,
        ntor=1 if lthreed else 0,
        ns=int(ns),
        nfp=1,
        lasym=bool(lasym),
        lthreed=bool(lthreed),
        lconm1=True,
        ntheta=4,
        nzeta=2 if lthreed else 1,
        free_boundary=FreeBoundaryConfig(
            enabled=bool(lfreeb),
            mgrid_file="mgrid.synthetic.nc" if lfreeb else "NONE",
            extcur=(1.0,) if lfreeb else (),
            nvacskip=1,
        ),
    )


def _indata(**values) -> InData:
    scalars = {
        "NITER": 2,
        "FTOL": 1.0e-8,
        "DELT": 0.25,
        "SIGNGS": 7,
        "PHIEDGE": 1.0,
    }
    scalars.update(values)
    return InData(scalars=scalars, indexed={})


def _state(ns: int, label: str = "state") -> SimpleNamespace:
    arr = np.zeros((int(ns), 1), dtype=float)
    return SimpleNamespace(
        label=label,
        layout=SimpleNamespace(ns=int(ns), K=1),
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
    n_iter: int = 0,
    diagnostics: dict | None = None,
) -> SolveVmecResidualResult:
    diag = {
        "converged": bool(converged),
        "ftol": 1.0e-8,
        "final_fsqr": float(fsq),
        "final_fsqz": 0.0,
        "final_fsql": 0.0,
        "resume_state": {"time_step": 0.5, "iter_offset": 4, "flip_sign": "-1"},
    }
    if diagnostics:
        diag.update(diagnostics)
    return SolveVmecResidualResult(
        state=state,
        n_iter=int(n_iter),
        w_history=np.asarray([float(fsq)], dtype=float),
        fsqr2_history=np.asarray([float(fsq)], dtype=float),
        fsqz2_history=np.asarray([0.0], dtype=float),
        fsql2_history=np.asarray([0.0], dtype=float),
        grad_rms_history=np.asarray([], dtype=float),
        step_history=np.asarray([], dtype=float),
        diagnostics=diag,
    )


def _prepared_mgrid() -> PreparedMGrid:
    return PreparedMGrid(
        metadata=MGridMetadata(
            path="mgrid.synthetic.nc",
            ir=2,
            jz=2,
            kp=1,
            nfp=1,
            nextcur=1,
            rmin=0.0,
            rmax=1.0,
            zmin=-1.0,
            zmax=1.0,
            mgrid_mode="S",
            coil_groups=("coil",),
            raw_coil_cur=(1.0,),
        ),
        extcur=(2.0,),
    )


def _install_light_driver(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cfg: VMECConfig,
    indata: InData,
    solver,
    prepared_fb=None,
) -> dict[str, list]:
    calls: dict[str, list] = {
        "build_static": [],
        "initial_guess": [],
        "interp": [],
        "prepare_mgrid": [],
        "validate_free_boundary": [],
        "final_flux": [],
    }
    monkeypatch.setenv("VMEC_JAX_COMPILATION_CACHE", "0")
    monkeypatch.setenv("VMEC_JAX_DISABLE_JIT_INIT", "1")
    monkeypatch.setattr(driver, "load_config", lambda _path: (cfg, indata))
    monkeypatch.setattr(driver, "_default_backend_name", lambda: "cpu")
    monkeypatch.setattr(
        driver,
        "validate_free_boundary_config",
        lambda cfg_in, *, strict: calls["validate_free_boundary"].append((cfg_in, strict)),
    )
    monkeypatch.setattr(
        driver,
        "prepare_mgrid_for_config",
        lambda cfg_in, *, load_fields, strict: calls["prepare_mgrid"].append((load_fields, strict))
        or prepared_fb,
    )

    def fake_static(cfg_in, **kwargs):
        calls["build_static"].append((cfg_in, kwargs))
        return SimpleNamespace(
            cfg=cfg_in,
            modes=SimpleNamespace(m=np.asarray([0]), n=np.asarray([0])),
            s=np.linspace(0.0, 1.0, int(cfg_in.ns)),
            grid=SimpleNamespace(theta=np.asarray([0.0]), zeta=np.asarray([0.0])),
        )

    def fake_initial_guess(static, boundary, indata_arg, **kwargs):
        calls["initial_guess"].append((static.cfg.ns, kwargs))
        return _state(static.cfg.ns, "initial")

    def fake_interp(state, **kwargs):
        calls["interp"].append((state.layout.ns, int(kwargs["ns_new"])))
        return _state(int(kwargs["ns_new"]), f"interp-{kwargs['ns_new']}")

    monkeypatch.setattr(driver, "build_static", fake_static)
    monkeypatch.setattr(driver, "boundary_from_indata", lambda *_args, **_kwargs: SimpleNamespace(boundary=True))
    monkeypatch.setattr(driver, "initial_guess_from_boundary", fake_initial_guess)
    monkeypatch.setattr(driver, "interp_vmec_state", fake_interp)
    monkeypatch.setattr(
        driver,
        "flux_profiles_from_indata",
        lambda _indata, s, *, signgs: SimpleNamespace(
            phipf=np.ones_like(np.asarray(s)),
            chipf=np.zeros_like(np.asarray(s)),
            phips=np.ones_like(np.asarray(s)),
            lamscale=np.asarray(1.0),
            signgs=int(signgs),
        ),
    )
    monkeypatch.setattr(driver, "eval_profiles", lambda _indata, s: {"pressure": np.zeros_like(np.asarray(s))})
    monkeypatch.setattr(driver, "_final_flux_profiles_from_state", lambda **kwargs: (kwargs["flux_local"], kwargs["prof_local"]))
    monkeypatch.setattr(driver, "solve_fixed_boundary_residual_iter", solver)
    return calls


def test_cli_staged_followup_records_policy_and_beats_single_grid(monkeypatch, tmp_path: Path) -> None:
    cfg = _cfg(ns=5, lthreed=True)
    indata = _indata(NS_ARRAY=[3, 5], NITER_ARRAY=[1, 1], FTOL_ARRAY=[1.0e-4, 1.0e-8])
    solve_calls: list[dict] = []

    def fake_solver(state, static, **kwargs):
        solve_calls.append({"ns": int(static.cfg.ns), **kwargs})
        fsq = 1.0e-3 if len(solve_calls) == 1 else 1.0e-10
        return _result(_state(static.cfg.ns, f"solve-{len(solve_calls)}"), fsq=fsq, converged=fsq < 1.0e-8)

    _install_light_driver(monkeypatch, cfg=cfg, indata=indata, solver=fake_solver)

    run = driver.run_fixed_boundary(
        tmp_path / "input.staged",
        solver="vmec2000_iter",
        solver_mode="accelerated",
        max_iter=2,
        multigrid=False,
        verbose=False,
        jit_forces=False,
        cli_fixed_boundary_mode=True,
    )

    diag = run.result.diagnostics
    assert [call["ns"] for call in solve_calls] == [5, 3, 5]
    assert diag["cli_fixed_boundary_staged_followup_used"] is True
    assert diag["cli_fixed_boundary_staged_followup_policy"] == "input_multigrid"
    np.testing.assert_array_equal(diag["cli_fixed_boundary_staged_followup_ns"], [3, 5])
    assert np.asarray(diag["cli_fixed_boundary_staged_followup_modes"]).tolist() == ["parity", "accelerated"]
    assert np.asarray(diag["cli_fixed_boundary_staged_followup_wall_s"]).shape == (2,)
    assert np.all(np.asarray(diag["cli_fixed_boundary_staged_followup_wall_s"]) >= 0.0)
    assert np.asarray(diag["cli_fixed_boundary_staged_followup_solve_total_s"]).shape == (2,)
    assert diag["converged"] is True


def test_staged_cli_full_parity_fallback_diagnostics_when_followup_does_not_improve(monkeypatch, tmp_path: Path) -> None:
    cfg = _cfg(ns=5)
    indata = _indata(NS_ARRAY=[3, 5], NITER_ARRAY=[1, 1])
    solve_calls: list[dict] = []

    def fake_solver(state, static, **kwargs):
        solve_calls.append({"ns": int(static.cfg.ns), **kwargs})
        return _result(_state(static.cfg.ns, f"solve-{len(solve_calls)}"), fsq=1.0e-3, converged=False)

    _install_light_driver(monkeypatch, cfg=cfg, indata=indata, solver=fake_solver)

    run = driver.run_fixed_boundary(
        tmp_path / "input.fallback",
        solver="vmec2000_iter",
        solver_mode="accelerated",
        max_iter=1,
        multigrid=False,
        verbose=False,
        jit_forces=False,
        cli_fixed_boundary_mode=True,
        use_scan=False,
    )

    diag = run.result.diagnostics
    assert diag["cli_fixed_boundary_full_parity_fallback"] is True
    assert diag["cli_fixed_boundary_finish_budget_exhausted"] is False
    assert np.asarray(diag["cli_fixed_boundary_finish_modes"]).tolist() == ["parity"]
    assert any(call["use_scan"] is False for call in solve_calls)


def test_solver_device_reroute_wraps_recursive_run_and_adds_diagnostics(monkeypatch, tmp_path: Path) -> None:
    cfg = _cfg()
    indata = _indata()
    contexts: list[str] = []

    class FakeJax:
        @staticmethod
        def devices(name):
            return [f"{name}:0"]

        @staticmethod
        @contextmanager
        def default_device(device):
            contexts.append(device)
            yield

        class config:
            @staticmethod
            def update(*_args, **_kwargs):
                return None

    monkeypatch.setitem(sys.modules, "jax", FakeJax)
    monkeypatch.setattr(driver, "_resolve_fixed_boundary_solver_device_name", lambda **_kwargs: "cpu")

    def fake_solver(state, static, **kwargs):
        return _result(_state(static.cfg.ns, "routed"), fsq=1.0e-10, converged=True, diagnostics={"use_scan": kwargs["use_scan"]})

    _install_light_driver(monkeypatch, cfg=cfg, indata=indata, solver=fake_solver)

    run = driver.run_fixed_boundary(
        tmp_path / "input.device",
        solver_mode="parity",
        max_iter=1,
        verbose=False,
        grid=object(),
        jit_forces=False,
        solver_device="cpu",
        _auto_cli_fixed_boundary_mode=False,
    )

    assert contexts == ["cpu:0"]
    assert run.result.diagnostics["solver_device"] == "cpu"
    assert run.result.diagnostics["solver_device_auto_reroute"] is False


def test_compilation_cache_success_and_config_failure_branches(monkeypatch, tmp_path: Path) -> None:
    cfg = _cfg()
    indata = _indata()
    updates: list[tuple] = []
    cache_dirs: list[str] = []

    jax_module = types.ModuleType("jax")

    class Config:
        def update(self, *args):
            updates.append(args)
            if args[0] == "jax_enable_compilation_cache":
                raise RuntimeError("synthetic config failure")

    jax_module.config = Config()
    experimental_module = types.ModuleType("jax.experimental")
    cache_module = types.ModuleType("jax.experimental.compilation_cache")
    cache_module.compilation_cache = SimpleNamespace(set_cache_dir=lambda path: cache_dirs.append(path))
    monkeypatch.setitem(sys.modules, "jax", jax_module)
    monkeypatch.setitem(sys.modules, "jax.experimental", experimental_module)
    monkeypatch.setitem(sys.modules, "jax.experimental.compilation_cache", cache_module)
    monkeypatch.setattr("vmec_jax._compat._default_compilation_cache_dir", lambda: tmp_path / "cache")
    monkeypatch.delenv("VMEC_JAX_COMPILATION_CACHE", raising=False)
    monkeypatch.delenv("VMEC_JAX_DISABLE_COMPILATION_CACHE", raising=False)
    monkeypatch.setenv("VMEC_JAX_CACHE_MIN_COMPILE_TIME_SECS", "0.5")
    monkeypatch.setenv("VMEC_JAX_CACHE_MIN_ENTRY_SIZE_BYTES", "bad-int")

    _install_light_driver(
        monkeypatch,
        cfg=cfg,
        indata=indata,
        solver=lambda state, static, **kwargs: _result(state, fsq=0.0, converged=True),
    )
    monkeypatch.delenv("VMEC_JAX_COMPILATION_CACHE", raising=False)

    run = driver.run_fixed_boundary(
        tmp_path / "input.cache",
        use_initial_guess=True,
        verbose=False,
        grid=object(),
        _auto_cli_fixed_boundary_mode=False,
    )

    assert run.result is None
    assert cache_dirs == [str(tmp_path / "cache")]
    assert updates[0] == ("jax_enable_compilation_cache", True)
    assert any(update[0] == "jax_persistent_cache_min_compile_time_secs" for update in updates)


def test_free_boundary_strict_env_and_prepared_mgrid_reach_static_kwargs(monkeypatch, tmp_path: Path) -> None:
    cfg = _cfg(lfreeb=True)
    indata = _indata(LFREEB=True, MGRID_FILE="mgrid.synthetic.nc")

    def fake_solver(state, static, **kwargs):
        return _result(_state(static.cfg.ns, "freeb"), fsq=1.0e-10, converged=True, diagnostics={"use_scan": kwargs["use_scan"]})

    calls = _install_light_driver(
        monkeypatch,
        cfg=cfg,
        indata=indata,
        solver=fake_solver,
        prepared_fb=_prepared_mgrid(),
    )
    monkeypatch.setenv("VMEC_JAX_FREEB_STRICT", "0")

    run = driver.run_fixed_boundary(
        tmp_path / "input.freeb",
        solver_mode="accelerated",
        max_iter=1,
        verbose=False,
        grid=object(),
        jit_forces=False,
        _auto_cli_fixed_boundary_mode=False,
    )

    assert run.result.diagnostics["use_scan"] is False
    assert calls["validate_free_boundary"] == [(cfg, False)]
    assert calls["prepare_mgrid"] == [(False, False)]
    build_kwargs = calls["build_static"][0][1]
    assert isinstance(build_kwargs["mgrid_metadata"], MGridMetadata)
    assert build_kwargs["free_boundary_extcur"] == (2.0,)


def test_run_free_boundary_delegates_only_after_lfreeb_guard(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "input.case"
    path.write_text("&INDATA\n/\n")
    monkeypatch.setattr(driver, "load_config", lambda _path: (_cfg(lfreeb=False), _indata()))

    with pytest.raises(ValueError, match="Use run_fixed_boundary"):
        driver.run_free_boundary(path, max_iter=1)

    free_cfg = _cfg(lfreeb=True)
    captured = {}
    monkeypatch.setattr(driver, "load_config", lambda _path: (free_cfg, _indata(LFREEB=True)))
    monkeypatch.setattr(driver, "run_fixed_boundary", lambda input_path, **kwargs: captured.setdefault("call", (input_path, kwargs)))

    out = driver.run_free_boundary(path, max_iter=2, verbose=False)

    assert out == (path, {"max_iter": 2, "verbose": False})
    assert captured["call"] == out


def test_cli_default_policy_uses_solver_device_backend_for_scan_selection(monkeypatch, tmp_path: Path) -> None:
    input_path = tmp_path / "input.case"
    input_path.write_text("&INDATA\n/\n")
    calls: list[tuple] = []
    indata = _indata(NITER=3)

    monkeypatch.setattr(cli, "read_indata", lambda _path: indata)
    monkeypatch.setattr(
        cli,
        "_default_non_autodiff_solver_policy_for_backend",
        lambda indata_arg, backend: calls.append(("policy", indata_arg, backend)) or ("accelerated", True),
    )
    monkeypatch.setattr(
        cli,
        "_default_use_scan_for_backend",
        lambda indata_arg, backend, mode: calls.append(("scan", indata_arg, backend, mode)) or True,
    )

    def fake_run_fixed_boundary(path, **kwargs):
        calls.append(("run", Path(path), kwargs))
        return SimpleNamespace(state=SimpleNamespace(Rcos=np.asarray([0.0])))

    monkeypatch.setattr(cli, "run_fixed_boundary", fake_run_fixed_boundary)
    monkeypatch.setattr(cli, "write_wout_from_fixed_boundary_run", lambda *args, **kwargs: calls.append(("write", args, kwargs)))

    assert cli.main([str(input_path), "--solver-device", "gpu", "--quiet"]) == 0

    assert calls[0] == ("policy", indata, "gpu")
    assert calls[1] == ("scan", indata, "gpu", "accelerated")
    run_kwargs = calls[2][2]
    assert run_kwargs["solver_device"] == "gpu"
    assert run_kwargs["use_scan"] is True
    assert run_kwargs["cli_fixed_boundary_mode"] is True
