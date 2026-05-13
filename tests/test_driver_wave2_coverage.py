from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import vmec_jax.driver as driver
import vmec_jax.solve as solve_module
from vmec_jax.driver import run_fixed_boundary
from vmec_jax.energy import FluxProfiles
from vmec_jax.solve import SolveVmecResidualResult
from vmec_jax.vmec_tomnsp import vmec_angle_grid


def _example_input(name: str = "input.circular_tokamak") -> Path:
    return Path(__file__).resolve().parents[1] / "examples" / "data" / name


def _small_grid(*, lasym: bool = False):
    return vmec_angle_grid(ntheta=6, nzeta=1 if not lasym else 4, nfp=1, lasym=lasym)


def _write_input(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text("&INDATA\n" + body + "/\n")
    return path


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
    monkeypatch.setattr(driver, "interp_vmec_state", lambda *_args, ns_new, **_kwargs: _fake_state(ns_new))
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


def _result(state, *, max_iter: int, fsq: float, converged: bool, diagnostics: dict | None = None):
    diag = {
        "converged": bool(converged),
        "resume_state": {
            "time_step": 0.25,
            "inv_tau": [1.0, 2.0],
            "iter_offset": 3,
            "vmec2000_cache_valid": True,
            "flip_sign": -1.0,
        },
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


@pytest.mark.parametrize(
    "solver",
    ["gd", "lbfgs", "vmec_lbfgs", "vmec_gn"],
)
def test_run_fixed_boundary_solver_wrappers_delegate_without_full_solve(monkeypatch, solver) -> None:
    calls = []

    def fake_solver(state, static, **kwargs):
        calls.append({"state": state, "static": static, "kwargs": dict(kwargs)})
        return _result(state, max_iter=kwargs.get("max_iter", 1), fsq=1.0e-12, converged=True)

    monkeypatch.setenv("VMEC_JAX_DISABLE_JIT_INIT", "1")
    _patch_lightweight_driver_core(monkeypatch)
    monkeypatch.setattr(driver, "solve_fixed_boundary_gd", fake_solver)
    monkeypatch.setattr(driver, "solve_fixed_boundary_lbfgs", fake_solver)
    monkeypatch.setattr(solve_module, "solve_fixed_boundary_lbfgs_vmec_residual", fake_solver)
    monkeypatch.setattr(solve_module, "solve_fixed_boundary_gn_vmec_residual", fake_solver)

    run = run_fixed_boundary(
        _example_input(),
        solver=solver,
        max_iter=3,
        step_size=0.125,
        history_size=4,
        gn_damping=0.5,
        gn_cg_tol=1.0e-4,
        gn_cg_maxiter=7,
        verbose=False,
        grid=_small_grid(),
        performance_mode=False,
    )

    assert run.result is not None
    assert len(calls) == 1
    kwargs = calls[0]["kwargs"]
    assert kwargs["max_iter"] == 3
    assert kwargs["step_size"] == pytest.approx(0.125)
    if solver == "lbfgs":
        assert kwargs["history_size"] == 4
    if solver == "vmec_gn":
        assert kwargs["damping"] == pytest.approx(0.5)
        assert kwargs["cg_tol"] == pytest.approx(1.0e-4)
        assert kwargs["cg_maxiter"] == 7
    assert run.state is calls[0]["state"]


def test_non_vmec_verbose_finish_reports_gradient_history(monkeypatch, capsys) -> None:
    def fake_solver(state, static, **kwargs):
        del static, kwargs
        return SolveVmecResidualResult(
            state=state,
            n_iter=2,
            w_history=np.asarray([0.5], dtype=float),
            fsqr2_history=np.asarray([0.5], dtype=float),
            fsqz2_history=np.asarray([0.0], dtype=float),
            fsql2_history=np.asarray([0.0], dtype=float),
            grad_rms_history=np.asarray([0.25], dtype=float),
            step_history=np.asarray([], dtype=float),
            diagnostics={"converged": True, "resume_state": {}},
        )

    monkeypatch.setenv("VMEC_JAX_DISABLE_JIT_INIT", "1")
    _patch_lightweight_driver_core(monkeypatch)
    monkeypatch.setattr(driver, "solve_fixed_boundary_gd", fake_solver)

    run = run_fixed_boundary(
        _example_input(),
        solver="gd",
        max_iter=3,
        verbose=True,
        grid=_small_grid(),
        performance_mode=False,
    )

    assert run.result.n_iter == 2
    assert "grad_rms=2.500e-01" in capsys.readouterr().out


def test_run_fixed_boundary_rejects_restart_state_ns_mismatch() -> None:
    restart_state = SimpleNamespace(layout=SimpleNamespace(ns=5))

    with pytest.raises(ValueError, match="restart_state ns=5 does not match ns_override=7"):
        run_fixed_boundary(
            _example_input(),
            restart_state=restart_state,
            ns_override=7,
            verbose=False,
            grid=_small_grid(),
        )


def test_run_fixed_boundary_restart_wout_path_uses_loaded_state(monkeypatch, tmp_path) -> None:
    restart_state = SimpleNamespace(layout=SimpleNamespace(ns=3), marker="restart")
    cfg = SimpleNamespace(ns=5)
    loaded = []

    _patch_lightweight_driver_core(monkeypatch)
    monkeypatch.setattr(driver, "read_wout", lambda path: loaded.append(Path(path)) or "wout")
    monkeypatch.setattr(driver, "state_from_wout", lambda wout: restart_state if wout == "wout" else None)

    run = run_fixed_boundary(
        _example_input(),
        restart_wout_path=tmp_path / "wout_fake.nc",
        use_initial_guess=True,
        verbose=False,
        grid=_small_grid(),
    )

    del cfg
    assert loaded == [tmp_path / "wout_fake.nc"]
    assert run.state is restart_state
    assert run.cfg.ns == 3


def test_solver_device_cpu_reroute_annotates_result_diagnostics(monkeypatch) -> None:
    calls = []

    def fake_solver(state, static, **kwargs):
        calls.append({"ns": int(static.cfg.ns), "max_iter": int(kwargs["max_iter"])})
        return _result(
            state,
            max_iter=int(kwargs["max_iter"]),
            fsq=1.0e-16,
            converged=True,
            diagnostics={"ftol": float(kwargs["ftol"]), "use_scan": bool(kwargs["use_scan"])},
        )

    monkeypatch.setenv("VMEC_JAX_DISABLE_JIT_INIT", "1")
    _patch_lightweight_driver_core(monkeypatch)
    monkeypatch.setattr(driver, "solve_fixed_boundary_residual_iter", fake_solver)
    monkeypatch.setattr(solve_module, "solve_fixed_boundary_residual_iter", fake_solver)

    run = run_fixed_boundary(
        _example_input(),
        solver="vmec2000_iter",
        solver_mode="parity",
        solver_device="cpu",
        max_iter=1,
        verbose=False,
        multigrid=False,
        grid=_small_grid(),
    )

    assert calls == [{"ns": run.static.cfg.ns, "max_iter": 1}]
    assert run.result.diagnostics["solver_device"] == "cpu"
    assert run.result.diagnostics["solver_device_auto_reroute"] is False


def test_run_fixed_boundary_dump_xc_init_writes_debug_file(monkeypatch, tmp_path) -> None:
    import vmec_jax.diagnostics as diagnostics

    monkeypatch.setenv("VMEC_JAX_DISABLE_JIT_INIT", "1")
    monkeypatch.setenv("VMEC_JAX_DUMP_XC_INIT", "1")
    monkeypatch.setenv("VMEC_JAX_DUMP_DIR", str(tmp_path))
    _patch_lightweight_driver_core(monkeypatch)
    monkeypatch.setattr(
        diagnostics,
        "vmec_internal_mn_from_state",
        lambda *_args, **_kwargs: {
            "rcc": np.asarray([1.0]),
            "rss": np.asarray([2.0]),
            "zsc": np.asarray([3.0]),
            "zcs": np.asarray([4.0]),
            "lsc": np.asarray([5.0]),
            "lcs": np.asarray([6.0]),
            "rsc": np.asarray([7.0]),
            "rcs": np.asarray([8.0]),
            "zcc": np.asarray([9.0]),
            "zss": np.asarray([10.0]),
            "lcc": np.asarray([11.0]),
            "lss": np.asarray([12.0]),
        },
    )
    monkeypatch.setattr(diagnostics, "vmec_xc_from_mn_blocks", lambda **_kwargs: np.asarray([1.5, -2.5]))

    run = run_fixed_boundary(
        _example_input(),
        use_initial_guess=True,
        verbose=False,
        vmec_project=False,
        grid=_small_grid(),
    )

    dump_path = tmp_path / f"xc_init_init_ns{run.static.cfg.ns}.dat"
    text = dump_path.read_text()
    assert "neqs=2" in text
    assert "1.5000000000000000e+00" in text
    assert "-2.5000000000000000e+00" in text


def test_compilation_cache_setup_uses_configured_cache_dir(monkeypatch, tmp_path) -> None:
    import jax
    import vmec_jax._compat as compat
    from jax.experimental.compilation_cache import compilation_cache

    cache_dir = tmp_path / "jax-cache"
    cache_calls = []
    config_calls = []

    monkeypatch.setenv("VMEC_JAX_COMPILATION_CACHE", "1")
    monkeypatch.setenv("VMEC_JAX_CACHE_MIN_COMPILE_TIME_SECS", "0.25")
    monkeypatch.setenv("VMEC_JAX_CACHE_MIN_ENTRY_SIZE_BYTES", "123")
    monkeypatch.setenv("VMEC_JAX_COMPILATION_CACHE_MAX_SIZE", "456")
    monkeypatch.setenv("VMEC_JAX_EXPLAIN_CACHE_MISSES", "1")
    monkeypatch.setattr(compat, "_default_compilation_cache_dir", lambda: str(cache_dir))
    monkeypatch.setattr(compilation_cache, "set_cache_dir", lambda value: cache_calls.append(value))
    monkeypatch.setattr(jax.config, "update", lambda key, value: config_calls.append((key, value)))
    _patch_lightweight_driver_core(monkeypatch)

    run = run_fixed_boundary(
        _example_input(),
        use_initial_guess=True,
        verbose=False,
        vmec_project=False,
    )

    assert run.state is not None
    assert cache_calls == [str(cache_dir)]
    assert ("jax_enable_compilation_cache", True) in config_calls
    assert ("jax_persistent_cache_min_compile_time_secs", 0.25) in config_calls
    assert ("jax_persistent_cache_min_entry_size_bytes", 123) in config_calls
    assert ("jax_compilation_cache_max_size", 456) in config_calls
    assert ("jax_explain_cache_misses", True) in config_calls


def test_compilation_cache_falls_back_to_tmp_when_default_cache_is_unwritable(monkeypatch, tmp_path) -> None:
    import jax
    import vmec_jax._compat as compat
    from jax.experimental.compilation_cache import compilation_cache

    cache_calls = []
    mkdir_calls = []

    def fake_mkdir(self, *args, **kwargs):
        del args, kwargs
        mkdir_calls.append(str(self))
        if len(mkdir_calls) == 1:
            raise OSError("primary cache unavailable")

    monkeypatch.setenv("VMEC_JAX_COMPILATION_CACHE", "1")
    monkeypatch.setattr(compat, "_default_compilation_cache_dir", lambda: str(tmp_path / "blocked-cache"))
    monkeypatch.setattr(driver.Path, "mkdir", fake_mkdir)
    monkeypatch.setattr(compilation_cache, "set_cache_dir", lambda value: cache_calls.append(value))
    monkeypatch.setattr(jax.config, "update", lambda *_args, **_kwargs: None)
    _patch_lightweight_driver_core(monkeypatch)

    run = run_fixed_boundary(
        _example_input(),
        use_initial_guess=True,
        verbose=False,
        vmec_project=False,
        grid=_small_grid(),
    )

    assert run.state is not None
    assert mkdir_calls == [str(tmp_path / "blocked-cache"), "/tmp/vmec_jax/jax_compilation_cache"]
    assert cache_calls == ["/tmp/vmec_jax/jax_compilation_cache"]


def test_driver_history_and_flux_profile_helper_branches(monkeypatch) -> None:
    import vmec_jax.vmec_bcovar as bcovar_module
    import vmec_jax.vmec_residue as residue_module
    import vmec_jax.wout as wout_module

    assert driver._vmec_history_relerr([2.0, 4.0], [1.0, 2.0]) == pytest.approx(1.0)
    assert np.isinf(driver._vmec_history_relerr([1.0], [1.0, 2.0]))

    same = SimpleNamespace(
        w_history=np.asarray([1.0]),
        fsqr2_history=np.asarray([2.0]),
        fsqz2_history=np.asarray([0.0]),
        fsql2_history=np.asarray([0.0]),
    )
    shifted = SimpleNamespace(
        w_history=np.asarray([1.0]),
        fsqr2_history=np.asarray([2.5]),
        fsqz2_history=np.asarray([0.0]),
        fsql2_history=np.asarray([0.0]),
    )
    assert driver._vmec_histories_match(same, same, rtol=1.0e-12, atol=0.0)
    assert not driver._vmec_histories_match(same, shifted, rtol=1.0e-12, atol=0.0)

    ns = 3
    indata = SimpleNamespace(
        get_int=lambda key, default=0: 1 if key == "NCURR" else default,
        get_float=lambda key, default=0.0: 1.0 if key == "GAMMA" else default,
        get_bool=lambda key, default=False: True if key == "LRFP" else default,
    )
    static = SimpleNamespace(
        s=np.linspace(0.0, 1.0, ns),
        modes=SimpleNamespace(m=np.asarray([0]), n=np.asarray([0])),
        grid=SimpleNamespace(theta=np.asarray([0.0]), zeta=np.asarray([0.0])),
        trig_vmec=object(),
        cfg=SimpleNamespace(nfp=1, mpol=1, ntor=0, lasym=False),
    )
    flux = FluxProfiles(
        phipf=np.asarray([1.0, 2.0, 3.0]),
        chipf=np.asarray([0.0, 2.0, 4.0]),
        phips=np.asarray([0.0, 2.0, 4.0]),
        signgs=1,
        lamscale=np.asarray(1.0),
    )

    monkeypatch.setattr(driver, "boundary_from_indata", lambda *_args, **_kwargs: SimpleNamespace(R_cos=np.asarray([2.0])))
    monkeypatch.setattr(
        bcovar_module,
        "vmec_bcovar_half_mesh_from_wout",
        lambda **_kwargs: SimpleNamespace(
            jac=SimpleNamespace(sqrtg=np.ones((ns, 1, 1))),
            guu=np.ones((ns, 1, 1)) * 2.0,
            guv=np.ones((ns, 1, 1)) * 0.5,
            bsupu=np.zeros((ns, 1, 1)),
            bsupv=np.zeros((ns, 1, 1)),
        ),
    )
    monkeypatch.setattr(residue_module, "vmec_pwint_from_trig", lambda *_args, **_kwargs: np.ones((ns, 1, 1)))
    monkeypatch.setattr(wout_module, "_icurv_full_mesh_from_indata", lambda **_kwargs: np.asarray([0.0, 4.0, 8.0]))
    monkeypatch.setattr(wout_module, "_chipf_from_chips", lambda chips: np.asarray(chips) + 0.25)
    monkeypatch.setattr(driver, "_iotaf_from_iotas", lambda iotas, *, lrfp: np.asarray(iotas) + (0.5 if lrfp else 0.0))

    flux_out, prof_out = driver._final_flux_profiles_from_state(
        indata=indata,
        static_in=static,
        state=_fake_state(ns),
        signgs=1,
        flux_local=flux,
        prof_local={"pressure": np.asarray([0.0, 1.0, 2.0])},
        pressure_local=np.asarray([0.0, 1.0, 2.0]),
    )

    np.testing.assert_allclose(prof_out["iota"], [0.0, 1.0, 1.0])
    np.testing.assert_allclose(prof_out["iotaf"], [0.5, 1.5, 1.5])
    np.testing.assert_allclose(flux_out.chipf, [0.25, 2.25, 4.25])


def test_cli_finisher_accepts_strict_residuals_even_when_converged_flag_is_false(monkeypatch, tmp_path) -> None:
    input_path = _write_input(
        tmp_path,
        "input.strict_finish",
        """
  LFREEB = F
  NFP = 1
  MPOL = 5
  NTOR = 0
  NS = 7
  NITER = 3
  FTOL = 1e-10
  PHIEDGE = 1.0
  RBC(0,0) = 1.0
  ZBS(1,0) = 0.1
""",
    )
    calls = []

    def fake_solver(state, static, **kwargs):
        calls.append(int(kwargs["max_iter"]))
        return _result(
            state,
            max_iter=int(kwargs["max_iter"]),
            fsq=1.0e-12,
            converged=False,
            diagnostics={
                "ftol": float(kwargs["ftol"]),
                "final_fsqr": 1.0e-12,
                "final_fsqz": 2.0e-12,
                "final_fsql": 3.0e-12,
            },
        )

    monkeypatch.setenv("VMEC_JAX_DISABLE_JIT_INIT", "1")
    _patch_lightweight_driver_core(monkeypatch)
    monkeypatch.setattr(driver, "solve_fixed_boundary_residual_iter", fake_solver)
    monkeypatch.setattr(solve_module, "solve_fixed_boundary_residual_iter", fake_solver)

    run = run_fixed_boundary(
        input_path,
        solver="vmec2000_iter",
        solver_mode="parity",
        verbose=False,
        cli_fixed_boundary_mode=True,
    )

    assert calls == [3]
    diag = run.result.diagnostics
    assert diag["converged"] is True
    assert diag["converged_strict"] is True
    assert np.asarray(diag["cli_fixed_boundary_finish_budgets"]).tolist() == []
    assert diag["cli_fixed_boundary_full_parity_fallback"] is False


def test_accelerated_multigrid_miss_uses_partial_and_full_parity_fallback(monkeypatch, tmp_path) -> None:
    input_path = _write_input(
        tmp_path,
        "input.partial_fallback",
        """
  LFREEB = F
  LASYM = F
  NFP = 2
  MPOL = 5
  NTOR = 1
  NS = 13
  NITER = 3
  FTOL = 1e-14
  NS_ARRAY = 5 9 13
  NITER_ARRAY = 1 1 1
  FTOL_ARRAY = 1e-14 1e-14 1e-14
  PHIEDGE = 1.0
  RBC(0,0) = 1.0
  ZBS(1,0) = 0.1
""",
    )
    calls = []

    def fake_solver(state, static, **kwargs):
        idx = len(calls)
        calls.append(
            {
                "ns": int(static.cfg.ns),
                "max_iter": int(kwargs["max_iter"]),
                "use_scan": bool(kwargs["use_scan"]),
            }
        )
        converged = idx >= 7
        fsq = 1.0e-16 if converged else (5.0e-4 if idx == 3 else 1.0e-3)
        return _result(
            state,
            max_iter=int(kwargs["max_iter"]),
            fsq=fsq,
            converged=converged,
            diagnostics={
                "ftol": float(kwargs["ftol"]),
                "use_scan": bool(kwargs["use_scan"]),
                "final_fsqr": fsq,
                "final_fsqz": 0.0,
                "final_fsql": 0.0,
            },
        )

    monkeypatch.setenv("VMEC_JAX_DISABLE_JIT_INIT", "1")
    _patch_lightweight_driver_core(monkeypatch)
    monkeypatch.setattr(driver, "solve_fixed_boundary_residual_iter", fake_solver)
    monkeypatch.setattr(solve_module, "solve_fixed_boundary_residual_iter", fake_solver)

    run = run_fixed_boundary(
        input_path,
        solver="vmec2000_iter",
        solver_mode="accelerated",
        step_size=None,
        verbose=False,
        cli_fixed_boundary_mode=True,
    )

    assert calls[:4] == [
        {"ns": 5, "max_iter": 1, "use_scan": True},
        {"ns": 9, "max_iter": 1, "use_scan": True},
        {"ns": 13, "max_iter": 1, "use_scan": True},
        {"ns": 13, "max_iter": 1, "use_scan": False},
    ]
    assert calls[4:7] == [
        {"ns": 5, "max_iter": 1, "use_scan": False},
        {"ns": 9, "max_iter": 1, "use_scan": False},
        {"ns": 13, "max_iter": 1, "use_scan": False},
    ]
    assert calls[-1] == {"ns": 13, "max_iter": 3, "use_scan": False}
    diag = run.result.diagnostics
    assert diag["cli_fixed_boundary_partial_parity_fallback"] is True
    assert diag["cli_fixed_boundary_full_parity_fallback"] is True
    assert diag["converged"] is True


def test_default_mode_single_stage_fast_path_uses_fake_solver(monkeypatch) -> None:
    calls = []

    def fake_solver(state, static, **kwargs):
        calls.append(
            {
                "max_iter": int(kwargs["max_iter"]),
                "use_scan": bool(kwargs["use_scan"]),
                "jit_forces": bool(kwargs["jit_forces"]),
            }
        )
        n = int(kwargs["max_iter"])
        hist = np.linspace(1.0, 0.5, n)
        return SolveVmecResidualResult(
            state=state,
            n_iter=max(0, n - 1),
            w_history=hist,
            fsqr2_history=hist,
            fsqz2_history=hist * 0.0,
            fsql2_history=hist * 0.0,
            grad_rms_history=np.asarray([], dtype=float),
            step_history=np.asarray([], dtype=float),
            diagnostics={"converged": True, "use_scan": bool(kwargs["use_scan"]), "resume_state": {}},
        )

    monkeypatch.setenv("VMEC_JAX_DISABLE_JIT_INIT", "1")
    _patch_lightweight_driver_core(monkeypatch)
    monkeypatch.setattr(driver, "solve_fixed_boundary_residual_iter", fake_solver)
    monkeypatch.setattr(solve_module, "solve_fixed_boundary_residual_iter", fake_solver)

    run = run_fixed_boundary(
        _example_input(),
        solver="vmec2000_iter",
        solver_mode="default",
        max_iter=4,
        verbose=False,
        multigrid=False,
        jit_forces=False,
        grid=_small_grid(),
    )

    assert [call["max_iter"] for call in calls] == [4]
    assert [call["use_scan"] for call in calls] == [False]
    assert run.result.diagnostics["use_scan"] is False


def test_accelerated_single_grid_runs_explicit_staged_followup(monkeypatch, tmp_path) -> None:
    input_path = _write_input(
        tmp_path,
        "input.single_grid_followup",
        """
  LFREEB = F
  LASYM = F
  NFP = 2
  MPOL = 5
  NTOR = 1
  NS = 9
  NITER = 5
  FTOL = 1e-14
  NS_ARRAY = 5 9
  NITER_ARRAY = 2 3
  FTOL_ARRAY = 1e-14 1e-14
  PHIEDGE = 1.0
  RBC(0,0) = 1.0
  ZBS(1,0) = 0.1
""",
    )
    calls = []

    def fake_solver(state, static, **kwargs):
        idx = len(calls)
        calls.append(
            {
                "ns": int(static.cfg.ns),
                "max_iter": int(kwargs["max_iter"]),
                "use_scan": bool(kwargs["use_scan"]),
            }
        )
        converged = idx == 2
        fsq = 1.0e-16 if converged else 1.0e-3
        return _result(
            state,
            max_iter=int(kwargs["max_iter"]),
            fsq=fsq,
            converged=converged,
            diagnostics={
                "ftol": float(kwargs["ftol"]),
                "use_scan": bool(kwargs["use_scan"]),
                "final_fsqr": fsq,
                "final_fsqz": 0.0,
                "final_fsql": 0.0,
            },
        )

    monkeypatch.setenv("VMEC_JAX_DISABLE_JIT_INIT", "1")
    _patch_lightweight_driver_core(monkeypatch)
    monkeypatch.setattr(driver, "solve_fixed_boundary_residual_iter", fake_solver)
    monkeypatch.setattr(solve_module, "solve_fixed_boundary_residual_iter", fake_solver)

    run = run_fixed_boundary(
        input_path,
        solver="vmec2000_iter",
        solver_mode="accelerated",
        step_size=None,
        verbose=False,
        multigrid=False,
        cli_fixed_boundary_mode=True,
    )

    assert calls == [
        {"ns": 9, "max_iter": 5, "use_scan": True},
        {"ns": 5, "max_iter": 2, "use_scan": False},
        {"ns": 9, "max_iter": 3, "use_scan": True},
    ]
    diag = run.result.diagnostics
    assert diag["cli_fixed_boundary_staged_followup_used"] is True
    assert diag["cli_fixed_boundary_staged_followup_policy"] == "input_multigrid"
    assert np.asarray(diag["cli_fixed_boundary_staged_followup_ns"]).tolist() == [5, 9]
    assert np.asarray(diag["cli_fixed_boundary_staged_followup_niter"]).tolist() == [2, 3]
    assert np.asarray(diag["cli_fixed_boundary_staged_followup_modes"]).tolist() == ["parity", "accelerated"]
    assert diag["converged"] is True


def test_accelerated_explicit_stage_monitor_switches_to_parity(monkeypatch, tmp_path) -> None:
    input_path = _write_input(
        tmp_path,
        "input.stage_monitor",
        """
  LFREEB = F
  NFP = 1
  MPOL = 5
  NTOR = 0
  NS = 9
  NITER = 250
  FTOL = 1e-14
  NS_ARRAY = 5 9
  NITER_ARRAY = 1 250
  FTOL_ARRAY = 1e-14 1e-14
  PHIEDGE = 1.0
  RBC(0,0) = 1.0
  ZBS(1,0) = 0.1
""",
    )
    calls = []

    def fake_solver(state, static, **kwargs):
        calls.append({"ns": int(static.cfg.ns), "max_iter": int(kwargs["max_iter"]), "use_scan": bool(kwargs["use_scan"])})
        idx = len(calls) - 1
        if idx == 1:
            return SolveVmecResidualResult(
                state=state,
                n_iter=199,
                w_history=np.asarray([10.0, 9.0]),
                fsqr2_history=np.asarray([10.0, 9.0]),
                fsqz2_history=np.asarray([0.0, 0.0]),
                fsql2_history=np.asarray([0.0, 0.0]),
                grad_rms_history=np.asarray([], dtype=float),
                step_history=np.asarray([], dtype=float),
                diagnostics={"converged": False, "resume_state": {"time_step": 0.2}},
            )
        fsq = 1.0e-16 if idx >= 2 else 1.0e-3
        return _result(
            state,
            max_iter=int(kwargs["max_iter"]),
            fsq=fsq,
            converged=idx >= 2,
            diagnostics={"use_scan": bool(kwargs["use_scan"]), "ftol": float(kwargs["ftol"])},
        )

    monkeypatch.setenv("VMEC_JAX_DISABLE_JIT_INIT", "1")
    _patch_lightweight_driver_core(monkeypatch)
    monkeypatch.setattr(driver, "solve_fixed_boundary_residual_iter", fake_solver)
    monkeypatch.setattr(solve_module, "solve_fixed_boundary_residual_iter", fake_solver)

    run = run_fixed_boundary(
        input_path,
        solver="vmec2000_iter",
        solver_mode="accelerated",
        step_size=None,
        verbose=False,
        cli_fixed_boundary_mode=True,
    )

    assert calls == [
        {"ns": 5, "max_iter": 1, "use_scan": True},
        {"ns": 9, "max_iter": 200, "use_scan": False},
        {"ns": 9, "max_iter": 250, "use_scan": False},
    ]
    diag = run.result.diagnostics
    assert diag["accelerated_stage_early_switch"] is True
    assert str(diag["accelerated_stage_switch_reason"]).startswith("projected_budget_miss:")
    assert diag["accelerated_stage_effective_mode"] == "parity"
    np.testing.assert_array_equal(diag["accelerated_stage_probe_chunk_iters"], [200])
    assert np.asarray(diag["multigrid_stage_modes"]).tolist() == ["accelerated", "parity"]
    assert diag["converged"] is True


def test_scan_wout_corrector_runs_one_non_scan_step(monkeypatch) -> None:
    calls = []

    def fake_solver(state, static, **kwargs):
        calls.append({"max_iter": int(kwargs["max_iter"]), "use_scan": bool(kwargs["use_scan"])})
        idx = len(calls) - 1
        return _result(
            state,
            max_iter=int(kwargs["max_iter"]),
            fsq=1.0e-16,
            converged=True,
            diagnostics={
                "use_scan": bool(kwargs["use_scan"]),
                "vmec2000_scan": idx == 0,
                "resume_state": {"time_step": 0.1},
                "ftol": float(kwargs["ftol"]),
            },
        )

    monkeypatch.setenv("VMEC_JAX_DISABLE_JIT_INIT", "1")
    _patch_lightweight_driver_core(monkeypatch)
    monkeypatch.setattr(driver, "solve_fixed_boundary_residual_iter", fake_solver)
    monkeypatch.setattr(solve_module, "solve_fixed_boundary_residual_iter", fake_solver)

    run = run_fixed_boundary(
        _example_input(),
        solver="vmec2000_iter",
        solver_mode="accelerated",
        max_iter=3,
        verbose=False,
        multigrid=False,
        scan_wout_corrector=True,
        grid=_small_grid(),
    )

    assert calls == [{"max_iter": 3, "use_scan": True}, {"max_iter": 1, "use_scan": False}]
    assert run.result.diagnostics["scan_wout_corrector"] is True
    assert run.result.diagnostics["scan_wout_corrector_iters"] == 0


def test_run_fixed_boundary_unknown_solver_reports_supported_modes(monkeypatch) -> None:
    _patch_lightweight_driver_core(monkeypatch)

    with pytest.raises(ValueError, match="expected 'gd', 'lbfgs', 'vmec_lbfgs', 'vmec_gn', or 'vmec2000_iter'"):
        run_fixed_boundary(
            _example_input(),
            solver="not-a-solver",
            max_iter=1,
            verbose=False,
            grid=_small_grid(),
        )
