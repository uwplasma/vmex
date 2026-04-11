from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import vmec_jax.cli as cli_module
import vmec_jax.driver as driver_module
import vmec_jax.solve as solve_module
from vmec_jax.boundary import boundary_from_indata
from vmec_jax.config import load_config
from vmec_jax.driver import (
    example_paths,
    load_example,
    run_fixed_boundary,
    save_npz,
    wout_from_fixed_boundary_run,
)
from vmec_jax.init_guess import initial_guess_from_boundary
from vmec_jax.solve import SolveVmecResidualResult
from vmec_jax.static import build_static
from vmec_jax.solve import solve_fixed_boundary_residual_iter
from vmec_jax.vmec_tomnsp import vmec_angle_grid


def _write_staged_no_niter_input(tmp_path: Path) -> Path:
    input_path = tmp_path / "input.staged_no_niter"
    input_path.write_text(
        "&INDATA\n"
        "  LFREEB = F\n"
        "  NFP = 1\n"
        "  MPOL = 5\n"
        "  NTOR = 0\n"
        "  NS = 13\n"
        "  NITER = 100\n"
        "  FTOL = 1e-14\n"
        "  NS_ARRAY = 5 9 13\n"
        "  FTOL_ARRAY = 1e-14 1e-14 1e-14\n"
        "  PHIEDGE = 1.0\n"
        "  RBC(0,0) = 1.0\n"
        "  ZBS(1,0) = 0.1\n"
        "/\n"
    )
    return input_path


def _write_staged_with_niter_input(tmp_path: Path) -> Path:
    input_path = tmp_path / "input.staged_with_niter"
    input_path.write_text(
        "&INDATA\n"
        "  LFREEB = F\n"
        "  NFP = 1\n"
        "  MPOL = 5\n"
        "  NTOR = 0\n"
        "  NS = 13\n"
        "  NITER = 100\n"
        "  FTOL = 1e-14\n"
        "  NS_ARRAY = 5 9 13\n"
        "  NITER_ARRAY = 10 20 40\n"
        "  FTOL_ARRAY = 1e-14 1e-14 1e-14\n"
        "  PHIEDGE = 1.0\n"
        "  RBC(0,0) = 1.0\n"
        "  ZBS(1,0) = 0.1\n"
        "/\n"
    )
    return input_path


def _write_staged_with_niter_nonaxis_input(tmp_path: Path) -> Path:
    input_path = tmp_path / "input.staged_with_niter_nonaxis"
    input_path.write_text(
        "&INDATA\n"
        "  LFREEB = F\n"
        "  LASYM = F\n"
        "  NFP = 2\n"
        "  MPOL = 5\n"
        "  NTOR = 1\n"
        "  NS = 13\n"
        "  NITER = 100\n"
        "  FTOL = 1e-14\n"
        "  NS_ARRAY = 5 9 13\n"
        "  NITER_ARRAY = 10 20 40\n"
        "  FTOL_ARRAY = 1e-14 1e-14 1e-14\n"
        "  PHIEDGE = 1.0\n"
        "  RBC(0,0) = 1.0\n"
        "  ZBS(1,0) = 0.1\n"
        "/\n"
    )
    return input_path


def _write_staged_with_niter_nonaxis_current_input(tmp_path: Path) -> Path:
    input_path = tmp_path / "input.staged_with_niter_nonaxis_current"
    input_path.write_text(
        "&INDATA\n"
        "  LFREEB = F\n"
        "  LASYM = F\n"
        "  NFP = 2\n"
        "  MPOL = 5\n"
        "  NTOR = 1\n"
        "  NCURR = 1\n"
        "  NS = 13\n"
        "  NITER = 100\n"
        "  FTOL = 1e-14\n"
        "  NS_ARRAY = 5 9 13\n"
        "  NITER_ARRAY = 10 20 40\n"
        "  FTOL_ARRAY = 1e-14 1e-14 1e-14\n"
        "  PHIEDGE = 1.0\n"
        "  RBC(0,0) = 1.0\n"
        "  ZBS(1,0) = 0.1\n"
        "/\n"
    )
    return input_path


def _write_staged_with_niter_nonaxis_lasym_current_input(tmp_path: Path) -> Path:
    input_path = tmp_path / "input.staged_with_niter_nonaxis_lasym_current"
    input_path.write_text(
        "&INDATA\n"
        "  LFREEB = F\n"
        "  LASYM = T\n"
        "  NFP = 2\n"
        "  MPOL = 5\n"
        "  NTOR = 1\n"
        "  NCURR = 1\n"
        "  NS = 13\n"
        "  NITER = 100\n"
        "  FTOL = 1e-14\n"
        "  NS_ARRAY = 5 9 13\n"
        "  NITER_ARRAY = 10 20 40\n"
        "  FTOL_ARRAY = 1e-14 1e-14 1e-14\n"
        "  PHIEDGE = 1.0\n"
        "  RBC(0,0) = 1.0\n"
        "  ZBS(1,0) = 0.1\n"
        "  RBS(1,0) = 0.02\n"
        "  ZBC(1,0) = 0.01\n"
        "/\n"
    )
    return input_path


def _write_two_stage_nonaxis_current_input(tmp_path: Path) -> Path:
    input_path = tmp_path / "input.two_stage_nonaxis_current"
    input_path.write_text(
        "&INDATA\n"
        "  LFREEB = F\n"
        "  LASYM = F\n"
        "  NFP = 2\n"
        "  MPOL = 5\n"
        "  NTOR = 1\n"
        "  NCURR = 1\n"
        "  NS = 13\n"
        "  NITER = 100\n"
        "  FTOL = 1e-14\n"
        "  NS_ARRAY = 5 13\n"
        "  NITER_ARRAY = 10 40\n"
        "  FTOL_ARRAY = 1e-14 1e-14\n"
        "  PHIEDGE = 1.0\n"
        "  RBC(0,0) = 1.0\n"
        "  ZBS(1,0) = 0.1\n"
        "/\n"
    )
    return input_path


def _write_single_stage_input(tmp_path: Path) -> Path:
    input_path = tmp_path / "input.single_stage"
    input_path.write_text(
        "&INDATA\n"
        "  LFREEB = F\n"
        "  NFP = 1\n"
        "  MPOL = 5\n"
        "  NTOR = 0\n"
        "  NS = 13\n"
        "  NITER = 100\n"
        "  FTOL = 1e-14\n"
        "  NS_ARRAY = 13\n"
        "  FTOL_ARRAY = 1e-14\n"
        "  PHIEDGE = 1.0\n"
        "  RBC(0,0) = 1.0\n"
        "  ZBS(1,0) = 0.1\n"
        "/\n"
    )
    return input_path


@pytest.mark.full
def test_example_paths_and_load_example():
    pytest.importorskip("netCDF4")

    input_path, wout_path = example_paths(
        "LandremanPaul2021_QH_reactorScale_lowres", root=Path(__file__).resolve().parents[1]
    )
    assert input_path.exists()
    assert wout_path is not None and wout_path.exists()

    ex = load_example(
        "LandremanPaul2021_QH_reactorScale_lowres", root=Path(__file__).resolve().parents[1], with_wout=True
    )
    assert ex.cfg.ns > 0
    assert ex.wout is not None
    assert ex.state is not None


def test_save_npz(tmp_path):
    path = save_npz(tmp_path / "demo.npz", a=[1, 2, 3], b=[4, 5, 6])
    assert path.exists()


def test_run_fixed_boundary_initial_guess():
    root = Path(__file__).resolve().parents[1]
    input_path = root / "examples/data/input.circular_tokamak"
    # Keep CI fast: use a small VMEC grid.
    grid = vmec_angle_grid(ntheta=10, nzeta=1, nfp=1, lasym=False)
    run = run_fixed_boundary(
        input_path,
        max_iter=1,
        use_initial_guess=True,
        vmec_project=False,
        verbose=False,
        grid=grid,
    )
    assert run.cfg.ns > 0
    assert run.state is not None
    assert run.result is None

    wout = wout_from_fixed_boundary_run(run, include_fsq=False, fast_bcovar=True)
    assert wout.ns == run.cfg.ns


def test_run_fixed_boundary_returns_current_driven_flux_profiles():
    root = Path(__file__).resolve().parents[1]
    input_path = root / "examples/data/input.basic_non_stellsym_pressure"
    grid = vmec_angle_grid(ntheta=10, nzeta=8, nfp=1, lasym=True)

    run = run_fixed_boundary(
        input_path,
        max_iter=2,
        verbose=False,
        multigrid=False,
        grid=grid,
    )
    wout = wout_from_fixed_boundary_run(run, include_fsq=False, fast_bcovar=True)

    chipf = np.asarray(run.flux.chipf, dtype=float)
    iota = np.asarray(run.profiles["iota"], dtype=float)
    chipf_wout_internal = np.asarray(wout.chipf, dtype=float) / float(2.0 * np.pi * run.signgs)
    assert chipf.shape == np.asarray(wout.chipf).shape
    assert iota.shape == np.asarray(wout.iotas).shape
    assert np.max(np.abs(chipf[1:])) > 0.0
    assert np.max(np.abs(iota[1:])) > 0.0
    np.testing.assert_allclose(chipf, chipf_wout_internal, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(iota, np.asarray(wout.iotas, dtype=float), rtol=1e-10, atol=1e-12)


def test_final_flux_profiles_from_state_supports_traced_lsin():
    pytest.importorskip("jax")

    from vmec_jax._compat import enable_x64, jax, jnp
    from vmec_jax.driver import _final_flux_profiles_from_state, run_fixed_boundary
    from vmec_jax.vmec_tomnsp import vmec_angle_grid

    enable_x64(True)

    root = Path(__file__).resolve().parents[1]
    input_path = root / "examples/data/input.basic_non_stellsym_pressure"
    grid = vmec_angle_grid(ntheta=10, nzeta=8, nfp=1, lasym=True)
    run = run_fixed_boundary(
        input_path,
        max_iter=2,
        verbose=False,
        multigrid=False,
        grid=grid,
    )

    radial_idx = min(5, int(np.asarray(run.state.Lsin).shape[0]) - 1)
    mode_idx = 1

    def scalar(alpha):
        state = type(run.state)(
            layout=run.state.layout,
            Rcos=jnp.asarray(run.state.Rcos),
            Rsin=jnp.asarray(run.state.Rsin),
            Zcos=jnp.asarray(run.state.Zcos),
            Zsin=jnp.asarray(run.state.Zsin),
            Lcos=jnp.asarray(run.state.Lcos),
            Lsin=jnp.asarray(run.state.Lsin).at[radial_idx, mode_idx].add(alpha),
        )
        _flux, prof = _final_flux_profiles_from_state(
            indata=run.indata,
            static_in=run.static,
            state=state,
            signgs=run.signgs,
            flux_local=run.flux,
            prof_local=run.profiles,
            pressure_local=run.profiles["pressure"],
        )
        return jnp.sum(jnp.asarray(prof["iota"]))

    grad = float(jax.grad(scalar)(0.0))
    assert np.isfinite(grad)


def test_host_update_assembly_matches_jax_update_path():
    root = Path(__file__).resolve().parents[1]
    input_path = root / "examples/data/input.LandremanPaul2021_QA_lowres"
    cfg, indata = load_config(str(input_path))
    grid = vmec_angle_grid(ntheta=10, nzeta=8, nfp=cfg.nfp, lasym=False)
    static = build_static(cfg, grid=grid)
    boundary = boundary_from_indata(indata, static.modes)
    state0 = initial_guess_from_boundary(static, boundary, indata, vmec_project=False)

    common = dict(
        indata=indata,
        signgs=-1,
        ftol=float(indata.get_float("FTOL", 1.0e-13)),
        max_iter=1,
        step_size=float(indata.get_float("DELT", 5e-3)),
        include_constraint_force=True,
        apply_m1_constraints=True,
        precond_radial_alpha=0.5,
        precond_lambda_alpha=0.5,
        mode_diag_exponent=0.0,
        auto_flip_force=False,
        divide_by_scalxc_for_update=False,
        lambda_update_scale=1.0,
        enforce_vmec_lambda_axis=True,
        vmec2000_control=True,
        strict_update=True,
        backtracking=False,
        reference_mode=False,
        use_restart_triggers=True,
        vmecpp_restart=False,
        use_direct_fallback=False,
        verbose=False,
        verbose_vmec2000_table=False,
        jit_forces=False,
        use_scan=False,
    )

    res_jax = solve_fixed_boundary_residual_iter(
        state0,
        static,
        host_update_assembly=False,
        **common,
    )
    res_host = solve_fixed_boundary_residual_iter(
        state0,
        static,
        host_update_assembly=True,
        **common,
    )

    np.testing.assert_allclose(np.asarray(res_host.state.Rcos), np.asarray(res_jax.state.Rcos), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(res_host.state.Zsin), np.asarray(res_jax.state.Zsin), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(res_host.state.Lsin), np.asarray(res_jax.state.Lsin), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(res_host.w_history), np.asarray(res_jax.w_history), rtol=1e-12, atol=1e-12)


def test_lasym_performance_mode_infers_axis_for_fast_path():
    root = Path(__file__).resolve().parents[1]
    input_path = root / "examples/data/input.up_down_asymmetric_tokamak"
    perf = run_fixed_boundary(
        input_path,
        use_initial_guess=True,
        verbose=False,
    )
    safe = run_fixed_boundary(
        input_path,
        use_initial_guess=True,
        verbose=False,
        performance_mode=False,
    )
    assert perf.result is None
    assert safe.result is None
    assert float(perf.state.Rcos[0, 0]) > 1.0
    assert abs(float(perf.state.Zcos[0, 0])) > 1e-3
    assert float(safe.state.Rcos[0, 0]) == 0.0
    assert float(safe.state.Zcos[0, 0]) == 0.0


def test_dynamic_scan_probe_settings_cpu(monkeypatch):
    monkeypatch.setattr(driver_module, "_default_backend_name", lambda: "cpu")
    monkeypatch.delenv("VMEC_JAX_DYNAMIC_SCAN_ITERS", raising=False)
    monkeypatch.delenv("VMEC_JAX_DYNAMIC_SCAN_TIMED", raising=False)

    pre_iters, timed_probe, backend = driver_module._dynamic_scan_probe_settings(50)
    assert pre_iters == 10
    assert timed_probe is True
    assert backend == "cpu"


def test_dynamic_scan_probe_settings_accelerator(monkeypatch):
    monkeypatch.setattr(driver_module, "_default_backend_name", lambda: "gpu")
    monkeypatch.delenv("VMEC_JAX_DYNAMIC_SCAN_ITERS", raising=False)
    monkeypatch.delenv("VMEC_JAX_DYNAMIC_SCAN_TIMED", raising=False)

    pre_iters, timed_probe, backend = driver_module._dynamic_scan_probe_settings(50)
    assert pre_iters == 3
    assert timed_probe is False
    assert backend == "gpu"


def test_dynamic_scan_probe_settings_env_override(monkeypatch):
    monkeypatch.setattr(driver_module, "_default_backend_name", lambda: "gpu")
    monkeypatch.setenv("VMEC_JAX_DYNAMIC_SCAN_ITERS", "7")
    monkeypatch.setenv("VMEC_JAX_DYNAMIC_SCAN_TIMED", "1")

    pre_iters, timed_probe, backend = driver_module._dynamic_scan_probe_settings(5)
    assert pre_iters == 4
    assert timed_probe is True
    assert backend == "gpu"


def test_normalize_solver_mode():
    assert driver_module._normalize_solver_mode(solver_mode=None, performance_mode=True) == "default"
    assert driver_module._normalize_solver_mode(solver_mode=None, performance_mode=False) == "parity"
    assert driver_module._normalize_solver_mode(solver_mode="accelerated", performance_mode=False) == "accelerated"
    assert driver_module._normalize_solver_mode(solver_mode="fast", performance_mode=False) == "default"
    with pytest.raises(ValueError):
        driver_module._normalize_solver_mode(solver_mode="unknown-mode", performance_mode=True)


def test_default_non_autodiff_solver_policy_matches_fixed_boundary_defaults(tmp_path):
    simple_input = Path(__file__).resolve().parents[1] / "examples/data/input.circular_tokamak"
    _cfg_simple, indata_simple = load_config(simple_input)
    assert driver_module.default_non_autodiff_solver_policy(indata_simple) == ("accelerated", True)

    staged_input = _write_staged_no_niter_input(tmp_path)
    _cfg_staged, indata_staged = load_config(staged_input)
    assert driver_module.default_non_autodiff_solver_policy(indata_staged) == ("parity", False)


def test_default_non_autodiff_solver_policy_keeps_free_boundary_on_robust_path():
    freeb_input = Path(__file__).resolve().parents[1] / "examples/data/input.cth_like_free_bdy"
    _cfg_freeb, indata_freeb = load_config(freeb_input)
    assert driver_module.default_non_autodiff_solver_policy(indata_freeb) == ("default", True)


def test_python_default_fixed_boundary_uses_optimized_controller(tmp_path):
    input_path = _write_staged_with_niter_input(tmp_path)
    run = run_fixed_boundary(
        input_path,
        solver="vmec2000_iter",
        max_iter=1,
        multigrid=False,
        verbose=False,
    )
    assert bool(run.result.diagnostics.get("cli_fixed_boundary_mode", False)) is True


def test_accelerated_cli_budgeted_stage_iters():
    total = driver_module._accelerated_cli_budgeted_total_iters(total_budget=5000, ns_stages=[16, 49, 100])
    assert total == 2000
    assert driver_module._accelerated_cli_budgeted_stage_iters(total_budget=total, ns_stages=[16, 49, 100]) == [130, 552, 1318]


def test_cli_solver_mode_conflicts_with_fast_flags():
    with pytest.raises(SystemExit):
        cli_module.main(["examples/data/input.circular_tokamak", "--solver-mode", "accelerated", "--fast"])


def test_cli_passes_cli_fixed_boundary_mode(monkeypatch, tmp_path):
    input_path = Path(__file__).resolve().parents[1] / "examples/data/input.circular_tokamak"
    captured = {}

    def _fake_run_fixed_boundary(input_path_arg, **kwargs):
        captured["input_path"] = str(input_path_arg)
        captured.update(kwargs)
        class _Run:
            state = type("S", (), {"Rcos": np.asarray([0.0])})()
            result = None
        return _Run()

    monkeypatch.setattr(cli_module, "run_fixed_boundary", _fake_run_fixed_boundary)
    monkeypatch.setattr(cli_module, "write_wout_from_fixed_boundary_run", lambda *args, **kwargs: None)

    rc = cli_module.main([str(input_path), "--output", str(tmp_path / "wout_test.nc"), "--quiet"])
    assert rc == 0
    assert captured["cli_fixed_boundary_mode"] is True


def test_cli_defaults_to_accelerated_on_simple_fixed_boundary(monkeypatch, tmp_path):
    input_path = Path(__file__).resolve().parents[1] / "examples/data/input.circular_tokamak"
    captured = {}

    def _fake_run_fixed_boundary(input_path_arg, **kwargs):
        captured["input_path"] = str(input_path_arg)
        captured.update(kwargs)

        class _Run:
            state = type("S", (), {"Rcos": np.asarray([0.0])})()
            result = None

        return _Run()

    monkeypatch.setattr(cli_module, "run_fixed_boundary", _fake_run_fixed_boundary)
    monkeypatch.setattr(cli_module, "write_wout_from_fixed_boundary_run", lambda *args, **kwargs: None)

    rc = cli_module.main([str(input_path), "--output", str(tmp_path / "wout_test.nc"), "--quiet"])
    assert rc == 0
    assert captured["solver_mode"] == "accelerated"
    assert captured["performance_mode"] is True


def test_cli_defaults_to_accelerated_on_staged_fixed_boundary_with_niter_array(monkeypatch, tmp_path):
    input_path = _write_staged_with_niter_input(tmp_path)
    captured = {}

    def _fake_run_fixed_boundary(input_path_arg, **kwargs):
        captured["input_path"] = str(input_path_arg)
        captured.update(kwargs)

        class _Run:
            state = type("S", (), {"Rcos": np.asarray([0.0])})()
            result = None

        return _Run()

    monkeypatch.setattr(cli_module, "run_fixed_boundary", _fake_run_fixed_boundary)
    monkeypatch.setattr(cli_module, "write_wout_from_fixed_boundary_run", lambda *args, **kwargs: None)

    rc = cli_module.main([str(input_path), "--output", str(tmp_path / "wout_test.nc"), "--quiet"])
    assert rc == 0
    assert captured["solver_mode"] == "accelerated"
    assert captured["performance_mode"] is True


def test_cli_defaults_to_parity_on_staged_fixed_boundary_without_niter_array(monkeypatch, tmp_path):
    input_path = _write_staged_no_niter_input(tmp_path)
    captured = {}

    def _fake_run_fixed_boundary(input_path_arg, **kwargs):
        captured["input_path"] = str(input_path_arg)
        captured.update(kwargs)

        class _Run:
            state = type("S", (), {"Rcos": np.asarray([0.0])})()
            result = None

        return _Run()

    monkeypatch.setattr(cli_module, "run_fixed_boundary", _fake_run_fixed_boundary)
    monkeypatch.setattr(cli_module, "write_wout_from_fixed_boundary_run", lambda *args, **kwargs: None)

    rc = cli_module.main([str(input_path), "--output", str(tmp_path / "wout_test.nc"), "--quiet"])
    assert rc == 0
    assert captured["solver_mode"] == "parity"
    assert captured["performance_mode"] is False


def test_run_fixed_boundary_accelerated_mode_uses_scan():
    root = Path(__file__).resolve().parents[1]
    input_path = root / "examples/data/input.circular_tokamak"
    grid = vmec_angle_grid(ntheta=10, nzeta=1, nfp=1, lasym=False)

    run = run_fixed_boundary(
        input_path,
        max_iter=6,
        verbose=False,
        multigrid=False,
        grid=grid,
        solver_mode="accelerated",
    )
    diag = run.result.diagnostics
    assert diag["solver_mode"] == "accelerated"
    assert diag["accelerated_mode"] is True
    assert diag["use_scan"] is True
    assert diag["accelerated_scan"] is True
    assert diag["light_history"] is True
    assert diag["resume_state_mode"] == "minimal"
    assert diag["fsq_total_target"] is not None
    assert float(diag["fsq_total_target"]) == driver_module._accelerated_fsq_total_target_from_ftol(1.0e-14)
    assert "vRcc" not in diag["resume_state"]
    assert "cache_precond_diag" not in diag["resume_state"]
    assert np.isfinite(np.asarray(run.result.w_history)).all()
    assert "converged" in diag


def test_run_fixed_boundary_cli_budgeted_multigrid_path(monkeypatch, tmp_path):
    input_path = _write_staged_no_niter_input(tmp_path)
    calls = []

    def _fake_solver(state, static, **kwargs):
        converged = bool(int(static.cfg.ns) == 13 and len(calls) == 3)
        calls.append(
            {
                "ns": int(static.cfg.ns),
                "max_iter": int(kwargs["max_iter"]),
                "use_scan": bool(kwargs["use_scan"]),
                "ftol": float(kwargs["ftol"]),
            }
        )
        return SolveVmecResidualResult(
            state=state,
            n_iter=max(0, int(kwargs["max_iter"]) - 1),
            w_history=np.asarray([max(1.0e-12, 1.0 / max(1, int(static.cfg.ns)))], dtype=float),
            fsqr2_history=np.asarray([1.0], dtype=float),
            fsqz2_history=np.asarray([0.0], dtype=float),
            fsql2_history=np.asarray([0.0], dtype=float),
            grad_rms_history=np.asarray([], dtype=float),
            step_history=np.asarray([], dtype=float),
            diagnostics={
                "use_scan": bool(kwargs["use_scan"]),
                "resume_state": {},
                "light_history": True,
                "resume_state_mode": "minimal",
                "converged": converged,
            },
        )

    monkeypatch.setattr(solve_module, "solve_fixed_boundary_residual_iter", _fake_solver)
    monkeypatch.setattr(driver_module, "solve_fixed_boundary_residual_iter", _fake_solver)

    run = run_fixed_boundary(
        input_path,
        solver="vmec2000_iter",
        solver_mode="accelerated",
        verbose=False,
        cli_fixed_boundary_mode=True,
    )

    assert [call["ns"] for call in calls] == [5, 9, 13, 13]
    assert [call["max_iter"] for call in calls] == [27, 18, 100, 100]
    # All budgeted multigrid stages use accelerated+scan; parity fallback is non-scan.
    assert [call["use_scan"] for call in calls] == [True, True, True, False]
    diag = run.result.diagnostics
    assert diag["cli_fixed_boundary_mode"] is True
    assert diag["cli_accelerated_fixed_policy"] == "budgeted_multigrid"
    assert np.asarray(diag["cli_accelerated_stage_ns"]).tolist() == [5, 9, 13]
    assert np.asarray(diag["cli_accelerated_stage_niter"]).tolist() == [27, 18, 100]
    assert np.asarray(diag["cli_accelerated_stage_modes"]).tolist() == ["accelerated", "accelerated", "accelerated"]
    assert diag["cli_accelerated_final_stage_budget"] == 100
    assert diag["cli_fixed_boundary_initial_policy"] == "budgeted_multigrid"
    assert np.asarray(diag["cli_fixed_boundary_finish_budgets"]).tolist() == [100]
    assert np.asarray(diag["cli_fixed_boundary_finish_converged"]).tolist() == [True]
    assert diag["cli_fixed_boundary_full_parity_fallback"] is False
    assert diag["solver_mode"] == "accelerated"


def test_run_fixed_boundary_cli_parity_finisher_uses_state_only_blocks(monkeypatch, tmp_path):
    input_path = _write_staged_no_niter_input(tmp_path)
    calls = []
    fsq_values = [
        1.0e-2,
        1.0e-3,
        1.0e-4,
        1.0e-5,
        1.0e-6,
        2.0e-6,
        9.0e-7,
        1.1e-6,
        2.0e-14,
    ]

    def _fake_solver(state, static, **kwargs):
        idx = len(calls)
        calls.append(
            {
                "ns": int(static.cfg.ns),
                "max_iter": int(kwargs["max_iter"]),
                "use_scan": bool(kwargs["use_scan"]),
                "resume_state": kwargs.get("resume_state", None),
            }
        )
        fsq = float(fsq_values[idx])
        converged = bool(fsq <= 3.0e-14)
        return SolveVmecResidualResult(
            state=state,
            n_iter=max(0, int(kwargs["max_iter"]) - 1),
            w_history=np.asarray([fsq], dtype=float),
            fsqr2_history=np.asarray([fsq], dtype=float),
            fsqz2_history=np.asarray([0.0], dtype=float),
            fsql2_history=np.asarray([0.0], dtype=float),
            grad_rms_history=np.asarray([], dtype=float),
            step_history=np.asarray([], dtype=float),
            diagnostics={
                "use_scan": bool(kwargs["use_scan"]),
                "resume_state": {"cache_norms": np.asarray([1.0])},
                "light_history": True,
                "resume_state_mode": "full",
                "converged": converged,
                "marker": idx,
            },
        )

    monkeypatch.setattr(solve_module, "solve_fixed_boundary_residual_iter", _fake_solver)
    monkeypatch.setattr(driver_module, "solve_fixed_boundary_residual_iter", _fake_solver)

    run = run_fixed_boundary(
        input_path,
        solver="vmec2000_iter",
        solver_mode="parity",
        verbose=False,
        cli_fixed_boundary_mode=True,
    )

    assert [call["ns"] for call in calls[:3]] == [5, 9, 13]
    assert all(call["max_iter"] == 100 for call in calls[:3])
    assert [call["max_iter"] for call in calls[3:]] == [100, 100, 100, 50, 50, 25]
    assert all(call["resume_state"] is None for call in calls[3:])
    diag = run.result.diagnostics
    assert diag["cli_fixed_boundary_mode"] is True
    assert diag["cli_fixed_boundary_initial_policy"] == "multigrid"
    assert np.asarray(diag["cli_fixed_boundary_finish_budgets"]).tolist() == [100, 100, 100, 50, 50, 25]
    assert np.asarray(diag["cli_fixed_boundary_finish_modes"]).tolist() == ["parity"] * 6
    assert np.asarray(diag["cli_fixed_boundary_finish_converged"]).tolist() == [False, False, False, False, False, True]
    assert diag["cli_fixed_boundary_full_parity_fallback"] is False
    assert diag["converged"] is True
    assert diag["marker"] == 8


def test_run_fixed_boundary_cli_single_grid_uses_accelerated_finish_first(monkeypatch, tmp_path):
    input_path = _write_single_stage_input(tmp_path)
    calls = []
    fsq_values = [1.0e-4, 1.0e-8, 2.0e-14]

    def _fake_solver(state, static, **kwargs):
        idx = len(calls)
        calls.append(
            {
                "ns": int(static.cfg.ns),
                "max_iter": int(kwargs["max_iter"]),
                "use_scan": bool(kwargs["use_scan"]),
                "resume_state_mode": str(kwargs.get("resume_state_mode", "")),
                "fsq_total_target": kwargs.get("fsq_total_target", None),
            }
        )
        fsq = float(fsq_values[idx])
        converged = bool(fsq <= 3.0e-14)
        return SolveVmecResidualResult(
            state=state,
            n_iter=max(0, int(kwargs["max_iter"]) - 1),
            w_history=np.asarray([fsq], dtype=float),
            fsqr2_history=np.asarray([fsq], dtype=float),
            fsqz2_history=np.asarray([0.0], dtype=float),
            fsql2_history=np.asarray([0.0], dtype=float),
            grad_rms_history=np.asarray([], dtype=float),
            step_history=np.asarray([], dtype=float),
            diagnostics={
                "use_scan": bool(kwargs["use_scan"]),
                "resume_state": {"cache_norms": np.asarray([1.0])},
                "light_history": True,
                "resume_state_mode": str(kwargs.get("resume_state_mode", "")),
                "converged": converged,
            },
        )

    monkeypatch.setattr(solve_module, "solve_fixed_boundary_residual_iter", _fake_solver)
    monkeypatch.setattr(driver_module, "solve_fixed_boundary_residual_iter", _fake_solver)

    run = run_fixed_boundary(
        input_path,
        solver="vmec2000_iter",
        solver_mode="accelerated",
        verbose=False,
        cli_fixed_boundary_mode=True,
    )

    assert [call["ns"] for call in calls] == [13, 13, 13]
    assert [call["max_iter"] for call in calls] == [100, 100, 100]
    # Scan is now enabled for CLI CPU runs: benchmarks show lax.scan with the JAX
    # compilation disk cache is 2-2.5× faster than the NumPy hot-path for CLI use.
    assert [call["use_scan"] for call in calls] == [True, True, True]
    assert [call["resume_state_mode"] for call in calls] == ["minimal", "minimal", "minimal"]
    # Single-grid: only stage is the final stage, so fsq_total_target=None for
    # exact per-component convergence matching VMEC2000. Retry attempts (mode=
    # "accelerated") still use a total target for their own early-exit heuristic.
    assert calls[0]["fsq_total_target"] is None
    assert calls[1]["fsq_total_target"] is not None
    assert calls[2]["fsq_total_target"] is not None
    diag = run.result.diagnostics
    assert diag["cli_fixed_boundary_mode"] is True
    assert diag["cli_fixed_boundary_initial_policy"] == "single_grid"
    assert np.asarray(diag["cli_fixed_boundary_finish_budgets"]).tolist() == [100, 100]
    assert np.asarray(diag["cli_fixed_boundary_finish_modes"]).tolist() == ["accelerated", "accelerated"]
    assert np.asarray(diag["cli_fixed_boundary_finish_converged"]).tolist() == [False, True]
    assert diag["cli_fixed_boundary_full_parity_fallback"] is False
    assert diag["converged"] is True


def test_run_fixed_boundary_cli_single_grid_requires_strict_ftol(monkeypatch, tmp_path):
    input_path = _write_single_stage_input(tmp_path)
    calls = []
    residuals = [
        (2.50e-14, 2.0e-16, 2.0e-16),
        (1.40e-14, 4.0e-15, 4.0e-15),
        (8.00e-15, 7.0e-15, 6.0e-15),
    ]

    def _fake_solver(state, static, **kwargs):
        idx = len(calls)
        fsqr, fsqz, fsql = residuals[idx]
        calls.append(
            {
                "ns": int(static.cfg.ns),
                "max_iter": int(kwargs["max_iter"]),
                "use_scan": bool(kwargs["use_scan"]),
                "resume_state_mode": str(kwargs.get("resume_state_mode", "")),
            }
        )
        w = float(fsqr + fsqz + fsql)
        return SolveVmecResidualResult(
            state=state,
            n_iter=max(0, int(kwargs["max_iter"]) - 1),
            w_history=np.asarray([w], dtype=float),
            fsqr2_history=np.asarray([fsqr], dtype=float),
            fsqz2_history=np.asarray([fsqz], dtype=float),
            fsql2_history=np.asarray([fsql], dtype=float),
            grad_rms_history=np.asarray([], dtype=float),
            step_history=np.asarray([], dtype=float),
            diagnostics={
                "use_scan": bool(kwargs["use_scan"]),
                "resume_state": {"cache_norms": np.asarray([1.0])},
                "light_history": True,
                "resume_state_mode": str(kwargs.get("resume_state_mode", "")),
                "ftol": 1.0e-14,
                "converged": bool(w <= 3.0e-14),
            },
        )

    monkeypatch.setattr(solve_module, "solve_fixed_boundary_residual_iter", _fake_solver)
    monkeypatch.setattr(driver_module, "solve_fixed_boundary_residual_iter", _fake_solver)

    run = run_fixed_boundary(
        input_path,
        solver="vmec2000_iter",
        solver_mode="accelerated",
        verbose=False,
        cli_fixed_boundary_mode=True,
    )

    assert [call["ns"] for call in calls] == [13, 13, 13]
    assert [call["max_iter"] for call in calls] == [100, 100, 100]
    # Scan is now the default for CLI CPU runs (faster with warm JAX disk cache).
    assert [call["use_scan"] for call in calls] == [True, True, True]
    diag = run.result.diagnostics
    assert np.asarray(diag["cli_fixed_boundary_finish_budgets"]).tolist() == [100, 100]
    assert np.asarray(diag["cli_fixed_boundary_finish_modes"]).tolist() == ["accelerated", "accelerated"]
    assert np.asarray(diag["cli_fixed_boundary_finish_converged"]).tolist() == [False, True]
    assert diag["converged"] is True
    assert diag["converged_strict"] is True
    assert float(diag["final_fsqr"]) <= 1.0e-14
    assert float(diag["final_fsqz"]) <= 1.0e-14
    assert float(diag["final_fsql"]) <= 1.0e-14


def test_run_fixed_boundary_accelerated_mode_defaults_to_single_grid():
    root = Path(__file__).resolve().parents[1]
    input_path = root / "examples/data/input.LandremanSenguptaPlunk_section5p3_low_res"

    run_acc = run_fixed_boundary(
        input_path,
        max_iter=1,
        verbose=False,
        solver_mode="accelerated",
    )
    run_parity = run_fixed_boundary(
        input_path,
        max_iter=1,
        verbose=False,
        solver_mode="parity",
    )

    diag_acc = run_acc.result.diagnostics
    diag_parity = run_parity.result.diagnostics

    assert diag_acc["multigrid_user_provided"] is False
    assert diag_acc["accelerated_single_grid_default"] is True
    assert np.asarray(diag_acc["multigrid_ns_stages"]).tolist() == [int(run_acc.cfg.ns)]
    assert diag_parity["accelerated_single_grid_default"] is False
    assert np.asarray(diag_parity["multigrid_ns_stages"]).tolist() == [11, 25]


def test_run_fixed_boundary_cli_explicit_staged_followup_after_single_grid_miss(monkeypatch, tmp_path):
    input_path = _write_staged_with_niter_input(tmp_path)
    calls = []
    fsq_values = [1.0e-4, 1.0e-2, 1.0e-4, 2.0e-14]
    converged_flags = [False, False, False, True]

    def _fake_solver(state, static, **kwargs):
        idx = len(calls)
        calls.append(
            {
                "ns": int(static.cfg.ns),
                "max_iter": int(kwargs["max_iter"]),
                "use_scan": bool(kwargs["use_scan"]),
            }
        )
        fsq = float(fsq_values[idx])
        return SolveVmecResidualResult(
            state=state,
            n_iter=max(0, int(kwargs["max_iter"]) - 1),
            w_history=np.asarray([fsq], dtype=float),
            fsqr2_history=np.asarray([fsq], dtype=float),
            fsqz2_history=np.asarray([0.0], dtype=float),
            fsql2_history=np.asarray([0.0], dtype=float),
            grad_rms_history=np.asarray([], dtype=float),
            step_history=np.asarray([], dtype=float),
            diagnostics={
                "use_scan": bool(kwargs["use_scan"]),
                "resume_state": {},
                "light_history": True,
                "resume_state_mode": "minimal",
                "converged": bool(converged_flags[idx]),
            },
        )

    monkeypatch.setattr(solve_module, "solve_fixed_boundary_residual_iter", _fake_solver)
    monkeypatch.setattr(driver_module, "solve_fixed_boundary_residual_iter", _fake_solver)

    run = run_fixed_boundary(
        input_path,
        solver="vmec2000_iter",
        solver_mode="accelerated",
        verbose=False,
        cli_fixed_boundary_mode=True,
    )

    assert [call["ns"] for call in calls] == [13, 5, 9, 13]
    assert [call["max_iter"] for call in calls] == [70, 10, 20, 40]
    # All stages now use accelerated (scan) mode; parity fallback is only used as last resort.
    assert [call["use_scan"] for call in calls] == [True, True, True, True]
    diag = run.result.diagnostics
    assert diag["cli_fixed_boundary_mode"] is True
    assert diag["cli_fixed_boundary_initial_policy"] == "single_grid"
    assert diag["cli_fixed_boundary_staged_followup_used"] is True
    assert diag["cli_fixed_boundary_staged_followup_policy"] == "input_multigrid"
    assert np.asarray(diag["cli_fixed_boundary_staged_followup_ns"]).tolist() == [5, 9, 13]
    assert np.asarray(diag["cli_fixed_boundary_staged_followup_niter"]).tolist() == [10, 20, 40]
    assert np.asarray(diag["cli_fixed_boundary_staged_followup_modes"]).tolist() == [
        "accelerated",
        "accelerated",
        "accelerated",
    ]
    assert np.asarray(diag["cli_fixed_boundary_finish_budgets"]).tolist() == []
    assert diag["converged"] is True


def test_run_fixed_boundary_cli_explicit_staged_followup_runs_for_converged_nonaxis_single_grid(monkeypatch, tmp_path):
    input_path = _write_staged_with_niter_nonaxis_input(tmp_path)
    calls = []
    fsq_values = [2.0e-14, 1.0e-5, 1.0e-8, 1.0e-14]
    converged_flags = [True, False, False, True]

    def _fake_solver(state, static, **kwargs):
        idx = len(calls)
        calls.append(
            {
                "ns": int(static.cfg.ns),
                "max_iter": int(kwargs["max_iter"]),
                "use_scan": bool(kwargs["use_scan"]),
            }
        )
        fsq = float(fsq_values[idx])
        return SolveVmecResidualResult(
            state=state,
            n_iter=max(0, int(kwargs["max_iter"]) - 1),
            w_history=np.asarray([fsq], dtype=float),
            fsqr2_history=np.asarray([fsq], dtype=float),
            fsqz2_history=np.asarray([0.0], dtype=float),
            fsql2_history=np.asarray([0.0], dtype=float),
            grad_rms_history=np.asarray([], dtype=float),
            step_history=np.asarray([], dtype=float),
            diagnostics={
                "use_scan": bool(kwargs["use_scan"]),
                "resume_state": {},
                "light_history": True,
                "resume_state_mode": "minimal",
                "converged": bool(converged_flags[idx]),
            },
        )

    monkeypatch.setattr(solve_module, "solve_fixed_boundary_residual_iter", _fake_solver)
    monkeypatch.setattr(driver_module, "solve_fixed_boundary_residual_iter", _fake_solver)

    run = run_fixed_boundary(
        input_path,
        solver="vmec2000_iter",
        solver_mode="accelerated",
        verbose=False,
        cli_fixed_boundary_mode=True,
    )

    assert [call["ns"] for call in calls] == [13, 5, 9, 13]
    assert [call["max_iter"] for call in calls] == [70, 10, 20, 40]
    diag = run.result.diagnostics
    assert diag["cli_fixed_boundary_initial_policy"] == "single_grid"
    assert diag["cli_fixed_boundary_staged_followup_used"] is True
    assert diag["cli_fixed_boundary_staged_followup_policy"] == "input_multigrid"
    assert np.asarray(diag["cli_fixed_boundary_staged_followup_ns"]).tolist() == [5, 9, 13]
    assert np.asarray(diag["cli_fixed_boundary_staged_followup_niter"]).tolist() == [10, 20, 40]
    assert np.asarray(diag["cli_fixed_boundary_staged_followup_modes"]).tolist() == [
        "parity",
        "accelerated",
        "accelerated",
    ]


def test_run_fixed_boundary_cli_current_driven_nonaxis_uses_direct_multigrid(monkeypatch, tmp_path):
    input_path = _write_staged_with_niter_nonaxis_current_input(tmp_path)
    calls = []
    fsq_values = [1.0e-8, 1.0e-10, 1.0e-14]
    converged_flags = [False, False, True]

    def _fake_solver(state, static, **kwargs):
        idx = len(calls)
        calls.append(
            {
                "ns": int(static.cfg.ns),
                "max_iter": int(kwargs["max_iter"]),
                "use_scan": bool(kwargs["use_scan"]),
            }
        )
        fsq = float(fsq_values[idx])
        return SolveVmecResidualResult(
            state=state,
            n_iter=max(0, int(kwargs["max_iter"]) - 1),
            w_history=np.asarray([fsq], dtype=float),
            fsqr2_history=np.asarray([fsq], dtype=float),
            fsqz2_history=np.asarray([0.0], dtype=float),
            fsql2_history=np.asarray([0.0], dtype=float),
            grad_rms_history=np.asarray([], dtype=float),
            step_history=np.asarray([], dtype=float),
            diagnostics={
                "use_scan": bool(kwargs["use_scan"]),
                "resume_state": {},
                "light_history": True,
                "resume_state_mode": "minimal",
                "converged": bool(converged_flags[idx]),
            },
        )

    monkeypatch.setattr(solve_module, "solve_fixed_boundary_residual_iter", _fake_solver)
    monkeypatch.setattr(driver_module, "solve_fixed_boundary_residual_iter", _fake_solver)

    run = run_fixed_boundary(
        input_path,
        solver="vmec2000_iter",
        solver_mode="accelerated",
        verbose=False,
        cli_fixed_boundary_mode=True,
    )

    assert [call["ns"] for call in calls] == [5, 9, 13]
    assert [call["max_iter"] for call in calls] == [10, 20, 40]
    # Scan is enabled for current_driven_3d_cli on CPU (benchmarks show lax.scan
    # is faster than the Python-loop NumPy hot-path for this case).
    assert [call["use_scan"] for call in calls] == [True, True, True]
    diag = run.result.diagnostics
    assert diag["cli_fixed_boundary_initial_policy"] == "multigrid"
    assert np.asarray(diag["multigrid_stage_modes"]).tolist() == [
        "accelerated",
        "accelerated",
        "accelerated",
    ]


def test_run_fixed_boundary_cli_two_stage_current_driven_nonaxis_uses_multigrid(monkeypatch, tmp_path):
    input_path = _write_two_stage_nonaxis_current_input(tmp_path)
    calls = []
    fsq_values = [1.0e-8, 1.0e-14]
    converged_flags = [False, True]

    def _fake_solver(state, static, **kwargs):
        idx = len(calls)
        calls.append(
            {
                "ns": int(static.cfg.ns),
                "max_iter": int(kwargs["max_iter"]),
                "use_scan": bool(kwargs["use_scan"]),
            }
        )
        fsq = float(fsq_values[idx])
        return SolveVmecResidualResult(
            state=state,
            n_iter=max(0, int(kwargs["max_iter"]) - 1),
            w_history=np.asarray([fsq], dtype=float),
            fsqr2_history=np.asarray([fsq], dtype=float),
            fsqz2_history=np.asarray([0.0], dtype=float),
            fsql2_history=np.asarray([0.0], dtype=float),
            grad_rms_history=np.asarray([], dtype=float),
            step_history=np.asarray([], dtype=float),
            diagnostics={
                "use_scan": bool(kwargs["use_scan"]),
                "resume_state": {},
                "light_history": True,
                "resume_state_mode": "minimal",
                "converged": bool(converged_flags[idx]),
            },
        )

    monkeypatch.setattr(solve_module, "solve_fixed_boundary_residual_iter", _fake_solver)
    monkeypatch.setattr(driver_module, "solve_fixed_boundary_residual_iter", _fake_solver)

    run = run_fixed_boundary(
        input_path,
        solver="vmec2000_iter",
        solver_mode="accelerated",
        verbose=False,
        cli_fixed_boundary_mode=True,
    )

    assert [call["ns"] for call in calls] == [5, 13]
    assert [call["max_iter"] for call in calls] == [10, 40]
    diag = run.result.diagnostics
    assert diag["cli_fixed_boundary_initial_policy"] == "multigrid"


def test_run_fixed_boundary_cli_three_stage_lasym_current_driven_nonaxis_uses_multigrid(
    monkeypatch, tmp_path
):
    input_path = _write_staged_with_niter_nonaxis_lasym_current_input(tmp_path)
    calls = []
    fsq_values = [1.0e-8, 1.0e-10, 1.0e-14]
    converged_flags = [False, False, True]

    def _fake_solver(state, static, **kwargs):
        idx = len(calls)
        calls.append(
            {
                "ns": int(static.cfg.ns),
                "max_iter": int(kwargs["max_iter"]),
                "use_scan": bool(kwargs["use_scan"]),
            }
        )
        fsq = float(fsq_values[idx])
        return SolveVmecResidualResult(
            state=state,
            n_iter=max(0, int(kwargs["max_iter"]) - 1),
            w_history=np.asarray([fsq], dtype=float),
            fsqr2_history=np.asarray([fsq], dtype=float),
            fsqz2_history=np.asarray([0.0], dtype=float),
            fsql2_history=np.asarray([0.0], dtype=float),
            grad_rms_history=np.asarray([], dtype=float),
            step_history=np.asarray([], dtype=float),
            diagnostics={
                "use_scan": bool(kwargs["use_scan"]),
                "resume_state": {},
                "light_history": True,
                "resume_state_mode": "minimal",
                "converged": bool(converged_flags[idx]),
            },
        )

    monkeypatch.setattr(solve_module, "solve_fixed_boundary_residual_iter", _fake_solver)
    monkeypatch.setattr(driver_module, "solve_fixed_boundary_residual_iter", _fake_solver)

    run = run_fixed_boundary(
        input_path,
        solver="vmec2000_iter",
        solver_mode="accelerated",
        verbose=False,
        cli_fixed_boundary_mode=True,
    )

    assert [call["ns"] for call in calls] == [5, 9, 13]
    assert [call["max_iter"] for call in calls] == [10, 20, 40]
    assert [call["use_scan"] for call in calls] == [False, False, False]
    diag = run.result.diagnostics
    assert diag["cli_fixed_boundary_initial_policy"] == "multigrid"
    assert "cli_fixed_boundary_staged_followup_used" not in diag
    assert np.asarray(diag["multigrid_stage_modes"]).tolist() == ["parity", "parity", "parity"]


def test_vmec2000_iter_histories_materialize_numeric_arrays():
    root = Path(__file__).resolve().parents[1]
    input_path = root / "examples/data/input.circular_tokamak"
    grid = vmec_angle_grid(ntheta=10, nzeta=1, nfp=1, lasym=False)

    run = run_fixed_boundary(
        input_path,
        max_iter=2,
        verbose=False,
        performance_mode=False,
        multigrid=False,
        grid=grid,
    )
    diag = run.result.diagnostics

    for key in (
        "update_rms_history",
        "fsq1_history",
        "fsqr1_history",
        "fsqz1_history",
        "fsql1_history",
        "rz_norm_history",
        "f_norm1_history",
        "gcr2_p_history",
        "gcz2_p_history",
        "gcl2_p_history",
    ):
        arr = np.asarray(diag[key])
        assert arr.ndim == 1
        assert arr.dtype.kind == "f"

    assert diag["update_rms_history"].shape == diag["dt_eff_history"].shape
