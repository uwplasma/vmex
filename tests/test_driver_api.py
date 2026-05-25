from __future__ import annotations

import builtins
from pathlib import Path

import numpy as np
import pytest

import vmec_jax as vj
import vmec_jax.api as api_module
import vmec_jax.cli as cli_module
import vmec_jax.driver as driver_module
import vmec_jax.solve as solve_module
from vmec_jax.boundary import boundary_from_indata
from vmec_jax.config import load_config
from vmec_jax.driver import (
    example_paths,
    load_example,
    run_free_boundary,
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


def test_host_update_assembly_matches_jax_update_path_lasym():
    root = Path(__file__).resolve().parents[1]
    input_path = root / "examples/data/input.basic_non_stellsym_pressure"
    cfg, indata = load_config(str(input_path))
    grid = vmec_angle_grid(ntheta=10, nzeta=8, nfp=cfg.nfp, lasym=True)
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

    for name in ("Rcos", "Rsin", "Zcos", "Zsin", "Lcos", "Lsin"):
        np.testing.assert_allclose(
            np.asarray(getattr(res_host.state, name)),
            np.asarray(getattr(res_jax.state, name)),
            rtol=1e-12,
            atol=1e-12,
        )
    np.testing.assert_allclose(np.asarray(res_host.w_history), np.asarray(res_jax.w_history), rtol=1e-12, atol=1e-12)


def test_host_update_assembly_driver_default_env_override(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    cfg, _indata = load_config(str(root / "examples/data/input.nfp4_QH_warm_start"))
    cfg_high_work, _indata_high_work = load_config(str(root / "examples/data/input.nfp4_QH_finite_beta"))
    cfg_lasym, _indata_lasym = load_config(str(root / "examples/data/input.basic_non_stellsym_pressure"))

    monkeypatch.delenv("VMEC_JAX_HOST_UPDATE_ASSEMBLY", raising=False)
    monkeypatch.delenv("VMEC_JAX_HOST_UPDATE_CPU_WORK_LIMIT", raising=False)
    assert driver_module._host_update_assembly_driver_default(
        cfg=cfg,
        performance_mode=True,
        backend="cpu",
        use_scan=False,
    )
    assert not driver_module._host_update_assembly_driver_default(
        cfg=cfg,
        performance_mode=True,
        backend="gpu",
        use_scan=False,
    )
    assert not driver_module._host_update_assembly_driver_default(
        cfg=cfg,
        performance_mode=True,
        backend="cpu",
        use_scan=True,
    )
    assert driver_module._host_update_assembly_driver_default(
        cfg=cfg_lasym,
        performance_mode=True,
        backend="cpu",
        use_scan=False,
    )
    assert not driver_module._host_update_assembly_driver_default(
        cfg=cfg_high_work,
        performance_mode=True,
        backend="cpu",
        use_scan=False,
    )

    monkeypatch.setenv("VMEC_JAX_HOST_UPDATE_ASSEMBLY", "0")
    assert not driver_module._host_update_assembly_driver_default(
        cfg=cfg,
        performance_mode=True,
        backend="cpu",
        use_scan=False,
    )
    monkeypatch.delenv("VMEC_JAX_HOST_UPDATE_ASSEMBLY", raising=False)
    monkeypatch.setenv("VMEC_JAX_HOST_UPDATE_CPU_WORK_LIMIT", "999999")
    assert driver_module._host_update_assembly_driver_default(
        cfg=cfg_high_work,
        performance_mode=True,
        backend="cpu",
        use_scan=False,
    )
    monkeypatch.setenv("VMEC_JAX_HOST_UPDATE_ASSEMBLY", "1")
    assert not driver_module._host_update_assembly_driver_default(
        cfg=cfg,
        performance_mode=True,
        backend="cpu",
        use_scan=True,
    )
    assert driver_module._host_update_assembly_driver_default(
        cfg=cfg,
        performance_mode=True,
        backend="cpu",
        use_scan=False,
    )


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


def test_default_backend_and_dynamic_scan_env_fallbacks(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "jax":
            raise RuntimeError("jax unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert driver_module._default_backend_name() == "cpu"

    monkeypatch.setattr(driver_module, "_default_backend_name", lambda: "cpu")
    monkeypatch.setenv("VMEC_JAX_DYNAMIC_SCAN_ITERS", "not-an-int")
    monkeypatch.setenv("VMEC_JAX_DYNAMIC_SCAN_TIMED", "maybe")
    pre_iters, timed_probe, backend = driver_module._dynamic_scan_probe_settings(4)
    assert pre_iters == 3
    assert timed_probe is True
    assert backend == "cpu"

    monkeypatch.setenv("VMEC_JAX_DYNAMIC_SCAN_ITERS", "0")
    monkeypatch.setenv("VMEC_JAX_DYNAMIC_SCAN_TIMED", "0")
    pre_iters, timed_probe, backend = driver_module._dynamic_scan_probe_settings(10)
    assert pre_iters == 1
    assert timed_probe is False
    assert backend == "cpu"


def test_normalize_solver_mode():
    assert driver_module._normalize_solver_mode(solver_mode=None, performance_mode=True) == "default"
    assert driver_module._normalize_solver_mode(solver_mode=None, performance_mode=False) == "parity"
    assert driver_module._normalize_solver_mode(solver_mode="accelerated", performance_mode=False) == "accelerated"
    assert driver_module._normalize_solver_mode(solver_mode="fast", performance_mode=False) == "default"
    assert driver_module._normalize_solver_mode(solver_mode="safe", performance_mode=True) == "parity"
    assert driver_module._normalize_solver_mode(solver_mode="reference", performance_mode=True) == "parity"
    assert driver_module._normalize_solver_mode(solver_mode="perf", performance_mode=False) == "accelerated"
    with pytest.raises(ValueError):
        driver_module._normalize_solver_mode(solver_mode="unknown-mode", performance_mode=True)


def test_driver_scalar_list_and_ftol_helpers():
    class _Input:
        def get_float(self, key, default):
            assert key == "FTOL"
            return 1.0e-9 if default is not None else 1.0e-9

    assert driver_module._as_float_list(None) is None
    assert driver_module._as_float_list(["1.0", 2]) == [1.0, 2.0]
    assert driver_module._as_float_list(object()) is None
    assert driver_module._as_list_like(None) is None
    assert driver_module._as_list_like((1, 2)) == [1, 2]
    assert driver_module._as_list_like(np.asarray([3, 4])).count(3) == 1
    assert driver_module._as_list_like(5) == [5]
    assert driver_module._as_list_like(object()) is None
    assert driver_module._requested_final_ftol(indata=_Input(), ftol_list_input=[1.0e-6, -2.0]) == 0.0
    assert driver_module._requested_final_ftol(indata=_Input(), ftol_list_input=None) == 1.0e-9


def test_default_non_autodiff_solver_policy_matches_fixed_boundary_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(driver_module, "_default_backend_name", lambda: "cpu")
    simple_input = Path(__file__).resolve().parents[1] / "examples/data/input.circular_tokamak"
    _cfg_simple, indata_simple = load_config(simple_input)
    assert driver_module.default_non_autodiff_solver_policy(indata_simple) == ("default", True)

    staged_input = _write_staged_no_niter_input(tmp_path)
    _cfg_staged, indata_staged = load_config(staged_input)
    assert driver_module.default_non_autodiff_solver_policy(indata_staged) == ("parity", False)


def test_default_use_scan_policy_is_backend_and_input_aware():
    simple_input = Path(__file__).resolve().parents[1] / "examples/data/input.circular_tokamak"
    _cfg_simple, indata_simple = load_config(simple_input)
    lasym_input = Path(__file__).resolve().parents[1] / "examples/data/input.up_down_asymmetric_tokamak"
    _cfg_lasym, indata_lasym = load_config(lasym_input)
    finite_beta_input = Path(__file__).resolve().parents[1] / "examples/data/input.nfp4_QH_finite_beta"
    _cfg_finite_beta, indata_finite_beta = load_config(finite_beta_input)

    assert driver_module._default_use_scan_for_backend(indata_simple, "cpu", "default") is False
    assert driver_module._default_use_scan_for_backend(indata_simple, "gpu", "accelerated") is True
    assert driver_module._default_use_scan_for_backend(indata_lasym, "cpu", "accelerated") is False
    assert driver_module._default_use_scan_for_backend(indata_simple, "gpu", "parity") is True
    assert driver_module._default_use_scan_for_backend(indata_finite_beta, "cpu", "accelerated") is True


def test_default_non_autodiff_solver_policy_uses_accelerated_for_gpu(monkeypatch):
    monkeypatch.setattr(driver_module, "_default_backend_name", lambda: "gpu")
    simple_input = Path(__file__).resolve().parents[1] / "examples/data/input.circular_tokamak"
    _cfg_simple, indata_simple = load_config(simple_input)
    assert driver_module.default_non_autodiff_solver_policy(indata_simple) == ("accelerated", True)


def test_default_non_autodiff_solver_policy_keeps_cpu_lasym_accelerated(monkeypatch):
    monkeypatch.setattr(driver_module, "_default_backend_name", lambda: "cpu")
    lasym_input = Path(__file__).resolve().parents[1] / "examples/data/input.up_down_asymmetric_tokamak"
    _cfg_lasym, indata_lasym = load_config(lasym_input)
    assert driver_module.default_non_autodiff_solver_policy(indata_lasym) == ("accelerated", True)


def test_default_non_autodiff_solver_policy_keeps_cpu_current_multigrid_accelerated(monkeypatch):
    monkeypatch.setattr(driver_module, "_default_backend_name", lambda: "cpu")
    qa_input = Path(__file__).resolve().parents[1] / "examples/data/input.LandremanPaul2021_QA_lowres"
    _cfg_qa, indata_qa = load_config(qa_input)
    assert driver_module.default_non_autodiff_solver_policy(indata_qa) == ("accelerated", True)


def test_gpu_lasym_current_staged_solver_device_auto_inherits_gpu_default():
    input_path = Path(__file__).resolve().parents[1] / "examples/data/input.basic_non_stellsym_pressure"
    cfg, indata = load_config(input_path)
    ns_array = driver_module._as_list_like(indata.get("NS_ARRAY", None))
    niter_array = driver_module._as_list_like(indata.get("NITER_ARRAY", None))

    device = driver_module._resolve_fixed_boundary_solver_device_name(
        solver_device=None,
        backend="gpu",
        cfg=cfg,
        indata=indata,
        solver_lower="vmec2000_iter",
        cli_fixed_boundary_mode=True,
        accelerated_mode=True,
        ns_list_input=ns_array,
        niter_list_input=niter_array,
        restart_state_present=False,
        restart_solver_state_present=False,
    )

    assert device is None


def test_gpu_lasym_current_staged_solver_device_default_opts_out():
    input_path = Path(__file__).resolve().parents[1] / "examples/data/input.basic_non_stellsym_pressure"
    cfg, indata = load_config(input_path)

    device = driver_module._resolve_fixed_boundary_solver_device_name(
        solver_device="default",
        backend="gpu",
        cfg=cfg,
        indata=indata,
        solver_lower="vmec2000_iter",
        cli_fixed_boundary_mode=True,
        accelerated_mode=True,
        ns_list_input=driver_module._as_list_like(indata.get("NS_ARRAY", None)),
        niter_list_input=driver_module._as_list_like(indata.get("NITER_ARRAY", None)),
        restart_state_present=False,
        restart_solver_state_present=False,
    )

    assert device is None


def test_gpu_lasym_current_staged_solver_device_cpu_is_explicit():
    input_path = Path(__file__).resolve().parents[1] / "examples/data/input.basic_non_stellsym_pressure"
    cfg, indata = load_config(input_path)

    device = driver_module._resolve_fixed_boundary_solver_device_name(
        solver_device="cpu",
        backend="gpu",
        cfg=cfg,
        indata=indata,
        solver_lower="vmec2000_iter",
        cli_fixed_boundary_mode=True,
        accelerated_mode=True,
        ns_list_input=driver_module._as_list_like(indata.get("NS_ARRAY", None)),
        niter_list_input=driver_module._as_list_like(indata.get("NITER_ARRAY", None)),
        restart_state_present=False,
        restart_solver_state_present=False,
    )

    assert device == "cpu"


def test_gpu_lasym_non_scan_uses_precomputed_tridi_default(monkeypatch):
    input_path = Path(__file__).resolve().parents[1] / "examples/data/input.basic_non_stellsym_pressure"
    cfg, _indata = load_config(input_path)
    monkeypatch.delenv("VMEC_JAX_TRIDI_PRECOMPUTE", raising=False)

    assert (
        driver_module._default_preconditioner_use_precomputed_tridi(
            cfg=cfg,
            backend="gpu",
            performance_mode=True,
            use_scan=False,
        )
        is True
    )
    assert (
        driver_module._default_preconditioner_use_precomputed_tridi(
            cfg=cfg,
            backend="cpu",
            performance_mode=True,
            use_scan=False,
        )
        is None
    )
    assert (
        driver_module._default_preconditioner_use_precomputed_tridi(
            cfg=cfg,
            backend="gpu",
            performance_mode=True,
            use_scan=True,
        )
        is True
    )
    monkeypatch.setenv("VMEC_JAX_TRIDI_PRECOMPUTE", "0")
    assert (
        driver_module._default_preconditioner_use_precomputed_tridi(
            cfg=cfg,
            backend="gpu",
            performance_mode=True,
            use_scan=False,
        )
        is None
    )


def test_low_mode_non_lasym_gpu_keeps_tridi_legacy_default(monkeypatch):
    input_path = Path(__file__).resolve().parents[1] / "examples/data/input.nfp4_QH_warm_start"
    cfg, _indata = load_config(input_path)
    monkeypatch.delenv("VMEC_JAX_TRIDI_PRECOMPUTE", raising=False)

    assert (
        driver_module._default_preconditioner_use_precomputed_tridi(
            cfg=cfg,
            backend="gpu",
            performance_mode=True,
            use_scan=False,
        )
        is None
    )


def test_high_mode_non_lasym_gpu_uses_precomputed_tridi_default(monkeypatch):
    input_path = Path(__file__).resolve().parents[1] / "examples/data/input.nfp4_QH_finite_beta"
    cfg, _indata = load_config(input_path)
    monkeypatch.delenv("VMEC_JAX_TRIDI_PRECOMPUTE", raising=False)

    assert (
        driver_module._default_preconditioner_use_precomputed_tridi(
            cfg=cfg,
            backend="gpu",
            performance_mode=True,
            use_scan=False,
        )
        is True
    )
    assert (
        driver_module._default_preconditioner_use_precomputed_tridi(
            cfg=cfg,
            backend="cpu",
            performance_mode=True,
            use_scan=False,
        )
        is None
    )
    assert (
        driver_module._default_preconditioner_use_precomputed_tridi(
            cfg=cfg,
            backend="gpu",
            performance_mode=True,
            use_scan=True,
        )
        is True
    )


def test_run_fixed_boundary_gpu_lasym_passes_precomputed_tridi_policy(monkeypatch):
    input_path = Path(__file__).resolve().parents[1] / "examples/data/input.basic_non_stellsym_pressure"
    monkeypatch.delenv("VMEC_JAX_TRIDI_PRECOMPUTE", raising=False)
    monkeypatch.setattr(driver_module, "_default_backend_name", lambda: "gpu")
    captured = {}

    def _fake_solver(state, static, **kwargs):
        captured["kwargs"] = dict(kwargs)
        return SolveVmecResidualResult(
            state=state,
            n_iter=1,
            w_history=np.asarray([1.0], dtype=float),
            fsqr2_history=np.asarray([1.0], dtype=float),
            fsqz2_history=np.asarray([0.0], dtype=float),
            fsql2_history=np.asarray([0.0], dtype=float),
            grad_rms_history=np.asarray([], dtype=float),
            step_history=np.asarray([], dtype=float),
            diagnostics={"converged": False, "use_scan": bool(kwargs["use_scan"])},
        )

    monkeypatch.setattr(solve_module, "solve_fixed_boundary_residual_iter", _fake_solver)
    monkeypatch.setattr(driver_module, "solve_fixed_boundary_residual_iter", _fake_solver)

    run_fixed_boundary(
        input_path,
        solver="vmec2000_iter",
        solver_mode="accelerated",
        max_iter=1,
        multigrid=False,
        use_scan=False,
        verbose=False,
    )

    assert captured["kwargs"]["preconditioner_use_precomputed_tridi"] is True


def test_run_fixed_boundary_gpu_high_mode_non_lasym_passes_precomputed_tridi_policy(monkeypatch):
    input_path = Path(__file__).resolve().parents[1] / "examples/data/input.nfp4_QH_finite_beta"
    monkeypatch.delenv("VMEC_JAX_TRIDI_PRECOMPUTE", raising=False)
    monkeypatch.setattr(driver_module, "_default_backend_name", lambda: "gpu")
    captured = {}

    def _fake_solver(state, static, **kwargs):
        captured["kwargs"] = dict(kwargs)
        return SolveVmecResidualResult(
            state=state,
            n_iter=1,
            w_history=np.asarray([1.0], dtype=float),
            fsqr2_history=np.asarray([1.0], dtype=float),
            fsqz2_history=np.asarray([0.0], dtype=float),
            fsql2_history=np.asarray([0.0], dtype=float),
            grad_rms_history=np.asarray([], dtype=float),
            step_history=np.asarray([], dtype=float),
            diagnostics={"converged": False, "use_scan": bool(kwargs["use_scan"])},
        )

    monkeypatch.setattr(solve_module, "solve_fixed_boundary_residual_iter", _fake_solver)
    monkeypatch.setattr(driver_module, "solve_fixed_boundary_residual_iter", _fake_solver)

    run_fixed_boundary(
        input_path,
        solver="vmec2000_iter",
        solver_mode="accelerated",
        max_iter=1,
        multigrid=False,
        use_scan=False,
        verbose=False,
    )

    assert captured["kwargs"]["preconditioner_use_precomputed_tridi"] is True


def test_run_fixed_boundary_gpu_scan_passes_precomputed_tridi_policy(monkeypatch):
    input_path = Path(__file__).resolve().parents[1] / "examples/data/input.nfp4_QH_warm_start"
    monkeypatch.delenv("VMEC_JAX_TRIDI_PRECOMPUTE", raising=False)
    monkeypatch.setattr(driver_module, "_default_backend_name", lambda: "gpu")
    captured = {}

    def _fake_solver(state, static, **kwargs):
        captured["kwargs"] = dict(kwargs)
        return SolveVmecResidualResult(
            state=state,
            n_iter=1,
            w_history=np.asarray([1.0], dtype=float),
            fsqr2_history=np.asarray([1.0], dtype=float),
            fsqz2_history=np.asarray([0.0], dtype=float),
            fsql2_history=np.asarray([0.0], dtype=float),
            grad_rms_history=np.asarray([], dtype=float),
            step_history=np.asarray([], dtype=float),
            diagnostics={"converged": False, "use_scan": bool(kwargs["use_scan"])},
        )

    monkeypatch.setattr(solve_module, "solve_fixed_boundary_residual_iter", _fake_solver)
    monkeypatch.setattr(driver_module, "solve_fixed_boundary_residual_iter", _fake_solver)

    run_fixed_boundary(
        input_path,
        solver="vmec2000_iter",
        solver_mode="accelerated",
        max_iter=1,
        multigrid=False,
        use_scan=True,
        verbose=False,
    )

    assert captured["kwargs"]["use_scan"] is True
    assert captured["kwargs"]["preconditioner_use_precomputed_tridi"] is True


def test_run_fixed_boundary_gpu_auto_policy_uses_scan_and_precomputed_tridi(monkeypatch):
    input_path = Path(__file__).resolve().parents[1] / "examples/data/input.nfp4_QH_warm_start"
    monkeypatch.delenv("VMEC_JAX_TRIDI_PRECOMPUTE", raising=False)
    monkeypatch.setattr(driver_module, "_default_backend_name", lambda: "gpu")
    captured = {}

    def _fake_solver(state, static, **kwargs):
        captured["kwargs"] = dict(kwargs)
        return SolveVmecResidualResult(
            state=state,
            n_iter=1,
            w_history=np.asarray([1.0], dtype=float),
            fsqr2_history=np.asarray([1.0], dtype=float),
            fsqz2_history=np.asarray([0.0], dtype=float),
            fsql2_history=np.asarray([0.0], dtype=float),
            grad_rms_history=np.asarray([], dtype=float),
            step_history=np.asarray([], dtype=float),
            diagnostics={
                "converged": True,
                "use_scan": bool(kwargs["use_scan"]),
                "resume_state": {},
                "light_history": True,
            },
        )

    monkeypatch.setattr(solve_module, "solve_fixed_boundary_residual_iter", _fake_solver)
    monkeypatch.setattr(driver_module, "solve_fixed_boundary_residual_iter", _fake_solver)

    run_fixed_boundary(
        input_path,
        solver="vmec2000_iter",
        max_iter=1,
        multigrid=False,
        verbose=False,
    )

    assert captured["kwargs"]["use_scan"] is True
    assert captured["kwargs"]["preconditioner_use_precomputed_tridi"] is True


def test_run_fixed_boundary_gpu_auto_policy_env_tridi_override_delegates(monkeypatch):
    input_path = Path(__file__).resolve().parents[1] / "examples/data/input.nfp4_QH_warm_start"
    monkeypatch.setenv("VMEC_JAX_TRIDI_PRECOMPUTE", "0")
    monkeypatch.setattr(driver_module, "_default_backend_name", lambda: "gpu")
    captured = {}

    def _fake_solver(state, static, **kwargs):
        captured["kwargs"] = dict(kwargs)
        return SolveVmecResidualResult(
            state=state,
            n_iter=1,
            w_history=np.asarray([1.0], dtype=float),
            fsqr2_history=np.asarray([1.0], dtype=float),
            fsqz2_history=np.asarray([0.0], dtype=float),
            fsql2_history=np.asarray([0.0], dtype=float),
            grad_rms_history=np.asarray([], dtype=float),
            step_history=np.asarray([], dtype=float),
            diagnostics={
                "converged": True,
                "use_scan": bool(kwargs["use_scan"]),
                "resume_state": {},
                "light_history": True,
            },
        )

    monkeypatch.setattr(solve_module, "solve_fixed_boundary_residual_iter", _fake_solver)
    monkeypatch.setattr(driver_module, "solve_fixed_boundary_residual_iter", _fake_solver)

    run_fixed_boundary(
        input_path,
        solver="vmec2000_iter",
        max_iter=1,
        multigrid=False,
        verbose=False,
    )

    assert captured["kwargs"]["use_scan"] is True
    assert captured["kwargs"]["preconditioner_use_precomputed_tridi"] is None


def test_default_non_autodiff_solver_policy_keeps_free_boundary_on_robust_path():
    freeb_input = Path(__file__).resolve().parents[1] / "examples/data/input.cth_like_free_bdy"
    _cfg_freeb, indata_freeb = load_config(freeb_input)
    assert driver_module.default_non_autodiff_solver_policy(indata_freeb) == ("default", True)


def test_run_free_boundary_rejects_fixed_boundary_input():
    input_path = Path(__file__).resolve().parents[1] / "examples/data/input.circular_tokamak"
    with pytest.raises(ValueError, match="not a free-boundary case"):
        run_free_boundary(input_path, verbose=False, use_initial_guess=True)


def test_run_free_boundary_delegates_to_shared_driver(monkeypatch, tmp_path):
    input_path = tmp_path / "input.freeb"
    input_path.write_text(
        "&INDATA\n"
        "  LFREEB = T\n"
        "  MGRID_FILE = 'mgrid_test.nc'\n"
        "/\n"
    )
    captured = {}
    sentinel = object()

    def _fake_run_fixed_boundary(input_path_arg, **kwargs):
        captured["input_path"] = str(input_path_arg)
        captured["kwargs"] = dict(kwargs)
        return sentinel

    monkeypatch.setattr(driver_module, "run_fixed_boundary", _fake_run_fixed_boundary)

    result = run_free_boundary(input_path, verbose=False, max_iter=3)
    assert result is sentinel
    assert captured["input_path"] == str(input_path)
    assert captured["kwargs"]["verbose"] is False
    assert captured["kwargs"]["max_iter"] == 3


def test_run_free_boundary_smoke_on_bundled_small_case():
    input_path = Path(__file__).resolve().parents[1] / "examples/data/input.cth_like_free_bdy_lasym_small"
    run = run_free_boundary(
        input_path,
        use_initial_guess=True,
        vmec_project=False,
        verbose=False,
    )
    assert run.cfg.lfreeb is True
    assert run.state is not None
    assert run.result is None


def test_run_fixed_boundary_keeps_supporting_free_boundary_inputs():
    input_path = Path(__file__).resolve().parents[1] / "examples/data/input.cth_like_free_bdy_lasym_small"
    run = run_fixed_boundary(
        input_path,
        use_initial_guess=True,
        vmec_project=False,
        verbose=False,
    )
    assert run.cfg.lfreeb is True
    assert run.state is not None
    assert run.result is None


def test_public_api_reexports_run_free_boundary():
    from vmec_jax.driver import ExampleData

    assert api_module.ExampleData is ExampleData
    assert api_module.example_paths is example_paths
    assert api_module.run_free_boundary is run_free_boundary
    assert vj.ExampleData is ExampleData
    assert vj.example_paths is example_paths
    assert vj.run_free_boundary is run_free_boundary


def test_public_api_reexports_qi_promotion_helpers():
    from vmec_jax.qi_diagnostics import QISeedSuitabilityTargets, qi_promotion_score
    from vmec_jax.optimization_workflow import QuasiIsodynamicResidualCeiling

    assert api_module.QISeedSuitabilityTargets is QISeedSuitabilityTargets
    assert api_module.qi_promotion_score is qi_promotion_score
    assert api_module.QuasiIsodynamicResidualCeiling is QuasiIsodynamicResidualCeiling
    assert vj.QISeedSuitabilityTargets is QISeedSuitabilityTargets
    assert vj.qi_promotion_score is qi_promotion_score
    assert vj.QuasiIsodynamicResidualCeiling is QuasiIsodynamicResidualCeiling


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
    assert driver_module._accelerated_cli_budgeted_total_iters(total_budget=17, ns_stages=[]) == 17
    assert driver_module._accelerated_cli_budgeted_stage_iters(total_budget=17, ns_stages=[]) == [17]


def test_driver_budget_and_residual_target_helpers():
    result = type(
        "Result",
        (),
        {
            "diagnostics": {"final_fsqr": "1e-8", "final_fsqz": 2.0e-8, "final_fsql": 3.0e-8, "ftol": 3.0e-8},
            "w_history": np.asarray([]),
        },
    )()
    assert driver_module._allocate_integer_budget(total=0, weights=[1, 2]) == [0, 0]
    assert driver_module._allocate_integer_budget(total=5, weights=[]) == []
    assert driver_module._allocate_integer_budget(total=5, weights=[0, 0, 0]) == [0, 0, 5]
    assert sum(driver_module._allocate_integer_budget(total=7, weights=[1, 1, 1])) == 7
    assert driver_module._accelerated_fsq_total_target_from_ftol(1.0e-8) == pytest.approx(3.0e-8)
    assert driver_module._result_final_residuals(result) == (1.0e-8, 2.0e-8, 3.0e-8)
    assert driver_module._result_final_fsq(result) == pytest.approx(6.0e-8)
    assert driver_module._result_meets_requested_ftol(result, ftol=3.0e-8) is True
    assert driver_module._result_meets_requested_ftol(result, ftol=1.0e-8) is False
    assert driver_module._result_hits_total_target(result, fsq_total_target=7.0e-8) is True
    assert driver_module._result_hits_total_target(result, fsq_total_target=5.0e-8) is False
    assert driver_module._result_hits_total_target(None, fsq_total_target=1.0) is False


def test_driver_result_helpers_use_history_and_convergence_flags():
    strict = type("Result", (), {"diagnostics": {"converged_strict": True}})()
    loose = type("Result", (), {"diagnostics": {"converged": True}})()
    history = type(
        "Result",
        (),
        {
            "diagnostics": {},
            "fsqr2_history": np.asarray([4.0, 1.0]),
            "fsqz2_history": np.asarray([5.0, 2.0]),
            "fsql2_history": np.asarray([6.0, 3.0]),
            "w_history": np.asarray([10.0, 0.5]),
        },
    )()

    assert driver_module._result_meets_requested_ftol(strict, ftol=0.0) is True
    assert driver_module._result_meets_requested_ftol(loose, ftol=0.0) is True
    assert driver_module._result_final_residuals(history) == (1.0, 2.0, 3.0)
    assert driver_module._result_final_fsq(history) == 0.5


def test_driver_result_helpers_fall_back_to_diagnostic_histories():
    diag_history = type(
        "Result",
        (),
        {
            "diagnostics": {
                "final_fsqr": "not-a-number",
                "final_fsqz": 2.0,
                "final_fsql": 3.0,
                "fsqr_full": np.asarray([4.0, 1.0]),
                "fsqz_full": np.asarray([5.0, 2.0]),
                "fsql_full": np.asarray([6.0, 3.0]),
                "ftol": 3.0,
            },
            "w_history": np.asarray([]),
        },
    )()
    no_residuals = type("Result", (), {"diagnostics": {"ftol": 1.0}})()
    bad_history = type(
        "Result",
        (),
        {
            "diagnostics": {},
            "fsqr2_history": object(),
            "fsqz2_history": object(),
            "fsql2_history": object(),
        },
    )()

    assert driver_module._result_final_residuals(diag_history) == (1.0, 2.0, 3.0)
    assert driver_module._result_final_fsq(diag_history) == 6.0
    assert driver_module._result_meets_requested_ftol(diag_history, ftol=3.0) is True
    assert driver_module._result_final_residuals(None) is None
    assert driver_module._result_final_residuals(no_residuals) is None
    assert driver_module._result_final_residuals(bad_history) is None
    assert np.isinf(driver_module._result_final_fsq(no_residuals))
    assert driver_module._result_meets_requested_ftol(None, ftol=1.0) is False
    assert driver_module._result_meets_requested_ftol(no_residuals, ftol=1.0) is False


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


def test_cli_defaults_to_cpu_default_on_simple_fixed_boundary(monkeypatch, tmp_path):
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
    assert captured["solver_mode"] == "default"
    assert captured["performance_mode"] is True
    assert captured["use_scan"] is False


def test_cli_defaults_to_cpu_default_on_staged_fixed_boundary_with_niter_array(monkeypatch, tmp_path):
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
    assert captured["solver_mode"] == "default"
    assert captured["performance_mode"] is True
    assert captured["use_scan"] is False


def test_cli_solver_device_cpu_uses_cpu_default_policy(monkeypatch, tmp_path):
    input_path = Path(__file__).resolve().parents[1] / "examples/data/input.nfp4_QH_warm_start"
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

    rc = cli_module.main(
        [
            str(input_path),
            "--solver-device",
            "cpu",
            "--output",
            str(tmp_path / "wout_test.nc"),
            "--quiet",
        ]
    )
    assert rc == 0
    assert captured["solver_mode"] == "default"
    assert captured["solver_device"] == "cpu"
    assert captured["use_scan"] is False


def test_cli_solver_device_gpu_uses_gpu_performance_policy(monkeypatch, tmp_path):
    input_path = Path(__file__).resolve().parents[1] / "examples/data/input.nfp4_QH_warm_start"
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

    rc = cli_module.main(
        [
            str(input_path),
            "--solver-device",
            "gpu",
            "--output",
            str(tmp_path / "wout_test.nc"),
            "--quiet",
        ]
    )
    assert rc == 0
    assert captured["solver_mode"] == "accelerated"
    assert captured["solver_device"] == "gpu"
    assert captured["use_scan"] is True


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


def test_run_fixed_boundary_cli_parity_finisher_caps_explicit_max_iter(monkeypatch, tmp_path):
    input_path = _write_staged_no_niter_input(tmp_path)
    calls = []
    fsq_values = [
        1.0e-2,
        1.0e-3,
        1.0e-4,
        1.0e-5,
        1.0e-6,
        1.0e-7,
        1.0e-8,
    ]

    def _fake_solver(state, static, **kwargs):
        idx = min(len(calls), len(fsq_values) - 1)
        calls.append(
            {
                "ns": int(static.cfg.ns),
                "max_iter": int(kwargs["max_iter"]),
                "use_scan": bool(kwargs["use_scan"]),
                "resume_state": kwargs.get("resume_state", None),
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
                "resume_state": {"cache_norms": np.asarray([1.0])},
                "light_history": True,
                "resume_state_mode": "full",
                "converged": False,
                "marker": len(calls) - 1,
            },
        )

    monkeypatch.setattr(solve_module, "solve_fixed_boundary_residual_iter", _fake_solver)
    monkeypatch.setattr(driver_module, "solve_fixed_boundary_residual_iter", _fake_solver)

    run = run_fixed_boundary(
        input_path,
        solver="vmec2000_iter",
        solver_mode="parity",
        max_iter=100,
        verbose=False,
        cli_fixed_boundary_mode=True,
    )

    assert [call["ns"] for call in calls[:3]] == [5, 9, 13]
    assert [call["max_iter"] for call in calls[:3]] == [34, 33, 33]
    assert [call["max_iter"] for call in calls[3:]] == [100, 100]
    assert all(call["resume_state"] is None for call in calls[3:])
    diag = run.result.diagnostics
    assert np.asarray(diag["cli_fixed_boundary_finish_budgets"]).tolist() == [100, 100]
    assert np.asarray(diag["cli_fixed_boundary_finish_modes"]).tolist() == ["parity", "parity"]
    assert diag["cli_fixed_boundary_finish_budget_cap"] == 200
    assert diag["cli_fixed_boundary_finish_budget_exhausted"] is True
    assert diag["converged"] is False


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


def test_run_fixed_boundary_cli_accelerated_finish_respects_use_scan_false(monkeypatch, tmp_path):
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
                "resume_state": {},
                "light_history": True,
                "converged": converged,
            },
        )

    monkeypatch.setattr(solve_module, "solve_fixed_boundary_residual_iter", _fake_solver)
    monkeypatch.setattr(driver_module, "solve_fixed_boundary_residual_iter", _fake_solver)

    run = run_fixed_boundary(
        input_path,
        solver="vmec2000_iter",
        solver_mode="accelerated",
        use_scan=False,
        verbose=False,
        cli_fixed_boundary_mode=True,
    )

    assert [call["ns"] for call in calls] == [13, 13, 13]
    assert [call["max_iter"] for call in calls] == [100, 100, 100]
    assert [call["use_scan"] for call in calls] == [False, False, False]
    diag = run.result.diagnostics
    assert np.asarray(diag["cli_fixed_boundary_finish_budgets"]).tolist() == [100, 100]
    assert np.asarray(diag["cli_fixed_boundary_finish_modes"]).tolist() == ["accelerated", "accelerated"]
    assert diag["converged"] is True


def test_run_fixed_boundary_cli_accelerated_finish_caps_explicit_max_iter(monkeypatch, tmp_path):
    input_path = _write_single_stage_input(tmp_path)
    calls = []
    fsq_values = [1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7]

    def _fake_solver(state, static, **kwargs):
        idx = min(len(calls), len(fsq_values) - 1)
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
                "converged": False,
            },
        )

    monkeypatch.setattr(solve_module, "solve_fixed_boundary_residual_iter", _fake_solver)
    monkeypatch.setattr(driver_module, "solve_fixed_boundary_residual_iter", _fake_solver)

    run = run_fixed_boundary(
        input_path,
        solver="vmec2000_iter",
        solver_mode="accelerated",
        max_iter=100,
        use_scan=False,
        verbose=False,
        cli_fixed_boundary_mode=True,
    )

    assert [call["ns"] for call in calls] == [13, 13, 13]
    assert [call["max_iter"] for call in calls] == [100, 100, 100]
    assert [call["use_scan"] for call in calls] == [False, False, False]
    diag = run.result.diagnostics
    assert np.asarray(diag["cli_fixed_boundary_finish_budgets"]).tolist() == [100, 100]
    assert np.asarray(diag["cli_fixed_boundary_finish_modes"]).tolist() == ["accelerated", "accelerated"]
    assert diag["cli_fixed_boundary_finish_budget_cap"] == 200
    assert diag["cli_fixed_boundary_finish_budget_exhausted"] is True
    assert diag["converged"] is False


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
    # When solver_mode is explicitly set to "accelerated" (not via CLI auto-
    # detection), the accelerated path uses the single-grid shortcut regardless
    # of whether NS_ARRAY/NITER_ARRAY are present.  The user_explicitly_staged_cli
    # path only activates in the CLI auto-detection code path (no explicit
    # solver_mode arg).
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


def test_run_fixed_boundary_cli_user_explicitly_staged_uses_direct_multigrid(monkeypatch, tmp_path):
    # When CLI mode detects both NS_ARRAY and NITER_ARRAY with multiple stages,
    # the user_explicitly_staged_cli path skips the single-grid shortcut and
    # runs the NS stages directly (matching xvmec2000 behavior).
    input_path = _write_staged_with_niter_input(tmp_path)
    calls = []
    fsq_values = [1.0e-4, 1.0e-2, 1.0e-14]
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

    # Direct multi-stage [5, 9, 13] — no single-grid first attempt.
    assert [call["ns"] for call in calls] == [5, 9, 13]
    assert [call["max_iter"] for call in calls] == [10, 20, 40]
    assert [call["use_scan"] for call in calls] == [True, True, True]
    diag = run.result.diagnostics
    assert diag["cli_fixed_boundary_mode"] is True
    assert diag["cli_fixed_boundary_initial_policy"] == "multigrid"
    # No staged followup — the initial run was already multigrid.
    assert diag.get("cli_fixed_boundary_staged_followup_used", False) is False
    assert diag["converged"] is True


def test_run_fixed_boundary_cli_explicit_staged_followup_runs_for_converged_nonaxis_single_grid(monkeypatch, tmp_path):
    # 3D non-current-driven case (NTOR=1, NCURR=0) with explicit NS_ARRAY+NITER_ARRAY:
    # user_explicitly_staged_cli triggers direct multi-stage execution, matching
    # xvmec2000 behavior (no single-grid first attempt).
    input_path = _write_staged_with_niter_nonaxis_input(tmp_path)
    calls = []
    fsq_values = [1.0e-5, 1.0e-8, 1.0e-14]
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

    # Direct multi-stage [5, 9, 13] — no single-grid first attempt.
    assert [call["ns"] for call in calls] == [5, 9, 13]
    assert [call["max_iter"] for call in calls] == [10, 20, 40]
    diag = run.result.diagnostics
    assert diag["cli_fixed_boundary_initial_policy"] == "multigrid"
    # No staged followup needed — initial path was already multigrid.
    assert diag.get("cli_fixed_boundary_staged_followup_used", False) is False
    assert diag["converged"] is True


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
