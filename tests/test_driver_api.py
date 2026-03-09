from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import vmec_jax.cli as cli_module
import vmec_jax.driver as driver_module
import vmec_jax.solve as solve_module
from vmec_jax.driver import (
    example_paths,
    load_example,
    run_fixed_boundary,
    save_npz,
    wout_from_fixed_boundary_run,
)
from vmec_jax.solve import SolveVmecResidualResult
from vmec_jax.vmec_tomnsp import vmec_angle_grid


def test_example_paths_and_load_example():
    pytest.importorskip("netCDF4")

    input_path, wout_path = example_paths("n3are_R7.75B5.7_lowres", root=Path(__file__).resolve().parents[1])
    assert input_path.exists()
    assert wout_path is not None and wout_path.exists()

    ex = load_example("n3are_R7.75B5.7_lowres", root=Path(__file__).resolve().parents[1], with_wout=True)
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


def test_cli_solver_mode_conflicts_with_fast_flags():
    with pytest.raises(SystemExit):
        cli_module.main(["examples/data/input.circular_tokamak", "--solver-mode", "accelerated", "--fast"])


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


def test_accelerated_single_grid_uses_total_niter_array_budget(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    input_path = root / "examples/data/input.LandremanPaul2021_QA_lowres"
    grid = vmec_angle_grid(ntheta=8, nzeta=8, nfp=2, lasym=False)
    calls = []

    def _fake_solver(state, static, **kwargs):
        calls.append(
            {
                "ns": int(static.cfg.ns),
                "max_iter": int(kwargs["max_iter"]),
                "ftol": float(kwargs["ftol"]),
                "use_scan": bool(kwargs["use_scan"]),
            }
        )
        return SolveVmecResidualResult(
            state=state,
            n_iter=0,
            w_history=np.asarray([1.0], dtype=float),
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
                "converged": False,
            },
        )

    monkeypatch.setattr(solve_module, "solve_fixed_boundary_residual_iter", _fake_solver)

    run = run_fixed_boundary(
        input_path,
        solver="vmec2000_iter",
        solver_mode="accelerated",
        verbose=False,
        grid=grid,
    )

    assert len(calls) == 1
    assert calls[0]["ns"] == int(run.cfg.ns)
    assert calls[0]["max_iter"] == 2600
    assert np.isclose(calls[0]["ftol"], 1.0e-13)
    assert calls[0]["use_scan"] is True
    assert np.asarray(run.result.diagnostics["multigrid_ns_stages"]).tolist() == [50]
    assert np.asarray(run.result.diagnostics["multigrid_niter_stages"]).tolist() == [2600]
    assert np.allclose(np.asarray(run.result.diagnostics["multigrid_ftol_stages"]), np.asarray([1.0e-13]))


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
