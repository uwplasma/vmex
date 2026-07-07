from __future__ import annotations

import builtins
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import vmec_jax.driver as driver
from vmec_jax.free_boundary import MGridMetadata, PreparedMGrid
from vmec_jax.kernels.tomnsp import vmec_angle_grid
from vmec_jax.solve import SolveVmecResidualResult


ROOT = Path(__file__).resolve().parents[2]


def _result(*, n: int = 3, bad_fsql: bool = False):
    fsql = np.asarray([0.3, 0.2, 0.1], dtype=object if bad_fsql else float)
    if bad_fsql:
        fsql[-1] = object()
    return SolveVmecResidualResult(
        state=object(),
        n_iter=n - 1,
        w_history=np.linspace(1.0, 0.1, n),
        fsqr2_history=np.linspace(0.4, 0.2, n),
        fsqz2_history=np.linspace(0.3, 0.1, n),
        fsql2_history=fsql,
        grad_rms_history=np.zeros((0,), dtype=float),
        step_history=np.zeros((0,), dtype=float),
        diagnostics={"converged": True},
    )


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


def test_example_paths_reports_missing_wout_as_none(tmp_path):
    data_dir = tmp_path / "examples" / "data"
    data_dir.mkdir(parents=True)
    input_path = data_dir / "input.synthetic"
    input_path.write_text("&INDATA\n/\n")

    actual_input, actual_wout = driver.example_paths("synthetic", root=tmp_path)

    assert actual_input == input_path
    assert actual_wout is None


def test_load_example_without_wout_skips_optional_wout_read():
    example = driver.load_example("circular_tokamak", root=ROOT, with_wout=False)

    assert example.input_path.exists()
    if example.wout_path is not None:
        assert example.wout_path.name in {"wout_circular_tokamak_reference.nc", "wout_circular_tokamak.nc"}
    assert example.wout is None
    assert example.state is None
    assert example.static.cfg.ns == example.cfg.ns


def test_save_npz_creates_parent_and_preserves_arrays(tmp_path):
    path = driver.save_npz(tmp_path / "nested" / "demo.npz", a=np.asarray([1, 2, 3]), b=np.asarray([[4.0], [5.0]]))

    with np.load(path) as data:
        np.testing.assert_array_equal(data["a"], np.asarray([1, 2, 3]))
        np.testing.assert_allclose(data["b"], np.asarray([[4.0], [5.0]]))


def test_run_fixed_boundary_initial_guess_verbose_vmec2000_mode(capsys):
    grid = vmec_angle_grid(ntheta=8, nzeta=1, nfp=1, lasym=False)

    run = driver.run_fixed_boundary(
        ROOT / "examples" / "data" / "input.circular_tokamak",
        solver="vmec2000_iter",
        max_iter=1,
        use_initial_guess=True,
        vmec_project=False,
        verbose=True,
        grid=grid,
    )

    out = capsys.readouterr().out
    assert "fixed-boundary run (initial guess)" in out
    assert "max_iter=" not in out
    assert run.result is None
    assert run.state is not None


def test_run_fixed_boundary_unknown_solver_raises_before_solver_dispatch():
    grid = vmec_angle_grid(ntheta=8, nzeta=1, nfp=1, lasym=False)

    with pytest.raises(ValueError, match="Unknown solver"):
        driver.run_fixed_boundary(
            ROOT / "examples" / "data" / "input.circular_tokamak",
            solver="bogus",
            max_iter=1,
            ns_override=3,
            vmec_project=False,
            verbose=False,
            grid=grid,
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


def test_list_coercion_and_resume_sanitizers_cover_numpy_scalar_and_bad_step_edges():
    assert driver._as_list_like(np.asarray([1, 2, 3])) == [1, 2, 3]
    assert driver._as_list_like(np.float64(2.5)) == [np.float64(2.5)]
    assert driver._as_list_like((v for v in [4, 5])) == [4, 5]

    resume_state = {"time_step": "0.25"}
    cross_grid = driver._sanitize_resume_state_for_grid_change(resume_state, step_size=object())
    same_grid = driver._sanitize_resume_state_for_same_grid(resume_state, step_size=object())

    assert cross_grid["time_step"] == pytest.approx(0.25)
    assert same_grid["time_step"] == pytest.approx(0.25)
    assert same_grid["inv_tau"] == [pytest.approx(0.6)] * 10
    assert driver._sanitize_resume_state_for_grid_change(None, step_size=1.0) is None
    assert driver._sanitize_resume_state_for_same_grid({}, step_size=1.0) is None


def test_stage_switch_reason_covers_zero_and_nonpositive_decay_edges():
    assert (
        driver._stage_switch_reason_from_progress(
            start_total_fsq=0.0,
            best_total_fsq=-1.0,
            target_total_fsq=-2.0,
            chunk_iters=4,
            remaining_budget=3,
        )
        is None
    )
    assert (
        driver._stage_switch_reason_from_progress(
            start_total_fsq=1.0,
            best_total_fsq=0.0,
            target_total_fsq=-1.0,
            chunk_iters=4,
            remaining_budget=3,
        )
        is None
    )


def test_result_residual_convergence_and_budget_helpers_cover_fallbacks():
    assert driver._result_final_residuals(None) is None
    explicit = SimpleNamespace(diagnostics={"final_fsqr": "1.0", "final_fsqz": 2.0, "final_fsql": np.float64(3.0)})
    assert driver._result_final_residuals(explicit) == (1.0, 2.0, 3.0)

    history = SimpleNamespace(
        diagnostics={"final_fsqr": object(), "final_fsqz": 2.0, "final_fsql": 3.0, "requested_ftol": 10.0},
        fsqr2_history=np.asarray([4.0, 5.0]),
        fsqz2_history=np.asarray([6.0, 7.0]),
        fsql2_history=np.asarray([8.0, 9.0]),
    )
    assert driver._result_final_residuals(history) == (5.0, 7.0, 9.0)

    diag_history = SimpleNamespace(
        diagnostics={
            "fsqr_full": np.asarray([1.25]),
            "fsqz_full": np.asarray([2.5]),
            "fsql_full": np.asarray([3.75]),
        },
        fsqr2_history=np.asarray([object()], dtype=object),
        fsqz2_history=np.asarray([1.0]),
        fsql2_history=np.asarray([1.0]),
    )
    assert driver._result_final_residuals(diag_history) == (1.25, 2.5, 3.75)

    no_residuals = SimpleNamespace(
        diagnostics={"fsqr_full": np.asarray([object()], dtype=object), "fsqz_full": [1.0], "fsql_full": [1.0]},
        fsqr2_history=np.asarray([object()], dtype=object),
        fsqz2_history=np.asarray([1.0]),
        fsql2_history=np.asarray([1.0]),
    )
    assert driver._result_final_residuals(no_residuals) is None
    assert driver._result_final_fsq(None) == np.inf
    assert driver._result_final_fsq(no_residuals) == np.inf

    assert driver._result_meets_requested_ftol(SimpleNamespace(diagnostics={"converged_strict": True}), ftol=0.0)
    assert not driver._result_meets_requested_ftol(SimpleNamespace(diagnostics={"converged_strict": False}), ftol=1.0)
    assert driver._result_meets_requested_ftol(SimpleNamespace(diagnostics={"converged": True}), ftol=0.0)
    assert not driver._result_meets_requested_ftol(no_residuals, ftol=1.0)
    assert driver._result_meets_requested_ftol(history, ftol=10.0)
    assert not driver._result_meets_requested_ftol(history, ftol=1.0)
    assert not driver._result_hits_total_target(None, fsq_total_target=1.0)
    assert not driver._result_hits_total_target(history, fsq_total_target=None)
    assert driver._result_hits_total_target(history, fsq_total_target=100.0)

    assert driver._allocate_integer_budget(total=0, weights=[1, 2]) == [0, 0]
    assert driver._allocate_integer_budget(total=5, weights=[]) == []
    assert driver._allocate_integer_budget(total=5, weights=[0, -1, 0]) == [0, 0, 5]
    assert sum(driver._allocate_integer_budget(total=7, weights=[1, 2, 3])) == 7
    assert driver._accelerated_cli_budgeted_total_iters(total_budget=10, ns_stages=[]) == 10
    assert driver._accelerated_cli_budgeted_total_iters(total_budget=100, ns_stages=[25, 100]) == 50
    assert driver._accelerated_cli_budgeted_stage_iters(total_budget=8, ns_stages=[]) == [8]
    assert driver._accelerated_cli_budgeted_stage_iters(total_budget=1, ns_stages=[8, 9])[-1] == 1
    assert driver._distribute_stage_iters(iters=0, nstep=4) == [0]
    assert driver._distribute_stage_iters(iters=3, nstep=1) == [3]
    assert driver._distribute_stage_iters(iters=2, nstep=5) == [2]
    assert driver._distribute_stage_iters(iters=7, nstep=3) == [3, 2, 2]


def test_stage_chunk_result_merging_preserves_histories_and_terminal_metadata():
    first = _result(n=2)
    first = replace(
        first,
        diagnostics={
            "step_status_history": np.asarray(["accepted"], dtype=object),
            "fsq_prev_history": np.asarray([2.0]),
        },
    )
    second = replace(
        _result(n=3),
        diagnostics={
            "step_status_history": np.asarray(["restart", "accepted"], dtype=object),
            "time_step_history": np.asarray([0.5, 0.25]),
        },
    )

    single = driver._merge_stage_chunk_results([first], mode_i="accelerated")
    assert single.diagnostics["accelerated_stage_chunked"] is False
    assert single.diagnostics["accelerated_stage_effective_mode"] == "accelerated"

    merged = driver._merge_stage_chunk_results([first, second], mode_i="parity")

    assert merged.state is second.state
    assert merged.n_iter == first.n_iter + second.n_iter + 1
    np.testing.assert_allclose(merged.w_history, np.concatenate([first.w_history, second.w_history]))
    np.testing.assert_array_equal(
        merged.diagnostics["step_status_history"],
        np.asarray(["accepted", "restart", "accepted"], dtype=object),
    )
    np.testing.assert_allclose(merged.diagnostics["fsq_prev_history"], [2.0])
    np.testing.assert_allclose(merged.diagnostics["time_step_history"], [0.5, 0.25])
    assert merged.diagnostics["accelerated_stage_chunked"] is True
    assert merged.diagnostics["accelerated_stage_effective_mode"] == "parity"
    np.testing.assert_array_equal(merged.diagnostics["accelerated_stage_chunk_iters"], [2, 3])


def test_wout_from_fixed_boundary_run_uses_sparse_history_sampling_and_restores_env(monkeypatch, tmp_path):
    captured = []

    def fake_wout_minimal_from_fixed_boundary(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(kind="wout", kwargs=kwargs)

    monkeypatch.setattr(driver, "residual_scalars_from_state", lambda **_kwargs: (9.0, 8.0, 7.0))
    monkeypatch.setattr("vmec_jax.wout.wout_minimal_from_fixed_boundary", fake_wout_minimal_from_fixed_boundary)
    monkeypatch.setenv("VMEC_JAX_WOUT_FAST_BCOVAR", "previous")

    run = driver.FixedBoundaryRun(
        cfg=object(),
        indata=object(),
        static=object(),
        state=object(),
        result=_result(n=150),
        flux=object(),
        profiles={},
        signgs=-1,
    )

    out = driver.wout_from_fixed_boundary_run(run, path=tmp_path / "wout.nc", fast_bcovar=False)

    assert out.kind == "wout"
    assert captured[-1]["signgs"] == -1
    assert captured[-1]["converged"] is True
    assert captured[-1]["fsqr"] == pytest.approx(0.2)
    assert captured[-1]["fsqz"] == pytest.approx(0.1)
    assert captured[-1]["fsql"] == pytest.approx(0.1)
    assert captured[-1]["fsqt"].shape == (100,)
    assert np.count_nonzero(captured[-1]["fsqt"]) == 75
    assert captured[-1]["fsqt"][0] == pytest.approx(run.result.fsqr2_history[1] + run.result.fsqz2_history[1])
    assert captured[-1]["fsqt"][74] == pytest.approx(run.result.fsqr2_history[149] + run.result.fsqz2_history[149])
    assert captured[-1]["fsqt"][75] == 0.0
    assert driver.os.environ["VMEC_JAX_WOUT_FAST_BCOVAR"] == "previous"


def test_wout_from_fixed_boundary_run_falls_back_for_bad_histories_and_include_fsq_false(monkeypatch, tmp_path):
    captured = []

    def fake_wout_minimal_from_fixed_boundary(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(kwargs=kwargs)

    monkeypatch.setattr(driver, "residual_scalars_from_state", lambda **_kwargs: (9.0, 8.0, 7.0))
    monkeypatch.setattr("vmec_jax.wout.wout_minimal_from_fixed_boundary", fake_wout_minimal_from_fixed_boundary)
    monkeypatch.delenv("VMEC_JAX_WOUT_FAST_BCOVAR", raising=False)

    run = driver.FixedBoundaryRun(
        cfg=object(),
        indata=object(),
        static=object(),
        state=object(),
        result=_result(bad_fsql=True),
        flux=object(),
        profiles={},
        signgs=1,
    )

    driver.wout_from_fixed_boundary_run(run, path=tmp_path / "fallback.nc", fast_bcovar=True)
    assert captured[-1]["fsqr"] == pytest.approx(9.0)
    assert captured[-1]["fsqz"] == pytest.approx(8.0)
    assert captured[-1]["fsql"] == pytest.approx(7.0)
    assert "VMEC_JAX_WOUT_FAST_BCOVAR" not in driver.os.environ

    driver.wout_from_fixed_boundary_run(run, path=tmp_path / "zeros.nc", include_fsq=False)
    assert captured[-1]["fsqr"] == 0.0
    assert captured[-1]["fsqz"] == 0.0
    assert captured[-1]["fsql"] == 0.0
    assert captured[-1]["fsqt"] is None


class _Input:
    def __init__(self, **values):
        self.values = dict(values)

    def get(self, key, default=None):
        return self.values.get(key, default)

    def get_bool(self, key, default=False):
        return bool(self.values.get(key, default))

    def get_float(self, key, default=0.0):
        return float(self.values.get(key, default))

    def get_int(self, key, default=0):
        return int(self.values.get(key, default))


def test_default_backend_name_falls_back_to_cpu_when_jax_import_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "jax":
            raise RuntimeError("synthetic jax import failure")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert driver._default_backend_name() == "cpu"


def test_dynamic_scan_probe_settings_parses_env_and_clamps_to_available_iterations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(driver, "_default_backend_name", lambda: "gpu")
    monkeypatch.setenv("VMEC_JAX_DYNAMIC_SCAN_TIMED", "yes")
    monkeypatch.setenv("VMEC_JAX_DYNAMIC_SCAN_ITERS", "bad-int")

    assert driver._dynamic_scan_probe_settings(5) == (1, True, "gpu")

    monkeypatch.setenv("VMEC_JAX_DYNAMIC_SCAN_TIMED", "maybe")
    monkeypatch.setenv("VMEC_JAX_DYNAMIC_SCAN_ITERS", "0")

    assert driver._dynamic_scan_probe_settings(10) == (1, False, "gpu")


def test_list_and_float_normalizers_cover_scalar_iterable_and_invalid_inputs() -> None:
    assert driver._as_float_list(np.asarray(["1.0", "2.5"])) == [1.0, 2.5]
    assert driver._as_float_list(object()) is None
    assert driver._as_list_like((1, 2)) == [1, 2]
    assert driver._as_list_like(np.asarray([3, 4])) == [3, 4]
    assert driver._as_list_like(np.int64(5)) == [np.int64(5)]
    assert driver._as_list_like(v for v in (6, 7)) == [6, 7]


def test_default_use_scan_for_backend_validates_solver_mode_and_gpu_policy() -> None:
    assert driver._default_use_scan_for_backend(_Input(), "cpu", "safe") is False
    assert driver._default_use_scan_for_backend(_Input(), "gpu", "safe") is True
    with pytest.raises(ValueError, match="Unknown solver_mode"):
        driver._default_use_scan_for_backend(_Input(), "cpu", "definitely-not-valid")


def test_resolve_vmec2000_stage_controls_caps_explicit_stage_budgets() -> None:
    niter, ftol, niter_input, ftol_input = driver._resolve_vmec2000_stage_controls(
        nstep=3,
        niter_list=[3, 4, 5],
        ftol_list=[1.0e-3, 1.0e-4, 1.0e-5],
        max_iter=6,
        max_iter_overridden=False,
        multigrid_use_input_niter=True,
        multigrid_user_provided=False,
        accelerated_single_grid_default=False,
        indata=_Input(NITER=99, FTOL=9.0e-9),
    )

    assert niter == [3, 3, 0]
    assert ftol == [1.0e-3, 1.0e-4, 1.0e-5]
    assert niter_input == [3, 4, 5]
    assert ftol_input == [1.0e-3, 1.0e-4, 1.0e-5]


def test_resolve_vmec2000_stage_controls_collapsed_accelerated_single_grid_budget() -> None:
    niter, ftol, niter_input, ftol_input = driver._resolve_vmec2000_stage_controls(
        nstep=1,
        niter_list=[2, 3],
        ftol_list=[1.0e-2, 1.0e-4],
        max_iter=5,
        max_iter_overridden=False,
        multigrid_use_input_niter=True,
        multigrid_user_provided=False,
        accelerated_single_grid_default=True,
        indata=_Input(NITER=99, FTOL=9.0e-9),
    )

    assert niter == [5]
    assert ftol == [1.0e-4]
    assert niter_input is None
    assert ftol_input is None


def test_resolve_vmec2000_stage_controls_distributes_overridden_budget_and_defaults_ftol() -> None:
    niter, ftol, niter_input, ftol_input = driver._resolve_vmec2000_stage_controls(
        nstep=3,
        niter_list=None,
        ftol_list=None,
        max_iter=5,
        max_iter_overridden=True,
        multigrid_use_input_niter=True,
        multigrid_user_provided=False,
        accelerated_single_grid_default=False,
        indata=_Input(NITER=99, FTOL=7.0e-7),
    )

    assert niter == [2, 2, 1]
    assert ftol == [7.0e-7, 7.0e-7, 7.0e-7]
    assert niter_input is None
    assert ftol_input is None


def test_resolve_vmec2000_stage_controls_uses_niter_per_stage_without_input_array() -> None:
    niter, ftol, _, _ = driver._resolve_vmec2000_stage_controls(
        nstep=2,
        niter_list=None,
        ftol_list=None,
        max_iter=9,
        max_iter_overridden=False,
        multigrid_use_input_niter=True,
        multigrid_user_provided=False,
        accelerated_single_grid_default=False,
        indata=_Input(NITER=4, FTOL=8.0e-8),
    )

    assert niter == [4, 4]
    assert ftol == [8.0e-8, 8.0e-8]


def test_resolve_vmec2000_stage_controls_manual_distribution_still_uses_ftol_array() -> None:
    niter, ftol, _, ftol_input = driver._resolve_vmec2000_stage_controls(
        nstep=3,
        niter_list=[10, 20, 30],
        ftol_list=[1.0e-1, 1.0e-2, 1.0e-3],
        max_iter=5,
        max_iter_overridden=False,
        multigrid_use_input_niter=False,
        multigrid_user_provided=True,
        accelerated_single_grid_default=True,
        indata=_Input(NITER=99, FTOL=9.0e-9),
    )

    assert niter == [2, 2, 1]
    assert ftol == [1.0e-1, 1.0e-2, 1.0e-3]
    assert ftol_input == [1.0e-1, 1.0e-2, 1.0e-3]


def test_budget_helpers_cover_empty_stage_and_weight_inputs() -> None:
    assert driver._allocate_integer_budget(total=5, weights=[]) == []
    assert driver._accelerated_cli_budgeted_total_iters(total_budget=7, ns_stages=[]) == 7


def test_sanitize_minimal_resume_state_for_finish_keeps_only_safe_scalars() -> None:
    marker = object()
    assert driver._sanitize_minimal_resume_state_for_finish(marker) is marker

    missing_time = {"inv_tau": [1.0]}
    assert driver._sanitize_minimal_resume_state_for_finish(missing_time) is missing_time

    bad_time = {"time_step": object(), "inv_tau": [1.0]}
    assert driver._sanitize_minimal_resume_state_for_finish(bad_time) is bad_time

    invalid_flip = driver._sanitize_minimal_resume_state_for_finish(
        {"time_step": -0.25, "iter_offset": "3", "vmec2000_cache_valid": 1, "flip_sign": object()}
    )
    assert invalid_flip["time_step"] == pytest.approx(-0.25)
    assert invalid_flip["iter_offset"] == 3
    assert invalid_flip["vmec2000_cache_valid"] is True
    assert invalid_flip["inv_tau"] == [pytest.approx(0.6)] * 10
    assert "flip_sign" not in invalid_flip

    valid = driver._sanitize_minimal_resume_state_for_finish(
        {"time_step": "0.5", "inv_tau": (1.0, 2.0), "iter_offset": 4, "flip_sign": "-1"}
    )
    assert valid == {
        "time_step": 0.5,
        "inv_tau": [1.0, 2.0],
        "iter_offset": 4,
        "vmec2000_cache_valid": False,
        "flip_sign": -1.0,
    }


class _FlakyFloat:
    def __init__(self, value: float):
        self.value = float(value)
        self.calls = 0

    def __float__(self):
        self.calls += 1
        if self.calls == 1:
            raise TypeError("synthetic first conversion failure")
        return self.value


def test_resume_state_sanitizers_cover_conversion_fallbacks_and_missing_time_step() -> None:
    cross = driver._sanitize_resume_state_for_grid_change({"time_step": _FlakyFloat(0.2)}, step_size=0.1)
    same = driver._sanitize_resume_state_for_same_grid({"time_step": _FlakyFloat(0.3)}, step_size=0.1)

    assert cross["time_step"] == pytest.approx(0.2)
    assert cross["inv_tau"] == [pytest.approx(0.75)] * 10
    assert same["time_step"] == pytest.approx(0.3)
    assert same["inv_tau"] == [pytest.approx(0.5)] * 10
    assert driver._sanitize_resume_state_for_grid_change(None, step_size=0.1) is None
    assert driver._sanitize_resume_state_for_same_grid({"inv_tau": [1.0]}, step_size=0.1) is None


def test_result_residual_helpers_cover_bad_payload_fallbacks() -> None:
    assert driver._result_final_residuals(None) is None

    bad_explicit = SimpleNamespace(
        diagnostics={"final_fsqr": object(), "final_fsqz": 2.0, "final_fsql": 3.0},
        fsqr2_history=np.asarray([4.0]),
        fsqz2_history=np.asarray([5.0]),
        fsql2_history=np.asarray([6.0]),
    )
    assert driver._result_final_residuals(bad_explicit) == (4.0, 5.0, 6.0)

    class _BadArray:
        def __array__(self, dtype=None):
            del dtype
            raise TypeError("synthetic array conversion failure")

    bad_diag_histories = SimpleNamespace(
        diagnostics={"fsqr_full": _BadArray(), "fsqz_full": [1.0], "fsql_full": [2.0]},
    )
    assert driver._result_final_residuals(bad_diag_histories) is None
    assert driver._result_final_fsq(SimpleNamespace(diagnostics={"final_fsqr": 1.0, "final_fsqz": 2.0, "final_fsql": 3.0})) == 6.0
    assert driver._result_meets_requested_ftol(None, ftol=1.0) is False
    assert driver._result_meets_requested_ftol(SimpleNamespace(diagnostics={"requested_ftol": 1.0}), ftol=1.0) is False


def test_vmec_history_relerr_reports_scaled_difference_for_matching_shapes() -> None:
    assert driver._vmec_history_relerr(np.asarray([1.0, 1.5]), np.asarray([1.0, 2.0])) == pytest.approx(0.25)


def test_wout_from_fixed_boundary_run_samples_strided_fsqt_and_recomputes_on_bad_fsql(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import vmec_jax.wout as wout_module

    captured = {}

    def fake_wout_minimal_from_fixed_boundary(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(kind="wout")

    class _BadFinalFsql:
        def __array__(self, dtype=None):
            return np.asarray([[1.0, 2.0]], dtype=dtype)

    result = SimpleNamespace(
        diagnostics={"converged": True},
        fsqr2_history=np.arange(101, dtype=float),
        fsqz2_history=np.arange(101, dtype=float) + 100.0,
        fsql2_history=_BadFinalFsql(),
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

    monkeypatch.setattr(wout_module, "wout_minimal_from_fixed_boundary", fake_wout_minimal_from_fixed_boundary)
    monkeypatch.setattr(driver, "residual_scalars_from_state", lambda **_kwargs: (7.0, 8.0, 9.0))

    out = driver.wout_from_fixed_boundary_run(run, path=tmp_path / "wout_strided.nc")

    assert out.kind == "wout"
    assert captured["fsqr"] == 7.0
    assert captured["fsqz"] == 8.0
    assert captured["fsql"] == 9.0
    assert captured["fsqt"][0] == pytest.approx(102.0)
    assert captured["fsqt"][1] == pytest.approx(106.0)


def test_example_paths_default_root_prefers_reference_then_plain_wout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    package_driver = tmp_path / "vmec_jax" / "driver.py"
    data_dir = tmp_path / "examples" / "data"
    data_dir.mkdir(parents=True)
    package_driver.parent.mkdir()
    input_path = data_dir / "input.case"
    ref_path = data_dir / "wout_case_reference.nc"
    plain_path = data_dir / "wout_case.nc"
    input_path.write_text("&INDATA\n/\n")
    ref_path.write_text("reference")
    plain_path.write_text("plain")
    monkeypatch.setattr(driver, "__file__", str(package_driver))

    assert driver.example_paths("case") == (input_path, ref_path)

    ref_path.unlink()
    assert driver.example_paths("case") == (input_path, plain_path)


def test_load_example_passes_prepared_mgrid_metadata_without_wout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "examples" / "data"
    data_dir.mkdir(parents=True)
    input_path = data_dir / "input.freeb"
    input_path.write_text("&INDATA\n/\n")
    cfg = SimpleNamespace(marker="cfg", lfreeb=True)
    indata = object()
    static = object()
    metadata = MGridMetadata(
        path="mgrid.synthetic.nc",
        ir=2,
        jz=3,
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
    )
    prepared = PreparedMGrid(metadata=metadata, extcur=(2.0,))
    captured = {}

    def fake_load_config(path):
        captured["load"] = path
        return cfg, indata

    def fake_prepare_mgrid_for_config(cfg_arg, **kwargs):
        captured["mgrid"] = (cfg_arg, kwargs)
        return prepared

    def fake_build_static(cfg_arg, **kwargs):
        captured["static"] = (cfg_arg, kwargs)
        return static

    monkeypatch.setattr(driver, "load_config", fake_load_config)
    monkeypatch.setattr(driver, "validate_free_boundary_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(driver, "prepare_mgrid_for_config", fake_prepare_mgrid_for_config)
    monkeypatch.setattr(driver, "build_static", fake_build_static)

    loaded = driver.load_example("freeb", root=tmp_path, with_wout=False, grid="grid")

    assert loaded.static is static
    assert loaded.wout is None
    assert loaded.state is None
    assert captured["load"] == str(input_path)
    assert captured["mgrid"] == (cfg, {"load_fields": False, "strict": False})
    assert captured["static"] == (
        cfg,
        {"grid": "grid", "mgrid_metadata": metadata, "free_boundary_extcur": (2.0,)},
    )


def test_save_npz_creates_parent_and_round_trips_arrays(tmp_path: Path) -> None:
    out_path = driver.save_npz(tmp_path / "nested" / "values.npz", values=np.asarray([1.0, 2.0]))

    assert out_path == tmp_path / "nested" / "values.npz"
    with np.load(out_path) as loaded:
        np.testing.assert_allclose(loaded["values"], [1.0, 2.0])
