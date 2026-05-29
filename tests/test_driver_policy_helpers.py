from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

import vmec_jax.driver as driver


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


@pytest.mark.parametrize(
    ("solver_mode", "performance_mode", "expected"),
    [
        (None, True, "default"),
        (None, False, "parity"),
        (" DEFAULT ", False, "default"),
        ("Fast", False, "default"),
        ("SAFE", True, "parity"),
        (" reference ", True, "parity"),
        (" PERF ", False, "accelerated"),
        ("accelerated", True, "accelerated"),
    ],
)
def test_normalize_solver_mode_handles_aliases_case_and_defaults(solver_mode, performance_mode, expected):
    assert driver._normalize_solver_mode(solver_mode=solver_mode, performance_mode=performance_mode) == expected


def test_normalize_solver_mode_reports_valid_modes():
    with pytest.raises(ValueError, match="Expected one of: accelerated, default, parity"):
        driver._normalize_solver_mode(solver_mode="not-a-mode", performance_mode=False)


@pytest.mark.parametrize(
    ("backend", "indata", "expected"),
    [
        ("gpu", _Input(LFREEB=True, NS_ARRAY=[5, 9]), ("default", True)),
        ("gpu", _Input(NS_ARRAY=[5, 9]), ("parity", False)),
        ("gpu", _Input(NS_ARRAY=[9]), ("accelerated", True)),
        ("cpu", _Input(LASYM=True), ("accelerated", True)),
        ("cpu", _Input(NCURR=1, NS_ARRAY=[5, 9], NITER_ARRAY=[10, 20]), ("accelerated", True)),
        ("cpu", _Input(NCURR=1, NS_ARRAY=[5, 9]), ("parity", False)),
        ("cpu", _Input(), ("default", True)),
    ],
)
def test_default_non_autodiff_solver_policy_for_backend_uses_input_structure(backend, indata, expected):
    assert driver._default_non_autodiff_solver_policy_for_backend(indata, backend) == expected


def test_resolve_jit_forces_auto_policy_preserves_explicit_flags():
    static = SimpleNamespace(
        modes=SimpleNamespace(m=np.arange(3)),
        cfg=SimpleNamespace(ns=3, ntheta=4, nzeta=5),
    )

    assert driver._resolve_jit_forces_auto_policy(False, static, niter_i=100) is False
    assert driver._resolve_jit_forces_auto_policy(True, static, niter_i=0) is True
    assert driver._resolve_jit_forces_auto_policy("false", static, niter_i=0) is True


def test_resolve_jit_forces_auto_policy_uses_work_and_iteration_thresholds():
    small = SimpleNamespace(
        modes=SimpleNamespace(m=np.arange(2)),
        cfg=SimpleNamespace(ns=3, ntheta=4, nzeta=5),
    )
    large = SimpleNamespace(
        modes=SimpleNamespace(m=np.arange(2_000)),
        cfg=SimpleNamespace(ns=50, ntheta=50, nzeta=1),
    )

    assert driver._resolve_jit_forces_auto_policy("auto", small, niter_i=4) is False
    assert driver._resolve_jit_forces_auto_policy(" auto ", small, niter_i=5) is True
    assert driver._resolve_jit_forces_auto_policy("AUTO", large, niter_i=1) is True
    assert driver._resolve_jit_forces_auto_policy("auto", object(), niter_i=1) is True


def test_host_update_default_and_scan_policy_handle_fallback_branches(monkeypatch):
    cfg = SimpleNamespace(ns=2, mpol=2, ntor=1, lasym=False)
    monkeypatch.setenv("VMEC_JAX_HOST_UPDATE_CPU_WORK_LIMIT", "not-an-int")

    assert driver._host_update_assembly_driver_default(
        cfg=cfg,
        performance_mode=True,
        backend="cpu",
        use_scan=False,
    ) is True

    assert driver._default_use_scan_for_backend(_Input(), "unknown-backend", "default") is False


def test_precomputed_tridi_policy_tolerates_bad_mode_counts():
    class BadModeCfg:
        lasym = False
        ntor = 16

        @property
        def mpol(self):
            raise RuntimeError("mode metadata unavailable")

    assert driver._default_preconditioner_use_precomputed_tridi(
        cfg=BadModeCfg(),
        backend="gpu",
        performance_mode=True,
        use_scan=False,
    ) is None


@pytest.mark.parametrize("solver_device", [None, "", " none ", "AUTO", "default"])
def test_resolve_fixed_boundary_solver_device_inherits_default_for_auto_values(solver_device):
    assert (
        driver._resolve_fixed_boundary_solver_device_name(
            solver_device=solver_device,
            backend="gpu",
            cfg=object(),
            indata=object(),
            solver_lower="vmec2000_iter",
            cli_fixed_boundary_mode=True,
            accelerated_mode=True,
            ns_list_input=[5, 9],
            niter_list_input=[10, 20],
            restart_state_present=True,
            restart_solver_state_present=True,
        )
        is None
    )


@pytest.mark.parametrize(("solver_device", "expected"), [(" cpu ", "cpu"), ("GPU", "gpu"), ("tpu", "tpu")])
def test_resolve_fixed_boundary_solver_device_preserves_explicit_names(solver_device, expected):
    assert (
        driver._resolve_fixed_boundary_solver_device_name(
            solver_device=solver_device,
            backend="cpu",
            cfg=object(),
            indata=object(),
            solver_lower="vmec2000_iter",
            cli_fixed_boundary_mode=False,
            accelerated_mode=False,
            ns_list_input=None,
            niter_list_input=None,
            restart_state_present=False,
            restart_solver_state_present=False,
        )
        == expected
    )


def test_dynamic_scan_probe_settings_clamps_single_iteration_probe(monkeypatch):
    monkeypatch.setattr(driver, "_default_backend_name", lambda: "cpu")
    monkeypatch.setenv("VMEC_JAX_DYNAMIC_SCAN_ITERS", "50")
    monkeypatch.setenv("VMEC_JAX_DYNAMIC_SCAN_TIMED", "off")

    pre_iters, timed_probe, backend = driver._dynamic_scan_probe_settings(1)

    assert pre_iters == 1
    assert timed_probe is False
    assert backend == "cpu"


def test_example_paths_reports_missing_wout_as_none(tmp_path):
    data_dir = tmp_path / "examples" / "data"
    data_dir.mkdir(parents=True)
    input_path = data_dir / "input.synthetic"
    input_path.write_text("&INDATA\n/\n")

    found_input, found_wout = driver.example_paths("synthetic", root=tmp_path)

    assert found_input == input_path
    assert found_wout is None


def test_requested_final_ftol_prefers_last_valid_list_value_and_clamps():
    fallback = _Input(FTOL=4.0e-7)
    negative_fallback = _Input(FTOL=-1.0)

    assert driver._requested_final_ftol(indata=fallback, ftol_list_input=("1e-4", 2.0e-5)) == 2.0e-5
    assert driver._requested_final_ftol(indata=fallback, ftol_list_input=[1.0e-4, -2.0]) == 0.0
    assert driver._requested_final_ftol(indata=fallback, ftol_list_input=object()) == 4.0e-7
    assert driver._requested_final_ftol(indata=negative_fallback, ftol_list_input=[]) == 0.0


def test_accelerated_total_target_clamps_negative_ftol():
    assert driver._accelerated_fsq_total_target_from_ftol(-1.0e-8) == 0.0
    assert driver._accelerated_fsq_total_target_from_ftol(2.0e-8) == pytest.approx(6.0e-8)


def test_allocate_integer_budget_clamps_inputs_and_distributes_remainder():
    assert driver._allocate_integer_budget(total=-3, weights=[1, 2, 3]) == [0, 0, 0]
    assert driver._allocate_integer_budget(total=6, weights=[-5, 1, 2]) == [0, 2, 4]
    assert driver._allocate_integer_budget(total=5, weights=[1, 2]) == [2, 3]
    assert driver._allocate_integer_budget(total=5, weights=[0, 0, 0]) == [0, 0, 5]


def test_as_list_like_tolerates_broken_numpy_type_check(monkeypatch):
    monkeypatch.setattr(driver.np, "ndarray", object())

    assert driver._as_list_like(object()) is None


def test_accelerated_cli_budget_helpers_scale_total_and_weight_stages():
    assert driver._accelerated_cli_budgeted_total_iters(total_budget=100, ns_stages=[9, 36]) == 50
    assert driver._accelerated_cli_budgeted_total_iters(total_budget=0, ns_stages=[4]) == 1
    assert driver._accelerated_cli_budgeted_stage_iters(total_budget=10, ns_stages=[4, 7, 10]) == [5, 3, 2]
    assert driver._accelerated_cli_budgeted_stage_iters(total_budget=0, ns_stages=[]) == [1]


def test_distribute_stage_iters_matches_vmec_budget_edges():
    assert driver._distribute_stage_iters(iters=0, nstep=3) == [0]
    assert driver._distribute_stage_iters(iters=2, nstep=5) == [2]
    assert driver._distribute_stage_iters(iters=7, nstep=3) == [3, 2, 2]
    assert driver._distribute_stage_iters(iters=5, nstep=1) == [5]


def test_resume_state_sanitizers_drop_unsafe_payloads_and_clamp_step():
    resume_state = {
        "time_step": 0.25,
        "inv_tau": [1.0, 2.0],
        "iter_offset": 17,
        "flip_sign": -1,
        "vmec2000_cache_valid": True,
        "cached_arrays": object(),
    }

    cross_grid = driver._sanitize_resume_state_for_grid_change(resume_state, step_size=0.1)
    same_grid = driver._sanitize_resume_state_for_same_grid(resume_state, step_size=0.1)

    assert cross_grid["time_step"] == pytest.approx(0.1)
    assert cross_grid["inv_tau"] == [pytest.approx(1.5)] * 10
    assert cross_grid["iter_offset"] == 0
    assert cross_grid["flip_sign"] == -1.0
    assert cross_grid["vmec2000_cache_valid"] is False
    assert "cached_arrays" not in cross_grid

    assert same_grid["time_step"] == pytest.approx(0.1)
    assert same_grid["inv_tau"] == [1.0, 2.0]
    assert same_grid["iter_offset"] == 17
    assert same_grid["vmec2000_cache_valid"] is False
    assert "cached_arrays" not in same_grid
    assert driver._sanitize_resume_state_for_grid_change({}, step_size=0.1) is None
    assert driver._sanitize_resume_state_for_same_grid(None, step_size=0.1) is None


def _stage_result(label, *, n_iter, w_history, diagnostics=None):
    zeros = np.zeros((0,), dtype=float)
    return driver.SolveVmecResidualResult(
        state=label,
        n_iter=int(n_iter),
        w_history=np.asarray(w_history, dtype=float),
        fsqr2_history=np.asarray(w_history, dtype=float) + 10.0,
        fsqz2_history=np.asarray(w_history, dtype=float) + 20.0,
        fsql2_history=np.asarray(w_history, dtype=float) + 30.0,
        grad_rms_history=zeros,
        step_history=zeros,
        diagnostics={} if diagnostics is None else dict(diagnostics),
    )


def test_merge_stage_chunk_results_concatenates_histories_and_diagnostics():
    first = _stage_result(
        "first",
        n_iter=1,
        w_history=[1.0, 0.5],
        diagnostics={
            "step_status_history": np.asarray([1.0]),
            "first_only": True,
            "timing": {
                "solve_total_s": 2.0,
                "iteration_loop_s": 1.5,
                "compute_forces_calls": 2,
                "iterations": 2,
            },
        },
    )
    second = _stage_result(
        "second",
        n_iter=2,
        w_history=[0.25, 0.125, 0.0625],
        diagnostics={
            "step_status_history": np.asarray([2.0, 3.0]),
            "time_step_history": np.asarray([0.1]),
            "timing": {
                "solve_total_s": 3.0,
                "iteration_loop_s": 2.5,
                "compute_forces_calls": 3,
                "iterations": 3,
            },
        },
    )

    merged = driver._merge_stage_chunk_results([first, second], mode_i="accelerated")

    assert merged.state == "second"
    assert merged.n_iter == 4
    np.testing.assert_allclose(merged.w_history, [1.0, 0.5, 0.25, 0.125, 0.0625])
    np.testing.assert_allclose(merged.diagnostics["step_status_history"], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(merged.diagnostics["time_step_history"], [0.1])
    np.testing.assert_array_equal(merged.diagnostics["accelerated_stage_chunk_iters"], [2, 3])
    assert merged.diagnostics["accelerated_stage_chunk_count"] == 2
    assert merged.diagnostics["accelerated_stage_chunked"] is True
    assert merged.diagnostics["accelerated_stage_effective_mode"] == "accelerated"
    assert merged.diagnostics["timing"]["solve_total_s"] == pytest.approx(5.0)
    assert merged.diagnostics["timing"]["iteration_loop_s"] == pytest.approx(4.0)
    assert merged.diagnostics["timing"]["compute_forces_calls"] == 5
    assert merged.diagnostics["timing"]["iterations"] == 5
    assert merged.diagnostics["timing"]["solve_total_per_iter_s"] == pytest.approx(1.0)
    np.testing.assert_allclose(merged.diagnostics["timing"]["chunk_solve_total_s"], [2.0, 3.0])


def test_merge_stage_chunk_results_marks_single_chunk_without_concatenation():
    result = _stage_result("single", n_iter=0, w_history=[1.0], diagnostics={"kept": "yes"})

    merged = driver._merge_stage_chunk_results([result], mode_i="parity")

    assert merged.state == "single"
    assert merged.diagnostics["kept"] == "yes"
    assert merged.diagnostics["accelerated_stage_chunked"] is False
    assert merged.diagnostics["accelerated_stage_effective_mode"] == "parity"


def test_stage_switch_reason_from_progress_reports_only_actionable_misses():
    assert (
        driver._stage_switch_reason_from_progress(
            start_total_fsq=10.0,
            best_total_fsq=9.0,
            target_total_fsq=1.0,
            chunk_iters=2,
            remaining_budget=0,
        )
        is None
    )
    assert (
        driver._stage_switch_reason_from_progress(
            start_total_fsq=10.0,
            best_total_fsq=np.inf,
            target_total_fsq=1.0,
            chunk_iters=2,
            remaining_budget=10,
        )
        == "nonfinite_total_fsq"
    )
    assert (
        driver._stage_switch_reason_from_progress(
            start_total_fsq=10.0,
            best_total_fsq=10.0,
            target_total_fsq=1.0,
            chunk_iters=2,
            remaining_budget=10,
        )
        == "nondecreasing_total_fsq"
    )
    assert (
        driver._stage_switch_reason_from_progress(
            start_total_fsq=10.0,
            best_total_fsq=0.5,
            target_total_fsq=1.0,
            chunk_iters=2,
            remaining_budget=10,
        )
        is None
    )
    assert (
        driver._stage_switch_reason_from_progress(
            start_total_fsq=0.0,
            best_total_fsq=-1.0,
            target_total_fsq=0.0,
            chunk_iters=2,
            remaining_budget=10,
        )
        is None
    )
    assert (
        driver._stage_switch_reason_from_progress(
            start_total_fsq=100.0,
            best_total_fsq=10.0,
            target_total_fsq=1.0,
            chunk_iters=1,
            remaining_budget=2,
        )
        is None
    )
    assert driver._stage_switch_reason_from_progress(
        start_total_fsq=100.0,
        best_total_fsq=90.0,
        target_total_fsq=1.0,
        chunk_iters=1,
        remaining_budget=3,
    ).startswith("projected_budget_miss:")


def test_vmec_history_comparison_helpers_cover_mismatch_and_tolerance():
    lhs = SimpleNamespace(
        w_history=np.asarray([1.0, 2.0]),
        fsqr2_history=np.asarray([1.0]),
        fsqz2_history=np.asarray([2.0]),
        fsql2_history=np.asarray([3.0]),
    )
    rhs = SimpleNamespace(
        w_history=np.asarray([1.0, 2.0 + 1.0e-7]),
        fsqr2_history=np.asarray([1.0]),
        fsqz2_history=np.asarray([2.0]),
        fsql2_history=np.asarray([3.0]),
    )
    wrong_shape = SimpleNamespace(
        w_history=np.asarray([1.0]),
        fsqr2_history=np.asarray([1.0]),
        fsqz2_history=np.asarray([2.0]),
        fsql2_history=np.asarray([3.0]),
    )

    assert driver._vmec_history_relerr(np.asarray([1.0]), np.asarray([[1.0]])) == np.inf
    assert driver._vmec_histories_match(lhs, rhs, rtol=1.0e-5, atol=0.0) is True
    assert driver._vmec_histories_match(lhs, rhs, rtol=1.0e-10, atol=0.0) is False
    assert driver._vmec_histories_match(lhs, wrong_shape, rtol=1.0, atol=1.0) is False


def test_result_final_residuals_prefers_explicit_diagnostics_over_histories():
    result = SimpleNamespace(
        diagnostics={"final_fsqr": "1.0", "final_fsqz": 2, "final_fsql": np.asarray(3.0)},
        fsqr2_history=np.asarray([10.0]),
        fsqz2_history=np.asarray([20.0]),
        fsql2_history=np.asarray([30.0]),
    )

    assert driver._result_final_residuals(result) == (1.0, 2.0, 3.0)


def test_result_final_residuals_falls_back_when_explicit_diagnostics_are_incomplete():
    result = SimpleNamespace(
        diagnostics={"final_fsqr": 1.0, "final_fsqz": None, "final_fsql": 3.0},
        fsqr2_history=np.asarray([[1.0, 4.0]]),
        fsqz2_history=np.asarray([[2.0, 5.0]]),
        fsql2_history=np.asarray([[3.0, 6.0]]),
    )

    assert driver._result_final_residuals(result) == (4.0, 5.0, 6.0)


def test_result_final_residuals_uses_flattened_diagnostic_histories():
    result = SimpleNamespace(
        diagnostics={
            "fsqr_full": np.asarray([[1.0, 2.0]]),
            "fsqz_full": np.asarray([[3.0, 4.0]]),
            "fsql_full": np.asarray([[5.0, 6.0]]),
        },
    )

    assert driver._result_final_residuals(result) == (2.0, 4.0, 6.0)


def test_result_final_residuals_ignores_unparseable_explicit_values_and_broken_histories():
    class BrokenHistoryResult:
        diagnostics = {
            "final_fsqr": "not-a-float",
            "final_fsqz": 2.0,
            "final_fsql": 3.0,
            "fsqr_full": [],
            "fsqz_full": [],
            "fsql_full": [],
        }

        @property
        def fsqr2_history(self):
            raise RuntimeError("history not materialized")

        @property
        def fsqz2_history(self):
            raise RuntimeError("history not materialized")

        @property
        def fsql2_history(self):
            raise RuntimeError("history not materialized")

    assert driver._result_final_residuals(BrokenHistoryResult()) is None


def test_result_final_fsq_prefers_weight_history_then_residual_sum():
    weighted = SimpleNamespace(
        diagnostics={"final_fsqr": 1.0, "final_fsqz": 2.0, "final_fsql": 3.0},
        w_history=np.asarray([99.0, 0.25]),
    )
    residual_only = SimpleNamespace(diagnostics={"final_fsqr": 1.0, "final_fsqz": 2.0, "final_fsql": 3.0})

    assert driver._result_final_fsq(weighted) == 0.25
    assert driver._result_final_fsq(residual_only) == 6.0
    assert np.isinf(driver._result_final_fsq(None))


def test_result_meets_requested_ftol_respects_strict_diagnostic_flag():
    result = SimpleNamespace(
        diagnostics={"converged_strict": False, "final_fsqr": 0.0, "final_fsqz": 0.0, "final_fsql": 0.0},
    )

    assert driver._result_meets_requested_ftol(result, ftol=1.0) is False


def test_result_meets_requested_ftol_uses_legacy_converged_only_without_ftol_metadata():
    result = SimpleNamespace(
        diagnostics={"converged": False, "final_fsqr": 0.0, "final_fsqz": 0.0, "final_fsql": 0.0},
    )

    assert driver._result_meets_requested_ftol(result, ftol=1.0) is False


def test_result_meets_requested_ftol_clamps_negative_requested_target():
    zero = SimpleNamespace(
        diagnostics={"requested_ftol": -1.0, "final_fsqr": 0.0, "final_fsqz": 0.0, "final_fsql": 0.0},
    )
    nonzero = SimpleNamespace(
        diagnostics={"requested_ftol": -1.0, "final_fsqr": 1.0e-16, "final_fsqz": 0.0, "final_fsql": 0.0},
    )

    assert driver._result_meets_requested_ftol(zero, ftol=-1.0) is True
    assert driver._result_meets_requested_ftol(nonzero, ftol=-1.0) is False


def test_result_hits_total_target_clamps_negative_target_and_handles_none():
    zero = SimpleNamespace(w_history=np.asarray([0.0]), diagnostics={})
    nonzero = SimpleNamespace(w_history=np.asarray([1.0e-12]), diagnostics={})

    assert driver._result_hits_total_target(zero, fsq_total_target=-1.0) is True
    assert driver._result_hits_total_target(nonzero, fsq_total_target=-1.0) is False
    assert driver._result_hits_total_target(zero, fsq_total_target=None) is False


def test_copy_final_force_payload_preserves_solver_payload_when_present():
    source = _stage_result("source", n_iter=1, w_history=[1.0])
    result = _stage_result("result", n_iter=1, w_history=[2.0])
    payload = {"fsqr": 1.0}
    object.__setattr__(source, "_final_force_payload", payload)

    out = driver._copy_final_force_payload(result, source)

    assert out is result
    assert getattr(out, "_final_force_payload") is payload


def test_wout_from_fixed_boundary_run_samples_fsqt_and_falls_back_to_residual_recompute(monkeypatch, tmp_path):
    import vmec_jax.wout as wout_module

    captured = {}

    def _fake_wout_minimal_from_fixed_boundary(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(path=kwargs["path"], marker="wout")

    monkeypatch.setattr(wout_module, "wout_minimal_from_fixed_boundary", _fake_wout_minimal_from_fixed_boundary)
    monkeypatch.setattr(driver, "residual_scalars_from_state", lambda **_kwargs: (7.0, 8.0, 9.0))
    monkeypatch.delenv("VMEC_JAX_WOUT_FAST_BCOVAR", raising=False)

    result = SimpleNamespace(
        diagnostics={"converged": True},
        fsqr2_history=np.asarray([1.0, 2.0, 3.0]),
        fsqz2_history=np.asarray([4.0, 5.0, 6.0]),
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

    out = driver.wout_from_fixed_boundary_run(
        run,
        include_fsq=True,
        path=tmp_path / "wout_synthetic.nc",
        fast_bcovar=True,
    )

    assert out.marker == "wout"
    assert captured["fsqr"] == 7.0
    assert captured["fsqz"] == 8.0
    assert captured["fsql"] == 9.0
    assert captured["converged"] is True
    np.testing.assert_allclose(captured["fsqt"][:3], [5.0, 7.0, 9.0])
    assert captured["fsqt"].shape == (100,)
    assert os.getenv("VMEC_JAX_WOUT_FAST_BCOVAR") is None


def test_wout_from_fixed_boundary_run_include_fsq_false_restores_existing_fast_env(monkeypatch, tmp_path):
    import vmec_jax.wout as wout_module

    captured = {}

    def _fake_wout_minimal_from_fixed_boundary(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(path=kwargs["path"])

    monkeypatch.setattr(wout_module, "wout_minimal_from_fixed_boundary", _fake_wout_minimal_from_fixed_boundary)
    monkeypatch.setattr(driver, "residual_scalars_from_state", lambda **_kwargs: pytest.fail("unexpected recompute"))
    monkeypatch.setenv("VMEC_JAX_WOUT_FAST_BCOVAR", "original")
    run = driver.FixedBoundaryRun(
        cfg=object(),
        indata=object(),
        static=object(),
        state=object(),
        result=None,
        flux=None,
        profiles={},
        signgs=1,
    )

    driver.wout_from_fixed_boundary_run(
        run,
        include_fsq=False,
        path=tmp_path / "wout_no_fsq.nc",
        fast_bcovar=False,
    )

    assert captured["fsqr"] == 0.0
    assert captured["fsqz"] == 0.0
    assert captured["fsql"] == 0.0
    assert captured["fsqt"] is None
    assert captured["converged"] is None
    assert os.environ["VMEC_JAX_WOUT_FAST_BCOVAR"] == "original"


def test_wout_from_fixed_boundary_run_parity_mode_uses_legacy_bcovar_by_default(monkeypatch, tmp_path):
    import vmec_jax.wout as wout_module

    captured_env = []

    def _fake_wout_minimal_from_fixed_boundary(**kwargs):
        captured_env.append(os.getenv("VMEC_JAX_WOUT_FAST_BCOVAR"))
        return SimpleNamespace(path=kwargs["path"])

    monkeypatch.setattr(wout_module, "wout_minimal_from_fixed_boundary", _fake_wout_minimal_from_fixed_boundary)
    monkeypatch.setattr(driver, "residual_scalars_from_state", lambda **_kwargs: pytest.fail("unexpected recompute"))
    monkeypatch.delenv("VMEC_JAX_WOUT_FAST_BCOVAR", raising=False)
    run = driver.FixedBoundaryRun(
        cfg=object(),
        indata=object(),
        static=object(),
        state=object(),
        result=SimpleNamespace(diagnostics={"solver_mode": "parity"}),
        flux=None,
        profiles={},
        signgs=1,
    )

    driver.wout_from_fixed_boundary_run(run, include_fsq=False, path=tmp_path / "wout_parity.nc")

    assert captured_env == ["0"]
    assert os.getenv("VMEC_JAX_WOUT_FAST_BCOVAR") is None


def test_wout_from_fixed_boundary_run_explicit_fast_bcovar_overrides_parity_mode(monkeypatch, tmp_path):
    import vmec_jax.wout as wout_module

    captured_env = []

    def _fake_wout_minimal_from_fixed_boundary(**kwargs):
        captured_env.append(os.getenv("VMEC_JAX_WOUT_FAST_BCOVAR"))
        return SimpleNamespace(path=kwargs["path"])

    monkeypatch.setattr(wout_module, "wout_minimal_from_fixed_boundary", _fake_wout_minimal_from_fixed_boundary)
    monkeypatch.setattr(driver, "residual_scalars_from_state", lambda **_kwargs: pytest.fail("unexpected recompute"))
    monkeypatch.delenv("VMEC_JAX_WOUT_FAST_BCOVAR", raising=False)
    run = driver.FixedBoundaryRun(
        cfg=object(),
        indata=object(),
        static=object(),
        state=object(),
        result=SimpleNamespace(diagnostics={"solver_mode": "parity"}),
        flux=None,
        profiles={},
        signgs=1,
    )

    driver.wout_from_fixed_boundary_run(
        run,
        include_fsq=False,
        path=tmp_path / "wout_parity_fast.nc",
        fast_bcovar=True,
    )

    assert captured_env == ["1"]
    assert os.getenv("VMEC_JAX_WOUT_FAST_BCOVAR") is None


def test_wout_from_fixed_boundary_run_uses_complete_result_histories_without_recompute(monkeypatch, tmp_path):
    import vmec_jax.wout as wout_module

    captured = {}

    def _fake_wout_minimal_from_fixed_boundary(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(path=kwargs["path"], marker="packed")

    monkeypatch.setattr(wout_module, "wout_minimal_from_fixed_boundary", _fake_wout_minimal_from_fixed_boundary)
    monkeypatch.setattr(driver, "residual_scalars_from_state", lambda **_kwargs: pytest.fail("unexpected recompute"))

    result = SimpleNamespace(
        diagnostics={"converged": False},
        fsqr2_history=np.asarray([10.0, 1.0]),
        fsqz2_history=np.asarray([20.0, 2.0]),
        fsql2_history=np.asarray([30.0, 3.0]),
    )
    run = driver.FixedBoundaryRun(
        cfg=object(),
        indata=object(),
        static=object(),
        state=object(),
        result=result,
        flux=None,
        profiles={},
        signgs=1,
    )

    out = driver.wout_from_fixed_boundary_run(run, include_fsq=True, path=tmp_path / "wout_hist.nc")

    assert out.marker == "packed"
    assert captured["fsqr"] == 1.0
    assert captured["fsqz"] == 2.0
    assert captured["fsql"] == 3.0
    assert captured["converged"] is False
    np.testing.assert_allclose(captured["fsqt"][:2], [30.0, 3.0])
    np.testing.assert_allclose(captured["fsqt"][2:], np.zeros(98))
