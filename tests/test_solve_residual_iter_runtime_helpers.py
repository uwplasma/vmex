import numpy as np
import pytest

from vmec_jax.solvers.fixed_boundary.residual.runtime import (
    _attach_free_boundary_external_field_diag,
    _build_residual_iter_timing_report,
    _build_resume_state_base,
    _converged_residuals_scan_fast,
    _device_get_floats,
    _format_ptau_dump_row,
    _format_residual_iter_timing_message,
    _initial_setup_phase_timings,
    _maybe_dump_ptau,
    _maybe_print_nonscan_state_debug,
    _nonscan_state_debug_payload,
    _new_residual_iter_timing_stats,
    _ptau_dump_enabled,
    _record_compute_force_timing,
    _record_setup_timing,
    _scan_block_until_ready,
    _scan_device_run_ready,
    _scan_print_uses_debug_callback,
    _scan_print_uses_debug_print,
    _scan_print_uses_io_callback,
    _setup_timer_start,
    _vmec_freeb_plascur_from_bcovar,
    dump_xc_with_velocity_blocks,
    initial_free_boundary_loop_state,
    record_elapsed_timing,
    record_update_state_ready_timing,
    record_update_total_timing,
    resolve_free_boundary_iteration_controls,
    resolve_residual_profile_window,
    resume_free_boundary_loop_state,
    trial_residual_total_runtime,
)
from vmec_jax.solvers.fixed_boundary.residual.update import ResidualVelocityBlocks


class _Result:
    def __init__(self, diagnostics=None):
        self.state = "state"
        self.n_iter = 3
        self.w_history = np.asarray([1.0, 0.5])
        self.fsqr2_history = np.asarray([0.1])
        self.fsqz2_history = np.asarray([0.2])
        self.fsql2_history = np.asarray([0.3])
        self.grad_rms_history = np.asarray([0.4])
        self.step_history = np.asarray([0.5])
        self.diagnostics = {} if diagnostics is None else diagnostics


def _result_type(**kwargs):
    out = _Result(kwargs["diagnostics"])
    out.state = kwargs["state"]
    out.n_iter = kwargs["n_iter"]
    out.w_history = kwargs["w_history"]
    out.fsqr2_history = kwargs["fsqr2_history"]
    out.fsqz2_history = kwargs["fsqz2_history"]
    out.fsql2_history = kwargs["fsql2_history"]
    out.grad_rms_history = kwargs["grad_rms_history"]
    out.step_history = kwargs["step_history"]
    return out


def test_device_get_floats_batches_scalar_materialization():
    class FakeJax:
        calls = []

        @classmethod
        def device_get(cls, vals):
            cls.calls.append(vals)
            return vals

    assert _device_get_floats(np.asarray(1.25), 2.5, jax_module=FakeJax) == (1.25, 2.5)
    assert len(FakeJax.calls) == 1


def test_setup_timing_helpers_initialize_and_accumulate():
    timings = _initial_setup_phase_timings()
    assert set(timings) == {
        "setup_static_grid_rebuild",
        "setup_freeb_policy",
        "setup_boundary_profiles",
        "setup_profile_data",
        "setup_trig_tables",
        "setup_cache_key_hash",
        "setup_ptau_constants",
        "setup_index_constants",
        "setup_update_constants",
    }
    assert all(value == 0.0 for value in timings.values())
    assert _setup_timer_start(timing_enabled=False, perf_counter=lambda: 10.0) is None
    assert _setup_timer_start(timing_enabled=True, perf_counter=lambda: 10.0) == 10.0
    assert not _record_setup_timing(timings, "setup_freeb_policy", None, perf_counter=lambda: 12.0)
    assert _record_setup_timing(timings, "setup_freeb_policy", 10.0, perf_counter=lambda: 12.5)
    assert timings["setup_freeb_policy"] == 2.5
    assert _record_setup_timing(timings, "setup_freeb_policy", 20.0, perf_counter=lambda: 21.0)
    assert timings["setup_freeb_policy"] == 3.5


def test_record_elapsed_timing_accumulates_named_bucket_without_touching_disabled_clock() -> None:
    stats = {"bucket": 1.0}

    assert record_elapsed_timing(True, stats, "bucket", 2.0, lambda: 2.75)
    assert stats["bucket"] == pytest.approx(1.75)

    assert not record_elapsed_timing(False, stats, "bucket", 2.0, lambda: pytest.fail("disabled timing"))
    assert not record_elapsed_timing(True, stats, "bucket", None, lambda: pytest.fail("missing start"))
    assert stats["bucket"] == pytest.approx(1.75)


def test_new_residual_iter_timing_stats_preserves_setup_phase_values():
    setup = _initial_setup_phase_timings()
    setup["setup_freeb_policy"] = 1.25
    setup["setup_profile_data"] = 0.25
    setup["setup_trig_tables"] = 0.125
    setup["setup_update_constants"] = 0.5

    stats = _new_residual_iter_timing_stats(setup)

    assert stats["setup_freeb_policy"] == pytest.approx(1.25)
    assert stats["setup_profile_data"] == pytest.approx(0.25)
    assert stats["setup_trig_tables"] == pytest.approx(0.125)
    assert stats["setup_update_constants"] == pytest.approx(0.5)
    assert stats["setup_total"] == pytest.approx(0.0)
    assert stats["iterations"] == 0
    assert stats["compute_forces_calls"] == 0
    assert stats["compute_forces"] == pytest.approx(0.0)
    assert stats["iteration_control_fsq1_payload_get"] == pytest.approx(0.0)
    assert stats["finalize_residual_device_get"] == pytest.approx(0.0)


def test_resolve_residual_profile_window_parses_iteration_windows():
    disabled = resolve_residual_profile_window(profile_window_env="", profile_dir_env="/tmp/prof")
    assert disabled.started is False
    assert disabled.active is False
    assert disabled.start_iter is None
    assert disabled.directory == ""

    active = resolve_residual_profile_window(profile_window_env="iter12", profile_dir_env="/tmp/prof")
    assert active.started is False
    assert active.active is True
    assert active.start_iter == 12
    assert active.directory.endswith("window_iter12")

    invalid = resolve_residual_profile_window(profile_window_env="iterbad", profile_dir_env="/tmp/prof")
    assert invalid.active is False
    assert invalid.start_iter is None


def test_resolve_free_boundary_iteration_controls_disabled_skips_trace() -> None:
    trace_calls = []

    out = resolve_free_boundary_iteration_controls(
        free_boundary_enabled=False,
        controls_cached=None,
        iter2=5,
        iter1=1,
        ivac=-1,
        ivacskip=0,
        nvacskip=3,
        nvskip0=3,
        prev_rz_fsq=float("nan"),
        activate_fsq=None,
        iter_controls_func=lambda **_kwargs: pytest.fail("disabled path should not call controls"),
        dump_freeb_control_trace=lambda **kwargs: trace_calls.append(kwargs),
    )

    assert out.ivac == -1
    assert out.ivacskip == 0
    assert out.nvacskip == 3
    assert out.controls_cached is None
    assert out.turnon_iter is False
    assert out.ivac_effective == -1
    assert trace_calls == []


def test_resolve_free_boundary_iteration_controls_computes_and_caches_turnon() -> None:
    control_calls = []
    trace_calls = []

    def iter_controls_func(**kwargs):
        control_calls.append(kwargs)
        return 0, 0, 4

    out = resolve_free_boundary_iteration_controls(
        free_boundary_enabled=True,
        controls_cached=None,
        iter2=7,
        iter1=3,
        ivac=-1,
        ivacskip=0,
        nvacskip=2,
        nvskip0=2,
        prev_rz_fsq=float("nan"),
        activate_fsq=1.0e-6,
        iter_controls_func=iter_controls_func,
        dump_freeb_control_trace=lambda **kwargs: trace_calls.append(kwargs),
    )

    assert out.ivac == 0
    assert out.ivacskip == 0
    assert out.nvacskip == 4
    assert out.controls_cached == (0, 0, 4)
    assert out.turnon_iter is True
    assert out.ivac_effective == 1
    assert control_calls[0]["fsq_rz_prev"] == pytest.approx(1.0)
    assert trace_calls == [
        {
            "iter2": 7,
            "iter1": 3,
            "ivac": 0,
            "ivacskip": 0,
            "nvacskip": 4,
            "fsq_rz_prev": 1.0,
            "cached": False,
        }
    ]


def test_resolve_free_boundary_iteration_controls_reuses_cached_values() -> None:
    trace_calls = []

    out = resolve_free_boundary_iteration_controls(
        free_boundary_enabled=True,
        controls_cached=(2, 5, 8),
        iter2=9,
        iter1=4,
        ivac=1,
        ivacskip=0,
        nvacskip=3,
        nvskip0=3,
        prev_rz_fsq=0.25,
        activate_fsq=None,
        iter_controls_func=lambda **_kwargs: pytest.fail("cached path should not call controls"),
        dump_freeb_control_trace=lambda **kwargs: trace_calls.append(kwargs),
    )

    assert out.ivac == 2
    assert out.ivacskip == 5
    assert out.nvacskip == 8
    assert out.controls_cached == (2, 5, 8)
    assert out.turnon_iter is False
    assert out.ivac_effective == 2
    assert trace_calls[0]["cached"] is True
    assert trace_calls[0]["fsq_rz_prev"] == pytest.approx(0.25)


def test_dump_xc_with_velocity_blocks_forwards_legacy_velocity_names() -> None:
    calls = []
    velocities = ResidualVelocityBlocks(*(f"v{idx}" for idx in range(12)))

    got = dump_xc_with_velocity_blocks(
        dump_xc=lambda **kwargs: calls.append(kwargs) or "dumped",
        state="state",
        velocities=velocities,
        static="static",
        iter_idx=17,
    )

    assert got == "dumped"
    assert calls == [
        {
            "state": "state",
            "vRcc": "v0",
            "vRss": "v1",
            "vZsc": "v4",
            "vZcs": "v5",
            "vLsc": "v8",
            "vLcs": "v9",
            "vRsc": "v2",
            "vRcs": "v3",
            "vZcc": "v6",
            "vZss": "v7",
            "vLcc": "v10",
            "vLss": "v11",
            "static": "static",
            "iter_idx": 17,
        }
    ]


def test_update_timing_helpers_record_device_ready_and_total_update_time() -> None:
    class State:
        Rcos = "leaf"

    class FakeJax:
        calls = []

        @classmethod
        def block_until_ready(cls, value):
            cls.calls.append(value)

    times = iter([10.25, 10.75, 12.0])
    stats = {"update_state_ready": 0.0, "update_state": 0.0, "update": 0.0}

    assert record_update_state_ready_timing(
        timing_enabled=True,
        timing_stats=stats,
        start=10.0,
        state=State(),
        perf_counter=lambda: next(times),
        has_jax=lambda: True,
        jax_module=FakeJax,
    )
    assert stats["update_state_ready"] == pytest.approx(0.5)
    assert stats["update_state"] == pytest.approx(0.75)
    assert FakeJax.calls == ["leaf"]

    assert record_update_total_timing(
        timing_enabled=True,
        timing_stats=stats,
        start=11.0,
        state=State(),
        perf_counter=lambda: next(times),
        has_jax=lambda: True,
        jax_module=FakeJax,
    )
    assert stats["update"] == pytest.approx(1.0)
    assert FakeJax.calls == ["leaf", "leaf"]

    assert not record_update_total_timing(
        timing_enabled=False,
        timing_stats=stats,
        start=11.0,
        state=State(),
        perf_counter=lambda: pytest.fail("disabled timing should not sample the clock"),
        has_jax=lambda: True,
        jax_module=FakeJax,
    )


def test_ptau_dump_enabled_requires_env_and_directory():
    assert not _ptau_dump_enabled(dump_ptau_env="", dump_dir="/tmp")
    assert not _ptau_dump_enabled(dump_ptau_env="0", dump_dir="/tmp")
    assert not _ptau_dump_enabled(dump_ptau_env="1", dump_dir="")
    assert _ptau_dump_enabled(dump_ptau_env="yes", dump_dir="/tmp")


def test_format_ptau_dump_row_matches_legacy_fields():
    row = _format_ptau_dump_row(
        iter_idx=7,
        ptau_min=-1.25,
        ptau_max=2.5,
        tau_min_state=None,
        tau_max_state=3.0,
        badjac_ptau=None,
        badjac_state=True,
        badjac_used=False,
        mode="scan",
        label="iter",
    )

    assert row == (
        "7 iter scan -1.2500000000000000e+00 2.5000000000000000e+00 "
        "nan 3.0000000000000000e+00 -1 1 0\n"
    )


def test_maybe_dump_ptau_writes_header_and_swallows_io_failures(tmp_path):
    assert not _maybe_dump_ptau(
        iter_idx=1,
        ptau_min=0.0,
        ptau_max=0.0,
        tau_min_state=None,
        tau_max_state=None,
        badjac_ptau=None,
        badjac_state=None,
        badjac_used=False,
        mode="host",
        label="skip",
        dump_ptau_env="0",
        dump_dir=str(tmp_path),
    )

    assert _maybe_dump_ptau(
        iter_idx=2,
        ptau_min=-0.5,
        ptau_max=0.75,
        tau_min_state=-0.25,
        tau_max_state=0.25,
        badjac_ptau=True,
        badjac_state=False,
        badjac_used=True,
        mode="host",
        label="probe",
        dump_ptau_env="1",
        dump_dir=str(tmp_path),
    )
    lines = (tmp_path / "ptau_minmax.log").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "iter label mode ptau_min ptau_max state_min state_max bad_ptau bad_state bad_used"
    assert lines[1].startswith("2 probe host -5.0000000000000000e-01 7.5000000000000000e-01")
    assert lines[1].endswith("1 0 1")

    assert not _maybe_dump_ptau(
        iter_idx=3,
        ptau_min=0.0,
        ptau_max=0.0,
        tau_min_state=None,
        tau_max_state=None,
        badjac_ptau=None,
        badjac_state=None,
        badjac_used=False,
        mode="host",
        label="bad",
        dump_ptau_env="1",
        dump_dir=str(tmp_path / "missing" / "nested"),
    )


def test_scan_block_until_ready_falls_back_to_tree_leaves():
    class ReadyLeaf:
        def __init__(self):
            self.ready = False

        def block_until_ready(self):
            self.ready = True
            return self

    leaf = ReadyLeaf()

    def block_until_ready(_value):
        raise RuntimeError("not a whole-tree value")

    def tree_map(fn, value):
        return {key: fn(item) for key, item in value.items()}

    result = _scan_block_until_ready(
        {"leaf": leaf, "plain": 3},
        block_until_ready=block_until_ready,
        tree_map=tree_map,
    )

    assert result["leaf"] is leaf
    assert leaf.ready
    assert result["plain"] == 3


def test_scan_device_run_ready_records_only_when_enabled():
    times = iter([10.25, 11.0])
    stats = {"scan_device_dispatch_s": 0.0, "scan_device_ready_s": 0.0, "scan_device_run_s": 0.0}
    calls = []

    assert _scan_device_run_ready(
        start=None,
        value="value",
        scan_timing_enabled=True,
        perf_counter=lambda: next(times),
        block_until_ready=lambda value: value,
        tree_map=lambda fn, value: value,
        record_ready=lambda **kwargs: calls.append(kwargs) or True,
        stats=stats,
    ) == "value"
    assert calls == []

    result = _scan_device_run_ready(
        start=10.0,
        value="value",
        scan_timing_enabled=True,
        perf_counter=lambda: next(times),
        block_until_ready=lambda value: f"{value}-ready",
        tree_map=lambda fn, value: value,
        record_ready=lambda **kwargs: calls.append(kwargs) or True,
        stats=stats,
    )

    assert result == "value-ready"
    assert calls == [
        {
            "start": 10.0,
            "dispatch_done": 10.25,
            "ready_done": 11.0,
            "stats": stats,
            "cache_status": None,
        }
    ]


def test_record_compute_force_timing_updates_main_and_labeled_counters():
    times = iter([2.5, 5.0, 7.0])
    stats = {
        "compute_forces": 0.0,
        "compute_forces_first": 0.0,
        "compute_forces_rest": 0.0,
        "compute_forces_calls": 0,
        "compute_forces_trial": 0.0,
        "compute_forces_trial_calls": 0,
    }
    ready = []

    assert not _record_compute_force_timing(
        "main",
        None,
        "skip",
        timing_enabled=True,
        timing_stats=stats,
        perf_counter=lambda: next(times),
        block_until_ready=lambda value: ready.append(value),
    )
    assert stats["compute_forces_calls"] == 0

    assert _record_compute_force_timing(
        "main",
        1.0,
        "first",
        timing_enabled=True,
        timing_stats=stats,
        perf_counter=lambda: next(times),
        block_until_ready=lambda value: ready.append(value),
    )
    assert stats["compute_forces"] == pytest.approx(1.5)
    assert stats["compute_forces_first"] == pytest.approx(1.5)
    assert stats["compute_forces_rest"] == pytest.approx(0.0)
    assert stats["compute_forces_calls"] == 1
    assert ready == ["first"]

    assert _record_compute_force_timing(
        "main",
        4.0,
        "second",
        timing_enabled=True,
        timing_stats=stats,
        perf_counter=lambda: next(times),
        block_until_ready=lambda value: ready.append(value),
    )
    assert stats["compute_forces"] == pytest.approx(2.5)
    assert stats["compute_forces_first"] == pytest.approx(1.5)
    assert stats["compute_forces_rest"] == pytest.approx(1.0)
    assert stats["compute_forces_calls"] == 2

    assert _record_compute_force_timing(
        "trial",
        6.25,
        "trial",
        timing_enabled=True,
        timing_stats=stats,
        perf_counter=lambda: next(times),
        block_until_ready=lambda value: (_ for _ in ()).throw(RuntimeError("sync failed")),
    )
    assert stats["compute_forces_trial"] == pytest.approx(0.75)
    assert stats["compute_forces_trial_calls"] == 1


def test_record_compute_force_timing_disabled_is_noop():
    stats = {"compute_forces": 0.0, "compute_forces_calls": 0}
    assert not _record_compute_force_timing(
        "main",
        1.0,
        "value",
        timing_enabled=False,
        timing_stats=stats,
        perf_counter=lambda: 2.0,
        block_until_ready=None,
    )
    assert stats == {"compute_forces": 0.0, "compute_forces_calls": 0}


def test_converged_residuals_scan_fast_supports_strict_and_total_target():
    assert _converged_residuals_scan_fast(0.1, 0.2, 0.3, ftol=0.25, fsq_total_target=None) is False
    assert _converged_residuals_scan_fast(0.1, 0.2, 0.3, ftol=0.4, fsq_total_target=None) is True
    assert _converged_residuals_scan_fast(0.4, 0.4, 0.1, ftol=0.2, fsq_total_target=1.0) is True

    arr = _converged_residuals_scan_fast(
        np.asarray([0.1, 0.4]),
        np.asarray([0.1, 0.4]),
        np.asarray([0.1, 0.4]),
        ftol=0.2,
        fsq_total_target=0.9,
    )
    np.testing.assert_array_equal(arr, np.asarray([True, False]))


def test_vmec_freeb_plascur_from_bcovar_uses_finite_value_or_fallback():
    def plascur_edge_from_bcovar(**kwargs):
        assert kwargs == {"bc": "bc", "trig": "trig", "wout": "wout", "s": "s"}
        return np.asarray(4.25)

    assert _vmec_freeb_plascur_from_bcovar(
        "bc",
        1.5,
        plascur_edge_from_bcovar=plascur_edge_from_bcovar,
        trig="trig",
        wout="wout",
        s="s",
    ) == 4.25

    assert _vmec_freeb_plascur_from_bcovar(
        "bc",
        1.5,
        plascur_edge_from_bcovar=lambda **_: np.nan,
        trig=None,
        wout=None,
        s=None,
    ) == 1.5
    assert _vmec_freeb_plascur_from_bcovar(
        "bc",
        1.5,
        plascur_edge_from_bcovar=lambda **_: (_ for _ in ()).throw(RuntimeError("bad")),
        trig=None,
        wout=None,
        s=None,
    ) == 1.5


def test_free_boundary_loop_state_initializes_and_resumes_vmec_cadence():
    class WoutLike:
        icurv = np.asarray([0.0, 2.0])

    loop_state = initial_free_boundary_loop_state(nvacskip=3, nvskip0=4, wout_like=WoutLike())
    assert loop_state.ivac == -1
    assert loop_state.ivacskip == 0
    assert loop_state.nvacskip == 3
    assert loop_state.nvskip0 == 4
    assert loop_state.last_model == "none"
    assert loop_state.last_diagnostics == {}
    assert loop_state.plascur == pytest.approx(4.0 * np.pi)

    disabled_resume = resume_free_boundary_loop_state(
        loop_state,
        resume_state={"freeb_ivac": 9, "freeb_ivacskip": 8, "freeb_nvacskip": 0, "freeb_nvskip0": 0},
        free_boundary_enabled=False,
    )
    assert disabled_resume == loop_state

    resumed = resume_free_boundary_loop_state(
        loop_state,
        resume_state={
            "freeb_ivac": 9,
            "freeb_ivacskip": 8,
            "freeb_nvacskip": 0,
            "freeb_nvskip0": 0,
            "freeb_model": "direct-coil",
        },
        free_boundary_enabled=True,
    )
    assert resumed.ivac == 9
    assert resumed.ivacskip == 8
    assert resumed.nvacskip == 1
    assert resumed.nvskip0 == 1
    assert resumed.last_model == "direct-coil"
    assert resumed.plascur == loop_state.plascur


def test_trial_residual_total_runtime_records_timing_and_sums_residuals():
    force_calls = []
    timing_calls = []

    def compute_forces_iter(candidate_state, **kwargs):
        force_calls.append((candidate_state, kwargs))
        return None, None, np.asarray(1.0), np.asarray(2.0), np.asarray(3.0), None, None, "norms"

    residual = trial_residual_total_runtime(
        "candidate",
        "bsqvac",
        zero_m1_value="zero-m1",
        timing_label="trial",
        compute_forces_iter_func=compute_forces_iter,
        include_edge=True,
        constraint_precond_diag="diag",
        constraint_tcon="tcon",
        constraint_precond_active=True,
        constraint_tcon_active=False,
        iter2=7,
        timing_detail_enabled=True,
        perf_counter=lambda: 1.25,
        record_compute_force_timing=lambda *args: timing_calls.append(args),
        residual_fsq_from_norms_func=lambda norms, **kwargs: (
            kwargs["gcr2"] + 10.0,
            kwargs["gcz2"] + 20.0,
            kwargs["gcl2"] + 30.0,
        ),
        numpy_module=np,
    )

    assert residual == pytest.approx(66.0)
    assert force_calls == [
        (
            "candidate",
            {
                "include_edge": True,
                "zero_m1": "zero-m1",
                "freeb_bsqvac_half": "bsqvac",
                "constraint_precond_diag": "diag",
                "constraint_tcon": "tcon",
                "constraint_precond_active": True,
                "constraint_tcon_active": False,
                "iter2": 7,
            },
        )
    ]
    assert len(timing_calls) == 1
    assert timing_calls[0][0] == "trial"
    assert timing_calls[0][1] == 1.25
    np.testing.assert_allclose(timing_calls[0][2], np.asarray(1.0))


def test_attach_free_boundary_external_field_diag_branches():
    res = _Result()
    assert (
        _attach_free_boundary_external_field_diag(
            res,
            free_boundary_enabled=False,
            external_field_provider_kind=None,
            freeb_sample_external=True,
            sample_external_field_func=lambda **_: {"unused": True},
            static="static",
            result_type=_result_type,
        )
        is res
    )

    direct = _attach_free_boundary_external_field_diag(
        _Result(),
        free_boundary_enabled=True,
        external_field_provider_kind="direct_coils",
        freeb_sample_external=True,
        sample_external_field_func=lambda **_: {"unused": True},
        static="static",
        result_type=_result_type,
    )
    assert direct.diagnostics["free_boundary_external_field"] == {
        "enabled": True,
        "available": False,
        "provider_kind": "direct_coils",
        "reason": "direct_provider_runtime_path",
    }

    sampled_calls = []
    sampled = _attach_free_boundary_external_field_diag(
        _Result(),
        free_boundary_enabled=True,
        external_field_provider_kind="mgrid",
        freeb_sample_external=True,
        sample_external_field_func=lambda **kwargs: sampled_calls.append(kwargs) or {"available": True},
        static="static",
        result_type=_result_type,
    )
    assert sampled.diagnostics["free_boundary_external_field"] == {"available": True}
    assert sampled_calls == [{"state": "state", "static": "static"}]

    disabled = _attach_free_boundary_external_field_diag(
        _Result(),
        free_boundary_enabled=True,
        external_field_provider_kind=None,
        freeb_sample_external=False,
        sample_external_field_func=lambda **_: {"unused": True},
        static="static",
        result_type=_result_type,
    )
    assert disabled.diagnostics["free_boundary_external_field"] == {
        "enabled": False,
        "available": False,
        "vacuum_stub": True,
        "reason": "disabled_by_env",
    }

    existing = _attach_free_boundary_external_field_diag(
        _Result({"free_boundary_external_field": {"kept": True}}),
        free_boundary_enabled=True,
        external_field_provider_kind="mgrid",
        freeb_sample_external=True,
        sample_external_field_func=lambda **_: {"unused": True},
        static="static",
        result_type=_result_type,
    )
    assert existing.diagnostics["free_boundary_external_field"] == {"kept": True}
    assert isinstance(existing.w_history, np.ndarray)


@pytest.mark.parametrize(
    ("mode", "debug_print_fn", "expected"),
    [("debug_print", object(), True), ("debug_print", None, False), ("debug_callback", object(), False)],
)
def test_scan_print_uses_debug_print(mode, debug_print_fn, expected):
    assert _scan_print_uses_debug_print(scan_print_mode=mode, debug_print_fn=debug_print_fn) is expected


@pytest.mark.parametrize(
    ("mode", "debug_module", "expected"),
    [("debug_callback", object(), True), ("debug_callback", None, False), ("debug_print", object(), False)],
)
def test_scan_print_uses_debug_callback(mode, debug_module, expected):
    assert _scan_print_uses_debug_callback(scan_print_mode=mode, debug_module=debug_module) is expected


@pytest.mark.parametrize(
    ("mode", "io_callback_fn", "expected"),
    [("io_callback", object(), True), ("io_callback", None, False), ("debug_print", object(), False)],
)
def test_scan_print_uses_io_callback(mode, io_callback_fn, expected):
    assert _scan_print_uses_io_callback(scan_print_mode=mode, io_callback_fn=io_callback_fn) is expected


class _State:
    def __init__(self):
        self.Rcos = np.asarray([[1.0, 2.0]])
        self.Zsin = np.asarray([[3.0, 4.0]])
        self.Lsin = np.asarray([[5.0, 6.0]])


class _Norms:
    fnorm = 2.0
    fnormL = 3.0
    r1 = 4.0


def test_nonscan_state_debug_payload_and_optional_print():
    state = _State()
    checkpoint = _State()
    checkpoint.Rcos = np.asarray([[10.0]])
    checkpoint.Zsin = np.asarray([[20.0]])
    checkpoint.Lsin = np.asarray([[30.0]])

    payload = _nonscan_state_debug_payload(
        state=state,
        state_checkpoint=checkpoint,
        gcr2=np.asarray(0.5),
        gcz2=np.asarray(0.25),
        gcl2=np.asarray(0.75),
        norms_used=_Norms(),
    )

    assert payload["rcos_sum"] == 3.0
    assert payload["zsin_ck"] == 20.0
    assert payload["fsqr"] == 4.0
    assert payload["fsqz"] == 2.0
    assert payload["fsql"] == 2.25

    rows = []
    emitted = _maybe_print_nonscan_state_debug(
        debug_iter_env="7",
        iter2=7,
        state=state,
        state_checkpoint=checkpoint,
        gcr2=0.5,
        gcz2=0.25,
        gcl2=0.75,
        norms_used=_Norms(),
        print_fn=lambda text, **kwargs: rows.append((text, kwargs)),
    )
    assert emitted
    assert rows[0][0].startswith("[nonscan-state] iter=7")
    assert rows[0][1] == {"flush": True}

    assert not _maybe_print_nonscan_state_debug(
        debug_iter_env="bad",
        iter2=7,
        state=state,
        state_checkpoint=checkpoint,
        gcr2=0.5,
        gcz2=0.25,
        gcl2=0.75,
        norms_used=_Norms(),
        print_fn=lambda *_args, **_kwargs: None,
    )
    assert not _maybe_print_nonscan_state_debug(
        debug_iter_env="7",
        iter2=7,
        state=state,
        state_checkpoint=checkpoint,
        gcr2=0.5,
        gcz2=0.25,
        gcl2=0.75,
        norms_used=_Norms(),
        print_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("closed pipe")),
    )


def test_residual_iter_timing_report_and_message():
    stats = {
        "iterations": 4,
        "setup_total": 10.0,
        "setup_axis_reset": 3.0,
        "setup_axis_reset_compute_forces": 1.0,
        "iteration_loop": 20.0,
        "iteration_prepare": 1.0,
        "compute_forces": 8.0,
        "compute_forces_first": 3.0,
        "compute_forces_rest": 5.0,
        "compute_forces_calls": 9,
        "compute_forces_main_reuse_count": 2,
        "iteration_residual_metrics": 2.0,
        "preconditioner": 3.0,
        "precond_refresh": 1.5,
        "update": 4.0,
        "update_state": 2.5,
        "update_trace_build": 0.25,
        "update_trace_finalize": 0.5,
        "iteration_post_update": 1.0,
        "finalize": 0.75,
        "precond_apply": 1.25,
        "precond_mode_scale": 0.5,
    }

    report = _build_residual_iter_timing_report(
        stats,
        solve_total_s=42.0,
        timing_detail_enabled=True,
    )

    assert report["iterations"] == 4
    assert report["setup_unattributed_s"] == 7.0
    assert report["setup_axis_reset_unattributed_s"] == 2.0
    assert report["iteration_loop_unattributed_s"] == 1.0
    assert report["compute_forces_per_iter_s"] == 2.0
    assert report["compute_forces_main_reuse_count"] == 2
    assert report["precond_apply_per_iter_s"] == 0.3125

    msg = _format_residual_iter_timing_message(report, timing_detail_enabled=True)
    assert "iters=4" in msg
    assert "compute_forces=8.000e+00s" in msg
    assert "precond_apply=1.250e+00s" in msg


def test_build_resume_state_base_counts_optional_free_boundary_runtime():
    class Runtime:
        update_count = "5"
        reuse_count = "6"

    base = _build_resume_state_base(
        time_step=0.25,
        inv_tau=[1.0, 2.0],
        fsq_prev=3.0,
        fsq0_prev=4.0,
        flip_sign=-1.0,
        iter1=8,
        last_iter2=9,
        ijacob=2,
        bad_resets=1,
        res0=0.1,
        res1=0.2,
        prev_rz_fsq=0.3,
        bad_growth_streak=4,
        huge_force_restart_count=5,
        vmec2000_cache_valid=True,
        freeb_ivac=1,
        freeb_ivacskip=2,
        freeb_nvacskip=3,
        freeb_nvskip0=4,
        freeb_last_model="nestor",
        freeb_nestor_runtime=Runtime(),
    )

    assert base["time_step"] == 0.25
    assert base["iter_offset"] == 9
    assert base["freeb_model"] == "nestor"
    assert base["freeb_nestor_update_count"] == 5
    assert base["freeb_nestor_reuse_count"] == 6

    base_none = _build_resume_state_base(
        time_step=0.25,
        inv_tau=[],
        fsq_prev=0.0,
        fsq0_prev=0.0,
        flip_sign=1.0,
        iter1=0,
        last_iter2=0,
        ijacob=0,
        bad_resets=0,
        res0=0.0,
        res1=0.0,
        prev_rz_fsq=0.0,
        bad_growth_streak=0,
        huge_force_restart_count=0,
        vmec2000_cache_valid=False,
        freeb_ivac=0,
        freeb_ivacskip=0,
        freeb_nvacskip=0,
        freeb_nvskip0=0,
        freeb_last_model="none",
        freeb_nestor_runtime=None,
    )
    assert base_none["freeb_nestor_update_count"] == 0
    assert base_none["freeb_nestor_reuse_count"] == 0
