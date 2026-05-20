from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pytest

import vmec_jax.discrete_adjoint as da
from vmec_jax.state import StateLayout, VMECState


def _fake_tape(nbytes: int):
    # A uint8 buffer makes the size accounting in _replay_column_chunk_default
    # easy to reason about without allocating large arrays in the test.
    return SimpleNamespace(
        dynamic_initial_carry=(np.zeros(nbytes, dtype=np.uint8),),
        dynamic_base_carries_stacked=None,
    )


def _fake_dynamic_trace(
    *,
    lambda_update_scale: float = 1.0,
    max_coeff_delta_rms_pre: float = 0.1,
    max_update_rms_pre: float = 0.01,
) -> dict:
    layout = StateLayout(ns=1, K=1, lasym=False)
    z = np.zeros((1, 1), dtype=float)
    state = VMECState(
        layout=layout,
        Rcos=z,
        Rsin=z,
        Zcos=z,
        Zsin=z,
        Lcos=z,
        Lsin=z,
    )
    return {
        "state_pre": state,
        "wout_like": "wout",
        "trig": "trig",
        "w_mode_mn": np.asarray([1.0]),
        "lambda_update_scale": float(lambda_update_scale),
        "dt_eff": 1.0,
        "b1": 1.0,
        "fac": 1.0,
        "force_scale": 1.0,
        "flip_sign": False,
        "time_step": 1.0,
        "fsq_prev_before": 1.0,
        "reset_inv_tau": False,
        "inv_tau_before": np.asarray([1.0]),
        "max_coeff_delta_rms_pre": float(max_coeff_delta_rms_pre),
        "max_update_rms_pre": float(max_update_rms_pre),
        "zero_m1": np.asarray([False]),
        "precond_mats": None,
        "precond_jmax": 1,
        "lam_prec": None,
        "apply_lforbal": False,
        "include_edge_residual": True,
        "apply_m1_constraints": True,
        "limit_update_rms": False,
        "limit_dt_from_force": False,
        "vmec2000_control": True,
        "divide_by_scalxc_for_update": False,
        "signgs": 1,
        "vRcc_before": z,
        "vRss_before": z,
        "vRsc_before": z,
        "vRcs_before": z,
        "vZsc_before": z,
        "vZcs_before": z,
        "vZcc_before": z,
        "vZss_before": z,
        "vLsc_before": z,
        "vLcs_before": z,
        "vLcc_before": z,
        "vLss_before": z,
    }


def _fake_supported_dynamic_trace(**overrides) -> dict:
    trace = _fake_dynamic_trace()
    trace.update(
        {
            "branch": "strict_update",
            "step_status": "momentum",
            "restart_reason": "none",
            "restart_path": "momentum_accept",
        }
    )
    trace.update(overrides)
    return trace


def _fake_restart_trace(**overrides) -> dict:
    trace = _fake_supported_dynamic_trace(
        step_status="restart_bad_progress",
        restart_reason="bad_progress",
        restart_path="catastrophic_growth",
    )
    trace.update(overrides)
    return trace


def _fake_carry_stacked(width: int = 3):
    return (np.zeros((1, width), dtype=float),) + tuple(np.zeros((1, 1), dtype=float) for _ in range(14))


def test_replay_column_chunk_default_honors_env_target(monkeypatch):
    monkeypatch.setenv("VMEC_JAX_REPLAY_COLUMN_TARGET_MB", "0.001")
    tape = _fake_tape(128)
    tangents = np.zeros((10, 1), dtype=float)

    chunk = da._replay_column_chunk_default(tape=tape, tangents=tangents)

    assert chunk == 8


def test_replay_column_chunk_default_uses_module_default(monkeypatch):
    monkeypatch.delenv("VMEC_JAX_REPLAY_COLUMN_TARGET_MB", raising=False)
    monkeypatch.setattr(da, "_DEFAULT_REPLAY_COLUMN_TARGET_MB", 0.0002)
    tape = _fake_tape(128)
    tangents = np.zeros((10, 1), dtype=float)

    chunk = da._replay_column_chunk_default(tape=tape, tangents=tangents)

    assert chunk == 1


def test_replay_column_target_default_is_relaxed_after_memory_fix():
    assert da._DEFAULT_REPLAY_COLUMN_TARGET_MB == 4096.0


def test_dynamic_replay_bucket_default_is_modest(monkeypatch):
    monkeypatch.delenv("VMEC_JAX_DYNAMIC_REPLAY_BUCKET", raising=False)

    assert da._dynamic_replay_bucket_size() == 32
    assert da._dynamic_replay_bucket_len(33) == 64


def test_dynamic_replay_bucket_default_is_larger_on_gpu(monkeypatch):
    monkeypatch.delenv("VMEC_JAX_DYNAMIC_REPLAY_BUCKET", raising=False)
    monkeypatch.setattr(da.jax, "default_backend", lambda: "gpu")

    assert da._dynamic_replay_bucket_size() == 128
    assert da._dynamic_replay_bucket_len(129) == 256


def test_dynamic_replay_bucket_malformed_env_uses_backend_default(monkeypatch):
    monkeypatch.setenv("VMEC_JAX_DYNAMIC_REPLAY_BUCKET", "bad")
    monkeypatch.setattr(da.jax, "default_backend", lambda: "gpu")

    assert da._dynamic_replay_bucket_size() == 128
    assert da._dynamic_replay_bucket_len(129) == 256


def test_dynamic_replay_bucket_honors_env(monkeypatch):
    monkeypatch.setenv("VMEC_JAX_DYNAMIC_REPLAY_BUCKET", "128")

    assert da._dynamic_replay_bucket_size() == 128
    assert da._dynamic_replay_bucket_len(129) == 256


def test_dynamic_replay_payload_passes_scalar_controls_as_trace_inputs(monkeypatch):
    monkeypatch.setenv("VMEC_JAX_DYNAMIC_REPLAY_BUCKET", "4")
    traces = (_fake_dynamic_trace(lambda_update_scale=0.5), _fake_dynamic_trace(lambda_update_scale=0.5))
    static_flags = da._static_flags_from_replay_step_traces(traces)

    stacked, dynamic_flags, _initial_carry, _base_carries = da._build_dynamic_replay_payload(
        traces,
        static_flags,
    )

    for key in da._DYNAMIC_REPLAY_SCALAR_TRACE_KEYS:
        assert key in stacked
        assert key not in dynamic_flags
        assert np.asarray(stacked[key]).shape == (4,)
    np.testing.assert_allclose(np.asarray(stacked["lambda_update_scale"])[:2], [0.5, 0.5])
    assert not bool(np.asarray(stacked["active"])[-1])


def test_dynamic_replay_payload_signature_reuses_scalar_value_changes(monkeypatch):
    monkeypatch.setenv("VMEC_JAX_DYNAMIC_REPLAY_BUCKET", "4")
    traces_a = (_fake_dynamic_trace(lambda_update_scale=0.5),)
    traces_b = (_fake_dynamic_trace(lambda_update_scale=0.75, max_update_rms_pre=0.03),)
    flags_a = da._static_flags_from_replay_step_traces(traces_a)
    flags_b = da._static_flags_from_replay_step_traces(traces_b)

    stacked_a, _dynamic_flags_a, _initial_a, _base_a = da._build_dynamic_replay_payload(traces_a, flags_a)
    stacked_b, _dynamic_flags_b, _initial_b, _base_b = da._build_dynamic_replay_payload(traces_b, flags_b)

    assert da._stacked_trace_signature(stacked_a) == da._stacked_trace_signature(stacked_b)
    assert float(np.asarray(stacked_a["lambda_update_scale"])[0]) == 0.5
    assert float(np.asarray(stacked_b["lambda_update_scale"])[0]) == 0.75
    assert float(np.asarray(stacked_b["max_update_rms_pre"])[0]) == 0.03


def test_single_state_jvp_uses_dynamic_column_replay(monkeypatch):
    calls = []

    def fake_columns(*, tape, static, initial_tangents, rebuild_preconditioner):
        calls.append((tape, static, bool(rebuild_preconditioner)))
        return np.asarray(initial_tangents) + 1.0

    monkeypatch.setattr(da, "checkpoint_tape_state_jvp_columns", fake_columns)
    tape = SimpleNamespace(dynamic_initial_carry=(np.zeros(1),), step_traces=())

    out = da.checkpoint_tape_state_jvp(
        tape=tape,
        static="static",
        initial_tangent=np.array([2.0, 3.0]),
        rebuild_preconditioner=True,
    )

    np.testing.assert_allclose(out, np.array([3.0, 4.0]))
    assert len(calls) == 1
    assert calls[0][0] is tape
    assert calls[0][1:] == ("static", True)


def test_scan_cache_limit_lru_and_clear(monkeypatch):
    monkeypatch.setenv("VMEC_JAX_SCAN_CACHE_LIMIT", "bad")
    assert da._scan_cache_limit() == 8

    monkeypatch.setenv("VMEC_JAX_SCAN_CACHE_LIMIT", "0")
    assert da._scan_cache_limit() == 1

    monkeypatch.setenv("VMEC_JAX_SCAN_CACHE_LIMIT", "2")
    cache = OrderedDict()
    da._lru_cache_put(cache, ("a",), "A")
    da._lru_cache_put(cache, ("b",), "B")
    assert da._lru_cache_get(cache, ("a",)) == "A"
    da._lru_cache_put(cache, ("c",), "C")
    assert list(cache.keys()) == [("a",), ("c",)]
    assert da._lru_cache_get(cache, ("missing",)) is None

    da._CHECKPOINT_TAPE_SCAN_CACHE[("scan",)] = object()
    da._CHECKPOINT_TAPE_DYNAMIC_SCAN_CACHE[("dynamic",)] = object()
    da._CHECKPOINT_TAPE_DYNAMIC_BASEPOINT_SCAN_CACHE[("basepoint",)] = object()
    da._CHECKPOINT_TAPE_DYNAMIC_BASEPOINT_VJP_SCAN_CACHE[("vjp",)] = object()
    da.clear_replay_scan_caches()
    assert not da._CHECKPOINT_TAPE_SCAN_CACHE
    assert not da._CHECKPOINT_TAPE_DYNAMIC_SCAN_CACHE
    assert not da._CHECKPOINT_TAPE_DYNAMIC_BASEPOINT_SCAN_CACHE
    assert not da._CHECKPOINT_TAPE_DYNAMIC_BASEPOINT_VJP_SCAN_CACHE


def test_dynamic_replay_mode_env_aliases(monkeypatch):
    monkeypatch.setenv("VMEC_JAX_DYNAMIC_REPLAY_MODE", "scan")
    assert da._dynamic_replay_mode() == "whole_scan"

    monkeypatch.setenv("VMEC_JAX_DYNAMIC_REPLAY_MODE", "unknown")
    assert da._dynamic_replay_mode() == "basepoint"


def test_replay_column_chunk_default_shape_guards(monkeypatch):
    monkeypatch.setenv("VMEC_JAX_REPLAY_COLUMN_TARGET_MB", "0")
    assert da._replay_column_chunk_default(tape=_fake_tape(128), tangents=np.zeros((10, 1))) is None

    monkeypatch.setenv("VMEC_JAX_REPLAY_COLUMN_TARGET_MB", "1")
    assert da._replay_column_chunk_default(tape=SimpleNamespace(), tangents=np.zeros((10, 1))) is None
    assert da._replay_column_chunk_default(tape=_fake_tape(128), tangents=np.zeros((1, 1))) is None
    assert da._replay_column_chunk_default(tape=_fake_tape(1), tangents=np.zeros((2, 1))) is None


def test_replay_column_chunk_override_parses_policy_values():
    assert da._replay_column_chunk_override(None) == (False, None)
    assert da._replay_column_chunk_override("auto") == (False, None)
    assert da._replay_column_chunk_override("bad") == (False, None)
    assert da._replay_column_chunk_override("off") == (True, None)
    assert da._replay_column_chunk_override("-3") == (True, None)
    assert da._replay_column_chunk_override("3") == (True, 3)


def test_residual_iteration_trace_guards_and_rejected_statuses():
    with pytest.raises(TypeError, match="diagnostics"):
        da.residual_iteration_trace_from_result(SimpleNamespace(diagnostics=None))

    with pytest.raises(ValueError, match="inconsistent residual trace lengths"):
        da.residual_iteration_trace_from_result(
            SimpleNamespace(
                diagnostics={
                    "iter2_history": np.asarray([1, 2]),
                    "step_status_history": np.asarray(["momentum"], dtype=object),
                }
            )
        )

    trace = da.residual_iteration_trace_from_result(
        SimpleNamespace(
            diagnostics={
                "step_status_history": np.asarray(
                    ["momentum", "rejected", "restart_bad_jacobian"],
                    dtype=object,
                )
            }
        )
    )
    np.testing.assert_array_equal(trace.state_advanced, np.asarray([True, False, False]))


def test_compact_tape_diagnostics_filters_host_only_values():
    out = da._compact_tape_diagnostics(
        {
            "timing": {
                "iterations": np.float64(4.0),
                "solve_s": np.float64(1.25),
                "enabled": True,
                "label": "skip",
                "bad": object(),
            },
            "converged": np.bool_(True),
            "converged_iter": np.int64(3),
            "final_fsq": np.float64(0.125),
            "final_fsqr": object(),
        }
    )

    assert out == {
        "timing": {"iterations": 4, "solve_s": 1.25},
        "converged": True,
        "converged_iter": 3,
        "final_fsq": 0.125,
    }


def test_direct_checkpoint_tape_records_build_leaf_timing(monkeypatch):
    trace = _fake_dynamic_trace()
    state = trace["state_pre"]

    def fake_solve(state0, static, *, max_iter, adjoint_trace, **kwargs):
        assert state0 is state
        assert static == "static"
        assert max_iter == 3
        assert adjoint_trace is True
        return SimpleNamespace(
            state=state,
            diagnostics={
                "timing": {
                    "setup_total_s": 0.2,
                    "iteration_loop_s": 0.7,
                    "compute_forces_s": 0.1,
                    "iteration_residual_metrics_s": 0.3,
                    "iteration_loop_unattributed_s": 0.1,
                    "finalize_s": 0.05,
                },
                "adjoint_step_trace": [trace],
                "converged": True,
            },
        )

    monkeypatch.setattr("vmec_jax.solve.solve_fixed_boundary_residual_iter", fake_solve)
    monkeypatch.setattr(da, "_dynamic_replay_supported", lambda *, tape, rebuild_preconditioner: True)
    monkeypatch.setattr(
        da,
        "_build_dynamic_replay_payload",
        lambda step_traces, static_flags: ("dynamic", {"dynamic": True}, "carry0", "base_carries"),
    )
    monkeypatch.setattr(da, "_stack_replay_step_traces", lambda step_traces: ("stacked", {"stacked": True}))

    tape = da.build_residual_checkpoint_tape_direct(
        state,
        "static",
        indata={},
        signgs=1,
        max_iter=3,
        ftol=0.0,
        store_full_step_traces=True,
    )

    timing = tape.diagnostics["timing"]
    assert timing["setup_total_s"] == 0.2
    assert timing["iteration_loop_s"] == 0.7
    assert timing["compute_forces_s"] == 0.1
    assert timing["iteration_residual_metrics_s"] == 0.3
    assert timing["iteration_loop_unattributed_s"] == 0.1
    assert timing["finalize_s"] == 0.05
    for key in (
        "tape_solve_call_s",
        "tape_final_state_pack_s",
        "tape_step_trace_extract_s",
        "tape_dynamic_payload_build_s",
        "tape_trace_stack_s",
    ):
        assert key in timing
        assert timing[key] >= 0.0
    assert tape.dynamic_initial_carry == "carry0"
    assert tape.dynamic_base_carries_stacked == "base_carries"
    assert tape.stacked_step_traces == "stacked"
    assert tape.step_trace_static_flags == {"stacked": True}


def test_direct_checkpoint_tape_reruns_full_trace_when_dynamic_trace_unsupported(monkeypatch):
    trace_dynamic = _fake_dynamic_trace()
    trace_full = _fake_supported_dynamic_trace(lambda_update_scale=0.75)
    state = trace_dynamic["state_pre"]
    calls = []

    def fake_solve(state0, static, *, adjoint_trace, **kwargs):
        assert state0 is state
        assert static == "static"
        assert adjoint_trace is True
        calls.append(kwargs.get("adjoint_trace_mode"))
        trace = trace_dynamic if len(calls) == 1 else trace_full
        return SimpleNamespace(
            state=state,
            diagnostics={
                "adjoint_step_trace": [trace],
                "timing": {"iterations": len(calls)},
            },
        )

    stack_calls = []

    def fake_stack(step_traces):
        stack_calls.append(step_traces)
        return {"stacked": len(step_traces)}, {"precond_jmax": 1}

    monkeypatch.setattr("vmec_jax.solve.solve_fixed_boundary_residual_iter", fake_solve)
    monkeypatch.setattr(da, "_stack_replay_step_traces", fake_stack)

    tape = da.build_residual_checkpoint_tape_direct(
        state,
        "static",
        indata={},
        signgs=1,
        max_iter=2,
        store_full_step_traces=False,
    )

    assert calls == ["dynamic", "full"]
    assert tape.step_traces == (trace_full,)
    assert tape.stacked_step_traces == {"stacked": 1}
    assert tape.step_trace_static_flags == {"precond_jmax": 1}
    assert stack_calls == [(trace_full,)]
    assert tape.diagnostics["timing"]["iterations"] == 2


def test_concat_residual_iteration_traces_empty_is_typed():
    trace = da.concat_residual_iteration_traces([])

    assert trace.iter2.dtype.kind in {"i", "u"}
    assert trace.step_status.dtype == object
    assert trace.state_advanced.dtype == bool
    assert trace.iter2.shape == (0,)


def test_static_flags_reject_varying_static_arrays_and_track_jmax():
    trace0 = _fake_dynamic_trace()
    trace1 = _fake_dynamic_trace()
    trace1["precond_jmax"] = 2

    flags = da._static_flags_from_replay_step_traces((trace0, trace1))

    assert flags["precond_jmax"] is None

    bad = _fake_dynamic_trace()
    bad["signgs"] = np.asarray([-1])
    with pytest.raises(ValueError, match="signgs"):
        da._static_flags_from_replay_step_traces((trace0, bad))


def test_dynamic_replay_payload_promotes_varying_constants(monkeypatch):
    monkeypatch.setenv("VMEC_JAX_DYNAMIC_REPLAY_BUCKET", "2")
    traces = (
        _fake_dynamic_trace(),
        _fake_dynamic_trace(),
    )
    traces[0]["w_mode_mn"] = np.asarray([1.0])
    traces[1]["w_mode_mn"] = np.asarray([2.0])
    traces[1]["wout_like"] = "wout-varying"
    static_flags = da._static_flags_from_replay_step_traces(traces)

    stacked, dynamic_flags, _initial_carry, _base_carries = da._build_dynamic_replay_payload(
        traces,
        static_flags,
    )

    assert "wout_like" in stacked
    assert "wout_like" not in dynamic_flags
    assert "w_mode_mn" in stacked
    assert "w_mode_mn" not in dynamic_flags
    assert dynamic_flags["trig"] == "trig"
    np.testing.assert_allclose(np.asarray(stacked["w_mode_mn"])[:, 0], [1.0, 2.0])


def test_dynamic_replay_support_and_restart_classifiers():
    supported = _fake_supported_dynamic_trace()
    restart = _fake_restart_trace()

    assert da._dynamic_replay_trace_supported(supported)
    assert da._dynamic_restart_trace_supported(restart)
    assert not da._dynamic_replay_trace_supported(restart)
    assert not da._dynamic_restart_trace_supported({**restart, "restart_path": "ordinary"})

    tape = SimpleNamespace(step_traces=(supported,), step_trace_static_flags={"precond_jmax": 1})
    assert da._dynamic_replay_supported(tape=tape, rebuild_preconditioner=True)
    assert not da._dynamic_replay_supported(tape=tape, rebuild_preconditioner=False)
    assert not da._dynamic_replay_supported(
        tape=SimpleNamespace(step_traces=(), step_trace_static_flags=None),
        rebuild_preconditioner=True,
    )
    assert not da._dynamic_replay_supported(
        tape=SimpleNamespace(step_traces=(supported,), step_trace_static_flags={"precond_jmax": None}),
        rebuild_preconditioner=True,
    )


def test_dynamic_initial_carry_zero_fills_missing_asymmetric_velocities():
    trace = _fake_dynamic_trace()
    for key in ("vRsc_before", "vRcs_before", "vZcc_before", "vZss_before", "vLcc_before", "vLss_before"):
        del trace[key]

    carry = da._dynamic_replay_initial_carry(trace)

    assert len(carry) == 15
    for idx in (5, 6, 9, 10, 13, 14):
        np.testing.assert_allclose(np.asarray(carry[idx]), 0.0)
        assert np.asarray(carry[idx]).shape == np.asarray(carry[3]).shape


def test_dynamic_safe_dt_from_force_arrays_limits_only_finite_nonzero_forces():
    z = np.zeros((2, 2), dtype=float)
    ones = np.ones((2, 2), dtype=float)

    nominal = da._dynamic_safe_dt_from_force_arrays(
        dt_nominal=2.0,
        max_coeff_delta_rms=0.25,
        frcc=z,
        frss=z,
        fzsc=z,
        fzcs=z,
        flsc=z,
        flcs=z,
        frsc=z,
        frcs=z,
        fzcc=z,
        fzss=z,
        flcc=z,
        flss=z,
    )
    assert float(np.asarray(nominal)) == pytest.approx(2.0)

    limited = da._dynamic_safe_dt_from_force_arrays(
        dt_nominal=2.0,
        max_coeff_delta_rms=0.25,
        frcc=ones,
        frss=ones,
        fzsc=ones,
        fzcs=ones,
        flsc=ones,
        flcs=ones,
        frsc=ones,
        frcs=ones,
        fzcc=ones,
        fzss=ones,
        flcc=ones,
        flss=ones,
    )
    assert 1.0e-12 < float(np.asarray(limited)) < 2.0

    nonfinite = da._dynamic_safe_dt_from_force_arrays(
        dt_nominal=2.0,
        max_coeff_delta_rms=0.25,
        frcc=np.full((2, 2), np.nan),
        frss=z,
        fzsc=z,
        fzcs=z,
        flsc=z,
        flcs=z,
        frsc=z,
        frcs=z,
        fzcc=z,
        fzss=z,
        flcc=z,
        flss=z,
    )
    assert float(np.asarray(nonfinite)) == pytest.approx(2.0)


def test_jvp_columns_chunks_env_before_dynamic_basepoint_runner(monkeypatch):
    monkeypatch.delenv("VMEC_JAX_DYNAMIC_REPLAY_MODE", raising=False)
    monkeypatch.setenv("VMEC_JAX_REPLAY_COLUMN_CHUNK", "2")
    calls = []

    def fake_runner(*, static, stacked, stacked_base_carries, static_flags):
        def run(carry_tangents0, stacked_base_carries_in, stacked_traces_in):
            calls.append(np.asarray(carry_tangents0[0]).shape)
            assert stacked_base_carries_in is stacked_base_carries
            assert stacked_traces_in is stacked
            return carry_tangents0

        return run

    monkeypatch.setattr(da, "_checkpoint_tape_dynamic_basepoint_scan_runner", fake_runner)
    tape = SimpleNamespace(
        step_traces=(),
        dynamic_initial_carry=(np.zeros(3, dtype=float),),
        dynamic_base_carries_stacked=_fake_carry_stacked(width=3),
        stacked_step_traces={"active": np.asarray([True])},
        step_trace_static_flags={"precond_jmax": 1},
    )
    tangents = np.arange(15, dtype=float).reshape(5, 3)

    out = da.checkpoint_tape_state_jvp_columns(
        tape=tape,
        static="static",
        initial_tangents=tangents,
        rebuild_preconditioner=True,
    )

    np.testing.assert_allclose(np.asarray(out), tangents)
    assert calls == [(2, 3), (2, 3), (1, 3)]


def test_jvp_columns_auto_chunk_falls_back_when_env_override_invalid(monkeypatch):
    monkeypatch.delenv("VMEC_JAX_DYNAMIC_REPLAY_MODE", raising=False)
    monkeypatch.setenv("VMEC_JAX_REPLAY_COLUMN_CHUNK", "not-an-int")
    monkeypatch.setenv("VMEC_JAX_REPLAY_COLUMN_TARGET_MB", "0.000001")
    calls = []

    def fake_runner(*, static, stacked, stacked_base_carries, static_flags):
        def run(carry_tangents0, stacked_base_carries_in, stacked_traces_in):
            calls.append(np.asarray(carry_tangents0[0]).shape)
            assert stacked_base_carries_in is stacked_base_carries
            assert stacked_traces_in is stacked
            return carry_tangents0

        return run

    monkeypatch.setattr(da, "_checkpoint_tape_dynamic_basepoint_scan_runner", fake_runner)
    tape = SimpleNamespace(
        step_traces=(),
        dynamic_initial_carry=(np.zeros(128, dtype=np.uint8),),
        dynamic_base_carries_stacked=_fake_carry_stacked(width=128),
        stacked_step_traces={"active": np.asarray([True])},
        step_trace_static_flags={"precond_jmax": 1},
    )
    tangents = np.arange(384, dtype=float).reshape(3, 128)

    out = da.checkpoint_tape_state_jvp_columns(
        tape=tape,
        static="static",
        initial_tangents=tangents,
        rebuild_preconditioner=True,
    )

    np.testing.assert_allclose(np.asarray(out), tangents)
    assert calls == [(1, 128), (1, 128), (1, 128)]


def test_jvp_columns_batches_supported_segments_across_restart(monkeypatch):
    monkeypatch.setenv("VMEC_JAX_DYNAMIC_REPLAY_BUCKET", "1")
    calls = []

    def fake_runner(*, static, stacked, stacked_base_carries, static_flags):
        def run(carry_tangents, _stacked_base_carries, _stacked_traces):
            calls.append(tuple(np.asarray(item).copy() for item in carry_tangents))
            if len(calls) == 1:
                return (
                    carry_tangents[0] + 1.0,
                    carry_tangents[1] + 5.0,
                    carry_tangents[2] + 7.0,
                    *(item + 9.0 for item in carry_tangents[3:]),
                )
            return (carry_tangents[0] + 3.0, *carry_tangents[1:])

        return run

    monkeypatch.setattr(da, "_checkpoint_tape_dynamic_basepoint_scan_runner", fake_runner)
    tape = SimpleNamespace(
        step_traces=(
            _fake_supported_dynamic_trace(),
            _fake_restart_trace(),
            _fake_supported_dynamic_trace(lambda_update_scale=0.75),
        ),
        dynamic_initial_carry=None,
        dynamic_base_carries_stacked=None,
        stacked_step_traces=None,
        step_trace_static_flags=None,
    )
    tangents = np.asarray([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])

    out = da.checkpoint_tape_state_jvp_columns(
        tape=tape,
        static="static",
        initial_tangents=tangents,
        rebuild_preconditioner=True,
    )

    np.testing.assert_allclose(np.asarray(out), tangents + 4.0)
    assert len(calls) == 2
    np.testing.assert_allclose(calls[1][0], tangents + 1.0)
    np.testing.assert_allclose(calls[1][1], 0.0)
    np.testing.assert_allclose(calls[1][2], 7.0)
    for velocity_tangent in calls[1][3:]:
        np.testing.assert_allclose(velocity_tangent, 0.0)


def test_state_vjp_basepoint_runner_receives_zero_aux_cotangents(monkeypatch):
    monkeypatch.setenv("VMEC_JAX_DYNAMIC_REPLAY_MODE", "basepoint")
    captured = []

    def fake_runner(*, static, stacked, stacked_base_carries, static_flags):
        def run(final_cotangents, stacked_base_carries_in, stacked_traces_in):
            captured.append(tuple(np.asarray(item).copy() for item in final_cotangents))
            assert stacked_base_carries_in is stacked_base_carries
            assert stacked_traces_in is stacked
            return (final_cotangents[0] + 2.0, *final_cotangents[1:])

        return run

    monkeypatch.setattr(da, "_checkpoint_tape_dynamic_basepoint_vjp_scan_runner", fake_runner)
    carry0 = (np.zeros(3, dtype=float),) + tuple(np.ones((2,), dtype=float) * idx for idx in range(1, 15))
    tape = SimpleNamespace(
        step_traces=(),
        dynamic_initial_carry=carry0,
        dynamic_base_carries_stacked=_fake_carry_stacked(width=3),
        stacked_step_traces={"active": np.asarray([True])},
        step_trace_static_flags={"precond_jmax": 1},
    )

    out = da.checkpoint_tape_state_vjp(
        tape=tape,
        static="static",
        final_cotangent=np.asarray([1.0, 2.0, 3.0]),
        rebuild_preconditioner=True,
    )

    np.testing.assert_allclose(np.asarray(out), [3.0, 4.0, 5.0])
    assert len(captured) == 1
    np.testing.assert_allclose(captured[0][0], [1.0, 2.0, 3.0])
    for aux_cotangent in captured[0][1:]:
        np.testing.assert_allclose(aux_cotangent, 0.0)


def test_state_vjp_whole_scan_uses_linearized_dynamic_runner(monkeypatch):
    monkeypatch.setenv("VMEC_JAX_DYNAMIC_REPLAY_MODE", "whole_scan")

    def fake_runner(*, static, stacked, static_flags):
        def run(carry_init, _stacked):
            return (carry_init[0] * 3.0, *carry_init[1:])

        return run

    monkeypatch.setattr(da, "_checkpoint_tape_dynamic_scan_runner", fake_runner)
    carry0 = (np.zeros(3, dtype=float),) + tuple(np.ones((2,), dtype=float) * idx for idx in range(1, 15))
    tape = SimpleNamespace(
        step_traces=(),
        dynamic_initial_carry=carry0,
        dynamic_base_carries_stacked=None,
        stacked_step_traces={"active": np.asarray([True])},
        step_trace_static_flags={"precond_jmax": 1},
    )

    out = da.checkpoint_tape_state_vjp(
        tape=tape,
        static="static",
        final_cotangent=np.asarray([1.0, 2.0, 3.0]),
        rebuild_preconditioner=True,
    )

    np.testing.assert_allclose(np.asarray(out), [3.0, 6.0, 9.0])


def test_empty_tape_jvp_and_vjp_are_identity():
    tape = SimpleNamespace(
        step_traces=(),
        dynamic_initial_carry=None,
        dynamic_base_carries_stacked=None,
        stacked_step_traces=None,
        step_trace_static_flags=None,
    )

    tangent = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    cotangent = np.asarray([5.0, 6.0])

    np.testing.assert_allclose(
        np.asarray(
            da.checkpoint_tape_state_jvp_columns(
                tape=tape,
                static="static",
                initial_tangents=tangent,
            )
        ),
        tangent,
    )
    np.testing.assert_allclose(
        np.asarray(da.checkpoint_tape_state_vjp(tape=tape, static="static", final_cotangent=cotangent)),
        cotangent,
    )


def test_packed_dynamic_replay_step_requires_layout_before_force_rebuild():
    carry = tuple(np.zeros(1, dtype=float) for _ in range(15))

    with pytest.raises(ValueError, match="stored VMEC layout"):
        da._packed_dynamic_replay_step_from_carry(
            carry,
            {},
            static="static",
            static_flags={},
            preconditioner_jmax_override=1,
        )
