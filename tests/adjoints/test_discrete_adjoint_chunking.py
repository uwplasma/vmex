from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pytest

import vmec_jax.discrete_adjoint as da
from vmec_jax.state import StateLayout, VMECState, pack_state, unpack_state


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


def _state_with_scalar(state: VMECState, value: float) -> VMECState:
    arr = np.full_like(np.asarray(state.Rcos, dtype=float), float(value))
    return VMECState(
        layout=state.layout,
        Rcos=arr.copy(),
        Rsin=arr.copy(),
        Zcos=arr.copy(),
        Zsin=arr.copy(),
        Lcos=arr.copy(),
        Lsin=arr.copy(),
    )


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


def _fake_jax_replay_trace(
    *,
    lambda_update_scale: float = 1.0,
    precond_jmax: int = 1,
    dynamic_supported: bool = True,
) -> dict:
    trace = _fake_supported_dynamic_trace(
        lambda_update_scale=lambda_update_scale,
        precond_jmax=int(precond_jmax),
    )
    trace.update(
        {
            "wout_like": np.asarray(0.0),
            "trig": np.asarray(0.0),
            "precond_mats": np.asarray(1.0),
            "lam_prec": np.asarray(1.0),
        }
    )
    if not dynamic_supported:
        trace.update(
            {
                "branch": "synthetic_fallback",
                "step_status": "synthetic_fallback",
                "restart_reason": "none",
                "restart_path": "none",
            }
        )
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


def _fake_carry(width: int = 3):
    return (np.zeros(width, dtype=float),) + tuple(np.zeros(1, dtype=float) for _ in range(14))


def _dynamic_scan_static_flags(*, precond_jmax: int = 1) -> dict:
    return {
        "apply_lforbal": False,
        "include_edge_residual": True,
        "apply_m1_constraints": True,
        "limit_update_rms": False,
        "limit_dt_from_force": False,
        "vmec2000_control": True,
        "divide_by_scalxc_for_update": False,
        "signgs": 1,
        "precond_jmax": int(precond_jmax),
    }


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


def test_replay_column_chunk_default_uses_initial_carry_for_jvp_only(monkeypatch):
    monkeypatch.setenv("VMEC_JAX_REPLAY_COLUMN_TARGET_MB", "0.001")
    tape = SimpleNamespace(
        dynamic_initial_carry=(np.zeros(128, dtype=np.uint8),),
        dynamic_base_carries_stacked=(np.zeros(4096, dtype=np.uint8),),
        jvp_only=True,
    )
    tangents = np.zeros((10, 1), dtype=float)

    chunk = da._replay_column_chunk_default(tape=tape, tangents=tangents)

    assert chunk == 8


def test_replay_column_target_default_is_relaxed_after_memory_fix():
    assert da._DEFAULT_REPLAY_COLUMN_TARGET_MB == 4096.0


def test_dynamic_replay_bucket_default_is_modest(monkeypatch):
    monkeypatch.delenv("VMEC_JAX_DYNAMIC_REPLAY_BUCKET", raising=False)

    assert da._dynamic_replay_bucket_size() == 32
    assert da._dynamic_replay_bucket_len(33) == 64


def test_dynamic_replay_bucket_backend_failure_and_empty_lengths(monkeypatch):
    monkeypatch.delenv("VMEC_JAX_DYNAMIC_REPLAY_BUCKET", raising=False)
    monkeypatch.setattr(da.jax, "default_backend", lambda: (_ for _ in ()).throw(RuntimeError("backend")))

    assert da._dynamic_replay_bucket_size() == 32
    assert da._dynamic_replay_bucket_len(0) == 0
    assert da._dynamic_replay_bucket_len(-3) == 0


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


def test_dynamic_replay_payload_keeps_constraint_active_flags_dynamic(monkeypatch):
    monkeypatch.setenv("VMEC_JAX_DYNAMIC_REPLAY_BUCKET", "2")
    diag = (np.asarray([1.0]), np.asarray([2.0]))
    traces = (
        _fake_supported_dynamic_trace(
            constraint_tcon0=2.0,
            constraint_precond_diag=diag,
            constraint_tcon=np.asarray([0.0]),
            constraint_precond_active=np.asarray(False),
            constraint_tcon_active=np.asarray(False),
        ),
        _fake_supported_dynamic_trace(
            constraint_tcon0=2.0,
            constraint_precond_diag=diag,
            constraint_tcon=np.asarray([3.0]),
            constraint_precond_active=np.asarray(True),
            constraint_tcon_active=np.asarray(True),
        ),
    )
    static_flags = da._static_flags_from_replay_step_traces(traces)

    stacked, dynamic_flags, _initial_carry, _base_carries = da._build_dynamic_replay_payload(
        traces,
        static_flags,
    )

    for key in ("constraint_precond_active", "constraint_tcon_active"):
        assert key in stacked
        assert key not in dynamic_flags
        np.testing.assert_array_equal(np.asarray(stacked[key]), np.asarray([False, True]))


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


def test_dynamic_replay_payload_can_store_invariant_preconditioner_cache_in_trace(monkeypatch):
    monkeypatch.setenv("VMEC_JAX_DYNAMIC_REPLAY_BUCKET", "2")
    monkeypatch.setenv("VMEC_JAX_OPT_STATIC_PRECOND_REPLAY_CARRY", "1")
    traces = (
        _fake_supported_dynamic_trace(
            precond_cache_update=False,
            constraint_cache_update=False,
            lam_prec=np.asarray([2.0]),
            precond_mats=np.asarray([3.0]),
        ),
        _fake_supported_dynamic_trace(
            precond_cache_update=False,
            constraint_cache_update=True,
            lam_prec=np.asarray([4.0]),
            precond_mats=np.asarray([5.0]),
        ),
    )
    flags = da._static_flags_from_replay_step_traces(traces)

    stacked, dynamic_flags, initial_carry, base_carries = da._build_dynamic_replay_payload(
        traces,
        flags,
    )

    assert dynamic_flags["preconditioner_cache_in_trace"] is True
    assert len(initial_carry) == 18
    assert len(base_carries) == 18
    assert "cache_lam_prec_static" in stacked
    assert "cache_prec_mats_static" in stacked
    np.testing.assert_allclose(np.asarray(stacked["cache_lam_prec_static"]), [[2.0], [4.0]])
    np.testing.assert_allclose(np.asarray(stacked["cache_prec_mats_static"]), [[3.0], [5.0]])


def test_dynamic_replay_payload_keeps_preconditioner_cache_in_carry_when_any_step_updates(monkeypatch):
    monkeypatch.setenv("VMEC_JAX_DYNAMIC_REPLAY_BUCKET", "2")
    monkeypatch.setenv("VMEC_JAX_OPT_STATIC_PRECOND_REPLAY_CARRY", "1")
    traces = (
        _fake_supported_dynamic_trace(
            precond_cache_update=False,
            lam_prec=np.asarray([2.0]),
            precond_mats=np.asarray([3.0]),
        ),
        _fake_supported_dynamic_trace(
            precond_cache_update=True,
            lam_prec=np.asarray([4.0]),
            precond_mats=np.asarray([5.0]),
        ),
    )
    flags = da._static_flags_from_replay_step_traces(traces)

    stacked, dynamic_flags, initial_carry, base_carries = da._build_dynamic_replay_payload(
        traces,
        flags,
    )

    assert dynamic_flags["preconditioner_cache_in_trace"] is False
    assert len(initial_carry) == 20
    assert len(base_carries) == 20
    assert "cache_lam_prec_static" not in stacked
    assert "cache_prec_mats_static" not in stacked


def test_dynamic_replay_payload_stacks_nested_preconditioner_cache_tree(monkeypatch):
    monkeypatch.setenv("VMEC_JAX_DYNAMIC_REPLAY_BUCKET", "2")
    monkeypatch.setenv("VMEC_JAX_OPT_STATIC_PRECOND_REPLAY_CARRY", "1")
    traces = (
        _fake_supported_dynamic_trace(
            precond_cache_update=False,
            lam_prec=np.asarray([2.0]),
            precond_mats={"ar": np.asarray([3.0]), "nested": {"az": np.asarray([4.0])}},
        ),
        _fake_supported_dynamic_trace(
            precond_cache_update=False,
            lam_prec=np.asarray([4.0]),
            precond_mats={"ar": np.asarray([5.0]), "nested": {"az": np.asarray([6.0])}},
        ),
    )
    flags = da._static_flags_from_replay_step_traces(traces)

    stacked, dynamic_flags, initial_carry, _base_carries = da._build_dynamic_replay_payload(
        traces,
        flags,
    )

    assert dynamic_flags["preconditioner_cache_in_trace"] is True
    assert len(initial_carry) == 18
    assert "cache_prec_mats_static" in stacked
    np.testing.assert_allclose(np.asarray(stacked["cache_prec_mats_static"]["ar"]), [[3.0], [5.0]])
    np.testing.assert_allclose(np.asarray(stacked["cache_prec_mats_static"]["nested"]["az"]), [[4.0], [6.0]])


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


def test_checkpoint_scan_runner_factories_reuse_cached_runners(monkeypatch):
    monkeypatch.setenv("VMEC_JAX_SCAN_CACHE_LIMIT", "8")
    monkeypatch.setenv("VMEC_JAX_TIMING", "1")
    da.clear_replay_scan_caches()
    assert all(value == 0 for value in da.replay_scan_cache_diagnostics(reset=False).values())
    static = object()

    trace = _fake_jax_replay_trace()
    stacked, generic_flags = da._stack_replay_step_traces((trace,))
    generic = da._checkpoint_tape_scan_runner(
        static=static,
        stacked=stacked,
        static_flags=generic_flags,
        rebuild_preconditioner=False,
    )
    generic_again = da._checkpoint_tape_scan_runner(
        static=static,
        stacked=stacked,
        static_flags=generic_flags,
        rebuild_preconditioner=False,
    )

    dynamic_flags = _dynamic_scan_static_flags(precond_jmax=2)
    dynamic_stacked = {
        "active": np.asarray([True, False]),
        "delta": np.asarray([1.0, 2.0]),
    }
    dynamic = da._checkpoint_tape_dynamic_scan_runner(
        static=static,
        stacked=dynamic_stacked,
        static_flags=dynamic_flags,
    )
    dynamic_again = da._checkpoint_tape_dynamic_scan_runner(
        static=static,
        stacked=dynamic_stacked,
        static_flags=dynamic_flags,
    )

    stacked_base_carries = (np.zeros((2, 1), dtype=float),)
    basepoint = da._checkpoint_tape_dynamic_basepoint_scan_runner(
        static=static,
        stacked=dynamic_stacked,
        stacked_base_carries=stacked_base_carries,
        static_flags=dynamic_flags,
    )
    basepoint_again = da._checkpoint_tape_dynamic_basepoint_scan_runner(
        static=static,
        stacked=dynamic_stacked,
        stacked_base_carries=stacked_base_carries,
        static_flags=dynamic_flags,
    )
    basepoint_vjp = da._checkpoint_tape_dynamic_basepoint_vjp_scan_runner(
        static=static,
        stacked=dynamic_stacked,
        stacked_base_carries=stacked_base_carries,
        static_flags=dynamic_flags,
    )
    basepoint_vjp_again = da._checkpoint_tape_dynamic_basepoint_vjp_scan_runner(
        static=static,
        stacked=dynamic_stacked,
        stacked_base_carries=stacked_base_carries,
        static_flags=dynamic_flags,
    )

    assert generic is generic_again
    assert dynamic is dynamic_again
    assert basepoint is basepoint_again
    assert basepoint_vjp is basepoint_vjp_again
    assert len(da._CHECKPOINT_TAPE_SCAN_CACHE) == 1
    assert len(da._CHECKPOINT_TAPE_DYNAMIC_SCAN_CACHE) == 1
    assert len(da._CHECKPOINT_TAPE_DYNAMIC_BASEPOINT_SCAN_CACHE) == 1
    assert len(da._CHECKPOINT_TAPE_DYNAMIC_BASEPOINT_VJP_SCAN_CACHE) == 1
    diagnostics = da.replay_scan_cache_diagnostics(reset=True)
    for label in ("checkpoint", "dynamic", "dynamic_basepoint", "dynamic_basepoint_vjp"):
        assert diagnostics[f"replay_{label}_scan_cache_miss_count"] == 1
        assert diagnostics[f"replay_{label}_scan_cache_hit_count"] == 1
        assert diagnostics[f"replay_{label}_scan_cache_lookup_s"] >= 0.0
        assert diagnostics[f"replay_{label}_scan_cache_build_s"] >= 0.0
    assert all(value == 0 for value in da.replay_scan_cache_diagnostics(reset=False).values())
    da.clear_replay_scan_caches()


def test_checkpoint_scan_runner_cache_keys_include_tridi_policies(monkeypatch):
    monkeypatch.setenv("VMEC_JAX_SCAN_CACHE_LIMIT", "8")
    da.clear_replay_scan_caches()
    static = object()
    trace = _fake_jax_replay_trace()
    stacked, static_flags = da._stack_replay_step_traces((trace,))

    base = da._checkpoint_tape_scan_runner(
        static=static,
        stacked=stacked,
        static_flags=static_flags,
        rebuild_preconditioner=False,
    )
    precomputed = da._checkpoint_tape_scan_runner(
        static=static,
        stacked=stacked,
        static_flags={**static_flags, "preconditioner_use_precomputed_tridi": True},
        rebuild_preconditioner=False,
    )
    lax_tridi = da._checkpoint_tape_scan_runner(
        static=static,
        stacked=stacked,
        static_flags={**static_flags, "preconditioner_use_lax_tridi": True},
        rebuild_preconditioner=False,
    )

    assert base is not precomputed
    assert base is not lax_tridi
    assert precomputed is not lax_tridi
    assert len(da._CHECKPOINT_TAPE_SCAN_CACHE) == 3
    da.clear_replay_scan_caches()


def test_checkpoint_tape_dynamic_scan_runner_skips_inactive_trace_entries(monkeypatch):
    from vmec_jax._compat import jnp

    da.clear_replay_scan_caches()
    static = object()

    def fake_dynamic_step(
        carry,
        trace,
        *,
        static,
        static_flags,
        preconditioner_jmax_override,
        preconditioner_use_precomputed_tridi=None,
        preconditioner_use_lax_tridi=None,
    ):
        assert static is not None
        assert static_flags["precond_jmax"] == 7
        assert preconditioner_jmax_override == 7
        assert preconditioner_use_precomputed_tridi is None
        assert preconditioner_use_lax_tridi is None
        delta = jnp.asarray(trace["delta"], dtype=jnp.asarray(carry[0]).dtype)
        return tuple(jnp.asarray(part) + delta for part in carry)

    monkeypatch.setattr(da, "_packed_dynamic_replay_step_from_carry", fake_dynamic_step)
    stacked = {
        "active": np.asarray([True, False, True]),
        "delta": np.asarray([2.0, 100.0, 3.0]),
    }
    static_flags = _dynamic_scan_static_flags(precond_jmax=7)
    run_scan = da._checkpoint_tape_dynamic_scan_runner(
        static=static,
        stacked=stacked,
        static_flags=static_flags,
    )
    assert run_scan is da._checkpoint_tape_dynamic_scan_runner(
        static=static,
        stacked=stacked,
        static_flags=static_flags,
    )

    carry_final = run_scan((np.asarray([1.0, 2.0]), np.asarray([10.0])), stacked)

    np.testing.assert_allclose(np.asarray(carry_final[0]), [6.0, 7.0])
    np.testing.assert_allclose(np.asarray(carry_final[1]), [15.0])
    da.clear_replay_scan_caches()


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
        lambda step_traces, static_flags, *, store_base_carries=True: (
            "dynamic",
            {"dynamic": True},
            "carry0",
            "base_carries",
        ),
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


def test_direct_checkpoint_tape_jvp_only_omits_dynamic_base_carries_by_default(monkeypatch):
    monkeypatch.delenv("VMEC_JAX_JVP_ONLY_EXACT_TAPE_BASEPOINT_CARRIES", raising=False)
    trace = _fake_dynamic_trace()
    state = trace["state_pre"]
    store_base_carry_requests = []

    def fake_solve(state0, static, *, adjoint_trace, **kwargs):
        assert state0 is state
        assert static == "static"
        assert adjoint_trace is True
        return SimpleNamespace(
            state=state,
            diagnostics={
                "adjoint_step_trace": [trace],
                "timing": {},
            },
        )

    def fake_dynamic_payload(step_traces, static_flags, *, store_base_carries=True):
        store_base_carry_requests.append(bool(store_base_carries))
        return "dynamic", {"dynamic": True}, "carry0", None

    monkeypatch.setattr("vmec_jax.solve.solve_fixed_boundary_residual_iter", fake_solve)
    monkeypatch.setattr(da, "_dynamic_replay_supported", lambda *, tape, rebuild_preconditioner: True)
    monkeypatch.setattr(da, "_build_dynamic_replay_payload", fake_dynamic_payload)

    tape = da.build_residual_checkpoint_tape_direct(
        state,
        "static",
        indata={},
        signgs=1,
        max_iter=3,
        ftol=0.0,
        store_full_step_traces=False,
        jvp_only=True,
    )

    assert store_base_carry_requests == [False]
    assert tape.step_traces == ()
    assert tape.stacked_step_traces == "dynamic"
    assert tape.dynamic_initial_carry == "carry0"
    assert tape.dynamic_base_carries_stacked is None
    assert tape.jvp_only is True
    assert tape.diagnostics["jvp_only_basepoint_carries_enabled"] is False
    assert tape.diagnostics["jvp_only_fast_basepoint_scan_available"] is False
    assert tape.diagnostics["jvp_only_replay_path"] == "dynamic_whole_scan_linearize"
    assert tape.diagnostics["jvp_only_replay_fallback_reason"] == "basepoint_carries_not_stored"
    with pytest.raises(ValueError, match="JVP-only checkpoint tapes"):
        da.checkpoint_tape_state_vjp(tape=tape, static="static", final_cotangent=np.zeros(1))


def test_direct_checkpoint_tape_jvp_only_can_preserve_basepoint_carries_for_fast_scan(monkeypatch):
    monkeypatch.setenv("VMEC_JAX_JVP_ONLY_EXACT_TAPE_BASEPOINT_CARRIES", "1")
    trace = _fake_dynamic_trace()
    state = trace["state_pre"]
    store_base_carry_requests = []

    def fake_solve(state0, static, *, adjoint_trace, **kwargs):
        assert state0 is state
        assert static == "static"
        assert adjoint_trace is True
        return SimpleNamespace(
            state=state,
            diagnostics={
                "adjoint_step_trace": [trace],
                "timing": {},
            },
        )

    def fake_dynamic_payload(step_traces, static_flags, *, store_base_carries=True):
        store_base_carry_requests.append(bool(store_base_carries))
        base_carries = _fake_carry_stacked(width=6) if store_base_carries else None
        return {"active": np.asarray([True])}, {"precond_jmax": 1}, _fake_carry(width=6), base_carries

    runner_calls = []

    def fake_runner(*, static, stacked, stacked_base_carries, static_flags):
        def run(carry_tangents0, stacked_base_carries_in, stacked_traces_in):
            runner_calls.append(np.asarray(carry_tangents0[0]).shape)
            assert stacked_base_carries_in is stacked_base_carries
            assert stacked_traces_in is stacked
            return carry_tangents0

        return run

    monkeypatch.setattr("vmec_jax.solve.solve_fixed_boundary_residual_iter", fake_solve)
    monkeypatch.setattr(da, "_dynamic_replay_supported", lambda *, tape, rebuild_preconditioner: True)
    monkeypatch.setattr(da, "_build_dynamic_replay_payload", fake_dynamic_payload)
    monkeypatch.setattr(da, "_checkpoint_tape_dynamic_basepoint_scan_runner", fake_runner)

    tape = da.build_residual_checkpoint_tape_direct(
        state,
        "static",
        indata={},
        signgs=1,
        max_iter=3,
        ftol=0.0,
        store_full_step_traces=False,
        jvp_only=True,
    )

    assert store_base_carry_requests == [True]
    assert tape.step_traces == ()
    assert tape.dynamic_base_carries_stacked is not None
    assert tape.jvp_only is True
    assert tape.diagnostics["jvp_only_basepoint_carries_enabled"] is True
    assert tape.diagnostics["jvp_only_fast_basepoint_scan_available"] is True
    assert tape.diagnostics["jvp_only_replay_path"] == "dynamic_basepoint_scan"

    tangents = np.arange(12, dtype=float).reshape(2, 6)
    out = da.checkpoint_tape_state_jvp_columns(
        tape=tape,
        static="static",
        initial_tangents=tangents,
        rebuild_preconditioner=True,
    )

    np.testing.assert_allclose(np.asarray(out), tangents)
    assert runner_calls == [(2, 6)]
    with pytest.raises(ValueError, match="forward-replay only"):
        da.checkpoint_tape_state_vjp(tape=tape, static="static", final_cotangent=np.zeros(6))


def test_direct_checkpoint_tape_jvp_only_without_steps_reports_identity(monkeypatch):
    monkeypatch.delenv("VMEC_JAX_JVP_ONLY_EXACT_TAPE_BASEPOINT_CARRIES", raising=False)
    state = _fake_dynamic_trace()["state_pre"]

    def fake_solve(state0, static, *, adjoint_trace, **kwargs):
        assert state0 is state
        assert static == "static"
        assert adjoint_trace is True
        return SimpleNamespace(
            state=state,
            diagnostics={
                "adjoint_step_trace": [],
                "timing": {"iterations": 0},
                "converged": True,
            },
        )

    monkeypatch.setattr("vmec_jax.solve.solve_fixed_boundary_residual_iter", fake_solve)

    tape = da.build_residual_checkpoint_tape_direct(
        state,
        "static",
        indata={},
        signgs=1,
        max_iter=0,
        store_full_step_traces=False,
        jvp_only=True,
    )

    tangent = np.arange(6.0)
    columns = np.arange(12.0).reshape(2, 6)
    assert tape.step_traces == ()
    assert tape.dynamic_initial_carry is None
    assert tape.dynamic_base_carries_stacked is None
    assert tape.jvp_only is True
    assert tape.diagnostics["jvp_only_basepoint_carries_enabled"] is False
    assert tape.diagnostics["jvp_only_fast_basepoint_scan_available"] is False
    assert tape.diagnostics["jvp_only_replay_path"] == "identity"
    np.testing.assert_allclose(
        np.asarray(da.checkpoint_tape_state_jvp(tape=tape, static="static", initial_tangent=tangent)),
        tangent,
    )
    np.testing.assert_allclose(
        np.asarray(da.checkpoint_tape_state_jvp_columns(tape=tape, static="static", initial_tangents=columns)),
        columns,
    )
    with pytest.raises(ValueError, match="forward-replay only"):
        da.checkpoint_tape_state_vjp(tape=tape, static="static", final_cotangent=tangent)


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


def test_checkpoint_tape_state_jvp_columns_runs_generic_scan_runner(monkeypatch):
    from vmec_jax._compat import jnp

    da.clear_replay_scan_caches()
    static = object()
    trace = _fake_jax_replay_trace(lambda_update_scale=2.0)
    stacked, static_flags = da._stack_replay_step_traces((trace,))
    calls = []

    def fake_strict_update(state, static, **kwargs):
        calls.append(
            {
                "static": static,
                "mats_is_none": kwargs["mats"] is None,
                "jmax": kwargs["jmax"],
                "lam_prec_is_none": kwargs["lam_prec"] is None,
                "w_mode_mn_is_none": kwargs["w_mode_mn"] is None,
                "override": kwargs["preconditioner_jmax_override"],
            }
        )
        x = pack_state(state)
        scale = jnp.asarray(kwargs["lambda_update_scale"], dtype=jnp.asarray(x).dtype)
        return {"step": {"state_post": unpack_state(scale * x, state.layout)}}

    monkeypatch.setattr(da, "strict_update_one_step_from_state", fake_strict_update)
    tape = da.ResidualCheckpointTape(
        final_packed_state=np.zeros(6),
        packed_states=np.zeros((0, 6)),
        trace=da.concat_residual_iteration_traces([]),
        resume_states=(),
        step_traces=(trace,),
        stacked_step_traces=stacked,
        step_trace_static_flags=static_flags,
    )
    tangents = np.arange(12.0).reshape(2, 6)

    run_scan = da._checkpoint_tape_scan_runner(
        static=static,
        stacked=stacked,
        static_flags=static_flags,
        rebuild_preconditioner=False,
    )
    assert run_scan is da._checkpoint_tape_scan_runner(
        static=static,
        stacked=stacked,
        static_flags=static_flags,
        rebuild_preconditioner=False,
    )
    out = da.checkpoint_tape_state_jvp_columns(
        tape=tape,
        static=static,
        initial_tangents=tangents,
        rebuild_preconditioner=False,
    )

    np.testing.assert_allclose(np.asarray(out), 2.0 * tangents)
    assert calls == [
        {
            "static": static,
            "mats_is_none": False,
            "jmax": 1,
            "lam_prec_is_none": False,
            "w_mode_mn_is_none": False,
            "override": None,
        }
    ]
    da.clear_replay_scan_caches()


def test_checkpoint_tape_state_jvp_columns_rebuild_preconditioner_with_variable_jmax(monkeypatch):
    from vmec_jax._compat import jnp

    trace0 = _fake_jax_replay_trace(lambda_update_scale=2.0, precond_jmax=1, dynamic_supported=False)
    trace1 = _fake_jax_replay_trace(lambda_update_scale=3.0, precond_jmax=4, dynamic_supported=False)
    calls = []

    def fake_strict_update(state, _static, **kwargs):
        calls.append(
            {
                "mats_is_none": kwargs["mats"] is None,
                "jmax": kwargs["jmax"],
                "lam_prec_is_none": kwargs["lam_prec"] is None,
                "w_mode_mn_is_none": kwargs["w_mode_mn"] is None,
                "override": kwargs["preconditioner_jmax_override"],
            }
        )
        x = pack_state(state)
        scale = jnp.asarray(kwargs["lambda_update_scale"], dtype=jnp.asarray(x).dtype)
        return {"step": {"state_post": unpack_state(scale * x, state.layout)}}

    monkeypatch.setattr(da, "strict_update_one_step_from_state", fake_strict_update)
    tape = da.ResidualCheckpointTape(
        final_packed_state=np.zeros(6),
        packed_states=np.zeros((0, 6)),
        trace=da.concat_residual_iteration_traces([]),
        resume_states=(),
        step_traces=(trace0, trace1),
    )
    tangents = np.ones((2, 6), dtype=float)

    out = da.checkpoint_tape_state_jvp_columns(
        tape=tape,
        static=object(),
        initial_tangents=tangents,
        rebuild_preconditioner=True,
    )

    np.testing.assert_allclose(np.asarray(out), 6.0 * tangents)
    assert calls == [
        {
            "mats_is_none": True,
            "jmax": None,
            "lam_prec_is_none": True,
            "w_mode_mn_is_none": True,
            "override": 1,
        },
        {
            "mats_is_none": True,
            "jmax": None,
            "lam_prec_is_none": True,
            "w_mode_mn_is_none": True,
            "override": 4,
        },
    ]


def test_checkpoint_tape_builder_collects_step_payloads_until_converged(monkeypatch):
    trace0 = _fake_supported_dynamic_trace(lambda_update_scale=0.5)
    state0 = trace0["state_pre"]
    state1 = _state_with_scalar(state0, 1.0)
    state2 = _state_with_scalar(state0, 2.0)
    calls = []

    def fake_replay_step(state, static, *, resume_state, solve_kwargs):
        calls.append((state, static, resume_state, dict(solve_kwargs)))
        idx = len(calls)
        converged = idx == 2
        next_state = state1 if idx == 1 else state2
        return SimpleNamespace(
            state=next_state,
            diagnostics={
                "iter2_history": [idx],
                "step_status_history": ["momentum"],
                "time_step_history": [0.25 * idx],
                "fsq_curr_history": [1.0 / idx],
                "state_advanced_history": [True],
                "resume_state": {"step": idx, "fsq": 1.0 / idx},
                "adjoint_step_trace": [{**trace0, "lambda_update_scale": 0.25 * idx}],
                "converged": converged,
            },
        )

    monkeypatch.setattr(da, "replay_residual_checkpoint_step", fake_replay_step)

    tape = da.build_residual_checkpoint_tape(
        state0,
        "static",
        indata={"nstep": 1},
        signgs=-1,
        max_iter=5,
        ftol=1.0e-6,
        step_size=0.75,
        light_history=True,
    )

    assert len(calls) == 2
    assert calls[0][2] is None
    assert calls[1][2] == {"step": 1, "fsq": 1.0}
    assert calls[0][3]["signgs"] == -1
    assert calls[0][3]["light_history"] is False
    assert calls[0][3]["resume_state_mode"] == "full"
    np.testing.assert_allclose(np.asarray(tape.final_packed_state), np.asarray(da.pack_state(state2)))
    assert tape.packed_states.shape == (2, int(state0.layout.size))
    assert [r["step"] for r in tape.resume_states] == [1, 2]
    assert len(tape.step_traces) == 2
    np.testing.assert_allclose(tape.trace.time_step, [0.25, 0.5])


def test_dynamic_fsq1_uses_vmec2000_lambda_norm_and_safe_zero_rz_norm(monkeypatch):
    import vmec_jax.kernels.residue as residue

    def fake_gcx2_from_tomnsps(**_kwargs):
        return (np.asarray(2.0), np.asarray(3.0), np.asarray(5.0))

    rz_norm_values = iter((np.asarray(4.0), np.asarray(0.0), np.asarray(4.0)))

    def fake_rz_norm_from_state(**_kwargs):
        return next(rz_norm_values)

    monkeypatch.setattr(residue, "vmec_gcx2_from_tomnsps", fake_gcx2_from_tomnsps)
    monkeypatch.setattr(residue, "vmec_rz_norm_from_state", fake_rz_norm_from_state)

    trace = _fake_dynamic_trace()
    state = trace["state_pre"]
    static = SimpleNamespace(s=np.asarray([0.0, 0.5, 1.0]), cfg=SimpleNamespace(lconm1=True))
    frzl = SimpleNamespace(
        flsc=np.asarray([[9.0], [1.0], [2.0]]),
        flcs=np.asarray([[9.0], [3.0], [4.0]]),
        flcc=np.asarray([[9.0], [5.0], [6.0]]),
        flss=np.asarray([[9.0], [7.0], [8.0]]),
    )

    vmec2000 = da._dynamic_fsq1_from_force_channels(
        state_pre=state,
        static=static,
        vmec2000_control=True,
        frzl_pre=frzl,
    )
    zero_norm = da._dynamic_fsq1_from_force_channels(
        state_pre=state,
        static=static,
        vmec2000_control=False,
        frzl_pre=frzl,
    )
    non_vmec2000 = da._dynamic_fsq1_from_force_channels(
        state_pre=state,
        static=static,
        vmec2000_control=False,
        frzl_pre=frzl,
    )

    expected_lambda_full = (1 + 4 + 9 + 16 + 25 + 36 + 49 + 64) * 0.5
    assert float(np.asarray(vmec2000)) == pytest.approx((2.0 + 3.0) / 4.0 + expected_lambda_full)
    assert float(np.asarray(zero_norm)) == pytest.approx(5.0 * 0.5)
    assert float(np.asarray(non_vmec2000)) == pytest.approx((2.0 + 3.0) / 4.0 + 5.0 * 0.5)


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


@pytest.mark.parametrize("backend", ["gpu", "cuda", "rocm", "cuda:0"])
def test_dynamic_replay_payload_stacks_on_device_for_accelerator_backend(monkeypatch, backend):
    monkeypatch.setenv("VMEC_JAX_DYNAMIC_REPLAY_BUCKET", "1")
    monkeypatch.setattr(da.jax, "default_backend", lambda: backend)
    traces = (_fake_dynamic_trace(lambda_update_scale=0.5),)
    static_flags = da._static_flags_from_replay_step_traces(traces)

    stacked, _dynamic_flags, _initial_carry, _base_carries = da._build_dynamic_replay_payload(
        traces,
        static_flags,
    )

    assert hasattr(stacked["lambda_update_scale"], "dtype")
    np.testing.assert_allclose(np.asarray(stacked["lambda_update_scale"]), [0.5])
    assert da._dynamic_replay_bucket_default() == 128


def test_generic_replay_stack_uses_device_arrays_for_accelerator_backend(monkeypatch):
    monkeypatch.setattr(da.jax, "default_backend", lambda: "gpu")
    trace = _fake_jax_replay_trace()

    stacked, _static_flags = da._stack_replay_step_traces((trace,))

    assert hasattr(stacked["lambda_update_scale"], "dtype")
    assert "jax" in type(stacked["lambda_update_scale"]).__module__.lower()
    np.testing.assert_allclose(np.asarray(stacked["lambda_update_scale"]), [1.0])


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


def test_replay_values_equal_handles_objects_and_fallback_equality():
    assert da._replay_values_equal(SimpleNamespace(a=np.asarray([1.0])), SimpleNamespace(a=np.asarray([1.0])))
    assert not da._replay_values_equal(SimpleNamespace(a=1, b=2), SimpleNamespace(a=1))

    class Unarrayable:
        def __array__(self, *_args, **_kwargs):
            raise TypeError("no array view")

        def __eq__(self, other):
            return isinstance(other, Unarrayable)

    assert da._replay_values_equal(Unarrayable(), Unarrayable())


def test_segmented_dynamic_replay_uses_static_flags_without_full_stack(monkeypatch):
    trace = _fake_supported_dynamic_trace()
    tape = SimpleNamespace(
        dynamic_initial_carry=None,
        dynamic_base_carries_stacked=None,
        stacked_step_traces=None,
        step_trace_static_flags=None,
        step_traces=(trace,),
    )
    calls = []

    def fail_stack(_segment):
        raise AssertionError("segmented dynamic replay should not build a generic stacked trace")

    def fake_dynamic_payload(segment, static_flags, **_kwargs):
        calls.append(static_flags)
        stacked = {"active": np.asarray([True])}
        base_carries = _fake_carry_stacked(width=2)
        return stacked, dict(static_flags), _fake_carry(width=2), base_carries

    def fake_runner(**_kwargs):
        return lambda carry_tangents, _base_carries, _stacked: carry_tangents

    monkeypatch.setattr(da, "_stack_replay_step_traces", fail_stack)
    monkeypatch.setattr(da, "_build_dynamic_replay_payload", fake_dynamic_payload)
    monkeypatch.setattr(da, "_checkpoint_tape_dynamic_basepoint_scan_runner", fake_runner)

    tangents = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    out = da.checkpoint_tape_state_jvp_columns(
        tape=tape,
        static=object(),
        initial_tangents=tangents,
        rebuild_preconditioner=True,
    )

    np.testing.assert_allclose(np.asarray(out), tangents)
    assert calls
    assert calls[0]["precond_jmax"] == 1


def test_stacked_trace_signature_uses_asarray_dtype_for_proxy_leaf():
    class ArrayProxy:
        shape = (2,)

        def __array__(self, dtype=None):
            return np.asarray([1, 2], dtype=dtype)

    assert da._stacked_trace_signature((ArrayProxy(),)) == (((2,), np.dtype(int).str),)


def test_dynamic_initial_carry_compacts_missing_symmetric_asymmetric_velocities():
    trace = _fake_dynamic_trace()
    for key in ("vRsc_before", "vRcs_before", "vZcc_before", "vZss_before", "vLcc_before", "vLss_before"):
        del trace[key]

    carry = da._dynamic_replay_initial_carry(trace)

    assert len(carry) == 20
    for idx in (5, 6, 9, 10, 13, 14):
        assert carry[idx] is None
    for idx in range(15, 20):
        np.testing.assert_allclose(np.asarray(carry[idx]), 0.0)


def test_dynamic_initial_carry_zero_fills_missing_lasym_asymmetric_velocities():
    trace = _fake_dynamic_trace()
    state = trace["state_pre"]
    trace["state_pre"] = VMECState(
        layout=StateLayout(ns=state.layout.ns, K=state.layout.K, lasym=True),
        Rcos=state.Rcos,
        Rsin=state.Rsin,
        Zcos=state.Zcos,
        Zsin=state.Zsin,
        Lcos=state.Lcos,
        Lsin=state.Lsin,
    )
    for key in ("vRsc_before", "vRcs_before", "vZcc_before", "vZss_before", "vLcc_before", "vLss_before"):
        del trace[key]

    carry = da._dynamic_replay_initial_carry(trace)

    assert len(carry) == 20
    for idx in (5, 6, 9, 10, 13, 14):
        np.testing.assert_allclose(np.asarray(carry[idx]), 0.0)
        assert np.asarray(carry[idx]).shape == np.asarray(carry[3]).shape


def test_cotangent_like_output_preserves_compacted_none_slots():
    output = (
        np.ones((2,), dtype=float),
        None,
        {"active": np.ones((1,), dtype=float), "inactive": None},
    )
    cotangent = (
        np.asarray([3.0, 4.0]),
        None,
        {"active": np.asarray([5.0]), "inactive": None},
    )

    padded = da._cotangent_like_output(cotangent, output)

    np.testing.assert_allclose(np.asarray(padded[0]), [3.0, 4.0])
    assert padded[1] is None
    np.testing.assert_allclose(np.asarray(padded[2]["active"]), [5.0])
    assert padded[2]["inactive"] is None

    missing = da._cotangent_like_output((), output)
    np.testing.assert_allclose(np.asarray(missing[0]), [0.0, 0.0])
    assert missing[1] is None
    np.testing.assert_allclose(np.asarray(missing[2]["active"]), [0.0])
    assert missing[2]["inactive"] is None


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
    monkeypatch.setenv("VMEC_JAX_TIMING", "1")
    da.clear_replay_scan_caches()
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
    diagnostics = da.replay_scan_cache_diagnostics(reset=True)
    assert diagnostics["replay_jvp_columns_chunked_call_count"] == 1
    assert diagnostics["replay_jvp_columns_chunk_count"] == 3
    assert diagnostics["replay_jvp_columns_last_chunk_size"] == 2
    assert diagnostics["replay_jvp_columns_dynamic_basepoint_count"] == 3
    assert diagnostics["replay_jvp_columns_leaf_call_count"] == 3
    assert diagnostics["replay_jvp_columns_input_column_count"] == 5


def test_jvp_columns_dynamic_basepoint_uses_runner_zero_aux(monkeypatch):
    monkeypatch.delenv("VMEC_JAX_DYNAMIC_REPLAY_MODE", raising=False)
    calls = []

    class FakeRunner:
        def __call__(self, *_args, **_kwargs):
            raise AssertionError("dynamic basepoint path should use runner zero_aux entrypoint")

        def zero_aux(self, state_tangents, stacked_base_carries_in, stacked_traces_in):
            calls.append(np.asarray(state_tangents).shape)
            assert stacked_base_carries_in is stacked_base_carries
            assert stacked_traces_in is stacked
            return (state_tangents + 2.0,)

    def fake_runner(*, static, stacked, stacked_base_carries, static_flags):
        return FakeRunner()

    monkeypatch.setattr(da, "_checkpoint_tape_dynamic_basepoint_scan_runner", fake_runner)
    stacked_base_carries = _fake_carry_stacked(width=3)
    stacked = {"active": np.asarray([True])}
    tape = SimpleNamespace(
        step_traces=(),
        dynamic_initial_carry=(np.zeros(3, dtype=float),),
        dynamic_base_carries_stacked=stacked_base_carries,
        stacked_step_traces=stacked,
        step_trace_static_flags={"precond_jmax": 1},
    )
    tangents = np.arange(6, dtype=float).reshape(2, 3)

    out = da.checkpoint_tape_state_jvp_columns(
        tape=tape,
        static="static",
        initial_tangents=tangents,
        rebuild_preconditioner=True,
    )

    np.testing.assert_allclose(np.asarray(out), tangents + 2.0)
    assert calls == [(2, 3)]


def test_jvp_columns_dynamic_basepoint_prefers_initial_carry_runner(monkeypatch):
    monkeypatch.delenv("VMEC_JAX_DYNAMIC_REPLAY_MODE", raising=False)
    calls = []

    class FakeRunner:
        def __call__(self, *_args, **_kwargs):
            raise AssertionError("dynamic basepoint path should not use full-stack callable")

        def zero_aux(self, *_args, **_kwargs):
            raise AssertionError("dynamic basepoint path should prefer zero_aux_initial")

        def zero_aux_initial(self, state_tangents, carry0_in, stacked_traces_in):
            calls.append((np.asarray(state_tangents).shape, np.asarray(carry0_in[0]).shape))
            assert stacked_traces_in is stacked
            return (state_tangents + 4.0,)

    def fake_runner(*, static, stacked, stacked_base_carries, static_flags):
        return FakeRunner()

    monkeypatch.setattr(da, "_checkpoint_tape_dynamic_basepoint_scan_runner", fake_runner)
    stacked_base_carries = _fake_carry_stacked(width=3)
    stacked = {"active": np.asarray([True])}
    tape = SimpleNamespace(
        step_traces=(),
        dynamic_initial_carry=(np.zeros(3, dtype=float),),
        dynamic_base_carries_stacked=stacked_base_carries,
        stacked_step_traces=stacked,
        step_trace_static_flags={"precond_jmax": 1},
    )
    tangents = np.arange(6, dtype=float).reshape(2, 3)

    out = da.checkpoint_tape_state_jvp_columns(
        tape=tape,
        static="static",
        initial_tangents=tangents,
        rebuild_preconditioner=True,
    )

    np.testing.assert_allclose(np.asarray(out), tangents + 4.0)
    assert calls == [((2, 3), (3,))]


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
        def _copy(item):
            return None if item is None else np.asarray(item).copy()

        def _advance_aux(item):
            return None if item is None else item + 9.0

        def run(carry_tangents, _stacked_base_carries, _stacked_traces):
            calls.append(tuple(_copy(item) for item in carry_tangents))
            if len(calls) == 1:
                return (
                    carry_tangents[0] + 1.0,
                    carry_tangents[1] + 5.0,
                    carry_tangents[2] + 7.0,
                    *(_advance_aux(item) for item in carry_tangents[3:]),
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
    inactive_asym_slots = {5, 6, 9, 10, 13, 14}
    for idx, velocity_tangent in enumerate(calls[1][3:15], start=3):
        if idx in inactive_asym_slots:
            assert velocity_tangent is None
        else:
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


def test_jvp_columns_whole_scan_uses_linearized_dynamic_runner(monkeypatch):
    monkeypatch.setenv("VMEC_JAX_DYNAMIC_REPLAY_MODE", "whole_scan")

    def fake_runner(*, static, stacked, static_flags):
        def run(carry_init, _stacked):
            return (carry_init[0] * 4.0, *carry_init[1:])

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

    tangents = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = da.checkpoint_tape_state_jvp_columns(
        tape=tape,
        static="static",
        initial_tangents=tangents,
        rebuild_preconditioner=True,
    )

    np.testing.assert_allclose(np.asarray(out), tangents * 4.0)


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


def test_replay_column_chunk_default_ignores_uninspectable_trees(monkeypatch):
    monkeypatch.setenv("VMEC_JAX_REPLAY_COLUMN_TARGET_MB", "1")
    monkeypatch.setattr(
        da.jax.tree_util,
        "tree_leaves",
        lambda _tree: (_ for _ in ()).throw(TypeError("opaque tree")),
    )

    chunk = da._replay_column_chunk_default(
        tape=SimpleNamespace(dynamic_initial_carry=object(), dynamic_base_carries_stacked=object()),
        tangents=np.zeros((4, 2)),
    )

    assert chunk is None


def test_compact_tape_diagnostics_skips_scalar_conversion_errors():
    class BadFloat(float):
        def __float__(self):
            raise TypeError("bad scalar")

    out = da._compact_tape_diagnostics({"final_fsq": BadFloat(1.0), "converged_iter": np.int64(2)})

    assert out == {"converged_iter": 2}


def test_dynamic_replay_trace_summary_counts_cache_update_flags():
    traces = (
        {"constraint_cache_update": True},
        {"constraint_cache_update": False, "precond_cache_update": True},
        {"constraint_cache_update": False, "precond_cache_update": False},
    )

    out = da._dynamic_replay_trace_summary(traces, supported=True)

    assert out["step_count"] == 3
    assert out["supported"] is True
    assert out["constraint_cache_update_true_count"] == 1
    assert out["constraint_cache_update_false_count"] == 2
    assert out["precond_cache_update_true_count"] == 2
    assert out["precond_cache_update_false_count"] == 1
    assert out["all_precond_cache_updates"] is False
    assert out["any_precond_cache_updates"] is True


def test_fallback_state_jvp_and_vjp_apply_linearized_step(monkeypatch):
    trace = _fake_dynamic_trace()

    def fake_step(state, *_args, **_kwargs):
        packed = da.pack_state(state)
        return {"step": {"state_post": da.unpack_state(2.0 * packed + 1.0, state.layout)}}

    monkeypatch.setattr(da, "strict_update_one_step_from_state", fake_step)
    tape = SimpleNamespace(
        step_traces=(trace,),
        dynamic_initial_carry=None,
        dynamic_base_carries_stacked=None,
        stacked_step_traces=None,
        step_trace_static_flags=None,
    )
    tangent = np.arange(trace["state_pre"].layout.size, dtype=float)
    cotangent = np.arange(trace["state_pre"].layout.size, dtype=float) + 1.0

    jvp_out = da.checkpoint_tape_state_jvp(
        tape=tape,
        static="static",
        initial_tangent=tangent,
        rebuild_preconditioner=False,
    )
    vjp_out = da.checkpoint_tape_state_vjp(
        tape=tape,
        static="static",
        final_cotangent=cotangent,
        rebuild_preconditioner=False,
    )

    np.testing.assert_allclose(np.asarray(jvp_out), 2.0 * tangent)
    np.testing.assert_allclose(np.asarray(vjp_out), 2.0 * cotangent)


def test_checkpoint_tape_param_jvp_and_vjp_bridge_boundary_params(monkeypatch):
    pytest.importorskip("jax")

    from vmec_jax._compat import jnp

    layout = StateLayout(ns=1, K=1, lasym=False)

    def fake_apply_boundary_params(boundary, specs, params):
        return SimpleNamespace(boundary=boundary, specs=specs, params=params)

    def fake_initial_guess_from_boundary(_static, boundary_p, *_args, **_kwargs):
        params = jnp.asarray(boundary_p.params)
        z = jnp.zeros((1, 1), dtype=params.dtype)
        return VMECState(
            layout=layout,
            Rcos=params[0:1, None],
            Rsin=z,
            Zcos=z,
            Zsin=params[1:2, None],
            Lcos=z,
            Lsin=z,
        )

    monkeypatch.setattr("vmec_jax.optimization.apply_boundary_params", fake_apply_boundary_params)
    monkeypatch.setattr("vmec_jax.init_guess.initial_guess_from_boundary", fake_initial_guess_from_boundary)
    tape = SimpleNamespace(
        step_traces=(),
        dynamic_initial_carry=None,
        dynamic_base_carries_stacked=None,
        stacked_step_traces=None,
        step_trace_static_flags=None,
    )

    params = jnp.asarray([2.0, 3.0])
    tangent = jnp.asarray([0.25, -0.5])
    final_cotangent = jnp.asarray([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])

    jvp_out = da.checkpoint_tape_param_jvp(
        tape=tape,
        static="static",
        boundary="boundary",
        indata="indata",
        specs="specs",
        params=params,
        axis_override=None,
        params_tangent=tangent,
    )
    vjp_out = da.checkpoint_tape_param_vjp(
        tape=tape,
        static="static",
        boundary="boundary",
        indata="indata",
        specs="specs",
        params=params,
        axis_override=None,
        final_cotangent=final_cotangent,
    )

    np.testing.assert_allclose(np.asarray(jvp_out), [0.25, 0.0, 0.0, -0.5, 0.0, 0.0])
    np.testing.assert_allclose(np.asarray(vjp_out), [10.0, 40.0])
