from types import SimpleNamespace

import numpy as np

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
