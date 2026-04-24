from types import SimpleNamespace

import numpy as np

import vmec_jax.discrete_adjoint as da


def _fake_tape(nbytes: int):
    # A uint8 buffer makes the size accounting in _replay_column_chunk_default
    # easy to reason about without allocating large arrays in the test.
    return SimpleNamespace(
        dynamic_initial_carry=(np.zeros(nbytes, dtype=np.uint8),),
        dynamic_base_carries_stacked=None,
    )


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
