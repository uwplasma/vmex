from __future__ import annotations

import numpy as np

import vmec_jax._solve_runtime as solve_runtime
import vmec_jax.solve as solve_module


def test_scan_chunk_settings_cpu_default(monkeypatch):
    monkeypatch.setattr(solve_runtime, "_scan_backend_name", lambda: "cpu")
    monkeypatch.delenv("VMEC_JAX_SCAN_CHUNK_SIZE", raising=False)

    chunk_size, cap_to_remaining = solve_runtime._scan_chunk_settings(
        max_iter_scan=783,
        nstep_screen=200,
        need_print=False,
        lthreed=False,
    )

    assert chunk_size == 783
    assert cap_to_remaining is True


def test_scan_chunk_settings_accelerator_default(monkeypatch):
    monkeypatch.setattr(solve_runtime, "_scan_backend_name", lambda: "gpu")
    monkeypatch.delenv("VMEC_JAX_SCAN_CHUNK_SIZE", raising=False)

    chunk_size, cap_to_remaining = solve_runtime._scan_chunk_settings(
        max_iter_scan=783,
        nstep_screen=200,
        need_print=False,
        lthreed=False,
    )

    assert chunk_size == 783
    assert cap_to_remaining is True


def test_scan_chunk_settings_accelerator_3d_default(monkeypatch):
    monkeypatch.setattr(solve_runtime, "_scan_backend_name", lambda: "gpu")
    monkeypatch.delenv("VMEC_JAX_SCAN_CHUNK_SIZE", raising=False)

    chunk_size, cap_to_remaining = solve_runtime._scan_chunk_settings(
        max_iter_scan=5000,
        nstep_screen=200,
        need_print=False,
        lthreed=True,
    )

    # GPU quiet runs now use the full iteration budget as a single chunk to
    # eliminate host/device sync overhead in the Python chunk loop.
    # Use VMEC_JAX_SCAN_CHUNK_SIZE to cap this when GPU memory is tight.
    assert chunk_size == 5000
    assert cap_to_remaining is True


def test_scan_chunk_settings_respects_override(monkeypatch):
    monkeypatch.setattr(solve_runtime, "_scan_backend_name", lambda: "gpu")
    monkeypatch.setenv("VMEC_JAX_SCAN_CHUNK_SIZE", "1000")

    chunk_size, cap_to_remaining = solve_runtime._scan_chunk_settings(
        max_iter_scan=3,
        nstep_screen=200,
        need_print=False,
        lthreed=True,
    )

    assert chunk_size == 1000
    assert cap_to_remaining is True


def test_default_scan_core_uses_minimal_accelerated_path():
    assert solve_runtime._default_scan_core(
        scan_core_env="",
        scan_minimal=True,
        fsq_total_target=3.0e-13,
    )


def test_default_scan_core_respects_override():
    assert not solve_runtime._default_scan_core(
        scan_core_env="0",
        scan_minimal=True,
        fsq_total_target=3.0e-13,
    )
    assert solve_runtime._default_scan_core(
        scan_core_env="1",
        scan_minimal=False,
        fsq_total_target=None,
    )


def test_solve_reexports_runtime_helpers():
    assert solve_module._hash_array_bytes is solve_runtime._hash_array_bytes
    assert solve_module._tree_has_tracer is solve_runtime._tree_has_tracer
    assert solve_module._scan_backend_name is solve_runtime._scan_backend_name
    assert solve_module._default_scan_core is solve_runtime._default_scan_core
    assert solve_module._parse_iter_list is solve_runtime._parse_iter_list


def test_solve_scan_chunk_settings_keeps_backend_monkeypatch_compat(monkeypatch):
    monkeypatch.setattr(solve_module, "_scan_backend_name", lambda: "gpu")
    monkeypatch.delenv("VMEC_JAX_SCAN_CHUNK_SIZE", raising=False)

    chunk_size, cap_to_remaining = solve_module._scan_chunk_settings(
        max_iter_scan=19,
        nstep_screen=5,
        need_print=True,
        lthreed=False,
    )

    assert chunk_size == 5
    assert cap_to_remaining is False


def test_hash_array_bytes_includes_shape_and_dtype():
    a = np.asarray([1, 2], dtype=np.int64)
    b = np.asarray([[1, 2]], dtype=np.int64)
    c = np.asarray([1, 2], dtype=np.float64)

    assert solve_runtime._hash_array_bytes(a) != solve_runtime._hash_array_bytes(b)
    assert solve_runtime._hash_array_bytes(a) != solve_runtime._hash_array_bytes(c)
    assert solve_runtime._hash_array_bytes(a) == solve_runtime._hash_array_bytes(a.copy())


def test_runtime_iter_list_parsing_matches_legacy_edges():
    assert solve_runtime._parse_iter_list("") is None
    assert solve_runtime._parse_iter_list(" 1, 3-5, 9 , bad, 8-6 ") == {1, 3, 4, 5, 6, 7, 8, 9}
    assert solve_runtime._parse_iter_list("bad, nope") is None


def test_runtime_dump_env_policy_preserves_legacy_truthiness():
    assert not solve_runtime._dump_env_enabled("")
    assert not solve_runtime._dump_env_enabled("0")
    assert solve_runtime._dump_env_enabled("1")
    assert solve_runtime._dump_env_enabled("false")
    assert solve_runtime._dump_env_enabled(" 0 ")


def test_runtime_dump_iter_selected_uses_optional_allowlist():
    assert solve_runtime._dump_iter_selected(iter_idx=3, iter_env="")
    assert solve_runtime._dump_iter_selected(iter_idx=3, iter_env="1,3-4")
    assert not solve_runtime._dump_iter_selected(iter_idx=5, iter_env="1,3-4")
