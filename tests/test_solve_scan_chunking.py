from __future__ import annotations

import vmec_jax.solve as solve_module


def test_scan_chunk_settings_cpu_default(monkeypatch):
    monkeypatch.setattr(solve_module, "_scan_backend_name", lambda: "cpu")
    monkeypatch.delenv("VMEC_JAX_SCAN_CHUNK_SIZE", raising=False)

    chunk_size, cap_to_remaining = solve_module._scan_chunk_settings(
        max_iter_scan=783,
        nstep_screen=200,
        need_print=False,
        lthreed=False,
    )

    assert chunk_size == 200
    assert cap_to_remaining is False


def test_scan_chunk_settings_accelerator_default(monkeypatch):
    monkeypatch.setattr(solve_module, "_scan_backend_name", lambda: "gpu")
    monkeypatch.delenv("VMEC_JAX_SCAN_CHUNK_SIZE", raising=False)

    chunk_size, cap_to_remaining = solve_module._scan_chunk_settings(
        max_iter_scan=783,
        nstep_screen=200,
        need_print=False,
        lthreed=False,
    )

    assert chunk_size == 783
    assert cap_to_remaining is True


def test_scan_chunk_settings_accelerator_3d_default(monkeypatch):
    monkeypatch.setattr(solve_module, "_scan_backend_name", lambda: "gpu")
    monkeypatch.delenv("VMEC_JAX_SCAN_CHUNK_SIZE", raising=False)

    chunk_size, cap_to_remaining = solve_module._scan_chunk_settings(
        max_iter_scan=5000,
        nstep_screen=200,
        need_print=False,
        lthreed=True,
    )

    assert chunk_size == 400
    assert cap_to_remaining is True


def test_scan_chunk_settings_respects_override(monkeypatch):
    monkeypatch.setattr(solve_module, "_scan_backend_name", lambda: "gpu")
    monkeypatch.setenv("VMEC_JAX_SCAN_CHUNK_SIZE", "1000")

    chunk_size, cap_to_remaining = solve_module._scan_chunk_settings(
        max_iter_scan=3,
        nstep_screen=200,
        need_print=False,
        lthreed=True,
    )

    assert chunk_size == 1000
    assert cap_to_remaining is True
