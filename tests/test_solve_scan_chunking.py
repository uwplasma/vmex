from __future__ import annotations

from collections import OrderedDict

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


def test_scan_chunk_settings_accelerator_low_mode_long_budget(monkeypatch):
    monkeypatch.setattr(solve_runtime, "_scan_backend_name", lambda: "gpu")
    monkeypatch.delenv("VMEC_JAX_SCAN_CHUNK_SIZE", raising=False)

    chunk_size, cap_to_remaining = solve_runtime._scan_chunk_settings(
        max_iter_scan=1500,
        nstep_screen=200,
        need_print=False,
        lthreed=True,
        spectral_mode_count=8,
    )

    assert chunk_size == 256
    assert cap_to_remaining is False


def test_scan_chunk_settings_accelerator_3d_default(monkeypatch):
    monkeypatch.setattr(solve_runtime, "_scan_backend_name", lambda: "gpu")
    monkeypatch.delenv("VMEC_JAX_SCAN_CHUNK_SIZE", raising=False)

    chunk_size, cap_to_remaining = solve_runtime._scan_chunk_settings(
        max_iter_scan=5000,
        nstep_screen=200,
        need_print=False,
        lthreed=True,
    )

    # Higher-mode GPU quiet runs use the full iteration budget as a single
    # chunk to avoid Python host/device sync overhead.
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
    assert solve_module._runtime_env_enabled is solve_runtime._runtime_env_enabled
    assert solve_module._edge_signature_key is solve_runtime._edge_signature_key
    assert solve_module._edge_value_key is solve_runtime._edge_value_key
    assert solve_module._scan_fallback_policy is solve_runtime._scan_fallback_policy
    assert solve_module._residual_convergence_flags is solve_runtime._residual_convergence_flags
    assert solve_module._scalar_history_array is solve_runtime._scalar_history_array


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


def test_edge_signature_key_ignores_values_but_not_shape_or_dtype():
    a = np.asarray([1.0, 2.0], dtype=np.float64)
    b = np.asarray([3.0, 4.0], dtype=np.float64)
    c = np.asarray([[1.0, 2.0]], dtype=np.float64)
    d = np.asarray([1.0, 2.0], dtype=np.float32)

    assert solve_runtime._edge_signature_key(a) == solve_runtime._edge_signature_key(b)
    assert solve_runtime._edge_signature_key(a) != solve_runtime._edge_signature_key(c)
    assert solve_runtime._edge_signature_key(a) != solve_runtime._edge_signature_key(d)
    assert solve_runtime._edge_value_key(a) != solve_runtime._edge_value_key(b)


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


def test_runtime_env_enabled_uses_modern_false_tokens():
    assert not solve_runtime._runtime_env_enabled("")
    assert not solve_runtime._runtime_env_enabled("0")
    assert not solve_runtime._runtime_env_enabled(" false ")
    assert not solve_runtime._runtime_env_enabled("NO")
    assert solve_runtime._runtime_env_enabled("1")
    assert solve_runtime._runtime_env_enabled("yes")


def test_scan_fallback_policy_backend_defaults():
    cpu = solve_runtime._scan_fallback_policy(
        backend_name="cpu",
        enabled_env=None,
        iters_env="50",
        badjac_limit_env="10",
        fsq_abs_env="1.0e-2",
        accept_frac_env="0.5",
        fsq_factor_env="50",
        improve_env="0.1",
    )
    gpu = solve_runtime._scan_fallback_policy(
        backend_name="gpu",
        enabled_env=None,
        iters_env="50",
        badjac_limit_env="10",
        fsq_abs_env="1.0e-2",
        accept_frac_env="0.5",
        fsq_factor_env="50",
        improve_env="0.1",
    )

    assert cpu.enabled is True
    assert gpu.enabled is False
    assert cpu.iters == 50
    assert cpu.badjac_limit == 10
    assert cpu.fsq_abs == 1.0e-2
    assert cpu.accept_frac == 0.5
    assert cpu.fsq_factor == 50.0
    assert cpu.improve == 0.1


def test_scan_fallback_policy_preserves_legacy_parse_fallbacks_and_clamps():
    policy = solve_runtime._scan_fallback_policy(
        backend_name="gpu",
        enabled_env="yes",
        iters_env="bad",
        badjac_limit_env="-3",
        fsq_abs_env="-1",
        accept_frac_env="2",
        fsq_factor_env="0.25",
        improve_env="bad",
    )

    assert policy.enabled is True
    assert policy.iters == 20
    assert policy.badjac_limit == 0
    assert policy.fsq_abs == 0.0
    assert policy.accept_frac == 1.0
    assert policy.fsq_factor == 1.0
    assert policy.improve == 0.9


def test_residual_convergence_flags_support_strict_and_total_target():
    assert solve_runtime._residual_convergence_flags(
        fsqr=1.0,
        fsqz=2.0,
        fsql=3.0,
        ftol=3.0,
        fsq_total_target=None,
    ) == (True, False, True)
    assert solve_runtime._residual_convergence_flags(
        fsqr=4.0,
        fsqz=1.0,
        fsql=1.0,
        ftol=3.0,
        fsq_total_target=6.0,
    ) == (False, True, True)
    assert solve_runtime._residual_convergence_flags(
        fsqr=4.0,
        fsqz=1.0,
        fsql=1.0,
        ftol=3.0,
        fsq_total_target=5.0,
    ) == (False, False, False)


def test_scalar_history_array_materializes_float_array():
    empty = solve_runtime._scalar_history_array([])
    assert empty.shape == (0,)
    assert empty.dtype == np.dtype(float)

    got = solve_runtime._scalar_history_array([np.asarray(1), np.asarray(2.5)])
    np.testing.assert_allclose(got, np.asarray([1.0, 2.5], dtype=float))


def test_jit_cache_put_bounds_and_marks_recent(monkeypatch):
    from vmec_jax import solve

    cache = OrderedDict()
    monkeypatch.setenv("VMEC_JAX_TEST_CACHE_SIZE", "2")

    solve._jit_cache_put(cache, ("a",), "A", env_name="VMEC_JAX_TEST_CACHE_SIZE", default=4)
    solve._jit_cache_put(cache, ("b",), "B", env_name="VMEC_JAX_TEST_CACHE_SIZE", default=4)
    assert solve._jit_cache_get(cache, ("a",)) == "A"
    solve._jit_cache_put(cache, ("c",), "C", env_name="VMEC_JAX_TEST_CACHE_SIZE", default=4)

    assert list(cache.keys()) == [("a",), ("c",)]
    assert solve._jit_cache_get(cache, ("b",)) is None


def test_jit_cache_limit_zero_disables_retention(monkeypatch):
    from vmec_jax import solve

    cache = OrderedDict()
    monkeypatch.setenv("VMEC_JAX_TEST_CACHE_SIZE", "0")

    value = solve._jit_cache_put(cache, ("a",), "A", env_name="VMEC_JAX_TEST_CACHE_SIZE", default=4)

    assert value == "A"
    assert cache == OrderedDict()
