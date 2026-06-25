from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import vmec_jax._solve_runtime as runtime
from vmec_jax.solvers.fixed_boundary.residual import runtime as residual_runtime


def test_scan_chunk_settings_invalid_env_uses_screen_floor(monkeypatch):
    monkeypatch.setenv("VMEC_JAX_SCAN_CHUNK_SIZE", "not-an-int")

    chunk_size, cap_to_remaining = runtime._scan_chunk_settings(
        max_iter_scan=20,
        nstep_screen=0,
        need_print=False,
        lthreed=True,
        backend_name="gpu",
    )

    assert chunk_size == 1
    assert cap_to_remaining is True


def test_scan_chunk_settings_printing_uses_screen_budget(monkeypatch):
    monkeypatch.delenv("VMEC_JAX_SCAN_CHUNK_SIZE", raising=False)

    chunk_size, cap_to_remaining = runtime._scan_chunk_settings(
        max_iter_scan=100,
        nstep_screen=7,
        need_print=True,
        lthreed=False,
        backend_name="cpu",
    )

    assert chunk_size == 7
    assert cap_to_remaining is False


def test_scan_chunk_settings_explicit_env_wins_and_cpu_quiet_uses_full_budget(monkeypatch):
    monkeypatch.setenv("VMEC_JAX_SCAN_CHUNK_SIZE", "3")

    chunk_size, cap_to_remaining = runtime._scan_chunk_settings(
        max_iter_scan=100,
        nstep_screen=7,
        need_print=True,
        lthreed=False,
        backend_name="cpu",
    )

    assert chunk_size == 3
    assert cap_to_remaining is False

    monkeypatch.delenv("VMEC_JAX_SCAN_CHUNK_SIZE")
    chunk_size, cap_to_remaining = runtime._scan_chunk_settings(
        max_iter_scan=100,
        nstep_screen=7,
        need_print=False,
        lthreed=False,
        backend_name="cpu",
    )

    assert chunk_size == 100
    assert cap_to_remaining is True


def test_scan_chunk_settings_accelerator_quiet_uses_full_budget(monkeypatch):
    monkeypatch.delenv("VMEC_JAX_SCAN_CHUNK_SIZE", raising=False)

    chunk_size, cap_to_remaining = runtime._scan_chunk_settings(
        max_iter_scan=12,
        nstep_screen=4,
        need_print=False,
        lthreed=True,
        backend_name="tpu",
    )

    assert chunk_size == 12
    assert cap_to_remaining is True


def test_edge_bsqvac_from_nestor_broadcasts_single_zeta_plane():
    nestor = SimpleNamespace(
        vac_total=SimpleNamespace(bsqvac=np.array([[1.0], [2.0]], dtype=float))
    )
    static = SimpleNamespace(cfg=SimpleNamespace(nzeta=3))

    out = residual_runtime.edge_bsqvac_from_nestor(nestor, static)

    assert out.shape == (2, 3)
    np.testing.assert_allclose(out, np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]))


def test_edge_bsqvac_from_nestor_keeps_full_zeta_grid():
    edge = np.array([[1.0, 1.5], [2.0, 2.5]], dtype=float)
    nestor = SimpleNamespace(vac_total=SimpleNamespace(bsqvac=edge))
    static = SimpleNamespace(cfg=SimpleNamespace(nzeta=3))

    out = residual_runtime.edge_bsqvac_from_nestor(nestor, static)

    assert out.shape == (2, 2)
    np.testing.assert_allclose(out, edge)


@pytest.mark.parametrize(
    ("scan_core_env", "scan_minimal", "fsq_total_target", "expected"),
    [
        ("", True, None, False),
        ("", False, 1.0, False),
        ("", True, 1.0, True),
        ("false", True, 1.0, False),
        ("yes", False, None, True),
    ],
)
def test_default_scan_core_policy(scan_core_env, scan_minimal, fsq_total_target, expected):
    assert (
        runtime._default_scan_core(
            scan_core_env=scan_core_env,
            scan_minimal=scan_minimal,
            fsq_total_target=fsq_total_target,
        )
        is expected
    )


def test_scan_fallback_policy_false_override_and_parse_defaults():
    policy = runtime._scan_fallback_policy(
        backend_name="cpu",
        enabled_env=" false ",
        iters_env="0",
        badjac_limit_env="bad",
        fsq_abs_env="bad",
        accept_frac_env="-0.25",
        fsq_factor_env="bad",
        improve_env="1.0",
    )

    assert policy.enabled is False
    assert policy.iters == 1
    assert policy.badjac_limit == 10
    assert policy.fsq_abs == 1.0e-2
    assert policy.accept_frac == 0.0
    assert policy.fsq_factor == 50.0
    assert policy.improve == 0.9


def test_scan_fallback_policy_accelerator_default_and_upper_clamps():
    policy = runtime._scan_fallback_policy(
        backend_name="gpu",
        enabled_env=None,
        iters_env="4",
        badjac_limit_env="-1",
        fsq_abs_env="-3.0",
        accept_frac_env="1.5",
        fsq_factor_env="0.25",
        improve_env="0.5",
    )

    assert policy.enabled is False
    assert policy.iters == 4
    assert policy.badjac_limit == 0
    assert policy.fsq_abs == 0.0
    assert policy.accept_frac == 1.0
    assert policy.fsq_factor == 1.0
    assert policy.improve == 0.5


def test_residual_convergence_flags_use_exact_thresholds_and_total_target():
    assert runtime._residual_convergence_flags(
        fsqr=1.0,
        fsqz=1.0,
        fsql=1.0,
        ftol=1.0,
        fsq_total_target=2.0,
    ) == (True, False, True)
    assert runtime._residual_convergence_flags(
        fsqr=2.0,
        fsqz=0.5,
        fsql=0.5,
        ftol=1.0,
        fsq_total_target=3.0,
    ) == (False, True, True)


def test_runtime_env_enabled_treats_none_as_nonempty_string():
    assert runtime._runtime_env_enabled(None) is True


def test_hash_array_bytes_reports_opaque_unarrayable_values():
    class Unarrayable:
        def __array__(self, dtype=None):
            raise TypeError("cannot materialize")

    assert runtime._hash_array_bytes(Unarrayable()) == "opaque:Unarrayable"


def test_array_keys_use_traced_aval_for_unarrayable_values(monkeypatch):
    class Unarrayable:
        def __array__(self, dtype=None):
            raise TypeError("cannot materialize")

    class Aval:
        shape = (2, 3)
        dtype = np.dtype("float32")
        weak_type = True

    class FakeJax:
        typeof = staticmethod(lambda _x: Aval())

    monkeypatch.setattr(runtime, "jax", FakeJax())

    value = Unarrayable()

    assert runtime._hash_array_bytes(value) == "traced:(2, 3):float32:True"
    assert runtime._array_signature_key(value) == ((2, 3), "float32")


def test_array_signature_key_falls_back_to_type_when_no_aval(monkeypatch):
    class Unarrayable:
        def __array__(self, dtype=None):
            raise TypeError("cannot materialize")

    class FakeCore:
        @staticmethod
        def get_aval(_x):
            raise TypeError("not a tracer")

    class FakeJax:
        typeof = None
        core = FakeCore()

    monkeypatch.setattr(runtime, "jax", FakeJax())

    assert runtime._array_signature_key(Unarrayable()) == ((), "Unarrayable")


def test_edge_signature_ignores_values_but_value_key_tracks_bytes():
    a = np.asarray([1.0, 2.0], dtype=np.float64)
    b = np.asarray([9.0, 8.0], dtype=np.float64)
    c = np.asarray([1.0, 2.0], dtype=np.float32)

    assert runtime._array_signature_key(a) == ((2,), "float64")
    assert runtime._edge_signature_key(a) == runtime._edge_signature_key(b)
    assert runtime._edge_signature_key(a) != runtime._edge_signature_key(c)
    assert runtime._edge_value_key(a) != runtime._edge_value_key(b)
    assert runtime._edge_value_key(a) != runtime._edge_value_key(c)


def test_dump_iter_selection_parses_ranges_and_invalid_chunks():
    assert runtime._parse_iter_list("") is None
    assert runtime._parse_iter_list("bad, 5-3, 8, 9-bad, ,") == {3, 4, 5, 8}
    assert runtime._dump_env_enabled("false") is True
    assert runtime._dump_env_enabled("0") is False
    assert runtime._dump_iter_selected(iter_idx=4, iter_env="1,3-5") is True
    assert runtime._dump_iter_selected(iter_idx=2, iter_env="1,3-5") is False
    assert runtime._dump_iter_selected(iter_idx=99, iter_env="bad") is True


def test_scan_backend_tree_and_scalar_history_helpers(monkeypatch):
    monkeypatch.setattr(runtime, "has_jax", lambda: False)
    assert runtime._scan_backend_name() == "cpu"
    assert runtime._tree_has_tracer({"x": np.asarray([1.0])}) is False

    monkeypatch.setattr(runtime, "has_jax", lambda: True)
    monkeypatch.setattr(runtime.jax, "default_backend", lambda: " GPU ")
    assert runtime._scan_backend_name() == "gpu"

    np.testing.assert_allclose(runtime._scalar_history_array([]), np.zeros((0,)))
    np.testing.assert_allclose(runtime._scalar_history_array([1, 2.5]), [1.0, 2.5])


def test_scan_backend_tree_and_history_exception_fallbacks(monkeypatch):
    monkeypatch.setattr(runtime, "jax", None)
    assert runtime._tree_has_tracer({"x": object()}) is False

    class _TreeUtil:
        @staticmethod
        def tree_leaves(_tree):
            raise RuntimeError("tree failure")

    class _Core:
        class Tracer:
            pass

    class _FakeJax:
        tree_util = _TreeUtil()
        core = _Core()

        @staticmethod
        def default_backend():
            raise RuntimeError("backend unavailable")

        @staticmethod
        def device_get(_vals):
            raise RuntimeError("device_get unavailable")

    monkeypatch.setattr(runtime, "jax", _FakeJax())
    monkeypatch.setattr(runtime, "has_jax", lambda: True)

    assert runtime._scan_backend_name() == "cpu"
    assert runtime._tree_has_tracer(object()) is False
    np.testing.assert_allclose(runtime._scalar_history_array([1, 2]), [1.0, 2.0])


def test_scan_fallback_policy_accept_frac_default_on_parse_error():
    policy = runtime._scan_fallback_policy(
        backend_name="cpu",
        enabled_env=None,
        iters_env="bad",
        badjac_limit_env="bad",
        fsq_abs_env="bad",
        accept_frac_env="bad",
        fsq_factor_env="bad",
        improve_env="bad",
    )

    assert policy.enabled is True
    assert policy.iters == 20
    assert policy.accept_frac == 0.5
