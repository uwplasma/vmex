from __future__ import annotations

import builtins
import importlib.util
import pathlib
import platform
import sys
import types

import numpy as np
import pytest

from vmec_jax import _compat as compat


def test_numpy_mode_context_and_noop_jit_behaviour() -> None:
    assert compat.numpy_mode_enabled() is False
    with compat.numpy_mode_context() as ctx:
        assert ctx is not None
        assert compat.numpy_mode_enabled() is True
        assert compat.has_jax() is False
        np.testing.assert_allclose(compat.einsum("i,i->", np.asarray([1.0]), np.asarray([2.0])), 2.0)
    assert compat.numpy_mode_enabled() is False

    def fn(x):
        return x + 1

    assert compat._noop_jit(fn)(1) == 2
    assert compat._noop_jit(static_argnames=("x",))(fn)(1) == 2


def test_default_compilation_cache_dir_respects_environment(monkeypatch, tmp_path) -> None:
    for key in (
        "JAX_COMPILATION_CACHE_DIR",
        "VMEC_JAX_COMPILATION_CACHE_DIR",
        "VMEC_JAX_COMPILATION_CACHE",
        "JAX_PLATFORM_NAME",
        "JAX_PLATFORMS",
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
    ):
        monkeypatch.delenv(key, raising=False)

    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", "/user/cache")
    assert compat._default_compilation_cache_dir() == "/user/cache"
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", "disabled")
    assert compat._default_compilation_cache_dir() is None
    monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR")

    monkeypatch.setenv("VMEC_JAX_COMPILATION_CACHE_DIR", "/vmec/cache")
    assert compat._default_compilation_cache_dir() == "/vmec/cache"
    monkeypatch.setenv("VMEC_JAX_COMPILATION_CACHE_DIR", "no")
    assert compat._default_compilation_cache_dir() is None
    monkeypatch.delenv("VMEC_JAX_COMPILATION_CACHE_DIR")

    monkeypatch.setenv("VMEC_JAX_COMPILATION_CACHE", "off")
    assert compat._default_compilation_cache_dir() is None
    monkeypatch.delenv("VMEC_JAX_COMPILATION_CACHE")
    assert compat._default_compilation_cache_dir() is None

    monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setenv("VMEC_JAX_COMPILATION_CACHE", "1")
    forced_cache = compat._default_compilation_cache_dir()
    assert forced_cache is not None
    assert forced_cache.startswith(str(tmp_path / ".cache" / "vmec_jax" / "jax_cache"))
    monkeypatch.delenv("VMEC_JAX_COMPILATION_CACHE")

    monkeypatch.setenv("JAX_PLATFORM_NAME", "gpu")
    platform_name_cache = compat._default_compilation_cache_dir()
    assert platform_name_cache is not None
    assert platform_name_cache.startswith(str(tmp_path / ".cache" / "vmec_jax" / "jax_cache"))
    monkeypatch.delenv("JAX_PLATFORM_NAME")

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert compat._default_compilation_cache_dir() is None

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setenv("JAX_PLATFORMS", "cpu,gpu")
    platform_cache = compat._default_compilation_cache_dir()
    assert platform_cache is not None
    assert platform_cache.startswith(str(tmp_path / ".cache" / "vmec_jax" / "jax_cache"))


def test_configure_compilation_cache_updates_available_jax_settings(monkeypatch) -> None:
    calls = []

    class _Config:
        def update(self, key, value):
            calls.append((key, value))

    class _Jax:
        config = _Config()

    monkeypatch.setenv("VMEC_JAX_CACHE_MIN_COMPILE_TIME_SECS", "1.5")
    monkeypatch.setenv("VMEC_JAX_CACHE_MIN_ENTRY_SIZE_BYTES", "123")
    monkeypatch.setenv("VMEC_JAX_PERSISTENT_CACHE_XLA_CACHES", "xla_gpu_per_fusion_autotune_cache_dir")
    monkeypatch.setenv("VMEC_JAX_COMPILATION_CACHE_MAX_SIZE", "456")
    monkeypatch.setenv("VMEC_JAX_EXPLAIN_CACHE_MISSES", "yes")

    compat._configure_compilation_cache(_Jax(), "/tmp/vmec-cache")

    assert ("jax_enable_compilation_cache", True) in calls
    assert ("jax_compilation_cache_dir", "/tmp/vmec-cache") in calls
    assert ("jax_persistent_cache_min_compile_time_secs", 1.5) in calls
    assert ("jax_persistent_cache_min_entry_size_bytes", 123) in calls
    assert ("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir") in calls
    assert ("jax_compilation_cache_max_size", 456) in calls
    assert ("jax_explain_cache_misses", True) in calls


def test_configure_compilation_cache_is_best_effort() -> None:
    class _BadConfig:
        def update(self, *_args, **_kwargs):
            raise RuntimeError("unsupported option")

    class _Jax:
        config = _BadConfig()

    compat._configure_compilation_cache(_Jax(), None)
    compat._configure_compilation_cache(_Jax(), "/tmp/cache")


def test_cache_machine_fingerprint_uses_cpuinfo_and_fallbacks(monkeypatch) -> None:
    monkeypatch.setattr(compat.importlib_metadata, "version", lambda _name: (_ for _ in ()).throw(RuntimeError("missing")))
    monkeypatch.setattr(compat.os.path, "exists", lambda path: path == "/proc/cpuinfo")

    cpuinfo = "\n".join(
        [
            "model name : synthetic cpu",
            "model name : duplicate should be ignored",
            "flags : avx avx2",
            "unrelated : ignored",
        ]
    )

    def fake_open(path, *args, **kwargs):
        assert path == "/proc/cpuinfo"
        return _StringContext(cpuinfo)

    monkeypatch.setattr(builtins, "open", fake_open)

    fp = compat._cache_machine_fingerprint()

    assert platform.system().lower() in fp
    assert len(fp.rsplit("-", 1)[-1]) == 16

    monkeypatch.setattr(compat.os.path, "exists", lambda _path: False)
    monkeypatch.setattr(compat.platform, "system", lambda: "")
    monkeypatch.setattr(compat.platform, "machine", lambda: "")
    monkeypatch.setattr(compat.platform, "processor", lambda: "")
    monkeypatch.setattr(compat.platform, "node", lambda: "node-fallback")
    assert compat._cache_machine_fingerprint().startswith("unknown-unknown-")


def test_cache_machine_fingerprint_ignores_malformed_cpuinfo_and_open_errors(monkeypatch) -> None:
    monkeypatch.setattr(compat.importlib_metadata, "version", lambda _name: "1.0")
    monkeypatch.setattr(compat.os.path, "exists", lambda path: path == "/proc/cpuinfo")

    def fake_open(path, *args, **kwargs):
        assert path == "/proc/cpuinfo"
        return _StringContext("no delimiter here\nFeatures : neon\n")

    monkeypatch.setattr(builtins, "open", fake_open)
    assert len(compat._cache_machine_fingerprint().rsplit("-", 1)[-1]) == 16

    def raising_open(*_args, **_kwargs):
        raise OSError("cpuinfo unavailable")

    monkeypatch.setattr(builtins, "open", raising_open)
    assert len(compat._cache_machine_fingerprint().rsplit("-", 1)[-1]) == 16


class _StringContext:
    def __init__(self, text: str):
        self._lines = text.splitlines(True)

    def __enter__(self):
        return iter(self._lines)

    def __exit__(self, *_args):
        return False


def test_default_compilation_cache_dir_handles_home_failure_and_rocm(monkeypatch) -> None:
    for key in (
        "JAX_COMPILATION_CACHE_DIR",
        "VMEC_JAX_COMPILATION_CACHE_DIR",
        "VMEC_JAX_COMPILATION_CACHE",
        "JAX_PLATFORM_NAME",
        "JAX_PLATFORMS",
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
    ):
        monkeypatch.delenv(key, raising=False)

    monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: (_ for _ in ()).throw(OSError("home missing"))))
    monkeypatch.setenv("VMEC_JAX_COMPILATION_CACHE", "1")
    assert compat._default_compilation_cache_dir() is None
    monkeypatch.delenv("VMEC_JAX_COMPILATION_CACHE")

    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    assert compat._default_compilation_cache_dir() is None

    monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: pathlib.Path("/tmp/synthetic-home")))
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "-1")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "1")
    assert compat._default_compilation_cache_dir() is None

    monkeypatch.setenv("JAX_PLATFORM_NAME", "rocm")
    assert compat._default_compilation_cache_dir() is not None


def test_configure_compilation_cache_gpu_defaults_and_bad_numeric_env(monkeypatch) -> None:
    calls = []

    class _Config:
        def update(self, key, value):
            calls.append((key, value))

    class _Jax:
        config = _Config()

    monkeypatch.delenv("VMEC_JAX_PERSISTENT_CACHE_XLA_CACHES", raising=False)
    monkeypatch.setenv("JAX_PLATFORM_NAME", "gpu")
    monkeypatch.setenv("VMEC_JAX_CACHE_MIN_COMPILE_TIME_SECS", "bad")
    monkeypatch.setenv("VMEC_JAX_CACHE_MIN_ENTRY_SIZE_BYTES", "bad")
    monkeypatch.setenv("VMEC_JAX_COMPILATION_CACHE_MAX_SIZE", "bad")
    monkeypatch.setenv("VMEC_JAX_EXPLAIN_CACHE_MISSES", "no")

    compat._configure_compilation_cache(_Jax(), "/tmp/cache")

    assert ("jax_enable_compilation_cache", True) in calls
    assert ("jax_compilation_cache_dir", "/tmp/cache") in calls
    assert ("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir") in calls
    assert not any(key == "jax_explain_cache_misses" for key, _value in calls)


def test_configure_compilation_cache_cpu_default_disables_extra_xla_caches(monkeypatch) -> None:
    calls = []

    class _Config:
        def update(self, key, value):
            calls.append((key, value))

    class _Jax:
        config = _Config()

    monkeypatch.delenv("VMEC_JAX_PERSISTENT_CACHE_XLA_CACHES", raising=False)
    monkeypatch.delenv("JAX_PLATFORM_NAME", raising=False)
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    compat._configure_compilation_cache(_Jax(), "/tmp/cache")

    assert ("jax_enable_compilation_cache", True) in calls
    assert ("jax_compilation_cache_dir", "/tmp/cache") in calls
    assert not any(key == "jax_persistent_cache_enable_xla_caches" for key, _value in calls)


def test_configure_compilation_cache_ignores_optional_update_failures(monkeypatch) -> None:
    calls = []

    class _Config:
        def update(self, key, value):
            calls.append((key, value))
            if key in {
                "jax_persistent_cache_enable_xla_caches",
                "jax_explain_cache_misses",
            }:
                raise RuntimeError("unsupported")

    class _Jax:
        config = _Config()

    monkeypatch.setenv("VMEC_JAX_PERSISTENT_CACHE_XLA_CACHES", "xla_gpu_kernel_cache_file")
    monkeypatch.setenv("VMEC_JAX_EXPLAIN_CACHE_MISSES", "1")

    compat._configure_compilation_cache(_Jax(), "/tmp/cache")

    assert ("jax_persistent_cache_enable_xla_caches", "xla_gpu_kernel_cache_file") in calls
    assert ("jax_explain_cache_misses", True) in calls


def test_try_import_jax_falls_back_when_import_is_mocked(monkeypatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "jax":
            return object()
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    jax_mod, jnp_mod, jit = compat._try_import_jax()

    assert jax_mod is None
    assert jnp_mod is np

    def fn(x):
        return x + 2

    assert jit(fn)(3) == 5


def test_module_import_installs_tree_util_fallback_when_jax_is_unavailable(monkeypatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "jax" or name.startswith("jax."):
            raise ImportError("jax unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    module_name = "_vmec_jax_compat_nojax_test"
    spec = importlib.util.spec_from_file_location(module_name, pathlib.Path(compat.__file__))
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        assert module.jax is None

        @module.tree_util.register_pytree_node_class
        class _SyntheticTree:
            pass

        assert _SyntheticTree.__name__ == "_SyntheticTree"
    finally:
        sys.modules.pop(module_name, None)


def test_try_import_jax_sets_cache_env_and_ignores_config_update_failures(monkeypatch) -> None:
    fake_jax = types.ModuleType("jax")
    fake_jnp = types.ModuleType("jax.numpy")
    updates = []

    class _Config:
        def update(self, key, value):
            updates.append((key, value))
            if key in {"jax_enable_x64", "jax_cpu_enable_async_dispatch"}:
                raise RuntimeError("read-only")

    fake_jax.config = _Config()
    fake_jax.jit = lambda fn=None, **_kwargs: (lambda inner: inner) if fn is None else fn
    fake_jax.numpy = fake_jnp
    monkeypatch.setitem(sys.modules, "jax", fake_jax)
    monkeypatch.setitem(sys.modules, "jax.numpy", fake_jnp)
    monkeypatch.setattr(compat, "_default_compilation_cache_dir", lambda: "/tmp/synthetic-cache")
    monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR", raising=False)

    jax_mod, jnp_mod, jit = compat._try_import_jax()

    assert jax_mod is fake_jax
    assert jnp_mod is fake_jnp
    assert jit(lambda x: x)(4) == 4
    assert ("jax_enable_x64", True) in updates
    assert ("jax_cpu_enable_async_dispatch", False) in updates
    assert compat.os.environ["JAX_COMPILATION_CACHE_DIR"] == "/tmp/synthetic-cache"


def test_x64_enabled_falls_back_to_environment(monkeypatch) -> None:
    if compat.jax is None:
        pytest.skip("JAX not available")

    class _BadConfig:
        def read(self, _name):
            raise RuntimeError("config unavailable")

    monkeypatch.setattr(compat.jax, "config", _BadConfig())
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    assert compat.x64_enabled() is True
    monkeypatch.setenv("JAX_ENABLE_X64", "0")
    assert compat.x64_enabled() is False


def test_x64_and_array_helpers_when_jax_missing_or_limited(monkeypatch) -> None:
    monkeypatch.setattr(compat, "jax", None)
    assert compat.x64_enabled() is True
    compat.enable_x64(False)
    np.testing.assert_allclose(compat.asarray([1, 2], dtype=float), np.asarray([1.0, 2.0]))

    class _BadConfig:
        def update(self, *_args, **_kwargs):
            raise RuntimeError("cannot update")

    monkeypatch.setattr(compat, "jax", types.SimpleNamespace(config=_BadConfig()))
    compat.enable_x64(True)


def test_einsum_jax_precision_fallbacks(monkeypatch) -> None:
    class _FakeJnp:
        def __init__(self):
            self.calls = []

        def einsum(self, expr, *operands, precision=None):
            self.calls.append(precision)
            if precision is not None:
                raise TypeError("precision unsupported")
            return np.einsum(expr, *operands)

    fake_jnp = _FakeJnp()
    monkeypatch.setattr(compat, "jax", object())
    monkeypatch.setattr(compat, "jnp", fake_jnp)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "jax" and args and args[2] == ("lax",):
            raise ImportError("no lax")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    np.testing.assert_allclose(compat.einsum("i,i->", np.asarray([2.0]), np.asarray([3.0])), 6.0)
    assert fake_jnp.calls[-1] is None

    np.testing.assert_allclose(
        compat.einsum("i,i->", np.asarray([2.0]), np.asarray([3.0]), precision="highest"),
        6.0,
    )
    assert fake_jnp.calls[-2:] == ["highest", None]


def test_einsum_uses_real_jax_highest_precision_when_available() -> None:
    if compat.jax is None:
        pytest.skip("JAX not available")

    np.testing.assert_allclose(
        np.asarray(compat.einsum("i,i->", np.asarray([2.0]), np.asarray([4.0]))),
        8.0,
    )
