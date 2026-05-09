from pathlib import Path

from vmec_jax._compat import _default_compilation_cache_dir


def test_compilation_cache_is_disabled_by_default_on_cpu(monkeypatch):
    monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR", raising=False)
    monkeypatch.delenv("VMEC_JAX_COMPILATION_CACHE_DIR", raising=False)
    monkeypatch.delenv("VMEC_JAX_COMPILATION_CACHE", raising=False)
    monkeypatch.delenv("JAX_PLATFORM_NAME", raising=False)
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)

    assert _default_compilation_cache_dir() is None


def test_compilation_cache_dir_env_enables_cache(monkeypatch, tmp_path):
    monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR", raising=False)
    monkeypatch.delenv("VMEC_JAX_COMPILATION_CACHE", raising=False)
    monkeypatch.setenv("VMEC_JAX_COMPILATION_CACHE_DIR", str(tmp_path))

    assert Path(_default_compilation_cache_dir()) == tmp_path


def test_compilation_cache_flag_uses_default_cache(monkeypatch):
    monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR", raising=False)
    monkeypatch.delenv("VMEC_JAX_COMPILATION_CACHE_DIR", raising=False)
    monkeypatch.setenv("VMEC_JAX_COMPILATION_CACHE", "1")

    cache_dir = _default_compilation_cache_dir()

    assert cache_dir is not None
    assert ".cache/vmec_jax/jax_cache/" in cache_dir


def test_compilation_cache_auto_enables_for_requested_gpu(monkeypatch):
    monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR", raising=False)
    monkeypatch.delenv("VMEC_JAX_COMPILATION_CACHE_DIR", raising=False)
    monkeypatch.delenv("VMEC_JAX_COMPILATION_CACHE", raising=False)
    monkeypatch.setenv("JAX_PLATFORM_NAME", "gpu")

    cache_dir = _default_compilation_cache_dir()

    assert cache_dir is not None
    assert ".cache/vmec_jax/jax_cache/" in cache_dir


def test_compilation_cache_auto_enables_for_requested_gpu_platforms(monkeypatch):
    monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR", raising=False)
    monkeypatch.delenv("VMEC_JAX_COMPILATION_CACHE_DIR", raising=False)
    monkeypatch.delenv("VMEC_JAX_COMPILATION_CACHE", raising=False)
    monkeypatch.delenv("JAX_PLATFORM_NAME", raising=False)
    monkeypatch.setenv("JAX_PLATFORMS", "cpu,gpu")

    cache_dir = _default_compilation_cache_dir()

    assert cache_dir is not None
    assert ".cache/vmec_jax/jax_cache/" in cache_dir


def test_compilation_cache_auto_enables_for_visible_cuda_device(monkeypatch):
    monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR", raising=False)
    monkeypatch.delenv("VMEC_JAX_COMPILATION_CACHE_DIR", raising=False)
    monkeypatch.delenv("VMEC_JAX_COMPILATION_CACHE", raising=False)
    monkeypatch.delenv("JAX_PLATFORM_NAME", raising=False)
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    cache_dir = _default_compilation_cache_dir()

    assert cache_dir is not None
    assert ".cache/vmec_jax/jax_cache/" in cache_dir


def test_compilation_cache_disable_overrides_default(monkeypatch):
    monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR", raising=False)
    monkeypatch.delenv("VMEC_JAX_COMPILATION_CACHE_DIR", raising=False)
    monkeypatch.setenv("JAX_PLATFORM_NAME", "gpu")
    monkeypatch.setenv("VMEC_JAX_COMPILATION_CACHE", "0")

    assert _default_compilation_cache_dir() is None
