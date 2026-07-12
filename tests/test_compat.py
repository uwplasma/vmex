"""Unit tests for :mod:`vmec_jax._compat` (JAX/NumPy backend shim).

Covers the numpy-mode thread-local override, the no-op jit fallback, the
machine-scoped compilation-cache policy (env-var precedence table in
``_default_compilation_cache_dir``), the cache wiring against a recording
fake JAX module, and the backend helpers (``asarray``/``einsum``/
``enable_x64``/``x64_enabled``).
"""

from __future__ import annotations

import re
import types

import numpy as np
import pytest

from vmec_jax import _compat


# ---------------------------------------------------------------------------
# numpy-mode override + jit fallback
# ---------------------------------------------------------------------------


def test_numpy_mode_context_toggles_has_jax():
    assert _compat.has_jax()  # suite runs with real JAX
    assert not _compat.numpy_mode_enabled()
    with _compat.numpy_mode_context():
        assert _compat.numpy_mode_enabled()
        assert not _compat.has_jax()
        # einsum dispatches to numpy inside the context
        out = _compat.einsum("i,i->", np.arange(3.0), np.arange(3.0))
        assert isinstance(out, np.ndarray | np.floating | float)
    assert not _compat.numpy_mode_enabled()
    assert _compat.has_jax()


def test_noop_jit_bare_and_decorator_factory_forms():
    fn = lambda x: x + 1  # noqa: E731
    assert _compat._noop_jit(fn) is fn
    assert _compat._noop_jit(fn, static_argnames=("x",)) is fn
    # @partial(jit, static_argnames=...) form: jit() returns a decorator
    deco = _compat._noop_jit(None, static_argnames=("x",))
    assert deco(fn) is fn
    deco = _compat._noop_jit()
    assert deco(fn) is fn


# ---------------------------------------------------------------------------
# compilation-cache directory policy
# ---------------------------------------------------------------------------

_CACHE_VARS = (
    "JAX_COMPILATION_CACHE_DIR", "VMEC_JAX_COMPILATION_CACHE_DIR",
    "VMEC_JAX_COMPILATION_CACHE", "JAX_PLATFORM_NAME", "JAX_PLATFORMS",
)


@pytest.fixture
def clean_cache_env(monkeypatch):
    for var in _CACHE_VARS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_cache_dir_env_precedence(clean_cache_env):
    mp = clean_cache_env
    # nothing requested on CPU -> no cache
    assert _compat._default_compilation_cache_dir() is None

    # explicit JAX var wins verbatim; 'disabled' turns it off
    mp.setenv("JAX_COMPILATION_CACHE_DIR", "/tmp/jaxcache")
    assert _compat._default_compilation_cache_dir() == "/tmp/jaxcache"
    mp.setenv("JAX_COMPILATION_CACHE_DIR", "disabled")
    assert _compat._default_compilation_cache_dir() is None
    mp.delenv("JAX_COMPILATION_CACHE_DIR")

    # vmec-specific dir override
    mp.setenv("VMEC_JAX_COMPILATION_CACHE_DIR", "/tmp/vmeccache")
    assert _compat._default_compilation_cache_dir() == "/tmp/vmeccache"
    mp.setenv("VMEC_JAX_COMPILATION_CACHE_DIR", "no")
    assert _compat._default_compilation_cache_dir() is None
    mp.delenv("VMEC_JAX_COMPILATION_CACHE_DIR")

    # forced on CPU -> machine-scoped default under ~/.cache
    mp.setenv("VMEC_JAX_COMPILATION_CACHE", "1")
    path = _compat._default_compilation_cache_dir()
    assert path is not None and "vmec_jax" in path and "jax_cache" in path
    mp.setenv("VMEC_JAX_COMPILATION_CACHE", "off")
    assert _compat._default_compilation_cache_dir() is None
    mp.delenv("VMEC_JAX_COMPILATION_CACHE")

    # accelerator request enables the default cache
    mp.setenv("JAX_PLATFORMS", "cuda,cpu")
    assert _compat._default_compilation_cache_dir() is not None
    mp.delenv("JAX_PLATFORMS")
    mp.setenv("JAX_PLATFORM_NAME", "tpu")
    assert _compat._default_compilation_cache_dir() is not None


def test_cache_machine_fingerprint_shape_and_stability():
    fp = _compat._cache_machine_fingerprint()
    assert re.fullmatch(r"[a-z0-9_]+-[a-z0-9_]+-[0-9a-f]{16}", fp)
    assert fp == _compat._cache_machine_fingerprint()


class _FakeConfig:
    def __init__(self, fail_keys=()):
        self.updates = {}
        self.fail_keys = set(fail_keys)

    def update(self, key, value):
        if key in self.fail_keys:
            raise RuntimeError(f"cannot set {key}")
        self.updates[key] = value


def test_configure_compilation_cache_wiring(monkeypatch):
    fake = types.SimpleNamespace(config=_FakeConfig())
    monkeypatch.delenv("JAX_PLATFORM_NAME", raising=False)
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("VMEC_JAX_CACHE_MIN_COMPILE_TIME_SECS", "2.5")
    monkeypatch.setenv("VMEC_JAX_CACHE_MIN_ENTRY_SIZE_BYTES", "1024")
    monkeypatch.setenv("VMEC_JAX_COMPILATION_CACHE_MAX_SIZE", "100000")
    monkeypatch.setenv("VMEC_JAX_EXPLAIN_CACHE_MISSES", "1")
    monkeypatch.setenv("VMEC_JAX_PERSISTENT_CACHE_XLA_CACHES", "all")

    _compat._configure_compilation_cache(fake, "/tmp/cachedir")
    ups = fake.config.updates
    assert ups["jax_enable_compilation_cache"] is True
    assert ups["jax_compilation_cache_dir"] == "/tmp/cachedir"
    assert ups["jax_persistent_cache_min_compile_time_secs"] == 2.5
    assert ups["jax_persistent_cache_min_entry_size_bytes"] == 1024
    assert ups["jax_compilation_cache_max_size"] == 100000
    assert ups["jax_explain_cache_misses"] is True
    assert ups["jax_persistent_cache_enable_xla_caches"] == "all"

    # cache_dir=None is a no-op; failing config keys are tolerated
    fake2 = types.SimpleNamespace(config=_FakeConfig())
    _compat._configure_compilation_cache(fake2, None)
    assert fake2.config.updates == {}
    fake3 = types.SimpleNamespace(
        config=_FakeConfig(fail_keys={"jax_enable_compilation_cache"}))
    _compat._configure_compilation_cache(fake3, "/tmp/x")  # must not raise
    assert fake3.config.updates["jax_compilation_cache_dir"] == "/tmp/x"


def test_configure_compilation_cache_gpu_autotune_default(monkeypatch):
    monkeypatch.delenv("VMEC_JAX_PERSISTENT_CACHE_XLA_CACHES", raising=False)
    monkeypatch.delenv("JAX_PLATFORM_NAME", raising=False)
    monkeypatch.setenv("JAX_PLATFORMS", "cuda")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    fake = types.SimpleNamespace(config=_FakeConfig())
    _compat._configure_compilation_cache(fake, "/tmp/cachedir")
    assert (fake.config.updates["jax_persistent_cache_enable_xla_caches"]
            == "xla_gpu_per_fusion_autotune_cache_dir")


# ---------------------------------------------------------------------------
# backend helpers
# ---------------------------------------------------------------------------


def test_asarray_and_einsum_jax_backend():
    a = _compat.asarray([1.0, 2.0, 3.0])
    assert float(a.sum()) == 6.0
    dot = _compat.einsum("i,i->", a, a)
    assert float(dot) == pytest.approx(14.0)
    # explicit precision path
    from jax import lax

    dot2 = _compat.einsum("i,i->", a, a, precision=lax.Precision.DEFAULT)
    assert float(dot2) == pytest.approx(14.0)


def test_enable_x64_roundtrip_and_query():
    assert _compat.has_jax()
    original = _compat.x64_enabled()
    try:
        _compat.enable_x64(True)
        assert _compat.x64_enabled()
        _compat.enable_x64(False)
        assert not _compat.x64_enabled()
    finally:
        _compat.enable_x64(original)
    assert _compat.x64_enabled() == original


def test_helpers_with_jax_unavailable(monkeypatch):
    """The no-JAX branches of enable_x64/x64_enabled/einsum."""
    monkeypatch.setattr(_compat, "jax", None)
    assert not _compat.has_jax()
    _compat.enable_x64(True)  # must be a silent no-op
    assert _compat.x64_enabled() is True  # default-true without JAX
    out = _compat.einsum("i,i->", np.arange(3.0), np.arange(3.0))
    assert float(out) == pytest.approx(5.0)


def test_x64_enabled_env_fallback(monkeypatch):
    """When jax.config.read raises, x64_enabled falls back to the env var."""
    class _Cfg:
        def read(self, key):
            raise RuntimeError("no such config")

        def update(self, key, value):
            raise RuntimeError("frozen")

    monkeypatch.setattr(_compat, "jax", types.SimpleNamespace(config=_Cfg()))
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    assert _compat.x64_enabled() is True
    monkeypatch.setenv("JAX_ENABLE_X64", "0")
    assert _compat.x64_enabled() is False
    _compat.enable_x64(True)  # update raising is swallowed


def test_try_import_jax_mocked_module_falls_back_to_numpy(monkeypatch):
    """A non-module ``jax`` in sys.modules (docs mocking) -> NumPy fallback."""
    import sys

    class _Mock:  # not a types.ModuleType instance
        pass

    monkeypatch.setitem(sys.modules, "jax", _Mock())
    jax_mod, np_mod, jit = _compat._try_import_jax()
    assert jax_mod is None
    assert np_mod is np
    fn = lambda x: x  # noqa: E731
    assert jit(fn) is fn  # _noop_jit


def test_module_level_backend_objects():
    assert _compat.jax is not None
    assert _compat.jnp is not None
    assert callable(_compat.jit)
    assert hasattr(_compat.tree_util, "register_pytree_node_class")
    # _try_import_jax is idempotent
    jax2, jnp2, jit2 = _compat._try_import_jax()
    assert jax2 is not None and callable(jit2)
