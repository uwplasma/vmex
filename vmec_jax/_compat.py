"""Small compatibility layer.

We want minimal dependencies, but also want the code to be importable in environments
without JAX (e.g. for parsing / numpy-only debugging). If JAX is available, we use it.

Notes on float64
----------------
VMEC historically relies on float64. JAX defaults to float32 unless x64 is enabled.
To keep results stable and reduce warning spam, we *default* to enabling x64 when
JAX is imported, unless the user has explicitly set ``JAX_ENABLE_X64``.
"""

from __future__ import annotations

from typing import Any, Callable, Tuple
import hashlib
from importlib import metadata as importlib_metadata
import sys
import threading
import types

import os
import platform

import numpy as _np


# ---------------------------------------------------------------------------
# Thread-local NumPy-mode override (for pure-NumPy force hot path).
# ---------------------------------------------------------------------------

_numpy_mode_local = threading.local()


def numpy_mode_enabled() -> bool:
    """Return True if the current thread is inside a numpy_mode_context."""
    return getattr(_numpy_mode_local, "active", False)


class numpy_mode_context:
    """Context manager to force pure-NumPy paths in has_jax()-gated code.

    Inside this context, ``has_jax()`` returns False so that all
    ``jnp.*`` dispatch is replaced by ``np.*`` (via the fallback path).
    This eliminates all JAX eager-dispatch overhead in the force hot loop.
    """

    def __enter__(self):
        _numpy_mode_local.active = True
        return self

    def __exit__(self, *_):
        _numpy_mode_local.active = False


def _noop_jit(f=None, *args, **_kwargs):
    """Fallback jit decorator when JAX is unavailable.

    Accepts arbitrary args/kwargs so @partial(jit, static_argnames=...) works
    in docs builds and numpy-only environments.
    """
    if f is None:
        def _wrap(fn):
            return fn
        return _wrap
    return f


def _cache_machine_fingerprint() -> str:
    """Return a short cache key for host-specific XLA CPU executables.

    XLA CPU persistent-cache entries are native executables.  On shared home
    directories, reusing an entry compiled on another CPU can trigger XLA AOT
    loader errors or even illegal-instruction failures.  The fingerprint keeps
    vmec_jax's default cache portable by separating entries by OS, machine, and
    CPU-feature/model signature.  Users who deliberately want a shared cache can
    still set ``VMEC_JAX_COMPILATION_CACHE_DIR`` or ``JAX_COMPILATION_CACHE_DIR``.
    """

    parts = [
        platform.system(),
        platform.machine(),
        platform.processor(),
        f"python={sys.version_info.major}.{sys.version_info.minor}",
    ]
    for package in ("jax", "jaxlib"):
        try:
            parts.append(f"{package}={importlib_metadata.version(package)}")
        except Exception:
            pass
    try:
        if os.path.exists("/proc/cpuinfo"):
            wanted = ("model name", "cpu family", "model", "stepping", "flags", "Features")
            seen: set[str] = set()
            with open("/proc/cpuinfo", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    if ":" not in line:
                        continue
                    key, value = (part.strip() for part in line.split(":", 1))
                    if key in wanted and key not in seen:
                        parts.append(f"{key}={value}")
                        seen.add(key)
    except Exception:
        pass
    if not any(str(part).strip() for part in parts[:3]):
        try:
            parts.append(platform.node())
        except Exception:
            pass
    digest = hashlib.sha256("|".join(parts).encode("utf-8", errors="ignore")).hexdigest()[:16]
    system = platform.system().lower() or "unknown"
    machine = platform.machine().lower() or "unknown"
    return f"{system}-{machine}-{digest}"


def _default_compilation_cache_dir() -> str | None:
    """Return the configured JAX compilation-cache directory.

    The persistent cache is enabled automatically for accelerator-requested
    runs so cold-process CLI/API runs can reuse compiled kernels across
    invocations.  CPU persistent-cache entries are native AOT executables and
    can emit host-feature mismatch errors on some XLA/JAX versions, so CPU cache
    use is opt-in with ``VMEC_JAX_COMPILATION_CACHE=1`` unless the user provides
    an explicit cache directory.
    """
    # Already set by the user — respect it.
    if "JAX_COMPILATION_CACHE_DIR" in os.environ:
        val = os.environ["JAX_COMPILATION_CACHE_DIR"].strip()
        if val.lower() in ("", "disabled", "0", "false", "no"):
            return None
        return val

    # User can opt out via VMEC_JAX_COMPILATION_CACHE_DIR=disabled
    vmec_val = os.environ.get("VMEC_JAX_COMPILATION_CACHE_DIR", "").strip()
    if vmec_val.lower() in ("disabled", "0", "false", "no"):
        return None
    if vmec_val:
        return vmec_val

    cache_flag = os.environ.get("VMEC_JAX_COMPILATION_CACHE", "").strip().lower()
    if cache_flag in ("disabled", "0", "false", "no", "off"):
        return None
    cache_forced = cache_flag in ("1", "true", "yes", "on")

    platform_name = os.environ.get("JAX_PLATFORM_NAME", "").strip().lower()
    platforms = os.environ.get("JAX_PLATFORMS", "").strip().lower()
    accelerator_requested = (
        platform_name in ("gpu", "cuda", "rocm", "tpu")
        or any(
            part.strip() in ("gpu", "cuda", "rocm", "tpu")
            for part in platforms.split(",")
        )
    )
    if not (cache_forced or accelerator_requested):
        return None

    # Default cache location: ~/.cache/vmec_jax/jax_cache/<machine-fingerprint>
    # The host-specific suffix prevents unsafe XLA:CPU AOT reuse on shared home
    # filesystems where different machines see the same ~/.cache directory.
    try:
        import pathlib
        return str(
            pathlib.Path.home()
            / ".cache"
            / "vmec_jax"
            / "jax_cache"
            / _cache_machine_fingerprint()
        )
    except Exception:
        return None


def _configure_compilation_cache(jax_module: Any, cache_dir: str | None) -> None:
    """Apply vmec_jax's persistent-cache defaults to an imported JAX module."""
    if cache_dir is None:
        return
    try:
        jax_module.config.update("jax_enable_compilation_cache", True)
    except Exception:
        pass
    try:
        jax_module.config.update("jax_compilation_cache_dir", cache_dir)
    except Exception:
        pass
    try:
        min_compile = os.environ.get("VMEC_JAX_CACHE_MIN_COMPILE_TIME_SECS", "0")
        jax_module.config.update("jax_persistent_cache_min_compile_time_secs", float(min_compile))
    except Exception:
        pass
    try:
        min_entry = os.environ.get("VMEC_JAX_CACHE_MIN_ENTRY_SIZE_BYTES", "-1")
        jax_module.config.update("jax_persistent_cache_min_entry_size_bytes", int(min_entry))
    except Exception:
        pass
    try:
        xla_caches = os.environ.get("VMEC_JAX_PERSISTENT_CACHE_XLA_CACHES", "").strip()
        if not xla_caches:
            platform_name = os.environ.get("JAX_PLATFORM_NAME", "").strip().lower()
            platforms = os.environ.get("JAX_PLATFORMS", "").strip().lower()
            visible_cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip().lower()
            gpu_requested = (
                platform_name in ("gpu", "cuda")
                or any(part.strip() in ("gpu", "cuda") for part in platforms.split(","))
                or visible_cuda not in ("", "-1", "none", "no")
            )
            xla_caches = "xla_gpu_per_fusion_autotune_cache_dir" if gpu_requested else "none"
        if xla_caches.lower() not in ("", "none", "0", "false", "no", "off"):
            jax_module.config.update("jax_persistent_cache_enable_xla_caches", xla_caches)
    except Exception:
        pass
    try:
        max_size = os.environ.get("VMEC_JAX_COMPILATION_CACHE_MAX_SIZE", "")
        if max_size:
            jax_module.config.update("jax_compilation_cache_max_size", int(max_size))
    except Exception:
        pass
    try:
        explain = os.environ.get("VMEC_JAX_EXPLAIN_CACHE_MISSES", "")
        if explain.strip().lower() not in ("", "0", "false", "no"):
            jax_module.config.update("jax_explain_cache_misses", True)
    except Exception:
        pass


def _try_import_jax() -> Tuple[Any, Any, Callable[[Callable[..., Any]], Callable[..., Any]]]:
    try:
        # Enable x64 by default for VMEC parity unless the user opted out.
        os.environ.setdefault("JAX_ENABLE_X64", "1")
        # VMEC/JAX optimization callbacks immediately materialize most results
        # on the host (SciPy residuals/Jacobians, history, wout writing).  On
        # CPU, asynchronous dispatch can leave completed XLA/PjRt work and
        # executable state queued across many exact-Jacobian callbacks in one
        # long-lived process.  Default CPU dispatch to synchronous execution so
        # memory is reclaimed at callback boundaries; users can still override
        # this before import with JAX_CPU_ENABLE_ASYNC_DISPATCH=true.
        os.environ.setdefault("JAX_CPU_ENABLE_ASYNC_DISPATCH", "false")
        # Suppress noisy C++ warnings from XLA/PjRt backend (e.g.
        # repeated "Assume version compatibility. PjRt-IFRT does not
        # track XLA executable versions." on persistent-cache hits).
        # These are harmless informational messages emitted by the XLA
        # runtime logging stack. Level 0=INFO, 1=WARNING, 2=ERROR — we
        # default to ERROR-only so that genuine errors still surface.
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")
        os.environ.setdefault("GLOG_minloglevel", "2")
        # JAX's default GPU allocator preallocates most device memory.  That
        # hurts vmec_jax's exact-optimizer workload in practice: it prevents
        # concurrent profiling/worker processes from starting and can make the
        # accepted-point replay path much slower.  Default to demand allocation
        # unless the user already set JAX's allocator env var or explicitly
        # asks vmec_jax to keep JAX's preallocation default.
        _vmec_gpu_prealloc = os.environ.get("VMEC_JAX_GPU_PREALLOCATE", "").strip().lower()
        if (
            "XLA_PYTHON_CLIENT_PREALLOCATE" not in os.environ
            and _vmec_gpu_prealloc not in ("1", "true", "yes", "on")
        ):
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

        # Enable the JAX disk compilation cache in a machine-scoped directory.
        # This avoids unsafe XLA:CPU AOT reuse across hosts while preserving
        # repeated cold-process speedups on the same machine.
        _cache_dir = _default_compilation_cache_dir()
        if _cache_dir is not None:
            os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", _cache_dir)

        import jax

        # If Sphinx (or other tooling) has inserted a mock, treat JAX as unavailable.
        if not isinstance(jax, types.ModuleType):
            raise ImportError("mocked jax module")

        # Also set via config. This must happen before importing `jax.numpy`
        # to reliably affect dtype defaults.
        try:
            jax.config.update("jax_enable_x64", os.environ.get("JAX_ENABLE_X64", "0") == "1")
        except Exception:
            pass
        try:
            _cpu_async = os.environ.get("JAX_CPU_ENABLE_ASYNC_DISPATCH", "true")
            jax.config.update(
                "jax_cpu_enable_async_dispatch",
                _cpu_async.strip().lower() not in ("0", "false", "no", "off"),
            )
        except Exception:
            pass

        # Wire up the compilation cache via jax.config too; the env-var path
        # alone does not cover all JAX/JAXLIB versions and cache thresholds.
        _configure_compilation_cache(jax, _cache_dir)

        import jax.numpy as jnp

        return jax, jnp, jax.jit
    except Exception:
        # numpy fallback: no autodiff, no jit
        return None, _np, _noop_jit


jax, jnp, jit = _try_import_jax()

try:
    if jax is None:
        raise ImportError
    from jax import tree_util as tree_util  # type: ignore
except Exception:
    class _TreeUtilFallback:
        @staticmethod
        def register_pytree_node_class(cls):
            return cls

    tree_util = _TreeUtilFallback()


def has_jax() -> bool:
    if getattr(_numpy_mode_local, "active", False):
        return False
    return jax is not None


def enable_x64(enable: bool = True) -> None:
    """Enable/disable float64 for JAX (no-op if JAX unavailable).

    VMEC historically relies on float64; we therefore enable x64 by default.
    This helper is useful in scripts/tests to be explicit.
    """
    if jax is None:
        return
    try:
        jax.config.update("jax_enable_x64", bool(enable))
    except Exception:
        # If JAX is already initialized in a way that disallows toggling,
        # we silently ignore (the existing dtype policy will apply).
        pass


def x64_enabled() -> bool:
    if jax is None:
        return True
    try:
        return bool(jax.config.read("jax_enable_x64"))
    except Exception:
        return bool(os.environ.get("JAX_ENABLE_X64", "0") == "1")


def asarray(x: Any, dtype: Any | None = None):
    """Create an array using the active backend (jax.numpy or numpy)."""
    return jnp.asarray(x, dtype=dtype)


def einsum(expr: str, *operands: Any, precision: Any | None = None):
    """Backend-aware einsum with high-precision accumulation when available."""
    if jax is None or getattr(_numpy_mode_local, "active", False):
        return _np.einsum(expr, *operands)
    if precision is None:
        try:
            from jax import lax

            precision = lax.Precision.HIGHEST
        except Exception:
            precision = None
    if precision is None:
        return jnp.einsum(expr, *operands)
    try:
        return jnp.einsum(expr, *operands, precision=precision)
    except TypeError:
        return jnp.einsum(expr, *operands)
