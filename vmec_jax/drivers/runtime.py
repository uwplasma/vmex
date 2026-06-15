"""Runtime setup helpers for CLI/API driver entry points."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable


def maybe_enable_compilation_cache(
    *,
    accelerator_requested: bool = False,
    default_compilation_cache_dir: Callable[[], str | None],
    path_cls: Callable[[Any], Any] = Path,
) -> None:
    """Enable JAX's persistent compilation cache when policy/env allow it."""

    if os.getenv("VMEC_JAX_COMPILATION_CACHE", "").strip().lower() in ("0", "false", "no", "off"):
        return
    if os.getenv("VMEC_JAX_DISABLE_COMPILATION_CACHE", "") not in ("", "0"):
        return

    cache_dir = default_compilation_cache_dir()
    if (
        not cache_dir
        and bool(accelerator_requested)
        and "JAX_COMPILATION_CACHE_DIR" not in os.environ
        and "VMEC_JAX_COMPILATION_CACHE_DIR" not in os.environ
        and "VMEC_JAX_COMPILATION_CACHE" not in os.environ
    ):
        # solver_device="gpu" is a vmec_jax runtime request, not a JAX env var,
        # so _compat cannot see it during package import. Enable the same
        # machine-scoped cache policy here before the first solve compilation.
        os.environ["VMEC_JAX_COMPILATION_CACHE"] = "1"
        try:
            cache_dir = default_compilation_cache_dir()
        finally:
            os.environ.pop("VMEC_JAX_COMPILATION_CACHE", None)
    if str(cache_dir).strip().lower() in ("disabled", "0", "false", "no", "off"):
        return
    if not cache_dir:
        return
    try:
        import jax
        from jax.experimental.compilation_cache import compilation_cache

        cache_path = path_cls(cache_dir)
        try:
            cache_path.mkdir(parents=True, exist_ok=True)
        except Exception:
            # Fall back to /tmp when the home cache is not writable.
            try:
                cache_path = path_cls("/tmp/vmec_jax/jax_compilation_cache")
                cache_path.mkdir(parents=True, exist_ok=True)
            except Exception:
                return
        cache_dir = str(cache_path)
        compilation_cache.set_cache_dir(cache_dir)
        try:
            jax.config.update("jax_enable_compilation_cache", True)
        except Exception:
            pass
        try:
            jax.config.update("jax_compilation_cache_dir", cache_dir)
        except Exception:
            pass
        try:
            min_compile = os.getenv("VMEC_JAX_CACHE_MIN_COMPILE_TIME_SECS", "0")
            jax.config.update("jax_persistent_cache_min_compile_time_secs", float(min_compile))
            min_entry = os.getenv("VMEC_JAX_CACHE_MIN_ENTRY_SIZE_BYTES", "-1")
            jax.config.update("jax_persistent_cache_min_entry_size_bytes", int(min_entry))
            xla_caches = os.getenv("VMEC_JAX_PERSISTENT_CACHE_XLA_CACHES", "").strip()
            if not xla_caches and bool(accelerator_requested):
                xla_caches = "xla_gpu_per_fusion_autotune_cache_dir"
            if xla_caches.lower() not in ("", "none", "0", "false", "no", "off"):
                jax.config.update("jax_persistent_cache_enable_xla_caches", xla_caches)
            max_size = os.getenv("VMEC_JAX_COMPILATION_CACHE_MAX_SIZE", "")
            if max_size:
                jax.config.update("jax_compilation_cache_max_size", int(max_size))
            explain = os.getenv("VMEC_JAX_EXPLAIN_CACHE_MISSES", "")
            if explain.strip().lower() not in ("", "0", "false", "no"):
                jax.config.update("jax_explain_cache_misses", True)
        except Exception:
            pass
    except Exception:
        return


__all__ = ["maybe_enable_compilation_cache"]
