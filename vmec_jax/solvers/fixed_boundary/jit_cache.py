"""Small JIT-cache helpers shared by VMEC solve hot paths."""

from __future__ import annotations

from collections import OrderedDict
import os
from typing import Any

from .performance import scan_cache_miss_category_counts


def jit_cache_limit(env_name: str, default: int) -> int:
    """Return a non-negative JIT-cache size limit from an environment variable."""

    raw = os.getenv(env_name, str(default)).strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return max(0, int(default))


def jit_cache_get(cache: OrderedDict[tuple, Any], key: tuple):
    """Return a cached value and mark it as recently used."""

    cached = cache.get(key)
    if cached is not None:
        cache.move_to_end(key)
    return cached


def jit_cache_put(cache: OrderedDict[tuple, Any], key: tuple, value, *, env_name: str, default: int):
    """Insert a cached value while respecting an environment-controlled LRU limit."""

    limit = jit_cache_limit(env_name, default)
    if limit == 0:
        return value
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > limit:
        cache.popitem(last=False)
    return value


def strict_update_static_cache_key(static) -> tuple[Any, ...]:
    """Return a structural cache key for strict-update kernels.

    The strict-update kernel only depends on VMEC mode/radial layout metadata,
    not on the Python object identity of ``static``. Keying on ``id(static)``
    caused accepted-point exact optimizer solves to compile/cache a new GPU
    update kernel for each otherwise-identical callback.
    """

    cfg = static.cfg
    modes = getattr(static, "modes", None)
    s = getattr(static, "s", None)
    s_shape = tuple(getattr(s, "shape", ()))
    s_dtype = str(getattr(s, "dtype", ""))
    return (
        int(getattr(cfg, "ns", 0)),
        int(getattr(cfg, "mpol", 0)),
        int(getattr(cfg, "ntor", 0)),
        int(getattr(cfg, "nfp", 0)),
        bool(getattr(cfg, "lasym", False)),
        bool(getattr(cfg, "lthreed", False)),
        int(getattr(modes, "K", 0)) if modes is not None else 0,
        s_shape,
        s_dtype,
    )


def record_scan_runner_cache_miss_categories(
    stats: dict[str, float | int],
    *,
    requested_key: tuple,
    existing_keys,
) -> None:
    """Record stable scan-runner miss causes into timing diagnostics."""

    try:
        counts = scan_cache_miss_category_counts(tuple(requested_key), tuple(existing_keys))
    except Exception:
        counts = {"unknown": 1}
    for category, count in counts.items():
        safe_category = "".join(ch if ch.isalnum() else "_" for ch in str(category).strip().lower()).strip("_")
        if not safe_category:
            safe_category = "unknown"
        key = f"scan_runner_cache_miss_category_{safe_category}_count"
        stats[key] = int(stats.get(key, 0)) + int(count)
