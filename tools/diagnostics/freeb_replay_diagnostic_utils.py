"""Shared helpers for optional free-boundary replay diagnostics."""

from __future__ import annotations

import time
from typing import Any

import numpy as np


def block_until_ready(value: Any) -> Any:
    """Synchronize a JAX value when possible and return it unchanged."""

    if hasattr(value, "block_until_ready"):
        return value.block_until_ready()
    return value


def json_ready(value: Any) -> Any:
    """Convert NumPy/JAX-friendly report values into JSON-safe Python values."""

    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, dict):
        return {str(key): json_ready(val) for key, val in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(item) for item in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def timed_call(fn: Any, *args: Any, warm_repeats: int) -> tuple[Any, float, list[float]]:
    """Return ``fn(*args)`` plus first-call and warm-call wall timings."""

    t0 = time.perf_counter()
    value = fn(*args)
    block_until_ready(value)
    first = time.perf_counter() - t0
    warm: list[float] = []
    for _ in range(max(0, int(warm_repeats))):
        t0 = time.perf_counter()
        value = fn(*args)
        block_until_ready(value)
        warm.append(time.perf_counter() - t0)
    return value, first, warm
