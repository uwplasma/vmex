"""JIT-cache seams for residual-iteration force evaluation."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable

__all__ = [
    "compute_forces_jit_cache_key",
    "select_compute_forces_callable",
]


def compute_forces_jit_cache_key(
    *,
    static_key: tuple[Any, ...],
    wout_key: tuple[Any, ...],
    signgs: int,
    apply_m1_constraints: bool,
) -> tuple[Any, ...]:
    """Return the structural cache key for the residual force JIT closure."""

    return (
        "compute_forces_v1",
        tuple(static_key),
        tuple(wout_key),
        int(signgs),
        bool(apply_m1_constraints),
    )


def select_compute_forces_callable(
    compute_forces_nodump: Callable[..., Any],
    *,
    differentiating_scan: bool,
    cache: OrderedDict[tuple, Any],
    cache_key: tuple[Any, ...],
    jit_func: Callable[..., Any],
    cache_get: Callable[[OrderedDict[tuple, Any], tuple[Any, ...]], Any],
    cache_put: Callable[..., Any],
    cache_env_name: str = "VMEC_JAX_COMPUTE_FORCES_CACHE_SIZE",
    cache_default: int = 32,
) -> Any:
    """Return a JIT-wrapped force callable without changing cache ownership."""

    if bool(differentiating_scan):
        # Do not store a jitted closure created while tracing the scan solve:
        # it can retain traced closure constants and leak them out of the
        # transformation.  Primal solves still reuse the global cache.
        return jit_func(
            compute_forces_nodump,
            static_argnames=("include_edge", "include_edge_residual"),
        )

    cached = cache_get(cache, cache_key)
    if cached is not None:
        return cached
    cached = jit_func(
        compute_forces_nodump,
        static_argnames=("include_edge", "include_edge_residual"),
    )
    return cache_put(
        cache,
        cache_key,
        cached,
        env_name=cache_env_name,
        default=int(cache_default),
    )
