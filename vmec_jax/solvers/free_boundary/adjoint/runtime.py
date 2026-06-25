"""Runtime utility helpers for free-boundary adjoint reports."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Callable


def block_until_ready_for_timing(value: Any, *, jax_module: Any, tree_util_module: Any) -> Any:
    """Synchronize JAX arrays before recording device timing diagnostics."""

    if jax_module is None:
        return value
    try:
        return jax_module.block_until_ready(value)
    except Exception:
        return tree_util_module.tree_map(lambda leaf: jax_module.block_until_ready(leaf), value)


def jax_named_scope(
    name: str,
    *,
    jax_module: Any,
    nullcontext_factory: Callable[[], Any] = nullcontext,
) -> Any:
    """Return a JAX named-scope context when supported, otherwise a no-op."""

    if jax_module is None or not hasattr(jax_module, "named_scope"):
        return nullcontext_factory()
    return jax_module.named_scope(name)
