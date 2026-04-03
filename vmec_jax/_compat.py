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
import types

import os

import numpy as _np


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


def _try_import_jax() -> Tuple[Any, Any, Callable[[Callable[..., Any]], Callable[..., Any]]]:
    try:
        # Enable x64 by default for VMEC parity unless the user opted out.
        os.environ.setdefault("JAX_ENABLE_X64", "1")
        # Suppress noisy C++ warnings from XLA/PjRt backend (e.g.
        # "Assume version compatibility. PjRt-IFRT does not track XLA
        # executable versions.").  These are harmless informational
        # messages emitted by abseil logging inside the XLA runtime.
        # Level 0=INFO, 1=WARNING, 2=ERROR — we default to ERROR-only
        # so that genuine errors still surface.
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
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
    if jax is None:
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
