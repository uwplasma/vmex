"""Objective algebra helpers for free-boundary adjoint validation."""

from __future__ import annotations

from typing import Any

import numpy as np

from vmec_jax._compat import jnp, tree_util


def weighted_half_norm(value: Any, weight: Any) -> Any:
    """Return ``0.5 * sum(weight * value**2)`` with scalar/array weights."""

    arr = jnp.asarray(value)
    w = jnp.asarray(weight, dtype=arr.dtype)
    return 0.5 * jnp.sum(w * arr * arr)


def static_weight_is_zero(weight: Any) -> bool:
    """Return true only for host-known scalar/array weights that are exactly zero."""

    try:
        arr = np.asarray(weight)
    except Exception:
        return False
    if arr.size == 0:
        return False
    try:
        return bool(np.all(arr == 0.0))
    except Exception:
        return False


def tree_weighted_half_norm(values: Any, weight: Any) -> Any:
    """Return the sum of weighted half-norms over numeric pytree leaves."""

    leaves = tree_util.tree_leaves(values)
    if not leaves:
        return jnp.asarray(0.0)
    total = jnp.asarray(0.0)
    for leaf in leaves:
        if leaf is None:
            continue
        try:
            total = total + weighted_half_norm(leaf, weight)
        except TypeError:
            continue
    return total
