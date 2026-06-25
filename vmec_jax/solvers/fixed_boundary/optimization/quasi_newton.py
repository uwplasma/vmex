"""Pure optimizer math helpers used by fixed-boundary solvers."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ...._compat import jnp


def lbfgs_curvature_tolerance(s_vec: Any, y_vec: Any) -> float:
    """Return the scale-aware curvature threshold for accepting an L-BFGS pair."""
    s_np = np.asarray(s_vec)
    y_np = np.asarray(y_vec)
    dtype = np.result_type(s_np.dtype, y_np.dtype)
    scale = float(np.linalg.norm(np.ravel(s_np)) * np.linalg.norm(np.ravel(y_np)))
    return float(np.finfo(np.dtype(dtype)).eps * scale)


def lbfgs_two_loop_direction(g_flat: Any, s_hist: Sequence[Any], y_hist: Sequence[Any]) -> Any:
    """Compute the L-BFGS search direction using the standard two-loop recursion."""
    if len(s_hist) == 0:
        return -g_flat

    q = g_flat
    alpha = []
    rho = []
    for s_i, y_i in zip(reversed(s_hist), reversed(y_hist)):
        ys = jnp.dot(y_i, s_i)
        rho_i = jnp.where(ys != 0, 1.0 / ys, 0.0)
        a_i = rho_i * jnp.dot(s_i, q)
        q = q - a_i * y_i
        alpha.append(a_i)
        rho.append(rho_i)

    s0 = s_hist[-1]
    y0 = y_hist[-1]
    ys0 = jnp.dot(y0, s0)
    yy0 = jnp.dot(y0, y0)
    gamma0 = jnp.where(yy0 != 0, ys0 / yy0, 1.0)
    r = gamma0 * q

    for s_i, y_i, a_i, rho_i in zip(s_hist, y_hist, reversed(alpha), reversed(rho)):
        beta = rho_i * jnp.dot(y_i, r)
        r = r + s_i * (a_i - beta)

    return -r


def ensure_descent_direction(g_flat: Any, p_flat: Any) -> tuple[Any, float, bool]:
    """Return ``p_flat`` unless it is not a finite descent direction."""
    gtp = float(np.asarray(jnp.dot(g_flat, p_flat)))
    if not np.isfinite(gtp) or gtp >= 0.0:
        return -g_flat, gtp, True
    return p_flat, gtp, False
