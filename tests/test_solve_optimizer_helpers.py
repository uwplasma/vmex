from __future__ import annotations

import numpy as np
import pytest

from vmec_jax._compat import jnp
from vmec_jax.solve import _resolve_lbfgs_curvature_tol
from vmec_jax.solve_optimizer_helpers import (
    ensure_descent_direction,
    lbfgs_curvature_tolerance,
    lbfgs_two_loop_direction,
)


def test_lbfgs_two_loop_direction_empty_history_is_steepest_descent():
    g = jnp.asarray([1.0, -2.0, 3.0])

    direction = lbfgs_two_loop_direction(g, [], [])

    np.testing.assert_allclose(np.asarray(direction), [-1.0, 2.0, -3.0])


def test_lbfgs_two_loop_direction_uses_latest_curvature_pair_scaling():
    g = jnp.asarray([4.0, 3.0])
    s_hist = [jnp.asarray([1.0, 0.0])]
    y_hist = [jnp.asarray([2.0, 0.0])]

    direction = lbfgs_two_loop_direction(g, s_hist, y_hist)

    np.testing.assert_allclose(np.asarray(direction), [-2.0, -1.5])


def test_lbfgs_curvature_tolerance_tracks_dtype_and_solve_alias():
    s = np.asarray([3.0, 4.0], dtype=np.float32)
    y = np.asarray([0.0, 6.0], dtype=np.float32)
    expected = np.finfo(np.float32).eps * np.linalg.norm(s.ravel()) * np.linalg.norm(y.ravel())

    assert lbfgs_curvature_tolerance(s, y) == pytest.approx(expected)
    assert _resolve_lbfgs_curvature_tol(s, y) == pytest.approx(expected)


def test_ensure_descent_direction_preserves_descent_and_falls_back_otherwise():
    g = jnp.asarray([1.0, 2.0])
    descent = jnp.asarray([-0.5, -1.0])
    ascent = jnp.asarray([0.5, 1.0])

    direction, gtp, fallback = ensure_descent_direction(g, descent)
    np.testing.assert_allclose(np.asarray(direction), np.asarray(descent))
    assert gtp == pytest.approx(-2.5)
    assert fallback is False

    direction, gtp, fallback = ensure_descent_direction(g, ascent)
    np.testing.assert_allclose(np.asarray(direction), [-1.0, -2.0])
    assert gtp == pytest.approx(2.5)
    assert fallback is True
