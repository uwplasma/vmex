"""The public closed-polyline sampler: wrap-around, endpoints and shape.

``sample_closed_polyline`` is the reference a spline or Fourier fit of a closed
magnetic axis is judged against, so its contract is worth pinning: the closing
segment is implied, arc length wraps, and the result lies on the polyline.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

from vmex.mirror import sample_closed_polyline  # noqa: E402


def _square():
    """Unit square in the z = 0 plane, perimeter 4, vertices given once."""
    return np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                     [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])


def test_vertices_are_reproduced_at_their_own_arc_lengths():
    points = _square()
    sampled = sample_closed_polyline(points, np.array([0.0, 1.0, 2.0, 3.0]))
    assert np.allclose(sampled, points)


def test_the_closing_segment_is_implied_not_repeated():
    """Arc length 3.5 is halfway along the segment from the last vertex to the first."""
    sampled = sample_closed_polyline(_square(), np.array([3.5]))
    assert np.allclose(sampled[0], [0.0, 0.5, 0.0])


def test_arc_length_wraps_in_both_directions():
    points = _square()
    inside = np.array([0.0, 0.7, 2.3, 3.9])
    for shift in (-8.0, -4.0, 4.0, 12.0):
        assert np.allclose(sample_closed_polyline(points, inside + shift),
                           sample_closed_polyline(points, inside))


def test_result_shape_follows_the_query_shape():
    points = _square()
    assert sample_closed_polyline(points, np.array([0.5])).shape == (1, 3)
    assert sample_closed_polyline(points, np.linspace(0.0, 4.0, 17)).shape == (17, 3)


def test_samples_lie_on_the_polyline():
    """Every sample of the square sits on one of its four edges."""
    sampled = sample_closed_polyline(_square(), np.linspace(0.0, 4.0, 401))
    x, y, z = sampled[:, 0], sampled[:, 1], sampled[:, 2]
    on_edge = (np.isclose(x, 0.0) | np.isclose(x, 1.0)
               | np.isclose(y, 0.0) | np.isclose(y, 1.0))
    assert np.all(on_edge) and np.allclose(z, 0.0)
    assert np.all((x >= -1e-12) & (x <= 1 + 1e-12))
    assert np.all((y >= -1e-12) & (y <= 1 + 1e-12))


def test_uniform_sampling_of_a_circle_recovers_its_length():
    """A fine closed polyline around a circle: sampled spacing is uniform in arc length."""
    angle = np.linspace(0.0, 2.0 * np.pi, 2001)[:-1]
    circle = np.stack([np.cos(angle), np.sin(angle), np.zeros_like(angle)], axis=1)
    total = float(np.sum(np.linalg.norm(
        np.diff(np.vstack([circle, circle[:1]]), axis=0), axis=1)))
    sampled = sample_closed_polyline(circle, np.linspace(0.0, total, 65)[:-1])
    steps = np.linalg.norm(np.diff(sampled, axis=0), axis=1)
    assert np.allclose(steps, steps[0], rtol=1e-6)
    assert np.allclose(np.linalg.norm(sampled, axis=1), 1.0, rtol=1e-5)
