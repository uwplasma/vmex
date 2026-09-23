"""The target-graded rule that evaluates the exterior field next to the surface.

The direct periodic trapezoid rule loses accuracy as ``exp(-2 pi d / h)`` and
is useless within a few node spacings of the surface; these tests hold the
graded replacement to known answers there, check the per-point switch of
``near_surface="auto"``, and check that the rule traces and differentiates.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

from tests.test_virtual_casing_physics import (  # noqa: E402
    _finest_spacing,
    _ring_field,
    _torus_points,
    _two_source_torus,
    _z_axis_field,
)
from vmex.core import virtual_casing as VC  # noqa: E402
from vmex.core.extender import ExteriorFieldAccuracyError, VmecExtender  # noqa: E402

pytestmark = [
    pytest.mark.usefixtures("_module_jit_enabled"),
    pytest.mark.skipif(not VC.have_virtual_casing_jax(), reason="requires virtual_casing_jax"),
]



def test_graded_rule_reaches_the_torus_oracle_next_to_the_surface_on_both_sides():
    """Known answers where the direct rule is useless, a few mm off a 0.3 m torus.

    Outside, the plasma field is the inside filament's field; inside, minus
    the z-axis field.  The direct rule on the same data is off by order one
    there.  Measured: 6e-10 of the field scale at 1 mm, 1e-10 at 1 cm; the
    gradient to 1.4e-5 at 1 mm.
    """
    surface = _two_source_torus(24, 24)
    scale = float(np.sqrt(np.mean(np.sum(np.asarray(surface.B_total) ** 2, axis=0))))
    for distance in (0.01, 0.001):
        outside = _torus_points(distance, count=4)
        inside = _torus_points(-distance, count=4)
        B_out, grad_out = VC.graded_plasma_field(surface, jnp.asarray(outside), order=1)
        B_in, = VC.graded_plasma_field(surface, jnp.asarray(inside))
        error = np.linalg.norm(np.asarray(B_out) - _ring_field(outside), axis=1) / scale
        assert error.max() <= 5e-9, (distance, error)
        error = np.linalg.norm(np.asarray(B_in) + _z_axis_field(inside), axis=1) / scale
        assert error.max() <= 5e-9, (distance, error)
        step = 1e-6
        numeric = np.stack([(_ring_field(outside + step * e) - _ring_field(outside - step * e))
                            / (2 * step) for e in np.eye(3)], axis=-1)
        error = np.abs(np.asarray(grad_out) - numeric).max() / np.abs(numeric).max()
        # the gradient kernel is one order more singular: 1.4e-5 at 1 mm
        assert error <= (1e-6 if distance > 0.005 else 1e-4), (distance, error)


def test_auto_switches_unresolved_points_to_the_graded_rule():
    """Where the direct estimate misses, ``B`` and its derivatives come from the graded rule."""
    digits = 6
    surface = _two_source_torus(24, 24)
    field = VmecExtender.from_surface_data(
        surface, digits=digits, levels=((48, 24), (96, 48)), accuracy_check="raise")
    assert field.near_surface == "auto"
    h = _finest_spacing(field)
    points = np.concatenate([_torus_points(0.3 * h, count=3), _torus_points(4.0 * h, count=3)])
    scale = float(np.sqrt(np.mean(np.sum(np.asarray(surface.B_total) ** 2, axis=0))))

    value = np.asarray(field.B(jnp.asarray(points)))  # raises if any point misses
    error = np.linalg.norm(value - _ring_field(points), axis=1) / scale
    assert error.max() <= 10.0 ** -digits, error
    assert np.asarray(field.B_error_estimate(jnp.asarray(points))).max() <= 10.0 ** -digits
    gradient = np.asarray(field.gradB(jnp.asarray(points)))
    step = 1e-6
    numeric = np.stack([(_ring_field(points + step * e) - _ring_field(points - step * e))
                        / (2 * step) for e in np.eye(3)], axis=-1)
    assert np.abs(gradient - numeric).max() <= 1e-5 * np.abs(numeric).max()

    field.near_surface = "direct"
    with pytest.raises(ExteriorFieldAccuracyError, match="3 of 6 points"):
        field.B(jnp.asarray(points))
    field.accuracy_check = "off"
    direct = np.asarray(field.B(jnp.asarray(points)))
    assert (np.linalg.norm(direct - _ring_field(points), axis=1) / scale)[:3].min() > 1e-2
    # far points are the direct path's own values in every mode
    np.testing.assert_allclose(direct[3:], value[3:], rtol=0, atol=1e-14)


def test_graded_quadrature_is_traceable_and_differentiable_in_the_surface():
    """``with_graded_quadrature`` works under ``jit`` and pulls back to the surface data."""
    surface = _two_source_torus(16, 16)
    points = jnp.asarray(_torus_points(0.005, count=2))
    field = VmecExtender.from_surface_data(surface, digits=4).with_graded_quadrature(
        nodes=(64, 256))
    assert field.near_surface == "graded"
    eager = np.asarray(field.B(points))
    np.testing.assert_allclose(np.asarray(jax.jit(field.B)(points)), eager, rtol=1e-12)

    weight = jnp.asarray(np.random.default_rng(3).normal(size=(2, 3)))

    def value(B_total):
        return jnp.vdot(VC.graded_plasma_field(
            replace(surface, B_total=B_total), points, nodes=(64, 256))[0], weight)

    direction = jnp.asarray(np.random.default_rng(4).normal(size=surface.B_total.shape))
    autodiff = float(jnp.vdot(jax.grad(value)(surface.B_total), direction))
    step = 1e-6
    numeric = (float(value(surface.B_total + step * direction))
               - float(value(surface.B_total - step * direction))) / (2 * step)
    assert abs(autodiff - numeric) <= 1e-6 * abs(numeric), (autodiff, numeric)
