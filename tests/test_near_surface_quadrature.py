"""The target-graded rule that evaluates the exterior field next to the surface.

The direct periodic trapezoid rule loses accuracy as ``exp(-2 pi d / h)`` and
is useless within a few node spacings of the surface; these tests hold the
graded replacement to known answers there, check the per-point switch of
``near_surface="auto"``, and check that the rule traces and differentiates.
"""

from __future__ import annotations

import copy
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
    pytest.mark.skipif(not VC._HAVE_VCJ, reason="requires virtual_casing_jax"),
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


def test_auto_resolves_identically_eager_and_traced_and_pulls_back_the_returned_map():
    """Near the surface eager, jitted and differentiated ``auto`` evaluate one map."""
    surface = _two_source_torus(16, 16)
    scaled = lambda params: replace(surface, B_total=params[0] * surface.B_total)  # noqa: E731
    points = jnp.asarray(np.concatenate([_torus_points(0.01, count=2, seed=5),
                                         _torus_points(0.15, count=1, seed=6)]))
    parameters = jnp.array([1.0])
    original = VmecExtender.from_parameterized_surface_data(
        scaled, parameters, accuracy_check="off")
    field = original.set_points(points)
    copied = copy.copy(field)
    for order, name in enumerate(("B", "gradB", "gradgradB", "gradgradgradB")):
        eager = getattr(field, name)()
        # one map: the grading centre is a Newton iterate, so compilations agree
        # to the rule's own error (1e-9 at third order), not to rounding
        np.testing.assert_allclose(jax.jit(getattr(field, name))(points), eager,
                                   rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(getattr(copied, name)(), eager, rtol=1e-8, atol=1e-10)
        weights = jnp.cos(jnp.arange(eager.size)).reshape(eager.shape)
        pullback = np.asarray(getattr(field, name + "_vjp")(weights)).ravel()[0]

        def scalar(p, name=name, weights=weights):
            rebuilt = VmecExtender.from_surface_data(
                scaled(p), accuracy_check="off").set_points(points)
            return float(jnp.vdot(getattr(rebuilt, name)(), weights))
        step = 1e-3  # the field is linear in the parameter; a small step only adds round-off
        numeric = (scalar(parameters + step) - scalar(parameters - step)) / (2 * step)
        assert abs(pullback - numeric) <= 1e-6 * max(abs(numeric), 1.0), (order, pullback, numeric)


def test_a_copy_pulls_back_its_own_quadrature_not_the_original_s():
    """A copied field whose ``near_surface`` changes differentiates that map, with its own points."""
    surface = _two_source_torus(16, 16)
    scaled = lambda params: replace(surface, B_total=params[0] * surface.B_total)  # noqa: E731
    points = jnp.asarray(_torus_points(0.01, count=2, seed=5))
    original = VmecExtender.from_parameterized_surface_data(
        scaled, jnp.array([1.0]), accuracy_check="off").set_points(points)
    weights = jnp.cos(jnp.arange(6.0)).reshape(2, 3)
    graded = float(original.B_vjp(weights)[0])
    copied = copy.copy(original)
    copied.near_surface = "direct"
    # the direct rule is off by order one this close; the field is linear in the scale
    direct = float(jnp.vdot(VmecExtender.from_surface_data(
        surface, accuracy_check="off", near_surface="direct").B(points), weights))
    assert abs(float(copied.B_vjp(weights)[0]) - direct) <= 1e-9 * abs(direct)
    assert abs(float(original.B_vjp(weights)[0]) - graded) <= 1e-12 * abs(graded)
    assert abs(graded - direct) > 1e-2 * abs(graded)


def test_direct_tier_stays_the_plain_schedule_whatever_the_virtual_casing_routing():
    """Newer virtual-casing-jax routes ``B_plasma_xyz`` near the surface; ``auto`` must not see that."""
    field = VmecExtender.from_surface_data(_two_source_torus(16, 16), digits=4)
    assert getattr(field.plasma_field.config, "near_surface", False) is False
