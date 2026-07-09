"""M3 pressure-closure consistency and anisotropy validity tests."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from vmec_jax.mirror import (  # noqa: E402
    BiMaxwellianPressureClosure,
    IsotropicPressureClosure,
    TabulatedPressureClosure,
    anisotropy_indicators,
)
from vmec_jax.mirror.forces import MU0  # noqa: E402


def test_isotropic_closure_is_exact_limit_and_has_positive_indicators() -> None:
    closure = IsotropicPressureClosure(jnp.asarray([2.0e4, -1.5e4]))
    s = jnp.linspace(0.0, 1.0, 9)[:, None]
    b = jnp.linspace(0.4, 1.2, 11)[None, :]
    moments = closure.moments(s, b)
    expected = 2.0e4 - 1.5e4 * s + jnp.zeros_like(b)
    np.testing.assert_allclose(moments.parallel, expected)
    np.testing.assert_allclose(moments.perpendicular, expected)
    indicators = anisotropy_indicators(closure, s, b)
    np.testing.assert_allclose(indicators.sigma, 1.0 / MU0, rtol=2.0e-14)
    np.testing.assert_allclose(indicators.mirror_ellipticity, 1.0 / MU0, rtol=2.0e-14)
    assert bool(indicators.valid)


def test_bimaxwellian_form_factor_matches_animec_passing_formula() -> None:
    closure = BiMaxwellianPressureClosure(
        thermal_coefficients=jnp.asarray([1.0e4, -5.0e3]),
        hot_fraction_coefficients=jnp.asarray([0.4]),
        temperature_ratio=0.3,
        critical_field=0.5,
    )
    b = jnp.asarray([0.55, 0.8, 1.1])
    normalized = b / 0.5
    expected = normalized / (1.0 - 0.3 * (1.0 - normalized))
    np.testing.assert_allclose(closure.form_factor(b), expected, rtol=2.0e-14, atol=2.0e-14)

    below = closure.form_factor(jnp.asarray([0.5 - 1.0e-8]))
    at = closure.form_factor(jnp.asarray([0.5]))
    assert np.isfinite(float(below[0]))
    np.testing.assert_allclose(below, at, rtol=3.0e-8, atol=3.0e-8)


def test_bimaxwellian_isotropic_passing_limit_and_parallel_force_identity() -> None:
    isotropic_passing = BiMaxwellianPressureClosure(
        thermal_coefficients=jnp.asarray([1.2e4, -2.0e3]),
        hot_fraction_coefficients=jnp.asarray([0.25, -0.1]),
        temperature_ratio=1.0,
        critical_field=0.2,
    )
    s = jnp.linspace(0.0, 1.0, 7)
    b = jnp.linspace(0.4, 1.0, 7)
    moments = isotropic_passing.moments(s, b)
    np.testing.assert_allclose(isotropic_passing.form_factor(b), 1.0, rtol=2.0e-14)
    np.testing.assert_allclose(moments.parallel, moments.perpendicular, rtol=3.0e-14, atol=3.0e-10)

    anisotropic = BiMaxwellianPressureClosure(
        thermal_coefficients=jnp.asarray([1.2e4, -2.0e3]),
        hot_fraction_coefficients=jnp.asarray([0.4]),
        temperature_ratio=0.25,
        critical_field=0.6,
    )
    b = jnp.linspace(0.35, 1.1, 9)
    s = jnp.full_like(b, 0.3)
    moments = anisotropic.moments(s, b)
    dp_db = jax.grad(lambda field: jnp.sum(anisotropic.parallel_pressure(s, field)))(b)
    np.testing.assert_allclose(
        moments.perpendicular,
        moments.parallel - b * dp_db,
        rtol=3.0e-14,
        atol=3.0e-10,
    )
    assert np.all(np.isfinite(np.asarray(moments.perpendicular)))


def test_tabulated_parallel_pressure_derives_perpendicular_moment() -> None:
    s_nodes = jnp.asarray([0.0, 0.4, 1.0])
    b_nodes = jnp.asarray([0.3, 0.7, 1.2])
    # p_parallel = 10000*(1-s) + 2000*B, exactly bilinear.
    values = 1.0e4 * (1.0 - s_nodes[:, None]) + 2.0e3 * b_nodes[None, :]
    closure = TabulatedPressureClosure(s_nodes, b_nodes, values)
    s = jnp.asarray([0.2, 0.6, 0.9])
    b = jnp.asarray([0.5, 0.8, 1.0])
    moments = closure.moments(s, b)
    np.testing.assert_allclose(moments.parallel, 1.0e4 * (1.0 - s) + 2.0e3 * b)
    np.testing.assert_allclose(moments.perpendicular, 1.0e4 * (1.0 - s), atol=2.0e-10)


def test_closure_coefficients_are_differentiable_leaves() -> None:
    s = jnp.asarray([0.2, 0.7])
    b = jnp.asarray([0.5, 0.9])

    def total_pressure(coefficients):
        closure = BiMaxwellianPressureClosure(
            thermal_coefficients=coefficients,
            hot_fraction_coefficients=jnp.asarray([0.3]),
            temperature_ratio=0.4,
            critical_field=0.6,
        )
        return jnp.sum(closure.moments(s, b).parallel)

    derivative = jax.grad(total_pressure)(jnp.asarray([1.0e4, -2.0e3]))
    assert derivative.shape == (2,)
    assert np.all(np.isfinite(np.asarray(derivative)))
    assert np.linalg.norm(np.asarray(derivative)) > 0.0
