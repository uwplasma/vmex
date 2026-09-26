"""ESSOS interoperability: physical units, serialization and traced derivatives.

Run unchanged with upstream ESSOS main or the research branch on PYTHONPATH.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vmex.core.coil_parameters import CoilParameters

essos = pytest.importorskip("essos.coils")


@pytest.fixture
def physical_coils():
    jax.config.update("jax_enable_x64", True)
    coefficients = np.random.default_rng(747).normal(size=(2, 3, 5))
    curves = essos.Curves(jnp.asarray(coefficients), 24, 2, True, scaling_factor=0.7, scale_fixed=2.5)
    currents = jnp.array([2.4e5, -3.1e5])
    return essos.Coils(curves, currents, currents_scale=8e4), coefficients, currents


def test_import_preserves_physical_units_and_symmetry(physical_coils):
    coils, coefficients, currents = physical_coils
    chart = CoilParameters.from_coils(coils, current_dofs=(1,), max_coil_mode=1)
    np.testing.assert_allclose(chart.coefficients, coefficients, rtol=2e-15, atol=2e-15)
    np.testing.assert_array_equal(chart.currents, currents)
    restored = chart.coils_from_x(chart.x0)
    np.testing.assert_allclose(restored.gamma, coils.gamma, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(restored.gamma_dash, coils.gamma_dash, rtol=2e-14, atol=2e-14)
    np.testing.assert_array_equal(restored.currents, coils.currents)
    assert (chart.nfp, chart.stellsym, chart.n_segments, chart.size) == (2, True, 24, 19)


def test_export_reload_keeps_physical_currents_and_geometry(physical_coils, tmp_path):
    coils, _, _ = physical_coils
    chart = CoilParameters.from_coils(coils, current_dofs=(1,))
    x = np.random.default_rng(483).normal(size=chart.size) * chart.scales
    exported = chart.coils_from_x(x)
    path = tmp_path / "coils.json"
    exported.to_json(str(path))
    reloaded = essos.Coils.from_json(str(path))
    rebuilt = CoilParameters.from_coils(reloaded, current_dofs=(1,))
    np.testing.assert_allclose(rebuilt.coefficients, chart.curve_dofs_at(x), rtol=2e-15, atol=2e-15)
    np.testing.assert_array_equal(rebuilt.currents, chart.base_currents_at(x))
    np.testing.assert_allclose(reloaded.gamma, exported.gamma, rtol=2e-14, atol=2e-14)


def test_field_derivatives_under_jit_and_nested_transforms(physical_coils):
    coils, _, _ = physical_coils
    chart = CoilParameters.from_coils(coils, current_dofs=(1,))
    scales = jnp.asarray(chart.scales)

    def field(u):
        return jnp.stack(
            chart(u * scales).b_cyl(jnp.array([0.8, 1.1, 1.4]), jnp.array([0.1, 0.3, 0.7]), jnp.array([0.2, -0.1, 0.3]))
        ).ravel()

    compiled = jax.jit(field)
    derivative = jax.jit(jax.jacfwd(compiled))
    rng = np.random.default_rng(748)
    for u in (np.zeros(chart.size), rng.normal(size=chart.size)):
        jacobian = derivative(u)
        assert np.isfinite(jacobian).all()
        # Exercise current and Fourier directions independently, at two states.
        for index in (0, 1, chart.size - 1):
            delta = np.eye(chart.size)[index] * 1e-4
            fd = (compiled(u + delta) - compiled(u - delta)) / 2e-4
            np.testing.assert_allclose(jacobian[:, index], fd, rtol=3e-6, atol=1e-10)
        weights = jnp.linspace(0.2, 1.0, 9)
        reverse = jax.jit(jax.grad(lambda v: jnp.vdot(compiled(v), weights)))(u)
        np.testing.assert_allclose(reverse, weights @ jacobian, rtol=2e-12, atol=1e-12)
