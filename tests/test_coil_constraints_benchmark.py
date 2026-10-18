"""Coil chart and ESSOS-backed coil rows of the coil-constraint single-stage example."""

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

essos_coils = pytest.importorskip("essos.coils")
EXAMPLE = (Path(__file__).resolve().parents[1] / "examples" / "optimization"
           / "single_stage_free_boundary_optimization_coil_constraints.py")
SPEC = importlib.util.spec_from_file_location("coil_constraints_example", EXAMPLE)
cc = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cc)


@pytest.fixture
def coils():
    jax.config.update("jax_enable_x64", True)
    coefficients = np.random.default_rng(747).normal(size=(2, 3, 5))
    curves = essos_coils.Curves(jnp.asarray(coefficients), 24, 2, True, scaling_factor=0.7, scale_fixed=2.5)
    return essos_coils.Coils(curves, jnp.array([2.4e5, -3.1e5]), currents_scale=8e4), coefficients


def test_chart_keeps_physical_units_and_selects_currents(coils):
    coils, coefficients = coils
    chart = cc.CoilChart(coils, current_dofs=(1,), scales=np.ones(31))
    assert chart.x0.size == len(chart.dof_names) == 31
    np.testing.assert_allclose(chart.coefficients, coefficients, rtol=2e-15, atol=2e-15)
    x = np.random.default_rng(38).normal(size=31) * 1e-3
    np.testing.assert_allclose(chart.base_currents_at(x), [2.4e5, -3.1e5 * (1 + x[0])])
    moved = chart.coils_from_x(x)
    np.testing.assert_allclose(moved.dofs_curves, coefficients + x[1:].reshape(2, 3, 5), rtol=2e-15, atol=2e-15)
    restored = chart.coils_from_x(chart.x0)
    np.testing.assert_allclose(restored.gamma, coils.gamma, rtol=2e-14, atol=2e-14)
    np.testing.assert_array_equal(restored.currents, coils.currents)


def test_chart_plasma_coordinates_come_first(coils):
    coils, coefficients = coils
    nominal = np.array([2.0, -4.0, 1.5e4])  # two AC_AUX_F values, then CURTOR [A]
    chart = cc.CoilChart(coils, current_dofs=(1,), phiedge=0.08, phiedge_scale=0.02, plasma_current=nominal,
                         plasma_current_spline=True, plasma_current_scale=0.1)
    assert chart.size == 35 and chart.dof_names[:4] == ("phiedge/nominal", "plasma_current[0]/unit",
                                                         "plasma_current[1]/unit", "curtor/nominal")
    assert chart.scales[0] == 0.02 and np.all(chart.scales[1:4] == 0.1)
    x = np.random.default_rng(40).normal(size=chart.size) * 0.01
    np.testing.assert_allclose(chart.phiedge_at(x), 0.08 * (1 + x[0]))

    @dataclass
    class Params:
        phiedge: float
        curtor: float
        ac: object
        ac_aux_f: object

    moved = chart.plasma_from_parameters(Params(0.0, 0.0, jnp.zeros(3), jnp.full(3, 7.0)), x)
    np.testing.assert_allclose(moved.phiedge, 0.08 * (1 + x[0]))
    np.testing.assert_allclose(moved.ac_aux_f, [2 + 4 * x[1], -4 + 4 * x[2], 7.0])
    np.testing.assert_allclose(moved.curtor, 1.5e4 * (1 + x[3]))
    np.testing.assert_allclose(chart.base_currents_at(x), [2.4e5, -3.1e5 * (1 + x[4])])
    np.testing.assert_allclose(chart.coils_from_x(x).dofs_curves, coefficients + x[5:].reshape(2, 3, 5),
                               rtol=2e-15, atol=2e-15)
    deck = SimpleNamespace(phiedge=0.08, ncurr=1, ac_aux_f=[2.0, -4.0, 9.0], ac=[], curtor=1.5e4)
    chart.check_input(deck)
    with pytest.raises(ValueError, match="nominal plasma current"):
        chart.check_input(SimpleNamespace(**{**vars(deck), "curtor": 1.0e4}))
    with pytest.raises(ValueError, match="does not vary PHIEDGE"):
        cc.CoilChart(coils, current_dofs=()).phiedge_at(np.zeros(30))
    with pytest.raises(ValueError, match="nonzero CURTOR"):
        cc.CoilChart(coils, current_dofs=(), plasma_current=[1.0, 0.0])
    assert cc.CoilChart(coils, current_dofs=()).plasma_from_parameters is None


def test_field_derivatives_under_jit_and_nested_transforms(coils):
    chart = cc.CoilChart(coils[0], current_dofs=(1,), scales=np.full(31, 1e-2))
    scales = jnp.asarray(chart.scales)

    def field(u):
        points = jnp.array([0.8, 1.1, 1.4]), jnp.array([0.1, 0.3, 0.7]), jnp.array([0.2, -0.1, 0.3])
        return jnp.stack(chart(u * scales).b_cyl(*points)).ravel()

    compiled = jax.jit(field)
    derivative = jax.jit(jax.jacfwd(compiled))
    for u in (np.zeros(31), np.random.default_rng(748).normal(size=31)):
        jacobian = derivative(u)
        for index in (0, 1, 30):  # the current and two Fourier directions
            delta = np.eye(31)[index] * 1e-4
            fd = (compiled(u + delta) - compiled(u - delta)) / 2e-4
            np.testing.assert_allclose(jacobian[:, index], fd, rtol=3e-6, atol=1e-10)
        weights = jnp.linspace(0.2, 1.0, 9)
        reverse = jax.jit(jax.grad(lambda v: jnp.vdot(compiled(v), weights)))(u)
        np.testing.assert_allclose(reverse, weights @ jacobian, rtol=2e-12, atol=1e-12)


def test_coil_metrics_of_a_circle_and_differentiable_rows():
    jax.config.update("jax_enable_x64", True)
    radius, order, n_coils = 0.3, cc.COIL_ORDER, 3
    raw = np.zeros((n_coils, 3, 2 * order + 1))
    raw[:, 0, 0] = 1.0 + np.arange(n_coils)  # centres 1 m apart, no overlap
    raw[:, 0, 2], raw[:, 2, 1] = radius, radius   # x = cos, z = sin: a circle of radius 0.3 m
    coils = essos_coils.Coils(essos_coils.Curves(jnp.asarray(raw), 64, 1, False), jnp.ones(n_coils))
    metrics = cc.coil_metrics(coils)
    np.testing.assert_allclose(metrics["length"], 2 * np.pi * radius, rtol=1e-13)
    np.testing.assert_allclose(metrics["peak"], 1 / radius, rtol=1e-12)
    np.testing.assert_allclose(metrics["msc"], 1 / radius**2, rtol=1e-12)
    np.testing.assert_allclose(metrics["coil_distance"], 1 - 2 * radius, rtol=1e-12)
    # Nonadjacent segments of the polygon are at least one chord apart.
    np.testing.assert_allclose(metrics["self_distance"], 2 * radius * np.sin(np.pi / cc.DISTANCE_POINTS), rtol=1e-10)
    jacobian = jax.jacrev(lambda r: cc.coil_inequalities(
        essos_coils.Coils(essos_coils.Curves(r, 64, 1, False), jnp.ones(n_coils))))(jnp.asarray(raw))
    assert np.all(np.isfinite(jacobian))
