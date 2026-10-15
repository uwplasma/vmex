"""VMEX composition tests; manufactured coefficients are not design results."""
import sys
from types import ModuleType, SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vmex.core.ash import HeliumAshObjective


def response(diffusion, _runtime):
    edges = jnp.linspace(0.0, 1.0, 41)
    return dict(
        radial_edges_m=edges, cell_volume_m3=jnp.diff(edges),
        volume_derivative_m2=jnp.ones_like(edges),
        diffusion_m2_s=jnp.ones_like(edges) * diffusion,
        convection_m_s=jnp.zeros_like(edges),
        source_m3_s=jnp.ones(40), boundary_kappa_m_s=jnp.inf,
    )


@pytest.mark.parametrize("reference", [0, -1, np.inf, np.nan])
def test_reference_must_be_fixed_positive_finite(reference):
    with pytest.raises(ValueError, match="reference_occupation_s"):
        HeliumAshObjective(response, reference, "coefficient-only")


def test_scope_and_callback_are_explicit():
    with pytest.raises(ValueError, match="gradient_scope"):
        HeliumAshObjective(response, 1.0, "")
    with pytest.raises(TypeError, match="callable"):
        HeliumAshObjective(None, 1.0, "coefficient-only")


@pytest.fixture
def supplied_radial_result(monkeypatch):
    """Test the adapter contract without substituting for the radial solver."""
    package = ModuleType("essos")
    package.__path__ = []
    module = ModuleType("essos.ash")
    monkeypatch.setitem(sys.modules, "essos", package)
    monkeypatch.setitem(sys.modules, "essos.ash", module)

    def supply(occupation, valid=True):
        result = SimpleNamespace(
            mean_occupation_s=jnp.asarray(occupation), valid=jnp.asarray(valid),
        )
        calls = []

        def evaluate(**kwargs):
            calls.append(kwargs)
            return result

        module.evaluate_helium_residence = evaluate
        return result, calls

    return supply


def test_adapter_forwards_callback_arguments_and_preserves_result(supplied_radial_result):
    result, calls = supplied_radial_result(3.0)
    state, runtime, coefficients = object(), object(), {"supplied_coefficients": object()}
    callback_calls = []

    def transport(given_state, given_runtime):
        callback_calls.append((given_state, given_runtime))
        return coefficients

    objective = HeliumAshObjective(transport, 2.0, "supplied response contract")
    assert objective.evaluate_state(state, runtime) is result
    assert callback_calls == [(state, runtime)]
    assert calls == [coefficients]


def test_adapter_normalizes_scalar_and_exposes_lazy_objective(supplied_radial_result):
    supplied_radial_result(3.0)
    from vmex import optimize as opt

    assert opt.HeliumAshObjective is HeliumAshObjective
    objective = opt.HeliumAshObjective(lambda state, runtime: {}, 2.0, "supplied scalar")
    np.testing.assert_allclose(objective.total_state(None, None), 1.5)
    rows = objective.residuals_state(None, None)
    assert rows.shape == (1,)
    np.testing.assert_allclose(rows**2, [1.5])
    np.testing.assert_array_equal(objective(None, None), rows)


@pytest.mark.parametrize("occupation,valid", [
    (3.0, False), (0.0, True), (-1.0, True), (np.nan, True), (np.inf, True),
])
def test_adapter_rejects_invalid_supplied_occupation(supplied_radial_result, occupation, valid):
    supplied_radial_result(occupation, valid)
    objective = HeliumAshObjective(lambda state, runtime: {}, 2.0, "supplied scalar")
    assert np.isnan(objective.total_state(None, None))
    assert np.all(np.isnan(objective.residuals_state(None, None)))


def test_adapter_rejects_nonscalar_occupation(supplied_radial_result):
    supplied_radial_result([1.0, 2.0])
    objective = HeliumAshObjective(lambda state, runtime: {}, 2.0, "supplied vector")
    with pytest.raises(ValueError, match="mean_occupation_s must be scalar"):
        objective.total_state(None, None)


def test_missing_optional_dependency_has_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "essos", None)
    monkeypatch.setitem(sys.modules, "essos.ash", None)
    objective = HeliumAshObjective(lambda state, runtime: {}, 2.0, "dependency contract")
    with pytest.raises(ImportError, match="requires ESSOS with essos.ash") as error:
        objective.evaluate_state(None, None)
    assert isinstance(error.value.__cause__, ImportError)


def test_positive_cost_and_vmex_residual_composition():
    pytest.importorskip("essos.ash")
    from vmex import optimize as opt

    objective = HeliumAshObjective(response, 2.0, "diffusion only; fixed geometry/source/Er")
    result = objective.evaluate_state(jnp.asarray(0.5), None)
    assert bool(result.valid)
    # Reflecting/absorbing slab, uniform births: mean residence = L^2/(3D).
    np.testing.assert_allclose(result.mean_residence_s, 2 / 3, rtol=0.001)
    total = objective.total_state(jnp.asarray(0.5), None)
    rows = opt.residuals_from_tuples(jnp.asarray(0.5), None, [(objective, 0.0, 3.0)])
    np.testing.assert_allclose(jnp.vdot(rows, rows), 3.0 * total, rtol=1e-12)
    assert np.asarray(rows).shape == (1,)


def test_coefficient_gradient_and_jit_match_finite_difference_ladder():
    pytest.importorskip("essos.ash")
    objective = HeliumAshObjective(response, 1.0, "diffusion only")
    def cost(diffusion):
        return objective.total_state(diffusion, None)
    value, derivative = jax.jit(jax.value_and_grad(cost))(jnp.asarray(0.7))
    np.testing.assert_allclose(derivative, -value / 0.7, rtol=1e-10)
    for step in (1e-3, 1e-4, 1e-5):
        fd = (cost(0.7 + step) - cost(0.7 - step)) / (2 * step)
        np.testing.assert_allclose(derivative, fd, rtol=3e-6)


@pytest.mark.parametrize("diffusion", [-0.1, np.nan, np.inf])
def test_invalid_physics_never_becomes_favorable_cost(diffusion):
    pytest.importorskip("essos.ash")
    objective = HeliumAshObjective(response, 1.0, "diffusion only")
    assert not np.isfinite(objective.total_state(jnp.asarray(diffusion), None))
    assert not np.all(np.isfinite(objective(jnp.asarray(diffusion), None)))


def test_closed_boundary_does_not_return_finite_objective():
    pytest.importorskip("essos.ash")
    def closed(state, runtime):
        return {**response(state, runtime), "boundary_kappa_m_s": 0.0}
    objective = HeliumAshObjective(closed, 1.0, "diffusion only")
    assert not np.isfinite(objective.total_state(jnp.asarray(1.0), None))
