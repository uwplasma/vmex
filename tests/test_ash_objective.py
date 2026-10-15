"""VMEX composition tests; manufactured coefficients are not design results."""
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
