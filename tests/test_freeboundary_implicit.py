"""Coupled solved-LCFS implicit-sensitivity contracts."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from vmec_jax.core.freeboundary_implicit import CoupledFreeBoundaryProblem
from vmec_jax.core.solver import SpectralState


def _state(value):
    arrays = [jnp.asarray(value * np.arange(1, 5).reshape(2, 2)) for _ in range(6)]
    return SpectralState(*arrays)


def test_forward_implicit_sensitivity_solves_linear_contract():
    """The forward IFT solve returns ``dx/dp`` without unrolling a solver."""
    coefficients = _state(1.0)
    current = jnp.asarray([2.0])

    class LinearProblem:
        reference_state = jax.tree.map(lambda x: current[0] * x, coefficients)
        external_field = SimpleNamespace(extcur=current)

        @staticmethod
        def residual(state, extcur):
            return jax.tree.map(lambda x, c: x - extcur[0] * c, state, coefficients)

    result = CoupledFreeBoundaryProblem.extcur_sensitivity(
        LinearProblem(),
        jnp.asarray([3.0]),
        rtol=1.0e-12,
        projector=lambda state: state,
    )
    assert result.converged
    assert result.residual_norm < 1.0e-12
    expected = jax.tree.map(lambda x: 3.0 * x, coefficients)
    for actual, target in zip(jax.tree.leaves(result.state), jax.tree.leaves(expected)):
        np.testing.assert_allclose(actual, target, rtol=1.0e-12, atol=1.0e-12)
