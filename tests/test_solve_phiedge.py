"""PHIEDGE root solve for a free-boundary geometric target.

Uses the portable axisymmetric DIII-D field of ``test_lasym_free_case`` on a
16-surface grid, where one warm-started solve takes about a second.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")

from tests.test_lasym_free_case import lasym_free_field  # noqa: E402

from vmex.core.errors import VmecConvergenceError  # noqa: E402
from vmex.core.freeboundary import _phiedge_metric, solve_phiedge  # noqa: E402
from vmex.core.input import VmecInput  # noqa: E402

pytestmark = pytest.mark.usefixtures("_module_jit_enabled")  # free-boundary solves: run jitted

DATA = Path(__file__).resolve().parents[1] / "examples" / "data"


@pytest.fixture(scope="module")
def case():
    """Stellarator-symmetric DIII-D deck and its compressed vacuum field."""
    inp = VmecInput.from_file(DATA / "input.DIII-D_lasym_false")
    inp = dataclasses.replace(
        inp, extcur=np.ones(1), mgrid_file="compressed_d3d", ns_array=[16],
        niter_array=[3000], ftol_array=[1.0e-8])
    return inp, lasym_free_field()


@pytest.mark.parametrize("metric,target", [("volume", 20.3), ("r_outboard", 2.254)])
def test_solve_phiedge_lands_on_target(case, metric, target):
    """The returned deck's equilibrium meets the target within rtol."""
    inp, field = case
    solved, result = solve_phiedge(inp, field, target, metric=metric, rtol=1.0e-4)
    value = _phiedge_metric(metric)(solved, result)
    assert bool(result.converged)
    assert abs(value - target) <= 1.0e-4 * target
    # Volume grows with |PHIEDGE|; the outboard edge recedes in this field.
    assert -4.03 < solved.phiedge < -3.29


def test_solve_phiedge_unreachable_target_raises_typed_error(case):
    """A target no nearby equilibrium reaches ends in a typed error, not a loop."""
    inp, field = case
    with pytest.raises(VmecConvergenceError, match="PHIEDGE solve did not reach"):
        solve_phiedge(inp, field, 1.0e3, metric="volume", max_iter=3)
    solved, _ = solve_phiedge(inp, field, 0.0, metric=lambda i, r: 0.0)  # callable metric
    assert solved.phiedge == inp.phiedge
    with pytest.raises(ValueError, match="metric must be"):
        solve_phiedge(inp, field, 1.0, metric="aspect")


@pytest.mark.full  # nightly: compiles the coupled adjoint (minutes on a loaded CPU)
def test_free_boundary_phiedge_derivative_matches_resolve_finite_difference(case):
    """d(LCFS R at theta = phi = 0)/d(PHIEDGE): implicit adjoint vs re-solves.

    Measured: adjoint 5.51187e-2, central difference 5.51187e-2 (h = 1e-4).
    """
    import jax
    import jax.numpy as jnp

    from vmex.core import implicit as im
    from vmex.core.freeboundary_implicit import (
        make_free_boundary_config, solve_free_boundary_implicit)

    inp, field = case
    params = im.params_from_input(inp)
    cfg = make_free_boundary_config(
        inp, field, ns=16, ftol=1.0e-9, max_iterations=3000,
        adjoint_tol=1.0e-9, adjoint_maxiter=200)

    def r_outboard(phiedge):
        state = solve_free_boundary_implicit(
            dataclasses.replace(params, phiedge=phiedge), field, cfg)
        return jnp.sum(state.R_cos[-1])

    phiedge, step = jnp.asarray(params.phiedge), 1.0e-4
    derivative = float(jax.grad(r_outboard)(phiedge))
    finite_difference = float(r_outboard(phiedge + step) - r_outboard(phiedge - step)) / (2.0 * step)
    np.testing.assert_allclose(derivative, finite_difference, rtol=1.0e-4)
