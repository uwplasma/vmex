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
    solved, result = solve_phiedge(*case, target, metric=metric, rtol=1.0e-4)
    assert abs(_phiedge_metric(metric)(solved, result) - target) <= 1.0e-4 * target


def test_solve_phiedge_host_loop_on_a_stub_solver(case, monkeypatch):
    """Failed trials halve back, a bracket bisects, and bad input raises typed errors."""
    import vmex.core.freeboundary as fb

    trials = []

    def stub(inp, **kwargs):
        trials.append(inp.phiedge)
        if inp.phiedge > 1.7:
            raise VmecConvergenceError("stub divergence")
        return dataclasses.make_dataclass("R", ["state", "iterations"])(None, 10)

    monkeypatch.setattr(fb, "solve_free_boundary", stub)
    metric = lambda i, r: 1.0 + np.cbrt(i.phiedge - 1.5)  # noqa: E731
    solved, _ = solve_phiedge(case[0], None, 1.0, metric=metric, phiedge0=1.0, rtol=1.0e-3, max_iter=40)
    assert abs(solved.phiedge - 1.5) < 1.0e-6 and max(trials) > 1.7
    with pytest.raises(VmecConvergenceError, match="stub"):  # no converged state to fall back on
        solve_phiedge(case[0], None, 1.0, phiedge0=2.0)
    with pytest.raises(VmecConvergenceError, match="PHIEDGE solve did not reach"):
        solve_phiedge(case[0], None, 5.0, metric=metric, phiedge0=1.0, max_iter=3)
    with pytest.raises(ValueError, match="metric must be"):
        solve_phiedge(*case, 1.0, metric="aspect")
    with pytest.raises(ValueError, match="nonzero"):
        solve_phiedge(*case, 1.0, phiedge0=0.0)


def test_phiedge_root_gradient_is_the_implicit_function_theorem(case):
    """phiedge_root: dphi/dp = -(dg/dp)/(dg/dphi) on an analytic residual."""
    import jax

    from vmex.core import implicit as im
    from vmex.core.freeboundary_implicit import phiedge_root

    def residual(params, field):  # root phiedge = curtor / a
        return params.phiedge * field["a"] - params.curtor

    params = dataclasses.replace(im.params_from_input(case[0]), phiedge=0.75, curtor=1.5)
    grad_p, grad_f = jax.grad(phiedge_root, argnums=(1, 2))(residual, params, {"a": 2.0})
    np.testing.assert_allclose([grad_p.curtor, grad_p.phiedge, grad_f["a"]], [0.5, 0.0, -0.375])
    assert phiedge_root(residual, params, {"a": 2.0}) == 0.75


@pytest.mark.full  # nightly: compiles the coupled adjoint (minutes on a loaded CPU)
def test_phiedge_root_extcur_gradient_matches_resolve_finite_difference(case):
    """d(PHIEDGE at a fixed edge observable)/d(extcur): one adjoint vs re-solved roots.

    The observable is the sum of the internal edge ``R_cos`` row, pinned at
    its value for the deck's PHIEDGE; the perturbed roots come from
    fixed-slope Newton steps on the Newton-anchored implicit forward.
    """
    import jax
    import jax.numpy as jnp

    from vmex.core import implicit as im
    from vmex.core.freeboundary_implicit import (
        make_free_boundary_config, phiedge_root, solve_free_boundary_implicit)

    inp, field = case
    inp = dataclasses.replace(inp, niter_array=[20000], ftol_array=[1.0e-13])
    cfg = make_free_boundary_config(
        inp, field, ns=16, ftol=1.0e-13, max_iterations=20000,
        adjoint_tol=1.0e-10, adjoint_maxiter=200)
    params = im.params_from_input(inp)

    def edge(params, field):
        return jnp.sum(solve_free_boundary_implicit(params, field, cfg).R_cos[-1])

    target = float(edge(params, field))

    def residual(params, field):
        return edge(params, field) - target

    derivative = float(jax.grad(lambda f: phiedge_root(residual, params, f))(field).extcur[0])
    step, roots = 1.0e-4, []
    for h in (step, -step):
        trial, shifted = params, dataclasses.replace(field, extcur=field.extcur + h)
        for _ in range(4):  # dg/dphiedge = 0.0551 here; converges to 1e-12 Wb
            trial = dataclasses.replace(
                trial, phiedge=trial.phiedge - float(residual(trial, shifted)) / 0.0551)
        roots.append(float(trial.phiedge))
    np.testing.assert_allclose(derivative, (roots[0] - roots[1]) / (2.0 * step), rtol=1.0e-4)
