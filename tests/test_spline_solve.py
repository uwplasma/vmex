"""B-spline Galerkin radial discretization of the fixed-boundary energy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from vmex import VmecInput  # noqa: E402
from vmex.core import spline_solve as ss  # noqa: E402
from vmex.core.radial_basis import BSplineBasis  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "examples" / "data"


@pytest.fixture(scope="module")
def solovev():
    inp = VmecInput.from_file(str(DATA / "input.solovev"))
    state = ss.initial_spline_state(inp, BSplineBasis.clamped(np.linspace(0.0, 1.0, 3), degree=3))
    plan = ss.make_spline_plan(state, ntheta=18)
    full, info = ss.solve_spline(plan)
    return plan, full, info


def test_structured_hessian_matches_autodiff(solovev):
    plan, full, _ = solovev
    index = jnp.asarray(plan.free)
    expected = jax.jit(jax.hessian(lambda x: ss.spline_energy(plan, full.at[index].set(x))))(full[index])
    actual = jax.jit(ss.spline_hessian)(plan, full)
    assert float(jnp.max(jnp.abs(actual - expected)) / jnp.max(jnp.abs(expected))) < 1e-12


def test_energy_matches_extrapolated_finite_difference_limit(solovev):
    # VMEX on the same deck and angular grid (NTHETA = 18) converges as ns^-2:
    # W(129) = 2027913.32760, W(257) = 2027913.24438 J, so W(inf) = 2027913.2166 J.
    # Two cubic spans (69 unknowns) land within 1e-10 of that limit.
    plan, full, info = solovev
    assert info["iterations"] <= 15 and info["history"][-1][1] < 1e-10
    assert abs(float(ss.spline_energy(plan, full)) / 2027913.2166 - 1.0) < 1e-9


def test_implicit_gradient_matches_central_differences(solovev):
    plan, full, info = solovev
    parameters = ss.boundary_indices(plan)
    m = np.asarray(plan.m)
    pick = int(np.flatnonzero(m == 1)[0])  # RBC(0,1)

    def axis(f):
        return f[0]  # R_00 at s = 0: the magnetic axis

    gradient = ss.implicit_gradient(plan, full, axis, parameters, info["multipliers"])[pick]
    h = 1e-4
    values = [float(axis(ss.solve_spline(plan, full.at[parameters[pick]].add(sign * h))[0]))
              for sign in (1.0, -1.0)]
    assert abs(float(gradient) - (values[0] - values[1]) / (2 * h)) < 1e-6 * abs(float(gradient))
