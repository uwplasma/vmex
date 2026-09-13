"""Regression tests for implicit-lane forward-solver controls."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)

from vmex.core import optimize as opt
from vmex.core.input import VmecInput


DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"


@pytest.mark.usefixtures("_module_jit_enabled")
def test_implicit_least_squares_honors_multigrid_solve_kwargs():
    """Implicit and FD lanes must see the same requested radial ladder.

    Before the regression fix, ``jac=None`` honored these controls while
    ``jac='implicit'`` silently used the deck's ``NS_ARRAY``/``FTOL_ARRAY``/
    ``NITER_ARRAY``.  The resulting Jacobians were compared at different
    equilibria and had different residual-row shapes.
    """
    inp = VmecInput.from_file(DATA_DIR / "input.solovev")
    inp = inp.change_resolution(mpol=3, ntor=0, ntheta=12, nzeta=4)
    inp = dataclasses.replace(
        inp,
        ns_array=np.asarray([5]),
        ftol_array=np.asarray([1.0e-10]),
        niter_array=np.asarray([1000]),
    )
    qh = opt.QuasisymmetryRatioResidual([0.5], 1, -1)
    strict = {
        "ns_array": [7],
        "ftol_array": [1.0e-11],
        "niter_array": [1200],
        "device": "cpu",
    }

    def run(solve_kwargs):
        return opt.least_squares(
            [(qh, 0.0, 1.0), (opt.aspect_ratio, 4.0, 1.0)],
            inp,
            max_mode=1,
            jac="implicit",
            hot_restart=True,
            warm_start=None,
            solve_kwargs=solve_kwargs,
            max_nfev=1,
        )

    got = run(strict)
    alternate_controls = {
        "ns_array": [5],
        "ftol_array": [1.0e-6],
        "niter_array": [5],
        "device": "cpu",
    }
    alternate_problem = opt.VmecProblem.from_tuples(
        inp, [(qh, 0.0, 1.0), (opt.aspect_ratio, 4.0, 1.0)],
        max_mode=1, hot_restart=True, warm_start=None,
        solve_kwargs=alternate_controls,
    )
    # Query the value-only graph: an intentionally exhausted forward solve
    # cannot supply an ordinary implicit Jacobian under the primal contract.
    alternate_value = np.asarray(
        alternate_problem.jax_residual(alternate_problem.x0))
    # Host optimizers receive the finite penalty pair without constructing
    # an equilibrium derivative; the actual primal remains explicitly rejected.
    rejected_residual, rejected_jacobian = alternate_problem.residual_and_jac(alternate_problem.x0)
    assert np.all(np.isfinite(rejected_residual))
    assert np.all(np.isfinite(rejected_jacobian))
    rejected = alternate_problem.evaluate(alternate_problem.x0)
    assert rejected.status == "under_converged"
    assert not rejected.diagnostics["derivative_certified"]
    # The two requested ladders must produce finite, observable residuals.
    # This covers both the implicit callback and its hot-restart path without
    # asserting how the input object stores the controls internally.
    assert np.all(np.isfinite(got.fun))
    assert np.all(np.isfinite(alternate_value))
    # Before the fix, both calls silently used the deck's [5], 1e-10, 1000
    # ladder and produced the same implicit residual.
    assert not np.isclose(
        np.linalg.norm(alternate_value),
        np.linalg.norm(np.asarray(got.fun, dtype=float)),
        rtol=1.0e-7,
        atol=1.0e-10,
    )
