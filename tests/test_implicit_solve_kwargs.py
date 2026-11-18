"""Regression tests for implicit-lane forward-solver controls."""

from __future__ import annotations

import dataclasses
import types
from pathlib import Path

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)

from vmex.core import optimize as opt
from vmex.core.input import VmecInput


DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"
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
    alternate = run({
        "ns_array": [5],
        "ftol_array": [1.0e-6],
        "niter_array": [1000],
        "device": "cpu",
    })
    # The two requested ladders must produce finite, observable residuals.
    # This covers both the implicit callback and its hot-restart path without
    # asserting how the input object stores the controls internally.
    assert np.all(np.isfinite(got.fun))
    assert np.all(np.isfinite(alternate.fun))
    # Before the fix, both calls silently used the deck's [5], 1e-10, 1000
    # ladder and produced the same implicit residual.
    assert not np.isclose(
        np.linalg.norm(np.asarray(alternate.fun, dtype=float)),
        np.linalg.norm(np.asarray(got.fun, dtype=float)),
        rtol=1.0e-7,
        atol=1.0e-10,
    )


def test_implicit_problem_passes_host_roots_to_terms():
    """A term with ``roots`` gets them from the host, as an argument of the jitted rows."""

    class Doubled:
        """Toy ambipolar term: the aspect ratio times its 'root' 2 (3 without roots)."""

        ambipolar = True
        calls = 0

        def roots(self, state, runtime):
            del state, runtime
            Doubled.calls += 1
            return (np.asarray([2.0]),)

        def residuals_state(self, state, runtime, roots=None):
            factor = 3.0 if roots is None else roots[0][0]
            return factor * opt.aspect_ratio(state, runtime)

    inp = VmecInput.from_file(DATA_DIR / "input.solovev")
    inp = inp.change_resolution(mpol=3, ntor=0, ntheta=12, nzeta=4)
    inp = dataclasses.replace(inp, ns_array=np.asarray([5]),
                              ftol_array=np.asarray([1.0e-10]), niter_array=np.asarray([1000]))
    problem = opt.VmecProblem.from_tuples(
        inp, [(Doubled().residuals_state, 0.0, 1.0), (opt.aspect_ratio, 0.0, 1.0)],
        max_mode=1, device="cpu")
    rows = problem.residual(problem.x0)
    jac = problem.residual_jac(problem.x0)
    np.testing.assert_allclose(rows[0], 2.0 * rows[1], rtol=1e-10)
    np.testing.assert_allclose(jac[0], 2.0 * jac[1], rtol=1e-8, atol=1e-12)
    assert Doubled.calls >= 1


def test_probe_workers_split_the_jacobian_columns_without_changing_them(monkeypatch):
    """Column chunks run on threads return the single-program Jacobian."""
    import concurrent.futures

    inp = VmecInput.from_file(DATA_DIR / "input.solovev")
    inp = inp.change_resolution(mpol=3, ntor=0, ntheta=12, nzeta=4)
    inp = dataclasses.replace(inp, ns_array=np.asarray([5]),
                              ftol_array=np.asarray([1.0e-10]), niter_array=np.asarray([1000]))
    pools = []

    class Pool(concurrent.futures.ThreadPoolExecutor):
        def __init__(self, workers):
            pools.append(workers)
            super().__init__(workers)

    monkeypatch.setattr(opt, "concurrent", types.SimpleNamespace(
        futures=types.SimpleNamespace(ThreadPoolExecutor=Pool)))
    jacobians = {}
    for workers in (1, 2):
        problem = opt.VmecProblem.from_tuples(
            inp, [(opt.aspect_ratio, 0.0, 1.0), (opt.mean_iota, 0.0, 1.0)], max_mode=1,
            device="cpu", jacobian_batch_size=1, probe_workers=workers)
        assert problem.x0.size >= 2
        jacobians[workers] = problem.residual_jac(problem.x0)
    assert pools == ([2] if problem.x0.size % 2 == 0 else [])
    np.testing.assert_allclose(jacobians[2], jacobians[1], rtol=1e-10, atol=1e-12)
