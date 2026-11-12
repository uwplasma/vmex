"""run_stage: step units, bounds and stagnation stop on a cheap analytic problem."""

import numpy as np

from vmex.core.stages import run_stage


class Rosenbrock:
    """Residuals of the Rosenbrock valley plus a flat third variable."""

    def __init__(self, x0):
        self.x0 = np.asarray(x0, dtype=float)
        self.calls = 0

    def residual(self, x):
        self.calls += 1
        return np.array([10 * (x[1] - x[0] ** 2), 1 - x[0], 0.1 * x[2]])

    def residual_jac(self, x):
        return np.array([[-20 * x[0], 10, 0], [-1, 0, 0], [0, 0, 0.1]])


def test_stage_reaches_the_minimum_in_step_units():
    problem = Rosenbrock([-1.2, 1.0, 3.0])
    result = run_stage(problem, max_nfev=80, step=[0.1, 0.1, 0.1], ftol=1e-14, patience=100)
    assert result.cost < 1e-12
    np.testing.assert_allclose(result.x, [1.0, 1.0, 0.0], atol=1e-5)


def test_change_bounds_the_stage_around_the_seed():
    problem = Rosenbrock([-1.2, 1.0, 3.0])
    result = run_stage(problem, max_nfev=20, step=0.1, change=2.0)
    assert np.all(np.abs(result.x - problem.x0) <= 0.2 + 1e-12)


def test_max_nfev_is_a_ceiling():
    result = run_stage(Rosenbrock([-1.2, 1.0, 3.0]), max_nfev=3, step=0.1)
    assert result.nfev == 3


def test_stagnation_stops_a_converged_stage_before_the_budget():
    problem = Rosenbrock([1.0, 1.0, 0.0])
    problem.residual = lambda x: np.array([1.0 + 1e-3 * np.sum(x**2), 0.0, 0.0])
    problem.residual_jac = lambda x: np.array([[2e-3 * x[0], 2e-3 * x[1], 2e-3 * x[2]]] + [[0, 0, 0]] * 2)
    result = run_stage(problem, max_nfev=50, step=1.0, patience=2, ftol=1e-14, xtol=1e-14)
    assert result.nfev < 50


def test_callback_sees_the_problems_own_coordinates():
    problem = Rosenbrock([-1.2, 1.0, 3.0])
    seen = []
    result = run_stage(problem, max_nfev=4, step=[0.1, 0.2, 0.3], callback=seen.append)
    assert seen and all(np.shape(r.x) == (3,) for r in seen)
    np.testing.assert_allclose(seen[-1].x, result.x)
    assert np.all(np.abs(seen[0].x - problem.x0) < 5.0)
