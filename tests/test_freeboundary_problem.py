"""Scalar free-boundary coil optimization: ownership, derivatives, reuse and polishing.

Analytic roots exercise the real API, dense/seed-LU kernels and optimizers;
they do not qualify a plasma equilibrium.
"""

from dataclasses import dataclass, replace
from types import SimpleNamespace
from types import SimpleNamespace as NS
from unittest.mock import Mock
import weakref

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np
import pytest

from vmex import optimize as opt
from vmex.core import freeboundary, freeboundary_implicit as fbi, implicit as im
from vmex.core import freeboundary_problem as api
from vmex.core.errors import TrialRejected, VmecError
from vmex.core.freeboundary_problem import CoilParameters, FreeBoundaryProblem

pytestmark = pytest.mark.usefixtures("_module_jit_enabled")


def test_public_names_share_the_core_classes():
    assert opt.TrialRejected is TrialRejected
    assert opt.FreeBoundaryProblem is FreeBoundaryProblem and opt.CoilParameters is CoilParameters


@dataclass(frozen=True, eq=False)
class Root:
    parameters: np.ndarray
    state: object
    rcon0: object = None
    zcon0: object = None
    dof_mask: object = None
    root_residual_norm: float = 0.0
    result: object = None


@dataclass(frozen=True, eq=False)
class Config:
    solver: object
    params: object
    _anchor: Root
    parameter_scales: object
    continuation_step: float = 0.1
    max_continuation_steps: int = 64


@pytest.fixture
def scalar(monkeypatch):
    """A joint state/coil loss on an analytic root: explicit plus implicit derivatives."""
    jax.config.update("jax_enable_x64", True)
    size = 5
    matrix = np.eye(size) + 0.03 * np.ones((size, size))
    base = np.array([0.2, 5.0, 0.7, -0.3, 0.8])

    def state(x):
        return jnp.asarray(base) + jnp.asarray(matrix) @ x + 0.1 * x * x

    def derivative(x):
        return matrix + 0.2 * np.diag(np.asarray(x))

    class Chart:
        x0 = np.zeros(size)
        scales = np.ones(size)
        dof_names = tuple(f"coil[{i}]" for i in range(size))

        def __call__(self, x):
            return x

        def coils_from_x(self, x):
            return jnp.asarray(x)

    chart = Chart()
    inp = SimpleNamespace(lfreeb=True)
    solver = SimpleNamespace(
        implicit=SimpleNamespace(inp=inp, ftol=1e-11, max_iterations=10),
        resolution=None, edge_force_tolerance=1e-11, field_from_parameters=chart,
        include_edge_in_convergence=True, adjoint_solver="forward_dense_jax", adjoint_fail="error",
    )
    cfg = Config(solver, None, Root(chart.x0.copy(), state(chart.x0)), chart.scales)
    stats = dict(rhs=[], solves=0, closed=0, fail=False, seeds=[], factors=[],
                 matrixfree_fail=False, dense_fail=False, predictions=[])
    monkeypatch.setattr(api.im, "runtime_from_params", lambda *a: None)

    def pullback(record, config, rhs, *, diagnostics=None, preconditioner=None):
        stats["rhs"].append(rhs.shape[0])
        if preconditioner is not None:
            assert not preconditioner.closed and preconditioner.config is config
            if stats["matrixfree_fail"]:
                raise api.AdjointSolveError("matrix-free failed")
        elif stats["dense_fail"]:
            raise api.AdjointSolveError("dense failed")
        jac = np.asarray(rhs) @ derivative(record.parameters)

        class Linearization:
            field_jacobian = jac
            closed = False

            def offload_factors(self):
                pass

            def preconditioner(self, **kwargs):
                class Seed:
                    closed = False

                    def close(self):
                        self.closed = True
                seed = Seed()
                seed.config = config
                stats["seeds"].append(seed)
                return seed

            def tangent(self, current, current_config, delta, **kwargs):
                assert current is record and current_config is config and not self.closed
                return jnp.asarray(derivative(record.parameters) @ delta)

            def close(self):
                stats["closed"] += 1
                self.closed = True

        root = Linearization()
        stats["factors"].append(root)
        return root

    def correct(inp, *, external_field, **kwargs):
        stats["solves"] += 1
        stats["predictions"].append(kwargs)
        return SimpleNamespace(result=SimpleNamespace(state=state(external_field), converged=not stats["fail"]),
                               rcon0=None, zcon0=None)

    def certify(config, point, value, **kwargs):
        np.testing.assert_allclose(value, state(point), rtol=0, atol=0)
        return Root(np.asarray(point).copy(), value)

    monkeypatch.setattr(api, "_pullback", pullback)
    monkeypatch.setattr(api, "_certify", certify)
    monkeypatch.setattr(freeboundary, "_solve_free_boundary_stage", correct)

    def loss(s, rt, coils):
        return 0.5 * jnp.sum(s * s) + jnp.sum(coils * s) + 3 * jnp.sum(coils * coils)

    problem = FreeBoundaryProblem(inp, chart, cfg, loss=loss,
                                  quantities=(lambda s, rt: s[0], lambda s, rt: s[1]))
    yield problem, stats, state, derivative
    problem.close()


def build(scalar, loss, **kwargs):
    """Another problem sharing the fixture's analytic root, chart and patches."""
    p = scalar[0]
    return FreeBoundaryProblem(p.inp, p.parameterization, p.cfg, loss=loss, **kwargs)


def test_total_gradient_constraints_and_independent_differences(scalar):
    p, stats, state, derivative = scalar
    x = np.full(5, .002)
    value, gradient = p.value_and_grad(x)
    expected = np.asarray(state(x))
    np.testing.assert_allclose(value, .5*expected@expected + x@expected + 3*x@x)
    np.testing.assert_allclose(gradient, (expected+x)@derivative(x) + expected + 6*x, rtol=1e-13)
    np.testing.assert_allclose(p.constraint_jac(x), derivative(x)[:2], rtol=1e-13)
    calls = len(stats["rhs"])
    direction = np.arange(1., 6.)/10
    h = 1e-5
    rows = [p.evaluate_trial(x + sign*h*direction, predict=False, ftol=1e-15)[1] for sign in (1, -1)]
    np.testing.assert_allclose((rows[0]-rows[1])/(2*h), np.r_[gradient@direction, derivative(x)[:2]@direction],
                               rtol=1e-8)
    assert len(stats["rhs"]) == calls  # Independent FD requests neither a tangent nor an adjoint.
    assert stats["predictions"][-1]["ftol"] == stats["predictions"][-1]["edge_force_tolerance"] == 1e-15
    np.testing.assert_array_equal(stats["predictions"][-1]["initial_state"], p.accepted.state)
    assert p.accepted_step == 0


def test_evaluation_never_promotes_and_failed_trials_keep_the_anchor(scalar):
    p, stats, _, _ = scalar
    anchor = p.accepted
    candidate, _ = p.evaluate_trial(np.ones(5) * 0.001, 1)
    assert p.accepted is anchor
    p.accept(candidate)
    assert p.accepted is candidate and stats["closed"] > 0
    with pytest.raises(ValueError, match="current evaluation"):
        p.accept(anchor)
    stats["fail"] = True
    with pytest.raises(TrialRejected, match="did not converge"):
        p.evaluate_trial(np.ones(5) * 0.002, 1)
    assert p.accepted is candidate
    with pytest.raises(ValueError, match="evaluate"):
        p.accept_x(np.ones(5))


def test_rejected_point_is_not_solved_again_until_the_anchor_moves(scalar):
    p, stats, _, _ = scalar
    x = np.full(5, .002)
    stats["fail"] = True
    with pytest.raises(TrialRejected):
        p.fun(x)
    solves = stats["solves"]
    with pytest.raises(TrialRejected):
        p.constraint_values(x)  # the optimizer's constraint query at the same probe
    assert stats["solves"] == solves
    stats["fail"] = False
    other = np.full(5, .001)
    p.value_and_grad(other)
    p.accept_x(other)
    p.fun(x)  # a new anchor gives the same point a fresh prediction and solve
    assert stats["solves"] == solves + 2


def test_returned_arrays_do_not_mutate_cache(scalar):
    p, *_ = scalar
    _, grad = p.value_and_grad(p.x0)
    expected = grad.copy()
    grad[:] = 999
    np.testing.assert_array_equal(p.grad(p.x0), expected)


def test_acceptance_reuses_current_gradient_and_tangent(scalar):
    p, stats, *_ = scalar
    assert p.enable_matrix_free() is None
    assert len(stats["rhs"]) == 1 and len(stats["seeds"]) == 1
    seed, config = stats["seeds"][0], p.cfg
    x = np.full(5, .002)
    p.value_and_grad(x)
    factor = stats["factors"][-1]
    p.accept_x(x)
    assert p.cfg is config and p.accepted_step == 1 and not factor.closed
    calls = len(stats["rhs"])
    p.fun(x+np.ones(5)*.001)
    assert len(stats["rhs"]) == calls and len(stats["seeds"]) == 1 and not seed.closed
    np.testing.assert_array_equal(p.accepted.parameters, x)


def test_acceptance_preserves_exact_requested_parameters(scalar):
    problem, stats, state, _ = scalar
    anchor = np.full(5, .1)
    problem.value_and_grad(anchor)
    problem.accept_x(anchor)
    target = np.full(5, -.2)
    # Reconstructing the target from its increment changes the last bit.
    assert not np.array_equal(anchor + (target - anchor), target)
    problem.value_and_grad(target)
    problem.accept_x(target)
    np.testing.assert_array_equal(problem.accepted.parameters, target)
    np.testing.assert_array_equal(problem.accepted.state, state(target))
    solves = stats["solves"]
    problem.fun(target)
    problem.constraint_values(target)
    assert stats["solves"] == solves and problem.accepted_step == 2


@pytest.mark.parametrize("refresh_horizon", [None, 10])
def test_dense_recovery_refreshes_only_when_candidate_accepted(scalar, refresh_horizon):
    p, stats, *_ = scalar
    p.enable_matrix_free(refresh_horizon=refresh_horizon)
    seed = stats["seeds"][0]
    stats["matrixfree_fail"] = True
    p.value_and_grad(np.full(5, .002))
    rejected = stats["factors"][-1]
    assert len(stats["seeds"]) == 1 and not seed.closed
    p.value_and_grad(np.full(5, .001))
    assert rejected.closed and not seed.closed
    p.accept_x(np.full(5, .001))
    assert seed.closed and len(stats["seeds"]) == 2 and not stats["seeds"][-1].closed
    if refresh_horizon is not None:
        assert p._lu_refresh.warmup == 0 and not p._lu_refresh.costs  # already compiled


def test_failed_recovery_does_not_promote_or_retry_forever(scalar):
    p, stats, *_ = scalar
    p.enable_matrix_free()
    anchor, seed = p.accepted, stats["seeds"][0]
    stats["matrixfree_fail"] = stats["dense_fail"] = True
    calls = len(stats["rhs"])
    with pytest.raises(api.AdjointSolveError, match="dense failed"):
        p.value_and_grad(np.full(5, .001))
    assert len(stats["rhs"]) == calls+2 and p.accepted is anchor and not seed.closed


@pytest.mark.parametrize("options", [dict(refresh_horizon=h) for h in (0, -1, True, 1.5, float("nan"))]
                         + [dict(refresh_horizon=10, refresh_max_steps=b) for b in (0, -1, True, 1.5)])
def test_invalid_refresh_policy_builds_no_factors(scalar, options):
    p, stats, *_ = scalar
    with pytest.raises(ValueError, match="refresh_"):
        p.enable_matrix_free(**options)
    assert not stats["rhs"] and not stats["seeds"]


def test_lu_refresh_ignores_compilation_spikes_and_final_step_rebuilds():
    refresh = api._LURefresh(horizon=10, dense_seconds=100.)
    # Neither first-use compilation nor a single spike replaces good factors.
    assert not any(refresh.observe(cost) for cost in [300., 200., 5., 5., 5., 100., 5., 5.])
    assert not refresh.observe(20.)
    assert refresh.observe(20.)
    policy = api._LURefresh(horizon=10, dense_seconds=100., warmup=0,
                            costs=[20., 20.], best_seconds=1., remaining=1)
    assert not policy.observe(20.)
    assert policy.remaining == 0
    policy.remaining = 3
    assert not policy.observe(20.)  # Two remaining steps cannot repay 100 s.


def test_dense_derivatives_seed_every_accepted_step(scalar):
    p, stats, *_ = scalar
    p.enable_matrix_free(dense_derivatives=True)
    assert p.solver_info["active_adjoint"] == p.solver.adjoint_solver
    for step in (1, 2):
        x = np.full(5, step * .001)
        p.value_and_grad(x)
        p.accept_x(x)
        assert len(stats["seeds"]) == step + 1 and not stats["seeds"][-1].closed
        assert all(seed.closed for seed in stats["seeds"][:-1])


def test_dense_derivative_that_misses_its_gate_retries_matrix_free(scalar, monkeypatch):
    p, stats, *_ = scalar
    monkeypatch.setattr(api._LURefresh, "observe", lambda self, cost: True)  # no cost refresh redoes that dense solve
    p.enable_matrix_free(dense_derivatives=True, refresh_horizon=10)
    seed = stats["seeds"][0]
    stats["dense_fail"] = True
    x = np.full(5, .001)
    p.value_and_grad(x)  # the dense solve fails; the accepted seed LU recovers it matrix-free
    p.accept_x(x)
    assert len(stats["seeds"]) == 1 and not seed.closed  # no dense factors to reseed from
    stats["dense_fail"] = False
    p.value_and_grad(np.full(5, .002))
    p.accept_x(np.full(5, .002))
    assert len(stats["seeds"]) == 2 and seed.closed


@pytest.mark.parametrize("failed_refresh", [None, "solve", "seed", "parity", "nonfinite"])
def test_adaptive_refresh_preserves_acceptance_and_derivatives(scalar, monkeypatch, failed_refresh):
    p, stats, *_ = scalar
    clock = [0.]
    cost = [5.]
    pullback = api._pullback

    def timed_pullback(*args, preconditioner=None, **kwargs):
        value = pullback(*args, preconditioner=preconditioner, **kwargs)
        clock[0] += 100. if preconditioner is None else cost[0]
        tangent = value.tangent

        def timed_tangent(*args, **kwargs):
            result = tangent(*args, **kwargs)
            clock[0] += 1.
            return result

        value.tangent = timed_tangent
        if failed_refresh == "seed" and preconditioner is None and stats["seeds"]:
            def fail_seed(**kwargs):
                raise api.AdjointSolveError("seed failed")
            value.preconditioner = fail_seed
        if failed_refresh == 'parity' and preconditioner is None and stats['seeds']:
            value.field_jacobian = value.field_jacobian + .1
        if failed_refresh == 'nonfinite' and preconditioner is None and stats['seeds']:
            value.field_jacobian = value.field_jacobian * np.nan
        return value

    monkeypatch.setattr(api.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(api, "_pullback", timed_pullback)
    events = []
    p._emit = lambda name, **data: events.append((name, data))
    p.enable_matrix_free(refresh_horizon=10)
    seed = stats["seeds"][0]
    for step in range(1, 6):
        x = np.full(5, step*.001)
        p.value_and_grad(x)
        p.accept_x(x)
    anchor = p.accepted
    cost[0] = 20.
    # Slow rejected trials do not advance the policy or replace its factors.
    for step in range(6, 10):
        x = np.full(5, step*.001)
        p.value_and_grad(x)
    assert p.accepted is anchor and len(stats["seeds"]) == 1
    assert p._lu_refresh.costs == [6., 6., 6.]
    p.accept_x(x)
    x = np.full(5, .010)
    expected = p.value_and_grad(x)[1].copy()
    anchor = p.accepted
    candidate_linearization = p._linearization
    policy_before = p._lu_refresh
    stats["dense_fail"] = failed_refresh == "solve"
    if failed_refresh is not None:
        error = FloatingPointError if failed_refresh == "nonfinite" else api.AdjointSolveError
        with pytest.raises(error, match="failed|changed derivative|nonfinite"):
            p.accept_x(x)
        assert p.accepted is anchor and not seed.closed
        assert p._lu_refresh is policy_before
        assert not candidate_linearization.closed and len(stats["seeds"]) == 1
        if failed_refresh in ('seed', 'parity', 'nonfinite'):
            assert stats["factors"][-1].closed
        np.testing.assert_array_equal(p.grad(x), expected)
    else:
        p.accept_x(x)
        assert p.accepted_step == 7 and seed.closed and candidate_linearization.closed
        assert len(stats["seeds"]) == 2 and p._lu_refresh.dense_seconds == 100.
        np.testing.assert_array_equal(p.grad(x), expected)
        calls = len(stats["rhs"])
        p.fun(x+.001)  # The next predictor uses the refreshed accepted factors.
        assert len(stats["rhs"]) == calls
        refresh_events = [data for name, data in events if name == "preconditioner_refresh"]
        assert len(refresh_events) == 1 and refresh_events[0]["reason"] == "cost"


@pytest.mark.parametrize("budget", [1, 5])
def test_shared_minimize_promotes_only_certified_points(scalar, budget):
    p, stats, *_ = scalar
    p.enable_matrix_free()
    initial = p.fun(p.x0)
    seen = []
    result = opt.minimize(p, method="SLSQP", callback=lambda x: seen.append(x.copy()),
                          constraints=p.nonlinear_constraint([-1e6, -1e6], [1e6, 1e6]),
                          options={"maxiter": budget, "ftol": 1e-10})
    assert 0 < p.accepted_step == len(seen) <= budget
    np.testing.assert_array_equal(p.accepted.parameters, result.x)
    assert p.fun(result.x) < initial and len(stats["seeds"]) == 1


def test_failed_equilibrium_stops_minimize_at_the_accepted_root(scalar):
    p, stats, *_ = scalar
    p.enable_matrix_free()
    anchor = p.accepted
    stats["fail"] = True
    result = opt.minimize(p, method="SLSQP", callback=lambda x: pytest.fail("failed trial promoted"),
                          options={"maxiter": 20})
    assert not result.success
    np.testing.assert_array_equal(result.x, anchor.parameters)
    assert p.accepted is anchor and p.accepted_step == 0


def test_coil_quantity_includes_direct_and_moving_equilibrium_derivatives(scalar):
    _, stats, state, derivative = scalar
    q = build(scalar, lambda s, rt, c: jnp.sum(s*s), quantities=(lambda s, rt: s[0],),
              coil_quantities=(lambda s, rt, c: jnp.dot(s, c) + 2*s[1],))
    try:
        x = np.arange(5)*.001
        expected = np.asarray(state(x))
        jac = np.vstack([derivative(x)[0], (x+2*np.eye(5)[1])@derivative(x)+expected])
        np.testing.assert_allclose(q.constraint_values(x), [expected[0], expected@x+2*expected[1]])
        np.testing.assert_allclose(q.constraint_jac(x), jac, rtol=1e-13)
        direction = np.arange(1., 6.)/10
        h = 1e-5
        fd = (q.constraint_values(x+h*direction)-q.constraint_values(x-h*direction))/(2*h)
        np.testing.assert_allclose(fd, jac@direction, rtol=1e-9)
        constraint = q.nonlinear_constraint([0., 0.], [np.inf, np.inf], scales=[.2, .4])
        np.testing.assert_allclose(constraint.jac(x), jac/np.array([.2, .4])[:, None])
        assert q.accepted_step == 0 and stats['rhs'][-1] == 3
        with pytest.raises(ValueError, match="positive constraint scales"):
            q.nonlinear_constraint(0., 1., scales=0.)
    finally:
        q.close()


@pytest.mark.parametrize("kwargs,error,match", [
    (dict(loss=lambda s, rt, coils: s), ValueError, "scalar"),
    (dict(loss=lambda s, rt, c: jnp.sum(s), coil_quantities=(lambda s, rt, c: jnp.outer(c, c),)), ValueError,
     "one-dimensional"),
])
def test_scalar_rows_are_validated_before_optimization(scalar, kwargs, error, match):
    with pytest.raises(error, match=match):
        build(scalar, **kwargs)


def test_vector_quantities_give_one_constraint_row_per_entry(scalar):
    _, _, state, derivative = scalar
    q = build(scalar, lambda s, rt, c: jnp.sum(s * s), quantities=(lambda s, rt: s[:3], lambda s, rt: s[4]))
    try:
        x = np.full(5, .001)
        expected = np.asarray(state(x))
        np.testing.assert_allclose(q.constraint_values(x), np.r_[expected[:3], expected[4]])
        np.testing.assert_allclose(q.constraint_jac(x), derivative(x)[[0, 1, 2, 4]], rtol=1e-13)
        assert q.nonlinear_constraint(np.zeros(4), np.full(4, np.inf)).jac(x).shape == (4, 5)
    finally:
        q.close()


def test_newton_corrected_trial_skips_the_ordinary_solve(scalar):
    p, stats, *_ = scalar
    x = np.full(5, .001)
    root = p.evaluate_trial(x)[0]  # an ordinary solve gives a certified root to return
    solves, calls, events = stats["solves"], [], []
    p._newton_options = dict(tolerance=1e-12, max_steps=1)
    p._newton_trial = lambda point, predicted, trial: calls.append(point) or root
    p._emit = lambda name, **data: events.append((name, data))
    candidate, rows = p.evaluate_trial(x)
    assert candidate is root and stats["solves"] == solves and len(calls) == 1
    np.testing.assert_array_equal(calls[0], x)
    assert ("certification", 1) in [(name, data.get("index")) for name, data in events]
    np.testing.assert_array_equal(rows, p.optimizer_rows(root))


def test_from_loss_rejects_invalid_definitions():
    inp = SimpleNamespace(lfreeb=True)
    with pytest.raises(TypeError, match="callable"):
        FreeBoundaryProblem.from_loss(inp, lambda *a: 0., coil_quantities=(1,))
    jax.config.update("jax_enable_x64", True)
    with pytest.raises(ValueError, match="select coil_current_dofs explicitly"):
        FreeBoundaryProblem.from_loss(inp, lambda *a: 0., coils=object())


def test_complex_parameters_are_rejected_without_silent_conversion(scalar):
    problem, *_ = scalar
    with pytest.raises(ValueError, match="real"):
        problem.fun(problem.x0.astype(complex) + 1j)
    with pytest.raises(ValueError, match="real"):
        CoilParameters(np.zeros((1, 3, 3), dtype=complex), [1.0], current_dofs=(0,))


def test_equilibrium_preserves_standard_type_and_lazy_fixed_geometry_wout(scalar, monkeypatch):
    problem, stats, *_ = scalar
    seen = []
    output = object()

    def wout(record):
        seen.append(record)
        return output

    monkeypatch.setattr(problem, "_wout", wout)
    eq = problem.equilibrium_from_x(problem.x0)
    assert isinstance(eq, opt.Equilibrium)
    assert eq.solution is problem.accepted.state and seen == []
    assert eq.wout is output and eq.wout is output
    assert seen == [problem.accepted] and stats["solves"] == 0


@pytest.mark.parametrize("failure", [None, "solve", "edge", "adjoint"])
def test_constructor_solves_once_and_checks_the_initial_root(scalar, monkeypatch, failure):
    p, stats, state, _ = scalar
    seen = {}

    def make_solver(inp, field, **options):
        seen["options"] = options
        return p.solver

    def solve(inp, *, external_field, **kwargs):
        stats["solves"] += 1
        seen["solve"] = kwargs
        return SimpleNamespace(result=SimpleNamespace(converged=failure != "solve", state=state(external_field),
                                                      fedge=1e-12), rcon0=None, zcon0=None)

    def certify(solver, params, point, **kwargs):
        seen["certify"] = kwargs
        anchor = Root(np.asarray(point), kwargs["state"],
                      result=SimpleNamespace(fedge=1e-7 if failure == "edge" else 1e-12))
        return replace(p.cfg, _anchor=anchor)

    monkeypatch.setattr(api.fbi, "make_free_boundary_config", make_solver)
    monkeypatch.setattr(api.im, "params_from_input", lambda inp: None)
    monkeypatch.setattr(freeboundary, "_solve_free_boundary_stage", solve)
    monkeypatch.setattr(api, "_config_from_state", certify)
    kwargs = dict(parameterization=p.parameterization, restart_from=p.accepted.state,
                  solver_options=dict(ftol=1e-11, **({"adjoint_solver": "coupled_gcrot"} if failure == "adjoint" else {})))
    loss = lambda s, rt, coils: jnp.sum(s)  # noqa: E731
    if failure == "adjoint":
        with pytest.raises(ValueError, match="forward_dense_jax"):
            FreeBoundaryProblem.from_loss(p.inp, loss, **kwargs)
        assert stats["solves"] == 0
        return
    if failure:
        with pytest.raises(VmecError):
            FreeBoundaryProblem.from_loss(p.inp, loss, **kwargs)
    else:
        FreeBoundaryProblem.from_loss(p.inp, loss, **kwargs).close()
        assert seen["certify"]["root_residual_atol"] == 2e-6
        assert seen["options"]["adjoint_solver"] == "forward_dense_jax"
    assert stats["solves"] == 1
    assert seen["solve"]["initial_state"] is p.accepted.state
    assert seen["solve"]["jacobian_retries"] == 0 and seen["solve"]["include_edge_in_convergence"]


def test_real_dense_and_matrixfree_pullbacks_through_the_public_api(monkeypatch):
    """The scalar API on the actual LU and seed-preconditioned GMRES kernels."""
    matrix = jnp.array([[3., 2.], [-1., 4.]])
    coupling = jnp.array([[1., -2.], [3., 1.]])
    residual = jax.jit(lambda z, p, x, *_: matrix @ z + coupling @ x - p)
    chart = SimpleNamespace(x0=np.zeros(2), scales=np.ones(2), dof_names=("a", "b"), coils_from_x=lambda x: x)
    inp = SimpleNamespace(lfreeb=True)
    solver = SimpleNamespace(implicit=SimpleNamespace(inp=inp, device=None, lconm1=False, adjoint_tol=1e-11,
        adjoint_maxiter=10, adjoint_gcrot_m=2, adjoint_gcrot_k=1),
        adjoint_dense_batch_size=2, adjoint_dense_max_dofs=10, adjoint_solver="forward_dense_jax",
        adjoint_fail="error", adjoint_residual_rtol=1e-9, include_edge_in_convergence=True, field_from_parameters=chart)
    owner = object()
    root = SimpleNamespace(_owner=owner, parameters=np.zeros(2), state=jnp.zeros(2), dof_mask=jnp.ones(2),
                           rcon0=None, zcon0=None)
    cfg = SimpleNamespace(_owner=owner, _anchor=root, params=jnp.zeros(2), solver=solver,
                          root_residual_atol=2e-6, parameter_scales=chart.scales)
    monkeypatch.setattr(fbi, "_projected_residual", lambda *_, **__: residual)
    monkeypatch.setattr(im, "_dof_projector", lambda _, mask: lambda x: x*mask)
    monkeypatch.setattr(im, "runtime_from_params", lambda *_: None)
    p = FreeBoundaryProblem(inp, chart, cfg, loss=lambda s, rt, c: jnp.sum((s-1)**2)+jnp.sum(c),
                            quantities=(lambda s, rt: s[0], lambda s, rt: s[1]))
    try:
        expected = -np.linalg.solve(matrix, coupling)
        np.testing.assert_allclose(p.grad(p.x0), -2*expected.sum(axis=0)+1, rtol=1e-13)
        assert p.solver_info["active_adjoint"] == "forward_dense_jax"
        np.testing.assert_allclose(p.constraint_jac(p.x0), expected, rtol=1e-13)
        p.enable_matrix_free(restart=2, max_restarts=2)
        assert p.solver_info["active_adjoint"] == "matrixfree_seed_lu"
        assert p.solver_info["recovery"] == "forward_dense_jax"
        reports = []
        seeded = api._pullback(root, cfg, jnp.eye(2), preconditioner=p._preconditioner, diagnostics=reports)
        try:
            np.testing.assert_allclose(seeded.field_jacobian, expected, rtol=1e-12)
            assert reports and all(r["accepted"] and r["backend"] == "matrixfree_seed_lu" for r in reports)
        finally:
            seeded.close()
    finally:
        p.close()


# Coupled root polishing with the real dense and seed-LU kernels.

@dataclass(frozen=True, eq=False)
class PolishRoot:
    parameters: object
    state: object
    dof_mask: object
    _owner: object
    root_residual_norm: float
    rcon0: object = None
    zcon0: object = None
    result: object = None


@pytest.fixture
def coupled(monkeypatch):
    jax.config.update('jax_enable_x64', True)
    # Third coordinate is inactive and must remain exactly unchanged.
    mask = jnp.array([1., 1., 0.])
    matrix = jnp.array([[3., 1., 0.], [-1., 4., 0.], [0., 0., 0.]])
    coupling = jnp.array([[1., -2.], [3., 1.], [0., 0.]])
    params = jnp.array([.6, -.2, 0.])
    residual = jax.jit(lambda z, p, x, *_: matrix@z + .1*z*z*mask - p - coupling@x)

    class Chart:
        x0 = np.zeros(2)
        scales = np.ones(2)
        dof_names = ('a', 'b')

        def __call__(self, x):
            return x

        def coils_from_x(self, x):
            return x

    chart = Chart()
    inp = SimpleNamespace(lfreeb=True)
    solver = SimpleNamespace(implicit=SimpleNamespace(inp=inp, device=None, lconm1=False,
        ftol=1e-15, max_iterations=10), resolution=None, edge_force_tolerance=1e-15,
        adjoint_dense_batch_size=2, adjoint_dense_max_dofs=10, adjoint_solver='forward_dense_jax',
        adjoint_fail='error', adjoint_residual_rtol=1e-9, include_edge_in_convergence=True,
        field_from_parameters=chart)
    owner = object()
    initial = jnp.array([.2, -.01, 7.])
    anchor = PolishRoot(chart.x0.copy(), initial, mask, owner,
        float(jnp.linalg.norm(residual(initial*mask, params, chart.x0))),
        jnp.array([2.]), jnp.array([3.]), SimpleNamespace(iterations=5))
    cfg = SimpleNamespace(_owner=owner, _anchor=anchor, params=params, solver=solver,
        parameter_scales=chart.scales, continuation_step=.1, max_continuation_steps=64,
        root_residual_atol=1.)
    monkeypatch.setattr(api.fbi, '_projected_residual', lambda *_, **__: residual)
    monkeypatch.setattr(api.im, '_dof_projector', lambda _, m: lambda x: x*m)
    monkeypatch.setattr(api.im, 'runtime_from_params', lambda *_: None)

    def certify(config, point, state, *, rcon0, zcon0, **kw):
        assert config is cfg
        assert np.array_equal(rcon0, anchor.rcon0) and np.array_equal(zcon0, anchor.zcon0)
        assert float(state[2]) == 7.
        norm = float(jnp.linalg.norm(residual(state*mask, params, point)))
        # Fresh force diagnostics, not the original ordinary solve's result.
        # refine uses dataclasses.replace on a real result; supply its equivalent.
        @dataclass
        class Result:
            iterations: int = 0
            converged: bool = True
            fsqr: float = 1e-22
            fsqz: float = 1e-22
            fsql: float = 1e-22
            fedge: float = 1e-22
        result = Result()
        return PolishRoot(np.asarray(point).copy(), state, mask, owner, norm, rcon0, zcon0, result)

    monkeypatch.setattr(api, '_certify', certify)
    problem = FreeBoundaryProblem(inp, chart, cfg,
        loss=lambda s, rt, c: .5*jnp.sum(s[:2]**2)+jnp.sum(c*c),
        quantities=(lambda s, rt: s[0], lambda s, rt: s[1]))
    yield problem, residual
    problem.close()


def test_polished_seed_and_total_gradient_use_actual_kernels(coupled):
    p, residual = coupled
    old = p.accepted
    old_gradient = p.grad(p.x0).copy()
    p.enable_root_polishing()
    assert p.accepted is not old and p.accepted_step == 0
    assert p.accepted.root_residual_norm <= 1e-12
    assert p.accepted.state[2] == 7.
    assert p.accepted.result.iterations == 5
    p.enable_matrix_free(refresh_horizon=10)
    z = p.accepted.state
    matrix = jax.jacfwd(residual, 0)(z*jnp.array([1., 1., 0.]), p.params, jnp.zeros(2))[:2,:2]
    coupling = jax.jacfwd(residual, 2)(z*jnp.array([1., 1., 0.]), p.params, jnp.zeros(2))[:2]
    expected = np.asarray(z[:2]) @ -np.linalg.solve(matrix, coupling)
    np.testing.assert_allclose(p.grad(p.x0), expected, rtol=1e-10, atol=1e-12)
    assert not np.allclose(old_gradient, expected)


def test_polish_failure_keeps_initial_root_and_caches(coupled):
    p, _ = coupled
    original = p.accepted
    grad = p.grad(p.x0).copy()
    with pytest.raises(api._RootPolishError, match='budget exhausted'):
        p.enable_root_polishing(tolerance=1e-14, max_steps=1)
    assert p.accepted is original and p._root_polish_options is None
    np.testing.assert_array_equal(p.grad(p.x0), grad)


def test_polished_coil_quantity_matches_independent_endpoints(coupled, monkeypatch):
    """A moving-surface-like row retains both direct and root-response terms."""
    p, _ = coupled
    q = FreeBoundaryProblem(p.inp, p.parameterization, p.cfg, loss=lambda s, rt, c: jnp.sum(s[:2]**2),
        coil_quantities=(lambda s, rt, c: s[0]*c[0] + s[1] + 2*c[1],))
    try:
        q.enable_root_polishing()
        anchor = q.accepted

        def ordinary(inp, *, initial_state, **kwargs):
            return SimpleNamespace(result=SimpleNamespace(state=initial_state, converged=True, iterations=3),
                                   rcon0=anchor.rcon0, zcon0=anchor.zcon0)

        monkeypatch.setattr(freeboundary, '_solve_free_boundary_stage', ordinary)
        direction = np.array([.2, -.3])
        expected = q.constraint_jac(q.x0) @ direction
        for h in (3e-4, 1e-4):
            plus, vp = q.evaluate_trial(h*direction, predict=False)
            minus, vm = q.evaluate_trial(-h*direction, predict=False)
            assert max(plus.root_residual_norm, minus.root_residual_norm) <= 1e-12
            np.testing.assert_allclose((vp[1:]-vm[1:])/(2*h), expected, rtol=1e-7, atol=1e-9)
            assert q.accepted is anchor and q.accepted_step == 0
    finally:
        q.close()


def test_trial_polishing_preserves_accepted_seed_and_recertifies(coupled, monkeypatch):
    p, residual = coupled
    p.enable_root_polishing()
    p.enable_matrix_free(refresh_horizon=10)
    anchor, seed = p.accepted, p._preconditioner
    calls = []

    def ordinary(inp, *, external_field, initial_state, **kwargs):
        calls.append(np.asarray(initial_state).copy())
        # An imperfect ordinary root: Newton polishing must finish the solve.
        return SimpleNamespace(result=SimpleNamespace(state=initial_state, converged=True, iterations=3),
                               rcon0=anchor.rcon0, zcon0=anchor.zcon0)

    monkeypatch.setattr(freeboundary, '_solve_free_boundary_stage', ordinary)
    point = np.array([.001, -.002])
    trial, rows = p.evaluate_trial(point, predict=False)
    assert trial.root_residual_norm < 1e-12 and np.all(np.isfinite(rows))
    np.testing.assert_array_equal(calls[-1], anchor.state)
    assert p.accepted is anchor and p._preconditioner is seed
    p.accept(trial)
    assert p.accepted is trial and p.accepted_step == 1


@pytest.mark.parametrize('tolerance,steps', [(0.,3),(float('nan'),3),(1e-12,0),(1e-12,True)])
def test_invalid_polishing_contract(coupled, tolerance, steps):
    p, _ = coupled
    with pytest.raises(ValueError, match='positive finite'):
        p.enable_root_polishing(tolerance=tolerance, max_steps=steps)


@pytest.mark.parametrize('failure', [None, 'retry', 'report', 'handoff'])
def test_one_dense_retry_is_bounded_and_releases_temporary_factors(monkeypatch, failure):
    """A stale-seed failure gets one dense retry; its factors are released
    whether the retry, the report or the recovery handoff succeeds or fails."""
    calls = []

    class Temporary:
        def __init__(self, name):
            self.name = name

        def close(self):
            calls.append(f'close {self.name}')

    def refine(record, cfg, seed, **kwargs):
        calls.append('refine')
        if seed == 'old' or failure == 'retry':
            raise api._RootPolishError('stale LU')
        return 'polished', {'seconds': .1}

    def report(data):
        if failure == 'report' and data['event'] == 'polished':
            raise RuntimeError('report failed')

    def handoff(*args):
        raise RuntimeError('handoff failed')

    monkeypatch.setattr(api, '_refine', refine)
    call = lambda: api._polish_with_recovery(  # noqa: E731
        'root', 'config', 'old', lambda root: (Temporary('dense'), Temporary('seed')), report,
        retain_recovery=handoff if failure == 'handoff' else None)
    if failure is None:
        assert call() == 'polished'
    else:
        with pytest.raises(api._RootPolishError if failure == 'retry' else RuntimeError,
                           match='stale LU' if failure == 'retry' else failure):
            call()
    assert calls == ['refine', 'close dense', 'refine', 'close seed']


@pytest.mark.parametrize('error_type', [api._RootPolishError, api.AdjointSolveError])
def test_failed_tape_and_dense_factors_released_before_retry(monkeypatch, error_type):
    """Recovery must fit without simultaneously retaining both failed/new tapes."""
    failed_tape = []
    events = []
    dense = SimpleNamespace(closed=False)
    seed = SimpleNamespace(closed=False)
    dense.close = lambda: setattr(dense, 'closed', True)
    seed.close = lambda: setattr(seed, 'closed', True)

    class Tape:
        pass

    def refine(record, cfg, preconditioner, **kwargs):
        if preconditioner == 'old':
            tape = Tape()
            failed_tape.append(weakref.ref(tape))
            raise error_type('linear solve failed')
        assert failed_tape[0]() is None
        assert dense.closed and not seed.closed
        return record, {'seconds': 0.}

    def build_dense(record):
        assert failed_tape[0]() is None
        return dense, seed

    monkeypatch.setattr(api, '_refine', refine)
    assert api._polish_with_recovery('root', 'cfg', 'old', build_dense, events.append) == 'root'
    assert seed.closed
    assert events[0]['failure'] == 'linear solve failed'
    assert events[-1]['recovered_with_dense']


def test_bootstrap_releases_dense_before_real_refinement(coupled, monkeypatch):
    p, _ = coupled
    pullback = api._pullback
    refine = api._refine
    dense = []

    def capture(*args, **kwargs):
        result = pullback(*args, **kwargs)
        dense.append(result)
        return result

    def checked(*args, **kwargs):
        assert dense[-1]._root is None
        return refine(*args, **kwargs)

    monkeypatch.setattr(api, '_pullback', capture)
    monkeypatch.setattr(api, '_refine', checked)
    p.enable_root_polishing()
    assert p.accepted.root_residual_norm <= 1e-12


def test_real_dense_recovery_keeps_accepted_root_and_seed(coupled, monkeypatch):
    p, _ = coupled
    p.enable_root_polishing()
    p.enable_matrix_free()
    anchor, seed = p.accepted, p._preconditioner
    gradient = p.grad(p.x0).copy()
    events = []
    monkeypatch.setattr(p, '_emit', lambda name, **data: events.append((name, data)))
    point = np.array([.002, -.003])
    candidate = api._certify(p.cfg, point, anchor.state,
        rcon0=anchor.rcon0, zcon0=anchor.zcon0)
    solve = api.dense.solve
    calls = []

    def stale_once(action, template, space, factors, rhs, **options):
        calls.append(options)
        if len(calls) == 1:
            return jnp.zeros_like(rhs), 300, jnp.linalg.norm(rhs), False
        return solve(action, template, space, factors, rhs, **options)

    monkeypatch.setattr(api.dense, 'solve', stale_once)
    refined = p._polish_record(candidate)
    assert refined.root_residual_norm <= 1e-12
    assert p.accepted is anchor and p._preconditioner is seed
    assert seed._seed.factors is not None
    np.testing.assert_array_equal(p.grad(p.x0), gradient)
    retry = next(data for name, data in events if data.get('phase') == 'dense_retry')
    assert retry['linear_failure']['linear_relative_error'] == pytest.approx(1.)
    assert retry['linear_failure']['krylov_iterations'] == 300
    assert not retry['linear_failure']['krylov_converged']
    assert retry['linear_failure']['root_residual'] == pytest.approx(candidate.root_residual_norm)
    assert events[-1][1]['recovered_with_dense']


def test_already_polished_initial_state_is_not_changed(coupled):
    p, _ = coupled
    p.enable_root_polishing()
    anchor = p.accepted
    p.enable_root_polishing()
    assert p.accepted is anchor and p.accepted_step == 0


def recovered_trial(p, monkeypatch):
    """Force a stale-seed failure, then run the actual dense/Newton kernels."""
    p.enable_root_polishing()
    p.enable_matrix_free(refresh_horizon=10, refresh_max_steps=20)
    anchor = p.accepted
    point = np.array([.002, -.003])
    candidate = api._certify(p.cfg, point, anchor.state,
        rcon0=anchor.rcon0, zcon0=anchor.zcon0)
    real_refine = api._refine

    def stale(record, cfg, seed, **options):
        if seed is p._preconditioner:
            raise api._RootPolishError('stale accepted LU')
        return real_refine(record, cfg, seed, **options)

    with monkeypatch.context() as patch:
        patch.setattr(api, '_refine', stale)
        refined = p._polish_record(candidate)
    p._records[p._key(point)] = refined
    assert p._trial_lu.record is refined
    return refined


def test_recovery_seed_used_at_polished_root_and_promoted_only_on_accept(coupled, monkeypatch):
    p, residual = coupled
    candidate = recovered_trial(p, monkeypatch)
    anchor, old_seed = p.accepted, p._preconditioner
    trial_seed = p._trial_lu.seed
    dense_seconds = p._trial_lu.seconds
    pullback = api._pullback
    seen = []

    def checked(record, cfg, rhs, **options):
        seen.append(record)
        assert record is candidate  # new operator at the polished root
        assert options['preconditioner'] is trial_seed
        return pullback(record, cfg, rhs, **options)

    monkeypatch.setattr(api, '_pullback', checked)
    jac = p._derivatives(candidate)
    z, x = candidate.state, candidate.parameters
    matrix = jax.jacfwd(residual, 0)(z*candidate.dof_mask, p.params, x)[:2, :2]
    coupling = jax.jacfwd(residual, 2)(z*candidate.dof_mask, p.params, x)[:2]
    response = -np.linalg.solve(matrix, coupling)
    expected = np.vstack((np.asarray(z[:2])@response+2*x, response))
    np.testing.assert_allclose(jac, expected, rtol=1e-9, atol=1e-12)
    assert p.accepted is anchor and p._preconditioner is old_seed
    assert old_seed._seed is not None and trial_seed._seed is not None
    p.accept(candidate)
    assert len(seen) == 1  # no second dense assembly on acceptance
    assert p._preconditioner is trial_seed and p._trial_lu is None
    assert old_seed._seed is None and p.accepted is candidate
    assert p._lu_refresh.remaining == 19
    assert p._lu_refresh.dense_seconds == dense_seconds
    tangent = p._linearization.tangent(candidate, p.cfg, jnp.array([.2, -.1]))
    np.testing.assert_allclose(tangent[:2], response@np.array([.2, -.1]), rtol=1e-9, atol=1e-12)
    p.close()
    assert trial_seed._seed is None


@pytest.mark.parametrize('abandon', ['new_trial', 'close'])
def test_abandoned_recovery_seed_is_released(coupled, monkeypatch, abandon):
    p, _ = coupled
    candidate = recovered_trial(p, monkeypatch)
    anchor, old_seed, trial_seed = p.accepted, p._preconditioner, p._trial_lu.seed
    p._derivatives(candidate)
    if abandon == 'close':
        p.close()
        assert trial_seed._seed is None
        return

    def failed_ordinary(*args, **kwargs):
        assert trial_seed._seed is None
        np.testing.assert_array_equal(kwargs['initial_state'], anchor.state)
        raise api.TrialRejected('reject new proposal')

    monkeypatch.setattr(freeboundary, '_solve_free_boundary_stage', failed_ordinary)
    with pytest.raises(api.TrialRejected, match='reject new proposal'):
        p.evaluate_trial(np.array([.003, -.001]), predict=False)
    assert p._trial_lu is None and p._preconditioner is old_seed and p.accepted is anchor
    assert np.all(np.isfinite(p.grad(p.x0)))


def test_accepted_gradient_query_does_not_lose_trial_recovery(coupled, monkeypatch):
    p, _ = coupled
    candidate = recovered_trial(p, monkeypatch)
    old_seed, trial_seed = p._preconditioner, p._trial_lu.seed
    jac = p._derivatives(candidate).copy()
    assert np.all(np.isfinite(p.grad(p.x0)))
    assert p._preconditioner is old_seed and p._trial_lu.seed is trial_seed
    np.testing.assert_allclose(p._derivatives(candidate), jac, rtol=1e-12, atol=1e-14)
    p.accept(candidate)
    assert p._preconditioner is trial_seed and old_seed._seed is None


@pytest.mark.parametrize('recovery', [False, True])
def test_polished_fd_trial_must_keep_requested_force_tolerance(coupled, monkeypatch, recovery):
    """A polished finite-difference trial is rejected above its requested
    tolerance, and a recovery seed built for it is released."""
    p, _ = coupled
    p.enable_root_polishing()
    p.enable_matrix_free()
    anchor, old_seed = p.accepted, p._preconditioner
    real_refine = api._refine
    seeds = []

    def stale(record, cfg, seed, **options):
        if seed is old_seed:
            raise api._RootPolishError('stale accepted LU')
        seeds.append(seed)
        return real_refine(record, cfg, seed, **options)

    if recovery:
        monkeypatch.setattr(api, '_refine', stale)
    monkeypatch.setattr(freeboundary, '_solve_free_boundary_stage', lambda inp, **kw:
        SimpleNamespace(result=SimpleNamespace(state=kw['initial_state'], converged=True),
                        rcon0=anchor.rcon0, zcon0=anchor.zcon0))
    with pytest.raises(api.TrialRejected, match='requested force tolerance'):
        p.evaluate_trial(np.array([.001, -.002]), predict=False, ftol=1e-24)
    assert len(seeds) == recovery and all(seed._seed is None for seed in seeds) and p._trial_lu is None
    assert p.accepted is anchor and p._preconditioner is old_seed and old_seed._seed is not None


@pytest.mark.parametrize('dense_fails', [False, True])
def test_trial_seed_adjoint_failure_still_has_one_dense_fallback(coupled, monkeypatch, dense_fails):
    p, _ = coupled
    candidate = recovered_trial(p, monkeypatch)
    anchor, old_seed, trial_seed = p.accepted, p._preconditioner, p._trial_lu.seed
    pullback = api._pullback
    calls = []

    def fail_seed(record, cfg, rhs, **options):
        seed = options.get('preconditioner')
        calls.append(seed)
        if seed is not None or dense_fails:
            raise api.AdjointSolveError('injected solve failure')
        return pullback(record, cfg, rhs, **options)

    monkeypatch.setattr(api, '_pullback', fail_seed)
    if dense_fails:
        with pytest.raises(api.AdjointSolveError, match='injected'):
            p.accept(candidate)
        assert p.accepted is anchor and p._preconditioner is old_seed
        assert p._trial_lu is None and trial_seed._seed is None
        assert old_seed._seed is not None
    else:
        p.accept(candidate)
        assert p.accepted is candidate and p._preconditioner is not trial_seed
        assert trial_seed._seed is None and old_seed._seed is None
        assert p._preconditioner._seed is not None
    assert calls == [trial_seed, None]


# Root linearizations and seed-LU matrix-free solves.

def linear_root(monkeypatch):
    matrix=jnp.array([[3.,2.],[-1.,4.]])
    coupling=jnp.array([[1.,-2.],[3.,1.]])
    residual=jax.jit(lambda z,p,f,*_:matrix@z+coupling@f-p)
    cfg=NS(implicit=NS(device=None,lconm1=False,adjoint_tol=1e-11,adjoint_maxiter=10,adjoint_gcrot_m=2,adjoint_gcrot_k=1),
        adjoint_dense_batch_size=2,adjoint_dense_max_dofs=10,adjoint_solver='forward_dense_jax',
        adjoint_fail='error',adjoint_residual_rtol=2e-5)
    monkeypatch.setattr(fbi,'_projected_residual',lambda *_, **__:residual)
    monkeypatch.setattr(im,'_dof_projector',lambda _,mask:lambda x:x*mask)
    monkeypatch.setattr(im,'runtime_from_params',lambda *_:None)
    owner=object();state=jnp.zeros(2)
    accepted=NS(_owner=owner,parameters=np.zeros(2),state=state,dof_mask=jnp.ones(2),rcon0=None,zcon0=None)
    continuation=NS(_owner=owner,params=jnp.zeros(2),solver=cfg,root_residual_atol=2e-6)
    return accepted,continuation,matrix,coupling


def test_shared_factorization_orientation_and_scaling(monkeypatch):
    accepted,cfg,a,b=linear_root(monkeypatch)
    calls=[];original=jsl.lu_factor
    def measured(x):calls.append(1);return original(x)
    monkeypatch.setattr(jsl,'lu_factor',measured)
    rows=[]
    root=api._pullback(accepted,cfg,jnp.eye(2),diagnostics=rows)
    np.testing.assert_allclose(root.field_jacobian,-np.linalg.solve(a,b),rtol=1e-13)
    tangent_rows=[]
    for alpha in (1.,.5,.25):
        direction=jnp.array([.2,-.3])*alpha
        response=root.tangent(accepted,cfg,direction,diagnostics=tangent_rows)
        np.testing.assert_allclose(response,np.linalg.solve(a,-b@direction),rtol=1e-13)
    assert len(calls)==1
    assert [x['scaled_reuse'] for x in tangent_rows]==[False,True,True]
    assert all(x['accepted'] and x['relative_residual']<1e-12 for x in rows+tangent_rows)
    direction=jnp.array([.1,.7])
    np.testing.assert_allclose(root.tangent(accepted,cfg,direction),np.linalg.solve(a,-b@direction),rtol=1e-13)
    assert len(calls)==1


def test_original_operator_catches_corrupt_factors(monkeypatch):
    accepted,cfg,_,_=linear_root(monkeypatch)
    root=api._pullback(accepted,cfg,jnp.eye(2))
    root._root.factors=jsl.lu_factor(jnp.eye(2))
    rows=[]
    with pytest.raises(Exception,match='reused_dense_lu'):
        root.tangent(accepted,cfg,jnp.ones(2),diagnostics=rows)
    assert rows and not rows[0]['accepted']


def test_stale_and_closed_roots_fail_without_fallback(monkeypatch):
    accepted,cfg,_,_=linear_root(monkeypatch)
    root=api._pullback(accepted,cfg,jnp.eye(2))
    for other,c in ((NS(**vars(accepted)),cfg),(accepted,NS(**vars(cfg)))):
        with pytest.raises(ValueError,match='different root'):
            root.tangent(other,c,jnp.ones(2))
    root.close();root.close()
    assert root.field_jacobian is None
    with pytest.raises(ValueError,match='closed'):
        root.tangent(accepted,cfg,jnp.ones(2))


def test_zero_invalid_directions_and_jit_rejection(monkeypatch):
    accepted,cfg,_,_=linear_root(monkeypatch)
    root=api._pullback(accepted,cfg,jnp.eye(2))
    np.testing.assert_array_equal(root.tangent(accepted,cfg,jnp.zeros(2)),np.zeros(2))
    for direction in (np.ones(3),np.array([np.nan,0.]),np.array([1j,0j])):
        with pytest.raises(ValueError):
            root.tangent(accepted,cfg,direction)
    with jax.disable_jit(False), pytest.raises(ValueError,match='host-eager'):
        jax.jit(lambda d:root.tangent(accepted,cfg,d))(jnp.ones(2))


@pytest.mark.parametrize('backend,fail', [(name, 'best_effort') for name in fbi._ADJOINT_SOLVERS]
                         + [('coupled_gcrot', 'error')])
def test_unsupported_policy_cannot_fall_back(monkeypatch,backend,fail):
    accepted,cfg,_,_=linear_root(monkeypatch)
    cfg.solver.adjoint_solver=backend;cfg.solver.adjoint_fail=fail
    with pytest.raises(ValueError,match='retained linearization requires'):
        api._pullback(accepted,cfg,jnp.eye(2))


def test_new_root_refactors_and_preserves_root_gate(monkeypatch):
    accepted,cfg,_,_=linear_root(monkeypatch)
    accepted.state=jnp.ones(2)
    with pytest.raises(ValueError,match='root residual'):
        api._pullback(accepted,cfg,jnp.eye(2))
    accepted.state=jnp.zeros(2)
    first=api._pullback(accepted,cfg,jnp.eye(2))
    other=NS(**vars(accepted))
    second=api._pullback(other,cfg,jnp.eye(2))
    assert first._root is not second._root
    first.close()
    assert np.all(np.isfinite(second.tangent(other,cfg,jnp.ones(2))))


@pytest.mark.parametrize("rhs_batch_size", [1, 3])
def test_current_nonlinear_root_arbitrary_rows_and_seed_lifetime(monkeypatch, rhs_batch_size):
    accepted, cfg, matrix, coupling = linear_root(monkeypatch)
    residual = jax.jit(lambda z, p, f, *_: matrix @ z + 0.1 * z * z + coupling @ f - p)
    monkeypatch.setattr(fbi, "_projected_residual", lambda *_, **__: residual)
    seed_root = api._pullback(accepted, cfg, jnp.eye(2))
    seed_root.offload_factors()
    seed = seed_root.preconditioner(rhs_batch_size=rhs_batch_size)
    factors = seed._seed.factors[0].copy()
    seed_root.close()
    monkeypatch.setattr(api.dense, "_assemble_device", lambda *a, **k: pytest.fail("unexpected dense assembly"))
    state = jnp.array([0.2, -0.3])
    field = np.linalg.solve(coupling, -matrix @ state - 0.1 * state * state)
    other = NS(**(vars(accepted) | dict(state=state, parameters=field)))
    # Four unique rows also exercise the final, shorter batch.
    rhs = jnp.array([[1.0, 2.0], [3.0, 4.0], [-3.0, -4.0], [1.0, 2.0], [0.0, 0.0], [2.0, 5.0], [5.0, 1.0]])
    diagnostics = []
    root = api._pullback(
        other, cfg, rhs, preconditioner=seed, diagnostics=diagnostics
    )
    current = np.asarray(matrix) + np.diag(0.2 * np.asarray(state))
    expected = -np.asarray(rhs) @ np.linalg.solve(current, coupling)
    np.testing.assert_allclose(root.field_jacobian, expected, rtol=1e-10, atol=1e-13)
    assert all(row["accepted"] for row in diagnostics)
    assert all(row["krylov_converged"] for row in diagnostics)
    assert diagnostics[4]["krylov_residual_norm"] == 0.0
    assert [row["iterations"] for row in diagnostics][2:5] == [0, 0, 0]
    direction = jnp.array([0.3, -0.4])
    notes = []
    for alpha in (1.0, 0.5, 0.0):
        np.testing.assert_allclose(
            root.tangent(other, cfg, alpha * direction, diagnostics=notes),
            -np.linalg.solve(current, coupling @ (alpha * direction)),
            rtol=1e-10,
            atol=1e-13,
        )
    assert notes[1]["scaled_reuse"] and notes[1]["iterations"] == 0
    np.testing.assert_array_equal(seed._seed.factors[0], factors)
    seed.close()
    # A live current-root predictor owns its factor reference independently.
    assert np.all(np.isfinite(root.tangent(other, cfg, direction)))
    with pytest.raises(ValueError, match="closed"):
        api._pullback(other, cfg, rhs, preconditioner=seed).field_jacobian
    root.close()
    with pytest.raises(ValueError, match="closed"):
        root.tangent(other, cfg, direction)


def test_preconditioner_rejects_foreign_config_mask_and_uncertified_root(monkeypatch):
    accepted, cfg, _, _ = linear_root(monkeypatch)
    root = api._pullback(accepted, cfg, jnp.eye(2))
    seed = root.preconditioner()
    with pytest.raises(ValueError, match="different configuration"):
        api._pullback(accepted, NS(**vars(cfg)), jnp.eye(2), preconditioner=seed)
    other = NS(**(vars(accepted) | dict(dof_mask=jnp.array([1.0, 0.0]))))
    with pytest.raises(ValueError, match="active space"):
        api._pullback(other, cfg, jnp.eye(2), preconditioner=seed).field_jacobian
    other = NS(**(vars(accepted) | dict(state=jnp.ones(2))))
    with pytest.raises(ValueError, match="root residual"):
        api._pullback(other, cfg, jnp.eye(2), preconditioner=seed).field_jacobian
    for kw in (
        dict(rtol=0.0),
        dict(rtol=np.nan),
        dict(restart=0),
        dict(max_restarts=1.5),
    ):
        with pytest.raises(ValueError):
            root.preconditioner(**kw)
    seed.close()
    root.close()


def test_exact_rhs_reuse_ignores_signed_zero_but_not_small_changes(monkeypatch):
    accepted, cfg, _, _ = linear_root(monkeypatch)
    dense_root = api._pullback(accepted, cfg, jnp.eye(2))
    seed = dense_root.preconditioner()
    rows = jnp.array([[1.0, 0.0], [-1.0, 0.0], [1.0, -0.0], [1.0, 1e-20]])
    reports = []
    result = api._pullback(accepted, cfg, rows, preconditioner=seed, diagnostics=reports).field_jacobian
    assert [r["reused_rhs"] for r in reports] == [False, True, True, False]
    assert reports[1]["iterations"] == reports[2]["iterations"] == 0
    np.testing.assert_array_equal(np.asarray(result)[1], -np.asarray(result)[0])
    assert all(r["accepted"] for r in reports)
    dense_root.close()
    seed.close()


def test_nonnormal_current_operator_rejects_short_solve_and_supports_two_recoveries(monkeypatch):
    """Exercise actual Krylov failure, a larger basis and fresh dense factors."""
    accepted, cfg, _, _ = linear_root(monkeypatch)
    n = 96
    matrix = np.eye(n) - np.diag(np.ones(n - 1), -1)
    for offset in (1, 2, 3):
        matrix += np.diag(np.ones(n - offset), offset)
    identity, current = jnp.eye(n), jnp.asarray(matrix)
    coupling = jnp.asarray(np.random.default_rng(2).normal(size=(n, 2)))
    residual = jax.jit(lambda z, p, f, *_: (identity + f[0] * (current - identity)) @ z + coupling @ f - p)
    monkeypatch.setattr(fbi, "_projected_residual", lambda *_, **__: residual)
    cfg.params = jnp.zeros(n)
    cfg.solver.adjoint_dense_max_dofs = 128
    cfg.solver.adjoint_dense_batch_size = 16
    cfg.solver.adjoint_residual_rtol = 1e-6
    accepted.state, accepted.dof_mask = jnp.zeros(n), jnp.ones(n)
    rhs = jnp.asarray(np.random.default_rng(0).normal(size=(1, n)))
    reference0 = api._pullback(accepted, cfg, rhs)
    short = reference0.preconditioner(restart=30, max_restarts=1)
    larger = reference0.preconditioner(restart=100, max_restarts=3)
    reference0.close()
    state = jnp.asarray(np.linalg.solve(matrix, -np.asarray(coupling)[:, 0]))
    point = NS(**(vars(accepted) | dict(state=state, parameters=np.array([1.0, 0.0]))))
    reports = []
    from vmex.core.errors import AdjointSolveError

    with pytest.raises(AdjointSolveError):
        api._pullback(point, cfg, rhs, preconditioner=short, diagnostics=reports).field_jacobian
    assert reports[0]["iterations"] == 30 and not reports[0]["accepted"]
    dense_reference = api._pullback(point, cfg, rhs)
    field_matrix = np.column_stack(
        ((matrix - np.eye(n)) @ np.asarray(state) + np.asarray(coupling)[:, 0], np.asarray(coupling)[:, 1])
    )
    expected = -np.asarray(rhs) @ np.linalg.solve(matrix, field_matrix)
    np.testing.assert_allclose(dense_reference.field_jacobian, expected, rtol=1e-10, atol=1e-11)
    refreshed = dense_reference.preconditioner()
    for seed in (larger, refreshed):
        answer = api._pullback(point, cfg, rhs, preconditioner=seed).field_jacobian
        np.testing.assert_allclose(answer, expected, rtol=1e-8, atol=1e-10)
    dense_reference.close()
    for seed in (short, larger, refreshed):
        seed.close()


def test_true_residual_rejects_bad_krylov_answer_in_both_directions(monkeypatch):
    accepted, cfg, _, _ = linear_root(monkeypatch)
    dense_root = api._pullback(accepted, cfg, jnp.eye(2))
    seed = dense_root.preconditioner()
    root = api._pullback(
        accepted, cfg, jnp.eye(2), preconditioner=seed
    )
    monkeypatch.setattr(
        api.dense,
        "solve",
        lambda action, z, space, lu, rhs, **kw: (
            (jnp.zeros_like(rhs), 1, 0.0, True) if kw.get("return_info") else (jnp.zeros_like(rhs), 1)
        ),
    )
    reports = []
    with pytest.raises(Exception, match="matrixfree_seed_lu") as error:
        api._pullback(
            accepted, cfg, jnp.eye(2), preconditioner=seed, diagnostics=reports
        ).field_jacobian
    assert reports and not reports[0]["accepted"]
    assert "restart=2, max_cycles=10" in str(error.value)
    assert "max_restarts=300" not in str(error.value)
    assert reports[0]["requested_rtol"] == 1e-11
    with pytest.raises(Exception, match="matrixfree_seed_lu"):
        root.tangent(accepted, cfg, jnp.ones(2))
    root.close()
    seed.close()
    dense_root.close()


def test_nonfinite_full_residual_has_boolean_rejection_diagnostics(monkeypatch):
    accepted, cfg, _, _ = linear_root(monkeypatch)
    reference = api._pullback(accepted, cfg, jnp.eye(2))
    seed = reference.preconditioner()
    # A finite primal root can still have an undefined derivative. A bogus
    # finite Krylov answer must not turn the rejection into a string in JSON.
    monkeypatch.setattr(api.dense, "prepare", lambda *args, **kwargs: lambda value: value * jnp.nan)
    monkeypatch.setattr(api.dense, "solve", lambda *args, **kwargs: (jnp.ones(2), 1, 0.0, True))
    reports = []
    from vmex.core.errors import AdjointSolveError

    with pytest.raises(AdjointSolveError):
        api._pullback(
            accepted, cfg, jnp.eye(2), preconditioner=seed, diagnostics=reports
        ).field_jacobian
    assert reports[0]["accepted"] is False
    assert reports[0]["full_residual_accepted"] is False
    assert not np.isfinite(reports[0]["residual_norm"])
    seed.close()
    reference.close()


def test_stable_tape_changes_numbers_without_new_executables():
    matrix = jnp.array([[3.0, 1.0], [-0.5, 2.0]])

    @jax.jit
    def residual(z, p, f, b, r, c):
        return matrix @ z + 0.01 * z * z + f

    space = api.dense._Space(jnp.arange(2), jnp.arange(2), jnp.zeros(2), jnp.ones(2))
    factors = jsl.lu_factor(matrix.T)  # retained factors are of the Jacobian transpose
    counts = []
    for i in range(3):
        z = jnp.array([0.2, 0.4]) + i * 0.1
        action = api.dense.prepare(z, None, jnp.zeros(2), z, None, None, residual=residual)
        current = np.asarray(matrix) + np.diag(0.02 * np.asarray(z))
        for transpose in (False, True):
            result, _ = api.dense.solve(
                action, z, space, factors, jnp.ones(2), transpose=transpose, rtol=1e-11, restart=30, max_restarts=10
            )
            np.testing.assert_allclose(
                result, np.linalg.solve(current.T if transpose else current, np.ones(2)), rtol=1e-10
            )
        counts.append(api.dense.solve._cache_size())
    assert counts[0] == counts[1] == counts[2]


def test_redundant_paired_coordinates():
    space = api.dense._Space(jnp.array([0, 1]), jnp.array([0, 2]), jnp.array([0.0, -1.0]), jnp.array([1.0, 1 / np.sqrt(2)]))
    q = np.array([[1.0, 0.0], [0.0, 1 / np.sqrt(2)], [0.0, -1 / np.sqrt(2)]])
    active = np.array([[2.0, 0.5], [-0.8, 3.0]])
    action = jax.tree_util.Partial(jnp.matmul, jnp.asarray(q @ active @ q.T))
    factors = jsl.lu_factor(jnp.asarray(active + np.eye(2) * 0.1).T)
    rhs = jnp.array([1.0, -2.0])
    for transpose in (False, True):
        solution, _ = api.dense.solve(
            action, jnp.zeros(3), space, factors, rhs, transpose=transpose, rtol=1e-11, restart=30, max_restarts=10
        )
        np.testing.assert_allclose(solution, np.linalg.solve(active.T if transpose else active, rhs), rtol=1e-10)


@pytest.mark.parametrize("size", [0, 5, True, 1.5])
def test_rhs_batch_size_is_bounded(monkeypatch, size):
    accepted, cfg, _, _ = linear_root(monkeypatch)
    root = api._pullback(accepted, cfg, jnp.eye(2))
    try:
        with pytest.raises(ValueError, match="rhs_batch_size"):
            root.preconditioner(rhs_batch_size=size)
    finally:
        root.close()


def test_batched_krylov_preserves_independent_convergence():
    matrix = jnp.diag(jnp.arange(1.0, 10.0))
    template = jnp.zeros(9)
    index = jnp.arange(9)
    space = api.dense._Space(index, index, jnp.zeros(9), jnp.ones(9))
    action = jax.tree_util.Partial(lambda a, x: a @ x, matrix)
    rhs = jnp.stack([jnp.eye(9)[0], jnp.ones(9), jnp.zeros(9)])
    x, iterations, norm, converged = api.dense._solve_many(
        action, template, space, jsl.lu_factor(jnp.eye(9)), rhs, rtol=1e-11, restart=9, max_restarts=2
    )
    np.testing.assert_allclose(x, np.asarray(rhs) / np.arange(1.0, 10.0), rtol=1e-10, atol=1e-12)
    assert list(map(int, iterations))[0] == 1 and int(iterations[1]) > 1 and int(iterations[2]) == 0
    assert bool(jnp.all(converged)) and bool(jnp.all(norm < 1e-10))


@pytest.mark.parametrize("rhs_batch_size", [1, 3])
def test_nonfinite_cotangents_are_rejected_before_krylov(monkeypatch, rhs_batch_size):
    accepted, cfg, _, _ = linear_root(monkeypatch)
    root = api._pullback(accepted, cfg, jnp.eye(2))
    seed = root.preconditioner(rhs_batch_size=rhs_batch_size)
    monkeypatch.setattr(api.dense, "solve", lambda *a, **k: pytest.fail("nonfinite RHS reached Krylov"))
    try:
        with pytest.raises(ValueError, match="nonfinite adjoint right-hand side"):
            api._pullback(
                accepted, cfg, jnp.array([[1.0, jnp.nan]]), preconditioner=seed
            ).field_jacobian
    finally:
        seed.close()
        root.close()


def test_closed_linearization_cannot_create_or_offload_seed(monkeypatch):
    accepted, cfg, _, _ = linear_root(monkeypatch)
    root = api._pullback(accepted, cfg, jnp.eye(2))
    dense_root = root._root
    root.close()
    for operation in (root.preconditioner, root.offload_factors):
        with pytest.raises(ValueError, match="closed"):
            operation()
    with pytest.raises(ValueError, match="live dense linearization"):
        api.dense.SeedLU.from_root(dense_root)


def test_seed_rejects_changed_solver_closed_storage_and_single_precision(monkeypatch):
    accepted, cfg, _, _ = linear_root(monkeypatch)
    root = api._pullback(accepted, cfg, jnp.eye(2))
    dense_root = root._root
    seed = root.preconditioner()
    with pytest.raises(ValueError, match="different solver or state layout"):
        seed._seed.validate(accepted.state, accepted.parameters, dense_root.space, NS(**vars(cfg.solver)))
    seed._seed.close()
    with pytest.raises(ValueError, match="closed"):
        seed._seed.validate(accepted.state, accepted.parameters, dense_root.space, cfg.solver)
    with pytest.raises(ValueError, match="Krylov rtol"):
        root.preconditioner(rtol=np.nan)
    factors, pivots = dense_root.factors
    dense_root.factors = (factors.astype(jnp.float32), pivots)
    with pytest.raises(TypeError, match="float64"):
        root.preconditioner()
    seed.close()
    root.close()


@pytest.mark.parametrize("backend, failure", [("coupled_gcrot", "error"), ("forward_dense_jax", "best_effort")])
def test_seed_cannot_bypass_strict_backend_policy(monkeypatch, backend, failure):
    accepted, cfg, _, _ = linear_root(monkeypatch)
    root = api._pullback(accepted, cfg, jnp.eye(2))
    seed = root.preconditioner()
    cfg.solver.adjoint_solver, cfg.solver.adjoint_fail = backend, failure
    try:
        with pytest.raises(ValueError, match="requires forward_dense_jax"):
            api._pullback(accepted, cfg, jnp.eye(2), preconditioner=seed).field_jacobian
    finally:
        seed.close()
        root.close()


# Trial correction from the accepted root.

def callback(monkeypatch, tmp_path, *, converge=True, certify=True):

    anchor = NS(parameters=np.zeros(1), state=10., rcon0=1., zcon0=2.)
    result = NS(state=20., converged=converge, iterations=30,
                fsqr=1e-12, fsqz=1e-12, fsql=1e-12, fedge=1e-12)
    solve = Mock(return_value=NS(result=result, rcon0=11., zcon0=12.))
    tangent = Mock(return_value=4.)
    cert = Mock(side_effect=lambda cfg, point, state, **kw: NS(
        parameters=np.asarray(point), state=state, rcon0=11., zcon0=12.,
        root_residual_norm=1e-7, result=result))
    if not certify:
        cert.side_effect = VmecError('root residual gate')
    monkeypatch.setattr(freeboundary, '_solve_free_boundary_stage', solve)
    monkeypatch.setattr(api, '_certify', cert)
    problem = FreeBoundaryProblem.__new__(FreeBoundaryProblem)
    problem._accepted_linearization = None
    problem._preconditioner = None
    problem._root_polish_options = None
    problem._trial_lu = None
    problem._newton_options = None
    problem.x0 = problem.scales = np.ones(1)
    problem.deadline = None
    problem.accepted = anchor
    problem.cfg = NS(continuation_step=.1, max_continuation_steps=64)
    problem.solver = NS(resolution=None, edge_force_tolerance=1e-11,
                        implicit=NS(ftol=1e-11, max_iterations=12000))
    problem.inp = None
    problem.parameterization = lambda p: p
    problem._linearization = NS(tangent=tangent)
    problem._linearization_record = anchor
    problem._compact_jac = np.zeros((4, 1))
    problem._scalar_rows = lambda state, x: np.zeros(4)
    problem.metadata = {'holder': {'failed_trials': 0}}
    problem._emit = Mock()
    return problem, solve, tangent, cert


def test_tangent_path_retains_prediction(monkeypatch, tmp_path):
    problem, solve, tangent, _ = callback(monkeypatch, tmp_path)
    problem.evaluate_trial(np.array([.15]), 1)
    tangent.assert_called_once()
    # One tangent prediction from the accepted root, then one correction solve.
    assert [c.kwargs['initial_state'] for c in solve.call_args_list] == [14.]


@pytest.mark.parametrize('converge,certify', [(False, True), (True, False)])
def test_reused_tangent_rejects_failed_equilibrium_and_certification(monkeypatch, tmp_path, converge, certify):
    problem, solve, tangent, cert = callback(monkeypatch, tmp_path, converge=converge, certify=certify)
    anchor = problem.accepted
    with pytest.raises(TrialRejected):
        problem.evaluate_trial(np.array([.05]), 1)
    tangent.assert_called_once()
    assert problem.accepted is anchor
    if not converge:
        cert.assert_not_called()
    emitted = problem._emit.call_args_list[-1]
    assert emitted.args == ('candidate',)
    assert emitted.kwargs['stage'] is solve.return_value


def test_each_backtracking_trial_restarts_at_the_accepted_anchor(monkeypatch, tmp_path):
    problem, solve, _, _ = callback(monkeypatch, tmp_path)
    problem.evaluate_trial(np.array([.15]), 1)
    problem.evaluate_trial(np.array([.05]), 2)
    assert [c.kwargs['initial_state'] for c in solve.call_args_list] == [14., 14.]
    assert [c.kwargs['constraint_continuation'] for c in solve.call_args_list] == [(1., 2.), (1., 2.)]
    for c in solve.call_args_list:
        assert c.kwargs['ftol'] == c.kwargs['edge_force_tolerance'] == 1e-11
        assert c.kwargs['include_edge_in_convergence'] and c.kwargs['jacobian_retries'] == 0


def test_failed_derivative_preparation_never_launches_correction(monkeypatch, tmp_path):
    problem, solve, tangent, cert = callback(monkeypatch, tmp_path)
    problem._linearization_record = None
    monkeypatch.setattr(problem, '_derivatives', Mock(side_effect=RuntimeError('derivative gate failed')))
    with pytest.raises(RuntimeError, match='derivative gate failed'):
        problem.evaluate_trial(np.array([.05]), 1)
    solve.assert_not_called()
    tangent.assert_not_called()
    cert.assert_not_called()


# Coil coordinates.

def test_coil_chart_selects_currents_and_keeps_nominal_data_immutable():
    rng = np.random.default_rng(38)
    coefficients = rng.normal(size=(2, 3, 5))
    currents = np.array([3e5, -2e5])
    chart = CoilParameters(coefficients, currents, current_dofs=(1,), nfp=3)
    x = rng.normal(size=chart.size) * 0.001
    assert chart.size == 31 and len(chart.dof_names) == 31
    np.testing.assert_allclose(chart.base_currents_at(x), [currents[0], currents[1] * (1 + x[0])])
    np.testing.assert_allclose(chart.curve_dofs_at(x), coefficients + x[1:].reshape(2, 3, 5))
    coefficients[:] = 0
    currents[:] = 0
    assert np.any(chart.coefficients) and np.any(chart.currents)


def test_essos_coil_parameter_roundtrip():
    pytest.importorskip("essos.coils")
    rng = np.random.default_rng(483)
    chart = CoilParameters(rng.normal(size=(2, 3, 5)), [2e5, -3e5], current_dofs=(1,), nfp=2, stellsym=True,
                           n_segments=24)
    reconstructed = CoilParameters.from_coils(chart.coils_from_x(chart.x0), current_dofs=(1,))
    np.testing.assert_array_equal(reconstructed.coefficients, chart.coefficients)
    np.testing.assert_array_equal(reconstructed.currents, chart.currents)
    assert reconstructed.nfp == 2 and reconstructed.stellsym and reconstructed.n_segments == 24


@pytest.fixture
def physical_coils():
    essos_coils = pytest.importorskip("essos.coils")
    jax.config.update("jax_enable_x64", True)
    coefficients = np.random.default_rng(747).normal(size=(2, 3, 5))
    curves = essos_coils.Curves(jnp.asarray(coefficients), 24, 2, True, scaling_factor=0.7, scale_fixed=2.5)
    currents = jnp.array([2.4e5, -3.1e5])
    return essos_coils.Coils(curves, currents, currents_scale=8e4), coefficients, currents


def test_import_preserves_physical_units_and_symmetry(physical_coils):
    coils, coefficients, currents = physical_coils
    chart = CoilParameters.from_coils(coils, current_dofs=(1,))
    np.testing.assert_allclose(chart.coefficients, coefficients, rtol=2e-15, atol=2e-15)
    np.testing.assert_array_equal(chart.currents, currents)
    restored = chart.coils_from_x(chart.x0)
    np.testing.assert_allclose(restored.gamma, coils.gamma, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(restored.gamma_dash, coils.gamma_dash, rtol=2e-14, atol=2e-14)
    np.testing.assert_array_equal(restored.currents, coils.currents)
    assert (chart.nfp, chart.stellsym, chart.n_segments, chart.size) == (2, True, 24, 31)


def test_export_reload_keeps_physical_currents_and_geometry(physical_coils, tmp_path):
    coils, _, _ = physical_coils
    chart = CoilParameters.from_coils(coils, current_dofs=(1,))
    x = np.random.default_rng(483).normal(size=chart.size) * chart.scales
    exported = chart.coils_from_x(x)
    path = tmp_path / "coils.json"
    exported.to_json(str(path))
    reloaded = pytest.importorskip("essos.coils").Coils.from_json(str(path))
    rebuilt = CoilParameters.from_coils(reloaded, current_dofs=(1,))
    np.testing.assert_allclose(rebuilt.coefficients, chart.curve_dofs_at(x), rtol=2e-15, atol=2e-15)
    np.testing.assert_array_equal(rebuilt.currents, chart.base_currents_at(x))
    np.testing.assert_allclose(reloaded.gamma, exported.gamma, rtol=2e-14, atol=2e-14)


def test_field_derivatives_under_jit_and_nested_transforms(physical_coils):
    coils, _, _ = physical_coils
    chart = CoilParameters.from_coils(coils, current_dofs=(1,))
    scales = jnp.asarray(chart.scales)

    def field(u):
        return jnp.stack(
            chart(u * scales).b_cyl(jnp.array([0.8, 1.1, 1.4]), jnp.array([0.1, 0.3, 0.7]), jnp.array([0.2, -0.1, 0.3]))
        ).ravel()

    compiled = jax.jit(field)
    derivative = jax.jit(jax.jacfwd(compiled))
    rng = np.random.default_rng(748)
    for u in (np.zeros(chart.size), rng.normal(size=chart.size)):
        jacobian = derivative(u)
        assert np.isfinite(jacobian).all()
        # Exercise current and Fourier directions independently, at two states.
        for index in (0, 1, chart.size - 1):
            delta = np.eye(chart.size)[index] * 1e-4
            fd = (compiled(u + delta) - compiled(u - delta)) / 2e-4
            np.testing.assert_allclose(jacobian[:, index], fd, rtol=3e-6, atol=1e-10)
        weights = jnp.linspace(0.2, 1.0, 9)
        reverse = jax.jit(jax.grad(lambda v: jnp.vdot(compiled(v), weights)))(u)
        np.testing.assert_allclose(reverse, weights @ jacobian, rtol=2e-12, atol=1e-12)
