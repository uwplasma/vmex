"""Public API, derivatives, trial ownership and bounds on an analytic root.

These tests exercise the real API/optimizer with a known implicit equilibrium;
they do not claim qualification of the 48x40 plasma equilibrium.
"""

from dataclasses import dataclass, replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vmex import optimize as opt
from vmex.core import freeboundary_problem as api
from vmex.core.coil_parameters import CoilParameters
from vmex.core.projected_optimization import TrialRejected


def test_trial_rejection_keeps_public_and_legacy_import_identity():
    from vmex.core.errors import TrialRejected as SharedTrialRejected

    assert opt.TrialRejected is TrialRejected is SharedTrialRejected
    with pytest.raises(TrialRejected, match="uncertified trial"):
        raise opt.TrialRejected("uncertified trial")


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
def analytic(monkeypatch):
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

        def motion_bounds(self, delta):
            return float(np.max(np.abs(delta))), 0.0

        def coils_from_x(self, x):
            return jnp.asarray(x)

    chart = Chart()
    inp = SimpleNamespace(lfreeb=True)
    solver = SimpleNamespace(
        implicit=SimpleNamespace(inp=inp, ftol=1e-11, max_iterations=10),
        resolution=None,
        edge_force_tolerance=1e-11,
        field_from_parameters=chart,
        include_edge_in_convergence=True,
        adjoint_solver="forward_dense_jax", adjoint_fail="error",
    )
    anchor = Root(chart.x0.copy(), state(chart.x0))
    cfg = Config(solver, None, anchor, chart.scales)
    stats = dict(rhs=[], solves=0, closed=0, fail=False, gradient_calls=0, seeds=[], factors=[],
                 matrixfree_fail=False, dense_fail=False, parity_error=0.0, predictions=[])
    monkeypatch.setattr(api.im, "runtime_from_params", lambda *a: None)

    def pullback(record, config, rhs, *, diagnostics=None, return_linearization=False, preconditioner=None):
        stats["rhs"].append(rhs.shape[0])
        if preconditioner is not None:
            assert not preconditioner.closed and preconditioner.config is config
            if stats["matrixfree_fail"]:
                raise api.AdjointSolveError("matrix-free failed")
        elif stats["dense_fail"]:
            raise api.AdjointSolveError("dense failed")
        jac = np.asarray(rhs) @ derivative(record.parameters)
        if preconditioner is not None:
            jac = jac + stats["parity_error"]
        if not return_linearization:
            return jac

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
        return SimpleNamespace(
            result=SimpleNamespace(state=state(external_field), converged=not stats["fail"]), rcon0=None, zcon0=None
        )

    def certify(config, point, value, **kwargs):
        np.testing.assert_allclose(value, state(point), rtol=0, atol=0)
        return Root(np.asarray(point).copy(), value)

    monkeypatch.setattr(api.fc, "free_boundary_continuation_state_pullback", pullback)
    monkeypatch.setattr(api.fc, "certify_free_boundary_continuation_state", certify)
    monkeypatch.setattr(
        api.fc, "reanchor_free_boundary_continuation_config", lambda cfg, root: replace(cfg, _anchor=root)
    )
    from vmex.core import freeboundary

    monkeypatch.setattr(freeboundary, "_solve_free_boundary_stage", correct)
    terms = [(lambda s, rt: s, 0.0, 2.0)]
    constraints = [
        opt.TargetBand(lambda s, rt: s[0], 0.2, scale=0.005),
        opt.TargetBand(lambda s, rt: s[1], 5.0, scale=0.05),
    ]
    problem = opt.FreeBoundaryProblem.from_tuples(
        inp, terms, parameterization=chart, continuation=cfg, constraints=constraints, objective_normalization=2.0
    )
    yield problem, stats, state, derivative
    problem.close()


def test_scalar_and_constraint_derivatives_share_compact_pullback(analytic):
    p, stats, state, derivative = analytic
    value, gradient = p.value_and_grad(p.x0)
    expected = np.asarray(state(p.x0))
    np.testing.assert_allclose(value, 0.25 * np.dot(expected, expected))
    np.testing.assert_allclose(gradient, 0.5 * expected @ derivative(p.x0), rtol=1e-13)
    np.testing.assert_allclose(p.constraint_jac(p.x0), derivative(p.x0)[:2], rtol=1e-13)
    p.value_and_grad(p.x0)
    assert stats["rhs"] == [3]  # Objective norm plus two constraints, not five residual rows.
    assert stats["solves"] == 0
    residual, jac = p.residual_and_jac(p.x0)
    np.testing.assert_allclose(jac.T @ residual, gradient, rtol=1e-13)
    assert stats["rhs"] == [3, 5]


def test_finite_differences_and_nonpromoting_evaluation(analytic):
    p, stats, _, _ = analytic
    anchor = p.accepted
    gradient = p.grad(p.x0)
    eps = 1e-5
    fd = []
    for index in range(p.x0.size):
        delta = np.eye(p.x0.size)[index] * eps
        fd.append((p.fun(p.x0 + delta) - p.fun(p.x0 - delta)) / (2 * eps))
    np.testing.assert_allclose(gradient, fd, rtol=2e-8, atol=1e-9)
    assert p.accepted is anchor
    candidate, _ = p.evaluate_trial(np.ones(5) * 0.001, 1)
    assert p.accepted is anchor
    p.accept(candidate)
    assert p.accepted is candidate and stats["closed"] > 0
    with pytest.raises(ValueError, match="current evaluation"):
        p.accept(anchor)


def test_failed_trial_keeps_anchor_and_reports_rejection(analytic):
    p, stats, _, _ = analytic
    before = p.accepted
    stats["fail"] = True
    with pytest.raises(TrialRejected, match="did not converge"):
        p.evaluate_trial(np.ones(5) * 0.001, 1)
    assert p.accepted is before
    result = opt.minimize_projected(p, maxiter=2)
    assert result.status == "stagnated" and result.nit == 0 and not result.success
    assert p.accepted is before


def test_optimizer_promotes_only_accepted_steps_and_preserves_reference(analytic):
    p, stats, _, _ = analytic
    events = []
    result = opt.minimize_projected(p, maxiter=2, initial_gradient_norm=2.0, callback=events.append)
    assert result.nit == 2 and result.status == "step_budget_reached" and not result.success
    assert result.initial_gradient_norm == 2.0
    assert len(events) == 2 and result.accepted is p.accepted
    assert result.fun < p.fun(p.x0)
    assert np.all(np.abs(result.constraints - p.targets) <= p.constraint_tolerances)


def test_returned_arrays_do_not_mutate_cache(analytic):
    p, *_ = analytic
    _, grad = p.value_and_grad(p.x0)
    expected = grad.copy()
    grad[:] = 999
    np.testing.assert_array_equal(p.grad(p.x0), expected)


def test_invalid_chart_owner_and_bands(analytic):
    p, *_ = analytic
    with pytest.raises(ValueError, match="identical field chart"):
        opt.FreeBoundaryProblem.from_tuples(
            p.inp, [(lambda s, rt: s, 0, 1)], parameterization=object(), continuation=p.cfg
        )
    with pytest.raises(ValueError):
        opt.TargetBand(lambda s, rt: s[0], 0.0, rtol=0.01)
    assert opt.TargetBand(lambda s, rt: s[0], 0.0, atol=0.2).tolerance == 0.2
    with pytest.raises(ValueError):
        opt.minimize_projected(p, maxiter=-1)


def test_general_coil_chart_and_continuous_motion_bound():
    rng = np.random.default_rng(38)
    shape = (2, 3, 7)
    coefficients = rng.normal(size=shape)
    currents = np.array([3e5, -2e5])
    chart = CoilParameters(coefficients, currents, current_dofs=(1,), max_coil_mode=2, nfp=3)
    x = rng.normal(size=chart.size) * 0.001
    assert chart.size == 31 and len(chart.dof_names) == 31
    np.testing.assert_allclose(chart.base_currents_at(x), [currents[0], currents[1] * (1 + x[0])])
    after = np.asarray(chart.curve_dofs_at(x))
    np.testing.assert_array_equal(after[:, :, 5:], coefficients[:, :, 5:])
    np.testing.assert_allclose(after[:, :, :5], coefficients[:, :, :5] + x[1:].reshape(2, 3, 5))
    t = np.arange(10001) / 10001
    basis = np.array(
        [np.ones_like(t), np.sin(2 * np.pi * t), np.cos(2 * np.pi * t), np.sin(4 * np.pi * t), np.cos(4 * np.pi * t)]
    )
    movement = np.einsum("cdk,ks->csd", x[1:].reshape(2, 3, 5), basis)
    bound, current = chart.motion_bounds(x)
    assert np.max(np.linalg.norm(movement, axis=-1)) <= bound * (1 + 1e-12)
    assert current == abs(x[0])
    coefficients[:] = 0
    currents[:] = 0
    assert np.any(chart.coefficients) and np.any(chart.currents)


def test_zero_objective_has_finite_zero_gradient_and_converges(analytic):
    p, stats, _, _ = analytic
    zero = opt.FreeBoundaryProblem.from_tuples(
        p.inp,
        [(lambda state, rt: jnp.zeros(3), 0.0, 1.0)],
        parameterization=p.parameterization,
        continuation=p.cfg,
        constraints=p.constraints,
    )
    try:
        value, grad = zero.value_and_grad(zero.x0)
        assert value == 0.0 and np.all(grad == 0.0)
        result = opt.minimize_projected(zero, maxiter=1)
        assert result.success and result.nit == 0 and stats["solves"] == 0
    finally:
        zero.close()


def test_callback_stop_preserves_accepted_result(analytic):
    p, *_ = analytic

    def stop(result):
        raise StopIteration

    result = opt.minimize_projected(p, maxiter=10, callback=stop)
    assert result.status == "callback_stopped" and result.nit == 1
    assert result.accepted is p.accepted


def test_continuation_scale_and_edge_gates(analytic):
    p, *_ = analytic
    terms = [(lambda state, rt: state, 0.0, 1.0)]
    with pytest.raises(ValueError, match="scales differ"):
        opt.FreeBoundaryProblem.from_tuples(
            p.inp,
            terms,
            parameterization=p.parameterization,
            continuation=replace(p.cfg, parameter_scales=np.full(5, 2.0)),
        )
    p.solver.include_edge_in_convergence = False
    with pytest.raises(ValueError, match="edge-certified"):
        opt.FreeBoundaryProblem.from_tuples(p.inp, terms, parameterization=p.parameterization, continuation=p.cfg)


@pytest.mark.parametrize("failure", [None, "solve", "edge"])
def test_standalone_constructor_solves_once_and_checks_initial_root(analytic, monkeypatch, failure):
    p, stats, state, _ = analytic
    from vmex.core import freeboundary

    seen = {}

    def make_solver(inp, field, **options):
        seen["options"] = options
        return p.solver

    def solve(inp, *, external_field, **kwargs):
        stats["solves"] += 1
        seen["solve"] = kwargs
        return SimpleNamespace(
            result=SimpleNamespace(converged=failure != "solve", state=state(external_field), fedge=1e-12),
            rcon0=None,
            zcon0=None,
        )

    def certify(solver, params, point, **kwargs):
        seen["certify"] = kwargs
        anchor = Root(
            np.asarray(point), kwargs["state"], result=SimpleNamespace(fedge=1e-7 if failure == "edge" else 1e-12)
        )
        return replace(p.cfg, _anchor=anchor)

    monkeypatch.setattr(api.fbi, "make_free_boundary_config", make_solver)
    monkeypatch.setattr(api.im, "params_from_input", lambda inp: None)
    monkeypatch.setattr(freeboundary, "_solve_free_boundary_stage", solve)
    monkeypatch.setattr(api.fc, "make_free_boundary_continuation_config_from_state", certify)
    kwargs = dict(
        parameterization=p.parameterization,
        restart_from=p.accepted.state,
        constraints=p.constraints,
        solver_options=dict(ftol=1e-11),
    )
    if failure:
        with pytest.raises(api.VmecError):
            opt.FreeBoundaryProblem.from_tuples(p.inp, [(lambda s, rt: s, 0.0, 1.0)], **kwargs)
    else:
        other = opt.FreeBoundaryProblem.from_tuples(p.inp, [(lambda s, rt: s, 0.0, 1.0)], **kwargs)
        other.close()
        assert seen["certify"]["root_residual_atol"] == 2e-6
        assert seen["options"]["adjoint_solver"] == "forward_dense_jax"
    assert stats["solves"] == 1
    assert seen["solve"]["initial_state"] is p.accepted.state
    assert seen["solve"]["jacobian_retries"] == 0
    assert seen["solve"]["include_edge_in_convergence"]


def test_essos_coil_parameter_roundtrip():
    pytest.importorskip("essos.coils")
    rng = np.random.default_rng(483)
    chart = CoilParameters(
        rng.normal(size=(2, 3, 5)), [2e5, -3e5], current_dofs=(1,), nfp=2, stellsym=True, n_segments=24
    )
    coils = chart.coils_from_x(chart.x0)
    reconstructed = CoilParameters.from_coils(coils, current_dofs=(1,))
    np.testing.assert_array_equal(reconstructed.coefficients, chart.coefficients)
    np.testing.assert_array_equal(reconstructed.currents, chart.currents)
    assert reconstructed.nfp == 2 and reconstructed.stellsym and reconstructed.n_segments == 24


def test_complex_parameters_are_rejected_without_silent_conversion(analytic):
    problem, *_ = analytic
    with pytest.raises(ValueError, match="real"):
        problem.fun(problem.x0.astype(complex) + 1j)
    with pytest.raises(ValueError, match="real"):
        CoilParameters(np.zeros((1, 3, 3), dtype=complex), [1.0], current_dofs=(0,))


def test_equilibrium_preserves_standard_type_and_lazy_fixed_geometry_wout(analytic, monkeypatch):
    problem, stats, *_ = analytic
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


@pytest.fixture
def scalar(analytic):
    """A joint state/coil loss has an explicit and an implicit derivative."""
    p, stats, state, derivative = analytic
    def loss(s, rt, coils):
        return 0.5*jnp.sum(s*s) + jnp.sum(coils*s) + 3*jnp.sum(coils*coils)
    problem = opt.FreeBoundaryProblem.from_loss(p.inp, loss, quantities=(lambda s, rt: s[0], lambda s, rt: s[1]),
                                               parameterization=p.parameterization, continuation=p.cfg)
    yield problem, stats, state, derivative
    problem.close()


def test_scalar_total_gradient_and_constraints_share_three_rows(scalar):
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
    np.testing.assert_allclose((rows[0]-rows[1])/(2*h), np.r_[gradient@direction, derivative(x)[:2]@direction], rtol=1e-8)
    assert len(stats["rhs"]) == calls  # Independent FD does not request a tangent or adjoint.
    assert stats["predictions"][-1]["ftol"] == stats["predictions"][-1]["edge_force_tolerance"] == 1e-15
    np.testing.assert_array_equal(stats["predictions"][-1]["initial_state"], p.accepted.state)
    assert p.accepted_step == 0


def test_scalar_acceptance_reuses_current_gradient_and_tangent(scalar):
    p, stats, *_ = scalar
    p.enable_matrix_free(np.ones(5)*.001)
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


@pytest.mark.parametrize("fixture", ["analytic", "scalar"])
def test_acceptance_preserves_exact_requested_parameters(request, fixture):
    problem, stats, state, _ = request.getfixturevalue(fixture)
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
    p.enable_matrix_free(np.ones(5)*.001, refresh_horizon=refresh_horizon)
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
        assert p._lu_refresh.warmup == 2 and not p._lu_refresh.costs


def test_failed_recovery_does_not_promote_or_retry_forever(scalar):
    p, stats, *_ = scalar
    p.enable_matrix_free(np.ones(5)*.001)
    anchor, seed = p.accepted, stats["seeds"][0]
    stats["matrixfree_fail"] = stats["dense_fail"] = True
    calls = len(stats["rhs"])
    with pytest.raises(api.AdjointSolveError, match="dense failed"):
        p.value_and_grad(np.full(5, .001))
    assert len(stats["rhs"]) == calls+2 and p.accepted is anchor and not seed.closed


def test_matrixfree_parity_failure_retains_dense_seed(scalar):
    p, stats, *_ = scalar
    p.value_and_grad(p.x0)
    dense = stats["factors"][-1]
    stats["parity_error"] = .1
    with pytest.raises(api.AdjointSolveError, match="dense reference"):
        p.enable_matrix_free(np.ones(5)*.001)
    assert stats["seeds"][0].closed and not dense.closed
    assert p._preconditioner is None
    p.fun(np.full(5, .001))


def test_scalar_api_rejects_invalid_loss_and_unseen_acceptance(analytic):
    p, *_ = analytic
    with pytest.raises(ValueError, match="scalar"):
        opt.FreeBoundaryProblem.from_loss(p.inp, lambda s, rt, coils: s,
            parameterization=p.parameterization, continuation=p.cfg)
    with pytest.raises(ValueError, match="evaluate"):
        p.accept_x(np.ones(5))
    with pytest.raises(ValueError, match="no normalization"):
        opt.FreeBoundaryProblem.from_loss(p.inp, lambda *a: 0., objective_normalization=1.)


@pytest.mark.parametrize("lasym", [False, True])
def test_public_boundary_coefficients_roundtrip_and_derivative(lasym):
    from pathlib import Path
    from vmex import VmecInput
    from vmex.core.solver import prepare_runtime, resolution_from_input, _initial_state

    inp = VmecInput.from_file(Path(__file__).resolve().parents[1] / "examples/single-stage-benchmarks/input.rotating_ellipse")
    inp = inp.change_resolution(mpol=3, ntor=2, ntheta=12, nzeta=12)
    inp = replace(inp, lasym=lasym)
    if lasym:
        rbs, zbc = np.zeros_like(inp.rbc), np.zeros_like(inp.zbs)
        rbs[inp.ntor+1, 1], zbc[inp.ntor+1, 1] = .01, .02
        inp = replace(inp, rbs=rbs, zbc=zbc)
    rt = prepare_runtime(inp, resolution_from_input(inp, ns=5))
    state = _initial_state(rt.setup)
    arrays = opt.boundary_from_state(state, rt)
    for actual, expected in zip(arrays, (inp.rbc, inp.zbs, inp.rbs, inp.zbc)):
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
    def loss(s):
        return sum(jnp.sum(a*a) for a in opt.boundary_from_state(s, rt))
    direction = jax.tree.map(jnp.ones_like, state)
    analytic = jax.jvp(loss, (state,), (direction,))[1]
    plus = jax.tree.map(lambda a, d: a+1e-5*d, state, direction)
    minus = jax.tree.map(lambda a, d: a-1e-5*d, state, direction)
    np.testing.assert_allclose(analytic, (loss(plus)-loss(minus))/(2e-5), rtol=1e-9, atol=1e-9)


@pytest.mark.usefixtures("_module_jit_enabled")
@pytest.mark.parametrize("backend", ["forward_dense_jax", "forward_dense", "coupled_gcrot"])
def test_scalar_api_with_real_dense_and_matrixfree_pullbacks(monkeypatch, backend):
    """Exercise the public scalar API through the actual LU/FGMRES machinery."""
    from vmex.core import freeboundary_implicit as fbi, implicit as im

    if backend == "forward_dense_jax":
        monkeypatch.setattr(fbi, "_tangent_gcrot_core",
                            lambda *a, **k: pytest.fail("production must not invoke GCROT"))
    matrix = jnp.array([[3., 2.], [-1., 4.]])
    coupling = jnp.array([[1., -2.], [3., 1.]])
    residual = jax.jit(lambda z, p, x, *_: matrix @ z + coupling @ x - p)
    chart = SimpleNamespace(x0=np.zeros(2), scales=np.ones(2), dof_names=("a", "b"), coils_from_x=lambda x: x)
    inp = SimpleNamespace(lfreeb=True)
    solver = SimpleNamespace(implicit=SimpleNamespace(inp=inp, device=None, lconm1=False, adjoint_tol=1e-11,
        adjoint_maxiter=10, adjoint_gcrot_m=2, adjoint_gcrot_k=1),
        adjoint_dense_batch_size=2, adjoint_dense_max_dofs=10, adjoint_solver=backend,
        adjoint_fail="error", adjoint_residual_rtol=1e-9, include_edge_in_convergence=True, field_from_parameters=chart)
    owner = object()
    root = SimpleNamespace(_owner=owner, parameters=np.zeros(2), state=jnp.zeros(2), dof_mask=jnp.ones(2), rcon0=None, zcon0=None)
    cfg = SimpleNamespace(_owner=owner, _anchor=root, params=jnp.zeros(2), solver=solver,
                          root_residual_atol=2e-6, parameter_scales=chart.scales)
    monkeypatch.setattr(fbi, "_projected_residual", lambda *_: residual)
    monkeypatch.setattr(im, "_dof_projector", lambda _, mask: lambda x: x*mask)
    monkeypatch.setattr(im, "runtime_from_params", lambda *_: None)
    p = opt.FreeBoundaryProblem.from_loss(inp, lambda s, rt, c: jnp.sum((s-1)**2)+jnp.sum(c),
        quantities=(lambda s, rt: s[0], lambda s, rt: s[1]), parameterization=chart, continuation=cfg)
    try:
        expected = -np.linalg.solve(matrix, coupling)
        np.testing.assert_allclose(p.grad(p.x0), -2*expected.sum(axis=0)+1, rtol=1e-13)
        assert p.solver_info["active_adjoint"] == backend
        np.testing.assert_allclose(p.constraint_jac(p.x0), expected, rtol=1e-13)
        if backend != "forward_dense_jax":
            with pytest.raises(ValueError, match="seed-LU reuse requires"):
                p.enable_matrix_free()
            assert p.accepted is root
            return
        report = p.enable_matrix_free(np.array([.01, -.02]), restart=2, max_restarts=2)
        assert report["passed"]
        assert p.solver_info["active_adjoint"] == "matrixfree_seed_lu"
        assert p.solver_info["recovery"] == "forward_dense_jax"
        np.testing.assert_allclose(p.grad(p.x0), -2*expected.sum(axis=0)+1, rtol=1e-13)
        np.testing.assert_allclose(p.constraint_jac(p.x0), expected, rtol=1e-13)
    finally:
        p.close()


def test_slsqp_scalar_problem_accepts_only_gradient_points(scalar):
    from scipy.optimize import minimize
    p, stats, *_ = scalar
    p.enable_matrix_free(np.ones(5)*.001)
    anchors = []

    def gradient(x):
        value, grad = p.value_and_grad(x)
        if not np.array_equal(x, p.accepted.parameters):
            p.accept_x(x)
            anchors.append((x.copy(), value))
        return grad

    # Tight box keeps this analytic test inside the production proposal bound.
    result = minimize(p.fun, p.x0, jac=gradient, method="SLSQP", bounds=[(-.01, .01)]*5,
                      constraints=[dict(type="ineq", fun=lambda x: p.constraint_values(x)[0]-.19,
                                        jac=lambda x: p.constraint_jac(x)[0])],
                      options=dict(maxiter=10, ftol=1e-10))
    gradient(result.x)
    assert result.success and anchors and p.accepted_step == len(anchors)
    np.testing.assert_array_equal(p.accepted.parameters, result.x)
    assert len(stats["seeds"]) == 1
    assert all(after[1] <= before[1] for before, after in zip(anchors, anchors[1:]))


def test_production_matrixfree_setup_uses_no_parity_experiment(scalar):
    p, stats, *_ = scalar
    assert p.enable_matrix_free() is None
    assert len(stats["rhs"]) == 1 and len(stats["seeds"]) == 1
    np.testing.assert_allclose(p.grad(p.x0), p._accepted_jac[0])
    assert len(stats["rhs"]) == 1
    p.value_and_grad(np.full(5, .001))
    assert len(stats["rhs"]) == 2


@pytest.mark.parametrize("horizon", [0, -1, True, 1.5, float("nan")])
def test_invalid_refresh_horizon_does_not_build_factors(scalar, horizon):
    p, stats, *_ = scalar
    with pytest.raises(ValueError, match="refresh_horizon"):
        p.enable_matrix_free(refresh_horizon=horizon)
    assert not stats["rhs"] and not stats["seeds"]


def test_lu_refresh_ignores_compilation_and_isolated_spikes():
    refresh = api._LURefresh(horizon=10, dense_seconds=100.)
    # Compilation is expensive; neither first-use time nor a single spike
    # should replace good factors. A sustained slowdown should.
    assert not any(refresh.observe(cost) for cost in [300., 200., 5., 5., 5., 100., 5., 5.])
    assert not refresh.observe(20.)
    assert refresh.observe(20.)


@pytest.mark.parametrize("failed_refresh", [None, "solve", "seed", "parity"])
def test_adaptive_refresh_preserves_acceptance_and_derivatives(scalar, monkeypatch, failed_refresh):
    p, stats, *_ = scalar
    clock = [0.]
    cost = [5.]
    pullback = api.fc.free_boundary_continuation_state_pullback

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
        return value

    monkeypatch.setattr(api.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(api.fc, "free_boundary_continuation_state_pullback", timed_pullback)
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
        with pytest.raises(api.AdjointSolveError, match="failed|changed derivative"):
            p.accept_x(x)
        assert p.accepted is anchor and not seed.closed
        assert p._lu_refresh is policy_before
        assert not candidate_linearization.closed and len(stats["seeds"]) == 1
        if failed_refresh in ('seed', 'parity'):
            assert stats["factors"][-1].closed
        np.testing.assert_array_equal(p.grad(x), expected)
    else:
        p.accept_x(x)
        assert p.accepted_step == 7 and seed.closed and candidate_linearization.closed
        assert len(stats["seeds"]) == 2 and p._lu_refresh.dense_seconds == 100.
        np.testing.assert_array_equal(p.grad(x), expected)
        calls = len(stats["rhs"])
        p.fun(x+.001)  # Next predictor uses the refreshed accepted factors.
        assert len(stats["rhs"]) == calls
        refresh_events = [data for name, data in events if name == "preconditioner_refresh"]
        assert len(refresh_events) == 1 and refresh_events[0]["reason"] == "cost"


def test_refresh_budget_avoids_final_step_rebuild():
    policy = api._LURefresh(horizon=10, dense_seconds=100., warmup=0,
                           costs=[20., 20.], best_seconds=1., remaining=1)
    assert not policy.observe(20.)
    assert policy.remaining == 0
    policy.remaining = 3
    assert not policy.observe(20.)  # Two remaining steps cannot repay 100 s.


@pytest.mark.parametrize('budget', [0, -1, True, 1.5])
def test_invalid_refresh_budget_builds_no_factors(scalar, budget):
    p, stats, *_ = scalar
    with pytest.raises(ValueError, match='refresh_max_steps'):
        p.enable_matrix_free(refresh_horizon=10, refresh_max_steps=budget)
    assert not stats['rhs']


def test_scalar_checkpoint_restores_without_ordinary_solve(analytic, monkeypatch, tmp_path):
    from vmex.core import _freeboundary_checkpoint as storage, freeboundary
    from vmex.core.solver import SpectralState
    p, _, _, _ = analytic
    state = SpectralState(*(jnp.ones((2, 3)) for _ in storage.FIELDS))
    baselines = np.zeros((2, 3, 1))
    root = Root(np.zeros(5), state, baselines, baselines, state)
    holder = SimpleNamespace(checkpoint_identity="contract", accepted=root, accepted_step=0,
                             initial_gradient_norm=None, loss_scale=1.)
    path = tmp_path/"seed.npz"
    info = storage.write(holder, path)
    monkeypatch.setattr(storage, "identity", lambda *a, **k: "contract")
    monkeypatch.setattr(api.fbi, "make_free_boundary_config", lambda *a, **k: p.solver)
    monkeypatch.setattr(api.im, "params_from_input", lambda *a: None)
    monkeypatch.setattr(api.fc, "make_free_boundary_continuation_config_from_state",
                        lambda *a, **k: replace(p.cfg, _anchor=root))
    monkeypatch.setattr(freeboundary, "_solve_free_boundary_stage", lambda *a, **k: pytest.fail("checkpoint must not solve again"))
    restored = opt.FreeBoundaryProblem.from_loss(p.inp, lambda s, rt, coils: jnp.sum(s.R_cos),
        parameterization=p.parameterization, checkpoint=path, checkpoint_sha256=info["sha256"], checkpoint_identity="case")
    try:
        assert restored.accepted_step == 0 and restored.fun(restored.x0) == 6
        np.testing.assert_array_equal(restored.accepted.state.R_cos, state.R_cos)
        anchor = restored.accepted
        replay = restored.state_from_checkpoint(path, sha256=info["sha256"], parameters=restored.x0)
        np.testing.assert_array_equal(replay.R_cos, state.R_cos)
        assert restored.accepted is anchor and restored.accepted_step == 0
        with pytest.raises(ValueError, match="SHA256"):
            restored.state_from_checkpoint(path, sha256="wrong")
        with pytest.raises(ValueError, match="parameters"):
            restored.state_from_checkpoint(path, sha256=info["sha256"], parameters=restored.x0+1)
    finally:
        restored.close()


@pytest.mark.parametrize("wrong_gradient", [False, True])
def test_separate_qualification_checks_shared_scalar_problem(scalar, monkeypatch, tmp_path, wrong_gradient):
    from pathlib import Path
    import importlib
    import json

    example = Path(__file__).resolve().parents[1]/"examples/single-stage-benchmarks"
    monkeypatch.syspath_prepend(str(example))
    verifier = importlib.import_module("verify_free_boundary_single_stage")
    p, stats, *_ = scalar
    anchor = p.accepted
    stage = SimpleNamespace(problem=p, chart=p.parameterization,
                            constraint_transform=np.array([[1., 0], [0, 1.], [0, -1.]]),
                            inequalities=lambda values: np.r_[values, -values[1]])
    if wrong_gradient:
        original = p.value_and_grad
        def corrupted(x):
            value, gradient = original(x)
            return value, gradient+1
        monkeypatch.setattr(p, "value_and_grad", corrupted)
        with pytest.raises(RuntimeError, match="derivative check failed"):
            verifier.verify_problem(stage, tmp_path)
        assert not stats["seeds"]
        assert not json.loads((tmp_path/"gradient_check.json").read_text())["passed"]
    else:
        report = verifier.verify_problem(stage, tmp_path)
        assert report["finite_difference"]["passed"] and report["matrix_free"]["passed"]
        assert len(stats["predictions"]) == 4
        assert all(item["ftol"] == 1e-20 for item in stats["predictions"])
        assert all(item["initial_state"] is anchor.state or np.array_equal(item["initial_state"], anchor.state)
                   for item in stats["predictions"])
    assert p.accepted is anchor and p.accepted_step == 0


def test_boundary_from_wout_preserves_physical_modes_and_asymmetry():
    wout = SimpleNamespace(mpol=3, ntor=1, nfp=2, ns=2,
        xm=np.array([0, 1, 1, 2]), xn=np.array([0, -2, 2, 0]),
        rmnc=np.array([[0., 0., 0., 0.], [1., .2, .3, .4]]),
        zmns=np.array([[0., 0., 0., 0.], [0., -.2, .3, .1]]),
        rmns=np.ones((2, 4))*.01, zmnc=np.ones((2, 4))*.02)
    rbc, zbs, rbs, zbc = opt.boundary_from_wout(wout)
    assert rbc.shape == (3, 3) and rbc[0, 1] == .2 and rbc[2, 1] == .3
    assert zbs[0, 1] == -.2 and rbs[2, 1] == .01 and zbc[0, 1] == .02
    cropped = opt.boundary_from_wout(wout, mpol=2, ntor=0)[0]
    np.testing.assert_array_equal(cropped, [[1., 0.]])
    padded = opt.boundary_from_wout(wout, mpol=4, ntor=2)[0]
    np.testing.assert_array_equal(padded[1:4, :3], rbc)
    assert not padded[:, 3].any()


@pytest.mark.parametrize("budget", [1, 20])
def test_lbfgsb_production_uses_real_problem_and_accepted_callbacks(scalar, monkeypatch, budget):
    from pathlib import Path
    import importlib

    example = Path(__file__).resolve().parents[1]/"examples/single-stage-benchmarks"
    monkeypatch.syspath_prepend(str(example))
    entry = importlib.import_module("free_boundary_single_stage_optimization_scalar")
    monkeypatch.setattr(entry, "PARAMETER_BOUND", .01)
    p, stats, *_ = scalar
    p.enable_matrix_free()
    initial = p.fun(p.x0)
    anchors = []
    result = entry.run_optimizer(SimpleNamespace(problem=p), SimpleNamespace(accepted_steps=budget),
        lambda: anchors.append((p.accepted.parameters.copy(), p.fun(p.accepted.parameters))),
        method="L-BFGS-B")
    assert anchors and p.accepted_step == len(anchors) <= budget
    assert anchors[-1][1] < initial
    np.testing.assert_array_equal(p.accepted.parameters, result.x)
    assert np.max(np.abs(result.x/p.scales)) <= .01 + 1e-14
    assert len(stats["seeds"]) == 1
    if budget == 1:
        assert not result.success and p.accepted_step == 1
    else:
        assert result.success


def test_lbfgsb_failed_equilibrium_keeps_accepted_root(scalar, monkeypatch):
    from pathlib import Path
    import importlib

    example = Path(__file__).resolve().parents[1]/"examples/single-stage-benchmarks"
    monkeypatch.syspath_prepend(str(example))
    entry = importlib.import_module("free_boundary_single_stage_optimization_scalar")
    p, stats, *_ = scalar
    p.enable_matrix_free()
    anchor = p.accepted
    stats["fail"] = True
    result = entry.run_optimizer(SimpleNamespace(problem=p), SimpleNamespace(accepted_steps=20),
                            lambda: pytest.fail("failed trial promoted"), method="L-BFGS-B")
    assert not result.success and result.stop_reason == "equilibrium_trial_rejected"
    np.testing.assert_array_equal(result.x, anchor.parameters)
    assert p.accepted is anchor and p.accepted_step == 0


@pytest.mark.parametrize("budget", [1, 5])
def test_shared_slsqp_promotes_only_certified_points(scalar, budget):
    p, stats, *_ = scalar
    seen = []
    constraints = p.nonlinear_constraint([-1e6, -1e6], [1e6, 1e6])
    result = opt.minimize(p, method="SLSQP", constraints=constraints,
        callback=lambda x: seen.append(x.copy()), options={"maxiter": budget, "ftol": 1e-10})
    assert 0 < p.accepted_step == len(seen) <= budget
    np.testing.assert_array_equal(p.accepted.parameters, result.x)
    for x in seen:
        assert np.all(np.isfinite(x))


def test_scalar_coil_current_alias_rejects_ambiguous_names(analytic):
    p, *_ = analytic
    with pytest.raises(ValueError, match="not both"):
        opt.FreeBoundaryProblem.from_loss(p.inp, lambda *a: 0.,
            coil_current_dofs=(), current_dofs=())


def test_coil_quantity_includes_direct_and_moving_equilibrium_derivatives(analytic):
    p, stats, state, derivative = analytic
    q = opt.FreeBoundaryProblem.from_loss(p.inp, lambda s, rt, c: jnp.sum(s*s),
        quantities=(lambda s, rt: s[0],),
        coil_quantities=(lambda s, rt, c: jnp.dot(s, c) + 2*s[1],),
        parameterization=p.parameterization, continuation=p.cfg)
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
        constraint = q.nonlinear_constraint([0.,0.], [np.inf,np.inf], scales=[.2, .4])
        np.testing.assert_allclose(constraint.jac(x), jac/np.array([.2,.4])[:,None])
        assert q.accepted_step == 0
        assert stats['rhs'][-1] == 3
    finally:
        q.close()


def test_coil_quantity_must_be_scalar(analytic):
    p, *_ = analytic
    with pytest.raises(ValueError, match='coil quantities must be scalar'):
        opt.FreeBoundaryProblem.from_loss(p.inp, lambda s, rt, c: jnp.sum(s),
            coil_quantities=(lambda s, rt, c: c,),
            parameterization=p.parameterization, continuation=p.cfg)


def test_coil_quantity_must_be_callable(analytic):
    p, *_ = analytic
    with pytest.raises(TypeError, match="callable"):
        opt.FreeBoundaryProblem.from_loss(p.inp, lambda s, rt, c: jnp.sum(s),
            coil_quantities=(1,), parameterization=p.parameterization, continuation=p.cfg)


def test_qualification_records_absent_optional_analysis_dependencies(monkeypatch, tmp_path):
    from importlib import metadata
    deck=tmp_path/'input';deck.write_text('vacuum')
    original=metadata.version
    def version(name):
        if name in ('booz_xform_jax','virtual-casing-jax'):
            raise metadata.PackageNotFoundError(name)
        return original(name)
    monkeypatch.setattr(metadata,'version',version)
    result=opt.OptimizationQualification.signature(parameters={},input_path=deck)
    assert result['dependencies']['booz_xform_jax'] is None
    assert result['dependencies']['virtual-casing-jax'] is None
    assert result['dependencies']['jax']==jax.__version__


def test_qualification_requires_core_dependency_metadata(monkeypatch, tmp_path):
    from importlib import metadata
    deck = tmp_path / "input"
    deck.write_text("vacuum")
    original = metadata.version

    def version(name):
        if name == "solvax":
            raise metadata.PackageNotFoundError(name)
        return original(name)

    monkeypatch.setattr(metadata, "version", version)
    with pytest.raises(metadata.PackageNotFoundError, match="solvax"):
        opt.OptimizationQualification.signature(parameters={}, input_path=deck)
