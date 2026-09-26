"""Coupled root refinement with real dense/FGMRES kernels on an analytic root."""
from dataclasses import dataclass
from types import SimpleNamespace
import weakref

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vmex import optimize as opt
from vmex.core import freeboundary_problem as api
from vmex.core import _freeboundary_root_polish as polish


@dataclass(frozen=True, eq=False)
class Root:
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
    anchor = Root(chart.x0.copy(), initial, mask, owner,
        float(jnp.linalg.norm(residual(initial*mask, params, chart.x0))),
        jnp.array([2.]), jnp.array([3.]), SimpleNamespace(iterations=5))
    cfg = SimpleNamespace(_owner=owner, _anchor=anchor, params=params, solver=solver,
        parameter_scales=chart.scales, continuation_step=.1, max_continuation_steps=64,
        root_residual_atol=1.)
    monkeypatch.setattr(polish.fbi, '_projected_residual', lambda *_: residual)
    monkeypatch.setattr(polish.im, '_dof_projector', lambda _, m: lambda x: x*m)
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
        return Root(np.asarray(point).copy(), state, mask, owner, norm, rcon0, zcon0, result)

    monkeypatch.setattr(api.fc, 'certify_free_boundary_continuation_state', certify)
    problem = opt.FreeBoundaryProblem.from_loss(inp,
        lambda s, rt, c: .5*jnp.sum(s[:2]**2)+jnp.sum(c*c),
        quantities=(lambda s, rt: s[0], lambda s, rt: s[1]),
        parameterization=chart, continuation=cfg)
    yield problem, residual
    problem.close()


@pytest.mark.usefixtures('_module_jit_enabled')
def test_polished_seed_and_total_gradient_use_actual_kernels(coupled):
    p, residual = coupled
    old = p.accepted
    old_gradient = p.grad(p.x0).copy()
    p.enable_root_polishing()
    assert p.accepted is not old and p.accepted_step == 0
    assert p.accepted.root_residual_norm <= 1e-12
    assert p.accepted.state[2] == 7.
    assert p.accepted.result.iterations == 5
    p.enable_matrix_free(np.array([.001, -.002]), refresh_horizon=10)
    z = p.accepted.state
    matrix = jax.jacfwd(residual, 0)(z*jnp.array([1., 1., 0.]), p.params, jnp.zeros(2))[:2,:2]
    coupling = jax.jacfwd(residual, 2)(z*jnp.array([1., 1., 0.]), p.params, jnp.zeros(2))[:2]
    expected = np.asarray(z[:2]) @ -np.linalg.solve(matrix, coupling)
    np.testing.assert_allclose(p.grad(p.x0), expected, rtol=1e-10, atol=1e-12)
    assert not np.allclose(old_gradient, expected)


@pytest.mark.usefixtures('_module_jit_enabled')
def test_polish_failure_keeps_initial_root_and_caches(coupled):
    p, _ = coupled
    original = p.accepted
    grad = p.grad(p.x0).copy()
    with pytest.raises(polish.RootPolishError, match='budget exhausted'):
        p.enable_root_polishing(tolerance=1e-14, max_steps=1)
    assert p.accepted is original and p._root_polish_options is None
    np.testing.assert_array_equal(p.grad(p.x0), grad)


@pytest.mark.usefixtures('_module_jit_enabled')
def test_polished_coil_quantity_matches_independent_endpoints(coupled, monkeypatch):
    """A moving-surface-like row retains both direct and root-response terms."""
    from vmex.core import freeboundary
    p, _ = coupled
    q = opt.FreeBoundaryProblem.from_loss(p.inp, lambda s, rt, c: jnp.sum(s[:2]**2),
        coil_quantities=(lambda s, rt, c: s[0]*c[0] + s[1] + 2*c[1],),
        parameterization=p.parameterization, continuation=p.cfg)
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


@pytest.mark.usefixtures('_module_jit_enabled')
def test_trial_polishing_preserves_accepted_seed_and_recertifies(coupled, monkeypatch):
    from vmex.core import freeboundary
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


@pytest.mark.usefixtures('_module_jit_enabled')
def test_polished_fd_trial_must_keep_requested_force_tolerance(coupled, monkeypatch):
    from vmex.core import freeboundary
    p, _ = coupled
    p.enable_root_polishing()
    p.enable_matrix_free()
    anchor = p.accepted
    monkeypatch.setattr(freeboundary, '_solve_free_boundary_stage', lambda inp, **kw:
        SimpleNamespace(result=SimpleNamespace(state=kw['initial_state'], converged=True),
                        rcon0=anchor.rcon0, zcon0=anchor.zcon0))
    with pytest.raises(api.TrialRejected, match='requested force tolerance'):
        p.evaluate_trial(np.array([.001, -.002]), predict=False, ftol=1e-24)
    assert p.accepted is anchor


@pytest.mark.parametrize('tolerance,steps', [(0.,3),(float('nan'),3),(1e-12,0),(1e-12,True)])
def test_invalid_polishing_contract(coupled, tolerance, steps):
    p, _ = coupled
    with pytest.raises(ValueError, match='positive finite'):
        p.enable_root_polishing(tolerance=tolerance, max_steps=steps)


def test_one_dense_retry_releases_temporary_factors(monkeypatch):
    calls = []
    class Temporary:
        def close(self):
            calls.append('close')
    def refine(record, cfg, seed, **kwargs):
        calls.append('refine')
        if seed == 'old':
            raise polish.RootPolishError('stale LU')
        return 'polished', {'seconds': .1}
    monkeypatch.setattr(polish, 'refine', refine)
    result = polish.polish_with_recovery('root','config','old',
        lambda root:(Temporary(),Temporary()), lambda event:None)
    assert result == 'polished' and calls == ['refine','close','refine','close']


def test_failed_dense_retry_is_bounded_and_releases_factors(monkeypatch):
    calls = []
    class Temporary:
        def close(self):
            calls.append('close')
    def fail(*args, **kwargs):
        calls.append('refine')
        raise polish.RootPolishError('root did not improve')
    monkeypatch.setattr(polish, 'refine', fail)
    with pytest.raises(polish.RootPolishError, match='did not improve'):
        polish.polish_with_recovery('root','config','old',
            lambda root:(Temporary(),Temporary()), lambda event:None)
    assert calls == ['refine','close','refine','close']


@pytest.mark.parametrize('error_type', [polish.RootPolishError, polish.AdjointSolveError])
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

    monkeypatch.setattr(polish, 'refine', refine)
    assert polish.polish_with_recovery('root', 'cfg', 'old', build_dense, events.append) == 'root'
    assert seed.closed
    assert events[0]['failure'] == 'linear solve failed'
    assert events[-1]['recovered_with_dense']


@pytest.mark.usefixtures('_module_jit_enabled')
def test_bootstrap_releases_dense_before_real_refinement(coupled, monkeypatch):
    p, _ = coupled
    pullback = api.fc.free_boundary_continuation_state_pullback
    refine = polish.refine
    dense = []

    def capture(*args, **kwargs):
        result = pullback(*args, **kwargs)
        dense.append(result)
        return result

    def checked(*args, **kwargs):
        assert dense[-1]._root is None
        return refine(*args, **kwargs)

    monkeypatch.setattr(api.fc, 'free_boundary_continuation_state_pullback', capture)
    monkeypatch.setattr(polish, 'refine', checked)
    p.enable_root_polishing()
    assert p.accepted.root_residual_norm <= 1e-12


@pytest.mark.usefixtures('_module_jit_enabled')
def test_real_dense_recovery_keeps_accepted_root_and_seed(coupled, monkeypatch):
    p, _ = coupled
    p.enable_root_polishing()
    p.enable_matrix_free()
    anchor, seed = p.accepted, p._preconditioner
    gradient = p.grad(p.x0).copy()
    events = []
    monkeypatch.setattr(p, '_emit', lambda name, **data: events.append((name, data)))
    point = np.array([.002, -.003])
    candidate = api.fc.certify_free_boundary_continuation_state(p.cfg, point, anchor.state,
        rcon0=anchor.rcon0, zcon0=anchor.zcon0)
    solve = polish.mf.solve
    calls = []

    def stale_once(action, template, space, factors, rhs, **options):
        calls.append(options)
        if len(calls) == 1:
            return jnp.zeros_like(rhs), 300, jnp.linalg.norm(rhs), False
        return solve(action, template, space, factors, rhs, **options)

    monkeypatch.setattr(polish.mf, 'solve', stale_once)
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


@pytest.mark.usefixtures('_module_jit_enabled')
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
    candidate = api.fc.certify_free_boundary_continuation_state(p.cfg, point, anchor.state,
        rcon0=anchor.rcon0, zcon0=anchor.zcon0)
    real_refine = polish.refine

    def stale(record, cfg, seed, **options):
        if seed is p._preconditioner:
            raise polish.RootPolishError('stale accepted LU')
        return real_refine(record, cfg, seed, **options)

    with monkeypatch.context() as patch:
        patch.setattr(polish, 'refine', stale)
        refined = p._polish_record(candidate)
    p._records[p._key(point)] = refined
    assert p._trial_lu.record is refined
    return refined


@pytest.mark.usefixtures('_module_jit_enabled')
def test_recovery_seed_used_at_polished_root_and_promoted_only_on_accept(coupled, monkeypatch):
    p, residual = coupled
    candidate = recovered_trial(p, monkeypatch)
    anchor, old_seed = p.accepted, p._preconditioner
    trial_seed = p._trial_lu.seed
    dense_seconds = p._trial_lu.seconds
    pullback = api.fc.free_boundary_continuation_state_pullback
    seen = []

    def checked(record, cfg, rhs, **options):
        seen.append(record)
        assert record is candidate  # new operator at the polished root
        assert options['preconditioner'] is trial_seed
        return pullback(record, cfg, rhs, **options)

    monkeypatch.setattr(api.fc, 'free_boundary_continuation_state_pullback', checked)
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


@pytest.mark.usefixtures('_module_jit_enabled')
@pytest.mark.parametrize('abandon', ['new_trial', 'close'])
def test_abandoned_recovery_seed_is_released(coupled, monkeypatch, abandon):
    from vmex.core import freeboundary
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


@pytest.mark.usefixtures('_module_jit_enabled')
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


@pytest.mark.usefixtures('_module_jit_enabled')
def test_post_polish_force_rejection_releases_recovery_seed(coupled, monkeypatch):
    from vmex.core import freeboundary
    p, _ = coupled
    p.enable_root_polishing()
    p.enable_matrix_free()
    anchor, old_seed = p.accepted, p._preconditioner
    real_refine = polish.refine
    seeds = []

    def stale(record, cfg, seed, **options):
        if seed is old_seed:
            raise polish.RootPolishError('stale accepted LU')
        seeds.append(seed)
        return real_refine(record, cfg, seed, **options)

    monkeypatch.setattr(polish, 'refine', stale)
    monkeypatch.setattr(freeboundary, '_solve_free_boundary_stage', lambda inp, **kw:
        SimpleNamespace(result=SimpleNamespace(state=kw['initial_state'], converged=True),
                        rcon0=anchor.rcon0, zcon0=anchor.zcon0))
    with pytest.raises(api.TrialRejected, match='requested force tolerance'):
        p.evaluate_trial(np.array([.001, -.002]), predict=False, ftol=1e-24)
    assert len(seeds) == 1 and seeds[0]._seed is None and p._trial_lu is None
    assert p.accepted is anchor and p._preconditioner is old_seed and old_seed._seed is not None


@pytest.mark.usefixtures('_module_jit_enabled')
@pytest.mark.parametrize('dense_fails', [False, True])
def test_trial_seed_adjoint_failure_still_has_one_dense_fallback(coupled, monkeypatch, dense_fails):
    p, _ = coupled
    candidate = recovered_trial(p, monkeypatch)
    anchor, old_seed, trial_seed = p.accepted, p._preconditioner, p._trial_lu.seed
    pullback = api.fc.free_boundary_continuation_state_pullback
    calls = []

    def fail_seed(record, cfg, rhs, **options):
        seed = options.get('preconditioner')
        calls.append(seed)
        if seed is not None or dense_fails:
            raise polish.AdjointSolveError('injected solve failure')
        return pullback(record, cfg, rhs, **options)

    monkeypatch.setattr(api.fc, 'free_boundary_continuation_state_pullback', fail_seed)
    if dense_fails:
        with pytest.raises(polish.AdjointSolveError, match='injected'):
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


@pytest.mark.parametrize('failure', ['report', 'handoff'])
def test_failed_recovery_handoff_releases_seed(monkeypatch, failure):
    closed = []
    class Seed:
        def close(self):
            closed.append('seed')
    class Dense:
        def close(self):
            closed.append('dense')
    def refine(record, cfg, seed, **options):
        if seed == 'old':
            raise polish.RootPolishError('stale')
        return 'polished', {'seconds': 0.}
    def report(data):
        if failure == 'report' and data['event'] == 'polished':
            raise RuntimeError('report failed')
    def handoff(*args):
        raise RuntimeError('handoff failed')
    monkeypatch.setattr(polish, 'refine', refine)
    with pytest.raises(RuntimeError, match=failure):
        polish.polish_with_recovery('root', 'cfg', 'old', lambda _: (Dense(), Seed()),
                                   report, retain_recovery=handoff)
    assert closed == ['dense', 'seed']
