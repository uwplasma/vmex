"""Public accepted-state composition, seed lineage and optimizer integration."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vmex import optimize as opt
from vmex.core import implicit as imp
from vmex.core.errors import TrialRejected
from vmex.core.fourier import mode_table
from vmex.core.input import VmecInput
from vmex.core.solver import SpectralState


def source_problem():
    inp = VmecInput.from_file(Path(__file__).resolve().parents[1] /
                             "examples/single-stage-benchmarks/input.rotating_ellipse")
    inp = inp.change_resolution(mpol=2, ntor=0, ntheta=8, nzeta=8)
    inp = replace(inp, ns_array=np.array([3]))

    class Config:
        hot_restart = True
        ftol = 1e-12
        max_fsq_ratio = 100
        resolution = SimpleNamespace(ns=3)

    cfg = Config()
    cfg.inp = inp
    shape = (3, mode_table(inp.mpol, inp.ntor).mnmax)
    visits = []
    failed = set()

    def equilibrium(x):
        visits.append((float(x[0]), imp._HOT_CACHE.get(cfg)))
        if float(x[0]) in failed:
            raise RuntimeError("fixture corrector failed")
        state = SpectralState(*(np.full(shape, x[0]) for _ in range(6)))
        return SimpleNamespace(inp=inp, state=state,
            result=SimpleNamespace(converged=True, fsqr=1e-15, fsqz=1e-15, fsql=1e-15))

    def value_gradient(x):
        equilibrium(x)
        return float((x[0]-1)**2), np.array([2*(x[0]-1)])

    p = opt.VmecProblem([0.], value_and_grad=value_gradient,
        residual=lambda x: np.array([equilibrium(x).state.R_cos[0, 0]]),
        residual_jac=lambda x: equilibrium(x) and np.ones((1, 1)),
        input_from_x=lambda x: inp, x_from_input=lambda deck: np.zeros(1),
        boundary_from_x=lambda x: (inp.rbc, inp.zbs), equilibrium_from_x=equilibrium,
        metadata=dict(config=cfg, holder={"lin": "stale predictor"}), scales=[.3])
    return p, cfg, visits, failed


def test_every_host_trial_uses_accepted_seed_and_only_acceptance_advances_it():
    source, cfg, visits, _ = source_problem()
    p = source.with_accepted_state()
    anchor = p.accepted.equilibrium
    for function, point in ((p.fun, .3), (p.value_and_grad, .4), (p.residual, .5),
                            (p.residual_jac, .6), (p.equilibrium_from_x, .7)):
        imp._HOT_CACHE[cfg] = "rejected trial"
        imp._PERTURB_SEED[cfg] = "unaccepted predictor"
        function(np.array([point]))
        assert visits[-1][1] is anchor.state
        assert cfg not in imp._PERTURB_SEED
        assert source.metadata["holder"]["lin"] is None
        np.testing.assert_array_equal(p.accepted.parameters, [0.])
    p.accept_x([.4])
    assert p.accepted_step == 1
    np.testing.assert_array_equal(p.accepted.parameters, [.4])
    p.fun([.2])
    assert visits[-1][1] is p.accepted.equilibrium.state
    detached = p.accepted.parameters
    detached[0] = 99
    np.testing.assert_array_equal(p.accepted.parameters, [.4])
    p.accept_x([.4])
    assert p.accepted_step == 1
    assert p.with_accepted_state() is p
    with pytest.raises(NotImplementedError, match="before"):
        p.subproblem(active=[0])


def test_failed_promotion_preserves_equilibrium_and_generic_composed_anchor():
    source, _, _, failed = source_problem()
    p = source.with_accepted_state()
    joint = opt.FunctionProblem.from_functions([0., 0.],
        value_and_grad=lambda x: (float(x @ x), 2*x)).with_acceptance(lambda x: p.accept_x(x[:1]))
    anchor = p.accepted.equilibrium
    failed.add(.3)
    # Prime the gradient before a corrector failure during promotion.
    p._vg_cache = p._key(np.array([.3])), (0.49, np.array([-1.4]))
    with pytest.raises(TrialRejected, match="corrector failed"):
        joint.accept_x([.3, .5])
    assert p.accepted.equilibrium is anchor and joint.accepted_step == p.accepted_step == 0
    np.testing.assert_array_equal(joint.accepted.parameters, [0., 0.])


def test_force_gate_and_invalid_restart_never_replace_anchor():
    source, cfg, _, _ = source_problem()
    p = source.with_accepted_state()
    anchor = p.accepted.equilibrium
    original = source._equilibrium_from_x

    def bad(x):
        eq = original(x)
        eq.result = SimpleNamespace(converged=False, fsqr=1., fsqz=0., fsql=0.)
        return eq

    source._equilibrium_from_x = bad
    with pytest.raises(TrialRejected, match="force gate"):
        p.accept_x([.2])
    assert p.accepted.equilibrium is anchor and p.accepted_step == 0
    with pytest.raises(ValueError, match="nfp"):
        source.restart_from(SimpleNamespace(inp=replace(anchor.inp, nfp=3), state=anchor.state))
    invalid = replace(anchor.state, R_cos=np.full_like(anchor.state.R_cos, np.nan))
    with pytest.raises(ValueError, match="finite"):
        source.restart_from(SimpleNamespace(inp=anchor.inp, state=invalid))
    cfg.hot_restart = False
    with pytest.raises(ValueError, match="hot_restart"):
        source.restart_from(anchor)


@pytest.mark.parametrize("method", ["BFGS", "L-BFGS-B", "SLSQP"])
def test_shared_minimizer_uses_composed_acceptance_and_physical_constraints(method):
    source, _, _, _ = source_problem()
    plasma = source.with_accepted_state()

    def value_gradient(x):
        value, gradient = plasma.value_and_grad(x[:1])
        return value+(x[1]-2)**2, np.r_[gradient, 2*(x[1]-2)]

    joint = opt.FunctionProblem.from_functions([0., 0.], scales=[.3, 2.],
        value_and_grad=value_gradient).with_acceptance(lambda x: plasma.accept_x(x[:1]))
    records = []
    constraints = ()
    if method == "SLSQP":
        quantities = opt.FunctionProblem.from_functions([0., 0.],
            residual=lambda x: x[:1], residual_jac=lambda x: np.array([[1., 0.]]))
        constraints = quantities.nonlinear_constraint([.2], [np.inf], scales=[.1])
    result = opt.minimize(joint, method=method, constraints=constraints,
        callback=lambda x: records.append((x.copy(), plasma.accepted.parameters.copy())),
        options=dict(maxiter=30, **({"ftol": 1e-12} if method != "BFGS" else {"gtol": 1e-10})))
    assert result.success
    np.testing.assert_allclose(result.x, [1., 2.], atol=1e-6)
    assert result.accepted_steps == joint.accepted_step == len(records)
    for point, accepted_boundary in records:
        np.testing.assert_array_equal(point[:1], accepted_boundary)
    assert not hasattr(source, "accept_x")  # Legacy stateless instances are unchanged.


@pytest.mark.parametrize("method", ["L-BFGS-B", "SLSQP"])
def test_budget_stop_and_nonfinite_proposals_keep_last_accepted_point(method):
    p = opt.FunctionProblem.from_functions([3.],
        value_and_grad=lambda x: (float((x[0]-1)**4), 4*(x-1)**3)).with_acceptance(lambda x: None)
    result = opt.minimize(p, method=method, options=dict(maxiter=1))
    assert result.stop_reason == "accepted_step_budget_reached"
    assert p.accepted_step == result.accepted_steps == 1
    np.testing.assert_array_equal(result.x, p.accepted.parameters)
    with pytest.raises(ValueError, match="finite"):
        p.accept_x([np.nan])
    with pytest.raises(ValueError, match="already owns"):
        p.with_acceptance(lambda x: None)


def test_public_factory_uses_restart_from_not_internal_initial_state(monkeypatch):
    from vmex.core import optimize as implementation

    source, _, _, _ = source_problem()
    eq = source.equilibrium_from_x([0.])
    captured = {}
    def construct(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(metadata={})
    monkeypatch.setattr(implementation, "_least_squares_implicit", construct)
    opt.VmecProblem.from_loss(eq.inp, lambda s, r: 0., restart_from=eq.state, warm_start="state")
    assert captured["initial_state"] is eq.state
    with pytest.raises(TypeError, match="initial_state"):
        opt.VmecProblem.from_loss(eq.inp, lambda s, r: 0., initial_state=eq.state)
