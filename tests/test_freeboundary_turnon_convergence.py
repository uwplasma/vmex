"""A strict solve must not certify a different, pre-restart plasma state."""
import dataclasses
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from vmex.core import solver
from vmex.core.errors import MORE_ITER_FLAG, SUCCESSFUL_TERM_FLAG
from vmex.core.residuals import ForceResiduals, PreconditionedResiduals


def model(monkeypatch, strict, max_iterations=8):
    # The returned/restored state fails; the pre-restart evaluation state passes.
    state = solver.SpectralState(*(jnp.zeros((3, 1)) for _ in range(6)))
    passing = dataclasses.replace(state, R_cos=jnp.ones((3, 1)))
    runtime = SimpleNamespace(ftol=1e-10, max_iterations=max_iterations,
        gamma=0., lfreeb=True, include_edge_in_convergence=strict,
        edge_force_tolerance=1e-10, lmove_axis=False,
        resolution=SimpleNamespace(ns=3), prec2d=None)
    fields = {f.name: jnp.asarray(0.) for f in dataclasses.fields(solver._LoopCarry)}
    fields.update(state=state, xstore=state, xcdot=state, cache=jnp.asarray(0.),
        time_step=jnp.asarray(.1), inv_tau=jnp.ones(solver.NDAMP),
        iteration=jnp.asarray(2), iter1=jnp.asarray(2), ijacob=jnp.asarray(1),
        done=jnp.asarray(False), ier=jnp.asarray(0),
        res0=jnp.asarray(jnp.inf), res1=jnp.asarray(jnp.inf),
        trajectory=jnp.zeros((max_iterations, solver._TRAJ_COLS)))
    carry = solver._LoopCarry(**fields)
    pipeline = solver.ForcePipelineHealth(**{
        f.name:jnp.asarray(True) for f in dataclasses.fields(solver.ForcePipelineHealth)})
    health = solver.NumericalHealth(**{
        f.name:pipeline if f.name=='pipeline' else jnp.asarray(True)
        for f in dataclasses.fields(solver.NumericalHealth)})
    def evaluate(current, cache, *args, **kwargs):
        edge = jnp.where(current.R_cos[0,0] > .5, 8e-11, 2e-10)
        zero = jnp.asarray(0.)
        return solver._EvalResult(gc=state,
            residuals=ForceResiduals(zero,zero,zero,edge,zero,zero,zero),
            pre=PreconditionedResiduals(zero,zero,zero),
            wb=zero,wp=zero,r00=zero,z00=zero,
            jacobian_sign_changed=jnp.asarray(False),cache=cache,health=health)
    monkeypatch.setattr(solver, '_evaluate', evaluate)
    return runtime,carry,passing


@pytest.mark.parametrize('compiled', [False, True])
@pytest.mark.parametrize('hoisted', [False, True])
def test_strict_turnon_waits_for_returned_state_evaluation(monkeypatch, compiled, hoisted):
    rt,carry,passing = model(monkeypatch, True)
    body=solver._make_body(rt,evaluation_state=passing,
        evaluation_synthesis=() if hoisted else None)
    with jax.disable_jit(not compiled):
        result=jax.jit(body)(carry) if compiled else body(carry)
        # A passing residual at the other state must not finish this solve.
        assert not bool(result.done)
        assert int(result.iteration)==3
        normal=solver._make_body(rt)
        failing=normal(result)
        assert not bool(failing.done)
        assert float(failing.fedge)>rt.edge_force_tolerance
        # A later ordinary pass can finish once the actual state passes.
        accepted=normal(dataclasses.replace(failing,state=passing))
        assert bool(accepted.done) and int(accepted.ier)==SUCCESSFUL_TERM_FLAG
        assert float(accepted.fedge)<=rt.edge_force_tolerance
        np.testing.assert_array_equal(accepted.state.R_cos,passing.R_cos)


def test_turnon_at_iteration_budget_is_not_success(monkeypatch):
    rt,carry,passing=model(monkeypatch,True,max_iterations=2)
    result=solver._make_body(rt,evaluation_state=passing)(carry)
    assert bool(result.done) and int(result.ier)==MORE_ITER_FLAG


def test_legacy_turnon_behavior_is_unchanged(monkeypatch):
    rt,carry,passing=model(monkeypatch,False)
    result=solver._make_body(rt,evaluation_state=passing)(carry)
    assert bool(result.done) and int(result.ier)==SUCCESSFUL_TERM_FLAG
