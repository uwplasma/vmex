"""Continuation ownership, rejection and derivative contracts."""
import dataclasses
from dataclasses import dataclass
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vmex.core import freeboundary_continuation as fc
from vmex.core import freeboundary_implicit as fbi, implicit as im
from vmex.core.errors import VmecError
from vmex.core.solver import SpectralState

pytestmark = pytest.mark.usefixtures('_module_jit_enabled')


@dataclass(frozen=True)
class Result:
    state: object
    converged: bool = True
    iterations: int = 3
    fsqr: float = 1e-15
    fsqz: float = 1e-15
    fsql: float = 1e-15
    output: object = None


@pytest.fixture
def family(monkeypatch):
    matrix = jnp.array([[2., .4], [-.3, 1.5]])
    controls = SimpleNamespace(device=None, inp=object(), ftol=1e-14,
        max_iterations=100, adjoint_tol=1e-11, adjoint_maxiter=20,
        adjoint_gcrot_m=6, adjoint_gcrot_k=1, lconm1=False)
    solver = SimpleNamespace(implicit=controls, resolution=object(),
        adjoint_fail='error', adjoint_solver='coupled_gcrot', field_from_parameters=lambda p:p,
        adjoint_dense_batch_size=3, adjoint_dense_max_dofs=100)
    calls = []
    mode = {'failure':None, 'shift':0.}
    def stage(inp, **kwargs):
        p = np.asarray(kwargs['external_field'])
        calls.append((p.copy(), kwargs['initial_state'], kwargs['constraint_continuation']))
        if mode['failure'] == 'exception' and p[0] >= .4:
            raise VmecError('synthetic failed trial')
        state = SpectralState(*(matrix @ p + mode['shift'] for _ in range(6)))
        result = Result(state, output=np.asarray(p).copy())
        if mode['failure'] == 'force':
            result = dataclasses.replace(result, fsqr=1e-8)
        if mode['failure'] == 'convergence':
            result = dataclasses.replace(result, converged=False)
        return SimpleNamespace(result=result, rcon0=jnp.array([.125]), zcon0=jnp.array([.25]))
    def residual(z, p, field, base, rcon, zcon):
        defect = 1. if mode['failure']=='residual' else 0.
        return jax.tree.map(lambda x:x-matrix@field+defect, z)
    monkeypatch.setattr(fbi, '_solve_free_boundary_stage', stage)
    monkeypatch.setattr(im, 'input_with_params', lambda inp,p:inp)
    monkeypatch.setattr(im, 'runtime_from_params', lambda *_:None)
    monkeypatch.setattr(im, '_dof_projector', lambda cfg,mask:lambda x:x)
    monkeypatch.setattr(fbi, '_projected_residual', lambda *_:residual)
    monkeypatch.setattr(fbi, '_linearization_from_stage', lambda c,p,s,**kw:(
        s.result.state, jax.tree.map(jnp.ones_like,s.result.state), s.rcon0, s.zcon0))
    shape = SpectralState(*(jax.ShapeDtypeStruct((2,),jnp.float64) for _ in range(6)))
    monkeypatch.setattr(im, '_state_struct', lambda _:shape)
    monkeypatch.setattr(im, '_callback_sharding', lambda _:None)
    monkeypatch.setattr(fbi, '_baseline_struct', lambda _:(
        jax.ShapeDtypeStruct((1,),jnp.float64),jax.ShapeDtypeStruct((1,),jnp.float64)))
    def make(**kwargs):
        options = dict(parameter_scales=np.array([2.,1.]), continuation_step=.1,
                       root_residual_atol=1e-10, cache_size=2)
        options.update(kwargs)
        return fc.make_free_boundary_continuation_config(solver, {'fixed':jnp.array(0.)},
            np.array([0.,0.]), **options)
    return SimpleNamespace(make=make, calls=calls, mode=mode, matrix=np.asarray(matrix))


def test_scaled_path_is_independent_of_trial_order_and_memo_is_bounded(family):
    cfg = family.make()
    a = fc.free_boundary_continuation_result(np.array([.4,.1]),cfg)
    assert a.continuation_steps == 2
    np.testing.assert_allclose([c[0] for c in family.calls],[[0,0],[.2,.05],[.4,.1]])
    assert family.calls[1][1] is cfg._anchor.state
    np.testing.assert_array_equal(family.calls[2][2][0],a.rcon0)
    before=len(family.calls)
    fc.free_boundary_continuation_result(np.array([.4,.1]),cfg)
    assert len(family.calls)==before
    fc.free_boundary_continuation_result(np.array([-.2,0.]),cfg)
    assert family.calls[-1][1] is cfg._anchor.state
    fc.free_boundary_continuation_result(np.array([0.,.1]),cfg)
    assert len(cfg._runtime.memo)==2
    fc.free_boundary_continuation_result(np.array([.4,.1]),cfg)
    assert len(family.calls)==before+4  # evicted endpoint is recomputed


@pytest.mark.parametrize('failure',['exception','force','convergence','residual'])
def test_failed_trial_or_refresh_preserves_anchor_and_accepted_endpoint(family,failure):
    cfg=family.make()
    p=np.array([.4,.1])
    accepted=fc.free_boundary_continuation_result(p,cfg)
    family.mode['failure']=failure
    with pytest.raises(VmecError):
        fc.free_boundary_continuation_result(p,cfg,force_recompute=True)
    family.mode['failure']=None
    before=len(family.calls)
    restored=fc.free_boundary_continuation_result(p,cfg)
    assert len(family.calls)==before
    np.testing.assert_array_equal(restored.state.R_cos,accepted.state.R_cos)
    np.testing.assert_array_equal(cfg.parameter_anchor,[0,0])
    assert fc.free_boundary_continuation_stats(cfg)['failures']==1


def test_validation_and_budget_reject_before_starting_any_path(family):
    cfg=family.make(max_continuation_steps=2,
        validate_parameters=lambda p:not (.09<p[0]<.11))
    before=len(family.calls)
    with pytest.raises(ValueError,match='validator'):
        fc.free_boundary_continuation_result(np.array([.2,.2]),cfg)
    with pytest.raises(ValueError,match='budget'):
        fc.free_boundary_continuation_result(np.array([1.,0.]),cfg)
    assert len(family.calls)==before


def test_memo_hits_still_validate_and_public_output_cannot_corrupt_memo(family):
    valid={'value':True}
    cfg=family.make(validate_parameters=lambda _:valid['value'])
    p=np.array([.2,0.])
    result=fc.free_boundary_continuation_result(p,cfg)
    result.result.output[0]=999
    np.testing.assert_array_equal(fc.free_boundary_continuation_result(p,cfg).result.output,p)
    valid['value']=False
    with pytest.raises(ValueError,match='validator'):
        fc.free_boundary_continuation_result(p,cfg)


def test_anchor_and_scales_are_immutable_and_reanchoring_is_explicit(family):
    scales=np.array([2.,1.]);cfg=family.make(parameter_scales=scales)
    scales[:]=99
    np.testing.assert_array_equal(cfg.parameter_scales,[2,1])
    with pytest.raises(ValueError):
        cfg.parameter_anchor.setflags(write=True)
    result=fc.free_boundary_continuation_result(np.array([.2,.1]),cfg)
    before=len(family.calls)
    new=fc.reanchor_free_boundary_continuation_config(cfg,result)
    assert len(family.calls)==before
    assert new._runtime is not cfg._runtime
    np.testing.assert_array_equal(cfg.parameter_anchor,[0,0])
    np.testing.assert_array_equal(new.parameter_anchor,[.2,.1])
    with pytest.raises(ValueError,match='different continuation'):
        fc.reanchor_free_boundary_continuation_config(new,result)


@pytest.mark.parametrize('point',[[1.],[[1.,2.]],[np.nan,0.],[1j,0.]])
def test_bad_target_is_rejected_before_solve(family,point):
    cfg=family.make();before=len(family.calls)
    with pytest.raises(ValueError):
        fc.free_boundary_continuation_result(point,cfg)
    assert len(family.calls)==before


@pytest.mark.parametrize('option,value',[
    ('continuation_step',0),('root_residual_atol',np.nan),
    ('parameter_scales',[1.,0.]),('max_continuation_steps',1.5),('cache_size',0)])
def test_invalid_controls_do_not_solve_anchor(family,option,value):
    with pytest.raises(ValueError):
        family.make(**{option:value})
    assert not family.calls


@pytest.mark.parametrize("backend", ["coupled_gcrot", "forward_dense", "forward_dense_jax"])
def test_scalar_custom_vjp_and_shared_pullback_match_analytic_derivative(family, backend):
    cfg=family.make();p=jnp.array([.2,.1])
    cfg.solver.adjoint_solver=backend
    def objective(point):
        state=fc.solve_free_boundary_continuation(point,cfg)
        return jnp.sum(state.R_cos**2)+3*point[0]
    expected=2*family.matrix.T@(family.matrix@np.asarray(p))+np.array([3.,0.])
    np.testing.assert_allclose(jax.grad(objective)(p),expected,rtol=1e-9,atol=1e-10)
    if backend.startswith("forward_dense"):
        with pytest.raises(ValueError, match="host-eager"):
            jax.jit(jax.grad(objective))(p)
    else:
        np.testing.assert_allclose(jax.jit(jax.grad(objective))(p),expected,rtol=1e-9,atol=1e-10)
    accepted=fc.free_boundary_continuation_result(p,cfg)
    rhs=SpectralState(jnp.eye(2),*(jnp.zeros((2,2)) for _ in range(5)))
    derivative=fc.free_boundary_continuation_state_pullback(accepted,cfg,rhs)
    np.testing.assert_allclose(derivative,family.matrix,rtol=1e-9,atol=1e-10)


def test_shared_pullback_uses_supplied_record_after_refresh(family,monkeypatch):
    cfg=family.make();p=np.array([.2,.1])
    saved=fc.free_boundary_continuation_result(p,cfg)
    # A new accepted memo record must not replace a previously saved VJP root.
    replacement=dataclasses.replace(cfg._runtime.memo[p.tobytes()],root_residual_norm=9e-11)
    cfg._runtime.memo[p.tobytes()]=replacement
    seen=[]
    monkeypatch.setattr(fbi,'free_boundary_state_pullback_multi_rhs',
        lambda params,point,solver,state,mask,rhs,**kw:(None,seen.append(state)))
    fc.free_boundary_continuation_state_pullback(saved,cfg,None)
    assert seen[0] is saved.state


def test_saved_scalar_backward_does_not_refetch_memo(family,monkeypatch):
    cfg=family.make();p=jnp.array([.2,.1])
    state,pb=jax.vjp(lambda point:fc.solve_free_boundary_continuation(point,cfg),p)
    monkeypatch.setattr(fc,'_root',lambda *_args,**_kw:pytest.fail('backward re-solved the forward root'))
    cotangent=jax.tree.map(jnp.ones_like,state)
    np.testing.assert_allclose(pb(cotangent)[0],6*family.matrix.T@np.ones(2),rtol=1e-9,atol=1e-10)


def test_public_exports():
    import vmex
    for name in fc.__all__:
        assert getattr(vmex,name) is getattr(fc,name)


def test_default_placement_keeps_saved_state_and_mask_immutable(family,monkeypatch):
    native=fbi._linearization_from_stage
    monkeypatch.setattr(fbi,'_linearization_from_stage',lambda *args,**kwargs:
        jax.tree.map(np.asarray,native(*args,**kwargs)))
    cfg=family.make()
    accepted=fc.free_boundary_continuation_result(np.array([0.,0.]),cfg)
    assert all(isinstance(x,jax.Array) for x in jax.tree.leaves(
        (accepted.state,accepted.dof_mask,accepted.rcon0,accepted.zcon0)))
    with pytest.raises(TypeError,match='immutable'):
        accepted.dof_mask.R_cos[0]=0.


def test_state_constructor_never_solves_and_preserves_state(family,monkeypatch):
    cfg=family.make()
    saved=cfg._anchor
    monkeypatch.setattr(fc,'_solve_point',lambda *a:pytest.fail('cold anchor solve'))
    seen=[]
    def certify(new, point, state, **kwargs):
        seen.append(state)
        return dataclasses.replace(saved,parameters=point,_owner=new._owner)
    monkeypatch.setattr(fc,'certify_free_boundary_continuation_state',certify)
    new=fc.make_free_boundary_continuation_config_from_state(cfg.solver,cfg.params,
        cfg.parameter_anchor,state=saved.state,rcon0=saved.rcon0,zcon0=saved.zcon0,
        continuation_step=.1,root_residual_atol=1e-10)
    assert seen[0] is saved.state
    assert new._anchor.state is saved.state
    fc.reanchor_free_boundary_continuation_config(new,new._anchor)
