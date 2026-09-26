"""Strict free-boundary force, adjoint and tangent regression checks."""
import dataclasses
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from vmex.core import freeboundary_implicit as fbi
from vmex.core.errors import AdjointSolveError

def test_explicit_tangent_gate_rejects_default_slack(monkeypatch):
    cfg=SimpleNamespace(adjoint_tol=2e-5,adjoint_gcrot_m=3,adjoint_gcrot_k=1,adjoint_maxiter=10)
    monkeypatch.setattr(fbi,'_tangent_gcrot_core',lambda *a,**k:(jnp.ones(3),jnp.array(1e-4),jnp.array(1.),jnp.array(2)))
    fbi._solve_gcrot_tangent(None,None,cfg) # unchanged default 2e-4
    diagnostics=[]
    with pytest.raises(AdjointSolveError):
        fbi._solve_gcrot_tangent(None,None,cfg,residual_rtol=2e-5,diagnostics=diagnostics)
    assert diagnostics[0]['accepted'] is False
    assert diagnostics[0]['tolerance']==2e-5


@pytest.mark.parametrize('nonlinear',[0.,.4])
def test_tangent_is_linear_response_without_equilibrium_solve(monkeypatch,nonlinear):
    from contextlib import nullcontext
    matrix=jnp.array([[2.,.3],[.1,3.]])
    coupling=jnp.array([[1.,2.],[.3,-1.]])
    warmed=[]
    def warm(params,cfg):
        warmed.append(cfg)
    def residual(z,p,f,base,r,zcon):
        assert warmed, 'runtime must be prepared before tracing'
        return matrix@z+nonlinear*z*z-coupling@f
    monkeypatch.setattr(fbi.im,'runtime_from_params',warm)
    monkeypatch.setattr(fbi.im,'_device_context',lambda *a:nullcontext())
    monkeypatch.setattr(fbi.im,'_device_pin',lambda cfg,values:values)
    monkeypatch.setattr(fbi,'_projected_residual',lambda *a:residual)
    monkeypatch.setattr(fbi.im,'_dof_projector',lambda *a:lambda x:x)
    monkeypatch.setattr(fbi,'_solve_free_boundary_stage',lambda *a,**k:pytest.fail('equilibrium solve'))
    cfg=SimpleNamespace(implicit=SimpleNamespace(adjoint_tol=1e-10,adjoint_gcrot_m=2,
        adjoint_gcrot_k=1,adjoint_maxiter=10),adjoint_residual_rtol=1e-10)
    direction=jnp.array([.2,-.1]); state=jnp.array([.1,-.2])
    field=jnp.linalg.solve(coupling,matrix@state+nonlinear*state*state)
    with jax.disable_jit(False),jax.checking_leaks():
        tangent=fbi.free_boundary_state_tangent(None,field,cfg,state,jnp.ones(2),direction,rcon0=None,zcon0=None)
    assert warmed == [cfg.implicit]
    expected=np.linalg.solve(matrix+jnp.diag(2*nonlinear*state),coupling@direction)
    np.testing.assert_allclose(tangent,expected,rtol=1e-9,atol=1e-12)


@pytest.mark.parametrize("edge",[1.1e-14,-1.,float('nan'),float('inf')])
def test_strict_edge_gate_rejects_without_changing_default(edge):
    from vmex.core.solver import _force_convergence
    args=(jnp.array(1e-16),jnp.array(1e-16),jnp.array(1e-16),jnp.array(edge),1e-14)
    assert bool(_force_convergence(*args))
    assert not bool(_force_convergence(*args,edge_tolerance=1e-14))


def test_strict_edge_gate_requires_vacuum_and_interior_channels():
    from vmex.core.solver import _force_convergence
    for i in range(3):
        channels=[1e-16]*3;channels[i]=2e-14
        assert not bool(_force_convergence(*channels,jnp.array(1e-16),1e-14,edge_tolerance=1e-14))
    assert not bool(_force_convergence(0.,0.,0.,jnp.array(0.),1e-14,edge_tolerance=1e-14,vacuum_active=False))
    assert bool(_force_convergence(1e-14,1e-14,1e-14,jnp.array(1e-14),1e-14,edge_tolerance=1e-14))


def test_strict_force_evaluation_discards_stale_normalization():
    from vmex.core.input import VmecInput
    from vmex.core.solver import prepare_runtime, resolution_from_input, _initial_state, evaluate_forces
    inp = VmecInput.from_file(Path(__file__).resolve().parent/'data/input.rotating_ellipse_m3n3_maxmode1_iota0p2')
    inp = dataclasses.replace(inp, ns_array=np.array([4]))
    rt = prepare_runtime(inp, resolution_from_input(inp))
    rt = dataclasses.replace(rt, lfreeb=True, jmax=4, presf_ns_scale=1.,
        bsqvac_edge=jnp.zeros((rt.resolution.ntheta3,rt.resolution.nzeta)))
    state = _initial_state(rt.setup)
    _, reference, diagnostics = evaluate_forces(state, rt, iteration=2)
    assert not bool(diagnostics.jacobian_sign_changed)
    stale = dataclasses.replace(diagnostics.cache,
        fnorm=diagnostics.cache.fnorm*.5, fnormL=diagnostics.cache.fnormL*.5)
    _, default, _ = evaluate_forces(state, rt, cache=stale, iteration=2)
    _, strict, _ = evaluate_forces(state, dataclasses.replace(rt, include_edge_in_convergence=True),
        cache=stale, iteration=2)
    for name in ('fsqr', 'fsqz', 'fsql', 'fedge'):
        np.testing.assert_allclose(getattr(strict,name), getattr(reference,name), rtol=1e-12, atol=1e-30)
    assert float(default.fedge) != float(reference.fedge)
