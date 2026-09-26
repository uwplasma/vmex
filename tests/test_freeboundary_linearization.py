"""One accepted-root factorization supplies checked adjoints and tangents."""
from types import SimpleNamespace as NS
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np
import pytest
from vmex.core import freeboundary_implicit as fbi, freeboundary_continuation as fc, implicit as im

pytestmark = pytest.mark.usefixtures('_module_jit_enabled')


def fixture(monkeypatch):
    matrix=jnp.array([[3.,2.],[-1.,4.]])
    coupling=jnp.array([[1.,-2.],[3.,1.]])
    residual=jax.jit(lambda z,p,f,*_:matrix@z+coupling@f-p)
    cfg=NS(implicit=NS(device=None,lconm1=False,adjoint_tol=1e-11,adjoint_maxiter=10,adjoint_gcrot_m=2,adjoint_gcrot_k=1),
        adjoint_dense_batch_size=2,adjoint_dense_max_dofs=10,adjoint_solver='forward_dense_jax',
        adjoint_fail='error',adjoint_residual_rtol=2e-5)
    monkeypatch.setattr(fbi,'_projected_residual',lambda *_:residual)
    monkeypatch.setattr(im,'_dof_projector',lambda _,mask:lambda x:x*mask)
    monkeypatch.setattr(im,'runtime_from_params',lambda *_:None)
    owner=object();state=jnp.zeros(2)
    accepted=NS(_owner=owner,parameters=np.zeros(2),state=state,dof_mask=jnp.ones(2),rcon0=None,zcon0=None)
    continuation=NS(_owner=owner,params=jnp.zeros(2),solver=cfg,root_residual_atol=2e-6)
    return accepted,continuation,matrix,coupling


def test_shared_factorization_orientation_and_scaling(monkeypatch):
    accepted,cfg,a,b=fixture(monkeypatch)
    calls=[];original=jsl.lu_factor
    def measured(x):calls.append(1);return original(x)
    monkeypatch.setattr(jsl,'lu_factor',measured)
    rows=[]
    root=fc.free_boundary_continuation_state_pullback(accepted,cfg,jnp.eye(2),diagnostics=rows, return_linearization=True)
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
    accepted,cfg,_,_=fixture(monkeypatch)
    root=fc.free_boundary_continuation_state_pullback(accepted,cfg,jnp.eye(2), return_linearization=True)
    root._root.factors=jsl.lu_factor(jnp.eye(2))
    rows=[]
    with pytest.raises(Exception,match='reused_dense_lu'):
        root.tangent(accepted,cfg,jnp.ones(2),diagnostics=rows)
    assert rows and not rows[0]['accepted']


def test_stale_and_closed_roots_fail_without_fallback(monkeypatch):
    accepted,cfg,_,_=fixture(monkeypatch)
    root=fc.free_boundary_continuation_state_pullback(accepted,cfg,jnp.eye(2), return_linearization=True)
    for other,c in ((NS(**vars(accepted)),cfg),(accepted,NS(**vars(cfg)))):
        with pytest.raises(ValueError,match='different accepted root'):
            root.tangent(other,c,jnp.ones(2))
    root.close();root.close()
    assert root.field_jacobian is None
    with pytest.raises(ValueError,match='closed'):
        root.tangent(accepted,cfg,jnp.ones(2))


def test_zero_invalid_directions_and_jit_rejection(monkeypatch):
    accepted,cfg,_,_=fixture(monkeypatch)
    root=fc.free_boundary_continuation_state_pullback(accepted,cfg,jnp.eye(2), return_linearization=True)
    np.testing.assert_array_equal(root.tangent(accepted,cfg,jnp.zeros(2)),np.zeros(2))
    for direction in (np.ones(3),np.array([np.nan,0.]),np.array([1j,0j])):
        with pytest.raises(ValueError):
            root.tangent(accepted,cfg,direction)
    with jax.disable_jit(False), pytest.raises(ValueError,match='host-eager'):
        jax.jit(lambda d:root.tangent(accepted,cfg,d))(jnp.ones(2))


@pytest.mark.parametrize('backend,fail', [(name, 'best_effort') for name in fbi._ADJOINT_SOLVERS])
def test_unsupported_policy_cannot_fall_back(monkeypatch,backend,fail):
    accepted,cfg,_,_=fixture(monkeypatch)
    cfg.solver.adjoint_solver=backend;cfg.solver.adjoint_fail=fail
    with pytest.raises(ValueError,match='retained linearization requires'):
        fc.free_boundary_continuation_state_pullback(accepted,cfg,jnp.eye(2), return_linearization=True)


def test_new_root_refactors_and_preserves_root_gate(monkeypatch):
    accepted,cfg,_,_=fixture(monkeypatch)
    accepted.state=jnp.ones(2)
    with pytest.raises(ValueError,match='root residual'):
        fc.free_boundary_continuation_state_pullback(accepted,cfg,jnp.eye(2), return_linearization=True)
    accepted.state=jnp.zeros(2)
    first=fc.free_boundary_continuation_state_pullback(accepted,cfg,jnp.eye(2), return_linearization=True)
    other=NS(**vars(accepted))
    second=fc.free_boundary_continuation_state_pullback(other,cfg,jnp.eye(2), return_linearization=True)
    assert first._root is not second._root
    first.close()
    assert np.all(np.isfinite(second.tangent(other,cfg,jnp.ones(2))))


@pytest.mark.parametrize("backend", fbi._ADJOINT_SOLVERS)
def test_all_backends_share_owned_gradient_and_predictor_interface(monkeypatch, backend):
    accepted, cfg, matrix, coupling = fixture(monkeypatch)
    cfg.solver.adjoint_solver = backend
    residual = jax.jit(lambda z, p, f, *_: matrix @ z + coupling @ f - p)
    raw = jax.jit(lambda z, p, f, *_: 7 * residual(z, p, f))
    monkeypatch.setattr(fbi, "_projected_residual",
        lambda *a, **k: raw if k.get("formulation") == "raw" else residual)
    calls = []
    def schur(solver, z, p, field, frozen, rcon, zcon, mask, rhs, **kwargs):
        calls.append(kwargs["row"])
        return jnp.linalg.solve(7 * matrix.T, rhs)
    monkeypatch.setattr(fbi, "_host_boundary_schur_adjoint", schur)
    monkeypatch.setattr(fbi, "_edge_response", lambda *a: object())
    monkeypatch.setattr(fbi, "_prepare_response_transpose", lambda z, p, f, base, rc, zc, *a, **k:
        jax.vjp(lambda zz: residual(zz, p, f, base, rc, zc), z)[1])
    root = fc.free_boundary_continuation_state_pullback(accepted, cfg, jnp.eye(2), return_linearization=True)
    np.testing.assert_allclose(root.field_jacobian, -np.linalg.solve(matrix, coupling), atol=1e-11)
    if backend == "boundary_schur":
        assert calls == [0, 1]  # Uses the upstream Schur entry, with raw parameter derivatives.
    root.offload_factors()
    reports = []
    direction = jnp.array([.2, -.3])
    np.testing.assert_allclose(root.tangent(accepted, cfg, direction, diagnostics=reports),
                              -np.linalg.solve(matrix, coupling @ direction), atol=1e-11)
    expected = "reused_dense_lu" if backend.startswith("forward_dense") else "gcrot_tangent"
    assert reports[-1]["backend"] == expected and reports[-1]["accepted"]
    if not backend.startswith("forward_dense"):
        with pytest.raises(ValueError, match="live dense"):
            root.preconditioner()
    with pytest.raises(ValueError, match="different accepted root"):
        root.tangent(NS(**vars(accepted)), cfg, direction)
    root.close()
    with pytest.raises(ValueError, match="closed"):
        root.tangent(accepted, cfg, direction)
