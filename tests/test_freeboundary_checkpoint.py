"""Checkpoint authentication, exact restoration, and spectral input parity."""
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace as NS
import hashlib
import numpy as np
import jax.numpy as jnp
import pytest

from vmex import VmecInput
from vmex.core import _freeboundary_checkpoint as storage
from vmex.core import freeboundary_problem as api
from vmex.core.coil_parameters import CoilParameters
from vmex.core.solver import SpectralState
from vmex.core.restart import restart_state

DATA = Path(__file__).resolve().parent / 'data/free_boundary_qa'


@pytest.fixture
def saved_problem(monkeypatch, tmp_path):
    inp = VmecInput.from_file(DATA/'input.rotating_ellipse')
    class Chart(CoilParameters):
        def __call__(self, x):
            return x
    chart = Chart(np.zeros((4, 3, 9)), np.ones(4), current_dofs=(1, 2, 3))
    state = SpectralState(*(jnp.ones((3, 2))*i for i in range(1, 7)))
    mask = SpectralState(*(jnp.ones((3, 2)) for _ in range(6)))
    root = NS(parameters=chart.x0, state=state, dof_mask=mask,
              rcon0=jnp.ones((3, 4, 2)), zcon0=jnp.zeros((3, 4, 2)),
              result=NS(fedge=0.))
    solver = NS(implicit=NS(inp=inp, ftol=1e-11, max_iterations=12000), resolution=None,
                edge_force_tolerance=1e-11, include_edge_in_convergence=True,
                adjoint_solver='forward_dense_jax', adjoint_fail='error', field_from_parameters=chart)
    monkeypatch.setattr(api.im, 'runtime_from_params', lambda *a: None)
    monkeypatch.setattr(api.im, 'params_from_input', lambda *a: None)
    monkeypatch.setattr(api.fbi, 'make_free_boundary_config', lambda *a, **kw: solver)
    def certify(solver, params, point, *, state, rcon0, zcon0, **kw):
        new_root = NS(parameters=np.asarray(point), state=state, dof_mask=mask,
                      rcon0=rcon0, zcon0=zcon0, result=NS(fedge=0.))
        return NS(solver=solver, params=params, _anchor=new_root, parameter_scales=chart.scales)
    monkeypatch.setattr(api.fc, 'make_free_boundary_continuation_config_from_state', certify)
    from vmex.core import freeboundary
    root.result.converged = True
    root.result.state = state
    monkeypatch.setattr(freeboundary, '_solve_free_boundary_stage', lambda *a, **kw: root)
    terms = [(lambda s, rt: s.R_cos, 0., 1.)]
    identity = {'objective':'test', 'optimizer':{'max_trials':6}}
    problem = api.FreeBoundaryProblem.from_tuples(inp, terms, parameterization=chart,
                                               checkpoint_identity=identity)
    monkeypatch.setattr(freeboundary, '_solve_free_boundary_stage', lambda *a, **kw: pytest.fail('resume launched a solve'))
    problem.accepted_step = 17
    problem.initial_gradient_norm = 2.75
    record = problem.save_checkpoint(tmp_path/'checkpoint.npz')
    return problem, record, terms, identity


def test_exact_resume_without_equilibrium_solve(saved_problem):
    old, record, terms, identity = saved_problem
    restored = api.FreeBoundaryProblem.from_tuples(old.inp, terms, parameterization=old.parameterization,
        checkpoint=record['path'], checkpoint_sha256=record['sha256'], checkpoint_identity=identity)
    assert restored.accepted_step == 17 and restored.initial_gradient_norm == 2.75
    assert restored.loss_scale == old.loss_scale
    for name in storage.FIELDS:
        np.testing.assert_array_equal(getattr(restored.accepted.state, name), getattr(old.accepted.state, name))
        np.testing.assert_array_equal(getattr(restored.accepted.dof_mask, name), getattr(old.accepted.dof_mask, name))
    np.testing.assert_array_equal(restored.accepted.rcon0, old.accepted.rcon0)
    np.testing.assert_array_equal(restored.accepted.zcon0, old.accepted.zcon0)
    with pytest.raises(FileExistsError):
        restored.save_checkpoint(record['path'])


def test_hash_and_changed_objective_fail_before_construction(saved_problem):
    old, record, terms, identity = saved_problem
    for digest, context, message in [('0'*64,identity,'SHA256'),
                                     (record['sha256'],{'objective':'changed'},'settings differ')]:
        with pytest.raises(ValueError, match=message):
            api.FreeBoundaryProblem.from_tuples(old.inp, terms, parameterization=old.parameterization,
                checkpoint=record['path'], checkpoint_sha256=digest, checkpoint_identity=context)


@pytest.mark.parametrize('field,value', [('R_cos',np.full((3,2),np.nan)),
    ('mask_R_cos',np.full((3,2),.5)),('loss_scale',np.array(-1.)),('accepted_step',np.array(-1)),
    ('zcon0',np.zeros((2,2))), ('initial_gradient_norm',np.asarray('-1'))])
def test_malformed_checkpoint_is_rejected(saved_problem, field, value):
    old, record, _, _ = saved_problem
    path = Path(record['path'])
    with np.load(path) as z:
        data = {k:z[k] for k in z.files}
    data[field] = value
    np.savez(path, **data)
    with pytest.raises(ValueError):
        storage.read(path, hashlib.sha256(path.read_bytes()).hexdigest(), old.checkpoint_identity)


def test_certification_cannot_change_saved_state(saved_problem, monkeypatch):
    old, record, terms, identity = saved_problem
    original = api.fc.make_free_boundary_continuation_config_from_state
    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        result._anchor.rcon0 = result._anchor.rcon0 + 1e-10
        return result
    monkeypatch.setattr(api.fc,'make_free_boundary_continuation_config_from_state',changed)
    with pytest.raises(ValueError, match='changed during certification'):
        api.FreeBoundaryProblem.from_tuples(old.inp,terms,parameterization=old.parameterization,
            checkpoint=record['path'],checkpoint_sha256=record['sha256'],checkpoint_identity=identity)


def test_normal_input_and_lossless_seed(tmp_path):
    inp = VmecInput.from_file(DATA/'input.rotating_ellipse')
    assert (inp.ntheta,inp.nzeta,inp.mpol,inp.ntor,list(inp.ns_array)) == (48,40,3,3,[31])
    assert inp.ftol_array[0] == 1e-11
    inp.to_json(tmp_path/'input.json')
    other = VmecInput.from_file(tmp_path/'input.json')
    for f in fields(inp):
        np.testing.assert_equal(getattr(inp,f.name),getattr(other,f.name))
    state = restart_state(DATA/'initial_state.npz',inp,ns=31)
    with np.load(DATA/'initial_state.npz') as seed:
        for name in storage.FIELDS:
            np.testing.assert_array_equal(getattr(state,name),seed[name])
    assert hashlib.sha256((DATA/'initial_state.npz').read_bytes()).hexdigest() == 'bbc25674652973ccdb2f0e1305216af6016a84c6a4ce9775afe6ca479290d9f1'



def test_checkpoint_identity_requires_explicit_solver_construction(saved_problem):
    old, _, terms, identity = saved_problem
    with pytest.raises(ValueError, match="construction with solver_options"):
        api.FreeBoundaryProblem.from_tuples(old.inp, terms, parameterization=old.parameterization,
            continuation=old.cfg, checkpoint_identity=identity)
