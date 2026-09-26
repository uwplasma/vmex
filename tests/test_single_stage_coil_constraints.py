"""CPU geometry/derivative checks; no equilibrium or optimization campaign."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('JAX_ENABLE_X64', 'true')
os.environ.setdefault('VMEX_COMPILATION_CACHE', 'disabled')

from pathlib import Path
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("essos")

import importlib.util
import sys
EXAMPLE = Path(__file__).resolve().parents[1] / "examples/coil-constraints-benchmarks"
sys.path.insert(0, str(EXAMPLE.parent))
from single_stage_support.common import DATA, resize_coils
# Load this example's settings without depending on another test's module cache.
spec = importlib.util.spec_from_file_location("parameters", EXAMPLE / "parameters.py")
parameters = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parameters)
previous = sys.modules.get("parameters")
sys.modules["parameters"] = parameters
try:
    spec = importlib.util.spec_from_file_location("coil_constraint_geometry", EXAMPLE / "_coil_constraints.py")
    C = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(C)
finally:
    if previous is None:
        sys.modules.pop("parameters", None)
    else:
        sys.modules["parameters"] = previous

jax.config.update('jax_enable_x64', True)


def circle(radius=.5, height=0.):
    a=np.zeros((3,3));a[0,2]=radius;a[1,1]=radius;a[2,0]=height
    return a


def coils(raw):
    return SimpleNamespace(curves=SimpleNamespace(curves=jnp.asarray(raw)))


def test_circle_units_and_scale():
    for radius in (.5,1.,2.):
        _,speed,k=C.geometry(jnp.asarray(circle(radius)[None]),512)
        np.testing.assert_allclose(speed.mean(),2*np.pi*radius,rtol=1e-14)
        np.testing.assert_allclose(k,1/radius,rtol=1e-14)
        np.testing.assert_allclose(jnp.sum(k*k*speed)/jnp.sum(speed),1/radius**2,rtol=1e-14)


def test_msc_is_arc_length_weighted_not_parameter_average():
    a=circle();a[0,2]=1.
    _,v,k=C.geometry(jnp.asarray(a[None]),1024)
    actual=jnp.sum(k*k*v)/jnp.sum(v)
    t=np.arange(200000)*2*np.pi/200000
    speed=np.sqrt(np.sin(t)**2+.25*np.cos(t)**2)
    expected=np.sum((.5/speed**3)**2*speed)/np.sum(speed)
    np.testing.assert_allclose(actual,expected,rtol=1e-12)
    assert abs(float(actual-jnp.mean(k*k)))>.1


def test_segment_interior_intersection_and_parallel_distance():
    a=jnp.array([[-1.,0,0],[1.,0,0]])
    b=jnp.array([[0.,-1,0],[0.,1,0]])
    assert float(C.segment_distances(a,b).min())<1e-12
    np.testing.assert_allclose(C.segment_distances(a,a+jnp.array([0,0,2.])),2.,atol=1e-14)


def test_self_crossing_detected():
    bow=jnp.array([[-1.,-1,0],[1.,1,0],[-1.,1,0],[1.,-1,0]])
    cc,own=C.separations(jnp.stack([bow,bow+jnp.array([0.,0,3.])]))
    assert float(own)<1e-12
    np.testing.assert_allclose(cc,3.,atol=1e-13)


def test_geometry_directional_derivative_and_scaled_coordinates():
    a=circle();a[0,2]=.65;a[0,0]=.03;a[2,1]=.06
    direction=np.random.default_rng(6).normal(size=a.shape)*.07
    def rows(u):
        _,v,k=C.geometry(jnp.asarray(a[None])+u*jnp.asarray(direction[None]),512)
        return jnp.array([v.mean(),(k*k*v).sum()/v.sum(),k.max()])
    derivative=jax.jacfwd(rows)(.013)
    h=1e-6
    np.testing.assert_allclose(derivative,(rows(.013+h)-rows(.013-h))/(2*h),rtol=3e-7,atol=2e-8)


def test_clearance_derivative_includes_moving_surface():
    raw=jnp.asarray(circle()[None]);target=jnp.array([[.031,.012,1.17]])
    def distance(x):
        shifted=raw.at[:,2,0].add(x[0])
        surface=SimpleNamespace(gamma=target+jnp.array([0.,0.,x[1]]))
        return C.surface_distance(coils(shifted),surface)
    x=jnp.array([.1,.03]);grad=jax.grad(distance)(x)
    assert abs(float(grad[0]))>.5
    np.testing.assert_allclose(grad[0],-grad[1],rtol=1e-13)
    h=1e-6
    for i in range(2):
        d=jnp.eye(2)[i]*h
        np.testing.assert_allclose(grad[i],(distance(x+d)-distance(x-d))/(2*h),rtol=1e-8)


def test_saved_coils_use_metres_and_match_essos_geometry():
    from essos.coils import Coils
    obj=Coils.from_json(str(DATA / 'coils.fitted.json'))
    xyz,_,k=C.geometry(obj.curves.curves,obj.n_segments)
    np.testing.assert_allclose(xyz,obj.gamma,atol=1e-12)
    np.testing.assert_allclose(k,obj.curvature,rtol=1e-12)
    assert xyz.shape[0]==12


def test_independent_verifier_and_continuous_clearance():
    from essos.surfaces import SurfaceRZFourier
    # Axisymmetric torus R=1, minor radius .2. Coaxial circle R=1.5
    # at z=.5 has analytic distance sqrt(.5**2+.5**2)-.2.
    surface=SurfaceRZFourier(jnp.array([1.,.2]),jnp.array([0.,.2]),1,1,0,
                            ntheta=32,nphi=31,close=False)
    raw=np.stack([circle(1.5,.5),circle(.5,3.),circle(.5,5.)])
    report=C.verify(coils(raw),surface)
    np.testing.assert_allclose(report['refined_coil_surface_distance_m'],np.sqrt(.5)-.2,atol=1e-8)
    np.testing.assert_allclose(report['coils'][0]['msc_per_m2'],1/1.5**2,rtol=1e-12)
    assert not report['checks']['length']
    assert report['checks']['clearance_refinement']
    assert report['checks']['self_intersection']
    assert not report['continuous_surface_clearance_certified']


def test_all_coil_rows_have_finite_scaled_jacobian():
    from essos.coils import Coils
    obj=Coils.from_json(str(DATA / 'coils.fitted.json'))
    obj=resize_coils(obj,16,256)
    x0=jnp.asarray(obj.dofs_curves).ravel()
    def from_x(x):
        return obj.with_dofs(jnp.r_[x,obj.dofs_currents])
    constrained=C.constraint(from_x)
    # Displace slightly to avoid an accidental tied extremum in the seed.
    direction=np.random.default_rng(14).normal(size=x0.size)*.05
    x=x0+.03*direction
    jac=constrained.jac(x)
    assert jac.shape==(14,len(x)) and np.all(np.isfinite(jac))
    h=1e-6
    fd=(constrained.fun(x+h*direction)-constrained.fun(x-h*direction))/(2*h)
    np.testing.assert_allclose(jac@direction,fd,rtol=2e-4,atol=2e-6)


def test_order16_padding_preserves_shape_currents_and_low_mode_scaling():
    from essos.coils import Coils
    seed=Coils.from_json(str(DATA / 'coils.fitted.json'))
    higher=resize_coils(seed,16,256)
    assert higher.order==16 and higher.n_segments==256
    assert higher.dofs_curves.size==297
    np.testing.assert_array_equal(higher.dofs_currents_raw,seed.dofs_currents_raw)
    np.testing.assert_array_equal(higher.dofs_curves[:,:,:11],seed.dofs_curves)
    np.testing.assert_array_equal(higher.dofs_curves[:,:,11:],0.)
    for actual,expected in zip(C.geometry(higher.curves.curves,1024),C.geometry(seed.curves.curves,1024)):
        np.testing.assert_allclose(actual,expected,rtol=1e-13,atol=1e-13)


def test_order16_field_quadrature_on_perturbed_seed():
    from essos.coils import Coils
    from essos.fields import BiotSavart
    seed=Coils.from_json(str(DATA / 'coils.fitted.json'))
    higher=resize_coils(seed,16,256)
    delta=np.random.default_rng(42).normal(size=higher.dofs_curves.shape)*1e-4
    higher=higher.with_dofs(jnp.r_[(higher.dofs_curves+delta).ravel(),higher.dofs_currents])
    fine=resize_coils(higher,16,512)
    phi=np.arange(21)*2*np.pi/21
    target=jnp.asarray(np.c_[np.cos(phi),np.sin(phi),.1*np.sin(2*phi)])
    a=jax.vmap(BiotSavart(higher).B)(target)
    b=jax.vmap(BiotSavart(fine).B)(target)
    np.testing.assert_allclose(a,b,rtol=1e-7,atol=1e-9)


@pytest.mark.parametrize("coefficient,expected_pass", [(1e6, True), (1e10, False)])
def test_free_qualification_requires_two_successive_refined_passes(tmp_path, coefficient, expected_pass):
    """A coarse failure must refine, while persistent mismatches still block reuse."""
    import ast
    import json
    source = (EXAMPLE / "verify_free_boundary_single_stage.py").read_text()
    function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "verify_problem")
    reports, calls, reuse = {}, [], []

    def write(path, value):
        reports[path.name] = json.loads(json.dumps(value))

    def evaluate(delta, **kwargs):
        assert kwargs == dict(predict=False, ftol=1e-20)
        calls.append(float(delta[0]))
        x = float(delta[0])
        return SimpleNamespace(root_residual_norm=1e-14,
            result=SimpleNamespace(fsqr=1e-22, fsqz=1e-22, fsql=1e-22, fedge=1e-22)), np.array([x*(1+coefficient*x*x)])

    anchor = SimpleNamespace(root_residual_norm=1e-13)
    problem = SimpleNamespace(accepted=anchor, x0=np.zeros(1),
        value_and_grad=lambda x: (0., np.ones(1)), constraint_jac=lambda x: np.empty((0, 1)),
        evaluate_trial=evaluate, enable_matrix_free=lambda *a, **k: reuse.append(True) or {"passed": True})
    stage = SimpleNamespace(problem=problem, chart=SimpleNamespace(scales=np.ones(1)),
        constraint_transform=np.empty((0, 0)), inequalities=lambda x: x,
        coil_constraint=SimpleNamespace(fun=lambda x: 2*x, jac=lambda x: np.array([[2.]])))
    namespace = dict(example=SimpleNamespace(write_json=write, MATRIXFREE_RTOL=1e-11,
        MATRIXFREE_RESTART=100, MATRIXFREE_MAX_CYCLES=3, MATRIXFREE_RHS_BATCH_SIZE=3),
        GRADIENT_STEPS=(3e-4, 1e-4, 3e-5, 1e-5), GRADIENT_RTOL=1e-3,
        GRADIENT_ATOL=1e-7,
        GRADIENT_CHECK_FTOL=1e-20, LINEARIZATION_PARITY_RTOL=1e-6)
    exec(compile(ast.Module(body=[function], type_ignores=[]), "qualification", "exec"), namespace)
    if expected_pass:
        namespace["verify_problem"](stage, tmp_path)
        assert reports["gradient_check.json"]["passed"]
        assert [r["passed"] for r in reports["gradient_check.json"]["checks"]] == [False, False, True, True]
        assert reuse == [True]
    else:
        with pytest.raises(RuntimeError, match="two successive"):
            namespace["verify_problem"](stage, tmp_path)
        assert not reports["gradient_check.json"]["passed"]
        assert reuse == []
    assert len(calls) == 8 and problem.accepted is anchor
