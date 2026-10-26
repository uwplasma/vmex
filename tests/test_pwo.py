"""Independent lattice, mapping and Fourier oracles for pwO constructions."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vmex.core.pwo import (
    PwOActionFamilies,
    canonical_pwo_field,
    canonical_pwo_residual_from_boozer,
    mapped_pwo_residual_from_boozer,
    physical_pwo_action_from_boozer,
    squeezed_pwo_coordinates,
)

PARAMS = dict(iota=.7, nfp=3, bmin=1.1, bmax=1.9, t1=-.3, t2=3.)


def _lattice_union(theta, zeta, *, iota=.7, nfp=3, bmin=1.1, bmax=1.9,
                   t1=-.3, t2=3., p=2, radius=5):
    """Direct NumPy product on a larger unwrapped lattice than the implementation."""
    width = np.pi * iota * (1-t1*t2) / (nfp * (iota+t2))
    complement = np.ones(np.broadcast_shapes(np.shape(theta), np.shape(zeta)))
    for k in range(-radius, radius+1):
        for period in range(-radius, radius+1):
            th = theta - 2*np.pi*k
            ze = zeta - np.pi/nfp - 2*np.pi*period/nfp
            u, v = (ze+t1*th)/width, (th+t2*ze)/np.pi
            with np.errstate(over="ignore"):
                complement *= -np.expm1(-u**(2*p)-v**(2*p))
    return bmin+(bmax-bmin)*(1-complement)


def _map(gamma, eta, *, iota=.7, nfp=3, a=.3, b=.2, c=-2.):
    """Published LC two branches, evaluated with NumPy scalar arithmetic."""
    x = np.where(eta <= np.pi, eta, 2*np.pi-eta)
    distance = c*x*x-(1+c*np.pi)*x+np.pi
    y = np.where(eta <= np.pi, gamma, -gamma)+iota/nfp*distance
    shape = a*x*np.sin(y+b*np.sin(y))
    return np.broadcast_arrays(gamma, eta)[0], (np.pi+np.where(eta <= np.pi, -1, 1)*(shape+distance))/nfp


def _spectrum(ns=1):
    # The toroidal frequencies are physical: 3 and -6 for NFP=3.
    return dict(bmnc_b=np.tile([1.5, .08, -.06], (ns, 1)),
                bmns_b=np.tile([0., -.04, .025], (ns, 1)),
                xm_b=np.array([0, 1, 2]), xn_b=np.array([0, 3, -6]),
                iota_b=np.full(ns, .7), nfp=3)


def test_canonical_matches_independent_large_lattice_and_bounds():
    theta = np.linspace(-np.pi, np.pi, 17, endpoint=False)[:, None]+.037
    zeta = np.linspace(0, 2*np.pi/3, 19, endpoint=False)[None, :]+.019
    result = canonical_pwo_field(theta, zeta, **PARAMS)
    np.testing.assert_allclose(result["B_target"], _lattice_union(theta, zeta), atol=2e-14, rtol=2e-14)
    assert np.all(np.asarray(result["B_target"]) >= PARAMS["bmin"])
    assert np.all(np.asarray(result["B_target"]) <= PARAMS["bmax"])
    assert np.all(result["valid"])


def test_canonical_periodicity_symmetry_and_center_gradient():
    def field(theta, zeta):
        return canonical_pwo_field(theta, zeta, **PARAMS)["B_target"]
    theta, zeta = jnp.array([.13, 1.3, -2.9]), jnp.array([.04, 1.7, .9])
    np.testing.assert_allclose(field(theta+2*np.pi, zeta), field(theta, zeta), atol=2e-14)
    np.testing.assert_allclose(field(theta, zeta+2*np.pi/3), field(theta, zeta), atol=2e-14)
    np.testing.assert_allclose(field(-theta, 2*np.pi/3-zeta), field(theta, zeta), atol=2e-14)
    assert field(0., np.pi/3) == pytest.approx(PARAMS["bmax"])
    gradient = jax.grad(lambda x: field(x[0], x[1]))(jnp.array([0., np.pi/3]))
    np.testing.assert_allclose(gradient, 0., atol=1e-14)


def test_canonical_jvp_vjp_and_finite_difference_include_live_iota():
    theta, zeta = jnp.array([.31, -1.2, 2.7]), jnp.array([.39, 1.27, 1.8])
    def field(x):
        return canonical_pwo_field(theta, zeta, iota=x[0], nfp=3, bmin=x[1], bmax=x[2],
                                   t1=x[3], t2=x[4])["B_target"]
    x = jnp.array([.7, 1.1, 1.9, -.3, 3.])
    direction = jnp.array([.13, -.04, .06, .02, -.17])
    cotangent = jnp.array([.3, -.7, .5])
    value, tangent = jax.jit(lambda q: jax.jvp(field, (q,), (direction,)))(x)
    _, pullback = jax.vjp(field, x)
    np.testing.assert_allclose(jnp.vdot(cotangent, tangent), jnp.vdot(pullback(cotangent)[0], direction), atol=1e-13)
    errors = []
    for step in (1e-3, 1e-4, 1e-5, 1e-6):
        finite = (_lattice_union(np.asarray(theta), np.asarray(zeta), **dict(zip(
            ("iota", "bmin", "bmax", "t1", "t2"), np.asarray(x+step*direction)))))
        finite -= _lattice_union(np.asarray(theta), np.asarray(zeta), **dict(zip(
            ("iota", "bmin", "bmax", "t1", "t2"), np.asarray(x-step*direction))))
        errors.append(np.linalg.norm(finite/(2*step)-tangent))
    assert min(errors) < 1e-8
    assert np.isfinite(value).all() and np.linalg.norm(tangent) > 1e-4


@pytest.mark.parametrize("nfp,iota", [(2, .51), (3, -.72), (5, .94)])
def test_lc_mapping_against_independent_formula_and_jacobian(nfp, iota):
    gamma = np.linspace(0, 2*np.pi, 7, endpoint=False)[:, None]
    eta = np.array([2.8, 2.91, 3.01, 3.24, 3.4])[None, :]
    kwargs = dict(nfp=nfp, iota=iota, a=.3, b=.2, c=-2.)
    result = squeezed_pwo_coordinates(gamma, eta, **kwargs)
    theta, zeta = _map(gamma, eta, **kwargs)
    np.testing.assert_allclose(result["theta"], theta, atol=1e-14)
    np.testing.assert_allclose(result["zeta"], zeta, atol=1e-14)
    step = 1e-6
    finite = (_map(gamma, eta+step, **kwargs)[1]-_map(gamma, eta-step, **kwargs)[1])/(2*step)
    np.testing.assert_allclose(result["jacobian"], finite, rtol=2e-8, atol=2e-9)
    assert np.all(result["jacobian"] > 0)


def test_mapped_residual_matches_manual_sine_cosine_variance():
    kwargs = _spectrum(2)
    result = mapped_pwo_residual_from_boozer(**kwargs, b_ref=1.5, a=.3, b=.2,
                                             c=-2., eta_min=2.8, ngamma=12, neta=10)
    gamma = np.linspace(0, 2*np.pi, 12, endpoint=False)[:, None]
    eta = (2.8+(np.arange(10)+.5)*(2*np.pi-5.6)/10)[None, :]
    theta, zeta = _map(gamma, eta)
    phase = np.array([0, 1, 2])[:, None, None]*theta-np.array([0, 3, -6])[:, None, None]*zeta
    field = np.sum(np.array([1.5, .08, -.06])[:, None, None]*np.cos(phase)+
                   np.array([0., -.04, .025])[:, None, None]*np.sin(phase), axis=0)
    expected = np.mean(((field-field.mean(axis=0))/1.5)**2)
    assert float(result["total"]) == pytest.approx(expected, rel=2e-12)
    assert np.all(result["valid"])


def test_mapped_qi_limit_has_zero_residual_but_generic_field_does_not():
    spectra = dict(bmnc_b=np.array([[1.5, .2]]), bmns_b=np.array([[0., .1]]),
                   xm_b=np.array([0, 0]), xn_b=np.array([0, 3]), iota_b=np.array([.7]), nfp=3)
    control = mapped_pwo_residual_from_boozer(**spectra, b_ref=1.5, a=0.)
    assert float(control["total"]) < 1e-28
    broken = mapped_pwo_residual_from_boozer(**_spectrum(), b_ref=1.5, a=0.)
    assert float(broken["total"]) > 1e-4


def test_canonical_residual_grid_normalization_and_both_parities():
    kwargs = dict(**_spectrum(), bmin=1.1, bmax=1.9, t1=-.3, t2=3.)
    coarse = canonical_pwo_residual_from_boozer(**kwargs, ntheta=24, nzeta=28)
    fine = canonical_pwo_residual_from_boozer(**kwargs, ntheta=48, nzeta=56)
    assert float(coarse["total"]) == pytest.approx(float(fine["total"]), rel=.02)
    symmetric = canonical_pwo_residual_from_boozer(**dict(kwargs, bmns_b=None), ntheta=48, nzeta=56)
    assert abs(float(symmetric["total"])-float(fine["total"])) > 1e-5


@pytest.mark.parametrize("kwargs", [dict(nfp=0), dict(bmin=0.), dict(bmax=1.1),
                                     dict(iota=-3.), dict(t1=1/3), dict(p=1.5)])
def test_invalid_canonical_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        canonical_pwo_field(0.1, 0.2, **dict(PARAMS, **kwargs))


def _sharp_family_lengths(width):
    """Intersect straight field lines with lattice parallelograms; label by winding."""
    groups = {}
    iota, t1, t2, period = .7, -.3, 3., 2*np.pi/3
    for alpha in np.linspace(-np.pi, np.pi, 129, endpoint=False):
        intervals = []
        for toroidal in range(-5, 6):
            for poloidal in range(-5, 6):
                first = (toroidal*period-t1*(alpha-2*np.pi*poloidal))/(1+t1*iota)
                second = (t2*toroidal*period-alpha+2*np.pi*poloidal)/(iota+t2)
                lo = max(first-width/abs(1+t1*iota), second-np.pi/abs(iota+t2))
                hi = min(first+width/abs(1+t1*iota), second+np.pi/abs(iota+t2))
                if hi > lo:
                    intervals.append((lo, hi, toroidal, poloidal))
        intervals.sort()
        for left, right in zip(intervals[:-1], intervals[1:]):
            if abs(left[1]) < 8 and abs(right[0]) < 8:
                winding = (right[2]-left[2], right[3]-left[3])
                groups.setdefault(winding, []).append(right[0]-left[1])
    return groups


def test_sharp_limit_has_unequal_invariant_families_and_detects_broken_connection():
    width = float(canonical_pwo_field(.1, .2, **PARAMS)["width"])
    families = _sharp_family_lengths(width)
    assert set(families) == {(1, 0), (2, 1), (3, 1)}
    for values in families.values():
        assert np.ptp(values) < 2e-13
    first, second, third = [np.mean(families[key]) for key in ((1, 0), (2, 1), (3, 1))]
    assert first+second == pytest.approx(third, abs=2e-13)
    # At fixed reflecting levels J is a positive constant times these lengths.
    # Pooling unequal families would incorrectly penalize this exact pwO field.
    assert np.var(np.concatenate(list(families.values()))) > .1
    assert np.ptp(_sharp_family_lengths(1.03*width)[(1, 0)]) > 1e-3


@pytest.mark.parametrize("name", ["theta_center", "zeta_center"])
def test_nonfinite_target_centers_are_not_valid(name):
    with pytest.raises(ValueError):
        canonical_pwo_residual_from_boozer(**_spectrum(), bmin=1.1, bmax=1.9, **{name: np.nan})


def test_overflowing_surface_weights_cannot_zero_the_objective():
    kwargs = dict(**_spectrum(2), b_ref=1.5)
    regular = mapped_pwo_residual_from_boozer(**kwargs)
    try:
        huge = mapped_pwo_residual_from_boozer(**kwargs, weights=np.array([1e308, 1e308]))
    except ValueError:
        return  # Explicit rejection is also valid for this unsupported normalization.
    assert float(huge["total"]) == pytest.approx(float(regular["total"]), rel=1e-12)


def test_traced_invalid_target_reports_false_and_nonfinite_residual():
    def evaluate(center):
        result = canonical_pwo_residual_from_boozer(**_spectrum(), bmin=1.1, bmax=1.9,
                                                     theta_center=center, ntheta=8, nzeta=8)
        return result["valid"], result["residuals1d"]
    valid, residual = jax.jit(evaluate)(jnp.nan)
    assert not np.any(valid)
    assert not np.isfinite(residual).all()


def test_large_sharpness_is_differentiable_or_explicitly_unsupported():
    def field(iota):
        return canonical_pwo_field(.31, .39, **dict(PARAMS, iota=iota, p=128))["B_target"]
    try:
        derivative = jax.grad(field)(.7)
    except ValueError:
        return
    assert np.isfinite(derivative)


def test_adapters_forward_one_live_transform_and_preserve_diagnostics(monkeypatch):
    from types import SimpleNamespace
    from vmex.core import pwo
    calls = []
    def transform(state, runtime, **kwargs):
        calls.append((state, runtime, kwargs))
        return dict(_spectrum(), s_b=np.array([.5]))
    monkeypatch.setattr(pwo, "boozer_spectrum_state", transform)
    eq = SimpleNamespace(state=object(), runtime=object())
    for target in (pwo.CanonicalPwOResidual([.5], bmin=1.1, bmax=1.9, ntheta=8, nzeta=8),
                   pwo.MappedPwOResidual([.5], b_ref=1.5, ngamma=8, neta=8)):
        count = len(calls)
        report = target.compute_state(eq.state, eq.runtime)
        assert len(calls) == count+1
        assert calls[-1][0] is eq.state and calls[-1][1] is eq.runtime
        np.testing.assert_array_equal(calls[-1][2]["surfaces"], [.5])
        assert report["s_b"][0] == .5
        np.testing.assert_allclose(target(eq), report["residuals1d"])
        assert target.total(eq) == pytest.approx(float(report["total"]))
        np.testing.assert_allclose(target.J(eq), report["residuals1d"])


@pytest.mark.parametrize("kwargs", [dict(surfaces=[]), dict(surfaces=[[.5]]), dict(surfaces=[np.nan]),
    dict(surfaces=[0.]), dict(surfaces=[1.01]), dict(mboz=0), dict(nboz=-1), dict(oversample=0)])
def test_invalid_adapter_configuration(kwargs):
    from vmex.core.pwo import CanonicalPwOResidual
    with pytest.raises(ValueError):
        CanonicalPwOResidual(**(dict(surfaces=[.5], bmin=1.1, bmax=1.9) | kwargs))


@pytest.mark.parametrize("kwargs", [dict(weights=[0.]), dict(weights=[-1.]), dict(weights=[np.nan]),
    dict(weights=[1., 1.]), dict(bmnc_b=np.ones(3)), dict(bmns_b=np.ones((2, 3))),
    dict(xn_b=np.array([0, 1, 2])), dict(xm_b=np.array([0., 1.1, 2.])), dict(iota_b=np.array([.7,.8]))])
def test_invalid_spectrum_contract(kwargs):
    with pytest.raises(ValueError):
        mapped_pwo_residual_from_boozer(**dict(_spectrum(), **kwargs), b_ref=1.5)


@pytest.mark.parametrize("kwargs", [dict(eta_min=0.), dict(eta_min=np.pi), dict(eta_min=np.nan),
    dict(c=0.), dict(a=10.), dict(b_ref=0.), dict(ngamma=3), dict(neta=3)])
def test_invalid_map_is_rejected(kwargs):
    with pytest.raises(ValueError):
        mapped_pwo_residual_from_boozer(**_spectrum(), **(dict(b_ref=1.5) | kwargs))


def test_lc_shape_parameter_derivatives_against_independent_map():
    gamma = jnp.array([.2, 1.3, 3.4])
    eta = jnp.array([2.83, 3.02, 3.32])
    x = jnp.array([.7, .3, .2, -2.])
    direction = jnp.array([.1, -.04, .03, .08])
    def mapped(values):
        return squeezed_pwo_coordinates(gamma, eta, iota=values[0], a=values[1],
                                         b=values[2], c=values[3], nfp=3)["zeta"]
    _, tangent = jax.jit(lambda q: jax.jvp(mapped, (q,), (direction,)))(x)
    step = 1e-5
    def oracle(values):
        return _map(np.asarray(gamma), np.asarray(eta), iota=values[0], a=values[1],
                    b=values[2], c=values[3], nfp=3)[1]
    finite = (oracle(x+step*direction)-oracle(x-step*direction))/(2*step)
    np.testing.assert_allclose(tangent, finite, rtol=1e-7, atol=1e-10)


def test_locally_nonfolding_map_with_global_overlap_is_rejected():
    eta_min = np.pi-.5+1e-6
    # At gamma=0 both boundary shape terms increase the toroidal span.
    _, left = _map(np.array([0.]), np.array([eta_min]), b=0.)
    _, right = _map(np.array([0.]), np.array([2*np.pi-eta_min]), b=0.)
    assert float(right[0]-left[0]) > 2*np.pi/3
    with pytest.raises(ValueError):
        mapped_pwo_residual_from_boozer(**_spectrum(), b_ref=1.5, eta_min=eta_min)


REFERENCE = dict(iota=-.9,nfp=5,bmin=1.1,bmax=1.9,t1=.441421356,t2=-3.141421356)


def test_published_initial_tilt_crosses_cut_without_overlap():
    theta=np.linspace(-np.pi,np.pi,21)[:,None]+.013
    zeta=np.linspace(-np.pi/5,np.pi/5,23)[None,:]-.017
    out=canonical_pwo_field(theta,zeta,**REFERENCE)
    assert out['corner_margin']<0
    assert out['lattice_margin']>.45
    assert out['omitted_image_log_bound']<np.log(1e-150)
    np.testing.assert_allclose(out['B_target'],_lattice_union(theta,zeta,**REFERENCE,radius=9),atol=2e-14)
    # Independently enumerate a much larger lattice in linear coordinates.
    iota,t1,t2=REFERENCE['iota'],REFERENCE['t1'],REFERENCE['t2']
    width=np.pi*iota*(1-t1*t2)/(5*(iota+t2))
    distances=[max(abs(2*np.pi*(t1*j+k/5)/width),abs(2*j+2*t2*k/5))
               for j in range(-20,21) for k in range(-20,21) if j or k]
    assert float(out['lattice_margin'])==pytest.approx(1-2/min(distances))


@pytest.mark.parametrize('p',[2,8,128])
def test_crossing_field_periodicity_reflection_and_finite_ad(p):
    theta,zeta=jnp.array([.31,-2.9,2.71]),jnp.array([.39,-.62,.53])
    def field(x):
        return canonical_pwo_field(theta,zeta,iota=x[0],t1=x[1],t2=x[2],
                                   nfp=5,bmin=1.1,bmax=1.9,p=p)['B_target']
    x=jnp.array([-.9,.441421356,-3.141421356]);direction=jnp.array([.1,.02,-.1])
    def reference(values):
        return _lattice_union(np.asarray(theta),np.asarray(zeta),
                              **dict(REFERENCE,iota=values[0],t1=values[1],t2=values[2]),p=p,radius=9)
    value,tangent=jax.jit(lambda q:jax.jvp(field,(q,),(direction,)))(x)
    _,pullback=jax.vjp(field,x)
    cotangent=jnp.array([.3,-.4,.7])
    assert np.all(np.isfinite(tangent))
    tail_gradient=jax.grad(lambda q:canonical_pwo_field(.2,.3,iota=q,t1=REFERENCE['t1'],t2=REFERENCE['t2'],nfp=5,bmin=1.1,bmax=1.9,p=p)['omitted_image_log_bound'])(x[0])
    assert np.isfinite(tail_gradient)
    np.testing.assert_allclose(cotangent@tangent,pullback(cotangent)[0]@direction,atol=1e-12)
    np.testing.assert_allclose(value,reference(np.asarray(x)),atol=2e-14)
    shifted=canonical_pwo_field(theta+2*np.pi,zeta+2*np.pi/5,p=p,**REFERENCE)['B_target']
    np.testing.assert_allclose(shifted,value,atol=3e-14)
    reflected=canonical_pwo_field(-theta,zeta,iota=.9,t1=-REFERENCE['t1'],t2=-REFERENCE['t2'],nfp=5,bmin=1.1,bmax=1.9,p=p)['B_target']
    np.testing.assert_allclose(reflected,value,atol=2e-14)
    if p==2:
        h=1e-5
        plus=reference(np.asarray(x+h*direction))
        minus=reference(np.asarray(x-h*direction))
        np.testing.assert_allclose(tangent,(plus-minus)/(2*h),atol=1e-8)


def test_stencil_and_explicit_containment_rejections():
    with pytest.raises(ValueError,match='stencil'):
        canonical_pwo_field(.2,.3,images=1,**REFERENCE)
    spec=dict(bmnc_b=np.array([[1.5,.1]]),xm_b=np.array([0,1]),xn_b=np.array([0,5]),iota_b=np.array([-.9]),nfp=5,bmin=1.1,bmax=1.9,t1=REFERENCE['t1'],t2=REFERENCE['t2'])
    assert np.all(canonical_pwo_residual_from_boozer(**spec)['valid'])
    with pytest.raises(ValueError):
        canonical_pwo_residual_from_boozer(**spec,minimum_corner_margin=0.)
    with pytest.raises(ValueError):
        canonical_pwo_residual_from_boozer(**spec,minimum_lattice_margin=.9)


def test_touching_sharp_translates_are_invalid_eager_and_traced():
    # w2=pi and t1=0 make the j=1,k=0 translation touch at v=2.
    params=dict(iota=.7,nfp=3,bmin=1.1,bmax=1.9,t1=0.,t2=3.)
    with pytest.raises(ValueError):
        canonical_pwo_field(.2,.3,**params)
    out=jax.jit(lambda t:canonical_pwo_field(.2,.3,**{**params,'t1':t}))(jnp.array(0.))
    assert not out['valid']
    assert np.isnan(out['B_target'])


def test_overlapping_lattice_translates_are_rejected():
    params=dict(iota=.7,nfp=3,bmin=1.1,bmax=1.9,t1=-2.,t2=-2.5)
    with pytest.raises(ValueError):
        canonical_pwo_field(.2,.3,**params)
    out=jax.jit(lambda t:canonical_pwo_field(.2,.3,**{**params,'t1':t}))(jnp.array(-2.))
    assert out['lattice_margin']<0
    assert not out['valid']


def _action_spectrum(*, broken=False):
    """A nonconstant QS field, optionally with a second independent helicity."""
    return dict(bmnc_b=np.array([[1.5, .2, 0.]]),
                bmns_b=np.array([[0., 0., .04 if broken else 0.]]),
                xm_b=np.array([0, 1, 1]), xn_b=np.array([0, 0, 1]),
                iota_b=np.array([.7]), G_b=np.array([2.]),
                I_b=np.array([.3]), nfp=1)


@pytest.mark.parametrize("current_sign", [-1., 1.])
@pytest.mark.parametrize("points_per_period", [1023, 1024])
def test_physical_action_qs_limit_and_boozer_line_element(current_sign, points_per_period):
    from scipy.integrate import quad

    spectrum = _action_spectrum()
    spectrum.update(G_b=current_sign*spectrum["G_b"], I_b=current_sign*spectrum["I_b"])
    result = physical_pwo_action_from_boozer(
        **spectrum, levels=[1.5], alphas=[.2, .5, .8], check_roots=True, points_per_period=points_per_period)
    # Integrate in the QS angle independently of the root/coordinate mapping.
    # Both legs give 2*|G+iota*I|, and dtheta/dzeta=iota.
    integral = quad(lambda theta: np.sqrt(1-(1.5+.2*np.cos(theta))/1.5)
                    /(1.5+.2*np.cos(theta)), np.pi/2, 3*np.pi/2,
                    epsabs=1e-11, epsrel=1e-11)[0]
    expected = 2*abs(2.+.7*.3)/.7*integral
    assert result["counts"] == [[1], [1], [1]]
    np.testing.assert_allclose([row["J"] for row in result["rows"]], expected, rtol=2e-9)
    np.testing.assert_allclose([row["dJ"] for row in result["rows"]], 0., atol=2e-10)
    assert result["components"]["raw_Jalpha2"] < 1e-19
    assert len(result["root_checks"]) == 3
    assert result["family_invariance_qualified"] is False


def test_physical_action_broken_qs_derivative_matches_moving_bounce_difference():
    spectrum = _action_spectrum(broken=True)
    alpha, step = .4, 2e-5
    options = dict(levels=[1.48, 1.55], alphas=[alpha])
    result = physical_pwo_action_from_boozer(**spectrum, **options)
    plus = physical_pwo_action_from_boozer(**spectrum, **dict(options, alphas=[alpha+step]))
    minus = physical_pwo_action_from_boozer(**spectrum, **dict(options, alphas=[alpha-step]))
    assert result["counts"] == plus["counts"] == minus["counts"] == [[1, 1]]
    derivative = np.array([row["dJ"] for row in result["rows"]])
    finite = (np.array([row["J"] for row in plus["rows"]])
              - np.array([row["J"] for row in minus["rows"]]))/(2*step)
    assert np.isfinite(derivative).all() and np.linalg.norm(derivative) > .01
    np.testing.assert_allclose(derivative, finite, rtol=2e-6, atol=2e-8)
    assert result["components"]["raw_Jalpha2"] == pytest.approx(np.mean(derivative**2))
    matched = physical_pwo_action_from_boozer(**spectrum, **options, reference=result)
    assert len(matched["matches"]) == 2


def test_physical_action_stationary_roots_resolve_unsampled_narrow_barrier():
    shift, level = .0023, 1.09999999
    spectrum = dict(bmnc_b=[1., .1*np.cos(shift)],
                    bmns_b=[0., -.1*np.sin(shift)], xm_b=[0, 0], xn_b=[0, 1],
                    iota_b=.7, G_b=1., I_b=0., nfp=1)
    grid = np.linspace(-8*np.pi, 16*np.pi, 12*1024+1)
    sampled = 1.+.1*np.cos(grid-shift)
    assert np.max(sampled) < level
    assert not np.any((sampled[:-1]-level)*(sampled[1:]-level) < 0)
    result = physical_pwo_action_from_boozer(
        **spectrum, levels=[level], alphas=[0.], check_roots=True)
    assert result["counts"] == [[1]]
    row = result["rows"][0]
    offset = np.arccos((level-1.)/.1)
    np.testing.assert_allclose([row["left"], row["right"]],
                               [shift+offset, shift+2*np.pi-offset], atol=2e-10)
    assert np.isfinite(row["J"]) and row["J"] > 0.
    assert row["dJ"] == 0.
    assert result["root_checks"][0]["roots"] == 24


@pytest.mark.parametrize("invalid", [dict(levels=[]), dict(levels=[np.nan]),
    dict(alphas=[]), dict(G_b=[1., 2.]), dict(bmnc_b=[[1.5, .2, 0.], [1.5, .2, 0.]]),
    dict(levels=[1.7]), dict(xn_b=[0, 0, .5]), dict(G_b=0., I_b=0.),
    dict(bmnc_b=[[-1.5, .2, 0.]]), dict(levels=[2.])])
def test_physical_action_rejects_invalid_or_tangent_inputs(invalid):
    with pytest.raises(ValueError):
        physical_pwo_action_from_boozer(
            **(dict(_action_spectrum(), levels=[1.5], alphas=[.4]) | invalid))


@pytest.mark.parametrize("mismatch", ["grid", "counts", "critical", "rows", "endpoints"])
def test_physical_action_rejects_mismatched_reference(mismatch):
    from copy import deepcopy

    spectrum = _action_spectrum()
    options = dict(levels=[1.5], alphas=[.4])
    reference = deepcopy(physical_pwo_action_from_boozer(**spectrum, **options))
    if mismatch == "grid":
        reference["alphas"] = [.41]
    elif mismatch == "counts":
        reference["counts"] = [[2]]
    elif mismatch == "critical":
        reference["critical_census"][0]["count"] += 1
    elif mismatch == "rows":
        reference["rows"] = []
    else:
        # Preserve well counts while shifting both roots by >20% of the width.
        spectrum["bmnc_b"][0, 1] = .2*np.cos(1.)
        spectrum["bmns_b"][0, 1] = -.2*np.sin(1.)
    with pytest.raises(ValueError):
        physical_pwo_action_from_boozer(**spectrum, **options, reference=reference, match_critical_census=True)


def test_physical_action_extends_long_well_without_truncation():
    spectrum = _action_spectrum()
    spectrum["iota_b"] = [.05]
    options = dict(levels=[1.5], alphas=[np.pi/2-.05], check_roots=True)
    result = physical_pwo_action_from_boozer(**spectrum, **options)
    np.testing.assert_allclose([result["rows"][0][k] for k in ("left", "right")],
                               [1., 1.+np.pi/.05], atol=1e-10)
    assert result["critical_census"][0]["right_extent_periods"] == 16
    assert result["counts"] == [[1]]
    assert abs(result["rows"][0]["dJ"]) < 1e-8
    with pytest.raises(ValueError, match="Incomplete well"):
        physical_pwo_action_from_boozer(**spectrum, **options, max_periods=8)


@pytest.mark.parametrize("edge", [0., 2*np.pi])
def test_physical_action_follows_reference_well_across_sampling_seam(edge):
    spectrum = _action_spectrum()
    sign = 1. if edge == 0 else -1.
    options = dict(levels=[1.5], alphas=[np.pi/2-.7*(edge+sign*.001)])
    before = physical_pwo_action_from_boozer(**spectrum, **options)
    phase = sign*.0014
    spectrum["bmnc_b"][0, 1] = .2*np.cos(phase)
    spectrum["bmns_b"][0, 1] = -.2*np.sin(phase)
    after = physical_pwo_action_from_boozer(**spectrum, **options, reference=before, check_roots=True)
    assert before["counts"] == after["counts"] == [[1]]
    np.testing.assert_allclose(after["rows"][0]["left"], edge-sign*.001, atol=1e-12)
    np.testing.assert_allclose(after["rows"][0]["J"], before["rows"][0]["J"], rtol=1e-10)
    with pytest.raises(ValueError, match="No complete wells"):
        physical_pwo_action_from_boozer(**spectrum, **options)


@pytest.mark.parametrize("merge", [False, True])
def test_physical_action_rejects_reference_well_split_or_merge(merge):
    spectrum = dict(bmnc_b=[1.5, .2, .3 if merge else 0.],
                    xm_b=[0, 1, 2], xn_b=[0, 0, 0],
                    iota_b=1., G_b=1., I_b=0., nfp=1)
    options = dict(levels=[1.55], alphas=[0.])
    reference = physical_pwo_action_from_boozer(**spectrum, **options)
    spectrum["bmnc_b"][2] = 0. if merge else .3
    with pytest.raises(ValueError, match="well counts differ"):
        physical_pwo_action_from_boozer(**spectrum, **options, reference=reference)


@pytest.mark.parametrize("legacy", [False, True])
def test_physical_action_preserves_reference_normalization(legacy):
    spectrum = _action_spectrum(broken=True)
    options = dict(levels=[1.48, 1.55], alphas=[.4])
    first = physical_pwo_action_from_boozer(**spectrum, **options)
    anchors = [r["J"] for r in first["rows"]]
    if legacy:
        for row in first["rows"]:
            del row["reference_J"]
    spectrum["bmnc_b"][0, 0] -= .002
    second = physical_pwo_action_from_boozer(**spectrum, **options, reference=first)
    spectrum["bmnc_b"][0, 0] -= .002
    third = physical_pwo_action_from_boozer(**spectrum, **options, reference=second)
    assert [r["reference_J"] for r in third["rows"]] == anchors
    fixed = np.mean([(r["dJ"]/anchor)**2 for r, anchor in zip(third["rows"], anchors)])
    assert third["components"]["fixed_relative_Jalpha2"] == pytest.approx(fixed)
    assert not np.isclose(fixed, third["components"]["relative_Jalpha2"], rtol=1e-3)


@pytest.mark.parametrize("anchor", [0., -1., np.nan, np.inf])
def test_physical_action_rejects_invalid_reference_normalization(anchor):
    spectrum = _action_spectrum(broken=True)
    options = dict(levels=[1.5], alphas=[.4])
    reference = physical_pwo_action_from_boozer(**spectrum, **options)
    reference["rows"][0]["reference_J"] = anchor
    with pytest.raises(ValueError, match="Reference actions"):
        physical_pwo_action_from_boozer(**spectrum, **options, reference=reference)


@pytest.mark.parametrize("alpha", [-.7, .7])
@pytest.mark.parametrize("depth", [1e-8, 1e-10])
def test_physical_action_shallow_well_has_finite_harmonic_limit(alpha, depth):
    # B=1+0.2*cos(zeta)+0.02*cos(alpha), with iota=0.
    minimum = .8+.02*np.cos(alpha)
    row = physical_pwo_action_from_boozer(
        bmnc_b=[1., .2, .02], xm_b=[0, 0, 1], xn_b=[0, 1, 0],
        iota_b=0., G_b=1., I_b=0., nfp=1,
        levels=[minimum+depth], alphas=[alpha])["rows"][0]
    # The local harmonic well gives J=C*depth and dJ/dalpha=-C*B_alpha.
    coefficient = np.pi/(np.sqrt(.1)*minimum**1.5)
    assert row["J"] == pytest.approx(coefficient*depth, rel=3e-6)
    assert row["dJ"] == pytest.approx(coefficient*.02*np.sin(alpha), rel=1e-6)


@pytest.mark.parametrize("alpha", [-.7, .7])
@pytest.mark.parametrize("depth", [1e-8, 1e-10])
def test_action_kernel_polishes_shallow_bounces_without_clipping_energy(alpha, depth):
    from math import fsum
    from scipy.integrate import quad

    amplitude = .2
    level = .8+.02*np.cos(alpha)+depth
    # B=1-0.2*cos(zeta)+0.02*cos(alpha). This independent sine-square
    # substitution has exact finite transformed integrands at both bounces.
    gap = fsum((level, -1., amplitude, -.02*np.cos(alpha)))
    k = np.sqrt(gap/(2*amplitude))
    half = 2*np.arcsin(k)
    kernel = PwOActionFamilies._kernel(dict(bmnc_b=[1., -amplitude, .02],
        xm_b=[0, 0, 1], xn_b=[0, 1, 0], iota_b=0., G_b=1., I_b=0., nfp=1))
    # Represent root uncertainty from a few field-evaluation ulps, amplified
    # by the small bounce slope. The original endpoint residual is retained.
    error = 8*np.finfo(float).eps/abs(kernel.field(half, alpha)[2])
    left, right = -half-error, half+error
    def transformed(theta, derivative):
        co, si = np.cos(theta), np.sin(theta)
        B = level-gap*co*co
        jacobian = 1/np.sqrt(1-k*k*si*si)
        if derivative:
            return 4*.02*np.sin(alpha)*k/np.sqrt(gap/level)*(1-B/(2*level))/B**2*jacobian
        return 4*k*np.sqrt(gap/level)*co*co/B*jacobian
    for derivative in (False, True):
        expected = quad(lambda t: transformed(t, derivative), -np.pi/2, np.pi/2,
                        epsabs=1e-12, epsrel=1e-12)[0]
        actual = kernel.integral(alpha, left, right, level, derivative, extrema=[0.])[0]
        assert actual == pytest.approx(expected, rel=3e-6)
    with pytest.raises(ValueError, match="correction exceeds numerical uncertainty"):
        kernel.integral(alpha, left-.01, right, level, False)
    # At the well minimum the two bounces merge and the bounce slope vanishes.
    with pytest.raises(ValueError, match="Unresolved bounce-root polishing"):
        kernel.integral(alpha, 0., 2*np.pi, kernel.field(0., alpha)[0], False)


def _near_reflecting_long_well(depth, *, iota=-.0377, periods=12):
    """Generated three-mode well with an almost-reflecting interior maximum."""
    from scipy.optimize import brentq

    alpha = .7
    kernel = PwOActionFamilies._kernel(dict(bmnc_b=[1., .1, -.08],
        xm_b=[0, 0, 1], xn_b=[0, -2, 0], iota_b=iota, G_b=1., I_b=0., nfp=2))
    critical = brentq(lambda z: kernel.field(z, alpha)[2], -.02, .02, xtol=1e-13)
    level = kernel.field(critical, alpha)[0]+depth
    left = brentq(lambda z: kernel.field(z, alpha)[0]-level, -np.pi, -np.pi/2, xtol=1e-13)
    right = brentq(lambda z: kernel.field(z, alpha)[0]-level,
                   (periods-.5)*np.pi, periods*np.pi, xtol=1e-13)
    # The slow mode's slope is below .0031, while the fast mode has opposite
    # signs exceeding .0079 at these bracket endpoints around each extremum.
    extrema = [brentq(lambda z: kernel.field(z, alpha)[2],
                     k*np.pi/2-.02, k*np.pi/2+.02, xtol=1e-13) for k in range(-2, 2*periods+1)]
    return kernel, alpha, left, right, level, [z for z in extrema if left < z < right]


@pytest.mark.parametrize("depth", [1e-7, 1e-8, 1e-9])
def test_near_reflecting_action_matches_independent_extrema_panel_sums(monkeypatch, depth):
    from math import fsum
    import scipy.integrate

    kernel, alpha, left, right, level, extrema = _near_reflecting_long_well(depth)
    actual = [kernel.integral(alpha, left, right, level, derivative, extrema=extrema)
              for derivative in (False, True)]
    quad = scipy.integrate.quad
    def independently_partitioned(fun, lower, upper, *, points, epsabs, epsrel, limit, full_output):
        edges = [lower, *sorted(points), upper]
        parts = [quad(fun, a, b, epsabs=epsabs*(b-a)/(upper-lower),
                      epsrel=epsrel, limit=limit, full_output=full_output)
                 for a, b in zip(edges, edges[1:])]
        assert all(len(part) == 3 for part in parts), "Independent panel quadrature did not converge"
        total, error = (fsum(part[index] for part in parts) for index in (0, 1))
        # Signed dJ may cancel across panels; enforce the ORIGINAL global
        # tolerance as well as every local call's unchanged failure flag.
        assert error <= max(epsabs, epsrel*abs(total))
        return total, error, {}
    monkeypatch.setattr(scipy.integrate, "quad", independently_partitioned)
    # The kernel captures quad at construction, so rebuild after replacement.
    kernel, alpha, left, right, level, extrema = _near_reflecting_long_well(depth)
    for derivative, (value, error) in zip((False, True), actual):
        expected, _ = kernel.integral(alpha, left, right, level, derivative, extrema=extrema)
        assert error <= max(1e-9, 1e-8*abs(value))
        assert abs(value-expected) <= max(1e-9, 1e-8*abs(expected))


@pytest.mark.parametrize("accepted", [False, True])
def test_natural_width_retry_preserves_success_and_failure(monkeypatch, accepted):
    import scipy.integrate

    calls = []
    def fake_quad(fun, lower, upper, **options):
        calls.append(options)
        if accepted:
            return 1., 0., {}
        return 0., 0., {}, "roundoff prevents the requested tolerance"
    monkeypatch.setattr(scipy.integrate, "quad", fake_quad)
    kernel, alpha, left, right, level, extrema = _near_reflecting_long_well(1e-8)
    if accepted:
        assert kernel.integral(alpha, left, right, level, False, extrema=extrema) == (1., 0.)
        assert len(calls) == 1
    else:
        with pytest.raises(ValueError, match="roundoff prevents the requested tolerance"):
            kernel.integral(alpha, left, right, level, False, extrema=extrema)
        assert len(calls) == 2
        assert len(calls[1]["points"]) > len(calls[0]["points"])
    assert all(call["epsabs"] == 1e-9 and call["epsrel"] == 1e-8 and call["limit"] == 150 for call in calls)


def test_natural_width_retry_keeps_original_failure_above_point_budget(monkeypatch):
    import scipy.integrate

    calls = []
    def failed_quad(*args, **kwargs):
        calls.append(kwargs)
        return 0., 0., {}, "original quadrature failure"
    monkeypatch.setattr(scipy.integrate, "quad", failed_quad)
    kernel, alpha, left, right, level, extrema = _near_reflecting_long_well(
        1e-8, iota=-.009, periods=50)
    assert 100 <= len(extrema) < 150
    with pytest.raises(ValueError, match="original quadrature failure"):
        kernel.integral(alpha, left, right, level, False, extrema=extrema)
    assert len(calls) == 1


def _action_family(level=.81):
    spectrum = dict(bmnc_b=[1., .2, .02], bmns_b=[0., 0., 0.],
                    xm_b=[0, 0, 1], xn_b=[0, 1, 0], iota_b=0., G_b=1., I_b=0., nfp=1)
    nodes, weights = np.polynomial.legendre.leggauss(8)
    rows = []
    for u, weight in zip((nodes+1)/2, weights/2):
        lower = np.arccos((level-.8)/.02)
        alpha = lower+u*(2*np.pi-2*lower)
        width = np.arccos((1+.02*np.cos(alpha)-level)/.2)
        rows.append(dict(u=u, weight=weight, left=np.pi-width, right=np.pi+width))
    chart = dict(level_T=level, weight=2*np.pi-2*lower,
                 endpoints=[[lower, np.pi], [2*np.pi-lower, np.pi]], rows=rows)
    return spectrum, chart


def test_action_family_matches_independent_fixed_alpha_wells():
    spectrum, chart = _action_family()
    objective = PwOActionFamilies(spectrum, [chart], length=.4)
    result = objective.initial
    alphas = [r["alpha"] for r in result["families"][0]["rows"]]
    independent = physical_pwo_action_from_boozer(**spectrum, alphas=alphas, levels=[.81])
    J = np.array([r["J"] for r in independent["rows"]]); w = np.array([r["weight"] for r in chart["rows"]])
    np.testing.assert_allclose([r["J"] for r in result["families"][0]["rows"]], J, rtol=1e-10)
    assert result["value"] == pytest.approx(w@((J-w@J)**2)/.4**2, rel=1e-10)
    assert not result["family_invariance_qualified"] and not result["global_topology_proof"]


@pytest.mark.parametrize("angle", [.03, -.03])
def test_action_family_tracks_translated_events_and_preserves_reference(angle):
    from copy import deepcopy
    spectrum, chart = _action_family()
    objective = PwOActionFamilies(spectrum, [chart], length=.4)
    before = deepcopy(objective.initial)
    translated = dict(spectrum, bmnc_b=[1., .2*np.cos(angle), .02*np.cos(angle)],
                      bmns_b=[0., -.2*np.sin(angle), .02*np.sin(angle)])
    after = objective(translated)
    for key in ("value", "raw_Jalpha2", "fixed_relative_variance"):
        assert after[key] == pytest.approx(before[key], rel=1e-9, abs=1e-12)
    np.testing.assert_allclose(after["families"][0]["endpoints"], np.array(chart["endpoints"])+angle, atol=1e-10)
    assert objective.initial == before


@pytest.mark.parametrize("problem", ["weight", "nodes", "root", "tangency", "length", "empty", "reversed"])
def test_action_family_rejects_invalid_reference(problem):
    spectrum, chart = _action_family()
    if problem == "weight":
        chart["weight"] = -1
    if problem == "nodes":
        chart["rows"][0]["u"] = 0
    if problem == "root":
        chart["rows"][0]["left"] += .1
    if problem == "tangency":
        chart["endpoints"][0][0] += .1
    if problem == "reversed":
        chart["endpoints"].reverse()
    with pytest.raises(ValueError):
        PwOActionFamilies(spectrum, [] if problem == "empty" else [chart], length=0 if problem == "length" else .4)


@pytest.mark.parametrize("change", [{"nfp": 2}, {"bmnc_b": [1., .2, 0.]}, {"G_b": 0.}, {"s_b": [.5]}])
def test_action_family_rejects_changed_geometry_or_surface(change):
    spectrum, chart = _action_family()
    objective = PwOActionFamilies(spectrum, [chart], length=.4)
    with pytest.raises(ValueError):
        objective(dict(spectrum, **change))


def test_action_families_keep_distinct_means_and_fixed_normalization():
    spectrum, first = _action_family(.81)
    _, second = _action_family(.805)
    objective = PwOActionFamilies(spectrum, [first, second], length=.4)
    initial = objective.initial
    means = [f["mean_action_m"] for f in initial["families"]]
    assert abs(means[0]-means[1]) > .01
    weights = np.array([first["weight"], second["weight"]]); weights /= weights.sum()
    assert initial["value"] == pytest.approx(sum(w*f["variance_m2"] for w, f in zip(weights, initial["families"]))/ .4**2)
    after = objective(dict(spectrum, G_b=2.))
    assert [p['level_T'] for p in initial['per_pitch']] == [.805, .81]
    for pitch, family in zip(initial['per_pitch'], reversed(initial['families'])):
        assert pitch['value'] == pytest.approx(family['variance_m2']/.4**2)
    for before, changed in zip(initial['per_pitch'], after['per_pitch']):
        assert changed['value'] == pytest.approx(4*before['value'], rel=1e-10)
    for key in ("value", "raw_Jalpha2", "fixed_relative_variance"):
        assert after[key] == pytest.approx(4*initial[key], rel=1e-10)
    assert objective(spectrum)["value"] == initial["value"]
    first['weight'], second['weight'] = 1e300, 1e-300
    extreme = PwOActionFamilies(spectrum, [first, second], length=.4).initial
    np.testing.assert_allclose([p['value'] for p in extreme['per_pitch']],
                               [p['value'] for p in initial['per_pitch']], rtol=1e-10)


@pytest.mark.parametrize("forbidden", [False, True])
def test_action_family_rejects_intervals_that_are_not_one_allowed_well(forbidden):
    spectrum, chart = _action_family()
    for row in chart["rows"]:
        left, right = row["left"], row["right"]
        row["left"] = right if forbidden else left
        row["right"] = (left if forbidden else right)+2*np.pi
    with pytest.raises(ValueError, match="orientation|reflecting barrier"):
        PwOActionFamilies(spectrum, [chart], length=.4)


def test_action_family_owns_reference_arrays_and_rejects_weight_overflow():
    from copy import deepcopy
    spectrum, chart = _action_family()
    spectrum["bmnc_b"] = np.array(spectrum["bmnc_b"])
    saved = deepcopy(spectrum)
    objective = PwOActionFamilies(spectrum, [chart], length=.4)
    spectrum["bmnc_b"][:] = 0
    chart["rows"][0]["left"] = 0
    assert objective(saved)["value"] == objective.initial["value"]
    _, chart = _action_family()
    chart["weight"] = 1e308
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="total family weight"):
        PwOActionFamilies(saved, [chart, chart], length=.4)


def test_action_family_allows_fourier_refinement_and_reordering():
    spectrum, chart = _action_family()
    objective = PwOActionFamilies(spectrum, [chart], length=.4)
    refined = dict(spectrum, bmnc_b=[.02, 0., 1., .2], bmns_b=[0.]*4,
                   xm_b=[1, 2, 0, 0], xn_b=[0, 3, 0, 1])
    result = objective(refined)
    for key in ('value', 'raw_Jalpha2', 'fixed_relative_variance'):
        assert result[key] == pytest.approx(objective.initial[key], rel=1e-10)
