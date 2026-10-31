"""Independent mapping, Fourier variance and derivative checks for the pwO proxy."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vmex.core.pwo import (
    MappedPwOResidual,
    mapped_pwo_residual_from_boozer,
    squeezed_pwo_coordinates,
)

jax.config.update("jax_enable_x64", True)


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


def test_default_live_mean_normalization_is_invariant_to_surface_field_scale():
    spectrum = _spectrum(2)
    # Put the constant mode last: its location is not part of the contract.
    permutation = [1, 2, 0]
    for key in ("bmnc_b", "bmns_b"):
        spectrum[key] = spectrum[key][:, permutation] * np.array([[1.], [1.3]])
    for key in ("xm_b", "xn_b"):
        spectrum[key] = spectrum[key][permutation]
    options = dict(ngamma=8, neta=8, weights=[1., 3.])
    baseline = mapped_pwo_residual_from_boozer(**spectrum, **options)
    explicit = mapped_pwo_residual_from_boozer(
        **spectrum, b_ref=[1.5, 1.95], **options)
    np.testing.assert_allclose(baseline["residuals1d"], explicit["residuals1d"], atol=1e-15)
    scale = np.array([[.2], [4.5]])
    scaled = mapped_pwo_residual_from_boozer(
        **(spectrum | {key: spectrum[key]*scale for key in ("bmnc_b", "bmns_b")}),
        **options)
    np.testing.assert_allclose(scaled["residuals1d"], baseline["residuals1d"], atol=1e-15)
    assert float(scaled["total"]) == pytest.approx(float(baseline["total"]), rel=1e-13)


def test_live_mean_jvp_and_vjp_include_the_normalizing_constant_mode():
    spectrum = _spectrum(2)
    cosine = jnp.array([[1.5, .08, -.06], [1.7, .04, -.09]])
    sine = jnp.array([[0., -.04, .025], [0., .03, -.02]])
    direction = jnp.array([[.13, 0., 0.], [-.07, 0., 0.]])

    def residual(cosine, sine):
        return mapped_pwo_residual_from_boozer(
            **(spectrum | dict(bmnc_b=cosine, bmns_b=sine)),
            weights=[1., 3.], ngamma=8, neta=8)["residuals1d"]

    value, tangent = jax.jit(lambda x: jax.jvp(
        lambda c: residual(c, sine), (x,), (direction,)))(cosine)
    # A constant-field perturbation cancels from B-mean_gamma(B). Only the
    # quotient's denominator changes: dr/dB00 = -r/B00 on each surface.
    rows = np.asarray(value).reshape(2, -1)
    expected = -rows*np.asarray(direction[:, :1]/cosine[:, :1])
    np.testing.assert_allclose(tangent, expected.ravel(), rtol=2e-12, atol=1e-15)
    cotangent = jnp.linspace(-.3, .5, value.size)
    _, pullback = jax.vjp(lambda c: residual(c, sine), cosine)
    expected_constant_gradient = -np.sum(
        np.asarray(cotangent).reshape(2, -1)*rows, axis=1)/np.asarray(cosine[:, 0])
    np.testing.assert_allclose(pullback(cotangent)[0][:, 0], expected_constant_gradient,
                               rtol=2e-12, atol=1e-15)
    # Scaling both Fourier parities has zero tangent; a detached denominator
    # would incorrectly leave a tangent equal to the residual itself.
    _, scale_tangent = jax.jit(lambda c, s: jax.jvp(
        residual, (c, s), (c, s)))(cosine, sine)
    np.testing.assert_allclose(scale_tangent, 0., atol=2e-15)


@pytest.mark.parametrize("constant", [0., -.1, np.nan, np.inf])
def test_default_live_mean_rejects_nonpositive_or_nonfinite_constant(constant):
    # This toroidal harmonic is positive on the mapped cut even when B00<=0.
    # Hence the default must reject its normalization, not merely negative B.
    spectrum = dict(bmnc_b=np.array([[constant, -1.]]),
        xm_b=np.array([0, 0]), xn_b=np.array([0, 3]), iota_b=np.array([.7]), nfp=3)
    options = dict(a=0., eta_min=3.05, ngamma=8, neta=8)
    if np.isfinite(constant):
        control = mapped_pwo_residual_from_boozer(**spectrum, b_ref=1., **options)
        assert np.all(control["valid"])
    with pytest.raises(ValueError):
        mapped_pwo_residual_from_boozer(**spectrum, **options)

    def traced(cosine):
        result = mapped_pwo_residual_from_boozer(
            **(spectrum | dict(bmnc_b=cosine)), **options)
        return result["valid"], result["residuals1d"]
    valid, residual = jax.jit(traced)(jnp.asarray(spectrum["bmnc_b"]))
    assert not np.any(valid)
    assert not np.isfinite(residual).all()


@pytest.mark.parametrize("constant_modes", [0, 2])
def test_default_live_mean_requires_exactly_one_constant_fourier_mode(constant_modes):
    spectrum = _spectrum()
    if constant_modes == 0:
        spectrum.update(bmnc_b=np.array([[-1.]]), bmns_b=np.array([[0.]]),
                        xm_b=np.array([0]), xn_b=np.array([3]))
    else:
        spectrum["xm_b"][1] = spectrum["xn_b"][1] = 0
    options = dict(a=0., eta_min=3.05, ngamma=8, neta=8)
    control = mapped_pwo_residual_from_boozer(**spectrum, b_ref=1., **options)
    assert np.all(control["valid"])
    with pytest.raises(ValueError):
        mapped_pwo_residual_from_boozer(**spectrum, **options)

    def traced(cosine):
        result = mapped_pwo_residual_from_boozer(
            **(spectrum | dict(bmnc_b=cosine)), **options)
        return result["valid"], result["residuals1d"]
    valid, residual = jax.jit(traced)(jnp.asarray(spectrum["bmnc_b"]))
    assert not np.any(valid)
    assert not np.isfinite(residual).all()


def test_mapped_qi_limit_has_zero_residual_but_generic_field_does_not():
    spectra = dict(bmnc_b=np.array([[1.5, .2]]), bmns_b=np.array([[0., .1]]),
                   xm_b=np.array([0, 0]), xn_b=np.array([0, 3]), iota_b=np.array([.7]), nfp=3)
    control = mapped_pwo_residual_from_boozer(**spectra, b_ref=1.5, a=0.)
    assert float(control["total"]) < 1e-28
    broken = mapped_pwo_residual_from_boozer(**_spectrum(), b_ref=1.5, a=0.)
    assert float(broken["total"]) > 1e-4


def test_overflowing_surface_weights_cannot_zero_the_objective():
    kwargs = dict(**_spectrum(2), b_ref=1.5)
    regular = mapped_pwo_residual_from_boozer(**kwargs)
    try:
        huge = mapped_pwo_residual_from_boozer(**kwargs, weights=np.array([1e308, 1e308]))
    except ValueError:
        return  # Explicit rejection is also valid for this unsupported normalization.
    assert float(huge["total"]) == pytest.approx(float(regular["total"]), rel=1e-12)


def test_adapters_forward_one_live_transform_and_preserve_diagnostics(monkeypatch):
    from types import SimpleNamespace
    from vmex.core import pwo
    calls = []
    def transform(state, runtime, **kwargs):
        calls.append((state, runtime, kwargs))
        return dict(_spectrum(), s_b=np.array([.5]))
    monkeypatch.setattr(pwo, "boozer_spectrum_state", transform)
    eq = SimpleNamespace(state=object(), runtime=object())
    target = pwo.MappedPwOResidual([.5], b_ref=1.5, ngamma=8, neta=8)
    report = target.compute_state(eq.state, eq.runtime)
    assert len(calls) == 1
    assert calls[-1][0] is eq.state and calls[-1][1] is eq.runtime
    np.testing.assert_array_equal(calls[-1][2]["surfaces"], [.5])
    assert report["s_b"][0] == .5
    np.testing.assert_allclose(target(eq), report["residuals1d"])
    assert target.total(eq) == pytest.approx(float(report["total"]))
    np.testing.assert_allclose(target.J(eq), report["residuals1d"])


@pytest.mark.parametrize("kwargs", [dict(surfaces=[]), dict(surfaces=[[.5]]), dict(surfaces=[np.nan]),
    dict(surfaces=[0.]), dict(surfaces=[1.01]), dict(mboz=0), dict(nboz=-1), dict(oversample=0)])
def test_invalid_adapter_configuration(kwargs):
    with pytest.raises(ValueError):
        MappedPwOResidual(**(dict(surfaces=[.5], b_ref=1.5) | kwargs))


@pytest.mark.parametrize("kwargs", [dict(weights=[0.]), dict(weights=[-1.]), dict(weights=[np.nan]),
    dict(weights=[1., 1.]), dict(bmnc_b=np.ones(3)), dict(bmns_b=np.ones((2, 3))),
    dict(xn_b=np.array([0, 1, 2])), dict(xm_b=np.array([0., 1.1, 2.])), dict(iota_b=np.array([.7,.8]))])
def test_invalid_spectrum_contract(kwargs):
    with pytest.raises(ValueError):
        mapped_pwo_residual_from_boozer(**dict(_spectrum(), **kwargs), b_ref=1.5)


@pytest.mark.parametrize("kwargs", [dict(eta_min=0.), dict(eta_min=np.pi), dict(eta_min=np.nan),
    dict(c=0.), dict(a=10.), dict(b_ref=0.), dict(b_ref=[1., 2.]),
    dict(ngamma=3), dict(neta=3)])
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


def test_mapped_residual_jvp_vjp_and_independent_finite_difference():
    # Different surface weights, both Fourier parities and signed live iota.
    x = jnp.array([1.5, .08, -.06, 1.7, .04, -.09,
                   0., -.04, .025, 0., .03, -.02, .7, -.6, 1.5, 1.7])
    direction = jnp.linspace(-.1, .1, x.size)
    modes = dict(xm_b=np.array([0, 1, 2]), xn_b=np.array([0, 3, -6]), nfp=3)
    def residual(values):
        return mapped_pwo_residual_from_boozer(
            bmnc_b=values[:6].reshape(2, 3), bmns_b=values[6:12].reshape(2, 3),
            iota_b=values[12:14], b_ref=values[14:], weights=[1., 3.],
            ngamma=12, neta=10, a=.3, b=.2, **modes)["residuals1d"]
    def oracle(values):
        values = np.asarray(values)
        gamma = 2*np.pi*np.arange(12)[None, :, None]/12
        eta = (2.8+(2*np.pi-5.6)*(np.arange(10)+.5)/10)[None, None, :]
        theta, zeta = _map(gamma, eta, iota=values[12:14, None, None])
        phase = theta[..., None]*modes["xm_b"]-zeta[..., None]*modes["xn_b"]
        field = np.sum(values[:6].reshape(2, 1, 1, 3)*np.cos(phase)
                       + values[6:12].reshape(2, 1, 1, 3)*np.sin(phase), axis=-1)
        centered = (field-field.mean(axis=1, keepdims=True))/values[14:, None, None]
        return (centered*np.sqrt(np.array([.25, .75])[:, None, None]/120)).ravel()
    value, tangent = jax.jit(lambda q: jax.jvp(residual, (q,), (direction,)))(x)
    np.testing.assert_allclose(value, oracle(x), atol=1e-15)
    cotangent = jnp.linspace(-.3, .5, value.size)
    _, pullback = jax.vjp(residual, x)
    np.testing.assert_allclose(jnp.vdot(cotangent, tangent),
                               jnp.vdot(pullback(cotangent)[0], direction), atol=1e-13)
    for step in (1e-4, 1e-5):
        finite = (oracle(x+step*direction)-oracle(x-step*direction))/(2*step)
        np.testing.assert_allclose(tangent, finite, rtol=2e-7, atol=1e-11)


@pytest.mark.parametrize("reference", [0., np.nan])
def test_traced_invalid_map_reports_false_and_nonfinite_residual(reference):
    def evaluate(b_ref):
        result = mapped_pwo_residual_from_boozer(**_spectrum(), b_ref=b_ref, ngamma=8, neta=8)
        return result["valid"], result["residuals1d"]
    valid, residual = jax.jit(evaluate)(reference)
    assert not np.any(valid)
    assert not np.isfinite(residual).all()


def test_full_surface_diagnostics_reuse_transform_and_include_sine_modes(monkeypatch):
    spectrum = _spectrum(ns=2)
    mapped = MappedPwOResidual([.25, .75], b_ref=[1.5, 1.5])
    calls = []

    def compute(state, runtime):
        calls.append((state, runtime))
        return {**spectrum, "total": 0.}

    monkeypatch.setattr(mapped, "compute_state", compute)
    _, low, high, mirror = mapped.diagnostics_state(None, None, ntheta=9, nzeta=11)
    theta, zeta = np.meshgrid(2*np.pi*np.arange(9)/9, 2*np.pi*np.arange(11)/33, indexing="ij")
    field = np.zeros((2, 9, 11))
    for j, (m, n) in enumerate(zip(spectrum["xm_b"], spectrum["xn_b"])):
        field += (spectrum["bmnc_b"][:, j, None, None]*np.cos(m*theta-n*zeta)
                  + spectrum["bmns_b"][:, j, None, None]*np.sin(m*theta-n*zeta))
    np.testing.assert_allclose(low, field.min(axis=(1, 2)), rtol=1e-14)
    np.testing.assert_allclose(high, field.max(axis=(1, 2)), rtol=1e-14)
    np.testing.assert_allclose(mirror, (high-low)/(high+low), rtol=1e-14)
    assert len(calls) == 1


@pytest.mark.parametrize("low, high, mirror, feasible", [
    (1., 2., 1/3, True), (-.01, 2., 1.01, False), (float("nan"), 2., .1, False),
    (1., float("inf"), .1, False), (1., 2., float("nan"), False),
])
def test_physical_metrics_reject_invalid_full_surface_field(monkeypatch, low, high, mirror, feasible):
    from types import SimpleNamespace as NS
    from vmex.core import statephysics

    eq = NS(state=None, runtime=NS(setup=NS(signgs=1)),
            wout=NS(aspect=10., iotaf=np.array([.6, .7]), betatotal=0.),
            result=NS(fsqr=1e-14, fsqz=1e-14, fsql=1e-14))
    jacobian = NS(sqrt_g=np.ones((3, 2, 2)))
    field = NS(pressure=np.zeros(3), total_pressure=np.ones((3, 2, 2)))
    monkeypatch.setattr(statephysics, "_field_chain", lambda state, rt: (None, jacobian, None, field, None))
    mapped = MappedPwOResidual([.5], b_ref=1.)
    result = mapped.physical_metrics(eq, bmax_ref=[3.], aspect_bounds=(9.9, 10.1), iota_floor=.41,
        fields=({"total": .01, "valid": np.array([True])}, [low], [high], [mirror]))
    assert result["feasible"] is feasible
