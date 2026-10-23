"""Analytic closed-transit oracles and differentiated moment envelopes."""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vmex.core.cyclometry import (
    moment_bound, moving_rational_surface, transit_action, transit_basis,
)


@pytest.fixture(autouse=True)
def float64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def mixed(nphase=32, ntransit=256):
    cosine, _ = transit_basis([0, 1, 1], [0, 0, 1], 1, 2,
                              nphase=nphase, ntransit=ntransit)
    return cosine @ jnp.array([.75, .07, .07])


def test_nonlinear_resonance():
    b = mixed()
    chi = 2 * np.pi * np.arange(len(b)) / len(b)
    exact = 1 / np.sqrt(.75**2 - 4 * .07**2 * np.cos(chi / 2)**2)
    np.testing.assert_allclose(transit_action(b, 0)[:, 0], exact, atol=2e-14)
    np.testing.assert_allclose(b.mean(axis=1), .75, atol=2e-14)
    assert np.ptp(exact) > .02


@pytest.mark.parametrize('q', [0., .5, .95, 1.])
@pytest.mark.parametrize('order', [1, 8, 32])
def test_pitch_envelope(q, order):
    b = mixed()
    actual = np.ptp(transit_action(b, np.linspace(0, q, 53)), axis=0).max()
    bound = moment_bound(b, order=order, lambda_max=q)
    assert actual <= bound + 2e-14
    smooth = moment_bound(b, order=order, lambda_max=q, softness=1e-4)
    assert bound <= smooth + 2e-14
    assert smooth - bound <= 4e-4 * np.log(len(b))
    # With a constant field every moment oscillation vanishes. The exact
    # binomial-series remainder at b=1 independently checks the tail.
    from scipy.special import binom
    truncated = sum(binom(.5, k) * (-q)**k for k in range(order + 1))
    tail = moment_bound(jnp.ones((2, 3)), order=order, lambda_max=q)
    assert 0 <= truncated - np.sqrt(1-q) <= tail + 2e-14


@pytest.mark.parametrize('N,M', [(1, 2), (2, 3), (-2, 3), (0, 1)])
def test_resonant_sine_and_physical_n(N, M):
    c, s = transit_basis([0, M], [0, N], N, M, nphase=24)
    b = c @ jnp.array([.8, .03]) + s @ jnp.array([0., .02])
    chi = 2 * np.pi * np.arange(24) / 24
    np.testing.assert_allclose(b.mean(axis=1), .8 + .03*np.cos(chi) + .02*np.sin(chi), atol=1e-14)


def test_equimeasurable_target_and_single_helicity():
    chi = 2 * np.pi * np.arange(32) / 32
    zeta = 4 * np.pi * np.arange(512) / 512
    for modulation in [0., .6]:
        b = .75 + .1*np.cos(zeta[None, :] + modulation*np.sin(chi[:, None]))
        assert np.max(np.ptp(transit_action(b, np.linspace(0, 1, 41)), axis=0)) < 2e-14


def test_continuum_error_and_gradients():
    c, s = transit_basis([0, 1, 1], [0, 0, 1], 1, 2)
    x = jnp.array([.75, .06, .08, .01, -.01])
    def loss(p):
        b = c @ p[:3] + s[..., 1:] @ p[3:]
        return moment_bound(b, order=16, softness=1e-4)
    value, grad = jax.jit(jax.value_and_grad(loss))(x)
    assert np.isfinite(value)
    for direction in np.eye(5):
        tangent = jax.jvp(loss, (x,), (jnp.asarray(direction),))[1]
        np.testing.assert_allclose(tangent, grad @ direction, atol=2e-12)
        for eps in [1e-5, 3e-6, 1e-6]:
            fd = (loss(x + eps*direction) - loss(x - eps*direction)) / (2*eps)
            np.testing.assert_allclose(tangent, fd, rtol=2e-4, atol=1e-8)
    b = mixed(16, 64)
    errors = np.r_[.61**-2, np.arange(1, 16)] * (.07*2*np.pi/16 + .07*4*np.pi/64)
    bound = moment_bound(b, order=16, oscillation_error=errors)
    dense = np.ptp(transit_action(mixed(128, 1024), np.linspace(0, .95, 31)), axis=0).max()
    assert dense <= bound


@pytest.mark.parametrize('bad', [0., -1., 1.1, np.nan, np.inf])
def test_invalid_field_eager_and_compiled(bad):
    b = jnp.full((2, 3), bad)
    for fn in [moment_bound, partial(transit_action, pitch_parameter=.5)]:
        with pytest.raises(ValueError, match='finite'):
            fn(b)
        assert np.all(np.isnan(jax.jit(fn)(b)))


@pytest.mark.parametrize('kwargs', [dict(order=0), dict(order=1.5), dict(lambda_max=-1),
                                    dict(lambda_max=np.nan), dict(softness=0), dict(softness=np.inf),
                                    dict(oscillation_error=[0]), dict(oscillation_error=np.full(32, -1)),
                                    dict(oscillation_error=np.full(32, np.nan))])
def test_invalid_bound_parameters(kwargs):
    with pytest.raises(ValueError):
        moment_bound(mixed(), **kwargs)


def test_shapes_and_invalid_pitch():
    for b in [jnp.ones(3), jnp.ones((0, 3))]:
        with pytest.raises(ValueError, match='shape'):
            transit_action(b, .5)
    for pitch in [-.1, 1.1, np.nan, [], [[.5]]]:
        with pytest.raises(ValueError):
            transit_action(mixed(), pitch)
    assert np.isnan(jax.jit(transit_action)(mixed(), -1.)).all()
    assert np.isnan(jax.jit(lambda error: moment_bound(mixed(), oscillation_error=error))(
        jnp.full(32, -1.)))


@pytest.mark.parametrize('args,kwargs', [
    (([0], [0], 2, 4), {}), (([0], [0], 1.5, 2), {}), (([0], [0], 1, 0), {}),
    (([], [], 1, 2), {}), (([0], [0, 1], 1, 2), {}), (([.5], [0], 1, 2), {}),
    (([0], [np.nan], 1, 2), {}), (([0], [0], 1, 2), dict(nphase=1)),
    (([0], [0], 1, 2), dict(ntransit=3.5)),
])
def test_invalid_basis(args, kwargs):
    with pytest.raises(ValueError):
        transit_basis(*args, **kwargs)


RADII = np.array([.1, .3, .5, .9])


def radial_table(p):
    s = jnp.asarray(RADII)
    return dict(s_b=s, iota_b=p[0] + p[1]*s, G_b=2 + p[2] + 3*s,
                I_b=-.2 + p[3]*s, bmnc_b=jnp.stack((.75 + p[2]*s, p[3] + .02*s), axis=1),
                bmns_b=jnp.stack((jnp.zeros_like(s), .01 + p[4] + .03*s), axis=1),
                psi_b=-2*s, psi_edge=-2., xm_b=np.array([0, 2]), xn_b=np.array([0, 1]), nfp=1)


def moving(table, **kwargs):
    return moving_rational_surface(table, 1, 2, radii=RADII, bracket=(.2, .8), **kwargs)


@pytest.mark.parametrize('sign', [1., -1.])
def test_moving_surface_implicit_derivatives(sign):
    p = jnp.array([.5 - sign*.25, sign*.6, .03, .06, .02])
    cosine, sine = transit_basis([0, 2], [0, 1], 1, 2, nphase=12, ntransit=16)

    def objective(table):
        b = cosine @ table['bmnc_b'][0] + sine @ table['bmns_b'][0]
        return (moment_bound(b, order=8, softness=1e-3)
                + table['G_b'][0] * table['I_b'][0] + table['psi_b'][0]**2
                + table['root_s']**2 + table['iota_shear']**2)

    def loss(x):
        return objective(moving(radial_table(x)))

    def exact(x):
        # Independent affine-profile oracle: no interpolation or root solver.
        s = (.5 - x[0]) / x[1]
        return objective(dict(bmnc_b=jnp.array([[.75 + x[2]*s, x[3] + .02*s]]),
                              bmns_b=jnp.array([[0., .01 + x[4] + .03*s]]),
                              G_b=jnp.array([2 + x[2] + 3*s]), I_b=jnp.array([-.2 + x[3]*s]),
                              psi_b=jnp.array([-2*s]), root_s=s, iota_shear=x[1]))

    surface = jax.jit(lambda x: moving(radial_table(x)))(p)
    assert surface['bmnc_b'].shape == (1, 2)
    assert surface['G_b'].shape == surface['s_b'].shape == (1,)
    np.testing.assert_allclose(surface['root_s'], 5/12, atol=1e-15)
    np.testing.assert_allclose(surface['iota_b'], [.5], atol=1e-15)
    np.testing.assert_allclose(surface['iota_shear'], sign*.6, atol=1e-15)
    assert abs(surface['root_residual']) < 1e-15
    expected_grad = jax.grad(exact)(p)
    value, pullback = jax.vjp(jax.jit(loss), p)
    np.testing.assert_allclose(value, exact(p), atol=2e-14)
    np.testing.assert_allclose(pullback(jnp.array(1.))[0], expected_grad, atol=2e-12)
    for direction in jnp.eye(5):
        tangent = jax.jvp(loss, (p,), (direction,))[1]
        np.testing.assert_allclose(tangent, expected_grad @ direction, atol=2e-12)
        for eps in [1e-4, 3e-5, 1e-5]:
            finite_difference = (loss(p + eps*direction) - loss(p - eps*direction)) / (2*eps)
            np.testing.assert_allclose(tangent, finite_difference, rtol=2e-6, atol=2e-9)


@pytest.mark.parametrize('sign', [1., -1.])
def test_moving_surface_uses_root_segment_and_ignores_outside_turns(sign):
    # The bracket spans three unequal slopes; the first table interval lies
    # outside it and turns in the opposite direction. Differentiate the
    # interior affine root independently of the interpolation implementation.
    radii = np.array([.1, .2, .4, .6, .9])
    iota = jnp.array([.7, .2, .35, .65, .8])
    iota = .5 + sign * (iota - .5)

    def root(values):
        table = dict(s_b=radii, iota_b=values, G_b=jnp.ones(5),
                     I_b=jnp.zeros(5), bmnc_b=jnp.ones((5, 1)))
        return moving_rational_surface(table, 1, 2, radii=radii,
                                       bracket=(.25, .85))['root_s']

    def exact(values):
        return .4 + .2 * (.5 - values[2]) / (values[3] - values[2])

    value, gradient = jax.jit(jax.value_and_grad(root))(iota)
    np.testing.assert_allclose(value, exact(iota), atol=1e-15)
    np.testing.assert_allclose(gradient, jax.grad(exact)(iota), atol=1e-14)


@pytest.mark.parametrize('case', ['missing', 'zero_shear', 'small_shear', 'multiple_roots',
                                 'endpoint', 'knot', 'radii_mismatch', 'iota_b', 'bmnc_b',
                                 'bmns_b', 'G_b', 'I_b', 'psi_b', 'psi_edge'])
def test_invalid_moving_branch_eager_and_compiled(case):
    table = radial_table(jnp.array([.25, .6, .03, .06, .02]))
    if case == 'missing':
        table['iota_b'] = table['iota_b'] + 2
    elif case in ('zero_shear', 'small_shear'):
        table['iota_b'] = .5 + (0. if case == 'zero_shear' else 1e-8)*(jnp.asarray(RADII) - .4)
    elif case == 'multiple_roots':
        table['iota_b'] = jnp.array([.1, .6, .4, .9])
    elif case in ('endpoint', 'knot'):
        table['iota_b'] = .5 + .6*(jnp.asarray(RADII) - (.2 if case == 'endpoint' else .3))
    elif case == 'radii_mismatch':
        table['s_b'] = table['s_b'].at[1].add(.01)
    else:
        table[case] = jnp.full_like(jnp.asarray(table[case]), jnp.nan)
    with pytest.raises(ValueError, match='invalid rational branch'):
        moving(table)
    result = jax.jit(moving)(table)
    for key in ('root_s', 's_b', 'iota_shear', 'root_residual', 'iota_b', 'bmnc_b', 'G_b', 'I_b'):
        assert np.all(np.isnan(result[key]))


@pytest.mark.parametrize('kwargs', [dict(radii=[.1, .1]), dict(radii=[.3, .1]),
                                    dict(radii=[0., .9]), dict(radii=[.1, np.nan]),
                                    dict(bracket=(.0, .8)), dict(bracket=(.8, .2)),
                                    dict(minimum_shear=0), dict(root_tolerance=np.inf),
                                    dict(numerator=2, denominator=4)])
def test_invalid_moving_static_contract(kwargs):
    arguments = dict(numerator=1, denominator=2, radii=RADII, bracket=(.2, .8))
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        moving_rational_surface(radial_table(jnp.array([.25, .6, .03, .06, .02])), **arguments)


def test_moving_surface_two_nodes_optional_parities_and_shape_guards():
    table = radial_table(jnp.array([.25, .6, .03, .06, .02]))
    for name in ('bmns_b', 'psi_b', 'psi_edge'):
        del table[name]
    two = {name: values[np.array([0, 3])] if name in ('s_b', 'iota_b', 'G_b', 'I_b', 'bmnc_b')
           else values for name, values in table.items()}
    root = moving_rational_surface(two, 1, 2, radii=RADII[[0, 3]], bracket=(.2, .8))
    np.testing.assert_allclose(root['root_s'], 5/12)
    for name in ('s_b', 'iota_b', 'G_b', 'I_b', 'bmnc_b'):
        bad = dict(table, **{name: table[name][:-1]})
        with pytest.raises(ValueError, match='shape|match'):
            moving(bad)
