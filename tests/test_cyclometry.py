"""Analytic closed-transit oracles and differentiated moment envelopes."""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vmex.core.cyclometry import moment_bound, transit_action, transit_basis


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
