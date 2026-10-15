"""Algebra controls only; these do not certify a stellarator proxy."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad
from deltares_reference import (helical_angles, normalized_bump, paper_average,
                               resonance_pairs, stabilized_power)


def test_fourier_normalization():
    eta = 2 * np.pi * np.arange(64) / 64
    score = stabilized_power(.3 + .07 * np.cos(3 * eta) + .02 * np.sin(3 * eta), 8)
    np.testing.assert_allclose(score[2], 64 * (.07**2 + .02**2) / (np.pi**2 * 9), rtol=1e-13)
    assert np.max(np.delete(np.asarray(score), 2)) < 1e-29


@pytest.mark.parametrize("n,q", [(16, 8), (15, 8), (8, 0)])
def test_alias_guard(n, q):
    with pytest.raises(ValueError):
        stabilized_power(jnp.ones(n), q)


def test_zero_signal_gradient():
    f = lambda x: jnp.sum(stabilized_power(x, 4))
    x = jnp.zeros(32)
    assert float(f(x)) == 0
    np.testing.assert_array_equal(jax.grad(f)(x), 0)


def test_signed_unreduced_harmonics():
    pairs = resonance_pairs(2, 4, -.75, .1)
    assert all(-.75 <= p / q <= .1 for p, q in pairs)
    assert (pairs[:, 0] == 0).sum() == 4
    assert (-1, 2) in map(tuple, pairs) and (-2, 4) in map(tuple, pairs)
    assert resonance_pairs(0, 1, 1., 2.).shape == (0, 2)


def test_bump_normalization():
    value, _ = quad(lambda x: float(normalized_bump(x, .07)), -.07, .07,
                    epsabs=1e-12, epsrel=1e-12)
    np.testing.assert_allclose(value, 1., rtol=5e-11)


def test_bump_support_gradient():
    x = jnp.array([-1e100, -.1, -.099999999, 0., .099999999, .1, 1e100])
    values = jax.jit(jax.vmap(lambda z: normalized_bump(z, .1)))(x)
    gradient = jax.jit(jax.vmap(jax.grad(lambda z: normalized_bump(z, .1))))(x)
    assert np.isfinite(values).all() and np.isfinite(gradient).all()
    np.testing.assert_array_equal(np.asarray(values)[[0, 1, 5, 6]], 0)
    assert np.isnan(float(normalized_bump(0., 0.)))


def test_pitch_jacobian_domain_replication():
    score, tau = jnp.ones((2, 3, 1)), jnp.full((2, 4, 3, 1), 2.)
    b, wb, L = jnp.array([1., 2., 4.]), jnp.array([.1, .2, .4]), jnp.array([2., 4.])
    got = paper_average(score, tau, b, wb, L)
    np.testing.assert_allclose(got, np.sum(np.asarray(wb / b**2)) / np.asarray(L))
    repeated = paper_average(jnp.tile(score, (1, 1, 2)), jnp.tile(tau, (1, 1, 1, 2)), b, wb, 2 * L)
    np.testing.assert_allclose(repeated, got)


def test_average_invalid():
    score, tau = jnp.ones((1, 2, 1)), jnp.ones((1, 2, 2, 1))
    assert np.isnan(paper_average(score, tau, jnp.array([0., 1.]), jnp.ones(2), jnp.ones(1))).all()
    assert np.isnan(paper_average(score, tau, jnp.ones(2), jnp.zeros(2), jnp.ones(1))).all()
    with pytest.raises(ValueError):
        paper_average(score, tau, jnp.ones(3), jnp.ones(3), jnp.ones(1))


@pytest.mark.parametrize("M,N,nfp,iota", [(1, 0, 2, .58), (1, 4, 4, 1.1), (0, 4, 4, .7)])
def test_helical_chart(M, N, nfp, iota):
    chi, eta = .73, .61
    theta, zeta = helical_angles(chi, eta, iota, nfp, M, N)
    np.testing.assert_allclose(M * theta - N * zeta, chi, rtol=1e-13)
    np.testing.assert_allclose(nfp * (theta - iota * zeta) / (N - M * iota), eta, rtol=1e-13)
    th2, ze2 = helical_angles(chi, eta + 2 * np.pi, iota, nfp, M, N)
    np.testing.assert_allclose(jnp.cos(3 * th2 + 2 * nfp * ze2),
                               jnp.cos(3 * theta + 2 * nfp * zeta), atol=2e-14)


def test_iota_chain_rule():
    f = lambda i: helical_angles(.7, .9, i, 2, 1, 0)[1]
    np.testing.assert_allclose(jax.grad(f)(.6), -.7 / .6**2, rtol=1e-13)
    assert np.isnan(helical_angles(.7, .9, 0., 2, 1, 0)[0])


def test_conditional_action_spectrum_identity():
    eta = 2 * jnp.pi * jnp.arange(64) / 64
    action = 3. + .003 * jnp.cos(2 * eta) + .001 * jnp.sin(3 * eta)
    derivative = -.006 * jnp.sin(2 * eta) + .003 * jnp.cos(3 * eta)
    coefficient = .027
    aq = jnp.fft.rfft(action)[1:9] / eta.size
    expected = 256 / jnp.pi**2 * coefficient**2 * (aq.real**2 + aq.imag**2)
    np.testing.assert_allclose(stabilized_power(coefficient * derivative, 8), expected, rtol=1e-10, atol=1e-30)


def test_gradient_and_transpose():
    eta = jnp.arange(40) * (2 * jnp.pi / 40)
    def rows(x):
        return stabilized_power(x[0] * jnp.sin(eta) + x[1] * jnp.cos(2 * eta), 4) * normalized_bump(.01 + .03 * x[0], .09)
    x, v, cotangent = jnp.array([.12, -.2]), jnp.array([.4, .6]), jnp.arange(1., 5.)
    scalar = lambda z: jnp.sum(rows(z))
    h = 1e-5
    fd = (scalar(x + h * v) - scalar(x - h * v)) / (2 * h)
    np.testing.assert_allclose(jax.grad(scalar)(x) @ v, fd, rtol=2e-8)
    jvp = jax.jvp(rows, (x,), (v,))[1]
    _, pullback = jax.vjp(rows, x)
    np.testing.assert_allclose(jvp @ cotangent, v @ pullback(cotangent)[0], rtol=1e-12)


def test_energy_strength_scaling():
    eta = jnp.arange(40) * (2 * jnp.pi / 40)
    for ratio in (.01, .5, 2.):
        np.testing.assert_allclose(stabilized_power(jnp.sqrt(ratio) * jnp.sin(eta), 4),
                                   ratio * stabilized_power(jnp.sin(eta), 4), atol=1e-29)


def test_ratio_of_means_not_mean_rotation():
    tau, precession = np.array([1., 2.]), np.array([.2, .6])
    assert abs(np.mean(precession * tau / (2 * np.pi)) - np.mean(precession) / np.mean(2 * np.pi / tau)) > .02


def test_deposited_square_grid_phase():
    assert 1.0**2 + 0.0**2 == 2 * (.5**2 + .5**2)


def test_published_normalization_complete_single_well():
    eps, bmax = .2, 1.2
    nodes, weights = np.polynomial.legendre.leggauss(160)
    bcrit, wb = 1 - eps + eps * (nodes + 1), eps * weights
    z, wz = np.polynomial.legendre.leggauss(96)
    angle, wa = np.pi / 2 * z, np.pi / 2 * wz
    roots = np.arccos((1 - bcrit) / eps)
    x = roots[:, None] * np.sin(angle)
    parallel = 1 - (1 - eps * np.cos(x)) / bcrit[:, None]
    vtau = 2 * np.sum(wa * roots[:, None] * np.cos(angle) / np.sqrt(parallel), axis=1)
    L = quad(lambda z: 1 / (1 - eps * np.cos(z)), -np.pi, np.pi, epsabs=1e-12)[0]
    ft = quad(lambda z: np.sqrt(max(0., 1 - (1 - eps * np.cos(z)) / bmax))
              / (1 - eps * np.cos(z)), -np.pi, np.pi, epsabs=1e-12)[0] / L
    published = float(paper_average(np.ones((1, 160, 1)), vtau[None, None, :, None],
                                    bcrit, wb, np.array([L]))[0])
    np.testing.assert_allclose(published, 2 * ft, rtol=3e-5)


def test_bounce_cancellation_before_squaring():
    derivative = np.sin(2 * np.pi * np.arange(128) / 128 + .31)
    assert np.mean(derivative)**2 < 1e-28
    np.testing.assert_allclose(np.mean(derivative**2), .5, rtol=1e-13)
