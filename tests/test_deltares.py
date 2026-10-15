"""Reduction algebra and restricted action controls, not a confinement certificate."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad
from vmex.core.deltares import (helical_angles, normalized_bump, paper_average,
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
    def f(x):
        return jnp.sum(stabilized_power(x, 4))
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
    def f(i):
        return helical_angles(.7, .9, i, 2, 1, 0)[1]
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
    def scalar(z):
        return jnp.sum(rows(z))
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


@pytest.mark.parametrize('omega,expected', [
    ([-1., 1.], [.5, .5]), ([1., -1.], [.5, .5]),
    ([-1., 0., 1.], [0., 1., 0.]), ([0., .2, 1.], [1., 0., 0.]),
    ([-1., -.2, 0.], [0., 0., 1.]), ([-1., 1., -1.], [.5, 1., .5]),
    ([-2., -1.], [0., 0.]), ([.2, .2], [0., 0.]),
])
def test_crossing_deposition(omega, expected):
    from vmex.core.deltares import linear_crossing_weights
    got = jax.jit(linear_crossing_weights)(jnp.array(omega), jnp.array([0.]))
    np.testing.assert_allclose(got[:, 0], expected)


@pytest.mark.parametrize('omega', [[0., 0.], [-1., 0., -1.], [1., 0., 1.], [0., np.nan]])
def test_degenerate_crossing_is_not_zero(omega):
    from vmex.core.deltares import linear_crossing_weights
    assert np.isnan(linear_crossing_weights(jnp.array(omega), jnp.array([0.]))).all()


def test_crossing_gaps_shapes_and_multiple_families():
    from vmex.core.deltares import linear_crossing_weights
    omega = jnp.array([[-1., -.2], [np.nan, .6], [1., 1.]])
    valid = jnp.isfinite(omega)
    got = linear_crossing_weights(omega, jnp.array([0.]), valid)
    np.testing.assert_allclose(got[..., 0], [[0., .75], [0., .25], [0., 0.]])
    assert np.isnan(linear_crossing_weights(jnp.array([0., 1.]), jnp.array([0.]),
                                          jnp.array([True, False]))).all()
    assert np.isnan(linear_crossing_weights(jnp.array([0., 1.]), jnp.array([np.nan]))).all()
    for om, res, mask in [(jnp.ones(1), jnp.ones(1), None),
                          (jnp.ones(2), jnp.ones((1, 1)), None),
                          (jnp.ones(2), jnp.ones(1), jnp.ones(3))]:
        with pytest.raises(ValueError):
            linear_crossing_weights(om, res, mask)


def _maps(amplitude=.02):
    eta = jnp.arange(24) * (2 * jnp.pi / 24)
    ds = jnp.broadcast_to(amplitude * jnp.sin(eta)[None, :, None, None], (3, 24, 1, 1))
    de = jnp.broadcast_to(2 * jnp.pi * jnp.array([-.11, .14, .31])[:, None, None, None], ds.shape)
    return ds, de, jnp.ones_like(ds) * 2, jnp.array([1.]), jnp.array([.2]), jnp.ones(3)


def test_maps_reduction_and_gradient():
    from vmex.core.deltares import deltares_from_maps
    def loss(a):
        return deltares_from_maps(*_maps(a), p_max=0, q_max=2)['loss']
    out = jax.jit(lambda a: deltares_from_maps(*_maps(a), p_max=0, q_max=2))(.02)
    strength = 64 * .02**2 / np.pi**2
    expected = strength * .2 * np.array([.56, .44, 0.])
    assert out['valid']
    np.testing.assert_allclose(out['delta_res'], expected, rtol=1e-13)
    np.testing.assert_allclose(out['loss'], .5 * np.sum(expected**2), rtol=1e-13)
    np.testing.assert_allclose(jax.grad(loss)(.02), 4 * loss(.02) / .02, rtol=1e-13)
    np.testing.assert_allclose(jax.grad(loss)(.02), (loss(.020001) - loss(.019999)) / .000002, rtol=1e-8)
    assert loss(0.) == 0 and jax.grad(loss)(0.) == 0


def test_maps_rotation_zero_precession_and_rejection():
    from vmex.core.deltares import deltares_from_maps
    ds, de, tau, *rest = _maps()
    tau = tau.at[:, ::2].set(4.)
    de = de.at[:, ::2].set(0.)
    ratio = deltares_from_maps(ds, de, tau, *rest, p_max=0, q_max=2)
    mean = deltares_from_maps(ds, de, tau, *rest, p_max=0, q_max=2, rotation='mean_map')
    np.testing.assert_allclose(ratio['omega'], mean['omega'] * 4 / 3)
    for bad_tau in [tau.at[0, 0, 0, 0].set(0.), tau * np.nan]:
        out = deltares_from_maps(ds, de, bad_tau, *rest, p_max=0, q_max=2)
        assert not out['valid'] and np.isnan(out['loss'])
    with pytest.raises(ValueError):
        deltares_from_maps(ds[0], de, tau, *rest)
    with pytest.raises(ValueError):
        deltares_from_maps(ds, de, tau, *rest, rotation='unknown')
    with pytest.raises(ValueError):
        resonance_pairs(-1, 2)


def _restricted_input(x=None):
    s = jnp.array([.2, .45, .7])
    x = jnp.array([0., .002, .001]) if x is None else x
    return dict(bmnc_b=jnp.stack((1 + .08 * (s - .4)**2 + x[0], -.2 - .01 * s,
                                 jnp.full(3, .01)), axis=1),
                bmns_b=jnp.zeros((3, 3)).at[:, 2].set(x[1]),
                xm_b=jnp.array([0, 1, 0]), xn_b=jnp.array([0, 0, 2]),
                iota_b=.5 + .01 * s + x[2], G_b=2 + .05 * s, I_b=jnp.full(3, .05),
                nfp=2, psi_b=2 * s, psi_edge=2., bcrit=jnp.array([1.02]),
                bcrit_weight=jnp.array([.04]), mass=1., speed=1., charge=1.,
                canonical_flux_sign=1, n_eta=9, n_t=129, quadrature_order=24, q_max=2)


def test_physical_flux_local_stencils():
    from vmex.core.deltares import _physical_flux_derivative
    s = jnp.array([.1, .17, .34, .58, .91])
    values = jnp.stack((1 + 3 * s + 4 * s**2, 2 - s), axis=1)
    expected = np.column_stack((3 + 8 * s, -np.ones(5)))
    np.testing.assert_allclose(_physical_flux_derivative(values, -2 * s, -2), expected, atol=2e-14)
    for psi in (jnp.ones(2), jnp.ones((2, 2))):
        with pytest.raises(ValueError):
            _physical_flux_derivative(values, psi, 1.)


def _cosine_action_oracle(parameters, pitch):
    from scipy.special import ellipk, elliprf, elliprj
    b0, eps, K, k = parameters
    low = b0 - eps
    delta = 1 / pitch - low
    r, n = delta / (2 * eps), -delta / low
    third = elliprf(0, 1 - r, 1) + n / 3 * elliprj(0, 1 - r, 1, 1 - n)
    action = 8 * K / k * np.sqrt(pitch * delta * r) * ((1 - 1 / n) * third + ellipk(r) / n) / low
    period = 8 * K / k * np.sqrt(r) / np.sqrt(pitch * delta) / low * third
    root = np.arccos((b0 - 1 / pitch) / eps) / k
    def derivative(j):
        def integrand(angle):
            t, dt = root * np.sin(angle), root * np.cos(angle)
            B = b0 - eps * np.cos(k * t)
            parallel = np.sqrt(max(1 - pitch * B, 1e-30))
            Bx = (1, -np.cos(k * t), 0, eps * t * np.sin(k * t))[j]
            line_x = (j == 2) / B - K * Bx / B**2
            return 2 * (parallel * line_x - pitch * Bx * K / B / (2 * parallel)) * dt
        return quad(integrand, -np.pi / 2, np.pi / 2, epsabs=2e-10, epsrel=2e-10)[0]
    return action, period, np.array([derivative(j) for j in range(4)])


def test_restricted_cosine_independent_action_period_and_derivatives():
    from vmex.core.deltares import restricted_action_deltares_from_boozer
    parameters = jnp.array([1.01, .21, 2.03, .51])
    s = jnp.array([.3, .5, .7])
    slopes = jnp.array([.02, .01, .03, .02])
    def integrals(p):
        profile = p[None] + (s - .5)[:, None] * slopes
        out = restricted_action_deltares_from_boozer(
            bmnc_b=jnp.stack((profile[:, 0], -profile[:, 1]), axis=1),
            xm_b=jnp.array([0, 1]), xn_b=jnp.array([0, 0]), iota_b=profile[:, 3],
            G_b=profile[:, 2], I_b=jnp.zeros(3), nfp=2, psi_b=2 * s, psi_edge=2.,
            bcrit=jnp.array([1.03]), bcrit_weight=jnp.array([.04]),
            mass=1., speed=1., charge=1., canonical_flux_sign=1,
            n_eta=5, n_t=513, quadrature_order=64, p_max=0, q_max=1)
        return jnp.stack((out['action'][1, 0, 0, 0], out['v_tau'][1, 0, 0, 0], out['valid']))
    value, derivative = jax.jit(lambda p: (integrals(p), jax.jacrev(integrals)(p)))(parameters)
    action, period, oracle_derivative = _cosine_action_oracle(np.asarray(parameters), 1 / 1.03)
    assert bool(value[2])
    np.testing.assert_allclose(value[:2], [action, period], rtol=7e-6)
    # Differentiate the smooth integral itself; moving-endpoint action terms vanish.
    np.testing.assert_allclose(derivative[0], oracle_derivative, rtol=4e-4)


def test_restricted_live_coefficient_gradient_parity_and_flux_sign():
    from vmex.core.deltares import restricted_action_deltares_from_boozer as evaluate
    x, direction = jnp.array([0., .002, .001]), jnp.array([.4, .3, -.2])
    scalar = jax.jit(lambda x: evaluate(**_restricted_input(x))['loss'])
    value, gradient = jax.jit(jax.value_and_grad(scalar))(x)
    assert np.isfinite(value) and value > 0
    expected = float(gradient @ direction)
    errors = [abs(float((scalar(x + h * direction) - scalar(x - h * direction)) / (2 * h)) / expected - 1)
              for h in (1e-5, 3e-6, 1e-6)]
    assert min(errors) < 3e-5
    out = evaluate(**_restricted_input(x))
    reversed_sign = evaluate(**dict(_restricted_input(x), canonical_flux_sign=-1))
    np.testing.assert_allclose(reversed_sign['omega'], -out['omega'], atol=1e-13)
    np.testing.assert_allclose(reversed_sign['harmonic_power'], out['harmonic_power'], atol=1e-13)
    mirrored = evaluate(**_restricted_input(x.at[1].set(-x[1])))
    np.testing.assert_allclose(mirrored['action'][:, ::-1], out['action'], rtol=1e-13)
    assert out['valid'] and not out['full_population']
    assert np.all(np.asarray(out['paper_pitch_band_mass']) > 0)
    ratio = evaluate(**dict(_restricted_input(x), rotation='frequency_ratio'))
    assert ratio['valid']
    assert np.max(np.abs(np.asarray(ratio['omega'] - out['omega']))) > 1e-6


@pytest.mark.parametrize('patch,certificate', [
    ({'mass': -1.}, 'particle_valid'), ({'speed': 0.}, 'particle_valid'),
    ({'charge': 0.}, 'particle_valid'), ({'psi_b': np.array([.2, .2, .7])}, 'flux_valid'),
    ({'psi_b': np.array([-.2, .4, .8])}, 'flux_valid'),
    ({'psi_b': np.array([.2, .4, 2.1])}, 'flux_valid'),
    ({'iota_b': np.zeros(3)}, 'chart_valid'),
    ({'iota_b': np.array([-.5, .5, .5])}, 'chart_valid'), ({'nfp': 3}, 'periodic_valid'),
    ({'t_bounds': (-1., 1.)}, None), ({'t_bounds': (-20., 20.)}, None),
    ({'root_residual_limit': 1e-15}, None), ({'root_slope_floor': 1.}, None),
    ({'boundary_forbidden_floor': 1.}, None), ({'endpoint_limit': 1e-6}, 'family_eta_valid'),
])
def test_restricted_invalid_physics_and_coverage(patch, certificate):
    from vmex.core.deltares import restricted_action_deltares_from_boozer
    out = restricted_action_deltares_from_boozer(**dict(_restricted_input(), **patch))
    assert not out['valid'] and np.isnan(out['loss']) and not out['full_population']
    if certificate is not None:
        assert not out[certificate]


@pytest.mark.parametrize('patch', [
    {'n_eta': 4}, {'quadrature_order': 1}, {'t_bounds': (1., -1.)}, {'M': 2},
    {'canonical_flux_sign': 0}, {'rotation': 'unknown'}, {'root_residual_limit': np.inf},
    {'bmnc_b': np.ones(3)}, {'psi_b': np.ones(2)}, {'bcrit_weight': np.ones(2)},
    {'mass': np.ones(2)}, {'speed': np.ones(2)}, {'charge': np.ones(2)},
])
def test_restricted_static_settings_and_shapes(patch):
    from vmex.core.deltares import restricted_action_deltares_from_boozer
    with pytest.raises(ValueError):
        restricted_action_deltares_from_boozer(**dict(_restricted_input(), **patch))


def test_restricted_exact_zero_rotation_is_retained():
    from vmex.core.deltares import restricted_action_deltares_from_boozer
    kwargs = _restricted_input()
    kwargs.update(bmnc_b=jnp.tile(jnp.array([1., -.2, 0.]), (3, 1)),
                  bmns_b=jnp.zeros((3, 3)), iota_b=jnp.full(3, .5),
                  G_b=jnp.full(3, 2.), I_b=jnp.zeros(3))
    out = restricted_action_deltares_from_boozer(**kwargs)
    np.testing.assert_array_equal(out['omega'], 0.)
    assert np.max(np.asarray(out['harmonic_power'])) < 1e-26
    assert np.all(np.asarray(out['v_tau']) > 0)
    # An exact rational plateau has undefined crossing deposition, not zero loss.
    assert not out['valid'] and np.isnan(out['loss'])


def test_fixed_grid_deposition_phase_factor():
    from vmex.core.deltares import linear_crossing_weights
    for phase in (0., .25, .5, .75, 1.):
        weights = linear_crossing_weights(jnp.array([-phase, 1 - phase]), jnp.array([0.]))[:, 0]
        np.testing.assert_allclose(weights.sum(), 1.)
        np.testing.assert_allclose(.5 * jnp.sum(weights**2), phase**2 - phase + .5)


def _alias_input():
    return dict(iota_b=np.ones(3), G_b=np.full(3, 10.), I_b=np.zeros(3), nfp=1,
                psi_b=np.array([.2, .4, .6]), psi_edge=1., bcrit=np.array([2.]),
                bcrit_weight=np.array([.1]), mass=1e-26, speed=1e7, charge=1e-19,
                n_eta=9, n_t=513, quadrature_order=48, q_max=1, p_max=1,
                canonical_flux_sign=1, t_bounds=(-np.pi, np.pi))


@pytest.mark.parametrize('amplitude', [.1, 0.])
def test_restricted_trace_alias_rejected_including_zero_modes(amplitude):
    from vmex.core.deltares import restricted_action_deltares_from_boozer
    # sin(512*t) vanishes on all 513 knots and at the cosine's roots. A small
    # spectral root residual cannot detect the omitted interior wells.
    t = np.linspace(-np.pi, np.pi, 513)
    np.testing.assert_allclose(np.sin(512 * t), 0., atol=4e-13)
    out = restricted_action_deltares_from_boozer(
        bmnc_b=np.array([[2, -.5, 0], [2, -.51, 0], [2, -.52, 0]]),
        bmns_b=np.tile([0, 0, amplitude], (3, 1)),
        xm_b=np.array([0, 1, 512]), xn_b=np.zeros(3), **_alias_input())
    assert out['periodic_valid'] and not out['sampling_valid']
    np.testing.assert_allclose(out['max_trace_phase_step'], 2 * np.pi)
    assert out['root_residual'] < 1e-10 and not out['hard_status']
    assert not out['valid'] and np.isnan(out['loss'])


def test_restricted_eta_alias_rejected_with_resolved_line_sampling():
    from vmex.core.deltares import restricted_action_deltares_from_boozer
    # n=nfp*n_eta has identical phases at every sampled eta, hiding its drive.
    eta = 2 * np.pi * np.arange(9) / 9
    np.testing.assert_allclose(np.sin(9 * eta), 0., atol=3e-14)
    out = restricted_action_deltares_from_boozer(
        bmnc_b=np.array([[2, -.5, 0], [2, -.51, 0], [2, -.52, 0]]),
        bmns_b=np.tile([0, 0, .001], (3, 1)),
        xm_b=np.array([0, 1, 0]), xn_b=np.array([0, 0, 9]), **_alias_input())
    assert out['periodic_valid'] and out['max_trace_phase_step'] < np.pi
    assert out['max_eta_harmonic'] == 9 and not out['sampling_valid']
    assert not out['valid'] and np.isnan(out['loss'])


def test_restricted_physical_mode_periodicity():
    from vmex.core.deltares import restricted_action_deltares_from_boozer
    out = restricted_action_deltares_from_boozer(
        bmnc_b=np.array([[2, -.5], [2.01, -.5], [2.02, -.5]]),
        xm_b=np.array([0, .5]), xn_b=np.zeros(2),
        **dict(_alias_input(), t_bounds=(-5.5, 5.5)))
    assert out['sampling_valid'] and not out['periodic_valid']
    assert not out['valid'] and np.isnan(out['loss'])
    # QI eta closure alone does not require n/nfp to be an integer. Even an
    # unused invalid mode label must fail the physical field-period contract.
    out = restricted_action_deltares_from_boozer(
        bmnc_b=np.array([[2, -.5, 0], [2, -.51, 0], [2, -.52, 0]]),
        xm_b=np.zeros(3), xn_b=np.array([0, 2, 1]),
        **dict(_alias_input(), nfp=2, M=0, N=2, t_bounds=(-np.pi / 2, np.pi / 2)))
    assert out['sampling_valid'] and not out['periodic_valid']
    assert not out['valid'] and np.isnan(out['loss'])


def test_restricted_subgrid_well_rejected_below_nyquist():
    from vmex.core.deltares import restricted_action_deltares_from_boozer
    # Two analytic minima at 0 and pi, separated by maxima at cos(t)=-1/12.
    # The shallow minimum at pi lies below Bcrit but between two above-level
    # knots, creating a hidden pair of roots even though both Nyquist tests pass.
    t = np.linspace(-np.pi / 2, 3 * np.pi / 2, 66)
    def field(t):
        return 2 - .3 * np.cos(2 * t) - .1 * np.cos(t)
    bcrit = 1.80001
    bracket = np.searchsorted(t, np.pi)
    assert field(np.pi) < bcrit < min(field(t[bracket - 1]), field(t[bracket]))
    inside = field(t) < bcrit
    assert np.count_nonzero(~inside[:-1] & inside[1:]) == 1
    out = restricted_action_deltares_from_boozer(
        bmnc_b=np.tile([2, -.3, -.1], (3, 1)), xm_b=np.array([0, 2, 1]), xn_b=np.zeros(3),
        **dict(_alias_input(), G_b=np.array([10., 10.1, 10.2]), bcrit=np.array([bcrit]),
               n_t=66, t_bounds=(-np.pi / 2, 3 * np.pi / 2)))
    assert out['sampling_valid'] and out['periodic_valid'] and not out['hard_status']
    assert out['root_residual'] < .005
    assert not out['well_count_valid'] and out['unresolved_interval_count'] > 0
    assert not out['valid'] and np.isnan(out['loss'])


def test_restricted_negative_subgrid_field_rejected():
    from vmex.core.deltares import restricted_action_deltares_from_boozer
    t = np.linspace(-np.pi / 2, 3 * np.pi / 2, 66)
    # All sampled values are positive, but the analytic minimum is negative.
    assert np.min(1 + 1.000001 * np.cos(t)) > 0
    assert 1 + 1.000001 * np.cos(np.pi) < 0
    out = restricted_action_deltares_from_boozer(
        bmnc_b=np.tile([1, 1.000001], (3, 1)), xm_b=np.array([0, 1]), xn_b=np.zeros(2),
        **dict(_alias_input(), G_b=np.array([10., 10.1, 10.2]), bcrit=np.array([.5]),
               n_t=66, t_bounds=(-np.pi / 2, 3 * np.pi / 2)))
    assert out['sampling_valid'] and out['well_count_valid'] and not out['hard_status']
    assert not out['field_positive_valid']
    assert not out['valid'] and np.isnan(out['loss'])
