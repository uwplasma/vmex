"""Zero-bootstrap control of the mapped pwO field: the unmapped-band rows.

The oracle is the collisionless Shaing-Callen G coefficient of f_s + <u B^2>
(I = 0) on a one-period Boozer grid, normalized as -iota Delta_G / f_t (1 for
a tokamak). Its two-level limit is Delta = sum_i <chi_i>/(iota - a_i).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vmex.core.pwo import mapped_pwo_residual_from_boozer, squeezed_pwo_coordinates

jax.config.update("jax_enable_x64", True)


def _bootstrap_ratio(B, iota, nfp, n_lambda=32):
    """-iota Delta_G/f_t: u, h solved per Fourier mode and zeroed at B_max."""
    nt, nz = B.shape
    weight = np.exp(1e3*(B-B.max())/np.ptp(B))
    weight /= weight.sum()
    m = (np.fft.fftfreq(nt)*nt)[:, None]
    k = (np.fft.fftfreq(nz)*nz*nfp)[None, :]
    d = iota*m+k
    coef = np.where((m != 0) | (k != 0), m*d/np.where(d == 0, 1., d*d+(1e-3*iota*m)**2), 0.)

    def solve(f):
        g = np.fft.ifft2(coef*np.fft.fft2(f, axes=(-2, -1)), axes=(-2, -1)).real
        return g-np.sum(weight*g, axis=(-2, -1), keepdims=True)

    inverse = B**-2
    volume = inverse.mean()
    nodes, quadrature = np.polynomial.legendre.leggauss(n_lambda)
    t = .5*(nodes+1)
    lam = (1-t*t)/B.max()
    xi = np.sqrt(1-lam[:, None, None]*B)
    passing = (xi*inverse).mean(axis=(1, 2))
    g4 = (xi*inverse*solve(1/xi)).mean(axis=(1, 2))/passing
    measure = .75/volume/B.max()*quadrature*t*lam
    f_t = 1-np.sum(measure*volume/passing)
    delta_g = np.sum(measure*g4)-solve(inverse).mean()/volume
    return -iota*delta_g/f_t


def _prototype(w2, *, q=6, nfp=2, t1=-.3, t2=1.8, n=128):
    """Two-level parallelogram field and its corner-connecting iota."""
    w1 = .3*np.pi/nfp
    iota = t2/(np.pi*(1-t1*t2)/(nfp*w1)-1)
    theta, zeta = np.meshgrid(2*np.pi*np.arange(n)/n, 2*np.pi*np.arange(n)/(nfp*n), indexing="ij")
    bump = sum(np.exp(-((u+t1*v)/w1)**(2*q)-((v+t2*u)/w2)**(2*q))
               for u in (zeta-np.pi/nfp+2*np.pi/nfp*j for j in (-1, 0, 1))
               for v in (theta-np.pi+2*np.pi*j for j in (-1, 0, 1)))
    return .95+.1*np.minimum(bump, 1.), iota


def _mapped_field(a, b, c, eta_min, iota, nfp, profile, band, n=128):
    """B(eta) on the mapped domain and ``band(theta, zeta)`` everywhere else."""
    theta = 2*np.pi*np.arange(n)/n
    zeta = 2*np.pi*np.arange(n)/(nfp*n)
    eta = np.linspace(eta_min, 2*np.pi-eta_min, 4001)
    lines = np.asarray(squeezed_pwo_coordinates(theta[:, None], eta[None], iota=iota, nfp=nfp,
                                                a=a, b=b, c=c)["zeta"])
    B = band(*np.meshgrid(theta, zeta, indexing="ij"))
    for row, line in enumerate(lines):
        assert np.all(np.diff(line) > 0) and line[-1]-line[0] < 2*np.pi/nfp
        for shift in 2*np.pi/nfp*np.arange(-1, 2):
            inside = (zeta+shift >= line[0]) & (zeta+shift <= line[-1])
            B[row, inside] = profile(np.interp(zeta[inside]+shift, line, eta))
    return B


def test_oracle_reproduces_two_level_prototype_zero_at_w2_pi():
    zero, iota = _prototype(np.pi)
    half, _ = _prototype(.5*np.pi)
    assert abs(_bootstrap_ratio(zero, iota, 2)) < 5e-3
    assert _bootstrap_ratio(half, iota, 2) > .05
    tokamak = 1+.1*np.cos(2*np.pi*np.arange(64)/64)[:, None]*np.ones((1, 64))
    assert _bootstrap_ratio(tokamak, .42, 2) == pytest.approx(1., abs=1e-2)


MAPS = [(2, .3, 0., -2., 2.9, .45), (2, .6, .8, -2., 3., .45), (3, .3, 0., -.6, 2.5, .7),
        (3, .8, .5, -3., 3.05, .42), (3, .4, -.5, -1., 2.4, .5)]


@pytest.mark.parametrize("nfp,a,b,c,eta_min,iota", MAPS)
def test_mapped_field_with_flat_band_has_zero_bootstrap_for_every_map(nfp, a, b, c, eta_min, iota):
    def flat(theta, zeta):
        return np.full(theta.shape, 1.1)
    two_level = _mapped_field(a, b, c, eta_min, iota, nfp, lambda e: .9+0*e, flat)
    smooth = _mapped_field(a, b, c, eta_min, iota, nfp, lambda e: 1+.1*np.cos(e),
                           lambda t, z: np.full(t.shape, 1+.1*np.cos(eta_min)))
    assert abs(_bootstrap_ratio(two_level, iota, nfp)) < 3e-3
    assert abs(_bootstrap_ratio(smooth, iota, nfp)) < 1e-3


@pytest.mark.parametrize("nfp,a,b,c,eta_min,iota", MAPS[:2])
def test_band_islands_carry_the_bootstrap_current(nfp, a, b, c, eta_min, iota):
    # B_max only on part of the band: lines through the gaps join two wells.
    def islands(theta, zeta):
        return np.where(np.cos(theta) > 0, 1.1, .9)
    field = _mapped_field(a, b, c, eta_min, iota, nfp, lambda e: .9+0*e, islands)
    assert abs(_bootstrap_ratio(field, iota, nfp)) > .03


def _kernel(x, nband=6):
    out = mapped_pwo_residual_from_boozer(
        bmnc_b=x[:3][None], bmns_b=x[3:6][None], xm_b=np.array([0, 1, 1]),
        xn_b=np.array([0, 0, 3]), iota_b=x[6:7], nfp=3, a=x[7], eta_min=x[8], b=.2,
        ngamma=8, neta=6, nband=nband)
    return out


def test_band_rows_match_a_manual_band_evaluation():
    x = jnp.array([1.4, .07, -.05, 0., .02, .03, -.66, .35, 2.9])
    out = _kernel(x)
    assert bool(out["valid"][0])
    gamma = 2*np.pi*np.arange(8)/8
    right = np.asarray(squeezed_pwo_coordinates(gamma, 2*np.pi-2.9, iota=.66, nfp=3, a=.35, b=.2)["zeta"])
    left = np.asarray(squeezed_pwo_coordinates(gamma, 2.9, iota=.66, nfp=3, a=.35, b=.2)["zeta"])+2*np.pi/3
    zeta = -(right[:, None]+(left-right)[:, None]*(np.arange(6)+.5)/6)   # negative iota: mirror
    theta = gamma[:, None]
    field = 1.4+.07*np.cos(theta)-.05*np.cos(theta-3*zeta)+.02*np.sin(theta)+.03*np.sin(theta-3*zeta)
    expected = (field-field.mean())/1.4/np.sqrt(48)
    np.testing.assert_allclose(out["band_residuals1d"], expected.ravel(), atol=1e-14)
    assert float(out["band_total"]) > 0


def test_flat_band_gives_zero_rows_and_axisymmetric_field_does_not():
    flat = _kernel(jnp.array([1.4, 0., 0., 0., 0., 0., .66, .3, 2.8]))
    assert float(flat["band_total"]) < 1e-28
    assert float(_kernel(jnp.array([1.4, .1, 0., 0., 0., 0., .66, .3, 2.8]))["band_total"]) > 1e-4


def test_band_rows_derivatives_match_finite_differences():
    x = jnp.array([1.4, .07, -.05, 0., .02, .03, .66, .35, 2.9])
    direction = jnp.array([.1, -.2, .3, .1, -.1, .2, .05, .2, -.3])
    assert bool(_kernel(x)["valid"][0])
    rows = jax.jit(lambda q: _kernel(q)["band_residuals1d"])
    value, tangent = jax.jvp(rows, (x,), (direction,))
    assert np.all(np.isfinite(tangent)) and np.linalg.norm(tangent) > 0
    _, pullback = jax.vjp(rows, x)
    cotangent = jnp.linspace(-1., 1., value.size)
    np.testing.assert_allclose(jnp.vdot(cotangent, tangent), jnp.vdot(pullback(cotangent)[0], direction),
                               rtol=1e-12)
    for step in (1e-4, 1e-5):
        finite = (rows(x+step*direction)-rows(x-step*direction))/(2*step)
        np.testing.assert_allclose(tangent, finite, rtol=1e-6, atol=1e-10)
