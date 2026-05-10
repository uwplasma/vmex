"""Finite-beta optimization diagnostics and residual helpers.

These helpers are intentionally VMEC-state based and JAX differentiable.  They
cover the global stage-one finite-beta quantities that are cheap and stable to
differentiate through the fixed-boundary discrete-adjoint path: aspect ratio,
iota bounds, volume-averaged field proxy, and total beta.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

from ._compat import jnp
from .energy import flux_profiles_from_indata
from .profiles import eval_profiles
from .solve import _half_mesh_from_full_mesh, _icurv_full_mesh_from_indata, _mass_half_mesh_from_indata
from .vmec_bcovar import vmec_bcovar_half_mesh_from_wout
from .vmec_residue import vmec_force_norms_from_bcovar_dynamic
from .wout import _chipf_from_chips, equilibrium_aspect_ratio_from_state, equilibrium_iota_profiles_from_state

MU0 = 4e-7 * np.pi
ELEMENTARY_CHARGE = 1.602176634e-19

_GL_REDL_X, _GL_REDL_W = np.polynomial.legendre.leggauss(32)
_GL_REDL_X = jnp.asarray(_GL_REDL_X, dtype=jnp.float64)
_GL_REDL_W = jnp.asarray(_GL_REDL_W, dtype=jnp.float64)


@dataclass(frozen=True)
class FiniteBetaTargets:
    """Targets and weights for stage-one finite-beta fixed-boundary objectives."""

    aspect_ratio: float
    min_iota: float
    min_average_iota: float
    max_iota: float
    volavgB: float
    beta_total: float
    aspect_weight: float = 1.0
    iota_weight: float = 1.0
    max_iota_weight: float = 1.0
    volavgB_weight: float = 1.0
    beta_weight: float = 1.0


def mercier_terms_from_profile_integrals(
    *,
    s,
    phips,
    iotas,
    vp,
    pres,
    torcur,
    tpp,
    tbb,
    tjb,
    tjj,
    signgs: int = 1,
) -> dict[str, Any]:
    """Return JAX-differentiable Mercier terms from 1D VMEC profile integrals.

    This is the algebraic core of VMEC's ``mercier.f`` calculation after the
    geometric surface averages have been assembled:

    ``DMerc = DShear + DCurr + DWell + DGeod``.

    The inputs ``tpp``, ``tbb``, ``tjb``, and ``tjj`` are the per-surface
    geometry/current integrals in the same normalization used by the Mercier
    formula, i.e. after the ``(2*pi)^2`` factor applied in ``wout._compute_mercier``.
    This helper is intentionally small and differentiable; the next porting
    step is to replace the remaining NumPy surface-integral assembly with a JAX
    path that feeds this function.
    """
    s = jnp.asarray(s, dtype=jnp.float64)
    phips = jnp.asarray(phips, dtype=jnp.float64)
    iotas = jnp.asarray(iotas, dtype=jnp.float64)
    vp = jnp.asarray(vp, dtype=jnp.float64)
    pres = jnp.asarray(pres, dtype=jnp.float64)
    torcur = jnp.asarray(torcur, dtype=jnp.float64)
    tpp = jnp.asarray(tpp, dtype=jnp.float64)
    tbb = jnp.asarray(tbb, dtype=jnp.float64)
    tjb = jnp.asarray(tjb, dtype=jnp.float64)
    tjj = jnp.asarray(tjj, dtype=jnp.float64)

    ns = int(s.shape[0])
    zeros = jnp.zeros_like(s, dtype=jnp.float64)
    if ns < 3:
        return {
            "DMerc": zeros,
            "Dshear": zeros,
            "Dcurr": zeros,
            "Dwell": zeros,
            "Dgeod": zeros,
            "shear": zeros,
            "vpp": zeros,
            "presp": zeros,
            "ip": zeros,
        }

    sign_jac = jnp.asarray(1.0 if int(signgs) >= 0 else -1.0, dtype=jnp.float64)
    twopi = jnp.asarray(2.0 * np.pi, dtype=jnp.float64)
    hs = jnp.asarray(1.0 / float(ns - 1), dtype=jnp.float64)
    phip_real = twopi * phips * sign_jac
    vp_real = jnp.where(phip_real != 0.0, sign_jac * twopi * twopi * vp / phip_real, 0.0)
    vp_real = vp_real.at[0].set(0.0)

    phip_full = 0.5 * (phip_real[2:] + phip_real[1:-1])
    denom = jnp.where(phip_full != 0.0, 1.0 / (hs * phip_full), 0.0)
    shear_inner = (iotas[2:] - iotas[1:-1]) * denom
    vpp_inner = (vp_real[2:] - vp_real[1:-1]) * denom
    presp_inner = (pres[2:] - pres[1:-1]) * denom
    ip_inner = (torcur[2:] - torcur[1:-1]) * denom

    dshear_inner = 0.25 * shear_inner * shear_inner
    dcurr_inner = -shear_inner * (tjb[1:-1] - ip_inner * tbb[1:-1])
    dwell_inner = presp_inner * (vpp_inner - presp_inner * tpp[1:-1]) * tbb[1:-1]
    dgeod_inner = tjb[1:-1] * tjb[1:-1] - tbb[1:-1] * tjj[1:-1]

    Dshear = zeros.at[1:-1].set(dshear_inner)
    Dcurr = zeros.at[1:-1].set(dcurr_inner)
    Dwell = zeros.at[1:-1].set(dwell_inner)
    Dgeod = zeros.at[1:-1].set(dgeod_inner)
    shear = zeros.at[1:-1].set(shear_inner)
    vpp = zeros.at[1:-1].set(vpp_inner)
    presp = zeros.at[1:-1].set(presp_inner)
    ip = zeros.at[1:-1].set(ip_inner)
    return {
        "DMerc": Dshear + Dcurr + Dwell + Dgeod,
        "Dshear": Dshear,
        "Dcurr": Dcurr,
        "Dwell": Dwell,
        "Dgeod": Dgeod,
        "shear": shear,
        "vpp": vpp,
        "presp": presp,
        "ip": ip,
    }


def mercier_surface_integrals_from_realspace(
    *,
    phips,
    sqrtg,
    b2,
    gpp,
    bdotk_merc,
    wint,
    signgs: int = 1,
) -> dict[str, Any]:
    """Return JAX-differentiable Mercier surface integrals.

    Inputs are real-space arrays on the full radial mesh.  ``b2`` is the
    pressure-subtracted field strength used by VMEC's Mercier formula
    (``2 * (bsq - pressure)`` in the wout path), ``gpp`` is the contravariant
    metric component on the half-mesh surface, ``bdotk_merc`` is VMEC's
    ``mu0 * sqrt(g) J.B`` channel, and ``wint`` are the VMEC quadrature weights
    over ``(theta, zeta)``.

    The returned ``tpp``, ``tbb``, ``tjb``, and ``tjj`` arrays feed directly into
    :func:`mercier_terms_from_profile_integrals`.
    """
    phips = jnp.asarray(phips, dtype=jnp.float64)
    sqrtg = jnp.asarray(sqrtg, dtype=jnp.float64)
    b2 = jnp.asarray(b2, dtype=jnp.float64)
    gpp = jnp.asarray(gpp, dtype=jnp.float64)
    bdotk_merc = jnp.asarray(bdotk_merc, dtype=jnp.float64)
    wint = jnp.asarray(wint, dtype=jnp.float64)
    ns = int(phips.shape[0])
    zeros = jnp.zeros_like(phips, dtype=jnp.float64)
    if ns < 3:
        return {"tpp": zeros, "tbb": zeros, "tjb": zeros, "tjj": zeros}

    sign_jac = jnp.asarray(1.0 if int(signgs) >= 0 else -1.0, dtype=jnp.float64)
    twopi = jnp.asarray(2.0 * np.pi, dtype=jnp.float64)
    phip_real = twopi * phips * sign_jac
    phip_full = 0.5 * (phip_real[2:] + phip_real[1:-1])
    gsqrt_raw = 0.5 * (sqrtg[2:] + sqrtg[1:-1])
    phip_full = phip_full[:, None, None]
    phip_safe = jnp.where(phip_full != 0.0, phip_full, 1.0)
    gsqrt_full = jnp.where(phip_full != 0.0, gsqrt_raw / phip_safe, 0.0)
    b2i = 0.5 * (b2[2:] + b2[1:-1])
    b2_safe = jnp.where(b2i != 0.0, b2i, jnp.asarray(1.0, dtype=jnp.float64))
    norm = twopi * twopi

    weighted_sum = lambda arr: jnp.sum(arr * wint[None, :, :], axis=(1, 2))
    tpp_inner = weighted_sum(gsqrt_full / b2_safe) * norm
    tbb_inner = weighted_sum(b2i * gsqrt_full * gpp[1:-1]) * norm
    bdotj_norm = jnp.where(gsqrt_raw != 0.0, bdotk_merc[1:-1] / gsqrt_raw, 0.0)
    jdotb = bdotj_norm * gpp[1:-1] * gsqrt_full
    tjb_inner = weighted_sum(jdotb) * norm
    tjj_inner = weighted_sum(jdotb * bdotj_norm / b2_safe) * norm

    return {
        "tpp": zeros.at[1:-1].set(tpp_inner),
        "tbb": zeros.at[1:-1].set(tbb_inner),
        "tjb": zeros.at[1:-1].set(tjb_inner),
        "tjj": zeros.at[1:-1].set(tjj_inner),
    }


def jxbforce_profiles_from_realspace(
    *,
    phips,
    sqrtg,
    bsq,
    pres,
    vp,
    bdotk,
    wint,
    sigma_an=None,
    signgs: int = 1,
) -> dict[str, Any]:
    """Return JAX-differentiable JXBFORCE 1D field/current profiles.

    This is the state-level counterpart of VMEC's ``jxbforce`` reductions for
    ``jdotb``, ``bdotb`` and ``bdotgradv``.  Inputs are full-radial-mesh
    real-space arrays in the same normalization used by VMEC's Mercier path.
    The returned arrays live on the full radial mesh, with VMEC's endpoint
    extrapolation convention.
    """
    phips = jnp.asarray(phips, dtype=jnp.float64)
    sqrtg = jnp.asarray(sqrtg, dtype=jnp.float64)
    bsq = jnp.asarray(bsq, dtype=jnp.float64)
    pres = jnp.asarray(pres, dtype=jnp.float64)
    vp = jnp.asarray(vp, dtype=jnp.float64)
    bdotk = jnp.asarray(bdotk, dtype=jnp.float64)
    wint = jnp.asarray(wint, dtype=jnp.float64)
    sigma = jnp.ones_like(sqrtg, dtype=jnp.float64) if sigma_an is None else jnp.asarray(sigma_an, dtype=jnp.float64)

    ns = int(phips.shape[0])
    zeros = jnp.zeros_like(phips, dtype=jnp.float64)
    if ns < 3:
        return {"jdotb": zeros, "bdotb": zeros, "bdotgradv": zeros}

    dnorm1 = jnp.asarray((2.0 * np.pi) ** 2, dtype=jnp.float64)
    sign_jac = jnp.asarray(1.0 if int(signgs) >= 0 else -1.0, dtype=jnp.float64)
    denom = vp[2:] + vp[1:-1]
    ovp = jnp.where(denom != 0.0, 2.0 / denom / dnorm1, 0.0)
    tjnorm = ovp * sign_jac
    weighted_sum = lambda arr: jnp.sum(arr * wint[None, :, :], axis=(1, 2))

    sqgb2 = sqrtg[2:] * (bsq[2:] - pres[2:, None, None]) + sqrtg[1:-1] * (
        bsq[1:-1] - pres[1:-1, None, None]
    )
    sigma_inner = sigma[1:-1]
    jdotb_inner = dnorm1 * tjnorm * weighted_sum(bdotk[1:-1] / sigma_inner)
    bdotb_inner = dnorm1 * tjnorm * weighted_sum(sqgb2 / sigma_inner)
    bdotgradv_inner = 0.5 * dnorm1 * tjnorm * (phips[1:-1] + phips[2:])

    jdotb = zeros.at[1:-1].set(jdotb_inner)
    bdotb = zeros.at[1:-1].set(bdotb_inner)
    bdotgradv = zeros.at[1:-1].set(bdotgradv_inner)

    jdotb = jdotb.at[0].set(2.0 * jdotb[1] - jdotb[2])
    jdotb = jdotb.at[-1].set(2.0 * jdotb[-2] - jdotb[-3])
    bdotb = bdotb.at[0].set(2.0 * bdotb[2] - bdotb[1])
    bdotb = bdotb.at[-1].set(2.0 * bdotb[-2] - bdotb[-3])
    bdotgradv = bdotgradv.at[0].set(2.0 * bdotgradv[1] - bdotgradv[2])
    bdotgradv = bdotgradv.at[-1].set(2.0 * bdotgradv[-2] - bdotgradv[-3])

    return {"jdotb": jdotb, "bdotb": bdotb, "bdotgradv": bdotgradv}


def polynomial_profile_and_derivative(coeffs, s) -> tuple[Any, Any]:
    """Evaluate a polynomial profile and ``d/ds`` derivative.

    Coefficients are in ascending power order, matching SIMSOPT's
    ``ProfilePolynomial`` convention used in the finite-beta stage-one scripts.
    A scalar input is treated as a constant profile.
    """

    coeffs = jnp.ravel(jnp.asarray(coeffs, dtype=jnp.float64))
    s = jnp.asarray(s, dtype=jnp.float64)
    value = jnp.zeros_like(s, dtype=jnp.float64)
    for power in range(int(coeffs.shape[0]) - 1, -1, -1):
        value = value * s + coeffs[power]
    deriv = jnp.zeros_like(s, dtype=jnp.float64)
    for power in range(int(coeffs.shape[0]) - 1, 0, -1):
        deriv = deriv * s + float(power) * coeffs[power]
    return value, deriv


def trapped_fraction_from_modb_sqrtg(*, modB, sqrtg, n_lambda: int = 32) -> dict[str, Any]:
    r"""Return Redl/Sauter geometry moments from ``|B|`` and ``sqrt(g)``.

    This is a differentiable fixed-quadrature counterpart of SIMSOPT's
    ``compute_trapped_fraction``.  Inputs use vmec_jax real-space layout
    ``(ns, ntheta, nzeta)``.  The returned effective trapped fraction is

    .. math::

       f_t = 1 - \frac{3}{4}\langle B^2\rangle
             \int_0^{1/B_{\max}}
             \frac{\lambda\,d\lambda}{\langle\sqrt{1-\lambda B}\rangle}.

    The extrema use JAX ``min``/``max`` on the angular grid rather than the
    cubic-spline refinement in SIMSOPT, so this helper is intended for smooth
    optimization diagnostics and residuals, not as a bitwise replacement for
    the legacy post-processing routine.
    """

    modB = jnp.asarray(modB, dtype=jnp.float64)
    sqrtg = jnp.abs(jnp.asarray(sqrtg, dtype=jnp.float64))
    if int(modB.ndim) != 3 or int(sqrtg.ndim) != 3:
        raise ValueError("modB and sqrtg must have shape (ns, ntheta, nzeta)")
    if modB.shape != sqrtg.shape:
        raise ValueError("modB and sqrtg shape mismatch")

    axes = (1, 2)
    weight = jnp.where(sqrtg > 0.0, sqrtg, jnp.asarray(0.0, dtype=jnp.float64))
    weight_sum = jnp.sum(weight, axis=axes)
    weight_safe = jnp.where(weight_sum != 0.0, weight_sum, jnp.asarray(1.0, dtype=jnp.float64))
    b_safe = jnp.maximum(modB, jnp.asarray(1.0e-300, dtype=jnp.float64))
    Bmin = jnp.min(b_safe, axis=axes)
    Bmax = jnp.max(b_safe, axis=axes)
    fsa_B2 = jnp.sum(b_safe * b_safe * weight, axis=axes) / weight_safe
    fsa_1overB = jnp.sum(weight / b_safe, axis=axes) / weight_safe
    ratio = jnp.where(Bmin > 0.0, Bmax / Bmin, jnp.asarray(1.0, dtype=jnp.float64))
    epsilon = (ratio - 1.0) / (ratio + 1.0)

    nquad = max(1, min(int(n_lambda), int(_GL_REDL_X.shape[0])))
    x = _GL_REDL_X[:nquad]
    w = _GL_REDL_W[:nquad]
    # Use y = sqrt(1 - lambda * Bmax) to remove the endpoint square-root
    # singularity at lambda=1/Bmax.  This is much more accurate for nearly
    # constant-B surfaces than direct lambda quadrature.
    y = 0.5 * (x[:, None] + 1.0)
    weights = 0.5 * w[:, None]
    inv_bmax = 1.0 / jnp.maximum(Bmax, jnp.asarray(1.0e-300, dtype=jnp.float64))
    lambdas = (1.0 - y * y) * inv_bmax[None, :]
    root = jnp.sqrt(jnp.maximum(1.0 - lambdas[:, :, None, None] * b_safe[None, :, :, :], 0.0))
    bounce_avg = jnp.sum(root * weight[None, :, :, :], axis=(2, 3)) / weight_safe[None, :]
    bounce_safe = jnp.where(bounce_avg > 0.0, bounce_avg, jnp.asarray(1.0e-300, dtype=jnp.float64))
    integral = jnp.sum(weights * (2.0 * y * (1.0 - y * y) * inv_bmax[None, :] ** 2) / bounce_safe, axis=0)
    f_t = 1.0 - 0.75 * fsa_B2 * integral

    return {
        "Bmin": Bmin,
        "Bmax": Bmax,
        "epsilon": epsilon,
        "fsa_B2": fsa_B2,
        "fsa_1overB": fsa_1overB,
        "f_t": jnp.clip(f_t, 0.0, 1.0),
    }


def redl_bootstrap_jdotb(
    *,
    s,
    G,
    R,
    iota,
    epsilon,
    f_t,
    psi_edge,
    nfp: int,
    helicity_n: int,
    ne_coeffs,
    Te_coeffs,
    Ti_coeffs=None,
    Zeff_coeffs=1.0,
) -> tuple[Any, dict[str, Any]]:
    r"""Return ``<J.B>`` from the Redl et al. bootstrap-current fit.

    The algebra follows the SIMSOPT implementation of Redl et al.,
    Physics of Plasmas 28, 022502 (2021), using polynomial density and
    temperature profiles in SI units: ``ne`` in ``m^-3`` and ``Te/Ti`` in eV.
    """

    s = jnp.asarray(s, dtype=jnp.float64)
    G = jnp.asarray(G, dtype=jnp.float64)
    R = jnp.asarray(R, dtype=jnp.float64)
    iota = jnp.asarray(iota, dtype=jnp.float64)
    epsilon = jnp.maximum(jnp.asarray(epsilon, dtype=jnp.float64), jnp.asarray(1.0e-8, dtype=jnp.float64))
    f_t = jnp.clip(jnp.asarray(f_t, dtype=jnp.float64), 0.0, 1.0)
    Ti_coeffs = Te_coeffs if Ti_coeffs is None else Ti_coeffs

    ne_s, d_ne_d_s = polynomial_profile_and_derivative(ne_coeffs, s)
    Te_s, d_Te_d_s = polynomial_profile_and_derivative(Te_coeffs, s)
    Ti_s, d_Ti_d_s = polynomial_profile_and_derivative(Ti_coeffs, s)
    Zeff_s, _d_Zeff_d_s = polynomial_profile_and_derivative(jnp.atleast_1d(jnp.asarray(Zeff_coeffs)), s)
    Zeff_s = jnp.maximum(Zeff_s, jnp.asarray(1.0, dtype=jnp.float64))
    ne_s = jnp.maximum(ne_s, jnp.asarray(1.0e-300, dtype=jnp.float64))
    Te_s = jnp.maximum(Te_s, jnp.asarray(1.0e-300, dtype=jnp.float64))
    Ti_s = jnp.maximum(Ti_s, jnp.asarray(1.0e-300, dtype=jnp.float64))
    ni_s = ne_s / Zeff_s
    pe_s = ne_s * Te_s
    pi_s = ni_s * Ti_s

    helicity_N = int(nfp) * int(helicity_n)
    iota_minus_N = iota - float(helicity_N)
    iota_safe = jnp.where(jnp.abs(iota_minus_N) > 1.0e-12, iota_minus_N, jnp.sign(iota_minus_N + 1.0e-300) * 1.0e-12)
    geometry_factor = jnp.abs(R / iota_safe)
    ln_Lambda_e = 31.3 - jnp.log(jnp.sqrt(ne_s) / Te_s)
    ln_Lambda_ii = 30.0 - jnp.log((Zeff_s**3) * jnp.sqrt(ni_s) / (Ti_s**1.5))
    nu_e = geometry_factor * 6.921e-18 * ne_s * Zeff_s * ln_Lambda_e / (Te_s * Te_s * epsilon**1.5)
    nu_i = geometry_factor * 4.90e-18 * ni_s * Zeff_s**4 * ln_Lambda_ii / (Ti_s * Ti_s * epsilon**1.5)
    sqrt_nue = jnp.sqrt(jnp.maximum(nu_e, 0.0))
    sqrt_nui = jnp.sqrt(jnp.maximum(nu_i, 0.0))
    sqrt_zeff_minus_1 = jnp.sqrt(jnp.maximum(Zeff_s - 1.0, 0.0))

    X31 = f_t / (
        1.0
        + (0.67 * (1.0 - 0.7 * f_t) * sqrt_nue) / (0.56 + 0.44 * Zeff_s)
        + (0.52 + 0.086 * sqrt_nue) * (1.0 + 0.87 * f_t) * nu_e / (1.0 + 1.13 * sqrt_zeff_minus_1)
    )
    Zfac = jnp.maximum(Zeff_s**1.2 - 0.71, jnp.asarray(1.0e-12, dtype=jnp.float64))
    L31 = (
        (1.0 + 0.15 / Zfac) * X31
        - 0.22 / Zfac * X31**2
        + 0.01 / Zfac * X31**3
        + 0.06 / Zfac * X31**4
    )

    X32e = f_t / (
        1.0
        + 0.23 * (1.0 - 0.96 * f_t) * sqrt_nue / jnp.sqrt(Zeff_s)
        + 0.13
        * (1.0 - 0.38 * f_t)
        * nu_e
        / (Zeff_s * Zeff_s)
        * (
            jnp.sqrt(1.0 + 2.0 * sqrt_zeff_minus_1)
            + f_t * f_t * jnp.sqrt((0.075 + 0.25 * (Zeff_s - 1.0) ** 2) * nu_e)
        )
    )
    F32ee = (
        (0.1 + 0.6 * Zeff_s) * (X32e - X32e**4) / (Zeff_s * (0.77 + 0.63 * (1.0 + (Zeff_s - 1.0) ** 1.1)))
        + 0.7 / (1.0 + 0.2 * Zeff_s) * (X32e**2 - X32e**4 - 1.2 * (X32e**3 - X32e**4))
        + 1.3 / (1.0 + 0.5 * Zeff_s) * X32e**4
    )

    X32ei = f_t / (
        1.0
        + 0.87 * (1.0 + 0.39 * f_t) * sqrt_nue / (1.0 + 2.95 * (Zeff_s - 1.0) ** 2)
        + 1.53 * (1.0 - 0.37 * f_t) * nu_e * (2.0 + 0.375 * (Zeff_s - 1.0))
    )
    F32ei = (
        -(0.4 + 1.93 * Zeff_s) / (Zeff_s * (0.8 + 0.6 * Zeff_s)) * (X32ei - X32ei**4)
        + 5.5 / (1.5 + 2.0 * Zeff_s) * (X32ei**2 - X32ei**4 - 0.8 * (X32ei**3 - X32ei**4))
        - 1.3 / (1.0 + 0.5 * Zeff_s) * X32ei**4
    )
    L32 = F32ei + F32ee
    L34 = L31

    alpha0 = -(0.62 + 0.055 * (Zeff_s - 1.0)) * (1.0 - f_t) / (
        (0.53 + 0.17 * (Zeff_s - 1.0))
        * (1.0 - (0.31 - 0.065 * (Zeff_s - 1.0)) * f_t - 0.25 * f_t * f_t)
    )
    alpha = ((alpha0 + 0.7 * Zeff_s * jnp.sqrt(f_t * nu_i)) / (1.0 + 0.18 * sqrt_nui) - 0.002 * nu_i * nu_i * f_t**6) / (
        1.0 + 0.004 * nu_i * nu_i * f_t**6
    )

    psi_safe = jnp.where(jnp.abs(psi_edge) > 1.0e-300, jnp.asarray(psi_edge, dtype=jnp.float64), 1.0e-300)
    redl_norm = -G * ELEMENTARY_CHARGE / (psi_safe * iota_safe)
    dnds_term = redl_norm * (ne_s * Te_s + ni_s * Ti_s) * L31 * (d_ne_d_s / ne_s)
    dTeds_term = redl_norm * pe_s * (L31 + L32) * (d_Te_d_s / Te_s)
    dTids_term = redl_norm * pi_s * (L31 + L34 * alpha) * (d_Ti_d_s / Ti_s)
    jdotB = dnds_term + dTeds_term + dTids_term

    details = {
        "s": s,
        "ne_s": ne_s,
        "ni_s": ni_s,
        "Zeff_s": Zeff_s,
        "Te_s": Te_s,
        "Ti_s": Ti_s,
        "d_ne_d_s": d_ne_d_s,
        "d_Te_d_s": d_Te_d_s,
        "d_Ti_d_s": d_Ti_d_s,
        "ln_Lambda_e": ln_Lambda_e,
        "ln_Lambda_ii": ln_Lambda_ii,
        "nu_e_star": nu_e,
        "nu_i_star": nu_i,
        "L31": L31,
        "L32": L32,
        "L34": L34,
        "alpha": alpha,
        "dnds_term": dnds_term,
        "dTeds_term": dTeds_term,
        "dTids_term": dTids_term,
        "jdotB": jdotB,
    }
    return jdotB, details


def _surface_indices(s_grid, surfaces: tuple[float, ...] | None):
    s_grid_np = np.asarray(s_grid, dtype=float)
    if surfaces is None:
        if int(s_grid_np.shape[0]) <= 2:
            return jnp.asarray([], dtype=jnp.int32)
        return jnp.arange(1, int(s_grid_np.shape[0]) - 1, dtype=jnp.int32)
    indices = [int(np.argmin(np.abs(s_grid_np - float(surface)))) for surface in surfaces]
    return jnp.asarray(indices, dtype=jnp.int32)


def redl_bootstrap_geometry_from_state(
    *,
    state,
    static,
    indata,
    signgs: int,
    surfaces: tuple[float, ...] | None = None,
    n_lambda: int = 32,
) -> dict[str, Any]:
    """Return VMEC-state geometry needed by the Redl bootstrap formula.

    The implementation uses the differentiable VMEC real-space field channels
    already used by the Mercier/JXBFORCE path.  Surface selection currently uses
    nearest full-mesh surfaces, matching the other profile objective objects.
    """

    from .vmec_tomnsp import vmec_trig_tables
    from .wout import _vmec_wint_from_trig_jax

    lasym = bool(getattr(static.cfg, "lasym", False))
    s_grid = jnp.asarray(static.s, dtype=jnp.float64)
    trig = getattr(static, "trig_vmec", None)
    if trig is None:
        trig = vmec_trig_tables(
            ntheta=int(static.cfg.ntheta),
            nzeta=int(static.cfg.nzeta),
            nfp=int(static.cfg.nfp),
            mmax=int(static.cfg.mpol) - 1,
            nmax=int(static.cfg.ntor),
            lasym=lasym,
            dtype=jnp.asarray(state.Rcos).dtype,
        )
    wout_like, pres = _wout_like_for_state(state=state, static=static, indata=indata, signgs=int(signgs))
    bc = vmec_bcovar_half_mesh_from_wout(
        state=state,
        static=static,
        wout=wout_like,
        pres=pres,
        use_wout_bsup=False,
        use_wout_bsub_for_lambda=False,
        use_wout_bmag_for_bsq=False,
        use_vmec_synthesis=True,
        trig=trig,
    )
    wint = _vmec_wint_from_trig_jax(trig)
    trapped = trapped_fraction_from_modb_sqrtg(
        modB=jnp.sqrt(jnp.maximum(jnp.asarray(bc.bsq, dtype=jnp.float64), 1.0e-300)),
        sqrtg=bc.jac.sqrtg,
        n_lambda=int(n_lambda),
    )
    I_full = jnp.sum(jnp.asarray(bc.bsubu, dtype=jnp.float64) * wint[None, :, :], axis=(1, 2))
    G_full = jnp.sum(jnp.asarray(bc.bsubv, dtype=jnp.float64) * wint[None, :, :], axis=(1, 2))
    iota_full = jnp.asarray(wout_like.iotas, dtype=jnp.float64)
    R_full = (G_full + iota_full * I_full) * trapped["fsa_1overB"]
    psi_edge = -jnp.asarray(indata.get_float("PHIEDGE", 1.0), dtype=jnp.float64) / jnp.asarray(2.0 * np.pi, dtype=jnp.float64)
    idx = _surface_indices(s_grid, surfaces)
    selected = {
        "indices": idx,
        "s": s_grid[idx],
        "G": G_full[idx],
        "I": I_full[idx],
        "R": R_full[idx],
        "iota": iota_full[idx],
        "psi_edge": psi_edge,
        "nfp": int(static.cfg.nfp),
    }
    for key, value in trapped.items():
        selected[key] = jnp.asarray(value, dtype=jnp.float64)[idx]
    return selected


def redl_bootstrap_mismatch_from_profiles(*, jdotB_vmec, jdotB_redl, eps: float = 1.0e-300) -> Any:
    r"""Return normalized Redl bootstrap-current mismatch residuals."""

    jdotB_vmec = jnp.asarray(jdotB_vmec, dtype=jnp.float64)
    jdotB_redl = jnp.asarray(jdotB_redl, dtype=jnp.float64)
    denominator = jnp.sqrt(jnp.maximum(jnp.sum((jdotB_vmec + jdotB_redl) ** 2), float(eps)))
    return (jdotB_vmec - jdotB_redl) / denominator


def redl_bootstrap_mismatch_from_state(
    *,
    state,
    static,
    indata,
    signgs: int,
    helicity_n: int,
    ne_coeffs,
    Te_coeffs,
    Ti_coeffs=None,
    Zeff_coeffs=1.0,
    surfaces: tuple[float, ...] | None = None,
    n_lambda: int = 32,
    mmax_force: int | None = None,
    nmax_force: int | None = None,
) -> dict[str, Any]:
    """Return differentiable VMEC-vs-Redl bootstrap-current mismatch.

    This is the user-facing finite-beta residual block corresponding to the
    SIMSOPT ``VmecRedlBootstrapMismatch`` objective.  The Redl algebra is the
    same fit formula; the geometry is evaluated from vmec_jax state channels on
    nearest full-mesh surfaces.
    """

    geom = redl_bootstrap_geometry_from_state(
        state=state,
        static=static,
        indata=indata,
        signgs=int(signgs),
        surfaces=surfaces,
        n_lambda=int(n_lambda),
    )
    jdotB_redl, redl_details = redl_bootstrap_jdotb(
        s=geom["s"],
        G=geom["G"],
        R=geom["R"],
        iota=geom["iota"],
        epsilon=geom["epsilon"],
        f_t=geom["f_t"],
        psi_edge=geom["psi_edge"],
        nfp=int(geom["nfp"]),
        helicity_n=int(helicity_n),
        ne_coeffs=ne_coeffs,
        Te_coeffs=Te_coeffs,
        Ti_coeffs=Ti_coeffs,
        Zeff_coeffs=Zeff_coeffs,
    )
    terms = mercier_terms_from_state(
        state=state,
        static=static,
        indata=indata,
        signgs=int(signgs),
        mmax_force=mmax_force,
        nmax_force=nmax_force,
    )
    jdotB_vmec = jnp.asarray(terms["jdotb"], dtype=jnp.float64)[geom["indices"]]
    residuals = redl_bootstrap_mismatch_from_profiles(jdotB_vmec=jdotB_vmec, jdotB_redl=jdotB_redl)
    return {
        "residuals1d": residuals,
        "total": jnp.dot(residuals, residuals),
        "jdotB_vmec": jdotB_vmec,
        "jdotB_redl": jdotB_redl,
        "geometry": geom,
        "redl": redl_details,
    }


def mercier_gpp_from_realspace_geometry(
    *,
    s,
    phips,
    sqrtg,
    R_even,
    R_odd,
    Ru_even,
    Ru_odd,
    Zu_even,
    Zu_odd,
    Rv_even,
    Rv_odd,
    Zv_even,
    Zv_odd,
    signgs: int = 1,
) -> Any:
    """Return VMEC Mercier contravariant ``gpp`` from real-space geometry.

    The inputs are the even/odd VMEC real-space channels used in the Mercier
    path after the internal parity conversion:
    ``X(s,theta,zeta) = X_even + sqrt(s) * X_odd``.  The returned array has the
    same ``(ns, ntheta, nzeta)`` shape as the input geometry and is populated on
    interior full-mesh surfaces.  Endpoints are zero because Mercier terms are
    only defined on ``1 <= js <= ns-2`` in VMEC's convention.
    """
    s = jnp.asarray(s, dtype=jnp.float64)
    phips = jnp.asarray(phips, dtype=jnp.float64)
    sqrtg = jnp.asarray(sqrtg, dtype=jnp.float64)
    R_even = jnp.asarray(R_even, dtype=jnp.float64)
    R_odd = jnp.asarray(R_odd, dtype=jnp.float64)
    Ru_even = jnp.asarray(Ru_even, dtype=jnp.float64)
    Ru_odd = jnp.asarray(Ru_odd, dtype=jnp.float64)
    Zu_even = jnp.asarray(Zu_even, dtype=jnp.float64)
    Zu_odd = jnp.asarray(Zu_odd, dtype=jnp.float64)
    Rv_even = jnp.asarray(Rv_even, dtype=jnp.float64)
    Rv_odd = jnp.asarray(Rv_odd, dtype=jnp.float64)
    Zv_even = jnp.asarray(Zv_even, dtype=jnp.float64)
    Zv_odd = jnp.asarray(Zv_odd, dtype=jnp.float64)

    ns = int(s.shape[0])
    zeros = jnp.zeros_like(sqrtg, dtype=jnp.float64)
    if ns < 3:
        return zeros

    sign_jac = jnp.asarray(1.0 if int(signgs) >= 0 else -1.0, dtype=jnp.float64)
    twopi = jnp.asarray(2.0 * np.pi, dtype=jnp.float64)
    phip_real = twopi * phips * sign_jac
    phip_full = 0.5 * (phip_real[2:] + phip_real[1:-1])
    gsqrt_raw = 0.5 * (sqrtg[2:] + sqrtg[1:-1])
    phip_full = phip_full[:, None, None]
    phip_safe = jnp.where(phip_full != 0.0, phip_full, 1.0)
    gsqrt_full = jnp.where(phip_full != 0.0, gsqrt_raw / phip_safe, 0.0)

    sqs = jnp.sqrt(jnp.maximum(s[1:-1], 0.0))[:, None, None]
    r1f = R_even[1:-1] + sqs * R_odd[1:-1]
    rtf = Ru_even[1:-1] + sqs * Ru_odd[1:-1]
    ztf = Zu_even[1:-1] + sqs * Zu_odd[1:-1]
    rzf = Rv_even[1:-1] + sqs * Rv_odd[1:-1]
    zzf = Zv_even[1:-1] + sqs * Zv_odd[1:-1]
    gtt = rtf * rtf + ztf * ztf
    denom = gtt * r1f * r1f + (rtf * zzf - rzf * ztf) ** 2
    denom_safe = jnp.where(denom != 0.0, denom, 1.0)
    gpp_inner = jnp.where(denom != 0.0, (gsqrt_full * gsqrt_full) / denom_safe, 0.0)
    return zeros.at[1:-1].set(gpp_inner)


def mercier_realspace_geometry_channels_from_state(
    *,
    state,
    modes,
    trig,
    s,
    lconm1: bool = True,
    lthreed: bool = True,
    lasym: bool = False,
    apply_scalxc: bool = True,
    phase_split: bool = False,
) -> dict[str, Any]:
    """Return VMEC even/odd real-space R/Z geometry channels for Mercier.

    This mirrors the geometry synthesis used by the NumPy ``wout`` Mercier
    parity path: VMEC internal Fourier coefficients are split by even/odd
    poloidal mode number, axis rules are applied, and base/theta/zeta
    derivatives are synthesized on the VMEC angular grid.  The odd channels are
    VMEC-internal channels; physical fields are recovered as
    ``X_even + sqrt(s) * X_odd``.
    """
    from .vmec_jacobian import _apply_vmec_axis_rules
    from .vmec_parity import vmec_m1_internal_to_physical_signed
    from .vmec_realspace import vmec_realspace_synthesis_multi

    s = jnp.asarray(s, dtype=jnp.float64)
    m_np = np.asarray(modes.m, dtype=int)
    Rcos = jnp.asarray(state.Rcos, dtype=jnp.float64)
    Rsin = jnp.asarray(state.Rsin, dtype=jnp.float64)
    Zcos = jnp.asarray(state.Zcos, dtype=jnp.float64)
    Zsin = jnp.asarray(state.Zsin, dtype=jnp.float64)

    if bool(lconm1):
        Rcos, Zsin, Rsin, Zcos = vmec_m1_internal_to_physical_signed(
            Rcos=Rcos,
            Zsin=Zsin,
            Rsin=Rsin,
            Zcos=Zcos,
            modes=modes,
            lthreed=bool(lthreed),
            lasym=bool(lasym),
            lconm1=bool(lconm1),
        )

    Rcos = _apply_vmec_axis_rules(Rcos, m_np)
    Rsin = _apply_vmec_axis_rules(Rsin, m_np)
    Zcos = _apply_vmec_axis_rules(Zcos, m_np)
    Zsin = _apply_vmec_axis_rules(Zsin, m_np)

    coeff_cos_stack = jnp.stack([Rcos, Zcos], axis=0)
    coeff_sin_stack = jnp.stack([Rsin, Zsin], axis=0)
    if bool(lasym) and bool(phase_split):
        # LASYM bss half-mesh geometry uses VMEC cos/sin phase channels, not
        # the Mercier/totzsps even/odd poloidal-mode split used for gpp.
        zeros = jnp.zeros_like(coeff_cos_stack)
        coeff_cos = jnp.stack([coeff_cos_stack, zeros], axis=0)
        coeff_sin = jnp.stack([zeros, coeff_sin_stack], axis=0)
        apply_scalxc_local = False
    else:
        dtype = Rcos.dtype
        mask_even = jnp.asarray((m_np % 2) == 0, dtype=dtype)
        mask_odd = 1.0 - mask_even
        mask_stack = jnp.stack([mask_even, mask_odd], axis=0)
        coeff_cos = coeff_cos_stack[None, ...] * mask_stack[:, None, None, :]
        coeff_sin = coeff_sin_stack[None, ...] * mask_stack[:, None, None, :]
        apply_scalxc_local = bool(apply_scalxc)

    stack, stack_t, stack_p = vmec_realspace_synthesis_multi(
        coeff_cos=coeff_cos,
        coeff_sin=coeff_sin,
        modes=modes,
        trig=trig,
        coeffs_internal=True,
        apply_scalxc=apply_scalxc_local,
        s=s,
        derivs=("base", "dtheta", "dzeta"),
    )
    even = stack[0]
    odd = stack[1]
    even_t = stack_t[0]
    odd_t = stack_t[1]
    even_p = stack_p[0]
    odd_p = stack_p[1]
    return {
        "R_even": even[0],
        "R_odd": odd[0],
        "Z_even": even[1],
        "Z_odd": odd[1],
        "Ru_even": even_t[0],
        "Ru_odd": odd_t[0],
        "Zu_even": even_t[1],
        "Zu_odd": odd_t[1],
        "Rv_even": even_p[0],
        "Rv_odd": odd_p[0],
        "Zv_even": even_p[1],
        "Zv_odd": odd_p[1],
    }


def _mercier_symoutput_split_jax(*, f, trig, reversed_sym: bool = False) -> tuple[Any, Any]:
    """JAX VMEC symoutput split into reduced-grid sym/asym channels."""
    f = jnp.asarray(f, dtype=jnp.float64)
    if f.ndim != 3:
        raise ValueError("Expected f with shape (ns, ntheta, nzeta)")
    nt2 = int(trig.ntheta2)
    nt1 = int(trig.ntheta1)
    if int(f.shape[1]) < nt2:
        raise ValueError("Input theta grid is smaller than VMEC ntheta2")
    nzeta = int(f.shape[2])
    i0 = jnp.arange(nt2)
    ir0 = jnp.where(i0 == 0, 0, nt1 - i0)
    kk = jnp.mod(nzeta - jnp.arange(nzeta), nzeta)
    f_half = f[:, :nt2, :]
    f_ref = jnp.take(jnp.take(f, ir0, axis=1), kk, axis=2)
    if bool(reversed_sym):
        sym = 0.5 * (f_half - f_ref)
        asym = 0.5 * (f_half + f_ref)
    else:
        sym = 0.5 * (f_half + f_ref)
        asym = 0.5 * (f_half - f_ref)
    return sym, asym


def _mercier_extend_parity_to_full_jax(*, par0, par1, trig) -> Any:
    """Expand reduced-grid VMEC parity channels to the full LASYM theta grid."""
    par0 = jnp.asarray(par0, dtype=jnp.float64)
    par1 = jnp.asarray(par1, dtype=jnp.float64)
    if par0.shape != par1.shape:
        raise ValueError("parity channel shape mismatch")
    ns, nt2, nzeta = par0.shape
    nt1 = int(trig.ntheta1)
    nt3 = int(getattr(trig, "ntheta3", nt2))
    full = jnp.zeros((int(ns), nt3, int(nzeta)), dtype=par0.dtype)
    full = full.at[:, :nt2, :].set(par0 + par1)
    if nt3 == int(nt2):
        return full

    i0 = np.arange(int(nt2), dtype=int)
    ir0 = np.where(i0 == 0, 0, nt1 - i0)
    mask = ir0 >= int(nt2)
    if not np.any(mask):
        return full
    ir = jnp.asarray(ir0[mask], dtype=jnp.int32)
    kk = jnp.asarray((int(nzeta) - np.arange(int(nzeta), dtype=int)) % int(nzeta), dtype=jnp.int32)
    ref0 = jnp.take(jnp.take(par0, jnp.asarray(np.nonzero(mask)[0], dtype=jnp.int32), axis=1), kk, axis=2)
    ref1 = jnp.take(jnp.take(par1, jnp.asarray(np.nonzero(mask)[0], dtype=jnp.int32), axis=1), kk, axis=2)
    return full.at[:, ir, :].set(ref0 - ref1)


def mercier_bss_geometry_channels_from_state(
    *,
    state,
    modes,
    trig,
    s,
    lthreed: bool = True,
    lasym: bool = False,
    apply_scalxc: bool = True,
) -> dict[str, Any]:
    """Return VMEC bss.f geometry channels for covariant ``B_s`` assembly."""
    if bool(lasym):
        return mercier_realspace_geometry_channels_from_state(
            state=state,
            modes=modes,
            trig=trig,
            s=s,
            lconm1=False,
            lthreed=bool(lthreed),
            lasym=True,
            apply_scalxc=False,
            phase_split=True,
        )

    from .vmec_jacobian import _apply_vmec_axis_rules
    from .vmec_realspace import vmec_realspace_synthesis_multi

    s = jnp.asarray(s, dtype=jnp.float64)
    m_np = np.asarray(modes.m, dtype=int)
    Rcos = _apply_vmec_axis_rules(jnp.asarray(state.Rcos, dtype=jnp.float64), m_np)
    Rsin = _apply_vmec_axis_rules(jnp.asarray(state.Rsin, dtype=jnp.float64), m_np)
    Zcos = _apply_vmec_axis_rules(jnp.asarray(state.Zcos, dtype=jnp.float64), m_np)
    Zsin = _apply_vmec_axis_rules(jnp.asarray(state.Zsin, dtype=jnp.float64), m_np)

    coeff_cos_stack = jnp.stack([Rcos, Zcos], axis=0)
    coeff_sin_stack = jnp.stack([Rsin, Zsin], axis=0)
    zeros = jnp.zeros_like(coeff_cos_stack)
    mask_even = jnp.asarray((m_np % 2) == 0, dtype=jnp.float64)
    mask_m1 = jnp.asarray(m_np == 1, dtype=jnp.float64)
    mask_odd_rest = jnp.asarray(((m_np % 2) == 1) & (m_np != 1), dtype=jnp.float64)

    def _synth(mask, *, apply_scalxc_local: bool):
        coeff_cos = coeff_cos_stack[None, ...] * mask[None, None, None, :]
        coeff_sin = coeff_sin_stack[None, ...] * mask[None, None, None, :]
        base, theta, zeta = vmec_realspace_synthesis_multi(
            coeff_cos=coeff_cos,
            coeff_sin=coeff_sin,
            modes=modes,
            trig=trig,
            coeffs_internal=True,
            apply_scalxc=bool(apply_scalxc_local),
            s=s,
            derivs=("base", "dtheta", "dzeta"),
        )
        return base[0], theta[0], zeta[0]

    even_base, even_t, even_p = _synth(mask_even, apply_scalxc_local=False)
    odd_m1_base, odd_m1_t, odd_m1_p = _synth(mask_m1, apply_scalxc_local=bool(apply_scalxc))
    odd_rest_base, odd_rest_t, odd_rest_p = _synth(mask_odd_rest, apply_scalxc_local=bool(apply_scalxc))
    odd_base = odd_m1_base + odd_rest_base
    odd_t = odd_m1_t + odd_rest_t
    odd_p = odd_m1_p + odd_rest_p
    if int(odd_base.shape[0]) >= 2:
        odd_base = odd_base.at[0].set(odd_m1_base[1])
        odd_t = odd_t.at[0].set(odd_m1_t[1])
        odd_p = odd_p.at[0].set(odd_m1_p[1])

    return {
        "R_even": even_base[0],
        "R_odd": odd_base[0],
        "Z_even": even_base[1],
        "Z_odd": odd_base[1],
        "Ru_even": even_t[0],
        "Ru_odd": odd_t[0],
        "Zu_even": even_t[1],
        "Zu_odd": odd_t[1],
        "Rv_even": even_p[0],
        "Rv_odd": odd_p[0],
        "Zv_even": even_p[1],
        "Zv_odd": odd_p[1],
    }


def mercier_bsubs_derivatives_lasym_false(
    *,
    bsubs,
    trig,
    mmax_force: int,
    nmax_force: int,
) -> dict[str, Any]:
    """Return VMEC jxbforce ``bsubsu``/``bsubsv`` for stellarator symmetry.

    ``bsubs`` must be the VMEC full-mesh covariant radial field channel after
    the jxbforce radial averaging/filtering convention.  This helper ports the
    vectorized stellarator-symmetric branch in :mod:`vmec_jax.wout` to JAX so
    the Mercier ``bdotk`` path can be assembled without NumPy postprocessing.
    """
    bsubs = jnp.asarray(bsubs, dtype=jnp.float64)
    ns, ntheta, nzeta = bsubs.shape
    nt2 = int(trig.ntheta2)
    if int(ntheta) < nt2:
        raise ValueError("bsubs grid smaller than trig.ntheta2")

    mmax = int(mmax_force)
    nmax = int(nmax_force)
    if mmax < 0 or nmax < 0:
        zeros = jnp.zeros((int(ns), nt2, int(nzeta)), dtype=jnp.float64)
        return {"bsubsu": zeros, "bsubsv": zeros}

    cosmui = jnp.asarray(trig.cosmui, dtype=jnp.float64)[:nt2, : mmax + 1]
    sinmui = jnp.asarray(trig.sinmui, dtype=jnp.float64)[:nt2, : mmax + 1]
    cosmu = jnp.asarray(trig.cosmu, dtype=jnp.float64)[:nt2, : mmax + 1]
    sinmu = jnp.asarray(trig.sinmu, dtype=jnp.float64)[:nt2, : mmax + 1]
    cosmum = jnp.asarray(trig.cosmum, dtype=jnp.float64)[:nt2, : mmax + 1]
    sinmum = jnp.asarray(trig.sinmum, dtype=jnp.float64)[:nt2, : mmax + 1]
    cosnv = jnp.asarray(trig.cosnv, dtype=jnp.float64)[:, : nmax + 1]
    sinnv = jnp.asarray(trig.sinnv, dtype=jnp.float64)[:, : nmax + 1]
    cosnvn = jnp.asarray(trig.cosnvn, dtype=jnp.float64)[:, : nmax + 1]
    sinnvn = jnp.asarray(trig.sinnvn, dtype=jnp.float64)[:, : nmax + 1]

    r0scale = float(getattr(trig, "r0scale", 1.0))
    dnorm = jnp.asarray(1.0 / (r0scale**2), dtype=jnp.float64)
    dmult = jnp.full((mmax + 1, nmax + 1), dnorm, dtype=jnp.float64)
    mnyq = int(np.asarray(trig.cosmui).shape[1] - 1)
    nnyq = int(np.asarray(trig.cosnv).shape[1] - 1)
    if mnyq > 0 and mnyq <= mmax:
        dmult = dmult.at[mnyq, :].multiply(0.5)
    if nnyq > 0 and nnyq <= nmax:
        dmult = dmult.at[:, nnyq].multiply(0.5)

    bsubs_nt2 = bsubs[:, :nt2, :]
    f_theta_sin = jnp.einsum("sik,im->smk", bsubs_nt2, sinmui, optimize=True)
    f_theta_cos = jnp.einsum("sik,im->smk", bsubs_nt2, cosmui, optimize=True)
    bsubsmn1 = jnp.einsum("smk,kn->smn", f_theta_sin, cosnv, optimize=True) * dmult[None, :, :]
    bsubsmn2 = jnp.einsum("smk,kn->smn", f_theta_cos, sinnv, optimize=True) * dmult[None, :, :]

    tmp_su_1 = jnp.einsum("smn,im->sin", bsubsmn1, cosmum, optimize=True)
    tmp_su_2 = jnp.einsum("smn,im->sin", bsubsmn2, sinmum, optimize=True)
    bsubsu = jnp.einsum("sin,kn->sik", tmp_su_1, cosnv, optimize=True) + jnp.einsum(
        "sin,kn->sik", tmp_su_2, sinnv, optimize=True
    )

    tmp_sv_1 = jnp.einsum("smn,im->sin", bsubsmn1, sinmu, optimize=True)
    tmp_sv_2 = jnp.einsum("smn,im->sin", bsubsmn2, cosmu, optimize=True)
    bsubsv = jnp.einsum("sin,kn->sik", tmp_sv_1, sinnvn, optimize=True) + jnp.einsum(
        "sin,kn->sik", tmp_sv_2, cosnvn, optimize=True
    )
    return {"bsubsu": bsubsu, "bsubsv": bsubsv}


def mercier_bsubs_derivatives_lasym_true(
    *,
    bsubs,
    trig,
    mmax_force: int,
    nmax_force: int,
) -> dict[str, Any]:
    """Return VMEC jxbforce ``bsubsu``/``bsubsv`` for LASYM equilibria."""
    bsubs = jnp.asarray(bsubs, dtype=jnp.float64)
    ns, ntheta, nzeta = bsubs.shape
    nt2 = int(trig.ntheta2)
    nt3 = int(getattr(trig, "ntheta3", nt2))
    if int(ntheta) < nt3:
        raise ValueError("LASYM bsubs grid smaller than trig.ntheta3")

    mmax = int(mmax_force)
    nmax = int(nmax_force)
    if mmax < 0 or nmax < 0:
        zeros = jnp.zeros((int(ns), nt3, int(nzeta)), dtype=jnp.float64)
        return {"bsubsu": zeros, "bsubsv": zeros}

    cosmui = jnp.asarray(trig.cosmui, dtype=jnp.float64)[:nt2, : mmax + 1]
    sinmui = jnp.asarray(trig.sinmui, dtype=jnp.float64)[:nt2, : mmax + 1]
    cosmu = jnp.asarray(trig.cosmu, dtype=jnp.float64)[:nt2, : mmax + 1]
    sinmu = jnp.asarray(trig.sinmu, dtype=jnp.float64)[:nt2, : mmax + 1]
    cosmum = jnp.asarray(trig.cosmum, dtype=jnp.float64)[:nt2, : mmax + 1]
    sinmum = jnp.asarray(trig.sinmum, dtype=jnp.float64)[:nt2, : mmax + 1]
    cosnv = jnp.asarray(trig.cosnv, dtype=jnp.float64)[:, : nmax + 1]
    sinnv = jnp.asarray(trig.sinnv, dtype=jnp.float64)[:, : nmax + 1]
    cosnvn = jnp.asarray(trig.cosnvn, dtype=jnp.float64)[:, : nmax + 1]
    sinnvn = jnp.asarray(trig.sinnvn, dtype=jnp.float64)[:, : nmax + 1]

    r0scale = float(getattr(trig, "r0scale", 1.0))
    dnorm = jnp.asarray(1.0 / (r0scale**2), dtype=jnp.float64)
    dmult = jnp.full((mmax + 1, nmax + 1), dnorm, dtype=jnp.float64)
    mnyq = int(np.asarray(trig.cosmui).shape[1] - 1)
    nnyq = int(np.asarray(trig.cosnv).shape[1] - 1)
    if mnyq > 0 and mnyq <= mmax:
        dmult = dmult.at[mnyq, :].multiply(0.5)
    if nnyq > 0 and nnyq <= nmax:
        dmult = dmult.at[:, nnyq].multiply(0.5)

    bsubs_sym, bsubs_asym = _mercier_symoutput_split_jax(f=bsubs, trig=trig, reversed_sym=True)

    f_theta_sin = jnp.einsum("sik,im->smk", bsubs_sym[:, :nt2, :], sinmui, optimize=True)
    f_theta_cos = jnp.einsum("sik,im->smk", bsubs_sym[:, :nt2, :], cosmui, optimize=True)
    bsubsmn1 = jnp.einsum("smk,kn->smn", f_theta_sin, cosnv, optimize=True) * dmult[None, :, :]
    bsubsmn2 = jnp.einsum("smk,kn->smn", f_theta_cos, sinnv, optimize=True) * dmult[None, :, :]

    tmp_su_1 = jnp.einsum("smn,im->sin", bsubsmn1, cosmum, optimize=True)
    tmp_su_2 = jnp.einsum("smn,im->sin", bsubsmn2, sinmum, optimize=True)
    bsubsu_s = jnp.einsum("sin,kn->sik", tmp_su_1, cosnv, optimize=True) + jnp.einsum(
        "sin,kn->sik", tmp_su_2, sinnv, optimize=True
    )

    tmp_sv_1 = jnp.einsum("smn,im->sin", bsubsmn1, sinmu, optimize=True)
    tmp_sv_2 = jnp.einsum("smn,im->sin", bsubsmn2, cosmu, optimize=True)
    bsubsv_s = jnp.einsum("sin,kn->sik", tmp_sv_1, sinnvn, optimize=True) + jnp.einsum(
        "sin,kn->sik", tmp_sv_2, cosnvn, optimize=True
    )

    f_theta_cos_a = jnp.einsum("sik,im->smk", bsubs_asym[:, :nt2, :], cosmui, optimize=True)
    f_theta_sin_a = jnp.einsum("sik,im->smk", bsubs_asym[:, :nt2, :], sinmui, optimize=True)
    bsubsmn3 = jnp.einsum("smk,kn->smn", f_theta_cos_a, cosnv, optimize=True) * dmult[None, :, :]
    bsubsmn4 = jnp.einsum("smk,kn->smn", f_theta_sin_a, sinnv, optimize=True) * dmult[None, :, :]

    tmp_su_3 = jnp.einsum("smn,im->sin", bsubsmn3, sinmum, optimize=True)
    tmp_su_4 = jnp.einsum("smn,im->sin", bsubsmn4, cosmum, optimize=True)
    bsubsu_a = jnp.einsum("sin,kn->sik", tmp_su_3, cosnv, optimize=True) + jnp.einsum(
        "sin,kn->sik", tmp_su_4, sinnv, optimize=True
    )

    tmp_sv_3 = jnp.einsum("smn,im->sin", bsubsmn3, cosmu, optimize=True)
    tmp_sv_4 = jnp.einsum("smn,im->sin", bsubsmn4, sinmu, optimize=True)
    bsubsv_a = jnp.einsum("sin,kn->sik", tmp_sv_3, sinnvn, optimize=True) + jnp.einsum(
        "sin,kn->sik", tmp_sv_4, cosnvn, optimize=True
    )

    return {
        "bsubsu": _mercier_extend_parity_to_full_jax(par0=bsubsu_s, par1=bsubsu_a, trig=trig),
        "bsubsv": _mercier_extend_parity_to_full_jax(par0=bsubsv_s, par1=bsubsv_a, trig=trig),
    }


def mercier_bsubs_half_mesh_from_geometry(
    *,
    bsupu,
    bsupv,
    rs12,
    zs12,
    ru12,
    zu12,
    rv12,
    zv12,
) -> dict[str, Any]:
    """Return VMEC half-mesh ``bsubs`` from geometry and contravariant B.

    This is the differentiable core of VMEC's ``bss.f`` radial covariant field
    assembly once the half-mesh geometric channels have been synthesized:

    ``B_s = B^u (R_s R_u + Z_s Z_u) + B^v (R_s R_v + Z_s Z_v)``.
    """
    bsupu = jnp.asarray(bsupu, dtype=jnp.float64)
    bsupv = jnp.asarray(bsupv, dtype=jnp.float64)
    rs12 = jnp.asarray(rs12, dtype=jnp.float64)
    zs12 = jnp.asarray(zs12, dtype=jnp.float64)
    ru12 = jnp.asarray(ru12, dtype=jnp.float64)
    zu12 = jnp.asarray(zu12, dtype=jnp.float64)
    rv12 = jnp.asarray(rv12, dtype=jnp.float64)
    zv12 = jnp.asarray(zv12, dtype=jnp.float64)

    g_su = rs12 * ru12 + zs12 * zu12
    g_sv = rs12 * rv12 + zs12 * zv12
    bsubs = bsupu * g_su + bsupv * g_sv
    return {"bsubs": bsubs, "g_su": g_su, "g_sv": g_sv}


def mercier_zeta_half_mesh_from_realspace_geometry(
    *,
    s,
    Rv_even,
    Rv_odd,
    Zv_even,
    Zv_odd,
) -> dict[str, Any]:
    """Return VMEC half-mesh ``rv12``/``zv12`` from parity geometry channels."""
    s = jnp.asarray(s, dtype=jnp.float64)
    Rv_even = jnp.asarray(Rv_even, dtype=jnp.float64)
    Rv_odd = jnp.asarray(Rv_odd, dtype=jnp.float64)
    Zv_even = jnp.asarray(Zv_even, dtype=jnp.float64)
    Zv_odd = jnp.asarray(Zv_odd, dtype=jnp.float64)

    zeros = jnp.zeros_like(Rv_even, dtype=jnp.float64)
    ns = int(s.shape[0])
    if ns < 2:
        return {"rv12": zeros, "zv12": zeros}

    sh = jnp.sqrt(jnp.maximum(0.5 * (s[1:] + s[:-1]), 0.0))[:, None, None]
    rv_inner = 0.5 * (Rv_even[1:] + Rv_even[:-1] + sh * (Rv_odd[1:] + Rv_odd[:-1]))
    zv_inner = 0.5 * (Zv_even[1:] + Zv_even[:-1] + sh * (Zv_odd[1:] + Zv_odd[:-1]))
    rv12 = zeros.at[1:].set(rv_inner)
    zv12 = zeros.at[1:].set(zv_inner)
    rv12 = rv12.at[0].set(rv_inner[0])
    zv12 = zv12.at[0].set(zv_inner[0])
    return {"rv12": rv12, "zv12": zv12}


def mercier_bss_half_mesh_geometry_from_realspace(
    *,
    s,
    rs,
    zs,
    R_odd,
    Z_odd,
    Rv_even,
    Rv_odd,
    Zv_even,
    Zv_odd,
) -> dict[str, Any]:
    """Return VMEC bss half-mesh geometry corrections used for ``B_s``."""
    s = jnp.asarray(s, dtype=jnp.float64)
    rs = jnp.asarray(rs, dtype=jnp.float64)
    zs = jnp.asarray(zs, dtype=jnp.float64)
    R_odd = jnp.asarray(R_odd, dtype=jnp.float64)
    Z_odd = jnp.asarray(Z_odd, dtype=jnp.float64)
    zeta = mercier_zeta_half_mesh_from_realspace_geometry(
        s=s,
        Rv_even=Rv_even,
        Rv_odd=Rv_odd,
        Zv_even=Zv_even,
        Zv_odd=Zv_odd,
    )

    ns = int(s.shape[0])
    rs12 = jnp.zeros_like(rs, dtype=jnp.float64)
    zs12 = jnp.zeros_like(zs, dtype=jnp.float64)
    if ns < 2:
        return {"rs12": rs12, "zs12": zs12, **zeta}

    sh = jnp.sqrt(jnp.maximum(0.5 * (s[1:] + s[:-1]), 0.0))[:, None, None]
    sh_safe = jnp.where(sh != 0.0, sh, 1.0)
    dphids = jnp.asarray(0.25, dtype=jnp.float64)
    rs_inner = rs[1:] + dphids * (R_odd[1:] + R_odd[:-1]) / sh_safe
    zs_inner = zs[1:] + dphids * (Z_odd[1:] + Z_odd[:-1]) / sh_safe
    rs12 = rs12.at[1:].set(rs_inner)
    zs12 = zs12.at[1:].set(zs_inner)
    rs12 = rs12.at[0].set(rs_inner[0])
    zs12 = zs12.at[0].set(zs_inner[0])
    return {"rs12": rs12, "zs12": zs12, **zeta}


def mercier_bsubs_full_mesh_from_half_mesh(*, bsubs_half) -> Any:
    """Average half-mesh ``bsubs`` to VMEC's jxbforce full-mesh convention."""
    bsubs_half = jnp.asarray(bsubs_half, dtype=jnp.float64)
    ns = int(bsubs_half.shape[0])
    bsubs_full = jnp.array(bsubs_half)
    if ns > 2:
        bsubs_full = bsubs_full.at[1:-1].set(0.5 * (bsubs_half[1:-1] + bsubs_half[2:]))
    if ns > 0:
        bsubs_full = bsubs_full.at[0].set(jnp.zeros_like(bsubs_full[0]))
    return bsubs_full


def mercier_bdotk_from_covariant_derivatives(
    *,
    bsubu,
    bsubv,
    bsubsu,
    bsubsv,
    s,
) -> dict[str, Any]:
    """Return VMEC Mercier ``bdotk`` channels from covariant field derivatives.

    This is the JAX equivalent of the small jxbforce block that forms
    ``itheta``, ``izeta``, ``bdotk``, and ``bdotk_merc`` once the filtered
    covariant fields and their angular derivatives are available.
    """
    bsubu = jnp.asarray(bsubu, dtype=jnp.float64)
    bsubv = jnp.asarray(bsubv, dtype=jnp.float64)
    bsubsu = jnp.asarray(bsubsu, dtype=jnp.float64)
    bsubsv = jnp.asarray(bsubsv, dtype=jnp.float64)
    s = jnp.asarray(s, dtype=jnp.float64)
    ns = int(s.shape[0])
    zeros = jnp.zeros_like(bsubu, dtype=jnp.float64)
    if ns < 3:
        return {
            "itheta": zeros,
            "izeta": zeros,
            "bdotk": zeros,
            "bdotk_merc": zeros,
        }

    hs = jnp.asarray(1.0 / float(ns - 1), dtype=jnp.float64)
    ohs = 1.0 / hs
    itheta_inner = bsubsv[1:-1] - ohs * (bsubv[2:] - bsubv[1:-1])
    izeta_inner = -bsubsu[1:-1] + ohs * (bsubu[2:] - bsubu[1:-1])
    itheta = zeros.at[1:-1].set(itheta_inner)
    izeta = zeros.at[1:-1].set(izeta_inner)
    izeta = izeta.at[0].set(2.0 * izeta[1] - izeta[2])
    izeta = izeta.at[-1].set(2.0 * izeta[-2] - izeta[-3])

    itheta = itheta / jnp.asarray(MU0, dtype=jnp.float64)
    izeta = izeta / jnp.asarray(MU0, dtype=jnp.float64)
    bsubu1 = 0.5 * (bsubu[2:] + bsubu[1:-1])
    bsubv1 = 0.5 * (bsubv[2:] + bsubv[1:-1])
    bdotk_inner = itheta[1:-1] * bsubu1 + izeta[1:-1] * bsubv1
    bdotk = zeros.at[1:-1].set(bdotk_inner)
    bdotk_merc = jnp.asarray(MU0, dtype=jnp.float64) * bdotk
    return {
        "itheta": itheta,
        "izeta": izeta,
        "bdotk": bdotk,
        "bdotk_merc": bdotk_merc,
    }


def _s_half_from_static(static):
    s = jnp.asarray(static.s)
    if int(s.shape[0]) < 2:
        return s
    return jnp.concatenate([s[:1], 0.5 * (s[1:] + s[:-1])], axis=0)


def _wout_like_for_state(*, state, static, indata, signgs: int):
    s = jnp.asarray(static.s)
    flux = flux_profiles_from_indata(indata, s, signgs=int(signgs))
    phips = jnp.asarray(flux.phips)
    if int(phips.shape[0]) > 0:
        phips = phips.at[0].set(0.0)

    s_half = _s_half_from_static(static)
    prof = eval_profiles(indata, s_half)
    pres = jnp.asarray(prof.get("pressure", jnp.zeros_like(s_half)))
    if int(pres.shape[0]) > 0:
        pres = pres.at[0].set(0.0)

    chips, iotas, iotaf = equilibrium_iota_profiles_from_state(
        state=state,
        static=static,
        indata=indata,
        signgs=int(signgs),
    )
    chipf = _chipf_from_chips(chips)

    from .boundary import boundary_from_indata

    boundary = boundary_from_indata(indata, static.modes)
    mode_m = np.asarray(static.modes.m)
    mode_n = np.asarray(static.modes.n)
    idx00 = np.where((mode_m == 0) & (mode_n == 0))[0]
    r00 = float(np.asarray(boundary.R_cos)[int(idx00[0])]) if idx00.size else float(np.asarray(boundary.R_cos)[0])

    gamma = float(indata.get_float("GAMMA", 0.0))
    lrfp = bool(indata.get_bool("LRFP", False))
    chips_half = _half_mesh_from_full_mesh(jnp.asarray(flux.chipf)) if lrfp else None
    mass = _mass_half_mesh_from_indata(
        indata=indata,
        s_full=s,
        phips=phips,
        r00=r00,
        gamma=gamma,
        lrfp=lrfp,
        chips=chips_half,
    )
    icurv = _icurv_full_mesh_from_indata(indata=indata, s_full=s, signgs=int(signgs))

    wout_like = SimpleNamespace(
        phipf=jnp.asarray(flux.phipf),
        phips=phips,
        chipf=jnp.asarray(chipf),
        iotaf=jnp.asarray(iotaf),
        iotas=jnp.asarray(iotas),
        signgs=int(signgs),
        nfp=int(static.cfg.nfp),
        mpol=int(static.cfg.mpol),
        ntor=int(static.cfg.ntor),
        lasym=bool(static.cfg.lasym),
        flux_is_internal=True,
        ncurr=int(indata.get_int("NCURR", 0)),
        lcurrent=bool(indata.get_int("NCURR", 0) == 1),
        icurv=jnp.asarray(icurv),
        mass=jnp.asarray(mass),
        gamma=gamma,
    )
    return wout_like, pres


def mercier_terms_from_state(
    *,
    state,
    static,
    indata,
    signgs: int,
    mmax_force: int | None = None,
    nmax_force: int | None = None,
    include_channels: bool = False,
) -> dict[str, Any]:
    """Return differentiable VMEC Mercier terms from a VMEC state.

    This state-level composition uses the JAX Mercier geometry and jxbforce
    derivative paths for both stellarator-symmetric and LASYM equilibria.
    """
    from .vmec_tomnsp import vmec_trig_tables
    from .wout import _vmec_wint_from_trig_jax

    lasym = bool(getattr(static.cfg, "lasym", False))
    s = jnp.asarray(static.s, dtype=jnp.float64)
    trig = getattr(static, "trig_vmec", None)
    if trig is None:
        trig = vmec_trig_tables(
            ntheta=int(static.cfg.ntheta),
            nzeta=int(static.cfg.nzeta),
            nfp=int(static.cfg.nfp),
            mmax=int(static.cfg.mpol) - 1,
            nmax=int(static.cfg.ntor),
            lasym=lasym,
            dtype=jnp.asarray(state.Rcos).dtype,
        )
    mmax = int(static.cfg.mpol) - 1 if mmax_force is None else int(mmax_force)
    nmax = int(static.cfg.ntor) if nmax_force is None else int(nmax_force)

    wout_like, pres = _wout_like_for_state(state=state, static=static, indata=indata, signgs=int(signgs))
    bc = vmec_bcovar_half_mesh_from_wout(
        state=state,
        static=static,
        wout=wout_like,
        pres=pres,
        use_wout_bsup=False,
        use_wout_bsub_for_lambda=False,
        use_wout_bmag_for_bsq=False,
        use_vmec_synthesis=True,
        trig=trig,
    )
    norms = vmec_force_norms_from_bcovar_dynamic(
        bc=bc,
        trig=trig,
        s=s,
        signgs=int(signgs),
    )
    geom = mercier_realspace_geometry_channels_from_state(
        state=state,
        modes=static.modes,
        trig=trig,
        s=s,
        lconm1=bool(getattr(static.cfg, "lconm1", True)),
        lthreed=bool(getattr(static.cfg, "lthreed", True)),
        lasym=lasym,
        apply_scalxc=True,
        phase_split=False,
    )
    bss_geom = mercier_bss_geometry_channels_from_state(
        state=state,
        modes=static.modes,
        trig=trig,
        s=s,
        lthreed=bool(getattr(static.cfg, "lthreed", True)),
        lasym=lasym,
        apply_scalxc=True,
    )
    bss_half_geom = mercier_bss_half_mesh_geometry_from_realspace(
        s=s,
        rs=bc.jac.rs,
        zs=bc.jac.zs,
        R_odd=bss_geom["R_odd"],
        Z_odd=bss_geom["Z_odd"],
        Rv_even=bss_geom["Rv_even"],
        Rv_odd=bss_geom["Rv_odd"],
        Zv_even=bss_geom["Zv_even"],
        Zv_odd=bss_geom["Zv_odd"],
    )
    bsubs_half = mercier_bsubs_half_mesh_from_geometry(
        bsupu=bc.bsupu,
        bsupv=bc.bsupv,
        rs12=bss_half_geom["rs12"],
        zs12=bss_half_geom["zs12"],
        ru12=bc.jac.ru12,
        zu12=bc.jac.zu12,
        rv12=bss_half_geom["rv12"],
        zv12=bss_half_geom["zv12"],
    )
    bsubs_full = mercier_bsubs_full_mesh_from_half_mesh(bsubs_half=bsubs_half["bsubs"])
    if lasym:
        bsubs_derivs = mercier_bsubs_derivatives_lasym_true(
            bsubs=bsubs_full,
            trig=trig,
            mmax_force=mmax,
            nmax_force=nmax,
        )
    else:
        bsubs_derivs = mercier_bsubs_derivatives_lasym_false(
            bsubs=bsubs_full,
            trig=trig,
            mmax_force=mmax,
            nmax_force=nmax,
        )
    bdotk = mercier_bdotk_from_covariant_derivatives(
        bsubu=bc.bsubu,
        bsubv=bc.bsubv,
        bsubsu=bsubs_derivs["bsubsu"],
        bsubsv=bsubs_derivs["bsubsv"],
        s=s,
    )
    gpp = mercier_gpp_from_realspace_geometry(
        s=s,
        phips=wout_like.phips,
        sqrtg=bc.jac.sqrtg,
        R_even=geom["R_even"],
        R_odd=geom["R_odd"],
        Ru_even=geom["Ru_even"],
        Ru_odd=geom["Ru_odd"],
        Zu_even=geom["Zu_even"],
        Zu_odd=geom["Zu_odd"],
        Rv_even=geom["Rv_even"],
        Rv_odd=geom["Rv_odd"],
        Zv_even=geom["Zv_even"],
        Zv_odd=geom["Zv_odd"],
        signgs=int(signgs),
    )
    b2 = 2.0 * (jnp.asarray(bc.bsq, dtype=jnp.float64) - jnp.asarray(pres, dtype=jnp.float64)[:, None, None])
    surface = mercier_surface_integrals_from_realspace(
        phips=wout_like.phips,
        sqrtg=bc.jac.sqrtg,
        b2=b2,
        gpp=gpp,
        bdotk_merc=bdotk["bdotk_merc"],
        wint=_vmec_wint_from_trig_jax(trig),
        signgs=int(signgs),
    )
    wint = _vmec_wint_from_trig_jax(trig)
    jxb = jxbforce_profiles_from_realspace(
        phips=wout_like.phips,
        sqrtg=bc.jac.sqrtg,
        bsq=bc.bsq,
        pres=pres,
        vp=norms.vp,
        bdotk=bdotk["bdotk"],
        wint=wint,
        signgs=int(signgs),
    )
    torcur = jnp.zeros_like(s, dtype=jnp.float64)
    if int(s.shape[0]) > 1:
        torcur_inner = jnp.asarray(float(signgs) * 2.0 * np.pi, dtype=jnp.float64) * jnp.sum(
            jnp.asarray(bc.bsubu, dtype=jnp.float64)[1:] * wint[None, :, :],
            axis=(1, 2),
        )
        torcur = torcur.at[1:].set(torcur_inner)

    terms = mercier_terms_from_profile_integrals(
        s=s,
        phips=wout_like.phips,
        iotas=wout_like.iotas,
        vp=norms.vp,
        pres=pres,
        torcur=torcur,
        tpp=surface["tpp"],
        tbb=surface["tbb"],
        tjb=surface["tjb"],
        tjj=surface["tjj"],
        signgs=int(signgs),
    )
    out = {
        **terms,
        **surface,
        **jxb,
        "torcur": torcur,
        "vp": norms.vp,
    }
    if include_channels:
        out.update(
            {
                "gpp": gpp,
                "bsubs_half": bsubs_half["bsubs"],
                "bsubs_full": bsubs_full,
                "bsubsu": bsubs_derivs["bsubsu"],
                "bsubsv": bsubs_derivs["bsubsv"],
                "itheta": bdotk["itheta"],
                "izeta": bdotk["izeta"],
                "bdotk": bdotk["bdotk"],
                "bdotk_merc": bdotk["bdotk_merc"],
                "sqrtg": bc.jac.sqrtg,
            }
        )
    return out


def finite_beta_scalars_from_state(*, state, static, indata, signgs: int) -> dict[str, Any]:
    """Return JAX-differentiable finite-beta scalar diagnostics from a VMEC state."""
    aspect = equilibrium_aspect_ratio_from_state(state=state, static=static)
    _chips, iotas, iotaf = equilibrium_iota_profiles_from_state(
        state=state,
        static=static,
        indata=indata,
        signgs=int(signgs),
    )
    iotaf = jnp.asarray(iotaf, dtype=jnp.float64)

    wout_like, pres = _wout_like_for_state(state=state, static=static, indata=indata, signgs=int(signgs))
    bc = vmec_bcovar_half_mesh_from_wout(
        state=state,
        static=static,
        wout=wout_like,
        pres=pres,
        use_wout_bsup=False,
        use_wout_bsub_for_lambda=False,
        use_wout_bmag_for_bsq=False,
        use_vmec_synthesis=True,
        trig=None,
    )
    norms = vmec_force_norms_from_bcovar_dynamic(
        bc=bc,
        trig=static.trig_vmec,
        s=jnp.asarray(static.s),
        signgs=int(signgs),
    )
    beta_total = jnp.where(norms.wb != 0.0, norms.wp / norms.wb, jnp.asarray(0.0, dtype=norms.wb.dtype))
    volavgB = jnp.sqrt(jnp.maximum(2.0 * norms.wb / jnp.maximum(norms.volume, 1e-300), 0.0))
    return {
        "aspect": aspect,
        "iotas": jnp.asarray(iotas, dtype=jnp.float64),
        "iotaf": iotaf,
        "mean_iota": jnp.mean(jnp.abs(iotaf[1:])) if int(iotaf.shape[0]) > 1 else jnp.asarray(0.0),
        "min_iota": jnp.min(jnp.abs(iotaf[1:])) if int(iotaf.shape[0]) > 1 else jnp.asarray(0.0),
        "max_iota": jnp.max(jnp.abs(iotaf[1:])) if int(iotaf.shape[0]) > 1 else jnp.asarray(0.0),
        "volavgB": volavgB,
        "betatotal": beta_total,
        "wb": norms.wb,
        "wp": norms.wp,
        "vp": getattr(norms, "vp", jnp.zeros_like(jnp.asarray(static.s))),
        "volume": norms.volume,
    }


def finite_beta_global_residuals_from_state(
    *,
    state,
    static,
    indata,
    signgs: int,
    targets: FiniteBetaTargets,
) -> jnp.ndarray:
    """Build global finite-beta residuals for stage-one surface optimization."""
    scalars = finite_beta_scalars_from_state(state=state, static=static, indata=indata, signgs=int(signgs))
    aspect_res = jnp.maximum(scalars["aspect"] - float(targets.aspect_ratio), 0.0)
    min_iota_res = jnp.minimum(scalars["min_iota"] - float(targets.min_iota), 0.0)
    mean_iota_res = jnp.minimum(scalars["mean_iota"] - float(targets.min_average_iota), 0.0)
    max_iota_res = jnp.maximum(scalars["max_iota"] - float(targets.max_iota), 0.0)
    return jnp.asarray(
        [
            float(targets.aspect_weight) * aspect_res,
            float(targets.iota_weight) * min_iota_res,
            float(targets.iota_weight) * mean_iota_res,
            float(targets.max_iota_weight) * max_iota_res,
            float(targets.volavgB_weight) * (scalars["volavgB"] - float(targets.volavgB)),
            float(targets.beta_weight) * (scalars["betatotal"] - float(targets.beta_total)),
        ],
        dtype=jnp.float64,
    )
