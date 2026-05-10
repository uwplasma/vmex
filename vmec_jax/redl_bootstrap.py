"""Differentiable Redl bootstrap-current algebra.

This module contains the pure profile, trapped-particle, and Redl fit helpers
used by finite-beta objectives.  VMEC-state geometry assembly remains in
``finite_beta`` so these functions stay small, testable, and independent of the
large Mercier/JXBFORCE state machinery.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._compat import jnp

ELEMENTARY_CHARGE = 1.602176634e-19

_GL_REDL_X, _GL_REDL_W = np.polynomial.legendre.leggauss(32)
_GL_REDL_X = jnp.asarray(_GL_REDL_X, dtype=jnp.float64)
_GL_REDL_W = jnp.asarray(_GL_REDL_W, dtype=jnp.float64)


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
    # singularity at lambda=1/Bmax. This improves nearly constant-B surfaces.
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


def redl_bootstrap_mismatch_from_profiles(*, jdotB_vmec, jdotB_redl, eps: float = 1.0e-300) -> Any:
    r"""Return normalized Redl bootstrap-current mismatch residuals."""

    jdotB_vmec = jnp.asarray(jdotB_vmec, dtype=jnp.float64)
    jdotB_redl = jnp.asarray(jdotB_redl, dtype=jnp.float64)
    denominator = jnp.sqrt(jnp.maximum(jnp.sum((jdotB_vmec + jdotB_redl) ** 2), float(eps)))
    return (jdotB_vmec - jdotB_redl) / denominator


__all__ = [
    "ELEMENTARY_CHARGE",
    "polynomial_profile_and_derivative",
    "redl_bootstrap_jdotb",
    "redl_bootstrap_mismatch_from_profiles",
    "trapped_fraction_from_modb_sqrtg",
]
