"""Differentiable Redl (2021) bootstrap current (plan.md R26.g, steps 1-2).

Implements the pure-formula layer and the two geometry lanes of
``notes_r26g_redl_spec.md``:

- :func:`compute_trapped_fraction` — effective trapped fraction, epsilon and
  the flux-surface averages ``<B^2>``, ``<1/B>`` from ``|B|``/``sqrt(g)`` on
  an angular grid.  Differentiable rewrite of simsopt
  ``simsopt.mhd.bootstrap.compute_trapped_fraction`` (fixed-order
  Gauss-Legendre lambda quadrature instead of adaptive ``scipy.integrate.quad``;
  plain grid max/min instead of spline-refined extrema; double-where guard on
  the ``sqrt(1 - lambda B)`` near-singularity so reverse-mode AD stays finite).
- :func:`j_dot_B_redl` — the Redl et al., Phys. Plasmas 28, 022502 (2021)
  ``<J.B>`` formula (eqs. 10-16, 19-21 with the Sauter 18b-18e
  collisionalities), transcribed verbatim from simsopt
  ``j_dot_B_Redl`` including the quasisymmetry isomorphism as simsopt applies
  it: ``iota -> iota - nfp*helicity_n`` everywhere iota appears, ``G``
  unshifted (spec section 3 note).
- :class:`KineticProfiles` / :func:`profile_value_and_dds` — polynomial
  ``ne/Te/Ti/Zeff`` profiles in ``s`` (lowest order first, simsopt
  ``ProfilePolynomial`` convention), value + analytic d/ds via Horner.
- :func:`redl_geometry_from_wout` — parity lane mirroring simsopt
  ``RedlGeomVmec.__call__``: half-mesh linear interpolation of
  ``iotas/bvco/buco/gmnc/bmnc`` onto the requested surfaces, cosine synthesis
  of ``|B|``/``sqrt(g)`` on a uniform ``(theta, phi)`` grid, then
  :func:`compute_trapped_fraction`.
- :func:`redl_geometry_from_state` — traceable lane on
  ``(SpectralState, SolverRuntime)``: ``|B|`` and ``sqrt(g)`` from the
  solver's half-mesh internal grid (mirrored from the reduced ``[0, pi]``
  theta grid exactly as ``QuasisymmetryRatioResidual._pointwise_state``),
  iota/G/I from ``_iotas_half``/``surface_currents``.

Units follow simsopt/the spec: ``ne`` [1/m^3], ``Te/Ti`` [eV], ``G/I/R``
[T*m], ``psi_edge`` [Wb/rad], output ``<J.B>`` [A*T/m^2].

Note (spec section 6.1b): the spec's stated decision was to hoist
``_field_chain``/``_iotas_half``/``_half_grid``/``_interp_half_grid`` into a
shared ``core/_state_diag.py``; because ``core/optimize.py`` is frozen while
this module lands, we take the spec's stated alternative and import the
private helpers from :mod:`vmec_jax.core.optimize` directly (no import cycle:
``optimize`` does not import this module).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import jax
import jax.numpy as jnp

from .fields import surface_currents
from .optimize import (
    _as_1d,
    _field_chain,
    _half_grid,
    _interp_half_grid,
    _iotas_half,
    _mode_matrix,
)
from .solver import SolverRuntime, SpectralState
from .wout import read_wout

__all__ = [
    "ELEMENTARY_CHARGE",
    "KineticProfiles",
    "profile_value_and_dds",
    "compute_trapped_fraction",
    "RedlGeometry",
    "redl_geometry_from_wout",
    "redl_geometry_from_state",
    "j_dot_B_redl",
]

Array = Any

#: CODATA elementary charge [C]; converts eV -> J in the <J.B> assembly.
ELEMENTARY_CHARGE = 1.602176634e-19


# ===========================================================================
# Kinetic profiles (polynomials in s, simsopt ProfilePolynomial convention)
# ===========================================================================


def profile_value_and_dds(coeffs: Array, s: Array) -> tuple[Array, Array]:
    """Polynomial profile value and analytic d/ds at ``s`` (traceable).

    ``coeffs`` are polynomial coefficients in ``s``, lowest order first
    (simsopt ``ProfilePolynomial``): ``p(s) = sum_k coeffs[k] * s**k``.
    Evaluated with the paired Horner recurrence, so the derivative is exact
    and both outputs are differentiable in ``coeffs`` and ``s``.
    """
    coeffs = jnp.atleast_1d(jnp.asarray(coeffs))
    s = jnp.asarray(s)
    value = jnp.zeros_like(s)
    dds = jnp.zeros_like(s)
    for k in range(int(coeffs.shape[0]) - 1, -1, -1):
        dds = dds * s + value
        value = value * s + coeffs[k]
    return value, dds


@dataclass(frozen=True)
class KineticProfiles:
    """Prescribed kinetic profiles for the Redl formula (frozen pytree).

    Polynomial coefficients in ``s`` (lowest order first).  These are *not*
    VMEC inputs; they parameterize the bootstrap objective (spec section 6.6).
    Units: ``ne_coeffs`` [1/m^3], ``Te_coeffs``/``Ti_coeffs`` [eV],
    ``Zeff_coeffs`` dimensionless (default: constant 1, hydrogen).

    Example (paper profiles): ``ne = n0*(1 - s^5)``, ``Te = Ti = T0*(1 - s)``::

        KineticProfiles(ne_coeffs=4.13e20 * np.array([1, 0, 0, 0, 0, -1]),
                        Te_coeffs=12.0e3 * np.array([1, -1]),
                        Ti_coeffs=12.0e3 * np.array([1, -1]))
    """

    ne_coeffs: Array
    Te_coeffs: Array
    Ti_coeffs: Array
    Zeff_coeffs: Array = 1.0


jax.tree_util.register_dataclass(
    KineticProfiles,
    data_fields=["ne_coeffs", "Te_coeffs", "Ti_coeffs", "Zeff_coeffs"],
    meta_fields=[],
)


# ===========================================================================
# Trapped fraction (differentiable rewrite of simsopt compute_trapped_fraction)
# ===========================================================================


def compute_trapped_fraction(modB: Array, sqrtg: Array, *, n_lambda: int = 64):
    r"""Effective trapped fraction and flux-surface averages per surface.

    ``f_t = 1 - (3/4) <B^2> \int_0^{1/Bmax} lambda dlambda / <sqrt(1 - lambda B)>``
    with ``<.>`` the flux-surface average (weight ``|sqrt(g)|``; only ratios
    of averages enter, so the uniform Jacobian sign cancels).

    Args:
        modB: ``(nsurf, ntheta, nzeta)`` array of :math:`|B|` on a uniform
            angular grid (leading surface axis — note simsopt uses trailing).
        sqrtg: same shape, the Jacobian :math:`\sqrt{g}` on the grid.
        n_lambda: fixed Gauss-Legendre quadrature order for the lambda
            integral on ``[0, 1/Bmax]`` (replaces simsopt's adaptive quad;
            the nodes exclude the ``lambda = 1/Bmax`` endpoint).

    Returns:
        ``(Bmin, Bmax, epsilon, fsa_B2, fsa_1overB, f_t)`` — 1D arrays of
        length ``nsurf`` (same tuple as simsopt).  ``Bmin/Bmax`` are hard grid
        extrema: piecewise-smooth gradients, adequate for trust-region least
        squares (same stance as :func:`vmec_jax.core.optimize.mirror_ratio`).
    """
    modB = jnp.asarray(modB)
    sqrtg = jnp.asarray(sqrtg)
    if modB.ndim != 3 or modB.shape != sqrtg.shape:
        raise ValueError("modB and sqrtg must both have shape (nsurf, ntheta, nzeta)")

    w = jnp.abs(sqrtg)
    Vp = jnp.mean(w, axis=(1, 2))
    fsa_B2 = jnp.mean(modB * modB * w, axis=(1, 2)) / Vp
    fsa_1overB = jnp.mean(w / modB, axis=(1, 2)) / Vp
    Bmax = jnp.max(modB, axis=(1, 2))
    Bmin = jnp.min(modB, axis=(1, 2))
    epsilon = (Bmax - Bmin) / (Bmax + Bmin)

    # Gauss-Legendre nodes/weights on [0, 1] (host constants, order is static).
    nodes, weights = np.polynomial.legendre.leggauss(int(n_lambda))
    x = jnp.asarray(0.5 * (nodes + 1.0), dtype=modB.dtype)      # (nl,) in (0, 1)
    wq = jnp.asarray(0.5 * weights, dtype=modB.dtype)

    lam = x[None, :] / Bmax[:, None]                            # (nsurf, nl)
    arg = 1.0 - lam[:, :, None, None] * modB[:, None, :, :]
    # Double-where guard: at the |B| = Bmax grid point arg -> 0 as
    # lambda -> 1/Bmax; d/dx sqrt(x) is infinite there, so reverse-mode AD
    # must never see sqrt evaluated at (or differentiated through) arg <= 0.
    positive = arg > 0.0
    safe = jnp.where(positive, arg, 1.0)
    root = jnp.where(positive, jnp.sqrt(safe), 0.0)
    fsa_root = (jnp.mean(root * w[:, None, :, :], axis=(2, 3))
                / Vp[:, None])                                  # <sqrt(1 - lam B)>
    integral = jnp.sum(wq[None, :] * lam / fsa_root, axis=1) / Bmax
    f_t = 1.0 - 0.75 * fsa_B2 * integral
    return Bmin, Bmax, epsilon, fsa_B2, fsa_1overB, f_t


# ===========================================================================
# Geometry container + the two lanes
# ===========================================================================


@dataclass(frozen=True)
class RedlGeometry:
    """Per-surface geometry inputs of the Redl formula (frozen pytree).

    All arrays are 1D over ``surfaces`` (values of normalized toroidal flux).
    ``G``/``I`` are the Boozer covariant averages (wout ``bvco``/``buco``,
    [T*m]); ``R = (G + iota*I) * <1/B>`` is the effective major radius for the
    Sauter collisionality; ``psi_edge = -phi(s=1)/(2 pi)`` [Wb/rad] (wout sign
    convention).  ``nfp`` is static metadata.
    """

    surfaces: Array
    iota: Array
    G: Array
    I: Array  # noqa: E741 - Boozer I
    R: Array
    epsilon: Array
    f_t: Array
    fsa_B2: Array
    fsa_1overB: Array
    Bmin: Array
    Bmax: Array
    psi_edge: Array
    nfp: int = 1


jax.tree_util.register_dataclass(
    RedlGeometry,
    data_fields=["surfaces", "iota", "G", "I", "R", "epsilon", "f_t",
                 "fsa_B2", "fsa_1overB", "Bmin", "Bmax", "psi_edge"],
    meta_fields=["nfp"],
)


def redl_geometry_from_wout(
    wout,
    surfaces,
    *,
    ntheta: int = 64,
    nphi: int = 65,
    n_lambda: int = 64,
) -> RedlGeometry:
    """Redl geometry inputs from a wout dataset (simsopt ``RedlGeomVmec`` lane).

    ``wout`` may be a :class:`~vmec_jax.core.wout.WoutData`, a path to a
    ``wout_*.nc`` file, or anything exposing a ``.wout`` attribute (e.g.
    :class:`~vmec_jax.core.optimize.Equilibrium`).  Mirrors simsopt
    ``RedlGeomVmec.__call__``: linear interpolation of the half-mesh tables
    ``iotas/bvco/buco/gmnc/bmnc`` (axis slot dropped) onto ``surfaces``,
    cosine synthesis of ``|B|``/``sqrt(g)`` from the Nyquist mode tables on
    ``theta in [0, 2 pi)`` x ``phi in [0, 2 pi/nfp)`` grids, then
    :func:`compute_trapped_fraction`.  jnp throughout; this is the
    validation/finite-difference lane, not the implicit path.
    """
    if isinstance(wout, (str, Path)):
        wout = read_wout(wout)
    wout = getattr(wout, "wout", wout)
    if bool(getattr(wout, "lasym", False)):
        raise NotImplementedError("redl_geometry_from_wout supports lasym = False only")

    surfaces = _as_1d(surfaces)
    nfp = int(wout.nfp)
    iotas = _as_1d(np.asarray(wout.iotas, dtype=float))
    ns = int(iotas.shape[0])
    s_half = _half_grid(ns, iotas.dtype)

    def half(values):
        return _interp_half_grid(values[1:], surfaces, s_half)

    iota = half(iotas)
    G = half(_as_1d(np.asarray(wout.bvco, dtype=float)))
    I = half(_as_1d(np.asarray(wout.buco, dtype=float)))  # noqa: E741 - Boozer I

    xm = _as_1d(np.asarray(wout.xm_nyq, dtype=float))
    xn = _as_1d(np.asarray(wout.xn_nyq, dtype=float))
    mn = int(xm.shape[0])
    bmnc = half(_mode_matrix(wout, "bmnc", ns=ns, mn=mn))
    gmnc = half(_mode_matrix(wout, "gmnc", ns=ns, mn=mn))

    theta1d = jnp.linspace(0.0, 2.0 * jnp.pi, int(ntheta), endpoint=False)
    phi1d = jnp.linspace(0.0, 2.0 * jnp.pi / nfp, int(nphi), endpoint=False)
    angle = (theta1d[:, None, None] * xm[None, None, :]
             - phi1d[None, :, None] * xn[None, None, :])
    cosangle = jnp.cos(angle)
    modB = jnp.einsum("sm,tpm->stp", bmnc, cosangle)
    sqrtg = jnp.einsum("sm,tpm->stp", gmnc, cosangle)

    Bmin, Bmax, epsilon, fsa_B2, fsa_1overB, f_t = compute_trapped_fraction(
        modB, sqrtg, n_lambda=n_lambda)
    R = (G + iota * I) * fsa_1overB
    psi_edge = -_as_1d(np.asarray(wout.phi, dtype=float))[-1] / (2.0 * jnp.pi)
    return RedlGeometry(
        surfaces=surfaces, iota=iota, G=G, I=I, R=R, epsilon=epsilon, f_t=f_t,
        fsa_B2=fsa_B2, fsa_1overB=fsa_1overB, Bmin=Bmin, Bmax=Bmax,
        psi_edge=psi_edge, nfp=nfp)


def redl_geometry_from_state(
    state: SpectralState,
    rt: SolverRuntime,
    *,
    surfaces=None,
    n_lambda: int = 64,
) -> RedlGeometry:
    """Redl geometry inputs as a traceable function of ``(state, runtime)``.

    ``|B|`` and ``sqrt(g)`` are taken on the solver's half-mesh internal grid
    (``|B|^2 = 2 (total_pressure - pressure)``, bcovar.f) with the reduced
    ``[0, pi]`` theta grid mirrored to the full circle exactly as
    ``QuasisymmetryRatioResidual._pointwise_state`` does; the axis row
    ``js = 0`` is dropped *before* any singular division (0*inf AD note,
    optimize.py).  iota comes from ``_iotas_half`` (ncurr=1-aware), G/I from
    :func:`~vmec_jax.core.fields.surface_currents` (``bvco``/``buco``), and
    ``psi_edge = -signgs*hs*sum(phipf[1:])``.  Per-surface scalars and the
    angular fields are linearly interpolated from the half mesh onto
    ``surfaces`` before :func:`compute_trapped_fraction`.

    Default ``surfaces``: ``linspace(0.05, 0.95, 16)`` — interior only; do
    not sample ``s -> 1`` where ``Te, Ti -> 0`` blows up the collisionality
    (spec section 6.1b).
    """
    setup = rt.setup
    if bool(setup.lasym):
        raise NotImplementedError(
            "redl_geometry_from_state supports lasym = False only")
    if surfaces is None:
        surfaces = np.linspace(0.05, 0.95, 16)
    surfaces = _as_1d(surfaces)

    s = jnp.asarray(setup.s_full)
    nfp = int(rt.resolution.nfp)
    _, jacobian, _, fields, _ = _field_chain(state, rt)

    # Mirror the reduced [0, pi] theta grid to the full circle
    # (stellarator-symmetry map X(2 pi - theta, -zeta) = X(theta, zeta)).
    ntheta2 = int(np.shape(fields.total_pressure)[1])
    nzeta = int(np.shape(fields.total_pressure)[2])
    ntheta1 = max(2 * (ntheta2 - 1), 1)
    i_full = np.arange(ntheta1)
    i_src = np.where(i_full < ntheta2, i_full, ntheta1 - i_full)
    k = np.arange(nzeta)
    k_src = np.where(i_full[:, None] < ntheta2, k[None, :],
                     (nzeta - k[None, :]) % nzeta)
    i_src = np.broadcast_to(i_src[:, None], (ntheta1, nzeta))

    def full(a):
        # Drop the zeroed axis row (js = 0) before the divisions downstream.
        return jnp.asarray(a)[1:, i_src, k_src]

    bsq2 = 2.0 * (jnp.asarray(fields.total_pressure)
                  - jnp.asarray(fields.pressure)[:, None, None])
    tiny = jnp.asarray(jnp.finfo(bsq2.dtype).tiny, dtype=bsq2.dtype)
    bmag = jnp.sqrt(jnp.maximum(full(bsq2), tiny))     # half mesh js = 1..ns-1
    w = jnp.abs(full(jacobian.sqrt_g))

    iota_h = _iotas_half(state, rt)[1:]
    cur = surface_currents(bsubu=fields.bsubu, bsubv=fields.bsubv,
                           trig=rt.trig, s=s, signgs=setup.signgs)
    G_h = jnp.asarray(cur.bvco)[1:]
    I_h = jnp.asarray(cur.buco)[1:]

    s_half = 0.5 * (s[:-1] + s[1:])
    iota = _interp_half_grid(iota_h, surfaces, s_half)
    G = _interp_half_grid(G_h, surfaces, s_half)
    I = _interp_half_grid(I_h, surfaces, s_half)  # noqa: E741 - Boozer I
    modB = _interp_half_grid(bmag, surfaces, s_half)
    sqrtg = _interp_half_grid(w, surfaces, s_half)

    Bmin, Bmax, epsilon, fsa_B2, fsa_1overB, f_t = compute_trapped_fraction(
        modB, sqrtg, n_lambda=n_lambda)
    R = (G + iota * I) * fsa_1overB

    hs = s[1] - s[0]
    psi_edge = -float(setup.signgs) * hs * jnp.sum(jnp.asarray(setup.phipf)[1:])
    return RedlGeometry(
        surfaces=surfaces, iota=iota, G=G, I=I, R=R, epsilon=epsilon, f_t=f_t,
        fsa_B2=fsa_B2, fsa_1overB=fsa_1overB, Bmin=Bmin, Bmax=Bmax,
        psi_edge=psi_edge, nfp=nfp)


# ===========================================================================
# Redl (2021) <J.B> formula (transcribed verbatim from simsopt j_dot_B_Redl)
# ===========================================================================


def j_dot_B_redl(
    profiles: KineticProfiles,
    geom: RedlGeometry,
    helicity_n: int,
) -> tuple[Array, dict[str, Array]]:
    """Bootstrap ``<J.B>`` [A*T/m^2] from the Redl (2021) formulae (pure jnp).

    Equation-by-equation transcription of simsopt ``j_dot_B_Redl`` (which
    generated the arXiv:2205.02914 results): Sauter eqs. (18b)-(18e) for the
    Coulomb logarithms and collisionalities, Redl eqs. (10)-(16) and
    (19)-(21) for ``L31/L32/L34/alpha``.  The quasisymmetry isomorphism is
    applied exactly as simsopt does: ``iota -> iota - N`` with
    ``N = nfp*helicity_n`` in the collisionality geometry factor and in the
    ``1/(psi_edge*(iota - N))`` prefactor; ``G`` (and the ``R`` provided by
    the geometry lanes) is used unshifted.

    ``helicity_n`` is 0 for quasi-axisymmetry, +/-1 for quasi-helical
    symmetry.  ``Te/Ti`` are clamped at 1 eV and ``ne`` at 1e17 1/m^3
    (belt-and-suspenders against sampling s -> 1 where the profiles vanish
    and the collisionality blows up; spec section 6.1b).

    Returns ``(jdotB, details)`` with ``details`` a dict of every
    intermediate quantity (simsopt names).
    """
    s = jnp.asarray(geom.surfaces)
    ne_s, d_ne_d_s = profile_value_and_dds(profiles.ne_coeffs, s)
    Te_s, d_Te_d_s = profile_value_and_dds(profiles.Te_coeffs, s)
    Ti_s, d_Ti_d_s = profile_value_and_dds(profiles.Ti_coeffs, s)
    Zeff_s, _ = profile_value_and_dds(profiles.Zeff_coeffs, s)

    ne_s = jnp.maximum(ne_s, 1e17)
    Te_s = jnp.maximum(Te_s, 1.0)
    Ti_s = jnp.maximum(Ti_s, 1.0)
    ni_s = ne_s / Zeff_s
    pe_s = ne_s * Te_s
    pi_s = ni_s * Ti_s

    # Sauter eqs. (18d)-(18e):
    ln_Lambda_e = 31.3 - jnp.log(jnp.sqrt(ne_s) / Te_s)
    ln_Lambda_ii = 30.0 - jnp.log(Zeff_s ** 3 * jnp.sqrt(ni_s) / (Ti_s ** 1.5))

    # Sauter eqs. (18b)-(18c) with the isomorphism substitution iota -> iota - N:
    helicity_N = int(geom.nfp) * int(helicity_n)
    iota_N = geom.iota - helicity_N
    geometry_factor = jnp.abs(geom.R / iota_N)
    nu_e = (geometry_factor * (6.921e-18) * ne_s * Zeff_s * ln_Lambda_e
            / (Te_s * Te_s * (geom.epsilon ** 1.5)))
    nu_i = (geometry_factor * (4.90e-18) * ni_s * (Zeff_s ** 4) * ln_Lambda_ii
            / (Ti_s * Ti_s * (geom.epsilon ** 1.5)))

    f_t = geom.f_t
    sqrt_nu_e = jnp.sqrt(nu_e)
    sqrt_nu_i = jnp.sqrt(nu_i)
    # AD-safe sqrt(Zeff - 1): Zeff = 1 exactly is the common case and
    # d/dx sqrt(x) is infinite at x = 0 (double-where guard).
    zm1 = Zeff_s - 1.0
    zm1_pos = zm1 > 0.0
    sqrt_zm1 = jnp.where(zm1_pos, jnp.sqrt(jnp.where(zm1_pos, zm1, 1.0)), 0.0)

    # Redl eq (11):
    X31 = f_t / (1 + (0.67 * (1 - 0.7 * f_t) * sqrt_nu_e) / (0.56 + 0.44 * Zeff_s)
                 + (0.52 + 0.086 * sqrt_nu_e) * (1 + 0.87 * f_t) * nu_e / (1 + 1.13 * sqrt_zm1))

    # Redl eq (10):
    Zfac = Zeff_s ** 1.2 - 0.71
    L31 = (1 + 0.15 / Zfac) * X31 \
        - 0.22 / Zfac * (X31 ** 2) \
        + 0.01 / Zfac * (X31 ** 3) \
        + 0.06 / Zfac * (X31 ** 4)

    # Redl eq (14):
    X32e = f_t / ((1 + 0.23 * (1 - 0.96 * f_t) * sqrt_nu_e / jnp.sqrt(Zeff_s)
                   + 0.13 * (1 - 0.38 * f_t) * nu_e / (Zeff_s * Zeff_s)
                   * (jnp.sqrt(1 + 2 * sqrt_zm1)
                      + f_t * f_t * jnp.sqrt((0.075 + 0.25 * zm1 ** 2) * nu_e))))

    # Redl eq (13):
    F32ee = (0.1 + 0.6 * Zeff_s) * (X32e - X32e ** 4) \
        / (Zeff_s * (0.77 + 0.63 * (1 + zm1 ** 1.1))) \
        + 0.7 / (1 + 0.2 * Zeff_s) * (X32e ** 2 - X32e ** 4 - 1.2 * (X32e ** 3 - X32e ** 4)) \
        + 1.3 / (1 + 0.5 * Zeff_s) * (X32e ** 4)

    # Redl eq (16):
    X32ei = f_t / (1 + 0.87 * (1 + 0.39 * f_t) * sqrt_nu_e / (1 + 2.95 * zm1 ** 2)
                   + 1.53 * (1 - 0.37 * f_t) * nu_e * (2 + 0.375 * zm1))

    # Redl eq (15):
    F32ei = -(0.4 + 1.93 * Zeff_s) / (Zeff_s * (0.8 + 0.6 * Zeff_s)) * (X32ei - X32ei ** 4) \
        + 5.5 / (1.5 + 2 * Zeff_s) * (X32ei ** 2 - X32ei ** 4 - 0.8 * (X32ei ** 3 - X32ei ** 4)) \
        - 1.3 / (1 + 0.5 * Zeff_s) * (X32ei ** 4)

    # Redl eq (12):
    L32 = F32ei + F32ee

    # Redl eq (19):
    L34 = L31

    # Redl eq (20):
    alpha0 = -(0.62 + 0.055 * zm1) * (1 - f_t) \
        / ((0.53 + 0.17 * zm1) * (1 - (0.31 - 0.065 * zm1) * f_t - 0.25 * f_t * f_t))
    # Redl eq (21):
    alpha = ((alpha0 + 0.7 * Zeff_s * jnp.sqrt(f_t * nu_i)) / (1 + 0.18 * sqrt_nu_i)
             - 0.002 * nu_i * nu_i * (f_t ** 6)) \
        / (1 + 0.004 * nu_i * nu_i * (f_t ** 6))

    # Factor of ELEMENTARY_CHARGE converts temperatures from eV to J:
    pref = -geom.G * ELEMENTARY_CHARGE / (geom.psi_edge * iota_N)
    dnds_term = pref * (ne_s * Te_s + ni_s * Ti_s) * L31 * (d_ne_d_s / ne_s)
    dTeds_term = pref * pe_s * (L31 + L32) * (d_Te_d_s / Te_s)
    dTids_term = pref * pi_s * (L31 + L34 * alpha) * (d_Ti_d_s / Ti_s)
    jdotB = dnds_term + dTeds_term + dTids_term

    details = {
        "s": s, "ne_s": ne_s, "ni_s": ni_s, "Zeff_s": Zeff_s,
        "Te_s": Te_s, "Ti_s": Ti_s,
        "d_ne_d_s": d_ne_d_s, "d_Te_d_s": d_Te_d_s, "d_Ti_d_s": d_Ti_d_s,
        "ln_Lambda_e": ln_Lambda_e, "ln_Lambda_ii": ln_Lambda_ii,
        "nu_e_star": nu_e, "nu_i_star": nu_i,
        "X31": X31, "X32e": X32e, "X32ei": X32ei,
        "F32ee": F32ee, "F32ei": F32ei,
        "L31": L31, "L32": L32, "L34": L34,
        "alpha0": alpha0, "alpha": alpha,
        "dnds_term": dnds_term, "dTeds_term": dTeds_term,
        "dTids_term": dTids_term, "jdotB": jdotB,
    }
    return jdotB, details
