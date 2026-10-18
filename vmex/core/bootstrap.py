"""Differentiable Redl (2021) bootstrap current.

Redl, Angioni, Belli & Sauter, Phys. Plasmas 28, 022502 (2021), eqs. 10-16
and 19-21, on the Sauter, Angioni & Lin-Liu, Phys. Plasmas 6, 2834 (1999)
collisionalities (eqs. 18b-18e); the self-consistent lane follows Landreman,
Buller & Drevlak, Phys. Plasmas 29, 082501 (2022).

- :func:`compute_trapped_fraction` — effective trapped fraction, epsilon and
  the flux-surface averages ``<B^2>``, ``<1/B>`` from ``|B|``/``sqrt(g)``.
  Differentiable rewrite of simsopt
  ``simsopt.mhd.bootstrap.compute_trapped_fraction`` (fixed-order
  Gauss-Legendre lambda quadrature instead of adaptive ``scipy.integrate.quad``;
  plain grid max/min instead of spline-refined extrema; double-where guard on
  the ``sqrt(1 - lambda B)`` near-singularity so reverse-mode AD stays finite).
- :func:`j_dot_B_redl` — the Redl et al., Phys. Plasmas 28, 022502 (2021)
  ``<J.B>`` formula (eqs. 10-16, 19-21 with the Sauter 18b-18e
  collisionalities), transcribed verbatim from simsopt ``j_dot_B_Redl``
  including its quasisymmetry isomorphism: ``iota -> iota - nfp*helicity_n``
  everywhere iota appears, ``G`` unshifted.
- :class:`KineticProfiles` / :func:`profile_value_and_dds` — polynomial
  ``ne/Te/Ti/Zeff`` profiles in ``s`` (lowest order first, simsopt
  ``ProfilePolynomial`` convention), value + analytic d/ds via Horner.
- Redl's geometry, as the bootstrap rows and :class:`RedlBootstrapMismatch`
  use it: the quasi-symmetric part of ``|B|`` (Boozer angles, simsopt
  ``RedlGeomBoozer``), from a state (:func:`_qs_profile_geometry`) or a wout
  (:func:`redl_qs_geometry_from_wout`); smooth in the equilibrium.
- :func:`redl_geometry_from_wout` — the full field, simsopt
  ``RedlGeomVmec.__call__`` parity check only (tests and validation): half-mesh
  interpolation of ``iotas/bvco/buco/gmnc/bmnc``, cosine synthesis of
  ``|B|``/``sqrt(g)``.
- :func:`redl_geometry_from_state` — the full field's geometry on
  ``(SpectralState, SolverRuntime)`` (the solver's half-mesh internal grid;
  iota/G/I from ``_iotas_half``/``surface_currents``), for geometry outputs;
  no Redl current path uses it.
- :func:`vmec_j_dot_B` / :func:`vmec_j_dot_B_from_wout` — equilibrium
  ``<J.B>`` via the MHD identity (validated against the Zenodo
  ``convertSfincsToVmecCurrentProfile`` script and the wout ``jdotb``):

      <J.B>(s) = [<B^2> dI/ds + mu0 I dp/ds] / (2 pi psi_a),
      I(s) = signgs*(2 pi/mu0)*buco(s) [A],  psi_a = phi(1)/(2 pi) [Wb/rad]

  (``signgs`` matches VMEC's ``ctor = signgs*(2 pi/mu0)*buco(ns)``
  convention, so the identity reproduces the sign of the wout ``jdotb``).
- :class:`RedlBootstrapMismatch` — the paper/simsopt ``f_boot`` normalized
  residual comparing ``<J.B>_vmec`` against :func:`j_dot_B_redl`; same wout /
  traceable-state dual-lane shape as ``QuasisymmetryRatioResidual`` (composes
  with ``least_squares`` at ``jac=None`` and ``jac="implicit"``).
- :func:`self_consistent_bootstrap` — fixed-boundary Picard iteration
  ``AC/CURTOR <- current profile implied by the Redl <J.B>`` (the Zenodo
  script's "smooth method").

Units follow simsopt/the spec: ``ne`` [1/m^3], ``Te/Ti`` [eV], ``G/I/R``
[T*m], ``psi_edge`` [Wb/rad], output ``<J.B>`` [A*T/m^2].  The shared
half-mesh helpers (``_field_chain``/``_iotas_half``/``_half_grid``/
``_interp_half_grid``) live in :mod:`vmex.core.statephysics`; ``optimize``
re-exports them for backward compatibility.
"""

from __future__ import annotations

import dataclasses
import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

import jax
import jax.numpy as jnp

from .fields import surface_currents
from .loops import map_columns
from .input import VmecInput
from .profiles import MU0
from .solver import SolverRuntime, SpectralState
from .statephysics import (
    _as_1d,
    _field_chain,
    _half_grid,
    _interp_half_grid,
    _iotas_half,
    _mode_matrix,
)
from .wout import read_wout

__all__ = [
    "ELEMENTARY_CHARGE",
    "KineticProfiles",
    "profile_value_and_dds",
    "compute_trapped_fraction",
    "trapped_fraction_from_state",
    "RedlGeometry",
    "redl_geometry_from_wout",
    "redl_geometry_from_state",
    "redl_qs_geometry_from_wout",
    "j_dot_B_redl",
    "vmec_j_dot_B",
    "vmec_j_dot_B_from_wout",
    "RedlBootstrapMismatch",
    "BootstrapPicardResult",
    "self_consistent_bootstrap",
]

Array = Any

#: CODATA elementary charge [C]; converts eV -> J in the <J.B> assembly.
ELEMENTARY_CHARGE = 1.602176634e-19

#: Gauss-Legendre order for the trapped-fraction lambda quadrature, shared by
#: every entry point so the geometry lanes and the objective agree by default.
#: At 64 nodes the Redl current derivative (:func:`redl_current_derivative`)
#: is converged to round-off.
N_LAMBDA = 64


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
    VMEC inputs; they parameterize the bootstrap objective.
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


def compute_trapped_fraction(modB: Array, sqrtg: Array, *, n_lambda: int = N_LAMBDA):
    r"""Effective trapped fraction and flux-surface averages per surface.

    ``f_t = 1 - (3/4) <B^2> \int_0^{1/Bmax} lambda dlambda / <sqrt(1 - lambda B)>``
    with ``<.>`` the flux-surface average (weight ``|sqrt(g)|``; only ratios
    of averages enter, so the uniform Jacobian sign cancels).

    Args:
        modB: ``(nsurf, ntheta, nzeta)`` array of :math:`|B|` on a uniform
            angular grid (leading surface axis — note simsopt uses trailing).
        sqrtg: same shape, the Jacobian :math:`\sqrt{g}` on the grid.
        n_lambda: fixed Gauss-Legendre order after the substitution
            :math:`y=\sqrt{1-\lambda B_{\max}}` (replaces simsopt's
            adaptive quadrature).

    Returns:
        ``(Bmin, Bmax, epsilon, fsa_B2, fsa_1overB, f_t)`` — 1D arrays of
        length ``nsurf`` (same tuple as simsopt).  ``Bmin/Bmax`` are hard grid
        extrema: piecewise-smooth gradients, adequate for trust-region least
        squares (same stance as :func:`vmex.core.optimize.mirror_ratio`).
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
    f_t = _trapped_fraction_integral(modB, w, Vp, fsa_B2, Bmax, n_lambda)
    return Bmin, Bmax, epsilon, fsa_B2, fsa_1overB, f_t


def _trapped_fraction_integral(modB, w, Vp, fsa_B2, Bmax, n_lambda):
    """``f_t = 1 - (3/4) <B^2> int_0^{1/Bmax} lambda dlambda / <sqrt(1 - lambda B)>`` on ``modB`` sampled with the
    surface-average weight ``w`` (``(nsurf, ...)`` arrays; ``Vp`` the weight's mean, ``Bmax`` per surface)."""
    axes = tuple(range(1, modB.ndim))
    # Gauss-Legendre nodes/weights on [0, 1] (host constants, order is static).
    nodes, weights = np.polynomial.legendre.leggauss(int(n_lambda))
    x = jnp.asarray(0.5 * (nodes + 1.0), dtype=modB.dtype)      # (nl,) in (0, 1)
    wq = jnp.asarray(0.5 * weights, dtype=modB.dtype)

    y = x[None, :]
    lam = (1.0 - y * y) / Bmax[:, None]
    expand = (slice(None), slice(None)) + (None,) * len(axes)
    arg = 1.0 - lam[expand] * modB[:, None]
    # Double-where guard: at the |B| = Bmax grid point arg -> 0 as
    # lambda -> 1/Bmax; d/dx sqrt(x) is infinite there, so reverse-mode AD
    # must never see sqrt evaluated at (or differentiated through) arg <= 0.
    positive = arg > 0.0
    safe = jnp.where(positive, arg, 1.0)
    root = jnp.where(positive, jnp.sqrt(safe), 0.0)
    fsa_root = (jnp.mean(root * w[:, None], axis=tuple(a + 1 for a in axes))
                / Vp[:, None])
    integral = jnp.sum(
        wq[None, :] * 2.0 * y * lam / fsa_root, axis=1) / Bmax
    return 1.0 - 0.75 * fsa_B2 * integral


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
    n_lambda: int = N_LAMBDA,
) -> RedlGeometry:
    """Redl geometry inputs from a wout dataset (simsopt ``RedlGeomVmec`` lane).

    ``wout`` may be a :class:`~vmex.core.wout.WoutData`, a path to a
    ``wout_*.nc`` file, or anything exposing a ``.wout`` attribute (e.g.
    :class:`~vmex.core.optimize.Equilibrium`).  Mirrors simsopt
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
    lasym = bool(getattr(wout, "lasym", False))

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
    bmns = half(_mode_matrix(wout, "bmns", ns=ns, mn=mn)) if lasym else None
    gmns = half(_mode_matrix(wout, "gmns", ns=ns, mn=mn)) if lasym else None
    axis_modes = (surfaces == 0.0)[:, None] & (xm != 0.0)[None, :]
    bmnc, gmnc = jnp.where(axis_modes, 0.0, bmnc), jnp.where(
        axis_modes, 0.0, gmnc)
    if lasym:
        bmns, gmns = jnp.where(axis_modes, 0.0, bmns), jnp.where(
            axis_modes, 0.0, gmns)

    theta1d = jnp.linspace(0.0, 2.0 * jnp.pi, int(ntheta), endpoint=False)
    phi1d = jnp.linspace(0.0, 2.0 * jnp.pi / nfp, int(nphi), endpoint=False)
    angle = (theta1d[:, None, None] * xm[None, None, :]
             - phi1d[None, :, None] * xn[None, None, :])
    cosangle = jnp.cos(angle)
    modB = jnp.einsum("sm,tpm->stp", bmnc, cosangle)
    sqrtg = jnp.einsum("sm,tpm->stp", gmnc, cosangle)
    if lasym:
        sinangle = jnp.sin(angle)
        modB += jnp.einsum("sm,tpm->stp", bmns, sinangle)
        sqrtg += jnp.einsum("sm,tpm->stp", gmns, sinangle)

    Bmin, Bmax, epsilon, fsa_B2, fsa_1overB, f_t = compute_trapped_fraction(
        modB, sqrtg, n_lambda=n_lambda)
    R = (G + iota * I) * fsa_1overB
    psi_edge = -_as_1d(np.asarray(wout.phi, dtype=float))[-1] / (2.0 * jnp.pi)
    return RedlGeometry(
        surfaces=surfaces, iota=iota, G=G, I=I, R=R, epsilon=epsilon, f_t=f_t,
        fsa_B2=fsa_B2, fsa_1overB=fsa_1overB, Bmin=Bmin, Bmax=Bmax,
        psi_edge=psi_edge, nfp=nfp)


class _HalfMeshFields(NamedTuple):
    """Traceable half-mesh (js = 1..ns-1) raw fields of one core state.

    ``bmag``/``w`` are ``(ns-1, ntheta1, nzeta)`` on the mirrored full theta
    circle; the per-surface arrays are 1D of length ``ns-1``.  ``p_int`` is
    the kinetic pressure in internal units (``mu0*Pa``); ``phi_edge`` is
    ``phi(s=1)/(2 pi)`` [Wb/rad] (the un-negated wout convention — the Redl
    ``psi_edge`` is its negative).
    """

    s_half: Array
    bmag: Array
    w: Array
    iota: Array
    G: Array
    I: Array  # noqa: E741 - Boozer I
    p_int: Array
    phi_edge: Array
    signgs: float
    nfp: int


def _full_surface_values(values: Array, *, lasym: bool) -> Array:
    """Drop the axis slot and restore the full poloidal surface."""
    values = jnp.asarray(values)[1:]
    if lasym:
        return values
    ntheta2, nzeta = int(values.shape[1]), int(values.shape[2])
    ntheta1 = max(2 * (ntheta2 - 1), 1)
    i_full = np.arange(ntheta1)
    i_src = np.where(i_full < ntheta2, i_full, ntheta1 - i_full)
    k = np.arange(nzeta)
    k_src = np.where(
        i_full[:, None] < ntheta2, k[None, :],
        (nzeta - k[None, :]) % nzeta,
    )
    return values[:, np.broadcast_to(i_src[:, None], k_src.shape), k_src]


def _trapped_fraction_from_fields(
    *,
    total_pressure,
    pressure,
    sqrt_g,
    s_full,
    surfaces,
    lasym: bool,
    n_lambda: int,
    serial_surfaces: bool = False,
):
    """Evaluate trapped-fraction quantities from an existing field pass."""
    b2 = 2.0 * (
        jnp.asarray(total_pressure) - jnp.asarray(pressure)[:, None, None])
    tiny = jnp.asarray(jnp.finfo(b2.dtype).tiny, dtype=b2.dtype)
    modB = _full_surface_values(jnp.sqrt(jnp.maximum(b2, tiny)), lasym=lasym)
    sqrtg = jnp.abs(_full_surface_values(sqrt_g, lasym=lasym))
    s_full = jnp.asarray(s_full)
    s_half = 0.5 * (s_full[:-1] + s_full[1:])

    def interpolate(values):
        result = _interp_half_grid(values, surfaces, s_half)
        axis_samples = jnp.mean(values, axis=1, keepdims=True)
        axis = _interp_half_grid(
            axis_samples, jnp.zeros((1,), dtype=surfaces.dtype), s_half)[0]
        mask = (surfaces == 0.0).reshape(
            (surfaces.shape[0],) + (1,) * (values.ndim - 1))
        result = jnp.where(mask, axis, result)
        return result

    modB = interpolate(modB)
    sqrtg = interpolate(sqrtg)
    if serial_surfaces:
        return map_columns(
            lambda x: tuple(v[0] for v in compute_trapped_fraction(
                x[0][None], x[1][None], n_lambda=n_lambda)),
            (modB, sqrtg),
        )
    return compute_trapped_fraction(modB, sqrtg, n_lambda=n_lambda)


@functools.partial(jax.jit, static_argnames=("lasym", "n_lambda"))
def _trapped_fraction_export_lane(
    *, total_pressure, pressure, sqrt_g, s_full, lasym: bool, n_lambda: int
):
    """WOUT-export trapped fraction f_t(s) on the full mesh, one XLA program.

    Module-level ``jax.jit``: eager, the interpolation + ``lax.map`` chain
    dispatches ~150 single-op XLA programs per export (~2 s of every cold
    CLI run that writes a WOUT file).  ``serial_surfaces=True`` keeps the
    per-surface pitch-angle grid out of memory at once, as before.
    """
    return _trapped_fraction_from_fields(
        total_pressure=total_pressure, pressure=pressure, sqrt_g=sqrt_g,
        s_full=s_full, surfaces=s_full, lasym=lasym, n_lambda=n_lambda,
        serial_surfaces=True,
    )[-1]


def _half_mesh_fields(state: SpectralState, rt: SolverRuntime) -> _HalfMeshFields:
    """One ``_field_chain`` pass -> the half-mesh inputs of both bootstrap lanes.

    ``|B|`` and ``sqrt(g)`` on the solver's half-mesh internal grid
    (``|B|^2 = 2 (total_pressure - pressure)``, bcovar.f) with the reduced
    ``[0, pi]`` theta grid mirrored to the full circle exactly as
    ``QuasisymmetryRatioResidual._pointwise_state`` does; the axis row
    ``js = 0`` is dropped *before* any singular division (0*inf AD note,
    optimize.py).  iota comes from ``_iotas_half`` (ncurr=1-aware), G/I from
    :func:`~vmex.core.fields.surface_currents` (``bvco``/``buco``).
    """
    setup = rt.setup
    s = jnp.asarray(setup.s_full)
    nfp = int(rt.resolution.nfp)
    _, jacobian, _, fields, _ = _field_chain(state, rt)

    bsq2 = 2.0 * (jnp.asarray(fields.total_pressure)
                  - jnp.asarray(fields.pressure)[:, None, None])
    tiny = jnp.asarray(jnp.finfo(bsq2.dtype).tiny, dtype=bsq2.dtype)
    bmag = _full_surface_values(
        jnp.sqrt(jnp.maximum(bsq2, tiny)), lasym=bool(setup.lasym))
    w = jnp.abs(_full_surface_values(
        jacobian.sqrt_g, lasym=bool(setup.lasym)))

    iota_h = _iotas_half(state, rt)[1:]
    cur = surface_currents(bsubu=fields.bsubu, bsubv=fields.bsubv,
                           trig=rt.trig, s=s, signgs=setup.signgs)
    G_h = jnp.asarray(cur.bvco)[1:]
    I_h = jnp.asarray(cur.buco)[1:]
    p_int = jnp.asarray(fields.pressure)[1:]           # mu0*Pa, half mesh

    hs = s[1] - s[0]
    phi_edge = float(setup.signgs) * hs * jnp.sum(jnp.asarray(setup.phipf)[1:])
    return _HalfMeshFields(
        s_half=0.5 * (s[:-1] + s[1:]), bmag=bmag, w=w, iota=iota_h, G=G_h,
        I=I_h, p_int=p_int, phi_edge=phi_edge, signgs=float(setup.signgs),
        nfp=nfp)


def _geometry_from_half(hm: _HalfMeshFields, surfaces, *, n_lambda: int) -> RedlGeometry:
    """Interpolate half-mesh fields onto ``surfaces`` -> :class:`RedlGeometry`."""
    iota = _interp_half_grid(hm.iota, surfaces, hm.s_half)
    G = _interp_half_grid(hm.G, surfaces, hm.s_half)
    I = _interp_half_grid(hm.I, surfaces, hm.s_half)  # noqa: E741 - Boozer I
    modB = _interp_half_grid(hm.bmag, surfaces, hm.s_half)
    sqrtg = _interp_half_grid(hm.w, surfaces, hm.s_half)

    Bmin, Bmax, epsilon, fsa_B2, fsa_1overB, f_t = compute_trapped_fraction(
        modB, sqrtg, n_lambda=n_lambda)
    R = (G + iota * I) * fsa_1overB
    return RedlGeometry(
        surfaces=surfaces, iota=iota, G=G, I=I, R=R, epsilon=epsilon, f_t=f_t,
        fsa_B2=fsa_B2, fsa_1overB=fsa_1overB, Bmin=Bmin, Bmax=Bmax,
        psi_edge=-hm.phi_edge, nfp=hm.nfp)


#: ``chi`` samples per poloidal grid point of the quasi-symmetric profile's averages (:func:`_qs_geometry`).
QS_PROFILE_OVERSAMPLE = 4
#: Refinement of the field grid for the Boozer-angle projection (:func:`_qs_harmonics`); 1 aliases a coarse grid.
QS_BOOZER_OVERSAMPLE = 2


def _upsample_spectrum(F: Array, *, factor: int) -> Array:
    """The 2D DFT ``F`` (``(..., nt, nz)``, numpy order) of grid samples, embedded in the DFT of the same
    trigonometric interpolant on a grid ``factor`` times finer in each angle (a single-point axis stays one point);
    an even axis's Nyquist coefficient is split between its two frequencies."""
    for axis in (-2, -1):
        n = int(F.shape[axis])
        if factor == 1 or n == 1:
            continue
        N = factor * n
        k = (np.fft.fftfreq(n, 1.0 / n)).astype(int)
        moved = jnp.moveaxis(F, axis, -1)
        out = jnp.zeros(moved.shape[:-1] + (N,), dtype=moved.dtype)
        regular = np.flatnonzero(~((n % 2 == 0) & (k == -(n // 2))))
        out = out.at[..., k[regular] % N].set(moved[..., regular])
        if n % 2 == 0:
            half = 0.5 * moved[..., n // 2]
            out = out.at[..., n // 2].add(half).at[..., N - n // 2].add(half)
        F = jnp.moveaxis(out * factor, -1, axis)
    return F


def _mirror_rows(values: Array, *, lasym: bool) -> Array:
    """Half-mesh rows ``(nsurf, ntheta2, nzeta)`` of a reflection-even quantity on the full poloidal circle
    (:func:`_full_surface_values` without dropping a row)."""
    if lasym:
        return values
    ntheta2, nzeta = int(values.shape[1]), int(values.shape[2])
    ntheta1 = max(2 * (ntheta2 - 1), 1)
    i_full = np.arange(ntheta1)
    i_src = np.where(i_full < ntheta2, i_full, ntheta1 - i_full)
    k = np.arange(nzeta)
    k_src = np.where(i_full[:, None] < ntheta2, k[None, :], (nzeta - k[None, :]) % nzeta)
    return values[:, np.broadcast_to(i_src[:, None], k_src.shape), k_src]


def _qs_boozer_harmonics(state: SpectralState, rt: SolverRuntime, helicity_n: int, hm: _HalfMeshFields):
    """The quasi-symmetric part of ``|B|`` on every half-mesh surface of a state (:func:`_qs_harmonics` of the
    solver's half-mesh grid: lambda's derivatives in the wout convention, the covariant ``B``, ``|B|``)."""
    setup = rt.setup
    lasym = bool(setup.lasym)
    geometry, _, _, fields, _ = _field_chain(state, rt)
    s = jnp.asarray(setup.s_full)
    root = jnp.sqrt(0.5 * (s[1:] + s[:-1]))[:, None, None]
    scale = (jnp.asarray(fields.lamscale) / jnp.asarray(setup.phips)[1:])[:, None, None]

    def half(even, odd):  # VMEC's half-mesh value of an (even, odd)-m split quantity
        even, odd = jnp.asarray(even), jnp.asarray(odd)
        return 0.5 * (even[1:] + even[:-1] + root * (odd[1:] + odd[:-1]))

    lam_t = _mirror_rows(half(geometry.dlambda_dtheta_even, geometry.dlambda_dtheta_odd) * scale, lasym=lasym)
    lam_z = _mirror_rows(half(geometry.dlambda_dzeta_even, geometry.dlambda_dzeta_odd) * scale, lasym=lasym)
    b_t = _mirror_rows(jnp.asarray(fields.bsubu)[1:], lasym=lasym)
    b_z = _mirror_rows(jnp.asarray(fields.bsubv)[1:], lasym=lasym)
    return _qs_harmonics(lam_t, lam_z, b_t, b_z, hm.bmag, hm.iota, int(rt.resolution.nfp), helicity_n)


def _qs_harmonics(lam_t, lam_z, b_t, b_z, modB, iota, nfp: int, helicity_n: int):
    """The quasi-symmetric part of ``|B|``: ``B_QS(chi) = sum_m c_m cos(m chi) + s_m sin(m chi)``,
    ``chi = theta_B - N zeta_B`` (Boozer angles, ``N = nfp * helicity_n``), ``m = 0 .. ntheta / 2``, from
    ``(nsurf, ntheta, nzeta)`` grids over ``theta in [0, 2 pi)`` x ``zeta in [0, 2 pi / nfp)`` of
    ``d(lambda)/d(theta, zeta)`` (``theta* = theta + lambda``), the covariant ``B_theta, B_zeta`` and ``|B|``.

    No Boozer transform's mode tables: ``theta_B = theta + lambda + iota nu``, ``zeta_B = zeta + nu``,
    ``nu = (w - I lambda) / (G + iota I)`` with ``dw/dtheta = B_theta - I``, ``dw/dzeta = B_zeta - G``
    (booz_xform's construction), the antiderivatives taken spectrally (FFT); ``c_m, s_m`` are the
    Boozer-angle averages of ``|B| cos(m chi)``, ``|B| sin(m chi)``, evaluated with the angle map's Jacobian.
    Memory and time are those of a few FFTs of the field grid per surface."""
    nsurf, nt, nz = modB.shape
    m = np.fft.fftfreq(nt, 1.0 / nt)[:, None]
    n = nfp * np.fft.fftfreq(nz, 1.0 / nz)[None, :]  # physical toroidal mode numbers
    nyquist = ((nt % 2 == 0) & (np.abs(m) == nt // 2)) | ((nz % 2 == 0) & (np.abs(n) == nfp * (nz // 2)))
    inv_m = jnp.asarray(np.where((m != 0) & ~nyquist, 1.0 / np.where(m != 0, m, 1), 0.0))
    inv_n = jnp.asarray(np.where((m == 0) & (n != 0) & ~nyquist, 1.0 / np.where(n != 0, n, 1), 0.0))

    def antiderivative(f_t, f_z):  # spectrum of g with dg/dtheta = f_t - <f_t>, dg/dzeta = f_z - <f_z>, <g> = 0
        F_t, F_z = jnp.fft.fft2(f_t, axes=(1, 2)), jnp.fft.fft2(f_z, axes=(1, 2))
        return -1j * (F_t * inv_m + F_z * inv_n)

    I_g, G_g = jnp.mean(b_t, axis=(1, 2)), jnp.mean(b_z, axis=(1, 2))
    lam_hat = antiderivative(lam_t, lam_z)
    nu_hat = (antiderivative(b_t, b_z) - I_g[:, None, None] * lam_hat) / (G_g + iota * I_g)[:, None, None]

    # the Boozer-angle projection on a finer grid (the fields' trigonometric interpolants): cos(m chi) of the
    # grid's poloidal band times |B| J is not resolved by the field grid itself, which aliases the top harmonics
    P = QS_BOOZER_OVERSAMPLE
    up = functools.partial(_upsample_spectrum, factor=P)
    lam_f, nu_f, B_f = up(lam_hat), up(nu_hat), up(jnp.fft.fft2(modB, axes=(1, 2)))
    NT, NZ = P * nt, (P * nz if nz > 1 else 1)
    mf = jnp.asarray(np.fft.fftfreq(NT, 1.0 / NT))[:, None] * 1j
    nf = jnp.asarray(nfp * np.fft.fftfreq(NZ, 1.0 / NZ))[None, :] * 1j

    def synth(F):
        return jnp.real(jnp.fft.ifft2(F, axes=(1, 2)))

    lam, nu, B = synth(lam_f), synth(nu_f), synth(B_f)
    lt, lz, nt_, nz_ = synth(mf * lam_f), synth(nf * lam_f), synth(mf * nu_f), synth(nf * nu_f)
    io = jnp.asarray(iota)[:, None, None]
    theta = jnp.asarray(2 * np.pi * np.arange(NT) / NT)[None, :, None]
    zeta = jnp.asarray(2 * np.pi * np.arange(NZ) / (NZ * nfp))[None, None, :]
    chi = theta + lam + io * nu - nfp * int(helicity_n) * (zeta + nu)
    jac = (1.0 + lt + io * nt_) * (1.0 + nz_) - (lz + io * nz_) * nt_
    weight = B * jac
    orders = np.arange(nt // 2 + 1)

    def harmonic(k):
        return (jnp.mean(weight * jnp.cos(k * chi), axis=(1, 2)), jnp.mean(weight * jnp.sin(k * chi), axis=(1, 2)))

    # checkpointed: a reverse pass keeps chi and |B| J, not every order's cos(m chi) on the fine grid
    c, sn = map_columns(jax.checkpoint(harmonic), jnp.asarray(orders, dtype=modB.dtype))
    norm = jnp.asarray(np.where(orders == 0, 1.0, 2.0))[:, None]
    return (c * norm).T, (sn * norm).T, jnp.asarray(orders, dtype=modB.dtype)


def _trig_extremum(c, sn, m, chi, sign, *, newton_steps: int = 8, candidates: int = 4):
    """``sign * max(sign * B)`` of ``B(chi) = sum c_m cos(m chi) + s_m sin(m chi)`` (one surface): Newton from the
    ``candidates`` largest local extrema of the samples ``chi``, the largest refined value; the location is held
    when differentiating (envelope theorem).  A quasi-symmetric profile has one maximum and one minimum a surface,
    so the candidates are margin."""
    cs, ss = jax.lax.stop_gradient(c), jax.lax.stop_gradient(sn)
    phase = m[:, None] * chi[None, :]
    B = sign * (cs @ jnp.cos(phase) + ss @ jnp.sin(phase))
    local = (B >= jnp.roll(B, 1)) & (B >= jnp.roll(B, -1))
    top, idx = jax.lax.top_k(jnp.where(local, B, -jnp.inf), min(int(candidates), chi.size))

    def refine(x):
        for _ in range(newton_steps):
            g = jnp.sum(m * (ss * jnp.cos(m * x) - cs * jnp.sin(m * x)))
            h = -jnp.sum(m * m * (cs * jnp.cos(m * x) + ss * jnp.sin(m * x)))
            x = x - jnp.where(sign * h < 0, g / jnp.where(h == 0, 1.0, h), 0.0)
        return x

    x = jax.vmap(refine)(chi[idx])
    values = jax.vmap(lambda a: jnp.sum(c * jnp.cos(m * a) + sn * jnp.sin(m * a)))(x)
    return sign * jnp.max(jnp.where(jnp.isfinite(top), sign * values, -jnp.inf))


def _qs_profile_geometry(state: SpectralState, rt: SolverRuntime, helicity_n: int, hm: _HalfMeshFields, *,
                         n_lambda: int, surfaces=None) -> RedlGeometry:
    """Redl's geometry from the quasi-symmetric part of ``|B|`` (:func:`_qs_boozer_harmonics`) on every half-mesh
    surface, or interpolated (harmonics and ``iota``, ``G``, ``I``, linearly in ``s``) onto ``surfaces``: see
    :func:`_qs_geometry`.  The profile has one maximum and one minimum a surface, where the full field on the
    solver grid can have several nearly equal local maxima, whose switching makes rows built on it non-smooth in
    the boundary."""
    c, sn, m = _qs_boozer_harmonics(state, rt, helicity_n, hm)
    iota, G, I = hm.iota, hm.G, hm.I  # noqa: E741 - Boozer I
    if surfaces is not None:
        surfaces = _as_1d(surfaces)
        c, sn, iota, G, I = (_interp_half_grid(a, surfaces, hm.s_half) for a in (c, sn, iota, G, I))  # noqa: E741
    return _qs_geometry(c, sn, m, int(hm.bmag.shape[1]), iota=iota, G=G, I=I,
                        surfaces=hm.s_half if surfaces is None else surfaces, psi_edge=-hm.phi_edge, nfp=hm.nfp,
                        n_lambda=n_lambda)


def _qs_geometry(c, sn, m, ntheta: int, *, iota, G, I, surfaces, psi_edge, nfp, n_lambda) -> RedlGeometry:  # noqa: E741
    """Redl's geometry of the quasi-symmetric profiles ``B(chi) = sum c_m cos(m chi) + s_m sin(m chi)``, as
    simsopt's ``RedlGeomBoozer`` builds it: ``f_t``, ``<B^2>``, ``<1/B>`` with the Boozer weight ``1/B^2`` on
    ``QS_PROFILE_OVERSAMPLE * ntheta`` samples of ``chi``, and ``B_max``/``B_min`` refined
    (:func:`_trig_extremum`), the former bounding f_t's ``lambda`` range."""
    nchi = QS_PROFILE_OVERSAMPLE * int(ntheta)
    chi = jnp.asarray(2 * np.pi * np.arange(nchi) / nchi, dtype=c.dtype)
    phase = m[:, None] * chi[None, :]
    B = c @ jnp.cos(phase) + sn @ jnp.sin(phase)  # (nsurf, nchi)
    w = 1.0 / (B * B)
    Vp = jnp.mean(w, axis=1)
    fsa_B2 = jnp.mean(B * B * w, axis=1) / Vp
    fsa_1overB = jnp.mean(w / B, axis=1) / Vp
    Bmax = jax.vmap(lambda a, b: _trig_extremum(a, b, m, chi, 1.0))(c, sn)
    Bmin = jax.vmap(lambda a, b: _trig_extremum(a, b, m, chi, -1.0))(c, sn)
    f_t = _trapped_fraction_integral(B, w, Vp, fsa_B2, Bmax, n_lambda)
    return RedlGeometry(
        surfaces=surfaces, iota=iota, G=G, I=I, R=(G + iota * I) * fsa_1overB,
        epsilon=(Bmax - Bmin) / (Bmax + Bmin), f_t=f_t, fsa_B2=fsa_B2, fsa_1overB=fsa_1overB, Bmin=Bmin,
        Bmax=Bmax, psi_edge=psi_edge, nfp=nfp)


def redl_qs_geometry_from_wout(wout, surfaces, helicity_n: int, *, ntheta: int = 64, nphi: int = 65,
                               n_lambda: int = N_LAMBDA) -> RedlGeometry:
    """:func:`_qs_geometry` from a wout dataset (the bootstrap rows' definition, wout lane): the half-mesh
    tables ``lmns``, ``bsubumnc``, ``bsubvmnc``, ``bmnc`` (and their LASYM partners) and ``iotas/bvco/buco``
    interpolated linearly onto ``surfaces``, synthesized on ``ntheta x nphi`` points of one field period, then
    :func:`_qs_harmonics`.  :func:`redl_geometry_from_wout` (the full field) is simsopt ``RedlGeomVmec``'s parity
    check only.

    Parameters
    ----------
    wout:
        A wout path or dataset.
    surfaces:
        Normalized toroidal flux values to evaluate on.
    helicity_n:
        Quasi-symmetry helicity, 0 (QA) or +/-1 (QH) (:func:`redl_helicity`).
    ntheta, nphi:
        Synthesis grid over one field period.
    n_lambda:
        Gauss-Legendre order of the trapped-fraction integral.

    Returns
    -------
    RedlGeometry
        The geometry on ``surfaces``.

    Raises
    ------
    ValueError
        If ``helicity_n`` is not that of QA or QH.
    """
    if isinstance(wout, (str, Path)):
        wout = read_wout(wout)
    wout = getattr(wout, "wout", wout)
    lasym = bool(getattr(wout, "lasym", False))
    surfaces = _as_1d(surfaces)
    nfp = int(wout.nfp)
    iotas = _as_1d(np.asarray(wout.iotas, dtype=float))
    ns = int(iotas.shape[0])
    s_half = _half_grid(ns, iotas.dtype)

    def half(values):
        return _interp_half_grid(values[1:], surfaces, s_half)

    theta = jnp.linspace(0.0, 2.0 * jnp.pi, int(ntheta), endpoint=False)
    phi = jnp.linspace(0.0, 2.0 * jnp.pi / nfp, int(nphi), endpoint=False)

    def tables(names, xm_name, xn_name):
        xm = _as_1d(np.asarray(getattr(wout, xm_name), dtype=float))
        xn = _as_1d(np.asarray(getattr(wout, xn_name), dtype=float))
        angle = theta[:, None, None] * xm[None, None, :] - phi[None, :, None] * xn[None, None, :]
        rows = {k: half(_mode_matrix(wout, k, ns=ns, mn=int(xm.size))) for k in names}
        return xm, xn, jnp.cos(angle), jnp.sin(angle), rows

    xm, xn, cos_a, sin_a, lam = tables(("lmns",) + (("lmnc",) if lasym else ()), "xm", "xn")
    synth = functools.partial(jnp.einsum, "sm,tpm->stp")
    lam_t = synth(lam["lmns"] * xm, cos_a)
    lam_z = synth(-lam["lmns"] * xn, cos_a)
    if lasym:
        lam_t, lam_z = lam_t - synth(lam["lmnc"] * xm, sin_a), lam_z + synth(lam["lmnc"] * xn, sin_a)
    names = ("bmnc", "bsubumnc", "bsubvmnc") + (("bmns", "bsubumns", "bsubvmns") if lasym else ())
    xm, xn, cos_a, sin_a, nyq = tables(names, "xm_nyq", "xn_nyq")
    modB, b_t, b_z = (synth(nyq[k], cos_a) for k in ("bmnc", "bsubumnc", "bsubvmnc"))
    if lasym:
        modB, b_t, b_z = (a + synth(nyq[k], sin_a)
                          for a, k in zip((modB, b_t, b_z), ("bmns", "bsubumns", "bsubvmns")))
    iota = half(iotas)
    c, sn, m = _qs_harmonics(lam_t, lam_z, b_t, b_z, modB, iota, nfp, redl_helicity(helicity_n))
    psi_edge = -_as_1d(np.asarray(wout.phi, dtype=float))[-1] / (2.0 * jnp.pi)
    return _qs_geometry(c, sn, m, int(ntheta), iota=iota, G=half(_as_1d(np.asarray(wout.bvco, dtype=float))),
                        I=half(_as_1d(np.asarray(wout.buco, dtype=float))), surfaces=surfaces, psi_edge=psi_edge,
                        nfp=nfp, n_lambda=n_lambda)


def redl_geometry_from_state(
    state: SpectralState,
    rt: SolverRuntime,
    *,
    surfaces=None,
    n_lambda: int = N_LAMBDA,
) -> RedlGeometry:
    """Redl geometry inputs as a traceable function of ``(state, runtime)``.

    ``|B|`` and ``sqrt(g)`` are taken on the solver's half-mesh internal grid
    (see :func:`_half_mesh_fields`); ``psi_edge = -signgs*hs*sum(phipf[1:])``.
    Per-surface scalars and the angular fields are linearly interpolated from
    the half mesh onto ``surfaces`` before :func:`compute_trapped_fraction`.

    Default ``surfaces``: ``linspace(0.05, 0.95, 16)`` — interior only; do
    not sample ``s -> 1`` where ``Te, Ti -> 0`` blows up the collisionality.
    """
    if surfaces is None:
        surfaces = np.linspace(0.05, 0.95, 16)
    surfaces = _as_1d(surfaces)
    return _geometry_from_half(_half_mesh_fields(state, rt), surfaces,
                               n_lambda=n_lambda)


def trapped_fraction_from_state(
    state: SpectralState,
    rt: SolverRuntime,
    *,
    surfaces=None,
    n_lambda: int = N_LAMBDA,
) -> Array:
    """Effective trapped fraction on requested normalized-flux surfaces.

    The default is VMEX's full radial mesh. The axis keeps only the regular
    poloidal ``m=0`` limit; the boundary uses linear half-mesh extrapolation.
    LASYM uses the full poloidal grid; symmetric runs restore it by parity.
    """
    if surfaces is None:
        surfaces = jnp.asarray(rt.setup.s_full)
    surfaces = _as_1d(surfaces)
    _, jacobian, _, fields, _ = _field_chain(state, rt)
    return _trapped_fraction_from_fields(
        total_pressure=fields.total_pressure,
        pressure=fields.pressure,
        sqrt_g=jacobian.sqrt_g,
        s_full=rt.setup.s_full,
        surfaces=surfaces,
        lasym=bool(rt.setup.lasym),
        n_lambda=n_lambda,
    )[-1]


# ===========================================================================
# Redl (2021) <J.B> formula (transcribed verbatim from simsopt j_dot_B_Redl)
# ===========================================================================


REDL_HELICITIES = (0, 1, -1)  # QA, QH: the quasi-symmetries Redl's fit (and simsopt's j_dot_B_Redl) covers


def redl_helicity(helicity_n) -> int:
    """``helicity_n`` checked for Redl's model: 0 (quasi-axisymmetry) or +/-1 (quasi-helical symmetry, units of
    ``nfp``).  Redl's formulae assume a quasi-symmetric field (the isomorphism ``iota -> iota - N``); a
    quasi-isodynamic or otherwise non-quasi-symmetric field has no such helicity and is refused, as is a missing
    one (no default: a QI deck must not silently take the QA isomorphism).

    Parameters
    ----------
    helicity_n:
        The requested helicity.

    Returns
    -------
    int
        ``helicity_n`` as an integer.

    Raises
    ------
    ValueError
        If ``helicity_n`` is missing or not 0, 1 or -1.
    """
    if helicity_n is None:
        raise ValueError("Redl's bootstrap model needs the field's quasi-symmetry helicity_n: 0 (QA) or +/-1 (QH); "
                         "it is not valid for quasi-isodynamic (QI) or other non-quasi-symmetric fields")
    if isinstance(helicity_n, bool) or int(helicity_n) != helicity_n or int(helicity_n) not in REDL_HELICITIES:
        raise ValueError(f"helicity_n={helicity_n!r}: Redl's bootstrap model covers QA (0) and QH (+/-1) only, "
                         "not quasi-isodynamic (QI) or other non-quasi-symmetric fields")
    return int(helicity_n)


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
    symmetry; anything else (a quasi-isodynamic field) is refused
    (:func:`redl_helicity`).  ``Te/Ti`` are clamped at 1 eV and ``ne`` at 1e17 1/m^3
    (belt-and-suspenders against sampling s -> 1 where the profiles vanish
    and the collisionality blows up).

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
    helicity_N = int(geom.nfp) * redl_helicity(helicity_n)
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


# ===========================================================================
# Traceable <J.B>_vmec — the MHD identity
# ===========================================================================


def _dds_half(f: Array, hs) -> Array:
    """d/ds of half-mesh radial samples on the half mesh itself.

    Central differences in the interior, one-sided two-point stencils at the
    ends (the spec's "half -> full central differences, ends extrapolated"
    evaluated back on the half mesh — identical values, no regridding).
    Works on 1D per-surface arrays (leading radial axis).
    """
    f = jnp.asarray(f)
    d0 = (f[1] - f[0]) / hs
    dn = (f[-1] - f[-2]) / hs
    return jnp.concatenate([d0[None], (f[2:] - f[:-2]) / (2.0 * hs), dn[None]])


def _jv_from_half(hm: _HalfMeshFields, surfaces) -> Array:
    """``<J.B>_vmec`` at ``surfaces`` from half-mesh fields (see :func:`vmec_j_dot_B`)."""
    Vp = jnp.mean(hm.w, axis=(1, 2))
    fsa_B2 = jnp.mean(hm.bmag * hm.bmag * hm.w, axis=(1, 2)) / Vp
    hs = hm.s_half[1] - hm.s_half[0]
    jv_half = hm.signgs * (fsa_B2 * _dds_half(hm.I, hs)
                           + hm.I * _dds_half(hm.p_int, hs)) / (MU0 * hm.phi_edge)
    return _interp_half_grid(jv_half, surfaces, hm.s_half)


def redl_current_derivative(profiles: KineticProfiles, helicity_n: int, state: SpectralState, rt: SolverRuntime,
                            *, n_lambda: int = N_LAMBDA) -> tuple[Array, Array]:
    """``(s, dI/ds)`` on the half mesh: the enclosed-current derivative [A] that makes ``<J.B>`` Redl's.

    The identity of :func:`vmec_j_dot_B`, ``<B^2> dI/ds + I dp/ds = mu0 Phi_a <J.B>`` (``I`` the Boozer
    ``buco``, ``p`` in ``mu0`` Pa), solved for ``dI/ds`` with Redl's ``<J.B>`` and the state's own ``I``,
    then converted to the toroidal current in A.  A current profile with this derivative at every
    half-mesh surface is self-consistent with Redl there; unlike ``<J.B>_vmec - <J.B>_Redl``, the
    difference of two such derivatives also sees a current that alternates from surface to surface.
    Traceable in ``(state, rt)``, smooth: Redl's geometry is that of the quasi-symmetric part of ``|B|``
    (:func:`_qs_profile_geometry`).

    Parameters
    ----------
    profiles:
        Kinetic profiles of Redl's formula.
    helicity_n:
        Quasi-symmetry helicity, 0 (QA) or +/-1 (QH).
    state, rt:
        A solved state and its runtime.
    n_lambda:
        Gauss-Legendre order of the trapped-fraction integral.

    Returns
    -------
    tuple
        ``(s, dI/ds)``: the half-mesh surfaces and the current derivative there [A].
    """
    return _redl_current_derivative(profiles, helicity_n, state, rt, n_lambda=n_lambda)[:2]


def _redl_current_derivative(profiles, helicity_n, state, rt, *, n_lambda=N_LAMBDA):
    """:func:`redl_current_derivative`'s ``(s, dI/ds)``, with the Redl geometry and ``<J.B>`` behind them.

    The geometry is the quasi-symmetric part of ``|B|`` (:func:`_qs_profile_geometry`, simsopt's
    ``RedlGeomBoozer`` definition): smooth in the equilibrium, unlike the full field's ``B_max`` on the solver
    grid, whose nearly equal local maxima switch.  The Redl current differs from the full-field one by the
    field's symmetry breaking."""
    hm = _half_mesh_fields(state, rt)
    geom = _qs_profile_geometry(state, rt, redl_helicity(helicity_n), hm, n_lambda=n_lambda)
    jr, _ = j_dot_B_redl(profiles, geom, helicity_n)
    return hm.s_half, _current_derivative(hm, jr), geom, jr


def near_axis_redl_ratio(profiles: KineticProfiles, helicity_n: int, geom: RedlGeometry, j_dot_b, x) -> Array:
    """Redl's ``<J.B>`` at ``s = x s_1`` over its value at ``s_1`` (``x < 1``), inside the first surface.

    The geometry is ``geom``'s at ``s_1`` (its first surface; ``j_dot_b`` Redl's ``<J.B>`` there) continued
    to the axis: ``eps ~ s^(1/2)`` and ``f_t ~ s^(1/4)`` (``f_t ~ eps^(1/2)``), ``iota``, ``G``, ``R`` and the
    flux-surface averages at their ``s_1`` values, while the kinetic profiles are evaluated at ``s``.  The
    banana-regime current follows ``s^(1/4)`` until the electron collisionality ``nu*_e ~ eps^(-3/2)``
    turns it into a collisional decay towards the axis.

    Parameters
    ----------
    profiles, helicity_n:
        As for :func:`j_dot_B_redl`.
    geom:
        Redl's geometry; its first surface is ``s_1``.
    j_dot_b:
        Redl's ``<J.B>`` on ``geom``'s surfaces.
    x:
        Fractions of ``s_1``.

    Returns
    -------
    Array
        The ratio at every ``x``.
    """
    x = jnp.asarray(x)

    def first(a):
        return jnp.broadcast_to(jnp.asarray(a)[0], x.shape)

    near = dataclasses.replace(
        geom, surfaces=first(geom.surfaces) * x, iota=first(geom.iota), G=first(geom.G), I=first(geom.I),
        R=first(geom.R), epsilon=first(geom.epsilon) * jnp.sqrt(x), f_t=first(geom.f_t) * x**0.25,
        fsa_B2=first(geom.fsa_B2), fsa_1overB=first(geom.fsa_1overB), Bmin=first(geom.Bmin), Bmax=first(geom.Bmax))
    return j_dot_B_redl(profiles, near, helicity_n)[0] / jnp.asarray(j_dot_b)[0]


#: Gauss-Legendre nodes of :func:`near_axis_redl_mean` in ``t = x^(1/4)``, converged to round-off in the banana
#: and collisional regimes.
N_AXIS = 32


def near_axis_redl_mean(profiles: KineticProfiles, helicity_n: int, geom: RedlGeometry, j_dot_b,
                        n_nodes: int = N_AXIS) -> Array:
    """The mean of :func:`near_axis_redl_ratio` over ``0 < s < s_1``: ``int_0^1 ratio(x) dx``.

    Gauss-Legendre in ``t = x^(1/4)`` (``dx = 4 t^3 dt``), where the banana-regime ``x^(1/4)`` is smooth.

    Parameters
    ----------
    profiles, helicity_n, geom, j_dot_b:
        As for :func:`near_axis_redl_ratio`.
    n_nodes:
        Gauss-Legendre nodes.

    Returns
    -------
    Array
        The mean ratio (a scalar).
    """
    nodes, weights = np.polynomial.legendre.leggauss(int(n_nodes))
    t = 0.5 * (nodes + 1.0)
    ratio = near_axis_redl_ratio(profiles, helicity_n, geom, j_dot_b, jnp.asarray(t**4))
    return jnp.sum(jnp.asarray(0.5 * weights * 4.0 * t**3) * ratio)


def current_derivative(j_dot_b, state: SpectralState, rt: SolverRuntime) -> tuple[Array, Array]:
    """``(s, dI/ds)`` on the half mesh that makes ``<J.B>`` equal ``j_dot_b`` [A T/m^2] there.

    :func:`redl_current_derivative` for any bootstrap model, e.g. a kinetic ``<J.B>``
    interpolated onto the half-mesh surfaces.  Traceable in ``(state, rt)``.

    Parameters
    ----------
    j_dot_b:
        The target ``<J.B>`` on the ``ns - 1`` half-mesh surfaces.
    state, rt:
        A solved state and its runtime.

    Returns
    -------
    tuple
        ``(s, dI/ds)``: the half-mesh surfaces and the current derivative there [A].
    """
    hm = _half_mesh_fields(state, rt)
    return hm.s_half, _current_derivative(hm, jnp.asarray(j_dot_b))


def _current_derivative(hm: _HalfMeshFields, j_dot_b: Array) -> Array:
    """dI/ds [A] solving ``<B^2> dI/ds + I dp/ds = mu0 Phi_a <J.B>`` on the half mesh (Boozer ``I``)."""
    w = jnp.abs(hm.w)  # |sqrt(g)|: the surface-average weight for either Jacobian sign (signgs)
    fsa_B2 = jnp.mean(hm.bmag * hm.bmag * w, axis=(1, 2)) / jnp.mean(w, axis=(1, 2))
    hs = hm.s_half[1] - hm.s_half[0]
    dI_boozer = (j_dot_b * MU0 * hm.phi_edge / hm.signgs - hm.I * _dds_half(hm.p_int, hs)) / fsa_B2
    return dI_boozer * 2.0 * jnp.pi / (MU0 * hm.signgs)


class HalfMeshCurrent:
    """A current profile with a value on every half-mesh surface, solved to be Redl's bootstrap current.

    ``I'(s)`` is a line-segment profile (``line_segment_ip``) with a knot on every half-mesh surface
    ``s_j = (j - 1/2) / (ns - 1)``.  Its values there (A per unit ``s``) are the unknowns, and each knot has the row
    ``sqrt(w_j) (I'(s_j) - I'_Redl(s_j)) / unit`` (:func:`redl_current_derivative`), a square block, so the current
    is Redl's wherever the equilibrium is.  The three-terms free boundary solves it together with the boundary
    (``ThreeTermsFreeBoundaryModel(bootstrap=)``), :func:`self_consistent_bootstrap` with ``profile="half_mesh"`` by
    fixed-point iteration at a fixed boundary.

    The rows' scale is the interface rows' own, not a weight.  ``w_j`` are the knots' trapezoid weights in the
    profile's integral (``1 / (ns - 1)`` on the half mesh), so the block's squared norm is the integral over ``s`` of
    ``((I' - I'_Redl) / unit)^2``, the same at any ``ns``; ``unit = L B0 / mu0`` (:meth:`set_unit`, by the model from
    its seed boundary: ``L`` the mean poloidal circumference, ``B0`` the area-RMS ``|B|`` that also scales the
    interface rows) is the toroidal current whose poloidal field on the boundary is ``B0``.  A current error is thus
    measured by the field it carries over ``|B|``, as ``B . n / B0`` and the net-current row ``(G - G_coil) /
    G_coil`` are.  The square Newton solve of the free boundary does not depend on the scale (its root is where every
    row vanishes), but its conditioning and merit function do: in these units the current block is conditioned
    like the boundary block, with no tuned weight and no dependence on the starting current.

    Inside the first surface VMEC sees the current only through the enclosed current at ``s_1`` (its current
    profile is sampled at the half-mesh surfaces), so the first segment carries exactly the enclosed current of
    Redl's current on the near-axis continuation of the ``s_1`` geometry: ``int_0^s1 I' ds = s_1 I'_Redl(s_1)
    R``, with ``R`` the mean of :func:`near_axis_redl_ratio` over ``0 < s < s_1`` (:func:`near_axis_redl_mean`;
    ``R = 4/5`` for the banana-regime ``s^(1/4)``, less where the collisionality diverges towards the axis).
    The knot at ``s = 0`` is the unknown that does so, with the row ``I'(0) - (2 R - 1) I'_Redl(s_1)``: its
    value is that of the straight segment with this integral, not the current density on the axis (which
    vanishes with the trapped fraction).  Fixed knots inside ``s_1`` would instead leave the piecewise-linear
    integral short of the model's.  The edge knot continues the last two linearly (it only enters CURTOR), and CURTOR is the profile's integral.  ``deck`` is the input
    on this profile, starting from its own current (``x0``, the initial condition only).

    The knot count ``ns + 1`` is not bound by VMEC2000's ``ndatafmax`` (101): VMEX's profiles have any length.
    Only an INDATA namelist or a WOUT holds at most 101 knots; writing one refits the profile to a subset of its
    knots, with a warning (:func:`~vmex.core.input.fit_aux_knots`).
    """

    def __init__(self, deck, profiles, helicity_n):
        from .profiles import current

        self.helicity_n = redl_helicity(helicity_n)  # QA or QH only, before any work
        if int(deck.ncurr) != 1:
            raise ValueError("a bootstrap current needs a prescribed current profile (ncurr = 1)")
        ns = int(np.asarray(deck.ns_array)[-1])
        s_half = (np.arange(1, ns) - 0.5) / (ns - 1)
        self.knots = np.r_[0.0, s_half, 1.0]
        self.current = functools.partial(current, "line_segment_ip", deck.ac, self.knots)  # (values, s) -> I(s)
        enclosed = functools.partial(current, deck.pcurr_type, deck.ac, deck.ac_aux_s, deck.ac_aux_f,
                                     bloat=deck.bloat)
        edge = float(enclosed(jnp.asarray(1.0)))
        slope = np.array(jax.vmap(jax.grad(enclosed))(jnp.asarray(self.knots[:-1])))
        # the axis knot: the straight first segment with the deck's enclosed current at s_1
        slope[0] = 2.0 * float(enclosed(jnp.asarray(s_half[0]))) / s_half[0] - slope[1]
        self.x0 = float(deck.curtor) / edge * slope if edge else np.zeros(slope.size)  # the deck's current, in A
        # d CURTOR / d v: the unknown knots' trapezoid weights (CURTOR is linear in v), the edge value included
        self.weights = np.asarray(jax.grad(lambda v: self.current(self.values(v), 1.0))(jnp.ones(slope.size)))
        self.unit = None
        self.profiles = profiles
        values = self.values(self.x0)  # knots and values in one replace: VmecInput trims them to a common length
        self.deck = dataclasses.replace(deck, pcurr_type="line_segment_ip", ac_aux_s=self.knots,
                                        ac_aux_f=np.asarray(values), curtor=float(self.current(values, 1.0)))

    def set_unit(self, length, field):
        """Measure the current in ``unit = length field / mu0`` [A]: ``length`` the boundary's mean poloidal
        circumference [m] and ``field`` its RMS ``|B|`` [T].  Also the coordinates' scale ``x_scale``."""
        self.unit = float(length) * float(field) / MU0
        if not (np.isfinite(self.unit) and self.unit > 0):
            raise ValueError(f"the current unit must be positive and finite, got {self.unit}")

    def _require_unit(self) -> float:
        if self.unit is None:
            raise ValueError("the current unit is not set (HalfMeshCurrent.set_unit)")
        return self.unit

    @property
    def x_scale(self):
        """The coordinates' scale: :attr:`unit` (set by :meth:`set_unit`) on every knot."""
        return np.full(self.x0.size, self._require_unit())

    def values(self, v):
        """Every knot value from the unknowns ``v`` (the axis knot's, then the half-mesh ones)."""
        v = jnp.asarray(v)
        # the edge value: the line through the last two half-mesh knots (1 - 3h/2, 1 - h/2) at s = 1
        return jnp.concatenate([v, (1.5 * v[-1] - 0.5 * v[-2])[None]])

    def deck_with(self, deck, v):
        """``deck`` with the knot values ``v``."""
        values = self.values(v)
        return dataclasses.replace(deck, ac_aux_f=np.asarray(values), curtor=float(self.current(values, 1.0)))

    def apply(self, params, v):
        """``ImplicitParams`` with the knot values ``v`` and their CURTOR (traceable)."""
        values = self.values(v)
        return dataclasses.replace(params, ac_aux_f=values, curtor=self.current(values, 1.0))

    def target(self, state, runtime):
        """The axis knot's ``(2 R - 1) I'_Redl(s_1)``, then ``I'_Redl`` on the half-mesh surfaces (traceable)."""
        _, target, geom, jr = _redl_current_derivative(self.profiles, self.helicity_n, state, runtime)
        mean = near_axis_redl_mean(self.profiles, self.helicity_n, geom, jr)
        return jnp.concatenate([((2.0 * mean - 1.0) * target[0])[None], target])

    def mismatch(self, state, runtime, params):
        """``(I' - I'_Redl, I'_Redl)`` at the axis knot and the half-mesh surfaces (:meth:`target`)."""
        target = self.target(state, runtime)
        return params.ac_aux_f[: target.size] - target, target

    def rows(self, state, runtime, params):
        """The self-consistency rows ``sqrt(w_j) (I' - I'_Redl) / unit`` (traceable)."""
        return jnp.asarray(np.sqrt(self.weights) / self._require_unit()) * self.mismatch(state, runtime, params)[0]


def vmec_j_dot_B(
    state: SpectralState,
    rt: SolverRuntime,
    *,
    surfaces=None,
) -> Array:
    """Traceable equilibrium ``<J.B>`` [A*T/m^2] via the MHD identity.

    For a VMEC equilibrium the flux-surface-averaged parallel current obeys
    (validated in the Zenodo
    ``convertSfincsToVmecCurrentProfile`` script against VMEC's own
    ``jdotb``)

        <J.B>(s) = [<B^2> dI/ds + mu0 I dp/ds] / (2 pi psi_a)

    with ``I(s) = signgs*(2 pi/mu0)*buco(s)`` [A] (VMEC's
    ``ctor = signgs*(2 pi/mu0)*buco(ns)`` sign convention), ``p`` in Pa and
    ``psi_a = phi(1)/(2 pi)`` [Wb/rad].  In core internal units
    (``fields.pressure`` is ``mu0*Pa``, ``buco`` [T*m]) this reduces to
    ``signgs*(<B^2> buco' + buco p_int') / (mu0 psi_a)``.  Radial derivatives
    are half-mesh central differences (one-sided at the ends), then the
    half-mesh profile is linearly interpolated onto ``surfaces`` (default:
    ``linspace(0.05, 0.95, 16)``, matching
    :func:`redl_geometry_from_state`).

    Agrees with the wout-engine ``jdotb`` (jxbforce.f) to a few 1e-4
    relative on the Zenodo finite-beta optima — no ``bsubs`` port needed.
    """
    if surfaces is None:
        surfaces = np.linspace(0.05, 0.95, 16)
    return _jv_from_half(_half_mesh_fields(state, rt), _as_1d(surfaces))


def vmec_j_dot_B_from_wout(wout, surfaces, *, geom: RedlGeometry | None = None,
                           ntheta: int = 64, nphi: int = 65) -> Array:
    """The section-6.2 identity evaluated from wout tables (parity lane).

    Same formula as :func:`vmec_j_dot_B` with ``buco``/``pres``/``phi`` read
    from the wout dataset and ``<B^2>`` synthesized from ``bmnc``/``gmnc``
    (plus the ``bmns``/``gmns`` partners when ``lasym``) at the requested
    ``surfaces``; pass ``geom`` (a :class:`RedlGeometry` from
    :func:`redl_geometry_from_wout` at the *same* surfaces) to reuse its
    ``fsa_B2`` instead.  This is the validation lane — the wout ``jdotb``
    itself (jxbforce.f) is what :class:`RedlBootstrapMismatch` consumes.
    """
    if isinstance(wout, (str, Path)):
        wout = read_wout(wout)
    wout = getattr(wout, "wout", wout)
    surfaces = _as_1d(surfaces)

    ns = int(wout.ns)
    s_full = jnp.linspace(0.0, 1.0, ns)
    hs = s_full[1] - s_full[0]
    s_half = _half_grid(ns, s_full.dtype)
    buco = _as_1d(np.asarray(wout.buco, dtype=float))[1:]
    p_int = MU0 * _as_1d(np.asarray(wout.pres, dtype=float))[1:]
    phi_edge = _as_1d(np.asarray(wout.phi, dtype=float))[-1] / (2.0 * jnp.pi)
    signgs = float(wout.signgs)

    if geom is not None:
        fsa_B2 = jnp.asarray(geom.fsa_B2)
    else:
        xm = _as_1d(np.asarray(wout.xm_nyq, dtype=float))
        xn = _as_1d(np.asarray(wout.xn_nyq, dtype=float))
        mn = int(xm.shape[0])

        # Every m != 0 harmonic vanishes at the magnetic axis, where the
        # surface degenerates to a point.  redl_geometry_from_wout zeroes them
        # there; without the same treatment the two lanes disagreed by 4.0e-4
        # at s = 0 while matching bit-for-bit everywhere else.
        axis_modes = (np.asarray(surfaces) == 0.0)[:, None] & (xm != 0.0)[None, :]

        def half(name):
            values = _interp_half_grid(
                _mode_matrix(wout, name, ns=ns, mn=mn)[1:], surfaces, s_half)
            return jnp.where(axis_modes, 0.0, values)

        theta1d = jnp.linspace(0.0, 2.0 * jnp.pi, int(ntheta), endpoint=False)
        phi1d = jnp.linspace(0.0, 2.0 * jnp.pi / int(wout.nfp), int(nphi),
                             endpoint=False)
        angle = (theta1d[:, None, None] * xm[None, None, :]
                 - phi1d[None, :, None] * xn[None, None, :])
        cosangle = jnp.cos(angle)
        modB = jnp.einsum("sm,tpm->stp", half("bmnc"), cosangle)
        sqrtg = jnp.einsum("sm,tpm->stp", half("gmnc"), cosangle)
        if bool(getattr(wout, "lasym", False)):
            # On a LASYM wout the sine families carry the up-down-asymmetric
            # half of |B| and sqrt(g); dropping them symmetrizes <B^2> (same
            # synthesis as :func:`redl_geometry_from_wout`).
            sinangle = jnp.sin(angle)
            modB += jnp.einsum("sm,tpm->stp", half("bmns"), sinangle)
            sqrtg += jnp.einsum("sm,tpm->stp", half("gmns"), sinangle)
        sqrtg = jnp.abs(sqrtg)
        fsa_B2 = (jnp.mean(modB * modB * sqrtg, axis=(1, 2))
                  / jnp.mean(sqrtg, axis=(1, 2)))

    dI = _interp_half_grid(_dds_half(buco, hs), surfaces, s_half)
    dp = _interp_half_grid(_dds_half(p_int, hs), surfaces, s_half)
    I = _interp_half_grid(buco, surfaces, s_half)  # noqa: E741 - Boozer I
    return signgs * (fsa_B2 * dI + I * dp) / (MU0 * phi_edge)


# ===========================================================================
# f_boot objective (simsopt VmecRedlBootstrapMismatch)
# ===========================================================================


class RedlBootstrapMismatch:
    """Normalized ``<J.B>_vmec`` vs ``<J.B>_Redl`` mismatch (paper ``f_boot``).

    Residual vector (simsopt ``VmecRedlBootstrapMismatch.residuals``
    verbatim)::

        R_j = (Jv(s_j) - Jr(s_j)) / sqrt(sum_k (Jv(s_k) + Jr(s_k))**2)

    over the geometry ``surfaces``, so ``sum(R**2)`` is the paper's

        f_boot = int ds [<J.B>_vmec - <J.B>_Redl]^2
                 / int ds [<J.B>_vmec + <J.B>_Redl]^2

    (bounded by 1; the denominator depends on the dofs — the simsopt/paper
    self-normalizing convention, kept as is).  Dual-lane shape mirroring
    :class:`~vmex.core.optimize.QuasisymmetryRatioResidual`:

    - :meth:`J`/:meth:`residuals` (wout lane, FD-only): ``Jr`` from
      :func:`redl_qs_geometry_from_wout`, ``Jv`` from the wout ``jdotb``
      (jxbforce.f) interpolated onto ``surfaces``.
    - :meth:`residuals_state` (traceable lane): one
      :func:`_half_mesh_fields` pass feeds both the Redl geometry and the
      MHD-identity ``Jv``, so the term composes with
      ``least_squares(jac="implicit")`` (profiles fixed, geometry traced).

    ``helicity_n`` is 0 for QA, -1/+1 for QH (units of ``nfp``, simsopt
    convention).  Default ``surfaces``: ``linspace(0.05, 0.95, 16)``
    (interior only — never sample ``s -> 1``).  Usage as
    an objective term (target 0; the normalization already scales it)::

        terms = [(qs.residuals_state, 0.0, 1.0),
                 (boot.residuals_state, 0.0, w_boot), ...]
    """

    name = "f_boot"

    def __init__(
        self,
        profiles: KineticProfiles,
        helicity_n: int,
        surfaces=None,
        *,
        ntheta: int = 64,
        nphi: int = 65,
        n_lambda: int = N_LAMBDA,
    ):
        self.profiles = profiles
        self.helicity_n = redl_helicity(helicity_n)
        self.surfaces = np.atleast_1d(np.asarray(
            np.linspace(0.05, 0.95, 16) if surfaces is None else surfaces,
            dtype=float))
        self.ntheta = int(ntheta)
        self.nphi = int(nphi)
        self.n_lambda = int(n_lambda)

    @staticmethod
    def _residual_vector(jv: Array, jr: Array) -> jnp.ndarray:
        denom_sq = jnp.sum((jv + jr) ** 2)
        tiny = jnp.asarray(jnp.finfo(jnp.asarray(denom_sq).dtype).tiny)
        return (jv - jr) / jnp.sqrt(jnp.maximum(denom_sq, tiny))

    # -- wout-table evaluation (simsopt parity; FD lane) -----------------------

    def current_profiles(self, wout) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Return ``(s, <J.B>_VMEX, <J.B>_Redl)`` for reporting and plots."""
        wout = getattr(wout, "wout", wout)
        if isinstance(wout, (str, Path)):
            wout = read_wout(wout)
        surfaces = _as_1d(self.surfaces)
        geom = redl_qs_geometry_from_wout(wout, surfaces, self.helicity_n, ntheta=self.ntheta,
                                          nphi=self.nphi, n_lambda=self.n_lambda)
        jr, _ = j_dot_B_redl(self.profiles, geom, self.helicity_n)
        s_full = jnp.linspace(0.0, 1.0, int(wout.ns))
        jv = jnp.interp(surfaces, s_full,
                        _as_1d(np.asarray(wout.jdotb, dtype=float)))
        return surfaces, jv, jr

    def residuals(self, wout) -> jnp.ndarray:
        """Residual vector from a wout-like object / path / ``Equilibrium``."""
        _surfaces, jv, jr = self.current_profiles(wout)
        return self._residual_vector(jv, jr)

    def profile(self, wout) -> jnp.ndarray:
        """Per-surface squared residuals (``sum = total``)."""
        r = self.residuals(wout)
        return r * r

    def total(self, wout) -> Array:
        """Scalar ``f_boot = sum(residuals**2)``."""
        r = self.residuals(wout)
        return jnp.sum(r * r)

    def J(self, eq) -> jnp.ndarray:
        """Objective-term entry point for :func:`~vmex.core.optimize.least_squares`."""
        return self.residuals(eq)

    __call__ = J  # the instance itself can be an objective term

    # -- traceable (state, runtime) evaluation --------------------------------

    def residuals_state(self, state: SpectralState, rt: SolverRuntime) -> jnp.ndarray:
        """Traceable residual vector (implicit lane; ``sum(r**2) = total_state``).

        Same ``R_j`` with ``Jr`` from :func:`_qs_profile_geometry` and
        ``Jv`` from :func:`vmec_j_dot_B` — one shared field-chain pass.
        Agrees with :meth:`residuals` at discretization level (solver
        internal grid + the identity ``Jv`` vs the 64x65 wout synthesis +
        jxbforce ``jdotb``).
        """
        surfaces = _as_1d(self.surfaces)
        hm = _half_mesh_fields(state, rt)
        geom = _qs_profile_geometry(state, rt, self.helicity_n, hm, n_lambda=self.n_lambda, surfaces=surfaces)
        jr, _ = j_dot_B_redl(self.profiles, geom, self.helicity_n)
        jv = _jv_from_half(hm, surfaces)
        return self._residual_vector(jv, jr)

    def profile_state(self, state: SpectralState, rt: SolverRuntime) -> Array:
        """Traceable per-surface squared residuals."""
        r = self.residuals_state(state, rt)
        return r * r

    def total_state(self, state: SpectralState, rt: SolverRuntime) -> Array:
        """Traceable scalar ``f_boot``."""
        r = self.residuals_state(state, rt)
        return jnp.sum(r * r)


# ===========================================================================
# Fixed-boundary Picard self-consistency loop (secondary lane)
# ===========================================================================


def _half_mesh_picard(inp, profiles, helicity_n, mismatch, *, n_iter, tol, relax, solve_kwargs, verbose):
    """:func:`self_consistent_bootstrap` on a :class:`HalfMeshCurrent` profile: ``I' <- I'_Redl`` per surface."""
    from . import optimize as opt

    block = HalfMeshCurrent(inp, profiles, helicity_n)
    inp, v = block.deck, block.x0
    history, state, eq, converged = [], None, None, False
    for it in range(int(n_iter)):
        eq = opt.solve_equilibrium(inp, initial_state=state, **solve_kwargs)
        state = eq.state
        target = np.asarray(block.target(eq.solution, eq.solver_context))
        delta = float(np.max(np.abs(target - v)) / np.max(np.abs(target)))
        history.append(dict(curtor=float(inp.curtor), delta=delta, f_boot=float(mismatch.total(eq.wout))))
        if verbose:
            print(f"[self_consistent_bootstrap] it {it}: curtor={inp.curtor:.6e} delta={delta:.3e}", flush=True)
        if delta <= float(tol):
            converged = True
            break
        v = (1.0 - relax) * v + relax * target
        inp = block.deck_with(inp, v)
    return BootstrapPicardResult(input=inp, equilibrium=eq, converged=converged, iterations=len(history),
                                 history=tuple(history))


@dataclass(frozen=True)
class BootstrapPicardResult:
    """Outcome of :func:`self_consistent_bootstrap`.

    ``input`` carries the final ``AC``/``CURTOR`` (``ncurr=1``,
    ``pcurr_type="power_series"``); ``equilibrium`` is the last solved
    :class:`~vmex.core.optimize.Equilibrium`; ``history`` is one dict per
    iteration (``curtor``, ``delta``, ``f_boot``).  ``delta`` is the
    self-consistency error ``max|I'_Redl - I'_applied| / max|I'_Redl|``
    between the Redl-implied and the currently applied current profile.
    """

    input: VmecInput
    equilibrium: Any
    converged: bool
    iterations: int
    history: tuple


def _picard_dds_matrix(ns: int, ds: float) -> np.ndarray:
    """4th-order d/ds collocation matrix on ``linspace(0, 1, ns)`` (host).

    Verbatim port of the finite-difference matrix of the Zenodo
    ``convertSfincsToVmecCurrentProfile`` script (centered 5-point interior
    stencil, one-sided 5-point rows at both boundaries); requires
    ``ns >= 5``.
    """
    dds = (np.diag(1.0 / (12 * ds) * np.ones(ns - 2), -2)
           - np.diag(2.0 / (3 * ds) * np.ones(ns - 1), -1)
           + np.diag(2.0 / (3 * ds) * np.ones(ns - 1), 1)
           - np.diag(1.0 / (12 * ds) * np.ones(ns - 2), 2))
    dds[0, :5] = np.array([-25.0 / 12, 4.0, -3.0, 4.0 / 3, -1.0 / 4]) / ds
    dds[1, :5] = np.array([-1.0 / 4, -5.0 / 6, 3.0 / 2, -1.0 / 2, 1.0 / 12]) / ds
    dds[-1, -5:] = np.array([1.0 / 4, -4.0 / 3, 3.0, -4.0, 25.0 / 12]) / ds
    dds[-2, -5:] = np.array([-1.0 / 12, 1.0 / 2, -3.0 / 2, 5.0 / 6, 1.0 / 4]) / ds
    return dds


def self_consistent_bootstrap(
    inp: VmecInput,
    profiles: KineticProfiles,
    helicity_n: int,
    *,
    n_iter: int = 10,
    tol: float = 1e-3,
    relax: float = 1.0,
    degree: int = 12,
    s_eval=None,
    solve_kwargs: dict | None = None,
    verbose: bool = False,
    profile: str = "power_series",
) -> BootstrapPicardResult:
    """Fixed-boundary Picard iteration to a bootstrap-consistent current profile.

    The Zenodo ``convertSfincsToVmecCurrentProfile`` "smooth method" with
    the Redl ``<J.B>`` in place of SFINCS, the self-consistent iteration of
    Landreman, Buller & Drevlak (2022) (secondary lane).  Host-side loop (no
    AD through it):

    1. solve the equilibrium (hot-restarted from the previous iterate);
    2. ``Jr = j_dot_B_redl(profiles, redl_geometry_from_wout(...))`` on the
       interior ``s_eval`` grid (default ``linspace(0.02, 0.98, 49)``),
       pinned to 0 at ``s = 0, 1`` and interpolated onto the full mesh;
    3. invert the section-6.2 identity for the enclosed current: solve
       ``[<B^2> d/ds + mu0 dp/ds] I = 2 pi psi_a Jr`` with ``I(0) = 0``
       (dense collocation solve, :func:`_picard_dds_matrix`), then recompute
       the *smooth* ``dI/ds = (2 pi psi_a Jr - mu0 I dp/ds)/<B^2>``;
    4. under-relax ``I' <- (1 - relax) I'_prev + relax I'_new``, refit the
       VMEC ``AC`` power series (``numpy.polynomial.polyfit``, ascending —
       ``pcurr_type="power_series"`` parameterizes ``I'``) and set
       ``CURTOR = I(1)``;
    5. stop when ``delta = max|I'_new - I'_applied| / max|I'_new| <= tol``.

    ``relax = 1.0`` is the plain fixed point (paper expectation: <= 5
    iterations at beta <= 2.5%); use ``relax = 0.5`` when the current
    dominates iota (e.g. tokamaks / the beta = 5% QH), where the
    ``<J.B> ~ 1/iota ~ 1/I`` feedback makes the undamped map marginally
    stable.  ``degree`` is clipped to ``ns - 2``.  The first solve uses
    ``inp``'s own current settings (any ``pcurr_type``); every refit
    switches to ``ncurr=1`` + ``power_series``.

    ``profile="half_mesh"`` instead iterates a :class:`HalfMeshCurrent`
    (a knot on every half-mesh surface and on the axis): each step sets
    ``I'`` there to :func:`redl_current_derivative` of the last equilibrium
    (under-relaxed), with no fit and no ``s_eval``; ``delta`` is the same
    relative error, on the half-mesh surfaces.  ``input`` is then on that
    ``line_segment_ip`` profile.
    """
    from . import optimize as opt

    solve_kwargs = dict(solve_kwargs or {})
    if s_eval is None:
        s_eval = np.linspace(0.02, 0.98, 49)
    s_eval = np.atleast_1d(np.asarray(s_eval, dtype=float))
    relax = float(relax)
    if not 0.0 < relax <= 1.0:
        raise ValueError(f"relax must be in (0, 1], got {relax}")
    if profile not in ("power_series", "half_mesh"):
        raise ValueError(f"profile must be 'power_series' or 'half_mesh', got {profile!r}")

    mismatch = RedlBootstrapMismatch(profiles, helicity_n,
                                     surfaces=np.clip(s_eval, 0.05, 0.95))
    if profile == "half_mesh":
        return _half_mesh_picard(inp, profiles, helicity_n, mismatch, n_iter=n_iter, tol=tol, relax=relax,
                                 solve_kwargs=solve_kwargs, verbose=verbose)
    history: list[dict] = []
    state = None
    dIds_applied = None   # I'(s_full) currently driving the equilibrium
    eq = None
    converged = False

    for it in range(int(n_iter)):
        try:
            eq = opt.solve_equilibrium(inp, initial_state=state, **solve_kwargs)
        except Exception:
            if state is None:
                raise
            eq = opt.solve_equilibrium(inp, **solve_kwargs)  # cold fallback
        state = eq.state
        w = eq.wout
        ns = int(w.ns)
        if ns < 5:
            raise ValueError("self_consistent_bootstrap requires ns >= 5")
        s_full = np.linspace(0.0, 1.0, ns)
        ds = s_full[1] - s_full[0]
        s_half = s_full[1:] - 0.5 * ds

        geom = redl_geometry_from_wout(w, s_eval)
        jr, _ = j_dot_B_redl(profiles, geom, helicity_n)
        jr_full = np.interp(s_full, np.concatenate([[0.0], s_eval, [1.0]]),
                            np.concatenate([[0.0], np.asarray(jr), [0.0]]))
        fsa_B2_full = np.interp(s_full, s_eval, np.asarray(geom.fsa_B2))

        pres = np.asarray(w.pres, dtype=float)[1:]                # Pa, half mesh
        dp_half = np.empty_like(pres)
        dp_half[1:-1] = (pres[2:] - pres[:-2]) / (2 * ds)
        dp_half[0] = (pres[1] - pres[0]) / ds
        dp_half[-1] = (pres[-1] - pres[-2]) / ds
        dpds = np.interp(s_full, s_half, dp_half)
        psi_a = float(np.asarray(w.phi)[-1]) / (2.0 * np.pi)

        dds = _picard_dds_matrix(ns, ds)
        matrix = (np.diag(fsa_B2_full) @ dds + np.diag(MU0 * dpds)) / (2 * np.pi * psi_a)
        matrix[0, :] = 0.0
        matrix[0, 0] = 1.0
        rhs = jr_full.copy()
        rhs[0] = 0.0
        I_solve = np.linalg.solve(matrix, rhs)
        dIds_new = (jr_full * 2 * np.pi * psi_a - MU0 * I_solve * dpds) / fsa_B2_full

        scale = max(float(np.max(np.abs(dIds_new))), np.finfo(float).tiny)
        delta = (np.inf if dIds_applied is None
                 else float(np.max(np.abs(dIds_new - dIds_applied))) / scale)
        f_boot = float(mismatch.total(w))
        history.append(dict(curtor=float(inp.curtor), delta=delta, f_boot=f_boot))
        if verbose:
            print(f"[self_consistent_bootstrap] it {it}: curtor={inp.curtor:.6e} "
                  f"delta={delta:.3e} f_boot={f_boot:.3e}", flush=True)
        if delta <= float(tol):
            converged = True
            break

        dIds_use = (dIds_new if dIds_applied is None
                    else (1.0 - relax) * dIds_applied + relax * dIds_new)
        deg = int(min(int(degree), ns - 2))
        coeffs = np.polynomial.polynomial.polyfit(s_full, dIds_use, deg)
        ac = np.zeros(max(21, deg + 1))
        ac[:deg + 1] = coeffs
        curtor = float(np.trapezoid(dIds_use, s_full))
        inp = dataclasses.replace(inp, ncurr=1, pcurr_type="power_series",
                                  ac=ac, curtor=curtor)
        dIds_applied = dIds_use

    return BootstrapPicardResult(
        input=inp, equilibrium=eq, converged=converged,
        iterations=len(history), history=tuple(history))
