"""Traceable omnigenity/QI objective.

A quasi-isodynamic (QI) omnigenity residual evaluated as a pure, traceable
function of a converged ``(SpectralState, SolverRuntime)`` pair, so it
composes with the implicit-gradient least-squares lane
(``vmex.core.optimize.least_squares(..., jac="implicit")``) exactly like
:class:`~vmex.core.optimize.QuasisymmetryRatioResidual` — no wout tables,
no host booz_xform round-trip.

Two pieces:

1. :func:`boozer_spectrum_state` — a *traceable* Boozer ``|B|`` transform.
   vmex owns only the equilibrium side: snapping requested surfaces to the
   half mesh and building wout-convention single-surface spectral tables
   from the live state (:func:`vmex.core.boozer_tables.boozer_input_tables`).
   The Boozer transform itself — generating potential, Boozer angles, and
   the angle-transform quadrature — executes in ``booz_xform_jax``'s
   jittable kernel (``booz_xform_jax.jax_api.booz_xform_jax_impl``) for
   symmetric and ``lasym`` states alike, so vmex carries no second
   implementation of the transform and the whole chain stays end-to-end
   differentiable.

2. :func:`omnigenity_residual` / :class:`QIResidual` — a smooth, lightweight
   surrogate for poloidally-closed-contour (``M = 0, N = 1``) omnigenity.
   It distills conditions used in the constructed-QI target of
   **Goodman et al., "Constructing precisely quasi-isodynamic magnetic
   fields", J. Plasma Phys. 89, 905890504 (2023), arXiv:2211.09829** into a
   level-set form: on each surface, ``|B|`` is sampled along
   Boozer field lines ``theta_B = alpha + iota * phi_B`` over one field
   period and the residual stacks, per surface,

   - **bounce-distance uniformity** (``well_weight``): for every trapping
     level ``B*``, the bounce distance ``delta(alpha, B*)`` between the two
     monotone branches of the magnetic well (smooth occupancy integrals of
     the running-maximum branch envelopes) minus its field-line average —
     the Cary–Shasharina omnigenity condition (Cary & Shasharina, PRL 78,
     674 (1997)) that Goodman's "shuffle" step enforces;
   - **extremum alignment** (``extremum_weight``): the per-field-line
     ``B_min``/``B_max`` minus their field-line averages — poloidal closure
     of the extremal ``|B|`` contours (Goodman's "align the maxima" step;
     also the flat-``B_max`` condition of Dudt et al., J. Plasma Phys. 90,
     905900120 (2024), arXiv:2305.08026);
   - **single-well monotonicity** (``squash_weight``): the pointwise distance
     between ``|B|`` and its monotone branch envelopes — Goodman's "squash"
     distance, penalizing side extrema (multiple wells per period).

   Each piece is an exact zero of an exactly QI field, every operation is
   smooth or piecewise-smooth (sigmoid occupancies, running maxima), and the
   full pipeline is jit/grad/jvp-transparent for the implicit lane.  The
   converse does not hold at finite sampling: this inexpensive surrogate can
   be driven low by a field that still has a large full squash-and-shuffle
   distance.  Use :class:`vmex.core.qi.ConstructedQIResidual` for production
   QI optimization and resolved reporting.

:func:`j_invariant_qi_maxj_residual_from_boozer` additionally provides the
constructed-well J-invariant QI/max-J surrogate used by NEOPAX, retaining
its normalization separately from the fixed-physical-pitch objectives.

Scope notes
-----------
- Symmetric and non-stellarator-symmetric (``lasym``) states are supported;
  the latter retain both cosine and sine Boozer ``|B|`` harmonics.
- Requested surfaces are snapped to the *nearest half-mesh surface* (the
  Boozer transform is a per-surface construction — same convention as
  :func:`vmex.core.optimize.boozer_modes_from_wout`), not interpolated.
- The wout-engine analogue for cross-checks is
  :func:`vmex.core.optimize.quasi_isodynamic_residual_from_wout`
  (host booz_xform_jax, finite-difference-only).
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any, Iterable

import numpy as np

import jax
import jax.numpy as jnp

from .boozer_tables import boozer_input_tables, high_order_boozer_input_tables
from .solver import SolverRuntime, SpectralState
from .statephysics import _as_1d

__all__ = [
    "boozer_spectrum_high_order",
    "boozer_spectrum_state",
    "omnigenity_residual",
    "j_invariant_qi_maxj_residual_from_boozer",
    "QIResidual",
]

Array = Any


# ---------------------------------------------------------------------------
# Traceable Boozer |B| spectrum from a core state
# ---------------------------------------------------------------------------


def _nearest_half_mesh_rows(ns: int, surfaces) -> tuple[np.ndarray, np.ndarray]:
    """Return half-mesh coordinates and one-based nearest row indices.

    Exact midpoint ties select the lower-flux surface, matching booz_xform's
    ``s_in`` lookup independently of round-off in the two grid constructions.
    """
    s_half = (np.arange(ns - 1) + 0.5) / (ns - 1)
    requested = np.atleast_1d(
        np.asarray(list(np.ravel(surfaces)), dtype=float)
    )
    if requested.size == 0:
        raise ValueError("surfaces must be non-empty")
    rows = []
    for surface in requested:
        distance = np.abs(s_half - surface)
        tolerance = 8.0 * np.finfo(float).eps * max(1.0, abs(float(surface)))
        nearest = np.flatnonzero(distance <= distance.min() + tolerance)[0]
        rows.append(int(nearest) + 1)
    return s_half, np.asarray(rows, dtype=int)


def _refine_booz_grids(constants, grids, oversample, nfp):
    """booz_xform_jax constants/grids on an ``oversample``-times finer grid.

    ``prepare_booz_xform_constants`` pins the angle-transform quadrature at
    ``2*(2*mboz+1)`` by ``2*(2*nboz+1)`` points, but the transform reads the
    counts back off ``constants`` for its Fourier normalization, so an integer
    refinement of the flattened ``(theta, zeta)`` grid is a drop-in reduction
    of the ``cos(m theta_B - n zeta_B)`` aliasing.
    """
    factor = int(oversample)
    if factor == 1:
        return constants, grids
    ntheta = factor * int(constants.ntheta)
    nzeta = factor * int(constants.nzeta) if int(constants.nzeta) > 1 else 1
    nu2_b = ntheta // 2 + 1
    # booz_xform's stellarator-symmetric grid spans theta in [0, pi] only
    # (nu2_b rows, boundary rows half-weighted and the normalization read
    # from nu2_b); the asymmetric grid spans the full circle (ntheta rows).
    nu3_b = ntheta if bool(constants.asym) else nu2_b
    theta = 2.0 * np.pi * np.arange(nu3_b) / ntheta
    zeta = 2.0 * np.pi * np.arange(nzeta) / (nzeta * int(nfp))
    return (
        dataclasses.replace(constants, ntheta=ntheta, nzeta=nzeta,
                            nu2_b=nu2_b),
        dataclasses.replace(grids,
                            theta_grid=jnp.asarray(np.repeat(theta, nzeta)),
                            zeta_grid=jnp.asarray(np.tile(zeta, nu3_b))),
    )


_BOOZER_GEOMETRY = ("rmnc_b", "rmns_b", "zmnc_b", "zmns_b", "pmnc_b", "pmns_b",
                    "gmnc_b", "gmns_b")


def _boozer_kernel_state(state, rt, *, rows, s_half, mboz, nboz, oversample,
                         geometry=False):
    """State tables through booz_xform_jax's validated full transform.

    One code path for both parities: symmetric states pass only the cosine
    families (the kernel returns an all-zero ``bmns_b`` block), ``lasym``
    states add the independent sine families.
    """
    from booz_xform_jax.jax_api import (
        booz_xform_jax_impl,
        prepare_booz_xform_constants,
    )

    lasym = bool(rt.setup.lasym)
    tables = [boozer_input_tables(state, rt, int(row)) for row in rows]
    stack = lambda name: jnp.stack([table[name] for table in tables])  # noqa: E731
    first = tables[0]
    xm, xn = np.asarray(first["xm"]), np.asarray(first["xn"])
    # An axisymmetric state (no toroidal input harmonics) has exactly no
    # n != 0 Boozer content -- the Boozer relabelling of an axisymmetric
    # field is axisymmetric -- so drop the toroidal band outright.  This
    # keeps the historical mode-list contract (tokamak decks return no
    # toroidal harmonics) and skips the dead quadrature.
    if not np.any(xn):
        nboz = 0
    # booz_xform's convenience wrapper prepares shape constants with Python
    # integer conversions. Do that work at trace time, then call its fully
    # jittable kernel so the objectives remain differentiable under JVP.
    with jax.ensure_compile_time_eval():
        constants, grids = prepare_booz_xform_constants(
            nfp=int(rt.resolution.nfp), mboz=int(mboz), nboz=int(nboz),
            asym=lasym, xm=xm, xn=xn, xm_nyq=xm, xn_nyq=xn)
        constants, grids = _refine_booz_grids(
            constants, grids, oversample, rt.resolution.nfp)
        xm_b = np.asarray(grids.xm_b, dtype=float)
        xn_b = np.asarray(grids.xn_b, dtype=float)
    magnetic_only = not geometry and (
        "magnetic_only" in inspect.signature(booz_xform_jax_impl).parameters)
    out = booz_xform_jax_impl(
        **({"magnetic_only": True} if magnetic_only else {}),
        rmnc=stack("rmnc"), zmns=stack("zmns"), lmns=stack("lmns"),
        bmnc=stack("bmnc"), bsubumnc=stack("bsubumnc"),
        bsubvmnc=stack("bsubvmnc"), iota=stack("iota"),
        xm=jnp.asarray(xm), xn=jnp.asarray(xn), xm_nyq=jnp.asarray(xm),
        xn_nyq=jnp.asarray(xn), constants=constants, grids=grids,
        **(dict(rmns=stack("rmns"), zmnc=stack("zmnc"), lmnc=stack("lmnc"),
                bmns=stack("bmns"), bsubumns=stack("bsubumns"),
                bsubvmns=stack("bsubvmns")) if lasym else {}),
    )
    setup = rt.setup
    extra = {key: out[key] for key in _BOOZER_GEOMETRY} if geometry else {}
    return extra | {
        "bmnc_b": out["bmnc_b"], "bmns_b": out["bmns_b"],
        "xm_b": xm_b, "xn_b": xn_b,
        "iota_b": stack("iota"), "G_b": stack("G"), "I_b": stack("I"),
        "nfp": int(rt.resolution.nfp),
        "s_b": jnp.asarray(s_half, dtype=jnp.asarray(setup.s_full).dtype)[rows - 1],
        "psi_b": jnp.asarray(setup.psi_half)[rows],
        "psi_edge": jnp.asarray(setup.psi_edge),
    }


def boozer_spectrum_high_order(
    state,
    *,
    surfaces,
    mboz: int = 16,
    nboz: int = 16,
    asym: bool = False,
    ntheta: int | None = None,
    nzeta: int | None = None,
) -> dict[str, Array]:
    """Boozer ``|B|`` spectrum of a continuous high-order state, traceable.

    The continuous-state counterpart of :func:`boozer_spectrum_state`.  Where
    that route reads the solver's discrete radial mesh and can only evaluate
    *half-mesh* surfaces, this one takes a
    :class:`~vmex.core.strong_force.HighOrderEquilibriumState` — the spline
    reconstruction produced by
    :func:`~vmex.core.strong_force.lift_high_order_state` or
    :func:`~vmex.core.strong_force.high_order_state_from_wout` — and evaluates
    *any* surface exactly, with no radial mesh, no interpolation, and no
    snapping.

    Per requested surface,
    :func:`~vmex.core.boozer_tables.high_order_boozer_input_tables` builds the
    wout-convention input tables at ``rho = sqrt(s)``: the geometry harmonics
    come straight from the spline coefficients, while ``|B|`` and the
    covariant field are Fourier-projected from the analytic field evaluator on
    a uniform ``(theta, zeta)`` grid.  ``booz_xform_jax``'s jittable kernel
    then performs the Boozer construction itself.  The whole chain is
    jit/grad-transparent — ``jax.grad`` through a returned ``bmnc_b`` entry
    back to the state's spline coefficients is exercised by the test suite.

    Parameters
    ----------
    state:
        Continuous reconstruction to transform.  Only its spectral tables,
        radial basis, ``m``/``n`` mode lists, ``nfp``, and ``phipf``/``chipf``
        are read.
    surfaces:
        Normalized toroidal flux values ``s = psi/psi_edge``, each in
        ``(0, 1]``; ``s = 0`` (the magnetic axis) and ``s > 1`` are rejected
        with :exc:`ValueError`.  Scalars are promoted to a one-element array.
        Values are used verbatim — unlike :func:`boozer_spectrum_state`, they
        are never snapped to a mesh.
    mboz, nboz:
        Poloidal and toroidal Boozer mode bounds.  They fix the output mode
        list ``(xm_b, xn_b)`` and, with it, booz_xform's angle-transform
        quadrature at ``2*(2*mboz+1)`` by ``2*(2*nboz+1)`` points.
    asym:
        Whether to run the transform's non-stellarator-symmetric branch.  The
        sine families are always handed to the kernel, but the kernel consumes
        them only when this is true; with the default ``False`` the returned
        ``bmns_b`` block is zero.  Set it for a state whose sine tables carry
        real content.
    ntheta, nzeta:
        Size of the uniform angular grid used to project ``|B|`` and the
        covariant field onto the Nyquist mode set — *not* the Boozer
        quadrature, which ``mboz``/``nboz`` set.  Both angles are in radians;
        ``theta`` spans ``[0, 2*pi)`` and ``zeta`` spans one field period.
        Defaults are ``max(12, 2*(max_m + 1))`` and ``max(8, 2*(max_n + 1))``
        from the state's own mode set; a grid too coarse to resolve that set
        raises.

    Returns
    -------
    A dict with ``bmnc_b`` and ``bmns_b`` of shape ``(nsurf, nmodes)`` in T,
    the mode lists ``xm_b``/``xn_b`` as ``numpy`` float arrays with ``xn_b``
    in physical units (already multiplied by ``nfp``), the per-surface
    ``iota_b``, the Boozer covariant coefficients ``G_b`` (``B_zeta``) and
    ``I_b`` (``B_theta``) in T m, the integer ``nfp``, and ``s_b`` — the
    requested ``surfaces`` themselves.  Note the difference from
    :func:`boozer_spectrum_state`: there is no ``psi_b``/``psi_edge`` here,
    and ``G_b``/``I_b`` are the transform's own ``bvco_b``/``buco_b`` rather
    than the input tables' values.

    ``bmnc_b``, ``bmns_b``, ``xm_b``, ``xn_b``, ``iota_b`` and ``nfp`` are
    exactly the inputs :func:`omnigenity_residual` expects.
    """

    from booz_xform_jax.jax_api import (
        booz_xform_jax_impl,
        prepare_booz_xform_constants,
    )

    surface_values = np.atleast_1d(np.asarray(surfaces, dtype=float))
    if np.any((surface_values <= 0.0) | (surface_values > 1.0)):
        raise ValueError("surfaces must satisfy 0 < s <= 1")
    from .strong_force import _carries_asymmetric_harmonics

    asym = bool(asym) or _carries_asymmetric_harmonics(state)
    tables = [
        high_order_boozer_input_tables(
            state,
            np.sqrt(surface),
            ntheta=ntheta,
            nzeta=nzeta,
        )
        for surface in surface_values
    ]
    first = tables[0]
    stack = lambda name: jnp.stack([table[name] for table in tables])  # noqa: E731
    constants, grids = prepare_booz_xform_constants(
        nfp=int(state.nfp),
        mboz=int(mboz),
        nboz=int(nboz),
        asym=asym,
        xm=first["xm"],
        xn=first["xn"],
        xm_nyq=first["xm_nyq"],
        xn_nyq=first["xn_nyq"],
    )
    magnetic_only = "magnetic_only" in inspect.signature(booz_xform_jax_impl).parameters
    out = booz_xform_jax_impl(
        **({"magnetic_only": True} if magnetic_only else {}),
        rmnc=stack("rmnc"),
        zmns=stack("zmns"),
        lmns=stack("lmns"),
        bmnc=stack("bmnc"),
        bsubumnc=stack("bsubumnc"),
        bsubvmnc=stack("bsubvmnc"),
        iota=stack("iota"),
        xm=jnp.asarray(first["xm"]),
        xn=jnp.asarray(first["xn"]),
        xm_nyq=jnp.asarray(first["xm_nyq"]),
        xn_nyq=jnp.asarray(first["xn_nyq"]),
        constants=constants,
        grids=grids,
        rmns=stack("rmns"),
        zmnc=stack("zmnc"),
        lmnc=stack("lmnc"),
        bmns=stack("bmns"),
        bsubumns=stack("bsubumns"),
        bsubvmns=stack("bsubvmns"),
    )
    return {
        "bmnc_b": out["bmnc_b"],
        "bmns_b": out["bmns_b"],
        "xm_b": np.asarray(grids.xm_b, dtype=float),
        "xn_b": np.asarray(grids.xn_b, dtype=float),
        "iota_b": stack("iota"),
        "G_b": out["bvco_b"],
        "I_b": out["buco_b"],
        "nfp": int(state.nfp),
        "s_b": jnp.asarray(surface_values),
    }


def boozer_spectrum_state(
    state: SpectralState,
    rt: SolverRuntime,
    *,
    surfaces,
    mboz: int = 16,
    nboz: int = 16,
    oversample: int = 2,
    geometry: bool = False,
) -> dict[str, Array]:
    """Boozer ``|B|`` spectrum of selected surfaces, fully traceable.

    The jnp analogue of :func:`vmex.core.optimize.boozer_modes_from_wout`
    evaluated without a wout file: vmex snaps the requested surfaces and
    builds the wout-convention single-surface spectral tables from the
    solver's internal state (:func:`vmex.core.boozer_tables
    .boozer_input_tables`), then ``booz_xform_jax``'s jittable kernel runs
    the Boozer construction itself — generating potential, Boozer angles,
    and the angle-transform quadrature — for symmetric and ``lasym`` states
    through the same code path.

    ``surfaces`` are normalized-flux values snapped to the nearest half-mesh
    surfaces (one Boozer construction per requested value, duplicates kept so
    outputs align with ``surfaces``).  ``oversample`` refines booz_xform's
    pinned ``(theta, zeta)`` quadrature grid by an integer factor before the
    angle transform, reducing the aliasing of ``cos(m theta_B - n zeta_B)``
    products.

    Returns ``{bmnc_b (nsurf, nmodes), bmns_b, xm_b, xn_b (physical),
    iota_b, G_b, I_b, nfp, s_b, psi_b, psi_edge}``; ``bmns_b`` is the
    all-zero cosine partner for symmetric states, and ``psi_b``/``psi_edge``
    are the signed toroidal flux divided by ``2*pi``.
    ``bmnc_b/xm_b/xn_b/iota_b/G_b/I_b/nfp`` are the inputs of
    :func:`omnigenity_residual` and of
    :func:`vmex.core.optimize.quasi_isodynamic_residual`.

    ``geometry=True`` also returns the geometry of the same transform in
    booz_xform's ``boozmn`` names: ``rmnc_b``, ``rmns_b``, ``zmnc_b``,
    ``zmns_b``, ``pmnc_b``, ``pmns_b`` (``phi - zeta_B``) and the Jacobian
    ``gmnc_b``, ``gmns_b``. Downstream codes call this instead of stacking
    :func:`~vmex.core.boozer_tables.boozer_input_tables` themselves.
    """
    setup = rt.setup
    if int(oversample) < 1:
        raise ValueError("oversample must be >= 1")
    s = jnp.asarray(setup.s_full)
    ns = int(s.shape[0])
    if ns < 3:
        raise ValueError(f"boozer_spectrum_state needs ns >= 3, got ns = {ns}")

    # -- surface selection: nearest half-mesh rows (static, shape-only) -----
    s_half_np, rows = _nearest_half_mesh_rows(ns, surfaces)
    return _boozer_kernel_state(
        state, rt, rows=rows, s_half=s_half_np, mboz=mboz, nboz=nboz,
        oversample=oversample, geometry=geometry)


# ---------------------------------------------------------------------------
# Omnigenity residual on Boozer |B| harmonics
# ---------------------------------------------------------------------------


def omnigenity_residual(
    *,
    bmnc_b,
    bmns_b=None,
    xm_b,
    xn_b,
    iota_b,
    nfp: int,
    weights: Iterable[float] | None = None,
    nphi: int = 97,
    nalpha: int = 25,
    n_levels: int = 16,
    softness: float = 2.0e-2,
    well_weight: float = 1.0,
    extremum_weight: float = 1.0,
    squash_weight: float = 1.0,
) -> dict[str, Array]:
    """Smooth constructed-QI-target omnigenity residual (module docstring).

    ``|B|`` is synthesized from Boozer harmonics along field lines
    ``theta_B = alpha + iota * phi_B`` on ``nalpha`` labels over one field
    period (``nphi`` periodic points), normalized per surface to ``[0, 1]``
    by the surface extrema.  Each field line is split at its minimum into
    two monotone branch envelopes (periodic running maxima); the residual
    stacks the three Goodman-construction distances:

    - ``well``: bounce distance ``delta(alpha, B*) = d_left + d_right``
      (sigmoid occupancy integrals of the branch envelopes at ``n_levels``
      trapping levels, in field-period fraction units) minus its
      ``alpha``-average — Cary–Shasharina bounce-distance omnigenity;
    - ``extremum``: per-line min/max of ``|B|`` minus their
      ``alpha``-averages — poloidally closed extremal contours;
    - ``squash``: pointwise ``envelope - |B|`` monotonicity defect —
      one magnetic well per field period.

    All three vanish on an exactly QI field.

    Parameters
    ----------
    bmnc_b:
        Cosine Boozer ``|B|`` harmonics in T, shape ``(nsurf, nmodes)``,
        indexed by the ``(m, n)`` pairs of ``xm_b``/``xn_b``.  Promoted to
        float64.
    bmns_b:
        The sine partner, same shape.  ``None`` (the default) means a
        stellarator-symmetric field and is treated as zeros.
    xm_b, xn_b:
        Boozer poloidal and toroidal mode numbers, shape ``(nmodes,)``.
        ``xn_b`` is in physical units — already multiplied by ``nfp``, as
        :func:`boozer_spectrum_state` and
        :func:`boozer_spectrum_high_order` return it.
    iota_b:
        Rotational transform on each surface, shape ``(nsurf,)``,
        dimensionless.  It sets the field-line slope
        ``theta_B = alpha + iota * phi_B``.
    nfp:
        Number of field periods; fixes the sampled ``phi_B`` interval to one
        period, ``[0, 2*pi/nfp)``.
    weights:
        One weight per surface, length ``nsurf``.  Its square root multiplies
        every residual entry of that surface, so a weight enters the objective
        linearly.  Default: all ones.
    nphi:
        Periodic samples of ``phi_B`` across the field period (radians), at
        least 8.
    nalpha:
        Field-line labels ``alpha``, uniform on ``[0, 2*pi)`` (radians), at
        least 2.  The residual's field-line averages are means over this axis.
    n_levels:
        Trapping levels ``B*`` at which the bounce distance is measured,
        placed strictly inside the normalized ``[0, 1]`` range, at least 2.
    softness:
        Width of the sigmoid that replaces the sharp ``|B| < B*`` occupancy,
        in *normalized* ``|B|`` units (each surface is rescaled to ``[0, 1]``
        by its own extrema, so this is a fraction of the surface's ``|B|``
        span).  Floored at machine epsilon.
    well_weight, extremum_weight, squash_weight:
        Multipliers on the three residual blocks.  Setting one to zero keeps
        the block's entries in the vector but zeroes them.

    Returns
    -------
    A dict whose ``residuals1d`` is the flat least-squares vector (the three
    blocks concatenated, each divided by the square root of its own entry
    count so the blocks stay comparable) and whose ``total`` is
    ``sum(residuals1d**2)``.  The remaining keys are diagnostics on the
    sampling grid: ``bhat`` ``(nsurf, nalpha, nphi)``, the normalized ``|B|``;
    ``delta`` ``(nsurf, nalpha, n_levels)``, the bounce distance in
    field-period fractions; ``line_min``/``line_max`` ``(nsurf, nalpha)``;
    and the grids ``levels`` ``(n_levels,)``, ``phi`` ``(nphi,)``,
    ``alpha`` ``(nalpha,)``.
    """
    bmnc_b = jnp.asarray(bmnc_b, dtype=jnp.float64)
    bmns_b = (jnp.zeros_like(bmnc_b) if bmns_b is None
              else jnp.asarray(bmns_b, dtype=bmnc_b.dtype))
    xm_b = jnp.asarray(np.asarray(xm_b, dtype=float))
    xn_b = jnp.asarray(np.asarray(xn_b, dtype=float))
    iota_b = jnp.atleast_1d(jnp.asarray(iota_b, dtype=jnp.float64))
    if bmnc_b.ndim != 2:
        raise ValueError(f"bmnc_b must have shape (nsurf, nmodes), got {bmnc_b.shape}")
    if bmns_b.shape != bmnc_b.shape:
        raise ValueError("bmns_b must have the same shape as bmnc_b")
    if nphi < 8 or nalpha < 2 or n_levels < 2:
        raise ValueError("omnigenity residual needs nphi >= 8, nalpha >= 2, n_levels >= 2")
    nsurf = int(bmnc_b.shape[0])
    dtype = bmnc_b.dtype
    w_arr = jnp.ones((nsurf,), dtype=dtype) if weights is None else _as_1d(weights)
    if int(w_arr.shape[0]) != nsurf:
        raise ValueError("weights must have the same length as surfaces")
    tiny = jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype)
    eps = jnp.maximum(jnp.asarray(float(softness), dtype=dtype),
                      jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype))

    # -- |B| along field lines over one field period (periodic phi grid) -----
    period = 2.0 * np.pi / float(nfp)
    phi = jnp.asarray(period * np.arange(nphi) / nphi, dtype=dtype)
    alpha = jnp.asarray(2.0 * np.pi * np.arange(nalpha) / nalpha, dtype=dtype)
    theta = alpha[None, :, None] + iota_b[:, None, None] * phi[None, None, :]
    angle = (theta[..., None] * xm_b - phi[None, None, :, None] * xn_b)
    b = (jnp.einsum("sapm,sm->sap", jnp.cos(angle), bmnc_b)
         + jnp.einsum("sapm,sm->sap", jnp.sin(angle), bmns_b))

    bmin = jnp.min(b, axis=(1, 2), keepdims=True)
    bmax = jnp.max(b, axis=(1, 2), keepdims=True)
    bhat = (b - bmin) / jnp.maximum(bmax - bmin, tiny)

    # -- monotone branch envelopes about the per-line minimum ----------------
    imin = jnp.argmin(bhat, axis=-1)                          # (nsurf, nalpha)
    nk = nphi // 2 + 1
    offs = jnp.arange(nk, dtype=jnp.int32)
    idx_l = jnp.mod(imin[:, :, None] - offs[None, None, :], nphi)
    idx_r = jnp.mod(imin[:, :, None] + offs[None, None, :], nphi)
    raw_l = jnp.take_along_axis(bhat, idx_l, axis=-1)         # (nsurf, nalpha, nk)
    raw_r = jnp.take_along_axis(bhat, idx_r, axis=-1)
    env_l = jax.lax.cummax(raw_l, axis=raw_l.ndim - 1)
    env_r = jax.lax.cummax(raw_r, axis=raw_r.ndim - 1)

    sqrt_w = jnp.sqrt(w_arr)[:, None, None]
    pieces: list[jnp.ndarray] = []

    # -- bounce-distance uniformity (Cary-Shasharina / Goodman "shuffle") ----
    levels = jnp.linspace(0.0, 1.0, int(n_levels) + 2, dtype=dtype)[1:-1]
    occ_l = jax.nn.sigmoid((levels[None, None, None, :] - env_l[..., None]) / eps)
    occ_r = jax.nn.sigmoid((levels[None, None, None, :] - env_r[..., None]) / eps)
    delta = (jnp.sum(occ_l, axis=2) + jnp.sum(occ_r, axis=2)) / float(nphi)
    well_res = (delta - jnp.mean(delta, axis=1, keepdims=True)) * sqrt_w * float(well_weight)
    pieces.append(jnp.ravel(well_res) / np.sqrt(float(nalpha * n_levels)))

    # -- extremum alignment (poloidally closed B_min / B_max contours) -------
    line_min = env_l[..., 0]                                  # = bhat at the minimum
    line_max = jnp.maximum(env_l[..., -1], env_r[..., -1])
    ext = jnp.stack([line_min, line_max], axis=-1)            # (nsurf, nalpha, 2)
    ext_res = (ext - jnp.mean(ext, axis=1, keepdims=True)) * sqrt_w * float(extremum_weight)
    pieces.append(jnp.ravel(ext_res) / np.sqrt(float(2 * nalpha)))

    # -- single-well monotonicity (Goodman "squash" distance) ----------------
    squash = jnp.concatenate([env_l - raw_l, (env_r - raw_r)[..., 1:]], axis=-1)
    squash_res = squash * sqrt_w * float(squash_weight)
    pieces.append(jnp.ravel(squash_res) / np.sqrt(float(nalpha * (2 * nk - 1))))

    residuals1d = jnp.concatenate(pieces)
    return {
        "residuals1d": residuals1d,
        "total": jnp.sum(residuals1d * residuals1d),
        "bhat": bhat,
        "delta": delta,
        "line_min": line_min,
        "line_max": line_max,
        "levels": levels,
        "phi": phi,
        "alpha": alpha,
    }


# ---------------------------------------------------------------------------
# Constructed-well J-invariant QI/max-J surrogate
# ---------------------------------------------------------------------------


def _soft_min_idx(values, beta: float = 50.0):
    values = jnp.asarray(values, dtype=jnp.float64)
    weights = jax.nn.softmax(-jnp.asarray(beta, dtype=values.dtype) * values)
    return jnp.sum(jnp.arange(values.shape[0], dtype=values.dtype) * weights)


def _cummin(values):
    values = jnp.asarray(values, dtype=jnp.float64)
    return jax.lax.associative_scan(jnp.minimum, values)


def _smooth_signed_sqrt(values, eps: float = 1.0e-9):
    values = jnp.asarray(values, dtype=jnp.float64)
    eps_arr = jnp.asarray(eps, dtype=values.dtype)
    abs_smooth = jnp.sqrt(values * values + eps_arr * eps_arr)
    return values / jnp.sqrt(abs_smooth + eps_arr)


def _smooth_positive_sqrt(values, eps: float = 1.0e-10):
    values = jnp.asarray(values, dtype=jnp.float64)
    eps_arr = jnp.asarray(eps, dtype=values.dtype)
    positive = jnp.maximum(values, 0.0)
    # Finite derivative at zero, while staying close to sqrt(max(values, 0)).
    return jnp.sqrt(positive + eps_arr) - jnp.sqrt(eps_arr)


def _smooth_abs_power(values, exponent: float, eps: float = 1.0e-12):
    """Differentiable surrogate for ``|x|**p`` that stays finite at zero."""

    values = jnp.asarray(values, dtype=jnp.float64)
    eps_arr = jnp.asarray(eps, dtype=values.dtype)
    return jnp.power(values * values + eps_arr * eps_arr, 0.5 * float(exponent))


def _periodic_angle_distance(alpha_a, alpha_b):
    """Shortest periodic distance on the 2π circle."""

    two_pi = jnp.asarray(2.0 * jnp.pi, dtype=jnp.result_type(alpha_a, alpha_b))
    delta = jnp.abs(alpha_a - alpha_b)
    return jnp.minimum(delta, two_pi - delta)


def _apply_smooth_goodman_transform(b_line, phi_coords):
    """Smooth squash/stretch surrogate of the Goodman constructed-QI well."""

    b_line = jnp.asarray(b_line, dtype=jnp.float64)
    phi_coords = jnp.asarray(phi_coords, dtype=jnp.float64)
    n = int(b_line.shape[0])
    indices = jnp.arange(n, dtype=b_line.dtype)
    s_indmin = _soft_min_idx(b_line)
    split_beta = jnp.asarray(10.0, dtype=b_line.dtype)
    peak_beta = jnp.asarray(120.0, dtype=b_line.dtype)
    cap_beta = jnp.asarray(35.0, dtype=b_line.dtype)
    mask_l = jax.nn.sigmoid(split_beta * (s_indmin - indices))
    mask_r = 1.0 - mask_l

    # Left branch: mirror the reference logic by identifying the LHS peak,
    # flattening before it, then squashing toward the minimum.
    lhs_gate = jax.nn.sigmoid(split_beta * (s_indmin - indices))
    lhs_penalty = jnp.asarray(10.0, dtype=b_line.dtype) * (1.0 - lhs_gate)
    lhs_weights = jax.nn.softmax(peak_beta * (b_line - lhs_penalty))
    lhs_peak_val = jnp.sum(lhs_weights * b_line)
    lhs_peak_idx = jnp.sum(lhs_weights * indices)
    before_peak = jax.nn.sigmoid(cap_beta * (lhs_peak_idx - indices))
    bl_base = before_peak * lhs_peak_val + (1.0 - before_peak) * b_line
    bl_sq = _cummin(bl_base)

    # Right branch: build a smoother analogue of the reference logic:
    # isolate the RHS of the well, locate its peak, flatten only after that
    # peak, then squash from right-to-left with a reverse cumulative minimum.
    rhs_gate = jax.nn.sigmoid(split_beta * (indices - s_indmin))
    rhs_penalty = jnp.asarray(10.0, dtype=b_line.dtype) * (1.0 - rhs_gate)
    rhs_weights = jax.nn.softmax(peak_beta * (b_line - rhs_penalty))
    rhs_peak_val = jnp.sum(rhs_weights * b_line)
    rhs_peak_idx = jnp.sum(rhs_weights * indices)
    after_peak = jax.nn.sigmoid(cap_beta * (indices - rhs_peak_idx))
    br_base = after_peak * rhs_peak_val + (1.0 - after_peak) * b_line
    br_sq = jnp.flip(_cummin(jnp.flip(br_base)))

    pmax = jnp.asarray(50.0, dtype=b_line.dtype)
    pmin = jnp.asarray(15.0, dtype=b_line.dtype)
    b_min_val = jnp.interp(s_indmin, indices, b_line)
    phi_mid = jnp.interp(s_indmin, indices, phi_coords)
    phi_start = phi_coords[0]
    phi_end = phi_coords[-1]
    x1_l = (phi_coords - phi_start) / (phi_mid - phi_start + 1.0e-10)
    x1_r = (phi_coords - phi_mid) / (phi_end - phi_mid + 1.0e-10)
    shape_l = (jnp.cos(2.0 * jnp.pi * x1_l) + 1.0) / 2.0
    shape_r = (jnp.cos(2.0 * jnp.pi * x1_r) + 1.0) / 2.0
    f_l = jnp.where(
        x1_l < 0.5,
        (1.0 - bl_sq) * (shape_l**pmax),
        (-b_min_val) * (shape_l**pmin),
    )
    f_r = jnp.where(
        x1_r < 0.5,
        (-b_min_val) * (shape_r**pmin),
        (1.0 - rhs_peak_val) * (shape_r**pmax),
    )
    out = mask_l * (bl_sq + f_l) + mask_r * (br_sq + f_r)
    out = jnp.clip(out, 0.0, 1.0)
    x_edge = (phi_coords - phi_start) / (phi_end - phi_start + 1.0e-10)
    edge_beta = jnp.asarray(120.0, dtype=b_line.dtype)
    edge_width = jnp.asarray(0.015, dtype=b_line.dtype)
    left_edge = jax.nn.sigmoid(edge_beta * (edge_width - x_edge))
    right_edge = jax.nn.sigmoid(edge_beta * (x_edge - (1.0 - edge_width)))
    edge_blend = left_edge + right_edge - left_edge * right_edge
    out = (1.0 - edge_blend) * out + edge_blend
    return out


def _branch_crossings(phi_coords, b_line, bj_level):
    """Smooth inverse-branch evaluation for ``B(phi)=Bj`` crossings."""

    phi_coords = jnp.asarray(phi_coords, dtype=jnp.float64)
    b_line = jnp.asarray(b_line, dtype=jnp.float64)
    bj_level = jnp.asarray(bj_level, dtype=jnp.float64)
    indices = jnp.arange(b_line.shape[0], dtype=jnp.float64)
    s_indmin = _soft_min_idx(b_line)
    branch_beta = jnp.asarray(10.0, dtype=b_line.dtype)
    left_mask = jax.nn.sigmoid(branch_beta * (s_indmin - indices))
    right_mask = 1.0 - left_mask

    def _invert_branch(phi_branch, b_branch, branch_mask):
        phi0 = phi_branch[:-1]
        phi1 = phi_branch[1:]
        b0 = b_branch[:-1]
        b1 = b_branch[1:]
        m0 = branch_mask[:-1]
        m1 = branch_mask[1:]
        db = b1 - b0
        scale = jnp.maximum(5.0e-3 * jnp.max(jnp.abs(b_branch)), 1.0e-6)

        # On both oriented branches, B should increase as we move away from the minimum.
        mono_gate = jax.nn.sigmoid(60.0 * db / scale)
        lo = jnp.minimum(b0, b1)
        hi = jnp.maximum(b0, b1)
        between_gate = jax.nn.sigmoid(40.0 * (bj_level - lo) / scale) * jax.nn.sigmoid(
            40.0 * (hi - bj_level) / scale
        )
        slope_gate = jnp.tanh(jnp.abs(db) / scale) ** 2
        branch_gate = m0 * m1
        valid = branch_gate * mono_gate * between_gate * slope_gate + 1.0e-14

        t_raw = (bj_level - b0) / (db + 1.0e-12)
        t = jnp.clip(t_raw, 0.0, 1.0)
        phi_seg = phi0 + t * (phi1 - phi0)
        return jnp.sum(valid * phi_seg) / jnp.sum(valid)

    phi_min = jnp.interp(s_indmin, indices, phi_coords)
    phi_lo = _invert_branch(jnp.flip(phi_coords), jnp.flip(b_line), jnp.flip(left_mask))
    phi_hi = _invert_branch(phi_coords, b_line, right_mask)
    # Match the normalized reference ``GetBranches(..., Bmax=1, Bmin=0)``.
    endpoint_beta = jnp.asarray(250.0, dtype=b_line.dtype)
    endpoint_width = jnp.asarray(0.01, dtype=b_line.dtype)
    low_blend = jax.nn.sigmoid(endpoint_beta * (endpoint_width - bj_level))
    high_blend = jax.nn.sigmoid(endpoint_beta * (bj_level - (1.0 - endpoint_width)))
    phi_lo = (1.0 - low_blend) * phi_lo + low_blend * phi_min
    phi_hi = (1.0 - low_blend) * phi_hi + low_blend * phi_min
    phi_lo = (1.0 - high_blend) * phi_lo + high_blend * phi_coords[0]
    phi_hi = (1.0 - high_blend) * phi_hi + high_blend * phi_coords[-1]
    return phi_lo, phi_hi


def _compute_j_pair(phi_coords, b_input, b_target, bj_levels, gi_value, *, nphi_int: int = 128):
    """Return ``(J_input, J_constructed)`` on the requested bounce levels."""

    b_input = jnp.asarray(b_input, dtype=jnp.float64)
    b_target = jnp.asarray(b_target, dtype=jnp.float64)
    phi_coords = jnp.asarray(phi_coords, dtype=jnp.float64)
    bj_levels = jnp.asarray(bj_levels, dtype=jnp.float64)
    gi_value = jnp.asarray(gi_value, dtype=jnp.float64)

    bmin = jnp.min(b_target)
    bmax = jnp.max(b_target)
    scale = jnp.maximum(bmax - bmin, 1.0e-12)
    b_target_norm = (b_target - bmin) / scale
    bj_norm = (bj_levels - bmin) / scale

    p1, p2 = jax.vmap(lambda bj: _branch_crossings(phi_coords, b_target_norm, bj))(bj_norm)

    t = jnp.linspace(0.0, 1.0, int(nphi_int), dtype=b_target.dtype)
    phi_grid = p1[:, None] + t[None, :] * (p2 - p1)[:, None]
    bi_g = jnp.interp(phi_grid, phi_coords, b_input)
    bc_g = jnp.interp(phi_grid, phi_coords, b_target)
    bi_safe = jnp.maximum(bi_g, 1.0e-9)
    bj_v = jnp.maximum(bj_levels[:, None], 1.0e-9)
    metric_factor = gi_value / bi_safe

    res_i = 1.0 - bi_g / bj_v
    res_c = 1.0 - bc_g / bj_v
    vi_g = _smooth_signed_sqrt(res_i)
    vc_g = _smooth_positive_sqrt(res_c)

    ji = jnp.trapezoid(vi_g * metric_factor, x=phi_grid, axis=1)
    jc = jnp.trapezoid(vc_g * metric_factor, x=phi_grid, axis=1)
    return ji, jc


def _synthesize_boozer_field_lines(
    *,
    bmnc_b,
    xm_b,
    xn_b,
    iota_b,
    nfp: int,
    nphi: int,
    nalpha: int,
):
    """Synthesize ``|B|`` along Boozer field lines over one field period."""

    bmnc_b = jnp.asarray(bmnc_b, dtype=jnp.float64)
    xm_b = jnp.asarray(np.asarray(xm_b, dtype=float))
    xn_b = jnp.asarray(np.asarray(xn_b, dtype=float))
    iota_b = jnp.atleast_1d(jnp.asarray(iota_b, dtype=jnp.float64))
    dtype = bmnc_b.dtype

    period = 2.0 * np.pi / float(nfp)
    phi = jnp.linspace(0.0, period, int(nphi), endpoint=True, dtype=dtype)
    alpha = jnp.linspace(0.0, 2.0 * jnp.pi, int(nalpha), endpoint=False, dtype=dtype)
    theta = alpha[None, :, None] + iota_b[:, None, None] * phi[None, None, :]
    angle = theta[..., None] * xm_b - phi[None, None, :, None] * xn_b
    b = jnp.einsum("sapm,sm->sap", jnp.cos(angle), bmnc_b)
    return phi, alpha, b


def j_invariant_qi_maxj_residual_from_boozer(
    *,
    bmnc_b,
    xm_b,
    xn_b,
    iota_b,
    gi_b,
    s_b,
    nfp: int,
    weights: Iterable[float] | None = None,
    nphi: int = 101,
    nalpha: int = 51,
    n_bounce: int = 66,
    p_j: float = 1.0,
    p_lambda: float = 1.0,
    nphi_int: int = 128,
    target_maxj: float = -0.06,
    qi_weight: float = 1.0,
    maxj_weight: float = 1.0,
    include_qi: bool = True,
    include_maxj: bool = True,
    maxj_pairing: str = "same_alpha",
    maxj_sigma_alpha: float | None = None,
) -> dict[str, Array]:
    """Shared-J QI/max-J residual blocks from precomputed Boozer spectra."""

    bmnc_b = jnp.asarray(bmnc_b, dtype=jnp.float64)
    iota_b = jnp.asarray(iota_b, dtype=jnp.float64)
    gi_b = jnp.asarray(gi_b, dtype=jnp.float64)
    s_b = jnp.asarray(s_b, dtype=jnp.float64)
    nsurf = int(bmnc_b.shape[0])
    if nsurf == 0:
        raise ValueError("Boozer surfaces must be non-empty")
    if int(nphi) < 8 or int(nalpha) < 2 or int(n_bounce) < 2:
        raise ValueError("shared-J QI/max-J residual needs nphi >= 8, nalpha >= 2, n_bounce >= 2")

    w_arr = jnp.ones((nsurf,), dtype=jnp.float64) if weights is None else _as_1d(weights)
    if int(w_arr.shape[0]) != nsurf:
        raise ValueError("weights must have the same length as the Boozer surfaces")

    phi, alpha, b_lines = _synthesize_boozer_field_lines(
        bmnc_b=bmnc_b,
        xm_b=xm_b,
        xn_b=xn_b,
        iota_b=iota_b,
        nfp=int(nfp),
        nphi=int(nphi),
        nalpha=int(nalpha),
    )
    bj_norm = jnp.power(
        jnp.arange(int(n_bounce), dtype=jnp.float64) / jnp.maximum(int(n_bounce) - 1, 1),
        float(p_lambda),
    )
    if str(maxj_pairing) not in {"all_to_all", "same_alpha", "soft_local"}:
        raise ValueError("maxj_pairing must be 'all_to_all', 'same_alpha', or 'soft_local'")
    if maxj_sigma_alpha is None:
        maxj_sigma_alpha = 2.0 * np.pi / max(int(nalpha), 1)
    sigma_alpha = max(float(maxj_sigma_alpha), 1.0e-8)

    def _per_surface(b_surface, gi_surface):
        def _per_line(b_line):
            bmin = jnp.min(b_line)
            bmax = jnp.max(b_line)
            scale = jnp.maximum(bmax - bmin, 1.0e-10)
            b_norm = (b_line - bmin) / scale
            b_target_norm = _apply_smooth_goodman_transform(b_norm, phi)
            b_target = b_target_norm * scale + bmin
            bj_phys = bj_norm * scale + bmin
            return _compute_j_pair(phi, b_line, b_target, bj_phys, gi_surface, nphi_int=int(nphi_int))

        ji_all, jc_all = jax.vmap(_per_line)(b_surface)
        return ji_all, jc_all

    ji_all, jc_all = jax.vmap(_per_surface, in_axes=(0, 0))(b_lines, gi_b)

    residual_blocks: list[jnp.ndarray] = []
    diagnostics: dict[str, Array] = {
        "phi": phi,
        "alpha": alpha,
        "surfaces": s_b,
        "ji": ji_all,
        "jc": jc_all,
    }

    if bool(include_qi):
        ji_pow = jnp.power(ji_all, float(p_j))
        jc_pow = jnp.power(jc_all, float(p_j))
        nalpha_f = jnp.asarray(float(nalpha), dtype=jnp.float64)
        sum_ji = jnp.sum(ji_pow, axis=1)
        sum_jc = jnp.sum(jc_pow, axis=1)
        sum_ji_sq = jnp.sum(ji_pow * ji_pow, axis=1)
        sum_jc_sq = jnp.sum(jc_pow * jc_pow, axis=1)
        qi_pair_sum = nalpha_f * (sum_ji_sq + sum_jc_sq) - 2.0 * sum_ji * sum_jc
        qi_num = jnp.sum(qi_pair_sum, axis=1)
        mean_denom = ((jnp.sum(sum_ji + sum_jc, axis=1) / (2.0 * float(n_bounce))) ** 2) + 1.0e-10
        qi_surface = jnp.sqrt(jnp.maximum(qi_num, 0.0) / mean_denom)
        qi_block = float(qi_weight) * jnp.sqrt(w_arr) * qi_surface
        residual_blocks.append(qi_block)
        diagnostics["qi_surface"] = qi_surface
        diagnostics["qi_objective"] = jnp.sum(qi_block * qi_block)
        diagnostics["ji_pow"] = ji_pow
        diagnostics["jc_pow"] = jc_pow
    else:
        diagnostics["qi_surface"] = jnp.zeros((nsurf,), dtype=jnp.float64)
        diagnostics["qi_objective"] = jnp.asarray(0.0, dtype=jnp.float64)
        diagnostics["ji_pow"] = jnp.zeros_like(ji_all)
        diagnostics["jc_pow"] = jnp.zeros_like(jc_all)

    if bool(include_maxj):
        if nsurf < 2:
            maxj_block = jnp.zeros((0,), dtype=jnp.float64)
            maxj_surface = jnp.zeros((0,), dtype=jnp.float64)
            jc_pow_maxj = jnp.zeros_like(jc_all)
            slope = jnp.zeros((0, 0, 0), dtype=jnp.float64)
            maxj_pair_weights = jnp.zeros((0, 0), dtype=jnp.float64)
        else:
            jc_pow_maxj = _smooth_abs_power(jc_all, float(p_j))
            ds = s_b[1:] - s_b[:-1]
            ds = jnp.where(jnp.abs(ds) > 0.0, ds, 1.0e-10)
            # jc_* shape is (surface, alpha, bounce). max-J compares adjacent
            # surfaces at fixed bounce level, with selectable lower-surface alpha pairing.
            jc_lo = jc_pow_maxj[:-1, :, 1:]
            jc_hi = jc_pow_maxj[1:, :, 1:]
            alpha_arr = jnp.asarray(alpha, dtype=jnp.float64)
            if str(maxj_pairing) == "all_to_all":
                alpha_weights = jnp.ones((int(nalpha), int(nalpha)), dtype=jnp.float64) / float(nalpha)
            elif str(maxj_pairing) == "same_alpha":
                alpha_weights = jnp.eye(int(nalpha), dtype=jnp.float64)
            else:
                alpha_dist = _periodic_angle_distance(alpha_arr[:, None], alpha_arr[None, :])
                alpha_weights = jax.nn.softmax(
                    -(alpha_dist * alpha_dist) / (2.0 * sigma_alpha * sigma_alpha), axis=1
                )

            def _surface_pair_slope(hi_surface, lo_surface, ds_surface):
                if str(maxj_pairing) == "soft_local":
                    hi_pairs = hi_surface[:, :, None]
                    lo_pairs = jnp.swapaxes(lo_surface, 0, 1)[None, :, :]
                    pair_slope = (hi_pairs - lo_pairs) / (
                        ds_surface * (0.5 * (hi_pairs + lo_pairs) + 1.0e-10)
                    )
                    pair_violation = jnp.maximum(0.0, pair_slope - float(target_maxj))
                    weighted_violation_sq = jnp.einsum(
                        "abl,al->ab", pair_violation * pair_violation, alpha_weights
                    )
                    slope_local = jnp.einsum("abl,al->ab", pair_slope, alpha_weights)
                    return slope_local, jnp.sqrt(weighted_violation_sq)

                lo_matched = alpha_weights @ lo_surface
                slope_local = (hi_surface - lo_matched) / (
                    ds_surface * (0.5 * (hi_surface + lo_matched) + 1.0e-10)
                )
                violation_local = jnp.maximum(0.0, slope_local - float(target_maxj))
                return slope_local, violation_local

            slope, violation = jax.vmap(_surface_pair_slope, in_axes=(0, 0, 0))(jc_hi, jc_lo, ds)
            maxj_surface = jnp.sqrt(jnp.sum(violation**2, axis=(1, 2)))
            maxj_block = float(maxj_weight) * jnp.sqrt(0.5 * (w_arr[:-1] + w_arr[1:])) * maxj_surface
            maxj_pair_weights = alpha_weights
        residual_blocks.append(maxj_block)
        diagnostics["maxj_surface"] = maxj_surface
        diagnostics["maxj_objective"] = jnp.sum(maxj_block * maxj_block)
        diagnostics["jc_pow_maxj"] = jc_pow_maxj
        diagnostics["maxj_slope"] = slope
        diagnostics["maxj_pair_weights"] = maxj_pair_weights
    else:
        diagnostics["maxj_surface"] = jnp.zeros((max(nsurf - 1, 0),), dtype=jnp.float64)
        diagnostics["maxj_objective"] = jnp.asarray(0.0, dtype=jnp.float64)
        diagnostics["jc_pow_maxj"] = jnp.zeros_like(jc_all)
        diagnostics["maxj_slope"] = jnp.zeros((max(nsurf - 1, 0), 0, 0), dtype=jnp.float64)
        diagnostics["maxj_pair_weights"] = jnp.zeros((0, 0), dtype=jnp.float64)

    if not residual_blocks:
        raise ValueError("At least one of include_qi/include_maxj must be True.")

    residuals1d = jnp.concatenate(residual_blocks)
    diagnostics["residual_block_sizes"] = jnp.asarray([block.size for block in residual_blocks], dtype=jnp.int32)
    return {
        "residuals1d": residuals1d,
        "total": jnp.sum(residuals1d * residuals1d),
        **diagnostics,
    }


# ---------------------------------------------------------------------------
# Objective-term wrapper (QuasisymmetryRatioResidual-style interface)
# ---------------------------------------------------------------------------


class _ResidualTerm:
    """Least-squares objective-term contract shared by the residual classes.

    A subclass defines ``compute_state(state, rt)`` returning a dict with
    ``"residuals1d"`` and ``"total"``, or overrides :meth:`residuals_state`
    and :meth:`total_state`.  Boozer-spectrum terms set ``surfaces``,
    ``mboz``, ``nboz`` and ``oversample`` and transform with ``_boozer``.
    Used by :class:`QIResidual`, the maximum-J terms and ``GammaC``.
    """

    def _boozer(self, state, rt):
        return boozer_spectrum_state(state, rt, surfaces=self.surfaces, mboz=self.mboz,
                                     nboz=self.nboz, oversample=self.oversample)

    def residuals_state(self, state, rt):
        """Return the 1-D residual rows of ``compute_state``."""
        return self.compute_state(state, rt)["residuals1d"]

    def total_state(self, state, rt):
        """Return the scalar ``total`` of ``compute_state``."""
        return self.compute_state(state, rt)["total"]

    def J(self, eq):
        """Residual rows of a converged :class:`~vmex.core.optimize.Equilibrium`.

        Also reachable as ``__call__`` and ``residuals``.
        """
        return self.residuals_state(eq.state, eq.runtime)

    __call__ = J
    residuals = J

    def total(self, eq):
        """Scalar cost of a converged :class:`~vmex.core.optimize.Equilibrium`."""
        return self.total_state(eq.state, eq.runtime)


class QIResidual(_ResidualTerm):
    """Traceable smooth quasi-isodynamic surrogate (module docstring).

    Composition of :func:`boozer_spectrum_state` (traceable Boozer ``|B|``
    spectrum on the requested surfaces) and :func:`omnigenity_residual`
    (a smooth level-set surrogate for the Goodman construction).  The
    interface mirrors
    :class:`~vmex.core.optimize.QuasisymmetryRatioResidual`: the instance
    is a :func:`~vmex.core.optimize.least_squares` objective term for
    both gradient modes — ``jac=None`` calls :meth:`J` on the converged
    :class:`~vmex.core.optimize.Equilibrium`, ``jac="implicit"`` picks up
    the traceable :meth:`residuals_state` vector (full pointwise
    Gauss-Newton geometry, exact implicit gradients).  Use
    :class:`vmex.core.qi.ConstructedQIResidual` when the optimized quantity
    must be the full squash-and-shuffle distance.

    Parameters
    ----------
    surfaces:
        Normalized toroidal flux values ``s = psi/psi_edge`` to evaluate.
        Each is snapped to the nearest half-mesh surface by
        :func:`boozer_spectrum_state`; duplicates are kept.
    weights:
        One weight per surface (same length as ``surfaces``), or ``None`` for
        uniform weighting.  Validated at construction.
    mboz, nboz, oversample:
        Boozer resolution passed straight to :func:`boozer_spectrum_state`:
        the poloidal and toroidal mode bounds, and the integer refinement of
        booz_xform's pinned angle-transform quadrature.
    nphi, nalpha, n_levels, softness:
        Field-line sampling of :func:`omnigenity_residual` — points per field
        period, field-line labels, trapping levels, and the sigmoid width in
        normalized ``|B|`` units.
    well_weight, extremum_weight, squash_weight:
        Multipliers on the bounce-distance, extremum-alignment, and
        single-well blocks of that residual.

    Example::

        qi = QIResidual(np.linspace(0.25, 0.9, 4))
        result = least_squares([(qi, 0.0, 10.0), ...], inp, jac="implicit")
    """

    name = "qi"

    def __init__(
        self,
        surfaces,
        *,
        weights: Iterable[float] | None = None,
        mboz: int = 16,
        nboz: int = 16,
        oversample: int = 2,
        nphi: int = 97,
        nalpha: int = 25,
        n_levels: int = 16,
        softness: float = 2.0e-2,
        well_weight: float = 1.0,
        extremum_weight: float = 1.0,
        squash_weight: float = 1.0,
    ):
        self.surfaces = np.atleast_1d(np.asarray(surfaces, dtype=float))
        self.weights = None if weights is None else np.asarray(list(weights), dtype=float)
        if self.weights is not None and self.weights.shape[0] != self.surfaces.shape[0]:
            raise ValueError("weights must have the same length as surfaces")
        self.mboz = int(mboz)
        self.nboz = int(nboz)
        self.oversample = int(oversample)
        self.nphi = int(nphi)
        self.nalpha = int(nalpha)
        self.n_levels = int(n_levels)
        self.softness = float(softness)
        self.well_weight = float(well_weight)
        self.extremum_weight = float(extremum_weight)
        self.squash_weight = float(squash_weight)

    # -- traceable (state, runtime) evaluation --------------------------------

    def compute_state(self, state: SpectralState, rt: SolverRuntime) -> dict[str, Array]:
        """Full diagnostics dict (Boozer spectrum + residual pieces)."""
        booz = self._boozer(state, rt)
        out = omnigenity_residual(
            bmnc_b=booz["bmnc_b"], bmns_b=booz.get("bmns_b"),
            xm_b=booz["xm_b"], xn_b=booz["xn_b"],
            iota_b=booz["iota_b"], nfp=booz["nfp"], weights=self.weights,
            nphi=self.nphi, nalpha=self.nalpha, n_levels=self.n_levels,
            softness=self.softness, well_weight=self.well_weight,
            extremum_weight=self.extremum_weight, squash_weight=self.squash_weight)
        out.update(booz)
        return out
