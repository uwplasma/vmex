"""Differentiable fast-ion confinement proxies ``Gamma_c`` and ``Gamma_alpha``.

``Gamma_c`` measures how far the contours of the second adiabatic invariant
``J`` deviate from flux surfaces: trapped particles whose bounce-averaged
drift has a radial component ride superbanana orbits out of the device.
The proxy of Nemov, Kasilov, Kernbichler, Leitold, Phys. Plasmas 15, 052501
(2008), equation 61, as organized by Velasco et al., Nucl. Fusion 61,
116059 (2021), equation 16, is

.. math::

   \\gamma_c = \\frac{2}{\\pi}\\arctan\\frac{v_r}{v_p},
   \\qquad
   \\Gamma_c = \\frac{\\pi}{4\\sqrt 2}\\left\\langle
     \\int_{1/B_{\\max}}^{1/B}\\!\\mathrm d\\lambda\\,
     \\frac{B}{\\sqrt{1-\\lambda B}}\\,\\gamma_c^2\\right\\rangle
   = \\frac{\\pi}{8\\sqrt 2}\\,
     \\frac{\\int\\mathrm d\\lambda\\sum_w (v\\tau_b\\,\\gamma_c^2)_w}
          {\\int \\mathrm dl/B},

with :math:`v_r, v_p` the bounce-averaged radial and poloidal projections of
the magnetic drift, ``w`` the wells along field lines and
:math:`\\langle\\cdot\\rangle` the flux-surface average.  ``Gamma_alpha`` is the
fraction of trapped particles lost promptly according to Velasco et al. 2021,
section 4.2, equation 25 (below).  Neither is a loss law: Velasco et al.
find ``Gamma_c`` itself cannot predict prompt losses and validate
``Gamma_alpha`` as the predictor.

The drift ratio is Nemov's, in DESC's single-valued rewrite of his equations
21-22 (``desc.compute._fast_ion``, bounce kernel of Unalmis et al., J. Plasma
Phys. 92(3), 2026):

.. math::

   \\tan\\frac{\\pi\\gamma_c}{2} =
   \\frac{\\int\\mathrm dl\\, w\\,
          (\\mathbf B\\times\\nabla B)\\cdot\\nabla\\psi/B^3}
        {T\\int \\mathrm dl\\,\\big[w\\,\\partial_\\rho B|_{\\vartheta,\\phi}
         + \\sqrt{1-\\lambda B}\\,K\\big]/B},

with :math:`w = (1-\\lambda B/2)/\\sqrt{1-\\lambda B}`, PEST angles
:math:`(\\vartheta,\\phi)`, :math:`\\rho=\\sqrt{s}`,
:math:`K = \\iota_\\rho\\,(\\nabla\\psi\\times\\mathbf b)\\cdot\\nabla\\phi
- 2\\,\\partial_\\rho B|_{\\vartheta,\\phi}
+ B\\,\\partial_\\rho B^\\phi|_{\\vartheta,\\phi}/B^\\phi` and
:math:`T=|\\nabla\\rho|\\,\\Vert\\mathbf e_\\alpha\\Vert` Nemov's surface-tangency
factor at the field minimum of the well.  Field lines are the PEST lines of
:mod:`vmex.core.stability`; every geometric ingredient is an exact spectral
point evaluation with JAX AD.

Numerics.  Each field line is a cubic Hermite spline (fourth-order slopes) of
the sampled ingredients.  Its stationary points split every grid interval
into monotone pieces, so every bounce point at every pitch level is found
exactly (a safeguarded Newton solve whose last step carries the implicit
function derivative) and wells smaller than a grid cell are not missed.
The pitch integral is split at the critical levels of each line (the values
of its local extrema and of its end points): between them the well topology
is constant and the integrand is analytic except for the logarithmic
divergence at barrier tops, which a smootherstep map of the Gauss-Legendre
nodes removes.  The levels depend on the field, so the Leibniz terms of the
moving breakpoints are differentiated exactly.  The bounce integrals use the
sine map that cancels the inverse-square-root endpoint singularity.

Two modelling choices make the finite-line estimator continuous; both tend to
the exact quantity under refinement:

- ``completeness``: wells cut by the ends of the finite line enter the
  normalising bounce time but their ``gamma_c**2`` is borrowed from the
  complete wells of the same level, ``tau * <gamma^2>_complete`` with the
  complete-well weight floored at ``completeness`` of the total, so that the
  last complete well leaving the window does not make the estimate jump.
- ``tangency_width``: Nemov's ``T`` is a point value at the *lowest* point of
  a well.  Wells with several dips of nearly equal depth (every omnigenous
  or quasisymmetric field has them) make that point jump discontinuously
  between dips, and ``T`` varies by factors of several along a line, so the
  exact ``Gamma_c`` is genuinely discontinuous there.  ``T`` is replaced by
  its Boltzmann average ``exp(-(B-B_min)/(tangency_width (b-B_min)))`` over
  the well, which tends to Nemov's value as ``tangency_width -> 0`` (Laplace
  principle) and is smooth for positive width.

``Gamma_alpha`` (Velasco et al. 2021, eq. 25; ripple wells ignored, as there)

.. math::

   \\Gamma_\\alpha = \\frac12\\left\\langle\\int_{1/B_{\\max}}^{1/B}
     \\!\\mathrm d\\lambda\\,\\frac{B}{\\sqrt{1-\\lambda B}}\\;
     H\\big((\\alpha_{out}-\\alpha)\\,v_\\alpha\\big)\\,
     H\\big((\\alpha-\\alpha_{in})\\,v_\\alpha\\big)\\right\\rangle

counts the orbits that, precessing in the direction of
:math:`v_\\alpha=\\mathbf v_M\\cdot\\nabla\\alpha`, meet an outward superbanana
(:math:`\\gamma_c^*>\\gamma_{th}`) before an inward one
(:math:`\\gamma_c^*<-\\gamma_{th}`), with
:math:`\\gamma_c^*=\\frac2\\pi\\arctan(v_r/|v_p|)` and ``gamma_th = 0.2``.
It lies between zero and the trapped fraction.  Per field line and pitch
level the well data are reduced to a signed ``gamma_c*`` and a precession
direction, the first outward-before-inward event is found along the periodic
``alpha`` grid with smoothed (smoothstep) thresholds of half-width
``threshold_width``, and the pitch integral is a midpoint rule on a common
level grid.  Both limits (``threshold_width -> 0``, finer levels and lines)
give eq. 25.  The precession direction is the sign of the poloidal drift
integral above for a positive charge.

Everything is jnp: both proxies are traceable end to end.  The field-line
parameterization requires nonzero ``iota`` on the target surfaces.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

import jax
import jax.numpy as jnp

from .solver import SolverRuntime, SpectralState
from .omnigenity import _ResidualTerm
from .statephysics import _field_line_frame
from .stability import (
    _ballooning_context, _pest_lambda, _surface_closures, _surface_tables,
    _theta_vmec_from_pest,
)

Array = Any

__all__ = [
    "GammaAlpha",
    "GammaC",
    "gamma_alpha_from_fieldlines",
    "gamma_alpha_from_wout",
    "gamma_alpha_state",
    "gamma_c_from_fieldlines",
    "gamma_c_from_wout",
    "gamma_c_state",
]

_INF = jnp.inf
_FIELDS = ("bmag", "radial_drift", "radial_gradient", "drift_correction",
           "dl_dx", "tangency")
_GAMMA_TH = 0.2     # Velasco et al. 2021, section 4.1


# --------------------------------------------------------------------------
# Line kernel: spline, bounce points, wells
# --------------------------------------------------------------------------

def _hermite(f: Array) -> Array:
    """Cubic Hermite coefficients ``(4, ..., n-1)`` of uniform samples ``f``.

    Slopes are the fourth-order central differences (cubic extrapolation at
    the ends); ``t in [0, 1]`` spans one grid interval.
    """
    f0, f1, f2, f3 = (f[..., i] for i in range(4))
    g0, g1, g2, g3 = (f[..., -1 - i] for i in range(4))
    pad = jnp.concatenate([
        jnp.stack([10 * f0 - 20 * f1 + 15 * f2 - 4 * f3,
                   4 * f0 - 6 * f1 + 4 * f2 - f3], -1), f,
        jnp.stack([4 * g0 - 6 * g1 + 4 * g2 - g3,
                   10 * g0 - 20 * g1 + 15 * g2 - 4 * g3], -1)], -1)
    m = (pad[..., :-4] - 8 * pad[..., 1:-3] + 8 * pad[..., 3:-1] - pad[..., 4:]) / 12
    d, m0, m1 = f[..., 1:] - f[..., :-1], m[..., :-1], m[..., 1:]
    return jnp.stack([f[..., :-1], m0, 3 * d - 2 * m0 - m1, -2 * d + m0 + m1])


def _peval(c, t):
    return ((c[3] * t + c[2]) * t + c[1]) * t + c[0]


def _pder(c, t):
    return (3 * c[3] * t + 2 * c[2]) * t + c[1]


def _stationary(c):
    """Stationary points ``t`` of the cubics ``c`` in ``[0, 1]``: ``(2, N)`` and validity."""
    b, cc, d = c[1], c[2], c[3]
    disc = cc * cc - 3 * b * d
    q = -(cc + jnp.where(cc >= 0, 1.0, -1.0)
          * jnp.sqrt(jnp.where(disc > 0, disc, 1.0)))
    t = jnp.stack([jnp.where(d != 0, q / jnp.where(d != 0, 3 * d, 1.0), _INF),
                   jnp.where(q != 0, b / jnp.where(q != 0, q, 1.0), _INF)])
    return t, (disc > 0) & (t >= 0) & (t <= 1)


def _root(c, tl, th, bl, bh, level, iters=40):
    """Root of ``cubic(t) = level`` in the monotone piece ``[tl, th]``.

    Safeguarded Newton on stopped values; the last step is differentiable,
    which gives the implicit-function derivative at the converged root.
    """
    sg = jax.lax.stop_gradient
    c0, tl, th, bl, bh, b0 = map(sg, (c, tl, th, bl, bh, level))
    inc = bh > bl
    t = tl + (th - tl) * (b0 - bl) / jnp.where(bh != bl, bh - bl, 1.0)
    lo, hi = jnp.broadcast_to(tl, t.shape), jnp.broadcast_to(th, t.shape)

    def body(_, s):
        t, lo, hi = s
        f = _peval(c0, t) - b0
        left = (f > 0) == inc
        hi, lo = jnp.where(left, t, hi), jnp.where(left, lo, t)
        dd = _pder(c0, t)
        tn = t - f / jnp.where(dd != 0, dd, 1.0)
        return jnp.where((tn > lo) & (tn < hi), tn, 0.5 * (lo + hi)), lo, hi

    t = sg(jax.lax.fori_loop(0, iters, body, (t, lo, hi))[0])
    dd = _pder(c, t)
    return t - (_peval(c, t) - level) / jnp.where(dd != 0, dd, 1.0)


def _breakpoint_nodes(F: Array, length, K: int, Q: int):
    """Pitch levels ``b = 1/lambda`` and weights ``db/b**2`` of one line.

    ``K`` level slots hold the critical levels (the values of the local
    extrema of ``B`` and of its two end values) and, in the remaining slots,
    repeated bisections of the widest gap.  Between slots the well topology
    is constant, and the integrand is analytic except for the logarithmic
    divergence at barrier tops; the drift integrals nevertheless vary on
    scales much finer than the gap between critical levels (a bounce point
    sweeping a plateau of ``B``, a zero of the poloidal drift), which the
    bisections resolve.  Each gap gets ``Q`` Gauss-Legendre nodes on the
    smootherstep ``s(t) = t^3 (10 - 15 t + 6 t^2)``, which flattens the
    logarithmic singularity.  Returns the nodes, the weights and the
    overflow flag (more critical levels than slots).
    """
    C = _hermite(F[0])
    t, ok = _stationary(C)
    # envelope theorem: B' = 0 at the extremum, so t is held fixed (its own
    # derivative diverges at the birth or death of a pair)
    t = jax.lax.stop_gradient(jnp.where(ok, t, 0.0))
    values = jnp.where(ok, _peval(C, t), _INF).ravel()
    values = jnp.concatenate([values, F[0, :1], F[0, -1:]])
    count = jnp.sum(jnp.isfinite(values))
    top = jnp.max(jnp.where(jnp.isfinite(values), values, -_INF))
    edges = jnp.sort(values)[:K]
    edges = jnp.where(jnp.isfinite(edges), edges, top)

    def bisect(i, edges):
        # widths are compared on a grid of 1e-8 of the range, so rounding
        # noise cannot flip the choice between equal gaps of mirror lines
        j = jnp.argmax(jnp.round((edges[1:] - edges[:-1]) / (1e-8 * (top - edges[0]) + 1e-300)))
        slot = jnp.minimum(count + i, K - 1)       # a free (padded) slot
        mid = jnp.where(count + i < K, 0.5 * (edges[j] + edges[j + 1]), edges[slot])
        return jnp.sort(edges.at[slot].set(mid))

    edges = jax.lax.fori_loop(0, K, bisect, edges)
    x, w = np.polynomial.legendre.leggauss(Q)
    t, w = 0.5 * (x + 1), 0.5 * w
    lo, hi = edges[:-1, None], edges[1:, None]
    b = (lo + (hi - lo) * t**3 * (10 - 15 * t + 6 * t * t)).ravel()
    weight = ((hi - lo) * 30 * t**2 * (1 - t)**2 * w).ravel() / b**2
    return b, weight, count > K


def _line_wells(F: Array, length, b: Array, *, W: int, qorder: int,
                width: float):
    """Bounce wells of one field line at the pitch levels ``b = 1/lambda``.

    ``F`` stacks the six ingredients ``(6, n)`` (module ``_FIELDS`` order) on
    a uniform grid ``[0, length]``.  Returns per level and well slot ``W`` the
    dict ``vt`` (bounce time ``v tau``), ``bottom`` (position of the well bottom), ``N``, ``D`` (drift integrals;
    ``tan(pi gamma_c / 2) = N / D``), ``g`` (completeness weight of a well
    away from the ends, 0 for cut wells), ``mask`` and the overflow flag.
    """
    n = F.shape[-1]
    N = n - 1
    step = length / N
    C = _hermite(F)
    cB = C[:, 0]
    ts, ok = _stationary(cB)
    edges = jnp.concatenate([jnp.zeros((1, N)),
                             jnp.sort(jnp.where(ok, ts, 1.0), axis=0),
                             jnp.ones((1, N))])
    tl, th = edges[:-1], edges[1:]                      # monotone pieces (3, N)
    bl_, bh_ = _peval(cB[:, None], tl), _peval(cB[:, None], th)
    lev = b[:, None, None]
    down = (bl_ >= lev) & (bh_ < lev)
    up = (bl_ < lev) & (bh_ >= lev)
    c3 = jnp.broadcast_to(cB[:, None, :], (4, 3, N))[:, None]
    t = _root(c3, tl, th, bl_, bh_, lev)
    x = ((jnp.arange(N) + t) * step).reshape(b.shape[0], 3 * N)
    # the line ends bound the wells that start or finish inside the window
    down = jnp.concatenate(
        [jnp.where(F[0, 0] < b, 0.0, _INF)[:, None],
         jnp.where(down.reshape(x.shape), x, _INF)], 1)
    up = jnp.concatenate(
        [jnp.where(up.reshape(x.shape), x, _INF),
         jnp.where(F[0, -1] < b, length, _INF)[:, None]], 1)
    down, up = jnp.sort(down, -1), jnp.sort(up, -1)
    left = down[:, :W]
    shift = (up[:, 0] < down[:, 0]).astype(jnp.int32)
    index = jnp.arange(W) + shift[:, None]
    right = jnp.where(index < up.shape[1], jnp.take_along_axis(
        up, jnp.minimum(index, up.shape[1] - 1), 1), _INF)
    mask = jnp.isfinite(left) & jnp.isfinite(right) & (right > left)
    overflow = jnp.any(jnp.sum(jnp.isfinite(down), -1) > W)
    left, right = jnp.where(mask, left, 0.0), jnp.where(mask, right, 0.0)
    half = 0.5 * (right - left)
    nodes, weights = np.polynomial.legendre.leggauss(qorder)
    angle, weights = 0.5 * np.pi * nodes, 0.5 * np.pi * weights
    xq = 0.5 * (left + right)[..., None] + half[..., None] * np.sin(angle)
    i = jnp.clip(jnp.floor(xq / step).astype(jnp.int32), 0, N - 1)
    B, f_rad, f_grad, corr, dl, tang = _peval(C[:, :, i], xq / step - i)
    parallel = jnp.where(mask[..., None], 1.0 - B / b[:, None, None], 1.0)
    inside = parallel > 0
    speed = jnp.where(inside, jnp.sqrt(jnp.where(inside, parallel, 1.0)), 0.0)
    inverse = jnp.where(inside, 1.0 / jnp.where(inside, speed, 1.0), 0.0)
    dx = weights * half[..., None] * np.cos(angle)       # sine-map measure
    drift = (1.0 - 0.5 * B / b[:, None, None]) * inverse
    # tangency factor: Boltzmann average over the well bottom
    bottom = jnp.min(jnp.where(mask[..., None], B, _INF), -1, keepdims=True)
    bottom = jnp.where(jnp.isfinite(bottom), bottom, 0.0)
    boltz = jnp.exp(-(B - bottom) / (width * jnp.maximum(
        b[:, None, None] - bottom, 1e-12))) * dx
    norm = jnp.where(mask, jnp.sum(boltz, -1), 1.0)
    tangency = jnp.sum(boltz * tang, -1) / norm
    cut = jnp.minimum(left, length - right) / jnp.minimum(length / 2, jnp.pi)
    complete = jnp.where((left > 0) & (right < length), jnp.sin(
        0.5 * jnp.pi * jnp.clip(cut, 0.0, 1.0))**2, 0.0)
    return dict(
        vt=jnp.where(mask, 2.0 * jnp.sum(inverse * dl * dx, -1), 0.0),
        N=jnp.sum(f_rad * drift * dl * dx, -1),
        D=tangency * (jnp.sum(f_grad * drift * dl * dx, -1)
                      + jnp.sum(corr * speed * dl * dx, -1)),
        g=complete, mask=mask, overflow=overflow,
        bottom=jnp.sum(boltz * xq, -1) / norm)


def _drift_angle(w: dict) -> Array:
    """Signed ``gamma_c* = (2/pi) arctan(N / |D|)`` of every well (0 off mask)."""
    m = w["mask"]
    return jnp.where(m, (2 / jnp.pi) * jnp.arctan2(
        jnp.where(m, w["N"], 1.0), jnp.abs(jnp.where(m, w["D"], 1.0))), 0.0)


def _line_length(F: Array, length):
    return jnp.trapezoid(F[4] / F[0], jnp.linspace(0.0, length, F.shape[-1]))


def _complete_mean(w: dict, value: Array, completeness: float):
    """``(total v tau, completeness-floored mean of value)`` per level."""
    vt = w["vt"]
    total, den = jnp.sum(vt, -1), jnp.sum(w["g"] * vt, -1)
    scale = den + completeness * total
    return total, jnp.sum(w["g"] * vt * value, -1) / jnp.where(scale > 0, scale, 1.0)


def _gamma_c_line(F, length, *, K, Q, W, qorder, width, completeness):
    b, weight, overflow = _breakpoint_nodes(F, length, K, Q)
    w = _line_wells(F, length, b, W=W, qorder=qorder, width=width)
    total, mean = _complete_mean(w, _drift_angle(w) ** 2, completeness)
    return (jnp.sum(weight * total * mean), _line_length(F, length),
            overflow | w["overflow"])


def _alpha_line(F, alpha, *, length, iota, nfp, b, W, qorder, width):
    """Wells of the line of label ``alpha``: bounce time, ``gamma_c*``,
    precession direction, completeness weight and the label ``alpha_eff`` of
    the well on the field-period-reduced surface (``alpha + 2 pi iota k / nfp``
    for a well bottom in field period ``k``)."""
    w = _line_wells(F, length, b, W=W, qorder=qorder, width=width)
    m, d = w["mask"], jnp.where(w["mask"], w["D"], 1.0)
    phi = (w["bottom"] - 0.5 * length) / iota
    k = jnp.floor(phi * nfp / (2 * jnp.pi))
    return dict(
        w, gamma=_drift_angle(w), alpha=alpha + 2 * jnp.pi * iota * k / nfp,
        direction=jnp.where(m, d / jnp.sqrt(d * d + w["N"]**2), 0.0))


def _check_lines(bmag, ingredients):
    bmag = jnp.asarray(bmag)
    if bmag.ndim != 2 or int(bmag.shape[-1]) < 8:
        raise ValueError("bmag must have shape (nline, nx >= 8)")
    return jnp.stack([jnp.broadcast_to(jnp.asarray(ingredients[k], bmag.dtype),
                                       bmag.shape) for k in _FIELDS], 1)


def _caps(num_transit: int, max_levels, max_wells) -> tuple[int, int]:
    """Level and well slots per line; the defaults (48 and 8 per transit)
    hold the shaped nfp = 2-4 boundaries tried (up to ~30 extrema of ``B`` per
    transit) and leave the rest of the level slots to the bisections."""
    return (48 * int(num_transit) if max_levels is None else int(max_levels),
            8 * int(num_transit) if max_wells is None else int(max_wells))


def gamma_c_from_fieldlines(
    *, bmag, radial_drift, radial_gradient, drift_correction, tangency, dl_dx,
    length, num_transit: int = 4, pitch_nodes: int = 4,
    quadrature_order: int = 64, max_levels: int | None = None,
    max_wells: int | None = None, tangency_width: float = 0.05,
    completeness: float = 0.05,
) -> dict[str, Array]:
    """Nemov ``Gamma_c`` of one flux surface from sampled field lines.

    ``bmag`` is ``(nline, nx)`` on a uniform grid covering ``[0, length]`` in
    the line parameter ``x`` with arc-length element ``dl_dx``; the other
    arrays are the module-docstring ingredients: ``radial_drift`` is
    ``(B x grad B) . grad psi / B^3``, ``radial_gradient`` is
    ``dB/drho at fixed PEST angles / B``, ``drift_correction`` is ``K / B``,
    ``tangency`` is ``T``.  ``pitch_nodes`` is the Gauss-Legendre order
    between consecutive critical levels of a line; ``max_levels`` and
    ``max_wells`` (defaults scale with ``num_transit``) bound the pitch-level
    slots per line (critical levels plus bisections) and the wells per level.  Any overflow turns ``gamma_c``
    into NaN rather than a falsely low cost.  Also returns ``gamma_c_lines``,
    the per-line values, and the ``overflow`` flag.
    """
    F = _check_lines(bmag, dict(
        bmag=bmag, radial_drift=radial_drift, radial_gradient=radial_gradient,
        drift_correction=drift_correction, dl_dx=dl_dx, tangency=tangency))
    K, W = _caps(num_transit, max_levels, max_wells)
    numer, line, over = jax.lax.map(jax.checkpoint(functools.partial(
        _gamma_c_line, length=length, K=K, Q=int(pitch_nodes), W=W,
        qorder=int(quadrature_order), width=float(tangency_width),
        completeness=float(completeness))), F)
    per_line = jnp.pi / (8 * jnp.sqrt(2.0)) * numer / line
    gamma_c = jnp.pi / (8 * jnp.sqrt(2.0)) * jnp.mean(numer) / jnp.mean(line)
    return {"gamma_c": jnp.where(jnp.any(over), jnp.nan, gamma_c),
            "gamma_c_lines": per_line, "overflow": jnp.any(over)}


def _lost_fraction(gamma, direction, width):
    """Probability that precession meets an outward superbanana first.

    ``gamma``, ``direction`` are ``(nbin, nlevel)`` on the periodic ``alpha``
    grid; the thresholds are smoothsteps rising over ``gamma_th +- width``
    (exactly 0 and 1 outside, so a quiet field loses nothing); the two precession
    directions are mixed with weights ``(1 +- direction) / 2``.  Eq. 25 pairs
    an inward with an outward event, so the result is scaled by the
    probability that an inward event exists at all.
    """
    n = gamma.shape[0]
    def step(g):
        u = jnp.clip((g - _GAMMA_TH) / (2 * width) + 0.5, 0.0, 1.0)
        return u * u * (3.0 - 2.0 * u)

    out, inward = step(gamma), step(-gamma)
    free = jnp.clip(1.0 - out - inward, 0.0, 1.0)
    steps = jnp.arange(n)

    def sweep(sign):
        k = (steps[:, None] + sign * steps[None, :]) % n          # (j, m)
        alive = jnp.cumprod(free[k], axis=1)
        alive = jnp.concatenate([jnp.ones_like(alive[:, :1]), alive[:, :-1]], 1)
        return jnp.sum(out[k] * alive, axis=1)

    plus = 0.5 * (1.0 + direction)
    pair = 1.0 - jnp.prod(1.0 - inward, axis=0)
    return (plus * sweep(1) + (1.0 - plus) * sweep(-1)) * pair


def gamma_alpha_from_fieldlines(
    *, bmag, radial_drift, radial_gradient, drift_correction, tangency, dl_dx,
    length, iota, nfp: int = 1, num_transit: int = 4, pitch_levels: int = 64,
    quadrature_order: int = 64, max_wells: int | None = None,
    tangency_width: float = 0.05, completeness: float = 0.05,
    threshold_width: float = 0.05,
) -> dict[str, Array]:
    """Velasco ``Gamma_alpha`` (module docstring) of one flux surface.

    Same ingredients as :func:`gamma_c_from_fieldlines`; the lines are
    equispaced in the field-line label ``alpha`` over one period and carry
    ``iota`` and the number of field periods ``nfp``.  Every well is
    assigned the label ``alpha_eff`` of its bottom reduced into the first
    field period (the surface position that eq. 25 calls ``alpha``); the
    ``gamma_c*`` and precession direction of the complete wells are binned on
    a periodic ``alpha`` grid (hat weights), and each well takes the lost
    probability of its position.  The pitch integral is a midpoint rule with
    ``pitch_levels`` levels uniform in ``B`` between the extremes of the
    lines.  Also returns the trapped fraction ``f_trapped`` (eq. 24).
    """
    F = _check_lines(bmag, dict(
        bmag=bmag, radial_drift=radial_drift, radial_gradient=radial_gradient,
        drift_correction=drift_correction, dl_dx=dl_dx, tangency=tangency))
    n = F.shape[0]
    _, W = _caps(num_transit, None, max_wells)
    b_lo, b_hi = jnp.min(F[:, 0]), jnp.max(F[:, 0])
    b = b_lo + (b_hi - b_lo) * (jnp.arange(pitch_levels) + 0.5) / pitch_levels
    weight = (b_hi - b_lo) / pitch_levels / b**2
    labels = 2 * jnp.pi * jnp.arange(n) / n
    w = jax.lax.map(jax.checkpoint(lambda a: _alpha_line(
        a[0], a[1], length=length, iota=iota, nfp=nfp, b=b, W=W,
        qorder=int(quadrature_order), width=float(tangency_width))), (F, labels))
    # hat weights of every well on the periodic alpha grid: (line, level, well, bin)
    u = jnp.mod(w["alpha"] / (2 * jnp.pi) * n, n)[..., None] - jnp.arange(n)
    hat = jnp.maximum(0.0, 1.0 - jnp.abs((u + n / 2) % n - n / 2))
    vt = w["vt"]
    # complete-well bin maps; the completeness floor mirrors _complete_mean
    mass = jnp.einsum("lkwi,lkw->ki", hat, w["g"] * vt)
    floor = completeness * jnp.einsum("lkwi,lkw->ki", hat, vt)
    maps = [jnp.einsum("lkwi,lkw->ki", hat, w["g"] * vt * w[name]) / jnp.where(
        mass + floor > 0, mass + floor, 1.0) for name in ("gamma", "direction")]
    lost = _lost_fraction(maps[0].T, maps[1].T, float(threshold_width))  # (bin, level)
    well_lost = jnp.einsum("lkwi,ik->lkw", hat, lost)
    line = jnp.mean(jax.vmap(lambda f: _line_length(f, length))(F))
    scale = 0.25 / (n * line)
    over = jnp.any(w["overflow"])
    return {"gamma_alpha": jnp.where(over, jnp.nan, scale * jnp.sum(
                weight * jnp.sum(vt * well_lost, (0, 2)))),
            "f_trapped": scale * jnp.sum(weight * jnp.sum(vt, (0, 2))),
            "overflow": over}


def _make_gamma_point_fn(m: Array, xn: Array, tabs: dict, iota: Array,
                         diota: Array, phipf_j: Array):
    """Point-evaluation closure for one flux surface (Nemov drift set).

    Built on :func:`vmex.core.stability._surface_closures`, so the sine-parity
    spectra of an asymmetric state are carried without a second
    implementation.  Returns at ``q = (t, theta, phi)`` (evaluated at
    ``t = s - s_j = 0``) the tuple

    ``(|B|, B^phi, B x grad|B| . grad s, d|B|/ds|PEST, dB^phi/ds|PEST,
    (grad s x b) . grad phi, |grad s|, ||e_theta|| / (1 + lambda_theta))``

    where the PEST-fixed radial derivatives follow the moving VMEC angle
    ``dtheta/ds = -lambda_s / (1 + lambda_theta)`` and the last entry is
    ``||e_alpha||`` at fixed ``(rho, phi)``.
    """
    pos_fn, lam_fn, _, modb_fn, bsupphi_fn = _surface_closures(
        m, xn, tabs, iota, diota, phipf_j)

    def point(q: Array):
        f = _field_line_frame(pos_fn, lam_fn, modb_fn, q, iota=iota, diota=diota, phipf_j=phipf_j)
        dB, dual = f["dB"], f["dual"]                # dB at fixed VMEC angles
        dG = jax.grad(bsupphi_fn)(q)
        one_lam = 1.0 + f["lam_g"][1]
        dtheta_ds = -f["lam_g"][0] / one_lam          # fixed PEST angle, moving VMEC angle
        grad_s = dual[0]
        return (
            f["modB"],
            f["b_sup_phi"],
            jnp.cross(f["B"], f["grad_modB"]) @ grad_s,
            dB[0] + dB[1] * dtheta_ds,
            dG[0] + dG[1] * dtheta_ds,
            jnp.cross(grad_s, f["B"] / f["modB"]) @ dual[2],
            jnp.linalg.norm(grad_s),
            jnp.linalg.norm(f["J"][:, 1]) / one_lam,
        )

    return point


def _surface_rows(surfaces, ns: int) -> tuple[int, ...]:
    """Map normalized-flux values to distinct interior full-mesh rows."""
    values = np.atleast_1d(np.asarray(surfaces, dtype=float))
    if values.size == 0 or np.any(~np.isfinite(values)) or np.any(
            (values <= 0.0) | (values >= 1.0)):
        raise ValueError("surfaces must be finite and strictly inside (0, 1)")
    rows = tuple(
        min(max(int(round(value * (ns - 1))), 2), ns - 2) for value in values)
    if len(set(rows)) != len(rows):
        raise ValueError(
            f"surfaces resolve to duplicate radial rows at ns = {ns}")
    return rows


def _sampling_grids(dtype, nalpha: int, num_transit: int,
                    points_per_transit: int):
    """Field-line-label and along-line grids of one budget.

    The along-line grid is centred on the field-line label, not ``[0, L]``:
    the stellarator reflection maps ``(alpha, x) -> (-alpha, -x)`` and the
    alpha set is closed under negation, so a window symmetric in ``x`` makes
    the estimator exactly invariant under it -- and ``Gamma_c`` is even while
    the sine-parity spectra are odd, so ``d(Gamma_c)/d(sine)`` then vanishes
    on a symmetric state as it must.
    """
    length = 2.0 * np.pi * int(num_transit)
    alpha = jnp.asarray(
        2.0 * np.pi * np.arange(int(nalpha)) / int(nalpha), dtype=dtype)
    x = jnp.linspace(
        -0.5 * length, 0.5 * length,
        int(points_per_transit) * int(num_transit) + 1, dtype=dtype)
    return length, alpha, x


def _line_ingredients(ctx: dict, j: int, zeta0: Array, alpha: Array,
                      x: Array) -> dict[str, Array]:
    """Sampled Nemov drift-set arrays on one full-mesh row's field lines.

    Returns the keyword arguments of :func:`gamma_c_from_fieldlines` and
    :func:`gamma_alpha_from_fieldlines` (all ``(nline, nx)``) plus the row
    ``iota``, so both proxies share one geometry evaluation.
    """
    hs, psi_edge = ctx["hs"], ctx["psi_edge"]
    iota = 0.5 * (ctx["iotas"][j] + ctx["iotas"][j + 1])
    diota = (ctx["iotas"][j + 1] - ctx["iotas"][j]) / hs
    tabs = _surface_tables(ctx, j)
    point = _make_gamma_point_fn(
        ctx["m"], ctx["xn"], tabs, iota, diota, ctx["phipf"][j])
    lmns0, lmnc0 = _pest_lambda(tabs)

    def line(a):
        theta_star = a + x
        phi = zeta0 + x / iota
        theta_v = _theta_vmec_from_pest(
            theta_star, phi, lmns0, ctx["m"], ctx["xn"], lmnc0)
        q = jnp.stack([jnp.zeros_like(theta_v), theta_v, phi], axis=-1)
        return jax.vmap(point)(q)

    (modB, b_sup_phi, bxgb_gs, modb_s_pest, bsupphi_s_pest,
     gs_cross_b_gphi, grad_s_norm, e_alpha_norm) = jax.vmap(line)(alpha)

    two_rho = 2.0 * jnp.sqrt(ctx["s"][j])
    modb_r = two_rho * modb_s_pest                # d|B|/drho at fixed PEST angles
    correction = (
        two_rho * diota * psi_edge * gs_cross_b_gphi
        - 2.0 * modb_r + modB * (two_rho * bsupphi_s_pest) / b_sup_phi)
    return dict(
        bmag=modB,
        radial_drift=psi_edge * bxgb_gs / modB**3,
        radial_gradient=modb_r / modB,
        drift_correction=correction / modB,
        tangency=grad_s_norm / two_rho * e_alpha_norm,
        dl_dx=modB / jnp.abs(iota * b_sup_phi),
        iota=iota,
    )


_OPTIONS = {
    "c": dict(nalpha=16, num_transit=4, points_per_transit=32, pitch_nodes=4,
              quadrature_order=64, max_levels=None, max_wells=None,
              tangency_width=0.05, completeness=0.05),
    "alpha": dict(nalpha=16, num_transit=4, points_per_transit=32,
                  pitch_levels=64, quadrature_order=64, max_wells=None,
                  tangency_width=0.05, completeness=0.05, threshold_width=0.05),
}
_OUTPUTS = {"c": ("gamma_c",), "alpha": ("gamma_alpha", "f_trapped")}
_KERNELS = {"c": gamma_c_from_fieldlines, "alpha": gamma_alpha_from_fieldlines}
_BOUNDARY = ("nalpha", "num_transit", "points_per_transit")


def _resolve(kind: str, options: dict) -> tuple:
    """Defaults + overrides as a hashable static tuple, validated."""
    unknown = set(options) - set(_OPTIONS[kind])
    if unknown:
        raise TypeError(f"unknown option(s) {sorted(unknown)}; valid: {sorted(_OPTIONS[kind])}")
    cfg = {**_OPTIONS[kind], **options}
    if cfg["nalpha"] < 2 or (kind == "alpha" and cfg["nalpha"] < 4):
        raise ValueError("nalpha must be >= 2 (>= 4 for Gamma_alpha)")
    if cfg["num_transit"] < 1 or cfg["points_per_transit"] < 16:
        raise ValueError("num_transit must be >= 1 and points_per_transit >= 16")
    if min(cfg.get("pitch_nodes", 2), cfg.get("pitch_levels", 2), cfg["quadrature_order"]) < 2:
        raise ValueError("pitch_nodes, pitch_levels and quadrature_order must be >= 2")
    if not (cfg["tangency_width"] > 0 and cfg["completeness"] >= 0
            and cfg.get("threshold_width", 1.0) > 0):
        raise ValueError("tangency_width and threshold_width must be positive, completeness >= 0")
    return tuple(sorted(cfg.items()))


def _rows_from_context(ctx: dict, zeta0: Array, *, rows: tuple[int, ...],
                       kind: str, cfg: tuple, map_rows: bool = False
                       ) -> dict[str, Array]:
    """Proxy of the full-mesh rows ``rows`` of one spectral context.

    ``ctx`` is the shared field-line context of
    :func:`vmex.core.stability._ballooning_context` -- from a live
    ``(state, runtime)`` pair or from a WOUT file -- so the optimization and
    the read-only plotting path run the identical numerics.
    """
    setting = dict(cfg)
    length, alpha, x = _sampling_grids(
        ctx["s"].dtype, setting["nalpha"], setting["num_transit"],
        setting["points_per_transit"])
    options = {k: v for k, v in setting.items() if k not in _BOUNDARY + ("nfp",)}

    def row(j):
        ing = _line_ingredients(ctx, j, jnp.asarray(zeta0, ctx["s"].dtype), alpha, x)
        iota = ing.pop("iota")
        extra = dict(iota=iota, nfp=setting["nfp"]) if kind == "alpha" else {}
        out = _KERNELS[kind](**ing, length=length, num_transit=setting["num_transit"],
                             **extra, **options)
        # The field-line map phi = x / iota is meaningless through iota ~ 0:
        # poison the result instead of returning a plausible number.
        return {**{k: jnp.where(jnp.abs(iota) > 1.0e-6, out[k], jnp.nan)
                   for k in _OUTPUTS[kind]}, "iota": iota,
                "overflow": out["overflow"].astype(ing["bmag"].dtype)}

    # WOUT diagnostics favor one small compilation; live optimization retains
    # the unrolled rows and their faster repeated execution.
    stacked: dict[str, Array] = (jax.lax.map(row, jnp.asarray(rows)) if map_rows else
               jax.tree.map(lambda *xs: jnp.stack(xs), *(row(j) for j in rows)))
    stacked["s"] = ctx["s"][jnp.asarray(rows)]
    return stacked


_ROW_STATICS = ("rows", "kind", "cfg")

#: Traced context entries consumed by :func:`_rows_from_context`; ``lasym`` is
#: Python control flow, so it rides as a static.
_CTX_KEYS = ("s", "hs", "psi_edge", "iotas", "phipf", "m", "xn",
             "rmnc", "zmns", "lmns", "rmns", "zmnc", "lmnc")


@functools.partial(jax.jit, static_argnames=_ROW_STATICS)
def _proxy_rows(state: SpectralState, rt: SolverRuntime, zeta0: Array, *,
                rows, kind, cfg) -> dict[str, Array]:
    """Live-state rows, one XLA executable per settings (module level so
    every boundary iterate of an optimization stage reuses it)."""
    return _rows_from_context(_ballooning_context(state, rt), zeta0,
                              rows=rows, kind=kind, cfg=cfg)


@functools.partial(jax.jit, static_argnames=_ROW_STATICS + ("lasym",))
def _gamma_c_rows_from_tables(tables: dict, zeta0: Array, *, lasym: bool,
                              rows, kind, cfg) -> dict[str, Array]:
    """WOUT-context rows; one executable per (shapes, settings)."""
    return _rows_from_context(dict(tables, lasym=lasym), zeta0, rows=rows,
                              kind=kind, cfg=cfg, map_rows=True)


def _evaluate(kind, state, rt, surfaces, zeta0, options):
    cfg = _resolve(kind, options) + (("nfp", int(rt.resolution.nfp)),)
    rows = _surface_rows(surfaces, int(np.shape(rt.setup.s_full)[0]))
    out = _proxy_rows(state, rt, jnp.asarray(float(zeta0)),
                      rows=rows, kind=kind, cfg=cfg)
    return {**out, "surface_rows": rows}


def _evaluate_wout(kind, wout, surfaces, zeta0, options):
    from .turbulence import _wout_ballooning_context
    from .wout import read_wout

    if isinstance(wout, (str, Path)):
        wout = read_wout(str(wout))
    cfg = _resolve(kind, options) + (("nfp", int(wout.nfp)),)
    ctx = _wout_ballooning_context(wout)
    rows = _surface_rows(surfaces, int(ctx["ns"]))
    out = _gamma_c_rows_from_tables(
        {key: ctx[key] for key in _CTX_KEYS}, jnp.asarray(float(zeta0)),
        lasym=bool(ctx["lasym"]), rows=rows, kind=kind, cfg=cfg)
    return {**out, "surface_rows": rows}


_STATE_DOC = """

    ``surfaces`` are normalized-flux values mapped to the nearest interior
    full-mesh rows (the convention of
    :func:`vmex.core.neoclassical.epsilon_effective_from_wout`).  Each surface
    samples ``nalpha`` equispaced field lines over ``num_transit`` poloidal
    turns at ``points_per_transit`` points per turn, centred on the field-line
    label so the stellarator reflection is an exact invariance of the
    estimator.  Further ``options`` (defaults in ``{table}``) are those of
    :func:`{kernel}`.  Returns per-surface arrays led by ``{name}``;
    ``overflow`` flags rows whose level or well slots were exhausted (their
    value is NaN, never a falsely low number).  Differentiable with
    ``jax.grad`` and through the implicit equilibrium solve.
    """


def gamma_c_state(state: SpectralState, rt: SolverRuntime, *,
                  surfaces: Sequence[float] = (0.35, 0.6, 0.85),
                  zeta0: float = 0.0, **options) -> dict[str, Array]:
    """Nemov ``Gamma_c`` on flux surfaces of a converged equilibrium."""
    return _evaluate("c", state, rt, surfaces, zeta0, options)


def gamma_alpha_state(state: SpectralState, rt: SolverRuntime, *,
                      surfaces: Sequence[float] = (0.35, 0.6, 0.85),
                      zeta0: float = 0.0, **options) -> dict[str, Array]:
    """Velasco ``Gamma_alpha`` on flux surfaces of a converged equilibrium."""
    return _evaluate("alpha", state, rt, surfaces, zeta0, options)


def gamma_c_from_wout(wout, *, surfaces: Sequence[float] = (0.35, 0.6, 0.85),
                      zeta0: float = 0.0, **options) -> dict[str, Array]:
    """``Gamma_c`` from a VMEC-compatible WOUT (object or path), without a solve."""
    return _evaluate_wout("c", wout, surfaces, zeta0, options)


def gamma_alpha_from_wout(wout, *, surfaces: Sequence[float] = (0.35, 0.6, 0.85),
                          zeta0: float = 0.0, **options) -> dict[str, Array]:
    """``Gamma_alpha`` from a VMEC-compatible WOUT (object or path), without a solve."""
    return _evaluate_wout("alpha", wout, surfaces, zeta0, options)


for _fn, _kind, _name, _table in (
        (gamma_c_state, "c", "gamma_c", "``_OPTIONS['c']``"),
        (gamma_alpha_state, "alpha", "gamma_alpha", "``_OPTIONS['alpha']``"),
        (gamma_c_from_wout, "c", "gamma_c", "``_OPTIONS['c']``"),
        (gamma_alpha_from_wout, "alpha", "gamma_alpha", "``_OPTIONS['alpha']``")):
    _fn.__doc__ += _STATE_DOC.format(
        table=_table, name=_name, kernel=_KERNELS[_kind].__name__)
del _fn, _kind, _name, _table


class _Proxy(_ResidualTerm):
    """Composable least-squares term: one proxy row per flux surface.

    ``residuals_state`` returns ``sqrt(weight) * value`` rows, so the total
    cost is the weighted sum of squares (a least-squares form, not a physical
    scaling).  Traceable in both gradient modes.
    """

    kind = ""
    name = ""

    def __init__(self, surfaces=(0.35, 0.6, 0.85), *,
                 weights: Iterable[float] | None = None, zeta0: float = 0.0,
                 **options):
        self.surfaces = np.atleast_1d(np.asarray(surfaces, dtype=float))
        if (self.surfaces.size == 0 or np.any(~np.isfinite(self.surfaces))
                or np.any((self.surfaces <= 0.0) | (self.surfaces >= 1.0))
                or np.any(np.diff(self.surfaces) <= 0.0)):
            raise ValueError(
                "increasing surfaces strictly inside (0, 1) are required")
        self.weights = None if weights is None else np.asarray(weights, dtype=float)
        if self.weights is not None:
            if self.weights.shape != self.surfaces.shape:
                raise ValueError("weights must have the same length as surfaces")
            if np.any(~np.isfinite(self.weights)) or np.any(self.weights < 0.0):
                raise ValueError("weights must be finite and non-negative")
        _resolve(self.kind, options)
        self.options, self.zeta0 = options, float(zeta0)

    def compute_state(self, state: SpectralState, rt: SolverRuntime) -> dict[str, Array]:
        """Return the per-surface proxy and sampling diagnostics."""
        return _evaluate(self.kind, state, rt, self.surfaces, self.zeta0, self.options)

    def residuals_state(self, state: SpectralState, rt: SolverRuntime) -> jnp.ndarray:
        """Return ``sqrt(weight) * value`` per surface (unweighted when ``weights`` is None)."""
        value = self.compute_state(state, rt)[self.name]
        if self.weights is None:
            return value
        return jnp.sqrt(jnp.asarray(self.weights, dtype=value.dtype)) * value

    def total_state(self, state: SpectralState, rt: SolverRuntime) -> Array:
        """Return the weighted sum of squares over the surfaces."""
        rows = self.residuals_state(state, rt)
        return jnp.vdot(rows, rows)


class GammaC(_Proxy):
    """Nemov ``Gamma_c`` (module docstring), the weighted rows of :func:`gamma_c_state`.

    Velasco et al. 2021, section 5.4, find the prompt-loss fraction follows
    their ``|gamma_c*|`` variant (eq. 21) more linearly and validate
    :class:`GammaAlpha` as the predictor.
    """

    kind, name = "c", "gamma_c"


class GammaAlpha(_Proxy):
    """Velasco ``Gamma_alpha`` (module docstring), the weighted rows of :func:`gamma_alpha_state`."""

    kind, name = "alpha", "gamma_alpha"
