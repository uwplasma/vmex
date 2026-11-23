r"""Effective ripple ``epsilon_eff`` (1/nu regime) from Nemov's bounce integrals, differentiable.

The 1/nu flux of a stellarator is proportional to :math:`\epsilon_\mathrm{eff}^{3/2}`
(Nemov et al., Phys. Plasmas 6, 4622 (1999); in the form of Unalmis et al.,
J. Plasma Phys. 92(3), 2026, eq. 2.12, as in ``desc.compute._neoclassical``):

.. math::

   \epsilon_\mathrm{eff}^{3/2} = \frac{\pi R_0^2 B_0^2}
   {4\sqrt2\,\langle|\nabla\psi|\rangle^2 V'}
   \Big\langle \int_{B_{\min}}^{B_{\max}} \frac{\mathrm d b}{b^3}
   \sum_\mathrm{wells} \frac{I_1^2}{I_2}\Big\rangle_\text{per toroidal transit},

.. math::

   I_1 = \int\!\mathrm d\ell\,\frac{\sqrt{1-B/b}}{B}\Big(\frac{4b}{B}-1\Big)
   |\nabla\psi|\kappa_g,\qquad
   I_2 = \int\!\mathrm d\ell\,\frac{\sqrt{1-B/b}}{B},

with :math:`b = 1/\lambda`, :math:`B_0 = B_{\max}` of the surface (the NEO default
``ref_swi = 2``), :math:`R_0 = R_{00}` of the surface and :math:`\psi = \Phi/2\pi`.
In Boozer coordinates :math:`\mathrm d\ell/B = (G+\iota I)\,\mathrm d\varphi/B^2` and
:math:`|\nabla\psi|\kappa_g = (G\,\partial_\theta B - I\,\partial_\varphi B)/(G+\iota I)`
(grad-B curvature), so the integrals need only the Boozer spectrum of ``|B|``
with ``iota``, ``G``, ``I``; ``<|grad psi|>`` and ``V' = 4 pi^2 <sqrt(g)>`` follow
from the Boozer ``R, Z`` and angle shift.

Algorithm.  ``n_lines`` field lines :math:`\theta=\alpha+\iota\varphi` start the
wells over ``n_transits`` toroidal transits (a well belongs to the line whose window
holds its first bounce point, so no well is clipped; each line is extended by
``n_extension`` transits to close it).  ``|B|`` along a line is a Hermite cubic on
``nodes_per_period`` nodes per field period, split at its stationary points into monotone
pieces: bounce points and the critical values of ``|B|`` (where wells merge)
are exact roots, and the ``b`` integral runs between the critical values of each
line with a Gauss rule on analytic pieces.  Each well uses a Gauss rule in the
variable ``u`` of :math:`\varphi=\varphi_m+\Delta\varphi\sin u`, which removes the
square-root behaviour at the bounce points.  Smooth window weights on the
first and last bounce points make the value a continuous function of the
equilibrium, so forward and reverse mode agree with finite differences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

import jax
import jax.numpy as jnp

__all__ = [
    "RippleBudget",
    "epsilon_effective_3_2_from_boozer",
    "epsilon_effective_3_2_from_state",
    "epsilon_effective_3_2_from_wout",
    "screen_transport",
    "transport_improves",
]

@dataclass(frozen=True)
class RippleBudget:
    """Resolution of the bounce integrals; the defaults hold 1% against a 3x larger budget.

    ``n_lines`` field lines start the wells over ``n_transits`` toroidal transits of the
    window and are extended by ``n_extension`` transits that close them;
    ``nodes_per_period`` Hermite nodes per field period; ``n_pitch`` Gauss nodes per
    interval between critical values of ``|B|``; ``n_well`` Gauss nodes per well;
    ``n_surface`` points per direction for ``B_max``, ``<|grad psi|>`` and ``V'``.
    ``max_wells`` (default ``2 nfp n_transits + 4``) caps the wells that start at one pitch
    (more returns NaN); ``max_critical`` (default ``4 nfp (n_transits + n_extension)``) caps
    the critical values per line, thinned evenly in rank beyond it. ``line_batch`` lines are
    evaluated together (a speed against memory trade: 1.4x on rotating ellipses, 1.5x more memory).
    """

    n_lines: int = 16
    n_transits: int = 3
    n_extension: int = 3
    nodes_per_period: int = 16
    n_pitch: int = 4
    n_well: int = 32
    n_surface: int = 64
    max_wells: int | None = None
    max_critical: int | None = None
    line_batch: int = 1


def _neo_imports():
    from .._compat import require_optional

    require_optional("neo-jax", "effective ripple")
    try:
        from neo_jax import NeoConfig, run_neo
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "the NEO screen requires NEO_JAX; install VMEX with "
            "`pip install vmex[neoclassical]`") from exc
    return NeoConfig, run_neo


_ROOT_ITERS = 12


def _gl(n, a=0.0, b=1.0):
    x, w = np.polynomial.legendre.leggauss(n)
    return a + (b - a) * 0.5 * (x + 1), 0.5 * (b - a) * w


def _series(amp, xm, xn, theta, phi, order=1):
    """Re sum amp*exp(i(m theta - n phi)) and its theta/phi derivatives (amp = c - i s)."""
    z = amp * jnp.exp(1j * (xm * theta[..., None] - xn * phi[..., None]))
    f, ft, fp = z.real.sum(-1), -(xm * z.imag).sum(-1), (xn * z.imag).sum(-1)
    if order == 1:
        return f, ft, fp
    return (f, ft, fp, -(xm**2 * z.real).sum(-1), (xm * xn * z.real).sum(-1), -(xn**2 * z.real).sum(-1))


def _peval(c, t):
    return c[..., 0] + t * (c[..., 1] + t * (c[..., 2] + t * c[..., 3]))


def _pder(c, t):
    return c[..., 1] + t * (2 * c[..., 2] + 3 * t * c[..., 3])


def _cubic(y, dy, h):
    """Power-basis coefficients (N, 4) of the Hermite cubic of each node interval, t in [0, 1]."""
    m0, m1, d = dy[:-1] * h, dy[1:] * h, y[1:] - y[:-1]
    return jnp.stack([y[:-1], m0, 3 * d - 2 * m0 - m1, m0 + m1 - 2 * d], -1)


def _stationary(c):
    """Sorted interior roots of p'(t) for cubics c (..., 4); absent roots sit at t = 1."""
    a, bq, cq = 3 * c[..., 3], 2 * c[..., 2], c[..., 1]
    disc = bq * bq - 4 * a * cq
    sq = jnp.sqrt(jnp.maximum(disc, 0.0))
    q = -0.5 * (bq + jnp.where(bq >= 0, sq, -sq))
    quad = jnp.abs(a) > 1e-300
    r1 = jnp.where(quad, q / jnp.where(quad, a, 1.0), -cq / jnp.where(bq == 0, 1.0, bq))
    r2 = jnp.where(jnp.abs(q) > 1e-300, cq / jnp.where(jnp.abs(q) > 1e-300, q, 1.0), 2.0)
    r1 = jnp.where((r1 > 0) & (r1 < 1) & (disc > 0), r1, 1.0)
    r2 = jnp.where((r2 > 0) & (r2 < 1) & (disc > 0) & quad, r2, 1.0)
    return jnp.minimum(r1, r2), jnp.maximum(r1, r2)


def _root(c, b, lo, hi):
    """Root of p(t) - b on the monotone stretch [lo, hi]: safeguarded Newton, then one
    differentiable Newton step (the exact implicit derivative)."""
    f = lambda t: _peval(c, t) - b  # noqa: E731
    s0 = jnp.sign(f(lo))

    def step(carry, _):
        lo_, hi_, t = carry
        p = f(t)
        same = jnp.sign(p) == s0
        lo_, hi_ = jnp.where(same, t, lo_), jnp.where(same, hi_, t)
        q = _pder(c, t)
        tn = t - p / jnp.where(q == 0, 1e-300, q)
        inside = (tn > jnp.minimum(lo_, hi_)) & (tn < jnp.maximum(lo_, hi_))
        return (lo_, hi_, jnp.where(inside, tn, 0.5 * (lo_ + hi_))), None

    t = jax.lax.stop_gradient(jax.lax.scan(step, (lo, hi, 0.5 * (lo + hi)), None, length=_ROOT_ITERS)[0][2])
    q = _pder(c, t)
    return t - f(t) / jnp.where(jnp.abs(q) < 1e-14, 1e-14, q)


_root_v = jax.vmap(_root, in_axes=(0, None, 0, 0))


def _smooth_step(x):
    y = jnp.clip(x, 0.0, 1.0)
    return y * y * (3 - 2 * y)


def _wells(cB, cQ, pcs, h, b, GI, nstart, ramp, nwell, nquad):
    """``sum_wells window * I1^2/I2`` of one line at pitch ``b``; also the number of starts beyond ``nwell``."""
    n = cB.shape[0]
    iv, t0, t1, v0 = pcs
    m = iv.shape[0]
    pos = jnp.concatenate([v0, _peval(cB[-1], 1.0)[None]]) >= b
    down = pos[:-1] & ~pos[1:] & (iv < nstart)                   # first bounce point inside the window
    up = ~pos[:-1] & pos[1:]
    d = jnp.nonzero(down, size=nwell, fill_value=m)[0]
    ups = jnp.nonzero(up, size=m // 2 + 1, fill_value=m)[0]
    u = ups[jnp.minimum(jnp.searchsorted(ups, d), m // 2)]      # the next up-crossing closes the well
    valid = (d < m) & (u < m)
    d, u = jnp.minimum(d, m - 1), jnp.minimum(u, m - 1)
    r1 = (iv[d] + _root_v(cB[iv[d]], b, t0[d], t1[d])) * h
    r2 = (iv[u] + _root_v(cB[iv[u]], b, t0[u], t1[u])) * h
    x, w = _gl(nquad, -1.0, 1.0)
    ang = 0.5 * jnp.pi * x
    mid, half = jnp.where(valid, 0.5 * (r1 + r2), 0.0), jnp.where(valid, 0.5 * (r2 - r1), 0.0)
    phi = mid[:, None] + half[:, None] * jnp.sin(ang)
    wq = half[:, None] * 0.5 * jnp.pi * jnp.cos(ang) * w
    idx = jnp.clip(jnp.floor(phi / h).astype(int), 0, n - 1)
    t = phi / h - idx
    Bq, Qq = _peval(cB[idx], t), _peval(cQ[idx], t)
    x2 = 1 - Bq / b
    rad = jnp.where(x2 > 0, jnp.sqrt(jnp.where(x2 > 0, x2, 1.0)), 0.0)
    I1 = (wq * rad * (4 * b / Bq - 1) * Qq / Bq**2).sum(-1)
    I2 = GI * (wq * rad / Bq**2).sum(-1)
    ok = valid & (I2 > 1e-30)
    win = (_smooth_step(r1 / ramp) * _smooth_step((nstart * h - r1) / ramp)
           * _smooth_step((n * h - r2) / (0.5 * (n - nstart) * h)))
    return jnp.where(ok, win * I1**2 / jnp.where(ok, I2, 1.0), 0.0).sum(), down.sum() > nwell


def _surface(amp, xm, xn, iota, cur, G, nfp, bud, ramp=0.25):
    """``epsilon_eff^(3/2)`` of one Boozer surface; NaN if more wells start than fit in one pitch node."""
    nturn, nper, nquad, nsurf = bud.n_transits, bud.nodes_per_period, bud.n_well, bud.n_surface
    n = nfp * (nturn + bud.n_extension) * nper
    nstart = nfp * nturn * nper
    L, h = 2 * jnp.pi * nturn, 2 * jnp.pi * (nturn + bud.n_extension) / n
    nwell = bud.max_wells or 2 * nfp * nturn + 4
    ncrit = bud.max_critical or 4 * nfp * (nturn + bud.n_extension)
    GI = G + iota * cur
    # surface quantities on a (theta, field period) grid
    th = jnp.arange(nsurf) * 2 * jnp.pi / nsurf
    TH, PH = jnp.meshgrid(th, th / nfp, indexing="ij")
    B = _series(amp["B"], xm, xn, TH, PH)[0]
    R, Rt, Rp = _series(amp["R"], xm, xn, TH, PH)
    _, Zt, Zp = _series(amp["Z"], xm, xn, TH, PH)
    _, pt, pp = _series(amp["p"], xm, xn, TH, PH)       # cylindrical angle = phi_B + p
    dS = jnp.linalg.norm(jnp.cross(jnp.stack([Rt, R * pt, Zt]), jnp.stack([Rp, R * (1 + pp), Zp]), axis=0), axis=0)
    sqrtg = GI / B**2
    gradpsi, Vp = dS.mean() / sqrtg.mean(), 4 * jnp.pi**2 * sqrtg.mean()
    R0 = amp["R"][np.flatnonzero((xm == 0) & (xn == 0))[0]].real
    pitch, wpitch = _gl(bud.n_pitch)
    sg, dsg = pitch * pitch * (3 - 2 * pitch), 6 * pitch * (1 - pitch)   # nodes cluster at the interval ends
    phi = jnp.arange(n + 1) * h

    def line(alpha):
        Bn, Bt, Bp, Btt, Btp, Bpp = _series(amp["B"], xm, xn, alpha + iota * phi, phi, order=2)
        Qn, dBn = G * Bt - cur * Bp, Bp + iota * Bt
        dQn = G * (Btp + iota * Btt) - cur * (Bpp + iota * Btp)
        cB, cQ = _cubic(Bn, dBn, h), _cubic(Qn, dQn, h)
        ta, tb = _stationary(cB)
        brk = jnp.stack([jnp.zeros(n), ta, tb, jnp.ones(n)], -1)
        iv = jnp.repeat(jnp.arange(n), 3)
        t0, t1 = brk[:, :3].ravel(), brk[:, 1:].ravel()
        pcs = (iv, t0, t1, _peval(cB[iv], t0))
        # critical values of B: the stationary points (envelope theorem: t carries no derivative)
        cand = jnp.concatenate([jnp.where(ta < 1, _peval(cB, jax.lax.stop_gradient(ta)), jnp.nan),
                                jnp.where(tb < 1, _peval(cB, jax.lax.stop_gradient(tb)), jnp.nan)])
        top, ncand = Bn.max(), (~jnp.isnan(cand)).sum()
        cs = jnp.sort(jnp.where(jnp.isnan(cand), top, cand))
        # keep every critical value, or ncrit of them evenly spread in rank when a nearly flat |B| has more
        pick = jnp.where(ncand > ncrit, jnp.round(jnp.linspace(0.0, 1.0, ncrit) * (ncand - 1)), jnp.arange(ncrit)).astype(int)
        crit = jnp.sort(jnp.concatenate([cs[jnp.minimum(pick, cand.size - 1)], Bn[:1], Bn[-1:], top[None]]))
        lo, hi = crit[:-1], crit[1:]
        bq = (lo[:, None] + (hi - lo)[:, None] * sg).ravel()
        wb = ((hi - lo)[:, None] * dsg * wpitch).ravel()
        val, over = jax.vmap(lambda b: _wells(cB, cQ, pcs, h, b, GI, nstart, ramp * L, nwell, nquad))(bq)
        return (wb * val / bq**3).sum() * 2 * jnp.pi / (L * (1 - ramp)), over.any()

    alpha = (jnp.arange(bud.n_lines) + 0.5) * 2 * jnp.pi / bud.n_lines
    S, over = jax.lax.map(jax.checkpoint(line), alpha, **({"batch_size": bud.line_batch} if bud.line_batch > 1 else {}))
    eps = (jnp.pi * R0)**2 * B.max()**2 / (4 * jnp.sqrt(2.0) * gradpsi**2 * Vp) * S.mean()
    return jnp.where(over.any(), jnp.nan, eps)


def epsilon_effective_3_2_from_boozer(booz: Any, budget: RippleBudget | None = None):
    r"""Return ``(s, epsilon_eff**(3/2))`` from Boozer-coordinate data (traceable).

    ``booz`` is a mapping with the NEO_JAX keys: ``nfp_b``, ``ixm_b``, ``ixn_b``
    (``n`` includes ``nfp``), ``iota_b``, ``buco_b`` (``I``), ``bvco_b`` (``G``),
    ``rmnc_b``, ``zmns_b``, ``pmns_b`` (``phi - zeta_B``), ``bmnc_b``, ``s_b``,
    optionally the sine partners ``rmns_b``, ``zmnc_b``, ``pmnc_b``, ``bmns_b``
    of an asymmetric equilibrium and ``mode_first`` for ``(mode, surface)``
    arrays.  ``budget`` is a :class:`RippleBudget`; the default is within 1% of a
    3x larger one on the validation set (six boundaries and a QI spectrum).  A
    surface that overflows ``max_wells`` returns NaN.
    """
    bud = budget or RippleBudget()
    t = (lambda a: jnp.asarray(a).T) if booz.get("mode_first") else jnp.asarray  # noqa: E731
    zero = lambda k, ref: t(booz[k]) if k in booz else 0 * ref  # noqa: E731
    bc, rc, zs, ps = t(booz["bmnc_b"]), t(booz["rmnc_b"]), t(booz["zmns_b"]), t(booz["pmns_b"])
    amps = {"B": bc - 1j * zero("bmns_b", bc), "R": rc - 1j * zero("rmns_b", rc),
            "Z": zero("zmnc_b", zs) - 1j * zs, "p": zero("pmnc_b", ps) - 1j * ps}
    xm, xn = np.asarray(booz["ixm_b"], float), np.asarray(booz["ixn_b"], float)
    per_row = (jnp.asarray(booz["iota_b"]), jnp.asarray(booz["buco_b"]), jnp.asarray(booz["bvco_b"]), amps)
    values = jax.lax.map(lambda a: _surface(a[3], xm, xn, a[0], a[1], a[2], int(booz["nfp_b"]), bud), per_row)
    return booz["s_b"], values


def _bracket(s_half, surfaces):
    """Static half-mesh rows ``rows`` and the weights that interpolate linearly in s to ``surfaces``."""
    s = np.atleast_1d(np.asarray(surfaces, float))
    hi = np.clip(np.searchsorted(s_half, s), 1, len(s_half) - 1)
    w = np.clip((s - s_half[hi - 1]) / (s_half[hi] - s_half[hi - 1]), 0.0, 1.0)
    rows = np.unique(np.concatenate([hi - 1, hi]))
    return rows, np.searchsorted(rows, hi - 1), np.searchsorted(rows, hi), w


def _interpolate(values, lo, hi, w):
    return (1 - w) * values[lo] + w * values[hi]


def epsilon_effective_3_2_from_state(state, rt, surfaces: Sequence[float] = (0.25, 0.5, 0.75), *,
                                     mboz: int = 12, nboz: int = 9, budget: RippleBudget | None = None):
    r"""``epsilon_eff**(3/2)`` of a solved VMEX state at ``surfaces``, traceable end to end.

    The path is the differentiable one of the optimizer: the state's Boozer tables
    (:func:`vmex.core.omnigenity.boozer_spectrum_state`, ``booz_xform_jax``'s jittable
    kernel) feed :func:`epsilon_effective_3_2_from_boozer`.  The value is evaluated on
    the two half-mesh rows that bracket each surface and interpolated linearly in
    ``s``.  ``jax.grad``, ``jax.jvp`` and ``jax.jacrev`` pass through
    :func:`vmex.core.implicit.solve_implicit` and this function alike.  For the
    effective ripple itself use ``epsilon_effective_3_2_from_state(...) ** (2 / 3)``.
    """
    from .omnigenity import boozer_spectrum_state

    ns = int(np.shape(rt.setup.s_full)[0])
    s_half = (np.arange(ns - 1) + 0.5) / (ns - 1)
    rows, lo, hi, w = _bracket(s_half, surfaces)
    b = boozer_spectrum_state(state, rt, surfaces=s_half[rows], mboz=mboz, nboz=nboz, geometry=True)
    booz = {"nfp_b": b["nfp"], "ixm_b": b["xm_b"], "ixn_b": b["xn_b"], "iota_b": b["iota_b"],
            "buco_b": b["I_b"], "bvco_b": b["G_b"], "s_b": b["s_b"]}
    booz |= {k: b[k] for k in ("bmnc_b", "bmns_b", "rmnc_b", "rmns_b", "zmnc_b", "zmns_b", "pmnc_b", "pmns_b")}
    return _interpolate(epsilon_effective_3_2_from_boozer(booz, budget)[1], lo, hi, w)


def epsilon_effective_3_2_from_wout(wout, surfaces: Sequence[float] = (0.2, 0.4, 0.6, 0.8, 0.95), *,
                                    mboz: int = 12, nboz: int = 9, budget: RippleBudget | None = None):
    """``epsilon_eff**(3/2)`` at ``surfaces`` from a (VMEX or VMEC) ``wout``, interpolated like the state route."""
    from booz_xform_jax import Booz_xform

    bx = Booz_xform(verbose=0, mboz=int(mboz), nboz=int(nboz))
    bx.read_wout_data(wout)
    rows, lo, hi, w = _bracket(np.asarray(bx.s_in, float), surfaces)
    bx.compute_surfs = [int(r) for r in rows]
    bx.run()
    values = epsilon_effective_3_2_from_boozer(_neo_boozer_data(bx, rows), budget)[1]
    return _interpolate(values, lo, hi, w)


def _neo_boozer_data(bx, indices):
    """Share one mode-first Boozer transform with NEO, including sine partners."""
    booz = {
        "nfp_b": int(bx.nfp), "ns_b": len(indices), "mode_first": True,
        "asym": bool(getattr(bx, "asym", False)),
        "ixm_b": np.asarray(bx.xm_b), "ixn_b": np.asarray(bx.xn_b),
        "iota_b": np.asarray(bx.iota)[indices],
        "buco_b": np.asarray(bx.Boozer_I), "bvco_b": np.asarray(bx.Boozer_G),
        "rmnc_b": np.asarray(bx.rmnc_b), "zmns_b": np.asarray(bx.zmns_b),
        "pmns_b": -np.asarray(bx.numns_b), "bmnc_b": np.asarray(bx.bmnc_b),
        "s_b": np.asarray(bx.s_b),
    }
    if booz["asym"]:
        booz.update(rmns_b=np.asarray(bx.rmns_b), zmnc_b=np.asarray(bx.zmnc_b),
                    pmnc_b=-np.asarray(bx.numnc_b), bmns_b=np.asarray(bx.bmns_b))
    return booz


def screen_transport(wout, surfaces, *, mboz=24, nboz=24, config=None):
    """Matched forward NEO screen; not a converged transport calculation."""
    from booz_xform_jax import Booz_xform
    NeoConfig, run_neo = _neo_imports()

    bx = Booz_xform(verbose=0, mboz=mboz, nboz=nboz)
    bx.read_wout_data(wout)
    indices = [int(np.argmin(abs(np.asarray(bx.s_in)-s))) for s in surfaces]
    if not np.allclose(np.asarray(bx.s_in)[indices], surfaces, rtol=0, atol=1e-12):
        raise ValueError("The NEO screen requires the requested half-mesh radii")
    bx.compute_surfs = indices
    bx.run()
    config = config if config is not None else NeoConfig(theta_n=192, phi_n=192, npart=80, multra=2, nstep_per=80,
                       nstep_min=800, nstep_max=2000, acc_req=.005, ref_swi=2,
                       max_rational_field_periods=100000, rational_surface_policy="error")
    results = run_neo(_neo_boozer_data(bx, indices), config=config, use_jax=True,
                      progress=False, jax_surface_scan=False)
    values = []
    for result in results:
        diag, value = result.diagnostics, float(result.epsilon_effective)
        if (not np.isfinite(value) or value <= 0 or not diag.get("rational_correction_valid")
                or diag.get("approximation_used") or diag.get("rational_work_limit_exceeded")):
            raise ValueError(f"Invalid NEO screen at s={float(result.s):.6g}, Q={value}; keep the incumbent")
        values.append(value)
    return np.asarray(values)


def transport_improves(baseline, trial):
    """Require a resolved mean gain and no radial tradeoff in the coarse screen."""
    a, b = np.asarray(baseline), np.asarray(trial)
    return bool(a.ndim == b.ndim == 1 and a.shape == b.shape and a.size > 0 and np.isfinite(a).all()
                and np.isfinite(b).all() and np.all(a > 0) and np.all(b > 0)
                and np.all(b <= a) and b.mean() < .99*a.mean())
