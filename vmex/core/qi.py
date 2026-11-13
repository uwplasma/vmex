"""Quasi-isodynamic residuals: Goodman-style constructed field and bounce-action invariance."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

import jax
import jax.numpy as jnp

from .bounce import bounce_action_from_boozer
from .omnigenity import boozer_spectrum_state
from .statephysics import _as_1d

Array = Any

__all__ = [
    "ConstructedQIResidual",
    "JInvariantQIResidual",
    "boozer_modes_from_wout",
    "j_invariant_alpha_derivative_from_boozer",
    "j_invariant_qi_residual_from_boozer",
    "quasi_isodynamic_residual",
    "quasi_isodynamic_residual_from_wout",
]


def _width(softness, dtype):
    """Sigmoid/softmax width ``softness`` floored at machine epsilon."""
    return jnp.maximum(jnp.asarray(float(softness), dtype=dtype),
                       jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype))


def _qi_grid(bmnc_b, xm_b, xn_b, iota_b, *, bmns_b=None, nfp: int, weights, nphi: int,
             nalpha: int, n_bounce: int, include_bounce_endpoints: bool,
             softness: float, phimin: float):
    """Normalized ``|B|`` along field lines and the bounce levels.

    ``theta = alpha + iota * phi`` samples ``nalpha`` field-line labels over
    one field period; ``bnorm`` rescales ``|B|`` to [0, 1] per surface.
    Returns ``(weights, phi0, phi1, phi, alpha, bmag, bnorm, levels, eps)``
    with ``eps`` the sigmoid width ``softness`` floored at machine epsilon.
    """
    bmnc_b = jnp.asarray(bmnc_b, dtype=jnp.float64)
    bmns_b = (jnp.zeros_like(bmnc_b) if bmns_b is None
              else jnp.asarray(bmns_b, dtype=bmnc_b.dtype))
    xm_b = jnp.asarray(xm_b, dtype=jnp.float64)
    xn_b = jnp.asarray(xn_b, dtype=jnp.float64)
    iota_b = jnp.asarray(iota_b, dtype=jnp.float64)
    if bmnc_b.ndim != 2:
        raise ValueError(f"bmnc_b must have shape (nsurf, nmodes), got {bmnc_b.shape}")
    if bmns_b.shape != bmnc_b.shape:
        raise ValueError("bmns_b must have the same shape as bmnc_b")
    if nphi < 4 or nalpha < 2 or n_bounce < 2:
        raise ValueError("QI residual requires nphi >= 4, nalpha >= 2, n_bounce >= 2")
    dtype = bmnc_b.dtype
    finfo = jnp.finfo(dtype)
    weights = jnp.ones((bmnc_b.shape[0],), dtype=dtype) if weights is None else _as_1d(weights)

    phi0 = jnp.asarray(float(phimin), dtype=dtype)
    phi1 = phi0 + 2.0 * np.pi / nfp
    phi = jnp.linspace(phi0, phi1, nphi, endpoint=True, dtype=dtype)
    alpha = jnp.linspace(0.0, 2.0 * jnp.pi, nalpha, endpoint=False, dtype=dtype)
    theta = alpha[None, None, :] + iota_b[:, None, None] * phi[None, :, None]
    angle = (theta[:, :, :, None] * xm_b[None, None, None, :]
             - phi[None, :, None, None] * xn_b[None, None, None, :])
    bmag = jnp.sum(
        bmnc_b[:, None, None, :] * jnp.cos(angle)
        + bmns_b[:, None, None, :] * jnp.sin(angle), axis=-1)

    bmin = jnp.min(bmag, axis=(1, 2), keepdims=True)
    bmax = jnp.max(bmag, axis=(1, 2), keepdims=True)
    bnorm = (bmag - bmin) / jnp.maximum(bmax - bmin, jnp.asarray(finfo.tiny, dtype=dtype))

    if include_bounce_endpoints:
        levels = jnp.linspace(0.0, 1.0, n_bounce, endpoint=True, dtype=dtype)
    else:
        levels = jnp.linspace(0.0, 1.0, n_bounce + 2, endpoint=True, dtype=dtype)[1:-1]
    return weights, phi0, phi1, phi, alpha, bmag, bnorm, levels, _width(softness, dtype)


def _minimum_position(b, softness: float, *, periodic: bool):
    """Index-space position of the minimum of ``b`` along its last axis.

    ``softness = 0`` is the grid argmin.  A positive value is the softmin
    centroid with weights ``exp(-(b - min b)/softness)`` (circular for a
    periodic line), which moves continuously as ``b`` changes and tends to
    the argmin as ``softness -> 0``.
    """
    n = b.shape[-1]
    if softness <= 0.0:
        return jnp.argmin(b, axis=-1).astype(b.dtype)
    weights = jax.nn.softmax(-(b - jnp.min(b, axis=-1, keepdims=True)) / softness, axis=-1)
    index = jnp.arange(n, dtype=b.dtype)
    if not periodic:
        return jnp.sum(weights * index, axis=-1)
    angle = 2.0 * jnp.pi * index / n
    centre = jnp.arctan2(jnp.sum(weights * jnp.sin(angle), axis=-1),
                         jnp.sum(weights * jnp.cos(angle), axis=-1))
    return jnp.mod(centre * n / (2.0 * jnp.pi), n)


def _sample(b, position, *, periodic: bool):
    """Linear interpolation of ``b`` at fractional index ``position``.

    Integer positions return the samples exactly.  A non-periodic line reads
    the normalized maximum 1 one step beyond either end and is clamped there,
    so a branch that leaves the field period ends on the well's wall.
    """
    n = b.shape[-1]
    if not periodic:
        one = jnp.ones(b.shape[:-1] + (1,), dtype=b.dtype)
        b = jnp.concatenate([one, b, one], axis=-1)
        position = jnp.clip(position + 1.0, 0.0, float(n + 1))
    lower = jnp.floor(position)
    frac = position - lower
    lower = lower.astype(jnp.int32)
    upper = lower + 1
    if periodic:
        lower, upper = jnp.mod(lower, n), jnp.mod(upper, n)
    else:
        upper = jnp.minimum(upper, n + 1)
    return ((1.0 - frac) * jnp.take_along_axis(b, lower, axis=-1)
            + frac * jnp.take_along_axis(b, upper, axis=-1))


def _branch_width_residual(bnorm, levels, sqrt_w, *, weight, softness, argmin_softness):
    """Variance over ``alpha`` of the trapped-well bounce width at each level."""
    nphi, nalpha = bnorm.shape[1], bnorm.shape[2]
    bper = jnp.swapaxes(bnorm[:, :-1, :], 1, 2)           # periodic, (nsurf, nalpha, nper)
    nper = nphi - 1
    offs = jnp.arange(max(1, nper // 2) + 1, dtype=bnorm.dtype)[None, None, :]
    pmin = _minimum_position(bper, argmin_softness, periodic=True)[:, :, None]
    left = jnp.maximum.accumulate(_sample(bper, pmin - offs, periodic=True), axis=-1)
    right = jnp.maximum.accumulate(_sample(bper, pmin + offs, periodic=True), axis=-1)
    tiny = jnp.asarray(jnp.finfo(bnorm.dtype).tiny, dtype=bnorm.dtype)
    left = (left - left[..., :1]) / jnp.maximum(left[..., -1:] - left[..., :1], tiny)
    right = (right - right[..., :1]) / jnp.maximum(right[..., -1:] - right[..., :1], tiny)
    distance = offs[0, 0] / jnp.asarray(nper, dtype=bnorm.dtype)
    beps = _width(softness, bnorm.dtype)

    def crossing(branch):
        logits = -((branch[:, :, :, None] - levels[None, None, None, :]) / beps) ** 2
        logits = logits - jnp.max(logits, axis=2, keepdims=True)
        w = jnp.exp(logits)
        w = w / jnp.sum(w, axis=2, keepdims=True)
        return jnp.sum(w * distance[None, None, :, None], axis=2)

    width = crossing(left) + crossing(right)              # (nsurf, nalpha, nlev)
    residual = (width - jnp.mean(width, axis=1, keepdims=True)) * sqrt_w * weight
    return jnp.ravel(residual) / jnp.sqrt(jnp.asarray(nalpha * levels.shape[0], dtype=jnp.float64))


def _shuffle_residual(bnorm, levels, sqrt_w, phi0, phi1, *, weight, softness, argmin_softness):
    """Constructed (shuffled) well and its residual against ``bnorm``.

    Each well's two branch crossings are shifted so every field line has the
    mean bounce width, then the shuffled well is reinterpolated onto the
    original grid.  Returns ``(shuffled, residual_vector)`` with ``shuffled``
    of shape ``(nsurf, nalpha, nphi)``.
    """
    dtype = bnorm.dtype
    nsurf, nphi, nalpha = bnorm.shape
    b_alpha = jnp.swapaxes(bnorm, 1, 2)                   # (nsurf, nalpha, nphi)
    offs = jnp.arange(nphi, dtype=dtype)
    dphi = (phi1 - phi0) / (nphi - 1)
    imin = _minimum_position(b_alpha, argmin_softness, periodic=False)
    left = jnp.maximum.accumulate(
        _sample(b_alpha, imin[:, :, None] - offs[None, None, :], periodic=False), axis=-1)
    right = jnp.maximum.accumulate(
        _sample(b_alpha, imin[:, :, None] + offs[None, None, :], periodic=False), axis=-1)

    seps = _width(softness, dtype)
    trapz_w = jnp.ones((nphi,), dtype=dtype).at[0].set(0.5).at[-1].set(0.5)

    def branch_crossing(branch):
        occ = jax.nn.sigmoid((levels[None, None, None, :] - branch[:, :, :, None]) / seps)
        return jnp.sum(occ * trapz_w[None, None, :, None], axis=2) * dphi

    lcross, rcross = branch_crossing(left), branch_crossing(right)
    bw = lcross + rcross

    min_phi = phi0 + imin * dphi
    lend = jnp.maximum(min_phi - phi0, 0.0)[:, :, None]
    rend = jnp.maximum(phi1 - min_phi, 0.0)[:, :, None]
    signed_phi = (offs[None, None, :] - imin[:, :, None]) * dphi

    level_full = jnp.concatenate([jnp.zeros((1,), dtype=dtype), levels, jnp.ones((1,), dtype=dtype)])
    y_target = jnp.concatenate([jnp.flip(level_full, axis=0), level_full[1:]], axis=0)

    delta = 0.5 * (bw - jnp.mean(bw, axis=1, keepdims=True))
    zeros = jnp.zeros((nsurf, nalpha, 1), dtype=dtype)
    lfull = jnp.maximum.accumulate(
        jnp.concatenate([zeros, jnp.clip(lcross - delta, 0.0, lend), lend], axis=-1), axis=-1)
    rfull = jnp.maximum.accumulate(
        jnp.concatenate([zeros, jnp.clip(rcross - delta, 0.0, rend), rend], axis=-1), axis=-1)
    x_target = jnp.concatenate([-jnp.flip(lfull, axis=-1), rfull[:, :, 1:]], axis=-1)
    x_target = x_target + jnp.arange(x_target.shape[-1], dtype=dtype) * 1.0e-14 * (phi1 - phi0)

    shuffled = jax.vmap(jax.vmap(lambda xp, x: jnp.interp(x, xp, y_target)))(x_target, signed_phi)
    # The first coordinate is exactly -lend (the wall); roundoff in phi - phimin
    # must not slide it into the artificial node ramp.
    shuffled = shuffled.at[..., 0].set(1.0)
    residual = (shuffled - b_alpha) * sqrt_w * weight
    if argmin_softness > 0.0:
        # The period ends sit on the constructed wall, where the target
        # jumps to 1 as a branch crossing reaches the end: drop them.
        residual = residual[..., 1:-1]
    return shuffled, jnp.ravel(residual) / jnp.sqrt(jnp.asarray(nalpha * nphi, dtype=jnp.float64))


def quasi_isodynamic_residual(
    *,
    bmnc_b,
    bmns_b=None,
    xm_b,
    xn_b,
    iota_b,
    nfp: int,
    weights: Iterable[float] | None = None,
    nphi: int = 151,
    nalpha: int = 31,
    n_bounce: int = 51,
    include_bounce_endpoints: bool = False,
    softness: float = 2.0e-2,
    width_weight: float = 1.0,
    branch_width_weight: float = 0.5,
    branch_width_softness: float = 1.0e-2,
    profile_weight: float = 0.1,
    shuffle_profile_weight: float = 1.0,
    shuffle_profile_softness: float = 2.0e-2,
    argmin_softness: float = 0.0,
    phimin: float = 0.0,
) -> dict[str, Array]:
    """Smooth Goodman-style quasi-isodynamic residual from Boozer ``|B|`` modes.

    A configuration is quasi-isodynamic when the ``|B|`` contours are
    poloidally closed and the trapped-particle bounce distance between the
    two branches of each magnetic well is independent of the field-line label
    ``alpha`` (omnigenity).  This residual samples the normalized ``|B|``
    along field lines ``theta = alpha + iota*phi`` over one field period and
    penalizes, per surface:

    - **level-set width variance** (``width_weight``): for each bounce level
      ``B*`` the smooth occupancy ``sigmoid((B* - bnorm)/softness)`` gives the
      fraction of the field line below ``B*``; its variance over ``alpha``
      measures misalignment of the ``|B|`` contours.
    - **branch width variance** (``branch_width_weight``): each field line is
      split at its ``|B|`` minimum, both branches are made monotone with a
      running maximum, and the (smooth) level-crossing distances of the two
      branches are summed — the trapped-well bounce width, whose variance
      over ``alpha`` is the classic omnigenity error.
    - **profile consistency** (``profile_weight``): small penalty on the
      variance of ``bnorm`` itself over ``alpha`` at fixed ``phi``, which
      keeps degenerate QH-like candidates from gaming the width terms.
    - **branch-shuffle profile** (``shuffle_profile_weight``): the "squash and
      shuffle" comparison — each well's branch crossings are shifted so every
      field line has the *mean* bounce width, the shuffled well is
      reinterpolated onto the original grid and compared pointwise to the
      original ``bnorm`` (the closest smooth analogue of Goodman et al.'s
      construction of the nearest omnigenous field).

    Both branch terms start each well at its ``|B|`` minimum.  With the
    default ``argmin_softness = 0`` that is the grid argmin, which jumps a
    whole grid cell as the field moves, so the residual is discontinuous
    there.  A positive ``argmin_softness`` (in units of the normalized
    ``|B|``) uses the softmin centroid and reads the branches by linear
    interpolation, and drops the two period-end samples of the shuffle
    comparison, which sit on the constructed well's wall; the residual is
    then continuous in the field.

    ``xn_b`` uses physical toroidal mode numbers (booz_xform convention).
    Returns ``residuals1d`` (least-squares vector) and ``total`` (its squared
    norm) with the sampled ``bmag``, ``bnorm``, ``constructed_bmag``,
    ``levels``, ``phi`` and ``alpha``.
    """
    (weights_arr, phi0, phi1, phi, alpha, bmag, bnorm, levels, eps) = _qi_grid(
        bmnc_b, xm_b, xn_b, iota_b, bmns_b=bmns_b,
        nfp=int(nfp), weights=weights, nphi=int(nphi),
        nalpha=int(nalpha), n_bounce=int(n_bounce),
        include_bounce_endpoints=bool(include_bounce_endpoints),
        softness=float(softness), phimin=float(phimin))
    nalpha_, nphi_ = bnorm.shape[2], bnorm.shape[1]
    sqrt_w = jnp.sqrt(weights_arr)[:, None, None]
    argmin_softness = float(argmin_softness)
    constructed = jnp.swapaxes(bnorm, 1, 2)

    occupancy = jax.nn.sigmoid((levels[None, None, None, :] - bnorm[:, :, :, None]) / eps)
    widths = jnp.mean(occupancy, axis=1)                      # (nsurf, nalpha, nlev)
    width_res = (widths - jnp.mean(widths, axis=1, keepdims=True)) * sqrt_w * width_weight
    profile_res = (bnorm - jnp.mean(bnorm, axis=2, keepdims=True)) * sqrt_w * profile_weight
    pieces = [jnp.ravel(width_res) / jnp.sqrt(jnp.asarray(nalpha_ * levels.shape[0], dtype=jnp.float64)),
              jnp.ravel(profile_res) / jnp.sqrt(jnp.asarray(nalpha_ * nphi_, dtype=jnp.float64))]
    if float(branch_width_weight) != 0.0:
        # Evaluated after the profile term: the summation order of the gradient
        # contributions to ``bnorm`` fixes the last bits of the Jacobian.
        pieces.insert(1, _branch_width_residual(
            bnorm, levels, sqrt_w, weight=branch_width_weight,
            softness=float(branch_width_softness), argmin_softness=argmin_softness))
    if float(shuffle_profile_weight) != 0.0:
        constructed, shuffle_res = _shuffle_residual(
            bnorm, levels, sqrt_w, phi0, phi1, weight=shuffle_profile_weight,
            softness=float(shuffle_profile_softness), argmin_softness=argmin_softness)
        pieces.append(shuffle_res)

    residuals1d = jnp.concatenate(pieces)
    bmin = jnp.min(bmag, axis=(1, 2), keepdims=True)
    bmax = jnp.max(bmag, axis=(1, 2), keepdims=True)
    return {
        "residuals1d": residuals1d,
        "total": jnp.sum(residuals1d * residuals1d),
        "bnorm": bnorm,
        "bmag": bmag,
        "constructed_bmag": bmin + jnp.swapaxes(constructed, 1, 2) * (bmax - bmin),
        "levels": levels,
        "phi": phi,
        "alpha": alpha,
    }


def boozer_modes_from_wout(
    wout,
    *,
    surfaces,
    mboz: int = 18,
    nboz: int = 18,
    jit: bool = False,
) -> dict[str, Any]:
    """Boozer ``|B|`` spectrum of selected surfaces via ``booz_xform_jax``.

    ``wout`` is a :class:`~vmex.core.wout.WoutData` (or any wout-like
    object accepted by ``Booz_xform.read_wout_data``); ``surfaces`` are
    normalized-flux values matched to the nearest half-mesh surfaces.
    Returns ``{bmnc_b, bmns_b, xm_b, xn_b, iota_b, nfp, s_b}`` with the
    spectra shaped ``(nsurf, nmodes)``. ``bmns_b`` is zero for symmetric
    equilibria and contains the independent sine spectrum for ``lasym``.

    ``booz_xform_jax`` is an optional dependency (soft import).
    """
    try:
        from booz_xform_jax import Booz_xform
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Boozer-based objectives require booz_xform_jax; "
            "run `pip install booz_xform_jax`.") from exc
    from .optimize import Equilibrium  # lazy: optimize re-exports this module

    if isinstance(wout, Equilibrium):
        wout = wout.wout
    bx = Booz_xform(verbose=0, mboz=int(mboz), nboz=int(nboz))
    bx.read_wout_data(wout)
    s_in = np.asarray(bx.s_in, dtype=float)
    values = np.atleast_1d(np.asarray(list(np.ravel(surfaces)), dtype=float))
    indices = sorted({int(np.argmin(np.abs(s_in - v))) for v in values})
    bx.compute_surfs = indices
    bx.run(jit=bool(jit))
    bmnc_b = np.asarray(bx.bmnc_b, dtype=float)
    bmns_raw = getattr(bx, "bmns_b", None)
    bmns_b = (np.zeros_like(bmnc_b) if bmns_raw is None
              else np.asarray(bmns_raw, dtype=float))
    xm_b = np.asarray(bx.xm_b, dtype=float)
    if bmnc_b.shape[0] == xm_b.shape[0]:      # (nmodes, nsurf) -> (nsurf, nmodes)
        bmnc_b = bmnc_b.T
    if bmns_b.shape[0] == xm_b.shape[0]:
        bmns_b = bmns_b.T
    return {
        "bmnc_b": bmnc_b,
        "bmns_b": bmns_b,
        "xm_b": xm_b,
        "xn_b": np.asarray(bx.xn_b, dtype=float),
        "iota_b": np.asarray(bx.iota, dtype=float)[indices],
        "nfp": int(bx.nfp),
        "s_b": s_in[indices],
    }


def quasi_isodynamic_residual_from_wout(
    wout,
    *,
    surfaces,
    mboz: int = 18,
    nboz: int = 18,
    jit_booz: bool = False,
    **qi_kwargs,
) -> dict[str, Array]:
    """QI residual of a converged equilibrium: wout -> Boozer -> residual.

    Convenience composition of :func:`boozer_modes_from_wout` and
    :func:`quasi_isodynamic_residual`; ``qi_kwargs`` are the residual's
    sampling/weight knobs.  Accepts a :class:`Equilibrium` too, so it can be
    used directly as a :func:`least_squares` objective term via
    ``lambda eq: quasi_isodynamic_residual_from_wout(eq, surfaces=...)["residuals1d"]``.
    """
    booz = boozer_modes_from_wout(wout, surfaces=surfaces, mboz=mboz, nboz=nboz,
                                  jit=jit_booz)
    return quasi_isodynamic_residual(
        bmnc_b=booz["bmnc_b"], bmns_b=booz["bmns_b"],
        xm_b=booz["xm_b"], xn_b=booz["xn_b"],
        iota_b=booz["iota_b"], nfp=booz["nfp"], **qi_kwargs)


def j_invariant_alpha_derivative_from_boozer(*, alpha, **bounce_options):
    """Return the local action derivative at fixed physical pitch.

    ``alpha`` is a scalar or nonempty one-dimensional array of Boozer
    field-line labels in radians; remaining arguments go to
    :func:`~vmex.core.bounce.bounce_action_from_boozer`. Its diagnostics gain
    ``action_alpha_derivative`` with the shape of ``action``. A JVP shifts
    each independent label, differentiating the sampled field, line element,
    and turning points without pairing wells between labels.

    Invalid wells contain NaN; exclude them from downstream objectives.
    Complete interior wells remain usable in a cut trace (``truncated``).
    Derivatives assume unchanged sampled topology; further parameter AD also
    assumes unchanged interpolation cells. Crossings and quadrature nodes
    changing cells can cause jumps, and near-separatrix derivatives can be
    ill-conditioned beyond the sampled marginal/merged guard. Check spatial,
    label, trace-length, and pitch convergence separately. This supplies no
    population weights or scalar transport measure.
    """
    alpha = jnp.atleast_1d(jnp.asarray(alpha, dtype=jnp.float64))
    if alpha.ndim != 1 or alpha.size == 0:
        raise ValueError("alpha must be scalar or nonempty one-dimensional")

    def actions(labels):
        out = bounce_action_from_boozer(alpha=labels, **bounce_options)
        return out["action"], out

    _, derivative, out = jax.jvp(
        actions, (alpha,), (jnp.ones_like(alpha),), has_aux=True)
    out["action_alpha_derivative"] = jnp.where(
        out["usable_mask"], derivative, jnp.nan)
    return out


def j_invariant_qi_residual_from_boozer(
    *,
    bmnc_b,
    xm_b,
    xn_b,
    iota_b,
    G_b,
    I_b,
    nfp: int,
    pitch,
    bmns_b=None,
    weights: Iterable[float] | None = None,
    nalpha: int = 17,
    points_per_period: int = 128,
    num_periods: int = 4,
    max_wells: int | None = None,
    quadrature_order: int = 64,
) -> dict[str, Array]:
    """Measure pooled complete-well action dispersion at shared physical pitch.

    For each surface and pitch, ``action_mean`` pools every usable well across
    field lines and periods. ``total`` sums the surface-weighted, mean-normalized
    variance about that pooled mean over pitches. Distinct well families can
    therefore give a positive residual even when each family's action is
    independent of the field-line label; this is not a matched-well measure of
    omnigenity or radial drift.

    Edge-cut wells are excluded, while marginal, merged, overflow, or absent
    pitch blocks return NaN instead of a plausible zero.
    """
    bmnc_b = jnp.asarray(bmnc_b, dtype=jnp.float64)
    if bmnc_b.ndim != 2:
        raise ValueError("bmnc_b must have shape (nsurface, nmode)")
    if nalpha < 2:
        raise ValueError("nalpha must be >= 2")
    nsurface = int(bmnc_b.shape[0])
    weight = (jnp.ones(nsurface, dtype=bmnc_b.dtype) if weights is None
              else _as_1d(weights))
    if int(weight.shape[0]) != nsurface:
        raise ValueError("weights must have the same length as surfaces")

    alpha = jnp.arange(nalpha, dtype=bmnc_b.dtype) * (2.0 * jnp.pi / nalpha)
    out = bounce_action_from_boozer(
        bmnc_b=bmnc_b, xm_b=xm_b, xn_b=xn_b, iota_b=iota_b,
        G_b=G_b, I_b=I_b, nfp=nfp, alpha=alpha,
        points_per_period=points_per_period, num_periods=num_periods,
        bmns_b=bmns_b, pitch=pitch, max_wells=max_wells,
        quadrature_order=quadrature_order)
    action, mask = out["action"], out["usable_mask"]
    line_count = jnp.sum(mask, axis=3)
    count = jnp.sum(mask, axis=(1, 3))
    total_action = jnp.sum(jnp.where(mask, action, 0.0), axis=(1, 3))
    mean = total_action / jnp.maximum(count, 1)
    scale = jnp.maximum(jnp.abs(mean), jnp.finfo(bmnc_b.dtype).eps)
    hard_invalid = out["marginal"] | out["merged"] | out["overflow"]
    valid_pitch = jnp.all(line_count > 0, axis=1) & ~jnp.any(
        hard_invalid, axis=1)
    safe_action = jnp.where(mask, action, mean[:, None, :, None])
    residual = jnp.where(
        mask,
        (safe_action - mean[:, None, :, None]) / scale[:, None, :, None]
        * jnp.sqrt(weight[:, None, None, None] / count[:, None, :, None]),
        0.0,
    )
    residual = jnp.where(valid_pitch[:, None, :, None], residual, jnp.nan)
    residuals1d = jnp.ravel(residual)
    out.update({
        "residuals1d": residuals1d,
        "total": jnp.sum(residuals1d * residuals1d),
        "action_mean": mean,
        "valid_pitch": valid_pitch,
        "usable_count": count,
        "usable_count_per_line": line_count,
    })
    return out


class _QIResidualTerm:
    """Shared least-squares objective interface."""

    def residuals_state(self, state, rt):
        return self.compute_state(state, rt)["residuals1d"]

    def total_state(self, state, rt):
        return self.compute_state(state, rt)["total"]

    def J(self, eq):
        return self.residuals_state(eq.state, eq.runtime)

    __call__ = J
    residuals = J

    def total(self, eq):
        return self.total_state(eq.state, eq.runtime)


class ConstructedQIResidual(_QIResidualTerm):
    """Goodman squash-and-shuffle residual with a traceable Boozer transform."""

    name = "constructed_qi"

    def __init__(
        self, surfaces, *, weights=None, mboz=16, nboz=16, oversample=2,
        **residual_options,
    ):
        self.surfaces = np.atleast_1d(np.asarray(surfaces, dtype=float))
        self.weights = None if weights is None else np.asarray(weights, dtype=float)
        if self.weights is not None and self.weights.shape != self.surfaces.shape:
            raise ValueError("weights must have the same length as surfaces")
        self.mboz, self.nboz = int(mboz), int(nboz)
        self.oversample = int(oversample)
        self.residual_options = residual_options

    def compute_state(self, state, rt):
        """Return the constructed-QI residual and Boozer diagnostics."""
        booz = boozer_spectrum_state(
            state, rt, surfaces=self.surfaces, mboz=self.mboz,
            nboz=self.nboz, oversample=self.oversample)
        out = quasi_isodynamic_residual(
            bmnc_b=booz["bmnc_b"], bmns_b=booz.get("bmns_b"),
            xm_b=booz["xm_b"], xn_b=booz["xn_b"],
            iota_b=booz["iota_b"], nfp=booz["nfp"], weights=self.weights,
            **self.residual_options)
        out.update(booz)
        return out


class JInvariantQIResidual(_QIResidualTerm):
    """Pooled complete-well action dispersion at supplied physical pitch.

    See :func:`j_invariant_qi_residual_from_boozer` for its interpretation.
    """

    name = "j_invariant_qi"

    def __init__(
        self, surfaces, pitch, *, weights=None, mboz=16, nboz=16,
        oversample=2, nalpha=17, points_per_period=128, num_periods=4,
        max_wells=None, quadrature_order=64,
    ):
        self.surfaces = np.atleast_1d(np.asarray(surfaces, dtype=float))
        self.pitch = np.atleast_1d(np.asarray(pitch, dtype=float))
        if (self.surfaces.size == 0 or self.pitch.size == 0
                or not np.all(np.isfinite(self.pitch)) or np.any(self.pitch <= 0)):
            raise ValueError("surfaces and positive finite physical pitch are required")
        self.weights = None if weights is None else np.asarray(weights, dtype=float)
        if self.weights is not None and self.weights.shape != self.surfaces.shape:
            raise ValueError("weights must have the same length as surfaces")
        self.mboz, self.nboz = int(mboz), int(nboz)
        self.oversample, self.nalpha = int(oversample), int(nalpha)
        self.points_per_period, self.num_periods = (
            int(points_per_period), int(num_periods))
        self.max_wells = None if max_wells is None else int(max_wells)
        self.quadrature_order = int(quadrature_order)

    def compute_state(self, state, rt):
        """Return action-invariance residuals and topology diagnostics."""
        booz = boozer_spectrum_state(
            state, rt, surfaces=self.surfaces, mboz=self.mboz,
            nboz=self.nboz, oversample=self.oversample)
        out = j_invariant_qi_residual_from_boozer(
            bmnc_b=booz["bmnc_b"], bmns_b=booz.get("bmns_b"),
            xm_b=booz["xm_b"], xn_b=booz["xn_b"],
            iota_b=booz["iota_b"], G_b=booz["G_b"], I_b=booz["I_b"],
            nfp=booz["nfp"], pitch=self.pitch, weights=self.weights,
            nalpha=self.nalpha, points_per_period=self.points_per_period,
            num_periods=self.num_periods, max_wells=self.max_wells,
            quadrature_order=self.quadrature_order)
        out.update(booz)
        return out
