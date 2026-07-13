"""Boozer-spectrum quasi-isodynamic optimization objective.

This module owns the distilled Goodman-style bounce-width and profile
residual plus the optional ``booz_xform_jax`` adapter.  It contains no solver
or optimization-driver state and can be evaluated independently from saved
WOUT data.
"""

from __future__ import annotations

from typing import Any, Iterable, TypedDict

import numpy as np

import jax
import jax.numpy as jnp

from .statephysics import _as_1d

Array = Any


class BoozerModes(TypedDict):
    """Boozer ``|B|`` spectrum and surface metadata."""

    bmnc_b: np.ndarray
    xm_b: np.ndarray
    xn_b: np.ndarray
    iota_b: np.ndarray
    nfp: int
    s_b: np.ndarray

__all__ = [
    "boozer_modes_from_wout",
    "quasi_isodynamic_residual",
    "quasi_isodynamic_residual_from_wout",
]

# ===========================================================================
# Quasi-isodynamic residual (Goodman-style; distilled legacy port)
# ===========================================================================


def _qi_grid(bmnc_b, xm_b, xn_b, iota_b, *, nfp: int, weights, nphi: int,
             nalpha: int, n_bounce: int, include_bounce_endpoints: bool,
             softness: float, phimin: float):
    """Normalized ``|B|`` along field lines + bounce levels (legacy `_qi_boozer_surface_grid`).

    ``theta = alpha + iota * phi`` samples ``nalpha`` field-line labels over
    one field period; ``bnorm`` rescales ``|B|`` to [0, 1] per surface.
    """
    bmnc_b = jnp.asarray(bmnc_b, dtype=jnp.float64)
    xm_b = jnp.asarray(xm_b, dtype=jnp.float64)
    xn_b = jnp.asarray(xn_b, dtype=jnp.float64)
    iota_b = jnp.asarray(iota_b, dtype=jnp.float64)
    if bmnc_b.ndim != 2:
        raise ValueError(f"bmnc_b must have shape (nsurf, nmodes), got {bmnc_b.shape}")
    if nphi < 4 or nalpha < 2 or n_bounce < 2:
        raise ValueError("QI residual requires nphi >= 4, nalpha >= 2, n_bounce >= 2")
    nsurf = int(bmnc_b.shape[0])
    dtype = bmnc_b.dtype
    weights_arr = jnp.ones((nsurf,), dtype=dtype) if weights is None else _as_1d(weights)

    phi0 = jnp.asarray(float(phimin), dtype=dtype)
    phi1 = phi0 + jnp.asarray(2.0 * np.pi / nfp, dtype=dtype)
    phi = jnp.linspace(phi0, phi1, nphi, endpoint=True, dtype=dtype)
    alpha = jnp.linspace(0.0, 2.0 * jnp.pi, nalpha, endpoint=False, dtype=dtype)
    theta = alpha[None, None, :] + iota_b[:, None, None] * phi[None, :, None]
    angle = (theta[:, :, :, None] * xm_b[None, None, None, :]
             - phi[None, :, None, None] * xn_b[None, None, None, :])
    bmag = jnp.sum(bmnc_b[:, None, None, :] * jnp.cos(angle), axis=-1)

    bmin = jnp.min(bmag, axis=(1, 2), keepdims=True)
    bmax = jnp.max(bmag, axis=(1, 2), keepdims=True)
    tiny = jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype)
    bnorm = (bmag - bmin) / jnp.maximum(bmax - bmin, tiny)

    if include_bounce_endpoints:
        levels = jnp.linspace(0.0, 1.0, n_bounce, endpoint=True, dtype=dtype)
    else:
        levels = jnp.linspace(0.0, 1.0, n_bounce + 2, endpoint=True, dtype=dtype)[1:-1]
    eps = jnp.maximum(jnp.asarray(float(softness), dtype=dtype),
                      jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype))
    return weights_arr, phi0, phi1, phi, alpha, bmag, bnorm, levels, eps


def quasi_isodynamic_residual(
    *,
    bmnc_b,
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
    phimin: float = 0.0,
) -> dict[str, Array]:
    """Smooth Goodman-style quasi-isodynamic residual from Boozer ``|B|`` modes.

    A configuration is quasi-isodynamic when the ``|B|`` contours are
    poloidally closed and the trapped-particle bounce distance between the
    two branches of each magnetic well is independent of the field-line label
    ``alpha`` (omnigenity).  This residual samples the normalized ``|B|``
    along field lines ``theta = alpha + iota*phi`` over one field period and
    penalizes, per surface (weights are the legacy defaults, i.e. exactly the
    terms the minimal-seed QI examples used):

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

    Ported from the legacy ``vmec_jax.quasi_isodynamic.objectives.
    quasi_isodynamic_residual_from_boozer_modes`` with the unused
    ``aligned_profile_*`` / ``weighted_shuffle_*`` / ``shuffle_profile_nphi_out``
    machinery removed (they defaulted to off in the QI examples).  ``xn_b``
    uses physical toroidal mode numbers (booz_xform convention).  Returns
    ``residuals1d`` (least-squares vector) and ``total`` (its squared norm).
    """
    (weights_arr, phi0, phi1, phi, alpha, bmag, bnorm, levels, eps) = _qi_grid(
        bmnc_b, xm_b, xn_b, iota_b, nfp=int(nfp), weights=weights, nphi=int(nphi),
        nalpha=int(nalpha), n_bounce=int(n_bounce),
        include_bounce_endpoints=bool(include_bounce_endpoints),
        softness=float(softness), phimin=float(phimin))
    dtype = bnorm.dtype
    nsurf, nphi_, nalpha_ = int(bnorm.shape[0]), int(bnorm.shape[1]), int(bnorm.shape[2])
    nlev = int(levels.shape[0])
    sqrt_w = jnp.sqrt(weights_arr)[:, None, None]
    tiny = jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype)
    pieces: list[jnp.ndarray] = []

    # -- level-set occupancy width variance + profile consistency ----------
    occupancy = jax.nn.sigmoid((levels[None, None, None, :] - bnorm[:, :, :, None]) / eps)
    widths = jnp.mean(occupancy, axis=1)                      # (nsurf, nalpha, nlev)
    width_res = (widths - jnp.mean(widths, axis=1, keepdims=True)) * sqrt_w * width_weight
    pieces.append(jnp.ravel(width_res) / jnp.sqrt(jnp.asarray(nalpha_ * nlev, dtype=dtype)))

    profile_res = (bnorm - jnp.mean(bnorm, axis=2, keepdims=True)) * sqrt_w * profile_weight
    pieces.append(jnp.ravel(profile_res) / jnp.sqrt(jnp.asarray(nalpha_ * nphi_, dtype=dtype)))

    # -- branch-based trapped-well width variance --------------------------
    if float(branch_width_weight) != 0.0:
        bper = jnp.swapaxes(bnorm[:, :-1, :], 1, 2)           # periodic, (nsurf, nalpha, nper)
        nper = nphi_ - 1
        offs = jnp.arange(max(1, nper // 2) + 1, dtype=jnp.int32)
        imin = jnp.argmin(bper, axis=-1)
        left = jnp.maximum.accumulate(
            jnp.take_along_axis(bper, jnp.mod(imin[:, :, None] - offs[None, None, :], nper), axis=-1), axis=-1)
        right = jnp.maximum.accumulate(
            jnp.take_along_axis(bper, jnp.mod(imin[:, :, None] + offs[None, None, :], nper), axis=-1), axis=-1)
        left = (left - left[..., :1]) / jnp.maximum(left[..., -1:] - left[..., :1], tiny)
        right = (right - right[..., :1]) / jnp.maximum(right[..., -1:] - right[..., :1], tiny)
        distance = jnp.asarray(offs, dtype=dtype) / jnp.asarray(nper, dtype=dtype)
        beps = jnp.maximum(jnp.asarray(float(branch_width_softness), dtype=dtype),
                           jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype))

        def crossing(branch):
            logits = -((branch[:, :, :, None] - levels[None, None, None, :]) / beps) ** 2
            logits = logits - jnp.max(logits, axis=2, keepdims=True)
            w = jnp.exp(logits)
            w = w / jnp.sum(w, axis=2, keepdims=True)
            return jnp.sum(w * distance[None, None, :, None], axis=2)

        bw = crossing(left) + crossing(right)                 # (nsurf, nalpha, nlev)
        bw_res = (bw - jnp.mean(bw, axis=1, keepdims=True)) * sqrt_w * branch_width_weight
        pieces.insert(1, jnp.ravel(bw_res) / jnp.sqrt(jnp.asarray(nalpha_ * nlev, dtype=dtype)))

    # -- branch-shuffle profile comparison ----------------------------------
    if float(shuffle_profile_weight) != 0.0:
        b_alpha = jnp.swapaxes(bnorm, 1, 2)                   # (nsurf, nalpha, nphi)
        offs = jnp.arange(nphi_, dtype=jnp.int32)
        offs_f = jnp.asarray(offs, dtype=dtype)
        dphi = (phi1 - phi0) / jnp.asarray(nphi_ - 1, dtype=dtype)
        period = phi1 - phi0
        imin = jnp.argmin(b_alpha, axis=-1)
        li_raw = imin[:, :, None] - offs[None, None, :]
        ri_raw = imin[:, :, None] + offs[None, None, :]
        lvalid, rvalid = li_raw >= 0, ri_raw < nphi_
        lraw = jnp.take_along_axis(b_alpha, jnp.clip(li_raw, 0, nphi_ - 1), axis=-1)
        rraw = jnp.take_along_axis(b_alpha, jnp.clip(ri_raw, 0, nphi_ - 1), axis=-1)
        one = jnp.asarray(1.0, dtype=dtype)
        left = jnp.maximum.accumulate(jnp.where(lvalid, lraw, one), axis=-1)
        right = jnp.maximum.accumulate(jnp.where(rvalid, rraw, one), axis=-1)

        seps = jnp.maximum(jnp.asarray(float(shuffle_profile_softness), dtype=dtype),
                           jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype))
        trapz_w = jnp.ones((nphi_,), dtype=dtype).at[0].set(0.5).at[-1].set(0.5)

        def branch_crossing(branch):
            occ = jax.nn.sigmoid((levels[None, None, None, :] - branch[:, :, :, None]) / seps)
            return jnp.sum(occ * trapz_w[None, None, :, None], axis=2) * dphi

        lcross, rcross = branch_crossing(left), branch_crossing(right)
        bw = lcross + rcross
        bw_mean = jnp.mean(bw, axis=1, keepdims=True)

        min_phi = phi0 + jnp.asarray(imin, dtype=dtype) * dphi
        lend = jnp.maximum(min_phi - phi0, 0.0)
        rend = jnp.maximum(phi1 - min_phi, 0.0)
        signed_phi = (offs_f[None, None, :] - jnp.asarray(imin[:, :, None], dtype=dtype)) * dphi

        level_full = jnp.concatenate([jnp.zeros((1,), dtype=dtype), levels,
                                      jnp.ones((1,), dtype=dtype)])
        y_target = jnp.concatenate([jnp.flip(level_full, axis=0), level_full[1:]], axis=0)

        delta = 0.5 * (bw - bw_mean)
        ltarget = jnp.clip(lcross - delta, 0.0, lend[:, :, None])
        rtarget = jnp.clip(rcross - delta, 0.0, rend[:, :, None])
        zeros = jnp.zeros((nsurf, nalpha_, 1), dtype=dtype)
        lfull = jnp.maximum.accumulate(
            jnp.concatenate([zeros, ltarget, lend[:, :, None]], axis=-1), axis=-1)
        rfull = jnp.maximum.accumulate(
            jnp.concatenate([zeros, rtarget, rend[:, :, None]], axis=-1), axis=-1)
        x_target = jnp.concatenate([-jnp.flip(lfull, axis=-1), rfull[:, :, 1:]], axis=-1)
        ramp = (jnp.arange(x_target.shape[-1], dtype=dtype)
                * jnp.asarray(1.0e-14, dtype=dtype) * period)
        x_target = x_target + ramp[None, None, :]

        def interp_one(xp, x):
            return jnp.interp(x, xp, y_target)

        shuffled = jax.vmap(jax.vmap(interp_one, in_axes=(0, 0)), in_axes=(0, 0))(
            x_target, signed_phi)
        shuffle_res = (shuffled - b_alpha) * sqrt_w * shuffle_profile_weight
        pieces.append(jnp.ravel(shuffle_res)
                      / jnp.sqrt(jnp.asarray(nalpha_ * nphi_, dtype=dtype)))

    residuals1d = jnp.concatenate(pieces)
    return {
        "residuals1d": residuals1d,
        "total": jnp.sum(residuals1d * residuals1d),
        "bnorm": bnorm,
        "bmag": bmag,
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
) -> BoozerModes:
    """Boozer ``|B|`` spectrum of selected surfaces via ``booz_xform_jax``.

    ``wout`` is a :class:`~vmec_jax.core.wout.WoutData` (or any wout-like
    object accepted by ``Booz_xform.read_wout_data``); ``surfaces`` are
    normalized-flux values matched to the nearest half-mesh surfaces.
    Returns ``{bmnc_b, xm_b, xn_b, iota_b, nfp, s_b}`` with ``bmnc_b`` shaped
    ``(nsurf, nmodes)`` — the inputs of :func:`quasi_isodynamic_residual`.

    ``booz_xform_jax`` is an optional dependency (soft import).
    """
    try:
        from booz_xform_jax import Booz_xform
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Boozer-based objectives require booz_xform_jax; "
            "run `pip install booz_xform_jax`.") from exc
    if all(hasattr(wout, name) for name in ("state", "runtime", "wout")):
        wout = wout.wout
    bx = Booz_xform(verbose=0, mboz=int(mboz), nboz=int(nboz))
    bx.read_wout_data(wout)
    s_in = np.asarray(bx.s_in, dtype=float)
    values = np.atleast_1d(np.asarray(list(np.ravel(surfaces)), dtype=float))
    indices = sorted({int(np.argmin(np.abs(s_in - v))) for v in values})
    bx.compute_surfs = indices
    bx.run(jit=bool(jit))
    bmnc_b = np.asarray(bx.bmnc_b, dtype=float)
    xm_b = np.asarray(bx.xm_b, dtype=float)
    if bmnc_b.shape[0] == xm_b.shape[0]:      # (nmodes, nsurf) -> (nsurf, nmodes)
        bmnc_b = bmnc_b.T
    return {
        "bmnc_b": bmnc_b,
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
        bmnc_b=booz["bmnc_b"], xm_b=booz["xm_b"], xn_b=booz["xn_b"],
        iota_b=booz["iota_b"], nfp=booz["nfp"], **qi_kwargs)

