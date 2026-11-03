"""Traceable maximum-J residuals."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

import jax
import jax.numpy as jnp

from .bounce import bounce_action, bounce_action_from_boozer
from .omnigenity import _ResidualTerm, boozer_spectrum_state
from .optimize import quasi_isodynamic_residual
from .qi import j_invariant_qi_residual_from_boozer
from .statephysics import _as_1d

Array = Any

__all__ = [
    "common_trapped_pitches",
    "common_trapped_pitches_state",
    "birth_pitches",
    "ConstructedMaximumJResidual",
    "JInvariantQIAndMaximumJResidual",
    "MaximumJResidual",
    "constructed_maximum_j_residual_from_boozer",
    "DriftConfinementResidual",
    "drift_orbits",
    "drift_surfaces_from_boozer",
    "maximum_j_residual_from_boozer",
    "qi_and_maximum_j_from_boozer",
]


def constructed_maximum_j_residual_from_boozer(
    *,
    bmnc_b,
    xm_b,
    xn_b,
    iota_b,
    G_b,
    I_b,
    nfp: int,
    psi_b,
    psi_edge,
    pitch,
    bmns_b=None,
    weights: Iterable[float] | None = None,
    pitch_weights: Iterable[float] | None = None,
    target: float = 0.0,
    max_wells: int | None = None,
    quadrature_order: int = 64,
    qi_options: dict[str, Any] | None = None,
) -> dict[str, Array]:
    """Penalize outward-increasing action in Goodman's constructed QI field.

    The squash-and-shuffle construction first gives every field-line label
    the mean well width at each field strength. Bounce actions are then
    evaluated in that smoother, approximately omnigenous field at one
    physical pitch shared by all radii. This is the differentiable ``g_J``
    continuation target used to establish a maximum-J direction before the
    resolved :func:`maximum_j_residual_from_boozer` certificate pairs wells
    in the actual field.
    """
    bmnc_b = jnp.asarray(bmnc_b, dtype=jnp.float64)
    nsurface = int(bmnc_b.shape[0])
    if bmnc_b.ndim != 2 or nsurface < 2:
        raise ValueError("bmnc_b needs at least two surfaces")
    weight = (jnp.ones(nsurface, dtype=bmnc_b.dtype) if weights is None
              else _as_1d(weights))
    if weight.shape != (nsurface,):
        raise ValueError("weights must have the same length as surfaces")
    pitch = jnp.atleast_1d(jnp.asarray(pitch, dtype=bmnc_b.dtype))
    pitch_weight = (jnp.ones_like(pitch) if pitch_weights is None
                    else _as_1d(pitch_weights))
    if pitch_weight.shape != pitch.shape:
        raise ValueError("pitch_weights must have the same length as pitch")

    qi = quasi_isodynamic_residual(
        bmnc_b=bmnc_b, bmns_b=bmns_b, xm_b=xm_b, xn_b=xn_b,
        iota_b=iota_b, nfp=nfp, weights=weight, **(qi_options or {}))
    # quasi_isodynamic_residual retains the duplicate field-period endpoint;
    # bounce_action receives the corresponding periodic grid without it.
    constructed_b = jnp.swapaxes(qi["constructed_bmag"][:, :-1, :], 1, 2)
    iota = jnp.asarray(iota_b, dtype=bmnc_b.dtype)
    current_factor = jnp.abs(
        jnp.asarray(G_b, dtype=bmnc_b.dtype)
        + iota * jnp.asarray(I_b, dtype=bmnc_b.dtype))
    bounce = bounce_action(
        constructed_b, pitch,
        dl_dphi=current_factor[:, None, None] / constructed_b,
        length=2.0 * np.pi / int(nfp), periodic=True,
        max_wells=max_wells, quadrature_order=quadrature_order)
    action, usable = bounce["action"], bounce["usable_mask"]
    count = jnp.sum(usable, axis=(1, 3))
    action_mean = jnp.sum(jnp.where(usable, action, 0.0), axis=(1, 3)) / jnp.maximum(count, 1)
    hard_invalid = bounce["marginal"] | bounce["merged"] | bounce["overflow"]
    valid_pitch = (jnp.all(jnp.sum(usable, axis=-1) > 0, axis=1)
                   & ~jnp.any(hard_invalid, axis=1))

    psi = jnp.asarray(psi_b, dtype=bmnc_b.dtype)
    psi_edge = jnp.asarray(psi_edge, dtype=bmnc_b.dtype)
    if psi.shape != (nsurface,):
        raise ValueError("psi_b must have the same length as surfaces")
    dpsi = jnp.diff(psi)
    eps = jnp.finfo(bmnc_b.dtype).eps
    valid_flux = (jnp.all(dpsi * psi_edge > 0.0) & jnp.isfinite(psi_edge)
                  & (jnp.abs(psi_edge) > eps) & jnp.all(jnp.isfinite(dpsi))
                  & jnp.all(jnp.abs(dpsi) > eps))
    valid_pair = valid_pitch[:-1] & valid_pitch[1:] & valid_flux
    dJ_ds = jnp.diff(action_mean, axis=0) / dpsi[:, None] * psi_edge
    mean_action = 0.5 * (jnp.abs(action_mean[:-1]) + jnp.abs(action_mean[1:]))
    relative_slope = dJ_ds / jnp.maximum(mean_action, eps)
    violation = jnp.maximum(relative_slope - float(target), 0.0)
    pair_weight = 0.5 * (weight[:-1] + weight[1:])
    normalized_pitch_weight = pitch_weight / jnp.maximum(jnp.sum(pitch_weight), eps)
    scale = jnp.sqrt(pair_weight[:, None] * normalized_pitch_weight[None, :])
    residual = jnp.where(valid_pair, violation * scale, jnp.nan)
    sample_weight = pair_weight[:, None] * pitch_weight[None, :]
    denominator = jnp.sum(jnp.where(valid_pair, sample_weight, 0.0))
    maximum_j_fraction = jnp.where(
        denominator > 0.0,
        jnp.sum(jnp.where(valid_pair & (dJ_ds < 0.0), sample_weight, 0.0))
        / denominator,
        jnp.nan)
    return {
        "residuals1d": jnp.ravel(residual),
        "total": jnp.sum(residual * residual),
        "dJ_ds": jnp.where(valid_pair, dJ_ds, jnp.nan),
        "relative_slope": jnp.where(valid_pair, relative_slope, jnp.nan),
        "maximum_j_fraction": maximum_j_fraction,
        "valid_pitch_pair": valid_pair,
        "action_mean": action_mean,
        "constructed_bmag": constructed_b,
        "qi": qi,
        **bounce,
    }


def common_trapped_pitches(bmag, trapping_depths=(0.35, 0.55, 0.75)):
    """Choose physical pitches trapped on every sampled field line.

    ``bmag`` has shape ``(surface, distance_along_line, field_line)``. Each
    returned ``lambda=1/B_*`` uses the same field strength ``B_*`` on every
    surface and field-line label, so adjacent radial actions describe the
    same trapped-particle population. ``trapping_depths`` runs from zero at
    the shallow common trapping threshold to one at the deepest threshold.
    """
    bmag = jnp.asarray(bmag)
    depths = jnp.atleast_1d(jnp.asarray(trapping_depths, dtype=bmag.dtype))
    if bmag.ndim != 3 or min(bmag.shape) < 1:
        raise ValueError("bmag must have shape (surface, distance, field_line)")
    if bool(jnp.any(~jnp.isfinite(depths)) | jnp.any((depths <= 0.0) | (depths >= 1.0))):
        raise ValueError("trapping_depths must be finite and strictly between zero and one")
    common_min = jnp.max(jnp.min(bmag, axis=1))
    common_max = jnp.min(jnp.max(bmag, axis=1))
    if bool(~jnp.isfinite(common_min) | ~jnp.isfinite(common_max)
            | (common_min >= common_max)):
        raise ValueError("the sampled field lines have no common trapped-particle pitch")
    b_star = common_max - depths * (common_max - common_min)
    return 1.0 / b_star


def _bmag_lines(booz, nalpha=9, points_per_period=64, num_periods=4):
    """``|B|`` on ``theta = alpha + iota phi`` lines, shape ``(alpha, distance)`` per surface."""
    dtype = jnp.asarray(booz["bmnc_b"]).dtype
    alpha = jnp.asarray(2.0 * np.pi * np.arange(int(nalpha)) / int(nalpha), dtype=dtype)
    nphi = int(points_per_period) * int(num_periods)
    phi = jnp.asarray(
        2.0 * np.pi * np.arange(nphi) / (int(booz["nfp"]) * int(points_per_period)),
        dtype=dtype)
    theta = alpha[None, :, None] + booz["iota_b"][:, None, None] * phi[None, None, :]
    angle = (theta[..., None] * booz["xm_b"]
             - phi[None, None, :, None] * booz["xn_b"])
    bmns = booz.get("bmns_b")
    if bmns is None:
        bmns = jnp.zeros_like(booz["bmnc_b"])
    return (jnp.einsum("sapm,sm->sap", jnp.cos(angle), booz["bmnc_b"])
            + jnp.einsum("sapm,sm->sap", jnp.sin(angle), bmns))


def birth_pitches(bmag, n=6, *, birth=0, margin=0.1, measure="xi"):
    """Pitches and weights from ``v_par/v`` at the field minimum of a birth surface.

    ``bmag`` has shape ``(surface, alpha, distance)``. A particle with
    ``xi0 = v_par/v`` at ``B_min`` of surface ``birth`` mirrors at
    ``B_* = B_min / (1 - xi0**2)``, so ``lambda = 1/B_* = (1 - xi0**2)/B_min``.
    The ``n`` Gauss-Legendre nodes in ``xi0`` span the window trapped on every
    sampled surface and line: ``B_*`` runs from the largest line minimum to
    ``margin`` of the common well depth below the smallest line maximum, so
    barely trapped pitches whose wells vanish on outer surfaces are dropped.
    Uniform ``xi0`` weights are the isotropic measure and emphasize deep
    trapping; ``measure="trapped"`` multiplies them by the trapped density per
    ``lambda``, ``<B / sqrt(1 - lambda B)>`` over the birth surface.
    Returns ``(pitch, weights)``, the weights summing to one.
    """
    bmag = jnp.asarray(bmag)
    b_min = jnp.min(bmag[birth])
    lo, hi = jnp.max(jnp.min(bmag, axis=-1)), jnp.min(jnp.max(bmag, axis=-1))
    lo, hi = lo + 0.02 * (hi - lo), hi - margin * (hi - lo)
    xi_lo = jnp.sqrt(jnp.maximum(1.0 - b_min / lo, 0.0))
    xi_hi = jnp.sqrt(jnp.maximum(1.0 - b_min / hi, 0.0))
    node, weight = np.polynomial.legendre.leggauss(int(n))
    xi0 = xi_lo + 0.5 * (node + 1.0) * (xi_hi - xi_lo)
    pitch = (1.0 - xi0 ** 2) / b_min
    weight = jnp.asarray(weight, dtype=bmag.dtype)
    if measure == "trapped":
        b = bmag[birth].ravel()[None, :]
        density = jnp.mean(jnp.where(pitch[:, None] * b < 1.0, b / jnp.sqrt(
            jnp.maximum(1.0 - pitch[:, None] * b, 1.0e-6)), 0.0), axis=1)
        weight = weight * xi0 * density
    return pitch, weight / jnp.sum(weight)


def common_trapped_pitches_state(
    state, rt, surfaces, trapping_depths=(0.35, 0.55, 0.75), *,
    mboz=12, nboz=12, nalpha=9, points_per_period=64, num_periods=4,
):
    """Choose common physical pitches directly from a converged equilibrium.

    The sampled field lines use ``theta=alpha+iota*phi`` in Boozer angles.
    A single ``1/lambda`` is selected from the trapping interval shared by
    every surface and field-line label; this prevents radial maximum-J
    comparisons from silently changing the particle population.
    """
    bmag = _bmag_lines(boozer_spectrum_state(
        state, rt, surfaces=surfaces, mboz=mboz, nboz=nboz),
        nalpha, points_per_period, num_periods)
    return common_trapped_pitches(jnp.swapaxes(bmag, 1, 2), trapping_depths)


def maximum_j_residual_from_boozer(
    *,
    bmnc_b,
    xm_b,
    xn_b,
    iota_b,
    G_b,
    I_b,
    nfp: int,
    psi_b,
    psi_edge,
    pitch,
    bmns_b=None,
    weights: Iterable[float] | None = None,
    pitch_weights: Iterable[float] | None = None,
    target: float = 0.0,
    nalpha: int = 17,
    points_per_period: int = 128,
    num_periods: int = 4,
    max_wells: int | None = None,
    quadrature_order: int = 64,
    match_tolerance: float | None = None,
) -> dict[str, Array]:
    """Penalize outward-increasing ``J`` at common physical pitch.

    Wells on adjacent surfaces are paired only when they are reciprocal
    nearest neighbours in Boozer toroidal angle. Every line must retain the
    same nonzero number of complete, usable wells on both surfaces. Ambiguous
    topology, a nonmonotone signed flux coordinate, or an excessive matching
    displacement invalidates the pitch block with NaN.

    ``target`` is the upper bound on the dimensionless logarithmic slope
    ``psi_edge / J * dJ/dpsi = (1/J) dJ/ds``, where
    ``s = psi / psi_edge`` increases from axis to boundary. The default
    enforces the maximum-J condition ``dJ/ds <= 0`` independently of VMEC's
    signed toroidal-flux convention. Reported
    fractions use uniform pitch-sample weights unless ``pitch_weights``
    supplies user quadrature weights. They are resolved-orbit summaries, not
    the bounce-time-weighted Maxwellian phase-space fraction.
    """
    bmnc_b = jnp.asarray(bmnc_b, dtype=jnp.float64)
    psi = jnp.atleast_1d(jnp.asarray(psi_b, dtype=bmnc_b.dtype))
    if bmnc_b.ndim != 2 or int(bmnc_b.shape[0]) < 2:
        raise ValueError("bmnc_b needs at least two surfaces")
    nsurface = int(bmnc_b.shape[0])
    if psi.shape != (nsurface,):
        raise ValueError("psi_b must have the same length as the Boozer surfaces")
    if nalpha < 2:
        raise ValueError("nalpha must be >= 2")
    weight = (jnp.ones(nsurface, dtype=bmnc_b.dtype) if weights is None
              else _as_1d(weights))
    if int(weight.shape[0]) != nsurface:
        raise ValueError("weights must have the same length as surfaces")
    pitch = jnp.atleast_1d(jnp.asarray(pitch, dtype=bmnc_b.dtype))
    pitch_weight = (jnp.ones_like(pitch) if pitch_weights is None
                    else _as_1d(pitch_weights))
    if pitch_weight.shape != pitch.shape:
        raise ValueError("pitch_weights must have the same length as pitch")

    alpha = jnp.arange(nalpha, dtype=bmnc_b.dtype) * (2.0 * jnp.pi / nalpha)
    out = bounce_action_from_boozer(
        bmnc_b=bmnc_b, xm_b=xm_b, xn_b=xn_b, iota_b=iota_b,
        G_b=G_b, I_b=I_b, nfp=nfp, alpha=alpha,
        points_per_period=points_per_period, num_periods=num_periods,
        bmns_b=bmns_b, pitch=pitch, max_wells=max_wells,
        quadrature_order=quadrature_order)
    action, usable = out["action"], out["usable_mask"]
    center = 0.5 * (out["well_left"] + out["well_right"])
    lo_center, hi_center = center[:-1], center[1:]
    lo_usable, hi_usable = usable[:-1], usable[1:]
    distance = jnp.abs(lo_center[..., :, None] - hi_center[..., None, :])
    candidates = lo_usable[..., :, None] & hi_usable[..., None, :]
    infinity = jnp.asarray(jnp.inf, dtype=bmnc_b.dtype)
    candidate_distance = jnp.where(candidates, distance, infinity)
    hi_index = jnp.argmin(candidate_distance, axis=-1)
    lo_index = jnp.argmin(candidate_distance, axis=-2)
    matched_hi_usable = jnp.take_along_axis(hi_usable, hi_index, axis=-1)
    reciprocal = jnp.take_along_axis(lo_index, hi_index, axis=-1)
    reciprocal = reciprocal == jnp.arange(action.shape[-1])
    match_distance = jnp.take_along_axis(
        candidate_distance, hi_index[..., None], axis=-1)[..., 0]
    tolerance = (np.pi / float(nfp) if match_tolerance is None
                 else float(match_tolerance))
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("match_tolerance must be positive and finite")
    matched = (
        lo_usable & matched_hi_usable & reciprocal
        & (match_distance <= tolerance)
    )

    hard_invalid = out["marginal"] | out["merged"] | out["overflow"]
    lo_count = jnp.sum(lo_usable, axis=-1)
    hi_count = jnp.sum(hi_usable, axis=-1)
    valid_line = (
        (lo_count > 0) & (lo_count == hi_count)
        & (jnp.sum(matched, axis=-1) == lo_count)
        & ~(hard_invalid[:-1] | hard_invalid[1:])
    )
    valid_pitch = jnp.all(valid_line, axis=1)

    dpsi = jnp.diff(psi)
    psi_edge = jnp.asarray(psi_edge, dtype=bmnc_b.dtype)
    eps = jnp.finfo(bmnc_b.dtype).eps
    outward = jnp.all(dpsi * psi_edge > 0.0)
    valid_flux = (
        outward & jnp.isfinite(psi_edge) & (jnp.abs(psi_edge) > eps)
        & jnp.isfinite(dpsi) & (jnp.abs(dpsi) > eps)
    )
    valid_pitch &= valid_flux[:, None]

    hi_action = jnp.take_along_axis(action[1:], hi_index, axis=-1)
    lo_action = jnp.where(matched, action[:-1], 1.0)
    hi_action = jnp.where(matched, hi_action, 1.0)
    dJ_dpsi = (hi_action - lo_action) / dpsi[:, None, None, None]
    mean_action = 0.5 * (jnp.abs(lo_action) + jnp.abs(hi_action))
    dJ_ds = dJ_dpsi * psi_edge
    relative_slope = dJ_ds / jnp.maximum(
        mean_action, eps)
    violation = jnp.maximum(relative_slope - float(target), 0.0)
    count = jnp.sum(matched, axis=(1, 3))
    pair_weight = 0.5 * (weight[:-1] + weight[1:])
    scale = jnp.sqrt(
        pair_weight[:, None] / jnp.maximum(count, 1))[:, None, :, None]
    residual = jnp.where(matched, violation * scale, 0.0)
    residual = jnp.where(valid_pitch[:, None, :, None], residual, jnp.nan)
    residuals1d = jnp.ravel(residual)
    sample_weight = pair_weight[:, None, None, None] * pitch_weight[
        None, None, :, None]

    def fraction(select, group=True):
        group = jnp.asarray(group)
        denominator = jnp.sum(jnp.where(matched & group, sample_weight, 0.0))
        numerator = jnp.sum(jnp.where(matched & group & select, sample_weight, 0.0))
        return jnp.where(denominator > 0.0, numerator / denominator, jnp.nan)

    maximum_j = dJ_ds < 0.0
    bmin = jnp.min(out["bmag"], axis=-1)
    bmax = jnp.max(out["bmag"], axis=-1)
    trapping_depth = jnp.clip(
        (bmax[..., None] - 1.0 / pitch[None, None, :])
        / jnp.maximum(bmax[..., None] - bmin[..., None], eps),
        0.0, 1.0)
    pair_depth = 0.5 * (trapping_depth[:-1] + trapping_depth[1:])
    shallow = pair_depth[..., None] < 0.5
    deep = pair_depth[..., None] >= 0.5
    out.update({
        "residuals1d": residuals1d,
        "total": jnp.sum(residuals1d * residuals1d),
        "dJ_dpsi": jnp.where(matched, dJ_dpsi, jnp.nan),
        "dJ_ds": jnp.where(matched, dJ_ds, jnp.nan),
        "relative_slope": jnp.where(matched, relative_slope, jnp.nan),
        "matched_well_mask": matched,
        "matched_well_index": hi_index,
        "match_distance": jnp.where(matched, match_distance, jnp.nan),
        "valid_pitch_pair": valid_pitch,
        "maximum_j_fraction": fraction(maximum_j),
        "target_fraction": fraction(relative_slope <= float(target)),
        "shallow_maximum_j_fraction": fraction(maximum_j, shallow),
        "deep_maximum_j_fraction": fraction(maximum_j, deep),
        "excluded_pitch_fraction": 1.0 - jnp.mean(valid_pitch),
        "trapping_depth": pair_depth,
        "psi_b": psi,
        "psi_edge": psi_edge,
    })
    return out


def qi_and_maximum_j_from_boozer(
    state,
    rt,
    *,
    surfaces,
    pitch,
    weights: Iterable[float] | None = None,
    mboz: int = 16,
    nboz: int = 16,
    oversample: int = 2,
    qi_options: dict[str, Any] | None = None,
    maxj_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate pooled action dispersion and maximum-J from one Boozer pass.

    :class:`~vmex.core.qi.JInvariantQIResidual` and :class:`MaximumJResidual`
    each run their own :func:`~vmex.core.omnigenity.boozer_spectrum_state`
    transform — the dominant cost when both objectives share the same
    surfaces and physical pitch (the usual campaign layout).  This helper
    runs the transform once and feeds both functional layers.

    ``qi_options`` / ``maxj_options`` forward extra keyword arguments to
    :func:`vmex.core.qi.j_invariant_qi_residual_from_boozer` and
    :func:`maximum_j_residual_from_boozer` (``nalpha``,
    ``points_per_period``, ``num_periods``, ``max_wells``, ``target``, ...).
    Returns ``{"boozer", "qi", "maximum_j"}``; the two residual entries are
    identical to the corresponding independent class evaluations.
    """
    booz = boozer_spectrum_state(
        state, rt, surfaces=surfaces, mboz=int(mboz), nboz=int(nboz),
        oversample=int(oversample))
    common = dict(
        bmnc_b=booz["bmnc_b"], bmns_b=booz.get("bmns_b"),
        xm_b=booz["xm_b"], xn_b=booz["xn_b"],
        iota_b=booz["iota_b"], G_b=booz["G_b"], I_b=booz["I_b"],
        nfp=booz["nfp"], pitch=pitch, weights=weights)
    qi_out = j_invariant_qi_residual_from_boozer(
        **common, **(qi_options or {}))
    maxj_out = maximum_j_residual_from_boozer(
        **common, psi_b=booz["psi_b"], psi_edge=booz["psi_edge"],
        **(maxj_options or {}))
    return {"boozer": booz, "qi": qi_out, "maximum_j": maxj_out}


class JInvariantQIAndMaximumJResidual(_ResidualTerm):
    """Joint pooled action dispersion and maximum-J rows from one Boozer transform."""

    name = "j_invariant_qi_and_maximum_j"

    def __init__(
        self, surfaces, pitch, *, weights=None, qi_weight=1.0, maxj_weight=1.0,
        mboz=16, nboz=16, oversample=2, qi_options=None, maxj_options=None,
    ):
        self.surfaces = np.atleast_1d(np.asarray(surfaces, dtype=float))
        self.pitch = np.atleast_1d(np.asarray(pitch, dtype=float))
        if (self.surfaces.size < 2 or np.any(~np.isfinite(self.surfaces))
                or np.any(np.diff(self.surfaces) <= 0.0)
                or self.pitch.size == 0 or np.any(~np.isfinite(self.pitch))
                or np.any(self.pitch <= 0.0)):
            raise ValueError(
                "increasing finite surfaces and positive finite pitch are required")
        self.weights = None if weights is None else np.asarray(weights, dtype=float)
        if self.weights is not None:
            if self.weights.shape != self.surfaces.shape:
                raise ValueError("weights must have the same length as surfaces")
            if np.any(~np.isfinite(self.weights)) or np.any(self.weights < 0.0):
                raise ValueError("weights must be finite and non-negative")
        if not np.isfinite(qi_weight) or not np.isfinite(maxj_weight):
            raise ValueError("qi_weight and maxj_weight must be finite")
        if qi_weight < 0.0 or maxj_weight < 0.0:
            raise ValueError("qi_weight and maxj_weight must be non-negative")
        self.qi_scale, self.maxj_scale = np.sqrt(qi_weight), np.sqrt(maxj_weight)
        self.mboz, self.nboz, self.oversample = int(mboz), int(nboz), int(oversample)
        if min(self.mboz, self.nboz, self.oversample) <= 0:
            raise ValueError("mboz, nboz, and oversample must be positive")
        self.qi_options = {} if qi_options is None else dict(qi_options)
        self.maxj_options = {} if maxj_options is None else dict(maxj_options)

    def compute_state(self, state, rt):
        """Return the ``{"boozer", "qi", "maximum_j"}`` dictionary of one Boozer transform."""
        return qi_and_maximum_j_from_boozer(
            state, rt, surfaces=self.surfaces, pitch=self.pitch, weights=self.weights,
            mboz=self.mboz, nboz=self.nboz, oversample=self.oversample,
            qi_options=self.qi_options, maxj_options=self.maxj_options,
        )

    def residuals_state(self, state, rt):
        """Return ``sqrt(qi_weight)`` QI rows followed by ``sqrt(maxj_weight)`` maximum-J rows."""
        out = self.compute_state(state, rt)
        return jnp.concatenate([
            self.qi_scale * out["qi"]["residuals1d"],
            self.maxj_scale * out["maximum_j"]["residuals1d"],
        ])

    def total_state(self, state, rt):
        """Return the sum of squared residual rows."""
        rows = self.residuals_state(state, rt)
        return jnp.vdot(rows, rows)


class MaximumJResidual(_ResidualTerm):
    """Composable maximum-J objective at fixed physical pitch."""

    name = "maximum_j"
    _from_boozer = staticmethod(maximum_j_residual_from_boozer)

    def __init__(
        self, surfaces, pitch, *, weights=None, mboz=16, nboz=16,
        oversample=2, **residual_options,
    ):
        self.surfaces = np.atleast_1d(np.asarray(surfaces, dtype=float))
        self.pitch = np.atleast_1d(np.asarray(pitch, dtype=float))
        if (self.surfaces.size < 2 or np.any(np.diff(self.surfaces) <= 0.0)
                or self.pitch.size == 0 or not np.all(np.isfinite(self.pitch))
                or np.any(self.pitch <= 0.0)):
            raise ValueError(
                "increasing surfaces and positive finite physical pitch are required")
        self.weights = None if weights is None else np.asarray(weights, dtype=float)
        if self.weights is not None and self.weights.shape != self.surfaces.shape:
            raise ValueError("weights must have the same length as surfaces")
        self.mboz, self.nboz, self.oversample = int(mboz), int(nboz), int(oversample)
        self.residual_options = residual_options

    def compute_state(self, state, rt):
        """Return the residuals, actions, and radial-slope diagnostics."""
        booz = self._boozer(state, rt)
        out = self._from_boozer(
            bmnc_b=booz["bmnc_b"], bmns_b=booz.get("bmns_b"),
            xm_b=booz["xm_b"], xn_b=booz["xn_b"],
            iota_b=booz["iota_b"], G_b=booz["G_b"], I_b=booz["I_b"],
            nfp=booz["nfp"], psi_b=booz["psi_b"], psi_edge=booz["psi_edge"],
            pitch=self.pitch, weights=self.weights, **self.residual_options)
        out.update(booz)
        return out


class ConstructedMaximumJResidual(MaximumJResidual):
    """Goodman constructed-field maximum-J continuation objective.

    Use this smooth target to establish a favorable radial-action direction,
    then certify the resulting actual field with :class:`MaximumJResidual`.
    """

    name = "constructed_maximum_j"
    _from_boozer = staticmethod(constructed_maximum_j_residual_from_boozer)


def drift_orbits(action, bounce_time, s, alpha, s0, alpha0=0.0, *,
                 t_final=None, length=None, nsteps: int = 256,
                 degree: int = 3, clamp: bool = False) -> dict[str, Array]:
    """Integrate the bounce-averaged drift ``(s(t), alpha(t))`` of one pitch.

    With ``J`` the action and ``tau`` the bounce time, both per unit speed
    (the ``action`` and ``bounce_time`` of :func:`bounce_action`, sampled on
    surfaces ``s = psi/psi_edge`` and a uniform ``alpha`` grid), the
    bounce-centre equations of motion are

    ``ds/dt = (dJ/dalpha) / tau``,  ``dalpha/dt = -(dJ/ds) / tau``,

    with dimensionless time ``t = t_phys m v**2 / (Z e psi_edge)``, the signs
    of a positive charge in ``B = grad(psi) x grad(alpha)``; the opposite
    charge or orientation reverses the direction, not the orbit. Their
    ratio ``ds/dalpha = -(dJ/dalpha) / (dJ/ds)`` is the action-transport law
    ``dJ/dalpha = 0``, so orbits follow ``J`` contours: the drift surfaces.

    ``J`` and ``tau`` are fitted by a Fourier series in ``alpha`` times a
    least-squares polynomial of ``degree`` in ``s``; surfaces with any
    non-finite sample get zero weight. The fit is evaluated analytically and
    integrated with ``nsteps`` RK4 steps, all in ``jnp``, so every output is
    differentiable in the inputs. ``t_final=None`` closes one turn of all
    but the slowest tenth of the grid, ``2 pi / percentile_10 |dalpha/dt|``.
    With ``length`` the contour is traced by arc length in the
    ``(s cos alpha, s sin alpha)`` plane instead, and ``t`` is that arc
    length. ``tau`` then drops out, every orbit gets the same resolution
    whatever its drift speed, and ``s_max`` is the reach of the drift surface
    along ``length``; time-based reach needs many RK4 steps per precession
    turn and ill-conditions the gradient. The LCFS absorbs: past ``s = 1`` the drift fades out over ``0.05`` in
    ``s``, so a lost orbit stops there rather than following the
    extrapolated fit. Each initial condition ``(s0, alpha0)`` (broadcast
    together) gives one orbit; ``s_max`` is its largest radius, and
    ``action`` the fitted ``J`` along it, constant up to the integration
    error inside the plasma. ``clamp=True`` evaluates the fit at the radius
    clipped to the finite surfaces, so an orbit that leaves the window where
    the well exists meets frozen edge fields instead of the polynomial's
    extrapolation.
    """
    action, bounce_time = jnp.asarray(action), jnp.asarray(bounce_time)
    dtype = action.dtype
    s = jnp.asarray(s, dtype=dtype)
    nalpha = int(action.shape[1])
    finite = jnp.all(jnp.isfinite(action) & jnp.isfinite(bounce_time), axis=1)
    weight = finite.astype(dtype)[:, None]
    power = jnp.arange(degree + 1)
    fit = jnp.linalg.pinv(weight * s[:, None] ** power)

    def coefficients(values):  # (degree + 1, nk) complex polynomial x Fourier
        spectrum = jnp.fft.rfft(jnp.where(finite[:, None], values, 0.0), axis=1) / nalpha
        k = jnp.arange(spectrum.shape[1])
        spectrum = spectrum * jnp.where((k == 0) | (2 * k == nalpha), 1.0, 2.0)
        return fit @ (weight * spectrum)

    cj, ct = coefficients(action), coefficients(bounce_time)
    k = jnp.arange(cj.shape[1])
    alpha_offset = jnp.asarray(alpha, dtype=dtype)[0]
    s_lo = jnp.min(jnp.where(finite, s, jnp.inf))
    s_hi = jnp.max(jnp.where(finite, s, -jnp.inf))

    def fields(y):
        radius, angle = y[0], y[1] - alpha_offset
        if clamp:
            radius = jnp.clip(radius, s_lo, s_hi)
        sp = radius ** power
        dsp = jnp.where(power > 0, power * radius ** jnp.maximum(power - 1, 0), 0.0)
        phase = jnp.exp(1j * k * angle)
        j_k, tau_k = sp @ cj, sp @ ct
        return (jnp.real(j_k @ phase), jnp.real((dsp @ cj) @ phase),
                jnp.real((1j * k * j_k) @ phase), jnp.real(tau_k @ phase))

    def velocity(y):
        _, j_s, j_alpha, tau = fields(y)
        return jnp.stack([j_alpha, -j_s]) / tau

    def rhs(y):
        # The LCFS absorbs: the drift fades out over 0.02 in s beyond it, so a
        # lost orbit stops near s = 1 instead of following an extrapolated J.
        v = velocity(y)
        if length is not None:  # unit speed in the (s cos alpha, s sin alpha) plane
            # slow down near stagnation points of J, where the direction is undefined
            v = v / jnp.sqrt(v[0] ** 2 + (y[0] * v[1]) ** 2 + (0.1 * speed) ** 2)
        return (0.5 - 0.5 * jnp.tanh((jnp.abs(y[0]) - 1.0) / 0.02)) * v

    grid = jnp.stack(jnp.broadcast_arrays(
        s[:, None], jnp.asarray(alpha, dtype=dtype)[None, :]), -1).reshape(-1, 2)
    grid_velocity = jax.vmap(velocity)(grid)
    usable = jnp.repeat(finite, nalpha)
    speed = jnp.nanmedian(jnp.where(usable, jnp.hypot(
        grid_velocity[:, 0], grid[:, 0] * grid_velocity[:, 1]), jnp.nan))
    if length is not None:
        t_final = length
    elif t_final is None:
        rate = jnp.where(usable, jnp.abs(grid_velocity[:, 1]), jnp.nan)
        t_final = 2.0 * jnp.pi / jnp.nanpercentile(rate, 10.0)
    dt = jnp.asarray(t_final, dtype=dtype) / nsteps

    def step(y, _):
        k1 = rhs(y); k2 = rhs(y + 0.5 * dt * k1)
        k3 = rhs(y + 0.5 * dt * k2); k4 = rhs(y + dt * k3)
        y = y + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return y, y

    s0, alpha0 = jnp.broadcast_arrays(jnp.atleast_1d(jnp.asarray(s0, dtype=dtype)),
                                      jnp.asarray(alpha0, dtype=dtype))
    y0 = jnp.stack([s0, alpha0], -1)
    path = jax.vmap(lambda y: jnp.concatenate(
        [y[None], jax.lax.scan(step, y, None, length=nsteps)[1]]))(y0)
    return {
        "t": dt * jnp.arange(nsteps + 1),
        "s": path[..., 0],
        "alpha": path[..., 1],
        "s_max": jnp.max(path[..., 0], axis=-1),
        "action": jax.vmap(jax.vmap(lambda y: fields(y)[0]))(path),
        "t_final": dt * nsteps,
    }


def drift_surfaces_from_boozer(
    *, bmnc_b, xm_b, xn_b, iota_b, G_b, I_b, nfp: int, s_b, pitch, s0,
    alpha0=0.0, bmns_b=None, t_final=None, length=None, nsteps: int = 256, degree: int = 3,
    nalpha: int = 32, points_per_period: int = 64, num_periods: int = 12,
    max_wells: int | None = None, quadrature_order: int = 32, clamp: bool = False,
) -> dict[str, Array]:
    """Drift surfaces of one physical ``pitch`` from a Boozer spectrum.

    Evaluates ``J`` and the bounce time on every surface ``s_b`` and
    ``nalpha`` field lines with :func:`~vmex.core.bounce.bounce_action_from_boozer`
    (on each traced line, the deepest usable well starting within one field
    period of the first, as in the ``vmex --plot`` J map) and integrates them
    with :func:`drift_orbits`. The grids are returned as ``j_map``/``tau_map``
    next to the orbits, with the ``mirror`` ratio ``(Bmax - Bmin)/(Bmax + Bmin)``
    of the traced lines on every surface. The traced
    line must cover a poloidal transit, about ``nfp / iota`` periods.
    """
    alpha = jnp.arange(nalpha) * (2.0 * jnp.pi / nalpha)
    out = bounce_action_from_boozer(
        bmnc_b=bmnc_b, bmns_b=bmns_b, xm_b=xm_b, xn_b=xn_b, iota_b=iota_b,
        G_b=G_b, I_b=I_b, nfp=nfp, alpha=alpha, pitch=jnp.atleast_1d(pitch)[:1],
        points_per_period=points_per_period, num_periods=num_periods,
        max_wells=2 * num_periods if max_wells is None else max_wells,
        quadrature_order=quadrature_order, drift_integrands={})
    # Later wells on a traced line sit at other field-line labels, so J(alpha)
    # is the deepest (largest-J) usable well starting within one field period
    # of the first; averaging all wells erases the alpha dependence that
    # moves the orbit, and the first well alone is often a shallow ripple.
    usable = out["usable_mask"][:, :, 0]
    left = jnp.where(usable, out["well_left"][:, :, 0], jnp.inf)
    usable &= left < jnp.min(left, axis=-1, keepdims=True) + 2.0 * jnp.pi / nfp
    first = jnp.argmax(jnp.where(usable, out["action"][:, :, 0], -jnp.inf), axis=-1)[..., None]

    def first_well(values):
        value = jnp.take_along_axis(values[:, :, 0], first, axis=-1)[..., 0]
        return jnp.where(jnp.any(usable, axis=-1), value, jnp.nan)

    j_map, tau_map = first_well(out["action"]), first_well(out["bounce_time"])
    orbits = drift_orbits(j_map, tau_map, s_b, alpha, s0, alpha0,
                          t_final=t_final, length=length, nsteps=nsteps, degree=degree,
                          clamp=clamp)
    b_max, b_min = jnp.max(out["bmag"], axis=(1, 2)), jnp.min(out["bmag"], axis=(1, 2))
    return {**orbits, "j_map": j_map, "tau_map": tau_map, "alpha_grid": alpha,
            "s_grid": jnp.asarray(s_b), "mirror": (b_max - b_min) / (b_max + b_min)}


class DriftConfinementResidual(MaximumJResidual):
    """Smooth drift-loop reach of trapped particles, after Sengupta et al.

    For each physical ``pitch`` (weighted by ``pitch_weights``, e.g. the
    trapped measure of each pitch), bounce-centre orbits are launched at
    every ``s0`` and ``launch_alpha`` field-line labels and followed for
    ``t_final`` with :func:`drift_surfaces_from_boozer`. One row per orbit is
    the reach

    ``R = tanh(max(s_hat - s0, 0) / (1 - s0))``,

    with ``s_hat = (logsumexp(sharpness * s) - ln(n_t)) / sharpness`` the
    smooth maximum radius over the ``n_t`` orbit samples. It lies between
    the mean and the maximum of ``s``, so a drift surface that stays on its
    flux surface has zero reach, one that reaches the LCFS about 0.76, and
    the reach is bounded beyond. The weighted
    mean of ``R``, returned as ``gamma_w``, is a smooth version of the reach
    proxy ``Gamma_W`` of arXiv:2608.20042. Over traced alpha losses of 13
    equilibria its rank correlation was 0.97, against 0.62 for the
    excursion rows.

    ``excursion_weight`` adds the rows
    ``max(|J - <J>| / (|d<J>/ds| + floor |<J>|) - (s_limit - s), 0)`` on
    every sampled surface and line; they need no orbit integration but
    reward a large ``|d<J>/ds|``, so keep them a weak regulariser. Orbits
    whose launch point has no trapped well add nothing: those particles
    are passing. Orbits are traced by arc length (``length=8``, see
    :func:`drift_orbits`), which reaches the end of every drift surface once
    and keeps the gradient well conditioned; pass ``t_final`` to follow
    physical time instead. The bounce-averaged model has no finite orbit
    width, so keep ``s0`` well inside and cross-check with
    guiding-centre tracing.

    Three optional terms, all off by default, address what the bounce-averaged
    reach cannot see. ``mirror_target`` adds a hinge
    ``mirror_weight * max(m - mirror_target, 0)`` on the mirror ratio
    ``m = (Bmax - Bmin)/(Bmax + Bmin)`` of every sampled surface, the form of
    :func:`~vmex.core.optimize.mirror_ratio` and of the usual
    ``mirror_excess`` residual. ``fow`` is the alpha gyroradius over the minor
    radius at the target scale (about 0.027 for 3.5 MeV at 5.9 T, ``a`` =
    1.7 m); an orbit then counts as lost, with a logistic of width 0.05, when
    ``s_hat + delta_s >= 1`` for the banana width
    ``delta_s = 2 sqrt(s) fow q / sqrt(eps)``, ``q = 1/|iota|``,
    ``eps = sqrt(s) a/R`` with ``R = G/B00``, ``a = sqrt(2 psi_edge/B00)``. The
    reach and these rows are different pictures of one loss, so the weighted
    mean of the second is returned as ``gamma_fow``. ``clamp`` is passed to
    :func:`drift_orbits`.
    """

    name = "drift_confinement"

    def __init__(self, surfaces, pitch, s0, *, pitch_weights=None, launch_alpha=8,
                 sharpness=50.0, excursion_weight=0.0, s_limit=1.0, floor=0.05,
                 mirror_target=None, mirror_weight=1.0, fow=None, fow_weight=1.0,
                 mboz=16, nboz=16, oversample=2, **drift_options):
        super().__init__(surfaces, pitch, mboz=mboz, nboz=nboz, oversample=oversample)
        alpha0 = np.arange(int(launch_alpha)) * (2.0 * np.pi / int(launch_alpha))
        s0, alpha0 = np.meshgrid(np.atleast_1d(np.asarray(s0, dtype=float)), alpha0)
        self.s0, self.alpha0 = s0.ravel(), alpha0.ravel()
        weight = np.ones(self.pitch.size) if pitch_weights is None else np.asarray(pitch_weights, float)
        self.row_scale = np.sqrt(weight / weight.sum() / self.s0.size)
        self.sharpness, self.excursion_weight = float(sharpness), float(excursion_weight)
        self.s_limit, self.floor = float(s_limit), float(floor)
        self.mirror_target, self.mirror_weight = mirror_target, float(mirror_weight)
        self.fow, self.fow_weight = fow, float(fow_weight)
        if "t_final" not in drift_options:
            drift_options.setdefault("length", 8.0)
        self.residual_options = drift_options

    def compute_state(self, state, rt):
        """Return the reach rows (and optional excursion rows) of every pitch."""
        booz = self._boozer(state, rt)
        s_b = booz["psi_b"] / booz["psi_edge"]
        orbits = [drift_surfaces_from_boozer(
            bmnc_b=booz["bmnc_b"], bmns_b=booz.get("bmns_b"), xm_b=booz["xm_b"],
            xn_b=booz["xn_b"], iota_b=booz["iota_b"], G_b=booz["G_b"],
            I_b=booz["I_b"], nfp=booz["nfp"], s_b=s_b, pitch=p, s0=self.s0,
            alpha0=self.alpha0, **self.residual_options) for p in self.pitch]
        path = jnp.stack([o["s"] for o in orbits])               # (pitch, orbit, time)
        s_hat = (jax.nn.logsumexp(self.sharpness * path, axis=-1)
                 - np.log(path.shape[-1])) / self.sharpness
        reach = jnp.tanh(jnp.maximum(s_hat - self.s0, 0.0) / (1.0 - self.s0))
        reach = jnp.where(jnp.isfinite(reach), reach, 0.0)        # passing at launch
        rows = [jnp.ravel(reach * self.row_scale[:, None])]
        out = {}
        if self.mirror_target is not None:
            mirror = orbits[0]["mirror"]
            excess = jnp.maximum(mirror - self.mirror_target, 0.0)
            rows.append(self.mirror_weight * excess / np.sqrt(excess.size))
            out["mirror"] = mirror
        if self.fow is not None:
            b00 = booz["bmnc_b"][0, jnp.argmin(jnp.abs(booz["xm_b"]) + jnp.abs(booz["xn_b"]))]
            eps = jnp.sqrt(s_hat) * jnp.sqrt(2.0 * jnp.abs(booz["psi_edge"]) / b00) * b00 / jnp.abs(booz["G_b"][0])
            q = 1.0 / jnp.interp(s_hat, s_b, jnp.abs(booz["iota_b"]))
            delta = 2.0 * jnp.sqrt(s_hat) * self.fow * q / jnp.sqrt(eps)
            lost = jax.nn.sigmoid((s_hat + delta - 1.0) / 0.05)
            lost = jnp.where(jnp.isfinite(s_hat), lost, 0.0)
            rows.append(self.fow_weight * jnp.ravel(lost * self.row_scale[:, None]))
            out["gamma_fow"] = jnp.sum(lost * self.row_scale[:, None] ** 2)
        if self.excursion_weight:
            j_map = jnp.stack([o["j_map"] for o in orbits])        # (pitch, s, alpha)
            finite = jnp.isfinite(j_map)
            j_mean = jnp.nanmean(j_map, axis=-1)
            j_mean = jnp.where(jnp.isfinite(j_mean), j_mean, 0.0)
            slope = jnp.abs(jnp.gradient(j_mean, s_b, axis=-1)) + self.floor * jnp.abs(j_mean)
            delta = jnp.abs(jnp.where(finite, j_map, j_mean[..., None]) - j_mean[..., None])
            delta = delta / jnp.maximum(slope, jnp.finfo(slope.dtype).tiny)[..., None]
            excursion = jnp.maximum(delta - (self.s_limit - s_b)[:, None], 0.0)
            rows.append(self.excursion_weight * jnp.ravel(excursion) / np.sqrt(excursion.size))
        residuals1d = jnp.concatenate(rows)
        return {**out, "residuals1d": residuals1d, "total": jnp.sum(residuals1d ** 2),
                "orbits": orbits, "reach": reach, "gamma_w": jnp.sum(
                    reach * self.row_scale[:, None] ** 2),
                "s_max": jnp.stack([o["s_max"] for o in orbits])}
