"""Differentiable quasi-isodynamic residuals.

The reference QI scripts in ``omnigenity_optimization`` measure whether the
bounce width and field-line profile of magnetic wells are independent of
field-line label.  Their implementation uses branch finding and splines, which
are useful diagnostics but not a smooth objective for JAX autodiff.  This
module keeps the same target and replaces hard branch extraction with smooth
level-set widths:

``width(alpha, B_j) = integral H(B_j - B(alpha, phi)) dphi``

where ``H`` is a logistic approximation to the step function.  A QI surface has
widths that are independent of ``alpha`` for every level ``B_j``.  The residual
can also include branch-width and profile-consistency terms.  The default
weights are calibrated so the smooth metric ranks the seed, the published
``omnigenity_optimization`` QI result, and current vmec_jax candidates in the
same order as the branch-squash/stretch/shuffle diagnostic from the reference
Goodman et al. omnigenity workflow, while remaining differentiable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from ._compat import jax, jnp


def _require_jax():
    if jnp is None:
        raise ImportError("vmec_jax.quasi_isodynamic requires JAX (jax + jaxlib).")


def _as_surface_array(surfaces) -> jnp.ndarray:
    try:
        values = list(surfaces)  # type: ignore[arg-type]
    except Exception:
        values = [surfaces]
    return jnp.asarray(values, dtype=jnp.float64)


def _as_surface_list(surfaces) -> list[float]:
    try:
        values = list(surfaces)  # type: ignore[arg-type]
    except Exception:
        values = [surfaces]
    return [float(value) for value in values]


def _as_weight_array(weights, nsurf: int) -> jnp.ndarray:
    if weights is None:
        return jnp.ones((nsurf,), dtype=jnp.float64)
    return jnp.asarray(list(weights), dtype=jnp.float64)


def _smooth_reduce_max(values, *, axis=None, softness: float = 0.0):
    """Return a hard or smooth maximum with stable normalization."""
    values = jnp.asarray(values)
    if float(softness) <= 0.0:
        return jnp.max(values, axis=axis)
    eps = jnp.asarray(float(softness), dtype=values.dtype)
    vmax = jnp.max(values, axis=axis, keepdims=True)
    shifted = jnp.exp((values - vmax) / eps)
    out = jnp.squeeze(vmax, axis=axis) + eps * jnp.log(jnp.mean(shifted, axis=axis))
    return out


def _smooth_reduce_min(values, *, axis=None, softness: float = 0.0):
    return -_smooth_reduce_max(-jnp.asarray(values), axis=axis, softness=softness)


def _positive_part(value, *, softness: float = 0.0):
    value = jnp.asarray(value)
    if float(softness) <= 0.0:
        return jnp.maximum(value, jnp.asarray(0.0, dtype=value.dtype))
    eps = jnp.asarray(float(softness), dtype=value.dtype)
    return eps * jax.nn.softplus(value / eps)


def _nearest_half_mesh_indices(surfaces: Iterable[float], *, n_half: int) -> np.ndarray:
    if int(n_half) < 1:
        raise ValueError("QI residual requires at least one half-mesh Boozer surface")
    surf = np.asarray(list(surfaces), dtype=float)
    s_half = 0.5 * (np.arange(int(n_half), dtype=float) + np.arange(1, int(n_half) + 1, dtype=float)) / float(n_half)
    return np.asarray([int(np.argmin(np.abs(s_half - value))) for value in surf], dtype=np.int32)


@dataclass(frozen=True)
class _QIBoozerSurfaceGrid:
    """Validated Boozer data and smooth-QI sampling grid."""

    bmnc_b: Any
    xm_b: Any
    xn_b: Any
    iota_b: Any
    nfp: int
    nphi: int
    nalpha: int
    n_bounce: int
    nsurf: int
    weights_arr: Any
    dtype: Any
    phi0: Any
    phi1: Any
    phi: Any
    alpha: Any
    bmag: Any
    bnorm: Any
    levels: Any
    level_count: int
    eps: Any


def _qi_boozer_surface_grid(
    *,
    bmnc_b,
    xm_b,
    xn_b,
    iota_b,
    nfp: int,
    weights,
    nphi: int,
    nalpha: int,
    n_bounce: int,
    include_bounce_endpoints: bool,
    softness: float,
    phimin: float,
) -> _QIBoozerSurfaceGrid:
    """Validate Boozer inputs and evaluate normalized ``|B|`` on field lines."""
    bmnc_b = jnp.asarray(bmnc_b, dtype=jnp.float64)
    xm_b = jnp.asarray(xm_b, dtype=jnp.float64)
    xn_b = jnp.asarray(xn_b, dtype=jnp.float64)
    iota_b = jnp.asarray(iota_b, dtype=jnp.float64)
    nfp = int(nfp)
    nphi = int(nphi)
    nalpha = int(nalpha)
    n_bounce = int(n_bounce)
    if bmnc_b.ndim != 2:
        raise ValueError(f"bmnc_b must have shape (nsurf, nmodes), got {bmnc_b.shape}")
    if int(bmnc_b.shape[1]) != int(xm_b.shape[0]) or int(xm_b.shape[0]) != int(xn_b.shape[0]):
        raise ValueError("Boozer mode arrays must have the same mode dimension as bmnc_b")
    if int(iota_b.shape[0]) != int(bmnc_b.shape[0]):
        raise ValueError("iota_b must have one value per Boozer surface")
    if nphi < 4 or nalpha < 2 or n_bounce < 2:
        raise ValueError("QI residual requires nphi >= 4, nalpha >= 2, and n_bounce >= 2")

    nsurf = int(bmnc_b.shape[0])
    weights_arr = _as_weight_array(weights, nsurf)
    if int(weights_arr.shape[0]) != nsurf:
        raise ValueError("weights must have the same length as the number of Boozer surfaces")

    dtype = bmnc_b.dtype
    phi0 = jnp.asarray(float(phimin), dtype=dtype)
    phi1 = phi0 + jnp.asarray(2.0 * np.pi / nfp, dtype=dtype)
    phi = jnp.linspace(phi0, phi1, nphi, endpoint=True, dtype=dtype)
    alpha = jnp.linspace(0.0, 2.0 * jnp.pi, nalpha, endpoint=False, dtype=dtype)

    theta = alpha[None, None, :] + iota_b[:, None, None] * phi[None, :, None]
    angle = theta[:, :, :, None] * xm_b[None, None, None, :] - phi[None, :, None, None] * xn_b[
        None, None, None, :
    ]
    bmag = jnp.sum(bmnc_b[:, None, None, :] * jnp.cos(angle), axis=-1)

    bmin = jnp.min(bmag, axis=(1, 2), keepdims=True)
    bmax = jnp.max(bmag, axis=(1, 2), keepdims=True)
    denom = jnp.maximum(bmax - bmin, jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype))
    bnorm = (bmag - bmin) / denom

    if bool(include_bounce_endpoints):
        levels = jnp.linspace(0.0, 1.0, n_bounce, endpoint=True, dtype=dtype)
    else:
        levels = jnp.linspace(0.0, 1.0, n_bounce + 2, endpoint=True, dtype=dtype)[1:-1]
    eps = jnp.maximum(jnp.asarray(float(softness), dtype=dtype), jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype))

    return _QIBoozerSurfaceGrid(
        bmnc_b=bmnc_b,
        xm_b=xm_b,
        xn_b=xn_b,
        iota_b=iota_b,
        nfp=nfp,
        nphi=nphi,
        nalpha=nalpha,
        n_bounce=n_bounce,
        nsurf=nsurf,
        weights_arr=weights_arr,
        dtype=dtype,
        phi0=phi0,
        phi1=phi1,
        phi=phi,
        alpha=alpha,
        bmag=bmag,
        bnorm=bnorm,
        levels=levels,
        level_count=int(levels.shape[0]),
        eps=eps,
    )


def _qi_branch_width_residuals(grid: _QIBoozerSurfaceGrid, *, branch_width_weight: float, branch_width_softness: float):
    """Return Goodman-style branch-width residuals on the smooth QI grid."""

    dtype, empty3 = grid.dtype, jnp.zeros((grid.nsurf, grid.nalpha, 0), dtype=grid.dtype)
    if float(branch_width_weight) == 0.0:
        return {"residuals1d": jnp.zeros((0,), dtype=dtype), "residuals3d": empty3,
                "branch_widths": empty3, "branch_widths_mean": jnp.zeros((grid.nsurf, 1, 0), dtype=dtype)}

    bperiodic = jnp.swapaxes(grid.bnorm[:, :-1, :], 1, 2)  # (nsurf, nalpha, nperiodic)
    nperiodic = int(grid.nphi - 1)
    offsets = jnp.arange(max(1, nperiodic // 2) + 1, dtype=jnp.int32)
    min_index = jnp.argmin(bperiodic, axis=-1)
    left_index = jnp.mod(min_index[:, :, None] - offsets[None, None, :], nperiodic)
    right_index = jnp.mod(min_index[:, :, None] + offsets[None, None, :], nperiodic)
    left_branch = jnp.maximum.accumulate(jnp.take_along_axis(bperiodic, left_index, axis=-1), axis=-1)
    right_branch = jnp.maximum.accumulate(jnp.take_along_axis(bperiodic, right_index, axis=-1), axis=-1)
    tiny = jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype)
    left_branch = (left_branch - left_branch[..., :1]) / jnp.maximum(left_branch[..., -1:] - left_branch[..., :1], tiny)
    right_branch = (right_branch - right_branch[..., :1]) / jnp.maximum(right_branch[..., -1:] - right_branch[..., :1], tiny)
    distance = jnp.asarray(offsets, dtype=dtype) / jnp.asarray(nperiodic, dtype=dtype)
    branch_eps = jnp.maximum(jnp.asarray(float(branch_width_softness), dtype=dtype), jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype))

    def _smooth_branch_crossing(branch):
        logits = -((branch[:, :, :, None] - grid.levels[None, None, None, :]) / branch_eps) ** 2
        logits = logits - jnp.max(logits, axis=2, keepdims=True)
        crossing_weights = jnp.exp(logits)
        crossing_weights = crossing_weights / jnp.sum(crossing_weights, axis=2, keepdims=True)
        return jnp.sum(crossing_weights * distance[None, None, :, None], axis=2)

    branch_widths = _smooth_branch_crossing(left_branch) + _smooth_branch_crossing(right_branch)
    branch_widths_mean = jnp.mean(branch_widths, axis=1, keepdims=True)
    residuals3d = ((branch_widths - branch_widths_mean) * jnp.sqrt(grid.weights_arr)[:, None, None]
                   * jnp.asarray(float(branch_width_weight), dtype=dtype))
    residuals1d = jnp.ravel(residuals3d) / jnp.sqrt(jnp.asarray(grid.nalpha * grid.level_count, dtype=dtype))
    return {"residuals1d": residuals1d, "residuals3d": residuals3d, "branch_widths": branch_widths,
            "branch_widths_mean": branch_widths_mean}


def _qi_width_profile_residuals(grid: _QIBoozerSurfaceGrid, *, width_weight: float, profile_weight: float):
    occupancy = jax_sigmoid((grid.levels[None, None, None, :] - grid.bnorm[:, :, :, None]) / grid.eps)
    widths = jnp.mean(occupancy, axis=1)
    widths_mean = jnp.mean(widths, axis=1, keepdims=True)
    width_residuals3d = ((widths - widths_mean) * jnp.sqrt(grid.weights_arr)[:, None, None]
                         * jnp.asarray(float(width_weight), dtype=grid.dtype))
    width_residuals1d = jnp.ravel(width_residuals3d) / jnp.sqrt(jnp.asarray(grid.nalpha * grid.level_count,
                                                                            dtype=grid.dtype))
    profile_mean = jnp.mean(grid.bnorm, axis=2, keepdims=True)
    profile_residuals3d = ((grid.bnorm - profile_mean) * jnp.sqrt(grid.weights_arr)[:, None, None]
                           * jnp.asarray(float(profile_weight), dtype=grid.dtype))
    profile_residuals1d = jnp.ravel(profile_residuals3d) / jnp.sqrt(jnp.asarray(grid.nalpha * grid.nphi,
                                                                                dtype=grid.dtype))
    return (width_residuals1d, width_residuals3d, widths, widths_mean,
            profile_residuals1d, profile_residuals3d, profile_mean)


def _qi_aligned_profile_residuals(grid: _QIBoozerSurfaceGrid, *, aligned_profile_weight: float,
                                  aligned_profile_softness: float, aligned_profile_trap_level: float,
                                  aligned_profile_trap_softness: float):
    """Return smooth minimum-aligned trapped-well profile residuals."""

    dtype, empty = grid.dtype, jnp.zeros((grid.nsurf, 0, grid.nalpha), dtype=grid.dtype)
    aligned_min_phi = jnp.zeros((grid.nsurf, grid.nalpha), dtype=dtype)
    if float(aligned_profile_weight) == 0.0:
        return {"residuals1d": jnp.zeros((0,), dtype=dtype), "residuals3d": empty, "profile": empty,
                "profile_mean": jnp.zeros((grid.nsurf, 0, 1), dtype=dtype), "trap_weight": empty,
                "min_phi": aligned_min_phi}

    # Drop the repeated endpoint before FFT-based periodic shifts.
    bperiodic = grid.bnorm[:, :-1, :]
    nperiodic = int(grid.nphi - 1)
    period = jnp.asarray(2.0 * np.pi / grid.nfp, dtype=dtype)
    min_temperature = jnp.maximum(jnp.asarray(float(aligned_profile_softness), dtype=dtype),
                                  jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype))
    shifted_for_weights = bperiodic - jnp.min(bperiodic, axis=1, keepdims=True)
    argmin_weights = jnp.exp(-shifted_for_weights / min_temperature)
    argmin_weights = argmin_weights / jnp.sum(argmin_weights, axis=1, keepdims=True)
    grid_angle = 2.0 * jnp.pi * jnp.arange(nperiodic, dtype=dtype) / jnp.asarray(nperiodic, dtype=dtype)
    min_angle = jnp.mod(jnp.angle(jnp.sum(argmin_weights * jnp.exp(1j * grid_angle)[None, :, None], axis=1)), 2.0 * jnp.pi)
    aligned_min_phi = min_angle * period / (2.0 * jnp.pi) + grid.phi0

    coeffs = jnp.fft.fft(bperiodic, axis=1)
    mode_numbers = jnp.fft.fftfreq(nperiodic) * jnp.asarray(nperiodic, dtype=dtype)
    phase = jnp.exp(1j * 2.0 * jnp.pi * mode_numbers[None, :, None] * (aligned_min_phi - grid.phi0)[:, None, :] / period)
    aligned_profile = jnp.real(jnp.fft.ifft(coeffs * phase, axis=1))
    aligned_profile_mean = jnp.mean(aligned_profile, axis=2, keepdims=True)
    trap_eps = jnp.maximum(jnp.asarray(float(aligned_profile_trap_softness), dtype=dtype), jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype))
    trap_weight = jax_sigmoid((jnp.asarray(float(aligned_profile_trap_level), dtype=dtype) - aligned_profile) / trap_eps)
    residuals3d = ((aligned_profile - aligned_profile_mean) * trap_weight * jnp.sqrt(grid.weights_arr)[:, None, None]
                   * jnp.asarray(float(aligned_profile_weight), dtype=dtype))
    residuals1d = jnp.ravel(residuals3d) / jnp.sqrt(jnp.asarray(grid.nalpha * nperiodic, dtype=dtype))
    return {"residuals1d": residuals1d, "residuals3d": residuals3d, "profile": aligned_profile,
            "profile_mean": aligned_profile_mean, "trap_weight": trap_weight, "min_phi": aligned_min_phi}


def _qi_shuffle_profile_residuals(grid: _QIBoozerSurfaceGrid, *, n_bounce: int, include_bounce_endpoints: bool,
                                  shuffle_profile_weight: float, shuffle_profile_softness: float,
                                  shuffle_profile_nphi_out: int | None,
                                  weighted_shuffle_profile_weight: float,
                                  weighted_shuffle_profile_softness: float):
    nsurf, nphi, nalpha, dtype = grid.nsurf, grid.nphi, grid.nalpha, grid.dtype
    shuffle_profile_residuals1d = jnp.zeros((0,), dtype=dtype)
    shuffle_profile_residuals3d = jnp.zeros((nsurf, nalpha, 0), dtype=dtype)
    shuffle_profile = jnp.zeros((nsurf, nalpha, 0), dtype=dtype)
    weighted_shuffle_profile_residuals1d = jnp.zeros((0,), dtype=dtype)
    weighted_shuffle_profile_residuals3d = jnp.zeros((nsurf, nalpha, 0), dtype=dtype)
    weighted_shuffle_profile = jnp.zeros((nsurf, nalpha, 0), dtype=dtype)
    weighted_shuffle_alpha_weights = jnp.zeros((nsurf, nalpha), dtype=dtype)
    shuffle_branch_widths = jnp.zeros((nsurf, nalpha, 0), dtype=dtype)
    shuffle_branch_widths_mean = jnp.zeros((nsurf, 1, 0), dtype=dtype)
    weighted_shuffle_branch_widths_mean = jnp.zeros((nsurf, 1, 0), dtype=dtype)
    if float(shuffle_profile_weight) == 0.0 and float(weighted_shuffle_profile_weight) == 0.0:
        return (shuffle_profile_residuals1d, shuffle_profile_residuals3d, shuffle_profile,
                weighted_shuffle_profile_residuals1d, weighted_shuffle_profile_residuals3d,
                weighted_shuffle_profile, weighted_shuffle_alpha_weights, shuffle_branch_widths,
                shuffle_branch_widths_mean, weighted_shuffle_branch_widths_mean)

    b_by_alpha = jnp.swapaxes(grid.bnorm, 1, 2)  # (nsurf, nalpha, nphi)
    offsets = jnp.arange(nphi, dtype=jnp.int32)
    offsets_float = jnp.asarray(offsets, dtype=dtype)
    dphi = (grid.phi1 - grid.phi0) / jnp.asarray(max(nphi - 1, 1), dtype=dtype)
    period = grid.phi1 - grid.phi0
    min_index = jnp.argmin(b_by_alpha, axis=-1)

    left_index_raw = min_index[:, :, None] - offsets[None, None, :]
    right_index_raw = min_index[:, :, None] + offsets[None, None, :]
    left_valid = left_index_raw >= 0
    right_valid = right_index_raw < nphi
    left_index = jnp.clip(left_index_raw, 0, nphi - 1)
    right_index = jnp.clip(right_index_raw, 0, nphi - 1)
    left_raw = jnp.take_along_axis(b_by_alpha, left_index, axis=-1)
    right_raw = jnp.take_along_axis(b_by_alpha, right_index, axis=-1)
    left_raw = jnp.where(left_valid, left_raw, jnp.asarray(1.0, dtype=dtype))
    right_raw = jnp.where(right_valid, right_raw, jnp.asarray(1.0, dtype=dtype))

    left_branch = jnp.maximum.accumulate(left_raw, axis=-1)
    right_branch = jnp.maximum.accumulate(right_raw, axis=-1)

    def _stretch_branches(left, right):
        pmax = jnp.asarray(50.0, dtype=dtype)
        pmin = jnp.asarray(15.0, dtype=dtype)
        denom = jnp.maximum(jnp.asarray(nphi - 1, dtype=dtype), jnp.asarray(1.0, dtype=dtype))
        x = offsets_float / denom
        window = ((jnp.cos(2.0 * jnp.pi * x) + 1.0) / 2.0)
        left_legacy = jnp.flip(left, axis=-1)
        f_left = jnp.where((x < 0.5)[None, None, :],
                           (1.0 - left_legacy[..., :1]) * window[None, None, :] ** pmax,
                           (-left_legacy[..., -1:]) * window[None, None, :] ** pmin)
        f_right = jnp.where((x < 0.5)[None, None, :],
                            (-right[..., :1]) * window[None, None, :] ** pmin,
                            (1.0 - right[..., -1:]) * window[None, None, :] ** pmax)
        return (jnp.maximum.accumulate(jnp.flip(left_legacy + f_left, axis=-1), axis=-1),
                jnp.maximum.accumulate(right + f_right, axis=-1))

    if bool(include_bounce_endpoints):
        shuffle_levels = jnp.linspace(0.0, 1.0, n_bounce, endpoint=True, dtype=dtype)
    else:
        shuffle_levels = jnp.linspace(0.0, 1.0, n_bounce + 2, endpoint=True, dtype=dtype)[1:-1]
    shuffle_eps = jnp.maximum(jnp.asarray(float(shuffle_profile_softness), dtype=dtype),
                              jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype))
    trapz_weights = jnp.ones((nphi,), dtype=dtype).at[0].set(0.5).at[-1].set(0.5)

    def _branch_crossing(branch):
        occupancy_local = jax_sigmoid((shuffle_levels[None, None, None, :] - branch[:, :, :, None]) / shuffle_eps)
        return jnp.sum(occupancy_local * trapz_weights[None, None, :, None], axis=2) * dphi

    def _linear_branch_crossing(branch):
        distance = offsets_float * dphi
        branch_ramp = jnp.arange(nphi, dtype=dtype) * jnp.asarray(1.0e-14, dtype=dtype)

        def _interp_branch_one(branch_1d):
            return jnp.interp(shuffle_levels, branch_1d + branch_ramp, distance)

        return jax.vmap(jax.vmap(_interp_branch_one, in_axes=0, out_axes=0), in_axes=0, out_axes=0)(branch)

    left_crossing = _branch_crossing(left_branch)
    right_crossing = _branch_crossing(right_branch)
    shuffle_branch_widths = left_crossing + right_crossing
    shuffle_branch_widths_mean = jnp.mean(shuffle_branch_widths, axis=1, keepdims=True)
    weighted_left_crossing = left_crossing
    weighted_right_crossing = right_crossing
    weighted_shuffle_branch_widths = shuffle_branch_widths
    if float(weighted_shuffle_profile_weight) != 0.0:
        weighted_left_branch, weighted_right_branch = _stretch_branches(left_branch, right_branch)
        weighted_left_crossing = _linear_branch_crossing(weighted_left_branch)
        weighted_right_crossing = _linear_branch_crossing(weighted_right_branch)
        weighted_shuffle_branch_widths = weighted_left_crossing + weighted_right_crossing
        valid_count = jnp.maximum(jnp.sum(left_valid.astype(dtype) + right_valid.astype(dtype), axis=-1),
                                  jnp.asarray(1.0, dtype=dtype))
        squash_error = jnp.sum(jnp.where(left_valid, (left_raw - weighted_left_branch) ** 2, 0.0)
                               + jnp.where(right_valid, (right_raw - weighted_right_branch) ** 2, 0.0),
                               axis=-1) / valid_count
        weighted_eps = jnp.maximum(jnp.asarray(float(weighted_shuffle_profile_softness), dtype=dtype) ** 2,
                                   jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype))
        inv_error = 1.0 / jnp.maximum(squash_error, weighted_eps)
        weighted_shuffle_alpha_weights = inv_error / jnp.maximum(
            jnp.sum(inv_error, axis=1, keepdims=True), jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype)
        )
        weighted_shuffle_branch_widths_mean = jnp.sum(
            weighted_shuffle_branch_widths * weighted_shuffle_alpha_weights[:, :, None], axis=1, keepdims=True
        )

    min_phi = grid.phi0 + jnp.asarray(min_index, dtype=dtype) * dphi
    left_endpoint = jnp.maximum(min_phi - grid.phi0, jnp.asarray(0.0, dtype=dtype))
    right_endpoint = jnp.maximum(grid.phi1 - min_phi, jnp.asarray(0.0, dtype=dtype))
    signed_phi = (offsets_float[None, None, :] - jnp.asarray(min_index[:, :, None], dtype=dtype)) * dphi
    shuffle_eval_count = nphi if shuffle_profile_nphi_out is None else int(shuffle_profile_nphi_out)
    if shuffle_eval_count == nphi:
        signed_phi_eval = signed_phi
        b_eval = b_by_alpha
    else:
        phi_eval = jnp.linspace(grid.phi0, grid.phi1, shuffle_eval_count, endpoint=True, dtype=dtype)
        signed_phi_eval = phi_eval[None, None, :] - min_phi[:, :, None]
        original_ramp = jnp.arange(nphi, dtype=dtype) * jnp.asarray(1.0e-14, dtype=dtype) * period
        signed_phi_interp = signed_phi + original_ramp[None, None, :]

        def _interp_original_one(xp, fp, x):
            return jnp.interp(x, xp, fp)

        b_eval = jax.vmap(
            jax.vmap(_interp_original_one, in_axes=(0, 0, 0), out_axes=0), in_axes=(0, 0, 0), out_axes=0
        )(signed_phi_interp, b_by_alpha, signed_phi_eval)

    level_full = jnp.concatenate([jnp.zeros((1,), dtype=dtype), shuffle_levels, jnp.ones((1,), dtype=dtype)])
    y_target = jnp.concatenate([jnp.flip(level_full, axis=0), level_full[1:]], axis=0)

    def _interp_one(xp, x):
        return jnp.interp(x, xp, y_target)

    def _profile_from_width_mean(left_cross, right_cross, branch_widths, width_mean, x_eval):
        delta_width = 0.5 * (branch_widths - width_mean)
        left_target = jnp.clip(left_cross - delta_width, 0.0, left_endpoint[:, :, None])
        right_target = jnp.clip(right_cross - delta_width, 0.0, right_endpoint[:, :, None])
        left_full = jnp.concatenate([jnp.zeros((nsurf, nalpha, 1), dtype=dtype), left_target,
                                     left_endpoint[:, :, None]], axis=-1)
        right_full = jnp.concatenate([jnp.zeros((nsurf, nalpha, 1), dtype=dtype), right_target,
                                      right_endpoint[:, :, None]], axis=-1)
        left_full = jnp.maximum.accumulate(left_full, axis=-1)
        right_full = jnp.maximum.accumulate(right_full, axis=-1)
        x_target = jnp.concatenate([-jnp.flip(left_full, axis=-1), right_full[:, :, 1:]], axis=-1)
        ramp = jnp.arange(x_target.shape[-1], dtype=dtype) * jnp.asarray(1.0e-14, dtype=dtype) * period
        x_target = x_target + ramp[None, None, :]
        return jax.vmap(jax.vmap(_interp_one, in_axes=(0, 0), out_axes=0), in_axes=(0, 0), out_axes=0)(
            x_target, x_eval
        )

    if float(shuffle_profile_weight) != 0.0:
        shuffle_profile = _profile_from_width_mean(
            left_crossing, right_crossing, shuffle_branch_widths, shuffle_branch_widths_mean, signed_phi_eval
        )
        shuffle_profile_residuals3d = ((shuffle_profile - b_eval) * jnp.sqrt(grid.weights_arr)[:, None, None]
                                       * jnp.asarray(float(shuffle_profile_weight), dtype=dtype))
        shuffle_profile_residuals1d = jnp.ravel(shuffle_profile_residuals3d) / jnp.sqrt(
            jnp.asarray(nalpha * shuffle_eval_count, dtype=dtype)
        )
    if float(weighted_shuffle_profile_weight) != 0.0:
        weighted_shuffle_profile = _profile_from_width_mean(
            weighted_left_crossing, weighted_right_crossing, weighted_shuffle_branch_widths,
            weighted_shuffle_branch_widths_mean, signed_phi_eval
        )
        weighted_shuffle_profile_residuals3d = (
            (weighted_shuffle_profile - b_eval) * jnp.sqrt(grid.weights_arr)[:, None, None]
            * jnp.asarray(float(weighted_shuffle_profile_weight), dtype=dtype)
        )
        weighted_shuffle_profile_residuals1d = jnp.ravel(weighted_shuffle_profile_residuals3d) / jnp.sqrt(
            jnp.asarray(nalpha * shuffle_eval_count, dtype=dtype)
        )
    return (shuffle_profile_residuals1d, shuffle_profile_residuals3d, shuffle_profile,
            weighted_shuffle_profile_residuals1d, weighted_shuffle_profile_residuals3d,
            weighted_shuffle_profile, weighted_shuffle_alpha_weights, shuffle_branch_widths,
            shuffle_branch_widths_mean, weighted_shuffle_branch_widths_mean)


def mirror_ratio_penalty_from_boozer_modes(
    *,
    bmnc_b,
    xm_b,
    xn_b,
    nfp: int,
    bmns_b=None,
    threshold: float = 0.21,
    weights: Iterable[float] | None = None,
    ntheta: int = 128,
    nphi: int = 128,
    phimin: float = 0.0,
    smooth_extrema: float = 0.0,
    smooth_penalty: float = 0.0,
):
    """Penalize the maximum mirror ratio from Boozer ``|B|`` modes.

    This is the JAX-native analogue of the ``MirrorRatioPen`` diagnostic used
    in the reference ``omnigenity_optimization`` QI script.  For every supplied
    Boozer surface it evaluates ``|B|(theta_B, phi_B)`` on a uniform grid and
    computes

    ``M = (Bmax - Bmin) / (Bmax + Bmin)``.

    The least-squares residual is ``max(0, M - threshold)`` per surface.  Set
    ``smooth_extrema`` and/or ``smooth_penalty`` to positive values when a fully
    smooth softmax/softplus surrogate is preferred.
    """
    _require_jax()

    bmnc_b = jnp.asarray(bmnc_b, dtype=jnp.float64)
    if bmnc_b.ndim == 1:
        bmnc_b = bmnc_b[None, :]
    bmns_b = jnp.zeros_like(bmnc_b) if bmns_b is None else jnp.asarray(bmns_b, dtype=jnp.float64)
    if bmns_b.ndim == 1:
        bmns_b = bmns_b[None, :]
    xm_b = jnp.asarray(xm_b, dtype=jnp.float64)
    xn_b = jnp.asarray(xn_b, dtype=jnp.float64)
    if int(bmnc_b.shape[1]) != int(xm_b.shape[0]) or int(xm_b.shape[0]) != int(xn_b.shape[0]):
        raise ValueError("Boozer mode arrays must have the same mode dimension as bmnc_b")
    if bmns_b.shape != bmnc_b.shape:
        raise ValueError("bmns_b must have the same shape as bmnc_b")
    if int(ntheta) < 4 or int(nphi) < 4:
        raise ValueError("mirror-ratio penalty requires ntheta >= 4 and nphi >= 4")

    nsurf = int(bmnc_b.shape[0])
    weights_arr = _as_weight_array(weights, nsurf)
    if int(weights_arr.shape[0]) != nsurf:
        raise ValueError("weights must have the same length as the number of Boozer surfaces")

    dtype = bmnc_b.dtype
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, int(ntheta), endpoint=False, dtype=dtype)
    phi = jnp.linspace(
        float(phimin),
        float(phimin) + 2.0 * jnp.pi / float(nfp),
        int(nphi),
        endpoint=False,
        dtype=dtype,
    )
    angle = theta[None, :, None, None] * xm_b[None, None, None, :] - phi[None, None, :, None] * xn_b[
        None, None, None, :
    ]
    bmag = jnp.sum(
        bmnc_b[:, None, None, :] * jnp.cos(angle) + bmns_b[:, None, None, :] * jnp.sin(angle),
        axis=-1,
    )
    bmax = _smooth_reduce_max(bmag, axis=(1, 2), softness=float(smooth_extrema))
    bmin = _smooth_reduce_min(bmag, axis=(1, 2), softness=float(smooth_extrema))
    tiny = jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype)
    # Truncated Boozer |B| spectra can undershoot and produce nonpositive
    # pointwise values during intermediate optimizer steps.  The physical
    # mirror-ratio target should then become a large finite penalty, not an
    # Inf/NaN that breaks the trust-region linear algebra.
    bmin_positive = jnp.maximum(bmin, tiny)
    denom = jnp.maximum(bmax + bmin_positive, tiny)
    mirror_ratio = (bmax - bmin_positive) / denom
    penalty = _positive_part(mirror_ratio - float(threshold), softness=float(smooth_penalty))
    residuals1d = penalty * jnp.sqrt(weights_arr)
    total = jnp.sum(residuals1d * residuals1d)
    return {
        "residuals1d": residuals1d,
        "total": total,
        "penalty": penalty,
        "mirror_ratio": mirror_ratio,
        "bmax": bmax,
        "bmin": bmin,
        "bmag": bmag,
        "theta": theta,
        "phi": phi,
        "threshold": jnp.asarray(float(threshold), dtype=dtype),
    }


def mirror_ratio_penalty_from_boozer_output(
    booz,
    *,
    nfp: int | None = None,
    threshold: float = 0.21,
    weights: Iterable[float] | None = None,
    ntheta: int = 128,
    nphi: int = 128,
    phimin: float = 0.0,
    smooth_extrema: float = 0.0,
    smooth_penalty: float = 0.0,
):
    """Evaluate :func:`mirror_ratio_penalty_from_boozer_modes` from Boozer output."""
    nfp_local = int(nfp) if nfp is not None else int(np.asarray(booz["nfp_b"]))
    bmns_b = booz.get("bmns_b")
    return mirror_ratio_penalty_from_boozer_modes(
        bmnc_b=booz["bmnc_b"],
        bmns_b=bmns_b,
        xm_b=booz["ixm_b"],
        xn_b=booz["ixn_b"],
        nfp=nfp_local,
        threshold=threshold,
        weights=weights,
        ntheta=ntheta,
        nphi=nphi,
        phimin=phimin,
        smooth_extrema=smooth_extrema,
        smooth_penalty=smooth_penalty,
    )


def boundary_max_elongation_from_rz(
    R,
    Z,
    *,
    phi=None,
    smooth_extrema: float = 0.0,
):
    """Estimate maximum LCFS cross-section elongation from ``R(theta, phi), Z``.

    The reference SIMSOPT QI script computes an effective elongation by solving
    for normal-plane cross-sections and fitting an ellipse from perimeter and
    area.  That path uses root finding and is not a good JAX objective.  This
    differentiable proxy computes, for each fixed toroidal angle, the covariance
    of the 3-D boundary curve and returns ``sqrt(lambda_major/lambda_minor)``
    using the two largest covariance eigenvalues.  It tracks the same geometric
    failure mode (very stretched boundary cross-sections) while remaining cheap
    and compatible with autodiff.
    """
    _require_jax()

    R = jnp.asarray(R, dtype=jnp.float64)
    Z = jnp.asarray(Z, dtype=jnp.float64)
    if R.shape != Z.shape or R.ndim != 2:
        raise ValueError("R and Z must both have shape (ntheta, nphi)")
    nphi = int(R.shape[1])
    dtype = R.dtype
    if phi is None:
        phi_arr = jnp.linspace(0.0, 2.0 * jnp.pi, nphi, endpoint=False, dtype=dtype)
    else:
        phi_arr = jnp.asarray(phi, dtype=dtype)
        if int(phi_arr.shape[0]) != nphi:
            raise ValueError("phi must have one value per toroidal grid point")

    X = R * jnp.cos(phi_arr)[None, :]
    Y = R * jnp.sin(phi_arr)[None, :]
    points = jnp.stack([X, Y, Z], axis=-1)  # (ntheta, nphi, 3)
    points = jnp.swapaxes(points, 0, 1)  # (nphi, ntheta, 3)
    centered = points - jnp.mean(points, axis=1, keepdims=True)
    cov = jnp.einsum("pti,ptj->pij", centered, centered) / jnp.asarray(R.shape[0], dtype=dtype)
    eigvals = jnp.linalg.eigvalsh(cov)
    tiny = jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype)
    elongation = jnp.sqrt(jnp.maximum(eigvals[:, -1], tiny) / jnp.maximum(eigvals[:, -2], tiny))
    max_elongation = _smooth_reduce_max(elongation, axis=0, softness=float(smooth_extrema))
    return {
        "max_elongation": max_elongation,
        "elongation": elongation,
        "phi": phi_arr,
    }


def max_elongation_penalty_from_state(
    *,
    state,
    static,
    threshold: float = 8.0,
    ntheta: int = 64,
    nphi: int = 24,
    s_index: int = -1,
    smooth_extrema: float = 0.0,
    smooth_penalty: float = 0.0,
):
    """Penalize excessive LCFS elongation from a solved VMEC state."""
    _require_jax()
    if int(ntheta) < 4 or int(nphi) < 3:
        raise ValueError("elongation penalty requires ntheta >= 4 and nphi >= 3")

    from .fourier import build_helical_basis, eval_fourier
    from .grids import AngleGrid
    from .state import VMECState
    from .vmec_parity import vmec_m1_internal_to_physical_signed

    nfp = int(static.cfg.nfp)
    # Keep the angular grid host-static. ``build_helical_basis`` intentionally
    # uses NumPy to cache static Fourier tables; feeding it traced JAX arrays
    # would fail when this objective is evaluated inside the optimizer JIT.
    theta = np.linspace(0.0, 2.0 * np.pi, int(ntheta), endpoint=False, dtype=float)
    phi = np.linspace(0.0, 2.0 * np.pi / float(nfp), int(nphi), endpoint=False, dtype=float)
    zeta = phi * float(nfp)
    grid = AngleGrid(theta=theta, zeta=zeta, nfp=nfp)
    basis = build_helical_basis(static.modes, grid, cache=False)

    cfg = static.cfg
    lconm1 = bool(getattr(cfg, "lconm1", True))
    lthreed = bool(getattr(cfg, "lthreed", int(getattr(cfg, "ntor", 0)) > 0))
    lasym = bool(getattr(cfg, "lasym", False))
    state_use = state
    if lconm1 and (lthreed or lasym) and int(getattr(cfg, "mpol", 0)) > 1:
        Rcos, Zsin, Rsin, Zcos = vmec_m1_internal_to_physical_signed(
            Rcos=state.Rcos,
            Zsin=state.Zsin,
            Rsin=state.Rsin,
            Zcos=state.Zcos,
            modes=static.modes,
            lthreed=lthreed,
            lasym=lasym,
            lconm1=lconm1,
        )
        state_use = VMECState(
            layout=state.layout,
            Rcos=Rcos,
            Rsin=Rsin,
            Zcos=Zcos,
            Zsin=Zsin,
            Lcos=state.Lcos,
            Lsin=state.Lsin,
        )

    s_idx = int(s_index)
    R = eval_fourier(state_use.Rcos[s_idx], state_use.Rsin[s_idx], basis, coeffs_internal=True)
    Z = eval_fourier(state_use.Zcos[s_idx], state_use.Zsin[s_idx], basis, coeffs_internal=True)
    out = boundary_max_elongation_from_rz(R, Z, phi=phi, smooth_extrema=smooth_extrema)
    penalty = _positive_part(out["max_elongation"] - float(threshold), softness=float(smooth_penalty))
    residuals1d = jnp.asarray([penalty], dtype=jnp.float64)
    return {
        **out,
        "residuals1d": residuals1d,
        "total": jnp.sum(residuals1d * residuals1d),
        "penalty": penalty,
        "threshold": jnp.asarray(float(threshold), dtype=jnp.float64),
    }


def _state_with_physical_m1_modes(state, static):
    """Return a state whose m=1 modes use the public VMEC convention."""
    from .state import VMECState
    from .vmec_parity import vmec_m1_internal_to_physical_signed

    cfg = static.cfg
    lconm1 = bool(getattr(cfg, "lconm1", True))
    lthreed = bool(getattr(cfg, "lthreed", int(getattr(cfg, "ntor", 0)) > 0))
    lasym = bool(getattr(cfg, "lasym", False))
    if not (lconm1 and (lthreed or lasym) and int(getattr(cfg, "mpol", 0)) > 1):
        return state
    Rcos, Zsin, Rsin, Zcos = vmec_m1_internal_to_physical_signed(
        Rcos=state.Rcos,
        Zsin=state.Zsin,
        Rsin=state.Rsin,
        Zcos=state.Zcos,
        modes=static.modes,
        lthreed=lthreed,
        lasym=lasym,
        lconm1=lconm1,
    )
    return VMECState(
        layout=state.layout,
        Rcos=Rcos,
        Rsin=Rsin,
        Zcos=Zcos,
        Zsin=Zsin,
        Lcos=state.Lcos,
        Lsin=state.Lsin,
    )


def _periodic_central_diff(values, *, spacing: float, axis: int):
    spacing_arr = jnp.asarray(float(spacing), dtype=jnp.asarray(values).dtype)
    return (jnp.roll(values, -1, axis=axis) - jnp.roll(values, 1, axis=axis)) / (2.0 * spacing_arr)


def _metric_inverse_at_surface(geom, *, s_index: int):
    g = jnp.stack(
        [
            jnp.stack([geom.g_ss[s_index], geom.g_st[s_index], geom.g_sp[s_index]], axis=-1),
            jnp.stack([geom.g_st[s_index], geom.g_tt[s_index], geom.g_tp[s_index]], axis=-1),
            jnp.stack([geom.g_sp[s_index], geom.g_tp[s_index], geom.g_pp[s_index]], axis=-1),
        ],
        axis=-2,
    )
    return jnp.linalg.inv(g)


def lgradb_from_state(
    *,
    state,
    static,
    indata,
    signgs: int,
    s_index: int = -1,
    ntheta: int = 9,
    nphi: int = 7,
    flux_local=None,
):
    """Evaluate the magnetic-gradient scale length on a VMEC surface.

    This is the JAX-native analogue of the ``L_grad_B`` diagnostic used in the
    reference SIMSOPT/``omnigenity_optimization`` examples.  It computes

    ``L_grad_B = |B| sqrt(2 / (nabla B : nabla B))``

    from the Cartesian magnetic field vector on a small VMEC grid.  Angular
    derivatives use periodic centered differences and the radial derivative
    uses the same differentiable finite-difference operator as the geometry
    kernel.  The returned arrays are differentiable with respect to the VMEC
    state and boundary parameters.
    """
    _require_jax()
    if int(ntheta) < 4 or int(nphi) < 4:
        raise ValueError("LgradB requires ntheta >= 4 and nphi >= 4")

    from .energy import flux_profiles_from_indata
    from .field import b_cartesian_from_bsup, bsup_from_geom
    from .fourier import build_helical_basis
    from .geom import _eval_geom_jit
    from .grids import AngleGrid
    from .radial import d_ds_coeffs

    nfp = int(static.cfg.nfp)
    theta = np.linspace(0.0, 2.0 * np.pi, int(ntheta), endpoint=False, dtype=float)
    phi = np.linspace(0.0, 2.0 * np.pi / float(nfp), int(nphi), endpoint=False, dtype=float)
    zeta = phi * float(nfp)
    grid = AngleGrid(theta=theta, zeta=zeta, nfp=nfp)
    basis = build_helical_basis(static.modes, grid, cache=False)
    s_grid = jnp.asarray(static.s, dtype=jnp.float64)

    state_use = _state_with_physical_m1_modes(state, static)
    geom = _eval_geom_jit(state_use, basis, s_grid, jnp.asarray(zeta, dtype=jnp.float64))
    flux = flux_profiles_from_indata(indata, s_grid, signgs=int(signgs)) if flux_local is None else flux_local
    bsupu, bsupv = bsup_from_geom(
        geom,
        phipf=flux.phipf,
        chipf=flux.chipf,
        nfp=nfp,
        signgs=int(signgs),
        lamscale=flux.lamscale,
        flux_is_internal=True,
    )
    bcart = b_cartesian_from_bsup(geom, bsupu, bsupv, zeta=jnp.asarray(zeta, dtype=jnp.float64), nfp=nfp)

    ns = int(bcart.shape[0])
    s_idx = int(s_index)
    if s_idx < 0:
        s_idx += ns
    if s_idx < 0 or s_idx >= ns:
        raise ValueError(f"s_index {s_index} is outside the radial grid with ns={ns}")

    db_ds = d_ds_coeffs(bcart, s_grid)[s_idx]
    db_dtheta = _periodic_central_diff(bcart, spacing=2.0 * np.pi / int(ntheta), axis=1)[s_idx]
    db_dphi = _periodic_central_diff(bcart, spacing=2.0 * np.pi / float(nfp) / int(nphi), axis=2)[s_idx]
    db_dcoords = jnp.stack([db_ds, db_dtheta, db_dphi], axis=-2)

    ginv = _metric_inverse_at_surface(geom, s_index=s_idx)
    grad_b_cart_sq = jnp.einsum("...ic,...ij,...jc->...c", db_dcoords, ginv, db_dcoords)
    grad_b_double_dot_grad_b = jnp.sum(grad_b_cart_sq, axis=-1)
    tiny = jnp.asarray(jnp.finfo(bcart.dtype).tiny, dtype=bcart.dtype)
    bmag = jnp.sqrt(jnp.maximum(jnp.sum(bcart[s_idx] * bcart[s_idx], axis=-1), tiny))
    lgradb = bmag * jnp.sqrt(2.0 / jnp.maximum(grad_b_double_dot_grad_b, tiny))
    return {
        "L_grad_B": lgradb,
        "grad_B_double_dot_grad_B": grad_b_double_dot_grad_b,
        "B_cartesian": bcart[s_idx],
        "Bmag": bmag,
        "theta": jnp.asarray(theta, dtype=jnp.float64),
        "phi": jnp.asarray(phi, dtype=jnp.float64),
        "s_index": jnp.asarray(s_idx, dtype=jnp.int32),
    }


def lgradb_penalty_from_state(
    *,
    state,
    static,
    indata,
    signgs: int,
    threshold: float = 0.30,
    s_index: int = -1,
    ntheta: int = 9,
    nphi: int = 7,
    smooth_penalty: float = 0.0,
    flux_local=None,
):
    """Penalize short magnetic-gradient scale length on a VMEC surface.

    The residual follows the reference omnigenity scripts:

    ``max(1/L_grad_B - 1/threshold, 0) / sqrt(ntheta*nphi)``.

    Use this as an independent least-squares block, e.g. with residual weight
    ``sqrt(0.01)`` to match the QI reference script.
    """
    out = lgradb_from_state(
        state=state,
        static=static,
        indata=indata,
        signgs=signgs,
        s_index=s_index,
        ntheta=ntheta,
        nphi=nphi,
        flux_local=flux_local,
    )
    lgradb = jnp.asarray(out["L_grad_B"], dtype=jnp.float64)
    tiny = jnp.asarray(jnp.finfo(lgradb.dtype).tiny, dtype=lgradb.dtype)
    excess = 1.0 / jnp.maximum(lgradb, tiny) - 1.0 / jnp.asarray(float(threshold), dtype=lgradb.dtype)
    penalty = _positive_part(excess, softness=float(smooth_penalty))
    residuals1d = jnp.ravel(penalty) / jnp.sqrt(jnp.asarray(int(ntheta) * int(nphi), dtype=lgradb.dtype))
    return {
        **out,
        "residuals1d": residuals1d,
        "total": jnp.sum(residuals1d * residuals1d),
        "penalty": penalty,
        "threshold": jnp.asarray(float(threshold), dtype=lgradb.dtype),
        "excess": excess,
        "min_L_grad_B": jnp.min(lgradb),
    }


def quasi_isodynamic_residual_from_boozer_modes(
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
    shuffle_profile_nphi_out: int | None = None,
    weighted_shuffle_profile_weight: float = 0.0,
    weighted_shuffle_profile_softness: float = 2.0e-2,
    aligned_profile_weight: float = 0.0,
    aligned_profile_softness: float = 2.0e-2,
    aligned_profile_trap_level: float = 0.65,
    aligned_profile_trap_softness: float = 5.0e-2,
    phimin: float = 0.0,
):
    """Evaluate a smooth QI residual from Boozer ``|B|`` Fourier modes.

    Parameters
    ----------
    bmnc_b:
        Cosine Boozer ``|B|`` coefficients with shape ``(nsurf, nmodes)``.
    xm_b, xn_b:
        Boozer mode numbers. ``xn_b`` should use the physical toroidal mode
        convention, matching ``booz_xform_jax`` and BOOZ_XFORM.
    iota_b:
        Rotational transform on the same Boozer surfaces.
    nfp:
        Number of field periods.
    weights:
        Optional surface weights.  They are applied as square roots to the
        least-squares residual vector.
    nphi, nalpha, n_bounce:
        Sampling resolution along toroidal angle, field-line label, and
        normalized bounce level.
    include_bounce_endpoints:
        If true, include normalized bounce levels 0 and 1 in the smooth
        level-set residuals, matching the legacy Goodman-style branch-shuffle
        diagnostic.  The default keeps the historical smooth objective behavior
        and samples only interior bounce levels.
    softness:
        Logistic smoothing width in normalized ``|B|`` units. Smaller values
        approach hard branch widths but increase stiffness.
    width_weight:
        Relative weight for the smooth level-set occupancy width residual.
    branch_width_weight:
        Relative weight for a branch-based trapped-well width residual.  This
        follows the reference omnigenity objective more closely than the
        occupancy width: each field line is split at its well minimum, each
        side is made monotone with a cumulative maximum, and level crossings
        are computed with a smooth inverse.
    branch_width_softness:
        Normalized ``|B|`` smoothing width for branch level crossings.
    profile_weight:
        Small relative weight for field-line profile consistency.  Width-only
        and branch-width-only surrogates can rank some QH-like candidates too
        favorably; keeping a small profile term restores the legacy
        branch-shuffle ranking without making this term dominate the objective.
    shuffle_profile_weight:
        Relative weight for a differentiable branch-shuffle profile residual.
        This term follows the reference Goodman/``omnigenity_optimization``
        diagnostic more directly than the occupancy-width residual: it builds
        left/right trapped-branch crossings, shifts them so every field-line
        label has the mean bounce width, and compares that shuffled well to the
        original profile.
    shuffle_profile_softness:
        Logistic smoothing width used to estimate branch crossing locations for
        ``shuffle_profile_weight``.
    shuffle_profile_nphi_out:
        Optional dense output grid for the branch-shuffle profile residual.  If
        set, the shuffled and original wells are compared on this many toroidal
        samples, matching the legacy ``arr_out=True`` Goodman objective more
        closely than the default base ``nphi`` grid.
    weighted_shuffle_profile_weight:
        Relative weight for a branch-shuffle profile residual whose mean bounce
        widths are weighted by a differentiable proxy for the legacy
        squash/stretch quality weights.  This is useful when the unweighted
        smooth residual ranks high-mirror false positives ahead of the
        Goodman-style branch diagnostic.
    weighted_shuffle_profile_softness:
        Logistic smoothing width used by ``weighted_shuffle_profile_weight``.
    aligned_profile_weight:
        Relative weight for a differentiable trapped-well profile residual.
        Each field-line profile is circularly shifted by its smooth minimum
        before comparing against the mean over field-line label.  This is a
        smooth surrogate for the branch/shuffle profile comparison in the
        reference QI scripts.
    aligned_profile_softness:
        Temperature used for the smooth circular argmin that locates each well
        minimum in normalized ``|B|`` units.
    aligned_profile_trap_level, aligned_profile_trap_softness:
        Logistic trapped-region window applied to the aligned profiles.  Values
        below ``aligned_profile_trap_level`` receive the most weight.
    phimin:
        Start of the toroidal interval. The interval length is one field period.

    Returns
    -------
    dict
        ``residuals1d`` is suitable for least-squares optimization. ``total``
        is its squared norm.
    """
    _require_jax()

    if int(nphi) < 4 or int(nalpha) < 2 or int(n_bounce) < 2:
        raise ValueError("QI residual requires nphi >= 4, nalpha >= 2, and n_bounce >= 2")
    if shuffle_profile_nphi_out is not None and int(shuffle_profile_nphi_out) < 4:
        raise ValueError("shuffle_profile_nphi_out must be >= 4 when supplied")
    qi_grid = _qi_boozer_surface_grid(
        bmnc_b=bmnc_b,
        xm_b=xm_b,
        xn_b=xn_b,
        iota_b=iota_b,
        nfp=int(nfp),
        weights=weights,
        nphi=int(nphi),
        nalpha=int(nalpha),
        n_bounce=int(n_bounce),
        include_bounce_endpoints=bool(include_bounce_endpoints),
        softness=float(softness),
        phimin=float(phimin),
    )
    iota_b = qi_grid.iota_b
    phi = qi_grid.phi
    alpha = qi_grid.alpha
    bmag = qi_grid.bmag
    bnorm = qi_grid.bnorm
    levels = qi_grid.levels
    width_residuals1d, width_residuals3d, widths, widths_mean, profile_residuals1d, profile_residuals3d, profile_mean = (
        _qi_width_profile_residuals(qi_grid, width_weight=width_weight, profile_weight=profile_weight)
    )

    branch_width = _qi_branch_width_residuals(
        qi_grid,
        branch_width_weight=branch_width_weight,
        branch_width_softness=branch_width_softness,
    )
    branch_width_residuals1d, branch_width_residuals3d = branch_width["residuals1d"], branch_width["residuals3d"]
    branch_widths, branch_widths_mean = branch_width["branch_widths"], branch_width["branch_widths_mean"]

    aligned = _qi_aligned_profile_residuals(
        qi_grid,
        aligned_profile_weight=aligned_profile_weight,
        aligned_profile_softness=aligned_profile_softness,
        aligned_profile_trap_level=aligned_profile_trap_level,
        aligned_profile_trap_softness=aligned_profile_trap_softness,
    )
    aligned_profile_residuals1d, aligned_profile_residuals3d = aligned["residuals1d"], aligned["residuals3d"]
    aligned_profile, aligned_profile_mean = aligned["profile"], aligned["profile_mean"]
    aligned_profile_trap_weight, aligned_min_phi = aligned["trap_weight"], aligned["min_phi"]

    (shuffle_profile_residuals1d, shuffle_profile_residuals3d, shuffle_profile,
     weighted_shuffle_profile_residuals1d, weighted_shuffle_profile_residuals3d,
     weighted_shuffle_profile, weighted_shuffle_alpha_weights, shuffle_branch_widths,
     shuffle_branch_widths_mean, weighted_shuffle_branch_widths_mean) = _qi_shuffle_profile_residuals(
        qi_grid,
        n_bounce=int(n_bounce),
        include_bounce_endpoints=bool(include_bounce_endpoints),
        shuffle_profile_weight=shuffle_profile_weight,
        shuffle_profile_softness=shuffle_profile_softness,
        shuffle_profile_nphi_out=shuffle_profile_nphi_out,
        weighted_shuffle_profile_weight=weighted_shuffle_profile_weight,
        weighted_shuffle_profile_softness=weighted_shuffle_profile_softness,
    )
    residuals1d = jnp.concatenate(
        [
            width_residuals1d,
            branch_width_residuals1d,
            profile_residuals1d,
            aligned_profile_residuals1d,
            shuffle_profile_residuals1d,
            weighted_shuffle_profile_residuals1d,
        ]
    )
    total = jnp.sum(residuals1d * residuals1d)
    return {
        "residuals1d": residuals1d,
        "residuals3d": width_residuals3d,
        "width_residuals1d": width_residuals1d,
        "width_residuals3d": width_residuals3d,
        "branch_width_residuals1d": branch_width_residuals1d,
        "branch_width_residuals3d": branch_width_residuals3d,
        "profile_residuals1d": profile_residuals1d,
        "profile_residuals3d": profile_residuals3d,
        "aligned_profile_residuals1d": aligned_profile_residuals1d,
        "aligned_profile_residuals3d": aligned_profile_residuals3d,
        "shuffle_profile_residuals1d": shuffle_profile_residuals1d,
        "shuffle_profile_residuals3d": shuffle_profile_residuals3d,
        "weighted_shuffle_profile_residuals1d": weighted_shuffle_profile_residuals1d,
        "weighted_shuffle_profile_residuals3d": weighted_shuffle_profile_residuals3d,
        "total": total,
        "widths": widths,
        "widths_mean": widths_mean,
        "branch_widths": branch_widths,
        "branch_widths_mean": branch_widths_mean,
        "profile_mean": profile_mean,
        "aligned_profile": aligned_profile,
        "aligned_profile_mean": aligned_profile_mean,
        "aligned_profile_trap_weight": aligned_profile_trap_weight,
        "aligned_min_phi": aligned_min_phi,
        "shuffle_profile": shuffle_profile,
        "weighted_shuffle_profile": weighted_shuffle_profile,
        "weighted_shuffle_alpha_weights": weighted_shuffle_alpha_weights,
        "shuffle_branch_widths": shuffle_branch_widths,
        "shuffle_branch_widths_mean": shuffle_branch_widths_mean,
        "weighted_shuffle_branch_widths_mean": weighted_shuffle_branch_widths_mean,
        "shuffle_profile_nphi_out": shuffle_profile_nphi_out,
        "bmag": bmag,
        "bnorm": bnorm,
        "levels": levels,
        "include_bounce_endpoints": bool(include_bounce_endpoints),
        "phi": phi,
        "alpha": alpha,
        "iota": iota_b,
    }


def jax_sigmoid(x):
    """Stable logistic used by QI objectives and their adjoints."""
    return jax.nn.sigmoid(x)


def quasi_isodynamic_residual_from_boozer_output(
    booz,
    *,
    nfp: int | None = None,
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
    shuffle_profile_nphi_out: int | None = None,
    weighted_shuffle_profile_weight: float = 0.0,
    weighted_shuffle_profile_softness: float = 2.0e-2,
    aligned_profile_weight: float = 0.0,
    aligned_profile_softness: float = 2.0e-2,
    aligned_profile_trap_level: float = 0.65,
    aligned_profile_trap_softness: float = 5.0e-2,
    phimin: float = 0.0,
):
    """Evaluate the smooth QI residual from a ``booz_xform_jax`` output dict."""
    nfp_local = int(nfp) if nfp is not None else int(np.asarray(booz["nfp_b"]))
    return quasi_isodynamic_residual_from_boozer_modes(
        bmnc_b=booz["bmnc_b"],
        xm_b=booz["ixm_b"],
        xn_b=booz["ixn_b"],
        iota_b=booz["iota_b"],
        nfp=nfp_local,
        weights=weights,
        nphi=nphi,
        nalpha=nalpha,
        n_bounce=n_bounce,
        include_bounce_endpoints=include_bounce_endpoints,
        softness=softness,
        width_weight=width_weight,
        branch_width_weight=branch_width_weight,
        branch_width_softness=branch_width_softness,
        profile_weight=profile_weight,
        shuffle_profile_weight=shuffle_profile_weight,
        shuffle_profile_softness=shuffle_profile_softness,
        shuffle_profile_nphi_out=shuffle_profile_nphi_out,
        weighted_shuffle_profile_weight=weighted_shuffle_profile_weight,
        weighted_shuffle_profile_softness=weighted_shuffle_profile_softness,
        aligned_profile_weight=aligned_profile_weight,
        aligned_profile_softness=aligned_profile_softness,
        aligned_profile_trap_level=aligned_profile_trap_level,
        aligned_profile_trap_softness=aligned_profile_trap_softness,
        phimin=phimin,
    )


def quasi_isodynamic_residual_from_state(
    *,
    state,
    static,
    indata,
    signgs: int,
    surfaces,
    weights: Iterable[float] | None = None,
    mboz: int = 12,
    nboz: int = 12,
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
    shuffle_profile_nphi_out: int | None = None,
    weighted_shuffle_profile_weight: float = 0.0,
    weighted_shuffle_profile_softness: float = 2.0e-2,
    aligned_profile_weight: float = 0.0,
    aligned_profile_softness: float = 2.0e-2,
    aligned_profile_trap_level: float = 0.65,
    aligned_profile_trap_softness: float = 5.0e-2,
    phimin: float = 0.0,
    flux_local=None,
    prof_local=None,
    pressure_local=None,
    jit_booz: bool = False,
    booz_constants=None,
    booz_grids=None,
    surface_indices=None,
):
    """Evaluate a differentiable QI residual directly from a solved VMEC state.

    This uses ``vmec_jax.booz_xform_inputs_from_state`` followed by the
    functional ``booz_xform_jax`` API.  ``booz_xform_jax`` is an optional
    runtime dependency; install it to use this state-level objective.
    """
    field = boozer_output_from_state(
        state=state,
        static=static,
        indata=indata,
        signgs=signgs,
        surfaces=surfaces,
        mboz=mboz,
        nboz=nboz,
        flux_local=flux_local,
        prof_local=prof_local,
        pressure_local=pressure_local,
        jit_booz=jit_booz,
        booz_constants=booz_constants,
        booz_grids=booz_grids,
        surface_indices=surface_indices,
    )
    out = quasi_isodynamic_residual_from_boozer_output(
        field["booz"],
        nfp=int(field["nfp"]),
        weights=weights,
        nphi=nphi,
        nalpha=nalpha,
        n_bounce=n_bounce,
        include_bounce_endpoints=include_bounce_endpoints,
        softness=softness,
        width_weight=width_weight,
        branch_width_weight=branch_width_weight,
        branch_width_softness=branch_width_softness,
        profile_weight=profile_weight,
        shuffle_profile_weight=shuffle_profile_weight,
        shuffle_profile_softness=shuffle_profile_softness,
        shuffle_profile_nphi_out=shuffle_profile_nphi_out,
        weighted_shuffle_profile_weight=weighted_shuffle_profile_weight,
        weighted_shuffle_profile_softness=weighted_shuffle_profile_softness,
        aligned_profile_weight=aligned_profile_weight,
        aligned_profile_softness=aligned_profile_softness,
        aligned_profile_trap_level=aligned_profile_trap_level,
        aligned_profile_trap_softness=aligned_profile_trap_softness,
        phimin=phimin,
    )
    return {
        **out,
        "booz": field["booz"],
        "surfaces": field["surfaces"],
        "surface_indices": field["surface_indices"],
    }


def boozer_output_from_state(
    *,
    state,
    static,
    indata,
    signgs: int,
    surfaces,
    mboz: int = 12,
    nboz: int = 12,
    flux_local=None,
    prof_local=None,
    pressure_local=None,
    jit_booz: bool = False,
    booz_constants=None,
    booz_grids=None,
    surface_indices=None,
):
    """Evaluate differentiable Boozer ``|B|`` output directly from a VMEC state."""
    _require_jax()
    try:
        from booz_xform_jax import booz_xform_from_inputs, prepare_booz_xform_constants_from_inputs
    except Exception as exc:  # pragma: no cover - exercised only when optional dep missing
        raise ImportError(
            "quasi_isodynamic_residual_from_state requires booz_xform_jax. "
            "Install it with `pip install booz_xform_jax` or from github.com/uwplasma/booz_xform_jax."
        ) from exc

    from .booz_input import booz_xform_inputs_from_state

    surface_list = _as_surface_list(surfaces)
    surface_values = jnp.asarray(surface_list, dtype=jnp.float64)
    inputs = booz_xform_inputs_from_state(
        state=state,
        static=static,
        indata=indata,
        signgs=int(signgs),
        flux=flux_local,
        profiles_half=prof_local,
    )
    if pressure_local is not None:
        # The argument is accepted for symmetry with quasisymmetry helpers; the
        # Boozer input adapter already consumes ``prof_local``/``flux_local``.
        del pressure_local

    if booz_constants is None or booz_grids is None:
        constants, grids = prepare_booz_xform_constants_from_inputs(
            inputs=inputs,
            mboz=int(mboz),
            nboz=int(nboz),
            asym=bool(static.cfg.lasym),
        )
    else:
        constants, grids = booz_constants, booz_grids
    if surface_indices is None:
        surface_indices = _nearest_half_mesh_indices(surface_list, n_half=int(inputs.rmnc.shape[0]))
    booz = booz_xform_from_inputs(
        inputs=inputs,
        constants=constants,
        grids=grids,
        surface_indices=jnp.asarray(surface_indices, dtype=jnp.int32),
        jit=bool(jit_booz),
    )
    return {
        "booz": booz,
        "surfaces": surface_values,
        "surface_indices": jnp.asarray(surface_indices),
        "nfp": int(inputs.nfp),
    }


def mirror_ratio_penalty_from_state(
    *,
    state,
    static,
    indata,
    signgs: int,
    surfaces,
    weights: Iterable[float] | None = None,
    mboz: int = 12,
    nboz: int = 12,
    ntheta: int = 128,
    nphi: int = 128,
    phimin: float = 0.0,
    smooth_extrema: float = 0.0,
    smooth_penalty: float = 0.0,
    threshold: float = 0.21,
    flux_local=None,
    prof_local=None,
    pressure_local=None,
    jit_booz: bool = False,
    booz_constants=None,
    booz_grids=None,
    surface_indices=None,
):
    """Penalize mirror ratio from a solved VMEC state without any QI options."""

    field = boozer_output_from_state(
        state=state,
        static=static,
        indata=indata,
        signgs=signgs,
        surfaces=surfaces,
        mboz=mboz,
        nboz=nboz,
        flux_local=flux_local,
        prof_local=prof_local,
        pressure_local=pressure_local,
        jit_booz=jit_booz,
        booz_constants=booz_constants,
        booz_grids=booz_grids,
        surface_indices=surface_indices,
    )
    out = mirror_ratio_penalty_from_boozer_output(
        field["booz"],
        nfp=int(field["nfp"]),
        weights=weights,
        ntheta=ntheta,
        nphi=nphi,
        phimin=phimin,
        smooth_extrema=smooth_extrema,
        smooth_penalty=smooth_penalty,
        threshold=threshold,
    )
    return {
        **out,
        "booz": field["booz"],
        "surfaces": field["surfaces"],
        "surface_indices": field["surface_indices"],
    }


__all__ = sorted(
    name
    for name, value in globals().items()
    if getattr(value, "__module__", None) == __name__
    and not name.startswith("_")
    and name != "jax_sigmoid"
)
