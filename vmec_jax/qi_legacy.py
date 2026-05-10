"""Legacy quasi-isodynamic diagnostics.

This module contains a NumPy/SciPy implementation of the branch
squash/stretch/shuffle diagnostic used by the reference
``omnigenity_optimization`` workflows.  It is intentionally not JAX
differentiable.  Use it to validate and rank differentiable QI objectives, not
inside an autodiff optimization loop.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.interpolate import UnivariateSpline

__all__ = [
    "legacy_qi_branch_shuffle_diagnostic_from_boozer_output",
]


def _legacy_get_branches(phi_b: np.ndarray, b_alpha: np.ndarray, b_level: float) -> tuple[float, float]:
    """Return the first and last bounce crossings used by qi_functions.py."""
    diffs = b_alpha - float(b_level)
    diffsgn = diffs[:-1] * diffs[1:]
    inds = np.where(diffsgn < 0)[0]
    if b_level <= np.min(b_alpha):
        phi_min = float(phi_b[int(np.argmin(b_alpha))])
        return phi_min, phi_min
    if b_level >= np.max(b_alpha):
        return float(phi_b[0]), float(phi_b[-1])
    if len(inds) < 2:
        inds = np.where(diffsgn <= 0)[0]
        split = None
        for idx in range(1, len(inds)):
            if inds[idx] != inds[idx - 1] + 1:
                split = [inds[idx - 1], inds[-1]]
                break
        if split is not None:
            inds = np.asarray(split, dtype=int)
    if len(inds) > 2:
        inds = np.asarray([inds[0], inds[-1]], dtype=int)
    if len(inds) < 2:
        return float(phi_b[0]), float(phi_b[-1])

    def _crossing(ind: int, *, right_endpoint: bool) -> float:
        dy = b_alpha[ind] - b_alpha[ind + 1]
        dx = phi_b[ind] - phi_b[ind + 1]
        slope = dy / dx
        intercept = b_alpha[ind] - slope * phi_b[ind]
        if slope == 0.0:
            return float(phi_b[ind + int(right_endpoint)])
        return float((b_level - intercept) / slope)

    return _crossing(int(inds[0]), right_endpoint=False), _crossing(int(inds[1]), right_endpoint=True)


def _nfp_from_boozer_output(booz: dict[str, Any]) -> int:
    if "nfp_b" not in booz:
        raise ValueError("nfp must be supplied when booz output does not include nfp_b")
    nfp_value = np.asarray(booz["nfp_b"]).ravel()
    if nfp_value.size == 0:
        raise ValueError("nfp_b is empty")
    return int(nfp_value[0])


def legacy_qi_branch_shuffle_diagnostic_from_boozer_output(
    booz: dict[str, Any],
    *,
    nfp: int | None = None,
    nphi: int = 151,
    nalpha: int = 31,
    n_bounce: int = 51,
    nphi_out: int = 2000,
    phimin: float = 0.0,
    weights=None,
) -> dict[str, Any]:
    """Evaluate the legacy branch-shuffle QI diagnostic on Boozer modes.

    Parameters are intentionally named to match the historical
    ``omnigenity_optimization.qi_functions`` controls.  The returned
    ``total`` is the squared norm of the residual vector.

    This diagnostic uses branch extraction and piecewise-linear splines.  It is
    useful as a reference-quality ranking metric, but it is not smooth and must
    not be used as the differentiable objective in ``jax.grad``/``vjp`` paths.
    """
    if nfp is None:
        nfp = _nfp_from_boozer_output(booz)
    nfp = int(nfp)
    if nfp < 1:
        raise ValueError("nfp must be positive")
    if int(nphi) < 4 or int(nalpha) < 2 or int(n_bounce) < 2 or int(nphi_out) < 4:
        raise ValueError("legacy QI diagnostic requires nphi>=4, nalpha>=2, n_bounce>=2, nphi_out>=4")

    bmnc_b = np.asarray(booz["bmnc_b"], dtype=float)
    bmns_raw = booz.get("bmns_b")
    bmns_b = None if bmns_raw is None else np.asarray(bmns_raw, dtype=float)
    xm_b = np.asarray(booz["ixm_b"], dtype=float)
    xn_b = np.asarray(booz["ixn_b"], dtype=float)
    iota_b = np.asarray(booz["iota_b"], dtype=float)
    if bmnc_b.ndim != 2:
        raise ValueError(f"bmnc_b must have shape (nsurf, nmodes), got {bmnc_b.shape}")
    if bmns_b is not None and bmns_b.shape != bmnc_b.shape:
        raise ValueError("bmns_b must have the same shape as bmnc_b")
    if bmnc_b.shape[1] != xm_b.shape[0] or xm_b.shape[0] != xn_b.shape[0]:
        raise ValueError("Boozer mode arrays must have the same mode dimension as bmnc_b")

    nsurf = bmnc_b.shape[0]
    if iota_b.shape[0] != nsurf:
        raise ValueError("iota_b must have one value per Boozer surface")
    if weights is None:
        weights_arr = np.ones(nsurf)
    else:
        weights_arr = np.sqrt(np.asarray(weights, dtype=float))
    if weights_arr.shape[0] != nsurf:
        raise ValueError("weights must have one value per Boozer surface")

    phimax = float(phimin) + 2.0 * np.pi / float(nfp)
    phi_1d = np.linspace(float(phimin), phimax, int(nphi))
    phis2d = np.tile(phi_1d, (int(nalpha), 1)).T
    b_levels = np.linspace(0.0, 1.0, int(n_bounce))
    out = np.zeros((nsurf, int(nalpha), int(nphi_out)))
    surface_totals = []

    for surf in range(nsurf):
        iota = float(iota_b[surf])
        theta_min = -iota * float(phimin)
        theta_max = theta_min + 2.0 * np.pi
        thetas2d = np.tile(np.linspace(theta_min, theta_max, int(nalpha)), (int(nphi), 1)) + iota * phis2d
        angle = thetas2d[:, :, None] * xm_b[None, None, :] - phis2d[:, :, None] * xn_b[None, None, :]
        bmag = np.sum(bmnc_b[surf][None, None, :] * np.cos(angle), axis=-1)
        if bmns_b is not None:
            bmag = bmag + np.sum(bmns_b[surf][None, None, :] * np.sin(angle), axis=-1)
        bmin = float(np.min(bmag))
        bmax = float(np.max(bmag))
        denom = max(bmax - bmin, np.finfo(float).tiny)
        bnorm = (bmag - bmin) / denom

        bounce_widths = np.zeros((int(nalpha), int(n_bounce)))
        phi_crossings = np.zeros((int(nalpha), 2 * int(n_bounce) - 1))
        shuffled_crossings = np.zeros_like(phi_crossings)
        weights_alpha = np.zeros(int(nalpha))

        for ialpha in range(int(nalpha)):
            profile = np.array(bnorm[:, ialpha], copy=True)
            phi_profile = phis2d[:, ialpha]
            min_index = int(np.argmin(profile))

            left = np.array(profile[: min_index + 1], copy=True)
            phi_left = phi_profile[: min_index + 1]
            right = np.array(profile[min_index:], copy=True)
            phi_right = phi_profile[min_index:]

            left_max = int(np.argmax(left))
            left[:left_max] = left[left_max]
            for idx in range(len(left) - 1):
                if left[idx] <= left[idx + 1]:
                    stop = len(left) - 1
                    for jdx in range(idx + 1, len(left)):
                        if left[jdx] < left[idx]:
                            stop = jdx
                            break
                    left[idx:stop] = left[idx]

            right_max = int(np.argmax(right))
            right[right_max:] = right[right_max]
            for jdx in range(len(right) - 1, 1, -1):
                if right[jdx - 1] >= right[jdx]:
                    stop = 0
                    for kdx in range(jdx - 1, 1, -1):
                        if right[kdx] < right[jdx]:
                            stop = kdx
                            break
                    right[stop + 1 : jdx] = right[jdx]

            pmax = 50
            pmin = 15
            if len(left) > 1:
                x_left = (phi_left - phi_left[0]) / max(phi_left[-1] - phi_left[0], np.finfo(float).eps)
                left_half = x_left < 0.5
                f_left = left_half * (1.0 - left[0]) * ((np.cos(2 * np.pi * x_left) + 1.0) / 2.0) ** pmax
                f_left += (~left_half) * (-left[-1]) * ((np.cos(2 * np.pi * x_left) + 1.0) / 2.0) ** pmin
                left = left + f_left
            if len(right) > 1:
                x_right = (phi_right - phi_right[0]) / max(phi_right[-1] - phi_right[0], np.finfo(float).eps)
                right_half = x_right < 0.5
                f_right = right_half * (-right[0]) * ((np.cos(2 * np.pi * x_right) + 1.0) / 2.0) ** pmin
                f_right += (~right_half) * (1.0 - right[-1]) * ((np.cos(2 * np.pi * x_right) + 1.0) / 2.0) ** pmax
                right = right + f_right

            squashed = np.concatenate((left[:-1], right))
            diff = profile - squashed
            weights_alpha[ialpha] = (phimax - float(phimin)) / max(
                float(UnivariateSpline(phi_profile, diff * diff, k=1, s=0).integral(float(phimin), phimax)),
                np.finfo(float).eps,
            )

            for jlevel, level in enumerate(b_levels):
                phi_left_cross, phi_right_cross = _legacy_get_branches(phi_profile, squashed, float(level))
                bounce_widths[ialpha, jlevel] = phi_right_cross - phi_left_cross
                phi_crossings[ialpha, int(n_bounce) - jlevel - 1] = phi_left_cross
                phi_crossings[ialpha, int(n_bounce) + jlevel - 1] = phi_right_cross

        weights_alpha = weights_alpha / max(float(np.sum(weights_alpha)), np.finfo(float).eps)
        mean_widths = np.sum(bounce_widths * weights_alpha[:, None], axis=0)
        shuffled_levels = np.concatenate((np.flip(b_levels), b_levels[1:]))
        phi_eval = np.linspace(float(phimin), phimax, int(nphi_out))

        for ialpha in range(int(nalpha)):
            delta_widths = 0.5 * (bounce_widths[ialpha, :] - mean_widths)
            left_crossings = np.array(phi_crossings[ialpha, : int(n_bounce)], copy=True)
            right_crossings = np.array(phi_crossings[ialpha, int(n_bounce) - 1 :], copy=True)
            left_crossings += np.flip(delta_widths)
            right_crossings -= delta_widths
            for idx in range(int(n_bounce) - 1):
                if left_crossings[idx + 1] - left_crossings[idx] < 0:
                    right_crossings[-idx - 2] += left_crossings[idx] - left_crossings[idx + 1] + 1.0e-12
                    left_crossings[idx + 1] = left_crossings[idx] + 1.0e-12
                if right_crossings[-idx - 1] - right_crossings[-idx - 2] < 0:
                    left_crossings[idx + 1] += right_crossings[-idx - 1] - right_crossings[-idx - 2] - 1.0e-12
                    right_crossings[-idx - 2] = right_crossings[-idx - 1] - 1.0e-12
            shuffled_crossings[ialpha, : int(n_bounce)] = left_crossings
            shuffled_crossings[ialpha, int(n_bounce) - 1 :] = right_crossings
            for idx in range(1, shuffled_crossings.shape[1]):
                if shuffled_crossings[ialpha, idx] <= shuffled_crossings[ialpha, idx - 1]:
                    shuffled_crossings[ialpha, idx] = shuffled_crossings[ialpha, idx - 1] + 1.0e-12

            original = UnivariateSpline(phis2d[:, ialpha], bnorm[:, ialpha], k=1, s=0)
            shuffled = UnivariateSpline(shuffled_crossings[ialpha, :], shuffled_levels, k=1, s=0)
            out[surf, ialpha, :] = weights_arr[surf] * (shuffled(phi_eval) - original(phi_eval)) / np.sqrt(
                int(nphi_out)
            )

        out[surf, :, :] = out[surf, :, :] / np.sqrt(int(nalpha))
        surface_totals.append(float(np.dot(out[surf].ravel(), out[surf].ravel())))

    residuals = out.ravel()
    return {
        "residuals1d": residuals,
        "total": float(np.dot(residuals, residuals)),
        "surface_totals": surface_totals,
        "residual_size": int(residuals.size),
    }
