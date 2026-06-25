"""Toroidal stellarator-mirror hybrid boundary helpers.

These helpers build ordinary VMEC fixed-boundary input data.  They are not part
of the open-ended mirror coordinate system: the surface is closed and toroidal,
with weakly shaped side arcs and localized stellarator-like shaping near the
corner arcs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .fourier import build_helical_basis, eval_fourier, project_to_modes
from .grids import AngleGrid
from .modes import vmec_mode_table
from .namelist import InData, minimal_fixed_boundary_indata


@dataclass(frozen=True)
class ToroidalHybridBoundarySamples:
    """Real-space samples for one VMEC field period."""

    theta: np.ndarray
    zeta: np.ndarray
    R: np.ndarray
    Z: np.ndarray
    side_weight: np.ndarray
    corner_weight: np.ndarray


def recommended_square_axis_nzeta(ntor: int, *, margin: int = 8, block: int = 8) -> int:
    """Return a conservative toroidal grid size for square-axis hybrids.

    The square-axis surface has localized side/corner structure, so VMEC runs
    are much less fragile when the toroidal collocation grid has room beyond
    the largest retained Fourier mode.  The result is rounded up to a small
    block size so CLI and VMEC2000 comparisons use reproducible grids.
    """

    ntor = int(ntor)
    margin = int(margin)
    block = int(block)
    if ntor < 0:
        raise ValueError("ntor must be nonnegative")
    if margin < 0:
        raise ValueError("margin must be nonnegative")
    if block <= 0:
        raise ValueError("block must be positive")
    raw = max(16, 2 * ntor + margin)
    return int(block * np.ceil(raw / block))


def sample_toroidal_stellarator_mirror_hybrid_boundary(
    *,
    ntheta: int = 64,
    nzeta: int = 64,
    major_radius: float = 1.15,
    minor_radius: float = 0.18,
    axis_oval: float = 0.10,
    side_minor_modulation: float = 0.10,
    side_elongation: float = 0.28,
    corner_amplitude: float = 0.035,
    corner_helicity: int = 1,
    corner_ellipticity: float = 0.18,
    corner_rotation: float = 0.35,
    side_power: float = 1.0,
    corner_power: float = 1.0,
) -> ToroidalHybridBoundarySamples:
    """Sample a toroidal hybrid LCFS over one field period.

    The side arcs, at ``zeta = 0`` and ``pi``, are nearly axisymmetric elongated
    cross sections.  The corner arcs, at ``zeta = pi/2`` and ``3*pi/2``, carry a
    localized finite-mode rotating ellipse plus a small optional ``m=2``
    helical perturbation.  ``side_power`` and ``corner_power`` sharpen or
    broaden those two regions without moving their centers.  The formula is
    stellarator symmetric, so it can be stored with the usual VMEC ``RBC``/``ZBS``
    boundary coefficients.
    """
    ntheta = int(ntheta)
    nzeta = int(nzeta)
    if ntheta < 8 or nzeta < 8:
        raise ValueError("ntheta and nzeta must be at least 8")
    if minor_radius <= 0.0 or major_radius <= minor_radius:
        raise ValueError("major_radius must exceed positive minor_radius")
    if int(corner_helicity) < 0:
        raise ValueError("corner_helicity must be nonnegative")
    corner_ellipticity = float(corner_ellipticity)
    corner_rotation = float(corner_rotation)
    if not np.isfinite(corner_ellipticity) or not (0.0 <= corner_ellipticity < 0.95):
        raise ValueError("corner_ellipticity must be finite and satisfy 0 <= corner_ellipticity < 0.95")
    if not np.isfinite(corner_rotation):
        raise ValueError("corner_rotation must be finite")
    side_power = float(side_power)
    corner_power = float(corner_power)
    if not np.isfinite(side_power) or side_power <= 0.0:
        raise ValueError("side_power must be finite and positive")
    if not np.isfinite(corner_power) or corner_power <= 0.0:
        raise ValueError("corner_power must be finite and positive")

    theta = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
    zeta = np.linspace(0.0, 2.0 * np.pi, nzeta, endpoint=False)
    theta2, zeta2 = np.meshgrid(theta, zeta, indexing="ij")

    side_weight = np.clip(np.cos(zeta2) ** 2, 0.0, 1.0) ** side_power
    corner_weight = np.clip(np.sin(zeta2) ** 2, 0.0, 1.0) ** corner_power
    axis = float(major_radius) + float(axis_oval) * np.cos(2.0 * zeta2)
    side_minor = float(minor_radius) * (1.0 + float(side_minor_modulation) * side_weight)
    elongation = 1.0 + float(side_elongation) * side_weight
    corner_shape = corner_ellipticity * corner_weight
    radial_semiaxis = side_minor * (1.0 + corner_shape)
    vertical_semiaxis = side_minor * elongation * (1.0 - corner_shape)
    rotation_harmonic = int(corner_helicity)
    corner_tilt = corner_rotation * corner_weight * np.sin(float(rotation_harmonic) * zeta2)
    corner_phase = 2.0 * theta2 - float(int(corner_helicity)) * zeta2

    R = (
        axis
        + radial_semiaxis * np.cos(theta2)
        - vertical_semiaxis * corner_tilt * np.sin(theta2)
        + float(corner_amplitude) * corner_weight * np.cos(corner_phase)
    )
    Z = (
        radial_semiaxis * corner_tilt * np.cos(theta2)
        + vertical_semiaxis * np.sin(theta2)
        + float(corner_amplitude) * corner_weight * np.sin(corner_phase)
    )

    if float(np.min(R)) <= 0.0:
        raise ValueError("boundary has nonpositive cylindrical R; reduce minor_radius or shaping amplitudes")

    return ToroidalHybridBoundarySamples(
        theta=theta,
        zeta=zeta,
        R=np.asarray(R, dtype=float),
        Z=np.asarray(Z, dtype=float),
        side_weight=np.asarray(side_weight, dtype=float),
        corner_weight=np.asarray(corner_weight, dtype=float),
    )


def sample_square_axis_stellarator_mirror_hybrid_boundary(
    *,
    ntheta: int = 64,
    nzeta: int = 128,
    axis_half_width: float = 1.5,
    axis_kind: str = "superellipse",
    axis_square_power: float = 5.0,
    axis_spline_corner_radius_factor: float = np.sqrt(2.0),
    minor_radius: float = 0.10,
    side_minor_modulation: float = 0.08,
    side_elongation: float = 0.25,
    corner_amplitude: float = 0.020,
    corner_helicity: int = 1,
    corner_ellipticity: float = 0.16,
    corner_rotation: float = 0.30,
    side_power: float = 1.4,
    corner_power: float = 1.4,
) -> ToroidalHybridBoundarySamples:
    """Sample a toroidal stellarator-mirror LCFS around a square-like axis.

    The magnetic axis is represented in polar form. ``axis_kind="superellipse"``
    keeps the original smooth polar superellipse. ``axis_kind="spline"`` uses a
    lower-bandwidth rounded-square envelope through side and corner radii; this
    is often easier to project to a compact VMEC Fourier boundary when the
    straight mirror-side intuition matters more than a sharp mathematical
    square. The surface is still stored in normal VMEC cylindrical coordinates,
    so the final equilibrium can use the ordinary toroidal fixed/free-boundary
    solver path.
    """

    ntheta = int(ntheta)
    nzeta = int(nzeta)
    if ntheta < 8 or nzeta < 16:
        raise ValueError("ntheta must be >= 8 and nzeta must be >= 16")
    if axis_half_width <= 0.0:
        raise ValueError("axis_half_width must be positive")
    axis_kind = str(axis_kind).strip().lower()
    if axis_kind not in {"superellipse", "spline", "spline_rounded_square", "rounded_square_spline"}:
        raise ValueError("axis_kind must be 'superellipse' or 'spline'")
    if axis_kind == "superellipse" and axis_square_power <= 2.0:
        raise ValueError("axis_square_power must exceed 2 for a square-like axis")
    axis_spline_corner_radius_factor = float(axis_spline_corner_radius_factor)
    if not np.isfinite(axis_spline_corner_radius_factor) or axis_spline_corner_radius_factor <= 1.0:
        raise ValueError("axis_spline_corner_radius_factor must be finite and greater than one")
    if minor_radius <= 0.0:
        raise ValueError("minor_radius must be positive")
    if int(corner_helicity) < 0:
        raise ValueError("corner_helicity must be nonnegative")

    theta = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
    zeta = np.linspace(0.0, 2.0 * np.pi, nzeta, endpoint=False)
    theta2, zeta2 = np.meshgrid(theta, zeta, indexing="ij")

    if axis_kind == "superellipse":
        c = np.cos(zeta)
        s = np.sin(zeta)
        axis_r = float(axis_half_width) / np.maximum(
            np.abs(c) ** float(axis_square_power) + np.abs(s) ** float(axis_square_power),
            np.finfo(float).tiny,
        ) ** (1.0 / float(axis_square_power))
    else:
        side_radius = float(axis_half_width)
        corner_boost = axis_spline_corner_radius_factor - 1.0
        # A single smooth fourfold envelope reaches its maximum on the rounded
        # corners and its minimum at side centers.  It deliberately avoids the
        # absolute-value cusp and high-mode tail of a sharp polar square.
        corner_profile = np.sin(2.0 * zeta) ** 2
        axis_r = side_radius * (1.0 + corner_boost * corner_profile)

    side_seed = 0.5 * (1.0 + np.cos(4.0 * zeta))
    side_weight_1d = np.clip(side_seed, 0.0, 1.0) ** float(side_power)
    corner_weight_1d = np.clip(1.0 - side_seed, 0.0, 1.0) ** float(corner_power)
    side_weight = np.broadcast_to(side_weight_1d[None, :], (ntheta, nzeta))
    corner_weight = np.broadcast_to(corner_weight_1d[None, :], (ntheta, nzeta))

    minor = float(minor_radius) * (1.0 + float(side_minor_modulation) * side_weight_1d)
    radial_semiaxis = minor * (1.0 + float(corner_ellipticity) * corner_weight_1d)
    vertical_semiaxis = minor * (1.0 + float(side_elongation) * side_weight_1d) * (
        1.0 - 0.5 * float(corner_ellipticity) * corner_weight_1d
    )
    tilt = float(corner_rotation) * corner_weight_1d * np.sin(float(int(corner_helicity)) * zeta)
    phase = 2.0 * theta2 - float(int(corner_helicity)) * zeta2
    local_r = radial_semiaxis[None, :] * np.cos(theta2)
    local_z = vertical_semiaxis[None, :] * np.sin(theta2)
    local_r = local_r + float(corner_amplitude) * corner_weight_1d[None, :] * np.cos(phase)
    local_z = local_z + float(corner_amplitude) * corner_weight_1d[None, :] * np.sin(phase)
    R = axis_r[None, :] + local_r * np.cos(tilt)[None, :] - local_z * np.sin(tilt)[None, :]
    Z = local_r * np.sin(tilt)[None, :] + local_z * np.cos(tilt)[None, :]
    if float(np.min(R)) <= 0.0:
        raise ValueError("boundary has nonpositive cylindrical R; reduce minor_radius or increase axis_half_width")

    return ToroidalHybridBoundarySamples(
        theta=theta,
        zeta=zeta,
        R=np.asarray(R, dtype=float),
        Z=np.asarray(Z, dtype=float),
        side_weight=np.asarray(side_weight, dtype=float),
        corner_weight=np.asarray(corner_weight, dtype=float),
    )


def _coeff_map_from_modes(
    values: np.ndarray, modes, *, coeff_tol: float, keep_00: bool = False
) -> dict[tuple[int, int], float]:
    out: dict[tuple[int, int], float] = {}
    for m_i, n_i, value in zip(
        np.asarray(modes.m, dtype=int), np.asarray(modes.n, dtype=int), np.asarray(values, dtype=float)
    ):
        if abs(float(value)) <= float(coeff_tol) and not (keep_00 and int(m_i) == 0 and int(n_i) == 0):
            continue
        out[(int(n_i), int(m_i))] = float(value)
    return out


def _indata_from_boundary_samples(
    *,
    samples: ToroidalHybridBoundarySamples,
    nfp: int,
    mpol: int,
    ntor: int,
    ns_array: int | list[int],
    niter_array: int | list[int],
    ftol_array: float | list[float],
    phiedge: float,
    coeff_tol: float,
) -> InData:
    modes = vmec_mode_table(mpol=mpol, ntor=ntor)
    grid = AngleGrid(theta=samples.theta, zeta=samples.zeta, nfp=nfp)
    basis = build_helical_basis(modes, grid)
    r_cos, r_sin = project_to_modes(samples.R, basis)
    z_cos, z_sin = project_to_modes(samples.Z, basis)
    r_cos = np.asarray(r_cos, dtype=float)
    r_sin = np.asarray(r_sin, dtype=float)
    z_cos = np.asarray(z_cos, dtype=float)
    z_sin = np.asarray(z_sin, dtype=float)

    rbs = _coeff_map_from_modes(r_sin, modes, coeff_tol=coeff_tol)
    zbc = _coeff_map_from_modes(z_cos, modes, coeff_tol=coeff_tol)
    if rbs or zbc:
        raise ValueError("sampled hybrid boundary is not stellarator symmetric at the requested tolerance")

    indata = minimal_fixed_boundary_indata(
        nfp=nfp,
        mpol=mpol,
        ntor=ntor,
        ns_array=ns_array,
        niter_array=niter_array,
        ftol_array=ftol_array,
        phiedge=phiedge,
    )
    indata.scalars.update(
        {
            "NFP": nfp,
            "MPOL": mpol,
            "NTOR": ntor,
            "LASYM": False,
            "PHIEDGE": float(phiedge),
            "NS_ARRAY": ns_array if isinstance(ns_array, list) else int(ns_array),
            "NITER_ARRAY": niter_array if isinstance(niter_array, list) else int(niter_array),
            "FTOL_ARRAY": ftol_array if isinstance(ftol_array, list) else float(ftol_array),
        }
    )
    indata.indexed = {
        "RBC": _coeff_map_from_modes(r_cos, modes, coeff_tol=coeff_tol, keep_00=True),
        "ZBS": _coeff_map_from_modes(z_sin, modes, coeff_tol=coeff_tol),
    }
    return indata


def toroidal_stellarator_mirror_hybrid_indata(
    *,
    nfp: int = 2,
    mpol: int = 5,
    ntor: int = 4,
    ntheta_fit: int = 64,
    nzeta_fit: int = 64,
    ns_array: int | list[int] = 15,
    niter_array: int | list[int] = 80,
    ftol_array: float | list[float] = 1.0e-9,
    phiedge: float = 0.05,
    coeff_tol: float = 1.0e-12,
    **sample_kwargs: Any,
) -> InData:
    """Return VMEC ``InData`` for the toroidal hybrid boundary.

    The boundary is sampled on a uniform tensor grid and projected onto the
    standard VMEC helical modes.  Defaults keep only low-order modes so the
    input remains small and useful for low-resolution solver smoke tests.
    """
    nfp = int(nfp)
    mpol = int(mpol)
    ntor = int(ntor)
    if nfp <= 0:
        raise ValueError("nfp must be positive")
    if mpol < 3:
        raise ValueError("mpol must be at least 3 so the corner m=2 shaping fits")
    corner_helicity = int(sample_kwargs.get("corner_helicity", 1))
    if ntor < corner_helicity + 2:
        raise ValueError("ntor must be at least corner_helicity + 2 to fit the localized corner shaping")

    samples = sample_toroidal_stellarator_mirror_hybrid_boundary(
        ntheta=int(ntheta_fit),
        nzeta=int(nzeta_fit),
        **sample_kwargs,
    )
    return _indata_from_boundary_samples(
        samples=samples,
        nfp=nfp,
        mpol=mpol,
        ntor=ntor,
        ns_array=ns_array,
        niter_array=niter_array,
        ftol_array=ftol_array,
        phiedge=phiedge,
        coeff_tol=coeff_tol,
    )


def square_axis_stellarator_mirror_hybrid_indata(
    *,
    nfp: int = 1,
    mpol: int = 5,
    ntor: int = 12,
    ntheta_fit: int = 64,
    nzeta_fit: int = 128,
    ns_array: int | list[int] = 9,
    niter_array: int | list[int] = 40,
    ftol_array: float | list[float] = 1.0e-8,
    phiedge: float = 0.04,
    coeff_tol: float = 1.0e-12,
    **sample_kwargs: Any,
) -> InData:
    """Return VMEC ``InData`` for the square-axis toroidal hybrid boundary."""

    nfp = int(nfp)
    mpol = int(mpol)
    ntor = int(ntor)
    if nfp <= 0:
        raise ValueError("nfp must be positive")
    if mpol < 3:
        raise ValueError("mpol must be at least 3 so the corner m=2 shaping fits")
    if ntor < 4:
        raise ValueError("ntor must be at least 4 to fit the square-like axis")
    samples = sample_square_axis_stellarator_mirror_hybrid_boundary(
        ntheta=int(ntheta_fit),
        nzeta=int(nzeta_fit),
        **sample_kwargs,
    )
    return _indata_from_boundary_samples(
        samples=samples,
        nfp=nfp,
        mpol=mpol,
        ntor=ntor,
        ns_array=ns_array,
        niter_array=niter_array,
        ftol_array=ftol_array,
        phiedge=phiedge,
        coeff_tol=coeff_tol,
    )


def square_axis_stellarator_mirror_hybrid_projection_error(
    *,
    nfp: int = 1,
    mpol: int = 5,
    ntor: int = 12,
    ntheta_fit: int = 64,
    nzeta_fit: int = 128,
    ntheta_eval: int | None = None,
    nzeta_eval: int | None = None,
    ns_array: int | list[int] = 9,
    niter_array: int | list[int] = 40,
    ftol_array: float | list[float] = 1.0e-8,
    phiedge: float = 0.04,
    coeff_tol: float = 1.0e-12,
    **sample_kwargs: Any,
) -> dict[str, float | int]:
    """Measure Fourier projection error for a square-axis hybrid boundary.

    The square-axis helper samples a smooth real-space boundary and then stores
    it as ordinary VMEC Fourier boundary coefficients.  This diagnostic reports
    how much the selected ``MPOL``/``NTOR`` truncation changes that sampled
    boundary before any equilibrium solve is attempted.
    """

    ntheta_eval = int(ntheta_fit if ntheta_eval is None else ntheta_eval)
    nzeta_eval = int(nzeta_fit if nzeta_eval is None else nzeta_eval)
    target = sample_square_axis_stellarator_mirror_hybrid_boundary(
        ntheta=ntheta_eval,
        nzeta=nzeta_eval,
        **sample_kwargs,
    )
    indata = square_axis_stellarator_mirror_hybrid_indata(
        nfp=nfp,
        mpol=mpol,
        ntor=ntor,
        ntheta_fit=ntheta_fit,
        nzeta_fit=nzeta_fit,
        ns_array=ns_array,
        niter_array=niter_array,
        ftol_array=ftol_array,
        phiedge=phiedge,
        coeff_tol=coeff_tol,
        **sample_kwargs,
    )
    reconstructed = evaluate_toroidal_hybrid_indata_boundary(
        indata,
        ntheta=ntheta_eval,
        nzeta=nzeta_eval,
    )
    dR = np.asarray(reconstructed.R, dtype=float) - np.asarray(target.R, dtype=float)
    dZ = np.asarray(reconstructed.Z, dtype=float) - np.asarray(target.Z, dtype=float)
    err = np.sqrt(dR * dR + dZ * dZ)
    target_scale = max(
        float(np.ptp(np.asarray(target.R, dtype=float))),
        float(np.ptp(np.asarray(target.Z, dtype=float))),
        np.finfo(float).tiny,
    )
    rms = float(np.sqrt(np.mean(err * err)))
    max_abs = float(np.max(err))
    max_abs_R = float(np.max(np.abs(dR)))
    max_abs_Z = float(np.max(np.abs(dZ)))
    max_abs_component = max(max_abs_R, max_abs_Z)
    return {
        "nfp": int(nfp),
        "mpol": int(mpol),
        "ntor": int(ntor),
        "ntheta_fit": int(ntheta_fit),
        "nzeta_fit": int(nzeta_fit),
        "ntheta_eval": int(ntheta_eval),
        "nzeta_eval": int(nzeta_eval),
        "max_abs_R_error": max_abs_R,
        "max_abs_Z_error": max_abs_Z,
        "max_abs_component_error": max_abs_component,
        "rms_R_error": float(np.sqrt(np.mean(dR * dR))),
        "rms_Z_error": float(np.sqrt(np.mean(dZ * dZ))),
        "max_abs_error": max_abs,
        "rms_error": rms,
        "max_abs_error_rel": float(max_abs / target_scale),
        "max_abs_component_error_rel": float(max_abs_component / target_scale),
        "rms_error_rel": float(rms / target_scale),
    }


def toroidal_stellarator_mirror_hybrid_metrics(samples: ToroidalHybridBoundarySamples) -> dict[str, float]:
    """Return lightweight geometry checks for a sampled hybrid boundary."""
    theta_reflect = (-np.arange(samples.theta.size)) % samples.theta.size
    zeta_reflect = (-np.arange(samples.zeta.size)) % samples.zeta.size
    R_reflect = samples.R[np.ix_(theta_reflect, zeta_reflect)]
    Z_reflect = samples.Z[np.ix_(theta_reflect, zeta_reflect)]
    side_cols = [0, samples.zeta.size // 2]
    corner_cols = [samples.zeta.size // 4, (3 * samples.zeta.size) // 4]
    side_r_span = float(np.mean(np.ptp(samples.R[:, side_cols], axis=0)))
    corner_r_span = float(np.mean(np.ptp(samples.R[:, corner_cols], axis=0)))
    orientation = toroidal_hybrid_cross_section_orientation(samples)
    anisotropy = toroidal_hybrid_cross_section_anisotropy(samples)
    side_weight = np.mean(samples.side_weight, axis=0)
    corner_weight = np.mean(samples.corner_weight, axis=0)
    side_region = side_weight >= 0.995
    corner_region = corner_weight >= 0.9
    anisotropy_threshold = 1.0e-14 + 1.0e-8 * float(np.max(anisotropy))
    valid_orientation = anisotropy > anisotropy_threshold
    side_valid = side_region & valid_orientation
    corner_valid = corner_region & valid_orientation
    return {
        "min_R": float(np.min(samples.R)),
        "max_R": float(np.max(samples.R)),
        "max_abs_Z": float(np.max(np.abs(samples.Z))),
        "stellsym_R_error": float(np.max(np.abs(samples.R - R_reflect))),
        "stellsym_Z_error": float(np.max(np.abs(samples.Z + Z_reflect))),
        "side_r_span_mean": side_r_span,
        "corner_r_span_mean": corner_r_span,
        "corner_weight_max": float(np.max(samples.corner_weight)),
        "side_weight_max": float(np.max(samples.side_weight)),
        "cross_section_orientation_span": float(np.ptp(orientation)),
        "side_orientation_span": float(np.ptp(orientation[side_region])) if np.any(side_region) else 0.0,
        "corner_orientation_span": float(np.ptp(orientation[corner_region])) if np.any(corner_region) else 0.0,
        "orientation_valid_fraction": float(np.mean(valid_orientation)) if valid_orientation.size else 0.0,
        "valid_cross_section_orientation_span": float(np.ptp(orientation[valid_orientation]))
        if np.any(valid_orientation)
        else 0.0,
        "valid_side_orientation_span": float(np.ptp(orientation[side_valid])) if np.any(side_valid) else 0.0,
        "valid_corner_orientation_span": float(np.ptp(orientation[corner_valid])) if np.any(corner_valid) else 0.0,
        "side_corner_weight_overlap_max": float(np.max(side_weight * corner_weight)),
        "cross_section_anisotropy_min": float(np.min(anisotropy)),
        "cross_section_anisotropy_max": float(np.max(anisotropy)),
    }


def _sample_RZ_arrays(samples: ToroidalHybridBoundarySamples) -> tuple[np.ndarray, np.ndarray]:
    R = np.asarray(samples.R, dtype=float)
    Z = np.asarray(samples.Z, dtype=float)
    if R.shape != Z.shape:
        raise ValueError("R and Z samples must have the same shape")
    if R.ndim != 2 or R.shape[0] < 3 or R.shape[1] < 1:
        raise ValueError("R and Z samples must have shape (ntheta, nzeta)")
    return R, Z


def toroidal_hybrid_cross_section_anisotropy(samples: ToroidalHybridBoundarySamples) -> np.ndarray:
    """Return the covariance anisotropy strength of each sampled cross section."""
    R, Z = _sample_RZ_arrays(samples)
    values = []
    for col in range(R.shape[1]):
        r = R[:, col] - float(np.mean(R[:, col]))
        z = Z[:, col] - float(np.mean(Z[:, col]))
        q1 = float(np.mean(r * r) - np.mean(z * z))
        q2 = float(2.0 * np.mean(r * z))
        values.append(np.hypot(q1, q2))
    return np.asarray(values, dtype=float)


def toroidal_hybrid_cross_section_orientation(samples: ToroidalHybridBoundarySamples) -> np.ndarray:
    """Return the unwrapped principal-axis angle of each sampled cross section.

    The angle is undefined where the cross-section covariance is isotropic.  Use
    `toroidal_hybrid_cross_section_anisotropy` to mask those points before
    interpreting orientation differences.
    """
    R, Z = _sample_RZ_arrays(samples)
    angles = []
    for col in range(R.shape[1]):
        r = R[:, col] - float(np.mean(R[:, col]))
        z = Z[:, col] - float(np.mean(Z[:, col]))
        rr = float(np.mean(r * r))
        zz = float(np.mean(z * z))
        rz = float(np.mean(r * z))
        angles.append(0.5 * np.arctan2(2.0 * rz, rr - zz))
    return 0.5 * np.unwrap(2.0 * np.asarray(angles, dtype=float))


def evaluate_toroidal_hybrid_indata_boundary(
    indata: InData,
    *,
    ntheta: int = 64,
    nzeta: int = 64,
) -> ToroidalHybridBoundarySamples:
    """Evaluate a generated hybrid input boundary on a uniform grid."""
    from .boundary import boundary_input_from_indata

    mpol = int(indata.get_int("MPOL", 5))
    ntor = int(indata.get_int("NTOR", 4))
    nfp = int(indata.get_int("NFP", 2))
    modes = vmec_mode_table(mpol=mpol, ntor=ntor)
    theta = np.linspace(0.0, 2.0 * np.pi, int(ntheta), endpoint=False)
    zeta = np.linspace(0.0, 2.0 * np.pi, int(nzeta), endpoint=False)
    grid = AngleGrid(theta=theta, zeta=zeta, nfp=nfp)
    basis = build_helical_basis(modes, grid)
    boundary = boundary_input_from_indata(indata, modes)
    R = np.asarray(eval_fourier(boundary.R_cos, boundary.R_sin, basis), dtype=float)
    Z = np.asarray(eval_fourier(boundary.Z_cos, boundary.Z_sin, basis), dtype=float)
    theta2, zeta2 = np.meshgrid(theta, zeta, indexing="ij")
    return ToroidalHybridBoundarySamples(
        theta=theta,
        zeta=zeta,
        R=R,
        Z=Z,
        side_weight=np.cos(zeta2) ** 2,
        corner_weight=np.sin(zeta2) ** 2,
    )
