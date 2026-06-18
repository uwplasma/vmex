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
    side_power: float = 1.0,
    corner_power: float = 1.0,
) -> ToroidalHybridBoundarySamples:
    """Sample a toroidal hybrid LCFS over one field period.

    The side arcs, at ``zeta = 0`` and ``pi``, are nearly axisymmetric elongated
    cross sections.  The corner arcs, at ``zeta = pi/2`` and ``3*pi/2``, carry a
    localized ``m=2`` helical perturbation.  ``side_power`` and
    ``corner_power`` sharpen or broaden those two regions without moving their
    centers.  The formula is stellarator symmetric, so it can be stored with the
    usual VMEC ``RBC``/``ZBS`` boundary coefficients.
    """
    ntheta = int(ntheta)
    nzeta = int(nzeta)
    if ntheta < 8 or nzeta < 8:
        raise ValueError("ntheta and nzeta must be at least 8")
    if minor_radius <= 0.0 or major_radius <= minor_radius:
        raise ValueError("major_radius must exceed positive minor_radius")
    if int(corner_helicity) < 0:
        raise ValueError("corner_helicity must be nonnegative")
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
    corner_phase = 2.0 * theta2 - float(int(corner_helicity)) * zeta2

    R = axis + side_minor * np.cos(theta2) + float(corner_amplitude) * corner_weight * np.cos(corner_phase)
    Z = side_minor * elongation * np.sin(theta2) + float(corner_amplitude) * corner_weight * np.sin(corner_phase)

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
    side_region = side_weight >= 0.5
    corner_region = corner_weight >= 0.5
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
