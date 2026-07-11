"""Toroidal stellarator-mirror hybrid geometry for the ordinary VMEC solver.

The magnetic axis is a smooth square-like closed curve in the horizontal
plane. Long, nearly straight sides act as mirror sections; localized rotating
ellipses on the four corners provide stellarator shaping. The real-space
target is projected to standard VMEC ``RBC/ZBS`` coefficients, so no second
equilibrium representation is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fourier import mode_table
from .input import VmecInput


@dataclass(frozen=True)
class HybridBoundarySamples:
    """Boundary and axis samples on a uniform ``(theta, zeta)`` grid."""

    theta: np.ndarray
    zeta: np.ndarray
    axis_radius: np.ndarray
    radius: np.ndarray
    height: np.ndarray
    side_weight: np.ndarray
    corner_weight: np.ndarray


def sample_stellarator_mirror_hybrid(
    *,
    ntheta: int = 64,
    nzeta: int = 256,
    axis_half_width: float = 1.5,
    axis_square_power: int = 6,
    minor_radius: float = 0.10,
    side_elongation: float = 0.25,
    corner_ellipticity: float = 0.18,
    corner_rotation: float = 0.35,
    corner_helicity: int = 1,
    corner_localization: float = 2.0,
) -> HybridBoundarySamples:
    """Sample one closed square-torus LCFS with stellarator-shaped corners."""

    ntheta, nzeta = int(ntheta), int(nzeta)
    power = int(axis_square_power)
    if ntheta < 8 or nzeta < 32:
        raise ValueError("ntheta must be >= 8 and nzeta must be >= 32")
    if axis_half_width <= 0.0 or minor_radius <= 0.0:
        raise ValueError("axis_half_width and minor_radius must be positive")
    if minor_radius >= axis_half_width:
        raise ValueError("minor_radius must be smaller than axis_half_width")
    if power < 4 or power % 2:
        raise ValueError("axis_square_power must be an even integer >= 4")
    if not 0.0 <= corner_ellipticity < 0.8:
        raise ValueError("corner_ellipticity must satisfy 0 <= value < 0.8")
    if corner_localization <= 0.0:
        raise ValueError("corner_localization must be positive")

    theta = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
    zeta = np.linspace(0.0, 2.0 * np.pi, nzeta, endpoint=False)
    theta2, zeta2 = np.meshgrid(theta, zeta, indexing="ij")
    cosine, sine = np.cos(zeta), np.sin(zeta)
    axis_radius = float(axis_half_width) / (
        cosine**power + sine**power
    ) ** (1.0 / power)

    side_seed = np.clip(0.5 * (1.0 + np.cos(4.0 * zeta)), 0.0, 1.0)
    side = side_seed**float(corner_localization)
    corner = (1.0 - side_seed) ** float(corner_localization)
    radial_semiaxis = float(minor_radius) * (1.0 + float(corner_ellipticity) * corner)
    vertical_semiaxis = float(minor_radius) * (
        1.0 + float(side_elongation) * side - 0.5 * float(corner_ellipticity) * corner
    )
    tilt = (
        float(corner_rotation)
        * corner
        * np.sin(float(int(corner_helicity)) * zeta)
    )
    local_r = radial_semiaxis[None, :] * np.cos(theta2)
    local_z = vertical_semiaxis[None, :] * np.sin(theta2)
    radius = (
        axis_radius[None, :]
        + local_r * np.cos(tilt)[None, :]
        - local_z * np.sin(tilt)[None, :]
    )
    height = local_r * np.sin(tilt)[None, :] + local_z * np.cos(tilt)[None, :]
    if np.min(radius) <= 0.0:
        raise ValueError("hybrid boundary reaches nonpositive cylindrical radius")
    return HybridBoundarySamples(
        theta=theta,
        zeta=zeta,
        axis_radius=axis_radius,
        radius=radius,
        height=height,
        side_weight=np.broadcast_to(side[None, :], radius.shape),
        corner_weight=np.broadcast_to(corner[None, :], radius.shape),
    )


def _project_samples(
    samples: HybridBoundarySamples, *, mpol: int, ntor: int, nfp: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    modes = mode_table(mpol, ntor)
    theta2, zeta2 = np.meshgrid(samples.theta, samples.zeta, indexing="ij")
    phase = (
        np.asarray(modes.m)[:, None, None] * theta2[None]
        - np.asarray(modes.n)[:, None, None] * int(nfp) * zeta2[None]
    )
    cosine = np.cos(phase).reshape(len(modes.m), -1).T
    sine = np.sin(phase).reshape(len(modes.m), -1).T
    r_coeff = np.linalg.lstsq(cosine, samples.radius.reshape(-1), rcond=None)[0]
    active_sine = np.any(np.abs(sine) > 32.0 * np.finfo(float).eps, axis=0)
    z_coeff = np.zeros(len(modes.m))
    z_coeff[active_sine] = np.linalg.lstsq(
        sine[:, active_sine], samples.height.reshape(-1), rcond=None
    )[0]
    rbc = np.zeros((2 * ntor + 1, mpol))
    zbs = np.zeros_like(rbc)
    for m, n, rc, zs in zip(modes.m, modes.n, r_coeff, z_coeff, strict=True):
        rbc[int(n) + ntor, int(m)] = rc
        zbs[int(n) + ntor, int(m)] = zs
    reconstructed_r = (cosine @ r_coeff).reshape(samples.radius.shape)
    reconstructed_z = (sine @ z_coeff).reshape(samples.height.shape)
    return rbc, zbs, reconstructed_r, reconstructed_z


def stellarator_mirror_hybrid_input(
    *,
    mpol: int = 6,
    ntor: int = 16,
    nfp: int = 1,
    ns_array: tuple[int, ...] = (9, 15),
    ftol_array: tuple[float, ...] = (1.0e-8, 1.0e-11),
    niter_array: tuple[int, ...] = (1000, 2000),
    phiedge: float = 0.04,
    **sample_kwargs,
) -> VmecInput:
    """Project the hybrid target into an ordinary fixed-boundary VMEC input."""

    mpol, ntor, nfp = int(mpol), int(ntor), int(nfp)
    if mpol < 3 or ntor < 4 or nfp != 1:
        raise ValueError("hybrid projection currently requires mpol>=3, ntor>=4, nfp=1")
    samples = sample_stellarator_mirror_hybrid(**sample_kwargs)
    rbc, zbs, _, _ = _project_samples(samples, mpol=mpol, ntor=ntor, nfp=nfp)
    axis_modes = np.column_stack(
        [np.cos(n * samples.zeta) for n in range(ntor + 1)]
    )
    raxis_c = np.linalg.lstsq(axis_modes, samples.axis_radius, rcond=None)[0]
    return VmecInput(
        nfp=nfp,
        mpol=mpol,
        ntor=ntor,
        ns_array=ns_array,
        ftol_array=ftol_array,
        niter_array=niter_array,
        phiedge=phiedge,
        lfreeb=False,
        rbc=rbc,
        zbs=zbs,
        raxis_c=raxis_c,
    )


def hybrid_projection_error(
    *, mpol: int, ntor: int, nfp: int = 1, **sample_kwargs
) -> dict[str, float]:
    """Return maximum and RMS component errors of the VMEC projection."""

    samples = sample_stellarator_mirror_hybrid(**sample_kwargs)
    _, _, radius, height = _project_samples(
        samples, mpol=int(mpol), ntor=int(ntor), nfp=int(nfp)
    )
    error_r = radius - samples.radius
    error_z = height - samples.height
    return {
        "maximum": float(max(np.max(np.abs(error_r)), np.max(np.abs(error_z)))),
        "rms": float(np.sqrt(np.mean(error_r**2 + error_z**2))),
    }
