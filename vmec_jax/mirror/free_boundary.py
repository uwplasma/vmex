"""Small free-boundary helpers for mirror coil-field studies.

This module is a bridge, not a free-boundary equilibrium solver.  It provides
ESSOS-compatible circular-loop coil parameters and field sampling on mirror
grids so later lanes can build LCFS and beta-scan drivers on tested pieces.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from vmec_jax._compat import jnp
from vmec_jax.external_fields import CoilFieldParams, sample_external_field_cylindrical

from .core.boundary import MirrorBoundary
from .core.grids import MirrorGrid
from .validation.coils import mirror_boundary_from_on_axis_bz


@dataclass(frozen=True)
class MirrorCircularCoils:
    """Axisymmetric circular coils for mirror free-boundary studies."""

    radii_m: Any
    z_centers_m: Any
    currents_a: Any
    n_segments: int = 128
    regularization_epsilon: float = 0.0
    chunk_size: int | None = None

    def __post_init__(self) -> None:
        radii = np.asarray(self.radii_m, dtype=float)
        z_centers = np.asarray(self.z_centers_m, dtype=float)
        currents = np.asarray(self.currents_a, dtype=float)
        if radii.ndim != 1 or z_centers.ndim != 1 or currents.ndim != 1:
            raise ValueError("radii_m, z_centers_m, and currents_a must be one-dimensional")
        if not (radii.shape == z_centers.shape == currents.shape):
            raise ValueError("radii_m, z_centers_m, and currents_a must have the same shape")
        if radii.size == 0:
            raise ValueError("at least one circular coil is required")
        if np.any(radii <= 0.0):
            raise ValueError("coil radii must be positive")
        if int(self.n_segments) < 8:
            raise ValueError("n_segments must be at least 8")
        if float(self.regularization_epsilon) < 0.0:
            raise ValueError("regularization_epsilon must be nonnegative")
        if self.chunk_size is not None and int(self.chunk_size) <= 0:
            raise ValueError("chunk_size must be positive when provided")
        object.__setattr__(self, "radii_m", radii)
        object.__setattr__(self, "z_centers_m", z_centers)
        object.__setattr__(self, "currents_a", currents)
        object.__setattr__(self, "n_segments", int(self.n_segments))
        object.__setattr__(self, "regularization_epsilon", float(self.regularization_epsilon))
        object.__setattr__(self, "chunk_size", None if self.chunk_size is None else int(self.chunk_size))

    @classmethod
    def symmetric_pair(
        cls,
        *,
        coil_radius_m: float,
        separation_m: float,
        current_a: float,
        center_z_m: float = 0.0,
        n_segments: int = 128,
        regularization_epsilon: float = 0.0,
        chunk_size: int | None = None,
    ) -> "MirrorCircularCoils":
        """Return two equal circular coils centered about ``center_z_m``."""

        separation = float(separation_m)
        if separation <= 0.0:
            raise ValueError("separation_m must be positive")
        half = 0.5 * separation
        return cls(
            radii_m=np.asarray([coil_radius_m, coil_radius_m], dtype=float),
            z_centers_m=np.asarray([float(center_z_m) - half, float(center_z_m) + half], dtype=float),
            currents_a=np.asarray([current_a, current_a], dtype=float),
            n_segments=n_segments,
            regularization_epsilon=regularization_epsilon,
            chunk_size=chunk_size,
        )

    def to_direct_coil_params(self) -> CoilFieldParams:
        """Return ESSOS-compatible direct-coil parameters."""

        return mirror_circular_coils_to_direct_params(self)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        return {
            "radii_m": self.radii_m.tolist(),
            "z_centers_m": self.z_centers_m.tolist(),
            "currents_a": self.currents_a.tolist(),
            "n_segments": int(self.n_segments),
            "regularization_epsilon": float(self.regularization_epsilon),
            "chunk_size": self.chunk_size,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MirrorCircularCoils":
        """Build circular coils from a JSON-friendly mapping."""

        return cls(
            radii_m=data["radii_m"],
            z_centers_m=data["z_centers_m"],
            currents_a=data["currents_a"],
            n_segments=int(data.get("n_segments", 128)),
            regularization_epsilon=float(data.get("regularization_epsilon", 0.0)),
            chunk_size=data.get("chunk_size"),
        )


@dataclass(frozen=True)
class MirrorExternalFieldSample:
    """External-field values sampled on a mirror axis or boundary grid."""

    r: Any
    theta: Any
    z: Any
    br: Any
    btheta: Any
    bz: Any
    bmag: Any


@dataclass(frozen=True)
class MirrorFreeBoundaryBetaCase:
    """One planned beta point for mirror free-boundary scans."""

    beta_percent: float
    beta_fraction: float
    pressure_scale: float

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-friendly representation."""

        return {
            "beta_percent": float(self.beta_percent),
            "beta_fraction": float(self.beta_fraction),
            "pressure_scale": float(self.pressure_scale),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MirrorFreeBoundaryBetaCase":
        """Build a beta case from a JSON-friendly mapping."""

        beta_percent = float(data["beta_percent"])
        beta_fraction = float(data.get("beta_fraction", 0.01 * beta_percent))
        pressure_scale = float(data["pressure_scale"])
        return cls(beta_percent=beta_percent, beta_fraction=beta_fraction, pressure_scale=pressure_scale)


@dataclass(frozen=True)
class MirrorFreeBoundaryCircularCoilScan:
    """Serializable setup for circular-coil mirror beta scans."""

    coils: MirrorCircularCoils
    beta_cases: tuple[MirrorFreeBoundaryBetaCase, ...]

    def __post_init__(self) -> None:
        if not self.beta_cases:
            raise ValueError("at least one beta case is required")
        object.__setattr__(self, "beta_cases", tuple(self.beta_cases))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        return {
            "coils": self.coils.to_dict(),
            "beta_cases": [case.to_dict() for case in self.beta_cases],
            "status": "setup_only_no_lcfs_solve",
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MirrorFreeBoundaryCircularCoilScan":
        """Build a scan setup from a JSON-friendly mapping."""

        return cls(
            coils=MirrorCircularCoils.from_dict(data["coils"]),
            beta_cases=tuple(MirrorFreeBoundaryBetaCase.from_dict(case) for case in data["beta_cases"]),
        )


def mirror_circular_coils_to_direct_params(coils: MirrorCircularCoils) -> CoilFieldParams:
    """Convert circular mirror coils to ESSOS-convention Fourier coil params."""

    radii = np.asarray(coils.radii_m, dtype=float)
    z_centers = np.asarray(coils.z_centers_m, dtype=float)
    dofs = np.zeros((radii.size, 3, 3), dtype=float)
    dofs[:, 0, 2] = radii
    dofs[:, 1, 1] = radii
    dofs[:, 2, 0] = z_centers
    return CoilFieldParams(
        base_curve_dofs=jnp.asarray(dofs),
        base_currents=jnp.asarray(coils.currents_a),
        n_segments=coils.n_segments,
        nfp=1,
        stellsym=False,
        current_scale=1.0,
        regularization_epsilon=coils.regularization_epsilon,
        chunk_size=coils.chunk_size,
    )


def _as_direct_coil_params(provider_params: Any) -> Any:
    if isinstance(provider_params, MirrorCircularCoils):
        return provider_params.to_direct_coil_params()
    return provider_params


def _sample_field(
    *,
    r: Any,
    theta: Any,
    z: Any,
    provider_params: Any,
    provider_kind: str,
    provider_static: Any | None,
) -> MirrorExternalFieldSample:
    br, btheta, bz = sample_external_field_cylindrical(
        provider_kind,
        provider_static,
        _as_direct_coil_params(provider_params),
        r,
        z,
        theta,
    )
    bmag = jnp.sqrt(br * br + btheta * btheta + bz * bz)
    return MirrorExternalFieldSample(r=r, theta=theta, z=z, br=br, btheta=btheta, bz=bz, bmag=bmag)


def sample_mirror_axis_external_field(
    grid: MirrorGrid,
    provider_params: Any,
    *,
    provider_kind: str = "direct_coils",
    provider_static: Any | None = None,
) -> MirrorExternalFieldSample:
    """Sample an external field on the mirror axis nodes."""

    z = jnp.asarray(grid.z)
    r = jnp.zeros_like(z)
    theta = jnp.zeros_like(z)
    return _sample_field(
        r=r,
        theta=theta,
        z=z,
        provider_params=provider_params,
        provider_kind=provider_kind,
        provider_static=provider_static,
    )


def sample_mirror_boundary_external_field(
    grid: MirrorGrid,
    boundary: MirrorBoundary,
    provider_params: Any,
    *,
    provider_kind: str = "direct_coils",
    provider_static: Any | None = None,
) -> MirrorExternalFieldSample:
    """Sample an external field on the mirror side boundary."""

    r = jnp.asarray(boundary.radius_on_grid_3d(grid))
    theta = jnp.broadcast_to(jnp.asarray(grid.theta)[:, None], r.shape)
    z = jnp.broadcast_to(jnp.asarray(grid.z)[None, :], r.shape)
    return _sample_field(
        r=r,
        theta=theta,
        z=z,
        provider_params=provider_params,
        provider_kind=provider_kind,
        provider_static=provider_static,
    )


def make_mirror_free_boundary_beta_cases(
    beta_percent: tuple[float, ...] = (1.0, 3.0, 10.0),
    *,
    pressure_scale_for_one_percent: float = 1.0,
) -> tuple[MirrorFreeBoundaryBetaCase, ...]:
    """Return planned pressure scales for a mirror beta scan."""

    scale = float(pressure_scale_for_one_percent)
    if scale <= 0.0:
        raise ValueError("pressure_scale_for_one_percent must be positive")
    cases = []
    for value in beta_percent:
        beta = float(value)
        if beta < 0.0:
            raise ValueError("beta_percent values must be nonnegative")
        cases.append(
            MirrorFreeBoundaryBetaCase(
                beta_percent=beta,
                beta_fraction=0.01 * beta,
                pressure_scale=scale * beta,
            )
        )
    return tuple(cases)


def make_mirror_free_boundary_circular_coil_scan(
    coils: MirrorCircularCoils,
    beta_percent: tuple[float, ...] = (1.0, 3.0, 10.0),
    *,
    pressure_scale_for_one_percent: float = 1.0,
) -> MirrorFreeBoundaryCircularCoilScan:
    """Return a serializable circular-coil beta-scan setup."""

    return MirrorFreeBoundaryCircularCoilScan(
        coils=coils,
        beta_cases=make_mirror_free_boundary_beta_cases(
            beta_percent,
            pressure_scale_for_one_percent=pressure_scale_for_one_percent,
        ),
    )


def write_mirror_free_boundary_circular_coil_scan(
    path: str | Path,
    scan: MirrorFreeBoundaryCircularCoilScan,
) -> Path:
    """Write a circular-coil beta-scan setup to JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scan.to_dict(), indent=2) + "\n")
    return path


def load_mirror_free_boundary_circular_coil_scan(path: str | Path) -> MirrorFreeBoundaryCircularCoilScan:
    """Load a circular-coil beta-scan setup from JSON."""

    return MirrorFreeBoundaryCircularCoilScan.from_dict(json.loads(Path(path).read_text()))


def mirror_boundary_from_external_axis_field(
    grid: MirrorGrid,
    axis_bz: Any,
    *,
    midplane_radius: float,
    radius_floor: float = 1.0e-4,
) -> MirrorBoundary:
    """Build an initial fixed boundary from sampled external on-axis ``Bz``."""

    z = np.asarray(grid.z, dtype=float)
    bz = np.asarray(axis_bz, dtype=float)
    if bz.shape != z.shape:
        raise ValueError(f"axis_bz must have shape {z.shape}, got {bz.shape}")
    radius = float(midplane_radius)
    if radius <= 0.0:
        raise ValueError("midplane_radius must be positive")
    midplane_bmag = float(np.interp(0.0, z, np.abs(bz)))
    if midplane_bmag <= 0.0:
        raise ValueError("midplane external field must be nonzero")
    psi_value = 0.5 * midplane_bmag * radius**2
    return mirror_boundary_from_on_axis_bz(psi_value, z, bz, radius_floor=radius_floor)


def initial_mirror_boundary_from_circular_coil_scan(
    grid: MirrorGrid,
    scan: MirrorFreeBoundaryCircularCoilScan,
    *,
    midplane_radius: float,
    radius_floor: float = 1.0e-4,
) -> MirrorBoundary:
    """Build the fixed-boundary baseline from a circular-coil scan setup."""

    axis_sample = sample_mirror_axis_external_field(grid, scan.coils)
    return mirror_boundary_from_external_axis_field(
        grid,
        axis_sample.bz,
        midplane_radius=midplane_radius,
        radius_floor=radius_floor,
    )
