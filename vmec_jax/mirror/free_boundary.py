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


@dataclass(frozen=True)
class MirrorLCFSDiagnostic:
    """Side-boundary diagnostic for future mirror free-boundary updates."""

    theta: Any
    z: Any
    boundary_r: Any
    boundary_dr_dz: Any
    external_bnormal: Any
    external_bnormal_rms: float
    external_bnormal_max: float
    pressure_balance: Any
    pressure_balance_rms: float
    pressure_balance_max: float
    internal_bmag: Any
    external_bmag: Any
    edge_pressure: float


@dataclass(frozen=True)
class MirrorLCFSMerit:
    """Dimensionless merit for accepting LCFS pilot updates."""

    value: float
    pressure_balance_rms: float
    external_bnormal_rms: float
    external_bmag_rms: float
    pressure_scale: float
    bnormal_scale: float
    bnormal_weight: float


@dataclass(frozen=True)
class MirrorLCFSUpdateProposal:
    """One damped axisymmetric side-boundary update proposal."""

    z: Any
    xi: Any
    old_radius: Any
    new_radius: Any
    delta_radius: Any
    pressure_response: Any
    pressure_balance_before: Any
    pressure_balance_predicted: Any
    pressure_balance_rms_before: float
    pressure_balance_rms_predicted: float
    damping: float
    max_relative_step: float
    cap_taper_power: float
    smoothing_passes: int
    preserve_caps: bool
    boundary: MirrorBoundary
    strategy: str


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


def mirror_lcfs_diagnostic(
    output: Any, external_sample: MirrorExternalFieldSample, *, mu0: float = 1.0
) -> MirrorLCFSDiagnostic:
    """Return side-boundary normal-field and total-pressure diagnostics.

    The diagnostic target is intentionally local to the side boundary.  It
    gives later LCFS update work a tested quantity to reduce without pretending
    that the current fixed-boundary baseline is already a free-boundary solve.
    """

    theta = np.asarray(output.theta, dtype=float)
    z = np.asarray(output.z, dtype=float)
    boundary_r = np.asarray(output.geometry.boundary_r, dtype=float)
    if boundary_r.shape != (theta.size, z.size):
        raise ValueError(f"boundary_r must have shape {(theta.size, z.size)}, got {boundary_r.shape}")
    if z.size < 2:
        raise ValueError("at least two axial nodes are required for LCFS diagnostics")
    edge_internal_bmag = np.asarray(output.field.bmag[-1], dtype=float)
    external_br = np.asarray(external_sample.br, dtype=float)
    external_bz = np.asarray(external_sample.bz, dtype=float)
    external_bmag = np.asarray(external_sample.bmag, dtype=float)
    expected_shape = boundary_r.shape
    if (
        external_br.shape != expected_shape
        or external_bz.shape != expected_shape
        or external_bmag.shape != expected_shape
    ):
        raise ValueError("external field sample must have shape (ntheta, nxi)")
    if edge_internal_bmag.shape != expected_shape:
        raise ValueError("internal edge |B| must have shape (ntheta, nxi)")
    if float(mu0) <= 0.0:
        raise ValueError("mu0 must be positive")

    external_bnormal, dr_dz = mirror_external_bnormal(boundary_r, z, external_sample, return_dr_dz=True)
    edge_pressure = float(np.asarray(output.profiles.pressure, dtype=float)[-1])
    pressure_balance = edge_pressure + (edge_internal_bmag**2 - external_bmag**2) / (2.0 * float(mu0))
    return MirrorLCFSDiagnostic(
        theta=theta,
        z=z,
        boundary_r=boundary_r,
        boundary_dr_dz=dr_dz,
        external_bnormal=external_bnormal,
        external_bnormal_rms=float(np.sqrt(np.mean(external_bnormal**2))),
        external_bnormal_max=float(np.max(np.abs(external_bnormal))),
        pressure_balance=pressure_balance,
        pressure_balance_rms=float(np.sqrt(np.mean(pressure_balance**2))),
        pressure_balance_max=float(np.max(np.abs(pressure_balance))),
        internal_bmag=edge_internal_bmag,
        external_bmag=external_bmag,
        edge_pressure=edge_pressure,
    )


def mirror_external_bnormal(
    boundary_r: Any,
    z: Any,
    external_sample: MirrorExternalFieldSample,
    *,
    return_dr_dz: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Return external normal field on a mirror side boundary."""

    boundary_r = np.asarray(boundary_r, dtype=float)
    z = np.asarray(z, dtype=float)
    external_br = np.asarray(external_sample.br, dtype=float)
    external_bz = np.asarray(external_sample.bz, dtype=float)
    if boundary_r.ndim != 2:
        raise ValueError("boundary_r must have shape (ntheta, nxi)")
    if z.ndim != 1 or z.size != boundary_r.shape[-1]:
        raise ValueError("z must be one-dimensional with length nxi")
    if z.size < 2:
        raise ValueError("at least two axial nodes are required")
    if external_br.shape != boundary_r.shape or external_bz.shape != boundary_r.shape:
        raise ValueError("external field sample must have shape (ntheta, nxi)")
    edge_order = 2 if z.size > 2 else 1
    dr_dz = np.gradient(boundary_r, z, axis=-1, edge_order=edge_order)
    normal_scale = np.sqrt(1.0 + dr_dz**2)
    external_bnormal = external_br / normal_scale - external_bz * dr_dz / normal_scale
    if return_dr_dz:
        return external_bnormal, dr_dz
    return external_bnormal


def mirror_external_pressure_balance_response(
    diagnostic: MirrorLCFSDiagnostic,
    provider_params: Any,
    *,
    provider_kind: str = "direct_coils",
    provider_static: Any | None = None,
    radius_step_fraction: float = 1.0e-3,
    radius_step_min: float = 1.0e-5,
    radius_floor: float = 1.0e-6,
    mu0: float = 1.0,
) -> np.ndarray:
    """Estimate ``d(pressure_balance)/dr`` from external magnetic pressure.

    This keeps the internal fixed-boundary equilibrium frozen and only measures
    how the external coil magnetic pressure changes when the side boundary is
    moved radially.  The resulting response is suitable for a damped first
    LCFS proposal, not for claiming a converged free-boundary equilibrium.
    """

    if float(mu0) <= 0.0:
        raise ValueError("mu0 must be positive")
    if float(radius_step_fraction) <= 0.0:
        raise ValueError("radius_step_fraction must be positive")
    if float(radius_step_min) <= 0.0:
        raise ValueError("radius_step_min must be positive")
    if float(radius_floor) <= 0.0:
        raise ValueError("radius_floor must be positive")

    radius = np.asarray(diagnostic.boundary_r, dtype=float)
    step = np.maximum(float(radius_step_fraction) * np.maximum(radius, float(radius_floor)), float(radius_step_min))
    r_plus = radius + step
    r_minus = np.maximum(radius - step, float(radius_floor))
    denominator = r_plus - r_minus
    if np.any(denominator <= 0.0):
        raise ValueError("finite-difference radius step collapsed")

    plus = _sample_field(
        r=jnp.asarray(r_plus),
        theta=jnp.asarray(diagnostic.theta),
        z=jnp.asarray(diagnostic.z),
        provider_params=provider_params,
        provider_kind=provider_kind,
        provider_static=provider_static,
    )
    minus = _sample_field(
        r=jnp.asarray(r_minus),
        theta=jnp.asarray(diagnostic.theta),
        z=jnp.asarray(diagnostic.z),
        provider_params=provider_params,
        provider_kind=provider_kind,
        provider_static=provider_static,
    )
    bmag_plus = np.asarray(plus.bmag, dtype=float)
    bmag_minus = np.asarray(minus.bmag, dtype=float)
    return -(bmag_plus**2 - bmag_minus**2) / (2.0 * float(mu0) * denominator)


def mirror_lcfs_merit(
    diagnostic: MirrorLCFSDiagnostic,
    *,
    pressure_scale: float | None = None,
    bnormal_scale: float | None = None,
    bnormal_weight: float = 1.0,
) -> MirrorLCFSMerit:
    """Return a dimensionless side-boundary merit for LCFS pilot steps."""

    pressure_rms = float(diagnostic.pressure_balance_rms)
    bnormal_rms = float(diagnostic.external_bnormal_rms)
    external_bmag_rms = float(np.sqrt(np.mean(np.asarray(diagnostic.external_bmag, dtype=float) ** 2)))
    pressure_scale = pressure_rms if pressure_scale is None else float(pressure_scale)
    bnormal_scale = external_bmag_rms if bnormal_scale is None else float(bnormal_scale)
    bnormal_weight = float(bnormal_weight)
    if pressure_scale <= 0.0:
        raise ValueError("pressure_scale must be positive")
    if bnormal_scale <= 0.0:
        raise ValueError("bnormal_scale must be positive")
    if bnormal_weight < 0.0:
        raise ValueError("bnormal_weight must be nonnegative")
    pressure_term = pressure_rms / pressure_scale
    bnormal_term = bnormal_rms / bnormal_scale
    return MirrorLCFSMerit(
        value=float(np.sqrt(pressure_term**2 + bnormal_weight * bnormal_term**2)),
        pressure_balance_rms=pressure_rms,
        external_bnormal_rms=bnormal_rms,
        external_bmag_rms=external_bmag_rms,
        pressure_scale=float(pressure_scale),
        bnormal_scale=float(bnormal_scale),
        bnormal_weight=bnormal_weight,
    )


def propose_axisymmetric_mirror_lcfs_update(
    diagnostic: MirrorLCFSDiagnostic,
    pressure_response: Any,
    *,
    damping: float = 0.25,
    max_relative_step: float = 0.05,
    radius_floor: float = 1.0e-4,
    preserve_caps: bool = True,
    cap_taper_power: float = 2.0,
    smoothing_passes: int = 1,
) -> MirrorLCFSUpdateProposal:
    """Return a damped axisymmetric radius proposal from pressure imbalance.

    The update is a clipped Newton step for the theta-averaged side-boundary
    pressure-balance residual.  Cap radii are preserved by default, and the
    update is tapered smoothly toward the caps to avoid creating large axial
    side-boundary slopes.
    """

    damping = float(damping)
    max_relative_step = float(max_relative_step)
    radius_floor = float(radius_floor)
    cap_taper_power = float(cap_taper_power)
    smoothing_passes = int(smoothing_passes)
    if not (0.0 < damping <= 1.0):
        raise ValueError("damping must be in (0, 1]")
    if max_relative_step <= 0.0:
        raise ValueError("max_relative_step must be positive")
    if radius_floor <= 0.0:
        raise ValueError("radius_floor must be positive")
    if cap_taper_power < 0.0:
        raise ValueError("cap_taper_power must be nonnegative")
    if smoothing_passes < 0:
        raise ValueError("smoothing_passes must be nonnegative")

    z = np.asarray(diagnostic.z, dtype=float)
    if z.ndim != 1 or z.size < 2 or not np.all(np.diff(z) > 0.0):
        raise ValueError("diagnostic z nodes must be a strictly increasing one-dimensional array")
    radius = np.mean(np.asarray(diagnostic.boundary_r, dtype=float), axis=0)
    residual = np.mean(np.asarray(diagnostic.pressure_balance, dtype=float), axis=0)
    response = np.asarray(pressure_response, dtype=float)
    if response.shape == np.asarray(diagnostic.boundary_r).shape:
        response = np.mean(response, axis=0)
    elif response.shape != radius.shape:
        raise ValueError("pressure_response must have shape (ntheta, nxi) or (nxi,)")
    if not (np.all(np.isfinite(radius)) and np.all(np.isfinite(residual)) and np.all(np.isfinite(response))):
        raise ValueError("LCFS update inputs must be finite")

    raw_delta = np.zeros_like(radius)
    active = np.abs(response) > np.finfo(float).eps
    raw_delta[active] = -residual[active] / response[active]
    limit = max_relative_step * np.maximum(radius, radius_floor)
    delta = np.clip(damping * raw_delta, -limit, limit)
    if preserve_caps and cap_taper_power > 0.0:
        normalized_z = (z - z[0]) / (z[-1] - z[0])
        delta *= np.sin(np.pi * normalized_z) ** cap_taper_power
    for _ in range(smoothing_passes):
        if delta.size > 2:
            smoothed = delta.copy()
            smoothed[1:-1] = 0.25 * delta[:-2] + 0.5 * delta[1:-1] + 0.25 * delta[2:]
            delta = np.clip(smoothed, -limit, limit)
    if preserve_caps:
        delta[0] = 0.0
        delta[-1] = 0.0
    new_radius = np.maximum(radius + delta, radius_floor)
    delta = new_radius - radius
    predicted = residual + response * delta
    xi = 2.0 * (z - z[0]) / (z[-1] - z[0]) - 1.0
    return MirrorLCFSUpdateProposal(
        z=z,
        xi=xi,
        old_radius=radius,
        new_radius=new_radius,
        delta_radius=delta,
        pressure_response=response,
        pressure_balance_before=residual,
        pressure_balance_predicted=predicted,
        pressure_balance_rms_before=float(np.sqrt(np.mean(residual**2))),
        pressure_balance_rms_predicted=float(np.sqrt(np.mean(predicted**2))),
        damping=damping,
        max_relative_step=max_relative_step,
        cap_taper_power=cap_taper_power,
        smoothing_passes=smoothing_passes,
        preserve_caps=bool(preserve_caps),
        boundary=MirrorBoundary.tabulated_radius(xi, new_radius),
        strategy="local_pressure",
    )


def propose_axisymmetric_mirror_lcfs_scale_update(
    diagnostic: MirrorLCFSDiagnostic,
    pressure_response: Any,
    *,
    max_relative_step: float = 0.05,
    radius_floor: float = 1.0e-4,
) -> MirrorLCFSUpdateProposal:
    """Return a shape-preserving radius-scale proposal.

    This candidate keeps the axial boundary shape smooth and changes the flux
    tube radius by one global scale factor chosen from the linearized pressure
    response.  It is useful as a normal-field-friendly alternative to nodal
    pressure updates.
    """

    max_relative_step = float(max_relative_step)
    radius_floor = float(radius_floor)
    if max_relative_step <= 0.0:
        raise ValueError("max_relative_step must be positive")
    if radius_floor <= 0.0:
        raise ValueError("radius_floor must be positive")

    z = np.asarray(diagnostic.z, dtype=float)
    if z.ndim != 1 or z.size < 2 or not np.all(np.diff(z) > 0.0):
        raise ValueError("diagnostic z nodes must be a strictly increasing one-dimensional array")
    radius = np.mean(np.asarray(diagnostic.boundary_r, dtype=float), axis=0)
    residual = np.mean(np.asarray(diagnostic.pressure_balance, dtype=float), axis=0)
    response = np.asarray(pressure_response, dtype=float)
    if response.shape == np.asarray(diagnostic.boundary_r).shape:
        response = np.mean(response, axis=0)
    elif response.shape != radius.shape:
        raise ValueError("pressure_response must have shape (ntheta, nxi) or (nxi,)")
    if not (np.all(np.isfinite(radius)) and np.all(np.isfinite(residual)) and np.all(np.isfinite(response))):
        raise ValueError("LCFS scale-update inputs must be finite")

    direction = radius.copy()
    linear_response = response * direction
    denom = float(np.dot(linear_response, linear_response))
    scale_step = 0.0 if denom <= np.finfo(float).eps else -float(np.dot(residual, linear_response)) / denom
    scale_step = float(np.clip(scale_step, -max_relative_step, max_relative_step))
    delta = scale_step * direction
    new_radius = np.maximum(radius + delta, radius_floor)
    delta = new_radius - radius
    predicted = residual + response * delta
    xi = 2.0 * (z - z[0]) / (z[-1] - z[0]) - 1.0
    return MirrorLCFSUpdateProposal(
        z=z,
        xi=xi,
        old_radius=radius,
        new_radius=new_radius,
        delta_radius=delta,
        pressure_response=response,
        pressure_balance_before=residual,
        pressure_balance_predicted=predicted,
        pressure_balance_rms_before=float(np.sqrt(np.mean(residual**2))),
        pressure_balance_rms_predicted=float(np.sqrt(np.mean(predicted**2))),
        damping=abs(scale_step),
        max_relative_step=max_relative_step,
        cap_taper_power=0.0,
        smoothing_passes=0,
        preserve_caps=False,
        boundary=MirrorBoundary.tabulated_radius(xi, new_radius),
        strategy="scale_pressure",
    )
