"""Trace diagnostics for fixed-boundary mirror solves."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...core.grids import MirrorGrid
from ...core.profiles import IPrimeProfile, PressureProfile, PsiPrimeProfile
from ...core.state import MirrorState3D, MirrorStateAxisym
from ...kernels.fields import evaluate_axisym_field, evaluate_field_3d
from ...kernels.forces import axisym_projected_energy_residual, projected_energy_residual_3d
from ...kernels.geometry import evaluate_axisym_geometry, evaluate_geometry_3d
from ...kernels.residuals import field_diagnostics


@dataclass(frozen=True)
class FixedBoundaryTraceRow:
    """One fixed-boundary solve trace row."""

    stage_index: int
    iteration: int
    pressure_scale: float
    energy_total: float
    residual_norm: float
    fsq: float
    normalized_force: float
    active_force_dof: int
    min_sqrtg: float
    max_sqrtg: float
    min_bmag: float
    max_bmag: float
    mirror_ratio: float
    step_size: float
    accepted: bool


def ensure_finite_pressure_scale(scale: float) -> float:
    """Validate and return a finite pressure continuation scale."""
    scale = float(scale)
    if not np.isfinite(scale):
        raise ValueError("pressure continuation scales must be finite")
    return scale


def trace_row_from_state(
    state: MirrorStateAxisym | MirrorState3D,
    grid: MirrorGrid,
    *,
    stage_index: int,
    iteration: int,
    pressure_scale: float,
    step_size: float,
    accepted: bool,
    psi_prime: PsiPrimeProfile,
    i_prime: IPrimeProfile,
    pressure: PressureProfile,
    mu0: float,
) -> FixedBoundaryTraceRow:
    """Build a diagnostic trace row from a state."""
    if np.asarray(state.a).ndim == 3:
        geometry = evaluate_geometry_3d(state, grid)
        field = evaluate_field_3d(state, grid, geometry, psi_prime=psi_prime, i_prime=i_prime)
        residual = projected_energy_residual_3d(
            state,
            grid,
            psi_prime=psi_prime,
            i_prime=i_prime,
            pressure=pressure,
            mu0=mu0,
        )
    else:
        geometry = evaluate_axisym_geometry(state, grid)
        field = evaluate_axisym_field(state, grid, geometry, psi_prime=psi_prime, i_prime=i_prime)
        residual = axisym_projected_energy_residual(
            state,
            grid,
            psi_prime=psi_prime,
            i_prime=i_prime,
            pressure=pressure,
            mu0=mu0,
        )
    diagnostics = field_diagnostics(field, grid)
    return FixedBoundaryTraceRow(
        stage_index=int(stage_index),
        iteration=int(iteration),
        pressure_scale=float(pressure_scale),
        energy_total=float(residual.energy),
        residual_norm=float(residual.norm),
        fsq=float(residual.fsq),
        normalized_force=float(residual.normalized_force),
        active_force_dof=int(residual.active_dof),
        min_sqrtg=float(np.min(geometry.sqrtg)),
        max_sqrtg=float(np.max(geometry.sqrtg)),
        min_bmag=diagnostics.min_bmag,
        max_bmag=diagnostics.max_bmag,
        mirror_ratio=diagnostics.mirror_ratio,
        step_size=float(step_size),
        accepted=bool(accepted),
    )
