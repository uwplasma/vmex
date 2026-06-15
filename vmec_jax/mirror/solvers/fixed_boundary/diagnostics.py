"""Trace diagnostics for fixed-boundary mirror solves."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...core.grids import MirrorGrid
from ...core.profiles import IPrimeProfile, PressureProfile, PsiPrimeProfile
from ...core.state import MirrorStateAxisym
from ...kernels.fields import evaluate_axisym_field
from ...kernels.forces import axisym_projected_energy_residual
from ...kernels.geometry import evaluate_axisym_geometry
from ...kernels.residuals import field_diagnostics


@dataclass(frozen=True)
class FixedBoundaryTraceRow:
    """One fixed-boundary solve trace row."""

    stage_index: int
    iteration: int
    pressure_scale: float
    energy_total: float
    residual_norm: float
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
    state: MirrorStateAxisym,
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
    geometry = evaluate_axisym_geometry(state, grid)
    field = evaluate_axisym_field(state, grid, geometry, psi_prime=psi_prime, i_prime=i_prime)
    diagnostics = field_diagnostics(field, grid)
    residual = axisym_projected_energy_residual(
        state,
        grid,
        psi_prime=psi_prime,
        i_prime=i_prime,
        pressure=pressure,
        mu0=mu0,
    )
    return FixedBoundaryTraceRow(
        stage_index=int(stage_index),
        iteration=int(iteration),
        pressure_scale=float(pressure_scale),
        energy_total=float(residual.energy),
        residual_norm=float(residual.norm),
        min_sqrtg=float(np.min(geometry.sqrtg)),
        max_sqrtg=float(np.max(geometry.sqrtg)),
        min_bmag=diagnostics.min_bmag,
        max_bmag=diagnostics.max_bmag,
        mirror_ratio=diagnostics.mirror_ratio,
        step_size=float(step_size),
        accepted=bool(accepted),
    )
