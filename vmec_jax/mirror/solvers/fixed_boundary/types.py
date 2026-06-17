"""Shared fixed-boundary optimizer data containers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...core.state import MirrorState3D, MirrorStateAxisym


@dataclass(frozen=True)
class OptimizerOptions:
    """Numerical options for fixed-boundary optimizer stages."""

    optimizer: str = "gradient_descent"
    maxiter: int = 50
    tolerance: float = 1.0e-8
    step_size: float = 1.0e-3
    min_step_size: float = 1.0e-12
    ftol: float | None = None
    line_search_steps: int = 16
    reduced_coordinate_scaling: str = "geometry"
    residual_linear_maxiter: int = 16
    residual_linear_maxiter_policy: str = "adaptive"
    residual_linear_adaptive_factor: float = 6.0
    residual_linear_solver: str = "lsmr"
    residual_preconditioner: str = "radial_xi_tridi"
    residual_radial_alpha: float = 0.5
    residual_lambda_alpha: float = 0.5
    residual_xi_alpha: float = 0.2
    mu0: float = 4.0e-7 * np.pi


@dataclass(frozen=True)
class OptimizerStep:
    """Accepted optimizer step payload."""

    state: MirrorStateAxisym | MirrorState3D
    energy: float
    residual_norm: float
    step_size: float
    accepted: bool


@dataclass(frozen=True)
class OptimizerRun:
    """Multi-step optimizer payload."""

    state: MirrorStateAxisym | MirrorState3D
    steps: tuple[OptimizerStep, ...]
    success: bool = False
    status: int = 0
    message: str = ""
    nit: int = 0
    nfev: int = 0
    njev: int = 0
    accepted: bool = True
    rejection_reason: str = ""
    candidate_energy_total: float | None = None
    candidate_residual_norm: float | None = None
    candidate_min_a: float | None = None
    candidate_min_sqrtg: float | None = None
    candidate_energy_improved: bool | None = None
    candidate_positive_radius: bool | None = None
    candidate_positive_jacobian: bool | None = None
    residual_linear_maxiter_policy: str = ""
    residual_linear_solver: str = ""
    residual_linear_maxiter_effective_max: int | None = None
    residual_linear_maxiter_effective_last: int | None = None


@dataclass(frozen=True)
class _CandidateDiagnostics:
    """Acceptance diagnostics for the raw optimizer candidate."""

    accepted: bool
    reason: str
    min_a: float
    min_sqrtg: float
    energy_improved: bool
    positive_radius: bool
    positive_jacobian: bool
