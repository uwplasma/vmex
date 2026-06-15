"""Nonlinear orchestration for fixed-boundary mirror stages."""

from __future__ import annotations

from dataclasses import dataclass

from ...core.boundary import MirrorBoundary
from ...core.grids import MirrorGrid
from ...core.profiles import IPrimeProfile, PressureProfile, PsiPrimeProfile
from ...core.state import MirrorStateAxisym
from ...kernels.constraints import project_axisym_state
from .diagnostics import FixedBoundaryTraceRow, trace_row_from_state
from .optimizers import OptimizerOptions, projected_gradient_step


@dataclass(frozen=True)
class FixedBoundaryStageResult:
    """Result from one pressure-continuation stage."""

    state: MirrorStateAxisym
    trace: tuple[FixedBoundaryTraceRow, ...]


def solve_axisym_fixed_boundary_stage(
    state: MirrorStateAxisym,
    grid: MirrorGrid,
    boundary: MirrorBoundary,
    *,
    psi_prime: PsiPrimeProfile,
    i_prime: IPrimeProfile,
    pressure: PressureProfile,
    options: OptimizerOptions,
    stage_index: int,
    pressure_scale: float,
) -> FixedBoundaryStageResult:
    """Run one projected fixed-boundary optimizer stage."""
    optimizer = options.optimizer.lower().replace("-", "_")
    if optimizer in {"lbfgs", "l_bfgs_b"}:
        # The first production-safe path is projected gradient descent.  Keep
        # the LBFGS spelling accepted for configuration compatibility while the
        # constrained reduced-coordinate LBFGS path is being validated.
        optimizer = "gradient_descent"
    if optimizer not in {"gradient_descent", "gd"}:
        raise ValueError(f"unsupported mirror fixed-boundary optimizer {options.optimizer!r}")

    state = project_axisym_state(state, grid, boundary)
    trace: list[FixedBoundaryTraceRow] = [
        trace_row_from_state(
            state,
            grid,
            stage_index=stage_index,
            iteration=0,
            pressure_scale=pressure_scale,
            step_size=0.0,
            accepted=True,
            psi_prime=psi_prime,
            i_prime=i_prime,
            pressure=pressure,
            mu0=options.mu0,
        )
    ]

    for iteration in range(1, int(options.maxiter) + 1):
        step = projected_gradient_step(
            state,
            grid,
            boundary,
            psi_prime=psi_prime,
            i_prime=i_prime,
            pressure=pressure,
            options=options,
        )
        state = step.state
        trace.append(
            trace_row_from_state(
                state,
                grid,
                stage_index=stage_index,
                iteration=iteration,
                pressure_scale=pressure_scale,
                step_size=step.step_size,
                accepted=step.accepted,
                psi_prime=psi_prime,
                i_prime=i_prime,
                pressure=pressure,
                mu0=options.mu0,
            )
        )
        if step.residual_norm <= options.tolerance or not step.accepted:
            break

    return FixedBoundaryStageResult(state=state, trace=tuple(trace))
