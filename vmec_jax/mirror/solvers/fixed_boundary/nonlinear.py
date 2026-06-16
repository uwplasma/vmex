"""Nonlinear orchestration for fixed-boundary mirror stages."""

from __future__ import annotations

from dataclasses import dataclass

from ...core.boundary import MirrorBoundary
from ...core.grids import MirrorGrid
from ...core.profiles import IPrimeProfile, PressureProfile, PsiPrimeProfile
from ...core.state import MirrorStateAxisym
from ...kernels.constraints import project_axisym_state
from .diagnostics import FixedBoundaryTraceRow, trace_row_from_state
from .optimizers import OptimizerOptions, projected_gradient_step, projected_lbfgs_solve


@dataclass(frozen=True)
class FixedBoundaryStageResult:
    """Result from one pressure-continuation stage."""

    state: MirrorStateAxisym
    trace: tuple[FixedBoundaryTraceRow, ...]


def _stage_trace_row(
    state: MirrorStateAxisym,
    grid: MirrorGrid,
    *,
    psi_prime: PsiPrimeProfile,
    i_prime: IPrimeProfile,
    pressure: PressureProfile,
    options: OptimizerOptions,
    stage_index: int,
    pressure_scale: float,
    iteration: int,
    step_size: float,
    accepted: bool,
) -> FixedBoundaryTraceRow:
    return trace_row_from_state(
        state,
        grid,
        stage_index=stage_index,
        iteration=iteration,
        pressure_scale=pressure_scale,
        step_size=step_size,
        accepted=accepted,
        psi_prime=psi_prime,
        i_prime=i_prime,
        pressure=pressure,
        mu0=options.mu0,
    )


def _lbfgs_stage(
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
    run = projected_lbfgs_solve(
        state,
        grid,
        boundary,
        psi_prime=psi_prime,
        i_prime=i_prime,
        pressure=pressure,
        options=options,
    )
    trace = [
        _stage_trace_row(
            step.state,
            grid,
            psi_prime=psi_prime,
            i_prime=i_prime,
            pressure=pressure,
            options=options,
            stage_index=stage_index,
            pressure_scale=pressure_scale,
            iteration=iteration,
            step_size=step.step_size,
            accepted=step.accepted,
        )
        for iteration, step in enumerate(run.steps, start=1)
    ]
    return FixedBoundaryStageResult(state=run.state, trace=tuple(trace))


def _gradient_descent_stage(
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
    trace: list[FixedBoundaryTraceRow] = []
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
            _stage_trace_row(
                state,
                grid,
                psi_prime=psi_prime,
                i_prime=i_prime,
                pressure=pressure,
                options=options,
                stage_index=stage_index,
                pressure_scale=pressure_scale,
                iteration=iteration,
                step_size=step.step_size,
                accepted=step.accepted,
            )
        )
        if step.residual_norm <= options.tolerance or not step.accepted:
            break
    return FixedBoundaryStageResult(state=state, trace=tuple(trace))


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
    if optimizer not in {"gradient_descent", "gd", "lbfgs", "l_bfgs_b"}:
        raise ValueError(f"unsupported mirror fixed-boundary optimizer {options.optimizer!r}")

    state = project_axisym_state(state, grid, boundary)
    initial_trace = (
        _stage_trace_row(
            state,
            grid,
            psi_prime=psi_prime,
            i_prime=i_prime,
            pressure=pressure,
            options=options,
            stage_index=stage_index,
            pressure_scale=pressure_scale,
            iteration=0,
            step_size=0.0,
            accepted=True,
        ),
    )

    if optimizer in {"lbfgs", "l_bfgs_b"}:
        result = _lbfgs_stage(
            state,
            grid,
            boundary,
            psi_prime=psi_prime,
            i_prime=i_prime,
            pressure=pressure,
            options=options,
            stage_index=stage_index,
            pressure_scale=pressure_scale,
        )
    else:
        result = _gradient_descent_stage(
            state,
            grid,
            boundary,
            psi_prime=psi_prime,
            i_prime=i_prime,
            pressure=pressure,
            options=options,
            stage_index=stage_index,
            pressure_scale=pressure_scale,
        )
    return FixedBoundaryStageResult(state=result.state, trace=initial_trace + result.trace)
