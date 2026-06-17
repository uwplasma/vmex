"""Nonlinear orchestration for fixed-boundary mirror stages."""

from __future__ import annotations

from dataclasses import dataclass

from ...core.boundary import MirrorBoundary
from ...core.grids import MirrorGrid
from ...core.profiles import IPrimeProfile, PressureProfile, PsiPrimeProfile
from ...core.state import MirrorState3D, MirrorStateAxisym
from ...kernels.constraints import project_axisym_state, project_state_3d
from .diagnostics import FixedBoundaryOptimizerSummary, FixedBoundaryTraceRow, trace_row_from_state
from .optimizers import (
    OptimizerOptions,
    projected_gradient_step,
    projected_gradient_step_3d,
    projected_lbfgs_solve,
    projected_lbfgs_solve_3d,
    projected_residual_newton_solve,
)


@dataclass(frozen=True)
class FixedBoundaryStageResult:
    """Result from one pressure-continuation stage."""

    state: MirrorStateAxisym | MirrorState3D
    trace: tuple[FixedBoundaryTraceRow, ...]
    optimizer_summary: FixedBoundaryOptimizerSummary | None = None


def _summary_from_run(
    run,
    *,
    options: OptimizerOptions,
    stage_index: int,
    pressure_scale: float,
) -> FixedBoundaryOptimizerSummary:
    return FixedBoundaryOptimizerSummary(
        stage_index=int(stage_index),
        pressure_scale=float(pressure_scale),
        optimizer=str(options.optimizer),
        success=bool(run.success),
        status=int(run.status),
        message=str(run.message),
        nit=int(run.nit),
        nfev=int(run.nfev),
        njev=int(run.njev),
        accepted=bool(run.accepted),
        rejection_reason=str(run.rejection_reason),
        candidate_energy_total=run.candidate_energy_total,
        candidate_residual_norm=run.candidate_residual_norm,
        candidate_min_a=run.candidate_min_a,
        candidate_min_sqrtg=run.candidate_min_sqrtg,
        candidate_energy_improved=run.candidate_energy_improved,
        candidate_positive_radius=run.candidate_positive_radius,
        candidate_positive_jacobian=run.candidate_positive_jacobian,
        residual_linear_maxiter_policy=run.residual_linear_maxiter_policy,
        residual_linear_solver=run.residual_linear_solver,
        residual_linear_maxiter_effective_max=run.residual_linear_maxiter_effective_max,
        residual_linear_maxiter_effective_last=run.residual_linear_maxiter_effective_last,
        residual_linear_istop_last=run.residual_linear_istop_last,
        residual_linear_iterations_last=run.residual_linear_iterations_last,
        residual_linear_iterations_total=run.residual_linear_iterations_total,
        residual_linear_residual_norm_last=run.residual_linear_residual_norm_last,
        residual_linear_normal_residual_norm_last=run.residual_linear_normal_residual_norm_last,
        residual_linear_condition_estimate_last=run.residual_linear_condition_estimate_last,
        residual_dense_step_norm_last=run.residual_dense_step_norm_last,
        residual_dense_step_cosine_last=run.residual_dense_step_cosine_last,
        residual_dense_step_relative_error_last=run.residual_dense_step_relative_error_last,
    )


def _summary_from_trace(
    trace: tuple[FixedBoundaryTraceRow, ...],
    *,
    options: OptimizerOptions,
    stage_index: int,
    pressure_scale: float,
) -> FixedBoundaryOptimizerSummary:
    final = trace[-1] if trace else None
    success = bool(final is not None and final.accepted and final.residual_norm <= options.tolerance)
    message = "projected residual is below tolerance" if success else "iteration stopped before tolerance"
    return FixedBoundaryOptimizerSummary(
        stage_index=int(stage_index),
        pressure_scale=float(pressure_scale),
        optimizer=str(options.optimizer),
        success=success,
        status=0 if success else 1,
        message=message,
        nit=len(trace),
        nfev=len(trace),
        njev=len(trace),
        accepted=bool(final is not None and final.accepted),
    )


def _stage_trace_row(
    state: MirrorStateAxisym | MirrorState3D,
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
    return FixedBoundaryStageResult(
        state=run.state,
        trace=tuple(trace),
        optimizer_summary=_summary_from_run(
            run,
            options=options,
            stage_index=stage_index,
            pressure_scale=pressure_scale,
        ),
    )


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
    trace_tuple = tuple(trace)
    return FixedBoundaryStageResult(
        state=state,
        trace=trace_tuple,
        optimizer_summary=_summary_from_trace(
            trace_tuple,
            options=options,
            stage_index=stage_index,
            pressure_scale=pressure_scale,
        ),
    )


def _residual_newton_stage(
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
    run = projected_residual_newton_solve(
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
    return FixedBoundaryStageResult(
        state=run.state,
        trace=tuple(trace),
        optimizer_summary=_summary_from_run(
            run,
            options=options,
            stage_index=stage_index,
            pressure_scale=pressure_scale,
        ),
    )


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
    if optimizer not in {"gradient_descent", "gd", "lbfgs", "l_bfgs_b", "residual_newton", "newton"}:
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
    elif optimizer in {"residual_newton", "newton"}:
        result = _residual_newton_stage(
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
    return FixedBoundaryStageResult(
        state=result.state,
        trace=initial_trace + result.trace,
        optimizer_summary=result.optimizer_summary,
    )


def _lbfgs_stage_3d(
    state: MirrorState3D,
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
    run = projected_lbfgs_solve_3d(
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
    return FixedBoundaryStageResult(
        state=run.state,
        trace=tuple(trace),
        optimizer_summary=_summary_from_run(
            run,
            options=options,
            stage_index=stage_index,
            pressure_scale=pressure_scale,
        ),
    )


def _gradient_descent_stage_3d(
    state: MirrorState3D,
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
        step = projected_gradient_step_3d(
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
    trace_tuple = tuple(trace)
    return FixedBoundaryStageResult(
        state=state,
        trace=trace_tuple,
        optimizer_summary=_summary_from_trace(
            trace_tuple,
            options=options,
            stage_index=stage_index,
            pressure_scale=pressure_scale,
        ),
    )


def solve_3d_fixed_boundary_stage(
    state: MirrorState3D,
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
    """Run one projected 3D fixed-boundary optimizer stage."""
    optimizer = options.optimizer.lower().replace("-", "_")
    if optimizer not in {"gradient_descent", "gd", "lbfgs", "l_bfgs_b"}:
        if optimizer in {"residual_newton", "newton"}:
            raise ValueError("optimizer='residual_newton' is currently implemented for axisymmetric mirror states only")
        raise ValueError(f"unsupported mirror fixed-boundary optimizer {options.optimizer!r}")

    state = project_state_3d(state, grid, boundary)
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
        result = _lbfgs_stage_3d(
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
        result = _gradient_descent_stage_3d(
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
    return FixedBoundaryStageResult(
        state=result.state,
        trace=initial_trace + result.trace,
        optimizer_summary=result.optimizer_summary,
    )
