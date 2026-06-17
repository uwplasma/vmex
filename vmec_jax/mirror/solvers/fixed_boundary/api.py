"""Public fixed-boundary mirror solver API."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...core.boundary import MirrorBoundary
from ...core.config import MirrorConfig
from ...core.grids import MirrorGrid
from ...core.profiles import IPrimeProfile, PressureProfile, PsiPrimeProfile
from ...core.state import MirrorState3D, MirrorStateAxisym
from ...kernels.constraints import project_axisym_state, project_state_3d
from .continuation import pressure_stage_profiles
from .diagnostics import FixedBoundaryOptimizerSummary, FixedBoundaryTraceRow, trace_row_from_state
from .nonlinear import solve_3d_fixed_boundary_stage, solve_axisym_fixed_boundary_stage
from .types import OptimizerOptions


@dataclass(frozen=True)
class MirrorSolveOptions:
    """Options for the first fixed-boundary mirror solve path."""

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
    residual_compare_dense_step: bool = False
    residual_preconditioner: str = "radial_xi_tridi"
    residual_radial_alpha: float = 0.5
    residual_lambda_alpha: float = 0.5
    residual_xi_alpha: float = 0.2
    pressure_continuation: tuple[float, ...] = (1.0,)
    mu0: float = 4.0e-7 * 3.141592653589793

    def optimizer_options(self) -> OptimizerOptions:
        """Return the optimizer-specific options object."""
        return OptimizerOptions(
            optimizer=self.optimizer,
            maxiter=self.maxiter,
            tolerance=self.tolerance,
            step_size=self.step_size,
            min_step_size=self.min_step_size,
            ftol=self.ftol,
            line_search_steps=self.line_search_steps,
            reduced_coordinate_scaling=self.reduced_coordinate_scaling,
            residual_linear_maxiter=self.residual_linear_maxiter,
            residual_linear_maxiter_policy=self.residual_linear_maxiter_policy,
            residual_linear_adaptive_factor=self.residual_linear_adaptive_factor,
            residual_linear_solver=self.residual_linear_solver,
            residual_compare_dense_step=self.residual_compare_dense_step,
            residual_preconditioner=self.residual_preconditioner,
            residual_radial_alpha=self.residual_radial_alpha,
            residual_lambda_alpha=self.residual_lambda_alpha,
            residual_xi_alpha=self.residual_xi_alpha,
            mu0=self.mu0,
        )


@dataclass(frozen=True)
class MirrorFixedBoundaryResult:
    """Result from the axisymmetric fixed-boundary mirror solver."""

    config: MirrorConfig
    grid: MirrorGrid
    boundary: MirrorBoundary
    state: MirrorStateAxisym | MirrorState3D
    psi_prime: PsiPrimeProfile
    i_prime: IPrimeProfile
    pressure: PressureProfile
    options: MirrorSolveOptions
    trace: tuple[FixedBoundaryTraceRow, ...]
    optimizer_summaries: tuple[FixedBoundaryOptimizerSummary, ...] = ()

    @property
    def final_trace(self) -> FixedBoundaryTraceRow:
        return self.trace[-1]


def run_mirror_fixed_boundary(
    config: MirrorConfig,
    boundary: MirrorBoundary,
    *,
    psi_prime: PsiPrimeProfile | None = None,
    i_prime: IPrimeProfile | None = None,
    pressure: PressureProfile | None = None,
    initial_state: MirrorStateAxisym | MirrorState3D | None = None,
    options: MirrorSolveOptions | None = None,
) -> MirrorFixedBoundaryResult:
    """Run the first axisymmetric fixed-boundary mirror solve workflow."""
    options = options or MirrorSolveOptions()
    psi_prime = psi_prime or PsiPrimeProfile.constant(0.01)
    i_prime = i_prime or IPrimeProfile.zero()
    pressure = pressure or PressureProfile.zero()

    grid = config.build_grid()
    use_3d = isinstance(initial_state, MirrorState3D) or (
        initial_state is None and (not boundary.is_axisymmetric or grid.ntheta > 1)
    )
    if initial_state is not None and np.asarray(initial_state.a).ndim == 3:
        use_3d = True
    if use_3d:
        if initial_state is not None and not isinstance(initial_state, MirrorState3D):
            raise ValueError("3D mirror solves require a MirrorState3D initial_state")
        state = initial_state or MirrorState3D.from_boundary(grid, boundary)
        state = project_state_3d(state, grid, boundary)
    else:
        if initial_state is not None and not isinstance(initial_state, MirrorStateAxisym):
            raise ValueError("axisymmetric mirror solves require a MirrorStateAxisym initial_state")
        state = initial_state or MirrorStateAxisym.from_boundary(grid, boundary)
        state = project_axisym_state(state, grid, boundary)

    trace: list[FixedBoundaryTraceRow] = []
    optimizer_summaries: list[FixedBoundaryOptimizerSummary] = []
    stages = pressure_stage_profiles(pressure, options.pressure_continuation)
    if options.maxiter == 0:
        stage_index, pressure_scale, stage_pressure = stages[-1]
        trace.append(
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
                pressure=stage_pressure,
                mu0=options.mu0,
            )
        )
        return MirrorFixedBoundaryResult(
            config=config,
            grid=grid,
            boundary=boundary,
            state=state,
            psi_prime=psi_prime,
            i_prime=i_prime,
            pressure=pressure,
            options=options,
            trace=tuple(trace),
            optimizer_summaries=(),
        )

    for stage_index, pressure_scale, stage_pressure in stages:
        if use_3d:
            stage_result = solve_3d_fixed_boundary_stage(
                state,
                grid,
                boundary,
                psi_prime=psi_prime,
                i_prime=i_prime,
                pressure=stage_pressure,
                options=options.optimizer_options(),
                stage_index=stage_index,
                pressure_scale=pressure_scale,
            )
        else:
            stage_result = solve_axisym_fixed_boundary_stage(
                state,
                grid,
                boundary,
                psi_prime=psi_prime,
                i_prime=i_prime,
                pressure=stage_pressure,
                options=options.optimizer_options(),
                stage_index=stage_index,
                pressure_scale=pressure_scale,
            )
        state = stage_result.state
        trace.extend(stage_result.trace)
        if stage_result.optimizer_summary is not None:
            optimizer_summaries.append(stage_result.optimizer_summary)

    return MirrorFixedBoundaryResult(
        config=config,
        grid=grid,
        boundary=boundary,
        state=state,
        psi_prime=psi_prime,
        i_prime=i_prime,
        pressure=pressure,
        options=options,
        trace=tuple(trace),
        optimizer_summaries=tuple(optimizer_summaries),
    )
