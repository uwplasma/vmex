"""Public fixed-boundary mirror solver API."""

from __future__ import annotations

from dataclasses import dataclass

from ...core.boundary import MirrorBoundary
from ...core.config import MirrorConfig
from ...core.grids import MirrorGrid
from ...core.profiles import IPrimeProfile, PressureProfile, PsiPrimeProfile
from ...core.state import MirrorStateAxisym
from ...kernels.constraints import project_axisym_state
from .continuation import pressure_stage_profiles
from .diagnostics import FixedBoundaryTraceRow, trace_row_from_state
from .nonlinear import solve_axisym_fixed_boundary_stage
from .optimizers import OptimizerOptions


@dataclass(frozen=True)
class MirrorSolveOptions:
    """Options for the first fixed-boundary mirror solve path."""

    optimizer: str = "gradient_descent"
    maxiter: int = 50
    tolerance: float = 1.0e-8
    step_size: float = 1.0e-3
    min_step_size: float = 1.0e-12
    line_search_steps: int = 16
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
            line_search_steps=self.line_search_steps,
            mu0=self.mu0,
        )


@dataclass(frozen=True)
class MirrorFixedBoundaryResult:
    """Result from the axisymmetric fixed-boundary mirror solver."""

    config: MirrorConfig
    grid: MirrorGrid
    boundary: MirrorBoundary
    state: MirrorStateAxisym
    psi_prime: PsiPrimeProfile
    i_prime: IPrimeProfile
    pressure: PressureProfile
    options: MirrorSolveOptions
    trace: tuple[FixedBoundaryTraceRow, ...]

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
    initial_state: MirrorStateAxisym | None = None,
    options: MirrorSolveOptions | None = None,
) -> MirrorFixedBoundaryResult:
    """Run the first axisymmetric fixed-boundary mirror solve workflow."""
    options = options or MirrorSolveOptions()
    psi_prime = psi_prime or PsiPrimeProfile.constant(0.01)
    i_prime = i_prime or IPrimeProfile.zero()
    pressure = pressure or PressureProfile.zero()

    grid = config.build_grid()
    state = initial_state or MirrorStateAxisym.from_boundary(grid, boundary)
    state = project_axisym_state(state, grid, boundary)

    trace: list[FixedBoundaryTraceRow] = []
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
        )

    for stage_index, pressure_scale, stage_pressure in stages:
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
    )
