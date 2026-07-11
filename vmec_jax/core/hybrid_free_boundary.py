"""Free-boundary continuation for the 16-coil toroidal mirror hybrid.

The routines here deliberately return only equilibria that pass the requested
force-residual gate.  A failed pressure step is reported in the result rather
than being turned into a prescribed or interpolated boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Sequence

import numpy as np

from .coils import CoilSet, square_mirror_coils, to_mgrid_data
from .errors import VmecError
from .freeboundary import solve_free_boundary
from .hybrid import (
    CoilInformedAxis,
    coil_informed_toroidal_flux,
    stellarator_mirror_hybrid_input,
    trace_square_coil_vacuum_axis,
)
from .mgrid import MgridField
from .multigrid import solve_multigrid
from .preconditioner_2d import Prec2DConfig
from .solver import SolveResult, solve
from .wout import WoutData, wout_from_state

MU0 = 4.0e-7 * np.pi


@dataclass(frozen=True)
class HybridContinuationPoint:
    """One accepted coupled equilibrium in a pressure continuation."""

    target_beta: float
    achieved_beta: float
    predictor_iterations: int
    corrector_iterations: int
    free_iterations: int
    wall_seconds: float
    predictor_result: SolveResult | None
    corrector_result: SolveResult | None
    result: SolveResult
    wout: WoutData

    @property
    def maximum_residual(self) -> float:
        """Largest of the physical R, Z, and lambda force residuals."""

        return max(self.result.fsqr, self.result.fsqz, self.result.fsql)


@dataclass(frozen=True)
class HybridFreeBoundaryScan:
    """Accepted equilibria and construction data for one hybrid scan."""

    coils: CoilSet
    axis: CoilInformedAxis
    external_field: MgridField
    points: tuple[HybridContinuationPoint, ...]
    failed_target_beta: float | None = None
    failure: str | None = None
    failed_predictor: SolveResult | None = None
    failed_corrector: SolveResult | None = None


def _corrector_config(target_beta: float) -> Prec2DConfig | None:
    """Validated equation scaling along the present low-beta branch."""

    if target_beta <= 0.003:
        return None
    common = dict(
        threshold=1.0,
        start_iteration=10,
        step=0.25,
        gmres_restart=(
            160 if target_beta > 0.0071 else 120 if target_beta > 0.006978125 else 80
        ),
        gmres_max_restarts=3,
        gmres_rtol=1.0e-2,
        backtracking=True,
    )
    if target_beta <= 0.005:
        return Prec2DConfig(interval=10, row_scales=(1.0, 1.0, 0.01), **common)
    if target_beta <= 0.005953125:
        return Prec2DConfig(interval=10, row_scales=(1.0, 1.0, 0.005), **common)
    if target_beta <= 0.006715625:
        return Prec2DConfig(interval=10, row_scales=(1.0, 1.0, 0.004), **common)
    return Prec2DConfig(
        interval=20,
        auto_balance_lambda=True,
        lambda_balance_target=0.1,
        **common,
    )


def _as_point(
    target_beta: float,
    result: SolveResult,
    inp,
    *,
    predictor_result: SolveResult | None,
    corrector_result: SolveResult | None,
    predictor_iterations: int,
    corrector_iterations: int,
    wall_seconds: float,
) -> HybridContinuationPoint:
    wout = wout_from_state(
        inp=inp,
        state=result.state,
        fsqr=result.fsqr,
        fsqz=result.fsqz,
        fsql=result.fsql,
        niter=result.iterations,
        converged=result.converged,
    )
    return HybridContinuationPoint(
        target_beta=float(target_beta),
        achieved_beta=float(wout.betatotal),
        predictor_iterations=int(predictor_iterations),
        corrector_iterations=int(corrector_iterations),
        free_iterations=int(result.iterations),
        wall_seconds=float(wall_seconds),
        predictor_result=predictor_result,
        corrector_result=corrector_result,
        result=result,
        wout=wout,
    )


def solve_square_coil_free_boundary_scan(
    beta_targets: Sequence[float],
    *,
    n_coils_per_side: int = 4,
    side_length: float = 3.0,
    coil_radius: float = 0.5,
    coil_current: float = 8.0e5,
    toroidal_current: float = 3.0e3,
    minor_radius: float = 0.1,
    mpol: int = 6,
    ntor: int = 20,
    ns_array: Sequence[int] = (3, 5),
    ntheta: int = 48,
    nzeta: int = 256,
    ftol: float = 1.0e-8,
    predictor_ftol: float = 1.0e-7,
    max_iterations: int = 5000,
    corrector_iterations: int = 1000,
    coil_segments: int = 96,
    mgrid_shape: tuple[int, int, int] = (49, 41, 96),
    stop_on_failure: bool = True,
) -> HybridFreeBoundaryScan:
    """Solve the square 16-coil hybrid from vacuum through ``beta_targets``.

    Each finite-pressure point uses a fixed-LCFS predictor/corrector followed
    by a NESTOR free-boundary release.  The previous *solved* LCFS is retained
    throughout.  The current Fourier branch is validated through target beta
    0.775%; higher targets remain useful for exposing the solver barrier,
    but are never reported as equilibria unless they converge.
    """

    targets = tuple(float(value) for value in beta_targets)
    if any(value <= 0.0 for value in targets) or any(b <= a for a, b in zip(targets, targets[1:])):
        raise ValueError("beta_targets must be positive and strictly increasing")

    coils = square_mirror_coils(
        n_per_side=n_coils_per_side,
        side_length=side_length,
        semi_major=coil_radius,
        semi_minor=coil_radius,
        current=coil_current,
        n_segments=coil_segments,
        regularization_epsilon=5.0e-7,
        chunk_size=256,
    )
    axis = trace_square_coil_vacuum_axis(coils, side_length=side_length, n_steps=4096, nzeta=nzeta)
    phiedge = coil_informed_toroidal_flux(axis, minor_radius)
    fixed_input = stellarator_mirror_hybrid_input(
        mpol=mpol,
        ntor=ntor,
        ns_array=tuple(ns_array),
        ftol_array=tuple(ftol for _ in ns_array),
        niter_array=tuple(max_iterations for _ in ns_array),
        phiedge=phiedge,
        curtor=toroidal_current,
        ntheta=ntheta,
        nzeta=nzeta,
        axis_radius_samples=axis.radius,
        minor_radius_samples=axis.toroidal_flux_scale,
        minor_radius=minor_radius,
        side_elongation=0.0,
        corner_ellipticity=0.0,
        corner_rotation=0.0,
    )
    fixed_input = replace(fixed_input, delt=0.02)
    fixed = solve_multigrid(fixed_input)

    ir, jz, kp = mgrid_shape
    grid = to_mgrid_data(
        coils,
        rmin=0.8 * side_length / 2.0,
        rmax=side_length / 2.0 + 1.4 * coil_radius,
        zmin=-1.1 * coil_radius,
        zmax=1.1 * coil_radius,
        ir=ir,
        jz=jz,
        kp=kp,
        mgrid_mode="N",
        single_group=True,
    )
    field = MgridField.from_mgrid_data(grid, extcur=np.asarray([1.0]))
    am = np.zeros_like(fixed_input.am)
    am[:2] = (1.0, -1.0)
    free_input = replace(
        fixed_input,
        lfreeb=True,
        mgrid_file="in_memory_square_mirror_coils.nc",
        extcur=np.asarray([1.0]),
        am=am,
        pres_scale=0.0,
        nvacskip=1,
        delt=0.002,
        ns_array=np.asarray([ns_array[-1]]),
        ftol_array=np.asarray([ftol]),
        niter_array=np.asarray([max_iterations]),
    )
    started = perf_counter()
    released = solve_free_boundary(
        free_input,
        external_field=field,
        initial_state=fixed.state,
        ftol=ftol,
        max_iterations=max_iterations,
        max_vacuum_skip=1,
    )
    points = [
        _as_point(
            0.0,
            released,
            free_input,
            predictor_result=fixed,
            corrector_result=None,
            predictor_iterations=fixed.iterations,
            corrector_iterations=0,
            wall_seconds=perf_counter() - started,
        )
    ]
    state = released.state
    reference_field = float(np.exp(np.mean(np.log(axis.field_strength))))
    pressure_per_beta = reference_field**2 / MU0

    for target in targets:
        started = perf_counter()
        predictor = None
        corrected = None
        try:
            fixed_beta_input = replace(
                free_input,
                lfreeb=False,
                pres_scale=pressure_per_beta * target,
                delt=0.02,
            )
            predictor = solve(
                fixed_beta_input,
                initial_state=state,
                boundary_from_initial_state=True,
                ftol=predictor_ftol,
                max_iterations=max_iterations,
            )
            corrected = solve(
                fixed_beta_input,
                initial_state=predictor.state,
                boundary_from_initial_state=True,
                ftol=ftol,
                max_iterations=corrector_iterations,
                prec2d=_corrector_config(target),
                error_on_no_convergence=False,
            )
            if not corrected.converged:
                raise RuntimeError(
                    "fixed corrector did not converge: "
                    f"({corrected.fsqr:.3e}, {corrected.fsqz:.3e}, {corrected.fsql:.3e})"
                )
            target_input = replace(
                free_input,
                pres_scale=pressure_per_beta * target,
                delt=0.002,
            )
            released = solve_free_boundary(
                target_input,
                external_field=field,
                initial_state=corrected.state,
                ftol=ftol,
                max_iterations=max_iterations,
                max_vacuum_skip=1,
                error_on_no_convergence=False,
            )
            if not released.converged:
                raise RuntimeError(
                    f"free release did not converge: ({released.fsqr:.3e}, {released.fsqz:.3e}, {released.fsql:.3e})"
                )
        except (RuntimeError, VmecError) as error:
            scan = HybridFreeBoundaryScan(
                coils=coils,
                axis=axis,
                external_field=field,
                points=tuple(points),
                failed_target_beta=target,
                failure=str(error),
                failed_predictor=predictor,
                failed_corrector=corrected,
            )
            if stop_on_failure:
                return scan
            raise

        points.append(
            _as_point(
                target,
                released,
                target_input,
                predictor_result=predictor,
                corrector_result=corrected,
                predictor_iterations=predictor.iterations,
                corrector_iterations=corrected.iterations,
                wall_seconds=perf_counter() - started,
            )
        )
        state = released.state
        free_input = target_input

    return HybridFreeBoundaryScan(coils, axis, field, tuple(points))
