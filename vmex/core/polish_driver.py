"""Force-balance polishing: entry point, report, and WOUT export.

``solve(..., polish=...)``, ``solve_file`` and ``vmex --polish`` call
:func:`polish_legacy_solution` after the ordinary solve.  It lifts the
converged state into the native quintic-spline representation, runs the
constrained force least squares of :mod:`vmex.core.polish_native`, and
certifies the result with the independent strong-force oracle of
:mod:`vmex.core.strong_force`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any, NamedTuple

import numpy as np

from .errors import StrongForceCertificationError, VmecInputError
from .polish_native import PolishConfig, native_polish_supported, physical_scales, polish_native
from .printing import emit_flushed, polish_certificate_summary
from .strong_force import (
    HighOrderEquilibriumState,
    StrongForceReport,
    certify_strong_force,
    force_error_measures,
    lift_high_order_state,
)


@dataclass(frozen=True)
class PolishReport:
    """Summary of one polish, as plain host values.

    ``force_norm`` is the independent volume-RMS force over the wout scale
    ``volavgB**2 / (mu0 Aminor_p)``; ``stationarity`` the Frobenius-scaled
    projected gradient ``eta``.  ``initial_*``/``final_*`` are the certificate
    of the lifted VMEC state and of the polished state: ``*_normalized_l2`` is
    the pointwise ``eps_F`` (bounded above by 2 by construction), the other
    two the non-saturating averages over ``normalization_window``.
    """

    converged: bool
    termination_reason: str
    force_norm: float
    stationarity: float
    nonlinear_iterations: int
    charts: int
    minimum_signed_jacobian: float
    solve_seconds: float
    initial_normalized_l2: float
    final_normalized_l2: float
    initial_volume_average_force: float
    final_volume_average_force: float
    initial_magnetic_relative_force_error: float
    final_magnetic_relative_force_error: float
    normalization_window: tuple[float, float]


class PolishResult(NamedTuple):
    """Certified native state, its certificate, the report, and a solve-mesh view."""

    native_equilibrium: HighOrderEquilibriumState
    strong_force: StrongForceReport
    polish_report: PolishReport
    compatibility_state: Any


def polish_legacy_solution(
    source,
    resolution,
    legacy_state,
    *,
    config: PolishConfig | None = None,
    lconm1: bool = True,
    auto: bool = False,
    verbose: bool = False,
    emit: Any = emit_flushed,
) -> PolishResult | None:
    """Polish one converged fixed-boundary solve (see the module docstring).

    Supported decks are axisymmetric, fixed-boundary, with prescribed pressure
    and iota (``NCURR = 0``, ``GAMMA = 0``) and no ``LASYM``.  For any other
    deck ``auto=True`` returns ``None`` (the solve stays unpolished) and an
    explicit request raises :class:`~vmex.core.errors.VmecInputError`.  A
    polish that misses its tolerances raises
    :class:`~vmex.core.errors.StrongForceCertificationError`, or with
    ``config.fail_policy="return_unpolished"`` returns the lifted state with
    ``polish_report.converged`` false.
    """

    from . import implicit

    config = PolishConfig() if config is None else config
    if not native_polish_supported(source):
        if auto:
            return None
        raise VmecInputError(
            "force-balance polishing supports fixed-boundary axisymmetric decks with "
            "NCURR = 0, GAMMA = 0 and LASYM = F",
            hint="drop the polish request for this deck",
        )
    started = perf_counter()
    implicit_config = implicit.make_config(source, ns=int(resolution.ns), lconm1=bool(lconm1), multigrid=False)
    runtime = implicit.runtime_from_params(implicit.params_from_input(source), implicit_config)
    if verbose:
        emit(" native polish: lifting to the quintic spline basis...")
    lifted = lift_high_order_state(legacy_state, runtime, inp=source, degree=config.degree)
    initial = certify_strong_force(lifted)
    force_scale, volume_scale = physical_scales(lifted)
    result = polish_native(lifted, force_scale=force_scale, volume_scale=volume_scale,
                           config=config, emit=emit if verbose else None)
    final = certify_strong_force(result.state)
    force_norm = float(final.absolute_l2) / force_scale
    converged = bool(result.stationarity <= config.stationarity_tolerance
                     and force_norm <= config.force_tolerance
                     and float(final.minimum_signed_jacobian) > 0.0)
    window = initial.window_normalizations
    report = PolishReport(
        converged=converged,
        termination_reason="certified" if converged else "not-converged",
        force_norm=force_norm,
        stationarity=result.stationarity,
        nonlinear_iterations=result.gauss_newton_steps + result.newton_steps,
        charts=result.charts,
        minimum_signed_jacobian=float(final.minimum_signed_jacobian),
        solve_seconds=perf_counter() - started,
        initial_normalized_l2=float(initial.normalized_l2),
        final_normalized_l2=float(final.normalized_l2),
        initial_volume_average_force=float(window.volume_average_force),
        final_volume_average_force=float(final.window_normalizations.volume_average_force),
        initial_magnetic_relative_force_error=float(window.magnetic_relative_force_error),
        final_magnetic_relative_force_error=float(final.window_normalizations.magnetic_relative_force_error),
        normalization_window=(float(window.s_min), float(window.s_max)),
    )
    if verbose:
        emit(f" native polish: |F|_rms / F* = {force_norm:.3E} (tolerance {config.force_tolerance:.1E}),"
             f" eta = {result.stationarity:.3E} (tolerance {config.stationarity_tolerance:.1E}),"
             f" {report.solve_seconds:.1f} s")
        emit(polish_certificate_summary(
            report.initial_normalized_l2, report.final_normalized_l2, config.force_tolerance,
            verdict="CERTIFIED" if converged else "FAILED",
            measures=force_error_measures(initial, final), window=report.normalization_window), end="")
    if not converged and config.fail_policy == "raise":
        raise StrongForceCertificationError(
            f"force-balance polish did not converge: |F|/F* = {force_norm:.3e}, "
            f"eta = {result.stationarity:.3e}",
            hint="inspect the polish report or use fail_policy='return_unpolished'",
            force_norm=force_norm,
            force_tolerance=config.force_tolerance,
            stationarity=result.stationarity,
            stationarity_tolerance=config.stationarity_tolerance,
        )
    native, certificate = (result.state, final) if converged else (lifted, initial)
    return PolishResult(native, certificate, report, _solve_mesh_state(native, source, runtime))


def _solve_mesh_state(native: HighOrderEquilibriumState, source, runtime):
    """The native state sampled on the solve mesh and the deck's modes.

    Padded poloidal modes the deck's ``MPOL`` cannot hold are dropped; this is
    the solve-grid view, not the certified object (see
    :func:`polished_wout_state` for the faithful export).
    """

    from .polish import sample_high_order_state

    keep = np.abs(np.asarray(native.m)) < int(source.mpol)
    fields = ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")
    truncated = replace(native, m=np.asarray(native.m)[keep], n=np.asarray(native.n)[keep],
                        **{name: getattr(native, name)[keep] for name in fields})
    return sample_high_order_state(truncated, runtime)


#: Minimum radial surfaces for a polished WOUT export: four samples per span
#: of the default 32-span wout reconstruction.
_POLISHED_WOUT_MIN_NS = 129


def polished_wout_ns(native: HighOrderEquilibriumState, *, solve_ns: int) -> int:
    """Radial export mesh on which the samples determine ``native``.

    ``max(solve_ns, 129, 2 * size + 1, 2 / ds_min + 1)``: never coarser than
    the solve, four samples per default reconstruction span, two per spline
    coefficient, and two uniform-``s`` samples in the narrowest span.
    """

    determined = 2 * int(native.radial_basis.size) + 1
    narrowest = float(np.min(np.diff(np.asarray(native.radial_basis.breakpoints))))
    resolving = int(np.ceil(2.0 / narrowest)) + 1
    return max(int(solve_ns), _POLISHED_WOUT_MIN_NS, determined, resolving)


def polished_wout_input(native: HighOrderEquilibriumState, source):
    """The deck whose ``MPOL`` holds every mode of ``native`` (a wider wout)."""

    mpol = int(np.max(np.abs(np.asarray(native.m)))) + 1
    if mpol <= int(source.mpol):
        return source
    return source.change_resolution(mpol=mpol, ntor=int(source.ntor))


def polished_wout_state(native: HighOrderEquilibriumState, source, *, solve_ns: int):
    """Sample ``native`` on the :func:`polished_wout_ns` mesh for the wout writer."""

    from .polish import sample_high_order_state
    from .solver import prepare_runtime, resolution_from_input

    source = polished_wout_input(native, source)
    ns = polished_wout_ns(native, solve_ns=solve_ns)
    return sample_high_order_state(native, prepare_runtime(source, resolution_from_input(source, ns=ns)))


__all__ = [
    "PolishConfig",
    "PolishReport",
    "PolishResult",
    "polish_legacy_solution",
    "polished_wout_input",
    "polished_wout_ns",
    "polished_wout_state",
]
