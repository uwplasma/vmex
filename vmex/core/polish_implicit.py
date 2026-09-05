"""Implicit derivatives of certified collocation least-squares polishing.

The nonlinear solve is deliberately not differentiated. Once its correction
is stationary, this module applies the implicit-function theorem to
``J(c, native).T @ r(c, native) = 0`` with matrix-free JVPs/VJPs. A gradient
therefore costs one Krylov solve rather than a replay of Gauss--Newton steps.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from solvax import gmres

from .errors import StrongForceCertificationError, StrongForceLinearSolveError
from .polish import (
    HighOrderCorrection,
    StrongPhysicalChart,
    StrongRootRuntime,
    strong_collocation_residual_at_native,
)
from .polish_driver import PolishContext
from .strong_force import HighOrderEquilibriumState


@dataclass(frozen=True)
class PolishLinearConfig:
    """Krylov and nonlinear-stationarity controls for polished derivatives.

    Stationarity requires ``||D*g/a**2|| <= max(stationarity_atol,
    stationarity_rtol * max(reference, 1))`` using the context's frozen primal
    scales. Eager failures raise typed errors by default; JIT failures return
    NaN derivatives and false report flags for either policy.
    """

    rtol: float = 1.0e-8
    atol: float = 1.0e-11
    restart: int = 30
    max_restarts: int = 30
    fail_policy: Literal["raise", "nan"] = "raise"
    stationarity_rtol: float = 1.0e-8
    stationarity_atol: float = 1.0e-11

    def __post_init__(self) -> None:
        if not np.isfinite(self.stationarity_rtol) or self.stationarity_rtol <= 0.0:
            raise ValueError("stationarity_rtol must be finite and positive")
        if not np.isfinite(self.stationarity_atol) or self.stationarity_atol < 0.0:
            raise ValueError("stationarity_atol must be finite and non-negative")
        if not np.isfinite(self.rtol) or self.rtol <= 0.0:
            raise ValueError("rtol must be finite and positive")
        if not np.isfinite(self.atol) or self.atol < 0.0:
            raise ValueError("atol must be finite and non-negative")
        if self.restart < 1:
            raise ValueError("restart must be positive")
        if self.max_restarts < 1:
            raise ValueError("max_restarts must be positive")
        if self.fail_policy not in ("raise", "nan"):
            raise ValueError("fail_policy must be 'raise' or 'nan'")


class PolishLinearReport(NamedTuple):
    """Stationarity and finite true-linear-residual certificates.

    Public tangent/adjoint ``converged`` requires both ``stationarity_converged``
    and ``linear_converged``. The latter certifies a finite unpreconditioned
    residual within ``tolerance``. A skipped Krylov solve has zero iterations,
    NaN linear norms and ``linear_converged=False``. Stationarity uses the
    primal scaling and its own reported norm and tolerance.
    """

    residual_norm: jax.Array
    tolerance: jax.Array
    iterations: jax.Array
    converged: jax.Array
    linear_converged: jax.Array | bool = False
    stationarity_norm: jax.Array | float = float("nan")
    stationarity_tolerance: jax.Array | float = float("nan")
    stationarity_converged: jax.Array | bool = False


class PolishTangentResult(NamedTuple):
    """Total polished-state tangent and its reduced correction response."""

    native_tangent: HighOrderEquilibriumState
    correction_tangent: jax.Array
    report: PolishLinearReport


class PolishAdjointResult(NamedTuple):
    """Native-state cotangent and the strong-equation adjoint variable."""

    native_cotangent: HighOrderEquilibriumState
    equation_adjoint: jax.Array
    report: PolishLinearReport


def _tree_norm(value) -> jax.Array:
    return jnp.sqrt(
        sum(
            (jnp.vdot(leaf, leaf).real for leaf in jax.tree.leaves(value)),
            jnp.asarray(0.0),
        )
    )


def _add_correction(
    native: HighOrderEquilibriumState,
    correction: HighOrderCorrection,
) -> HighOrderEquilibriumState:
    """Add correction data while preserving the native PyTree metadata."""

    return replace(
        native,
        R_cos=native.R_cos + correction.R_cos,
        R_sin=native.R_sin + correction.R_sin,
        Z_cos=native.Z_cos + correction.Z_cos,
        Z_sin=native.Z_sin + correction.Z_sin,
        L_cos=native.L_cos + correction.L_cos,
        L_sin=native.L_sin + correction.L_sin,
    )


def _linear_report(operator, rhs, solution, config: PolishLinearConfig):
    applied = operator(solution.x)
    residual_norm = jnp.linalg.norm(rhs - applied)
    rhs_norm = jnp.linalg.norm(rhs)
    tolerance = jnp.maximum(config.atol, config.rtol * rhs_norm)
    # Krylov flags can describe a preconditioned estimate. Only the finite,
    # recomputed unpreconditioned residual certifies a derivative here.
    finite = (
        jnp.all(jnp.isfinite(solution.x))
        & jnp.all(jnp.isfinite(rhs))
        & jnp.all(jnp.isfinite(applied))
        & jnp.isfinite(rhs_norm)
        & jnp.isfinite(residual_norm)
        & jnp.isfinite(tolerance)
    )
    converged = finite & (residual_norm <= tolerance)
    return PolishLinearReport(
        residual_norm=residual_norm,
        tolerance=tolerance,
        iterations=solution.iterations,
        converged=converged,
        linear_converged=converged,
    )


def _checked_solution(
    value: jax.Array,
    report: PolishLinearReport,
    config: PolishLinearConfig,
    solve_kind: str,
) -> jax.Array:
    traced = any(
        isinstance(item, jax.core.Tracer)
        for item in (value, report.residual_norm, report.tolerance, report.converged)
    )
    if not traced and not bool(np.asarray(report.converged)):
        if config.fail_policy == "raise":
            raise StrongForceLinearSolveError(
                message=(
                    f"strong-root {solve_kind} solve did not converge: finite "
                    f"true-residual certificate failed (residual "
                    f"{float(report.residual_norm):.3e}, tolerance "
                    f"{float(report.tolerance):.3e}) after "
                    f"{int(report.iterations)} Krylov iterations"
                ),
                hint=(
                    "increase max_restarts/restart, loosen the derivative "
                    "tolerance, or refresh the polish preconditioner"
                ),
                solve_kind=solve_kind,
                iterations=int(report.iterations),
                residual_norm=float(report.residual_norm),
                tolerance=float(report.tolerance),
            )
        return jnp.full_like(value, jnp.nan)
    return jnp.where(report.converged, value, jnp.full_like(value, jnp.nan))


def _solve_linear(operator, rhs, preconditioner, config, solve_kind):
    size = int(rhs.shape[0])
    solution = gmres(
        operator,
        rhs,
        precond=preconditioner,
        restart=min(config.restart, size),
        rtol=config.rtol,
        atol=config.atol,
        max_restarts=config.max_restarts,
    )
    report = _linear_report(operator, rhs, solution, config)
    return _checked_solution(solution.x, report, config, solve_kind), report


def _stationarity_certificate(gradient, context, config):
    """Match the primal coordinates c=D*y and residual r/a without another JVP."""
    scale = jnp.asarray(context.residual_scale)
    reference = jnp.asarray(context.stationarity_reference)
    diagonal = jnp.asarray(context.variable_scale)
    if scale.ndim or reference.ndim:
        raise ValueError("polish residual scale and stationarity reference must be scalars")
    if diagonal.shape != gradient.shape or context.correction.shape != gradient.shape:
        raise ValueError("polish stationarity and coordinate scales must match correction shape")
    norm = jnp.linalg.norm(diagonal * gradient / scale / scale)
    tolerance = jnp.maximum(
        config.stationarity_atol,
        config.stationarity_rtol * jnp.maximum(reference, 1.0),
    )
    finite = (
        jnp.all(jnp.isfinite(gradient))
        & jnp.all(jnp.isfinite(context.correction))
        & jnp.all(jnp.isfinite(diagonal) & (diagonal > 0.0))
        & jnp.isfinite(scale) & (scale > 0.0)
        & jnp.isfinite(reference) & (reference >= 0.0)
        & jnp.isfinite(norm) & jnp.isfinite(tolerance)
    )
    return norm, tolerance, finite & (norm <= tolerance)


def _solve_stationary_linear(operator, rhs, preconditioner, config, solve_kind, certificate):
    norm, tolerance, stationary = certificate
    if not isinstance(stationary, jax.core.Tracer) and not bool(stationary):
        if config.fail_policy == "raise":
            raise StrongForceCertificationError(
                f"{solve_kind} requires a stationary polish correction: "
                f"scaled gradient {float(norm):.3e}, tolerance {float(tolerance):.3e}",
                hint="refine the nonlinear polish for the current native inputs",
                stationarity_norm=float(norm),
                stationarity_tolerance=float(tolerance),
            )
    value, linear = jax.lax.cond(
        stationary,
        lambda _: _solve_linear(operator, rhs, preconditioner, config, solve_kind),
        lambda _: (
            jnp.full_like(rhs, jnp.nan),
            PolishLinearReport(jnp.asarray(jnp.nan, rhs.real.dtype),
                               jnp.asarray(jnp.nan, rhs.real.dtype),
                               jnp.asarray(0, jnp.int32), jnp.asarray(False)),
        ),
        operand=None,
    )
    report = linear._replace(
        converged=stationary & linear.converged,
        stationarity_norm=norm, stationarity_tolerance=tolerance,
        stationarity_converged=stationary,
    )
    return _checked_solution(value, report, config, solve_kind), report


def _certified_tangent(value, report):
    return jax.tree.map(
        lambda leaf: jnp.where(report.converged, leaf, jnp.full_like(leaf, jnp.nan)),
        value,
    )


def _collocation_corrected_state(
    native: HighOrderEquilibriumState,
    correction: jax.Array,
    runtime: StrongRootRuntime,
    chart: StrongPhysicalChart,
) -> HighOrderEquilibriumState:
    full = chart.lift(correction)
    high = runtime.layout.unpack(jnp.asarray(runtime.coordinate_scale) * full)
    return _add_correction(native, high)


def _collocation_stationarity(
    correction: jax.Array,
    native: HighOrderEquilibriumState,
    runtime: StrongRootRuntime,
    chart: StrongPhysicalChart,
) -> jax.Array:
    """Return the exact gradient of one-half the collocation residual norm."""

    from solvax import least_squares_stationarity

    return least_squares_stationarity(
        lambda value: strong_collocation_residual_at_native(
            value, native, runtime, chart
        ),
        correction,
    )


def collocation_polish_tangent(
    context: PolishContext,
    native_tangent: HighOrderEquilibriumState,
    *,
    config: PolishLinearConfig = PolishLinearConfig(),
) -> PolishTangentResult:
    """Apply the IFT tangent of the rectangular polish stationarity equation."""

    runtime, chart = context.runtime, context.chart
    correction = jnp.asarray(context.correction)
    if correction.shape != (chart.size,):
        raise ValueError(
            f"correction has shape {correction.shape}; expected {(chart.size,)}"
        )
    if jax.tree.structure(native_tangent) != jax.tree.structure(runtime.native):
        raise ValueError("native_tangent must have the runtime native-state structure")
    stationarity = lambda value: _collocation_stationarity(  # noqa: E731
        value, runtime.native, runtime, chart
    )
    gradient, operator = jax.linearize(stationarity, correction)
    _, parameter_direction = jax.jvp(
        lambda native: _collocation_stationarity(
            correction, native, runtime, chart
        ),
        (runtime.native,),
        (native_tangent,),
    )
    diagonal_inverse = jnp.asarray(context.variable_scale) ** 2
    response, report = _solve_stationary_linear(
        operator,
        -parameter_direction,
        lambda rhs: diagonal_inverse * rhs,
        config,
        "least-squares tangent",
        _stationarity_certificate(gradient, context, config),
    )
    _, correction_tangent = jax.jvp(
        lambda value: _collocation_corrected_state(
            runtime.native, value, runtime, chart
        ),
        (correction,),
        (response,),
    )
    return PolishTangentResult(
        native_tangent=_certified_tangent(
            jax.tree.map(jnp.add, native_tangent, correction_tangent), report),
        correction_tangent=response,
        report=report,
    )


def collocation_polish_adjoint(
    context: PolishContext,
    polished_cotangent: HighOrderEquilibriumState,
    *,
    config: PolishLinearConfig = PolishLinearConfig(),
) -> PolishAdjointResult:
    """Apply the IFT pullback of the rectangular polish stationarity equation."""

    runtime, chart = context.runtime, context.chart
    correction = jnp.asarray(context.correction)
    if correction.shape != (chart.size,):
        raise ValueError(
            f"correction has shape {correction.shape}; expected {(chart.size,)}"
        )
    if jax.tree.structure(polished_cotangent) != jax.tree.structure(runtime.native):
        raise ValueError(
            "polished_cotangent must have the runtime native-state structure"
        )
    stationarity = lambda value: _collocation_stationarity(  # noqa: E731
        value, runtime.native, runtime, chart
    )
    gradient, stationarity_pullback = jax.vjp(stationarity, correction)
    transpose_operator = lambda value: stationarity_pullback(value)[0]  # noqa: E731
    _, correction_pullback = jax.vjp(
        lambda value: _collocation_corrected_state(
            runtime.native, value, runtime, chart
        ),
        correction,
    )
    correction_cotangent = correction_pullback(polished_cotangent)[0]
    diagonal_inverse = jnp.asarray(context.variable_scale) ** 2
    equation_adjoint, report = _solve_stationary_linear(
        transpose_operator,
        correction_cotangent,
        lambda rhs: diagonal_inverse * rhs,
        config,
        "least-squares adjoint",
        _stationarity_certificate(gradient, context, config),
    )
    _, stationarity_native_pullback = jax.vjp(
        lambda native: _collocation_stationarity(
            correction, native, runtime, chart
        ),
        runtime.native,
    )
    force_cotangent = stationarity_native_pullback(equation_adjoint)[0]
    _, direct_native_pullback = jax.vjp(
        lambda native: _collocation_corrected_state(
            native, correction, runtime, chart
        ),
        runtime.native,
    )
    direct_cotangent = direct_native_pullback(polished_cotangent)[0]
    native_cotangent = jax.tree.map(
        jnp.subtract, direct_cotangent, force_cotangent
    )
    return PolishAdjointResult(
        _certified_tangent(native_cotangent, report), equation_adjoint, report)


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5))
def _implicit_collocation_leaves(
    native_leaves: tuple[jax.Array, ...],
    correction: jax.Array,
    scales: tuple[jax.Array, ...],
    runtime: StrongRootRuntime,
    chart: StrongPhysicalChart,
    config: PolishLinearConfig,
) -> tuple[jax.Array, ...]:
    del scales, config
    native = jax.tree.unflatten(jax.tree.structure(runtime.native), native_leaves)
    correction = jax.lax.stop_gradient(jnp.asarray(correction))
    return tuple(
        jax.tree.leaves(
            _collocation_corrected_state(native, correction, runtime, chart)
        )
    )


def _implicit_collocation_leaves_fwd(
    native_leaves, correction, scales, runtime, chart, config
):
    output = _implicit_collocation_leaves(
        native_leaves,
        correction,
        scales,
        runtime,
        chart,
        config,
    )
    return output, (native_leaves, correction, scales)


def _implicit_collocation_leaves_bwd(
    runtime,
    chart,
    config,
    saved,
    output_cotangent_leaves,
):
    native_leaves, correction, scales = saved
    # The frozen discretization may be reused, but the adjoint must linearize
    # at the native inputs of this forward call, not the runtime's old state.
    native = jax.tree.unflatten(jax.tree.structure(runtime.native), native_leaves)
    runtime = replace(runtime, native=native)
    output_cotangent = jax.tree.unflatten(
        jax.tree.structure(runtime.native), output_cotangent_leaves
    )
    result = collocation_polish_adjoint(
        PolishContext(runtime, chart, correction, *scales),
        output_cotangent,
        config=config,
    )
    return (
        tuple(jax.tree.leaves(result.native_cotangent)),
        jnp.zeros_like(correction),
        jax.tree.map(jnp.zeros_like, scales),
    )


_implicit_collocation_leaves.defvjp(
    _implicit_collocation_leaves_fwd,
    _implicit_collocation_leaves_bwd,
)


def implicit_collocation_polished_state(
    native: HighOrderEquilibriumState,
    context: PolishContext,
    config: PolishLinearConfig = PolishLinearConfig(),
) -> HighOrderEquilibriumState:
    """Attach a stationarity-checked VJP to a supplied polish correction.

    The primal correction comes from :func:`polish_collocation_least_squares`.
    Reverse mode solves the exact transposed least-squares stationarity
    equation once; it never differentiates through the nonlinear iterations.
    The forward evaluates the supplied correction; the pullback checks its
    stationarity for the current native inputs. Use
    :func:`collocation_polish_tangent` for forward sensitivities.
    """

    if jax.tree.structure(native) != jax.tree.structure(context.runtime.native):
        raise ValueError("native must have the polish context native-state structure")
    leaves = tuple(jax.tree.leaves(native))
    polished_leaves = _implicit_collocation_leaves(
        leaves,
        context.correction,
        (context.variable_scale, jnp.asarray(context.residual_scale),
         jnp.asarray(context.stationarity_reference)),
        context.runtime,
        context.chart,
        config,
    )
    return jax.tree.unflatten(jax.tree.structure(native), polished_leaves)


__all__ = [
    "PolishAdjointResult",
    "PolishLinearConfig",
    "PolishLinearReport",
    "PolishTangentResult",
    "collocation_polish_adjoint",
    "collocation_polish_tangent",
    "implicit_collocation_polished_state",
]
