"""Objective term containers and residual callbacks for fixed-boundary workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from vmec_jax._compat import jnp
from vmec_jax.optimizers.fixed_boundary.parameterization import BoundaryParamSpec


@dataclass(frozen=True)
class StageContext:
    """Objects needed by objective callbacks for one mode-continuation stage."""

    static: object
    indata: object
    boundary_input: object
    specs: Sequence[BoundaryParamSpec]
    signgs: int
    flux: object
    pressure: object


@dataclass(frozen=True)
class ObjectiveTerm:
    """One weighted least-squares objective block."""

    name: str
    evaluate: Callable[[StageContext, object], object]
    target: float | np.ndarray = 0.0
    weight: float = 1.0
    total: Callable[[StageContext, object], object] | None = None
    track_iota: bool = False
    metadata: dict[str, object] = field(default_factory=dict)
    prepare: Callable[[StageContext], "ObjectiveTerm"] | None = None

    def residual(self, ctx: StageContext, state) -> object:
        """Evaluate residual for fixed-boundary VMEC solve and implicit differentiation."""
        value = as_vector(self.evaluate(ctx, state))
        target = jnp.asarray(self.target, dtype=jnp.float64)
        if int(target.ndim) == 0:
            target = jnp.full_like(value, target)
        else:
            target = jnp.ravel(target)
        return float(self.weight) * (value - target)

    def bind(self, ctx: StageContext) -> "ObjectiveTerm":
        """Return a stage-specialized term when the objective has static setup."""

        return self if self.prepare is None else self.prepare(ctx)


@dataclass(frozen=True)
class FixedBoundaryObjectiveStage:
    """Prepared optimizer and metadata for one active boundary-mode stage."""

    mode: int
    ctx: StageContext
    optimizer: object
    specs: Sequence[BoundaryParamSpec]
    boundary_input: object


@dataclass(frozen=True)
class QIObjectiveTerm:
    """One field-quality objective that shares a Boozer/QI field evaluation."""

    name: str
    evaluate: Callable[[StageContext, object, dict], tuple[object, object]]
    qi_options: "QuasiIsodynamicOptions | None" = None

    def residual_and_total(self, ctx: StageContext, state, field: dict) -> tuple[object, object]:
        """Evaluate residual and total for fixed-boundary VMEC solve and implicit differentiation."""
        residuals, total = self.evaluate(ctx, state, field)
        return as_vector(residuals), total


def as_vector(value):
    """Return a float64 one-dimensional JAX array for scalar/vector objectives."""

    arr = jnp.asarray(value, dtype=jnp.float64)
    return arr.reshape((1,)) if int(arr.ndim) == 0 else jnp.ravel(arr)


def residuals_from_objectives(objectives: Sequence[ObjectiveTerm], ctx: StageContext):
    """Create the state residual callback consumed by ``FixedBoundaryExactOptimizer``."""

    bound_objectives = tuple(term.bind(ctx) for term in objectives)

    def residuals_from_state(state, *, ctx=ctx, objectives=bound_objectives):
        """Evaluate residuals from state for fixed-boundary VMEC solve and implicit differentiation."""
        return jnp.concatenate([term.residual(ctx, state) for term in objectives])

    field_totals = tuple(term.total for term in bound_objectives if term.total is not None)
    residuals_from_state._n_non_qs = sum(1 for term in bound_objectives if term.total is None)
    residuals_from_state._qs_total_from_state = (
        lambda state, ctx=ctx, field_totals=field_totals: float(
            sum(float(total(ctx, state)) for total in field_totals)
        )
        if field_totals
        else lambda _state: 0.0
    )
    family = next(
        (
            term.metadata.get("objective_family")
            for term in bound_objectives
            if term.metadata.get("objective_family")
        ),
        None,
    )
    if family is not None:
        residuals_from_state._objective_family = str(family)
    helicity_m = next(
        (term.metadata.get("helicity_m") for term in bound_objectives if "helicity_m" in term.metadata),
        None,
    )
    helicity_n = next(
        (term.metadata.get("helicity_n") for term in bound_objectives if "helicity_n" in term.metadata),
        None,
    )
    if helicity_m is not None:
        residuals_from_state._helicity_m = int(helicity_m)
    if helicity_n is not None:
        residuals_from_state._helicity_n = int(helicity_n)
    return attach_packed_state_autodiff_hooks(residuals_from_state)


def attach_packed_state_autodiff_hooks(residuals_from_state: Callable) -> Callable:
    """Attach generic packed-state VJP hooks to an objective residual callback."""

    def _residuals_from_packed(packed_state, layout):
        from vmec_jax.state import unpack_state

        state = unpack_state(packed_state, layout)
        return jnp.asarray(residuals_from_state(state), dtype=jnp.float64).reshape(-1)

    def state_cotangent_operator_from_packed(packed_state, layout):
        """Evaluate state cotangent operator from packed for fixed-boundary VMEC solve and implicit differentiation."""
        from vmec_jax._compat import jax, jnp as _jnp

        packed_state = _jnp.asarray(packed_state, dtype=_jnp.float64)

        def _packed_residuals(packed):
            return _residuals_from_packed(packed, layout)

        _, residual_vjp = jax.vjp(_packed_residuals, packed_state)

        def _apply(residual_cotangent):
            cotangent = _jnp.asarray(residual_cotangent, dtype=_jnp.float64).reshape(-1)
            state_cotangent = residual_vjp(cotangent)[0]
            return _jnp.nan_to_num(state_cotangent, nan=0.0, posinf=0.0, neginf=0.0)

        return _apply

    def state_cotangent_from_packed(packed_state, layout, residual_cotangent):
        """Evaluate state cotangent from packed for fixed-boundary VMEC solve and implicit differentiation."""
        return state_cotangent_operator_from_packed(packed_state, layout)(residual_cotangent)

    def state_objective_value_and_cotangent_from_packed(packed_state, layout):
        """Evaluate state objective value and cotangent from packed for fixed-boundary VMEC solve and implicit differentiation."""
        from vmec_jax._compat import jax, jnp as _jnp

        packed_state = _jnp.asarray(packed_state, dtype=_jnp.float64)

        def _objective(packed):
            residuals = _residuals_from_packed(packed, layout)
            return 0.5 * _jnp.vdot(residuals, residuals)

        value, cotangent = jax.value_and_grad(_objective)(packed_state)
        cotangent = _jnp.nan_to_num(cotangent, nan=0.0, posinf=0.0, neginf=0.0)
        return value, cotangent

    residuals_from_state._state_cotangent_from_packed = state_cotangent_from_packed
    residuals_from_state._state_cotangent_operator_from_packed = state_cotangent_operator_from_packed
    residuals_from_state._state_objective_value_and_cotangent_from_packed = (
        state_objective_value_and_cotangent_from_packed
    )
    return residuals_from_state


__all__ = [
    "FixedBoundaryObjectiveStage",
    "ObjectiveTerm",
    "QIObjectiveTerm",
    "StageContext",
    "residuals_from_objectives",
]
