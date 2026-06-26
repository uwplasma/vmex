"""Free-boundary solve helpers, diagnostics, and derivative reports."""

from .derivatives import (
    DEFAULT_FREE_BOUNDARY_DERIVATIVE_OUTPUTS,
    FreeBoundaryDerivativeOptions,
    canonical_free_boundary_output_keys,
    coil_direction,
    contract_free_boundary_vjp,
    free_boundary_value_and_jacobian,
    free_boundary_value_and_jvp,
)
from .reduced_controls import (
    ReducedControlMap,
    ReducedControlState,
    ReducedControlStep,
    reduced_control_decode,
    reduced_control_least_squares_step,
    reduced_control_pullback,
)

__all__ = [
    "DEFAULT_FREE_BOUNDARY_DERIVATIVE_OUTPUTS",
    "FreeBoundaryDerivativeOptions",
    "ReducedControlMap",
    "ReducedControlState",
    "ReducedControlStep",
    "canonical_free_boundary_output_keys",
    "coil_direction",
    "contract_free_boundary_vjp",
    "free_boundary_value_and_jacobian",
    "free_boundary_value_and_jvp",
    "reduced_control_decode",
    "reduced_control_least_squares_step",
    "reduced_control_pullback",
]
