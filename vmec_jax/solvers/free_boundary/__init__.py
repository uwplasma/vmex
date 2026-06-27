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
from .control import (
    FreeBoundaryNativeSplineState,
    FreeBoundaryNativeSplineUpdate,
    FreeBoundaryReducedEdgeState,
    free_boundary_reduced_edge_state_from_vmec_state,
    free_boundary_reduced_edge_state_to_vmec_state,
)

__all__ = [
    "DEFAULT_FREE_BOUNDARY_DERIVATIVE_OUTPUTS",
    "FreeBoundaryDerivativeOptions",
    "FreeBoundaryNativeSplineState",
    "FreeBoundaryNativeSplineUpdate",
    "FreeBoundaryReducedEdgeState",
    "ReducedControlMap",
    "ReducedControlState",
    "ReducedControlStep",
    "canonical_free_boundary_output_keys",
    "coil_direction",
    "contract_free_boundary_vjp",
    "free_boundary_value_and_jacobian",
    "free_boundary_value_and_jvp",
    "free_boundary_reduced_edge_state_from_vmec_state",
    "free_boundary_reduced_edge_state_to_vmec_state",
    "reduced_control_decode",
    "reduced_control_least_squares_step",
    "reduced_control_pullback",
]
