"""Compatibility facade for free-boundary adjoint controller primitives."""

from __future__ import annotations

from vmec_jax.solvers.free_boundary.adjoint.controller import (
    _pytree_vdot_jax,
    jax_visible_accepted_nonlinear_controller_directional_check_jax,
    jax_visible_accepted_nonlinear_controller_jax,
    jax_visible_accepted_only_nonlinear_controller_jax,
    jax_visible_masked_nonlinear_controller_directional_check_jax,
    jax_visible_masked_nonlinear_controller_jax,
    jax_visible_nonlinear_controller_directional_check_jax,
    jax_visible_nonlinear_controller_jax,
    jax_visible_segmented_accepted_nonlinear_controller_jax,
    jax_visible_segmented_state_only_accepted_nonlinear_controller_jax,
    jax_visible_state_only_accepted_nonlinear_controller_jax,
    jax_visible_state_only_accepted_only_nonlinear_controller_jax,
    jax_visible_unrolled_accepted_only_nonlinear_controller_jax,
    jax_visible_unrolled_state_only_accepted_only_nonlinear_controller_jax,
    pytree_directional_derivative_check_jax,
)

__all__ = [
    "_pytree_vdot_jax",
    "jax_visible_accepted_nonlinear_controller_directional_check_jax",
    "jax_visible_accepted_nonlinear_controller_jax",
    "jax_visible_accepted_only_nonlinear_controller_jax",
    "jax_visible_masked_nonlinear_controller_directional_check_jax",
    "jax_visible_masked_nonlinear_controller_jax",
    "jax_visible_nonlinear_controller_directional_check_jax",
    "jax_visible_nonlinear_controller_jax",
    "jax_visible_segmented_accepted_nonlinear_controller_jax",
    "jax_visible_segmented_state_only_accepted_nonlinear_controller_jax",
    "jax_visible_state_only_accepted_nonlinear_controller_jax",
    "jax_visible_state_only_accepted_only_nonlinear_controller_jax",
    "jax_visible_unrolled_accepted_only_nonlinear_controller_jax",
    "jax_visible_unrolled_state_only_accepted_only_nonlinear_controller_jax",
    "pytree_directional_derivative_check_jax",
]
