"""Initial state/update constants for residual-iteration solves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from vmec_jax._compat import has_jax, jnp
from vmec_jax.solvers.fixed_boundary.optimization.constraints import (
    enforce_fixed_boundary_and_axis as _enforce_fixed_boundary_and_axis,
    enforce_fixed_boundary_and_axis_np as _enforce_fixed_boundary_and_axis_np,
)


@dataclass(frozen=True)
class ResidualStateSetup:
    """State and cached constants used by the host/JAX residual update paths."""

    state: Any
    precomputed_axis_mask_np: Any
    jnp_state_dtype: Any
    jnp_zero_m1_0: Any
    jnp_zero_m1_1: Any
    jnp_true_bool: Any
    jnp_false_bool: Any
    zeros_coeff_np: Any
    zeros_dR_np: Any
    delta_s: Any


def build_residual_state_setup(
    *,
    state0: Any,
    static: Any,
    s: Any,
    edge_Rcos: Any,
    edge_Rsin: Any,
    edge_Zcos: Any,
    edge_Zsin: Any,
    free_boundary_enabled: bool,
    host_update_assembly: bool,
    setup_host_enforce: bool,
    idx00: int,
    mpol: int,
    nrange: int,
    state0_dtype: Any,
    apply_lambda_axis_rules: Callable[[Any], Any],
    tree_has_tracer: Callable[[Any], bool],
    has_jax_func: Callable[[], bool] = has_jax,
) -> ResidualStateSetup:
    """Build the initial constrained state and per-iteration cached constants."""

    if bool(host_update_assembly) or bool(setup_host_enforce):
        if getattr(static, "m_is_m0", None) is not None:
            precomputed_axis_mask_np = np.asarray(static.m_is_m0, dtype=state0_dtype)
        else:
            precomputed_axis_mask_np = (np.asarray(static.modes.m) == 0).astype(state0_dtype)
    else:
        precomputed_axis_mask_np = None

    if bool(host_update_assembly) and bool(has_jax_func()):
        if tree_has_tracer(state0):
            jnp_state_dtype = jnp.asarray(state0.Rcos).dtype
            jnp_zero_m1_0 = jnp.asarray(0.0, dtype=jnp_state_dtype)
            jnp_zero_m1_1 = jnp.asarray(1.0, dtype=jnp_state_dtype)
            jnp_true_bool = jnp.asarray(True, dtype=bool)
            jnp_false_bool = jnp.asarray(False, dtype=bool)
        else:
            jnp_state_dtype = np.asarray(state0.Rcos).dtype
            jnp_zero_m1_0 = np.asarray(0.0, dtype=jnp_state_dtype)
            jnp_zero_m1_1 = np.asarray(1.0, dtype=jnp_state_dtype)
            jnp_true_bool = np.asarray(True, dtype=bool)
            jnp_false_bool = np.asarray(False, dtype=bool)
    else:
        jnp_state_dtype = None
        jnp_zero_m1_0 = None
        jnp_zero_m1_1 = None
        jnp_true_bool = None
        jnp_false_bool = None

    zeros_coeff_np = None
    zeros_dR_np = None
    if bool(host_update_assembly):
        coeff_shape_np = (int(np.asarray(state0.Rcos).shape[0]), int(mpol), int(nrange))
        zeros_coeff_np = np.zeros(coeff_shape_np, dtype=state0_dtype)
        zeros_dR_np = np.zeros_like(np.asarray(state0.Rcos))

    if bool(host_update_assembly) and (not tree_has_tracer(s)) and (not tree_has_tracer(state0.Rcos)):
        s_np = np.asarray(s)
        delta_s = (
            np.asarray(s_np[1] - s_np[0], dtype=state0_dtype)
            if int(s_np.shape[0]) > 1
            else np.asarray(1.0, dtype=state0_dtype)
        )
    else:
        s_j = jnp.asarray(s)
        delta_s = (
            jnp.asarray(s_j[1] - s_j[0], dtype=jnp.asarray(state0.Rcos).dtype)
            if int(s_j.shape[0]) > 1
            else jnp.asarray(1.0, dtype=jnp.asarray(state0.Rcos).dtype)
        )

    if bool(host_update_assembly) or bool(setup_host_enforce):
        state = _enforce_fixed_boundary_and_axis_np(
            state0,
            static,
            edge_Rcos=edge_Rcos,
            edge_Rsin=edge_Rsin,
            edge_Zcos=edge_Zcos,
            edge_Zsin=edge_Zsin,
            enforce_edge=not bool(free_boundary_enabled),
            enforce_lambda_axis=True,
            idx00=idx00,
            precomputed_axis_mask=precomputed_axis_mask_np,
        )
    else:
        state = _enforce_fixed_boundary_and_axis(
            state0,
            static,
            edge_Rcos=edge_Rcos,
            edge_Rsin=edge_Rsin,
            edge_Zcos=edge_Zcos,
            edge_Zsin=edge_Zsin,
            enforce_edge=not bool(free_boundary_enabled),
            enforce_lambda_axis=True,
            idx00=idx00,
        )

    return ResidualStateSetup(
        state=apply_lambda_axis_rules(state),
        precomputed_axis_mask_np=precomputed_axis_mask_np,
        jnp_state_dtype=jnp_state_dtype,
        jnp_zero_m1_0=jnp_zero_m1_0,
        jnp_zero_m1_1=jnp_zero_m1_1,
        jnp_true_bool=jnp_true_bool,
        jnp_false_bool=jnp_false_bool,
        zeros_coeff_np=zeros_coeff_np,
        zeros_dR_np=zeros_dR_np,
        delta_s=delta_s,
    )


__all__ = ["ResidualStateSetup", "build_residual_state_setup"]
