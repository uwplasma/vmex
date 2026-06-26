"""Initial state/update constants for residual-iteration solves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, NamedTuple

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


class ResidualIndexStateSetup(NamedTuple):
    """Mode-transform and initial-state setup for residual iteration."""

    mpol: int
    ntor: int
    nrange: int
    ncoeff: int
    setup_host_enforce: bool
    mode_context: Any
    m0_mask: Any
    w_mode_mn: Any
    w_mode_mn_np: Any
    state0_dtype: Any
    mn_cos_to_signed: Any
    mn_sin_to_signed: Any
    mn_cos_to_signed_physical: Any
    mn_sin_to_signed_physical: Any
    mn_sin_to_signed_physical_lambda: Any
    mn_cos_to_signed_physical_lambda: Any
    physical_delta_transforms: tuple[Any, Any, Any, Any]
    internal_delta_transforms: tuple[Any, Any, Any, Any]
    rz_norm_np: Any
    rz_norm: Any
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


def prepare_residual_index_state_setup(
    *,
    static: Any,
    state0: Any,
    s: Any,
    edge_Rcos: Any,
    edge_Rsin: Any,
    edge_Zcos: Any,
    edge_Zsin: Any,
    free_boundary_enabled: bool,
    host_update_assembly: bool,
    use_scan: bool,
    state0_has_tracer: bool,
    divide_by_scalxc_for_update: bool,
    mode_diag_exponent: float,
    idx00: int,
    apply_lambda_axis_rules: Callable[[Any], Any],
    vmec_scalxc_from_s_func: Callable[[Any], Any],
    setup_host_enforce_env: str,
    backend_name: str,
    build_mode_transform_context_func: Callable[..., Any],
    resolve_setup_host_enforce_func: Callable[..., bool],
    tree_has_tracer_func: Callable[[Any], bool],
    has_jax_func: Callable[[], bool] = has_jax,
) -> ResidualIndexStateSetup:
    """Build mode transforms and the constrained initial residual state."""

    mpol = int(static.cfg.mpol)
    ntor = int(static.cfg.ntor)
    nrange = ntor + 1
    ncoeff = int(jnp.asarray(state0.Rcos).shape[1])
    setup_host_enforce = resolve_setup_host_enforce_func(
        setup_host_enforce_env=setup_host_enforce_env,
        host_update_assembly=bool(host_update_assembly),
        use_scan=bool(use_scan),
        state_has_tracer=state0_has_tracer,
        backend_name=backend_name,
    )
    mode_context = build_mode_transform_context_func(
        static=static,
        state0=state0,
        s=s,
        host_update_assembly=bool(host_update_assembly),
        setup_host_enforce=bool(setup_host_enforce),
        divide_by_scalxc_for_update=bool(divide_by_scalxc_for_update),
        mode_diag_exponent=mode_diag_exponent,
        tree_has_tracer=tree_has_tracer_func,
        vmec_scalxc_from_s=vmec_scalxc_from_s_func,
    )
    state_setup = build_residual_state_setup(
        state0=state0,
        static=static,
        s=s,
        edge_Rcos=edge_Rcos,
        edge_Rsin=edge_Rsin,
        edge_Zcos=edge_Zcos,
        edge_Zsin=edge_Zsin,
        free_boundary_enabled=bool(free_boundary_enabled),
        host_update_assembly=bool(host_update_assembly),
        setup_host_enforce=bool(setup_host_enforce),
        idx00=idx00,
        mpol=mpol,
        nrange=nrange,
        state0_dtype=mode_context.state0_dtype,
        apply_lambda_axis_rules=apply_lambda_axis_rules,
        tree_has_tracer=tree_has_tracer_func,
        has_jax_func=has_jax_func,
    )
    physical_delta_transforms = (
        mode_context.mn_cos_to_signed_physical,
        mode_context.mn_sin_to_signed_physical,
        mode_context.mn_cos_to_signed_physical_lambda,
        mode_context.mn_sin_to_signed_physical_lambda,
    )
    internal_delta_transforms = (
        mode_context.mn_cos_to_signed,
        mode_context.mn_sin_to_signed,
        mode_context.mn_cos_to_signed,
        mode_context.mn_sin_to_signed,
    )
    return ResidualIndexStateSetup(
        mpol=mpol,
        ntor=ntor,
        nrange=nrange,
        ncoeff=ncoeff,
        setup_host_enforce=bool(setup_host_enforce),
        mode_context=mode_context,
        m0_mask=mode_context.m0_mask,
        w_mode_mn=mode_context.w_mode_mn,
        w_mode_mn_np=mode_context.w_mode_mn_np,
        state0_dtype=mode_context.state0_dtype,
        mn_cos_to_signed=mode_context.mn_cos_to_signed,
        mn_sin_to_signed=mode_context.mn_sin_to_signed,
        mn_cos_to_signed_physical=mode_context.mn_cos_to_signed_physical,
        mn_sin_to_signed_physical=mode_context.mn_sin_to_signed_physical,
        mn_sin_to_signed_physical_lambda=mode_context.mn_sin_to_signed_physical_lambda,
        mn_cos_to_signed_physical_lambda=mode_context.mn_cos_to_signed_physical_lambda,
        physical_delta_transforms=physical_delta_transforms,
        internal_delta_transforms=internal_delta_transforms,
        rz_norm_np=mode_context.rz_norm_np,
        rz_norm=mode_context.rz_norm,
        state=state_setup.state,
        precomputed_axis_mask_np=state_setup.precomputed_axis_mask_np,
        jnp_state_dtype=state_setup.jnp_state_dtype,
        jnp_zero_m1_0=state_setup.jnp_zero_m1_0,
        jnp_zero_m1_1=state_setup.jnp_zero_m1_1,
        jnp_true_bool=state_setup.jnp_true_bool,
        jnp_false_bool=state_setup.jnp_false_bool,
        zeros_coeff_np=state_setup.zeros_coeff_np,
        zeros_dR_np=state_setup.zeros_dR_np,
        delta_s=state_setup.delta_s,
    )


__all__ = [
    "ResidualIndexStateSetup",
    "ResidualStateSetup",
    "build_residual_state_setup",
    "prepare_residual_index_state_setup",
]
