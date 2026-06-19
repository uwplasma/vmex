"""Pure update helpers for the residual-iteration VMEC solve."""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from ...._compat import jax, jnp
from ...._solve_runtime import _tree_has_tracer


class ResidualVelocityBlocks(NamedTuple):
    rcc: Any
    rss: Any
    rsc: Any
    rcs: Any
    zsc: Any
    zcs: Any
    zcc: Any
    zss: Any
    lsc: Any
    lcs: Any
    lcc: Any
    lss: Any


class HostMomentumUpdate(NamedTuple):
    velocities: ResidualVelocityBlocks
    update_rms: Any


class HostCatastrophicRestartUpdate(NamedTuple):
    """Scalar state after a host-loop catastrophic trial restart."""

    time_step: float
    ijacob: int
    restart_reason: str
    step_status: str
    restart_path: str
    max_coeff_delta_rms: float
    max_update_rms: float
    bad_resets: int
    iter1: int
    fsq_prev: float
    fsq0_prev: float
    inv_tau: list[float]
    update_rms: float


class BacktrackingMomentumSearchResult(NamedTuple):
    """Result of the non-strict host backtracking momentum search."""

    state: Any
    velocities: ResidualVelocityBlocks
    dt_eff: float
    update_rms: float
    step_status: str
    accepted: bool


def candidate_state_from_deltas(
    *,
    state: Any,
    static: Any,
    dR_value: Any,
    dR_sin_value: Any,
    dZ_cos_value: Any,
    dZ_value: Any,
    dL_cos_value: Any,
    dL_value: Any,
    use_numpy_arrays: bool,
    use_numpy_enforce: bool,
    edge_Rcos: Any,
    edge_Rsin: Any,
    edge_Zcos: Any,
    edge_Zsin: Any,
    free_boundary_enabled: bool,
    idx00: int,
    precomputed_axis_mask: Any,
    enforce_fixed_boundary_and_axis: Any,
    enforce_fixed_boundary_and_axis_np: Any,
    apply_vmec_lambda_axis_rules: Any,
):
    """Build a candidate VMEC state after one residual update proposal."""

    from ....state import VMECState

    array = np.asarray if use_numpy_arrays else jnp.asarray
    candidate = VMECState(
        layout=state.layout,
        Rcos=array(state.Rcos) + array(dR_value),
        Rsin=array(state.Rsin) + array(dR_sin_value),
        Zcos=array(state.Zcos) + array(dZ_cos_value),
        Zsin=array(state.Zsin) + array(dZ_value),
        Lcos=array(state.Lcos) + array(dL_cos_value),
        Lsin=array(state.Lsin) + array(dL_value),
    )
    if use_numpy_enforce:
        candidate = enforce_fixed_boundary_and_axis_np(
            candidate,
            static,
            edge_Rcos=edge_Rcos,
            edge_Rsin=edge_Rsin,
            edge_Zcos=edge_Zcos,
            edge_Zsin=edge_Zsin,
            enforce_edge=not bool(free_boundary_enabled),
            enforce_lambda_axis=True,
            idx00=idx00,
            precomputed_axis_mask=precomputed_axis_mask,
        )
    else:
        candidate = enforce_fixed_boundary_and_axis(
            candidate,
            static,
            edge_Rcos=edge_Rcos,
            edge_Rsin=edge_Rsin,
            edge_Zcos=edge_Zcos,
            edge_Zsin=edge_Zsin,
            enforce_edge=not bool(free_boundary_enabled),
            enforce_lambda_axis=True,
            idx00=idx00,
        )
    return apply_vmec_lambda_axis_rules(candidate)


def delta_tuple_from_blocks(
    dt,
    transforms,
    *blocks,
    lasym: bool,
    zeros_dR_np: Any | None = None,
    use_numpy_lasym_zeros: bool = False,
):
    """Transform velocity blocks into physical R/Z/lambda update arrays."""

    rcc, rss, rsc, rcs, zsc, zcs, zcc, zss, lsc, lcs, lcc, lss = blocks
    mn_cos_to_signed, mn_sin_to_signed, mn_cos_to_signed_lambda, mn_sin_to_signed_lambda = transforms
    dR = dt * mn_cos_to_signed(rcc, rss)
    dZ = dt * mn_sin_to_signed(zsc, zcs)
    dL = dt * mn_sin_to_signed_lambda(lsc, lcs)
    if bool(lasym):
        dR_sin = dt * mn_sin_to_signed(rsc, rcs)
        dZ_cos = dt * mn_cos_to_signed(zcc, zss)
        dL_cos = dt * mn_cos_to_signed_lambda(lcc, lss)
    elif use_numpy_lasym_zeros:
        dR_sin = zeros_dR_np
        dZ_cos = zeros_dR_np
        dL_cos = zeros_dR_np
    else:
        dR_sin = jnp.zeros_like(dR)
        dZ_cos = jnp.zeros_like(dR)
        dL_cos = jnp.zeros_like(dR)
    return (dR, dR_sin, dZ_cos, dZ, dL_cos, dL)


def candidate_state_from_delta_tuple(
    deltas,
    *,
    scale: float,
    use_numpy_arrays: bool,
    use_numpy_enforce: bool,
    candidate_from_deltas: Any,
):
    """Build a candidate state from an already transformed delta tuple."""

    if float(scale) != 1.0:
        deltas = tuple(float(scale) * value for value in deltas)
    dR, dR_sin, dZ_cos, dZ, dL_cos, dL = deltas
    return candidate_from_deltas(
        dR_value=dR,
        dR_sin_value=dR_sin,
        dZ_cos_value=dZ_cos,
        dZ_value=dZ,
        dL_cos_value=dL_cos,
        dL_value=dL,
        use_numpy_arrays=use_numpy_arrays,
        use_numpy_enforce=use_numpy_enforce,
    )


def zero_velocity_blocks_like(*blocks):
    """Return zeroed velocity blocks with each input block's shape and dtype."""

    out = []
    for block in blocks:
        if _tree_has_tracer(block):
            out.append(jnp.zeros_like(block))
            continue
        try:
            if jax is not None and isinstance(block, jax.Array):
                out.append(jnp.zeros_like(block))
                continue
        except Exception:
            pass
        out.append(np.zeros_like(np.asarray(block)))
    return tuple(out)


def scale_velocity_blocks(scale: float, *blocks):
    """Scale velocity blocks uniformly while preserving JAX array semantics."""

    return tuple(float(scale) * block for block in blocks)


def force_update_rms(scale: float, *blocks):
    """Return the JAX-visible RMS coefficient update implied by scaled blocks."""

    if not blocks:
        return jnp.asarray(0.0)
    total = None
    scale_j = jnp.asarray(scale)
    for block in blocks:
        term = (scale_j * block) ** 2
        total = term if total is None else total + term
    return jnp.sqrt(jnp.mean(total))


def host_force_update_rms(scale: float, *blocks) -> float:
    """Return the host scalar RMS coefficient update implied by scaled blocks."""

    return float(np.asarray(force_update_rms(scale, *blocks)))


def momentum_update_jax(
    *,
    velocities: ResidualVelocityBlocks,
    forces: ResidualVelocityBlocks,
    b1: float,
    fac: float,
    force_scale: float,
    flip_sign: float,
    dt_eff: float,
    compute_update_rms: bool,
) -> HostMomentumUpdate:
    """Apply the JAX-visible strict momentum update to all velocity blocks."""

    updated = ResidualVelocityBlocks(
        *(
            fac * (b1 * velocity + force_scale * (flip_sign * jnp.asarray(force)))
            for velocity, force in zip(velocities, forces)
        )
    )
    if compute_update_rms:
        update_rms = force_update_rms(dt_eff, *updated)
    else:
        update_rms = jnp.asarray(0.0, dtype=jnp.asarray(updated.rcc).dtype)
    return HostMomentumUpdate(velocities=updated, update_rms=update_rms)


def host_momentum_update_np(
    *,
    velocities: ResidualVelocityBlocks,
    forces: ResidualVelocityBlocks,
    b1: float,
    fac: float,
    force_scale: float,
    flip_sign: float,
    dt_eff: float,
    compute_update_rms: bool,
) -> HostMomentumUpdate:
    """Apply the host strict momentum update to all velocity blocks."""
    velocity_stack = np.stack([np.asarray(block) for block in velocities])
    force_stack = np.stack([np.asarray(block) for block in forces])

    np.multiply(velocity_stack, float(fac) * float(b1), out=velocity_stack)
    np.multiply(force_stack, float(fac) * float(force_scale) * float(flip_sign), out=force_stack)
    np.add(velocity_stack, force_stack, out=velocity_stack)

    if compute_update_rms:
        flat = velocity_stack.ravel()
        update_rms = abs(float(dt_eff)) * np.sqrt(np.dot(flat, flat) / velocity_stack.size)
    else:
        update_rms = np.asarray(0.0, dtype=velocity_stack.dtype)

    return HostMomentumUpdate(
        velocities=ResidualVelocityBlocks(*velocity_stack),
        update_rms=update_rms,
    )


def host_catastrophic_restart_update(
    *,
    probe_bad_jacobian: bool,
    w_try: float,
    time_step: float,
    restart_badjac_factor: float,
    restart_badprog_factor: float,
    step_size: float,
    ijacob: int,
    bad_resets: int,
    iter2: int,
    fsq_prev_before: float,
    fsq0_prev_before: float,
    k_ndamp: int,
    max_coeff_delta_rms: float,
    max_update_rms: float,
) -> HostCatastrophicRestartUpdate:
    """Apply VMEC-style scalar updates after rejecting a catastrophic trial.

    The caller owns the large state rollback and velocity zeroing.  This helper
    keeps the scalar branch policy in one place so scan/replay fingerprints can
    reason about the same restart semantics.
    """

    max_coeff_delta_rms_next = max(0.5 * float(max_coeff_delta_rms), 1.0e-12)
    max_update_rms_next = max(0.8 * float(max_update_rms), 1.0e-6)

    if bool(probe_bad_jacobian) or (not np.isfinite(float(w_try))):
        time_step_next = max(float(restart_badjac_factor) * float(time_step), 1.0e-12)
        ijacob_next = int(ijacob) + 1
        restart_reason = "bad_jacobian"
        step_status = "restart_bad_jacobian"
        restart_path = "catastrophic_nonfinite"
    else:
        time_step_next = max(float(time_step) / float(restart_badprog_factor), 1.0e-12)
        ijacob_next = int(ijacob)
        restart_reason = "bad_progress"
        step_status = "restart_bad_progress"
        restart_path = "catastrophic_growth"

    if ijacob_next in (25, 50):
        scale = 0.98 if ijacob_next < 50 else 0.96
        time_step_next = max(scale * float(step_size), 1.0e-12)

    return HostCatastrophicRestartUpdate(
        time_step=float(time_step_next),
        ijacob=int(ijacob_next),
        restart_reason=restart_reason,
        step_status=step_status,
        restart_path=restart_path,
        max_coeff_delta_rms=float(max_coeff_delta_rms_next),
        max_update_rms=float(max_update_rms_next),
        bad_resets=int(bad_resets) + 1,
        iter1=int(iter2),
        fsq_prev=float(fsq_prev_before),
        fsq0_prev=float(fsq0_prev_before),
        inv_tau=[0.15 / float(time_step_next)] * int(k_ndamp),
        update_rms=0.0,
    )


def backtracking_momentum_search(
    *,
    state: Any,
    velocities: ResidualVelocityBlocks,
    forces: ResidualVelocityBlocks,
    time_step: float,
    step_size: float,
    b1: float,
    fac: float,
    flip_sign: float,
    w_curr: float,
    delta_transforms: tuple,
    delta_tuple_from_blocks: Any,
    candidate_state_from_delta_tuple: Any,
    freeb_bsqvac_half_for_trial_state: Any,
    trial_residual_total: Any,
    max_backtracks: int = 6,
    accept_ratio: float = 1.05,
) -> BacktrackingMomentumSearchResult:
    """Try non-strict momentum updates, halving the step until residual growth is acceptable."""

    accepted = False
    step_status = "rejected"
    step_factor = 1.0
    best_state = state
    best_velocities = velocities
    dt_eff = float(time_step)
    update_rms = 0.0

    for _ in range(int(max_backtracks)):
        dt_try = float(time_step) * step_factor
        trial_velocities = ResidualVelocityBlocks(
            *(
                fac * (b1 * velocity + dt_try * (flip_sign * jnp.asarray(force)))
                for velocity, force in zip(velocities, forces)
            )
        )
        state_try = candidate_state_from_delta_tuple(
            delta_tuple_from_blocks(dt_try, delta_transforms, *trial_velocities),
            use_numpy_arrays=False,
            use_numpy_enforce=False,
        )
        freeb_bsqvac_half_trial = freeb_bsqvac_half_for_trial_state(state_try)
        w_try = trial_residual_total(state_try, freeb_bsqvac_half_trial)
        if np.isfinite(w_try) and (w_try <= float(accept_ratio) * float(w_curr)):
            accepted = True
            step_status = "momentum"
            best_state = state_try
            best_velocities = trial_velocities
            dt_eff = float(dt_try)
            update_rms = host_force_update_rms(dt_try, *trial_velocities)
            break
        step_factor *= 0.5

    if not accepted:
        best_velocities = ResidualVelocityBlocks(*scale_velocity_blocks(0.5, *best_velocities))
        dt_eff = float(step_size) * step_factor
        update_rms = 0.0

    return BacktrackingMomentumSearchResult(
        state=best_state,
        velocities=best_velocities,
        dt_eff=float(dt_eff),
        update_rms=float(update_rms),
        step_status=step_status,
        accepted=bool(accepted),
    )


_ResidualVelocityBlocks = ResidualVelocityBlocks
_HostCatastrophicRestartUpdate = HostCatastrophicRestartUpdate
_zero_velocity_blocks_like = zero_velocity_blocks_like
_scale_velocity_blocks = scale_velocity_blocks
_host_force_update_rms = host_force_update_rms
_momentum_update_jax = momentum_update_jax
_host_momentum_update_np = host_momentum_update_np
_host_catastrophic_restart_update = host_catastrophic_restart_update
