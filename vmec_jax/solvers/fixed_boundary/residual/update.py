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


def host_force_update_rms(scale: float, *blocks) -> float:
    """Return the RMS coefficient update implied by scaled force blocks."""

    if not blocks:
        return 0.0
    total = None
    for block in blocks:
        term = (float(scale) * block) ** 2
        total = term if total is None else total + term
    return float(np.asarray(jnp.sqrt(jnp.mean(total))))


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


_ResidualVelocityBlocks = ResidualVelocityBlocks
_HostCatastrophicRestartUpdate = HostCatastrophicRestartUpdate
_zero_velocity_blocks_like = zero_velocity_blocks_like
_scale_velocity_blocks = scale_velocity_blocks
_host_force_update_rms = host_force_update_rms
_host_momentum_update_np = host_momentum_update_np
_host_catastrophic_restart_update = host_catastrophic_restart_update
