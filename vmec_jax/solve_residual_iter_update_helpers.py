"""Pure update helpers for the residual-iteration VMEC solve."""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np


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


_ResidualVelocityBlocks = ResidualVelocityBlocks
_host_momentum_update_np = host_momentum_update_np
