"""Small free-boundary iteration-control helpers for VMEC solves."""

from __future__ import annotations

import os

import numpy as np

from .solvers.fixed_boundary.residual.update import (
    scale_velocity_blocks,
    zero_velocity_blocks_like,
)


def free_boundary_iter_controls(iter2: int, iter1: int, nvacskip: int) -> tuple[int, int]:
    """Return the reduced legacy ``(ivac, ivacskip)`` free-boundary cadence."""

    nv = max(1, int(nvacskip))
    ivs = int((int(iter2) - int(iter1)) % nv)
    ivac = 1 if ivs == 0 else 2
    return ivac, ivs


def free_boundary_iter_controls_vmec(
    *,
    iter2: int,
    iter1: int,
    ivac: int,
    nvacskip: int,
    nvskip0: int,
    fsq_rz_prev: float,
    activate_fsq: float | None = None,
) -> tuple[int, int, int]:
    """VMEC2000-style ``ivac/ivacskip/nvacskip`` update for ``funct3d`` cadence."""

    i2 = int(iter2)
    i1 = int(iter1)
    iv = int(ivac)
    nv = max(1, int(nvacskip))
    nv0 = max(1, int(nvskip0))
    fs = float(fsq_rz_prev)
    if not np.isfinite(fs) or fs < 0.0:
        fs = 1.0

    if activate_fsq is None:
        activate_threshold = float(os.getenv("VMEC_JAX_FREEB_ACTIVATE_FSQ", "1.0e-3") or 1.0e-3)
    else:
        activate_threshold = float(activate_fsq)

    if i2 > 1 and fs <= activate_threshold:
        iv += 1

    if iv < 0:
        return iv, 0, nv

    ivs = int((i2 - i1) % nv)
    if iv <= 2:
        ivs = 0

    if ivs == 0:
        nv_est = int(1.0 / max(1.0e-1, 1.0e11 * fs))
        nv = max(nv0, max(1, nv_est))

    return iv, ivs, nv


def free_boundary_prev_rz_fsq_next(
    *,
    prev_fsq_before: float,
    fsq_rz_curr: float,
    turnon_restart: bool,
    preserve_turnon_restart: bool,
) -> float:
    """Optionally carry the pre-turn-on residual into the next cadence step."""

    if bool(turnon_restart) and bool(preserve_turnon_restart):
        return float(prev_fsq_before)
    return float(fsq_rz_curr)


def free_boundary_should_damp_constraint_baseline(
    *,
    freeb_ivac: int,
    freeb_turnon_iter: bool,
    lthreed: bool,
) -> bool:
    """Return whether VMEC should damp persistent free-boundary constraint baselines."""

    if not bool(lthreed):
        return int(freeb_ivac) >= 0
    return int(freeb_ivac) >= 0 and (not bool(freeb_turnon_iter))


def free_boundary_turnon_resets_iter1_immediately(*, lthreed: bool, lasym: bool) -> bool:
    """Return whether turn-on should immediately reset ``iter1`` for cadence."""

    return (not bool(lthreed)) or (not bool(lasym))


_zero_velocity_blocks_like = zero_velocity_blocks_like
_scale_velocity_blocks = scale_velocity_blocks
