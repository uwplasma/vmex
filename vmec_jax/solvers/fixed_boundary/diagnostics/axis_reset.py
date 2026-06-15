"""Initial magnetic-axis reset helpers for VMEC residual solves."""

from __future__ import annotations

from pathlib import Path
import os
from typing import NamedTuple

import numpy as np

from ...._compat import jnp
from ....state import VMECState


class InitialAxisResetDecision(NamedTuple):
    """Pure control decision for VMEC-style initial magnetic-axis resets."""

    bad_jacobian: bool
    force_reset: bool
    reset: bool


def merge_axis_reset_state(*, st: VMECState, st_axis: VMECState, static, full_reset: bool) -> VMECState:
    """Return an axis-reset state, preserving non-axis coefficients unless requested."""

    if full_reset:
        return st_axis
    if getattr(static, "m_is_m0", None) is None:
        mask_m0 = jnp.asarray(np.asarray(static.modes.m, dtype=int) == 0, dtype=jnp.asarray(st.Rcos).dtype)
    else:
        mask_m0 = jnp.asarray(static.m_is_m0, dtype=jnp.asarray(st.Rcos).dtype)
    Rcos = jnp.where(mask_m0[None, :] != 0, jnp.asarray(st_axis.Rcos), jnp.asarray(st.Rcos))
    Rsin = jnp.where(mask_m0[None, :] != 0, jnp.asarray(st_axis.Rsin), jnp.asarray(st.Rsin))
    Zcos = jnp.where(mask_m0[None, :] != 0, jnp.asarray(st_axis.Zcos), jnp.asarray(st.Zcos))
    Zsin = jnp.where(mask_m0[None, :] != 0, jnp.asarray(st_axis.Zsin), jnp.asarray(st.Zsin))
    return VMECState(
        layout=st.layout,
        Rcos=Rcos,
        Rsin=Rsin,
        Zcos=Zcos,
        Zsin=Zsin,
        Lcos=st.Lcos,
        Lsin=st.Lsin,
    )


def initial_axis_reset_decision(
    *,
    bad_jacobian_ptau: bool | None,
    bad_jacobian_state: bool,
    badjac_use_state: bool,
    fsq_phys: float | None,
    axis_reset_fsq_min: float,
    force_axis_reset: bool,
    axis_reset_always_3d: bool,
    lthreed: bool,
    vmec2000_control: bool = True,
    lmove_axis: bool = True,
    axis_reset_enabled: bool = True,
) -> InitialAxisResetDecision:
    """Pure control-flow gate for VMEC-style initial magnetic-axis resets."""

    if bad_jacobian_ptau is None:
        bad_jacobian = bool(bad_jacobian_state)
    elif bool(badjac_use_state):
        bad_jacobian = bool(bad_jacobian_ptau) and bool(bad_jacobian_state)
    else:
        bad_jacobian = bool(bad_jacobian_ptau)

    fsq_min = max(0.0, float(axis_reset_fsq_min))
    if bad_jacobian and fsq_min > 0.0:
        if fsq_phys is None:
            bad_jacobian = False
        else:
            fsq_val = float(fsq_phys)
            if (not np.isfinite(fsq_val)) or (fsq_val < fsq_min):
                bad_jacobian = False

    force_reset = bool(force_axis_reset) or (
        bool(vmec2000_control) and bool(lmove_axis) and bool(lthreed) and bool(axis_reset_always_3d)
    )
    return InitialAxisResetDecision(
        bad_jacobian=bool(bad_jacobian),
        force_reset=bool(force_reset),
        reset=bool(axis_reset_enabled) and (bool(bad_jacobian) or bool(force_reset)),
    )


def write_axis_reset_dump(
    *,
    axis_dump_dir: str | os.PathLike[str] | None,
    ns: int,
    ntor: int,
    used_state_guess: bool,
    raxis_cc,
    raxis_cs,
    zaxis_cc,
    zaxis_cs,
) -> bool:
    """Write optional magnetic-axis reset coefficients for diagnostics."""

    if axis_dump_dir is None or str(axis_dump_dir).strip() == "":
        return False
    try:
        p = Path(axis_dump_dir).expanduser().resolve()
        ntor_i = int(ntor)
        rcc = np.asarray(raxis_cc)
        rcs = np.asarray(raxis_cs)
        zcc = np.asarray(zaxis_cc)
        zcs = np.asarray(zaxis_cs)
        if min(rcc.size, rcs.size, zcc.size, zcs.size) < ntor_i + 1:
            return False
        p.mkdir(parents=True, exist_ok=True)
        out = p / f"axis_reset_ns{int(ns)}.dat"
        with out.open("w", encoding="utf-8") as f:
            f.write(f"# used_state_guess={int(bool(used_state_guess))}\n")
            f.write("n raxis_cc raxis_cs zaxis_cc zaxis_cs\n")
            for n in range(ntor_i + 1):
                f.write(
                    f"{n:4d} "
                    f"{float(rcc[n]): .16e} "
                    f"{float(rcs[n]): .16e} "
                    f"{float(zcc[n]): .16e} "
                    f"{float(zcs[n]): .16e}\n"
                )
        return True
    except Exception:
        return False
