"""Utilities for the discrete-adjoint recovery path.

The first step is a structured view of the existing fixed-boundary residual
iteration history. This keeps the initial refactor narrow: no solver behavior
changes, only a stable extraction layer over the primal trace data already
recorded in diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ResidualIterationTrace:
    """Structured view of one fixed-boundary residual solve history."""

    iter2: np.ndarray
    step_status: np.ndarray
    restart_reason: np.ndarray
    pre_restart_reason: np.ndarray
    time_step: np.ndarray
    dt_eff: np.ndarray
    update_rms: np.ndarray
    include_edge: np.ndarray
    zero_m1: np.ndarray
    fsq_curr: np.ndarray
    fsq_try: np.ndarray
    fsq_prev: np.ndarray
    r00: np.ndarray
    z00: np.ndarray
    wb: np.ndarray
    wp: np.ndarray
    w_vmec: np.ndarray
    state_advanced: np.ndarray


def _array_from_diag(diagnostics: dict[str, Any], key: str, *, dtype=None) -> np.ndarray:
    value = diagnostics.get(key, np.zeros((0,), dtype=float))
    arr = np.asarray(value)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def residual_iteration_trace_from_result(result) -> ResidualIterationTrace:
    """Extract a compact, typed residual-iteration trace from a solver result."""
    diagnostics = getattr(result, "diagnostics", None)
    if not isinstance(diagnostics, dict):
        raise TypeError("result.diagnostics must be a dict")

    iter2 = _array_from_diag(diagnostics, "iter2_history", dtype=int)
    step_status = _array_from_diag(diagnostics, "step_status_history", dtype=object)
    restart_reason = _array_from_diag(diagnostics, "restart_reason_history", dtype=object)
    pre_restart_reason = _array_from_diag(diagnostics, "pre_restart_reason_history", dtype=object)
    time_step = _array_from_diag(diagnostics, "time_step_history", dtype=float)
    dt_eff = _array_from_diag(diagnostics, "dt_eff_history", dtype=float)
    update_rms = _array_from_diag(diagnostics, "update_rms_history", dtype=float)
    include_edge = _array_from_diag(diagnostics, "include_edge_history", dtype=int)
    zero_m1 = _array_from_diag(diagnostics, "zero_m1_history", dtype=int)
    fsq_curr = _array_from_diag(diagnostics, "w_curr_history", dtype=float)
    fsq_try = _array_from_diag(diagnostics, "w_try_history", dtype=float)
    fsq_prev = _array_from_diag(diagnostics, "fsq_prev_history", dtype=float)
    r00 = _array_from_diag(diagnostics, "r00_history", dtype=float)
    z00 = _array_from_diag(diagnostics, "z00_history", dtype=float)
    wb = _array_from_diag(diagnostics, "wb_history", dtype=float)
    wp = _array_from_diag(diagnostics, "wp_history", dtype=float)
    w_vmec = _array_from_diag(diagnostics, "w_vmec_history", dtype=float)

    lengths = {
        int(arr.shape[0])
        for arr in (
            iter2,
            step_status,
            restart_reason,
            pre_restart_reason,
            time_step,
            dt_eff,
            update_rms,
            include_edge,
            zero_m1,
            fsq_curr,
            fsq_try,
            fsq_prev,
            r00,
            z00,
            wb,
            wp,
            w_vmec,
        )
        if arr.ndim >= 1 and arr.shape[0] > 0
    }
    if len(lengths) > 1:
        raise ValueError(f"inconsistent residual trace lengths: {sorted(lengths)}")

    rejected = np.isin(
        step_status,
        np.asarray(["rejected", "restart_bad_progress", "restart_bad_jacobian"], dtype=object),
    )
    state_advanced = ~rejected

    return ResidualIterationTrace(
        iter2=iter2,
        step_status=step_status,
        restart_reason=restart_reason,
        pre_restart_reason=pre_restart_reason,
        time_step=time_step,
        dt_eff=dt_eff,
        update_rms=update_rms,
        include_edge=include_edge,
        zero_m1=zero_m1,
        fsq_curr=fsq_curr,
        fsq_try=fsq_try,
        fsq_prev=fsq_prev,
        r00=r00,
        z00=z00,
        wb=wb,
        wp=wp,
        w_vmec=w_vmec,
        state_advanced=state_advanced,
    )


__all__ = ["ResidualIterationTrace", "residual_iteration_trace_from_result"]
