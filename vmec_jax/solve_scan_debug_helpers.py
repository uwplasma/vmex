"""Small debug/printing helpers for residual-iteration scan paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from .solve_diagnostics_io import (
    _format_axis_coeff,
    _format_time_control_trace_row,
    _format_vmec2000_iter_row,
)


def _print_vmec2000_row(
    *,
    iter_idx: int,
    fsqr: float,
    fsqz: float,
    fsql: float,
    delt0r: float,
    r00: float,
    w_mhd: float,
    lasym: bool,
    z00: float | None = None,
    verbose: bool = True,
    vmec2000_control: bool = True,
    verbose_vmec2000_table: bool = True,
    print_live: bool = True,
) -> bool:
    """Print one VMEC2000-style iteration row when controls allow it."""

    if not (bool(verbose) and bool(vmec2000_control) and bool(verbose_vmec2000_table) and bool(print_live)):
        return False
    print(
        _format_vmec2000_iter_row(
            iter_idx=iter_idx,
            fsqr=fsqr,
            fsqz=fsqz,
            fsql=fsql,
            delt0r=delt0r,
            r00=r00,
            w_mhd=w_mhd,
            lasym=bool(lasym),
            z00=z00,
        ),
        flush=True,
    )
    return True


def _axis_guess_lines(raxis_cc: Any, zaxis_cs: Any) -> tuple[str, str, str, str]:
    r_line = "      RAXIS_CC =    " + "   ".join(_format_axis_coeff(v) for v in np.ravel(raxis_cc))
    z_line = "      ZAXIS_CS =    " + "   ".join(_format_axis_coeff(v) for v in np.ravel(zaxis_cs))
    return (
        "  ---- Improved AXIS Guess ----",
        r_line,
        z_line,
        "  -----------------------------",
    )


def _print_axis_guess(raxis_cc: Any, zaxis_cs: Any) -> bool:
    """Print improved axis coefficients, matching the historic best-effort behavior."""

    try:
        for line in _axis_guess_lines(raxis_cc, zaxis_cs):
            print(line, flush=True)
    except Exception:
        return False
    return True


def _timecontrol_scan_stage_name(stage_id: int) -> str:
    return {0: "init", 1: "pre", 2: "checkpoint", 3: "restart"}.get(int(stage_id), "pre")


def _append_timecontrol_scan_trace_row(
    path: str | Path,
    *,
    stage_id: int,
    iter2: int,
    iter1: int,
    fsq: float,
    fsq0: float,
    res0: float,
    res1: float,
    time_step: float,
    irst: int,
) -> bool:
    """Append a scan time-control trace row, swallowing I/O failures like solve.py."""

    try:
        with Path(path).open("a", encoding="utf-8") as f:
            f.write(
                _format_time_control_trace_row(
                    stage=_timecontrol_scan_stage_name(stage_id),
                    iter2=int(iter2),
                    iter1=int(iter1),
                    fsq=float(fsq),
                    fsq0=float(fsq0),
                    res0=float(res0),
                    res1=float(res1),
                    time_step=float(time_step),
                    irst=int(irst),
                )
            )
    except Exception:
        return False
    return True


def _emit_scan_prints(
    *,
    hist_np: tuple[Any, ...],
    it_start: int,
    max_iter_local: int,
    scan_minimal: bool,
    scan_light: bool,
    ftol: float,
    fsq_total_target: float | None,
    iter_offset0: int,
    should_print: Callable[[int, int], bool],
    print_row: Callable[..., Any],
) -> bool:
    """Emit deferred scan print rows from collected scan history arrays."""

    if scan_minimal:
        return False
    if scan_light:
        (
            fsqr_h,
            fsqz_h,
            fsql_h,
            _accepted_h,
            r00_h,
            z00_h,
            w_mhd_h,
            dt_h,
            _bad_jac_h,
        ) = hist_np
    else:
        (
            fsqr_h,
            fsqz_h,
            fsql_h,
            _fsqr1_h,
            _fsqz1_h,
            _fsql1_h,
            _accepted_h,
            r00_h,
            z00_h,
            w_mhd_h,
            dt_h,
            _zero_m1_h,
            _include_edge_h,
            _res0_h,
            _res1_h,
            _iter1_h,
            _bad_jac_h,
            _min_tau_h,
            _max_tau_h,
            _ptau_min_h,
            _ptau_max_h,
            _tau_min_state_h,
            _tau_max_state_h,
            _badjac_ptau_h,
            _badjac_state_h,
        ) = hist_np
    conv_mask = (fsqr_h <= float(ftol)) & (fsqz_h <= float(ftol)) & (fsql_h <= float(ftol))
    if fsq_total_target is not None:
        conv_mask = conv_mask | ((fsqr_h + fsqz_h + fsql_h) <= float(fsq_total_target))
    conv_idx = int(np.argmax(conv_mask)) if bool(np.any(conv_mask)) else None
    n_iter_local = int(fsqr_h.shape[0])
    for i in range(n_iter_local):
        iter2 = int(it_start + i + 1 + int(iter_offset0))
        if iter2 > int(max_iter_local):
            break
        conv_now = (conv_idx is not None) and (i == conv_idx)
        if should_print(iter2, int(max_iter_local)) or conv_now:
            r00_val = float(f"{float(r00_h[i]):.3E}")
            z00_val = float(f"{float(z00_h[i]):.3E}")
            print_row(
                iter_idx=int(iter2),
                fsqr=float(fsqr_h[i]),
                fsqz=float(fsqz_h[i]),
                fsql=float(fsql_h[i]),
                delt0r=float(dt_h[i]),
                r00=r00_val,
                w_mhd=float(w_mhd_h[i]),
                z00=z00_val,
            )
        if conv_now:
            return True
    return False


def _record_scan_device_ready(
    *,
    start: float | None,
    dispatch_done: float,
    ready_done: float,
    stats: dict[str, float],
) -> bool:
    """Accumulate scan dispatch/ready timing counters."""

    if start is None:
        return False
    stats["scan_device_dispatch_s"] += dispatch_done - float(start)
    stats["scan_device_ready_s"] += ready_done - dispatch_done
    stats["scan_device_run_s"] += ready_done - float(start)
    return True
