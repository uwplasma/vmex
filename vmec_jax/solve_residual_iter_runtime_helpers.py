"""Runtime helper seams for ``solve_fixed_boundary_residual_iter``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np


def _ptau_dump_enabled(*, dump_ptau_env: str, dump_dir: str) -> bool:
    return str(dump_ptau_env).strip() not in ("", "0") and bool(str(dump_dir).strip())


def _format_ptau_dump_row(
    *,
    iter_idx: int,
    ptau_min: float,
    ptau_max: float,
    tau_min_state: float | None,
    tau_max_state: float | None,
    badjac_ptau: bool | None,
    badjac_state: bool | None,
    badjac_used: bool,
    mode: str,
    label: str,
) -> str:
    return (
        f"{int(iter_idx)} {label} {mode} "
        f"{float(ptau_min):.16e} {float(ptau_max):.16e} "
        f"{float(tau_min_state if tau_min_state is not None else float('nan')):.16e} "
        f"{float(tau_max_state if tau_max_state is not None else float('nan')):.16e} "
        f"{int(badjac_ptau) if badjac_ptau is not None else -1} "
        f"{int(badjac_state) if badjac_state is not None else -1} "
        f"{int(bool(badjac_used))}\n"
    )


def _maybe_dump_ptau(
    *,
    iter_idx: int,
    ptau_min: float,
    ptau_max: float,
    tau_min_state: float | None,
    tau_max_state: float | None,
    badjac_ptau: bool | None,
    badjac_state: bool | None,
    badjac_used: bool,
    mode: str,
    label: str,
    dump_ptau_env: str,
    dump_dir: str,
) -> bool:
    """Append a ptau diagnostic row, preserving solve.py's best-effort behavior."""

    if not _ptau_dump_enabled(dump_ptau_env=dump_ptau_env, dump_dir=dump_dir):
        return False
    try:
        path = Path(str(dump_dir).strip()) / "ptau_minmax.log"
        if not path.exists():
            with path.open("w", encoding="utf-8") as f:
                f.write("iter label mode ptau_min ptau_max state_min state_max bad_ptau bad_state bad_used\n")
        with path.open("a", encoding="utf-8") as f:
            f.write(
                _format_ptau_dump_row(
                    iter_idx=iter_idx,
                    ptau_min=ptau_min,
                    ptau_max=ptau_max,
                    tau_min_state=tau_min_state,
                    tau_max_state=tau_max_state,
                    badjac_ptau=badjac_ptau,
                    badjac_state=badjac_state,
                    badjac_used=badjac_used,
                    mode=mode,
                    label=label,
                )
            )
    except Exception:
        return False
    return True


def _scan_block_until_ready(
    value: Any,
    *,
    block_until_ready: Callable[[Any], Any],
    tree_map: Callable[[Callable[[Any], Any], Any], Any],
) -> Any:
    """Synchronize a JAX value, falling back to per-leaf ``block_until_ready``."""

    try:
        return block_until_ready(value)
    except Exception:
        return tree_map(
            lambda a: a.block_until_ready() if hasattr(a, "block_until_ready") else a,
            value,
        )


def _scan_device_run_ready(
    *,
    start: float | None,
    value: Any,
    scan_timing_enabled: bool,
    perf_counter: Callable[[], float],
    block_until_ready: Callable[[Any], Any],
    tree_map: Callable[[Callable[[Any], Any], Any], Any],
    record_ready: Callable[..., bool],
    stats: dict[str, float],
) -> Any:
    """Block for scan completion and record dispatch/ready timing when enabled."""

    if not bool(scan_timing_enabled) or start is None:
        return value
    dispatch_done = perf_counter()
    value = _scan_block_until_ready(
        value,
        block_until_ready=block_until_ready,
        tree_map=tree_map,
    )
    ready_done = perf_counter()
    record_ready(
        start=start,
        dispatch_done=dispatch_done,
        ready_done=ready_done,
        stats=stats,
    )
    return value


def _converged_residuals_scan_fast(
    fsqr: Any,
    fsqz: Any,
    fsql: Any,
    *,
    ftol: Any,
    fsq_total_target: Any | None,
) -> Any:
    strict = (fsqr <= ftol) & (fsqz <= ftol) & (fsql <= ftol)
    if fsq_total_target is None:
        return strict
    return strict | ((fsqr + fsqz + fsql) <= fsq_total_target)


def _vmec_freeb_plascur_from_bcovar(
    bc_obj: Any,
    fallback: float,
    *,
    plascur_edge_from_bcovar: Callable[..., Any],
    trig: Any,
    wout: Any,
    s: Any,
) -> float:
    """Best-effort VMEC ``ctor`` proxy used by NESTOR."""

    try:
        ctor = plascur_edge_from_bcovar(
            bc=bc_obj,
            trig=trig,
            wout=wout,
            s=s,
        )
        ctor_f = float(np.asarray(ctor))
        if np.isfinite(ctor_f):
            return float(ctor_f)
    except Exception:
        pass
    return float(fallback)


def _scan_print_uses_debug_print(*, scan_print_mode: str, debug_print_fn: Any) -> bool:
    return str(scan_print_mode) == "debug_print" and debug_print_fn is not None


def _scan_print_uses_debug_callback(*, scan_print_mode: str, debug_module: Any) -> bool:
    return str(scan_print_mode) == "debug_callback" and debug_module is not None


def _scan_print_uses_io_callback(*, scan_print_mode: str, io_callback_fn: Any) -> bool:
    return str(scan_print_mode) == "io_callback" and io_callback_fn is not None
