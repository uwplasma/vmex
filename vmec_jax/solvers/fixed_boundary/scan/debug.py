"""Small debug/printing helpers for residual-iteration scan paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from ...._compat import jnp
from ..diagnostics.io import (
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


def _emit_vmec2000_iter_row(
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
    scan_print_mode: str = "debug_print",
    scan_print_ordered: bool = False,
    jax_debug: Any | None = None,
    io_callback: Any | None = None,
    print_row: Callable[..., Any] = _print_vmec2000_row,
) -> bool:
    """Emit one VMEC2000-style iteration row through the selected JAX-safe path."""

    if not (bool(verbose) and bool(vmec2000_control) and bool(verbose_vmec2000_table)):
        return False
    if not bool(print_live):
        return False

    z_val = float("nan") if z00 is None else float(z00)
    if jax_debug is None:
        print_row(
            iter_idx=iter_idx,
            fsqr=fsqr,
            fsqz=fsqz,
            fsql=fsql,
            delt0r=delt0r,
            r00=r00,
            w_mhd=w_mhd,
            lasym=bool(lasym),
            z00=z_val if lasym else z00,
        )
        return True

    if bool(lasym):
        if scan_print_mode == "debug_print":
            jax_debug.print(
                "{i:5d}{fsqr:10.2E}{fsqz:10.2E}{fsql:10.2E}{r00:11.3E}{z00:11.3E}{dt:10.2E}{w:12.4E}",
                i=iter_idx,
                fsqr=fsqr,
                fsqz=fsqz,
                fsql=fsql,
                r00=r00,
                z00=z_val,
                dt=delt0r,
                w=w_mhd,
                ordered=bool(scan_print_ordered),
            )
            return True
        if scan_print_mode == "debug_callback":

            def _cb(i, fsqr_v, fsqz_v, fsql_v, r00_v, z00_v, dt_v, w_v):
                print_row(
                    iter_idx=int(i),
                    fsqr=float(fsqr_v),
                    fsqz=float(fsqz_v),
                    fsql=float(fsql_v),
                    delt0r=float(dt_v),
                    r00=float(r00_v),
                    w_mhd=float(w_v),
                    lasym=True,
                    z00=float(z00_v),
                )
                return None

            jax_debug.callback(
                _cb,
                iter_idx,
                fsqr,
                fsqz,
                fsql,
                r00,
                z_val,
                delt0r,
                w_mhd,
                ordered=bool(scan_print_ordered),
            )
            return True

        if io_callback is None:
            return False

        def _cb_io(i, fsqr_v, fsqz_v, fsql_v, r00_v, z00_v, dt_v, w_v):
            print_row(
                iter_idx=int(i),
                fsqr=float(fsqr_v),
                fsqz=float(fsqz_v),
                fsql=float(fsql_v),
                delt0r=float(dt_v),
                r00=float(r00_v),
                w_mhd=float(w_v),
                lasym=True,
                z00=float(z00_v),
            )
            return ()

        io_callback(  # type: ignore[misc]
            _cb_io,
            None,
            iter_idx,
            fsqr,
            fsqz,
            fsql,
            r00,
            z_val,
            delt0r,
            w_mhd,
            ordered=bool(scan_print_ordered),
        )
        return True

    if scan_print_mode == "debug_print":
        jax_debug.print(
            "{i:5d}{fsqr:10.2E}{fsqz:10.2E}{fsql:10.2E}{r00:11.3E}{dt:10.2E}{w:12.4E}",
            i=iter_idx,
            fsqr=fsqr,
            fsqz=fsqz,
            fsql=fsql,
            r00=r00,
            dt=delt0r,
            w=w_mhd,
            ordered=bool(scan_print_ordered),
        )
        return True
    if scan_print_mode == "debug_callback":

        def _cb(i, fsqr_v, fsqz_v, fsql_v, r00_v, dt_v, w_v):
            print_row(
                iter_idx=int(i),
                fsqr=float(fsqr_v),
                fsqz=float(fsqz_v),
                fsql=float(fsql_v),
                delt0r=float(dt_v),
                r00=float(r00_v),
                w_mhd=float(w_v),
                lasym=False,
            )
            return None

        jax_debug.callback(
            _cb,
            iter_idx,
            fsqr,
            fsqz,
            fsql,
            r00,
            delt0r,
            w_mhd,
            ordered=bool(scan_print_ordered),
        )
        return True

    if io_callback is None:
        return False

    def _cb_io(i, fsqr_v, fsqz_v, fsql_v, r00_v, dt_v, w_v):
        print_row(
            iter_idx=int(i),
            fsqr=float(fsqr_v),
            fsqz=float(fsqz_v),
            fsql=float(fsql_v),
            delt0r=float(dt_v),
            r00=float(r00_v),
            w_mhd=float(w_v),
            lasym=False,
        )
        return ()

    io_callback(  # type: ignore[misc]
        _cb_io,
        None,
        iter_idx,
        fsqr,
        fsqz,
        fsql,
        r00,
        delt0r,
        w_mhd,
        ordered=bool(scan_print_ordered),
    )
    return True


def emit_live_scan_vmec2000_row(
    *,
    enabled: bool,
    sample_vmec: Any,
    iter_idx: Any,
    fsqr: Any,
    fsqz: Any,
    fsql: Any,
    delt0r: Any,
    r00: Any,
    w_mhd: Any,
    scan_print_mode: str,
    scan_print_ordered: bool,
    jax_debug: Any | None,
    io_callback: Any | None,
    cond: Callable[..., Any],
    emit_iter_row: Callable[..., Any] = _emit_vmec2000_iter_row,
    print_row: Callable[..., Any] = _print_vmec2000_row,
) -> Any:
    """Emit one live VMEC scan row inside a JAX conditional when requested."""

    if not bool(enabled):
        return None

    def _do_print(_):
        emit_iter_row(
            iter_idx=iter_idx,
            fsqr=fsqr,
            fsqz=fsqz,
            fsql=fsql,
            delt0r=delt0r,
            r00=r00,
            w_mhd=w_mhd,
            lasym=False,
            verbose=True,
            vmec2000_control=True,
            verbose_vmec2000_table=True,
            print_live=True,
            scan_print_mode=scan_print_mode,
            scan_print_ordered=bool(scan_print_ordered),
            jax_debug=jax_debug,
            io_callback=io_callback,
            print_row=print_row,
        )
        return jnp.asarray(0, dtype=jnp.int32)

    return cond(
        sample_vmec,
        _do_print,
        lambda _: jnp.asarray(0, dtype=jnp.int32),
        operand=None,
    )


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


def emit_vmec2000_post_scan_rows(
    *,
    enabled: bool,
    scan_histories: Any,
    fsqr_full: Any,
    fsqz_full: Any,
    fsql_full: Any,
    conv_idx_print: int,
    max_iter: int,
    should_print: Callable[[int, int], bool],
    print_row: Callable[..., Any],
) -> int:
    """Replay VMEC2000-style iteration rows after a non-printing scan."""

    if not bool(enabled):
        return 0
    r00_full = np.asarray(scan_histories.r00)
    z00_full = np.asarray(scan_histories.z00)
    w_mhd_full = np.asarray(scan_histories.w_mhd)
    dt_full = np.asarray(scan_histories.dt)
    fsqr_arr = np.asarray(fsqr_full)
    fsqz_arr = np.asarray(fsqz_full)
    fsql_arr = np.asarray(fsql_full)
    last_iter = int(conv_idx_print) if int(conv_idx_print) > 0 else int(max_iter)
    printed = 0
    for i in range(last_iter):
        iter2 = i + 1
        if should_print(int(iter2), int(last_iter)):
            r00_val = float(f"{float(r00_full[i]):.3E}")
            z00_val = float(f"{float(z00_full[i]):.3E}")
            print_row(
                iter_idx=int(iter2),
                fsqr=float(fsqr_arr[i]),
                fsqz=float(fsqz_arr[i]),
                fsql=float(fsql_arr[i]),
                delt0r=float(dt_full[i]),
                r00=r00_val,
                w_mhd=float(w_mhd_full[i]),
                z00=z00_val,
            )
            printed += 1
    return printed


def dump_vmec2000_scan_ptau_rows(
    *,
    enabled: bool,
    scan_histories: Any,
    conv_idx_print: int,
    max_iter: int,
    iter_offset0: int,
    badjac_mode: str,
    dump_ptau: Callable[..., Any],
) -> int:
    """Replay VMEC2000 scan p-tau min/max diagnostics to the host dump path."""

    if not bool(enabled):
        return 0
    last_iter = int(conv_idx_print) if int(conv_idx_print) > 0 else int(max_iter)
    ptau_min_full = np.asarray(scan_histories.ptau_min)
    ptau_max_full = np.asarray(scan_histories.ptau_max)
    tau_min_state_full = np.asarray(scan_histories.tau_min_state)
    tau_max_state_full = np.asarray(scan_histories.tau_max_state)
    badjac_ptau_full = np.asarray(scan_histories.badjac_ptau).astype(int)
    badjac_state_full = np.asarray(scan_histories.badjac_state).astype(int)
    bad_jac_full = np.asarray(scan_histories.bad_jac)
    dumped = 0
    for i in range(last_iter):
        iter2 = i + 1 + int(iter_offset0)
        dumped += int(
            bool(
                dump_ptau(
                    iter_idx=int(iter2),
                    ptau_min=float(ptau_min_full[i]),
                    ptau_max=float(ptau_max_full[i]),
                    tau_min_state=float(tau_min_state_full[i]) if np.isfinite(tau_min_state_full[i]) else None,
                    tau_max_state=float(tau_max_state_full[i]) if np.isfinite(tau_max_state_full[i]) else None,
                    badjac_ptau=bool(badjac_ptau_full[i]),
                    badjac_state=bool(badjac_state_full[i]),
                    badjac_used=bool(bad_jac_full[i]),
                    mode=badjac_mode,
                    label="scan",
                )
            )
        )
    return dumped


def maybe_debug_scan_force_first_iter(
    *,
    enabled: bool,
    iter2: Any,
    frzl: Any,
    carry_state: Any,
    use_cached_precond: Any,
    need_bcovar_update: Any,
    norms_used: Any,
    gcr2: Any,
    gcz2: Any,
    fsqr: Any,
    fsqz: Any,
    jnp_module: Any,
    cond: Callable[..., Any],
    debug_print: Callable[..., Any] | None = None,
) -> bool:
    """Emit optional first-iteration force-channel diagnostics in scan mode."""

    if not bool(enabled):
        return False
    if debug_print is None:
        try:
            from jax import debug as jax_debug  # type: ignore

            debug_print = jax_debug.print
        except Exception:
            return False
    if debug_print is None:
        return False

    def _dbg(_):
        fzsc2 = jnp_module.sum(frzl.fzsc * frzl.fzsc)
        fzcs2 = (
            jnp_module.sum(frzl.fzcs * frzl.fzcs)
            if frzl.fzcs is not None
            else jnp_module.asarray(0.0, dtype=fsqz.dtype)
        )
        fzcs_m1 = (
            jnp_module.sum(frzl.fzcs[:, 1, :] * frzl.fzcs[:, 1, :])
            if frzl.fzcs is not None and int(jnp_module.asarray(frzl.fzcs).shape[1]) > 1
            else jnp_module.asarray(0.0, dtype=fsqz.dtype)
        )
        rcos_sum = jnp_module.sum(carry_state.Rcos)
        zsin_sum = jnp_module.sum(carry_state.Zsin)
        debug_print(
            "[scan-debug] iter={i} gcr2={gcr:.6e} gcz2={gcz:.6e} fzsc2={fzsc2:.6e} fzcs2={fzcs2:.6e} fzcs_m1={fzcsm1:.6e} rcos_sum={rcsum:.6e} zsin_sum={zssum:.6e} use_cached={uc} need_bcovar={nb} fnorm={fn:.6e} r1={r1:.6e} fsqr={fsqr:.6e} fsqz={fsqz:.6e}",
            i=iter2,
            gcr=gcr2,
            gcz=gcz2,
            fzsc2=fzsc2,
            fzcs2=fzcs2,
            fzcsm1=fzcs_m1,
            rcsum=rcos_sum,
            zssum=zsin_sum,
            uc=jnp_module.asarray(use_cached_precond, dtype=jnp_module.int32),
            nb=jnp_module.asarray(need_bcovar_update, dtype=jnp_module.int32),
            fn=norms_used.fnorm,
            r1=norms_used.r1,
            fsqr=fsqr,
            fsqz=fsqz,
        )
        return 0

    _ = cond(iter2 == 1, _dbg, lambda _: 0, operand=None)
    return True


def maybe_debug_scan_state_iter(
    *,
    scan_debug_iter: int,
    iter2: Any,
    carry_adv: Any,
    use_cached_precond: Any,
    need_bcovar_update: Any,
    norms_used: Any,
    gcr2: Any,
    gcz2: Any,
    gcl2: Any,
    jnp_module: Any,
    cond: Callable[..., Any],
    debug_print: Callable[..., Any] | None = None,
) -> bool:
    """Emit optional state/checkpoint diagnostics for one requested scan iteration."""

    if int(scan_debug_iter) <= 0:
        return False
    if debug_print is None:
        try:
            from jax import debug as jax_debug  # type: ignore

            debug_print = jax_debug.print
        except Exception:
            return False
    if debug_print is None:
        return False

    def _dbg_state(_):
        rcos_sum = jnp_module.sum(carry_adv.state.Rcos)
        zsin_sum = jnp_module.sum(carry_adv.state.Zsin)
        lsin_sum = jnp_module.sum(carry_adv.state.Lsin)
        rcos_ck = jnp_module.sum(carry_adv.state_checkpoint.Rcos)
        zsin_ck = jnp_module.sum(carry_adv.state_checkpoint.Zsin)
        lsin_ck = jnp_module.sum(carry_adv.state_checkpoint.Lsin)
        fsqr_dbg = norms_used.r1 * norms_used.fnorm * gcr2
        fsqz_dbg = norms_used.r1 * norms_used.fnorm * gcz2
        fsql_dbg = norms_used.fnormL * gcl2
        debug_print(
            "[scan-state] iter={i} rcos_sum={rc:.6e} zsin_sum={zs:.6e} lsin_sum={ls:.6e} "
            "rcos_ck={rck:.6e} zsin_ck={zck:.6e} lsin_ck={lck:.6e} "
            "use_cached={uc} need_bcovar={nb} gcr2={gcr:.6e} gcz2={gcz:.6e} gcl2={gcl:.6e} "
            "fnorm={fn:.6e} r1={r1:.6e} fsqr={fsqr:.6e} fsqz={fsqz:.6e} fsql={fsql:.6e}",
            i=iter2,
            rc=rcos_sum,
            zs=zsin_sum,
            ls=lsin_sum,
            rck=rcos_ck,
            zck=zsin_ck,
            lck=lsin_ck,
            uc=jnp_module.asarray(use_cached_precond, dtype=jnp_module.int32),
            nb=jnp_module.asarray(need_bcovar_update, dtype=jnp_module.int32),
            gcr=gcr2,
            gcz=gcz2,
            gcl=gcl2,
            fn=norms_used.fnorm,
            r1=norms_used.r1,
            fsqr=fsqr_dbg,
            fsqz=fsqz_dbg,
            fsql=fsql_dbg,
        )
        return 0

    _ = cond(iter2 == int(scan_debug_iter), _dbg_state, lambda _: 0, operand=None)
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


def _maybe_dump_timecontrol_scan(
    *,
    cond: Any,
    stage_id: Any,
    iter2: Any,
    iter1: Any,
    fsq: Any,
    fsq0: Any,
    res0: Any,
    res1: Any,
    time_step: Any,
    irst: Any,
    dump_timecontrol_scan: bool,
    timecontrol_callback: Any | None,
    timecontrol_path: str | Path | None,
    jax_module: Any,
    jnp_module: Any,
) -> Any:
    """Conditionally append one time-control trace row from inside a JAX scan."""

    if not bool(dump_timecontrol_scan) or timecontrol_callback is None or timecontrol_path is None:
        return jnp_module.asarray(0, dtype=jnp_module.int32)

    def _emit(args):
        (iter2_v, iter1_v, fsq_v, fsq0_v, res0_v, res1_v, time_step_v, irst_v, stage_id_v) = args
        _append_timecontrol_scan_trace_row(
            timecontrol_path,
            stage_id=int(stage_id_v),
            iter2=int(iter2_v),
            iter1=int(iter1_v),
            fsq=float(fsq_v),
            fsq0=float(fsq0_v),
            res0=float(res0_v),
            res1=float(res1_v),
            time_step=float(time_step_v),
            irst=int(irst_v),
        )
        return np.int32(0)

    def _call(_):
        return timecontrol_callback(
            _emit,
            jax_module.ShapeDtypeStruct((), jnp_module.int32),
            (
                iter2,
                iter1,
                fsq,
                fsq0,
                res0,
                res1,
                time_step,
                irst,
                stage_id,
            ),
            ordered=True,
        )

    return jax_module.lax.cond(
        cond,
        _call,
        lambda _: jnp_module.asarray(0, dtype=jnp_module.int32),
        operand=None,
    )


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
    stats: dict[str, float | int],
    cache_status: str | None = None,
) -> bool:
    """Accumulate scan dispatch/ready timing counters."""

    if start is None:
        return False
    dispatch_s = dispatch_done - float(start)
    ready_s = ready_done - dispatch_done
    run_s = ready_done - float(start)
    stats["scan_device_dispatch_s"] += dispatch_s
    stats["scan_device_ready_s"] += ready_s
    stats["scan_device_run_s"] += run_s
    status = str(cache_status or "").strip().lower()
    if status in ("hit", "miss", "bypass"):
        prefix = f"scan_runner_cache_{status}"
        stats[f"{prefix}_dispatch_s"] += dispatch_s
        stats[f"{prefix}_ready_s"] += ready_s
        stats[f"{prefix}_device_run_s"] += run_s
    return True
