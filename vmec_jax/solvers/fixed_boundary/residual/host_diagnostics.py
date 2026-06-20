"""Host-side VMEC2000 print/diagnostic contexts for residual iteration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from vmec_jax.solvers.fixed_boundary.diagnostics.io import (
    _format_residual_converged_message,
    _format_residual_iteration_update_message,
    _format_residual_physical_status_message,
)


@dataclass(frozen=True)
class Vmec2000PrintContext:
    """Bound host-side VMEC2000 row-print helpers for one residual solve."""

    nstep_screen: int
    print_iter_row: Callable[..., None]
    should_print: Callable[[int, int], bool]


@dataclass(frozen=True)
class VmecIterationScalars:
    """Screen/history scalars sampled from one residual iteration."""

    r00: float
    z00: float
    wb: float
    wp: float
    w_vmec: float


def resolve_vmec2000_print_context(
    *,
    cfg: Any,
    indata: Any,
    verbose: bool,
    vmec2000_control: bool,
    verbose_vmec2000_table: bool,
    getenv: Callable[[str, str], str],
    resolve_debug_print_config: Callable[..., Any],
    resolve_nstep_screen: Callable[..., int],
    emit_iter_row: Callable[..., None],
    should_print_row: Callable[..., bool],
    print_row: Callable[..., None],
) -> Vmec2000PrintContext:
    """Resolve row-printing policy and return bound print/cadence helpers."""

    debug_print_config = resolve_debug_print_config(
        print_env=getenv("VMEC_JAX_SCAN_PRINT", "1"),
        mode_env=getenv("VMEC_JAX_SCAN_PRINT_MODE", "debug_print"),
        ordered_env=getenv("VMEC_JAX_SCAN_PRINT_ORDERED", "0"),
    )
    scan_print_mode = debug_print_config.mode
    scan_print_ordered = debug_print_config.ordered
    print_live = debug_print_config.print_live
    jax_debug = None
    io_callback = None
    if print_live:
        try:
            from jax import debug as jax_debug  # type: ignore[assignment]
        except Exception:
            jax_debug = None
    if scan_print_mode == "io_callback":
        try:
            from jax.experimental import io_callback as io_callback  # type: ignore[assignment]
        except Exception:
            scan_print_mode = resolve_debug_print_config(
                print_env="1",
                mode_env=scan_print_mode,
                ordered_env="0",
                io_callback_available=False,
            ).mode
            io_callback = None

    nstep_screen = resolve_nstep_screen(
        indata_nstep=int(indata.get_int("NSTEP", 1)) if indata is not None else 1,
        override_env=getenv("VMEC_JAX_NSTEP_OVERRIDE", ""),
    )

    def print_iter_row(
        *,
        iter_idx: int,
        fsqr: float,
        fsqz: float,
        fsql: float,
        fsqr1: float,
        fsqz1: float,
        fsql1: float,
        delt0r: float,
        r00: float,
        w_mhd: float,
        z00: float | None = None,
    ) -> None:
        del fsqr1, fsqz1, fsql1
        emit_iter_row(
            iter_idx=iter_idx,
            fsqr=fsqr,
            fsqz=fsqz,
            fsql=fsql,
            delt0r=delt0r,
            r00=r00,
            w_mhd=w_mhd,
            lasym=bool(cfg.lasym),
            z00=z00,
            verbose=bool(verbose),
            vmec2000_control=bool(vmec2000_control),
            verbose_vmec2000_table=bool(verbose_vmec2000_table),
            print_live=bool(print_live),
            scan_print_mode=scan_print_mode,
            scan_print_ordered=bool(scan_print_ordered),
            jax_debug=jax_debug,
            io_callback=io_callback,
            print_row=print_row,
        )

    def should_print(iter_idx: int, max_iter: int) -> bool:
        return should_print_row(
            iter_idx=iter_idx,
            max_iter=max_iter,
            nstep_screen=nstep_screen,
            verbose=bool(verbose),
            vmec2000_control=bool(vmec2000_control),
            verbose_vmec2000_table=bool(verbose_vmec2000_table),
        )

    return Vmec2000PrintContext(
        nstep_screen=int(nstep_screen),
        print_iter_row=print_iter_row,
        should_print=should_print,
    )


def sample_vmec_iteration_scalars(
    *,
    need_scalar: bool,
    k: Any,
    state: Any,
    norms_current: Any,
    m0_mask: Any,
    lasym: bool,
    host_update_assembly: bool,
    vmec2000_control: bool,
    gamma: float,
    twopi: float,
    previous_r00: float,
    previous_z00: float,
    previous_wb: float,
    previous_wp: float,
    tree_has_tracer: Callable[[Any], bool],
    device_get_floats: Callable[..., tuple[float, ...]],
    jnp_module: Any,
) -> VmecIterationScalars:
    """Sample VMEC screen/history scalars with host/device parity rules."""

    if not bool(need_scalar):
        r00_val = float(previous_r00)
        z00_val = float(previous_z00)
        wb_val = float(previous_wb)
        wp_val = float(previous_wp)
    elif bool(host_update_assembly) and (not tree_has_tracer(k)):
        try:
            r00_val = float(np.asarray(k.pr1_even)[0, 0, 0])
            z00_val = float(np.asarray(k.pz1_even)[0, 0, 0]) if bool(lasym) else 0.0
        except Exception:
            if not np.any(m0_mask):
                r00_val = float("nan")
                z00_val = float("nan")
            else:
                r00_val = float(np.sum(np.asarray(state.Rcos)[0, m0_mask]))
                z00_val = float(np.sum(np.asarray(state.Zcos)[0, m0_mask])) if bool(lasym) else 0.0
        wb_val = float(np.asarray(norms_current.wb))
        wp_val = float(np.asarray(norms_current.wp))
    else:
        try:
            r00_j = jnp_module.asarray(k.pr1_even)[0, 0, 0]
            if bool(lasym):
                z00_j = jnp_module.asarray(k.pz1_even)[0, 0, 0]
            else:
                z00_j = jnp_module.asarray(0.0, dtype=jnp_module.asarray(r00_j).dtype)
        except Exception:
            if not np.any(m0_mask):
                r00_j = jnp_module.asarray(float("nan"))
                z00_j = jnp_module.asarray(float("nan"))
            else:
                r00_j = jnp_module.sum(jnp_module.asarray(state.Rcos)[0, m0_mask])
                if bool(lasym):
                    z00_j = jnp_module.sum(jnp_module.asarray(state.Zcos)[0, m0_mask])
                else:
                    z00_j = jnp_module.asarray(0.0, dtype=jnp_module.asarray(r00_j).dtype)
        # `norms_current` reflects the current bcovar state and therefore
        # matches VMEC's printed wb/wp even when the preconditioner norm is cached.
        wb_j = jnp_module.asarray(norms_current.wb)
        wp_j = jnp_module.asarray(norms_current.wp)
        r00_val, z00_val, wb_val, wp_val = device_get_floats(r00_j, z00_j, wb_j, wp_j)

    if bool(vmec2000_control):
        # Match VMEC's printed precision (E11.3) for parity checks.
        r00_val = float(f"{float(r00_val):.3E}")
        z00_val = float(f"{float(z00_val):.3E}")
    w_vmec = (float(wb_val) + float(wp_val) / (float(gamma) - 1.0)) * float(twopi) * float(twopi)
    return VmecIterationScalars(
        r00=float(r00_val),
        z00=float(z00_val),
        wb=float(wb_val),
        wp=float(wp_val),
        w_vmec=float(w_vmec),
    )


def print_compact_physical_residual_status(
    *,
    verbose: bool,
    vmec2000_control: bool,
    verbose_vmec2000_table: bool,
    iter_idx: int,
    fsqr: float,
    fsqz: float,
    fsql: float,
    include_edge: bool,
    print_func: Callable[..., None] = print,
) -> bool:
    """Print the compact non-VMEC physical residual status line."""

    if not (bool(verbose) and not (bool(vmec2000_control) and bool(verbose_vmec2000_table))):
        return False
    print_func(
        _format_residual_physical_status_message(
            iter_idx=iter_idx,
            fsqr=fsqr,
            fsqz=fsqz,
            fsql=fsql,
            include_edge=include_edge,
        ),
        flush=True,
    )
    return True


def print_compact_converged_status(
    *,
    verbose: bool,
    vmec2000_control: bool,
    verbose_vmec2000_table: bool,
    fsqr: float,
    fsqz: float,
    fsql: float,
    target: float,
    print_func: Callable[..., None] = print,
) -> bool:
    """Print the compact non-VMEC convergence status line."""

    if not (bool(verbose) and not (bool(vmec2000_control) and bool(verbose_vmec2000_table))):
        return False
    print_func(
        _format_residual_converged_message(
            fsqr=fsqr,
            fsqz=fsqz,
            fsql=fsql,
            target=target,
        ),
        flush=True,
    )
    return True


def print_compact_residual_iteration_update_status(
    *,
    verbose: bool,
    vmec2000_control: bool,
    verbose_vmec2000_table: bool,
    precond_diag_floats: Callable[[], tuple[float, float, float]],
    iter_idx: int,
    dt_eff: float,
    update_rms: float,
    step_status: str,
    print_func: Callable[..., None] = print,
) -> bool:
    """Print compact update status only on the non-VMEC table path."""

    if not (bool(verbose) and not (bool(vmec2000_control) and bool(verbose_vmec2000_table))):
        return False
    fsqr1_f, fsqz1_f, fsql1_f = precond_diag_floats()
    print_func(
        _format_residual_iteration_update_message(
            iter_idx=int(iter_idx),
            dt_eff=dt_eff,
            update_rms=update_rms,
            fsqr1=fsqr1_f,
            fsqz1=fsqz1_f,
            fsql1=fsql1_f,
            step_status=step_status,
        ),
        flush=True,
    )
    return True


def print_residual_iteration_update_status(
    *,
    verbose: bool,
    vmec2000_control: bool,
    verbose_vmec2000_table: bool,
    should_print_vmec2000: Callable[[int, int], bool],
    print_vmec2000_iter_row: Callable[..., None],
    precond_diag_floats: Callable[[], tuple[float, float, float]],
    iter_idx: int,
    max_iter: int,
    compact_iter_idx: int,
    fsqr: float,
    fsqz: float,
    fsql: float,
    dt_eff: float,
    update_rms: float,
    time_step: float,
    r00: float,
    z00: float,
    w_mhd: float,
    step_status: str,
    force_vmec2000_row: bool = False,
    compact_status: bool = True,
    print_func: Callable[..., None] = print,
) -> bool:
    """Print either a VMEC2000 row or a compact residual update line."""

    if not bool(verbose):
        return False
    if bool(vmec2000_control) and bool(verbose_vmec2000_table):
        if (not bool(force_vmec2000_row)) and (not should_print_vmec2000(int(iter_idx), int(max_iter))):
            return False
        fsqr1_f, fsqz1_f, fsql1_f = precond_diag_floats()
        print_vmec2000_iter_row(
            iter_idx=int(iter_idx),
            fsqr=fsqr,
            fsqz=fsqz,
            fsql=fsql,
            fsqr1=fsqr1_f,
            fsqz1=fsqz1_f,
            fsql1=fsql1_f,
            delt0r=float(time_step),
            r00=float(r00),
            w_mhd=float(w_mhd),
            z00=float(z00),
        )
        return True

    if not bool(compact_status):
        return False
    fsqr1_f, fsqz1_f, fsql1_f = precond_diag_floats()
    print_func(
        _format_residual_iteration_update_message(
            iter_idx=int(compact_iter_idx),
            dt_eff=dt_eff,
            update_rms=update_rms,
            fsqr1=fsqr1_f,
            fsqz1=fsqz1_f,
            fsql1=fsql1_f,
            step_status=step_status,
        ),
        flush=True,
    )
    return True


__all__ = [
    "Vmec2000PrintContext",
    "VmecIterationScalars",
    "print_compact_converged_status",
    "print_compact_physical_residual_status",
    "print_compact_residual_iteration_update_status",
    "print_residual_iteration_update_status",
    "resolve_vmec2000_print_context",
    "sample_vmec_iteration_scalars",
]
