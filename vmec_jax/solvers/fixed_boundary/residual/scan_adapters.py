"""Small adapter objects for residual-iteration scan solves.

The VMEC2000 scan controller is intentionally kept close to the original
algorithm.  This module owns only the host/JAX plumbing around that controller:
device synchronization for timing, VMEC table print routing, time-control trace
dumps, convergence predicates, and the ``m=1`` preconditioner RHS scaling hook.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from vmec_jax.solvers.fixed_boundary.diagnostics.io import _should_print_vmec2000_row
from vmec_jax.solvers.fixed_boundary.scan.debug import (
    _maybe_dump_timecontrol_scan,
    _print_axis_guess,
    _print_vmec2000_row,
)
from vmec_jax.solvers.fixed_boundary.residual.runtime import (
    _scan_block_until_ready,
    _scan_device_run_ready,
)


@dataclass(frozen=True)
class ScanDeviceRuntime:
    """Device synchronization hooks used by scan timing instrumentation."""

    scan_timing_enabled: bool
    stats: dict[str, Any]
    perf_counter: Callable[[], float]
    block_until_ready: Callable[[Any], Any]
    tree_map: Callable[..., Any]
    record_ready: Callable[..., Any]

    def ready(self, start: float | None, value: Any, *, cache_status: str | None = None):
        """Block scan output when timing is enabled and record device-ready time."""

        return _scan_device_run_ready(
            start=start,
            value=value,
            scan_timing_enabled=bool(self.scan_timing_enabled),
            perf_counter=self.perf_counter,
            block_until_ready=self.block_until_ready,
            tree_map=self.tree_map,
            record_ready=self.record_ready,
            stats=self.stats,
            cache_status=cache_status,
        )

    def block_value(self, value: Any):
        """Synchronize a pytree value with the active JAX backend."""

        return _scan_block_until_ready(
            value,
            block_until_ready=self.block_until_ready,
            tree_map=self.tree_map,
        )


@dataclass(frozen=True)
class ScanVmec2000PrintContext:
    """VMEC2000-style scan print cadence and row emitters."""

    nstep_screen: int
    lasym: bool
    verbose: bool
    vmec2000_control: bool
    verbose_vmec2000_table: bool

    def should_print(self, iter_idx: int, max_iter_local: int) -> bool:
        """Return whether a VMEC2000 row should be emitted for this iteration."""

        return _should_print_vmec2000_row(
            iter_idx=iter_idx,
            max_iter=max_iter_local,
            nstep_screen=int(self.nstep_screen),
            verbose=bool(self.verbose),
            vmec2000_control=bool(self.vmec2000_control),
            verbose_vmec2000_table=bool(self.verbose_vmec2000_table),
        )

    def print_row(
        self,
        *,
        iter_idx: int,
        fsqr: float,
        fsqz: float,
        fsql: float,
        delt0r: float,
        r00: float,
        w_mhd: float,
        z00: float | None = None,
    ) -> None:
        """Print one VMEC2000-style iteration row."""

        _print_vmec2000_row(
            iter_idx=iter_idx,
            fsqr=fsqr,
            fsqz=fsqz,
            fsql=fsql,
            delt0r=delt0r,
            r00=r00,
            w_mhd=w_mhd,
            lasym=bool(self.lasym),
            z00=z00,
            verbose=bool(self.verbose),
            vmec2000_control=bool(self.vmec2000_control),
            verbose_vmec2000_table=bool(self.verbose_vmec2000_table),
        )

    @staticmethod
    def print_axis_guess(raxis_cc: Any, zaxis_cs: Any) -> None:
        """Print an improved magnetic-axis guess with VMEC formatting."""

        _print_axis_guess(raxis_cc, zaxis_cs)


@dataclass(frozen=True)
class ScanTimeControlDumper:
    """JAX-safe time-control trace dumper for VMEC2000 scan solves."""

    enabled: bool
    timecontrol_callback: Any | None
    timecontrol_path: Any | None
    jax_module: Any
    jnp_module: Any

    def __call__(self, *, cond, stage_id, iter2, iter1, fsq, fsq0, res0, res1, time_step, irst):
        """Dump a time-control trace row when the configured condition is true."""

        return _maybe_dump_timecontrol_scan(
            cond=cond,
            stage_id=stage_id,
            iter2=iter2,
            iter1=iter1,
            fsq=fsq,
            fsq0=fsq0,
            res0=res0,
            res1=res1,
            time_step=time_step,
            irst=irst,
            dump_timecontrol_scan=bool(self.enabled),
            timecontrol_callback=self.timecontrol_callback,
            timecontrol_path=self.timecontrol_path,
            jax_module=self.jax_module,
            jnp_module=self.jnp_module,
        )


@dataclass(frozen=True)
class ScanConvergencePredicate:
    """Callable residual convergence predicate with fixed scan tolerances."""

    ftol: Any
    fsq_total_target: Any | None
    converged_func: Callable[..., Any]

    def __call__(self, fsqr, fsqz, fsql):
        return self.converged_func(
            fsqr,
            fsqz,
            fsql,
            ftol=self.ftol,
            fsq_total_target=self.fsq_total_target,
        )


def scan_m1_preconditioner_rhs(
    frzl_in,
    mats: dict[str, Any],
    *,
    cfg: Any,
    scale_m1_precond_rhs_from_mats: Callable[..., Any],
):
    """Apply VMEC's ``m=1`` preconditioner RHS scaling for scan payloads."""

    return scale_m1_precond_rhs_from_mats(
        frzl_in,
        mats,
        lconm1=getattr(cfg, "lconm1", True),
        mpol=int(cfg.mpol),
        host_update_assembly=False,
    )
