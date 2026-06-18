"""Small adapter objects for residual-iteration scan solves.

The VMEC2000 scan controller is intentionally kept close to the original
algorithm.  This module owns only the host/JAX plumbing around that controller:
device synchronization for timing, VMEC table print routing, time-control trace
dumps, convergence predicates, and the ``m=1`` preconditioner RHS scaling hook.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
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


@dataclass(frozen=True)
class Vmec2000ScanRuntimeSetup:
    """Host-side setup values for the VMEC2000-style scan controller."""

    timing_enabled: bool
    timing_stats: dict[str, Any]
    total_start: float | None
    device_runtime: ScanDeviceRuntime
    setup: Any
    scan_differentiated: bool
    controller_constants: Any
    iter_offset0: int
    nstep_screen: int
    options: Any
    hooks: Any
    dump_timecontrol_scan: bool
    timecontrol_callback: Any | None
    timecontrol_path: Any | None
    io_callback: Any | None
    chunked_print: bool
    print_in_scan: bool
    scan_print_mode: str
    scan_trace: bool
    jax_debug: Any
    jax_debug_print: Any
    axis_reset_enabled: bool
    axis_reset_repeat: bool
    print_context: ScanVmec2000PrintContext
    dtype: Any
    timecontrol_dumper: ScanTimeControlDumper
    time_step0: Any
    flip_sign0: Any
    converged: ScanConvergencePredicate
    resume_fields: Any
    scale_m1_precond_rhs: Any
    jit_forces_scan: bool
    compute_forces_scan: Any
    trace_context_or_null: Any

    @property
    def state_only_scan(self) -> bool:
        return bool(self.setup.state_only_scan)

    @property
    def scan_fallback_enabled_run(self) -> bool:
        return bool(self.setup.scan_fallback_enabled_run)

    @property
    def force_chunked_scan_run(self) -> bool:
        return bool(self.setup.force_chunked_scan_run)

    def maybe_trace(self, label: str):
        """Return the configured scan trace context manager for a label."""

        return self.trace_context_or_null(self.hooks, label)


def build_vmec2000_scan_runtime_setup(
    *,
    env: Any,
    state_init: Any,
    indata: Any,
    cfg: Any,
    mpol: int,
    nrange: int,
    resume_state: dict[str, Any] | None,
    state_only: bool,
    scan_fallback_enabled: bool,
    force_chunked_scan: bool,
    preconditioner_use_precomputed_tridi: bool | None,
    preconditioner_use_lax_tridi: bool | None,
    verbose: bool,
    vmec2000_control: bool,
    verbose_vmec2000_table: bool,
    light_history: bool,
    scan_minimal_default: bool | None,
    dump_any: bool,
    fsq_total_target: float | None,
    axis_reset_done: bool,
    lmove_axis: bool,
    step_size: float,
    initial_flip_sign: float,
    ftol: float,
    jit_forces: bool,
    compute_forces: Any,
    compute_forces_impl: Any,
    scan_timing_enabled_func: Any,
    new_scan_timing_stats_func: Any,
    scan_backend_name_func: Any,
    tree_has_tracer_func: Any,
    validate_vmec2000_scan_guards_func: Any,
    resolve_vmec2000_scan_setup_func: Any,
    default_vmec2000_controller_constants_func: Any,
    resolve_scan_runtime_hooks_from_env_func: Any,
    scan_jit_forces_enabled_func: Any,
    scan_trace_context_or_null_func: Any,
    initialize_scan_resume_state_func: Any,
    scan_m1_preconditioner_rhs_func: Any,
    scale_m1_precond_rhs_from_mats_func: Any,
    converged_func: Any,
    record_scan_device_ready_func: Any,
    has_jax_func: Any,
    jax_module: Any,
    jnp_module: Any,
    time_module: Any,
    backtracking: bool,
    limit_dt_from_force: bool,
    limit_update_rms: bool,
    use_direct_fallback: bool,
    reference_mode: bool,
    strict_update: bool,
    auto_flip_force: bool,
) -> Vmec2000ScanRuntimeSetup:
    """Assemble host/JAX plumbing for a VMEC2000-control scan solve.

    The scan recurrence remains in the residual solver.  This helper only
    resolves environment-driven policy, print hooks, timing, resume fields, and
    the force callable selected for the scan path.
    """

    scan_timing_enabled = scan_timing_enabled_func(env.get("VMEC_JAX_TIMING", ""))
    scan_timing_stats = new_scan_timing_stats_func()
    scan_total_start = time_module.perf_counter() if scan_timing_enabled else None
    scan_device_runtime = ScanDeviceRuntime(
        scan_timing_enabled=bool(scan_timing_enabled),
        stats=scan_timing_stats,
        perf_counter=time_module.perf_counter,
        block_until_ready=jax_module.block_until_ready,
        tree_map=jax_module.tree_util.tree_map,
        record_ready=record_scan_device_ready_func,
    )

    validate_vmec2000_scan_guards_func(
        backtracking=bool(backtracking),
        limit_dt_from_force=bool(limit_dt_from_force),
        limit_update_rms=bool(limit_update_rms),
        use_direct_fallback=bool(use_direct_fallback),
        reference_mode=bool(reference_mode),
        strict_update=bool(strict_update),
        auto_flip_force=bool(auto_flip_force),
    )

    scan_differentiated = tree_has_tracer_func(state_init)
    scan_setup = resolve_vmec2000_scan_setup_func(
        env=env,
        state_only=bool(state_only),
        scan_differentiated=bool(scan_differentiated),
        scan_fallback_enabled=bool(scan_fallback_enabled),
        force_chunked_scan=bool(force_chunked_scan),
        indata_nstep=int(indata.get_int("NSTEP", 1)) if indata is not None else 1,
        preconditioner_use_precomputed_tridi=preconditioner_use_precomputed_tridi,
        preconditioner_use_lax_tridi=preconditioner_use_lax_tridi,
        verbose=bool(verbose),
        vmec2000_control=bool(vmec2000_control),
        verbose_vmec2000_table=bool(verbose_vmec2000_table),
        light_history=bool(light_history),
        scan_minimal_default=scan_minimal_default,
        dump_any=bool(dump_any),
        fsq_total_target=fsq_total_target,
        backend_name=scan_backend_name_func(),
    )
    controller_constants = default_vmec2000_controller_constants_func()
    iter_offset0 = 0
    scan_options = scan_setup.options
    scan_hooks = resolve_scan_runtime_hooks_from_env_func(
        env,
        print_in_scan=scan_options.print_in_scan,
        scan_print_mode=scan_options.scan_print_mode,
        scan_trace=scan_options.scan_trace,
    )
    if resume_state is not None:
        try:
            iter_offset0 = int(resume_state.get("iter_offset", iter_offset0))
        except Exception:
            pass

    axis_reset_enabled = bool(vmec2000_control) and (not axis_reset_done) and bool(lmove_axis)
    scan_print_context = ScanVmec2000PrintContext(
        nstep_screen=int(scan_setup.nstep_screen),
        lasym=bool(cfg.lasym),
        verbose=bool(verbose),
        vmec2000_control=bool(vmec2000_control),
        verbose_vmec2000_table=bool(verbose_vmec2000_table),
    )

    dtype = jnp_module.asarray(state_init.Rcos).dtype
    scan_timecontrol_dumper = ScanTimeControlDumper(
        enabled=bool(scan_hooks.dump_timecontrol_scan),
        timecontrol_callback=scan_hooks.timecontrol_callback,
        timecontrol_path=scan_hooks.timecontrol_path,
        jax_module=jax_module,
        jnp_module=jnp_module,
    )
    time_step0 = jnp_module.asarray(float(step_size), dtype=dtype)
    flip_sign0 = jnp_module.asarray(float(initial_flip_sign), dtype=dtype)
    ftol_j = jnp_module.asarray(float(ftol), dtype=dtype)
    fsq_total_target_j = None
    if fsq_total_target is not None:
        fsq_total_target_j = jnp_module.asarray(float(fsq_total_target), dtype=dtype)

    scan_converged = ScanConvergencePredicate(
        ftol=ftol_j,
        fsq_total_target=fsq_total_target_j,
        converged_func=converged_func,
    )
    scan_resume0 = initialize_scan_resume_state_func(
        resume_state,
        dtype=dtype,
        velocity_shape=(int(state_init.Rcos.shape[0]), int(mpol), int(nrange)),
        k_ndamp=controller_constants.ndamp,
        time_step_default=time_step0,
        flip_sign_default=flip_sign0,
        state_checkpoint_default=state_init,
    )
    flip_sign0 = scan_resume0.flip_sign
    scale_m1_precond_rhs = partial(
        scan_m1_preconditioner_rhs_func,
        cfg=cfg,
        scale_m1_precond_rhs_from_mats=scale_m1_precond_rhs_from_mats_func,
    )

    scan_jit_env = env.get("VMEC_JAX_SCAN_JIT_FORCES")
    jit_forces_scan = scan_jit_forces_enabled_func(env_value=scan_jit_env, jit_forces=bool(jit_forces))
    compute_forces_scan = compute_forces if jit_forces_scan else compute_forces_impl
    if scan_timing_enabled and scan_total_start is not None:
        scan_timing_stats["scan_setup_s"] += time_module.perf_counter() - float(scan_total_start)

    return Vmec2000ScanRuntimeSetup(
        timing_enabled=bool(scan_timing_enabled),
        timing_stats=scan_timing_stats,
        total_start=scan_total_start,
        device_runtime=scan_device_runtime,
        setup=scan_setup,
        scan_differentiated=bool(scan_differentiated),
        controller_constants=controller_constants,
        iter_offset0=int(iter_offset0),
        nstep_screen=int(scan_setup.nstep_screen),
        options=scan_options,
        hooks=scan_hooks,
        dump_timecontrol_scan=bool(scan_hooks.dump_timecontrol_scan),
        timecontrol_callback=scan_hooks.timecontrol_callback,
        timecontrol_path=scan_hooks.timecontrol_path,
        io_callback=scan_hooks.io_callback,
        chunked_print=bool(scan_options.chunked_print),
        print_in_scan=bool(scan_hooks.print_in_scan),
        scan_print_mode=scan_hooks.scan_print_mode,
        scan_trace=bool(scan_hooks.scan_trace),
        jax_debug=scan_hooks.jax_debug,
        jax_debug_print=scan_hooks.jax_debug_print,
        axis_reset_enabled=bool(axis_reset_enabled),
        axis_reset_repeat=False,
        print_context=scan_print_context,
        dtype=dtype,
        timecontrol_dumper=scan_timecontrol_dumper,
        time_step0=time_step0,
        flip_sign0=flip_sign0,
        converged=scan_converged,
        resume_fields=scan_resume0,
        scale_m1_precond_rhs=scale_m1_precond_rhs,
        jit_forces_scan=bool(jit_forces_scan),
        compute_forces_scan=compute_forces_scan,
        trace_context_or_null=scan_trace_context_or_null_func,
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
