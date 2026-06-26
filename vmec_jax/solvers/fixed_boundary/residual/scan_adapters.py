"""Small adapter objects for residual-iteration scan solves.

The VMEC2000 scan controller is intentionally kept close to the original
algorithm.  This module owns only the host/JAX plumbing around that controller:
device synchronization for timing, VMEC table print routing, time-control trace
dumps, convergence predicates, and the ``m=1`` preconditioner RHS scaling hook.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import os
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
        """Evaluate this callable objective for fixed-boundary VMEC solve and implicit differentiation."""
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
        """Evaluate state only scan for fixed-boundary VMEC solve and implicit differentiation."""
        return bool(self.setup.state_only_scan)

    @property
    def scan_fallback_enabled_run(self) -> bool:
        """Evaluate scan fallback enabled run for fixed-boundary VMEC solve and implicit differentiation."""
        return bool(self.setup.scan_fallback_enabled_run)

    @property
    def force_chunked_scan_run(self) -> bool:
        """Evaluate force chunked scan run for fixed-boundary VMEC solve and implicit differentiation."""
        return bool(self.setup.force_chunked_scan_run)

    def maybe_trace(self, label: str):
        """Return the configured scan trace context manager for a label."""

        return self.trace_context_or_null(self.hooks, label)


@dataclass(frozen=True)
class ResidualScanPathHooks:
    """Injected scan hooks that preserve residual-iteration monkeypatch seams."""

    run_vmec2000_scan: Callable[..., Any]
    scan_context_type: Any
    scan_fallback_decision: Callable[..., Any]
    scan_fallback_message: Callable[..., str]
    run_accelerated_scan: Callable[..., Any]
    converged_residuals_func: Callable[..., Any]
    scan_device_ready_recorder: Callable[..., Any]
    get_or_build_scan_runner: Callable[..., Any]
    jit_cache_get: Callable[..., Any]
    jit_cache_put: Callable[..., Any]
    record_scan_runner_cache_miss_categories: Callable[..., Any]
    scan_timing_enabled: Callable[..., Any]
    new_scan_timing_stats: Callable[..., Any]
    build_scan_timing_report: Callable[..., Any]
    runtime_env_enabled: Callable[..., Any]
    scan_backend_name: Callable[..., str]
    scan_chunk_settings: Callable[..., Any]
    tree_has_tracer: Callable[..., bool]
    scan_runner_cache: Any
    enforce_fixed_boundary_and_axis: Callable[..., Any]
    jax_module: Any
    jnp_module: Any
    jit_func: Callable[..., Any]
    perf_counter: Callable[[], float]


@dataclass(frozen=True)
class ResidualScanPathResult:
    """Outcome of trying the scan path before falling back to host iteration."""

    handled: bool
    result: Any | None
    use_scan: bool
    state: Any
    resume_state: dict[str, Any] | None


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


def _dispatch_vmec2000_residual_scan(
    *,
    namespace: dict[str, Any],
    state: Any,
    hooks: ResidualScanPathHooks,
) -> ResidualScanPathResult:
    """Run the VMEC2000 residual scan and decide whether to fall back."""

    startup_policy = namespace["startup_policy"]
    scan_result = hooks.run_vmec2000_scan(
        hooks.scan_context_type.from_namespace(
            namespace,
            _SCAN_RUNNER_CACHE=hooks.scan_runner_cache,
            _runtime_env_enabled=hooks.runtime_env_enabled,
            _scan_backend_name=hooks.scan_backend_name,
            _scan_chunk_settings=hooks.scan_chunk_settings,
            _tree_has_tracer=hooks.tree_has_tracer,
            axis_reset_coeffs=_axis_reset_coeffs_from_namespace(namespace),
        ),
        state,
    )
    attach_freeb_diag = namespace["_attach_freeb_diag"]
    if startup_policy.scan_fallback_enabled and (not bool(namespace["state_only"])):
        fallback_decision = hooks.scan_fallback_decision(
            diagnostics=scan_result.diagnostics,
            fsqr_history=scan_result.fsqr2_history,
            fsqz_history=scan_result.fsqz2_history,
            fsql_history=scan_result.fsql2_history,
            max_iter=int(namespace["max_iter"]),
            fallback_iters=int(startup_policy.scan_fallback_iters),
            badjac_limit=int(startup_policy.scan_fallback_badjac_limit),
            fsq_abs=float(startup_policy.scan_fallback_fsq_abs),
            accept_frac=float(startup_policy.scan_fallback_accept_frac),
            fsq_factor=float(startup_policy.scan_fallback_fsq_factor),
        )
        if fallback_decision.fallback:
            if namespace["verbose"]:
                print(hooks.scan_fallback_message(fallback_decision), flush=True)
            return ResidualScanPathResult(
                handled=False,
                result=None,
                use_scan=False,
                state=namespace["state0"],
                resume_state=None,
            )
    return ResidualScanPathResult(
        handled=True,
        result=attach_freeb_diag(scan_result),
        use_scan=True,
        state=state,
        resume_state=namespace["resume_state"],
    )


def _dispatch_accelerated_residual_scan(
    *,
    namespace: dict[str, Any],
    state: Any,
    hooks: ResidualScanPathHooks,
) -> ResidualScanPathResult:
    """Run the non-VMEC2000 accelerated residual scan."""

    result = hooks.run_accelerated_scan(
        state=state,
        state0=namespace["state0"],
        static=namespace["static"],
        cfg=namespace["cfg"],
        max_iter=int(namespace["max_iter"]),
        step_size=float(namespace["step_size"]),
        initial_flip_sign=float(namespace["initial_flip_sign"]),
        lambda_update_scale=float(namespace["lambda_update_scale"]),
        lambda_update_scale_j=namespace["lambda_update_scale_j"],
        ftol=float(namespace["ftol"]),
        fsq_total_target=namespace["fsq_total_target"],
        precond_radial_alpha=float(namespace["precond_radial_alpha"]),
        precond_lambda_alpha=float(namespace["precond_lambda_alpha"]),
        apply_m1_constraints=bool(namespace["apply_m1_constraints"]),
        jit_forces=bool(namespace["jit_forces"]),
        free_boundary_enabled=bool(namespace["free_boundary_enabled"]),
        static_key=namespace["static_key"],
        wout_key=namespace["wout_key"],
        edge_value_key=namespace["edge_value_key"],
        edge_Rcos=namespace["edge_Rcos"],
        edge_Rsin=namespace["edge_Rsin"],
        edge_Zcos=namespace["edge_Zcos"],
        edge_Zsin=namespace["edge_Zsin"],
        idx00=int(namespace["idx00"]),
        w_mode_mn=namespace["w_mode_mn"],
        mode_context=namespace["_mode_context"],
        compute_forces=namespace["_compute_forces"],
        compute_forces_impl=namespace["_compute_forces_impl"],
        apply_radial_tridi_batched=namespace["_apply_radial_tridi_batched"],
        mn_cos_to_signed_physical=namespace["_mn_cos_to_signed_physical"],
        mn_sin_to_signed_physical=namespace["_mn_sin_to_signed_physical"],
        mn_cos_to_signed_physical_lambda=namespace["_mn_cos_to_signed_physical_lambda"],
        enforce_fixed_boundary_and_axis=hooks.enforce_fixed_boundary_and_axis,
        apply_vmec_lambda_axis_rules=namespace["_apply_vmec_lambda_axis_rules"],
        attach_freeb_diag=namespace["_attach_freeb_diag"],
        scan_timing_env=os.getenv("VMEC_JAX_TIMING", ""),
        jax_module=hooks.jax_module,
        jnp_module=hooks.jnp_module,
        jit_func=hooks.jit_func,
        scan_timing_enabled_func=hooks.scan_timing_enabled,
        new_scan_timing_stats_func=hooks.new_scan_timing_stats,
        build_scan_timing_report_func=hooks.build_scan_timing_report,
        converged_residuals_func=hooks.converged_residuals_func,
        scan_device_ready_recorder=hooks.scan_device_ready_recorder,
        get_or_build_scan_runner_func=hooks.get_or_build_scan_runner,
        scan_runner_cache=hooks.scan_runner_cache,
        jit_cache_get_func=hooks.jit_cache_get,
        jit_cache_put_func=hooks.jit_cache_put,
        record_scan_runner_cache_miss_categories_func=hooks.record_scan_runner_cache_miss_categories,
        perf_counter=hooks.perf_counter,
        differentiating_scan=bool(namespace["startup_policy"].differentiating_scan),
    )
    return ResidualScanPathResult(
        handled=True,
        result=result,
        use_scan=True,
        state=state,
        resume_state=namespace["resume_state"],
    )


def _axis_reset_coeffs_from_namespace(namespace: dict[str, Any]) -> Any:
    """Return the current axis-reset coefficients carried by the reset callback."""

    reset_callback = namespace.get("_reset_axis_from_boundary")
    coeffs_func = getattr(reset_callback, "coeffs", None)
    if callable(coeffs_func):
        return coeffs_func()
    return namespace.get("axis_reset_coeffs")


def dispatch_residual_scan_path(
    *,
    namespace: dict[str, Any],
    state: Any,
    hooks: ResidualScanPathHooks,
) -> ResidualScanPathResult:
    """Try the configured scan path before the host residual loop."""

    if not bool(namespace["use_scan"]):
        return ResidualScanPathResult(
            handled=False,
            result=None,
            use_scan=False,
            state=state,
            resume_state=namespace["resume_state"],
        )
    if namespace["vmec2000_control"]:
        scan_outcome = _dispatch_vmec2000_residual_scan(
            namespace=namespace,
            state=state,
            hooks=hooks,
        )
        if scan_outcome.handled:
            return scan_outcome
        return scan_outcome

    startup_policy = namespace["startup_policy"]
    if (
        namespace["backtracking"]
        or startup_policy.use_restart_triggers
        or startup_policy.auto_flip_force
        or namespace["limit_dt_from_force"]
        or namespace["limit_update_rms"]
        or namespace["strict_update"]
        or namespace["use_direct_fallback"]
        or namespace["reference_mode"]
    ):
        raise ValueError(
            "use_scan requires vmec2000_control=False, backtracking=False, "
            "use_restart_triggers=False, auto_flip_force=False, "
            "limit_dt_from_force=False, limit_update_rms=False, strict_update=False, "
            "use_direct_fallback=False, reference_mode=False."
        )
    return _dispatch_accelerated_residual_scan(
        namespace=namespace,
        state=state,
        hooks=hooks,
    )
