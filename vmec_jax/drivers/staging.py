"""Staged fixed-boundary driver runners.

These helpers implement CLI continuation policies that repeatedly call the
public fixed-boundary driver on coarser/finer radial grids.  They deliberately
receive the recursive driver call, interpolation helper, timing reducer, and
finish callback as dependencies so :mod:`vmec_jax.driver` keeps its historical
monkeypatch seams while the long public workflow remains readable.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import time
from typing import Any, Callable

import numpy as np

from vmec_jax._solve_runtime import _dataclass_from_namespace


@dataclass(frozen=True)
class FixedBoundaryStageRunnerContext:
    """Shared inputs for CLI staged fixed-boundary continuation runners."""

    input_path: Any
    cfg: Any
    step_size: Any
    history_size: int
    gn_damping: Any
    gn_cg_tol: Any
    gn_cg_maxiter: int
    vmec_project: bool
    use_restart_triggers: Any
    vmecpp_restart: bool
    use_direct_fallback: Any
    free_boundary_edge_control_projection: Any
    verbose: bool
    jit_forces: Any
    jit_precompile: Any
    use_scan: bool
    scan_wout_corrector: Any
    stage_transition_heuristic: Any
    stage_transition_factor: float
    stage_transition_scale: float
    grid: Any
    solver_mode_eff: str
    cli_fixed_boundary_finish_enabled: bool
    run_fixed_boundary: Callable[..., Any]
    interp_vmec_state: Callable[..., Any]
    maybe_finish_cli_fixed_boundary_run: Callable[..., Any]
    sanitize_resume_state_for_stage: Callable[[Any], Any]
    timing_solve_total_s: Callable[[dict[str, Any]], float]
    accelerated_cli_budgeted_stage_iters: Callable[..., list[int]]

    @classmethod
    def from_namespace(cls, namespace: dict[str, Any], /, **overrides: Any) -> "FixedBoundaryStageRunnerContext":
        return _dataclass_from_namespace(cls, namespace, label="fixed-boundary stage-runner", overrides=overrides)


@dataclass(frozen=True)
class Vmec2000StagedSolveContext:
    """Inputs and hooks for the public VMEC2000-style staged solve path."""

    input_path: Any
    cfg: Any
    indata: Any
    solver: str
    solver_mode_eff: str
    accelerated_mode: bool
    performance_mode: bool
    use_scan: bool
    cli_fixed_boundary_mode: bool
    direct_staged_current_driven_3d_cli: bool
    multigrid: bool
    multigrid_user_provided: bool
    accelerated_single_grid_default: bool
    direct_external_provider: bool
    policy_backend: str
    stage_transition_heuristic: bool
    stage_transition_factor: float
    stage_transition_scale: float
    scan_minimal_default: Any
    scan_wout_corrector: Any
    jit_forces: Any
    jit_precompile: Any
    precompile_stages: bool
    use_restart_triggers: Any
    vmecpp_restart: bool
    limit_update_rms: Any
    light_history: Any
    external_field_provider_kind: Any
    external_field_provider_static: Any
    external_field_provider_params: Any
    free_boundary_activate_fsq: Any
    free_boundary_edge_control_projection: Any
    verbose: bool
    signgs: int
    step_size_val: float
    ns_stages: list[int]
    niter_stages: list[int]
    ftol_stages: list[float]
    niter_stages_input: Any
    ftol_list_input: Any
    restart_state_eff: Any
    restart_solver_state: Any
    boundary_coeffs: Any
    t_start: float
    build_static_cfg: Callable[[Any], Any]
    initial_guess_with_optional_nojit: Callable[..., Any]
    resolve_jit_forces: Callable[..., bool]
    sanitize_resume_state_for_stage: Callable[[Any], Any]
    sanitize_resume_state_for_same_stage: Callable[[Any], Any]
    interp_vmec_state: Callable[..., Any]
    mode_table_func: Callable[..., Any]
    maybe_dump_xc_init: Callable[..., None]
    maybe_disable_scan_by_parity_guard: Callable[..., bool]
    resolve_stage_jit_settings: Callable[..., Any]
    accelerated_fsq_total_target_from_ftol: Callable[[float], float]
    host_update_assembly_driver_default: Callable[..., bool]
    default_preconditioner_use_precomputed_tridi: Callable[..., bool]
    default_preconditioner_use_lax_tridi: Callable[..., bool]
    solve_fixed_boundary_residual_iter: Callable[..., Any]
    maybe_select_dynamic_scan_mode: Callable[..., bool]
    dynamic_scan_probe_settings: Callable[..., Any]
    vmec_histories_match: Callable[..., bool]
    vmec_history_relerr: Callable[..., float]
    maybe_precompile_fixed_boundary_stage: Callable[..., None]
    run_fixed_boundary_stage_solve: Callable[..., Any]
    result_meets_requested_ftol: Callable[..., bool]
    stage_switch_reason_from_progress: Callable[..., str | None]
    merge_stage_chunk_results: Callable[..., Any]
    result_with_diag: Callable[..., Any]
    maybe_rerun_scan_abort_stage: Callable[..., Any]
    assemble_multigrid_stage_result: Callable[..., Any]
    maybe_apply_scan_wout_corrector: Callable[..., Any]
    copy_final_force_payload: Callable[..., Any]
    timing_solve_total_s: Callable[[dict[str, Any]], float]
    requested_final_ftol: Callable[..., float]
    result_final_residuals: Callable[..., Any]
    result_hits_total_target: Callable[..., bool]
    finalize_fixed_boundary_convergence_result: Callable[..., Any]
    print_vmec2000_run_summary: Callable[..., None]
    default_backend_name: Callable[[], str]
    deepcopy_func: Callable[[Any], Any]
    getenv: Callable[[str, str], str]
    perf_counter: Callable[[], float] = time.perf_counter

    @classmethod
    def from_namespace(cls, namespace: dict[str, Any], /, **overrides: Any) -> "Vmec2000StagedSolveContext":
        return _dataclass_from_namespace(cls, namespace, label="VMEC2000 staged-solve", overrides=overrides)


@dataclass(frozen=True)
class Vmec2000StagedSolveResult:
    """Result bundle from the VMEC2000 staged solve driver seam."""

    result: Any
    static: Any
    stage_results: list[Any]
    stage_statics: list[Any]


@dataclass(frozen=True)
class Vmec2000StageSolvePlan:
    """Resolved controls for one VMEC2000 stage solve."""

    scan_mode: bool
    solve_kwargs: dict[str, Any]
    jit_forces_eff: bool
    jit_forces_base: bool
    jit_precompile_noscan: bool
    jit_warmup_noscan: int
    explicit_stage_monitor: bool
    explicit_stage_chunk: int
    explicit_stage_target: float
    explicit_stage_monitor_jit_forces: bool


def _stage_timing_s(run: Any, timing_solve_total_s: Callable[[dict[str, Any]], float]) -> float:
    try:
        timing = run.result.diagnostics.get("timing", {}) if run.result is not None else {}
    except Exception:
        timing = {}
    return float(timing_solve_total_s(timing))


def _append_stage_summary(
    *,
    run: Any,
    mode: str,
    elapsed_s: float,
    stage_runs: list[Any],
    stage_modes: list[str],
    stage_wall_s: list[float],
    stage_solve_total_s: list[float],
    timing_solve_total_s: Callable[[dict[str, Any]], float],
) -> None:
    stage_wall_s.append(float(elapsed_s))
    stage_solve_total_s.append(_stage_timing_s(run, timing_solve_total_s))
    stage_runs.append(run)
    stage_modes.append(str(mode))


def run_stage_with_optional_explicit_monitor(
    *,
    monitor_enabled: bool,
    stage_mode: str,
    ns: int,
    niter: int,
    ftol: float,
    explicit_stage_chunk: int,
    explicit_stage_target: float,
    policy_backend: str,
    scan_mode: bool,
    state: Any,
    state_stage_start: Any,
    resume_state_stage: Any,
    stage_prev_fsq: float | None,
    solve_kwargs: dict[str, Any],
    jit_forces_eff: bool,
    jit_forces_base: bool,
    explicit_stage_monitor_jit_forces: bool,
    jit_warmup_noscan: int,
    jit_precompile_noscan: bool,
    run_stage_solve: Callable[..., Any],
    sanitize_resume_state_for_same_stage: Callable[[Any], Any],
    result_meets_requested_ftol: Callable[..., bool],
    stage_switch_reason_from_progress: Callable[..., str | None],
    merge_stage_chunk_results: Callable[..., Any],
    result_with_diag: Callable[..., Any],
    maybe_rerun_scan_abort_stage: Callable[..., Any],
    scan_abort_fallback_enabled: bool,
    verbose: bool,
    print_func: Callable[..., None] = print,
) -> tuple[Any, str]:
    """Run one stage, optionally with accelerated explicit-stage monitoring."""

    effective_mode = str(stage_mode)
    if bool(monitor_enabled) and int(explicit_stage_chunk) < int(niter):
        chunk_results: list[Any] = []
        chunk_state = state
        chunk_resume_state = resume_state_stage
        stage_switch_reason = None
        stage_monitor_used = True
        stage_monitor_scan = bool(scan_mode) and str(policy_backend).lower() in (
            "gpu",
            "cuda",
            "rocm",
            "tpu",
        )
        remaining_budget = int(niter)
        stage_first_chunk = True

        chunk_budget = min(int(explicit_stage_chunk), int(remaining_budget))
        chunk_kwargs = dict(solve_kwargs)
        chunk_kwargs.update(
            {
                "max_iter": int(chunk_budget),
                "resume_state": chunk_resume_state,
                "stage_prev_fsq": stage_prev_fsq if bool(stage_first_chunk) else None,
                "use_scan": bool(stage_monitor_scan),
                "jit_warmup_iters": int(jit_warmup_noscan),
                "jit_precompile": bool(jit_precompile_noscan),
            }
        )
        res_chunk = run_stage_solve(
            state=chunk_state,
            solve_kwargs=chunk_kwargs,
            jit_forces=bool(explicit_stage_monitor_jit_forces),
        )
        chunk_results.append(res_chunk)
        stage_first_chunk = False

        completed_chunk_iters = min(int(chunk_budget), int(res_chunk.n_iter) + 1)
        remaining_budget = max(0, int(remaining_budget) - int(completed_chunk_iters))
        chunk_state = res_chunk.state
        chunk_resume_state = sanitize_resume_state_for_same_stage(res_chunk.diagnostics.get("resume_state"))

        strict_chunk = bool(result_meets_requested_ftol(res_chunk, ftol=float(ftol)))
        if (not bool(strict_chunk)) and int(remaining_budget) > 0:
            try:
                chunk_w = np.asarray(res_chunk.w_history, dtype=float).reshape(-1)
                if chunk_w.size > 0:
                    stage_switch_reason = stage_switch_reason_from_progress(
                        start_total_fsq=float(chunk_w[0]),
                        best_total_fsq=float(np.min(chunk_w)),
                        target_total_fsq=float(explicit_stage_target),
                        chunk_iters=int(completed_chunk_iters),
                        remaining_budget=int(remaining_budget),
                    )
            except Exception:
                stage_switch_reason = None

        if (stage_switch_reason is None) and (not bool(strict_chunk)) and int(remaining_budget) > 0:
            tail_kwargs = dict(solve_kwargs)
            tail_kwargs.update(
                {
                    "max_iter": int(remaining_budget),
                    "resume_state": chunk_resume_state,
                    "stage_prev_fsq": None,
                    "use_scan": bool(stage_monitor_scan),
                    "jit_warmup_iters": 0,
                    "jit_precompile": False,
                }
            )
            res_tail = run_stage_solve(
                state=chunk_state,
                solve_kwargs=tail_kwargs,
                jit_forces=bool(explicit_stage_monitor_jit_forces),
            )
            chunk_results.append(res_tail)
        elif (stage_switch_reason is None) and (not bool(strict_chunk)):
            stage_switch_reason = "budget_exhausted"

        if stage_switch_reason is not None:
            if bool(verbose):
                print_func(
                    "[vmec_jax] accelerated staged solve cannot meet requested FTOL; "
                    f"switching stage ns={int(ns)} to parity mode ({stage_switch_reason}).",
                    flush=True,
                )
            fallback_kwargs = dict(solve_kwargs)
            fallback_kwargs.update(
                {
                    "use_scan": False,
                    "resume_state": resume_state_stage,
                    "max_iter": int(niter),
                    "jit_warmup_iters": int(jit_warmup_noscan),
                    "jit_precompile": bool(jit_precompile_noscan),
                    "light_history": None,
                    "resume_state_mode": None,
                    "fsq_total_target": None,
                    "host_update_assembly": False,
                }
            )
            result = run_stage_solve(
                state=state_stage_start,
                solve_kwargs=fallback_kwargs,
                jit_forces=bool(jit_forces_base),
            )
            result = result_with_diag(
                result,
                accelerated_stage_chunked=bool(stage_monitor_used or len(chunk_results) > 0),
                accelerated_stage_early_switch=True,
                accelerated_stage_switch_reason=str(stage_switch_reason),
                accelerated_stage_probe_chunk_iters=np.asarray(
                    [int(r.n_iter) + 1 for r in chunk_results],
                    dtype=int,
                ),
                accelerated_stage_effective_mode="parity",
            )
            return result, "parity"
        return merge_stage_chunk_results(chunk_results, mode_i=effective_mode), effective_mode

    result = run_stage_solve(
        state=state,
        solve_kwargs=solve_kwargs,
        jit_forces=bool(jit_forces_eff),
    )
    result = maybe_rerun_scan_abort_stage(
        result=result,
        enabled=bool(scan_abort_fallback_enabled),
        state_stage_start=state_stage_start,
        resume_state_stage=resume_state_stage,
        solve_kwargs=solve_kwargs,
        jit_warmup_noscan=int(jit_warmup_noscan),
        jit_precompile_noscan=bool(jit_precompile_noscan),
        jit_forces_base=bool(jit_forces_base),
        run_stage_solve_func=run_stage_solve,
        verbose=bool(verbose),
    )
    return result, effective_mode


def _build_vmec2000_stage_solve_plan(
    ctx: Vmec2000StagedSolveContext,
    *,
    stage_index: int,
    nstep: int,
    niter: int,
    ftol: float,
    static_i: Any,
    state: Any,
    state_stage_start: Any,
    resume_state_stage: Any,
    stage_accelerated_mode: bool,
    scan_mode: bool,
    jit_forces_base: bool,
    jit_forces_eff: bool,
    jit_precompile_eff: bool,
    jit_warmup_iters: int,
    jit_precompile_noscan: bool,
    jit_warmup_noscan: int,
    stage_prev_fsq: float | None,
    is_last_stage: bool,
) -> Vmec2000StageSolvePlan:
    """Build one stage's scan policy, solve kwargs, and precompile hook."""

    final_cpu_scan_env = ctx.getenv("VMEC_JAX_FINAL_STAGE_CPU_SCAN", "1").strip().lower()
    final_cpu_scan_disabled = final_cpu_scan_env in ("0", "false", "no")
    if bool(ctx.cli_fixed_boundary_mode) and scan_mode and (ctx.default_backend_name() == "cpu") and final_cpu_scan_disabled:
        scan_mode = False
    stage_fsq_total_target = (
        ctx.accelerated_fsq_total_target_from_ftol(float(ftol))
        if (stage_accelerated_mode and not is_last_stage)
        else None
    )
    stage_light_history = (
        ctx.light_history
        if ctx.light_history is not None
        else (
            True
            if (
                bool(ctx.performance_mode)
                and (not bool(ctx.verbose))
                and ((not bool(ctx.cfg.lfreeb)) or bool(ctx.direct_external_provider))
            )
            else None
        )
    )
    stage_host_update_assembly = ctx.host_update_assembly_driver_default(
        cfg=static_i.cfg,
        performance_mode=bool(ctx.performance_mode),
        backend=ctx.default_backend_name(),
        use_scan=bool(scan_mode),
    )
    stage_preconditioner_use_precomputed_tridi = ctx.default_preconditioner_use_precomputed_tridi(
        cfg=static_i.cfg,
        backend=ctx.policy_backend,
        performance_mode=bool(ctx.performance_mode),
        use_scan=bool(scan_mode),
        direct_external_provider=bool(ctx.direct_external_provider),
    )
    stage_preconditioner_use_lax_tridi = ctx.default_preconditioner_use_lax_tridi(
        cfg=static_i.cfg,
        backend=ctx.policy_backend,
        performance_mode=bool(ctx.performance_mode),
        use_scan=bool(scan_mode),
        direct_external_provider=bool(ctx.direct_external_provider),
    )
    stage_limit_update_rms = False if ctx.limit_update_rms is None else bool(ctx.limit_update_rms)
    solve_kwargs = dict(
        indata=ctx.indata,
        signgs=ctx.signgs,
        ftol=float(ftol),
        max_iter=int(niter),
        step_size=float(ctx.step_size_val),
        include_constraint_force=True,
        apply_m1_constraints=True,
        precond_radial_alpha=0.5,
        precond_lambda_alpha=0.5,
        mode_diag_exponent=0.0,
        auto_flip_force=False,
        divide_by_scalxc_for_update=False,
        lambda_update_scale=1.0,
        enforce_vmec_lambda_axis=True,
        vmec2000_control=True,
        strict_update=True,
        backtracking=False,
        limit_update_rms=stage_limit_update_rms,
        reference_mode=False,
        use_restart_triggers=True if ctx.use_restart_triggers is None else bool(ctx.use_restart_triggers),
        vmecpp_restart=bool(ctx.vmecpp_restart),
        use_direct_fallback=False,
        stage_prev_fsq=stage_prev_fsq,
        stage_transition_factor=float(ctx.stage_transition_factor),
        stage_transition_scale=float(ctx.stage_transition_scale),
        resume_state=resume_state_stage,
        verbose=bool(ctx.verbose),
        verbose_vmec2000_table=bool(ctx.verbose),
        use_scan=bool(scan_mode),
        jit_warmup_iters=int(jit_warmup_iters),
        jit_precompile=bool(jit_precompile_eff),
        scan_minimal_default=ctx.scan_minimal_default,
        light_history=stage_light_history,
        resume_state_mode="minimal" if stage_accelerated_mode else None,
        fsq_total_target=stage_fsq_total_target,
        host_update_assembly=stage_host_update_assembly,
        preconditioner_use_precomputed_tridi=stage_preconditioner_use_precomputed_tridi,
        preconditioner_use_lax_tridi=stage_preconditioner_use_lax_tridi,
        external_field_provider_kind=ctx.external_field_provider_kind,
        external_field_provider_static=ctx.external_field_provider_static,
        external_field_provider_params=ctx.external_field_provider_params,
        free_boundary_activate_fsq=ctx.free_boundary_activate_fsq,
        free_boundary_edge_control_projection=ctx.free_boundary_edge_control_projection,
        return_final_force_payload=True,
    )
    scan_mode = ctx.maybe_select_dynamic_scan_mode(
        cfg=ctx.cfg,
        accelerated_mode=bool(ctx.accelerated_mode),
        performance_mode=bool(ctx.performance_mode),
        scan_mode=bool(scan_mode),
        vmec2000_control=True,
        niter=int(niter),
        solve_kwargs=solve_kwargs,
        state_stage_start=state_stage_start,
        static_stage=static_i,
        resume_state_stage=resume_state_stage,
        jit_forces_base=bool(jit_forces_base),
        solve_fixed_boundary_residual_iter=ctx.solve_fixed_boundary_residual_iter,
        dynamic_scan_probe_settings=ctx.dynamic_scan_probe_settings,
        vmec_histories_match=ctx.vmec_histories_match,
        vmec_history_relerr=ctx.vmec_history_relerr,
        verbose=bool(ctx.verbose),
        getenv=ctx.getenv,
        deepcopy_func=ctx.deepcopy_func,
    )
    solve_kwargs["use_scan"] = bool(scan_mode)
    ctx.maybe_precompile_fixed_boundary_stage(
        enabled=bool(ctx.precompile_stages) and bool(jit_forces_eff),
        state=state,
        static=static_i,
        solve_kwargs=solve_kwargs,
        solve_fixed_boundary_residual_iter_func=ctx.solve_fixed_boundary_residual_iter,
    )
    explicit_stage_monitor = (
        bool(stage_accelerated_mode)
        and (ctx.niter_stages_input is not None)
        and int(nstep) > 1
        and int(stage_index) > 0
    )
    explicit_stage_chunk = min(int(niter), max(int(ctx.indata.get_int("NSTEP", 1)), 200))
    explicit_stage_target = ctx.accelerated_fsq_total_target_from_ftol(float(ftol))
    return Vmec2000StageSolvePlan(
        scan_mode=bool(scan_mode),
        solve_kwargs=solve_kwargs,
        jit_forces_eff=bool(jit_forces_eff),
        jit_forces_base=bool(jit_forces_base),
        jit_precompile_noscan=bool(jit_precompile_noscan),
        jit_warmup_noscan=int(jit_warmup_noscan),
        explicit_stage_monitor=bool(explicit_stage_monitor),
        explicit_stage_chunk=int(explicit_stage_chunk),
        explicit_stage_target=float(explicit_stage_target),
        explicit_stage_monitor_jit_forces=bool(jit_forces_base),
    )


def run_vmec2000_staged_solve(ctx: Vmec2000StagedSolveContext) -> Vmec2000StagedSolveResult:
    """Run the multigrid/single-grid VMEC2000-style staged solve path."""

    nstep = len(ctx.ns_stages)
    stage_results: list[Any] = []
    stage_statics: list[Any] = []
    stage_offsets: list[int] = []

    header_modes = ctx.mode_table_func(ctx.cfg.mpol, ctx.cfg.ntor)
    nmodes_header = int(np.asarray(header_modes.m).size)

    state = ctx.restart_state_eff
    static_prev = None
    resume_state_stage = ctx.restart_solver_state
    multigrid_resume = False
    if ctx.multigrid:
        env_resume = ctx.getenv("VMEC_JAX_MULTIGRID_RESUME", "0")
        multigrid_resume = env_resume.strip().lower() not in ("", "0", "false", "no")

    prev_stage_fsq = None
    stage_mode_history: list[str] = []
    stage_wall_s: list[float] = []
    stage_solve_total_s: list[float] = []
    ftol_last = None
    step_size_last = None
    last_niter_stage = 0

    for i, (ns_i, niter_i, ftol_i) in enumerate(zip(ctx.ns_stages, ctx.niter_stages, ctx.ftol_stages)):
        if int(niter_i) <= 0:
            continue
        last_niter_stage = int(niter_i)
        stage_t0 = ctx.perf_counter()
        stage_accelerated_mode = bool(ctx.accelerated_mode)
        if bool(stage_accelerated_mode) and bool(ctx.direct_staged_current_driven_3d_cli) and bool(ctx.cfg.lasym):
            # LASYM current-driven 3D staged runs remain more sensitive in
            # lambda than geometry. Keep this class on the conservative
            # controller until the lambda mismatch is closed numerically.
            stage_accelerated_mode = False
        stage_mode_i = "accelerated" if bool(stage_accelerated_mode) else "parity"
        stage_mode_history.append(stage_mode_i)
        if ctx.verbose:
            print(
                f"  NS = {int(ns_i):4d} NO. FOURIER MODES = {nmodes_header:4d} "
                f"FTOLV = {float(ftol_i):10.3E} NITER = {int(niter_i):6d}",
                flush=True,
            )
            print("  PROCESSOR COUNT - RADIAL:    1", flush=True)
            print("", flush=True)
            if bool(ctx.cfg.lasym):
                print(
                    "  ITER    FSQR      FSQZ      FSQL    RAX(v=0)  ZAX(v=0)    DELT       WMHD",
                    flush=True,
                )
            else:
                print(
                    "  ITER    FSQR      FSQZ      FSQL    RAX(v=0)    DELT       WMHD",
                    flush=True,
                )

        cfg_i = replace(ctx.cfg, ns=int(ns_i))
        static_i = ctx.build_static_cfg(cfg_i)
        scan_mode = bool(ctx.use_scan) if bool(stage_accelerated_mode) else False
        if stage_accelerated_mode and bool(ctx.use_scan):
            scan_mode = not bool(cfg_i.lfreeb)
        if bool(ctx.cfg.lasym):
            lasym_scan_env = ctx.getenv("VMEC_JAX_LASYM_USE_SCAN", "auto").strip().lower()
            if lasym_scan_env in ("0", "false", "no", "off"):
                scan_mode = False
            elif lasym_scan_env not in ("", "auto"):
                scan_mode = True
        scan_mode = ctx.maybe_disable_scan_by_parity_guard(
            accelerated_mode=bool(ctx.accelerated_mode),
            scan_mode=bool(scan_mode),
            niter=int(niter_i),
            state_stage_start=state,
            static_stage=static_i,
            indata=ctx.indata,
            signgs=ctx.signgs,
            ftol=float(ftol_i),
            step_size=float(ctx.step_size_val),
            use_restart_triggers=ctx.use_restart_triggers,
            vmecpp_restart=bool(ctx.vmecpp_restart),
            stage_transition_factor=float(ctx.stage_transition_factor),
            stage_transition_scale=float(ctx.stage_transition_scale),
            scan_minimal_default=ctx.scan_minimal_default,
            jit_forces=ctx.jit_forces,
            resolve_jit_forces=ctx.resolve_jit_forces,
            solve_fixed_boundary_residual_iter=ctx.solve_fixed_boundary_residual_iter,
            verbose=bool(ctx.verbose),
            getenv=ctx.getenv,
        )
        jit_forces_base = ctx.resolve_jit_forces(ctx.jit_forces, static_i, int(niter_i))
        jit_settings = ctx.resolve_stage_jit_settings(
            jit_forces_base=bool(jit_forces_base),
            scan_mode=bool(scan_mode),
            solver=ctx.solver,
            performance_mode=bool(ctx.performance_mode),
            jit_precompile=ctx.jit_precompile,
        )
        jit_forces_eff = bool(jit_settings.jit_forces_eff)
        jit_precompile_eff = bool(jit_settings.jit_precompile_eff)
        jit_warmup_iters = int(jit_settings.jit_warmup_iters)
        jit_precompile_noscan = bool(jit_settings.jit_precompile_noscan)
        jit_warmup_noscan = int(jit_settings.jit_warmup_noscan)
        if i == 0:
            if state is None:
                if ctx.boundary_coeffs is None:
                    raise ValueError("boundary_coeffs missing; cannot build initial guess")
                state = ctx.initial_guess_with_optional_nojit(
                    static_i,
                    ctx.boundary_coeffs,
                    force_disable_jit=bool(jit_warmup_iters > 0),
                )
                ctx.maybe_dump_xc_init(state=state, static=static_i, label="stage0")
        else:
            state = ctx.interp_vmec_state(
                state,
                m=static_prev.modes.m,
                n=static_prev.modes.n,
                lthreed=bool(static_prev.cfg.lthreed),
                lconm1=bool(getattr(static_prev.cfg, "lconm1", True)),
                ns_new=int(ns_i),
            )
        state_stage_start = state
        static_prev = static_i

        stage_offsets.append(sum(int(np.asarray(r.w_history).size) for r in stage_results))
        stage_prev_fsq = prev_stage_fsq if bool(ctx.stage_transition_heuristic) else None
        is_last_stage = i == len(ctx.ns_stages) - 1
        stage_plan = _build_vmec2000_stage_solve_plan(
            ctx,
            stage_index=int(i),
            nstep=int(nstep),
            niter=int(niter_i),
            ftol=float(ftol_i),
            static_i=static_i,
            state=state,
            state_stage_start=state_stage_start,
            resume_state_stage=resume_state_stage,
            stage_accelerated_mode=bool(stage_accelerated_mode),
            scan_mode=bool(scan_mode),
            jit_forces_base=bool(jit_forces_base),
            jit_forces_eff=bool(jit_forces_eff),
            jit_precompile_eff=bool(jit_precompile_eff),
            jit_warmup_iters=int(jit_warmup_iters),
            jit_precompile_noscan=bool(jit_precompile_noscan),
            jit_warmup_noscan=int(jit_warmup_noscan),
            stage_prev_fsq=stage_prev_fsq,
            is_last_stage=bool(is_last_stage),
        )

        def run_stage_solve(*, state, solve_kwargs, jit_forces):
            return ctx.run_fixed_boundary_stage_solve(
                state=state,
                static=static_i,
                solve_kwargs=solve_kwargs,
                jit_forces=jit_forces,
                solve_fixed_boundary_residual_iter_func=ctx.solve_fixed_boundary_residual_iter,
            )

        res_i, stage_mode_i = run_stage_with_optional_explicit_monitor(
            monitor_enabled=bool(stage_plan.explicit_stage_monitor),
            stage_mode=str(stage_mode_i),
            ns=int(ns_i),
            niter=int(niter_i),
            ftol=float(ftol_i),
            explicit_stage_chunk=int(stage_plan.explicit_stage_chunk),
            explicit_stage_target=float(stage_plan.explicit_stage_target),
            policy_backend=str(ctx.policy_backend),
            scan_mode=bool(stage_plan.scan_mode),
            state=state,
            state_stage_start=state_stage_start,
            resume_state_stage=resume_state_stage,
            stage_prev_fsq=stage_prev_fsq,
            solve_kwargs=stage_plan.solve_kwargs,
            jit_forces_eff=bool(stage_plan.jit_forces_eff),
            jit_forces_base=bool(stage_plan.jit_forces_base),
            explicit_stage_monitor_jit_forces=bool(stage_plan.explicit_stage_monitor_jit_forces),
            jit_warmup_noscan=int(stage_plan.jit_warmup_noscan),
            jit_precompile_noscan=bool(stage_plan.jit_precompile_noscan),
            run_stage_solve=run_stage_solve,
            sanitize_resume_state_for_same_stage=ctx.sanitize_resume_state_for_same_stage,
            result_meets_requested_ftol=ctx.result_meets_requested_ftol,
            stage_switch_reason_from_progress=ctx.stage_switch_reason_from_progress,
            merge_stage_chunk_results=ctx.merge_stage_chunk_results,
            result_with_diag=ctx.result_with_diag,
            maybe_rerun_scan_abort_stage=ctx.maybe_rerun_scan_abort_stage,
            scan_abort_fallback_enabled=(not ctx.accelerated_mode) and bool(ctx.performance_mode) and bool(stage_plan.scan_mode),
            verbose=bool(ctx.verbose),
        )
        stage_mode_history[-1] = str(stage_mode_i)
        stage_wall_s.append(float(ctx.perf_counter() - stage_t0))
        try:
            stage_timing = res_i.diagnostics.get("timing", {})
        except Exception:
            stage_timing = {}
        stage_solve_total_s.append(ctx.timing_solve_total_s(stage_timing))
        stage_results.append(res_i)
        stage_statics.append(static_i)
        try:
            w_hist = np.asarray(res_i.w_history)
            prev_stage_fsq = float(w_hist[-1]) if w_hist.size else None
        except Exception:
            prev_stage_fsq = None
        if multigrid_resume and i < (nstep - 1):
            resume_state_stage = ctx.sanitize_resume_state_for_stage(res_i.diagnostics.get("resume_state"))
        state = stage_results[-1].state
        static_prev = static_i
        ftol_last = float(ftol_i)
        step_size_last = float(ctx.step_size_val)

    res = ctx.assemble_multigrid_stage_result(
        stage_results=stage_results,
        state=state,
        solver_mode=str(ctx.solver_mode_eff),
        accelerated_mode=bool(ctx.accelerated_mode),
        multigrid_user_provided=bool(ctx.multigrid_user_provided),
        accelerated_single_grid_default=bool(ctx.accelerated_single_grid_default),
        ns_stages=list(ctx.ns_stages),
        niter_stages=list(ctx.niter_stages),
        ftol_stages=list(ctx.ftol_stages),
        stage_offsets=stage_offsets,
        stage_mode_history=stage_mode_history,
        stage_wall_s=stage_wall_s,
        stage_solve_total_s=stage_solve_total_s,
        niter_stages_input=ctx.niter_stages_input,
    )
    res = ctx.maybe_apply_scan_wout_corrector(
        result=res,
        stage_results=stage_results,
        scan_wout_corrector=ctx.scan_wout_corrector,
        accelerated_mode=bool(ctx.accelerated_mode),
        static_prev=static_prev,
        build_static_func=ctx.build_static_cfg,
        cfg=ctx.cfg,
        ftol_last=ftol_last,
        step_size_last=step_size_last,
        indata=ctx.indata,
        signgs=ctx.signgs,
        use_restart_triggers=ctx.use_restart_triggers,
        vmecpp_restart=bool(ctx.vmecpp_restart),
        stage_transition_factor=float(ctx.stage_transition_factor),
        stage_transition_scale=float(ctx.stage_transition_scale),
        scan_minimal_default=ctx.scan_minimal_default,
        free_boundary_edge_control_projection=ctx.free_boundary_edge_control_projection,
        jit_forces=ctx.jit_forces,
        resolve_jit_forces=ctx.resolve_jit_forces,
        solve_fixed_boundary_residual_iter_func=ctx.solve_fixed_boundary_residual_iter,
        accelerated_fsq_total_target_from_ftol=ctx.accelerated_fsq_total_target_from_ftol,
        copy_final_force_payload=ctx.copy_final_force_payload,
        getenv=ctx.getenv,
    )
    final_requested_ftol = ctx.requested_final_ftol(indata=ctx.indata, ftol_list_input=ctx.ftol_list_input)
    final_target_fsq = ctx.accelerated_fsq_total_target_from_ftol(float(final_requested_ftol))
    res = ctx.finalize_fixed_boundary_convergence_result(
        res,
        requested_ftol=float(final_requested_ftol),
        fsq_total_target=float(final_target_fsq),
        accelerated_mode=bool(ctx.accelerated_mode),
        result_final_residuals=ctx.result_final_residuals,
        result_meets_requested_ftol=ctx.result_meets_requested_ftol,
        result_hits_total_target=ctx.result_hits_total_target,
    )
    static = static_prev if static_prev is not None else ctx.build_static_cfg(ctx.cfg)
    if ctx.verbose and ctx.solver == "vmec2000_iter":
        ctx.print_vmec2000_run_summary(
            input_path=ctx.input_path,
            result=res,
            niter_stage=int(last_niter_stage),
            total_time=ctx.perf_counter() - ctx.t_start,
        )
    return Vmec2000StagedSolveResult(
        result=res,
        static=static,
        stage_results=stage_results,
        stage_statics=stage_statics,
    )

def run_cli_accelerated_budgeted_multigrid(
    ctx: FixedBoundaryStageRunnerContext,
    *,
    ns_stage_list: list[int],
    warm_start_budget: int,
    final_stage_budget: int,
) -> Any:
    """Run the accelerated CLI budgeted-multigrid policy."""

    stage_budgets = ctx.accelerated_cli_budgeted_stage_iters(
        total_budget=int(warm_start_budget),
        ns_stages=ns_stage_list,
    )
    if stage_budgets:
        stage_budgets[-1] = max(int(stage_budgets[-1]), int(final_stage_budget))

    stage_runs: list[Any] = []
    stage_state = None
    stage_static_prev = None
    stage_modes: list[str] = []
    stage_wall_s: list[float] = []
    stage_solve_total_s: list[float] = []

    for ns_i, niter_i in zip(ns_stage_list, stage_budgets):
        stage_mode_i = "accelerated"
        if stage_state is not None and int(stage_static_prev.cfg.ns) != int(ns_i):
            stage_state = ctx.interp_vmec_state(
                stage_state,
                m=stage_static_prev.modes.m,
                n=stage_static_prev.modes.n,
                lthreed=bool(stage_static_prev.cfg.lthreed),
                lconm1=bool(getattr(stage_static_prev.cfg, "lconm1", True)),
                ns_new=int(ns_i),
            )
        kwargs = dict(
            solver="vmec2000_iter",
            solver_mode=stage_mode_i,
            max_iter=int(niter_i),
            step_size=ctx.step_size,
            history_size=int(ctx.history_size),
            gn_damping=ctx.gn_damping,
            gn_cg_tol=ctx.gn_cg_tol,
            gn_cg_maxiter=int(ctx.gn_cg_maxiter),
            use_initial_guess=False,
            vmec_project=bool(ctx.vmec_project),
            use_restart_triggers=ctx.use_restart_triggers,
            vmecpp_restart=bool(ctx.vmecpp_restart),
            use_direct_fallback=ctx.use_direct_fallback,
            free_boundary_edge_control_projection=ctx.free_boundary_edge_control_projection,
            multigrid=False,
            multigrid_use_input_niter=False,
            verbose=bool(ctx.verbose),
            jit_forces=ctx.jit_forces,
            jit_precompile=ctx.jit_precompile,
            use_scan=bool(ctx.use_scan),
            performance_mode=True,
            scan_wout_corrector=ctx.scan_wout_corrector,
            stage_transition_heuristic=ctx.stage_transition_heuristic,
            stage_transition_factor=float(ctx.stage_transition_factor),
            stage_transition_scale=float(ctx.stage_transition_scale),
            grid=ctx.grid,
            cli_fixed_boundary_mode=False,
            _auto_cli_fixed_boundary_mode=False,
        )
        if stage_state is None:
            kwargs["ns_override"] = int(ns_i)
        else:
            kwargs["restart_state"] = stage_state
        stage_t0 = time.perf_counter()
        stage_run = ctx.run_fixed_boundary(ctx.input_path, **kwargs)
        _append_stage_summary(
            run=stage_run,
            mode=stage_mode_i,
            elapsed_s=float(time.perf_counter() - stage_t0),
            stage_runs=stage_runs,
            stage_modes=stage_modes,
            stage_wall_s=stage_wall_s,
            stage_solve_total_s=stage_solve_total_s,
            timing_solve_total_s=ctx.timing_solve_total_s,
        )
        stage_state = stage_run.state
        stage_static_prev = stage_run.static

    final_run = stage_runs[-1]
    if final_run.result is None:
        return final_run
    diag = dict(final_run.result.diagnostics)
    diag["solver_mode"] = str(ctx.solver_mode_eff)
    diag["accelerated_mode"] = True
    diag["cli_fixed_boundary_mode"] = True
    diag["cli_accelerated_fixed_policy"] = "budgeted_multigrid"
    diag["cli_accelerated_stage_ns"] = np.asarray(ns_stage_list, dtype=int)
    diag["cli_accelerated_stage_niter"] = np.asarray(stage_budgets, dtype=int)
    diag["cli_accelerated_stage_modes"] = np.asarray(stage_modes, dtype=object)
    diag["cli_accelerated_stage_wall_s"] = np.asarray(stage_wall_s, dtype=float)
    diag["cli_accelerated_stage_solve_total_s"] = np.asarray(stage_solve_total_s, dtype=float)
    diag["cli_accelerated_stage_fsq"] = np.asarray(
        [float(np.asarray(stage_run.result.w_history)[-1]) for stage_run in stage_runs],
        dtype=float,
    )
    diag["cli_accelerated_budget_total"] = int(warm_start_budget)
    diag["cli_accelerated_final_stage_budget"] = int(final_stage_budget)
    diag["multigrid_ns_stages"] = np.asarray(ns_stage_list, dtype=int)
    diag["multigrid_niter_stages"] = np.asarray(stage_budgets, dtype=int)
    diag["accelerated_single_grid_default"] = False
    final_run = replace(final_run, result=replace(final_run.result, diagnostics=diag))
    return ctx.maybe_finish_cli_fixed_boundary_run(
        final_run,
        initial_policy="budgeted_multigrid",
        enabled=bool(ctx.cli_fixed_boundary_finish_enabled),
    )


def run_cli_explicit_staged_followup(
    ctx: FixedBoundaryStageRunnerContext,
    *,
    ns_stage_list: list[int],
    niter_stage_list: list[int],
    ftol_stage_list: list[float],
    start_stage_index: int = 0,
    restart_state=None,
    restart_static_prev=None,
    restart_resume_state=None,
    stage_mode_override: str | None = None,
    use_scan_override: bool | None = None,
    performance_mode_override: bool | None = None,
    policy_name: str = "input_multigrid",
) -> Any:
    """Run an explicit staged follow-up using VMEC input stage budgets."""

    stage_runs: list[Any] = []
    stage_state = restart_state
    stage_static_prev = restart_static_prev
    stage_resume_state = restart_resume_state
    stage_modes: list[str] = []
    stage_wall_s: list[float] = []
    stage_solve_total_s: list[float] = []

    for idx, (ns_i, niter_i, _ftol_i) in enumerate(zip(ns_stage_list, niter_stage_list, ftol_stage_list)):
        if int(idx) < int(start_stage_index) or int(niter_i) <= 0:
            continue
        if stage_mode_override is not None:
            stage_mode_i = str(stage_mode_override)
        elif bool(ctx.cfg.lthreed) and int(idx) == 0:
            stage_mode_i = "parity"
        else:
            stage_mode_i = "accelerated"
        if stage_state is not None and int(stage_static_prev.cfg.ns) != int(ns_i):
            stage_state = ctx.interp_vmec_state(
                stage_state,
                m=stage_static_prev.modes.m,
                n=stage_static_prev.modes.n,
                lthreed=bool(stage_static_prev.cfg.lthreed),
                lconm1=bool(getattr(stage_static_prev.cfg, "lconm1", True)),
                ns_new=int(ns_i),
            )
        kwargs = dict(
            solver="vmec2000_iter",
            solver_mode=stage_mode_i,
            max_iter=int(niter_i),
            step_size=ctx.step_size,
            history_size=int(ctx.history_size),
            gn_damping=ctx.gn_damping,
            gn_cg_tol=ctx.gn_cg_tol,
            gn_cg_maxiter=int(ctx.gn_cg_maxiter),
            use_initial_guess=False,
            vmec_project=bool(ctx.vmec_project),
            use_restart_triggers=ctx.use_restart_triggers,
            vmecpp_restart=bool(ctx.vmecpp_restart),
            use_direct_fallback=ctx.use_direct_fallback,
            free_boundary_edge_control_projection=ctx.free_boundary_edge_control_projection,
            multigrid=False,
            multigrid_use_input_niter=False,
            verbose=bool(ctx.verbose),
            jit_forces=ctx.jit_forces,
            jit_precompile=ctx.jit_precompile,
            use_scan=bool(ctx.use_scan if use_scan_override is None else use_scan_override),
            performance_mode=True if performance_mode_override is None else bool(performance_mode_override),
            scan_wout_corrector=ctx.scan_wout_corrector,
            stage_transition_heuristic=ctx.stage_transition_heuristic,
            stage_transition_factor=float(ctx.stage_transition_factor),
            stage_transition_scale=float(ctx.stage_transition_scale),
            grid=ctx.grid,
            cli_fixed_boundary_mode=False,
            _auto_cli_fixed_boundary_mode=False,
        )
        if stage_state is None:
            kwargs["ns_override"] = int(ns_i)
        else:
            kwargs["restart_state"] = stage_state
            if stage_resume_state is not None:
                kwargs["restart_solver_state"] = stage_resume_state
        stage_t0 = time.perf_counter()
        stage_run = ctx.run_fixed_boundary(ctx.input_path, **kwargs)
        _append_stage_summary(
            run=stage_run,
            mode=stage_mode_i,
            elapsed_s=float(time.perf_counter() - stage_t0),
            stage_runs=stage_runs,
            stage_modes=stage_modes,
            stage_wall_s=stage_wall_s,
            stage_solve_total_s=stage_solve_total_s,
            timing_solve_total_s=ctx.timing_solve_total_s,
        )
        stage_state = stage_run.state
        stage_static_prev = stage_run.static
        stage_resume_state = ctx.sanitize_resume_state_for_stage(
            stage_run.result.diagnostics.get("resume_state") if stage_run.result is not None else None
        )

    final_run = stage_runs[-1]
    if final_run.result is None:
        return final_run
    diag = dict(final_run.result.diagnostics)
    diag["solver_mode"] = str(ctx.solver_mode_eff)
    diag["accelerated_mode"] = True
    diag["cli_fixed_boundary_mode"] = True
    diag["cli_staged_followup_policy"] = str(policy_name)
    diag["cli_staged_followup_stage_ns"] = np.asarray(ns_stage_list, dtype=int)
    diag["cli_staged_followup_stage_niter"] = np.asarray(niter_stage_list, dtype=int)
    diag["cli_staged_followup_executed_stage_ns"] = np.asarray(
        [int(ns_stage_list[i]) for i in range(int(start_stage_index), len(ns_stage_list)) if int(niter_stage_list[i]) > 0],
        dtype=int,
    )
    diag["cli_staged_followup_executed_stage_niter"] = np.asarray(
        [
            int(niter_stage_list[i])
            for i in range(int(start_stage_index), len(niter_stage_list))
            if int(niter_stage_list[i]) > 0
        ],
        dtype=int,
    )
    diag["cli_staged_followup_stage_modes"] = np.asarray(stage_modes, dtype=object)
    diag["cli_staged_followup_stage_wall_s"] = np.asarray(stage_wall_s, dtype=float)
    diag["cli_staged_followup_stage_solve_total_s"] = np.asarray(stage_solve_total_s, dtype=float)
    diag["cli_staged_followup_start_stage_index"] = int(start_stage_index)
    diag["cli_staged_followup_stage_fsq"] = np.asarray(
        [float(np.asarray(stage_run.result.w_history)[-1]) for stage_run in stage_runs],
        dtype=float,
    )
    return replace(final_run, result=replace(final_run.result, diagnostics=diag))


__all__ = [
    "FixedBoundaryStageRunnerContext",
    "Vmec2000StagedSolveContext",
    "Vmec2000StagedSolveResult",
    "run_cli_accelerated_budgeted_multigrid",
    "run_cli_explicit_staged_followup",
    "run_vmec2000_staged_solve",
]
