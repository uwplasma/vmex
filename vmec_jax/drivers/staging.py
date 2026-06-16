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
    "run_cli_accelerated_budgeted_multigrid",
    "run_cli_explicit_staged_followup",
]
