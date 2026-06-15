"""CLI fixed-boundary finish policy for :mod:`vmec_jax.driver`.

The public driver owns input loading, staging, and solver dispatch. This module
owns the final CLI-only convergence/output policy so the driver can delegate the
bounded retry/fallback bookkeeping without changing public behavior.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FixedBoundaryFinishContext:
    """Driver state and callbacks needed by the CLI fixed-boundary finisher."""

    input_path: Any
    cfg: Any
    indata: Any
    solver_mode_eff: str
    accelerated_mode: bool
    ftol_list_input: Any
    ns_list_input: Any
    niter_list_input: Any
    deferred_staged_current_driven_3d_cli: bool
    max_iter: int
    max_iter_overridden: bool
    step_size: Any
    step_size_sentinel: Any
    history_size: int
    gn_damping: Any
    gn_cg_tol: Any
    gn_cg_maxiter: int
    vmec_project: bool
    use_restart_triggers: bool | None
    vmecpp_restart: bool
    use_direct_fallback: bool | None
    multigrid: bool
    multigrid_use_input_niter: bool
    multigrid_user_provided: bool
    verbose: bool
    jit_forces: bool | str
    jit_precompile: bool | None
    use_scan: bool | None
    scan_wout_corrector: bool | None
    stage_transition_heuristic: bool | None
    stage_transition_factor: float
    stage_transition_scale: float
    grid: Any
    policy_backend: str
    direct_external_provider: bool
    accelerated_single_grid_default: bool
    run_fixed_boundary: Callable[..., Any]
    run_cli_explicit_staged_followup: Callable[..., Any]
    get_stage_results: Callable[[], Sequence[Any]]
    get_stage_statics: Callable[[], Sequence[Any]]
    sanitize_resume_state_for_stage: Callable[[Any], Any]
    solve_fixed_boundary_residual_iter: Callable[..., Any]
    default_backend_name: Callable[[], str]
    host_update_assembly_driver_default: Callable[..., bool]
    default_preconditioner_use_precomputed_tridi: Callable[..., bool]
    default_preconditioner_use_lax_tridi: Callable[..., bool]
    resolve_jit_forces_auto_policy: Callable[..., bool]
    requested_final_ftol: Callable[..., float]
    accelerated_fsq_total_target_from_ftol: Callable[[float], float]
    result_final_fsq: Callable[[Any], float]
    result_final_residuals: Callable[[Any], tuple[float, float, float] | None]
    result_hits_total_target: Callable[..., bool]
    result_meets_requested_ftol: Callable[..., bool]
    sanitize_minimal_resume_state_for_finish: Callable[[Any], Any]


def _empty_finish_diagnostics(diag: dict, *, converged: bool, strict: bool, total: bool) -> dict:
    diag["cli_fixed_boundary_finish_budgets"] = np.zeros((0,), dtype=int)
    diag["cli_fixed_boundary_finish_fsq"] = np.zeros((0,), dtype=float)
    diag["cli_fixed_boundary_finish_converged"] = np.zeros((0,), dtype=bool)
    diag["cli_fixed_boundary_finish_modes"] = np.asarray([], dtype=object)
    diag["cli_fixed_boundary_full_parity_fallback"] = False
    diag["converged"] = bool(converged)
    diag["converged_strict"] = bool(strict)
    diag["converged_by_total_fsq"] = bool(total)
    return diag


def maybe_finish_cli_fixed_boundary_run(
    run_in: Any,
    *,
    initial_policy: str,
    enabled: bool,
    context: FixedBoundaryFinishContext,
) -> Any:
    """Apply the CLI fixed-boundary finish policy to an already-produced run."""

    ctx = context
    if not bool(enabled):
        return run_in
    if run_in.result is None:
        return run_in
    base_diag = dict(run_in.result.diagnostics)
    base_diag["solver_mode"] = str(ctx.solver_mode_eff)
    base_diag["accelerated_mode"] = bool(ctx.accelerated_mode)
    base_diag["cli_fixed_boundary_mode"] = True
    base_diag["cli_fixed_boundary_initial_policy"] = str(initial_policy)
    requested_ftol = ctx.requested_final_ftol(indata=ctx.indata, ftol_list_input=ctx.ftol_list_input)
    target_fsq = ctx.accelerated_fsq_total_target_from_ftol(float(requested_ftol))
    base_diag["requested_ftol"] = float(requested_ftol)
    base_diag["fsq_total_target"] = float(target_fsq)
    staged_input = (ctx.ns_list_input is not None) and (len(ctx.ns_list_input) > 1)
    explicit_niter_stages = (
        [int(v) for v in ctx.niter_list_input]
        if (ctx.niter_list_input is not None) and (len(ctx.niter_list_input) == len(ctx.ns_list_input or []))
        else None
    )
    require_staged_followup = (
        bool(ctx.accelerated_mode)
        and str(initial_policy) == "single_grid"
        and bool(staged_input)
        and (explicit_niter_stages is not None)
        and bool(ctx.cfg.lthreed)
        and (not bool(ctx.deferred_staged_current_driven_3d_cli))
    )
    run_in_strict = ctx.result_meets_requested_ftol(run_in.result, ftol=float(requested_ftol))
    run_in_total = ctx.result_hits_total_target(run_in.result, fsq_total_target=float(target_fsq))
    if (
        bool(run_in.result.diagnostics.get("converged", False))
        and bool(run_in_strict)
        and (not bool(require_staged_followup))
    ):
        _empty_finish_diagnostics(base_diag, converged=True, strict=True, total=bool(run_in_total))
        return replace(run_in, result=replace(run_in.result, diagnostics=base_diag))
    if bool(run_in_strict) and (not bool(require_staged_followup)):
        _empty_finish_diagnostics(base_diag, converged=True, strict=True, total=bool(run_in_total))
        return replace(run_in, result=replace(run_in.result, diagnostics=base_diag))

    base_total_budget = max(1, int(ctx.max_iter))
    max_fallback_budget = int(2 * base_total_budget)

    best_run = run_in
    best_fsq = float(ctx.result_final_fsq(run_in.result))
    attempt_budgets: list[int] = []
    attempt_fsq: list[float] = []
    attempt_converged: list[bool] = []
    attempt_modes: list[str] = []
    fallback_used = False
    partial_fallback_used = False
    staged_followup_used = False
    staged_followup_policy = ""
    staged_followup_ns = np.zeros((0,), dtype=int)
    staged_followup_niter = np.zeros((0,), dtype=int)
    staged_followup_modes = np.asarray([], dtype=object)
    staged_followup_fsq = np.zeros((0,), dtype=float)
    staged_followup_wall_s = np.zeros((0,), dtype=float)
    staged_followup_solve_total_s = np.zeros((0,), dtype=float)

    def _resolve_finish_jit_forces(static_i, niter_i: int) -> bool:
        return ctx.resolve_jit_forces_auto_policy(ctx.jit_forces, static_i, niter_i)

    def _run_finish_attempt(*, budget_i: int, mode_i: str, use_scan_i: bool, performance_mode_i: bool):
        static_i = best_run.static
        mode_i_l = str(mode_i).strip().lower()
        scan_minimal_default_i = True if (bool(performance_mode_i) and (not bool(ctx.verbose))) else None
        host_update_assembly_i = ctx.host_update_assembly_driver_default(
            cfg=static_i.cfg,
            performance_mode=bool(performance_mode_i),
            backend=ctx.default_backend_name(),
            use_scan=bool(use_scan_i),
        )
        preconditioner_use_precomputed_tridi_i = ctx.default_preconditioner_use_precomputed_tridi(
            cfg=static_i.cfg,
            backend=ctx.policy_backend,
            performance_mode=bool(performance_mode_i),
            use_scan=bool(use_scan_i),
            direct_external_provider=bool(ctx.direct_external_provider),
        )
        preconditioner_use_lax_tridi_i = ctx.default_preconditioner_use_lax_tridi(
            cfg=static_i.cfg,
            backend=ctx.policy_backend,
            performance_mode=bool(performance_mode_i),
            use_scan=bool(use_scan_i),
            direct_external_provider=bool(ctx.direct_external_provider),
        )
        if ctx.step_size is ctx.step_size_sentinel or ctx.step_size is None:
            step_size_finish = float(ctx.indata.get_float("DELT", 5e-3))
        else:
            step_size_finish = float(ctx.step_size)
        finish_fsq_total_target = float(target_fsq) if mode_i_l == "accelerated" else None
        finish_resume_state_mode = "minimal" if mode_i_l == "accelerated" else "full"
        res_i = ctx.solve_fixed_boundary_residual_iter(
            best_run.state,
            static_i,
            indata=ctx.indata,
            signgs=best_run.signgs,
            ftol=float(ctx.indata.get_float("FTOL", 1.0e-13)),
            max_iter=int(budget_i),
            step_size=float(step_size_finish),
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
            reference_mode=False,
            use_restart_triggers=True if ctx.use_restart_triggers is None else bool(ctx.use_restart_triggers),
            vmecpp_restart=bool(ctx.vmecpp_restart),
            stage_prev_fsq=None,
            stage_transition_factor=float(ctx.stage_transition_factor),
            stage_transition_scale=float(ctx.stage_transition_scale),
            use_direct_fallback=ctx.use_direct_fallback,
            # CLI finish attempts deliberately restart from the current
            # equilibrium state only. Reusing nonlinear-controller caches was
            # materially less robust on the hard staged inputs.
            resume_state=None,
            verbose=False,
            verbose_vmec2000_table=False,
            jit_precompile=False,
            jit_warmup_iters=0,
            use_scan=bool(use_scan_i),
            scan_minimal_default=scan_minimal_default_i,
            light_history=True,
            resume_state_mode=finish_resume_state_mode,
            fsq_total_target=finish_fsq_total_target,
            host_update_assembly=host_update_assembly_i,
            preconditioner_use_precomputed_tridi=preconditioner_use_precomputed_tridi_i,
            preconditioner_use_lax_tridi=preconditioner_use_lax_tridi_i,
            return_final_force_payload=True,
            jit_forces=_resolve_finish_jit_forces(static_i, int(budget_i)),
        )
        return replace(best_run, state=res_i.state, result=res_i)

    if staged_input and bool(ctx.accelerated_mode) and str(initial_policy) == "single_grid":
        explicit_ftol_stages = (
            [float(v) for v in ctx.ftol_list_input]
            if (ctx.ftol_list_input is not None) and (len(ctx.ftol_list_input) == len(ctx.ns_list_input))
            else [float(ctx.indata.get_float("FTOL", 1.0e-13))] * len(ctx.ns_list_input)
        )
        missed_target = not bool(ctx.result_meets_requested_ftol(best_run.result, ftol=float(requested_ftol)))
        should_run_staged_followup = bool(explicit_niter_stages is not None) and (
            bool(require_staged_followup) or bool(missed_target)
        )
        if should_run_staged_followup:
            staged_followup = ctx.run_cli_explicit_staged_followup(
                ns_stage_list=[int(v) for v in ctx.ns_list_input],
                niter_stage_list=explicit_niter_stages,
                ftol_stage_list=explicit_ftol_stages,
                policy_name="input_multigrid",
            )
            staged_followup_used = True
            staged_followup_policy = "input_multigrid"
            staged_diag = dict(staged_followup.result.diagnostics)
            staged_followup_ns = np.asarray(staged_diag.get("cli_staged_followup_stage_ns", []), dtype=int)
            staged_followup_niter = np.asarray(staged_diag.get("cli_staged_followup_stage_niter", []), dtype=int)
            staged_followup_modes = np.asarray(staged_diag.get("cli_staged_followup_stage_modes", []), dtype=object)
            staged_followup_fsq = np.asarray(staged_diag.get("cli_staged_followup_stage_fsq", []), dtype=float)
            staged_followup_wall_s = np.asarray(
                staged_diag.get("cli_staged_followup_stage_wall_s", []),
                dtype=float,
            )
            staged_followup_solve_total_s = np.asarray(
                staged_diag.get("cli_staged_followup_stage_solve_total_s", []),
                dtype=float,
            )
            staged_fsq_val = float(ctx.result_final_fsq(staged_followup.result))
            staged_conv = bool(ctx.result_meets_requested_ftol(staged_followup.result, ftol=float(requested_ftol)))
            if staged_conv or (staged_fsq_val < float(best_fsq)):
                best_run = staged_followup
                best_fsq = float(staged_fsq_val)

    # Accelerated multigrid can still miss the correct branch on some explicit
    # staged inputs even though xvmec2000 converges with the same sequence.
    if (
        bool(staged_input)
        and bool(ctx.accelerated_mode)
        and str(initial_policy) == "multigrid"
        and (explicit_niter_stages is not None)
        and (not bool(ctx.result_meets_requested_ftol(best_run.result, ftol=float(requested_ftol))))
    ):
        partial_start_stage = int(max(1, len(ctx.ns_list_input) - 1))
        partial_restart_state = None
        partial_restart_static_prev = None
        partial_restart_resume_state = None
        try:
            stage_results = ctx.get_stage_results()
            stage_statics = ctx.get_stage_statics()
            if len(stage_results) >= int(partial_start_stage):
                prev_idx = int(partial_start_stage) - 1
                partial_restart_state = stage_results[prev_idx].state
                partial_restart_static_prev = stage_statics[prev_idx]
                partial_restart_resume_state = ctx.sanitize_resume_state_for_stage(
                    stage_results[prev_idx].diagnostics.get("resume_state")
                )
        except Exception:
            partial_restart_state = None
            partial_restart_static_prev = None
            partial_restart_resume_state = None

        if (partial_restart_state is not None) and (partial_restart_static_prev is not None):
            partial_fallback = ctx.run_cli_explicit_staged_followup(
                ns_stage_list=[int(v) for v in ctx.ns_list_input],
                niter_stage_list=explicit_niter_stages,
                ftol_stage_list=(
                    [float(v) for v in ctx.ftol_list_input]
                    if (ctx.ftol_list_input is not None) and (len(ctx.ftol_list_input) == len(ctx.ns_list_input))
                    else [float(ctx.indata.get_float("FTOL", 1.0e-13))] * len(ctx.ns_list_input)
                ),
                start_stage_index=int(partial_start_stage),
                restart_state=partial_restart_state,
                restart_static_prev=partial_restart_static_prev,
                restart_resume_state=partial_restart_resume_state,
                stage_mode_override="parity",
                use_scan_override=False,
                performance_mode_override=False,
                policy_name="partial_parity_multigrid",
            )
            partial_fallback_used = True
            partial_fallback_fsq = float(ctx.result_final_fsq(partial_fallback.result))
            partial_fallback_conv = bool(
                ctx.result_meets_requested_ftol(partial_fallback.result, ftol=float(requested_ftol))
            )
            if partial_fallback_conv or partial_fallback_fsq < best_fsq:
                best_run = partial_fallback
                best_fsq = float(partial_fallback_fsq)

        if not bool(ctx.result_meets_requested_ftol(best_run.result, ftol=float(requested_ftol))):
            fallback_used = True
            fallback = ctx.run_fixed_boundary(
                ctx.input_path,
                solver="vmec2000_iter",
                solver_mode="parity",
                max_iter=int(max_fallback_budget),
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
                multigrid=True,
                multigrid_use_input_niter=bool(ctx.multigrid_use_input_niter),
                verbose=bool(ctx.verbose),
                jit_forces=ctx.jit_forces,
                jit_precompile=ctx.jit_precompile,
                use_scan=False,
                performance_mode=False,
                scan_wout_corrector=ctx.scan_wout_corrector,
                stage_transition_heuristic=ctx.stage_transition_heuristic,
                stage_transition_factor=float(ctx.stage_transition_factor),
                stage_transition_scale=float(ctx.stage_transition_scale),
                grid=ctx.grid,
                cli_fixed_boundary_mode=False,
                _auto_cli_fixed_boundary_mode=False,
            )
            fallback_fsq = float(ctx.result_final_fsq(fallback.result))
            fallback_conv = bool(ctx.result_meets_requested_ftol(fallback.result, ftol=float(requested_ftol)))
            if fallback_conv or fallback_fsq < best_fsq:
                best_run = fallback
                best_fsq = float(fallback_fsq)

    improvement_floor = np.finfo(float).eps * max(1.0, abs(float(best_fsq)), abs(float(target_fsq)))
    finish_budget_cap = int(max_fallback_budget) if bool(ctx.max_iter_overridden) else None
    finish_budget_used = 0
    accelerated_finish_uses_scan = False if ctx.use_scan is False else True
    if (
        bool(ctx.accelerated_mode)
        and str(initial_policy) == "single_grid"
        and (not bool(staged_followup_used))
        and not bool(ctx.result_meets_requested_ftol(best_run.result, ftol=float(requested_ftol)))
    ):
        accel_budget_i = int(base_total_budget)
        accel_budget_used = 0
        while int(accel_budget_i) >= 1 and int(accel_budget_used) < int(max_fallback_budget):
            if finish_budget_cap is not None:
                remaining_finish_budget = int(finish_budget_cap) - int(finish_budget_used)
                if remaining_finish_budget <= 0:
                    break
                budget_this = min(int(accel_budget_i), int(remaining_finish_budget))
            else:
                budget_this = int(accel_budget_i)
            prev_best_fsq = float(best_fsq)
            trial = _run_finish_attempt(
                budget_i=budget_this,
                mode_i="accelerated",
                use_scan_i=bool(accelerated_finish_uses_scan),
                performance_mode_i=True,
            )
            trial_fsq = float(ctx.result_final_fsq(trial.result))
            trial_conv = bool(ctx.result_meets_requested_ftol(trial.result, ftol=float(requested_ftol)))
            attempt_budgets.append(int(budget_this))
            attempt_fsq.append(float(trial_fsq))
            attempt_converged.append(bool(trial_conv))
            attempt_modes.append("accelerated")
            accel_budget_used += int(budget_this)
            finish_budget_used += int(budget_this)
            improved = trial_conv or (float(trial_fsq) < float(prev_best_fsq - improvement_floor))
            if improved:
                best_run = trial
                best_fsq = float(trial_fsq)
            if trial_conv or (not improved):
                break
    # For multigrid paths where the final stage exhausted its NITER budget, skip
    # extra parity iterations to match VMEC2000's normal-termination behavior.
    multigrid_niter_exhausted = (
        str(initial_policy) == "multigrid"
        and bool(best_run.result.diagnostics.get("multigrid_final_stage_niter_exhausted", False))
    )
    if not bool(multigrid_niter_exhausted) and not bool(
        ctx.result_meets_requested_ftol(best_run.result, ftol=float(requested_ftol))
    ):
        budget_i = int(base_total_budget)
        while int(budget_i) >= 1:
            if finish_budget_cap is not None:
                remaining_finish_budget = int(finish_budget_cap) - int(finish_budget_used)
                if remaining_finish_budget <= 0:
                    break
                budget_this = min(int(budget_i), int(remaining_finish_budget))
            else:
                budget_this = int(budget_i)
            prev_best_fsq = float(best_fsq)
            trial = _run_finish_attempt(
                budget_i=budget_this,
                mode_i="parity",
                use_scan_i=False,
                performance_mode_i=False,
            )
            trial_fsq = float(ctx.result_final_fsq(trial.result))
            trial_conv = bool(ctx.result_meets_requested_ftol(trial.result, ftol=float(requested_ftol)))
            attempt_budgets.append(int(budget_this))
            attempt_fsq.append(float(trial_fsq))
            attempt_converged.append(bool(trial_conv))
            attempt_modes.append("parity")
            finish_budget_used += int(budget_this)
            improved = trial_conv or (float(trial_fsq) < float(prev_best_fsq - improvement_floor))
            if improved:
                best_run = trial
                best_fsq = float(trial_fsq)
            if trial_conv:
                break
            if improved:
                continue
            next_budget = max(1, int(np.ceil(float(budget_i) / 2.0)))
            if int(next_budget) == int(budget_i):
                break
            budget_i = int(next_budget)

    if (
        staged_input
        and not bool(ctx.result_meets_requested_ftol(best_run.result, ftol=float(requested_ftol)))
        and bool(ctx.accelerated_mode)
        and not bool(multigrid_niter_exhausted)
    ):
        fallback_used = True
        fallback = ctx.run_fixed_boundary(
            ctx.input_path,
            solver="vmec2000_iter",
            solver_mode="parity",
            max_iter=int(max_fallback_budget),
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
            multigrid=ctx.multigrid if bool(ctx.multigrid_user_provided) else None,
            multigrid_use_input_niter=bool(ctx.multigrid_use_input_niter),
            verbose=bool(ctx.verbose),
            jit_forces=ctx.jit_forces,
            jit_precompile=ctx.jit_precompile,
            use_scan=False,
            performance_mode=False,
            scan_wout_corrector=ctx.scan_wout_corrector,
            stage_transition_heuristic=ctx.stage_transition_heuristic,
            stage_transition_factor=float(ctx.stage_transition_factor),
            stage_transition_scale=float(ctx.stage_transition_scale),
            grid=ctx.grid,
            cli_fixed_boundary_mode=False,
            _auto_cli_fixed_boundary_mode=False,
        )
        fallback_fsq = float(ctx.result_final_fsq(fallback.result))
        fallback_conv = bool(ctx.result_meets_requested_ftol(fallback.result, ftol=float(requested_ftol)))
        if fallback_conv or fallback_fsq < best_fsq:
            best_run = fallback
            best_fsq = float(fallback_fsq)

    diag = dict(base_diag)
    diag.update(best_run.result.diagnostics)
    diag["solver_mode"] = str(ctx.solver_mode_eff)
    diag["accelerated_mode"] = bool(ctx.accelerated_mode)
    diag["cli_fixed_boundary_mode"] = True
    diag["cli_fixed_boundary_initial_policy"] = str(initial_policy)
    final_residuals = ctx.result_final_residuals(best_run.result)
    strict_converged = bool(ctx.result_meets_requested_ftol(best_run.result, ftol=float(requested_ftol)))
    total_converged = bool(ctx.result_hits_total_target(best_run.result, fsq_total_target=float(target_fsq)))
    diag["requested_ftol"] = float(requested_ftol)
    if final_residuals is not None:
        diag["final_fsqr"] = float(final_residuals[0])
        diag["final_fsqz"] = float(final_residuals[1])
        diag["final_fsql"] = float(final_residuals[2])
    diag["converged"] = bool(strict_converged)
    diag["converged_strict"] = bool(strict_converged)
    diag["converged_by_total_fsq"] = bool(total_converged)
    diag["cli_fixed_boundary_partial_parity_fallback"] = bool(partial_fallback_used)
    diag["cli_fixed_boundary_finish_budgets"] = np.asarray(attempt_budgets, dtype=int)
    diag["cli_fixed_boundary_finish_fsq"] = np.asarray(attempt_fsq, dtype=float)
    diag["cli_fixed_boundary_finish_converged"] = np.asarray(attempt_converged, dtype=bool)
    diag["cli_fixed_boundary_finish_modes"] = np.asarray(attempt_modes)
    diag["cli_fixed_boundary_finish_budget_cap"] = -1 if finish_budget_cap is None else int(finish_budget_cap)
    diag["cli_fixed_boundary_finish_budget_exhausted"] = bool(
        (finish_budget_cap is not None)
        and int(finish_budget_used) >= int(finish_budget_cap)
        and not bool(strict_converged)
    )
    diag["cli_fixed_boundary_full_parity_fallback"] = bool(fallback_used)
    diag["cli_fixed_boundary_staged_followup_used"] = bool(staged_followup_used)
    diag["cli_fixed_boundary_staged_followup_policy"] = str(staged_followup_policy)
    diag["cli_fixed_boundary_staged_followup_ns"] = staged_followup_ns
    diag["cli_fixed_boundary_staged_followup_niter"] = staged_followup_niter
    diag["cli_fixed_boundary_staged_followup_modes"] = staged_followup_modes
    diag["cli_fixed_boundary_staged_followup_fsq"] = staged_followup_fsq
    diag["cli_fixed_boundary_staged_followup_wall_s"] = staged_followup_wall_s
    diag["cli_fixed_boundary_staged_followup_solve_total_s"] = staged_followup_solve_total_s
    diag["multigrid_user_provided"] = bool(ctx.multigrid_user_provided)
    diag["accelerated_single_grid_default"] = bool(ctx.accelerated_single_grid_default)
    if bool(ctx.accelerated_mode):
        diag["resume_state_mode"] = "minimal"
        diag["resume_state"] = ctx.sanitize_minimal_resume_state_for_finish(diag.get("resume_state"))
    best_run = replace(best_run, result=replace(best_run.result, diagnostics=diag))
    return best_run
