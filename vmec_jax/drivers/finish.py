"""CLI fixed-boundary finish policy for :mod:`vmec_jax.driver`.

The public driver owns input loading, staging, and solver dispatch. This module
owns the final CLI-only convergence/output policy so the driver can delegate the
bounded retry/fallback bookkeeping without changing public behavior.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from vmec_jax._solve_runtime import _dataclass_from_namespace


@dataclass
class FinishAttemptLog:
    """Bookkeeping for CLI finish retry attempts."""

    budgets: list[int] = field(default_factory=list)
    fsq: list[float] = field(default_factory=list)
    converged: list[bool] = field(default_factory=list)
    modes: list[str] = field(default_factory=list)

    def record(
        self,
        ctx: "FixedBoundaryFinishContext",
        trial: Any,
        *,
        requested_ftol: float,
        budget_i: int,
        mode_i: str,
    ) -> tuple[float, bool]:
        trial_fsq = float(ctx.result_final_fsq(trial.result))
        trial_conv = bool(ctx.result_meets_requested_ftol(trial.result, ftol=float(requested_ftol)))
        self.budgets.append(int(budget_i))
        self.fsq.append(float(trial_fsq))
        self.converged.append(bool(trial_conv))
        self.modes.append(str(mode_i))
        return trial_fsq, trial_conv


@dataclass
class FinishBudgetTracker:
    """Apply the optional explicit ``max_iter`` cap across finish attempts."""

    cap: int | None
    used: int = 0

    def bounded(self, requested: int) -> int:
        if self.cap is None:
            return int(requested)
        remaining = int(self.cap) - int(self.used)
        return 0 if remaining <= 0 else min(int(requested), int(remaining))

    def add(self, budget: int) -> None:
        self.used += int(budget)


@dataclass(frozen=True)
class StagedFollowupDiagnostics:
    """Diagnostics copied from an explicit staged followup run."""

    used: bool = False
    policy: str = ""
    ns: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=int))
    niter: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=int))
    modes: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=object))
    fsq: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=float))
    wall_s: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=float))
    solve_total_s: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=float))

    @classmethod
    def from_run(cls, run: Any, *, policy: str) -> "StagedFollowupDiagnostics":
        diag = dict(run.result.diagnostics)
        return cls(
            used=True,
            policy=str(policy),
            ns=np.asarray(diag.get("cli_staged_followup_stage_ns", []), dtype=int),
            niter=np.asarray(diag.get("cli_staged_followup_stage_niter", []), dtype=int),
            modes=np.asarray(diag.get("cli_staged_followup_stage_modes", []), dtype=object),
            fsq=np.asarray(diag.get("cli_staged_followup_stage_fsq", []), dtype=float),
            wall_s=np.asarray(diag.get("cli_staged_followup_stage_wall_s", []), dtype=float),
            solve_total_s=np.asarray(diag.get("cli_staged_followup_stage_solve_total_s", []), dtype=float),
        )


@dataclass(frozen=True)
class FinishDiagnosticInputs:
    """Explicit inputs needed to stamp CLI finish diagnostics on the selected run."""

    ctx: "FixedBoundaryFinishContext"
    best_run: Any
    requested_ftol: float
    target_fsq: float
    base_diag: dict[str, Any]
    initial_policy: str
    partial_fallback_used: bool
    fallback_used: bool
    staged_followup: StagedFollowupDiagnostics
    attempt_log: FinishAttemptLog
    finish_budget: FinishBudgetTracker


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

    @classmethod
    def from_namespace(cls, namespace: dict[str, Any], /, **overrides: Any) -> "FixedBoundaryFinishContext":
        return _dataclass_from_namespace(cls, namespace, label="fixed-boundary finish", overrides=overrides)


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


def _run_full_parity_fallback(ctx: FixedBoundaryFinishContext, *, max_fallback_budget: int, multigrid: Any) -> Any:
    """Run the conservative full-parity CLI fallback with one shared policy."""
    return ctx.run_fixed_boundary(
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
        multigrid=multigrid,
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


def _run_finish_attempt(
    ctx: FixedBoundaryFinishContext,
    *,
    best_run: Any,
    target_fsq: float,
    budget_i: int,
    mode_i: str,
    use_scan_i: bool,
    performance_mode_i: bool,
) -> Any:
    """Run one state-only finish attempt from the current best equilibrium."""

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
        jit_forces=ctx.resolve_jit_forces_auto_policy(ctx.jit_forces, static_i, int(budget_i)),
    )
    return replace(best_run, state=res_i.state, result=res_i)


def _finish_run_with_diagnostics(info: FinishDiagnosticInputs) -> Any:
    """Attach CLI finish diagnostics to the selected fixed-boundary run."""

    ctx = info.ctx
    best_run = info.best_run
    requested_ftol = float(info.requested_ftol)
    target_fsq = float(info.target_fsq)
    final_residuals = ctx.result_final_residuals(best_run.result)
    strict_converged = bool(ctx.result_meets_requested_ftol(best_run.result, ftol=requested_ftol))
    total_converged = bool(ctx.result_hits_total_target(best_run.result, fsq_total_target=target_fsq))
    attempt_log = info.attempt_log
    finish_budget = info.finish_budget
    staged_followup = info.staged_followup
    diag = dict(info.base_diag)
    diag.update(best_run.result.diagnostics)
    diag["solver_mode"] = str(ctx.solver_mode_eff)
    diag["accelerated_mode"] = bool(ctx.accelerated_mode)
    diag["cli_fixed_boundary_mode"] = True
    diag["cli_fixed_boundary_initial_policy"] = str(info.initial_policy)
    diag["requested_ftol"] = requested_ftol
    if final_residuals is not None:
        diag["final_fsqr"] = float(final_residuals[0])
        diag["final_fsqz"] = float(final_residuals[1])
        diag["final_fsql"] = float(final_residuals[2])
    diag["converged"] = bool(strict_converged)
    diag["converged_strict"] = bool(strict_converged)
    diag["converged_by_total_fsq"] = bool(total_converged)
    diag["cli_fixed_boundary_partial_parity_fallback"] = bool(info.partial_fallback_used)
    diag["cli_fixed_boundary_finish_budgets"] = np.asarray(attempt_log.budgets, dtype=int)
    diag["cli_fixed_boundary_finish_fsq"] = np.asarray(attempt_log.fsq, dtype=float)
    diag["cli_fixed_boundary_finish_converged"] = np.asarray(attempt_log.converged, dtype=bool)
    diag["cli_fixed_boundary_finish_modes"] = np.asarray(attempt_log.modes)
    diag["cli_fixed_boundary_finish_budget_cap"] = -1 if finish_budget.cap is None else int(finish_budget.cap)
    diag["cli_fixed_boundary_finish_budget_exhausted"] = bool(
        (finish_budget.cap is not None)
        and int(finish_budget.used) >= int(finish_budget.cap)
        and not bool(strict_converged)
    )
    diag["cli_fixed_boundary_full_parity_fallback"] = bool(info.fallback_used)
    diag["cli_fixed_boundary_staged_followup_used"] = bool(staged_followup.used)
    diag["cli_fixed_boundary_staged_followup_policy"] = str(staged_followup.policy)
    diag["cli_fixed_boundary_staged_followup_ns"] = staged_followup.ns
    diag["cli_fixed_boundary_staged_followup_niter"] = staged_followup.niter
    diag["cli_fixed_boundary_staged_followup_modes"] = staged_followup.modes
    diag["cli_fixed_boundary_staged_followup_fsq"] = staged_followup.fsq
    diag["cli_fixed_boundary_staged_followup_wall_s"] = staged_followup.wall_s
    diag["cli_fixed_boundary_staged_followup_solve_total_s"] = staged_followup.solve_total_s
    diag["multigrid_user_provided"] = bool(ctx.multigrid_user_provided)
    diag["accelerated_single_grid_default"] = bool(ctx.accelerated_single_grid_default)
    if bool(ctx.accelerated_mode):
        diag["resume_state_mode"] = "minimal"
        diag["resume_state"] = ctx.sanitize_minimal_resume_state_for_finish(diag.get("resume_state"))
    return replace(best_run, result=replace(best_run.result, diagnostics=diag))


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
    ns_stage_count = len(ctx.ns_list_input) if staged_input else 0
    explicit_niter_stages = (
        [int(v) for v in ctx.niter_list_input]
        if (ctx.niter_list_input is not None) and (len(ctx.niter_list_input) == ns_stage_count)
        else None
    )
    explicit_ftol_stages = (
        [float(v) for v in ctx.ftol_list_input]
        if (ctx.ftol_list_input is not None) and (len(ctx.ftol_list_input) == ns_stage_count)
        else [float(ctx.indata.get_float("FTOL", 1.0e-13))] * ns_stage_count
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
    if bool(run_in_strict) and (not bool(require_staged_followup)):
        _empty_finish_diagnostics(base_diag, converged=True, strict=True, total=bool(run_in_total))
        return replace(run_in, result=replace(run_in.result, diagnostics=base_diag))

    base_total_budget = max(1, int(ctx.max_iter))
    max_fallback_budget = int(2 * base_total_budget)

    best_run = run_in
    best_fsq = float(ctx.result_final_fsq(run_in.result))
    attempt_log = FinishAttemptLog()
    fallback_used = False
    partial_fallback_used = False
    staged_followup_diag = StagedFollowupDiagnostics()

    if staged_input and bool(ctx.accelerated_mode) and str(initial_policy) == "single_grid":
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
            staged_followup_diag = StagedFollowupDiagnostics.from_run(staged_followup, policy="input_multigrid")
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
                ftol_stage_list=explicit_ftol_stages,
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
            fallback = _run_full_parity_fallback(ctx, max_fallback_budget=int(max_fallback_budget), multigrid=True)
            fallback_fsq = float(ctx.result_final_fsq(fallback.result))
            fallback_conv = bool(ctx.result_meets_requested_ftol(fallback.result, ftol=float(requested_ftol)))
            if fallback_conv or fallback_fsq < best_fsq:
                best_run = fallback
                best_fsq = float(fallback_fsq)

    improvement_floor = np.finfo(float).eps * max(1.0, abs(float(best_fsq)), abs(float(target_fsq)))
    finish_budget = FinishBudgetTracker(cap=int(max_fallback_budget) if bool(ctx.max_iter_overridden) else None)
    accelerated_finish_uses_scan = False if ctx.use_scan is False else True

    if (
        bool(ctx.accelerated_mode)
        and str(initial_policy) == "single_grid"
        and (not bool(staged_followup_diag.used))
        and not bool(ctx.result_meets_requested_ftol(best_run.result, ftol=float(requested_ftol)))
    ):
        accel_budget_i = int(base_total_budget)
        accel_budget_used = 0
        while int(accel_budget_i) >= 1 and int(accel_budget_used) < int(max_fallback_budget):
            budget_this = finish_budget.bounded(accel_budget_i)
            if int(budget_this) <= 0:
                break
            prev_best_fsq = float(best_fsq)
            trial = _run_finish_attempt(
                ctx,
                best_run=best_run,
                target_fsq=target_fsq,
                budget_i=budget_this,
                mode_i="accelerated",
                use_scan_i=bool(accelerated_finish_uses_scan),
                performance_mode_i=True,
            )
            trial_fsq, trial_conv = attempt_log.record(
                ctx,
                trial,
                requested_ftol=requested_ftol,
                budget_i=budget_this,
                mode_i="accelerated",
            )
            accel_budget_used += int(budget_this)
            finish_budget.add(budget_this)
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
            budget_this = finish_budget.bounded(budget_i)
            if int(budget_this) <= 0:
                break
            prev_best_fsq = float(best_fsq)
            trial = _run_finish_attempt(
                ctx,
                best_run=best_run,
                target_fsq=target_fsq,
                budget_i=budget_this,
                mode_i="parity",
                use_scan_i=False,
                performance_mode_i=False,
            )
            trial_fsq, trial_conv = attempt_log.record(
                ctx,
                trial,
                requested_ftol=requested_ftol,
                budget_i=budget_this,
                mode_i="parity",
            )
            finish_budget.add(budget_this)
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
        fallback = _run_full_parity_fallback(
            ctx,
            max_fallback_budget=int(max_fallback_budget),
            multigrid=ctx.multigrid if bool(ctx.multigrid_user_provided) else None,
        )
        fallback_fsq = float(ctx.result_final_fsq(fallback.result))
        fallback_conv = bool(ctx.result_meets_requested_ftol(fallback.result, ftol=float(requested_ftol)))
        if fallback_conv or fallback_fsq < best_fsq:
            best_run = fallback
            best_fsq = float(fallback_fsq)

    return _finish_run_with_diagnostics(
        FinishDiagnosticInputs(
            ctx=ctx,
            best_run=best_run,
            requested_ftol=float(requested_ftol),
            target_fsq=float(target_fsq),
            base_diag=base_diag,
            initial_policy=str(initial_policy),
            partial_fallback_used=bool(partial_fallback_used),
            fallback_used=bool(fallback_used),
            staged_followup=staged_followup_diag,
            attempt_log=attempt_log,
            finish_budget=finish_budget,
        )
    )
