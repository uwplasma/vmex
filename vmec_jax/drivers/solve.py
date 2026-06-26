"""Driver helpers for lightweight fixed-boundary solve entry points."""

from __future__ import annotations

import os
from typing import Any, Callable


FIXED_BOUNDARY_OPTIMIZER_SOLVERS = frozenset({"gd", "lbfgs", "vmec_lbfgs", "vmec_gn"})


def initial_guess_with_optional_nojit(
    static,
    boundary,
    indata,
    *,
    vmec_project: bool,
    infer_axis_if_missing: bool,
    performance_mode: bool,
    force_disable_jit: bool = False,
    initial_guess_from_boundary_func: Callable,
    default_backend_name_func: Callable[[], str],
    getenv: Callable[[str, str], str] = os.getenv,
    contains_jax_tracer_func: Callable[[object], bool] | None = None,
    numpy_module_patch_func: Callable[[], object] | None = None,
):
    """Build the initial VMEC state with optional CPU NumPy/no-JIT fallbacks."""

    disable_env = getenv("VMEC_JAX_DISABLE_JIT_INIT", "") not in ("", "0")
    use_numpy_init = False
    if bool(performance_mode) and (default_backend_name_func() == "cpu") and not bool(force_disable_jit):
        env_numpy_init = getenv("VMEC_JAX_CPU_NUMPY_INIT_GUESS", "1").strip().lower()
        if env_numpy_init not in ("", "0", "false", "no"):
            try:
                if contains_jax_tracer_func is None:
                    from ..multigrid import _contains_jax_tracer as contains_jax_tracer_func

                use_numpy_init = not contains_jax_tracer_func(boundary)
            except Exception:
                use_numpy_init = False
    if use_numpy_init:
        try:
            if numpy_module_patch_func is None:
                from ..kernels.numpy_forces import _numpy_module_patch as numpy_module_patch_func

            with numpy_module_patch_func():
                return initial_guess_from_boundary_func(
                    static,
                    boundary,
                    indata,
                    vmec_project=bool(vmec_project),
                    infer_axis_if_missing=bool(infer_axis_if_missing),
                )
        except Exception:
            # Fall through to the standard JAX path if the NumPy-compatible
            # shim is missing an operation for an uncommon initialization.
            pass
    if not (disable_env or bool(force_disable_jit)):
        return initial_guess_from_boundary_func(
            static,
            boundary,
            indata,
            vmec_project=bool(vmec_project),
            infer_axis_if_missing=bool(infer_axis_if_missing),
        )
    try:
        import jax

        with jax.disable_jit():
            return initial_guess_from_boundary_func(
                static,
                boundary,
                indata,
                vmec_project=bool(vmec_project),
                infer_axis_if_missing=bool(infer_axis_if_missing),
            )
    except Exception:
        return initial_guess_from_boundary_func(
            static,
            boundary,
            indata,
            vmec_project=bool(vmec_project),
            infer_axis_if_missing=bool(infer_axis_if_missing),
        )


def solve_fixed_boundary_from_boundary(
    *,
    boundary,
    static,
    indata,
    flux,
    pressure,
    signgs: int,
    max_iter: int = 2,
    step_size: float = 5e-3,
    jacobian_penalty: float = 1e3,
    jit_grad: bool = False,
    differentiable: bool = True,
    stop_grad_in_update: bool = True,
    verbose: bool = False,
    vmec_project: bool = False,
    initial_guess_from_boundary_func: Callable,
    solve_fixed_boundary_gd_func: Callable,
):
    """Solve VMEC fixed-boundary starting from boundary coefficients."""

    st_guess = initial_guess_from_boundary_func(static, boundary, indata, vmec_project=vmec_project)
    res = solve_fixed_boundary_gd_func(
        st_guess,
        static,
        phipf=flux.phipf,
        chipf=flux.chipf,
        signgs=signgs,
        lamscale=flux.lamscale,
        pressure=pressure,
        gamma=float(indata.get_float("GAMMA", 0.0)),
        max_iter=int(max_iter),
        step_size=float(step_size),
        jacobian_penalty=float(jacobian_penalty),
        jit_grad=bool(jit_grad),
        differentiable=bool(differentiable),
        stop_grad_in_update=bool(stop_grad_in_update),
        verbose=bool(verbose),
    )
    return res.state


def fixed_boundary_initial_guess_run(
    *,
    cfg: Any,
    indata: Any,
    static: Any,
    boundary: Any,
    flux: Any,
    profiles: dict,
    signgs: int,
    restart_state: Any | None,
    initial_guess_func: Callable[[Any, Any], Any],
    maybe_dump_xc_init: Callable[..., None],
    run_container: Callable[..., Any],
) -> Any:
    """Return a driver run containing only the initial VMEC state."""

    if restart_state is not None:
        st0 = restart_state
    else:
        st0 = initial_guess_func(static, boundary)
        maybe_dump_xc_init(state=st0, static=static, label="init")
    return run_container(
        cfg=cfg,
        indata=indata,
        static=static,
        state=st0,
        result=None,
        flux=flux,
        profiles=profiles,
        signgs=signgs,
    )


def maybe_print_optimizer_summary(result: Any, *, solver: str, verbose: bool, print_func: Callable[..., None] = print) -> None:
    """Print the short non-VMEC2000 optimizer summary used by the driver."""

    if not bool(verbose) or str(solver) == "vmec2000_iter":
        return
    n_iter = int(getattr(result, "n_iter", -1))
    w_history = getattr(result, "w_history", None)
    w_final = float(w_history[-1]) if w_history is not None else float("nan")
    grad_history = getattr(result, "grad_rms_history", None)
    if grad_history is not None and len(grad_history) > 0:
        grad_final = float(grad_history[-1])
    else:
        grad_final = float("nan")
    print_func(f"[vmec_jax] finished: n_iter={n_iter} w={w_final:.8e} grad_rms={grad_final:.3e}")


def run_fixed_boundary_optimizer_solver(
    *,
    solver: str,
    restart_state: Any | None,
    static: Any,
    boundary: Any,
    indata: Any,
    flux: Any,
    pressure: Any,
    signgs: int,
    gamma: float,
    max_iter: int,
    step_size: float,
    history_size: int,
    gn_damping: float | None,
    gn_cg_tol: float | None,
    gn_cg_maxiter: int,
    verbose: bool,
    initial_guess_func: Callable[[Any, Any], Any],
    solve_fixed_boundary_gd_func: Callable[..., Any],
    solve_fixed_boundary_lbfgs_func: Callable[..., Any],
    solve_fixed_boundary_lbfgs_vmec_residual_func: Callable[..., Any] | None = None,
    solve_fixed_boundary_gn_vmec_residual_func: Callable[..., Any] | None = None,
) -> Any | None:
    """Run non-VMEC2000 fixed-boundary optimizer solvers.

    ``run_fixed_boundary`` owns input loading, staging, and VMEC2000 parity
    policy.  This helper isolates the simpler optimizer dispatch so that the
    public driver keeps one clear branch: optimizer-style solvers here,
    VMEC2000-style staged iteration in the main path.
    """

    solver_lower = str(solver).lower()
    if solver_lower not in FIXED_BOUNDARY_OPTIMIZER_SOLVERS:
        return None

    st0 = restart_state if restart_state is not None else initial_guess_func(static, boundary)
    if solver_lower == "gd":
        return solve_fixed_boundary_gd_func(
            st0,
            static,
            phipf=flux.phipf,
            chipf=flux.chipf,
            signgs=signgs,
            lamscale=flux.lamscale,
            pressure=pressure,
            gamma=gamma,
            max_iter=int(max_iter),
            step_size=float(step_size),
            jacobian_penalty=1e3,
            jit_grad=True,
            verbose=bool(verbose),
        )
    if solver_lower == "lbfgs":
        return solve_fixed_boundary_lbfgs_func(
            st0,
            static,
            phipf=flux.phipf,
            chipf=flux.chipf,
            signgs=signgs,
            lamscale=flux.lamscale,
            pressure=pressure,
            gamma=gamma,
            max_iter=int(max_iter),
            step_size=float(step_size),
            history_size=int(history_size),
            jit_grad=True,
            verbose=bool(verbose),
        )
    if solver_lower == "vmec_lbfgs":
        if solve_fixed_boundary_lbfgs_vmec_residual_func is None:
            from ..solve import solve_fixed_boundary_lbfgs_vmec_residual as solve_fixed_boundary_lbfgs_vmec_residual_func

        return solve_fixed_boundary_lbfgs_vmec_residual_func(
            st0,
            static,
            indata=indata,
            signgs=signgs,
            history_size=int(history_size),
            max_iter=int(max_iter),
            step_size=float(step_size),
            jit_grad=True,
            preconditioner="mode_diag+radial_tridi",
            precond_exponent=1.0,
            precond_radial_alpha=0.2,
            verbose=bool(verbose),
        )
    if solve_fixed_boundary_gn_vmec_residual_func is None:
        from ..solve import solve_fixed_boundary_gn_vmec_residual as solve_fixed_boundary_gn_vmec_residual_func

    return solve_fixed_boundary_gn_vmec_residual_func(
        st0,
        static,
        indata=indata,
        signgs=signgs,
        max_iter=int(max_iter),
        step_size=float(step_size),
        damping=None if gn_damping is None else float(gn_damping),
        cg_tol=None if gn_cg_tol is None else float(gn_cg_tol),
        cg_maxiter=int(gn_cg_maxiter),
        jit_kernels=True,
        verbose=bool(verbose),
    )


def run_fixed_boundary_stage_solve(
    *,
    state: Any,
    static: Any,
    solve_kwargs: dict[str, Any],
    jit_forces: bool,
    solve_fixed_boundary_residual_iter_func: Callable[..., Any],
) -> Any:
    """Run one VMEC2000-style stage with the requested JIT policy."""

    if not bool(jit_forces):
        try:
            import jax

            with jax.disable_jit():
                return solve_fixed_boundary_residual_iter_func(
                    state,
                    static,
                    jit_forces=False,
                    **solve_kwargs,
                )
        except Exception:
            return solve_fixed_boundary_residual_iter_func(
                state,
                static,
                jit_forces=False,
                **solve_kwargs,
            )
    return solve_fixed_boundary_residual_iter_func(
        state,
        static,
        jit_forces=True,
        **solve_kwargs,
    )


def maybe_precompile_fixed_boundary_stage(
    *,
    enabled: bool,
    state: Any,
    static: Any,
    solve_kwargs: dict[str, Any],
    solve_fixed_boundary_residual_iter_func: Callable[..., Any],
) -> None:
    """Optionally precompile a single-stage VMEC2000 iteration."""

    if not bool(enabled):
        return
    try:
        precompile_kwargs = dict(solve_kwargs)
        precompile_kwargs.update(
            {
                "precompile_only": True,
                "verbose": False,
                "verbose_vmec2000_table": False,
                "jit_warmup_iters": 0,
                "jit_precompile": True,
                "max_iter": 1,
            }
        )
        solve_fixed_boundary_residual_iter_func(
            state,
            static,
            jit_forces=True,
            **precompile_kwargs,
        )
    except Exception:
        return


def maybe_rerun_scan_abort_stage(
    *,
    result: Any,
    enabled: bool,
    state_stage_start: Any,
    resume_state_stage: Any,
    solve_kwargs: dict[str, Any],
    jit_warmup_noscan: int,
    jit_precompile_noscan: bool,
    jit_forces_base: bool,
    run_stage_solve_func: Callable[..., Any],
    verbose: bool,
    print_func: Callable[..., None] = print,
) -> Any:
    """Rerun a bad scan stage in non-scan parity mode when requested."""

    if not bool(enabled):
        return result
    try:
        if not (
            bool(result.diagnostics.get("vmec2000_scan", False))
            and bool(result.diagnostics.get("abort_scan", False))
        ):
            return result
        if bool(verbose):
            print_func(
                "[vmec_jax] scan abort detected; rerunning stage in parity mode.",
                flush=True,
            )
        solve_kwargs_fallback = dict(solve_kwargs)
        solve_kwargs_fallback.update(
            {
                "use_scan": False,
                "resume_state": resume_state_stage,
                "jit_warmup_iters": int(jit_warmup_noscan),
                "jit_precompile": bool(jit_precompile_noscan),
            }
        )
        return run_stage_solve_func(
            state=state_stage_start,
            solve_kwargs=solve_kwargs_fallback,
            jit_forces=bool(jit_forces_base),
        )
    except Exception:
        return result


def maybe_apply_scan_wout_corrector(
    *,
    result: Any,
    stage_results: list[Any],
    scan_wout_corrector: bool | None,
    accelerated_mode: bool,
    static_prev: Any,
    build_static_func: Callable[[Any], Any],
    cfg: Any,
    ftol_last: float | None,
    step_size_last: float | None,
    indata: Any,
    signgs: int,
    use_restart_triggers: bool | None,
    vmecpp_restart: bool,
    stage_transition_factor: float,
    stage_transition_scale: float,
    scan_minimal_default: bool | None,
    jit_forces: Any,
    resolve_jit_forces: Callable[[Any, Any, int], bool],
    solve_fixed_boundary_residual_iter_func: Callable[..., Any],
    accelerated_fsq_total_target_from_ftol: Callable[[float], float],
    copy_final_force_payload: Callable[..., Any],
    getenv: Callable[[str, str], str] = os.getenv,
) -> Any:
    """Optionally run one non-scan corrector step before writing WOUT output."""

    try:
        use_scan_any = any(bool(r.diagnostics.get("vmec2000_scan", False)) for r in stage_results)
    except Exception:
        use_scan_any = False
    if bool(accelerated_mode) and scan_wout_corrector is None:
        scan_wout_corrector = False
    if scan_wout_corrector is None:
        scan_wout_env = getenv("VMEC_JAX_SCAN_WOUT_CORRECTOR", "0").strip().lower()
        scan_wout_corrector = scan_wout_env not in ("", "0", "false", "no")
    if not (bool(use_scan_any) and bool(scan_wout_corrector)):
        return result

    try:
        from ..solve import SolveVmecResidualResult

        resume_state_corr = result.diagnostics.get("resume_state", None)
        static_corr = static_prev if static_prev is not None else build_static_func(cfg)
        ftol_corr = float(ftol_last) if ftol_last is not None else float(indata.get_float("FTOL", 1e-13))
        step_corr = float(step_size_last) if step_size_last is not None else 1.0
        corr_kwargs = dict(
            indata=indata,
            signgs=signgs,
            ftol=ftol_corr,
            max_iter=1,
            step_size=step_corr,
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
            use_restart_triggers=True if use_restart_triggers is None else bool(use_restart_triggers),
            vmecpp_restart=bool(vmecpp_restart),
            stage_prev_fsq=None,
            stage_transition_factor=float(stage_transition_factor),
            stage_transition_scale=float(stage_transition_scale),
            use_direct_fallback=False,
            resume_state=resume_state_corr,
            verbose=False,
            verbose_vmec2000_table=False,
            jit_precompile=False,
            jit_warmup_iters=0,
            use_scan=False,
            scan_minimal_default=scan_minimal_default,
            light_history=True if accelerated_mode else None,
            resume_state_mode="minimal" if accelerated_mode else None,
            fsq_total_target=(
                accelerated_fsq_total_target_from_ftol(float(ftol_corr)) if bool(accelerated_mode) else None
            ),
            return_final_force_payload=True,
        )
        res_corr = solve_fixed_boundary_residual_iter_func(
            result.state,
            static_corr,
            jit_forces=resolve_jit_forces(jit_forces, static_corr, 1),
            **corr_kwargs,
        )
        diag = dict(result.diagnostics)
        diag["scan_wout_corrector"] = True
        diag["scan_wout_corrector_iters"] = int(res_corr.n_iter)
        corrected = SolveVmecResidualResult(
            state=res_corr.state,
            n_iter=result.n_iter,
            w_history=result.w_history,
            fsqr2_history=result.fsqr2_history,
            fsqz2_history=result.fsqz2_history,
            fsql2_history=result.fsql2_history,
            grad_rms_history=result.grad_rms_history,
            step_history=result.step_history,
            diagnostics=diag,
        )
        return copy_final_force_payload(corrected, res_corr)
    except Exception:
        return result


def maybe_run_fixed_boundary_in_solver_device_context(
    *,
    input_path: Any,
    solver_device: str | None,
    solver_device_context_active: bool,
    cfg: Any,
    indata: Any,
    solver_lower: str,
    cli_fixed_boundary_mode: bool,
    accelerated_mode: bool,
    restart_state_present: bool,
    restart_solver_state_present: bool,
    recursive_run_kwargs: dict[str, Any],
    run_fixed_boundary_func: Callable[..., Any],
    replace_func: Callable[..., Any],
    as_list_like_func: Callable[[Any], Any],
    default_backend_name_func: Callable[[], str],
    resolve_solver_device_name_func: Callable[..., str | None],
    getenv: Callable[[str, str | None], str | None] = os.getenv,
) -> Any | None:
    """Run a fixed-boundary solve inside a requested JAX device context.

    The public driver owns policy resolution and restart setup.  This helper
    owns the narrow reroute mechanics: resolve a device name, enter JAX's
    default-device context, recursively call the driver once, and annotate the
    returned diagnostics.  Returning ``None`` means no reroute was possible or
    requested, so the caller should continue with the ordinary path.
    """

    backend_for_device = default_backend_name_func()
    solver_device_name = resolve_solver_device_name_func(
        solver_device=solver_device,
        backend=backend_for_device,
        cfg=cfg,
        indata=indata,
        solver_lower=solver_lower,
        cli_fixed_boundary_mode=bool(cli_fixed_boundary_mode),
        accelerated_mode=bool(accelerated_mode),
        ns_list_input=as_list_like_func(indata.get("NS_ARRAY", None)),
        niter_list_input=as_list_like_func(indata.get("NITER_ARRAY", None)),
        restart_state_present=bool(restart_state_present),
        restart_solver_state_present=bool(restart_solver_state_present),
    )
    if solver_device_name is None or bool(solver_device_context_active):
        return None
    try:
        import jax

        devices = jax.devices(str(solver_device_name))
    except Exception:
        devices = []
    if not devices:
        return None

    from ..kernels.tomnsp import tomnsps_fft_policy_override

    tomnsps_fft_override = (
        str(solver_device_name).strip().lower() in ("gpu", "cuda", "rocm", "tpu")
        if getenv("VMEC_JAX_TOMNSPS_FFT", None) is None
        else None
    )
    run_kwargs = dict(recursive_run_kwargs)
    run_kwargs["solver_device"] = str(solver_device_name)
    run_kwargs["_solver_device_context_active"] = True
    with jax.default_device(devices[0]), tomnsps_fft_policy_override(tomnsps_fft_override):
        routed_run = run_fixed_boundary_func(input_path, **run_kwargs)
    if getattr(routed_run, "result", None) is not None:
        diag = dict(getattr(routed_run.result, "diagnostics", {}) or {})
        diag["solver_device"] = str(solver_device_name)
        diag["solver_device_auto_reroute"] = (str(solver_device or "auto").strip().lower() == "auto")
        diag["solver_device_requested_backend"] = str(backend_for_device)
        routed_run = replace_func(routed_run, result=replace_func(routed_run.result, diagnostics=diag))
    return routed_run
