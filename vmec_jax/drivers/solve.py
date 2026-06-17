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
                from ..vmec_numpy_forces import _numpy_module_patch as numpy_module_patch_func

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

    from ..vmec_tomnsp import tomnsps_fft_policy_override

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
