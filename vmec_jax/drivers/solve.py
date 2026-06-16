"""Driver helpers for lightweight fixed-boundary solve entry points."""

from __future__ import annotations

import os
from typing import Callable


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
