"""Driver helpers for lightweight fixed-boundary solve entry points."""

from __future__ import annotations

from typing import Callable


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
