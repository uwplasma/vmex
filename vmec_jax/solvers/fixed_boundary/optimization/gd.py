"""Gradient-descent fixed-boundary magnetic-energy optimizer."""

from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np

from ..results import SolveFixedBoundaryResult
from ....state import VMECState


def solve_fixed_boundary_gd_impl(
    state0: VMECState,
    static,
    *,
    phipf,
    chipf,
    signgs: int,
    lamscale,
    edge_Rcos: Any | None = None,
    edge_Rsin: Any | None = None,
    edge_Zcos: Any | None = None,
    edge_Zsin: Any | None = None,
    pressure: Any | None = None,
    gamma: float = 0.0,
    jacobian_penalty: float = 1e3,
    max_iter: int = 25,
    step_size: float = 5e-3,
    scale_rz: float = 1.0,
    scale_l: float = 1.0,
    grad_tol: float | None = None,
    max_backtracks: int = 16,
    bt_factor: float = 0.5,
    jit_grad: bool = False,
    preconditioner: str = "none",
    precond_exponent: float = 1.0,
    precond_radial_alpha: float = 0.0,
    differentiable: bool = False,
    stop_grad_in_update: bool = False,
    verbose: bool = True,
    has_jax_func: Callable[[], bool] | None = None,
    validate_options_func: Callable[..., Any] | None = None,
    prepare_energy_context_func: Callable[..., Any] | None = None,
    enforce_fixed_boundary_and_axis_func: Callable[..., VMECState] | None = None,
    mask_grad_for_constraints_func: Callable[..., VMECState] | None = None,
    apply_preconditioner_func: Callable[..., VMECState] | None = None,
    update_state_gd_func: Callable[..., VMECState] | None = None,
    grad_rms_state_func: Callable[..., float] | None = None,
    resolve_grad_tol_func: Callable[..., float] | None = None,
    mode00_index_func: Callable[..., int] | None = None,
    eval_geom_func: Callable[..., Any] | None = None,
    bsup_from_geom_func: Callable[..., tuple[Any, Any]] | None = None,
    b2_from_bsup_func: Callable[..., Any] | None = None,
    angle_steps_func: Callable[..., tuple[float, float]] | None = None,
    validate_pressure_shape_func: Callable[..., Any] | None = None,
    jax_module: Any | None = None,
    jnp_module: Any | None = None,
    jit_func: Callable[..., Any] | None = None,
) -> SolveFixedBoundaryResult:
    """Minimize fixed-boundary magnetic energy with gradient descent."""

    if has_jax_func is None or jax_module is None or jnp_module is None or jit_func is None:
        from ...._compat import has_jax as _has_jax
        from ...._compat import jax as _jax
        from ...._compat import jit as _jit
        from ...._compat import jnp as _jnp

        has_jax_func = _has_jax if has_jax_func is None else has_jax_func
        jax_module = _jax if jax_module is None else jax_module
        jnp_module = _jnp if jnp_module is None else jnp_module
        jit_func = _jit if jit_func is None else jit_func

    if not has_jax_func():
        raise ImportError("solve_fixed_boundary_gd requires JAX (jax + jaxlib)")

    if validate_options_func is None:
        from ..options import validate_fixed_boundary_gd_options as validate_options_func
    if prepare_energy_context_func is None:
        from .energy import (
            prepare_fixed_boundary_energy_context as prepare_energy_context_func,
        )
    if enforce_fixed_boundary_and_axis_func is None:
        from .constraints import (
            enforce_fixed_boundary_and_axis as enforce_fixed_boundary_and_axis_func,
        )
    if mask_grad_for_constraints_func is None:
        from .gradient import mask_grad_for_constraints as mask_grad_for_constraints_func
    if apply_preconditioner_func is None:
        from ..preconditioning.operators import (
            apply_preconditioner as apply_preconditioner_func,
        )
    if update_state_gd_func is None:
        from .gradient import update_state_gd as update_state_gd_func
    if grad_rms_state_func is None:
        from .constraints import grad_rms_state as grad_rms_state_func
    if resolve_grad_tol_func is None:
        from .tolerances import resolve_grad_tol as resolve_grad_tol_func

    opts = validate_options_func(
        max_iter=max_iter,
        max_backtracks=max_backtracks,
        bt_factor=bt_factor,
        gamma=gamma,
    )
    max_iter = opts.max_iter
    max_backtracks = opts.max_backtracks
    bt_factor = opts.bt_factor
    gamma = opts.gamma

    energy = prepare_energy_context_func(
        state0,
        static,
        phipf=phipf,
        chipf=chipf,
        signgs=signgs,
        lamscale=lamscale,
        edge_Rcos=edge_Rcos,
        edge_Rsin=edge_Rsin,
        edge_Zcos=edge_Zcos,
        edge_Zsin=edge_Zsin,
        pressure=pressure,
        gamma=gamma,
        jacobian_penalty=jacobian_penalty,
        jit_grad=jit_grad,
        mode00_index_func=mode00_index_func,
        eval_geom_func=eval_geom_func,
        bsup_from_geom_func=bsup_from_geom_func,
        b2_from_bsup_func=b2_from_bsup_func,
        angle_steps_func=angle_steps_func,
        validate_pressure_shape_func=validate_pressure_shape_func,
        jax_module=jax_module,
        jnp_module=jnp_module,
        jit_func=jit_func,
    )
    idx00 = energy.idx00
    signgs = energy.signgs
    gamma = energy.gamma
    edge_Rcos = energy.edge_Rcos
    edge_Rsin = energy.edge_Rsin
    edge_Zcos = energy.edge_Zcos
    edge_Zsin = energy.edge_Zsin
    obj_and_grad = energy.objective_and_grad
    objective = energy.objective
    w_terms = energy.w_terms

    # Start from a constraint-satisfying state.
    state = enforce_fixed_boundary_and_axis_func(
        state0,
        static,
        edge_Rcos=edge_Rcos,
        edge_Rsin=edge_Rsin,
        edge_Zcos=edge_Zcos,
        edge_Zsin=edge_Zsin,
        enforce_lambda_axis=False,
        idx00=idx00,
    )

    grad_tol_eff: float | None = None

    if differentiable:
        wb_history = []
        wp_history = []
        w_history = []
        grad_rms_history = []
        step_history = []

        def _grad_rms_jax(grad_state: VMECState):
            g = (
                jnp_module.asarray(grad_state.Rcos) ** 2
                + jnp_module.asarray(grad_state.Rsin) ** 2
                + jnp_module.asarray(grad_state.Zcos) ** 2
                + jnp_module.asarray(grad_state.Zsin) ** 2
                + jnp_module.asarray(grad_state.Lcos) ** 2
                + jnp_module.asarray(grad_state.Lsin) ** 2
            )
            return jnp_module.sqrt(jnp_module.mean(g))

        for _ in range(max_iter):
            wb_t, wp_t, w_t = w_terms(state)
            w_history.append(w_t)
            wb_history.append(wb_t)
            wp_history.append(wp_t)

            _obj_t, grad_t = obj_and_grad(state)
            grad_t = mask_grad_for_constraints_func(grad_t, static, idx00=idx00)
            grad_t = apply_preconditioner_func(
                grad_t,
                static,
                kind=preconditioner,
                exponent=precond_exponent,
                radial_alpha=precond_radial_alpha,
            )
            if stop_grad_in_update:
                grad_t = jax_module.lax.stop_gradient(grad_t)
            grad_rms_history.append(_grad_rms_jax(grad_t))
            step_history.append(jnp_module.asarray(step_size, dtype=jnp_module.asarray(state.Rcos).dtype))

            state = update_state_gd_func(
                state,
                grad_t,
                step=step_size,
                scale_rz=scale_rz,
                scale_l=scale_l,
            )
            state = enforce_fixed_boundary_and_axis_func(
                state,
                static,
                edge_Rcos=edge_Rcos,
                edge_Rsin=edge_Rsin,
                edge_Zcos=edge_Zcos,
                edge_Zsin=edge_Zsin,
                idx00=idx00,
            )
    else:
        wb0, wp0, w0 = w_terms(state)
        wb0 = float(np.asarray(wb0))
        wp0 = float(np.asarray(wp0))
        w0 = float(np.asarray(w0))
        wb_history = [wb0]
        wp_history = [wp0]
        grad_rms_history = []
        step_history = []

        obj0, grad0 = obj_and_grad(state)
        obj0 = float(np.asarray(obj0))
        w_history = [obj0]

        for it in range(max_iter):
            grad0m = mask_grad_for_constraints_func(grad0, static, idx00=idx00)
            grad_raw = grad0m
            grad0m = apply_preconditioner_func(
                grad0m,
                static,
                kind=preconditioner,
                exponent=precond_exponent,
                radial_alpha=precond_radial_alpha,
            )
            grad_rms = grad_rms_state_func(grad0m)
            grad_rms_history.append(grad_rms)
            if grad_tol_eff is None:
                grad_tol_eff = resolve_grad_tol_func(
                    grad_tol,
                    grad_rms0=grad_rms,
                    dtype=np.asarray(state.Rcos).dtype,
                )

            if verbose:
                print(f"[solve_fixed_boundary_gd] iter={it:03d} w={w_history[-1]:.8e} grad_rms={grad_rms:.3e}")

            if grad_rms < float(grad_tol_eff):
                break

            step = float(step_size)

            def _try_line_search(grad_step):
                step_local = float(step_size)
                for bt in range(max_backtracks + 1):
                    if bt > 0:
                        step_local *= bt_factor
                    trial = update_state_gd_func(
                        state,
                        grad_step,
                        step=step_local,
                        scale_rz=scale_rz,
                        scale_l=scale_l,
                    )
                    trial = enforce_fixed_boundary_and_axis_func(
                        trial,
                        static,
                        edge_Rcos=edge_Rcos,
                        edge_Rsin=edge_Rsin,
                        edge_Zcos=edge_Zcos,
                        edge_Zsin=edge_Zsin,
                        idx00=idx00,
                    )
                    obj_t = objective(trial)
                    obj_t = float(np.asarray(obj_t))
                    if np.isfinite(obj_t) and obj_t < w_history[-1]:
                        return True, trial, obj_t, step_local
                return False, None, None, step_local

            accepted, trial, obj_t, step = _try_line_search(grad0m)
            if not accepted and preconditioner != "none":
                accepted, trial, obj_t, step = _try_line_search(grad_raw)
                if accepted and verbose:
                    print("[solve_fixed_boundary_gd] fallback to unpreconditioned gradient")

            step_history.append(step)

            if not accepted:
                if verbose:
                    print("[solve_fixed_boundary_gd] line search failed to improve objective; stopping")
                break

            state = trial
            obj0 = obj_t

            wb_t, wp_t, _w_t = w_terms(state)
            w_history.append(obj0)
            wb_history.append(float(np.asarray(wb_t)))
            wp_history.append(float(np.asarray(wp_t)))

            obj0, grad0 = obj_and_grad(state)

    diag: Dict[str, Any] = {
        "idx00": idx00,
        "signgs": signgs,
        "gamma": gamma,
        "jacobian_penalty": float(jacobian_penalty),
        "scale_rz": float(scale_rz),
        "scale_l": float(scale_l),
        "preconditioner": str(preconditioner),
        "precond_exponent": float(precond_exponent),
        "precond_radial_alpha": float(precond_radial_alpha),
        "grad_tol": None if grad_tol_eff is None else float(grad_tol_eff),
    }
    if differentiable:
        return SolveFixedBoundaryResult(
            state=state,
            n_iter=len(w_history),
            w_history=jnp_module.asarray(w_history),
            wb_history=jnp_module.asarray(wb_history),
            wp_history=jnp_module.asarray(wp_history),
            grad_rms_history=jnp_module.asarray(grad_rms_history),
            step_history=jnp_module.asarray(step_history),
            diagnostics=diag,
        )
    return SolveFixedBoundaryResult(
        state=state,
        n_iter=len(w_history) - 1,
        w_history=np.asarray(w_history, dtype=float),
        wb_history=np.asarray(wb_history, dtype=float),
        wp_history=np.asarray(wp_history, dtype=float),
        grad_rms_history=np.asarray(grad_rms_history, dtype=float),
        step_history=np.asarray(step_history, dtype=float),
        diagnostics=diag,
    )
