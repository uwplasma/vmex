"""L-BFGS fixed-boundary magnetic-energy optimizer."""

from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np

from .solve_result_types import SolveFixedBoundaryResult
from .state import VMECState


def solve_fixed_boundary_lbfgs_impl(
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
    history_size: int = 10,
    max_iter: int = 40,
    step_size: float = 1.0,
    grad_tol: float | None = None,
    max_backtracks: int = 12,
    bt_factor: float = 0.5,
    jit_grad: bool = False,
    preconditioner: str = "none",
    precond_exponent: float = 1.0,
    precond_radial_alpha: float = 0.0,
    verbose: bool = True,
    has_jax_func: Callable[[], bool] | None = None,
    validate_options_func: Callable[..., Any] | None = None,
    prepare_energy_context_func: Callable[..., Any] | None = None,
    enforce_fixed_boundary_and_axis_func: Callable[..., VMECState] | None = None,
    mask_grad_for_constraints_func: Callable[..., VMECState] | None = None,
    apply_preconditioner_func: Callable[..., VMECState] | None = None,
    grad_rms_state_func: Callable[..., float] | None = None,
    resolve_grad_tol_func: Callable[..., float] | None = None,
    lbfgs_two_loop_direction_func: Callable[..., Any] | None = None,
    ensure_descent_direction_func: Callable[..., tuple[Any, Any, bool]] | None = None,
    resolve_lbfgs_curvature_tol_func: Callable[..., float] | None = None,
    pack_state_func: Callable[..., Any] | None = None,
    unpack_state_func: Callable[..., VMECState] | None = None,
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
    """Minimize fixed-boundary magnetic energy with L-BFGS."""

    if has_jax_func is None or jnp_module is None or jit_func is None:
        from ._compat import has_jax as _has_jax
        from ._compat import jit as _jit
        from ._compat import jnp as _jnp

        has_jax_func = _has_jax if has_jax_func is None else has_jax_func
        jnp_module = _jnp if jnp_module is None else jnp_module
        jit_func = _jit if jit_func is None else jit_func

    if not has_jax_func():
        raise ImportError("solve_fixed_boundary_lbfgs requires JAX (jax + jaxlib)")

    if validate_options_func is None:
        from .solve_options import validate_fixed_boundary_lbfgs_options as validate_options_func
    if prepare_energy_context_func is None:
        from .solve_fixed_boundary_energy_helpers import (
            prepare_fixed_boundary_energy_context as prepare_energy_context_func,
        )
    if enforce_fixed_boundary_and_axis_func is None:
        from .solve_constraint_helpers import (
            enforce_fixed_boundary_and_axis as enforce_fixed_boundary_and_axis_func,
        )
    if mask_grad_for_constraints_func is None:
        from .solve_gradient_helpers import mask_grad_for_constraints as mask_grad_for_constraints_func
    if apply_preconditioner_func is None:
        from .solve_preconditioner_helpers import apply_preconditioner as apply_preconditioner_func
    if grad_rms_state_func is None:
        from .solve_constraint_helpers import grad_rms_state as grad_rms_state_func
    if resolve_grad_tol_func is None:
        from .solve_tolerance_helpers import resolve_grad_tol as resolve_grad_tol_func
    if lbfgs_two_loop_direction_func is None:
        from .solve_optimizer_helpers import lbfgs_two_loop_direction as lbfgs_two_loop_direction_func
    if ensure_descent_direction_func is None:
        from .solve_optimizer_helpers import ensure_descent_direction as ensure_descent_direction_func
    if resolve_lbfgs_curvature_tol_func is None:
        from .solve_optimizer_helpers import (
            lbfgs_curvature_tolerance as resolve_lbfgs_curvature_tol_func,
        )
    if pack_state_func is None or unpack_state_func is None:
        from .state import pack_state as _pack_state
        from .state import unpack_state as _unpack_state

        pack_state_func = _pack_state if pack_state_func is None else pack_state_func
        unpack_state_func = _unpack_state if unpack_state_func is None else unpack_state_func

    opts = validate_options_func(
        history_size=history_size,
        max_iter=max_iter,
        max_backtracks=max_backtracks,
        bt_factor=bt_factor,
        gamma=gamma,
    )
    history_size = opts.history_size
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
        jacobian_penalty=0.0,
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
    w_and_grad = energy.objective_and_grad
    w_terms = energy.w_terms_and_jacmin

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

    wb0, wp0, w0, jacmin0 = w_terms(state)
    w0 = float(np.asarray(w0))
    wb0 = float(np.asarray(wb0))
    wp0 = float(np.asarray(wp0))
    jacmin0 = float(np.asarray(jacmin0))
    if not np.isfinite(w0) or jacmin0 <= 0.0:
        raise ValueError("Initial state has invalid Jacobian sign or non-finite energy")

    w_history = [w0]
    wb_history = [wb0]
    wp_history = [wp0]
    grad_rms_history = []
    step_history = []

    _w_val, grad = w_and_grad(state)
    grad = mask_grad_for_constraints_func(grad, static, idx00=idx00, mask_lambda_axis=False)
    grad = apply_preconditioner_func(
        grad,
        static,
        kind=preconditioner,
        exponent=precond_exponent,
        radial_alpha=precond_radial_alpha,
    )

    x = pack_state_func(state)
    g_flat = pack_state_func(grad)

    s_hist: list[Any] = []
    y_hist: list[Any] = []

    step0 = float(step_size)
    grad_tol_eff: float | None = None

    for it in range(max_iter):
        grad_rms = grad_rms_state_func(grad)
        grad_rms_history.append(grad_rms)
        if grad_tol_eff is None:
            grad_tol_eff = resolve_grad_tol_func(
                grad_tol,
                grad_rms0=grad_rms,
                dtype=np.asarray(state.Rcos).dtype,
            )

        if verbose:
            print(f"[solve_fixed_boundary_lbfgs] iter={it:03d} w={w_history[-1]:.8e} grad_rms={grad_rms:.3e}")

        if grad_rms < float(grad_tol_eff):
            break

        p_flat = lbfgs_two_loop_direction_func(g_flat, s_hist, y_hist)
        p_flat, _gtp, _fallback_to_descent = ensure_descent_direction_func(g_flat, p_flat)

        accepted = False
        step = step0

        x_old = x
        g_old = g_flat

        for bt in range(max_backtracks + 1):
            if bt > 0:
                step *= bt_factor
            x_try = x_old + jnp_module.asarray(step, dtype=x_old.dtype) * p_flat
            st_try = unpack_state_func(x_try, state.layout)
            st_try = enforce_fixed_boundary_and_axis_func(
                st_try,
                static,
                edge_Rcos=edge_Rcos,
                edge_Rsin=edge_Rsin,
                edge_Zcos=edge_Zcos,
                edge_Zsin=edge_Zsin,
                enforce_lambda_axis=False,
                idx00=idx00,
            )

            wb_t, wp_t, w_t, jacmin_t = w_terms(st_try)
            w_tf = float(np.asarray(w_t))
            jacmin_tf = float(np.asarray(jacmin_t))
            if np.isfinite(w_tf) and jacmin_tf > 0.0 and w_tf < w_history[-1]:
                state = st_try
                x = pack_state_func(state)
                accepted = True
                break

        step_history.append(step)

        if not accepted:
            if verbose:
                print("[solve_fixed_boundary_lbfgs] line search failed; stopping")
            break

        # New value/grad at accepted state.
        wb_t, wp_t, w_t, _jacmin_t = w_terms(state)
        w_history.append(float(np.asarray(w_t)))
        wb_history.append(float(np.asarray(wb_t)))
        wp_history.append(float(np.asarray(wp_t)))

        _w_val, grad_new = w_and_grad(state)
        grad_new = mask_grad_for_constraints_func(grad_new, static, idx00=idx00)
        grad_new = apply_preconditioner_func(
            grad_new,
            static,
            kind=preconditioner,
            exponent=precond_exponent,
            radial_alpha=precond_radial_alpha,
        )
        g_flat_new = pack_state_func(grad_new)

        s_k = x - x_old
        y_k = g_flat_new - g_old
        ys = float(np.asarray(jnp_module.dot(y_k, s_k)))
        if np.isfinite(ys) and ys > resolve_lbfgs_curvature_tol_func(s_k, y_k):
            s_hist.append(s_k)
            y_hist.append(y_k)
            if len(s_hist) > history_size:
                s_hist.pop(0)
                y_hist.pop(0)

        grad = grad_new
        g_flat = g_flat_new
        step0 = float(step)

    diag: Dict[str, Any] = {
        "idx00": idx00,
        "signgs": signgs,
        "gamma": gamma,
        "history_size": int(history_size),
        "preconditioner": str(preconditioner),
        "precond_exponent": float(precond_exponent),
        "precond_radial_alpha": float(precond_radial_alpha),
        "grad_tol": None if grad_tol_eff is None else float(grad_tol_eff),
    }
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
