"""L-BFGS optimizer for VMEC-style fixed-boundary residual objectives."""

from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np

from .solve_result_types import SolveVmecResidualResult
from .state import VMECState


def solve_fixed_boundary_lbfgs_vmec_residual_impl(
    state0: VMECState,
    static,
    *,
    indata,
    signgs: int,
    w_rz: float = 1.0,
    w_l: float = 1.0,
    include_constraint_force: bool = True,
    objective_scale: float | None = None,
    apply_m1_constraints: bool = True,
    history_size: int = 10,
    max_iter: int = 40,
    step_size: float = 1.0,
    scale_rz: float = 1.0,
    scale_l: float = 1.0,
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
    prepare_residual_force_context_func: Callable[..., Any] | None = None,
    mode00_index_func: Callable[..., int] | None = None,
    half_mesh_from_full_mesh_func: Callable[..., Any] | None = None,
    mass_half_mesh_from_indata_func: Callable[..., Any] | None = None,
    pressure_half_mesh_from_indata_func: Callable[..., Any] | None = None,
    icurv_full_mesh_from_indata_func: Callable[..., Any] | None = None,
    vmec_force_flux_profiles_func: Callable[..., tuple[Any, Any, Any]] | None = None,
    wout_like_cls: type | None = None,
    assemble_residual_objective_terms_func: Callable[..., Any] | None = None,
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
    jax_module: Any | None = None,
    jnp_module: Any | None = None,
    jit_func: Callable[..., Any] | None = None,
) -> SolveVmecResidualResult:
    """Minimize VMEC-style fixed-boundary force residuals with L-BFGS."""

    if has_jax_func is None or jax_module is None or jnp_module is None or jit_func is None:
        from ._compat import has_jax as _has_jax
        from ._compat import jax as _jax
        from ._compat import jit as _jit
        from ._compat import jnp as _jnp

        has_jax_func = _has_jax if has_jax_func is None else has_jax_func
        jax_module = _jax if jax_module is None else jax_module
        jnp_module = _jnp if jnp_module is None else jnp_module
        jit_func = _jit if jit_func is None else jit_func

    if not has_jax_func():
        raise ImportError("solve_fixed_boundary_lbfgs_vmec_residual requires JAX (jax + jaxlib)")

    if validate_options_func is None:
        from .solve_options import validate_residual_lbfgs_options as validate_options_func
    if prepare_residual_force_context_func is None:
        from .solve_residual_force_context import (
            prepare_residual_force_context as prepare_residual_force_context_func,
        )
    if mode00_index_func is None:
        from .solve_constraint_helpers import mode00_index as mode00_index_func
    if assemble_residual_objective_terms_func is None:
        from .solve_residual_objective_helpers import (
            assemble_residual_objective_terms as assemble_residual_objective_terms_func,
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
        w_rz=w_rz,
        w_l=w_l,
        objective_scale=objective_scale,
        scale_rz=scale_rz,
        scale_l=scale_l,
        history_size=history_size,
        max_iter=max_iter,
        max_backtracks=max_backtracks,
        bt_factor=bt_factor,
    )
    w_rz = opts.w_rz
    w_l = opts.w_l
    objective_scale = opts.objective_scale
    scale_rz = opts.scale_rz
    scale_l = opts.scale_l
    history_size = opts.history_size
    max_iter = opts.max_iter
    max_backtracks = opts.max_backtracks
    bt_factor = opts.bt_factor

    idx00 = mode00_index_func(static.modes)
    residual_context = prepare_residual_force_context_func(
        state0,
        static,
        indata=indata,
        signgs=signgs,
        idx00=idx00,
        include_constraint_force=bool(include_constraint_force),
        mode00_index_func=mode00_index_func,
        half_mesh_from_full_mesh_func=half_mesh_from_full_mesh_func,
        mass_half_mesh_from_indata_func=mass_half_mesh_from_indata_func,
        pressure_half_mesh_from_indata_func=pressure_half_mesh_from_indata_func,
        icurv_full_mesh_from_indata_func=icurv_full_mesh_from_indata_func,
        vmec_force_flux_profiles_func=vmec_force_flux_profiles_func,
        wout_like_cls=wout_like_cls,
        jnp_module=jnp_module,
    )
    idx00 = residual_context.idx00
    signgs = residual_context.signgs
    s = residual_context.s
    wout_like = residual_context.wout_like
    trig = residual_context.trig
    constraint_tcon0 = residual_context.constraint_tcon0
    apply_lforbal = residual_context.apply_lforbal
    ftol_target = residual_context.ftol_target
    edge_Rcos = residual_context.edge_Rcos
    edge_Rsin = residual_context.edge_Rsin
    edge_Zcos = residual_context.edge_Zcos
    edge_Zsin = residual_context.edge_Zsin
    mask_pack = residual_context.mask_pack

    from .vmec_forces import vmec_forces_rz_from_wout, vmec_residual_internal_from_kernels
    from .vmec_residue import (
        vmec_force_norms_from_bcovar_dynamic,
    )

    objective_scale_f = float(objective_scale) if objective_scale is not None else None

    def _build_terms_fn(scale: float | None):
        def _fsq2_terms_and_jacmin(state: VMECState, zero_m1_zforce: Any):
            k = vmec_forces_rz_from_wout(
                state=state,
                static=static,
                wout=wout_like,
                indata=None,
                constraint_tcon0=constraint_tcon0,
                use_vmec_synthesis=True,
                trig=trig,
            )
            rzl = vmec_residual_internal_from_kernels(
                k,
                cfg_ntheta=int(static.cfg.ntheta),
                cfg_nzeta=int(static.cfg.nzeta),
                wout=wout_like,
                trig=trig,
                apply_lforbal=apply_lforbal,
                include_edge=False,
                masks=mask_pack,
            )
            norms = vmec_force_norms_from_bcovar_dynamic(bc=k.bc, trig=trig, s=s, signgs=signgs)
            terms = assemble_residual_objective_terms_func(
                frzl=rzl,
                norms=norms,
                s=s,
                w_rz=w_rz,
                w_l=w_l,
                zero_m1_zforce=zero_m1_zforce,
                lconm1=bool(getattr(static.cfg, "lconm1", True)),
                apply_m1_constraints=bool(apply_m1_constraints),
                zero_m1_after_m1_constraints=False,
                include_edge=False,
                apply_scalxc=True,
                zero_edge_rz_blocks=False,
                objective_scale=scale,
            )

            jac = signgs * jnp_module.asarray(k.bc.jac.sqrtg)
            jac_min = jnp_module.min(jac) if jac.shape[0] <= 1 else jnp_module.min(jac[1:, :, :])
            return terms.fsqr2, terms.fsqz2, terms.fsql2, terms.w, jac_min

        return _fsq2_terms_and_jacmin

    def _make_objective(scale: float | None):
        w_terms_local = _build_terms_fn(scale)

        def _w_only(state: VMECState, zero_m1_zforce: Any):
            return w_terms_local(state, zero_m1_zforce)[3]

        w_and_grad_local = jax_module.value_and_grad(_w_only)
        if jit_grad:
            return jit_func(w_and_grad_local), jit_func(w_terms_local)
        return w_and_grad_local, w_terms_local

    w_and_grad, w_terms = _make_objective(objective_scale_f)

    state = enforce_fixed_boundary_and_axis_func(
        state0,
        static,
        edge_Rcos=edge_Rcos,
        edge_Rsin=edge_Rsin,
        edge_Zcos=edge_Zcos,
        edge_Zsin=edge_Zsin,
        idx00=idx00,
    )

    zero_m1 = jnp_module.asarray(1.0, dtype=jnp_module.asarray(state0.Rcos).dtype)
    fsqr2_0, fsqz2_0, fsql2_0, w0, jacmin0 = w_terms(state, zero_m1)
    w0 = float(np.asarray(w0))
    jacmin0 = float(np.asarray(jacmin0))
    if not np.isfinite(w0):
        raise ValueError("Initial state has non-finite residual objective")
    if jacmin0 <= 0.0 and verbose:
        print("[solve_fixed_boundary_lbfgs_vmec_residual] warning: initial Jacobian has non-positive entries")

    if objective_scale_f is None:
        # Auto-scale the objective to be O(1) on the initial iterate.
        objective_scale_f = 1.0 / max(abs(w0), 1.0)
        w_and_grad, w_terms = _make_objective(objective_scale_f)
        fsqr2_0, fsqz2_0, fsql2_0, w0, jacmin0 = w_terms(state, zero_m1)
        w0 = float(np.asarray(w0))

    w_history = [w0]
    fsqr2_history = [float(np.asarray(fsqr2_0))]
    fsqz2_history = [float(np.asarray(fsqz2_0))]
    fsql2_history = [float(np.asarray(fsql2_0))]
    grad_rms_history = []
    step_history = []

    _w_val, grad = w_and_grad(state, zero_m1)
    grad = mask_grad_for_constraints_func(grad, static, idx00=idx00, mask_lambda_axis=False)
    grad = apply_preconditioner_func(
        grad,
        static,
        kind=preconditioner,
        exponent=precond_exponent,
        radial_alpha=precond_radial_alpha,
    )
    sr = jnp_module.asarray(scale_rz, dtype=jnp_module.asarray(grad.Rcos).dtype)
    sl = jnp_module.asarray(scale_l, dtype=jnp_module.asarray(grad.Lcos).dtype)
    grad = VMECState(
        layout=grad.layout,
        Rcos=jnp_module.asarray(grad.Rcos) * sr,
        Rsin=jnp_module.asarray(grad.Rsin) * sr,
        Zcos=jnp_module.asarray(grad.Zcos) * sr,
        Zsin=jnp_module.asarray(grad.Zsin) * sr,
        Lcos=jnp_module.asarray(grad.Lcos) * sl,
        Lsin=jnp_module.asarray(grad.Lsin) * sl,
    )

    x = pack_state_func(state)
    g_flat = pack_state_func(grad)

    s_hist: list[Any] = []
    y_hist: list[Any] = []

    step0 = float(step_size)
    grad_tol_eff: float | None = None
    zero_m1_fsqz_target = float(ftol_target)

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
            print(
                f"[solve_fixed_boundary_lbfgs_vmec_residual] iter={it:03d} "
                f"w={w_history[-1]:.8e} grad_rms={grad_rms:.3e}"
            )

        if grad_rms < float(grad_tol_eff):
            break

        p_flat = lbfgs_two_loop_direction_func(g_flat, s_hist, y_hist)
        p_flat, _gtp, _fallback_to_descent = ensure_descent_direction_func(g_flat, p_flat)

        accepted = False
        step = step0
        best_w = np.inf
        best_state = None
        best_step = None
        best_fsqr2 = None
        best_fsqz2 = None
        best_fsql2 = None

        x_old = x
        g_old = g_flat

        zero_m1 = jnp_module.asarray(
            1.0 if ((len(step_history) == 0) or (fsqz2_history[-1] < zero_m1_fsqz_target)) else 0.0,
            dtype=jnp_module.asarray(state.Rcos).dtype,
        )
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
                idx00=idx00,
            )

            fsqr2_t, fsqz2_t, fsql2_t, w_t, jacmin_t = w_terms(st_try, zero_m1)
            w_tf = float(np.asarray(w_t))
            jacmin_tf = float(np.asarray(jacmin_t))
            if np.isfinite(w_tf) and w_tf < best_w:
                best_w = w_tf
                best_state = st_try
                best_step = step
                best_fsqr2 = float(np.asarray(fsqr2_t))
                best_fsqz2 = float(np.asarray(fsqz2_t))
                best_fsql2 = float(np.asarray(fsql2_t))
            if np.isfinite(w_tf) and jacmin_tf > 0.0 and w_tf < w_history[-1]:
                state = st_try
                x = pack_state_func(state)
                accepted = True
                fsqr2_accept = float(np.asarray(fsqr2_t))
                fsqz2_accept = float(np.asarray(fsqz2_t))
                fsql2_accept = float(np.asarray(fsql2_t))
                break

        step_history.append(step)

        if not accepted:
            if best_state is not None and np.isfinite(best_w):
                if verbose:
                    print("[solve_fixed_boundary_lbfgs_vmec_residual] line search failed; accepting best finite step")
                state = best_state
                x = pack_state_func(state)
                w_t = best_w
                fsqr2_accept = best_fsqr2 if best_fsqr2 is not None else float(np.asarray(fsqr2_t))
                fsqz2_accept = best_fsqz2 if best_fsqz2 is not None else float(np.asarray(fsqz2_t))
                fsql2_accept = best_fsql2 if best_fsql2 is not None else float(np.asarray(fsql2_t))
                step_history[-1] = best_step
            else:
                if verbose:
                    print("[solve_fixed_boundary_lbfgs_vmec_residual] line search failed; stopping")
                break

        w_history.append(float(np.asarray(w_t)))
        fsqr2_history.append(fsqr2_accept)
        fsqz2_history.append(fsqz2_accept)
        fsql2_history.append(fsql2_accept)

        _w_val, grad_new = w_and_grad(state, zero_m1)
        grad_new = mask_grad_for_constraints_func(grad_new, static, idx00=idx00, mask_lambda_axis=False)
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
        "w_rz": float(w_rz),
        "w_l": float(w_l),
        "objective_scale": float(objective_scale_f),
        "include_constraint_force": bool(include_constraint_force),
        "scale_rz": float(scale_rz),
        "scale_l": float(scale_l),
        "apply_m1_constraints": bool(apply_m1_constraints),
        "history_size": int(history_size),
        "preconditioner": str(preconditioner),
        "precond_exponent": float(precond_exponent),
        "precond_radial_alpha": float(precond_radial_alpha),
        "grad_tol": None if grad_tol_eff is None else float(grad_tol_eff),
        "zero_m1_fsqz_thresh": float(zero_m1_fsqz_target),
    }
    return SolveVmecResidualResult(
        state=state,
        n_iter=len(w_history) - 1,
        w_history=np.asarray(w_history, dtype=float),
        fsqr2_history=np.asarray(fsqr2_history, dtype=float),
        fsqz2_history=np.asarray(fsqz2_history, dtype=float),
        fsql2_history=np.asarray(fsql2_history, dtype=float),
        grad_rms_history=np.asarray(grad_rms_history, dtype=float),
        step_history=np.asarray(step_history, dtype=float),
        diagnostics=diag,
    )
