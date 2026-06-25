"""Gauss-Newton optimizer for VMEC-style fixed-boundary residual objectives."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..results import SolveVmecResidualResult
from ..results import solve_vmec_residual_result_from_history as _solve_vmec_residual_result_from_history
from ..options import validate_residual_gn_options as _validate_residual_gn_options
from .constraints import enforce_fixed_boundary_and_axis as _enforce_fixed_boundary_and_axis
from .constraints import grad_rms_state as _grad_rms_state
from .constraints import mode00_index as _mode00_index
from .gradient import mask_grad_for_constraints as _mask_grad_for_constraints
from .residual_context import prepare_residual_force_context as _prepare_residual_force_context
from .residual_context import residual_terms_from_force_context as _residual_terms_from_force_context
from .residual_objective import assemble_residual_objective_terms as _assemble_residual_objective_terms
from .residual_objective import residual_objective_vector as _residual_objective_vector
from .tolerances import dtype_tiny as _dtype_tiny
from .tolerances import resolve_cg_tol as _resolve_cg_tol
from .tolerances import resolve_lm_damping as _resolve_lm_damping
from ...._compat import has_jax as _has_jax
from ...._compat import jax as _jax
from ...._compat import jit as _jit
from ...._compat import jnp as _jnp
from ....state import VMECState
from ....state import pack_state as _pack_state
from ....state import unpack_state as _unpack_state


def _state_plus_scaled_step(state: VMECState, dx_state: VMECState, *, step: float, jnp_module: Any) -> VMECState:
    """Apply one projected optimizer step while preserving per-array dtypes."""
    return VMECState(
        layout=state.layout,
        Rcos=jnp_module.asarray(state.Rcos)
        + jnp_module.asarray(step, dtype=jnp_module.asarray(state.Rcos).dtype) * jnp_module.asarray(dx_state.Rcos),
        Rsin=jnp_module.asarray(state.Rsin)
        + jnp_module.asarray(step, dtype=jnp_module.asarray(state.Rsin).dtype) * jnp_module.asarray(dx_state.Rsin),
        Zcos=jnp_module.asarray(state.Zcos)
        + jnp_module.asarray(step, dtype=jnp_module.asarray(state.Zcos).dtype) * jnp_module.asarray(dx_state.Zcos),
        Zsin=jnp_module.asarray(state.Zsin)
        + jnp_module.asarray(step, dtype=jnp_module.asarray(state.Zsin).dtype) * jnp_module.asarray(dx_state.Zsin),
        Lcos=jnp_module.asarray(state.Lcos)
        + jnp_module.asarray(step, dtype=jnp_module.asarray(state.Lcos).dtype) * jnp_module.asarray(dx_state.Lcos),
        Lsin=jnp_module.asarray(state.Lsin)
        + jnp_module.asarray(step, dtype=jnp_module.asarray(state.Lsin).dtype) * jnp_module.asarray(dx_state.Lsin),
    )


def solve_fixed_boundary_gn_vmec_residual_impl(
    state0: VMECState,
    static,
    *,
    indata,
    signgs: int,
    w_rz: float = 1.0,
    w_l: float = 1.0,
    include_constraint_force: bool = True,
    apply_m1_constraints: bool = True,
    objective_scale: float | None = None,
    damping: float | None = None,
    damping_increase: float = 10.0,
    damping_decrease: float = 0.5,
    max_damping: float | None = None,
    max_retries: int = 6,
    zero_m1_iters: int | None = None,
    zero_m1_fsqz_thresh: float | None = None,
    max_iter: int = 20,
    cg_tol: float | None = None,
    cg_maxiter: int = 80,
    step_size: float = 1.0,
    max_backtracks: int = 12,
    bt_factor: float = 0.5,
    jit_kernels: bool = True,
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
    residual_objective_vector_func: Callable[..., Any] | None = None,
    enforce_fixed_boundary_and_axis_func: Callable[..., VMECState] | None = None,
    mask_grad_for_constraints_func: Callable[..., VMECState] | None = None,
    grad_rms_state_func: Callable[..., float] | None = None,
    resolve_cg_tol_func: Callable[..., float] | None = None,
    resolve_lm_damping_func: Callable[..., float] | None = None,
    dtype_tiny_func: Callable[..., float] | None = None,
    pack_state_func: Callable[..., Any] | None = None,
    unpack_state_func: Callable[..., VMECState] | None = None,
    jax_module: Any | None = None,
    jnp_module: Any | None = None,
    jit_func: Callable[..., Any] | None = None,
) -> SolveVmecResidualResult:
    """Solve a VMEC residual least-squares system with Gauss-Newton steps."""

    has_jax_func = has_jax_func or _has_jax
    jax_module = jax_module or _jax
    jnp_module = jnp_module or _jnp
    jit_func = jit_func or _jit

    if not has_jax_func():
        raise ImportError("solve_fixed_boundary_gn_vmec_residual requires JAX (jax + jaxlib)")

    validate_options_func = validate_options_func or _validate_residual_gn_options
    prepare_residual_force_context_func = prepare_residual_force_context_func or _prepare_residual_force_context
    mode00_index_func = mode00_index_func or _mode00_index
    assemble_residual_objective_terms_func = assemble_residual_objective_terms_func or _assemble_residual_objective_terms
    residual_objective_vector_func = residual_objective_vector_func or _residual_objective_vector
    enforce_fixed_boundary_and_axis_func = enforce_fixed_boundary_and_axis_func or _enforce_fixed_boundary_and_axis
    mask_grad_for_constraints_func = mask_grad_for_constraints_func or _mask_grad_for_constraints
    grad_rms_state_func = grad_rms_state_func or _grad_rms_state
    resolve_cg_tol_func = resolve_cg_tol_func or _resolve_cg_tol
    resolve_lm_damping_func = resolve_lm_damping_func or _resolve_lm_damping
    dtype_tiny_func = dtype_tiny_func or _dtype_tiny
    pack_state_func = pack_state_func or _pack_state
    unpack_state_func = unpack_state_func or _unpack_state

    opts = validate_options_func(
        damping=damping,
        damping_increase=damping_increase,
        damping_decrease=damping_decrease,
        max_damping=max_damping,
        max_retries=max_retries,
        zero_m1_iters=zero_m1_iters,
        zero_m1_fsqz_thresh=zero_m1_fsqz_thresh,
        w_rz=w_rz,
        w_l=w_l,
        max_iter=max_iter,
        cg_maxiter=cg_maxiter,
        max_backtracks=max_backtracks,
        bt_factor=bt_factor,
        objective_scale=objective_scale,
    )
    damping = opts.damping
    damping_increase = opts.damping_increase
    damping_decrease = opts.damping_decrease
    max_damping_eff = opts.max_damping_eff
    max_retries = opts.max_retries
    zero_m1_iters_eff = opts.zero_m1_iters_eff
    zero_m1_fsqz_thresh = opts.zero_m1_fsqz_thresh
    w_rz = opts.w_rz
    w_l = opts.w_l
    max_iter = opts.max_iter
    cg_maxiter = opts.cg_maxiter
    max_backtracks = opts.max_backtracks
    bt_factor = opts.bt_factor
    objective_scale = opts.objective_scale

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
    ftol_target = residual_context.ftol_target
    edge_Rcos = residual_context.edge_Rcos
    edge_Rsin = residual_context.edge_Rsin
    edge_Zcos = residual_context.edge_Zcos
    edge_Zsin = residual_context.edge_Zsin
    zero_m1_fsqz_thresh_eff = float(ftol_target) if zero_m1_fsqz_thresh is None else float(zero_m1_fsqz_thresh)

    try:
        from jax.scipy.sparse.linalg import cg  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("solve_fixed_boundary_gn_vmec_residual requires jax.scipy.sparse.linalg.cg") from e

    def _project_step(d: VMECState) -> VMECState:
        return mask_grad_for_constraints_func(d, static, idx00=idx00, mask_lambda_axis=True)

    def _enforce_state(st: VMECState) -> VMECState:
        return enforce_fixed_boundary_and_axis_func(
            st,
            static,
            edge_Rcos=edge_Rcos,
            edge_Rsin=edge_Rsin,
            edge_Zcos=edge_Zcos,
            edge_Zsin=edge_Zsin,
            enforce_lambda_axis=True,
            idx00=idx00,
        )

    def _residual_blocks(state: VMECState, zero_m1_zforce: Any):
        terms, _jac_min = _residual_terms_from_force_context(
            context=residual_context,
            state=state,
            static=static,
            zero_m1_zforce=zero_m1_zforce,
            w_rz=w_rz,
            w_l=w_l,
            apply_m1_constraints=bool(apply_m1_constraints),
            zero_m1_after_m1_constraints=True,
            include_edge=True,
            zero_edge_rz_blocks=True,
            objective_scale=None,
            assemble_residual_objective_terms_func=assemble_residual_objective_terms_func,
            jnp_module=jnp_module,
        )
        return terms.frzl, terms.fsqr2, terms.fsqz2, terms.fsql2, terms.norms

    def _residual_vec(state: VMECState, zero_m1_zforce: Any) -> Any:
        frzl, *_vals = _residual_blocks(state, zero_m1_zforce)
        norms = _vals[-1]
        return residual_objective_vector_func(frzl=frzl, norms=norms, w_rz=w_rz, w_l=w_l)

    def _obj_terms(state: VMECState, zero_m1_zforce: Any):
        _frzl, fsqr2, fsqz2, fsql2, _norms = _residual_blocks(state, zero_m1_zforce)
        w = (w_rz * (fsqr2 + fsqz2)) + (w_l * fsql2)
        return fsqr2, fsqz2, fsql2, w

    if bool(jit_kernels):
        _residual_vec_jit = jit_func(_residual_vec)
        _obj_terms_jit = jit_func(_obj_terms)
    else:
        _residual_vec_jit = _residual_vec
        _obj_terms_jit = _obj_terms

    state = _enforce_state(state0)
    zero_m1 = jnp_module.asarray(1.0, dtype=jnp_module.asarray(state0.Rcos).dtype)
    fsqr2_0, fsqz2_0, fsql2_0, w0 = _obj_terms_jit(state, zero_m1)
    w0_f = float(np.asarray(w0))
    if not np.isfinite(w0_f):
        raise ValueError("Initial state has non-finite residual objective")

    scale_f = float(objective_scale) if objective_scale is not None else (1.0 / max(abs(w0_f), 1.0))

    w_history = [float(scale_f * w0_f)]
    fsqr2_history = [float(np.asarray(fsqr2_0))]
    fsqz2_history = [float(np.asarray(fsqz2_0))]
    fsql2_history = [float(np.asarray(fsql2_0))]
    grad_rms_history = []
    step_history = []

    for it in range(int(max_iter)):
        zero_m1_active = (len(step_history) < int(zero_m1_iters_eff)) or (len(step_history) == 0)
        zero_m1_active = zero_m1_active or (fsqz2_history[-1] < float(zero_m1_fsqz_thresh_eff))
        zero_m1 = jnp_module.asarray(
            1.0 if zero_m1_active else 0.0,
            dtype=jnp_module.asarray(state.Rcos).dtype,
        )
        r, pullback = jax_module.vjp(_residual_vec_jit, state, zero_m1)
        # Gradient of 0.5*||r||^2 is J^T r.
        g_state = pullback(r)[0]
        g_state = _project_step(g_state)
        grad_rms_history.append(grad_rms_state_func(g_state))

        b_flat = -pack_state_func(g_state)
        dtype_state = np.asarray(state.Rcos).dtype
        current_w = float(w_history[-1])
        cg_tol_it = resolve_cg_tol_func(
            cg_tol,
            current_obj=current_w,
            initial_obj=float(w_history[0]),
            target_obj=float(ftol_target),
            dtype=dtype_state,
        )
        g_norm_sq = float(np.asarray(jnp_module.dot(b_flat, b_flat)))
        if np.isfinite(g_norm_sq) and g_norm_sq > dtype_tiny_func(dtype_state):
            zero_tangent = jnp_module.zeros_like(zero_m1)
            jg = jax_module.jvp(_residual_vec_jit, (state, zero_m1), (g_state, zero_tangent))[1]
            jt_jg = pullback(jg)[0]
            jt_jg = _project_step(jt_jg)
            curvature_num = float(np.asarray(jnp_module.dot(pack_state_func(g_state), pack_state_func(jt_jg))))
            curvature_scale = max(0.0, curvature_num / max(g_norm_sq, dtype_tiny_func(dtype_state)))
        else:
            curvature_scale = 0.0
        damping_it = resolve_lm_damping_func(damping, curvature_scale=curvature_scale, dtype=dtype_state)

        accepted = False
        step = float(step_size)
        w_curr = w_history[-1]
        retry = 0
        while True:
            dmp = float(damping_it)

            def _matvec(v_flat):
                v_state = unpack_state_func(v_flat, state.layout)
                v_state = _project_step(v_state)
                zero_tangent = jnp_module.zeros_like(zero_m1)
                jv = jax_module.jvp(_residual_vec_jit, (state, zero_m1), (v_state, zero_tangent))[1]
                jt_jv = pullback(jv)[0]
                jt_jv = _project_step(jt_jv)
                if dmp != 0.0:
                    jt_jv = VMECState(
                        layout=jt_jv.layout,
                        Rcos=jt_jv.Rcos + dmp * v_state.Rcos,
                        Rsin=jt_jv.Rsin + dmp * v_state.Rsin,
                        Zcos=jt_jv.Zcos + dmp * v_state.Zcos,
                        Zsin=jt_jv.Zsin + dmp * v_state.Zsin,
                        Lcos=jt_jv.Lcos + dmp * v_state.Lcos,
                        Lsin=jt_jv.Lsin + dmp * v_state.Lsin,
                    )
                return pack_state_func(jt_jv)

            dx_flat, _info = cg(_matvec, b_flat, tol=float(cg_tol_it), maxiter=int(cg_maxiter))
            dx_state = unpack_state_func(dx_flat, state.layout)
            dx_state = _project_step(dx_state)

            step = float(step_size)
            for bt in range(int(max_backtracks) + 1):
                if bt > 0:
                    step *= float(bt_factor)
                st_try = _state_plus_scaled_step(state, dx_state, step=step, jnp_module=jnp_module)
                st_try = _enforce_state(st_try)
                fsqr2_t, fsqz2_t, fsql2_t, w_t = _obj_terms_jit(st_try, zero_m1)
                w_tf = float(np.asarray(w_t))
                w_scaled = float(scale_f * w_tf)
                if np.isfinite(w_scaled) and w_scaled < w_curr:
                    state = st_try
                    accepted = True
                    w_history.append(w_scaled)
                    fsqr2_history.append(float(np.asarray(fsqr2_t)))
                    fsqz2_history.append(float(np.asarray(fsqz2_t)))
                    fsql2_history.append(float(np.asarray(fsql2_t)))
                    break

            if accepted:
                # Levenberg-Marquardt style: relax damping after success.
                damping_it = max(damping_it * damping_decrease, 0.0)
                break

            if retry >= max_retries or damping_it >= max_damping_eff:
                break
            # Increase damping and try again from the same state.
            damping_it = min(max_damping_eff, damping_it * damping_increase)
            retry += 1

        if not accepted:
            # Robust fallback: take a small steepest-descent step on 0.5*||r||^2
            # using the already-computed gradient g_state = J^T r.
            dx_state = unpack_state_func(b_flat, state.layout)  # b_flat = -grad_flat
            dx_state = _project_step(dx_state)
            step = float(step_size)
            for bt in range(int(max_backtracks) + 1):
                if bt > 0:
                    step *= float(bt_factor)
                st_try = _state_plus_scaled_step(state, dx_state, step=step, jnp_module=jnp_module)
                st_try = _enforce_state(st_try)
                fsqr2_t, fsqz2_t, fsql2_t, w_t = _obj_terms_jit(st_try, zero_m1)
                w_tf = float(np.asarray(w_t))
                w_scaled = float(scale_f * w_tf)
                if np.isfinite(w_scaled) and w_scaled < w_curr:
                    state = st_try
                    accepted = True
                    w_history.append(w_scaled)
                    fsqr2_history.append(float(np.asarray(fsqr2_t)))
                    fsqz2_history.append(float(np.asarray(fsqz2_t)))
                    fsql2_history.append(float(np.asarray(fsql2_t)))
                    break

        step_history.append(step)
        if verbose:
            print(
                f"[solve_fixed_boundary_gn_vmec_residual] iter={it:03d} w={w_history[-1]:.8e} "
                f"step={step:.3e} accepted={accepted} damping={damping_it:.3e} cg_tol={cg_tol_it:.3e} retries={retry}"
            )

        if not accepted:
            break

    diag = {
        "idx00": idx00,
        "signgs": signgs,
        "w_rz": float(w_rz),
        "w_l": float(w_l),
        "objective_scale": float(scale_f),
        "apply_m1_constraints": bool(apply_m1_constraints),
        "damping": None if damping is None else float(damping),
        "damping_mode": "adaptive" if damping is None else "fixed",
        "cg_tol": None if cg_tol is None else float(cg_tol),
        "cg_tol_mode": "adaptive" if cg_tol is None else "fixed",
        "cg_maxiter": int(cg_maxiter),
        "zero_m1_iters": None if zero_m1_iters is None else int(zero_m1_iters),
        "zero_m1_fsqz_thresh": float(zero_m1_fsqz_thresh_eff),
    }
    return _solve_vmec_residual_result_from_history(
        state=state,
        w_history=w_history,
        fsqr2_history=fsqr2_history,
        fsqz2_history=fsqz2_history,
        fsql2_history=fsql2_history,
        grad_rms_history=grad_rms_history,
        step_history=step_history,
        diagnostics=diag,
    )
