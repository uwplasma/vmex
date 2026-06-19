"""Implicit differentiation utilities.

This module provides *custom VJP* wrappers for equilibrium sub-solves so that
outer objectives can differentiate through equilibrium states without
backpropagating through many optimization iterations.

Initial scope
=============

Implicit differentiation for the **lambda-only** fixed-geometry solve.

Future work
===========

Extend to the full fixed-boundary solve over (R,Z,lambda) using the same
implicit-function machinery.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import os
import time
from typing import Any, Callable, Tuple

import numpy as np

from ._compat import has_jax, jax, jnp
from .implicit_adjoint_helpers import (
    full_active_keep_indices,
    make_full_normal_map,
    pack_stellsym_feasible_state as _pack_stellsym_feasible_state,
    pack_stellsym_reduced_state as _pack_stellsym_reduced_state,
    select_active_packing_strategy,
    stellsym_feasible_indices as _stellsym_feasible_indices,  # noqa: F401 - compatibility export
    stellsym_feasible_indices_np as _stellsym_feasible_indices_np,
    stellsym_lambda_mn_indices as _stellsym_lambda_mn_indices,
    stellsym_reduced_z_indices as _stellsym_reduced_z_indices,
    stellsym_structural_active_keep_indices as _stellsym_structural_active_keep_indices,  # noqa: F401 - compatibility export
    update_stellsym_feasible_state as _update_stellsym_feasible_state,
    update_stellsym_reduced_state as _update_stellsym_reduced_state,
    validate_full_adjoint_shapes,
)
from .implicit_residual_adjoint_helpers import (
    lineax_bicgstab_solve as _lineax_bicgstab_solve_impl,
    linear_map_jacobian_columns as _linear_map_jacobian_columns_impl,
    solve_active_residual_adjoint_linearized as _solve_active_residual_adjoint_linearized,
    solve_active_residual_tangent_linearized as _solve_active_residual_tangent_linearized,
    solve_full_residual_adjoint_linearized as _solve_full_residual_adjoint_linearized,
)

try:
    import lineax as lx
except Exception:  # pragma: no cover - optional dependency
    lx = None

from .field import TWOPI, b2_from_bsup, bsup_from_geom, bsup_from_sqrtg_lambda, signgs_from_sqrtg
from .energy import flux_profiles_from_indata
from .fourier import eval_fourier_dtheta, eval_fourier_dzeta_phys
from .geom import eval_geom
from .grids import angle_steps
from .solve import (
    _WoutLikeVmecForces,
    _enforce_fixed_boundary_and_axis,
    _mask_grad_for_constraints,
    _mode00_index,
    _zero_edge_rz_force_blocks,
    solve_fixed_boundary_gd,
    solve_fixed_boundary_lbfgs,
    solve_fixed_boundary_residual_iter,
    solve_lambda_gd,
)
from .solvers.fixed_boundary.profiles import (
    _half_mesh_from_full_mesh,
    _icurv_full_mesh_from_indata,
    _mass_half_mesh_from_indata,
    _pressure_half_mesh_from_indata,
    _vmec_force_flux_profiles,
)
from .state import VMECState, pack_state, unpack_state


def _env_flag_enabled(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() not in ("", "0", "false", "no")


def _vmec_backward_profile_enabled() -> bool:
    return _env_flag_enabled("VMEC_JAX_PROFILE_BACKWARD")


def _profile_log(*, prefix: str, enabled: bool, stage: str, start: float | None, extra: dict[str, Any]) -> None:
    if not enabled:
        return
    payload = {"stage": stage}
    if start is not None:
        payload["elapsed_s"] = time.perf_counter() - start
    payload.update(extra)
    print(f"[vmec_jax {prefix}] {payload}", flush=True)


def _vmec_backward_profile_log(stage: str, start: float | None = None, **extra) -> None:
    _profile_log(prefix="backward", enabled=_vmec_backward_profile_enabled(), stage=stage, start=start, extra=extra)


def _vmec_residual_profile_enabled() -> bool:
    return _env_flag_enabled("VMEC_JAX_PROFILE_RESIDUAL")


def _vmec_residual_profile_log(stage: str, start: float | None = None, **extra) -> None:
    _profile_log(prefix="residual", enabled=_vmec_residual_profile_enabled(), stage=stage, start=start, extra=extra)


def _vmec_keep_all_active_enabled() -> bool:
    return _env_flag_enabled("VMEC_JAX_IMPLICIT_KEEP_ALL_ACTIVE")


def _vmec_disable_reduced_active_enabled() -> bool:
    return _env_flag_enabled("VMEC_JAX_IMPLICIT_DISABLE_REDUCED_ACTIVE")


def _contains_jax_tracer(*xs) -> bool:
    return any(isinstance(x, jax.core.Tracer) for x in xs)


def _jax_bicgstab_solve(*args, **kwargs):
    from jax.scipy.sparse.linalg import bicgstab

    return bicgstab(*args, **kwargs)


def _dense_transpose_lstsq_host(J, b, damping):
    """Host-side least-squares solve for J^T lam ~= b with optional Tikhonov damping."""
    J_host = np.asarray(J)
    b_host = np.asarray(b)
    damping_host = float(np.asarray(damping))
    A_host = J_host.T
    if damping_host > 0.0:
        eye = np.eye(int(A_host.shape[1]), dtype=A_host.dtype)
        A_host = np.concatenate(
            [A_host, np.sqrt(damping_host) * eye],
            axis=0,
        )
        b_host = np.concatenate(
            [b_host, np.zeros((int(eye.shape[0]),), dtype=b_host.dtype)],
            axis=0,
        )
    lam_host, *_ = np.linalg.lstsq(A_host, b_host, rcond=None)
    return np.asarray(lam_host, dtype=J_host.dtype)


def _pack_named_residual_parts(parts, projector=None):
    """Flatten named residual blocks, optionally keeping structural indices."""
    def _pack_one(name, arr):
        flat = jnp.ravel(jnp.asarray(arr))
        keep = None if projector is None else projector.get(name)
        return flat if keep is None else jnp.take(flat, keep)

    return jnp.concatenate([_pack_one(name, arr) for name, arr in parts], axis=0)


def _zero_m1_zforce_flag_from_result(res, dtype) -> np.ndarray:
    """Return the VMEC residual tangent flag used after the host primal solve."""
    fsqz_hist = np.asarray(getattr(res, "fsqz2_history", []), dtype=float)
    n_iter = int(getattr(res, "n_iter", 0))
    enabled = n_iter < 2 or (fsqz_hist.size > 0 and float(fsqz_hist[-1]) < 1.0e-6)
    return np.asarray(1.0 if enabled else 0.0, dtype=dtype)


@dataclass(frozen=True)
class ImplicitLambdaOptions:
    """Controls for the implicit backward pass."""

    cg_max_iter: int = 80
    cg_tol: float = 1e-10
    damping: float = 1e-6


@dataclass(frozen=True)
class ImplicitFixedBoundaryOptions:
    """Controls for the implicit backward pass (fixed-boundary solve)."""

    cg_max_iter: int = 80
    cg_tol: float = 1e-10
    damping: float = 1e-6
    residual_adjoint_mode: str = "auto"
    residual_tangent_mode: str = "opaque"
    jac_chunk_size: int | None = None


@dataclass(frozen=True)
class _VmecResidualSetup:
    """Passive data needed by the VMEC residual implicit-adjoint closure."""

    s: Any
    wout_like: Any
    trig: Any
    apply_lforbal: bool
    mask_pack: Any
    stellsym_residual_projector: Any


@dataclass(frozen=True)
class _VmecResidualVectorContext:
    """Static metadata for the VMEC residual vector used by implicit AD."""

    static: Any
    indata: Any
    signgs_i: int
    s: Any
    wout_like: Any
    trig: Any
    apply_lforbal: bool
    mask_pack: Any
    stellsym_residual_projector: Any


@dataclass(frozen=True)
class _VmecResidualHostSolveSettings:
    """Host-only inputs for the primal VMEC residual solve used by pure_callback."""

    static: Any
    indata: Any
    state_layout: Any
    state0_host: VMECState | None
    max_iter: int
    step_size: float
    ftol: float | None


def _build_vmec_residual_setup(*, state0_c: VMECState, static, indata, signgs_i: int, idx00: int) -> _VmecResidualSetup:
    """Build profile, force, and structural residual metadata for implicit VMEC residual AD."""
    from .boundary import boundary_from_indata
    from .solvers.fixed_boundary.optimization.residual_context import prepare_residual_force_context
    from .vmec_tomnsp import vmec_trig_tables

    context = prepare_residual_force_context(
        state0_c,
        static,
        indata=indata,
        signgs=int(signgs_i),
        idx00=int(idx00),
        include_constraint_force=True,
        mode00_index_func=_mode00_index,
        half_mesh_from_full_mesh_func=_half_mesh_from_full_mesh,
        mass_half_mesh_from_indata_func=_mass_half_mesh_from_indata,
        pressure_half_mesh_from_indata_func=_pressure_half_mesh_from_indata,
        icurv_full_mesh_from_indata_func=_icurv_full_mesh_from_indata,
        vmec_force_flux_profiles_func=_vmec_force_flux_profiles,
        wout_like_cls=_WoutLikeVmecForces,
        flux_profiles_from_indata_func=flux_profiles_from_indata,
        boundary_from_indata_func=boundary_from_indata,
        vmec_trig_tables_func=vmec_trig_tables,
        jnp_module=jnp,
    )
    mask_pack = getattr(static, "tomnsps_masks", None)
    stellsym_residual_projector = None
    if (not bool(static.cfg.lasym)) and (mask_pack is not None):
        ns_mask = int(static.cfg.ns)
        mpol_mask = int(static.cfg.mpol)
        ntor1_mask = int(static.cfg.ntor) + 1
        mask_rz_np = np.broadcast_to(np.asarray(mask_pack.mask_rz) > 0, (ns_mask, mpol_mask, ntor1_mask))
        mask_l_np = np.broadcast_to(np.asarray(mask_pack.mask_l) > 0, (ns_mask, mpol_mask, ntor1_mask))
        m0_mask = np.broadcast_to((np.arange(mpol_mask)[None, :, None] == 0), (ns_mask, mpol_mask, ntor1_mask))
        n0_mask = np.broadcast_to((np.arange(ntor1_mask)[None, None, :] == 0), (ns_mask, mpol_mask, ntor1_mask))
        stellsym_residual_projector = {
            "frcc": jnp.asarray(np.flatnonzero(mask_rz_np.reshape(-1)), dtype=jnp.int32),
            "fzsc": jnp.asarray(np.flatnonzero((mask_rz_np & ~m0_mask).reshape(-1)), dtype=jnp.int32),
            "flsc": jnp.asarray(np.flatnonzero((mask_l_np & ~m0_mask).reshape(-1)), dtype=jnp.int32),
            "frss": jnp.asarray(np.flatnonzero((mask_rz_np & ~m0_mask & ~n0_mask).reshape(-1)), dtype=jnp.int32),
            "fzcs": jnp.asarray(np.flatnonzero((mask_rz_np & ~n0_mask).reshape(-1)), dtype=jnp.int32),
            "flcs": jnp.asarray(np.flatnonzero((mask_l_np & ~n0_mask).reshape(-1)), dtype=jnp.int32),
        }
    return _VmecResidualSetup(
        s=context.s,
        wout_like=context.wout_like,
        trig=context.trig,
        apply_lforbal=bool(context.apply_lforbal),
        mask_pack=mask_pack,
        stellsym_residual_projector=stellsym_residual_projector,
    )


def _vmec_residual_vector_from_state(
    *,
    state: VMECState,
    zero_m1_zforce,
    eRcos,
    eRsin,
    eZcos,
    eZsin,
    context: _VmecResidualVectorContext,
    enforce_state_func: Callable[..., VMECState],
    project_stellsym: bool = False,
):
    """Assemble the scaled VMEC residual vector for an implicit fixed-boundary solve."""
    from .preconditioner_1d_jax import (
        lambda_preconditioner_cached,
    )
    from .vmec_forces import vmec_forces_rz_from_wout, vmec_residual_internal_from_kernels
    from .vmec_residue import (
        vmec_apply_m1_constraints,
        vmec_apply_scalxc_to_tomnsps,
        vmec_force_norms_from_bcovar_dynamic,
        vmec_zero_m1_zforce,
    )

    residual_start = time.perf_counter()
    state = enforce_state_func(state, eRcos, eRsin, eZcos, eZsin)
    forces_start = time.perf_counter()
    k = vmec_forces_rz_from_wout(
        state=state,
        static=context.static,
        wout=context.wout_like,
        indata=None,
        constraint_tcon0=float(context.indata.get_float("TCON0", 1.0)),
        use_vmec_synthesis=True,
        trig=context.trig,
    )
    _vmec_residual_profile_log("forces_done", forces_start)
    tomnsps_start = time.perf_counter()
    frzl = vmec_residual_internal_from_kernels(
        k,
        cfg_ntheta=int(context.static.cfg.ntheta),
        cfg_nzeta=int(context.static.cfg.nzeta),
        wout=context.wout_like,
        trig=context.trig,
        apply_lforbal=context.apply_lforbal,
        include_edge=False,
        masks=context.mask_pack,
    )
    _vmec_residual_profile_log("tomnsps_done", tomnsps_start)
    post_start = time.perf_counter()
    frzl = vmec_apply_m1_constraints(frzl=frzl, lconm1=bool(getattr(context.static.cfg, "lconm1", True)))
    frzl = vmec_zero_m1_zforce(frzl=frzl, enabled=zero_m1_zforce)
    frzl = vmec_apply_scalxc_to_tomnsps(frzl=frzl, s=context.s)
    frzl = _zero_edge_rz_force_blocks(frzl, preserve_numpy=False)
    norms = vmec_force_norms_from_bcovar_dynamic(
        bc=k.bc,
        trig=context.trig,
        s=context.s,
        signgs=context.signgs_i,
    )
    scale_rz = jnp.sqrt(norms.r1 * norms.fnorm)
    scale_l = jnp.sqrt(norms.fnormL)
    lam_prec = lambda_preconditioner_cached(
        bc=k.bc,
        trig=context.trig,
        s=context.s,
        cfg=context.static.cfg,
    )
    parts = [
        ("frcc", scale_rz * frzl.frcc),
        ("fzsc", scale_rz * frzl.fzsc),
        ("flsc", scale_l * jnp.asarray(frzl.flsc) * jnp.asarray(lam_prec)),
    ]
    if frzl.frss is not None:
        parts.append(("frss", scale_rz * frzl.frss))
    if frzl.fzcs is not None:
        parts.append(("fzcs", scale_rz * frzl.fzcs))
    if frzl.flcs is not None:
        parts.append(("flcs", scale_l * jnp.asarray(frzl.flcs) * jnp.asarray(lam_prec)))
    for name in ["frsc", "fzcc", "flcc", "frcs", "fzss", "flss"]:
        arr = getattr(frzl, name, None)
        if arr is not None:
            scale = scale_l if name.startswith("fl") else scale_rz
            if name.startswith("fl"):
                arr = jnp.asarray(arr) * jnp.asarray(lam_prec)
            parts.append((name, scale * arr))
    projector = context.stellsym_residual_projector if bool(project_stellsym) else None
    packed = _pack_named_residual_parts(parts, projector=projector)
    _vmec_residual_profile_log("postprocess_done", post_start)
    _vmec_residual_profile_log(
        "residual_done",
        residual_start,
        projected=bool(project_stellsym),
        output_size=int(np.prod(np.shape(packed))),
    )
    return packed


def _solve_vmec_residual_host_from_boundary(eRcos, eRsin, eZcos, eZsin, settings: _VmecResidualHostSolveSettings):
    """Run the authoritative host VMEC residual solve for a trial boundary."""
    from .boundary import BoundaryCoeffs
    from .init_guess import initial_guess_from_boundary

    eRcos_np = np.asarray(eRcos)
    eRsin_np = np.asarray(eRsin)
    eZcos_np = np.asarray(eZcos)
    eZsin_np = np.asarray(eZsin)
    dtype_host = (
        np.asarray(settings.state0_host.Rcos).dtype
        if settings.state0_host is not None
        else np.asarray(eRcos_np).dtype
    )
    boundary_host = BoundaryCoeffs(
        R_cos=np.array(eRcos_np, copy=True),
        R_sin=np.array(eRsin_np, copy=True),
        Z_cos=np.array(eZcos_np, copy=True),
        Z_sin=np.array(eZsin_np, copy=True),
    )
    state_init = initial_guess_from_boundary(
        settings.static,
        boundary_host,
        settings.indata,
        dtype=dtype_host,
        vmec_project=True,
    )
    geom0 = eval_geom(state_init, settings.static)
    signgs0 = signgs_from_sqrtg(np.asarray(geom0.sqrtg), axis_index=1)
    state_init = VMECState(
        layout=state_init.layout,
        Rcos=np.asarray(jax.device_get(state_init.Rcos)),
        Rsin=np.asarray(jax.device_get(state_init.Rsin)),
        Zcos=np.asarray(jax.device_get(state_init.Zcos)),
        Zsin=np.asarray(jax.device_get(state_init.Zsin)),
        Lcos=np.asarray(jax.device_get(state_init.Lcos)),
        Lsin=np.asarray(jax.device_get(state_init.Lsin)),
    )
    res = solve_fixed_boundary_residual_iter(
        state_init,
        settings.static,
        indata=settings.indata,
        signgs=int(signgs0),
        ftol=settings.ftol,
        max_iter=int(settings.max_iter),
        step_size=float(settings.step_size),
        vmec2000_control=True,
        reference_mode=False,
        backtracking=True,
        limit_dt_from_force=True,
        limit_update_rms=True,
        verbose=False,
        verbose_vmec2000_table=False,
        jit_forces="auto",
        use_scan=False,
    )
    zero_m1 = _zero_m1_zforce_flag_from_result(res, dtype=dtype_host)
    return np.asarray(pack_state(res.state)), zero_m1


def _project_boundary_edge_rows(*, static, indata, dtype, eRcos, eRsin, eZcos, eZsin):
    """Project raw boundary coefficients to VMEC edge rows."""
    from .boundary import BoundaryCoeffs
    from .init_guess import initial_guess_from_boundary

    boundary_state = initial_guess_from_boundary(
        static,
        BoundaryCoeffs(
            R_cos=jnp.asarray(eRcos),
            R_sin=jnp.asarray(eRsin),
            Z_cos=jnp.asarray(eZcos),
            Z_sin=jnp.asarray(eZsin),
        ),
        indata,
        dtype=dtype,
        vmec_project=True,
    )
    return (
        jnp.asarray(boundary_state.Rcos)[-1, :],
        jnp.asarray(boundary_state.Rsin)[-1, :],
        jnp.asarray(boundary_state.Zcos)[-1, :],
        jnp.asarray(boundary_state.Zsin)[-1, :],
    )


def _boundary_edge_rows_vjp(boundary_edge_rows_func: Callable[..., tuple[Any, Any, Any, Any]], residual_edges, ct_state):
    """Apply the VJP of the boundary-edge projection to an output-state cotangent."""
    eRcos_star, eRsin_star, eZcos_star, eZsin_star = residual_edges
    ct_edge = (
        jnp.asarray(ct_state.Rcos)[-1, :],
        jnp.asarray(ct_state.Rsin)[-1, :],
        jnp.asarray(ct_state.Zcos)[-1, :],
        jnp.asarray(ct_state.Zsin)[-1, :],
    )
    _, edge_vjp_fun = jax.vjp(
        boundary_edge_rows_func,
        eRcos_star,
        eRsin_star,
        eZcos_star,
        eZsin_star,
    )
    return edge_vjp_fun(ct_edge)


def _stop_gradient_tree(x):
    return jax.tree_util.tree_map(jax.lax.stop_gradient, x)


def _zero_state_like(state: VMECState) -> VMECState:
    return VMECState(
        layout=state.layout,
        Rcos=jnp.zeros_like(jnp.asarray(state.Rcos)),
        Rsin=jnp.zeros_like(jnp.asarray(state.Rsin)),
        Zcos=jnp.zeros_like(jnp.asarray(state.Zcos)),
        Zsin=jnp.zeros_like(jnp.asarray(state.Zsin)),
        Lcos=jnp.zeros_like(jnp.asarray(state.Lcos)),
        Lsin=jnp.zeros_like(jnp.asarray(state.Lsin)),
    )


def _flatten_L(Lcos, Lsin) -> Any:
    return jnp.concatenate([jnp.ravel(Lcos), jnp.ravel(Lsin)], axis=0)


def _unflatten_L(x, *, shape: Tuple[int, int]):
    ns, K = shape
    n = ns * K
    x = jnp.asarray(x)
    Lcos = jnp.reshape(x[:n], (ns, K))
    Lsin = jnp.reshape(x[n:], (ns, K))
    return Lcos, Lsin


def _cg_solve(
    matvec: Callable[[Any], Any],
    b: Any,
    *,
    x0: Any | None = None,
    tol: float,
    max_iter: int,
) -> Any:
    """Conjugate gradients for SPD systems (JAX-friendly)."""
    b = jnp.asarray(b)
    x = jnp.zeros_like(b) if x0 is None else jnp.asarray(x0)
    r = b - matvec(x)
    p = r
    rs = jnp.dot(r, r)

    tol2 = jnp.asarray(float(tol) ** 2, dtype=b.dtype)
    max_iter = int(max_iter)

    def cond_fun(carry):
        i, x, r, p, rs = carry
        return jnp.logical_and(i < max_iter, rs > tol2)

    def body_fun(carry):
        i, x, r, p, rs = carry
        Ap = matvec(p)
        alpha = rs / jnp.dot(p, Ap)
        x_new = x + alpha * p
        r_new = r - alpha * Ap
        rs_new = jnp.dot(r_new, r_new)
        beta = rs_new / rs
        p_new = r_new + beta * p
        return i + 1, x_new, r_new, p_new, rs_new

    _i, x, _r, _p, _rs = jax.lax.while_loop(cond_fun, body_fun, (0, x, r, p, rs))
    return x


def _lineax_bicgstab_solve(
    matvec: Callable[[Any], Any],
    b: Any,
    *,
    x0: Any | None = None,
    tol: float,
    max_iter: int,
):
    """Solve a square linear system with lineax when available.

    This is kept as an optional feature-gated path so we can benchmark whether
    lineax materially improves the residual adjoint solve before depending on it
    more broadly.
    """
    return _lineax_bicgstab_solve_impl(
        matvec,
        b,
        x0=x0,
        tol=tol,
        max_iter=max_iter,
        lineax_module=lx,
        jax_module=jax,
    )


def _linear_map_jacobian_columns(
    linear_map: Callable[[Any], Any],
    *,
    input_size: int,
    output_size: int,
    dtype: Any,
    chunk_size: int,
):
    """Build a dense Jacobian by batching JVP columns in chunks.

    This mirrors DESC's chunked Jacobian construction pattern: apply the same
    linearized map to blocks of basis directions instead of tracing a separate
    Jacobian transform for every column.
    """
    return _linear_map_jacobian_columns_impl(
        linear_map,
        input_size=input_size,
        output_size=output_size,
        dtype=dtype,
        chunk_size=chunk_size,
    )


def solve_lambda_state_implicit(
    state0: VMECState,
    static,
    *,
    phipf,
    chipf,
    signgs: int,
    lamscale,
    sqrtg: Any | None = None,
    max_iter: int = 60,
    step_size: float = 0.05,
    grad_tol: float | None = None,
    bt_factor: float = 0.5,
    max_backtracks: int = 12,
    implicit: ImplicitLambdaOptions | None = None,
) -> VMECState:
    """Solve lambda with a custom VJP that uses implicit differentiation.

    Notes
    -----
    - This wrapper intentionally treats ``state0`` and ``static`` as *constants*
      for differentiation purposes. Gradients are provided w.r.t. ``phipf``,
      ``chipf``, and ``lamscale``.
    - The backward pass solves a damped linear system involving the Hessian of
      ``wb(L)`` w.r.t. lambda coefficients using conjugate gradients and Hessian-
      vector products computed via ``jax.jvp``.
    """
    if not has_jax():
        raise ImportError("solve_lambda_state_implicit requires JAX (jax + jaxlib)")

    implicit = implicit or ImplicitLambdaOptions()

    state0_c = _stop_gradient_tree(state0)

    idx00 = _mode00_index(static.modes)
    nfp = int(static.cfg.nfp)

    # Metric depends only on R/Z, so compute it once from the fixed geometry.
    g0 = eval_geom(state0_c, static)
    gtt = jnp.asarray(g0.g_tt)
    gtp = jnp.asarray(g0.g_tp)
    gpp = jnp.asarray(g0.g_pp)
    sqrtg_use = jnp.asarray(g0.sqrtg if sqrtg is None else sqrtg)

    s = jnp.asarray(static.s)
    theta = jnp.asarray(static.grid.theta)
    zeta = jnp.asarray(static.grid.zeta)
    if s.shape[0] < 2:
        ds = jnp.asarray(1.0, dtype=s.dtype)
    else:
        ds = s[1] - s[0]
    dtheta_f, dzeta_f = angle_steps(ntheta=int(theta.shape[0]), nzeta=int(zeta.shape[0]))
    dtheta = jnp.asarray(dtheta_f, dtype=s.dtype)
    dzeta = jnp.asarray(dzeta_f, dtype=s.dtype)
    weight = ds * dtheta * dzeta

    signgs_i = int(signgs)

    def _wb_from_L(Lcos, Lsin, phipf, chipf, lamscale):
        lam_u = eval_fourier_dtheta(Lcos, Lsin, static.basis, coeffs_internal=True)
        lam_v = eval_fourier_dzeta_phys(Lcos, Lsin, static.basis, coeffs_internal=True) / nfp
        bsupu, bsupv = bsup_from_sqrtg_lambda(
            sqrtg=sqrtg_use,
            lam_u=lam_u,
            lam_v=lam_v,
            phipf=phipf,
            chipf=chipf,
            signgs=signgs_i,
            lamscale=lamscale,
        )
        B2 = gtt * bsupu**2 + 2.0 * gtp * bsupu * bsupv + gpp * bsupv**2
        jac = signgs_i * sqrtg_use
        E_total = jnp.sum(0.5 * B2 * jac) * weight
        return E_total / (TWOPI * TWOPI)

    def _solve(phipf, chipf, lamscale):
        # Forward solve uses the existing robust optimizer (not differentiated through).
        res = solve_lambda_gd(
            state0_c,
            static,
            phipf=phipf,
            chipf=chipf,
            signgs=signgs_i,
            lamscale=lamscale,
            sqrtg=sqrtg_use,
            max_iter=int(max_iter),
            step_size=float(step_size),
            grad_tol=None if grad_tol is None else float(grad_tol),
            bt_factor=float(bt_factor),
            max_backtracks=int(max_backtracks),
        )
        return res.state

    @jax.custom_vjp
    def _solve_cust(phipf, chipf, lamscale):
        return _solve(phipf, chipf, lamscale)

    def fwd(phipf, chipf, lamscale):
        st = _solve(phipf, chipf, lamscale)
        return st, (jnp.asarray(st.Lcos), jnp.asarray(st.Lsin), jnp.asarray(phipf), jnp.asarray(chipf), jnp.asarray(lamscale))

    def bwd(residual, ct_state):
        Lcos_star, Lsin_star, phipf_star, chipf_star, lamscale_star = residual
        ns, K = Lcos_star.shape

        # Gauge mask: exclude (m,n)=(0,0) everywhere.
        mask = jnp.ones((ns, K), dtype=Lcos_star.dtype)
        if idx00 is not None:
            mask = mask.at[:, int(idx00)].set(0.0)

        def grad_L_flat(Lcos, Lsin, phipf, chipf, lamscale):
            gcos, gsin = jax.grad(_wb_from_L, argnums=(0, 1))(Lcos, Lsin, phipf, chipf, lamscale)
            gcos = gcos * mask
            gsin = gsin * mask
            return _flatten_L(gcos, gsin)

        def Hvp(u_flat):
            ucos, usin = _unflatten_L(u_flat, shape=(ns, K))
            ucos = ucos * mask
            usin = usin * mask

            def grad_pair(Lcos, Lsin):
                gcos, gsin = jax.grad(_wb_from_L, argnums=(0, 1))(Lcos, Lsin, phipf_star, chipf_star, lamscale_star)
                gcos = gcos * mask
                gsin = gsin * mask
                return gcos, gsin

            (_gcos, _gsin), (tcos, tsin) = jax.jvp(grad_pair, (Lcos_star, Lsin_star), (ucos, usin))
            t = _flatten_L(tcos * mask, tsin * mask)
            return t + jnp.asarray(float(implicit.damping), dtype=t.dtype) * u_flat

        # Right-hand side is cotangent w.r.t. the output state's lambda coefficients.
        ct_Lcos = jnp.asarray(ct_state.Lcos) * mask
        ct_Lsin = jnp.asarray(ct_state.Lsin) * mask
        b = _flatten_L(ct_Lcos, ct_Lsin)

        v = _cg_solve(Hvp, b, tol=float(implicit.cg_tol), max_iter=int(implicit.cg_max_iter))

        # Gradient w.r.t. parameters: dL/dp = - v^T (∂/∂p grad_L).
        def F_params(phipf, chipf, lamscale):
            return grad_L_flat(Lcos_star, Lsin_star, phipf, chipf, lamscale)

        (_out, vjp_fun) = jax.vjp(F_params, phipf_star, chipf_star, lamscale_star)
        dphipf, dchipf, dlamscale = vjp_fun(v)
        return (-dphipf, -dchipf, -dlamscale)

    _solve_cust.defvjp(fwd, bwd)

    return _solve_cust(jnp.asarray(phipf), jnp.asarray(chipf), jnp.asarray(lamscale))


def solve_fixed_boundary_state_implicit(
    state0: VMECState,
    static,
    *,
    phipf,
    chipf,
    signgs: int,
    lamscale,
    pressure,
    gamma: float = 0.0,
    jacobian_penalty: float = 1e3,
    solver: str = "lbfgs",
    max_iter: int = 25,
    step_size: float = 5e-3,
    history_size: int = 10,
    grad_tol: float | None = None,
    max_backtracks: int = 12,
    bt_factor: float = 0.5,
    preconditioner: str = "none",
    precond_exponent: float = 1.0,
    precond_radial_alpha: float = 0.0,
    edge_Rcos: Any | None = None,
    edge_Rsin: Any | None = None,
    edge_Zcos: Any | None = None,
    edge_Zsin: Any | None = None,
    implicit_converge_tol: float | None = None,
    implicit_zero_unconverged: bool = True,
    implicit: ImplicitFixedBoundaryOptions | None = None,
) -> VMECState:
    """Fixed-boundary solve with a custom VJP using implicit differentiation.

    This is a building block: it returns an equilibrium state while
    exposing *implicit* gradients w.r.t. the 1D profiles/fluxes.

    Differentiable inputs (by design)
    ---------------------------------
    - ``phipf(s)``, ``chipf(s)``, ``pressure(s)``, and ``lamscale``.
    - Optional boundary edge coefficients (``edge_Rcos``/``edge_Rsin``/``edge_Zcos``/``edge_Zsin``).

    Notes
    -----
    - ``state0`` and ``static`` are treated as constants for differentiation.
    - The backward pass solves a damped linear system involving the Hessian of
      the total objective w.r.t. the (masked) Fourier coefficients using CG and
      Hessian-vector products computed via ``jax.jvp``.
    """
    if not has_jax():
        raise ImportError("solve_fixed_boundary_state_implicit requires JAX (jax + jaxlib)")

    implicit = implicit or ImplicitFixedBoundaryOptions()

    solver = str(solver).strip().lower()
    if solver not in ("gd", "lbfgs"):
        raise ValueError(f"solver must be 'gd' or 'lbfgs', got {solver!r}")

    state0_c = _stop_gradient_tree(state0)
    idx00 = _mode00_index(static.modes)

    signgs_i = int(signgs)
    nfp = int(static.cfg.nfp)
    gamma = float(gamma)
    jacobian_penalty = float(jacobian_penalty)

    s = jnp.asarray(static.s)
    theta = jnp.asarray(static.grid.theta)
    zeta = jnp.asarray(static.grid.zeta)
    if s.shape[0] < 2:
        ds = jnp.asarray(1.0, dtype=s.dtype)
    else:
        ds = s[1] - s[0]
    dtheta_f, dzeta_f = angle_steps(ntheta=int(theta.shape[0]), nzeta=int(zeta.shape[0]))
    dtheta = jnp.asarray(dtheta_f, dtype=s.dtype)
    dzeta = jnp.asarray(dzeta_f, dtype=s.dtype)
    weight = ds * dtheta * dzeta

    edge_any = any(x is not None for x in (edge_Rcos, edge_Rsin, edge_Zcos, edge_Zsin))
    if edge_any and not all(x is not None for x in (edge_Rcos, edge_Rsin, edge_Zcos, edge_Zsin)):
        raise ValueError("edge_Rcos/edge_Rsin/edge_Zcos/edge_Zsin must be provided together")

    edge_Rcos_use = jnp.asarray(edge_Rcos) if edge_any else jnp.asarray(state0_c.Rcos)[-1, :]
    edge_Rsin_use = jnp.asarray(edge_Rsin) if edge_any else jnp.asarray(state0_c.Rsin)[-1, :]
    edge_Zcos_use = jnp.asarray(edge_Zcos) if edge_any else jnp.asarray(state0_c.Zcos)[-1, :]
    edge_Zsin_use = jnp.asarray(edge_Zsin) if edge_any else jnp.asarray(state0_c.Zsin)[-1, :]

    def _objective(state: VMECState, phipf, chipf, pressure, lamscale, edge_Rcos, edge_Rsin, edge_Zcos, edge_Zsin):
        state_use = _enforce_fixed_boundary_and_axis(
            state,
            static,
            edge_Rcos=edge_Rcos,
            edge_Rsin=edge_Rsin,
            edge_Zcos=edge_Zcos,
            edge_Zsin=edge_Zsin,
            enforce_axis=True,
            enforce_edge=True,
            enforce_lambda_axis=True,
            idx00=idx00,
        )
        g = eval_geom(state_use, static)
        bsupu, bsupv = bsup_from_geom(g, phipf=phipf, chipf=chipf, nfp=nfp, signgs=signgs_i, lamscale=lamscale)
        B2 = b2_from_bsup(g, bsupu, bsupv)
        jac = signgs_i * g.sqrtg
        wb = (jnp.sum(0.5 * B2 * jac) * weight) / (TWOPI * TWOPI)
        wp = (jnp.sum(pressure[:, None, None] * jac) * weight) / (TWOPI * TWOPI)
        w = wb + wp / (gamma - 1.0)
        # Softly enforce a consistent Jacobian sign away from the axis.
        jac2 = jac.at[0, :, :].set(0.0)
        neg = jnp.minimum(jac2, 0.0)
        penalty = jacobian_penalty * jnp.mean(neg * neg)
        return w + penalty

    def _grad_flat(state: VMECState, phipf, chipf, pressure, lamscale, edge_Rcos, edge_Rsin, edge_Zcos, edge_Zsin):
        g = jax.grad(_objective)(state, phipf, chipf, pressure, lamscale, edge_Rcos, edge_Rsin, edge_Zcos, edge_Zsin)
        g = _mask_grad_for_constraints(g, static, idx00=idx00)
        return pack_state(g)

    def _solve(phipf, chipf, pressure, lamscale, *, edge_Rcos, edge_Rsin, edge_Zcos, edge_Zsin):
        traced = _contains_jax_tracer(phipf, chipf, pressure, lamscale, edge_Rcos, edge_Rsin, edge_Zcos, edge_Zsin)
        solver_use = solver
        if traced and solver_use != "gd":
            solver_use = "gd"

        if solver_use == "gd":
            res = solve_fixed_boundary_gd(
                state0_c,
                static,
                phipf=phipf,
                chipf=chipf,
                signgs=signgs_i,
                lamscale=lamscale,
                edge_Rcos=edge_Rcos,
                edge_Rsin=edge_Rsin,
                edge_Zcos=edge_Zcos,
                edge_Zsin=edge_Zsin,
                pressure=pressure,
                gamma=gamma,
                jacobian_penalty=jacobian_penalty,
                max_iter=int(max_iter),
                step_size=float(step_size),
                grad_tol=None if grad_tol is None else float(grad_tol),
                max_backtracks=int(max_backtracks),
                bt_factor=float(bt_factor),
                preconditioner=str(preconditioner),
                precond_exponent=float(precond_exponent),
                precond_radial_alpha=float(precond_radial_alpha),
                differentiable=traced,
                jit_grad=traced,
                verbose=False,
            )
        else:
            res = solve_fixed_boundary_lbfgs(
                state0_c,
                static,
                phipf=phipf,
                chipf=chipf,
                signgs=signgs_i,
                lamscale=lamscale,
                edge_Rcos=edge_Rcos,
                edge_Rsin=edge_Rsin,
                edge_Zcos=edge_Zcos,
                edge_Zsin=edge_Zsin,
                pressure=pressure,
                gamma=gamma,
                history_size=int(history_size),
                max_iter=int(max_iter),
                step_size=float(step_size),
                grad_tol=None if grad_tol is None else float(grad_tol),
                max_backtracks=int(max_backtracks),
                bt_factor=float(bt_factor),
                preconditioner=str(preconditioner),
                precond_exponent=float(precond_exponent),
                precond_radial_alpha=float(precond_radial_alpha),
                verbose=False,
            )
            if int(getattr(res, "n_iter", 0)) <= 0:
                # L-BFGS failed to find a decreasing step; fall back to a more
                # conservative GD run to keep implicit gradients meaningful.
                max_iter_fb = max(50, int(max_iter))
                step_size_fb = min(0.2, float(step_size) * 0.2)
                res = solve_fixed_boundary_gd(
                    state0_c,
                    static,
                    phipf=phipf,
                    chipf=chipf,
                    signgs=signgs_i,
                    lamscale=lamscale,
                    edge_Rcos=edge_Rcos,
                    edge_Rsin=edge_Rsin,
                    edge_Zcos=edge_Zcos,
                    edge_Zsin=edge_Zsin,
                    pressure=pressure,
                    gamma=gamma,
                    jacobian_penalty=jacobian_penalty,
                    max_iter=max_iter_fb,
                    step_size=step_size_fb,
                    grad_tol=None if grad_tol is None else float(grad_tol),
                    max_backtracks=int(max_backtracks),
                    bt_factor=float(bt_factor),
                    preconditioner=str(preconditioner),
                    precond_exponent=float(precond_exponent),
                    precond_radial_alpha=float(precond_radial_alpha),
                    verbose=False,
                )
        grad_hist = getattr(res, "grad_rms_history", None)
        converged = False
        if grad_hist is not None and len(grad_hist) > 0:
            try:
                if implicit_converge_tol is not None:
                    tol_check = float(implicit_converge_tol)
                else:
                    tol_check = getattr(getattr(res, "diagnostics", {}), "get", lambda *_args, **_kwargs: None)("grad_tol")
                    tol_check = None if tol_check is None else float(tol_check)
                converged = bool((tol_check is not None) and (float(grad_hist[-1]) < float(tol_check)))
            except Exception:
                converged = False
        return res.state, converged

    @jax.custom_vjp
    def _solve_cust(phipf, chipf, pressure, lamscale, edge_Rcos, edge_Rsin, edge_Zcos, edge_Zsin):
        return _solve(
            phipf,
            chipf,
            pressure,
            lamscale,
            edge_Rcos=edge_Rcos,
            edge_Rsin=edge_Rsin,
            edge_Zcos=edge_Zcos,
            edge_Zsin=edge_Zsin,
        )[0]

    def fwd(phipf, chipf, pressure, lamscale, edge_Rcos, edge_Rsin, edge_Zcos, edge_Zsin):
        st, converged = _solve(
            phipf,
            chipf,
            pressure,
            lamscale,
            edge_Rcos=edge_Rcos,
            edge_Rsin=edge_Rsin,
            edge_Zcos=edge_Zcos,
            edge_Zsin=edge_Zsin,
        )
        return st, (
            _stop_gradient_tree(st),
            jnp.asarray(phipf),
            jnp.asarray(chipf),
            jnp.asarray(pressure),
            jnp.asarray(lamscale),
            jnp.asarray(edge_Rcos),
            jnp.asarray(edge_Rsin),
            jnp.asarray(edge_Zcos),
            jnp.asarray(edge_Zsin),
            bool(edge_any),
            bool(converged),
        )

    def bwd(residual, ct_state):
        (
            st_star,
            phipf_star,
            chipf_star,
            pressure_star,
            lamscale_star,
            edge_Rcos_star,
            edge_Rsin_star,
            edge_Zcos_star,
            edge_Zsin_star,
            edge_active,
            converged,
        ) = residual
        if (not bool(converged)) and bool(implicit_zero_unconverged):
            z = jnp.zeros_like(jnp.asarray(phipf_star))
            zc = jnp.zeros_like(jnp.asarray(chipf_star))
            zp = jnp.zeros_like(jnp.asarray(pressure_star))
            zl = jnp.zeros_like(jnp.asarray(lamscale_star))
            zr = jnp.zeros_like(jnp.asarray(edge_Rcos_star))
            zs = jnp.zeros_like(jnp.asarray(edge_Rsin_star))
            zc2 = jnp.zeros_like(jnp.asarray(edge_Zcos_star))
            zz = jnp.zeros_like(jnp.asarray(edge_Zsin_star))
            return (z, zc, zp, zl, zr, zs, zc2, zz)
        layout = st_star.layout

        ct_state_full = ct_state
        ct_state = _mask_grad_for_constraints(ct_state, static, idx00=idx00)
        b = pack_state(ct_state)

        def Hvp(u_flat):
            u_state = unpack_state(u_flat, layout)
            u_state = _mask_grad_for_constraints(u_state, static, idx00=idx00)
            _, hvp = jax.jvp(
                lambda st: _grad_flat(
                    st,
                    phipf_star,
                    chipf_star,
                    pressure_star,
                    lamscale_star,
                    edge_Rcos_star,
                    edge_Rsin_star,
                    edge_Zcos_star,
                    edge_Zsin_star,
                ),
                (st_star,),
                (u_state,),
            )
            return hvp + jnp.asarray(float(implicit.damping), dtype=hvp.dtype) * u_flat

        v = _cg_solve(Hvp, b, tol=float(implicit.cg_tol), max_iter=int(implicit.cg_max_iter))

        def F_params(phipf, chipf, pressure, lamscale, edge_Rcos, edge_Rsin, edge_Zcos, edge_Zsin):
            return _grad_flat(
                st_star,
                phipf,
                chipf,
                pressure,
                lamscale,
                edge_Rcos,
                edge_Rsin,
                edge_Zcos,
                edge_Zsin,
            )

        (_out, vjp_fun) = jax.vjp(
            F_params,
            phipf_star,
            chipf_star,
            pressure_star,
            lamscale_star,
            edge_Rcos_star,
            edge_Rsin_star,
            edge_Zcos_star,
            edge_Zsin_star,
        )
        dphipf, dchipf, dpressure, dlamscale, dRcos, dRsin, dZcos, dZsin = vjp_fun(v)
        # Direct dependence of the output state on boundary parameters.
        ct_edge_Rcos = jnp.asarray(ct_state_full.Rcos)[-1, :]
        ct_edge_Rsin = jnp.asarray(ct_state_full.Rsin)[-1, :]
        ct_edge_Zcos = jnp.asarray(ct_state_full.Zcos)[-1, :]
        ct_edge_Zsin = jnp.asarray(ct_state_full.Zsin)[-1, :]
        dRcos = dRcos + ct_edge_Rcos
        dRsin = dRsin + ct_edge_Rsin
        dZcos = dZcos + ct_edge_Zcos
        dZsin = dZsin + ct_edge_Zsin
        if not bool(edge_active):
            dRcos = jnp.zeros_like(dRcos)
            dRsin = jnp.zeros_like(dRsin)
            dZcos = jnp.zeros_like(dZcos)
            dZsin = jnp.zeros_like(dZsin)
        return (-dphipf, -dchipf, -dpressure, -dlamscale, -dRcos, -dRsin, -dZcos, -dZsin)

    _solve_cust.defvjp(fwd, bwd)

    return _solve_cust(
        jnp.asarray(phipf),
        jnp.asarray(chipf),
        jnp.asarray(pressure),
        jnp.asarray(lamscale),
        jnp.asarray(edge_Rcos_use),
        jnp.asarray(edge_Rsin_use),
        jnp.asarray(edge_Zcos_use),
        jnp.asarray(edge_Zsin_use),
    )


def solve_fixed_boundary_state_implicit_vmec_residual(
    state0: VMECState,
    static,
    *,
    indata,
    signgs: int,
    state0_host: VMECState | None = None,
    max_iter: int = 50,
    step_size: float = 1.0,
    ftol: float | None = None,
    implicit: ImplicitFixedBoundaryOptions | None = None,
    edge_Rcos: Any | None = None,
    edge_Rsin: Any | None = None,
    edge_Zcos: Any | None = None,
    edge_Zsin: Any | None = None,
):
    """Implicitly differentiate a VMEC residual fixed-point solve.

    The forward solve uses ``solve_fixed_boundary_residual_iter`` with the
    VMEC2000-style control path. The backward pass differentiates the VMEC
    residual vector itself rather than an auxiliary energy objective.

    ``implicit.residual_adjoint_mode='auto'`` selects a cheaper reduced-
    coordinate adjoint for the common ``lasym=False`` path when the residual
    dimension matches the active state dimension, falling back to the legacy
    normal-equation CG solve otherwise.
    """
    if not has_jax():
        raise ImportError("solve_fixed_boundary_state_implicit_vmec_residual requires JAX (jax + jaxlib)")

    implicit = implicit or ImplicitFixedBoundaryOptions()
    state0_c = _stop_gradient_tree(state0)
    idx00 = _mode00_index(static.modes)
    signgs_i = int(signgs)

    edge_Rcos_use = jnp.asarray(edge_Rcos) if edge_Rcos is not None else jnp.asarray(state0_c.Rcos)[-1, :]
    edge_Rsin_use = jnp.asarray(edge_Rsin) if edge_Rsin is not None else jnp.asarray(state0_c.Rsin)[-1, :]
    edge_Zcos_use = jnp.asarray(edge_Zcos) if edge_Zcos is not None else jnp.asarray(state0_c.Zcos)[-1, :]
    edge_Zsin_use = jnp.asarray(edge_Zsin) if edge_Zsin is not None else jnp.asarray(state0_c.Zsin)[-1, :]

    setup = _build_vmec_residual_setup(
        state0_c=state0_c,
        static=static,
        indata=indata,
        signgs_i=signgs_i,
        idx00=idx00,
    )
    residual_vector_context = _VmecResidualVectorContext(
        static=static,
        indata=indata,
        signgs_i=signgs_i,
        s=setup.s,
        wout_like=setup.wout_like,
        trig=setup.trig,
        apply_lforbal=setup.apply_lforbal,
        mask_pack=setup.mask_pack,
        stellsym_residual_projector=setup.stellsym_residual_projector,
    )

    def _boundary_state_edge_rows(eRcos, eRsin, eZcos, eZsin):
        return _project_boundary_edge_rows(
            static=static, indata=indata, dtype=jnp.asarray(state0_c.Rcos).dtype,
            eRcos=eRcos, eRsin=eRsin, eZcos=eZcos, eZsin=eZsin,
        )

    def _enforce_state(st, eRcos, eRsin, eZcos, eZsin):
        sRcos, sRsin, sZcos, sZsin = _boundary_state_edge_rows(eRcos, eRsin, eZcos, eZsin)
        return _enforce_fixed_boundary_and_axis(
            st,
            static,
            edge_Rcos=sRcos,
            edge_Rsin=sRsin,
            edge_Zcos=sZcos,
            edge_Zsin=sZsin,
            enforce_lambda_axis=True,
            idx00=idx00,
        )

    _project_state = partial(_mask_grad_for_constraints, static=static, idx00=idx00, mask_lambda_axis=True)

    def _residual_vec(state, zero_m1_zforce, eRcos, eRsin, eZcos, eZsin, *, project_stellsym: bool = False):
        return _vmec_residual_vector_from_state(
            state=state,
            zero_m1_zforce=zero_m1_zforce,
            eRcos=eRcos,
            eRsin=eRsin,
            eZcos=eZcos,
            eZsin=eZsin,
            context=residual_vector_context,
            enforce_state_func=_enforce_state,
            project_stellsym=project_stellsym,
        )

    def _stationarity_state(state, zero_m1_zforce, eRcos, eRsin, eZcos, eZsin, *, project_stellsym: bool = False):
        def _objective_from_state(st):
            residual = _residual_vec(
                st,
                zero_m1_zforce,
                eRcos,
                eRsin,
                eZcos,
                eZsin,
                project_stellsym=project_stellsym,
            )
            residual = jnp.asarray(residual)
            return 0.5 * jnp.sum(residual * residual)

        return _project_state(jax.grad(_objective_from_state)(state))

    host_solve_settings = _VmecResidualHostSolveSettings(
        static=static,
        indata=indata,
        state_layout=state0_c.layout,
        state0_host=state0_host,
        max_iter=int(max_iter),
        step_size=float(step_size),
        ftol=ftol,
    )

    _solve_host = partial(_solve_vmec_residual_host_from_boundary, settings=host_solve_settings)

    def _solve(eRcos, eRsin, eZcos, eZsin):
        traced = _contains_jax_tracer(eRcos, eRsin, eZcos, eZsin)
        if traced:
            out_shape = (
                jax.ShapeDtypeStruct((int(host_solve_settings.state_layout.size),), jnp.asarray(state0_c.Rcos).dtype),
                jax.ShapeDtypeStruct((), jnp.asarray(state0_c.Rcos).dtype),
            )
            x_flat, zero_m1 = jax.pure_callback(_solve_host, out_shape, eRcos, eRsin, eZcos, eZsin)
            return unpack_state(x_flat, host_solve_settings.state_layout), zero_m1
        x_flat, zero_m1 = _solve_host(eRcos, eRsin, eZcos, eZsin)
        return unpack_state(jnp.asarray(x_flat), host_solve_settings.state_layout), jnp.asarray(zero_m1)

    if not _contains_jax_tracer(edge_Rcos_use, edge_Rsin_use, edge_Zcos_use, edge_Zsin_use):
        x_flat, _zero_m1 = _solve_host(edge_Rcos_use, edge_Rsin_use, edge_Zcos_use, edge_Zsin_use)
        return unpack_state(jnp.asarray(x_flat), host_solve_settings.state_layout)

    residual_tangent_mode = str(getattr(implicit, "residual_tangent_mode", "opaque")).strip().lower()

    def _state_tangent_from_boundary_tangent(
        st_star,
        zero_m1_star,
        eRcos_star,
        eRsin_star,
        eZcos_star,
        eZsin_star,
        deRcos,
        deRsin,
        deZcos,
        deZsin,
    ):
        if bool(static.cfg.lasym):
            raise NotImplementedError(
                "residual_tangent_mode='linearize' currently supports only lasym=False residual solves"
            )

        tangent_mode = str(getattr(implicit, "residual_tangent_mode", "auto")).strip().lower()
        rz_idx_np, lam_idx_np, ns_active, K_active = _stellsym_feasible_indices_np(
            static,
            idx00=idx00,
            mask_lambda_axis=True,
        )
        rz_idx = jnp.asarray(rz_idx_np, dtype=jnp.int32)
        lam_idx = jnp.asarray(lam_idx_np, dtype=jnp.int32)
        st_active_ref = _stop_gradient_tree(st_star)
        if _vmec_keep_all_active_enabled():
            x_active_star_full = _pack_stellsym_feasible_state(st_active_ref, rz_idx=rz_idx, lam_idx=lam_idx)
            active_keep_idx = jnp.arange(int(x_active_star_full.shape[0]), dtype=jnp.int32)
            x_active_star = jnp.take(x_active_star_full, active_keep_idx)

            def stationarity_fun_active(x_active):
                x_active_full = x_active_star_full.at[active_keep_idx].set(
                    x_active,
                    indices_are_sorted=True,
                    unique_indices=True,
                )
                st_active = _update_stellsym_feasible_state(
                    st_active_ref,
                    x_active_full,
                    rz_idx=rz_idx,
                    lam_idx=lam_idx,
                    ns=ns_active,
                    K=K_active,
                )
                grad_state = _stationarity_state(
                    st_active,
                    zero_m1_star,
                    eRcos_star,
                    eRsin_star,
                    eZcos_star,
                    eZsin_star,
                )
                grad_active_full = _pack_stellsym_feasible_state(grad_state, rz_idx=rz_idx, lam_idx=lam_idx)
                return jnp.take(grad_active_full, active_keep_idx)

            def stationarity_params_active(a, b, c, d):
                grad_state = _stationarity_state(
                    st_star,
                    zero_m1_star,
                    a,
                    b,
                    c,
                    d,
                )
                grad_active_full = _pack_stellsym_feasible_state(grad_state, rz_idx=rz_idx, lam_idx=lam_idx)
                return jnp.take(grad_active_full, active_keep_idx)
        else:
            z_idx = _stellsym_reduced_z_indices(rz_idx=rz_idx_np, K=int(K_active), idx00=idx00)
            lam_sc_idx, lam_cs_idx, lam_maps = _stellsym_lambda_mn_indices(
                static,
                idx00=idx00,
                mask_lambda_axis=True,
            )
            x_active_star = _pack_stellsym_reduced_state(
                st_active_ref,
                rz_idx=rz_idx,
                z_idx=z_idx,
                lam_sc_idx=lam_sc_idx,
                lam_cs_idx=lam_cs_idx,
                lam_maps=lam_maps,
            )

            def stationarity_fun_active(x_active):
                st_active = _update_stellsym_reduced_state(
                    st_active_ref,
                    x_active,
                    rz_idx=rz_idx,
                    z_idx=z_idx,
                    lam_sc_idx=lam_sc_idx,
                    lam_cs_idx=lam_cs_idx,
                    lam_maps=lam_maps,
                    ns=ns_active,
                    K=K_active,
                )
                grad_state = _stationarity_state(
                    st_active,
                    zero_m1_star,
                    eRcos_star,
                    eRsin_star,
                    eZcos_star,
                    eZsin_star,
                )
                return _pack_stellsym_reduced_state(
                    grad_state,
                    rz_idx=rz_idx,
                    z_idx=z_idx,
                    lam_sc_idx=lam_sc_idx,
                    lam_cs_idx=lam_cs_idx,
                    lam_maps=lam_maps,
                )

            def stationarity_params_active(a, b, c, d):
                grad_state = _stationarity_state(
                    st_star,
                    zero_m1_star,
                    a,
                    b,
                    c,
                    d,
                )
                return _pack_stellsym_reduced_state(
                    grad_state,
                    rz_idx=rz_idx,
                    z_idx=z_idx,
                    lam_sc_idx=lam_sc_idx,
                    lam_cs_idx=lam_cs_idx,
                    lam_maps=lam_maps,
                )

        stationarity_star_active, stationarity_jvp_active = jax.linearize(stationarity_fun_active, x_active_star)
        stationarity_vjp_active = jax.linear_transpose(stationarity_jvp_active, x_active_star)
        boundary_tangent = jax.jvp(
            stationarity_params_active,
            (eRcos_star, eRsin_star, eZcos_star, eZsin_star),
            (deRcos, deRsin, deZcos, deZsin),
        )[1]
        rhs = jnp.asarray(boundary_tangent)

        active_tangent_result = _solve_active_residual_tangent_linearized(
            stationarity_jvp_active,
            stationarity_vjp_active,
            stationarity_star_active=stationarity_star_active,
            x_active_star=x_active_star,
            rhs=rhs,
            tangent_mode=tangent_mode,
            damping=float(implicit.damping),
            cg_tol=float(implicit.cg_tol),
            cg_max_iter=int(implicit.cg_max_iter),
            jac_chunk_size=getattr(implicit, "jac_chunk_size", None),
            cg_solve=_cg_solve,
            bicgstab_solve=_jax_bicgstab_solve,
            lineax_solve=_lineax_bicgstab_solve,
            jacobian_columns=_linear_map_jacobian_columns,
        )
        dx_active = active_tangent_result.dx

        if _vmec_keep_all_active_enabled():
            dx_active_full = jnp.zeros_like(x_active_star_full).at[active_keep_idx].set(
                dx_active,
                indices_are_sorted=True,
                unique_indices=True,
            )
            tangent_state = _update_stellsym_feasible_state(
                _zero_state_like(st_star),
                dx_active_full,
                rz_idx=rz_idx,
                lam_idx=lam_idx,
                ns=ns_active,
                K=K_active,
            )
        else:
            tangent_state = _update_stellsym_reduced_state(
                _zero_state_like(st_star),
                dx_active,
                rz_idx=rz_idx,
                z_idx=z_idx,
                lam_sc_idx=lam_sc_idx,
                lam_cs_idx=lam_cs_idx,
                lam_maps=lam_maps,
                ns=ns_active,
                K=K_active,
            )
        dsRcos, dsRsin, dsZcos, dsZsin = jax.jvp(
            _boundary_state_edge_rows,
            (eRcos_star, eRsin_star, eZcos_star, eZsin_star),
            (deRcos, deRsin, deZcos, deZsin),
        )[1]
        return VMECState(
            layout=tangent_state.layout,
            Rcos=(-jnp.asarray(tangent_state.Rcos)).at[-1].set(jnp.asarray(dsRcos)),
            Rsin=(-jnp.asarray(tangent_state.Rsin)).at[-1].set(jnp.asarray(dsRsin)),
            Zcos=(-jnp.asarray(tangent_state.Zcos)).at[-1].set(jnp.asarray(dsZcos)),
            Zsin=(-jnp.asarray(tangent_state.Zsin)).at[-1].set(jnp.asarray(dsZsin)),
            Lcos=jnp.asarray(tangent_state.Lcos),
            Lsin=-jnp.asarray(tangent_state.Lsin),
        )

    if residual_tangent_mode not in ("", "0", "false", "no", "opaque"):
        @jax.custom_jvp
        def _solve_cust_jvp(eRcos, eRsin, eZcos, eZsin):
            return _solve(eRcos, eRsin, eZcos, eZsin)[0]

        @_solve_cust_jvp.defjvp
        def _solve_cust_jvp_rule(primals, tangents):
            eRcos, eRsin, eZcos, eZsin = primals
            deRcos, deRsin, deZcos, deZsin = tangents
            st, zero_m1 = _solve(eRcos, eRsin, eZcos, eZsin)
            tangent_state = _state_tangent_from_boundary_tangent(
                st,
                zero_m1,
                eRcos,
                eRsin,
                eZcos,
                eZsin,
                jnp.asarray(deRcos),
                jnp.asarray(deRsin),
                jnp.asarray(deZcos),
                jnp.asarray(deZsin),
            )
            return st, tangent_state

        return _solve_cust_jvp(
            jnp.asarray(edge_Rcos_use),
            jnp.asarray(edge_Rsin_use),
            jnp.asarray(edge_Zcos_use),
            jnp.asarray(edge_Zsin_use),
        )

    @jax.custom_vjp
    def _solve_cust(eRcos, eRsin, eZcos, eZsin):
        return _solve(eRcos, eRsin, eZcos, eZsin)[0]

    def fwd(eRcos, eRsin, eZcos, eZsin):
        st, zero_m1 = _solve(eRcos, eRsin, eZcos, eZsin)
        return st, (
            _stop_gradient_tree(st),
            jnp.asarray(zero_m1),
            jnp.asarray(eRcos),
            jnp.asarray(eRsin),
            jnp.asarray(eZcos),
            jnp.asarray(eZsin),
        )

    def bwd(residual, ct_state):
        bwd_start = time.perf_counter()
        _vmec_backward_profile_log("bwd_start")
        st_star, zero_m1_star, eRcos_star, eRsin_star, eZcos_star, eZsin_star = residual
        ct_state_full = ct_state
        ct_state = _project_state(ct_state)
        b = pack_state(ct_state)

        stationarity_fun = lambda st: pack_state(
            _stationarity_state(st, zero_m1_star, eRcos_star, eRsin_star, eZcos_star, eZsin_star)
        )

        def _edge_boundary_vjp():
            return _boundary_edge_rows_vjp(
                _boundary_state_edge_rows,
                (eRcos_star, eRsin_star, eZcos_star, eZsin_star),
                ct_state_full,
            )

        def _boundary_param_vjp_result(vjp_start, cotangent):
            dRcos, dRsin, dZcos, dZsin = cotangent
            edge_dRcos, edge_dRsin, edge_dZcos, edge_dZsin = _edge_boundary_vjp()
            _vmec_backward_profile_log("boundary_param_vjp_done", vjp_start)
            return (
                edge_dRcos - dRcos,
                edge_dRsin - dRsin,
                edge_dZcos - dZcos,
                edge_dZsin - dZsin,
            )

        residual_adjoint_mode = str(getattr(implicit, "residual_adjoint_mode", "auto")).strip().lower()
        if (not bool(static.cfg.lasym)) and (not _vmec_disable_reduced_active_enabled()):
            active_setup_start = time.perf_counter()
            rz_idx_np, lam_idx_np, ns_active, K_active = _stellsym_feasible_indices_np(
                static,
                idx00=idx00,
                mask_lambda_axis=True,
            )
            rz_idx = jnp.asarray(rz_idx_np, dtype=jnp.int32)
            lam_idx = jnp.asarray(lam_idx_np, dtype=jnp.int32)
            active_packing_strategy = select_active_packing_strategy(
                keep_all_active=_vmec_keep_all_active_enabled()
            )
            if active_packing_strategy == "full":
                b_active_full = _pack_stellsym_feasible_state(ct_state, rz_idx=rz_idx, lam_idx=lam_idx)
                active_keep_idx = full_active_keep_indices(b_active_full, dtype=jnp.int32)
                st_active_ref = _stop_gradient_tree(st_star)
                x_active_star_full = _pack_stellsym_feasible_state(st_star, rz_idx=rz_idx, lam_idx=lam_idx)
                x_active_star = jnp.take(x_active_star_full, active_keep_idx)
                b_active = jnp.take(b_active_full, active_keep_idx)

                _vmec_backward_profile_log(
                    "active_setup_done",
                    active_setup_start,
                    active_size=int(np.shape(b_active)[0]),
                    active_full_size=int(np.shape(b_active_full)[0]),
                    residual_mode=residual_adjoint_mode,
                )

                def stationarity_fun_active(x_active):
                    x_active_full = x_active_star_full.at[active_keep_idx].set(
                        x_active,
                        indices_are_sorted=True,
                        unique_indices=True,
                    )
                    st_active = _update_stellsym_feasible_state(
                        st_active_ref,
                        x_active_full,
                        rz_idx=rz_idx,
                        lam_idx=lam_idx,
                        ns=ns_active,
                        K=K_active,
                    )
                    grad_state = _stationarity_state(
                        st_active,
                        zero_m1_star,
                        eRcos_star,
                        eRsin_star,
                        eZcos_star,
                        eZsin_star,
                    )
                    grad_active_full = _pack_stellsym_feasible_state(grad_state, rz_idx=rz_idx, lam_idx=lam_idx)
                    return jnp.take(grad_active_full, active_keep_idx)

                def _boundary_param_vjp_active(lam):
                    vjp_start = time.perf_counter()

                    def G_params(eRcos, eRsin, eZcos, eZsin):
                        grad_state = _stationarity_state(
                            st_star,
                            zero_m1_star,
                            eRcos,
                            eRsin,
                            eZcos,
                            eZsin,
                        )
                        grad_active_full = _pack_stellsym_feasible_state(grad_state, rz_idx=rz_idx, lam_idx=lam_idx)
                        return jnp.take(grad_active_full, active_keep_idx)

                    _, vjp_fun = jax.vjp(G_params, eRcos_star, eRsin_star, eZcos_star, eZsin_star)
                    return _boundary_param_vjp_result(vjp_start, vjp_fun(jnp.asarray(lam)))
            else:
                z_idx = _stellsym_reduced_z_indices(rz_idx=rz_idx_np, K=int(K_active), idx00=idx00)
                lam_sc_idx, lam_cs_idx, lam_maps = _stellsym_lambda_mn_indices(
                    static,
                    idx00=idx00,
                    mask_lambda_axis=True,
                )
                st_active_ref = _stop_gradient_tree(st_star)
                x_active_star = _pack_stellsym_reduced_state(
                    st_active_ref,
                    rz_idx=rz_idx,
                    z_idx=z_idx,
                    lam_sc_idx=lam_sc_idx,
                    lam_cs_idx=lam_cs_idx,
                    lam_maps=lam_maps,
                )
                b_active = _pack_stellsym_reduced_state(
                    ct_state,
                    rz_idx=rz_idx,
                    z_idx=z_idx,
                    lam_sc_idx=lam_sc_idx,
                    lam_cs_idx=lam_cs_idx,
                    lam_maps=lam_maps,
                )

                _vmec_backward_profile_log(
                    "active_setup_done",
                    active_setup_start,
                    active_size=int(np.shape(b_active)[0]),
                    active_full_size=int(np.shape(b_active)[0]),
                    residual_mode=residual_adjoint_mode,
                )

                def stationarity_fun_active(x_active):
                    st_active = _update_stellsym_reduced_state(
                        st_active_ref,
                        x_active,
                        rz_idx=rz_idx,
                        z_idx=z_idx,
                        lam_sc_idx=lam_sc_idx,
                        lam_cs_idx=lam_cs_idx,
                        lam_maps=lam_maps,
                        ns=ns_active,
                        K=K_active,
                    )
                    grad_state = _stationarity_state(
                        st_active,
                        zero_m1_star,
                        eRcos_star,
                        eRsin_star,
                        eZcos_star,
                        eZsin_star,
                    )
                    return _pack_stellsym_reduced_state(
                        grad_state,
                        rz_idx=rz_idx,
                        z_idx=z_idx,
                        lam_sc_idx=lam_sc_idx,
                        lam_cs_idx=lam_cs_idx,
                        lam_maps=lam_maps,
                    )

                def _boundary_param_vjp_active(lam):
                    vjp_start = time.perf_counter()

                    def G_params(eRcos, eRsin, eZcos, eZsin):
                        grad_state = _stationarity_state(
                            st_star,
                            zero_m1_star,
                            eRcos,
                            eRsin,
                            eZcos,
                            eZsin,
                        )
                        return _pack_stellsym_reduced_state(
                            grad_state,
                            rz_idx=rz_idx,
                            z_idx=z_idx,
                            lam_sc_idx=lam_sc_idx,
                            lam_cs_idx=lam_cs_idx,
                            lam_maps=lam_maps,
                        )

                    _, vjp_fun = jax.vjp(G_params, eRcos_star, eRsin_star, eZcos_star, eZsin_star)
                    return _boundary_param_vjp_result(vjp_start, vjp_fun(jnp.asarray(lam)))

            active_linearize_start = time.perf_counter()
            residual_star_active, residual_jvp_active = jax.linearize(stationarity_fun_active, x_active_star)
            residual_vjp_active = jax.linear_transpose(residual_jvp_active, x_active_star)
            _vmec_backward_profile_log(
                "active_linearize_done",
                active_linearize_start,
                residual_size=int(np.prod(np.shape(residual_star_active))),
            )

            active_adjoint_result = _solve_active_residual_adjoint_linearized(
                residual_jvp_active,
                residual_vjp_active,
                residual_star_active=residual_star_active,
                b_active=b_active,
                x_active_star=x_active_star,
                residual_adjoint_mode=residual_adjoint_mode,
                damping=float(implicit.damping),
                cg_tol=float(implicit.cg_tol),
                cg_max_iter=int(implicit.cg_max_iter),
                jac_chunk_size=getattr(implicit, "jac_chunk_size", None),
                dense_transpose_lstsq_host=_dense_transpose_lstsq_host,
                is_traced=_contains_jax_tracer,
                cg_solve=_cg_solve,
                bicgstab_solve=_jax_bicgstab_solve,
                lineax_solve=_lineax_bicgstab_solve,
                jacobian_columns=_linear_map_jacobian_columns,
                profile_log=_vmec_backward_profile_log,
                time_module=time,
            )
            result = _boundary_param_vjp_active(active_adjoint_result.lam)
            done_stage = {
                "dense": "bwd_done_chunked",
                "bicgstab": "bwd_done_direct",
                "lineax": "bwd_done_lineax",
                "cg": "bwd_done_active",
            }.get(active_adjoint_result.route, "bwd_done_active")
            _vmec_backward_profile_log(done_stage, bwd_start)
            return result

        def _boundary_param_vjp_full(lam):
            vjp_start = time.perf_counter()

            def G_params(eRcos, eRsin, eZcos, eZsin):
                return pack_state(
                    _stationarity_state(st_star, zero_m1_star, eRcos, eRsin, eZcos, eZsin)
                )

            _, vjp_fun = jax.vjp(G_params, eRcos_star, eRsin_star, eZcos_star, eZsin_star)
            return _boundary_param_vjp_result(vjp_start, vjp_fun(jnp.asarray(lam)))

        linearize_start = time.perf_counter()
        residual_star, residual_jvp = jax.linearize(stationarity_fun, st_star)
        residual_vjp = jax.linear_transpose(residual_jvp, st_star)
        _vmec_backward_profile_log("full_linearize_done", linearize_start, residual_size=int(np.prod(np.shape(residual_star))))
        full_adjoint_result = _solve_full_residual_adjoint_linearized(
            residual_jvp,
            residual_vjp,
            residual_star=residual_star,
            b=b,
            st_star=st_star,
            damping=float(implicit.damping),
            cg_tol=float(implicit.cg_tol),
            cg_max_iter=int(implicit.cg_max_iter),
            cg_solve=_cg_solve,
            unpack_state=unpack_state,
            pack_state=pack_state,
            project_state=_project_state,
            make_full_normal_map_func=make_full_normal_map,
            validate_full_shapes=validate_full_adjoint_shapes,
            profile_log=_vmec_backward_profile_log,
            time_module=time,
        )
        result = _boundary_param_vjp_full(full_adjoint_result.lam)
        _vmec_backward_profile_log("bwd_done_full", bwd_start)
        return result

    _solve_cust.defvjp(fwd, bwd)
    return _solve_cust(
        jnp.asarray(edge_Rcos_use),
        jnp.asarray(edge_Rsin_use),
        jnp.asarray(edge_Zcos_use),
        jnp.asarray(edge_Zsin_use),
    )
