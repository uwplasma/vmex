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
from typing import Any, Callable, Tuple

import numpy as np

from ._compat import has_jax, jax, jnp
from .field import TWOPI, b2_from_bsup, bsup_from_geom, bsup_from_sqrtg_lambda
from .energy import flux_profiles_from_indata
from .fourier import eval_fourier_dtheta, eval_fourier_dzeta_phys
from .geom import eval_geom
from .grids import angle_steps
from .solve import (
    _WoutLikeVmecForces,
    _enforce_fixed_boundary_and_axis,
    _half_mesh_from_full_mesh,
    _icurv_full_mesh_from_indata,
    _mass_half_mesh_from_indata,
    _enforce_lambda_gauge,
    _mask_grad_for_constraints,
    _mode00_index,
    _pressure_half_mesh_from_indata,
    solve_fixed_boundary_gd,
    solve_fixed_boundary_lbfgs,
    solve_fixed_boundary_residual_iter,
    solve_lambda_gd,
)
from .state import VMECState, pack_state, unpack_state


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


def _stop_gradient_tree(x):
    return jax.tree_util.tree_map(jax.lax.stop_gradient, x)


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

        x_star = _flatten_L(Lcos_star, Lsin_star)

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
        def _is_traced(x):
            return isinstance(x, jax.core.Tracer)

        traced = (
            _is_traced(phipf)
            or _is_traced(chipf)
            or _is_traced(pressure)
            or _is_traced(lamscale)
            or _is_traced(edge_Rcos)
            or _is_traced(edge_Rsin)
            or _is_traced(edge_Zcos)
            or _is_traced(edge_Zsin)
        )
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
    """
    if not has_jax():
        raise ImportError("solve_fixed_boundary_state_implicit_vmec_residual requires JAX (jax + jaxlib)")

    implicit = implicit or ImplicitFixedBoundaryOptions()
    state0_c = _stop_gradient_tree(state0)
    if state0_host is None:
        state0_host = VMECState(
            layout=state0_c.layout,
            Rcos=np.asarray(jax.device_get(state0_c.Rcos)),
            Rsin=np.asarray(jax.device_get(state0_c.Rsin)),
            Zcos=np.asarray(jax.device_get(state0_c.Zcos)),
            Zsin=np.asarray(jax.device_get(state0_c.Zsin)),
            Lcos=np.asarray(jax.device_get(state0_c.Lcos)),
            Lsin=np.asarray(jax.device_get(state0_c.Lsin)),
        )
    idx00 = _mode00_index(static.modes)
    signgs_i = int(signgs)

    edge_Rcos_use = jnp.asarray(edge_Rcos) if edge_Rcos is not None else jnp.asarray(state0_c.Rcos)[-1, :]
    edge_Rsin_use = jnp.asarray(edge_Rsin) if edge_Rsin is not None else jnp.asarray(state0_c.Rsin)[-1, :]
    edge_Zcos_use = jnp.asarray(edge_Zcos) if edge_Zcos is not None else jnp.asarray(state0_c.Zcos)[-1, :]
    edge_Zsin_use = jnp.asarray(edge_Zsin) if edge_Zsin is not None else jnp.asarray(state0_c.Zsin)[-1, :]

    from .boundary import boundary_from_indata
    from .vmec_forces import vmec_forces_rz_from_wout, vmec_residual_internal_from_kernels
    from .vmec_residue import (
        vmec_apply_m1_constraints,
        vmec_apply_scalxc_to_tomnsps,
        vmec_force_norms_from_bcovar_dynamic,
        vmec_zero_m1_zforce,
    )
    from .vmec_tomnsp import TomnspsRZL, vmec_trig_tables

    s = jnp.asarray(static.s)
    flux = flux_profiles_from_indata(indata, s, signgs=signgs_i)
    phips = jnp.asarray(flux.phips)
    if phips.shape[0] >= 1:
        phips = phips.at[0].set(0.0)
    chipf_wout = jnp.asarray(flux.chipf)

    boundary = boundary_from_indata(indata, static.modes)
    r00 = float(np.asarray(boundary.R_cos)[int(idx00)]) if int(idx00) >= 0 else float(np.asarray(boundary.R_cos)[0])
    gamma = float(indata.get_float("GAMMA", 0.0))
    lrfp = bool(indata.get_bool("LRFP", False))
    chips = _half_mesh_from_full_mesh(chipf_wout) if lrfp else None
    mass = _mass_half_mesh_from_indata(
        indata=indata,
        s_full=s,
        phips=phips,
        r00=r00,
        gamma=gamma,
        lrfp=lrfp,
        chips=chips,
    )
    pres = _pressure_half_mesh_from_indata(indata=indata, s_full=s)
    ncurr = int(indata.get_int("NCURR", 0))
    icurv = _icurv_full_mesh_from_indata(indata=indata, s_full=s, signgs=signgs_i)
    wout_like = _WoutLikeVmecForces(
        nfp=int(static.cfg.nfp),
        mpol=int(static.cfg.mpol),
        ntor=int(static.cfg.ntor),
        lasym=bool(static.cfg.lasym),
        signgs=signgs_i,
        phipf=jnp.asarray(flux.phipf),
        phips=phips,
        chipf=chipf_wout,
        pres=pres,
        mass=mass,
        gamma=gamma,
        ncurr=ncurr,
        lcurrent=True,
        icurv=icurv,
    )
    trig = getattr(static, "trig_vmec", None)
    if trig is None:
        trig = vmec_trig_tables(
            ntheta=int(static.cfg.ntheta),
            nzeta=int(static.cfg.nzeta),
            nfp=int(wout_like.nfp),
            mmax=int(wout_like.mpol) - 1,
            nmax=int(wout_like.ntor),
            lasym=bool(wout_like.lasym),
            dtype=jnp.asarray(state0.Rcos).dtype,
        )
    apply_lforbal = bool(indata.get_bool("LFORBAL", False))
    mask_pack = getattr(static, "tomnsps_masks", None)

    def _enforce_state(st, eRcos, eRsin, eZcos, eZsin):
        return _enforce_fixed_boundary_and_axis(
            st,
            static,
            edge_Rcos=eRcos,
            edge_Rsin=eRsin,
            edge_Zcos=eZcos,
            edge_Zsin=eZsin,
            enforce_lambda_axis=True,
            idx00=idx00,
        )

    def _project_state(st):
        return _mask_grad_for_constraints(st, static, idx00=idx00, mask_lambda_axis=True)

    def _zero_edge_rz(a):
        a = jnp.asarray(a)
        if a.shape[0] < 2:
            return a
        return a.at[-1].set(jnp.zeros_like(a[-1]))

    def _residual_vec(state, zero_m1_zforce, eRcos, eRsin, eZcos, eZsin):
        state = _enforce_state(state, eRcos, eRsin, eZcos, eZsin)
        k = vmec_forces_rz_from_wout(
            state=state,
            static=static,
            wout=wout_like,
            indata=None,
            constraint_tcon0=float(indata.get_float("TCON0", 1.0)),
            use_vmec_synthesis=True,
            trig=trig,
        )
        frzl = vmec_residual_internal_from_kernels(
            k,
            cfg_ntheta=int(static.cfg.ntheta),
            cfg_nzeta=int(static.cfg.nzeta),
            wout=wout_like,
            trig=trig,
            apply_lforbal=apply_lforbal,
            include_edge=False,
            masks=mask_pack,
        )
        frzl = vmec_apply_m1_constraints(frzl=frzl, lconm1=bool(getattr(static.cfg, "lconm1", True)))
        frzl = vmec_zero_m1_zforce(frzl=frzl, enabled=zero_m1_zforce)
        frzl = vmec_apply_scalxc_to_tomnsps(frzl=frzl, s=s)
        frzl = TomnspsRZL(
            frcc=_zero_edge_rz(frzl.frcc),
            frss=_zero_edge_rz(frzl.frss) if frzl.frss is not None else None,
            fzsc=_zero_edge_rz(frzl.fzsc),
            fzcs=_zero_edge_rz(frzl.fzcs) if frzl.fzcs is not None else None,
            flsc=frzl.flsc,
            flcs=frzl.flcs,
            frsc=_zero_edge_rz(getattr(frzl, "frsc", None)) if getattr(frzl, "frsc", None) is not None else None,
            frcs=_zero_edge_rz(getattr(frzl, "frcs", None)) if getattr(frzl, "frcs", None) is not None else None,
            fzcc=_zero_edge_rz(getattr(frzl, "fzcc", None)) if getattr(frzl, "fzcc", None) is not None else None,
            fzss=_zero_edge_rz(getattr(frzl, "fzss", None)) if getattr(frzl, "fzss", None) is not None else None,
            flcc=getattr(frzl, "flcc", None),
            flss=getattr(frzl, "flss", None),
        )
        norms = vmec_force_norms_from_bcovar_dynamic(bc=k.bc, trig=trig, s=s, signgs=signgs_i)
        scale_rz = jnp.sqrt(norms.r1 * norms.fnorm)
        scale_l = jnp.sqrt(norms.fnormL)
        parts = [scale_rz * frzl.frcc, scale_rz * frzl.fzsc, scale_l * frzl.flsc]
        if frzl.frss is not None:
            parts.append(scale_rz * frzl.frss)
        if frzl.fzcs is not None:
            parts.append(scale_rz * frzl.fzcs)
        if frzl.flcs is not None:
            parts.append(scale_l * frzl.flcs)
        for name in ["frsc", "fzcc", "flcc", "frcs", "fzss", "flss"]:
            arr = getattr(frzl, name, None)
            if arr is not None:
                parts.append((scale_l if name.startswith("fl") else scale_rz) * arr)
        return jnp.concatenate([jnp.ravel(jnp.asarray(p)) for p in parts], axis=0)

    def _solve_host(eRcos, eRsin, eZcos, eZsin):
        eRcos_np = np.asarray(eRcos)
        eRsin_np = np.asarray(eRsin)
        eZcos_np = np.asarray(eZcos)
        eZsin_np = np.asarray(eZsin)
        state_init = VMECState(
            layout=state0_host.layout,
            Rcos=np.array(state0_host.Rcos, copy=True),
            Rsin=np.array(state0_host.Rsin, copy=True),
            Zcos=np.array(state0_host.Zcos, copy=True),
            Zsin=np.array(state0_host.Zsin, copy=True),
            Lcos=np.array(state0_host.Lcos, copy=True),
            Lsin=np.array(state0_host.Lsin, copy=True),
        )
        state_init = _enforce_state(state_init, eRcos_np, eRsin_np, eZcos_np, eZsin_np)
        res = solve_fixed_boundary_residual_iter(
            state_init,
            static,
            indata=indata,
            signgs=signgs_i,
            ftol=ftol,
            max_iter=int(max_iter),
            step_size=float(step_size),
            vmec2000_control=True,
            reference_mode=True,
            backtracking=True,
            limit_dt_from_force=True,
            limit_update_rms=True,
            verbose=False,
            verbose_vmec2000_table=False,
            jit_forces="auto",
            use_scan=False,
        )
        fsqz_hist = np.asarray(getattr(res, "fsqz2_history", []), dtype=float)
        zero_m1 = 1.0 if (int(getattr(res, "n_iter", 0)) < 2 or (fsqz_hist.size > 0 and float(fsqz_hist[-1]) < 1.0e-6)) else 0.0
        return np.asarray(pack_state(res.state)), np.asarray(zero_m1, dtype=state0_host.Rcos.dtype)

    def _is_traced(*xs):
        return any(isinstance(x, jax.core.Tracer) for x in xs)

    def _solve(eRcos, eRsin, eZcos, eZsin):
        traced = _is_traced(eRcos, eRsin, eZcos, eZsin)
        if traced:
            out_shape = (
                jax.ShapeDtypeStruct((int(state0_c.layout.size),), jnp.asarray(state0_c.Rcos).dtype),
                jax.ShapeDtypeStruct((), jnp.asarray(state0_c.Rcos).dtype),
            )
            x_flat, zero_m1 = jax.pure_callback(_solve_host, out_shape, eRcos, eRsin, eZcos, eZsin)
            return unpack_state(x_flat, state0_c.layout), zero_m1
        x_flat, zero_m1 = _solve_host(eRcos, eRsin, eZcos, eZsin)
        return unpack_state(jnp.asarray(x_flat), state0_c.layout), jnp.asarray(zero_m1)

    if not _is_traced(edge_Rcos_use, edge_Rsin_use, edge_Zcos_use, edge_Zsin_use):
        x_flat, _zero_m1 = _solve_host(edge_Rcos_use, edge_Rsin_use, edge_Zcos_use, edge_Zsin_use)
        return unpack_state(jnp.asarray(x_flat), state0_c.layout)

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
        st_star, zero_m1_star, eRcos_star, eRsin_star, eZcos_star, eZsin_star = residual
        ct_state_full = ct_state
        ct_state = _project_state(ct_state)
        b = pack_state(ct_state)

        def Hvp(u_flat):
            u_state = _project_state(unpack_state(u_flat, st_star.layout))
            jv = jax.jvp(
                lambda st: _residual_vec(st, zero_m1_star, eRcos_star, eRsin_star, eZcos_star, eZsin_star),
                (st_star,),
                (u_state,),
            )[1]
            _, vjp_fun = jax.vjp(
                lambda st: _residual_vec(st, zero_m1_star, eRcos_star, eRsin_star, eZcos_star, eZsin_star),
                st_star,
            )
            jt_jv = vjp_fun(jv)[0]
            jt_jv = _project_state(jt_jv)
            return pack_state(jt_jv) + jnp.asarray(float(implicit.damping), dtype=jnp.asarray(jv).dtype) * u_flat

        u = _cg_solve(Hvp, b, tol=float(implicit.cg_tol), max_iter=int(implicit.cg_max_iter))
        u_state = _project_state(unpack_state(u, st_star.layout))
        lam = jax.jvp(
            lambda st: _residual_vec(st, zero_m1_star, eRcos_star, eRsin_star, eZcos_star, eZsin_star),
            (st_star,),
            (u_state,),
        )[1]

        def F_params(eRcos, eRsin, eZcos, eZsin):
            return _residual_vec(st_star, zero_m1_star, eRcos, eRsin, eZcos, eZsin)

        _, vjp_fun = jax.vjp(F_params, eRcos_star, eRsin_star, eZcos_star, eZsin_star)
        dRcos, dRsin, dZcos, dZsin = vjp_fun(lam)

        dRcos = dRcos + jnp.asarray(ct_state_full.Rcos)[-1, :]
        dRsin = dRsin + jnp.asarray(ct_state_full.Rsin)[-1, :]
        dZcos = dZcos + jnp.asarray(ct_state_full.Zcos)[-1, :]
        dZsin = dZsin + jnp.asarray(ct_state_full.Zsin)[-1, :]
        return (-dRcos, -dRsin, -dZcos, -dZsin)

    _solve_cust.defvjp(fwd, bwd)
    return _solve_cust(
        jnp.asarray(edge_Rcos_use),
        jnp.asarray(edge_Rsin_use),
        jnp.asarray(edge_Zcos_use),
        jnp.asarray(edge_Zsin_use),
    )
