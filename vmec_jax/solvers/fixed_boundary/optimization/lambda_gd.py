"""Lambda-only fixed-geometry optimizer implementation."""

from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np

from ....field import TWOPI
from ..results import SolveLambdaResult
from ....state import VMECState


def solve_lambda_gd_impl(
    state0: VMECState,
    static,
    *,
    phipf,
    chipf,
    signgs: int,
    lamscale,
    sqrtg: Any | None = None,
    max_iter: int = 50,
    step_size: float = 0.05,
    grad_tol: float | None = None,
    max_backtracks: int = 16,
    bt_factor: float = 0.5,
    jit_grad: bool = False,
    preconditioner: str = "none",
    precond_exponent: float = 1.0,
    precond_radial_alpha: float = 0.0,
    verbose: bool = True,
    has_jax_func: Callable[[], bool] | None = None,
    validate_options_func: Callable[..., Any] | None = None,
    mode00_index_func: Callable[..., int] | None = None,
    eval_geom_func: Callable[..., Any] | None = None,
    eval_fourier_dtheta_func: Callable[..., Any] | None = None,
    eval_fourier_dzeta_phys_func: Callable[..., Any] | None = None,
    bsup_from_sqrtg_lambda_func: Callable[..., tuple[Any, Any]] | None = None,
    angle_steps_func: Callable[..., tuple[float, float]] | None = None,
    enforce_lambda_gauge_func: Callable[..., tuple[Any, Any]] | None = None,
    resolve_grad_tol_func: Callable[..., float] | None = None,
    jax_module: Any | None = None,
    jnp_module: Any | None = None,
    jit_func: Callable[..., Any] | None = None,
) -> SolveLambdaResult:
    """Solve for VMEC lambda with fixed R/Z geometry.

    The public wrapper in :mod:`vmec_jax.solve` injects its module-level aliases
    so existing tests and downstream private monkeypatch hooks keep their
    historical behavior while the implementation lives outside the solver
    monolith.
    """

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
        raise ImportError("solve_lambda_gd requires JAX (jax + jaxlib)")

    if validate_options_func is None:
        from ..options import validate_lambda_gd_options as validate_options_func
    if mode00_index_func is None:
        from .constraints import mode00_index as mode00_index_func
    if eval_geom_func is None:
        from ....geom import eval_geom as eval_geom_func
    if eval_fourier_dtheta_func is None:
        from ....fourier import eval_fourier_dtheta as eval_fourier_dtheta_func
    if eval_fourier_dzeta_phys_func is None:
        from ....fourier import eval_fourier_dzeta_phys as eval_fourier_dzeta_phys_func
    if bsup_from_sqrtg_lambda_func is None:
        from ....field import bsup_from_sqrtg_lambda as bsup_from_sqrtg_lambda_func
    if angle_steps_func is None:
        from ....grids import angle_steps as angle_steps_func
    if enforce_lambda_gauge_func is None:
        from .constraints import enforce_lambda_gauge as enforce_lambda_gauge_func
    if resolve_grad_tol_func is None:
        from .tolerances import resolve_grad_tol as resolve_grad_tol_func

    opts = validate_options_func(
        max_iter=max_iter,
        max_backtracks=max_backtracks,
        bt_factor=bt_factor,
        preconditioner=preconditioner,
        precond_exponent=precond_exponent,
    )
    max_iter = opts.max_iter
    max_backtracks = opts.max_backtracks
    bt_factor = opts.bt_factor
    preconditioner = opts.preconditioner
    precond_exponent = opts.precond_exponent

    idx00 = mode00_index_func(static.modes)

    # Metric depends only on R/Z, so compute it once.
    g0 = eval_geom_func(state0, static)
    gtt = jnp_module.asarray(g0.g_tt)
    gtp = jnp_module.asarray(g0.g_tp)
    gpp = jnp_module.asarray(g0.g_pp)

    sqrtg_use = jnp_module.asarray(g0.sqrtg if sqrtg is None else sqrtg)

    phipf = jnp_module.asarray(phipf)
    chipf = jnp_module.asarray(chipf)
    lamscale = jnp_module.asarray(lamscale)
    signgs = int(signgs)
    nfp = int(static.cfg.nfp)

    s = jnp_module.asarray(static.s)
    theta = jnp_module.asarray(static.grid.theta)
    zeta = jnp_module.asarray(static.grid.zeta)
    if s.shape[0] < 2:
        ds = jnp_module.asarray(1.0, dtype=s.dtype)
    else:
        ds = s[1] - s[0]
    dtheta_f, dzeta_f = angle_steps_func(ntheta=int(theta.shape[0]), nzeta=int(zeta.shape[0]))
    dtheta = jnp_module.asarray(dtheta_f, dtype=s.dtype)
    dzeta = jnp_module.asarray(dzeta_f, dtype=s.dtype)
    weight = ds * dtheta * dzeta

    def _wb_from_L(Lcos, Lsin):
        lam_u = eval_fourier_dtheta_func(Lcos, Lsin, static.basis, coeffs_internal=True)
        lam_v = (
            eval_fourier_dzeta_phys_func(Lcos, Lsin, static.basis, coeffs_internal=True)
            / nfp
        )
        bsupu, bsupv = bsup_from_sqrtg_lambda_func(
            sqrtg=sqrtg_use,
            lam_u=lam_u,
            lam_v=lam_v,
            phipf=phipf,
            chipf=chipf,
            signgs=signgs,
            lamscale=lamscale,
        )
        B2 = gtt * bsupu**2 + 2.0 * gtp * bsupu * bsupv + gpp * bsupv**2
        jac = signgs * sqrtg_use
        E_total = jnp_module.sum(0.5 * B2 * jac) * weight
        return E_total / (TWOPI * TWOPI)

    wb_and_grad = jax_module.value_and_grad(_wb_from_L, argnums=(0, 1))
    wb_only = _wb_from_L
    if jit_grad:
        wb_and_grad = jit_func(wb_and_grad)
        wb_only = jit_func(wb_only)

    Lcos = jnp_module.asarray(state0.Lcos)
    Lsin = jnp_module.asarray(state0.Lsin)
    Lcos, Lsin = enforce_lambda_gauge_func(Lcos, Lsin, idx00=idx00)

    wb0, (gcos, gsin) = wb_and_grad(Lcos, Lsin)
    wb_history = [float(np.asarray(wb0))]
    grad_rms_history = []
    step_history = []
    grad_tol_eff: float | None = None

    for it in range(max_iter):
        # Optional mode-diagonal preconditioning for the lambda subproblem.
        if preconditioner == "mode_diag":
            m = jnp_module.asarray(static.modes.m)
            n = jnp_module.asarray(static.modes.n)
            k2 = m.astype(jnp_module.float64) ** 2 + (
                n.astype(jnp_module.float64) * float(static.cfg.nfp)
            ) ** 2
            w = (1.0 + k2) ** (-precond_exponent)
            w = w.astype(jnp_module.asarray(Lcos).dtype)
            gcos_p = gcos * w[None, :]
            gsin_p = gsin * w[None, :]
        else:
            gcos_p = gcos
            gsin_p = gsin

        grad_rms = float(np.sqrt(np.mean(np.asarray(gcos_p) ** 2 + np.asarray(gsin_p) ** 2)))
        grad_rms_history.append(grad_rms)
        if grad_tol_eff is None:
            grad_tol_eff = resolve_grad_tol_func(
                grad_tol,
                grad_rms0=grad_rms,
                dtype=np.asarray(Lcos).dtype,
            )

        if verbose:
            print(f"[solve_lambda_gd] iter={it:03d} wb={wb_history[-1]:.8e} grad_rms={grad_rms:.3e}")

        if grad_rms < float(grad_tol_eff):
            break

        step = float(step_size)
        accepted = False

        for bt in range(max_backtracks + 1):
            if bt > 0:
                step *= bt_factor
            Lcos_t = Lcos - step * gcos_p
            Lsin_t = Lsin - step * gsin_p
            Lcos_t, Lsin_t = enforce_lambda_gauge_func(Lcos_t, Lsin_t, idx00=idx00)
            wb_t = wb_only(Lcos_t, Lsin_t)
            if float(np.asarray(wb_t)) < wb_history[-1]:
                accepted = True
                Lcos, Lsin, wb0 = Lcos_t, Lsin_t, wb_t
                break

        step_history.append(step)

        if not accepted:
            if verbose:
                print("[solve_lambda_gd] line search failed to improve objective; stopping")
            break

        wb_history.append(float(np.asarray(wb0)))
        wb0, (gcos, gsin) = wb_and_grad(Lcos, Lsin)

    st = VMECState(
        layout=state0.layout,
        Rcos=state0.Rcos,
        Rsin=state0.Rsin,
        Zcos=state0.Zcos,
        Zsin=state0.Zsin,
        Lcos=Lcos,
        Lsin=Lsin,
    )
    diag: Dict[str, Any] = {
        "idx00": idx00,
        "grad_tol": None if grad_tol_eff is None else float(grad_tol_eff),
    }
    return SolveLambdaResult(
        state=st,
        n_iter=len(wb_history) - 1,
        wb_history=np.asarray(wb_history, dtype=float),
        grad_rms_history=np.asarray(grad_rms_history, dtype=float),
        step_history=np.asarray(step_history, dtype=float),
        diagnostics=diag,
    )
