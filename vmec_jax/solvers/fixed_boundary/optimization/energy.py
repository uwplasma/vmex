"""Shared fixed-boundary magnetic-energy objective setup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Tuple

from ....field import TWOPI
from ....state import VMECState


@dataclass(frozen=True)
class FixedBoundaryEnergyContext:
    """Prepared arrays and evaluators for fixed-boundary energy optimizers."""

    idx00: int
    signgs: int
    gamma: float
    edge_Rcos: Any
    edge_Rsin: Any
    edge_Zcos: Any
    edge_Zsin: Any
    objective_and_grad: Callable[[VMECState], tuple[Any, Any]]
    objective: Callable[[VMECState], Any]
    w_terms: Callable[[VMECState], tuple[Any, Any, Any]]
    w_terms_and_jacmin: Callable[[VMECState], tuple[Any, Any, Any, Any]]


def prepare_fixed_boundary_energy_context(
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
    jacobian_penalty: float = 0.0,
    jit_grad: bool = False,
    mode00_index_func: Callable[..., int] | None = None,
    eval_geom_func: Callable[..., Any] | None = None,
    bsup_from_geom_func: Callable[..., tuple[Any, Any]] | None = None,
    b2_from_bsup_func: Callable[..., Any] | None = None,
    angle_steps_func: Callable[..., tuple[float, float]] | None = None,
    validate_pressure_shape_func: Callable[..., Any] | None = None,
    jax_module: Any | None = None,
    jnp_module: Any | None = None,
    jit_func: Callable[..., Any] | None = None,
) -> FixedBoundaryEnergyContext:
    """Prepare common fixed-boundary energy objective evaluators.

    The caller injects solve-module aliases so existing monkeypatch-based tests
    and downstream private hooks continue to observe the same behavior.
    """

    if jax_module is None or jnp_module is None or jit_func is None:
        from ...._compat import jax as _jax
        from ...._compat import jit as _jit
        from ...._compat import jnp as _jnp

        jax_module = _jax if jax_module is None else jax_module
        jnp_module = _jnp if jnp_module is None else jnp_module
        jit_func = _jit if jit_func is None else jit_func

    if mode00_index_func is None:
        from .constraints import mode00_index as mode00_index_func
    if eval_geom_func is None:
        from ....geom import eval_geom as eval_geom_func
    if bsup_from_geom_func is None:
        from ....field import bsup_from_geom as bsup_from_geom_func
    if b2_from_bsup_func is None:
        from ....field import b2_from_bsup as b2_from_bsup_func
    if angle_steps_func is None:
        from ....grids import angle_steps as angle_steps_func
    if validate_pressure_shape_func is None:
        from ..options import validate_pressure_shape as validate_pressure_shape_func

    idx00 = mode00_index_func(static.modes)

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

    if pressure is None:
        pressure = jnp_module.zeros_like(s)
    pressure = jnp_module.asarray(pressure)
    validate_pressure_shape_func(tuple(pressure.shape), tuple(s.shape))

    edge_Rcos = (
        jnp_module.asarray(edge_Rcos)
        if edge_Rcos is not None
        else jnp_module.asarray(state0.Rcos)[-1, :]
    )
    edge_Rsin = (
        jnp_module.asarray(edge_Rsin)
        if edge_Rsin is not None
        else jnp_module.asarray(state0.Rsin)[-1, :]
    )
    edge_Zcos = (
        jnp_module.asarray(edge_Zcos)
        if edge_Zcos is not None
        else jnp_module.asarray(state0.Zcos)[-1, :]
    )
    edge_Zsin = (
        jnp_module.asarray(edge_Zsin)
        if edge_Zsin is not None
        else jnp_module.asarray(state0.Zsin)[-1, :]
    )

    def _wb_wp_from_geom(g) -> Tuple[Any, Any]:
        bsupu, bsupv = bsup_from_geom_func(
            g,
            phipf=phipf,
            chipf=chipf,
            nfp=nfp,
            signgs=signgs,
            lamscale=lamscale,
        )
        B2 = b2_from_bsup_func(g, bsupu, bsupv)
        jac = signgs * g.sqrtg
        wb = (jnp_module.sum(0.5 * B2 * jac) * weight) / (TWOPI * TWOPI)
        wp = (jnp_module.sum(pressure[:, None, None] * jac) * weight) / (TWOPI * TWOPI)
        return wb, wp

    def _w_total_from_wb_wp(wb, wp) -> Any:
        return wb + wp / (gamma - 1.0)

    def _objective(state: VMECState) -> Any:
        g = eval_geom_func(state, static)
        wb, wp = _wb_wp_from_geom(g)
        w = _w_total_from_wb_wp(wb, wp)
        if float(jacobian_penalty) <= 0.0:
            return w
        # Softly enforce a consistent Jacobian sign away from the axis (s=0).
        jac = signgs * g.sqrtg
        jac = jac.at[0, :, :].set(0.0)
        neg = jnp_module.minimum(jac, 0.0)
        return w + float(jacobian_penalty) * jnp_module.mean(neg * neg)

    def _w_terms(state: VMECState) -> Tuple[Any, Any, Any]:
        g = eval_geom_func(state, static)
        wb, wp = _wb_wp_from_geom(g)
        return wb, wp, _w_total_from_wb_wp(wb, wp)

    def _w_terms_and_jacmin(state: VMECState) -> Tuple[Any, Any, Any, Any]:
        g = eval_geom_func(state, static)
        wb, wp = _wb_wp_from_geom(g)
        w = _w_total_from_wb_wp(wb, wp)
        jac = signgs * g.sqrtg
        if jac.shape[0] <= 1:
            jac_min = jnp_module.min(jac)
        else:
            jac_min = jnp_module.min(jac[1:, :, :])
        return wb, wp, w, jac_min

    objective_and_grad = jax_module.value_and_grad(_objective)
    w_terms = _w_terms
    w_terms_and_jacmin = _w_terms_and_jacmin
    if jit_grad:
        objective_and_grad = jit_func(objective_and_grad)
        w_terms = jit_func(w_terms)
        w_terms_and_jacmin = jit_func(w_terms_and_jacmin)

    return FixedBoundaryEnergyContext(
        idx00=idx00,
        signgs=signgs,
        gamma=float(gamma),
        edge_Rcos=edge_Rcos,
        edge_Rsin=edge_Rsin,
        edge_Zcos=edge_Zcos,
        edge_Zsin=edge_Zsin,
        objective_and_grad=objective_and_grad,
        objective=_objective,
        w_terms=w_terms,
        w_terms_and_jacmin=w_terms_and_jacmin,
    )
