"""Opt-in Newton solve of free-boundary equilibria.

The default solver is steepest descent on the MHD energy ``W``.  This module
puts the result (or another start) on the root of the coupled plasma--vacuum
force balance with the Krylov Newton anchor of
:mod:`vmex.core.freeboundary_implicit` (Deuflhard-damped, GMRES with the
block-tridiagonal plus NESTOR preconditioner), so a start need not be a
minimum of ``W``.  No stability count is offered: the inertia of VMEC's force
Jacobian was checked against a radially unstable ring (decay index 2.5) and
did not flag it, so it is not an ideal-MHD stability test.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from . import freeboundary_implicit as fbi
from . import implicit as im
from .input import VmecInput
from .restart import state_from_wout
from .solver import evaluate_forces
from .wout import WoutData, wout_from_state

__all__ = ["NewtonResult", "solve_free_boundary_newton"]


@dataclass(frozen=True)
class NewtonResult:
    """Outcome of :func:`solve_free_boundary_newton`.

    ``wout`` and ``state`` are the anchored equilibrium (``wout.fsqr/fsqz/
    fsql`` are VMEC's normalized invariant residuals there); ``residual`` is
    the 2-norm of the projected raw force residual and ``converged`` means the
    anchor certified ``residual <= refine_tol``.
    """

    wout: WoutData
    state: Any
    residual: float
    converged: bool
    newton_steps: int = 0


def solve_free_boundary_newton(
    inp: VmecInput,
    field: Any,
    *,
    start: Any = None,
    refine_tol: float = 1.0e-10,
) -> NewtonResult:
    """Free-boundary equilibrium of ``inp`` in ``field`` by Newton.

    ``inp`` is the deck (``lfreeb`` is forced on) and ``field`` the external
    field, an :class:`~vmex.core.mgrid.MgridField`.  ``start=None`` runs the
    free-boundary descent of the deck (set its ``ftol`` to 1e-6 or tighter)
    and anchors it.  A :class:`~vmex.core.wout.WoutData`, a wout path or a
    :class:`~vmex.core.solver.SpectralState` on the deck's radial grid is
    anchored directly.  Every start is anchored on the same residual: the
    free-boundary loop damps the constraint baseline ``rcon0/zcon0`` by 0.9
    per iteration, so at a converged free-boundary state it is zero, and a
    supplied start uses that zero baseline.  The anchor needs a start
    converged to about 1e-6 or tighter.  Needs ``jax_enable_x64``.
    """
    if not jax.config.jax_enable_x64:
        raise ValueError("solve_free_boundary_newton needs jax_enable_x64")
    free = dataclasses.replace(inp, lfreeb=True)
    cfg = fbi.make_free_boundary_config(free, field, refine_tol=refine_tol)
    icfg = cfg.implicit
    ns = int(icfg.resolution.ns)
    params = im.params_from_input(free)
    base_rt = dataclasses.replace(
        im.runtime_from_params(params, icfg), lfreeb=True, jmax=ns,
        presf_ns_scale=jnp.asarray(fbi._presf_ns_scale(free, ns)))

    def runtime(state, rcon0, zcon0):
        rt = dataclasses.replace(base_rt, rcon0=rcon0, zcon0=zcon0)
        return dataclasses.replace(rt, bsqvac_edge=jax.lax.stop_gradient(
            cfg.vacuum_program.bsq(state, rt, field)))

    if start is None:
        state, mask, rcon0, zcon0 = jax.tree.map(jnp.asarray, fbi._host_solve_and_mask_impl(
            cfg, jax.tree.map(np.asarray, params), jax.tree.map(np.asarray, field),
            error_on_no_convergence=False))
        report = fbi._FREE_LAST_ANCHOR.get(cfg)
    else:
        if isinstance(start, (WoutData, str)) or hasattr(start, "__fspath__"):
            state0 = state_from_wout(start, inp=free, ns=ns)
        else:
            state0 = start
        rcon0 = zcon0 = jnp.zeros_like(base_rt.rcon0)
        rt0 = runtime(state0, rcon0, zcon0)
        mask = jax.tree.map(lambda a: jnp.asarray(np.asarray(a)), im._dof_mask(
            state0, rt0, icfg, evaluator=lambda x: im.evaluate_forces(x, rt0)[0],
            fixed_edge=False))
        state, report = fbi._anchor_root(cfg, params, field, state0, mask, rcon0, zcon0)

    converged = report is not None and bool(report.certified)
    project = im._dof_projector(icfg, mask)
    residual = float(fbi._anchor_norm(project(state), params, field, state, rcon0, zcon0,
                                      mask, cfg=cfg))
    _, norms, _ = evaluate_forces(state, runtime(state, rcon0, zcon0))
    wout = wout_from_state(inp=free, state=state, fsqr=float(norms.fsqr),
                           fsqz=float(norms.fsqz), fsql=float(norms.fsql), converged=converged)
    return NewtonResult(wout=wout, state=state, residual=residual, converged=converged,
                        newton_steps=0 if report is None else int(report.steps))
