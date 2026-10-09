"""Opt-in Newton solve of free-boundary equilibria with an ideal-MHD stability count.

The default solver is steepest descent on the MHD energy ``W``, which cannot
settle on an unstable equilibrium (a saddle of ``W``).  Newton on the same
force balance converges to it.  At the converged state the force Jacobian is
``J = dF/dz = K d2W`` with ``K > 0`` (VMEC's force metric), so ``J`` has the
inertia of the energy Hessian: each negative eigenvalue is a direction with
``delta W < 0``, an ideal-MHD instability.  Eigenvalue magnitudes are in
VMEC's force metric and are not growth rates.

The solve reuses the coupled plasma--vacuum residual and the Krylov Newton
anchor of :mod:`vmex.core.freeboundary_implicit`, and falls back to damped
dense Newton in a reduced orthonormal dof basis when the anchor stalls.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field as _field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from . import freeboundary_implicit as fbi
from . import implicit as im
from .input import VmecInput
from .restart import state_from_wout
from .wout import WoutData, wout_from_state

__all__ = ["NewtonResult", "solve_free_boundary_newton", "count_unstable_modes"]

_CONVERGED = 1.0e-9


@dataclass(frozen=True)
class NewtonResult:
    """Outcome of :func:`solve_free_boundary_newton`.

    ``wout`` and ``state`` are the Newton-converged equilibrium; ``residual``
    is the 2-norm of the projected raw force residual; ``n_unstable`` counts
    negative eigenvalues of the reduced force Jacobian (``-1`` when the
    eigenvalues were not requested); ``modes`` lists the ``nev`` most negative
    eigenvalues as ``(eigenvalue, [(energy_fraction, m, n), ...])``.
    """

    wout: WoutData
    state: Any
    residual: float
    converged: bool
    n_unstable: int
    modes: list = _field(default_factory=list)
    newton_steps: int = 0


def _basis(cfg, mask):
    """Orthonormal basis (n_full x n_dof) of the range of the dof projector."""
    projector = im._dof_projector(cfg, mask)
    flat, unravel = ravel_pytree(jax.tree.map(jnp.zeros_like, mask))
    active = np.nonzero(np.asarray(ravel_pytree(mask)[0]) > 0)[0]
    eye = np.zeros((len(active), flat.size))
    eye[np.arange(len(active)), active] = 1.0
    cols = np.asarray(jax.jit(jax.vmap(
        lambda v: ravel_pytree(projector(unravel(v)))[0]))(jnp.asarray(eye)))
    # one column per support pattern, then orthonormalize
    _, keep = np.unique(np.abs(cols) > 1e-14, axis=0, return_index=True)
    return np.linalg.qr(cols[np.sort(keep)].T)[0], unravel


def _jacobian_operator(f, basis, chunk=32):
    """Callable ``z -> Q^T (dF/dz) Q`` by forward-mode products.

    The batched JVP is compiled once and reused at every ``z``; compiling it
    is the dominant fixed cost of the dense Newton path.
    """
    jvp = jax.jit(lambda z, tangents: jax.vmap(
        lambda t: jax.jvp(f, (z,), (t,))[1])(tangents))
    n = basis.shape[1]

    def jacobian(z):
        cols = []
        for i in range(0, n, chunk):
            block = np.pad(basis[:, i:i + chunk],
                           ((0, 0), (0, max(0, i + chunk - n))))
            cols.append(np.asarray(jvp(z, jnp.asarray(block.T)))
                        [:min(chunk, n - i)])
        return basis.T @ np.concatenate(cols).T

    return jacobian


def _modes(vec, unravel, cfg, top=2):
    """Dominant ``(energy_fraction, m, n)`` of an eigenvector."""
    state = unravel(jnp.asarray(vec))
    table = im._static_tables(cfg.resolution)[0]
    m, n = np.asarray(table.m), np.asarray(table.n)
    energy = sum((np.asarray(getattr(state, name)) ** 2).sum(0)
                 for name in im._STATE_FIELDS)
    order = np.argsort(energy)[::-1][:top]
    return [(round(float(energy[k] / energy.sum()), 2), int(m[k]), int(n[k]))
            for k in order]


def count_unstable_modes(jac, basis, unravel, cfg, *, nev=4):
    """Negative-eigenvalue count and the ``nev`` most negative modes of ``jac``."""
    w, v = np.linalg.eig(jac)
    order = np.argsort(w.real)
    modes = [(float(w[k].real), _modes(basis @ v[:, k].real, unravel, cfg))
             for k in order[:nev]]
    return int((w.real < 0).sum()), modes


def solve_free_boundary_newton(
    inp: VmecInput,
    field: Any,
    *,
    start: Any = None,
    fixed_ftol: float = 3.0e-9,
    max_newton: int = 15,
    newton_tol: float = 1.0e-11,
    count_modes: bool = True,
    nev: int = 4,
) -> NewtonResult:
    """Free-boundary equilibrium of ``inp`` in ``field`` by Newton, with a stability count.

    Opt-in polish: the default descent solvers are unchanged.  Newton reaches
    ``|F| ~ 1e-13`` in 2-10x fewer iterations and converges current-carrying
    cases where descent drifts, at the price of 14-52 s of compile time and
    1.5-3x the memory.  It is fragile from a cold start or from a state
    converged only to 1e-4: it must start from a state converged to about
    1e-6 or tighter.  ``start=None`` (default) first runs the free-boundary
    descent solve of ``inp`` (so set the deck's ``ftol`` to 1e-6 or tighter)
    and anchors it by Krylov Newton.  ``start="fixed"`` starts instead from a
    fixed-boundary descent solve to ``fixed_ftol``, which converges
    current-carrying cases where free-boundary descent drifts.  A
    :class:`~vmex.core.wout.WoutData`, a wout path, or a
    :class:`~vmex.core.solver.SpectralState` on the deck's radial grid
    starts from that state (its constraint reference is taken from a
    fixed-boundary solve of ``inp``).

    ``inp`` is the deck (``lfreeb`` is forced on) and ``field`` the external
    field, an :class:`~vmex.core.mgrid.MgridField`.  Returns a
    :class:`NewtonResult`.  ``converged`` means ``residual < 1e-9``.
    ``count_modes=False`` skips the eigen-decomposition.
    """
    jax.config.update("jax_enable_x64", True)
    free = dataclasses.replace(inp, lfreeb=True)
    cfg = fbi.make_free_boundary_config(free, field)
    icfg = cfg.implicit
    params = im.params_from_input(free)
    if start is None:
        # Free-boundary descent to the deck's ftol, then the Krylov anchor.
        state, mask, rcon0, zcon0 = fbi._host_solve_and_mask_impl(
            cfg, jax.tree.map(np.asarray, params),
            jax.tree.map(np.asarray, field), error_on_no_convergence=False)
        state, mask, rcon0, zcon0 = jax.tree.map(
            jnp.asarray, (state, mask, rcon0, zcon0))
    else:
        fixed = im.run(dataclasses.replace(
            inp, lfreeb=False, mgrid_file="NONE", ftol_array=[fixed_ftol],
            niter_array=[40000]))
        rcon0, zcon0 = fixed.runtime.rcon0, fixed.runtime.zcon0
        ns = int(icfg.resolution.ns)
        if isinstance(start, str) and start == "fixed":
            state0 = fixed.state
        elif isinstance(start, (WoutData, str)) or hasattr(start, "__fspath__"):
            state0 = state_from_wout(start, inp=free, ns=ns)
        else:
            state0 = start
        rt = dataclasses.replace(
            im.runtime_from_params(params, icfg), rcon0=rcon0, zcon0=zcon0,
            lfreeb=True, jmax=ns,
            presf_ns_scale=jnp.asarray(fbi._presf_ns_scale(free, ns)))
        rt = dataclasses.replace(rt, bsqvac_edge=jax.lax.stop_gradient(
            cfg.vacuum_program.bsq(state0, rt, field)))
        mask = im._dof_mask(
            state0, rt, icfg,
            evaluator=lambda x: im.evaluate_forces(x, rt)[0], fixed_edge=False)
        mask = jax.tree.map(lambda a: jnp.asarray(np.asarray(a)), mask)
        state, _ = fbi._anchor_root(
            cfg, params, field, state0, mask, rcon0, zcon0)

    frozen = jax.lax.stop_gradient(state)
    projector = im._dof_projector(icfg, mask)
    basis, unravel = _basis(icfg, mask)
    residual = fbi._projected_residual(cfg, mask, formulation="raw")

    def force(z):
        return ravel_pytree(
            residual(unravel(z), params, field, frozen, rcon0, zcon0))[0]

    force_j = jax.jit(force)
    jacobian = _jacobian_operator(force, basis)
    z = ravel_pytree(projector(state))[0]
    steps = 0
    for _ in range(max_newton):  # damped dense Newton when the anchor stalls
        r0 = float(jnp.linalg.norm(force_j(z)))
        if r0 < newton_tol:
            break
        dz = jnp.asarray(basis @ -np.linalg.solve(
            jacobian(z), basis.T @ np.asarray(force_j(z))))
        alpha = 1.0
        while alpha > 1e-3 and float(
                jnp.linalg.norm(force_j(z + alpha * dz))) >= r0:
            alpha /= 2
        z = z + alpha * dz
        steps += 1
    state = jax.tree.map(jnp.add, frozen, projector(
        jax.tree.map(jnp.subtract, unravel(z), frozen)))
    r = float(jnp.linalg.norm(force_j(z)))
    n_unstable, modes = -1, []
    if count_modes:
        n_unstable, modes = count_unstable_modes(
            jacobian(z), basis, unravel, icfg, nev=nev)
    ok = bool(r < _CONVERGED)
    wout = wout_from_state(inp=free, state=state, fsqr=r, fsqz=0.0, fsql=0.0,
                           converged=ok)
    return NewtonResult(wout=wout, state=state, residual=r, converged=ok,
                        n_unstable=n_unstable, modes=modes, newton_steps=steps)
