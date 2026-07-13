"""Implicit-Jacobian backend for fixed-boundary least squares.

This module owns the exact fixed-point linearization, chunked Jacobian
assembly, block-Thomas factorization, Krylov recycling, and perturbation warm
starts.  The public scheduling and finite-difference driver remains in
:mod:`vmec_jax.core.optimize` and imports this backend only when requested.
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any, Callable, Sequence

import numpy as np

import jax
import jax.numpy as jnp
from solvax import (
    auto_chunk_size,
    block_thomas_factor,
    block_thomas_solve,
    chunk_map,
)

from .input import VmecInput
from .optimization_parameters import (
    _CURTOR_SCALE,
    _apply_current,
    _current_dof_setup,
    _dof_modes,
    _pack_current,
    pack_boundary,
    unpack_boundary,
)
from .optimize import solve_equilibrium
from .solver import SpectralState

__all__ = ["least_squares_implicit"]

# ---------------------------------------------------------------------------
# Implicit-gradient mode (vmec_jax.core.implicit wiring)
# ---------------------------------------------------------------------------


def _traceable_term(fun: Callable) -> Callable:
    """Objective callable -> traceable ``(state, runtime)`` function.

    Terms exposing ``residuals_state`` (:class:`QuasisymmetryRatioResidual`
    instances or their bound ``J``/``residuals`` methods) contribute their
    full traceable pointwise residual vector — same least-squares cost as
    the finite-difference stacked residuals (internal-grid sampling instead
    of the 63x64 wout grid), same Gauss-Newton geometry.
    Two-positional-argument callables (the scalar targets) are used as-is.
    Anything else (wout-table objectives — host NumPy) is rejected with a
    pointer to ``jac=None``.
    """
    owner = getattr(fun, "__self__", fun)
    if hasattr(owner, "residuals_state"):
        return owner.residuals_state
    try:
        params = [p for p in inspect.signature(fun).parameters.values()
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        two_positional = len(params) >= 2 and params[1].default is inspect.Parameter.empty
    except (TypeError, ValueError):
        two_positional = False
    if two_positional:
        return fun
    raise ValueError(
        f"objective term {fun!r} is not implicit-differentiable: jac='implicit' "
        "needs traceable (state, runtime) callables or a residuals_state method. "
        "Wout-engine terms (d_merc, l_grad_b, the Boozer QI residual) run on "
        "host NumPy — use jac=None (finite differences) for those.")


def least_squares_implicit(
    objective_terms: Sequence[tuple[Callable, float, float]],
    inp: VmecInput,
    *,
    max_mode: int,
    x0: np.ndarray | None,
    initial_state: SpectralState | None,
    current_dofs: int | None = None,
    jac_chunk_size: int | str | None = "auto",
    jac_solver: str = "block",
    recycle: bool = False,
    warm_start: str | None = "perturbation",
    solve_kwargs: dict,
    device: Any = None,
    verbose: int = 0,
    **scipy_kwargs,
):
    """Single-stage boundary least squares with implicit-gradient Jacobians.

    ``fun`` maps the dof vector through the traceable boundary update ->
    :func:`~vmec_jax.core.implicit.solve_implicit` (host solver behind
    ``pure_callback``, warm-started per ``warm_start`` — see
    :func:`least_squares`) ->
    :func:`~vmec_jax.core.implicit.runtime_from_params` -> the stacked
    objective rows: one warm host solve per trial ``x``.  ``jac`` computes
    the exact residual Jacobian by *forward* implicit differentiation:
    by default (``jac_solver="block"``, see ``jacobian_rows_block``) one
    amortized block-tridiagonal factorization backsolves every boundary-dof
    column at once; ``jac_solver="gmres"`` (see ``jacobian_rows``) runs one
    preconditioned GMRES per boundary dof, batched.  Either way this is far
    below one full equilibrium solve per dof (finite differences) while
    keeping the full pointwise Gauss-Newton residual geometry.  Both are
    jit-compiled once per stage.

    The residual and Jacobian graphs run on the device chosen by
    :func:`vmec_jax.core.device.resolve_implicit_device` — the CPU by default,
    where the per-dof vmapped adjoint GMRES is far faster than the
    launch-bound, dof-count-scaling GPU compile (plan.md R1); an explicit
    ``device=`` overrides this.  The forward equilibrium solve is a host
    callback and always runs on the CPU regardless.
    """
    import scipy.optimize

    from . import implicit as imp
    from .device import resolve_implicit_device

    if bool(inp.lasym):
        raise NotImplementedError(
            "jac='implicit' requires lasym = False (the implicit parameter map "
            "does not implement the lasym readin.f boundary rotation)")
    terms = [(_traceable_term(f), float(t), float(w)) for (f, t, w) in objective_terms]
    modes = _dof_modes(inp, max_mode)
    nm = len(modes)
    ntor = int(inp.ntor)
    row_idx = np.asarray([n + ntor for (_, n) in modes], dtype=int)
    col_idx = np.asarray([m for (m, _) in modes], dtype=int)
    # Optional AC/CURTOR dof block (spec 6.4): k + 1 trailing dofs, one-hot
    # tangents through ImplicitParams.ac / .curtor (runtime_from_params
    # already traces both).
    k_cur, ac_scale = _current_dof_setup(inp, current_dofs)
    ndof = 2 * nm + (k_cur + 1 if k_cur else 0)
    # multigrid=True routes the host solve through solve_multigrid (even for
    # single-stage ladders) so NITER-exhausted trials are penalized instead
    # of raising, matching the finite-difference path's trial policy.
    # Loose adjoint budget: the trust-region optimizer only needs ~1e-3
    # gradient accuracy; measured row-norm deviation vs the tight
    # (1e-11, 300) diagnostics default is <~1e-4 at a fraction of the cost.
    # hot_restart seeds each trial's host solve from the stage's previous
    # converged state (same fixed points, far fewer iterations) — the
    # implicit-mode analogue of the finite-difference path's hot restart.
    # warm_start="perturbation" (R25.4) sharpens that seed to the DESC-style
    # first-order prediction; see least_squares.
    if warm_start not in ("perturbation", "state", None):
        raise ValueError(
            "warm_start must be 'perturbation', 'state' or None, "
            f"got {warm_start!r}")
    if recycle and warm_start == "perturbation":
        warm_start = "state"  # the recycled variant carries (C, U) instead
    cfg = imp.make_config(inp, multigrid=True,
                          hot_restart=(warm_start is not None),
                          adjoint_tol=1e-6, adjoint_maxiter=30)
    if initial_state is not None:
        imp._HOT_CACHE[cfg] = initial_state
    # Pin the residual/Jacobian graphs to the fastest device for this launch-
    # bound path (CPU by default; explicit device= honored) — committing the
    # input dof vector to it makes both jits compile and run there, and their
    # uncommitted constants follow.  ``None`` leaves placement untouched.
    jac_device = resolve_implicit_device(device, cfg.resolution)

    def _place(x: np.ndarray) -> jnp.ndarray:
        a = jnp.asarray(x, dtype=jnp.float64)
        return a if jac_device is None else jax.device_put(a, jac_device)

    params0 = imp.params_from_input(inp)
    imp._template_runtime(cfg)  # host-built template: warm the per-cfg cache
    # eagerly so runtime_from_params stays traceable under jit below

    def params_of(x: jnp.ndarray):
        rbc = params0.rbc.at[row_idx, col_idx].set(x[:nm])
        zbs = params0.zbs.at[row_idx, col_idx].set(x[nm:2 * nm])
        params = dataclasses.replace(params0, rbc=rbc, zbs=zbs)
        if k_cur:
            ac = params0.ac.at[:k_cur].set(x[2 * nm:2 * nm + k_cur] * ac_scale)
            params = dataclasses.replace(
                params, ac=ac, curtor=x[2 * nm + k_cur] * _CURTOR_SCALE)
        return params

    def term_rows(state, rt) -> jnp.ndarray:
        return jnp.concatenate([
            jnp.atleast_1d(w * (jnp.asarray(f(state, rt)) - t)).ravel()
            for (f, t, w) in terms])

    def residual_rows(x: jnp.ndarray) -> jnp.ndarray:
        params = params_of(x)
        state = imp.solve_implicit(params, cfg)
        return term_rows(state, imp.runtime_from_params(params, cfg))

    rows_jit = jax.jit(residual_rows)

    # The evolved-dof mask is a *structural* per-config constant; fetch it
    # once (first host solve, cached in implicit._MASK_CACHE) so the Jacobian
    # graph below can close over it.
    if x0 is None:
        x0 = pack_boundary(inp, max_mode)
        if k_cur:
            x0 = np.concatenate([x0, _pack_current(inp, k_cur, ac_scale)])
    params0_np = jax.tree.map(lambda a: np.asarray(a, dtype=np.float64),
                              params_of(jnp.asarray(x0, dtype=jnp.float64)))
    _, mask_np = imp._host_solve_and_mask(cfg, params0_np)
    mask_const = jax.tree.map(jnp.asarray, mask_np)

    # One-hot dof tangents in ImplicitParams space, stacked over dofs
    # (leading axis ndof) so chunk_map can process them in fixed-size chunks:
    # boundary rbc/zbs rows first, then the scaled AC/CURTOR rows.
    t_rbc = np.zeros((ndof,) + np.shape(params0.rbc))
    t_zbs = np.zeros((ndof,) + np.shape(params0.zbs))
    t_ac = np.zeros((ndof,) + np.shape(params0.ac))
    t_curtor = np.zeros((ndof,))
    for j in range(nm):
        t_rbc[j, row_idx[j], col_idx[j]] = 1.0
        t_zbs[nm + j, row_idx[j], col_idx[j]] = 1.0
    for j in range(k_cur):
        t_ac[2 * nm + j, j] = ac_scale
    if k_cur:
        t_curtor[2 * nm + k_cur] = _CURTOR_SCALE
    zerop = jax.tree.map(jnp.zeros_like, params0)
    tangent_stack = (jnp.asarray(t_rbc), jnp.asarray(t_zbs),
                     jnp.asarray(t_ac), jnp.asarray(t_curtor))

    # R17.1 memory knob: chunk_size None == one wide vmap (current behavior),
    # an int / "auto" caps peak Jacobian memory at that many dofs at a time.
    if jac_chunk_size == "auto":
        chunk = int(auto_chunk_size(ndof))
    elif jac_chunk_size is None or isinstance(jac_chunk_size, int):
        chunk = jac_chunk_size
    else:
        raise ValueError(
            "jac_chunk_size must be None, a positive int, or 'auto', "
            f"got {jac_chunk_size!r}")

    def _jac_parts(x: jnp.ndarray):
        """Shared per-x setup of the implicit-Jacobian maps.

        At the fixed point, ``dz_j = -(dF/dz)^{-1} dF/dp t_j`` per boundary
        dof tangent ``t_j`` (F's linearization is plain JAX, so forward mode
        is available even though the solve itself is an opaque custom-VJP
        callback), then ``J[:, j] = G_z dz_j + G_p t_j`` with ``G`` the
        residual rows of the assembled state.  Returns the linearized
        operator ``Fz`` plus the per-dof tangent/RHS/column maps shared by
        all Jacobian variants below, and the ``(params, frozen, P, z_star)``
        linearization point (the block variant re-linearizes the *raw*
        residual formulation there).
        """
        params = params_of(x)
        frozen = jax.lax.stop_gradient(imp.solve_implicit(params, cfg))
        P = imp._dof_projector(cfg, mask_const)
        edge = imp._edge_mask(cfg)
        F = imp.residual_fn(cfg, frozen, mask_const)
        z_star = P(frozen)

        def G(z, prm):
            rt_p = imp.runtime_from_params(prm, cfg)
            return term_rows(imp._assemble(z, rt_p, frozen, P, edge), rt_p)

        def Fz(dz):
            return jax.jvp(lambda z: F(z, params), (z_star,), (dz,))[1]

        def tangent_of(tp):
            return dataclasses.replace(zerop, rbc=tp[0], zbs=tp[1],
                                       ac=tp[2], curtor=tp[3])

        def rhs_of(tp):
            b = jax.jvp(lambda prm: F(z_star, prm), (params,), (tp,))[1]
            return jax.tree.map(jnp.negative, b)

        def column_of(dz, tp):
            return jax.jvp(G, (z_star, params), (P(dz), tp))[1]

        return Fz, tangent_of, rhs_of, column_of, (params, frozen, P, z_star)

    def jacobian_rows(x: jnp.ndarray):
        """Exact residual Jacobian by *forward* implicit differentiation.

        One batched preconditioned GMRES per boundary dof (see
        ``_jac_parts``) — far below one forward solve per dof (finite
        differences) — while exposing the *full* pointwise Gauss-Newton
        geometry to scipy.  Columns are mathematically independent, so the
        result is identical across chunk sizes to float64 round-off.
        Also returns the per-dof state responses ``dz_j`` (leading axis
        ``2*nm``): they are the R25.4 perturbation warm-start linearization,
        already paid for by the column solves.
        """
        Fz, tangent_of, rhs_of, column_of, _ = _jac_parts(x)

        def column(tp_stack):
            tp = tangent_of(tp_stack)
            dz, _ = imp._adjoint_solve(Fz, rhs_of(tp), cfg)
            return column_of(dz, tp), dz

        cols, dz_cols = chunk_map(column, tangent_stack, chunk_size=chunk)
        return jnp.transpose(cols), dz_cols

    # R25.2 amortized block-tridiagonal variant.  The *raw* residual
    # formulation (un-preconditioned scalxc-scaled spectral force; see
    # implicit.residual_fn) has a Jacobian that is exactly block-tridiagonal
    # in the radial index (verified numerically: per-surface probe response
    # is 0.0 beyond |i-j| = 1 — the radial coupling is the nearest-neighbor
    # full/half-mesh FD stencil; the *preconditioned* formulation is dense
    # in radius because the 1D preconditioner applies per-mode radial
    # tridiagonal *solves*).  Both formulations share the fixed point, so
    #   dz_j = -(dF/dz)^{-1} dF/dp t_j
    # is the same solution through either: assemble the raw blocks once with
    # 3-colored jvp probes (cost ~3*(3*mn) residual linearizations,
    # independent of the dof count), factor once (solvax block Thomas), and
    # backsolve all 2*nm right-hand sides — then one short preconditioned
    # GMRES pass per column (warm-started at the direct solution) certifies
    # cfg.adjoint_tol in the same norm as the default path: solvax checks
    # the initial residual before the first Arnoldi cycle, so columns whose
    # direct solve already meets tolerance cost one matvec.
    mn_state = int(np.asarray(mask_np.R_cos).shape[1])
    ns_state = int(cfg.resolution.ns)
    active_fields = tuple(f for f in imp._STATE_FIELDS
                          if np.asarray(getattr(mask_np, f)).any())
    n_act = len(active_fields)
    m_block = n_act * mn_state
    # Probe (color, field, column) index triples, color-major so the probe
    # axis reshapes to (3, m_block, ...) below.
    probe_color = jnp.asarray(np.repeat(np.arange(3), m_block))
    probe_field = jnp.asarray(np.tile(np.repeat(np.arange(n_act), mn_state), 3))
    probe_col = jnp.asarray(np.tile(np.tile(np.arange(mn_state), n_act), 3))
    if jac_chunk_size == "auto":
        probe_chunk = int(auto_chunk_size(3 * m_block))
    else:
        probe_chunk = chunk

    def _pack(t) -> jnp.ndarray:
        """SpectralState -> (ns, m_block): active fields side by side."""
        return jnp.concatenate([getattr(t, f) for f in active_fields], axis=1)

    def _unpack(mat: jnp.ndarray) -> SpectralState:
        """(ns, m_block) -> SpectralState (structurally-zero fields zero)."""
        parts = dict(zip(active_fields, jnp.split(mat, n_act, axis=1)))
        return SpectralState(**{
            f: parts.get(f, jnp.zeros((ns_state, mn_state), mat.dtype))
            for f in imp._STATE_FIELDS})

    def jacobian_rows_block(x: jnp.ndarray):
        """``jacobian_rows`` via one block-tridiagonal factorization (R25.2).

        Same Jacobian as the default path to ``cfg.adjoint_tol`` (the GMRES
        corrector runs against the identical preconditioned system) at a
        cost that does not grow with the boundary-dof count.  Returns the
        certified per-dof responses ``dz_j`` alongside the rows (the R25.4
        perturbation warm-start linearization, same contract as
        ``jacobian_rows``).
        """
        Fz, tangent_of, rhs_of, column_of, (params, frozen, P, z_star) = \
            _jac_parts(x)
        F_raw = imp.residual_fn(cfg, frozen, mask_const, formulation="raw")

        def Fz_raw(dz):
            return jax.jvp(lambda z: F_raw(z, params), (z_star,), (dz,))[1]

        def probe_response(spec):
            c, fi, k = spec
            rows = (jnp.arange(ns_state) % 3 == c)
            mat = jnp.where(rows[:, None],
                            jax.nn.one_hot(k, mn_state, dtype=x.dtype)[None, :],
                            0.0)
            stack = (jax.nn.one_hot(fi, n_act, dtype=x.dtype)[:, None, None]
                     * mat[None])
            dz = _unpack(jnp.concatenate(
                [stack[i] for i in range(n_act)], axis=1))
            # F_raw projects both sides onto the evolved-dof subspace, so
            # its linearization is singular on the (I - P) complement; the
            # identity fill (dz - P(dz)) makes the assembled blocks
            # invertible without changing the solution for P-masked RHS.
            return _pack(jax.tree.map(lambda a, b, p: a + (b - p),
                                      Fz_raw(dz), dz, P(dz)))

        probes = chunk_map(probe_response,
                           (probe_color, probe_field, probe_col),
                           chunk_size=probe_chunk)
        # probes[c*m_block + q, i, :] = rows i of A(dz) for the color-c
        # one-hot-q tangent; for row i the unique in-stencil source surface
        # of color c is j = i + d with d = the offset satisfying
        # (i + d) % 3 == c, so gathering at color (i + d) % 3 reads off the
        # d-band blocks A[i, i+d] for every surface at once.
        probes = probes.reshape((3, m_block, ns_state, m_block))
        ii = jnp.arange(ns_state)

        def band(d):
            g = probes[(ii + d) % 3, :, ii, :]  # (ns, col q, row)
            return jnp.swapaxes(g, 1, 2)  # (ns, row, col)

        # lower[0] / upper[-1] gather out-of-stencil (zero) responses and
        # are ignored by the factorization anyway.
        factors = block_thomas_factor(band(-1), band(0), band(1))

        def raw_rhs(tp_stack):
            tp = tangent_of(tp_stack)
            b = jax.jvp(lambda prm: F_raw(z_star, prm), (params,), (tp,))[1]
            return _pack(jax.tree.map(jnp.negative, b))

        rhs = chunk_map(raw_rhs, tangent_stack, chunk_size=chunk)
        dz0 = block_thomas_solve(factors, jnp.moveaxis(rhs, 0, -1))

        def column(args):
            *tp_stack_j, dz0_mat = args
            tp = tangent_of(tuple(tp_stack_j))
            dz, _ = imp._adjoint_solve(
                Fz, rhs_of(tp), cfg, x0=_unpack(dz0_mat),
                max_restarts=min(3, cfg.adjoint_maxiter))
            return column_of(dz, tp), dz

        cols, dz_cols = chunk_map(
            column, (*tangent_stack, jnp.moveaxis(dz0, -1, 0)),
            chunk_size=chunk)
        return jnp.transpose(cols), dz_cols

    # R25.3 recycled variant: all 2*nm solves share the operator Fz (and Fz
    # drifts slowly between accepted trust-region iterates), so a GCROT
    # deflation pair (C, U) is threaded through a lax.scan over fixed-size
    # dof chunks — vmapped *within* a chunk with the incoming pair shared
    # read-only, then advanced from one representative (first) lane — and
    # returned to the caller, which stashes it between jac_jit calls.  The
    # dof axis is zero-padded to a whole number of chunks; padded columns
    # have zero RHS (gcrot converges in zero cycles) and are discarded.
    n_flat = sum(int(np.prod(s.shape))
                 for s in jax.tree.leaves(imp._state_struct(cfg)))
    csize = int(chunk) if chunk else ndof
    nchunks = -(-ndof // csize)
    pad = nchunks * csize - ndof

    def jacobian_rows_recycled(x: jnp.ndarray, C: jnp.ndarray,
                               U: jnp.ndarray):
        """``jacobian_rows`` with GCROT recycle carry (plan R25.3).

        Same ``cfg.adjoint_tol`` / ``cfg.adjoint_maxiter`` budget per solve
        as the default path; the Jacobian matches to solver tolerance *when
        the solves converge within budget*.  See the ``recycle`` note in
        :func:`least_squares` for why this is opt-in: the solvax v0.1
        recycle space measurably slows warm-started columns on the
        production operator, so budget-capped columns can come back with
        larger residuals than the GMRES path.
        """
        Fz, tangent_of, rhs_of, column_of, _ = _jac_parts(x)

        def column(tp_stack_j, rec):
            tp = tangent_of(tp_stack_j)
            dz, sol = imp._recycled_solve(Fz, rhs_of(tp), cfg, rec)
            return column_of(dz, tp), sol.recycle

        def scan_body(carry, tp_chunk):
            cols_chunk, recs = jax.vmap(
                column, in_axes=(0, None))(tp_chunk, carry)
            # Lane 0 is always a real dof (pad < csize): its updated pair
            # seeds the next chunk / the next Jacobian evaluation.
            return jax.tree.map(lambda a: a[0], recs), cols_chunk

        def pad_stack(t):
            t = jnp.concatenate(
                [t, jnp.zeros((pad,) + t.shape[1:], t.dtype)])
            return t.reshape((nchunks, csize) + t.shape[1:])

        (C, U), cols = jax.lax.scan(
            scan_body, (C, U),
            tuple(pad_stack(t) for t in tangent_stack))
        cols = cols.reshape((nchunks * csize,) + cols.shape[2:])[:ndof]
        return jnp.transpose(cols), C, U

    if jac_solver not in ("block", "gmres"):
        raise ValueError(
            f"jac_solver must be 'block' or 'gmres', got {jac_solver!r}")
    if recycle:
        jac_impl = jacobian_rows_recycled  # opt-in R25.3 experiment wins
    elif jac_solver == "block":
        jac_impl = jacobian_rows_block
    else:
        jac_impl = jacobian_rows
    jac_jit = jax.jit(jac_impl)

    holder: dict[str, Any] = {"nres": None, "lin": None}
    if recycle:
        # An all-zero pair is a cold start (gcrot's warm-start QR masks the
        # rank-deficient columns out); shapes are static so jac_jit compiles
        # once and the carried pair never triggers a re-trace.
        holder["recycle"] = (_place(np.zeros((n_flat, imp._RECYCLE_K))),
                             _place(np.zeros((n_flat, imp._RECYCLE_K))))

    # R25.4 perturbation warm start (DESC arXiv:2203.15927 ``eq.perturb``
    # before ``eq.solve``): each jac(x_ref) call stashes its linearization —
    # the converged state plus the per-dof responses dz_j its columns just
    # solved — and every subsequent trial fun(x) deposits the first-order
    # predicted state in implicit._PERTURB_SEED for the host solve to
    # consume, instead of restarting from the unmoved last converged state.
    P_seed = imp._dof_projector(cfg, mask_const)
    edge_seed = imp._edge_mask(cfg)

    @jax.jit
    def predicted_state(x_trial, x_ref, frozen, dz_cols):
        """First-order trial-state prediction around the stashed jac point.

        ``x_pred = frozen + P(sum_j (x_trial - x_ref)_j dz_j) +
        edge*(boundary(p_trial) - frozen)`` through the same dof-projector /
        assemble machinery the implicit residual uses, so the edge row lands
        exactly on the trial boundary (the solver's ``hot_restart_state``
        boundary shift becomes a no-op) and frozen directions stay frozen.
        """
        rt_p = imp.runtime_from_params(params_of(x_trial), cfg)
        dz = jax.tree.map(
            lambda d: jnp.tensordot(x_trial - x_ref, d, axes=1), dz_cols)
        z = jax.tree.map(jnp.add, P_seed(frozen), dz)
        return imp._assemble(z, rt_p, frozen, P_seed, edge_seed)

    def _stash_linearization(x: np.ndarray, dz_cols) -> None:
        """Record ``(x_ref, converged state, dz columns)`` for trial seeding."""
        hit = imp._LAST_SOLVE.get(cfg)
        params_np = jax.tree.map(lambda a: np.asarray(a, dtype=np.float64),
                                 params_of(jnp.asarray(x, dtype=jnp.float64)))
        if hit is not None and hit[0] == imp._params_key(params_np):
            holder["lin"] = (np.array(x, dtype=float), hit[1].state, dz_cols)
        else:  # unexpected call pattern: better no seed than a wrong one
            holder["lin"] = None

    def fun(x: np.ndarray) -> np.ndarray:
        lin = holder["lin"]
        if lin is not None and lin[0].shape == np.shape(x):
            seed = jax.tree.map(
                lambda a: np.asarray(a, dtype=np.float64),
                jax.device_get(predicted_state(
                    _place(x), _place(lin[0]), lin[1], lin[2])))
            if all(np.all(np.isfinite(a)) for a in jax.tree.leaves(seed)):
                imp._PERTURB_SEED[cfg] = seed
        try:
            residual = np.asarray(
                jax.device_get(rows_jit(_place(x))), dtype=float)
        except Exception as exc:  # zero-crash policy: penalize, don't die
            if holder["nres"] is None:
                raise
            if verbose:
                print(f"[least_squares] trial solve failed: {exc}")
            return np.full((holder["nres"],), 1.0e6)
        finally:
            imp._PERTURB_SEED.pop(cfg, None)  # one-shot: never leak a seed
        if not np.all(np.isfinite(residual)):
            residual = np.where(np.isfinite(residual), residual, 1.0e6)
        holder["nres"] = residual.size
        if verbose:
            print(f"[least_squares] cost = {0.5 * float(residual @ residual):.6e}")
        return residual

    def jac_fn(x: np.ndarray) -> np.ndarray:
        if recycle:
            rows, C, U = jac_jit(_place(x), *holder["recycle"])
            holder["recycle"] = (C, U)  # deflate the next jac evaluation
            return np.asarray(jax.device_get(rows), dtype=float)
        rows, dz_cols = jac_jit(_place(x))
        if warm_start == "perturbation":
            _stash_linearization(np.asarray(x, dtype=float), dz_cols)
        return np.asarray(jax.device_get(rows), dtype=float)

    result = scipy.optimize.least_squares(fun, np.asarray(x0, dtype=float),
                                          jac=jac_fn, **scipy_kwargs)
    result.input = unpack_boundary(inp, result.x[:2 * nm], max_mode)
    if k_cur:
        result.input = _apply_current(result.input, result.x[2 * nm:],
                                      k_cur, ac_scale)
    stats = imp._SOLVE_STATS.get(cfg)
    result.solve_stats = None if stats is None else dict(stats)
    try:
        # Hot-seed the diagnostic re-solve from the stage's last converged
        # trial state (plan R25.1): the optimizer's final x was just solved
        # by the implicit path, so this converges in ~1 sweep instead of
        # repeating a full cold solve per continuation stage.
        seed = imp._HOT_CACHE.get(cfg)
        try:
            result.equilibrium = solve_equilibrium(
                result.input, initial_state=seed, **solve_kwargs)
        except Exception:
            if seed is None:
                raise
            # ns-mismatched seed (different ladder) must not cost the
            # diagnostic: fall back to the plain cold solve.
            result.equilibrium = solve_equilibrium(result.input, **solve_kwargs)
    except Exception:  # pragma: no cover - diagnostic attribute only
        result.equilibrium = None
    return result
