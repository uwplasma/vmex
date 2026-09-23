"""Implicit derivatives of a coupled free-boundary VMEX equilibrium.

The forward pass uses the ordinary host-driven free-boundary solver.  The
reverse pass differentiates the converged plasma--vacuum root: NESTOR is
re-evaluated on the current edge and its vacuum pressure enters the evolved
VMEC edge-force rows.  Solver iterations are therefore absent from the AD
tape; one matrix-free adjoint supplies derivatives with respect to plasma
profiles and explicit external-field parameters (including ESSOS coil shape
and current degrees of freedom or an :class:`~vmex.core.mgrid.MgridField`
current vector).

The forward pass also Newton-anchors the returned state on that coupled
root (:func:`_anchor_root`), because VMEC's ``ftol`` test does not put it
there, and a trial whose restart from the configuration's reference cannot
converge falls back to a cold solve after a bounded budget
(:func:`_restart_budget`).
"""

from __future__ import annotations

import dataclasses
import functools
import warnings
from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from scipy.sparse import bsr_matrix
from scipy.sparse.linalg import LinearOperator, gcrotmk
from solvax import SpluFactorization, block_thomas_factor, block_thomas_solve
from solvax import gmres as _solvax_gmres

from . import implicit as im
from .device import AUTO, resolve_implicit_device
from .freeboundary import (
    _edge_fourier_jax,
    _presf_ns_scale,
    _presf_ns_scale_traceable,
    _solve_free_boundary_stage,
    _vacuum_executables,
    _vacuum_scalars,
    free_boundary_resolution,
)
from .errors import VmecConvergenceError, VmecError
from .input import VmecInput
from .solver import SpectralState, evaluate_forces

Array = Any

#: ``coupled_gcrot`` is the certified default; ``boundary_schur`` eliminates
#: the radial bulk and assembles the edge system column by column;
#: ``edge_response`` iterates the coupled transpose on a dense model of
#: NESTOR.
_ADJOINT_SOLVERS = ("boundary_schur", "coupled_gcrot", "edge_response")


@dataclass(frozen=True, eq=False)
class FreeBoundaryImplicitConfig:
    """Static controls for :func:`solve_free_boundary_implicit`.

    ``field_from_parameters`` reconstructs the differentiable field from the
    second solve argument. ``implicit`` holds the shared Krylov tolerances and
    differentiable input-to-runtime map.
    """

    implicit: im.ImplicitConfig
    field_from_parameters: Callable[[Any], Any]
    adjoint_solver: str = "coupled_gcrot"
    adjoint_fail: str = "error"
    schur_probe_chunk_size: int = 1
    vacuum_program: Any = None

    @property
    def resolution(self):
        """Solve resolution of the shared implicit configuration (static)."""
        return self.implicit.resolution


def make_free_boundary_config(
    inp: VmecInput,
    external_field: Any,
    *,
    ns: int | None = None,
    ftol: float | None = None,
    max_iterations: int | None = None,
    adjoint_tol: float = 1e-10,
    adjoint_maxiter: int = 300,
    adjoint_gcrot_m: int = 30,
    adjoint_gcrot_k: int = 5,
    adjoint_solver: str = "coupled_gcrot",
    adjoint_fail: str = "error",
    schur_probe_chunk_size: int = 1,
    field_from_parameters: Callable[[Any], Any] | None = None,
    device: Any = AUTO,
    max_fsq_ratio: float = 1.0,
    refine_tol: float = 1.0e-10,
) -> FreeBoundaryImplicitConfig:
    """Build a coupled free-boundary derivative configuration.

    By default the second solve argument is an external-field pytree. For a
    smaller AD graph, pass ``field_from_parameters`` and then supply only the
    actual current/coil parameters to :func:`solve_free_boundary_implicit`.
    ``external_field`` here is the concrete reference used to fix resolution.
    ``device="auto"`` uses the CPU for the coupled implicit response on an
    accelerator host unless the process already pins JAX placement; pass an
    explicit device to override that measured lower-memory default.
    ``adjoint_solver="coupled_gcrot"`` is the certified default;
    ``"boundary_schur"`` selects the advanced radial-elimination path, which
    stays well conditioned on marginally converged roots where the coupled
    Krylov solve stalls.  ``"edge_response"`` is the coupled Krylov solve
    with NESTOR's response to the edge built once as a dense matrix instead
    of re-swept in reverse mode on every iteration: the matvec then costs no
    NESTOR call, the answer is certified on the exact coupled transpose, and
    a certificate that misses continues on that exact operator, so the lane
    is never less accurate than the default -- only cheaper.
    ``adjoint_fail="best_effort"`` returns the stalled
    Krylov solution with a warning instead of raising, so one bad trial in an
    optimization is a poor search direction the line search rejects rather
    than a dead run; a non-finite adjoint always raises.

    ``max_fsq_ratio`` is the largest ``(fsqr + fsqz + fsql) / ftol`` at which
    :func:`solve_free_boundary_implicit_status` still certifies a solve that
    ran out of iterations.  It governs only those: a solve VMEC calls
    converged is certified whatever the ratio, and because that flag is per
    component (``fsqr``, ``fsqz`` and ``fsql`` each under ``ftol``) a converged
    solve routinely carries a summed ratio near 1.5 and up to 3.  The default
    of 1 therefore certifies the converged solves and nothing else, where the
    previous 1e6 also admitted roots that stopped 49x past ``ftol``; an
    uncertified state is a failed trial (status 2), because its adjoint is
    taken off the root and its value depends on the path there.

    ``refine_tol`` is the residual at which a certifiable solve counts as
    anchored on the coupled plasma--vacuum root: VMEC's ``ftol`` does not
    test that root (see :func:`_anchor_root`), so every such solve is
    Newton-refined until ``|F| <= refine_tol`` and differentiated there.  A
    solve the anchor cannot land is status 3.  ``inf`` skips the anchor and
    restores the unanchored behaviour.
    """
    if not inp.lfreeb:
        raise ValueError("free-boundary implicit differentiation requires LFREEB=T")
    resolution = free_boundary_resolution(inp, external_field, ns=ns)
    solve_device = resolve_implicit_device(device, resolution)
    cfg = im.make_config(
        inp, ns=resolution.ns, ftol=ftol, max_iterations=max_iterations,
        adjoint_tol=adjoint_tol, adjoint_maxiter=adjoint_maxiter,
        adjoint_gcrot_m=adjoint_gcrot_m, adjoint_gcrot_k=adjoint_gcrot_k,
        device=solve_device, max_fsq_ratio=max_fsq_ratio,
        refine_tol=refine_tol,
    )
    if cfg.resolution != resolution:
        cfg = dataclasses.replace(cfg, resolution=resolution)
    if adjoint_solver not in _ADJOINT_SOLVERS:
        raise ValueError(
            "adjoint_solver must be one of " + ", ".join(
                repr(name) for name in sorted(_ADJOINT_SOLVERS)))
    if adjoint_fail not in {"error", "best_effort"}:
        raise ValueError("adjoint_fail must be 'error' or 'best_effort'")
    if schur_probe_chunk_size < 1:
        raise ValueError("schur_probe_chunk_size must be positive")
    config = FreeBoundaryImplicitConfig(
        implicit=cfg,
        field_from_parameters=(lambda value: value) if field_from_parameters is None
        else field_from_parameters,
        adjoint_solver=adjoint_solver,
        adjoint_fail=adjoint_fail,
        schur_probe_chunk_size=int(schur_probe_chunk_size),
    )
    return dataclasses.replace(config, vacuum_program=_vacuum_program(config))


def _vacuum_program(cfg: FreeBoundaryImplicitConfig):
    """Return the cached differentiable NESTOR program for ``cfg``."""
    icfg = cfg.implicit
    rt = im._template_runtime(icfg)
    # Some free-boundary decks intentionally leave the axis guess blank. The
    # executable only needs a non-degenerate static topology here; its actual
    # axis coordinates remain dynamic inputs to every NESTOR call.
    r00 = float(np.asarray(icfg.inp.rbc)[int(icfg.inp.ntor), 0])
    axis_r = im._device_pin(
        icfg, jnp.full((icfg.resolution.nzeta,), r00))
    axis_z = im._device_pin(icfg, jnp.zeros_like(axis_r))
    return _vacuum_executables(
        icfg.resolution, mf=int(icfg.inp.mpol) + 1,
        nf=int(icfg.inp.ntor), signgs=int(rt.setup.signgs),
        wint=np.asarray(rt.trig.wint), modes=rt.modes,
        axis_r0=axis_r, axis_z0=axis_z, use_fft=False,
        solve_on_plasma_device=True,
    )[1]


def _vacuum_inputs(state: SpectralState, rt) -> tuple:
    """NESTOR's own plasma inputs ``(edge coefficients, ctor, axis R, axis Z)``.

    Everything the plasma sends to the vacuum solver passes through this
    short tuple, and reaching it from the state costs no NESTOR work.
    """
    ctor, _, axis_r, axis_z, _, _ = _vacuum_scalars(state, rt)
    return (*_edge_fourier_jax(state, rt), ctor, axis_r, axis_z)


# Module scope with ``cfg`` the only static key: every per-gradient array is
# an argument, so one executable serves a whole optimization and no trial's
# arrays are baked into a cached trace.
@functools.partial(jax.jit, static_argnames=("cfg",))
def _edge_response(cfg: FreeBoundaryImplicitConfig, params, field_parameters,
                   frozen, rcon0, zcon0):
    """Dense NESTOR edge response ``(value, dbsqvac/dh, h)``, built once.

    ``bsqvac`` depends on the plasma only through :func:`_vacuum_inputs`, so
    one forward-mode column per edge coefficient, one for ``ctor`` and two
    per toroidal axis point capture NESTOR's whole linearization exactly.
    The plasma-side ``dh/dz`` that follows is VMEC-only and costs no vacuum
    solve, so every later coupled matvec becomes a dense multiply instead of
    a NESTOR reverse sweep.
    """
    icfg = cfg.implicit
    field = cfg.field_from_parameters(field_parameters)
    rt = dataclasses.replace(
        im.runtime_from_params(params, icfg), rcon0=rcon0, zcon0=zcon0,
        lfreeb=True, jmax=int(icfg.resolution.ns),
        presf_ns_scale=_presf_ns_scale_traceable(
            params, icfg.inp, int(icfg.resolution.ns)),
    )
    flat, unflatten = ravel_pytree(_vacuum_inputs(frozen, rt))

    def kernel(vector):
        return cfg.vacuum_program.bsq_edge(*unflatten(vector), field)

    value, jacobian = jax.vmap(
        lambda tangent: jax.jvp(kernel, (flat,), (tangent,)),
        out_axes=(None, -1),
    )(jnp.eye(flat.size, dtype=flat.dtype))
    return jax.lax.stop_gradient((value, jacobian, flat))


def _linearized_bsqvac(state, rt, response):
    """NESTOR's edge pressure through its dense response at the root."""
    value, jacobian, base = response
    delta = ravel_pytree(_vacuum_inputs(state, rt))[0] - base
    return value + jnp.tensordot(jacobian, delta, axes=1)


def _projected_residual(
    cfg: FreeBoundaryImplicitConfig,
    dof_mask: SpectralState,
    *,
    formulation: str = "preconditioned",
    fixed_bsqvac: Array | None = None,
    response: tuple | None = None,
) -> Callable:
    """Return a projected coupled root in preconditioned or raw form.

    ``fixed_bsqvac`` freezes only NESTOR's edge pressure.  The resulting raw
    Jacobian is exactly block tridiagonal in radius and is the bulk operator
    used by the boundary-Schur adjoint.  ``response`` instead replaces the
    NESTOR call by :func:`_edge_response`'s dense model of it, which is exact
    in value and first derivative at the root the response was built on and
    is therefore an exact linearization there, at the cost of a multiply.

    The returned closure is memoized on ``(cfg, formulation, mask content)``
    (``fixed_bsqvac=None`` only): the host callback hands each backward pass
    a fresh numpy mask tree of identical content, and a stable closure
    identity lets :func:`_prepare_transpose` reuse its compiled
    linearization across gradient calls.  A ``fixed_bsqvac`` closure changes
    value every iterate and is never a jit key, so it is not cached; the
    pressure still enters the lane as a traced argument, never a baked
    constant.
    """
    if formulation not in {"preconditioned", "raw"}:
        raise ValueError(f"unknown formulation {formulation!r}")

    def residual(z, params, field_parameters, frozen, rcon0, zcon0):
        return _projected_residual_lane(
            z, params, field_parameters, frozen, rcon0, zcon0, dof_mask,
            fixed_bsqvac, response, cfg=cfg, formulation=formulation)

    leaves = jax.tree.leaves(dof_mask)
    if fixed_bsqvac is not None or response is not None or any(
            isinstance(leaf, jax.core.Tracer) for leaf in leaves):
        # Under an outer jax.jit the mask is a tracer, so it has no bytes to
        # key on; the whole pullback is staged once anyway, which is what the
        # cache buys the eager path.
        return residual
    key = (cfg, formulation, tuple(
        np.asarray(leaf).tobytes() for leaf in leaves))
    cached = _RESIDUAL_CLOSURE_CACHE.get(key)
    if cached is None:
        _RESIDUAL_CLOSURE_CACHE[key] = cached = residual
        while len(_RESIDUAL_CLOSURE_CACHE) > _RESIDUAL_CLOSURE_CACHE_MAX:
            _RESIDUAL_CLOSURE_CACHE.pop(next(iter(_RESIDUAL_CLOSURE_CACHE)))
    return cached


_RESIDUAL_CLOSURE_CACHE: dict[tuple, Callable] = {}
_RESIDUAL_CLOSURE_CACHE_MAX = 8



# Module scope with ``cfg``/``formulation`` static and the per-iterate arrays
# (mask included) as traced arguments, the ``implicit.
# _preconditioned_residual_lane`` idiom: the previous per-call ``@jax.jit``
# closure inside :func:`_projected_residual` was a fresh function object, so
# every backward pass of a free-boundary optimization re-traced and
# recompiled the coupled NESTOR+VMEC residual up to four times (both
# formulations plus the Schur frozen root) before its first adjoint matvec.
@functools.partial(jax.jit, static_argnames=("cfg", "formulation"))
def _projected_residual_lane(z, params, field_parameters, frozen, rcon0,
                             zcon0, dof_mask, fixed_bsqvac, response=None, *,
                             cfg: FreeBoundaryImplicitConfig,
                             formulation: str):
    icfg = cfg.implicit
    project = im._dof_projector(icfg, dof_mask)
    # Unlike fixed boundary, every active edge coefficient comes from z;
    # the input boundary is only the forward solver's initial guess.
    dz = project(jax.tree.map(lambda a, b: a - b, z, frozen))
    state = jax.tree.map(jnp.add, frozen, dz)
    rt = dataclasses.replace(
        im.runtime_from_params(params, icfg), rcon0=rcon0, zcon0=zcon0,
        lfreeb=True, jmax=int(icfg.resolution.ns),
        presf_ns_scale=_presf_ns_scale_traceable(
            params, icfg.inp, int(icfg.resolution.ns)),
    )
    if response is not None:
        bsqvac = _linearized_bsqvac(state, rt, response)
    elif fixed_bsqvac is None:
        # The executable/topology was fixed concretely when the config was
        # built; all equilibrium and coil values stay dynamic traced arrays.
        external_field = cfg.field_from_parameters(field_parameters)
        bsqvac = cfg.vacuum_program.bsq(state, rt, external_field)
    else:
        bsqvac = fixed_bsqvac
    rt = dataclasses.replace(rt, bsqvac_edge=bsqvac)
    if formulation == "preconditioned":
        force, _, _ = evaluate_forces(state, rt)
    else:
        force = im._raw_force_state(state, rt, include_edge=True)
    return project(force)


_FREE_MASK_CACHE: dict[tuple, SpectralState] = {}

#: One reference stage per configuration; every solve restarts from it.  It is
#: kept whether or not it converged: a configuration with no reachable root
#: must still answer deterministically, and :func:`_host_solve_and_mask_status`
#: is what reports the miss.
_FREE_HOT_CACHE: dict[FreeBoundaryImplicitConfig, Any] = {}
_FREE_LAST_RESULT: dict[FreeBoundaryImplicitConfig, Any] = {}
#: What :func:`_anchor_root` reported for the last solve (``None``: skipped).
_FREE_LAST_ANCHOR: dict[FreeBoundaryImplicitConfig, Any] = {}


def _mask_key(cfg: FreeBoundaryImplicitConfig) -> tuple:
    icfg = cfg.implicit
    return (icfg.resolution, bool(icfg.lconm1), int(icfg.inp.ncurr), "free")


def _host_solve_and_mask(
    cfg, params_np, field_parameters_np, *, error_on_no_convergence=True,
):
    """Run the callback on the implicit config's explicitly selected device."""
    with im._device_context(cfg.implicit):
        return _host_solve_and_mask_impl(
            cfg, params_np, field_parameters_np,
            error_on_no_convergence=error_on_no_convergence,
        )


def _cold_reference(solve, icfg, inp, field):
    """Solve cold, falling back to a coarse rung when one rung cannot converge.

    A single fine rung started from the input guess can stall above ``ftol``
    at any iteration budget, where a ``[ns // 2, ns]`` ladder converges in a
    few hundred iterations: at ns = 31 on the finite-beta free-boundary deck
    fsq is 9.3e-8 after 2500 iterations and 9.3e-7 after 12000, against
    ftol = 1e-9, while the ladder reaches 1.4e-9 in 106; on the ns = 8 lasym
    deck at ftol = 1e-8 the single rung stops at fsq 4.9e-7 after 2500 where
    the ``[4, 8]`` ladder converges in 163.

    The ladder is not free -- it compiles and solves a second resolution -- so
    a deck whose single rung already converges never pays for it.
    """
    stage = solve(initial_state=None)
    ns = int(icfg.resolution.ns)
    if bool(stage.result.converged) or ns < 8:
        return stage
    from .multigrid import solve_free_boundary_multigrid

    return solve(initial_state=solve_free_boundary_multigrid(
        inp, ns_array=np.array([max(3, ns // 2), ns]),
        ftol_array=np.full(2, icfg.ftol),
        niter_array=np.full(2, icfg.max_iterations), external_field=field,
        raise_on_max_iterations=False).state)


def _continuation(stage) -> dict:
    """Restart arguments that continue ``stage`` instead of repeating turn-on.

    Every solve of a configuration restarts from the same reference, never
    from the previous trial, so the returned state is a function of the
    parameters alone.  The constraint and residual continuation travel with
    the state, as they do between multigrid rungs; the vacuum continuation is
    a starting guess for this trial's field, which differs from the
    reference's, so its caches are rebuilt rather than reused.
    """
    return dict(
        initial_state=stage.continuation_state, vacuum_continuation=stage.vacuum,
        constraint_continuation=(stage.rcon0, stage.zcon0),
        residual_continuation=(
            stage.result.fsqr, stage.result.fsqz, stage.result.fsql))


def _host_solve_and_mask_impl(
    cfg, params_np, field_parameters_np, *, error_on_no_convergence=True,
):
    """Opaque forward solve plus one structural free-boundary dof mask."""
    icfg = cfg.implicit
    params = im._device_pin(icfg, jax.tree.map(jnp.asarray, params_np))
    field_parameters = im._device_pin(
        icfg, jax.tree.map(jnp.asarray, field_parameters_np))
    field = cfg.field_from_parameters(field_parameters)
    inp = im.input_with_params(icfg.inp, params)
    solve = functools.partial(
        _solve_free_boundary_stage, inp, external_field=field,
        resolution=icfg.resolution, ftol=icfg.ftol,
        max_iterations=icfg.max_iterations,
        error_on_no_convergence=error_on_no_convergence, use_fft=False)
    reference = _FREE_HOT_CACHE.get(cfg)
    if reference is None:
        reference = _FREE_HOT_CACHE[cfg] = _cold_reference(solve, icfg, inp, field)
    try:
        stage = solve(**_continuation(reference),
                      max_iterations=_restart_budget(icfg, reference))
    except VmecError:
        # The reference may be too far from this trial; start over cold.
        stage = _cold_reference(solve, icfg, inp, field)
    else:
        if not bool(stage.result.converged):
            # A restart carries the reference's state, and far enough from it
            # that state is a worse start than none: measured, a 2 % change in
            # every coil current stalls at the iteration cap and lands 9.2e-3
            # away from the cold answer, which converges in 76. Solve this
            # trial cold instead -- after at most ``_restart_budget``
            # iterations, not the full ``max_iterations``. The stored
            # reference is deliberately NOT replaced, so every call stays a
            # function of its own parameters and the one reference, making
            # repeated calls bit-identical.
            stage = _cold_reference(solve, icfg, inp, field)
    _FREE_LAST_RESULT[cfg] = stage.result
    state = stage.result.state
    rcon0, zcon0 = stage.rcon0, stage.zcon0

    # Prime the static runtime/NESTOR closures before a transformed residual
    # sees them, then identify only structurally active state entries.
    rt = im.runtime_from_params(params, icfg)
    key = _mask_key(cfg)
    mask = _FREE_MASK_CACHE.get(key)
    if mask is None:
        # The active mode families are fixed by VMEC symmetry/constraints,
        # not by the dense NESTOR response. Freeze the converged edge pressure
        # while finding structural force support; tracing NESTOR here would
        # compile its LU pullback once merely to rediscover the same mask.
        rt_mask = dataclasses.replace(
            rt, rcon0=rcon0, zcon0=zcon0, lfreeb=True,
            jmax=int(icfg.resolution.ns),
            bsqvac_edge=jax.lax.stop_gradient(stage.vacuum.bsqvac),
            # Host value on purpose: this runtime only discovers which dofs
            # have structural force support, a discrete answer that no
            # derivative is taken through.  The adjoint lanes above use the
            # traceable ratio instead.
            presf_ns_scale=jnp.asarray(
                _presf_ns_scale(inp, int(icfg.resolution.ns))
            ),
        )
        force = lambda x: evaluate_forces(x, rt_mask)[0]  # noqa: E731
        mask = im._dof_mask(
            state, rt_mask, icfg, evaluator=force, fixed_edge=False
        )
        _FREE_MASK_CACHE[key] = mask

    # Anchor a certifiable state on the coupled root the adjoint linearizes,
    # so the value, the objective's cotangent and the linearization all read
    # the same point (see _anchor_root).  An uncertifiable solve is never
    # differentiated, so it is not worth the Newton work.
    report = None
    result = stage.result
    fsq = float(result.fsqr) + float(result.fsqz) + float(result.fsql)
    if bool(result.converged) or fsq / icfg.ftol <= icfg.max_fsq_ratio:
        with im._timed(icfg, "anchor"):
            state, report = _anchor_root(
                cfg, params, field_parameters, state, mask, rcon0, zcon0)
        if error_on_no_convergence and not report.certified:
            raise VmecConvergenceError(
                "the free-boundary solve met ftol but Newton did not anchor "
                f"it on the coupled root: |F| {report.residual:.3e} > "
                f"refine_tol {icfg.refine_tol:.1e} after {report.steps} "
                "steps",
                hint="tighten ftol, or pass refine_tol=inf to "
                     "make_free_boundary_config to skip the anchor")
    _FREE_LAST_ANCHOR[cfg] = report

    to_numpy = lambda tree: jax.tree.map(  # noqa: E731
        lambda value: np.asarray(value, dtype=np.float64), tree
    )
    return to_numpy(state), to_numpy(mask), to_numpy(rcon0), to_numpy(zcon0)


def _restart_budget(icfg, reference) -> int:
    """Iterations a restart from the reference may spend before going cold.

    The restart exists to be cheaper than a cold solve, so it gets the
    iterations the configuration's own cold reference needed, never less
    than a tenth of ``max_iterations`` and never more than all of it.  On the
    finite-beta single-stage deck (reference: 943 iterations) restarts that
    converge do so in 329 or 1305 where the cold solve takes 823-1425, and
    from halfway to the optimum every restart ran to the 4000 cap without
    converging -- 19-21 s of each 25 s trial -- before the cold solve that
    certified it.  A budget spent is deterministic: the restart is a fixed
    function of the parameters and the one reference, so the same parameters
    always take the same branch.
    """
    cap = int(icfg.max_iterations)
    iterations = getattr(reference.result, "iterations", None)
    if iterations is None:
        return cap
    return int(min(cap, max(int(iterations), cap // 10, 1)))


def _host_solve_and_mask_status(cfg, params_np, field_parameters_np):
    """Exception-free free-boundary callback for optimizer trial points."""
    try:
        state, mask, rcon0, zcon0 = _host_solve_and_mask(
            cfg, params_np, field_parameters_np, error_on_no_convergence=False,
        )
    except VmecError:
        icfg = cfg.implicit
        reference = _FREE_HOT_CACHE.get(cfg)
        runtime = im._template_runtime(icfg)
        state = (im._initial_state(runtime.setup) if reference is None
                 else reference.continuation_state)
        mask = _FREE_MASK_CACHE.get(_mask_key(cfg))
        if mask is None:
            mask = jax.tree.map(jnp.zeros_like, state)
        to_numpy = lambda tree: jax.tree.map(  # noqa: E731
            lambda value: np.asarray(value, dtype=np.float64), tree
        )
        return (to_numpy(state), to_numpy(mask), to_numpy(runtime.rcon0),
                to_numpy(runtime.zcon0), np.int32(1), np.float64(np.inf),
                np.float64(np.inf))

    result = _FREE_LAST_RESULT[cfg]
    fsq = float(result.fsqr) + float(result.fsqz) + float(result.fsql)
    ratio = fsq / cfg.implicit.ftol
    status = 0 if bool(result.converged) or ratio <= cfg.implicit.max_fsq_ratio else 2
    report = _FREE_LAST_ANCHOR.get(cfg)
    if status == 0 and report is not None and not report.certified:
        status = 3
    return state, mask, rcon0, zcon0, np.int32(status), np.float64(fsq), np.float64(ratio)


def _baseline_struct(cfg: FreeBoundaryImplicitConfig):
    rt = im._template_runtime(cfg.implicit)
    return jax.tree.map(
        lambda value: jax.ShapeDtypeStruct(value.shape, jnp.float64), rt.rcon0
    ), jax.tree.map(
        lambda value: jax.ShapeDtypeStruct(value.shape, jnp.float64), rt.zcon0
    )


def _callback(params, field_parameters, cfg):
    rcon_struct, zcon_struct = _baseline_struct(cfg)
    return jax.pure_callback(
        functools.partial(_host_solve_and_mask, cfg),
        (im._state_struct(cfg.implicit), im._state_struct(cfg.implicit),
         rcon_struct, zcon_struct),
        params, field_parameters,
        sharding=im._callback_sharding(cfg.implicit),
    )


def _callback_status(params, field_parameters, cfg):
    """Return the free-boundary state, linearization data, and solve status."""
    rcon_struct, zcon_struct = _baseline_struct(cfg)
    scalar = jax.ShapeDtypeStruct((), jnp.float64)
    return jax.pure_callback(
        functools.partial(_host_solve_and_mask_status, cfg),
        (im._state_struct(cfg.implicit), im._state_struct(cfg.implicit),
         rcon_struct, zcon_struct, jax.ShapeDtypeStruct((), jnp.int32),
         scalar, scalar),
        params, field_parameters,
        sharding=im._callback_sharding(cfg.implicit),
    )


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def solve_free_boundary_implicit(
    params: im.ImplicitParams,
    field_parameters: Any,
    cfg: FreeBoundaryImplicitConfig,
) -> SpectralState:
    """Return a differentiable converged free-boundary spectral state.

    Solves the coupled plasma--vacuum root: the VMEC force balance in the
    interior together with NESTOR's vacuum pressure on the moving edge, for
    the boundary and profile parameters in ``params`` and the coil or
    current parameters in ``field_parameters``.  The forward pass is the
    ordinary host free-boundary solver behind a ``jax.pure_callback``, so
    the solver's own iterations never enter the AD tape; the reverse pass
    is one matrix-free adjoint of the converged root.

    Parameters
    ----------
    params:
        Differentiable equilibrium parameters, an
        :class:`~vmex.core.implicit.ImplicitParams` pytree: the dense INDATA
        boundary arrays ``rbc``/``rbs``/``zbc``/``zbs`` in metres, the
        profile coefficient arrays ``am`` (pressure, Pa before
        ``pres_scale``), ``ai`` (rotational transform, dimensionless) and
        ``ac`` (current), the optimizable current-spline knot values
        ``ac_aux_f``, and the scalars ``phiedge`` (total enclosed toroidal
        flux, Wb), ``pres_scale`` and ``curtor`` (total toroidal current,
        A).  The boundary here is only the forward solver's initial guess:
        in a free-boundary solve every active edge coefficient is an
        unknown of the root.
    field_parameters:
        Second differentiable argument, passed through
        ``cfg.field_from_parameters`` to build the external field.  With the
        default identity map this *is* the external-field pytree — an
        :class:`~vmex.core.mgrid.MgridField` (differentiable in its
        ``extcur`` currents, A) or a coil field closing over its own dofs.
        Pass a ``field_from_parameters`` to
        :func:`make_free_boundary_config` instead and this becomes just the
        coil shape and current degrees of freedom, which keeps the AD graph
        small.
    cfg:
        The static :class:`FreeBoundaryImplicitConfig` from
        :func:`make_free_boundary_config`.  It is a non-differentiable
        argument of the custom VJP and a jit key, so it must be a stable
        object: build it once and reuse it across the optimization.  It
        fixes the resolution, the forward tolerances, the adjoint solver,
        and the compiled NESTOR program.

    Returns
    -------
    The converged :class:`~vmex.core.solver.SpectralState` — the spectral
    coefficient arrays of the equilibrium, differentiable with respect to
    both ``params`` and ``field_parameters``.

    A forward solve that does not converge raises
    :class:`~vmex.core.errors.VmecError` (retried once from a cold start
    when a hot-restart seed was in play).  That makes this entry point
    unsuitable for an optimizer that probes infeasible trial points; use
    :func:`solve_free_boundary_implicit_status`, which reports failure as a
    status value instead of raising and suppresses the pullback for a trial
    whose derivatives are not certified.
    """
    icfg = cfg.implicit
    with im._device_context(icfg):
        params, field_parameters = im._device_pin(
            icfg, (params, field_parameters))
        state, _, _, _ = _callback(params, field_parameters, cfg)
    return state


def _solve_fwd(params, field_parameters, cfg):
    icfg = cfg.implicit
    with im._device_context(icfg):
        params, field_parameters = im._device_pin(
            icfg, (params, field_parameters))
        state, mask, rcon0, zcon0 = _callback(
            params, field_parameters, cfg)
        state, mask, rcon0, zcon0 = im._device_pin(
            icfg, (state, mask, rcon0, zcon0))
    return state, (params, field_parameters, state, mask, rcon0, zcon0)


def _solve_bwd(cfg, saved, state_bar):
    icfg = cfg.implicit
    with im._device_context(icfg):
        saved, state_bar = im._device_pin(icfg, (saved, state_bar))
        return _solve_bwd_impl(cfg, saved, state_bar)


def _solve_bwd_impl(cfg, saved, state_bar):
    params, field_parameters, state, mask, rcon0, zcon0 = saved
    frozen = jax.lax.stop_gradient(state)
    project = im._dof_projector(cfg.implicit, mask)
    residual = _projected_residual(cfg, mask)
    z_star = project(state)

    rhs = project(state_bar)
    traced = any(
        isinstance(value, jax.core.Tracer) for value in jax.tree.leaves(rhs)
    )
    if traced:
        # An outer jax.jit needs a staged Krylov loop. Ordinary SciPy/JAXopt
        # drivers call the concrete lane below, which compiles only one
        # transpose matvec and has a much smaller cold memory peak.
        if cfg.adjoint_solver == "boundary_schur":
            warnings.warn(
                "adjoint_solver='boundary_schur' is a host lane and is not "
                "available under jax.jit; this pullback uses the staged "
                "coupled GCROT solve instead. That is a different solver, so "
                "the gradient agrees only to the Krylov tolerance, and it is "
                "the slower of the two -- measured on the free-boundary "
                "single-stage deck at ns = 25, one warm value-and-gradient "
                "costs 5.1 s through the Schur lane against 38.6 s staged. "
                "Call the objective eagerly to keep the solver you asked for.",
                RuntimeWarning, stacklevel=2)
        _, state_pullback = jax.vjp(
            lambda z: residual(
                z, params, field_parameters, frozen, rcon0, zcon0), z_star
        )
        lam, _ = im._adjoint_solve_gcrot(
            lambda cotangent: state_pullback(cotangent)[0], rhs, cfg.implicit)
    elif cfg.adjoint_solver == "boundary_schur":
        lam = _host_boundary_schur_adjoint(
            cfg, z_star, params, field_parameters, frozen, rcon0, zcon0,
            mask, rhs, fail=cfg.adjoint_fail,
        )
        residual = _projected_residual(cfg, mask, formulation="raw")
    elif cfg.adjoint_solver == "edge_response":
        response = _edge_response(
            cfg, params, field_parameters, frozen, rcon0, zcon0)
        lam = _host_adjoint(
            residual, z_star, params, field_parameters, frozen, rcon0, zcon0,
            rhs, cfg.implicit, fail=cfg.adjoint_fail,
            pullback=_prepare_response_transpose(
                z_star, params, field_parameters, frozen, rcon0, zcon0, mask,
                response, cfg=cfg),
            certify=True)
    else:
        lam = _host_adjoint(
            residual, z_star, params, field_parameters, frozen, rcon0, zcon0,
            rhs, cfg.implicit, fail=cfg.adjoint_fail)

    _, parameter_pullback = jax.vjp(
        lambda p, field: residual(
            z_star, p, field, frozen, rcon0, zcon0),
        params, field_parameters,
    )
    params_bar, field_bar = parameter_pullback(
        jax.tree.map(jnp.negative, lam)
    )
    return params_bar, field_bar


def _packers(cfg: FreeBoundaryImplicitConfig, dof_mask):
    """``(project, pack, unpack)`` for one mask, built at trace time.

    Called only from inside the jitted helpers below, so the arrays these
    close over are that trace's own arguments and never a cached constant.
    """
    icfg = cfg.implicit
    fields = im._active_state_fields(icfg)
    ns, mn = int(icfg.resolution.ns), int(dof_mask.R_cos.shape[1])
    project = im._dof_projector(icfg, dof_mask)

    def pack(tree):
        return jnp.concatenate(
            [getattr(tree, name) for name in fields], axis=1)

    def unpack(matrix):
        parts = dict(zip(fields, jnp.split(matrix, len(fields), axis=1)))
        return SpectralState(**{
            name: parts.get(name, jnp.zeros((ns, mn), matrix.dtype))
            for name in im._STATE_FIELDS})

    return project, pack, unpack


# Every helper below is at module scope with ``cfg`` its only static key, and
# takes each per-gradient array -- the bulk blocks, the mask, the saved
# pullback -- as an argument. A jitted closure defined inside the adjoint
# instead is a fresh jit key on every backward pass, which both recompiles the
# lane and strands that trial's arrays in JAX's trace cache as jaxpr
# constants; the boundary-Schur lane used to leak a whole block system
# (ns x block x block) per successful gradient that way.
@functools.partial(jax.jit, static_argnames=("cfg", "probe_chunk_size"))
def _frozen_bulk_blocks(params, field_parameters, frozen, rcon0, zcon0,
                        dof_mask, z_star, bsqvac, *, cfg, probe_chunk_size=None):
    """Radial block tridiagonal of the raw Jacobian at frozen edge pressure.

    ``probe_chunk_size`` overrides ``cfg.schur_probe_chunk_size``; the Newton
    anchor passes the fixed-boundary refinement's ``ceil(sqrt(block))``.
    """
    icfg = cfg.implicit
    runtime = dataclasses.replace(
        im.runtime_from_params(params, icfg), rcon0=rcon0, zcon0=zcon0,
        lfreeb=True, jmax=int(icfg.resolution.ns),
        presf_ns_scale=_presf_ns_scale_traceable(
            params, icfg.inp, int(icfg.resolution.ns)),
        bsqvac_edge=bsqvac,
    )

    def frozen_root(z, payload):
        return _projected_residual_lane(
            z, payload[0], payload[1], frozen, rcon0, zcon0, dof_mask,
            bsqvac, None, cfg=cfg, formulation="raw")

    system = im._raw_block_system(
        (params, field_parameters), icfg, frozen, dof_mask,
        im._active_state_fields(icfg),
        probe_chunk_size=(cfg.schur_probe_chunk_size if probe_chunk_size is None
                          else probe_chunk_size), residual=frozen_root,
        z_star=z_star, runtime=runtime, physical_state=frozen,
        include_edge=True, factor=False,
    )
    return (system.lower, system.diagonal, system.upper,
            system.row_scale, system.column_scale)


@functools.partial(jax.jit, static_argnames=("cfg", "chunk"))
def _edge_probe_columns(edge_values, pullback, lower, diagonal, upper,
                        dof_mask, edge_basis, *, cfg, chunk):
    """``(J^T - A^T)`` applied to a batch of edge directions, packed.

    ``pullback`` is a saved transpose: the exact coupled one assembles the
    true Schur complement, a response-linearized one a preconditioner for it.
    """
    project, pack, unpack = _packers(cfg, dof_mask)
    ns, block_size = diagonal.shape[0], diagonal.shape[1]

    def band(tangent):
        return project(unpack(im.block_tridiag_matvec(
            lower, diagonal, upper, pack(project(tangent)))))

    zero = unpack(jnp.zeros((ns, block_size), edge_basis.dtype))
    band_t = jax.vjp(band, zero)[1]

    def column(edge_value):
        matrix = jnp.zeros((ns, block_size), edge_basis.dtype)
        cotangent = project(unpack(matrix.at[-1].set(edge_basis @ edge_value)))
        return pack(project(jax.tree.map(
            jnp.subtract, pullback(cotangent)[0], band_t(cotangent)[0])))

    return im.chunk_map(
        column, edge_values,
        chunk_size=min(chunk, int(edge_values.shape[0])))


@functools.partial(jax.jit, static_argnames=("cfg",))
def _edge_rows_of_tree(tree, dof_mask, edge_basis, *, cfg):
    """Edge-basis coordinates of a state tree."""
    project, pack, _ = _packers(cfg, dof_mask)
    return edge_basis.T @ pack(project(tree))[-1]


@functools.partial(jax.jit, static_argnames=("cfg",))
def _edge_rows(packed, dof_mask, edge_basis, *, cfg):
    """Edge-basis coordinates of a batch of packed radial rows."""
    project, pack, unpack = _packers(cfg, dof_mask)
    return jax.vmap(
        lambda matrix: edge_basis.T @ pack(project(unpack(matrix)))[-1])(packed)


@functools.partial(jax.jit, static_argnames=("cfg",))
def _pack_projected(tree, dof_mask, *, cfg):
    """Pack a projected state into radial rows."""
    project, pack, _ = _packers(cfg, dof_mask)
    return pack(project(tree))


@functools.partial(jax.jit, static_argnames=("cfg",))
def _pack_state(tree, dof_mask, *, cfg):
    """Pack a state into radial rows without projecting it."""
    _, pack, _ = _packers(cfg, dof_mask)
    return pack(tree)


@jax.jit
def _apply_pullback(pullback, cotangent):
    """Apply a saved transpose to one state cotangent."""
    return pullback(cotangent)[0]


@functools.partial(jax.jit, static_argnames=("cfg",))
def _unpack_projected(matrix, dof_mask, *, cfg):
    """Unpack radial rows into a projected state."""
    project, _, unpack = _packers(cfg, dof_mask)
    return project(unpack(matrix))


def _edge_basis(cfg: FreeBoundaryImplicitConfig, dof_mask, packed_mask, dtype):
    """Orthonormal columns spanning the active evolved edge directions."""
    icfg = cfg.implicit
    active_fields = im._active_state_fields(icfg)
    mn = int(dof_mask.R_cos.shape[1])
    paired, columns = {}, []
    if bool(icfg.lconm1) and int(icfg.resolution.ntor) > 0:
        positive, negative = im._m1_pair_columns(icfg)
        for pos, neg in zip(positive, negative):
            paired[("Z_sin", int(pos))] = (int(neg), 1.0)
            if bool(icfg.resolution.lasym):
                paired[("Z_cos", int(pos))] = (int(neg), -1.0)
    for field_index, name in enumerate(active_fields):
        for mode in range(mn):
            index = field_index * mn + mode
            if packed_mask[index] == 0.0:
                continue
            pair = paired.get((name, mode))
            if any(name == pair_name and mode == pair_value[0]
                   for (pair_name, _), pair_value in paired.items()):
                continue
            column = np.zeros_like(packed_mask)
            if pair is None:
                column[index] = 1.0
            else:
                other, sign = pair
                column[index] = 1.0 / np.sqrt(2.0)
                column[field_index * mn + other] = sign / np.sqrt(2.0)
            columns.append(column)
    return jnp.asarray(np.stack(columns, axis=1), dtype=dtype)


def _balanced_dense_solver(schur):
    """Two-sided balanced dense solve of one small edge Schur matrix."""
    tiny = np.finfo(schur.dtype).tiny
    row_scale = 1.0 / np.maximum(np.max(np.abs(schur), axis=1), tiny)
    row_scaled = row_scale[:, None] * schur
    column_scale = 1.0 / np.maximum(np.max(np.abs(row_scaled), axis=0), tiny)
    balanced = row_scaled * column_scale[None, :]
    condition = np.linalg.cond(balanced)

    def solve_reduced(value):
        scaled_rhs = row_scale * value
        if np.isfinite(condition) and condition < 1.0 / np.finfo(
                schur.dtype).eps:
            balanced_solution = np.linalg.solve(balanced, scaled_rhs)
        else:
            # The edge system can inherit redundant m=1 directions. A
            # rank-revealing solve avoids amplifying them; the exact
            # coupled-residual certificate below remains authoritative.
            balanced_solution = np.linalg.lstsq(
                balanced, scaled_rhs,
                rcond=np.finfo(schur.dtype).eps * max(schur.shape))[0]
        return column_scale * balanced_solution

    return solve_reduced, condition


def _host_boundary_schur_adjoint(
    cfg, z_star, params, field_parameters, frozen, rcon0, zcon0, mask, rhs,
    *, fail="error",
):
    """Solve the coupled adjoint through an exact edge Schur complement.

    With NESTOR's converged edge pressure frozen, the raw VMEC Jacobian ``A``
    is block tridiagonal in radius.  The difference ``E = J - A`` has nonzero
    rows only at the free boundary.  Eliminating the bulk gives

    ``(I + U.T @ A.T^-1 @ E.T @ U) mu = U.T @ A.T^-1 @ rhs``,

    where ``U`` injects the evolved edge row. Direct three-surface assembly
    already retains every terminal VMEC stencil coupling in ``A``; only
    NESTOR's response to the moving edge remains in ``E``. One sparse bulk
    factorization and the edge solve recover the full adjoint. ``E``'s
    columns come from NESTOR's dense edge response (:func:`_edge_response`),
    its exact linearization at this root, so each costs a VMEC transpose
    instead of a NESTOR reverse sweep: on the finite-beta single-stage deck
    the warm adjoint falls from 3.1 s to 1.6-1.9 s and its compile from 40 s
    to 31 s. The final answer is certified against the original coupled
    transpose operator.
    """
    icfg = cfg.implicit
    field = cfg.field_from_parameters(field_parameters)
    rt = dataclasses.replace(
        im.runtime_from_params(params, icfg), rcon0=rcon0, zcon0=zcon0,
        lfreeb=True, jmax=int(icfg.resolution.ns),
        presf_ns_scale=_presf_ns_scale_traceable(
            params, icfg.inp, int(icfg.resolution.ns)),
    )
    bsqvac = jax.lax.stop_gradient(cfg.vacuum_program.bsq(frozen, rt, field))
    lower_blocks, diagonal, upper_blocks, row_scale, column_scale = (
        _frozen_bulk_blocks(params, field_parameters, frozen, rcon0, zcon0,
                            mask, z_star, bsqvac, cfg=cfg,
                            probe_chunk_size=_bulk_probe_chunk(cfg, mask)))
    coupled_residual = _projected_residual(cfg, mask, formulation="raw")
    coupled_pullback = _prepare_transpose(
        z_star, params, field_parameters, frozen, rcon0, zcon0,
        residual=coupled_residual)
    # The Schur columns come from NESTOR's dense edge response, which is its
    # exact linearization at this root: a VMEC transpose per column instead
    # of a NESTOR reverse sweep.  The answer is still certified on the exact
    # coupled transpose below.
    edge_pullback = _prepare_response_transpose(
        z_star, params, field_parameters, frozen, rcon0, zcon0, mask,
        _edge_response(cfg, params, field_parameters, frozen, rcon0, zcon0),
        cfg=cfg, formulation="raw")

    dtype = rhs.R_cos.dtype
    packed_mask = np.asarray(_pack_state(mask, mask, cfg=cfg)[-1])
    edge_basis = _edge_basis(cfg, mask, packed_mask, dtype)
    nedge = int(edge_basis.shape[1])
    chunk = int(cfg.schur_probe_chunk_size)

    # The raw radial system is strongly scaled near the magnetic axis. A
    # globally pivoted sparse LU is materially more accurate there than the
    # no-pivot block-Thomas elimination, while retaining O(ns) block storage.
    ns, block_size = np.asarray(diagonal).shape[:2]
    bulk_row_scale = np.asarray(row_scale)
    bulk_column_scale = np.asarray(column_scale)
    previous = np.maximum(np.arange(ns) - 1, 0)
    following = np.minimum(np.arange(ns) + 1, ns - 1)
    lower = (bulk_row_scale[:, :, None] * np.asarray(lower_blocks)
             * bulk_column_scale[previous, None, :])
    middle = (bulk_row_scale[:, :, None] * np.asarray(diagonal)
              * bulk_column_scale[:, None, :])
    upper = (bulk_row_scale[:, :, None] * np.asarray(upper_blocks)
             * bulk_column_scale[following, None, :])
    blocks, indices, indptr = [], [], [0]
    for radial_row in range(ns):
        if radial_row:
            blocks.append(lower[radial_row]); indices.append(radial_row - 1)
        blocks.append(middle[radial_row]); indices.append(radial_row)
        if radial_row + 1 < ns:
            blocks.append(upper[radial_row]); indices.append(radial_row + 1)
        indptr.append(len(blocks))
    sparse_bulk = bsr_matrix(
        (np.asarray(blocks), np.asarray(indices), np.asarray(indptr)),
        shape=(ns * block_size, ns * block_size)).tocsc()
    bulk_lu = SpluFactorization(sparse_bulk)

    def sparse_inverse_packed(packed, *, transpose):
        values = np.asarray(packed)
        batched = values.ndim == 3
        values = values if batched else values[None]
        scale_rhs = bulk_column_scale if transpose else bulk_row_scale
        scale_solution = bulk_row_scale if transpose else bulk_column_scale
        flat_rhs = np.moveaxis(values * scale_rhs[None], 0, -1).reshape(
            ns * block_size, -1)
        flat_solution = np.asarray(bulk_lu.solve(
            flat_rhs, trans="T" if transpose else "N"))
        solution = np.moveaxis(
            flat_solution.reshape(ns, block_size, -1), -1, 0)
        solution = solution * scale_solution[None]
        return solution if batched else solution[0]

    def sparse_inverse(tree, *, transpose):
        packed = sparse_inverse_packed(
            np.asarray(_pack_projected(tree, mask, cfg=cfg)),
            transpose=transpose)
        return _unpack_projected(jnp.asarray(packed), mask, cfg=cfg)

    def probe(edge_values, pullback):
        """Schur columns for a batch of edge directions, exact or modelled."""
        packed = _edge_probe_columns(
            jnp.asarray(edge_values, dtype), pullback, lower_blocks, diagonal,
            upper_blocks, mask, edge_basis, cfg=cfg, chunk=chunk)
        solved = sparse_inverse_packed(np.asarray(packed), transpose=True)
        return np.asarray(edge_values) + np.asarray(
            _edge_rows(jnp.asarray(solved), mask, edge_basis, cfg=cfg))

    base = sparse_inverse(rhs, transpose=True)
    edge_rhs = np.asarray(_edge_rows_of_tree(base, mask, edge_basis, cfg=cfg))
    calls = 0

    def apply(value):
        nonlocal calls
        calls += 1
        return probe(np.atleast_2d(value), edge_pullback)[0]

    apply(edge_rhs)

    identity = np.eye(nedge, dtype=np.asarray(edge_rhs).dtype)
    schur = probe(identity, edge_pullback).T
    calls += nedge
    solve_reduced, condition = _balanced_dense_solver(schur)
    edge_solution = solve_reduced(edge_rhs)
    # Dense iterative refinement is cheap at edge size and recovers the
    # residual digits lost to the raw near-axis scaling.
    for _ in range(3):
        edge_solution += solve_reduced(edge_rhs - schur @ edge_solution)
    if im._adjoint_debug_enabled():
        print(f"[vmex adjoint] balanced Schur condition={condition:.3e}")

    correction_rows = _edge_probe_columns(
        jnp.asarray(np.atleast_2d(edge_solution), dtype), edge_pullback,
        lower_blocks, diagonal, upper_blocks, mask, edge_basis, cfg=cfg,
        chunk=chunk)[0]
    correction = _unpack_projected(
        jnp.asarray(sparse_inverse_packed(
            np.asarray(correction_rows), transpose=True)), mask, cfg=cfg)
    solution = jax.tree.map(jnp.subtract, base, correction)
    defect = jax.tree.map(
        jnp.subtract, rhs, _apply_pullback(coupled_pullback, solution))
    residual_norm = float(im._tree_norm(defect))
    rhs_norm = float(im._tree_norm(rhs))
    tolerance = float(im._adjoint_acceptance(icfg, rhs_norm))
    if not np.isfinite(residual_norm) or residual_norm > tolerance:
        # The raw near-axis scaling can leave the reduced solve a few ulps
        # outside the strict certificate. Continue from it with the original
        # coupled operator; this changes no mathematics and usually needs only
        # a small correction rather than a cold whole-state Krylov search.
        im._count(icfg, adjoint_certificate_fallbacks=1)
        return _host_adjoint(
            coupled_residual, z_star, params, field_parameters, frozen, rcon0,
            zcon0, rhs, icfg, x0=solution, fail=fail)
    return solution


# ---------------------------------------------------------------------------
# Newton anchoring of the coupled root
# ---------------------------------------------------------------------------
#: Newton steps one attempt may take.  From solves at ``ftol = 1e-10`` along
#: the finite-beta single-stage optimization path it lands in 7 to 15, from
#: ``ftol = 1e-9`` in 7 to 23; the budget leaves room for a slower point
#: without letting a non-contracting one run on.
_ANCHOR_MAX_STEPS = 25

#: Inexact-Newton forcing term of each step's Krylov solve, as in the
#: fixed-boundary refinement (``implicit._REFINE_FORCING``).  A forcing that
#: loosens to 1e-2 while the contraction is slow (Eisenstat--Walker style)
#: was measured: it cut the Krylov work but failed to anchor five of eight
#: points on that path that this fixed one lands.
_ANCHOR_FORCING = 1.0e-6

#: GMRES iterations per solve (one unrestarted cycle).  The preconditioner
#: is exact where it was built, so the Krylov solve only covers its
#: staleness: 1-3 iterations there and at most about 70 on the single-stage
#: deck; the bulk factors alone needed 40-110.
_ANCHOR_KRYLOV = 150

#: A Newton solve that took more Krylov iterations than this rebuilds the
#: preconditioner at the next iterate.  A rebuild costs about 1 s on the
#: single-stage deck; 20 and 40 cost the same there (Krylov 183 against 310).
_ANCHOR_REFACTOR_KRYLOV = 40

#: Relative accuracy of the simplified correction behind the damping test;
#: the test compares two lengths, so two digits suffice.
_ANCHOR_TEST_FORCING = 1.0e-2

#: Damping of the first step, one per attempt: an attempt that fails starts
#: again from the host state with a smaller first step.  On a finite-difference
#: leg of ``take_free_boundary_gradients.py`` a first step of 0.5 passed the
#: monotonicity test and still led where the next Newton corrections grew
#: from 0.11 to 1.8; a quarter step from the same state lands.
_ANCHOR_FIRST_DAMPING = (1.0, 0.25, 0.0625)

#: Add NESTOR's coupling to the bulk preconditioner (:func:`_anchor_coupling`);
#: off, the bulk factors alone precondition (4-8x the Krylov iterations).
_ANCHOR_WOODBURY = True

#: Smallest Newton damping factor before an attempt gives up.  Steps are
#: damped on Deuflhard's natural monotonicity test, not on ``|F|`` (see
#: :func:`_anchor_root`).
_ANCHOR_MIN_DAMPING = 1.0 / 64.0


@functools.partial(jax.jit, static_argnames=("cfg",))
def _anchor_raw_residual(z, params, field_parameters, frozen, rcon0, zcon0,
                         dof_mask, *, cfg):
    """Packed raw coupled residual (exact NESTOR) at ``z``."""
    _, pack, _ = _packers(cfg, dof_mask)
    return pack(_projected_residual_lane(
        z, params, field_parameters, frozen, rcon0, zcon0, dof_mask, None,
        None, cfg=cfg, formulation="raw"))


@functools.partial(jax.jit, static_argnames=("cfg",))
def _anchor_norm(z, params, field_parameters, frozen, rcon0, zcon0, dof_mask,
                 *, cfg):
    """``|F(z)|`` of the preconditioned coupled residual the adjoint uses."""
    return im._tree_norm(_projected_residual_lane(
        z, params, field_parameters, frozen, rcon0, zcon0, dof_mask, None,
        None, cfg=cfg, formulation="preconditioned"))


@jax.jit
def _anchor_factor(lower, diagonal, upper, row_scale, column_scale):
    """Block-Thomas factors of the scaled bulk block tridiagonal.

    The same two-sided scaling the Schur adjoint's sparse LU uses; here the
    factors only precondition, so the fixed-boundary refinement's JAX block
    elimination (``implicit._refine_block_factor_core``) is accurate enough
    and keeps each Krylov solve inside one compiled program.
    """
    ns = diagonal.shape[0]
    previous = jnp.maximum(jnp.arange(ns) - 1, 0)
    following = jnp.minimum(jnp.arange(ns) + 1, ns - 1)
    return block_thomas_factor(
        row_scale[:, :, None] * lower * column_scale[previous][:, None, :],
        row_scale[:, :, None] * diagonal * column_scale[:, None, :],
        row_scale[:, :, None] * upper * column_scale[following][:, None, :])


@functools.partial(jax.jit, static_argnames=("cfg",))
def _anchor_coupling(z, params, field_parameters, frozen, rcon0, zcon0,
                     dof_mask, response, factors, row_scale, column_scale, *,
                     cfg):
    """Woodbury terms that add NESTOR's coupling to the bulk factors.

    At frozen edge pressure the raw Jacobian is the block tridiagonal ``A``
    the factors invert.  NESTOR adds ``E = P Q``: ``Q = dh/dz`` maps the
    state to NESTOR's own inputs ``h`` (:func:`_vacuum_inputs`: edge
    coefficients, ``ctor``, the axis) and ``P`` is the edge force's response
    to them through :func:`_edge_response`.  Its rank is ``h``'s length (129
    on the single-stage deck), so ``(A + P Q)^-1`` is applied exactly by the
    Woodbury identity with one small capacitance factorization.  Returns
    ``(A^-1 P, Q, lu(I + Q A^-1 P))``, packed and flattened.
    """
    project, pack, unpack = _packers(cfg, dof_mask)
    icfg = cfg.implicit
    value, jacobian, _ = response
    shape = row_scale.shape
    rt = dataclasses.replace(
        im.runtime_from_params(params, icfg), rcon0=rcon0, zcon0=zcon0,
        lfreeb=True, jmax=int(icfg.resolution.ns),
        presf_ns_scale=_presf_ns_scale_traceable(
            params, icfg.inp, int(icfg.resolution.ns)),
    )
    _, force_of_pressure = jax.linearize(
        lambda bsq: pack(_projected_residual_lane(
            z, params, field_parameters, frozen, rcon0, zcon0, dof_mask, bsq,
            None, cfg=cfg, formulation="raw")), value)
    columns = jax.vmap(force_of_pressure, in_axes=-1, out_axes=-1)(jacobian)
    columns = columns.reshape(-1, columns.shape[-1])

    def inputs(packed):
        state = jax.tree.map(jnp.add, frozen, project(unpack(packed)))
        return ravel_pytree(_vacuum_inputs(state, rt))[0]

    _, inputs_t = jax.vjp(inputs, jnp.zeros(shape, row_scale.dtype))
    rows = jax.vmap(lambda e: inputs_t(e)[0])(
        jnp.eye(columns.shape[-1], dtype=columns.dtype))
    rows = rows.reshape(rows.shape[0], -1)

    def bulk_inverse(flat):
        scaled = flat.reshape(shape) * row_scale
        solved = block_thomas_solve(factors, scaled[..., None])[..., 0]
        return (solved * column_scale).ravel()

    solved = jax.vmap(bulk_inverse, in_axes=-1, out_axes=-1)(columns)
    capacitance = jnp.eye(rows.shape[0], dtype=rows.dtype) + rows @ solved
    return solved, rows, jax.scipy.linalg.lu_factor(capacitance)


@functools.partial(jax.jit, static_argnames=("cfg",))
def _anchor_linear_solve(force, z, params, field_parameters, frozen, rcon0,
                         zcon0, dof_mask, response, factors, row_scale,
                         column_scale, coupling, *, cfg, rtol):
    """``-J(z)^-1 force`` for the packed raw coupled residual, staged.

    GMRES on the raw linearization at ``z``, preconditioned by the bulk
    factors (possibly from an earlier iterate), with NESTOR through its
    dense edge response ``response`` -- a VMEC force tangent per matvec
    instead of a NESTOR sweep (9 ms against 45-70 ms on the single-stage
    deck) -- or exactly when ``response`` is ``None``.  One compiled program
    per ``(cfg, rtol)`` runs the whole solve, as the fixed-boundary
    refinement's ``_refine_block_step_core`` does; driving the same Krylov
    loop from the host cost 22 ms per iteration.  Inactive entries get an
    identity equation.  ``rtol`` is traced, so the Newton and the damping
    solves share one executable.  Returns the packed correction, the
    iteration count and the final relative residual.
    """
    project, pack, unpack = _packers(cfg, dof_mask)
    shape = force.shape
    _, linear = jax.linearize(
        lambda zz: _projected_residual_lane(
            zz, params, field_parameters, frozen, rcon0, zcon0, dof_mask,
            None, response, cfg=cfg, formulation="raw"), z)

    def operator(flat):
        packed = flat.reshape(shape)
        tangent = project(unpack(packed))
        return (pack(project(linear(tangent))) + packed
                - pack(tangent)).ravel()

    def precondition(flat):
        scaled = flat.reshape(shape) * row_scale
        solved = block_thomas_solve(factors, scaled[..., None])[..., 0]
        solved = (solved * column_scale).ravel()
        if coupling is None:
            return solved
        bulk_p, q, lu = coupling
        return solved - bulk_p @ jax.scipy.linalg.lu_solve(lu, q @ solved)

    rhs = -force.ravel()
    solution = _solvax_gmres(
        operator, rhs, precond=precondition,
        restart=min(_ANCHOR_KRYLOV, int(rhs.shape[0])), rtol=rtol, atol=0.0,
        max_restarts=1)
    return (solution.x.reshape(shape), solution.iterations,
            solution.residual_norm / jnp.linalg.norm(rhs))


@dataclass(frozen=True)
class _AnchorReport:
    """What the Newton anchor did to one host state."""

    certified: bool
    initial_residual: float
    residual: float
    steps: int = 0
    krylov_iterations: int = 0
    factorizations: int = 0
    displacement: float = 0.0
    attempts: int = 0


def _anchor_root(cfg, params, field_parameters, state, mask, rcon0, zcon0):
    """Newton-anchor a converged free-boundary state on its coupled root.

    VMEC's convergence test is not a root test for this residual.  ``ftol``
    bounds summed squares of the invariant forces, and the free-boundary
    edge row joins that sum only during the first 50 iterations after a
    (re)start (``residuals.edge_force_condition``; ``getfsq`` in
    VMEC2000's ``residue.f90``).  A small sum of squares does not mean a
    nearby root where ``dF/dz`` has a small singular value.  On the
    finite-beta single-stage deck a solve converged at ``ftol = 1e-10``
    leaves ``|F| ~ 3e-06`` and lies 3.5e-02 to 1.3e-01 from the root in
    coefficient norm, mostly lambda ``(m, n) = (2, 0)`` with low-``m`` R/Z --
    the pattern of a poloidal-angle relabelling, which only the weak
    spectral-condensation force pins.  The objective there differs from the
    objective at the root by 0.2-17 %, while the adjoint differentiates the
    root.

    The anchor is the free-boundary counterpart of
    ``implicit._refined_state``: Newton on the projected coupled residual.
    Each step solves the raw linearization by GMRES, with NESTOR entering
    through its dense edge response, rebuilt at every iterate.  The
    preconditioner is the exact Jacobian where it is built: the block
    factors of the radial block tridiagonal at frozen vacuum pressure (the
    Schur adjoint's bulk operator) plus NESTOR's low-rank coupling through
    the Woodbury identity (:func:`_anchor_coupling`).  It is rebuilt only
    when GMRES says it has gone stale.  Steps are damped on
    Deuflhard's natural monotonicity test (the NLEQ-ERR strategy): the
    simplified correction ``J^-1 F(z + a dz)`` must be shorter than ``dz``.
    ``|F|`` is no guide here -- the first full step raises it by three
    decades along the gauge direction and still lands, while undamped steps
    from other points diverge.

    The state is certified once ``|F| <= refine_tol``; like the
    fixed-boundary refinement it keeps stepping while a step still gains a
    decade, so it lands at the residual floor (3e-14 to 9e-14 on that deck).
    The anchored root does not depend on where VMEC stopped: from solves at
    ``ftol = 1e-10`` and ``1e-12`` the anchored objectives agree to 1e-10
    (optimized coils) and 8e-9 (a quarter of the way), where the unanchored
    ones differ by 2.8 % and 5 %.  A run that does not certify returns the
    host state with ``certified=False``; the callers turn that into an
    uncertified trial (status 3) or an error, never a silent pass.
    ``refine_tol = inf`` disables the anchor.
    """
    icfg = cfg.implicit
    tol = float(icfg.refine_tol)
    if not np.isfinite(tol) or tol <= 0.0:
        return state, _AnchorReport(True, np.nan, np.nan)
    arrays = (params, field_parameters, state, mask, rcon0, zcon0)
    params, field_parameters, state, mask, rcon0, zcon0 = im._device_pin(
        icfg, jax.tree.map(jnp.asarray, arrays))
    project = im._dof_projector(icfg, mask)
    z = project(state)
    lane = (params, field_parameters, state, rcon0, zcon0, mask)
    base = float(_anchor_norm(z, *lane, cfg=cfg))
    if not np.isfinite(base) or base <= tol:
        return state, _AnchorReport(bool(np.isfinite(base)), base, base)

    field = cfg.field_from_parameters(field_parameters)
    ns = int(icfg.resolution.ns)
    rt = dataclasses.replace(
        im.runtime_from_params(params, icfg), rcon0=rcon0, zcon0=zcon0,
        lfreeb=True, jmax=ns,
        presf_ns_scale=_presf_ns_scale_traceable(params, icfg.inp, ns))
    chunk = _bulk_probe_chunk(cfg, mask)
    debug = im._adjoint_debug_enabled()

    def iterate_state(z_at):
        return jax.tree.map(jnp.add, state, jax.tree.map(
            jnp.subtract, z_at, project(state)))

    def factor(z_at, response_at):
        """Preconditioner at the iterate ``z_at``: bulk factors + coupling."""
        at = iterate_state(z_at)
        bsqvac = jax.lax.stop_gradient(cfg.vacuum_program.bsq(at, rt, field))
        lower, diagonal, upper, row_scale, column_scale = _frozen_bulk_blocks(
            params, field_parameters, at, rcon0, zcon0, mask, z_at, bsqvac,
            cfg=cfg, probe_chunk_size=chunk)
        factors = _anchor_factor(lower, diagonal, upper, row_scale,
                                 column_scale)
        coupling = (_anchor_coupling(
            z_at, params, field_parameters, state, rcon0, zcon0, mask,
            response_at, factors, row_scale, column_scale, cfg=cfg)
            if _ANCHOR_WOODBURY else None)
        return factors, row_scale, column_scale, coupling

    def raw(z_at):
        return _anchor_raw_residual(z_at, *lane, cfg=cfg)

    def correction(force, z_at, rtol):
        """``(-J(z_at)^-1 F, Krylov iterations, relative residual)``."""
        packed, iterations, linear = _anchor_linear_solve(
            force, z_at, params, field_parameters, state, rcon0, zcon0, mask,
            response, *bulk, cfg=cfg, rtol=rtol)
        return packed, int(iterations), float(linear)

    initial_response = _edge_response(
        cfg, params, field_parameters, state, rcon0, zcon0)
    initial_bulk = bulk = factor(z, initial_response)
    z0, force0 = z, raw(z)
    best_z, best = z, base
    factorizations = 1
    steps = krylov = attempts = 0
    # An attempt whose first step would start at or above the factor the
    # previous attempt's first step accepted retraces that attempt exactly
    # (the halvings are the same deterministic tests), so it is skipped.
    first_accepted = np.inf
    for first_damping in _ANCHOR_FIRST_DAMPING:
        if best <= tol:
            break
        if first_damping >= first_accepted:
            continue
        # Every attempt starts again from the host state, so the anchored
        # state is still a function of that state alone.
        attempts += 1
        z, force, bulk, previous = z0, force0, initial_bulk, base
        damping = 0.25 * first_damping
        refactor = False
        for attempt_step in range(_ANCHOR_MAX_STEPS):
            # NESTOR's dense response is cheap (0.34 s) and exact at ``z``,
            # so it is rebuilt every step; the preconditioner is kept until
            # GMRES says it has gone stale.
            response = initial_response if z is z0 else _edge_response(
                cfg, params, field_parameters, iterate_state(z), rcon0, zcon0)
            if refactor:
                bulk = factor(z, response)
                factorizations += 1
            newton, iterations, linear = correction(force, z, _ANCHOR_FORCING)
            krylov += iterations
            refactor = iterations > _ANCHOR_REFACTOR_KRYLOV
            level = float(jnp.linalg.norm(newton))
            step = _unpack_projected(newton, mask, cfg=cfg)
            steps += 1
            # Damping on Deuflhard's natural monotonicity test (NLEQ-ERR): the
            # simplified correction ``J^-1 F(z + a dz)``, with this step's
            # Jacobian, must be shorter than ``dz``.  ``|F|`` itself is no guide:
            # the first full step routinely raises it by three decades along the
            # poloidal-angle gauge direction and still lands, while an undamped
            # run from some points diverges (|F| 2.4e-03 -> 2.2e+01).  The test
            # needs only a couple of digits.  Each step starts from four times
            # the last accepted factor and halves: Deuflhard's Kantorovich
            # prediction was measured here and under-damps the gauge direction
            # into 1/64 steps that never finish.
            damping = min(1.0, 4.0 * damping)
            trials = 0
            while True:
                trials += 1
                trial = jax.tree.map(lambda a, b: a + damping * b, z, step)
                trial_force = raw(trial)
                simplified, iterations, _ = correction(
                    trial_force, z, _ANCHOR_TEST_FORCING)
                krylov += iterations
                trial_level = float(jnp.linalg.norm(simplified))
                passed = bool(np.isfinite(trial_level)) and trial_level <= (
                    1.0 - 0.25 * damping) * level
                if passed or damping <= _ANCHOR_MIN_DAMPING:
                    break
                damping = max(_ANCHOR_MIN_DAMPING, 0.5 * damping)
            if attempt_step == 0:
                first_accepted = damping if passed else 0.0
            if not passed:
                if debug:
                    print(f"[vmex anchor] step {steps}: no damped step passes "
                          f"the natural monotonicity test (|dz| {level:.3e})")
                break
            z, force = trial, trial_force
            residual = float(_anchor_norm(z, *lane, cfg=cfg))
            if debug:
                print(f"[vmex anchor] step {steps}: |F| {previous:.3e} -> "
                      f"{residual:.3e}, |dz| {level:.3e} -> {trial_level:.3e}, "
                      f"damping {damping:.3g} after {trials} trial(s), Newton "
                      f"GMRES to {linear:.1e}, {krylov} Krylov iterations so far")
            gained = residual < 0.1 * previous
            previous = residual
            if residual < best:
                best_z, best = z, residual
            if best <= tol and (not gained or residual == 0.0):
                break
    im._count(icfg, anchors=1, anchor_steps=steps,
              anchor_krylov_iterations=krylov,
              anchor_factorizations=factorizations)
    if best > tol:
        return state, _AnchorReport(False, base, best, steps, krylov,
                                    factorizations, attempts=attempts)
    correction = jax.tree.map(jnp.subtract, best_z, project(state))
    return jax.tree.map(jnp.add, state, correction), _AnchorReport(
        True, base, best, steps, krylov, factorizations,
        float(im._tree_norm(correction)), attempts)


def _bulk_probe_chunk(cfg, mask) -> int:
    """Bulk-block probe chunk of the anchor and the Schur adjoint: one block.

    Each probe differentiates a three-surface kernel, so a full block of
    cotangents is small; on the single-stage deck (block 150) the blocks
    assemble in 1.1 s against 2.6 s at the fixed-boundary refinement's
    ``ceil(sqrt(block))`` and 6.6 s one column at a time.
    """
    return len(im._active_state_fields(cfg.implicit)) * int(mask.R_cos.shape[1])


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def solve_free_boundary_implicit_status(
    params: im.ImplicitParams,
    field_parameters: Any,
    cfg: FreeBoundaryImplicitConfig,
) -> tuple[SpectralState, Array, Array, Array]:
    """Differentiable state with an exception-free optimizer-trial status.

    Status 0 is derivative-certified, 1 denotes a failed solve, 2 an
    under-converged solve, and 3 a solve that met ``ftol`` but that Newton
    could not anchor on the coupled root (``refine_tol``). Only status 0
    evaluates the implicit pullback.

    The arguments are exactly those of :func:`solve_free_boundary_implicit`.
    The difference is the failure contract: a solve that would raise there
    returns here with status 1, the last hot-restart state (or a fresh
    initial state) in place of a converged one, and zero cotangents for both
    differentiable arguments, so an optimizer may probe infeasible points
    without an exception and without picking up a meaningless gradient.

    Returns
    -------
    ``(state, status, fsq, ratio)``.  ``state`` is the
    :class:`~vmex.core.solver.SpectralState`, differentiable only at status
    0.  ``status`` is the int32 code above.  ``fsq`` is the summed final
    force residual ``fsqr + fsqz + fsql``, and ``ratio`` is ``fsq / ftol``;
    status 2 is exactly ``ratio`` exceeding the configuration's
    ``max_fsq_ratio`` on a solve that did not converge.  Both are infinite
    on a failed solve.
    """
    icfg = cfg.implicit
    with im._device_context(icfg):
        params, field_parameters = im._device_pin(
            icfg, (params, field_parameters))
        state, _, _, _, status, fsq, ratio = _callback_status(
            params, field_parameters, cfg)
    return state, status, fsq, ratio


def _solve_status_fwd(params, field_parameters, cfg):
    icfg = cfg.implicit
    with im._device_context(icfg):
        params, field_parameters = im._device_pin(
            icfg, (params, field_parameters))
        state, mask, rcon0, zcon0, status, fsq, ratio = _callback_status(
            params, field_parameters, cfg)
        state, mask, rcon0, zcon0 = im._device_pin(
            icfg, (state, mask, rcon0, zcon0))
    saved = (params, field_parameters, state, mask, rcon0, zcon0, status)
    return (state, status, fsq, ratio), saved


def _solve_status_bwd(cfg, saved, cotangents):
    params, field_parameters, state, mask, rcon0, zcon0, status = saved
    state_bar, _, _, _ = cotangents
    zeros = (jax.tree.map(jnp.zeros_like, params),
             jax.tree.map(jnp.zeros_like, field_parameters))

    def success(values):
        prm, field, solved, dof_mask, rcon, zcon, bar = values
        return _solve_bwd(
            cfg, (prm, field, solved, dof_mask, rcon, zcon), bar)

    if not isinstance(status, jax.core.Tracer):
        return success(
            (params, field_parameters, state, mask, rcon0, zcon0, state_bar)
        ) if int(status) == 0 else zeros
    return jax.lax.cond(
        status == 0, success, lambda _: zeros,
        (params, field_parameters, state, mask, rcon0, zcon0, state_bar),
    )


# ``residual`` is a static jit key, so its identity must be stable across
# gradient calls — :func:`_projected_residual`'s memo provides exactly that.
# The previous per-call ``@jax.jit`` closure re-lowered and recompiled this
# transpose (the largest program of the backward pass) on every host adjoint.
@functools.partial(jax.jit, static_argnames=("residual",))
def _prepare_transpose(z, p, field, base, rcon, zcon, *, residual):
    """Save the coupled primal once; return a pytree of pullback residuals."""
    return jax.vjp(
        lambda zz: residual(zz, p, field, base, rcon, zcon), z
    )[1]


# ``response`` is a per-gradient value, so the response-linearized lane cannot
# be keyed on a closure the way :func:`_prepare_transpose` is without either
# recompiling every backward pass or reading a stale linearization. Taking the
# mask and the response as traced arguments leaves ``cfg`` the only static key,
# so one compiled transpose serves every gradient in a process.
@functools.partial(jax.jit, static_argnames=("cfg", "formulation"))
def _prepare_response_transpose(z, p, field, base, rcon, zcon, dof_mask,
                                response, *, cfg, formulation="preconditioned"):
    """Save the response-linearized transpose of the coupled root."""
    return jax.vjp(
        lambda zz: _projected_residual_lane(
            zz, p, field, base, rcon, zcon, dof_mask, None, response,
            cfg=cfg, formulation=formulation), z,
    )[1]


@jax.jit
def _transpose_matvec(value, pullback, template):
    """Apply the saved transpose without repeating the NESTOR/VMEC primal."""
    _, unravel = ravel_pytree(template)
    return ravel_pytree(pullback(unravel(value))[0])[0]


def _host_adjoint(
    residual, z_star, params, field_parameters, frozen, rcon0, zcon0, rhs, cfg,
    *, x0=None, fail="error", pullback=None, certify=False,
):
    """Solve one adjoint while reusing a separately compiled JAX matvec.

    Staging GCROT together with the coupled NESTOR--VMEC transpose makes XLA
    inline that large operator into every Arnoldi loop and greatly increases
    cold compilation memory. SciPy keeps the small Krylov bookkeeping on the
    host and calls one compiled JAX operator; only vectors cross the boundary.
    Saved primal intermediates stay on the device for this solve and are
    rebuilt at the next linearization point.

    ``pullback`` iterates on a cheaper transpose than ``residual``'s own --
    today the response-linearized one.  ``certify`` then re-measures the
    accepted solution on ``residual``'s exact transpose and, if the cheaper
    operator's answer misses the same acceptance every lane uses, continues
    from it on that exact operator instead of returning it.
    """
    rhs_flat, unravel = ravel_pytree(rhs)

    if pullback is None:
        pullback = _prepare_transpose(
            z_star, params, field_parameters, frozen, rcon0, zcon0,
            residual=residual)

    def matvec(value):
        return _transpose_matvec(value, pullback, z_star)

    matvec(rhs_flat).block_until_ready()
    dtype = np.asarray(rhs_flat).dtype
    shape = rhs_flat.shape
    calls = 0

    def apply(value):
        nonlocal calls
        calls += 1
        return np.asarray(matvec(jnp.asarray(value, dtype=rhs_flat.dtype)))

    matrix = LinearOperator((shape[0], shape[0]), matvec=apply, dtype=dtype)
    x0_flat = None if x0 is None else np.asarray(ravel_pytree(x0)[0])
    solution, _info = gcrotmk(
        matrix, np.asarray(rhs_flat), rtol=cfg.adjoint_tol, atol=0.0,
        m=min(cfg.adjoint_gcrot_m, shape[0]),
        k=min(cfg.adjoint_gcrot_k, shape[0]),
        maxiter=cfg.adjoint_maxiter, x0=x0_flat,
    )
    tolerance = float(im._adjoint_acceptance(
        cfg, np.linalg.norm(np.asarray(rhs_flat))))
    if certify:
        exact = _prepare_transpose(
            z_star, params, field_parameters, frozen, rcon0, zcon0,
            residual=residual)
        residual_norm = float(np.linalg.norm(
            np.asarray(rhs_flat) - np.asarray(_transpose_matvec(
                jnp.asarray(solution, dtype=rhs_flat.dtype), exact, z_star))))
        if np.isfinite(residual_norm) and residual_norm <= tolerance:
            return unravel(jnp.asarray(solution, dtype=rhs_flat.dtype))
        # The cheaper operator did not certify here; finish on the exact one
        # rather than hand back a derivative this lane cannot vouch for, and
        # count it: a lane that always lands here has silently lost its gain.
        im._count(cfg, adjoint_certificate_fallbacks=1)
        warm = (unravel(jnp.asarray(solution, dtype=rhs_flat.dtype))
                if np.all(np.isfinite(solution)) else None)
        return _host_adjoint(
            residual, z_star, params, field_parameters, frozen, rcon0, zcon0,
            rhs, cfg, x0=warm, fail=fail)
    residual_norm = float(np.linalg.norm(np.asarray(rhs_flat) - apply(solution)))
    if not np.isfinite(residual_norm) or residual_norm > tolerance:
        if fail != "best_effort" or not np.isfinite(residual_norm):
            im._raise_adjoint_unconverged(
                cfg, iterations=calls, residual_norm=residual_norm,
                tolerance=tolerance, method="host GCROT",
            )
        warnings.warn(
            "free-boundary adjoint stalled: residual "
            f"{residual_norm:.3e} > acceptance {tolerance:.3e} after {calls} "
            "Krylov iterations; returning the best-effort solution because "
            "adjoint_fail='best_effort'. The gradient at this point is "
            "inaccurate; a line search should reject it.",
            RuntimeWarning, stacklevel=2,
        )
    return unravel(jnp.asarray(solution, dtype=rhs_flat.dtype))


solve_free_boundary_implicit.defvjp(_solve_fwd, _solve_bwd)
solve_free_boundary_implicit_status.defvjp(_solve_status_fwd, _solve_status_bwd)


__all__ = [
    "FreeBoundaryImplicitConfig",
    "make_free_boundary_config",
    "solve_free_boundary_implicit",
    "solve_free_boundary_implicit_status",
]
