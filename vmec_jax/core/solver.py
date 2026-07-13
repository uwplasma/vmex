"""Single-grid fixed-boundary solve loop: funct3d evaluation + eqsolve iteration.

Wires the ported core modules (:mod:`geometry`, :mod:`fields`, :mod:`forces`,
:mod:`residuals`, :mod:`preconditioner`, :mod:`step`) into one force
evaluation (:func:`evaluate_forces`) and the damped Richardson iteration
(:func:`solve`) with VMEC2000 restart/escalation semantics.

VMEC2000 counterparts
---------------------
- ``Sources/General/funct3d.f`` — one force evaluation: odd-m internal
  synthesis (``totzsps``), constraint baselines ``rcon0/zcon0``, Jacobian
  (early exit on sign change), ``bcovar`` (with the ``ns4 = 25``-iteration
  refresh of the preconditioner / force norms / ``tcon``), constraint force
  (``alias``), MHD forces, ``tomnsps``, ``gc = gc*scalxc`` and ``residue``:
  :func:`evaluate_forces`.
- ``Sources/TimeStep/evolve.f`` — damping window + momentum update (via
  :mod:`vmec_jax.core.step`) and ``TimeStepControl`` (store/restore of the
  best state, residual-growth back-off ``irst = 3``): the iteration body.
- ``Sources/TimeStep/eqsolve.f`` — the outer force-iteration loop: axis
  re-guess after a first bad Jacobian (``ijacob == 0``), time-step resets at
  ``ijacob = 25/50``, abort at ``ijacob >= 75`` (``jac75_flag``), convergence
  when ``fsqr, fsqz, fsql <= ftolv`` simultaneously: :func:`solve`.
- ``Sources/TimeStep/restart.f`` — state restore semantics (zero velocity,
  ``delt`` rescaling) via :mod:`vmec_jax.core.step`.

Execution lanes (plan.md §5.3)
------------------------------
Both lanes share one traced single-iteration body:

- ``mode="cli"``: Python ``while`` over a jitted ``lax.scan`` block of
  ``block_size = 10`` iterations, with host residual checks and VMEC2000
  screen printing (``printout.f`` cadence) between blocks;
- ``mode="jit"``: a single ``lax.while_loop`` over the same body (fully
  traced iteration; the setup and error raising remain host code).

The per-iteration ``(fsqr, fsqz, fsql, fsqr1, fsqz1, fsql1, ...)`` trajectory
is recorded in a preallocated buffer carried through the loop, so the two
lanes produce identical histories.

Structural executable reuse (2026-07-09, plan.md Phase 2 item (1))
------------------------------------------------------------------
:class:`SolverRuntime` is a registered pytree passed as an *argument* to the
module-level jitted lanes :func:`_while_lane`/:func:`_block_lane` (previously
per-runtime closures cached by object identity).  Array-valued run data
(:class:`~vmec_jax.core.setup.RunSetup`, ``rcon0/zcon0``) are pytree data;
the hashable configuration (:class:`~vmec_jax.core.fourier.Resolution`,
``gamma/tcon0/ftol/max_iterations/time_step0/nstep/jmax``) is pytree meta;
the NumPy mode/trig/weight/gather tables — consumed with ``np.*``/fancy
indexing at trace time — are derived from the meta resolution through the
cached :func:`_static_tables` and never enter the pytree.  Consequence: two
different runtimes with equal structure (e.g. different boundary values at
one resolution, hot restarts, multigrid re-runs) share one XLA executable
per lane.  :func:`solve` additionally accepts ``initial_state`` (hot
restart), and the stage machinery is factored into :func:`_solve_stage`/
:func:`_finalize` for reuse by :func:`vmec_jax.core.multigrid.solve_multigrid`.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)  # float64 mandatory (plan.md §7.7)


def _harden_compilation_cache() -> None:
    """Idempotent persistent compile-cache setup at ``core.solver`` import.

    benchmarks/gpu_baseline.json pitfall (2026-07-09): launching Python from a
    cwd that contains the repo checkout can resolve ``vmec_jax`` as a
    *namespace* package, so ``vmec_jax/__init__.py`` — which configures the
    persistent XLA compilation cache — never runs and every ``solve()`` pays a
    full recompile (~7 s vs ~1.7 s warm on CUDA for solovev).  This module
    always executes on any core solve path, so the cache policy is re-applied
    here: warn on the shadowed import, then configure the ``_compat`` cache
    defaults *only* when neither the user (``JAX_COMPILATION_CACHE_DIR`` env /
    an explicit ``jax.config.update``) nor ``vmec_jax/__init__`` already set a
    cache directory.
    """
    import os
    import sys
    import warnings

    top = sys.modules.get(__name__.partition(".")[0])
    if top is not None and getattr(top, "__file__", None) is None:
        warnings.warn(
            "vmec_jax was imported as a namespace package (vmec_jax.__file__ is "
            "missing) — its __init__.py never ran.  This usually means the "
            "current working directory shadows the installed package (e.g. "
            "running Python from the directory containing the vmec_jax "
            "checkout).  Change directory or fix sys.path; package-level "
            "defaults such as the persistent compilation cache are otherwise "
            "skipped.",
            RuntimeWarning,
            stacklevel=3,
        )
    try:
        current = jax.config.jax_compilation_cache_dir
    except AttributeError:  # pragma: no cover - very old jax
        return
    if current:  # user env/jax.config or vmec_jax/__init__ already set it
        return
    try:
        from .._compat import _configure_compilation_cache, _default_compilation_cache_dir
    except Exception:  # pragma: no cover - core used standalone
        return
    cache_dir = _default_compilation_cache_dir()
    if cache_dir is None:  # policy says no cache (e.g. CPU-only, not forced)
        return
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except OSError:  # pragma: no cover - unwritable cache location
        return
    _configure_compilation_cache(jax, cache_dir)


_harden_compilation_cache()

import jax.numpy as jnp
from jax import lax

from .errors import (
    BAD_JACOBIAN_FLAG, JAC75_FLAG, MISC_ERROR_FLAG, MORE_ITER_FLAG,
    NORM_TERM_FLAG, SUCCESSFUL_TERM_FLAG,
)
from .fields import (
    constraint_scaling, energies_and_force_norms, magnetic_fields,
    metric_elements, preconditioned_force_norm,
)
from .forces import mhd_forces, spectral_mhd_forces
from .geometry import half_mesh_jacobian
from .preconditioner import lamcal, precondn, scalfor_matrices
from .preconditioner_2d import Prec2DConfig, newton_direction
from .residuals import (
    ForceResiduals, apply_lambda_preconditioner,
    apply_radial_preconditioner, edge_force_condition, force_residuals,
    m1_residue_rotation, m1_zero_condition,
    preconditioned_residuals, scale_m1_preconditioner_rhs, scalxc_scale_force,
    zero_m1_z_force,
)
from .step import (
    DAMPING_CAP, GROWTH_BACKOFF_DIVISOR, GROWTH_LIMIT, GROWTH_MIN_ITERATIONS,
    JACOBIAN_RESET_FACTOR, NDAMP, RESTART_GROWTH, RESTART_JACOBIAN, STEP_OK,
    StepControl, damping_coefficients, momentum_update,
)

__all__ = [
    "NS4", "SpectralState", "PreconditionerCache", "FunctDiagnostics",
    "SolverRuntime", "SolveResult", "Prec2DConfig", "resolution_from_input",
    "prepare_runtime", "evaluate_forces", "solve",
    "hot_restart_state", "runtime_with_baselines",
]

Array = Any

#: bcovar.f ``ns4``: preconditioner / force-norm / tcon refresh cadence.
NS4 = 25

#: Number of iterations per jitted CLI-lane scan block (plan.md §5.3).
BLOCK_SIZE = 10

#: Trajectory buffer columns:
#: iter, fsqr, fsqz, fsql, fsqr1, fsqz1, fsql1, r00, z00, wmhd, delt.
_TRAJ_COLS = 14

_TWO_PI_SQ = (2.0 * np.pi) ** 2


def _select(mask: Array, new, old):
    """Elementwise pytree select ``mask ? new : old`` (scalar traced mask)."""
    return jax.tree.map(lambda a, b: jnp.where(mask, a, b), new, old)


# -- State containers ------------------------------------------------------------------------------------------------


from .solver_runtime import (
    FunctDiagnostics,
    PreconditionerCache,
    SolverRuntime,
    SpectralState,
    _constraint_baselines as _constraint_baselines,
    _EvalResult,
    _force_to_state,
    _geometry,
    _initial_state as _initial_state,
    _LoopCarry,
    _physical_coefficients as _physical_coefficients,
    _static_tables as _static_tables,
    _zero_cache,
    hot_restart_state,
    prepare_runtime,
    resolution_from_input,
    runtime_with_baselines,
)


# -- One funct3d pass ------------------------------------------------------------------------------------------------


def _force_pipeline(
    *,
    geometry, jacobian, metrics, fields,
    R_cos: Array, R_sin: Array, Z_cos: Array, Z_sin: Array,
    cache: PreconditionerCache, rt: SolverRuntime,
    iteration: Array, fsqz_previous: Array,
) -> tuple[Any, Any]:
    """MHD forces -> residue.f90 chain -> scalfor/faclam preconditioning.

    The ``funct3d.f`` -> ``forces.f`` -> ``tomnsps`` -> ``residue.f90`` segment
    that both :func:`_evaluate` (the iteration body) and
    :func:`_preconditioned_force_signed` (the 2D-preconditioner force map)
    consume, factored out so they cannot drift.  Returns ``(scaled,
    preconditioned)``: the ``scalxc``-scaled force (input to the invariant
    residuals ``getfsq``) and the 1D-preconditioned force (input to
    ``fsqr1/fsqz1`` and the update direction ``gc``).  All preconditioner
    matrices come from the frozen ``cache`` (``ns4`` cadence).
    """
    setup = rt.setup
    res = rt.resolution
    s = setup.s_full
    hs = setup.hs

    forces = mhd_forces(
        geometry=geometry, jacobian=jacobian, metrics=metrics, fields=fields,
        R_cos=R_cos, R_sin=R_sin, Z_cos=Z_cos, Z_sin=Z_sin,
        modes=rt.modes, trig=rt.trig, s=s, phipf=setup.phipf,
        tcon=cache.tcon, signgs=setup.signgs, rcon0=rt.rcon0, zcon0=rt.zcon0,
    )
    if rt.lfreeb:
        # forces.f (ivac >= 1): vacuum-pressure edge force.  funct3d.f builds
        # rbsq = (bsqvac + presf_ns) * (r1(ns,0) + r1(ns,1)) * ohs with
        # presf_ns = [pmass(1)/pmass(hs*(ns-1.5))] * pres(ns), then forces.f
        # adds zu0*rbsq to the even AND odd armn edge rows (and -ru0*rbsq to
        # azmn).  sqrts(ns) = 1, so even+odd sums are the physical edge row.
        presf_ns = jnp.asarray(rt.presf_ns_scale) * fields.pressure[-1]
        gcon_edge = jnp.asarray(rt.bsqvac_edge) + presf_ns
        r1_edge = geometry.R_even[-1] + geometry.R_odd[-1]
        rbsq = gcon_edge * r1_edge / hs
        ru0, zu0 = geometry.theta_derivatives_full(s)
        forces = replace(
            forces,
            force_R_even=jnp.asarray(forces.force_R_even).at[-1].add(zu0[-1] * rbsq),
            force_R_odd=jnp.asarray(forces.force_R_odd).at[-1].add(zu0[-1] * rbsq),
            force_Z_even=jnp.asarray(forces.force_Z_even).at[-1].add(-ru0[-1] * rbsq),
            force_Z_odd=jnp.asarray(forces.force_Z_odd).at[-1].add(-ru0[-1] * rbsq),
        )
    spectral = spectral_mhd_forces(
        forces, mpol=res.mpol, ntor=res.ntor, trig=rt.trig, include_edge=bool(rt.lfreeb)
    )

    # -- residue.f90 chain ---------------------------------------------------
    rotated = m1_residue_rotation(spectral, lconm1=setup.lconm1)
    zero_gate = m1_zero_condition(fsqz_previous=fsqz_previous, iterations_since_restart=iteration)
    released = zero_m1_z_force(rotated, zero_gate)
    scaled = scalxc_scale_force(released, s=s)
    if setup.lthreed or setup.lasym:
        rhs = scale_m1_preconditioner_rhs(
            scaled, coefficients_R=cache.coefficients_R,
            coefficients_Z=cache.coefficients_Z, lconm1=setup.lconm1,
        )
    else:
        rhs = scaled
    solved = apply_radial_preconditioner(rhs, matrices_R=cache.matrices_R, matrices_Z=cache.matrices_Z, jmax=rt.jmax)
    preconditioned = apply_lambda_preconditioner(solved, cache.faclam)
    return scaled, preconditioned


def _force_maps_at_state(
    state: SpectralState, cache: PreconditionerCache, rt: SolverRuntime,
    *, iteration: Array, fsqz_previous: Array,
) -> tuple[Any, Any, Any, Any]:
    """Jacobian, energies, raw force, and preconditioned force at ``state``.

    This is the shared fixed-cache evaluation used by both the Newton operator
    and its physical-force line search. It deliberately omits the expensive
    preconditioner refresh candidates in :func:`_evaluate`.
    """
    setup = rt.setup
    s = setup.s_full
    (R_cos, R_sin, Z_cos, Z_sin), geometry = _geometry(state, rt)
    jacobian = half_mesh_jacobian(geometry, s=s)
    metrics = metric_elements(geometry, s=s)
    fields = magnetic_fields(
        geometry=geometry, jacobian=jacobian, metrics=metrics, trig=rt.trig,
        s=s, phips=setup.phips, phipf=setup.phipf, chips=setup.chips,
        signgs=setup.signgs, gamma=rt.gamma, mass=setup.mass,
        ncurr=setup.ncurr, enclosed_current=setup.icurv,
    )
    energies = energies_and_force_norms(
        jacobian=jacobian, metrics=metrics, fields=fields, trig=rt.trig,
        s=s, signgs=setup.signgs,
    )
    scaled, preconditioned = _force_pipeline(
        geometry=geometry, jacobian=jacobian, metrics=metrics, fields=fields,
        R_cos=R_cos, R_sin=R_sin, Z_cos=Z_cos, Z_sin=Z_sin,
        cache=cache, rt=rt, iteration=iteration, fsqz_previous=fsqz_previous,
    )
    return jacobian, energies, scaled, preconditioned


def _preconditioned_force_signed(
    state: SpectralState, cache: PreconditionerCache, rt: SolverRuntime,
    *, iteration: Array, fsqz_previous: Array,
) -> SpectralState:
    """1D-preconditioned force map ``state -> gc`` at a frozen ``cache``.

    Reproduces the ``gc`` that :func:`_evaluate` returns (same
    :func:`_force_pipeline`, same signed packing via :func:`_force_to_state`),
    but as a standalone pure function of ``state`` with everything else frozen:
    exactly the map whose Jacobian the 2D block preconditioner needs.
    :func:`jax.jvp` of this function is the exact block-tridiagonal
    Hessian-vector product (VMEC2000 ``precon2d.f`` ``compute_blocks``, without
    the finite-difference jogs).  Fixed-boundary only (``lfreeb=False``); the
    frozen ``iteration``/``fsqz_previous`` fix the ``residue.f90`` m=1 release
    gate so the map matches the ``gc`` produced at the linearization point.
    """
    _, _, _, preconditioned = _force_maps_at_state(
        state, cache, rt, iteration=iteration, fsqz_previous=fsqz_previous,
    )
    return _force_to_state(preconditioned, rt)


_ALL_CHANNELS = ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")


def _newton_active_indices(rt: SolverRuntime, channels: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Flat indices of physical degrees of freedom in the Newton system."""

    ns, mnmax = rt.resolution.ns, rt.modes.mnmax
    m = np.asarray(rt.modes.m)
    zero_mode = (m == 0) & (np.asarray(rt.modes.n) == 0)
    indices: dict[str, np.ndarray] = {}
    for channel in channels:
        mask = np.zeros((ns, mnmax), dtype=bool)
        if channel.startswith(("R_", "Z_")):
            mask[: rt.jmax] = True
            mask[0, m > 0] = False
        else:
            mask[1:] = True  # lambda axis values are closure/gauge data
        if channel.endswith("_sin") or channel.startswith("L_"):
            mask[:, zero_mode] = False
        indices[channel] = np.flatnonzero(mask.reshape(-1))
    return indices


def _newton_step(
    rt: SolverRuntime, state: SpectralState, gc_signed: SpectralState,
    cache: PreconditionerCache, iteration: Array, fsqz_previous: Array,
    fsq_raw: Array, gate: Array,
) -> tuple[SpectralState, Array, Array, Array, Array]:
    """2D-preconditioner Newton direction, gated on the activation predicate.

    Returns ``(direction, active, accepted_step, linear_residual,
    lambda_row_scale)``. When ``active`` (finest grid,
    ``fsq_raw < threshold``, past ``start_iteration``, on the configured
    cadence, and the base ``gate``), ``direction`` is the damped-Newton
    replacement for the 1D force
    ``gc_signed``: it solves ``J delta = -gc_signed`` with matrix-free GMRES
    (``J = d(preconditioned force)/d(state)`` at ``state``, exact HVP via
    ``jax.jvp``), so ``state += cfg.step * delta`` is the block-preconditioned
    update (``precon2d.f`` ``block_precond``: ``gc <- -H^{-1} gc``).  Otherwise
    ``direction = gc_signed`` and ``active = False``.  The linear solve runs
    only over physical, evolved entries of the non-trivial spectral channels
    (symmetric: R_cos/Z_sin/L_sin). Fixed R/Z edge rows, axis-null harmonics,
    lambda-axis values, and identically zero/gauge modes are omitted. Optional
    backtracking minimizes the largest physical force component and rejects
    sign-changing Jacobians; a rejected correction returns to the regular VMEC
    update for that iteration.
    Only reached when ``rt.prec2d is not None`` (the branch is otherwise never
    traced, keeping the 1D-only path byte-identical).
    """
    cfg = rt.prec2d
    if not cfg.finest:  # non-finest multigrid stage: never activate (static)
        dtype = rt.setup.s_full.dtype
        return (
            gc_signed,
            jnp.zeros((), dtype=bool),
            jnp.asarray(-1.0, dtype=dtype),
            jnp.asarray(jnp.nan, dtype=dtype),
            jnp.asarray(cfg.row_scales[2], dtype=dtype),
        )

    cadence = ((iteration - cfg.start_iteration) % max(int(cfg.interval), 1)) == 0
    active = (
        gate & (fsq_raw < cfg.threshold) & (iteration >= cfg.start_iteration)
        & cadence
    )
    channels = _ALL_CHANNELS if rt.setup.lasym else ("R_cos", "Z_sin", "L_sin")
    indices = _newton_active_indices(rt, channels)
    dtype = rt.setup.s_full.dtype

    def active_norm(prefix: str) -> Array:
        norm_squared = jnp.asarray(0.0, dtype=dtype)
        for channel in channels:
            if channel.startswith(prefix):
                values = getattr(gc_signed, channel).reshape(-1)[indices[channel]]
                norm_squared = norm_squared + jnp.vdot(values, values).real
        return jnp.sqrt(norm_squared)

    lambda_row_scale = jnp.asarray(cfg.row_scales[2], dtype=dtype)
    if cfg.auto_balance_lambda:
        geometry_norm = jnp.maximum(active_norm("R_"), active_norm("Z_"))
        lambda_norm = active_norm("L_")
        low, high = cfg.lambda_scale_bounds
        lambda_row_scale = jnp.clip(
            cfg.lambda_balance_target * geometry_norm
            / jnp.maximum(lambda_norm, jnp.finfo(dtype).tiny),
            low, high,
        )
    row_scale = {
        channel: jnp.asarray(
            cfg.row_scales[0 if channel.startswith("R_") else
                           1 if channel.startswith("Z_") else 2], dtype=dtype,
        )
        for channel in channels
    }
    for channel in channels:
        if channel.startswith("L_"):
            row_scale[channel] = lambda_row_scale

    def to_full(reduced: dict) -> SpectralState:
        full = {}
        for channel in _ALL_CHANNELS:
            base = getattr(state, channel)
            if channel in reduced:
                flat = base.reshape(-1).at[indices[channel]].set(reduced[channel])
                base = flat.reshape(base.shape)
            full[channel] = base
        return SpectralState(**full)

    def g_reduced(reduced: dict) -> dict:
        gc_full = _preconditioned_force_signed(
            to_full(reduced), cache, rt, iteration=iteration, fsqz_previous=fsqz_previous,
        )
        return {c: getattr(gc_full, c).reshape(-1)[indices[c]] for c in channels}

    def g_scaled(reduced: dict) -> dict:
        force = g_reduced(reduced)
        return {channel: row_scale[channel] * force[channel] for channel in channels}

    x0 = {c: getattr(state, c).reshape(-1)[indices[c]] for c in channels}
    rhs = {
        c: -row_scale[c] * getattr(gc_signed, c).reshape(-1)[indices[c]]
        for c in channels
    }

    def do_newton(_):
        delta, sol = newton_direction(g_scaled, x0, rhs, cfg)
        accepted = jnp.ones((), dtype=bool)
        factor = jnp.ones((), dtype=rt.setup.s_full.dtype)
        if cfg.backtracking:
            factors = jnp.asarray([
                1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0,
            ])

            def objective(factor):
                candidate = {
                    c: x0[c] + cfg.step * factor * delta[c]
                    for c in channels
                }
                jacobian, energies, scaled, _ = _force_maps_at_state(
                    to_full(candidate), cache, rt,
                    iteration=iteration, fsqz_previous=fsqz_previous,
                )
                residuals = force_residuals(
                    scaled, fnorm=cache.fnorm, fnormL=cache.fnormL,
                    r1=energies.r1, include_edge=False,
                )
                merit = jnp.maximum(
                    residuals.fsqr, jnp.maximum(residuals.fsqz, residuals.fsql)
                )
                return jnp.where(
                    jacobian.jacobian_sign_changed, jnp.asarray(jnp.inf), merit,
                )

            objectives = lax.map(objective, factors)
            factor = factors[jnp.argmin(objectives)]
            delta = {c: factor * value for c, value in delta.items()}
            accepted = factor > 0.0
        full = {}
        for channel in _ALL_CHANNELS:
            value = jnp.zeros_like(getattr(gc_signed, channel))
            if channel in delta:
                flat = value.reshape(-1).at[indices[channel]].set(delta[channel])
                value = flat.reshape(value.shape)
            full[channel] = value
        accepted_step = jnp.where(accepted, cfg.step * factor, 0.0)
        return (
            SpectralState(**full), accepted, accepted_step,
            sol.residual_norm, lambda_row_scale,
        )

    direction, accepted, accepted_step, linear_residual, used_lambda_scale = lax.cond(
        active, do_newton,
        lambda _: (
            gc_signed,
            jnp.zeros((), dtype=bool),
            jnp.asarray(-1.0, dtype=dtype),
            jnp.asarray(jnp.nan, dtype=dtype),
            lambda_row_scale,
        ),
        operand=None,
    )
    return (
        direction, active & accepted, accepted_step,
        linear_residual, used_lambda_scale,
    )


def _evaluate(
    state: SpectralState, cache: PreconditionerCache, iteration: Array,
    iter_last_reset: Array, fsqz_previous: Array, rt: SolverRuntime,
    fsq_rz_previous: Array | None = None,
) -> _EvalResult:
    """One funct3d.f pass (fixed boundary), pure and jit-friendly.

    Order of operations follows ``funct3d.f``: synthesis -> Jacobian (sign
    flag carried, never branched on) -> ``bcovar`` fields/energies with the
    ``mod(iter2-iter1, ns4) == 0`` cache refresh -> constraint force with the
    cached ``tcon`` and static ``rcon0/zcon0`` -> ``forces`` -> ``tomnsps`` ->
    ``gc*scalxc`` -> ``residue`` (m=1 rotation + conditional Z-force zeroing,
    ``getfsq``, ``scale_m1`` + ``scalfor`` + ``faclam`` preconditioning,
    ``fsqr1/fsqz1/fsql1``).  On a Jacobian sign change VMEC skips everything
    past the Jacobian; here the computation proceeds (traced) and the caller
    discards results, while the cache refresh is suppressed exactly.
    """
    setup = rt.setup
    res = rt.resolution
    s = setup.s_full
    ns = int(s.shape[0])
    hs = setup.hs

    (R_cos, R_sin, Z_cos, Z_sin), geometry = _geometry(state, rt)
    jacobian = half_mesh_jacobian(geometry, s=s)
    jac_changed = jacobian.jacobian_sign_changed

    metrics = metric_elements(geometry, s=s)
    fields = magnetic_fields(
        geometry=geometry, jacobian=jacobian, metrics=metrics, trig=rt.trig,
        s=s, phips=setup.phips, phipf=setup.phipf, chips=setup.chips,
        signgs=setup.signgs, gamma=rt.gamma, mass=setup.mass,
        ncurr=setup.ncurr, enclosed_current=setup.icurv,
    )
    energies = energies_and_force_norms(
        jacobian=jacobian, metrics=metrics, fields=fields, trig=rt.trig, s=s, signgs=setup.signgs,
    )

    # -- ns4-cadence refresh candidates (bcovar.f) --------------------------
    tcon_new = constraint_scaling(
        tcon0=rt.tcon0, geometry=geometry, jacobian=jacobian,
        total_pressure=fields.total_pressure, trig=rt.trig, s=s,
    )
    common = dict(
        r12_half=jacobian.r12[1:], bsq_half=fields.total_pressure[1:],
        bsupv_half=fields.bsupv[1:], sqrt_g_half=jacobian.sqrt_g[1:],
        angular_weight=rt.weights, delta_s=hs, ns=ns,
    )
    coefficients_R = precondn(
        dxds_half=jacobian.dZ_ds[1:], dxdu_half=jacobian.zu12[1:],
        dxdu_even_full=geometry.dZ_dtheta_even, dxdu_odd_full=geometry.dZ_dtheta_odd,
        x_odd_full=geometry.Z_odd, **common,
    )
    coefficients_Z = precondn(
        dxds_half=jacobian.dR_ds[1:], dxdu_half=jacobian.ru12[1:],
        dxdu_even_full=geometry.dR_dtheta_even, dxdu_odd_full=geometry.dR_dtheta_odd,
        x_odd_full=geometry.R_odd, **common,
    )
    # jmax follows scalfor.f: ns-1 fixed boundary (rt.jmax default), ns once
    # the vacuum field is on (free-boundary lane) — activates the
    # EDGE_PEDESTAL / ZC(0,0) edge stiffening inside scalfor_matrices.
    mat_kwargs = dict(delta_s=hs, mpol=res.mpol, ntor=res.ntor, nfp=res.nfp, ns=ns,
                      jmax=int(rt.jmax))
    matrices_R = scalfor_matrices(coefficients_R, stabilize_edge_zc00=False, **mat_kwargs)
    matrices_Z = scalfor_matrices(coefficients_Z, stabilize_edge_zc00=True, **mat_kwargs)
    faclam_new = lamcal(
        guu_half=metrics.guu, guv_half=metrics.guv, gvv_half=metrics.gvv,
        sqrt_g_half=jacobian.sqrt_g, lamscale=fields.lamscale,
        angular_weight=rt.weights, mpol=res.mpol, ntor=res.ntor, nfp=res.nfp,
        lthreed=setup.lthreed,
    )
    fnorm1_new = preconditioned_force_norm(
        R_cos=state.R_cos, Z_sin=state.Z_sin, modes=rt.modes,
        R_sin=state.R_sin if setup.lasym else None,
        Z_cos=state.Z_cos if setup.lasym else None,
    )
    fresh = PreconditionerCache(
        tcon=tcon_new, fnorm=energies.fnorm, fnormL=energies.fnormL,
        fnorm1=fnorm1_new, coefficients_R=coefficients_R,
        coefficients_Z=coefficients_Z, matrices_R=matrices_R,
        matrices_Z=matrices_Z, faclam=faclam_new,
    )
    refresh = (((iteration - iter_last_reset) % NS4) == 0) & (~jac_changed)
    cache = _select(refresh, fresh, cache)

    # -- constraint + MHD forces + residue.f90 preconditioning chain --------
    # The mhd_forces -> tomnsps -> residue -> scalfor/faclam pipeline is shared
    # verbatim with the 2D-preconditioner force map (:func:`_force_pipeline`)
    # so the two can never drift; ``scaled`` feeds the invariant residuals,
    # ``preconditioned`` the fsqr1/fsqz1 residuals and the update force ``gc``.
    scaled, preconditioned = _force_pipeline(
        geometry=geometry, jacobian=jacobian, metrics=metrics, fields=fields,
        R_cos=R_cos, R_sin=R_sin, Z_cos=Z_cos, Z_sin=Z_sin,
        cache=cache, rt=rt, iteration=iteration, fsqz_previous=fsqz_previous,
    )
    if rt.lfreeb:
        # residue.f90 medge rule: the edge rows join fsqr/fsqz only when
        # iter2 - iter1 < 50 and the previous fsqr+fsqz < 1e-6 (traced
        # condition; both variants are cheap sums).
        res_int = force_residuals(scaled, fnorm=cache.fnorm, fnormL=cache.fnormL,
                                  r1=energies.r1, include_edge=False)
        res_edge = force_residuals(scaled, fnorm=cache.fnorm, fnormL=cache.fnormL,
                                   r1=energies.r1, include_edge=True)
        fsq_rz_prev = jnp.asarray(1.0, dtype=s.dtype) if fsq_rz_previous is None \
            else jnp.asarray(fsq_rz_previous)
        medge = edge_force_condition(
            fsq_rz_previous=fsq_rz_prev,
            iterations_since_restart=iteration - iter_last_reset,
            free_boundary=True,
        )
        residuals = _select(medge, res_edge, res_int)
    else:
        residuals = force_residuals(scaled, fnorm=cache.fnorm, fnormL=cache.fnormL, r1=energies.r1, include_edge=False)
    pre = preconditioned_residuals(preconditioned, fnorm1=cache.fnorm1, delta_s=hs)

    sqrt_s0 = jnp.sqrt(jnp.maximum(s[0], 0.0))
    r00 = geometry.R_even[0, 0, 0] + sqrt_s0 * geometry.R_odd[0, 0, 0]
    z00 = geometry.Z_even[0, 0, 0] + sqrt_s0 * geometry.Z_odd[0, 0, 0]

    return _EvalResult(
        gc=_force_to_state(preconditioned, rt),
        residuals=residuals, pre=pre, wb=energies.wb, wp=energies.wp,
        r00=r00, z00=z00, jacobian_sign_changed=jac_changed, cache=cache,
    )


def evaluate_forces(
    state: SpectralState,
    runtime: SolverRuntime,
    *,
    cache: PreconditionerCache | None = None,
    iteration: int = 1, iter_last_reset: int = 1, fsqz_previous: float = 1.0,
) -> tuple[SpectralState, ForceResiduals, FunctDiagnostics]:
    """One funct3d pass: preconditioned force ``gc``, residuals, diagnostics.

    VMEC2000: ``funct3d.f`` + ``residue.f90``.  ``cache=None`` forces a
    preconditioner/norm/tcon refresh (as VMEC does whenever
    ``iter2 == iter1``); pass the returned ``diagnostics.cache`` back in to
    reproduce the ``ns4 = 25`` cadence.  The returned ``gc`` is in the same
    signed spectral packing as ``state`` and feeds
    :func:`vmec_jax.core.step.momentum_update` directly.
    """
    if cache is None:
        cache = _zero_cache(runtime)
        iter_last_reset = iteration  # force refresh
    result = _evaluate(
        state, cache, jnp.asarray(iteration), jnp.asarray(iter_last_reset),
        jnp.asarray(fsqz_previous), runtime,
    )
    diagnostics = FunctDiagnostics(
        preconditioned=result.pre,
        wb=result.wb, wp=result.wp, r00=result.r00, z00=result.z00,
        jacobian_sign_changed=result.jacobian_sign_changed, cache=result.cache,
    )
    return result.gc, result.residuals, diagnostics


# -- Iteration body (evolve.f + TimeStepControl + the eqsolve.f checks) ----------------------------------------------


def _make_body(rt: SolverRuntime) -> Callable[[_LoopCarry], _LoopCarry]:
    """Build the traced single-iteration body shared by both lanes."""
    ftol = rt.ftol
    max_iter = rt.max_iterations
    gamma = rt.gamma

    def body(carry: _LoopCarry) -> _LoopCarry:
        it = carry.iteration
        running = jnp.logical_not(carry.done)

        # ---- funct3d (evolve.f) -------------------------------------------
        e1 = _evaluate(carry.state, carry.cache, it, carry.iter1, carry.fsqz, rt,
                       carry.fsqr + carry.fsqz)
        jac1 = e1.jacobian_sign_changed
        # On irst=2 funct3d skips residue: the module residuals stay stale.
        fsqr_c = jnp.where(jac1, carry.fsqr, e1.residuals.fsqr)
        fsqz_c = jnp.where(jac1, carry.fsqz, e1.residuals.fsqz)
        fsql_c = jnp.where(jac1, carry.fsql, e1.residuals.fsql)
        fsq0 = fsqr_c + fsqz_c + fsql_c

        converged = (~jac1) & (fsqr_c <= ftol) & (fsqz_c <= ftol) & (fsql_c <= ftol)
        bad_init = jac1 & (it == 1)
        stepping = running & (~converged) & (~bad_init)

        # ---- TimeStepControl (evolve.f) ------------------------------------
        first = it == carry.iter1
        fsq_prev = carry.fsq
        res0_f = jnp.where(first, fsq_prev, carry.res0)
        res1_f = _select(first, fsq0, carry.res1)
        record_low = (fsq_prev <= res0_f) & (fsq0 <= res1_f)
        res0_n = jnp.minimum(res0_f, fsq_prev)
        res1_n = jnp.minimum(res1_f, fsq0)
        growth_gate = ~(record_low & (~jac1))     # IF/ELSE-IF chain in Fortran
        grew = (
            growth_gate
            & ((it - carry.iter1) > GROWTH_MIN_ITERATIONS)
            & ((fsq_prev > GROWTH_LIMIT * res0_n) | (fsq0 > GROWTH_LIMIT * res1_n))
        )
        kind = jnp.where(grew, RESTART_GROWTH,
                         jnp.where(jac1, RESTART_JACOBIAN, STEP_OK))
        restart = stepping & (kind != STEP_OK)
        store = stepping & (first | (record_low & (~jac1)))

        xstore_n = _select(store, carry.state, carry.xstore)
        state_r = _select(restart, xstore_n, carry.state)
        xcdot_r = _select(restart, jax.tree.map(jnp.zeros_like, carry.xcdot), carry.xcdot)
        delt_r = carry.time_step * jnp.where(
            restart & (kind == RESTART_JACOBIAN), JACOBIAN_RESET_FACTOR, 1.0
        ) * jnp.where(
            restart & (kind == RESTART_GROWTH), 1.0 / GROWTH_BACKOFF_DIVISOR, 1.0
        )
        ijacob_r = carry.ijacob + (restart & (kind == RESTART_JACOBIAN)).astype(carry.ijacob.dtype)
        iter1_r = jnp.where(restart, it, carry.iter1)

        # Re-evaluate at the restored state (TimeStepControl calls funct3d).
        e2 = lax.cond(
            restart,
            lambda args: _evaluate(args[0], args[1], it, it, args[2], rt, args[3]),
            lambda args: e1,
            (state_r, e1.cache, fsqz_c, fsqr_c + fsqz_c),
        )
        reeval_bad = restart & e2.jacobian_sign_changed

        fsqr_f = _select(restart, e2.residuals.fsqr, fsqr_c)
        fsqz_f = _select(restart, e2.residuals.fsqz, fsqz_c)
        fsql_f = _select(restart, e2.residuals.fsql, fsql_c)
        fsqr1_f = jnp.where(restart, e2.pre.fsqr1, e1.pre.fsqr1)
        fsqz1_f = jnp.where(restart, e2.pre.fsqz1, e1.pre.fsqz1)
        fsql1_f = jnp.where(restart, e2.pre.fsql1, e1.pre.fsql1)
        wb_f = jnp.where(restart, e2.wb, e1.wb)
        wp_f = jnp.where(restart, e2.wp, e1.wp)
        r00_f = jnp.where(restart, e2.r00, e1.r00)
        z00_f = jnp.where(restart, e2.z00, e1.z00)
        cache_f = _select(restart, e2.cache, e1.cache)
        gc_f = _select(restart, e2.gc, e1.gc)

        # ---- damping + momentum step (evolve.f) ----------------------------
        fsq1 = fsqr1_f + fsqz1_f + fsql1_f
        control = StepControl(
            time_step=delt_r, inv_tau=carry.inv_tau, fsq_total_prev=fsq_prev,
            residual_best_precond=jnp.asarray(jnp.inf),
            residual_best_raw=jnp.asarray(jnp.inf),
            iter_last_reset=iter1_r, jacobian_resets=ijacob_r,
        )
        b1, fac, control2 = damping_coefficients(control, it, fsq1)
        state_n, xcdot_n = momentum_update(state_r, xcdot_r, gc_f, b1, fac, delt_r)

        # ---- 2D block preconditioner: matrix-free Newton step (precon2d.f) --
        # Only traced when enabled; otherwise the momentum step above stands
        # and the default 1D path is byte-identical.  When active, take a
        # damped Newton step (state += cfg.step * (-J^{-1} gc)) with zeroed
        # velocity, mirroring evolve.f's xcdot reset on prec2d activation.
        newton_step = jnp.asarray(-1.0, dtype=fsqr_f.dtype)
        newton_linear_residual = jnp.asarray(jnp.nan, dtype=fsqr_f.dtype)
        newton_lambda_scale = jnp.asarray(jnp.nan, dtype=fsqr_f.dtype)
        if rt.prec2d is not None:
            fsqz_prev_used = _select(restart, fsqz_c, carry.fsqz)
            (newton_dir, prec2d_active, newton_step,
             newton_linear_residual, newton_lambda_scale) = _newton_step(
                rt, state_r, gc_f, cache_f, it, fsqz_prev_used,
                fsqr_f + fsqz_f + fsql_f, stepping & (~reeval_bad),
            )
            state_newton = jax.tree.map(
                lambda x, d: x + rt.prec2d.step * d, state_r, newton_dir
            )
            xcdot_newton = jax.tree.map(jnp.zeros_like, xcdot_r)
            state_n = _select(prec2d_active, state_newton, state_n)
            xcdot_n = _select(prec2d_active, xcdot_newton, xcdot_n)

        # ---- eqsolve.f escalation ------------------------------------------
        eq_reset = stepping & ((ijacob_r == 25) | (ijacob_r == 50))
        state_out = _select(eq_reset, xstore_n, _select(stepping, state_n, carry.state))
        xcdot_out = _select(
            eq_reset, jax.tree.map(jnp.zeros_like, carry.xcdot),
            _select(stepping, xcdot_n, carry.xcdot),
        )
        delt_n = delt_r * jnp.where(eq_reset, JACOBIAN_RESET_FACTOR, 1.0)
        ijacob_n = ijacob_r + eq_reset.astype(ijacob_r.dtype)
        iter1_n = _select(eq_reset, it, iter1_r)

        jac75 = stepping & (ijacob_n >= 75)
        maxed = stepping & (~eq_reset) & (~jac75) & (it >= max_iter)

        stop_now = running & (converged | bad_init) | jac75 | maxed | reeval_bad
        done_n = carry.done | stop_now
        ier_n = jnp.where(
            carry.done, carry.ier,
            jnp.where(running & converged, SUCCESSFUL_TERM_FLAG,
            jnp.where(running & bad_init, BAD_JACOBIAN_FLAG,
            jnp.where(jac75, JAC75_FLAG,
            jnp.where(reeval_bad, MISC_ERROR_FLAG,
            jnp.where(maxed, MORE_ITER_FLAG, carry.ier))))),
        ).astype(carry.ier.dtype)

        advance = stepping & (~eq_reset) & (~jac75) & (~maxed) & (~reeval_bad)
        iteration_n = jnp.where(advance, it + 1, it)

        # ---- trajectory row (printout.f values) ----------------------------
        w0 = wb_f + wp_f / (gamma - 1.0)
        row = jnp.stack([
            it.astype(wb_f.dtype), fsqr_f, fsqz_f, fsql_f,
            fsqr1_f, fsqz1_f, fsql1_f, r00_f, z00_f, w0 * _TWO_PI_SQ, delt_r,
            newton_step, newton_linear_residual, newton_lambda_scale,
        ])
        idx = jnp.clip(it - 1, 0, max_iter - 1)
        old_row = lax.dynamic_slice_in_dim(carry.trajectory, idx, 1, axis=0)[0]
        new_row = jnp.where(running, row, old_row)
        trajectory_n = lax.dynamic_update_slice_in_dim(carry.trajectory, new_row[None, :], idx, axis=0)

        gate = lambda mask, new, old: jnp.where(mask, new, old)  # noqa: E731
        return _LoopCarry(
            state=state_out, xcdot=xcdot_out,
            xstore=_select(stepping, xstore_n, carry.xstore),
            cache=_select(running, cache_f, carry.cache),
            time_step=gate(stepping, delt_n, carry.time_step),
            inv_tau=gate(stepping, control2.inv_tau, carry.inv_tau),
            fsq=gate(stepping, control2.fsq_total_prev, carry.fsq),
            res0=gate(stepping, res0_n, carry.res0),
            res1=gate(stepping, res1_n, carry.res1),
            fsqr=gate(running, fsqr_f, carry.fsqr),
            fsqz=gate(running, fsqz_f, carry.fsqz),
            fsql=gate(running, fsql_f, carry.fsql),
            fsqr1=gate(running, fsqr1_f, carry.fsqr1),
            fsqz1=gate(running, fsqz1_f, carry.fsqz1),
            fsql1=gate(running, fsql1_f, carry.fsql1),
            wb=gate(running, wb_f, carry.wb), wp=gate(running, wp_f, carry.wp),
            r00=gate(running, r00_f, carry.r00),
            iteration=iteration_n, iter1=gate(stepping, iter1_n, carry.iter1),
            ijacob=gate(stepping, ijacob_n, carry.ijacob),
            done=done_n, ier=ier_n, trajectory=trajectory_n,
        )

    return body


def _initial_carry(state: SpectralState, rt: SolverRuntime, *, ijacob: int) -> _LoopCarry:
    """Initial loop carry (reset_params.f / initialize_radial.f values)."""
    dtype = rt.setup.s_full.dtype
    one = jnp.asarray(1.0, dtype=dtype)
    zeros = jax.tree.map(jnp.zeros_like, state)
    delt0 = jnp.asarray(rt.time_step0, dtype=dtype)
    zero, inf = jnp.zeros((), dtype=dtype), jnp.asarray(jnp.inf, dtype=dtype)
    # NOTE: scalar counters/flags carry explicit (non-weak) dtypes so that the
    # initial carry has exactly the avals of the carry the jitted lanes
    # return; weak-typed Python scalars here would force a second lane
    # compilation on the first block round-trip.
    int_ = lambda v: jnp.asarray(v, dtype=jnp.int64)  # noqa: E731
    return _LoopCarry(
        state=state, xcdot=zeros, xstore=state, cache=_zero_cache(rt),
        time_step=delt0,
        inv_tau=jnp.full((NDAMP,), DAMPING_CAP, dtype=dtype) / delt0,
        fsq=one, res0=inf, res1=inf,
        fsqr=one, fsqz=one, fsql=one, fsqr1=one, fsqz1=one, fsql1=one,
        wb=zero, wp=zero, r00=zero,
        iteration=int_(1), iter1=int_(1),
        ijacob=int_(int(ijacob)),
        done=jnp.zeros((), dtype=bool), ier=int_(NORM_TERM_FLAG),
        trajectory=jnp.zeros((rt.max_iterations, _TRAJ_COLS), dtype=dtype),
    )


# -- Result container and lanes --------------------------------------------------------------------------------------


from .solver_driver import (
    SolveResult,
    _finalize as _finalize,
    _result_from_carry as _result_from_carry,
    _solve_stage as _solve_stage,
    solve,
)
