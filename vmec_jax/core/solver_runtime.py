"""Spectral state containers and immutable single-grid runtime setup.

This module owns the data model shared by fixed boundary, free boundary,
multigrid, diagnostics, and implicit differentiation.  It contains no
nonlinear iteration loop; :mod:`vmec_jax.core.solver` consumes these objects
to evaluate forces and advance the equilibrium.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

import jax.numpy as jnp

from .forces import constraint_force
from .fourier import ModeTable, Resolution, TrigTables, mode_table, trig_tables
from .geometry import apply_lambda_axis_closure, real_space_geometry
from .input import VmecInput
from .preconditioner import (
    RadialPreconditionerCoefficients,
    TridiagonalMatrices,
    angular_integration_weights,
)
from .preconditioner_2d import Prec2DConfig
from .residuals import ForceResiduals, PreconditionedResiduals, m1_constrained_to_physical
from .setup import RunSetup, run_setup
from .transforms import (
    SpectralForce,
    register_pytree_dataclass as _register,
)

Array = Any

__all__ = [
    "FunctDiagnostics",
    "PreconditionerCache",
    "SolverRuntime",
    "SpectralState",
    "hot_restart_state",
    "prepare_runtime",
    "resolution_from_input",
    "runtime_with_baselines",
]

@dataclass(frozen=True)
class SpectralState:
    """Evolved spectral state in the signed-(m, n) internal packing.

    VMEC2000 ``xc``: internal normalization (``mscale*nscale`` divided out),
    m = 1-*constrained* basis (``residue.f90`` ``lconm1``), odd-m WITHOUT the
    ``scalxc`` factor (``totzsps`` applies it during synthesis).  For
    stellarator-symmetric runs ``R_sin/Z_cos/L_cos`` are identically zero.
    All arrays have shape ``(ns, mnmax)``.
    """

    R_cos: Array
    R_sin: Array
    Z_cos: Array
    Z_sin: Array
    L_cos: Array
    L_sin: Array


@dataclass(frozen=True)
class PreconditionerCache:
    """Quantities refreshed on the ``ns4 = 25`` cadence (VMEC2000 ``bcovar.f``).

    ``tcon`` (constraint scaling), the residual norms ``fnorm/fnormL/fnorm1``,
    the :func:`precondn` coefficients and assembled :func:`scalfor_matrices`
    for the R and Z force families, and the ``lamcal`` diagonal ``faclam``.
    """

    tcon: Array
    fnorm: Array
    fnormL: Array
    fnorm1: Array
    coefficients_R: RadialPreconditionerCoefficients
    coefficients_Z: RadialPreconditionerCoefficients
    matrices_R: TridiagonalMatrices
    matrices_Z: TridiagonalMatrices
    faclam: Array


@dataclass(frozen=True)
class FunctDiagnostics:
    """Per-evaluation diagnostics returned by :func:`evaluate_forces`.

    ``preconditioned`` are the residue.f90 ``fsqr1/fsqz1/fsql1``; ``wb/wp``
    the MHD energies (``bcovar.f``); ``r00/z00`` the axis position at
    ``theta = zeta = 0`` (``funct3d.f``); ``jacobian_sign_changed`` the
    ``jacobian.f`` ``irst = 2`` flag; ``cache`` the (possibly refreshed)
    :class:`PreconditionerCache` to carry into the next iteration.
    """

    preconditioned: PreconditionedResiduals
    wb: Array
    wp: Array
    r00: Array
    z00: Array
    jacobian_sign_changed: Array
    cache: PreconditionerCache


@dataclass(frozen=True)
class _EvalResult:
    """Internal bundle: force + residuals + diagnostics for the loop body."""

    gc: SpectralState
    residuals: ForceResiduals
    pre: PreconditionedResiduals
    wb: Array
    wp: Array
    r00: Array
    z00: Array
    jacobian_sign_changed: Array
    cache: PreconditionerCache


@dataclass(frozen=True)
class _LoopCarry:
    """Traced carry of the eqsolve iteration loop (one per solve).

    VMEC2000 names: ``time_step = delt``, ``inv_tau = otau`` (the ndamp
    damping window), ``fsq`` (previous ``fsqr1+fsqz1+fsql1``), ``res0/res1``
    (best preconditioned/raw totals, ``TimeStepControl``), ``iteration =
    iter2``, ``iter1`` (last restart), ``ijacob``; ``trajectory`` is the
    ``(max_iterations, _TRAJ_COLS)`` per-iteration history buffer.
    """

    state: SpectralState; xcdot: SpectralState; xstore: SpectralState
    cache: PreconditionerCache
    time_step: Array; inv_tau: Array; fsq: Array; res0: Array; res1: Array
    fsqr: Array; fsqz: Array; fsql: Array
    fsqr1: Array; fsqz1: Array; fsql1: Array
    wb: Array; wp: Array; r00: Array
    iteration: Array; iter1: Array; ijacob: Array
    done: Array; ier: Array
    trajectory: Array


for _cls in (SpectralState, PreconditionerCache, FunctDiagnostics, _EvalResult,
             _LoopCarry):
    _register(_cls)


# -- Runtime (per-solve context: a jit-passable pytree) ---------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _static_tables(resolution: Resolution):
    """Cached trace-time-static tables derived from a :class:`Resolution`.

    Returns ``(modes, trig, weights, gather_m, gather_n, cos_w, sin_w)`` —
    all host NumPy objects consumed *at trace time* (fancy-index tables,
    trig-table matmuls inside :mod:`vmec_jax.core.transforms`, angular
    weights).  The lru_cache guarantees that two runtimes built from the same
    ``Resolution`` share the *identical* table objects, so the runtime pytree
    treedefs compare equal and ``jax.jit`` reuses one executable across
    solves with different boundary/profile values (plan.md Phase 2 item (1)).
    """
    modes = mode_table(resolution.mpol, resolution.ntor)
    trig = trig_tables(resolution)
    weights = angular_integration_weights(
        ntheta=resolution.ntheta, nzeta=resolution.nzeta, lasym=resolution.lasym
    )
    gather_m, gather_n, cos_w, sin_w = _force_gather_tables(modes)
    return modes, trig, weights, gather_m, gather_n, cos_w, sin_w


@dataclass(frozen=True, eq=False)
class SolverRuntime:
    """Per-solve context, registered as a JAX pytree.

    Passed *as an argument* to the module-level jitted lanes
    (:func:`_while_lane` / :func:`_block_lane`):

    - **data fields** (traced): the :class:`RunSetup` arrays (profiles,
      radial grids, boundary, initial state — everything that changes with
      boundary values) and the fixed-boundary constraint baselines
      ``rcon0/zcon0`` (``funct3d.f`` — constant per run because the edge
      spectral row never evolves, but boundary-value dependent);
    - **meta fields** (static, hashable): the :class:`Resolution` plus the
      scalar configuration (``gamma`` is consumed concretely by
      ``fields.magnetic_fields``; ``max_iterations`` sizes the trajectory
      buffer; the rest are loop-control constants).

    The NumPy mode/trig/weight/gather tables are *derived* from the meta
    ``resolution`` via the cached :func:`_static_tables` (exposed as
    properties), so they never enter the pytree: they are used with
    ``np.*``/fancy indexing at trace time and must stay concrete.  Two
    runtimes with equal structure (same resolution + scalars, same array
    shapes) therefore share one XLA executable per lane.
    """

    resolution: Resolution
    setup: RunSetup
    rcon0: Array; zcon0: Array
    gamma: float; tcon0: float; ftol: float
    max_iterations: int; time_step0: float; nstep: int
    jmax: int                           # evolved radial rows (fixed: ns-1)

    # -- free-boundary seam (core/freeboundary.py; funct3d.f/forces.f) ------
    # lfreeb=True selects the vacuum-coupled lane: the edge row is evolved
    # (jmax = ns is passed by the free-boundary driver), tomnsps keeps the
    # edge row, and forces.f's `armn(ns) += zu0*rbsq` / `azmn(ns) -= ru0*rbsq`
    # edge terms are injected with rbsq = (bsqvac + presf_ns)*R(edge)*ohs.
    # `bsqvac_edge` is the NESTOR 0.5*|B|^2 on the (ntheta3, nzeta) boundary
    # grid, refreshed by the host driver between iterations (data field: no
    # retrace).  `presf_ns_scale` is the static funct3d.f edge-pressure
    # factor pmass(1)/pmass(hs*(ns-1.5)) applied to pres(ns).
    lfreeb: bool = False
    bsqvac_edge: Array | None = None
    presf_ns_scale: Array | None = None

    # -- 2D block preconditioner seam (core/preconditioner_2d.py; precon2d.f) -
    # None disables it (the default 1D-only path is then byte-identical); a
    # Prec2DConfig switches on the matrix-free Newton step (finest grid, once
    # fsqr+fsqz+fsql < threshold).  Static meta: the branch is only traced when
    # present, so a NONE run never compiles the GMRES/HVP graph.
    prec2d: Any = None

    # -- trace-time-static tables, derived from the meta resolution ---------
    @property
    def modes(self) -> ModeTable:
        """Return the cached Fourier mode table for this resolution."""
        return _static_tables(self.resolution)[0]

    @property
    def trig(self) -> TrigTables:
        """Return the cached trigonometric transform tables."""
        return _static_tables(self.resolution)[1]

    @property
    def weights(self) -> np.ndarray:    # angular integration weights (wint)
        """Return the angular quadrature weights (VMEC ``wint``)."""
        return _static_tables(self.resolution)[2]

    # force-block gather tables (static, from _force_gather_tables):
    # cos_w weights the (cc, ss) blocks; sin_w the (sc, cs) blocks.
    @property
    def gather_m(self) -> np.ndarray:
        """Return poloidal indices for gathering signed force modes."""
        return _static_tables(self.resolution)[3]

    @property
    def gather_n(self) -> np.ndarray:
        """Return toroidal indices for gathering signed force modes."""
        return _static_tables(self.resolution)[4]

    @property
    def cos_w(self) -> np.ndarray:
        """Return weights for gathering cosine-family force blocks."""
        return _static_tables(self.resolution)[5]

    @property
    def sin_w(self) -> np.ndarray:
        """Return weights for gathering sine-family force blocks."""
        return _static_tables(self.resolution)[6]


_register(SolverRuntime, meta=(
    "resolution", "gamma", "tcon0", "ftol", "max_iterations", "time_step0",
    "nstep", "jmax", "lfreeb", "prec2d",
))


def _force_gather_tables(modes: ModeTable) -> tuple[np.ndarray, ...]:
    """Static tables mapping VMEC ``(m, n>=0)`` force blocks to signed modes.

    Inverse of the signed -> block relations used throughout VMEC (see the
    parity-proven ``vmec_jax.kernels.parity`` maps): for the cos family
    (``cc/ss`` blocks) and the sin family (``sc/cs`` blocks),

    - ``n = 0``:            signed = A                      -> (1, 0)
    - ``n > 0, m > 0``:     cos: (cc + ss)/2 -> (1/2, 1/2); sin: (sc - cs)/2
    - ``n > 0, m = 0``:     cos: cc -> (1, 0);              sin: -cs -> (0, -1)
    - ``n < 0`` (m > 0):    cos: (cc - ss)/2;               sin: (sc + cs)/2
    """
    m = np.asarray(modes.m, dtype=int)
    n = np.asarray(modes.n, dtype=int)
    cos_w = np.zeros((m.size, 2))
    sin_w = np.zeros((m.size, 2))
    for k in range(m.size):
        if n[k] == 0:
            cos_w[k] = (1.0, 0.0); sin_w[k] = (1.0, 0.0)
        elif n[k] > 0 and m[k] > 0:
            cos_w[k] = (0.5, 0.5); sin_w[k] = (0.5, -0.5)
        elif n[k] > 0:  # m == 0
            cos_w[k] = (1.0, 0.0); sin_w[k] = (0.0, -1.0)
        else:           # n < 0 (m > 0 only in the VMEC mode table)
            cos_w[k] = (0.5, -0.5); sin_w[k] = (0.5, 0.5)
    return m.astype(np.int32), np.abs(n).astype(np.int32), cos_w, sin_w


def _blocks_to_signed(block_a, block_b, rt: SolverRuntime, w: np.ndarray) -> Array:
    """Gather one ``(ns, mpol, ntor+1)`` block pair into signed coefficients."""
    ns = int(rt.setup.s_full.shape[0])
    dtype = rt.setup.s_full.dtype
    if block_a is None:
        return jnp.zeros((ns, rt.modes.mnmax), dtype=dtype)
    a = jnp.asarray(block_a)[:, rt.gather_m, rt.gather_n]
    out = jnp.asarray(w[:, 0], dtype=a.dtype)[None, :] * a
    if block_b is not None:
        b = jnp.asarray(block_b)[:, rt.gather_m, rt.gather_n]
        out = out + jnp.asarray(w[:, 1], dtype=b.dtype)[None, :] * b
    return out


def _force_to_state(force: SpectralForce, rt: SolverRuntime) -> SpectralState:
    """Preconditioned :class:`SpectralForce` -> signed increments (``gc``).

    VMEC evolves ``xc`` in the same block layout as ``gc``; the signed
    packing is an equivalent linear reparametrization, so the momentum step
    commutes with this conversion.
    """
    return SpectralState(
        R_cos=_blocks_to_signed(force.force_R_cc, force.force_R_ss, rt, rt.cos_w),
        R_sin=_blocks_to_signed(force.force_R_sc, force.force_R_cs, rt, rt.sin_w),
        Z_cos=_blocks_to_signed(force.force_Z_cc, force.force_Z_ss, rt, rt.cos_w),
        Z_sin=_blocks_to_signed(force.force_Z_sc, force.force_Z_cs, rt, rt.sin_w),
        L_cos=_blocks_to_signed(force.force_lambda_cc, force.force_lambda_ss, rt, rt.cos_w),
        L_sin=_blocks_to_signed(force.force_lambda_sc, force.force_lambda_cs, rt, rt.sin_w),
    )


# -- Runtime construction --------------------------------------------------------------------------------------------


def resolution_from_input(inp: VmecInput, *, ns: int | None = None) -> Resolution:
    """Resolve the internal grid sizes from an input (``read_indata.f``).

    ``ntheta <= 0 -> 2*mpol + 6``; ``nzeta = 1`` for axisymmetric inputs with
    ``nzeta = 0``, else ``nzeta <= 0 -> 2*ntor + 4``.  ``ns`` defaults to the
    first ``ns_array`` stage (single-grid solve).
    """
    ntheta = int(inp.ntheta) if int(inp.ntheta) > 0 else 2 * int(inp.mpol) + 6
    nzeta = int(inp.nzeta)
    if int(inp.ntor) == 0 and nzeta == 0:
        nzeta = 1
    elif nzeta <= 0:
        nzeta = 2 * int(inp.ntor) + 4
    return Resolution(
        mpol=int(inp.mpol), ntor=int(inp.ntor), ntheta=ntheta, nzeta=nzeta,
        nfp=int(inp.nfp), lasym=bool(inp.lasym), ns=int(inp.ns_array[0]) if ns is None else int(ns),
    )


def _initial_state(setup: RunSetup) -> SpectralState:
    return SpectralState(
        R_cos=setup.R_cos, R_sin=setup.R_sin, Z_cos=setup.Z_cos,
        Z_sin=setup.Z_sin, L_cos=setup.lambda_cos, L_sin=setup.lambda_sin,
    )


def _physical_coefficients(state: SpectralState, *, modes, lthreed, lasym, lconm1):
    """Undo the m=1 constraint; return the geometry-synthesis coefficients."""
    R_cos, Z_sin, R_sin, Z_cos = m1_constrained_to_physical(
        state.R_cos, state.Z_sin, state.R_sin, state.Z_cos,
        modes=modes, lthreed=lthreed, lasym=lasym, lconm1=lconm1,
    )
    return R_cos, R_sin, Z_cos, Z_sin


def _geometry(state: SpectralState, rt: SolverRuntime):
    """Constrained state -> physical coefficients + real-space geometry."""
    setup = rt.setup
    R_cos, R_sin, Z_cos, Z_sin = _physical_coefficients(
        state, modes=rt.modes, lthreed=setup.lthreed, lasym=setup.lasym,
        lconm1=setup.lconm1,
    )
    lambda_sin = apply_lambda_axis_closure(state.L_sin, modes=rt.modes, ntor=rt.resolution.ntor)
    geometry = real_space_geometry(
        R_cos=R_cos, R_sin=R_sin, Z_cos=Z_cos, Z_sin=Z_sin,
        lambda_cos=state.L_cos, lambda_sin=lambda_sin,
        modes=rt.modes, trig=rt.trig, s=setup.s_full,
    )
    return (R_cos, R_sin, Z_cos, Z_sin), geometry


def _resolve_prec2d(
    source: VmecInput | RunSetup,
    prec2d: Prec2DConfig | None,
    precon_type: str | None,
    prec2d_threshold: float | None,
    *,
    finest: bool = True,
) -> Prec2DConfig | None:
    """Resolve the 2D-preconditioner config (``precon2d.f`` / ``evolve.f``).

    An explicit ``prec2d`` config wins (used verbatim); otherwise it is built
    from ``precon_type``/``prec2d_threshold`` (input defaults, overridable) when
    ``precon_type != "NONE"``.  Returns ``None`` (the default 1D-only path)
    when the 2D preconditioner is off.
    """
    if prec2d is not None:
        return prec2d
    if isinstance(source, VmecInput):
        pt = source.precon_type if precon_type is None else precon_type
        thr = source.prec2d_threshold if prec2d_threshold is None else prec2d_threshold
    else:
        pt = "NONE" if precon_type is None else precon_type
        thr = 1e-30 if prec2d_threshold is None else prec2d_threshold
    if str(pt).strip().upper() == "NONE":
        return None
    return Prec2DConfig(threshold=float(thr), finest=bool(finest))


def prepare_runtime(
    source: VmecInput | RunSetup,
    resolution: Resolution | None = None,
    *,
    ftol: float | None = None, max_iterations: int | None = None,
    time_step: float | None = None, tcon0: float | None = None,
    gamma: float | None = None, nstep: int | None = None,
    lconm1: bool = True, setup: RunSetup | None = None,
    precon_type: str | None = None, prec2d_threshold: float | None = None,
    prec2d: Prec2DConfig | None = None,
) -> SolverRuntime:
    """Build the static solver context from an input file or a RunSetup.

    Defaults come from the input (``delt``, ``tcon0``, ``gamma``, ``nstep``,
    ``ftol_array(1)``, ``niter_array(1)``); explicit keywords override.  The
    fixed-boundary constraint baselines ``rcon0/zcon0 = s * rcon(ns)``
    (``funct3d.f``) are computed once here from the initial state — the edge
    spectral row never evolves in fixed-boundary mode, so they are constants
    of the run.

    ``precon_type``/``prec2d_threshold`` (or an explicit ``prec2d``
    :class:`~vmec_jax.core.preconditioner_2d.Prec2DConfig`) switch on the
    optional 2D block preconditioner (``precon2d.f``); ``None``/``"NONE"``
    (the default) leaves the 1D-only path byte-identical.
    """
    if isinstance(source, RunSetup):
        if resolution is None:
            raise ValueError("prepare_runtime(RunSetup) requires a Resolution")
        setup = source
        defaults = dict(ftol=1e-10, niter=100, delt=1.0, tcon0=1.0, gamma=0.0,
                        nstep=200)
    else:
        inp = source
        if resolution is None:
            resolution = resolution_from_input(inp)
        if setup is None:
            setup = run_setup(inp, resolution, lconm1=lconm1)
        defaults = dict(ftol=float(inp.ftol_array[0]), niter=int(inp.niter_array[0]),
                        delt=float(inp.delt), tcon0=float(inp.tcon0),
                        gamma=float(inp.gamma), nstep=int(inp.nstep))

    rt = SolverRuntime(
        resolution=resolution, setup=setup,
        gamma=float(defaults["gamma"] if gamma is None else gamma),
        tcon0=float(defaults["tcon0"] if tcon0 is None else tcon0),
        ftol=float(defaults["ftol"] if ftol is None else ftol),
        max_iterations=int(defaults["niter"] if max_iterations is None else max_iterations),
        time_step0=float(defaults["delt"] if time_step is None else time_step),
        nstep=int(defaults["nstep"] if nstep is None else nstep),
        jmax=int(resolution.ns) - 1,
        rcon0=jnp.zeros(()), zcon0=jnp.zeros(()),  # placeholder, replaced below
        prec2d=_resolve_prec2d(source, prec2d, precon_type, prec2d_threshold),
    )
    rcon0, zcon0 = _constraint_baselines(_initial_state(setup), rt)
    return replace(rt, rcon0=rcon0, zcon0=zcon0)


def hot_restart_state(rt: SolverRuntime, state: SpectralState) -> SpectralState:
    """Adapt a previous solve's state to this runtime's boundary (hot restart).

    In fixed-boundary mode the R/Z edge row never evolves, so restarting from
    ``state`` unchanged would silently re-solve the OLD boundary.  Replacing
    only the edge row injects a discontinuous shear between the last two
    surfaces (measured: initial ``fsqr ~ 0.5`` on cth, i.e. *worse* than the
    fresh interior guess).  Instead the boundary delta is spread smoothly
    into the volume with the ``profil3d.f`` interior-guess radial profile —
    ``sqrts**m`` for ``m > 0``, linear in ``s`` for ``m = 0`` (the axis is
    held fixed) — which lands the edge row exactly on the new boundary
    (``facj(ns) = 1``) and keeps the interior near equilibrium (measured:
    initial ``fsqr ~ 4e-6`` for a 1% ``RBC(0,1)`` perturbation on cth).
    Lambda is carried over unchanged.
    """
    setup = rt.setup
    s = jnp.asarray(setup.s_full)
    dtype = s.dtype
    m = np.asarray(rt.modes.m, dtype=int)
    rho = jnp.sqrt(jnp.maximum(s, 0.0)).at[-1].set(jnp.asarray(1.0, dtype=dtype))
    m_j = jnp.asarray(m)[None, :]
    facj = jnp.where(m_j > 0, rho[:, None] ** m_j,
                     s[:, None] * jnp.ones((1, m.size), dtype=dtype))

    def shift(old, new_boundary):
        old = jnp.asarray(old, dtype=dtype)
        delta = jnp.asarray(new_boundary, dtype=dtype)[-1] - old[-1]
        return old + facj * delta[None, :]

    return replace(
        state,
        R_cos=shift(state.R_cos, setup.R_cos),
        R_sin=shift(state.R_sin, setup.R_sin),
        Z_cos=shift(state.Z_cos, setup.Z_cos),
        Z_sin=shift(state.Z_sin, setup.Z_sin),
    )


def runtime_with_baselines(rt: SolverRuntime, state: SpectralState) -> SolverRuntime:
    """Rebind ``rcon0/zcon0`` to a new starting state (``funct3d.f``).

    VMEC2000 sets the constraint baselines from the *current* state whenever
    ``iter2 == iter1`` — i.e. at the start of every grid.  A runtime from
    :func:`prepare_runtime` carries baselines for the ``profil3d.f`` interior
    guess; callers that start from a different state (hot restart via
    ``solve(initial_state=...)``, multigrid stages starting from the
    ``interp.f`` interpolant) must rebind them, or the constraint force —
    and hence the converged equilibrium — is subtly wrong (observed as a
    ~1e-8 relative ``wb`` shift on the nfp4_QH ladder).
    """
    rcon0, zcon0 = _constraint_baselines(state, rt)
    return replace(rt, rcon0=rcon0, zcon0=zcon0)


def _constraint_baselines(state: SpectralState, rt: SolverRuntime):
    """One-time ``rcon0/zcon0 = s * rcon(ns)`` (funct3d.f, iter2 == iter1)."""
    (R_cos, R_sin, Z_cos, Z_sin), geometry = _geometry(state, rt)
    ns = int(rt.setup.s_full.shape[0])
    _, _, _, rcon0, zcon0 = constraint_force(
        R_cos=R_cos, R_sin=R_sin, Z_cos=Z_cos, Z_sin=Z_sin,
        geometry=geometry, modes=rt.modes, trig=rt.trig, s=rt.setup.s_full,
        tcon=jnp.zeros((ns,), dtype=rt.setup.s_full.dtype),
        signgs=rt.setup.signgs,
    )
    return rcon0, zcon0


def _zero_cache(rt: SolverRuntime) -> PreconditionerCache:
    """Zero-filled cache (shapes only; iteration 1 always refreshes it)."""
    res = rt.resolution
    ns, mpol, nr = res.ns, res.mpol, res.ntor + 1
    dtype = rt.setup.s_full.dtype
    z = lambda shape: jnp.zeros(shape, dtype=dtype)  # noqa: E731
    coeffs = RadialPreconditionerCoefficients(
        axm=z((ns - 1, 2)), axd=z((ns, 2)), bxm=z((ns - 1, 2)), bxd=z((ns, 2)),
        cx=z((ns,)),
    )
    mats = TridiagonalMatrices(ax=z((rt.jmax, mpol, nr)), bx=z((rt.jmax, mpol, nr)),
                               dx=z((rt.jmax, mpol, nr)))
    return PreconditionerCache(
        tcon=z((ns,)), fnorm=z(()), fnormL=z(()), fnorm1=z(()),
        coefficients_R=coeffs, coefficients_Z=coeffs,
        matrices_R=mats, matrices_Z=mats, faclam=z((ns, mpol, nr)),
    )

