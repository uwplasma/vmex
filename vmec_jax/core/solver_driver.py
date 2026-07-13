"""Host iteration lanes, result assembly, and fixed-boundary solve entry point.

The traced force and update kernels live in :mod:`vmec_jax.core.solver`; this
module owns host orchestration, CLI printing cadence, typed termination, axis
retry, and conversion of the final carry into WOUT-convention arrays.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax

from .device import device_context
from .errors import (
    BAD_JACOBIAN_FLAG,
    JAC75_FLAG,
    MORE_ITER_FLAG,
    SUCCESSFUL_TERM_FLAG,
    VmecConvergenceError,
    VmecJacobianError,
    WERROR_MESSAGES,
)
from .fields import magnetic_fields, metric_elements
from .fourier import Resolution
from .geometry import half_mesh_jacobian
from .input import VmecInput
from .preconditioner_2d import Prec2DConfig
from .printing import FORCE_ITERATIONS_BANNER, screen_header, screen_line, stage_banner
from .setup import RunSetup, guess_axis, interior_guess
from .solver_runtime import (
    SolverRuntime,
    SpectralState,
    _geometry,
    _initial_state,
    _LoopCarry,
    _physical_coefficients,
    hot_restart_state,
    prepare_runtime,
    runtime_with_baselines,
)
from .transforms import physical_to_internal_scale

# Imported after solver defines these kernels; solver late-binds this module.
from .solver import (
    BLOCK_SIZE,
    _initial_carry,
    _make_body,
    _TWO_PI_SQ,
)

__all__ = ["SolveResult", "solve"]

@dataclass(frozen=True)
class SolveResult:
    """Converged fixed-boundary solve output.

    ``rmnc/zmns`` (+ ``rmns/zmnc`` when ``lasym``) are physical (wout
    convention) spectral coefficients on the full mesh, mode-ordered like the
    wout ``xm/xn`` arrays; ``iotaf`` follows ``add_fluxes.f90`` for
    ``ncurr = 1``.  ``fsq_history`` has one row per iteration:
    ``(fsqr, fsqz, fsql, fsqr1, fsqz1, fsql1)``.  ``wmhd`` is the printed
    ``WMHD = (wb + wp/(gamma-1)) * (2 pi)^2``. ``newton_history`` stores
    ``(accepted_step, linear_residual, lambda_row_scale)``; step ``-1`` means
    no attempt and ``0`` means a rejected correction followed by the regular
    VMEC update. Free-boundary solves additionally retain their final NESTOR
    cadence/cache object in ``vacuum_state``; fixed-boundary solves leave it
    ``None``.
    """

    converged: bool; iterations: int; ier_flag: int
    fsqr: float; fsqz: float; fsql: float
    wb: float; wp: float; wmhd: float; r00: float
    time_step: float; jacobian_resets: int
    state: SpectralState
    xm: np.ndarray; xn: np.ndarray
    rmnc: np.ndarray; zmns: np.ndarray
    rmns: np.ndarray | None; zmnc: np.ndarray | None
    iotaf: np.ndarray; fsq_history: np.ndarray; newton_history: np.ndarray
    vacuum_state: Any | None = None


def _result_from_carry(carry: _LoopCarry, rt: SolverRuntime) -> SolveResult:
    """Host-side result assembly (wout-convention outputs)."""
    setup = rt.setup
    state = carry.state
    R_cos, R_sin, Z_cos, Z_sin = _physical_coefficients(
        carry.state, modes=rt.modes, lthreed=setup.lthreed, lasym=setup.lasym,
        lconm1=setup.lconm1,
    )
    scale = 1.0 / physical_to_internal_scale(rt.modes, rt.trig)
    rmnc = np.asarray(R_cos) * scale[None, :]
    zmns = np.asarray(Z_sin) * scale[None, :]
    rmns = np.asarray(R_sin) * scale[None, :] if setup.lasym else None
    zmnc = np.asarray(Z_cos) * scale[None, :] if setup.lasym else None

    # iotaf (add_fluxes.f90): prescribed profile for ncurr = 0; reconstructed
    # from the converged current-constrained chips for ncurr = 1.
    if int(setup.ncurr) == 1:
        _, geometry = _geometry(carry.state, rt)
        jacobian = half_mesh_jacobian(geometry, s=setup.s_full)
        metrics = metric_elements(geometry, s=setup.s_full)
        fields = magnetic_fields(
            geometry=geometry, jacobian=jacobian, metrics=metrics, trig=rt.trig,
            s=setup.s_full, phips=setup.phips, phipf=setup.phipf,
            chips=setup.chips, signgs=setup.signgs, gamma=rt.gamma,
            mass=setup.mass, ncurr=setup.ncurr, enclosed_current=setup.icurv,
        )
        chips = np.asarray(fields.chips)
        phips = np.asarray(setup.phips)
        iotas = np.divide(chips, phips, out=np.zeros_like(chips), where=phips != 0.0)
        iotaf = np.zeros_like(iotas)
        iotaf[0] = 1.5 * iotas[1] - 0.5 * iotas[2]
        iotaf[1:-1] = 0.5 * (iotas[1:-1] + iotas[2:])
        iotaf[-1] = 1.5 * iotas[-1] - 0.5 * iotas[-2]
    else:
        iotaf = np.asarray(setup.iotaf)

    iterations = int(carry.iteration)
    trajectory = np.asarray(carry.trajectory)[:iterations]
    xm = np.asarray(rt.modes.m, dtype=float)
    xn = np.asarray(rt.modes.n, dtype=float) * float(rt.resolution.nfp)
    gamma = rt.gamma
    wb = float(carry.wb)
    wp = float(carry.wp)
    return SolveResult(
        converged=bool(int(carry.ier) == SUCCESSFUL_TERM_FLAG),
        iterations=iterations,
        ier_flag=int(carry.ier),
        fsqr=float(carry.fsqr), fsqz=float(carry.fsqz), fsql=float(carry.fsql),
        wb=wb, wp=wp, wmhd=float((wb + wp / (gamma - 1.0)) * _TWO_PI_SQ),
        r00=float(carry.r00),
        time_step=float(carry.time_step),
        jacobian_resets=int(carry.ijacob),
        state=state, xm=xm, xn=xn,
        rmnc=rmnc, zmns=zmns, rmns=rmns, zmnc=zmnc, iotaf=iotaf,
        fsq_history=trajectory[:, 1:7].copy(),
        newton_history=trajectory[:, 11:14].copy(),
    )


def _emit_lines(rt: SolverRuntime, trajectory: np.ndarray, upto: int,
                printed: set[int], final: bool, emit) -> None:
    """Print screen lines at the VMEC2000 cadence (eqsolve.f/printout.f)."""
    lasym = rt.resolution.lasym
    for it in range(1, upto + 1):
        due = (it == 1) or (it % rt.nstep == 0) or (final and it == upto)
        if not due or it in printed:
            continue
        row = trajectory[it - 1]
        if int(row[0]) != it:      # row not (yet) written for this iteration
            continue
        emit(screen_line(
            it, float(row[1]), float(row[2]), float(row[3]),
            float(row[7]), float(row[10]), float(row[9]),
            z_axis=float(row[8]) if lasym else None,
        ), end="")
        printed.add(it)


@jax.jit
def _while_lane(carry: _LoopCarry, rt: SolverRuntime) -> _LoopCarry:
    """Whole-solve ``lax.while_loop`` lane, keyed structurally on ``rt``.

    Module-level ``jax.jit`` with the runtime passed as a pytree argument:
    two DIFFERENT runtimes with equal structure (same meta, same leaf
    shapes/dtypes) — e.g. two boundaries at one :class:`Resolution`, hot
    restarts, optimization iterates — share one XLA executable.
    """
    body = _make_body(rt)
    return lax.while_loop(lambda c: jnp.logical_not(c.done), body, carry)


@functools.partial(jax.jit, donate_argnums=(0,))
def _block_lane(carry: _LoopCarry, rt: SolverRuntime) -> _LoopCarry:
    """One ``BLOCK_SIZE``-iteration ``lax.scan`` block (CLI lane), structural.

    ``donate_argnums=(0,)`` (R16.3): the CLI lane drives the solve as a Python
    loop ``carry = _block_lane(carry, rt)``, so the input carry is dead after
    each call — donating it lets XLA alias the (multi-array) carry's output
    onto the input buffers instead of allocating a fresh copy per block,
    removing the transient 2x-carry high-water mark.  ``rt`` (argument 1) is
    reused across blocks and is *not* donated.  Numerically identical to the
    non-donated lane.
    """
    body = _make_body(rt)
    return lax.scan(lambda cc, _: (body(cc), None), carry, None, length=BLOCK_SIZE)[0]


def _run_loop(state0: SpectralState, rt: SolverRuntime, *, mode: str,
              ijacob: int, verbose: bool, emit) -> _LoopCarry:
    """Run the iteration loop in the requested lane; return the final carry."""
    carry = _initial_carry(state0, rt, ijacob=ijacob)

    if mode == "jit":
        return _while_lane(carry, rt)

    if mode != "cli":
        raise ValueError(f"unknown mode {mode!r}; expected 'cli' or 'jit'")
    # The donated CLI lane (_block_lane, donate_argnums=0) requires every leaf
    # of the input carry to be a distinct buffer; _initial_carry aliases some
    # (xstore=state, shared cache zeros).  One copy to distinct buffers here
    # (values bit-for-bit unchanged) makes the per-block donation valid and is
    # amortized over the whole solve.
    carry = jax.tree.map(jnp.array, carry)
    if verbose:
        # initialize_radial.f prints the total Fourier mode count (mnmax), not mpol.
        emit(stage_banner(rt.resolution.ns, rt.resolution.mnmax, rt.ftol, rt.max_iterations), end="")
        emit(FORCE_ITERATIONS_BANNER, end="")
        emit(screen_header(lasym=rt.resolution.lasym, lfreeb=False), end="")

    printed: set[int] = set()
    max_passes = rt.max_iterations + 200
    for _ in range(max_passes):
        carry = _block_lane(carry, rt)
        done = bool(carry.done)
        upto = int(carry.iteration) if done else int(carry.iteration) - 1
        if verbose:
            trajectory = np.asarray(carry.trajectory[:max(upto, 0)])
            _emit_lines(rt, trajectory, upto, printed, done, emit)
        if done:
            break
    return carry


def _solve_stage(rt: SolverRuntime, state0: SpectralState | None, *,
                 mode: str, verbose: bool, emit,
                 try_axis_reguess: bool = True) -> _LoopCarry:
    """Run one solve at a fixed runtime, with the eqsolve.f axis-retry.

    ``state0=None`` starts from the runtime's ``profil3d.f`` interior guess.
    On a first-iteration Jacobian sign change with ``ijacob == 0``
    (``eqsolve.f``), the axis is re-guessed from the failing geometry and the
    loop restarted once (``try_axis_reguess``).  Returns the final carry;
    the caller maps ``carry.ier`` to results/exceptions (:func:`_finalize`).
    """
    setup = rt.setup
    if state0 is None:
        state0 = _initial_state(setup)
    carry = _run_loop(state0, rt, mode=mode, ijacob=0, verbose=verbose, emit=emit)

    # eqsolve.f: on a first-iteration Jacobian sign change with ijacob == 0,
    # re-guess the axis from the current geometry and restart once.
    if try_axis_reguess and int(carry.ier) == BAD_JACOBIAN_FLAG \
            and int(carry.ijacob) == 0 and rt.resolution.ns >= 3:
        if verbose:
            emit(" INITIAL JACOBIAN CHANGED SIGN!")
            emit(" TRYING TO IMPROVE INITIAL MAGNETIC AXIS GUESS")
        _, geometry = _geometry(state0, rt)
        axis = guess_axis(geometry, s=setup.s_full, trig=rt.trig, signgs=setup.signgs)
        new_state = interior_guess(
            boundary_R_cos=setup.boundary_R_cos, boundary_R_sin=setup.boundary_R_sin,
            boundary_Z_cos=setup.boundary_Z_cos, boundary_Z_sin=setup.boundary_Z_sin,
            raxis_c=axis[0], raxis_s=axis[1], zaxis_c=axis[2], zaxis_s=axis[3],
            modes=rt.modes, trig=rt.trig, s=setup.s_full,
        )
        state0 = SpectralState(
            R_cos=new_state[0], R_sin=new_state[1], Z_cos=new_state[2],
            Z_sin=new_state[3], L_cos=new_state[4], L_sin=new_state[5],
        )
        carry = _run_loop(state0, rt, mode=mode, ijacob=1, verbose=verbose,
                          emit=emit)
    return carry


def _finalize(carry: _LoopCarry, rt: SolverRuntime) -> SolveResult:
    """Map the final carry to a :class:`SolveResult` or a typed exception."""
    ier = int(carry.ier)
    fsq = (float(carry.fsqr), float(carry.fsqz), float(carry.fsql))
    if ier == SUCCESSFUL_TERM_FLAG:
        return _result_from_carry(carry, rt)
    if ier == MORE_ITER_FLAG:
        raise VmecConvergenceError(
            WERROR_MESSAGES[MORE_ITER_FLAG],
            hint="increase NITER or loosen FTOL",
            iteration=int(carry.iteration), fsq=fsq, ftol=rt.ftol,
        )
    raise VmecJacobianError(
        WERROR_MESSAGES.get(ier, WERROR_MESSAGES[JAC75_FLAG]),
        hint="decrease DELT or improve the axis guess",
        ier_flag=ier if ier in WERROR_MESSAGES else JAC75_FLAG,
        iteration=int(carry.iteration), jacobian_resets=int(carry.ijacob),
        fsq=fsq,
    )


def solve(
    source: VmecInput | RunSetup,
    resolution: Resolution | None = None,
    *,
    ftol: float | None = None, max_iterations: int | None = None,
    mode: str = "cli",
    time_step: float | None = None, tcon0: float | None = None,
    gamma: float | None = None, nstep: int | None = None,
    lconm1: bool = True, verbose: bool = False, emit=print,
    initial_state: SpectralState | None = None,
    boundary_from_initial_state: bool = False,
    error_on_no_convergence: bool = True,
    device: Any = None,
    precon_type: str | None = None, prec2d_threshold: float | None = None,
    prec2d: Prec2DConfig | None = None,
) -> SolveResult:
    """Single-grid fixed-boundary solve (VMEC2000 ``eqsolve.f``).

    ``source`` is a parsed :class:`vmec_jax.core.input.VmecInput`
    (recommended; supplies the ``delt/tcon0/gamma/nstep/ftol/niter`` defaults,
    with the keywords overriding) or a prebuilt
    :class:`vmec_jax.core.setup.RunSetup` (requires ``resolution``).  The
    resolution defaults to the first ``ns_array`` stage (``read_indata.f``
    grid rules).  Convergence requires ``fsqr, fsqz, fsql <= ftol``
    *simultaneously* (``evolve.f``).  ``mode="cli"`` runs a Python loop over
    jitted 10-iteration blocks with host residual checks and VMEC2000-format
    printing (``verbose=True``); ``mode="jit"`` runs one ``lax.while_loop``
    over the same traced body.

    Returns a :class:`SolveResult` on convergence.  Raises
    :class:`VmecJacobianError` when the initial Jacobian changes sign twice
    (after one ``guess_axis`` retry — the ``eqsolve.f`` ``ijacob == 0`` path)
    or at ``ijacob >= 75`` (``jac75_flag``), and :class:`VmecConvergenceError`
    when ``max_iterations`` is exhausted (``more_iter_flag``); both carry the
    final iteration and ``(fsqr, fsqz, fsql)`` diagnostics.

    ``initial_state`` hot-restarts the solve from a previous
    :class:`SpectralState` at the *same* resolution (e.g. ``result.state`` of
    an earlier solve on a perturbed boundary — VMEC++-style hot restart; use
    :func:`vmec_jax.core.multigrid.interpolate_state` first when ``ns``
    differs).  The R/Z *edge row* of the provided state is replaced by the
    input's processed boundary (the edge never evolves in fixed-boundary
    mode, so keeping the old row would silently re-solve the old boundary);
    the interior and lambda are kept.

    Set ``boundary_from_initial_state=True`` to hold the provided state's R/Z
    edge fixed instead. This is intended for a fixed-boundary predictor on a
    previously solved free-boundary LCFS. It requires ``initial_state`` and
    leaves the default hot-restart behavior unchanged.

    Set ``error_on_no_convergence=False`` to return the final iterate with
    ``converged=False`` and ``ier_flag=2`` when the iteration budget is
    exhausted. Jacobian and input failures still raise. This supports explicit
    checkpoint-and-continue workflows without weakening the convergence gate.

    ``device`` places the jitted iteration lanes: ``"cpu"``/``"gpu"``/
    ``"cuda"``/``"tpu"`` or a ``jax.Device`` (always honored), or ``None``
    (default) to apply the measured small-work-to-CPU policy of
    :mod:`vmec_jax.core.device` — which never overrides a user-pinned
    ``JAX_PLATFORMS``/``JAX_PLATFORM_NAME``.

    ``precon_type`` (``"NONE"`` default) with a finite ``prec2d_threshold`` —
    or an explicit ``prec2d``
    :class:`~vmec_jax.core.preconditioner_2d.Prec2DConfig` — switches on the
    optional **2D block preconditioner** (VMEC2000 ``precon2d.f``): once
    ``fsqr + fsqz + fsql < prec2d_threshold`` the iteration replaces the 1D
    radial force direction by a matrix-free Newton step (exact Hessian-vector
    products via ``jax.jvp``, solved with :func:`solvax.gmres`), converging
    stiff cases (high beta/aspect/mode-number) in far fewer iterations.  The
    default (``NONE``) path is byte-identical to the 1D-only solver.
    """
    if boundary_from_initial_state and initial_state is None:
        raise ValueError("boundary_from_initial_state requires initial_state")
    rt = prepare_runtime(
        source, resolution, ftol=ftol, max_iterations=max_iterations,
        time_step=time_step, tcon0=tcon0, gamma=gamma, nstep=nstep,
        lconm1=lconm1, precon_type=precon_type,
        prec2d_threshold=prec2d_threshold, prec2d=prec2d,
    )
    if initial_state is not None:
        ns, mnmax = rt.resolution.ns, rt.modes.mnmax
        if tuple(initial_state.R_cos.shape) != (ns, mnmax):
            raise ValueError(
                f"initial_state has shape {tuple(initial_state.R_cos.shape)}, "
                f"expected ({ns}, {mnmax}); interpolate with "
                "vmec_jax.core.multigrid.interpolate_state first"
            )
        if not boundary_from_initial_state:
            initial_state = hot_restart_state(rt, initial_state)
        rt = runtime_with_baselines(rt, initial_state)  # funct3d.f iter2==iter1
    with device_context(device, rt.resolution):
        carry = _solve_stage(rt, initial_state, mode=mode, verbose=verbose, emit=emit)
    if int(carry.ier) == MORE_ITER_FLAG and not error_on_no_convergence:
        return _result_from_carry(carry, rt)
    return _finalize(carry, rt)
