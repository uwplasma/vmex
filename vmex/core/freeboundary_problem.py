"""Scalar free-boundary optimization of the external field from an accepted equilibrium.

:class:`FreeBoundaryProblem` varies a design vector ``x`` that sets the
external field through ``field_from_parameters(x)`` (coils, currents, or any
differentiable field; coil charts live with the caller, for example in ESSOS)
and, optionally, plasma parameters such as PHIEDGE through
``plasma_from_parameters(params, x)``. Every trial is predicted from the
accepted root, solved with strict edge convergence and freshly certified;
derivatives use the dense ``forward_dense_jax`` adjoint, optionally reusing
the accepted LU as a matrix-free preconditioner. Only
:meth:`FreeBoundaryProblem.accept` changes the accepted root; ordinary
function evaluation, reporting and rejected trials never promote it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import cached_property
from pathlib import Path
import time
import traceback
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from . import _freeboundary_dense as dense, freeboundary as fb, freeboundary_implicit as fbi, implicit as im
from .errors import AdjointSolveError, TrialRejected, VmecError
from .optimize import Equilibrium
from .problem import FunctionProblem
from .solver import SolveResult, SpectralState, evaluate_forces
from .statephysics import volume
from .wout import wout_from_state

__all__ = ["FreeBoundaryProblem"]

#: Relative residual at which :func:`_refine` solves each Newton step with
#: GMRES on the seed LU.
_NEWTON_KRYLOV_RTOL = 1e-6
#: Largest relative defect of a Newton step in the raw linear equation it
#: solves; a larger defect means the seed LU no longer preconditions the root.
_NEWTON_LINEAR_RTOL = 1e-5
#: Step halvings tried before a Newton step counts as failed.
_NEWTON_BACKTRACKS = 8
#: Largest change of a structurally inactive coordinate a refinement may make.
_INACTIVE_DRIFT_ATOL = 1e-12
#: Defaults of :meth:`FreeBoundaryProblem.enable_matrix_free`; they also build
#: the temporary seed of a root polish that runs before it.
_MATRIX_FREE_DEFAULTS = dict(rtol=1e-11, restart=100, max_restarts=3, rhs_batch_size=3)


def _nonlinear_constraint(values, jacobian, lower, upper, scales):
    from scipy.optimize import NonlinearConstraint

    lo, hi, scale = np.broadcast_arrays(np.asarray(lower, dtype=float),
        np.asarray(upper, dtype=float), np.asarray(scales, dtype=float))
    if np.any(np.isnan(lo) | np.isnan(hi) | (lo > hi)) or np.any(~np.isfinite(scale) | (scale <= 0)):
        raise ValueError("ordered bounds and finite positive constraint scales required")
    return NonlinearConstraint(lambda x: np.asarray(values(x))/scale, lo/scale, hi/scale,
                               jac=lambda x: np.asarray(jacobian(x))/scale[..., None])


def _vector(value, shape=None):
    source = np.asarray(value)
    if (source.ndim != 1 or source.size == 0 or np.iscomplexobj(source)
            or not np.issubdtype(source.dtype, np.number)):
        raise ValueError("parameters must be a nonempty real vector")
    if shape is not None and source.shape != shape:
        raise ValueError(f"parameter shape must be {shape}, got {source.shape}")
    array = np.asarray(source, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("parameters must be finite")
    # Immutable backing bytes keep accepted parameters and cache keys fixed.
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)


def _positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _tree_norm(tree):
    """Euclidean norm of the flattened tree, as a host float.

    Kept separate from ``implicit._tree_norm`` (a sum of per-leaf dot
    products), whose rounding differs.
    """
    return float(jnp.linalg.norm(ravel_pytree(tree)[0]))


@dataclass(frozen=True, eq=False)
class _Root:
    """One certified root with the exact data needed for its pullback."""

    parameters: np.ndarray
    result: SolveResult
    dof_mask: SpectralState
    rcon0: Any
    zcon0: Any
    root_residual_norm: float
    _owner: Any = field(repr=False)

    @property
    def state(self):
        return self.result.state


@dataclass(frozen=True, eq=False)
class _Config:
    """Fixed profiles, solver (with the field map), accepted anchor and trial step bounds."""

    solver: fbi.FreeBoundaryImplicitConfig
    params: im.ImplicitParams
    parameter_scales: np.ndarray
    continuation_step: float
    max_continuation_steps: int
    root_residual_atol: float
    _anchor: _Root | None = field(default=None, repr=False)
    _owner: Any = field(default_factory=object, repr=False)


def _params_at(cfg, point):
    """``cfg.params`` with the plasma parameters the design point sets, if any."""
    plasma = cfg.solver.plasma_from_parameters
    return cfg.params if plasma is None else plasma(cfg.params, jnp.asarray(point))


def _config_from_state(solver, params, anchor, *, state, rcon0, zcon0, parameter_scales,
                       continuation_step, max_continuation_steps, root_residual_atol):
    """Certify an unchanged state as the accepted anchor, without solving."""
    if solver.adjoint_fail != "error":
        raise ValueError("optimization requires adjoint_fail='error'")
    anchor = _vector(anchor)
    scales = _vector(parameter_scales, anchor.shape)
    if np.any(scales <= 0):
        raise ValueError("parameter_scales must be positive")
    if (isinstance(max_continuation_steps, (bool, np.bool_))
            or int(max_continuation_steps) != max_continuation_steps or max_continuation_steps < 1):
        raise ValueError("max_continuation_steps must be a positive integer")
    with im._device_context(solver.implicit):
        fixed_params = im._device_pin(solver.implicit, jax.tree.map(jnp.array, params))
    cfg = _Config(solver, fixed_params, scales, _positive(continuation_step, "continuation_step"),
                  int(max_continuation_steps), _positive(root_residual_atol, "root_residual_atol"))
    root = _certify(cfg, anchor, state, rcon0=rcon0, zcon0=zcon0)
    return replace(cfg, _anchor=root)


def _certify(cfg, parameters, state, *, rcon0, zcon0, iterations=0, result=None):
    """Evaluate fixed-geometry vacuum, force and root checks; never change state."""
    solver, icfg = cfg.solver, cfg.solver.implicit
    if result is not None:
        if not isinstance(result, SolveResult) or not result.converged:
            raise VmecError("cannot certify a nonconverged solve")
        if any(not np.array_equal(np.asarray(a), np.asarray(b))
               for a, b in zip(jax.tree.leaves(state), jax.tree.leaves(result.state), strict=True)):
            raise ValueError("result and supplied state differ")
    point = _vector(parameters, cfg.parameter_scales.shape)
    params = _params_at(cfg, point)
    rt = im.runtime_from_params(params, icfg)
    expected = (icfg.resolution.ns, rt.modes.mnmax)
    if not isinstance(state, SpectralState):
        raise TypeError("state must be SpectralState")
    for leaf in jax.tree.leaves(state):
        array = np.asarray(leaf)
        if array.shape != expected or array.dtype != np.dtype("float64") or not np.all(np.isfinite(array)):
            raise ValueError("state must contain finite float64 arrays at the configured resolution")
    for baseline, reference in ((rcon0, rt.rcon0), (zcon0, rt.zcon0)):
        array = np.asarray(baseline)
        if array.shape != reference.shape or array.dtype != np.dtype("float64") or not np.all(np.isfinite(array)):
            raise ValueError("constraint baselines must be finite float64 arrays of the runtime shape")
    with im._device_context(icfg):
        state, rcon0, zcon0 = im._device_pin(icfg, jax.tree.map(jnp.asarray, (state, rcon0, zcon0)))
        rt = replace(rt, rcon0=rcon0, zcon0=zcon0, lfreeb=True, jmax=int(icfg.resolution.ns),
                     presf_ns_scale=fbi._presf_ns_scale_traceable(params, icfg.inp, int(icfg.resolution.ns)))
        external = solver.field_from_parameters(jnp.asarray(point))
        bsqvac = solver.vacuum_program.bsq(state, rt, external)
        if not np.all(np.isfinite(np.asarray(bsqvac))):
            raise VmecError("fresh vacuum evaluation is non-finite")
        rt = replace(rt, bsqvac_edge=bsqvac)
        _, forces, diagnostics = evaluate_forces(state, rt)
        values = {name: float(getattr(forces, name)) for name in ("fsqr", "fsqz", "fsql", "fedge")}
        if (not all(np.isfinite(v) and v >= 0 for v in values.values())
                or bool(diagnostics.jacobian_sign_changed)):
            raise VmecError("fresh force/geometry certification failed")
        for name in ("fsqr", "fsqz", "fsql"):
            if values[name] > float(icfg.ftol):
                raise VmecError(f"fresh {name}={values[name]:.6e} exceeds {float(icfg.ftol):.6e}")
        edge_tol = solver.edge_force_tolerance or float(icfg.ftol)
        if solver.include_edge_in_convergence and values["fedge"] > edge_tol:
            raise VmecError(f"fresh fedge={values['fedge']:.6e} exceeds {edge_tol:.6e}")
        mask = im._dof_mask(state, rt, icfg, evaluator=lambda x: evaluate_forces(x, rt)[0], fixed_edge=False)
        mask = jax.tree.map(jnp.asarray, mask)
        residual = fbi._projected_residual(solver, mask)
        project = im._dof_projector(icfg, mask)
        defect = residual(project(state), cfg.params, jnp.asarray(point), state, rcon0, zcon0)
        norm = _tree_norm(defect)
        if not np.isfinite(norm) or norm > cfg.root_residual_atol:
            raise VmecError(f"fresh root residual {norm:.6e} exceeds {cfg.root_residual_atol:.6e}")
        vol = float(volume(state, rt))
        if not np.isfinite(vol) or vol <= 0:
            raise VmecError("fresh volume must be finite and positive")
        if result is None:
            inp = im.input_with_params(icfg.inp, params)
            w = wout_from_state(inp=inp, state=state, niter=iterations,
                                **{n: values[n] for n in ("fsqr", "fsqz", "fsql")})
            wb, wp = float(diagnostics.wb), float(diagnostics.wp)
            result = SolveResult(converged=True, iterations=iterations, ier_flag=0,
                **values, wb=wb, wp=wp, wmhd=(wb + wp / (float(rt.gamma) - 1)) * (2 * np.pi)**2,
                r00=float(diagnostics.r00), time_step=0., jacobian_resets=0, state=state,
                xm=np.asarray(w.xm), xn=np.asarray(w.xn), rmnc=np.asarray(w.rmnc),
                zmns=np.asarray(w.zmns), rmns=w.rmns, zmnc=w.zmnc,
                iotaf=np.asarray(w.iotaf), fsq_history=np.empty((0, 6)))
        else:
            result = replace(result, state=state, **values)
        return _Root(point, result, mask, rcon0, zcon0, norm, cfg._owner)


@dataclass(eq=False)
class _LUPreconditioner:
    """Seed LU factors shared across roots of one configuration."""

    _cfg: Any = field(repr=False)
    _seed: Any = field(repr=False)

    def close(self):
        if self._seed is not None:
            self._seed.close()
        self._seed = self._cfg = None


@dataclass(eq=False)
class _Linearization:
    """A gradient and checked tangent solver owned by one root."""

    field_jacobian: Any
    _accepted: Any = field(repr=False)
    _cfg: Any = field(repr=False)
    _root: Any = field(repr=False)

    def preconditioner(self, **options):
        """Copy certified dense seed factors for bounded matrix-free solves.

        ``options`` are the GMRES controls of :meth:`SeedLU.from_root
        <vmex.core._freeboundary_dense.SeedLU.from_root>`.
        """
        if self._root is None:
            raise ValueError("linearization is closed")
        return _LUPreconditioner(self._cfg, dense.SeedLU.from_root(self._root, **options))

    def offload_factors(self):
        if self._root is None:
            raise ValueError("linearization is closed")
        self._root.offload_factors()

    def tangent(self, accepted, cfg, direction, *, diagnostics=None):
        if self._root is None:
            raise ValueError("linearization is closed")
        if accepted is not self._accepted or cfg is not self._cfg or accepted._owner is not cfg._owner:
            raise ValueError("linearization belongs to a different root or configuration")
        return self._root.tangent(direction, diagnostics=diagnostics)

    def close(self):
        if self._root is not None:
            self._root.close()
        self._root = self._accepted = self._cfg = self.field_jacobian = None


def _pullback(accepted, cfg, state_cotangents, *, diagnostics=None, preconditioner=None, structured_options=None):
    """Field-parameter derivatives at the supplied root, retaining its linearization.

    ``structured_options`` are the GMRES controls of a structured-factor solve
    (``adjoint_factorization="structured"``; see ``FreeBoundaryProblem._factor_options``).
    """
    if accepted._owner is not cfg._owner:
        raise ValueError("root belongs to a different configuration")
    options = {} if structured_options is None else {"structured_options": structured_options}
    if preconditioner is not None:
        if not isinstance(preconditioner, _LUPreconditioner) or preconditioner._cfg is not cfg:
            raise ValueError("preconditioner is closed or belongs to a different configuration")
        options["preconditioner"] = preconditioner._seed
    (_, field_bar), linearization = fbi.free_boundary_state_pullback_multi_rhs(
        cfg.params, jnp.asarray(accepted.parameters), cfg.solver,
        accepted.state, accepted.dof_mask, state_cotangents,
        rcon0=accepted.rcon0, zcon0=accepted.zcon0,
        root_residual_atol=cfg.root_residual_atol, diagnostics=diagnostics,
        return_linearization=True, **options)
    return _Linearization(field_bar, accepted, cfg, linearization)


@dataclass
class _RootPolishError(VmecError):
    """A bounded numerical root-refinement failure.

    Attributes
    ----------
    diagnostics:
        The failed Newton step's linear-solve report, when there is one.
    """

    diagnostics: dict | None = None


def _refine(accepted, cfg, preconditioner, *, tolerance=1e-12, max_steps=3, check_time=lambda: None):
    """Newton-refine the coupled plasma/vacuum root; return a newly certified root."""
    started = time.perf_counter()
    solver = cfg.solver
    seed = preconditioner._seed
    if preconditioner._cfg is not cfg:
        raise ValueError("preconditioner configuration mismatch")
    frozen = accepted.state
    project = im._dof_projector(solver.implicit, accepted.dof_mask)
    # Steps solve the raw coupled system, whose Jacobian the seed LU factors;
    # convergence is judged by the preconditioned residual that certification uses.
    residual = fbi._projected_residual(solver, accepted.dof_mask)
    raw = fbi._projected_residual(solver, accepted.dof_mask, formulation="raw")
    field_x = jnp.asarray(accepted.parameters)
    z = project(frozen)
    space = dense._active_space(solver.implicit, accepted.dof_mask, dense._max_dofs(solver))
    seed.validate(z, field_x, space, solver)
    space = jax.tree.map(jnp.asarray, space)
    factors = jax.tree.map(jnp.asarray, seed.factors)

    def evaluate(value, function=residual):
        return function(value, cfg.params, field_x, frozen, accepted.rcon0, accepted.zcon0)

    f = evaluate(z, raw)
    initial = magnitude = _tree_norm(evaluate(z))
    log = []
    for iteration in range(max_steps):
        check_time()
        if magnitude <= tolerance:
            break
        tick = time.perf_counter()
        action = dense.prepare(z, cfg.params, field_x, frozen, accepted.rcon0, accepted.zcon0,
                               residual=raw)
        correction, its, krylov_norm, converged = dense.solve(
            action, z, space, factors, -dense._compress(f, space), transpose=False,
            rtol=_NEWTON_KRYLOV_RTOL, restart=seed.restart, max_restarts=seed.max_restarts,
            return_info=True)
        delta = dense._expand(correction, z, space)
        defect_norm = _tree_norm(jax.tree.map(jnp.add, action(delta), f))
        relative_linear_error = defect_norm / _tree_norm(f)
        if not np.isfinite(relative_linear_error) or relative_linear_error > _NEWTON_LINEAR_RTOL:
            raise _RootPolishError(f"Newton linear residual failed: {relative_linear_error}",
                diagnostics=dict(iteration=iteration + 1, root_residual=magnitude,
                    defect_norm=defect_norm, linear_relative_error=relative_linear_error,
                    linear_relative_tolerance=_NEWTON_LINEAR_RTOL, krylov_iterations=int(its),
                    krylov_converged=bool(converged), krylov_norm=float(krylov_norm),
                    krylov_rtol=_NEWTON_KRYLOV_RTOL))
        # Accept a step that lowers the raw residual it solves or the
        # preconditioned one: a full Newton step can raise the preconditioned
        # norm once before converging quadratically, and near the root the raw
        # norm sits on its rounding floor.
        raw_norm = _tree_norm(f)
        for backtrack in range(_NEWTON_BACKTRACKS):
            check_time()
            alpha = 0.5**backtrack
            trial = jax.tree.map(lambda a, b: a + alpha * b, z, delta)
            trial_f = evaluate(trial, raw)
            trial_raw, trial_norm = _tree_norm(trial_f), _tree_norm(evaluate(trial))
            if np.isfinite(trial_raw) and np.isfinite(trial_norm) and (
                    trial_raw < raw_norm or trial_norm < magnitude):
                break
        else:
            raise _RootPolishError(f"Newton refinement did not reduce residual {magnitude}")
        log.append(dict(iteration=iteration + 1, before=magnitude, after=trial_norm,
                        alpha=alpha, krylov_iterations=int(its), krylov_converged=bool(converged),
                        krylov_norm=float(krylov_norm), linear_relative_error=relative_linear_error,
                        seconds=time.perf_counter() - tick))
        z, f, magnitude = trial, trial_f, trial_norm
    if not np.isfinite(magnitude) or magnitude > tolerance:
        raise _RootPolishError(f"Newton budget exhausted: {magnitude} > {tolerance}; steps={log}")
    displacement = project(jax.tree.map(jnp.subtract, z, frozen))
    state = jax.tree.map(jnp.add, frozen, displacement)
    inactive_change = _tree_norm(jax.tree.map(jnp.subtract, displacement, project(displacement)))
    if inactive_change > _INACTIVE_DRIFT_ATOL:
        raise _RootPolishError(f"inactive coordinate drift: {inactive_change}")
    tick = time.perf_counter()
    # Recompute vacuum, force diagnostics, geometry and root residual for the new state.
    refined = _certify(cfg, accepted.parameters, state, rcon0=accepted.rcon0, zcon0=accepted.zcon0)
    if not np.isfinite(refined.root_residual_norm) or refined.root_residual_norm > tolerance:
        raise _RootPolishError("refinement failed independently recomputed root gate")
    refined = replace(refined, result=replace(refined.result, iterations=accepted.result.iterations))
    return refined, dict(initial_residual=initial, final_residual=float(refined.root_residual_norm),
        tolerance=tolerance, steps=log, inactive_change=inactive_change,
        state_change=_tree_norm(displacement), certification_seconds=time.perf_counter() - tick,
        seconds=time.perf_counter() - started)


def _polish_with_recovery(record, cfg, preconditioner, build_dense, report, *,
                          retain_recovery=None, **options):
    """Refine with the accepted seed; on failure retry once with a fresh dense seed."""
    started = time.perf_counter()
    failure = None
    try:
        polished, evidence = _refine(record, cfg, preconditioner, **options)
    except (_RootPolishError, AdjointSolveError) as exc:
        failure = str(exc)
        diagnostics = getattr(exc, "diagnostics", None)
        # Failed frame locals hold a device LU and tape; release them first.
        traceback.clear_frames(exc.__traceback__)
        report(dict(event="dense_retry", failure=failure, linear_failure=diagnostics))
    if failure is not None:
        dense_root = seed = None
        try:
            tick = time.perf_counter()
            dense_root, seed = build_dense(record)
            dense_root.close()
            dense_root = None
            dense_seconds = time.perf_counter() - tick
            polished, evidence = _refine(record, cfg, seed, **options)
            report(dict(event="polished", recovered_with_dense=True,
                        first_failure=failure, total_seconds=time.perf_counter() - started, **evidence))
            if retain_recovery is not None:
                retain_recovery(polished, seed, dense_seconds)
                seed = None
            return polished
        finally:
            if seed is not None:
                seed.close()
            if dense_root is not None:
                dense_root.close()
    report(dict(event="polished", recovered_with_dense=False,
                first_failure=failure, total_seconds=time.perf_counter() - started, **evidence))
    return polished


@dataclass(frozen=True)
class _Equilibrium(Equilibrium):
    _wout_factory: Callable = field(kw_only=True, repr=False)

    @cached_property
    def wout(self):
        return self._wout_factory()


@dataclass
class _TrialLU:
    """Host factors from root recovery, owned by one unaccepted candidate."""

    record: object
    seed: _LUPreconditioner
    seconds: float


#: Accepted steps after a new seed whose timings are ignored: they can compile
#: new adjoint and predictor paths.
_REFRESH_WARMUP_STEPS = 2
#: Accepted steps in the running median of derivative costs.
_REFRESH_WINDOW = 3


@dataclass
class _LURefresh:
    """Estimate whether warm solve savings can repay a dense rebuild."""

    horizon: int
    dense_seconds: float
    warmup: int = _REFRESH_WARMUP_STEPS
    costs: list = field(default_factory=list)
    best_seconds: float = float("inf")
    remaining: int | None = None

    def observe(self, seconds):
        """Record one accepted step's cost; return whether a rebuild pays off."""
        if self.remaining is not None:
            self.remaining = max(0, self.remaining - 1)
        if self.warmup:
            self.warmup -= 1
            return False
        self.costs = (self.costs + [seconds])[-_REFRESH_WINDOW:]
        if len(self.costs) < _REFRESH_WINDOW:
            return False
        recent = float(np.median(self.costs))
        self.best_seconds = min(self.best_seconds, recent)
        horizon = self.horizon if self.remaining is None else min(self.horizon, self.remaining)
        return (recent - self.best_seconds) * horizon > self.dense_seconds


class FreeBoundaryProblem(FunctionProblem):
    """Weighted plasma objectives and derivatives with respect to field variables.

    Build with :meth:`from_loss`. The equilibrium and adjoint machinery is
    host-eager; the state objective functions are JIT compiled. No files,
    environment variables, CLI state or signal handlers are owned by this
    class. Pass it to :func:`vmex.core.optimize.minimize` (SLSQP), which calls
    :meth:`accept_x` only for iterates SciPy accepts.

    Attributes
    ----------
    accepted:
        The accepted, certified root (parameters, state, result, DOF mask,
        constraint baselines and root residual norm).
    accepted_step:
        Number of accepted optimizer steps.
    metadata:
        ``metadata["holder"]["failed_trials"]`` counts rejected trials.
    """

    @classmethod
    def from_loss(cls, inp, loss, x0, *, field_from_parameters, plasma_from_parameters=None,
                  scales=None, names=None, restart_from=None, solver_options=None,
                  quantities=(), parameter_quantities=(),
                  continuation_step=0.1, max_continuation_steps=64,
                  root_residual_atol=2e-6, event=None, deadline=None,
                  boundary_condition="nestor", three_term_options=None):
        """Solve and certify the seed equilibrium of a scalar loss of ``x``.

        Parameters
        ----------
        inp:
            Free-boundary input (``lfreeb``); it fixes the pressure and plasma
            current profiles, except where ``plasma_from_parameters`` sets them.
        loss:
            Scalar ``loss(state, runtime, x)``, used without normalization.
            Its gradient includes the equilibrium response and the explicit
            dependence on ``x``.
        x0:
            Initial design vector.
        field_from_parameters:
            Differentiable ``x -> external field`` (anything with ``b_cyl``,
            for example a Biot-Savart field of coils built from ``x``), as in
            :func:`~vmex.core.freeboundary_implicit.make_free_boundary_config`.
        plasma_from_parameters:
            Optional differentiable ``(params, x) -> params`` setting plasma
            parameters (for example PHIEDGE or the prescribed current profile)
            from ``x``; ``params`` are the input's
            :class:`~vmex.core.implicit.ImplicitParams`.
        scales, names:
            Coordinate scales (default ones) and names of ``x``.
        restart_from:
            Spectral state or WOUT path seeding the one ordinary solve.
        solver_options:
            Keyword arguments of
            :func:`~vmex.core.freeboundary_implicit.make_free_boundary_config`
            (for example ``device``, ``ftol``, ``edge_force_tolerance``,
            ``max_iterations``, ``adjoint_dense_batch_size`` (default 32),
            ``adjoint_dense_max_dofs`` and ``adjoint_residual_rtol``). Strict
            edge convergence, ``adjoint_solver="forward_dense_jax"`` and
            ``adjoint_fail="error"`` are required.
        quantities:
            Scalar or 1-D ``function(state, runtime)`` observables, the rows
            of :meth:`constraint_values`; the optimizer defines their bounds.
        parameter_quantities:
            Scalar or 1-D ``function(state, runtime, x)`` observables,
            appended after ``quantities``; their derivatives include the
            explicit and the equilibrium terms. Constraints on ``x`` alone
            (coil geometry, say) belong in ordinary optimizer constraints,
            which need no equilibrium adjoint.
        continuation_step, max_continuation_steps:
            A trial whose largest scaled step ``max |dx / scales|`` exceeds
            ``continuation_step * max_continuation_steps`` is rejected
            without a solve.
        root_residual_atol:
            Bound on the projected root residual of every certified root.
        event:
            Optional ``event(name, **data)`` callback for progress and timing.
            Names: ``proposal``, ``tangent``, ``correction``,
            ``certification``, ``candidate``, ``newton_correction``,
            ``root_polish``, ``adjoint_start``, ``adjoint``,
            ``matrixfree_recovery``, ``dense_recovery``,
            ``preconditioner_refresh_start`` and ``preconditioner_refresh``;
            data carries ``seconds`` where a phase is timed.
        deadline:
            Optional :func:`time.monotonic` value; construction, trials and
            derivatives raise :class:`TimeoutError` once it has passed.
        boundary_condition:
            ``"nestor"`` (VMEC + NESTOR, this class) or ``"three_term"``:
            every trial is the free boundary on which ``B.n``, the pressure
            jump and the sheet current vanish, and the problem is a
            :class:`~vmex.core.freeboundary_vc.ThreeTermFreeBoundaryProblem`
            with the same optimizer interface. It takes ``loss``, ``x0``, the
            parameter maps, ``scales``, ``names``, the quantities and
            ``three_term_options`` (keywords of
            :meth:`~vmex.core.freeboundary_vc.ThreeTermFreeBoundaryProblem.from_loss`: the
            boundary fit's ``boundary_ftol``, ``boundary_max_nfev`` and
            ``boundary_max_residual``, and the
            :class:`~vmex.core.freeboundary_vc.ThreeTermFreeBoundaryModel` keywords);
            the other arguments are NESTOR's.

        Returns
        -------
        FreeBoundaryProblem
            A problem whose accepted root is the certified seed equilibrium.
            Ordinary evaluations never promote a root.
        """
        if boundary_condition == "three_term":
            from .freeboundary_vc import ThreeTermFreeBoundaryProblem

            return ThreeTermFreeBoundaryProblem.from_loss(
                inp, loss, x0, field_from_parameters=field_from_parameters,
                plasma_from_parameters=plasma_from_parameters, scales=scales, names=names,
                quantities=quantities, parameter_quantities=parameter_quantities,
                **dict(three_term_options or {}))
        if boundary_condition != "nestor":
            raise ValueError(f"boundary_condition must be 'nestor' or 'three_term', got {boundary_condition!r}")
        quantities = tuple(quantities)
        parameter_quantities = tuple(parameter_quantities)
        if not all(callable(f) for f in (loss, field_from_parameters, *quantities, *parameter_quantities)) or (
                plasma_from_parameters is not None and not callable(plasma_from_parameters)):
            raise TypeError("loss, parameter maps and quantities must be callable")
        if not jax.config.x64_enabled:
            raise ValueError("free-boundary implicit optimization requires JAX_ENABLE_X64=1")
        if not inp.lfreeb:
            raise ValueError("input must enable free-boundary equilibrium")
        point = _vector(x0)
        opts = dict(solver_options or {})
        if opts.get("include_edge_in_convergence", True) is not True:
            raise ValueError("free-boundary optimization requires strict edge convergence")
        opts["include_edge_in_convergence"] = True
        opts.setdefault("adjoint_solver", "forward_dense_jax")
        opts.setdefault("adjoint_fail", "error")
        opts.setdefault("adjoint_dense_batch_size", 32)
        if opts["adjoint_solver"] != "forward_dense_jax" or opts["adjoint_fail"] != "error":
            raise ValueError("optimization requires adjoint_solver='forward_dense_jax' and adjoint_fail='error'")
        solver = fbi.make_free_boundary_config(
            inp, field_from_parameters(jnp.asarray(point)), field_from_parameters=field_from_parameters,
            plasma_from_parameters=plasma_from_parameters, **opts)
        params = im.params_from_input(inp)
        seed_inp = inp if plasma_from_parameters is None else im.input_with_params(
            inp, plasma_from_parameters(params, jnp.asarray(point)))

        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("walltime")
        if isinstance(restart_from, (str, Path)):
            from .restart import restart_state
            restart_from = restart_state(restart_from, inp, ns=solver.resolution.ns)
        stage = fb._solve_free_boundary_stage(
            seed_inp,
            external_field=field_from_parameters(jnp.asarray(point)),
            resolution=solver.resolution,
            ftol=solver.implicit.ftol,
            max_iterations=solver.implicit.max_iterations,
            initial_state=restart_from,
            include_edge_in_convergence=True,
            edge_force_tolerance=solver.edge_force_tolerance,
            error_on_no_convergence=False,
            jacobian_retries=0,
            allow_initial_axis_reguess=False,
            use_fft=False,
        )
        if not stage.result.converged:
            forces = ", ".join(f"{k}={float(getattr(stage.result, k, np.nan)):.3e}" for k in ("fsqr", "fsqz", "fsql", "fedge"))
            raise VmecError(f"initial ordinary equilibrium did not converge: {forces}")
        cfg = _config_from_state(
            solver,
            params,
            point,
            state=stage.result.state,
            rcon0=stage.rcon0,
            zcon0=stage.zcon0,
            parameter_scales=np.ones_like(point) if scales is None else scales,
            continuation_step=continuation_step,
            max_continuation_steps=max_continuation_steps,
            root_residual_atol=root_residual_atol,
        )
        if any(not np.array_equal(np.asarray(a), np.asarray(b)) for a, b in
               zip(jax.tree.leaves(stage.result.state), jax.tree.leaves(cfg._anchor.state))):
            raise VmecError("initial certification changed the ordinary state")
        if not np.isclose(stage.result.fedge, cfg._anchor.result.fedge, rtol=1e-5, atol=1e-15):
            raise VmecError("ordinary/fresh edge residual disagreement")
        return cls(inp, cfg, loss=loss, names=names, quantities=quantities,
                   parameter_quantities=parameter_quantities, event=event, deadline=deadline)

    def __init__(self, inp, cfg, *, loss, names=None, quantities=(), parameter_quantities=(),
                 event=None, deadline=None):
        self._accepted_linearization = self._accepted_jac = None
        self._preconditioner = self._matrixfree_options = None
        self._trial_lu = None
        self._lu_refresh = None
        self._adjoint_seconds = self._tangent_seconds = self._dense_seconds = 0.0
        self._polish_seconds = 0.0
        self._root_polish_options = None
        self._newton_options = None
        self._rejected = None
        self._refresh_parity_rtol = 1e-6
        self._dense_derivatives = self._matrixfree_fallback = False
        self._recovered = False
        self.inp, self.cfg = inp, cfg
        self.solver, self.params = cfg.solver, cfg.params
        self.rt = im.runtime_from_params(self.params, self.solver.implicit)
        if self.solver.plasma_from_parameters is None:
            def runtime(x):
                return self.rt
        else:  # rows see the design point's plasma parameters; self.rt keeps the input's
            def runtime(x):
                return im.runtime_from_params(_params_at(cfg, x), self.solver.implicit)
        self.accepted = cfg._anchor
        self.accepted_step = 0
        self._emit = event or (lambda *args, **kwargs: None)
        self.deadline = deadline
        self._linearization = self._linearization_record = None
        self._compact_jac = None
        self._records = {self._key(self.accepted.parameters): self.accepted}
        def scalar_rows(state, x):
            rt = runtime(x)
            value = jnp.asarray(loss(state, rt, x))
            if value.shape != ():
                raise ValueError("loss must return a scalar")
            # A quantity may be a scalar or a vector of rows (e.g. iota per surface).
            values = [jnp.asarray(function(state, rt)) for function in quantities]
            values += [jnp.asarray(function(state, rt, x)) for function in parameter_quantities]
            if any(v.ndim > 1 for v in values):
                raise ValueError("quantities must be scalars or one-dimensional arrays")
            return jnp.concatenate([value[None], *(jnp.ravel(v) for v in values)])

        self._scalar_rows = jax.jit(scalar_rows)
        self._scalar_jac = jax.jit(jax.jacrev(scalar_rows, argnums=(0, 1)))
        # Validate quantities eagerly; an invalid definition must not start an optimizer.
        initial = self.optimizer_rows(self.accepted)
        if not np.all(np.isfinite(initial)):
            raise ValueError("nonfinite objective or physical constraints")
        super().__init__(
            self.accepted.parameters,
            names=names,
            scales=cfg.parameter_scales,
            fun=self._value,
            value_and_grad=self._value_gradient,
            metadata={"holder": {"failed_trials": 0}},
        )

    def _check_time(self):
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise TimeoutError("walltime")

    @staticmethod
    def _x(x):
        if np.iscomplexobj(x):
            raise ValueError("parameters must be real")
        return FunctionProblem._x(x)

    def _validate_x(self, x):
        if np.iscomplexobj(x):
            raise ValueError("parameters must be real")
        x = np.asarray(x, dtype=float)
        if x.shape != self.x0.shape or not np.all(np.isfinite(x)):
            raise ValueError("invalid parameter vector")
        return x

    def _record(self, x):
        x = self._validate_x(x)
        key = self._key(x)
        if key not in self._records:
            # An optimizer may query values and constraints at the same rejected
            # point; reuse the rejection instead of repeating the failed solve.
            if self._rejected is not None and self._rejected[0] == key:
                raise TrialRejected(self._rejected[1])
            try:
                self._records[key] = self._trial(x, 0)
            except TrialRejected as exc:
                self._rejected = (key, str(exc))
                raise
            # Retain at most the anchor and most recently evaluated trial.
            self._records = {self._key(self.accepted.parameters): self.accepted, key: self._records[key]}
        return self._records[key]

    def _close_linearization(self):
        if self._linearization is not None and self._linearization is not self._accepted_linearization:
            self._linearization.close()
        self._linearization = self._linearization_record = self._compact_jac = None

    def _close_trial_lu(self):
        if self._trial_lu is not None:
            self._trial_lu.seed.close()
            self._trial_lu = None

    def _derivatives(self, record):
        """Total compact derivative rows at ``record``, retaining its linearization.

        A trial recovered by dense polishing uses its own LU; otherwise dense
        mode factors afresh and matrix-free mode uses the accepted seed LU.
        A failed solve gets one recovery at this certified root.
        """
        self._check_time()
        if self._linearization_record is record:
            return self._compact_jac
        self._close_linearization()
        if record is self.accepted and self._accepted_linearization is not None:
            self._linearization, self._compact_jac = self._accepted_linearization, self._accepted_jac
            self._linearization_record = record
            return self._compact_jac
        diagnostics = []
        trial_lu = self._trial_lu
        preconditioner = (trial_lu.seed if trial_lu is not None and trial_lu.record is record
                          else None if self._dense_derivatives else self._preconditioner)
        self._emit("adjoint_start")
        started = time.monotonic()
        dense_started = started if preconditioner is None else None
        self._recovered = self._matrixfree_fallback = False
        try:
            rhs, direct = self._scalar_jac(record.state, jnp.asarray(record.parameters))
            options = {} if preconditioner is None else {"preconditioner": preconditioner}
            try:
                linearization = _pullback(
                    record, self.cfg, rhs, diagnostics=diagnostics, **self._factor_options,
                    **options)
            except AdjointSolveError as error:
                fallback = self._recovery_seed(preconditioner)
                if preconditioner is None and fallback is None:
                    raise
                traceback.clear_frames(error.__traceback__)
                self._emit("matrixfree_recovery" if fallback is not None else "dense_recovery",
                           candidate=record, error=str(error))
                # A matrix-free retry is not timed as a dense solve.
                dense_started = None if fallback is not None else time.monotonic()
                linearization = _pullback(
                    record, self.cfg, rhs, diagnostics=diagnostics, **self._factor_options,
                    **({} if fallback is None else {"preconditioner": fallback}))
                self._matrixfree_fallback, self._recovered = fallback is not None, fallback is None
            jac = np.asarray(linearization.field_jacobian) + np.asarray(direct)
            if not np.all(np.isfinite(jac)):
                linearization.close()
                raise FloatingPointError("nonfinite total derivative")
            self._linearization = linearization
            self._linearization_record, self._compact_jac = record, jac
            linearization.offload_factors()
            if record is self.accepted:
                self._accepted_linearization, self._accepted_jac = linearization, jac
        except BaseException:
            if trial_lu is not None and trial_lu.record is record:
                self._close_trial_lu()
            raise
        finally:
            finished = time.monotonic()
            self._adjoint_seconds = finished - started
            if dense_started is not None:
                self._dense_seconds = finished - dense_started
            self._emit("adjoint", seconds=self._adjoint_seconds, rows=diagnostics, solver=self.solver_info)
        return self._compact_jac

    def _recovery_seed(self, preconditioner):
        """Choose the one retry after a failed derivative solve.

        Returns the accepted seed LU for a matrix-free retry of a dense
        derivative that missed its gate (dense mode), or ``None`` for a
        checked dense retry of a failed seed-LU solve. A dense derivative
        without a seed has no retry (``preconditioner`` and the result are
        both ``None``). A rejected candidate never replaces the accepted seed.
        """
        return self._preconditioner if preconditioner is None and self._dense_derivatives else None

    @property
    def solver_info(self):
        """Configured adjoint, reuse policy and predictor; actual solves report rows."""
        method = self.solver.adjoint_solver
        reuse = self._preconditioner is not None
        return dict(adjoint_solver=method,
            active_adjoint="matrixfree_seed_lu" if reuse and not self._dense_derivatives else method,
            preconditioner="seed_lu" if reuse else None,
            recovery="forward_dense_jax" if reuse else None,
            predictor="matrixfree_seed_lu" if reuse else "reused_dense_lu",
            refresh_horizon=None if self._lu_refresh is None else self._lu_refresh.horizon,
            root_polish_atol=None if self._root_polish_options is None else self._root_polish_options['tolerance'],
            adjoint_residual_rtol=getattr(self.solver, "adjoint_residual_rtol", None),
            factorization=getattr(self.solver, "adjoint_factorization", "dense"))

    @property
    def _factor_options(self):
        """``_pullback`` options: structured factors solve by GMRES with the matrix-free controls."""
        if getattr(self.solver, "adjoint_factorization", "dense") != "structured":
            return {}
        return {"structured_options": self._matrixfree_options or _MATRIX_FREE_DEFAULTS}

    def enable_root_polishing(self, *, tolerance=1e-12, max_steps=3):
        """Newton-polish the accepted root and every later trial before use.

        Call before :meth:`enable_matrix_free` and before any accepted step.
        Ordinary equilibrium convergence is still required. Each bounded
        Newton refinement keeps inactive coordinates and constraint baselines,
        then freshly certifies the coupled residual and physical forces.
        Independent finite-difference trials get the same polishing, without
        a predictor. A refinement that fails on the accepted seed LU is retried
        once with a fresh dense LU. No optimizer step is accepted.

        Parameters
        ----------
        tolerance:
            Projected root-residual norm every polished root must reach.
        max_steps:
            Newton steps allowed per root.

        Returns
        -------
        root
            The (possibly polished) accepted root. On failure the original
            root and derivative caches are kept.
        """
        if self._preconditioner is not None or self.accepted_step != 0:
            raise ValueError('enable root polishing before matrix-free setup and optimization')
        if (not np.isfinite(tolerance) or tolerance <= 0 or isinstance(max_steps, bool)
                or not isinstance(max_steps, int) or max_steps < 1):
            raise ValueError('positive finite tolerance and integer max_steps required')
        options = dict(tolerance=tolerance, max_steps=max_steps)
        polished = self._polish_record(self.accepted, options=options)
        # Initialization is transactional: failures leave the original root and
        # derivative caches usable; changing a root invalidates all old caches.
        if polished is not self.accepted:
            self._close_linearization()
            if self._accepted_linearization is not None:
                self._accepted_linearization.close()
            self._accepted_linearization = self._accepted_jac = None
            self.accepted = polished
            self._records = {self._key(polished.parameters): polished}
            self._vg_cache = self._rj_cache = None
        self._root_polish_options = options
        return self.accepted

    def enable_newton_correction(self, *, max_steps=8):
        """Correct predicted trials by Newton on the coupled root before any ordinary solve.

        Requires :meth:`enable_root_polishing` and :meth:`enable_matrix_free`.
        Newton starts from the tangent prediction, uses the accepted seed LU
        and must reach the polishing tolerance; the result is certified
        freshly. Any failure falls back to the ordinary equilibrium solve for
        that trial. Finite-difference trials (``predict=False`` or an explicit
        ``ftol``) always use the ordinary solve.

        Parameters
        ----------
        max_steps:
            Newton steps allowed per trial.
        """
        if self._root_polish_options is None or self._preconditioner is None:
            raise ValueError("enable root polishing and matrix-free reuse before Newton correction")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        self._newton_options = dict(tolerance=self._root_polish_options["tolerance"], max_steps=max_steps)

    def _newton_trial(self, x, predicted, trial):
        """Return a certified Newton-corrected root, or None to use the ordinary solve."""
        started = time.monotonic()
        guess = _Root(_vector(x), replace(self.accepted.result, state=predicted, iterations=0),
                      self.accepted.dof_mask, self.accepted.rcon0, self.accepted.zcon0, np.inf, self.cfg._owner)
        try:
            candidate, evidence = _refine(guess, self.cfg, self._preconditioner, check_time=self._check_time,
                                          **self._newton_options)
        except (_RootPolishError, AdjointSolveError, VmecError, ValueError) as exc:
            traceback.clear_frames(exc.__traceback__)
            self._emit("newton_correction", trial=trial, accepted=False, error=str(exc)[:300],
                       seconds=time.monotonic() - started)
            return None
        self._emit("newton_correction", trial=trial, accepted=True, steps=len(evidence["steps"]),
                   final_residual=evidence["final_residual"], seconds=time.monotonic() - started)
        return candidate

    def _polish_record(self, record, *, options=None):
        options = self._root_polish_options if options is None else options
        if options is None or record.root_residual_norm <= options['tolerance']:
            return record

        self._check_time()
        started = time.monotonic()

        def build_dense(root):
            self._check_time()
            rhs, _ = self._scalar_jac(root.state, jnp.asarray(root.parameters))
            dense_root = _pullback(
                root, self.cfg, rhs, **self._factor_options)
            try:
                seed = dense_root.preconditioner(**(self._matrixfree_options or _MATRIX_FREE_DEFAULTS))
            except BaseException:
                dense_root.close()
                raise
            return dense_root, seed

        def report(evidence):
            details = dict(evidence)
            details['phase'] = details.pop('event')
            self._emit('root_polish', **details)

        def retain_recovery(polished, seed, seconds):
            self._close_trial_lu()
            self._trial_lu = _TrialLU(polished, seed, seconds)

        try:
            if self._preconditioner is not None:
                return _polish_with_recovery(record, self.cfg, self._preconditioner,
                    build_dense, report, retain_recovery=retain_recovery,
                    check_time=self._check_time, **options)
            # Bootstrap uses temporary factors. No failed or rejected trial
            # can replace the accepted root's retained preconditioner.
            dense_root, seed = build_dense(record)
            try:
                # preconditioner() owns independent host factors; discard the
                # temporary dense GPU factors/tape before creating the polish tape.
                dense_root.close()
                polished, evidence = _refine(record, self.cfg, seed,
                    check_time=self._check_time, **options)
                report(dict(event='polished', recovered_with_dense=False, **evidence))
                return polished
            finally:
                seed.close()
                dense_root.close()
        finally:
            self._polish_seconds += time.monotonic() - started

    def enable_matrix_free(self, *, rtol=1e-11, restart=100, max_restarts=3,
                           rhs_batch_size=3, parity_rtol=1e-6, refresh_horizon=None,
                           refresh_max_steps=None, dense_derivatives=False):
        """Keep a checked dense LU of the accepted root as a seed preconditioner.

        Call once. The seed preconditions matrix-free GMRES solves of trial
        adjoints, tangents and Newton corrections; full residuals still obey
        ``solver_options["adjoint_residual_rtol"]``. A failed trial adjoint
        gets one dense retry, whose LU replaces the seed only if that trial is
        accepted.

        Parameters
        ----------
        rtol, restart, max_restarts:
            GMRES relative tolerance, restart length and restart cycles.
        rhs_batch_size:
            Distinct right-hand sides solved together (1 to 8).
        parity_rtol:
            Largest relative change of any derivative row when a cost refresh
            rebuilds the dense LU.
        refresh_horizon:
            Number of future accepted steps over which estimated warm-solve
            savings should repay a dense rebuild. Only accepted steps
            contribute timings: the first two after each seed are skipped,
            then the latest three-step median is compared with the best one.
            ``None`` refreshes only on failure. A refresh never relaxes a
            residual gate.
        refresh_max_steps:
            Optional remaining accepted-step budget (requires
            ``refresh_horizon``), so no rebuild is started that cannot pay
            back before the optimizer stops.
        dense_derivatives:
            Compute every derivative with the dense solve and make each
            accepted step's LU the new seed: no matrix-free adjoint on older
            factors, and trial Newton corrections start from the latest
            root's LU. A dense derivative that misses its gate retries
            matrix-free on the seed, and that root's LU is then not used as a
            seed.
        """
        if self._preconditioner is not None:
            raise ValueError("enable matrix-free once")
        if refresh_horizon is not None and (
            isinstance(refresh_horizon, bool) or not isinstance(refresh_horizon, int) or refresh_horizon < 1
        ):
            raise ValueError("refresh_horizon must be a positive integer or None")
        if refresh_max_steps is not None and (
            refresh_horizon is None or isinstance(refresh_max_steps, bool)
            or not isinstance(refresh_max_steps, int) or refresh_max_steps < 1
        ):
            raise ValueError('refresh_max_steps requires a refresh horizon and a positive integer')
        if not np.isfinite(parity_rtol) or parity_rtol <= 0:
            raise ValueError('positive finite parity tolerance required')
        self._refresh_parity_rtol = parity_rtol
        self._dense_derivatives = bool(dense_derivatives)
        options = dict(rtol=rtol, restart=restart, max_restarts=max_restarts, rhs_batch_size=rhs_batch_size)
        self._derivatives(self.accepted)
        self._preconditioner = self._linearization.preconditioner(**options)
        self._matrixfree_options = options
        if refresh_horizon is not None:
            self._lu_refresh = _LURefresh(refresh_horizon, self._dense_seconds, remaining=refresh_max_steps)

    def optimizer_rows(self, record):
        """Return compact objective/constraint rows without a derivative solve."""
        return np.asarray(self._scalar_rows(record.state, jnp.asarray(record.parameters)))

    def _trial(self, x, trial, *, predict=True, ftol=None):
        """Predict, correct and certify one trial point; never promote it.

        The trial starts from the tangent prediction at the accepted root
        (``predict=False``: from the accepted state itself), is Newton
        corrected when enabled, and otherwise solved once with strict edge
        convergence, certified and polished. Any failure is a
        :class:`~vmex.core.errors.TrialRejected`.
        """
        self._check_time()
        # A new proposal abandons the previous candidate's recovery factors.
        # Accepted factors and its predictor linearization remain independent.
        self._close_trial_lu()
        self._polish_seconds = 0.0
        x = np.asarray(x, dtype=float)
        delta = x - self.accepted.parameters
        # Bound the distance from the accepted root that a single tangent
        # prediction is asked to cover.
        substeps = max(1, int(np.ceil(np.max(np.abs(delta / self.scales)) / self.cfg.continuation_step)))
        if substeps > self.cfg.max_continuation_steps:
            raise TrialRejected("continuation budget exceeded")
        if predict:
            self._derivatives(self.accepted)
        self._emit("proposal", delta=delta.copy(), trial=trial, points=1,
                   jacobian=self._compact_jac.copy() if predict else None)
        tolerance = self.solver.implicit.ftol if ftol is None else float(ftol)
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("positive finite force tolerance required")
        stage = point = None
        try:
            diagnostics = []
            started = time.monotonic()
            try:
                tangent = (self._linearization.tangent(
                    self.accepted, self.cfg, jnp.asarray(delta), diagnostics=diagnostics
                ) if predict else jax.tree.map(jnp.zeros_like, self.accepted.state))
            finally:
                seconds = time.monotonic() - started
                self._tangent_seconds = seconds if predict else 0.0
                self._emit("tangent", trial=trial, seconds=seconds, rows=diagnostics)
            anchor = self.accepted
            self._check_time()
            # Solve at the exact requested x: rebuilding it from the accepted
            # point and delta can change its last bit and break cache identity.
            point = x
            predicted = jax.tree.map(lambda value, change: value + change, anchor.state, tangent)
            if self._newton_options is not None and predict and ftol is None:
                corrected = self._newton_trial(point, predicted, trial)
                if corrected is not None:
                    self._emit("certification", candidate=corrected, trial=trial, index=1)
                    return corrected
            started = time.monotonic()
            try:
                stage = fb._solve_free_boundary_stage(
                    self._inp_at(point),
                    external_field=self.solver.field_from_parameters(jnp.asarray(point)),
                    resolution=self.solver.resolution,
                    ftol=tolerance,
                    max_iterations=self.solver.implicit.max_iterations,
                    initial_state=predicted,
                    constraint_continuation=(anchor.rcon0, anchor.zcon0),
                    include_edge_in_convergence=True,
                    edge_force_tolerance=self.solver.edge_force_tolerance if ftol is None else tolerance,
                    error_on_no_convergence=False,
                    jacobian_retries=0,
                    allow_initial_axis_reguess=False,
                    use_fft=False,
                )
            except BaseException:
                # A failed solve is still a correction cost; record it before rejecting.
                self._emit("correction", stage=None, parameters=point, trial=trial, index=1,
                           points=1, failed=True, seconds=time.monotonic() - started)
                raise
            self._emit("correction", stage=stage, parameters=point, trial=trial, index=1,
                       points=1, seconds=time.monotonic() - started)
            if not stage.result.converged:
                raise TrialRejected("ordinary equilibrium did not converge")
            started = time.monotonic()
            candidate = _certify(self.cfg, point, stage.result.state, rcon0=stage.rcon0,
                                 zcon0=stage.zcon0, result=stage.result)
            certified = time.monotonic() - started
            candidate = self._polish_record(candidate)
            if ftol is not None and self._root_polish_options is not None:
                forces = [float(getattr(candidate.result, name))
                          for name in ('fsqr', 'fsqz', 'fsql', 'fedge')]
                if not all(np.isfinite(value) and value <= tolerance for value in forces):
                    raise TrialRejected('polished trial exceeds requested force tolerance')
            self._emit("certification", candidate=candidate, trial=trial, index=1, seconds=certified)
            return candidate
        except (VmecError, TrialRejected) as exc:
            self._close_trial_lu()
            self.metadata["holder"]["failed_trials"] += 1
            if isinstance(exc, TrialRejected):
                raise
            raise TrialRejected(f"{type(exc).__name__}: {exc}") from exc
        except BaseException:
            self._close_trial_lu()
            raise
        finally:
            self._emit("candidate", stage=stage, parameters=point, trial=trial)

    def evaluate_trial(self, delta, trial=0, *, predict=True, ftol=None):
        """Evaluate the point ``accepted.parameters + delta`` without promoting it.

        This is the finite-difference verification interface; optimizers use
        ``fun``/``value_and_grad`` and :meth:`accept_x`.

        Parameters
        ----------
        delta:
            Step from the accepted parameters.
        trial:
            Label passed to the ``event`` callback.
        predict:
            ``False`` starts the solve from the accepted state instead of the
            tangent prediction, and needs no derivative, for independent
            finite differences.
        ftol:
            Optional tighter tolerance for both the ordinary and the edge
            force convergence (also checked after polishing).

        Returns
        -------
        tuple
            ``(candidate, rows)``: the certified root, which :meth:`accept`
            can promote, and its loss and quantity rows.

        Raises
        ------
        vmex.core.errors.TrialRejected
            If the trial cannot be solved or certified.
        """
        delta = self._validate_x(delta)
        candidate = self._trial(self.accepted.parameters + delta, trial, predict=predict, ftol=ftol)
        self._records = {self._key(self.accepted.parameters): self.accepted, self._key(candidate.parameters): candidate}
        return candidate, self.optimizer_rows(candidate)

    def accept(self, candidate):
        """Promote a candidate returned by this problem after optimizer acceptance.

        The candidate's derivative is computed (or reused) first; its dense LU
        becomes the new seed preconditioner after a recovery, in dense mode,
        or when a cost refresh pays off. Nothing changes if that fails.

        Parameters
        ----------
        candidate:
            A root from :meth:`evaluate_trial`, or the root of the most
            recent evaluation.
        """
        if self._records.get(self._key(candidate.parameters)) is not candidate:
            raise ValueError("candidate is not a current evaluation of this problem")
        if candidate is self.accepted:
            return
        self._derivatives(candidate)
        linearization, jac = self._linearization, self._compact_jac
        trial_lu = self._trial_lu
        recovered_root = trial_lu is not None and trial_lu.record is candidate
        reason = "recovery" if self._recovered else "root_recovery" if recovered_root else None
        if reason is None and self._dense_derivatives and not self._matrixfree_fallback:
            reason = "dense"  # the accepted derivative was a dense solve: its LU is the new seed
        refresh = (None if self._lu_refresh is None else
                   replace(self._lu_refresh, costs=list(self._lu_refresh.costs)))
        # After a matrix-free fallback the dense solve at this root just missed its gate: do not refresh with it.
        if refresh is not None and reason is None and not self._matrixfree_fallback and refresh.observe(
            self._adjoint_seconds + self._tangent_seconds + self._polish_seconds
        ):
            reason = "cost"
            linearization, jac = self._refresh_linearization(candidate, refresh)
        try:
            replacement = (trial_lu.seed if reason == "root_recovery" else
                           linearization.preconditioner(**self._matrixfree_options)
                           if reason is not None else None)
        except BaseException:
            if linearization is not self._linearization:
                linearization.close()
            raise
        if linearization is not self._linearization:
            self._linearization.close()
        if self._accepted_linearization is not None:
            self._accepted_linearization.close()
        self.accepted = candidate
        self._lu_refresh = refresh
        self._accepted_linearization = self._linearization = linearization
        self._accepted_jac = self._compact_jac = jac
        if replacement is not None:
            if reason == "root_recovery":
                self._dense_seconds = trial_lu.seconds
                self._trial_lu = None  # ownership moves to the accepted seed
            self._preconditioner.close()
            self._preconditioner = replacement
            if refresh is not None:
                remaining = refresh.remaining
                if reason in ("recovery", "root_recovery") and remaining is not None:
                    remaining = max(0, remaining - 1)
                # Compilation happens once; after a refresh the next steps are already warm.
                self._lu_refresh = _LURefresh(refresh.horizon, self._dense_seconds, warmup=0, remaining=remaining)
            # A dense derivative was already timed as the adjoint; only a separate rebuild costs more.
            self._emit("preconditioner_refresh", reason=reason,
                       seconds=0.0 if reason == "dense" else self._dense_seconds)
        self._close_trial_lu()
        # Seeds and tapes never cross configuration identities; every later
        # proposal is predicted from this accepted root.
        self.accepted_step += 1
        self._rejected = None  # a rejection holds only for the anchor it was predicted from
        self._records = {self._key(self.accepted.parameters): self.accepted}
        self._vg_cache = self._rj_cache = None

    def _refresh_linearization(self, candidate, refresh):
        """Rebuild the dense LU at ``candidate`` for a cost refresh.

        The rebuilt derivative rows must match the current ones within the
        parity tolerance; the caller installs the new seed.
        """
        self._check_time()
        diagnostics = []
        started = time.monotonic()
        self._emit("preconditioner_refresh_start", reason="cost",
            recent_seconds=float(np.median(refresh.costs)), best_seconds=refresh.best_seconds,
            dense_seconds=refresh.dense_seconds, horizon=refresh.horizon)
        rhs, direct = self._scalar_jac(candidate.state, jnp.asarray(candidate.parameters))
        linearization = _pullback(
            candidate, self.cfg, rhs, diagnostics=diagnostics, **self._factor_options)
        try:
            jac = np.asarray(linearization.field_jacobian) + np.asarray(direct)
            if not np.all(np.isfinite(jac)):
                raise FloatingPointError("nonfinite total derivative")
            errors = np.linalg.norm(jac-self._compact_jac, axis=1) / np.maximum(
                np.linalg.norm(jac, axis=1), 1e-30)
            if not np.all(errors <= self._refresh_parity_rtol):
                raise AdjointSolveError(f'dense refresh changed derivative rows: {errors}')
            linearization.offload_factors()
        except BaseException:
            linearization.close()
            raise
        self._dense_seconds = time.monotonic() - started
        self._emit("adjoint", seconds=self._dense_seconds, rows=diagnostics,
            solver=self.solver_info, reason="preconditioner_refresh",
            gradient_relative_errors=errors.tolist())
        return linearization, jac

    def accept_x(self, x):
        """Promote an already evaluated point after the optimizer accepts it."""
        candidate = self._records.get(self._key(self._validate_x(x)))
        if candidate is None:
            raise ValueError("evaluate the candidate before accepting it")
        self.accept(candidate)

    def _value(self, x):
        rows = self.optimizer_rows(self._record(x))
        return float(rows[0])

    def _value_gradient(self, x):
        record = self._record(x)
        rows = self.optimizer_rows(record)
        jac = self._derivatives(record)
        return float(rows[0]), jac[0].copy()

    def constraint_values(self, x):
        """Return unscaled physical quantities, in their original units."""
        return self.optimizer_rows(self._record(x))[1:]

    def constraint_jac(self, x):
        """Return derivatives of the physical quantities with respect to x."""
        record = self._record(x)
        return self._derivatives(record)[1:].copy()

    def nonlinear_constraint(self, lower, upper, *, scales=1.0):
        """Bound the physical quantities supplied to from_loss, in their units."""
        return _nonlinear_constraint(self.constraint_values, self.constraint_jac, lower, upper, scales)

    def _inp_at(self, x):
        """The input deck with the plasma parameters of design point ``x``."""
        if self.solver.plasma_from_parameters is None:
            return self.inp
        return im.input_with_params(self.inp, _params_at(self.cfg, x))

    def equilibrium_from_x(self, x):
        """Return a certified equilibrium; WOUT uses its exact fixed-geometry vacuum."""
        record = self._record(x)
        rt = (self.rt if self.solver.plasma_from_parameters is None else
              im.runtime_from_params(_params_at(self.cfg, record.parameters), self.solver.implicit))
        return _Equilibrium(self._inp_at(record.parameters), record.state, rt, record.result,
                            _wout_factory=lambda: self._wout(record))

    def close(self):
        """Release retained derivative factors without altering accepted results."""
        self._close_linearization()
        self._close_trial_lu()
        if self._accepted_linearization is not None:
            self._accepted_linearization.close()
        if self._preconditioner is not None:
            self._preconditioner.close()
        self._accepted_linearization = self._accepted_jac = self._preconditioner = None

    def _wout(self, record):
        """WOUT of ``record``; external currents stay with the caller's field (no EXTCUR)."""
        # Re-evaluate the vacuum on this exact fixed plasma/coil geometry for
        # every exported pair. Imported anchors have no attached VacuumOutput;
        # ordinary results can carry cadence caches from a preceding geometry.
        params = _params_at(self.cfg, record.parameters)
        export_rt = replace(
            im.runtime_from_params(params, self.solver.implicit),
            rcon0=record.rcon0,
            zcon0=record.zcon0,
            lfreeb=True,
            jmax=int(self.solver.resolution.ns),
            presf_ns_scale=fbi._presf_ns_scale_traceable(params, self.inp, int(self.solver.resolution.ns)),
        )
        axis_r = jnp.full((self.solver.resolution.nzeta,), float(np.asarray(self.inp.rbc)[self.inp.ntor, 0]))
        basis, program, _ = fb._vacuum_executables(
            self.solver.resolution,
            mf=int(self.inp.mpol) + 1,
            nf=int(self.inp.ntor),
            signgs=int(self.rt.setup.signgs),
            wint=np.asarray(self.rt.trig.wint),
            modes=self.rt.modes,
            axis_r0=axis_r,
            axis_z0=jnp.zeros_like(axis_r),
            use_fft=False,
            solve_on_plasma_device=True,
        )
        field = self.solver.field_from_parameters(jnp.asarray(record.parameters))
        vacuum_values = program.full(record.state, export_rt, field)
        vacuum = fb._vacuum_output(
            fb.FreeBoundaryState(potvac=vacuum_values["potvac"], surface_fields=vacuum_values["surface_fields"]), basis
        )
        if vacuum is None or any(
            not np.all(np.isfinite(np.asarray(x)))
            for x in (vacuum.potsin, vacuum.bsubu, vacuum.bsubv, vacuum.bsupu, vacuum.bsupv)
        ):
            raise ValueError("snapshot requires finite fixed-geometry vacuum output")
        return wout_from_state(
            inp=self._inp_at(record.parameters),
            state=record.state,
            niter=record.result.iterations,
            fsqr=record.result.fsqr,
            fsqz=record.result.fsqz,
            fsql=record.result.fsql,
            vacuum_output=vacuum,
        )
