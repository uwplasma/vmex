"""Host-controlled dense adjoints of the canonical projected free-boundary root."""
from __future__ import annotations

import functools
import warnings
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np
import scipy.linalg as sl
from jax.flatten_util import ravel_pytree

from . import implicit as im


class _Space(NamedTuple):
    left: object
    right: object
    sign: object
    weight: object


def _active_space(cfg, mask, max_dofs):
    """Sparse orthonormal Q with QQ.T equal to main's DOF projector.

    Equal Z_sin pairs and opposite Z_cos pairs each supply one coordinate.
    The mask must already have equal support within each constrained pair.
    """
    flat = np.asarray(ravel_pytree(mask)[0])
    if not np.all(np.isfinite(flat)) or not np.all((flat == 0) | (flat == 1)):
        raise ValueError("dense adjoints require a finite binary DOF mask")
    active = flat.astype(bool).copy()
    pairs = []
    if bool(cfg.lconm1) and int(cfg.resolution.ntor) > 0:
        pos, neg = im._m1_pair_columns(cfg)
        offset = 0
        for name in im._STATE_FIELDS:
            shape = getattr(mask, name).shape
            sign = 1 if name == 'Z_sin' else -1
            if name == 'Z_sin' or (name == 'Z_cos' and cfg.resolution.lasym):
                for row in range(shape[0]):
                    for p, n in zip(pos, neg):
                        left, right = offset + row*shape[1] + int(p), offset + row*shape[1] + int(n)
                        if active[left] != active[right]:
                            raise ValueError("dense adjoint mask has unequal m=1 pair support")
                        if active[left]:
                            active[left] = active[right] = False
                            pairs.append((left, right, sign))
            offset += int(np.prod(shape))
    singles = np.flatnonzero(active)
    size = len(singles) + len(pairs)
    if not 0 < size <= max_dofs:
        raise ValueError(f"dense adjoint active dimension {size} must be in [1, {max_dofs}]; "
                         "adjust adjoint_dense_max_dofs explicitly for a larger matrix")
    left = np.asarray([*singles, *(p[0] for p in pairs)], dtype=np.int32)
    right = np.asarray([*singles, *(p[1] for p in pairs)], dtype=np.int32)
    sign = np.asarray([0]*len(singles) + [p[2] for p in pairs], dtype=np.float64)
    weight = np.asarray([1.]*len(singles) + [1/np.sqrt(2.)]*len(pairs), dtype=np.float64)
    return _Space(left, right, sign, weight)


def _expand(value, template, space):
    flat, unravel = ravel_pytree(template)
    weighted = value * space.weight
    result = jnp.zeros_like(flat).at[space.left].set(weighted)
    result = result.at[space.right].add(weighted * space.sign)
    return unravel(result)


def _compress(value, space):
    flat = ravel_pytree(value)[0]
    return space.weight * (flat[space.left] + space.sign * flat[space.right])


def _prepare_tangent(z, params, field, frozen, rcon, zcon, *, residual):
    # Main's residual is already compiled. Keep linearize host-eager: wrapping
    # the returned JVP closure in another jit can leak nested force-kernel
    # tracers through its static callable data on supported JAX versions.
    return jax.linearize(lambda x:residual(x, params, field, frozen, rcon, zcon), z)[1]


@jax.jit
def _columns(tangent, template, space, indices):
    size = space.left.shape[0]
    def column(index):
        seed = jax.nn.one_hot(index, size, dtype=ravel_pytree(template)[0].dtype)
        return _compress(tangent(_expand(seed, template, space)), space)
    return jax.vmap(column)(indices)


@functools.partial(jax.jit, static_argnames=('batch_size',))
def _assemble_device(tangent, template, space, *, batch_size):
    size = space.left.shape[0]
    chunks = (size + batch_size - 1)//batch_size
    def chunk(index):
        return _columns(tangent, template, space, index*batch_size+jnp.arange(batch_size))
    rows = jax.lax.map(chunk, jnp.arange(chunks)).reshape((-1, size))[:size]
    return rows.T


def _assemble_host(tangent, template, space, batch_size):
    size = space.left.shape[0]
    matrix = np.empty((size, size), dtype=np.asarray(ravel_pytree(template)[0]).dtype)
    for start in range(0, size, batch_size):
        count = min(batch_size, size-start)
        matrix[:,start:start+count] = np.asarray(
            _columns(tangent,template,space,jnp.arange(batch_size)+start))[:count].T
    return matrix


def _factor_solve(matrix, rhs, backend, *, return_factors=False):
    """One factorization, all transpose RHS columns, without assuming symmetry."""
    if backend == 'forward_dense':
        with warnings.catch_warnings():
            warnings.simplefilter('error', sl.LinAlgWarning)
            factors = sl.lu_factor(matrix)
            solution = sl.lu_solve(factors, np.asarray(rhs).T, trans=1).T
    else:
        factors = jsl.lu_factor(matrix)
        solution = jsl.lu_solve(factors, rhs.T, trans=1).T
    return (solution, factors) if return_factors else solution


def _refine_dense_solution(matrix, rhs, solution, factors, backend, tolerances):
    """Correct failed rows with the same LU; retain only residual improvements."""
    xp = np if backend == 'forward_dense' else jnp
    solve = sl.lu_solve if backend == 'forward_dense' else jsl.lu_solve
    rhs, solution = xp.asarray(rhs), xp.asarray(solution)
    defect = rhs - solution @ matrix
    norms = xp.linalg.norm(defect, axis=1)
    initial = np.asarray(norms).copy()
    steps = np.zeros(rhs.shape[0], dtype=int)
    for _ in range(3):
        failed = (norms > tolerances) & xp.isfinite(norms) & xp.all(xp.isfinite(solution), axis=1)
        if not bool(xp.any(failed)):
            break
        correction = solve(factors, xp.where(failed[:, None], defect, 0.).T, trans=1).T
        candidate = solution + correction
        candidate_defect = rhs - candidate @ matrix
        candidate_norms = xp.linalg.norm(candidate_defect, axis=1)
        improved = failed & (candidate_norms < norms) & xp.all(xp.isfinite(candidate), axis=1)
        steps += np.asarray(failed, dtype=int)
        if not bool(xp.any(improved)):
            break
        solution = xp.where(improved[:, None], candidate, solution)
        defect = xp.where(improved[:, None], candidate_defect, defect)
        norms = xp.where(improved, candidate_norms, norms)
    return solution, norms, initial, steps


@dataclass(eq=False)
class DenseRootLinearization:
    """Owned numerical factors and tape for exactly one certified root.

    Created only after successful adjoint certification. Its caller owns the
    accepted-state lifetime; there is no global numerical cache or slow fallback.
    """
    residual: object
    z: object
    params: object
    field: object
    frozen: object
    rcon: object
    zcon: object
    space: object
    factors: object
    action: object
    cfg: object
    _direction: object = None
    _response: object = None
    _rhs: object = None
    _tangent_backend = 'reused_dense_lu'

    def _solve_tangent(self, rhs):
        solve = sl.lu_solve if self.cfg.adjoint_solver == "forward_dense" else jsl.lu_solve
        solution = solve(self.factors, -_compress(rhs, self.space), trans=0)
        return _expand(solution, self.z, self.space), 1

    def offload_factors(self):
        """Move dense factors to immutable host storage between predictor calls."""
        if self.factors is None:
            raise ValueError("root linearization is closed")
        if not all(isinstance(x, np.ndarray) and not x.flags.writeable for x in self.factors):
            factors = tuple(np.array(x, copy=True) for x in self.factors)
            for value in factors:
                value.setflags(write=False)
            self.factors = factors

    def close(self):
        """Release root-specific arrays/tapes and permanently invalidate reuse."""
        for name in self.__dataclass_fields__:
            setattr(self, name, None)

    def tangent(self, direction, *, diagnostics=None):
        """Solve and independently certify the response to a field direction."""
        if self.factors is None:
            raise ValueError('dense root linearization is closed')
        if any(isinstance(x, jax.core.Tracer) for x in jax.tree.leaves(direction)):
            raise ValueError('dense root tangent requires host-eager inputs')
        vector = np.asarray(direction)
        if vector.shape != self.field.shape or not np.all(np.isfinite(vector)):
            raise ValueError('tangent direction must be finite and match field parameters')
        if np.iscomplexobj(vector):
            raise ValueError('tangent direction must be real')
        scale = None
        if self._direction is not None:
            index = int(np.argmax(np.abs(self._direction)))
            if self._direction.flat[index] != 0:
                candidate = float(vector.flat[index]/self._direction.flat[index])
                # Backtracking uses exact binary halvings. A changed direction,
                # even a nearby one, needs its own RHS and direct solve.
                if np.array_equal(vector, candidate*self._direction):
                    scale = candidate
            elif not np.any(vector):
                scale = 0.
        with im._device_context(self.cfg.implicit):
            if scale is None:
                direction = im._device_pin(self.cfg.implicit, jnp.asarray(vector))
                rhs = jax.jvp(lambda p:self.residual(self.z,self.params,p,self.frozen,self.rcon,self.zcon),
                              (self.field,), (direction,))[1]
                response, iterations = self._solve_tangent(rhs)
            else:
                response = jax.tree.map(lambda x:scale*x,self._response)
                rhs = jax.tree.map(lambda x:scale*x,self._rhs)
                iterations = 0
            # Independent full matrix-free residual, also for scaled responses.
            defect = jax.tree.map(lambda ax,b:ax+b,self.action(response),rhs)
            norm = float(jnp.linalg.norm(ravel_pytree(defect)[0]))
            rhs_norm = float(jnp.linalg.norm(ravel_pytree(rhs)[0]))
            rtol = self.cfg.adjoint_residual_rtol
            tolerance = (float(im._adjoint_acceptance(self.cfg.implicit,rhs_norm))
                         if rtol is None else rtol*rhs_norm)
            finite = all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(response))
            accepted = finite and np.isfinite(norm) and norm <= tolerance
            if diagnostics is not None:
                diagnostics.append(dict(row=None,residual_norm=norm,rhs_norm=rhs_norm,
                    relative_residual=norm/rhs_norm if rhs_norm else (0. if norm == 0 else float('inf')),
                    tolerance=tolerance,iterations=iterations,accepted=bool(accepted),
                    backend=self._tangent_backend,scaled_reuse=scale is not None))
            if not accepted:
                im._raise_adjoint_unconverged(self.cfg.implicit,iterations=iterations,
                    residual_norm=norm,tolerance=tolerance,method=self._tangent_backend+' tangent')
            if scale is None:
                self._direction = vector.copy()
                self._response, self._rhs = response, rhs
            return response


def solve_dense_adjoint(residual, z, params, field, frozen, rcon, zcon,
                        rhs_batch, mask, cfg, *, diagnostics=None, return_linearization=False):
    """Forward assembly and certified transpose solve for scalar or shared callers.

    This interface is host-eager even for the JAX matrix backend: the runtime
    mask determines the active dimension. No Krylov fallback is substituted.
    """
    if return_linearization and (cfg.adjoint_solver not in {'forward_dense', 'forward_dense_jax'} or cfg.adjoint_fail != 'error'):
        raise ValueError('retained dense linearization requires a dense backend with adjoint_fail=error')
    values = (z,params,field,frozen,rcon,zcon,rhs_batch,mask)
    if any(isinstance(x,jax.core.Tracer) for x in jax.tree.leaves(values)):
        raise ValueError("dense free-boundary adjoints require host-eager inputs; "
                         "do not wrap the scalar gradient in jax.jit")
    dtype = ravel_pytree(z)[0].dtype
    if np.dtype(dtype) != np.dtype(np.float64):
        raise TypeError("dense free-boundary adjoints require float64; enable JAX x64")
    space = _active_space(cfg.implicit, mask, cfg.adjoint_dense_max_dofs)
    # Numerical arrays follow the caller's selected device; nothing is cached
    # across roots, masks, or changes in the constraint baselines.
    space = jax.tree.map(jnp.asarray, space)
    batch_size = min(cfg.adjoint_dense_batch_size, space.left.size)
    tangent = _prepare_tangent(z,params,field,frozen,rcon,zcon,residual=residual)
    if cfg.adjoint_solver == 'forward_dense':
        matrix = _assemble_host(tangent,z,space,batch_size)
    else:
        matrix = _assemble_device(tangent,z,space,batch_size=batch_size)
    rhs = jax.vmap(lambda value:_compress(value,space))(rhs_batch)
    def reject(row, norm, tolerance):
        im._raise_adjoint_unconverged(cfg.implicit,iterations=1,
            residual_norm=float(norm),tolerance=float(tolerance),
            method=f'{cfg.adjoint_solver} row {row}')
    if not bool(jnp.all(jnp.isfinite(matrix))):
        reject(0,np.inf,0.)
    try:
        solution, factors = _factor_solve(matrix,rhs,cfg.adjoint_solver,return_factors=True)
    except (np.linalg.LinAlgError, sl.LinAlgWarning):
        reject(0,np.inf,0.)
    # Check the actual dense equation after the solve, independently of the
    # factorization's status. Singular/nonfinite solutions always fail closed.
    rhs_norms = jnp.linalg.norm(rhs,axis=1)
    residual_rtol = getattr(cfg, 'adjoint_residual_rtol', None)
    tolerances = (im._adjoint_acceptance(cfg.implicit,rhs_norms) if residual_rtol is None
                  else residual_rtol * rhs_norms)
    solution, norms, initial_norms, refinement_steps = _refine_dense_solution(
        matrix, rhs, solution, factors, cfg.adjoint_solver, np.asarray(tolerances))
    for row in range(rhs.shape[0]):
        finite = bool(jnp.isfinite(norms[row]) & jnp.all(jnp.isfinite(solution[row])))
        accepted = finite and float(norms[row]) <= float(tolerances[row])
        if diagnostics is not None:
            report = im._adjoint_diagnostic(cfg.implicit,
                residual_norm=norms[row], rhs_norm=rhs_norms[row], iterations=1,
                row=row, backend=cfg.adjoint_solver, residual_rtol=residual_rtol, finite=finite)
            report.update(initial_residual_norm=float(initial_norms[row]),
                          refinement_steps=int(refinement_steps[row]))
            diagnostics.append(report)
        if not accepted:
            if cfg.adjoint_fail != 'best_effort' or not finite:
                reject(row,norms[row],tolerances[row])
            warnings.warn(f'{cfg.adjoint_solver} row {row} residual exceeds acceptance; '
                          "returning an inaccurate best-effort adjoint",RuntimeWarning,stacklevel=2)
    adjoints = jax.vmap(lambda value:_expand(value,z,space))(jnp.asarray(solution))
    if return_linearization:
        return adjoints, DenseRootLinearization(
            residual,z,params,field,frozen,rcon,zcon,space,factors,tangent,cfg)
    return adjoints


BATCH_SIZES = (32, 64)


def tune_adjoint_batch(evaluate, *, deadline, event, clock=None):
    """Return (batch, owned linearization, report) at one unchanged root.

    ``evaluate(batch)`` must return a certified native linearization. Compile
    both shapes, then measure three pairs in alternating order. Choose 64 only
    if it beats 32 by at least 5% in every pair. The first 32 gradient is the
    numerical reference; no candidate may change any gradient row by 1e-8.
    All objects except the returned winner are closed, including on failure.
    No timings or numerical factors are shared across runs/devices.
    """
    import statistics
    import time
    import numpy as np

    clock = time.monotonic if clock is None else clock
    roots, records = {}, []
    reference = None
    started = clock()
    baseline, candidate = BATCH_SIZES
    try:
        # Discard the first call for each static batch shape (includes JIT).
        schedule = [(batch, None) for batch in BATCH_SIZES]
        schedule += [(batch, pair) for pair, order in enumerate(
            (BATCH_SIZES, BATCH_SIZES[::-1], BATCH_SIZES)) for batch in order]
        for batch, pair in schedule:
            if clock() >= deadline:
                raise TimeoutError('walltime during adjoint batch tuning')
            if batch in roots:
                roots.pop(batch).close()
            event('adjoint_batch_probe_start', batch=batch, pair=pair)
            start = clock()
            root = evaluate(batch)
            roots[batch] = root
            jac = np.asarray(root.field_jacobian)  # synchronize before timing
            seconds = clock() - start
            if clock() >= deadline:
                raise TimeoutError('walltime during adjoint batch tuning')
            if not np.isfinite(seconds) or seconds <= 0:
                raise ValueError('invalid adjoint batch timing')
            if not np.all(np.isfinite(jac)):
                raise ValueError('nonfinite adjoint batch gradient')
            if reference is None:
                reference = jac.copy()
            if jac.shape != reference.shape:
                raise ValueError('adjoint batch gradient shape changed')
            norm = np.linalg.norm(reference, axis=1)
            difference = np.linalg.norm(jac - reference, axis=1)
            if not np.all(np.isfinite(norm)) or not np.all(np.isfinite(difference)):
                raise ValueError('nonfinite adjoint batch gradient norm')
            # Zero reference rows require exact agreement; never hide them
            # behind an absolute tolerance or a division-by-zero workaround.
            relative = np.divide(difference, norm, out=np.zeros_like(norm), where=norm > 0)
            if np.any((norm == 0) & (difference != 0)) or np.any(relative > 1e-8):
                raise ValueError('adjoint batch gradient agreement failed')
            record = dict(batch=batch, pair=pair, warmup=pair is None,
                          seconds=seconds, maximum_gradient_row_relative_difference=float(max(relative)))
            records.append(record)
            event('adjoint_batch_probe', **record)
        timings = {batch: [r['seconds'] for r in records
                          if r['batch'] == batch and not r['warmup']] for batch in BATCH_SIZES}
        ratios = [b / a for a, b in zip(timings[baseline], timings[candidate])]
        selected = candidate if all(r < 0.95 for r in ratios) else baseline
        report = dict(requested='auto', selected_batch_size=selected,
                      candidates=list(BATCH_SIZES), records=records, paired_64_over_32=ratios,
                      warm_median_seconds={str(b): statistics.median(t) for b, t in timings.items()},
                      minimum_consistent_speedup_fraction=0.05,
                      gradient_agreement_rtol=1e-8, elapsed_seconds=clock() - started,
                      reason='consistent_gain' if selected == candidate else 'no_consistent_gain')
        event('adjoint_batch_selected', **report)
        winner = roots.pop(selected)
        return selected, winner, report
    finally:
        for root in roots.values():
            root.close()
