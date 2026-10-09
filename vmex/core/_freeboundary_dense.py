"""Dense and seed-LU matrix-free adjoints of the projected free-boundary root.

The dense path factors the active-space Jacobian at each root. The
matrix-free path reuses only an explicitly retained seed LU as a GMRES
preconditioner; acceptance checks always use the full current operator.
A structured seed (:class:`StructuredFactors`) replaces the dense LU by the
radial block-tridiagonal factors plus NESTOR's low-rank coupling, O(ns)
instead of O(ns^2) storage, under the same full-operator gates.
"""
from __future__ import annotations

import dataclasses
import functools
from functools import partial
import warnings
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np
import scipy.linalg as sl
from jax.flatten_util import ravel_pytree
from solvax import block_thomas_solve
from solvax.krylov import gmres

from . import implicit as im
from .errors import AdjointSolveError


class _Space(NamedTuple):
    left: object
    right: object
    sign: object
    weight: object


def _active_space(cfg, mask, max_dofs):
    """Sparse orthonormal Q with QQ.T equal to :func:`vmex.core.implicit._dof_projector`.

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


def _max_dofs(cfg):
    """Active-dimension cap: the dense matrix's, none for structured factors."""
    return np.inf if getattr(cfg, "adjoint_factorization", "dense") == "structured" else cfg.adjoint_dense_max_dofs


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
    # The projected residual is already compiled. Keep linearize host-eager: wrapping
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
    """Return the transpose of the active Jacobian, written in place row by row.

    Row i is column i of the Jacobian. One n-by-n buffer is live; the last
    batch is shifted back to end at n, recomputing a few rows, so no padded
    copy or transpose is materialized.
    """
    size = space.left.shape[0]
    batch_size = min(batch_size, size)
    chunks = (size + batch_size - 1)//batch_size
    dtype = ravel_pytree(template)[0].dtype

    def body(index, transpose):
        start = jnp.minimum(index*batch_size, size-batch_size)
        rows = _columns(tangent, template, space, start+jnp.arange(batch_size))
        return jax.lax.dynamic_update_slice(transpose, rows, (start, 0))

    return jax.lax.fori_loop(0, chunks, body, jnp.zeros((size, size), dtype))


RADIAL_BANDWIDTH = 2  # coupled raw rows reach two surfaces at the free boundary
#: Largest relative ``J v`` mismatch at which a colored assembly is accepted;
#: above it the Jacobian is rebuilt column by column.
_COLORED_ASSEMBLY_RTOL = 1e-9


def _radial_probes(template, space, bandwidth=RADIAL_BANDWIDTH, edge_axis=False):
    """Group active coordinates whose radial surfaces are more than 2*bandwidth apart.

    The raw coupled Jacobian is banded in radius, so one JVP of the summed seed
    recovers every column of the group. With a prescribed plasma current the axis
    unknowns also reach the edge equations; ``edge_axis`` gives the axis surface a
    color of its own, so those rows see no other column. Returns (seeds, probe_of, surface).
    """
    leaves = jax.tree.leaves(template)
    ns, mnmax = leaves[0].shape
    surface = (np.asarray(space.left) % (ns * mnmax)) // mnmax
    colors = 2 * bandwidth + 1
    color = np.where(edge_axis & (surface == 0), colors, surface % colors)
    rank = np.zeros(surface.size, dtype=int)
    for s in np.unique(surface):
        members = np.flatnonzero(surface == s)
        rank[members] = np.arange(members.size)
    key = color * (rank.max() + 1) + rank
    groups, probe_of = np.unique(key, return_inverse=True)
    seeds = np.zeros((groups.size, surface.size))
    seeds[probe_of, np.arange(surface.size)] = 1.0
    return seeds, probe_of, surface


@functools.partial(jax.jit, static_argnames=('batch_size', 'bandwidth', 'edge_axis'))
def _assemble_colored(tangent, template, space, seeds, probe_of, surface, *, batch_size, bandwidth,
                      edge_axis=False):
    """Return the Jacobian transpose from radially colored JVP probes."""
    def probe(seed):
        return _compress(tangent(_expand(seed, template, space)), space)
    rows = jax.lax.map(probe, seeds, batch_size=batch_size)
    band = jnp.abs(surface[:, None] - surface[None, :]) <= bandwidth  # [column, row]
    if edge_axis:  # axis unknowns (columns) also reach the edge equations (rows)
        edge = jax.tree.leaves(template)[0].shape[0] - 1
        band = band | ((surface[:, None] == 0) & (surface[None, :] == edge))
    return jnp.where(band, rows[probe_of], 0.0)


def _assembly_error(tangent, template, space, transpose, *, probes=2, seed=0):
    """Relative mismatch of assembled J v against independent JVPs at random v."""
    size = space.left.shape[0]
    vectors = jax.random.normal(jax.random.PRNGKey(seed), (probes, size), dtype=transpose.dtype)
    exact = jax.vmap(lambda v: _compress(tangent(_expand(v, template, space)), space))(vectors)
    return float(jnp.max(jnp.linalg.norm(vectors @ transpose - exact, axis=1)
                         / jnp.maximum(jnp.linalg.norm(exact, axis=1), 1e-300)))


# (ns, mnmax, active size) -> whether the edge-axis coloring was needed there last time
_EDGE_AXIS_LAYOUT: dict[tuple, bool] = {}


def _assemble(tangent, template, space, *, batch_size):
    """Colored assembly for banded state-space Jacobians, verified; else column by column."""
    from .solver import SpectralState
    if isinstance(template, SpectralState):
        key = (*jax.tree.leaves(template)[0].shape, space.left.shape[0])
        error = None
        for edge_axis in (True,) if _EDGE_AXIS_LAYOUT.get(key) else (False, True):
            seeds, probe_of, surface = _radial_probes(template, space, edge_axis=edge_axis)
            transpose = _assemble_colored(tangent, template, space, jnp.asarray(seeds), jnp.asarray(probe_of),
                                          jnp.asarray(surface), batch_size=batch_size,
                                          bandwidth=RADIAL_BANDWIDTH, edge_axis=edge_axis)
            error = _assembly_error(tangent, template, space, transpose)
            if np.isfinite(error) and error <= _COLORED_ASSEMBLY_RTOL:
                _EDGE_AXIS_LAYOUT[key] = edge_axis
                return transpose
            del transpose
        warnings.warn(f"colored Jacobian assembly mismatch {error:.2e}; assembling every column",
                      RuntimeWarning, stacklevel=2)
    return _assemble_device(tangent, template, space, batch_size=batch_size)


def _factor_solve(matrix, rhs):
    """Factor the Jacobian transpose once and solve every RHS row.

    Returns ``(solution, factors)``; the LU is retained for every later solve
    at this root. No symmetry is assumed.
    """
    factors = jsl.lu_factor(matrix)
    # A CUDA factorization can return an invalid pivot buffer even when the
    # LU entries are finite. Never feed sentinel indices to lu_solve: refactor
    # the identical matrix on the host. All residual gates still apply.
    pivots = np.asarray(factors[1])
    size = matrix.shape[0]
    if not (pivots.shape == (size,) and np.issubdtype(pivots.dtype, np.integer)
            and np.all((pivots >= np.arange(size)) & (pivots < size))):
        warnings.warn("GPU LU returned invalid pivot indices; refactoring the same matrix on CPU",
                      RuntimeWarning, stacklevel=2)
        del factors
        with warnings.catch_warnings():
            warnings.simplefilter('error', sl.LinAlgWarning)
            factors = sl.lu_factor(np.asarray(matrix))
            solution = jnp.asarray(sl.lu_solve(factors, np.asarray(rhs).T, trans=0).T)
    else:
        solution = jsl.lu_solve(factors, rhs.T, trans=0).T
    return solution, factors


_OPERATOR_NORM_ITERATIONS = 20  # power iterations for the backward-error scale ||A||_2
# Without a matrix each iteration applies the full coupled operator twice
# (~0.3 s at ns = 51). Five iterations reach 82-96% of the 20-iteration bound at
# finite-beta QA roots (early to late in a run), a gate only that much stricter:
# there the rows pass it by more than 10^6, and each step takes ~9 s less.
_OPERATOR_NORM_MATVEC_ITERATIONS = 5


def _operator_norm(matrix):
    """A lower bound on ``||A||_2`` from a fixed-start power iteration.

    It scales the backward-error gate below; underestimating ``||A||`` only
    makes that gate stricter.
    """
    return _operator_norm_matvec(lambda v: matrix @ v, lambda v: matrix.T @ v, matrix.shape[1], matrix.dtype)


def _operator_norm_matvec(apply, apply_transpose, size, dtype, iterations=_OPERATOR_NORM_ITERATIONS):
    """:func:`_operator_norm` of the operator ``apply`` with transpose ``apply_transpose``."""
    vector = jnp.ones(size, dtype) / np.sqrt(size)
    for _ in range(iterations):
        product = apply_transpose(apply(vector))
        norm = float(jnp.linalg.norm(product))
        if not np.isfinite(norm) or norm == 0.0:
            return 0.0
        vector = product / norm
    return float(jnp.linalg.norm(apply(vector)))


def _acceptance(cfg, rhs_norm, solution_norm, operator_norm):
    """Largest accepted residual: the normwise backward-error gate.

    ``||r|| <= tol (||A|| ||x|| + ||b||)``, where ``tol ||b||`` is the shared
    relative-residual acceptance (``10 adjoint_tol``, or
    ``adjoint_residual_rtol``). It is never tighter than that relative gate, and
    it still bounds the forward error by ``cond(A) tol``; a row whose
    ``||b||`` is small against ``||A|| ||x||`` no longer fails at the rounding
    floor of ``A x``.
    """
    scale = rhs_norm + operator_norm * solution_norm
    rtol = getattr(cfg, "adjoint_residual_rtol", None)
    return im._adjoint_acceptance(cfg.implicit, scale) if rtol is None else rtol * scale


def _backward_error_report(cfg, *, residual_norm, rhs_norm, solution_norm, operator_norm, **details):
    """``im._adjoint_diagnostic`` judged by :func:`_acceptance`, with its norms."""
    norm, rhs_norm = float(residual_norm), float(rhs_norm)
    solution_norm, operator_norm = float(solution_norm), float(operator_norm)
    scale = rhs_norm + operator_norm * solution_norm
    report = im._adjoint_diagnostic(cfg.implicit, residual_norm=norm, rhs_norm=scale,
                                    residual_rtol=getattr(cfg, "adjoint_residual_rtol", None), **details)
    report.update(rhs_norm=rhs_norm, solution_norm=solution_norm, operator_norm=operator_norm,
                  relative_residual=norm / rhs_norm if rhs_norm else (0.0 if norm == 0 else float("inf")),
                  backward_error=norm / scale if scale else (0.0 if norm == 0 else float("inf")))
    return report


def _refine_dense_solution(matrix, rhs, solution, factors, tolerances):
    """Correct failed rows with the same LU; retain only residual improvements."""
    xp, solve = jnp, jsl.lu_solve
    rhs, solution = xp.asarray(rhs), xp.asarray(solution)
    defect = rhs - solution @ matrix.T
    norms = xp.linalg.norm(defect, axis=1)
    initial = np.asarray(norms).copy()
    steps = np.zeros(rhs.shape[0], dtype=int)
    for _ in range(3):
        failed = (norms > tolerances) & xp.isfinite(norms) & xp.all(xp.isfinite(solution), axis=1)
        if not bool(xp.any(failed)):
            break
        correction = solve(factors, xp.where(failed[:, None], defect, 0.).T, trans=0).T
        candidate = solution + correction
        candidate_defect = rhs - candidate @ matrix.T
        candidate_norms = xp.linalg.norm(candidate_defect, axis=1)
        improved = failed & (candidate_norms < norms) & xp.all(xp.isfinite(candidate), axis=1)
        steps += np.asarray(failed, dtype=int)
        if not bool(xp.any(improved)):
            break
        solution = xp.where(improved[:, None], candidate, solution)
        defect = xp.where(improved[:, None], candidate_defect, defect)
        norms = xp.where(improved, candidate_norms, norms)
    return solution, norms, initial, steps


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class StructuredFactors:
    """Inverse of the raw coupled Jacobian ``J = A + P Q`` at the root where it was built.

    ``A`` is the radial block tridiagonal at frozen vacuum pressure, held as
    block-Thomas factors of its two-sided scaling ``row_scale A column_scale``;
    ``P Q`` is NESTOR's low-rank edge coupling, added by the Woodbury identity
    with ``bulk_p = A^-1 P``, ``q = Q`` and ``lu`` the factors of
    ``I + Q A^-1 P``. Storage is O(ns) where the dense LU is O(ns^2). Rows are
    packed surface by surface from ``fields``; inactive coordinates carry an
    identity equation. Built by
    :func:`vmex.core.freeboundary_implicit._structured_factors`.
    """

    blocks: object
    bulk_p: object
    q: object
    lu: object
    row_scale: object
    column_scale: object
    fields: tuple = dataclasses.field(metadata=dict(static=True))


def _structured_inverse(factors, packed, transpose):
    """``J^-1 packed`` (``J^-T`` with ``transpose``) for flat packed rows."""
    shape, dtype = factors.row_scale.shape, packed.dtype
    if not transpose:
        scaled = packed.reshape(shape) * factors.row_scale
        bulk = block_thomas_solve(factors.blocks, scaled[..., None])[..., 0].astype(dtype)
        bulk = (bulk * factors.column_scale).ravel()
        return bulk - factors.bulk_p @ jsl.lu_solve(factors.lu, factors.q @ bulk)
    # J^-1 = (I - Bp C^-1 Q) S with S = diag(cs) T^-1 diag(rs), so J^-T = S^T (I - Q^T C^-T Bp^T).
    corrected = packed - factors.q.T @ jsl.lu_solve(factors.lu, factors.bulk_p.T @ packed, trans=1)
    scaled = corrected.reshape(shape) * factors.column_scale
    bulk = block_thomas_solve(factors.blocks, scaled[..., None], transpose=True)[..., 0].astype(dtype)
    return (bulk * factors.row_scale).ravel()


def _precondition(factors, template, space, value, *, transpose):
    """Apply retained factors to an active-space vector: ``J^-T`` with ``transpose``, else ``J^-1``."""
    if isinstance(factors, StructuredFactors):
        tree = _expand(value, template, space)
        packed = jnp.concatenate([getattr(tree, name) for name in factors.fields], axis=1).ravel()
        rows = _structured_inverse(factors, packed, transpose).reshape(factors.row_scale.shape)
        parts = dict(zip(factors.fields, jnp.split(rows, len(factors.fields), axis=1)))
        return _compress(type(tree)(**{name: parts.get(name, jnp.zeros_like(getattr(tree, name)))
                                       for name in im._STATE_FIELDS}), space)
    # Dense factors are of the Jacobian transpose: a transposed solve uses trans=0.
    return jsl.lu_solve(factors, value, trans=int(not transpose))


def _frozen_host_copy(value):
    copy = np.array(value, copy=True)
    copy.setflags(write=False)
    return copy


@dataclass(eq=False)
class DenseRootLinearization:
    """Owned numerical factors and tape for exactly one certified root.

    Created only after successful adjoint certification. Its caller owns the
    accepted-state lifetime; nothing is cached globally across roots.
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
    operator_norm: float = 0.0  # ||A||_2 estimate of the dense operator, for the backward-error gate
    fresh_factors: bool = True  # the factors were built at this root (a seed may be taken from it)
    _direction: object = None
    _response: object = None
    _rhs: object = None
    _tangent_backend = 'reused_dense_lu'

    def _solve_tangent(self, rhs):
        solution = jsl.lu_solve(self.factors, -_compress(rhs, self.space), trans=1)
        return _expand(solution, self.z, self.space), 1

    def offload_factors(self):
        """Move dense factors to immutable host storage between predictor calls."""
        if self.factors is None:
            raise ValueError("root linearization is closed")
        if not all(isinstance(x, np.ndarray) and not x.flags.writeable for x in jax.tree.leaves(self.factors)):
            self.factors = jax.tree.map(_frozen_host_copy, self.factors)

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
            finite = all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(response))
            report = _backward_error_report(self.cfg, residual_norm=norm, rhs_norm=rhs_norm,
                solution_norm=float(jnp.linalg.norm(ravel_pytree(response)[0])),
                operator_norm=self.operator_norm, iterations=iterations,
                backend=self._tangent_backend, finite=finite, scaled_reuse=scale is not None)
            if diagnostics is not None:
                diagnostics.append(report)
            if not report["accepted"]:
                im._raise_adjoint_unconverged(self.cfg.implicit,iterations=iterations,
                    residual_norm=norm,tolerance=report["tolerance"],
                    method=self._tangent_backend+' tangent')
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
    if return_linearization and (cfg.adjoint_solver != 'forward_dense_jax' or cfg.adjoint_fail != 'error'):
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
    matrix = _assemble(tangent,z,space,batch_size=batch_size)
    rhs = jax.vmap(lambda value:_compress(value,space))(rhs_batch)
    def reject(row, norm, tolerance):
        im._raise_adjoint_unconverged(cfg.implicit,iterations=1,
            residual_norm=float(norm),tolerance=float(tolerance),
            method=f'{cfg.adjoint_solver} row {row}')
    if not bool(jnp.all(jnp.isfinite(matrix))):
        reject(0,np.inf,0.)
    try:
        solution, factors = _factor_solve(matrix,rhs)
    except (np.linalg.LinAlgError, sl.LinAlgWarning):
        reject(0,np.inf,0.)
    # Check the actual dense equation after the solve, independently of the
    # factorization's status. Singular/nonfinite solutions always fail closed.
    operator_norm = _operator_norm(matrix)
    rhs_norms = jnp.linalg.norm(rhs,axis=1)
    tolerances = _acceptance(cfg, rhs_norms, jnp.linalg.norm(solution, axis=1), operator_norm)
    solution, norms, initial_norms, refinement_steps = _refine_dense_solution(
        matrix, rhs, solution, factors, np.asarray(tolerances))
    solution_norms = jnp.linalg.norm(solution, axis=1)
    for row in range(rhs.shape[0]):
        finite = bool(jnp.isfinite(norms[row]) & jnp.all(jnp.isfinite(solution[row])))
        report = _backward_error_report(cfg, residual_norm=norms[row], rhs_norm=rhs_norms[row],
            solution_norm=solution_norms[row], operator_norm=operator_norm, iterations=1,
            row=row, backend=cfg.adjoint_solver, finite=finite)
        report.update(initial_residual_norm=float(initial_norms[row]),
                      refinement_steps=int(refinement_steps[row]))
        if diagnostics is not None:
            diagnostics.append(report)
        if not report["accepted"]:
            if cfg.adjoint_fail != 'best_effort' or not finite:
                reject(row,norms[row],report["tolerance"])
            warnings.warn(f'{cfg.adjoint_solver} row {row} residual exceeds acceptance; '
                          "returning an inaccurate best-effort adjoint",RuntimeWarning,stacklevel=2)
    adjoints = jax.vmap(lambda value:_expand(value,z,space))(jnp.asarray(solution))
    if return_linearization:
        return adjoints, DenseRootLinearization(
            residual,z,params,field,frozen,rcon,zcon,space,factors,tangent,cfg,operator_norm=operator_norm)
    return adjoints

def _signature(tree):
    return jax.tree.structure(tree), [(x.shape, x.dtype) for x in jax.tree.leaves(tree)]


def _rhs_key(value):
    """Identify exact RHS equality while treating signed zeros as equal."""
    canonical = np.array(value, copy=True)
    canonical[canonical == 0] = 0.0
    return canonical.tobytes()


@dataclass(eq=False)
class SeedLU:
    """An immutable host copy of certified factors, independent of root lifetime."""

    factors: object
    space: object
    signature: object
    field_shape: tuple
    cfg: object
    rtol: float
    restart: int
    max_restarts: int
    rhs_batch_size: int = 1
    operator_norm: float = 0.0  # ||A||_2 estimate at the seed root, scaling the backward-error gate

    @classmethod
    def from_root(
        cls,
        root,
        *,
        rtol=1e-11,
        restart=30,
        max_restarts=10,
        rhs_batch_size=1,
    ):
        """Snapshot a dense seed without retaining its numerical differentiation tape."""
        if not isinstance(root, DenseRootLinearization) or not root.fresh_factors or root.factors is None:
            raise ValueError("a live dense linearization (or structured factors built at its root) is required "
                             "to create a seed LU")
        if not np.isfinite(rtol) or not 0 < rtol < 1:
            raise ValueError("Krylov rtol must be finite and in (0, 1)")
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (restart, max_restarts)):
            raise ValueError("restart and max_restarts must be positive integers")
        if isinstance(rhs_batch_size, bool) or not isinstance(rhs_batch_size, int) or not 1 <= rhs_batch_size <= 8:
            raise ValueError("rhs_batch_size must be an integer in [1, 8]")
        factors = jax.tree.map(_frozen_host_copy, root.factors)
        space = jax.tree.map(_frozen_host_copy, root.space)
        if not isinstance(factors, StructuredFactors) and factors[0].dtype != np.float64:
            raise TypeError("seed LU requires float64")
        return cls(
            factors=factors,
            space=space,
            signature=_signature(root.z),
            field_shape=root.field.shape,
            cfg=root.cfg,
            rtol=rtol,
            restart=restart,
            max_restarts=max_restarts,
            rhs_batch_size=rhs_batch_size,
            operator_norm=root.operator_norm,
        )

    def validate(self, z, field, space, cfg):
        """Reject closed seeds and changes to the solver, state layout or active basis."""
        if self.factors is None:
            raise ValueError("seed LU preconditioner is closed")
        if cfg is not self.cfg or _signature(z) != self.signature or field.shape != self.field_shape:
            raise ValueError("seed LU belongs to a different solver or state layout")
        if any(not np.array_equal(a, b) for a, b in zip(space, self.space)):
            raise ValueError("seed LU active space differs from the current root")

    def close(self):
        """Release the seed; already-created root linearizations remain independent."""
        self.factors = self.space = self.signature = self.cfg = None


@jax.jit
def _forward_from_transpose(transpose, template, vector):
    return jax.linear_transpose(lambda value: transpose(value)[0], template)(vector)[0]


def prepare(z, params, field, frozen, rcon, zcon, *, residual):
    """Keep tape arrays dynamic so changed roots do not create new JIT callables."""
    from .freeboundary_implicit import _prepare_linearized_transpose

    transpose = _prepare_linearized_transpose(z, params, field, frozen, rcon, zcon, residual=residual)
    return jax.tree_util.Partial(_forward_from_transpose, transpose, z)


@partial(jax.jit, static_argnames=("transpose", "rtol", "restart", "max_restarts", "return_info"))
def solve(action, template, space, factors, rhs, *, transpose, rtol, restart, max_restarts, return_info=False):
    """Bounded right-preconditioned FGMRES; the caller certifies the full equation."""

    def operator(vector):
        value = _expand(vector, template, space)
        result = jax.linear_transpose(action, template)(value)[0] if transpose else action(value)
        return _compress(result, space)

    answer = gmres(
        operator,
        rhs,
        precond=lambda value: _precondition(factors, template, space, value, transpose=transpose),
        rtol=rtol,
        atol=0.0,
        restart=min(restart, rhs.size),
        max_restarts=max_restarts,
    )
    if return_info:
        return answer.x, answer.iterations, answer.residual_norm, answer.converged
    return answer.x, answer.iterations


@partial(jax.jit, static_argnames=("rtol", "restart", "max_restarts"))
def _solve_many(action, template, space, factors, rhs, *, rtol, restart, max_restarts):
    """Batch independent solves, retaining each row's stopping diagnostics."""
    return jax.vmap(
        lambda vector: solve(
            action,
            template,
            space,
            factors,
            vector,
            transpose=True,
            rtol=rtol,
            restart=restart,
            max_restarts=max_restarts,
            return_info=True,
        )
    )(rhs)


def _solve_unique(action, template, space, factors, reduced, *, batch_size, **options):
    """Group distinct nonzero rows without solving equal or opposite RHS twice."""
    unique, seen = [], set()
    for vector in reduced:
        host = np.asarray(vector)
        if not np.all(np.isfinite(host)):
            raise ValueError("nonfinite adjoint right-hand side")
        key, opposite = _rhs_key(host), _rhs_key(-host)
        if np.any(host) and key not in seen and opposite not in seen:
            unique.append((key, vector))
            seen.add(key)
    solved = {}
    for start in range(0, len(unique), batch_size):
        group = unique[start : start + batch_size]
        result = _solve_many(action, template, space, factors, jnp.stack([v for _, v in group]), **options)
        for index, (key, _) in enumerate(group):
            solved[key] = tuple(value[index] for value in result)
    return solved


@dataclass(eq=False)
class MatrixFreeRootLinearization(DenseRootLinearization):
    """Retain a current-root tape and the seed factors for checked predictor solves."""

    rtol: float = 1e-11
    restart: int = 30
    max_restarts: int = 10
    fresh_factors: bool = False  # True only when the seed's structured factors were built at this root
    _tangent_backend = "matrixfree_seed_lu"

    def _solve_tangent(self, rhs):
        solution, iterations = solve(
            self.action,
            self.z,
            self.space,
            self.factors,
            -_compress(rhs, self.space),
            transpose=False,
            rtol=self.rtol,
            restart=self.restart,
            max_restarts=self.max_restarts,
        )
        return _expand(solution, self.z, self.space), int(iterations)


def solve_matrixfree_adjoint(
    residual,
    z,
    params,
    field,
    frozen,
    rcon,
    zcon,
    rhs_batch,
    mask,
    cfg,
    *,
    preconditioner,
    diagnostics=None,
    return_linearization=False,
):
    """Solve arbitrary RHS batches; reuse only exact equal/opposite RHS solutions."""
    if cfg.adjoint_solver != "forward_dense_jax" or cfg.adjoint_fail != "error":
        raise ValueError("seed LU requires forward_dense_jax with adjoint_fail=error")
    space = _active_space(cfg.implicit, mask, _max_dofs(cfg))
    preconditioner.validate(z, field, space, cfg)
    space = jax.tree.map(jnp.asarray, space)
    factors = jax.tree.map(jnp.asarray, preconditioner.factors)
    action = prepare(z, params, field, frozen, rcon, zcon, residual=residual)
    reduced = jax.vmap(lambda value: _compress(value, space))(rhs_batch)
    options = dict(rtol=preconditioner.rtol, restart=preconditioner.restart, max_restarts=preconditioner.max_restarts)
    solved = (
        _solve_unique(action, z, space, factors, reduced, batch_size=preconditioner.rhs_batch_size, **options)
        if preconditioner.rhs_batch_size > 1
        else None
    )
    adjoints, cache = [], {}
    for row in range(reduced.shape[0]):
        vector = reduced[row]
        host = np.asarray(vector)
        if not np.all(np.isfinite(host)):
            raise ValueError("nonfinite adjoint right-hand side")
        key, opposite = _rhs_key(host), _rhs_key(-host)
        reused = key in cache or opposite in cache
        if not np.any(host):
            solution, iterations = jnp.zeros_like(vector), 0
            krylov_norm, converged = 0.0, True
        elif reused:
            solution, krylov_norm, converged = cache[key] if key in cache else cache[opposite]
            if key not in cache:
                solution = -solution
            iterations = 0
        else:
            if solved is None:
                solution, iterations, krylov_norm, converged = solve(
                    action, z, space, factors, vector, transpose=True, return_info=True, **options
                )
            else:
                solution, iterations, krylov_norm, converged = solved[key]
            krylov_norm, converged = float(krylov_norm), bool(converged)
        adjoint = _expand(solution, z, space)
        rhs = jax.tree.map(lambda value: value[row], rhs_batch)
        applied = jax.linear_transpose(action, z)(adjoint)[0]
        defect = jax.tree.map(jnp.subtract, applied, rhs)
        norm = float(jnp.linalg.norm(ravel_pytree(defect)[0]))
        rhs_norm = float(jnp.linalg.norm(ravel_pytree(rhs)[0]))
        # The seed root's ||A|| scales the gate: the current operator is a nearby one.
        report = _backward_error_report(cfg, row=row, residual_norm=norm, rhs_norm=rhs_norm,
            solution_norm=float(jnp.linalg.norm(solution)), operator_norm=preconditioner.operator_norm,
            iterations=iterations, backend="matrixfree_seed_lu", finite=bool(jnp.all(jnp.isfinite(solution))))
        tolerance, full_passed = report["tolerance"], report["accepted"]
        passed = full_passed
        if diagnostics is not None:
            report.update(accepted=passed, full_residual_accepted=full_passed,
                krylov_converged=converged, krylov_residual_norm=krylov_norm,
                krylov_tolerance=preconditioner.rtol * float(np.linalg.norm(host)),
                rhs_batch_size=preconditioner.rhs_batch_size, reused_rhs=reused,
                requested_rtol=preconditioner.rtol, restart=min(preconditioner.restart, vector.size),
                max_cycles=preconditioner.max_restarts)
            diagnostics.append(report)
        if not passed:
            reason = f"full residual {norm:.3e} exceeds acceptance {tolerance:.3e} or is nonfinite"
            raise AdjointSolveError(
                message=(
                    f"implicit adjoint matrixfree_seed_lu row {row} solve did not converge: "
                    f"{reason} after {int(iterations)} "
                    f"Krylov iterations (restart={min(preconditioner.restart, vector.size)}, "
                    f"max_cycles={preconditioner.max_restarts}, requested_rtol={preconditioner.rtol:g})"
                ),
                hint="Inspect the current operator and preconditioner; the failed adjoint was not returned.",
                iterations=int(iterations),
                residual_norm=norm,
                tolerance=tolerance,
            )
        cache[key] = solution, krylov_norm, converged
        adjoints.append(adjoint)
    result = jax.tree.map(lambda *values: jnp.stack(values), *adjoints)
    if return_linearization:
        root = MatrixFreeRootLinearization(
            residual,
            z,
            params,
            field,
            frozen,
            rcon,
            zcon,
            space,
            preconditioner.factors,
            action,
            cfg,
            operator_norm=preconditioner.operator_norm,
            **options,
        )
        return result, root
    return result


@jax.jit
def _active_apply(linear, template, space, vector):
    """Active-space ``J v`` from a forward linearization."""
    return _compress(linear(_expand(vector, template, space)), space)


@jax.jit
def _active_apply_transpose(transpose, template, space, vector):
    """Active-space ``J^T v`` from a saved transpose."""
    return _compress(transpose(_expand(vector, template, space))[0], space)


def structured_seed(residual, z, params, field, frozen, rcon, zcon, mask, cfg, factors, *,
                    rtol=1e-11, restart=100, max_restarts=3, rhs_batch_size=3):
    """A :class:`SeedLU` holding :class:`StructuredFactors` built at this root.

    It plays the role of a dense root's seed: :func:`solve_matrixfree_adjoint`
    with it solves the root's own adjoint, its tangents and Newton steps by
    GMRES preconditioned with an inverse that is exact where it was built, and
    every row passes the same full-operator backward-error gate, scaled by this
    root's ``||J||_2`` (a power-iteration lower bound).
    """
    from .freeboundary_implicit import _prepare_linearized_transpose

    space = jax.tree.map(jnp.asarray, _active_space(cfg.implicit, mask, _max_dofs(cfg)))
    transpose = _prepare_linearized_transpose(z, params, field, frozen, rcon, zcon, residual=residual)
    forward = jax.tree_util.Partial(_forward_from_transpose, transpose, z)
    operator_norm = _operator_norm_matvec(
        lambda v: _active_apply(forward, z, space, v), lambda v: _active_apply_transpose(transpose, z, space, v),
        int(space.left.shape[0]), ravel_pytree(z)[0].dtype, _OPERATOR_NORM_MATVEC_ITERATIONS)
    return SeedLU(
        factors=jax.tree.map(_frozen_host_copy, factors),
        space=jax.tree.map(_frozen_host_copy, space),
        signature=_signature(z),
        field_shape=field.shape,
        cfg=cfg,
        rtol=rtol,
        restart=restart,
        max_restarts=max_restarts,
        rhs_batch_size=rhs_batch_size,
        operator_norm=operator_norm,
    )
