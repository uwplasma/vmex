"""Current-root Krylov adjoints and tangents with an explicitly retained seed LU.

Only the preconditioner is reused. Numerical tapes are rebuilt at every root,
and acceptance checks use the full projected operator, not the old factors.
"""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jl
import numpy as np
from jax.flatten_util import ravel_pytree
from solvax.krylov import gmres

from . import implicit as im
from ._freeboundary_dense import DenseRootLinearization, _active_space, _compress, _expand
from .errors import AdjointSolveError


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
    require_adjoint_convergence: bool = False
    rhs_batch_size: int = 1
    tangent_rtol: float = 1e-11

    @classmethod
    def from_root(
        cls,
        root,
        *,
        rtol=1e-11,
        restart=30,
        max_restarts=10,
        require_adjoint_convergence=False,
        rhs_batch_size=1,
        tangent_rtol=None,
    ):
        """Snapshot a dense seed without retaining its numerical differentiation tape."""
        if type(root) is not DenseRootLinearization or root.factors is None:
            raise ValueError("a live dense linearization is required to create a seed LU")
        tangent_rtol = rtol if tangent_rtol is None else tangent_rtol
        if not np.isfinite(tangent_rtol) or not 0 < tangent_rtol < 1:
            raise ValueError("tangent_rtol must be finite and in (0, 1)")
        if not np.isfinite(rtol) or not 0 < rtol < 1:
            raise ValueError("Krylov rtol must be finite and in (0, 1)")
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (restart, max_restarts)):
            raise ValueError("restart and max_restarts must be positive integers")
        if not isinstance(require_adjoint_convergence, bool):
            raise ValueError("require_adjoint_convergence must be boolean")
        if isinstance(rhs_batch_size, bool) or not isinstance(rhs_batch_size, int) or not 1 <= rhs_batch_size <= 4:
            raise ValueError("rhs_batch_size must be an integer in [1, 4]")
        factors = tuple(np.array(x, copy=True) for x in root.factors)
        space = jax.tree.map(lambda x: np.array(x, copy=True), root.space)
        if factors[0].dtype != np.float64:
            raise TypeError("seed LU requires float64")
        for value in (*factors, *space):
            value.setflags(write=False)
        return cls(
            factors=factors,
            space=space,
            signature=_signature(root.z),
            field_shape=root.field.shape,
            cfg=root.cfg,
            rtol=rtol,
            restart=restart,
            max_restarts=max_restarts,
            require_adjoint_convergence=require_adjoint_convergence,
            rhs_batch_size=rhs_batch_size,
            tangent_rtol=tangent_rtol,
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
        precond=lambda value: jl.lu_solve(factors, value, trans=int(transpose)),
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
    space = _active_space(cfg.implicit, mask, cfg.adjoint_dense_max_dofs)
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
        report = im._adjoint_diagnostic(cfg.implicit, row=row, residual_norm=norm,
            rhs_norm=rhs_norm, iterations=iterations, backend="matrixfree_seed_lu",
            residual_rtol=cfg.adjoint_residual_rtol, finite=bool(jnp.all(jnp.isfinite(solution))))
        tolerance, full_passed = report["tolerance"], report["accepted"]
        passed = full_passed and (converged or not preconditioner.require_adjoint_convergence)
        if diagnostics is not None:
            report.update(accepted=passed, full_residual_accepted=full_passed,
                krylov_converged=converged, krylov_residual_norm=krylov_norm,
                krylov_tolerance=preconditioner.rtol * float(np.linalg.norm(host)),
                require_adjoint_convergence=preconditioner.require_adjoint_convergence,
                rhs_batch_size=preconditioner.rhs_batch_size, reused_rhs=reused,
                requested_rtol=preconditioner.rtol, restart=min(preconditioner.restart, vector.size),
                max_cycles=preconditioner.max_restarts)
            diagnostics.append(report)
        if not passed:
            reason = (
                f"full residual {norm:.3e} exceeds acceptance {tolerance:.3e} or is nonfinite"
                if not full_passed
                else "requested Krylov tolerance was not met"
            )
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
            **(options | {"rtol": preconditioner.tangent_rtol}),
        )
        return result, root
    return result
