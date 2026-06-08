"""Adjoint scaffolding for free-boundary vacuum solves.

Phase 1 intentionally keeps this module small and explicit.  It validates the
linear-solve differentiation contract that the production NESTOR replacement
will need: solve the primal system in the forward pass and use transpose solves
in the backward pass rather than differentiating through an iterative solver.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping
from contextlib import nullcontext
from typing import Any

import numpy as np

from vmec_jax._compat import jax, jnp, tree_util

from .free_boundary_adjoint_controller import (
    jax_visible_accepted_only_nonlinear_controller_jax,
    jax_visible_accepted_nonlinear_controller_directional_check_jax,
    jax_visible_accepted_nonlinear_controller_jax,
    jax_visible_masked_nonlinear_controller_directional_check_jax,
    jax_visible_masked_nonlinear_controller_jax,
    jax_visible_nonlinear_controller_directional_check_jax,
    jax_visible_nonlinear_controller_jax,
    jax_visible_segmented_accepted_nonlinear_controller_jax,
    jax_visible_segmented_state_only_accepted_nonlinear_controller_jax,
    jax_visible_state_only_accepted_nonlinear_controller_jax,
    jax_visible_state_only_accepted_only_nonlinear_controller_jax,
    jax_visible_unrolled_state_only_accepted_only_nonlinear_controller_jax,
    jax_visible_unrolled_accepted_only_nonlinear_controller_jax,
    pytree_directional_derivative_check_jax,
)

__all__ = [
    "direct_coil_accepted_trace_branch_metadata",
    "direct_coil_accepted_trace_controller_custom_vjp_scalars_jax",
    "direct_coil_accepted_trace_controller_replay_plan",
    "direct_coil_accepted_trace_replay_graph_metadata",
    "direct_coil_accepted_trace_step_controls_jax",
    "direct_coil_accepted_trace_step_policy_segments",
    "direct_coil_boundary_replay_context_for_shape",
    "direct_coil_adaptive_full_loop_same_branch_gate_report",
    "direct_coil_run_free_boundary_branch_local_scalar_value_and_grad_jax",
    "direct_coil_run_free_boundary_branch_local_scalars_value_and_jacobian_jax",
    "direct_coil_same_branch_physical_scalar_gate_report",
    "direct_coil_same_branch_replay_gate_report",
    "direct_coil_same_branch_controller_scalars_custom_vjp_report",
    "free_boundary_adjoint_trace_replay_diagnostics",
    "jax_visible_accepted_only_nonlinear_controller_jax",
    "jax_visible_accepted_nonlinear_controller_directional_check_jax",
    "jax_visible_accepted_nonlinear_controller_jax",
    "jax_visible_masked_nonlinear_controller_directional_check_jax",
    "jax_visible_masked_nonlinear_controller_jax",
    "jax_visible_nonlinear_controller_directional_check_jax",
    "jax_visible_nonlinear_controller_jax",
    "jax_visible_segmented_accepted_nonlinear_controller_jax",
    "jax_visible_segmented_state_only_accepted_nonlinear_controller_jax",
    "jax_visible_state_only_accepted_nonlinear_controller_jax",
    "jax_visible_state_only_accepted_only_nonlinear_controller_jax",
    "jax_visible_unrolled_state_only_accepted_only_nonlinear_controller_jax",
    "jax_visible_unrolled_accepted_only_nonlinear_controller_jax",
    "pytree_directional_derivative_check_jax",
]


def _block_until_ready_for_timing(value: Any) -> Any:
    """Synchronize JAX arrays before recording device timing diagnostics."""

    if jax is None:
        return value
    try:
        return jax.block_until_ready(value)
    except Exception:
        # Some older JAX versions do not accept every pytree container at the
        # top level.  Fall back to synchronizing individual leaves.
        return tree_util.tree_map(lambda leaf: jax.block_until_ready(leaf), value)


def _jax_named_scope(name: str) -> Any:
    """Return a JAX named-scope context when supported, otherwise a no-op."""

    if jax is None or not hasattr(jax, "named_scope"):
        return nullcontext()
    return jax.named_scope(name)


def dense_vacuum_solve_jax(A: Any, b: Any, *, symmetric: bool = False) -> Any:
    """Solve a dense toy vacuum linear system with an implicit adjoint.

    Parameters
    ----------
    A:
        Dense square matrix.
    b:
        Right-hand side vector or matrix.
    symmetric:
        If true, the transpose solve is the same as the primal solve.

    Notes
    -----
    This is a scaffold for small tests and future NESTOR refactoring.  It does
    not imply that the current production NESTOR path is fully differentiable.
    The production path should eventually expose a JAX-native matrix-free
    operator and pass it through ``jax.lax.custom_linear_solve`` or equivalent.
    """

    A_arr = jnp.asarray(A)
    b_arr = jnp.asarray(b)
    if A_arr.ndim != 2 or A_arr.shape[0] != A_arr.shape[1]:
        raise ValueError("A must be a square dense matrix")
    if b_arr.shape[0] != A_arr.shape[0]:
        raise ValueError(f"b leading dimension {b_arr.shape[0]} does not match A size {A_arr.shape[0]}")

    if jax is None:  # pragma: no cover - dependency fallback.
        return jnp.linalg.solve(A_arr, b_arr)

    def matvec(x):
        return A_arr @ x

    def solve_fn(_matvec, rhs):
        return jnp.linalg.solve(A_arr, rhs)

    def transpose_solve_fn(_matvec, rhs):
        matrix = A_arr if bool(symmetric) else A_arr.T
        return jnp.linalg.solve(matrix, rhs)

    return jax.lax.custom_linear_solve(
        matvec,
        b_arr,
        solve_fn,
        transpose_solve=transpose_solve_fn,
        symmetric=bool(symmetric),
    )


def dense_vacuum_residual(A: Any, x: Any, b: Any) -> Any:
    """Return ``A @ x - b`` for tests and diagnostics."""

    return jnp.asarray(A) @ jnp.asarray(x) - jnp.asarray(b)


def dense_nonlinear_solve_jax(
    residual_fn: Any,
    initial: Any,
    params: Any,
    *,
    max_iter: int = 10,
    damping: float = 1.0,
) -> Any:
    """Solve a small nonlinear residual with an implicit-root adjoint.

    Parameters
    ----------
    residual_fn:
        Callable ``residual_fn(x, params)`` returning a 1D residual array.
        ``params`` may be any JAX pytree.
    initial:
        Initial state for Newton iterations.
    params:
        Differentiable residual parameters.
    max_iter:
        Number of dense Newton iterations.
    damping:
        Scalar multiplier applied to each Newton step.

    Notes
    -----
    This is the nonlinear analogue of :func:`dense_vacuum_solve_jax` for the
    free-boundary phase-2 validation ladder.  The forward pass still uses an
    explicit dense Newton iteration, but the reverse pass applies the implicit
    function theorem at the converged root,

    ``F_x.T @ lambda = dJ/dx`` and ``dJ/dp = -F_p.T @ lambda``.

    It is intentionally limited to dense toy systems and validation gates.  It
    does not claim that the production VMEC/NESTOR nonlinear iteration loop has
    a full custom adjoint; it provides the tested primitive that loop should be
    refactored toward.
    """

    x0 = jnp.asarray(initial)
    if x0.ndim != 1:
        raise ValueError("initial must be a 1D state vector")
    max_iter_i = int(max_iter)
    if max_iter_i < 0:
        raise ValueError("max_iter must be non-negative")
    damping_f = float(damping)

    def _newton_solve(init, prm):
        def _step(_i, x):
            residual = jnp.asarray(residual_fn(x, prm))
            if residual.shape != x.shape:
                raise ValueError("residual_fn must return the same shape as initial")
            jac_x = jax.jacfwd(lambda y: jnp.asarray(residual_fn(y, prm)))(x)
            delta = jnp.linalg.solve(jac_x, residual)
            return x - damping_f * delta

        if jax is None:  # pragma: no cover - JAX-free import fallback.
            x = init
            for _ in range(max_iter_i):
                residual = jnp.asarray(residual_fn(x, prm))
                jac_x = _finite_difference_jacobian(lambda y: residual_fn(y, prm), x)
                x = x - damping_f * jnp.linalg.solve(jac_x, residual)
            return x
        return jax.lax.fori_loop(0, max_iter_i, _step, init)

    if jax is None:  # pragma: no cover - dependency fallback.
        return _newton_solve(x0, params)

    @jax.custom_vjp
    def _solve(init, prm):
        return _newton_solve(init, prm)

    def _solve_fwd(init, prm):
        root = _newton_solve(init, prm)
        return root, (root, prm, jnp.zeros_like(init))

    def _solve_bwd(saved, root_bar):
        root, prm, init_zero = saved
        jac_x = jax.jacfwd(lambda y: jnp.asarray(residual_fn(y, prm)))(root)
        lam = jnp.linalg.solve(jac_x.T, jnp.asarray(root_bar))
        _, pullback_params = jax.vjp(lambda pp: jnp.asarray(residual_fn(root, pp)), prm)
        grad_params = pullback_params(lam)[0]
        grad_params = tree_util.tree_map(lambda value: -value, grad_params)
        return init_zero, grad_params

    _solve.defvjp(_solve_fwd, _solve_bwd)
    return _solve(x0, params)


def dense_fixed_point_solve_jax(
    update_fn: Any,
    initial: Any,
    params: Any,
    *,
    max_iter: int = 10,
    damping: float = 1.0,
) -> Any:
    """Solve ``x = update_fn(x, params)`` with the nonlinear implicit adjoint.

    This is the small JAX-visible fixed-point wrapper used by the
    free-boundary phase-2 validation ladder.  It models the production coupling
    pattern, in which the accepted plasma state changes the boundary on which
    the external field is sampled and the vacuum response updates the state.
    Gradients are supplied by :func:`dense_nonlinear_solve_jax` through the
    residual ``x - update_fn(x, params)``.

    The helper is intentionally dense and validation-scale.  It should not be
    mistaken for the production ``run_free_boundary`` adjoint; it is the tested
    primitive that a future JAX-visible free-boundary fixed-point loop should
    reduce to.
    """

    def residual(state, prm):
        state_arr = jnp.asarray(state)
        update = jnp.asarray(update_fn(state_arr, prm))
        if update.shape != state_arr.shape:
            raise ValueError("update_fn must return the same shape as initial")
        return state_arr - update

    return dense_nonlinear_solve_jax(
        residual,
        initial,
        params,
        max_iter=max_iter,
        damping=damping,
    )

def _finite_difference_jacobian(fn: Any, x: Any, eps: float = 1.0e-6) -> Any:
    """Small NumPy/JAX-free fallback Jacobian for import-only environments."""

    x_arr = jnp.asarray(x)
    eye = jnp.eye(int(x_arr.size), dtype=x_arr.dtype)
    cols = []
    for k in range(int(x_arr.size)):
        step = eps * eye[k]
        cols.append((jnp.asarray(fn(x_arr + step)) - jnp.asarray(fn(x_arr - step))) / (2.0 * eps))
    return jnp.stack(cols, axis=1)


def vmec_source_from_gsource_jax(
    gsource: Any,
    *,
    onp: float,
    lasym: bool,
    nuv3: int | None = None,
    nuv_full: int | None = None,
    imirr: Any | None = None,
    imirr_full: Any | None = None,
) -> Any:
    """JAX version of VMEC/NESTOR source symmetrization.

    ``gsource`` is the weighted normal-field source used by the VMEC-like
    NESTOR bridge.  For stellarator-symmetric solves VMEC anti-symmetrizes the
    source with its mirror point before projecting onto sine modes.  For LASYM
    solves it uses the source directly.  This helper is intentionally small and
    side-effect free so the source-to-mode-RHS stage can be differentiated and
    finite-difference checked independently of the current host NESTOR path.
    """

    gsrc = jnp.reshape(jnp.asarray(gsource), (-1,))
    n_source = int(gsrc.shape[0])
    n3 = int(nuv3) if nuv3 is not None else n_source
    nfull = int(nuv_full) if nuv_full is not None else n3

    if bool(lasym):
        return float(onp) * gsrc[:n3]

    if n_source >= nfull and imirr_full is not None:
        mirror = jnp.asarray(imirr_full, dtype=jnp.int32)[:n3]
        mirrored = gsrc[mirror]
    elif imirr is not None:
        mirror = jnp.asarray(imirr, dtype=jnp.int32)[:n3]
        mirrored = gsrc[mirror]
    else:
        raise ValueError("non-LASYM source symmetrization requires imirr or imirr_full")
    return 0.5 * float(onp) * (gsrc[:n3] - mirrored)


def mode_rhs_from_gsource_jax(
    gsource: Any,
    *,
    sin_basis: Any,
    xmpot: Any,
    n_raw: Any,
    onp: float,
    lasym: bool,
    cos_basis: Any | None = None,
    nuv3: int | None = None,
    nuv_full: int | None = None,
    imirr: Any | None = None,
    imirr_full: Any | None = None,
) -> Any:
    """Project a VMEC/NESTOR grid source into mode-space RHS coefficients.

    This mirrors the production ``_vmec_bvec_from_gsource`` contract with JAX
    arrays.  It is a validation rung for the future production adjoint:
    differentiable external fields can feed this source projection, then a
    custom-linear-solve vacuum primitive, before the full NESTOR operator is
    ported.
    """

    src = vmec_source_from_gsource_jax(
        gsource,
        onp=float(onp),
        lasym=bool(lasym),
        nuv3=nuv3,
        nuv_full=nuv_full,
        imirr=imirr,
        imirr_full=imirr_full,
    )
    sin = jnp.asarray(sin_basis)
    if sin.ndim != 2:
        raise ValueError("sin_basis must be a 2D array")
    bsin = sin.T @ src

    xmpot_arr = jnp.asarray(xmpot)
    n_raw_arr = jnp.asarray(n_raw)
    skip_mask = jnp.logical_and(xmpot_arr == 0, n_raw_arr < 0)
    bsin = jnp.where(skip_mask, 0.0, bsin)

    if not bool(lasym):
        return bsin
    if cos_basis is None:
        raise ValueError("cos_basis is required for LASYM mode RHS projection")
    cos = jnp.asarray(cos_basis)
    if cos.shape != sin.shape:
        raise ValueError("cos_basis must match sin_basis shape")
    bcos = cos.T @ src
    bcos = jnp.where(skip_mask, 0.0, bcos)
    return jnp.concatenate([bsin, bcos], axis=0)


def mode_matrix_from_grpmn_jax(
    grpmn: Any,
    *,
    sin_basis: Any,
    xmpot: Any,
    n_raw: Any,
    lasym: bool,
    cos_basis: Any | None = None,
    mn0: int = 0,
) -> Any:
    """Build the VMEC/NESTOR mode matrix from Green-function mode samples.

    This is the JAX equivalent of the host ``_vmec_mode_matrix_from_grpmn``
    helper.  It validates the matrix-assembly half of the NESTOR adjoint
    contract: once the Green-function kernel samples ``grpmn`` are available in
    JAX, the mode matrix can be assembled, solved, and differentiated without
    crossing back to NumPy.
    """

    g = jnp.asarray(grpmn)
    sin = jnp.asarray(sin_basis)
    if g.ndim != 2:
        raise ValueError("grpmn must be a 2D array")
    if sin.ndim != 2:
        raise ValueError("sin_basis must be a 2D array")
    mnpd = int(sin.shape[1])
    if g.shape[0] < mnpd:
        raise ValueError("invalid_grpmn_shape")

    xmpot_arr = jnp.asarray(xmpot)
    n_raw_arr = jnp.asarray(n_raw)
    skip_col = jnp.logical_and(xmpot_arr == 0, n_raw_arr < 0)
    pi3 = float(4.0 * (jnp.pi**3))

    gsin = g[:mnpd, :]
    a11 = gsin @ sin
    a11 = jnp.where(skip_col[None, :], 0.0, a11)
    a11 = a11 + pi3 * jnp.eye(mnpd, dtype=a11.dtype)

    if not bool(lasym):
        return a11

    if g.shape[0] < 2 * mnpd:
        raise ValueError("invalid_grpmn_shape_lasym")
    if cos_basis is None:
        raise ValueError("cos_basis is required for LASYM mode matrix assembly")
    cos = jnp.asarray(cos_basis)
    if cos.shape != sin.shape:
        raise ValueError("cos_basis must match sin_basis shape")

    gcos = g[mnpd : 2 * mnpd, :]
    a12 = jnp.where(skip_col[None, :], 0.0, gsin @ cos)
    a21 = jnp.where(skip_col[None, :], 0.0, gcos @ sin)
    a22 = jnp.where(skip_col[None, :], 0.0, gcos @ cos)
    a22 = a22 + pi3 * jnp.eye(mnpd, dtype=a22.dtype)
    if 0 <= int(mn0) < mnpd:
        a22 = a22.at[int(mn0), int(mn0)].add(pi3)

    top = jnp.concatenate([a11, a12], axis=1)
    bottom = jnp.concatenate([a21, a22], axis=1)
    return jnp.concatenate([top, bottom], axis=0)


def mode_matrix_matvec_from_grpmn_jax(
    vector: Any,
    grpmn: Any,
    *,
    sin_basis: Any,
    xmpot: Any,
    n_raw: Any,
    lasym: bool,
    cos_basis: Any | None = None,
    mn0: int = 0,
    transpose: bool = False,
) -> Any:
    """Apply the VMEC/NESTOR mode operator without materializing it.

    This is the matrix-free counterpart to
    :func:`mode_matrix_from_grpmn_jax`.  It still consumes the JAX source
    response samples ``grpmn`` but avoids assembling the dense mode matrix
    before a Krylov solve.  The helper is intentionally opt-in and is used as a
    validation seam toward a future fully matrix-free NESTOR/source response.
    """

    g = jnp.asarray(grpmn)
    sin = jnp.asarray(sin_basis)
    x = jnp.asarray(vector)
    if g.ndim != 2:
        raise ValueError("grpmn must be a 2D array")
    if sin.ndim != 2:
        raise ValueError("sin_basis must be a 2D array")
    mnpd = int(sin.shape[1])
    if g.shape[0] < mnpd:
        raise ValueError("invalid_grpmn_shape")
    if x.shape[0] != (2 * mnpd if bool(lasym) else mnpd):
        raise ValueError("vector size does not match mode operator")

    xmpot_arr = jnp.asarray(xmpot)
    n_raw_arr = jnp.asarray(n_raw)
    skip_col = jnp.logical_and(xmpot_arr == 0, n_raw_arr < 0)
    pi3 = float(4.0 * (jnp.pi**3))
    gsin = g[:mnpd, :]

    if not bool(lasym):
        if bool(transpose):
            projected = sin.T @ (gsin.T @ x)
            return jnp.where(skip_col, 0.0, projected) + pi3 * x
        return gsin @ (sin @ jnp.where(skip_col, 0.0, x)) + pi3 * x

    if g.shape[0] < 2 * mnpd:
        raise ValueError("invalid_grpmn_shape_lasym")
    if cos_basis is None:
        raise ValueError("cos_basis is required for LASYM mode matrix application")
    cos = jnp.asarray(cos_basis)
    if cos.shape != sin.shape:
        raise ValueError("cos_basis must match sin_basis shape")

    gcos = g[mnpd : 2 * mnpd, :]
    xs = x[:mnpd]
    xc = x[mnpd:]
    if bool(transpose):
        grid = gsin.T @ xs + gcos.T @ xc
        ys = jnp.where(skip_col, 0.0, sin.T @ grid) + pi3 * xs
        yc = jnp.where(skip_col, 0.0, cos.T @ grid) + pi3 * xc
    else:
        grid = sin @ jnp.where(skip_col, 0.0, xs) + cos @ jnp.where(skip_col, 0.0, xc)
        ys = gsin @ grid + pi3 * xs
        yc = gcos @ grid + pi3 * xc
    if 0 <= int(mn0) < mnpd:
        yc = yc.at[int(mn0)].add(pi3 * xc[int(mn0)])
    return jnp.concatenate([ys, yc], axis=0)


def mode_operator_vacuum_solve_jax(
    grpmn: Any,
    rhs_mode: Any,
    *,
    sin_basis: Any,
    xmpot: Any,
    n_raw: Any,
    lasym: bool,
    cos_basis: Any | None = None,
    mn0: int = 0,
    include_phi_flat: bool = True,
    include_residual: bool = True,
    solver: str = "gmres",
    tol: float = 1.0e-11,
    atol: float = 1.0e-13,
    maxiter: int | None = None,
    restart: int | None = None,
) -> dict[str, Any]:
    """Solve the mode response through a matrix-free mode operator.

    The default production path still uses the dense mode matrix.  This helper
    is an opt-in validation/optimization seam for the low-resolution JAX
    NESTOR lane: forward and transpose solves apply
    :func:`mode_matrix_matvec_from_grpmn_jax` directly, so no dense mode matrix
    is assembled for the solve or residual.
    """

    rhs = jnp.asarray(rhs_mode)
    sin = jnp.asarray(sin_basis)
    if rhs.ndim != 1:
        raise ValueError("matrix-free mode solve requires a 1D rhs_mode")
    if sin.ndim != 2:
        raise ValueError("sin_basis must be a 2D array")
    grpmn_arr = jnp.asarray(grpmn)
    solver_name = str(solver).strip().lower()
    if solver_name not in ("gmres", "bicgstab"):
        raise ValueError("solver must be 'gmres' or 'bicgstab'")

    def _matvec(vec):
        return mode_matrix_matvec_from_grpmn_jax(
            vec,
            grpmn_arr,
            sin_basis=sin,
            cos_basis=cos_basis,
            xmpot=xmpot,
            n_raw=n_raw,
            lasym=bool(lasym),
            mn0=int(mn0),
            transpose=False,
        )

    def _transpose_matvec(vec):
        return mode_matrix_matvec_from_grpmn_jax(
            vec,
            grpmn_arr,
            sin_basis=sin,
            cos_basis=cos_basis,
            xmpot=xmpot,
            n_raw=n_raw,
            lasym=bool(lasym),
            mn0=int(mn0),
            transpose=True,
        )

    def _iterative_solve(matvec, vector):
        from jax.scipy.sparse.linalg import bicgstab, gmres

        if solver_name == "gmres":
            kwargs = {"tol": float(tol), "atol": float(atol), "maxiter": maxiter}
            if restart is not None:
                kwargs["restart"] = int(restart)
            return gmres(matvec, vector, **kwargs)[0]
        return bicgstab(matvec, vector, tol=float(tol), atol=float(atol), maxiter=maxiter)[0]

    if jax is None:  # pragma: no cover - dependency fallback.
        matrix = mode_matrix_from_grpmn_jax(
            grpmn_arr,
            sin_basis=sin,
            cos_basis=cos_basis,
            xmpot=xmpot,
            n_raw=n_raw,
            lasym=bool(lasym),
            mn0=int(mn0),
        )
        coeffs = jnp.linalg.solve(matrix, rhs)
    else:
        coeffs = jax.lax.custom_linear_solve(
            _matvec,
            rhs,
            lambda matvec, vector: _iterative_solve(matvec, vector),
            transpose_solve=lambda _matvec_unused, vector: _iterative_solve(_transpose_matvec, vector),
            symmetric=False,
        )

    if cos_basis is None:
        if coeffs.shape[0] != sin.shape[1]:
            raise ValueError("rhs size must match sin_basis columns")
        phi_flat = sin @ coeffs if bool(include_phi_flat) else None
    else:
        cos = jnp.asarray(cos_basis)
        if cos.shape != sin.shape:
            raise ValueError("cos_basis must match sin_basis shape")
        nmodes = int(sin.shape[1])
        if coeffs.shape[0] != 2 * nmodes:
            raise ValueError("doubled rhs size must be 2 * sin_basis columns")
        phi_flat = sin @ coeffs[:nmodes] + cos @ coeffs[nmodes:] if bool(include_phi_flat) else None

    out = {
        "mode_coeffs": coeffs,
        "solve_mode": f"matrix_free_{solver_name}",
        "mode_matrix_materialized": False,
    }
    if bool(include_phi_flat):
        out["phi_flat"] = phi_flat
    if bool(include_residual):
        out["residual"] = _matvec(coeffs) - rhs
    return out


def vmec_nonsingular_terms_from_bexni_jax(
    *,
    R: Any,
    Z: Any,
    Ru: Any,
    Zu: Any,
    Rv: Any,
    Zv: Any,
    ruu: Any,
    ruv: Any,
    rvv: Any,
    zuu: Any,
    zuv: Any,
    zvv: Any,
    bexni: Any,
    basis: dict[str, Any],
    tables: dict[str, Any],
    signgs: int,
    nvper: int,
) -> tuple[Any, Any]:
    """JAX VMEC/NESTOR nonsingular Green-function source/matrix assembly.

    This mirrors the core low-resolution algebra in
    ``free_boundary._vmec_nonsingular_terms_from_bexni`` once the boundary
    geometry and second derivatives are already sampled on the full VMEC
    angular grid.  The trigonometric and tangent tables are treated as static
    constants, while the boundary geometry and external normal-field source are
    differentiable JAX inputs.

    The helper is intentionally explicit rather than performance-tuned.  It is
    the validation bridge between the phase-1 dense mode-space adjoint scaffold
    and the future production NESTOR operator: tests can now differentiate
    through Green-kernel source assembly, mode projection, matrix assembly, and
    the implicit dense solve without crossing to NumPy.
    """

    R2 = jnp.asarray(R)
    Z2 = jnp.asarray(Z)
    Ru2 = jnp.asarray(Ru)
    Zu2 = jnp.asarray(Zu)
    Rv2 = jnp.asarray(Rv)
    Zv2 = jnp.asarray(Zv)
    ruu2 = jnp.asarray(ruu)
    ruv2 = jnp.asarray(ruv)
    rvv2 = jnp.asarray(rvv)
    zuu2 = jnp.asarray(zuu)
    zuv2 = jnp.asarray(zuv)
    zvv2 = jnp.asarray(zvv)
    if R2.ndim != 2:
        raise ValueError("R must be a 2D full-grid array")
    for name, arr in (
        ("Z", Z2),
        ("Ru", Ru2),
        ("Zu", Zu2),
        ("Rv", Rv2),
        ("Zv", Zv2),
        ("ruu", ruu2),
        ("ruv", ruv2),
        ("rvv", rvv2),
        ("zuu", zuu2),
        ("zuv", zuv2),
        ("zvv", zvv2),
    ):
        if arr.shape != R2.shape:
            raise ValueError(f"{name} must match R shape")

    nu = int(R2.shape[0])
    nv = int(R2.shape[1])
    nuv_full = int(nu * nv)
    nuv3 = int(basis["nuv3"])
    mf = int(basis["mf"])
    nf = int(basis["nf"])
    mnpd = int(basis["mnpd"])
    lasym = bool(basis["lasym"])
    mnpd2 = int(basis["mnpd2"])
    onp = float(basis["onp"])
    sign = float(int(signgs))
    nvper = max(1, int(nvper))
    if int(basis.get("nu_full", nu)) != nu:
        raise ValueError("R grid must use basis['nu_full'] rows")

    Rf = jnp.reshape(R2, (-1,))
    Zf = jnp.reshape(Z2, (-1,))
    R_uf = jnp.reshape(Ru2, (-1,))
    Z_uf = jnp.reshape(Zu2, (-1,))
    R_vf = jnp.reshape(Rv2, (-1,))
    Z_vf = jnp.reshape(Zv2, (-1,))
    ruuf = jnp.reshape(ruu2, (-1,))
    ruvf = jnp.reshape(ruv2, (-1,))
    rvvf = jnp.reshape(rvv2, (-1,))
    zuuf = jnp.reshape(zuu2, (-1,))
    zuvf = jnp.reshape(zuv2, (-1,))
    zvvf = jnp.reshape(zvv2, (-1,))

    snr = sign * Rf * Z_uf
    snv = sign * (R_uf * Z_vf - R_vf * Z_uf)
    snz = -sign * Rf * R_uf
    drv = -(Rf * snr + Zf * snz)
    guu_b = R_uf * R_uf + Z_uf * Z_uf
    guv_b = (R_uf * R_vf + Z_uf * Z_vf) * onp * 2.0
    gvv_b = (R_vf * R_vf + Z_vf * Z_vf + Rf * Rf) * (onp * onp)
    auu = 0.5 * (snr * ruuf + snz * zuuf)
    auv = (snr * ruvf + snv * R_uf + snz * zuvf) * onp
    avv = (snv * R_vf + 0.5 * (snr * (rvvf - Rf) + snz * zvvf)) * (onp * onp)
    rzb2 = Rf * Rf + Zf * Zf

    idx_all = jnp.asarray(tables["idx_all"], dtype=jnp.int32)
    tanu = jnp.asarray(tables["tanu"])
    tanv = jnp.asarray(tables["tanv"])
    cosuv = jnp.asarray(tables["cosuv"])
    sinuv = jnp.asarray(tables["sinuv"])
    cosper = jnp.asarray(tables["cosper"])
    sinper = jnp.asarray(tables["sinper"])
    cosv_tab = jnp.asarray(tables["cosv_tab"])
    sinv_tab = jnp.asarray(tables["sinv_tab"])
    cosui = jnp.asarray(tables["cosui"])
    sinui = jnp.asarray(tables["sinui"])
    nu_fourp = int(cosui.shape[1])
    if nu_fourp <= 0:
        raise ValueError("invalid nonsingular table shape")

    rcosuv = Rf * cosuv
    rsinuv = Rf * sinuv
    bex = jnp.reshape(jnp.asarray(bexni), (-1,))
    if int(bex.shape[0]) < nuv3:
        raise ValueError("bexni must contain at least basis['nuv3'] entries")
    bex = bex[:nuv3]

    if "iuv_grid" in tables:
        iuv_grid = jnp.asarray(tables["iuv_grid"], dtype=jnp.int32)
        iref_grid = jnp.asarray(tables["iref_grid"], dtype=jnp.int32)
        cosv_modes = jnp.asarray(tables["cosv_modes"])
        sinv_modes = jnp.asarray(tables["sinv_modes"])
        idx_p_flat = jnp.asarray(tables["idx_p_flat"], dtype=jnp.int32)
        idx_m_negative = jnp.asarray(tables["idx_m_negative"], dtype=jnp.int32)
        negative_positions_arr = jnp.asarray(tables["negative_positions"], dtype=jnp.int32)
        sinm_sym = jnp.asarray(tables["sinm_sym"])
        cosm_sym = jnp.asarray(tables["cosm_sym"])
        sinm_asym = jnp.asarray(tables["sinm_asym"])
        cosm_asym = jnp.asarray(tables["cosm_asym"])
    else:
        imirr_full = jnp.asarray(basis["imirr_full"], dtype=jnp.int32)
        idx_u = jnp.arange(nu_fourp, dtype=jnp.int32)
        idx_v = jnp.arange(nv, dtype=jnp.int32)
        iuv_grid = idx_u[:, None] * int(nv) + idx_v[None, :]
        iref_grid = imirr_full[iuv_grid]
        cosv_modes = 0.5 * onp * cosv_tab[: nf + 1, :]
        sinv_modes = 0.5 * onp * sinv_tab[: nf + 1, :]
        mf1 = int(mf + 1)
        idx_p_rows: list[int] = []
        idx_m_rows: list[int] = []
        negative_positions: list[int] = []
        flat_pos = 0
        for m in range(mf + 1):
            for n in range(nf + 1):
                idx_p_rows.append(int(m + (n + nf) * mf1))
                if n != 0 and m != 0:
                    idx_m_rows.append(int(m + ((-n) + nf) * mf1))
                    negative_positions.append(int(flat_pos))
                flat_pos += 1
        idx_p_flat = jnp.asarray(idx_p_rows, dtype=jnp.int32)
        idx_m_negative = jnp.asarray(idx_m_rows, dtype=jnp.int32)
        negative_positions_arr = jnp.asarray(negative_positions, dtype=jnp.int32)
        sinm_sym = sinui[: mf + 1, :]
        cosm_sym = -cosui[: mf + 1, :]
        sinm_asym = cosui[: mf + 1, :]
        cosm_asym = sinui[: mf + 1, :]

    gstore = jnp.zeros((nuv_full,), dtype=Rf.dtype)
    grpmn = jnp.zeros((mnpd2, nuv3), dtype=Rf.dtype)

    def _ip_body(carry: tuple[Any, Any], ip: Any) -> tuple[tuple[Any, Any], None]:
        gstore_acc, grpmn_acc = carry
        ip = jnp.asarray(ip, dtype=jnp.int32)
        xip = rcosuv[ip]
        yip = rsinuv[ip]
        ivoff = jnp.asarray(nuv_full, dtype=jnp.int32) - ip
        iskip = ip // jnp.asarray(max(1, nv), dtype=jnp.int32)
        iuoff = jnp.asarray(nuv_full, dtype=jnp.int32) - jnp.asarray(nv, dtype=jnp.int32) * iskip
        gsave = rzb2[ip] + rzb2 - 2.0 * Zf[ip] * Zf
        dsave = drv[ip] + Zf * snz[ip]
        delgr = jnp.zeros((nuv_full,), dtype=Rf.dtype)
        delgrp = jnp.zeros((nuv_full,), dtype=Rf.dtype)

        for kp in range(nvper):
            xper = xip * cosper[kp] - yip * sinper[kp]
            yper = yip * cosper[kp] + xip * sinper[kp]
            sxsave = (snr[ip] * xper - snv[ip] * yper) / Rf[ip]
            sysave = (snr[ip] * yper + snv[ip] * xper) / Rf[ip]
            base = gsave - 2.0 * (xper * rcosuv + yper * rsinuv)
            deriv_num = rcosuv * sxsave + rsinuv * sysave + dsave

            if kp == 0 or nv == 1:
                tidx_u = idx_all + iuoff
                ivoff_k = ivoff + jnp.asarray(2 * nu * kp if nv == 1 else 0, dtype=jnp.int32)
                tidx_v = idx_all + ivoff_k
                tanu_use = tanu[tidx_u]
                tanv_use = tanv[tidx_v]
                ga1 = tanu_use * (guu_b[ip] * tanu_use + guv_b[ip] * tanv_use) + gvv_b[ip] * tanv_use * tanv_use
                ga2 = tanu_use * (auu[ip] * tanu_use + auv[ip] * tanv_use) + avv[ip] * tanv_use * tanv_use
                ga2 = ga2 / ga1
                ga1s = 1.0 / jnp.sqrt(ga1)
                mask = idx_all != ip if kp == 0 else jnp.ones((nuv_full,), dtype=bool)
                safe_base = jnp.where(mask, base, 1.0)
                ftemp = 1.0 / safe_base
                htemp = jnp.sqrt(ftemp)
                deriv = ftemp * htemp * deriv_num
                delgr = delgr + jnp.where(mask, htemp - ga1s, 0.0)
                delgrp = delgrp + jnp.where(mask, deriv - ga2 * ga1s, 0.0)
            else:
                ftemp = 1.0 / base
                htemp = jnp.sqrt(ftemp)
                delgr = delgr + htemp
                delgrp = delgrp + ftemp * htemp * deriv_num

        if nv == 1 and nvper > 1:
            scale = 1.0 / float(nvper)
            delgr = delgr * scale
            delgrp = delgrp * scale

        gstore_next = gstore_acc + bex[ip] * delgr
        del_iuv = delgrp[iuv_grid]
        del_ref = delgrp[iref_grid]
        ka_grid = del_iuv - del_ref
        g1_sym = jnp.einsum("uv,fv->uf", ka_grid, cosv_modes)
        g2_sym = jnp.einsum("uv,fv->uf", ka_grid, sinv_modes)

        gcos = jnp.einsum("mu,uf->mf", sinm_sym, g1_sym)
        gsin = jnp.einsum("mu,uf->mf", cosm_sym, g2_sym)
        total_plus = jnp.reshape(gcos + gsin, (-1,))
        total_minus = jnp.reshape(gcos - gsin, (-1,))
        cols_p = jnp.full_like(idx_p_flat, ip)
        cols_m = jnp.full_like(idx_m_negative, ip)
        grpmn_next = grpmn_acc.at[(idx_p_flat, cols_p)].add(total_plus)
        grpmn_next = grpmn_next.at[(idx_m_negative, cols_m)].add(total_minus[negative_positions_arr])

        if lasym:
            ks_grid = del_iuv + del_ref
            g1_asym = jnp.einsum("uv,fv->uf", ks_grid, cosv_modes)
            g2_asym = jnp.einsum("uv,fv->uf", ks_grid, sinv_modes)
            gcos_asym = jnp.einsum("mu,uf->mf", sinm_asym, g1_asym)
            gsin_asym = jnp.einsum("mu,uf->mf", cosm_asym, g2_asym)
            total_plus_asym = jnp.reshape(gcos_asym + gsin_asym, (-1,))
            total_minus_asym = jnp.reshape(gcos_asym - gsin_asym, (-1,))
            row_off = int(mnpd)
            grpmn_next = grpmn_next.at[(row_off + idx_p_flat, cols_p)].add(total_plus_asym)
            grpmn_next = grpmn_next.at[(row_off + idx_m_negative, cols_m)].add(
                total_minus_asym[negative_positions_arr]
            )

        return (gstore_next, grpmn_next), None

    if bool(tables.get("use_ip_scan", True)):
        (gstore, grpmn), _ = jax.lax.scan(
            _ip_body,
            (gstore, grpmn),
            jnp.arange(nuv3, dtype=jnp.int32),
        )
    else:
        for ip in range(nuv3):
            (gstore, grpmn), _ = _ip_body((gstore, grpmn), jnp.asarray(ip, dtype=jnp.int32))

    return gstore, grpmn


def vmec_analytic_terms_from_geometry_jax(
    *,
    R: Any,
    Ru: Any,
    Rv: Any,
    Zu: Any,
    Zv: Any,
    ruu: Any,
    ruv: Any,
    rvv: Any,
    zuu: Any,
    zuv: Any,
    zvv: Any,
    bexni: Any,
    basis: dict[str, Any],
    signgs: int,
) -> tuple[Any, Any]:
    """JAX VMEC/NESTOR analytic singular-source terms from ``analyt.f``.

    This helper ports the analytic/singular mode-source contribution used by
    the VMEC-like free-boundary bridge when first and second boundary
    derivatives are already available on the active VMEC angular grid.  The
    recurrence coefficients and mode tables are static, while the boundary
    metric/curvature channels and external source are differentiable.
    """

    R_arr = jnp.asarray(R)
    Ru_arr = jnp.asarray(Ru)
    Rv_arr = jnp.asarray(Rv)
    Zu_arr = jnp.asarray(Zu)
    Zv_arr = jnp.asarray(Zv)
    ruu_arr = jnp.asarray(ruu)
    ruv_arr = jnp.asarray(ruv)
    rvv_arr = jnp.asarray(rvv)
    zuu_arr = jnp.asarray(zuu)
    zuv_arr = jnp.asarray(zuv)
    zvv_arr = jnp.asarray(zvv)
    if R_arr.ndim != 2:
        raise ValueError("R must be a 2D active-grid array")
    for name, arr in (
        ("Ru", Ru_arr),
        ("Rv", Rv_arr),
        ("Zu", Zu_arr),
        ("Zv", Zv_arr),
        ("ruu", ruu_arr),
        ("ruv", ruv_arr),
        ("rvv", rvv_arr),
        ("zuu", zuu_arr),
        ("zuv", zuv_arr),
        ("zvv", zvv_arr),
    ):
        if arr.shape != R_arr.shape:
            raise ValueError(f"{name} must match R shape")

    mnpd = int(basis["mnpd"])
    lasym = bool(basis["lasym"])
    mf = int(basis["mf"])
    nf = int(basis["nf"])
    onp = float(basis["onp"])
    sign = float(int(signgs))
    npts = int(jnp.size(R_arr))
    theta = jnp.asarray(basis["theta"])
    zeta = jnp.asarray(basis["zeta"])
    if int(theta.size) != npts or int(zeta.size) != npts:
        raise ValueError("basis theta/zeta size must match active grid")
    bex = jnp.reshape(jnp.asarray(bexni), (-1,))
    if int(bex.shape[0]) < npts:
        raise ValueError("bexni must contain at least one active-grid value per point")
    bex = bex[:npts]

    Rf = jnp.reshape(R_arr, (-1,))
    Ruf = jnp.reshape(Ru_arr, (-1,))
    Rvf = jnp.reshape(Rv_arr, (-1,))
    Zuf = jnp.reshape(Zu_arr, (-1,))
    Zvf = jnp.reshape(Zv_arr, (-1,))
    ruuf = jnp.reshape(ruu_arr, (-1,))
    ruvf = jnp.reshape(ruv_arr, (-1,))
    rvvf = jnp.reshape(rvv_arr, (-1,))
    zuuf = jnp.reshape(zuu_arr, (-1,))
    zuvf = jnp.reshape(zuv_arr, (-1,))
    zvvf = jnp.reshape(zvv_arr, (-1,))

    guu_b = Ruf * Ruf + Zuf * Zuf
    guv_b = (Ruf * Rvf + Zuf * Zvf) * (2.0 * onp)
    gvv_b = (Rvf * Rvf + Zvf * Zvf + Rf * Rf) * (onp * onp)
    adp = guu_b + guv_b + gvv_b
    adm = guu_b - guv_b + gvv_b
    cma = gvv_b - guu_b
    sqrtc = 2.0 * jnp.sqrt(gvv_b)
    sqrta = 2.0 * jnp.sqrt(guu_b)
    sqad1 = jnp.sqrt(adp)
    sqad2 = jnp.sqrt(adm)
    tlp = (1.0 / sqad1) * jnp.log((sqad1 * sqrtc + adp + cma) / (sqad1 * sqrta - adp + cma))
    tlm = (1.0 / sqad2) * jnp.log((sqad2 * sqrtc + adm + cma) / (sqad2 * sqrta - adm + cma))
    tlp_prev = jnp.zeros_like(tlp)
    tlm_prev = jnp.zeros_like(tlm)
    tlpm = tlp + tlm

    snr = sign * Rf * Zuf
    snv = sign * (Ruf * Zvf - Rvf * Zuf)
    snz = -sign * Rf * Ruf
    auu = 0.5 * (snr * ruuf + snz * zuuf)
    auv = (snr * ruvf + snv * Ruf + snz * zuvf) * onp
    avv = (snv * Rvf + 0.5 * (snr * (rvvf - Rf) + snz * zvvf)) * (onp * onp)
    azp1u = auu + auv + avv
    azm1u = auu - auv + avv
    cma11u = avv - auu
    delt1u = adp * adm - cma * cma
    r1p = (azp1u * (delt1u - cma * cma) / adp - azm1u * adp + 2.0 * cma11u * cma) / delt1u
    r1m = (azm1u * (delt1u - cma * cma) / adm - azp1u * adm + 2.0 * cma11u * cma) / delt1u
    r0p = (-azp1u * adm * cma / adp - azm1u * cma + 2.0 * cma11u * adm) / delt1u
    r0m = (-azm1u * adp * cma / adm - azp1u * cma + 2.0 * cma11u * adp) / delt1u
    ra1p = azp1u / adp
    ra1m = azm1u / adm

    bsin = jnp.zeros((mf + 1, 2 * nf + 1), dtype=Rf.dtype)
    bcos = jnp.zeros((mf + 1, 2 * nf + 1), dtype=Rf.dtype)
    gsin = jnp.zeros((mf + 1, 2 * nf + 1, npts), dtype=Rf.dtype)
    gcos = jnp.zeros((mf + 1, 2 * nf + 1, npts), dtype=Rf.dtype)
    # ``cmns`` is a static VMEC analytic-integral coefficient table, not a
    # differentiable variable.  Keep it as a host constant so the compiled
    # closure can skip exact-zero coefficients without tracer booleans.
    cmns = np.asarray(basis["cmns"])

    sign1 = 1.0
    fl1 = 0.0
    for l in range(0, mf + nf + 1):
        fl = fl1
        slp = (r1p * fl + ra1p) * tlp + r0p * fl * tlp_prev - (r1p + r0p) / sqrtc + sign1 * (r0p - r1p) / sqrta
        slm = (r1m * fl + ra1m) * tlm + r0m * fl * tlm_prev - (r1m + r0m) / sqrtc + sign1 * (r0m - r1m) / sqrta
        slpm = slp + slm
        for nabs in range(0, nf + 1):
            zv = float(nabs) * zeta
            cosv = jnp.cos(zv)
            sinv = jnp.sin(zv)
            for m in range(0, mf + 1):
                cm = float(cmns[l, m, nabs])
                if cm == 0.0:
                    continue
                mu = float(m) * theta
                sinu = jnp.sin(mu)
                cosu = jnp.cos(mu)
                col_p = int(nabs + nf)
                col_m = int((-nabs) + nf)
                if nabs == 0 or m == 0:
                    sinp = (sinu * cosv - sinv * cosu) * cm
                    bsin = bsin.at[m, col_p].add(jnp.sum(tlpm * bex * sinp))
                    gsin = gsin.at[m, col_p, :].add(slpm * sinp)
                    if lasym:
                        cosp = (cosu * cosv + sinv * sinu) * cm
                        bcos = bcos.at[m, col_p].add(jnp.sum(tlpm * bex * cosp))
                        gcos = gcos.at[m, col_p, :].add(slpm * cosp)
                else:
                    sinp0 = sinu * cosv * cm
                    temp = -cosu * sinv * cm
                    sinm = sinp0 - temp
                    sinp = sinp0 + temp
                    bsin = bsin.at[m, col_p].add(jnp.sum(tlm * bex * sinp))
                    bsin = bsin.at[m, col_m].add(jnp.sum(tlp * bex * sinm))
                    gsin = gsin.at[m, col_p, :].add(slm * sinp)
                    gsin = gsin.at[m, col_m, :].add(slp * sinm)
                    if lasym:
                        cosp0 = cosu * cosv * cm
                        temp2 = sinu * sinv * cm
                        cosm = cosp0 - temp2
                        cosp = cosp0 + temp2
                        bcos = bcos.at[m, col_p].add(jnp.sum(tlm * bex * cosp))
                        bcos = bcos.at[m, col_m].add(jnp.sum(tlp * bex * cosm))
                        gcos = gcos.at[m, col_p, :].add(slm * cosp)
                        gcos = gcos.at[m, col_m, :].add(slp * cosm)

        fl1 = fl1 + 1.0
        fl2 = 2.0 * fl1 - 1.0
        sign1 = -sign1
        tlp_next = ((sqrtc + sign1 * sqrta) - fl2 * cma * tlp - fl * adm * tlp_prev) / (adp * fl1)
        tlm_next = ((sqrtc + sign1 * sqrta) - fl2 * cma * tlm - fl * adp * tlm_prev) / (adm * fl1)
        tlp_prev = tlp
        tlm_prev = tlm
        tlp = tlp_next
        tlm = tlm_next
        tlpm = tlp + tlm

    xmpot = np.asarray(basis["xmpot"], dtype=np.int32)
    n_raw = np.asarray(basis["n_raw"], dtype=np.int32)
    out_s = jnp.zeros((mnpd,), dtype=Rf.dtype)
    out_c = jnp.zeros((mnpd,), dtype=Rf.dtype)
    gr_s = jnp.zeros((mnpd, npts), dtype=Rf.dtype)
    gr_c = jnp.zeros((mnpd, npts), dtype=Rf.dtype)
    for j in range(mnpd):
        m = int(xmpot[j])
        n = int(n_raw[j])
        col = int(n + nf)
        out_s = out_s.at[j].set(bsin[m, col])
        gr_s = gr_s.at[j, :].set(gsin[m, col, :])
        if lasym:
            out_c = out_c.at[j].set(bcos[m, col])
            gr_c = gr_c.at[j, :].set(gcos[m, col, :])

    if lasym:
        return jnp.concatenate([out_s, out_c], axis=0), jnp.concatenate([gr_s, gr_c], axis=0)
    return out_s, gr_s


def dense_mode_vacuum_solve_jax(
    mode_matrix: Any,
    rhs_mode: Any,
    sin_basis: Any,
    cos_basis: Any | None = None,
    *,
    symmetric: bool = False,
    include_phi_flat: bool = True,
    include_residual: bool = True,
) -> dict[str, Any]:
    """Solve a dense mode-space vacuum system and reconstruct a grid potential.

    This is the next scaffold between the dense toy solve and the production
    NESTOR path.  The current VMEC-like NESTOR implementation eventually builds
    a dense mode-space matrix and right-hand side before reconstructing a
    scalar potential on the boundary grid.  This helper makes that contract
    JAX-transformable and differentiable while the full source/operator assembly
    remains in the host implementation.

    Parameters
    ----------
    mode_matrix:
        Dense mode-space matrix ``A``.
    rhs_mode:
        Right-hand side vector ``b``.
    sin_basis, cos_basis:
        Flattened boundary-grid basis arrays with shape ``(npoints, nmodes)``.
        For stellarator-symmetric mode vectors pass only ``sin_basis``.  For
        LASYM-style doubled vectors pass both basis blocks; the first block of
        ``mode_coeffs`` multiplies ``sin_basis`` and the second multiplies
        ``cos_basis``.
    symmetric:
        Forwarded to :func:`dense_vacuum_solve_jax`.
    include_phi_flat:
        Reconstruct the scalar potential on the boundary grid. Compact
        accepted-state replay paths can disable this because field
        reconstruction uses the mode coefficients directly.
    include_residual:
        Include the dense linear residual in the returned diagnostics. Compact
        accepted-state replay paths can disable this since only ``bsqvac`` is
        needed for the strict VMEC update.
    """

    A = jnp.asarray(mode_matrix)
    rhs = jnp.asarray(rhs_mode)
    sin = jnp.asarray(sin_basis)
    if sin.ndim != 2:
        raise ValueError("sin_basis must be a 2D array")
    coeffs = dense_vacuum_solve_jax(A, rhs, symmetric=bool(symmetric))

    if cos_basis is None:
        if coeffs.shape[0] != sin.shape[1]:
            raise ValueError("rhs/mode_matrix size must match sin_basis columns")
        phi_flat = sin @ coeffs if bool(include_phi_flat) else None
    else:
        cos = jnp.asarray(cos_basis)
        if cos.shape != sin.shape:
            raise ValueError("cos_basis must match sin_basis shape")
        nmodes = int(sin.shape[1])
        if coeffs.shape[0] != 2 * nmodes:
            raise ValueError("doubled rhs/mode_matrix size must be 2 * sin_basis columns")
        phi_flat = sin @ coeffs[:nmodes] + cos @ coeffs[nmodes:] if bool(include_phi_flat) else None

    out = {"mode_coeffs": coeffs}
    if bool(include_phi_flat):
        out["phi_flat"] = phi_flat
    if bool(include_residual):
        out["residual"] = dense_vacuum_residual(A, coeffs, rhs)
    return out


def _nonsingular_full_grid_from_active_jax(
    *,
    R: Any,
    Z: Any,
    Ru: Any,
    Zu: Any,
    Rv: Any,
    Zv: Any,
    ruu: Any,
    ruv: Any,
    rvv: Any,
    zuu: Any,
    zuv: Any,
    zvv: Any,
    basis: dict[str, Any],
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any]:
    """Return the full grid expected by VMEC's nonsingular Green block.

    The host bridge expands stellarator-symmetric active-grid geometry before
    calling the nonsingular Green-function assembly, but keeps the analytic
    singular terms on the active grid. This helper mirrors that convention in
    JAX for the combined low-resolution operator.
    """

    R2 = jnp.asarray(R)
    Z2 = jnp.asarray(Z)
    Ru2 = jnp.asarray(Ru)
    Zu2 = jnp.asarray(Zu)
    Rv2 = jnp.asarray(Rv)
    Zv2 = jnp.asarray(Zv)
    ruu2 = jnp.asarray(ruu)
    ruv2 = jnp.asarray(ruv)
    rvv2 = jnp.asarray(rvv)
    zuu2 = jnp.asarray(zuu)
    zuv2 = jnp.asarray(zuv)
    zvv2 = jnp.asarray(zvv)
    nu_full = int(basis["nu_full"])
    ntheta3, nv = int(R2.shape[0]), int(R2.shape[1])
    if bool(basis["lasym"]) or nu_full == ntheta3:
        return R2, Z2, Ru2, Zu2, Rv2, Zv2, ruu2, ruv2, rvv2, zuu2, zuv2, zvv2

    shape_full = (nu_full, nv)
    zeros = jnp.zeros(shape_full, dtype=R2.dtype)
    Rf = zeros.at[:ntheta3, :].set(R2)
    Zf = zeros.at[:ntheta3, :].set(Z2)
    Ruf = zeros.at[:ntheta3, :].set(Ru2)
    Zuf = zeros.at[:ntheta3, :].set(Zu2)
    Rvf = zeros.at[:ntheta3, :].set(Rv2)
    Zvf = zeros.at[:ntheta3, :].set(Zv2)
    ruuf = zeros.at[:ntheta3, :].set(ruu2)
    ruvf = zeros.at[:ntheta3, :].set(ruv2)
    rvvf = zeros.at[:ntheta3, :].set(rvv2)
    zuuf = zeros.at[:ntheta3, :].set(zuu2)
    zuvf = zeros.at[:ntheta3, :].set(zuv2)
    zvvf = zeros.at[:ntheta3, :].set(zvv2)

    kv_m = (nv - jnp.arange(nv, dtype=jnp.int32)) % max(1, nv)
    for ku in range(1, max(1, ntheta3 - 1)):
        km = (nu_full - ku) % max(1, nu_full)
        if km < ntheta3:
            continue
        Rf = Rf.at[km, :].set(R2[ku, kv_m])
        Zf = Zf.at[km, :].set(-Z2[ku, kv_m])
        Ruf = Ruf.at[km, :].set(-Ru2[ku, kv_m])
        Zuf = Zuf.at[km, :].set(Zu2[ku, kv_m])
        Rvf = Rvf.at[km, :].set(-Rv2[ku, kv_m])
        Zvf = Zvf.at[km, :].set(Zv2[ku, kv_m])

    return Rf, Zf, Ruf, Zuf, Rvf, Zvf, ruuf, ruvf, rvvf, zuuf, zuvf, zvvf


def dense_vmec_nestor_mode_solve_jax(
    *,
    R: Any,
    Z: Any,
    Ru: Any,
    Zu: Any,
    Rv: Any,
    Zv: Any,
    ruu: Any,
    ruv: Any,
    rvv: Any,
    zuu: Any,
    zuv: Any,
    zvv: Any,
    bexni: Any,
    basis: dict[str, Any],
    tables: dict[str, Any],
    signgs: int,
    nvper: int,
    include_analytic: bool = True,
    symmetric: bool = False,
    include_phi_flat: bool = True,
    include_residual: bool = True,
    solve_mode: str = "dense",
    operator_solver: str = "gmres",
    operator_tol: float = 1.0e-11,
    operator_atol: float = 1.0e-13,
    operator_maxiter: int | None = None,
    operator_restart: int | None = None,
) -> dict[str, Any]:
    """Assemble and solve the dense JAX VMEC/NESTOR mode operator.

    This is the first cohesive JAX-native operator API for the free-boundary
    adjoint lane.  It combines the nonsingular Green-function contribution, the
    analytic/singular ``analyt.f`` contribution, VMEC mode projection, and the
    implicit dense mode-space solve.  It is meant for low-resolution validation
    and finite-difference gates before replacing the production host NESTOR
    bridge with a matrix-free/custom-transpose implementation.
    """

    full_grid = _nonsingular_full_grid_from_active_jax(
        R=R,
        Z=Z,
        Ru=Ru,
        Zu=Zu,
        Rv=Rv,
        Zv=Zv,
        ruu=ruu,
        ruv=ruv,
        rvv=rvv,
        zuu=zuu,
        zuv=zuv,
        zvv=zvv,
        basis=basis,
    )
    gsource_nonsing, grpmn_nonsing = vmec_nonsingular_terms_from_bexni_jax(
        R=full_grid[0],
        Z=full_grid[1],
        Ru=full_grid[2],
        Zu=full_grid[3],
        Rv=full_grid[4],
        Zv=full_grid[5],
        ruu=full_grid[6],
        ruv=full_grid[7],
        rvv=full_grid[8],
        zuu=full_grid[9],
        zuv=full_grid[10],
        zvv=full_grid[11],
        bexni=bexni,
        basis=basis,
        tables=tables,
        signgs=signgs,
        nvper=nvper,
    )
    rhs = mode_rhs_from_gsource_jax(
        gsource_nonsing,
        sin_basis=basis["sinmni"],
        cos_basis=basis["cosmni"],
        xmpot=basis["xmpot"],
        n_raw=basis["n_raw"],
        onp=float(basis["onp"]),
        lasym=bool(basis["lasym"]),
        nuv3=int(basis["nuv3"]),
        nuv_full=int(basis["nuv_full"]),
        imirr=basis["imirr"],
        imirr_full=basis["imirr_full"],
    )
    grpmn = grpmn_nonsing
    if bool(include_analytic):
        bvec_analytic, grpmn_analytic = vmec_analytic_terms_from_geometry_jax(
            R=R,
            Ru=Ru,
            Rv=Rv,
            Zu=Zu,
            Zv=Zv,
            ruu=ruu,
            ruv=ruv,
            rvv=rvv,
            zuu=zuu,
            zuv=zuv,
            zvv=zvv,
            bexni=bexni,
            basis=basis,
            signgs=signgs,
        )
        rhs = rhs + bvec_analytic
        grpmn = grpmn + grpmn_analytic

    solve_mode_name = str(solve_mode).strip().lower()
    if solve_mode_name in ("dense", "matrix", "mode_matrix"):
        mode_matrix = mode_matrix_from_grpmn_jax(
            grpmn,
            sin_basis=basis["sinmni"],
            cos_basis=basis["cosmni"],
            xmpot=basis["xmpot"],
            n_raw=basis["n_raw"],
            lasym=bool(basis["lasym"]),
            mn0=int(basis["mn0"]),
        )
        solved = dense_mode_vacuum_solve_jax(
            mode_matrix,
            rhs,
            basis["sinmni"],
            basis["cosmni"] if bool(basis["lasym"]) else None,
            symmetric=symmetric,
            include_phi_flat=bool(include_phi_flat),
            include_residual=bool(include_residual),
        )
        solved["solve_mode"] = "dense"
        solved["mode_matrix_materialized"] = True
    elif solve_mode_name in ("matrix_free", "operator", "operator_gmres", "gmres", "bicgstab"):
        solver_name = "bicgstab" if solve_mode_name == "bicgstab" else str(operator_solver).strip().lower()
        mode_matrix = None
        solved = mode_operator_vacuum_solve_jax(
            grpmn,
            rhs,
            sin_basis=basis["sinmni"],
            cos_basis=basis["cosmni"] if bool(basis["lasym"]) else None,
            xmpot=basis["xmpot"],
            n_raw=basis["n_raw"],
            lasym=bool(basis["lasym"]),
            mn0=int(basis["mn0"]),
            include_phi_flat=bool(include_phi_flat),
            include_residual=bool(include_residual),
            solver=solver_name,
            tol=float(operator_tol),
            atol=float(operator_atol),
            maxiter=operator_maxiter,
            restart=operator_restart,
        )
    else:
        raise ValueError("solve_mode must be 'dense' or 'matrix_free'")
    return {
        **solved,
        "rhs_mode": rhs,
        "mode_matrix": mode_matrix,
        "gsource_nonsing": gsource_nonsing,
        "grpmn": grpmn,
    }


def vacuum_boundary_fields_from_cylindrical_jax(
    *,
    br: Any,
    bp: Any,
    bz: Any,
    R: Any,
    Ru: Any,
    Zu: Any,
    Rv: Any,
    Zv: Any,
    det_floor: float = 1.0e-30,
    include_bnormal_unit: bool = True,
    include_contravariant: bool = True,
) -> dict[str, Any]:
    """JAX version of the VMEC boundary-field projection scaffold.

    This mirrors ``free_boundary.vacuum_boundary_fields_from_cylindrical`` for
    derivative tests.  It intentionally returns a plain dict rather than the
    NumPy dataclass used by the production bridge, so it can be transformed by
    ``jax.grad``/``jax.jacfwd`` while the full NESTOR path is still being
    ported.  Set ``include_contravariant=False`` when the caller only needs
    the covariant boundary channels and normal field before a later mode-space
    reconstruction; the public default keeps the full VMEC-compatible channel
    set.
    """

    br_arr = jnp.asarray(br)
    bp_arr = jnp.asarray(bp)
    bz_arr = jnp.asarray(bz)
    R_arr = jnp.asarray(R)
    Ru_arr = jnp.asarray(Ru)
    Zu_arr = jnp.asarray(Zu)
    Rv_arr = jnp.asarray(Rv)
    Zv_arr = jnp.asarray(Zv)

    g_uu = Ru_arr * Ru_arr + Zu_arr * Zu_arr
    g_uv = Ru_arr * Rv_arr + Zu_arr * Zv_arr
    g_vv = R_arr * R_arr + Rv_arr * Rv_arr + Zv_arr * Zv_arr
    det = g_uu * g_vv - g_uv * g_uv

    bu = br_arr * Ru_arr + bz_arr * Zu_arr
    bv = br_arr * Rv_arr + bp_arr * R_arr + bz_arr * Zv_arr
    n_r = -R_arr * Zu_arr
    n_phi = Zu_arr * Rv_arr - Ru_arr * Zv_arr
    n_z = R_arr * Ru_arr
    bnormal = br_arr * n_r + bp_arr * n_phi + bz_arr * n_z

    result = {
        "bu": bu,
        "bv": bv,
        "bnormal": bnormal,
        "g_uu": g_uu,
        "g_uv": g_uv,
        "g_vv": g_vv,
        "det_guv": det,
    }
    if bool(include_contravariant):
        det_safe = jnp.where(
            jnp.abs(det) >= float(det_floor),
            det,
            jnp.sign(det + 1.0e-300) * float(det_floor),
        )
        bsupu = (g_vv * bu - g_uv * bv) / det_safe
        bsupv = (g_uu * bv - g_uv * bu) / det_safe
        bsqvac = 0.5 * (bu * bsupu + bv * bsupv)
        result.update(
            {
                "bsupu": bsupu,
                "bsupv": bsupv,
                "bsqvac": bsqvac,
            }
        )
    if bool(include_bnormal_unit):
        n_norm = jnp.sqrt(n_r * n_r + n_phi * n_phi + n_z * n_z)
        result["bnormal_unit"] = bnormal / jnp.where(n_norm > 0.0, n_norm, 1.0)
    return result


def vacuum_boundary_fields_from_mode_coeffs_jax(
    mode_coeffs: Any,
    *,
    basis: dict[str, Any],
    bu_ext: Any,
    bv_ext: Any,
    g_uu: Any,
    g_uv: Any,
    g_vv: Any,
) -> dict[str, Any]:
    """JAX replay of VMEC vacuum channels from NESTOR mode coefficients.

    This mirrors the production ``_vacuum_channels_from_sample_potvac`` bridge
    but keeps the calculation transformable for accepted-update gradient tests.
    ``mode_coeffs`` contains the sine potential coefficients followed by cosine
    coefficients when ``basis["lasym"]`` is true.
    """

    pot = jnp.ravel(jnp.asarray(mode_coeffs))
    mnpd = int(basis["mnpd"])
    if int(pot.shape[0]) < mnpd:
        raise ValueError("mode_coeffs_too_small")
    potsin = pot[:mnpd]
    if bool(basis["lasym"]) and int(pot.shape[0]) >= 2 * mnpd:
        potcos = pot[mnpd : 2 * mnpd]
    else:
        potcos = jnp.zeros((mnpd,), dtype=pot.dtype)

    xmpot = jnp.asarray(basis["xmpot"], dtype=pot.dtype)
    n_raw = jnp.asarray(basis["n_raw"], dtype=pot.dtype)
    nfp = jnp.asarray(float(int(basis["nfp"])), dtype=pot.dtype)
    cos_phase = jnp.asarray(basis["cos_phase"], dtype=pot.dtype)
    sin_phase = jnp.asarray(basis["sin_phase"], dtype=pot.dtype)

    potu = cos_phase @ (xmpot * potsin)
    potv = cos_phase @ ((-n_raw * nfp) * potsin)
    if bool(basis["lasym"]):
        potu = potu - (sin_phase @ (xmpot * potcos))
        potv = potv - (sin_phase @ ((-n_raw * nfp) * potcos))

    bu_ext = jnp.asarray(bu_ext)
    bv_ext = jnp.asarray(bv_ext)
    potu = jnp.reshape(potu, bu_ext.shape)
    potv = jnp.reshape(potv, bv_ext.shape)
    bu = bu_ext + potu
    bv = bv_ext + potv
    g_uu = jnp.asarray(g_uu, dtype=bu.dtype)
    g_uv = jnp.asarray(g_uv, dtype=bu.dtype)
    g_vv = jnp.asarray(g_vv, dtype=bu.dtype)
    det = g_uu * g_vv - g_uv * g_uv
    det_safe = jnp.where(jnp.abs(det) > 1.0e-30, det, jnp.sign(det + 1.0e-300) * 1.0e-30)
    bsupu = (g_vv * bu - g_uv * bv) / det_safe
    bsupv = (g_uu * bv - g_uv * bu) / det_safe
    bsqvac = 0.5 * (bu * bsupu + bv * bsupv)
    return {
        "bu": bu,
        "bv": bv,
        "bsupu": bsupu,
        "bsupv": bsupv,
        "bsqvac": bsqvac,
        "det_guv": det,
    }


def direct_coil_boundary_bnormal_rms_jax(
    params: Any,
    *,
    R: Any,
    Z: Any,
    phi: Any,
    Ru: Any,
    Zu: Any,
    Rv: Any,
    Zv: Any,
    br_add: Any = 0.0,
    bp_add: Any = 0.0,
    bz_add: Any = 0.0,
) -> Any:
    """Replay the accepted-boundary direct-coil normal-field RMS in JAX.

    This is the smallest differentiable accepted-output primitive needed by the
    free-boundary coil-adjoint validation ladder.  It holds the VMEC boundary
    geometry fixed, samples the direct Biot-Savart coil field on that accepted
    boundary, projects it into VMEC/NESTOR boundary channels, and returns the
    RMS of ``B_ext · dS``.  It does not differentiate through the nonlinear VMEC
    iteration loop.
    """

    from .external_fields import sample_coil_field_cylindrical

    br, bp, bz = sample_coil_field_cylindrical(
        params,
        jnp.asarray(R),
        jnp.asarray(Z),
        jnp.asarray(phi),
    )
    br = br + jnp.asarray(br_add, dtype=br.dtype)
    bp = bp + jnp.asarray(bp_add, dtype=bp.dtype)
    bz = bz + jnp.asarray(bz_add, dtype=bz.dtype)
    vac = vacuum_boundary_fields_from_cylindrical_jax(
        br=br,
        bp=bp,
        bz=bz,
        R=R,
        Ru=Ru,
        Zu=Zu,
        Rv=Rv,
        Zv=Zv,
        include_contravariant=False,
    )
    bnormal = jnp.ravel(jnp.asarray(vac["bnormal"]))
    return jnp.sqrt(jnp.mean(bnormal * bnormal))


def free_boundary_boundary_geometry_jax(
    state: Any,
    static: Any,
    *,
    sample_nzeta: int | None = None,
) -> dict[str, Any]:
    """Synthesize accepted free-boundary geometry through JAX.

    This helper mirrors the geometry portion of the host-side
    ``_sample_external_boundary_arrays`` path: it applies VMEC's m=1
    internal-to-physical coefficient conversion, evaluates the last radial
    surface on the VMEC/NESTOR angular grid, and returns first and exact modal
    second derivatives.  It intentionally stops before external-field
    sampling, axis overrides, and legacy mgrid interpolation.

    The function is the phase-2 bridge between accepted-state replay and a
    future fully JAX-visible free-boundary loop.  Gradients with respect to the
    accepted VMEC state and direct-coil parameters can pass through this
    geometry, while production ``run_free_boundary`` still uses the established
    host sampler until the full NESTOR loop is ported.
    """

    from .free_boundary import _freeb_boundary_sample_setup
    from .vmec_parity import vmec_m1_internal_to_physical_signed
    from .vmec_realspace import vmec_realspace_synthesis_multi

    cfg = static.cfg
    if sample_nzeta is None:
        sample_nzeta = 1 if (not bool(getattr(cfg, "lthreed", True))) else int(cfg.nzeta)
    setup = _freeb_boundary_sample_setup(static=static, sample_nzeta=int(sample_nzeta))
    trig = setup.trig

    Rcos = jnp.asarray(state.Rcos)
    Rsin = jnp.asarray(state.Rsin)
    Zcos = jnp.asarray(state.Zcos)
    Zsin = jnp.asarray(state.Zsin)
    Rcos, Zsin, Rsin, Zcos = vmec_m1_internal_to_physical_signed(
        Rcos=Rcos,
        Zsin=Zsin,
        Rsin=Rsin,
        Zcos=Zcos,
        modes=static.modes,
        lthreed=bool(getattr(cfg, "lthreed", True)),
        lasym=bool(getattr(cfg, "lasym", False)),
        lconm1=bool(getattr(cfg, "lconm1", True)),
    )

    coeff_cos = jnp.stack([Rcos[-1:, :], Zcos[-1:, :]], axis=0)
    coeff_sin = jnp.stack([Rsin[-1:, :], Zsin[-1:, :]], axis=0)
    base, dtheta, dzeta = vmec_realspace_synthesis_multi(
        coeff_cos=coeff_cos,
        coeff_sin=coeff_sin,
        modes=static.modes,
        trig=trig,
        coeffs_internal=True,
        apply_scalxc=False,
        derivs=("base", "dtheta", "dzeta"),
    )

    second_facs = jnp.asarray(setup.second_facs, dtype=coeff_cos.dtype)
    second_cos = jnp.stack([Rcos[-1:, :], Zcos[-1:, :]], axis=0)[:, None, :, :] * second_facs[None, :, :, :]
    second_sin = jnp.stack([Rsin[-1:, :], Zsin[-1:, :]], axis=0)[:, None, :, :] * second_facs[None, :, :, :]
    second_base = vmec_realspace_synthesis_multi(
        coeff_cos=second_cos,
        coeff_sin=second_sin,
        modes=static.modes,
        trig=trig,
        coeffs_internal=True,
        apply_scalxc=False,
        derivs=("base",),
    )[0]

    R = base[0, 0]
    Z = base[1, 0]
    return {
        "R": R,
        "Z": Z,
        "phi": jnp.asarray(setup.phi_grid, dtype=R.dtype),
        "Ru": dtheta[0, 0],
        "Zu": dtheta[1, 0],
        "Rv": dzeta[0, 0],
        "Zv": dzeta[1, 0],
        "ruu": second_base[0, 0, 0],
        "ruv": second_base[0, 1, 0],
        "rvv": second_base[0, 2, 0],
        "zuu": second_base[1, 0, 0],
        "zuv": second_base[1, 1, 0],
        "zvv": second_base[1, 2, 0],
    }


def _with_jax_nonsingular_replay_tables(
    *,
    basis: dict[str, Any],
    tables: dict[str, Any],
    nv: int,
) -> dict[str, Any]:
    """Add JAX NESTOR replay tables that depend only on static grid data."""

    if "iuv_grid" in tables:
        return tables

    mf = int(basis["mf"])
    nf = int(basis["nf"])
    onp = float(basis["onp"])
    cosv_tab = np.asarray(tables["cosv_tab"], dtype=float)
    sinv_tab = np.asarray(tables["sinv_tab"], dtype=float)
    cosui = np.asarray(tables["cosui"], dtype=float)
    sinui = np.asarray(tables["sinui"], dtype=float)
    nu_fourp = int(cosui.shape[1])
    if nu_fourp <= 0:
        raise ValueError("invalid nonsingular table shape")

    iuv_grid = (np.arange(nu_fourp, dtype=np.int32)[:, None] * int(nv)) + np.arange(int(nv), dtype=np.int32)[
        None, :
    ]
    imirr_full = np.asarray(basis["imirr_full"], dtype=np.int32)
    mf1 = int(mf + 1)
    idx_p_rows: list[int] = []
    idx_m_rows: list[int] = []
    negative_positions: list[int] = []
    flat_pos = 0
    for m in range(mf + 1):
        for n in range(nf + 1):
            idx_p_rows.append(int(m + (n + nf) * mf1))
            if n != 0 and m != 0:
                idx_m_rows.append(int(m + ((-n) + nf) * mf1))
                negative_positions.append(int(flat_pos))
            flat_pos += 1

    enriched = dict(tables)
    enriched.update(
        {
            "iuv_grid": np.asarray(iuv_grid, dtype=np.int32),
            "iref_grid": np.asarray(imirr_full[iuv_grid], dtype=np.int32),
            "cosv_modes": 0.5 * onp * np.asarray(cosv_tab[: nf + 1, :], dtype=float),
            "sinv_modes": 0.5 * onp * np.asarray(sinv_tab[: nf + 1, :], dtype=float),
            "idx_p_flat": np.asarray(idx_p_rows, dtype=np.int32),
            "idx_m_negative": np.asarray(idx_m_rows, dtype=np.int32),
            "negative_positions": np.asarray(negative_positions, dtype=np.int32),
            "sinm_sym": np.asarray(sinui[: mf + 1, :], dtype=float),
            "cosm_sym": -np.asarray(cosui[: mf + 1, :], dtype=float),
            "sinm_asym": np.asarray(cosui[: mf + 1, :], dtype=float),
            "cosm_asym": np.asarray(sinui[: mf + 1, :], dtype=float),
        }
    )
    return enriched


def direct_coil_boundary_replay_context_for_shape(
    static: Any,
    *,
    ntheta: int,
    nzeta: int,
) -> dict[str, Any]:
    """Build shape/static NESTOR replay data for accepted-boundary replay.

    The returned mapping contains the VMEC quadrature weights, mode basis,
    nonsingular-kernel tables, and `nvper` value needed by
    :func:`direct_coil_boundary_bsqvac_from_trace_jax`.  It is intentionally
    separated from the differentiable coil/geometry replay: this setup depends
    only on grid shapes and VMEC static metadata, while the returned arrays are
    treated as fixed context for AD validation and future custom-VJP work.
    """

    from .free_boundary import (
        _build_vmec_mode_basis,
        _ensure_vmec_nonsingular_kernel_tables,
        _vmec_boundary_wint,
    )

    ntheta = int(ntheta)
    nzeta = int(nzeta)
    wint = _vmec_boundary_wint(static=static, ntheta=ntheta, nzeta=nzeta)
    basis = _build_vmec_mode_basis(
        ntheta=ntheta,
        nzeta=nzeta,
        nfp=int(static.cfg.nfp),
        mf=int(static.cfg.mpol) + 1,
        nf=int(static.cfg.ntor),
        lasym=bool(static.cfg.lasym),
        wint=wint,
    )
    nvper = 64 if nzeta == 1 else max(1, int(static.cfg.nfp))
    tables = _ensure_vmec_nonsingular_kernel_tables(basis=basis, nv=nzeta, nvper=nvper)
    tables = _with_jax_nonsingular_replay_tables(basis=basis, tables=tables, nv=nzeta)
    return {
        "basis": basis,
        "tables": tables,
        "wint": wint,
        "nvper": nvper,
        "ntheta": ntheta,
        "nzeta": nzeta,
    }


def direct_coil_boundary_replay_context(
    static: Any,
    geometry: dict[str, Any],
) -> dict[str, Any]:
    """Build static NESTOR replay data for an accepted boundary geometry."""

    R = geometry["R"]
    ntheta, nzeta = (int(v) for v in R.shape)
    return direct_coil_boundary_replay_context_for_shape(
        static,
        ntheta=ntheta,
        nzeta=nzeta,
    )


def _direct_coil_trace_boundary_shape(trace: Mapping[str, Any]) -> tuple[int, int] | None:
    """Infer the active NESTOR boundary grid shape from accepted trace data."""

    nestor_trace = trace.get("freeb_nestor_trace")
    if isinstance(nestor_trace, Mapping):
        for key in ("br_axis", "bp_axis", "bz_axis"):
            axis = nestor_trace.get(key)
            if axis is None:
                continue
            shape = tuple(int(value) for value in np.shape(axis))
            if len(shape) == 2:
                return shape
    bsqvac = trace.get("freeb_bsqvac_half")
    if bsqvac is not None:
        shape = tuple(int(value) for value in np.shape(bsqvac))
        if len(shape) == 2:
            return shape
    return None


def _direct_coil_trace_vacuum_field_override(trace: Mapping[str, Any]) -> dict[str, Any]:
    """Return accepted vacuum-projection arrays from a production NESTOR trace."""

    nestor_trace = trace.get("freeb_nestor_trace", trace)
    if not isinstance(nestor_trace, Mapping):
        raise ValueError("trace must be a NESTOR trace or contain 'freeb_nestor_trace'")
    required_key_map = {
        "bnormal": ("bnormal",),
        "g_uu": ("g_uu",),
        "g_uv": ("g_uv",),
        "g_vv": ("g_vv",),
    }
    missing = tuple(
        source_keys[0] for source_keys in required_key_map.values() if not any(key in nestor_trace for key in source_keys)
    )
    if missing:
        raise ValueError(f"trace is missing vacuum-field override arrays: {missing}")
    out = {
        target_key: jnp.asarray(next(nestor_trace[source_key] for source_key in source_keys if source_key in nestor_trace))
        for target_key, source_keys in required_key_map.items()
    }
    zero_tangent = jnp.zeros_like(out["bnormal"])
    out["bu"] = jnp.asarray(nestor_trace["bexu_ext"]) if "bexu_ext" in nestor_trace else jnp.asarray(nestor_trace.get("bu", zero_tangent))
    out["bv"] = jnp.asarray(nestor_trace["bexv_ext"]) if "bexv_ext" in nestor_trace else jnp.asarray(nestor_trace.get("bv", zero_tangent))
    return out


def direct_coil_boundary_bsqvac_jax(
    params: Any,
    *,
    R: Any,
    Z: Any,
    phi: Any,
    Ru: Any,
    Zu: Any,
    Rv: Any,
    Zv: Any,
    ruu: Any,
    ruv: Any,
    rvv: Any,
    zuu: Any,
    zuv: Any,
    zvv: Any,
    basis: dict[str, Any],
    tables: dict[str, Any],
    signgs: int,
    nvper: int,
    br_add: Any = 0.0,
    bp_add: Any = 0.0,
    bz_add: Any = 0.0,
    wint: Any | None = None,
    include_analytic: bool = True,
    include_diagnostics: bool = True,
    include_mode_diagnostics: bool = True,
    vac_override: Mapping[str, Any] | None = None,
    coil_geometry: Any | None = None,
    nestor_solve_mode: str = "dense",
    nestor_operator_solver: str = "gmres",
    nestor_operator_tol: float = 1.0e-11,
    nestor_operator_atol: float = 1.0e-13,
    nestor_operator_maxiter: int | None = None,
    nestor_operator_restart: int | None = None,
) -> dict[str, Any]:
    """Replay accepted-boundary direct-coil ``bsqvac`` through JAX NESTOR.

    This is the reusable phase-2 validation primitive for the production
    accepted-output ladder.  It holds a VMEC plasma boundary fixed, samples the
    differentiable direct-coil Biot-Savart field on that boundary, projects the
    normal field into VMEC/NESTOR source space, solves the dense JAX mode-space
    vacuum response, and reconstructs ``bsqvac`` on the boundary.

    The helper validates and exposes the differentiable accepted-boundary
    replay contract.  It intentionally does **not** differentiate through the
    outer host-controlled nonlinear VMEC iteration loop.
    """

    from .external_fields import sample_coil_field_cylindrical, sample_coil_field_cylindrical_from_geometry

    R_j = jnp.asarray(R)
    if vac_override is None:
        with _jax_named_scope("vmec_jax.free_boundary.direct_coil_sample"):
            if coil_geometry is None:
                br, bp, bz = sample_coil_field_cylindrical(
                    params,
                    R_j,
                    jnp.asarray(Z),
                    jnp.asarray(phi),
                )
            else:
                br, bp, bz = sample_coil_field_cylindrical_from_geometry(
                    coil_geometry,
                    R_j,
                    jnp.asarray(Z),
                    jnp.asarray(phi),
                    regularization_epsilon=float(getattr(params, "regularization_epsilon", 0.0)),
                    chunk_size=getattr(params, "chunk_size", None),
                )
            br = br + jnp.asarray(br_add, dtype=br.dtype)
            bp = bp + jnp.asarray(bp_add, dtype=bp.dtype)
            bz = bz + jnp.asarray(bz_add, dtype=bz.dtype)
        with _jax_named_scope("vmec_jax.free_boundary.vacuum_boundary_projection"):
            vac = vacuum_boundary_fields_from_cylindrical_jax(
                br=br,
                bp=bp,
                bz=bz,
                R=R_j,
                Ru=Ru,
                Zu=Zu,
                Rv=Rv,
                Zv=Zv,
                include_bnormal_unit=False,
                include_contravariant=False,
            )
    else:
        vac = {
            "bu": jnp.asarray(vac_override["bu"]),
            "bv": jnp.asarray(vac_override["bv"]),
            "bnormal": jnp.asarray(vac_override["bnormal"]),
            "g_uu": jnp.asarray(vac_override["g_uu"]),
            "g_uv": jnp.asarray(vac_override["g_uv"]),
            "g_vv": jnp.asarray(vac_override["g_vv"]),
        }
    if wint is None:
        wint_j = jnp.ones_like(R_j)
    else:
        wint_j = jnp.asarray(wint, dtype=jnp.asarray(vac["bnormal"]).dtype)
    bexni = -jnp.asarray(vac["bnormal"]) * wint_j * ((2.0 * jnp.pi) ** 2)
    with _jax_named_scope("vmec_jax.free_boundary.dense_nestor_mode_solve"):
        mode_solution = dense_vmec_nestor_mode_solve_jax(
            R=R_j,
            Z=Z,
            Ru=Ru,
            Zu=Zu,
            Rv=Rv,
            Zv=Zv,
            ruu=ruu,
            ruv=ruv,
            rvv=rvv,
            zuu=zuu,
            zuv=zuv,
            zvv=zvv,
            bexni=jnp.ravel(bexni),
            basis=basis,
            tables=tables,
            signgs=int(signgs),
            nvper=int(nvper),
            include_analytic=bool(include_analytic),
            include_phi_flat=bool(include_mode_diagnostics),
            include_residual=bool(include_mode_diagnostics),
            solve_mode=str(nestor_solve_mode),
            operator_solver=str(nestor_operator_solver),
            operator_tol=float(nestor_operator_tol),
            operator_atol=float(nestor_operator_atol),
            operator_maxiter=nestor_operator_maxiter,
            operator_restart=nestor_operator_restart,
        )
    with _jax_named_scope("vmec_jax.free_boundary.mode_field_reconstruction"):
        channels = vacuum_boundary_fields_from_mode_coeffs_jax(
            mode_solution["mode_coeffs"],
            basis=basis,
            bu_ext=vac["bu"],
            bv_ext=vac["bv"],
            g_uu=vac["g_uu"],
            g_uv=vac["g_uv"],
            g_vv=vac["g_vv"],
        )
    out = {"bsqvac": channels["bsqvac"]}
    if bool(include_diagnostics):
        out.update(
            {
                "channels": channels,
                "mode_solution": mode_solution,
                "vac": vac,
                "bexni": bexni,
            }
        )
    return out


def direct_coil_boundary_bsqvac_from_trace_jax(
    params: Any,
    geometry: dict[str, Any],
    trace: dict[str, Any],
    *,
    basis: dict[str, Any],
    tables: dict[str, Any],
    signgs: int,
    nvper: int,
    wint: Any,
    include_analytic: bool = True,
    include_diagnostics: bool = True,
    include_mode_diagnostics: bool = True,
    freeze_vacuum_field: bool = False,
    coil_geometry: Any | None = None,
    nestor_solve_mode: str = "dense",
    nestor_operator_solver: str = "gmres",
    nestor_operator_tol: float = 1.0e-11,
    nestor_operator_atol: float = 1.0e-13,
    nestor_operator_maxiter: int | None = None,
    nestor_operator_restart: int | None = None,
) -> dict[str, Any]:
    """Replay direct-coil ``bsqvac`` on accepted geometry using trace metadata.

    ``trace`` may be either a full residual-step trace containing
    ``freeb_nestor_trace`` or the nested NESTOR trace itself.  This keeps the
    production validation ladder from duplicating trace-to-replay plumbing in
    every test while keeping the differentiated path explicit: accepted
    geometry and direct-coil parameters remain JAX-visible, while basis/tables
    and axis-additive fields are captured trace data.
    """

    nestor_trace = trace.get("freeb_nestor_trace", trace)
    if not isinstance(nestor_trace, dict):
        raise ValueError("trace must be a NESTOR trace or contain 'freeb_nestor_trace'")

    vac_override = _direct_coil_trace_vacuum_field_override(trace) if bool(freeze_vacuum_field) else None
    return direct_coil_boundary_bsqvac_jax(
        params,
        R=geometry["R"],
        Z=geometry["Z"],
        phi=geometry["phi"],
        Ru=geometry["Ru"],
        Zu=geometry["Zu"],
        Rv=geometry["Rv"],
        Zv=geometry["Zv"],
        ruu=geometry["ruu"],
        ruv=geometry["ruv"],
        rvv=geometry["rvv"],
        zuu=geometry["zuu"],
        zuv=geometry["zuv"],
        zvv=geometry["zvv"],
        basis=basis,
        tables=tables,
        signgs=int(signgs),
        nvper=int(nvper),
        br_add=jnp.asarray(nestor_trace["br_axis"]),
        bp_add=jnp.asarray(nestor_trace["bp_axis"]),
        bz_add=jnp.asarray(nestor_trace["bz_axis"]),
        wint=jnp.asarray(wint),
        include_analytic=bool(include_analytic),
        include_diagnostics=bool(include_diagnostics),
        include_mode_diagnostics=bool(include_mode_diagnostics),
        vac_override=vac_override,
        coil_geometry=coil_geometry,
        nestor_solve_mode=nestor_solve_mode,
        nestor_operator_solver=nestor_operator_solver,
        nestor_operator_tol=nestor_operator_tol,
        nestor_operator_atol=nestor_operator_atol,
        nestor_operator_maxiter=nestor_operator_maxiter,
        nestor_operator_restart=nestor_operator_restart,
    )


def direct_coil_accepted_trace_replay_objective_jax(
    params: Any,
    initial_state: Any,
    *,
    static: Any,
    traces: Any,
    signgs: int,
    max_steps: int | None = None,
    sample_nzeta: int | None = None,
    include_analytic: bool = True,
    enforce_edge: bool = False,
    state_weight: Any = 1.0,
    force_weight: Any = 0.0,
    bsqvac_weight: Any = 0.0,
    coil_geometry: Any | None = None,
) -> dict[str, Any]:
    """Replay fixed accepted free-boundary traces with differentiable coils.

    This helper is the reusable bridge between accepted-boundary replay and a
    future full nonlinear ``run_free_boundary`` custom adjoint.  A production
    solve supplies accepted trace metadata: step controls, preconditioner
    matrices, axis-additive fields, and NESTOR replay context.  This function
    keeps those controls fixed, while recomputing at every replayed step

    ``state -> boundary geometry -> direct-coil Biot-Savart -> JAX NESTOR
    bsqvac -> strict VMEC update``.

    The result is a small differentiable fixed-control nonlinear replay.  It is
    appropriate for AD-vs-central-FD validation of accepted-output
    sensitivities, but it intentionally does not claim gradients through the
    adaptive host controller that selected the accepted production traces.
    """

    from .discrete_adjoint import strict_update_one_step_from_trace
    from .state import pack_state

    trace_seq = list(traces)
    if max_steps is not None:
        trace_seq = trace_seq[: int(max_steps)]
    if not trace_seq:
        raise ValueError("at least one accepted trace is required")
    reset_flags = _accepted_trace_reset_flags(trace_seq)

    state = initial_state
    objective_components: dict[str, Any] = {
        "state": jnp.asarray(0.0),
        "force": jnp.asarray(0.0),
        "bsqvac": jnp.asarray(0.0),
    }
    context_cache: dict[tuple[int, int], dict[str, Any]] = {}

    def _precomputed_context_for_trace(trace: Mapping[str, Any]) -> dict[str, Any] | None:
        shape = _direct_coil_trace_boundary_shape(trace)
        if shape is None:
            return None
        if shape not in context_cache:
            context_cache[shape] = direct_coil_boundary_replay_context_for_shape(
                static,
                ntheta=shape[0],
                nzeta=shape[1],
            )
        return context_cache[shape]

    steps: list[dict[str, Any]] = []
    bsqvac_values: list[Any] = []
    for trace, reset_to_trace_pre in zip(trace_seq, reset_flags, strict=True):
        if reset_to_trace_pre:
            # VMEC free-boundary turn-on/restart control can reset the working
            # state between accepted trace entries. Preserve that fixed host
            # control transition instead of incorrectly chaining state_post.
            state = trace["state_pre"]
        has_active_freeb_replay = trace.get("freeb_bsqvac_half") is not None and trace.get("freeb_nestor_trace") is not None
        if has_active_freeb_replay:
            with _jax_named_scope("vmec_jax.free_boundary.boundary_geometry"):
                geometry = free_boundary_boundary_geometry_jax(
                    state,
                    static,
                    sample_nzeta=sample_nzeta,
                )
            context = _precomputed_context_for_trace(trace)
            if context is None or tuple(int(v) for v in geometry["R"].shape) != (
                int(context["ntheta"]),
                int(context["nzeta"]),
            ):
                with _jax_named_scope("vmec_jax.free_boundary.replay_context"):
                    context = direct_coil_boundary_replay_context(static, geometry)
            with _jax_named_scope("vmec_jax.free_boundary.direct_coil_bsqvac_replay"):
                replay = direct_coil_boundary_bsqvac_from_trace_jax(
                    params,
                    geometry,
                    trace,
                    basis=context["basis"],
                    tables=context["tables"],
                    signgs=int(signgs),
                    nvper=int(context["nvper"]),
                    wint=jnp.asarray(context["wint"]),
                    include_analytic=bool(include_analytic),
                    coil_geometry=coil_geometry,
                )
            freeb_bsqvac_half = replay["bsqvac"]
        else:
            # Full accepted-trace replay must preserve non-vacuum/setup steps.
            # These steps do not have enough NESTOR metadata to resample coils,
            # so replay the original trace payload and keep coil derivatives
            # zero for that step.
            replay = None
            freeb_bsqvac_half = trace.get("freeb_bsqvac_half", None)
        with _jax_named_scope("vmec_jax.free_boundary.strict_update_one_step_from_trace"):
            step = strict_update_one_step_from_trace(
                state,
                static,
                trace,
                freeb_bsqvac_half=freeb_bsqvac_half,
                enforce_edge=bool(enforce_edge),
            )
        state = step["step"]["state_post"]
        steps.append(step)
        bsqvac_values.append(freeb_bsqvac_half)
        objective_components["force"] = objective_components["force"] + _tree_weighted_half_norm(
            step["force"],
            force_weight,
        )
        if replay is not None:
            objective_components["bsqvac"] = objective_components["bsqvac"] + _weighted_half_norm(
                replay["bsqvac"],
                bsqvac_weight,
            )

    objective_components["state"] = _weighted_half_norm(
        pack_state(state),
        state_weight,
    )
    objective = sum(objective_components.values())
    return {
        "objective": objective,
        "objective_components": objective_components,
        "state": state,
        "steps": steps,
        "bsqvac": bsqvac_values,
        "state_reset_flags": tuple(reset_flags),
    }


def _accepted_trace_state_reset_between(prev_trace: dict[str, Any], trace: dict[str, Any]) -> bool:
    from .state import pack_state

    prev_post = prev_trace.get("state_post")
    next_pre = trace.get("state_pre")
    if prev_post is None or next_pre is None:
        return False
    try:
        prev_packed = np.asarray(pack_state(prev_post), dtype=float)
        next_packed = np.asarray(pack_state(next_pre), dtype=float)
    except Exception:
        return False
    if prev_packed.shape != next_packed.shape:
        return True
    return not np.allclose(prev_packed, next_packed, rtol=1.0e-13, atol=1.0e-13)


def _accepted_trace_reset_flags(trace_seq: Any) -> tuple[bool, ...]:
    traces_tuple = tuple(trace_seq)
    if not traces_tuple:
        return ()
    return (False,) + tuple(
        _accepted_trace_state_reset_between(prev_trace, trace)
        for prev_trace, trace in zip(traces_tuple[:-1], traces_tuple[1:], strict=False)
    )


def direct_coil_accepted_trace_controller_controls_jax(
    traces: Any,
    *,
    accept_mask: Any | None = None,
    done_mask: Any | None = None,
) -> dict[str, Any]:
    """Return stacked JAX-visible controls for fixed accepted trace replay.

    The production trace payloads are still fixed Python data at this rung, but
    control decisions that are naturally stackable are exposed as arrays:
    ``step_index``, ``accept``, ``done``, ``reset_to_trace_pre``, and
    ``has_active_freeb_replay``.  This is the intermediate payload shape that
    the later full stacked replay can extend with update fields accepted by
    ``strict_update_one_step_from_state``.
    """

    trace_seq = tuple(traces)
    if not trace_seq:
        raise ValueError("at least one accepted trace is required")
    step_count = len(trace_seq)
    if accept_mask is None:
        accept_arr = jnp.ones(step_count, dtype=bool)
    else:
        if np.shape(accept_mask) != (step_count,):
            raise ValueError("accept_mask must have shape (n_steps,)")
        accept_arr = jnp.asarray(accept_mask, dtype=bool)
    if done_mask is None:
        done_arr = jnp.arange(step_count, dtype=jnp.int32) == jnp.asarray(step_count - 1, dtype=jnp.int32)
    else:
        if np.shape(done_mask) != (step_count,):
            raise ValueError("done_mask must have shape (n_steps,)")
        done_arr = jnp.asarray(done_mask, dtype=bool)
    return {
        "step_index": jnp.arange(step_count, dtype=jnp.int32),
        "accept": accept_arr,
        "done": done_arr,
        "reset_to_trace_pre": jnp.asarray(_accepted_trace_reset_flags(trace_seq), dtype=bool),
        "has_active_freeb_replay": jnp.asarray(
            [
                trace.get("freeb_bsqvac_half") is not None and trace.get("freeb_nestor_trace") is not None
                for trace in trace_seq
            ],
            dtype=bool,
        ),
    }


_ACCEPTED_TRACE_NUMERIC_CONTROL_KEYS = (
    "dt_eff",
    "b1",
    "fac",
    "force_scale",
    "max_update_rms_pre",
    "lambda_update_scale",
)

_ACCEPTED_TRACE_BOOL_CONTROL_KEYS = (
    "flip_sign",
    "limit_update_rms",
    "divide_by_scalxc_for_update",
    "preconditioner_use_precomputed_tridi",
    "preconditioner_use_lax_tridi",
)

_ACCEPTED_TRACE_REQUIRED_ARRAY_CONTROL_KEYS = (
    "vRcc_before",
    "vRss_before",
    "vZsc_before",
    "vZcs_before",
    "vLsc_before",
    "vLcs_before",
)

_ACCEPTED_TRACE_OPTIONAL_ARRAY_CONTROL_KEYS = (
    "vRsc_before",
    "vRcs_before",
    "vZcc_before",
    "vZss_before",
    "vLcc_before",
    "vLss_before",
)


def _stack_trace_control_field(trace_seq: tuple[dict[str, Any], ...], key: str, *, dtype: Any | None = None) -> Any:
    if not trace_seq:
        raise ValueError("at least one accepted trace is required")
    arrays = []
    for index, trace in enumerate(trace_seq):
        if key not in trace:
            raise KeyError(f"accepted trace {index} is missing control field {key!r}")
        arrays.append(jnp.asarray(trace[key], dtype=dtype))
    shapes = {tuple(arr.shape) for arr in arrays}
    if len(shapes) != 1:
        raise ValueError(f"accepted trace control field {key!r} must have consistent shape")
    return jnp.stack(arrays, axis=0)


def _stack_trace_pytree_field(trace_seq: tuple[dict[str, Any], ...], key: str) -> Any:
    if not trace_seq:
        raise ValueError("at least one accepted trace is required")
    values = []
    for index, trace in enumerate(trace_seq):
        if key not in trace:
            raise KeyError(f"accepted trace {index} is missing control field {key!r}")
        values.append(trace[key])
    treedef = tree_util.tree_structure(values[0])
    for index, value in enumerate(values[1:], start=1):
        if tree_util.tree_structure(value) != treedef:
            raise ValueError(f"accepted trace pytree field {key!r} has inconsistent structure at step {index}")

    def _stack_leaf(*leaves):
        arrays = [jnp.asarray(leaf) for leaf in leaves]
        shapes = {tuple(arr.shape) for arr in arrays}
        if len(shapes) != 1:
            raise ValueError(f"accepted trace pytree field {key!r} must have consistent leaf shapes")
        return jnp.stack(arrays, axis=0)

    return tree_util.tree_map(_stack_leaf, *values)


def _stack_optional_trace_pytree_field(trace_seq: tuple[dict[str, Any], ...], key: str) -> Any | None:
    values = [trace.get(key) for trace in trace_seq]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        return None

    def _stack_leaf(*leaves):
        arrays = [jnp.asarray(leaf) for leaf in leaves]
        shapes = {tuple(arr.shape) for arr in arrays}
        if len(shapes) != 1:
            raise ValueError(f"accepted trace optional field {key!r} must have consistent leaf shapes")
        return jnp.stack(arrays, axis=0)

    return tree_util.tree_map(_stack_leaf, *values)


def direct_coil_accepted_trace_scalar_controls_jax(traces: Any) -> dict[str, Any]:
    """Return stacked scalar/update controls consumed by accepted trace replay.

    This is the next phase-2 payload after the accepted/rejected controller
    masks: fixed host decisions and update scalars are represented as JAX
    arrays with leading dimension ``n_steps``.  The current replay still calls
    ``strict_update_one_step_from_trace`` for behavior parity; this payload is
    the validated interface for replacing per-step trace dictionary reads with
    a fully stacked state-update kernel.
    """

    trace_seq = tuple(traces)
    if not trace_seq:
        raise ValueError("at least one accepted trace is required")
    payload: dict[str, Any] = {}
    for key in _ACCEPTED_TRACE_NUMERIC_CONTROL_KEYS:
        payload[key] = _stack_trace_control_field(trace_seq, key)
    for key in _ACCEPTED_TRACE_BOOL_CONTROL_KEYS:
        payload[key] = _stack_trace_control_field(trace_seq, key, dtype=bool)
    return payload


def direct_coil_accepted_trace_preconditioner_controls_jax(traces: Any) -> dict[str, Any]:
    """Return stacked preconditioner/mode payloads for accepted replay.

    ``precond_jmax`` is intentionally not included yet because the current
    preconditioner application still consumes it via Python ``int(jmax)``.  The
    stacked payload covers fixed array pytrees whose leading scan axis can be
    sliced safely by ``lax.scan``.
    """

    trace_seq = tuple(traces)
    if not trace_seq:
        raise ValueError("at least one accepted trace is required")
    return {
        "precond_mats": _stack_trace_pytree_field(trace_seq, "precond_mats"),
        "lam_prec": _stack_trace_control_field(trace_seq, "lam_prec"),
        "w_mode_mn": _stack_trace_control_field(trace_seq, "w_mode_mn"),
    }


def _trace_preconditioner_policy_value(trace: dict[str, Any], key: str) -> int:
    value = trace.get(key, None)
    if value is None:
        return -1
    arr = np.asarray(value)
    if arr.size == 0:
        return -1
    return 1 if bool(arr.reshape(-1)[0]) else 0


def _trace_preconditioner_static_signature(trace: dict[str, Any]) -> tuple[Any, ...]:
    """Return the static preconditioner branch signature for one trace.

    The current radial preconditioner resolves Python/static XLA dispatch from
    the precomputed-Thomas policy, the ``lax.tridiagonal_solve`` policy,
    ``precond_jmax``, and matrix/mode payload shapes.  Accepted-trace replay
    may differentiate through values inside those arrays, but not through a
    change in this signature.
    """

    return (
        _trace_preconditioner_policy_value(trace, "preconditioner_use_precomputed_tridi"),
        _trace_preconditioner_policy_value(trace, "preconditioner_use_lax_tridi"),
        int(trace.get("precond_jmax", -1)),
        _trace_pytree_shape_signature(trace.get("precond_mats")),
        tuple(np.asarray(trace.get("lam_prec", [])).shape),
        tuple(np.asarray(trace.get("w_mode_mn", [])).shape),
    )


def _trace_static_value_shape_signature(value: Any) -> tuple[Any, ...]:
    """Return a compact signature for static trace payload structure/value."""

    if value is None:
        return ()
    try:
        leaves = tree_util.tree_leaves(value)
    except Exception:
        leaves = [value]
    signature = []
    for leaf in leaves:
        arr = np.asarray(leaf)
        if arr.dtype == object:
            digest = hashlib.sha256(repr(arr.tolist()).encode("utf-8")).hexdigest()
        else:
            digest = hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()
        signature.append((tuple(arr.shape), str(arr.dtype), digest))
    return tuple(signature)


def _trace_optional_presence_signature(trace: dict[str, Any], keys: tuple[str, ...]) -> tuple[int, ...]:
    return tuple(0 if trace.get(key) is None else 1 for key in keys)


_ACCEPTED_TRACE_OPTIONAL_STEP_PYTREE_CONTROL_KEYS = (
    "force_state_pre",
    "freeb_pres_scale",
    "constraint_rcon0",
    "constraint_zcon0",
    "constraint_tcon0",
    "constraint_precond_diag",
    "constraint_tcon",
    "constraint_precond_active",
    "constraint_tcon_active",
)

_ACCEPTED_TRACE_NESTOR_AXIS_KEYS = ("br_axis", "bp_axis", "bz_axis")


def _stack_trace_nestor_axis_controls(trace_seq: tuple[dict[str, Any], ...]) -> dict[str, Any] | None:
    active_nestor = [
        trace.get("freeb_nestor_trace")
        if trace.get("freeb_bsqvac_half") is not None and isinstance(trace.get("freeb_nestor_trace"), dict)
        else None
        for trace in trace_seq
    ]
    if all(nestor_trace is None for nestor_trace in active_nestor):
        return None
    payload = {}
    for key in _ACCEPTED_TRACE_NESTOR_AXIS_KEYS:
        template = None
        for nestor_trace in active_nestor:
            if nestor_trace is not None:
                if key not in nestor_trace:
                    raise KeyError(f"active NESTOR trace is missing axis field {key!r}")
                template = jnp.asarray(nestor_trace[key])
                break
        if template is None:
            continue
        values = []
        for nestor_trace in active_nestor:
            if nestor_trace is None:
                values.append(jnp.zeros_like(template))
                continue
            value = jnp.asarray(nestor_trace[key])
            if tuple(value.shape) != tuple(template.shape):
                raise ValueError(f"NESTOR axis field {key!r} must have consistent shape for stacked replay")
            values.append(value)
        payload[key] = jnp.stack(values, axis=0)
    return payload


def _trace_step_policy_static_signature(trace: dict[str, Any]) -> tuple[Any, ...]:
    """Return the static-dispatch signature for one accepted replay step.

    This signature is stricter than the preconditioner-only segmentation.  It
    separates steps whenever a field consumed through Python/static dispatch or
    a non-stacked optional payload changes shape or presence.  Values inside
    stacked JAX arrays remain differentiable/dynamic controls.
    """

    has_active_freeb_replay = trace.get("freeb_bsqvac_half") is not None and trace.get("freeb_nestor_trace") is not None
    return (
        _trace_preconditioner_static_signature(trace),
        int(bool(trace.get("apply_lforbal", False))),
        int(bool(trace.get("include_edge_residual", False))),
        int(bool(trace.get("apply_m1_constraints", False))),
        int(bool(has_active_freeb_replay)),
        _trace_static_value_shape_signature(trace.get("wout_like")),
        _trace_static_value_shape_signature(trace.get("trig")),
        _trace_static_value_shape_signature(trace.get("zero_m1")),
        _trace_optional_presence_signature(trace, _ACCEPTED_TRACE_OPTIONAL_STEP_PYTREE_CONTROL_KEYS),
        tuple(
            _trace_static_value_shape_signature(trace.get(key))
            for key in _ACCEPTED_TRACE_OPTIONAL_STEP_PYTREE_CONTROL_KEYS
            if trace.get(key) is not None
        ),
    )


def direct_coil_accepted_trace_step_policy_segments(
    traces: Any,
    *,
    max_steps: int | None = None,
) -> list[dict[str, Any]]:
    """Return consecutive static step-policy segments for stacked replay.

    These segments are the safe opt-in seam for moving accepted production
    traces from per-step ``trace`` dictionaries toward stacked JAX controls.
    A segment may call ``strict_update_one_step_from_state`` directly because
    all Python/static dispatch fields have the same signature inside it.
    """

    trace_seq = tuple(traces)
    if max_steps is not None:
        trace_seq = trace_seq[: int(max_steps)]
    if not trace_seq:
        raise ValueError("at least one accepted trace is required")

    segments: list[dict[str, Any]] = []
    start = 0
    current_signature = _trace_step_policy_static_signature(trace_seq[0])
    for index, trace in enumerate(trace_seq[1:], start=1):
        signature = _trace_step_policy_static_signature(trace)
        if signature == current_signature:
            continue
        segments.append(
            {
                "start": start,
                "stop": index,
                "n_steps": index - start,
                "signature": current_signature,
            }
        )
        start = index
        current_signature = signature
    segments.append(
        {
            "start": start,
            "stop": len(trace_seq),
            "n_steps": len(trace_seq) - start,
            "signature": current_signature,
        }
    )
    return segments


def direct_coil_accepted_trace_step_policy_segment_summary(
    traces: Any,
    *,
    accept_mask: Any | None = None,
    done_mask: Any | None = None,
    max_steps: int | None = None,
) -> list[dict[str, Any]]:
    """Return JSON-safe diagnostics for stacked step-policy segments."""

    trace_seq = tuple(traces)
    if max_steps is not None:
        trace_seq = trace_seq[: int(max_steps)]
    if not trace_seq:
        raise ValueError("at least one accepted trace is required")

    n_steps = len(trace_seq)
    if accept_mask is not None:
        accept_mask = np.asarray(accept_mask, dtype=bool)[:n_steps]
    if done_mask is not None:
        done_mask = np.asarray(done_mask, dtype=bool)[:n_steps]
    controls = direct_coil_accepted_trace_controller_controls_jax(
        trace_seq,
        accept_mask=accept_mask,
        done_mask=done_mask,
    )
    accepted = np.asarray(controls["accept"], dtype=bool)
    done = np.asarray(controls["done"], dtype=bool)
    reset = np.asarray(controls["reset_to_trace_pre"], dtype=bool)
    freeb = np.asarray(controls["has_active_freeb_replay"], dtype=bool)

    summaries: list[dict[str, Any]] = []
    for index, segment in enumerate(direct_coil_accepted_trace_step_policy_segments(trace_seq)):
        start = int(segment["start"])
        stop = int(segment["stop"])
        segment_accept = accepted[start:stop]
        segment_done = done[start:stop]
        segment_reset = reset[start:stop]
        segment_freeb = freeb[start:stop]
        summaries.append(
            {
                "index": int(index),
                "start": start,
                "stop": stop,
                "n_steps": int(stop - start),
                "accepted_steps": int(np.count_nonzero(segment_accept)),
                "rejected_steps": int(segment_accept.size - np.count_nonzero(segment_accept)),
                "done_markers": int(np.count_nonzero(segment_done)),
                "state_resets": int(np.count_nonzero(segment_reset)),
                "free_boundary_replay_steps": int(np.count_nonzero(segment_freeb)),
                "signature_repr": repr(segment["signature"]),
            }
        )
    return summaries


def direct_coil_accepted_trace_preconditioner_policy_segments(
    traces: Any,
    *,
    max_steps: int | None = None,
) -> list[dict[str, Any]]:
    """Return consecutive static-preconditioner-policy segments.

    This is the explicit phase-2 planning primitive for replacing one
    per-accepted-step ``lax.switch`` branch with a smaller set of static
    subcontrollers.  Each returned segment has a half-open ``[start, stop)``
    step range whose traces share the same preconditioner policy, active
    radial solve size, and preconditioner/mode payload shapes.  A future
    production controller can use these ranges to keep the existing
    preconditioner JIT-cache dispatch static while moving the surrounding
    nonlinear controller into JAX-visible control flow.
    """

    trace_seq = tuple(traces)
    if max_steps is not None:
        trace_seq = trace_seq[: int(max_steps)]
    if not trace_seq:
        raise ValueError("at least one accepted trace is required")

    segments: list[dict[str, Any]] = []
    start = 0
    current_signature = _trace_preconditioner_static_signature(trace_seq[0])
    for index, trace in enumerate(trace_seq[1:], start=1):
        signature = _trace_preconditioner_static_signature(trace)
        if signature == current_signature:
            continue
        segments.append(
            {
                "start": start,
                "stop": index,
                "n_steps": index - start,
                "signature": current_signature,
            }
        )
        start = index
        current_signature = signature
    segments.append(
        {
            "start": start,
            "stop": len(trace_seq),
            "n_steps": len(trace_seq) - start,
            "signature": current_signature,
        }
    )
    return segments


def direct_coil_accepted_trace_preconditioner_policy_segment_summary(
    traces: Any,
    *,
    accept_mask: Any | None = None,
    done_mask: Any | None = None,
    max_steps: int | None = None,
) -> list[dict[str, Any]]:
    """Return JSON-safe preconditioner-policy segment diagnostics.

    The raw segment signatures are intentionally precise Python tuples for
    equality checks.  This summary is the user-facing diagnostic payload for
    accepted-controller replay: each entry records the half-open step range,
    static preconditioner policy, and how many accepted, rejected, free-boundary
    replay, reset, and done-marker slots live in that range.
    """

    trace_seq = tuple(traces)
    if max_steps is not None:
        trace_seq = trace_seq[: int(max_steps)]
    if not trace_seq:
        raise ValueError("at least one accepted trace is required")

    n_steps = len(trace_seq)
    if accept_mask is not None:
        accept_mask = np.asarray(accept_mask, dtype=bool)[:n_steps]
    if done_mask is not None:
        done_mask = np.asarray(done_mask, dtype=bool)[:n_steps]
    controls = direct_coil_accepted_trace_controller_controls_jax(
        trace_seq,
        accept_mask=accept_mask,
        done_mask=done_mask,
    )
    accepted = np.asarray(controls["accept"], dtype=bool)
    done = np.asarray(controls["done"], dtype=bool)
    reset = np.asarray(controls["reset_to_trace_pre"], dtype=bool)
    freeb = np.asarray(controls["has_active_freeb_replay"], dtype=bool)

    summaries: list[dict[str, Any]] = []
    for index, segment in enumerate(direct_coil_accepted_trace_preconditioner_policy_segments(trace_seq)):
        start = int(segment["start"])
        stop = int(segment["stop"])
        signature = segment["signature"]
        segment_accept = accepted[start:stop]
        segment_done = done[start:stop]
        segment_reset = reset[start:stop]
        segment_freeb = freeb[start:stop]
        summaries.append(
            {
                "index": int(index),
                "start": start,
                "stop": stop,
                "n_steps": int(stop - start),
                "accepted_steps": int(np.count_nonzero(segment_accept)),
                "rejected_steps": int(segment_accept.size - np.count_nonzero(segment_accept)),
                "done_markers": int(np.count_nonzero(segment_done)),
                "state_resets": int(np.count_nonzero(segment_reset)),
                "free_boundary_replay_steps": int(np.count_nonzero(segment_freeb)),
                "preconditioner_use_precomputed_tridi": int(signature[0]),
                "preconditioner_use_lax_tridi": int(signature[1]),
                "precond_jmax": int(signature[2]),
                "signature_repr": repr(signature),
            }
        )
    return summaries


def _accepted_trace_effective_controller_masks(controls: Mapping[str, Any]) -> dict[str, Any]:
    """Return effective accepted/rejected/done masks for controller controls."""

    accept_control = np.asarray(controls["accept"], dtype=bool)
    done_control = np.asarray(controls["done"], dtype=bool)
    active_values = []
    accepted_values = []
    rejected_values = []
    done_values = []
    done = False
    for accept_i, done_i in zip(accept_control, done_control, strict=True):
        active = not done
        accepted = bool(active and accept_i)
        rejected = bool(active and not accept_i)
        done = bool(done or (accepted and done_i))
        active_values.append(active)
        accepted_values.append(accepted)
        rejected_values.append(rejected)
        done_values.append(done)
    return {
        "accept_control": jnp.asarray(accept_control, dtype=bool),
        "done_control": jnp.asarray(done_control, dtype=bool),
        "active": jnp.asarray(active_values, dtype=bool),
        "accepted": jnp.asarray(accepted_values, dtype=bool),
        "rejected": jnp.asarray(rejected_values, dtype=bool),
        "done": jnp.asarray(done_values, dtype=bool),
        "reset_to_trace_pre": jnp.asarray(controls["reset_to_trace_pre"], dtype=bool),
        "has_active_freeb_replay": jnp.asarray(controls["has_active_freeb_replay"], dtype=bool),
    }


def _accepted_trace_segment_is_unconditionally_accepted(
    masks: Mapping[str, Any],
    *,
    start: int,
    stop: int,
) -> bool:
    """Return whether a controller segment can skip accept/reject conditionals."""

    active = np.asarray(masks["active"], dtype=bool)[int(start) : int(stop)]
    accepted = np.asarray(masks["accepted"], dtype=bool)[int(start) : int(stop)]
    rejected = np.asarray(masks["rejected"], dtype=bool)[int(start) : int(stop)]
    done = np.asarray(masks["done"], dtype=bool)[int(start) : int(stop)]
    if active.size == 0:
        return False
    if not bool(np.all(active)):
        return False
    if not bool(np.all(accepted)):
        return False
    if bool(np.any(rejected)):
        return False
    # A final done marker is allowed. Any earlier done marker would make later
    # scan entries inactive in the ordinary controller, so it must not fast-path.
    if done.size > 1 and bool(np.any(done[:-1])):
        return False
    return True


def direct_coil_accepted_trace_branch_metadata(
    traces: Any,
    *,
    accept_mask: Any | None = None,
    done_mask: Any | None = None,
    max_steps: int | None = None,
    json_safe: bool = False,
) -> dict[str, Any]:
    """Return branch metadata for a fixed accepted free-boundary trace.

    This is the production-facing phase-2 seam between the host adaptive
    free-boundary controller and any fixed-branch custom-VJP wrapper.  It
    packages the branch-control fingerprint, accepted/done/reset masks, active
    direct-coil replay cadence, and static preconditioner segments in one
    payload so derivative gates can fail explicitly when a finite-difference
    perturbation follows a different branch.
    """

    trace_seq = tuple(traces)
    if max_steps is not None:
        trace_seq = trace_seq[: int(max_steps)]
    if not trace_seq:
        raise ValueError("at least one accepted trace is required")

    n_steps = len(trace_seq)
    if accept_mask is not None:
        accept_mask = np.asarray(accept_mask, dtype=bool)[:n_steps]
    if done_mask is not None:
        done_mask = np.asarray(done_mask, dtype=bool)[:n_steps]

    controls = direct_coil_accepted_trace_controller_controls_jax(
        trace_seq,
        accept_mask=accept_mask,
        done_mask=done_mask,
    )
    masks = _accepted_trace_effective_controller_masks(controls)
    freeb = jnp.asarray(controls["has_active_freeb_replay"], dtype=bool)
    active_freeb = jnp.logical_and(jnp.asarray(masks["accepted"], dtype=bool), freeb)
    metadata = {
        "n_steps": int(n_steps),
        "n_free_boundary_replay_steps": int(np.count_nonzero(np.asarray(active_freeb, dtype=bool))),
        "fingerprint": direct_coil_accepted_trace_fingerprint(trace_seq),
        "controller_controls": controls,
        "masks": masks,
        "accepted_mask": jnp.asarray(masks["accepted"], dtype=bool),
        "rejected_mask": jnp.asarray(masks["rejected"], dtype=bool),
        "done_mask": jnp.asarray(masks["done"], dtype=bool),
        "reset_to_trace_pre": jnp.asarray(masks["reset_to_trace_pre"], dtype=bool),
        "has_active_freeb_replay": freeb,
        "active_free_boundary_mask": active_freeb,
        "preconditioner_policy_segments": direct_coil_accepted_trace_preconditioner_policy_segments(trace_seq),
        "preconditioner_policy_segment_summary": direct_coil_accepted_trace_preconditioner_policy_segment_summary(
            trace_seq,
            accept_mask=accept_mask,
            done_mask=done_mask,
        ),
    }
    if json_safe:
        return _json_safe_fingerprint_value(metadata)
    return metadata


def _unique_shape_list(shapes: list[tuple[int, ...]]) -> list[list[int]]:
    seen: set[tuple[int, ...]] = set()
    unique: list[list[int]] = []
    for shape in shapes:
        if shape in seen:
            continue
        seen.add(shape)
        unique.append([int(value) for value in shape])
    return unique


def _compact_segment_summaries(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop exact static signatures from timing metadata.

    The full signatures remain available through fingerprint diagnostics.  The
    replay graph timing payload only needs counts and ranges; keeping hashes of
    every static array makes JSON reports noisy and unnecessarily large.
    """

    return [{key: value for key, value in summary.items() if key != "signature_repr"} for summary in summaries]


def direct_coil_accepted_trace_replay_graph_metadata(
    traces: Any,
    *,
    static: Any | None = None,
    accept_mask: Any | None = None,
    done_mask: Any | None = None,
    max_steps: int | None = None,
    sample_nzeta: int | None = None,
    include_analytic: bool = True,
    use_stacked_step_controls: bool = True,
    use_accepted_only_fast_path: bool = True,
    json_safe: bool = False,
) -> dict[str, Any]:
    """Return profiling metadata for the fixed accepted-branch replay graph.

    The values here are deliberately structural: they describe the replay graph
    being traced rather than its physics result.  This makes timing reports
    comparable across CPU/GPU runs and distinguishes a true kernel/runtime
    regression from a changed trace length, segmentation, or active
    free-boundary cadence.
    """

    trace_seq = tuple(traces)
    if max_steps is not None:
        trace_seq = trace_seq[: int(max_steps)]
    if not trace_seq:
        raise ValueError("at least one accepted trace is required")

    n_steps = len(trace_seq)
    if accept_mask is not None:
        accept_mask = np.asarray(accept_mask, dtype=bool)[:n_steps]
    if done_mask is not None:
        done_mask = np.asarray(done_mask, dtype=bool)[:n_steps]
    controls = direct_coil_accepted_trace_controller_controls_jax(
        trace_seq,
        accept_mask=accept_mask,
        done_mask=done_mask,
    )
    masks = _accepted_trace_effective_controller_masks(controls)
    accepted = np.asarray(masks["accepted"], dtype=bool)
    rejected = np.asarray(masks["rejected"], dtype=bool)
    done = np.asarray(masks["done"], dtype=bool)
    reset = np.asarray(masks["reset_to_trace_pre"], dtype=bool)
    freeb = np.asarray(controls["has_active_freeb_replay"], dtype=bool)
    active_freeb = np.logical_and(accepted, freeb)

    boundary_shapes: list[tuple[int, ...]] = []
    bsqvac_half_shapes: list[tuple[int, ...]] = []
    nestor_axis_shapes: list[tuple[int, ...]] = []
    for trace in trace_seq:
        if trace.get("freeb_bsqvac_half") is not None:
            bsqvac_half_shapes.append(tuple(int(v) for v in np.shape(trace["freeb_bsqvac_half"])))
        nestor_trace = trace.get("freeb_nestor_trace")
        if isinstance(nestor_trace, Mapping):
            for key in ("br_axis", "bp_axis", "bz_axis"):
                if nestor_trace.get(key) is None:
                    continue
                shape = tuple(int(v) for v in np.shape(nestor_trace[key]))
                nestor_axis_shapes.append(shape)
                if len(shape) == 2:
                    boundary_shapes.append(shape)
                break

    inferred_boundary_shape = boundary_shapes[0] if boundary_shapes else None
    static_cfg = getattr(static, "cfg", None)
    nfp = None if static_cfg is None else int(static_cfg.nfp)
    mpol = None if static_cfg is None else int(static_cfg.mpol)
    ntor = None if static_cfg is None else int(static_cfg.ntor)
    lasym = None if static_cfg is None else bool(static_cfg.lasym)
    nvper = None
    if inferred_boundary_shape is not None and nfp is not None:
        nzeta = int(inferred_boundary_shape[1])
        nvper = 64 if nzeta == 1 else max(1, int(nfp))

    metadata = {
        "contract": "fixed accepted-branch replay graph metadata",
        "differentiates_adaptive_controller": False,
        "n_steps": int(n_steps),
        "accepted_steps": int(np.count_nonzero(accepted)),
        "rejected_steps": int(np.count_nonzero(rejected)),
        "done_markers": int(np.count_nonzero(done)),
        "state_resets": int(np.count_nonzero(reset)),
        "free_boundary_trace_steps": int(np.count_nonzero(freeb)),
        "active_free_boundary_replay_steps": int(np.count_nonzero(active_freeb)),
        "step_policy_n_segments": int(len(direct_coil_accepted_trace_step_policy_segments(trace_seq))),
        "step_policy_segment_summary": _compact_segment_summaries(
            direct_coil_accepted_trace_step_policy_segment_summary(
                trace_seq,
                accept_mask=accept_mask,
                done_mask=done_mask,
            )
        ),
        "preconditioner_policy_n_segments": int(len(direct_coil_accepted_trace_preconditioner_policy_segments(trace_seq))),
        "preconditioner_policy_segment_summary": _compact_segment_summaries(
            direct_coil_accepted_trace_preconditioner_policy_segment_summary(
                trace_seq,
                accept_mask=accept_mask,
                done_mask=done_mask,
            )
        ),
        "boundary_shapes": _unique_shape_list(boundary_shapes),
        "bsqvac_half_shapes": _unique_shape_list(bsqvac_half_shapes),
        "nestor_axis_shapes": _unique_shape_list(nestor_axis_shapes),
        "inferred_boundary_shape": None
        if inferred_boundary_shape is None
        else [int(value) for value in inferred_boundary_shape],
        "sample_nzeta": None if sample_nzeta is None else int(sample_nzeta),
        "nfp": nfp,
        "mpol": mpol,
        "ntor": ntor,
        "lasym": lasym,
        "nvper": nvper,
        "include_analytic": bool(include_analytic),
        "use_stacked_step_controls": bool(use_stacked_step_controls),
        "use_accepted_only_fast_path": bool(use_accepted_only_fast_path),
    }
    if json_safe:
        return _json_safe_fingerprint_value(metadata)
    return metadata


def _direct_coil_boundary_replay_contexts_by_shape(static: Any, trace_seq: tuple[Any, ...]) -> dict[tuple[int, int], Any]:
    """Precompute fixed NESTOR replay contexts keyed by active boundary shape."""

    contexts: dict[tuple[int, int], Any] = {}
    for trace in trace_seq:
        shape = _direct_coil_trace_boundary_shape(trace)
        if shape is None or shape in contexts:
            continue
        contexts[shape] = direct_coil_boundary_replay_context_for_shape(
            static,
            ntheta=shape[0],
            nzeta=shape[1],
        )
    return contexts


def _slice_replay_controls(controls: Mapping[str, Any], *, start: int, stop: int) -> dict[str, Any]:
    """Slice stacked replay controls without rebuilding them from traces."""

    return tree_util.tree_map(
        lambda value, start=start, stop=stop: jnp.asarray(value)[start:stop],
        controls,
    )


def direct_coil_accepted_trace_controller_replay_plan(
    traces: Any,
    *,
    static: Any,
    accept_mask: Any | None = None,
    done_mask: Any | None = None,
    max_steps: int | None = None,
    use_preconditioner_policy_segments: bool = False,
    use_segment_preconditioner_controls: bool = False,
    use_stacked_step_controls: bool = False,
    use_accepted_only_fast_path: bool = True,
) -> dict[str, Any]:
    """Build fixed accepted-branch replay controls outside AD transforms.

    Branch-local production reports repeatedly replay a saved complete-solve
    branch under ``jax.jvp`` or ``jax.vjp``.  This plan hoists trace-derived
    masks, stacked controls, static segment ranges, and boundary replay
    contexts out of the transformed replay function.  It does not change the
    derivative contract: the adaptive host controller remains fixed and
    fingerprint-gated, while only the accepted direct-coil replay branch is
    differentiated.
    """

    trace_seq = tuple(traces)
    if max_steps is not None:
        trace_seq = trace_seq[: int(max_steps)]
    if not trace_seq:
        raise ValueError("at least one accepted trace is required")

    controller_controls = direct_coil_accepted_trace_controller_controls_jax(
        trace_seq,
        accept_mask=accept_mask,
        done_mask=done_mask,
    )
    effective_masks = _accepted_trace_effective_controller_masks(controller_controls)
    scalar_controls = direct_coil_accepted_trace_scalar_controls_jax(trace_seq)
    array_controls = direct_coil_accepted_trace_array_controls_jax(trace_seq)
    step_controls = direct_coil_accepted_trace_step_controls_jax(trace_seq) if bool(use_stacked_step_controls) else None
    step_scalar_controls = {
        key: value
        for key, value in scalar_controls.items()
        if key
        not in (
            "preconditioner_use_precomputed_tridi",
            "preconditioner_use_lax_tridi",
        )
    }
    controls = {**controller_controls, "step_scalars": step_scalar_controls, "step_arrays": array_controls}
    preconditioner_controls = None
    preconditioner_controls_stacked = True
    try:
        preconditioner_controls = direct_coil_accepted_trace_preconditioner_controls_jax(trace_seq)
    except (KeyError, ValueError):
        preconditioner_controls_stacked = False
    else:
        controls = {**controls, "step_preconditioner": preconditioner_controls}
    if step_controls is not None:
        controls = {**controls, "step_controls": step_controls}

    preconditioner_policy_segments = direct_coil_accepted_trace_preconditioner_policy_segments(trace_seq)
    preconditioner_policy_segment_summary = direct_coil_accepted_trace_preconditioner_policy_segment_summary(
        trace_seq,
        accept_mask=accept_mask,
        done_mask=done_mask,
    )
    step_policy_segments = direct_coil_accepted_trace_step_policy_segments(trace_seq)
    step_policy_segment_summary = direct_coil_accepted_trace_step_policy_segment_summary(
        trace_seq,
        accept_mask=accept_mask,
        done_mask=done_mask,
    )

    control_segments: tuple[dict[str, Any], ...] | None = None
    segment_preconditioner_controls_stacked: tuple[bool, ...] = ()
    accepted_only_fast_path_segments: tuple[bool, ...] = ()
    segment_source = "none"
    if bool(use_stacked_step_controls):
        segment_source = "step_policy"
        control_segments_list = []
        segment_preconditioner_controls_stacked_list = []
        accepted_only_fast_path_segments_list = []
        for segment in step_policy_segments:
            start = int(segment["start"])
            stop = int(segment["stop"])
            segment_controls = _slice_replay_controls(controls, start=start, stop=stop)
            if preconditioner_controls is None:
                try:
                    segment_preconditioner_controls = direct_coil_accepted_trace_preconditioner_controls_jax(
                        trace_seq[start:stop]
                    )
                except (KeyError, ValueError) as exc:
                    raise ValueError("stacked step replay requires stackable preconditioner controls per segment") from exc
                segment_controls = {**segment_controls, "step_preconditioner": segment_preconditioner_controls}
            segment_preconditioner_controls_stacked_list.append(True)
            accepted_only_fast_path_segments_list.append(
                bool(use_accepted_only_fast_path)
                and _accepted_trace_segment_is_unconditionally_accepted(effective_masks, start=start, stop=stop)
            )
            control_segments_list.append(segment_controls)
        control_segments = tuple(control_segments_list)
        segment_preconditioner_controls_stacked = tuple(segment_preconditioner_controls_stacked_list)
        accepted_only_fast_path_segments = tuple(accepted_only_fast_path_segments_list)
    elif bool(use_preconditioner_policy_segments):
        segment_source = "preconditioner_policy"
        control_segments_list = []
        segment_preconditioner_controls_stacked_list = []
        accepted_only_fast_path_segments_list = []
        for segment in preconditioner_policy_segments:
            start = int(segment["start"])
            stop = int(segment["stop"])
            segment_controls = _slice_replay_controls(controls, start=start, stop=stop)
            if preconditioner_controls is None and bool(use_segment_preconditioner_controls):
                try:
                    segment_preconditioner_controls = direct_coil_accepted_trace_preconditioner_controls_jax(
                        trace_seq[start:stop]
                    )
                except (KeyError, ValueError):
                    segment_preconditioner_controls_stacked_list.append(False)
                else:
                    segment_controls = {**segment_controls, "step_preconditioner": segment_preconditioner_controls}
                    segment_preconditioner_controls_stacked_list.append(True)
            elif preconditioner_controls is None:
                segment_preconditioner_controls_stacked_list.append(False)
            else:
                segment_preconditioner_controls_stacked_list.append(True)
            accepted_only_fast_path_segments_list.append(
                bool(use_accepted_only_fast_path)
                and _accepted_trace_segment_is_unconditionally_accepted(effective_masks, start=start, stop=stop)
            )
            control_segments_list.append(segment_controls)
        control_segments = tuple(control_segments_list)
        segment_preconditioner_controls_stacked = tuple(segment_preconditioner_controls_stacked_list)
        accepted_only_fast_path_segments = tuple(accepted_only_fast_path_segments_list)
    else:
        accepted_only_fast_path_segments = (
            bool(use_accepted_only_fast_path)
            and _accepted_trace_segment_is_unconditionally_accepted(effective_masks, start=0, stop=len(trace_seq)),
        )

    return {
        "contract": "fixed accepted-branch controller replay plan",
        "differentiates_adaptive_controller": False,
        "traces": trace_seq,
        "controls": controls,
        "effective_masks": effective_masks,
        "scalar_controls": scalar_controls,
        "array_controls": array_controls,
        "step_controls": step_controls,
        "preconditioner_controls": preconditioner_controls,
        "preconditioner_controls_stacked": bool(preconditioner_controls_stacked),
        "preconditioner_policy_segments": preconditioner_policy_segments,
        "preconditioner_policy_segment_summary": preconditioner_policy_segment_summary,
        "step_policy_segments": step_policy_segments,
        "step_policy_segment_summary": step_policy_segment_summary,
        "control_segments": control_segments,
        "segment_source": segment_source,
        "preconditioner_controls_segment_stacked": segment_preconditioner_controls_stacked,
        "accepted_only_fast_path_segments": accepted_only_fast_path_segments,
        "boundary_replay_contexts_by_shape": _direct_coil_boundary_replay_contexts_by_shape(static, trace_seq),
        "options": {
            "max_steps": None if max_steps is None else int(max_steps),
            "use_preconditioner_policy_segments": bool(use_preconditioner_policy_segments),
            "use_segment_preconditioner_controls": bool(use_segment_preconditioner_controls),
            "use_stacked_step_controls": bool(use_stacked_step_controls),
            "use_accepted_only_fast_path": bool(use_accepted_only_fast_path),
        },
    }


def _extract_adjoint_step_trace(source: Any) -> tuple[Any, ...]:
    if isinstance(source, Mapping):
        if "adjoint_step_trace" in source:
            return tuple(source["adjoint_step_trace"])
        if "diagnostics" in source and isinstance(source["diagnostics"], Mapping):
            diagnostics = source["diagnostics"]
            if "adjoint_step_trace" in diagnostics:
                return tuple(diagnostics["adjoint_step_trace"])
    diagnostics = getattr(source, "diagnostics", None)
    if isinstance(diagnostics, Mapping) and "adjoint_step_trace" in diagnostics:
        return tuple(diagnostics["adjoint_step_trace"])
    result = getattr(source, "result", None)
    result_diagnostics = getattr(result, "diagnostics", None)
    if isinstance(result_diagnostics, Mapping) and "adjoint_step_trace" in result_diagnostics:
        return tuple(result_diagnostics["adjoint_step_trace"])
    if isinstance(source, (str, bytes)):
        raise RuntimeError(
            "No adjoint_step_trace found. Run the residual solver with "
            "adjoint_trace=True and adjoint_trace_mode='full'."
        )
    try:
        traces = tuple(source)
    except TypeError as exc:
        raise RuntimeError(
            "No adjoint_step_trace found. Run the residual solver with "
            "adjoint_trace=True and adjoint_trace_mode='full'."
        ) from exc
    if traces and all(isinstance(trace, Mapping) for trace in traces):
        return traces
    raise RuntimeError(
        "No adjoint_step_trace found. Run the residual solver with "
        "adjoint_trace=True and adjoint_trace_mode='full'."
    )


def _stackability_probe(name: str, fn: Any, traces: tuple[Any, ...]) -> tuple[bool, str | None]:
    try:
        fn(traces)
    except Exception as exc:
        return False, f"{name}: {exc}"
    return True, None


def free_boundary_adjoint_trace_replay_diagnostics(
    source: Any,
    *,
    accept_mask: Any | None = None,
    done_mask: Any | None = None,
    max_steps: int | None = None,
    json_safe: bool = False,
) -> dict[str, Any]:
    """Return diagnostics for fixed accepted-trace free-boundary replay.

    The returned contract is intentionally conservative: it describes a fixed
    accepted-branch replay payload and explicitly does *not* claim that the
    adaptive host controller is differentiated.  Callers should use it to gate
    complete-solve finite-difference comparisons before invoking any
    branch-local custom VJP.
    """

    traces = _extract_adjoint_step_trace(source)
    if max_steps is not None:
        traces = traces[: int(max_steps)]
    if not traces:
        raise RuntimeError(
            "adjoint_step_trace is empty. Run the residual solver with "
            "adjoint_trace=True and adjoint_trace_mode='full'."
        )
    metadata = direct_coil_accepted_trace_branch_metadata(
        traces,
        accept_mask=accept_mask,
        done_mask=done_mask,
        max_steps=max_steps,
        json_safe=False,
    )
    scalar_ok, scalar_error = _stackability_probe(
        "scalar_controls",
        direct_coil_accepted_trace_scalar_controls_jax,
        traces,
    )
    array_ok, array_error = _stackability_probe(
        "array_controls",
        direct_coil_accepted_trace_array_controls_jax,
        traces,
    )
    preconditioner_ok, preconditioner_error = _stackability_probe(
        "preconditioner_controls",
        direct_coil_accepted_trace_preconditioner_controls_jax,
        traces,
    )
    errors = {
        key: value
        for key, value in {
            "scalar_controls": scalar_error,
            "array_controls": array_error,
            "preconditioner_controls": preconditioner_error,
        }.items()
        if value is not None
    }
    diagnostics = {
        "contract": "fixed accepted-trace replay diagnostics only",
        "differentiates_adaptive_controller": False,
        "n_steps": metadata["n_steps"],
        "branch_fingerprint": metadata["fingerprint"],
        "masks": metadata["masks"],
        "replay_diagnostics": {
            "preconditioner_policy_n_segments": len(metadata["preconditioner_policy_segments"]),
            "preconditioner_policy_segment_summary": metadata["preconditioner_policy_segment_summary"],
            "scalar_controls_stackable": bool(scalar_ok),
            "array_controls_stackable": bool(array_ok),
            "preconditioner_controls_stackable": bool(preconditioner_ok),
            "errors": errors,
        },
    }
    if json_safe:
        return _json_safe_fingerprint_value(diagnostics)
    return diagnostics


def direct_coil_accepted_trace_array_controls_jax(traces: Any) -> dict[str, Any]:
    """Return stacked array-valued update controls for accepted trace replay.

    The accepted VMEC state update uses velocity-history arrays captured before
    each accepted step.  These arrays are fixed host-control data, not outputs
    of the direct-coil replay.  Stacking them here moves another payload class
    into the JAX-visible scan while preserving the legacy trace fallback for
    optional asymmetric channels.
    """

    trace_seq = tuple(traces)
    if not trace_seq:
        raise ValueError("at least one accepted trace is required")
    payload: dict[str, Any] = {}
    for key in _ACCEPTED_TRACE_REQUIRED_ARRAY_CONTROL_KEYS:
        payload[key] = _stack_trace_control_field(trace_seq, key)
    for key in _ACCEPTED_TRACE_OPTIONAL_ARRAY_CONTROL_KEYS:
        values = [trace.get(key) for trace in trace_seq]
        if all(value is None for value in values):
            continue
        if any(value is None for value in values):
            raise ValueError(f"accepted trace optional array field {key!r} must be present for every step or none")
        payload[key] = _stack_trace_control_field(trace_seq, key)
    return payload


def direct_coil_accepted_trace_step_controls_jax(
    traces: Any,
    *,
    include_state_pre: bool = True,
    include_force_state_pre: bool = True,
    include_nestor_axes: bool = True,
    include_constraints: bool = True,
) -> dict[str, Any]:
    """Return stacked state/constraint controls for direct accepted replay.

    The controls in this payload are sliced by ``lax.scan`` and passed directly
    to ``strict_update_one_step_from_state``.  This removes one layer of
    per-step trace dictionary indirection while keeping all host branch
    decisions fixed and fingerprint-gated.
    """

    trace_seq = tuple(traces)
    if not trace_seq:
        raise ValueError("at least one accepted trace is required")
    payload: dict[str, Any] = {}
    if bool(include_state_pre):
        payload["state_pre"] = _stack_trace_pytree_field(trace_seq, "state_pre")
    if bool(include_force_state_pre):
        force_state_pre = _stack_optional_trace_pytree_field(trace_seq, "force_state_pre")
        if force_state_pre is not None:
            payload["force_state_pre"] = force_state_pre
    if bool(include_nestor_axes):
        nestor_axes = _stack_trace_nestor_axis_controls(trace_seq)
        if nestor_axes is not None:
            payload["freeb_nestor_axes"] = nestor_axes
    freeb_pres_scale = _stack_optional_trace_pytree_field(trace_seq, "freeb_pres_scale")
    if freeb_pres_scale is not None:
        payload["freeb_pres_scale"] = freeb_pres_scale
    if bool(include_constraints):
        for key in _ACCEPTED_TRACE_OPTIONAL_STEP_PYTREE_CONTROL_KEYS:
            if key in ("force_state_pre", "freeb_pres_scale"):
                continue
            value = _stack_optional_trace_pytree_field(trace_seq, key)
            if value is not None:
                payload[key] = value
    return payload


def direct_coil_accepted_trace_controller_replay_objective_jax(
    params: Any,
    initial_state: Any,
    *,
    static: Any,
    traces: Any,
    signgs: int,
    max_steps: int | None = None,
    sample_nzeta: int | None = None,
    include_analytic: bool = True,
    enforce_edge: bool = False,
    state_weight: Any = 1.0,
    force_weight: Any = 0.0,
    bsqvac_weight: Any = 0.0,
    checkpoint_steps: bool = False,
    accept_mask: Any | None = None,
    done_mask: Any | None = None,
    use_preconditioner_policy_segments: bool = False,
    use_segment_preconditioner_controls: bool = False,
    use_stacked_step_controls: bool = False,
    use_accepted_only_fast_path: bool = True,
    replay_plan: Mapping[str, Any] | None = None,
    include_replay_aux: bool = True,
    state_only_replay: bool = False,
    freeze_vacuum_field: bool = False,
    freeze_freeb_bsqvac: bool = False,
    include_mode_diagnostics: bool = False,
    nestor_solve_mode: str = "dense",
    nestor_operator_solver: str = "gmres",
    nestor_operator_tol: float = 1.0e-11,
    nestor_operator_atol: float = 1.0e-13,
    nestor_operator_maxiter: int | None = None,
    nestor_operator_restart: int | None = None,
    jit_preconditioner_apply: bool = True,
    unroll_accepted_only_segments_below: int = 0,
    coil_geometry: Any | None = None,
) -> dict[str, Any]:
    """Replay fixed production traces through a JAX-visible accept controller.

    This is the bridge between the legacy Python-loop
    :func:`direct_coil_accepted_trace_replay_objective_jax` and a future full
    nonlinear free-boundary controller.  The production traces remain fixed
    data, but the replayed state, per-step accepted masks, and objective
    history are carried through :func:`jax_visible_accepted_nonlinear_controller_jax`.
    If ``use_preconditioner_policy_segments`` is true, the same controls are
    split into consecutive static-preconditioner-policy segments and run
    through :func:`jax_visible_segmented_accepted_nonlinear_controller_jax`.
    The segmented path is behavior-preserving and opt-in while production
    preconditioner dispatch remains partially branch-local.
    ``use_segment_preconditioner_controls`` is a narrower performance
    diagnostic: when the full trace cannot stack preconditioner controls, it
    tries stacking them independently inside each static segment.  It is kept
    opt-in because current tiny production traces show parity but not a speed
    win.
    ``use_stacked_step_controls`` is the next rung: it segments by the full
    static step-policy signature and calls ``strict_update_one_step_from_state``
    directly with stacked state/update/constraint controls.
    ``use_accepted_only_fast_path`` removes the per-step accept/reject proposal
    conditional only for segments whose effective controller masks prove that
    every slot is active and accepted. Rejected, inactive, or post-convergence
    padded slots automatically use the ordinary controller path.
    ``state_only_replay`` is a narrower production-report fast path: it still
    replays the direct-coil vacuum field and VMEC state update, but it omits
    per-step force/vacuum objective history needed only by history-dependent
    scalars such as accepted Bnormal/Bsqvac RMS.
    ``freeze_freeb_bsqvac`` is a diagnostic-only cost split: it reuses the
    accepted trace's ``bsqvac`` array instead of differentiably recomputing the
    direct-coil/NESTOR vacuum response.  This keeps the strict VMEC accepted
    update in the graph while intentionally removing coil sensitivity through
    the external-field replay, so it must not be used as a promoted derivative
    or optimization path.
    ``freeze_vacuum_field`` is an intermediate diagnostic split: it reuses the
    accepted trace's normal/tangential vacuum-field projection arrays but still
    runs the JAX dense NESTOR mode solve and field reconstruction.  This
    separates Biot-Savart/projection graph cost from NESTOR/source assembly
    graph cost, and is also not a promoted derivative path.
    ``include_mode_diagnostics`` controls dense-mode diagnostic outputs such as
    ``phi_flat`` and residual vectors.  Accepted-controller replay only needs
    ``bsqvac`` and optionally boundary RMS diagnostics, so branch-local reports
    default this to false to avoid building unused dense-solve outputs.
    ``nestor_solve_mode`` and the ``nestor_operator_*`` options expose the
    opt-in matrix-free NESTOR/source response inside fixed accepted-branch
    replay.  Dense remains the default; matrix-free replay is a validated
    performance/research seam until size-triggered promotion is justified.

    The helper intentionally keeps every trace accepted.  It does not
    differentiate through the host policy that selected the traces; it validates
    that a production accepted-trace replay can be represented as a static
    JAX-visible accepted-control scan.
    """

    from .discrete_adjoint import strict_update_one_step_from_state, strict_update_one_step_from_trace
    from .state import pack_state

    if replay_plan is None:
        trace_seq = tuple(traces)
        if max_steps is not None:
            trace_seq = trace_seq[: int(max_steps)]
    else:
        trace_seq = tuple(replay_plan["traces"])
    if not trace_seq:
        raise ValueError("at least one accepted trace is required")
    if jax is None:  # pragma: no cover - dependency fallback.
        raise RuntimeError("JAX is required for controller replay.")

    if replay_plan is None:
        replay_plan = direct_coil_accepted_trace_controller_replay_plan(
            trace_seq,
            static=static,
            accept_mask=accept_mask,
            done_mask=done_mask,
            max_steps=None,
            use_preconditioner_policy_segments=bool(use_preconditioner_policy_segments),
            use_segment_preconditioner_controls=bool(use_segment_preconditioner_controls),
            use_stacked_step_controls=bool(use_stacked_step_controls),
            use_accepted_only_fast_path=bool(use_accepted_only_fast_path),
        )

    controls = replay_plan["controls"]
    effective_masks = replay_plan["effective_masks"]
    preconditioner_policy_segments = replay_plan["preconditioner_policy_segments"]
    preconditioner_policy_segment_summary = replay_plan["preconditioner_policy_segment_summary"]
    scalar_controls = replay_plan["scalar_controls"]
    array_controls = replay_plan["array_controls"]
    step_controls = replay_plan["step_controls"]
    step_policy_segments = replay_plan["step_policy_segments"]
    step_policy_segment_summary = replay_plan["step_policy_segment_summary"]
    preconditioner_controls = replay_plan["preconditioner_controls"]
    preconditioner_controls_stacked = bool(replay_plan["preconditioner_controls_stacked"])
    plan_options = replay_plan.get("options", {})

    context_cache: dict[tuple[int, int], dict[str, Any]] = dict(replay_plan.get("boundary_replay_contexts_by_shape", {}))

    def _precomputed_context_for_trace(trace: Mapping[str, Any]) -> dict[str, Any] | None:
        shape = _direct_coil_trace_boundary_shape(trace)
        if shape is None:
            return None
        if shape not in context_cache:
            context_cache[shape] = direct_coil_boundary_replay_context_for_shape(
                static,
                ntheta=shape[0],
                nzeta=shape[1],
            )
        return context_cache[shape]

    def _step_control(control: Mapping[str, Any], key: str) -> Any:
        return control["step_controls"][key] if key in control.get("step_controls", {}) else None

    def _branch_for_trace(
        trace: dict[str, Any],
        state: Any,
        coil_params: Any,
        control: dict[str, Any],
        replay_context: dict[str, Any] | None,
    ):
        reset_to_trace_pre = jnp.asarray(control["reset_to_trace_pre"], dtype=bool)
        state_in = jax.lax.cond(
            reset_to_trace_pre,
            lambda _: trace["state_pre"],
            lambda _: state,
            operand=None,
        )
        has_active_freeb_replay = trace.get("freeb_bsqvac_half") is not None and trace.get("freeb_nestor_trace") is not None
        if has_active_freeb_replay:
            if bool(freeze_freeb_bsqvac):
                freeb_bsqvac_half = jnp.asarray(trace["freeb_bsqvac_half"])
            else:
                with _jax_named_scope("vmec_jax.free_boundary.boundary_geometry"):
                    geometry = free_boundary_boundary_geometry_jax(
                        state_in,
                        static,
                        sample_nzeta=sample_nzeta,
                    )
                context = replay_context
                if context is None or tuple(int(v) for v in geometry["R"].shape) != (
                    int(context["ntheta"]),
                    int(context["nzeta"]),
                ):
                    with _jax_named_scope("vmec_jax.free_boundary.replay_context"):
                        context = direct_coil_boundary_replay_context(static, geometry)
                nestor_axes = _step_control(control, "freeb_nestor_axes")
                if nestor_axes is None:
                    with _jax_named_scope("vmec_jax.free_boundary.direct_coil_bsqvac_replay"):
                        replay = direct_coil_boundary_bsqvac_from_trace_jax(
                            coil_params,
                            geometry,
                            trace,
                            basis=context["basis"],
                            tables=context["tables"],
                            signgs=int(signgs),
                            nvper=int(context["nvper"]),
                            wint=jnp.asarray(context["wint"]),
                            include_analytic=bool(include_analytic),
                            include_diagnostics=not bool(state_only_replay),
                            include_mode_diagnostics=bool(include_mode_diagnostics),
                            freeze_vacuum_field=bool(freeze_vacuum_field),
                            nestor_solve_mode=str(nestor_solve_mode),
                            nestor_operator_solver=str(nestor_operator_solver),
                            nestor_operator_tol=float(nestor_operator_tol),
                            nestor_operator_atol=float(nestor_operator_atol),
                            nestor_operator_maxiter=nestor_operator_maxiter,
                            nestor_operator_restart=nestor_operator_restart,
                            coil_geometry=coil_geometry,
                        )
                else:
                    with _jax_named_scope("vmec_jax.free_boundary.direct_coil_bsqvac_replay"):
                        replay = direct_coil_boundary_bsqvac_jax(
                            coil_params,
                            R=geometry["R"],
                            Z=geometry["Z"],
                            phi=geometry["phi"],
                            Ru=geometry["Ru"],
                            Zu=geometry["Zu"],
                            Rv=geometry["Rv"],
                            Zv=geometry["Zv"],
                            ruu=geometry["ruu"],
                            ruv=geometry["ruv"],
                            rvv=geometry["rvv"],
                            zuu=geometry["zuu"],
                            zuv=geometry["zuv"],
                            zvv=geometry["zvv"],
                            basis=context["basis"],
                            tables=context["tables"],
                            signgs=int(signgs),
                            nvper=int(context["nvper"]),
                            br_add=jnp.asarray(nestor_axes["br_axis"]),
                            bp_add=jnp.asarray(nestor_axes["bp_axis"]),
                            bz_add=jnp.asarray(nestor_axes["bz_axis"]),
                            wint=jnp.asarray(context["wint"]),
                            include_analytic=bool(include_analytic),
                            include_diagnostics=not bool(state_only_replay),
                            include_mode_diagnostics=bool(include_mode_diagnostics),
                            vac_override=(
                                _direct_coil_trace_vacuum_field_override(trace)
                                if bool(freeze_vacuum_field)
                                else None
                            ),
                            coil_geometry=coil_geometry,
                            nestor_solve_mode=str(nestor_solve_mode),
                            nestor_operator_solver=str(nestor_operator_solver),
                            nestor_operator_tol=float(nestor_operator_tol),
                            nestor_operator_atol=float(nestor_operator_atol),
                            nestor_operator_maxiter=nestor_operator_maxiter,
                            nestor_operator_restart=nestor_operator_restart,
                        )
                freeb_bsqvac_half = replay["bsqvac"]
            if bool(state_only_replay):
                bsqvac_objective = jnp.asarray(0.0)
                bsqvac_rms = jnp.asarray(0.0)
                bnormal_rms = jnp.asarray(0.0)
            elif bool(freeze_freeb_bsqvac):
                bsqvac_objective = _weighted_half_norm(freeb_bsqvac_half, bsqvac_weight)
                bsqvac_rms = jnp.sqrt(jnp.mean(jnp.square(jnp.asarray(freeb_bsqvac_half))))
                bnormal_rms = jnp.asarray(0.0)
            else:
                bsqvac_objective = _weighted_half_norm(replay["bsqvac"], bsqvac_weight)
                bsqvac_rms = jnp.sqrt(jnp.mean(jnp.square(jnp.asarray(replay["bsqvac"]))))
                bnormal_rms = jnp.sqrt(jnp.mean(jnp.square(jnp.asarray(replay["vac"]["bnormal"]))))
        else:
            freeb_bsqvac_half = trace.get("freeb_bsqvac_half", None)
            bsqvac_objective = jnp.asarray(0.0)
            bsqvac_rms = jnp.asarray(0.0)
            bnormal_rms = jnp.asarray(0.0)
        with _jax_named_scope("vmec_jax.free_boundary.strict_update_one_step_from_trace"):
            step = strict_update_one_step_from_trace(
                state_in,
                static,
                trace,
                scalar_controls=control["step_scalars"],
                array_controls=control["step_arrays"],
                preconditioner_controls=control["step_preconditioner"] if "step_preconditioner" in control else None,
                freeb_bsqvac_half=freeb_bsqvac_half,
                enforce_edge=bool(enforce_edge),
                jit_preconditioner_apply=bool(jit_preconditioner_apply),
            )
        if bool(state_only_replay):
            return step["step"]["state_post"], {
                "state_reset": reset_to_trace_pre,
            }
        return step["step"]["state_post"], {
            "force": _tree_weighted_half_norm(step["force"], force_weight),
            "bsqvac": bsqvac_objective,
            "bsqvac_rms": bsqvac_rms,
            "bnormal_rms": bnormal_rms,
            "state_reset": reset_to_trace_pre,
        }

    def _branch_from_stacked_controls(
        trace: dict[str, Any],
        state: Any,
        coil_params: Any,
        control: dict[str, Any],
        replay_context: dict[str, Any] | None,
    ):
        if "step_preconditioner" not in control:
            raise ValueError("stacked step replay requires stackable preconditioner controls")
        reset_to_trace_pre = jnp.asarray(control["reset_to_trace_pre"], dtype=bool)
        stacked_state_pre = _step_control(control, "state_pre")
        if stacked_state_pre is None:
            raise ValueError("stacked step replay requires state_pre controls")
        state_in = jax.lax.cond(
            reset_to_trace_pre,
            lambda _: stacked_state_pre,
            lambda _: state,
            operand=None,
        )
        has_active_freeb_replay = trace.get("freeb_bsqvac_half") is not None and trace.get("freeb_nestor_trace") is not None
        if has_active_freeb_replay:
            if bool(freeze_freeb_bsqvac):
                freeb_bsqvac_half = jnp.asarray(trace["freeb_bsqvac_half"])
            else:
                with _jax_named_scope("vmec_jax.free_boundary.boundary_geometry"):
                    geometry = free_boundary_boundary_geometry_jax(
                        state_in,
                        static,
                        sample_nzeta=sample_nzeta,
                    )
                context = replay_context
                if context is None or tuple(int(v) for v in geometry["R"].shape) != (
                    int(context["ntheta"]),
                    int(context["nzeta"]),
                ):
                    with _jax_named_scope("vmec_jax.free_boundary.replay_context"):
                        context = direct_coil_boundary_replay_context(static, geometry)
                nestor_axes = _step_control(control, "freeb_nestor_axes")
                if nestor_axes is None:
                    with _jax_named_scope("vmec_jax.free_boundary.direct_coil_bsqvac_replay"):
                        replay = direct_coil_boundary_bsqvac_from_trace_jax(
                            coil_params,
                            geometry,
                            trace,
                            basis=context["basis"],
                            tables=context["tables"],
                            signgs=int(signgs),
                            nvper=int(context["nvper"]),
                            wint=jnp.asarray(context["wint"]),
                            include_analytic=bool(include_analytic),
                            include_diagnostics=not bool(state_only_replay),
                            include_mode_diagnostics=bool(include_mode_diagnostics),
                            freeze_vacuum_field=bool(freeze_vacuum_field),
                            nestor_solve_mode=str(nestor_solve_mode),
                            nestor_operator_solver=str(nestor_operator_solver),
                            nestor_operator_tol=float(nestor_operator_tol),
                            nestor_operator_atol=float(nestor_operator_atol),
                            nestor_operator_maxiter=nestor_operator_maxiter,
                            nestor_operator_restart=nestor_operator_restart,
                            coil_geometry=coil_geometry,
                        )
                else:
                    with _jax_named_scope("vmec_jax.free_boundary.direct_coil_bsqvac_replay"):
                        replay = direct_coil_boundary_bsqvac_jax(
                            coil_params,
                            R=geometry["R"],
                            Z=geometry["Z"],
                            phi=geometry["phi"],
                            Ru=geometry["Ru"],
                            Zu=geometry["Zu"],
                            Rv=geometry["Rv"],
                            Zv=geometry["Zv"],
                            ruu=geometry["ruu"],
                            ruv=geometry["ruv"],
                            rvv=geometry["rvv"],
                            zuu=geometry["zuu"],
                            zuv=geometry["zuv"],
                            zvv=geometry["zvv"],
                            basis=context["basis"],
                            tables=context["tables"],
                            signgs=int(signgs),
                            nvper=int(context["nvper"]),
                            br_add=jnp.asarray(nestor_axes["br_axis"]),
                            bp_add=jnp.asarray(nestor_axes["bp_axis"]),
                            bz_add=jnp.asarray(nestor_axes["bz_axis"]),
                            wint=jnp.asarray(context["wint"]),
                            include_analytic=bool(include_analytic),
                            include_diagnostics=not bool(state_only_replay),
                            include_mode_diagnostics=bool(include_mode_diagnostics),
                            vac_override=(
                                _direct_coil_trace_vacuum_field_override(trace)
                                if bool(freeze_vacuum_field)
                                else None
                            ),
                            coil_geometry=coil_geometry,
                            nestor_solve_mode=str(nestor_solve_mode),
                            nestor_operator_solver=str(nestor_operator_solver),
                            nestor_operator_tol=float(nestor_operator_tol),
                            nestor_operator_atol=float(nestor_operator_atol),
                            nestor_operator_maxiter=nestor_operator_maxiter,
                            nestor_operator_restart=nestor_operator_restart,
                        )
                freeb_bsqvac_half = replay["bsqvac"]
            if bool(state_only_replay):
                bsqvac_objective = jnp.asarray(0.0)
                bsqvac_rms = jnp.asarray(0.0)
                bnormal_rms = jnp.asarray(0.0)
            elif bool(freeze_freeb_bsqvac):
                bsqvac_objective = _weighted_half_norm(freeb_bsqvac_half, bsqvac_weight)
                bsqvac_rms = jnp.sqrt(jnp.mean(jnp.square(jnp.asarray(freeb_bsqvac_half))))
                bnormal_rms = jnp.asarray(0.0)
            else:
                bsqvac_objective = _weighted_half_norm(replay["bsqvac"], bsqvac_weight)
                bsqvac_rms = jnp.sqrt(jnp.mean(jnp.square(jnp.asarray(replay["bsqvac"]))))
                bnormal_rms = jnp.sqrt(jnp.mean(jnp.square(jnp.asarray(replay["vac"]["bnormal"]))))
        else:
            freeb_bsqvac_half = trace.get("freeb_bsqvac_half", None)
            bsqvac_objective = jnp.asarray(0.0)
            bsqvac_rms = jnp.asarray(0.0)
            bnormal_rms = jnp.asarray(0.0)
        preconditioner_use_precomputed_tridi = trace.get("preconditioner_use_precomputed_tridi")
        preconditioner_use_lax_tridi = trace.get("preconditioner_use_lax_tridi")
        step_step_controls = control["step_controls"]
        with _jax_named_scope("vmec_jax.free_boundary.strict_update_one_step_from_state"):
            step = strict_update_one_step_from_state(
                state_in,
                static,
                force_state_pre=step_step_controls.get("force_state_pre"),
                wout_like=trace["wout_like"],
                trig=trace["trig"],
                apply_lforbal=bool(trace["apply_lforbal"]),
                include_edge_residual=bool(trace["include_edge_residual"]),
                apply_m1_constraints=bool(trace["apply_m1_constraints"]),
                zero_m1=trace["zero_m1"],
                mats=control["step_preconditioner"]["precond_mats"],
                jmax=int(trace["precond_jmax"]),
                lam_prec=control["step_preconditioner"]["lam_prec"],
                w_mode_mn=control["step_preconditioner"]["w_mode_mn"],
                lambda_update_scale=control["step_scalars"]["lambda_update_scale"],
                dt_eff=control["step_scalars"]["dt_eff"],
                b1=control["step_scalars"]["b1"],
                fac=control["step_scalars"]["fac"],
                force_scale=control["step_scalars"]["force_scale"],
                flip_sign=control["step_scalars"]["flip_sign"],
                vRcc_before=control["step_arrays"]["vRcc_before"],
                vRss_before=control["step_arrays"]["vRss_before"],
                vZsc_before=control["step_arrays"]["vZsc_before"],
                vZcs_before=control["step_arrays"]["vZcs_before"],
                vLsc_before=control["step_arrays"]["vLsc_before"],
                vLcs_before=control["step_arrays"]["vLcs_before"],
                vRsc_before=control["step_arrays"].get("vRsc_before"),
                vRcs_before=control["step_arrays"].get("vRcs_before"),
                vZcc_before=control["step_arrays"].get("vZcc_before"),
                vZss_before=control["step_arrays"].get("vZss_before"),
                vLcc_before=control["step_arrays"].get("vLcc_before"),
                vLss_before=control["step_arrays"].get("vLss_before"),
                max_update_rms=control["step_scalars"]["max_update_rms_pre"],
                limit_update_rms=control["step_scalars"]["limit_update_rms"],
                divide_by_scalxc_for_update=control["step_scalars"]["divide_by_scalxc_for_update"],
                preconditioner_use_precomputed_tridi=(
                    None if preconditioner_use_precomputed_tridi is None else bool(preconditioner_use_precomputed_tridi)
                ),
                preconditioner_use_lax_tridi=(
                    None if preconditioner_use_lax_tridi is None else bool(preconditioner_use_lax_tridi)
                ),
                freeb_bsqvac_half=freeb_bsqvac_half,
                freeb_pres_scale=step_step_controls.get("freeb_pres_scale", trace.get("freeb_pres_scale", None)),
                constraint_rcon0=step_step_controls.get("constraint_rcon0", trace.get("constraint_rcon0")),
                constraint_zcon0=step_step_controls.get("constraint_zcon0", trace.get("constraint_zcon0")),
                constraint_tcon0=step_step_controls.get("constraint_tcon0", trace.get("constraint_tcon0")),
                constraint_precond_diag=step_step_controls.get(
                    "constraint_precond_diag",
                    trace.get("constraint_precond_diag"),
                ),
                constraint_tcon=step_step_controls.get("constraint_tcon", trace.get("constraint_tcon")),
                constraint_precond_active=step_step_controls.get(
                    "constraint_precond_active",
                    trace.get("constraint_precond_active"),
                ),
                constraint_tcon_active=step_step_controls.get(
                    "constraint_tcon_active",
                    trace.get("constraint_tcon_active"),
                ),
                enforce_edge=bool(enforce_edge),
                jit_preconditioner_apply=bool(jit_preconditioner_apply),
            )
        if bool(state_only_replay):
            return step["step"]["state_post"], {
                "state_reset": reset_to_trace_pre,
            }
        return step["step"]["state_post"], {
            "force": _tree_weighted_half_norm(step["force"], force_weight),
            "bsqvac": bsqvac_objective,
            "bsqvac_rms": bsqvac_rms,
            "bnormal_rms": bnormal_rms,
            "state_reset": reset_to_trace_pre,
        }

    def _make_step_fn(
        segment_traces: tuple[dict[str, Any], ...],
        *,
        index_offset: int = 0,
        stacked_step_controls: bool = False,
        accepted_only: bool = False,
    ):
        if bool(stacked_step_controls):
            representative_trace = segment_traces[0]
            representative_context = _precomputed_context_for_trace(representative_trace)

            def _step_fn(state, coil_params, control):
                if bool(accepted_only):
                    return _branch_from_stacked_controls(
                        representative_trace,
                        state,
                        coil_params,
                        control,
                        representative_context,
                    )
                do_propose = jnp.asarray(control["accept"], dtype=bool)

                def _propose(_unused):
                    return _branch_from_stacked_controls(
                        representative_trace,
                        state,
                        coil_params,
                        control,
                        representative_context,
                    )

                def _skip(_unused):
                    if bool(state_only_replay):
                        return state, {"state_reset": jnp.asarray(False, dtype=bool)}
                    return (
                        state,
                        {
                            "force": jnp.asarray(0.0),
                            "bsqvac": jnp.asarray(0.0),
                            "bsqvac_rms": jnp.asarray(0.0),
                            "bnormal_rms": jnp.asarray(0.0),
                            "state_reset": jnp.asarray(False, dtype=bool),
                        },
                    )

                return jax.lax.cond(do_propose, _propose, _skip, operand=None)

            return _step_fn

        branches = tuple(
            (
                lambda operand, trace=trace, replay_context=_precomputed_context_for_trace(trace): _branch_for_trace(
                    trace,
                    operand[0],
                    operand[1],
                    operand[2],
                    replay_context,
                )
            )
            for trace in segment_traces
        )

        def _step_fn(state, coil_params, control):
            step_index = jnp.asarray(control["step_index"], dtype=jnp.int32) - jnp.asarray(index_offset, dtype=jnp.int32)
            if bool(accepted_only):
                return jax.lax.switch(step_index, branches, (state, coil_params, control))
            do_propose = jnp.asarray(control["accept"], dtype=bool)

            def _propose(_unused):
                return jax.lax.switch(step_index, branches, (state, coil_params, control))

            def _skip(_unused):
                if bool(state_only_replay):
                    return state, {"state_reset": jnp.asarray(False, dtype=bool)}
                return (
                    state,
                    {
                        "force": jnp.asarray(0.0),
                        "bsqvac": jnp.asarray(0.0),
                        "bsqvac_rms": jnp.asarray(0.0),
                        "bnormal_rms": jnp.asarray(0.0),
                        "state_reset": jnp.asarray(False, dtype=bool),
                    },
                )

            return jax.lax.cond(do_propose, _propose, _skip, operand=None)

        return _step_fn

    def accept_fn(_state, _proposed_state, _params, control, _aux):
        return control["accept"]

    def converged_fn(_accepted_state, _params, control, _aux):
        return control["done"]

    segment_preconditioner_controls_stacked: tuple[bool, ...] = ()
    accepted_only_fast_path_segments: tuple[bool, ...] = ()
    if use_stacked_step_controls:
        if replay_plan.get("segment_source") != "step_policy":
            replay_plan = direct_coil_accepted_trace_controller_replay_plan(
                trace_seq,
                static=static,
                accept_mask=accept_mask,
                done_mask=done_mask,
                max_steps=None,
                use_stacked_step_controls=True,
                use_accepted_only_fast_path=bool(use_accepted_only_fast_path),
            )
            controls = replay_plan["controls"]
            preconditioner_controls = replay_plan["preconditioner_controls"]
            step_policy_segments = replay_plan["step_policy_segments"]
            context_cache = dict(replay_plan.get("boundary_replay_contexts_by_shape", {}))
        control_segments = tuple(replay_plan["control_segments"])
        segment_preconditioner_controls_stacked = tuple(replay_plan["preconditioner_controls_segment_stacked"])
        accepted_only_fast_path_segments = tuple(replay_plan["accepted_only_fast_path_segments"])
        step_fns = tuple(
            _make_step_fn(
                trace_seq[int(segment["start"]) : int(segment["stop"])],
                index_offset=int(segment["start"]),
                stacked_step_controls=True,
                accepted_only=bool(accepted_only_fast_path_segments[index]),
            )
            for index, segment in enumerate(step_policy_segments)
        )
        segmented_runner = (
            jax_visible_segmented_state_only_accepted_nonlinear_controller_jax
            if bool(state_only_replay)
            else jax_visible_segmented_accepted_nonlinear_controller_jax
        )
        run = segmented_runner(
            step_fns,
            accept_fn,
            converged_fn,
            initial_state,
            params,
            control_segments,
            checkpoint_steps=checkpoint_steps,
            accepted_only_segments=accepted_only_fast_path_segments,
            unroll_accepted_only_segments_below=int(unroll_accepted_only_segments_below),
        )
    elif use_preconditioner_policy_segments:
        if replay_plan.get("segment_source") != "preconditioner_policy" or bool(
            plan_options.get("use_segment_preconditioner_controls", False)
        ) != bool(use_segment_preconditioner_controls):
            replay_plan = direct_coil_accepted_trace_controller_replay_plan(
                trace_seq,
                static=static,
                accept_mask=accept_mask,
                done_mask=done_mask,
                max_steps=None,
                use_preconditioner_policy_segments=True,
                use_segment_preconditioner_controls=bool(use_segment_preconditioner_controls),
                use_accepted_only_fast_path=bool(use_accepted_only_fast_path),
            )
            controls = replay_plan["controls"]
            preconditioner_controls = replay_plan["preconditioner_controls"]
            preconditioner_policy_segments = replay_plan["preconditioner_policy_segments"]
            context_cache = dict(replay_plan.get("boundary_replay_contexts_by_shape", {}))
        control_segments = tuple(replay_plan["control_segments"])
        segment_preconditioner_controls_stacked = tuple(replay_plan["preconditioner_controls_segment_stacked"])
        accepted_only_fast_path_segments = tuple(replay_plan["accepted_only_fast_path_segments"])
        step_fns = tuple(
            _make_step_fn(
                trace_seq[int(segment["start"]) : int(segment["stop"])],
                index_offset=int(segment["start"]),
                accepted_only=bool(accepted_only_fast_path_segments[index]),
            )
            for index, segment in enumerate(preconditioner_policy_segments)
        )
        segmented_runner = (
            jax_visible_segmented_state_only_accepted_nonlinear_controller_jax
            if bool(state_only_replay)
            else jax_visible_segmented_accepted_nonlinear_controller_jax
        )
        run = segmented_runner(
            step_fns,
            accept_fn,
            converged_fn,
            initial_state,
            params,
            control_segments,
            checkpoint_steps=checkpoint_steps,
            accepted_only_segments=accepted_only_fast_path_segments,
            unroll_accepted_only_segments_below=int(unroll_accepted_only_segments_below),
        )
    else:
        accepted_only_fast_path_segments = (
            bool(use_accepted_only_fast_path)
            and _accepted_trace_segment_is_unconditionally_accepted(effective_masks, start=0, stop=len(trace_seq)),
        )
        step_fn = _make_step_fn(trace_seq, accepted_only=accepted_only_fast_path_segments[0])
        if accepted_only_fast_path_segments[0]:
            use_unrolled = int(unroll_accepted_only_segments_below) > 0 and len(trace_seq) <= int(
                unroll_accepted_only_segments_below
            )
            if bool(state_only_replay):
                accepted_only_runner = (
                    jax_visible_unrolled_state_only_accepted_only_nonlinear_controller_jax
                    if use_unrolled
                    else jax_visible_state_only_accepted_only_nonlinear_controller_jax
                )
            else:
                accepted_only_runner = (
                    jax_visible_unrolled_accepted_only_nonlinear_controller_jax
                    if use_unrolled
                    else jax_visible_accepted_only_nonlinear_controller_jax
                )
            run = accepted_only_runner(
                step_fn,
                converged_fn,
                initial_state,
                params,
                controls,
                checkpoint_steps=checkpoint_steps,
            )
        else:
            accepted_runner = (
                jax_visible_state_only_accepted_nonlinear_controller_jax
                if bool(state_only_replay)
                else jax_visible_accepted_nonlinear_controller_jax
            )
            run = accepted_runner(
                step_fn,
                accept_fn,
                converged_fn,
                initial_state,
                params,
                controls,
                checkpoint_steps=checkpoint_steps,
            )
    state_objective = (
        jnp.asarray(0.0)
        if _static_weight_is_zero(state_weight)
        else _weighted_half_norm(pack_state(run["state"]), state_weight)
    )
    if bool(state_only_replay):
        objective_components = {
            "state": state_objective,
            "force": jnp.asarray(0.0),
            "bsqvac": jnp.asarray(0.0),
        }
    else:
        accepted = jnp.asarray(run["history"]["accepted"], dtype=jnp.asarray(state_objective).dtype)
        objective_components = {
            "state": state_objective,
            "force": jnp.sum(accepted * jnp.asarray(run["history"]["force"])),
            "bsqvac": jnp.sum(accepted * jnp.asarray(run["history"]["bsqvac"])),
        }
    objective = sum(objective_components.values())
    result = {
        "objective": objective,
        "objective_components": objective_components,
        "state": run["state"],
        "history": run["history"],
        "used_state_only_replay": bool(state_only_replay),
    }
    if not bool(include_replay_aux):
        return {
            **result,
            "controls": {
                "has_active_freeb_replay": controls["has_active_freeb_replay"],
            },
        }
    return {
        **result,
        "controls": controls,
        "scalar_controls": scalar_controls,
        "array_controls": array_controls,
        "step_controls": step_controls,
        "preconditioner_controls": preconditioner_controls,
        "preconditioner_controls_stacked": bool(preconditioner_controls_stacked),
        "preconditioner_policy_segments": preconditioner_policy_segments,
        "preconditioner_policy_n_segments": len(preconditioner_policy_segments),
        "preconditioner_policy_segment_summary": preconditioner_policy_segment_summary,
        "step_policy_segments": step_policy_segments,
        "step_policy_n_segments": len(step_policy_segments),
        "step_policy_segment_summary": step_policy_segment_summary,
        "preconditioner_controls_segment_stacked": segment_preconditioner_controls_stacked,
        "used_preconditioner_policy_segments": bool(use_preconditioner_policy_segments),
        "used_stacked_step_controls": bool(use_stacked_step_controls),
        "used_accepted_only_fast_path": bool(any(accepted_only_fast_path_segments)),
        "accepted_only_fast_path_segments": accepted_only_fast_path_segments,
        "state_reset_flags": tuple(bool(flag) for flag in np.asarray(controls["reset_to_trace_pre"], dtype=bool)),
    }


def _trace_scalar(trace: dict[str, Any], key: str, *, default: float = np.nan) -> float:
    value = trace.get(key, default)
    if value is None:
        return float(default)
    arr = np.asarray(value)
    if arr.size == 0:
        return float(default)
    return float(arr.reshape(-1)[0])


def _trace_bool(trace: dict[str, Any], key: str) -> int:
    value = trace.get(key, False)
    if value is None:
        return 0
    arr = np.asarray(value)
    if arr.size == 0:
        return 0
    return int(bool(arr.reshape(-1)[0]))


def _trace_pack_size(value: Any) -> int:
    if value is None:
        return 0
    from .state import pack_state

    try:
        return int(np.asarray(pack_state(value)).size)
    except Exception:
        return int(np.asarray(value).size)


def _trace_array_size(value: Any) -> int:
    if value is None:
        return 0
    return int(np.asarray(value).size)


def _trace_pytree_shape_signature(value: Any) -> tuple[tuple[int, ...], ...]:
    if value is None:
        return ()
    try:
        leaves = tree_util.tree_leaves(value)
    except Exception:
        leaves = [value]
    return tuple(tuple(np.asarray(leaf).shape) for leaf in leaves)


def direct_coil_accepted_trace_fingerprint(
    traces: Any,
    *,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Return a branch-control fingerprint for accepted free-boundary traces.

    The fixed-trace direct-coil adjoint differentiates a frozen local model:
    accepted controller choices, time-step scalars, limiter policy, and NESTOR
    trace structure are fixed while coil fields are resampled.  This
    fingerprint captures those *discrete/control* choices so a complete-solve
    finite-difference check can reject perturbations that moved onto a
    different adaptive branch before comparing derivatives.

    Differentiable values that should vary with coil parameters, such as the
    actual ``freeb_bsqvac_half`` entries, are intentionally not included except
    for presence/size metadata.
    """

    trace_seq = list(traces)
    if max_steps is not None:
        trace_seq = trace_seq[: int(max_steps)]

    scalar_keys = (
        "dt_eff",
        "b1",
        "fac",
        "force_scale",
        "max_update_rms_pre",
        "limit_update_rms",
    )
    bool_keys = (
        "flip_sign",
        "divide_by_scalxc_for_update",
        "preconditioner_use_precomputed_tridi",
        "preconditioner_use_lax_tridi",
    )
    scalars = {
        key: np.asarray([_trace_scalar(trace, key) for trace in trace_seq], dtype=float)
        for key in scalar_keys
    }
    flags = {
        key: np.asarray([_trace_bool(trace, key) for trace in trace_seq], dtype=int)
        for key in bool_keys
    }
    freeb_sizes = np.asarray(
        [_trace_array_size(trace.get("freeb_bsqvac_half")) for trace in trace_seq],
        dtype=int,
    )
    nestor_sizes = np.asarray(
        [
            len(trace.get("freeb_nestor_trace", {}) or {})
            if isinstance(trace.get("freeb_nestor_trace", {}), dict)
            else 0
            for trace in trace_seq
        ],
        dtype=int,
    )
    state_pre_sizes = np.asarray(
        [_trace_pack_size(trace.get("state_pre")) for trace in trace_seq],
        dtype=int,
    )
    state_post_sizes = np.asarray(
        [_trace_pack_size(trace.get("state_post")) for trace in trace_seq],
        dtype=int,
    )
    precond_jmax = np.asarray([int(trace.get("precond_jmax", -1)) for trace in trace_seq], dtype=int)
    precond_mats_shapes = tuple(_trace_pytree_shape_signature(trace.get("precond_mats")) for trace in trace_seq)
    lam_prec_shapes = tuple(tuple(np.asarray(trace.get("lam_prec", [])).shape) for trace in trace_seq)
    w_mode_shapes = tuple(tuple(np.asarray(trace.get("w_mode_mn", [])).shape) for trace in trace_seq)
    reset_flags = []
    for prev_trace, trace in zip(trace_seq[:-1], trace_seq[1:], strict=False):
        try:
            prev_packed = np.asarray(pack_state(prev_trace.get("state_post")), dtype=float)
            next_packed = np.asarray(pack_state(trace.get("state_pre")), dtype=float)
            reset_flags.append(
                int(
                    prev_packed.shape != next_packed.shape
                    or (not np.allclose(prev_packed, next_packed, rtol=1.0e-13, atol=1.0e-13))
                )
            )
        except Exception:
            reset_flags.append(0)
    return {
        "n_steps": int(len(trace_seq)),
        "n_freeb_steps": int(np.count_nonzero(freeb_sizes)),
        "scalars": scalars,
        "flags": flags,
        "freeb_sizes": freeb_sizes,
        "nestor_trace_key_counts": nestor_sizes,
        "state_pre_sizes": state_pre_sizes,
        "state_post_sizes": state_post_sizes,
        "precond_jmax": precond_jmax,
        "precond_mats_shapes": precond_mats_shapes,
        "lam_prec_shapes": lam_prec_shapes,
        "w_mode_mn_shapes": w_mode_shapes,
        "state_reset_flags": np.asarray(reset_flags, dtype=int),
    }


def direct_coil_accepted_trace_fingerprint_delta(
    reference: Any,
    candidate: Any,
    *,
    rtol: float = 1.0e-10,
    atol: float = 1.0e-12,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Compare two accepted-trace fingerprints.

    Returns a small diagnostic dictionary with ``compatible=True`` only when
    the accepted-step structure and fixed controller scalars agree within the
    requested tolerances.  This is a guard for fixed-trace AD-vs-FD promotion;
    incompatibility means the perturbation exercised a different host-control
    branch and should not be used to validate the frozen-trace derivative.
    """

    ref = direct_coil_accepted_trace_fingerprint(reference, max_steps=max_steps)
    cand = direct_coil_accepted_trace_fingerprint(candidate, max_steps=max_steps)
    changed: list[str] = []
    max_abs = 0.0
    max_rel = 0.0

    for key in ("n_steps", "n_freeb_steps"):
        if int(ref[key]) != int(cand[key]):
            changed.append(key)

    for group in ("flags",):
        for key, ref_values in ref[group].items():
            cand_values = cand[group].get(key, np.asarray([], dtype=ref_values.dtype))
            if ref_values.shape != cand_values.shape or not np.array_equal(ref_values, cand_values):
                changed.append(f"{group}.{key}")

    for key in (
        "freeb_sizes",
        "nestor_trace_key_counts",
        "state_pre_sizes",
        "state_post_sizes",
        "precond_jmax",
        "state_reset_flags",
    ):
        ref_values = np.asarray(ref[key])
        cand_values = np.asarray(cand[key])
        if ref_values.shape != cand_values.shape or not np.array_equal(ref_values, cand_values):
            changed.append(key)

    for key in ("precond_mats_shapes", "lam_prec_shapes", "w_mode_mn_shapes"):
        if ref[key] != cand[key]:
            changed.append(key)

    for key, ref_values in ref["scalars"].items():
        cand_values = cand["scalars"].get(key, np.asarray([], dtype=float))
        if ref_values.shape != cand_values.shape:
            changed.append(f"scalars.{key}")
            continue
        abs_delta = np.abs(cand_values - ref_values)
        finite = np.isfinite(abs_delta)
        if np.any(finite):
            max_abs = max(max_abs, float(np.max(abs_delta[finite])))
            denom = np.maximum(np.abs(ref_values[finite]), float(atol))
            max_rel = max(max_rel, float(np.max(abs_delta[finite] / denom)))
        if not np.allclose(cand_values, ref_values, rtol=float(rtol), atol=float(atol), equal_nan=True):
            changed.append(f"scalars.{key}")

    return {
        "compatible": len(changed) == 0,
        "changed_fields": tuple(changed),
        "max_abs_scalar_delta": float(max_abs),
        "max_rel_scalar_delta": float(max_rel),
        "reference": ref,
        "candidate": cand,
    }


def _json_safe_fingerprint_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_safe_fingerprint_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe_fingerprint_value(value.item())
    if isinstance(value, dict):
        return {str(key): _json_safe_fingerprint_value(val) for key, val in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe_fingerprint_value(item) for item in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return _json_safe_fingerprint_value(value.tolist())
        except Exception:
            pass
    return value


def direct_coil_accepted_trace_fingerprint_delta_summary(
    reference: Any,
    candidate: Any,
    *,
    rtol: float = 1.0e-10,
    atol: float = 1.0e-12,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Return a strict-JSON-safe accepted-trace fingerprint delta summary.

    The raw :func:`direct_coil_accepted_trace_fingerprint_delta` output keeps
    NumPy arrays for in-process diagnostics.  Comparison scripts and reviewer
    artifacts need a payload that can be written with
    ``json.dumps(..., allow_nan=False)``; this helper converts arrays, tuples,
    NumPy scalars, and non-finite values into JSON-safe Python objects.
    """

    delta = direct_coil_accepted_trace_fingerprint_delta(
        reference,
        candidate,
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
    )
    return _json_safe_fingerprint_value(delta)


def direct_coil_complete_solve_trace(
    input_path: Any,
    params: Any,
    *,
    init_kwargs: dict[str, Any] | None = None,
    solve_kwargs: dict[str, Any] | None = None,
    require_active_trace: bool = True,
) -> dict[str, Any]:
    """Run a direct-coil free-boundary solve and return accepted traces.

    This is a validation helper for phase-2 same-branch adjoint promotion.  It
    runs the same direct-coil initialization plus accepted residual iteration
    used by the complete-solve finite-difference gates and returns the
    initialization result, final solve result, and recorded adjoint traces.

    The helper intentionally does not decide whether perturbations are on the
    same adaptive branch.  Use
    :func:`direct_coil_same_branch_complete_solve_fd_report` or
    :func:`direct_coil_accepted_trace_fingerprint_delta` for that gate.
    """

    from .driver import run_free_boundary
    from .solve import solve_fixed_boundary_residual_iter

    init_options: dict[str, Any] = {
        "use_initial_guess": True,
        "verbose": False,
        "external_field_provider_kind": "direct_coils",
        "external_field_provider_params": params,
    }
    if init_kwargs:
        init_options.update(init_kwargs)
    init = run_free_boundary(input_path, **init_options)

    solve_options: dict[str, Any] = {
        "max_iter": 2,
        "ftol": 1.0e-8,
        "vmec2000_control": True,
        "auto_flip_force": False,
        "use_direct_fallback": True,
        "verbose": False,
        "verbose_vmec2000_table": False,
        "jit_forces": False,
        "use_scan": False,
        "host_update_assembly": False,
        "adjoint_trace": True,
        "adjoint_trace_mode": "full",
        "external_field_provider_kind": "direct_coils",
        "external_field_provider_params": params,
        "free_boundary_activate_fsq": 1.0e99,
    }
    if solve_kwargs:
        solve_options.update(solve_kwargs)
    solve_options["external_field_provider_params"] = params
    result = solve_fixed_boundary_residual_iter(
        init.state,
        init.static,
        indata=init.indata,
        signgs=init.signgs,
        **solve_options,
    )
    traces = list(result.diagnostics.get("adjoint_step_trace", []))
    if not traces:
        raise RuntimeError("direct-coil solve did not record adjoint_step_trace")
    active_trace = any(trace.get("freeb_bsqvac_half") is not None for trace in traces)
    if bool(require_active_trace) and not active_trace:
        raise RuntimeError("direct-coil solve did not record an active free-boundary trace")
    return {
        "init": init,
        "result": result,
        "traces": traces,
        "params": params,
        "active_trace": bool(active_trace),
    }


def _complete_solve_objective_values(value: Any) -> dict[str, float]:
    """Normalize one scalar or a mapping of scalar diagnostics."""

    if isinstance(value, Mapping):
        if not value:
            raise ValueError("objective_fn returned an empty mapping")
        values: dict[str, float] = {}
        for key, item in value.items():
            arr = np.asarray(item, dtype=float)
            if arr.size != 1:
                raise ValueError(f"objective_fn mapping entry {key!r} must be scalar")
            values[str(key)] = float(arr.reshape(-1)[0])
        return values

    arr = np.asarray(value, dtype=float)
    if arr.size != 1:
        raise ValueError("objective_fn must return a scalar or a mapping of scalars")
    return {"objective": float(arr.reshape(-1)[0])}


def direct_coil_same_branch_complete_solve_fd_report(
    input_path: Any,
    base_params: Any,
    *,
    params_for: Any,
    objective_fn: Any,
    eps: float = 1.0e-4,
    init_kwargs: dict[str, Any] | None = None,
    solve_kwargs: dict[str, Any] | None = None,
    fingerprint_rtol: float = 1.0e-6,
    fingerprint_atol: float = 1.0e-9,
    require_active_trace: bool = True,
) -> dict[str, Any]:
    """Return same-branch complete-solve finite-difference diagnostics.

    ``params_for(scale)`` must return the coil parameters for ``base + scale *
    direction``.  ``objective_fn(payload)`` receives each payload returned by
    :func:`direct_coil_complete_solve_trace` and returns either one scalar or a
    mapping of scalar diagnostics.  The result contains raw base/plus/minus
    payloads, branch fingerprint deltas, scalar values, and central
    finite-difference slopes.  For backward compatibility, ``values`` reports
    the primary scalar.  ``objective_values`` reports every scalar returned by
    ``objective_fn``.

    This helper is deliberately a validation seam rather than a production
    adjoint: it rejects branch changes using accepted-trace and residual
    controller fingerprints and leaves the differentiated frozen-branch replay
    to the caller.
    """

    from .discrete_adjoint import residual_branch_fingerprint

    eps_f = float(eps)
    if eps_f == 0.0:
        raise ValueError("eps must be nonzero")
    base = direct_coil_complete_solve_trace(
        input_path,
        base_params,
        init_kwargs=init_kwargs,
        solve_kwargs=solve_kwargs,
        require_active_trace=require_active_trace,
    )
    plus = direct_coil_complete_solve_trace(
        input_path,
        params_for(eps_f),
        init_kwargs=init_kwargs,
        solve_kwargs=solve_kwargs,
        require_active_trace=require_active_trace,
    )
    minus = direct_coil_complete_solve_trace(
        input_path,
        params_for(-eps_f),
        init_kwargs=init_kwargs,
        solve_kwargs=solve_kwargs,
        require_active_trace=require_active_trace,
    )
    plus_branch = direct_coil_accepted_trace_fingerprint_delta(
        base["traces"],
        plus["traces"],
        rtol=float(fingerprint_rtol),
        atol=float(fingerprint_atol),
    )
    minus_branch = direct_coil_accepted_trace_fingerprint_delta(
        base["traces"],
        minus["traces"],
        rtol=float(fingerprint_rtol),
        atol=float(fingerprint_atol),
    )
    base_fingerprint = direct_coil_accepted_trace_fingerprint(base["traces"])
    plus_fingerprint = direct_coil_accepted_trace_fingerprint(plus["traces"])
    minus_fingerprint = direct_coil_accepted_trace_fingerprint(minus["traces"])
    base_residual_fingerprint = residual_branch_fingerprint(base["result"])
    plus_residual_fingerprint = residual_branch_fingerprint(plus["result"])
    minus_residual_fingerprint = residual_branch_fingerprint(minus["result"])
    same_residual_branch = bool(
        base_residual_fingerprint == plus_residual_fingerprint
        and base_residual_fingerprint == minus_residual_fingerprint
    )
    trace_replay_diagnostics = {
        "base": free_boundary_adjoint_trace_replay_diagnostics(base["traces"]),
        "plus": free_boundary_adjoint_trace_replay_diagnostics(plus["traces"]),
        "minus": free_boundary_adjoint_trace_replay_diagnostics(minus["traces"]),
    }
    base_values = _complete_solve_objective_values(objective_fn(base))
    plus_values = _complete_solve_objective_values(objective_fn(plus))
    minus_values = _complete_solve_objective_values(objective_fn(minus))
    if base_values.keys() != plus_values.keys() or base_values.keys() != minus_values.keys():
        raise ValueError("objective_fn returned different scalar keys for base/plus/minus solves")
    primary_key = "objective" if "objective" in base_values else next(iter(base_values))
    objective_values = {
        key: {
            "base": float(base_values[key]),
            "plus": float(plus_values[key]),
            "minus": float(minus_values[key]),
            "central_fd_directional": float((plus_values[key] - minus_values[key]) / (2.0 * eps_f)),
        }
        for key in base_values
    }
    return {
        "base": base,
        "plus": plus,
        "minus": minus,
        "branch_compatibility": {
            "same_branch": bool(plus_branch["compatible"] and minus_branch["compatible"] and same_residual_branch),
            "same_accepted_trace_branch": bool(plus_branch["compatible"] and minus_branch["compatible"]),
            "same_residual_branch": same_residual_branch,
            "plus": plus_branch,
            "minus": minus_branch,
            "base_fingerprint": base_fingerprint,
            "plus_fingerprint": plus_fingerprint,
            "minus_fingerprint": minus_fingerprint,
            "base_residual_fingerprint": base_residual_fingerprint,
            "plus_residual_fingerprint": plus_residual_fingerprint,
            "minus_residual_fingerprint": minus_residual_fingerprint,
        },
        "trace_replay_diagnostics": trace_replay_diagnostics,
        "primary_objective": primary_key,
        "values": objective_values[primary_key],
        "objective_values": objective_values,
    }


def direct_coil_same_branch_replay_gate_report(
    complete_report: Mapping[str, Any],
    *,
    require_active_free_boundary: bool = True,
    require_scalar_controls_stackable: bool = True,
    require_array_controls_stackable: bool = True,
    json_safe: bool = False,
) -> dict[str, Any]:
    """Return the branch gate for promoting a fixed-trace replay derivative.

    The gate consumes the output of
    :func:`direct_coil_same_branch_complete_solve_fd_report` and checks only
    discrete/control-flow compatibility: same accepted branch, matching replay
    fingerprints, active direct-coil free-boundary replay when requested, and
    stackable controller payloads.  Passing this gate means a branch-local
    custom VJP can be compared against complete-solve finite differences; it
    still does not differentiate the adaptive host controller.
    """

    errors: list[str] = []
    branch = complete_report.get("branch_compatibility", {})
    same_branch = bool(branch.get("same_branch", False))
    if not same_branch:
        errors.append("branch_compatibility.same_branch is false")

    trace_diags = complete_report.get("trace_replay_diagnostics", {})
    expected_labels = ("base", "plus", "minus")
    if set(trace_diags) != set(expected_labels):
        errors.append("trace_replay_diagnostics must contain base, plus, and minus")

    for label in expected_labels:
        diag = trace_diags.get(label)
        fingerprint = branch.get(f"{label}_fingerprint")
        if not isinstance(diag, Mapping):
            errors.append(f"{label}: missing replay diagnostics")
            continue
        if not isinstance(fingerprint, Mapping):
            errors.append(f"{label}: missing branch fingerprint")
            continue
        if bool(diag.get("differentiates_adaptive_controller", True)):
            errors.append(f"{label}: diagnostics unexpectedly claim adaptive-controller differentiation")
        diag_fingerprint = diag.get("branch_fingerprint", {})
        if int(diag.get("n_steps", -1)) != int(fingerprint.get("n_steps", -2)):
            errors.append(f"{label}: n_steps mismatch")
        if int(diag_fingerprint.get("n_steps", -1)) != int(fingerprint.get("n_steps", -2)):
            errors.append(f"{label}: fingerprint n_steps mismatch")
        if int(diag_fingerprint.get("n_freeb_steps", -1)) != int(fingerprint.get("n_freeb_steps", -2)):
            errors.append(f"{label}: fingerprint n_freeb_steps mismatch")
        try:
            if not np.array_equal(
                np.asarray(diag_fingerprint.get("freeb_sizes")),
                np.asarray(fingerprint.get("freeb_sizes")),
            ):
                errors.append(f"{label}: freeb_sizes mismatch")
        except Exception:
            errors.append(f"{label}: freeb_sizes comparison failed")

        masks = diag.get("masks", {})
        n_steps = int(fingerprint.get("n_steps", -1))
        for mask_key in ("active", "accepted", "rejected", "done", "has_active_freeb_replay"):
            mask = np.asarray(masks.get(mask_key, []), dtype=bool)
            if mask.shape != (n_steps,):
                errors.append(f"{label}: mask {mask_key!r} has shape {mask.shape}, expected {(n_steps,)}")
        if require_active_free_boundary:
            if int(fingerprint.get("n_freeb_steps", 0)) <= 0:
                errors.append(f"{label}: no active free-boundary replay steps in fingerprint")
            active_freeb = np.logical_and(
                np.asarray(masks.get("accepted", []), dtype=bool),
                np.asarray(masks.get("has_active_freeb_replay", []), dtype=bool),
            )
            if not bool(np.any(active_freeb)):
                errors.append(f"{label}: no accepted active free-boundary replay slots")

        replay_diag = diag.get("replay_diagnostics", {})
        if require_scalar_controls_stackable and not bool(replay_diag.get("scalar_controls_stackable", False)):
            errors.append(f"{label}: scalar controls are not stackable")
        if require_array_controls_stackable and not bool(replay_diag.get("array_controls_stackable", False)):
            errors.append(f"{label}: array controls are not stackable")
        if int(replay_diag.get("preconditioner_policy_n_segments", 0)) < 1:
            errors.append(f"{label}: no preconditioner policy segments")

    gate = {
        "contract": "same-branch accepted-trace replay gate",
        "passed": len(errors) == 0,
        "differentiates_adaptive_controller": False,
        "same_branch": same_branch,
        "errors": tuple(errors),
    }
    if json_safe:
        return _json_safe_fingerprint_value(gate)
    return gate


def direct_coil_same_branch_controller_scalar_custom_vjp_report(
    complete_report: dict[str, Any],
    base_params: Any,
    direction: Any,
    *,
    replay_scalar_fn: Any,
    scalar_key: str | None = None,
    eps: float = 1.0e-4,
    replay_kwargs: dict[str, Any] | None = None,
    rtol: float = 5.0e-3,
    atol: float = 1.0e-8,
    base_value_atol: float = 2.0e-3,
    compute_frozen_fd: bool = True,
) -> dict[str, Any]:
    """Compare a branch-local scalar custom VJP with complete-solve FD.

    ``complete_report`` must be returned by
    :func:`direct_coil_same_branch_complete_solve_fd_report`.  ``scalar_key``
    selects one scalar from its ``objective_values`` block; by default the
    report's primary scalar is used.  ``replay_scalar_fn(replay, base_payload)``
    receives the JAX-visible accepted-controller replay and the base complete
    solve payload, and must return the same scalar in replay coordinates.

    This is still a same-branch validation helper.  It proves that the frozen
    accepted-controller custom VJP agrees with complete-solve central
    differences when the accepted-trace fingerprint is unchanged.  It does not
    differentiate through an arbitrary adaptive host-controller branch change.
    Set ``compute_frozen_fd=False`` when the caller only needs the exact
    branch-local custom-VJP slope versus the complete-solve FD slope and wants
    to avoid two additional frozen replay evaluations.
    """

    if jax is None:  # pragma: no cover - JAX is required for custom VJP.
        raise RuntimeError("JAX is required for same-branch custom-VJP reports.")

    key = str(scalar_key or complete_report.get("primary_objective") or "objective")
    objective_values = complete_report.get("objective_values", {})
    if key not in objective_values:
        raise KeyError(f"scalar_key {key!r} not present in complete_report['objective_values']")

    replay_gate = direct_coil_same_branch_replay_gate_report(complete_report)
    branch = complete_report.get("branch_compatibility", {})
    same_branch = bool(branch.get("same_branch", False))
    base = complete_report["base"]
    traces = tuple(base["traces"])
    if not traces:
        raise ValueError("complete_report base payload contains no accepted traces")
    replay_options: dict[str, Any] = {
        "static": base["init"].static,
        "traces": traces,
        "signgs": int(base["init"].signgs),
        "state_weight": 0.0,
        "bsqvac_weight": 0.0,
        "force_weight": 0.0,
        "enforce_edge": False,
        "use_preconditioner_policy_segments": True,
    }
    if replay_kwargs:
        replay_options.update(replay_kwargs)

    def _controller_scalar(coil_params):
        return direct_coil_accepted_trace_controller_custom_vjp_scalar_jax(
            coil_params,
            traces[0]["state_pre"],
            scalar_fn=lambda replay: replay_scalar_fn(replay, base),
            **replay_options,
        )

    check = pytree_directional_derivative_check_jax(
        _controller_scalar,
        base_params,
        direction,
        eps=float(eps),
        compute_fd=bool(compute_frozen_fd),
    )
    value = float(np.asarray(check["value"], dtype=float))
    exact = float(np.asarray(check["exact_directional"], dtype=float))
    frozen_fd = float(np.asarray(check["fd_directional"], dtype=float))
    complete_values = objective_values[key]
    complete_base = float(complete_values["base"])
    complete_fd = float(complete_values["central_fd_directional"])
    abs_error = abs(exact - complete_fd)
    rel_error = abs_error / max(1.0, abs(complete_fd))
    base_abs_delta = abs(value - complete_base)
    passed = bool(
        replay_gate["passed"]
        and np.isfinite(exact)
        and np.isfinite(complete_fd)
        and abs_error <= float(atol) + float(rtol) * abs(complete_fd)
        and base_abs_delta <= float(base_value_atol)
    )
    return {
        "scalar_key": key,
        "passed": passed,
        "same_branch": same_branch,
        "replay_gate": replay_gate,
        "value": check["value"],
        "grad": check["grad"],
        "exact_directional": check["exact_directional"],
        "frozen_trace_fd_directional": check["fd_directional"],
        "complete_fd_directional": complete_fd,
        "abs_error": abs_error,
        "rel_error": rel_error,
        "base_value": value,
        "complete_base_value": complete_base,
        "base_abs_delta": base_abs_delta,
        "complete_values": complete_values,
    }


def _pytree_batched_directional_vdot_jax(jacobian_tree: Any, direction: Any, n_outputs: int) -> Any:
    """Contract a vector-output pytree Jacobian with one pytree direction."""

    leaves = tree_util.tree_leaves(
        tree_util.tree_map(
            lambda jac_leaf, direction_leaf: jnp.sum(
                jnp.reshape(jnp.asarray(jac_leaf), (int(n_outputs), -1))
                * jnp.reshape(jnp.asarray(direction_leaf), (1, -1)),
                axis=1,
            ),
            jacobian_tree,
            direction,
        )
    )
    if not leaves:
        return jnp.zeros((int(n_outputs),), dtype=float)
    total = leaves[0]
    for leaf in leaves[1:]:
        total = total + leaf
    return total


def _pytree_pullback_basis_jax(pullback: Any, basis: Any) -> Any:
    """Apply a VJP pullback to all basis cotangents with one batched transform.

    ``jax.vjp`` returns a pullback that accepts one output cotangent.  Several
    free-boundary validation paths need one gradient per scalar output.  Calling
    the pullback in a Python loop is correct but introduces avoidable host
    overhead and can inflate dispatch timing.  ``vmap`` batches the cotangents
    while preserving a leading scalar-output axis on every gradient leaf.

    Some unusual pytrees/backend combinations may not be vmappable; in that
    case fall back to the previous loop so this remains a performance
    improvement, not a semantic requirement.
    """

    try:
        return jax.vmap(lambda cotangent: pullback(cotangent)[0])(basis)
    except Exception:  # pragma: no cover - defensive fallback for exotic pytrees.
        basis_gradients = tuple(pullback(basis[index])[0] for index in range(int(basis.shape[0])))
        return tree_util.tree_map(
            lambda *parts: jnp.stack([jnp.asarray(part) for part in parts], axis=0),
            *basis_gradients,
        )


def _pytree_unstack_leading_axis_jax(pytree: Any, n_outputs: int) -> tuple[Any, ...]:
    """Return one pytree per leading output axis from a batched pytree."""

    return tuple(
        tree_util.tree_map(lambda leaf, index=index: jnp.asarray(leaf)[index], pytree)
        for index in range(int(n_outputs))
    )


def direct_coil_same_branch_controller_scalars_custom_vjp_report(
    complete_report: dict[str, Any],
    base_params: Any,
    direction: Any,
    *,
    replay_scalar_fns: Mapping[str, Any],
    eps: float = 1.0e-4,
    replay_kwargs: dict[str, Any] | None = None,
    rtol: float | Mapping[str, float] = 5.0e-3,
    atol: float | Mapping[str, float] = 1.0e-8,
    base_value_atol: float | Mapping[str, float] = 2.0e-3,
    compute_frozen_fd: bool = False,
) -> dict[str, Any]:
    """Batch same-branch custom-VJP reports for several replay scalars.

    This helper preserves the same branch-local contract as
    :func:`direct_coil_same_branch_controller_scalar_custom_vjp_report`, but
    groups multiple scalar pullbacks through one vector-valued custom-VJP seam.
    It is intended for expensive promotion tests that compare several physical
    outputs against the same complete-solve finite-difference report.
    """

    if jax is None:  # pragma: no cover - JAX is required for custom VJP.
        raise RuntimeError("JAX is required for same-branch custom-VJP reports.")

    keys = tuple(str(key) for key in replay_scalar_fns)
    if not keys:
        raise ValueError("replay_scalar_fns must contain at least one scalar")
    objective_values = complete_report.get("objective_values", {})
    for key in keys:
        if key not in objective_values:
            raise KeyError(f"scalar_key {key!r} not present in complete_report['objective_values']")

    def _option_for(option: float | Mapping[str, float], key: str) -> float:
        if isinstance(option, Mapping):
            return float(option[key])
        return float(option)

    replay_gate = direct_coil_same_branch_replay_gate_report(complete_report)
    branch = complete_report.get("branch_compatibility", {})
    same_branch = bool(branch.get("same_branch", False))
    base = complete_report["base"]
    traces = tuple(base["traces"])
    if not traces:
        raise ValueError("complete_report base payload contains no accepted traces")
    replay_options: dict[str, Any] = {
        "static": base["init"].static,
        "traces": traces,
        "signgs": int(base["init"].signgs),
        "state_weight": 0.0,
        "bsqvac_weight": 0.0,
        "force_weight": 0.0,
        "enforce_edge": False,
        "use_preconditioner_policy_segments": True,
    }
    if replay_kwargs:
        replay_options.update(replay_kwargs)
    replay_traces = tuple(replay_options.get("traces", traces))
    if not replay_traces:
        raise ValueError("replay traces must contain at least one accepted trace")
    replay_branch_metadata = direct_coil_accepted_trace_branch_metadata(
        replay_traces,
        accept_mask=replay_options.get("accept_mask"),
        done_mask=replay_options.get("done_mask"),
        max_steps=replay_options.get("max_steps"),
        json_safe=False,
    )

    scalar_fns = tuple(
        (lambda replay, fn=fn: fn(replay, base))
        for fn in replay_scalar_fns.values()
    )

    def _controller_scalars(coil_params):
        return direct_coil_accepted_trace_controller_custom_vjp_scalars_jax(
            coil_params,
            replay_traces[0]["state_pre"],
            scalar_fns=scalar_fns,
            **replay_options,
        )

    def _shifted(scale):
        return tree_util.tree_map(
            lambda value, delta: jnp.asarray(value) + float(scale) * jnp.asarray(delta),
            base_params,
            direction,
        )

    values, pullback = jax.vjp(_controller_scalars, base_params)
    basis = jnp.eye(len(keys), dtype=jnp.asarray(values).dtype)
    jacobian = _pytree_pullback_basis_jax(pullback, basis)
    exact_directionals = _pytree_batched_directional_vdot_jax(jacobian, direction, len(keys))
    if bool(compute_frozen_fd):
        step = float(eps)
        if not step > 0.0:
            raise ValueError("eps must be positive.")
        frozen_fd_directionals = (
            _controller_scalars(_shifted(step)) - _controller_scalars(_shifted(-step))
        ) / (2.0 * step)
    else:
        frozen_fd_directionals = jnp.full_like(exact_directionals, jnp.nan)

    scalar_reports: dict[str, dict[str, Any]] = {}
    passed_values: list[bool] = []
    for index, key in enumerate(keys):
        value = float(np.asarray(values[index], dtype=float))
        exact = float(np.asarray(exact_directionals[index], dtype=float))
        frozen_fd = float(np.asarray(frozen_fd_directionals[index], dtype=float))
        complete_values = objective_values[key]
        complete_base = float(complete_values["base"])
        complete_fd = float(complete_values["central_fd_directional"])
        abs_error = abs(exact - complete_fd)
        rel_error = abs_error / max(1.0, abs(complete_fd))
        base_abs_delta = abs(value - complete_base)
        key_passed = bool(
            replay_gate["passed"]
            and np.isfinite(exact)
            and np.isfinite(complete_fd)
            and abs_error <= _option_for(atol, key) + _option_for(rtol, key) * abs(complete_fd)
            and base_abs_delta <= _option_for(base_value_atol, key)
        )
        passed_values.append(key_passed)
        scalar_reports[key] = {
            "scalar_key": key,
            "passed": key_passed,
            "same_branch": same_branch,
            "replay_gate": replay_gate,
            "value": values[index],
            "exact_directional": exact_directionals[index],
            "frozen_trace_fd_directional": frozen_fd_directionals[index],
            "complete_fd_directional": complete_fd,
            "abs_error": abs_error,
            "rel_error": rel_error,
            "base_value": value,
            "complete_base_value": complete_base,
            "base_abs_delta": base_abs_delta,
            "complete_values": complete_values,
        }
    return {
        "scalar_keys": keys,
        "passed": bool(all(passed_values)),
        "same_branch": same_branch,
        "replay_gate": replay_gate,
        "replay_option_flags": {
            "use_preconditioner_policy_segments": bool(replay_options.get("use_preconditioner_policy_segments", False)),
            "use_stacked_step_controls": bool(replay_options.get("use_stacked_step_controls", False)),
            "use_accepted_only_fast_path": bool(replay_options.get("use_accepted_only_fast_path", True)),
            "include_analytic": bool(replay_options.get("include_analytic", True)),
            "include_mode_diagnostics": bool(replay_options.get("include_mode_diagnostics", False)),
            "freeze_vacuum_field": bool(replay_options.get("freeze_vacuum_field", False)),
            "freeze_freeb_bsqvac": bool(replay_options.get("freeze_freeb_bsqvac", False)),
        },
        "replay_branch_metadata": replay_branch_metadata,
        "values": values,
        "jacobian": jacobian,
        "exact_directionals": exact_directionals,
        "frozen_trace_fd_directionals": frozen_fd_directionals,
        "scalar_reports": scalar_reports,
    }


def direct_coil_same_branch_physical_scalar_gate_report(
    complete_report: Mapping[str, Any],
    scalars_report: Mapping[str, Any],
    *,
    scalar_keys: tuple[str, ...] | list[str] | None = None,
    json_safe: bool = False,
) -> dict[str, Any]:
    """Return a reviewer-facing same-branch physical-scalar promotion gate.

    This helper composes the complete-solve central-FD report with the
    branch-local accepted-controller custom-VJP scalar report.  Passing this
    gate means the named physical scalars agree with complete-solve central
    finite differences *under an unchanged accepted-trace fingerprint*.  It is
    intentionally explicit that this is not yet differentiation through an
    arbitrary adaptive host-controller branch change.
    """

    replay_gate = direct_coil_same_branch_replay_gate_report(complete_report)
    scalar_reports = scalars_report.get("scalar_reports", {})
    if scalar_keys is None:
        scalar_keys = tuple(str(key) for key in scalars_report.get("scalar_keys", tuple(scalar_reports)))
    else:
        scalar_keys = tuple(str(key) for key in scalar_keys)

    errors: list[str] = []
    if not bool(replay_gate.get("passed", False)):
        errors.append("same-branch replay gate failed")
    if bool(replay_gate.get("differentiates_adaptive_controller", True)):
        errors.append("replay gate unexpectedly claims adaptive-controller differentiation")
    if not bool(scalars_report.get("same_branch", False)):
        errors.append("scalar report is not same-branch")

    objective_values = complete_report.get("objective_values", {})
    branch = complete_report.get("branch_compatibility", {})
    same_accepted_trace_branch = bool(branch.get("same_accepted_trace_branch", branch.get("same_branch", False)))
    same_residual_branch = bool(branch.get("same_residual_branch", branch.get("same_branch", False)))
    if not same_accepted_trace_branch:
        errors.append("accepted-trace branch fingerprint changed")
    if not same_residual_branch:
        errors.append("residual-controller branch fingerprint changed")

    scalar_summaries: dict[str, dict[str, float | bool]] = {}
    for key in scalar_keys:
        scalar_report = scalar_reports.get(key)
        if not isinstance(scalar_report, Mapping):
            errors.append(f"{key}: missing scalar report")
            continue
        if key not in objective_values:
            errors.append(f"{key}: missing complete-solve objective values")
            continue
        if not bool(scalar_report.get("passed", False)):
            errors.append(f"{key}: scalar AD-vs-FD report failed")
        if not bool(scalar_report.get("same_branch", False)):
            errors.append(f"{key}: scalar report is not same-branch")
        complete_fd = float(objective_values[key]["central_fd_directional"])
        exact = float(np.asarray(scalar_report.get("exact_directional"), dtype=float))
        base_abs_delta = float(scalar_report.get("base_abs_delta", np.nan))
        if not np.isfinite(complete_fd):
            errors.append(f"{key}: non-finite complete-solve FD slope")
        if not np.isfinite(exact):
            errors.append(f"{key}: non-finite custom-VJP slope")
        scalar_summaries[key] = {
            "passed": bool(scalar_report.get("passed", False)),
            "complete_fd_directional": complete_fd,
            "exact_directional": exact,
            "abs_error": float(scalar_report.get("abs_error", np.nan)),
            "rel_error": float(scalar_report.get("rel_error", np.nan)),
            "base_abs_delta": base_abs_delta,
        }

    result = {
        "contract": "same-branch complete-solve physical-scalar AD-vs-FD gate",
        "passed": len(errors) == 0,
        "same_branch": bool(branch.get("same_branch", False)),
        "same_accepted_trace_branch": same_accepted_trace_branch,
        "same_residual_branch": same_residual_branch,
        "differentiates_adaptive_controller": False,
        "scalar_keys": scalar_keys,
        "replay_gate": replay_gate,
        "errors": tuple(errors),
        "scalars": scalar_summaries,
    }
    if json_safe:
        return _json_safe_fingerprint_value(result)
    return result


def _accepted_step_policy_signature_for_complete_payload(payload: Mapping[str, Any]) -> tuple[Any, ...]:
    traces = tuple(payload.get("traces", ()))
    if not traces:
        return ()
    return tuple(
        (
            int(segment["start"]),
            int(segment["stop"]),
            int(segment["n_steps"]),
            segment["signature"],
        )
        for segment in direct_coil_accepted_trace_step_policy_segments(traces)
    )


def _accepted_step_policy_summary_for_complete_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    traces = tuple(payload.get("traces", ()))
    if not traces:
        return {"n_segments": 0, "segments": ()}
    segments = direct_coil_accepted_trace_step_policy_segments(traces)
    return {
        "n_segments": len(segments),
        "segments": tuple(
            {
                "start": int(segment["start"]),
                "stop": int(segment["stop"]),
                "n_steps": int(segment["n_steps"]),
            }
            for segment in segments
        ),
    }


def direct_coil_adaptive_full_loop_same_branch_gate_report(
    complete_report: Mapping[str, Any],
    scalars_report: Mapping[str, Any],
    *,
    scalar_keys: tuple[str, ...] | list[str] | None = None,
    require_stacked_step_controls: bool = True,
    require_fixed_rejected_controller_slot: bool = False,
    json_safe: bool = False,
) -> dict[str, Any]:
    """Report whether complete-loop FD is compatible with stacked replay AD.

    This is deliberately a report-only production seam.  It validates that the
    complete host-loop finite-difference triplet stayed on the same adaptive
    branch and that the frozen accepted-branch stacked replay custom VJP
    matches physical scalar finite differences.  It is *not* a custom VJP for
    :func:`vmec_jax.driver.run_free_boundary` and must keep the adaptive
    controller-differentiation flags false.
    """

    physical_gate = direct_coil_same_branch_physical_scalar_gate_report(
        complete_report,
        scalars_report,
        scalar_keys=scalar_keys,
        json_safe=False,
    )
    branch = complete_report.get("branch_compatibility", {})
    branch_fingerprints = {
        "base": branch.get("base_fingerprint", {}),
        "plus": branch.get("plus_fingerprint", {}),
        "minus": branch.get("minus_fingerprint", {}),
    }
    residual_branch_fingerprints = {
        "base": branch.get("base_residual_fingerprint", {}),
        "plus": branch.get("plus_residual_fingerprint", {}),
        "minus": branch.get("minus_residual_fingerprint", {}),
    }
    same_full_loop_branch_fingerprint = bool(branch.get("same_branch", False)) and all(
        isinstance(branch_fingerprints[label], Mapping) and bool(branch_fingerprints[label])
        for label in ("base", "plus", "minus")
    )
    same_residual_branch_fingerprint = bool(branch.get("same_residual_branch", False))
    errors = [f"physical scalar gate: {error}" for error in physical_gate.get("errors", ())]
    if not same_full_loop_branch_fingerprint:
        errors.append("complete-loop branch fingerprints are missing or changed")
    replay_option_flags = scalars_report.get("replay_option_flags", {})
    used_stacked_step_controls = bool(replay_option_flags.get("use_stacked_step_controls", False))
    if bool(require_stacked_step_controls) and not used_stacked_step_controls:
        errors.append("stacked step-control replay was not used")
    replay_branch_metadata = scalars_report.get("replay_branch_metadata", {})
    fixed_rejected_controller_slots = 0
    if isinstance(replay_branch_metadata, Mapping):
        rejected_mask = replay_branch_metadata.get("rejected_mask")
        if rejected_mask is not None:
            fixed_rejected_controller_slots = int(np.count_nonzero(np.asarray(rejected_mask, dtype=bool)))
    fixed_rejected_controller_slot_present = fixed_rejected_controller_slots > 0
    if bool(require_fixed_rejected_controller_slot):
        if not fixed_rejected_controller_slot_present:
            errors.append("fixed rejected controller slot was not replayed")
        if bool(replay_option_flags.get("use_accepted_only_fast_path", True)):
            errors.append("accepted-only fast path was used for a rejected-slot replay gate")

    labels = ("base", "plus", "minus")
    step_policy_signatures: dict[str, tuple[Any, ...]] = {}
    step_policy_summaries: dict[str, dict[str, Any]] = {}
    for label in labels:
        payload = complete_report.get(label)
        if not isinstance(payload, Mapping):
            errors.append(f"{label}: missing complete-solve payload")
            step_policy_signatures[label] = ()
            step_policy_summaries[label] = {"n_segments": 0, "segments": ()}
            continue
        signature = _accepted_step_policy_signature_for_complete_payload(payload)
        if not signature:
            errors.append(f"{label}: no accepted step-policy segments")
        step_policy_signatures[label] = signature
        step_policy_summaries[label] = _accepted_step_policy_summary_for_complete_payload(payload)

    same_stacked_step_policy_branch = (
        bool(step_policy_signatures.get("base"))
        and step_policy_signatures.get("base") == step_policy_signatures.get("plus")
        and step_policy_signatures.get("base") == step_policy_signatures.get("minus")
    )
    if not same_stacked_step_policy_branch:
        errors.append("stacked step-policy branch changed")

    result = {
        "contract": "same-branch adaptive full-loop seam report",
        "passed": len(errors) == 0,
        "ad_vs_fd_gate": "complete-loop central FD vs branch-local stacked replay custom VJP",
        "adaptive_loop_scope": "fingerprint-gated branch-local accepted/rejected replay slots",
        "unclaimed_adaptive_controller_reason": (
            "host adaptive branch selection remains outside the custom VJP; "
            "same-branch finite differences validate the fixed accepted/rejected controller slots"
        ),
        "differentiates_adaptive_controller": False,
        "differentiates_run_free_boundary": False,
        "fingerprint_gated": True,
        "same_branch": bool(branch.get("same_branch", False)),
        "same_accepted_trace_branch": bool(branch.get("same_accepted_trace_branch", branch.get("same_branch", False))),
        "same_residual_branch": bool(branch.get("same_residual_branch", branch.get("same_branch", False))),
        "same_full_loop_branch_fingerprint": bool(same_full_loop_branch_fingerprint),
        "same_residual_branch_fingerprint": bool(same_residual_branch_fingerprint),
        "branch_fingerprints": branch_fingerprints,
        "residual_branch_fingerprints": residual_branch_fingerprints,
        "same_stacked_step_policy_branch": bool(same_stacked_step_policy_branch),
        "requires_stacked_step_controls": bool(require_stacked_step_controls),
        "used_stacked_step_controls": used_stacked_step_controls,
        "requires_fixed_rejected_controller_slot": bool(require_fixed_rejected_controller_slot),
        "fixed_rejected_controller_slot_present": bool(fixed_rejected_controller_slot_present),
        "fixed_rejected_controller_slots": int(fixed_rejected_controller_slots),
        "replay_option_flags": replay_option_flags,
        "replay_branch_metadata": replay_branch_metadata,
        "scalar_keys": physical_gate.get("scalar_keys", ()),
        "physical_scalar_gate": physical_gate,
        "step_policy_segments": step_policy_summaries,
        "errors": tuple(errors),
    }
    if json_safe:
        return _json_safe_fingerprint_value(result)
    return result


def direct_coil_run_free_boundary_branch_local_scalar_value_and_grad_jax(
    input_path: Any | None = None,
    params: Any | None = None,
    *,
    scalar_fn: Any,
    replay_scalar_fn: Any,
    scalar_key: str | None = None,
    production_values: Mapping[str, Any] | None = None,
    replay_payload: Mapping[str, Any] | None = None,
    replay_plan: Mapping[str, Any] | None = None,
    complete_payload: Mapping[str, Any] | None = None,
    init_kwargs: dict[str, Any] | None = None,
    solve_kwargs: dict[str, Any] | None = None,
    replay_kwargs: dict[str, Any] | None = None,
    replay_ad_mode: str = "direct",
    include_trace_replay_diagnostics: bool = True,
    include_payload: bool = True,
    include_replay_graph_metadata: bool = True,
    use_replay_plan: bool = True,
    require_active_trace: bool = True,
) -> dict[str, Any]:
    """Return a production-forward branch-local scalar value and gradient.

    The forward value is evaluated from an actual direct-coil free-boundary
    solve payload, either supplied as ``complete_payload`` or obtained by
    calling :func:`direct_coil_complete_solve_trace`.  The gradient is computed
    by replaying the saved accepted branch through the stacked accepted
    controller custom-VJP path.  This is the narrow production seam currently
    validated by complete-loop finite differences: it differentiates direct
    coils through a *fixed accepted branch*, not arbitrary adaptive host
    controller branch changes.

    ``scalar_fn(payload)`` must return the production scalar from the complete
    solve payload.  Callers that already evaluated the production scalar, for
    example through :func:`direct_coil_same_branch_complete_solve_fd_report`,
    can pass ``production_values`` to avoid recomputing it.  The
    ``replay_scalar_fn(replay, payload)`` must return the same scalar from the
    JAX-visible replay dictionary.  ``replay_payload`` can be supplied to pass a
    slim context into that function, avoiding closure capture of a full complete
    solve payload during cold replay/JVP graph construction.  Set
    ``include_payload=False`` for production reports that only need scalar
    values/derivatives and should not retain the full complete-solve payload.
    Set ``include_replay_graph_metadata=False`` when a compact production
    report does not need structural replay metadata.
    """

    if jax is None:  # pragma: no cover - JAX is required for this helper.
        raise RuntimeError("JAX is required for branch-local scalar gradients.")

    ad_mode = str(replay_ad_mode).strip().lower()
    if ad_mode not in {"direct", "custom_vjp"}:
        raise ValueError("replay_ad_mode must be 'direct' or 'custom_vjp'")

    timings: dict[str, float] = {}
    total_start = time.perf_counter()
    if complete_payload is None:
        if input_path is None or params is None:
            raise ValueError("input_path and params are required when complete_payload is not supplied")
        t0 = time.perf_counter()
        payload = direct_coil_complete_solve_trace(
            input_path,
            params,
            init_kwargs=init_kwargs,
            solve_kwargs=solve_kwargs,
            require_active_trace=require_active_trace,
        )
        timings["complete_solve_trace_wall_s"] = float(time.perf_counter() - t0)
    else:
        t0 = time.perf_counter()
        payload = dict(complete_payload)
        if params is None:
            params = payload.get("params")
        if params is None:
            raise ValueError("params must be supplied when complete_payload does not contain params")
        timings["payload_copy_wall_s"] = float(time.perf_counter() - t0)

    traces = tuple(payload.get("traces", ()))
    if not traces:
        raise ValueError("complete payload contains no accepted traces")
    active_trace = any(trace.get("freeb_bsqvac_half") is not None for trace in traces)
    if bool(require_active_trace) and not active_trace:
        raise RuntimeError("complete payload contains no active free-boundary trace")
    init = payload.get("init")
    if init is None:
        raise ValueError("complete payload is missing the initialization result")

    t0 = time.perf_counter()
    values = _complete_solve_objective_values(
        scalar_fn(payload) if production_values is None else production_values
    )
    timings["production_scalar_eval_wall_s"] = float(time.perf_counter() - t0)
    production_values_source = "scalar_fn" if production_values is None else "precomputed"
    key = str(scalar_key or ("objective" if "objective" in values else next(iter(values))))
    if key not in values:
        raise KeyError(f"scalar_key {key!r} not returned by scalar_fn")

    replay_options: dict[str, Any] = {
        "static": init.static,
        "traces": traces,
        "signgs": int(init.signgs),
        "state_weight": 0.0,
        "bsqvac_weight": 0.0,
        "force_weight": 0.0,
        "enforce_edge": False,
        "use_preconditioner_policy_segments": True,
        "use_stacked_step_controls": True,
        "include_replay_aux": False,
        "unroll_accepted_only_segments_below": 8,
    }
    if replay_kwargs:
        replay_options.update(replay_kwargs)
    replay_traces_for_scalars = tuple(replay_options.get("traces", traces))
    replay_branch_metadata = direct_coil_accepted_trace_branch_metadata(
        replay_traces_for_scalars,
        accept_mask=replay_options.get("accept_mask"),
        done_mask=replay_options.get("done_mask"),
        max_steps=replay_options.get("max_steps"),
        json_safe=True,
    )
    replay_payload_for_scalars = payload if replay_payload is None else replay_payload
    replay_payload_source = "complete_payload" if replay_payload is None else "user"
    replay_plan_for_scalars = replay_plan
    if replay_plan_for_scalars is None and bool(use_replay_plan):
        t0 = time.perf_counter()
        replay_plan_for_scalars = direct_coil_accepted_trace_controller_replay_plan(
            replay_traces_for_scalars,
            static=init.static,
            accept_mask=replay_options.get("accept_mask"),
            done_mask=replay_options.get("done_mask"),
            max_steps=replay_options.get("max_steps"),
            use_preconditioner_policy_segments=bool(
                replay_options.get("use_preconditioner_policy_segments", False)
            ),
            use_segment_preconditioner_controls=bool(
                replay_options.get("use_segment_preconditioner_controls", False)
            ),
            use_stacked_step_controls=bool(replay_options.get("use_stacked_step_controls", False)),
            use_accepted_only_fast_path=bool(replay_options.get("use_accepted_only_fast_path", True)),
        )
        timings["replay_plan_build_wall_s"] = float(time.perf_counter() - t0)
    else:
        timings["replay_plan_build_wall_s"] = 0.0

    if bool(include_replay_graph_metadata):
        graph_metadata = direct_coil_accepted_trace_replay_graph_metadata(
            replay_traces_for_scalars,
            static=init.static,
            accept_mask=replay_options.get("accept_mask"),
            done_mask=replay_options.get("done_mask"),
            max_steps=replay_options.get("max_steps"),
            sample_nzeta=replay_options.get("sample_nzeta"),
            include_analytic=bool(replay_options.get("include_analytic", True)),
            use_stacked_step_controls=bool(replay_options.get("use_stacked_step_controls", False)),
            use_accepted_only_fast_path=bool(replay_options.get("use_accepted_only_fast_path", True)),
            json_safe=True,
        )
    else:
        graph_metadata = {
            "contract": "fixed accepted-branch replay graph metadata",
            "omitted": True,
            "reason": "include_replay_graph_metadata=False",
            "differentiates_adaptive_controller": False,
        }

    def _replay_scalar_direct(coil_params):
        replay = direct_coil_accepted_trace_controller_replay_objective_jax(
            coil_params,
            replay_traces_for_scalars[0]["state_pre"],
            replay_plan=replay_plan_for_scalars,
            **replay_options,
        )
        return replay_scalar_fn(replay, replay_payload_for_scalars)

    def _replay_scalar_custom_vjp(coil_params):
        return direct_coil_accepted_trace_controller_custom_vjp_scalar_jax(
            coil_params,
            replay_traces_for_scalars[0]["state_pre"],
            scalar_fn=lambda replay: replay_scalar_fn(replay, replay_payload_for_scalars),
            replay_plan=replay_plan_for_scalars,
            **replay_options,
        )

    _replay_scalar = _replay_scalar_direct if ad_mode == "direct" else _replay_scalar_custom_vjp

    t0 = time.perf_counter()
    replay_value, grad = jax.value_and_grad(_replay_scalar)(params)
    timings["replay_value_and_grad_dispatch_s"] = float(time.perf_counter() - t0)
    t0 = time.perf_counter()
    replay_value, grad = _block_until_ready_for_timing((replay_value, grad))
    timings["replay_value_and_grad_ready_s"] = float(time.perf_counter() - t0)
    timings["replay_value_and_grad_wall_s"] = (
        timings["replay_value_and_grad_dispatch_s"] + timings["replay_value_and_grad_ready_s"]
    )
    t0 = time.perf_counter()
    if bool(include_trace_replay_diagnostics):
        diagnostics = free_boundary_adjoint_trace_replay_diagnostics(traces)
    else:
        diagnostics = {
            "contract": "fixed accepted-trace replay diagnostics only",
            "omitted": True,
            "reason": "include_trace_replay_diagnostics=False",
            "differentiates_adaptive_controller": False,
        }
    timings["trace_replay_diagnostics_wall_s"] = float(time.perf_counter() - t0)
    timings["total_wall_s"] = float(time.perf_counter() - total_start)
    return {
        "contract": "production-forward branch-local run_free_boundary scalar value/gradient",
        "uses_production_forward": True,
        "differentiates_adaptive_controller": False,
        "differentiates_run_free_boundary": False,
        "differentiates_fixed_accepted_branch": True,
        "replay_ad_mode": ad_mode,
        "scalar_key": key,
        "value": float(values[key]),
        "all_values": values,
        "production_values_source": production_values_source,
        "replay_payload_source": replay_payload_source,
        "replay_value": replay_value,
        "base_abs_delta": abs(float(np.asarray(replay_value, dtype=float)) - float(values[key])),
        "grad": grad,
        "payload": payload if bool(include_payload) else None,
        "includes_payload": bool(include_payload),
        "includes_replay_graph_metadata": bool(include_replay_graph_metadata),
        "timings": timings,
        "trace_replay_diagnostics": diagnostics,
        "replay_graph_metadata": graph_metadata,
        "replay_branch_metadata": replay_branch_metadata,
        "replay_option_flags": {
            "use_preconditioner_policy_segments": bool(
                replay_options.get("use_preconditioner_policy_segments", False)
            ),
            "use_stacked_step_controls": bool(replay_options.get("use_stacked_step_controls", False)),
            "use_accepted_only_fast_path": bool(replay_options.get("use_accepted_only_fast_path", True)),
            "use_replay_plan": bool(replay_plan_for_scalars is not None),
            "include_replay_aux": bool(replay_options.get("include_replay_aux", True)),
            "include_analytic": bool(replay_options.get("include_analytic", True)),
            "include_mode_diagnostics": bool(replay_options.get("include_mode_diagnostics", False)),
            "nestor_solve_mode": str(replay_options.get("nestor_solve_mode", "dense")),
            "nestor_operator_solver": str(replay_options.get("nestor_operator_solver", "gmres")),
            "nestor_operator_tol": float(replay_options.get("nestor_operator_tol", 1.0e-11)),
            "nestor_operator_atol": float(replay_options.get("nestor_operator_atol", 1.0e-13)),
            "nestor_operator_maxiter": (
                None
                if replay_options.get("nestor_operator_maxiter") is None
                else int(replay_options.get("nestor_operator_maxiter"))
            ),
            "nestor_operator_restart": (
                None
                if replay_options.get("nestor_operator_restart") is None
                else int(replay_options.get("nestor_operator_restart"))
            ),
            "freeze_vacuum_field": bool(replay_options.get("freeze_vacuum_field", False)),
            "freeze_freeb_bsqvac": bool(replay_options.get("freeze_freeb_bsqvac", False)),
            "state_only_replay": bool(replay_options.get("state_only_replay", False)),
            "jit_preconditioner_apply": bool(replay_options.get("jit_preconditioner_apply", True)),
            "unroll_accepted_only_segments_below": int(
                replay_options.get("unroll_accepted_only_segments_below", 0)
            ),
            "replay_ad_mode": ad_mode,
        },
    }


def direct_coil_run_free_boundary_branch_local_scalars_value_and_jacobian_jax(
    input_path: Any | None = None,
    params: Any | None = None,
    *,
    direction_params: Any | None = None,
    scalar_fn: Any,
    replay_scalar_fns: Mapping[str, Any],
    scalar_keys: tuple[str, ...] | list[str] | None = None,
    production_values: Mapping[str, Any] | None = None,
    replay_payload: Mapping[str, Any] | None = None,
    replay_plan: Mapping[str, Any] | None = None,
    complete_payload: Mapping[str, Any] | None = None,
    init_kwargs: dict[str, Any] | None = None,
    solve_kwargs: dict[str, Any] | None = None,
    replay_kwargs: dict[str, Any] | None = None,
    replay_ad_mode: str = "direct",
    include_trace_replay_diagnostics: bool = True,
    include_payload: bool = True,
    include_replay_graph_metadata: bool = True,
    use_replay_plan: bool = True,
    require_active_trace: bool = True,
) -> dict[str, Any]:
    """Return production-forward branch-local values and a scalar Jacobian.

    This is the vector-valued counterpart of
    :func:`direct_coil_run_free_boundary_branch_local_scalar_value_and_grad_jax`.
    The values are evaluated from a real direct-coil complete solve payload,
    while the Jacobian is computed by replaying the fixed accepted branch with
    a vector-output custom-VJP seam.  The contract is intentionally narrow: it
    differentiates direct-coil parameters through the saved accepted branch and
    does not differentiate adaptive host-controller branch changes.

    If ``direction_params`` is supplied, the helper computes only the
    directional derivatives ``J @ direction_params`` using ``jax.jvp`` instead
    of materializing the full Jacobian.  This is the fast path for production
    validation reports that compare against one complete-solve central
    finite-difference direction.

    ``scalar_fn(payload)`` must return a mapping of production scalar values.
    Callers that already have the production base values can pass
    ``production_values`` to avoid recomputing them from ``scalar_fn``.
    ``replay_scalar_fns`` maps the same scalar keys to callables of the form
    ``fn(replay, payload)`` that evaluate those scalars from the JAX-visible
    accepted-controller replay.  ``replay_payload`` can be supplied to pass a
    slim context into those functions, avoiding closure capture of a full
    complete-solve payload during cold replay/JVP graph construction.  Set
    ``include_payload=False`` for production reports that only need scalar
    values/derivatives and should not retain the full complete-solve payload.
    Set ``include_replay_graph_metadata=False`` when a compact production
    report does not need structural replay metadata.
    """

    if jax is None:  # pragma: no cover - JAX is required for this helper.
        raise RuntimeError("JAX is required for branch-local scalar gradients.")
    if not replay_scalar_fns:
        raise ValueError("replay_scalar_fns must contain at least one scalar")
    ad_mode = str(replay_ad_mode).strip().lower()
    if ad_mode not in {"direct", "custom_vjp"}:
        raise ValueError("replay_ad_mode must be 'direct' or 'custom_vjp'")
    if direction_params is not None and ad_mode != "direct":
        raise ValueError("direction_params directional mode requires replay_ad_mode='direct'")

    timings: dict[str, float] = {}
    total_start = time.perf_counter()
    if complete_payload is None:
        if input_path is None or params is None:
            raise ValueError("input_path and params are required when complete_payload is not supplied")
        t0 = time.perf_counter()
        payload = direct_coil_complete_solve_trace(
            input_path,
            params,
            init_kwargs=init_kwargs,
            solve_kwargs=solve_kwargs,
            require_active_trace=require_active_trace,
        )
        timings["complete_solve_trace_wall_s"] = float(time.perf_counter() - t0)
    else:
        t0 = time.perf_counter()
        payload = dict(complete_payload)
        if params is None:
            params = payload.get("params")
        if params is None:
            raise ValueError("params must be supplied when complete_payload does not contain params")
        timings["payload_copy_wall_s"] = float(time.perf_counter() - t0)

    traces = tuple(payload.get("traces", ()))
    if not traces:
        raise ValueError("complete payload contains no accepted traces")
    active_trace = any(trace.get("freeb_bsqvac_half") is not None for trace in traces)
    if bool(require_active_trace) and not active_trace:
        raise RuntimeError("complete payload contains no active free-boundary trace")
    init = payload.get("init")
    if init is None:
        raise ValueError("complete payload is missing the initialization result")

    t0 = time.perf_counter()
    all_values = _complete_solve_objective_values(
        scalar_fn(payload) if production_values is None else production_values
    )
    timings["production_scalar_eval_wall_s"] = float(time.perf_counter() - t0)
    production_values_source = "scalar_fn" if production_values is None else "precomputed"
    keys = tuple(str(key) for key in (scalar_keys if scalar_keys is not None else tuple(replay_scalar_fns)))
    if not keys:
        raise ValueError("scalar_keys must contain at least one scalar")
    for key in keys:
        if key not in all_values:
            raise KeyError(f"scalar_key {key!r} not returned by scalar_fn")
        if key not in replay_scalar_fns:
            raise KeyError(f"scalar_key {key!r} not present in replay_scalar_fns")

    replay_options: dict[str, Any] = {
        "static": init.static,
        "traces": traces,
        "signgs": int(init.signgs),
        "state_weight": 0.0,
        "bsqvac_weight": 0.0,
        "force_weight": 0.0,
        "enforce_edge": False,
        "use_preconditioner_policy_segments": True,
        "use_stacked_step_controls": True,
        "include_replay_aux": False,
        "unroll_accepted_only_segments_below": 8,
    }
    if replay_kwargs:
        replay_options.update(replay_kwargs)
    replay_traces_for_scalars = tuple(replay_options.get("traces", traces))
    replay_branch_metadata = direct_coil_accepted_trace_branch_metadata(
        replay_traces_for_scalars,
        accept_mask=replay_options.get("accept_mask"),
        done_mask=replay_options.get("done_mask"),
        max_steps=replay_options.get("max_steps"),
        json_safe=True,
    )
    replay_payload_for_scalars = payload if replay_payload is None else replay_payload
    replay_payload_source = "complete_payload" if replay_payload is None else "user"
    replay_plan_for_scalars = replay_plan
    if replay_plan_for_scalars is None and bool(use_replay_plan):
        t0 = time.perf_counter()
        replay_plan_for_scalars = direct_coil_accepted_trace_controller_replay_plan(
            replay_traces_for_scalars,
            static=init.static,
            accept_mask=replay_options.get("accept_mask"),
            done_mask=replay_options.get("done_mask"),
            max_steps=replay_options.get("max_steps"),
            use_preconditioner_policy_segments=bool(
                replay_options.get("use_preconditioner_policy_segments", False)
            ),
            use_segment_preconditioner_controls=bool(
                replay_options.get("use_segment_preconditioner_controls", False)
            ),
            use_stacked_step_controls=bool(replay_options.get("use_stacked_step_controls", False)),
            use_accepted_only_fast_path=bool(replay_options.get("use_accepted_only_fast_path", True)),
        )
        timings["replay_plan_build_wall_s"] = float(time.perf_counter() - t0)
    else:
        timings["replay_plan_build_wall_s"] = 0.0

    if bool(include_replay_graph_metadata):
        graph_metadata = direct_coil_accepted_trace_replay_graph_metadata(
            replay_traces_for_scalars,
            static=init.static,
            accept_mask=replay_options.get("accept_mask"),
            done_mask=replay_options.get("done_mask"),
            max_steps=replay_options.get("max_steps"),
            sample_nzeta=replay_options.get("sample_nzeta"),
            include_analytic=bool(replay_options.get("include_analytic", True)),
            use_stacked_step_controls=bool(replay_options.get("use_stacked_step_controls", False)),
            use_accepted_only_fast_path=bool(replay_options.get("use_accepted_only_fast_path", True)),
            json_safe=True,
        )
    else:
        graph_metadata = {
            "contract": "fixed accepted-branch replay graph metadata",
            "omitted": True,
            "reason": "include_replay_graph_metadata=False",
            "differentiates_adaptive_controller": False,
        }

    scalar_fn_seq = tuple(
        (lambda replay, key=key: replay_scalar_fns[key](replay, replay_payload_for_scalars)) for key in keys
    )

    def _replay_scalars_direct(coil_params):
        replay = direct_coil_accepted_trace_controller_replay_objective_jax(
            coil_params,
            replay_traces_for_scalars[0]["state_pre"],
            replay_plan=replay_plan_for_scalars,
            **replay_options,
        )
        return jnp.asarray([fn(replay) for fn in scalar_fn_seq])

    def _replay_scalars_custom_vjp(coil_params):
        return direct_coil_accepted_trace_controller_custom_vjp_scalars_jax(
            coil_params,
            replay_traces_for_scalars[0]["state_pre"],
            scalar_fns=scalar_fn_seq,
            replay_plan=replay_plan_for_scalars,
            **replay_options,
        )

    _replay_scalars = _replay_scalars_direct if ad_mode == "direct" else _replay_scalars_custom_vjp

    derivative_mode = "full_jacobian_vjp"
    jacobian = None
    gradients: dict[str, Any] = {}
    directional_values = None
    directional_fast_path = "none"
    directional_uses_fixed_coil_geometry = False
    if direction_params is not None:
        derivative_mode = "directional_jvp"
        current_only_direction = False
        current_direction_leaf = None
        current_base_leaf = None
        try:
            from .external_fields import CoilFieldParams

            if isinstance(params, CoilFieldParams) and isinstance(direction_params, CoilFieldParams):
                direction_dofs = np.asarray(direction_params.base_curve_dofs, dtype=float)
                current_only_direction = not np.any(direction_dofs)
                if current_only_direction:
                    current_base_leaf = jnp.asarray(params.base_currents)
                    current_direction_leaf = jnp.asarray(direction_params.base_currents)
        except Exception:
            current_only_direction = False
            current_direction_leaf = None
            current_base_leaf = None

        if current_only_direction and current_base_leaf is not None and current_direction_leaf is not None:
            directional_fast_path = "current_only"
            directional_uses_fixed_coil_geometry = True
            from .external_fields import build_coil_field_geometry, apply_stellarator_symmetry_to_currents

            fixed_gamma, fixed_gamma_dash, _fixed_currents = build_coil_field_geometry(params)

            def _fixed_geometry_for_currents(base_currents):
                expanded_currents = params.current_scale * apply_stellarator_symmetry_to_currents(
                    base_currents,
                    nfp=params.nfp,
                    stellsym=params.stellsym,
                )
                return fixed_gamma, fixed_gamma_dash, expanded_currents

            def _replay_scalars_current_only(base_currents):
                replay = direct_coil_accepted_trace_controller_replay_objective_jax(
                    params.with_arrays(base_currents=base_currents),
                    replay_traces_for_scalars[0]["state_pre"],
                    replay_plan=replay_plan_for_scalars,
                    coil_geometry=_fixed_geometry_for_currents(base_currents),
                    **replay_options,
                )
                return jnp.asarray([fn(replay) for fn in scalar_fn_seq])

            jvp_primal = (current_base_leaf,)
            jvp_tangent = (current_direction_leaf,)
            jvp_fn = _replay_scalars_current_only
        else:
            jvp_primal = (params,)
            jvp_tangent = (direction_params,)
            jvp_fn = _replay_scalars

        t0 = time.perf_counter()
        replay_values, directional_values = jax.jvp(
            jvp_fn,
            jvp_primal,
            jvp_tangent,
        )
        timings["replay_jvp_dispatch_s"] = float(time.perf_counter() - t0)
        t0 = time.perf_counter()
        replay_values, directional_values = _block_until_ready_for_timing((replay_values, directional_values))
        timings["replay_jvp_ready_s"] = float(time.perf_counter() - t0)
        timings["replay_jvp_wall_s"] = timings["replay_jvp_dispatch_s"] + timings["replay_jvp_ready_s"]
        # Compatibility timing keys: no full VJP/Jacobian was built.
        timings["replay_vjp_wall_s"] = 0.0
        timings["replay_pullbacks_wall_s"] = 0.0
        timings["jacobian_stack_ready_s"] = 0.0
    else:
        t0 = time.perf_counter()
        replay_values, pullback = jax.vjp(_replay_scalars, params)
        timings["replay_vjp_dispatch_s"] = float(time.perf_counter() - t0)
        t0 = time.perf_counter()
        replay_values = _block_until_ready_for_timing(replay_values)
        timings["replay_vjp_ready_s"] = float(time.perf_counter() - t0)
        basis = jnp.eye(len(keys), dtype=jnp.asarray(replay_values).dtype)
        t0 = time.perf_counter()
        jacobian = _pytree_pullback_basis_jax(pullback, basis)
        timings["replay_pullbacks_dispatch_s"] = float(time.perf_counter() - t0)
        t0 = time.perf_counter()
        jacobian = _block_until_ready_for_timing(jacobian)
        timings["replay_pullbacks_ready_s"] = float(time.perf_counter() - t0)
        timings["replay_vjp_wall_s"] = timings["replay_vjp_dispatch_s"] + timings["replay_vjp_ready_s"]
        timings["replay_pullbacks_wall_s"] = (
            timings["replay_pullbacks_dispatch_s"] + timings["replay_pullbacks_ready_s"]
        )
        # Pullback readiness already materialized the full Jacobian pytree.
        # Keep the timing key for report compatibility without re-walking it.
        timings["jacobian_stack_ready_s"] = 0.0
        basis_gradients = _pytree_unstack_leading_axis_jax(jacobian, len(keys))
        gradients = {key: basis_gradients[index] for index, key in enumerate(keys)}
    values = {key: float(all_values[key]) for key in keys}
    replay_value_map = {key: replay_values[index] for index, key in enumerate(keys)}
    base_abs_delta = {
        key: abs(float(np.asarray(replay_values[index], dtype=float)) - float(values[key]))
        for index, key in enumerate(keys)
    }
    directional_derivatives = (
        None
        if directional_values is None
        else {key: directional_values[index] for index, key in enumerate(keys)}
    )
    t0 = time.perf_counter()
    if bool(include_trace_replay_diagnostics):
        diagnostics = free_boundary_adjoint_trace_replay_diagnostics(traces)
    else:
        diagnostics = {
            "contract": "fixed accepted-trace replay diagnostics only",
            "omitted": True,
            "reason": "include_trace_replay_diagnostics=False",
            "differentiates_adaptive_controller": False,
        }
    timings["trace_replay_diagnostics_wall_s"] = float(time.perf_counter() - t0)
    timings["total_wall_s"] = float(time.perf_counter() - total_start)
    return {
        "contract": "production-forward branch-local run_free_boundary scalar values/Jacobian",
        "uses_production_forward": True,
        "differentiates_adaptive_controller": False,
        "differentiates_run_free_boundary": False,
        "differentiates_fixed_accepted_branch": True,
        "replay_ad_mode": ad_mode,
        "derivative_mode": derivative_mode,
        "scalar_keys": keys,
        "values": values,
        "all_values": all_values,
        "production_values_source": production_values_source,
        "replay_payload_source": replay_payload_source,
        "replay_values": replay_values,
        "replay_value_map": replay_value_map,
        "base_abs_delta": base_abs_delta,
        "max_base_abs_delta": max(base_abs_delta.values()) if base_abs_delta else 0.0,
        "jacobian": jacobian,
        "grads": gradients,
        "directional_derivatives": directional_derivatives,
        "payload": payload if bool(include_payload) else None,
        "includes_payload": bool(include_payload),
        "includes_replay_graph_metadata": bool(include_replay_graph_metadata),
        "timings": timings,
        "trace_replay_diagnostics": diagnostics,
        "replay_graph_metadata": graph_metadata,
        "replay_branch_metadata": replay_branch_metadata,
        "replay_option_flags": {
            "use_preconditioner_policy_segments": bool(
                replay_options.get("use_preconditioner_policy_segments", False)
            ),
            "use_stacked_step_controls": bool(replay_options.get("use_stacked_step_controls", False)),
            "use_accepted_only_fast_path": bool(replay_options.get("use_accepted_only_fast_path", True)),
            "use_replay_plan": bool(replay_plan_for_scalars is not None),
            "include_replay_aux": bool(replay_options.get("include_replay_aux", True)),
            "include_analytic": bool(replay_options.get("include_analytic", True)),
            "include_mode_diagnostics": bool(replay_options.get("include_mode_diagnostics", False)),
            "nestor_solve_mode": str(replay_options.get("nestor_solve_mode", "dense")),
            "nestor_operator_solver": str(replay_options.get("nestor_operator_solver", "gmres")),
            "nestor_operator_tol": float(replay_options.get("nestor_operator_tol", 1.0e-11)),
            "nestor_operator_atol": float(replay_options.get("nestor_operator_atol", 1.0e-13)),
            "nestor_operator_maxiter": (
                None
                if replay_options.get("nestor_operator_maxiter") is None
                else int(replay_options.get("nestor_operator_maxiter"))
            ),
            "nestor_operator_restart": (
                None
                if replay_options.get("nestor_operator_restart") is None
                else int(replay_options.get("nestor_operator_restart"))
            ),
            "freeze_vacuum_field": bool(replay_options.get("freeze_vacuum_field", False)),
            "freeze_freeb_bsqvac": bool(replay_options.get("freeze_freeb_bsqvac", False)),
            "state_only_replay": bool(replay_options.get("state_only_replay", False)),
            "jit_preconditioner_apply": bool(replay_options.get("jit_preconditioner_apply", True)),
            "unroll_accepted_only_segments_below": int(
                replay_options.get("unroll_accepted_only_segments_below", 0)
            ),
            "replay_ad_mode": ad_mode,
            "directional_jvp_fast_path": directional_fast_path,
            "directional_uses_fixed_coil_geometry": directional_uses_fixed_coil_geometry,
        },
    }


def direct_coil_fixed_trace_custom_vjp_objective_jax(
    params: Any,
    initial_state: Any,
    **replay_kwargs: Any,
) -> Any:
    """Return a scalar fixed-trace objective with an explicit custom VJP seam.

    This is the production-adjacent phase-2 bridge for direct-coil
    free-boundary adjoints.  The forward objective is the same fixed accepted
    trace replay used by :func:`direct_coil_accepted_trace_replay_objective_jax`.
    The custom backward rule differentiates only that frozen trace replay with
    respect to ``params``.  It deliberately does not differentiate through the
    adaptive host controller that chose accepted/rejected steps, activation
    cadence, limiters, or preconditioner policy.

    The helper is useful for call sites that need a scalar custom-VJP primitive
    while the full production ``run_free_boundary`` nonlinear controller is
    being refactored into a JAX-visible loop.  Use finite-difference trace
    fingerprint checks before promoting gradients from complete solves.
    """

    if jax is None:  # pragma: no cover - JAX is required for custom VJP.
        return direct_coil_accepted_trace_replay_objective_jax(
            params,
            initial_state,
            **replay_kwargs,
        )["objective"]

    def objective(coil_params):
        replay = direct_coil_accepted_trace_replay_objective_jax(
            coil_params,
            initial_state,
            **replay_kwargs,
        )
        return replay["objective"]

    @jax.custom_vjp
    def _wrapped(coil_params):
        return objective(coil_params)

    def _wrapped_fwd(coil_params):
        return objective(coil_params), coil_params

    def _wrapped_bwd(coil_params, cotangent):
        grad_params = jax.grad(objective)(coil_params)
        scaled_grad = tree_util.tree_map(
            lambda value: jnp.asarray(cotangent) * jnp.asarray(value),
            grad_params,
        )
        return (scaled_grad,)

    _wrapped.defvjp(_wrapped_fwd, _wrapped_bwd)
    return _wrapped(params)


def direct_coil_accepted_trace_controller_custom_vjp_objective_jax(
    params: Any,
    initial_state: Any,
    **replay_kwargs: Any,
) -> Any:
    """Return a scalar stacked-controller replay objective with custom VJP.

    This is the preferred phase-2 production-adjacent seam after the accepted
    trace controls have been lifted into a JAX-visible scan.  The forward path
    is :func:`direct_coil_accepted_trace_controller_replay_objective_jax`; the
    backward rule differentiates the same frozen accepted-controller replay
    with respect to coil parameters.  As with the older fixed-trace wrapper,
    adaptive host-control choices must be fingerprint-gated before complete
    solve finite differences are promoted.
    """

    if jax is None:  # pragma: no cover - JAX is required for custom VJP.
        return direct_coil_accepted_trace_controller_replay_objective_jax(
            params,
            initial_state,
            **replay_kwargs,
        )["objective"]

    def objective(coil_params):
        replay = direct_coil_accepted_trace_controller_replay_objective_jax(
            coil_params,
            initial_state,
            **replay_kwargs,
        )
        return replay["objective"]

    @jax.custom_vjp
    def _wrapped(coil_params):
        return objective(coil_params)

    def _wrapped_fwd(coil_params):
        return objective(coil_params), coil_params

    def _wrapped_bwd(coil_params, cotangent):
        grad_params = jax.grad(objective)(coil_params)
        scaled_grad = tree_util.tree_map(
            lambda value: jnp.asarray(cotangent) * jnp.asarray(value),
            grad_params,
        )
        return (scaled_grad,)

    _wrapped.defvjp(_wrapped_fwd, _wrapped_bwd)
    return _wrapped(params)


def direct_coil_accepted_trace_controller_custom_vjp_scalar_jax(
    params: Any,
    initial_state: Any,
    *,
    scalar_fn: Any,
    **replay_kwargs: Any,
) -> Any:
    """Return a scalar of accepted-controller replay with a custom VJP seam.

    ``scalar_fn`` is called with the replay dictionary returned by
    :func:`direct_coil_accepted_trace_controller_replay_objective_jax`; it can
    extract the replayed final state, objective history, or vacuum terms and
    return any scalar JAX expression.  The backward rule differentiates the
    same frozen accepted-controller replay with respect to coil parameters.

    This is a branch-local production-adjacent helper.  It deliberately does
    not differentiate the host policy that selected accepted/rejected steps,
    reset points, limiters, activation cadence, or preconditioner dispatch.
    Complete-solve promotion must therefore be guarded by accepted-trace
    fingerprints before comparing against finite differences.
    """

    if jax is None:  # pragma: no cover - JAX is required for custom VJP.
        replay = direct_coil_accepted_trace_controller_replay_objective_jax(
            params,
            initial_state,
            **replay_kwargs,
        )
        return scalar_fn(replay)

    def objective(coil_params):
        replay = direct_coil_accepted_trace_controller_replay_objective_jax(
            coil_params,
            initial_state,
            **replay_kwargs,
        )
        return scalar_fn(replay)

    @jax.custom_vjp
    def _wrapped(coil_params):
        return objective(coil_params)

    def _wrapped_fwd(coil_params):
        return objective(coil_params), coil_params

    def _wrapped_bwd(coil_params, cotangent):
        grad_params = jax.grad(objective)(coil_params)
        scaled_grad = tree_util.tree_map(
            lambda value: jnp.asarray(cotangent) * jnp.asarray(value),
            grad_params,
        )
        return (scaled_grad,)

    _wrapped.defvjp(_wrapped_fwd, _wrapped_bwd)
    return _wrapped(params)


def direct_coil_accepted_trace_controller_custom_vjp_scalars_jax(
    params: Any,
    initial_state: Any,
    *,
    scalar_fns: Any,
    **replay_kwargs: Any,
) -> Any:
    """Return several accepted-controller replay scalars with one custom VJP.

    The output is a one-dimensional JAX array whose entries are the scalars
    returned by ``scalar_fns``.  The backward rule differentiates the same
    frozen accepted-controller replay and supports vector cotangents, so tests
    can validate several physical scalar pullbacks against one complete-solve
    finite-difference branch report.
    """

    scalar_fn_seq = tuple(scalar_fns)
    if not scalar_fn_seq:
        raise ValueError("scalar_fns must contain at least one scalar function")
    if jax is None:  # pragma: no cover - JAX is required for custom VJP.
        replay = direct_coil_accepted_trace_controller_replay_objective_jax(
            params,
            initial_state,
            **replay_kwargs,
        )
        return jnp.asarray([fn(replay) for fn in scalar_fn_seq])

    def objective(coil_params):
        replay = direct_coil_accepted_trace_controller_replay_objective_jax(
            coil_params,
            initial_state,
            **replay_kwargs,
        )
        return jnp.asarray([fn(replay) for fn in scalar_fn_seq])

    @jax.custom_vjp
    def _wrapped(coil_params):
        return objective(coil_params)

    def _wrapped_fwd(coil_params):
        return objective(coil_params), coil_params

    def _wrapped_bwd(coil_params, cotangent):
        _, pullback = jax.vjp(objective, coil_params)
        return pullback(jnp.asarray(cotangent))

    _wrapped.defvjp(_wrapped_fwd, _wrapped_bwd)
    return _wrapped(params)


def direct_coil_accepted_trace_directional_check_jax(
    params: Any,
    direction: Any,
    initial_state: Any,
    *,
    eps: float = 1.0e-4,
    compute_fd: bool = True,
    **replay_kwargs: Any,
) -> dict[str, Any]:
    """Validate accepted-trace replay coil gradients by central FD.

    This wraps :func:`direct_coil_accepted_trace_replay_objective_jax` with the
    common AD-vs-central-FD contract used throughout the phase-2 free-boundary
    adjoint ladder.  The differentiated path includes direct-coil sampling,
    accepted-boundary geometry resampling, JAX NESTOR replay, and strict VMEC
    accepted updates under fixed production trace controls.

    The helper is production-adjacent but still intentionally scoped: the
    adaptive host controller that created the accepted traces is fixed data, so
    this is not yet a full custom VJP for :func:`vmec_jax.driver.run_free_boundary`.
    """

    def objective(coil_params):
        replay = direct_coil_accepted_trace_replay_objective_jax(
            coil_params,
            initial_state,
            **replay_kwargs,
        )
        return replay["objective"]

    check = pytree_directional_derivative_check_jax(
        objective,
        params,
        direction,
        eps=eps,
        compute_fd=compute_fd,
    )
    replay = direct_coil_accepted_trace_replay_objective_jax(
        params,
        initial_state,
        **replay_kwargs,
    )
    return {
        **check,
        "replay": replay,
        "objective_components": replay["objective_components"],
    }


def direct_coil_accepted_trace_controller_directional_check_jax(
    params: Any,
    direction: Any,
    initial_state: Any,
    *,
    eps: float = 1.0e-4,
    compute_fd: bool = True,
    **replay_kwargs: Any,
) -> dict[str, Any]:
    """Validate stacked accepted-controller replay gradients by central FD.

    This is the scan-controller counterpart to
    :func:`direct_coil_accepted_trace_directional_check_jax`.  It validates the
    differentiated path that carries accepted/rejected masks plus stacked
    scalar, velocity-history, and preconditioner controls through
    :func:`jax_visible_accepted_nonlinear_controller_jax`.  Passing
    ``use_preconditioner_policy_segments=True`` in ``replay_kwargs`` validates
    the segmented static-policy controller path used as the next staging/fusion
    rung for longer accepted traces.
    """

    def objective(coil_params):
        replay = direct_coil_accepted_trace_controller_replay_objective_jax(
            coil_params,
            initial_state,
            **replay_kwargs,
        )
        return replay["objective"]

    check = pytree_directional_derivative_check_jax(
        objective,
        params,
        direction,
        eps=eps,
        compute_fd=compute_fd,
    )
    replay = direct_coil_accepted_trace_controller_replay_objective_jax(
        params,
        initial_state,
        **replay_kwargs,
    )
    return {
        **check,
        "replay": replay,
        "objective_components": replay["objective_components"],
    }


def direct_coil_projected_mode_fixed_point_jax(
    params: Any,
    initial_state: Any,
    *,
    boundary_from_state: Any,
    update_from_response: Any,
    mode_matrix: Any,
    sin_basis: Any,
    xmpot: Any,
    n_raw: Any,
    imirr: Any | None = None,
    imirr_full: Any | None = None,
    cos_basis: Any | None = None,
    onp: float = 1.0,
    lasym: bool = False,
    nuv3: int | None = None,
    nuv_full: int | None = None,
    max_iter: int = 10,
    damping: float = 1.0,
    symmetric: bool = False,
) -> dict[str, Any]:
    """Solve a small direct-coil free-boundary fixed-point validation loop.

    ``boundary_from_state(state)`` must return a mapping with ``R``, ``Z``,
    ``phi``, ``Ru``, ``Zu``, ``Rv``, and ``Zv`` arrays.  At each fixed-point
    step this helper samples the direct Biot-Savart field on that moving
    boundary, projects it into VMEC boundary channels, projects the normal
    source into mode space, solves the dense vacuum mode system, and passes the
    response to ``update_from_response(state, response, vac, boundary, params)``.

    This is a production-adjacent phase-2 validation primitive.  It exercises
    the same dependency graph as a free-boundary coil solve at tiny dense scale,
    while keeping the true production ``run_free_boundary`` loop out of scope
    until that loop is made JAX-visible or receives its own custom VJP.
    """

    from .external_fields import sample_coil_field_cylindrical

    required = ("R", "Z", "phi", "Ru", "Zu", "Rv", "Zv")

    def _mode_response_for_state(state, coil_params):
        boundary = boundary_from_state(state)
        missing = [name for name in required if name not in boundary]
        if missing:
            raise ValueError(f"boundary_from_state missing keys: {missing}")
        br, bp, bz = sample_coil_field_cylindrical(
            coil_params,
            jnp.asarray(boundary["R"]),
            jnp.asarray(boundary["Z"]),
            jnp.asarray(boundary["phi"]),
        )
        vac = vacuum_boundary_fields_from_cylindrical_jax(
            br=br,
            bp=bp,
            bz=bz,
            R=boundary["R"],
            Ru=boundary["Ru"],
            Zu=boundary["Zu"],
            Rv=boundary["Rv"],
            Zv=boundary["Zv"],
        )
        rhs_mode = mode_rhs_from_gsource_jax(
            vac["bnormal"],
            sin_basis=sin_basis,
            cos_basis=cos_basis,
            xmpot=xmpot,
            n_raw=n_raw,
            onp=float(onp),
            lasym=bool(lasym),
            imirr=imirr,
            imirr_full=imirr_full,
            nuv3=nuv3,
            nuv_full=nuv_full,
        )
        response = dense_mode_vacuum_solve_jax(
            mode_matrix,
            rhs_mode,
            sin_basis,
            cos_basis,
            symmetric=bool(symmetric),
        )
        response = {**response, "rhs_mode": rhs_mode}
        return boundary, vac, response

    def _update(state, coil_params):
        boundary, vac, response = _mode_response_for_state(state, coil_params)
        return update_from_response(state, response, vac, boundary, coil_params)

    root = dense_fixed_point_solve_jax(
        _update,
        initial_state,
        params,
        max_iter=max_iter,
        damping=damping,
    )
    boundary, vac, response = _mode_response_for_state(root, params)
    fixed_update = _update(root, params)
    return {
        "state": root,
        "fixed_point_residual": root - fixed_update,
        "update": fixed_update,
        "boundary": boundary,
        "vac": vac,
        "response": response,
    }


def direct_coil_projected_mode_fixed_point_objective_jax(
    params: Any,
    initial_state: Any,
    *,
    boundary_from_state: Any,
    update_from_response: Any,
    mode_matrix: Any,
    sin_basis: Any,
    xmpot: Any,
    n_raw: Any,
    imirr: Any | None = None,
    imirr_full: Any | None = None,
    cos_basis: Any | None = None,
    onp: float = 1.0,
    lasym: bool = False,
    nuv3: int | None = None,
    nuv_full: int | None = None,
    max_iter: int = 10,
    damping: float = 1.0,
    symmetric: bool = False,
    state_weights: Any = 1.0,
    update_weights: Any = 0.0,
    mode_weights: Any = 0.0,
    rhs_mode_weights: Any = 0.0,
    bnormal_weight: float = 0.0,
    fixed_point_residual_weight: float = 1.0,
) -> dict[str, Any]:
    """Return a scalar objective for the projected-mode fixed-point helper.

    This wraps :func:`direct_coil_projected_mode_fixed_point_jax` with the
    quadratic objective shape used by the phase-2 AD-vs-FD gates.  It is useful
    for optimizer-facing validation because it exposes the differentiable
    contract as a scalar objective while still returning the solved state and
    component values for diagnostics.

    The default objective is a weighted half-norm of the solved fixed-point
    state plus a small residual guard.  Additional weights can include the
    fixed-point update, vacuum mode coefficients, mode RHS, and boundary normal
    field.  All weights may be scalars or arrays broadcastable to the
    corresponding component.
    """

    solved = direct_coil_projected_mode_fixed_point_jax(
        params,
        initial_state,
        boundary_from_state=boundary_from_state,
        update_from_response=update_from_response,
        mode_matrix=mode_matrix,
        sin_basis=sin_basis,
        xmpot=xmpot,
        n_raw=n_raw,
        imirr=imirr,
        imirr_full=imirr_full,
        cos_basis=cos_basis,
        onp=onp,
        lasym=lasym,
        nuv3=nuv3,
        nuv_full=nuv_full,
        max_iter=max_iter,
        damping=damping,
        symmetric=symmetric,
    )
    components = {
        "state": _weighted_half_norm(solved["state"], state_weights),
        "update": _weighted_half_norm(solved["update"], update_weights),
        "mode": _weighted_half_norm(solved["response"]["mode_coeffs"], mode_weights),
        "rhs_mode": _weighted_half_norm(solved["response"]["rhs_mode"], rhs_mode_weights),
        "bnormal": _weighted_half_norm(solved["vac"]["bnormal"], bnormal_weight),
        "fixed_point_residual": _weighted_half_norm(
            solved["fixed_point_residual"],
            fixed_point_residual_weight,
        ),
    }
    objective = sum(components.values())
    return {
        **solved,
        "objective": objective,
        "objective_components": components,
    }


def direct_coil_projected_mode_fixed_point_directional_check_jax(
    params: Any,
    direction: Any,
    initial_state: Any,
    *,
    eps: float = 1.0e-4,
    **objective_kwargs: Any,
) -> dict[str, Any]:
    """Validate projected-mode fixed-point coil gradients by central FD.

    This is the reusable phase-2/phase-3 validation rung for the direct-coil
    free-boundary adjoint path.  It wraps
    :func:`direct_coil_projected_mode_fixed_point_objective_jax`, computes the
    exact JAX directional derivative with respect to the coil-parameter pytree,
    and compares it with a central finite difference along ``direction``.

    The helper intentionally targets the tiny JAX-visible projected-mode
    fixed-point surrogate.  It exercises the important dependency chain

    ``coil parameters -> Biot-Savart field -> moving boundary projection ->
    dense vacuum solve -> fixed-point state -> scalar objective``

    without overclaiming a production custom VJP for the full
    :func:`vmec_jax.driver.run_free_boundary` control loop.  The returned
    ``solved`` dictionary contains the same state, vacuum, response, and
    objective-component diagnostics as
    :func:`direct_coil_projected_mode_fixed_point_objective_jax`.
    """

    def objective(coil_params):
        solved = direct_coil_projected_mode_fixed_point_objective_jax(
            coil_params,
            initial_state,
            **objective_kwargs,
        )
        return solved["objective"]

    check = pytree_directional_derivative_check_jax(
        objective,
        params,
        direction,
        eps=eps,
    )
    solved = direct_coil_projected_mode_fixed_point_objective_jax(
        params,
        initial_state,
        **objective_kwargs,
    )
    return {
        **check,
        "solved": solved,
        "objective_components": solved["objective_components"],
    }


def _weighted_half_norm(value: Any, weight: Any) -> Any:
    """Return ``0.5 * sum(weight * value**2)`` with scalar/array weights."""

    arr = jnp.asarray(value)
    w = jnp.asarray(weight, dtype=arr.dtype)
    return 0.5 * jnp.sum(w * arr * arr)


def _static_weight_is_zero(weight: Any) -> bool:
    """Return true only for host-known scalar/array weights that are exactly zero."""

    try:
        arr = np.asarray(weight)
    except Exception:
        return False
    if arr.size == 0:
        return False
    try:
        return bool(np.all(arr == 0.0))
    except Exception:
        return False


def _tree_weighted_half_norm(values: Any, weight: Any) -> Any:
    """Return the sum of weighted half-norms over numeric pytree leaves."""

    leaves = tree_util.tree_leaves(values)
    if not leaves:
        return jnp.asarray(0.0)
    total = jnp.asarray(0.0)
    for leaf in leaves:
        if leaf is None:
            continue
        try:
            total = total + _weighted_half_norm(leaf, weight)
        except TypeError:
            continue
    return total
