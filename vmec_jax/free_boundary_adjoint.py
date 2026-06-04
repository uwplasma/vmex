"""Adjoint scaffolding for free-boundary vacuum solves.

Phase 1 intentionally keeps this module small and explicit.  It validates the
linear-solve differentiation contract that the production NESTOR replacement
will need: solve the primal system in the forward pass and use transpose solves
in the backward pass rather than differentiating through an iterative solver.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from vmec_jax._compat import jax, jnp, tree_util

from .free_boundary_adjoint_controller import (
    jax_visible_accepted_nonlinear_controller_directional_check_jax,
    jax_visible_accepted_nonlinear_controller_jax,
    jax_visible_masked_nonlinear_controller_directional_check_jax,
    jax_visible_masked_nonlinear_controller_jax,
    jax_visible_nonlinear_controller_directional_check_jax,
    jax_visible_nonlinear_controller_jax,
    jax_visible_segmented_accepted_nonlinear_controller_jax,
    pytree_directional_derivative_check_jax,
)

__all__ = [
    "direct_coil_accepted_trace_branch_metadata",
    "direct_coil_accepted_trace_controller_custom_vjp_scalars_jax",
    "direct_coil_same_branch_replay_gate_report",
    "direct_coil_same_branch_controller_scalars_custom_vjp_report",
    "free_boundary_adjoint_trace_replay_diagnostics",
    "jax_visible_accepted_nonlinear_controller_directional_check_jax",
    "jax_visible_accepted_nonlinear_controller_jax",
    "jax_visible_masked_nonlinear_controller_directional_check_jax",
    "jax_visible_masked_nonlinear_controller_jax",
    "jax_visible_nonlinear_controller_directional_check_jax",
    "jax_visible_nonlinear_controller_jax",
    "jax_visible_segmented_accepted_nonlinear_controller_jax",
    "pytree_directional_derivative_check_jax",
]


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

    for ip in range(nuv3):
        xip = rcosuv[ip]
        yip = rsinuv[ip]
        ivoff = int(nuv_full - ip)
        iskip = int(ip // max(1, nv))
        iuoff = int(nuv_full - nv * iskip)
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
                ivoff_k = ivoff + (2 * nu * kp if nv == 1 else 0)
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

        gstore = gstore + bex[ip] * delgr
        del_iuv = delgrp[iuv_grid]
        del_ref = delgrp[iref_grid]
        ka_grid = del_iuv - del_ref
        g1_sym = jnp.einsum("uv,fv->uf", ka_grid, cosv_modes)
        g2_sym = jnp.einsum("uv,fv->uf", ka_grid, sinv_modes)

        gcos = jnp.einsum("mu,uf->mf", sinm_sym, g1_sym)
        gsin = jnp.einsum("mu,uf->mf", cosm_sym, g2_sym)
        total_plus = jnp.reshape(gcos + gsin, (-1,))
        total_minus = jnp.reshape(gcos - gsin, (-1,))
        grpmn = grpmn.at[idx_p_flat, ip].add(total_plus)
        grpmn = grpmn.at[idx_m_negative, ip].add(total_minus[negative_positions_arr])

        if lasym:
            ks_grid = del_iuv + del_ref
            g1_asym = jnp.einsum("uv,fv->uf", ks_grid, cosv_modes)
            g2_asym = jnp.einsum("uv,fv->uf", ks_grid, sinv_modes)
            gcos_asym = jnp.einsum("mu,uf->mf", sinm_asym, g1_asym)
            gsin_asym = jnp.einsum("mu,uf->mf", cosm_asym, g2_asym)
            total_plus_asym = jnp.reshape(gcos_asym + gsin_asym, (-1,))
            total_minus_asym = jnp.reshape(gcos_asym - gsin_asym, (-1,))
            row_off = int(mnpd)
            grpmn = grpmn.at[row_off + idx_p_flat, ip].add(total_plus_asym)
            grpmn = grpmn.at[row_off + idx_m_negative, ip].add(total_minus_asym[negative_positions_arr])

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
        phi_flat = sin @ coeffs
    else:
        cos = jnp.asarray(cos_basis)
        if cos.shape != sin.shape:
            raise ValueError("cos_basis must match sin_basis shape")
        nmodes = int(sin.shape[1])
        if coeffs.shape[0] != 2 * nmodes:
            raise ValueError("doubled rhs/mode_matrix size must be 2 * sin_basis columns")
        phi_flat = sin @ coeffs[:nmodes] + cos @ coeffs[nmodes:]

    return {
        "mode_coeffs": coeffs,
        "phi_flat": phi_flat,
        "residual": dense_vacuum_residual(A, coeffs, rhs),
    }


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
    )
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
) -> dict[str, Any]:
    """JAX version of the VMEC boundary-field projection scaffold.

    This mirrors ``free_boundary.vacuum_boundary_fields_from_cylindrical`` for
    derivative tests.  It intentionally returns a plain dict rather than the
    NumPy dataclass used by the production bridge, so it can be transformed by
    ``jax.grad``/``jax.jacfwd`` while the full NESTOR path is still being
    ported.
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
    det_safe = jnp.where(
        jnp.abs(det) >= float(det_floor),
        det,
        jnp.sign(det + 1.0e-300) * float(det_floor),
    )

    bu = br_arr * Ru_arr + bz_arr * Zu_arr
    bv = br_arr * Rv_arr + bp_arr * R_arr + bz_arr * Zv_arr
    bsupu = (g_vv * bu - g_uv * bv) / det_safe
    bsupv = (g_uu * bv - g_uv * bu) / det_safe
    bsqvac = 0.5 * (bu * bsupu + bv * bsupv)

    n_r = -R_arr * Zu_arr
    n_phi = Zu_arr * Rv_arr - Ru_arr * Zv_arr
    n_z = R_arr * Ru_arr
    bnormal = br_arr * n_r + bp_arr * n_phi + bz_arr * n_z
    n_norm = jnp.sqrt(n_r * n_r + n_phi * n_phi + n_z * n_z)
    bnormal_unit = bnormal / jnp.where(n_norm > 0.0, n_norm, 1.0)

    return {
        "bu": bu,
        "bv": bv,
        "bsupu": bsupu,
        "bsupv": bsupv,
        "bsqvac": bsqvac,
        "bnormal": bnormal,
        "bnormal_unit": bnormal_unit,
        "g_uu": g_uu,
        "g_uv": g_uv,
        "g_vv": g_vv,
        "det_guv": det,
    }


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


def direct_coil_boundary_replay_context(
    static: Any,
    geometry: dict[str, Any],
) -> dict[str, Any]:
    """Build static NESTOR replay data for an accepted boundary geometry.

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

    R = geometry["R"]
    ntheta, nzeta = (int(v) for v in R.shape)
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
    return {
        "basis": basis,
        "tables": tables,
        "wint": wint,
        "nvper": nvper,
        "ntheta": ntheta,
        "nzeta": nzeta,
    }


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

    from .external_fields import sample_coil_field_cylindrical

    R_j = jnp.asarray(R)
    br, bp, bz = sample_coil_field_cylindrical(
        params,
        R_j,
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
        R=R_j,
        Ru=Ru,
        Zu=Zu,
        Rv=Rv,
        Zv=Zv,
    )
    if wint is None:
        wint_j = jnp.ones_like(R_j)
    else:
        wint_j = jnp.asarray(wint, dtype=jnp.asarray(vac["bnormal"]).dtype)
    bexni = -jnp.asarray(vac["bnormal"]) * wint_j * ((2.0 * jnp.pi) ** 2)
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
    )
    channels = vacuum_boundary_fields_from_mode_coeffs_jax(
        mode_solution["mode_coeffs"],
        basis=basis,
        bu_ext=vac["bu"],
        bv_ext=vac["bv"],
        g_uu=vac["g_uu"],
        g_uv=vac["g_uv"],
        g_vv=vac["g_vv"],
    )
    return {
        "bsqvac": channels["bsqvac"],
        "channels": channels,
        "mode_solution": mode_solution,
        "vac": vac,
        "bexni": bexni,
    }


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
            geometry = free_boundary_boundary_geometry_jax(
                state,
                static,
                sample_nzeta=sample_nzeta,
            )
            context = direct_coil_boundary_replay_context(static, geometry)
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
            )
            freeb_bsqvac_half = replay["bsqvac"]
        else:
            # Full accepted-trace replay must preserve non-vacuum/setup steps.
            # These steps do not have enough NESTOR metadata to resample coils,
            # so replay the original trace payload and keep coil derivatives
            # zero for that step.
            replay = None
            freeb_bsqvac_half = trace.get("freeb_bsqvac_half", None)
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

    The helper intentionally keeps every trace accepted.  It does not
    differentiate through the host policy that selected the traces; it validates
    that a production accepted-trace replay can be represented as a static
    JAX-visible accepted-control scan.
    """

    from .discrete_adjoint import strict_update_one_step_from_trace
    from .state import pack_state

    trace_seq = tuple(traces)
    if max_steps is not None:
        trace_seq = trace_seq[: int(max_steps)]
    if not trace_seq:
        raise ValueError("at least one accepted trace is required")
    if jax is None:  # pragma: no cover - dependency fallback.
        raise RuntimeError("JAX is required for controller replay.")

    controls = direct_coil_accepted_trace_controller_controls_jax(
        trace_seq,
        accept_mask=accept_mask,
        done_mask=done_mask,
    )
    preconditioner_policy_segments = direct_coil_accepted_trace_preconditioner_policy_segments(trace_seq)
    preconditioner_policy_segment_summary = direct_coil_accepted_trace_preconditioner_policy_segment_summary(
        trace_seq,
        accept_mask=accept_mask,
        done_mask=done_mask,
    )
    scalar_controls = direct_coil_accepted_trace_scalar_controls_jax(trace_seq)
    array_controls = direct_coil_accepted_trace_array_controls_jax(trace_seq)
    # These preconditioner policy flags still feed Python bool/int dispatch in
    # the radial preconditioner implementation. Keep them as branch-local
    # static trace data until the full preconditioner path is JAX-visible.
    step_scalar_controls = {
        key: value
        for key, value in scalar_controls.items()
        if key
        not in (
            "preconditioner_use_precomputed_tridi",
            "preconditioner_use_lax_tridi",
        )
    }
    preconditioner_controls = None
    preconditioner_controls_stacked = True
    try:
        preconditioner_controls = direct_coil_accepted_trace_preconditioner_controls_jax(trace_seq)
    except (KeyError, ValueError):
        # Some production accepted traces change the active radial solve size
        # across steps. Those preconditioner matrices cannot be represented as a
        # single scan-stacked pytree without padding, so keep the branch-local
        # trace payload for this rung while still scanning scalar/velocity
        # controls.
        preconditioner_controls_stacked = False
    controls = {**controls, "step_scalars": step_scalar_controls, "step_arrays": array_controls}
    if preconditioner_controls is not None:
        controls = {**controls, "step_preconditioner": preconditioner_controls}

    def _branch_for_trace(trace: dict[str, Any], state: Any, coil_params: Any, control: dict[str, Any]):
        reset_to_trace_pre = jnp.asarray(control["reset_to_trace_pre"], dtype=bool)
        state_in = jax.lax.cond(
            reset_to_trace_pre,
            lambda _: trace["state_pre"],
            lambda _: state,
            operand=None,
        )
        has_active_freeb_replay = trace.get("freeb_bsqvac_half") is not None and trace.get("freeb_nestor_trace") is not None
        if has_active_freeb_replay:
            geometry = free_boundary_boundary_geometry_jax(
                state_in,
                static,
                sample_nzeta=sample_nzeta,
            )
            context = direct_coil_boundary_replay_context(static, geometry)
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
            )
            freeb_bsqvac_half = replay["bsqvac"]
            bsqvac_objective = _weighted_half_norm(replay["bsqvac"], bsqvac_weight)
            bsqvac_rms = jnp.sqrt(jnp.mean(jnp.square(jnp.asarray(replay["bsqvac"]))))
            bnormal_rms = jnp.sqrt(jnp.mean(jnp.square(jnp.asarray(replay["vac"]["bnormal"]))))
        else:
            freeb_bsqvac_half = trace.get("freeb_bsqvac_half", None)
            bsqvac_objective = jnp.asarray(0.0)
            bsqvac_rms = jnp.asarray(0.0)
            bnormal_rms = jnp.asarray(0.0)
        step = strict_update_one_step_from_trace(
            state_in,
            static,
            trace,
            scalar_controls=control["step_scalars"],
            array_controls=control["step_arrays"],
            preconditioner_controls=control["step_preconditioner"] if "step_preconditioner" in control else None,
            freeb_bsqvac_half=freeb_bsqvac_half,
            enforce_edge=bool(enforce_edge),
        )
        return step["step"]["state_post"], {
            "force": _tree_weighted_half_norm(step["force"], force_weight),
            "bsqvac": bsqvac_objective,
            "bsqvac_rms": bsqvac_rms,
            "bnormal_rms": bnormal_rms,
            "state_reset": reset_to_trace_pre,
        }

    def _make_step_fn(segment_traces: tuple[dict[str, Any], ...], *, index_offset: int = 0):
        branches = tuple(
            (
                lambda operand, trace=trace: _branch_for_trace(
                    trace,
                    operand[0],
                    operand[1],
                    operand[2],
                )
            )
            for trace in segment_traces
        )

        def _step_fn(state, coil_params, control):
            step_index = jnp.asarray(control["step_index"], dtype=jnp.int32) - jnp.asarray(index_offset, dtype=jnp.int32)
            do_propose = jnp.asarray(control["accept"], dtype=bool)

            def _propose(_unused):
                return jax.lax.switch(step_index, branches, (state, coil_params, control))

            def _skip(_unused):
                return state, {
                    "force": jnp.asarray(0.0),
                    "bsqvac": jnp.asarray(0.0),
                    "bsqvac_rms": jnp.asarray(0.0),
                    "bnormal_rms": jnp.asarray(0.0),
                    "state_reset": jnp.asarray(False, dtype=bool),
                }

            return jax.lax.cond(do_propose, _propose, _skip, operand=None)

        return _step_fn

    def accept_fn(_state, _proposed_state, _params, control, _aux):
        return control["accept"]

    def converged_fn(_accepted_state, _params, control, _aux):
        return control["done"]

    segment_preconditioner_controls_stacked: tuple[bool, ...] = ()
    if use_preconditioner_policy_segments:
        control_segments_list = []
        segment_preconditioner_controls_stacked_list: list[bool] = []
        for segment in preconditioner_policy_segments:
            start = int(segment["start"])
            stop = int(segment["stop"])
            segment_controls = tree_util.tree_map(
                lambda value, start=start, stop=stop: jnp.asarray(value)[start:stop],
                controls,
            )
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
            control_segments_list.append(segment_controls)
        control_segments = tuple(control_segments_list)
        segment_preconditioner_controls_stacked = tuple(segment_preconditioner_controls_stacked_list)
        step_fns = tuple(
            _make_step_fn(
                trace_seq[int(segment["start"]) : int(segment["stop"])],
                index_offset=int(segment["start"]),
            )
            for segment in preconditioner_policy_segments
        )
        run = jax_visible_segmented_accepted_nonlinear_controller_jax(
            step_fns,
            accept_fn,
            converged_fn,
            initial_state,
            params,
            control_segments,
            checkpoint_steps=checkpoint_steps,
        )
    else:
        run = jax_visible_accepted_nonlinear_controller_jax(
            _make_step_fn(trace_seq),
            accept_fn,
            converged_fn,
            initial_state,
            params,
            controls,
            checkpoint_steps=checkpoint_steps,
        )
    accepted = jnp.asarray(run["history"]["accepted"], dtype=jnp.asarray(pack_state(run["state"])).dtype)
    objective_components = {
        "state": _weighted_half_norm(pack_state(run["state"]), state_weight),
        "force": jnp.sum(accepted * jnp.asarray(run["history"]["force"])),
        "bsqvac": jnp.sum(accepted * jnp.asarray(run["history"]["bsqvac"])),
    }
    objective = sum(objective_components.values())
    return {
        "objective": objective,
        "objective_components": objective_components,
        "state": run["state"],
        "history": run["history"],
        "controls": controls,
        "scalar_controls": scalar_controls,
        "array_controls": array_controls,
        "preconditioner_controls": preconditioner_controls,
        "preconditioner_controls_stacked": bool(preconditioner_controls_stacked),
        "preconditioner_policy_segments": preconditioner_policy_segments,
        "preconditioner_policy_n_segments": len(preconditioner_policy_segments),
        "preconditioner_policy_segment_summary": preconditioner_policy_segment_summary,
        "preconditioner_controls_segment_stacked": segment_preconditioner_controls_stacked,
        "used_preconditioner_policy_segments": bool(use_preconditioner_policy_segments),
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
    adjoint: it rejects branch changes using accepted-trace fingerprints and
    leaves the differentiated frozen-branch replay to the caller.
    """

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
            "same_branch": bool(plus_branch["compatible"] and minus_branch["compatible"]),
            "plus": plus_branch,
            "minus": minus_branch,
            "base_fingerprint": base_fingerprint,
            "plus_fingerprint": plus_fingerprint,
            "minus_fingerprint": minus_fingerprint,
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

    scalar_fns = tuple(
        (lambda replay, fn=fn: fn(replay, base))
        for fn in replay_scalar_fns.values()
    )

    def _controller_scalars(coil_params):
        return direct_coil_accepted_trace_controller_custom_vjp_scalars_jax(
            coil_params,
            traces[0]["state_pre"],
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
    basis_gradients = tuple(pullback(basis[index])[0] for index in range(len(keys)))
    jacobian = tree_util.tree_map(
        lambda *parts: jnp.stack([jnp.asarray(part) for part in parts], axis=0),
        *basis_gradients,
    )
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
        "values": values,
        "jacobian": jacobian,
        "exact_directionals": exact_directionals,
        "frozen_trace_fd_directionals": frozen_fd_directionals,
        "scalar_reports": scalar_reports,
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
