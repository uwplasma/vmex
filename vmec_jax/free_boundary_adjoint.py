"""Adjoint scaffolding for free-boundary vacuum solves.

Phase 1 intentionally keeps this module small and explicit.  It validates the
linear-solve differentiation contract that the production NESTOR replacement
will need: solve the primal system in the forward pass and use transpose solves
in the backward pass rather than differentiating through an iterative solver.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vmec_jax._compat import jax, jnp, tree_util


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
