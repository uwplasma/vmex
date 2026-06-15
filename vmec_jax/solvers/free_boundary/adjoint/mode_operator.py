"""JAX VMEC/NESTOR mode-source and matrix-free operator helpers."""

from __future__ import annotations

from typing import Any

from vmec_jax._compat import jax, jnp


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
    """JAX version of VMEC/NESTOR source symmetrization."""

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
    """Project a VMEC/NESTOR grid source into mode-space RHS coefficients."""

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
    """Build the VMEC/NESTOR mode matrix from Green-function mode samples."""

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
    """Apply the VMEC/NESTOR mode operator without materializing it."""

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
    """Solve the mode response through a matrix-free mode operator."""

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


__all__ = [
    "mode_matrix_from_grpmn_jax",
    "mode_matrix_matvec_from_grpmn_jax",
    "mode_operator_vacuum_solve_jax",
    "mode_rhs_from_gsource_jax",
    "vmec_source_from_gsource_jax",
]
