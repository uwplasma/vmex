"""Dense mode-space vacuum solve helpers for free-boundary adjoint gates."""

from __future__ import annotations

from typing import Any

from vmec_jax._compat import jnp

from .dense import dense_vacuum_residual, dense_vacuum_solve_jax


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
    """Solve a dense mode-space vacuum system and reconstruct a grid potential."""

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


__all__ = ["dense_mode_vacuum_solve_jax"]
