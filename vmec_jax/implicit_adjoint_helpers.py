"""Small helper seams for implicit-adjoint linear algebra.

The routines here are intentionally VMEC-state agnostic where possible.  They
make the fixed-boundary residual backward pass easier to test with synthetic
linear maps without changing the numerical solve performed by ``implicit.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ._compat import jax, jnp


@dataclass(frozen=True)
class ActiveAdjointModeSelection:
    """Branch decision for the active-coordinate residual adjoint solve."""

    requested_mode: str
    active_is_square: bool
    use_chunked_active: bool
    use_lineax_active: bool
    use_direct_stellsym: bool

    @property
    def falls_back_to_cg(self) -> bool:
        return not (self.use_chunked_active or self.use_lineax_active or self.use_direct_stellsym)


def normalize_residual_adjoint_mode(mode: Any) -> str:
    """Normalize user-provided residual-adjoint mode strings."""
    return str(mode).strip().lower()


def select_active_adjoint_mode(mode: Any, *, active_is_square: bool) -> ActiveAdjointModeSelection:
    """Return the fixed-boundary active-branch solver flags used by ``implicit.py``."""
    requested_mode = normalize_residual_adjoint_mode(mode)
    return ActiveAdjointModeSelection(
        requested_mode=requested_mode,
        active_is_square=bool(active_is_square),
        use_chunked_active=requested_mode in ("chunked", "dense"),
        use_lineax_active=requested_mode == "lineax" and bool(active_is_square),
        use_direct_stellsym=requested_mode in ("direct", "bicgstab") and bool(active_is_square),
    )


def select_active_packing_strategy(*, keep_all_active: bool) -> str:
    """Select the stellarator-symmetric active coordinate packing strategy."""
    return "full" if bool(keep_all_active) else "reduced"


def full_active_keep_indices(full_vector_or_size: Any, *, dtype: Any = jnp.int32):
    """Return keep indices for the non-reduced active packing path."""
    if isinstance(full_vector_or_size, (int, np.integer)):
        size = int(full_vector_or_size)
    else:
        shape = np.shape(full_vector_or_size)
        if len(shape) != 1:
            raise ValueError(f"full active vector must be one-dimensional, got shape {shape}")
        size = int(shape[0])
    if size < 0:
        raise ValueError(f"full active vector size must be non-negative, got {size}")
    return jnp.arange(size, dtype=dtype)


def default_jac_chunk_size(x_active_star: Any, configured_chunk_size: Any) -> int:
    """Resolve the dense Jacobian chunk size while preserving legacy defaults."""
    if configured_chunk_size is None:
        shape = np.shape(x_active_star)
        if len(shape) != 1:
            raise ValueError(f"x_active_star must be one-dimensional, got shape {shape}")
        return min(int(shape[0]), 64)
    chunk_size = int(configured_chunk_size)
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    return chunk_size


def active_residual_is_square(residual_star_active: Any, b_active: Any) -> bool:
    """Match the legacy active square check exactly."""
    return tuple(np.shape(residual_star_active)) == tuple(np.shape(b_active))


def validate_active_adjoint_shapes(residual_star_active: Any, b_active: Any, x_active_star: Any) -> bool:
    """Validate active vectors and return whether the residual map is square."""
    b_shape = tuple(np.shape(b_active))
    x_shape = tuple(np.shape(x_active_star))
    residual_shape = tuple(np.shape(residual_star_active))
    if len(b_shape) != 1:
        raise ValueError(f"b_active must be one-dimensional, got shape {b_shape}")
    if len(x_shape) != 1:
        raise ValueError(f"x_active_star must be one-dimensional, got shape {x_shape}")
    if int(np.prod(residual_shape, dtype=np.int64)) <= 0:
        raise ValueError(f"residual_star_active must be non-empty, got shape {residual_shape}")
    return active_residual_is_square(residual_star_active, b_active)


def validate_full_adjoint_shapes(residual_star: Any, b: Any) -> None:
    """Validate full-coordinate cotangent vectors before matrix-free CG routing."""
    b_shape = tuple(np.shape(b))
    residual_shape = tuple(np.shape(residual_star))
    if len(b_shape) != 1:
        raise ValueError(f"full cotangent vector must be one-dimensional, got shape {b_shape}")
    if int(np.prod(residual_shape, dtype=np.int64)) <= 0:
        raise ValueError(f"full residual must be non-empty, got shape {residual_shape}")


def first_transpose_result(result: Any) -> Any:
    """Extract the primal transpose result from JAX linear-transpose outputs."""
    if isinstance(result, tuple):
        if not result:
            raise ValueError("linear transpose returned an empty tuple")
        return result[0]
    return result


def make_damped_transpose_map(residual_vjp: Callable[[Any], Any], *, damping: Any) -> Callable[[Any], Any]:
    """Wrap a VJP/linear-transpose map as ``J^T v + damping * v``."""

    def matvec(v):
        v_arr = jnp.asarray(v)
        return first_transpose_result(residual_vjp(v_arr)) + jnp.asarray(damping, dtype=v_arr.dtype) * v_arr

    return matvec


def make_active_normal_map(
    residual_jvp_active: Callable[[Any], Any],
    residual_vjp_active: Callable[[Any], Any],
    *,
    damping: Any,
) -> Callable[[Any], Any]:
    """Build ``(J J^T + damping I) lam`` for active-coordinate least squares."""

    def matvec(lam):
        lam_arr = jnp.asarray(lam)
        jt_lam = first_transpose_result(residual_vjp_active(lam_arr))
        j_jt_lam = residual_jvp_active(jt_lam)
        return j_jt_lam + jnp.asarray(damping, dtype=lam_arr.dtype) * lam_arr

    return matvec


def active_normal_rhs(residual_jvp_active: Callable[[Any], Any], b_active: Any) -> Any:
    """Build the active least-squares right-hand side ``J b``."""
    return residual_jvp_active(b_active)


def make_full_normal_map(
    residual_jvp: Callable[[Any], Any],
    residual_vjp: Callable[[Any], Any],
    *,
    unpack_state: Callable[[Any, Any], Any],
    pack_state: Callable[[Any], Any],
    project_state: Callable[[Any], Any],
    layout: Any,
    damping: Any,
) -> Callable[[Any], Any]:
    """Build the full-state matrix-free normal map used by the residual adjoint."""

    def matvec(u_flat):
        u_state = project_state(unpack_state(u_flat, layout))
        jv = residual_jvp(u_state)
        jt_jv = first_transpose_result(residual_vjp(jv))
        jt_jv = project_state(jt_jv)
        return pack_state(jt_jv) + jnp.asarray(damping, dtype=jnp.asarray(jv).dtype) * u_flat

    return matvec


def dense_adjoint_from_jacobian(
    J_active: Any,
    b_active: Any,
    *,
    damping: Any,
    mode: Any,
    dense_transpose_lstsq_host: Callable[[Any, Any, Any], Any],
    is_traced: Callable[..., bool],
):
    """Solve the active dense/chunked adjoint system from an explicit Jacobian."""
    mode = normalize_residual_adjoint_mode(mode)
    J_active = jnp.asarray(J_active)
    b_active = jnp.asarray(b_active)
    damping = jnp.asarray(damping, dtype=J_active.dtype)
    if mode == "dense":
        if is_traced(J_active, b_active, damping):
            out_shape = jax.ShapeDtypeStruct((int(J_active.shape[0]),), J_active.dtype)
            return jax.pure_callback(
                dense_transpose_lstsq_host,
                out_shape,
                J_active,
                b_active,
                damping,
            )
        return jnp.asarray(
            dense_transpose_lstsq_host(J_active, b_active, damping),
            dtype=J_active.dtype,
        )

    H_active = J_active @ J_active.T
    H_active = H_active + damping * jnp.eye(int(H_active.shape[0]), dtype=H_active.dtype)
    rhs_active = J_active @ b_active
    return jnp.linalg.solve(H_active, rhs_active)
