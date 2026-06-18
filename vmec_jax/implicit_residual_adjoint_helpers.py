"""Residual-adjoint routing helpers for implicit fixed-boundary solves.

These helpers keep branchy linear-solver selection testable without needing to
construct a full VMEC state.  They operate only on supplied linear maps.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable

import numpy as np

from ._compat import jax, jnp
from .implicit_adjoint_helpers import (
    active_normal_rhs,
    default_jac_chunk_size,
    dense_adjoint_from_jacobian,
    make_active_normal_map,
    make_damped_transpose_map,
    select_active_adjoint_mode,
    validate_active_adjoint_shapes,
)


@dataclass(frozen=True)
class ActiveResidualAdjointSolveResult:
    """Result and route metadata for an active residual-adjoint solve."""

    lam: Any
    route: str
    info: Any = None


@dataclass(frozen=True)
class FullResidualAdjointSolveResult:
    """Result and route metadata for a full-state residual-adjoint solve."""

    lam: Any
    route: str = "cg"
    state_update: Any | None = None


@dataclass(frozen=True)
class ActiveResidualTangentSolveResult:
    """Tangent update and route metadata for an active residual solve."""

    dx: Any
    route: str
    info: Any = None


def lineax_bicgstab_solve(
    matvec: Callable[[Any], Any],
    b: Any,
    *,
    x0: Any | None = None,
    tol: float,
    max_iter: int,
    lineax_module: Any = None,
    jax_module: Any = jax,
):
    """Solve a square linear system with lineax when an implementation is available."""
    lx = lineax_module
    if lx is None:
        return None, False, {}

    b = jnp.asarray(b)
    input_structure = jax_module.ShapeDtypeStruct(tuple(b.shape), b.dtype)
    operator = lx.FunctionLinearOperator(matvec, input_structure)
    options = {}
    if x0 is not None:
        options["y0"] = jnp.asarray(x0)
    solution = lx.linear_solve(
        operator,
        b,
        solver=lx.BiCGStab(rtol=float(tol), atol=0.0, max_steps=int(max_iter)),
        options=options,
        throw=False,
    )
    value = jnp.asarray(solution.value)
    try:
        success = bool(np.all(np.isfinite(np.asarray(jax_module.device_get(value)))))
    except Exception:
        success = False
    return value, success, getattr(solution, "stats", {})


def linear_map_jacobian_columns(
    linear_map: Callable[[Any], Any],
    *,
    input_size: int,
    output_size: int,
    dtype: Any,
    chunk_size: int,
):
    """Build a dense Jacobian by batching linear-map columns in chunks."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    eye_idx = jnp.arange(int(input_size), dtype=jnp.int32)
    chunks = []
    for start in range(0, int(input_size), int(chunk_size)):
        stop = min(start + int(chunk_size), int(input_size))
        rows = jnp.arange(start, stop, dtype=jnp.int32)
        basis = (rows[:, None] == eye_idx[None, :]).astype(dtype)
        chunk = jax.vmap(linear_map)(basis).T
        chunk = jnp.reshape(chunk, (int(output_size), stop - start))
        chunks.append(chunk)
    return jnp.concatenate(chunks, axis=1)


def solve_active_residual_tangent_linearized(
    stationarity_jvp_active: Callable[[Any], Any],
    stationarity_vjp_active: Callable[[Any], Any],
    *,
    stationarity_star_active: Any,
    x_active_star: Any,
    rhs: Any,
    tangent_mode: str,
    damping: float,
    cg_tol: float,
    cg_max_iter: int,
    jac_chunk_size: Any,
    cg_solve: Callable[..., Any],
    bicgstab_solve: Callable[..., tuple[Any, Any]] | None = None,
    lineax_solve: Callable[..., tuple[Any, bool, Any]] | None = None,
    jacobian_columns: Callable[..., Any] = linear_map_jacobian_columns,
) -> ActiveResidualTangentSolveResult:
    """Solve the active tangent equation used by the residual custom JVP."""
    tangent_mode = str(tangent_mode).strip().lower()
    damping_arr = jnp.asarray(float(damping), dtype=jnp.asarray(x_active_star).dtype)
    active_is_square = tuple(stationarity_star_active.shape) == tuple(x_active_star.shape)

    def stationarity_jvp_active_damped(u_active):
        return stationarity_jvp_active(u_active) + damping_arr * u_active

    if active_is_square and tangent_mode in ("auto", "lineax") and lineax_solve is not None:
        dx_lineax, success, stats = lineax_solve(
            stationarity_jvp_active_damped,
            rhs,
            tol=float(cg_tol),
            max_iter=int(cg_max_iter),
        )
        if bool(success):
            return ActiveResidualTangentSolveResult(dx=dx_lineax, route="lineax", info=stats)

    if active_is_square and tangent_mode in ("direct", "bicgstab") and bicgstab_solve is not None:
        dx_bicgstab, info = bicgstab_solve(
            stationarity_jvp_active_damped,
            rhs,
            tol=float(cg_tol),
            atol=0.0,
            maxiter=int(cg_max_iter),
        )
        if info is None:
            return ActiveResidualTangentSolveResult(dx=dx_bicgstab, route="bicgstab", info=info)

    if tangent_mode == "chunked":
        chunk_size = default_jac_chunk_size(x_active_star, jac_chunk_size)
        J_active = jacobian_columns(
            stationarity_jvp_active,
            input_size=int(x_active_star.shape[0]),
            output_size=int(np.prod(np.shape(stationarity_star_active))),
            dtype=jnp.asarray(x_active_star).dtype,
            chunk_size=int(chunk_size),
        )
        eye = jnp.eye(int(J_active.shape[1]), dtype=J_active.dtype)
        if active_is_square:
            dx_active = jnp.linalg.solve(
                J_active + damping_arr * eye,
                rhs,
            )
        else:
            dx_active = jnp.linalg.solve(
                J_active.T @ J_active + damping_arr * eye,
                J_active.T @ rhs,
            )
        return ActiveResidualTangentSolveResult(dx=dx_active, route="dense")

    Hvp_active = make_active_normal_map(
        stationarity_jvp_active,
        stationarity_vjp_active,
        damping=float(damping),
    )
    rhs_normal = stationarity_vjp_active(rhs)[0]
    dx_active = cg_solve(
        Hvp_active,
        rhs_normal,
        tol=float(cg_tol),
        max_iter=int(cg_max_iter),
    )
    return ActiveResidualTangentSolveResult(dx=dx_active, route="cg")


def solve_active_residual_adjoint_linearized(
    residual_jvp_active: Callable[[Any], Any],
    residual_vjp_active: Callable[[Any], Any],
    *,
    residual_star_active: Any,
    b_active: Any,
    x_active_star: Any,
    residual_adjoint_mode: Any,
    damping: float,
    cg_tol: float,
    cg_max_iter: int,
    jac_chunk_size: Any,
    dense_transpose_lstsq_host: Callable[[Any, Any, Any], Any],
    is_traced: Callable[..., bool],
    cg_solve: Callable[..., Any],
    bicgstab_solve: Callable[..., tuple[Any, Any]] | None = None,
    lineax_solve: Callable[..., tuple[Any, bool, Any]] | None = None,
    jacobian_columns: Callable[..., Any] = linear_map_jacobian_columns,
    profile_log: Callable[..., None] | None = None,
    time_module: Any = time,
) -> ActiveResidualAdjointSolveResult:
    """Route the active residual adjoint solve across dense/direct/fallback paths."""

    def _start():
        return time_module.perf_counter() if profile_log is not None else None

    def _log(stage: str, start: float | None = None, **extra) -> None:
        if profile_log is not None:
            profile_log(stage, start, **extra)

    active_is_square = validate_active_adjoint_shapes(residual_star_active, b_active, x_active_star)
    active_mode = select_active_adjoint_mode(
        residual_adjoint_mode,
        active_is_square=active_is_square,
    )

    if active_mode.use_chunked_active:
        chunk_size = default_jac_chunk_size(x_active_star, jac_chunk_size)
        dense_start = _start()
        J_active = jacobian_columns(
            residual_jvp_active,
            input_size=int(x_active_star.shape[0]),
            output_size=int(np.prod(np.shape(residual_star_active))),
            dtype=jnp.asarray(x_active_star).dtype,
            chunk_size=chunk_size,
        )
        _log(
            "active_dense_jacobian_done",
            dense_start,
            chunk_size=chunk_size,
            jac_shape=tuple(int(x) for x in J_active.shape),
        )
        solve_start = _start()
        lam = dense_adjoint_from_jacobian(
            J_active,
            b_active,
            damping=jnp.asarray(float(damping), dtype=J_active.dtype),
            mode=residual_adjoint_mode,
            dense_transpose_lstsq_host=dense_transpose_lstsq_host,
            is_traced=is_traced,
        )
        _log("active_dense_solve_done", solve_start)
        return ActiveResidualAdjointSolveResult(lam=lam, route="dense")

    if active_mode.use_direct_stellsym and bicgstab_solve is not None:
        JT_active = make_damped_transpose_map(
            residual_vjp_active,
            damping=float(damping),
        )
        direct_solve_start = _start()
        lam, info = bicgstab_solve(
            JT_active,
            b_active,
            tol=float(cg_tol),
            atol=0.0,
            maxiter=int(cg_max_iter),
        )
        _log("direct_bicgstab_done", direct_solve_start, info=str(info))
        if info is None:
            return ActiveResidualAdjointSolveResult(lam=lam, route="bicgstab", info=info)

    if active_mode.use_lineax_active and lineax_solve is not None:
        direct_solve_start = _start()
        JT_active_lineax = make_damped_transpose_map(
            residual_vjp_active,
            damping=float(damping),
        )
        lam, success, stats = lineax_solve(
            JT_active_lineax,
            b_active,
            tol=float(cg_tol),
            max_iter=int(cg_max_iter),
        )
        num_steps = stats.get("num_steps") if isinstance(stats, dict) else None
        if num_steps is not None:
            try:
                num_steps = int(np.asarray(jax.device_get(num_steps)))
            except Exception:
                num_steps = None
        _log(
            "direct_lineax_done",
            direct_solve_start,
            success=bool(success),
            num_steps=num_steps,
        )
        if bool(success):
            return ActiveResidualAdjointSolveResult(lam=lam, route="lineax", info=stats)

    Hvp_active = make_active_normal_map(
        residual_jvp_active,
        residual_vjp_active,
        damping=float(damping),
    )
    rhs_active = active_normal_rhs(residual_jvp_active, b_active)
    cg_start = _start()
    lam = cg_solve(Hvp_active, rhs_active, tol=float(cg_tol), max_iter=int(cg_max_iter))
    _log("active_cg_done", cg_start)
    return ActiveResidualAdjointSolveResult(lam=lam, route="cg")


def solve_full_residual_adjoint_linearized(
    residual_jvp: Callable[[Any], Any],
    residual_vjp: Callable[[Any], Any],
    *,
    residual_star: Any,
    b: Any,
    st_star: Any,
    damping: float,
    cg_tol: float,
    cg_max_iter: int,
    cg_solve: Callable[..., Any],
    unpack_state: Callable[..., Any],
    pack_state: Callable[..., Any],
    project_state: Callable[[Any], Any],
    make_full_normal_map_func: Callable[..., Any],
    validate_full_shapes: Callable[..., None],
    profile_log: Callable[..., None] | None = None,
    time_module: Any = time,
) -> FullResidualAdjointSolveResult:
    """Solve the full-state residual adjoint with the matrix-free normal map."""

    def _start():
        return time_module.perf_counter() if profile_log is not None else None

    def _log(stage: str, start: float | None = None, **extra) -> None:
        if profile_log is not None:
            profile_log(stage, start, **extra)

    validate_full_shapes(residual_star, b)
    Hvp = make_full_normal_map_func(
        residual_jvp,
        residual_vjp,
        unpack_state=unpack_state,
        pack_state=pack_state,
        project_state=project_state,
        layout=st_star.layout,
        damping=float(damping),
    )

    cg_start = _start()
    u = cg_solve(Hvp, b, tol=float(cg_tol), max_iter=int(cg_max_iter))
    _log("full_cg_done", cg_start)
    u_state = project_state(unpack_state(u, st_star.layout))
    jvp_start = _start()
    lam = residual_jvp(u_state)
    _log("full_jvp_done", jvp_start)
    return FullResidualAdjointSolveResult(lam=lam, route="cg", state_update=u_state)
