"""Native spline-vector residual microprofile for square-coil decks."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from examples.toroidal_stellarator_mirror_hybrid_square_coils_free_boundary import (
    ExampleConfig,
    make_free_boundary_indata,
)
from vmec_jax.boundary import boundary_from_indata
from vmec_jax.config import config_from_indata
from vmec_jax.init_guess import initial_guess_from_boundary
from vmec_jax.solvers.free_boundary import (
    FreeBoundaryNativeSplineResidualProblem,
    free_boundary_native_spline_matrix_free_normal_step_jax,
    free_boundary_native_spline_unknown_vector_from_vmec_state,
    free_boundary_native_spline_vector_projected_residual_jax,
    free_boundary_native_spline_vector_to_vmec_state_jax,
)
from vmec_jax.solvers.free_boundary.control import _prepare_freeb_edge_control_projection
from vmec_jax.state import VMECState
from vmec_jax.static import build_static


def _block_until_ready(value: Any) -> Any:
    """Block JAX arrays produced by the microprofile."""

    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
        return value
    if isinstance(value, VMECState):
        for part in (value.Rcos, value.Rsin, value.Zcos, value.Zsin, value.Lcos, value.Lsin):
            _block_until_ready(part)
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            _block_until_ready(item)
    return value


def _time_call(func: Any) -> tuple[Any, float]:
    """Return ``func()`` and elapsed wall time, blocking JAX outputs first."""

    start = time.perf_counter()
    result = func()
    _block_until_ready(result)
    return result, float(time.perf_counter() - start)


def _state_delta(state: VMECState) -> VMECState:
    """Build a deterministic same-shape residual surrogate for profiling."""

    return VMECState(
        layout=state.layout,
        Rcos=0.25 * state.Rcos,
        Rsin=0.20 * state.Rsin,
        Zcos=0.15 * state.Zcos,
        Zsin=0.10 * state.Zsin,
        Lcos=0.05 * state.Lcos,
        Lsin=0.04 * state.Lsin,
    )


def _norms(values: Any) -> dict[str, float]:
    """Return compact finite-vector norms."""

    arr = np.asarray(values, dtype=float).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"l2": 0.0, "linf": 0.0}
    return {
        "l2": float(np.linalg.norm(finite)),
        "linf": float(np.max(np.abs(finite))),
    }


def _json_scalar(value: Any) -> float | int | str | None:
    """Return a compact JSON-safe scalar from an optional JAX/NumPy value."""

    if value is None:
        return None
    try:
        arr = np.asarray(value)
    except Exception:
        return str(value)
    if arr.shape == ():
        item = arr.item()
        if isinstance(item, (bool, int, np.integer)):
            return int(item)
        if isinstance(item, (float, np.floating)):
            return float(item) if np.isfinite(float(item)) else None
        return str(item)
    return str(value)


def _native_matrix_free_profile(
    *,
    vector: Any,
    direction: Any,
    template_state: VMECState,
    projection: dict[str, Any],
) -> dict[str, Any]:
    """Time JVP/VJP/CG pieces for a native spline-vector normal step."""

    import jax

    residual_fn = lambda values: free_boundary_native_spline_vector_projected_residual_jax(
        values,
        template_state,
        projection,
        _state_delta,
        edge_metric="pullback",
    )

    (vjp_residual, vjp_pullback), vjp_s = _time_call(
        lambda: (
            lambda residual_value, vjp_fun: (residual_value, vjp_fun(residual_value)[0])
        )(*jax.vjp(residual_fn, vector))
    )

    def apply_normal_matvec() -> Any:
        _residual_value, jvp_fun = jax.linearize(residual_fn, vector)
        _residual_for_vjp, vjp_fun = jax.vjp(residual_fn, vector)
        return vjp_fun(jvp_fun(direction))[0]

    normal_matvec, normal_matvec_s = _time_call(apply_normal_matvec)

    problem = FreeBoundaryNativeSplineResidualProblem(
        template_state=template_state,
        projection=projection,
        residual_fn=_state_delta,
        edge_metric="pullback",
    )
    damping = 1.0e-10
    linear_tol = 1.0e-10
    linear_maxiter = 8
    step, cg_step_s = _time_call(
        lambda: free_boundary_native_spline_matrix_free_normal_step_jax(
            problem,
            vector,
            damping=damping,
            tol=linear_tol,
            maxiter=linear_maxiter,
        )
    )
    next_residual, next_residual_s = _time_call(lambda: problem.residual(step.next_vector))
    residual_l2 = float(step.residual_l2)
    next_residual_norm = _norms(np.asarray(next_residual, dtype=float))
    reduction = (
        None
        if residual_l2 <= np.finfo(float).tiny
        else float(next_residual_norm["l2"] / residual_l2)
    )

    return {
        "status": "completed",
        "method": "jax.linearize_vjp_cg_damped_normal_equations",
        "strict_target_ftol": 1.0e-12,
        "dense_jacobian_formed": False,
        "damping": damping,
        "linear_tol": linear_tol,
        "linear_maxiter": linear_maxiter,
        "vjp_wall_s": float(vjp_s),
        "normal_matvec_wall_s": float(normal_matvec_s),
        "cg_step_wall_s": float(cg_step_s),
        "post_step_residual_wall_s": float(next_residual_s),
        "vjp_residual_l2": _norms(np.asarray(vjp_residual, dtype=float))["l2"],
        "vjp_pullback_l2": _norms(np.asarray(vjp_pullback, dtype=float))["l2"],
        "normal_matvec_l2": _norms(np.asarray(normal_matvec, dtype=float))["l2"],
        "normal_matvec_linf": _norms(np.asarray(normal_matvec, dtype=float))["linf"],
        "step_l2": float(step.step_l2),
        "residual_l2_before_step": residual_l2,
        "residual_l2_after_step": next_residual_norm["l2"],
        "residual_linf_after_step": next_residual_norm["linf"],
        "residual_reduction_factor": reduction,
        "cg_info": _json_scalar(step.cg_info),
        "recommended_solver_lane": "matrix_free_native_spline_normal_or_adjoint_solve",
        "next_action": "feed_real_vmec_force_residual_into_native_matrix_free_loop",
    }


def native_spline_vector_residual_profile_payload(
    *,
    config: ExampleConfig,
    beta_percent: float,
    edge_control_projection_payload: dict[str, Any] | None,
    edge_control_requested: str,
) -> dict[str, Any]:
    """Measure native-vector residual mechanics on a real deck-shaped state.

    This is a no-solve diagnostic. It uses the selected square-coil deck,
    boundary projection, and initial VMEC state to time native encode, decode,
    projected residual packing, and a forward-mode JVP through that projected
    residual. The residual itself is a deterministic same-shape surrogate so
    this function stays fast and does not claim equilibrium convergence.
    """

    if edge_control_projection_payload is None:
        return {
            "status": "blocked",
            "reason": "edge_control_projection_not_enabled",
            "equilibrium_solve_performed": False,
            "next_action": "rerun_with_freeb_edge_control_projection",
        }

    try:
        import jax
        import jax.numpy as jnp
    except Exception as exc:  # pragma: no cover - JAX is expected in this repo.
        return {
            "status": "blocked",
            "reason": "jax_unavailable",
            "error": repr(exc),
            "equilibrium_solve_performed": False,
        }

    indata = make_free_boundary_indata(config, beta_percent=float(beta_percent))
    vmec_config = config_from_indata(indata)
    static = build_static(vmec_config)
    boundary = boundary_from_indata(indata, static.modes)
    state = initial_guess_from_boundary(static, boundary, indata, vmec_project=True)
    projection = _prepare_freeb_edge_control_projection(
        edge_control_projection_payload,
        indata=indata,
        static=static,
        state0=state,
        free_boundary_enabled=True,
    )
    info = dict(projection.get("info", {}))
    if not bool(projection.get("enabled", False)):
        return {
            "status": "blocked",
            "reason": info.get("reason", "projection_disabled"),
            "edge_control_projection": info,
            "equilibrium_solve_performed": False,
            "next_action": "repair_edge_control_projection_payload",
        }

    unknowns, encode_s = _time_call(
        lambda: free_boundary_native_spline_unknown_vector_from_vmec_state(state, projection)
    )
    vector = jnp.asarray(unknowns.vector)
    direction = jnp.linspace(
        1.0,
        2.0,
        int(unknowns.native_unknown_size),
        dtype=vector.dtype,
    )

    decoded, decode_s = _time_call(
        lambda: free_boundary_native_spline_vector_to_vmec_state_jax(
            vector,
            state,
            projection,
        )
    )
    decoded_host = unknowns.to_vmec_state()
    decode_delta = np.asarray(
        decoded_host.layout.pack(
            np.asarray(decoded.Rcos) - np.asarray(decoded_host.Rcos),
            np.asarray(decoded.Rsin) - np.asarray(decoded_host.Rsin),
            np.asarray(decoded.Zcos) - np.asarray(decoded_host.Zcos),
            np.asarray(decoded.Zsin) - np.asarray(decoded_host.Zsin),
            np.asarray(decoded.Lcos) - np.asarray(decoded_host.Lcos),
            np.asarray(decoded.Lsin) - np.asarray(decoded_host.Lsin),
        )
    )
    residual_host = _state_delta(decoded_host)
    host_pullback = unknowns.vector_from_delta_tuple(residual_host, edge_metric="pullback")

    projected, projected_s = _time_call(
        lambda: free_boundary_native_spline_vector_projected_residual_jax(
            vector,
            state,
            projection,
            _state_delta,
            edge_metric="pullback",
        )
    )
    projected_np = np.asarray(projected, dtype=float)
    residual_delta = projected_np - np.asarray(host_pullback, dtype=float)
    residual_norm = _norms(projected_np)
    residual_delta_norm = _norms(residual_delta)
    residual_rel = (
        None
        if residual_norm["l2"] <= np.finfo(float).tiny
        else float(residual_delta_norm["l2"] / residual_norm["l2"])
    )

    (_jvp_value, jvp), jvp_s = _time_call(
        lambda: jax.jvp(
            lambda values: free_boundary_native_spline_vector_projected_residual_jax(
                values,
                state,
                projection,
                _state_delta,
                edge_metric="pullback",
            ),
            (vector,),
            (direction,),
        )
    )
    jvp_norm = _norms(np.asarray(jvp, dtype=float))
    matrix_free_profile = _native_matrix_free_profile(
        vector=vector,
        direction=direction,
        template_state=state,
        projection=projection,
    )
    full_vmec_size = int(unknowns.full_vmec_size)
    native_unknown_size = int(unknowns.native_unknown_size)
    bytes_per_float = int(np.asarray(unknowns.vector).dtype.itemsize)

    return {
        "status": "completed",
        "equilibrium_solve_performed": False,
        "residual_surrogate": "linear_same_shape_state_delta",
        "edge_control_projection_requested": str(edge_control_requested),
        "edge_control_projection": info,
        "native_state_schema": "FreeBoundaryNativeSplineUnknownVector.v1",
        "native_unknown_size": native_unknown_size,
        "full_vmec_size": full_vmec_size,
        "interior_unknown_size": int(unknowns.interior_size),
        "edge_control_size": int(unknowns.edge_control_size),
        "removed_fourier_edge_dofs": int(unknowns.removed_fourier_edge_dofs),
        "unknown_reduction_fraction": float(native_unknown_size / full_vmec_size),
        "native_vector_bytes": int(native_unknown_size * bytes_per_float),
        "full_vmec_vector_bytes": int(full_vmec_size * bytes_per_float),
        "encode_wall_s": float(encode_s),
        "decode_wall_s": float(decode_s),
        "projected_residual_wall_s": float(projected_s),
        "projected_residual_jvp_wall_s": float(jvp_s),
        "decode_parity_linf": _norms(decode_delta)["linf"],
        "projected_residual_l2": residual_norm["l2"],
        "projected_residual_linf": residual_norm["linf"],
        "projected_residual_host_parity_l2": residual_delta_norm["l2"],
        "projected_residual_host_parity_linf": residual_delta_norm["linf"],
        "projected_residual_host_parity_rel": residual_rel,
        "jvp_l2": jvp_norm["l2"],
        "jvp_linf": jvp_norm["linf"],
        "autodiff_method_profiled": "jax.jvp_forward_mode",
        "matrix_free_normal_step_profile": matrix_free_profile,
        "next_action": "promote_packed_native_vector_into_opt_in_solver_loop",
    }
