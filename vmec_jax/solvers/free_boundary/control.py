"""Small free-boundary iteration-control helpers for VMEC solves."""

from __future__ import annotations

import os
from typing import Any, Callable, NamedTuple

import numpy as np

from vmec_jax._compat import jnp
from vmec_jax.boundary import boundary_from_indata
from vmec_jax.state import VMECState
from vmec_jax.solvers.free_boundary.reduced_controls import (
    ReducedControlMap,
    ReducedControlState,
    reduced_control_decode,
    reduced_control_least_squares_step,
)

from ..fixed_boundary.residual.update import (
    scale_velocity_blocks,
    zero_velocity_blocks_like,
)


class FreeBoundaryNativeControlStep(NamedTuple):
    """One solver-native reduced edge-control update."""

    state: VMECState
    update_deltas: Any
    control_velocity: np.ndarray
    control_coordinates: np.ndarray
    control_update: np.ndarray
    control_force: np.ndarray
    target_l2: float
    control_force_l2: float
    control_velocity_l2: float
    control_update_l2: float
    trust_scale: float
    force_metric: str


def free_boundary_iter_controls(iter2: int, iter1: int, nvacskip: int) -> tuple[int, int]:
    """Return the reduced legacy ``(ivac, ivacskip)`` free-boundary cadence."""

    nv = max(1, int(nvacskip))
    ivs = int((int(iter2) - int(iter1)) % nv)
    ivac = 1 if ivs == 0 else 2
    return ivac, ivs


def free_boundary_iter_controls_vmec(
    *,
    iter2: int,
    iter1: int,
    ivac: int,
    nvacskip: int,
    nvskip0: int,
    fsq_rz_prev: float,
    activate_fsq: float | None = None,
) -> tuple[int, int, int]:
    """VMEC2000-style ``ivac/ivacskip/nvacskip`` update for ``funct3d`` cadence."""

    i2 = int(iter2)
    i1 = int(iter1)
    iv = int(ivac)
    nv = max(1, int(nvacskip))
    nv0 = max(1, int(nvskip0))
    fs = float(fsq_rz_prev)
    if not np.isfinite(fs) or fs < 0.0:
        fs = 1.0

    if activate_fsq is None:
        activate_threshold = float(os.getenv("VMEC_JAX_FREEB_ACTIVATE_FSQ", "1.0e-3") or 1.0e-3)
    else:
        activate_threshold = float(activate_fsq)

    if i2 > 1 and fs <= activate_threshold:
        iv += 1

    if iv < 0:
        return iv, 0, nv

    ivs = int((i2 - i1) % nv)
    if iv <= 2:
        ivs = 0

    if ivs == 0:
        nv_est = int(1.0 / max(1.0e-1, 1.0e11 * fs))
        nv = max(nv0, max(1, nv_est))

    return iv, ivs, nv


def free_boundary_prev_rz_fsq_next(
    *,
    prev_fsq_before: float,
    fsq_rz_curr: float,
    turnon_restart: bool,
    preserve_turnon_restart: bool,
) -> float:
    """Optionally carry the pre-turn-on residual into the next cadence step."""

    if bool(turnon_restart) and bool(preserve_turnon_restart):
        return float(prev_fsq_before)
    return float(fsq_rz_curr)


def free_boundary_should_damp_constraint_baseline(
    *,
    freeb_ivac: int,
    freeb_turnon_iter: bool,
    lthreed: bool,
) -> bool:
    """Return whether VMEC should damp persistent free-boundary constraint baselines."""

    if not bool(lthreed):
        return int(freeb_ivac) >= 0
    return int(freeb_ivac) >= 0 and (not bool(freeb_turnon_iter))


def free_boundary_turnon_resets_iter1_immediately(*, lthreed: bool, lasym: bool) -> bool:
    """Return whether turn-on should immediately reset ``iter1`` for cadence."""

    return (not bool(lthreed)) or (not bool(lasym))


def _freeb_edge_control_mode_scale(static: Any) -> np.ndarray:
    """Return the VMEC internal-to-physical scale for one LCFS coefficient row."""

    modes = getattr(static, "modes", None)
    m = np.asarray(getattr(modes, "m"), dtype=float)
    n = np.asarray(getattr(modes, "n"), dtype=float)
    sqrt2 = np.sqrt(2.0)
    return np.where(m == 0.0, 1.0, sqrt2) * np.where(np.abs(n) == 0.0, 1.0, sqrt2)


def _prepare_freeb_edge_control_projection(
    payload: Any,
    *,
    indata: Any,
    static: Any,
    state0: VMECState,
    free_boundary_enabled: bool,
) -> dict[str, Any]:
    """Prepare an LCFS projection onto reduced free-boundary controls.

    ``payload`` supplies a dense Jacobian from reduced controls to physical VMEC
    edge-coefficient deltas stacked as ``R_cos, R_sin, Z_cos, Z_sin``.  The
    nonlinear solver stores edge rows with VMEC's internal normalization, so the
    projection converts only that row back and forth.
    """

    info: dict[str, Any] = {
        "enabled": False,
        "requested": bool(payload is not None),
        "reason": "not_requested",
    }
    if payload is None:
        return {"enabled": False, "info": info}
    if not isinstance(payload, dict):
        info.update({"reason": "invalid_payload_type", "payload_type": type(payload).__name__})
        return {"enabled": False, "info": info}
    if not bool(payload.get("enabled", True)):
        info["reason"] = "disabled_by_payload"
        return {"enabled": False, "info": info}
    if not bool(free_boundary_enabled):
        info["reason"] = "not_free_boundary"
        return {"enabled": False, "info": info}

    jacobian_raw = payload.get("control_jacobian", payload.get("jacobian"))
    try:
        jacobian = np.asarray(jacobian_raw, dtype=float)
    except Exception as exc:
        info.update({"reason": "invalid_jacobian", "error": repr(exc)})
        return {"enabled": False, "info": info}

    modes = getattr(static, "modes", None)
    mode_count = int(np.asarray(getattr(modes, "m")).size)
    if jacobian.ndim != 2 or jacobian.shape[0] != 4 * mode_count:
        info.update(
            {
                "reason": "jacobian_shape_mismatch",
                "jacobian_shape": [int(value) for value in getattr(jacobian, "shape", ())],
                "expected_rows": int(4 * mode_count),
            }
        )
        return {"enabled": False, "info": info}
    if jacobian.shape[1] <= 0:
        info.update(
            {
                "reason": "empty_control_basis",
                "jacobian_shape": [int(value) for value in jacobian.shape],
            }
        )
        return {"enabled": False, "info": info}

    mode_scale = _freeb_edge_control_mode_scale(static)
    if mode_scale.shape[0] != mode_count:
        info.update({"reason": "mode_scale_shape_mismatch"})
        return {"enabled": False, "info": info}

    update_mode = str(payload.get("update_mode", payload.get("edge_update_mode", ""))).strip().lower()
    coordinate_mode = update_mode in {
        "coordinate",
        "coordinates",
        "native",
        "native_coordinate",
        "native-coordinate",
        "reduced",
        "reduced_coordinate",
        "reduced-coordinate",
        "solver_native",
        "solver-native",
    }
    initial_source = "state0_edge"
    try:
        initial_payload = payload.get("initial_boundary")
        if isinstance(initial_payload, dict):
            initial = {
                "R_cos": np.asarray(initial_payload["R_cos"], dtype=float).reshape(-1),
                "R_sin": np.asarray(initial_payload["R_sin"], dtype=float).reshape(-1),
                "Z_cos": np.asarray(initial_payload["Z_cos"], dtype=float).reshape(-1),
                "Z_sin": np.asarray(initial_payload["Z_sin"], dtype=float).reshape(-1),
            }
            initial_source = "payload"
        elif indata is not None and not coordinate_mode:
            boundary = boundary_from_indata(indata, static.modes)
            initial = {
                "R_cos": np.asarray(boundary.R_cos, dtype=float).reshape(-1),
                "R_sin": np.asarray(boundary.R_sin, dtype=float).reshape(-1),
                "Z_cos": np.asarray(boundary.Z_cos, dtype=float).reshape(-1),
                "Z_sin": np.asarray(boundary.Z_sin, dtype=float).reshape(-1),
            }
            initial_source = "indata"
        else:
            initial = {
                "R_cos": np.asarray(state0.Rcos, dtype=float)[-1] * mode_scale,
                "R_sin": np.asarray(state0.Rsin, dtype=float)[-1] * mode_scale,
                "Z_cos": np.asarray(state0.Zcos, dtype=float)[-1] * mode_scale,
                "Z_sin": np.asarray(state0.Zsin, dtype=float)[-1] * mode_scale,
            }
    except Exception as exc:
        info.update({"reason": "initial_boundary_failed", "error": repr(exc)})
        return {"enabled": False, "info": info}

    for name, values in initial.items():
        values_arr = np.asarray(values)
        if values_arr.shape != (mode_count,):
            info.update(
                {
                    "reason": "initial_boundary_shape_mismatch",
                    "component": name,
                    "shape": [int(value) for value in values_arr.shape],
                    "expected_shape": [int(mode_count)],
                }
            )
            return {"enabled": False, "info": info}

    try:
        rcond = float(payload.get("rcond", 1.0e-12))
    except Exception:
        rcond = 1.0e-12
    try:
        ridge = float(payload.get("ridge", payload.get("control_ridge", 0.0)))
    except Exception:
        ridge = 0.0
    if not np.isfinite(ridge) or ridge < 0.0:
        info.update({"reason": "invalid_ridge"})
        return {"enabled": False, "info": info}
    trust_radius_raw = payload.get("trust_radius", payload.get("control_trust_radius"))
    try:
        trust_radius = None if trust_radius_raw is None else float(trust_radius_raw)
    except Exception:
        info.update({"reason": "invalid_trust_radius"})
        return {"enabled": False, "info": info}
    if trust_radius is not None and (not np.isfinite(trust_radius) or trust_radius <= 0.0):
        info.update({"reason": "invalid_trust_radius"})
        return {"enabled": False, "info": info}
    force_metric_raw = str(
        payload.get("native_force_metric", payload.get("control_force_metric", "pullback"))
    ).strip().lower()
    if force_metric_raw in {"", "default", "pullback", "adjoint", "gradient", "jtf", "j.t"}:
        native_force_metric = "pullback"
    elif force_metric_raw in {"least_squares", "least-squares", "ls", "coordinate", "pinv", "pseudoinverse"}:
        native_force_metric = "least_squares"
    else:
        info.update({"reason": "invalid_native_force_metric", "native_force_metric": force_metric_raw})
        return {"enabled": False, "info": info}

    try:
        pinv = np.linalg.pinv(jacobian, rcond=rcond)
        if ridge > 0.0:
            lhs = jacobian.T @ jacobian + ridge * np.eye(jacobian.shape[1])
            rhs = jacobian.T
            try:
                control_operator = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                control_operator = np.linalg.pinv(lhs, rcond=rcond) @ rhs
        else:
            control_operator = pinv
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
    except Exception as exc:
        info.update({"reason": "jacobian_factorization_failed", "error": repr(exc)})
        return {"enabled": False, "info": info}

    smax = float(np.max(singular_values)) if singular_values.size else 0.0
    rank = int(np.sum(singular_values > max(float(rcond) * smax, np.finfo(float).eps)))
    condition_number = (
        None
        if singular_values.size == 0 or float(np.min(singular_values)) <= 0.0
        else float(np.max(singular_values) / np.min(singular_values))
    )

    info.update(
        {
            "enabled": True,
            "reason": "enabled",
            "mode": "edge_delta_least_squares",
            "basis_symmetry": payload.get("basis_symmetry", payload.get("symmetry")),
            "labels": [str(value) for value in payload.get("labels", [])],
            "control_count": int(jacobian.shape[1]),
            "mode_count": int(mode_count),
            "jacobian_shape": [int(value) for value in jacobian.shape],
            "rank": int(rank),
            "rcond": float(rcond),
            "ridge": float(ridge),
            "trust_radius": None if trust_radius is None else float(trust_radius),
            "native_force_metric": native_force_metric,
            "singular_values": [float(value) for value in singular_values],
            "condition_number": condition_number,
            "initial_boundary_source": initial_source,
            "coordinate_mode_anchors_state0": bool(coordinate_mode and initial_source == "state0_edge"),
            "native_coordinate_mode": update_mode
            in {"native", "native_coordinate", "native-coordinate", "solver_native", "solver-native"},
        }
    )
    return {
        "enabled": True,
        "info": info,
        "mode_count": int(mode_count),
        "jacobian_np": jacobian,
        "pinv_np": pinv,
        "control_operator_np": np.asarray(control_operator, dtype=float),
        "ridge": float(ridge),
        "trust_radius": trust_radius,
        "native_force_metric": native_force_metric,
        "mode_scale_np": mode_scale,
        "initial_np": initial,
    }


def _freeb_edge_control_trust_radius(projection: dict[str, Any]) -> float | None:
    """Return the optional reduced-control trust radius from a projection."""

    value = projection.get("trust_radius", dict(projection.get("info", {})).get("trust_radius"))
    return None if value is None else float(value)


def _freeb_edge_control_scale_control_jax(control: Any, projection: dict[str, Any]):
    """Apply the optional reduced-control trust radius with JAX operations."""

    trust_radius = _freeb_edge_control_trust_radius(projection)
    if trust_radius is None:
        return control
    trust = jnp.asarray(float(trust_radius), dtype=jnp.asarray(control).dtype)
    norm = jnp.linalg.norm(control)
    tiny = jnp.asarray(np.finfo(float).tiny, dtype=jnp.asarray(control).dtype)
    scale = jnp.minimum(jnp.asarray(1.0, dtype=jnp.asarray(control).dtype), trust / jnp.maximum(norm, tiny))
    return control * scale


def _freeb_edge_control_control_delta_jax(target: Any, projection: dict[str, Any]):
    """Map a physical edge target to reduced controls with JAX operations."""

    dtype = jnp.asarray(target).dtype
    operator = jnp.asarray(projection.get("control_operator_np", projection["pinv_np"]), dtype=dtype)
    return _freeb_edge_control_scale_control_jax(operator @ target, projection)


def _project_freeb_edge_control_state(
    state: VMECState,
    projection: dict[str, Any],
    *,
    host_update: bool,
) -> VMECState:
    """Project a state LCFS edge row onto an enabled reduced-control subspace."""

    if not bool(projection.get("enabled", False)):
        return state
    k = int(projection["mode_count"])
    if bool(host_update):
        scale = np.asarray(projection["mode_scale_np"], dtype=float)
        initial = {name: np.asarray(value, dtype=float) for name, value in projection["initial_np"].items()}
        Rcos = np.asarray(state.Rcos, dtype=float)
        Rsin = np.asarray(state.Rsin, dtype=float)
        Zcos = np.asarray(state.Zcos, dtype=float)
        Zsin = np.asarray(state.Zsin, dtype=float)
        target = np.concatenate(
            [
                Rcos[-1] * scale - initial["R_cos"],
                Rsin[-1] * scale - initial["R_sin"],
                Zcos[-1] * scale - initial["Z_cos"],
                Zsin[-1] * scale - initial["Z_sin"],
            ],
            axis=0,
        )
        projected = _freeb_edge_control_project_vector_np(target, projection).predicted_delta
        edge_values = np.concatenate(
            [
                initial["R_cos"] + projected[0:k],
                initial["R_sin"] + projected[k : 2 * k],
                initial["Z_cos"] + projected[2 * k : 3 * k],
                initial["Z_sin"] + projected[3 * k : 4 * k],
            ],
            axis=0,
        )
        return _freeb_edge_control_state_from_edge_values(state, projection, edge_values, host_update=True)

    dtype = jnp.asarray(state.Rcos).dtype
    scale = jnp.asarray(projection["mode_scale_np"], dtype=dtype)
    initial = {name: jnp.asarray(value, dtype=dtype) for name, value in projection["initial_np"].items()}
    jacobian = jnp.asarray(projection["jacobian_np"], dtype=dtype)
    Rcos = jnp.asarray(state.Rcos)
    Rsin = jnp.asarray(state.Rsin)
    Zcos = jnp.asarray(state.Zcos)
    Zsin = jnp.asarray(state.Zsin)
    target = jnp.concatenate(
        [
            Rcos[-1] * scale - initial["R_cos"],
            Rsin[-1] * scale - initial["R_sin"],
            Zcos[-1] * scale - initial["Z_cos"],
            Zsin[-1] * scale - initial["Z_sin"],
        ],
        axis=0,
    )
    control_delta = _freeb_edge_control_control_delta_jax(target, projection)
    projected = jacobian @ control_delta
    edge_values = jnp.concatenate(
        [
            initial["R_cos"] + projected[0:k],
            initial["R_sin"] + projected[k : 2 * k],
            initial["Z_cos"] + projected[2 * k : 3 * k],
            initial["Z_sin"] + projected[3 * k : 4 * k],
        ],
        axis=0,
    )
    return _freeb_edge_control_state_from_edge_values(state, projection, edge_values, host_update=False)


def _freeb_edge_control_state_from_edge_values(
    state: VMECState,
    projection: dict[str, Any],
    edge_values: Any,
    *,
    host_update: bool,
) -> VMECState:
    """Return ``state`` with its physical LCFS edge row replaced."""

    k = int(projection["mode_count"])
    if bool(host_update):
        edge = np.asarray(edge_values, dtype=float).reshape(-1)
        if edge.size != 4 * k:
            raise ValueError("edge_values has the wrong size for the edge-control projection")
        scale = np.asarray(projection["mode_scale_np"], dtype=float)
        Rcos = np.array(state.Rcos, dtype=float, copy=True)
        Rsin = np.array(state.Rsin, dtype=float, copy=True)
        Zcos = np.array(state.Zcos, dtype=float, copy=True)
        Zsin = np.array(state.Zsin, dtype=float, copy=True)
        Rcos[-1] = edge[0:k] / scale
        Rsin[-1] = edge[k : 2 * k] / scale
        Zcos[-1] = edge[2 * k : 3 * k] / scale
        Zsin[-1] = edge[3 * k : 4 * k] / scale
        return VMECState(
            layout=state.layout,
            Rcos=Rcos,
            Rsin=Rsin,
            Zcos=Zcos,
            Zsin=Zsin,
            Lcos=state.Lcos,
            Lsin=state.Lsin,
        )

    dtype = jnp.asarray(state.Rcos).dtype
    edge = jnp.asarray(edge_values, dtype=dtype).reshape((-1,))
    if int(edge.shape[0]) != 4 * k:
        raise ValueError("edge_values has the wrong size for the edge-control projection")
    scale = jnp.asarray(projection["mode_scale_np"], dtype=dtype)
    Rcos = jnp.asarray(state.Rcos)
    Rsin = jnp.asarray(state.Rsin)
    Zcos = jnp.asarray(state.Zcos)
    Zsin = jnp.asarray(state.Zsin)
    Rcos = Rcos.at[-1, :].set(edge[0:k] / scale)
    Rsin = Rsin.at[-1, :].set(edge[k : 2 * k] / scale)
    Zcos = Zcos.at[-1, :].set(edge[2 * k : 3 * k] / scale)
    Zsin = Zsin.at[-1, :].set(edge[3 * k : 4 * k] / scale)
    return VMECState(
        layout=state.layout,
        Rcos=Rcos,
        Rsin=Rsin,
        Zcos=Zcos,
        Zsin=Zsin,
        Lcos=state.Lcos,
        Lsin=state.Lsin,
    )


def _freeb_edge_control_state_from_coordinates(
    state: VMECState,
    projection: dict[str, Any],
    control_delta: Any,
    *,
    host_update: bool,
) -> VMECState:
    """Decode reduced edge-control coordinates into the LCFS edge row."""

    if not bool(projection.get("enabled", False)):
        return state
    if bool(host_update):
        edge_values = _freeb_edge_control_reduced_map(projection).decode(control_delta)
        return _freeb_edge_control_state_from_edge_values(state, projection, edge_values, host_update=True)

    dtype = jnp.asarray(state.Rcos).dtype
    initial = jnp.concatenate(
        [
            jnp.asarray(projection["initial_np"]["R_cos"], dtype=dtype),
            jnp.asarray(projection["initial_np"]["R_sin"], dtype=dtype),
            jnp.asarray(projection["initial_np"]["Z_cos"], dtype=dtype),
            jnp.asarray(projection["initial_np"]["Z_sin"], dtype=dtype),
        ],
        axis=0,
    )
    edge_values = reduced_control_decode(initial, projection["jacobian_np"], control_delta)
    return _freeb_edge_control_state_from_edge_values(state, projection, edge_values, host_update=False)


def _freeb_edge_control_apply_coordinate_update(
    state: VMECState,
    projection: dict[str, Any],
    control_update: Any,
    *,
    host_update: bool,
) -> VMECState:
    """Apply a reduced edge-control update to the current LCFS edge state."""

    if not bool(projection.get("enabled", False)):
        return state
    if bool(host_update):
        control_map = _freeb_edge_control_reduced_map(projection)
        current = control_map.encode(_freeb_edge_control_state_edge_values(state, projection)).control_delta
        update = np.asarray(control_update, dtype=float).reshape(-1)
        if update.size != control_map.control_count:
            raise ValueError("control_update size must match the reduced edge-control count")
        return _freeb_edge_control_state_from_coordinates(
            state,
            projection,
            current + update,
            host_update=True,
        )

    dtype = jnp.asarray(state.Rcos).dtype
    k = int(projection["mode_count"])
    scale = jnp.asarray(projection["mode_scale_np"], dtype=dtype)
    initial = jnp.concatenate(
        [
            jnp.asarray(projection["initial_np"]["R_cos"], dtype=dtype),
            jnp.asarray(projection["initial_np"]["R_sin"], dtype=dtype),
            jnp.asarray(projection["initial_np"]["Z_cos"], dtype=dtype),
            jnp.asarray(projection["initial_np"]["Z_sin"], dtype=dtype),
        ],
        axis=0,
    )
    edge_values = jnp.concatenate(
        [
            jnp.asarray(state.Rcos)[-1] * scale,
            jnp.asarray(state.Rsin)[-1] * scale,
            jnp.asarray(state.Zcos)[-1] * scale,
            jnp.asarray(state.Zsin)[-1] * scale,
        ],
        axis=0,
    )
    if int(edge_values.shape[0]) != 4 * k:
        raise ValueError("state edge row has the wrong size for the edge-control projection")
    pinv = jnp.asarray(projection["pinv_np"], dtype=dtype)
    current = pinv @ (edge_values - initial)
    update = jnp.asarray(control_update, dtype=dtype).reshape((-1,))
    if int(update.shape[0]) != int(current.shape[0]):
        raise ValueError("control_update size must match the reduced edge-control count")
    return _freeb_edge_control_state_from_coordinates(
        state,
        projection,
        current + update,
        host_update=False,
    )


def _freeb_edge_control_reduced_state_from_state(
    state: VMECState,
    projection: dict[str, Any],
) -> ReducedControlState:
    """Encode the current LCFS edge as a reduced-control state."""

    control_map = _freeb_edge_control_reduced_map(projection)
    return ReducedControlState.from_full_values(
        control_map,
        _freeb_edge_control_state_edge_values(state, projection),
    )


def _freeb_edge_control_pullback_force_np(force_deltas: Any, projection: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Pull one physical LCFS edge force/update direction into controls."""

    target = _freeb_edge_control_delta_tuple_target(force_deltas, projection)
    jacobian = np.asarray(projection["jacobian_np"], dtype=float)
    if target.size != jacobian.shape[0]:
        raise ValueError("force_deltas and edge-control Jacobian have incompatible sizes")
    return np.asarray(jacobian.T @ target, dtype=float), target


def _freeb_edge_control_native_force_np(force_deltas: Any, projection: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, str]:
    """Map one physical LCFS edge force/update direction into native controls."""

    target = _freeb_edge_control_delta_tuple_target(force_deltas, projection)
    jacobian = np.asarray(projection["jacobian_np"], dtype=float)
    if target.size != jacobian.shape[0]:
        raise ValueError("force_deltas and edge-control Jacobian have incompatible sizes")

    metric = str(projection.get("native_force_metric", "pullback")).strip().lower()
    if metric == "least_squares":
        operator = np.asarray(projection.get("control_operator_np", projection["pinv_np"]), dtype=float)
        return np.asarray(operator @ target, dtype=float), target, metric
    return np.asarray(jacobian.T @ target, dtype=float), target, "pullback"


def _freeb_edge_control_scale_control_np(control: Any, projection: dict[str, Any]) -> tuple[np.ndarray, float]:
    """Apply the optional control trust radius to a host control vector."""

    values = np.asarray(control, dtype=float).reshape(-1)
    if not np.all(np.isfinite(values)):
        raise ValueError("control vector must be finite")
    trust_radius = _freeb_edge_control_trust_radius(projection)
    if trust_radius is None:
        return values, 1.0
    norm = float(np.linalg.norm(values))
    if norm <= float(trust_radius):
        return values, 1.0
    scale = float(trust_radius) / max(norm, np.finfo(float).tiny)
    return values * scale, scale


def _freeb_edge_control_delta_tuple_from_control_update(
    deltas: Any,
    projection: dict[str, Any],
    control_update: Any,
    *,
    host_update: bool,
) -> Any:
    """Replace the LCFS edge part of an update tuple with decoded controls."""

    if deltas is None:
        return deltas
    k = int(projection["mode_count"])
    dR, dR_sin, dZ_cos, dZ, dL_cos, dL = deltas
    if bool(host_update):
        scale = np.asarray(projection["mode_scale_np"], dtype=float)
        decoded = np.asarray(projection["jacobian_np"], dtype=float) @ np.asarray(control_update, dtype=float).reshape(-1)
        if decoded.size != 4 * k:
            raise ValueError("decoded edge-control update has the wrong size")
        dR_out = np.array(dR, dtype=float, copy=True)
        dR_sin_out = np.array(dR_sin, dtype=float, copy=True)
        dZ_cos_out = np.array(dZ_cos, dtype=float, copy=True)
        dZ_out = np.array(dZ, dtype=float, copy=True)
        dR_out[-1] = decoded[0:k] / scale
        dR_sin_out[-1] = decoded[k : 2 * k] / scale
        dZ_cos_out[-1] = decoded[2 * k : 3 * k] / scale
        dZ_out[-1] = decoded[3 * k : 4 * k] / scale
        return (dR_out, dR_sin_out, dZ_cos_out, dZ_out, dL_cos, dL)

    dtype = jnp.asarray(dR).dtype
    scale = jnp.asarray(projection["mode_scale_np"], dtype=dtype)
    decoded = jnp.asarray(projection["jacobian_np"], dtype=dtype) @ jnp.asarray(control_update, dtype=dtype).reshape((-1,))
    if int(decoded.shape[0]) != 4 * k:
        raise ValueError("decoded edge-control update has the wrong size")
    dR_out = jnp.asarray(dR).at[-1, :].set(decoded[0:k] / scale)
    dR_sin_out = jnp.asarray(dR_sin).at[-1, :].set(decoded[k : 2 * k] / scale)
    dZ_cos_out = jnp.asarray(dZ_cos).at[-1, :].set(decoded[2 * k : 3 * k] / scale)
    dZ_out = jnp.asarray(dZ).at[-1, :].set(decoded[3 * k : 4 * k] / scale)
    return (dR_out, dR_sin_out, dZ_cos_out, dZ_out, dL_cos, dL)


def _freeb_edge_control_native_coordinate_step(
    *,
    state_current: VMECState,
    state_candidate: VMECState,
    update_deltas: Any,
    force_deltas: Any,
    projection: dict[str, Any],
    control_velocity: Any | None,
    control_coordinates: Any | None,
    dt_eff: float,
    b1: float,
    fac: float,
    force_scale: float,
    flip_sign: float,
    host_update: bool,
) -> FreeBoundaryNativeControlStep:
    """Advance the LCFS edge in solver-native reduced-control coordinates."""

    control_force, target, force_metric = _freeb_edge_control_native_force_np(force_deltas, projection)
    previous = (
        np.zeros_like(control_force)
        if control_velocity is None
        else np.asarray(control_velocity, dtype=float).reshape(control_force.shape)
    )
    next_velocity = float(fac) * (float(b1) * previous + float(force_scale) * float(flip_sign) * control_force)
    control_update = float(dt_eff) * next_velocity
    control_update, trust_scale = _freeb_edge_control_scale_control_np(control_update, projection)
    if trust_scale != 1.0:
        next_velocity = control_update / max(float(dt_eff), np.finfo(float).tiny)

    current_state = (
        _freeb_edge_control_reduced_state_from_state(state_current, projection)
        if control_coordinates is None
        else ReducedControlState(
            control_map=_freeb_edge_control_reduced_map(projection),
            control_delta=control_coordinates,
        )
    )
    next_control_state = current_state.update(control_update)
    native_edge_state = _freeb_edge_control_state_from_coordinates(
        state_current,
        projection,
        next_control_state.control_delta,
        host_update=bool(host_update),
    )
    native_edge_values = _freeb_edge_control_state_edge_values(native_edge_state, projection)
    state_out = _freeb_edge_control_state_from_edge_values(
        state_candidate,
        projection,
        native_edge_values,
        host_update=bool(host_update),
    )
    update_deltas_out = _freeb_edge_control_delta_tuple_from_control_update(
        update_deltas,
        projection,
        control_update,
        host_update=bool(host_update),
    )
    return FreeBoundaryNativeControlStep(
        state=state_out,
        update_deltas=update_deltas_out,
        control_velocity=np.asarray(next_velocity, dtype=float),
        control_coordinates=np.asarray(next_control_state.control_delta, dtype=float),
        control_update=np.asarray(control_update, dtype=float),
        control_force=np.asarray(control_force, dtype=float),
        target_l2=float(np.linalg.norm(target)),
        control_force_l2=float(np.linalg.norm(control_force)),
        control_velocity_l2=float(np.linalg.norm(next_velocity)),
        control_update_l2=float(np.linalg.norm(control_update)),
        trust_scale=float(trust_scale),
        force_metric=str(force_metric),
    )


def _freeb_edge_control_project_vector_np(
    target: Any,
    projection: dict[str, Any],
):
    """Project one physical edge vector using the shared reduced-control solver."""

    target_arr = np.asarray(target, dtype=float).reshape(-1)
    jacobian = np.asarray(projection["jacobian_np"], dtype=float)
    if target_arr.size != jacobian.shape[0]:
        raise ValueError("edge-control target and Jacobian have incompatible sizes")
    labels = tuple(str(label) for label in dict(projection.get("info", {})).get("labels", []))
    info = dict(projection.get("info", {}))
    return reduced_control_least_squares_step(
        jacobian,
        target_arr,
        labels=labels if labels else None,
        ridge=float(info.get("ridge", projection.get("ridge", 0.0)) or 0.0),
        rcond=info.get("rcond"),
        trust_radius=projection.get("trust_radius", info.get("trust_radius")),
    )


def _freeb_edge_control_reduced_map(projection: dict[str, Any]) -> ReducedControlMap:
    """Return the affine reduced-control map for a prepared edge projection."""

    if not bool(projection.get("enabled", False)):
        raise ValueError("edge-control projection is not enabled")
    initial = {name: np.asarray(value, dtype=float) for name, value in projection["initial_np"].items()}
    initial_vector = np.concatenate(
        [
            initial["R_cos"],
            initial["R_sin"],
            initial["Z_cos"],
            initial["Z_sin"],
        ],
        axis=0,
    )
    info = dict(projection.get("info", {}))
    labels = tuple(str(label) for label in info.get("labels", []))
    return ReducedControlMap(
        initial=initial_vector,
        jacobian=np.asarray(projection["jacobian_np"], dtype=float),
        labels=labels,
        rcond=info.get("rcond"),
    )


def _freeb_edge_control_state_edge_values(state: VMECState, projection: dict[str, Any]) -> np.ndarray:
    """Return stacked physical LCFS edge coefficients for a prepared projection."""

    k = int(projection["mode_count"])
    scale = np.asarray(projection["mode_scale_np"], dtype=float)
    edge_values = np.concatenate(
        [
            np.asarray(state.Rcos, dtype=float)[-1] * scale,
            np.asarray(state.Rsin, dtype=float)[-1] * scale,
            np.asarray(state.Zcos, dtype=float)[-1] * scale,
            np.asarray(state.Zsin, dtype=float)[-1] * scale,
        ],
        axis=0,
    )
    if edge_values.size != 4 * k:
        raise ValueError("state edge row has the wrong size for the edge-control projection")
    return edge_values


def _freeb_edge_control_state_coordinates(state: VMECState, projection: dict[str, Any]) -> dict[str, Any]:
    """Fit the accepted LCFS edge row in reduced-control coordinates."""

    if not bool(projection.get("enabled", False)):
        return {"enabled": False, "status": "disabled"}
    control_map = _freeb_edge_control_reduced_map(projection)
    edge_values = _freeb_edge_control_state_edge_values(state, projection)
    step = control_map.encode(edge_values)
    residual = step.residual_after
    finite = residual[np.isfinite(residual)]
    residual_linf = float(np.max(np.abs(finite))) if finite.size else 0.0
    residual_rms = float(np.sqrt(np.mean(finite * finite))) if finite.size else 0.0
    map_diag = control_map.to_dict()
    info = dict(projection.get("info", {}))
    return {
        "enabled": True,
        "status": "measured",
        "mode": "affine_edge_control_coordinates",
        "basis_symmetry": info.get("basis_symmetry"),
        "initial_boundary_source": info.get("initial_boundary_source"),
        "mode_count": int(projection["mode_count"]),
        "full_size": int(map_diag["full_size"]),
        "control_count": int(map_diag["control_count"]),
        "rank": int(map_diag["rank"]),
        "rank_deficient": bool(map_diag["rank_deficient"]),
        "condition_number": map_diag["condition_number"],
        "rcond": map_diag["rcond"],
        "labels": list(step.labels),
        "coordinates": [float(value) for value in step.control_delta],
        "coordinate_by_label": step.control_delta_by_label,
        "coordinate_l2": float(step.control_l2),
        "coordinate_linf": float(step.control_linf),
        "target_l2": float(step.target_l2),
        "reconstruction_l2": float(step.predicted_l2),
        "reconstruction_residual_l2": float(step.residual_l2),
        "reconstruction_residual_linf": residual_linf,
        "reconstruction_residual_rms": residual_rms,
        "reconstruction_residual_rel": step.residual_rel,
    }


def _freeb_edge_control_reduced_unknown_vector_diagnostics(
    state: VMECState,
    projection: dict[str, Any],
) -> dict[str, Any]:
    """Return the reduced edge-state vector implied by the accepted LCFS."""

    if not bool(projection.get("enabled", False)):
        return {"enabled": False, "status": "disabled"}
    control_map = _freeb_edge_control_reduced_map(projection)
    edge_values = _freeb_edge_control_state_edge_values(state, projection)
    step = control_map.encode(edge_values)
    decoded = control_map.decode(step.control_delta)
    residual = edge_values - decoded
    finite = residual[np.isfinite(residual)]
    residual_linf = float(np.max(np.abs(finite))) if finite.size else 0.0
    residual_rms = float(np.sqrt(np.mean(finite * finite))) if finite.size else 0.0
    full_size = int(control_map.full_size)
    control_count = int(control_map.control_count)
    return {
        "enabled": True,
        "status": "measured",
        "mode": "reduced_edge_unknown_vector",
        "full_edge_size": full_size,
        "reduced_unknown_size": control_count,
        "full_to_reduced_size_ratio": None if control_count == 0 else float(full_size / control_count),
        "reduction_fraction": None if full_size == 0 else float(control_count / full_size),
        "labels": list(step.labels),
        "unknown_vector": [float(value) for value in step.control_delta],
        "unknown_by_label": step.control_delta_by_label,
        "unknown_l2": float(step.control_l2),
        "unknown_linf": float(step.control_linf),
        "rank": int(step.rank),
        "condition_number": step.condition_number,
        "decoded_residual_l2": float(np.linalg.norm(residual)),
        "decoded_residual_linf": residual_linf,
        "decoded_residual_rms": residual_rms,
        "decoded_residual_rel": step.residual_rel,
    }


def _freeb_edge_control_vector_projection_metrics(
    target: Any,
    projection: dict[str, Any],
    *,
    status: str,
) -> dict[str, Any]:
    """Project one stacked physical edge vector onto reduced controls."""

    jacobian = np.asarray(projection["jacobian_np"], dtype=float)
    step = _freeb_edge_control_project_vector_np(target, projection)
    projected = step.predicted_delta
    residual = step.residual_after
    finite = residual[np.isfinite(residual)]
    target_l2 = step.target_l2
    projected_l2 = step.predicted_l2
    residual_l2 = step.residual_l2 if finite.size else 0.0
    residual_linf = float(np.max(np.abs(finite))) if finite.size else 0.0
    residual_rms = float(np.sqrt(np.mean(finite * finite))) if finite.size else 0.0
    residual_rel = step.residual_rel
    captured_fraction = None if target_l2 <= np.finfo(float).tiny else float(projected_l2 / target_l2)
    residual_energy_fraction = None if target_l2 <= np.finfo(float).tiny else float((residual_l2 / target_l2) ** 2)
    return {
        "enabled": True,
        "status": str(status),
        "mode": "edge_delta_least_squares",
        "mode_count": int(projection["mode_count"]),
        "control_count": int(jacobian.shape[1]),
        "rank": int(step.rank),
        "condition_number": step.condition_number,
        "target_l2": target_l2,
        "projected_l2": projected_l2,
        "residual_l2": residual_l2,
        "residual_linf": residual_linf,
        "residual_rms": residual_rms,
        "residual_rel": residual_rel,
        "captured_fraction": captured_fraction,
        "residual_energy_fraction": residual_energy_fraction,
        "control_delta_l2": step.control_l2,
        "control_delta_linf": step.control_linf,
        "control_delta_by_label": step.control_delta_by_label,
        "ridge": float(step.ridge),
        "trust_radius": None if step.trust_radius is None else float(step.trust_radius),
        "trust_scale": float(step.trust_scale),
    }


def _freeb_edge_control_delta_tuple_target(deltas: Any, projection: dict[str, Any]) -> np.ndarray:
    """Return the stacked physical LCFS edge update from a VMEC delta tuple."""

    k = int(projection["mode_count"])
    scale = np.asarray(projection["mode_scale_np"], dtype=float)
    dR, dR_sin, dZ_cos, dZ, _dL_cos, _dL = deltas
    target = np.concatenate(
        [
            np.asarray(dR, dtype=float)[-1] * scale,
            np.asarray(dR_sin, dtype=float)[-1] * scale,
            np.asarray(dZ_cos, dtype=float)[-1] * scale,
            np.asarray(dZ, dtype=float)[-1] * scale,
        ],
        axis=0,
    )
    if target.size != 4 * k:
        raise ValueError("edge-control update direction has the wrong size")
    return target


def _freeb_edge_control_delta_tuple_projection_metrics(deltas: Any, projection: dict[str, Any]) -> dict[str, Any]:
    """Measure how much an edge update direction lies outside controls."""

    if not bool(projection.get("enabled", False)):
        return {"enabled": False, "status": "disabled"}
    target = _freeb_edge_control_delta_tuple_target(deltas, projection)
    return _freeb_edge_control_vector_projection_metrics(target, projection, status="measured")


def _freeb_edge_control_reduced_update_direction_diagnostics(
    deltas: Any,
    projection: dict[str, Any],
) -> dict[str, Any]:
    """Return the reduced edge-control vector fitted to one update direction."""

    if not bool(projection.get("enabled", False)):
        return {"enabled": False, "status": "disabled"}
    target = _freeb_edge_control_delta_tuple_target(deltas, projection)
    step = _freeb_edge_control_project_vector_np(target, projection)
    decoded = step.predicted_delta
    residual = target - decoded
    finite = residual[np.isfinite(residual)]
    residual_linf = float(np.max(np.abs(finite))) if finite.size else 0.0
    residual_rms = float(np.sqrt(np.mean(finite * finite))) if finite.size else 0.0
    full_size = int(target.size)
    reduced_size = int(np.asarray(step.control_delta).size)
    captured_fraction = None if step.target_l2 <= np.finfo(float).tiny else float(step.predicted_l2 / step.target_l2)
    return {
        "enabled": True,
        "status": "measured",
        "mode": "reduced_edge_update_direction",
        "full_update_size": full_size,
        "reduced_update_size": reduced_size,
        "full_to_reduced_size_ratio": None if reduced_size == 0 else float(full_size / reduced_size),
        "reduction_fraction": None if full_size == 0 else float(reduced_size / full_size),
        "labels": list(step.labels),
        "update_vector": [float(value) for value in step.control_delta],
        "update_by_label": step.control_delta_by_label,
        "update_l2": float(step.control_l2),
        "update_linf": float(step.control_linf),
        "rank": int(step.rank),
        "condition_number": step.condition_number,
        "target_l2": float(step.target_l2),
        "decoded_update_l2": float(step.predicted_l2),
        "decoded_residual_l2": float(np.linalg.norm(residual)),
        "decoded_residual_linf": residual_linf,
        "decoded_residual_rms": residual_rms,
        "decoded_residual_rel": step.residual_rel,
        "captured_fraction": captured_fraction,
    }


def _project_freeb_edge_control_delta_tuple(
    deltas: Any,
    projection: dict[str, Any],
    *,
    host_update: bool,
) -> Any:
    """Project an LCFS geometry update direction onto reduced controls.

    The reduced-control basis is defined in physical edge-coefficient space, so
    only the last radial row of the geometry deltas is changed. Lambda and
    interior geometry updates are left untouched.
    """

    if not bool(projection.get("enabled", False)):
        return deltas
    k = int(projection["mode_count"])
    dR, dR_sin, dZ_cos, dZ, dL_cos, dL = deltas
    if bool(host_update):
        scale = np.asarray(projection["mode_scale_np"], dtype=float)
        dR_out = np.array(dR, dtype=float, copy=True)
        dR_sin_out = np.array(dR_sin, dtype=float, copy=True)
        dZ_cos_out = np.array(dZ_cos, dtype=float, copy=True)
        dZ_out = np.array(dZ, dtype=float, copy=True)
        target = np.concatenate(
            [
                dR_out[-1] * scale,
                dR_sin_out[-1] * scale,
                dZ_cos_out[-1] * scale,
                dZ_out[-1] * scale,
            ],
            axis=0,
        )
        projected = _freeb_edge_control_project_vector_np(target, projection).predicted_delta
        dR_out[-1] = projected[0:k] / scale
        dR_sin_out[-1] = projected[k : 2 * k] / scale
        dZ_cos_out[-1] = projected[2 * k : 3 * k] / scale
        dZ_out[-1] = projected[3 * k : 4 * k] / scale
        return (dR_out, dR_sin_out, dZ_cos_out, dZ_out, dL_cos, dL)

    dtype = jnp.asarray(dR).dtype
    scale = jnp.asarray(projection["mode_scale_np"], dtype=dtype)
    jacobian = jnp.asarray(projection["jacobian_np"], dtype=dtype)
    dR_out = jnp.asarray(dR)
    dR_sin_out = jnp.asarray(dR_sin)
    dZ_cos_out = jnp.asarray(dZ_cos)
    dZ_out = jnp.asarray(dZ)
    target = jnp.concatenate(
        [
            dR_out[-1] * scale,
            dR_sin_out[-1] * scale,
            dZ_cos_out[-1] * scale,
            dZ_out[-1] * scale,
        ],
        axis=0,
    )
    control_delta = _freeb_edge_control_control_delta_jax(target, projection)
    projected = jacobian @ control_delta
    dR_out = dR_out.at[-1, :].set(projected[0:k] / scale)
    dR_sin_out = dR_sin_out.at[-1, :].set(projected[k : 2 * k] / scale)
    dZ_cos_out = dZ_cos_out.at[-1, :].set(projected[2 * k : 3 * k] / scale)
    dZ_out = dZ_out.at[-1, :].set(projected[3 * k : 4 * k] / scale)
    return (dR_out, dR_sin_out, dZ_cos_out, dZ_out, dL_cos, dL)


def _freeb_edge_control_state_residual_metrics(state: VMECState, projection: dict[str, Any]) -> dict[str, Any]:
    """Measure how far the LCFS edge row sits outside reduced controls."""

    if not bool(projection.get("enabled", False)):
        return {"enabled": False, "status": "disabled"}
    control_map = _freeb_edge_control_reduced_map(projection)
    edge_values = _freeb_edge_control_state_edge_values(state, projection)
    target = edge_values - control_map.initial
    return _freeb_edge_control_vector_projection_metrics(target, projection, status="measured")


def _zero_freeb_edge_control_velocity_blocks(
    blocks: Any,
    projection: dict[str, Any],
    *,
    host_update: bool,
) -> Any:
    """Clear LCFS geometry velocity memory for reduced edge-control solves.

    The reduced edge-control projector constrains accepted boundary states to a
    low-dimensional spline-control subspace.  VMEC's momentum memory is stored
    in parity-channel velocity blocks, not directly in that reduced basis, so
    carrying old LCFS geometry velocities can reintroduce uncontrolled Fourier
    directions on the next iteration.  Clearing only the LCFS geometry rows is a
    conservative first-order edge update while preserving interior and lambda
    momentum memory.
    """

    if not bool(projection.get("enabled", False)):
        return blocks
    geometry_names = ("rcc", "rss", "rsc", "rcs", "zsc", "zcs", "zcc", "zss")

    def _zero_edge_row(value: Any) -> Any:
        if bool(host_update):
            arr = np.array(value, dtype=float, copy=True)
            if arr.ndim == 0:
                return arr
            arr[-1, ...] = 0.0
            return arr
        arr = jnp.asarray(value)
        if arr.ndim == 0:
            return arr
        return arr.at[-1, ...].set(jnp.zeros_like(arr[-1, ...]))

    updates = {name: _zero_edge_row(getattr(blocks, name)) for name in geometry_names}
    return blocks._replace(**updates)


class FreeBoundaryNestorIterationResult(NamedTuple):
    """External-vacuum coupling state for one residual-loop iteration."""

    bsqvac_half_current: Any
    runtime: Any
    trace_arrays: Any
    reused: bool
    solve_time: float
    sample_time: float
    last_model: str
    last_diagnostics: dict[str, Any]
    ivac: int
    ivac_effective: int
    controls_cached: tuple[int, int, int] | None


def free_boundary_nestor_iteration_coupling(
    *,
    free_boundary_enabled: bool,
    freeb_couple_edge: bool,
    state: Any,
    static: Any,
    freeb_ivac: int,
    freeb_ivacskip: int,
    iter2: int,
    freeb_nestor_runtime: Any,
    freeb_plascur: float,
    external_field_provider_kind: str | None,
    external_field_provider_static: Any,
    external_field_provider_params: Any,
    collect_trace_arrays: bool,
    freeb_turnon_iter: bool,
    freeb_ivac_effective: int,
    freeb_nvacskip: int,
    controls_cached: tuple[int, int, int] | None,
    last_model: str,
    last_diagnostics: dict[str, Any],
    env_freeb_raise: bool,
    nestor_external_only_step_func: Callable[..., Any],
    edge_bsqvac_from_nestor_func: Callable[..., Any],
    source_reused_history: list[int],
    provider_allows_source_reuse_history: list[int],
    bnormal_rms_history: list[float],
    gsource_rms_history: list[float],
    bsqvac_rms_history: list[float],
) -> FreeBoundaryNestorIterationResult:
    """Run the VMEC-style free-boundary external-vacuum step when active."""

    bsqvac_half_current = None
    trace_arrays = None
    reused = False
    solve_time = 0.0
    sample_time = 0.0
    runtime_out = freeb_nestor_runtime
    model = str(last_model)
    diagnostics = dict(last_diagnostics)
    ivac = int(freeb_ivac)
    ivac_effective = int(freeb_ivac_effective)
    controls = controls_cached

    if not bool(free_boundary_enabled and freeb_couple_edge):
        return FreeBoundaryNestorIterationResult(
            bsqvac_half_current,
            freeb_nestor_runtime,
            trace_arrays,
            reused,
            solve_time,
            sample_time,
            model,
            diagnostics,
            ivac,
            ivac_effective,
            controls,
        )

    try:
        # VMEC enters NESTOR once control is active (`ivac >= 0`); vacuum.f
        # promotes ivac=0 -> 1 internally on the first turn-on.
        if ivac < 0:
            return FreeBoundaryNestorIterationResult(
                bsqvac_half_current,
                freeb_nestor_runtime,
                trace_arrays,
                reused,
                solve_time,
                sample_time,
                model,
                diagnostics,
                ivac,
                ivac_effective,
                controls,
            )
        nestor_res, runtime_out = nestor_external_only_step_func(
            state=state,
            static=static,
            ivac=ivac,
            ivacskip=int(freeb_ivacskip),
            iter_idx=int(iter2),
            runtime=freeb_nestor_runtime,
            extcur=tuple(getattr(static, "free_boundary_extcur", ()) or ()),
            plascur=float(freeb_plascur),
            external_field_provider_kind=external_field_provider_kind,
            external_field_provider_static=external_field_provider_static,
            external_field_provider_params=external_field_provider_params,
            collect_trace_arrays=bool(collect_trace_arrays),
        )
        model = str(getattr(nestor_res, "model", "spectral_poisson_external_only"))
        reused = bool(getattr(nestor_res, "reused", False))
        solve_time = float(getattr(nestor_res, "solve_time_s", 0.0))
        sample_time = float(getattr(nestor_res, "sample_time_s", 0.0))
        trace_arrays = getattr(nestor_res, "trace_arrays", None)
        diag_nestor = getattr(nestor_res, "diagnostics", None)
        if isinstance(diag_nestor, dict):
            diagnostics = dict(diag_nestor)
            source_reused_history.append(1 if bool(diag_nestor.get("source_reused", False)) else 0)
            provider_allows_source_reuse_history.append(
                1 if bool(diag_nestor.get("provider_allows_source_reuse", False)) else 0
            )
            for key, history in (
                ("bnormal_rms", bnormal_rms_history),
                ("gsource_rms", gsource_rms_history),
                ("bsqvac_rms", bsqvac_rms_history),
            ):
                try:
                    history.append(float(diag_nestor.get(key, float("nan"))))
                except Exception:
                    history.append(float("nan"))
        else:
            source_reused_history.append(0)
            provider_allows_source_reuse_history.append(0)
            bnormal_rms_history.append(float("nan"))
            gsource_rms_history.append(float("nan"))
            bsqvac_rms_history.append(float("nan"))
        bsqvac_half_current = edge_bsqvac_from_nestor_func(nestor_res, static)
        if freeb_turnon_iter:
            ivac = 1
            ivac_effective = 1
            controls = (ivac, int(freeb_ivacskip), int(freeb_nvacskip))
    except Exception:
        if env_freeb_raise:
            raise
        bsqvac_half_current = None
        trace_arrays = None
        reused = False
        solve_time = 0.0
        sample_time = 0.0

    return FreeBoundaryNestorIterationResult(
        bsqvac_half_current,
        runtime_out,
        trace_arrays,
        reused,
        solve_time,
        sample_time,
        model,
        diagnostics,
        ivac,
        ivac_effective,
        controls,
    )


_zero_velocity_blocks_like = zero_velocity_blocks_like
_scale_velocity_blocks = scale_velocity_blocks
