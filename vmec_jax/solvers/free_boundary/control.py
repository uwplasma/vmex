"""Small free-boundary iteration-control helpers for VMEC solves."""

from __future__ import annotations

import os
from typing import Any, Callable, NamedTuple

import numpy as np

from vmec_jax._compat import jnp
from vmec_jax.boundary import boundary_from_indata
from vmec_jax.state import VMECState

from ..fixed_boundary.residual.update import (
    scale_velocity_blocks,
    zero_velocity_blocks_like,
)


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
        elif indata is not None:
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
        pinv = np.linalg.pinv(jacobian, rcond=rcond)
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
            "singular_values": [float(value) for value in singular_values],
            "condition_number": condition_number,
            "initial_boundary_source": initial_source,
        }
    )
    return {
        "enabled": True,
        "info": info,
        "mode_count": int(mode_count),
        "jacobian_np": jacobian,
        "pinv_np": pinv,
        "mode_scale_np": mode_scale,
        "initial_np": initial,
    }


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
        jacobian = np.asarray(projection["jacobian_np"], dtype=float)
        pinv = np.asarray(projection["pinv_np"], dtype=float)
        Rcos = np.array(state.Rcos, dtype=float, copy=True)
        Rsin = np.array(state.Rsin, dtype=float, copy=True)
        Zcos = np.array(state.Zcos, dtype=float, copy=True)
        Zsin = np.array(state.Zsin, dtype=float, copy=True)
        target = np.concatenate(
            [
                Rcos[-1] * scale - initial["R_cos"],
                Rsin[-1] * scale - initial["R_sin"],
                Zcos[-1] * scale - initial["Z_cos"],
                Zsin[-1] * scale - initial["Z_sin"],
            ],
            axis=0,
        )
        projected = jacobian @ (pinv @ target)
        Rcos[-1] = (initial["R_cos"] + projected[0:k]) / scale
        Rsin[-1] = (initial["R_sin"] + projected[k : 2 * k]) / scale
        Zcos[-1] = (initial["Z_cos"] + projected[2 * k : 3 * k]) / scale
        Zsin[-1] = (initial["Z_sin"] + projected[3 * k : 4 * k]) / scale
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
    scale = jnp.asarray(projection["mode_scale_np"], dtype=dtype)
    initial = {name: jnp.asarray(value, dtype=dtype) for name, value in projection["initial_np"].items()}
    jacobian = jnp.asarray(projection["jacobian_np"], dtype=dtype)
    pinv = jnp.asarray(projection["pinv_np"], dtype=dtype)
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
    projected = jacobian @ (pinv @ target)
    Rcos = Rcos.at[-1, :].set((initial["R_cos"] + projected[0:k]) / scale)
    Rsin = Rsin.at[-1, :].set((initial["R_sin"] + projected[k : 2 * k]) / scale)
    Zcos = Zcos.at[-1, :].set((initial["Z_cos"] + projected[2 * k : 3 * k]) / scale)
    Zsin = Zsin.at[-1, :].set((initial["Z_sin"] + projected[3 * k : 4 * k]) / scale)
    return VMECState(
        layout=state.layout,
        Rcos=Rcos,
        Rsin=Rsin,
        Zcos=Zcos,
        Zsin=Zsin,
        Lcos=state.Lcos,
        Lsin=state.Lsin,
    )


def _freeb_edge_control_vector_projection_metrics(
    target: Any,
    projection: dict[str, Any],
    *,
    status: str,
) -> dict[str, Any]:
    """Project one stacked physical edge vector onto reduced controls."""

    target = np.asarray(target, dtype=float).reshape(-1)
    jacobian = np.asarray(projection["jacobian_np"], dtype=float)
    pinv = np.asarray(projection["pinv_np"], dtype=float)
    if target.size != jacobian.shape[0]:
        raise ValueError("edge-control target and Jacobian have incompatible sizes")
    control_delta = pinv @ target
    projected = jacobian @ control_delta
    residual = target - projected
    finite = residual[np.isfinite(residual)]
    target_l2 = float(np.linalg.norm(target))
    projected_l2 = float(np.linalg.norm(projected))
    residual_l2 = float(np.linalg.norm(finite)) if finite.size else 0.0
    residual_linf = float(np.max(np.abs(finite))) if finite.size else 0.0
    residual_rms = float(np.sqrt(np.mean(finite * finite))) if finite.size else 0.0
    residual_rel = None if target_l2 <= np.finfo(float).tiny else float(residual_l2 / target_l2)
    labels = list(dict(projection.get("info", {})).get("labels", []))
    return {
        "enabled": True,
        "status": str(status),
        "mode": "edge_delta_least_squares",
        "mode_count": int(projection["mode_count"]),
        "control_count": int(jacobian.shape[1]),
        "target_l2": target_l2,
        "projected_l2": projected_l2,
        "residual_l2": residual_l2,
        "residual_linf": residual_linf,
        "residual_rms": residual_rms,
        "residual_rel": residual_rel,
        "control_delta_l2": float(np.linalg.norm(control_delta)),
        "control_delta_linf": float(np.max(np.abs(control_delta))) if control_delta.size else 0.0,
        "control_delta_by_label": {
            str(label): float(value) for label, value in zip(labels, control_delta, strict=False)
        },
    }


def _freeb_edge_control_delta_tuple_projection_metrics(deltas: Any, projection: dict[str, Any]) -> dict[str, Any]:
    """Measure how much an edge update direction lies outside controls."""

    if not bool(projection.get("enabled", False)):
        return {"enabled": False, "status": "disabled"}
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
    return _freeb_edge_control_vector_projection_metrics(target, projection, status="measured")


def _freeb_edge_control_state_residual_metrics(state: VMECState, projection: dict[str, Any]) -> dict[str, Any]:
    """Measure how far the LCFS edge row sits outside reduced controls."""

    if not bool(projection.get("enabled", False)):
        return {"enabled": False, "status": "disabled"}
    k = int(projection["mode_count"])
    scale = np.asarray(projection["mode_scale_np"], dtype=float)
    initial = {name: np.asarray(value, dtype=float) for name, value in projection["initial_np"].items()}
    target = np.concatenate(
        [
            np.asarray(state.Rcos, dtype=float)[-1] * scale - initial["R_cos"],
            np.asarray(state.Rsin, dtype=float)[-1] * scale - initial["R_sin"],
            np.asarray(state.Zcos, dtype=float)[-1] * scale - initial["Z_cos"],
            np.asarray(state.Zsin, dtype=float)[-1] * scale - initial["Z_sin"],
        ],
        axis=0,
    )
    if target.size != 4 * k:
        raise ValueError("state edge row has the wrong size for the edge-control projection")
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
