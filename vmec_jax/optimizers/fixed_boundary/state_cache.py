"""Cache and boundary-state helpers for fixed-boundary optimization."""

from __future__ import annotations

from collections import OrderedDict
import math
import time

import numpy as np

from ...boundary import BoundaryCoeffs
from ...namelist import InData
from ...state import VMECState
from .parameterization import apply_boundary_params
from .parameterization import apply_boundary_params_numpy
from .parameterization import indexed_boundary_maps_from_boundary


def exact_cache_key(params) -> bytes:
    """Stable byte key for a boundary-parameter vector."""

    return np.asarray(params, dtype=float).reshape(-1).tobytes()


def callback_point_id(optimizer, cache_key: bytes) -> int:
    """Return a stable callback point id for trace summaries."""

    point_ids = getattr(optimizer, "_callback_point_ids", None)
    if point_ids is None:
        optimizer._callback_point_ids = {}
        point_ids = optimizer._callback_point_ids
    point_id = point_ids.get(cache_key)
    if point_id is None:
        point_id = len(point_ids)
        point_ids[cache_key] = point_id
    return int(point_id)


def remember_initial_state(optimizer, params, state: VMECState) -> None:
    """Store a projected initial VMEC state in the small LRU cache."""

    cache = getattr(optimizer, "_initial_state_cache", None)
    if cache is None:
        optimizer._initial_state_cache = OrderedDict()
        cache = optimizer._initial_state_cache
    cache_key = optimizer._exact_cache_key(params)
    cache[cache_key] = state
    cache.move_to_end(cache_key)
    max_size = max(0, int(getattr(optimizer, "_initial_state_cache_max", 0)))
    while max_size and len(cache) > max_size:
        cache.popitem(last=False)
    if max_size == 0:
        cache.clear()


def initial_state_from_params(
    optimizer,
    params,
    *,
    profile_name: str,
    initial_guess_from_boundary_func,
) -> VMECState:
    """Return a projected initial state for the current boundary parameters."""

    cache_key = optimizer._exact_cache_key(params)
    cache = getattr(optimizer, "_initial_state_cache", None)
    if cache is not None and cache_key in cache:
        state0 = cache.pop(cache_key)
        cache[cache_key] = state0
        optimizer._profile_add(f"{profile_name}_cache_hit", 0.0)
        return state0

    t_guess = time.perf_counter()
    state0 = optimizer._initial_state_from_params_jit(params)
    if state0 is None:
        boundary_now = optimizer._boundary_from_params(params)
        axis_override = getattr(optimizer, "_initial_axis_override", None)
        if axis_override is None:
            state0 = initial_guess_from_boundary_func(
                optimizer._static,
                boundary_now,
                optimizer._indata,
                vmec_project=True,
            )
        else:
            state0 = initial_guess_from_boundary_func(
                optimizer._static,
                boundary_now,
                optimizer._indata,
                vmec_project=True,
                axis_override=axis_override,
            )
    optimizer._remember_initial_state(params, state0)
    optimizer._profile_add(profile_name, time.perf_counter() - t_guess)
    return state0


def remember_exact_state(optimizer, cache_key: bytes, state: VMECState) -> None:
    """Remember the exact accepted state and invalidate stale residuals."""

    optimizer._exact_state_cache = {cache_key: state}
    if not hasattr(optimizer, "_exact_state_key_by_id"):
        optimizer._exact_state_key_by_id = {}
    optimizer._exact_state_key_by_id[id(state)] = cache_key
    residual_cache = getattr(optimizer, "_exact_residual_cache", None)
    if residual_cache is not None and cache_key not in residual_cache:
        residual_cache.clear()


def state_matches_params(optimizer, state: VMECState, params) -> bool:
    """Return true when *state* is a known exact solve for *params*."""

    state_keys = getattr(optimizer, "_exact_state_key_by_id", {})
    return state_keys.get(id(state)) == optimizer._exact_cache_key(params)


def remember_exact_residual(optimizer, cache_key: bytes, residual: np.ndarray) -> None:
    """Remember the most recent exact residual for same-point callbacks."""

    optimizer._exact_residual_cache = {cache_key: np.asarray(residual, dtype=float).reshape(-1).copy()}


def remember_exact_jacobian(optimizer, cache_key: bytes, jacobian: np.ndarray, residual: np.ndarray) -> None:
    """Keep the most recent dense accepted-point Jacobian for same-point callbacks."""

    optimizer._exact_jacobian_cache = {
        cache_key: (
            np.asarray(jacobian, dtype=float).copy(),
            np.asarray(residual, dtype=float).reshape(-1).copy(),
        )
    }


def remember_best_exact_point(
    optimizer,
    params,
    residual: np.ndarray,
    cost: float | None = None,
    *,
    state: VMECState | None = None,
) -> None:
    """Track the best exact accepted-point residual seen during one run."""

    residual_arr = np.asarray(residual, dtype=float).reshape(-1)
    if cost is None:
        cost = 0.5 * float(np.dot(residual_arr, residual_arr))
    if not np.isfinite(float(cost)) or not np.all(np.isfinite(residual_arr)):
        return
    if float(cost) < float(getattr(optimizer, "_best_exact_cost", math.inf)):
        cache_key = optimizer._exact_cache_key(params)
        optimizer._best_exact_cost = float(cost)
        optimizer._best_exact_params = np.asarray(params, dtype=float).reshape(-1).copy()
        optimizer._best_exact_residual = residual_arr.copy()
        best_state = state
        if best_state is not None and not optimizer._state_matches_params(best_state, params):
            best_state = None
        if best_state is None:
            exact_cache = getattr(optimizer, "_exact_cache", {})
            if cache_key in exact_cache:
                best_state = exact_cache[cache_key][0]
            else:
                best_state = getattr(optimizer, "_exact_state_cache", {}).get(cache_key)
        optimizer._best_exact_state = best_state


def append_exact_history_entry(
    optimizer,
    params,
    entry: dict,
    *,
    exact_residual: np.ndarray | None,
    state: VMECState | None,
) -> bool:
    """Append or reject one exact accepted-point history entry."""

    entry_cost = float(entry["cost"])
    if not optimizer._exact_history_accepts(entry_cost):
        optimizer._exact_history_rejected_count += 1
        return False
    optimizer._history.append(entry)
    if exact_residual is not None:
        optimizer._remember_best_exact_point(params, exact_residual, entry_cost, state=state)
    return True


def final_history_wall_time(optimizer) -> float:
    """Return a final history timestamp that never goes backwards."""

    final_wall_time_s = float(time.perf_counter() - optimizer._wall_t0)
    history = getattr(optimizer, "_history", None)
    if history:
        final_wall_time_s = max(final_wall_time_s, float(history[-1].get("wall_time_s", 0.0)))
    return final_wall_time_s


def evaluate_and_record_final_exact_point(
    optimizer,
    result: dict,
    *,
    selected_best_exact: bool,
):
    """Select the final exact accepted point and append its history entry.

    Final artifacts must come from an exact accepted solve.  If the optimizer's
    nominal final point cannot be reconstructed, or if a prior exact accepted
    point has a lower exact cost, use that best exact point instead of a
    relaxed trial solve.
    """

    best_exact_params = getattr(optimizer, "_best_exact_params", None)
    best_exact_state = getattr(optimizer, "_best_exact_state", None)
    best_exact_residual = getattr(optimizer, "_best_exact_residual", None)
    best_exact_cost = float(getattr(optimizer, "_best_exact_cost", math.inf))

    final_key = optimizer._exact_cache_key(result["x"])
    res_final = optimizer._cached_exact_residual(cache_key=final_key)
    if (
        res_final is None
        and best_exact_params is not None
        and best_exact_residual is not None
        and final_key == optimizer._exact_cache_key(best_exact_params)
    ):
        res_final = np.asarray(best_exact_residual, dtype=float).reshape(-1)
        optimizer._remember_exact_residual(final_key, res_final)

    state_final = optimizer._cached_exact_state(result["x"])
    if state_final is None:
        try:
            state_final = optimizer._solve_exact_state(result["x"])
        except Exception as exc:
            if best_exact_params is not None and best_exact_residual is not None and np.isfinite(best_exact_cost):
                selected_best_exact = True
                result["x"] = np.asarray(best_exact_params, dtype=float).copy()
                final_key = optimizer._exact_cache_key(result["x"])
                res_final = np.asarray(best_exact_residual, dtype=float).reshape(-1)
                state_final = optimizer._best_exact_state_or_solve(result["x"], best_exact_state)
            else:
                raise RuntimeError(
                    "Final exact accepted-point solve failed and no prior exact "
                    "accepted point is available for final output."
                ) from exc

    if state_final is not None:
        optimizer._remember_exact_state(final_key, state_final)

    final_wall_time_s = final_history_wall_time(optimizer)
    entry_final = optimizer._history_entry_from_state_or_residual(
        state_final,
        res_final,
        wall_time_s=final_wall_time_s,
        cache_key=final_key,
    )
    cost_final = float(entry_final["cost"])
    qs_total_final = float(entry_final["qs_objective"])
    aspect_final = float(entry_final["aspect"])

    exact_improvement_tol = max(
        1.0e-14,
        1.0e-9
        * max(
            1.0,
            abs(cost_final) if np.isfinite(cost_final) else 1.0,
            abs(best_exact_cost) if np.isfinite(best_exact_cost) else 1.0,
        ),
    )
    if (
        best_exact_params is not None
        and best_exact_residual is not None
        and np.isfinite(best_exact_cost)
        and (not np.isfinite(cost_final) or best_exact_cost < cost_final - exact_improvement_tol)
    ):
        selected_best_exact = True
        result["x"] = np.asarray(best_exact_params, dtype=float).copy()
        final_key = optimizer._exact_cache_key(result["x"])
        res_final = np.asarray(best_exact_residual, dtype=float).reshape(-1)
        try:
            state_final = optimizer._best_exact_state_or_solve(result["x"], best_exact_state)
        except Exception as exc:
            raise RuntimeError(
                "Best exact accepted point was selected for final output, "
                "but its exact state could not be reconstructed."
            ) from exc
        final_wall_time_s = final_history_wall_time(optimizer)
        entry_final = optimizer._history_entry_from_state_or_residual(
            state_final,
            res_final,
            wall_time_s=final_wall_time_s,
            cache_key=final_key,
        )
        cost_final = float(entry_final["cost"])
        qs_total_final = float(entry_final["qs_objective"])
        aspect_final = float(entry_final["aspect"])

    if state_final is not None:
        optimizer._remember_exact_state(final_key, state_final)

    result["cost"] = float(cost_final)
    result["objective"] = float(2.0 * cost_final)
    optimizer._history.append(entry_final)
    return (
        state_final,
        entry_final,
        cost_final,
        qs_total_final,
        aspect_final,
        final_wall_time_s,
        selected_best_exact,
    )


def reset_run_state(optimizer, *, trace_callbacks: bool | None, iota_fn) -> None:
    """Reset mutable per-run caches, traces, and accepted-point bookkeeping."""

    optimizer._history = []
    optimizer._profile = {}
    optimizer._trial_residual_cache.clear()
    optimizer._exact_jacobian_cache = getattr(optimizer, "_exact_jacobian_cache", {})
    optimizer._exact_jacobian_cache.clear()
    optimizer._callback_trace_enabled = bool(
        optimizer._env_bool_override("VMEC_JAX_OPT_TRACE_CALLBACKS") if trace_callbacks is None else trace_callbacks
    )
    optimizer._callback_trace = []
    optimizer._callback_point_ids = {}
    optimizer._callback_previous_key = None
    optimizer._wall_t0 = time.perf_counter()
    optimizer._iota_fn = iota_fn
    optimizer._best_exact_params = optimizer._best_exact_state = optimizer._best_exact_residual = None
    optimizer._best_exact_cost = math.inf
    optimizer._exact_history_rejected_count = 0


def attach_run_private_payload(
    optimizer,
    result: dict,
    *,
    state_initial,
    state_final,
    history_dump: dict,
) -> dict:
    """Attach non-serializable state/profile payloads used by examples."""

    result["_state_initial"] = state_initial
    result["_state_final"] = state_final
    result["_profile"] = optimizer._profile_dump()
    result["_history_dump"] = history_dump
    return result


def initial_run_evaluation(optimizer, params0_arr: np.ndarray):
    """Evaluate and record the exact initial point for an optimization run."""

    residual0 = optimizer.residual_fun(params0_arr)
    state0, _ = optimizer._solve_exact_state(params0_arr, return_payload=True)
    entry0 = optimizer._history_entry_from_state_or_residual(
        state0,
        residual0,
        wall_time_s=0.0,
        cache_key=optimizer._exact_cache_key(params0_arr),
    )
    cost0 = float(entry0["cost"])
    qs_total0 = float(entry0["qs_objective"])
    aspect0 = float(entry0["aspect"])
    optimizer._history.append(entry0)
    optimizer._remember_best_exact_point(params0_arr, residual0, cost0, state=state0)
    return state0, entry0, cost0, qs_total0, aspect0


def cached_exact_residual(optimizer, params=None, *, cache_key: bytes | None = None) -> np.ndarray | None:
    """Return a same-point exact residual if already available."""

    if cache_key is None:
        if params is None:
            return None
        cache_key = optimizer._exact_cache_key(params)
    last_key = getattr(optimizer, "_last_jacobian_key", [None])[0]
    if last_key == cache_key and getattr(optimizer, "_last_jacobian_residual", None) is not None:
        return np.asarray(optimizer._last_jacobian_residual, dtype=float).reshape(-1)
    cache = getattr(optimizer, "_exact_residual_cache", None)
    if cache is not None and cache_key in cache:
        optimizer._profile_add("exact_residual_cache_hit", 0.0)
        return np.asarray(cache[cache_key], dtype=float).reshape(-1)
    return None


def cached_exact_state(optimizer, params):
    """Return a cached exact state for the parameter point if available."""

    cache_key = optimizer._exact_cache_key(params)
    if cache_key in optimizer._exact_cache:
        state = optimizer._exact_cache[cache_key][0]
        optimizer._remember_exact_state(cache_key, state)
        optimizer._profile_add("exact_cache_hit", 0.0)
        return state
    if cache_key in getattr(optimizer, "_exact_state_cache", {}):
        optimizer._profile_add("exact_state_cache_hit", 0.0)
        state = optimizer._exact_state_cache[cache_key]
        optimizer._remember_exact_state(cache_key, state)
        return state
    return None


def cached_trial_residual(optimizer, params) -> np.ndarray | None:
    """Return a cached trial residual and refresh its LRU position."""

    cache_key = optimizer._exact_cache_key(params)
    cache = getattr(optimizer, "_trial_residual_cache", None)
    if cache is None or cache_key not in cache:
        return None
    residual = cache.pop(cache_key)
    cache[cache_key] = residual
    optimizer._profile_add("trial_residual_cache_hit", 0.0)
    return np.asarray(residual, dtype=float)


def remember_trial_residual(optimizer, params, residual: np.ndarray) -> None:
    """Store a relaxed trial residual in the bounded LRU cache."""

    cache_key = optimizer._exact_cache_key(params)
    cache = getattr(optimizer, "_trial_residual_cache", None)
    if cache is None:
        optimizer._trial_residual_cache = OrderedDict()
        cache = optimizer._trial_residual_cache
    cache[cache_key] = np.asarray(residual, dtype=float).copy()
    cache.move_to_end(cache_key)
    max_size = max(0, int(getattr(optimizer, "_trial_residual_cache_max", 0)))
    while max_size and len(cache) > max_size:
        cache.popitem(last=False)
    if max_size == 0:
        cache.clear()


def boundary_from_params(optimizer, params):
    """Return VMEC-internal boundary coefficients for parameter values."""

    from ..._compat import jnp as _jnp

    boundary = apply_boundary_params(
        optimizer._boundary_input if optimizer._boundary_input is not None else optimizer._boundary,
        optimizer._specs,
        _jnp.asarray(params, dtype=_jnp.float64),
    )
    if optimizer._boundary_input is None:
        return boundary
    from ... import boundary as boundary_module

    return boundary_module.boundary_from_input_convention(
        boundary,
        optimizer._static.modes,
        lasym=bool(optimizer._static.cfg.lasym),
        apply_m1_constraint=False,
    )


def boundary_from_params_numpy(optimizer, params) -> BoundaryCoeffs:
    """Host-side boundary update for cache keys and non-AD logic."""

    boundary = apply_boundary_params_numpy(
        optimizer._boundary_input if optimizer._boundary_input is not None else optimizer._boundary,
        optimizer._specs,
        np.asarray(params, dtype=float),
    )
    if optimizer._boundary_input is None:
        return boundary
    from ... import boundary as boundary_module

    return boundary_module.boundary_from_input_convention(
        boundary,
        optimizer._static.modes,
        lasym=bool(optimizer._static.cfg.lasym),
        apply_m1_constraint=False,
    )


def boundary_input_from_params(optimizer, params) -> BoundaryCoeffs:
    """Boundary coefficients in VMEC input convention for ``params``."""

    from ..._compat import jnp as _jnp

    base_boundary = optimizer._boundary_input if optimizer._boundary_input is not None else optimizer._boundary
    return apply_boundary_params(
        base_boundary,
        optimizer._specs,
        _jnp.asarray(params, dtype=_jnp.float64),
    )


def initial_tangent_cache_key(optimizer, params):
    """Cache key for affine initial-state tangent maps."""

    from ...init_guess import _vmec_lflip_from_boundary

    try:
        boundary = optimizer._boundary_from_params_numpy(np.asarray(params, dtype=float))
    except Exception:
        try:
            boundary = optimizer._boundary_from_params(params)
        except Exception:
            return None
    try:
        lflip = _vmec_lflip_from_boundary(optimizer._static, boundary)
    except Exception:
        return None
    if lflip is None:
        lflip = False
    return (
        int(np.asarray(params).size),
        bool(lflip),
        bool(optimizer._boundary_input is not None),
        bool(optimizer._static.cfg.lasym),
        int(optimizer._static.cfg.ns),
        int(optimizer._static.modes.K),
    )


def indata_from_params(optimizer, params) -> InData:
    """Return a VMEC namelist with boundary coefficients updated for ``params``."""

    boundary_input = optimizer._boundary_input_from_params(params)
    indexed = {name: dict(values) for name, values in optimizer._indata.indexed.items()}
    indexed.update(indexed_boundary_maps_from_boundary(boundary_input, optimizer._static.modes))
    return InData(
        scalars=dict(optimizer._indata.scalars),
        indexed=indexed,
        source_path=optimizer._indata.source_path,
    )


def base_params_vector(optimizer) -> np.ndarray:
    """Return reference free coefficients aligned with ``optimizer._specs``."""

    boundary = optimizer._boundary_input if optimizer._boundary_input is not None else optimizer._boundary
    base = np.empty(len(optimizer._specs), dtype=float)
    for idx, spec in enumerate(optimizer._specs):
        if spec.kind == "rc":
            base[idx] = float(boundary.R_cos[spec.index])
        elif spec.kind == "rs":
            base[idx] = float(boundary.R_sin[spec.index])
        elif spec.kind == "zc":
            base[idx] = float(boundary.Z_cos[spec.index])
        elif spec.kind == "zs":
            base[idx] = float(boundary.Z_sin[spec.index])
        else:  # pragma: no cover - guarded by boundary_param_specs
            raise ValueError(f"Unknown boundary parameter kind '{spec.kind}'")
    return base
