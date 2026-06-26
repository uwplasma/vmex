"""Dynamic replay payload helpers for fixed-boundary discrete adjoints."""

from __future__ import annotations

from typing import Any

import numpy as np

from vmec_jax._compat import jax, jnp
from vmec_jax.state import pack_state
from vmec_jax.solvers.fixed_boundary.adjoint.replay_policy import (
    _DYNAMIC_REPLAY_SCALAR_TRACE_KEYS,
    _OPTIONAL_REPLAY_STEP_TRACE_KEYS,
    _REPLAY_STEP_TRACE_KEYS,
    _REPLAY_STEP_TRACE_STATIC_KEYS,
    _backend_is_accelerator,
    _dynamic_replay_bucket_len,
    _tridi_policy_cache_value,
)

def _looks_array_like(value) -> bool:
    return isinstance(value, np.ndarray) or (
        hasattr(value, "shape") and hasattr(value, "dtype")
    )


def _static_flags_from_replay_step_traces(step_traces: tuple[dict[str, Any], ...]):
    step_traces = tuple(_trace_with_replay_defaults(trace) for trace in step_traces)
    static_flags = {key: step_traces[0][key] for key in _REPLAY_STEP_TRACE_STATIC_KEYS}
    for trace in step_traces[1:]:
        for key, value in static_flags.items():
            other = trace[key]
            if _looks_array_like(value) or _looks_array_like(other):
                same = np.array_equal(np.asarray(other), np.asarray(value))
            else:
                same = other == value
            if not same:
                raise ValueError(f"Replay step trace key {key} must be constant across the tape for scan replay.")
    precond_jmax0 = int(step_traces[0]["precond_jmax"])
    precond_jmax_constant = all(int(trace["precond_jmax"]) == precond_jmax0 for trace in step_traces[1:])
    static_flags["precond_jmax"] = precond_jmax0 if precond_jmax_constant else None
    tridi_policy0 = step_traces[0].get("preconditioner_use_precomputed_tridi", None)
    if any(trace.get("preconditioner_use_precomputed_tridi", None) != tridi_policy0 for trace in step_traces[1:]):
        raise ValueError("Replay step trace preconditioner tridiagonal policy must be constant across scan replay.")
    static_flags["preconditioner_use_precomputed_tridi"] = tridi_policy0
    lax_tridi_policy0 = step_traces[0].get("preconditioner_use_lax_tridi", None)
    if any(trace.get("preconditioner_use_lax_tridi", None) != lax_tridi_policy0 for trace in step_traces[1:]):
        raise ValueError("Replay step trace lax tridiagonal policy must be constant across scan replay.")
    static_flags["preconditioner_use_lax_tridi"] = lax_tridi_policy0
    return static_flags


def _trace_with_replay_defaults(trace: dict[str, Any]) -> dict[str, Any]:
    """Return a trace with backward-compatible defaults for replay controls."""
    out = dict(trace)
    constraint_update = out.get("constraint_cache_update", False)
    out.setdefault("constraint_cache_update", constraint_update)
    out.setdefault("precond_cache_update", constraint_update)
    return out


def _replay_stack_values_fn():
    """Return a tree-map stack function that preserves accelerator-resident arrays."""

    use_device_stack = _backend_is_accelerator(jax.default_backend())
    jax_array_type = getattr(jax, "Array", ())

    def _as_stack_array(x):
        if use_device_stack:
            if jax_array_type and isinstance(x, jax_array_type):
                return x
            return jnp.asarray(x)
        if isinstance(x, np.ndarray):
            return x
        return np.asarray(x)

    def _stack_values(*xs):
        arrays = [_as_stack_array(x) for x in xs]
        return jnp.stack(arrays, axis=0) if use_device_stack else np.stack(arrays, axis=0)

    return _stack_values


def _stack_replay_step_traces(step_traces: tuple[dict[str, Any], ...]):
    step_traces = tuple(_trace_with_replay_defaults(trace) for trace in step_traces)
    optional_keys = _OPTIONAL_REPLAY_STEP_TRACE_KEYS
    optional_present: dict[str, list[bool]] = {
        key: [trace.get(key, None) is not None for trace in step_traces]
        for key in optional_keys
    }
    active_optional_keys = set()
    for key, present in optional_present.items():
        if all(present):
            active_optional_keys.add(key)
        elif any(present):
            raise ValueError(
                f"Replay requires optional trace key {key} to be present "
                "on every active trace or none."
            )
    filtered = tuple(
        {
            key: trace[key]
            for key in _REPLAY_STEP_TRACE_KEYS
            if key not in optional_keys or key in active_optional_keys
        }
        for trace in step_traces
    )
    stacked = jax.tree_util.tree_map(_replay_stack_values_fn(), *filtered)
    static_flags = _static_flags_from_replay_step_traces(step_traces)
    return stacked, static_flags


def _replay_values_equal(a, b) -> bool:
    if a is b:
        return True
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b))
    if hasattr(a, "__dict__") and hasattr(b, "__dict__"):
        a_dict = vars(a)
        b_dict = vars(b)
        if a_dict.keys() != b_dict.keys():
            return False
        return all(_replay_values_equal(a_dict[key], b_dict[key]) for key in a_dict)
    try:
        return np.array_equal(np.asarray(a), np.asarray(b))
    except Exception:
        return a == b


def _build_dynamic_replay_payload(
    step_traces: tuple[dict[str, Any], ...],
    static_flags: dict[str, Any],
    *,
    store_base_carries: bool = True,
):
    step_traces = tuple(_trace_with_replay_defaults(trace) for trace in step_traces)
    stack_values = _replay_stack_values_fn()

    dynamic_static_flags = dict(static_flags)
    dynamic_static_flags["layout"] = step_traces[0]["state_pre"].layout
    constant_candidates = (
        "wout_like",
        "trig",
        "w_mode_mn",
        "constraint_tcon0",
        "constraint_precond_diag",
        "constraint_tcon",
        "constraint_precond_active",
        "constraint_tcon_active",
        "constraint_rcon0",
        "constraint_zcon0",
    )
    varying_keys = [
        "time_step",
        "flip_sign",
        "reset_inv_tau",
        "constraint_cache_update",
        "precond_cache_update",
        "zero_m1",
        *_DYNAMIC_REPLAY_SCALAR_TRACE_KEYS,
    ]
    forced_varying_optional_keys = {
        "constraint_precond_active",
        "constraint_tcon_active",
    }
    for key in forced_varying_optional_keys:
        present = [trace.get(key, None) is not None for trace in step_traces]
        if all(present):
            varying_keys.append(key)
        elif any(present):
            raise ValueError(
                f"Dynamic replay requires {key} to be present on every active trace or none."
            )
    for optional_key in _OPTIONAL_REPLAY_STEP_TRACE_KEYS:
        optional_present = [trace.get(optional_key, None) is not None for trace in step_traces]
        if optional_key in forced_varying_optional_keys:
            continue
        if optional_key in constant_candidates:
            continue
        if all(optional_present):
            varying_keys.append(optional_key)
        elif any(optional_present):
            raise ValueError(
                f"Dynamic replay requires {optional_key} to be present on every active trace or none."
            )
    for key in constant_candidates:
        if key in forced_varying_optional_keys:
            continue
        present = [trace.get(key, None) is not None for trace in step_traces]
        if not any(present):
            continue
        if not all(present):
            raise ValueError(
                f"Dynamic replay requires {key} to be present on every active trace or none."
            )
        first = step_traces[0][key]
        if all(_replay_values_equal(trace[key], first) for trace in step_traces[1:]):
            dynamic_static_flags[key] = first
        else:
            varying_keys.append(key)
    filtered = tuple({key: trace[key] for key in varying_keys} | {"active": True} for trace in step_traces)
    target_len = _dynamic_replay_bucket_len(len(filtered))
    if target_len > len(filtered):
        pad_trace = dict(filtered[-1])
        pad_trace["active"] = False
        filtered = filtered + tuple(dict(pad_trace) for _ in range(target_len - len(filtered)))
    stacked = {
        key: stack_values(*(trace[key] for trace in filtered))
        for key in filtered[0]
    }
    initial_carry = _dynamic_replay_initial_carry(step_traces[0])
    stacked_base_carries = None
    if store_base_carries:
        base_carries = (initial_carry,) + tuple(_dynamic_replay_initial_carry(trace) for trace in step_traces[1:])
        if target_len > len(base_carries):
            pad_carry = base_carries[-1]
            base_carries = base_carries + (pad_carry,) * (target_len - len(base_carries))
        stacked_base_carries = jax.tree_util.tree_map(stack_values, *base_carries)
    return stacked, dynamic_static_flags, initial_carry, stacked_base_carries


def _stacked_trace_signature(stacked) -> tuple[tuple[tuple[int, ...], str], ...]:
    leaves = jax.tree_util.tree_leaves(stacked)
    signature = []
    for leaf in leaves:
        shape = tuple(getattr(leaf, "shape", np.shape(leaf)))
        dtype = getattr(leaf, "dtype", None)
        if dtype is None:
            dtype = np.asarray(leaf).dtype
        signature.append((shape, np.dtype(dtype).str))
    return tuple(signature)


def _stacked_leading_axis_size(stacked) -> int | None:
    sizes: set[int] = set()
    for leaf in jax.tree_util.tree_leaves(stacked):
        shape = tuple(getattr(leaf, "shape", np.shape(leaf)))
        if not shape:
            return None
        sizes.add(int(shape[0]))
    if len(sizes) != 1:
        return None
    return next(iter(sizes))


def _dynamic_replay_cache_key(*, static, stacked, static_flags, stacked_base_carries=None) -> tuple[Any, ...]:
    key = (
        id(static),
        bool(static_flags["apply_lforbal"]),
        bool(static_flags["include_edge_residual"]),
        bool(static_flags["apply_m1_constraints"]),
        bool(static_flags["limit_update_rms"]),
        bool(static_flags["limit_dt_from_force"]),
        bool(static_flags["vmec2000_control"]),
        bool(static_flags["divide_by_scalxc_for_update"]),
        int(static_flags["signgs"]),
        int(static_flags["precond_jmax"]),
        _tridi_policy_cache_value(static_flags.get("preconditioner_use_precomputed_tridi", None)),
        _tridi_policy_cache_value(static_flags.get("preconditioner_use_lax_tridi", None)),
        _stacked_trace_signature(stacked),
    )
    if stacked_base_carries is not None:
        key += (_stacked_trace_signature(stacked_base_carries),)
    return key


def _dynamic_basepoint_payload_shapes_match(stacked, stacked_base_carries) -> bool:
    trace_len = _stacked_leading_axis_size(stacked)
    carry_len = _stacked_leading_axis_size(stacked_base_carries)
    return trace_len is not None and trace_len == carry_len


def _dynamic_replay_supported(
    *,
    tape: ResidualCheckpointTape,
    rebuild_preconditioner: bool,
) -> bool:
    if (not rebuild_preconditioner) or (not tape.step_traces):
        return False
    if tape.step_trace_static_flags is not None and tape.step_trace_static_flags.get("precond_jmax") is None:
        return False
    for freeb_key in ("freeb_bsqvac_half", "freeb_pres_scale"):
        freeb_present = [trace.get(freeb_key, None) is not None for trace in tape.step_traces]
        if any(freeb_present) and not all(freeb_present):
            return False
    return all(_dynamic_replay_trace_supported(trace) for trace in tape.step_traces)


def _dynamic_replay_trace_supported(trace) -> bool:
    return (
        trace.get("branch") == "strict_update"
        and trace.get("step_status") == "momentum"
        and trace.get("restart_reason") == "none"
        and trace.get("restart_path") == "momentum_accept"
    )


def _dynamic_restart_trace_supported(trace) -> bool:
    return (
        trace.get("branch") == "strict_update"
        and trace.get("step_status") in ("restart_bad_progress", "restart_bad_jacobian")
        and trace.get("restart_path") in ("catastrophic_growth", "catastrophic_nonfinite")
    )


def _restart_carry_tangents(carry_tangents):
    packed_state_tangent, inv_tau_tangent, fsq_prev_tangent, *velocity_tangents = carry_tangents
    zero_like = lambda arr: jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), arr)
    return (
        packed_state_tangent,
        zero_like(inv_tau_tangent),
        fsq_prev_tangent,
        *(zero_like(arr) for arr in velocity_tangents),
    )


def _carry_tangents_with_zero_aux(state_tangents, carry0):
    def _zeros_like(arr):
        return jax.tree_util.tree_map(
            lambda x: jnp.zeros((state_tangents.shape[0],) + jnp.asarray(x).shape, dtype=jnp.asarray(x).dtype),
            arr,
        )

    return (state_tangents,) + tuple(_zeros_like(arr) for arr in carry0[1:])


def _dynamic_replay_initial_carry(trace):
    trace = _trace_with_replay_defaults(trace)
    packed_state = jnp.asarray(pack_state(trace["state_pre"]))
    dtype = packed_state.dtype

    def _arr(name: str):
        value = trace.get(name)
        if value is None:
            return jnp.zeros_like(jnp.asarray(trace["vRcc_before"], dtype=dtype))
        return jnp.asarray(value, dtype=dtype)

    def _lam_prec_from_trace():
        value = trace.get("lam_prec", None)
        if value is None:
            return jnp.zeros_like(_arr("vLsc_before"))
        return jnp.asarray(value, dtype=dtype)

    def _precond_mats_from_trace():
        value = trace.get("precond_mats", None)
        if value is None:
            return jnp.zeros_like(_lam_prec_from_trace())
        return jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=dtype), value)

    return (
        packed_state,
        jnp.asarray(trace["inv_tau_before"], dtype=dtype),
        jnp.asarray(trace["fsq_prev_before"], dtype=dtype),
        _arr("vRcc_before"),
        _arr("vRss_before"),
        _arr("vRsc_before"),
        _arr("vRcs_before"),
        _arr("vZsc_before"),
        _arr("vZcs_before"),
        _arr("vZcc_before"),
        _arr("vZss_before"),
        _arr("vLsc_before"),
        _arr("vLcs_before"),
        _arr("vLcc_before"),
        _arr("vLss_before"),
        *_constraint_cache_from_trace(trace, dtype=dtype),
        _lam_prec_from_trace(),
        _precond_mats_from_trace(),
    )


def _constraint_precond_diag_as_tuple(value):
    if value is None or isinstance(value, tuple):
        return value
    arr = jnp.asarray(value)
    if len(arr.shape) >= 1 and int(arr.shape[0]) == 2:
        return (arr[0], arr[1])
    return value


def _constraint_cache_from_trace(trace, *, dtype):
    diag = _constraint_precond_diag_as_tuple(trace.get("constraint_precond_diag", None))
    tcon_value = trace.get("constraint_tcon", None)
    if diag is None:
        template = (
            jnp.asarray(tcon_value, dtype=dtype)
            if tcon_value is not None
            else jnp.asarray(trace["inv_tau_before"], dtype=dtype)
        )
        ard1 = jnp.zeros((int(template.shape[0]),), dtype=dtype)
        azd1 = jnp.zeros_like(ard1)
    else:
        ard1 = jnp.asarray(diag[0], dtype=dtype)
        azd1 = jnp.asarray(diag[1], dtype=dtype)
    if tcon_value is None:
        tcon = jnp.zeros_like(ard1)
    else:
        tcon = jnp.asarray(tcon_value, dtype=dtype)
    return ard1, azd1, tcon


def _dynamic_replay_value(trace, static_flags, key: str, default=None):
    if isinstance(trace, dict) and key in trace:
        return trace[key]
    return static_flags.get(key, default)


def _dynamic_replay_values(trace, static_flags, *keys: str):
    return tuple(_dynamic_replay_value(trace, static_flags, key) for key in keys)


def _dynamic_constraint_cache_current(
    trace,
    static_flags,
    ard1_before,
    azd1_before,
    tcon_before,
):
    constraint_precond_active = _dynamic_replay_value(trace, static_flags, "constraint_precond_active", False)
    constraint_tcon_active = _dynamic_replay_value(trace, static_flags, "constraint_tcon_active", False)
    precond_active = jnp.asarray(constraint_precond_active, dtype=bool)
    tcon_active = jnp.asarray(constraint_tcon_active, dtype=bool)
    return (
        constraint_precond_active,
        constraint_tcon_active,
        (
            jnp.where(precond_active, jnp.asarray(ard1_before), jnp.zeros_like(jnp.asarray(ard1_before))),
            jnp.where(precond_active, jnp.asarray(azd1_before), jnp.zeros_like(jnp.asarray(azd1_before))),
        ),
        jnp.where(tcon_active, jnp.asarray(tcon_before), jnp.zeros_like(jnp.asarray(tcon_before))),
    )


def _dynamic_preconditioner_cache_current(refreshed, trace, lam_prec_before, mats_before):
    constraint_cache_update = _dynamic_replay_value(trace, {}, "constraint_cache_update", False)
    precond_cache_update = _dynamic_replay_value(trace, {}, "precond_cache_update", constraint_cache_update)
    update_cache = jnp.asarray(precond_cache_update, dtype=bool)
    lam_prec = jnp.where(update_cache, jnp.asarray(refreshed["lam_prec"]), jnp.asarray(lam_prec_before))

    def _merge_mats(refreshed_value, cached_value):
        if isinstance(refreshed_value, dict):
            cached_dict = cached_value if isinstance(cached_value, dict) else {}
            keys = tuple(sorted(set(refreshed_value) | set(cached_dict)))
            merged = {}
            for key in keys:
                if key in refreshed_value:
                    refreshed_leaf = refreshed_value[key]
                else:
                    refreshed_leaf = cached_dict[key]
                if key in cached_dict:
                    cached_leaf = cached_dict[key]
                else:
                    cached_leaf = refreshed_value[key]
                merged[key] = _merge_mats(refreshed_leaf, cached_leaf)
            return merged
        if isinstance(cached_value, dict):
            return {
                key: _merge_mats(
                    refreshed_value,
                    cached_leaf,
                )
                for key, cached_leaf in sorted(cached_value.items())
            }
        return jnp.where(
            update_cache,
            jnp.asarray(refreshed_value),
            jnp.asarray(cached_value),
        )

    mats = _merge_mats(refreshed["mats"], mats_before)
    return {
        "lam_prec": lam_prec,
        "mats": mats,
        "jmax": refreshed["jmax"],
        "w_mode_mn": refreshed["w_mode_mn"],
    }


def _dynamic_safe_dt_from_force_arrays(
    *,
    dt_nominal,
    max_coeff_delta_rms,
    frcc,
    frss,
    fzsc,
    fzcs,
    flsc,
    flcs,
    frsc,
    frcs,
    fzcc,
    fzss,
    flcc,
    flss,
):
    dtype = jnp.asarray(frcc).dtype
    dt_nominal = jnp.asarray(dt_nominal, dtype=dtype)
    max_coeff_delta_rms = jnp.asarray(max_coeff_delta_rms, dtype=dtype)
    rms = jnp.sqrt(
        jnp.mean(
            jnp.asarray(frcc) * jnp.asarray(frcc)
            + jnp.asarray(frss) * jnp.asarray(frss)
            + jnp.asarray(frsc) * jnp.asarray(frsc)
            + jnp.asarray(frcs) * jnp.asarray(frcs)
            + jnp.asarray(fzsc) * jnp.asarray(fzsc)
            + jnp.asarray(fzcs) * jnp.asarray(fzcs)
            + jnp.asarray(fzcc) * jnp.asarray(fzcc)
            + jnp.asarray(fzss) * jnp.asarray(fzss)
            + jnp.asarray(flsc) * jnp.asarray(flsc)
            + jnp.asarray(flcs) * jnp.asarray(flcs)
            + jnp.asarray(flcc) * jnp.asarray(flcc)
            + jnp.asarray(flss) * jnp.asarray(flss)
        )
    )
    dt_lim = jnp.sqrt(max_coeff_delta_rms / jnp.maximum(rms, jnp.asarray(1.0e-30, dtype=dtype)))
    dt_eff = jnp.where(
        jnp.isfinite(rms) & (rms > 0.0),
        jnp.minimum(dt_nominal, dt_lim),
        dt_nominal,
    )
    return jnp.maximum(dt_eff, jnp.asarray(1.0e-12, dtype=dtype))


def _dynamic_fsq1_from_force_channels(
    *,
    state_pre,
    static,
    vmec2000_control: bool,
    frzl_pre,
):
    from vmec_jax.kernels.residue import vmec_gcx2_from_tomnsps, vmec_rz_norm_from_state

    s = jnp.asarray(static.s)
    gcr2_p, gcz2_p, gcl2_p = vmec_gcx2_from_tomnsps(
        frzl=frzl_pre,
        lconm1=bool(getattr(static.cfg, "lconm1", True)),
        apply_m1_constraints=False,
        include_edge=True,
        apply_scalxc=False,
        s=s,
    )
    rz_norm = vmec_rz_norm_from_state(
        state=state_pre,
        static=static,
        s=s,
        apply_scalxc=False,
        ns_min=0,
        ns_max=int(s.shape[0]),
    )
    nonzero_norm = rz_norm != 0.0
    safe_rz_norm = jnp.where(nonzero_norm, rz_norm, jnp.asarray(1.0, dtype=rz_norm.dtype))
    f_norm1 = jnp.where(nonzero_norm, 1.0 / safe_rz_norm, jnp.asarray(0.0, dtype=rz_norm.dtype))
    fsqr1 = gcr2_p * f_norm1
    fsqz1 = gcz2_p * f_norm1
    delta_s = jnp.asarray(s[1] - s[0], dtype=jnp.asarray(gcr2_p).dtype) if int(s.shape[0]) >= 2 else jnp.asarray(1.0, dtype=jnp.asarray(gcr2_p).dtype)
    if bool(vmec2000_control):
        gcl2_full = jnp.sum(jnp.asarray(frzl_pre.flsc)[1:] ** 2)
        if frzl_pre.flcs is not None:
            gcl2_full = gcl2_full + jnp.sum(jnp.asarray(frzl_pre.flcs)[1:] ** 2)
        if getattr(frzl_pre, "flcc", None) is not None:
            gcl2_full = gcl2_full + jnp.sum(jnp.asarray(frzl_pre.flcc)[1:] ** 2)
        if getattr(frzl_pre, "flss", None) is not None:
            gcl2_full = gcl2_full + jnp.sum(jnp.asarray(frzl_pre.flss)[1:] ** 2)
        fsql1 = gcl2_full * delta_s
    else:
        fsql1 = gcl2_p * delta_s
    return fsqr1 + fsqz1 + fsql1
