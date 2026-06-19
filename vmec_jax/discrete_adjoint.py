"""Utilities for the discrete-adjoint recovery path.

The first step is a structured view of the existing fixed-boundary residual
iteration history. This keeps the initial refactor narrow: no solver behavior
changes, only a stable extraction layer over the primal trace data already
recorded in diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Any

import numpy as np

from ._compat import jax, jnp
from .state import pack_state, unpack_state

from vmec_jax.solvers.fixed_boundary.adjoint import replay_policy as _replay_policy
from vmec_jax.solvers.fixed_boundary.adjoint.replay_policy import (
    _CHECKPOINT_TAPE_DYNAMIC_BASEPOINT_SCAN_CACHE,
    _CHECKPOINT_TAPE_DYNAMIC_BASEPOINT_VJP_SCAN_CACHE,
    _CHECKPOINT_TAPE_DYNAMIC_SCAN_CACHE,
    _CHECKPOINT_TAPE_SCAN_CACHE,
    _DIRECT_TAPE_TIMING_KEYS,
    _DYNAMIC_REPLAY_SCALAR_TRACE_KEYS,  # noqa: F401 - private compatibility facade
    _DynamicBasepointScanRunner,
    _DEFAULT_REPLAY_COLUMN_TARGET_MB,
    _OPTIONAL_REPLAY_STEP_TRACE_KEYS,  # noqa: F401 - private compatibility facade
    _REPLAY_STEP_TRACE_KEYS,  # noqa: F401 - private compatibility facade
    _REPLAY_STEP_TRACE_STATIC_KEYS,  # noqa: F401 - private compatibility facade
    _TRACE_OVERRIDE_UNSET,
    _backend_is_accelerator,  # noqa: F401 - private compatibility facade
    _dynamic_replay_bucket_default,  # noqa: F401 - private compatibility facade
    _dynamic_replay_bucket_len,  # noqa: F401 - private compatibility facade
    _dynamic_replay_bucket_size,  # noqa: F401 - private compatibility facade
    _dynamic_replay_mode,
    _get_replay_scan_runner,
    _jvp_only_exact_tape_basepoint_carries_enabled,
    _lru_cache_get,  # noqa: F401 - private compatibility facade
    _lru_cache_put,  # noqa: F401 - private compatibility facade
    _put_replay_scan_runner,
    _record_replay_jvp_columns_chunking,
    _record_replay_jvp_columns_path,
    _replay_column_chunk_override,
    _trace_preconditioner_use_lax_tridi,
    _trace_preconditioner_use_precomputed_tridi,
    _tridi_policy_cache_value,
    clear_replay_scan_caches,  # noqa: F401 - public compatibility facade
    replay_scan_cache_diagnostics,
)
from vmec_jax.solvers.fixed_boundary.adjoint.replay_payload import (
    _build_dynamic_replay_payload,
    _carry_tangents_with_zero_aux,
    _constraint_cache_from_trace,  # noqa: F401 - private compatibility facade
    _constraint_precond_diag_as_tuple,  # noqa: F401 - private compatibility facade
    _dynamic_basepoint_payload_shapes_match,
    _dynamic_constraint_cache_current,
    _dynamic_fsq1_from_force_channels,
    _dynamic_preconditioner_cache_current,
    _dynamic_replay_cache_key,
    _dynamic_replay_initial_carry,
    _dynamic_replay_supported,
    _dynamic_replay_trace_supported,
    _dynamic_replay_value,
    _dynamic_replay_values,
    _dynamic_restart_trace_supported,
    _dynamic_safe_dt_from_force_arrays,
    _looks_array_like,  # noqa: F401 - private compatibility facade
    _replay_values_equal,  # noqa: F401 - private compatibility facade
    _restart_carry_tangents,
    _stack_replay_step_traces,
    _stacked_leading_axis_size,  # noqa: F401 - private compatibility facade
    _stacked_trace_signature,
    _static_flags_from_replay_step_traces,
    _trace_with_replay_defaults,  # noqa: F401 - private compatibility facade
)


def _scan_cache_limit() -> int:
    return _replay_policy._scan_cache_limit()


def _replay_column_chunk_default(*, tape, tangents) -> int | None:
    original = _replay_policy._DEFAULT_REPLAY_COLUMN_TARGET_MB
    _replay_policy._DEFAULT_REPLAY_COLUMN_TARGET_MB = _DEFAULT_REPLAY_COLUMN_TARGET_MB
    try:
        return _replay_policy._replay_column_chunk_default(tape=tape, tangents=tangents)
    finally:
        _replay_policy._DEFAULT_REPLAY_COLUMN_TARGET_MB = original


@dataclass(frozen=True)
class ResidualIterationTrace:
    """Structured view of one fixed-boundary residual solve history."""

    iter2: np.ndarray
    step_status: np.ndarray
    restart_reason: np.ndarray
    pre_restart_reason: np.ndarray
    time_step: np.ndarray
    dt_eff: np.ndarray
    update_rms: np.ndarray
    include_edge: np.ndarray
    zero_m1: np.ndarray
    fsq_curr: np.ndarray
    fsq_try: np.ndarray
    fsq_prev: np.ndarray
    r00: np.ndarray
    z00: np.ndarray
    wb: np.ndarray
    wp: np.ndarray
    w_vmec: np.ndarray
    state_advanced: np.ndarray


@dataclass(frozen=True)
class ResidualCheckpointTape:
    """Replay-friendly checkpoints from repeated one-step residual solves."""

    final_packed_state: Any
    packed_states: np.ndarray
    trace: ResidualIterationTrace
    resume_states: tuple[dict[str, Any] | None, ...]
    step_traces: tuple[dict[str, Any], ...]
    stacked_step_traces: Any | None = None
    step_trace_static_flags: dict[str, Any] | None = None
    dynamic_initial_carry: Any | None = None
    dynamic_base_carries_stacked: Any | None = None
    diagnostics: dict[str, Any] | None = None
    jvp_only: bool = False


_RESIDUAL_TRACE_DIAGNOSTIC_FIELDS = (
    ("iter2", "iter2_history", int),
    ("step_status", "step_status_history", object),
    ("restart_reason", "restart_reason_history", object),
    ("pre_restart_reason", "pre_restart_reason_history", object),
    ("time_step", "time_step_history", float),
    ("dt_eff", "dt_eff_history", float),
    ("update_rms", "update_rms_history", float),
    ("include_edge", "include_edge_history", int),
    ("zero_m1", "zero_m1_history", int),
    ("fsq_curr", "w_curr_history", float),
    ("fsq_try", "w_try_history", float),
    ("fsq_prev", "fsq_prev_history", float),
    ("r00", "r00_history", float),
    ("z00", "z00_history", float),
    ("wb", "wb_history", float),
    ("wp", "wp_history", float),
    ("w_vmec", "w_vmec_history", float),
)
_RESIDUAL_TRACE_OUTPUT_FIELDS = _RESIDUAL_TRACE_DIAGNOSTIC_FIELDS + (("state_advanced", None, bool),)


def _empty_trace() -> ResidualIterationTrace:
    return ResidualIterationTrace(
        **{name: np.zeros((0,), dtype=dtype) for name, _diagnostic_key, dtype in _RESIDUAL_TRACE_OUTPUT_FIELDS}
    )


def _array_from_diag(diagnostics: dict[str, Any], key: str, *, dtype=None) -> np.ndarray:
    value = diagnostics.get(key, np.zeros((0,), dtype=float))
    arr = np.asarray(value)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


_FINGERPRINT_TRACE_KEYS = (
    "branch",
    "step_status",
    "restart_reason",
    "pre_restart_reason",
    "restart_path",
    "include_edge_residual",
    "apply_m1_constraints",
    "vmec2000_control",
    "limit_dt_from_force",
    "zero_m1",
    "precond_jmax",
    "preconditioner_use_precomputed_tridi",
    "preconditioner_use_lax_tridi",
)

_FINGERPRINT_DIAGNOSTIC_KEYS = (
    "step_status_history",
    "restart_reason_history",
    "pre_restart_reason_history",
    "restart_path_history",
    "include_edge_history",
    "zero_m1_history",
    "state_advanced_history",
    "freeb_ivac_history",
    "freeb_ivacskip_history",
    "freeb_full_update_history",
    "freeb_nestor_reused_history",
    "freeb_nestor_source_reused_history",
    "freeb_nestor_provider_allows_source_reuse_history",
    "freeb_nestor_trial_reused_history",
    "freeb_nestor_trial_failed_history",
)


def _fingerprint_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (str, bytes)):
        return value.decode() if isinstance(value, bytes) else value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    arr = np.asarray(value)
    if arr.ndim == 0:
        return _fingerprint_scalar(arr.item())
    if arr.dtype.kind in ("b",):
        return tuple(bool(x) for x in arr.reshape(-1).tolist())
    if arr.dtype.kind in ("i", "u"):
        return tuple(int(x) for x in arr.reshape(-1).tolist())
    if arr.dtype.kind in ("f",):
        return tuple(float(x) for x in arr.reshape(-1).tolist())
    return tuple(_fingerprint_scalar(x) for x in arr.reshape(-1).tolist())


def residual_branch_fingerprint(result_or_diagnostics: Any) -> tuple[tuple[str, Any], ...]:
    """Return categorical solver-control data for same-branch FD checks.

    The fingerprint intentionally includes controller decisions, restart paths,
    trace branches, free-boundary cadence/reuse flags, and preconditioner policy,
    but not residual magnitudes, wall times, or floating-point state values.  It
    is meant to guard local derivative comparisons: matching fingerprints support
    a same-branch AD-vs-FD claim; differing fingerprints mean the comparison is
    across an adaptive branch switch and should be skipped or treated separately.
    """

    diagnostics = getattr(result_or_diagnostics, "diagnostics", result_or_diagnostics)
    if not isinstance(diagnostics, dict):
        raise TypeError("result_or_diagnostics must be a diagnostics dict or an object with diagnostics")

    pieces: list[tuple[str, Any]] = []
    for key in _FINGERPRINT_DIAGNOSTIC_KEYS:
        if key in diagnostics:
            pieces.append((key, _fingerprint_scalar(diagnostics.get(key))))

    freeb = diagnostics.get("free_boundary")
    if isinstance(freeb, dict):
        for key in ("enabled", "nvacskip", "nvskip0", "ivac", "ivacskip", "couple_edge", "provider_kind"):
            if key in freeb:
                pieces.append((f"free_boundary.{key}", _fingerprint_scalar(freeb.get(key))))

    traces = diagnostics.get("adjoint_step_trace", ())
    trace_fingerprints = []
    for trace in tuple(traces or ()):
        if not isinstance(trace, dict):
            continue
        trace_fingerprints.append(
            tuple(
                (key, _fingerprint_scalar(trace.get(key)))
                for key in _FINGERPRINT_TRACE_KEYS
                if key in trace
            )
        )
    pieces.append(("adjoint_step_trace", tuple(trace_fingerprints)))
    return tuple(pieces)


def residual_iteration_trace_from_result(result) -> ResidualIterationTrace:
    """Extract a compact, typed residual-iteration trace from a solver result."""
    diagnostics = getattr(result, "diagnostics", None)
    if not isinstance(diagnostics, dict):
        raise TypeError("result.diagnostics must be a dict")

    fields = {
        name: _array_from_diag(diagnostics, diagnostic_key, dtype=dtype)
        for name, diagnostic_key, dtype in _RESIDUAL_TRACE_DIAGNOSTIC_FIELDS
    }
    lengths = {
        int(arr.shape[0])
        for arr in fields.values()
        if arr.ndim >= 1 and arr.shape[0] > 0
    }
    if len(lengths) > 1:
        raise ValueError(f"inconsistent residual trace lengths: {sorted(lengths)}")

    rejected = np.isin(
        fields["step_status"],
        np.asarray(["rejected", "restart_bad_progress", "restart_bad_jacobian"], dtype=object),
    )
    fields["state_advanced"] = ~rejected
    return ResidualIterationTrace(**fields)


def _compact_tape_diagnostics(diagnostics: dict[str, Any]) -> dict[str, Any]:
    """Keep lightweight solver diagnostics needed by exact-tape profilers."""
    out: dict[str, Any] = {}
    timing = diagnostics.get("timing")
    if isinstance(timing, dict):
        compact_timing: dict[str, float | int] = {}
        for key, value in timing.items():
            if isinstance(value, (bool, str)):
                continue
            try:
                if key == "iterations":
                    compact_timing[str(key)] = int(value)
                else:
                    compact_timing[str(key)] = float(value)
            except Exception:
                continue
        out["timing"] = compact_timing
    for key in ("converged", "converged_iter", "final_fsq", "final_fsqz", "final_fsqr", "final_fsql"):
        if key not in diagnostics:
            continue
        value = diagnostics[key]
        try:
            if isinstance(value, (bool, np.bool_)):
                out[key] = bool(value)
            elif isinstance(value, (int, np.integer)):
                out[key] = int(value)
            elif isinstance(value, (float, np.floating)):
                out[key] = float(value)
        except Exception:
            continue
    return out


def concat_residual_iteration_traces(traces: list[ResidualIterationTrace]) -> ResidualIterationTrace:
    """Concatenate per-call residual traces into one longer trace."""
    if not traces:
        return _empty_trace()

    def _cat(name: str) -> np.ndarray:
        parts = [np.asarray(getattr(trace, name)) for trace in traces]
        return np.concatenate(parts, axis=0)

    return ResidualIterationTrace(
        **{name: _cat(name).astype(dtype, copy=False) for name, _diagnostic_key, dtype in _RESIDUAL_TRACE_OUTPUT_FIELDS}
    )


def build_residual_checkpoint_tape(
    state0,
    static,
    *,
    indata,
    signgs: int,
    max_iter: int,
    ftol: float | None = None,
    step_size: float = 1.0,
    resume_state_mode: str = "minimal",
    light_history: bool = True,
    store_packed_states: bool = True,
    store_trace: bool = True,
    store_resume_states: bool = True,
    solver_kwargs: dict[str, Any] | None = None,
) -> ResidualCheckpointTape:
    """Replay the residual solver in one-step chunks and collect checkpoints."""
    solve_kwargs = dict(solver_kwargs or {})
    for key, value in (("indata", indata), ("signgs", int(signgs)), ("ftol", ftol), ("step_size", float(step_size))):
        solve_kwargs.setdefault(key, value)
    solve_kwargs["light_history"] = False if store_trace else bool(light_history)
    # Multi-step replay needs the full resume checkpoint carried between steps.
    solve_kwargs["resume_state_mode"] = "full"
    state = state0
    resume_state = None
    traces: list[ResidualIterationTrace] = []
    packed_states: list[np.ndarray] = []
    resume_states: list[dict[str, Any] | None] = []
    step_traces: list[dict[str, Any]] = []

    for _ in range(int(max_iter)):
        result = replay_residual_checkpoint_step(
            state,
            static,
            resume_state=resume_state,
            solve_kwargs=solve_kwargs,
        )
        state = result.state
        if store_packed_states:
            packed_states.append(np.asarray(pack_state(state), dtype=float))
        if store_trace:
            traces.append(residual_iteration_trace_from_result(result))
        resume_state = result.diagnostics.get("resume_state")
        if store_resume_states:
            resume_states.append(resume_state)
        step_traces.extend(list(result.diagnostics.get("adjoint_step_trace", [])))
        if bool(result.diagnostics.get("converged", False)):
            break

    final_packed_state = np.asarray(pack_state(state), dtype=float)
    if packed_states:
        packed_states_arr = np.stack(packed_states, axis=0)
    else:
        packed_states_arr = np.zeros((0, int(state0.layout.size)), dtype=float)

    trace = concat_residual_iteration_traces(traces)

    stacked_step_traces = None
    step_trace_static_flags = None
    dynamic_base_carries_stacked = None
    if step_traces:
        stacked_step_traces, step_trace_static_flags = _stack_replay_step_traces(tuple(step_traces))
    return ResidualCheckpointTape(
        final_packed_state=final_packed_state,
        packed_states=packed_states_arr,
        trace=trace,
        resume_states=tuple(resume_states),
        step_traces=tuple(step_traces),
        stacked_step_traces=stacked_step_traces,
        step_trace_static_flags=step_trace_static_flags,
        dynamic_base_carries_stacked=dynamic_base_carries_stacked,
    )


def build_residual_checkpoint_tape_direct(
    state0,
    static,
    *,
    indata,
    signgs: int,
    max_iter: int,
    ftol: float | None = None,
    step_size: float = 1.0,
    light_history: bool = True,
    store_trace: bool = False,
    store_full_step_traces: bool = True,
    jvp_only: bool = False,
    solver_kwargs: dict[str, Any] | None = None,
) -> ResidualCheckpointTape:
    """Build a replay tape from one direct residual solve with adjoint tracing."""
    from .solve import solve_fixed_boundary_residual_iter

    solve_kwargs = dict(solver_kwargs or {})
    for key, value in (("indata", indata), ("signgs", int(signgs)), ("ftol", ftol), ("step_size", float(step_size))):
        solve_kwargs.setdefault(key, value)
    solve_kwargs["light_history"] = False if store_trace else bool(light_history)
    solve_kwargs.setdefault(
        "adjoint_trace_mode",
        "full" if bool(store_full_step_traces) else "dynamic",
    )

    tape_timing = {key: 0.0 for key in _DIRECT_TAPE_TIMING_KEYS}

    def _record_timing(key: str, start: float) -> None:
        tape_timing[key] = float(tape_timing.get(key, 0.0)) + (time.perf_counter() - start)

    def _solve_with_timing(current_solve_kwargs):
        start = time.perf_counter()
        out = solve_fixed_boundary_residual_iter(
            state0,
            static,
            max_iter=int(max_iter),
            adjoint_trace=True,
            **current_solve_kwargs,
        )
        _record_timing("tape_solve_call_s", start)
        return out

    def _extract_result_payload(current_result):
        pack_start = time.perf_counter()
        packed_state = jnp.asarray(pack_state(current_result.state), dtype=jnp.float64)
        _record_timing("tape_final_state_pack_s", pack_start)
        trace_extract_start = time.perf_counter()
        traces = tuple(current_result.diagnostics.get("adjoint_step_trace", ()))
        _record_timing("tape_step_trace_extract_s", trace_extract_start)
        compact = _compact_tape_diagnostics(current_result.diagnostics)
        iter_trace = residual_iteration_trace_from_result(current_result) if store_trace else _empty_trace()
        return packed_state, traces, compact, iter_trace

    result = _solve_with_timing(solve_kwargs)
    final_packed_state, step_traces, compact_diagnostics, trace = _extract_result_payload(result)
    stacked_step_traces = None
    step_trace_static_flags = None
    dynamic_initial_carry = None
    dynamic_base_carries_stacked = None
    preserve_jvp_basepoint_carries = bool(jvp_only and _jvp_only_exact_tape_basepoint_carries_enabled())
    if step_traces:
        step_trace_static_flags = _static_flags_from_replay_step_traces(step_traces)
        tentative_tape = ResidualCheckpointTape(
            final_packed_state=final_packed_state,
            packed_states=np.zeros((0, int(state0.layout.size)), dtype=float),
            trace=trace,
            resume_states=(),
            step_traces=step_traces,
            stacked_step_traces=None,
            step_trace_static_flags=step_trace_static_flags,
            dynamic_base_carries_stacked=dynamic_base_carries_stacked,
        )
        if _dynamic_replay_supported(tape=tentative_tape, rebuild_preconditioner=True):
            dynamic_payload_start = time.perf_counter()
            dynamic_stacked, dynamic_static_flags, dynamic_initial_carry, dynamic_base_carries_stacked = _build_dynamic_replay_payload(
                step_traces,
                step_trace_static_flags,
                store_base_carries=(not bool(jvp_only)) or preserve_jvp_basepoint_carries,
            )
            _record_timing("tape_dynamic_payload_build_s", dynamic_payload_start)
            if not store_full_step_traces:
                step_traces = ()
                stacked_step_traces = dynamic_stacked
                step_trace_static_flags = dynamic_static_flags
            else:
                trace_stack_start = time.perf_counter()
                stacked_step_traces, step_trace_static_flags = _stack_replay_step_traces(step_traces)
                _record_timing("tape_trace_stack_s", trace_stack_start)
        else:
            if solve_kwargs.get("adjoint_trace_mode") == "dynamic":
                # The compact dynamic trace intentionally omits the large
                # force/preconditioner fields needed by the generic replay
                # fallback. Rare restart/fallback paths therefore rerun once
                # with a full trace to preserve exactness.
                solve_kwargs_full = dict(solve_kwargs)
                solve_kwargs_full["adjoint_trace_mode"] = "full"
                result = _solve_with_timing(solve_kwargs_full)
                final_packed_state, step_traces, compact_diagnostics, trace = _extract_result_payload(result)
            trace_stack_start = time.perf_counter()
            stacked_step_traces, step_trace_static_flags = _stack_replay_step_traces(step_traces)
            _record_timing("tape_trace_stack_s", trace_stack_start)
    timing = compact_diagnostics.setdefault("timing", {})
    for key, value in tape_timing.items():
        timing[key] = float(timing.get(key, 0.0)) + float(value)
    if jvp_only:
        fast_basepoint_available = bool(
            dynamic_base_carries_stacked is not None
            and stacked_step_traces is not None
            and step_trace_static_flags is not None
            and _dynamic_basepoint_payload_shapes_match(stacked_step_traces, dynamic_base_carries_stacked)
        )
        compact_diagnostics["jvp_only_basepoint_carries_enabled"] = bool(preserve_jvp_basepoint_carries)
        compact_diagnostics["jvp_only_fast_basepoint_scan_available"] = fast_basepoint_available
        if fast_basepoint_available:
            compact_diagnostics["jvp_only_replay_path"] = "dynamic_basepoint_scan"
        elif dynamic_initial_carry is not None and not step_traces:
            compact_diagnostics["jvp_only_replay_path"] = "dynamic_whole_scan_linearize"
            compact_diagnostics["jvp_only_replay_fallback_reason"] = "basepoint_carries_not_stored"
        elif step_traces:
            compact_diagnostics["jvp_only_replay_path"] = "step_trace_replay"
            compact_diagnostics["jvp_only_replay_fallback_reason"] = "dynamic_exact_tape_unsupported"
        else:
            compact_diagnostics["jvp_only_replay_path"] = "identity"
    return ResidualCheckpointTape(
        final_packed_state=final_packed_state,
        packed_states=np.zeros((0, int(state0.layout.size)), dtype=float),
        trace=trace,
        resume_states=(),
        step_traces=step_traces,
        stacked_step_traces=stacked_step_traces,
        step_trace_static_flags=step_trace_static_flags,
        dynamic_initial_carry=dynamic_initial_carry,
        dynamic_base_carries_stacked=dynamic_base_carries_stacked,
        diagnostics=compact_diagnostics,
        jvp_only=bool(jvp_only and not step_traces),
    )


def replay_residual_checkpoint_step(
    state,
    static,
    *,
    resume_state: dict[str, Any] | None,
    solve_kwargs: dict[str, Any],
):
    """Replay exactly one residual-solver step from a stored checkpoint."""
    from .solve import solve_fixed_boundary_residual_iter

    return solve_fixed_boundary_residual_iter(
        state,
        static,
        max_iter=1,
        resume_state=resume_state,
        adjoint_trace=True,
        **solve_kwargs,
    )


def _carry_cotangents_with_zero_aux(final_cotangent, carry0):
    zero_cotangent = lambda value: jax.tree_util.tree_map(lambda x: jnp.zeros_like(jnp.asarray(x)), value)
    return (jnp.asarray(final_cotangent, dtype=jnp.asarray(carry0[0]).dtype), *(zero_cotangent(value) for value in carry0[1:]))


def checkpoint_tape_state_vjp(
    *,
    tape: ResidualCheckpointTape,
    static,
    final_cotangent,
    rebuild_preconditioner: bool = False,
):
    """Reverse a packed-state cotangent through the extracted step tape."""
    if bool(getattr(tape, "jvp_only", False)):
        raise ValueError(
            "JVP-only checkpoint tapes are forward-replay only; rebuild the tape with jvp_only=False."
        )

    if (
        _dynamic_replay_mode() == "whole_scan"
        and rebuild_preconditioner
        and tape.dynamic_initial_carry is not None
        and tape.stacked_step_traces is not None
        and tape.step_trace_static_flags is not None
    ):
        carry0 = tape.dynamic_initial_carry
        final_carry_cotangents = _carry_cotangents_with_zero_aux(final_cotangent, carry0)
        run_scan = _checkpoint_tape_dynamic_scan_runner(
            static=static,
            stacked=tape.stacked_step_traces,
            static_flags=tape.step_trace_static_flags,
        )

        def _run(carry_init):
            return run_scan(carry_init, tape.stacked_step_traces)

        _, vjp_fun = jax.vjp(_run, carry0)
        initial_carry_cotangents = vjp_fun(final_carry_cotangents)[0]
        return initial_carry_cotangents[0]

    if (
        rebuild_preconditioner
        and tape.dynamic_initial_carry is not None
        and tape.dynamic_base_carries_stacked is not None
        and tape.stacked_step_traces is not None
        and tape.step_trace_static_flags is not None
        and _dynamic_basepoint_payload_shapes_match(tape.stacked_step_traces, tape.dynamic_base_carries_stacked)
    ):
        carry0 = tape.dynamic_initial_carry
        final_carry_cotangents = _carry_cotangents_with_zero_aux(final_cotangent, carry0)
        run_scan = _checkpoint_tape_dynamic_basepoint_vjp_scan_runner(
            static=static,
            stacked=tape.stacked_step_traces,
            stacked_base_carries=tape.dynamic_base_carries_stacked,
            static_flags=tape.step_trace_static_flags,
        )
        initial_carry_cotangents = run_scan(
            final_carry_cotangents,
            tape.dynamic_base_carries_stacked,
            tape.stacked_step_traces,
        )
        return initial_carry_cotangents[0]

    if not tape.step_traces:
        return jnp.asarray(final_cotangent)

    cotangent = jnp.asarray(final_cotangent)
    for trace in reversed(tape.step_traces):
        x0 = jnp.asarray(pack_state(trace["state_pre"]))

        def _step_map(x):
            state = unpack_state(x, trace["state_pre"].layout)
            out = strict_update_one_step_from_state(
                state,
                static,
                wout_like=trace["wout_like"],
                trig=trace["trig"],
                apply_lforbal=trace["apply_lforbal"],
                include_edge_residual=trace["include_edge_residual"],
                apply_m1_constraints=trace["apply_m1_constraints"],
                zero_m1=trace["zero_m1"],
                mats=None if rebuild_preconditioner else trace["precond_mats"],
                jmax=None if rebuild_preconditioner else trace["precond_jmax"],
                lam_prec=None if rebuild_preconditioner else trace["lam_prec"],
                w_mode_mn=None if rebuild_preconditioner else trace["w_mode_mn"],
                preconditioner_jmax_override=int(trace["precond_jmax"]) if rebuild_preconditioner else None,
                preconditioner_use_precomputed_tridi=_trace_preconditioner_use_precomputed_tridi(trace),
                preconditioner_use_lax_tridi=_trace_preconditioner_use_lax_tridi(trace),
                lambda_update_scale=trace["lambda_update_scale"],
                dt_eff=trace["dt_eff"],
                b1=trace["b1"],
                fac=trace["fac"],
                force_scale=trace["force_scale"],
                flip_sign=trace["flip_sign"],
                vRcc_before=trace["vRcc_before"],
                vRss_before=trace["vRss_before"],
                vZsc_before=trace["vZsc_before"],
                vZcs_before=trace["vZcs_before"],
                vLsc_before=trace["vLsc_before"],
                vLcs_before=trace["vLcs_before"],
                max_update_rms=trace["max_update_rms_pre"],
                limit_update_rms=trace["limit_update_rms"],
                need_update_rms=False,
                divide_by_scalxc_for_update=trace["divide_by_scalxc_for_update"],
                freeb_bsqvac_half=trace.get("freeb_bsqvac_half", None),
                freeb_pres_scale=trace.get("freeb_pres_scale", None),
            )
            return pack_state(out["step"]["state_post"])

        _, vjp_fun = jax.vjp(_step_map, x0)
        cotangent = vjp_fun(cotangent)[0]
    return cotangent


def checkpoint_tape_state_jvp(
    *,
    tape: ResidualCheckpointTape,
    static,
    initial_tangent,
    rebuild_preconditioner: bool = False,
):
    """Push a packed-state tangent forward through the extracted step tape."""
    if (
        tape.dynamic_initial_carry is not None
        or _dynamic_replay_supported(tape=tape, rebuild_preconditioner=rebuild_preconditioner)
    ):
        tangents = checkpoint_tape_state_jvp_columns(
            tape=tape,
            static=static,
            initial_tangents=jnp.asarray(initial_tangent)[None, :],
            rebuild_preconditioner=rebuild_preconditioner,
        )
        return tangents[0]
    if not tape.step_traces:
        return jnp.asarray(initial_tangent)

    tangent = jnp.asarray(initial_tangent)
    for trace in tape.step_traces:
        x0 = jnp.asarray(pack_state(trace["state_pre"]))

        def _step_map(x):
            state = unpack_state(x, trace["state_pre"].layout)
            out = strict_update_one_step_from_state(
                state,
                static,
                wout_like=trace["wout_like"],
                trig=trace["trig"],
                apply_lforbal=trace["apply_lforbal"],
                include_edge_residual=trace["include_edge_residual"],
                apply_m1_constraints=trace["apply_m1_constraints"],
                zero_m1=trace["zero_m1"],
                mats=None if rebuild_preconditioner else trace["precond_mats"],
                jmax=None if rebuild_preconditioner else trace["precond_jmax"],
                lam_prec=None if rebuild_preconditioner else trace["lam_prec"],
                w_mode_mn=None if rebuild_preconditioner else trace["w_mode_mn"],
                preconditioner_jmax_override=int(trace["precond_jmax"]) if rebuild_preconditioner else None,
                preconditioner_use_precomputed_tridi=_trace_preconditioner_use_precomputed_tridi(trace),
                preconditioner_use_lax_tridi=_trace_preconditioner_use_lax_tridi(trace),
                lambda_update_scale=trace["lambda_update_scale"],
                dt_eff=trace["dt_eff"],
                b1=trace["b1"],
                fac=trace["fac"],
                force_scale=trace["force_scale"],
                flip_sign=trace["flip_sign"],
                vRcc_before=trace["vRcc_before"],
                vRss_before=trace["vRss_before"],
                vZsc_before=trace["vZsc_before"],
                vZcs_before=trace["vZcs_before"],
                vLsc_before=trace["vLsc_before"],
                vLcs_before=trace["vLcs_before"],
                max_update_rms=trace["max_update_rms_pre"],
                limit_update_rms=trace["limit_update_rms"],
                need_update_rms=False,
                divide_by_scalxc_for_update=trace["divide_by_scalxc_for_update"],
                freeb_bsqvac_half=trace.get("freeb_bsqvac_half", None),
                freeb_pres_scale=trace.get("freeb_pres_scale", None),
            )
            return pack_state(out["step"]["state_post"])

        _, tangent = jax.jvp(_step_map, (x0,), (tangent,))
    return tangent


def _packed_replay_step_from_trace(
    packed_state,
    trace,
    *,
    static,
    rebuild_preconditioner: bool,
    apply_lforbal,
    include_edge_residual,
    apply_m1_constraints,
    limit_update_rms,
    divide_by_scalxc_for_update,
    preconditioner_jmax_override,
    preconditioner_use_precomputed_tridi: bool | None = None,
    preconditioner_use_lax_tridi: bool | None = None,
):
    state = unpack_state(packed_state, trace["state_pre"].layout)
    stored_jmax = preconditioner_jmax_override if preconditioner_jmax_override is not None else trace["precond_jmax"]
    out = strict_update_one_step_from_state(
        state,
        static,
        wout_like=trace["wout_like"],
        trig=trace["trig"],
        apply_lforbal=apply_lforbal,
        include_edge_residual=include_edge_residual,
        apply_m1_constraints=apply_m1_constraints,
        zero_m1=trace["zero_m1"],
        mats=None if rebuild_preconditioner else trace["precond_mats"],
        jmax=None if rebuild_preconditioner else stored_jmax,
        lam_prec=None if rebuild_preconditioner else trace["lam_prec"],
        w_mode_mn=None if rebuild_preconditioner else trace["w_mode_mn"],
        preconditioner_jmax_override=preconditioner_jmax_override if rebuild_preconditioner else None,
        preconditioner_use_precomputed_tridi=(
            _trace_preconditioner_use_precomputed_tridi(trace)
            if preconditioner_use_precomputed_tridi is None
            else preconditioner_use_precomputed_tridi
        ),
        preconditioner_use_lax_tridi=(
            _trace_preconditioner_use_lax_tridi(trace)
            if preconditioner_use_lax_tridi is None
            else preconditioner_use_lax_tridi
        ),
        freeb_bsqvac_half=trace.get("freeb_bsqvac_half", None),
        freeb_pres_scale=trace.get("freeb_pres_scale", None),
        lambda_update_scale=trace["lambda_update_scale"],
        dt_eff=trace["dt_eff"],
        b1=trace["b1"],
        fac=trace["fac"],
        force_scale=trace["force_scale"],
        flip_sign=trace["flip_sign"],
        vRcc_before=trace["vRcc_before"],
        vRss_before=trace["vRss_before"],
        vZsc_before=trace["vZsc_before"],
        vZcs_before=trace["vZcs_before"],
        vLsc_before=trace["vLsc_before"],
        vLcs_before=trace["vLcs_before"],
        max_update_rms=trace["max_update_rms_pre"],
        limit_update_rms=limit_update_rms,
        need_update_rms=False,
        divide_by_scalxc_for_update=divide_by_scalxc_for_update,
    )
    return pack_state(out["step"]["state_post"])

def _packed_dynamic_replay_step_from_carry(
    carry,
    trace,
    *,
    static,
    static_flags,
    preconditioner_jmax_override,
    preconditioner_use_precomputed_tridi: bool | None = None,
    preconditioner_use_lax_tridi: bool | None = None,
):
    if len(carry) != 20:
        raise ValueError("dynamic replay requires a stored VMEC layout and complete replay carry")
    (
        packed_state,
        inv_tau,
        fsq_prev,
        vRcc_before,
        vRss_before,
        vRsc_before,
        vRcs_before,
        vZsc_before,
        vZcs_before,
        vZcc_before,
        vZss_before,
        vLsc_before,
        vLcs_before,
        vLcc_before,
        vLss_before,
        constraint_ard1_before,
        constraint_azd1_before,
        constraint_tcon_before,
        cache_lam_prec_before,
        cache_prec_mats_before,
    ) = carry
    layout = static_flags.get("layout", trace["state_pre"].layout if isinstance(trace, dict) and "state_pre" in trace else None)
    if layout is None:
        raise ValueError("dynamic replay requires a stored VMEC layout")
    state_pre = unpack_state(packed_state, layout)
    (
        wout_like,
        trig,
        w_mode_mn,
        lambda_update_scale,
        max_coeff_delta_rms_pre,
        max_update_rms_pre,
    ) = _dynamic_replay_values(
        trace,
        static_flags,
        "wout_like",
        "trig",
        "w_mode_mn",
        "lambda_update_scale",
        "max_coeff_delta_rms_pre",
        "max_update_rms_pre",
    )
    (
        constraint_precond_active,
        constraint_tcon_active,
        constraint_precond_diag_current,
        constraint_tcon_current,
    ) = _dynamic_constraint_cache_current(
        trace,
        static_flags,
        constraint_ard1_before,
        constraint_azd1_before,
        constraint_tcon_before,
    )

    residual_out = raw_force_residual_from_state(
        state_pre,
        static,
        wout_like=wout_like,
        trig=trig,
        apply_lforbal=static_flags["apply_lforbal"],
        include_edge_residual=static_flags["include_edge_residual"],
        apply_m1_constraints=static_flags["apply_m1_constraints"],
        zero_m1=trace["zero_m1"],
        constraint_tcon0=_dynamic_replay_value(trace, static_flags, "constraint_tcon0"),
        constraint_tcon=constraint_tcon_current,
        constraint_precond_diag=constraint_precond_diag_current,
        constraint_precond_active=constraint_precond_active,
        constraint_tcon_active=constraint_tcon_active,
        constraint_rcon0=_dynamic_replay_value(trace, static_flags, "constraint_rcon0"),
        constraint_zcon0=_dynamic_replay_value(trace, static_flags, "constraint_zcon0"),
        freeb_bsqvac_half=trace.get("freeb_bsqvac_half", None) if isinstance(trace, dict) else None,
        freeb_pres_scale=trace.get("freeb_pres_scale", None) if isinstance(trace, dict) else static_flags.get("freeb_pres_scale", None),
    )
    tridi_policy = (
        _trace_preconditioner_use_precomputed_tridi(trace, static_flags)
        if preconditioner_use_precomputed_tridi is None
        else preconditioner_use_precomputed_tridi
    )
    lax_tridi_policy = (
        _trace_preconditioner_use_lax_tridi(trace, static_flags)
        if preconditioner_use_lax_tridi is None
        else preconditioner_use_lax_tridi
    )
    refreshed_preconditioner_out = state_dependent_preconditioner_from_forces(
        k=residual_out["k"],
        static=static,
        trig=trig,
        dtype=jnp.asarray(packed_state).dtype,
        jmax_override=preconditioner_jmax_override,
        w_mode_mn=w_mode_mn,
        use_precomputed=tridi_policy,
        use_lax_tridi=lax_tridi_policy,
    )
    preconditioner_out = _dynamic_preconditioner_cache_current(
        refreshed_preconditioner_out,
        trace,
        cache_lam_prec_before,
        cache_prec_mats_before,
    )
    constraint_tcon0 = _dynamic_replay_value(trace, static_flags, "constraint_tcon0")
    if constraint_tcon0 is None or float(constraint_tcon0) == 0.0:
        refreshed_ard1 = jnp.zeros_like(jnp.asarray(constraint_ard1_before))
        refreshed_azd1 = jnp.zeros_like(jnp.asarray(constraint_azd1_before))
        refreshed_tcon = jnp.zeros_like(jnp.asarray(constraint_tcon_before))
    else:
        from .vmec_constraints import precondn_diag_axd1_from_bcovar

        refreshed_ard1, refreshed_azd1 = precondn_diag_axd1_from_bcovar(
            trig=trig,
            s=jnp.asarray(static.s),
            bsq=residual_out["k"].bc.bsq,
            r12=residual_out["k"].bc.jac.r12,
            sqrtg=residual_out["k"].bc.jac.sqrtg,
            ru12=residual_out["k"].bc.jac.ru12,
            zu12=residual_out["k"].bc.jac.zu12,
        )
        refreshed_tcon = jnp.asarray(residual_out["k"].tcon, dtype=jnp.asarray(constraint_tcon_before).dtype)
        refreshed_ard1 = jnp.asarray(refreshed_ard1, dtype=jnp.asarray(constraint_ard1_before).dtype)
        refreshed_azd1 = jnp.asarray(refreshed_azd1, dtype=jnp.asarray(constraint_azd1_before).dtype)
    update_cache = jnp.asarray(_dynamic_replay_value(trace, {}, "constraint_cache_update", False), dtype=bool)
    constraint_ard1_next = jnp.where(update_cache, refreshed_ard1, jnp.asarray(constraint_ard1_before))
    constraint_azd1_next = jnp.where(update_cache, refreshed_azd1, jnp.asarray(constraint_azd1_before))
    constraint_tcon_next = jnp.where(update_cache, refreshed_tcon, jnp.asarray(constraint_tcon_before))
    cache_lam_prec_next = preconditioner_out["lam_prec"]
    cache_prec_mats_next = preconditioner_out["mats"]
    force_out = preconditioned_force_channels_from_raw_forces(
        frzl=residual_out["frzl"],
        mats=preconditioner_out["mats"],
        jmax=preconditioner_out["jmax"],
        cfg=static.cfg,
        lam_prec=preconditioner_out["lam_prec"],
        w_mode_mn=preconditioner_out["w_mode_mn"],
        lambda_update_scale=lambda_update_scale,
        use_precomputed=tridi_policy,
        use_lax_tridi=lax_tridi_policy,
    )
    fsq1 = _dynamic_fsq1_from_force_channels(
        state_pre=state_pre,
        static=static,
        vmec2000_control=bool(static_flags["vmec2000_control"]),
        frzl_pre=force_out["frzl_pre"],
    )
    time_step = jnp.asarray(trace["time_step"], dtype=jnp.asarray(packed_state).dtype)
    invtau_reset = jnp.full_like(inv_tau, jnp.asarray(0.15, dtype=time_step.dtype) / time_step)
    invtau_num = jnp.where(
        fsq1 == 0.0,
        jnp.asarray(0.0, dtype=time_step.dtype),
        jnp.minimum(jnp.abs(jnp.log(fsq1 / fsq_prev)), jnp.asarray(0.15, dtype=time_step.dtype)),
    )
    invtau_shift = jnp.concatenate([inv_tau[1:], (invtau_num / time_step)[None]], axis=0)
    inv_tau_next = jnp.where(jnp.asarray(trace["reset_inv_tau"]), invtau_reset, invtau_shift)
    otav = jnp.sum(inv_tau_next) / jnp.asarray(inv_tau_next.shape[0], dtype=time_step.dtype)
    dtau = time_step * otav / jnp.asarray(2.0, dtype=time_step.dtype)
    b1 = jnp.asarray(1.0, dtype=time_step.dtype) - dtau
    fac = jnp.asarray(1.0, dtype=time_step.dtype) / (jnp.asarray(1.0, dtype=time_step.dtype) + dtau)
    if bool(static_flags["limit_dt_from_force"]):
        dt_eff = _dynamic_safe_dt_from_force_arrays(
            dt_nominal=time_step,
            max_coeff_delta_rms=max_coeff_delta_rms_pre,
            frcc=force_out["frcc_u"],
            frss=force_out["frss_u"],
            fzsc=force_out["fzsc_u"],
            fzcs=force_out["fzcs_u"],
            flsc=force_out["flsc_u"],
            flcs=force_out["flcs_u"],
            frsc=force_out["frsc_u"],
            frcs=force_out["frcs_u"],
            fzcc=force_out["fzcc_u"],
            fzss=force_out["fzss_u"],
            flcc=force_out["flcc_u"],
            flss=force_out["flss_u"],
        )
    else:
        dt_eff = time_step
    step_out = strict_update_accepted_step(
        state_pre,
        static,
        dt_eff=dt_eff,
        b1=b1,
        fac=fac,
        force_scale=dt_eff,
        flip_sign=trace["flip_sign"],
        vRcc_before=vRcc_before,
        vRss_before=vRss_before,
        vZsc_before=vZsc_before,
        vZcs_before=vZcs_before,
        vLsc_before=vLsc_before,
        vLcs_before=vLcs_before,
        frcc_u=force_out["frcc_u"],
        frss_u=force_out["frss_u"],
        fzsc_u=force_out["fzsc_u"],
        fzcs_u=force_out["fzcs_u"],
        flsc_u=force_out["flsc_u"],
        flcs_u=force_out["flcs_u"],
        vRsc_before=vRsc_before,
        vRcs_before=vRcs_before,
        vZcc_before=vZcc_before,
        vZss_before=vZss_before,
        vLcc_before=vLcc_before,
        vLss_before=vLss_before,
        frsc_u=force_out.get("frsc_u"),
        frcs_u=force_out.get("frcs_u"),
        fzcc_u=force_out.get("fzcc_u"),
        fzss_u=force_out.get("fzss_u"),
        flcc_u=force_out.get("flcc_u"),
        flss_u=force_out.get("flss_u"),
        max_update_rms=max_update_rms_pre,
        limit_update_rms=static_flags["limit_update_rms"],
        need_update_rms=False,
        divide_by_scalxc_for_update=static_flags["divide_by_scalxc_for_update"],
    )
    return (
        pack_state(step_out["state_post"]),
        inv_tau_next,
        fsq1,
        step_out["vRcc_after"],
        step_out["vRss_after"],
        step_out["vRsc_after"],
        step_out["vRcs_after"],
        step_out["vZsc_after"],
        step_out["vZcs_after"],
        step_out["vZcc_after"],
        step_out["vZss_after"],
        step_out["vLsc_after"],
        step_out["vLcs_after"],
        step_out["vLcc_after"],
        step_out["vLss_after"],
        constraint_ard1_next,
        constraint_azd1_next,
        constraint_tcon_next,
        cache_lam_prec_next,
        cache_prec_mats_next,
    )


def _packed_dynamic_replay_step_with_flags(carry, trace, *, static, static_flags):
    replay_flags = trace if static_flags is None else static_flags
    jmax_override = (
        int(trace["precond_jmax"])
        if static_flags is None or static_flags.get("precond_jmax") is None
        else int(static_flags["precond_jmax"])
    )
    return _packed_dynamic_replay_step_from_carry(
        carry,
        trace,
        static=static,
        static_flags=replay_flags,
        preconditioner_jmax_override=jmax_override,
        preconditioner_use_precomputed_tridi=_trace_preconditioner_use_precomputed_tridi(trace, static_flags),
        preconditioner_use_lax_tridi=_trace_preconditioner_use_lax_tridi(trace, static_flags),
    )


def _checkpoint_tape_scan_runner(*, static, stacked, static_flags, rebuild_preconditioner: bool):
    key = (
        id(static),
        bool(rebuild_preconditioner),
        bool(static_flags["apply_lforbal"]),
        bool(static_flags["include_edge_residual"]),
        bool(static_flags["apply_m1_constraints"]),
        bool(static_flags["limit_update_rms"]),
        bool(static_flags["divide_by_scalxc_for_update"]),
        None if static_flags["precond_jmax"] is None else int(static_flags["precond_jmax"]),
        _tridi_policy_cache_value(static_flags.get("preconditioner_use_precomputed_tridi", None)),
        _tridi_policy_cache_value(static_flags.get("preconditioner_use_lax_tridi", None)),
        _stacked_trace_signature(stacked),
    )
    cached, cache_miss = _get_replay_scan_runner("checkpoint", _CHECKPOINT_TAPE_SCAN_CACHE, key)
    if cached is not None:
        return cached

    def _step_scan(carry, trace):
        tangents = carry
        x0 = jnp.asarray(pack_state(trace["state_pre"]))

        def _step_map(x):
            return _packed_replay_step_from_trace(
                x,
                trace,
                static=static,
                rebuild_preconditioner=rebuild_preconditioner,
                apply_lforbal=static_flags["apply_lforbal"],
                include_edge_residual=static_flags["include_edge_residual"],
                apply_m1_constraints=static_flags["apply_m1_constraints"],
                limit_update_rms=static_flags["limit_update_rms"],
                divide_by_scalxc_for_update=static_flags["divide_by_scalxc_for_update"],
                preconditioner_jmax_override=static_flags["precond_jmax"],
                preconditioner_use_precomputed_tridi=static_flags.get(
                    "preconditioner_use_precomputed_tridi",
                    None,
                ),
                preconditioner_use_lax_tridi=static_flags.get(
                    "preconditioner_use_lax_tridi",
                    None,
                ),
            )

        _, linear_step = jax.linearize(_step_map, x0)
        tangents = jax.vmap(linear_step)(tangents)
        return tangents, None

    @jax.jit
    def _run_scan(tangents, stacked_traces):
        tangents, _ = jax.lax.scan(_step_scan, tangents, stacked_traces)
        return tangents

    return _put_replay_scan_runner("checkpoint", _CHECKPOINT_TAPE_SCAN_CACHE, key, _run_scan, cache_miss)


def _checkpoint_tape_dynamic_scan_runner(*, static, stacked, static_flags):
    key = _dynamic_replay_cache_key(static=static, stacked=stacked, static_flags=static_flags)
    cached, cache_miss = _get_replay_scan_runner("dynamic", _CHECKPOINT_TAPE_DYNAMIC_SCAN_CACHE, key)
    if cached is not None:
        return cached

    def _step_scan(carry, trace):
        active = jnp.asarray(trace["active"], dtype=bool) if "active" in trace else jnp.asarray(True, dtype=bool)

        def _advance(carry_in):
            return _packed_dynamic_replay_step_with_flags(
                carry_in,
                trace,
                static=static,
                static_flags=static_flags,
            )

        carry = jax.lax.cond(active, _advance, lambda carry_in: carry_in, carry)
        return carry, None

    @jax.jit
    def _run_scan(carry0, stacked_traces):
        carry, _ = jax.lax.scan(_step_scan, carry0, stacked_traces)
        return carry

    return _put_replay_scan_runner("dynamic", _CHECKPOINT_TAPE_DYNAMIC_SCAN_CACHE, key, _run_scan, cache_miss)


def _checkpoint_tape_dynamic_basepoint_scan_runner(*, static, stacked, stacked_base_carries, static_flags):
    key = _dynamic_replay_cache_key(
        static=static,
        stacked=stacked,
        static_flags=static_flags,
        stacked_base_carries=stacked_base_carries,
    )
    cached, cache_miss = _get_replay_scan_runner("dynamic_basepoint", _CHECKPOINT_TAPE_DYNAMIC_BASEPOINT_SCAN_CACHE, key)
    if cached is not None:
        return cached

    def _step_scan(pair, trace):
        carry_base, carry_tangents = pair
        active = jnp.asarray(trace["active"], dtype=bool) if "active" in trace else jnp.asarray(True, dtype=bool)

        def _step(carry):
            return _packed_dynamic_replay_step_with_flags(
                carry,
                trace,
                static=static,
                static_flags=static_flags,
            )

        def _advance(pair_in):
            # Propagate the base carry inside the scan instead of linearizing at
            # the saved host carry for each step.  Saved carries can differ from
            # replayed carries in auxiliary cache slots, while the accepted-state
            # tangent must follow the replayed branch exactly.
            carry_base_in, carry_tangents_in = pair_in
            carry_base_out, linear_step = jax.linearize(_step, carry_base_in)
            return carry_base_out, jax.vmap(linear_step)(carry_tangents_in)

        pair = jax.lax.cond(active, _advance, lambda pair_in: pair_in, pair)
        return pair, None

    def _scan_from_carry(carry_tangents0, stacked_base_carries_in, stacked_traces_in):
        carry0 = jax.tree_util.tree_map(lambda x: x[0], stacked_base_carries_in)
        (_carry_base, carry_tangents), _ = jax.lax.scan(_step_scan, (carry0, carry_tangents0), stacked_traces_in)
        return carry_tangents

    @jax.jit
    def _run_scan(carry_tangents0, stacked_base_carries_in, stacked_traces_in):
        return _scan_from_carry(carry_tangents0, stacked_base_carries_in, stacked_traces_in)

    @jax.jit
    def _run_scan_zero_aux(state_tangents0, stacked_base_carries_in, stacked_traces_in):
        carry0 = jax.tree_util.tree_map(lambda x: x[0], stacked_base_carries_in)
        carry_tangents0 = _carry_tangents_with_zero_aux(state_tangents0, carry0)
        return _scan_from_carry(carry_tangents0, stacked_base_carries_in, stacked_traces_in)

    runner = _DynamicBasepointScanRunner(
        from_carry=_run_scan,
        from_state_tangents=_run_scan_zero_aux,
    )
    return _put_replay_scan_runner(
        "dynamic_basepoint",
        _CHECKPOINT_TAPE_DYNAMIC_BASEPOINT_SCAN_CACHE,
        key,
        runner,
        cache_miss,
    )


def _run_dynamic_basepoint_scan_zero_aux(*, run_scan, state_tangents, stacked_base_carries, stacked):
    run_zero_aux = getattr(run_scan, "zero_aux", None)
    if run_zero_aux is not None:
        return run_zero_aux(state_tangents, stacked_base_carries, stacked)

    carry0 = jax.tree_util.tree_map(lambda x: x[0], stacked_base_carries)
    carry_tangents0 = _carry_tangents_with_zero_aux(state_tangents, carry0)
    return run_scan(carry_tangents0, stacked_base_carries, stacked)


def _checkpoint_tape_dynamic_basepoint_vjp_scan_runner(*, static, stacked, stacked_base_carries, static_flags):
    key = _dynamic_replay_cache_key(
        static=static,
        stacked=stacked,
        static_flags=static_flags,
        stacked_base_carries=stacked_base_carries,
    )
    cached, cache_miss = _get_replay_scan_runner(
        "dynamic_basepoint_vjp",
        _CHECKPOINT_TAPE_DYNAMIC_BASEPOINT_VJP_SCAN_CACHE,
        key,
    )
    if cached is not None:
        return cached

    def _step_scan(carry_cotangents, inputs):
        carry_base, trace = inputs
        active = jnp.asarray(trace["active"], dtype=bool) if "active" in trace else jnp.asarray(True, dtype=bool)

        def _step(carry):
            return _packed_dynamic_replay_step_with_flags(
                carry,
                trace,
                static=static,
                static_flags=static_flags,
            )

        def _advance(cotangents_in):
            _, vjp_fun = jax.vjp(_step, carry_base)
            return vjp_fun(cotangents_in)[0]

        carry_cotangents = jax.lax.cond(
            active,
            _advance,
            lambda cotangents_in: cotangents_in,
            carry_cotangents,
        )
        return carry_cotangents, None

    @jax.jit
    def _run_scan(final_cotangents, stacked_base_carries_in, stacked_traces_in):
        reverse = lambda x: jnp.flip(x, axis=0)
        reversed_base_carries = jax.tree_util.tree_map(reverse, stacked_base_carries_in)
        reversed_traces = jax.tree_util.tree_map(reverse, stacked_traces_in)
        initial_cotangents, _ = jax.lax.scan(
            _step_scan,
            final_cotangents,
            (reversed_base_carries, reversed_traces),
        )
        return initial_cotangents

    return _put_replay_scan_runner(
        "dynamic_basepoint_vjp",
        _CHECKPOINT_TAPE_DYNAMIC_BASEPOINT_VJP_SCAN_CACHE,
        key,
        _run_scan,
        cache_miss,
    )


def checkpoint_tape_state_jvp_columns(
    *,
    tape: ResidualCheckpointTape,
    static,
    initial_tangents,
    rebuild_preconditioner: bool = False,
    column_chunk: int | None = None,
    _allow_chunking: bool = True,
):
    """Push multiple packed-state tangents forward through the extracted step tape."""
    if not tape.step_traces and tape.dynamic_initial_carry is None:
        n_columns = int(getattr(initial_tangents, "shape", (0,))[0]) if getattr(initial_tangents, "shape", ()) else 0
        _record_replay_jvp_columns_path("identity", n_columns=n_columns)
        return jnp.asarray(initial_tangents)

    tangents = jnp.asarray(initial_tangents)
    if _allow_chunking:
        chunk_env = os.environ.get("VMEC_JAX_REPLAY_COLUMN_CHUNK")
        env_handled, env_chunk = _replay_column_chunk_override(chunk_env)
        if env_handled:
            active_column_chunk = env_chunk
        elif column_chunk is not None:
            active_column_chunk = max(1, int(column_chunk))
        else:
            active_column_chunk = _replay_column_chunk_default(tape=tape, tangents=tangents)
        if active_column_chunk is not None and tangents.shape[0] > active_column_chunk:
            outputs = []
            n_chunks = (int(tangents.shape[0]) + int(active_column_chunk) - 1) // int(active_column_chunk)
            _record_replay_jvp_columns_chunking(n_chunks=n_chunks, chunk_size=int(active_column_chunk))
            for start in range(0, int(tangents.shape[0]), active_column_chunk):
                outputs.append(
                    checkpoint_tape_state_jvp_columns(
                        tape=tape,
                        static=static,
                        initial_tangents=tangents[start : start + active_column_chunk],
                        rebuild_preconditioner=rebuild_preconditioner,
                        column_chunk=active_column_chunk,
                        _allow_chunking=False,
                    )
                )
            return jnp.concatenate(outputs, axis=0)
    stacked = tape.stacked_step_traces
    static_flags = tape.step_trace_static_flags
    if (
        _dynamic_replay_mode() != "whole_scan"
        and tape.dynamic_base_carries_stacked is not None
        and stacked is not None
        and static_flags is not None
        and rebuild_preconditioner
        and _dynamic_basepoint_payload_shapes_match(stacked, tape.dynamic_base_carries_stacked)
    ):
        stacked_base_carries = tape.dynamic_base_carries_stacked
        run_scan = _checkpoint_tape_dynamic_basepoint_scan_runner(
            static=static,
            stacked=stacked,
            stacked_base_carries=stacked_base_carries,
            static_flags=static_flags,
        )
        carry_tangents_final = _run_dynamic_basepoint_scan_zero_aux(
            run_scan=run_scan,
            state_tangents=tangents,
            stacked_base_carries=stacked_base_carries,
            stacked=stacked,
        )
        _record_replay_jvp_columns_path("dynamic_basepoint", n_columns=int(tangents.shape[0]))
        return carry_tangents_final[0]
    if tape.step_traces and rebuild_preconditioner:
        carry_tangents = None
        idx = 0
        while idx < len(tape.step_traces):
            trace = tape.step_traces[idx]
            if _dynamic_replay_trace_supported(trace):
                end = idx + 1
                while end < len(tape.step_traces) and _dynamic_replay_trace_supported(tape.step_traces[end]):
                    end += 1
                segment = tuple(tape.step_traces[idx:end])
                segment_static_flags = _static_flags_from_replay_step_traces(segment)
                segment_stacked, segment_dynamic_flags, _segment_initial_carry, segment_base_carries = _build_dynamic_replay_payload(
                    segment,
                    segment_static_flags,
                )
                run_scan = _checkpoint_tape_dynamic_basepoint_scan_runner(
                    static=static,
                    stacked=segment_stacked,
                    stacked_base_carries=segment_base_carries,
                    static_flags=segment_dynamic_flags,
                )
                if carry_tangents is None:
                    carry_tangents = _run_dynamic_basepoint_scan_zero_aux(
                        run_scan=run_scan,
                        state_tangents=tangents,
                        stacked_base_carries=segment_base_carries,
                        stacked=segment_stacked,
                    )
                else:
                    carry_tangents = run_scan(carry_tangents, segment_base_carries, segment_stacked)
                idx = end
                continue
            if _dynamic_restart_trace_supported(trace):
                if carry_tangents is None:
                    carry_tangents = _carry_tangents_with_zero_aux(tangents, _dynamic_replay_initial_carry(trace))
                carry_tangents = _restart_carry_tangents(carry_tangents)
                idx += 1
                continue
            carry_tangents = None
            break
        if carry_tangents is not None and idx == len(tape.step_traces):
            _record_replay_jvp_columns_path("segmented_dynamic_basepoint", n_columns=int(tangents.shape[0]))
            return carry_tangents[0]
    if tape.step_traces and _dynamic_replay_supported(tape=tape, rebuild_preconditioner=rebuild_preconditioner):
        carry0 = _dynamic_replay_initial_carry(tape.step_traces[0])
        carry_tangents = _carry_tangents_with_zero_aux(tangents, carry0)
        for trace in tape.step_traces:
            carry_base = _dynamic_replay_initial_carry(trace)

            def _step(carry):
                return _packed_dynamic_replay_step_with_flags(
                    carry,
                    trace,
                    static=static,
                    static_flags=static_flags,
                )

            _, linear_step = jax.linearize(_step, carry_base)
            carry_tangents = jax.vmap(linear_step)(carry_tangents)
        _record_replay_jvp_columns_path("dynamic_linearize", n_columns=int(tangents.shape[0]))
        return carry_tangents[0]
    if tape.dynamic_initial_carry is not None and stacked is not None and static_flags is not None and rebuild_preconditioner:
        carry0 = tape.dynamic_initial_carry
        carry_tangents0 = _carry_tangents_with_zero_aux(tangents, carry0)
        run_scan = _checkpoint_tape_dynamic_scan_runner(
            static=static,
            stacked=stacked,
            static_flags=static_flags,
        )

        def _run(carry_init):
            return run_scan(carry_init, stacked)

        _, linear_step = jax.linearize(_run, carry0)
        carry_tangents_final = jax.vmap(linear_step)(carry_tangents0)
        _record_replay_jvp_columns_path("dynamic_scan_linearize", n_columns=int(tangents.shape[0]))
        return carry_tangents_final[0]

    if stacked is None or static_flags is None:
        stacked, static_flags = _stack_replay_step_traces(tape.step_traces)
    if rebuild_preconditioner and static_flags["precond_jmax"] is None:
        for trace in tape.step_traces:
            x0 = jnp.asarray(pack_state(trace["state_pre"]))

            def _step_map(x):
                return _packed_replay_step_from_trace(
                    x,
                    trace,
                    static=static,
                    rebuild_preconditioner=True,
                    apply_lforbal=trace["apply_lforbal"],
                    include_edge_residual=trace["include_edge_residual"],
                    apply_m1_constraints=trace["apply_m1_constraints"],
                    limit_update_rms=trace["limit_update_rms"],
                    divide_by_scalxc_for_update=trace["divide_by_scalxc_for_update"],
                    preconditioner_jmax_override=(
                        static_flags["precond_jmax"]
                        if static_flags["precond_jmax"] is not None
                        else int(trace["precond_jmax"])
                    ),
                    preconditioner_use_precomputed_tridi=_trace_preconditioner_use_precomputed_tridi(
                        trace,
                        static_flags,
                    ),
                    preconditioner_use_lax_tridi=_trace_preconditioner_use_lax_tridi(
                        trace,
                        static_flags,
                    ),
                )

            _, linear_step = jax.linearize(_step_map, x0)
            tangents = jax.vmap(linear_step)(tangents)
        _record_replay_jvp_columns_path("generic_per_trace", n_columns=int(tangents.shape[0]))
        return tangents

    run_scan = _checkpoint_tape_scan_runner(
        static=static,
        stacked=stacked,
        static_flags=static_flags,
        rebuild_preconditioner=rebuild_preconditioner,
    )
    _record_replay_jvp_columns_path("generic_scan", n_columns=int(tangents.shape[0]))
    return run_scan(tangents, stacked)


def checkpoint_tape_param_vjp(
    *,
    tape: ResidualCheckpointTape,
    static,
    boundary,
    indata,
    specs,
    params,
    axis_override,
    final_cotangent,
    vmec_project: bool = True,
    rebuild_preconditioner: bool = True,
):
    """Reverse a packed-state cotangent back to boundary parameters."""
    from .init_guess import initial_guess_from_boundary
    from .optimization import apply_boundary_params

    state_cotangent = checkpoint_tape_state_vjp(
        tape=tape,
        static=static,
        final_cotangent=final_cotangent,
        rebuild_preconditioner=rebuild_preconditioner,
    )

    params0 = jnp.asarray(params)

    def _state_from_params(p):
        boundary_p = apply_boundary_params(boundary, specs, p)
        state = initial_guess_from_boundary(
            static,
            boundary_p,
            indata,
            vmec_project=vmec_project,
            axis_override=axis_override,
        )
        return pack_state(state)

    _, vjp_fun = jax.vjp(_state_from_params, params0)
    return vjp_fun(jnp.asarray(state_cotangent))[0]


def checkpoint_tape_param_jvp(
    *,
    tape: ResidualCheckpointTape,
    static,
    boundary,
    indata,
    specs,
    params,
    axis_override,
    params_tangent,
    vmec_project: bool = True,
    rebuild_preconditioner: bool = True,
):
    """Push a parameter tangent forward to the final packed state."""
    from .init_guess import initial_guess_from_boundary
    from .optimization import apply_boundary_params

    params0 = jnp.asarray(params)
    params_tangent = jnp.asarray(params_tangent)

    def _state_from_params(p):
        boundary_p = apply_boundary_params(boundary, specs, p)
        state = initial_guess_from_boundary(
            static,
            boundary_p,
            indata,
            vmec_project=vmec_project,
            axis_override=axis_override,
        )
        return pack_state(state)

    _, state_tangent = jax.jvp(_state_from_params, (params0,), (params_tangent,))
    return checkpoint_tape_state_jvp(
        tape=tape,
        static=static,
        initial_tangent=state_tangent,
        rebuild_preconditioner=rebuild_preconditioner,
    )


from vmec_jax.solvers.fixed_boundary.adjoint.strict_updates import (
    preconditioned_force_channels_from_raw_forces,
    preconditioned_force_channels_from_rz_output,
    raw_force_residual_from_state,
    state_dependent_preconditioner_from_forces,
    strict_update_accepted_step,
    strict_update_velocity_block,
    strict_update_velocity_limit,
    strict_update_velocity_state_advance,
)
def strict_update_one_step_from_state(
    state_pre,
    static,
    *,
    force_state_pre=None,
    wout_like,
    trig,
    apply_lforbal: bool,
    include_edge_residual: bool,
    apply_m1_constraints: bool,
    zero_m1,
    mats=None,
    jmax=None,
    lam_prec=None,
    w_mode_mn=None,
    lambda_update_scale,
    dt_eff,
    b1,
    fac,
    force_scale,
    flip_sign,
    vRcc_before,
    vRss_before,
    vZsc_before,
    vZcs_before,
    vLsc_before,
    vLcs_before,
    vRsc_before=None,
    vRcs_before=None,
    vZcc_before=None,
    vZss_before=None,
    vLcc_before=None,
    vLss_before=None,
    max_update_rms=5.0e-3,
    limit_update_rms: bool = True,
    need_update_rms: bool = True,
    divide_by_scalxc_for_update: bool = False,
    preconditioner_jmax_override: int | None = None,
    preconditioner_use_precomputed_tridi: bool | None = None,
    preconditioner_use_lax_tridi: bool | None = None,
    jit_preconditioner_apply: bool = True,
    freeb_bsqvac_half=None,
    freeb_pres_scale=None,
    constraint_tcon0=None,
    constraint_tcon=None,
    constraint_precond_diag=None,
    constraint_precond_active=None,
    constraint_tcon_active=None,
    constraint_rcon0=None,
    constraint_zcon0=None,
    enforce_edge: bool = True,
):
    """Compose the exact QH one-step map from state through accepted update."""
    residual_state = state_pre if force_state_pre is None else force_state_pre
    residual_out = raw_force_residual_from_state(
        residual_state,
        static,
        wout_like=wout_like,
        trig=trig,
        apply_lforbal=apply_lforbal,
        include_edge_residual=include_edge_residual,
        apply_m1_constraints=apply_m1_constraints,
        zero_m1=zero_m1,
        freeb_bsqvac_half=freeb_bsqvac_half,
        freeb_pres_scale=freeb_pres_scale,
        constraint_tcon0=constraint_tcon0,
        constraint_tcon=constraint_tcon,
        constraint_precond_diag=constraint_precond_diag,
        constraint_precond_active=constraint_precond_active,
        constraint_tcon_active=constraint_tcon_active,
        constraint_rcon0=constraint_rcon0,
        constraint_zcon0=constraint_zcon0,
    )
    preconditioner_out = None
    if mats is None or jmax is None or lam_prec is None or w_mode_mn is None:
        preconditioner_out = state_dependent_preconditioner_from_forces(
            k=residual_out["k"],
            static=static,
            trig=trig,
            dtype=jnp.asarray(state_pre.Rcos).dtype,
            jmax_override=preconditioner_jmax_override,
            use_precomputed=preconditioner_use_precomputed_tridi,
            use_lax_tridi=preconditioner_use_lax_tridi,
        )
        mats = preconditioner_out["mats"]
        jmax = preconditioner_out["jmax"]
        lam_prec = preconditioner_out["lam_prec"]
        w_mode_mn = preconditioner_out["w_mode_mn"]
    force_out = preconditioned_force_channels_from_raw_forces(
        frzl=residual_out["frzl"],
        mats=mats,
        jmax=jmax,
        cfg=static.cfg,
        lam_prec=lam_prec,
        w_mode_mn=w_mode_mn,
        lambda_update_scale=lambda_update_scale,
        use_precomputed=preconditioner_use_precomputed_tridi,
        use_lax_tridi=preconditioner_use_lax_tridi,
        jit_preconditioner_apply=jit_preconditioner_apply,
    )
    step_out = strict_update_accepted_step(
        state_pre,
        static,
        dt_eff=dt_eff,
        b1=b1,
        fac=fac,
        force_scale=force_scale,
        flip_sign=flip_sign,
        vRcc_before=vRcc_before,
        vRss_before=vRss_before,
        vZsc_before=vZsc_before,
        vZcs_before=vZcs_before,
        vLsc_before=vLsc_before,
        vLcs_before=vLcs_before,
        vRsc_before=vRsc_before,
        vRcs_before=vRcs_before,
        vZcc_before=vZcc_before,
        vZss_before=vZss_before,
        vLcc_before=vLcc_before,
        vLss_before=vLss_before,
        frcc_u=force_out["frcc_u"],
        frss_u=force_out["frss_u"],
        fzsc_u=force_out["fzsc_u"],
        fzcs_u=force_out["fzcs_u"],
        flsc_u=force_out["flsc_u"],
        flcs_u=force_out["flcs_u"],
        frsc_u=force_out.get("frsc_u"),
        frcs_u=force_out.get("frcs_u"),
        fzcc_u=force_out.get("fzcc_u"),
        fzss_u=force_out.get("fzss_u"),
        flcc_u=force_out.get("flcc_u"),
        flss_u=force_out.get("flss_u"),
        max_update_rms=max_update_rms,
        limit_update_rms=limit_update_rms,
        need_update_rms=need_update_rms,
        divide_by_scalxc_for_update=divide_by_scalxc_for_update,
        enforce_edge=enforce_edge,
    )
    return {
        "residual": residual_out,
        "preconditioner": preconditioner_out,
        "force": force_out,
        "step": step_out,
    }


def strict_update_one_step_from_trace(
    state_pre,
    static,
    trace: dict[str, Any],
    *,
    scalar_controls: dict[str, Any] | None = None,
    array_controls: dict[str, Any] | None = None,
    preconditioner_controls: dict[str, Any] | None = None,
    freeb_bsqvac_half: Any = _TRACE_OVERRIDE_UNSET,
    freeb_pres_scale: Any = _TRACE_OVERRIDE_UNSET,
    enforce_edge: bool = True,
    jit_preconditioner_apply: bool = True,
) -> dict[str, Any]:
    """Replay one strict residual step using fields captured in a trace dict.

    This source helper is intentionally thin: it maps the diagnostic trace
    schema produced by ``adjoint_trace=True`` onto
    :func:`strict_update_one_step_from_state` and allows callers to replace the
    free-boundary ``bsqvac`` channel with a differentiable replay.  Optional
    ``scalar_controls``, ``array_controls``, and ``preconditioner_controls``
    let JAX-visible controller scans pass step-sliced update controls without
    changing the default trace-dictionary contract.  It keeps phase-2
    direct-coil validation tests from duplicating trace plumbing while
    preserving the explicit accepted-step contract.
    """

    def _control(key: str) -> Any:
        if scalar_controls is not None and key in scalar_controls:
            return scalar_controls[key]
        if array_controls is not None and key in array_controls:
            return array_controls[key]
        return trace[key]

    def _optional_control(key: str) -> Any:
        if array_controls is not None and key in array_controls:
            return array_controls[key]
        return trace.get(key)

    def _preconditioner_control(key: str) -> Any:
        if preconditioner_controls is not None and key in preconditioner_controls:
            return preconditioner_controls[key]
        return trace[key]

    bsqvac = trace.get("freeb_bsqvac_half", None) if freeb_bsqvac_half is _TRACE_OVERRIDE_UNSET else freeb_bsqvac_half
    pres_scale = trace.get("freeb_pres_scale", None) if freeb_pres_scale is _TRACE_OVERRIDE_UNSET else freeb_pres_scale
    force_state_pre = trace.get("force_state_pre", None)
    return strict_update_one_step_from_state(
        state_pre,
        static,
        force_state_pre=force_state_pre,
        wout_like=trace["wout_like"],
        trig=trace["trig"],
        apply_lforbal=trace["apply_lforbal"],
        include_edge_residual=trace["include_edge_residual"],
        apply_m1_constraints=trace["apply_m1_constraints"],
        zero_m1=trace["zero_m1"],
        mats=_preconditioner_control("precond_mats"),
        jmax=trace["precond_jmax"],
        lam_prec=_preconditioner_control("lam_prec"),
        w_mode_mn=_preconditioner_control("w_mode_mn"),
        lambda_update_scale=_control("lambda_update_scale"),
        dt_eff=_control("dt_eff"),
        b1=_control("b1"),
        fac=_control("fac"),
        force_scale=_control("force_scale"),
        flip_sign=_control("flip_sign"),
        vRcc_before=_control("vRcc_before"),
        vRss_before=_control("vRss_before"),
        vZsc_before=_control("vZsc_before"),
        vZcs_before=_control("vZcs_before"),
        vLsc_before=_control("vLsc_before"),
        vLcs_before=_control("vLcs_before"),
        vRsc_before=_optional_control("vRsc_before"),
        vRcs_before=_optional_control("vRcs_before"),
        vZcc_before=_optional_control("vZcc_before"),
        vZss_before=_optional_control("vZss_before"),
        vLcc_before=_optional_control("vLcc_before"),
        vLss_before=_optional_control("vLss_before"),
        max_update_rms=_control("max_update_rms_pre"),
        limit_update_rms=_control("limit_update_rms"),
        divide_by_scalxc_for_update=_control("divide_by_scalxc_for_update"),
        preconditioner_use_precomputed_tridi=_control("preconditioner_use_precomputed_tridi"),
        preconditioner_use_lax_tridi=_control("preconditioner_use_lax_tridi"),
        jit_preconditioner_apply=jit_preconditioner_apply,
        freeb_bsqvac_half=bsqvac,
        freeb_pres_scale=pres_scale,
        constraint_rcon0=trace.get("constraint_rcon0"),
        constraint_zcon0=trace.get("constraint_zcon0"),
        constraint_tcon0=trace.get("constraint_tcon0"),
        constraint_precond_diag=trace.get("constraint_precond_diag"),
        constraint_tcon=trace.get("constraint_tcon"),
        constraint_precond_active=trace.get("constraint_precond_active"),
        constraint_tcon_active=trace.get("constraint_tcon_active"),
        enforce_edge=bool(enforce_edge),
    )


__all__ = [
    "ResidualIterationTrace",
    "ResidualCheckpointTape",
    "build_residual_checkpoint_tape",
    "checkpoint_tape_param_jvp",
    "checkpoint_tape_param_vjp",
    "checkpoint_tape_state_jvp",
    "checkpoint_tape_state_vjp",
    "concat_residual_iteration_traces",
    "preconditioned_force_channels_from_raw_forces",
    "preconditioned_force_channels_from_rz_output",
    "raw_force_residual_from_state",
    "replay_scan_cache_diagnostics",
    "replay_residual_checkpoint_step",
    "residual_branch_fingerprint",
    "strict_update_accepted_step",
    "strict_update_one_step_from_trace",
    "strict_update_one_step_from_state",
    "strict_update_velocity_limit",
    "strict_update_velocity_block",
    "strict_update_velocity_state_advance",
    "residual_iteration_trace_from_result",
]
