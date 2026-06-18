"""Utilities for the discrete-adjoint recovery path.

The first step is a structured view of the existing fixed-boundary residual
iteration history. This keeps the initial refactor narrow: no solver behavior
changes, only a stable extraction layer over the primal trace data already
recorded in diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
import os
import time
from typing import Any

import numpy as np

from ._compat import jax, jnp
from .state import pack_state, unpack_state
from .vmec_tomnsp import TomnspsRZL

_REPLAY_STEP_TRACE_KEYS = (
    "state_pre",
    "wout_like",
    "trig",
    "zero_m1",
    "precond_mats",
    "precond_jmax",
    "lam_prec",
    "w_mode_mn",
    "lambda_update_scale",
    "dt_eff",
    "b1",
    "fac",
    "force_scale",
    "flip_sign",
    "time_step",
    "fsq_prev_before",
    "reset_inv_tau",
    "constraint_cache_update",
    "precond_cache_update",
    "inv_tau_before",
    "max_coeff_delta_rms_pre",
    "vRcc_before",
    "vRss_before",
    "vRsc_before",
    "vRcs_before",
    "vZsc_before",
    "vZcs_before",
    "vZcc_before",
    "vZss_before",
    "vLsc_before",
    "vLcs_before",
    "vLcc_before",
    "vLss_before",
    "max_update_rms_pre",
    "freeb_bsqvac_half",
    "freeb_pres_scale",
    "constraint_tcon0",
    "constraint_precond_diag",
    "constraint_tcon",
    "constraint_precond_active",
    "constraint_tcon_active",
    "constraint_rcon0",
    "constraint_zcon0",
)
_OPTIONAL_REPLAY_STEP_TRACE_KEYS = (
    "freeb_bsqvac_half",
    "freeb_pres_scale",
    "constraint_tcon0",
    "constraint_precond_diag",
    "constraint_tcon",
    "constraint_precond_active",
    "constraint_tcon_active",
    "constraint_rcon0",
    "constraint_zcon0",
)

_REPLAY_STEP_TRACE_STATIC_KEYS = (
    "apply_lforbal",
    "include_edge_residual",
    "apply_m1_constraints",
    "limit_update_rms",
    "limit_dt_from_force",
    "vmec2000_control",
    "divide_by_scalxc_for_update",
    "signgs",
)
_DYNAMIC_REPLAY_SCALAR_TRACE_KEYS = (
    "lambda_update_scale",
    "max_coeff_delta_rms_pre",
    "max_update_rms_pre",
)
_CHECKPOINT_TAPE_SCAN_CACHE: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
_CHECKPOINT_TAPE_DYNAMIC_SCAN_CACHE: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
_CHECKPOINT_TAPE_DYNAMIC_BASEPOINT_SCAN_CACHE: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
_CHECKPOINT_TAPE_DYNAMIC_BASEPOINT_VJP_SCAN_CACHE: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
_REPLAY_SCAN_CACHE_LABELS: tuple[str, ...] = (
    "checkpoint",
    "dynamic",
    "dynamic_basepoint",
    "dynamic_basepoint_vjp",
)
_REPLAY_SCAN_CACHE_DIAGNOSTICS: dict[str, int | float] = {
    f"replay_{label}_scan_cache_{suffix}": 0.0 if suffix.endswith("_s") else 0
    for label in _REPLAY_SCAN_CACHE_LABELS
    for suffix in ("lookup_s", "build_s", "hit_count", "miss_count")
}
_REPLAY_JVP_COLUMN_PATH_LABELS: tuple[str, ...] = (
    "identity",
    "dynamic_basepoint",
    "segmented_dynamic_basepoint",
    "dynamic_linearize",
    "dynamic_scan_linearize",
    "generic_per_trace",
    "generic_scan",
)
_REPLAY_SCAN_CACHE_DIAGNOSTICS.update(
    {
        **{f"replay_jvp_columns_{label}_count": 0 for label in _REPLAY_JVP_COLUMN_PATH_LABELS},
        "replay_jvp_columns_leaf_call_count": 0,
        "replay_jvp_columns_input_column_count": 0,
        "replay_jvp_columns_chunked_call_count": 0,
        "replay_jvp_columns_chunk_count": 0,
        "replay_jvp_columns_last_chunk_size": 0,
    }
)


@dataclass(frozen=True)
class _DynamicBasepointScanRunner:
    from_carry: Any
    from_state_tangents: Any

    def __call__(self, carry_tangents0, stacked_base_carries_in, stacked_traces_in):
        return self.from_carry(carry_tangents0, stacked_base_carries_in, stacked_traces_in)

    def zero_aux(self, state_tangents0, stacked_base_carries_in, stacked_traces_in):
        return self.from_state_tangents(state_tangents0, stacked_base_carries_in, stacked_traces_in)


_DIRECT_TAPE_TIMING_KEYS = (
    "tape_solve_call_s",
    "tape_final_state_pack_s",
    "tape_step_trace_extract_s",
    "tape_dynamic_payload_build_s",
    "tape_trace_stack_s",
)
_TRACE_OVERRIDE_UNSET = object()


def _scan_cache_limit() -> int:
    env = os.getenv("VMEC_JAX_SCAN_CACHE_LIMIT", "").strip()
    if not env:
        return 8
    try:
        value = int(env)
    except Exception:
        return 8
    return max(1, value)


def _lru_cache_get(cache: OrderedDict[tuple[Any, ...], Any], key: tuple[Any, ...]):
    value = cache.get(key)
    if value is None:
        return None
    cache.move_to_end(key)
    return value


def _lru_cache_put(cache: OrderedDict[tuple[Any, ...], Any], key: tuple[Any, ...], value):
    cache[key] = value
    cache.move_to_end(key)
    limit = _scan_cache_limit()
    while len(cache) > limit:
        cache.popitem(last=False)


def replay_scan_cache_diagnostics(*, reset: bool = False) -> dict[str, int | float]:
    """Return optional replay-scan cache diagnostics used for performance triage."""
    out = dict(_REPLAY_SCAN_CACHE_DIAGNOSTICS)
    if bool(reset):
        for key in _REPLAY_SCAN_CACHE_DIAGNOSTICS:
            _REPLAY_SCAN_CACHE_DIAGNOSTICS[key] = 0.0 if key.endswith("_s") else 0
    return out


def _replay_scan_cache_diagnostics_enabled() -> bool:
    env = os.getenv("VMEC_JAX_TIMING", "").strip().lower()
    return env not in ("", "0", "false", "no")


def _record_replay_scan_cache_lookup(
    label: str,
    *,
    hit: bool,
    lookup_s: float,
    build_s: float = 0.0,
) -> None:
    if not _replay_scan_cache_diagnostics_enabled():
        return
    prefix = f"replay_{label}_scan_cache"
    _REPLAY_SCAN_CACHE_DIAGNOSTICS[f"{prefix}_lookup_s"] += float(lookup_s)
    _REPLAY_SCAN_CACHE_DIAGNOSTICS[f"{prefix}_build_s"] += float(build_s)
    count_key = f"{prefix}_{'hit' if bool(hit) else 'miss'}_count"
    _REPLAY_SCAN_CACHE_DIAGNOSTICS[count_key] = int(_REPLAY_SCAN_CACHE_DIAGNOSTICS[count_key]) + 1


def _record_replay_jvp_columns_path(label: str, *, n_columns: int) -> None:
    if not _replay_scan_cache_diagnostics_enabled():
        return
    key = f"replay_jvp_columns_{label}_count"
    if key not in _REPLAY_SCAN_CACHE_DIAGNOSTICS:
        return
    _REPLAY_SCAN_CACHE_DIAGNOSTICS[key] = int(_REPLAY_SCAN_CACHE_DIAGNOSTICS[key]) + 1
    _REPLAY_SCAN_CACHE_DIAGNOSTICS["replay_jvp_columns_leaf_call_count"] = (
        int(_REPLAY_SCAN_CACHE_DIAGNOSTICS["replay_jvp_columns_leaf_call_count"]) + 1
    )
    _REPLAY_SCAN_CACHE_DIAGNOSTICS["replay_jvp_columns_input_column_count"] = (
        int(_REPLAY_SCAN_CACHE_DIAGNOSTICS["replay_jvp_columns_input_column_count"]) + max(0, int(n_columns))
    )


def _record_replay_jvp_columns_chunking(*, n_chunks: int, chunk_size: int) -> None:
    if not _replay_scan_cache_diagnostics_enabled():
        return
    _REPLAY_SCAN_CACHE_DIAGNOSTICS["replay_jvp_columns_chunked_call_count"] = (
        int(_REPLAY_SCAN_CACHE_DIAGNOSTICS["replay_jvp_columns_chunked_call_count"]) + 1
    )
    _REPLAY_SCAN_CACHE_DIAGNOSTICS["replay_jvp_columns_chunk_count"] = (
        int(_REPLAY_SCAN_CACHE_DIAGNOSTICS["replay_jvp_columns_chunk_count"]) + max(0, int(n_chunks))
    )
    _REPLAY_SCAN_CACHE_DIAGNOSTICS["replay_jvp_columns_last_chunk_size"] = max(0, int(chunk_size))


def clear_replay_scan_caches() -> None:
    """Drop cached replay-scan runners to release compiled executable refs."""
    _CHECKPOINT_TAPE_SCAN_CACHE.clear()
    _CHECKPOINT_TAPE_DYNAMIC_SCAN_CACHE.clear()
    _CHECKPOINT_TAPE_DYNAMIC_BASEPOINT_SCAN_CACHE.clear()
    _CHECKPOINT_TAPE_DYNAMIC_BASEPOINT_VJP_SCAN_CACHE.clear()
    replay_scan_cache_diagnostics(reset=True)


_DEFAULT_REPLAY_COLUMN_TARGET_MB = 4096.0


def _backend_is_accelerator(backend: str) -> bool:
    """Return true for JAX accelerator backend names that should stay on device."""

    normalized = str(backend).strip().lower()
    return normalized in {"gpu", "cuda", "rocm"} or normalized.startswith(("gpu:", "cuda:", "rocm:"))


def _dynamic_replay_bucket_default() -> int:
    try:
        backend = str(jax.default_backend()).lower()
    except Exception:
        backend = ""
    # GPU/XLA currently compiles and replays the basepoint scan faster when
    # the dynamic tape length is bucketed more coarsely. CPU profiling shows
    # the smaller bucket is still faster there, so keep the default
    # backend-sensitive instead of requiring users to tune an environment
    # variable for normal CPU/GPU optimization runs.
    return 128 if _backend_is_accelerator(backend) else 32


def _dynamic_replay_bucket_size() -> int:
    env = os.getenv("VMEC_JAX_DYNAMIC_REPLAY_BUCKET", "").strip()
    if not env:
        return _dynamic_replay_bucket_default()
    try:
        bucket = int(env)
    except Exception:
        return _dynamic_replay_bucket_default()
    return max(1, bucket)


def _dynamic_replay_bucket_len(length: int) -> int:
    bucket = _dynamic_replay_bucket_size()
    if length <= 0:
        return 0
    return int(((int(length) + bucket - 1) // bucket) * bucket)


def _dynamic_replay_mode() -> str:
    """Replay linearization strategy for compact dynamic tapes.

    ``basepoint`` linearizes each VMEC step at the saved base carry inside the
    scan.  ``whole_scan`` differentiates the entire replay scan at once.  The
    default remains ``basepoint`` because it has been the most stable CPU path;
    the alternate path is useful for GPU profiling where XLA may make different
    fusion/transpose tradeoffs.
    """
    mode = os.environ.get("VMEC_JAX_DYNAMIC_REPLAY_MODE", "basepoint").strip().lower()
    if mode in ("whole_scan", "scan", "full_scan"):
        return "whole_scan"
    return "basepoint"


def _jvp_only_exact_tape_basepoint_carries_enabled() -> bool:
    flag = os.getenv("VMEC_JAX_JVP_ONLY_EXACT_TAPE_BASEPOINT_CARRIES", "").strip().lower()
    return flag in ("1", "true", "yes", "on")


def _replay_column_chunk_default(*, tape, tangents) -> int | None:
    """Return an automatic replay-column chunk size for large exact Jacobians.

    This is a memory-control fallback, not an accuracy feature. If the user
    sets ``VMEC_JAX_REPLAY_COLUMN_CHUNK`` explicitly we always respect it.
    """
    target_env = os.environ.get("VMEC_JAX_REPLAY_COLUMN_TARGET_MB", "").strip()
    if target_env == "":
        # The replay memory leak is fixed, so a 1 GB target is now too
        # conservative for the common 24/48-DOF QA/QH exact Jacobians. A
        # 4 GB heuristic keeps the seed mode-3 runs in the ~1-1.5 GB RSS range
        # on CPU while materially reducing replay segmentation overhead.
        target_mb = _DEFAULT_REPLAY_COLUMN_TARGET_MB
    else:
        target_mb = float(target_env)
    if target_mb <= 0.0:
        return None

    def _tree_nbytes(tree) -> int:
        if tree is None:
            return 0
        try:
            leaves = jax.tree_util.tree_leaves(tree)
        except Exception:
            return 0
        total = 0
        for leaf in leaves:
            arr = jnp.asarray(leaf)
            total += int(arr.size) * max(1, int(arr.dtype.itemsize))
        return total

    bytes_per_col = max(
        _tree_nbytes(getattr(tape, "dynamic_initial_carry", None)),
        _tree_nbytes(getattr(tape, "dynamic_base_carries_stacked", None)),
    )
    if bytes_per_col <= 0:
        return None

    ncols = int(tangents.shape[0])
    if ncols <= 1:
        return None

    target_bytes = int(target_mb * (1024**2))
    auto_chunk = max(1, min(ncols, target_bytes // bytes_per_col))
    if auto_chunk >= ncols:
        return None
    return int(auto_chunk)


def _replay_column_chunk_override(value: str | None) -> tuple[bool, int | None]:
    """Parse a user replay-column chunk override.

    Returns ``(handled, chunk)``.  ``handled=True, chunk=None`` explicitly
    disables chunking.  Malformed values are left unhandled so production
    profiling jobs fall back to the automatic memory guard instead of failing
    midway through tape replay.
    """
    if value is None:
        return False, None
    text = str(value).strip().lower()
    if text in ("", "auto", "default"):
        return False, None
    if text in ("0", "none", "off", "false", "no"):
        return True, None
    try:
        chunk = int(text)
    except (TypeError, ValueError):
        return False, None
    if chunk <= 0:
        return True, None
    return True, int(chunk)


def _tridi_policy_cache_value(value: bool | None) -> int:
    """Encode the replay tridiagonal policy for JIT-cache keys."""
    if value is None:
        return -1
    return 1 if bool(value) else 0


def _trace_preconditioner_use_precomputed_tridi(
    trace: dict[str, Any],
    static_flags: dict[str, Any] | None = None,
) -> bool | None:
    """Return the precomputed-Thomas policy recorded by the primal solve."""
    if static_flags is not None and "preconditioner_use_precomputed_tridi" in static_flags:
        value = static_flags["preconditioner_use_precomputed_tridi"]
    else:
        value = trace.get("preconditioner_use_precomputed_tridi", None)
    if value is None:
        return None
    return bool(value)


def _trace_preconditioner_use_lax_tridi(
    trace: dict[str, Any],
    static_flags: dict[str, Any] | None = None,
) -> bool | None:
    """Return the lax tridiagonal-solver policy recorded by the primal solve."""
    if static_flags is not None and "preconditioner_use_lax_tridi" in static_flags:
        value = static_flags["preconditioner_use_lax_tridi"]
    else:
        value = trace.get("preconditioner_use_lax_tridi", None)
    if value is None:
        return None
    return bool(value)


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


def _empty_trace() -> ResidualIterationTrace:
    empty_i = np.zeros((0,), dtype=int)
    empty_f = np.zeros((0,), dtype=float)
    empty_o = np.zeros((0,), dtype=object)
    empty_b = np.zeros((0,), dtype=bool)
    return ResidualIterationTrace(
        iter2=empty_i,
        step_status=empty_o,
        restart_reason=empty_o,
        pre_restart_reason=empty_o,
        time_step=empty_f,
        dt_eff=empty_f,
        update_rms=empty_f,
        include_edge=empty_i,
        zero_m1=empty_i,
        fsq_curr=empty_f,
        fsq_try=empty_f,
        fsq_prev=empty_f,
        r00=empty_f,
        z00=empty_f,
        wb=empty_f,
        wp=empty_f,
        w_vmec=empty_f,
        state_advanced=empty_b,
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

    iter2 = _array_from_diag(diagnostics, "iter2_history", dtype=int)
    step_status = _array_from_diag(diagnostics, "step_status_history", dtype=object)
    restart_reason = _array_from_diag(diagnostics, "restart_reason_history", dtype=object)
    pre_restart_reason = _array_from_diag(diagnostics, "pre_restart_reason_history", dtype=object)
    time_step = _array_from_diag(diagnostics, "time_step_history", dtype=float)
    dt_eff = _array_from_diag(diagnostics, "dt_eff_history", dtype=float)
    update_rms = _array_from_diag(diagnostics, "update_rms_history", dtype=float)
    include_edge = _array_from_diag(diagnostics, "include_edge_history", dtype=int)
    zero_m1 = _array_from_diag(diagnostics, "zero_m1_history", dtype=int)
    fsq_curr = _array_from_diag(diagnostics, "w_curr_history", dtype=float)
    fsq_try = _array_from_diag(diagnostics, "w_try_history", dtype=float)
    fsq_prev = _array_from_diag(diagnostics, "fsq_prev_history", dtype=float)
    r00 = _array_from_diag(diagnostics, "r00_history", dtype=float)
    z00 = _array_from_diag(diagnostics, "z00_history", dtype=float)
    wb = _array_from_diag(diagnostics, "wb_history", dtype=float)
    wp = _array_from_diag(diagnostics, "wp_history", dtype=float)
    w_vmec = _array_from_diag(diagnostics, "w_vmec_history", dtype=float)

    lengths = {
        int(arr.shape[0])
        for arr in (
            iter2,
            step_status,
            restart_reason,
            pre_restart_reason,
            time_step,
            dt_eff,
            update_rms,
            include_edge,
            zero_m1,
            fsq_curr,
            fsq_try,
            fsq_prev,
            r00,
            z00,
            wb,
            wp,
            w_vmec,
        )
        if arr.ndim >= 1 and arr.shape[0] > 0
    }
    if len(lengths) > 1:
        raise ValueError(f"inconsistent residual trace lengths: {sorted(lengths)}")

    rejected = np.isin(
        step_status,
        np.asarray(["rejected", "restart_bad_progress", "restart_bad_jacobian"], dtype=object),
    )
    state_advanced = ~rejected

    return ResidualIterationTrace(
        iter2=iter2,
        step_status=step_status,
        restart_reason=restart_reason,
        pre_restart_reason=pre_restart_reason,
        time_step=time_step,
        dt_eff=dt_eff,
        update_rms=update_rms,
        include_edge=include_edge,
        zero_m1=zero_m1,
        fsq_curr=fsq_curr,
        fsq_try=fsq_try,
        fsq_prev=fsq_prev,
        r00=r00,
        z00=z00,
        wb=wb,
        wp=wp,
        w_vmec=w_vmec,
        state_advanced=state_advanced,
    )


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
        iter2=_cat("iter2").astype(int, copy=False),
        step_status=_cat("step_status").astype(object, copy=False),
        restart_reason=_cat("restart_reason").astype(object, copy=False),
        pre_restart_reason=_cat("pre_restart_reason").astype(object, copy=False),
        time_step=_cat("time_step").astype(float, copy=False),
        dt_eff=_cat("dt_eff").astype(float, copy=False),
        update_rms=_cat("update_rms").astype(float, copy=False),
        include_edge=_cat("include_edge").astype(int, copy=False),
        zero_m1=_cat("zero_m1").astype(int, copy=False),
        fsq_curr=_cat("fsq_curr").astype(float, copy=False),
        fsq_try=_cat("fsq_try").astype(float, copy=False),
        fsq_prev=_cat("fsq_prev").astype(float, copy=False),
        r00=_cat("r00").astype(float, copy=False),
        z00=_cat("z00").astype(float, copy=False),
        wb=_cat("wb").astype(float, copy=False),
        wp=_cat("wp").astype(float, copy=False),
        w_vmec=_cat("w_vmec").astype(float, copy=False),
        state_advanced=_cat("state_advanced").astype(bool, copy=False),
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
    solver_kwargs = dict(solver_kwargs or {})
    solve_kwargs = dict(solver_kwargs)
    solve_kwargs.setdefault("indata", indata)
    solve_kwargs.setdefault("signgs", int(signgs))
    solve_kwargs.setdefault("ftol", ftol)
    solve_kwargs.setdefault("step_size", float(step_size))
    # Full scalar histories are only required when the caller intends to keep
    # the compact trace diagnostics. The replay/JVP path only needs the
    # step-level adjoint traces plus the internal resume checkpoint carried
    # between one-step solves.
    solve_kwargs["light_history"] = False if store_trace else bool(light_history)
    # Multi-step replay currently needs the cached preconditioner/control state
    # carried in the full resume checkpoint. A later optimization pass can
    # shrink this once exact replay coverage is in place.
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

    solver_kwargs = dict(solver_kwargs or {})
    solve_kwargs = dict(solver_kwargs)
    solve_kwargs.setdefault("indata", indata)
    solve_kwargs.setdefault("signgs", int(signgs))
    solve_kwargs.setdefault("ftol", ftol)
    solve_kwargs.setdefault("step_size", float(step_size))
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

    result = _solve_with_timing(solve_kwargs)
    pack_start = time.perf_counter()
    final_packed_state = jnp.asarray(pack_state(result.state), dtype=jnp.float64)
    _record_timing("tape_final_state_pack_s", pack_start)
    trace_extract_start = time.perf_counter()
    step_traces = tuple(result.diagnostics.get("adjoint_step_trace", ()))
    _record_timing("tape_step_trace_extract_s", trace_extract_start)
    compact_diagnostics = _compact_tape_diagnostics(result.diagnostics)
    trace = residual_iteration_trace_from_result(result) if store_trace else _empty_trace()
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
                pack_start = time.perf_counter()
                final_packed_state = jnp.asarray(pack_state(result.state), dtype=jnp.float64)
                _record_timing("tape_final_state_pack_s", pack_start)
                trace_extract_start = time.perf_counter()
                step_traces = tuple(result.diagnostics.get("adjoint_step_trace", ()))
                _record_timing("tape_step_trace_extract_s", trace_extract_start)
                compact_diagnostics = _compact_tape_diagnostics(result.diagnostics)
                trace = residual_iteration_trace_from_result(result) if store_trace else _empty_trace()
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
        zero_cotangent = lambda value: jax.tree_util.tree_map(lambda x: jnp.zeros_like(jnp.asarray(x)), value)
        final_carry_cotangents = (
            jnp.asarray(final_cotangent, dtype=jnp.asarray(carry0[0]).dtype),
            *(zero_cotangent(value) for value in carry0[1:]),
        )
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
        zero_cotangent = lambda value: jax.tree_util.tree_map(lambda x: jnp.zeros_like(jnp.asarray(x)), value)
        final_carry_cotangents = (
            jnp.asarray(final_cotangent, dtype=jnp.asarray(carry0[0]).dtype),
            *(zero_cotangent(value) for value in carry0[1:]),
        )
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
        if use_device_stack:
            return jnp.stack(arrays, axis=0)
        return np.stack(arrays, axis=0)

    stacked = jax.tree_util.tree_map(_stack_values, *filtered)
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

    def _stack_dynamic_values(*xs):
        arrays = [_as_stack_array(x) for x in xs]
        if use_device_stack:
            return jnp.stack(arrays, axis=0)
        return np.stack(arrays, axis=0)

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
        key: _stack_dynamic_values(*(trace[key] for trace in filtered))
        for key in filtered[0]
    }
    initial_carry = _dynamic_replay_initial_carry(step_traces[0])
    stacked_base_carries = None
    if store_base_carries:
        base_carries = (initial_carry,) + tuple(_dynamic_replay_initial_carry(trace) for trace in step_traces[1:])
        if target_len > len(base_carries):
            pad_carry = base_carries[-1]
            base_carries = base_carries + (pad_carry,) * (target_len - len(base_carries))
        stacked_base_carries = jax.tree_util.tree_map(_stack_dynamic_values, *base_carries)
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
    from .vmec_residue import vmec_gcx2_from_tomnsps, vmec_rz_norm_from_state

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
    wout_like = trace["wout_like"] if isinstance(trace, dict) and "wout_like" in trace else static_flags["wout_like"]
    trig = trace["trig"] if isinstance(trace, dict) and "trig" in trace else static_flags["trig"]
    w_mode_mn = trace["w_mode_mn"] if isinstance(trace, dict) and "w_mode_mn" in trace else static_flags["w_mode_mn"]
    lambda_update_scale = (
        trace["lambda_update_scale"]
        if isinstance(trace, dict) and "lambda_update_scale" in trace
        else static_flags["lambda_update_scale"]
    )
    max_coeff_delta_rms_pre = (
        trace["max_coeff_delta_rms_pre"]
        if isinstance(trace, dict) and "max_coeff_delta_rms_pre" in trace
        else static_flags["max_coeff_delta_rms_pre"]
    )
    max_update_rms_pre = (
        trace["max_update_rms_pre"]
        if isinstance(trace, dict) and "max_update_rms_pre" in trace
        else static_flags["max_update_rms_pre"]
    )
    def _optional_replay_value(key: str):
        if isinstance(trace, dict) and key in trace:
            return trace[key]
        return static_flags.get(key, None)

    constraint_precond_active = _optional_replay_value("constraint_precond_active")
    constraint_tcon_active = _optional_replay_value("constraint_tcon_active")
    if constraint_precond_active is None:
        constraint_precond_active = jnp.asarray(False)
    if constraint_tcon_active is None:
        constraint_tcon_active = jnp.asarray(False)
    constraint_precond_diag_current = (
        jnp.where(
            jnp.asarray(constraint_precond_active, dtype=bool),
            jnp.asarray(constraint_ard1_before),
            jnp.zeros_like(jnp.asarray(constraint_ard1_before)),
        ),
        jnp.where(
            jnp.asarray(constraint_precond_active, dtype=bool),
            jnp.asarray(constraint_azd1_before),
            jnp.zeros_like(jnp.asarray(constraint_azd1_before)),
        ),
    )
    constraint_tcon_current = jnp.where(
        jnp.asarray(constraint_tcon_active, dtype=bool),
        jnp.asarray(constraint_tcon_before),
        jnp.zeros_like(jnp.asarray(constraint_tcon_before)),
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
        constraint_tcon0=_optional_replay_value("constraint_tcon0"),
        constraint_tcon=constraint_tcon_current,
        constraint_precond_diag=constraint_precond_diag_current,
        constraint_precond_active=constraint_precond_active,
        constraint_tcon_active=constraint_tcon_active,
        constraint_rcon0=_optional_replay_value("constraint_rcon0"),
        constraint_zcon0=_optional_replay_value("constraint_zcon0"),
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
    constraint_cache_update = (
        trace["constraint_cache_update"]
        if isinstance(trace, dict) and "constraint_cache_update" in trace
        else False
    )
    precond_cache_update = (
        trace["precond_cache_update"]
        if isinstance(trace, dict) and "precond_cache_update" in trace
        else constraint_cache_update
    )
    update_precond_cache = jnp.asarray(precond_cache_update, dtype=bool)
    lam_prec_current = jnp.where(
        update_precond_cache,
        jnp.asarray(refreshed_preconditioner_out["lam_prec"]),
        jnp.asarray(cache_lam_prec_before),
    )
    mats_current = jax.tree_util.tree_map(
        lambda refreshed, cached: jnp.where(
            update_precond_cache,
            jnp.asarray(refreshed),
            jnp.asarray(cached),
        ),
        refreshed_preconditioner_out["mats"],
        cache_prec_mats_before,
    )
    preconditioner_out = {
        "lam_prec": lam_prec_current,
        "mats": mats_current,
        "jmax": refreshed_preconditioner_out["jmax"],
        "w_mode_mn": refreshed_preconditioner_out["w_mode_mn"],
    }
    constraint_tcon0 = _optional_replay_value("constraint_tcon0")
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
    update_cache = jnp.asarray(constraint_cache_update, dtype=bool)
    constraint_ard1_next = jnp.where(update_cache, refreshed_ard1, jnp.asarray(constraint_ard1_before))
    constraint_azd1_next = jnp.where(update_cache, refreshed_azd1, jnp.asarray(constraint_azd1_before))
    constraint_tcon_next = jnp.where(update_cache, refreshed_tcon, jnp.asarray(constraint_tcon_before))
    cache_lam_prec_next = lam_prec_current
    cache_prec_mats_next = mats_current
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
    diagnostics_enabled = _replay_scan_cache_diagnostics_enabled()
    lookup_start = time.perf_counter() if diagnostics_enabled else None
    cached = _lru_cache_get(_CHECKPOINT_TAPE_SCAN_CACHE, key)
    lookup_s = time.perf_counter() - float(lookup_start) if lookup_start is not None else 0.0
    if cached is not None:
        _record_replay_scan_cache_lookup("checkpoint", hit=True, lookup_s=lookup_s)
        return cached
    build_start = time.perf_counter() if diagnostics_enabled else None

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

    _lru_cache_put(_CHECKPOINT_TAPE_SCAN_CACHE, key, _run_scan)
    _record_replay_scan_cache_lookup(
        "checkpoint",
        hit=False,
        lookup_s=lookup_s,
        build_s=time.perf_counter() - float(build_start) if build_start is not None else 0.0,
    )
    return _run_scan


def _checkpoint_tape_dynamic_scan_runner(*, static, stacked, static_flags):
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
    diagnostics_enabled = _replay_scan_cache_diagnostics_enabled()
    lookup_start = time.perf_counter() if diagnostics_enabled else None
    cached = _lru_cache_get(_CHECKPOINT_TAPE_DYNAMIC_SCAN_CACHE, key)
    lookup_s = time.perf_counter() - float(lookup_start) if lookup_start is not None else 0.0
    if cached is not None:
        _record_replay_scan_cache_lookup("dynamic", hit=True, lookup_s=lookup_s)
        return cached
    build_start = time.perf_counter() if diagnostics_enabled else None

    def _step_scan(carry, trace):
        active = jnp.asarray(trace["active"], dtype=bool) if "active" in trace else jnp.asarray(True, dtype=bool)

        def _advance(carry_in):
            return _packed_dynamic_replay_step_from_carry(
                carry_in,
                trace,
                static=static,
                static_flags=static_flags,
                preconditioner_jmax_override=int(static_flags["precond_jmax"]),
                preconditioner_use_precomputed_tridi=static_flags.get(
                    "preconditioner_use_precomputed_tridi",
                    None,
                ),
                preconditioner_use_lax_tridi=static_flags.get(
                    "preconditioner_use_lax_tridi",
                    None,
                ),
            )

        carry = jax.lax.cond(active, _advance, lambda carry_in: carry_in, carry)
        return carry, None

    @jax.jit
    def _run_scan(carry0, stacked_traces):
        carry, _ = jax.lax.scan(_step_scan, carry0, stacked_traces)
        return carry

    _lru_cache_put(_CHECKPOINT_TAPE_DYNAMIC_SCAN_CACHE, key, _run_scan)
    _record_replay_scan_cache_lookup(
        "dynamic",
        hit=False,
        lookup_s=lookup_s,
        build_s=time.perf_counter() - float(build_start) if build_start is not None else 0.0,
    )
    return _run_scan


def _checkpoint_tape_dynamic_basepoint_scan_runner(*, static, stacked, stacked_base_carries, static_flags):
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
        _stacked_trace_signature(stacked_base_carries),
    )
    diagnostics_enabled = _replay_scan_cache_diagnostics_enabled()
    lookup_start = time.perf_counter() if diagnostics_enabled else None
    cached = _lru_cache_get(_CHECKPOINT_TAPE_DYNAMIC_BASEPOINT_SCAN_CACHE, key)
    lookup_s = time.perf_counter() - float(lookup_start) if lookup_start is not None else 0.0
    if cached is not None:
        _record_replay_scan_cache_lookup("dynamic_basepoint", hit=True, lookup_s=lookup_s)
        return cached
    build_start = time.perf_counter() if diagnostics_enabled else None

    def _step_scan(pair, trace):
        carry_base, carry_tangents = pair
        active = jnp.asarray(trace["active"], dtype=bool) if "active" in trace else jnp.asarray(True, dtype=bool)

        def _step(carry):
            return _packed_dynamic_replay_step_from_carry(
                carry,
                trace,
                static=static,
                static_flags=static_flags,
                preconditioner_jmax_override=int(static_flags["precond_jmax"]),
                preconditioner_use_precomputed_tridi=static_flags.get(
                    "preconditioner_use_precomputed_tridi",
                    None,
                ),
                preconditioner_use_lax_tridi=static_flags.get(
                    "preconditioner_use_lax_tridi",
                    None,
                ),
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

        def _zeros_like_base(arr):
            return jax.tree_util.tree_map(
                lambda x: jnp.zeros((state_tangents0.shape[0],) + jnp.asarray(x).shape, dtype=jnp.asarray(x).dtype),
                arr,
            )

        carry_tangents0 = (state_tangents0,) + tuple(_zeros_like_base(arr) for arr in carry0[1:])
        return _scan_from_carry(carry_tangents0, stacked_base_carries_in, stacked_traces_in)

    runner = _DynamicBasepointScanRunner(
        from_carry=_run_scan,
        from_state_tangents=_run_scan_zero_aux,
    )
    _lru_cache_put(_CHECKPOINT_TAPE_DYNAMIC_BASEPOINT_SCAN_CACHE, key, runner)
    _record_replay_scan_cache_lookup(
        "dynamic_basepoint",
        hit=False,
        lookup_s=lookup_s,
        build_s=time.perf_counter() - float(build_start) if build_start is not None else 0.0,
    )
    return runner


def _run_dynamic_basepoint_scan_zero_aux(*, run_scan, state_tangents, stacked_base_carries, stacked):
    run_zero_aux = getattr(run_scan, "zero_aux", None)
    if run_zero_aux is not None:
        return run_zero_aux(state_tangents, stacked_base_carries, stacked)

    carry0 = jax.tree_util.tree_map(lambda x: x[0], stacked_base_carries)

    def _zeros_like_base(arr):
        return jax.tree_util.tree_map(
            lambda x: jnp.zeros((state_tangents.shape[0],) + jnp.asarray(x).shape, dtype=jnp.asarray(x).dtype),
            arr,
        )

    carry_tangents0 = (state_tangents,) + tuple(_zeros_like_base(arr) for arr in carry0[1:])
    return run_scan(carry_tangents0, stacked_base_carries, stacked)


def _checkpoint_tape_dynamic_basepoint_vjp_scan_runner(*, static, stacked, stacked_base_carries, static_flags):
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
        _stacked_trace_signature(stacked_base_carries),
    )
    diagnostics_enabled = _replay_scan_cache_diagnostics_enabled()
    lookup_start = time.perf_counter() if diagnostics_enabled else None
    cached = _lru_cache_get(_CHECKPOINT_TAPE_DYNAMIC_BASEPOINT_VJP_SCAN_CACHE, key)
    lookup_s = time.perf_counter() - float(lookup_start) if lookup_start is not None else 0.0
    if cached is not None:
        _record_replay_scan_cache_lookup("dynamic_basepoint_vjp", hit=True, lookup_s=lookup_s)
        return cached
    build_start = time.perf_counter() if diagnostics_enabled else None

    def _step_scan(carry_cotangents, inputs):
        carry_base, trace = inputs
        active = jnp.asarray(trace["active"], dtype=bool) if "active" in trace else jnp.asarray(True, dtype=bool)

        def _step(carry):
            return _packed_dynamic_replay_step_from_carry(
                carry,
                trace,
                static=static,
                static_flags=static_flags,
                preconditioner_jmax_override=int(static_flags["precond_jmax"]),
                preconditioner_use_precomputed_tridi=static_flags.get(
                    "preconditioner_use_precomputed_tridi",
                    None,
                ),
                preconditioner_use_lax_tridi=static_flags.get(
                    "preconditioner_use_lax_tridi",
                    None,
                ),
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

    _lru_cache_put(_CHECKPOINT_TAPE_DYNAMIC_BASEPOINT_VJP_SCAN_CACHE, key, _run_scan)
    _record_replay_scan_cache_lookup(
        "dynamic_basepoint_vjp",
        hit=False,
        lookup_s=lookup_s,
        build_s=time.perf_counter() - float(build_start) if build_start is not None else 0.0,
    )
    return _run_scan


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
                    carry0 = _dynamic_replay_initial_carry(trace)
                    zeros_like = lambda arr: jax.tree_util.tree_map(
                        lambda x: jnp.zeros((tangents.shape[0],) + jnp.asarray(x).shape, dtype=jnp.asarray(x).dtype),
                        arr,
                    )
                    carry_tangents = (tangents,) + tuple(zeros_like(arr) for arr in carry0[1:])
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
        zeros_like = lambda arr: jax.tree_util.tree_map(
            lambda x: jnp.zeros((tangents.shape[0],) + jnp.asarray(x).shape, dtype=jnp.asarray(x).dtype),
            arr,
        )
        carry_tangents = (tangents,) + tuple(zeros_like(arr) for arr in carry0[1:])
        for trace in tape.step_traces:
            carry_base = _dynamic_replay_initial_carry(trace)

            def _step(carry):
                return _packed_dynamic_replay_step_from_carry(
                    carry,
                    trace,
                    static=static,
                    static_flags=trace if static_flags is None else static_flags,
                    preconditioner_jmax_override=(
                        int(trace["precond_jmax"])
                        if static_flags is None or static_flags.get("precond_jmax") is None
                        else int(static_flags["precond_jmax"])
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

            _, linear_step = jax.linearize(_step, carry_base)
            carry_tangents = jax.vmap(linear_step)(carry_tangents)
        _record_replay_jvp_columns_path("dynamic_linearize", n_columns=int(tangents.shape[0]))
        return carry_tangents[0]
    if tape.dynamic_initial_carry is not None and stacked is not None and static_flags is not None and rebuild_preconditioner:
        carry0 = tape.dynamic_initial_carry
        zeros_like = lambda arr: jax.tree_util.tree_map(
            lambda x: jnp.zeros((tangents.shape[0],) + jnp.asarray(x).shape, dtype=jnp.asarray(x).dtype),
            arr,
        )
        carry_tangents0 = (tangents,) + tuple(zeros_like(arr) for arr in carry0[1:])
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


def strict_update_velocity_block(
    *,
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
    frcc_u,
    frss_u,
    fzsc_u,
    fzcs_u,
    flsc_u,
    flcs_u,
    vRsc_before=None,
    vRcs_before=None,
    vZcc_before=None,
    vZss_before=None,
    vLcc_before=None,
    vLss_before=None,
    frsc_u=None,
    frcs_u=None,
    fzcc_u=None,
    fzss_u=None,
    flcc_u=None,
    flss_u=None,
):
    """Apply the strict-update velocity recurrence for one solver step."""
    b1 = jnp.asarray(b1, dtype=jnp.asarray(vRcc_before).dtype)
    fac = jnp.asarray(fac, dtype=jnp.asarray(vRcc_before).dtype)
    force_scale = jnp.asarray(force_scale, dtype=jnp.asarray(vRcc_before).dtype)
    flip_sign = jnp.asarray(flip_sign, dtype=jnp.asarray(vRcc_before).dtype)
    scale = fac * force_scale * flip_sign
    memory = fac * b1

    def _update(v_before, force):
        return memory * jnp.asarray(v_before) + scale * jnp.asarray(force)

    vRcc_after = _update(vRcc_before, frcc_u)
    vRss_after = _update(vRss_before, frss_u)
    vZsc_after = _update(vZsc_before, fzsc_u)
    vZcs_after = _update(vZcs_before, fzcs_u)
    vLsc_after = _update(vLsc_before, flsc_u)
    vLcs_after = _update(vLcs_before, flcs_u)
    if vRsc_before is None:
        vRsc_after = None
        vRcs_after = None
        vZcc_after = None
        vZss_after = None
        vLcc_after = None
        vLss_after = None
    else:
        vRsc_after = _update(vRsc_before, frsc_u)
        vRcs_after = _update(vRcs_before, frcs_u)
        vZcc_after = _update(vZcc_before, fzcc_u)
        vZss_after = _update(vZss_before, fzss_u)
        vLcc_after = _update(vLcc_before, flcc_u)
        vLss_after = _update(vLss_before, flss_u)
    return {
        "vRcc_after": vRcc_after,
        "vRss_after": vRss_after,
        "vZsc_after": vZsc_after,
        "vZcs_after": vZcs_after,
        "vLsc_after": vLsc_after,
        "vLcs_after": vLcs_after,
        "vRsc_after": vRsc_after,
        "vRcs_after": vRcs_after,
        "vZcc_after": vZcc_after,
        "vZss_after": vZss_after,
        "vLcc_after": vLcc_after,
        "vLss_after": vLss_after,
    }


def strict_update_velocity_limit(
    *,
    dt_eff,
    max_update_rms,
    limit_update_rms,
    need_update_rms: bool = True,
    vRcc,
    vRss,
    vZsc,
    vZcs,
    vLsc,
    vLcs,
    vRsc=None,
    vRcs=None,
    vZcc=None,
    vZss=None,
    vLcc=None,
    vLss=None,
):
    """Apply the strict-update velocity RMS limiter for one solver step."""
    dt_eff = jnp.asarray(dt_eff, dtype=jnp.asarray(vRcc).dtype)
    max_update_rms = jnp.asarray(max_update_rms, dtype=jnp.asarray(vRcc).dtype)
    limit_update_rms = jnp.asarray(limit_update_rms, dtype=bool)
    need_update_rms = jnp.asarray(need_update_rms, dtype=bool)
    base = jnp.asarray(vRcc)
    zeros = jnp.zeros_like(base)
    pieces = [
        base,
        jnp.asarray(vRss),
        jnp.asarray(vRsc) if vRsc is not None else zeros,
        jnp.asarray(vRcs) if vRcs is not None else zeros,
        jnp.asarray(vZsc),
        jnp.asarray(vZcs),
        jnp.asarray(vZcc) if vZcc is not None else zeros,
        jnp.asarray(vZss) if vZss is not None else zeros,
        jnp.asarray(vLsc),
        jnp.asarray(vLcs),
        jnp.asarray(vLcc) if vLcc is not None else zeros,
        jnp.asarray(vLss) if vLss is not None else zeros,
    ]
    sq = sum((dt_eff * p) ** 2 for p in pieces)
    raw_update_rms = jnp.sqrt(jnp.mean(sq))
    report_update_rms = jnp.where(
        jnp.logical_or(limit_update_rms, need_update_rms),
        raw_update_rms,
        jnp.asarray(0.0, dtype=raw_update_rms.dtype),
    )
    clipped_scale = jnp.where(
        jnp.isfinite(raw_update_rms) & (raw_update_rms > max_update_rms),
        max_update_rms / jnp.maximum(raw_update_rms, jnp.asarray(1.0e-30, dtype=raw_update_rms.dtype)),
        jnp.asarray(1.0, dtype=raw_update_rms.dtype),
    )
    scale = jnp.where(limit_update_rms, clipped_scale, jnp.asarray(1.0, dtype=raw_update_rms.dtype))

    def _scale(x):
        return None if x is None else scale * jnp.asarray(x)

    out = {
        "vRcc": _scale(vRcc),
        "vRss": _scale(vRss),
        "vZsc": _scale(vZsc),
        "vZcs": _scale(vZcs),
        "vLsc": _scale(vLsc),
        "vLcs": _scale(vLcs),
        "vRsc": _scale(vRsc),
        "vRcs": _scale(vRcs),
        "vZcc": _scale(vZcc),
        "vZss": _scale(vZss),
        "vLcc": _scale(vLcc),
        "vLss": _scale(vLss),
        "update_rms_preclip": report_update_rms,
        "update_rms_scale": scale,
        "update_rms_postclip": scale * report_update_rms,
    }
    return out


def preconditioned_force_channels_from_rz_output(
    *,
    frzl_rz,
    lam_prec,
    w_mode_mn,
    lambda_update_scale=1.0,
):
    """Map R/Z preconditioner output into solver force channels for one step."""
    frcc = jnp.asarray(frzl_rz.frcc)
    zeros_r = jnp.zeros_like(frcc)
    fzsc = jnp.asarray(frzl_rz.fzsc)
    zeros_z = jnp.zeros_like(fzsc)
    flsc = jnp.asarray(frzl_rz.flsc) * jnp.asarray(lam_prec)
    zeros_l = jnp.zeros_like(flsc)

    frss = None if frzl_rz.frss is None else jnp.asarray(frzl_rz.frss)
    fzcs = None if frzl_rz.fzcs is None else jnp.asarray(frzl_rz.fzcs)
    flcs = None if frzl_rz.flcs is None else (jnp.asarray(frzl_rz.flcs) * jnp.asarray(lam_prec))
    frsc = jnp.asarray(frzl_rz.frsc) if getattr(frzl_rz, "frsc", None) is not None else zeros_r
    frcs = jnp.asarray(frzl_rz.frcs) if getattr(frzl_rz, "frcs", None) is not None else zeros_r
    fzcc = jnp.asarray(frzl_rz.fzcc) if getattr(frzl_rz, "fzcc", None) is not None else zeros_z
    fzss = jnp.asarray(frzl_rz.fzss) if getattr(frzl_rz, "fzss", None) is not None else zeros_z
    flcc = (
        jnp.asarray(frzl_rz.flcc) * jnp.asarray(lam_prec)
        if getattr(frzl_rz, "flcc", None) is not None
        else zeros_l
    )
    flss = (
        jnp.asarray(frzl_rz.flss) * jnp.asarray(lam_prec)
        if getattr(frzl_rz, "flss", None) is not None
        else zeros_l
    )

    frzl_pre = TomnspsRZL(
        frcc=frcc,
        frss=frss,
        fzsc=fzsc,
        fzcs=fzcs,
        flsc=flsc,
        flcs=flcs,
        frsc=frsc,
        frcs=frcs,
        fzcc=fzcc,
        fzss=fzss,
        flcc=flcc,
        flss=flss,
    )
    w = jnp.asarray(w_mode_mn)[None, :, :]
    frcc_u = frcc * w
    frss_u = (frss if frss is not None else zeros_r) * w
    fzsc_u = fzsc * w
    fzcs_u = (fzcs if fzcs is not None else zeros_z) * w
    flsc_u = flsc * w
    flcs_u = (flcs if flcs is not None else zeros_l) * w
    frsc_u = frsc * w
    frcs_u = frcs * w
    fzcc_u = fzcc * w
    fzss_u = fzss * w
    flcc_u = flcc * w
    flss_u = flss * w
    scale = jnp.asarray(lambda_update_scale, dtype=flsc_u.dtype)
    flsc_u = flsc_u * scale
    flcs_u = flcs_u * scale
    flcc_u = flcc_u * scale
    flss_u = flss_u * scale
    return {
        "frzl_pre": frzl_pre,
        "frcc_u": frcc_u,
        "frss_u": frss_u,
        "fzsc_u": fzsc_u,
        "fzcs_u": fzcs_u,
        "flsc_u": flsc_u,
        "flcs_u": flcs_u,
        "frsc_u": frsc_u,
        "frcs_u": frcs_u,
        "fzcc_u": fzcc_u,
        "fzss_u": fzss_u,
        "flcc_u": flcc_u,
        "flss_u": flss_u,
    }


def preconditioned_force_channels_from_raw_forces(
    *,
    frzl,
    mats,
    jmax,
    cfg,
    lam_prec,
    w_mode_mn,
    lambda_update_scale=1.0,
    use_precomputed: bool | None = None,
    use_lax_tridi: bool | None = None,
    jit_preconditioner_apply: bool = True,
):
    """Apply the radial preconditioner, lambda scaling, and mode scaling."""
    from .preconditioner_1d_jax import rz_preconditioner_apply, rz_preconditioner_apply_jit
    from .solve import _scale_mode_slice, _vmec_scale_m1_factors_from_mats

    frzl_rhs = frzl
    if bool(getattr(cfg, "lconm1", True)) and int(cfg.mpol) > 1:
        fac_r, fac_z = _vmec_scale_m1_factors_from_mats(mats)
        if int(jnp.asarray(fac_r).size) > 0:
            fac_r = jnp.asarray(fac_r, dtype=jnp.asarray(frzl.frcc).dtype)
            fac_z = jnp.asarray(fac_z, dtype=jnp.asarray(frzl.fzsc).dtype)
            ns_full = int(jnp.asarray(frzl.frcc).shape[0])
            nsolve = min(ns_full, int(fac_r.shape[0]))
            ones_r = jnp.ones((max(ns_full - nsolve, 0),), dtype=jnp.asarray(frzl.frcc).dtype)
            ones_z = jnp.ones((max(ns_full - nsolve, 0),), dtype=jnp.asarray(frzl.fzsc).dtype)
            fac_r_full = fac_r[:nsolve] if nsolve == ns_full else jnp.concatenate([fac_r[:nsolve], ones_r], axis=0)
            fac_z_full = fac_z[:nsolve] if nsolve == ns_full else jnp.concatenate([fac_z[:nsolve], ones_z], axis=0)
            frzl_rhs = TomnspsRZL(
                frcc=frzl.frcc,
                frss=_scale_mode_slice(frzl.frss, mode_idx=1, scale=fac_r_full),
                fzsc=frzl.fzsc,
                fzcs=_scale_mode_slice(frzl.fzcs, mode_idx=1, scale=fac_z_full),
                flsc=frzl.flsc,
                flcs=frzl.flcs,
                frsc=_scale_mode_slice(getattr(frzl, "frsc", None), mode_idx=1, scale=fac_r_full),
                frcs=getattr(frzl, "frcs", None),
                fzcc=_scale_mode_slice(getattr(frzl, "fzcc", None), mode_idx=1, scale=fac_z_full),
                fzss=getattr(frzl, "fzss", None),
                flcc=getattr(frzl, "flcc", None),
                flss=getattr(frzl, "flss", None),
            )

    apply_preconditioner = rz_preconditioner_apply_jit if bool(jit_preconditioner_apply) else rz_preconditioner_apply
    frzl_rz = apply_preconditioner(
        frzl_in=frzl_rhs,
        mats=mats,
        jmax=int(jmax),
        cfg=cfg,
        use_precomputed=use_precomputed,
        use_lax_tridi=use_lax_tridi,
    )
    out = preconditioned_force_channels_from_rz_output(
        frzl_rz=frzl_rz,
        lam_prec=lam_prec,
        w_mode_mn=w_mode_mn,
        lambda_update_scale=lambda_update_scale,
    )
    return {"frzl_rhs": frzl_rhs, "frzl_rz": frzl_rz, **out}


def raw_force_residual_from_state(
    state,
    static,
    *,
    wout_like,
    trig,
    apply_lforbal: bool,
    include_edge_residual: bool,
    apply_m1_constraints: bool,
    zero_m1,
    constraint_tcon0=None,
    constraint_tcon=None,
    constraint_precond_diag=None,
    constraint_precond_active=None,
    constraint_tcon_active=None,
    constraint_rcon0=None,
    constraint_zcon0=None,
    freeb_bsqvac_half=None,
    freeb_pres_scale=None,
):
    """Rebuild the solver's raw VMEC residual blocks for one fixed-boundary step."""
    from dataclasses import replace as dc_replace

    from .vmec_forces import vmec_forces_rz_from_wout, vmec_residual_internal_from_kernels
    from .vmec_residue import vmec_apply_m1_constraints, vmec_apply_scalxc_to_tomnsps, vmec_zero_m1_zforce

    k = vmec_forces_rz_from_wout(
        state=state,
        static=static,
        wout=wout_like,
        indata=None,
        constraint_tcon0=constraint_tcon0,
        constraint_tcon=constraint_tcon,
        constraint_precond_diag=constraint_precond_diag,
        constraint_precond_active=constraint_precond_active,
        constraint_tcon_active=constraint_tcon_active,
        constraint_rcon0=constraint_rcon0,
        constraint_zcon0=constraint_zcon0,
        freeb_bsqvac_half=freeb_bsqvac_half,
        freeb_pres_scale=freeb_pres_scale,
        use_vmec_synthesis=True,
        trig=trig,
        iter_idx=None,
    )
    mask_pack = None
    if getattr(static, "tomnsps_masks", None) is not None:
        mask_pack = static.tomnsps_masks_edge if bool(include_edge_residual) else static.tomnsps_masks
    frzl = vmec_residual_internal_from_kernels(
        k,
        cfg_ntheta=int(static.cfg.ntheta),
        cfg_nzeta=int(static.cfg.nzeta),
        wout=wout_like,
        trig=trig,
        apply_lforbal=bool(apply_lforbal),
        include_edge=bool(include_edge_residual),
        masks=mask_pack,
    )
    if bool(apply_m1_constraints):
        frzl = vmec_apply_m1_constraints(frzl=frzl, lconm1=bool(getattr(static.cfg, "lconm1", True)))
    frzl = vmec_zero_m1_zforce(frzl=frzl, enabled=zero_m1)
    frzl = vmec_apply_scalxc_to_tomnsps(frzl=frzl, s=jnp.asarray(static.s))

    def _nan_guard(x):
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            return x
        x = jnp.asarray(x)
        return jnp.where(jnp.isnan(x), x, x)

    frzl = dc_replace(
        frzl,
        frcc=_nan_guard(frzl.frcc),
        frss=_nan_guard(frzl.frss),
        fzsc=_nan_guard(frzl.fzsc),
        fzcs=_nan_guard(frzl.fzcs),
        flsc=_nan_guard(frzl.flsc),
        flcs=_nan_guard(frzl.flcs),
        frsc=_nan_guard(getattr(frzl, "frsc", None)),
        frcs=_nan_guard(getattr(frzl, "frcs", None)),
        fzcc=_nan_guard(getattr(frzl, "fzcc", None)),
        fzss=_nan_guard(getattr(frzl, "fzss", None)),
        flcc=_nan_guard(getattr(frzl, "flcc", None)),
        flss=_nan_guard(getattr(frzl, "flss", None)),
    )
    return {"k": k, "frzl": frzl}


def state_dependent_preconditioner_from_forces(
    *,
    k,
    static,
    trig,
    dtype=None,
    jmax_override: int | None = None,
    w_mode_mn=None,
    mode_diag_exponent: float = 0.0,
    use_precomputed: bool | None = None,
    use_lax_tridi: bool | None = None,
):
    """Rebuild the solver's state-dependent preconditioner objects."""
    from .preconditioner_1d_jax import lambda_preconditioner, rz_preconditioner_matrices

    cfg = static.cfg
    s = jnp.asarray(static.s)
    lam_prec = lambda_preconditioner(
        bc=k.bc,
        trig=trig,
        s=s,
        cfg=cfg,
        return_faclam=False,
    )
    mats, _jmin, jmax = rz_preconditioner_matrices(
        bc=k.bc,
        k=k,
        trig=trig,
        s=s,
        cfg=cfg,
        jmax_override=jmax_override,
        use_precomputed=use_precomputed,
        use_lax_tridi=use_lax_tridi,
    )
    if dtype is None:
        dtype = jnp.asarray(lam_prec).dtype
    if w_mode_mn is None:
        m = jnp.arange(int(cfg.mpol), dtype=jnp.float64)
        n = jnp.arange(int(cfg.ntor) + 1, dtype=jnp.float64) * float(cfg.nfp)
        k2 = (m[:, None] * m[:, None]) + (n[None, :] * n[None, :])
        w_mode_mn = (1.0 + k2).astype(jnp.float64) ** (-float(mode_diag_exponent))
        w_mode_mn = w_mode_mn.astype(dtype)
    else:
        w_mode_mn = jnp.asarray(w_mode_mn, dtype=dtype)
    return {
        "lam_prec": lam_prec,
        "mats": mats,
        "jmax": int(jmax_override) if (jmax_override is not None) else int(jmax),
        "w_mode_mn": w_mode_mn,
    }


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


def strict_update_accepted_step(
    state_pre,
    static,
    *,
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
    frcc_u,
    frss_u,
    fzsc_u,
    fzcs_u,
    flsc_u,
    flcs_u,
    vRsc_before=None,
    vRcs_before=None,
    vZcc_before=None,
    vZss_before=None,
    vLcc_before=None,
    vLss_before=None,
    frsc_u=None,
    frcs_u=None,
    fzcc_u=None,
    fzss_u=None,
    flcc_u=None,
    flss_u=None,
    max_update_rms=5.0e-3,
    limit_update_rms: bool = True,
    need_update_rms: bool = True,
    divide_by_scalxc_for_update: bool = False,
    enforce_edge: bool = True,
):
    """Compose the accepted strict-update velocity and state-advance blocks.

    ``enforce_edge`` preserves the historical fixed-boundary default.  Free
    boundary callers pass ``False`` so the fused update does not pin the LCFS.
    """
    velocity_raw = strict_update_velocity_block(
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
        frcc_u=frcc_u,
        frss_u=frss_u,
        fzsc_u=fzsc_u,
        fzcs_u=fzcs_u,
        flsc_u=flsc_u,
        flcs_u=flcs_u,
        vRsc_before=vRsc_before,
        vRcs_before=vRcs_before,
        vZcc_before=vZcc_before,
        vZss_before=vZss_before,
        vLcc_before=vLcc_before,
        vLss_before=vLss_before,
        frsc_u=frsc_u,
        frcs_u=frcs_u,
        fzcc_u=fzcc_u,
        fzss_u=fzss_u,
        flcc_u=flcc_u,
        flss_u=flss_u,
    )
    velocity_out = strict_update_velocity_limit(
        dt_eff=dt_eff,
        max_update_rms=max_update_rms,
        limit_update_rms=limit_update_rms,
        need_update_rms=need_update_rms,
        vRcc=velocity_raw["vRcc_after"],
        vRss=velocity_raw["vRss_after"],
        vZsc=velocity_raw["vZsc_after"],
        vZcs=velocity_raw["vZcs_after"],
        vLsc=velocity_raw["vLsc_after"],
        vLcs=velocity_raw["vLcs_after"],
        vRsc=velocity_raw["vRsc_after"],
        vRcs=velocity_raw["vRcs_after"],
        vZcc=velocity_raw["vZcc_after"],
        vZss=velocity_raw["vZss_after"],
        vLcc=velocity_raw["vLcc_after"],
        vLss=velocity_raw["vLss_after"],
    )
    state_post = strict_update_velocity_state_advance(
        state_pre,
        static,
        dt_eff=dt_eff,
        vRcc=velocity_out["vRcc"],
        vRss=velocity_out["vRss"],
        vZsc=velocity_out["vZsc"],
        vZcs=velocity_out["vZcs"],
        vLsc=velocity_out["vLsc"],
        vLcs=velocity_out["vLcs"],
        vRsc=velocity_out["vRsc"],
        vRcs=velocity_out["vRcs"],
        vZcc=velocity_out["vZcc"],
        vZss=velocity_out["vZss"],
        vLcc=velocity_out["vLcc"],
        vLss=velocity_out["vLss"],
        edge_Rcos=jnp.asarray(state_pre.Rcos)[-1, :],
        edge_Rsin=jnp.asarray(state_pre.Rsin)[-1, :],
        edge_Zcos=jnp.asarray(state_pre.Zcos)[-1, :],
        edge_Zsin=jnp.asarray(state_pre.Zsin)[-1, :],
        divide_by_scalxc_for_update=divide_by_scalxc_for_update,
        enforce_edge=enforce_edge,
    )
    return {
        "state_post": state_post,
        "vRcc_after": velocity_out["vRcc"],
        "vRss_after": velocity_out["vRss"],
        "vZsc_after": velocity_out["vZsc"],
        "vZcs_after": velocity_out["vZcs"],
        "vLsc_after": velocity_out["vLsc"],
        "vLcs_after": velocity_out["vLcs"],
        "vRsc_after": velocity_out["vRsc"],
        "vRcs_after": velocity_out["vRcs"],
        "vZcc_after": velocity_out["vZcc"],
        "vZss_after": velocity_out["vZss"],
        "vLcc_after": velocity_out["vLcc"],
        "vLss_after": velocity_out["vLss"],
        "update_rms_preclip": velocity_out["update_rms_preclip"],
        "update_rms_scale": velocity_out["update_rms_scale"],
        "update_rms_postclip": velocity_out["update_rms_postclip"],
    }


def strict_update_velocity_state_advance(
    state,
    static,
    *,
    dt_eff,
    vRcc,
    vRss,
    vZsc,
    vZcs,
    vLsc,
    vLcs,
    edge_Rcos,
    edge_Rsin,
    edge_Zcos,
    edge_Zsin,
    vRsc=None,
    vRcs=None,
    vZcc=None,
    vZss=None,
    vLcc=None,
    vLss=None,
    divide_by_scalxc_for_update: bool = False,
    enforce_edge: bool = True,
):
    """Apply the strict-update state-advance block from VMEC residual iteration.

    This is the accepted geometry/lambda update map after the velocity blocks
    have already been formed for the current step. It excludes force assembly
    and acceptance/restart logic and is intended as the first local reverse-mode
    target for the discrete-adjoint refactor.
    """
    from .solve import _apply_vmec_lambda_axis_rules_to_state, _enforce_fixed_boundary_and_axis, _mode00_index
    from .vmec_parity import _mn_cos_to_signed_cached, _mn_sin_to_signed_cached, signed_maps_from_modes
    from .vmec_residue import vmec_scalxc_from_s

    dt_eff = jnp.asarray(dt_eff, dtype=jnp.asarray(state.Rcos).dtype)
    scalxc = vmec_scalxc_from_s(s=jnp.asarray(static.s), mpol=int(static.cfg.mpol)).astype(jnp.asarray(state.Rcos).dtype)
    scalxc = scalxc[:, :, None]
    scalxc = jnp.where(jnp.asarray(divide_by_scalxc_for_update, dtype=bool), scalxc, jnp.ones_like(scalxc))
    maps = static.signed_maps if getattr(static, "signed_maps", None) is not None else signed_maps_from_modes(static.modes)
    ncoeff = int(static.modes.K)
    idx00 = _mode00_index(static.modes)

    def _cos_phys(cc, ss):
        cc = jnp.asarray(cc) / scalxc
        ss = jnp.asarray(ss) / scalxc if ss is not None else None
        return _mn_cos_to_signed_cached(cc, ss, maps=maps, ncoeff=ncoeff)

    def _sin_phys(sc, cs):
        sc = jnp.asarray(sc) / scalxc
        cs = jnp.asarray(cs) / scalxc if cs is not None else None
        return _mn_sin_to_signed_cached(sc, cs, maps=maps, ncoeff=ncoeff)

    dR = dt_eff * _cos_phys(vRcc, vRss)
    dZ = dt_eff * _sin_phys(vZsc, vZcs)
    dL = dt_eff * _sin_phys(vLsc, vLcs)
    if bool(static.cfg.lasym):
        dR_sin = dt_eff * _sin_phys(vRsc, vRcs)
        dZ_cos = dt_eff * _cos_phys(vZcc, vZss)
        dL_cos = dt_eff * _cos_phys(vLcc, vLss)
    else:
        dR_sin = jnp.zeros_like(dR)
        dZ_cos = jnp.zeros_like(dR)
        dL_cos = jnp.zeros_like(dR)

    state_try = type(state)(
        layout=state.layout,
        Rcos=jnp.asarray(state.Rcos) + dR,
        Rsin=jnp.asarray(state.Rsin) + dR_sin,
        Zcos=jnp.asarray(state.Zcos) + dZ_cos,
        Zsin=jnp.asarray(state.Zsin) + dZ,
        Lcos=jnp.asarray(state.Lcos) + dL_cos,
        Lsin=jnp.asarray(state.Lsin) + dL,
    )
    state_out = _enforce_fixed_boundary_and_axis(
        state_try,
        static,
        edge_Rcos=edge_Rcos,
        edge_Rsin=edge_Rsin,
        edge_Zcos=edge_Zcos,
        edge_Zsin=edge_Zsin,
        enforce_edge=enforce_edge,
        enforce_lambda_axis=True,
        idx00=idx00,
    )
    return _apply_vmec_lambda_axis_rules_to_state(
        state_out,
        enforce_vmec_lambda_axis=True,
        host_update_assembly=False,
        idx00=idx00,
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
