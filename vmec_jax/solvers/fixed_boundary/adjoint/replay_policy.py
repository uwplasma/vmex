"""Replay-cache and runtime policy helpers for fixed-boundary adjoints."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import os
import time
from typing import Any

from vmec_jax._compat import jax, jnp

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
    from_initial_carry: Any | None = None
    from_state_tangents_initial: Any | None = None

    def __call__(self, carry_tangents0, stacked_base_carries_in, stacked_traces_in):
        """Evaluate this callable objective for fixed-boundary VMEC solve and implicit differentiation."""
        return self.from_carry(carry_tangents0, stacked_base_carries_in, stacked_traces_in)

    def zero_aux(self, state_tangents0, stacked_base_carries_in, stacked_traces_in):
        """Evaluate zero aux for fixed-boundary VMEC solve and implicit differentiation."""
        return self.from_state_tangents(state_tangents0, stacked_base_carries_in, stacked_traces_in)

    def initial_carry(self, carry_tangents0, carry0_in, stacked_traces_in):
        """Replay from one initial base carry, avoiding full base-history inputs."""
        if self.from_initial_carry is None:
            raise AttributeError("initial-carry replay is not available")
        return self.from_initial_carry(carry_tangents0, carry0_in, stacked_traces_in)

    def zero_aux_initial(self, state_tangents0, carry0_in, stacked_traces_in):
        """Replay zero auxiliary tangents from one initial base carry."""
        if self.from_state_tangents_initial is None:
            raise AttributeError("initial-carry zero-aux replay is not available")
        return self.from_state_tangents_initial(state_tangents0, carry0_in, stacked_traces_in)


_DIRECT_TAPE_TIMING_KEYS = (
    "tape_solve_call_s",
    "tape_final_state_pack_s",
    "tape_step_trace_extract_s",
    "tape_dynamic_payload_build_s",
    "tape_trace_stack_s",
)
_TRACE_OVERRIDE_UNSET = object()


def _positive_int_env(name: str, default) -> int:
    fallback = int(default() if callable(default) else default)
    raw = os.getenv(name, "").strip()
    if not raw:
        return fallback
    try:
        value = int(raw)
    except Exception:
        return fallback
    return max(1, value)


def _scan_cache_limit() -> int:
    return _positive_int_env("VMEC_JAX_SCAN_CACHE_LIMIT", 8)


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


def _get_replay_scan_runner(
    label: str,
    cache: OrderedDict[tuple[Any, ...], Any],
    key: tuple[Any, ...],
):
    diagnostics_enabled = _replay_scan_cache_diagnostics_enabled()
    lookup_start = time.perf_counter() if diagnostics_enabled else None
    cached = _lru_cache_get(cache, key)
    lookup_s = time.perf_counter() - float(lookup_start) if lookup_start is not None else 0.0
    if cached is not None:
        _record_replay_scan_cache_lookup(label, hit=True, lookup_s=lookup_s)
        return cached, None
    return None, (lookup_s, time.perf_counter() if diagnostics_enabled else None)


def _put_replay_scan_runner(
    label: str,
    cache: OrderedDict[tuple[Any, ...], Any],
    key: tuple[Any, ...],
    runner,
    miss,
):
    lookup_s, build_start = miss
    _lru_cache_put(cache, key, runner)
    build_s = 0.0 if build_start is None else time.perf_counter() - build_start
    _record_replay_scan_cache_lookup(label, hit=False, lookup_s=lookup_s, build_s=build_s)
    return runner


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


def _add_replay_diag(key: str, amount: int = 1) -> None:
    _REPLAY_SCAN_CACHE_DIAGNOSTICS[key] = int(_REPLAY_SCAN_CACHE_DIAGNOSTICS[key]) + max(0, int(amount))


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
    _add_replay_diag(count_key)


def _record_replay_jvp_columns_path(label: str, *, n_columns: int) -> None:
    if not _replay_scan_cache_diagnostics_enabled():
        return
    key = f"replay_jvp_columns_{label}_count"
    if key not in _REPLAY_SCAN_CACHE_DIAGNOSTICS:
        return
    _add_replay_diag(key)
    _add_replay_diag("replay_jvp_columns_leaf_call_count")
    _add_replay_diag("replay_jvp_columns_input_column_count", n_columns)


def _record_replay_jvp_columns_chunking(*, n_chunks: int, chunk_size: int) -> None:
    if not _replay_scan_cache_diagnostics_enabled():
        return
    _add_replay_diag("replay_jvp_columns_chunked_call_count")
    _add_replay_diag("replay_jvp_columns_chunk_count", n_chunks)
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
    return _positive_int_env("VMEC_JAX_DYNAMIC_REPLAY_BUCKET", _dynamic_replay_bucket_default)


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


def _trace_bool_policy(trace: dict[str, Any], static_flags: dict[str, Any] | None, key: str) -> bool | None:
    value = static_flags[key] if static_flags is not None and key in static_flags else trace.get(key, None)
    return None if value is None else bool(value)


def _trace_preconditioner_use_precomputed_tridi(
    trace: dict[str, Any],
    static_flags: dict[str, Any] | None = None,
) -> bool | None:
    """Return the precomputed-Thomas policy recorded by the primal solve."""
    return _trace_bool_policy(trace, static_flags, "preconditioner_use_precomputed_tridi")


def _trace_preconditioner_use_lax_tridi(
    trace: dict[str, Any],
    static_flags: dict[str, Any] | None = None,
) -> bool | None:
    """Return the lax tridiagonal-solver policy recorded by the primal solve."""
    return _trace_bool_policy(trace, static_flags, "preconditioner_use_lax_tridi")
