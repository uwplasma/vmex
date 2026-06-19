"""Pure runtime helpers for :mod:`vmec_jax.solve`.

This module keeps environment and tracing policy helpers out of the main solver
implementation while avoiding imports of solver physics code.
"""

from __future__ import annotations

from dataclasses import fields
import hashlib
import os
from typing import Any, NamedTuple

import numpy as np

from ._compat import has_jax, jax


class ScanFallbackPolicy(NamedTuple):
    enabled: bool
    iters: int
    badjac_limit: int
    fsq_abs: float
    accept_frac: float
    fsq_factor: float
    improve: float


def _dataclass_from_namespace(cls, namespace: dict[str, Any], /, *, label: str, overrides: dict[str, Any]):
    """Build a dataclass context from local variables plus explicit overrides."""

    names = [field.name for field in fields(cls)]
    missing = [name for name in names if name not in overrides and name not in namespace]
    if missing:
        raise KeyError(f"Missing {label} context fields: {', '.join(missing)}")
    return cls(**{name: overrides[name] if name in overrides else namespace[name] for name in names})


def _hash_array_bytes(a: Any) -> str:
    try:
        arr = np.asarray(a)
    except Exception:
        try:
            typeof = getattr(jax, "typeof", None)
            aval = typeof(a) if typeof is not None else jax.core.get_aval(a)
        except Exception:
            aval = None
        if aval is None:
            return f"opaque:{type(a).__name__}"
        shape = tuple(getattr(aval, "shape", ()))
        dtype = getattr(aval, "dtype", None)
        weak_type = bool(getattr(aval, "weak_type", False))
        return f"traced:{shape}:{dtype}:{weak_type}"
    h = hashlib.blake2b(digest_size=16)
    h.update(arr.tobytes())
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())
    return h.hexdigest()


def _array_signature_key(a: Any) -> tuple[tuple[int, ...], str]:
    """Return the JIT-relevant shape/dtype signature for an array-like value."""
    try:
        arr = np.asarray(a)
        return tuple(int(v) for v in arr.shape), str(arr.dtype)
    except Exception:
        try:
            typeof = getattr(jax, "typeof", None)
            aval = typeof(a) if typeof is not None else jax.core.get_aval(a)
        except Exception:
            aval = None
        if aval is None:
            return (), type(a).__name__
        shape = tuple(int(v) for v in getattr(aval, "shape", ()))
        dtype = getattr(aval, "dtype", None)
        return shape, str(dtype)


def _edge_signature_key(*edges: Any) -> tuple[tuple[tuple[int, ...], str], ...]:
    """Key fixed-boundary edge rows by shape/dtype, not by coefficient values."""
    return tuple(_array_signature_key(edge) for edge in edges)


def _edge_value_key(*edges: Any) -> tuple[str, ...]:
    """Key fixed-boundary edge rows by values for legacy value-specialized paths."""
    return tuple(_hash_array_bytes(edge) for edge in edges)


def _tree_has_tracer(tree: Any) -> bool:
    if jax is None:
        return False
    try:
        leaves = jax.tree_util.tree_leaves(tree)
    except Exception:
        leaves = (tree,)
    return any(isinstance(leaf, jax.core.Tracer) for leaf in leaves)


def _scan_backend_name() -> str:
    if not has_jax() or jax is None:
        return "cpu"
    try:
        return str(jax.default_backend()).strip().lower() or "cpu"
    except Exception:
        return "cpu"


def _scan_chunk_settings(
    *,
    max_iter_scan: int,
    nstep_screen: int,
    need_print: bool,
    lthreed: bool,
    backend_name: str | None = None,
    chunk_size_env: str | None = None,
    spectral_mode_count: int | None = None,
) -> tuple[int, bool]:
    chunk_size_env = os.getenv("VMEC_JAX_SCAN_CHUNK_SIZE", "") if chunk_size_env is None else chunk_size_env
    chunk_size_env = str(chunk_size_env).strip()
    backend = _scan_backend_name() if backend_name is None else str(backend_name).strip().lower()
    long_quiet_accelerator = (
        backend not in ("", "cpu")
        and not bool(need_print)
        and int(max_iter_scan) > 512
    )
    if chunk_size_env:
        try:
            chunk_size = max(1, int(chunk_size_env))
        except Exception:
            chunk_size = max(1, int(nstep_screen))
    elif (backend == "cpu") and (not bool(need_print)):
        # Quiet CPU scan runs do not need host-side chunk boundaries for
        # printing, so use one remaining-iteration chunk and avoid the Python
        # chunk loop overhead entirely.
        chunk_size = max(1, int(max_iter_scan))
    elif long_quiet_accelerator:
        # Quiet accelerator runs can be dominated by compiling and dispatching
        # one very large scan executable.  Keep a fixed 512-iteration chunk so
        # the body is smaller and reused inside the solve.  The 2026-05-30
        # office RTX A4000 sweep showed this is neutral for the low-mode QH
        # warm-start case and much faster for the finite-beta high-mode case.
        chunk_size = min(max(1, int(max_iter_scan)), 512)
    elif (backend != "cpu") and (not bool(need_print)):
        # Short quiet accelerator runs use the full budget as one chunk; for
        # longer runs, the branch above intentionally limits chunk size.
        chunk_size = max(1, int(max_iter_scan))
    else:
        chunk_size = max(1, int(nstep_screen))
    cap_to_remaining = (not bool(need_print)) and (not long_quiet_accelerator)
    return chunk_size, cap_to_remaining


def _default_scan_core(*, scan_core_env: str, scan_minimal: bool, fsq_total_target: float | None) -> bool:
    """Choose the lean scan carry for accelerated quiet runs unless overridden."""
    if scan_core_env:
        return scan_core_env not in ("", "0", "false", "no")
    return bool(scan_minimal) and (fsq_total_target is not None)


def _parse_iter_list(val: str) -> set[int] | None:
    """Parse a comma-separated list of ints/ranges like '1,2,5-7'."""
    if not val:
        return None
    out: set[int] = set()
    for chunk in val.replace(" ", "").split(","):
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            try:
                lo = int(a)
                hi = int(b)
            except ValueError:
                continue
            if hi < lo:
                lo, hi = hi, lo
            out.update(range(lo, hi + 1))
        else:
            try:
                out.add(int(chunk))
            except ValueError:
                continue
    return out if out else None


def _dump_env_enabled(val: str) -> bool:
    """Match solve.py's legacy debug-dump enablement check exactly."""
    return bool(val) and val != "0"


def _dump_iter_selected(*, iter_idx: int, iter_env: str) -> bool:
    """Return whether a dump should run for an iteration env selector."""
    iters = _parse_iter_list(iter_env)
    return iters is None or int(iter_idx) in iters


def _runtime_env_enabled(val: str) -> bool:
    """Parse modern runtime env flags that use strip/lower false tokens."""
    return str(val).strip().lower() not in ("", "0", "false", "no")


def _scan_fallback_policy(
    *,
    backend_name: str,
    enabled_env: str | None,
    iters_env: str,
    badjac_limit_env: str,
    fsq_abs_env: str,
    accept_frac_env: str,
    fsq_factor_env: str,
    improve_env: str,
) -> ScanFallbackPolicy:
    """Parse scan fallback env policy while preserving legacy defaults/clamps."""
    backend = str(backend_name).strip().lower()
    default_enabled = "1" if backend == "cpu" else "0"
    enabled = _runtime_env_enabled(default_enabled if enabled_env is None else enabled_env)

    try:
        iters = max(1, int(str(iters_env).strip()))
    except Exception:
        iters = 20
    try:
        badjac_limit = max(0, int(str(badjac_limit_env).strip()))
    except Exception:
        badjac_limit = 10
    try:
        fsq_abs = float(str(fsq_abs_env).strip())
    except Exception:
        fsq_abs = 1.0e-2
    if fsq_abs < 0.0:
        fsq_abs = 0.0

    try:
        accept_frac = float(str(accept_frac_env).strip())
    except Exception:
        accept_frac = 0.5
    try:
        fsq_factor = float(str(fsq_factor_env).strip())
    except Exception:
        fsq_factor = 50.0
    try:
        improve = float(str(improve_env).strip())
    except Exception:
        improve = 0.9

    if accept_frac < 0.0:
        accept_frac = 0.0
    if accept_frac > 1.0:
        accept_frac = 1.0
    if fsq_factor < 1.0:
        fsq_factor = 1.0
    if improve <= 0.0 or improve >= 1.0:
        improve = 0.9

    return ScanFallbackPolicy(
        enabled=enabled,
        iters=iters,
        badjac_limit=badjac_limit,
        fsq_abs=fsq_abs,
        accept_frac=accept_frac,
        fsq_factor=fsq_factor,
        improve=improve,
    )


def _residual_convergence_flags(
    *,
    fsqr: float,
    fsqz: float,
    fsql: float,
    ftol: float,
    fsq_total_target: float | None,
) -> tuple[bool, bool, bool]:
    """Return strict, total-FSQ, and combined host convergence flags."""
    fsqr_f = float(fsqr)
    fsqz_f = float(fsqz)
    fsql_f = float(fsql)
    ftol_f = float(ftol)
    strict = (fsqr_f <= ftol_f) and (fsqz_f <= ftol_f) and (fsql_f <= ftol_f)
    total = bool((fsq_total_target is not None) and ((fsqr_f + fsqz_f + fsql_f) <= float(fsq_total_target)))
    return bool(strict), bool(total), bool(strict or total)


def _scalar_history_array(vals: Any) -> np.ndarray:
    """Materialize deferred scalar histories in one batch."""
    if not vals:
        return np.zeros((0,), dtype=float)
    if has_jax() and jax is not None:
        try:
            vals = jax.device_get(tuple(vals))
        except Exception:
            pass
    return np.asarray(vals, dtype=float)
