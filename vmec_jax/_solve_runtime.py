"""Pure runtime helpers for :mod:`vmec_jax.solve`.

This module keeps environment and tracing policy helpers out of the main solver
implementation while avoiding imports of solver physics code.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any

import numpy as np

from ._compat import has_jax, jax


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
) -> tuple[int, bool]:
    chunk_size_env = os.getenv("VMEC_JAX_SCAN_CHUNK_SIZE", "").strip()
    backend = _scan_backend_name() if backend_name is None else str(backend_name).strip().lower()
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
    elif (backend != "cpu") and (not bool(need_print)):
        # Quiet accelerator runs: use the full iteration budget as a single
        # chunk to eliminate the Python outer-loop host/device sync overhead.
        # Like the CPU quiet path, this compiles one program for the entire
        # solve, then breaks early via the carry.converged flag with no further
        # host syncs. The env override VMEC_JAX_SCAN_CHUNK_SIZE can cap this
        # when GPU memory is tight.
        chunk_size = max(1, int(max_iter_scan))
    else:
        chunk_size = max(1, int(nstep_screen))
    cap_to_remaining = not bool(need_print)
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
