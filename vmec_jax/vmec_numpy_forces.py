"""Pure-NumPy force computation hot path for CPU solves.

This module provides a context manager and helper that temporarily replaces
JAX array operations with NumPy equivalents across the full force computation
call chain.  The goal is to eliminate all JAX dispatch overhead from the
per-iteration CPU hot loop when ``host_update_assembly=True``.

Design
------
All force-path modules bind ``jnp`` at import time (``from ._compat import
jnp``).  We temporarily replace that module-level name with ``_NpModule``,
a thin wrapper around numpy that:
  - Returns ``_NpArray`` objects (numpy arrays with ``.at[].set()`` support).
  - Provides ``jax.lax.cond`` semantics via a Python if/else shim.

Thread-safety: This context **must not** be re-entered concurrently from
multiple threads (it mutates module-level globals).  The VMEC iteration loop
is single-threaded, so this is safe in practice.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any
import weakref

import numpy as np


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Helper: detect fancy (integer-array) indexing
# ---------------------------------------------------------------------------

def _idx_has_fancy(idx) -> bool:
    """Return True when *idx* contains integer-array (fancy) indexing.

    For simple slices/integers, NumPy's buffered ``+=`` is safe and faster.
    For fancy integer-array indices, ``np.add.at`` must be used to guarantee
    that duplicate indices are all accumulated (matching JAX behaviour).
    """
    if isinstance(idx, np.ndarray):
        return idx.ndim > 0  # 0-d array is effectively a scalar index
    if isinstance(idx, tuple):
        return any(_idx_has_fancy(i) for i in idx)
    # slice, int, or None — not fancy
    return False


# ---------------------------------------------------------------------------
# _AtIndexer: supports arr.at[idx].set(val) and arr.at[idx].add(val)
# ---------------------------------------------------------------------------

class _AtIndexer:
    """Provides JAX-style ``.at[idx].set(val)`` semantics for NumPy arrays.

    Correctness note: ``.set()`` copies the underlying array before mutating,
    matching JAX's functional/immutable semantics.  This is required for
    patterns like::

        rss_m1 = rss[:, 1, :]          # NumPy view of rss
        rss = rss.at[:, 1, :].set(...)  # must not corrupt rss_m1

    where a view captured before the ``.set()`` is used in a subsequent
    expression (vmec_parity.py lines 642–645).

    ``.add()`` can be done in-place because the JAX pattern is always
    ``arr = arr.at[idx].add(val)`` with no aliased views involved in the
    VMEC force path.
    """

    __slots__ = ("_arr", "_idx")

    def __init__(self, arr: np.ndarray, idx):
        self._arr = arr
        self._idx = idx

    def set(self, val) -> np.ndarray:
        out = self._arr.copy()
        out[self._idx] = np.asarray(val)
        return _wrap(out)

    def add(self, val) -> np.ndarray:
        # For simple (non-fancy) indices we use += which is faster.  For fancy
        # integer-array indices we must use np.add.at to correctly handle
        # duplicate indices (NumPy's buffered += only applies the last update
        # for duplicates, unlike JAX's .at[].add() which applies all of them).
        idx = self._idx
        use_add_at = _idx_has_fancy(idx)
        if use_add_at:
            np.add.at(self._arr, idx, np.asarray(val))
        else:
            self._arr[idx] += np.asarray(val)
        return self._arr


class _AtAccessor:
    """Attribute ``.at`` on a numpy array that returns an ``_AtIndexer``."""

    __slots__ = ("_arr",)

    def __init__(self, arr: np.ndarray):
        self._arr = arr

    def __getitem__(self, idx) -> _AtIndexer:
        return _AtIndexer(self._arr, idx)


# ---------------------------------------------------------------------------
# _NpArray: numpy ndarray subclass with .at support
# ---------------------------------------------------------------------------

class _NpArray(np.ndarray):
    """NumPy ndarray subclass that provides a JAX-compatible ``.at`` accessor."""

    @property
    def at(self) -> _AtAccessor:
        return _AtAccessor(self)

    def __array_finalize__(self, obj):
        pass


def _wrap(arr) -> _NpArray:
    """Wrap a plain numpy array (or scalar) in _NpArray."""
    if isinstance(arr, _NpArray):
        return arr
    a = np.asarray(arr)
    return a.view(_NpArray)


# ---------------------------------------------------------------------------
# _NpModule: drop-in replacement for jax.numpy used in the force hot path
# ---------------------------------------------------------------------------

class _NpModule:
    """Minimal jnp-compatible module backed by NumPy + _NpArray."""

    # --- dtype singletons (same as jax.numpy exposes) ---
    float32 = np.float32
    float64 = np.float64
    int32 = np.int32
    int64 = np.int64
    bool_ = np.bool_

    # Provide jnp.fft.rfft / jnp.fft.irfft as _NpFftModule instance.
    # (defined after _NpFftModule class below; patched at module load time)
    fft: "_NpFftModule" = None  # type: ignore[assignment]  # set after class def

    @staticmethod
    def asarray(x, dtype=None) -> _NpArray:
        if dtype is not None:
            # Fast path: already a _NpArray with the requested dtype — no copy.
            # This avoids repeated bool→float conversions for pre-converted mask
            # arrays (static.m_is_even, m_is_m1, etc.) on every iteration.
            if isinstance(x, _NpArray) and x.dtype == np.dtype(dtype):
                return x
            return _wrap(np.asarray(x, dtype=dtype))
        if isinstance(x, _NpArray):
            return x
        return _wrap(np.asarray(x))

    @staticmethod
    def array(x, dtype=None) -> _NpArray:
        if dtype is not None:
            return _wrap(np.array(x, dtype=dtype))
        return _wrap(np.array(x))

    @staticmethod
    def zeros(shape, dtype=None) -> _NpArray:
        return _wrap(np.zeros(shape, dtype=dtype or np.float64))

    @staticmethod
    def zeros_like(x, dtype=None) -> _NpArray:
        return _wrap(np.zeros_like(x, dtype=dtype))

    @staticmethod
    def ones(shape, dtype=None) -> _NpArray:
        return _wrap(np.ones(shape, dtype=dtype or np.float64))

    @staticmethod
    def ones_like(x, dtype=None) -> _NpArray:
        return _wrap(np.ones_like(x, dtype=dtype))

    @staticmethod
    def empty_like(x, dtype=None) -> _NpArray:
        return _wrap(np.empty_like(x, dtype=dtype))

    @staticmethod
    def full(shape, fill_value, dtype=None) -> _NpArray:
        return _wrap(np.full(shape, fill_value, dtype=dtype))

    @staticmethod
    def full_like(x, fill_value, dtype=None) -> _NpArray:
        return _wrap(np.full_like(x, fill_value, dtype=dtype))

    @staticmethod
    def arange(*args, dtype=None) -> _NpArray:
        return _wrap(np.arange(*args, dtype=dtype))

    @staticmethod
    def linspace(*args, **kwargs) -> _NpArray:
        return _wrap(np.linspace(*args, **kwargs))

    @staticmethod
    def reshape(x, shape) -> _NpArray:
        return _wrap(np.reshape(x, shape))

    @staticmethod
    def stack(arrays, axis=0) -> _NpArray:
        # Cache by identity of each input array + axis.  For the common case
        # where input arrays are long-lived cached phase tables, this avoids
        # re-running np.stack on every iteration (~55 000 calls saved per run).
        # Do not cache large force-path stacks: those inputs are short-lived,
        # and retaining their outputs pins O(GB) host memory across exact
        # Jacobian callbacks in long optimization processes.
        #
        # We validate cache entries with weakrefs so that Python id()-reuse
        # after garbage collection cannot produce stale (wrong) results.
        arr_list = list(arrays)
        arr_np = [np.asarray(a) for a in arr_list]
        total_bytes = sum(int(a.size) * int(a.dtype.itemsize) for a in arr_np)
        if total_bytes > _NP_STACK_CACHE_MAX_BYTES:
            return _wrap(np.stack(arr_np, axis=axis))
        key = tuple(id(a) for a in arr_list) + (axis,)
        entry = _NP_STACK_CACHE.get(key)
        if entry is not None:
            refs, result = entry
            # refs[i]() returns the original array if still alive, else None.
            # "is" comparison ensures it's the exact same Python object, not
            # a newly-allocated object that happens to share the same id().
            if all(r() is a for r, a in zip(refs, arr_list)):
                return result
            _NP_STACK_CACHE.pop(key, None)
        result = _wrap(np.stack(arr_np, axis=axis))
        try:
            refs = [weakref.ref(a) for a in arr_list]
            _NP_STACK_CACHE[key] = (refs, result)
            _prune_np_stack_cache()
        except TypeError:
            pass  # not weakly referenceable — skip caching
        return result

    @staticmethod
    def concatenate(arrays, axis=0) -> _NpArray:
        return _wrap(np.concatenate([np.asarray(a) for a in arrays], axis=axis))

    @staticmethod
    def where(cond, x=None, y=None) -> _NpArray:
        if x is None and y is None:
            return _wrap(np.where(np.asarray(cond)))
        return _wrap(np.where(np.asarray(cond), np.asarray(x), np.asarray(y)))

    @staticmethod
    def maximum(x, y) -> _NpArray:
        return _wrap(np.maximum(np.asarray(x), np.asarray(y)))

    @staticmethod
    def minimum(x, y) -> _NpArray:
        return _wrap(np.minimum(np.asarray(x), np.asarray(y)))

    @staticmethod
    def abs(x) -> _NpArray:
        return _wrap(np.abs(np.asarray(x)))

    @staticmethod
    def sqrt(x) -> _NpArray:
        return _wrap(np.sqrt(np.asarray(x)))

    @staticmethod
    def sum(x, axis=None, keepdims=False) -> _NpArray:
        return _wrap(np.sum(np.asarray(x), axis=axis, keepdims=keepdims))

    @staticmethod
    def all(x, axis=None) -> _NpArray:
        return _wrap(np.all(np.asarray(x), axis=axis))

    @staticmethod
    def any(x, axis=None) -> _NpArray:
        return _wrap(np.any(np.asarray(x), axis=axis))

    @staticmethod
    def isfinite(x) -> _NpArray:
        return _wrap(np.isfinite(np.asarray(x)))

    @staticmethod
    def isnan(x) -> _NpArray:
        return _wrap(np.isnan(np.asarray(x)))

    @staticmethod
    def isinf(x) -> _NpArray:
        return _wrap(np.isinf(np.asarray(x)))

    @staticmethod
    def mean(x, axis=None) -> _NpArray:
        return _wrap(np.mean(np.asarray(x), axis=axis))

    @staticmethod
    def max(x, axis=None) -> _NpArray:
        return _wrap(np.max(np.asarray(x), axis=axis))

    @staticmethod
    def min(x, axis=None) -> _NpArray:
        return _wrap(np.min(np.asarray(x), axis=axis))

    @staticmethod
    def einsum(expr, *operands, precision=None) -> _NpArray:
        return _wrap(np.einsum(expr, *[np.asarray(op) for op in operands]))

    @staticmethod
    def take(arr, indices, axis=None) -> _NpArray:
        return _wrap(np.take(np.asarray(arr), np.asarray(indices), axis=axis))

    @staticmethod
    def broadcast_to(arr, shape) -> _NpArray:
        return _wrap(np.broadcast_to(np.asarray(arr), shape))

    @staticmethod
    def moveaxis(arr, source, destination) -> _NpArray:
        return _wrap(np.moveaxis(np.asarray(arr), source, destination))

    @staticmethod
    def expand_dims(arr, axis) -> _NpArray:
        return _wrap(np.expand_dims(np.asarray(arr), axis=axis))

    @staticmethod
    def squeeze(arr, axis=None) -> _NpArray:
        if axis is None:
            return _wrap(np.squeeze(np.asarray(arr)))
        return _wrap(np.squeeze(np.asarray(arr), axis=axis))

    @staticmethod
    def transpose(arr, axes=None) -> _NpArray:
        return _wrap(np.transpose(np.asarray(arr), axes=axes))

    @staticmethod
    def clip(x, a_min=None, a_max=None) -> _NpArray:
        return _wrap(np.clip(np.asarray(x), a_min=a_min, a_max=a_max))

    @staticmethod
    def sin(x) -> _NpArray:
        return _wrap(np.sin(np.asarray(x)))

    @staticmethod
    def cos(x) -> _NpArray:
        return _wrap(np.cos(np.asarray(x)))

    @staticmethod
    def exp(x) -> _NpArray:
        return _wrap(np.exp(np.asarray(x)))

    @staticmethod
    def log(x) -> _NpArray:
        return _wrap(np.log(np.asarray(x)))

    @staticmethod
    def sign(x) -> _NpArray:
        return _wrap(np.sign(np.asarray(x)))

    @staticmethod
    def dot(a, b) -> _NpArray:
        return _wrap(np.dot(np.asarray(a), np.asarray(b)))

    @staticmethod
    def matmul(a, b) -> _NpArray:
        return _wrap(np.matmul(np.asarray(a), np.asarray(b)))

    @staticmethod
    def vstack(arrays) -> _NpArray:
        return _wrap(np.vstack([np.asarray(a) for a in arrays]))

    @staticmethod
    def hstack(arrays) -> _NpArray:
        return _wrap(np.hstack([np.asarray(a) for a in arrays]))

    @staticmethod
    def roll(arr, shift, axis=None) -> _NpArray:
        return _wrap(np.roll(np.asarray(arr), shift, axis=axis))

    @staticmethod
    def pad(arr, pad_width, **kwargs) -> _NpArray:
        return _wrap(np.pad(np.asarray(arr), pad_width, **kwargs))

    @staticmethod
    def sort(arr, axis=-1) -> _NpArray:
        return _wrap(np.sort(np.asarray(arr), axis=axis))

    @staticmethod
    def argsort(arr, axis=-1) -> _NpArray:
        return _wrap(np.argsort(np.asarray(arr), axis=axis))

    @staticmethod
    def floor(x) -> _NpArray:
        return _wrap(np.floor(np.asarray(x)))

    @staticmethod
    def ceil(x) -> _NpArray:
        return _wrap(np.ceil(np.asarray(x)))

    @staticmethod
    def round(x, decimals=0) -> _NpArray:
        return _wrap(np.round(np.asarray(x), decimals=decimals))

    @staticmethod
    def prod(x, axis=None) -> _NpArray:
        return _wrap(np.prod(np.asarray(x), axis=axis))

    @staticmethod
    def cumsum(x, axis=None) -> _NpArray:
        return _wrap(np.cumsum(np.asarray(x), axis=axis))

    @staticmethod
    def real(x) -> _NpArray:
        return _wrap(np.real(np.asarray(x)))

    @staticmethod
    def imag(x) -> _NpArray:
        return _wrap(np.imag(np.asarray(x)))


class _NpFftModule:
    """Minimal jnp.fft namespace backed by NumPy FFT.

    jnp.fft is accessed as a sub-namespace (e.g. jnp.fft.rfft).  When jnp is
    replaced by _NP_MODULE, attribute lookups like ``jnp.fft.rfft`` hit
    ``_NP_MODULE.fft.rfft``, which requires ``fft`` to be an object with the
    relevant methods rather than a bare function.
    """

    @staticmethod
    def fft(a, n=None, axis=-1, norm=None) -> _NpArray:
        return _wrap(np.fft.fft(np.asarray(a), n=n, axis=axis, norm=norm))

    @staticmethod
    def ifft(a, n=None, axis=-1, norm=None) -> _NpArray:
        return _wrap(np.fft.ifft(np.asarray(a), n=n, axis=axis, norm=norm))

    @staticmethod
    def rfft(a, n=None, axis=-1, norm=None) -> _NpArray:
        return _wrap(np.fft.rfft(np.asarray(a), n=n, axis=axis, norm=norm))

    @staticmethod
    def irfft(a, n=None, axis=-1, norm=None) -> _NpArray:
        return _wrap(np.fft.irfft(np.asarray(a), n=n, axis=axis, norm=norm))


# Singleton instance.
_NP_MODULE = _NpModule()
_NP_MODULE.fft = _NpFftModule()


# ---------------------------------------------------------------------------
# Shim for jax.lax.cond and other jax.* patterns in vmec_forces
# ---------------------------------------------------------------------------

class _NumpyLaxShim:
    """Minimal shim for ``jax.lax`` that handles patterns used in vmec_forces."""

    @staticmethod
    def cond(pred, true_fun, false_fun, operand=None):
        if bool(np.asarray(pred)):
            return true_fun(operand)
        return false_fun(operand)

    @staticmethod
    def dot_general(lhs, rhs, dimension_numbers, precision=None):
        (c_lhs, c_rhs), _ = dimension_numbers
        lhs_np = np.asarray(lhs)
        rhs_np = np.asarray(rhs)
        return _wrap(np.tensordot(lhs_np, rhs_np, axes=(list(c_lhs), list(c_rhs))))

    class Precision:
        HIGHEST = None


class _NumpyJaxShim:
    """Minimal shim for the ``jax`` module used in vmec_forces / vmec_tomnsp."""

    lax = _NumpyLaxShim()

    @staticmethod
    def named_scope(name: str):
        from contextlib import nullcontext
        return nullcontext()

    class profiler:
        @staticmethod
        def TraceAnnotation(name: str):
            from contextlib import nullcontext
            return nullcontext()

    @staticmethod
    def block_until_ready(x):
        return x

    @staticmethod
    def default_backend():
        return "cpu"


_JAX_SHIM = _NumpyJaxShim()


# ---------------------------------------------------------------------------
# Module-level patch tables (populated lazily)
# ---------------------------------------------------------------------------

_PATCHES: list[tuple[Any, list[tuple[str, Any]]]] | None = None

# Separate caches for NumPy-mode versions of module-level JAX array caches.
# These are populated lazily with _NpArray objects so that mask indexing
# (e.g. mask_even[None, :, None]) does not trigger JAX dispatch.
#
# Key insight: when _numpy_module_patch() swaps module-level 'jnp' with
# _NP_MODULE, newly computed cache entries use NumPy arrays.  By also swapping
# the cache dict itself we prevent the NumPy-mode code from finding (and using)
# stale JAX-array entries computed during warm-up before the patch was active.
_NP_MPARITY_CACHE: dict = {}
_NP_MASKS_CACHE: dict = {}
_NP_TOMNSPS_MASK_CACHE: dict = {}
# Force FFT path in NumPy mode by injecting a pre-populated cache that returns
# True.  The real _TOMNSPS_FFT_CACHE returns False on CPU (JAX backend = cpu),
# but numpy.fft.rfft is always available and FFT is strictly faster than DFT
# for ntheta >= 8.
_NP_TOMNSPS_FFT_CACHE: list = [True]

# vmec_parity: SignedModeMaps has '_j' JAX fields (mask_pos_j, idx_pos_safe_j …)
# that are indexed as n0[None,:,:] inside _mn_sin_to_signed_cached, triggering
# ~865 JAX __getitem__ dispatches per iteration for 3-D cases.  Patching the
# cache makes signed_maps_from_modes() build NumPy maps on first use in NumPy
# mode, eliminating all those dispatches.
_NP_SIGNED_MAP_CACHE: dict = {}

# vmec_realspace: phase table caches hold JAX arrays by default.  Patching them
# ensures that phase tables computed inside the NumPy-mode context are NumPy
# arrays and cached for subsequent iterations.
_NP_PHASE_CACHE: dict = {}
_NP_PHASE_DTHETA_CACHE: dict = {}
_NP_PHASE_DZETA_CACHE: dict = {}
_NP_PHASE_STACK_CACHE: dict = {}
_NP_PHASE_DTHETA_STACK_CACHE: dict = {}
_NP_PHASE_DZETA_STACK_CACHE: dict = {}

# vmec_residue caches: _WINT_CACHE, _PWINT_CACHE, _SCALXC_CACHE store JAX
# arrays under the real module. Redirect to separate NumPy-mode dicts so that
# _NpArray entries computed during the NumPy hot-path never leak into the JAX
# cache (which would cause type errors when the JAX scan path later hits the
# cache and tries to use the _NpArray inside a jit/lax.scan trace).
_NP_WINT_CACHE: dict = {}
_NP_PWINT_CACHE: dict = {}
_NP_SCALXC_CACHE: dict = {}

# fourier._HELICAL_BASIS_CACHE stores HelicalBasis objects containing JAX arrays.
# Redirect to a separate NumPy-mode cache to prevent _NpArray-containing
# HelicalBasis objects from polluting the JAX cache.
_NP_HELICAL_BASIS_CACHE: dict = {}

# Generic stack cache: keyed by (id(a0), id(a1), ..., axis).  The phase-table
# arrays returned by _vmec_phase_tables_stacked_cached are long-lived Python
# objects (held in _NP_PHASE_*_STACK_CACHE), so their id()s are stable for the
# lifetime of a VMEC solve. Caching here avoids 55 000+ redundant np.stack
# calls (0.44 s in QA_lowres) where the input arrays don't change between
# iterations.
_NP_STACK_CACHE: dict = {}
_NP_STACK_CACHE_LIMIT = 1024
_NP_STACK_CACHE_MAX_BYTES = 64 * 1024


def _prune_np_stack_cache() -> None:
    """Drop dead and excess generic NumPy stack-cache entries."""
    if not _NP_STACK_CACHE:
        return
    dead = []
    for key, (refs, _result) in _NP_STACK_CACHE.items():
        if any(ref() is None for ref in refs):
            dead.append(key)
    for key in dead:
        _NP_STACK_CACHE.pop(key, None)
    while len(_NP_STACK_CACHE) > _NP_STACK_CACHE_LIMIT:
        _NP_STACK_CACHE.pop(next(iter(_NP_STACK_CACHE)), None)


def clear_numpy_force_caches() -> None:
    """Release per-process NumPy-mode force caches that can hold arrays."""
    _NP_STACK_CACHE.clear()


def _np_einsum(expr: str, *operands, **kwargs) -> _NpArray:
    """NumPy einsum that always returns _NpArray, with BLAS fast-paths.

    Replaces ``np.einsum`` for the common VMEC contraction patterns so that
    numpy dispatches to BLAS DGEMM instead of its slow loop-based C kernel.
    Each fast-path is mathematically equivalent to the corresponding einsum.
    """
    # Normalise operands to plain ndarray once.
    ops = [np.asarray(op) for op in operands]

    # -----------------------------------------------------------------------
    # vmec_tomnsp hot-path patterns (the dominant cost for large cases)
    # -----------------------------------------------------------------------
    if expr == "apsik,im->apsmk" and len(ops) == 2:
        # arr (a,p,s,i,k), mat (i,m) → result (a,p,s,m,k)
        # numpy's einsum with optimize=True is ~2× faster than manual
        # reshape-transpose-GEMM because it avoids the non-contiguous strided copy.
        return _wrap(np.einsum("apsik,im->apsmk", ops[0], ops[1], optimize=True))

    if expr == "psmk,kn->psmn" and len(ops) == 2:
        # arr (p,s,m,k), mat (k,n) → result (p,s,m,n)
        arr, mat = ops
        p, s, m, k_ = arr.shape
        arr2 = arr.reshape(p * s * m, k_)  # k already last → no copy needed if C-contiguous
        out2 = arr2 @ mat  # (p*s*m, n)
        return _wrap(out2.reshape(p, s, m, mat.shape[1]))

    # -----------------------------------------------------------------------
    # vmec_constraints patterns (DFT basis projections / reconstructions)
    # -----------------------------------------------------------------------
    if expr == "sik,im->smk" and len(ops) == 2:
        # arr (s,i,k), mat (i,m) → result (s,m,k)
        arr, mat = ops
        s, i_, k = arr.shape
        # Bring i to last: (s,k,i) then contract with mat(i,m)
        arr2 = arr.transpose(0, 2, 1).reshape(s * k, i_)  # (s*k, i)
        out2 = arr2 @ mat  # (s*k, m)
        return _wrap(out2.reshape(s, k, mat.shape[1]).transpose(0, 2, 1))

    if expr == "smk,kn->smn" and len(ops) == 2:
        # arr (s,m,k), mat (k,n) → result (s,m,n)
        arr, mat = ops
        s, m, k_ = arr.shape
        arr2 = arr.reshape(s * m, k_)
        return _wrap((arr2 @ mat).reshape(s, m, mat.shape[1]))

    if expr == "smn,kn->smk" and len(ops) == 2:
        # arr (s,m,n), mat (k,n) → result (s,m,k)
        arr, mat = ops
        s, m, n = arr.shape
        arr2 = arr.reshape(s * m, n)
        return _wrap((arr2 @ mat.T).reshape(s, m, mat.shape[0]))

    if expr == "smk,im->sik" and len(ops) == 2:
        # arr (s,m,k), mat (i,m) → result (s,i,k)
        arr, mat = ops
        s, m, k = arr.shape
        # Bring m to last: (s,k,m) then contract with mat.T(m,i)
        arr2 = arr.transpose(0, 2, 1).reshape(s * k, m)  # (s*k, m)
        out2 = arr2 @ mat.T  # (s*k, i)
        return _wrap(out2.reshape(s, k, mat.shape[0]).transpose(0, 2, 1))

    # -----------------------------------------------------------------------
    # fourier / vmec_realspace patterns (synthesis and analysis transforms)
    # -----------------------------------------------------------------------
    if expr == "...k,kij->...ij" and len(ops) == 2:
        # coeff (...,k), phase (k,i,j) → result (...,i,j)
        coeff, phase = ops
        k = coeff.shape[-1]
        prefix = coeff.shape[:-1]
        i, j = phase.shape[1], phase.shape[2]
        out = coeff.reshape(-1, k) @ phase.reshape(k, i * j)
        return _wrap(out.reshape(*prefix, i, j))

    if expr == "...ij,kij->...k" and len(ops) == 2:
        # f (...,i,j), phase (k,i,j) → result (...,k)
        f, phase = ops
        ij = f.shape[-2] * f.shape[-1]
        prefix = f.shape[:-2]
        k = phase.shape[0]
        out = f.reshape(-1, ij) @ phase.reshape(k, ij).T
        return _wrap(out.reshape(*prefix, k))

    if expr == "sij,kij->sk" and len(ops) == 2:
        # f (s,i,j), phase (k,i,j) → result (s,k)
        f, phase = ops
        s = f.shape[0]
        ij = f.shape[1] * f.shape[2]
        k = phase.shape[0]
        out = f.reshape(s, ij) @ phase.reshape(k, ij).T
        return _wrap(out)

    if expr == "...k,tkij->t...ij" and len(ops) == 2:
        # coeff (...,k), phase_all (t,k,i,j) → result (t,...,i,j)
        # vmec_realspace_synthesis_multi: stacked synthesis for multiple derivs
        coeff, phase_all = ops
        k = coeff.shape[-1]
        prefix = coeff.shape[:-1]
        t, _, i, j = phase_all.shape
        # Batch matmul via numpy broadcast: (1,n,k) @ (t,k,ij) → (t,n,ij)
        n = int(np.prod(prefix)) if prefix else 1
        coeff2 = coeff.reshape(n, k)           # (n, k)
        phase2 = phase_all.reshape(t, k, i * j)  # (t, k, i*j) — view if C-contiguous
        # numpy batched matmul: (1,n,k) @ (t,k,i*j) = (t,n,i*j)
        out2 = coeff2[np.newaxis] @ phase2      # broadcast over t
        return _wrap(out2.reshape(t, *prefix, i, j))

    # Fallback: generic numpy einsum
    return _wrap(np.einsum(expr, *ops))


def _build_patches() -> list[tuple[Any, list[tuple[str, Any]]]]:
    """Build the patch list from already-imported modules."""
    import vmec_jax.vmec_forces as _vmec_forces
    import vmec_jax.vmec_bcovar as _vmec_bcovar
    import vmec_jax.vmec_jacobian as _vmec_jacobian
    import vmec_jax.vmec_tomnsp as _vmec_tomnsp
    import vmec_jax.vmec_parity as _vmec_parity
    import vmec_jax.vmec_realspace as _vmec_realspace
    import vmec_jax.vmec_residue as _vmec_residue
    import vmec_jax.vmec_constraints as _vmec_constraints
    import vmec_jax.fourier as _fourier
    import vmec_jax.field as _field
    import vmec_jax.solve as _solve
    import vmec_jax.solve_force_payload_helpers as _solve_force_payload_helpers

    patches = [
        (_solve, [
            ("jnp", _NP_MODULE),
            ("jax", _JAX_SHIM),
        ]),
        (_solve_force_payload_helpers, [
            ("jnp", _NP_MODULE),
        ]),
        (_vmec_forces, [
            ("jnp", _NP_MODULE),
            ("jax", _JAX_SHIM),
        ]),
        (_vmec_bcovar, [
            ("jnp", _NP_MODULE),
        ]),
        (_vmec_jacobian, [
            ("jnp", _NP_MODULE),
        ]),
        (_vmec_tomnsp, [
            ("jnp", _NP_MODULE),
            ("_JNP_EINSUM", _np_einsum),
            # Replace the JAX-array parity/mask caches with NumPy-array caches.
            # This prevents mask_even[None, :, None] inside _select_mparity from
            # triggering JAX __getitem__ dispatch (saves ~6500 JAX dispatches/iter).
            ("_MPARITY_CACHE", _NP_MPARITY_CACHE),
            ("_TOMNSPS_MASK_CACHE", _NP_TOMNSPS_MASK_CACHE),
            # Override the FFT-enable cache to True so _get_tomnsps_fft() returns
            # True in NumPy mode.  numpy.fft.rfft is always available and faster
            # than the DFT-GEMM fallback for ntheta >= ~8.
            ("_TOMNSPS_FFT_CACHE", _NP_TOMNSPS_FFT_CACHE),
        ]),
        (_vmec_parity, [
            ("jnp", _NP_MODULE),
            # Replace the SignedModeMaps cache so that _build_signed_maps()
            # produces _NpArray _j fields (e.g. m0_mask_j, idx_pos_safe_j …)
            # instead of JAX arrays.  Without this patch, each _j field access
            # (n0[None,:,:] etc.) triggers JAX __getitem__ dispatch — ~865
            # dispatches/iteration for 3-D stellarator cases.
            ("_MN_SIGNED_MAP_CACHE", _NP_SIGNED_MAP_CACHE),
        ]),
        (_vmec_realspace, [
            ("jnp", _NP_MODULE),
            ("einsum", _np_einsum),
            # Replace phase table caches so that NumPy-computed phases are
            # stored separately from the JAX-computed entries.
            ("_PHASE_CACHE", _NP_PHASE_CACHE),
            ("_PHASE_DTHETA_CACHE", _NP_PHASE_DTHETA_CACHE),
            ("_PHASE_DZETA_CACHE", _NP_PHASE_DZETA_CACHE),
            ("_PHASE_STACK_CACHE", _NP_PHASE_STACK_CACHE),
            ("_PHASE_DTHETA_STACK_CACHE", _NP_PHASE_DTHETA_STACK_CACHE),
            ("_PHASE_DZETA_STACK_CACHE", _NP_PHASE_DZETA_STACK_CACHE),
        ]),
        (_vmec_residue, [
            ("jnp", _NP_MODULE),
            # Redirect residue caches so _NpArray values don't leak into JAX caches.
            ("_WINT_CACHE", _NP_WINT_CACHE),
            ("_PWINT_CACHE", _NP_PWINT_CACHE),
            ("_SCALXC_CACHE", _NP_SCALXC_CACHE),
        ]),
        (_vmec_constraints, [
            ("jnp", _NP_MODULE),
            ("einsum", _np_einsum),
        ]),
        (_fourier, [
            ("jnp", _NP_MODULE),
            ("einsum", _np_einsum),
            # Redirect the HelicalBasis cache so _NpArray-backed basis objects
            # don't leak into the JAX cache.
            ("_HELICAL_BASIS_CACHE", _NP_HELICAL_BASIS_CACHE),
        ]),
        (_field, [
            ("jnp", _NP_MODULE),
        ]),
    ]
    return patches


@contextmanager
def _numpy_module_patch():
    """Context manager: temporarily patch module-level ``jnp`` with _NP_MODULE."""
    global _PATCHES
    if _PATCHES is None:
        _PATCHES = _build_patches()

    # Save originals and apply patches.
    saved: list[tuple[Any, list[tuple[str, Any]]]] = []
    for mod, attrs in _PATCHES:
        orig = [(name, getattr(mod, name, None)) for name, _ in attrs]
        saved.append((mod, orig))
        for name, new_val in attrs:
            setattr(mod, name, new_val)

    # Activate _compat numpy_mode so has_jax() returns False,
    # routing control-flow branches (FFT vs DFT, cache checks, etc.) to NumPy.
    from vmec_jax._compat import _numpy_mode_local
    prev_active = getattr(_numpy_mode_local, "active", False)
    _numpy_mode_local.active = True

    try:
        yield
    finally:
        for mod, orig in saved:
            for name, old_val in orig:
                if old_val is None:
                    try:
                        delattr(mod, name)
                    except AttributeError:
                        pass
                else:
                    setattr(mod, name, old_val)
        _numpy_mode_local.active = prev_active


def _to_numpy_recursive(obj):
    """Recursively convert all array fields in a frozen dataclass to _NpArray.

    Handles ``VmecTrigTables``, ``_WoutLikeVmecForces``, and any other frozen
    dataclass whose fields may contain JAX arrays.  Non-array fields (ints,
    floats, bools, ``None``, nested dataclasses) are preserved or recursed into
    as appropriate.

    Parameters
    ----------
    obj:
        A frozen dataclass (or any Python object).  If not a dataclass, returned
        unchanged.

    Returns
    -------
    A new instance of the same type with all array-like fields converted to
    ``_NpArray`` (NumPy arrays with ``.at`` support).
    """
    import dataclasses

    if not dataclasses.is_dataclass(obj) or isinstance(obj, type):
        # Plain arrays or scalars at the top level.
        try:
            return _wrap(np.asarray(obj))
        except Exception:
            return obj

    field_values = {}
    for f in dataclasses.fields(obj):
        val = getattr(obj, f.name)
        if val is None:
            field_values[f.name] = None
        elif isinstance(val, (int, float, bool, str)):
            field_values[f.name] = val
        elif dataclasses.is_dataclass(val) and not isinstance(val, type):
            field_values[f.name] = _to_numpy_recursive(val)
        else:
            # Try converting to NumPy array; fall back to original if not possible.
            try:
                arr = np.asarray(val)
                field_values[f.name] = _wrap(arr)
            except Exception:
                field_values[f.name] = val

    try:
        return dataclasses.replace(obj, **field_values)
    except Exception:
        # Last resort: construct a new instance directly.
        try:
            return type(obj)(**field_values)
        except Exception:
            return obj


def compute_forces_numpy(
    compute_forces_impl,
    state,
    *,
    include_edge: bool,
    include_edge_residual: bool | None = None,
    zero_m1: Any,
    freeb_bsqvac_half: Any | None = None,
    constraint_rcon0: Any | None = None,
    constraint_zcon0: Any | None = None,
    constraint_precond_diag: tuple[Any, Any] | None = None,
    constraint_tcon: Any | None = None,
    constraint_precond_active: Any | None = None,
    constraint_tcon_active: Any | None = None,
    iter_idx: int | None = None,
):
    """Call ``compute_forces_impl`` with all JAX ops replaced by NumPy.

    Parameters
    ----------
    compute_forces_impl:
        The ``_compute_forces_impl`` closure from ``solve.py`` (non-JIT version).
    state:
        Current ``VMECState`` (may hold JAX or NumPy arrays).
    All other kwargs are forwarded verbatim to ``compute_forces_impl``.

    Returns
    -------
    Same 8-tuple as ``_compute_forces``:
    ``(k, frzl, gcr2, gcz2, gcl2, rz_scale, l_scale, norms_current)``
    """
    # Convert state arrays to NumPy once before entering the patch context.
    try:
        from dataclasses import replace as _dc_replace
        state_np = _dc_replace(
            state,
            Rcos=_wrap(np.asarray(state.Rcos)),
            Rsin=_wrap(np.asarray(state.Rsin)),
            Zcos=_wrap(np.asarray(state.Zcos)),
            Zsin=_wrap(np.asarray(state.Zsin)),
            Lcos=_wrap(np.asarray(state.Lcos)),
            Lsin=_wrap(np.asarray(state.Lsin)),
        )
    except Exception:
        state_np = state

    # Convert constraint arrays to NumPy.
    if constraint_precond_diag is not None:
        try:
            ard1, azd1 = constraint_precond_diag
            constraint_precond_diag = (_wrap(np.asarray(ard1)), _wrap(np.asarray(azd1)))
        except Exception:
            pass
    if constraint_tcon is not None:
        try:
            constraint_tcon = _wrap(np.asarray(constraint_tcon))
        except Exception:
            pass
    if constraint_rcon0 is not None:
        try:
            constraint_rcon0 = _wrap(np.asarray(constraint_rcon0))
        except Exception:
            pass
    if constraint_zcon0 is not None:
        try:
            constraint_zcon0 = _wrap(np.asarray(constraint_zcon0))
        except Exception:
            pass
    if constraint_precond_active is not None:
        try:
            constraint_precond_active = bool(np.asarray(constraint_precond_active))
        except Exception:
            pass
    if constraint_tcon_active is not None:
        try:
            constraint_tcon_active = bool(np.asarray(constraint_tcon_active))
        except Exception:
            pass
    zero_m1_val = float(np.asarray(zero_m1))

    with _numpy_module_patch():
        result = compute_forces_impl(
            state_np,
            include_edge=include_edge,
            include_edge_residual=include_edge_residual,
            zero_m1=zero_m1_val,
            freeb_bsqvac_half=freeb_bsqvac_half,
            constraint_rcon0=constraint_rcon0,
            constraint_zcon0=constraint_zcon0,
            constraint_precond_diag=constraint_precond_diag,
            constraint_tcon=constraint_tcon,
            constraint_precond_active=constraint_precond_active,
            constraint_tcon_active=constraint_tcon_active,
            iter_idx=iter_idx,
        )
    return result
