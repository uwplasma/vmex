"""Ahead-of-time lane executables that skip tracing and lowering on warm runs.

A warm persistent compilation cache still pays JAX tracing plus MLIR lowering
for every lane before it can even compute the cache key (about half of a warm
CLI call).  :func:`aot_lane` is ``jax.jit`` plus a second cache level: the
first call compiles through the ordinary jitted lane, serializes the
executable next to the persistent cache (``<cache>/aot``), and later
processes deserialize it directly, with no trace, lower or key hash.

Entries are keyed on the lane name, a hash of the vmex sources, the JAX
version, the backend and the call structure (pytree definition, leaf shapes,
dtypes, weak types, commitment, static arguments).  The lowering is the one
the jitted lane would build, so results are bit-identical.  Any miss, load
error or argument mismatch falls back to the jitted lane; tracers (calls
inside another transformation) and a disabled cache bypass it.  The
directory is kept to ``MAX_FILES`` executables, oldest first.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable

import jax
import numpy as np
from jax.experimental import serialize_executable as _se

_LANES: list = []               #: every AotLane, for :func:`release`
MAX_FILES = 64                  #: executables kept per cache directory
_SOURCE_HASH: str | None = None
_BUILD_LOCK = threading.Lock()


def _compile(lower: Callable[[], Any]) -> Any:
    """Compile ``lower()`` with the persistent compilation cache bypassed.

    An executable served by that cache serializes into a blob which fails at
    execution (``call_kernel not found``) once loaded, after its donated
    inputs are already consumed.  A fresh compile serializes into a loadable
    one; it happens once per source hash, after which the blob is reused.
    """
    from jax._src import compilation_cache as cc
    with _BUILD_LOCK:
        jax.config.update("jax_enable_compilation_cache", False)
        cc.reset_cache()
        try:
            return lower().compile()
        finally:
            jax.config.update("jax_enable_compilation_cache", True)
            cc.reset_cache()


def _source_hash() -> str:
    """Digest of the vmex sources (any edit invalidates every entry)."""
    global _SOURCE_HASH
    if _SOURCE_HASH is None:
        h = hashlib.sha256()
        root = Path(__file__).resolve().parents[1]
        for path in sorted(root.rglob("*.py")):
            h.update(path.relative_to(root).as_posix().encode())
            h.update(path.read_bytes())
        _SOURCE_HASH = h.hexdigest()[:16]
    return _SOURCE_HASH


def _directory() -> Path | None:
    """``<persistent cache>/aot``, or None when the persistent cache is off."""
    try:
        if not jax.config.jax_enable_compilation_cache:
            return None
        base = jax.config.jax_compilation_cache_dir
    except AttributeError:
        return None
    return Path(base) / "aot" if base else None


def _initialize_lapack() -> None:
    """Bind the CPU LAPACK kernels, which JAX otherwise does only while lowering.

    A loaded executable skips lowering, so its first LU or solve would call
    an unbound kernel and crash the process.
    """
    from jaxlib import lapack
    lapack._lapack.initialize()


def _device(leaves: list) -> Any:
    """The one device ``jax.jit`` runs these leaves on, or None (several).

    Committed arrays fix it; otherwise the default device (a
    ``jax.default_device`` context or ``jax.config``), else the default
    backend's first device.  A blob loads onto exactly this device: left to
    itself the loader targets every device of the backend, and on a host with
    several (GPUs, or XLA:CPU split into one device per core) the executable
    then rejects every call.
    """
    found = {d for leaf in leaves if getattr(leaf, "committed", False)
             for d in leaf.devices()}
    if len(found) > 1:
        return None
    if found:
        return found.pop()
    default = jax.config.jax_default_device
    if default is None or isinstance(default, str):
        return jax.devices(default)[0]
    return default


def _structure(args: tuple, static: dict) -> tuple[str, Any] | None:
    """Exact call-structure string and target device, or None when not safely hashable."""
    leaves, treedef = jax.tree_util.tree_flatten(args)
    parts = [str(treedef), repr(sorted(static.items()))]
    for leaf in leaves:
        if isinstance(leaf, jax.core.Tracer):
            return None
        aval = jax.api_util.shaped_abstractify(leaf)
        parts.append(f"{aval.dtype}{aval.shape}{int(aval.weak_type)}"
                     f"{int(getattr(leaf, 'committed', True))}")
    device = _device(leaves)
    if device is None:
        return None
    parts.append(f"{device.platform}:{device.id}")
    text = "|".join(parts)
    # Elided arrays or object ids in a repr would make distinct calls collide.
    return None if ("..." in text or " at 0x" in text) else (text, device)


class _Flat:
    """Run a leaf-argument executable on the lane's structured arguments."""

    def __init__(self, compiled: Any) -> None:
        self.compiled = compiled

    def __call__(self, *args: Any) -> Any:
        return self.compiled(*jax.tree_util.tree_leaves(args))


class AotLane:
    """``jax.jit(fn, **jit_kwargs)`` with a serialized-executable cache."""

    def __init__(self, fn: Callable, *, identity: str = "", **jit_kwargs: Any) -> None:
        """``identity`` names whatever ``fn`` closes over (module-level lanes: none)."""
        self.jit = jax.jit(fn, **jit_kwargs)
        self.fn = fn
        self.donate = tuple(jit_kwargs.get("donate_argnums") or ())
        self.name = fn.__name__
        self.identity = identity
        self.__doc__ = fn.__doc__
        self._static_names = tuple(jit_kwargs.get("static_argnames") or ())
        self._memory: dict[str, Any] = {}
        _LANES.append(self)

    def __getattr__(self, name: str) -> Any:     # lower, eval_shape, ...
        return getattr(self.jit, name)

    def _cache_size(self) -> int:
        """Executables held in memory (jit cache plus AOT-loaded)."""
        return self.jit._cache_size() + len(self._memory)

    def clear_cache(self) -> None:
        """Drop the jit cache and the in-memory AOT executables."""
        self.jit.clear_cache()
        self._memory.clear()

    def executable(self, *args: Any, **static: Any) -> Any:
        """Compiled lane for this call structure (memory, disk, then build).

        Returns None when the call cannot be cached; the caller then uses
        the jitted lane.
        """
        directory = _directory()
        found = None if directory is None else _structure(args, static)
        if found is None:
            return None
        text, device = found
        hit = self._memory.get(text)
        if hit is not None:
            return hit
        digest = hashlib.sha256("\n".join(
            (self.name, self.identity, _source_hash(), jax.__version__,
             jax.default_backend(), os.environ.get("XLA_FLAGS", ""),
             text)).encode()).hexdigest()[:32]
        path = directory / f"{self.name}-{digest}.aot"
        compiled = None
        try:
            _initialize_lapack()
            payload, in_tree, out_tree = pickle.loads(path.read_bytes())
            compiled = _se.deserialize_and_load(
                payload, in_tree, out_tree, backend=device.client,
                execution_devices=[device])
            os.utime(path)
        except Exception:                        # absent, stale or corrupt
            compiled = None
        if compiled is None:
            compiled = _compile(lambda: self._flat(args, static).lower(
                *jax.tree_util.tree_leaves(args)))
            self._store(directory, path, compiled)
        compiled = _Flat(compiled)
        self._memory[text] = compiled
        return compiled

    def _flat(self, args: tuple, static: dict) -> Any:
        """The lane jitted on the leaves of ``args``.

        Lowering a structured call unflattens the argument pytrees around
        placeholder ``ArgInfo`` leaves, which a pytree computing in its
        unflatten (an ESSOS field's coils) cannot survive; the leaf lowering
        is the same program.
        """
        treedef = jax.tree_util.tree_structure(args)
        sizes = [len(jax.tree_util.tree_leaves(a)) for a in args]
        starts = np.cumsum([0, *sizes])
        donate = tuple(j for i in self.donate for j in range(int(starts[i]), int(starts[i + 1])))

        def flat(*leaves):
            return self.fn(*jax.tree_util.tree_unflatten(treedef, leaves), **static)

        flat.__name__ = self.name
        return jax.jit(flat, donate_argnums=donate)

    @staticmethod
    def _store(directory: Path, path: Path, compiled: Any) -> None:
        try:
            blob = pickle.dumps(_se.serialize(compiled))
            directory.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
            with os.fdopen(fd, "wb") as f:
                f.write(blob)
            os.replace(tmp, path)
            old = sorted(directory.glob("*.aot"), key=lambda p: p.stat().st_mtime)
            for stale in old[:-MAX_FILES]:
                stale.unlink(missing_ok=True)
        except Exception:                        # read-only or full disk
            pass

    def __call__(self, *args: Any, **static: Any) -> Any:
        """Run the cached executable, falling back to the jitted lane.

        Argument checks precede donation, so a rejected call leaves the inputs
        intact and the jitted lane may take over.  Once a donated buffer is
        gone the jitted lane could only fail on it and mask the real error,
        which is raised instead.
        """
        try:
            compiled = self.executable(*args, **static)
        except Exception:
            compiled = None
        if compiled is not None:
            try:
                return compiled(*args)
            except Exception:
                if any(getattr(leaf, "is_deleted", bool)()
                       for leaf in jax.tree_util.tree_leaves(args)):
                    raise
        return self.jit(*args, **static)


def aot_lane(fn: Callable | None = None, **jit_kwargs: Any):
    """Decorator: ``@aot_lane(donate_argnums=(0,), static_argnames=...)``."""
    if fn is None:
        return lambda f: AotLane(f, **jit_kwargs)
    return AotLane(fn, **jit_kwargs)


def release() -> None:
    """Drop in-memory executables (the disk entries stay); frees memory between rungs."""
    for lane in _LANES:
        lane._memory.clear()


def compile_lane(lane: Any, *args: Any, **static: Any) -> Any:
    """Executable for ``lane(*args, **static)``: AOT-cached when possible."""
    compiled = lane.executable(*args, **static) if isinstance(lane, AotLane) else None
    return compiled if compiled is not None else lane.lower(*args, **static).compile()
