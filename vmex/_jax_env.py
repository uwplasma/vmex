"""JAX environment defaults, persistent compilation-cache policy, optional floors.

- :func:`_configure_jax_environment` runs at import, before
  ``vmex/__init__`` imports JAX, and sets what must precede XLA
  initialization: float64 (``JAX_ENABLE_X64``, VMEC parity), synchronous CPU
  dispatch, quiet XLA logging, GPU demand allocation, XLA:CPU flags, the
  machine-scoped persistent cache and the memory-map pressure guard;
- the cache policy (:func:`_default_compilation_cache_dir`,
  :func:`_apply_compilation_cache_policy`) is re-applied by
  ``core.solver._harden_compilation_cache`` on every solve path;
- :func:`require_optional` checks the floors of the optional extras.
"""

from __future__ import annotations

from typing import Any
import contextlib
import hashlib
from importlib import metadata as importlib_metadata
import os
import pathlib
import platform
import re
import sys
import time


_CACHE_FORMAT_VERSION = "4"
_CACHE_MAX_ENTRIES = 1024            # resident executables; see _prune_cache_entries
_CACHE_RECENT_SECONDS = 24 * 3600    # entries used this recently survive the cap, up to 4x it
_CACHE_SIZE_FLOOR = 2 << 30          # 2 GiB
_CACHE_SIZE_CEILING = 20 << 30       # 20 GiB
_CACHE_DISK_FRACTION = 0.10
_CACHE_ENTRY_SUFFIX = "-cache"       # jax._src.lru_cache naming
_CACHE_ATIME_SUFFIX = "-atime"


def _default_cache_max_size(path: str | None = None) -> int:
    """Bytes to retain in the persistent compilation cache.

    The bound has to be finite: JAX takes its cross-process cache lock only
    when eviction is enabled, so an unbounded cache lets concurrent VMEX runs
    race writing the same executable.  It also has to be large.  One
    free-boundary or single-stage executable is tens of megabytes and a single
    optimization walks a whole family of them, so the historical 1 GiB bound
    sat permanently at its cap and evicted the executables the next stage
    asked for -- every run paid a cold compile that the cache existed to
    avoid.  Scale with the filesystem that holds the cache and cap at a size
    any current workstation can spare.
    """
    try:
        import shutil

        free = shutil.disk_usage(path or os.path.expanduser("~")).free
    except Exception:  # unreadable path, exotic filesystem
        return _CACHE_SIZE_FLOOR
    scaled = int(_CACHE_DISK_FRACTION * float(free))
    return int(min(_CACHE_SIZE_CEILING, max(_CACHE_SIZE_FLOOR, scaled)))


def _prune_cache_entries(cache_dir: str, max_entries: int) -> int:
    """Drop the least recently used cache entries above ``max_entries``.

    JAX's LRU cache re-scans the whole directory on every write (glob, stat,
    atime sidecars, under the directory lock): 0.028 ms per entry on an
    Apple M4, so a mature 10,880-entry cache turned a 7.5 s QA solve into
    31.3 s.  Its own bound is on bytes and never fires for these small
    entries, so the entry count is bounded here, once per process.  Entries
    used within ``_CACHE_RECENT_SECONDS`` survive up to four times
    ``max_entries`` -- the QI example writes 1,342 executables and a plain
    1,024 cap made every rerun recompile 318 of them -- and older ones are
    trimmed to ``max_entries``.  Housekeeping only: any failure (missing
    directory, no ``filelock``, a lock another process holds, an entry that
    cannot be removed) skips the work.  Returns the number of entries removed.
    """
    path = pathlib.Path(cache_dir)
    try:
        entries = list(path.glob(f"*{_CACHE_ENTRY_SUFFIX}"))
        if len(entries) <= max_entries:
            return 0
        import filelock

        lock = filelock.FileLock(path / ".lockfile")
        lock.acquire(timeout=5)
    except Exception:
        return 0

    def sidecar(entry: pathlib.Path) -> pathlib.Path:
        return entry.with_name(entry.name[: -len(_CACHE_ENTRY_SUFFIX)] + _CACHE_ATIME_SUFFIX)

    def atime(entry: pathlib.Path) -> int:
        try:
            return int.from_bytes(sidecar(entry).read_bytes(), "little")
        except OSError:
            return 0

    removed = 0
    try:
        stamps = {entry: atime(entry) for entry in entries}
        ordered = sorted(entries, key=stamps.__getitem__, reverse=True)
        cutoff = time.time_ns() - _CACHE_RECENT_SECONDS * 1_000_000_000
        recent = sum(stamp >= cutoff for stamp in stamps.values())
        for entry in ordered[max(max_entries, min(recent, 4 * max_entries)):]:
            with contextlib.suppress(OSError):
                entry.unlink()
                removed += 1
                sidecar(entry).unlink(missing_ok=True)
    finally:
        with contextlib.suppress(Exception):
            lock.release()
    return removed


def _env(name: str, default: str = "") -> str:
    """Read ``VMEX_<name>``, falling back to the legacy ``VMEC_JAX_<name>``.

    The package was renamed vmec_jax -> vmex; environment variables a user may
    have set in their shell profile (the ``*_COMPILATION_CACHE*`` knobs in
    particular) keep working under their old names for one release.
    """
    val = os.environ.get(f"VMEX_{name}")
    if val is not None:
        return val
    return os.environ.get(f"VMEC_JAX_{name}", default)


_CACHE_DESERIALIZE_SAFE_JAXLIB = (0, 10)


def _jaxlib_version_tuple() -> tuple[int, ...] | None:
    """Leading numeric components of the jaxlib Python will run, or None.

    The runtime ``jaxlib.version`` constant (pure Python, no native load)
    wins over distribution metadata: a stale or shadowed install can report
    one version while the interpreter imports another.
    """
    try:
        import jaxlib.version
        raw = jaxlib.version.__version__
    except Exception:
        try:
            raw = importlib_metadata.version("jaxlib")
        except Exception:
            return None
    parts: list[int] = []
    for token in raw.split(".")[:3]:
        digits = ""
        for char in token:
            if not char.isdigit():
                break
            digits += char
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts) or None


def _cache_deserialize_unsafe() -> bool:
    """True when *reading* the persistent cache can kill this process.

    jaxlib < 0.10 dies inside ``PyClient::DeserializeExecutable`` when
    loading a cached XLA:CPU executable holding more than a few hundred
    kernels (SIGBUS/SIGILL on macOS, SIGSEGV on Linux): LLVM ORC
    materializes the per-kernel objects recursively on one fixed-size
    worker-thread stack (RTDyldObjectLinkingLayer::emit ->
    ExecutionSession::lookup -> dispatchOutstandingMUs -> emit -> ...),
    and every vmex solve/adjoint executable is large enough to overflow
    it deterministically on the first warm rerun.  Reproduced on jaxlib
    0.9.2 with a 300-kernel jit program on both platforms; the same
    program reloads cleanly on jaxlib 0.10.
    An unknown jaxlib version counts as unsafe: losing the cache costs a
    recompile, trusting it can cost the process.
    """
    version = _jaxlib_version_tuple()
    return version is None or version < _CACHE_DESERIALIZE_SAFE_JAXLIB


_EXT_DIGEST_CHUNK = 1 << 20


def _jaxlib_backend_identity() -> list[str]:
    """Fingerprint parts tied to the jaxlib build that will actually run.

    Distribution metadata can survive an in-place up/downgrade stale or
    broken — and that is exactly when reusing the old cache is most
    dangerous: XLA:CPU AOT entries record the target features the compiling
    backend chose (e.g. ``+prefer-no-gather``), and a different jaxlib's AOT
    loader rejects them or segfaults.  Record the version constant of the
    module Python will import, plus a bounded content digest (size, leading
    and trailing :data:`_EXT_DIGEST_CHUNK` bytes) of the native XLA
    extension, so any compiler/loader change lands in a fresh cache
    directory.  ``jaxlib.version`` is pure Python; nothing here loads the
    native extension.
    """
    parts: list[str] = []
    try:
        import jaxlib.version
        parts.append(f"jaxlib-runtime={jaxlib.version.__version__}")
    except Exception:
        return parts
    try:
        ext_dir = os.path.dirname(jaxlib.__file__ or "")
        digest = hashlib.sha256()
        for name in sorted(os.listdir(ext_dir)):
            if (name.split(".", 1)[0] in ("_jax", "xla_extension")
                    and name.endswith((".so", ".pyd", ".dylib"))):
                path = os.path.join(ext_dir, name)
                size = os.path.getsize(path)
                digest.update(f"{name}:{size}".encode())
                with open(path, "rb") as fh:
                    digest.update(fh.read(_EXT_DIGEST_CHUNK))
                    fh.seek(max(size - _EXT_DIGEST_CHUNK, _EXT_DIGEST_CHUNK))
                    digest.update(fh.read(_EXT_DIGEST_CHUNK))
        parts.append(f"jaxlib-ext={digest.hexdigest()[:16]}")
    except Exception:
        pass
    return parts


def _host_identity() -> str:
    """Stable name of this physical machine (one cache per machine).

    Linux and Windows: the hostname, which on a cluster names the node.
    macOS: the hardware UUID (``gethostuuid``), because a laptop's hostname
    follows the network it joins; the hostname is the fallback.
    """
    if platform.system() == "Darwin":
        import ctypes
        import uuid

        buf = ctypes.create_string_buffer(16)
        with contextlib.suppress(Exception):
            if ctypes.CDLL(None).gethostuuid(buf, (ctypes.c_long * 2)()) == 0:
                return str(uuid.UUID(bytes=buf.raw))
    return platform.node()


def _cache_machine_fingerprint() -> str:
    """Return the per-machine subdirectory name of the compilation cache.

    Persistent-cache entries are native XLA:CPU executables.  JAX keys them
    on the host CPU it detects, but a cache on a shared home filesystem is
    read by every node of a heterogeneous cluster, and a loader that meets
    an entry built for other CPU features only logs "Target machine feature
    ... is not supported on the host machine" and runs it anyway.  So one
    directory per machine: the host identity (:func:`_host_identity`) never
    lets two machines share an entry, and the OS, architecture, CPU model
    and flags, Python and jaxlib build identity
    (:func:`_jaxlib_backend_identity`) move the cache whenever the machine
    or the compiler under it changes.  The price is one cold compile per
    cluster node (and per container, whose hostname is new each run); a
    workstation or laptop keeps one warm cache.
    """

    parts = [
        f"vmex-cache={_CACHE_FORMAT_VERSION}",
        platform.system(),
        platform.machine(),
        platform.processor(),
        f"host={_host_identity()}",
        f"python={sys.version_info.major}.{sys.version_info.minor}",
    ]
    for package in ("jax", "jaxlib"):
        try:
            parts.append(f"{package}={importlib_metadata.version(package)}")
        except Exception:
            pass
    parts.extend(_jaxlib_backend_identity())
    try:
        if os.path.exists("/proc/cpuinfo"):
            wanted = ("model name", "cpu family", "model", "stepping", "flags", "Features")
            seen: set[str] = set()
            with open("/proc/cpuinfo", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    if ":" not in line:
                        continue
                    key, value = (part.strip() for part in line.split(":", 1))
                    if key in wanted and key not in seen:
                        parts.append(f"{key}={value}")
                        seen.add(key)
    except Exception:
        pass
    if platform.system() == "Darwin":
        try:
            import subprocess
            for key in ("machdep.cpu.brand_string", "hw.optional.arm.FEAT_SME",
                        "hw.cpufamily"):
                out = subprocess.run(["sysctl", "-n", key], capture_output=True,
                                     text=True, timeout=2)
                if out.returncode == 0 and out.stdout.strip():
                    parts.append(f"{key}={out.stdout.strip()}")
        except Exception:
            pass
    digest = hashlib.sha256("|".join(parts).encode("utf-8", errors="ignore")).hexdigest()[:16]
    system = platform.system().lower() or "unknown"
    machine = platform.machine().lower() or "unknown"
    return f"{system}-{machine}-{digest}"


_FINGERPRINT_NAME = re.compile(r"[a-z0-9_]+-[a-z0-9_.]+-[0-9a-f]{16}")


def _machine_scoped(directory: str) -> str:
    """Return ``directory/<this machine's fingerprint>``.

    Idempotent, and never nests one machine inside another: a path whose
    last component is already a fingerprint -- this machine's, or another's
    inherited through the environment of a job launched from a different
    node -- is re-rooted at its parent.
    """
    path = pathlib.Path(directory).expanduser()
    if _FINGERPRINT_NAME.fullmatch(path.name):
        path = path.parent
    return str(path / _cache_machine_fingerprint())


_OFF = ("disabled", "0", "false", "no", "off")
_ON = ("1", "true", "yes", "on", "enabled")


def _default_compilation_cache_dir() -> str | None:
    """Return this machine's persistent-cache directory, or None for no cache.

    Zero configuration gives ``~/.cache/vmex/jax_cache/<fingerprint>``
    (:func:`_cache_machine_fingerprint`); a repeated run then reuses compiled
    kernels instead of recompiling (a solovev CLI rerun drops 4.3 s -> 1.2 s).
    In order of precedence:

    - ``VMEX_COMPILATION_CACHE=disabled`` (or ``0/false/no/off``, or
      ``VMEX_COMPILATION_CACHE_DIR=disabled``, or an empty/disabled
      ``JAX_COMPILATION_CACHE_DIR``) turns the cache off, whatever else is set;
    - a jaxlib whose deserializer can crash the process
      (:func:`_cache_deserialize_unsafe`) keeps it off unless
      ``VMEX_COMPILATION_CACHE=1`` asks for it -- a cache path alone, often
      set by a cluster module, is not that request;
    - ``VMEX_COMPILATION_CACHE_DIR``, then ``JAX_COMPILATION_CACHE_DIR``,
      replace the parent directory, still split per machine.
    """
    flag = _env("COMPILATION_CACHE").strip().lower()
    vmex_dir = _env("COMPILATION_CACHE_DIR").strip()
    jax_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR")
    if (flag in _OFF or vmex_dir.lower() in _OFF
            or (jax_dir is not None and jax_dir.strip().lower() in _OFF + ("",))):
        return None
    if flag not in _ON and _cache_deserialize_unsafe():
        return None
    parent = vmex_dir or (jax_dir or "").strip() or str(
        pathlib.Path("~/.cache/vmex/jax_cache").expanduser())
    return _machine_scoped(parent)


def _apply_compilation_cache_policy(jax_module: Any) -> str | None:
    """Make JAX's persistent cache match :func:`_default_compilation_cache_dir`.

    JAX reads ``JAX_COMPILATION_CACHE_DIR`` itself at import, so without
    this a raw shared path (or a cache the policy turned off) would stay
    live.  A directory the program itself set through ``jax.config`` (neither
    that variable's value nor one vmex applied) is the caller's choice and is
    left as it is.  Idempotent: a config already matching the policy is left
    alone.
    """
    cache_dir = _default_compilation_cache_dir()
    current = jax_module.config.jax_compilation_cache_dir
    enabled = jax_module.config.jax_enable_compilation_cache
    env_dir = (os.environ.get("JAX_COMPILATION_CACHE_DIR") or "").strip()
    if current and current not in (env_dir, cache_dir, _APPLIED[0]):
        return current
    if cache_dir is not None and (current != cache_dir or not enabled):
        try:
            os.makedirs(cache_dir, exist_ok=True)
        except OSError:  # unwritable: no cache rather than JAX's raw path
            cache_dir = None
        else:
            _configure_compilation_cache(jax_module, cache_dir)
            _reset_jax_cache()
    if cache_dir is None and (current or enabled):
        jax_module.config.update("jax_enable_compilation_cache", False)
        jax_module.config.update("jax_compilation_cache_dir", None)
        _reset_jax_cache()
    _APPLIED[0] = cache_dir
    return cache_dir


_APPLIED: list[str | None] = [None]   # the directory this module last set


def _reset_jax_cache() -> None:
    """Drop JAX's already-opened cache object so the new setting applies."""
    with contextlib.suppress(Exception):
        from jax._src import compilation_cache

        compilation_cache.reset_cache()


def _accelerator_requested(names: tuple[str, ...]) -> bool:
    """True when the environment asks JAX for one of the ``names`` platforms."""
    requested = re.split(r"[\s,]+", (os.environ.get("JAX_PLATFORM_NAME", "") + ","
                                      + os.environ.get("JAX_PLATFORMS", "")).lower())
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip().lower()
    return any(name in requested for name in names) or visible not in ("", "-1", "none", "no")


def _configure_compilation_cache(jax_module: Any, cache_dir: str | None) -> None:
    """Apply vmex's persistent-cache defaults to an imported JAX module.

    Each option is applied on its own: a malformed ``VMEX_*`` value or a JAX
    without that option skips only that option.
    """
    if cache_dir is None:
        return

    def update(name: str, value) -> None:
        with contextlib.suppress(Exception):
            value = value()
            if value is not None:
                jax_module.config.update(name, value)

    def xla_caches() -> str | None:
        value = _env("PERSISTENT_CACHE_XLA_CACHES").strip() or (
            "xla_gpu_per_fusion_autotune_cache_dir"
            if _accelerator_requested(("gpu", "cuda")) else "none")
        return None if value.lower() in ("", "none", "0", "false", "no", "off") else value

    with contextlib.suppress(Exception):
        # Keep the directory small before anything writes to it: every write
        # re-scans it (see _prune_cache_entries).  0 disables the bound.
        max_entries = int(_env("CACHE_MAX_ENTRIES", str(_CACHE_MAX_ENTRIES)))
        if max_entries > 0:
            _prune_cache_entries(cache_dir, max_entries)
    update("jax_enable_compilation_cache", lambda: True)
    update("jax_compilation_cache_dir", lambda: cache_dir)
    # Store every compilation.  JAX's default 1 s floor left hundreds of
    # sub-second polishing programs unstored: the polished shaped-tokamak CLI
    # rerun measured 60.6 s at floor 1 versus 31.6 s at floor 0 (first runs
    # 81.5 s versus 90.7 s, the write cost, paid once).
    update("jax_persistent_cache_min_compile_time_secs",
           lambda: float(_env("CACHE_MIN_COMPILE_TIME_SECS", "0")))
    update("jax_persistent_cache_min_entry_size_bytes",
           lambda: int(_env("CACHE_MIN_ENTRY_SIZE_BYTES", "-1")))
    update("jax_persistent_cache_enable_xla_caches", xla_caches)
    # JAX's file cache takes its cross-process lock only when eviction is
    # enabled, so a finite default also keeps concurrent VMEX runs from
    # writing the same executable at once.
    update("jax_compilation_cache_max_size", lambda: int(size) if (
        size := _env("COMPILATION_CACHE_MAX_SIZE", str(_default_cache_max_size(cache_dir)))) else None)
    update("jax_explain_cache_misses", lambda: True if _env("EXPLAIN_CACHE_MISSES").strip().lower()
           not in ("", "0", "false", "no") else None)


_MAP_LIMIT_FILE = "/proc/sys/vm/max_map_count"
_MAPS_FILE = "/proc/self/maps"
_MAP_PRESSURE = 0.5     # release compiled executables above this share of the limit
_MAP_HEADROOM = 0.25    # ... and only after this much growth since the last release
_MAP_STATE: dict[str, Any] = {"floor": 0, "warned": False}


def _map_count() -> int | None:
    """Memory mappings of this process, or None where Linux /proc is absent."""
    with contextlib.suppress(OSError), open(_MAPS_FILE, "rb") as fh:
        return fh.read().count(b"\n")
    return None


def _relieve_map_pressure(jax_module: Any) -> bool:
    """Drop compiled executables before the process runs out of mappings.

    Each XLA:CPU kernel holds about three memory mappings until its
    executable dies, and JAX keeps every executable, so a long run reaches
    Linux's ``vm.max_map_count`` (65530 on most clusters), after which every
    compile or load fails with ENOMEM.  Called after each compilation (a JAX
    monitoring listener), this clears JAX's executable caches once the count
    passes half the limit and again only after another quarter of growth, so
    a working set near the limit recompiles instead of thrashing.  Never
    while a vmex prefetch thread compiles.  Returns True when it released.
    """
    import threading

    try:
        with open(_MAP_LIMIT_FILE, encoding="ascii") as fh:
            limit = int(fh.read().strip())
    except (OSError, ValueError):
        return False
    count = _map_count()
    if (count is None or count < max(_MAP_PRESSURE * limit, _MAP_STATE["floor"] + _MAP_HEADROOM * limit)
            or any("prefetch" in t.name and t.is_alive() for t in threading.enumerate())):
        return False
    import gc

    jax_module.clear_caches()
    gc.collect()
    _MAP_STATE["floor"] = after = _map_count() or 0
    if not _MAP_STATE["warned"]:
        _MAP_STATE["warned"] = True
        import warnings

        warnings.warn(
            f"vmex: this process held {count} memory mappings against the kernel limit "
            f"vm.max_map_count={limit}; compiled executables were released ({after} left) and "
            "will be recompiled when needed.  A larger limit (e.g. 1048576) avoids that.",
            RuntimeWarning, stacklevel=2)
    return True


def _after_compile(event: str, *_: Any, **__: Any) -> None:
    """JAX monitoring listener: the mapping check, at most once a second."""
    now = time.monotonic()
    if (event == "/jax/core/compile/backend_compile_duration"
            and now - _MAP_STATE.get("checked", -1.0) >= 1.0):
        _MAP_STATE["checked"] = now
        _relieve_map_pressure(sys.modules["jax"])


def _configure_jax_environment() -> None:
    """Set JAX/XLA environment defaults, then import + configure JAX.

    Runs once at ``vmex._jax_env`` import time — before
    ``vmex/__init__`` (or anything else in the package) imports JAX — so
    the env-var defaults reliably reach XLA backend initialization.  Every
    default uses ``setdefault``: an explicit user environment always wins.
    """
    try:
        # Default fresh JAX to x64; preserve an existing runtime precision
        # choice when the caller has not supplied an environment override.
        if "jax" not in sys.modules:
            os.environ.setdefault("JAX_ENABLE_X64", "1")
        # VMEC/JAX optimization callbacks immediately materialize most results
        # on the host (SciPy residuals/Jacobians, history, wout writing).  On
        # CPU, asynchronous dispatch can leave completed XLA/PjRt work and
        # executable state queued across many exact-Jacobian callbacks in one
        # long-lived process.  Default CPU dispatch to synchronous execution so
        # memory is reclaimed at callback boundaries; users can still override
        # this before import with JAX_CPU_ENABLE_ASYNC_DISPATCH=true.
        os.environ.setdefault("JAX_CPU_ENABLE_ASYNC_DISPATCH", "false")
        # Suppress harmless informational C++ logs from XLA/PjRt.  Level
        # 0=INFO, 1=WARNING, 2=ERROR — default to ERROR-only so genuine errors
        # still surface.  JAX/jaxlib 0.9.1-0.9.2 can nevertheless print a
        # spurious PJRT warning on persistent-cache hits; the upstream fix is
        # in 0.10.0+, and the WSL driver-parser fix joins it in 0.10.1.  Do not
        # raise this to level 3 merely to hide the separate CUDA error stream.
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")
        os.environ.setdefault("GLOG_minloglevel", "2")
        # JAX's default GPU allocator preallocates most device memory.  That
        # hurts vmex's exact-optimizer workload in practice: it prevents
        # concurrent profiling/worker processes from starting and can make the
        # accepted-point replay path much slower.  Default to demand allocation
        # unless the user already set JAX's allocator env var or explicitly
        # asks vmex to keep JAX's preallocation default.
        _vmec_gpu_prealloc = _env("GPU_PREALLOCATE").strip().lower()
        if (
            "XLA_PYTHON_CLIENT_PREALLOCATE" not in os.environ
            and _vmec_gpu_prealloc not in ("1", "true", "yes", "on")
        ):
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

        # XLA:CPU compile-time flags.  The differentiable/optimization pipeline
        # is COMPILE-dominated (the fused adjoint VJP + GMRES graph dominates a
        # cold ``value_and_grad``); backend optimization level 1 plus disabling
        # expensive LLVM passes cuts compile wall-time ~1.3-2x at the cost of
        # slightly slower *warm* kernels.  Applied on CPU only (LLVM codegen),
        # never with fast-math (that would break float64 parity/determinism),
        # skipped if the user set XLA_FLAGS, and opt-in via
        # VMEX_FAST_COMPILE=1.  Pre-import environment hints cannot reliably
        # distinguish a normally discovered GPU installation, so VMEX must
        # not inject optional CPU tuning by default.  The macOS linker guard
        # below is a separate correctness default for large graphs.
        _fast_compile = _env("FAST_COMPILE", "0").strip().lower()
        _on_accel = _accelerator_requested(("cuda", "gpu", "tpu", "rocm"))
        if "XLA_FLAGS" not in os.environ:
            # A solver iteration is many small kernels, so on a GPU it is
            # launch-bound: let XLA capture every fused run into a CUDA graph,
            # not only runs of five or more kernels (8-11 % per iteration on
            # an RTX A4000, ns 16-101).  The CPU backend ignores GPU flags.
            _xla_flags = ["--xla_gpu_graph_min_graph_size=1"]
            if not _on_accel:
                # Large differentiated single-stage graphs can exhaust the small
                # macOS worker-thread stack while LLVM links its default 32 object
                # partitions.  Finer partitioning bounds linker recursion without
                # changing the executable's numerical operations.
                if platform.system() == "Darwin":
                    _xla_flags.append("--xla_cpu_parallel_codegen_split_count=128")
                if _fast_compile not in ("0", "false", "no", "off"):
                    _xla_flags.extend((
                        "--xla_backend_optimization_level=1",
                        "--xla_llvm_disable_expensive_passes=true",
                    ))
            os.environ["XLA_FLAGS"] = " ".join(_xla_flags)

        import jax

        try:
            if "JAX_ENABLE_X64" in os.environ:
                # JAX's own boolean spellings; anything else keeps the runtime.
                value = os.environ["JAX_ENABLE_X64"].lower()
                truthy = ("y", "yes", "t", "true", "on", "1")
                if value in truthy or value in ("n", "no", "f", "false", "off", "0"):
                    jax.config.update("jax_enable_x64", value in truthy)
        except Exception:
            pass
        try:
            _cpu_async = os.environ.get("JAX_CPU_ENABLE_ASYNC_DISPATCH", "true")
            jax.config.update(
                "jax_cpu_enable_async_dispatch",
                _cpu_async.strip().lower() not in ("0", "false", "no", "off"),
            )
        except Exception:
            pass

        # The persistent cache is configured through jax.config only, never
        # exported: a job launched from this process may run on another node.
        _apply_compilation_cache_policy(jax)
        if os.path.exists(_MAP_LIMIT_FILE) and not _MAP_STATE.get("listening"):
            from jax import monitoring

            monitoring.register_event_duration_secs_listener(_after_compile)
            _MAP_STATE["listening"] = True
    except Exception:
        # Never block a vmex import over environment tuning (e.g. docs
        # builds with a mocked JAX): core.solver enforces the hard
        # requirements (x64, cache hardening) on every solve path anyway.
        pass


_configure_jax_environment()


# Floors of the optional extras, equal to pyproject.toml (a test pins them).
# Checked where each is first used, so that an environment left behind by
# `git pull` over an editable install fails with the fix, not a TypeError.
OPTIONAL_MINIMUMS = {"essos": "0.21.0", "virtual-casing-jax": "0.0.9",
                     "neo-jax": "1.0.5", "gkx": "2.5.0",
                     "dkx": "2.8.0"}


def require_optional(name: str, feature: str) -> None:
    """Raise ImportError naming the fix if ``name`` is missing or below its floor."""
    from packaging.version import Version

    minimum = OPTIONAL_MINIMUMS[name]
    try:
        found = importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        raise ImportError(f'{feature} needs {name}>={minimum}; run: pip install "{name}>={minimum}"') from None
    if Version(found) < Version(minimum):
        raise ImportError(f'{feature} needs {name}>={minimum} (found {found}); '
                          f'run: pip install -U "{name}>={minimum}"')
