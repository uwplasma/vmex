"""JAX environment defaults + persistent compilation-cache policy.

The old JAX/NumPy backend shim is gone (the core is JAX-only); what remains —
and is actually used — is:

- :func:`_configure_jax_environment` (run at import, i.e. before
  ``vmex/__init__`` does ``import jax``): environment defaults that must
  be set before JAX/XLA initializes — float64 (``JAX_ENABLE_X64``, VMEC
  parity), synchronous CPU dispatch, quiet XLA/PjRt C++ logging, GPU
  demand allocation, the machine-scoped persistent compilation-cache
  directory, and the XLA:CPU compiler flags/guards;
- the compilation-cache policy helpers
  :func:`_default_compilation_cache_dir` / :func:`_cache_machine_fingerprint`
  / :func:`_configure_compilation_cache`, consumed by ``vmex/__init__``
  and re-applied by ``core.solver._harden_compilation_cache`` on every solve
  path (namespace-package shadowing guard).
"""

from __future__ import annotations

from typing import Any
import contextlib
import hashlib
import re
from importlib import metadata as importlib_metadata
import sys

import os
import platform


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

    JAX's LRU cache re-scans the whole cache directory on every *write*: each
    ``put`` globs the directory, stats every entry and reads every atime
    sidecar, all under the directory-wide lock.  The cost is linear in the
    number of resident entries and, measured on an Apple M4, is 0.028 ms per
    entry -- 5.7 ms per write at 250 entries but 304 ms at 10880, the size a
    developer machine reaches in a few weeks.  With vmex's floor-0 policy (we
    store every program, which is worth it on reload) a cold QA solve writes
    ~170 entries, so a mature cache turned a 7.5 s solve into 31.3 s, 82% of
    it directory scans.  JAX's own bound is on bytes, and these entries are
    small, so it never fires.

    Bounding the entry count instead costs one scan per process rather than
    one per write.  A fixed bound alone evicts a large workload's own working
    set: the QI optimization example writes 1,342 executables, so a 1,024 cap
    dropped 318 of them at every import, and every returning run recompiled
    and rewrote the same 318.  Entries used within ``_CACHE_RECENT_SECONDS``
    are therefore kept up to four times ``max_entries``, and only older ones
    are trimmed to ``max_entries``.  Measured on a 36-thread Xeon: the QI
    example's warm run misses nothing (its compile 68.0 s -> 38.6 s); a cold
    seed-deck solve against 4,026 stale entries still sees the cache pruned to
    1,024 (10.2 s, against 9.7 s with the plain cap); and the worst case, 4,026
    recent entries, costs 17.9 s, where a fixed 4,096 cap costs 19.0 s on any
    mature cache (a write scans 99 ms at 4,026 entries against 26 ms at 1,024).
    Returns the number of entries removed.
    """
    try:
        import pathlib

        path = pathlib.Path(cache_dir)
        entries = list(path.glob(f"*{_CACHE_ENTRY_SUFFIX}"))
        if len(entries) <= max_entries:
            return 0
    except Exception:  # missing directory, unreadable filesystem
        return 0

    lock = None
    try:
        import filelock

        lock = filelock.FileLock(path / ".lockfile")
        lock.acquire(timeout=5)
    except Exception:
        # No filelock, or another process holds it and is likely pruning too.
        return 0

    removed = 0
    try:
        def _atime(entry: "pathlib.Path") -> int:
            sidecar = entry.with_name(
                entry.name[: -len(_CACHE_ENTRY_SUFFIX)] + _CACHE_ATIME_SUFFIX
            )
            try:
                return int.from_bytes(sidecar.read_bytes(), "little")
            except Exception:
                return 0

        import time

        atimes = {entry: _atime(entry) for entry in entries}
        ordered = sorted(entries, key=atimes.__getitem__, reverse=True)
        recent_cutoff = time.time_ns() - _CACHE_RECENT_SECONDS * 1_000_000_000
        recent = sum(1 for entry in ordered if atimes[entry] >= recent_cutoff)
        keep = max(max_entries, min(recent, 4 * max_entries))
        for entry in ordered[keep:]:
            sidecar = entry.with_name(
                entry.name[: -len(_CACHE_ENTRY_SUFFIX)] + _CACHE_ATIME_SUFFIX
            )
            try:
                entry.unlink()
                removed += 1
            except OSError:
                continue
            try:
                sidecar.unlink()
            except OSError:
                pass
    finally:
        try:
            lock.release()
        except Exception:
            pass
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
    import pathlib

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
    import pathlib

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


def _configure_compilation_cache(jax_module: Any, cache_dir: str | None) -> None:
    """Apply vmex's persistent-cache defaults to an imported JAX module."""
    if cache_dir is None:
        return
    try:
        # Keep the directory small before anything writes to it: every write
        # re-scans it (see _prune_cache_entries).  0 disables the bound.
        max_entries = int(_env("CACHE_MAX_ENTRIES", str(_CACHE_MAX_ENTRIES)))
        if max_entries > 0:
            _prune_cache_entries(cache_dir, max_entries)
    except Exception:
        pass
    try:
        jax_module.config.update("jax_enable_compilation_cache", True)
    except Exception:
        pass
    try:
        jax_module.config.update("jax_compilation_cache_dir", cache_dir)
    except Exception:
        pass
    try:
        # Store every compilation. The 1 s floor JAX defaults to left most of
        # the polishing path's programs unstored - hundreds of sub-second
        # compiles whose reload savings add up: the polished shaped-tokamak
        # CLI rerun measured 60.6 s at floor 1 versus 31.6 s at floor 0
        # (first runs 81.5 s versus 90.7 s, the write cost, paid once). Disk
        # stays bounded by the eviction cap configured below.
        min_compile = _env("CACHE_MIN_COMPILE_TIME_SECS", "0")
        jax_module.config.update("jax_persistent_cache_min_compile_time_secs", float(min_compile))
    except Exception:
        pass
    try:
        min_entry = _env("CACHE_MIN_ENTRY_SIZE_BYTES", "-1")
        jax_module.config.update("jax_persistent_cache_min_entry_size_bytes", int(min_entry))
    except Exception:
        pass
    try:
        xla_caches = _env("PERSISTENT_CACHE_XLA_CACHES").strip()
        if not xla_caches:
            platform_name = os.environ.get("JAX_PLATFORM_NAME", "").strip().lower()
            platforms = os.environ.get("JAX_PLATFORMS", "").strip().lower()
            visible_cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip().lower()
            gpu_requested = (
                platform_name in ("gpu", "cuda")
                or any(part.strip() in ("gpu", "cuda") for part in platforms.split(","))
                or visible_cuda not in ("", "-1", "none", "no")
            )
            xla_caches = "xla_gpu_per_fusion_autotune_cache_dir" if gpu_requested else "none"
        if xla_caches.lower() not in ("", "none", "0", "false", "no", "off"):
            jax_module.config.update("jax_persistent_cache_enable_xla_caches", xla_caches)
    except Exception:
        pass
    try:
        # JAX's file cache takes its cross-process lock only when eviction is
        # enabled.  A finite default therefore prevents concurrent VMEX runs
        # from writing the same executable at once, as well as bounding disk.
        max_size = _env("COMPILATION_CACHE_MAX_SIZE",
                        str(_default_cache_max_size(cache_dir)))
        if max_size:
            jax_module.config.update("jax_compilation_cache_max_size", int(max_size))
    except Exception:
        pass
    try:
        explain = _env("EXPLAIN_CACHE_MISSES")
        if explain.strip().lower() not in ("", "0", "false", "no"):
            jax_module.config.update("jax_explain_cache_misses", True)
    except Exception:
        pass


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

    Every XLA:CPU kernel lives in its own JIT object with separate code,
    read-only and data mappings -- about three per kernel -- until its
    executable dies, and JAX keeps every executable it compiled.  A long
    free-boundary or optimization run reaches tens of thousands of mappings;
    at Linux's per-process ``vm.max_map_count`` (65530 by default on most
    clusters) every further ``mmap`` fails, so the next compile or load
    fails ("LLVM compilation error: Cannot allocate memory", "Failed to
    materialize symbols") or aborts the process.  Run after every XLA
    compilation or cache load (a JAX monitoring listener), this clears JAX's
    executable caches once the count passes half the limit, and again only
    after another quarter of growth, so a working set near the limit
    degrades to recompiling instead of thrashing.  Never while a vmex
    prefetch thread compiles.  Returns True when it released.
    """
    import threading

    try:
        with open(_MAP_LIMIT_FILE, encoding="ascii") as fh:
            limit = int(fh.read().strip())
    except (OSError, ValueError):
        return False
    count = _map_count()
    if (count is None or count < _MAP_PRESSURE * limit
            or count < _MAP_STATE["floor"] + _MAP_HEADROOM * limit):
        return False
    if any("prefetch" in t.name and t.is_alive() for t in threading.enumerate()):
        return False
    import gc

    jax_module.clear_caches()
    gc.collect()
    after = _map_count() or 0
    _MAP_STATE["floor"] = after
    if not _MAP_STATE["warned"]:
        _MAP_STATE["warned"] = True
        import warnings

        warnings.warn(
            f"vmex: this process held {count} memory mappings against the "
            f"kernel limit vm.max_map_count={limit}; compiled executables "
            f"were released ({after} mappings left) and will be recompiled "
            "when needed.  A larger limit (e.g. vm.max_map_count=1048576, "
            "set by an administrator) avoids the recompilation.",
            RuntimeWarning, stacklevel=2)
    return True


def _after_compile(event: str, *_: Any, **__: Any) -> None:
    """JAX monitoring listener: the mapping check, at most once a second."""
    import time

    now = time.monotonic()
    if (event == "/jax/core/compile/backend_compile_duration"
            and now - _MAP_STATE.get("checked", -1.0) >= 1.0):
        _MAP_STATE["checked"] = now
        _relieve_map_pressure(sys.modules["jax"])


def _configure_jax_environment() -> None:
    """Set JAX/XLA environment defaults, then import + configure JAX.

    Runs once at ``vmex._compat`` import time — before
    ``vmex/__init__`` (or anything else in the package) imports JAX — so
    the env-var defaults reliably reach XLA backend initialization.  Every
    default uses ``setdefault``: an explicit user environment always wins.
    """
    try:
        # Enable x64 by default for VMEC parity unless the user opted out.
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
        _accel_req = os.environ.get("JAX_PLATFORM_NAME", "").strip().lower()
        _accel_reqs = os.environ.get("JAX_PLATFORMS", "").strip().lower()
        _cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        _on_accel = (
            any(a in f"{_accel_req} {_accel_reqs}" for a in ("cuda", "gpu", "tpu", "rocm"))
            or (_cuda_vis not in ("", "-1"))
        )
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
            jax.config.update("jax_enable_x64", os.environ.get("JAX_ENABLE_X64", "0") == "1")
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
OPTIONAL_MINIMUMS = {"essos": "0.20.0", "virtual-casing-jax": "0.0.9",
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
