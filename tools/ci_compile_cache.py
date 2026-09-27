"""Carry the persistent XLA compilation cache from one CI run to the next.

CI restores a lane's cache with ``actions/cache``, runs the lane, and saves
it again only from ``main``. Three rules keep that safe:

- The restored cache must have been built on the same CPU (model and
  flags), Python and jaxlib build: XLA:CPU entries are native executables,
  and loading one compiled for another CPU is the crash vmex's own cache
  fingerprint exists to prevent. ``key`` prints that fingerprint for the
  ``actions/cache`` key, and ``prepare`` discards a seed recorded under any
  other fingerprint.
- Nothing is shared between processes while tests run. ``tests/conftest.py``
  gives every pytest process a private hard-linked copy of the seed and
  opens it with eviction off, so JAX takes no cross-process lock (the lock
  that serialized pytest-xdist workers until lanes hit their timeout) and
  no file is ever rewritten in place.
- The saved cache holds only what this run used. ``prepare`` backdates the
  access time of every restored entry; ``collect`` merges the per-process
  copies back and keeps the entries read or written during the run.

Usage::

    python tools/ci_compile_cache.py key            # fingerprint=<hex>
    python tools/ci_compile_cache.py prepare ROOT
    python tools/ci_compile_cache.py collect ROOT   # entries=<n>
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import os
import platform
import shutil
import sys
import time
from pathlib import Path

_ENTRY = "-cache"             # jax._src.lru_cache entry suffix
_OLD_ATIME_NS = 946_684_800 * 10**9  # 2000-01-01: older than any entry's mtime


def fingerprint() -> str:
    """Digest of everything an XLA:CPU executable is only valid for."""
    parts = [platform.system(), platform.machine(),
             f"python={sys.version_info.major}.{sys.version_info.minor}"]
    try:
        parts.append(f"jaxlib={importlib.metadata.version('jaxlib')}")
        spec = importlib.util.find_spec("jaxlib")
        for folder in (spec.submodule_search_locations or []) if spec else []:
            for lib in sorted(Path(folder).glob("_jax*.so")):
                with lib.open("rb") as fh:
                    parts.append(f"{lib.name}:{lib.stat().st_size}:"
                                 f"{hashlib.sha256(fh.read(1 << 20)).hexdigest()}")
    except Exception as exc:  # no jaxlib: nothing worth caching
        parts.append(f"no-jaxlib={type(exc).__name__}")
    try:
        seen = set()
        for line in Path("/proc/cpuinfo").read_text(errors="ignore").splitlines():
            key, _, value = (s.strip() for s in line.partition(":"))
            if key in ("model name", "flags", "Features") and key not in seen:
                parts.append(f"{key}={value}")
                seen.add(key)
    except OSError:
        parts.append(platform.processor())
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _entries(folder: Path) -> list[Path]:
    return sorted(folder.glob(f"*{_ENTRY}")) if folder.is_dir() else []


def _size_mb(paths: list[Path]) -> float:
    return sum(p.stat().st_size for p in paths) / 2**20


def _report(root: Path, text: str) -> None:
    print(text, file=sys.stderr)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(f"- XLA cache ({root.name}): {text}\n")


def prepare(root: Path) -> None:
    """Validate the restored seed, backdate its access times, and mark now."""
    seed, stamp = root / "seed", root / "fingerprint"
    mine = fingerprint()
    if seed.exists() and (not stamp.exists() or stamp.read_text().strip() != mine):
        shutil.rmtree(seed)
        print(f"discarded a seed recorded under another fingerprint (now {mine})")
    seed.mkdir(parents=True, exist_ok=True)
    stamp.write_text(mine + "\n")
    for proc in root.glob("proc-*"):
        shutil.rmtree(proc)
    entries = _entries(seed)
    for entry in entries:
        os.utime(entry, ns=(_OLD_ATIME_NS, entry.stat().st_mtime_ns))
    # Pruning by use needs the filesystem to record reads (relatime does,
    # noatime does not); without that, collect keeps every entry.
    probe = root / "atime-probe"
    probe.write_bytes(b"x")
    os.utime(probe, ns=(_OLD_ATIME_NS, probe.stat().st_mtime_ns))
    marker = time.time_ns()
    probe.read_bytes()
    tracked = probe.stat().st_atime_ns >= marker - 10**9
    probe.unlink()
    (root / "marker").write_text(f"{marker - 10**9} {int(tracked)}\n")
    _report(root, f"restored {len(entries)} entries, {_size_mb(entries):.0f} MB; "
                  f"access times {'tracked' if tracked else 'NOT tracked'}")


def collect(root: Path) -> None:
    """Merge every process's copy into the seed and drop unused entries."""
    seed = root / "seed"
    marker, tracked = (int(v) for v in (root / "marker").read_text().split())
    restored = {p.name for p in _entries(seed)}
    for proc in sorted(root.glob("proc-*")):
        for entry in _entries(proc):
            target = seed / entry.name
            if not target.exists():
                os.link(entry, target)
        shutil.rmtree(proc)
    kept, dropped, hits, new = [], 0, 0, 0
    for entry in _entries(seed):
        st = entry.stat()
        fresh = entry.name not in restored
        used = fresh or not tracked or st.st_atime_ns >= marker
        hits += (not fresh) and st.st_atime_ns >= marker
        new += fresh
        if used:
            kept.append(entry)
        else:
            entry.unlink()
            dropped += 1
    _report(root, f"{hits} of {len(restored)} restored entries read, {new} new, "
                  f"{dropped} unused dropped; saving {len(kept)} entries, "
                  f"{_size_mb(kept):.0f} MB")
    print(f"entries={len(kept)}")  # the save step skips an empty cache


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("key", "prepare", "collect"))
    parser.add_argument("root", nargs="?", type=Path)
    args = parser.parse_args(argv)
    if args.command == "key":
        cpuinfo = Path("/proc/cpuinfo")
        lines = cpuinfo.read_text(errors="ignore").splitlines() if cpuinfo.exists() else []
        cpu = next((line.split(":", 1)[1].strip() for line in lines
                    if line.startswith("model name")), platform.processor())
        print(f"XLA cache for {cpu}", file=sys.stderr)
        print(f"fingerprint={fingerprint()}")
    elif args.root is None:
        parser.error(f"{args.command} needs ROOT")
    elif args.command == "prepare":
        prepare(args.root)
    else:
        collect(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
