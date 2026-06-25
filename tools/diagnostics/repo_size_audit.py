#!/usr/bin/env python
"""Audit tracked repository size and, optionally, ignored local artifacts.

This script intentionally uses only the Python standard library.  It reports
the size of files tracked by git, grouped by top-level directory, and can fail
when a total or per-file threshold is exceeded.  Use it while moving generated
figures and large reference outputs out of the repository.

Pass ``--include-ignored`` during release cleanup to quantify local Sphinx
builds, WOUTs, BOOZ files, optimization outputs, and fetched assets that are
ignored by git but can make a working tree look much larger than a fresh clone.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys


BYTES_PER_MIB = 1024 * 1024


@dataclass(frozen=True)
class TrackedFile:
    path: Path
    size: int


def _run_git_files(args: list[str], *, nul: bool = False) -> list[Path]:
    proc = subprocess.run(
        ["git", *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if nul:
        return [Path(line) for line in proc.stdout.split("\0") if line.strip()]
    return [Path(line) for line in proc.stdout.splitlines() if line.strip()]


def _tracked_files() -> list[TrackedFile]:
    files: list[TrackedFile] = []
    for path in _run_git_files(["ls-files"]):
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            continue
        files.append(TrackedFile(path=path, size=size))
    return files


def _ignored_files() -> list[TrackedFile]:
    files: list[TrackedFile] = []
    for path in _run_git_files(["ls-files", "--others", "--ignored", "--exclude-standard", "-z"], nul=True):
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            continue
        if path.is_file():
            files.append(TrackedFile(path=path, size=size))
    return files


def _mib(size: int) -> float:
    return float(size) / float(BYTES_PER_MIB)


def _prefix(path: Path) -> str:
    return path.parts[0] if path.parts else "."


def _print_report(files: list[TrackedFile], *, top: int, label: str = "Tracked") -> None:
    total = sum(item.size for item in files)
    by_prefix: dict[str, int] = defaultdict(int)
    for item in files:
        by_prefix[_prefix(item.path)] += item.size

    label_lower = label.lower()
    print(f"{label} files: {len(files)}")
    print(f"{label} size:  {_mib(total):.2f} MiB")
    print()
    print("By top-level path:")
    for prefix, size in sorted(by_prefix.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {_mib(size):8.2f} MiB  {prefix}")

    print()
    print(f"Largest {top} {label_lower} files:")
    for item in sorted(files, key=lambda entry: entry.size, reverse=True)[:top]:
        print(f"  {_mib(item.size):8.2f} MiB  {item.path}")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=30, help="Number of largest tracked files to print.")
    parser.add_argument(
        "--max-total-mib",
        type=float,
        default=None,
        help="Fail if total tracked size exceeds this many MiB.",
    )
    parser.add_argument(
        "--max-file-mib",
        type=float,
        default=None,
        help="Fail if any single tracked file exceeds this many MiB.",
    )
    parser.add_argument(
        "--include-ignored",
        action="store_true",
        help="Also report ignored local artifacts. This is informational and does not affect tracked-size failures.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    files = _tracked_files()
    _print_report(files, top=max(0, int(args.top)))
    if args.include_ignored:
        print()
        print("Ignored local artifact report:")
        _print_report(_ignored_files(), top=max(0, int(args.top)), label="Ignored")

    failed = False
    total = sum(item.size for item in files)
    if args.max_total_mib is not None and _mib(total) > float(args.max_total_mib):
        print(
            f"\nFAIL: tracked size {_mib(total):.2f} MiB exceeds "
            f"--max-total-mib={float(args.max_total_mib):.2f}",
            file=sys.stderr,
        )
        failed = True
    if args.max_file_mib is not None:
        max_file_bytes = float(args.max_file_mib) * BYTES_PER_MIB
        offenders = [item for item in files if item.size > max_file_bytes]
        if offenders:
            print(
                f"\nFAIL: {len(offenders)} tracked files exceed "
                f"--max-file-mib={float(args.max_file_mib):.2f}",
                file=sys.stderr,
            )
            for item in sorted(offenders, key=lambda entry: entry.size, reverse=True)[:10]:
                print(f"  {_mib(item.size):8.2f} MiB  {item.path}", file=sys.stderr)
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
