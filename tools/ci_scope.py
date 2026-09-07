#!/usr/bin/env python3
"""Classify which numerical lanes a pull request needs, and whether coverage applies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from collections.abc import Iterable


DOCUMENTATION_SUFFIXES = {
    ".gif",
    ".jpeg",
    ".jpg",
    ".md",
    ".pdf",
    ".png",
    ".rst",
    ".svg",
    ".webp",
}


def classify(paths: Iterable[str], *, force_all: bool = False) -> tuple[bool, bool]:
    """Return ``(run_tests, run_coverage)`` for changed repository paths.

    Only documentation and rendered media bypass the numerical matrix. Unknown
    files remain conservative and run it. Changed-line coverage is necessary
    only when executable package code changed; test, workflow, and tool changes
    still run the matrix but do not spend another job combining coverage.
    """

    if force_all:
        return True, True
    changed = tuple(path.strip("/") for path in paths if path.strip("/"))
    if not changed:
        return True, True
    documentation_only = all(
        "/." not in path
        and any(path.lower().endswith(suffix) for suffix in DOCUMENTATION_SUFFIXES)
        for path in changed
    )
    run_tests = not documentation_only
    run_coverage = any(
        path.startswith("vmex/") and path.lower().endswith(".py")
        for path in changed
    )
    return run_tests, run_coverage


# -- lane selection ----------------------------------------------------------
#
# Whole-matrix runs are the safe default. Only changes confined to test,
# benchmark, example and documentation files can be narrowed, because nothing
# there is imported by the package itself: the lanes that own the touched test
# modules are sufficient. Package code, tooling, workflows and packaging all
# keep the full matrix, since any of them can change any lane's result.

ROOT = Path(__file__).resolve().parents[1]
NARROWABLE_PREFIXES = ("tests/", "benchmarks/", "examples/", "docs/")
_IMPORTS = re.compile(r"^\s*(?:from|import)\s+(benchmarks|examples)\b", re.MULTILINE)


def _owning_lanes(modules: Iterable[str], manifest: dict) -> set[str]:
    """PR lanes that own any of ``modules``, by the manifest's own ownership."""
    wanted = set(modules)
    lanes: set[str] = set()
    for record in manifest["records"]:
        path, lane_list = record[0], record[-1]
        if path in wanted:
            lanes |= {lane for lane in lane_list if lane.startswith("pr-")}
    return lanes


def _tests_referencing(paths: Iterable[str], root: Path) -> set[str]:
    """Test modules that import or name any of ``paths``.

    Tests reach benchmark and example code both ways: ``from benchmarks...``
    for helpers, and a literal path for scripts they execute. Missing either
    would silently drop a lane, so both are matched.
    """
    targets = [path for path in paths if path.endswith(".py")]
    if not targets:
        return set()
    packages = {path.split("/", 1)[0] for path in targets}
    stems = {Path(path).stem for path in targets}
    found = set()
    for path in sorted((root / "tests").rglob("test_*.py")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        imported = any(m.group(1) in packages for m in _IMPORTS.finditer(text))
        named = any(stem in text for stem in stems)
        if imported or named:
            found.add(path.relative_to(root).as_posix())
    return found


def needed_lanes(paths: Iterable[str], *, manifest=None, root=None) -> set[str] | None:
    """PR lanes this change needs, or ``None`` when the full matrix is required.

    An empty set means no numerical lane is needed at all, which only happens
    for documentation and data files that no test names.
    """
    root = Path(root) if root is not None else ROOT
    changed = tuple(path.strip("/") for path in paths if path.strip("/"))
    if not changed or not all(path.startswith(NARROWABLE_PREFIXES) for path in changed):
        return None
    if manifest is None:
        manifest = json.loads((root / "tests" / "manifest.json").read_text(encoding="utf-8"))

    modules = {path for path in changed if path.startswith("tests/") and path.endswith(".py")}
    scripts = [path for path in changed
               if path.startswith(("benchmarks/", "examples/")) and path.endswith(".py")]
    modules |= _tests_referencing(scripts, root)
    unattributed = [path for path in changed
                    if path.endswith(".py") and path not in modules
                    and path not in scripts]
    if unattributed:
        return None
    if scripts and not modules:           # a script no test names: stay safe
        return None
    if not modules:
        return set()
    lanes = _owning_lanes(modules, manifest)
    return lanes or None                  # an unowned module: stay safe


def select_lanes(paths: Iterable[str], entries, *, manifest=None, root=None) -> list:
    """Return the matrix ``entries`` this change needs, conservatively."""
    lanes = needed_lanes(paths, manifest=manifest, root=root)
    if lanes is None:
        return list(entries)
    return [entry for entry in entries
            if any(one in lanes for one in str(entry.get("selector", "")).split())]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="select the full main-branch gate")
    parser.add_argument("--null", action="store_true", help="read NUL-delimited paths")
    parser.add_argument("--matrix", help="emit the needed subset of this JSON matrix")
    args = parser.parse_args(argv)
    payload = sys.stdin.buffer.read()
    separator = b"\0" if args.null else b"\n"
    paths = [
        item.decode("utf-8", errors="surrogateescape")
        for item in payload.split(separator)
    ]
    run_tests, run_coverage = classify(paths, force_all=args.all)
    print(f"run_tests={str(run_tests).lower()}")
    print(f"run_coverage={str(run_coverage).lower()}")
    if args.matrix:
        entries = json.loads(args.matrix)
        needed = entries if args.all else select_lanes(paths, entries)
        print(f"matrix={json.dumps(needed, separators=(',', ':'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
