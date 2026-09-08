#!/usr/bin/env python3
"""Render ``benchmarks/INDEX.md``: every committed benchmark artifact, its owner and its readers.

The index answers three questions that the tree alone cannot:

1.  Which script produced the artifact (``GENERATORS`` below; the cell is left
    blank for the records that no script in this tree writes).
2.  Which revision and date it was measured at, read out of whatever
    provenance block the artifact carries -- the schemas differ by age, so the
    keys are searched rather than assumed.
3.  Which pages, tests or scripts cite it.  An artifact nothing cites is
    listed as such instead of being quietly deleted: an uncited record may
    still be evidence, but the reader is entitled to know it stands alone.

The 54 committed workflow baselines under ``benchmarks/baselines/m4/`` are one
grouped row -- one line per file would bury the 33 records that pages cite --
but the group still carries its file count, so a file added or removed there
changes the index.

Usage::

    python tools/render_benchmark_index.py           # rewrite in place
    python tools/render_benchmark_index.py --check   # exit 1 when stale
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BENCHMARKS = REPO / "benchmarks"
INDEX = BENCHMARKS / "INDEX.md"
SELF = Path(__file__).resolve().relative_to(REPO).as_posix()

#: Suffixes that make a file under ``benchmarks/`` an artifact (a record) as
#: opposed to a generator (``.py``).
ARTIFACT_SUFFIXES = (".json", ".md")

#: Directories rendered as a single grouped row instead of one row per file.
GROUPS = ("benchmarks/baselines/m4",)

#: Artifact glob -> the in-tree script that writes it.  Most artifacts do not
#: name their own writer (only the newer schemas record a ``command``), so the
#: mapping lives here, in code, and ``tests/test_cited_paths.py`` fails when an
#: artifact matches no pattern or a pattern names a script that does not exist.
#: First match wins, so the specific patterns come before the general ones.
GENERATORS: tuple[tuple[str, str], ...] = (
    ("benchmarks/baselines/m4/*.json", "benchmarks/profile_workflows.py"),
    ("benchmarks/baseline.json", "benchmarks/run_baseline.py"),
    ("benchmarks/gpu_baseline.json", "benchmarks/run_gpu_matrix.py"),
    ("benchmarks/freeboundary_multigrid.json", "benchmarks/run_freeboundary_multigrid.py"),
    ("benchmarks/high_mode_fft.json", "benchmarks/run_high_mode_fft.py"),
    ("benchmarks/convergence_nfp4_ns51.json", "benchmarks/make_readme_figures.py"),
    ("benchmarks/preconditioner_2d_stiff_cases.json", "benchmarks/preconditioner_2d_stiff.py"),
    ("benchmarks/capabilities.json", "tools/render_capabilities.py"),
    ("benchmarks/profile.json", "benchmarks/profile_production.py"),
    ("benchmarks/device_cache_reload_m4.json", "benchmarks/device_cache_reload.py"),
    ("benchmarks/polish_cost_*.json", "benchmarks/polish_cost.py"),
    ("benchmarks/polish_memory_*.json", "benchmarks/polish_memory.py"),
    ("benchmarks/polish_implicit_*.json", "benchmarks/polish_implicit.py"),
    ("benchmarks/polish_preconditioner_*.json", "benchmarks/polish_preconditioner.py"),
    ("benchmarks/polish_force_error_*.json", "benchmarks/strong_polish.py"),
    ("benchmarks/qa_optimization_startup_*.json", "benchmarks/qa_optimization_startup.py"),
    ("benchmarks/strong_force_m4.json", "benchmarks/strong_force.py"),
    ("benchmarks/strong_force_cases_m4.json", "benchmarks/make_strong_force_comparison.py"),
    ("benchmarks/strong_force_comparison_m4.json", "benchmarks/make_strong_force_comparison.py"),
    ("benchmarks/strong_root_*.json", "benchmarks/strong_root.py"),
    # Hand-recorded records: no script in the tree writes them.  They are
    # status and narrative records gated by tests or quoted by docs.
    ("benchmarks/mirror_*.json", ""),
    ("benchmarks/cache_entry_scaling_*.json", ""),
    ("benchmarks/desc_native_vs_lifted_*.json", ""),
    ("benchmarks/fresh_decks_vs_vmec2000_*.json", ""),
    ("benchmarks/review_*.json", ""),
    ("benchmarks/*.md", ""),
)

#: Provenance containers, searched in order; ``None`` means the top level.
PROVENANCE_KEYS = ("_provenance", "provenance", None)
COMMIT_KEYS = ("measurement_commit", "vmex_commit", "commit")
DATE_KEYS = ("measurement_date", "date", "created_utc", "review_date_utc", "utc")

#: Where a citation may come from.  ``benchmarks`` is included because a script
#: that reads another script's record is a consumer of it.
CITING_ROOTS = ("README.md", "CHANGELOG.md", "plan.md", "docs", "tests", "tools",
                "vmex", "examples", "benchmarks", ".github")
CITING_SUFFIXES = (".md", ".rst", ".py", ".json", ".yml", ".yaml", ".toml", ".cfg", ".txt")

#: A file or glob token naming a record, and a token naming a directory.
FILE_TOKEN = re.compile(r"(?:\.\./)*([A-Za-z0-9_*][A-Za-z0-9_./*-]*\.(?:json|md))")
DIR_TOKEN = re.compile(r"(?<![\w/])(?:\.\./)*(benchmarks/[A-Za-z0-9_*][A-Za-z0-9_/*-]*)")

#: A token has to name something.  ``*.md`` and ``*.json`` appear in every
#: directory walk in the tree and would make each tool that globs the docs
#: look like a reader of every record, so a token qualifies only when its own
#: text carries a run of at least three name characters.
NAMED_ENOUGH = re.compile(r"[A-Za-z0-9_]{3,}")


def artifacts() -> list[str]:
    """Every committed record under ``benchmarks/``, repository-relative and sorted."""
    found = [
        path.relative_to(REPO).as_posix()
        for path in BENCHMARKS.rglob("*")
        if path.is_file() and path.suffix in ARTIFACT_SUFFIXES
    ]
    return sorted(rel for rel in found if rel != INDEX.relative_to(REPO).as_posix())


def group_of(rel: str) -> str | None:
    """The grouped directory that owns ``rel``, or ``None`` when it stands alone."""
    for group in GROUPS:
        if rel.startswith(group + "/"):
            return group
    return None


def generator_of(rel: str) -> str:
    """The script that writes ``rel``; ``""`` when the record is hand-made.

    Raises when no pattern matches, which is the signal that a new artifact
    arrived without anyone saying where it came from.
    """
    for pattern, script in GENERATORS:
        if fnmatch.fnmatch(rel, pattern):
            return script
    raise KeyError(f"{rel} matches no pattern in GENERATORS (tools/render_benchmark_index.py)")


def _walk(node: object):
    """Yield every ``(key, value)`` pair in a nested JSON document."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield key, value
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


def provenance_of(rel: str) -> tuple[str, str]:
    """The ``(commit, date)`` this record was measured at; ``""`` where unrecorded.

    Provenance moved between the top level, ``provenance`` and ``_provenance``
    as the schemas evolved, and the date is spelled five ways, so both are
    looked up by key rather than by a fixed path.  Only the containers are
    searched for the commit -- a commit recorded for an embedded external
    source is not this record's own.
    """
    path = REPO / rel
    if path.suffix != ".json":
        return "", ""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError):
        return "", ""
    if not isinstance(document, dict):
        return "", ""

    commit = ""
    for container in PROVENANCE_KEYS:
        block = document if container is None else document.get(container)
        if not isinstance(block, dict):
            continue
        for key in COMMIT_KEYS:
            value = block.get(key)
            if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{7,40}", value):
                commit = value
                break
        if commit:
            break

    date = ""
    for key in DATE_KEYS:
        for found, value in _walk(document):
            if found == key and isinstance(value, str):
                match = re.match(r"(\d{4}-\d{2}-\d{2})", value)
                if match:
                    date = match.group(1)
                    break
        if date:
            break
    return commit[:8], date


def _named(token: str) -> bool:
    """Whether a token's own last component names something specific."""
    return bool(NAMED_ENOUGH.search(token.rsplit("/", 1)[-1].rsplit(".", 1)[0]))


def _tokens(text: str) -> tuple[set[str], set[str]]:
    """File-like and directory-like path tokens appearing in ``text``."""
    # A directory token is qualified by its whole path (``benchmarks/baselines/m4``
    # names a place even though its last segment is two characters long).
    return ({m.group(1) for m in FILE_TOKEN.finditer(text) if _named(m.group(1))},
            {m.group(1).rstrip("/") for m in DIR_TOKEN.finditer(text)
             if any(_named(part) for part in m.group(1).split("/")[1:])})


def _citing_files() -> list[tuple[str, set[str], set[str]]]:
    """Every scannable file with the path tokens it mentions."""
    scanned: list[tuple[str, set[str], set[str]]] = []
    seen: set[str] = set()
    for root in CITING_ROOTS:
        base = REPO / root
        candidates = [base] if base.is_file() else sorted(base.rglob("*"))
        for path in candidates:
            if not path.is_file() or path.suffix not in CITING_SUFFIXES:
                continue
            if "_build" in path.parts or ".git" in path.parts:
                continue
            rel = path.relative_to(REPO).as_posix()
            # The index and its generator name every artifact by construction;
            # counting them as readers would make nothing look orphaned.
            if rel in seen or rel in (SELF, INDEX.relative_to(REPO).as_posix()):
                continue
            seen.add(rel)
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            files, dirs = _tokens(text)
            scanned.append((rel, files, dirs))
    return sorted(scanned)


def _cites(rel: str, files: set[str], dirs: set[str]) -> bool:
    """Whether the token sets name ``rel`` by path, by base name or by directory."""
    name = rel.rsplit("/", 1)[-1]
    for token in files:
        if "/" in token:
            if fnmatch.fnmatch(rel, token) or fnmatch.fnmatch(rel, "*/" + token):
                return True
        elif fnmatch.fnmatch(name, token):
            return True
    return any(rel.startswith(token + "/") or fnmatch.fnmatch(rel, token + "/*") for token in dirs)


def citations(records: list[str]) -> dict[str, list[str]]:
    """Map each record (grouped directories included) to the files that cite it.

    A record's own generator is not a citation: a writer naming its default
    output would otherwise make every artifact look consumed.
    """
    keys = sorted({group_of(rel) or rel for rel in records})
    members: dict[str, list[str]] = {key: [] for key in keys}
    for rel in records:
        members[group_of(rel) or rel].append(rel)
    scanned = _citing_files()

    result: dict[str, list[str]] = {}
    for key in keys:
        owned = set(members[key])
        writers = {generator_of(rel) for rel in members[key]} - {""}
        citing = [
            source for source, files, dirs in scanned
            if source not in owned and source not in writers
            and any(_cites(rel, files, dirs) for rel in members[key])
        ]
        result[key] = citing
    return result


def _ident_order(ident: str) -> tuple[str, int]:
    """Sort workflow identifiers as ``F2`` before ``F10``, not after it."""
    letters = ident.rstrip("0123456789")
    return letters, int(ident[len(letters):])


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _cited_cell(sources: list[str]) -> str:
    return ", ".join(f"`{source}`" for source in sources) if sources else "*nothing*"


def render() -> str:
    """The whole of ``benchmarks/INDEX.md``."""
    records = artifacts()
    cited = citations(records)
    singles = [rel for rel in records if group_of(rel) is None]
    grouped = sorted({group for group in (group_of(rel) for rel in records) if group})

    uncited = sorted(key for key, sources in cited.items() if not sources)
    total_cited = len(cited) - len(uncited)

    lines = [
        "# Benchmark artifact index",
        "",
        "Generated by `tools/render_benchmark_index.py`; do not edit by hand. Run",
        "that script after adding, moving or re-measuring anything under",
        "`benchmarks/`. `tests/test_cited_paths.py` fails when an artifact is",
        "missing from this file, when a path named here does not exist, and when a",
        "grouped directory holds a different number of files than the count below.",
        "",
        *textwrap.wrap(
            f"{len(records)} committed artifacts: {len(singles)} standalone records "
            f"and {len(grouped)} grouped {'directory' if len(grouped) == 1 else 'directories'} "
            f"holding {len(records) - len(singles)} files. {total_cited} of the "
            f"{len(cited)} entries below are cited by a page, a test or another "
            f"script; {len(uncited)} are cited by nothing.",
            width=78,
        ),
        "",
        "`commit` is the revision recorded inside the artifact (short form) and",
        "`date` its recorded measurement date; both are blank where the schema of",
        "the day did not record them. `generator` is blank for records written by",
        "hand rather than by a script in this tree.",
        "",
        "## Records",
        "",
        _row(["artifact", "generator", "commit", "date", "cited by"]),
        _row(["---"] * 5),
    ]
    for rel in singles:
        generator = generator_of(rel)
        commit, date = provenance_of(rel)
        lines.append(_row([
            f"`{rel}`",
            f"`{generator}`" if generator else "",
            f"`{commit}`" if commit else "",
            date,
            _cited_cell(cited[rel]),
        ]))

    lines += [
        "",
        "## Grouped directories",
        "",
        _row(["directory", "files", "generator", "commits", "dates", "cited by"]),
        _row(["---"] * 6),
    ]
    for group in grouped:
        members = [rel for rel in records if group_of(rel) == group]
        provenances = [provenance_of(rel) for rel in members]
        commits = sorted({commit for commit, _ in provenances if commit})
        dates = sorted({date for _, date in provenances if date})
        span = dates[0] if len(dates) == 1 else f"{dates[0]} to {dates[-1]}" if dates else ""
        lines.append(_row([
            f"`{group}/`",
            str(len(members)),
            f"`{generator_of(members[0])}`",
            ", ".join(f"`{commit}`" for commit in commits),
            span,
            _cited_cell(cited[group]),
        ]))
        idents, regimes, others = set(), set(), []
        for rel in members:
            name = rel.rsplit("/", 1)[-1]
            match = re.fullmatch(r"([A-Z]+\d+)_([a-z_]+)\.json", name)
            if match:
                idents.add(match.group(1))
                regimes.add(match.group(2))
            else:
                others.append(name)
        lines += [
            "",
            f"Workflows in `{group}/`: {', '.join(sorted(idents, key=_ident_order))} "
            f"in the regimes {', '.join(sorted(regimes))}"
            + (f", plus {', '.join(f'`{group}/{name}`' for name in sorted(others))}." if others else "."),
        ]

    lines += ["", "## Cited by nothing", ""]
    if uncited:
        lines.append("These records are committed evidence that no page, test or script")
        lines.append("reads. They are kept, not deleted; the list exists so that stays visible.")
        lines.append("")
        for key in uncited:
            lines.append(f"- `{key}`")
    else:
        lines.append("Every committed artifact is cited.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    """Rewrite the index, or report that it is stale."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="exit 1 when the index is stale instead of rewriting")
    args = parser.parse_args()

    new = render()
    old = INDEX.read_text(encoding="utf-8") if INDEX.exists() else ""
    if args.check:
        if new != old:
            print("benchmarks/INDEX.md is stale; run python tools/render_benchmark_index.py", file=sys.stderr)
            return 1
        print("benchmark index is current")
        return 0
    if new != old:
        INDEX.write_text(new, encoding="utf-8")
        print(f"rewrote {INDEX.relative_to(REPO)}")
    else:
        print("benchmark index already current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
