#!/usr/bin/env python3
"""One-off history slimming for a bare clone of this repository.

Drops every media/data blob that is not in the tip tree of a kept ref (old
figure versions, deleted benchmark records, old netCDF fixtures), applies the
owner's message/text/path rules, and checks that the tip trees of the kept
refs are unchanged. Code history is kept whole. Nothing is pushed.

    git clone --bare https://github.com/uwplasma/vmex.git vmex.git
    python tools/rewrite_history.py vmex.git --keep main --keep <pr-branch> \
        --message-rules msg.txt --text-rules text.txt --mailmap mailmap.txt \
        --path-rule OLD=NEW

Rule files use git-filter-repo's ``--replace-message``/``--replace-text``
syntax. ``<repo>/filter-repo/commit-map`` maps old to new commit ids.
Requires git-filter-repo. Delete this script once the rewrite has shipped.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

MEDIA = re.compile(
    r"\.(png|webp|gif|jpe?g|mp4|webm|mov|pdf|nc|npz|npy|h5|hdf5|pkl|json|csv|svg)$",
    re.IGNORECASE,
)


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, text=True).stdout


def tip_trees(repo: Path, refs: list[str]) -> dict[str, str]:
    return {ref: git(repo, "rev-parse", f"{ref}^{{tree}}").strip() for ref in refs}


def dead_blobs(repo: Path, refs: list[str]) -> set[str]:
    """Media/data blobs reachable from history but absent from every kept tip."""
    keep = {line.split()[2] for ref in refs
            for line in git(repo, "ls-tree", "-r", ref).splitlines()}
    blobs = {line.split()[0] for line in git(
        repo, "cat-file", "--batch-check=%(objectname) %(objecttype)",
        "--batch-all-objects").splitlines() if line.endswith(" blob")}
    dead = set()
    for line in git(repo, "rev-list", "--objects", "--all").splitlines():
        sha, _, path = line.partition(" ")
        if path and MEDIA.search(path) and sha not in keep:
            dead.add(sha)
    return dead & blobs


def pack_size(repo: Path) -> str:
    git(repo, "gc", "--aggressive", "--prune=now", "-q")
    stats = dict(line.split(": ") for line in git(repo, "count-objects", "-vH").splitlines())
    return stats["size-pack"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("repo", type=Path, help="bare clone to rewrite in place")
    ap.add_argument("--keep", action="append", default=[],
                    help="ref whose tip tree must survive unchanged (repeatable; main is always kept)")
    ap.add_argument("--message-rules", type=Path)
    ap.add_argument("--text-rules", type=Path)
    ap.add_argument("--mailmap", type=Path)
    ap.add_argument("--path-rule", action="append", default=[], metavar="OLD=NEW",
                    help="case-insensitive regex substitution on every historical path")
    args = ap.parse_args()
    repo, refs = args.repo.resolve(), ["main", *args.keep]

    before_trees, before = tip_trees(repo, refs), pack_size(repo)
    strip = repo / "strip-blobs.txt"
    dead = dead_blobs(repo, refs)
    strip.write_text("\n".join(sorted(dead)) + "\n")

    cmd = ["git", "filter-repo", "--force", "--strip-blobs-with-ids", str(strip)]
    for flag, path in (("--replace-message", args.message_rules),
                       ("--replace-text", args.text_rules), ("--mailmap", args.mailmap)):
        if path:
            cmd += [flag, str(path.resolve())]
    if args.path_rule:
        subs = "".join(f"filename = re.sub(rb'(?i){old}', b'{new}', filename)\n"
                       for old, new in (rule.split("=", 1) for rule in args.path_rule))
        cmd += ["--filename-callback", subs + "return filename"]
    subprocess.run(cmd, cwd=repo, check=True)

    after_trees = tip_trees(repo, refs)
    changed = [ref for ref in refs if before_trees[ref] != after_trees[ref]]
    print(f"stripped blobs: {len(dead)}; pack {before} -> {pack_size(repo)}")
    if changed:
        raise SystemExit(f"tip trees changed for {changed}: land the tree cleanups first")
    print("kept tip trees identical")


if __name__ == "__main__":
    main()
