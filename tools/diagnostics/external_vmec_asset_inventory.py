"""Summarize optional external VMEC validation assets.

The manifest is metadata-only by design: it points to pinned upstream SIMSOPT
and Landreman VMEC equilibria assets without vendoring large or license-unclear
files.  This helper makes the manifest actionable for local/nightly validation
jobs by filtering assets by physics family and, when a local checkout is
provided, checking that the referenced files are present.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib


DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / "validation" / "external_vmec_asset_manifest.toml"


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _parse_source_root(values: list[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--source-root must be repository_key=/path, got {value!r}")
        key, raw_path = value.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"--source-root repository key cannot be empty: {value!r}")
        roots[key] = Path(raw_path).expanduser().resolve()
    return roots


def _asset_matches(asset: dict[str, Any], *, repository: str | None, family: set[str], recommendation: str | None) -> bool:
    if repository and asset.get("repository") != repository:
        return False
    families = set(asset.get("families", ()))
    if family and not family <= families:
        return False
    if recommendation and str(asset.get("recommendation", "")) != recommendation:
        return False
    return True


def build_inventory(
    manifest: dict[str, Any],
    *,
    source_roots: dict[str, Path] | None = None,
    repository: str | None = None,
    family: set[str] | None = None,
    recommendation: str | None = None,
) -> dict[str, Any]:
    """Return a machine-readable inventory for selected external assets."""

    roots = dict(source_roots or {})
    required_family = set(family or ())
    repositories = manifest.get("repositories", {})
    selected: list[dict[str, Any]] = []
    missing: list[str] = []
    present: list[str] = []

    for asset in manifest.get("assets", ()):
        if not _asset_matches(
            asset,
            repository=repository,
            family=required_family,
            recommendation=recommendation,
        ):
            continue
        item = {
            "id": asset["id"],
            "repository": asset["repository"],
            "path": asset["path"],
            "families": list(asset.get("families", ())),
            "recommendation": asset.get("recommendation"),
            "size_class": asset.get("size_class"),
            "raw_url": asset.get("raw_url"),
            "companion_paths": list(asset.get("companion_paths", ())),
        }
        root = roots.get(str(asset["repository"]))
        if root is not None:
            paths = [Path(asset["path"])] + [Path(p) for p in asset.get("companion_paths", ())]
            path_status = {str(path): (root / path).exists() for path in paths}
            item["local_root"] = str(root)
            item["path_status"] = path_status
            if all(path_status.values()):
                present.append(asset["id"])
            else:
                missing.append(asset["id"])
        selected.append(item)

    family_counts: dict[str, int] = {}
    for item in selected:
        for fam in item["families"]:
            family_counts[fam] = family_counts.get(fam, 0) + 1

    return {
        "manifest": {
            "version": manifest.get("version"),
            "name": manifest.get("name"),
        },
        "repositories": {
            key: {
                "name": repo.get("name"),
                "url": repo.get("url"),
                "ref": repo.get("ref"),
                "license": repo.get("license"),
            }
            for key, repo in repositories.items()
        },
        "selected_count": len(selected),
        "family_counts": dict(sorted(family_counts.items())),
        "present_count": len(present),
        "missing_count": len(missing),
        "present": present,
        "missing": missing,
        "assets": selected,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--repository", choices=("simsopt", "landreman_vmec_equilibria"))
    parser.add_argument(
        "--family",
        action="append",
        default=[],
        help="Require a physics-family tag. Repeat for intersections, e.g. --family stellarator --family free_boundary.",
    )
    parser.add_argument("--recommendation", help="Require an exact recommendation value.")
    parser.add_argument(
        "--source-root",
        action="append",
        default=[],
        help="Optional local checkout for path verification, e.g. simsopt=~/local/simsopt.",
    )
    parser.add_argument("--json-out", type=Path, help="Write inventory JSON to this path.")
    parser.add_argument("--fail-missing", action="store_true", help="Exit nonzero if any selected local paths are missing.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    manifest = _load_manifest(args.manifest)
    inventory = build_inventory(
        manifest,
        source_roots=_parse_source_root(args.source_root),
        repository=args.repository,
        family=set(args.family),
        recommendation=args.recommendation,
    )
    text = json.dumps(inventory, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n")
    if args.fail_missing and inventory["missing_count"]:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
