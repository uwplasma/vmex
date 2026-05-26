#!/usr/bin/env python3
"""Download large example/reference netCDF assets.

Generated optimization figures are intentionally not part of this asset bundle:
rerun the optimization renderers when report-quality panels are needed.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import tarfile
import urllib.request
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TAG = "assets-20260316-nc"
ASSET_NAME = "vmec_jax_assets_20260316_nc_only.tar.gz"
DEFAULT_URL = (
    "https://github.com/uwplasma/vmec_jax/releases/download/"
    f"{DEFAULT_TAG}/{ASSET_NAME}"
)
DEFAULT_SHA256 = "3344fc2401fffed240ee57ae741ec521594c592627c76dae203503f485e4c0d8"

COMMON_ASSET_PATHS = (
    "examples/data/mgrid_cth_like.nc",
    "examples/data/mgrid_d3d_ef.nc",
    "examples/data/wout_*_reference.nc",
    "examples_single_grid/data/mgrid_cth_like.nc",
    "examples_single_grid/data/mgrid_d3d_ef.nc",
    "examples_single_grid/data/wout_*_reference.nc",
)


def _sha256(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _safe_extract(tf: tarfile.TarFile, dest: Path) -> None:
    """Extract ``tf`` under ``dest`` without allowing path traversal."""
    dest_resolved = dest.resolve()
    for member in tf.getmembers():
        target = (dest_resolved / member.name).resolve()
        if target != dest_resolved and dest_resolved not in target.parents:
            raise SystemExit(f"Refusing to extract path outside destination: {member.name}")
    tf.extractall(dest_resolved)


def _print_bundle_info(url: str, sha256: str) -> None:
    print(f"Asset bundle URL: {url}")
    print(f"Expected SHA256:  {sha256 or '(not checked)'}")
    print("Common installed paths:")
    for path in COMMON_ASSET_PATHS:
        print(f"  {path}")


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", type=str, default=DEFAULT_URL, help="Asset tarball URL.")
    p.add_argument("--sha256", type=str, default=DEFAULT_SHA256, help="Expected SHA256.")
    p.add_argument("--dest", type=str, default=str(REPO_ROOT), help="Destination repo root.")
    p.add_argument("--force", action="store_true", help="Re-download even if files already exist.")
    p.add_argument("--list", action="store_true", help="Print the default bundle location and common paths.")
    p.add_argument("--dry-run", action="store_true", help="Print what would be downloaded without fetching it.")
    args = p.parse_args(argv)

    if args.list or args.dry_run:
        _print_bundle_info(args.url, args.sha256)
        if args.dry_run:
            print("Dry run: no files downloaded or extracted.")
        return 0

    dest = Path(args.dest).expanduser().resolve()
    marker = dest / ".assets_installed"
    if marker.exists() and not args.force:
        print(f"Assets already installed at {dest}. Use --force to re-download.")
        return 0

    print(f"Downloading assets from: {args.url}")
    with urllib.request.urlopen(args.url) as resp:
        data = resp.read()

    digest = _sha256(data)
    if args.sha256 and digest != args.sha256:
        raise SystemExit(f"SHA256 mismatch: expected {args.sha256}, got {digest}")

    print("Extracting assets...")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        _safe_extract(tf, dest)

    marker.write_text(f"{args.url}\n{digest}\n")
    print("Assets installed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
