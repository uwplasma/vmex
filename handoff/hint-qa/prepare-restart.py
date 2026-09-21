#!/usr/bin/env python3
"""Stage a hash-verified, writable HINT follow-mode restart directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile

import f90nml


FIELD_KEYS = ("mag_file", "vac_file", "flx_file", "limiter_file")
RECORD_NAME = "restart-preflight.json"


def sha256(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_expected(path: Path, names: set[str]) -> dict[str, dict[str, object]]:
    """Load an exact basename-to-checksum contract."""
    document = json.loads(path.read_text())
    records = document.get("files") if isinstance(document, dict) else None
    if not isinstance(records, dict) or set(records) != names:
        raise ValueError("checksum manifest files must exactly match the deck and its four FOPEN files")
    expected: dict[str, dict[str, object]] = {}
    for name, record in records.items():
        if isinstance(record, str):
            record = {"sha256": record}
        if not isinstance(record, dict) or not re.fullmatch(r"[0-9a-fA-F]{64}", str(record.get("sha256", ""))):
            raise ValueError(f"invalid SHA-256 record for {name}")
        expected[name] = {"sha256": str(record["sha256"]).lower()}
        if "bytes" in record:
            expected[name]["bytes"] = int(record["bytes"])
    return expected


def checked_source(path: Path, record: dict[str, object]) -> dict[str, object]:
    """Verify one retained input without changing it."""
    if not path.is_file() or not os.access(path, os.R_OK):
        raise FileNotFoundError(f"missing or unreadable input: {path.name}")
    actual = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    if actual["sha256"] != record["sha256"] or (
        "bytes" in record and actual["bytes"] != record["bytes"]
    ):
        raise ValueError(f"input checksum mismatch: {path.name}")
    return {**actual, "mode": stat.S_IMODE(path.stat().st_mode)}


def local_name(value: object, key: str) -> str:
    """Return one unambiguous local filename from an FOPEN member."""
    if not isinstance(value, str) or value != value.strip():
        raise ValueError(f"{key} must be a nonempty string without surrounding whitespace")
    if (
        not value
        or value in {".", "..", RECORD_NAME}
        or Path(value).name != value
        or "/" in value
        or "\\" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError(f"invalid local FOPEN filename for {key}")
    return value


def main() -> None:
    """Stage and validate one isolated follow-mode restart directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", type=Path, required=True, help="follow-mode HINT input deck")
    parser.add_argument("--source-dir", type=Path, required=True, help="directory containing all four FOPEN files")
    parser.add_argument("--sha256-manifest", type=Path, required=True, help="exact checksum contract for the deck and FOPEN files")
    parser.add_argument("--output", type=Path, required=True, help="new staging directory; must not exist")
    args = parser.parse_args()

    if args.output.exists():
        parser.error("--output already exists; refusing to overwrite or merge results")
    deck = args.deck.resolve()
    source_dir = args.source_dir.resolve()
    if not deck.is_file() or not source_dir.is_dir():
        parser.error("--deck must be a file and --source-dir must be a directory")

    parsed = f90nml.read(deck)
    try:
        controls = parsed["nlinp1"]
        opened = parsed["fopen"]
        run_mode = str(controls["run_mode"]).strip().lower()
        flx_type = str(controls["flx_type"]).strip().lower()
        file_type = str(opened["file_type"]).strip().lower()
        field_names = {key: local_name(opened[key], key) for key in FIELD_KEYS}
    except (KeyError, TypeError) as error:
        parser.error(f"deck lacks a required restart control or FOPEN member: {error}")
    except ValueError as error:
        parser.error(str(error))
    if run_mode != "follow" or flx_type != "file" or file_type != "netcdf":
        parser.error(
            "restart staging requires run_mode='follow', flx_type='file', "
            "and file_type='netcdf'"
        )
    if len(set(field_names.values())) != len(field_names):
        parser.error("FOPEN filenames must be distinct")
    if deck.name in {*field_names.values(), RECORD_NAME}:
        parser.error("the deck, FOPEN files, and generated preflight record must have distinct names")

    names = {deck.name, *field_names.values()}
    try:
        expected = load_expected(args.sha256_manifest.resolve(), names)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    sources = {deck.name: deck}
    sources.update({name: (source_dir / name).resolve() for name in field_names.values()})
    try:
        source_records = {name: checked_source(path, expected[name]) for name, path in sources.items()}
    except (OSError, ValueError) as error:
        parser.error(str(error))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=args.output.name + ".tmp-", dir=args.output.parent))
    try:
        for name, source in sources.items():
            shutil.copyfile(source, temporary / name)
            os.chmod(temporary / name, 0o400)

        restart_name = field_names["mag_file"]
        restart = temporary / restart_name
        os.chmod(restart, 0o600)
        if not restart.is_file() or restart.is_symlink():
            raise ValueError("staged restart must be a regular file")
        mode = stat.S_IMODE(restart.stat().st_mode)
        if mode != 0o600 or not os.access(restart, os.R_OK | os.W_OK):
            raise PermissionError("staged restart is not owner-readable and owner-writable")
        with restart.open("r+b"):
            pass

        staged = {}
        for name in sorted(names):
            path = temporary / name
            staged[name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "mode": format(stat.S_IMODE(path.stat().st_mode), "04o"),
            }
            if staged[name]["sha256"] != expected[name]["sha256"]:
                raise ValueError(f"staged checksum mismatch: {name}")
        for name, source in sources.items():
            current = checked_source(source, expected[name])
            if current != source_records[name]:
                raise ValueError(f"retained input changed while staging: {name}")

        access = {
            field_names["mag_file"]: "read_write",
            field_names["limiter_file"]: "read",
            field_names["flx_file"]: "read",
            field_names["vac_file"]: "named_not_read_in_follow_mode",
        }
        result = {
            "schema": 1,
            "status": "restart_preflight_complete_no_simulation",
            "deck": {"name": deck.name, "sha256": staged[deck.name]["sha256"]},
            "controls": {
                "run_mode": run_mode,
                "flx_type": flx_type,
                "file_type": file_type,
            },
            "field_inventory": [
                {"name": name, "access": access[name], **staged[name]}
                for name in sorted(field_names.values())
            ],
            "retained_sources_unchanged": True,
            "restart_copy": {
                "name": restart_name,
                "regular_file": True,
                "owner_read_write": True,
                "r+b_open_verified": True,
                "sha256": staged[restart_name]["sha256"],
            },
        }
        (temporary / RECORD_NAME).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        if args.output.exists():
            raise FileExistsError("--output appeared during staging; refusing to overwrite it")
        temporary.rename(args.output)
        print(json.dumps({"output": args.output.name, "status": result["status"], "restart": restart_name}, sort_keys=True))
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
