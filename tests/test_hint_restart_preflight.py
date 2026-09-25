from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat
import subprocess
import sys

import pytest


pytest.importorskip("f90nml")
SCRIPT = Path(__file__).parents[1] / "handoff" / "hint-qa" / "prepare-restart.py"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "source"
    source.mkdir()
    deck = tmp_path / "control.input"
    deck.write_text(
        "&nlinp1 run_mode='follow', flx_type='file' /\n"
        "&fopen file_type='netcdf', vac_file='vacuum.nc', flx_file='flux.nc', "
        "limiter_file='limiter.nc', mag_file='hint.nc' /\n"
    )
    for name in ("vacuum.nc", "flux.nc", "limiter.nc", "hint.nc"):
        (source / name).write_bytes((name + "\n").encode())
    (source / "hint.nc").chmod(0o400)
    files = {deck.name: deck, **{name: source / name for name in ("vacuum.nc", "flux.nc", "limiter.nc", "hint.nc")}}
    manifest = tmp_path / "SHA256.json"
    manifest.write_text(json.dumps({"files": {name: {"sha256": digest(path), "bytes": path.stat().st_size} for name, path in files.items()}}))
    return deck, source, manifest


def run_preflight(deck: Path, source: Path, manifest: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--deck", str(deck), "--source-dir", str(source),
         "--sha256-manifest", str(manifest), "--output", str(output)],
        text=True, capture_output=True, check=False,
    )


def test_stages_complete_inventory_and_writable_restart_without_mutating_source(tmp_path: Path) -> None:
    deck, source, manifest = fixture(tmp_path)
    retained = [deck, *source.iterdir()]
    before = {
        path: (digest(path), stat.S_IMODE(path.stat().st_mode)) for path in retained
    }
    output = tmp_path / "staged"

    result = run_preflight(deck, source, manifest, output)

    assert result.returncode == 0, result.stderr
    record = json.loads((output / "restart-preflight.json").read_text())
    assert {item["name"] for item in record["field_inventory"]} == {
        "hint.nc", "vacuum.nc", "flux.nc", "limiter.nc"
    }
    assert record["restart_copy"]["r+b_open_verified"] is True
    assert stat.S_IMODE((output / "hint.nc").stat().st_mode) == 0o600
    assert digest(output / "hint.nc") == before[source / "hint.nc"][0]
    assert {
        path: (digest(path), stat.S_IMODE(path.stat().st_mode)) for path in retained
    } == before
    assert before[source / "hint.nc"][1] == 0o400


@pytest.mark.parametrize("failure", ["missing", "hash", "existing_output"])
def test_refuses_incomplete_or_ambiguous_staging(tmp_path: Path, failure: str) -> None:
    deck, source, manifest = fixture(tmp_path)
    output = tmp_path / "staged"
    if failure == "missing":
        (source / "flux.nc").unlink()
    elif failure == "hash":
        (source / "flux.nc").write_bytes(b"changed")
    else:
        output.mkdir()

    result = run_preflight(deck, source, manifest, output)

    assert result.returncode != 0
    if failure != "existing_output":
        assert not output.exists()


@pytest.mark.parametrize(
    ("replacement", "expected"),
    [
        ("mag_file='control.input'", "must have distinct names"),
        ("mag_file='restart-preflight.json'", "invalid local FOPEN filename"),
        ("mag_file='../hint.nc'", "invalid local FOPEN filename"),
        ("mag_file=7", "must be a nonempty string"),
        ("file_type='binary', mag_file='hint.nc'", "file_type='netcdf'"),
    ],
)
def test_refuses_colliding_or_malformed_names(
    tmp_path: Path, replacement: str, expected: str
) -> None:
    deck, source, manifest = fixture(tmp_path)
    deck.write_text(deck.read_text().replace("mag_file='hint.nc'", replacement))
    output = tmp_path / "staged"

    result = run_preflight(deck, source, manifest, output)

    assert result.returncode != 0
    assert expected in result.stderr
    assert not output.exists()


def test_refuses_deck_collision_with_generated_record(tmp_path: Path) -> None:
    deck, source, manifest = fixture(tmp_path)
    colliding_deck = deck.with_name("restart-preflight.json")
    deck.rename(colliding_deck)
    output = tmp_path / "staged"

    result = run_preflight(colliding_deck, source, manifest, output)

    assert result.returncode != 0
    assert "must have distinct names" in result.stderr
    assert not output.exists()
