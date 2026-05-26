from __future__ import annotations

import importlib.util
import io
from pathlib import Path
import sys
import tarfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "fetch_assets.py"


def _load_fetch_assets():
    spec = importlib.util.spec_from_file_location("fetch_assets", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_fetch_assets_dry_run_lists_bundle(capsys) -> None:
    module = _load_fetch_assets()

    assert module.main(["--dry-run"]) == 0

    out = capsys.readouterr().out
    assert "Asset bundles:" in out
    assert "reference-nc" in out
    assert "wout-fixtures" in out
    assert "Expected SHA256:" in out
    assert "examples/data/mgrid_cth_like.nc" in out
    assert "examples/data/wout_*.nc" in out
    assert "Dry run: no files downloaded or extracted." in out


def test_fetch_assets_can_select_one_default_bundle(capsys) -> None:
    module = _load_fetch_assets()

    assert module.main(["--bundle", "wout-fixtures", "--dry-run"]) == 0

    out = capsys.readouterr().out
    assert "wout-fixtures" in out
    assert "reference-nc" not in out
    assert "docs/_static/readme_best_cases/*/wout_*.nc" in out


def test_fetch_assets_safe_extract_rejects_path_traversal(tmp_path) -> None:
    module = _load_fetch_assets()
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as tf:
        data = b"bad"
        info = tarfile.TarInfo("../outside.txt")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    payload.seek(0)

    with tarfile.open(fileobj=payload, mode="r:gz") as tf:
        with pytest.raises(SystemExit, match="outside destination"):
            module._safe_extract(tf, tmp_path)

    assert not (tmp_path.parent / "outside.txt").exists()
