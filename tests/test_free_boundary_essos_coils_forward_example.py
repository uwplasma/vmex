from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "examples" / "free_boundary_essos_coils_forward.py"


def _load_forward_module():
    spec = importlib.util.spec_from_file_location("free_boundary_essos_coils_forward_example", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_forward_direct_coil_example_imports_without_essos_import_side_effects():
    module = _load_forward_module()

    assert module.DEFAULT_OUTDIR.name == "free_boundary_essos_coils_forward"
    assert callable(module.main)
    assert callable(module._summarize_run)
