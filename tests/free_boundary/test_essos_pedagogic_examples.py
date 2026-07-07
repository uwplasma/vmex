from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from conftest import circular_coil_params
from vmec_jax.external_fields import CoilFieldParams


ROOT = Path(__file__).resolve().parents[2]
MGRID_SCRIPT = ROOT / "examples" / "free_boundary_essos_mgrid_forward.py"
DIRECT_SCRIPT = ROOT / "examples" / "free_boundary_essos_direct_forward.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakeEssosCoils:
    nfp = 2
    n_segments = 12
    stellsym = True
    currents_scale = 1.0

    def __init__(self) -> None:
        self.to_mgrid_calls: list[dict[str, object]] = []

    def to_mgrid(self, path, **kwargs) -> None:
        self.to_mgrid_calls.append({"path": Path(path), **kwargs})
        Path(path).write_text("fake mgrid placeholder\n")


def _simple_coil_params() -> CoilFieldParams:
    return circular_coil_params(
        current=1.0e6,
        radius=1.5,
        n_segments=12,
        nfp=2,
        stellsym=True,
        chunk_size=64,
    )


def test_pedagogic_essos_mgrid_example_dry_run(monkeypatch, tmp_path: Path) -> None:
    module = _load_module(MGRID_SCRIPT, "free_boundary_essos_mgrid_forward_example")
    fake_coils = _FakeEssosCoils()
    monkeypatch.setattr(module, "load_essos_coils", lambda _path=None: fake_coils)

    rc = module.main(["--dry-run", "--outdir", str(tmp_path), "--max-iter", "1", "--mgrid-nr", "4", "--mgrid-nz", "4"])

    assert rc == 0
    assert fake_coils.to_mgrid_calls
    call = fake_coils.to_mgrid_calls[0]
    assert call["path"] == tmp_path / "mgrid_lpqa_from_essos.nc"
    assert int(call["nfp"]) == 2

    input_text = (tmp_path / "input.lpqa_mgrid").read_text()
    assert "LFREEB = .TRUE." in input_text
    assert "MGRID_FILE = 'mgrid_lpqa_from_essos.nc'" in input_text
    assert "NITER_ARRAY = 1" in input_text

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["backend"] == "mgrid"
    assert summary["flow"] == "essos_generated_mgrid_compatibility"
    assert summary["workflow"]["field_backend"] == "mgrid"
    assert summary["workflow"]["python_provider_required"] is False
    assert summary["workflow"]["uses_mgrid_file"] is True
    assert summary["workflow"]["direct_coil_example"].endswith("free_boundary_essos_direct_forward.py")
    assert summary["dry_run"] is True
    assert summary["surface_dofs_optimized"] is False
    assert Path(summary["mgrid"]).name == "mgrid_lpqa_from_essos.nc"
    assert summary["external_field_provider_kind"] == "mgrid"
    assert Path(summary["mgrid_file"]).name == "mgrid_lpqa_from_essos.nc"
    assert summary["uses_generated_mgrid"] is True
    assert summary["mgrid_bounds"]["rmin"] < summary["mgrid_bounds"]["rmax"]
    assert summary["mgrid_bounds"]["zmin"] < summary["mgrid_bounds"]["zmax"]


def test_pedagogic_essos_direct_example_dry_run(monkeypatch, tmp_path: Path) -> None:
    module = _load_module(DIRECT_SCRIPT, "free_boundary_essos_direct_forward_example")
    monkeypatch.setattr(module, "load_essos_coils", lambda _path=None: _FakeEssosCoils())
    monkeypatch.setattr(module, "from_essos_coils", lambda _coils, chunk_size=None: _simple_coil_params())

    rc = module.main(["--dry-run", "--outdir", str(tmp_path), "--max-iter", "1", "--chunk-size", "64"])

    assert rc == 0
    assert module.DEFAULT_INPUT.name == "input.LandremanPaul2021_QA_lowres"
    input_text = (tmp_path / "input.lpqa_direct_coils").read_text()
    assert "LFREEB = .TRUE." in input_text
    assert "MGRID_FILE = 'DIRECT_COILS'" in input_text
    assert "NITER_ARRAY = 1" in input_text

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["backend"] == "direct_essos_coils"
    assert summary["flow"] == "direct_essos_coils_no_mgrid"
    assert summary["workflow"]["field_backend"] == "direct_coils"
    assert summary["workflow"]["python_provider_required"] is True
    assert summary["workflow"]["uses_mgrid_file"] is False
    assert summary["workflow"]["mgrid_compatibility_example"].endswith("free_boundary_essos_mgrid_forward.py")
    assert "Python-provider tag" in summary["workflow"]["vmec_input_replay"]
    assert summary["dry_run"] is True
    assert summary["surface_dofs_optimized"] is False
    assert summary["mgrid"] is None
    assert summary["external_field_provider_kind"] == "direct_coils"
    assert summary["mgrid_file"] == "DIRECT_COILS"
    assert summary["uses_generated_mgrid"] is False
    assert np.isfinite(float(summary["coil_current_norm"]))
    assert np.isfinite(float(summary["coil_length_mean"]))
    assert not list(tmp_path.glob("mgrid*"))
