from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from vmec_jax.external_fields import CoilFieldParams


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "examples" / "free_boundary_direct_coils_forward.py"


def _load_forward_module():
    spec = importlib.util.spec_from_file_location("free_boundary_direct_coils_forward_example", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_direct_coil_forward_example_imports_and_dry_runs_without_essos(tmp_path):
    module = _load_forward_module()

    params = module.make_circular_coil_params(current=1.0, radius=1.2, n_segments=12)
    assert isinstance(params, CoilFieldParams)
    assert int(params.n_segments) == 12
    np.testing.assert_allclose(np.asarray(params.base_currents), [1.0])

    rc = module.main(
        [
            "--dry-run",
            "--outdir",
            str(tmp_path),
            "--max-iter",
            "1",
            "--n-segments",
            "12",
        ]
    )

    assert rc == 0
    input_text = (tmp_path / "input.direct_coils").read_text()
    assert "MGRID_FILE = 'DIRECT_COILS'" in input_text
    assert "NITER_ARRAY = 1" in input_text

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["backend"] == "direct_coils"
    assert summary["dry_run"] is True
    assert summary["wout"] is None
    assert summary["coil"]["n_base_coils"] == 1
    assert summary["coil"]["n_segments"] == 12
    assert np.isfinite(float(summary["coil"]["sample_R1_Z0_phi0"]["bz"]))


def test_direct_coil_forward_example_wires_direct_provider_without_running_solver(monkeypatch, tmp_path):
    module = _load_forward_module()
    captured = {}

    def _fake_run_free_boundary(input_path, **kwargs):
        captured["input_path"] = Path(input_path)
        captured["kwargs"] = dict(kwargs)
        return SimpleNamespace(
            state=None,
            static=None,
            indata=None,
            signgs=1,
            result=SimpleNamespace(
                n_iter=0,
                diagnostics={
                    "final_fsqr": 1.0,
                    "final_fsqz": 2.0,
                    "final_fsql": 3.0,
                    "free_boundary": {
                        "vacuum_stub": False,
                        "nestor_model": "test",
                        "last_nestor_diagnostics": {"bnormal_rms": 4.0, "bsqvac_rms": 5.0},
                    },
                },
            ),
        )

    def _fake_write_wout(path, _run, *, include_fsq):
        captured["wout"] = (Path(path), bool(include_fsq))
        Path(path).write_text("placeholder wout\n")

    monkeypatch.setattr(module, "run_free_boundary", _fake_run_free_boundary)
    monkeypatch.setattr(module, "write_wout_from_fixed_boundary_run", _fake_write_wout)

    rc = module.main(
        [
            "--outdir",
            str(tmp_path),
            "--max-iter",
            "1",
            "--ns",
            "7",
            "--n-segments",
            "8",
            "--coil-current",
            "2.5",
        ]
    )

    assert rc == 0
    assert captured["input_path"] == tmp_path / "input.direct_coils"
    assert captured["kwargs"]["max_iter"] == 1
    assert captured["kwargs"]["multigrid"] is False
    assert captured["kwargs"]["jit_forces"] is False
    assert captured["kwargs"]["external_field_provider_kind"] == "direct_coils"
    assert isinstance(captured["kwargs"]["external_field_provider_params"], CoilFieldParams)
    np.testing.assert_allclose(np.asarray(captured["kwargs"]["external_field_provider_params"].base_currents), [2.5])
    assert captured["wout"] == (tmp_path / "wout_direct_coils.nc", True)

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["dry_run"] is False
    assert summary["fsqr"] == 1.0
    assert summary["free_boundary_nestor_model"] == "test"
