from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "examples" / "free_boundary_essos_coils_forward.py"
BETA_SCAN_SCRIPT_PATH = ROOT / "examples" / "free_boundary_essos_coils_beta_scan.py"


def _load_forward_module():
    spec = importlib.util.spec_from_file_location("free_boundary_essos_coils_forward_example", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_beta_scan_module():
    spec = importlib.util.spec_from_file_location("free_boundary_essos_coils_beta_scan", BETA_SCAN_SCRIPT_PATH)
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


def test_beta_scan_pressure_continuation_helpers_convert_wout_boundary(tmp_path):
    module = _load_beta_scan_module()
    base_indata = module.read_indata(ROOT / "examples" / "data" / "input.LandremanPaul2021_QA_lowres")
    base_indata.scalars["MPOL"] = 5
    base_indata.scalars["NTOR"] = 5
    wout = SimpleNamespace(
        nfp=2,
        lasym=False,
        xm=np.array([0, 1, 1, 6]),
        xn=np.array([0.0, -2.0, 2.0, 0.0]),
        rmnc=np.array([[1.2, 0.1, 0.2, 9.0]]),
        zmns=np.array([[0.0, -0.01, 0.02, 9.0]]),
        raxis_cc=np.array([1.1, 0.01]),
        zaxis_cs=np.array([0.0, 0.02]),
    )

    warmed = module.continue_indata_from_wout_boundary(base_indata, wout)

    assert module._vmec_input_n_from_wout_xn(4.0, nfp=2) == 2
    assert warmed.indexed["RBC"][(0, 0)] == 1.2
    assert warmed.indexed["RBC"][(-1, 1)] == 0.1
    assert warmed.indexed["RBC"][(1, 1)] == 0.2
    assert (0, 6) not in warmed.indexed["RBC"]
    assert warmed.indexed["ZBS"][(-1, 1)] == -0.01
    assert warmed.scalars["RAXIS_CC"] == [1.1, 0.01]
    assert warmed.scalars["ZAXIS_CS"] == [0.0, 0.02]

    wout_path = tmp_path / "wout_ok.nc"
    wout_path.write_text("placeholder")
    ok_summary = {"fsqr": 1.0e-8, "fsqz": 2.0e-8, "fsql": 3.0e-8, "wout": str(wout_path)}
    bad_summary = {"fsqr": 1.0e-3, "fsqz": 0.0, "fsql": 0.0, "wout": str(wout_path)}
    assert module._summary_is_promotable_for_pressure_continuation(ok_summary, max_fsq=1.0e-6)
    assert not module._summary_is_promotable_for_pressure_continuation(bad_summary, max_fsq=1.0e-6)
