from __future__ import annotations

import importlib.util
import json
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


def test_beta_scan_summary_checkpoint_preserves_partial_runs(tmp_path):
    module = _load_beta_scan_module()
    args = SimpleNamespace(
        coil_current_scale=1.0,
        phiedge=-0.025,
        pressure_scale_for_one_percent_beta=1000.0,
        pressure_continuation=True,
        pressure_continuation_max_fsq=1.0e-6,
        disable_direct_coil_source_reuse=False,
        direct_coil_trial_resample=False,
        direct_coil_limit_update_rms=False,
        ns=12,
        max_iter=20,
        ftol=1.0e-8,
    )
    summary_path = tmp_path / "summary.json"
    runs = [
        {
            "backend": "direct",
            "nominal_beta_percent": 0.0,
            "fsqr": 1.0e-12,
            "fsqz": 2.0e-12,
            "fsql": 3.0e-12,
        }
    ]

    module._write_summary_checkpoint(
        summary_path,
        coils_json=tmp_path / "coils.json",
        mgrid_file=tmp_path / "mgrid.nc",
        args=args,
        scale_summary={"coil_r_mean": 1.0},
        ns_array=[16, 101],
        niter_array=[600, 3000],
        ftol_array=[1.0e-8, 1.0e-12],
        summaries=runs,
        complete=False,
    )
    partial = json.loads(summary_path.read_text())
    assert partial["complete"] is False
    assert partial["runs"] == runs
    assert partial["ns_array"] == [16, 101]

    module._write_summary_checkpoint(
        summary_path,
        coils_json=tmp_path / "coils.json",
        mgrid_file=tmp_path / "mgrid.nc",
        args=args,
        scale_summary={"coil_r_mean": 1.0},
        ns_array=[16, 101],
        niter_array=[600, 3000],
        ftol_array=[1.0e-8, 1.0e-12],
        summaries=runs,
        complete=True,
    )
    complete = json.loads(summary_path.read_text())
    assert complete["complete"] is True


def test_beta_scan_resume_existing_case_uses_wout_path(monkeypatch, tmp_path):
    module = _load_beta_scan_module()
    assert (
        module._resume_existing_case(
            output_dir=tmp_path,
            backend="direct",
            beta_percent=0.5,
            pressure_scale_for_one_percent_beta=1000.0,
        )
        is None
    )

    wout_path = tmp_path / "wout_direct_beta_0.500.nc"
    wout_path.write_text("placeholder")

    def fake_summarize_existing_wout(path, *, backend, beta_percent):
        return {
            "wout": str(path),
            "backend": backend,
            "nominal_beta_percent": float(beta_percent),
            "fsqr": 1.0e-12,
            "fsqz": 2.0e-12,
            "fsql": 3.0e-12,
        }

    monkeypatch.setattr(module, "summarize_existing_wout", fake_summarize_existing_wout)
    summary = module._resume_existing_case(
        output_dir=tmp_path,
        backend="direct",
        beta_percent=0.5,
        pressure_scale_for_one_percent_beta=1200.0,
    )
    assert summary["wout"] == str(wout_path)
    assert summary["pressure_scale"] == 600.0
