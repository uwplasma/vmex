from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "optimization" / "qi_optimization_support.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("qi_optimization_support_checkpoint_test", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_qi_stage_checkpoint_preserves_partial_metrics_gates_and_provenance(tmp_path: Path) -> None:
    mod = _load_module()
    mod.configure({"OUTPUT_DIR": tmp_path})
    stage_dir = tmp_path / "mirror_ramp_01_cleanup"
    stage_result = SimpleNamespace(
        history={
            "objective_initial": 4.0,
            "objective_final": 1.25,
            "qs_initial": 3.0e-3,
            "qs_final": 1.7e-3,
            "aspect_initial": 9.7,
            "aspect_final": 10.1,
            "iota_initial": 0.38,
            "iota_final": 0.49,
            "nfev": 5,
        }
    )
    diagnostics = {
        "qi_smooth_total": 1.7e-3,
        "qi_legacy_total": 1.2e-3,
        "qi_mirror_ratio_max": 0.28,
        "qi_mirror_ratio_target": 0.30,
        "qi_seed_gate_passed": True,
        "qi_engineering_gate_passed": True,
        "qi_gate_failures": [],
    }
    promotion = {
        "qi_cleanup_promoted": True,
        "qi_smooth_total": 1.7e-3,
        "qi_legacy_total": 1.2e-3,
        "qi_mirror_ratio_max": 0.28,
    }

    checkpoint_path = mod.write_qi_stage_checkpoint(
        stage_dir,
        stage_index=1,
        stage_name="cleanup",
        stage_modes=(1, 2, 3),
        stage_result=stage_result,
        diagnostics=diagnostics,
        promotion=promotion,
        role="mirror_ramp",
    )

    checkpoint = json.loads(checkpoint_path.read_text())
    root_checkpoint = json.loads((tmp_path / "stage_checkpoint.json").read_text())
    stage_diagnostics = json.loads((stage_dir / "diagnostics.json").read_text())

    assert root_checkpoint == checkpoint
    assert stage_diagnostics["qi_engineering_gate_passed"] is True
    assert checkpoint["partial"] is True
    assert checkpoint["role"] == "mirror_ramp"
    assert checkpoint["history"]["objective_initial"] == 4.0
    assert checkpoint["history"]["objective_final"] == 1.25
    assert checkpoint["diagnostics"]["qi_smooth_total"] == 1.7e-3
    assert checkpoint["diagnostics"]["qi_legacy_total"] == 1.2e-3
    assert checkpoint["diagnostics"]["qi_mirror_ratio_max"] == 0.28
    assert checkpoint["diagnostics"]["qi_seed_gate_passed"] is True
    assert checkpoint["diagnostics"]["qi_engineering_gate_passed"] is True
    assert checkpoint["promotion"]["qi_cleanup_promoted"] is True
    assert checkpoint["input_path"] == checkpoint["final_input_path"]
    assert checkpoint["wout_path"] == checkpoint["final_wout_path"]
    assert checkpoint["provenance"] == {
        "stage_output_dir": str(stage_dir),
        "initial_input_path": str(stage_dir / "input.initial"),
        "final_input_path": str(stage_dir / "input.final"),
        "initial_wout_path": str(stage_dir / "wout_initial.nc"),
        "final_wout_path": str(stage_dir / "wout_final.nc"),
    }
