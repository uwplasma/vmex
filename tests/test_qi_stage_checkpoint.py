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
    stage_history = json.loads((stage_dir / "history.json").read_text())
    stage_diagnostics = json.loads((stage_dir / "diagnostics.json").read_text())

    assert root_checkpoint == checkpoint
    assert stage_history["objective_initial"] == 4.0
    assert stage_history["objective_final"] == 1.25
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


def test_qi_stage_checkpoint_writes_history_and_partial_diagnostics_before_audit(tmp_path: Path) -> None:
    mod = _load_module()
    mod.configure({"OUTPUT_DIR": tmp_path})
    stage_result = SimpleNamespace(
        history={
            "history": [
                {"objective": 3.0, "qs_objective": 2.0e-3, "aspect": 8.8, "iota": 0.41, "wall_time_s": 0.0},
                {"objective": 1.5, "qs_objective": 1.1e-3, "aspect": 7.2, "iota": 0.46, "wall_time_s": 4.0},
            ],
            "objective_final": 1.5,
            "qs_final": 1.1e-3,
            "aspect_final": 7.2,
            "iota_final": 0.46,
            "nfev": 3,
            "njev": 2,
            "total_wall_time_s": 4.0,
        }
    )

    checkpoint_path = mod.write_qi_stage_checkpoint(
        tmp_path,
        stage_index=1,
        stage_name="qi_optimization",
        stage_modes=(3,),
        stage_result=stage_result,
        diagnostics={},
        promotion={"diagnostics_pending": True},
        role="stage_pre_diagnostics",
    )

    history = json.loads((tmp_path / "history.json").read_text())
    diagnostics = json.loads((tmp_path / "diagnostics.json").read_text())
    checkpoint = json.loads(checkpoint_path.read_text())

    assert history["history"][-1]["objective"] == 1.5
    assert diagnostics["partial"] is True
    assert diagnostics["diagnostics_pending"] is True
    assert diagnostics["objective_final"] == 1.5
    assert diagnostics["qs_final"] == 1.1e-3
    assert diagnostics["aspect"] == 7.2
    assert diagnostics["mean_iota"] == 0.46
    assert checkpoint["history_path"] == str(tmp_path / "history.json")
