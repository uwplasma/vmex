from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "validation" / "qi_seed_robustness_plan.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("qi_seed_robustness_plan", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_plan_keeps_external_validation_optional():
    mod = _load_module()

    plan = mod.build_plan()
    lanes = {lane["lane_id"]: lane for lane in plan["lanes"]}

    assert plan["required_ci_baseline"]["status"] == "success"
    assert plan["required_ci_policy"]["heavy_external_validation_required"] is False
    assert lanes["simsopt-optional"]["required_ci"] is False
    assert lanes["vmec2000-optional"]["required_ci"] is False
    assert "RUN_SIMSOPT_VALIDATION=1" in lanes["simsopt-optional"]["command"]
    assert "VMEC2000_INTEGRATION=1" in lanes["vmec2000-optional"]["command"]
    assert "pytest" in lanes["required-fast-ci"]["command"]


def test_plan_covers_family_representatives_and_next_gates():
    mod = _load_module()

    plan = mod.build_plan()
    families = {row["family"] for row in plan["family_representatives"]}
    optional_families = {
        row["family"] for row in plan["family_representatives"] if not row["required_for_family_probe"]
    }
    gate_names = {gate["gate"] for gate in plan["next_parity_gates"]}

    assert {"qi", "qp", "qh", "qa", "simple"} <= families
    assert optional_families == {"qp"}
    assert {"VMEC2000 executable parity", "SIMSOPT diagnostic parity", "Family-prefine probes"} <= gate_names


def test_cli_writes_json_and_markdown(tmp_path):
    mod = _load_module()

    json_path = tmp_path / "plan.json"
    md_path = tmp_path / "plan.md"

    assert mod.main(["--output", str(json_path)]) == 0
    assert mod.main(["--output", str(md_path), "--format", "markdown"]) == 0

    payload = json.loads(json_path.read_text())
    markdown = md_path.read_text()

    assert payload["mode"] == "optional_qi_seed_robustness_validation_plan"
    assert payload["required_ci_policy"]["heavy_external_validation_required"] is False
    assert "Optional QI Seed Robustness Validation Plan" in markdown
    assert "vmec2000-optional" in markdown
