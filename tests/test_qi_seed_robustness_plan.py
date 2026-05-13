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


def test_optional_parity_commands_are_concrete_and_bounded():
    mod = _load_module()

    plan = mod.build_plan()
    commands = {command["command_id"]: command for command in plan["optional_parity_commands"]}

    assert plan["schema_version"] == 2
    assert {
        "simsopt-qh-formula-smoke",
        "vmec2000-stage-trace-smoke",
        "vmec2000-cli-five-iter",
        "bundled-wout-two-case-smoke",
    } <= set(commands)

    for command in commands.values():
        assert command["required_ci"] is False
        assert command["env"], command["command_id"]
        assert command["bounded_by"], command["command_id"]
        assert command["validates"], command["command_id"]
        assert "pytest -q tests/test_" in command["command"]
        assert "-m vmec2000" not in command["command"]

    assert commands["simsopt-qh-formula-smoke"]["env"] == ["RUN_SIMSOPT_VALIDATION=1"]
    assert "RUN_SIMSOPT_VALIDATION=1" in commands["simsopt-qh-formula-smoke"]["command"]
    assert (
        "::test_qh_quasisymmetry_residual_matches_simsopt_wout_formula"
        in commands["simsopt-qh-formula-smoke"]["command"]
    )

    vmec_stage = commands["vmec2000-stage-trace-smoke"]
    assert "VMEC2000_INTEGRATION=1" in vmec_stage["command"]
    assert "::test_fast_vmec2000_stage_trace_validation_cases" in vmec_stage["command"]
    assert any("--max-iter 2" in bound for bound in vmec_stage["bounded_by"])

    vmec_cli = commands["vmec2000-cli-five-iter"]
    assert "VMEC2000_CLI_NITER=5" in vmec_cli["command"]
    assert any("five iterations" in bound for bound in vmec_cli["bounded_by"])


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
    assert plan["next_parity_gates"][0]["status"] == "covered-fast-ci"


def test_plan_lists_fast_qi_diagnostic_artifacts():
    mod = _load_module()

    plan = mod.build_plan()
    artifacts = {artifact["artifact_id"]: artifact for artifact in plan["fast_diagnostic_artifacts"]}

    artifact = artifacts["qi-seed-suitability-annotation"]
    assert artifact["required_ci"] is True
    assert any("test_qi_seed_suitability_annotation_reports_gate_failures" in test for test in artifact["tests"])
    assert any("legacy Goodman-style QI totals" in item for item in artifact["validates"])


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
    assert "Fast Diagnostic Artifacts" in markdown
    assert "qi-seed-suitability-annotation" in markdown
    assert "Optional Parity Commands" in markdown
    assert "simsopt-qh-formula-smoke" in markdown
    assert "vmec2000-optional" in markdown
