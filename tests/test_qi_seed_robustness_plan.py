from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "validation" / "qi_seed_robustness_plan.py"
REPO_ROOT = SCRIPT.parents[1]


def _load_module():
    spec = importlib.util.spec_from_file_location("qi_seed_robustness_plan", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_test_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _test_nodes_from_command(command: str) -> list[tuple[Path, str | None]]:
    nodes: list[tuple[Path, str | None]] = []
    for match in re.finditer(r"(tests/[-_a-zA-Z0-9/]+\.py)(?:::([-_a-zA-Z0-9]+))?", command):
        nodes.append((REPO_ROOT / match.group(1), match.group(2)))
    return nodes


def test_plan_keeps_external_validation_optional():
    mod = _load_module()

    plan = mod.build_plan()
    lanes = {lane["lane_id"]: lane for lane in plan["lanes"]}

    assert plan["required_ci_baseline"]["status"] == "unverified"
    assert plan["required_ci_baseline"]["verification_required"] is True
    assert "gh run list" in plan["required_ci_baseline"]["verification_command"]
    assert plan["required_ci_policy"]["heavy_external_validation_required"] is False
    assert lanes["simsopt-optional"]["required_ci"] is False
    assert lanes["vmec2000-optional"]["required_ci"] is False
    assert "RUN_SIMSOPT_VALIDATION=1" in lanes["simsopt-optional"]["command"]
    assert "test_redl_bootstrap_formula_matches_simsopt_when_available" in lanes["simsopt-optional"]["command"]
    assert "VMEC2000_INTEGRATION=1" in lanes["vmec2000-optional"]["command"]
    assert "pytest" in lanes["required-fast-ci"]["command"]


def test_optional_parity_commands_reference_collectable_tests():
    mod = _load_module()

    plan = mod.build_plan()
    for command in plan["optional_parity_commands"]:
        nodes = _test_nodes_from_command(command["command"])
        assert nodes, command["command_id"]
        for path, test_name in nodes:
            assert path.exists(), f"{command['command_id']} references missing {path}"
            if test_name is not None:
                module = _load_test_module(path)
                assert hasattr(module, test_name), f"{command['command_id']} references missing {test_name}"


def test_optional_parity_commands_are_concrete_and_bounded():
    mod = _load_module()

    plan = mod.build_plan()
    commands = {command["command_id"]: command for command in plan["optional_parity_commands"]}

    assert plan["schema_version"] == 2
    assert {
        "simsopt-qs-family-formula",
        "simsopt-qs-family-state",
        "simsopt-redl-formula",
        "vmec2000-converged-wout-smoke",
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

    assert commands["simsopt-qs-family-formula"]["env"] == ["RUN_SIMSOPT_VALIDATION=1"]
    assert "RUN_SIMSOPT_VALIDATION=1" in commands["simsopt-qs-family-formula"]["command"]
    assert (
        "::test_quasisymmetry_residual_family_matches_simsopt_wout_formula"
        in commands["simsopt-qs-family-formula"]["command"]
    )
    assert (
        "::test_quasisymmetry_state_diagnostic_family_matches_simsopt_converged_wout"
        in commands["simsopt-qs-family-state"]["command"]
    )
    assert commands["simsopt-redl-formula"]["env"] == ["RUN_SIMSOPT_VALIDATION=1"]
    assert "::test_redl_bootstrap_formula_matches_simsopt_when_available" in commands["simsopt-redl-formula"]["command"]
    assert any("synthetic three-point" in bound for bound in commands["simsopt-redl-formula"]["bounded_by"])

    vmec_converged = commands["vmec2000-converged-wout-smoke"]
    assert "VMEC2000_INTEGRATION=1" in vmec_converged["command"]
    assert "::test_vmec2000_converged_wout_diagnostics_validation" in vmec_converged["command"]
    assert any("converged end-state" in bound for bound in vmec_converged["bounded_by"])

    vmec_stage = commands["vmec2000-stage-trace-smoke"]
    assert "::test_fast_vmec2000_stage_trace_validation_cases" in vmec_stage["command"]

    vmec_cli = commands["vmec2000-cli-five-iter"]
    assert "VMEC2000_CLI_NITER=5" in vmec_cli["command"]
    assert any("five iterations" in bound for bound in vmec_cli["bounded_by"])


def test_optional_vmec2000_commands_preserve_timeout_and_iteration_bounds():
    mod = _load_module()
    vmec_tests = _load_test_module(REPO_ROOT / "tests" / "test_vmec2000_exec_fast_validation.py")

    plan = mod.build_plan()
    commands = {command["command_id"]: command for command in plan["optional_parity_commands"]}
    stage = commands["vmec2000-stage-trace-smoke"]
    converged = commands["vmec2000-converged-wout-smoke"]

    stage_bounds = " ".join(stage["bounded_by"])
    assert f"--single-ns {vmec_tests.VMEC2000_STAGE_TRACE_SINGLE_NS}" in stage_bounds
    assert f"--max-iter {vmec_tests.VMEC2000_STAGE_TRACE_MAX_ITER}" in stage_bounds
    assert f"{int(vmec_tests.VMEC2000_STAGE_TRACE_TIMEOUT_S)}s executable timeout" in stage_bounds
    assert len(vmec_tests.VMEC2000_STAGE_TRACE_CASES) == 2

    converged_bounds = " ".join(converged["bounded_by"])
    assert f"{int(vmec_tests.VMEC2000_CONVERGED_TIMEOUT_S)}s executable timeout" in converged_bounds
    assert len(vmec_tests.VMEC2000_CONVERGED_WOUT_CASES) == 3
    assert {
        "nfp4_QH_warm_start",
        "circular_tokamak",
        "shaped_tokamak_pressure",
    } == {case[0] for case in vmec_tests.VMEC2000_CONVERGED_WOUT_CASES}
    assert all("NS_ARRAY" in updates and "NITER_ARRAY" in updates for _case, _input, updates in vmec_tests.VMEC2000_CONVERGED_WOUT_CASES)


def test_simsopt_optional_commands_cover_qa_and_qh_family_cases():
    mod = _load_module()
    simsopt_tests = _load_test_module(REPO_ROOT / "tests" / "test_simsopt_optional_validation.py")

    plan = mod.build_plan()
    commands = {command["command_id"]: command for command in plan["optional_parity_commands"]}
    family_cases = {case["case"]: case for case in simsopt_tests.QS_FAMILY_CASES}

    assert set(family_cases) == {"qh_warm_start", "qa_landreman_paul_lowres"}
    assert family_cases["qh_warm_start"]["helicity_n"] == -1
    assert family_cases["qa_landreman_paul_lowres"]["helicity_n"] == 0
    assert simsopt_tests.QS_LOW_GRID == {"n_surfaces": 3, "ntheta": 15, "nphi": 16}
    assert "QA and QH" in " ".join(commands["simsopt-qs-family-formula"]["bounded_by"])
    assert "QA and QH" in " ".join(commands["simsopt-qs-family-state"]["bounded_by"])
    assert "15x16" in " ".join(commands["simsopt-qs-family-formula"]["bounded_by"])


def test_required_ci_lane_excludes_full_vmec2000_and_simsopt_markers():
    mod = _load_module()

    plan = mod.build_plan()
    lane = {item["lane_id"]: item for item in plan["lanes"]}["required-fast-ci"]
    command = lane["command"]

    assert "VMEC2000_INTEGRATION" not in command
    assert "RUN_SIMSOPT_VALIDATION" not in command
    assert "-m vmec2000" not in command
    assert "-m simsopt" not in command
    assert "not vmec2000" in command
    assert "not simsopt" in command


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
    assert "simsopt-qs-family-formula" in markdown
    assert "vmec2000-converged-wout-smoke" in markdown
    assert "vmec2000-optional" in markdown
