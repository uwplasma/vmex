from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "optimization" / "qi_optimization_cases.py"


def _load_cases_module(name: str = "qi_optimization_cases_test"):
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_resolve_qi_case_defaults_and_aliases(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("VMEC_JAX_QI_INPUT", raising=False)
    monkeypatch.delenv("VMEC_JAX_QI_RUN_CASE", raising=False)
    monkeypatch.delenv("VMEC_JAX_QI_OUTPUT_DIR", raising=False)
    mod = _load_cases_module("qi_optimization_cases_default_test")

    run_case, case = mod.resolve_qi_case()
    assert run_case == "minimal_nfp2_qi"
    assert case["input_file"].name == "input.minimal_seed_nfp2"

    monkeypatch.setenv("VMEC_JAX_QI_RUN_CASE", "nfp3_qi")
    monkeypatch.setenv("VMEC_JAX_QI_OUTPUT_DIR", str(tmp_path / "alias_out"))
    run_case, case = mod.resolve_qi_case()
    assert run_case == "nfp3_qi"
    assert case["input_file"].name == "input.minimal_seed_nfp3"
    assert case["case_goal"] == "NFP=3 minimal-seed QI lane"
    assert case["target_aspect"] == pytest.approx(mod.SEED3127_REVIEWED_TARGET_ASPECT)
    assert case["output_dir"] == tmp_path / "alias_out"


def test_nfp3_qi_catalog_matches_reviewed_seed3127_target_aspect(monkeypatch) -> None:
    monkeypatch.delenv("VMEC_JAX_QI_INPUT", raising=False)
    monkeypatch.delenv("VMEC_JAX_QI_OUTPUT_DIR", raising=False)
    monkeypatch.setenv("VMEC_JAX_QI_RUN_CASE", "nfp3_qi")
    mod = _load_cases_module("qi_optimization_cases_nfp3_reviewed_target_test")

    run_case, case = mod.resolve_qi_case()

    assert run_case == "nfp3_qi"
    assert case["target_aspect"] == pytest.approx(mod.SEED3127_REVIEWED_TARGET_ASPECT)
    assert case["boundary_reference_preconditioner"]["target_aspect"] == pytest.approx(
        mod.SEED3127_REVIEWED_TARGET_ASPECT
    )
    assert "minimal_nfp3" in str(case["output_dir"])


def test_minimal_and_circular_qi_cases_require_reference_seeded_local_stage() -> None:
    mod = _load_cases_module("qi_optimization_cases_minimal_local_stage_test")

    for case_name in (
        "circular_nfp1_qi",
        "minimal_nfp1_qi",
        "minimal_nfp2_qi",
        "minimal_nfp3_qi",
        "minimal_nfp4_qi",
    ):
        case = mod.QI_CASES[case_name]
        boundary_reference = case["boundary_reference_preconditioner"]
        stages = case["mirror_ramp_stages"]

        assert boundary_reference["enabled"] is True
        assert boundary_reference["accept_as_baseline"] is True
        assert stages
        assert all(int(stage["max_nfev"]) >= mod.MINIMAL_QI_LOCAL_STAGE_MIN_NFEV for stage in stages)
        assert all(stage["use_showcase_max_nfev"] is True for stage in stages)
        assert all(stage["use_showcase_max_mode"] is True for stage in stages)
        assert all(float(stage["aspect_weight"]) >= 4.0 for stage in stages)
        assert all("stage_modes" in stage or "stage_mode_limits" in stage for stage in stages)
        if case_name != "minimal_nfp3_qi":
            assert all(stage["stage_modes"] == (case["max_mode"],) for stage in stages)
            assert all(stage["use_mode_continuation"] is False for stage in stages)


def test_resolve_qi_case_external_input_uses_far_seed_policy_without_reference(monkeypatch, tmp_path: Path) -> None:
    input_path = tmp_path / "input.custom_seed"
    output_dir = tmp_path / "custom_out"
    monkeypatch.setenv("VMEC_JAX_QI_INPUT", str(input_path))
    monkeypatch.setenv("VMEC_JAX_QI_LABEL", "custom_seed")
    monkeypatch.setenv("VMEC_JAX_QI_OUTPUT_DIR", str(output_dir))
    monkeypatch.delenv("VMEC_JAX_QI_RUN_CASE", raising=False)
    monkeypatch.delenv("VMEC_JAX_QI_REFERENCE_INPUT", raising=False)
    mod = _load_cases_module("qi_optimization_cases_external_test")

    run_case, case = mod.resolve_qi_case()

    assert run_case == "custom_seed"
    assert case["input_file"] == input_path
    assert case["output_dir"] == output_dir
    assert case["case_goal"] == "external VMEC input using the far-seed QI+iota robustness policy"
    assert case["boundary_reference_preconditioner"] == {"enabled": False}
    assert case["mirror_ramp_stages"] == mod.QI_CASES["qi_stel_seed_3127"]["mirror_ramp_stages"]


def test_resolve_qi_case_external_reference_overrides_policy_controls(monkeypatch, tmp_path: Path) -> None:
    input_path = tmp_path / "input.custom"
    reference_path = tmp_path / "input.reference"
    monkeypatch.setenv("VMEC_JAX_QI_INPUT", str(input_path))
    monkeypatch.setenv("VMEC_JAX_QI_RUN_CASE", "external_reference")
    monkeypatch.setenv("VMEC_JAX_QI_POLICY_CASE", "nfp4_qi_finite_beta")
    monkeypatch.setenv("VMEC_JAX_QI_REFERENCE_INPUT", str(reference_path))
    monkeypatch.setenv("VMEC_JAX_QI_REFERENCE_LAMBDAS", "0.99, 1.0 1.01")
    monkeypatch.setenv("VMEC_JAX_QI_MAX_MODE", "2")
    monkeypatch.setenv("VMEC_JAX_QI_INNER_MAX_ITER", "7")
    mod = _load_cases_module("qi_optimization_cases_reference_test")

    run_case, case = mod.resolve_qi_case()
    boundary_reference = case["boundary_reference_preconditioner"]

    assert run_case == "external_reference"
    assert case["input_file"] == input_path
    assert boundary_reference["enabled"] is True
    assert boundary_reference["reference_input"] == reference_path
    assert boundary_reference["max_mode"] == 2
    assert boundary_reference["target_aspect"] == pytest.approx(mod.DEFAULT_QI_TARGET_ASPECT)
    assert boundary_reference["max_mirror_ratio"] == pytest.approx(0.35)
    assert boundary_reference["smooth_qi_max"] == pytest.approx(3.0e-3)
    assert boundary_reference["legacy_qi_max"] == pytest.approx(2.0e-3)
    assert boundary_reference["max_iter"] == 7
    assert boundary_reference["lambdas"] == (0.99, 1.0, 1.01)
    assert boundary_reference["prefer_qi_safe_candidates"] is True


def test_resolve_qi_case_rejects_unknown_policy_and_bad_reference_lambdas(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VMEC_JAX_QI_INPUT", str(tmp_path / "input.custom"))
    monkeypatch.setenv("VMEC_JAX_QI_POLICY_CASE", "not_a_case")
    mod = _load_cases_module("qi_optimization_cases_error_test")

    with pytest.raises(KeyError, match="Unknown VMEC_JAX_QI_POLICY_CASE"):
        mod.resolve_qi_case()

    monkeypatch.setenv("VMEC_JAX_QI_POLICY_CASE", "qi_stel_seed_3127")
    monkeypatch.setenv("VMEC_JAX_QI_REFERENCE_INPUT", str(tmp_path / "input.reference"))
    monkeypatch.setenv("VMEC_JAX_QI_REFERENCE_LAMBDAS", "0.99, bad")
    with pytest.raises(ValueError, match="VMEC_JAX_QI_REFERENCE_LAMBDAS"):
        mod.resolve_qi_case()
