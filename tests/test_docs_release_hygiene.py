from __future__ import annotations

from pathlib import Path

from tools.diagnostics.parity_sweep_manifest import DEFAULT_MANIFEST, _parse_manifest


ROOT = Path(__file__).resolve().parents[1]


def test_root_readme_stays_concise_and_defers_extended_claims() -> None:
    readme = (ROOT / "README.md").read_text()

    assert len(readme.splitlines()) < 220
    assert "docs/optimization.rst" in readme
    assert "docs/optimization_sweep_results.rst" in readme
    assert "docs/performance.rst" in readme
    assert "docs/release_checklist.rst" in readme
    assert "Latest repository release tag:" in readme
    assert "readme_best_optimization_qa.png" in readme
    assert "readme_best_optimization_qh.png" in readme
    assert "readme_best_optimization_qp.png" in readme
    assert "readme_best_optimization_qi.png" in readme

    forbidden_fragments = (
        "## Optimization from Different Initial Conditions",
        "## VMEC++ notes",
        "readme_runtime_compare.png",
        "case-timeout-s 1200",
        "generate_qs_ess_sweep.py --backend-label",
        "VMEC_JAX_QI_OUTPUT_DIR",
        "latest verified",
    )
    for fragment in forbidden_fragments:
        assert fragment not in readme


def test_optimization_docs_explain_explicit_final_output_control() -> None:
    guide = (ROOT / "docs" / "optimization.rst").read_text()
    examples_readme = (ROOT / "examples" / "optimization" / "README.md").read_text()

    assert "save_final_outputs=False" in guide
    assert "save_final_outputs=False" in examples_readme
    assert "selected exact accepted point" in guide
    assert "selected exact accepted point" in examples_readme
    assert "unreplayed relaxed trial point" in guide
    assert "unreplayed relaxed trial point" in examples_readme


def test_optional_validation_plan_uses_live_ci_verification() -> None:
    plan = (ROOT / "docs" / "optional_validation_plan.rst").read_text()
    validation = (ROOT / "docs" / "validation.rst").read_text()
    testing_strategy = (ROOT / "docs" / "testing_strategy.rst").read_text()

    assert "gh run list --repo uwplasma/vmec_jax --branch main --workflow CI" in plan
    assert "gh run view RUN_ID --repo uwplasma/vmec_jax" in plan
    assert "latest verified ``main`` CI run checked during this update was green" not in plan
    assert "head SHA:" not in plan
    assert "verified green CI baseline" not in validation
    assert "Last recorded local CI-equivalent coverage baseline" not in testing_strategy


def test_optional_validation_lasym_freeb_example_matches_manifest_case() -> None:
    plan = (ROOT / "docs" / "optional_validation_plan.rst").read_text()
    _meta, cases = _parse_manifest(DEFAULT_MANIFEST)
    case = next(case for case in cases if case["id"] == "freeb_nonaxis_lasym_true_cth_like_local")

    assert "--ids freeb_nonaxis_lasym_true_cth_like_local" in plan
    assert "--manifest tools/diagnostics/parity_manifest.toml" in plan
    assert "--vmec-exec \"$VMEC2000_EXEC\"" in plan
    assert "freeb_scalpot" in plan
    assert "VMEC_DUMP_*" in plan

    assert case["tier"] == "planning"
    assert case["compare"] == "freeb_scalpot"
    assert case["source"] == "vmec_jax/examples"
    assert bool(case["lfreeb"]) is True
    assert bool(case["lasym"]) is True
    assert bool(case["axisymmetric"]) is False
    assert set(case["runtime_thresholds_s_by_iter"]) == {"80", "100"}
    assert set(case["metric_thresholds_rel_scaled_by_iter"]) == {"80", "100"}


def test_qi_case_specific_artifacts_are_not_documented_as_aspect6_promotions() -> None:
    readme = (ROOT / "README.md").read_text()
    optimization = (ROOT / "docs" / "optimization.rst").read_text()
    sweep_results = (ROOT / "docs" / "optimization_sweep_results.rst").read_text()
    validation = (ROOT / "docs" / "validation.rst").read_text()
    qi_cases_csv = (ROOT / "docs" / "_static" / "figures" / "readme_qi_optimization_cases.csv").read_text()

    for text in (readme, optimization, sweep_results, validation):
        assert "case-specific" in text
        assert "aspect-6 README" in text
        assert "best-row promotion" in text

    for text in (readme, optimization, sweep_results):
        assert "NFP=1/2/3 have passing saved diagnostics" not in text
        assert "passing QI lanes" not in text
        assert "validation_status=promoted" not in text

    assert ",promoted," not in qi_cases_csv
    assert ",case-gated," in qi_cases_csv
