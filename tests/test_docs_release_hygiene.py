from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_root_readme_stays_concise_and_defers_extended_claims() -> None:
    readme = (ROOT / "README.md").read_text()

    assert len(readme.splitlines()) < 220
    assert "docs/optimization.rst" in readme
    assert "docs/optimization_sweep_results.rst" in readme
    assert "docs/performance.rst" in readme
    assert "docs/release_checklist.rst" in readme
    assert "readme_best_optimization_qa.png" in readme
    assert "readme_best_optimization_qh.png" in readme
    assert "readme_best_optimization_qp.png" in readme
    assert "readme_best_optimization_qi.png" in readme

    forbidden_fragments = (
        "## Optimization from Different Initial Conditions",
        "## VMEC++ notes",
        "readme_runtime_compare.png",
        "case-timeout-s 1200",
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
