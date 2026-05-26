from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "diagnostics" / "local_ci_gate.py"


def test_local_ci_gate_documents_required_local_lanes() -> None:
    text = SCRIPT.read_text()

    required_fragments = (
        "COVERAGE_FAIL_UNDER = 95",
        '"cli-smoke-help"',
        '"cli-smoke-solve"',
        '"compile"',
        '"repo-size-audit"',
        '"fast-pytest-coverage"',
        '"fetch-assets"',
        '"physics-smoke"',
        '"build"',
        '"docs-fast"',
        '"docs-full"',
        "not full and not vmec2000 and not simsopt",
        "--cov-fail-under={COVERAGE_FAIL_UNDER}",
        "tools/diagnostics/repo_size_audit.py",
        '"--max-total-mib"',
        '"50"',
        '"--max-file-mib"',
        '"2"',
        "SPHINX_FAST",
        "READTHEDOCS",
    )
    for fragment in required_fragments:
        assert fragment in text


def test_local_ci_gate_has_opt_in_selection_controls() -> None:
    text = SCRIPT.read_text()

    assert "--dry-run" in text
    assert "--only" in text
    assert "--skip" in text
    assert "--list" in text
