"""P0: published benchmark citations and prose limits remain checkable offline."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("docs_gate", ROOT / "tools/check_docs_prose.py")
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def test_published_benchmark_paths_exist() -> None:
    pages = [ROOT / "README.md", ROOT / "CHANGELOG.md"]
    pages += [p for p in (ROOT / "docs").rglob("*")
              if p.suffix in (".md", ".rst") and "_build" not in p.parts]
    errors: list[str] = []
    for page in pages:
        gate.check_cited_paths(page, errors)
    assert not errors, "\n".join(errors)


@pytest.mark.parametrize("suffix,content", [
    (".md", "[record](../../benchmarks/present.json) and `benchmarks/missing.json`"),
    (".rst", "``benchmarks/present.json`` and ``benchmarks/missing.json``"),
    (".rst", "``benchmarks/present*.json`` and ``benchmarks/missing*.json``"),
])
def test_citations_reject_missing_records(tmp_path, monkeypatch, suffix, content):
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    (tmp_path / "benchmarks").mkdir()
    (tmp_path / "benchmarks/present.json").write_text("{}")
    page = tmp_path / f"page{suffix}"
    page.write_text(content)
    errors: list[str] = []
    gate.check_cited_paths(page, errors)
    assert len(errors) == 1 and "benchmarks/missing" in errors[0]


@pytest.mark.parametrize("suffix,content", [
    (".md", "```bash\nwrite --out benchmarks/generated.json\n```\n"),
    (".rst", ".. code-block:: bash\n\n   write --out benchmarks/generated.json\n\n"),
    (".rst", "Generate output::\n\n   write --out benchmarks/generated.json\n\n"),
])
def test_output_commands_are_not_claimed_records(tmp_path, monkeypatch, suffix, content):
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    page = tmp_path / f"page{suffix}"
    page.write_text(content + "Missing evidence: benchmarks/missing.json\n")
    errors: list[str] = []
    gate.check_cited_paths(page, errors)
    assert len(errors) == 1 and "benchmarks/missing.json" in errors[0]


@pytest.mark.parametrize("name,cap", [("README.md", 300), ("CHANGELOG.md", 200)])
def test_root_line_caps(tmp_path, name, cap):
    page = tmp_path / name
    page.write_text("text\n" * cap)
    errors: list[str] = []
    gate.check_root_limits(page, errors)
    assert not errors
    page.write_text(page.read_text() + "extra\n")
    gate.check_root_limits(page, errors)
    assert len(errors) == 1 and "line cap" in errors[0]


@pytest.mark.parametrize("claim", ["26-fold", "26 fold", "26×", "26x"])
def test_changelog_rejects_withdrawn_gain(tmp_path, claim):
    page = tmp_path / "CHANGELOG.md"
    page.write_text(f"Polish improves force error {claim}.\n")
    errors: list[str] = []
    gate.check_root_limits(page, errors)
    assert len(errors) == 1 and "withdrawn polish gain" in errors[0]


def test_rst_prose_scan_preserves_code_math_and_reference_titles(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    monkeypatch.setattr(gate, "DOCS", tmp_path)
    page = tmp_path / "page.rst"
    page.write_text(".. code-block:: python\n\n   robust = True\n\n"
                    ".. math::\n\n   x!\n\n``robust``\nRobust prose.\n")
    errors: list[str] = []
    gate.check_prose(page, errors)
    assert len(errors) == 1 and "banned phrase" in errors[0]
    reference = tmp_path / "project/references.rst"
    reference.parent.mkdir()
    reference.write_text("Author, “Robust\n  method”, Journal.\nRobust prose.\n")
    errors.clear()
    gate.check_prose(reference, errors)
    assert len(errors) == 1 and ":3:" in errors[0]
