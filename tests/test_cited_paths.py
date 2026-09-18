"""P0: published benchmark citations and prose limits remain checkable offline."""

from __future__ import annotations

import importlib.util
import re
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


# --- benchmarks/INDEX.md: every artifact accounted for, every path real ------

INDEX_SPEC = importlib.util.spec_from_file_location(
    "benchmark_index", ROOT / "tools/render_benchmark_index.py"
)
assert INDEX_SPEC is not None and INDEX_SPEC.loader is not None
index_tool = importlib.util.module_from_spec(INDEX_SPEC)
INDEX_SPEC.loader.exec_module(index_tool)

#: A code span in INDEX.md that names a repository path (commits and column
#: headings are backticked too, but only paths carry a separator).
INDEX_PATH = re.compile(r"`((?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]*)`")


def _index_text() -> str:
    return (ROOT / "benchmarks/INDEX.md").read_text(encoding="utf-8")


def test_every_benchmark_artifact_appears_in_the_index() -> None:
    """No committed record may be missing from the index.

    An artifact is named outright or covered by a grouped directory whose
    file count the index states, so neither a new record nor a new file in
    a grouped directory can arrive unlisted.
    """
    text = _index_text()
    counts = {
        match.group(1): int(match.group(2))
        for match in re.finditer(r"\| `(benchmarks/\S+?)/` \| (\d+) \|", text)
    }
    missing: list[str] = []
    for rel in index_tool.artifacts():
        group = index_tool.group_of(rel)
        if group is None:
            if f"`{rel}`" not in text:
                missing.append(rel)
        elif group not in counts:
            missing.append(rel)
    assert not missing, (
        "absent from benchmarks/INDEX.md; run python tools/render_benchmark_index.py:\n"
        + "\n".join(missing)
    )
    for group, stated in counts.items():
        actual = sum(index_tool.group_of(rel) == group for rel in index_tool.artifacts())
        assert stated == actual, (
            f"{group}/ holds {actual} files but INDEX.md states {stated}; "
            "run python tools/render_benchmark_index.py"
        )


def test_index_names_no_path_that_is_missing() -> None:
    """Every path the index cites must exist in this checkout."""
    missing = sorted(
        {
            target
            for target in INDEX_PATH.findall(_index_text())
            if not (ROOT / target.rstrip("/")).exists()
        }
    )
    assert not missing, "benchmarks/INDEX.md cites paths that do not exist:\n" + "\n".join(missing)


def test_index_is_regenerated_from_the_tree() -> None:
    """The committed index must be exactly what the generator renders."""
    assert index_tool.render() == _index_text(), (
        "benchmarks/INDEX.md is stale; run python tools/render_benchmark_index.py"
    )


def test_every_artifact_has_an_attributed_generator() -> None:
    """A new artifact must say which script writes it, or admit that none does."""
    for rel in index_tool.artifacts():
        script = index_tool.generator_of(rel)  # raises when unattributed
        assert not script or (ROOT / script).is_file(), f"{rel}: generator {script} is missing"


def test_index_detects_an_artifact_the_index_forgot(tmp_path, monkeypatch) -> None:
    """The membership check has to actually fail on an unlisted record."""
    monkeypatch.setattr(index_tool, "REPO", tmp_path)
    monkeypatch.setattr(index_tool, "BENCHMARKS", tmp_path / "benchmarks")
    monkeypatch.setattr(index_tool, "INDEX", tmp_path / "benchmarks/INDEX.md")
    (tmp_path / "benchmarks").mkdir()
    (tmp_path / "benchmarks/baseline.json").write_text("{}")
    (tmp_path / "benchmarks/INDEX.md").write_text("no records here\n")
    assert index_tool.artifacts() == ["benchmarks/baseline.json"]
    assert "`benchmarks/baseline.json`" not in (tmp_path / "benchmarks/INDEX.md").read_text()
    assert "`benchmarks/baseline.json`" in index_tool.render()
