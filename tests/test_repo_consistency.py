"""Repository consistency: packaging, docs, committed records, figures and CI.

Solve-free invariants that keep what the repository says in line with what it
ships.  Each test pins one invariant; the regeneration command for a stale
artifact is in its failure message.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from packaging.requirements import Requirement
from packaging.version import Version

import vmex

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
FIGURES = DOCS / "_static" / "figures"
sys.path.insert(0, str(ROOT / "tools"))

import check_docs_prose as gate  # noqa: E402
import ci_scope  # noqa: E402
import render_capabilities as capabilities  # noqa: E402
import render_performance_docs as performance  # noqa: E402
import test_manifest  # noqa: E402
import update_figure_manifest as figure_tool  # noqa: E402

PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())
PROJECT = PYPROJECT["project"]
EXTRAS = PROJECT["optional-dependencies"]
WORKFLOWS = {p.name: p.read_text() for p in (ROOT / ".github" / "workflows").glob("*.yml")}


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def _record(rel: str) -> dict:
    return json.loads(_read(rel))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _doc_pages() -> list[Path]:
    pages = [p for p in DOCS.rglob("*") if p.suffix in (".md", ".rst") and "_build" not in p.parts]
    return [ROOT / "README.md", ROOT / "CHANGELOG.md", *sorted(pages)]


def _tool(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# -- packaging ----------------------------------------------------------------


def test_release_metadata_agrees() -> None:
    citation = next(line.split(":", 1)[1].strip() for line in _read("CITATION.cff").splitlines()
                    if line.startswith("version:"))
    assert vmex.__version__ == PROJECT["version"] == citation
    find = PYPROJECT["tool"]["setuptools"]["packages"]["find"]
    assert find["include"] == ["vmex*"]
    assert {"tests*", "docs*", "examples*", "tools*"} <= set(find["exclude"])
    assert PROJECT["scripts"] == {"vmex": "vmex.core.cli:main", "vmec": "vmex.core.cli:main"}
    assert {"setuptools", "packaging"} <= set(PYPROJECT["build-system"]["requires"])
    assert PROJECT["requires-python"] == ">=3.11"
    assert PROJECT["urls"]["Repository"] == "https://github.com/uwplasma/vmex"
    names = {Requirement(dep).name for dep in PROJECT["dependencies"]}
    assert {"matplotlib", "booz_xform_jax", "numpy", "packaging"} <= names
    assert not {"plots", "plot", "qi", "booz"} & set(EXTRAS)


def _floors(*extras: str) -> dict[str, str]:
    floors: dict[str, str] = {}
    for dep in [*PROJECT["dependencies"], *(r for name in extras for r in EXTRAS[name])]:
        requirement = Requirement(dep)
        for spec in requirement.specifier:
            floors[requirement.name] = max(floors.get(requirement.name, spec.version),
                                           spec.version, key=Version)
    return floors


def test_declared_floors_match_every_copy() -> None:
    from vmex._compat import OPTIONAL_MINIMUMS

    core = _floors()
    assert {name: Version(core[name]).release for name in vmex._MINIMUM_VERSIONS} \
        == vmex._MINIMUM_VERSIONS
    everything = _floors(*EXTRAS)
    assert {name: everything[name] for name in OPTIONAL_MINIMUMS} == OPTIONAL_MINIMUMS
    readme = _read("README.md")
    quoted = dict(re.findall(r'"([A-Za-z0-9_.-]+)>=([0-9][^"]*)"', readme))
    assert {"solvax", "booz_xform_jax", "essos", "virtual-casing-jax", "gkx", "dkx"} <= set(quoted)
    assert all(everything[n] == f for n, f in quoted.items() if n not in {"jax", "jaxlib"})
    for extra in ("all", "coils", "freeb", "neoclassical", "turbulence", "optimizers"):
        assert f'pip install "vmex[{extra}]"' in readme, extra
    nightly = WORKFLOWS["nightly.yml"]
    lane = nightly[nightly.index("if: matrix.install == 'floors'"):]
    pins = dict(re.findall(r"([A-Za-z0-9_.-]+)==(\S+)", lane[:lane.index("- name:")]))
    for name, floor in _floors("coils", "freeb", "neoclassical", "turbulence").items():
        assert Version(pins[name]) == Version(floor), (name, floor)


@pytest.mark.parametrize("found, message", [
    (None, 'essos>=0.21.0; run: pip install "essos>=0.21.0"'),
    ("0.19.2", 'essos>=0.21.0 (found 0.19.2); run: pip install -U "essos>=0.21.0"'),
])
def test_require_optional_names_the_fix(monkeypatch, found, message) -> None:
    from vmex import _compat

    def version(name):
        if found is None:
            raise _compat.importlib_metadata.PackageNotFoundError(name)
        return found

    monkeypatch.setattr(_compat.importlib_metadata, "version", version)
    with pytest.raises(ImportError, match=re.escape("alpha-particle tracing needs " + message)):
        _compat.require_optional("essos", "alpha-particle tracing")


def test_import_guard_names_the_fix_and_skips_unknown_metadata(monkeypatch) -> None:
    monkeypatch.setattr(vmex, "_package_version",
                        lambda name: "1.15.3" if name == "scipy" else "0.11.1")
    with pytest.raises(ImportError, match=r'scipy >= 1\.16 \(found 1\.15\.3\).*pip install -U "scipy>=1\.16"'):
        vmex._check_supported_versions()

    def missing(name):
        if name == "scipy":
            raise vmex._PackageNotFoundError(name)
        return "99.0"

    monkeypatch.setattr(vmex, "_package_version", missing)
    assert vmex._check_supported_versions() is None
    monkeypatch.setattr(vmex, "_sys", SimpleNamespace(
        version_info=(3, 10, 14), version="3.10.14 (main, Mar 1 2026)"))
    with pytest.raises(ImportError, match=r"Python >= 3\.11 \(found 3\.10\.14\)"):
        vmex._check_supported_versions()


# -- docs ---------------------------------------------------------------------

_LINK = re.compile(r"\]\(([^)#\s]+)\)|\{doc\}`(?:[^<`]*<)?([^>`]+)>?`|:doc:`(?:[^<`]*<)?([^>`]+)>?`")


def _links(docname: str) -> set[str]:
    """Docs pages one click away from ``docname`` (links, doc roles, toctrees)."""
    path = next((p for p in (DOCS / f"{docname}.md", DOCS / f"{docname}.rst") if p.exists()), None)
    if path is None:
        return set()
    text = path.read_text(encoding="utf-8")
    raw = [next(g for g in m.groups() if g) for m in _LINK.finditer(text)]
    blocks = re.findall(r"```\{toctree\}(.*?)```", text, flags=re.DOTALL)
    blocks += re.findall(r"^\.\. toctree::\n((?:[ \t]+.*\n?)*)", text, flags=re.MULTILINE)
    raw += [line.strip() for block in blocks for line in block.splitlines()
            if line.strip() and not line.strip().startswith(":")]
    found = set()
    for target in raw:
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        target = re.sub(r"\.(md|rst|html)$", "", target)
        try:
            found.add(target.lstrip("/") if target.startswith("/") else
                      (path.parent / target).resolve().relative_to(DOCS.resolve()).as_posix())
        except ValueError:
            continue
    return found


@pytest.mark.parametrize("target", [
    "howto/restart-from-previous-run", "howto/run-on-gpu", "reference/vmec2000-compatibility",
])
def test_key_questions_are_two_clicks_from_the_landing_page(target) -> None:
    first = _links("index")
    assert target in first.union(*(_links(page) for page in first))


def test_cited_record_paths_exist() -> None:
    errors: list[str] = []
    for page in _doc_pages():
        gate.check_cited_paths(page, errors)
    assert not errors, "\n".join(errors)


@pytest.mark.parametrize("check, name, content, needle", [
    ("check_cited_paths", "p.md", "[r](../../benchmarks/present.json) and `benchmarks/missing.json`",
     "benchmarks/missing.json"),
    ("check_cited_paths", "p.rst", "``benchmarks/present*.json`` and ``benchmarks/missing*.json``",
     "benchmarks/missing*"),
    ("check_cited_paths", "p.md", "```bash\nwrite --out benchmarks/generated.json\n```\n"
     "Missing: benchmarks/missing.json\n", "benchmarks/missing.json"),
    ("check_cited_paths", "p.rst", "Generate::\n\n   write --out benchmarks/generated.json\n\n"
     "Missing: benchmarks/missing.json\n", "benchmarks/missing.json"),
    ("check_root_limits", "README.md", "text\n" * 701, "line cap"),
    ("check_root_limits", "CHANGELOG.md", "Polish improves force error 26-fold.\n", "withdrawn polish gain"),
    ("check_prose", "p.rst", ".. code-block:: python\n\n   robust = True\n\n``robust``\nRobust prose.\n",
     "banned phrase"),
])
def test_docs_gate_reports_exactly_one_problem(tmp_path, monkeypatch, check, name, content, needle):
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    monkeypatch.setattr(gate, "DOCS", tmp_path)
    (tmp_path / "benchmarks").mkdir()
    (tmp_path / "benchmarks/present.json").write_text("{}")
    page = tmp_path / name
    page.write_text(content)
    errors: list[str] = []
    getattr(gate, check)(page, errors)
    assert len(errors) == 1 and needle in errors[0], errors


_AUTOMODULE = re.compile(r"^\s*\.\.\s+automodule::\s+([A-Za-z_][\w.]*)\s*$", re.MULTILINE)


def _public_modules() -> dict[str, Path]:
    modules = {}
    for path in sorted((ROOT / "vmex").rglob("*.py")):
        parts = path.relative_to(ROOT).with_suffix("").parts
        parts = parts[:-1] if parts[-1] == "__init__" else parts
        if not any(part.startswith("_") for part in parts):
            modules[".".join(parts)] = path
    return modules


def _documented_modules() -> set[str]:
    return {name for page in _doc_pages()[2:] for name in _AUTOMODULE.findall(page.read_text())}


def test_every_public_module_is_in_the_api_reference() -> None:
    assert not sorted(set(_public_modules()) - _documented_modules())


def _rendered(tree: ast.AST, depth: int = 0, public_class: bool = False):
    """Definitions ``automodule :members:`` renders: public, top level or in a public class."""
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.If, ast.Try, ast.With, ast.For, ast.While)):
            yield from _rendered(node, depth, public_class)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            public = not node.name.startswith("_")
            overload = any(getattr(d, "id", getattr(d, "attr", None)) == "overload"
                           for d in node.decorator_list)
            if public and not overload and (depth == 0 or (depth == 1 and public_class)):
                yield node
            yield from _rendered(node, depth + 1, isinstance(node, ast.ClassDef) and public)


def test_rendered_api_definitions_have_docstrings() -> None:
    missing = []
    for module, path in _public_modules().items():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if ast.get_docstring(tree) is None:
            missing.append(f"{module}:<module>")
        missing += [f"{module}:{node.name}:{node.lineno}" for node in _rendered(tree)
                    if ast.get_docstring(node) is None]
    assert not missing, "\n".join(missing)


def test_exports_are_documented_and_indexed() -> None:
    exports = [name for name in vmex.__all__ if name != "__version__"]
    assert not [n for n in exports if not (inspect.getdoc(getattr(vmex, n)) or "").strip()]
    assert not [n for n in exports if not re.search(rf"\b{re.escape(n)}\b", vmex.__doc__ or "")]


# -- committed records ----------------------------------------------------------


def test_capability_contract_is_rendered_and_evidenced() -> None:
    data = _record("benchmarks/capabilities.json")
    assert data["schema"] == "vmex.capabilities/1"
    assert capabilities.DOC.read_text() == capabilities.render(data), "run tools/render_capabilities.py"
    ids = [row["id"] for row in data["rows"]]
    assert len(ids) == len(set(ids))
    for row in data["rows"]:
        assert row["status"] and row["scope"] and row["evidence"]
        assert all(row[field] in capabilities.MARK for field in capabilities.FIELDS)
        assert all((ROOT / path).is_file() for path in row["evidence"])
    free = [row for row in data["rows"] if row["topology"] == "toroidal" and row["boundary"] == "free"]
    assert free and all(row["coupled_ad"] and row["jvp"] == "not-available" and row["vjp"] == "limited"
                        and "experimental CPU" in row["scope"] for row in free)
    mirror = next(row for row in data["rows"] if row["id"] == "mirror-open-free-axisymmetric-supported")
    assert mirror["beta_max"] == _record(
        "benchmarks/mirror_free_boundary_axisymmetric.json")["case"]["supported_beta_max"]
    assert _read("README.md").count("https://vmex.readthedocs.io/en/latest/reference/capabilities.html") == 1


def test_performance_table_is_rendered_from_the_baseline() -> None:
    text = performance.DOC.read_text()
    block = performance.BEGIN + text.split(performance.BEGIN, 1)[1].split(performance.END, 1)[0] + performance.END
    assert block == performance.render(json.loads(performance.BASELINE.read_text())), \
        "run python tools/render_performance_docs.py"


def test_records_carry_portable_provenance() -> None:
    for path in sorted((ROOT / "benchmarks").glob("*.json")):
        record = json.loads(path.read_text())
        provenance = record.get("_provenance") or record.get("provenance")
        if isinstance(provenance, dict) and "measurement_commit" in provenance:
            assert re.fullmatch(r"[0-9a-f]{8,40}", provenance["measurement_commit"]), path.name
            assert provenance.get("input_data_embedded") is not True, path.name
    for directory in ("benchmarks", "docs", "examples", "tools"):
        for path in (ROOT / directory).rglob("*"):
            if path.suffix in {".json", ".md", ".py", ".rst", ".toml"} and "_build" not in path.parts:
                text = path.read_text(errors="replace")
                for private in ("/Users/", "/home/", "MacBook-Pro.local"):
                    assert private not in text, (path, private)
                assert "office" not in text.lower(), path


def test_strong_force_figure_matches_its_sources() -> None:
    metadata = _record("benchmarks/strong_force_comparison_m4.json")
    assert metadata["schema"] == "vmex.strong-force-readme-figure/4"
    assert _sha(ROOT / metadata["figure"]) == metadata["figure_sha256"]
    assert set(metadata["cases"]) == {"shaped_tokamak_pressure", "nfp2_QA_finite_beta"}
    for case in metadata["cases"].values():
        for source in case["sources"].values():
            assert _sha(ROOT / source["path"]) == source["sha256"]
    tokamak = metadata["cases"]["shaped_tokamak_pressure"]["sources"]
    assert len(tokamak) == 4 and {"VMEX", "VMEC2000", "DESC"} <= set(tokamak)
    assert tokamak["VMEX"]["normalized_l2"] < tokamak["DESC"]["normalized_l2"]
    bundle = _record(metadata["cases"]["nfp2_QA_finite_beta"]["sources"]["DESC"]["path"])
    desc = bundle["cases"]["nfp2_QA_finite_beta"]["sources"]["DESC"]
    representation = desc["external_source"]["representation"]
    assert desc["external_source"]["success"] and representation["L"] >= 16
    assert representation["M"] >= 10 and representation["N"] >= 10
    assert desc["metrics"]["radial_refinement_difference"] < 1.0e-3
    assert metadata["figure"].removeprefix("docs/") in _read("docs/explanation/validation.md")


@pytest.mark.parametrize("page, phrase", [
    ("README.md", "bounded above by 2 by construction"),
    ("README.md", "docs/explanation/validation.md"),
    ("docs/explanation/validation.md", "not a ranking of native solvers"),
    ("docs/explanation/validation.md", "selected for successful tokamak polishing"),
    ("docs/explanation/high-order-force-balance.rst", "bounded above by 2 by construction"),
    ("docs/explanation/high-order-force-balance.rst", "never be quoted on its own"),
])
def test_force_certificate_claims_keep_their_bounds(page, phrase) -> None:
    assert phrase in _read(page)
    assert "demonstrably wins" not in _read("README.md")


def test_preconditioner_table_quotes_its_record() -> None:
    record = _record("benchmarks/preconditioner_2d_stiff_cases.json")
    assert record["schema"] == "vmex.preconditioner-2d-stiff-cases/2"
    provenance = record["provenance"]
    assert provenance["measurement_dirty"] is False and provenance["versions"]["jax"]
    page = _read("docs/reference/performance.rst")
    for case in record["cases"]:
        assert case["iterations_2d"] < case["iterations_1d"]
        assert case["wb_relative_difference"] < 1e-5
        if case["in_readme_figure"]:
            for value in (case["iterations_1d"], case["iterations_2d"]):
                assert f"     - {value}\n" in page, (case["key"], value)


def test_diagnostics_figure_matches_its_recorded_solve() -> None:
    record = _record("docs/_static/figures/readme_diagnostics.json")
    assert record["schema"] == "vmex.readme-diagnostics-figures/1"
    assert record["provenance"]["measurement_dirty"] is False
    evidence = _read("docs/explanation/validation.md")
    for case in record["cases"]:
        assert _sha(ROOT / case["figure"]) == case["figure_sha256"]
        assert _sha(ROOT / case["deck"]) == case["deck_sha256"]
        assert f"{100 * case['scalars']['betatotal']:.2f}" in evidence, case["key"]


# -- figures --------------------------------------------------------------------


def test_figure_manifest_matches_the_tree() -> None:
    manifest = json.loads(figure_tool.MANIFEST.read_text())
    assert manifest["schema"] == figure_tool.SCHEMA
    listed = [row["path"] for row in manifest["figures"]]
    assert sorted(listed) == sorted(p.relative_to(ROOT).as_posix() for p in figure_tool.figure_paths())
    assert figure_tool.refreshed(manifest) == manifest, "run python tools/update_figure_manifest.py"
    embedded = {name for page in _doc_pages()[:1] + _doc_pages()[2:]
                for name in re.findall(r"[\w./-]*?([\w-]+\.webp)", page.read_text())}
    assert embedded <= {Path(path).name for path in listed}


def test_figure_provenance_points_at_real_files() -> None:
    for row in json.loads(figure_tool.MANIFEST.read_text())["figures"]:
        where = row["path"]
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", row["generated"]), where
        assert row["reproducible"] in {"command", "manual", "external-data"}, where
        assert isinstance(row["bytes_verified"], bool), where
        assert not row["bytes_verified"] or row["reproducible"] == "command", where
        assert row["note"] and row["command"] and row["generator"], where
        for target in [row["generator"], row["record"], *row["inputs"]]:
            assert target is None or (ROOT / target).exists(), (where, target)


# -- benchmark drivers ------------------------------------------------------------


def test_baseline_driver_does_not_count_niter_exhaustion_as_convergence() -> None:
    result = _tool("run_baseline").parse_vmec2000({"ok": True, "stdout": (
        "\n 1000  1.29E-01  1.09E-01  3.95E-02\n"
        " Try increasing NITER or PRE_NITER if the preconditioner is on.\n"
        " EXECUTION TERMINATED NORMALLY\n")})
    assert result["ok"] and not result["converged"] and result["iterations"] == 1000


def test_device_parity_driver_contracts(tmp_path) -> None:
    parity = _tool("device_parity")
    assert parity._requested_devices("cpu,gpu", {"cpu": object()}) == (
        ["cpu"], {"gpu": "no GPU JAX device is available"})
    for name in ("dmerc_interior_mean", "jdotb_interior_mean", "glasser_d_r_interior_mean"):
        assert name in parity.METRIC_NAMES and name in parity._metrics(quick=True)
    cpu = {"state": np.array([1.0, 2.0]), "metrics": {"mhd_energy": (3.0, 4.0)}}
    gpu = {"state": np.array([1.0, 2.0 + 1e-10]), "metrics": {"mhd_energy": (3.0, 4.0)}}
    comparison = parity._compare_lanes(cpu, gpu, rtol=1e-7)
    assert comparison["status"] == "passed"
    assert comparison["forward"]["state_relative_l2"] == pytest.approx(1e-10 / np.sqrt(5.0))
    assert comparison["metrics"]["mhd_energy"]["gradient_relative_difference"] == 0.0

    reload = _tool("device_cache_reload")
    assert reload._parse_devices("gpu,cpu,gpu") == ("gpu", "cpu")
    with pytest.raises(ValueError, match="subset of cpu,gpu"):
        reload._parse_devices("tpu")
    (tmp_path / "nested").mkdir()
    (tmp_path / "first").write_bytes(b"abc")
    (tmp_path / "nested" / "second").write_bytes(b"12345")
    assert reload._cache_stats(tmp_path) == {"file_count": 2, "bytes": 8}

    def run(wall, forward, gradient):
        lane = {"forward": {"cold_wall_s": forward, "warm_wall_s": 2.0},
                "gradients": {"mhd_energy": {"cold_wall_s": gradient, "warm_wall_s": 1.0}}}
        return {"subprocess_wall_s": wall, "child": {"lanes": {"gpu": lane}}}

    assert reload._speedups(run(30.0, 12.0, 10.0), run(12.0, 4.0, 3.0), "gpu") == {
        "process_cache_reload_speedup": 2.5, "forward_cache_reload_speedup": 3.0,
        "gradient_cache_reload_speedup": pytest.approx(10.0 / 3.0),
        "reload_forward_warm_speedup": 2.0, "reload_gradient_warm_speedup": 3.0}
    assert reload._speedups({}, run(12.0, 4.0, 3.0), "gpu") is None


# -- test manifest and CI -------------------------------------------------------


def test_collected_suite_has_exact_manifest_ownership() -> None:
    assert not test_manifest.validate(test_manifest.collect())


def _invoked_lanes() -> set[str]:
    """Lanes the workflows hand to ``test_manifest.py select``, in every spelling."""
    text = "\n".join(WORKFLOWS.values())
    invoked = {lane for value in re.findall(r"^\s*(?:-\s*)?selector:\s*(.+?)\s*$", text, re.M)
               for lane in value.split()}
    invoked |= set(re.findall(r"test_manifest\.py select ([a-z][A-Za-z0-9-]*)", text))
    for declaration in re.findall(r"^\s*ALL:\s*'(\[.*\])'\s*$", text, re.M):
        invoked |= {lane for entry in json.loads(declaration)
                    for lane in str(entry.get("selector", "")).split()}
    for body in WORKFLOWS.values():
        values = [v.strip() for group in re.findall(r"^\s*campaign:\s*\[(.*)\]", body, re.M)
                  for v in group.split(",")]
        for prefix in re.findall(r"select ([a-z-]+)\$\{\{ matrix\.campaign \}\}", body):
            invoked |= {prefix + value for value in values}
    return invoked


def test_every_manifest_lane_runs_in_a_workflow_or_is_local_only() -> None:
    data, records = test_manifest.load()
    named = {lane for record in records for lane in record["lanes"]}
    named |= set(data.get("selectors", {})) | set(data["campaigns"])
    invoked, local = _invoked_lanes(), data.get("local_only", {})
    for lane, entry in local.items():
        assert lane in named and lane not in invoked, lane
        assert entry.get("reason") and f"select {lane}" in entry.get("command", ""), lane
    assert not sorted(named - invoked - set(local))
    assert "self-hosted" not in "".join(WORKFLOWS.values())
    for lane in ("full-polish-gn", "full-run-options", "full-free-boundary-adjoint"):
        assert test_manifest.select(lane) and f"selector: {lane}" in WORKFLOWS["nightly.yml"]


def _declared(job: str) -> set[str]:
    block = WORKFLOWS["ci.yml"].split(f"- id: {job}", 1)[1]
    declaration = re.search(r"^\s*ALL:\s*'(\[.*\])'\s*$", block, re.M).group(1)
    return {lane for entry in json.loads(declaration) for lane in str(entry.get("selector", "")).split()}


def test_workflow_contracts() -> None:
    for name in ("ci.yml", "nightly.yml", "weekly.yml"):
        assert "tools/test_manifest.py select" in WORKFLOWS[name]
    ci, nightly = WORKFLOWS["ci.yml"], WORKFLOWS["nightly.yml"]
    assert "name: PR gate" in ci and 'jax: ["0.9.2", "0.11.1"]' in ci
    assert "device, jax-compatibility, changed-coverage]" in ci
    assert "campaign: [opt-qi, opt-qa, opt-qh, opt-qp]" in nightly
    assert "tools/test_manifest.py select optional" in nightly and 'RUN_FULL: "1"' in nightly
    assert "--cov=vmex" not in nightly and "timeout-minutes: 45" in nightly
    bundles = _record("assets/manifest.json")["bundles"]
    ncsx = next(b["name"] for b in bundles if "examples/data/mgrid_ncsx_c09r00_small.nc" in b["common_paths"])
    assert f"tools/fetch_assets.py --bundle {ncsx}" in nightly
    physics, parity = _declared("physics"), _declared("parity")
    assert "pr-mirror-spline" in physics and not physics & parity
    assert "tests/mirror/test_output.py" in test_manifest.select("pr-mirror-spline")
    assert len(test_manifest.select("weekly-hmfb-fixed")) == len(test_manifest.select("weekly-hmfb-free")) == 2


def test_ci_scope_narrows_only_attributable_changes() -> None:
    entries = [{"lane": "a", "selector": "pr-parity-a1"}, {"lane": "c2", "selector": "pr-parity-c2"}]
    for paths in (["vmex/core/solver.py"], [".github/workflows/ci.yml"], ["tools/ci_scope.py"],
                  ["pyproject.toml"], ["vmex/core/solver.py", "tests/test_gammac.py"]):
        assert ci_scope.select_lanes(paths, entries) == entries, paths
    assert "pr-parity-c4" in ci_scope.needed_lanes(["tests/test_gammac.py"])
    assert "pr-physics-field" in ci_scope.needed_lanes(["tests/test_monitoring.py"])
    assert ci_scope.needed_lanes(["examples/optimization/QA_optimization.py"])
    assert ci_scope.needed_lanes(["tests/test_not_in_the_manifest.py"]) is None
    assert ci_scope.needed_lanes(["benchmarks/baseline.json", "docs/_static/figures/figures.json"]) == set()
    prose = ci_scope.needed_lanes(["CHANGELOG.md"])
    assert prose is not None and ci_scope.needed_lanes(["README.md"]) is not None
    assert ci_scope.needed_lanes(["examples/optimization/single_stage_optimization.py",
                                  "CHANGELOG.md"]) >= prose
    assert ci_scope.needed_lanes(["vmex/core/solver.py", "CHANGELOG.md"]) is None
    assert ci_scope.classify(["docs/howto/gpu.rst", "README.md", "docs/figure.webp"]) == (False, False)
    assert ci_scope.classify(["docs/howto/gpu.rst", "vmex/doctor.py"]) == (True, True)
    assert ci_scope.classify(["tests/test_doctor.py"]) == ci_scope.classify([".github/workflows/ci.yml"]) \
        == (True, False)
    assert ci_scope.classify([]) == ci_scope.classify(["README.md"], force_all=True) == (True, True)


def _pytest(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    env = {key: value for key, value in (env or os.environ).items() if key != "RUN_FULL"}
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", *args], cwd=ROOT,
                            env=env, text=True, capture_output=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_manifest_report_lists_timings_and_every_skip(tmp_path) -> None:
    report = tmp_path / "report.json"
    _pytest("tests/test_lasym_free_convergence.py", "tests/test_fetch_assets.py::test_golden_bundle_targets"
            "_the_user_cache", f"--vmex-report={report}")
    data = json.loads(report.read_text())
    assert data["schema"] == "vmex.test-report/1" and data["collected"] == 2 and data["slowest"]
    assert [skip["nodeid"].split("::")[0] for skip in data["skips"]] == ["tests/test_lasym_free_convergence.py"]
    for record in data["slowest"] + data["skips"]:
        assert {"owner", "primary", "duration", "device", "asset", "oracle"} <= record.keys()


def test_modules_restore_jit_and_full_markers_split_nonlinear_work(tmp_path) -> None:
    (tmp_path / "jit_probe.py").write_text(
        "import jax, pytest\n"
        "@pytest.hookimpl(wrapper=True)\n"
        "def pytest_runtest_teardown(item, nextitem):\n"
        "    result = yield\n"
        "    if nextitem is None or nextitem.module is not item.module:\n"
        "        assert not jax.config.jax_disable_jit, item.nodeid\n"
        "    return result\n")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(tmp_path), os.environ.get("PYTHONPATH", "")]))
    result = _pytest("-p", "jit_probe",
                     "tests/test_scaling.py::test_input_scaling_changes_only_dimensional_quantities",
                     "tests/test_cli_freeboundary.py::test_free_boundary_keeps_the_state_on_iteration_exhaustion",
                     "tests/test_optimize.py::test_public_problem_factory_validation", env=env)
    assert "3 passed" in result.stdout
    nodes = _pytest("--collect-only", "-m", "full", "tests/test_run_options.py",
                    "tests/test_freeboundary_implicit.py").stdout
    assert "tests/test_run_options.py::test_solve_file_polish_directive_activates_polishing" in nodes
    assert ("tests/test_freeboundary_implicit.py::test_free_boundary_current_gradient_matches_resolve"
            "_finite_difference") in nodes
    assert "::test_solve_file_directives_reach_driver_once" not in nodes
