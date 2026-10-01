"""Guard: docs/reference/performance.rst is generated from the benchmark artifact.

The baseline table between the generated-block markers must match what
``tools/render_performance_docs.py`` renders from
``benchmarks/baseline.json`` — the review finding this prevents: a
hand-maintained narrative table silently disagreeing with the committed
measurement artifact.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import render_performance_docs as rpd  # noqa: E402
from benchmarks.profile_resources import (  # noqa: E402
    _mirror_ladder,
    _parser,
    _peak_rss_bytes,
    _repeat_error,
)


def test_performance_table_matches_baseline_artifact() -> None:
    baseline = json.loads(rpd.BASELINE.read_text())
    text = rpd.DOC.read_text()
    head, rest = text.split(rpd.BEGIN, 1)
    _inner, _tail = rest.split(rpd.END, 1)
    rendered = rpd.render(baseline)
    assert rpd.BEGIN + rest.split(rpd.END, 1)[0] + rpd.END == rendered, (
        "docs/reference/performance.rst baseline table is stale; run python tools/render_performance_docs.py"
    )


def test_render_marks_wins_and_footnotes() -> None:
    baseline = json.loads(rpd.BASELINE.read_text())
    rendered = rpd.render(baseline)
    assert "**" in rendered, "no winning rows marked — renderer broken?"
    assert "VMEC++" in rendered
    row_count = sum(not key.startswith("_") for key in baseline)
    assert str(row_count) in rendered  # the computed row count


def test_benchmark_scripts_import_this_checkout_from_any_cwd(
    tmp_path: Path,
) -> None:
    """A benchmark must not silently import an installed VMEX distribution."""
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    for script in (
        "run_baseline.py",
        "run_external_equilibrium.py",
        "run_freeboundary_multigrid.py",
        "run_high_mode_fft.py",
        "make_strong_force_comparison.py",
        "strong_certificate.py",
        "profile_resources.py",
    ):
        proc = subprocess.run(
            [sys.executable, str(ROOT / "benchmarks" / script), "--help"],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 0, proc.stderr


def test_resource_profiler_parses_platform_memory_and_mirror_ladders() -> None:
    assert _peak_rss_bytes("12345 maximum resident set size") == 12345
    assert (
        _peak_rss_bytes("Maximum resident set size (kbytes): 12345")
        == 12345 * 1024
    )
    assert _mirror_ladder("5:7:4,9:17:9") == [(5, 7, 4), (9, 17, 9)]
    assert _parser().parse_args(["--device", "gpu", "--device-index", "1"]).device_index == 1
    absolute, relative = _repeat_error([1.0, 2.0], [1.0, 2.0 + 1e-12])
    assert absolute == pytest.approx(1e-12)
    assert relative == pytest.approx(5e-13)


def test_benchmark_artifacts_disclose_redacted_provenance() -> None:
    artifacts = (
        ROOT / "benchmarks" / "baseline.json",
        ROOT / "benchmarks" / "freeboundary_multigrid.json",
        ROOT / "benchmarks" / "high_mode_fft.json",
        ROOT / "benchmarks" / "gpu_baseline.json",
        ROOT / "benchmarks" / "convergence_nfp4_ns51.json",
    )
    for artifact in artifacts:
        report = json.loads(artifact.read_text())
        provenance = report.get("_provenance") or report["provenance"]
        assert re.fullmatch(r"[0-9a-f]{8,40}", provenance["measurement_commit"])
        assert provenance["input_data_embedded"] is False
        encoded = json.dumps(provenance)
        assert "/Users/" not in encoded
        assert "/home/" not in encoded


def test_validation_strong_force_figure_matches_committed_sources() -> None:
    metadata = json.loads(
        (ROOT / "benchmarks" / "strong_force_comparison_m4.json").read_text()
    )
    assert metadata["schema"] == "vmex.strong-force-readme-figure/4"
    figure = ROOT / metadata["figure"]
    assert figure.is_file()
    assert hashlib.sha256(figure.read_bytes()).hexdigest() == metadata[
        "figure_sha256"
    ]
    cases = metadata["cases"]
    assert set(cases) == {"shaped_tokamak_pressure", "nfp2_QA_finite_beta"}
    for case in cases.values():
        for source in case["sources"].values():
            path = ROOT / source["path"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"]
    tokamak = cases["shaped_tokamak_pressure"]["sources"]
    assert set(tokamak) == {"VMEX", "VMEC2000", "VMEC++", "DESC"}
    # Preserve the recorded reconstruction ordering; this is not native DESC.
    assert tokamak["VMEX"]["normalized_l2"] < tokamak["DESC"]["normalized_l2"]

    stellarator = cases["nfp2_QA_finite_beta"]["sources"]
    bundle = json.loads((ROOT / stellarator["DESC"]["path"]).read_text())
    desc = bundle["cases"]["nfp2_QA_finite_beta"]["sources"]["DESC"]
    assert desc["external_source"]["success"] is True
    representation = desc["external_source"]["representation"]
    assert representation["L"] >= 16
    assert representation["M"] >= 10 and representation["N"] >= 10
    assert desc["metrics"]["radial_refinement_difference"] < 1.0e-3
    # Historical WOUT reconstruction records, not a native solver ranking.
    # Runtime evidence lives in benchmarks/baselines, not in these guards.
    evidence = (ROOT / "docs/explanation/validation.md").read_text()
    assert metadata["figure"].removeprefix("docs/") in evidence


def test_fresh_deck_parity_artifact_is_provenanced_and_cited() -> None:
    """The fresh-deck xvmec2000 table is a hashed record, not prose.

    Every row the docs quote must trace to a deck hash, a reference-binary
    hash, and a vmex commit, and the page must cite the record by path so
    a reader can check the numbers rather than trust the phrasing.
    """
    path = ROOT / "benchmarks" / "fresh_decks_vs_vmec2000_2026-09-02.json"
    record = json.loads(path.read_text())
    assert record["schema"] == "vmex.fresh-deck-parity/1"
    provenance = record["provenance"]
    for key in ("vmex_commit", "vmex_version", "jax", "host", "protocol"):
        assert provenance[key]
    assert re.fullmatch(r"[0-9a-f]{16}", provenance["reference"]["sha256_prefix"])
    decks = record["decks"]
    assert len(decks) == 6
    for deck in decks:
        assert re.fullmatch(r"[0-9a-f]{16}", deck["sha256_prefix"]), deck["deck"]
        walls = deck["wall_s"]
        assert 0.0 < walls["vmex_warm"] <= walls["vmex_cold"], deck["deck"]
        assert walls["xvmec2000"] > 0.0
        assert deck["max_rel_diff"] and all(
            0.0 <= value < 1.0e-9 for value in deck["max_rel_diff"].values()
        ), deck["deck"]
    page = (ROOT / "docs" / "reference" / "performance.rst").read_text()
    assert path.name in page
    assert "machine precision" not in page.split("Fresh decks against")[1].split(
        "Numerical reproducibility")[0]


#: The two committed polish force-error records: the shaped tokamak whose
#: before/after pair the validation page quotes, and the bundled solovev deck that
#: shows what ``eps_F`` looks like when its denominator has collapsed.
def _prose_number(value: float, digits: int = 3) -> str:
    """Format a measurement the way the validation page prints it.

    ``f"{v:.3e}"`` pads the exponent (``1.284e-02``); prose writes
    ``1.284e-2``.  Comparing through one formatter keeps the prose pinned
    to the artifact digit for digit without dictating its typography.
    """
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def test_readme_states_the_certificate_ceiling_and_its_selection() -> None:
    """P0/P1: the two claims the README must not make silently.

    eps_F may not appear without its bound, and a figure built from
    hand-picked cases may not be presented as general evidence.
    """
    readme = (ROOT / "README.md").read_text()
    assert "bounded above by 2 by construction" in readme
    assert "demonstrably wins" not in readme
    assert "docs/explanation/validation.md" in readme
    evidence = (ROOT / "docs/explanation/validation.md").read_text()
    assert "not a ranking of native solvers" in evidence
    assert "selected for successful tokamak polishing" in evidence
    page = (
        ROOT / "docs" / "explanation" / "high-order-force-balance.rst"
    ).read_text()
    assert "bounded above by 2 by construction" in page
    assert "never be quoted on its own" in page


def test_committed_reports_do_not_expose_personal_paths() -> None:
    """Release-facing text must remain portable between contributors."""
    text_suffixes = {".json", ".md", ".py", ".rst", ".toml"}
    for directory in ("benchmarks", "docs", "examples"):
        for path in (ROOT / directory).rglob("*"):
            if path.is_file() and path.suffix in text_suffixes and "_build" not in path.parts:
                text = path.read_text(errors="replace")
                assert "/Users/" not in text, path
                assert "/home/" not in text, path
                assert "MacBook-Pro.local" not in text, path
                assert "office" not in text.lower(), path
