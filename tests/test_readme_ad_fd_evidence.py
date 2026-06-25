from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.diagnostics import readme_ad_fd_evidence as evidence


PUBLIC_EVIDENCE_JSON = evidence.REPO_ROOT / "docs/_static/figures/readme_ad_fd_evidence.json"
PUBLIC_REQUIRED_SCALARS = {
    "aspect ratio",
    "iota profile",
    "QS residual",
    "smooth QI residual",
    "DMerc",
    "D_R",
    "free-boundary aspect",
    "free-boundary qs_total",
    "free-boundary mean_iota",
    "free-boundary lcfs_boundary_moment",
}


def _write_branch_report(path: Path, *, scalars: dict, passed: bool = True) -> Path:
    payload = {
        "branch_local_vector_gate": {
            "physical_scalar_gate": {
                "passed": passed,
                "scalars": scalars,
            }
        }
    }
    path.write_text(json.dumps(payload))
    return path


def _scalar(ad: float, fd: float | None = None, *, passed: bool = True) -> dict:
    if fd is None:
        fd = ad
    return {
        "exact_directional": ad,
        "complete_fd_directional": fd,
        "abs_error": abs(ad - fd),
        "rel_error": abs(ad - fd) / max(abs(ad), abs(fd), 1.0),
        "passed": passed,
    }


def test_branch_local_evidence_requires_promoted_physical_scalars(tmp_path: Path) -> None:
    scalars = {
        "aspect": _scalar(0.10),
        "qs_total": _scalar(-0.02),
        "mean_iota": _scalar(0.03),
        "lcfs_boundary_moment": _scalar(-0.04),
    }
    report = _write_branch_report(tmp_path / "same_branch_report.json", scalars=scalars)

    rows = evidence._branch_local_rows(report)

    assert [row.scalar for row in rows] == [
        "free-boundary aspect",
        "free-boundary qs_total",
        "free-boundary mean_iota",
        "free-boundary lcfs_boundary_moment",
    ]
    assert {row.tolerance for row in rows} == {evidence.STRICT_DETERMINISTIC_TOL}
    assert all(row.passed for row in rows)
    assert all("same-branch/fingerprint-gated" in row.note for row in rows)


def test_checked_in_public_evidence_keeps_promoted_rows_and_strict_tolerances() -> None:
    payload = json.loads(PUBLIC_EVIDENCE_JSON.read_text())
    records = payload["records"]
    by_scalar = {record["scalar"]: record for record in records}

    assert set(by_scalar) == PUBLIC_REQUIRED_SCALARS
    assert all(record["passed"] is True for record in records)
    assert all(float(record["tolerance"]) <= evidence.STRICT_DETERMINISTIC_TOL for record in records)
    assert max(float(record["rel_error"]) for record in records) <= evidence.STRICT_DETERMINISTIC_TOL

    metadata = payload["metadata"]
    assert metadata["branch_local_report"].endswith(
        "outputs/final_tranche_adfd_evidence/same_branch_complete_solve_report.json"
    )
    assert "same-branch/fingerprint-gated" in metadata["contract"]
    assert "arbitrary adaptive" not in metadata["contract"].lower()
    assert all(
        "not arbitrary adaptive branch differentiation" in by_scalar[scalar]["note"]
        for scalar in PUBLIC_REQUIRED_SCALARS
        if scalar.startswith("free-boundary")
    )


def test_branch_local_evidence_rejects_incomplete_or_failed_reports(tmp_path: Path) -> None:
    incomplete = _write_branch_report(
        tmp_path / "incomplete.json",
        scalars={
            "aspect": _scalar(0.10),
            "qs_total": _scalar(-0.02),
            "mean_iota": _scalar(0.03),
        },
    )
    with pytest.raises(ValueError, match="lcfs_boundary_moment"):
        evidence._branch_local_rows(incomplete)

    failed_gate = _write_branch_report(
        tmp_path / "failed_gate.json",
        scalars={
            "aspect": _scalar(0.10),
            "qs_total": _scalar(-0.02),
            "mean_iota": _scalar(0.03),
            "lcfs_boundary_moment": _scalar(-0.04),
        },
        passed=False,
    )
    with pytest.raises(ValueError, match="did not pass"):
        evidence._branch_local_rows(failed_gate)
