from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = REPO_ROOT / "validation/artifacts/qi_seed_multifamily_prefine_20260607.json"


def test_multifamily_prefine_artifact_records_all_seed_families_without_qi_worsening():
    artifact = json.loads(ARTIFACT.read_text())
    rows = artifact["rows"]
    by_label = {row["label"]: row for row in rows}

    assert artifact["status"] == "bounded_first_pass_evidence_not_full_seed_robustness_claim"
    assert artifact["summary"]["statuses"] == {"completed": 5}
    assert artifact["summary"]["qi_worsened_count"] == 0
    assert artifact["summary"]["scalar_improved_qi_worsened_count"] == 0
    assert {row["family"] for row in rows} == {"qp", "qi", "qh", "qa", "simple"}
    assert {
        "qp_from_omnigenity_nfp2_qi",
        "qi_omnigenity_nfp3",
        "qh_nfp4_warm_start",
        "qa_landreman_paul_lowres",
        "simple_circular_tokamak",
    } == set(by_label)

    for row in rows:
        assert row["accepted"] is True
        assert row["qi_worsened"] is False
        assert row["smooth_qi_final"] <= row["smooth_qi_initial"] + 1.0e-15
        assert row["legacy_qi_final"] <= row["legacy_qi_initial"] + 1.0e-15


def test_multifamily_prefine_artifact_keeps_near_qi_rows_as_diagnostic_baselines():
    artifact = json.loads(ARTIFACT.read_text())
    baseline_rows = [row for row in artifact["rows"] if row["diagnostic_baseline"]]

    assert {row["label"] for row in baseline_rows} == {
        "qp_from_omnigenity_nfp2_qi",
        "qi_omnigenity_nfp3",
    }
    for row in baseline_rows:
        assert row["prefine_policy"] == "near_qi_diagnostic_baseline"
        assert row["decision"] == "accepted_stable_low_objective"
        assert row["smooth_qi_final"] == row["smooth_qi_initial"]
        assert row["legacy_qi_final"] == row["legacy_qi_initial"]
        assert row["objective_final"] == row["objective_initial"]


def test_multifamily_prefine_artifact_does_not_claim_full_seed_robustness():
    artifact = json.loads(ARTIFACT.read_text())
    claim_boundary = artifact["claim_boundary"]

    unsupported = " ".join(claim_boundary["does_not_support"])
    purpose = artifact["purpose"]

    assert "full seed-robust QI claim" in unsupported
    assert "reviewed Boozer |B| contour plots" in unsupported
    assert "full seed-robust QI still requires" in purpose
