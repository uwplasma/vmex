from __future__ import annotations

import json

from tools.diagnostics.qi.qi_objective_component_report import (
    _annotate_wout_case_gates,
    _wout_rankings,
    build_synthetic_report,
    main,
)


def test_qi_objective_component_report_ranks_synthetic_cases_consistently():
    report = build_synthetic_report(nphi=21, nalpha=7, n_bounce=5, nphi_out=61)

    smooth_order = [row["case"] for row in report["rankings"]["smooth"]]
    legacy_order = [row["case"] for row in report["rankings"]["legacy"]]

    assert smooth_order == legacy_order
    assert set(smooth_order[:2]) == {"qi_cosine_well", "qi_shifted_well"}
    assert smooth_order[-1] == "helical_qh_like"

    by_case = {case["case"]: case for case in report["cases"]}
    qh = min(by_case["helical_qh_like"]["rows"], key=lambda row: row["smooth_total"])
    qi = min(by_case["qi_cosine_well"]["rows"], key=lambda row: row["smooth_total"])
    mixed = min(by_case["mixed_qi_helical"]["rows"], key=lambda row: row["smooth_total"])

    assert qi["smooth_total"] < mixed["smooth_total"] < qh["smooth_total"]
    assert qi["legacy_total"] < mixed["legacy_total"] < qh["legacy_total"]
    assert qh["reference_qi_closure_rms"] > qi["reference_qi_closure_rms"]


def test_qi_objective_component_report_cli_writes_json(tmp_path):
    out = tmp_path / "report.json"

    main([
        "--output",
        str(out),
        "--synthetic",
        "--nphi",
        "21",
        "--nalpha",
        "7",
        "--n-bounce",
        "5",
        "--nphi-out",
        "61",
    ])

    report = json.loads(out.read_text())
    assert report["mode"] == "synthetic"
    assert report["resolution"] == {
        "nphi": 21,
        "nalpha": 7,
        "n_bounce": 5,
        "nphi_out": 61,
    }
    assert len(report["cases"]) == 5


def test_wout_gate_annotations_rank_qi_preserving_mirror_cleanup():
    promoted = _annotate_wout_case_gates(
        {
            "case": "promoted",
            "smooth_total": 1.9e-3,
            "legacy_total": 2.8e-4,
            "mirror_ratio_max": 0.304,
            "mirror_ratio_target": 0.21,
            "max_elongation": 7.0,
            "elongation_target": 8.0,
            "aspect": 5.0,
            "mean_iota": -0.50,
        },
        smooth_gate=2.0e-3,
        legacy_gate=1.0e-3,
        abs_iota_min=0.41,
    )
    cleanup = _annotate_wout_case_gates(
        {
            "case": "safe_cleanup",
            "smooth_total": 1.8e-3,
            "legacy_total": 2.7e-4,
            "mirror_ratio_max": 0.300,
            "mirror_ratio_target": 0.21,
            "max_elongation": 7.2,
            "elongation_target": 8.0,
            "aspect": 5.0,
            "mean_iota": -0.50,
        },
        smooth_gate=2.0e-3,
        legacy_gate=1.0e-3,
        abs_iota_min=0.41,
    )
    bad_qi = _annotate_wout_case_gates(
        {
            "case": "bad_qi_low_mirror",
            "smooth_total": 5.0e-2,
            "legacy_total": 4.0e-2,
            "mirror_ratio_max": 0.25,
            "mirror_ratio_target": 0.21,
            "max_elongation": 5.0,
            "elongation_target": 8.0,
            "aspect": 5.0,
            "mean_iota": -0.50,
        },
        smooth_gate=2.0e-3,
        legacy_gate=1.0e-3,
        abs_iota_min=0.41,
    )

    rankings = _wout_rankings([promoted, cleanup, bad_qi])

    assert cleanup["qi_iota_gate_passed"] is True
    assert cleanup["qi_engineering_gate_passed"] is False
    assert cleanup["qi_mirror_cleanup_candidate"] is True
    assert bad_qi["qi_iota_gate_passed"] is False
    assert bad_qi["qi_gate_failures"][:2] == ["smooth_qi", "legacy_qi"]
    assert [row["case"] for row in rankings["mirror_cleanup"]] == [
        "safe_cleanup",
        "promoted",
        "bad_qi_low_mirror",
    ]
    assert [row["case"] for row in rankings["qi_iota_candidates"]] == [
        "safe_cleanup",
        "promoted",
    ]
