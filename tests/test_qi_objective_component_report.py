from __future__ import annotations

import json

from tools.diagnostics.qi_objective_component_report import build_synthetic_report, main


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
