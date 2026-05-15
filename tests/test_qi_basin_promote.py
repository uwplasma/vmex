from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.diagnostics.qi_basin_promote import (
    default_promotion_policies,
    load_candidate_records,
    main,
    write_summary,
)


def _write_candidates(path: Path, input_file: Path) -> None:
    records = [
        {
            "rank": 2,
            "label": "worse",
            "score": 2.0,
            "input_path": str(input_file),
            "metrics": {"qi_smooth_total": 2.0e-2},
        },
        {
            "rank": 1,
            "label": "best",
            "score": 1.0,
            "input_path": str(input_file),
            "metrics": {"qi_smooth_total": 1.0e-2},
        },
        {
            "rank": 3,
            "label": "missing",
            "score": 3.0,
            "input_path": str(input_file.parent / "does_not_exist"),
            "metrics": {"qi_smooth_total": 3.0e-2},
        },
    ]
    path.write_text(json.dumps(records) + "\n")


def test_default_promotion_policies_cover_global_local_lanes() -> None:
    policies = {policy.name: policy for policy in default_promotion_policies(max_nfev=2)}

    assert {
        "guarded_iota_ramp",
        "direct_matrix_free",
        "repeat_continuation",
        "qi_then_al_cleanup",
        "soft_wall_cleanup",
    } <= set(policies)
    assert policies["guarded_iota_ramp"].stages[0].iota_weight == 0.0
    assert policies["guarded_iota_ramp"].stages[1].qi_ceiling_weight > 0.0
    assert policies["guarded_iota_ramp"].stages[1].iota_weight > 0.0
    assert policies["direct_matrix_free"].stages[0].stage_modes == (3,)
    assert policies["repeat_continuation"].stages[0].stage_modes == (1, 1, 2, 2, 3, 3)
    assert len(policies["qi_then_al_cleanup"].stages) == 2
    assert policies["qi_then_al_cleanup"].stages[1].use_augmented_lagrangian is True
    assert policies["soft_wall_cleanup"].stages[0].qi_ceiling_weight > 0.0


def test_load_candidate_records_filters_missing_inputs_and_sorts(tmp_path: Path) -> None:
    input_file = tmp_path / "input.candidate"
    input_file.write_text("&INDATA\n/")
    candidates_file = tmp_path / "top_candidates.json"
    _write_candidates(candidates_file, input_file)

    records = load_candidate_records(candidates_file, top_n=2)

    assert [record["label"] for record in records] == ["best", "worse"]


def test_cli_dry_run_writes_promotion_plan(tmp_path: Path) -> None:
    input_file = tmp_path / "input.candidate"
    input_file.write_text("&INDATA\n/")
    candidates_file = tmp_path / "top_candidates.json"
    _write_candidates(candidates_file, input_file)

    rc = main(
        [
            "--candidates",
            str(candidates_file),
            "--out-root",
            str(tmp_path / "out"),
            "--top-n",
            "1",
            "--policy",
            "direct_matrix_free",
            "--max-nfev",
            "1",
        ]
    )

    assert rc == 0
    plan = json.loads((tmp_path / "out" / "promotion_plan.json").read_text())
    assert plan["execute"] is False
    assert plan["candidates"][0]["label"] == "best"
    assert plan["policies"][0]["name"] == "direct_matrix_free"


def test_write_summary_ranks_selected_and_writes_csv(tmp_path: Path) -> None:
    records = [
        {
            "rank": 1,
            "candidate_label": "bad",
            "policy": "p",
            "selected": False,
            "selection_reason": "failed",
            "smooth_qi": 1.0,
            "mirror": 1.0,
            "output_dir": "bad",
        },
        {
            "rank": 2,
            "candidate_label": "good",
            "policy": "p",
            "selected": True,
            "selection_reason": "passed",
            "smooth_qi": 1.0e-4,
            "mirror": 0.2,
            "output_dir": "good",
        },
    ]

    write_summary(records, tmp_path)

    summary = json.loads((tmp_path / "promotion_summary.json").read_text())
    assert summary[0]["candidate_label"] == "good"
    assert "candidate_label" in (tmp_path / "promotion_summary.csv").read_text()


def test_cli_requires_candidate_inputs(tmp_path: Path) -> None:
    candidates_file = tmp_path / "top_candidates.json"
    candidates_file.write_text(json.dumps([{"rank": 1, "label": "no-input"}]) + "\n")

    with pytest.raises(ValueError, match="No promotable candidate"):
        main(["--candidates", str(candidates_file), "--out-root", str(tmp_path / "out")])
