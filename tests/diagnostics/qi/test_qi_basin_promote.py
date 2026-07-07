from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.diagnostics.qi.qi_basin_promote import (
    default_promotion_policies,
    load_candidate_records,
    main,
    _should_continue_after_stage,
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
        "iota_seed_qi_cleanup",
        "direct_matrix_free",
        "repeat_continuation",
        "qi_then_al_cleanup",
        "soft_wall_cleanup",
    } <= set(policies)
    assert policies["guarded_iota_ramp"].stages[0].iota_weight == 0.0
    assert policies["guarded_iota_ramp"].stages[1].qi_ceiling_weight > 0.0
    assert policies["guarded_iota_ramp"].stages[1].iota_weight > 0.0
    assert policies["guarded_iota_ramp"].stages[0].continue_if_qi_aspect_pass is True
    assert policies["direct_matrix_free"].stages[0].stage_modes == (3,)
    assert policies["iota_seed_qi_cleanup"].stages[0].iota_weight > 0.0
    assert policies["iota_seed_qi_cleanup"].stages[0].qi_weight > policies["direct_matrix_free"].stages[0].qi_weight
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


def test_should_continue_after_stage_allows_qi_preserve_failures_only() -> None:
    stage = default_promotion_policies(max_nfev=2)[0].stages[0]

    assert _should_continue_after_stage(
        stage,
        {"qi_seed_gate_passed": False, "qi_gate_failures": ["iota", "mirror", "elongation"]},
    )
    assert not _should_continue_after_stage(
        stage,
        {"qi_seed_gate_passed": False, "qi_gate_failures": ["smooth_qi", "iota"]},
    )


def test_qi_example_stage_promotion_allows_guarded_iota_gain() -> None:
    import examples.optimization.qi_optimization_support as support

    support.configure(
        {
            "QI_GATE_SMOOTH_MAX": 2.0e-3,
            "QI_GATE_LEGACY_MAX": 1.0e-3,
        }
    )
    stage = {
        "accept_if_iota_improves": True,
        "iota_improvement_min": 0.05,
        "qi_relax_for_iota": 2.0,
    }
    reference = {
        "mean_iota": -0.42,
        "qi_smooth_total": 1.0e-3,
        "qi_legacy_total": 5.0e-4,
    }
    promotion = {
        "mean_iota": -0.50,
        "qi_smooth_total": 1.5e-3,
        "qi_legacy_total": 7.0e-4,
        "qi_cleanup_promoted": False,
        "qi_cleanup_rejection_reasons": ["mirror ratio did not improve"],
    }

    out = support.stage_promotes_candidate(stage, promotion, reference)

    assert out["qi_cleanup_promoted"] is True
    assert out["qi_cleanup_rejection_reasons"] == []
    assert "iota increased" in out["qi_iota_promotion_reason"]


def test_qi_example_stage_promotion_reports_rank_and_engineering_regressions() -> None:
    import examples.optimization.qi_optimization_support as support

    support.configure(
        {
            "QI_GATE_SMOOTH_MAX": 2.0e-3,
            "QI_GATE_LEGACY_MAX": 1.0e-3,
        }
    )
    stage = {
        "accept_if_rank_improves": True,
        "accept_if_engineering_score_improves": True,
        "mirror_improvement_min": 0.03,
    }
    reference = {
        "qi_seed_gate_passed": True,
        "qi_engineering_gate_passed": True,
        "qi_rank_score": 1.0,
        "qi_constraint_score": 1.0,
        "qi_mirror_ratio_max": 0.20,
    }
    promotion = {
        "qi_seed_gate_passed": True,
        "qi_engineering_gate_passed": True,
        "qi_rank_score": 1.2,
        "qi_constraint_score": 1.0,
        "qi_mirror_ratio_max": 0.19,
        "qi_cleanup_promoted": True,
        "qi_cleanup_rejection_reasons": [],
    }

    out = support.stage_promotes_candidate(stage, promotion, reference)

    assert out["qi_cleanup_promoted"] is False
    reasons = "\n".join(out["qi_cleanup_rejection_reasons"])
    assert "rank score did not improve" in reasons
    assert "engineering score did not improve" in reasons
    assert "mirror ratio did not improve enough" in reasons


def test_qi_reference_safe_filter_uses_summary_mirror_field() -> None:
    import examples.optimization.qi_optimization_support as support

    assert support.boundary_reference_record_is_qi_safe(
        {"mirror": 0.29, "mean_iota": 0.43, "aspect": 5.2},
        max_mirror_ratio=0.30,
        abs_iota_min=0.41,
        target_aspect=5.0,
    )
    assert not support.boundary_reference_record_is_qi_safe(
        {"mirror": 0.34, "mean_iota": 0.43, "aspect": 5.2},
        max_mirror_ratio=0.30,
        abs_iota_min=0.41,
        target_aspect=5.0,
    )
    assert not support.boundary_reference_record_is_qi_safe(
        {"mirror": 0.29, "mean_iota": 0.43, "aspect": 6.6},
        max_mirror_ratio=0.30,
        abs_iota_min=0.41,
        target_aspect=5.0,
    )


def test_cli_requires_candidate_inputs(tmp_path: Path) -> None:
    candidates_file = tmp_path / "top_candidates.json"
    candidates_file.write_text(json.dumps([{"rank": 1, "label": "no-input"}]) + "\n")

    with pytest.raises(ValueError, match="No promotable candidate"):
        main(["--candidates", str(candidates_file), "--out-root", str(tmp_path / "out")])
