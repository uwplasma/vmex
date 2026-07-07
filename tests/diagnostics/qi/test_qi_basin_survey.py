from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.diagnostics.qi.qi_basin_survey import (
    BasinCandidate,
    SurveyTargets,
    basin_score,
    generate_basin_candidates,
    main,
    rank_candidate_records,
)


def test_generate_basin_candidates_is_deterministic_and_scaled() -> None:
    names = ["rc10", "zs10", "rc20"]
    x_scale = np.asarray([1.0, 0.5, 0.1])

    first = generate_basin_candidates(
        names=names,
        x_scale=x_scale,
        radii=(0.1,),
        n_random=2,
        rng_seed=7,
        axis_count=2,
        directions=("axes", "rademacher"),
    )
    second = generate_basin_candidates(
        names=names,
        x_scale=x_scale,
        radii=(0.1,),
        n_random=2,
        rng_seed=7,
        axis_count=2,
        directions=("axes", "rademacher"),
    )

    assert [candidate.as_record(names) for candidate in first] == [
        candidate.as_record(names) for candidate in second
    ]
    assert first[0].label == "zero"
    axis_plus = next(candidate for candidate in first if candidate.label == "axis+:rc10:0.1")
    np.testing.assert_allclose(axis_plus.params, [0.1, 0.0, 0.0])
    axis_minus = next(candidate for candidate in first if candidate.label == "axis-:zs10:0.1")
    np.testing.assert_allclose(axis_minus.params, [0.0, -0.05, 0.0])


def test_basin_score_prioritizes_qi_and_engineering_gates() -> None:
    targets = SurveyTargets(
        smooth_qi_max=1.0e-3,
        legacy_qi_max=1.0e-3,
        mirror_ratio_max=0.3,
        max_elongation=8.0,
        abs_iota_min=0.41,
        target_aspect=10.0,
        aspect_tolerance=2.0,
    )
    good = {
        "qi_smooth_total": 5.0e-4,
        "qi_legacy_total": 5.0e-4,
        "qi_mirror_ratio_max": 0.25,
        "qi_max_elongation": 7.0,
        "mean_iota": 0.5,
        "aspect": 9.0,
    }
    bad_mirror = dict(good, qi_mirror_ratio_max=0.8)
    bad_qi = dict(good, qi_smooth_total=5.0e-3, qi_legacy_total=5.0e-3)

    assert basin_score(good, targets=targets) < basin_score(bad_mirror, targets=targets)
    assert basin_score(bad_mirror, targets=targets) < basin_score(bad_qi, targets=targets)


def test_rank_candidate_records_sorts_failures_last() -> None:
    records = [
        {
            "label": "bad",
            "metrics": {
                "qi_smooth_total": 1.0,
                "qi_legacy_total": 1.0,
                "qi_mirror_ratio_max": 1.0,
                "qi_max_elongation": 10.0,
                "mean_iota": 0.0,
                "aspect": 10.0,
            },
        },
        {
            "label": "good",
            "metrics": {
                "qi_smooth_total": 1.0e-4,
                "qi_legacy_total": 1.0e-4,
                "qi_mirror_ratio_max": 0.2,
                "qi_max_elongation": 6.0,
                "mean_iota": 0.6,
                "aspect": 10.0,
            },
        },
        {"label": "failed", "metrics": {}, "error": "solve failed"},
    ]

    ranked = rank_candidate_records(records)

    assert [record["label"] for record in ranked] == ["good", "bad", "failed"]
    assert [record["rank"] for record in ranked] == [1, 2, 3]


def test_cli_dry_run_writes_candidate_plan(tmp_path: Path) -> None:
    input_file = Path(__file__).resolve().parents[3] / "examples" / "data" / "input.QI_stel_seed_3127"
    if not input_file.exists():
        pytest.skip("Bundled QI seed input is unavailable")

    rc = main(
        [
            "--input",
            str(input_file),
            "--output-dir",
            str(tmp_path),
            "--radius",
            "0.01",
            "--n-random",
            "1",
            "--axis-count",
            "1",
            "--directions",
            "axes",
        ]
    )

    assert rc == 0
    plan = json.loads((tmp_path / "plan.json").read_text())
    assert plan["kind"] == "qi_basin_survey"
    assert plan["execute"] is False
    assert plan["radii"] == [0.01]
    assert plan["candidates"][0]["label"] == "zero"
    assert any(candidate["kind"] == "axis_positive" for candidate in plan["candidates"])


def test_basin_candidate_record_omits_zero_deltas() -> None:
    candidate = BasinCandidate(label="test", kind="unit", radius=0.1, params=(0.0, 0.2))

    assert candidate.as_record(["rc10", "zs10"])["deltas"] == {"zs10": 0.2}
