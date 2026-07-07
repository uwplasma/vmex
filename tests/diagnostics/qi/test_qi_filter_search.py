from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.diagnostics.qi.qi_filter_search import (
    FilterSearchOptions,
    SurveyTargets,
    filter_decision,
    filter_phase,
    gate_status,
    generate_trial_directions,
    main,
)


def _metrics(**overrides):
    data = {
        "qi_smooth_total": 1.0e-3,
        "qi_legacy_total": 5.0e-4,
        "qi_mirror_ratio_max": 0.25,
        "qi_max_elongation": 7.0,
        "mean_iota": 0.5,
        "aspect": 10.0,
    }
    data.update(overrides)
    return data


def test_gate_status_and_phase_order() -> None:
    targets = SurveyTargets()
    options = FilterSearchOptions()

    assert filter_phase(_metrics(qi_smooth_total=1.0e-2), targets=targets, options=options) == "qi"
    assert filter_phase(_metrics(mean_iota=0.01), targets=targets, options=options) == "iota"
    assert filter_phase(_metrics(qi_mirror_ratio_max=0.8), targets=targets, options=options) == "engineering"
    assert filter_phase(_metrics(), targets=targets, options=options) == "polish"
    assert gate_status(_metrics(), targets=targets, options=options)["qi"] is True


def test_filter_decision_accepts_only_active_gate_improvement() -> None:
    targets = SurveyTargets()
    options = FilterSearchOptions(min_iota_gain=1.0e-2)
    current = _metrics(mean_iota=0.02)

    accepted = filter_decision(current, _metrics(mean_iota=0.04), targets=targets, options=options)
    rejected = filter_decision(current, _metrics(mean_iota=0.025), targets=targets, options=options)
    lost_qi = filter_decision(
        current,
        _metrics(mean_iota=0.08, qi_smooth_total=1.0e-1),
        targets=targets,
        options=options,
    )

    assert accepted.accepted is True
    assert accepted.phase == "iota"
    assert rejected.accepted is False
    assert "iota gain" in rejected.reason
    assert lost_qi.accepted is False
    assert "QI" in lost_qi.reason


def test_filter_decision_engineering_preserves_iota() -> None:
    targets = SurveyTargets(mirror_ratio_max=0.35, max_elongation=8.0)
    options = FilterSearchOptions(min_engineering_gain=1.0e-3)
    current = _metrics(qi_mirror_ratio_max=0.8, qi_max_elongation=10.0, mean_iota=0.5)

    accepted = filter_decision(
        current,
        _metrics(qi_mirror_ratio_max=0.6, qi_max_elongation=9.0, mean_iota=0.45),
        targets=targets,
        options=options,
    )
    rejected = filter_decision(
        current,
        _metrics(qi_mirror_ratio_max=0.6, qi_max_elongation=9.0, mean_iota=0.1),
        targets=targets,
        options=options,
    )

    assert accepted.accepted is True
    assert rejected.accepted is False
    assert "iota" in rejected.reason


def test_filter_decision_polish_preserves_all_hard_gates() -> None:
    targets = SurveyTargets(mirror_ratio_max=0.35, max_elongation=8.0)
    current = _metrics(qi_smooth_total=1.0e-3, qi_legacy_total=5.0e-4, mean_iota=0.5)

    rejected = filter_decision(
        current,
        _metrics(qi_smooth_total=5.0e-4, qi_legacy_total=2.0e-4, mean_iota=0.1),
        targets=targets,
    )

    assert rejected.accepted is False
    assert rejected.phase == "polish"
    assert "iota" in rejected.reason


def test_generate_trial_directions_is_deterministic() -> None:
    rng = np.random.default_rng(3)
    directions = generate_trial_directions(
        names=["rc10", "zs10", "rc20"],
        x_scale=[1.0, 0.5, 0.1],
        axis_count=2,
        n_random=1,
        rng=rng,
        direction_families=("axes", "rademacher"),
    )

    labels = [label for label, _direction in directions]
    assert labels[:4] == ["axis+:rc10", "axis-:rc10", "axis+:zs10", "axis-:zs10"]
    assert labels[-1] == "rademacher:000"
    np.testing.assert_allclose(directions[0][1], [1.0, 0.0, 0.0])
    assert set(np.unique(directions[-1][1])) <= {-1.0, 1.0}


def test_cli_dry_run_writes_plan(tmp_path: Path) -> None:
    input_file = Path(__file__).resolve().parents[3] / "examples" / "data" / "input.QI_stel_seed_3127"
    if not input_file.exists():
        pytest.skip("Bundled QI seed input is unavailable")

    rc = main(
        [
            "--input",
            str(input_file),
            "--output-dir",
            str(tmp_path),
            "--max-iterations",
            "1",
            "--directions",
            "axes",
            "--axis-count",
            "1",
        ]
    )

    assert rc == 0
    plan = json.loads((tmp_path / "plan.json").read_text())
    assert plan["kind"] == "qi_filter_search"
    assert plan["execute"] is False
    assert plan["directions"] == ["axes"]
