from __future__ import annotations

import builtins
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import vmec_jax.qi_optimization as qio


def test_load_basin_prefilter_tools_reports_source_checkout_requirement(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def blocked_import(name, globals_=None, locals_=None, fromlist=(), level=0):  # noqa: ANN001
        if name in {"tools.diagnostics.qi_basin_survey", "tools.diagnostics.qi_landscape_scan"}:
            raise ModuleNotFoundError(name)
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    with pytest.raises(RuntimeError, match="source checkout"):
        qio._load_basin_prefilter_tools()


def test_target_helicity_seed_terms_are_deterministic_and_disable_cleanly() -> None:
    expected_mode_1 = (
        ("RBC", (1, 0), 2.5e-5),
        ("ZBS", (1, 0), 2.5e-5),
        ("RBC", (-1, 1), 2.5e-5),
        ("ZBS", (-1, 1), 2.5e-5),
        ("RBC", (1, 1), 2.5e-5),
        ("ZBS", (1, 1), 2.5e-5),
    )

    assert qio.target_helicity_seed_terms(max_mode=0) == ()
    assert qio.target_helicity_seed_terms(max_mode=3, amplitude=0.0) == ()
    assert qio.target_helicity_seed_terms(max_mode=1, amplitude=2.5e-5) == expected_mode_1


def test_target_helicity_seed_preconditioner_returns_input_when_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_file = tmp_path / "input.seed"

    def fail_read(_path):
        pytest.fail("disabled target-helicity seed should not read input")

    monkeypatch.setattr(qio.vj, "read_indata", fail_read)

    assert qio.run_target_helicity_seed_preconditioner(input_file, tmp_path, {"enabled": False}) == input_file


def test_target_helicity_seed_preconditioner_writes_seeded_input_and_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_file = tmp_path / "input.seed"
    source = SimpleNamespace(
        scalars={"NFP": 2},
        indexed={
            "RBC": {(1, 0): 9.0},
            "ZBS": {},
        },
    )
    writes = {}

    monkeypatch.setattr(qio.vj, "read_indata", lambda path: source)
    monkeypatch.setattr(qio.vj, "write_indata", lambda path, indata: writes.update(path=Path(path), indata=indata))

    out = qio.run_target_helicity_seed_preconditioner(
        input_file,
        tmp_path,
        {
            "only_if_abs_below": 1.0e-8,
            "terms": (
                ("rbc", (1, 0), 1.0e-5),
                ("zbs", (1, 0), 2.0e-5),
                ("rbc", (2, -1), 3.0e-5),
            ),
        },
    )

    metadata = json.loads((tmp_path / "target_helicity_seed" / "metadata.json").read_text())
    seeded = writes["indata"]

    assert out == tmp_path / "target_helicity_seed" / "input.target_helicity_seed"
    assert writes["path"] == out
    assert seeded.scalars == {"NFP": 2}
    assert seeded.source_path == str(input_file)
    assert seeded.indexed["RBC"][(1, 0)] == 9.0
    assert seeded.indexed["RBC"][(2, -1)] == 3.0e-5
    assert seeded.indexed["ZBS"][(1, 0)] == 2.0e-5
    assert metadata["inserted"] == [
        {"family": "ZBS", "n": 1, "m": 0, "value": 2.0e-5},
        {"family": "RBC", "n": 2, "m": -1, "value": 3.0e-5},
    ]
    assert len(metadata["terms"]) == 3


def test_jsonable_parse_float_sequence_and_partial_history_helpers() -> None:
    class UncooperativeArray:
        def __array__(self):
            raise TypeError("not array-like")

        def __str__(self) -> str:
            return "fallback-string"

    history = {
        "objective_final": np.float64(1.25),
        "qs_final": np.float64(2.5e-3),
        "aspect_final": np.float64(7.5),
        "iota_final": np.float64(-0.42),
        "nfev": np.int64(8),
        "njev": np.int64(3),
        "total_wall_time_s": np.float64(12.0),
    }
    partial = qio._partial_diagnostics_from_history(history, {})
    existing = {"qi_smooth_total": 1.0e-3}

    assert qio._parse_float_sequence(None, name="radii") is None
    assert qio._parse_float_sequence("", name="radii") is None
    assert qio._parse_float_sequence("1, 2.5 3", name="radii") == (1.0, 2.5, 3.0)
    with pytest.raises(ValueError, match="radii must be"):
        qio._parse_float_sequence("1, nope", name="radii")
    assert qio._partial_diagnostics_from_history(history, existing) == {
        "qi_smooth_total": 1.0e-3,
        "objective_final": np.float64(1.25),
        "qs_final": np.float64(2.5e-3),
        "aspect": np.float64(7.5),
        "mean_iota": np.float64(-0.42),
        "nfev": np.int64(8),
        "njev": np.int64(3),
        "total_wall_time_s": np.float64(12.0),
        "partial": True,
    }
    assert partial == {
        "objective_final": np.float64(1.25),
        "qs_final": np.float64(2.5e-3),
        "aspect": np.float64(7.5),
        "mean_iota": np.float64(-0.42),
        "nfev": np.int64(8),
        "njev": np.int64(3),
        "total_wall_time_s": np.float64(12.0),
        "partial": True,
        "diagnostics_pending": True,
    }
    assert qio.jsonable(
        {
            "path": Path("input.demo"),
            "finite_scalar": np.float64(4.0),
            "nonfinite_scalar": float("inf"),
            "array": np.asarray([1.0, np.nan]),
            "nested": (np.int64(2), UncooperativeArray()),
        }
    ) == {
        "path": "input.demo",
        "finite_scalar": 4.0,
        "nonfinite_scalar": None,
        "array": [1.0, None],
        "nested": [2, "fallback-string"],
    }
    assert qio.diagnostic_float({"aspect": np.float64(6.0)}, "aspect") == pytest.approx(6.0)
    assert np.isnan(qio.diagnostic_float({}, "missing"))


def test_basin_prefilter_score_weights_qi_iota_and_engineering_terms() -> None:
    targets = SimpleNamespace(
        smooth_qi_max=2.0,
        legacy_qi_max=4.0,
        mirror_ratio_max=0.3,
        max_elongation=5.0,
        abs_iota_min=0.4,
        target_aspect=10.0,
    )
    metrics = {
        "qi_smooth_total": 1.0,
        "qi_legacy_total": 2.0,
        "qi_mirror_ratio_max": 0.6,
        "qi_max_elongation": 6.0,
        "mean_iota": 0.2,
        "aspect": 12.0,
    }
    config = {
        "qi_weight": 2.0,
        "iota_gap_weight": 3.0,
        "mirror_weight": 0.25,
        "elongation_weight": 0.1,
        "aspect_weight": 0.1,
    }

    assert qio.basin_prefilter_score(metrics, targets, config) == pytest.approx(3.79)


def test_stage_modes_for_uses_limits_explicit_modes_and_default_repetition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(qio, "MAX_MODE", 4, raising=False)
    monkeypatch.setattr(qio, "USE_MODE_CONTINUATION", True, raising=False)
    monkeypatch.setattr(qio, "CONTINUATION_NFEV", 2, raising=False)
    monkeypatch.setattr(qio, "STAGE_REPEATS", 3, raising=False)
    monkeypatch.setattr(qio.vj, "normalize_boundary_mode_limits", lambda mode: f"normalised:{mode}")

    def repeated_stage_modes(*, max_mode, use_mode_continuation, continuation_nfev, repeats):
        return [max_mode, use_mode_continuation, continuation_nfev, repeats]

    monkeypatch.setattr(qio.vj, "repeated_stage_modes", repeated_stage_modes)

    assert qio.stage_modes_for({"stage_mode_limits": ("nfirst", (2, 3))}) == [
        "normalised:nfirst",
        "normalised:(2, 3)",
    ]
    assert qio.stage_modes_for({"stage_modes": (1, "2", 3)}) == [1, 2, 3]
    assert qio.stage_modes_for({}) == [4, True, 2, 3]
    assert qio.stage_modes_for({"stage_repeats": 5}) == [4, True, 2, 5]


def test_promotion_scores_apply_gate_penalties_and_nonfinite_fallbacks() -> None:
    gated = {
        "qi_seed_gate_passed": True,
        "qi_engineering_gate_passed": True,
        "qi_rank_score": 1.0,
        "qi_constraint_score": 4.0,
        "qi_mirror_ratio_max": 0.2,
    }
    ungated = {
        "qi_seed_gate_passed": False,
        "qi_engineering_gate_passed": False,
        "qi_rank_score": 1.0,
        "qi_constraint_score": 4.0,
        "qi_mirror_ratio_max": 0.2,
    }

    assert qio.promotion_score(gated) == pytest.approx(2.0)
    assert qio.engineering_promotion_score(gated) == pytest.approx(2.4)
    assert qio.promotion_score(ungated) == pytest.approx(112.0)
    assert qio.engineering_promotion_score(ungated) == pytest.approx(1102.4)
    assert np.isinf(qio.promotion_score({"qi_seed_gate_passed": True, "qi_rank_score": "bad"}))
    assert np.isinf(qio.engineering_promotion_score({"qi_mirror_ratio_max": np.nan}))


def test_stage_promotes_candidate_accepts_first_baseline_when_reference_is_missing() -> None:
    promotion = {
        "qi_cleanup_promoted": True,
        "qi_cleanup_rejection_reasons": [],
    }

    out = qio.stage_promotes_candidate(
        {
            "accept_if_rank_improves": True,
            "accept_if_engineering_score_improves": True,
        },
        promotion,
        None,
    )

    assert out is promotion


def test_stage_promotes_candidate_rejects_failed_iota_relaxed_qi_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(qio, "QI_GATE_SMOOTH_MAX", 1.0e-3, raising=False)
    monkeypatch.setattr(qio, "QI_GATE_LEGACY_MAX", 1.0e-3, raising=False)
    stage = {
        "accept_if_iota_improves": True,
        "iota_improvement_min": 0.10,
        "qi_relax_for_iota": 1.5,
    }
    promotion = {
        "qi_cleanup_promoted": True,
        "qi_cleanup_rejection_reasons": [],
        "mean_iota": 0.45,
        "qi_smooth_total": 1.0e-3,
        "qi_legacy_total": 1.0e-3,
    }
    reference = {
        "mean_iota": 0.40,
        "qi_smooth_total": 1.0e-3,
        "qi_legacy_total": 1.0e-3,
    }

    out = qio.stage_promotes_candidate(stage, promotion, reference)

    assert out["qi_cleanup_promoted"] is False
    assert "iota ramp did not satisfy relaxed QI promotion" in out["qi_cleanup_rejection_reasons"][0]


def test_stage_promotes_candidate_keeps_candidate_when_rank_and_engineering_improve() -> None:
    promotion = {
        "qi_cleanup_promoted": True,
        "qi_cleanup_rejection_reasons": [],
        "qi_seed_gate_passed": True,
        "qi_engineering_gate_passed": True,
        "qi_rank_score": 0.7,
        "qi_constraint_score": 1.0,
        "qi_mirror_ratio_max": 0.20,
    }
    reference = {
        "qi_seed_gate_passed": True,
        "qi_engineering_gate_passed": True,
        "qi_rank_score": 1.0,
        "qi_constraint_score": 1.0,
        "qi_mirror_ratio_max": 0.26,
    }

    out = qio.stage_promotes_candidate(
        {
            "accept_if_rank_improves": True,
            "rank_score_relax": 0.0,
            "accept_if_engineering_score_improves": True,
            "engineering_score_relax": 0.0,
            "mirror_improvement_min": 0.05,
        },
        promotion,
        reference,
    )

    assert out is promotion
