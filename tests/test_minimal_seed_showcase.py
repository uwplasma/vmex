from __future__ import annotations

import importlib.util
from dataclasses import fields
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, script_name: str):
    script = ROOT / "examples" / "optimization" / script_name
    spec = importlib.util.spec_from_file_location(name, script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_minimal_seed_showcase_case_map_uses_three_coefficient_inputs() -> None:
    generator = _load_module("generate_minimal_seed_showcase", "generate_minimal_seed_showcase.py")

    case_fields = {field.name for field in fields(generator.MinimalSeedCase)}
    assert {"default_max_mode", "default_use_ess", "default_policy"}.isdisjoint(case_fields)
    assert generator.DEFAULT_CASE_ORDER == ("qi_nfp1", "qi_nfp2", "qi_nfp3", "qa_nfp2", "qh_nfp4", "qp_nfp2")
    for case_name in generator.DEFAULT_CASE_ORDER:
        case = generator.SHOWCASE_CASES[case_name]
        text = Path(case.input_file).read_text()
        assert f"NFP = {case.nfp}" in text
        assert "RBC(0,0)" in text
        assert "RBC(0,1)" in text
        assert "ZBS(0,1)" in text
        assert text.count("RBC(") == 2
        assert text.count("ZBS(") == 1
        assert "RBS(" not in text
        assert "ZBC(" not in text


def test_minimal_seed_showcase_config_patch_is_bounded_and_non_mutating() -> None:
    generator = _load_module("generate_minimal_seed_showcase_cfg", "generate_minimal_seed_showcase.py")
    budget = generator.MinimalSeedBudget(
        max_nfev=2,
        continuation_nfev=3,
        inner_max_iter=11,
        inner_ftol=1.0e-8,
        trial_max_iter=12,
        trial_ftol=2.0e-8,
    )
    case = generator.SHOWCASE_CASES["qi_nfp3"]
    original = generator.sweep.PROBLEM_CONFIGS["qi"]

    patched = generator._problem_config_for_case(case, max_mode=3, budget=budget)

    assert generator.sweep.PROBLEM_CONFIGS["qi"] is original
    assert patched.input_file.name == "input.minimal_seed_nfp3"
    assert patched.max_nfev == 2
    assert patched.continuation_nfev == 3
    assert patched.inner_max_iter == 11
    assert patched.trial_max_iter == 12
    assert patched.project_input_boundary_to_max_mode is True
    assert patched.min_vmec_mode >= 5
    assert patched.qi_preseed_qp is True


def test_minimal_seed_renderer_loads_records_and_returns_monotone_segments(tmp_path: Path) -> None:
    generator = _load_module("generate_minimal_seed_showcase_for_renderer", "generate_minimal_seed_showcase.py")
    renderer = _load_module("render_minimal_seed_showcase", "render_minimal_seed_showcase.py")

    output_dir = tmp_path / "cpu" / "qa_nfp2" / "continuation" / "mode3" / "ess"
    output_dir.mkdir(parents=True)
    case = generator.SHOWCASE_CASES["qa_nfp2"]
    (output_dir / "showcase_case.json").write_text(
        json.dumps(
            {
                "minimal_seed_case": {
                    "name": case.name,
                    "problem": case.problem,
                    "nfp": case.nfp,
                    "input_file": str(case.input_file),
                },
                "policy": "continuation",
                "max_mode": 3,
                "use_ess": True,
            }
        )
    )
    (output_dir / "case_result.json").write_text(
        json.dumps(
            {
                "backend": "cpu",
                "problem": "qa",
                "max_mode": 3,
                "use_ess": True,
                "success": True,
                "crashed": False,
                "objective_final": 2.0,
                "aspect_final": 5.0,
                "iota_final": 0.42,
                "total_wall_time_s": 123.0,
                "policy": "continuation",
                "output_dir": str(output_dir),
            }
        )
    )
    (output_dir / "history.json").write_text(
        json.dumps(
            {
                "history": [
                    {"stage": "QA mode 1", "wall_time_s": 0.0, "objective": 10.0},
                    {"stage": "QA mode 1", "wall_time_s": 10.0, "objective": 8.0},
                    {"stage": "QA mode 1", "wall_time_s": 20.0, "objective": 9.0},
                    {"stage": "QA mode 3", "wall_time_s": 30.0, "objective": 7.0},
                    {"stage": "QA mode 3", "wall_time_s": 40.0, "objective": 7.5},
                    {"stage": "QA mode 3", "wall_time_s": 50.0, "objective": 4.0},
                ]
            }
        )
    )

    records = renderer.best_records(renderer.load_records(tmp_path))
    assert len(records) == 1
    assert records[0].case_name == "qa_nfp2"

    failed_dir = tmp_path / "cpu" / "qh_nfp4" / "continuation" / "mode3" / "ess"
    failed_dir.mkdir(parents=True)
    (failed_dir / "showcase_case.json").write_text(
        json.dumps(
            {
                "minimal_seed_case": {
                    "name": "qh_nfp4",
                    "problem": "qh",
                    "nfp": 4,
                    "input_file": "input.minimal_seed_nfp4",
                },
                "policy": "continuation",
                "max_mode": 3,
                "use_ess": True,
            }
        )
    )
    (failed_dir / "case_result.json").write_text(
        json.dumps(
            {
                "backend": "cpu",
                "problem": "qh",
                "max_mode": 3,
                "use_ess": True,
                "success": False,
                "crashed": True,
                "policy": "continuation",
                "output_dir": str(failed_dir),
            }
        )
    )
    all_records = renderer.best_records(renderer.load_records(tmp_path), successful_only=False)
    assert [record.case_name for record in all_records] == ["qa_nfp2", "qh_nfp4"]

    segments = renderer.objective_segments(records[0])
    assert len(segments) == 2
    np.testing.assert_allclose(segments[0][1], [10.0, 8.0, 8.0])
    np.testing.assert_allclose(segments[1][1], [7.0, 7.0, 4.0])

    summary = tmp_path / "summary.csv"
    renderer.write_summary_csv(records, summary)
    assert "qa_nfp2" in summary.read_text()
