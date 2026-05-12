from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "examples/optimization/audit_qi_seed_suitability.py"


def _load_module():
    pytest.importorskip("jax")
    spec = importlib.util.spec_from_file_location("audit_qi_seed_suitability", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_case_and_default_families_cover_seed_types():
    mod = _load_module()

    case = mod.parse_case("candidate_qi:QI:/tmp/input:/tmp/wout.nc")
    families = mod.parse_seed_families("QI,qp,simple")
    stage_modes = mod.parse_stage_modes("1,1,2,2,3")

    assert case.label == "candidate_qi"
    assert case.family == "qi"
    assert case.input_path == Path("/tmp/input")
    assert case.wout_path == Path("/tmp/wout.nc")
    assert families == ("qi", "qp", "simple")
    assert stage_modes == (1, 1, 2, 2, 3)

    default_cases, _skipped = mod.default_seed_cases()
    families = {case.family for case in default_cases}
    assert {"qi", "qh", "qa", "simple"}.issubset(families)
    if (mod.OMNIGENITY_ROOT / "wouts_QI/wout_nfp2_QI_fixed_resolution_final.nc").exists():
        assert "qp" in families


def test_constraint_status_flags_seed_quality():
    mod = _load_module()
    targets = mod.SuitabilityTargets(
        target_aspect=5.0,
        abs_iota_min=0.41,
        max_mirror_ratio=0.21,
        max_elongation=8.0,
    )

    good = mod._constraint_status(
        {
            "aspect": 5.2,
            "mean_iota": -0.45,
            "qi_mirror_ratio_max": 0.18,
            "qi_max_elongation": 7.5,
            "qi_smooth_total": 0.2,
            "qi_legacy_total": 0.3,
        },
        targets,
    )
    poor = mod._constraint_status(
        {
            "aspect": 8.5,
            "mean_iota": 0.12,
            "qi_mirror_ratio_max": 0.5,
            "qi_max_elongation": 11.0,
            "qi_smooth_total": None,
            "qi_legacy_total": None,
            "qi_smooth_error": "failed",
        },
        targets,
    )

    assert good["seed_suitability"] == "pass"
    assert good["failed_constraints"] == []
    assert poor["seed_suitability"] == "needs_attention"
    assert {"aspect", "iota", "mirror", "elongation", "smooth_qi", "legacy_qi"}.issubset(
        set(poor["failed_constraints"])
    )
    assert poor["constraint_score"] > good["constraint_score"]


def test_build_seed_audit_ranks_and_writes_csv(monkeypatch, tmp_path):
    mod = _load_module()
    targets = mod.SuitabilityTargets()
    cases = [
        mod.SeedCase("rough_qh", "qh", Path("/tmp/input_qh"), Path("/tmp/wout_qh.nc")),
        mod.SeedCase("better_qi", "qi", Path("/tmp/input_qi"), Path("/tmp/wout_qi.nc")),
    ]

    def fake_evaluate(case, **_kwargs):
        base = {
            "label": case.label,
            "family": case.family,
            "input": str(case.input_path),
            "wout": str(case.wout_path),
            "aspect": 5.0,
            "mean_iota": 0.45,
            "qi_mirror_ratio_max": 0.18,
            "qi_max_elongation": 7.0,
            "qi_legacy_total": 0.4,
        }
        if case.label == "better_qi":
            base["qi_smooth_total"] = 0.1
        else:
            base["qi_smooth_total"] = 0.5
            base["qi_mirror_ratio_max"] = 0.4
        base.update(mod._constraint_status(base, targets))
        return base

    monkeypatch.setattr(mod, "evaluate_seed_case", fake_evaluate)

    report = mod.build_seed_audit(
        cases=cases,
        skipped_defaults=[{"label": "missing_qp", "family": "qp", "missing": "/missing"}],
        surfaces=(0.5,),
        targets=targets,
        nphi=11,
        nalpha=5,
        n_bounce=5,
        nphi_out=21,
        mboz=6,
        nboz=6,
        phimin=0.0,
        mirror_ntheta=8,
        mirror_nphi=8,
        elongation_ntheta=8,
        elongation_nphi=4,
    )

    assert report["no_optimization"] is True
    assert report["cases"][0]["label"] == "better_qi"
    assert report["cases"][0]["suitability_rank"] == 1
    assert report["cases"][0]["qi_smooth_rank"] == 1
    assert report["cases"][0]["qi_seed_score"] < report["cases"][1]["qi_seed_score"]
    assert report["cases"][1]["seed_suitability"] == "needs_attention"
    assert report["skipped_defaults"][0]["family"] == "qp"
    assert report["resolution"]["include_bounce_endpoints"] is True

    csv_path = tmp_path / "audit.csv"
    mod._write_csv(report["cases"], csv_path)
    rows = list(csv.DictReader(csv_path.open()))
    assert rows[0]["label"] == "better_qi"
    assert float(rows[0]["qi_seed_score"]) < float(rows[1]["qi_seed_score"])
    assert rows[1]["failed_constraints"] == "mirror"


def test_build_seed_audit_can_select_best_well_phase(monkeypatch):
    mod = _load_module()
    targets = mod.SuitabilityTargets()
    case = mod.SeedCase("phase_sensitive_qi", "qi", Path("/tmp/input_qi"), Path("/tmp/wout_qi.nc"))

    monkeypatch.setattr(
        mod,
        "_phimin_candidates_for_case",
        lambda case, *, phimin, phimin_policy: (0.0, 0.5),
    )

    def fake_evaluate(case, **kwargs):
        phimin = float(kwargs["phimin"])
        smooth = 0.5 if phimin == 0.0 else 0.05
        record = {
            "label": case.label,
            "family": case.family,
            "input": str(case.input_path),
            "wout": str(case.wout_path),
            "aspect": 5.0,
            "mean_iota": 0.45,
            "qi_phimin": phimin,
            "qi_smooth_total": smooth,
            "qi_legacy_total": smooth,
            "qi_mirror_ratio_max": 0.18,
            "qi_max_elongation": 7.0,
        }
        record.update(mod._constraint_status(record, targets))
        return record

    monkeypatch.setattr(mod, "evaluate_seed_case", fake_evaluate)

    report = mod.build_seed_audit(
        cases=[case],
        skipped_defaults=[],
        surfaces=(0.5,),
        targets=targets,
        nphi=11,
        nalpha=5,
        n_bounce=5,
        nphi_out=21,
        mboz=6,
        nboz=6,
        phimin=0.0,
        mirror_ntheta=8,
        mirror_nphi=8,
        elongation_ntheta=8,
        elongation_nphi=4,
        phimin_policy="well-phase",
    )

    selected = report["cases"][0]
    assert selected["selected_phimin"] == 0.5
    assert selected["qi_seed_score"] == pytest.approx(0.1)
    assert selected["phimin_candidates"] == [0.0, 0.5]
    assert [row["qi_phimin"] for row in selected["phimin_candidate_metrics"]] == [0.0, 0.5]


def test_prefine_probe_manifest_selects_top_rows_and_stays_dry(tmp_path):
    mod = _load_module()
    report = {
        "resolution": {"include_bounce_endpoints": True},
        "cases": [
            {
                "suitability_rank": 1,
                "label": "better qi",
                "family": "qi",
                "input": "/tmp/input_qi",
                "wout": "/tmp/wout_qi.nc",
                "qi_seed_score": 0.2,
                "qi_smooth_total": 0.1,
                "qi_legacy_total": 0.1,
                "selected_phimin": 0.5,
                "phimin_policy": "well-phase",
                "phimin_candidates": [0.0, 0.5],
                "failed_constraints": [],
            },
            {
                "suitability_rank": 2,
                "label": "rough/qh",
                "family": "qh",
                "input": "/tmp/input_qh",
                "wout": "/tmp/wout_qh.nc",
                "qi_seed_score": 1.5,
                "failed_constraints": ["mirror"],
            },
        ]
    }
    config = mod.QIPrefineProbeConfig(top_n=2, output_dir=tmp_path / "probes")

    manifest = mod.build_qi_prefine_probe_manifest(
        report,
        config=config,
        manifest_path=tmp_path / "manifest.json",
        dry_run=True,
    )

    assert manifest["dry_run"] is True
    assert manifest["selection"]["planned_rows"] == 2
    assert manifest["selection"]["covered_families"] == ["qh", "qi"]
    assert [plan["status"] for plan in manifest["plans"]] == ["planned", "planned"]
    assert manifest["plans"][0]["label"] == "better qi"
    assert manifest["plans"][0]["selection_reasons"] == ["top_n", "family_representative"]
    assert manifest["plans"][0]["representative_families"] == ["qi"]
    assert manifest["plans"][0]["output_dir"].endswith("01_better_qi")
    assert "--prefine-probes run" in manifest["plans"][0]["run_command"]
    assert "--prefine-reviewed" in manifest["plans"][0]["run_command"]
    assert manifest["plans"][0]["optimization"]["max_nfev"] <= mod.MAX_PREFINE_MAX_NFEV
    assert manifest["plans"][0]["optimization"]["stage_modes"] == [1, 1, 2, 2, 3]
    assert manifest["plans"][0]["optimization"]["stage_count"] == 5
    assert manifest["plans"][0]["optimization"]["total_nfev_cap"] == 6
    assert manifest["plans"][0]["qi_options"]["nphi"] <= mod.MAX_PREFINE_QI_NPHI
    assert manifest["plans"][0]["qi_options"]["include_bounce_endpoints"] is True
    assert manifest["plans"][0]["qi_options"]["endpoint_mode"] == "include_bounce_endpoints"
    assert manifest["plans"][0]["qi_options"]["phimin"] == 0.5
    assert manifest["plans"][0]["phimin"] == {
        "value": 0.5,
        "source": "audit_selected_phimin",
        "audit_policy": "well-phase",
        "audit_candidates": [0.0, 0.5],
    }
    assert manifest["plans"][0]["endpoint_mode"] == "include_bounce_endpoints"
    assert manifest["plans"][0]["endpoint_alignment"] == {
        "audit_include_bounce_endpoints": True,
        "prefine_include_bounce_endpoints": True,
        "aligned": True,
        "endpoint_mode": "include_bounce_endpoints",
    }
    assert manifest["plans"][0]["stages"][0]["mode"] == 1
    assert manifest["plans"][0]["stages"][1]["repeat_index_for_mode"] == 2
    assert manifest["plans"][0]["stages"][-1]["mode"] == 3
    assert manifest["plans"][0]["stages"][-1]["nfev_cap"] == 2
    assert manifest["plans"][0]["caps"]["per_probe_total_nfev"] == 6
    assert manifest["hard_caps"]["stage_count"] == mod.MAX_PREFINE_STAGE_COUNT
    assert manifest["effective_caps"]["per_probe_total_nfev"] == 6
    assert manifest["endpoint_alignment"]["aligned"] is True
    assert manifest["review"]["status"] == "requires_review"
    assert manifest["summary"]["statuses"] == {"planned": 2}
    assert manifest["summary"]["recommendation"]["action"] == "review_manifest"
    assert manifest["summary"]["recommendation"]["label"] == "better qi"
    assert manifest["summary"]["best_candidate_by_final_objective"] is None

    bad_config = mod.QIPrefineProbeConfig(top_n=mod.MAX_PREFINE_TOP_N + 1)
    with pytest.raises(ValueError, match="top_n"):
        mod.build_qi_prefine_probe_manifest(
            report,
            config=bad_config,
            manifest_path=tmp_path / "bad.json",
            dry_run=True,
        )

    reversed_stage_config = mod.QIPrefineProbeConfig(stage_modes=(1, 2, 1), max_mode=2)
    with pytest.raises(ValueError, match="nondecreasing"):
        mod.build_qi_prefine_probe_manifest(
            report,
            config=reversed_stage_config,
            manifest_path=tmp_path / "bad_stages.json",
            dry_run=True,
        )


def test_prefine_probe_manifest_adds_family_representatives_after_top_n(tmp_path):
    mod = _load_module()
    report = {
        "cases": [
            {
                "suitability_rank": 1,
                "label": "best_qi",
                "family": "qi",
                "input": "/tmp/input_qi",
                "wout": "/tmp/wout_qi.nc",
                "qi_seed_score": 0.1,
            },
            {
                "suitability_rank": 2,
                "label": "second_qi",
                "family": "qi",
                "input": "/tmp/input_qi2",
                "wout": "/tmp/wout_qi2.nc",
                "qi_seed_score": 0.2,
            },
            {
                "suitability_rank": 3,
                "label": "best_qp",
                "family": "qp",
                "input": "/tmp/input_qp",
                "wout": "/tmp/wout_qp.nc",
                "qi_seed_score": 0.3,
            },
            {
                "suitability_rank": 4,
                "label": "best_qh",
                "family": "qh",
                "input": "/tmp/input_qh",
                "wout": "/tmp/wout_qh.nc",
                "qi_seed_score": 0.4,
            },
            {
                "suitability_rank": 5,
                "label": "best_qa",
                "family": "qa",
                "input": "/tmp/input_qa",
                "wout": "/tmp/wout_qa.nc",
                "qi_seed_score": 0.5,
            },
            {
                "suitability_rank": 6,
                "label": "best_simple",
                "family": "simple",
                "input": "/tmp/input_simple",
                "wout": "/tmp/wout_simple.nc",
                "qi_seed_score": 0.6,
            },
        ]
    }

    manifest = mod.build_qi_prefine_probe_manifest(
        report,
        config=mod.QIPrefineProbeConfig(top_n=1, output_dir=tmp_path / "probes"),
        manifest_path=tmp_path / "manifest.json",
        dry_run=True,
    )

    assert [plan["label"] for plan in manifest["plans"]] == [
        "best_qi",
        "best_qp",
        "best_qh",
        "best_qa",
        "best_simple",
    ]
    assert manifest["selection"]["planned_rows"] == 5
    assert manifest["selection"]["top_rows"] == 1
    assert manifest["selection"]["covered_families"] == ["qa", "qh", "qi", "qp", "simple"]
    assert manifest["plans"][0]["selection_reasons"] == ["top_n", "family_representative"]
    assert manifest["plans"][1]["selection_reasons"] == ["family_representative"]
    assert manifest["plans"][1]["representative_families"] == ["qp"]

    top_only = mod.build_qi_prefine_probe_manifest(
        report,
        config=mod.QIPrefineProbeConfig(
            top_n=2,
            include_family_representatives=False,
            output_dir=tmp_path / "top_only",
        ),
        manifest_path=tmp_path / "top_only_manifest.json",
        dry_run=True,
    )
    assert [plan["label"] for plan in top_only["plans"]] == ["best_qi", "second_qi"]
    assert top_only["selection"]["covered_families"] == []


def test_run_qi_prefine_probe_dispatches_tiny_qi_solve(tmp_path):
    mod = _load_module()
    report = {
        "cases": [
            {
                "suitability_rank": 1,
                "label": "candidate_qi",
                "family": "qi",
                "input": "/tmp/input_qi",
                "wout": "/tmp/wout_qi.nc",
                "qi_seed_score": 0.2,
            },
        ]
    }
    manifest = mod.build_qi_prefine_probe_manifest(
        report,
        config=mod.QIPrefineProbeConfig(output_dir=tmp_path / "probes"),
        manifest_path=tmp_path / "manifest.json",
        dry_run=False,
    )
    calls = {}

    class FakeQuasiIsodynamicOptions:
        def __init__(self, **kwargs):
            calls["qi_options"] = kwargs

    class FakeQuasiIsodynamicResidual:
        def __init__(self, options):
            self.options = options

        def J(self, _ctx, _state):
            return 0.0

    class FakeLeastSquaresProblem:
        @classmethod
        def from_tuples(cls, tuples):
            calls["tuples"] = tuples
            return "problem"

    class FakeFixedBoundaryVMEC:
        @classmethod
        def from_input(cls, input_file, **kwargs):
            calls["from_input"] = {"input_file": input_file, **kwargs}
            return "vmec"

    def fake_least_squares_solve(vmec, problem, **kwargs):
        calls["solve"] = {"vmec": vmec, "problem": problem, **kwargs}
        return SimpleNamespace(
            final_result={
                "_history_dump": {
                    "objective_initial": 3.0,
                    "objective_final": 1.0,
                    "qs_final": 0.4,
                    "total_wall_time_s": 0.5,
                    "nfev": 3,
                    "njev": 2,
                    "success": True,
                    "message": "ok",
                    "history": [
                        {"objective": 3.0},
                        {"objective": 2.0},
                        {"objective": 1.0},
                    ],
                }
            },
            stage_modes=[1, 1, 2, 2, 3],
        )

    fake_workflow = SimpleNamespace(
        QuasiIsodynamicOptions=FakeQuasiIsodynamicOptions,
        QuasiIsodynamicResidual=FakeQuasiIsodynamicResidual,
        LeastSquaresProblem=FakeLeastSquaresProblem,
        FixedBoundaryVMEC=FakeFixedBoundaryVMEC,
        least_squares_solve=fake_least_squares_solve,
    )

    completed = mod.run_qi_prefine_probe(manifest["plans"][0], workflow=fake_workflow)

    assert completed["status"] == "completed"
    assert completed["result"]["objective_final"] == 1.0
    assert completed["result"]["requested_stage_modes"] == [1, 1, 2, 2, 3]
    assert completed["result"]["completed_stage_modes"] == [1, 1, 2, 2, 3]
    assert completed["result"]["stage_count_requested"] == 5
    assert completed["result"]["total_nfev_cap"] == 6
    assert completed["result"]["endpoint_mode"] == "include_bounce_endpoints"
    assert completed["result"]["history_summary"]["history_present"] is True
    assert completed["result"]["history_summary"]["objective_monotonic_nonincreasing"] is True
    assert completed["result"]["history_summary"]["objective_regression_count"] == 0
    assert completed["result"]["nfev"] == 3
    assert calls["qi_options"]["nphi"] == 31
    assert calls["qi_options"]["include_bounce_endpoints"] is True
    assert calls["from_input"]["max_mode"] == 3
    assert calls["solve"]["max_nfev"] == 2
    assert calls["solve"]["stage_modes"] == (1, 1, 2, 2, 3)
    assert calls["solve"]["save_stage_wouts"] is False


def test_prefine_probe_manifest_run_can_require_review(tmp_path):
    mod = _load_module()
    manifest = mod.build_qi_prefine_probe_manifest(
        {
            "cases": [
                {
                    "suitability_rank": 1,
                    "label": "candidate_qi",
                    "family": "qi",
                    "input": "/tmp/input_qi",
                    "wout": "/tmp/wout_qi.nc",
                    "qi_seed_score": 0.2,
                },
            ]
        },
        config=mod.QIPrefineProbeConfig(output_dir=tmp_path / "probes"),
        manifest_path=tmp_path / "manifest.json",
        dry_run=False,
    )

    with pytest.raises(ValueError, match="reviewed manifest"):
        mod.run_qi_prefine_probe_manifest(manifest, require_review=True, workflow=SimpleNamespace())


def test_prefine_probe_summary_ranks_acceptance_failures_and_regressions():
    mod = _load_module()
    manifest = {
        "dry_run": False,
        "selection": {"planned_rows": 5},
        "plans": [
            {
                "status": "completed",
                "label": "bigger_improvement_qi",
                "family": "qi",
                "audit_rank": 1,
                "optimization": {"stage_modes": [1, 1, 2, 2, 3]},
                "result": {
                    "objective_initial": 10.0,
                    "objective_final": 3.0,
                    "requested_stage_modes": [1, 1, 2, 2, 3],
                    "completed_stage_modes": [1, 1, 2, 2, 3],
                    "history": [
                        {"objective": 10.0},
                        {"objective": 7.0},
                        {"objective": 3.0},
                    ],
                },
            },
            {
                "status": "completed",
                "label": "best_final_qp",
                "family": "qp",
                "audit_rank": 2,
                "optimization": {"stage_modes": [1, 1, 2, 2, 3]},
                "result": {
                    "objective_initial": 5.0,
                    "objective_final": 2.0,
                    "requested_stage_modes": [1, 1, 2, 2, 3],
                    "completed_stage_modes": [1, 1, 2, 2, 3],
                    "history": [
                        {"objective": 5.0},
                        {"objective": 3.0},
                        {"objective": 4.0},
                        {"objective": 2.0},
                    ],
                },
            },
            {
                "status": "failed",
                "label": "failed_qh",
                "family": "qh",
                "error_type": "RuntimeError",
                "error": "linear solve failed",
            },
            {
                "status": "failed",
                "label": "timeout_qa",
                "family": "qa",
                "error_type": "TimeoutError",
                "error": "worker timed out after 1200.0 s",
            },
            {
                "status": "pending",
                "label": "pending_simple",
                "family": "simple",
                "optimization": {"stage_modes": [1, 1, 2, 2, 3]},
            },
        ],
    }

    summary = mod.summarize_qi_prefine_probe_manifest(manifest)

    assert summary["statuses"] == {"completed": 2, "failed": 2, "pending": 1}
    assert summary["completed_count"] == 2
    assert summary["completed_stage_modes"][0]["completed_stage_modes"] == [1, 1, 2, 2, 3]
    assert summary["best_candidate_by_final_objective"]["label"] == "best_final_qp"
    assert summary["best_improvement"]["label"] == "bigger_improvement_qi"
    assert summary["failure_count"] == 2
    assert summary["timeout_count"] == 1
    assert summary["timeouts"][0]["label"] == "timeout_qa"
    assert summary["history_regression_plan_count"] == 1
    assert summary["history_regressions"][0]["label"] == "best_final_qp"
    assert summary["history_regressions"][0]["history_summary"]["objective_regression_count"] == 1
    assert summary["accepted_candidate"]["label"] == "bigger_improvement_qi"
    assert summary["acceptance"]["accepted"] is False
    assert "timeout" in summary["acceptance"]["blocking_issues"][0]
    assert summary["recommendation"]["action"] == "inspect_timeout"
    assert summary["recommendation"]["label"] == "timeout_qa"


def test_prefine_result_summary_ranks_and_flags_regressions():
    mod = _load_module()
    manifest = {
        "dry_run": False,
        "selection": {"planned_rows": 4},
        "plans": [
            {
                "status": "completed",
                "label": "qi_seed",
                "family": "qi",
                "result": {
                    "objective_initial": 5.0,
                    "objective_final": 1.0,
                    "qi_final": 0.2,
                    "wall_time_s": 3.0,
                    "requested_stage_modes": [1, 1, 2],
                    "completed_stage_modes": [1, 1, 2],
                    "objective_history": [5.0, 2.5, 1.0],
                },
            },
            {
                "status": "completed",
                "label": "qp_seed",
                "family": "qp",
                "result": {
                    "objective_initial": 10.0,
                    "objective_final": 2.0,
                    "qi_final": 0.3,
                    "wall_time_s": 2.0,
                    "requested_stage_modes": [1, 1, 2],
                    "completed_stage_modes": [1, 1],
                    "objective_history": [10.0, 7.0, 7.5, 2.0],
                },
            },
            {"status": "failed", "label": "qa_seed", "family": "qa", "error_type": "RuntimeError", "error": "bad"},
            {"label": "planned_seed", "family": "simple"},
        ],
    }

    summary = mod.summarize_qi_prefine_results(manifest)

    assert summary["dry_run"] is False
    assert summary["planned_rows"] == 4
    assert summary["completed_count"] == 2
    assert summary["failure_count"] == 1
    assert summary["statuses"] == {"completed": 2, "failed": 1, "unknown": 1}
    assert summary["completed_stage_modes"][0]["completed_stage_modes"] == [1, 1, 2]
    assert summary["best_candidate_by_final_objective"]["label"] == "qi_seed"
    assert summary["best_improvement"]["label"] == "qp_seed"
    assert summary["history_regressions"][0]["label"] == "qp_seed"
    assert summary["recommendation"]["action"] == "inspect_failure"


def test_prefine_manifest_run_attaches_summary_on_success(tmp_path):
    mod = _load_module()
    manifest = {
        "dry_run": False,
        "review": {"operator_confirmed": True, "status": "reviewed"},
        "selection": {"planned_rows": 1},
        "plans": [{"label": "candidate_qi", "family": "qi", "result": {}}],
    }

    def fake_run(plan, *, workflow=None):
        del workflow
        return {
            **plan,
            "status": "completed",
            "result": {
                "objective_initial": 4.0,
                "objective_final": 1.0,
                "completed_stage_modes": [1, 2, 3],
            },
        }

    original = mod.run_qi_prefine_probe
    try:
        mod.run_qi_prefine_probe = fake_run
        executed = mod.run_qi_prefine_probe_manifest(manifest, require_review=True, workflow=SimpleNamespace())
    finally:
        mod.run_qi_prefine_probe = original

    assert executed["result_summary"]["completed_count"] == 1
    assert executed["result_summary"]["best_candidate_by_final_objective"]["objective_final"] == 1.0
