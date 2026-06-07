from __future__ import annotations

import csv
import importlib.util
import json
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
    all_surface = mod.parse_optional_surface_index("all")
    indexed_surface = mod.parse_optional_surface_index("2")

    assert case.label == "candidate_qi"
    assert case.family == "qi"
    assert case.input_path == Path("/tmp/input")
    assert case.wout_path == Path("/tmp/wout.nc")
    assert families == ("qi", "qp", "simple")
    assert stage_modes == (1, 1, 2, 2, 3)
    assert all_surface is None
    assert indexed_surface == 2
    with pytest.raises(Exception, match="surface index"):
        mod.parse_optional_surface_index("-1")

    default_cases, _skipped = mod.default_seed_cases()
    families = {case.family for case in default_cases}
    assert {"qi", "qh", "qa", "simple"}.issubset(families)
    if (mod.OMNIGENITY_ROOT / "wouts_QI/wout_nfp2_QI_fixed_resolution_final.nc").exists():
        assert "qp" in families
    if (
        (mod.DATA_DIR / "input.QI_stel_seed_3127").exists()
        and (mod.DATA_DIR / "wout_QI_stel_seed_3127.nc").exists()
    ):
        labels = {case.label for case in default_cases}
        assert "qi_stel_seed_3127" in labels


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
            "qi_smooth_total": 1.0e-3,
            "qi_legacy_total": 1.5e-3,
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

    noisy_qi = mod._constraint_status(
        {
            "aspect": 5.0,
            "mean_iota": -0.45,
            "qi_mirror_ratio_max": 0.18,
            "qi_max_elongation": 7.5,
            "qi_smooth_total": 3.0e-3,
            "qi_legacy_total": 4.0e-3,
        },
        targets,
    )
    assert noisy_qi["seed_suitability"] == "needs_attention"
    assert {"smooth_qi", "legacy_qi"}.issubset(set(noisy_qi["failed_constraints"]))
    assert noisy_qi["smooth_qi_excess"] == pytest.approx(1.0e-3)
    assert noisy_qi["legacy_qi_excess"] == pytest.approx(2.0e-3)


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
            "aspect": mod.DEFAULT_TARGET_ASPECT,
            "mean_iota": 0.45,
            "qi_mirror_ratio_max": 0.18,
            "qi_max_elongation": 7.0,
            "qi_legacy_total": 1.0e-3,
        }
        if case.label == "better_qi":
            base["qi_smooth_total"] = 5.0e-4
        else:
            base["qi_smooth_total"] = 1.5e-3
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


def test_main_accepts_custom_case_with_bundled_qi_seed(monkeypatch, tmp_path):
    mod = _load_module()
    input_path = mod.DATA_DIR / "input.QI_stel_seed_3127"
    wout_path = mod.DATA_DIR / "wout_QI_stel_seed_3127.nc"
    if not input_path.exists() or not wout_path.exists():
        pytest.skip("bundled QI seed fixture is not available")

    targets = mod.SuitabilityTargets()
    seen = {}

    def fake_evaluate(case, **kwargs):
        seen["case"] = case
        seen["kwargs"] = kwargs
        assert case.input_path == input_path
        assert case.wout_path == wout_path
        assert case.input_path.exists()
        assert case.wout_path.exists()
        record = {
            "label": case.label,
            "family": case.family,
            "input": str(case.input_path),
            "wout": str(case.wout_path),
            "aspect": mod.DEFAULT_TARGET_ASPECT,
            "mean_iota": 0.45,
            "qi_mirror_ratio_max": 0.18,
            "qi_max_elongation": 7.0,
            "qi_smooth_total": 1.0e-3,
            "qi_legacy_total": 1.0e-3,
        }
        record.update(mod._constraint_status(record, targets))
        return record

    monkeypatch.setattr(mod, "evaluate_seed_case", fake_evaluate)
    output = tmp_path / "summary.json"
    csv_path = tmp_path / "summary.csv"

    rc = mod.main(
        [
            "--quick",
            "--case",
            f"custom_qi:qi:{input_path}:{wout_path}",
            "--output",
            str(output),
            "--csv",
            str(csv_path),
        ]
    )

    report = json.loads(output.read_text())
    rows = list(csv.DictReader(csv_path.open()))

    assert rc == 0
    assert seen["case"].label == "custom_qi"
    assert seen["case"].family == "qi"
    assert seen["kwargs"]["nphi"] == 51
    assert report["no_optimization"] is True
    assert report["skipped_defaults"] == []
    assert report["cases"][0]["label"] == "custom_qi"
    assert report["cases"][0]["suitability_rank"] == 1
    assert rows[0]["label"] == "custom_qi"
    assert rows[0]["seed_suitability"] == "pass"


def test_main_writes_prefine_manifest_for_custom_case_without_running_probe(monkeypatch, tmp_path):
    mod = _load_module()
    input_path = tmp_path / "input.custom"
    wout_path = tmp_path / "wout_custom.nc"
    input_path.write_text("&INDATA\n/")
    wout_path.write_text("placeholder")
    targets = mod.SuitabilityTargets()

    def fake_evaluate(case, **_kwargs):
        record = {
            "label": case.label,
            "family": case.family,
            "input": str(case.input_path),
            "wout": str(case.wout_path),
            "aspect": mod.DEFAULT_TARGET_ASPECT,
            "mean_iota": 0.45,
            "qi_phimin": 0.0,
            "qi_mirror_ratio_max": 0.18,
            "qi_max_elongation": 7.0,
            "qi_smooth_total": 1.0e-3,
            "qi_legacy_total": 1.0e-3,
        }
        record.update(mod._constraint_status(record, targets))
        return record

    monkeypatch.setattr(mod, "evaluate_seed_case", fake_evaluate)
    output = tmp_path / "summary.json"
    manifest_path = tmp_path / "prefine_manifest.json"

    rc = mod.main(
        [
            "--quick",
            "--case",
            f"custom case:qi:{input_path}:{wout_path}",
            "--output",
            str(output),
            "--prefine-probes",
            "plan",
            "--prefine-manifest",
            str(manifest_path),
            "--prefine-output-dir",
            str(tmp_path / "probes"),
            "--no-prefine-family-representatives",
        ]
    )

    report = json.loads(output.read_text())
    manifest = json.loads(manifest_path.read_text())

    assert rc == 0
    assert report["prefine_probe_mode"] == "plan"
    assert report["prefine_probe_manifest"] == str(manifest_path)
    assert report["prefine_probe_summary"]["dry_run"] is True
    assert manifest["dry_run"] is True
    assert manifest["plans"][0]["label"] == "custom case"
    assert manifest["plans"][0]["input"] == str(input_path)
    assert manifest["plans"][0]["wout"] == str(wout_path)
    assert manifest["plans"][0]["output_dir"].endswith("01_custom_case")
    assert manifest["plans"][0]["status"] == "planned"
    assert manifest["plans"][0]["selection_reasons"] == ["top_n"]
    assert manifest["summary"]["recommendation"]["action"] == "review_manifest"


def test_build_seed_audit_can_select_best_well_phase(monkeypatch):
    mod = _load_module()
    targets = mod.SuitabilityTargets(smooth_qi_max=None, legacy_qi_max=None)
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
    assert "--prefine-use-ess" in manifest["plans"][0]["run_command"]
    assert "--prefine-ess-alpha 1.2" in manifest["plans"][0]["run_command"]
    assert f"--prefine-mirror-weight {mod.DEFAULT_PREFINE_MIRROR_WEIGHT}" in manifest["plans"][0]["run_command"]
    assert (
        f"--prefine-elongation-weight {mod.DEFAULT_PREFINE_ELONGATION_WEIGHT}"
        in manifest["plans"][0]["run_command"]
    )
    assert "--prefine-mirror-surface-index all" in manifest["plans"][0]["run_command"]
    assert manifest["plans"][0]["optimization"]["objective"] == "qi_constrained_prefine_probe"
    assert manifest["plans"][0]["optimization"]["max_nfev"] <= mod.MAX_PREFINE_MAX_NFEV
    assert manifest["plans"][0]["optimization"]["use_ess"] is True
    assert manifest["plans"][0]["optimization"]["ess_alpha"] == 1.2
    assert manifest["plans"][0]["optimization"]["stage_modes"] == [1, 1, 2, 2, 3]
    assert manifest["plans"][0]["optimization"]["stage_count"] == 5
    assert manifest["plans"][0]["optimization"]["total_nfev_cap"] == 6
    assert manifest["plans"][0]["qi_options"]["nphi"] <= mod.MAX_PREFINE_QI_NPHI
    assert manifest["plans"][0]["qi_options"]["include_bounce_endpoints"] is True
    assert manifest["plans"][0]["qi_options"]["endpoint_mode"] == "include_bounce_endpoints"
    assert manifest["plans"][0]["qi_options"]["phimin"] == 0.5
    assert manifest["plans"][0]["qi_options"]["qi_ceiling_weight"] == 100.0
    assert manifest["plans"][0]["qi_options"]["mirror_weight"] == mod.DEFAULT_PREFINE_MIRROR_WEIGHT
    assert manifest["plans"][0]["qi_options"]["elongation_weight"] == mod.DEFAULT_PREFINE_ELONGATION_WEIGHT
    assert manifest["plans"][0]["qi_options"]["mirror_surface_index"] is None
    assert manifest["plans"][0]["qi_options"]["mirror_surface_mode"] == "all"
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


def test_prefine_probe_manifest_preserves_near_qi_before_aux_cleanup(tmp_path):
    mod = _load_module()
    report = {
        "cases": [
            {
                "suitability_rank": 1,
                "label": "near_qi",
                "family": "qi",
                "input": "/tmp/input_near_qi",
                "wout": "/tmp/wout_near_qi.nc",
                "qi_smooth_total": 1.0e-3,
                "qi_legacy_total": 5.0e-4,
            },
            {
                "suitability_rank": 2,
                "label": "far_qi",
                "family": "qh",
                "input": "/tmp/input_far",
                "wout": "/tmp/wout_far.nc",
                "qi_smooth_total": 0.2,
                "qi_legacy_total": 0.3,
            },
        ]
    }
    config = mod.QIPrefineProbeConfig(
        top_n=2,
        include_family_representatives=False,
        output_dir=tmp_path / "probes",
        mirror_weight=2.0,
        elongation_weight=0.5,
    )

    manifest = mod.build_qi_prefine_probe_manifest(
        report,
        config=config,
        manifest_path=tmp_path / "manifest.json",
        dry_run=True,
    )

    near = manifest["plans"][0]
    far = manifest["plans"][1]
    assert near["prefine_policy"]["name"] == "near_qi_preservation"
    assert near["prefine_policy"]["near_qi"] is True
    assert near["prefine_policy"]["project_input_boundary_to_max_mode"] is False
    assert near["optimization"]["project_input_boundary_to_max_mode"] is False
    assert "smooth_qi_below_preservation_threshold" in near["prefine_policy"]["reasons"]
    assert near["qi_options"]["qi_ceiling_weight"] == mod.DEFAULT_PREFINE_NEAR_QI_CEILING_WEIGHT
    assert near["qi_options"]["qi_ceiling_max"] == pytest.approx(1.25e-3)
    assert near["optimization"]["objective"] == "qi_only_prefine_probe"
    assert near["qi_options"]["mirror_weight"] == 0.0
    assert near["qi_options"]["elongation_weight"] == 0.0
    assert "--prefine-mirror-weight 0.0" in near["run_command"]
    assert "--prefine-elongation-weight 0.0" in near["run_command"]
    assert "--prefine-qi-ceiling-weight 500.0" in near["run_command"]
    assert "--no-prefine-preserve-near-qi" in near["run_command"]
    assert "--no-prefine-project-input-boundary-to-max-mode" in near["run_command"]

    assert far["prefine_policy"]["name"] == "constrained_recovery"
    assert far["prefine_policy"]["near_qi"] is False
    assert far["prefine_policy"]["project_input_boundary_to_max_mode"] is True
    assert far["optimization"]["project_input_boundary_to_max_mode"] is True
    assert far["qi_options"]["qi_ceiling_weight"] == 100.0
    assert far["qi_options"]["mirror_weight"] == 2.0
    assert far["qi_options"]["elongation_weight"] == 0.5


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
        config=mod.QIPrefineProbeConfig(
            output_dir=tmp_path / "probes",
            mirror_weight=0.0,
            elongation_weight=0.0,
        ),
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
                    "objective_diagnostics": {
                        "initial": {"qi_residual": 0.3, "qi_legacy_total": 0.4},
                        "final": {"qi_residual": 0.2, "qi_legacy_total": 0.3},
                    },
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
    assert completed["result"]["objective_diagnostics"]["final"]["qi_residual"] == 0.2
    assert completed["result"]["nfev"] == 3
    assert len(calls["tuples"]) == 1
    assert calls["qi_options"]["nphi"] == 31
    assert calls["qi_options"]["include_bounce_endpoints"] is True
    assert calls["from_input"]["max_mode"] == 3
    assert calls["solve"]["max_nfev"] == 2
    assert calls["solve"]["stage_modes"] == (1, 1, 2, 2, 3)
    assert calls["solve"]["save_stage_wouts"] is False


def test_run_qi_prefine_probe_can_dispatch_constrained_qi_terms(tmp_path):
    mod = _load_module()
    report = {
        "cases": [
            {
                "suitability_rank": 1,
                "label": "candidate_qp",
                "family": "qp",
                "input": "/tmp/input_qp",
                "wout": "/tmp/wout_qp.nc",
                "qi_seed_score": 0.2,
            },
        ]
    }
    manifest = mod.build_qi_prefine_probe_manifest(
        report,
        config=mod.QIPrefineProbeConfig(
            output_dir=tmp_path / "probes",
            mirror_weight=2.0,
            mirror_threshold=0.21,
            mirror_ntheta=12,
            mirror_nphi=10,
            elongation_weight=0.5,
            elongation_threshold=8.0,
            elongation_ntheta=14,
            elongation_nphi=6,
        ),
        manifest_path=tmp_path / "manifest.json",
        dry_run=False,
    )
    plan = manifest["plans"][0]
    assert plan["optimization"]["objective"] == "qi_constrained_prefine_probe"
    assert plan["qi_options"]["qi_ceiling_weight"] == 100.0
    assert plan["qi_options"]["qi_ceiling_max"] == 2.0e-3
    assert plan["qi_options"]["mirror_weight"] == 2.0
    assert plan["qi_options"]["elongation_weight"] == 0.5

    calls = {}

    class FakeQuasiIsodynamicOptions:
        def __init__(self, **kwargs):
            calls["qi_options"] = kwargs

    class FakeObjective:
        requires_qi_field = True

        def __init__(self, name, **kwargs):
            self.name = name
            self.kwargs = kwargs

        def J(self, _ctx, _state):
            return 0.0

    class FakeQuasiIsodynamicResidual(FakeObjective):
        def __init__(self, options):
            super().__init__("qi", options=options)

    class FakeQuasiIsodynamicResidualCeiling(FakeObjective):
        def __init__(self, **kwargs):
            super().__init__("qi_ceiling", **kwargs)
            calls["qi_ceiling"] = kwargs

    class FakeMirrorRatio(FakeObjective):
        def __init__(self, **kwargs):
            super().__init__("mirror", **kwargs)
            calls["mirror"] = kwargs

    class FakeMaxElongation(FakeObjective):
        def __init__(self, **kwargs):
            super().__init__("elongation", **kwargs)
            calls["elongation"] = kwargs

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
                    "objective_final": 2.0,
                    "qs_final": 2.0,
                    "total_wall_time_s": 0.5,
                    "history": [{"objective": 3.0}, {"objective": 2.0}],
                }
            },
            stage_modes=[1, 1, 2, 2, 3],
        )

    fake_workflow = SimpleNamespace(
        QuasiIsodynamicOptions=FakeQuasiIsodynamicOptions,
        QuasiIsodynamicResidual=FakeQuasiIsodynamicResidual,
        QuasiIsodynamicResidualCeiling=FakeQuasiIsodynamicResidualCeiling,
        MirrorRatio=FakeMirrorRatio,
        MaxElongation=FakeMaxElongation,
        LeastSquaresProblem=FakeLeastSquaresProblem,
        FixedBoundaryVMEC=FakeFixedBoundaryVMEC,
        least_squares_solve=fake_least_squares_solve,
    )

    completed = mod.run_qi_prefine_probe(plan, workflow=fake_workflow)

    assert completed["status"] == "completed"
    assert len(calls["tuples"]) == 4
    assert [entry[2] for entry in calls["tuples"]] == [1.0, 100.0, 2.0, 0.5]
    assert calls["qi_ceiling"]["maximum"] == 2.0e-3
    assert calls["qi_ceiling"]["smooth_penalty"] == 2.0e-3
    assert calls["mirror"]["threshold"] == 0.21
    assert calls["mirror"]["ntheta"] == 12
    assert calls["mirror"]["nphi"] == 10
    assert calls["mirror"]["surface_index"] is None
    assert isinstance(calls["mirror"]["qi_options"], FakeQuasiIsodynamicOptions)
    assert calls["elongation"]["threshold"] == 8.0
    assert calls["elongation"]["ntheta"] == 14
    assert calls["elongation"]["nphi"] == 6
    assert calls["from_input"]["project_input_boundary_to_max_mode"] is True


def test_qi_cleanup_candidate_promotion_requires_qi_gate_and_mirror_improvement():
    import vmec_jax as vj

    targets = vj.QISeedSuitabilityTargets(
        smooth_qi_max=2.0e-3,
        legacy_qi_max=1.0e-3,
        target_aspect=5.0,
        abs_iota_min=0.41,
        mirror_ratio_max=0.30,
        max_elongation=8.0,
    )
    reference = {
        "qi_smooth_total": 1.0e-3,
        "qi_legacy_total": 5.0e-4,
        "aspect": 5.0,
        "mean_iota": -0.50,
        "qi_mirror_ratio_max": 0.35,
        "qi_max_elongation": 7.0,
        "qi_mirror_ratio_target": 0.30,
        "qi_elongation_target": 8.0,
    }
    better_mirror = {**reference, "qi_mirror_ratio_max": 0.29}
    promoted = vj.qi_cleanup_candidate_promotable(
        better_mirror,
        reference=reference,
        targets=targets,
    )

    assert promoted["qi_cleanup_promoted"] is True
    assert promoted["qi_cleanup_rejection_reasons"] == []

    worse_mirror = {**reference, "qi_mirror_ratio_max": 0.36}
    rejected = vj.qi_cleanup_candidate_promotable(
        worse_mirror,
        reference=reference,
        targets=targets,
    )

    assert rejected["qi_cleanup_promoted"] is False
    assert any("mirror ratio did not improve" in reason for reason in rejected["qi_cleanup_rejection_reasons"])

    broken_qi = {**better_mirror, "qi_smooth_total": 1.0e-2}
    rejected_qi = vj.qi_cleanup_candidate_promotable(
        broken_qi,
        reference=reference,
        targets=targets,
    )

    assert rejected_qi["qi_cleanup_promoted"] is False
    assert any("QI seed gate failed" in reason for reason in rejected_qi["qi_cleanup_rejection_reasons"])


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


def test_prefine_probe_summary_accepts_stable_low_objective_seed():
    mod = _load_module()
    manifest = {
        "dry_run": False,
        "selection": {"planned_rows": 2},
        "plans": [
            {
                "status": "completed",
                "label": "already_good_qi",
                "family": "qi",
                "audit_rank": 2,
                "optimization": {"stage_modes": [1]},
                "result": {
                    "objective_initial": 2.5e-2,
                    "objective_final": 2.5e-2,
                    "requested_stage_modes": [1],
                    "completed_stage_modes": [1],
                    "history": [
                        {"objective": 2.5e-2},
                        {"objective": 2.5e-2},
                    ],
                },
            },
            {
                "status": "completed",
                "label": "improved_but_worse_qp",
                "family": "qp",
                "audit_rank": 1,
                "optimization": {"stage_modes": [1]},
                "result": {
                    "objective_initial": 1.0e-1,
                    "objective_final": 6.0e-2,
                    "requested_stage_modes": [1],
                    "completed_stage_modes": [1],
                    "history": [
                        {"objective": 1.0e-1},
                        {"objective": 6.0e-2},
                    ],
                },
            },
        ],
    }

    summary = mod.summarize_qi_prefine_probe_manifest(manifest)

    assert summary["accepted_candidate"]["label"] == "already_good_qi"
    assert summary["accepted_candidate"]["acceptance"]["decision"] == "accepted_stable_low_objective"
    assert summary["acceptance"]["accepted"] is True
    assert summary["recommendation"]["action"] == "promote_best_candidate"


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


def test_prefine_summary_decomposes_objective_and_flags_qi_worsening():
    mod = _load_module()
    manifest = {
        "dry_run": False,
        "selection": {"planned_rows": 1},
        "plans": [
            {
                "status": "completed",
                "label": "scalar_better_qi_worse",
                "family": "qi",
                "audit_rank": 1,
                "audit_metrics": {
                    "qi_smooth_total": 0.2,
                    "qi_legacy_total": 0.25,
                    "qi_mirror_ratio_max": 0.18,
                    "qi_mirror_ratio_target": 0.21,
                    "qi_max_elongation": 7.5,
                    "qi_elongation_target": 8.0,
                    "aspect": 5.1,
                    "target_aspect": 5.0,
                    "mean_iota": -0.45,
                    "abs_iota_min": 0.41,
                },
                "qi_options": {
                    "mirror_weight": 2.0,
                    "mirror_threshold": 0.21,
                    "elongation_weight": 0.5,
                    "elongation_threshold": 8.0,
                },
                "optimization": {"stage_modes": [1]},
                "result": {
                    "objective_initial": 10.0,
                    "objective_final": 4.0,
                    "requested_stage_modes": [1],
                    "completed_stage_modes": [1],
                    "final_diagnostics": {
                        "qi_smooth_total": 0.35,
                        "qi_legacy_total": 0.4,
                        "qi_mirror_ratio_max": 0.25,
                        "qi_mirror_ratio_target": 0.21,
                        "qi_mirror_excess_max": 0.04,
                        "qi_max_elongation": 8.5,
                        "qi_elongation_target": 8.0,
                        "qi_elongation_excess": 0.5,
                        "aspect": 5.3,
                        "mean_iota": -0.42,
                    },
                    "history": [{"objective": 10.0}, {"objective": 4.0}],
                },
            }
        ],
    }

    summary = mod.summarize_qi_prefine_probe_manifest(manifest)
    row = summary["plan_summaries"][0]
    diagnostics = row["objective_diagnostics"]

    assert diagnostics["initial"]["qi_residual"] == pytest.approx(0.2)
    assert diagnostics["final"]["qi_residual"] == pytest.approx(0.35)
    assert diagnostics["final"]["mirror_penalty"] == pytest.approx(0.04**2)
    assert diagnostics["final"]["mirror_weighted_penalty"] == pytest.approx(2.0 * 0.04**2)
    assert diagnostics["final"]["elongation_penalty"] == pytest.approx(0.5**2)
    assert diagnostics["final"]["elongation_weighted_penalty"] == pytest.approx(0.5 * 0.5**2)
    assert diagnostics["final"]["aspect"] == pytest.approx(5.3)
    assert diagnostics["final"]["mean_iota"] == pytest.approx(-0.42)
    assert diagnostics["delta"]["qi_residual"] == pytest.approx(0.15)
    assert diagnostics["flags"]["scalar_improved_but_qi_worsened"] is True
    assert diagnostics["flags"]["worsened_qi_terms"] == ["smooth_qi", "legacy_qi"]
    assert row["acceptance"]["accepted"] is False
    assert "smooth/legacy QI worsened" in row["acceptance"]["reasons"][0]
    assert summary["scalar_improved_qi_worsened_count"] == 1
    assert summary["scalar_improved_qi_worsened"][0]["label"] == "scalar_better_qi_worse"
    assert summary["acceptance"]["accepted"] is False
    assert summary["recommendation"]["action"] == "review_qi_worsening"

    result_summary = mod.summarize_qi_prefine_results(manifest)
    assert result_summary["scalar_improved_qi_worsened"][0]["label"] == "scalar_better_qi_worse"
    assert result_summary["legacy_recommendation"] == "inspect_scalar_improved_qi_worsening_before_promoting_seed"


def test_prefine_summary_merges_history_diagnostics_with_final_artifacts():
    mod = _load_module()
    manifest = {
        "dry_run": False,
        "selection": {"planned_rows": 1},
        "plans": [
            {
                "status": "completed",
                "label": "merged_diagnostics",
                "family": "qi",
                "audit_rank": 1,
                "audit_metrics": {
                    "qi_smooth_total": 0.8,
                    "qi_legacy_total": 0.9,
                    "qi_mirror_ratio_max": 0.18,
                    "qi_max_elongation": 7.7,
                    "aspect": 5.0,
                    "mean_iota": -0.45,
                },
                "qi_options": {
                    "mirror_weight": 2.0,
                    "mirror_threshold": 0.21,
                    "elongation_weight": 0.5,
                    "elongation_threshold": 8.0,
                },
                "optimization": {"stage_modes": [1]},
                "result": {
                    "objective_initial": 3.0,
                    "objective_final": 1.0,
                    "requested_stage_modes": [1],
                    "completed_stage_modes": [1],
                    "objective_diagnostics": {
                        "initial": {"qi_residual": 0.3, "qi_legacy_total": 0.4},
                        "final": {"qi_residual": 0.2, "qi_legacy_total": 0.35},
                    },
                    "final_diagnostics": {
                        "mirror_ratio": 0.31,
                        "elongation": 8.4,
                        "aspect": 5.4,
                        "mean_iota": -0.39,
                    },
                    "history": [{"objective": 3.0}, {"objective": 1.0}],
                },
            }
        ],
    }

    summary = mod.summarize_qi_prefine_probe_manifest(manifest)
    diagnostics = summary["plan_summaries"][0]["objective_diagnostics"]

    assert diagnostics["initial"]["qi_residual"] == pytest.approx(0.3)
    assert diagnostics["final"]["qi_residual"] == pytest.approx(0.2)
    assert diagnostics["final"]["mirror_ratio"] == pytest.approx(0.31)
    assert diagnostics["final"]["mirror_threshold"] == pytest.approx(0.21)
    assert diagnostics["final"]["mirror_excess"] == pytest.approx(0.1)
    assert diagnostics["final"]["mirror_weighted_penalty"] == pytest.approx(2.0 * 0.1**2)
    assert diagnostics["final"]["elongation"] == pytest.approx(8.4)
    assert diagnostics["final"]["elongation_threshold"] == pytest.approx(8.0)
    assert diagnostics["final"]["elongation_excess"] == pytest.approx(0.4)
    assert diagnostics["final"]["elongation_weighted_penalty"] == pytest.approx(0.5 * 0.4**2)
    assert diagnostics["final"]["aspect"] == pytest.approx(5.4)
    assert diagnostics["final"]["mean_iota"] == pytest.approx(-0.39)
    assert diagnostics["flags"]["scalar_improved_but_qi_worsened"] is False
    assert summary["accepted_candidate"]["label"] == "merged_diagnostics"


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
