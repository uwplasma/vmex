from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import sys

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

    assert case.label == "candidate_qi"
    assert case.family == "qi"
    assert case.input_path == Path("/tmp/input")
    assert case.wout_path == Path("/tmp/wout.nc")

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

    csv_path = tmp_path / "audit.csv"
    mod._write_csv(report["cases"], csv_path)
    rows = list(csv.DictReader(csv_path.open()))
    assert rows[0]["label"] == "better_qi"
    assert float(rows[0]["qi_seed_score"]) < float(rows[1]["qi_seed_score"])
    assert rows[1]["failed_constraints"] == "mirror"
