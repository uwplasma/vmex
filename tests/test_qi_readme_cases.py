from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "optimization" / "render_qi_readme_cases.py"
CASES_SCRIPT = ROOT / "examples" / "optimization" / "qi_optimization_cases.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("render_qi_readme_cases_test", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_cases_module():
    spec = importlib.util.spec_from_file_location("qi_optimization_cases_test", CASES_SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_synthetic_input(path: Path, *, nfp: int) -> None:
    path.write_text(
        "&INDATA\n"
        f"  NFP = {nfp}\n"
        "  RBC(0,0) = 1.0\n"
        "  RBC(0,1) = 0.2\n"
        "  ZBS(0,1) = 0.3\n"
        "/\n"
    )


def _synthetic_wout(*, nfp: int, rbc01: float = 0.2):
    class Wout:
        pass

    wout = Wout()
    wout.nfp = nfp
    wout.xm = np.asarray([0, 1], dtype=int)
    wout.xn = np.asarray([0, 0], dtype=int)
    wout.rmnc = np.asarray([[1.0, rbc01]], dtype=float)
    wout.rmns = np.zeros((1, 2), dtype=float)
    wout.zmnc = np.zeros((1, 2), dtype=float)
    wout.zmns = np.asarray([[0.0, 0.3]], dtype=float)
    return wout


def _patch_matching_wout(monkeypatch, mod, *, nfp: int = 2) -> None:
    monkeypatch.setattr(mod, "read_wout", lambda _path: _synthetic_wout(nfp=nfp))


def _synthetic_case(mod, tmp_path: Path, label: str, *, nfp: int, validation_status: str = "case-gated"):
    case_dir = tmp_path / f"case_nfp{nfp}"
    case_dir.mkdir(parents=True)
    input_file = tmp_path / "examples" / "data" / f"input.nfp{nfp}_QI"
    input_file.parent.mkdir(parents=True, exist_ok=True)
    initial_wout = case_dir / "wout_initial.nc"
    final_wout = case_dir / "wout_final.nc"
    _write_synthetic_input(input_file, nfp=nfp)
    for path in (initial_wout, final_wout):
        path.write_text("synthetic\n")
    return mod.QICase(
        label=label,
        input_file=input_file,
        output_dir=case_dir,
        initial_wout=initial_wout,
        note=f"synthetic NFP={nfp}",
        validation_status=validation_status,
    )


def _passing_diagnostics(*, nfp: int = 2) -> dict:
    return {
        "qi_smooth_total": 1.0e-3,
        "qi_smooth_gate": 2.0e-3,
        "qi_legacy_total": 1.5e-3,
        "qi_legacy_gate": 2.0e-3,
        "qi_mirror_ratio_max": 0.24,
        "qi_mirror_ratio_target": 0.30,
        "qi_max_elongation": 6.5,
        "qi_elongation_target": 8.2,
        "aspect": 9.8,
        "target_aspect": 10.0,
        "mean_iota": 0.47,
        "abs_iota_min": 0.41,
        "qi_nfp": nfp,
        "qi_seed_gate_passed": True,
        "qi_engineering_gate_passed": True,
        "qi_gate_failures": [],
    }


def test_readme_renderer_records_case_gated_nfp123_gate_status(monkeypatch, tmp_path: Path) -> None:
    mod = _load_module()
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(mod, "_history_summary", lambda _case: (120.0, 2, 5))
    monkeypatch.setattr(mod, "_preconditioner_summary", lambda _case: (0, None, None, None))
    _patch_matching_wout(monkeypatch, mod)

    def fake_load_json(path: Path):
        if path.name == "history.json":
            return {"objective_final": 1.5, "total_wall_time_s": 60.0}
        return _passing_diagnostics(nfp=int(path.parent.name.removeprefix("case_nfp")))

    monkeypatch.setattr(mod, "_load_json", fake_load_json)

    records = [
        mod._case_record(_synthetic_case(mod, tmp_path, "NFP=1 QI", nfp=1)),
        mod._case_record(_synthetic_case(mod, tmp_path, "NFP=2 bundled QI", nfp=2)),
        mod._case_record(_synthetic_case(mod, tmp_path, "NFP=3 seed 3127", nfp=3)),
    ]

    assert {record["qi_nfp"] for record in records} == {1, 2, 3}
    assert {record["validation_status"] for record in records} == {"case-gated"}
    assert all(record["qi_seed_gate_passed"] is True for record in records)
    assert all(record["qi_engineering_gate_passed"] is True for record in records)
    assert all(record["qi_gate_failures"] == "" for record in records)


def test_readme_renderer_keeps_deferred_case_from_promotion(
    monkeypatch, tmp_path: Path
) -> None:
    mod = _load_module()
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(mod, "_history_summary", lambda _case: (180.0, 1, 3))
    monkeypatch.setattr(mod, "_preconditioner_summary", lambda _case: (3, 1.0, 5.0e-3, 0.31))
    _patch_matching_wout(monkeypatch, mod, nfp=4)

    def fake_load_json(path: Path):
        if path.name == "history.json":
            return {"objective_final": 2.5, "total_wall_time_s": 90.0}
        return {
            "qi_smooth_total": 8.0e-3,
            "qi_legacy_total": 5.0e-3,
            "qi_mirror_ratio_max": 0.36,
            "qi_mirror_ratio_target": 0.35,
            "qi_max_elongation": 6.8,
            "qi_elongation_target": 8.2,
            "aspect": 6.2,
            "target_aspect": 6.0,
            "mean_iota": 0.52,
            "qi_nfp": 4,
            "qi_case_expected_gate_status": "candidate",
            "qi_seed_gate_passed": False,
            "qi_engineering_gate_passed": False,
            "qi_gate_failures": ["smooth_qi", "legacy_qi", "mirror"],
        }

    monkeypatch.setattr(mod, "_load_json", fake_load_json)

    record = mod._case_record(
        _synthetic_case(
            mod,
            tmp_path,
            "NFP=4 minimal + QI-reference proposal",
            nfp=4,
            validation_status="deferred",
        )
    )

    assert record["qi_nfp"] == 4
    assert record["validation_status"] == "deferred"
    assert record["expected_gate_status"] == "candidate"
    assert record["qi_seed_gate_passed"] is False
    assert record["qi_engineering_gate_passed"] is False
    assert record["qi_gate_failures"] == "smooth_qi;legacy_qi;mirror"


def test_readme_renderer_points_nfp4_row_at_reference_proposal_seed() -> None:
    mod = _load_module()

    nfp4 = mod.CASES[-1]

    assert nfp4.label == "NFP=4 minimal + QI-reference proposal"
    assert nfp4.input_file.name == "input.minimal_seed_nfp4"
    assert nfp4.output_dir == ROOT / "docs" / "_static" / "qi_readme_cases" / "nfp4_minimal"
    assert "nfp4_qi_finite_beta" not in str(nfp4.output_dir)
    assert nfp4.initial_wout == nfp4.output_dir / "wout_initial.nc"
    assert nfp4.validation_status == "case-gated"
    assert nfp4.preconditioner_summary is not None
    assert nfp4.preconditioner_summary.name == "summary.json"


def test_readme_renderer_points_nfp3_at_bundled_raw_initial_artifact() -> None:
    mod = _load_module()

    nfp3 = next(case for case in mod.CASES if case.label == "NFP=3 seed 3127")

    bundled_raw_wout = ROOT / "examples" / "data" / "wout_QI_stel_seed_3127.nc"
    assert nfp3.initial_wout == ROOT / "docs" / "_static" / "qi_readme_cases" / "nfp3_seed3127" / "wout_initial.nc"
    assert nfp3.initial_wout.read_bytes() == bundled_raw_wout.read_bytes()


def test_real_nfp3_raw_initial_wout_matches_input_boundary() -> None:
    mod = _load_module()

    nfp3 = next(case for case in mod.CASES if case.label == "NFP=3 seed 3127")

    assert mod._boundary_mismatches(nfp3.input_file, nfp3.initial_wout) == []
    mod._validate_case_initial_wout(nfp3)


def test_nfp3_case_catalog_matches_checked_in_readme_metadata() -> None:
    cases_mod = _load_cases_module()
    artifact_dir = ROOT / "docs" / "_static" / "qi_readme_cases" / "nfp3_seed3127"
    history = json.loads((artifact_dir / "history.json").read_text())
    diagnostics = json.loads(
        (artifact_dir / "diagnostics.json").read_text()
    )

    case = cases_mod.QI_CASES["nfp3_qi"]

    assert case["target_aspect"] == pytest.approx(history["target_aspect"])
    assert case["target_aspect"] == pytest.approx(diagnostics["target_aspect"])
    assert case["target_aspect"] == pytest.approx(cases_mod.SEED3127_REVIEWED_TARGET_ASPECT)
    assert "aspect4" in str(case["output_dir"])


def test_real_nfp4_raw_initial_wout_matches_minimal_seed_and_final_differs() -> None:
    mod = _load_module()

    nfp4 = next(case for case in mod.CASES if case.label == "NFP=4 minimal + QI-reference proposal")
    final_wout = nfp4.output_dir / "wout_final.nc"

    assert nfp4.initial_wout.read_bytes() != final_wout.read_bytes()
    assert mod._boundary_mismatches(nfp4.input_file, nfp4.initial_wout) == []
    mod._validate_case_initial_wout(nfp4)


def test_readme_renderer_detects_flat_objective_history() -> None:
    mod = _load_module()

    flat = [
        {
            "objective": np.asarray([1.0, 1.0 - 1.0e-8, 1.0 - 2.0e-8]),
            "wall_time_s": np.asarray([0.0, 1.0, 2.0]),
            "label": "flat",
            "path": Path("flat"),
        }
    ]
    moving = [
        {
            "objective": np.asarray([1.0, 0.8, 0.7]),
            "wall_time_s": np.asarray([0.0, 1.0, 2.0]),
            "label": "moving",
            "path": Path("moving"),
        }
    ]

    assert mod._history_is_effectively_flat(flat)
    assert not mod._history_is_effectively_flat(moving)


def test_qi_case_catalog_defines_nfp4_minimal_seed_candidate() -> None:
    cases_mod = _load_cases_module()

    case = cases_mod.QI_CASES["nfp4_qi"]

    assert case["input_file"].name == "input.minimal_seed_nfp4"
    assert case["target_aspect"] == cases_mod.DEFAULT_QI_TARGET_ASPECT
    assert case["mirror_threshold"] == pytest.approx(0.35)
    assert case["qi_gate_legacy_max"] == pytest.approx(2.0e-3)
    assert "minimal_nfp4_to_qi_finite_beta_reference" in str(case["output_dir"])
    assert case["boundary_reference_preconditioner"]["reference_input"].name == "input.nfp4_QI_finite_beta"
    assert case["boundary_reference_preconditioner"]["accept_as_baseline"] is True
    assert {family for family, _index, _value in case["target_helicity_seed_terms"]} == {"RBC", "ZBS"}


def test_readme_renderer_rejects_case_gated_case_with_failed_engineering_gate(
    monkeypatch, tmp_path: Path
) -> None:
    mod = _load_module()
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(mod, "_history_summary", lambda _case: (10.0, 1, 1))
    monkeypatch.setattr(mod, "_preconditioner_summary", lambda _case: (0, None, None, None))
    _patch_matching_wout(monkeypatch, mod)

    def fake_load_json(path: Path):
        if path.name == "history.json":
            return {"objective_final": 1.0, "total_wall_time_s": 10.0}
        return {
            "qi_raw_total": 1.0e-3,
            "qi_legacy_total": 1.0e-3,
            "qi_mirror_ratio_max": 0.5,
            "qi_mirror_ratio_target": 0.3,
            "qi_max_elongation": 7.0,
            "qi_elongation_target": 8.2,
            "aspect": 10.0,
            "target_aspect": 10.0,
            "mean_iota": 0.5,
            "qi_nfp": 2,
            "qi_engineering_gate_passed": False,
        }

    monkeypatch.setattr(mod, "_load_json", fake_load_json)

    with pytest.raises(RuntimeError, match="case-gated but failed"):
        mod._case_record(_synthetic_case(mod, tmp_path, "NFP=2 bundled QI", nfp=2))


def test_readme_renderer_rejects_case_gated_case_with_failed_seed_gate(monkeypatch, tmp_path: Path) -> None:
    mod = _load_module()
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(mod, "_history_summary", lambda _case: (10.0, 1, 1))
    monkeypatch.setattr(mod, "_preconditioner_summary", lambda _case: (0, None, None, None))
    _patch_matching_wout(monkeypatch, mod)

    def fake_load_json(path: Path):
        if path.name == "history.json":
            return {"objective_final": 1.0, "total_wall_time_s": 10.0}
        diagnostics = _passing_diagnostics()
        diagnostics["qi_seed_gate_passed"] = False
        return diagnostics

    monkeypatch.setattr(mod, "_load_json", fake_load_json)

    with pytest.raises(RuntimeError, match="failed the QI seed gate"):
        mod._case_record(_synthetic_case(mod, tmp_path, "NFP=2 bundled QI", nfp=2))


def test_readme_renderer_rejects_case_gated_case_with_gate_failures(monkeypatch, tmp_path: Path) -> None:
    mod = _load_module()
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(mod, "_history_summary", lambda _case: (10.0, 1, 1))
    monkeypatch.setattr(mod, "_preconditioner_summary", lambda _case: (0, None, None, None))
    _patch_matching_wout(monkeypatch, mod)

    def fake_load_json(path: Path):
        if path.name == "history.json":
            return {"objective_final": 1.0, "total_wall_time_s": 10.0}
        diagnostics = _passing_diagnostics()
        diagnostics["qi_gate_failures"] = ["smooth_qi"]
        return diagnostics

    monkeypatch.setattr(mod, "_load_json", fake_load_json)

    with pytest.raises(RuntimeError, match="has QI gate failures"):
        mod._case_record(_synthetic_case(mod, tmp_path, "NFP=2 bundled QI", nfp=2))


def test_readme_renderer_rejects_case_gated_case_with_nonfinite_metric(monkeypatch, tmp_path: Path) -> None:
    mod = _load_module()
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(mod, "_history_summary", lambda _case: (10.0, 1, 1))
    monkeypatch.setattr(mod, "_preconditioner_summary", lambda _case: (0, None, None, None))
    _patch_matching_wout(monkeypatch, mod)

    def fake_load_json(path: Path):
        if path.name == "history.json":
            return {"objective_final": 1.0, "total_wall_time_s": 10.0}
        diagnostics = _passing_diagnostics()
        diagnostics["qi_smooth_total"] = float("nan")
        return diagnostics

    monkeypatch.setattr(mod, "_load_json", fake_load_json)

    with pytest.raises(RuntimeError, match="non-finite qi_smooth_total"):
        mod._case_record(_synthetic_case(mod, tmp_path, "NFP=2 bundled QI", nfp=2))


def test_real_qi_readme_csv_contains_only_clean_case_gated_rows() -> None:
    csv_path = ROOT / "docs" / "_static" / "figures" / "readme_qi_optimization_cases.csv"
    rows = list(csv.DictReader(csv_path.open()))

    assert {int(row["qi_nfp"]) for row in rows} == {1, 2, 3, 4}
    assert {row["validation_status"] for row in rows} == {"case-gated"}
    assert {row["qi_gate_failures"] for row in rows} == {""}
    assert {row["qi_seed_gate_passed"] for row in rows} == {"True"}
    assert {row["qi_engineering_gate_passed"] for row in rows} == {"True"}
    labels = {row["case"] for row in rows}
    assert "NFP=2 target-helicity seed" in labels
    assert "NFP=4 minimal + QI-reference proposal" in labels
    nfp4 = next(row for row in rows if int(row["qi_nfp"]) == 4)
    assert int(nfp4["preconditioner_points"]) == 1
    assert float(nfp4["selected_lambda"]) == pytest.approx(1.0)


def test_readme_renderer_rejects_initial_wout_that_does_not_match_input(monkeypatch, tmp_path: Path) -> None:
    mod = _load_module()
    case = _synthetic_case(mod, tmp_path, "NFP=2 bundled QI", nfp=2)
    monkeypatch.setattr(mod, "read_wout", lambda _path: _synthetic_wout(nfp=2, rbc01=0.95))

    with pytest.raises(RuntimeError, match="not the raw input boundary"):
        mod._validate_case_initial_wout(case)


def test_readme_renderer_accepts_vmec_canonical_phase_equivalent_wout(monkeypatch, tmp_path: Path) -> None:
    mod = _load_module()
    input_file = tmp_path / "input.phase"
    input_file.write_text(
        "&INDATA\n"
        "  NFP = 3\n"
        "  RBC(0,0) = 1.0\n"
        "  RBC(1,0) = -0.2\n"
        "  ZBS(1,0) = -0.3\n"
        "  RBC(-2,1) = -0.04\n"
        "  ZBS(-2,1) = -0.05\n"
        "  RBC(-2,2) = 0.06\n"
        "  ZBS(-2,2) = -0.07\n"
        "/\n"
    )

    class Wout:
        pass

    wout = Wout()
    wout.nfp = 3
    wout.xm = np.asarray([0, 0, 1, 2], dtype=int)
    wout.xn = np.asarray([0, 3, 6, 6], dtype=int)
    wout.rmnc = np.asarray([[1.0, -0.2, 0.04, 0.06]], dtype=float)
    wout.rmns = np.zeros((1, 4), dtype=float)
    wout.zmnc = np.zeros((1, 4), dtype=float)
    wout.zmns = np.asarray([[0.0, -0.3, -0.05, 0.07]], dtype=float)
    monkeypatch.setattr(mod, "read_wout", lambda _path: wout)

    assert mod._boundary_mismatches(input_file, tmp_path / "wout.nc") == []


def test_readme_renderer_has_no_raw_wout_artifact_exception_bypass(monkeypatch, tmp_path: Path) -> None:
    mod = _load_module()
    case = _synthetic_case(mod, tmp_path, "NFP=3 seed 3127", nfp=3)
    monkeypatch.setattr(mod, "read_wout", lambda _path: _synthetic_wout(nfp=3, rbc01=0.95))

    assert not hasattr(case, "raw_initial_wout_exception")
    assert not hasattr(mod, "KNOWN_RAW_WOUT_ARTIFACTS")
    with pytest.raises(RuntimeError, match="not the raw input boundary"):
        mod._validate_case_initial_wout(case)
