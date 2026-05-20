from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "optimization" / "render_qi_readme_cases.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("render_qi_readme_cases_test", SCRIPT)
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


def _synthetic_case(mod, tmp_path: Path, label: str, *, nfp: int, validation_status: str = "promoted"):
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


def test_readme_renderer_records_promoted_nfp123_gate_status(monkeypatch, tmp_path: Path) -> None:
    mod = _load_module()
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(mod, "_history_summary", lambda _case: (120.0, 2, 5))
    monkeypatch.setattr(mod, "_preconditioner_summary", lambda _case: (0, None, None, None))
    _patch_matching_wout(monkeypatch, mod)

    def fake_load_json(path: Path):
        if path.name == "history.json":
            return {"objective_final": 1.5, "total_wall_time_s": 60.0}
        return {
            "qi_smooth_total": 1.0e-3,
            "qi_legacy_total": 1.5e-3,
            "qi_mirror_ratio_max": 0.24,
            "qi_mirror_ratio_target": 0.30,
            "qi_max_elongation": 6.5,
            "qi_elongation_target": 8.2,
            "aspect": 9.8,
            "target_aspect": 10.0,
            "mean_iota": 0.47,
            "qi_nfp": int(path.parent.name.removeprefix("case_nfp")),
            "qi_seed_gate_passed": True,
            "qi_engineering_gate_passed": True,
            "qi_gate_failures": [],
        }

    monkeypatch.setattr(mod, "_load_json", fake_load_json)

    records = [
        mod._case_record(_synthetic_case(mod, tmp_path, "NFP=1 QI", nfp=1)),
        mod._case_record(_synthetic_case(mod, tmp_path, "NFP=2 bundled QI", nfp=2)),
        mod._case_record(_synthetic_case(mod, tmp_path, "NFP=3 seed 3127", nfp=3)),
    ]

    assert {record["qi_nfp"] for record in records} == {1, 2, 3}
    assert {record["validation_status"] for record in records} == {"promoted"}
    assert all(record["qi_seed_gate_passed"] is True for record in records)
    assert all(record["qi_engineering_gate_passed"] is True for record in records)
    assert all(record["qi_gate_failures"] == "" for record in records)


def test_readme_renderer_keeps_nfp4_deferred_with_gate_failures(monkeypatch, tmp_path: Path) -> None:
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
            "qi_case_expected_gate_status": "non_passing_stress_fixture",
            "qi_case_stress_fixture": True,
            "qi_seed_gate_passed": False,
            "qi_engineering_gate_passed": False,
            "qi_gate_failures": ["smooth_qi", "legacy_qi", "mirror"],
        }

    monkeypatch.setattr(mod, "_load_json", fake_load_json)

    record = mod._case_record(
        _synthetic_case(mod, tmp_path, "NFP=4 minimal seed", nfp=4, validation_status="deferred")
    )

    assert record["qi_nfp"] == 4
    assert record["validation_status"] == "deferred"
    assert record["expected_gate_status"] == "non_passing_stress_fixture"
    assert record["qi_seed_gate_passed"] is False
    assert record["qi_engineering_gate_passed"] is False
    assert record["qi_gate_failures"] == "smooth_qi;legacy_qi;mirror"


def test_readme_renderer_rejects_promoted_case_with_failed_engineering_gate(monkeypatch, tmp_path: Path) -> None:
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

    with pytest.raises(RuntimeError, match="promoted but failed"):
        mod._case_record(_synthetic_case(mod, tmp_path, "NFP=2 bundled QI", nfp=2))


def test_readme_renderer_rejects_initial_wout_that_does_not_match_input(monkeypatch, tmp_path: Path) -> None:
    mod = _load_module()
    case = _synthetic_case(mod, tmp_path, "NFP=2 bundled QI", nfp=2)
    monkeypatch.setattr(mod, "read_wout", lambda _path: _synthetic_wout(nfp=2, rbc01=0.95))

    with pytest.raises(RuntimeError, match="not the raw input boundary"):
        mod._validate_case_initial_wout(case)


def test_readme_renderer_allows_only_known_raw_wout_artifact_exception(monkeypatch, tmp_path: Path) -> None:
    mod = _load_module()
    case = _synthetic_case(mod, tmp_path, "NFP=3 seed 3127", nfp=3)
    case = mod.QICase(
        label=case.label,
        input_file=case.input_file,
        output_dir=case.output_dir,
        initial_wout=case.initial_wout,
        note=case.note,
        raw_initial_wout_exception="synthetic exception",
    )
    monkeypatch.setattr(mod, "read_wout", lambda _path: _synthetic_wout(nfp=3, rbc01=0.95))

    with pytest.raises(RuntimeError, match="not the raw input boundary"):
        mod._validate_case_initial_wout(case)

    monkeypatch.setattr(mod, "KNOWN_RAW_WOUT_ARTIFACTS", {case.initial_wout.resolve()})
    mod._validate_case_initial_wout(case)
