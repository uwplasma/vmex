from __future__ import annotations

import json

from tools.diagnostics import compare_exact_path_profiles as tool


def _profile(*walls: float, exact_path: str = "auto") -> dict:
    return {
        "report_kind": "exact_optimizer_callback_profile",
        "callback": "jacobian",
        "dofs": 24,
        "exact_path_requested": exact_path,
        "solver_device_resolved": "gpu",
        "total_wall_time_s": sum(walls),
        "samples": [
            {"repeat": index, "wall_time_s": wall}
            for index, wall in enumerate(walls)
        ],
        "profile": {
            "exact_tape_build": {"wall_time_s": 1.0},
            "jacobian_tape_replay": {"wall_time_s": 2.0},
            "exact_tape_solver_preconditioner": {"wall_time_s": 0.5},
        },
    }


def test_break_even_callbacks_for_high_cold_fast_warm_scan() -> None:
    report = tool.compare(
        _profile(10.0, 2.5, exact_path="tape"),
        _profile(110.0, 1.25, exact_path="scan"),
    )

    assert report["break_even_callbacks"] == 82
    assert report["recommendation"] == "use_scan_only_for_long_warm_gpu_runs"
    assert report["tape"]["warm_min_wall_s"] == 2.5
    assert report["scan"]["warm_min_wall_s"] == 1.25


def test_no_break_even_when_scan_warm_is_not_faster() -> None:
    report = tool.compare(
        _profile(10.0, 2.0, exact_path="tape"),
        _profile(8.0, 2.1, exact_path="scan"),
    )

    assert report["break_even_callbacks"] is None
    assert report["recommendation"] == "keep_tape_default"


def test_cli_writes_json(tmp_path, capsys) -> None:
    tape = tmp_path / "tape.json"
    scan = tmp_path / "scan.json"
    out = tmp_path / "out.json"
    tape.write_text(json.dumps(_profile(10.0, 2.0, exact_path="tape")), encoding="utf-8")
    scan.write_text(json.dumps(_profile(12.0, 1.0, exact_path="scan")), encoding="utf-8")

    assert tool.main(["--tape", str(tape), "--scan", str(scan), "--json-out", str(out)]) == 0

    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["break_even_callbacks"] == 4
    assert "break-even" in capsys.readouterr().out
