from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.diagnostics import compare_profile_reports as compare_tool


MIB = 1024 * 1024


def _callback_report(
    *,
    total_wall_time_s: float,
    samples: int,
    rss_peak_mib: int,
    replay_wall_time_s: float,
    accepted_replays: int,
    solve_count: int,
    cache_entry_growth: int,
    solver_device: str,
) -> dict:
    return {
        "schema_version": 2,
        "report_kind": "exact_optimizer_callback_profile",
        "problem": "qh",
        "max_mode": 2,
        "callback": "jacobian",
        "solver_device_resolved": solver_device,
        "total_wall_time_s": total_wall_time_s,
        "rss_before_bytes": 120 * MIB,
        "rss_after_bytes": rss_peak_mib * MIB,
        "samples": [{"repeat": repeat, "wall_time_s": total_wall_time_s / samples} for repeat in range(samples)],
        "profile": {
            "xla_compile": {"count": 1, "wall_time_s": 1.25, "mean_wall_time_s": 1.25},
            "exact_tape_build": {"count": accepted_replays, "wall_time_s": 3.0, "mean_wall_time_s": 1.5},
            "jacobian_tape_replay": {
                "count": accepted_replays,
                "wall_time_s": replay_wall_time_s,
                "mean_wall_time_s": replay_wall_time_s / accepted_replays,
            },
            "exact_unpack_cache": {"count": accepted_replays, "wall_time_s": 0.4, "mean_wall_time_s": 0.2},
            "exact_solve_with_tape_total": {
                "count": accepted_replays,
                "wall_time_s": 3.5,
                "mean_wall_time_s": 1.75,
            },
            "solve_forward_trial": {
                "count": solve_count - accepted_replays,
                "wall_time_s": 1.0,
                "mean_wall_time_s": 1.0,
            },
        },
        "cache": {
            "growth": {
                "total_entries_after": 12 + cache_entry_growth,
                "total_entries_delta": cache_entry_growth,
            }
        },
    }


def test_callback_report_summary_extracts_bottleneck_metrics() -> None:
    summary = compare_tool.summarize_payload(
        _callback_report(
            total_wall_time_s=10.0,
            samples=2,
            rss_peak_mib=256,
            replay_wall_time_s=2.0,
            accepted_replays=2,
            solve_count=3,
            cache_entry_growth=4,
            solver_device="cpu",
        ),
        path=Path("cpu.json"),
        label="cpu",
    )

    metrics = summary["metrics"]
    assert metrics["total_runtime_s"] == 10.0
    assert metrics["compile_time_s"] == 1.25
    assert metrics["replay_time_s"] == 2.0
    assert metrics["cache_time_s"] == 0.4
    assert metrics["callback_count"] == 2
    assert metrics["rss_peak_mib"] == 256.0
    assert metrics["solve_count"] == 3
    assert metrics["accepted_point_replay_count"] == 2
    assert metrics["cache_entry_growth"] == 4
    assert summary["top_profile"][0]["name"] == "exact_solve_with_tape_total"


def test_comparison_reports_ratios_against_baseline() -> None:
    cpu = compare_tool.summarize_payload(
        _callback_report(
            total_wall_time_s=10.0,
            samples=2,
            rss_peak_mib=256,
            replay_wall_time_s=2.0,
            accepted_replays=2,
            solve_count=3,
            cache_entry_growth=4,
            solver_device="cpu",
        ),
        label="cpu",
    )
    gpu = compare_tool.summarize_payload(
        _callback_report(
            total_wall_time_s=25.0,
            samples=3,
            rss_peak_mib=512,
            replay_wall_time_s=5.0,
            accepted_replays=3,
            solve_count=5,
            cache_entry_growth=6,
            solver_device="gpu",
        ),
        label="gpu",
    )

    comparison = compare_tool.build_comparison([cpu, gpu], baseline="cpu")
    ratios = comparison["comparisons"][0]["ratios"]

    assert ratios["total_runtime_s"] == 2.5
    assert ratios["replay_time_s"] == 2.5
    assert ratios["callback_count"] == 1.5
    assert ratios["rss_peak_mib"] == 2.0
    assert ratios["solve_count"] == pytest.approx(5.0 / 3.0)
    assert ratios["accepted_point_replay_count"] == 1.5

    text = compare_tool.format_text(comparison)
    assert "Profile report comparison" in text
    assert "gpu" in text
    assert "2.500x" in text


def test_repeated_run_report_aggregates_runs() -> None:
    payload = {
        "problem": "qa",
        "max_mode": 1,
        "run_repeats": 2,
        "runs": [
            {
                "total_wall_time_s": 2.0,
                "nfev": 2,
                "njev": 1,
                "profile": {
                    "jacobian_tape_replay": {"count": 1, "wall_time_s": 0.75},
                    "exact_solve_with_tape_total": {"count": 1, "wall_time_s": 1.25},
                },
            },
            {
                "total_wall_time_s": 3.0,
                "nfev": 1,
                "njev": 1,
                "profile": {
                    "jacobian_tape_replay": {"count": 1, "wall_time_s": 1.25},
                    "solve_forward_trial": {"count": 2, "wall_time_s": 0.8},
                },
            },
        ],
    }

    summary = compare_tool.summarize_payload(payload, label="warm")

    assert summary["metadata"]["source_report_kind"] == "exact_optimizer_run_repeats"
    assert summary["metrics"]["total_runtime_s"] == 5.0
    assert summary["metrics"]["callback_count"] == 5
    assert summary["metrics"]["replay_time_s"] == 2.0
    assert summary["metrics"]["accepted_point_replay_count"] == 2
    assert summary["metrics"]["solve_count"] == 3


def test_cli_prints_text_and_writes_json(tmp_path: Path, capsys) -> None:
    cpu_path = tmp_path / "cpu.json"
    gpu_path = tmp_path / "gpu.json"
    out_path = tmp_path / "comparison.json"
    cpu_path.write_text(
        json.dumps(
            _callback_report(
                total_wall_time_s=10.0,
                samples=2,
                rss_peak_mib=256,
                replay_wall_time_s=2.0,
                accepted_replays=2,
                solve_count=3,
                cache_entry_growth=4,
                solver_device="cpu",
            )
        ),
        encoding="utf-8",
    )
    gpu_path.write_text(
        json.dumps(
            _callback_report(
                total_wall_time_s=25.0,
                samples=3,
                rss_peak_mib=512,
                replay_wall_time_s=5.0,
                accepted_replays=3,
                solve_count=5,
                cache_entry_growth=6,
                solver_device="gpu",
            )
        ),
        encoding="utf-8",
    )

    rc = compare_tool.main(
        [
            str(cpu_path),
            str(gpu_path),
            "--label",
            "cpu",
            "--label",
            "gpu",
            "--json-out",
            str(out_path),
        ]
    )

    assert rc == 0
    stdout = capsys.readouterr().out
    assert "Ratios vs baseline" in stdout
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written["report_kind"] == "profile_report_comparison"
    assert written["baseline_label"] == "cpu"
    assert written["comparisons"][0]["ratios"]["total_runtime_s"] == 2.5
