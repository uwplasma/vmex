#!/usr/bin/env python3
"""Compare tape and scan exact-optimizer callback profiles.

This is an offline report helper for profiles produced by
``tools/diagnostics/profile_exact_optimizer.py``.  It answers the practical GPU
question: how many accepted exact callbacks are needed before the high-compile
``exact_path="scan"`` path amortizes relative to the default tape path?
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tape", required=True, help="JSON callback profile for exact_path=tape/default.")
    parser.add_argument("--scan", required=True, help="JSON callback profile for exact_path=scan.")
    parser.add_argument("--json-out", default="", help="Optional path to write the normalized comparison JSON.")
    return parser


def _load(path: str | Path) -> dict[str, Any]:
    with open(Path(path).expanduser(), encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("report_kind") != "exact_optimizer_callback_profile":
        raise ValueError(f"{path} is not an exact optimizer callback profile")
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"{path} does not contain callback samples")
    return payload


def _wall_times(payload: dict[str, Any]) -> list[float]:
    return [float(sample["wall_time_s"]) for sample in payload["samples"]]


def _profile_metric(payload: dict[str, Any], *names: str) -> float:
    profile = payload.get("profile", {})
    if not isinstance(profile, dict):
        return 0.0
    return float(sum(float(profile.get(name, {}).get("wall_time_s", 0.0)) for name in names))


def _summary(payload: dict[str, Any]) -> dict[str, Any]:
    walls = _wall_times(payload)
    warm = walls[1:] if len(walls) > 1 else []
    return {
        "path": str(payload.get("exact_path_requested") or payload.get("runtime", {}).get("vmec_jax_opt_exact_path") or "unknown"),
        "callback": payload.get("callback"),
        "solver_device": payload.get("solver_device_resolved"),
        "dofs": int(payload.get("dofs", 0)),
        "cold_wall_s": walls[0],
        "warm_mean_wall_s": (sum(warm) / len(warm)) if warm else None,
        "warm_min_wall_s": min(warm) if warm else None,
        "sample_count": len(walls),
        "total_wall_s": float(payload.get("total_wall_time_s", sum(walls))),
        "accepted_replay_s": _profile_metric(
            payload,
            "jacobian_tape_replay",
            "jacobian_projected_replay_total",
            "scan_jacobian_total",
        ),
        "exact_tape_build_s": _profile_metric(payload, "exact_tape_build"),
        "preconditioner_s": _profile_metric(payload, "exact_tape_solver_preconditioner"),
    }


def _break_even_callbacks(tape: dict[str, Any], scan: dict[str, Any]) -> int | None:
    tape_warm = tape["warm_min_wall_s"]
    scan_warm = scan["warm_min_wall_s"]
    if tape_warm is None or scan_warm is None:
        return None
    delta_cold = float(scan["cold_wall_s"]) - float(tape["cold_wall_s"])
    delta_warm = float(tape_warm) - float(scan_warm)
    if delta_warm <= 0.0:
        return None
    # total(n) = cold + (n - 1) * warm.  Return the first integer n where
    # scan total is strictly lower than tape total.
    n_float = 1.0 + delta_cold / delta_warm
    return max(2, int(math.floor(n_float) + 1))


def compare(tape_payload: dict[str, Any], scan_payload: dict[str, Any]) -> dict[str, Any]:
    tape = _summary(tape_payload)
    scan = _summary(scan_payload)
    break_even = _break_even_callbacks(tape, scan)
    return {
        "schema_version": 1,
        "report_kind": "exact_path_profile_comparison",
        "tape": tape,
        "scan": scan,
        "break_even_callbacks": break_even,
        "recommendation": _recommendation(tape, scan, break_even),
    }


def _recommendation(tape: dict[str, Any], scan: dict[str, Any], break_even: int | None) -> str:
    if break_even is None:
        return "keep_tape_default"
    if break_even <= 3:
        return "consider_scan_for_short_gpu_runs_after_validation"
    return "use_scan_only_for_long_warm_gpu_runs"


def _format(report: dict[str, Any]) -> str:
    tape = report["tape"]
    scan = report["scan"]
    lines = [
        "Exact optimizer path comparison",
        f"  callback:       {tape['callback']}",
        f"  device:         {tape['solver_device']}",
        f"  dofs:           {tape['dofs']}",
        f"  tape cold/warm: {tape['cold_wall_s']:.3f}s / {tape['warm_min_wall_s']:.3f}s",
        f"  scan cold/warm: {scan['cold_wall_s']:.3f}s / {scan['warm_min_wall_s']:.3f}s",
        f"  break-even:     {report['break_even_callbacks'] or 'not reached'} accepted callbacks",
        f"  recommendation: {report['recommendation']}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = compare(_load(args.tape), _load(args.scan))
    print(_format(report))
    if args.json_out:
        out = Path(args.json_out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
