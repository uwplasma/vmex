#!/usr/bin/env python3
"""Compare vmec_jax profiling JSON reports.

The exact optimizer emits a few JSON shapes: callback-only profiles, short run
histories, and same-process repeated runs.  This tool normalizes those shapes
into production bottleneck metrics that can be compared across CPU/GPU or
before/after runs without launching another solver.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import json
from pathlib import Path
from typing import Any


PROFILE_TIME_GROUPS = {
    "compile_time_s": ("compile", "compilation"),
    "replay_time_s": ("replay",),
    "cache_time_s": ("cache",),
}

DIRECT_TIME_FIELDS = {
    "compile_time_s": (
        "compile_time_s",
        "compilation_time_s",
        "xla_compile_time_s",
        "compile_wall_time_s",
    ),
    "replay_time_s": ("replay_time_s", "tape_replay_time_s"),
    "cache_time_s": ("cache_time_s", "cache_wall_time_s"),
}

SOLVE_PROFILE_NAMES = {
    "solve_forward_trial",
    "solve_forward_exact",
    "exact_solve_with_tape_total",
    "scan_exact_state_solve",
}

ACCEPTED_REPLAY_PROFILE_NAMES = {
    "jacobian_tape_replay",
    "gradient_tape_replay",
    "state_tangent_tape_replay",
    "b_cartesian_tangent_tape_replay",
    "linear_operator_tape_vjp",
}

METRIC_ORDER = (
    "total_runtime_s",
    "vmec_solve_s",
    "qi_first_call_s",
    "qi_warm_min_s",
    "qi_warm_mean_s",
    "compile_time_s",
    "replay_time_s",
    "cache_time_s",
    "contamination_warning_count",
    "callback_count",
    "rss_peak_mib",
    "solve_count",
    "accepted_point_replay_count",
    "cache_entry_growth",
    "cache_entries_after",
)

METRIC_LABELS = {
    "total_runtime_s": "total runtime",
    "vmec_solve_s": "VMEC solve",
    "qi_first_call_s": "QI first call",
    "qi_warm_min_s": "QI warm min",
    "qi_warm_mean_s": "QI warm mean",
    "compile_time_s": "compile time",
    "replay_time_s": "replay time",
    "cache_time_s": "cache time",
    "contamination_warning_count": "warnings",
    "callback_count": "callbacks",
    "rss_peak_mib": "RSS peak",
    "solve_count": "solves",
    "accepted_point_replay_count": "accepted replays",
    "cache_entry_growth": "cache entry growth",
    "cache_entries_after": "cache entries after",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two or more vmec_jax profiling JSON reports and emit "
            "CPU/GPU or before/after bottleneck ratios."
        )
    )
    parser.add_argument("reports", nargs="+", type=Path, help="Profile JSON report paths.")
    parser.add_argument(
        "--label",
        action="append",
        default=None,
        help="Label for a report. Repeat once per input path.",
    )
    parser.add_argument(
        "--baseline",
        default="0",
        help="Baseline label or zero-based report index for ratios (default: 0).",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format printed to stdout.",
    )
    parser.add_argument("--json-out", type=Path, default=None, help="Optional path for machine-readable JSON.")
    parser.add_argument(
        "--top-profile",
        type=int,
        default=5,
        help="Number of largest profile terms to include per report.",
    )
    return parser


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _get_path(tree: Any, keys: Iterable[str]) -> Any:
    value = tree
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _sum_optional(values: Iterable[float | None]) -> float | None:
    total = 0.0
    found = False
    for value in values:
        if value is None:
            continue
        total += float(value)
        found = True
    return total if found else None


def _profile_from_payload(payload: dict[str, Any]) -> dict[str, dict[str, float | int]]:
    profile = payload.get("profile")
    if isinstance(profile, dict):
        return _normalize_profile(profile)

    runs = payload.get("runs")
    if isinstance(runs, list):
        merged: dict[str, dict[str, float | int]] = {}
        for run in runs:
            if not isinstance(run, dict):
                continue
            _merge_profile(merged, _profile_from_payload(run))
        return _finalize_profile(merged)

    return {}


def _normalize_profile(profile: dict[str, Any]) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    for name, rec in profile.items():
        if not isinstance(rec, dict):
            continue
        count = _as_int(rec.get("count")) or 0
        wall = _as_float(rec.get("wall_time_s")) or 0.0
        out[str(name)] = {
            "count": count,
            "wall_time_s": wall,
            "mean_wall_time_s": wall / count if count else 0.0,
        }
    return out


def _merge_profile(
    target: dict[str, dict[str, float | int]],
    source: dict[str, dict[str, float | int]],
) -> None:
    for name, rec in source.items():
        out = target.setdefault(name, {"count": 0, "wall_time_s": 0.0})
        out["count"] = int(out.get("count", 0)) + int(rec.get("count", 0))
        out["wall_time_s"] = float(out.get("wall_time_s", 0.0)) + float(rec.get("wall_time_s", 0.0))


def _finalize_profile(profile: dict[str, dict[str, float | int]]) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    for name, rec in sorted(profile.items()):
        count = int(rec.get("count", 0))
        wall = float(rec.get("wall_time_s", 0.0))
        out[name] = {
            "count": count,
            "wall_time_s": wall,
            "mean_wall_time_s": wall / count if count else 0.0,
        }
    return out


def _source_kind(payload: dict[str, Any]) -> str:
    kind = payload.get("report_kind")
    if isinstance(kind, str):
        return kind
    if isinstance(payload.get("runs"), list):
        return "exact_optimizer_run_repeats"
    if "wall_time_sec" in payload and "diagnostics" in payload:
        return "fixed_boundary_profile"
    if "qi_evaluations" in payload or "qi_resolution" in payload:
        return "qi_boozer_profile"
    if "history" in payload and "profile" in payload:
        return "exact_optimizer_run_history"
    if "profile" in payload:
        return "profile_report"
    return "unknown"


def _total_runtime(payload: dict[str, Any]) -> float | None:
    direct = next(
        (
            value
            for value in (
                _as_float(payload.get("total_wall_time_s")),
                _as_float(payload.get("wall_time_sec")),
                _as_float(payload.get("wall_time_s")),
                _as_float(payload.get("runtime_s")),
            )
            if value is not None
        ),
        None,
    )
    if direct is not None:
        return direct

    runs = payload.get("runs")
    if isinstance(runs, list):
        return _sum_optional(_total_runtime(run) for run in runs if isinstance(run, dict))
    return None


def _wall_time_metric(payload: dict[str, Any], key: str) -> float | None:
    value = _as_float(_get_path(payload, ("wall_time_s", key)))
    if value is not None:
        return value
    runs = payload.get("runs")
    if isinstance(runs, list):
        return _sum_optional(_wall_time_metric(run, key) for run in runs if isinstance(run, dict))
    return None


def _contamination_warning_count(payload: dict[str, Any]) -> int | None:
    warnings = payload.get("contamination_warnings")
    if isinstance(warnings, list):
        return len(warnings)
    runs = payload.get("runs")
    if isinstance(runs, list):
        values = [_contamination_warning_count(run) for run in runs if isinstance(run, dict)]
        if any(value is not None for value in values):
            return sum(int(value or 0) for value in values)
    return None


def _first_present(*values: float | None) -> float | None:
    for value in values:
        if value is not None:
            return value
    return None


def _direct_time(payload: dict[str, Any], field: str) -> float | None:
    candidates = list(DIRECT_TIME_FIELDS[field])
    current = next(
        (
            value
            for key in candidates
            for value in (
                _as_float(payload.get(key)),
                _as_float(_get_path(payload, ("timing", key))),
                _as_float(_get_path(payload, ("diagnostics", "timing", key))),
            )
            if value is not None
        ),
        None,
    )
    if current is not None:
        return current
    runs = payload.get("runs")
    if isinstance(runs, list):
        return _sum_optional(_direct_time(run, field) for run in runs if isinstance(run, dict))
    return None


def _profile_time(profile: dict[str, dict[str, float | int]], field: str) -> float | None:
    tokens = PROFILE_TIME_GROUPS[field]
    values = [
        float(rec.get("wall_time_s", 0.0))
        for name, rec in profile.items()
        if any(token in name.lower() for token in tokens)
    ]
    return sum(values) if values else None


def _callback_count(payload: dict[str, Any]) -> int | None:
    trace = payload.get("callback_trace")
    if isinstance(trace, dict):
        events = trace.get("events")
        if isinstance(events, list):
            return len(events)
        summary = trace.get("summary")
        if isinstance(summary, dict):
            total = 0
            found = False
            for rec in summary.values():
                if isinstance(rec, dict) and "count" in rec:
                    total += int(rec.get("count", 0))
                    found = True
            if found:
                return total

    samples = payload.get("samples")
    if isinstance(samples, list):
        return len(samples)

    runs = payload.get("runs")
    if isinstance(runs, list):
        values = [_callback_count(run) for run in runs if isinstance(run, dict)]
        if any(value is not None for value in values):
            return sum(int(value or 0) for value in values)

    nfev = _as_int(payload.get("nfev"))
    njev = _as_int(payload.get("njev"))
    if nfev is not None or njev is not None:
        return int(nfev or 0) + int(njev or 0)
    return None


def _accepted_replay_count(payload: dict[str, Any], profile: dict[str, dict[str, float | int]]) -> int | None:
    trace = payload.get("callback_trace")
    if isinstance(trace, dict):
        summary = trace.get("summary")
        if isinstance(summary, dict):
            total = 0
            found = False
            for key, rec in summary.items():
                if not isinstance(rec, dict):
                    continue
                key_l = str(key).lower()
                if "exact_tape_replay" in key_l or key_l.endswith(":tape_replay"):
                    total += int(rec.get("count", 0))
                    found = True
            if found:
                return total

    if any(name in profile for name in ACCEPTED_REPLAY_PROFILE_NAMES):
        return sum(int(profile[name].get("count", 0)) for name in ACCEPTED_REPLAY_PROFILE_NAMES if name in profile)

    runs = payload.get("runs")
    if isinstance(runs, list):
        values = [
            _accepted_replay_count(run, _profile_from_payload(run))
            for run in runs
            if isinstance(run, dict)
        ]
        if any(value is not None for value in values):
            return sum(int(value or 0) for value in values)
    return None


def _solve_count(payload: dict[str, Any], profile: dict[str, dict[str, float | int]]) -> int | None:
    direct = _as_int(payload.get("solve_count"))
    if direct is not None:
        return direct

    if any(name in profile for name in SOLVE_PROFILE_NAMES):
        return sum(int(profile[name].get("count", 0)) for name in SOLVE_PROFILE_NAMES if name in profile)

    trace = payload.get("callback_trace")
    if isinstance(trace, dict):
        summary = trace.get("summary")
        if isinstance(summary, dict):
            total = 0
            found = False
            for key, rec in summary.items():
                if not isinstance(rec, dict):
                    continue
                key_l = str(key).lower()
                if "trial_solve" in key_l or "exact_tape_replay" in key_l:
                    total += int(rec.get("count", 0))
                    found = True
            if found:
                return total

    runs = payload.get("runs")
    if isinstance(runs, list):
        values = [_solve_count(run, _profile_from_payload(run)) for run in runs if isinstance(run, dict)]
        if any(value is not None for value in values):
            return sum(int(value or 0) for value in values)
    return None


def _rss_peak_bytes(payload: dict[str, Any]) -> int | None:
    values: list[int] = []

    def walk(value: Any, key: str | None = None) -> None:
        if isinstance(value, dict):
            for child_key, child in value.items():
                walk(child, str(child_key))
            return
        if isinstance(value, list):
            for child in value:
                walk(child, key)
            return
        if key is None:
            return
        key_l = key.lower()
        numeric = _as_float(value)
        if numeric is None:
            return
        if key_l in {"rss_peak_bytes", "max_rss_bytes", "rss_before_bytes", "rss_after_bytes"}:
            values.append(int(numeric))
        elif key_l in {"rss_peak_mb", "rss_peak_mib", "max_rss_mb", "max_rss_mib"}:
            values.append(int(float(numeric) * 1024 * 1024))

    walk(payload)
    return max(values) if values else None


def _cache_entries_after(payload: dict[str, Any]) -> int | None:
    candidates = (
        _get_path(payload, ("cache", "growth", "total_entries_after")),
        _get_path(payload, ("cache_growth", "total_entries_after")),
        payload.get("cache_entries_after"),
    )
    values = [_as_int(candidate) for candidate in candidates]
    value = next((item for item in values if item is not None), None)
    if value is not None:
        return value
    runs = payload.get("runs")
    if isinstance(runs, list):
        run_values = [_cache_entries_after(run) for run in runs if isinstance(run, dict)]
        run_values = [item for item in run_values if item is not None]
        if run_values:
            return int(run_values[-1])
    return None


def _cache_entry_growth(payload: dict[str, Any]) -> int | None:
    candidates = (
        _get_path(payload, ("cache", "growth", "total_entries_delta")),
        _get_path(payload, ("cache_growth", "total_entries_delta")),
        payload.get("cache_entry_growth"),
    )
    values = [_as_int(candidate) for candidate in candidates]
    value = next((item for item in values if item is not None), None)
    if value is not None:
        return value
    runs = payload.get("runs")
    if isinstance(runs, list):
        run_values = [_cache_entry_growth(run) for run in runs if isinstance(run, dict)]
        if any(item is not None for item in run_values):
            return sum(int(item or 0) for item in run_values)
    return None


def _top_profile(
    profile: dict[str, dict[str, float | int]],
    *,
    limit: int,
) -> list[dict[str, float | int | str]]:
    rows = [
        {
            "name": name,
            "count": int(rec.get("count", 0)),
            "wall_time_s": float(rec.get("wall_time_s", 0.0)),
            "mean_wall_time_s": float(rec.get("mean_wall_time_s", 0.0)),
        }
        for name, rec in profile.items()
    ]
    rows.sort(key=lambda row: float(row["wall_time_s"]), reverse=True)
    return rows[: max(0, int(limit))]


def summarize_payload(
    payload: dict[str, Any],
    *,
    path: Path | None = None,
    label: str | None = None,
    top_profile: int = 5,
) -> dict[str, Any]:
    profile = _profile_from_payload(payload)
    compile_time = _direct_time(payload, "compile_time_s")
    replay_time = _direct_time(payload, "replay_time_s")
    cache_time = _direct_time(payload, "cache_time_s")
    if compile_time is None:
        compile_time = _profile_time(profile, "compile_time_s")
    if replay_time is None:
        replay_time = _profile_time(profile, "replay_time_s")
    if cache_time is None:
        cache_time = _profile_time(profile, "cache_time_s")

    rss_peak = _rss_peak_bytes(payload)
    metrics = {
        "total_runtime_s": _total_runtime(payload),
        "vmec_solve_s": _wall_time_metric(payload, "vmec_solve"),
        "qi_first_call_s": _first_present(
            _wall_time_metric(payload, "qi_first_call"),
            _wall_time_metric(payload, "qi_first"),
        ),
        "qi_warm_min_s": _wall_time_metric(payload, "qi_warm_min"),
        "qi_warm_mean_s": _wall_time_metric(payload, "qi_warm_mean"),
        "compile_time_s": compile_time,
        "replay_time_s": replay_time,
        "cache_time_s": cache_time,
        "contamination_warning_count": _contamination_warning_count(payload),
        "callback_count": _callback_count(payload),
        "rss_peak_bytes": rss_peak,
        "rss_peak_mib": None if rss_peak is None else rss_peak / (1024.0 * 1024.0),
        "solve_count": _solve_count(payload, profile),
        "accepted_point_replay_count": _accepted_replay_count(payload, profile),
        "cache_entries_after": _cache_entries_after(payload),
        "cache_entry_growth": _cache_entry_growth(payload),
    }

    runtime = payload.get("runtime") if isinstance(payload.get("runtime"), dict) else {}
    metadata = {
        "source_report_kind": _source_kind(payload),
        "problem": payload.get("problem"),
        "max_mode": payload.get("max_mode"),
        "callback": payload.get("callback"),
        "method": payload.get("method"),
        "solver_device": (
            payload.get("solver_device_resolved")
            or payload.get("solver_device")
            or _get_path(payload, ("args", "solver_device"))
        ),
        "jax_default_backend": payload.get("jax_default_backend") or runtime.get("default_backend"),
        "jax_version": payload.get("jax_version") or runtime.get("jax_version"),
        "active_gpu": payload.get("active_gpu") if "active_gpu" in payload else runtime.get("active_gpu"),
        "jit_booz": _get_path(payload, ("qi_resolution", "jit_booz")),
        "contamination_warnings": payload.get("contamination_warnings"),
        "run_repeats": payload.get("run_repeats"),
    }
    return {
        "label": label,
        "path": None if path is None else str(path),
        "metadata": metadata,
        "metrics": metrics,
        "top_profile": _top_profile(profile, limit=top_profile),
    }


def _ratio(value: Any, baseline: Any) -> float | None:
    value_f = _as_float(value)
    baseline_f = _as_float(baseline)
    if value_f is None or baseline_f is None or baseline_f == 0.0:
        return None
    return value_f / baseline_f


def build_comparison(
    summaries: list[dict[str, Any]],
    *,
    baseline: str = "0",
) -> dict[str, Any]:
    if len(summaries) < 2:
        raise ValueError("at least two reports are required for comparison")
    baseline_index = _resolve_baseline(summaries, baseline)
    base = summaries[baseline_index]
    base_metrics = base["metrics"]
    comparisons: list[dict[str, Any]] = []
    for index, summary in enumerate(summaries):
        if index == baseline_index:
            continue
        metrics = summary["metrics"]
        ratios = {
            key: _ratio(metrics.get(key), base_metrics.get(key))
            for key in METRIC_ORDER
            if key in metrics and key in base_metrics
        }
        deltas = {
            key: (
                None
                if _as_float(metrics.get(key)) is None or _as_float(base_metrics.get(key)) is None
                else float(metrics[key]) - float(base_metrics[key])
            )
            for key in METRIC_ORDER
            if key in metrics and key in base_metrics
        }
        comparisons.append(
            {
                "label": summary["label"],
                "baseline_label": base["label"],
                "ratios": ratios,
                "deltas": deltas,
            }
        )
    return {
        "schema_version": 1,
        "report_kind": "profile_report_comparison",
        "baseline_label": base["label"],
        "reports": summaries,
        "comparisons": comparisons,
    }


def _resolve_baseline(summaries: list[dict[str, Any]], baseline: str) -> int:
    try:
        index = int(str(baseline))
    except ValueError:
        index = -1
    if 0 <= index < len(summaries):
        return index
    for idx, summary in enumerate(summaries):
        if str(summary.get("label")) == str(baseline):
            return idx
    labels = ", ".join(str(summary.get("label")) for summary in summaries)
    raise ValueError(f"baseline {baseline!r} does not match index or label; available labels: {labels}")


def _format_value(value: Any, metric: str) -> str:
    if value is None:
        return "n/a"
    value_f = _as_float(value)
    if value_f is None:
        return str(value)
    if metric.endswith("_count") or metric in {
        "callback_count",
        "solve_count",
        "cache_entry_growth",
        "cache_entries_after",
    }:
        return str(int(round(value_f)))
    if metric.endswith("_mib"):
        return f"{value_f:.1f}"
    if metric.endswith("_s"):
        return f"{value_f:.3f}"
    return f"{value_f:.3f}"


def _format_ratio(value: Any) -> str:
    value_f = _as_float(value)
    if value_f is None:
        return "n/a"
    return f"{value_f:.3f}x"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [
        max(len(headers[col]), *(len(row[col]) for row in rows)) if rows else len(headers[col])
        for col in range(len(headers))
    ]
    lines = ["  ".join(headers[col].ljust(widths[col]) for col in range(len(headers)))]
    lines.append("  ".join("-" * widths[col] for col in range(len(headers))))
    for row in rows:
        lines.append("  ".join(row[col].ljust(widths[col]) for col in range(len(headers))))
    return "\n".join(lines)


def format_text(comparison: dict[str, Any]) -> str:
    reports = list(comparison["reports"])
    baseline_label = str(comparison["baseline_label"])
    lines = [
        "Profile report comparison",
        f"Baseline: {baseline_label}",
        "",
        "Reports:",
    ]
    report_rows = []
    for report in reports:
        metadata = report["metadata"]
        metrics = report["metrics"]
        report_rows.append(
            [
                str(report["label"]),
                str(metadata.get("source_report_kind") or "unknown"),
                str(metadata.get("solver_device") or metadata.get("jax_default_backend") or "unknown"),
                str(metadata.get("problem") or ""),
                str(metadata.get("callback") or metadata.get("method") or ""),
                _format_value(metrics.get("total_runtime_s"), "total_runtime_s"),
                _format_value(metrics.get("callback_count"), "callback_count"),
                _format_value(metrics.get("rss_peak_mib"), "rss_peak_mib"),
                _format_value(metrics.get("solve_count"), "solve_count"),
                _format_value(metrics.get("accepted_point_replay_count"), "accepted_point_replay_count"),
            ]
        )
    lines.append(
        _table(
            [
                "label",
                "kind",
                "device",
                "problem",
                "mode",
                "total_s",
                "callbacks",
                "rss_mib",
                "solves",
                "accepted_replays",
            ],
            report_rows,
        )
    )

    lines.extend(["", "Ratios vs baseline:"])
    ratio_headers = ["metric"] + [str(item["label"]) for item in comparison["comparisons"]]
    ratio_rows = []
    for metric in METRIC_ORDER:
        row = [METRIC_LABELS.get(metric, metric)]
        for item in comparison["comparisons"]:
            row.append(_format_ratio(item["ratios"].get(metric)))
        ratio_rows.append(row)
    lines.append(_table(ratio_headers, ratio_rows))

    lines.extend(["", "Top profile terms:"])
    for report in reports:
        entries = report.get("top_profile") or []
        if not entries:
            lines.append(f"  {report['label']}: n/a")
            continue
        total = _as_float(report["metrics"].get("total_runtime_s")) or 0.0
        formatted = []
        for entry in entries:
            wall = float(entry["wall_time_s"])
            share = "" if total <= 0.0 else f", {100.0 * wall / total:.1f}%"
            formatted.append(f"{entry['name']}={wall:.3f}s{share}")
        lines.append(f"  {report['label']}: " + "; ".join(formatted))
    return "\n".join(lines)


def _default_labels(paths: list[Path], labels: list[str] | None) -> list[str]:
    if labels is not None:
        if len(labels) != len(paths):
            raise ValueError("--label must be provided once per report when used")
        if len(set(labels)) != len(labels):
            raise ValueError("--label values must be unique")
        return labels

    counts: dict[str, int] = {}
    out: list[str] = []
    for path in paths:
        base = path.stem
        count = counts.get(base, 0)
        counts[base] = count + 1
        out.append(base if count == 0 else f"{base}_{count + 1}")
    return out


def load_summary(path: Path, *, label: str, top_profile: int = 5) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return summarize_payload(payload, path=path.resolve(), label=label, top_profile=top_profile)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        labels = _default_labels(args.reports, args.label)
        summaries = [
            load_summary(path.expanduser().resolve(), label=label, top_profile=int(args.top_profile))
            for path, label in zip(args.reports, labels, strict=True)
        ]
        comparison = build_comparison(summaries, baseline=str(args.baseline))
    except Exception as exc:
        raise SystemExit(f"compare_profile_reports: {exc}") from exc

    json_text = json.dumps(comparison, indent=2, sort_keys=True)
    if args.json_out is not None:
        json_path = args.json_out.expanduser().resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json_text + "\n", encoding="utf-8")

    if args.format == "json":
        print(json_text)
    else:
        print(format_text(comparison))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
