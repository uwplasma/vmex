"""Compare two ``example_runtime_memory_matrix.py`` summaries.

The PR-readiness benchmark uses this script to keep the current-branch versus
``origin/main`` comparison reproducible.  The summary producer records detailed
process metrics but not a solved-equilibrium residual, so this comparison only
claims status/runtime/memory regressions; WOUT residual parity remains a
separate gate.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUNTIME_REGRESSION_RATIO = 1.10
MEMORY_REGRESSION_RATIO = 1.15


@dataclass(frozen=True)
class ComparisonRow:
    case_id: str
    backend: str
    current_ok: bool | None
    baseline_ok: bool | None
    convergence_changed: bool
    cold_runtime_current_s: float | None
    cold_runtime_baseline_s: float | None
    cold_runtime_ratio: float | None
    warm_runtime_current_s: float | None
    warm_runtime_baseline_s: float | None
    warm_runtime_ratio: float | None
    peak_memory_current_bytes: int | None
    peak_memory_baseline_bytes: int | None
    peak_memory_ratio: float | None
    regression: bool
    classification: str
    residual_available: bool
    note: str


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _records(summary: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(row["case_id"]), str(row["backend"])): row for row in summary.get("results", [])}


def _float(row: dict[str, Any] | None, key: str) -> float | None:
    if row is None:
        return None
    value = row.get(key)
    if isinstance(value, (int, float)) and value > 0:
        return float(value)
    return None


def _mem(row: dict[str, Any] | None) -> int | None:
    if row is None:
        return None
    for key in ("peak_footprint_bytes", "max_rss_bytes"):
        value = row.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _ratio(current: float | int | None, baseline: float | int | None) -> float | None:
    if current is None or baseline is None or baseline <= 0:
        return None
    return float(current) / float(baseline)


def _classify(
    *,
    backend: str,
    current: dict[str, Any] | None,
    baseline: dict[str, Any] | None,
    cold_ratio: float | None,
    warm_ratio: float | None,
    mem_ratio: float | None,
) -> tuple[bool, str, str]:
    if current is None:
        return True, "missing_current", "Current branch did not produce this backend/case record."
    if baseline is None:
        return False, "new_current_record", "Baseline did not produce this backend/case record."
    current_ok = bool(current.get("ok"))
    baseline_ok = bool(baseline.get("ok"))
    if current_ok != baseline_ok:
        return True, "convergence_status_changed", "Backend success/convergence status changed."
    if not current_ok and not baseline_ok:
        return False, "both_unavailable", "Backend unavailable or non-converged on both branches."
    if backend != "vmec_jax":
        return False, "external_backend_reference", "Reference-backend timing is recorded for context, not PR regression gating."

    regressions: list[str] = []
    if cold_ratio is not None and cold_ratio > RUNTIME_REGRESSION_RATIO:
        regressions.append("cold_runtime")
    if warm_ratio is not None and warm_ratio > RUNTIME_REGRESSION_RATIO:
        regressions.append("warm_runtime")
    if mem_ratio is not None and mem_ratio > MEMORY_REGRESSION_RATIO:
        regressions.append("peak_memory")
    if regressions:
        return True, "pending_profile:" + ",".join(regressions), "Material regression requires profiling/classification."
    return False, "within_threshold", "No material runtime or memory regression."


def compare(current_path: Path, baseline_path: Path) -> list[ComparisonRow]:
    current_summary = _load(current_path)
    baseline_summary = _load(baseline_path)
    current = _records(current_summary)
    baseline = _records(baseline_summary)
    keys = sorted(set(current) | set(baseline))
    rows: list[ComparisonRow] = []
    for case_id, backend in keys:
        cur = current.get((case_id, backend))
        base = baseline.get((case_id, backend))
        cold_current = _float(cur, "runtime_cold_s") or _float(cur, "runtime_s") or _float(cur, "time_real_s")
        cold_baseline = _float(base, "runtime_cold_s") or _float(base, "runtime_s") or _float(base, "time_real_s")
        warm_current = _float(cur, "runtime_warm_s")
        warm_baseline = _float(base, "runtime_warm_s")
        mem_current = _mem(cur)
        mem_baseline = _mem(base)
        cold_ratio = _ratio(cold_current, cold_baseline)
        warm_ratio = _ratio(warm_current, warm_baseline)
        mem_ratio = _ratio(mem_current, mem_baseline)
        regression, classification, note = _classify(
            backend=backend,
            current=cur,
            baseline=base,
            cold_ratio=cold_ratio,
            warm_ratio=warm_ratio,
            mem_ratio=mem_ratio,
        )
        rows.append(
            ComparisonRow(
                case_id=case_id,
                backend=backend,
                current_ok=None if cur is None else bool(cur.get("ok")),
                baseline_ok=None if base is None else bool(base.get("ok")),
                convergence_changed=(cur is None)
                or (base is None)
                or (bool(cur.get("ok")) != bool(base.get("ok"))),
                cold_runtime_current_s=cold_current,
                cold_runtime_baseline_s=cold_baseline,
                cold_runtime_ratio=cold_ratio,
                warm_runtime_current_s=warm_current,
                warm_runtime_baseline_s=warm_baseline,
                warm_runtime_ratio=warm_ratio,
                peak_memory_current_bytes=mem_current,
                peak_memory_baseline_bytes=mem_baseline,
                peak_memory_ratio=mem_ratio,
                regression=regression,
                classification=classification,
                residual_available=False,
                note=note,
            )
        )
    return rows


def _write_csv(rows: list[ComparisonRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _write_json(rows: list[ComparisonRow], path: Path, *, current: Path, baseline: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "current_summary": str(current),
            "baseline_summary": str(baseline),
            "runtime_regression_ratio": RUNTIME_REGRESSION_RATIO,
            "memory_regression_ratio": MEMORY_REGRESSION_RATIO,
            "residual_note": "example_runtime_memory_matrix.py does not record final residuals; use WOUT parity gates.",
        },
        "records": [asdict(row) for row in rows],
        "regressions": [asdict(row) for row in rows if row.regression],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()

    rows = compare(args.current, args.baseline)
    _write_csv(rows, args.csv_out)
    _write_json(rows, args.json_out, current=args.current, baseline=args.baseline)
    regressions = [row for row in rows if row.regression]
    print(f"rows={len(rows)} regressions={len(regressions)}")
    for row in regressions:
        print(f"{row.case_id} {row.backend}: {row.classification} {row.note}")


if __name__ == "__main__":
    main()
