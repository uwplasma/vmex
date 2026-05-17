#!/usr/bin/env python
"""Render completed common-minimal-seed showcase results.

The renderer is deliberately conservative: it only writes figures when
completed case directories exist under ``results/minimal_seed_showcase``.  The
history panel shows best-so-far objective values within each stage so rejected
trial points and objective switches are not displayed as false increases.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = SCRIPT_DIR / "results" / "minimal_seed_showcase"
FIGURE_DIR = ROOT / "docs" / "_static" / "figures"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_minimal_seed_showcase import DEFAULT_CASE_ORDER


@dataclass(frozen=True)
class ShowcaseRecord:
    """Completed or attempted minimal-seed optimization record."""

    case_name: str
    nfp: int
    problem: str
    output_dir: Path
    success: bool
    crashed: bool
    objective_final: float | None
    aspect_final: float | None
    iota_final: float | None
    total_wall_time_s: float | None
    policy: str
    max_mode: int
    use_ess: bool
    qi_legacy_total: float | None = None
    qi_mirror_ratio_max: float | None = None
    qi_max_elongation: float | None = None


def _float_or_none(value) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _bool_value(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _metadata_for_result(result_path: Path) -> dict:
    meta_path = result_path.parent / "showcase_case.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return {}


def load_records(output_root: Path = RESULTS_ROOT) -> list[ShowcaseRecord]:
    """Load all minimal-seed case records under ``output_root``."""

    records: list[ShowcaseRecord] = []
    for result_path in sorted(Path(output_root).glob("**/case_result.json")):
        try:
            result = json.loads(result_path.read_text())
        except json.JSONDecodeError:
            continue
        metadata = _metadata_for_result(result_path)
        case_meta = metadata.get("minimal_seed_case", {})
        case_name = str(case_meta.get("name") or result_path.parent.name)
        nfp = int(case_meta.get("nfp") or result.get("input_nfp") or 0)
        records.append(
            ShowcaseRecord(
                case_name=case_name,
                nfp=nfp,
                problem=str(result.get("problem", "")),
                output_dir=result_path.parent,
                success=_bool_value(result.get("success")),
                crashed=_bool_value(result.get("crashed")),
                objective_final=_float_or_none(result.get("objective_final")),
                aspect_final=_float_or_none(result.get("aspect_final")),
                iota_final=_float_or_none(result.get("iota_final")),
                total_wall_time_s=_float_or_none(result.get("total_wall_time_s")),
                policy=str(result.get("policy", metadata.get("policy", ""))),
                max_mode=int(result.get("max_mode", metadata.get("max_mode", 0)) or 0),
                use_ess=_bool_value(result.get("use_ess", metadata.get("use_ess", False))),
                qi_legacy_total=_float_or_none(result.get("qi_legacy_total")),
                qi_mirror_ratio_max=_float_or_none(result.get("qi_mirror_ratio_max")),
                qi_max_elongation=_float_or_none(result.get("qi_max_elongation")),
            )
        )
    return records


def best_records(records: list[ShowcaseRecord], *, successful_only: bool = True) -> list[ShowcaseRecord]:
    """Return one successful lowest-objective record per minimal-seed case."""

    selected: list[ShowcaseRecord] = []
    for case_name in DEFAULT_CASE_ORDER:
        candidates = [
            record
            for record in records
            if record.case_name == case_name
            and (
                not successful_only
                or (record.success and not record.crashed and record.objective_final is not None)
            )
        ]
        if not candidates:
            continue
        selected.append(
            min(
                candidates,
                key=lambda record: (
                    not (record.success and not record.crashed),
                    float("inf") if record.objective_final is None else float(record.objective_final),
                ),
            )
        )
    return selected


def _history_stage_segments(history: list[dict]) -> list[list[dict]]:
    segments: list[list[dict]] = []
    current: list[dict] = []
    current_stage = object()
    for item in history:
        stage = item.get("stage", "")
        if current and stage != current_stage:
            segments.append(current)
            current = []
        current.append(item)
        current_stage = stage
    if current:
        segments.append(current)
    return segments


def objective_segments(record: ShowcaseRecord) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return monotone best-so-far objective segments for one record."""

    history_path = record.output_dir / "history.json"
    if not history_path.exists():
        return []
    try:
        data = json.loads(history_path.read_text())
    except json.JSONDecodeError:
        return []
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    for segment in _history_stage_segments(list(data.get("history", []))):
        if not segment:
            continue
        wall_min = np.asarray([float(item.get("wall_time_s", 0.0)) / 60.0 for item in segment], dtype=float)
        values = np.asarray(
            [max(float(item.get("objective", item.get("cost", np.nan))), 1.0e-16) for item in segment],
            dtype=float,
        )
        finite = np.isfinite(wall_min) & np.isfinite(values)
        if not finite.any():
            continue
        wall_min = wall_min[finite]
        values = values[finite]
        segments.append((wall_min, np.minimum.accumulate(values)))
    return segments


def write_summary_csv(records: list[ShowcaseRecord], path: Path) -> None:
    """Write a compact CSV summary for README/docs integration."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(
            [
                "case",
                "problem",
                "nfp",
                "policy",
                "max_mode",
                "ess",
                "success",
                "objective_final",
                "aspect_final",
                "iota_final",
                "wall_time_min",
                "qi_legacy_total",
                "qi_mirror_ratio_max",
                "qi_max_elongation",
                "output_dir",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record.case_name,
                    record.problem,
                    record.nfp,
                    record.policy,
                    record.max_mode,
                    "yes" if record.use_ess else "no",
                    "yes" if record.success and not record.crashed else "no",
                    "" if record.objective_final is None else f"{record.objective_final:.16e}",
                    "" if record.aspect_final is None else f"{record.aspect_final:.16e}",
                    "" if record.iota_final is None else f"{record.iota_final:.16e}",
                    "" if record.total_wall_time_s is None else f"{record.total_wall_time_s / 60.0:.6f}",
                    "" if record.qi_legacy_total is None else f"{record.qi_legacy_total:.16e}",
                    "" if record.qi_mirror_ratio_max is None else f"{record.qi_mirror_ratio_max:.16e}",
                    "" if record.qi_max_elongation is None else f"{record.qi_max_elongation:.16e}",
                    str(record.output_dir),
                ]
            )


def render_objective_panel(records: list[ShowcaseRecord], out_png: Path) -> Path | None:
    """Render best-so-far objective histories for completed records."""

    if not records:
        return None
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    ncols = 3
    nrows = int(np.ceil(len(records) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.2 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, record in zip(axes.ravel(), records, strict=False):
        ax.axis("on")
        segments = objective_segments(record)
        if not segments:
            ax.text(0.5, 0.5, "history missing", ha="center", va="center", transform=ax.transAxes)
        for wall_min, values in segments:
            ax.semilogy(wall_min, values, color="#1f4e79", linewidth=1.8)
            ax.scatter(wall_min[-1], values[-1], s=16, color="#d95f02", zorder=3)
        status = "ok" if record.success and not record.crashed else "failed"
        title = f"{record.case_name}: {record.policy}, m={record.max_mode}, {'ESS' if record.use_ess else 'no ESS'}, {status}"
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Wall time (min)")
        ax.set_ylabel("Best objective")
        ax.grid(True, alpha=0.25, linestyle=":")
    fig.suptitle("Common minimal-seed optimization histories", fontsize=13)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_png


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument("--summary-only", action="store_true", help="Write CSV only; skip Matplotlib rendering.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    all_records = best_records(load_records(args.output_root), successful_only=False)
    if not all_records:
        print(f"No minimal-seed showcase records found under {args.output_root}")
        return
    summary_csv = Path(args.figure_dir) / "minimal_seed_showcase_summary.csv"
    write_summary_csv(all_records, summary_csv)
    print(f"Wrote {summary_csv}")
    if not bool(args.summary_only):
        out_png = Path(args.figure_dir) / "minimal_seed_showcase_objective_panel.png"
        rendered = render_objective_panel(all_records, out_png)
        if rendered is not None:
            print(f"Wrote {rendered}")


if __name__ == "__main__":
    main()
