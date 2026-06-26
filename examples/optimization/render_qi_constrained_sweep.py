#!/usr/bin/env python
"""Render the constrained QI sweep matrix.

The QA/QH/QP/QI publication renderer has axes for backend, policy, max_mode,
and ESS.  QI has one additional policy axis: whether the constrained QI solve
starts from a same-mode QP preseed.  This renderer keeps that comparison
explicit and writes a diagnostics table used by the README selector.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import csv
import json
from pathlib import Path
import re

import numpy as np

from vmec_jax.quasi_isodynamic.diagnostics import QISeedSuitabilityTargets, qi_promotion_score


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
OUTPUT_ROOT = SCRIPT_DIR / "results" / "qs_ess_sweep"
FIGURE_DIR = REPO_ROOT / "docs" / "_static" / "figures"

BACKENDS = ("cpu", "gpu")
POLICIES = ("continuation", "direct")
PRESEED_OPTIONS = (True, False)
MODES = (1, 2, 3)
ESS_OPTIONS = (False, True)
QI_INPUT_NFP = 2
TARGET_ASPECT = 6.0
TARGET_ABS_IOTA_MIN = 0.41
QI_PROMOTION_MAX = 2.0e-2

SUMMARY_CSV = FIGURE_DIR / "qi_constrained_summary.csv"
SUMMARY_JSON = FIGURE_DIR / "qi_constrained_summary.json"
BEST_JSON = FIGURE_DIR / "qi_constrained_best.json"
OBJECTIVE_PNG = FIGURE_DIR / "qi_constrained_objective_panel.png"
OBJECTIVE_PDF = FIGURE_DIR / "qi_constrained_objective_panel.pdf"


SUMMARY_FIELDS = [
    "backend",
    "policy",
    "qi_qp_preseed",
    "qi_qi_preseed",
    "max_mode",
    "use_ess",
    "success",
    "crashed",
    "objective_final",
    "qi_raw_total",
    "qi_legacy_total",
    "qi_legacy_source",
    "qi_mirror_ratio_max",
    "qi_mirror_ratio_target",
    "qi_mirror_excess_max",
    "qi_max_elongation",
    "qi_elongation_target",
    "qi_elongation_excess",
    "qi_lgradb_min",
    "qi_lgradb_threshold",
    "qi_lgradb_excess_max",
    "aspect_final",
    "iota_final",
    "nfev",
    "njev",
    "total_wall_time_s",
    "jax_backend",
    "jax_device_kind",
    "input_file",
    "input_nfp",
    "project_input_boundary_to_max_mode",
    "target_aspect",
    "target_iota",
    "iota_abs_min",
    "iota_weight",
    "qi_lgradb_weight",
    "message",
    "output_dir",
]


_NFP_RE = re.compile(r"^\s*NFP\s*=\s*([0-9]+)", re.IGNORECASE | re.MULTILINE)


def _bool_value(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _float_value(value, default: float | None = None) -> float | None:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _qi_metric(row: dict, default: float | None = None, *, require_true_legacy: bool = False) -> float | None:
    """Return the explicitly labeled legacy-ranked QI metric.

    Older sweep artifacts used ``qi_raw_total`` for this value.  New artifacts
    also write ``qi_legacy_total`` so tables do not hide what is being ranked.
    """

    value = row.get("qi_legacy_total")
    if value in (None, ""):
        if require_true_legacy:
            return default
        value = row.get("qi_raw_total")
    return _float_value(value, default)


def _path_from_record(value: str | None, fallback: Path) -> Path:
    if not value:
        return fallback
    path = Path(value)
    resolved = path if path.is_absolute() else (REPO_ROOT / path).resolve()
    return resolved if resolved.exists() else fallback


def _repo_relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _input_nfp_from_output_dir(record: dict, path: Path) -> int | None:
    value = record.get("input_nfp")
    if value not in (None, ""):
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
    output_dir = _path_from_record(record.get("output_dir"), path.parent)
    for name in ("input.initial", "input.final"):
        candidate = output_dir / name
        if not candidate.exists():
            continue
        match = _NFP_RE.search(candidate.read_text(errors="ignore"))
        if match is not None:
            return int(match.group(1))
    return None


def _infer_qp_preseed(record: dict, path: Path) -> bool | None:
    value = record.get("qi_qp_preseed")
    if value not in (None, ""):
        return _bool_value(value)
    parts = set(path.parts)
    if "qp_preseed" in parts:
        return True
    if "no_qp_preseed" in parts:
        return False
    return None


def _discover_qi_results() -> list[dict]:
    rows_by_key: dict[tuple[str, str, bool, int, bool], tuple[float, dict]] = {}
    for path in sorted(OUTPUT_ROOT.glob("**/case_result.json")):
        record = json.loads(path.read_text())
        target_aspect = record.get("target_aspect")
        if target_aspect in (None, "") or abs(float(target_aspect) - TARGET_ASPECT) > 1.0e-8:
            continue
        if record.get("problem") != "qi":
            continue
        input_nfp = _input_nfp_from_output_dir(record, path)
        if input_nfp != QI_INPUT_NFP:
            continue
        if _bool_value(record.get("stellarator_asymmetric")):
            continue
        backend = str(record.get("backend") or "").lower()
        if backend not in BACKENDS:
            continue
        policy = str(record.get("policy") or "").lower()
        if policy not in POLICIES:
            continue
        qp_preseed = _infer_qp_preseed(record, path)
        if qp_preseed is None:
            continue
        row = {field: record.get(field) for field in SUMMARY_FIELDS}
        if row.get("qi_legacy_total") in (None, ""):
            row["qi_legacy_total"] = record.get("qi_raw_total")
            row["qi_legacy_source"] = "raw_fallback"
        else:
            row["qi_legacy_source"] = "legacy"
        row["backend"] = backend
        row["policy"] = policy
        row["qi_qp_preseed"] = bool(qp_preseed)
        row["max_mode"] = int(record.get("max_mode"))
        row["use_ess"] = bool(_bool_value(record.get("use_ess")))
        row["success"] = bool(_bool_value(record.get("success")))
        row["crashed"] = bool(_bool_value(record.get("crashed")))
        input_file = _path_from_record(
            row.get("input_file"),
            REPO_ROOT / "examples" / "data" / "input.minimal_seed_nfp2",
        )
        row["input_file"] = _repo_relative_path(input_file)
        row["input_nfp"] = input_nfp
        row["project_input_boundary_to_max_mode"] = (
            row.get("project_input_boundary_to_max_mode")
            if row.get("project_input_boundary_to_max_mode") not in (None, "")
            else True
        )
        row["output_dir"] = _repo_relative_path(_path_from_record(record.get("output_dir"), path.parent))
        if int(row["max_mode"]) not in MODES:
            continue
        key = (backend, policy, bool(qp_preseed), int(row["max_mode"]), bool(row["use_ess"]))
        previous = rows_by_key.get(key)
        mtime = path.stat().st_mtime
        if previous is None or mtime >= previous[0]:
            rows_by_key[key] = (mtime, row)
    return [
        row
        for _mtime, row in sorted(
            rows_by_key.values(),
            key=lambda item: (
                BACKENDS.index(item[1]["backend"]),
                POLICIES.index(item[1]["policy"]),
                0 if item[1]["qi_qp_preseed"] else 1,
                int(item[1]["max_mode"]),
                bool(item[1]["use_ess"]),
            ),
        )
    ]


def _write_summaries(rows: list[dict]) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(rows, indent=2) + "\n")
    with SUMMARY_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _history_for(row: dict) -> dict | None:
    output_dir = _path_from_record(row["output_dir"], REPO_ROOT)
    path = output_dir / "history.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


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


def _plotted_history_segments(row: dict, history: list[dict]) -> list[list[dict]]:
    """Return objective-history segments with a consistent objective definition."""

    segments = _history_stage_segments(history)
    if _bool_value(row.get("qi_qp_preseed")):
        qi_segments = [
            segment
            for segment in segments
            if segment and str(segment[0].get("stage", "")).startswith("QI ")
        ]
        return qi_segments or segments
    return segments


def _case_label(row: dict) -> str:
    return (
        f"{'ESS' if _bool_value(row['use_ess']) else 'No ESS'}: "
        f"J={_float_value(row.get('objective_final'), np.nan):.2e}, "
        f"QI={_qi_metric(row, np.nan):.2e}, "
        f"M={_float_value(row.get('qi_mirror_ratio_max'), np.nan):.2f}, "
        f"E={_float_value(row.get('qi_max_elongation'), np.nan):.1f}, "
        f"|i|={abs(_float_value(row.get('iota_final'), 0.0) or 0.0):.2f}, "
        f"{(_float_value(row.get('total_wall_time_s'), 0.0) or 0.0) / 60.0:.1f} min"
    )


def _plot_objective_panel(rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": ":",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )

    facets = [
        (backend, policy, qp_preseed)
        for backend in BACKENDS
        for policy in POLICIES
        for qp_preseed in PRESEED_OPTIONS
        if any(
            row["backend"] == backend and row["policy"] == policy and row["qi_qp_preseed"] == qp_preseed
            for row in rows
        )
    ]
    if not facets:
        raise FileNotFoundError(f"No constrained QI rows found under {OUTPUT_ROOT}")

    fig, axes = plt.subplots(
        len(facets),
        len(MODES),
        figsize=(4.8 * len(MODES), 3.35 * len(facets)),
        squeeze=False,
        sharey="row",
    )
    colors = {False: "#1f77b4", True: "#d95f02"}
    labels = {False: "No ESS", True: "ESS"}

    for row_index, (backend, policy, qp_preseed) in enumerate(facets):
        for col_index, max_mode in enumerate(MODES):
            ax = axes[row_index, col_index]
            if row_index == 0:
                ax.set_title(f"max_mode={max_mode}")
            if col_index == 0:
                ax.set_ylabel(
                    f"{backend.upper()} | {policy}\n"
                    f"{'QP preseed' if qp_preseed else 'no QP preseed'}\n"
                    "total objective"
                )
            annotation = []
            for use_ess in ESS_OPTIONS:
                matches = [
                    row
                    for row in rows
                    if row["backend"] == backend
                    and row["policy"] == policy
                    and row["qi_qp_preseed"] == qp_preseed
                    and int(row["max_mode"]) == max_mode
                    and bool(row["use_ess"]) == use_ess
                ]
                if not matches:
                    continue
                result = matches[0]
                hist = _history_for(result)
                if hist is None or not hist.get("history"):
                    annotation.append(f"{labels[use_ess]}: missing history")
                    continue
                segments = []
                for segment in _plotted_history_segments(result, hist["history"]):
                    values = np.minimum.accumulate(
                        np.asarray(
                            [
                                max(float(entry.get("objective", entry.get("cost", np.nan))), 1e-16)
                                for entry in segment
                            ],
                            dtype=float,
                        )
                    )
                    wall = np.asarray(
                        [float(entry.get("wall_time_s", 0.0)) / 60.0 for entry in segment],
                        dtype=float,
                    )
                    segments.append((wall, values))
                if not segments:
                    continue
                x_label = "wall time (min)" if any(np.any(wall) for wall, _values in segments) else "history index"
                linestyle = "-" if not result["crashed"] else "--"
                first_segment = True
                last_x = None
                last_value = None
                for wall, values in segments:
                    x_values = wall if x_label == "wall time (min)" else np.arange(len(values), dtype=float)
                    ax.semilogy(
                        x_values,
                        values,
                        color=colors[use_ess],
                        linestyle=linestyle,
                        linewidth=2.0 if use_ess else 1.75,
                        label=labels[use_ess] if first_segment and row_index == 0 and col_index == 0 else None,
                    )
                    first_segment = False
                    last_x = x_values[-1]
                    last_value = values[-1]
                if last_x is not None and last_value is not None:
                    ax.scatter(last_x, last_value, color=colors[use_ess], s=22, zorder=4)
                for boundary in hist.get("stage_boundaries", [])[:-1]:
                    try:
                        idx = int(boundary)
                    except (TypeError, ValueError):
                        continue
                    if 0 <= idx < len(hist["history"]):
                        x_boundary = (
                            float(hist["history"][idx].get("wall_time_s", 0.0)) / 60.0
                            if x_label == "wall time (min)"
                            else float(idx)
                        )
                        ax.axvline(x_boundary, color="0.7", linestyle=":", linewidth=0.9)
                annotation.append(_case_label(result))
                ax.set_xlabel(x_label)
            if annotation:
                ax.text(
                    0.02,
                    0.02,
                    "\n".join(annotation),
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=7.3,
                    bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "0.85", "alpha": 0.92},
                )
            ax.grid(True, which="both", alpha=0.22)

    handles, labels_out = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels_out, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.998))
    fig.suptitle("Constrained QI sweep: ESS, QP preseed, continuation/direct, CPU/GPU", y=1.004, fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    fig.savefig(OBJECTIVE_PNG, dpi=220, bbox_inches="tight")
    fig.savefig(OBJECTIVE_PDF, bbox_inches="tight")
    plt.close(fig)


def _best_score(row: dict) -> tuple:
    targets = QISeedSuitabilityTargets(
        smooth_qi_max=QI_PROMOTION_MAX,
        legacy_qi_max=QI_PROMOTION_MAX,
        target_aspect=TARGET_ASPECT,
        aspect_relative_tolerance=0.05,
        abs_iota_min=TARGET_ABS_IOTA_MIN,
        mirror_ratio_max=_float_value(row.get("qi_mirror_ratio_target"), 0.21),
        max_elongation=_float_value(row.get("qi_elongation_target"), 8.0),
    )
    return qi_promotion_score(row, targets=targets, require_legacy_source=True)


def _write_best(rows: list[dict]) -> dict:
    if not rows:
        raise FileNotFoundError("No QI rows available for best-case selection")
    best = min(rows, key=_best_score)
    payload = dict(best)
    score = []
    for item in _best_score(best):
        if isinstance(item, float) and not np.isfinite(item):
            score.append(None)
        else:
            score.append(item)
    payload["selection_score"] = score
    if not _bool_value(payload.get("success")) or _bool_value(payload.get("crashed")):
        payload["promotion_note"] = (
            "No passing target-aspect constrained-QI row was available; this "
            "record is a status artifact and must not be promoted."
        )
    BEST_JSON.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Write CSV/JSON summaries and skip the Matplotlib objective panel.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    rows = _discover_qi_results()
    if not rows:
        raise FileNotFoundError(
            "No current target-6 constrained QI rows found under "
            f"{OUTPUT_ROOT}. Rerun generate_qs_ess_sweep.py for --problems qi "
            "with the current objective policy before rendering."
        )
    _write_summaries(rows)
    if not args.summary_only:
        _plot_objective_panel(rows)
    best = _write_best(rows)
    print(f"Wrote {SUMMARY_CSV}")
    if not args.summary_only:
        print(f"Wrote {OBJECTIVE_PNG}")
    print(
        "Best QI: "
        f"{best['backend']} {best['policy']} qp_preseed={best['qi_qp_preseed']} "
        f"mode={best['max_mode']} ess={best['use_ess']} "
        f"J={best.get('objective_final')} QI={best.get('qi_legacy_total', best.get('qi_raw_total'))} "
        f"mirror={best.get('qi_mirror_ratio_max')} elong={best.get('qi_max_elongation')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
