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


def _repo_relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_minimal_seed_showcase import DEFAULT_CASE_ORDER, SHOWCASE_CASES

PUBLICATION_CASE_ORDER = (
    "qa_nfp2",
    "qa_nfp3",
    "qh_nfp3",
    "qh_nfp4",
    "qp_nfp2",
    "qp_nfp3",
    "qp_nfp4",
    "qi_nfp1",
    "qi_nfp2",
    "qi_nfp3",
    "qi_nfp4",
)
PUBLICATION_STRESS_CASE_ORDER = ("qp_nfp1",)
PUBLICATION_POLICY = "continuation"
PUBLICATION_MAX_MODE = 5


@dataclass(frozen=True)
class ShowcaseRecord:
    """Completed or attempted minimal-seed optimization record."""

    case_name: str
    nfp: int
    problem: str
    output_dir: Path
    success: bool
    crashed: bool
    message: str | None
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
    stale_reason: str | None = None


@dataclass(frozen=True)
class ShowcaseProvenance:
    """Input/output paths needed to distinguish raw seed and stage seed states."""

    initial_kind: str
    initial_input: Path | None
    stage_seed_kind: str
    stage_seed_input: Path | None
    initial_wout: Path | None
    final_wout: Path | None


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


def _case_filter(raw_cases: str) -> tuple[str, ...] | None:
    """Parse a comma-separated case filter; ``all`` keeps the full matrix."""

    if raw_cases.strip().lower() == "all":
        return None
    return tuple(case.strip() for case in raw_cases.split(",") if case.strip())


def _unknown_case_filter(cases: tuple[str, ...] | None) -> tuple[str, ...]:
    if cases is None:
        return ()
    known = set(SHOWCASE_CASES)
    return tuple(sorted({case for case in cases if case not in known}))


def _filter_records_by_case(records: list[ShowcaseRecord], cases: tuple[str, ...] | None) -> list[ShowcaseRecord]:
    if cases is None:
        return records
    requested = set(cases)
    return [record for record in records if record.case_name in requested]


def _filter_case_order(case_order: tuple[str, ...], cases: tuple[str, ...] | None) -> tuple[str, ...]:
    if cases is None:
        return case_order
    requested = set(cases)
    return tuple(case_name for case_name in case_order if case_name in requested)


def _metadata_for_result(result_path: Path) -> dict:
    meta_path = result_path.parent / "showcase_case.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return {}


def _stale_reason_for_record(
    *,
    case_name: str,
    metadata: dict,
    case_meta: dict,
    result: dict,
    result_path: Path,
) -> str | None:
    """Return why a result does not match the current showcase contract."""

    case = SHOWCASE_CASES.get(case_name)
    if case is None:
        return f"unknown minimal-seed case {case_name!r}"
    if str(result.get("problem", "")) != case.problem:
        return f"problem changed from {result.get('problem')!r} to {case.problem!r}"
    preseed_meta = metadata.get("reference_preseed") or {}
    expected_ref = case.reference_preseed_input
    if expected_ref is not None and float(case.reference_preseed_blend) != 0.0:
        recorded_ref = preseed_meta.get("reference_input")
        if not _bool_value(preseed_meta.get("enabled")) or not recorded_ref:
            return "record predates reference-family preseed provenance"
        if Path(str(recorded_ref)).name != expected_ref.name:
            return (
                "reference-family preseed input changed from "
                f"{recorded_ref!r} to {str(expected_ref)!r}"
            )
        recorded_blend = _float_or_none(preseed_meta.get("blend"))
        if recorded_blend is None or not np.isclose(recorded_blend, float(case.reference_preseed_blend)):
            return (
                "reference-family preseed blend changed from "
                f"{recorded_blend!r} to {case.reference_preseed_blend!r}"
            )
    if case.problem != "qi":
        return None

    expected_policy_case = case.qi_policy_case
    if not expected_policy_case:
        return None
    path_parts = set(result_path.parent.parts)
    recorded_policy_case = case_meta.get("qi_policy_case")
    if recorded_policy_case != expected_policy_case and expected_policy_case not in path_parts:
        return (
            "QI record predates staged dispatch; expected "
            f"{expected_policy_case!r} policy-case provenance"
        )
    if _bool_value(result.get("qi_qp_preseed")):
        return "QI record used the old QP-preseed sweep path instead of qi_staged_runner"
    return None


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
        stale_reason = _stale_reason_for_record(
            case_name=case_name,
            metadata=metadata,
            case_meta=case_meta,
            result=result,
            result_path=result_path,
        )
        records.append(
            ShowcaseRecord(
                case_name=case_name,
                nfp=nfp,
                problem=str(result.get("problem", "")),
                output_dir=result_path.parent,
                success=_bool_value(result.get("success")),
                crashed=_bool_value(result.get("crashed")),
                message=None if result.get("message") is None else str(result.get("message")),
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
                stale_reason=stale_reason,
            )
        )
    return records


def best_records(
    records: list[ShowcaseRecord],
    *,
    successful_only: bool = True,
    include_stale: bool = False,
) -> list[ShowcaseRecord]:
    """Return one successful lowest-objective record per minimal-seed case."""

    selected: list[ShowcaseRecord] = []
    for case_name in DEFAULT_CASE_ORDER:
        candidates = [
            record
            for record in records
            if record.case_name == case_name
            and (include_stale or record.stale_reason is None)
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


def publication_records(
    records: list[ShowcaseRecord],
    *,
    successful_only: bool = True,
    include_stale: bool = False,
    include_stress: bool = False,
) -> list[ShowcaseRecord]:
    """Return the current aspect-5/mode-5 README promotion matrix records."""

    case_order = PUBLICATION_CASE_ORDER + (PUBLICATION_STRESS_CASE_ORDER if include_stress else ())
    selected: list[ShowcaseRecord] = []
    for case_name in case_order:
        candidates = [
            record
            for record in records
            if record.case_name == case_name
            and record.policy == PUBLICATION_POLICY
            and int(record.max_mode) == PUBLICATION_MAX_MODE
            and bool(record.use_ess)
            and (include_stale or record.stale_reason is None)
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


def record_status(record: ShowcaseRecord) -> str:
    """Compact status label for README/docs tables."""

    if record.stale_reason is not None:
        return "stale"
    if record.success and not record.crashed:
        return "ok"
    if "partial" in str(record.message or "").lower():
        return "partial"
    if not record.crashed:
        return "incomplete"
    return "failed"


def display_message(record: ShowcaseRecord) -> str:
    """Return a stable human-facing status message for static docs tables."""

    message = "" if record.message is None else str(record.message)
    if record.crashed and "case still running" in message.lower():
        return "partial checkpoint metrics recorded; not promoted in refreshed static summary"
    return message


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

    stage_history_paths = sorted(record.output_dir.glob("*/history.json"))
    history_paths = stage_history_paths or [record.output_dir / "history.json"]
    if not any(path.exists() for path in history_paths):
        return []
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    seen_history_payloads: set[str] = set()
    offset_min = 0.0
    for history_path in history_paths:
        if not history_path.exists():
            continue
        try:
            data = json.loads(history_path.read_text())
        except json.JSONDecodeError:
            continue
        history = list(data.get("history", []))
        history_payload = json.dumps(history, sort_keys=True, separators=(",", ":"))
        if history_payload in seen_history_payloads:
            continue
        seen_history_payloads.add(history_payload)
        for segment in _history_stage_segments(history):
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
            if np.any(np.diff(wall_min) < 0.0):
                wall_min = wall_min - float(np.nanmin(wall_min))
            wall_min = wall_min + offset_min
            segments.append((wall_min, np.minimum.accumulate(values)))
            offset_min = float(wall_min[-1])
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
                "status",
                "success",
                "crashed",
                "objective_final",
                "aspect_final",
                "iota_final",
                "wall_time_min",
                "qi_legacy_total",
                "qi_mirror_ratio_max",
                "qi_max_elongation",
                "message",
                "initial_kind",
                "initial_input",
                "initial_wout",
                "stage_seed_kind",
                "stage_seed_input",
                "final_wout",
                "output_dir",
            ]
        )
        for record in records:
            provenance = provenance_for_record(record)
            writer.writerow(
                [
                    record.case_name,
                    record.problem,
                    record.nfp,
                    record.policy,
                    record.max_mode,
                    "yes" if record.use_ess else "no",
                    record_status(record),
                    "yes" if record.success and not record.crashed else "no",
                    "yes" if record.crashed else "no",
                    "" if record.objective_final is None else f"{record.objective_final:.16e}",
                    "" if record.aspect_final is None else f"{record.aspect_final:.16e}",
                    "" if record.iota_final is None else f"{record.iota_final:.16e}",
                    "" if record.total_wall_time_s is None else f"{record.total_wall_time_s / 60.0:.6f}",
                    "" if record.qi_legacy_total is None else f"{record.qi_legacy_total:.16e}",
                    "" if record.qi_mirror_ratio_max is None else f"{record.qi_mirror_ratio_max:.16e}",
                    "" if record.qi_max_elongation is None else f"{record.qi_max_elongation:.16e}",
                    display_message(record),
                    provenance.initial_kind,
                    "" if provenance.initial_input is None else _repo_relative_path(provenance.initial_input),
                    "" if provenance.initial_wout is None else _repo_relative_path(provenance.initial_wout),
                    provenance.stage_seed_kind,
                    "" if provenance.stage_seed_input is None else _repo_relative_path(provenance.stage_seed_input),
                    "" if provenance.final_wout is None else _repo_relative_path(provenance.final_wout),
                    _repo_relative_path(record.output_dir),
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
        status = record_status(record)
        title = f"{record.case_name}: {record.policy}, m={record.max_mode}, {'ESS' if record.use_ess else 'no ESS'}, {status}"
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Wall time (min)")
        ax.set_ylabel("Best objective")
        ax.grid(True, alpha=0.25, linestyle=":")
    fig.suptitle("Common minimal-seed optimization histories", fontsize=13)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    _save_compact_png(fig, out_png, dpi=200)
    plt.close(fig)
    return out_png


def _save_compact_png(fig, path: Path, *, dpi: int) -> None:
    """Save a tracked PNG with deterministic lossless compression."""

    try:
        fig.savefig(
            path,
            dpi=dpi,
            bbox_inches="tight",
            pil_kwargs={"optimize": True, "compress_level": 9},
        )
    except TypeError:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")


def _local_repo_path(value: str | Path | None, *, output_dir: Path | None = None) -> Path | None:
    """Map local or remote provenance paths back onto this checkout when possible."""

    if value in (None, ""):
        return None
    path = Path(str(value))
    if path.exists():
        return path
    if output_dir is not None:
        candidate = output_dir / path.name
        if candidate.exists():
            return candidate
    text = str(value).replace("\\", "/")
    for marker in ("examples/data/", "examples/optimization/results/"):
        if marker in text:
            candidate = ROOT / marker / text.split(marker, 1)[1]
            if candidate.exists():
                return candidate
    return path


def _record_metadata(record: ShowcaseRecord) -> dict:
    return _metadata_for_result(record.output_dir / "case_result.json")


def _initial_input_for_record(record: ShowcaseRecord) -> Path | None:
    """Return the raw VMEC seed deck for a minimal-seed showcase record.

    This deliberately ignores optimization-time target-helicity and
    reference-family preseed files.  README/docs state panels should show the
    user-facing seed, not a later stage input that already contains hints.
    """

    metadata = _record_metadata(record)
    case_meta = metadata.get("minimal_seed_case") or {}
    return _local_repo_path(case_meta.get("input_file"), output_dir=record.output_dir)


def _stage_seed_for_record(record: ShowcaseRecord) -> tuple[str, Path | None]:
    """Return the first optimization-time input and its provenance kind."""

    raw_input = _initial_input_for_record(record)
    metadata = _record_metadata(record)
    preseed_meta = metadata.get("reference_preseed") or {}
    seed_meta = metadata.get("target_helicity_seed") or {}
    seeded = _local_repo_path(seed_meta.get("seeded_input_file"), output_dir=record.output_dir)
    preseeded = _local_repo_path(preseed_meta.get("preseeded_input_file"), output_dir=record.output_dir)
    has_seed = seeded is not None and seeded.exists()
    preseed_enabled = _bool_value(preseed_meta.get("enabled"))
    preseed_is_distinct = (
        preseeded is not None
        and preseed_enabled
        and (raw_input is None or preseeded.resolve() != raw_input.resolve())
    )
    if has_seed and preseed_is_distinct:
        return "reference_preseed+target_helicity_seed", seeded
    if has_seed:
        return "target_helicity_seed", seeded
    if preseed_is_distinct:
        return "reference_preseed", preseeded
    return "raw_seed", raw_input


def _stage_checkpoint_wout(record: ShowcaseRecord, name: str) -> Path | None:
    checkpoint = record.output_dir / "stage_checkpoint.json"
    if not checkpoint.exists():
        return None
    try:
        data = json.loads(checkpoint.read_text())
    except json.JSONDecodeError:
        return None
    if name == "wout_final.nc":
        for raw_path in (
            data.get("wout_path"),
            data.get("diagnostics", {}).get("boundary_reference_wout_path")
            if isinstance(data.get("diagnostics"), dict)
            else None,
        ):
            candidate = _local_repo_path(raw_path, output_dir=record.output_dir)
            if candidate is not None and candidate.exists():
                return candidate
    diagnostics_path = _local_repo_path(data.get("diagnostics_path"), output_dir=record.output_dir)
    if diagnostics_path is None:
        return None
    candidate = diagnostics_path.parent / name
    return candidate if candidate.exists() else None


def _final_wout_for_record(record: ShowcaseRecord) -> Path | None:
    for candidate in (
        record.output_dir / "wout_final.nc",
        _stage_checkpoint_wout(record, "wout_final.nc"),
    ):
        if candidate is not None and candidate.exists():
            return candidate
    nested = sorted(record.output_dir.glob("*/wout_final.nc"))
    return nested[-1] if nested else None


def _initial_wout_for_record(record: ShowcaseRecord) -> Path | None:
    input_file = _initial_input_for_record(record)
    if input_file is None or not input_file.exists():
        return None
    try:
        from render_readme_best_optimizations import _validated_or_derived_raw_initial_wout
    except ImportError:
        return None
    candidate = record.output_dir / "wout_original.nc"
    if not candidate.exists() and record.policy == "continuation":
        mode1 = record.output_dir.parent.parent / "mode1" / record.output_dir.name / "wout_initial.nc"
        if mode1.exists():
            candidate = mode1
    if not candidate.exists():
        candidate = record.output_dir / "wout_initial.nc"
    try:
        return _validated_or_derived_raw_initial_wout(
            candidate,
            input_file,
            record.output_dir,
            context=f"{record.case_name} minimal-seed initial wout",
        )
    except Exception:
        return None


def _existing_initial_wout_for_record(record: ShowcaseRecord) -> Path | None:
    """Return a proven raw-seed initial WOUT path only if it already exists."""

    candidate = record.output_dir / "wout_original.nc"
    return candidate if candidate.exists() else None


def provenance_for_record(record: ShowcaseRecord) -> ShowcaseProvenance:
    """Return README/docs provenance paths without running VMEC."""

    stage_seed_kind, stage_seed_input = _stage_seed_for_record(record)
    return ShowcaseProvenance(
        initial_kind="raw_seed",
        initial_input=_initial_input_for_record(record),
        stage_seed_kind=stage_seed_kind,
        stage_seed_input=stage_seed_input,
        initial_wout=_existing_initial_wout_for_record(record),
        final_wout=_final_wout_for_record(record),
    )


def render_state_panel(records: list[ShowcaseRecord], out_png: Path) -> Path | None:
    """Render initial/final LCFS, Boozer contours, and objective histories."""

    candidates = [
        (record, _initial_wout_for_record(record), _final_wout_for_record(record))
        for record in records
        if record.stale_reason is None
    ]
    candidates = [
        (record, initial_wout, final_wout)
        for record, initial_wout, final_wout in candidates
        if initial_wout is not None and final_wout is not None and final_wout.exists()
    ]
    if not candidates:
        return None

    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        from render_readme_best_optimizations import _plot_boozer_bmag, _plot_lcfs
        from vmec_jax.wout import read_wout
    except Exception:
        return None

    nrows = len(candidates)
    fig = plt.figure(figsize=(22.0, 4.4 * nrows), constrained_layout=True)
    gs = fig.add_gridspec(nrows, 5, width_ratios=(1.05, 1.05, 1.0, 1.05, 1.05))
    for row, (record, initial_wout, final_wout) in enumerate(candidates):
        ax0 = fig.add_subplot(gs[row, 0], projection="3d")
        ax1 = fig.add_subplot(gs[row, 1], projection="3d")
        ax2 = fig.add_subplot(gs[row, 2])
        ax3 = fig.add_subplot(gs[row, 3])
        ax4 = fig.add_subplot(gs[row, 4])

        stage_seed_kind, _stage_seed_input = _stage_seed_for_record(record)
        seed_label = "raw minimal seed"
        if stage_seed_kind != "raw_seed":
            seed_label = f"raw minimal seed\n(stage seed: {stage_seed_kind})"
        _plot_lcfs(ax0, read_wout(initial_wout), f"{record.case_name}: {seed_label}")
        _plot_lcfs(ax1, read_wout(final_wout), "final")
        segments = objective_segments(record)
        if not segments:
            ax2.text(0.5, 0.5, "history missing", ha="center", va="center", transform=ax2.transAxes)
        for wall_min, values in segments:
            ax2.semilogy(wall_min, values, color="#1f4e79", linewidth=1.8)
            ax2.scatter(wall_min[-1], values[-1], s=16, color="#d95f02", zorder=3)
        ax2.set_title("Best objective history", fontsize=9)
        ax2.set_xlabel("Wall time (min)")
        ax2.set_ylabel("Best objective")
        ax2.grid(True, alpha=0.25, linestyle=":")
        _plot_boozer_bmag(ax3, initial_wout, r"Initial $|B|$")
        _plot_boozer_bmag(ax4, final_wout, r"Final $|B|$")

    fig.suptitle("Common minimal-seed initial/final optimization states", fontsize=13, x=0.01, y=1.01, ha="left")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    _save_compact_png(fig, out_png, dpi=170)
    plt.close(fig)
    return out_png


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument("--cases", type=str, default="all", help="Comma-separated cases to render, or 'all'.")
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Do not print missing-case warnings; useful for bounded smoke renders.",
    )
    parser.add_argument("--summary-only", action="store_true", help="Write CSV only; skip Matplotlib rendering.")
    parser.add_argument("--skip-state-panel", action="store_true", help="Skip initial/final geometry and Boozer panel.")
    parser.add_argument(
        "--include-stale",
        action="store_true",
        help="Include pre-dispatch or otherwise stale records instead of skipping them.",
    )
    parser.add_argument(
        "--publication-matrix",
        action="store_true",
        help="Render only successful current aspect-5/mode-5 README promotion rows.",
    )
    parser.add_argument(
        "--include-stress",
        action="store_true",
        help="With --publication-matrix, include optional stress rows such as qp_nfp1.",
    )
    args = parser.parse_args()
    unknown_cases = _unknown_case_filter(_case_filter(str(args.cases)))
    if unknown_cases:
        parser.error(
            "unknown --cases value(s): "
            + ", ".join(unknown_cases)
            + ". Available cases: "
            + ", ".join(DEFAULT_CASE_ORDER)
        )
    return args


def main() -> None:
    args = _parse_args()
    case_filter = _case_filter(str(args.cases))
    loaded_records = _filter_records_by_case(load_records(args.output_root), case_filter)
    stale_records = [record for record in loaded_records if record.stale_reason is not None]
    if not bool(args.include_stale):
        for record in stale_records:
            print(f"Skipping stale {record.case_name} record at {record.output_dir}: {record.stale_reason}")
    if bool(args.publication_matrix):
        all_records = publication_records(
            loaded_records,
            successful_only=True,
            include_stale=bool(args.include_stale),
            include_stress=bool(args.include_stress),
        )
        expected_case_order = PUBLICATION_CASE_ORDER + (
            PUBLICATION_STRESS_CASE_ORDER if bool(args.include_stress) else ()
        )
    else:
        all_records = best_records(
            loaded_records,
            successful_only=False,
            include_stale=bool(args.include_stale),
        )
        expected_case_order = DEFAULT_CASE_ORDER
    expected_case_order = _filter_case_order(expected_case_order, case_filter)
    if not all_records:
        if expected_case_order and not bool(args.skip_missing):
            print("Missing current minimal-seed records: " + ", ".join(expected_case_order))
        elif expected_case_order:
            print(
                "No selected minimal-seed showcase records found under "
                f"{args.output_root} for cases: {', '.join(expected_case_order)}"
            )
        else:
            print(f"No minimal-seed showcase records found under {args.output_root}")
        return
    present_cases = {record.case_name for record in all_records}
    missing = [case_name for case_name in expected_case_order if case_name not in present_cases]
    if missing and not bool(args.skip_missing):
        print("Missing current minimal-seed records: " + ", ".join(missing))
    summary_csv = Path(args.figure_dir) / "minimal_seed_showcase_summary.csv"
    write_summary_csv(all_records, summary_csv)
    print(f"Wrote {summary_csv}")
    if not bool(args.summary_only):
        out_png = Path(args.figure_dir) / "minimal_seed_showcase_objective_panel.png"
        rendered = render_objective_panel(all_records, out_png)
        if rendered is not None:
            print(f"Wrote {rendered}")
        if not bool(args.skip_state_panel):
            state_png = Path(args.figure_dir) / "minimal_seed_showcase_state_panel.png"
            rendered_state = render_state_panel(all_records, state_png)
            if rendered_state is not None:
                print(f"Wrote {rendered_state}")
            else:
                print("No current minimal-seed state panel rendered; missing non-stale initial/final wout artifacts.")


if __name__ == "__main__":
    main()
