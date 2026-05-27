#!/usr/bin/env python
"""Render README/docs coverage for the reviewed QI_optimization inputs.

This renderer intentionally consumes existing reviewed outputs instead of
launching new optimization jobs.  The initial/final Boozer |B| panels use line
contours only.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vmec_jax.namelist import read_indata
from vmec_jax.plotting import fix_matplotlib_3d, prepare_matplotlib_3d, vmecplot2_lcfs_3d_grid
from vmec_jax.wout import read_wout


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
FIGURE_DIR = REPO_ROOT / "docs" / "_static" / "figures"
ARTIFACT_DIR = REPO_ROOT / "docs" / "_static" / "qi_readme_cases"
OUT_PNG = FIGURE_DIR / "readme_qi_optimization_cases.png"
OUT_CSV = FIGURE_DIR / "readme_qi_optimization_cases.csv"


@dataclass(frozen=True)
class QICase:
    label: str
    input_file: Path
    output_dir: Path
    initial_wout: Path
    note: str
    validation_status: str = "case-gated"
    history_paths: tuple[Path, ...] = ()
    preconditioner_summary: Path | None = None


CASES = (
    QICase(
        label="NFP=1 QI",
        input_file=REPO_ROOT / "examples" / "data" / "input.nfp1_QI",
        output_dir=ARTIFACT_DIR / "nfp1",
        initial_wout=ARTIFACT_DIR / "nfp1" / "wout_initial.nc",
        note="mirror-aware QI lane",
        history_paths=(
            ARTIFACT_DIR / "nfp1" / "mirror_ramp_01_matrix_free_mirror030" / "history.json",
        ),
    ),
    QICase(
        label="NFP=2 target-helicity seed",
        input_file=REPO_ROOT / "examples" / "data" / "input.minimal_seed_nfp2_target_helicity",
        output_dir=ARTIFACT_DIR / "nfp2_target_helicity",
        initial_wout=ARTIFACT_DIR / "nfp2_target_helicity" / "wout_initial.nc",
        note="aspect-5 target-helicity hinted mirror-aware QI lane",
        history_paths=(
            ARTIFACT_DIR
            / "nfp2_target_helicity"
            / "mirror_ramp_01_matrix_free_mirror030"
            / "history.json",
        ),
        preconditioner_summary=ARTIFACT_DIR
        / "nfp2_target_helicity"
        / "boundary_reference_preconditioner"
        / "summary.json",
    ),
    QICase(
        label="NFP=3 seed 3127",
        input_file=REPO_ROOT / "examples" / "data" / "input.QI_stel_seed_3127",
        output_dir=ARTIFACT_DIR / "nfp3_seed3127",
        initial_wout=ARTIFACT_DIR / "nfp3_seed3127" / "wout_initial.nc",
        note="passing reference-family QI lane",
        history_paths=(
            ARTIFACT_DIR / "nfp3_seed3127" / "history.json",
            ARTIFACT_DIR / "nfp3_seed3127" / "boundary_reference_baseline" / "history.json",
            ARTIFACT_DIR
            / "nfp3_seed3127"
            / "mirror_ramp_01_prefiltered_mirror_qi_iota_cleanup"
            / "history.json",
        ),
        preconditioner_summary=ARTIFACT_DIR
        / "nfp3_seed3127"
        / "boundary_reference_preconditioner"
        / "summary.json",
    ),
    QICase(
        label="NFP=4 minimal + QI-reference proposal",
        input_file=REPO_ROOT / "examples" / "data" / "input.minimal_seed_nfp4",
        output_dir=ARTIFACT_DIR / "nfp4_minimal",
        initial_wout=ARTIFACT_DIR / "nfp4_minimal" / "wout_initial.nc",
        note="minimal seed with same-NFP finite-beta QI reference-family preconditioner",
        history_paths=(
            ARTIFACT_DIR
            / "nfp4_minimal"
            / "mirror_ramp_01_finite_beta_qi_audit_refine"
            / "history.json",
        ),
        preconditioner_summary=ARTIFACT_DIR / "nfp4_minimal" / "boundary_reference_preconditioner" / "summary.json",
    ),
)

BOUNDARY_FAMILIES = ("RBC", "RBS", "ZBC", "ZBS")


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _boundary_maps_from_input(input_file: Path) -> dict[str, dict[tuple[int, int], float]]:
    indata = read_indata(input_file)
    return {
        family: {tuple(key): float(value) for key, value in indata.indexed.get(family, {}).items()}
        for family in BOUNDARY_FAMILIES
    }


def _boundary_maps_from_wout(wout_path: Path) -> dict[str, dict[tuple[int, int], float]]:
    wout = read_wout(wout_path)
    arrays = {
        "RBC": np.asarray(wout.rmnc[-1], dtype=float),
        "RBS": np.asarray(wout.rmns[-1], dtype=float),
        "ZBC": np.asarray(wout.zmnc[-1], dtype=float),
        "ZBS": np.asarray(wout.zmns[-1], dtype=float),
    }
    nfp = int(wout.nfp)
    maps: dict[str, dict[tuple[int, int], float]] = {family: {} for family in BOUNDARY_FAMILIES}
    for family, values in arrays.items():
        for m_i, xn_i, value in zip(np.asarray(wout.xm, dtype=int), np.asarray(wout.xn, dtype=int), values):
            n_i = int(round(float(xn_i) / float(nfp))) if nfp else int(xn_i)
            maps[family][(n_i, int(m_i))] = float(value)
    return maps


def _boundary_maps_from_wout_vmec_input_convention(wout_path: Path) -> dict[str, dict[tuple[int, int], float]]:
    """Return WOUT boundary modes in the VMEC input phase convention.

    Some externally generated VMEC input decks use a poloidal phase convention
    equivalent to VMEC's canonical output after a theta shift. The physical LCFS
    is the same, but coefficients appear under ``n -> -n`` and parity-dependent
    signs for ``m > 0``. Accepting this map prevents the README renderer from
    rejecting a true raw WOUT just because VMEC wrote the equivalent canonical
    representation.
    """
    wout = read_wout(wout_path)
    arrays = {
        "RBC": np.asarray(wout.rmnc[-1], dtype=float),
        "RBS": np.asarray(wout.rmns[-1], dtype=float),
        "ZBC": np.asarray(wout.zmnc[-1], dtype=float),
        "ZBS": np.asarray(wout.zmns[-1], dtype=float),
    }
    nfp = int(wout.nfp)
    maps: dict[str, dict[tuple[int, int], float]] = {family: {} for family in BOUNDARY_FAMILIES}
    for family, values in arrays.items():
        for m_i_raw, xn_i, value in zip(np.asarray(wout.xm, dtype=int), np.asarray(wout.xn, dtype=int), values):
            m_i = int(m_i_raw)
            n_wout = int(round(float(xn_i) / float(nfp))) if nfp else int(xn_i)
            if m_i == 0:
                n_i = n_wout
                sign = 1.0
            else:
                n_i = -n_wout
                sign = float((-1) ** m_i) if family in ("RBC", "RBS") else float((-1) ** (m_i + 1))
            maps[family][(n_i, m_i)] = sign * float(value)
    return maps


def _boundary_mismatches_for_actual(
    expected: dict[str, dict[tuple[int, int], float]],
    actual: dict[str, dict[tuple[int, int], float]],
    *,
    abs_tol: float,
    rel_tol: float,
) -> list[str]:
    mismatches: list[str] = []
    for family in BOUNDARY_FAMILIES:
        keys = set(expected[family]) | {key for key, value in actual[family].items() if abs(value) > abs_tol}
        for key in sorted(keys):
            lhs = expected[family].get(key, 0.0)
            rhs = actual[family].get(key, 0.0)
            tol = abs_tol + rel_tol * max(abs(lhs), abs(rhs))
            if abs(lhs - rhs) > tol:
                mismatches.append(f"{family}{key}: input={lhs:.16e}, wout={rhs:.16e}")
                if len(mismatches) >= 8:
                    return mismatches
    return mismatches


def _boundary_mismatches(
    input_file: Path,
    wout_path: Path,
    *,
    abs_tol: float = 5.0e-8,
    rel_tol: float = 5.0e-8,
) -> list[str]:
    expected = _boundary_maps_from_input(input_file)
    candidates = (
        _boundary_maps_from_wout(wout_path),
        _boundary_maps_from_wout_vmec_input_convention(wout_path),
    )
    mismatch_sets = tuple(
        _boundary_mismatches_for_actual(
            expected,
            actual,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
        )
        for actual in candidates
    )
    return min(mismatch_sets, key=len)


def _assert_wout_matches_input_boundary(wout_path: Path, input_file: Path, *, context: str) -> None:
    mismatches = _boundary_mismatches(input_file, wout_path)
    if mismatches:
        joined = "; ".join(mismatches)
        raise RuntimeError(
            f"{context} is not the raw input boundary for {input_file}: {joined}. "
            "Regenerate or provide a raw initial wout that matches the paired input deck."
        )


def _validate_case_initial_wout(case: QICase) -> None:
    _assert_wout_matches_input_boundary(
        case.initial_wout,
        case.input_file,
        context=f"{case.label} initial_wout",
    )


def _history_paths(case: QICase) -> tuple[Path, ...]:
    if case.history_paths:
        return case.history_paths
    return (case.output_dir / "history.json",)


def _history_value(item: dict) -> float:
    value = item.get("objective", item.get("cost"))
    if value is None:
        return np.nan
    return float(value)


def _short_history_label(label: str) -> str:
    label = label.removeprefix("QI ")
    replacements = {
        "optimization (max_mode=3, ESS)": "seed solve",
        "qi_basin (max_mode=3, ESS)": "QI basin",
        "matrix_free_mirror030 (max_mode=3, ESS)": "mirror<=0.30",
        "lcfs_mirror_030 (max_mode=3, ESS)": "LCFS mirror check",
        "boundary-reference baseline (max_mode=4)": "boundary baseline",
        "prefiltered_mirror_qi_iota_cleanup (max_mode=4, ESS)": "rejected cleanup",
    }
    return replacements.get(label, label[:34])


def _history_segments(case: QICase) -> list[dict[str, np.ndarray | str | Path]]:
    segments: list[dict[str, np.ndarray | str | Path]] = []
    offset_s = 0.0
    seen_history_payloads: set[str] = set()
    for path in _history_paths(case):
        history = _load_json(path)
        entries = history.get("history", [])
        if not entries:
            raise RuntimeError(f"Missing history entries in {path}")
        # Some curated QI outputs keep a root history.json as a provenance
        # alias for a named stage directory.  Skip byte-equivalent histories so
        # README/docs plots do not double-count the same accepted points.
        history_payload = json.dumps(entries, sort_keys=True, separators=(",", ":"))
        if history_payload in seen_history_payloads:
            continue
        seen_history_payloads.add(history_payload)
        raw_time = np.asarray([float(item.get("wall_time_s", 0.0)) for item in entries], dtype=float)
        values = np.asarray([max(_history_value(item), 1.0e-16) for item in entries], dtype=float)
        keep = np.isfinite(raw_time) & np.isfinite(values)
        raw_time = raw_time[keep]
        values = values[keep]
        if raw_time.size == 0:
            raise RuntimeError(f"No finite history entries in {path}")
        if np.any(np.diff(raw_time) < 0.0):
            raw_time = raw_time - float(np.nanmin(raw_time))
        shifted_time = raw_time + offset_s
        label = _short_history_label(str(history.get("label") or path.parent.name))
        segments.append({"path": path, "label": label, "wall_time_s": shifted_time, "objective": values})
        offset_s = float(shifted_time[-1])
    return segments


def _history_summary(case: QICase) -> tuple[float, int, int]:
    segments = _history_segments(case)
    total_wall_s = 0.0
    total_points = 0
    for segment in segments:
        wall_time_s = np.asarray(segment["wall_time_s"], dtype=float)
        total_wall_s = max(total_wall_s, float(wall_time_s[-1]))
        total_points += int(wall_time_s.size)
    return total_wall_s, len(segments), total_points


def _stage_normalized_best_so_far(objective: np.ndarray) -> np.ndarray:
    """Normalize a stage objective and return its monotone best-so-far trace."""

    objective = np.asarray(objective, dtype=float)
    if objective.size == 0:
        return objective
    scale = max(float(objective[0]), 1.0e-16)
    normalized = np.maximum(objective / scale, 1.0e-16)
    return np.minimum.accumulate(normalized)


def _history_is_effectively_flat(segments: list[dict[str, np.ndarray | str | Path]]) -> bool:
    """Return True when normalized objective history has no visible movement."""

    for segment in segments:
        objective = np.asarray(segment["objective"], dtype=float)
        if objective.size < 2:
            continue
        best_so_far = _stage_normalized_best_so_far(objective)
        if np.nanmax(best_so_far) - np.nanmin(best_so_far) > 1.0e-4:
            return False
    return True


def _preconditioner_summary(case: QICase) -> tuple[int, float | None, float | None, float | None]:
    if case.preconditioner_summary is None:
        return 0, None, None, None
    rows = _load_json(case.preconditioner_summary)
    if not isinstance(rows, list):
        raise RuntimeError(f"Expected a list in {case.preconditioner_summary}")
    selected = [row for row in rows if bool(row.get("selected"))]
    selected_row = selected[-1] if selected else (rows[-1] if rows else {})
    return (
        len(rows),
        None if "lambda" not in selected_row else float(selected_row["lambda"]),
        None if "legacy_qi" not in selected_row else float(selected_row["legacy_qi"]),
        None if "mirror" not in selected_row else float(selected_row["mirror"]),
    )


def _selected_preconditioner_row(case: QICase) -> dict | None:
    if case.preconditioner_summary is None:
        return None
    rows = _load_json(case.preconditioner_summary)
    if not isinstance(rows, list) or not rows:
        return None
    selected = [row for row in rows if bool(row.get("selected"))]
    return dict(selected[-1] if selected else rows[-1])


def _optional_bool(diagnostics: dict, key: str) -> str | bool:
    if key not in diagnostics:
        return ""
    return bool(diagnostics[key])


def _qi_smooth_total(diagnostics: dict) -> float:
    if "qi_smooth_total" in diagnostics:
        return float(diagnostics["qi_smooth_total"])
    return float(diagnostics["qi_raw_total"])


def _require_finite_metric(case: QICase, name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise RuntimeError(f"{case.label} has non-finite {name}: {value!r}")
    return value


def _require_case_gated_diagnostics(
    case: QICase,
    diagnostics: dict,
    *,
    objective_final: float,
    qi_smooth_total: float,
    qi_legacy_total: float,
    qi_mirror_ratio_max: float,
    qi_mirror_ratio_target: float,
    qi_max_elongation: float,
    qi_elongation_target: float,
    aspect: float,
    target_aspect: float,
    mean_iota: float,
) -> None:
    if diagnostics.get("qi_seed_gate_passed") is not True:
        raise RuntimeError(f"{case.label} is case-gated but failed the QI seed gate")
    if diagnostics.get("qi_engineering_gate_passed") is not True:
        raise RuntimeError(f"{case.label} is case-gated but failed the QI engineering gate")
    gate_failures = diagnostics.get("qi_gate_failures", [])
    if gate_failures:
        raise RuntimeError(f"{case.label} is case-gated but has QI gate failures: {gate_failures}")

    metrics = {
        "objective_final": objective_final,
        "qi_smooth_total": qi_smooth_total,
        "qi_legacy_total": qi_legacy_total,
        "qi_mirror_ratio_max": qi_mirror_ratio_max,
        "qi_mirror_ratio_target": qi_mirror_ratio_target,
        "qi_max_elongation": qi_max_elongation,
        "qi_elongation_target": qi_elongation_target,
        "aspect": aspect,
        "target_aspect": target_aspect,
        "mean_iota": mean_iota,
    }
    for name, value in metrics.items():
        _require_finite_metric(case, name, value)

    if "qi_smooth_gate" in diagnostics and qi_smooth_total > float(diagnostics["qi_smooth_gate"]):
        raise RuntimeError(f"{case.label} is case-gated but smooth QI exceeds its gate")
    if "qi_legacy_gate" in diagnostics and qi_legacy_total > float(diagnostics["qi_legacy_gate"]):
        raise RuntimeError(f"{case.label} is case-gated but legacy QI exceeds its gate")
    if qi_mirror_ratio_max > qi_mirror_ratio_target:
        raise RuntimeError(f"{case.label} is case-gated but mirror ratio exceeds its target")
    if qi_max_elongation > qi_elongation_target:
        raise RuntimeError(f"{case.label} is case-gated but elongation exceeds its target")
    if "abs_iota_min" in diagnostics and abs(mean_iota) < float(diagnostics["abs_iota_min"]):
        raise RuntimeError(f"{case.label} is case-gated but mean iota is below its floor")


def _case_record(case: QICase) -> dict[str, str | float]:
    diagnostics = _load_json(case.output_dir / "diagnostics.json")
    history = _load_json(case.output_dir / "history.json")
    full_wall_s, history_segment_count, history_point_count = _history_summary(case)
    preconditioner_points, selected_lambda, selected_lambda_qi, selected_lambda_mirror = _preconditioner_summary(case)
    final_wout = case.output_dir / "wout_final.nc"
    for path in (case.input_file, case.initial_wout, final_wout):
        if not path.exists():
            raise FileNotFoundError(path)
    _validate_case_initial_wout(case)
    stress_fixture = bool(diagnostics.get("qi_case_stress_fixture", False))
    expected_gate_status = str(diagnostics.get("qi_case_expected_gate_status", "candidate"))
    if case.validation_status not in {"case-gated", "deferred"}:
        raise RuntimeError(f"{case.label} has unsupported validation_status={case.validation_status!r}")
    validation_status = "deferred" if case.validation_status == "deferred" or stress_fixture else "case-gated"
    objective_final = float(history["objective_final"])
    qi_smooth_total = _qi_smooth_total(diagnostics)
    qi_legacy_total = float(diagnostics["qi_legacy_total"])
    qi_mirror_ratio_max = float(diagnostics["qi_mirror_ratio_max"])
    qi_mirror_ratio_target = float(diagnostics["qi_mirror_ratio_target"])
    qi_max_elongation = float(diagnostics["qi_max_elongation"])
    qi_elongation_target = float(diagnostics["qi_elongation_target"])
    aspect = float(diagnostics["aspect"])
    target_aspect = float(diagnostics["target_aspect"])
    mean_iota = float(diagnostics["mean_iota"])
    if validation_status == "case-gated":
        _require_case_gated_diagnostics(
            case,
            diagnostics,
            objective_final=objective_final,
            qi_smooth_total=qi_smooth_total,
            qi_legacy_total=qi_legacy_total,
            qi_mirror_ratio_max=qi_mirror_ratio_max,
            qi_mirror_ratio_target=qi_mirror_ratio_target,
            qi_max_elongation=qi_max_elongation,
            qi_elongation_target=qi_elongation_target,
            aspect=aspect,
            target_aspect=target_aspect,
            mean_iota=mean_iota,
        )
    return {
        "case": case.label,
        "input_file": str(case.input_file.relative_to(REPO_ROOT)),
        "output_dir": str(case.output_dir.relative_to(REPO_ROOT)),
        "initial_wout": str(case.initial_wout.relative_to(REPO_ROOT)),
        "note": case.note,
        "validation_status": validation_status,
        "expected_gate_status": expected_gate_status,
        "qi_seed_gate_passed": _optional_bool(diagnostics, "qi_seed_gate_passed"),
        "qi_engineering_gate_passed": _optional_bool(diagnostics, "qi_engineering_gate_passed"),
        "qi_gate_failures": ";".join(str(item) for item in diagnostics.get("qi_gate_failures", [])),
        "objective_final": objective_final,
        "qi_smooth_total": qi_smooth_total,
        "qi_legacy_total": qi_legacy_total,
        "qi_mirror_ratio_max": qi_mirror_ratio_max,
        "qi_mirror_ratio_target": qi_mirror_ratio_target,
        "qi_max_elongation": qi_max_elongation,
        "qi_elongation_target": qi_elongation_target,
        "aspect": aspect,
        "target_aspect": target_aspect,
        "mean_iota": mean_iota,
        "qi_nfp": int(diagnostics["qi_nfp"]),
        "cpu_time_min": full_wall_s / 60.0,
        "published_stage_cpu_time_min": float(history["total_wall_time_s"]) / 60.0,
        "history_segments": history_segment_count,
        "history_points": history_point_count,
        "preconditioner_points": preconditioner_points,
        "selected_lambda": "" if selected_lambda is None else selected_lambda,
        "selected_lambda_qi": "" if selected_lambda_qi is None else selected_lambda_qi,
        "selected_lambda_mirror": "" if selected_lambda_mirror is None else selected_lambda_mirror,
        "final_wout": str(final_wout.relative_to(REPO_ROOT)),
    }


def _write_csv(records: list[dict[str, str | float]]) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fields = [
        "case",
        "input_file",
        "output_dir",
        "initial_wout",
        "note",
        "validation_status",
        "expected_gate_status",
        "qi_seed_gate_passed",
        "qi_engineering_gate_passed",
        "qi_gate_failures",
        "objective_final",
        "qi_smooth_total",
        "qi_legacy_total",
        "qi_mirror_ratio_max",
        "qi_mirror_ratio_target",
        "qi_max_elongation",
        "qi_elongation_target",
        "aspect",
        "target_aspect",
        "mean_iota",
        "qi_nfp",
        "cpu_time_min",
        "published_stage_cpu_time_min",
        "history_segments",
        "history_points",
        "preconditioner_points",
        "selected_lambda",
        "selected_lambda_qi",
        "selected_lambda_mirror",
        "final_wout",
    ]
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def _lcfs_xyz(R: np.ndarray, Z: np.ndarray, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return R * np.cos(phi[None, :]), R * np.sin(phi[None, :]), Z


def _plot_lcfs(ax, wout_path: Path, title: str) -> None:
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    wout = read_wout(wout_path)
    _theta, phi, R, Z, B = vmecplot2_lcfs_3d_grid(
        wout,
        s_index=int(wout.ns) - 1,
        ntheta=40,
        nzeta=max(72, 30 * int(wout.nfp)),
    )
    X, Y, Zp = _lcfs_xyz(R, Z, phi)
    norm = Normalize(vmin=float(np.nanmin(B)), vmax=float(np.nanmax(B)))
    colors = ScalarMappable(cmap="viridis", norm=norm).to_rgba(B)
    ax.plot_surface(
        X,
        Y,
        Zp,
        facecolors=colors,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=False,
        shade=False,
    )
    ax.set_title(title, fontsize=8, pad=4)
    ax.set_xlabel("X", labelpad=-8)
    ax.set_ylabel("Y", labelpad=-8)
    ax.set_zlabel("Z", labelpad=-8)
    ax.tick_params(axis="both", which="major", labelsize=6, pad=-2)
    fix_matplotlib_3d(ax)


def _plot_history(ax, case: QICase) -> None:
    colors = ("#1f4e79", "#d95f02", "#2ca25f", "#756bb1", "#636363")
    segments = _history_segments(case)
    if _history_is_effectively_flat(segments) and case.preconditioner_summary is not None:
        _plot_reference_transition(ax, case)
        return
    for idx, segment in enumerate(segments):
        wall_min = np.asarray(segment["wall_time_s"], dtype=float) / 60.0
        objective = np.asarray(segment["objective"], dtype=float)
        best_so_far = _stage_normalized_best_so_far(objective)
        color = colors[idx % len(colors)]
        label = str(segment["label"])
        ax.semilogy(wall_min, best_so_far, color=color, linewidth=1.35, marker="o", markersize=2.4, label=label)
        ax.scatter(wall_min[-1], best_so_far[-1], s=16, color=color, zorder=3)
        if idx > 0:
            ax.axvline(wall_min[0], color="0.75", linewidth=0.7, linestyle="--", zorder=0)
    _plot_preconditioner_sweep(ax, case)
    ax.set_title("Best-so-far staged objective history", fontsize=8, pad=4)
    ax.set_xlabel("Wall time (min)")
    ax.set_ylabel("Best objective / stage start")
    ax.grid(True, alpha=0.22, linestyle=":")
    ax.legend(fontsize=4.8, frameon=False, loc="upper right", handlelength=1.1, labelspacing=0.25)


def _plot_reference_transition(ax, case: QICase) -> None:
    """Show reference-family proposal progress when the local audit objective is flat."""

    row = _selected_preconditioner_row(case)
    diagnostics = _load_json(case.output_dir / "diagnostics.json")
    segments = _history_segments(case)
    wall_min = max(float(np.asarray(segment["wall_time_s"], dtype=float)[-1]) for segment in segments) / 60.0
    x = np.asarray([0.0, max(wall_min, 1.0e-6)], dtype=float)
    smooth = np.asarray(
        [
            max(float(row["smooth_qi"]), 1.0e-16) if row and "smooth_qi" in row else np.nan,
            max(_qi_smooth_total(diagnostics), 1.0e-16),
        ],
        dtype=float,
    )
    legacy = np.asarray(
        [
            max(float(row["legacy_qi"]), 1.0e-16) if row and "legacy_qi" in row else np.nan,
            max(float(diagnostics["qi_legacy_total"]), 1.0e-16),
        ],
        dtype=float,
    )
    mirror = np.asarray(
        [
            max(float(row["mirror"]), 1.0e-16) if row and "mirror" in row else np.nan,
            max(float(diagnostics["qi_mirror_ratio_max"]), 1.0e-16),
        ],
        dtype=float,
    )

    ax.semilogy(x, smooth, color="#1f4e79", linewidth=1.35, marker="o", markersize=2.6, label="smooth QI")
    ax.semilogy(x, legacy, color="#756bb1", linewidth=1.2, marker="o", markersize=2.4, label="legacy QI")
    ax2 = ax.twinx()
    ax2.plot(x, mirror, color="#d95f02", linewidth=1.1, marker="s", markersize=2.4, label="mirror")
    ax.axhline(float(diagnostics.get("qi_smooth_gate", 3.0e-3)), color="#1f4e79", linewidth=0.7, linestyle=":")
    ax.axhline(float(diagnostics.get("qi_legacy_gate", 2.0e-3)), color="#756bb1", linewidth=0.7, linestyle=":")
    ax2.axhline(float(diagnostics["qi_mirror_ratio_target"]), color="#d95f02", linewidth=0.7, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels(("reference\nproposal", "final\naudit"), fontsize=5.6)
    ax.set_title("Reference-family proposal + audit gates", fontsize=8, pad=4)
    ax.set_xlabel("Stage")
    ax.set_ylabel("QI residual")
    ax2.set_ylabel("Mirror ratio", fontsize=7)
    ax.grid(True, alpha=0.22, linestyle=":")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, fontsize=4.8, frameon=False, loc="upper right", handlelength=1.1)
    ax2.tick_params(axis="y", labelsize=5.4, pad=1)


def _plot_preconditioner_sweep(ax, case: QICase) -> None:
    if case.preconditioner_summary is None:
        return
    rows = _load_json(case.preconditioner_summary)
    if not rows:
        return
    lambdas = np.asarray([float(row["lambda"]) for row in rows], dtype=float)
    legacy_qi = np.asarray([max(float(row["legacy_qi"]), 1.0e-16) for row in rows], dtype=float)
    selected = np.asarray([bool(row.get("selected")) for row in rows], dtype=bool)
    inset = ax.inset_axes([0.52, 0.08, 0.43, 0.33])
    inset.semilogy(lambdas, legacy_qi, color="#6a3d9a", marker="o", markersize=2.0, linewidth=1.0)
    if np.any(selected):
        inset.scatter(lambdas[selected], legacy_qi[selected], color="#e31a1c", s=16, zorder=3)
    inset.axhline(1.0e-3, color="0.55", linestyle=":", linewidth=0.7)
    inset.set_title("reference scan QI", fontsize=5.6, pad=1)
    inset.set_xlabel(r"$\lambda$", fontsize=5.4, labelpad=0)
    inset.set_ylabel("legacy QI", fontsize=5.4, labelpad=0)
    inset.tick_params(axis="both", labelsize=4.8, pad=0)
    inset.grid(True, alpha=0.18, linestyle=":")


def _booz_xform_on_outer_surface(wout_path: Path):
    try:
        from booz_xform_jax import Booz_xform
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "render_qi_readme_cases.py requires booz_xform_jax. "
            "Install it with `pip install .` from the repository root."
        ) from exc

    bx = Booz_xform(verbose=0)
    bx.read_wout(str(wout_path))
    bx.compute_surfs = [int(bx.ns_in) - 1]
    bx.mboz = max(16, 2 * int(bx.mpol) + 4)
    bx.nboz = max(16, 2 * int(bx.ntor) + 4)
    bx.run()
    return bx


def _booz_bmag_grid(bx, *, ntheta: int = 128, nphi: int = 192) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * np.pi, ntheta)
    phi = np.linspace(0.0, 2.0 * np.pi / float(bx.nfp), nphi)
    phi2d, theta2d = np.meshgrid(phi, theta)
    xm = np.asarray(bx.xm_b, dtype=float)
    xn = np.asarray(bx.xn_b, dtype=float)
    bmnc = np.asarray(bx.bmnc_b, dtype=float)[:, 0]
    B = np.tensordot(
        bmnc,
        np.cos(xm[:, None, None] * theta2d[None, :, :] - xn[:, None, None] * phi2d[None, :, :]),
        axes=(0, 0),
    )
    if bool(bx.asym) and bx.bmns_b is not None:
        bmns = np.asarray(bx.bmns_b, dtype=float)[:, 0]
        B = B + np.tensordot(
            bmns,
            np.sin(xm[:, None, None] * theta2d[None, :, :] - xn[:, None, None] * phi2d[None, :, :]),
            axes=(0, 0),
        )
    return theta, phi, np.asarray(B)


def _plot_boozer_bmag(ax, wout_path: Path, nfp: int, title: str) -> None:
    bx = _booz_xform_on_outer_surface(wout_path)
    theta, phi, B = _booz_bmag_grid(bx)
    PHI, THETA = np.meshgrid(phi, theta)
    vmin = float(np.nanmin(B))
    vmax = float(np.nanmax(B))
    if vmax <= vmin:
        pad = max(abs(vmin), 1.0) * 1.0e-12
        vmin -= pad
        vmax += pad
    levels = np.linspace(vmin, vmax, 24)
    cs = ax.contour(PHI, THETA, B, levels=levels, cmap="viridis", linewidths=0.9)
    ax.set_title(title, fontsize=8, pad=4)
    ax.set_xlabel(r"$\phi_{B}$ (one field period)")
    ax.set_ylabel(r"$\theta_{B}$")
    ax.set_xlim(0.0, 2.0 * np.pi / float(nfp))
    ax.set_ylim(0.0, 2.0 * np.pi)
    ax.set_yticks([0, np.pi, 2 * np.pi])
    ax.set_yticklabels(["0", r"$\pi$", r"$2\pi$"])
    ax.grid(True, alpha=0.18, linestyle=":")
    ax.figure.colorbar(cs, ax=ax, fraction=0.046, pad=0.018, label="|B|")


def _row_title(record: dict[str, str | float]) -> str:
    status = str(record["validation_status"])
    suffix = f", status={status}"
    return (
        f"{record['case']} | J={record['objective_final']:.2e}, "
        f"QI={record['qi_legacy_total']:.2e}, smooth={record['qi_smooth_total']:.2e}, "
        f"mirror={record['qi_mirror_ratio_max']:.3f}, "
        f"elong={record['qi_max_elongation']:.2f}, "
        f"A={record['aspect']:.3f}, "
        f"iota={record['mean_iota']:.4f}, "
        f"{record['cpu_time_min']:.1f} CPU min"
        f"{suffix}"
    )


def _render(records: list[dict[str, str | float]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "DejaVu Serif",
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    row_count = len(CASES)
    fig = plt.figure(figsize=(21.5, 4.45 * row_count + 0.8))
    gs = fig.add_gridspec(
        row_count,
        5,
        left=0.045,
        right=0.975,
        bottom=0.035,
        top=0.92,
        wspace=0.35,
        hspace=1.05,
        width_ratios=(1.05, 1.05, 1.0, 1.08, 1.08),
    )
    fig.suptitle(
        "QI_optimization coverage for NFP=1, 2, 3, plus an NFP=4 reference-proposal case",
        fontsize=13,
        x=0.02,
        y=0.992,
        ha="left",
    )
    for row, (case, record) in enumerate(zip(CASES, records, strict=True)):
        ax0 = fig.add_subplot(gs[row, 0], projection="3d")
        ax1 = fig.add_subplot(gs[row, 1], projection="3d")
        ax2 = fig.add_subplot(gs[row, 2])
        ax3 = fig.add_subplot(gs[row, 3])
        ax4 = fig.add_subplot(gs[row, 4])
        _plot_lcfs(ax0, case.initial_wout, "Raw input LCFS")
        _plot_lcfs(ax1, case.output_dir / "wout_final.nc", "Final LCFS")
        _plot_history(ax2, case)
        _plot_boozer_bmag(ax3, case.initial_wout, int(record["qi_nfp"]), r"Initial Boozer $|B|$")
        _plot_boozer_bmag(ax4, case.output_dir / "wout_final.nc", int(record["qi_nfp"]), r"Final Boozer $|B|$")
        row_top = max(ax.get_position().y1 for ax in (ax0, ax1, ax2, ax3, ax4))
        title_y = row_top + 0.034
        fig.text(0.045, title_y, _row_title(record), fontsize=10, ha="left", va="bottom")
        fig.text(
            0.045,
            title_y - 0.025,
            f"{record['input_file']} -> {record['output_dir']}",
            fontsize=7,
            ha="left",
            va="bottom",
        )
    fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    prepare_matplotlib_3d()
    records = [_case_record(case) for case in CASES]
    _write_csv(records)
    _render(records)
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_CSV}")
    for record in records:
        print(
            f"{record['case']}: QI={record['qi_legacy_total']:.6e} "
            f"smooth={record['qi_smooth_total']:.6e} "
            f"mirror={record['qi_mirror_ratio_max']:.6g} "
            f"elong={record['qi_max_elongation']:.6g} "
            f"iota={record['mean_iota']:.6g} "
            f"aspect={record['aspect']:.6g} "
            f"cpu_min={record['cpu_time_min']:.3f}"
        )


if __name__ == "__main__":
    main()
