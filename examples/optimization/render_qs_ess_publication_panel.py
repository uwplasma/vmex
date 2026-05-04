#!/usr/bin/env python
"""Render publication-style figures for the QA/QH/QP/QI policy sweep."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
import re
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vmec_jax.plotting import (
    fix_matplotlib_3d,
    prepare_matplotlib_3d,
    vmecplot2_bmag_grid,
    vmecplot2_lcfs_3d_grid,
)
from vmec_jax.wout import read_wout


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR / "results" / "qs_ess_sweep"
REPO_ROOT = SCRIPT_DIR.parents[1]
FIGURE_DIR = REPO_ROOT / "docs" / "_static" / "figures"
PROBLEMS = ("qa", "qh", "qp", "qi")
ESS_OPTIONS = (False, True)
POLICIES = ("continuation", "direct")
ROW_SPECS = (
    ("qa", "continuation"),
    ("qa", "direct"),
    ("qh", "continuation"),
    ("qh", "direct"),
    ("qp", "continuation"),
    ("qp", "direct"),
    ("qi", "continuation"),
    ("qi", "direct"),
)
MODES_BY_POLICY = {
    "continuation": (1, 2, 3),
    "direct": (1, 2, 3),
}
QI_INPUT_NFP = 2
TARGET_ASPECT = 5.0
PROBLEM_TARGET_ASPECT = {
    "qa": TARGET_ASPECT,
    "qh": TARGET_ASPECT,
    "qp": TARGET_ASPECT,
    "qi": TARGET_ASPECT,
}
_TIMEOUT_SECONDS_RE = re.compile(r"timed out after\s+([0-9]+(?:\.[0-9]+)?)\s*s")
_NFP_RE = re.compile(r"^\s*NFP\s*=\s*([0-9]+)", re.IGNORECASE | re.MULTILINE)


@dataclass(frozen=True)
class CaseResult:
    problem: str
    max_mode: int
    use_ess: bool
    success: bool
    crashed: bool
    message: str
    backend: str = "cpu"
    policy: str = "continuation"
    objective_final: float | None = None
    qs_final: float | None = None
    aspect_final: float | None = None
    iota_final: float | None = None
    nfev: int | None = None
    njev: int | None = None
    total_wall_time_s: float | None = None
    output_dir: str | None = None
    jax_backend: str | None = None
    jax_device_kind: str | None = None
    solver_device: str | None = None
    jax_platforms: str | None = None
    stellarator_asymmetric: bool = False
    asymmetry_seed: float = 0.0
    input_file: str | None = None
    input_nfp: int | None = None
    project_input_boundary_to_max_mode: bool | None = None
    target_aspect: float | None = None
    target_iota: float | None = None
    iota_abs_min: float | None = None
    iota_weight: float | None = None
    lgradb_weight: float | None = None
    qi_lgradb_weight: float | None = None
    asymmetric_dof_count: int = 0
    asymmetric_param_norm_initial: float | None = None
    asymmetric_param_norm_final: float | None = None
    asymmetric_param_norm_delta: float | None = None
    bmag_min: float | None = None
    bmag_max: float | None = None
    bmag_nonpositive_fraction: float | None = None
    bmag_finite: bool | None = None
    lgradb_min: float | None = None
    lgradb_threshold: float | None = None
    lgradb_excess_max: float | None = None
    lgradb_diagnostic_error: str | None = None
    qi_qp_preseed: bool | None = None
    qi_qi_preseed: bool | None = None
    qi_raw_total: float | None = None
    qi_mirror_ratio_max: float | None = None
    qi_mirror_ratio_target: float | None = None
    qi_mirror_excess_max: float | None = None
    qi_max_elongation: float | None = None
    qi_elongation_target: float | None = None
    qi_elongation_excess: float | None = None
    qi_lgradb_min: float | None = None
    qi_lgradb_threshold: float | None = None
    qi_lgradb_excess_max: float | None = None
    qi_diagnostic_error: str | None = None


@dataclass
class PlotPayload:
    result: CaseResult
    X: np.ndarray
    Y: np.ndarray
    Z: np.ndarray
    B_surface: np.ndarray
    theta: np.ndarray
    zeta: np.ndarray
    B_contour: np.ndarray
    nfp: int


def _style_publication():
    prepare_matplotlib_3d()
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.17,
            "grid.linestyle": ":",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "DejaVu Serif",
            "font.size": 11,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    return plt


def _ess_label(use_ess: bool) -> str:
    return "ESS" if use_ess else "No ESS"


def _panel_label(index: int) -> str:
    label = ""
    index += 1
    while index:
        index, rem = divmod(index - 1, 26)
        label = chr(ord("A") + rem) + label
    return label


def _policy_label(policy: str) -> str:
    return "Continuation" if policy == "continuation" else "Direct start"


def _format_optional_float(value: float | None, fmt: str, *, missing: str = "-") -> str:
    if value is None:
        return missing
    try:
        return format(float(value), fmt)
    except (TypeError, ValueError):
        return missing


def _format_wall_minutes(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value) / 60.0:.1f}"


def _is_zero_iota_direct_limit(result: CaseResult) -> bool:
    return (
        result.problem == "qa"
        and result.policy == "direct"
        and int(result.max_mode) >= 3
        and result.iota_final is not None
        and abs(float(result.iota_final)) < 5.0e-2
        and result.objective_final is not None
        and float(result.objective_final) > 1.0e-2
    )


def _status_label(result: CaseResult) -> str:
    if result.crashed:
        return "failed"
    if result.bmag_finite is False:
        return "bad |B|"
    if (
        result.bmag_nonpositive_fraction is not None
        and float(result.bmag_nonpositive_fraction) > 0.0
    ):
        return "bad |B|"
    if _is_zero_iota_direct_limit(result):
        return "zero-iota"
    return "ok" if result.success else "stopped"


def _symmetry_label(stellarator_asymmetric: bool) -> str:
    return "LASYM" if bool(stellarator_asymmetric) else "Sym"


def _symmetry_file_label(stellarator_asymmetric: bool) -> str:
    return "asymmetric" if bool(stellarator_asymmetric) else "symmetric"


def _row_label(
    problem: str,
    policy: str,
    backend: str | None = None,
    *,
    stellarator_asymmetric: bool = False,
) -> str:
    prefix = "" if backend is None else f"{backend.upper()} | "
    return f"{prefix}{_symmetry_label(stellarator_asymmetric)} | {problem.upper()} {_policy_label(policy)}"


def _result_key(result: CaseResult) -> tuple[str, bool, str, str, int, bool]:
    return (
        result.backend,
        bool(result.stellarator_asymmetric),
        result.policy,
        result.problem,
        int(result.max_mode),
        bool(result.use_ess),
    )


def _relative_result_parts(path: Path) -> tuple[str, ...]:
    try:
        return path.relative_to(OUTPUT_ROOT).parts
    except ValueError:
        return path.parts


def _normalize_backend_label(value: str) -> str:
    label = str(value).strip().lower()
    if label.startswith("gpu"):
        return "gpu"
    if label.startswith("cpu"):
        return "cpu"
    return label


def _infer_backend(path: Path, record: dict) -> tuple[str, bool, bool]:
    if record.get("backend"):
        return _normalize_backend_label(str(record["backend"])), True, True
    parts = tuple(part.lower() for part in _relative_result_parts(path))
    for backend in ("cpu", "gpu"):
        if backend in parts:
            return _normalize_backend_label(backend), False, True
    return "cpu", False, False


def _infer_policy(path: Path, record: dict) -> tuple[str, bool]:
    if record.get("policy"):
        return str(record["policy"]), True
    parts = tuple(part.lower() for part in _relative_result_parts(path))
    return ("direct", False) if "direct" in parts else ("continuation", False)


def _infer_stellarator_asymmetric(path: Path, record: dict) -> tuple[bool, bool]:
    if "stellarator_asymmetric" in record:
        return bool(record["stellarator_asymmetric"]), True
    parts = tuple(part.lower() for part in _relative_result_parts(path))
    return "asymmetric" in parts or "lasym" in parts, False


def _discovery_priority(path: Path, raw_record: dict) -> int:
    _backend, backend_explicit, backend_in_path = _infer_backend(path, raw_record)
    _policy, policy_explicit = _infer_policy(path, raw_record)
    _asym, asym_explicit = _infer_stellarator_asymmetric(path, raw_record)
    priority = 0
    if backend_in_path:
        priority += 10
    if policy_explicit:
        priority += 2
    if backend_explicit:
        priority += 20
    if asym_explicit:
        priority += 4
    return priority


def _infer_missing_wall_time(record: dict) -> None:
    if record.get("total_wall_time_s") is not None:
        return
    match = _TIMEOUT_SECONDS_RE.search(str(record.get("message", "")))
    if match is None:
        return
    record["total_wall_time_s"] = float(match.group(1))


def _path_from_record_output(record: dict, path: Path) -> Path:
    output_dir = record.get("output_dir")
    if output_dir:
        candidate = Path(str(output_dir))
        if not candidate.is_absolute():
            candidate = REPO_ROOT / candidate
        if candidate.exists():
            return candidate
    return path.parent


def _input_nfp_from_result(record: dict, path: Path) -> int | None:
    value = record.get("input_nfp")
    if value not in (None, ""):
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
    output_dir = _path_from_record_output(record, path)
    for name in ("input.initial", "input.final"):
        candidate = output_dir / name
        if not candidate.exists():
            continue
        match = _NFP_RE.search(candidate.read_text(errors="ignore"))
        if match is not None:
            return int(match.group(1))
    return None


def _lcfs_xyz(R: np.ndarray, Z: np.ndarray, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = R * np.cos(phi[None, :])
    Y = R * np.sin(phi[None, :])
    return X, Y, Z


def _pi_label(v: float) -> str:
    from fractions import Fraction

    if abs(v) < 1e-14:
        return "0"
    frac = Fraction(v / float(np.pi)).limit_denominator(64)
    n, d = frac.numerator, frac.denominator
    if d == 1:
        return "pi" if n == 1 else f"{n}pi"
    return f"pi/{d}" if n == 1 else f"{n}pi/{d}"


def _discover_results() -> list[CaseResult]:
    results_by_key: dict[tuple[str, bool, str, str, int, bool], tuple[int, float, CaseResult]] = {}
    for path in sorted(OUTPUT_ROOT.glob("**/case_result.json")):
        raw_record = json.loads(path.read_text())
        problem_name = str(raw_record.get("problem", ""))
        expected_aspect = float(PROBLEM_TARGET_ASPECT.get(problem_name, TARGET_ASPECT))
        target_aspect = raw_record.get("target_aspect")
        if target_aspect not in (None, "") and abs(float(target_aspect) - expected_aspect) > 1.0e-8:
            continue
        priority = _discovery_priority(path, raw_record)
        record = dict(raw_record)
        backend, backend_explicit, backend_in_path = _infer_backend(path, raw_record)
        if not backend_explicit and not backend_in_path:
            # Ignore pre-backend-layout results. They caused stale timeout/zero-iota
            # records to leak into regenerated CPU/GPU panels after new sweeps.
            continue
        if record.get("problem") == "qi":
            input_nfp = _input_nfp_from_result(record, path)
            if input_nfp != QI_INPUT_NFP:
                continue
            record["input_file"] = record.get("input_file") or "examples/data/input.nfp2_QI"
            record["input_nfp"] = input_nfp
            if record.get("project_input_boundary_to_max_mode") in (None, ""):
                record["project_input_boundary_to_max_mode"] = True
        policy, _policy_explicit = _infer_policy(path, raw_record)
        record["backend"] = backend
        record["policy"] = policy
        asym, _asym_explicit = _infer_stellarator_asymmetric(path, raw_record)
        record["stellarator_asymmetric"] = asym
        _infer_missing_wall_time(record)
        output_dir = record.get("output_dir")
        if output_dir is None or not Path(str(output_dir)).exists():
            record["output_dir"] = str(path.parent)
        result = CaseResult(**record)
        key = _result_key(result)
        mtime = path.stat().st_mtime
        previous = results_by_key.get(key)
        if previous is None or (priority, mtime) >= (previous[0], previous[1]):
            results_by_key[key] = (priority, mtime, result)
    results = [result for _priority, _mtime, result in results_by_key.values()]
    if not results:
        raise FileNotFoundError(f"No case_result.json files found under {OUTPUT_ROOT}")
    return results


def _write_combined_summary(results: list[CaseResult]) -> None:
    ordered = sorted(
        results,
        key=lambda r: (
            r.backend,
            bool(r.stellarator_asymmetric),
            POLICIES.index(r.policy),
            r.problem,
            r.max_mode,
            r.use_ess,
        ),
    )
    records = [_summary_record(r) for r in ordered]
    summary_json = OUTPUT_ROOT / "summary_all.json"
    summary_csv = OUTPUT_ROOT / "summary_all.csv"
    summary_json.write_text(json.dumps(records, indent=2))
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            lineterminator="\n",
            fieldnames=[
                "policy",
                "backend",
                "stellarator_asymmetric",
                "asymmetry_seed",
                "asymmetric_dof_count",
                "asymmetric_param_norm_initial",
                "asymmetric_param_norm_final",
                "asymmetric_param_norm_delta",
                "bmag_min",
                "bmag_max",
                "bmag_nonpositive_fraction",
                "bmag_finite",
                "lgradb_min",
                "lgradb_threshold",
                "lgradb_excess_max",
                "lgradb_diagnostic_error",
                "qi_qp_preseed",
                "qi_qi_preseed",
                "qi_raw_total",
                "qi_mirror_ratio_max",
                "qi_mirror_ratio_target",
                "qi_mirror_excess_max",
                "qi_max_elongation",
                "qi_elongation_target",
                "qi_elongation_excess",
                "qi_lgradb_min",
                "qi_lgradb_threshold",
                "qi_lgradb_excess_max",
                "qi_diagnostic_error",
                "problem",
                "max_mode",
                "use_ess",
                "success",
                "crashed",
                "objective_final",
                "qs_final",
                "aspect_final",
                "iota_final",
                "nfev",
                "njev",
                "total_wall_time_s",
                "jax_backend",
                "jax_device_kind",
                "solver_device",
                "jax_platforms",
                "input_file",
                "input_nfp",
                "project_input_boundary_to_max_mode",
                "target_aspect",
                "target_iota",
                "iota_abs_min",
                "iota_weight",
                "lgradb_weight",
                "qi_lgradb_weight",
                "message",
                "output_dir",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(record)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(summary_csv, FIGURE_DIR / "qs_ess_summary_all.csv")
    shutil.copy2(summary_json, FIGURE_DIR / "qs_ess_summary_all.json")


def _summary_record(result: CaseResult) -> dict:
    record = asdict(result)
    output_dir = record.get("output_dir")
    if output_dir:
        try:
            record["output_dir"] = str(Path(str(output_dir)).resolve().relative_to(REPO_ROOT))
        except ValueError:
            record["output_dir"] = str(output_dir)
    return record


def _result_lookup(results: list[CaseResult]) -> dict[tuple[str, bool, str, str, int, bool], CaseResult]:
    return {_result_key(result): result for result in results}


def _draw_placeholder(ax, message: str, *, title: str | None = None) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("#f6f7f9")
    text_kwargs = {
        "transform": ax.transAxes,
        "ha": "center",
        "va": "center",
        "fontsize": 10,
        "color": "0.42",
        "style": "italic",
    }
    if message:
        if hasattr(ax, "text2D"):
            ax.text2D(0.5, 0.5, message, **text_kwargs)
        else:
            ax.text(0.5, 0.5, message, **text_kwargs)
    if title is not None:
        ax.set_title(title, fontsize=11)


def _history_for(result: CaseResult) -> dict | None:
    if result.output_dir is None:
        return None
    path = Path(result.output_dir) / "history.json"
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


def _lookup_result(
    lookup: dict[tuple[str, bool, str, str, int, bool], CaseResult],
    *,
    backend: str,
    stellarator_asymmetric: bool,
    problem: str,
    max_mode: int,
    use_ess: bool,
    policy: str,
    allow_mode1_baseline: bool = False,
) -> CaseResult | None:
    result = lookup.get((backend, bool(stellarator_asymmetric), policy, problem, max_mode, use_ess))
    if result is not None:
        return result
    if allow_mode1_baseline and policy == "direct" and max_mode == 1:
        return lookup.get((backend, bool(stellarator_asymmetric), "continuation", problem, 1, use_ess))
    return None


def _row_has_history(
    lookup: dict[tuple[str, bool, str, str, int, bool], CaseResult],
    *,
    backend: str,
    stellarator_asymmetric: bool,
    problem: str,
    policy: str,
) -> bool:
    has_any = False
    for max_mode in MODES_BY_POLICY[policy]:
        mode_results = []
        for use_ess in ESS_OPTIONS:
            result = _lookup_result(
                lookup,
                backend=backend,
                stellarator_asymmetric=stellarator_asymmetric,
                problem=problem,
                max_mode=max_mode,
                use_ess=use_ess,
                policy=policy,
                allow_mode1_baseline=True,
            )
            if result is not None:
                has_any = True
            mode_results.append(result)
        if all(result is not None for result in mode_results):
            return True
    # Partial LASYM matrices are publishable: a 1200 s timeout or OOM is useful
    # sweep information.  Missing lanes are kept as blank cells instead of
    # emitting stale "pending" labels into README figures.
    return has_any


def _available_row_specs(results: list[CaseResult]) -> list[tuple[str, bool, str, str]]:
    lookup = _result_lookup(results)
    backends = sorted({result.backend for result in results})
    symmetries = sorted({bool(result.stellarator_asymmetric) for result in results})
    return [
        (backend, stellarator_asymmetric, problem, policy)
        for backend in backends
        for stellarator_asymmetric in symmetries
        for problem, policy in ROW_SPECS
        if _row_has_history(
            lookup,
            backend=backend,
            stellarator_asymmetric=stellarator_asymmetric,
            problem=problem,
            policy=policy,
        )
    ]


def _policy_matrix_complete(
    results: list[CaseResult],
    *,
    backend: str,
    stellarator_asymmetric: bool,
    policy: str,
) -> bool:
    if not bool(stellarator_asymmetric):
        return True
    expected = {
        (problem, mode, use_ess)
        for problem in PROBLEMS
        for mode in MODES_BY_POLICY[policy]
        for use_ess in ESS_OPTIONS
    }
    present = {
        (result.problem, int(result.max_mode), bool(result.use_ess))
        for result in results
        if (
            result.backend == backend
            and bool(result.stellarator_asymmetric) is True
            and result.policy == policy
        )
    }
    return expected <= present


def _publication_results(results: list[CaseResult]) -> list[CaseResult]:
    """Return rows for the currently published mode policy."""

    return [result for result in results if int(result.max_mode) in MODES_BY_POLICY[result.policy]]


def _problems_with_payloads(
    payloads: dict[tuple[str, bool, str, str, int, bool], PlotPayload],
    *,
    backend: str,
    stellarator_asymmetric: bool,
    policy: str,
) -> tuple[str, ...]:
    modes = MODES_BY_POLICY[policy]
    return tuple(
        problem
        for problem in PROBLEMS
        if any(
            (backend, bool(stellarator_asymmetric), policy, problem, mode, use_ess) in payloads
            for mode in modes
            for use_ess in ESS_OPTIONS
        )
    )


def _load_payloads(results: list[CaseResult]) -> dict[tuple[str, bool, str, str, int, bool], PlotPayload]:
    payloads: dict[tuple[str, bool, str, str, int, bool], PlotPayload] = {}
    for result in results:
        if result.crashed or result.output_dir is None:
            continue
        wout_path = Path(result.output_dir) / "wout_final.nc"
        if not wout_path.exists():
            continue
        wout = read_wout(str(wout_path))
        ns = int(np.asarray(wout.ns))
        theta3d, phi3d, R3d, Z3d, B3d = vmecplot2_lcfs_3d_grid(
            wout,
            s_index=ns - 1,
            ntheta=48,
            nzeta=96,
        )
        X3d, Y3d, Z3d = _lcfs_xyz(R3d, Z3d, phi3d)
        zeta_max = 2.0 * np.pi / int(np.asarray(wout.nfp))
        theta2d, zeta2d, B2d = vmecplot2_bmag_grid(
            wout,
            s_index=ns - 1,
            ntheta=128,
            nzeta=192,
            zeta_max=zeta_max,
        )
        payloads[_result_key(result)] = PlotPayload(
            result=result,
            X=X3d,
            Y=Y3d,
            Z=Z3d,
            B_surface=B3d,
            theta=theta2d,
            zeta=zeta2d,
            B_contour=B2d,
            nfp=int(np.asarray(wout.nfp)),
        )
    return payloads


def _plot_objective_panel_all_policies(results: list[CaseResult], outpath_png: Path, outpath_pdf: Path) -> None:
    plt = _style_publication()

    lookup = _result_lookup(results)
    row_specs = _available_row_specs(results)
    if not row_specs:
        raise ValueError("No optimization histories are available to plot")
    modes_to_plot = tuple(sorted({mode for modes in MODES_BY_POLICY.values() for mode in modes}))
    ncols = len(modes_to_plot)
    fig, axes = plt.subplots(
        len(row_specs),
        ncols,
        figsize=(5.9 * ncols, 3.65 * len(row_specs)),
        sharey="row",
    )
    if len(row_specs) == 1:
        axes = np.asarray([axes])
    colors = {False: "#1f77b4", True: "#d95f02"}
    line_labels = {False: "No ESS", True: "ESS"}
    mode_titles = {mode: ("Mode 1 baseline" if mode == 1 else f"Mode {mode}") for mode in modes_to_plot}

    for row_index, (backend, stellarator_asymmetric, problem, policy) in enumerate(row_specs):
        for col_index, max_mode in enumerate(modes_to_plot):
            ax = axes[row_index, col_index]
            panel_label = _panel_label(row_index * ncols + col_index)
            ax.text(
                0.01,
                0.99,
                panel_label,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=13,
                fontweight="bold",
            )
            if row_index == 0:
                ax.set_title(mode_titles[max_mode], fontsize=12)
            annotation_lines = []
            plotted_any = False
            baseline_note = policy == "direct" and max_mode == 1
            for use_ess in ESS_OPTIONS:
                result = _lookup_result(
                    lookup,
                    backend=backend,
                    stellarator_asymmetric=stellarator_asymmetric,
                    problem=problem,
                    max_mode=max_mode,
                    use_ess=use_ess,
                    policy=policy,
                    allow_mode1_baseline=True,
                )
                if result is None:
                    continue
                history = _history_for(result)
                if history is None:
                    continue
                segments = []
                start_index = 0
                for segment in _history_stage_segments(history["history"]):
                    stop_index = start_index + len(segment)
                    segments.append(
                        (
                            np.arange(start_index, stop_index, dtype=float),
                            np.minimum.accumulate(
                                np.asarray(
                                    [max(float(entry["objective"]), 1e-16) for entry in segment],
                                    dtype=float,
                                )
                            ),
                        )
                    )
                    start_index = stop_index
                linestyle = "-" if result.success and not result.crashed else "--"
                alpha = 0.70 if baseline_note else 1.0
                linewidth = 2.5 if use_ess else 2.1
                label = line_labels[use_ess] if (row_index == 0 and col_index == 0) else None
                first_segment = True
                last_x = None
                last_objective = None
                for x, objective in segments:
                    ax.semilogy(
                        x,
                        objective,
                        color=colors[use_ess],
                        linestyle=linestyle,
                        linewidth=linewidth,
                        alpha=alpha,
                        label=label if first_segment else None,
                    )
                    first_segment = False
                    last_x = x[-1]
                    last_objective = objective[-1]
                plotted_any = True
                if last_x is not None and last_objective is not None:
                    ax.scatter(last_x, last_objective, color=colors[use_ess], s=28, zorder=4, alpha=alpha)
                for boundary in history.get("stage_boundaries", [])[:-1]:
                    ax.axvline(float(boundary), color="0.75", linestyle=":", linewidth=1.0, zorder=0)

                wall_min = float(result.total_wall_time_s) / 60.0
                objective_text = (
                    f"{float(result.objective_final):.2e}" if result.objective_final is not None else "n/a"
                )
                meta = f"{line_labels[use_ess]}: J={objective_text}, {wall_min:.1f}m"
                if result.aspect_final is not None:
                    meta += f", A={float(result.aspect_final):.3f}"
                if result.iota_final is not None:
                    meta += f", iota={float(result.iota_final):.4f}"
                if result.solver_device:
                    meta += f", dev={result.solver_device}"
                if _is_zero_iota_direct_limit(result):
                    meta += ", zero-iota branch"
                annotation_lines.append(meta)

            if baseline_note:
                ax.text(
                    0.98,
                    0.98,
                    "shared mode-1 baseline",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8.5,
                    color="0.35",
                )
            row_title = _row_label(
                problem,
                policy,
                backend,
                stellarator_asymmetric=stellarator_asymmetric,
            )
            if col_index == 0:
                ax.set_ylabel(f"{row_title}\nTotal objective", fontsize=11)
            if row_index == len(row_specs) - 1:
                ax.set_xlabel("History index", fontsize=11)
            ax.set_xlim(left=0)
            ax.grid(True, which="both", alpha=0.20)
            if not plotted_any:
                fallback_results = [
                    _lookup_result(
                        lookup,
                        backend=backend,
                        stellarator_asymmetric=stellarator_asymmetric,
                        problem=problem,
                        max_mode=max_mode,
                        use_ess=use_ess,
                        policy=policy,
                        allow_mode1_baseline=True,
                    )
                    for use_ess in ESS_OPTIONS
                ]
                fallback_lines = []
                for use_ess, result in zip(ESS_OPTIONS, fallback_results, strict=True):
                    if result is None:
                        continue
                    if result.crashed:
                        status = "timed out" if "timed out" in result.message.lower() else "failed"
                    else:
                        status = "no history"
                    wall = _format_wall_minutes(result.total_wall_time_s)
                    if wall != "-":
                        status = f"{status}, {wall} min"
                    fallback_lines.append(f"{line_labels[use_ess]}: {status}")
                _draw_placeholder(ax, "\n".join(fallback_lines))
                ax.text(
                    0.01,
                    0.99,
                    panel_label,
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=13,
                    fontweight="bold",
                )
            if annotation_lines:
                ax.text(
                    0.02,
                    0.02,
                    "\n".join(annotation_lines),
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=8.3,
                    bbox={"boxstyle": "round,pad=0.24", "facecolor": "white", "edgecolor": "0.86", "alpha": 0.92},
                )

    from matplotlib.lines import Line2D

    handles, labels = axes[0, 0].get_legend_handles_labels()
    status_handles = [
        Line2D([0], [0], color="0.25", linestyle="-", linewidth=2.2, label="solid: optimizer success"),
        Line2D([0], [0], color="0.25", linestyle="--", linewidth=2.2, label="dashed: stopped, failed, or budgeted"),
    ]
    if handles:
        fig.legend(
            handles + status_handles,
            labels + [handle.get_label() for handle in status_handles],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.985),
            ncol=4,
            frameon=False,
        )
    fig.suptitle(
        "QA/QH/QP/QI optimization histories by backend and boundary symmetry",
        y=0.998,
        fontsize=16,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    fig.savefig(outpath_png, dpi=220, bbox_inches="tight")
    fig.savefig(outpath_pdf, bbox_inches="tight")
    plt.close(fig)


def _plot_state_atlas(
    results: list[CaseResult],
    payloads: dict[tuple[str, bool, str, str, int, bool], PlotPayload],
    *,
    policy: str,
    backend: str = "cpu",
    stellarator_asymmetric: bool = False,
    outpath_png: Path,
    outpath_pdf: Path,
) -> bool:
    plt = _style_publication()
    from matplotlib import cm
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    lookup = _result_lookup(results)
    modes = MODES_BY_POLICY[policy]
    problems = _problems_with_payloads(
        payloads,
        backend=backend,
        stellarator_asymmetric=stellarator_asymmetric,
        policy=policy,
    )
    if not problems:
        return False
    columns = [(mode, use_ess) for mode in modes for use_ess in ESS_OPTIONS]
    ncols = len(columns)
    nrows = 2 * len(problems)
    fig = plt.figure(figsize=(3.95 * ncols, 3.15 * nrows))
    grid = fig.add_gridspec(nrows, ncols, wspace=0.14, hspace=0.22)

    def _finite_range(values: np.ndarray) -> tuple[float, float]:
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            vmin, vmax = 0.0, 1.0
        if vmax <= vmin:
            pad = max(abs(vmin), 1.0) * 1e-12
            vmin -= pad
            vmax += pad
        return vmin, vmax

    def _inset_colorbar(mappable, ax, *, label: str, height: str) -> None:
        cax = inset_axes(ax, width="4.5%", height=height, loc="lower right", borderpad=0.62)
        cbar = fig.colorbar(mappable, cax=cax)
        cbar.set_label(label, fontsize=7.2)
        cbar.ax.tick_params(labelsize=6.7, length=2)

    for col_index, (max_mode, use_ess) in enumerate(columns):
        for problem_index, problem in enumerate(problems):
            row_surface = problem_index * 2
            row_contour = row_surface + 1
            result = _lookup_result(
                lookup,
                backend=backend,
                stellarator_asymmetric=stellarator_asymmetric,
                problem=problem,
                max_mode=max_mode,
                use_ess=use_ess,
                policy=policy,
                allow_mode1_baseline=True,
            )
            payload = payloads.get(_result_key(result)) if result is not None else None
            title = None
            if problem_index == 0:
                title = f"mode {max_mode} | {_ess_label(use_ess)}"
            if result is None or payload is None:
                ax3d = fig.add_subplot(grid[row_surface, col_index], projection="3d")
                _draw_placeholder(ax3d, "", title=title)
                if col_index == 0:
                    ax3d.text2D(
                        -0.16,
                        0.5,
                        f"{problem.upper()} LCFS",
                        transform=ax3d.transAxes,
                        rotation=90,
                        va="center",
                        ha="center",
                        fontsize=12,
                        fontweight="bold",
                    )
                ax2d = fig.add_subplot(grid[row_contour, col_index])
                _draw_placeholder(ax2d, "")
                if col_index == 0:
                    ax2d.set_ylabel("theta")
                    ax2d.text(
                        -0.19,
                        0.5,
                        f"{problem.upper()} |B|",
                        transform=ax2d.transAxes,
                        rotation=90,
                        va="center",
                        ha="center",
                        fontsize=12,
                        fontweight="bold",
                    )
                ax2d.set_xlabel("zeta")
                continue

            ax3d = fig.add_subplot(grid[row_surface, col_index], projection="3d")
            b3_min, b3_max = _finite_range(payload.B_surface)
            surface_norm = Normalize(vmin=b3_min, vmax=b3_max)
            facecolors = cm.viridis(surface_norm(payload.B_surface))
            ax3d.plot_surface(
                payload.X,
                payload.Y,
                payload.Z,
                facecolors=facecolors,
                rstride=1,
                cstride=1,
                linewidth=0,
                antialiased=False,
                shade=False,
            )
            ax3d.view_init(elev=24, azim=42)
            fix_matplotlib_3d(ax3d)
            ax3d.set_xticks([])
            ax3d.set_yticks([])
            ax3d.set_zticks([])
            sm = ScalarMappable(norm=surface_norm, cmap=cm.viridis)
            sm.set_array([])
            _inset_colorbar(sm, ax3d, label="|B| (T)", height="46%")
            if problem_index == 0:
                wall_min = float(result.total_wall_time_s) / 60.0
                objective_text = (
                    f"{float(result.objective_final):.2e}" if result.objective_final is not None else "n/a"
                )
                ax3d.set_title(
                    f"mode {max_mode} | {_ess_label(use_ess)}\nJ={objective_text}, {wall_min:.1f} min",
                    pad=10,
                )
            if col_index == 0:
                ax3d.text2D(
                    -0.16,
                    0.5,
                    f"{problem.upper()} LCFS",
                    transform=ax3d.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=12,
                    fontweight="bold",
                )

            ax2d = fig.add_subplot(grid[row_contour, col_index])
            zeta_mesh, theta_mesh = np.meshgrid(payload.zeta, payload.theta)
            b2_min, b2_max = _finite_range(payload.B_contour)
            contour_levels = np.linspace(b2_min, b2_max, 22)
            contours = ax2d.contour(
                zeta_mesh,
                theta_mesh,
                payload.B_contour,
                levels=contour_levels,
                cmap="viridis",
                linewidths=1.0,
            )
            _inset_colorbar(contours, ax2d, label="|B| (T)", height="78%")
            ax2d.set_ylim(0.0, 2.0 * np.pi)
            ax2d.set_xlim(0.0, float(np.max(payload.zeta)))
            ax2d.set_yticks([0.0, np.pi, 2.0 * np.pi])
            if col_index == 0:
                ax2d.set_yticklabels(["0", "pi", "2pi"])
                ax2d.set_ylabel("theta")
                ax2d.text(
                    -0.19,
                    0.5,
                    f"{problem.upper()} |B|",
                    transform=ax2d.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=12,
                    fontweight="bold",
                )
            else:
                ax2d.set_yticklabels([])
            xticks = [0.0, float(np.max(payload.zeta)) / 2.0, float(np.max(payload.zeta))]
            ax2d.set_xticks(xticks)
            ax2d.set_xticklabels([_pi_label(v) for v in xticks])
            ax2d.set_xlabel("zeta")
            ax2d.grid(False)
            wall_min = float(result.total_wall_time_s) / 60.0
            meta = f"A={float(result.aspect_final):.3f}\nwall={wall_min:.1f} min"
            if result.iota_final is not None:
                meta = f"A={float(result.aspect_final):.3f}\niota={float(result.iota_final):.4f}\nwall={wall_min:.1f} min"
            if _is_zero_iota_direct_limit(result):
                meta += "\nzero-iota branch"
            ax2d.text(
                0.02,
                0.02,
                meta,
                transform=ax2d.transAxes,
                ha="left",
                va="bottom",
                fontsize=8.1,
                bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "0.86", "alpha": 0.92},
            )

    fig.suptitle(
        (
            f"Final-state atlas: {backend.upper()} {_symmetry_label(stellarator_asymmetric)} "
            f"{_policy_label(policy)} policy"
        ),
        y=0.995,
        fontsize=16,
    )
    fig.subplots_adjust(left=0.045, right=0.985, top=0.93, bottom=0.05)
    fig.savefig(outpath_png, dpi=220, bbox_inches="tight")
    fig.savefig(outpath_pdf, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_summary_tables(results: list[CaseResult], outpath_png: Path, outpath_pdf: Path) -> None:
    plt = _style_publication()

    row_specs = _available_row_specs(results)
    if not row_specs:
        raise ValueError("No optimization summary rows are available to plot")
    ncols = 2
    nrows = int(np.ceil(len(row_specs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5.25 * nrows))
    axes_arr = np.asarray(axes).ravel()
    for ax in axes_arr[len(row_specs):]:
        ax.axis("off")
    for ax, (backend, stellarator_asymmetric, problem, policy) in zip(axes_arr, row_specs):
        ax.axis("off")
        lookup = _result_lookup(results)
        group = [
            result
            for max_mode in MODES_BY_POLICY[policy]
            for use_ess in ESS_OPTIONS
            if (
                result := _lookup_result(
                    lookup,
                    backend=backend,
                    stellarator_asymmetric=stellarator_asymmetric,
                    problem=problem,
                    max_mode=max_mode,
                    use_ess=use_ess,
                    policy=policy,
                    allow_mode1_baseline=True,
                )
            )
            is not None
        ]
        if not group:
            _draw_placeholder(ax, "")
            ax.set_title(
                _row_label(problem, policy, backend, stellarator_asymmetric=stellarator_asymmetric),
                fontsize=12,
                pad=8,
            )
            continue
        finite_objective_indices = [
            i for i, result in enumerate(group) if result.objective_final is not None and not result.crashed
        ]
        best_index = (
            min(finite_objective_indices, key=lambda i: float(group[i].objective_final))
            if finite_objective_indices
            else -1
        )
        if problem in ("qa", "qp"):
            columns = ["Configuration", "Status", "Final J", "Aspect", "Iota", "nfev", "Wall (min)"]
            widths = [0.27, 0.13, 0.16, 0.11, 0.11, 0.08, 0.14]
            rows = [
                [
                    f"mode {result.max_mode} | {_ess_label(result.use_ess)}",
                    _status_label(result),
                    _format_optional_float(result.objective_final, ".2e"),
                    _format_optional_float(result.aspect_final, ".4f"),
                    _format_optional_float(result.iota_final, ".4f"),
                    "-" if result.nfev is None else f"{int(result.nfev)}",
                    _format_wall_minutes(result.total_wall_time_s),
                ]
                for result in group
            ]
        else:
            columns = ["Configuration", "Status", "Final J", "Aspect", "nfev", "Wall (min)"]
            widths = [0.32, 0.14, 0.18, 0.13, 0.09, 0.14]
            rows = [
                [
                    f"mode {result.max_mode} | {_ess_label(result.use_ess)}",
                    _status_label(result),
                    _format_optional_float(result.objective_final, ".2e"),
                    _format_optional_float(result.aspect_final, ".4f"),
                    "-" if result.nfev is None else f"{int(result.nfev)}",
                    _format_wall_minutes(result.total_wall_time_s),
                ]
                for result in group
            ]
        table = ax.table(
            cellText=rows,
            colLabels=columns,
            loc="center",
            cellLoc="center",
            colLoc="center",
            colWidths=widths,
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9.4)
        table.scale(1.0, 1.55)
        for (row_index, _col_index), cell in table.get_celld().items():
            if row_index == 0:
                cell.set_facecolor("#e9eef5")
                cell.set_text_props(weight="bold")
            else:
                cell.set_edgecolor("0.82")
                is_ess_row = rows[row_index - 1][0].split("|", 1)[1].strip() == "ESS"
                is_failed_row = rows[row_index - 1][1] == "failed"
                is_limited_row = rows[row_index - 1][1] == "zero-iota"
                if row_index - 1 == best_index:
                    cell.set_facecolor("#e6f4ea")
                elif is_failed_row:
                    cell.set_facecolor("#f8d7da")
                elif is_limited_row:
                    cell.set_facecolor("#fff3cd")
                elif is_ess_row:
                    cell.set_facecolor("#fff3e8")
                else:
                    cell.set_facecolor("#eef5ff")
        ax.set_title(
            _row_label(problem, policy, backend, stellarator_asymmetric=stellarator_asymmetric),
            fontsize=12,
            pad=8,
        )

    fig.suptitle("Sweep summary tables", y=0.99, fontsize=16)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(outpath_png, dpi=220, bbox_inches="tight")
    fig.savefig(outpath_pdf, bbox_inches="tight")
    plt.close(fig)


def _assemble_full_panel(
    image_paths: list[Path],
    titles: list[str],
    outpath_png: Path,
    outpath_pdf: Path,
) -> None:
    plt = _style_publication()
    import matplotlib.image as mpimg

    images = [mpimg.imread(path) for path in image_paths]
    if not images:
        raise ValueError("No image paths were provided for the full panel")
    height_ratios = [1.0]
    if len(images) > 2:
        height_ratios.extend([1.05] * (len(images) - 2))
    if len(images) > 1:
        height_ratios.append(0.62)
    fig, axes = plt.subplots(
        len(images),
        1,
        figsize=(22, 9.5 + 8.2 * len(images)),
        gridspec_kw={"height_ratios": height_ratios},
    )
    axes = np.asarray(axes).ravel()
    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(title, loc="left", fontsize=16, pad=10, fontweight="bold")
    fig.suptitle("Reviewer-facing QA/QH/QP/QI optimization policy sweep panel", y=0.996, fontsize=19)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.99))
    fig.savefig(outpath_png, dpi=220, bbox_inches="tight")
    fig.savefig(outpath_pdf, bbox_inches="tight")
    plt.close(fig)


def _render_publication_set(
    *,
    results: list[CaseResult],
    payloads: dict[tuple[str, bool, str, str, int, bool], PlotPayload],
    output_suffix: str,
    title_suffix: str,
    alias_legacy: bool = False,
) -> None:
    objective_stem = "objective_panel_all_policies" if not output_suffix else f"objective_panel_{output_suffix}_all_policies"
    summary_stem = "summary_tables_all_policies" if not output_suffix else f"summary_tables_{output_suffix}_all_policies"
    panel_stem = "publication_panel_full" if not output_suffix else f"publication_panel_{output_suffix}_full"
    objective_png = OUTPUT_ROOT / f"{objective_stem}.png"
    objective_pdf = OUTPUT_ROOT / f"{objective_stem}.pdf"
    summary_png = OUTPUT_ROOT / f"{summary_stem}.png"
    summary_pdf = OUTPUT_ROOT / f"{summary_stem}.pdf"
    panel_png = OUTPUT_ROOT / f"{panel_stem}.png"
    panel_pdf = OUTPUT_ROOT / f"{panel_stem}.pdf"

    _plot_objective_panel_all_policies(results, objective_png, objective_pdf)
    atlas_paths: list[tuple[Path, Path, str]] = []
    for backend in sorted({result.backend for result in results}):
        for stellarator_asymmetric in sorted({bool(result.stellarator_asymmetric) for result in results}):
            symmetry_file_label = _symmetry_file_label(stellarator_asymmetric)
            for policy in POLICIES:
                if not _problems_with_payloads(
                    payloads,
                    backend=backend,
                    stellarator_asymmetric=stellarator_asymmetric,
                    policy=policy,
                ):
                    continue
                atlas_stem = (
                    f"final_state_atlas_{backend}_{policy}"
                    if not stellarator_asymmetric
                    else f"final_state_atlas_{backend}_{symmetry_file_label}_{policy}"
                )
                if output_suffix:
                    atlas_stem = f"final_state_atlas_{output_suffix}_{backend}_{policy}"
                atlas_png = OUTPUT_ROOT / f"{atlas_stem}.png"
                atlas_pdf = OUTPUT_ROOT / f"{atlas_stem}.pdf"
                wrote_atlas = _plot_state_atlas(
                    results,
                    payloads,
                    policy=policy,
                    backend=backend,
                    stellarator_asymmetric=stellarator_asymmetric,
                    outpath_png=atlas_png,
                    outpath_pdf=atlas_pdf,
                )
                if not wrote_atlas:
                    continue
                atlas_paths.append(
                    (
                        atlas_png,
                        atlas_pdf,
                        (
                            f"Final-state atlas: {backend.upper()} "
                            f"{_symmetry_label(stellarator_asymmetric)} "
                            f"{_policy_label(policy).lower()} policy"
                        ),
                    )
                )
                if alias_legacy and backend == "cpu" and not stellarator_asymmetric:
                    shutil.copy2(atlas_png, OUTPUT_ROOT / f"final_state_atlas_{policy}.png")
                    shutil.copy2(atlas_pdf, OUTPUT_ROOT / f"final_state_atlas_{policy}.pdf")
                    if policy == "continuation":
                        _copy_alias(OUTPUT_ROOT / f"final_state_atlas_{policy}", OUTPUT_ROOT / "geometry_atlas")

    _plot_summary_tables(results, summary_png, summary_pdf)
    image_paths = [objective_png] + [path for path, _pdf, _title in atlas_paths] + [summary_png]
    panel_titles = [f"A. Objective histories: {title_suffix}"]
    for index, (_path, _pdf, title) in enumerate(atlas_paths, start=1):
        panel_titles.append(f"{chr(ord('A') + index)}. {title}")
    panel_titles.append(f"{chr(ord('A') + len(panel_titles))}. Sweep summary tables")
    _assemble_full_panel(image_paths, panel_titles, panel_png, panel_pdf)

    if alias_legacy:
        _copy_alias(OUTPUT_ROOT / "objective_panel_all_policies", OUTPUT_ROOT / "objective_panel")
        _copy_alias(OUTPUT_ROOT / "summary_tables_all_policies", OUTPUT_ROOT / "summary_table")
        _copy_alias(OUTPUT_ROOT / "publication_panel_full", OUTPUT_ROOT / "publication_panel")

    print(f"Wrote {objective_png}")
    for atlas_png, _atlas_pdf, _title in atlas_paths:
        print(f"Wrote {atlas_png}")
    print(f"Wrote {summary_png}")
    print(f"Wrote {panel_png}")


def _copy_alias(src_stem: Path, dst_stem: Path) -> None:
    """Copy a png/pdf figure pair so legacy names do not stay stale."""
    for suffix in (".png", ".pdf"):
        src = src_stem.with_suffix(suffix)
        if src.exists():
            shutil.copy2(src, dst_stem.with_suffix(suffix))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Sweep result root to read and write rendered panels. Defaults to examples/optimization/results/qs_ess_sweep.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help=(
            "Only regenerate summary_all.csv/json and the docs copies. "
            "This avoids loading all final wout files for heavyweight atlas rendering."
        ),
    )
    return parser.parse_args()


def main() -> None:
    global OUTPUT_ROOT
    args = _parse_args()
    OUTPUT_ROOT = args.output_root.resolve()

    results = _discover_results()
    results = _publication_results(results)
    _write_combined_summary(results)
    if args.summary_only:
        print(f"Wrote {OUTPUT_ROOT / 'summary_all.csv'}")
        print(f"Wrote {OUTPUT_ROOT / 'summary_all.json'}")
        print(f"Wrote {FIGURE_DIR / 'qs_ess_summary_all.csv'}")
        print(f"Wrote {FIGURE_DIR / 'qs_ess_summary_all.json'}")
        return
    payloads = _load_payloads(results)

    _render_publication_set(
        results=results,
        payloads=payloads,
        output_suffix="",
        title_suffix="all symmetric and LASYM policies",
        alias_legacy=True,
    )
    for backend in sorted({result.backend for result in results}):
        backend_results = [result for result in results if result.backend == backend]
        _plot_objective_panel_all_policies(
            backend_results,
            OUTPUT_ROOT / f"objective_panel_{backend}_policies.png",
            OUTPUT_ROOT / f"objective_panel_{backend}_policies.pdf",
        )
        _plot_summary_tables(
            backend_results,
            OUTPUT_ROOT / f"summary_tables_{backend}_policies.png",
            OUTPUT_ROOT / f"summary_tables_{backend}_policies.pdf",
        )
    for stellarator_asymmetric in sorted({bool(result.stellarator_asymmetric) for result in results}):
        if not stellarator_asymmetric:
            continue
        subset = [result for result in results if bool(result.stellarator_asymmetric) is stellarator_asymmetric]
        subset_payloads = {key: payload for key, payload in payloads.items() if key[1] is stellarator_asymmetric}
        _render_publication_set(
            results=subset,
            payloads=subset_payloads,
            output_suffix=_symmetry_file_label(stellarator_asymmetric),
            title_suffix="non-stellarator-symmetric LASYM policies",
            alias_legacy=False,
        )


if __name__ == "__main__":
    main()
