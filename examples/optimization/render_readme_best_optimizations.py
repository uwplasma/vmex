#!/usr/bin/env python
"""Render the compact README panel of best symmetric QA/QH/QP/QI runs.

The full optimization matrix lives in the documentation.  This script keeps the
README focused on one representative stellarator-symmetric result for each
target and evaluates the initial/final |B| contours in Boozer coordinates
through ``booz_xform_jax``.  QA/QH/QP use the best CPU rows from the all-policy
sweep; QI uses the constrained QI matrix when available so mirror ratio and
elongation diagnostics enter the selection.
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
SWEEP_ROOT = SCRIPT_DIR / "results" / "qs_ess_sweep"
FIGURE_DIR = REPO_ROOT / "docs" / "_static" / "figures"
SUMMARY_CSV = FIGURE_DIR / "qs_ess_summary_all.csv"
QI_CONSTRAINED_CSV = FIGURE_DIR / "qi_constrained_summary.csv"
QI_CONSTRAINED_BEST_JSON = FIGURE_DIR / "qi_constrained_best.json"
OUT_CSV = FIGURE_DIR / "readme_best_optimizations.csv"
QI_DEFAULT_RESULT_DIR = REPO_ROOT / "results" / "qi_opt" / "ess" / "nfp2_qi_aspect6"
README_CASES_ROOT = REPO_ROOT / "docs" / "_static" / "readme_best_cases"
README_CASES_SUMMARY = README_CASES_ROOT / "summary.csv"

PROBLEMS = ("qa", "qh", "qp", "qi")
PROBLEM_TITLES = {
    "qa": "QA",
    "qh": "QH",
    "qp": "QP",
    "qi": "QI",
}
TARGET_ASPECT = 6.0
QI_TARGET_ASPECT = TARGET_ASPECT
PROBLEM_TARGET_ASPECT = {
    "qa": TARGET_ASPECT,
    "qh": TARGET_ASPECT,
    "qp": TARGET_ASPECT,
    "qi": QI_TARGET_ASPECT,
}
TARGET_ABS_IOTA_MIN = 0.41
TARGET_QA_IOTA = 0.42


@dataclass(frozen=True)
class BestRun:
    problem: str
    policy: str
    max_mode: int
    use_ess: bool
    objective_final: float
    aspect_final: float
    iota_final: float
    total_wall_time_s: float
    output_dir: Path
    backend: str = "cpu"
    qi_qp_preseed: bool | None = None
    qi_raw_total: float | None = None
    qi_legacy_total: float | None = None
    qi_mirror_ratio_max: float | None = None
    qi_max_elongation: float | None = None
    lgradb_min: float | None = None
    qi_lgradb_min: float | None = None
    input_file: Path | None = None


def _read_summary_rows() -> list[dict[str, str]]:
    with SUMMARY_CSV.open(newline="") as f:
        return list(csv.DictReader(f))


def _repo_relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _read_qi_constrained_rows() -> list[dict[str, str]]:
    if not QI_CONSTRAINED_CSV.exists():
        return []
    with QI_CONSTRAINED_CSV.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["problem"] = "qi"
    return rows


def _bool_value(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _path_from_summary(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _float_value(row: dict[str, str], key: str, default: float | None = None) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _qi_metric(row: dict[str, str], default: float | None = None) -> float | None:
    value = row.get("qi_legacy_total")
    if value in (None, ""):
        value = row.get("qi_raw_total")
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _target_aspect_for(problem: str) -> float:
    return float(PROBLEM_TARGET_ASPECT.get(problem, TARGET_ASPECT))


def _qi_selection_score(row: dict[str, str]) -> tuple:
    failed = 1 if _bool_value(row.get("crashed")) or not _bool_value(row.get("success")) else 0
    mirror = _float_value(row, "qi_mirror_ratio_max", np.inf) or np.inf
    mirror_target = _float_value(row, "qi_mirror_ratio_target", 0.21) or 0.21
    elong = _float_value(row, "qi_max_elongation", np.inf) or np.inf
    elong_target = _float_value(row, "qi_elongation_target", 8.0) or 8.0
    iota = abs(_float_value(row, "iota_final", 0.0) or 0.0)
    aspect = _float_value(row, "aspect_final", np.inf) or np.inf
    qi_raw = _qi_metric(row, np.inf) or np.inf
    objective = _float_value(row, "objective_final", np.inf) or np.inf
    mirror_violation = max(0.0, mirror - mirror_target) / max(mirror_target, 1.0e-12)
    elong_violation = max(0.0, elong - elong_target) / max(elong_target, 1.0e-12)
    iota_violation = max(0.0, TARGET_ABS_IOTA_MIN - iota) / TARGET_ABS_IOTA_MIN
    aspect_target = _target_aspect_for("qi")
    aspect_violation = abs(aspect - aspect_target) / aspect_target
    qp_preseed = 1 if _bool_value(row.get("qi_qp_preseed")) else 0
    hard_ok = int(
        mirror_violation <= 0.10
        and elong_violation <= 0.05
        and iota_violation <= 0.025
        and aspect_violation <= 0.05
    )
    return (
        failed,
        1 - hard_ok,
        qi_raw,
        objective,
        qp_preseed,
        iota_violation + mirror_violation + elong_violation + 0.25 * aspect_violation,
        _float_value(row, "total_wall_time_s", np.inf),
    )


def _qs_selection_score(row: dict[str, str]) -> tuple:
    """Rank QA/QH/QP rows by field quality before secondary penalties."""

    failed = 1 if _bool_value(row.get("crashed")) or not _bool_value(row.get("success")) else 0
    problem = row.get("problem", "")
    qs = _float_value(row, "qs_final", np.inf) or np.inf
    objective = _float_value(row, "objective_final", np.inf) or np.inf
    aspect = _float_value(row, "aspect_final", np.inf) or np.inf
    aspect_target = _target_aspect_for(problem)
    aspect_violation = abs(aspect - aspect_target) / max(aspect_target, 1.0e-12)
    iota_violation = 0.0
    if problem == "qa":
        iota = _float_value(row, "iota_final", np.nan)
        iota_violation = (
            abs((iota or 0.0) - TARGET_QA_IOTA) / TARGET_QA_IOTA if np.isfinite(iota or np.nan) else 1.0
        )
    elif problem in {"qh", "qp"}:
        iota = abs(_float_value(row, "iota_final", 0.0) or 0.0)
        iota_violation = max(0.0, TARGET_ABS_IOTA_MIN - iota) / TARGET_ABS_IOTA_MIN
    hard_ok = int(aspect_violation <= 0.10 and iota_violation <= 0.10)
    return (
        failed,
        1 - hard_ok,
        qs,
        objective,
        iota_violation,
        aspect_violation,
        _float_value(row, "total_wall_time_s", np.inf),
    )


def _is_current_qs_row(row: dict[str, str]) -> bool:
    problem = row.get("problem", "")
    input_name = Path(row.get("input_file", "")).name
    if problem == "qh" and input_name != "input.nfp4_QH_warm_start":
        return False
    target_aspect = _float_value(row, "target_aspect", None)
    expected_aspect = _target_aspect_for(problem)
    if target_aspect is not None and abs(target_aspect - expected_aspect) > 1.0e-8:
        return False
    if problem in {"qh", "qp"}:
        target_floor = _float_value(row, "iota_abs_min", None)
        if target_floor is not None and abs(target_floor - TARGET_ABS_IOTA_MIN) > 1.0e-8:
            return False
    aspect = _float_value(row, "aspect_final", None)
    if aspect is not None and abs(aspect - expected_aspect) / expected_aspect > 0.25:
        return False
    return True


def _is_current_qi_row(row: dict[str, str]) -> bool:
    target_aspect = _float_value(row, "target_aspect", None)
    expected_aspect = _target_aspect_for("qi")
    if target_aspect is not None and abs(target_aspect - expected_aspect) > 1.0e-8:
        return False
    target_floor = _float_value(row, "iota_abs_min", None)
    if target_floor is not None and abs(target_floor - TARGET_ABS_IOTA_MIN) > 1.0e-8:
        return False
    aspect = _float_value(row, "aspect_final", None)
    if aspect is not None and abs(aspect - expected_aspect) / expected_aspect > 0.30:
        return False
    return True


def _run_from_row(row: dict[str, str]) -> BestRun:
    input_value = row.get("input_file")
    return BestRun(
        problem=row.get("problem", "qi"),
        backend=row.get("backend", "cpu"),
        policy=row["policy"],
        max_mode=int(row["max_mode"]),
        use_ess=_bool_value(row["use_ess"]),
        objective_final=float(row["objective_final"]),
        aspect_final=float(row["aspect_final"]),
        iota_final=float(_float_value(row, "iota_final", np.nan)),
        total_wall_time_s=float(row["total_wall_time_s"]),
        output_dir=_path_from_summary(row["output_dir"]),
        qi_qp_preseed=(
            _bool_value(row.get("qi_qp_preseed")) if row.get("problem") == "qi" and row.get("qi_qp_preseed") else None
        ),
        qi_raw_total=_float_value(row, "qi_raw_total"),
        qi_legacy_total=_qi_metric(row),
        qi_mirror_ratio_max=_float_value(row, "qi_mirror_ratio_max"),
        qi_max_elongation=_float_value(row, "qi_max_elongation"),
        lgradb_min=_float_value(row, "lgradb_min"),
        qi_lgradb_min=_float_value(row, "qi_lgradb_min"),
        input_file=_path_from_summary(input_value) if input_value else None,
    )


def _bundled_best_runs() -> list[BestRun]:
    """Return the checked-in README result bundle when available.

    Full optimization outputs are intentionally ignored by git.  The compact
    README panel is therefore driven by a small reviewed artifact bundle so a
    clean clone can regenerate the figures without re-running multi-minute
    optimizations.
    """

    if not README_CASES_SUMMARY.exists():
        return []
    with README_CASES_SUMMARY.open(newline="") as f:
        rows = list(csv.DictReader(f))
    runs = [_run_from_row(row) for row in rows]
    expected = set(PROBLEMS)
    found = {run.problem for run in runs}
    if found != expected:
        missing = ", ".join(sorted(expected - found))
        extra = ", ".join(sorted(found - expected))
        raise RuntimeError(
            f"{README_CASES_SUMMARY} must contain exactly {sorted(expected)}; "
            f"missing=[{missing}], extra=[{extra}]"
        )
    for run in runs:
        required = (
            run.output_dir / "history.json",
            run.output_dir / "wout_original.nc",
            run.output_dir / "wout_final.nc",
        )
        missing = [path for path in required if not path.exists()]
        if missing:
            joined = ", ".join(_repo_relative_path(path) for path in missing)
            raise FileNotFoundError(f"Incomplete README bundle for {run.problem}: {joined}")
    return sorted(runs, key=lambda run: PROBLEMS.index(run.problem))


def _qi_default_row_from_result_dir(result_dir: Path = QI_DEFAULT_RESULT_DIR) -> dict[str, str] | None:
    diagnostics_path = result_dir / "diagnostics.json"
    history_path = result_dir / "history.json"
    if not diagnostics_path.exists() or not history_path.exists():
        return None
    diagnostics = json.loads(diagnostics_path.read_text())
    history = json.loads(history_path.read_text())
    target_aspect = _float_value(diagnostics, "target_aspect", None)
    if target_aspect is not None and abs(target_aspect - _target_aspect_for("qi")) > 1.0e-8:
        return None
    return {
        "problem": "qi",
        "backend": "cpu",
        "policy": "qi_default",
        "max_mode": "3",
        "use_ess": "true",
        "success": "true",
        "crashed": "false",
        "target_aspect": f"{_target_aspect_for('qi'):.16e}",
        "iota_abs_min": f"{TARGET_ABS_IOTA_MIN:.16e}",
        "input_file": _repo_relative_path(REPO_ROOT / "examples" / "data" / "input.nfp2_QI"),
        "qi_qp_preseed": "false",
        "objective_final": f"{float(history['objective_final']):.16e}",
        "aspect_final": f"{float(diagnostics['aspect']):.16e}",
        "iota_final": f"{float(diagnostics['mean_iota']):.16e}",
        "total_wall_time_s": f"{float(history['total_wall_time_s']):.16e}",
        "output_dir": _repo_relative_path(result_dir),
        "qi_raw_total": f"{float(diagnostics['qi_raw_total']):.16e}",
        "qi_legacy_total": f"{float(diagnostics['qi_legacy_total']):.16e}",
        "qi_mirror_ratio_max": f"{float(diagnostics['qi_mirror_ratio_max']):.16e}",
        "qi_mirror_ratio_target": f"{float(diagnostics['qi_mirror_ratio_target']):.16e}",
        "qi_max_elongation": f"{float(diagnostics['qi_max_elongation']):.16e}",
        "qi_elongation_target": f"{float(diagnostics['qi_elongation_target']):.16e}",
    }


def _best_runs() -> list[BestRun]:
    bundled = _bundled_best_runs()
    if bundled:
        return bundled

    rows = [
        row
        for row in _read_summary_rows()
        if row.get("backend", "").lower() == "cpu"
        and not _bool_value(row.get("stellarator_asymmetric"))
        and _bool_value(row.get("success"))
        and not _bool_value(row.get("crashed"))
        and row.get("output_dir")
    ]
    best: list[BestRun] = []
    for problem in ("qa", "qh", "qp"):
        candidates = [row for row in rows if row.get("problem") == problem and _is_current_qs_row(row)]
        if not candidates:
            raise RuntimeError(f"No successful CPU symmetric rows found for {problem!r}")
        row = min(candidates, key=_qs_selection_score)
        best.append(_run_from_row(row))

    qi_rows = [
        row
        for row in _read_qi_constrained_rows()
        if row.get("output_dir")
        and (row.get("qi_legacy_total") not in (None, "") or row.get("qi_raw_total") not in (None, ""))
        and not _bool_value(row.get("stellarator_asymmetric"))
        and _is_current_qi_row(row)
    ]
    default_qi_row = _qi_default_row_from_result_dir()
    if default_qi_row is not None:
        qi_rows.append(default_qi_row)
    if qi_rows:
        row = min(qi_rows, key=_qi_selection_score)
        best.append(_run_from_row(row))
    else:
        candidates = [row for row in rows if row.get("problem") == "qi" and _is_current_qi_row(row)]
        if not candidates:
            raise RuntimeError("No successful symmetric rows found for 'qi'")
        row = min(candidates, key=lambda item: float(item["objective_final"]))
        best.append(_run_from_row(row))
    return best


def _lcfs_xyz(R: np.ndarray, Z: np.ndarray, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return R * np.cos(phi[None, :]), R * np.sin(phi[None, :]), Z


def _plot_lcfs(ax, wout, title: str) -> None:
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    theta, phi, R, Z, B = vmecplot2_lcfs_3d_grid(
        wout,
        s_index=int(wout.ns) - 1,
        ntheta=44,
        nzeta=max(80, 34 * int(wout.nfp)),
    )
    del theta
    X, Y, Zp = _lcfs_xyz(R, Z, phi)
    norm = Normalize(vmin=float(np.nanmin(B)), vmax=float(np.nanmax(B)))
    cmap = "viridis"
    colors = ScalarMappable(cmap=cmap, norm=norm).to_rgba(B)
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
    ax.set_title(title, fontsize=9, pad=4)
    ax.set_xlabel("X", labelpad=-8)
    ax.set_ylabel("Y", labelpad=-8)
    ax.set_zlabel("Z", labelpad=-8)
    ax.tick_params(axis="both", which="major", labelsize=7, pad=-2)
    fix_matplotlib_3d(ax)
    ax.figure.colorbar(
        ScalarMappable(cmap=cmap, norm=norm),
        ax=ax,
        fraction=0.035,
        pad=0.01,
        shrink=0.62,
        label="|B|",
    )


BOUNDARY_FAMILIES = ("RBC", "RBS", "ZBC", "ZBS")


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
    """Return WOUT boundary modes in the VMEC input phase convention."""

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
            "Use wout_original.nc or regenerate a raw-input initial wout."
        )


def _input_file_from_case_result(output_dir: Path) -> Path | None:
    result_path = output_dir / "case_result.json"
    if not result_path.exists():
        return None
    try:
        record = json.loads(result_path.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Cannot parse {result_path}") from exc
    value = record.get("input_file")
    if not value:
        return None
    return _path_from_summary(str(value))


def _raw_input_file_for_run(run: BestRun) -> Path:
    input_file = run.input_file or _input_file_from_case_result(run.output_dir)
    if input_file is None:
        raise RuntimeError(
            f"Cannot prove raw initial wout provenance for {run.output_dir}: "
            "missing input_file in summary row and case_result.json"
        )
    if not input_file.exists():
        raise FileNotFoundError(input_file)
    return input_file


def _validated_raw_initial_wout(candidate: Path, input_file: Path, *, context: str) -> Path:
    if not candidate.exists():
        raise FileNotFoundError(candidate)
    _assert_wout_matches_input_boundary(candidate, input_file, context=context)
    return candidate


def _derive_raw_initial_wout(input_file: Path, output_dir: Path) -> Path:
    """Solve the raw input deck once and persist it as ``wout_original.nc``."""

    from vmec_jax.driver import run_fixed_boundary, write_wout_from_fixed_boundary_run

    derived = output_dir / "wout_original.nc"
    derived.parent.mkdir(parents=True, exist_ok=True)
    run = run_fixed_boundary(input_file, verbose=False)
    write_wout_from_fixed_boundary_run(str(derived), run, include_fsq=False, fast_bcovar=True)
    _assert_wout_matches_input_boundary(derived, input_file, context="derived wout_original")
    return derived


def _validated_or_derived_raw_initial_wout(candidate: Path, input_file: Path, output_dir: Path, *, context: str) -> Path:
    try:
        return _validated_raw_initial_wout(candidate, input_file, context=context)
    except (FileNotFoundError, RuntimeError):
        return _derive_raw_initial_wout(input_file, output_dir)


def _preoptimization_wout_path(run: BestRun) -> Path:
    """Return the deck state before any mode-1 optimization work.

    Continuation case directories store ``wout_initial.nc`` for the final
    continuation stage, not the original deck.  For README panels we want the
    actual user-provided starting equilibrium before the first mode-1 solve.
    Never use a stage-local ``wout_initial.nc`` unless its boundary matches the
    raw VMEC input deck.
    """

    original = run.output_dir / "wout_original.nc"
    input_file = _raw_input_file_for_run(run)
    if original.exists():
        return _validated_raw_initial_wout(original, input_file, context=f"{run.problem} wout_original")
    if run.policy == "continuation":
        mode1 = run.output_dir.parent.parent / "mode1" / run.output_dir.name / "wout_initial.nc"
        if mode1.exists():
            return _validated_or_derived_raw_initial_wout(
                mode1,
                input_file,
                run.output_dir,
                context=f"{run.problem} continuation mode1 wout_initial",
            )
    return _validated_or_derived_raw_initial_wout(
        run.output_dir / "wout_initial.nc",
        input_file,
        run.output_dir,
        context=f"{run.problem} {run.policy} wout_initial",
    )


def _plot_history(ax, run: BestRun) -> None:
    with (run.output_dir / "history.json").open() as f:
        data = json.load(f)
    history = data.get("history", [])
    if not history:
        raise RuntimeError(f"Missing history entries in {run.output_dir / 'history.json'}")
    last_wall = None
    last_objective = None
    segments = _history_stage_segments(history)
    if run.problem == "qi" and run.qi_qp_preseed:
        qi_segments = [
            segment
            for segment in segments
            if segment and str(segment[0].get("stage", "")).startswith("QI ")
        ]
        segments = qi_segments or segments
    for segment in segments:
        wall_min = np.asarray([float(item.get("wall_time_s", 0.0)) / 60.0 for item in segment])
        objective = np.minimum.accumulate(
            np.asarray([max(float(item.get("objective", item.get("cost", np.nan))), 1.0e-16) for item in segment])
        )
        ax.plot(wall_min, objective, color="#1f4e79", linewidth=1.8)
        last_wall = wall_min[-1]
        last_objective = objective[-1]
    if last_wall is not None and last_objective is not None:
        ax.scatter(last_wall, last_objective, s=18, color="#d95f02", zorder=3)
    for boundary in data.get("stage_boundaries", []) or []:
        try:
            idx = int(boundary)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(history):
            ax.axvline(float(history[idx].get("wall_time_s", 0.0)) / 60.0, color="0.65", linestyle=":", linewidth=0.9)
    ax.set_yscale("log")
    ax.set_xlabel("Wall time (min)")
    ax.set_ylabel("Total objective")
    title = "QI refinement objective" if run.problem == "qi" and run.qi_qp_preseed else "Objective history"
    ax.set_title(title, fontsize=9, pad=4)
    ax.grid(True, alpha=0.22, linestyle=":")


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


def _booz_xform_on_outer_surface(wout_path: Path):
    try:
        from booz_xform_jax import Booz_xform
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "render_readme_best_optimizations.py requires booz_xform_jax. "
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


def _plot_boozer_bmag(ax, wout_path: Path, title: str) -> None:
    bx = _booz_xform_on_outer_surface(wout_path)
    theta, phi, B = _booz_bmag_grid(bx)
    PHI, THETA = np.meshgrid(phi, theta)
    vmin = float(np.nanmin(B))
    vmax = float(np.nanmax(B))
    if vmax <= vmin:
        pad = max(abs(vmin), 1.0) * 1e-12
        vmin -= pad
        vmax += pad
    levels = np.linspace(vmin, vmax, 24)
    cs = ax.contour(PHI, THETA, B, levels=levels, cmap="viridis", linewidths=0.9)
    ax.set_title(title, fontsize=9, pad=4)
    ax.set_xlabel(r"Boozer $\phi_B$ (one field period)")
    ax.set_ylabel(r"Boozer $\theta_B$")
    ax.set_xlim(0.0, 2.0 * np.pi / float(bx.nfp))
    ax.set_ylim(0.0, 2.0 * np.pi)
    ax.set_yticks([0, np.pi, 2 * np.pi])
    ax.set_yticklabels(["0", r"$\pi$", r"$2\pi$"])
    ax.grid(True, alpha=0.18, linestyle=":")
    ax.figure.colorbar(cs, ax=ax, fraction=0.046, pad=0.018, label="|B|")


def _write_readme_summary(runs: list[BestRun]) -> None:
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(
            [
                "problem",
                "backend",
                "policy",
                "max_mode",
                "ess",
                "qi_qp_preseed",
                "objective_final",
                "qi_legacy_total",
                "qi_raw_total",
                "qi_mirror_ratio_max",
                "qi_max_elongation",
                "aspect_final",
                "iota_final",
                "wall_time_min",
                "output_dir",
                "input_file",
                "initial_wout",
                "final_wout",
            ]
        )
        for run in runs:
            initial_wout = _preoptimization_wout_path(run)
            final_wout = run.output_dir / "wout_final.nc"
            writer.writerow(
                [
                    run.problem,
                    run.backend,
                    run.policy,
                    run.max_mode,
                    "yes" if run.use_ess else "no",
                    "" if run.qi_qp_preseed is None else ("yes" if run.qi_qp_preseed else "no"),
                    f"{run.objective_final:.16e}",
                    "" if run.qi_legacy_total is None else f"{run.qi_legacy_total:.16e}",
                    "" if run.qi_raw_total is None else f"{run.qi_raw_total:.16e}",
                    "" if run.qi_mirror_ratio_max is None else f"{run.qi_mirror_ratio_max:.16e}",
                    "" if run.qi_max_elongation is None else f"{run.qi_max_elongation:.16e}",
                    f"{run.aspect_final:.16e}",
                    f"{run.iota_final:.16e}",
                    f"{run.total_wall_time_s / 60.0:.6f}",
                    _repo_relative_path(run.output_dir),
                    "" if run.input_file is None else _repo_relative_path(run.input_file),
                    _repo_relative_path(initial_wout),
                    _repo_relative_path(final_wout),
                ]
            )


def _run_title(run: BestRun) -> str:
    extras = ""
    if run.problem == "qi":
        qi_raw = (
            float("nan")
            if (run.qi_legacy_total is None and run.qi_raw_total is None)
            else (run.qi_legacy_total if run.qi_legacy_total is not None else run.qi_raw_total)
        )
        mirror = float("nan") if run.qi_mirror_ratio_max is None else run.qi_mirror_ratio_max
        elong = float("nan") if run.qi_max_elongation is None else run.qi_max_elongation
        extras = (
            f", QP preseed={'yes' if run.qi_qp_preseed else 'no'}, "
            f"QI={qi_raw:.2e}, "
            f"mirror={mirror:.2f}, "
            f"elong={elong:.1f}"
        )
    return (
        f"{PROBLEM_TITLES[run.problem]} best symmetric {run.backend.upper()} run: "
        f"{run.policy}, max_mode={run.max_mode}, "
        f"{'ESS' if run.use_ess else 'no ESS'}, "
        f"J={run.objective_final:.2e}, "
        f"A={run.aspect_final:.3f}, "
        f"iota={run.iota_final:.4f}, "
        f"{run.total_wall_time_s / 60.0:.1f} min"
        f"{extras}"
    )


def _render_single_run(run: BestRun, out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig = plt.figure(figsize=(22, 4.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 5, width_ratios=(1.05, 1.05, 1.0, 1.05, 1.05))
    ax0 = fig.add_subplot(gs[0, 0], projection="3d")
    ax1 = fig.add_subplot(gs[0, 1], projection="3d")
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[0, 3])
    ax4 = fig.add_subplot(gs[0, 4])

    initial_wout_path = _preoptimization_wout_path(run)
    final_wout_path = run.output_dir / "wout_final.nc"
    wout_initial = read_wout(initial_wout_path)
    wout_final = read_wout(final_wout_path)
    _plot_lcfs(ax0, wout_initial, "Initial deck LCFS")
    _plot_lcfs(ax1, wout_final, "Final LCFS")
    _plot_history(ax2, run)
    _plot_boozer_bmag(ax3, initial_wout_path, r"Initial $|B|(\theta_B,\phi_B)$")
    _plot_boozer_bmag(ax4, final_wout_path, r"Final $|B|(\theta_B,\phi_B)$")
    fig.suptitle(_run_title(run), fontsize=13, x=0.01, y=1.02, ha="left")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    prepare_matplotlib_3d()
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    )

    runs = _best_runs()
    _write_readme_summary(runs)
    for run in runs:
        out_png = FIGURE_DIR / f"readme_best_optimization_{run.problem}.png"
        _render_single_run(run, out_png)
        print(f"Wrote {out_png}")
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
