#!/usr/bin/env python
"""Summarize square-coil free-boundary backend profile JSON files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vmec_jax.vmec2000_exec import _parse_vmec2000_threed1
from vmec_jax.solvers.free_boundary.validation import free_boundary_promotion_status
from vmec_jax.toroidal_hybrid import (
    square_axis_resolution_deck_status,
    square_axis_stellarator_mirror_hybrid_projection_error,
)


DEFAULT_GLOB = "results/square_coil_freeb_backend_profile_*/square_coil_free_boundary_backend_profile.json"
FINAL_REPORT_NAME = "square_coil_free_boundary_backend_profile.json"
PARTIAL_VMEC2000_NAME = "_partial_vmec2000_payload.json"
LAUNCHER_LOG_NAME = "launcher.log"
CASE_PREFIXES = (
    "square_coil_freeb_backend_profile_",
    "square_coil_direct_gpu_",
    "square_coil_",
)
FORCE_COMPONENTS = ("fsqr", "fsqz", "fsql")
INFERRED_SQUARE_AXIS_PROJECTION_GATE = 5.0e-12


def _ceil_tail_iteration_estimate(value: float) -> int | None:
    """Ceil a positive iteration estimate without platform-dependent off-by-one noise."""

    try:
        estimate = float(value)
    except Exception:
        return None
    if not np.isfinite(estimate):
        return None
    nearest = round(estimate)
    if abs(estimate - nearest) <= 1.0e-10 * max(1.0, abs(estimate)):
        return max(0, int(nearest))
    return max(0, int(np.ceil(estimate)))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Profile JSON files or directories. Empty defaults to the standard results glob.",
    )
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV output path.")
    parser.add_argument("--markdown", action="store_true", help="Print a Markdown table instead of TSV.")
    return parser


def _profile_paths(paths: list[Path]) -> list[Path]:
    if not paths:
        return sorted(Path(".").glob(DEFAULT_GLOB))
    out: list[Path] = []
    for path in paths:
        if path.is_dir():
            candidate = path / FINAL_REPORT_NAME
            if candidate.exists():
                out.append(candidate)
                continue
            partial = path / PARTIAL_VMEC2000_NAME
            if partial.exists():
                out.append(partial)
                continue
            threed1_matches = sorted((path / "vmec2000_mgrid").glob("threed1*"))
            if threed1_matches:
                out.append(threed1_matches[0])
                continue
            launcher_log = path / LAUNCHER_LOG_NAME
            if launcher_log.exists():
                out.append(launcher_log)
                continue
            out.extend(sorted(path.glob(f"**/{FINAL_REPORT_NAME}")))
        elif path.exists():
            out.append(path)
    return sorted(dict.fromkeys(out))


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if out == out and abs(out) != float("inf") else None


def _as_bool_list(value: Any) -> list[bool]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [bool(item) for item in value]
    return [bool(value)]


def _bool_last(value: Any) -> bool | None:
    values = _as_bool_list(value)
    return values[-1] if values else None


def _bool_any(value: Any) -> bool | None:
    values = _as_bool_list(value)
    return any(values) if values else None


def _last_sequence_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return value[-1] if value else None
    return value


def _vmec2000_total(backend: dict[str, Any]) -> tuple[float | None, float | None, int | None]:
    last = backend.get("last_row")
    last_total = None
    last_iter = None
    if isinstance(last, dict):
        last_total = _finite_float(last.get("total"))
        if last_total is None:
            parts = [_finite_float(last.get(key)) for key in ("fsqr", "fsqz", "fsql")]
            if all(value is not None for value in parts):
                last_total = float(sum(parts))  # type: ignore[arg-type]
        try:
            last_iter = int(last.get("it"))
        except Exception:
            last_iter = None
    return last_total, _finite_float(backend.get("min_total")), last_iter


def _file_status_payload(path: Path | None, *, prefix: str) -> dict[str, Any]:
    if path is None or not Path(path).exists():
        return {f"{prefix}_size_bytes": None, f"{prefix}_mtime_unix_s": None}
    stat = Path(path).stat()
    return {f"{prefix}_size_bytes": int(stat.st_size), f"{prefix}_mtime_unix_s": float(stat.st_mtime)}


def _vmec2000_final_max_component(backend: dict[str, Any]) -> float | None:
    last = backend.get("last_row")
    if not isinstance(last, dict):
        return None
    parts = [_finite_float(last.get(key)) for key in ("fsqr", "fsqz", "fsql")]
    if not all(value is not None for value in parts):
        return None
    return float(max(parts))  # type: ignore[arg-type]


def _jax_total(backend: dict[str, Any]) -> tuple[float | None, float | None, int | None]:
    last_total = _finite_float(backend.get("final_fsq_component_sum"))
    best_total = _finite_float(backend.get("best_scored_fsq"))
    try:
        last_iter = int(backend.get("n_iter"))
    except Exception:
        last_iter = None
    return last_total, best_total, last_iter


def _jax_final_max_component(backend: dict[str, Any]) -> float | None:
    parts = [_finite_float(backend.get(key)) for key in ("final_fsqr", "final_fsqz", "final_fsql")]
    if not all(value is not None for value in parts):
        return None
    return float(max(parts))  # type: ignore[arg-type]


def _strict_components_met(final_max_component: float | None, requested_ftol: float | None) -> bool | None:
    if final_max_component is None or requested_ftol is None:
        return None
    return bool(float(final_max_component) <= float(requested_ftol))


def _strict_gap(final_max_component: float | None, requested_ftol: float | None) -> float | None:
    if final_max_component is None or requested_ftol is None:
        return None
    requested = float(requested_ftol)
    if not np.isfinite(requested) or requested <= 0.0:
        return None
    return float(float(final_max_component) / requested)


def _remaining_iterations(max_iter: Any, final_iter: Any) -> int | None:
    try:
        max_iter_i = int(max_iter)
        final_iter_i = int(final_iter)
    except Exception:
        return None
    if max_iter_i < 0 or final_iter_i < 0:
        return None
    return max(0, max_iter_i - final_iter_i)


def _recommended_next_action(
    *,
    status: Any,
    progress_phase: Any,
    force_rows_started: Any,
    strict_components_met: bool | None,
    tail_plateau_status: Any,
    iters_to_target: float | None,
    remaining_iterations: int | None,
    vacuum_grid_exceeded_count: Any,
    strict_tail_projection_status: Any = None,
    strict_tail_iters_to_target: float | None = None,
) -> str:
    """Classify the next convergence action from compact summary evidence."""

    if strict_components_met is True:
        return "strict_converged"
    if force_rows_started is False or progress_phase in {
        "startup_or_pre_iteration_output",
        "axis_repair_or_pre_iteration_output",
        "waiting_for_threed1",
    }:
        return "wait_for_force_rows"
    try:
        grid_exceeded = int(vacuum_grid_exceeded_count or 0)
    except Exception:
        grid_exceeded = 0
    if grid_exceeded > 0:
        return "widen_mgrid_before_interpreting_residual"

    tail_status = "" if tail_plateau_status is None else str(tail_plateau_status)
    strict_tail_status = "" if strict_tail_projection_status is None else str(strict_tail_projection_status)
    running = str(status or "").startswith("running")
    if strict_tail_status in {"flat_or_growing_above_target", "weak_or_oscillatory_above_target"}:
        return (
            "let_current_run_finish_then_scan_delt_or_control_basis"
            if running
            else "scan_delt_or_promote_native_spline_controls"
        )
    if strict_tail_status == "projected_to_target":
        estimate = strict_tail_iters_to_target
        if estimate is not None:
            try:
                estimate = float(estimate)
            except Exception:
                estimate = np.nan
            if np.isfinite(estimate):
                if remaining_iterations is not None and estimate <= float(remaining_iterations):
                    return "continue_current_schedule"
                return "increase_final_stage_budget_or_change_schedule"
    if tail_status == "flat_above_stage_ftol":
        return "let_current_run_finish_then_scan_delt_or_stage_budget" if running else "scan_delt_or_stage_budget"
    if tail_status == "oscillatory":
        return "scan_delt_stage_budget_or_pressure_acceleration"
    if iters_to_target is not None:
        try:
            estimate = float(iters_to_target)
        except Exception:
            estimate = np.nan
        if np.isfinite(estimate):
            if remaining_iterations is not None and estimate <= float(remaining_iterations):
                return "continue_current_schedule"
            return "increase_final_stage_budget_or_change_schedule"
    if tail_status == "monotone_decreasing":
        return "continue_or_extend_if_budget_exhausts"
    return "inspect_tail_and_solver_diagnostics"


def _case_name(path: Path) -> str:
    for parent in (path.parent, path.parent.parent):
        name = parent.name
        for prefix in CASE_PREFIXES:
            if name.startswith(prefix):
                return name[len(prefix) :]
    if path.name == LAUNCHER_LOG_NAME and path.parent.name not in {"", "."}:
        return path.parent.name
    if path.name != FINAL_REPORT_NAME:
        return path.stem
    return path.parent.name


def _config_from_case_name(case: str) -> dict[str, Any]:
    """Infer resolution hints from standard profile directory names."""

    cfg: dict[str, Any] = {}
    for key in ("mpol", "ntor", "nzeta"):
        match = re.search(rf"(?:^|_){key}(\d+)(?:_|$)", case)
        if match is not None:
            cfg[key] = int(match.group(1))
    ns_match = re.search(r"(?:^|_)ns((?:\d+_)*\d+)(?:_mpol|_ntor|_nzeta|_|$)", case)
    if ns_match is not None:
        ns_values = [int(tok) for tok in ns_match.group(1).split("_") if tok]
        if ns_values:
            cfg["ns"] = ns_values[-1]
    niter_match = re.search(r"(?:^|_)niter(\d+)(k?)(?:_|$)", case)
    if niter_match is not None:
        scale = 1000 if niter_match.group(2) == "k" else 1
        cfg["max_iter"] = int(niter_match.group(1)) * scale
    mgrid_match = re.search(r"(?:^|_)mgrid\d+x\d+x(\d+)(?:_|$)", case)
    if mgrid_match is not None:
        cfg["mgrid_nphi"] = int(mgrid_match.group(1))
    if "control_spline" in case:
        cfg["axis_kind"] = "control_spline"
    elif "superellipse" in case:
        cfg["axis_kind"] = "superellipse"
    elif "spline" in case:
        cfg["axis_kind"] = "spline"
    return cfg


_SQUARE_BUILD_RE = re.compile(
    r"building square-coil configuration beta=(?P<beta>[+\-0-9.Ee]+)%,\s*"
    r"mpol=(?P<mpol>\d+),\s*ntor=(?P<ntor>\d+),\s*ns=(?P<ns>\[[^\]]+\]|\d+),\s*"
    r"nzeta=(?P<nzeta>\d+),\s*side_power=(?P<side>[+\-0-9.Ee]+),\s*"
    r"corner_power=(?P<corner>[+\-0-9.Ee]+)"
)


def _parse_square_build_config(text: str) -> dict[str, Any]:
    """Extract square-coil profile settings from a live launcher log."""

    match = _SQUARE_BUILD_RE.search(text)
    if match is None:
        return {}
    ns_text = match.group("ns")
    ns_values = [int(value) for value in re.findall(r"\d+", ns_text)]
    return {
        "beta_percent": float(match.group("beta")),
        "mpol": int(match.group("mpol")),
        "ntor": int(match.group("ntor")),
        "ns": ns_values[-1] if ns_values else None,
        "nzeta": int(match.group("nzeta")),
        "side_power": float(match.group("side")),
        "corner_power": float(match.group("corner")),
    }


def _infer_square_axis_resolution(
    cfg: dict[str, Any],
    *,
    case: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Infer cheap square-axis projection gates for live/partial rows.

    Completed profile JSON files carry exact ``boundary_projection`` and
    ``resolution_deck`` blocks.  Launcher-log and partial-VMEC2000 summaries do
    not, so this helper reconstructs the standard square-axis preflight from
    the encoded case/config values and marks the result as inferred.
    """

    required = ("mpol", "ntor", "nzeta")
    if any(cfg.get(key) is None for key in required):
        return {}, {}
    axis_kind = str(cfg.get("axis_kind") or "control_spline").strip().lower()
    if axis_kind not in {"control_spline", "spline", "superellipse"}:
        return {}, {}
    try:
        mpol = int(cfg["mpol"])
        ntor = int(cfg["ntor"])
        nzeta = int(cfg["nzeta"])
        ns = None if cfg.get("ns") is None else int(cfg["ns"])
        projection = square_axis_stellarator_mirror_hybrid_projection_error(
            nfp=1,
            mpol=mpol,
            ntor=ntor,
            ntheta_fit=max(64, 4 * mpol),
            nzeta_fit=max(128, 8 * ntor),
            ns_array=ns if ns is not None else 17,
            niter_array=1,
            ftol_array=1.0e-12,
            phiedge=-0.04,
            axis_half_width=1.5,
            axis_kind=axis_kind,
            axis_square_power=3.0,
            axis_spline_corner_radius_factor=1.14,
            minor_radius=0.03,
            side_elongation=0.08,
            side_minor_modulation=0.08,
            side_power=float(cfg.get("side_power", 1.0)),
            corner_power=float(cfg.get("corner_power", 1.0)),
            corner_ellipticity=0.04,
            corner_amplitude=0.004,
            corner_rotation=0.30,
            corner_helicity=1,
        )
        deck = square_axis_resolution_deck_status(
            projection=projection,
            mpol=mpol,
            ntor=ntor,
            nzeta=nzeta,
            ns=ns,
            mgrid_nphi=cfg.get("mgrid_nphi"),
            target_max_component_error=INFERRED_SQUARE_AXIS_PROJECTION_GATE,
        )
    except Exception:
        return {}, {}
    projection = {
        **projection,
        "inferred": True,
        "inferred_from": "case_or_launcher_log",
    }
    deck = {
        **deck,
        "inferred": True,
        "inferred_from": "case_or_launcher_log",
    }
    return projection, deck


def _stat(backend: dict[str, Any], history_key: str, stat_key: str) -> float | None:
    history = backend.get("history")
    if not isinstance(history, dict):
        return None
    stats = history.get(history_key)
    if not isinstance(stats, dict):
        return None
    return _finite_float(stats.get(stat_key))


def _history_counts(backend: dict[str, Any], history_key: str) -> dict[str, int]:
    history = backend.get("history")
    if not isinstance(history, dict):
        return {}
    counts = history.get(history_key)
    if not isinstance(counts, dict):
        return {}
    out: dict[str, int] = {}
    for key, value in counts.items():
        try:
            out[str(key)] = int(value)
        except Exception:
            continue
    return out


def _counts_text(counts: dict[str, int]) -> str | None:
    if not counts:
        return None
    return ",".join(f"{key}:{counts[key]}" for key in sorted(counts))


def _count_sum(counts: dict[str, int], *keys: str, prefix: str | None = None) -> int | None:
    if not counts:
        return None
    total = 0
    for key, value in counts.items():
        if key in keys or (prefix is not None and key.startswith(prefix)):
            total += int(value)
    return total


def _tail_projection(backend: dict[str, Any], key: str, *, target: float | None = None) -> float | None:
    history = backend.get("history")
    if not isinstance(history, dict):
        return None
    projection = history.get("fsq_component_sum_tail_projection")
    if not isinstance(projection, dict):
        return None
    if target is None:
        return _finite_float(projection.get(key))
    estimates = projection.get("estimated_additional_iterations_to_target")
    if not isinstance(estimates, dict):
        return None
    return _finite_float(estimates.get(f"{float(target):.0e}"))


def _component_tail_projection(
    backend: dict[str, Any],
    component: str,
    key: str,
    *,
    target: float | None = None,
) -> float | None:
    history = backend.get("history")
    if not isinstance(history, dict):
        return None
    by_component = history.get("fsq_component_tail_projection_by_component")
    if not isinstance(by_component, dict):
        return None
    payload = by_component.get(component)
    if not isinstance(payload, dict):
        return None
    if target is None:
        return _finite_float(payload.get(key))
    estimates = payload.get("estimated_additional_iterations_to_target")
    if not isinstance(estimates, dict):
        return None
    return _finite_float(estimates.get(f"{float(target):.0e}"))


def _tail_projection_payload(backend: dict[str, Any], *, component: str | None = None) -> dict[str, Any]:
    history = backend.get("history")
    if not isinstance(history, dict):
        return {}
    if component is None:
        projection = history.get("fsq_component_sum_tail_projection")
    else:
        by_component = history.get("fsq_component_tail_projection_by_component")
        projection = by_component.get(component) if isinstance(by_component, dict) else None
    return projection if isinstance(projection, dict) else {}


def _tail_projection_estimate(
    projection: dict[str, Any],
    *,
    target: float = 1.0e-12,
) -> float | None:
    estimates = projection.get("estimated_additional_iterations_to_target")
    if not isinstance(estimates, dict):
        return None
    return _finite_float(estimates.get(f"{float(target):.0e}"))


def _tail_projection_status(
    projection: dict[str, Any],
    *,
    target: float = 1.0e-12,
) -> str:
    """Classify whether a residual tail can plausibly reach the strict target."""

    last = _finite_float(projection.get("last"))
    if last is None:
        return "missing_tail"
    target_f = float(target)
    if last <= target_f:
        return "target_met"
    try:
        window = int(projection.get("window") or 0)
    except Exception:
        window = 0
    if window < 2:
        return "insufficient_tail"
    if _tail_projection_estimate(projection, target=target_f) is not None:
        return "projected_to_target"
    factor = _finite_float(projection.get("per_iter_factor"))
    if factor is not None and float(factor) >= 1.0:
        return "flat_or_growing_above_target"
    monotone_fraction = _finite_float(projection.get("monotone_decrease_fraction"))
    if monotone_fraction is not None and float(monotone_fraction) <= 0.25:
        return "weak_or_oscillatory_above_target"
    return "unprojected_above_target"


def _strict_tail_projection_payload(
    backend: dict[str, Any],
    *,
    limiting_component: str | None,
    target: float = 1.0e-12,
) -> dict[str, Any]:
    """Return compact component-wise strict-tail evidence."""

    history = backend.get("history")
    if isinstance(history, dict) and isinstance(history.get("strict_tail_projection_status"), str):
        status_by_component = history.get("strict_tail_component_status_by_component")
        limiting = history.get("strict_tail_limiting_component") or limiting_component
        return {
            "status": history.get("strict_tail_projection_status"),
            "target": _finite_float(history.get("strict_tail_projection_target")) or float(target),
            "component_statuses": status_by_component if isinstance(status_by_component, dict) else {},
            "limiting_component": limiting,
            "limiting_component_status": history.get("strict_tail_limiting_component_status"),
            "limiting_component_factor": _finite_float(
                history.get("strict_tail_limiting_component_factor")
            ),
            "limiting_component_iters_to_target": _finite_float(
                history.get("strict_tail_limiting_component_estimated_additional_iterations")
            ),
        }

    statuses: dict[str, str] = {}
    for component in FORCE_COMPONENTS:
        projection = _tail_projection_payload(backend, component=component)
        if projection:
            statuses[component] = _tail_projection_status(projection, target=target)
    limiting = limiting_component if limiting_component in FORCE_COMPONENTS else None
    if limiting is None and statuses:
        limiting = next(iter(statuses))
    limiting_projection = _tail_projection_payload(backend, component=limiting) if limiting else {}
    return {
        "status": statuses.get(limiting or "", "missing_tail"),
        "target": float(target),
        "component_statuses": statuses,
        "limiting_component": limiting,
        "limiting_component_status": statuses.get(limiting or "", "missing_tail"),
        "limiting_component_factor": _finite_float(limiting_projection.get("per_iter_factor")),
        "limiting_component_iters_to_target": _tail_projection_estimate(
            limiting_projection, target=target
        ),
    }


def _final_force_components(backend: dict[str, Any]) -> dict[str, float | None]:
    out = {name: _finite_float(backend.get(f"final_{name}")) for name in FORCE_COMPONENTS}
    last = backend.get("last_row")
    if isinstance(last, dict):
        for name in FORCE_COMPONENTS:
            if out[name] is None:
                out[name] = _finite_float(last.get(name))
    return out


def _strict_gap_for_component(value: float | None, requested_ftol: float | None) -> float | None:
    if value is None or requested_ftol is None:
        return None
    requested = float(requested_ftol)
    if not np.isfinite(requested) or requested <= 0.0:
        return None
    return float(float(value) / requested)


def _limiting_component(backend: dict[str, Any], components: dict[str, float | None]) -> str | None:
    history = backend.get("history")
    if isinstance(history, dict):
        value = history.get("fsq_limiting_component")
        if isinstance(value, str) and value in FORCE_COMPONENTS:
            return value
    finite = {name: value for name, value in components.items() if value is not None and np.isfinite(value)}
    if not finite:
        return None
    return max(finite, key=lambda name: float(finite[name]))


def _last_stage_value(backend: dict[str, Any], key: str) -> Any:
    stages = backend.get("stage_summaries")
    if not isinstance(stages, list) or not stages:
        return None
    last = stages[-1]
    if not isinstance(last, dict):
        return None
    return last.get(key)


def _stage_schedule_payload(backend: dict[str, Any], *, final_iter: int | None) -> dict[str, Any]:
    """Return compact multigrid schedule fields for live and final summaries."""

    stages = backend.get("stage_summaries")
    if not isinstance(stages, list) or not stages:
        return {
            "stage_count": None,
            "stage_ns_array": None,
            "stage_niter_array": None,
            "stage_ftol_array": None,
            "stage_budget_total": None,
            "stage_budget_final": None,
            "current_stage_index": None,
            "current_stage_niter": None,
            "current_stage_ftol": None,
            "current_stage_last_iter": None,
            "current_stage_iteration_row_count": None,
            "remaining_stage_budget": None,
            "remaining_total_stage_budget": None,
        }

    ns_values: list[int] = []
    niter_values: list[int] = []
    ftol_values: list[float] = []
    row_counts: list[int] = []
    current_stage_index: int | None = None
    for idx, stage in enumerate(stages):
        if not isinstance(stage, dict):
            continue
        try:
            ns_values.append(int(stage.get("ns")))
        except Exception:
            pass
        try:
            niter_values.append(int(stage.get("niter")))
        except Exception:
            pass
        ftol = _finite_float(stage.get("ftolv"))
        if ftol is not None:
            ftol_values.append(float(ftol))
        try:
            row_count = int(stage.get("iteration_row_count") or 0)
        except Exception:
            row_count = 0
        row_counts.append(row_count)
        if row_count > 0:
            current_stage_index = idx

    last_row = backend.get("last_row")
    if isinstance(last_row, dict):
        try:
            current_stage_index = int(last_row.get("stage_index"))
        except Exception:
            pass
    if current_stage_index is None and stages:
        current_stage_index = len(stages) - 1

    def _stage_item(key: str, index: int | None) -> Any:
        if index is None or not (0 <= int(index) < len(stages)):
            return None
        stage = stages[int(index)]
        return stage.get(key) if isinstance(stage, dict) else None

    current_niter = None
    try:
        current_niter = int(_stage_item("niter", current_stage_index))
    except Exception:
        current_niter = None
    current_ftol = _finite_float(_stage_item("ftolv", current_stage_index))
    current_rows = None
    try:
        current_rows = int(_stage_item("iteration_row_count", current_stage_index) or 0)
    except Exception:
        current_rows = None
    current_last_iter = None
    current_stage = None
    if current_stage_index is not None and 0 <= int(current_stage_index) < len(stages):
        current_stage = stages[int(current_stage_index)]
    if isinstance(current_stage, dict):
        current_last_row = current_stage.get("last_row")
        if isinstance(current_last_row, dict):
            try:
                current_last_iter = int(current_last_row.get("it"))
            except Exception:
                current_last_iter = None
    if current_last_iter is None and isinstance(last_row, dict):
        try:
            current_last_iter = int(last_row.get("it"))
        except Exception:
            current_last_iter = None

    stage_budget_total = int(sum(niter_values)) if niter_values else None
    stage_budget_final = int(niter_values[-1]) if niter_values else None
    current_elapsed = current_last_iter if current_last_iter is not None else current_rows
    remaining_stage_budget = (
        None if current_niter is None or current_elapsed is None else max(0, current_niter - current_elapsed)
    )
    future_budget = None
    if current_stage_index is not None and niter_values:
        try:
            future_budget = int(sum(niter_values[int(current_stage_index) + 1 :]))
        except Exception:
            future_budget = None
    remaining_total_stage_budget = (
        None
        if remaining_stage_budget is None or future_budget is None
        else int(remaining_stage_budget) + int(future_budget)
    )

    return {
        "stage_count": int(len(stages)),
        "stage_ns_array": ",".join(str(value) for value in ns_values) if ns_values else None,
        "stage_niter_array": ",".join(str(value) for value in niter_values) if niter_values else None,
        "stage_ftol_array": ",".join(f"{value:.0e}" for value in ftol_values) if ftol_values else None,
        "stage_budget_total": stage_budget_total,
        "stage_budget_final": stage_budget_final,
        "current_stage_index": current_stage_index,
        "current_stage_niter": current_niter,
        "current_stage_ftol": current_ftol,
        "current_stage_last_iter": current_last_iter,
        "current_stage_iteration_row_count": current_rows,
        "remaining_stage_budget": remaining_stage_budget,
        "remaining_total_stage_budget": remaining_total_stage_budget,
    }


def _virtual_casing_payload(backend: dict[str, Any]) -> dict[str, Any]:
    payload = backend.get("virtual_casing")
    return payload if isinstance(payload, dict) else {}


def _accepted_provider_parity_payload(backend: dict[str, Any]) -> dict[str, Any]:
    payload = backend.get("accepted_provider_parity")
    return payload if isinstance(payload, dict) else {"status": "not_run"}


def _solver_overrides_payload(backend: dict[str, Any]) -> dict[str, Any]:
    payload = backend.get("free_boundary_solver_overrides")
    return payload if isinstance(payload, dict) else {}


def _edge_projection_enabled_from_requested(value: Any) -> bool:
    return str(value or "").strip().lower() not in {"", "none", "off", "false", "no", "0"}


def _promotion_payload(
    *,
    backend_name: str,
    backend: dict[str, Any],
    cfg: dict[str, Any],
    strict_components_met: bool | None,
    virtual_casing: dict[str, Any],
) -> dict[str, Any]:
    payload = backend.get("free_boundary_promotion")
    if isinstance(payload, dict):
        return payload
    direct_backend = str(backend_name) in {"vmec_jax_direct", "vmec_jax_direct_live"}
    require_fresh = not str(backend_name).startswith("vmec2000")
    return free_boundary_promotion_status(
        beta_percent=cfg.get("beta_percent", 0.0),
        strict_components_met=strict_components_met,
        final_residual_recomputed=backend.get("final_residual_recomputed_on_accepted_state"),
        virtual_casing_status=virtual_casing.get("status"),
        virtual_casing_grid_adequacy_status=virtual_casing.get("grid_adequacy_status"),
        direct_coil_backend=direct_backend,
        require_fresh_residual=require_fresh,
    )


def _backend_role(backend_name: str) -> str:
    """Return the intended use of one profile backend."""

    key = str(backend_name).strip().lower()
    if key == "vmec2000_mgrid":
        return "vmec2000_mgrid_reference"
    if key in {"vmec_jax_mgrid", "vmec_jax_mgrid_live"}:
        return "vmec_jax_mgrid_parity"
    if key in {"vmec_jax_direct", "vmec_jax_direct_live"}:
        return "vmec_jax_direct_research"
    return "diagnostic_backend"


def _edge_force_capture_status(force_direction: dict[str, Any]) -> str:
    """Classify how well the reduced edge basis captures the force direction."""

    status = str(force_direction.get("status", "") or "")
    if status and status != "measured":
        return status
    captured = _finite_float(force_direction.get("captured_fraction"))
    rel = _finite_float(force_direction.get("residual_rel"))
    if captured is None and rel is None:
        return "not_measured"
    if (captured is not None and captured < 0.9) or (rel is not None and rel > 0.5):
        return "basis_underfit"
    if (captured is not None and captured >= 0.95) and (rel is None or rel <= 0.35):
        return "captured"
    return "marginal"


def _edge_force_capture_next_basis(
    *,
    edge_enabled: bool,
    basis: Any,
    capture_status: str,
) -> str | None:
    """Return the next reduced edge basis to try after a stalled profile."""

    if not bool(edge_enabled) or str(capture_status) != "basis_underfit":
        return None
    basis_key = str(basis or "").strip().lower()
    if basis_key in {"", "square"}:
        return "stellarator"
    if basis_key == "stellarator":
        return "full"
    return "native_spline_controls"


def _as_text_list(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        return [values] if values else []
    if isinstance(values, (list, tuple)):
        return [str(value) for value in values if str(value)]
    return [str(values)]


def _strict_evidence_payload(
    *,
    status: Any,
    backend_name: str,
    requested_ftol: float | None,
    strict_components_met: bool | None,
    strict_gap: float | None,
    next_action: str,
    production_candidate: Any,
    promotion_blockers: Any,
    resolution_deck: dict[str, Any],
    cfg: dict[str, Any],
    tail_plateau_status: Any,
) -> dict[str, Any]:
    """Classify whether one row is evidence for a strict production solve.

    VMEC's convergence contract is component-wise: fsqr, fsqz, and fsql must
    each be below FTOL.  This helper combines that force gate with the cheap
    geometry/grid gates so diagnostic rows are not confused with strict
    ``1e-12`` convergence evidence.
    """

    blockers: list[str] = []
    if requested_ftol is None:
        blockers.append("requested_ftol_unknown")
    elif float(requested_ftol) > 1.0e-12 * (1.0 + 1.0e-10):
        blockers.append("requested_ftol_above_1e-12")

    resolution_status = resolution_deck.get("status")
    if resolution_status in {"diagnostic_underresolved", "diagnostic_gate_disabled"}:
        blockers.append(f"resolution_deck_{resolution_status}")
    blockers.extend(f"resolution_{reason}" for reason in _as_text_list(resolution_deck.get("reasons")))
    if cfg.get("nzeta_underrecommended") is True:
        blockers.append("nzeta_below_square_axis_recommendation")
    if resolution_deck.get("mgrid_nphi_multiple_of_nzeta") is False:
        blockers.append("mgrid_nphi_not_multiple_of_nzeta")

    if strict_components_met is not True:
        blockers.append("strict_components_unknown" if strict_components_met is None else "strict_components_not_met")

    if production_candidate is not True:
        blockers.append("promotion_not_ready")
    blockers.extend(f"promotion_{reason}" for reason in _as_text_list(promotion_blockers))

    if str(status or "").startswith("running"):
        blockers.append("run_still_running")

    unique_blockers = list(dict.fromkeys(blockers))
    tail_status = "" if tail_plateau_status is None else str(tail_plateau_status)
    if not unique_blockers:
        evidence_status = "strict_production_evidence"
    elif "resolution_deck_diagnostic_underresolved" in unique_blockers or any(
        reason.startswith("resolution_") and reason != "resolution_deck_diagnostic_gate_disabled"
        for reason in unique_blockers
    ):
        evidence_status = "diagnostic_underresolved"
    elif "requested_ftol_above_1e-12" in unique_blockers:
        evidence_status = "non_strict_ftol"
    elif tail_status == "flat_above_stage_ftol":
        evidence_status = "stalled_above_strict_ftol"
    elif str(status or "").startswith("running") and next_action in {
        "continue_current_schedule",
        "continue_or_extend_if_budget_exhausts",
        "increase_final_stage_budget_or_change_schedule",
    }:
        evidence_status = "running_toward_strict"
    elif strict_gap is not None and np.isfinite(float(strict_gap)) and float(strict_gap) <= 100.0:
        evidence_status = "near_strict_not_met"
    elif strict_gap is not None and np.isfinite(float(strict_gap)) and float(strict_gap) > 100.0:
        evidence_status = "underconverged"
    else:
        evidence_status = "not_strict_evidence"

    return {
        "backend_role": _backend_role(str(backend_name)),
        "strict_evidence_status": evidence_status,
        "strict_evidence_blockers": ",".join(unique_blockers),
    }


def _recommended_followup_payload(
    *,
    backend_name: str,
    status: Any,
    next_action: str,
    strict_evidence_status: Any,
    accepted_provider_parity_status: Any,
    freeb_jax_nestor_operator: Any,
    freeb_edge_control_projection: dict[str, Any],
    freeb_edge_force_capture_status: str,
    freeb_edge_force_capture_next_basis: str | None,
    vacuum_grid_exceeded_count: Any,
) -> dict[str, str]:
    """Return the next profile kind that best matches the summary evidence."""

    evidence = "" if strict_evidence_status is None else str(strict_evidence_status)
    status_text = "" if status is None else str(status)
    next_action_text = "" if next_action is None else str(next_action)
    backend = str(backend_name)
    accepted_status = (
        "" if accepted_provider_parity_status is None else str(accepted_provider_parity_status)
    )
    try:
        grid_exceeded = int(vacuum_grid_exceeded_count or 0)
    except Exception:
        grid_exceeded = 0
    edge_enabled = bool(freeb_edge_control_projection.get("enabled"))
    edge_basis = str(freeb_edge_control_projection.get("basis_symmetry") or "").strip().lower()
    edge_update_mode = str(freeb_edge_control_projection.get("update_mode") or "").strip().lower()
    edge_capture = str(freeb_edge_force_capture_status or "")
    edge_next_basis = (
        None
        if freeb_edge_force_capture_next_basis is None
        else str(freeb_edge_force_capture_next_basis)
    )

    if evidence == "strict_production_evidence":
        return {
            "recommended_followup_profile_kind": "none",
            "recommended_followup_reason": "strict_evidence",
        }
    if status_text.startswith("running"):
        return {
            "recommended_followup_profile_kind": "wait_current_run",
            "recommended_followup_reason": "active_profile_still_running",
        }
    if grid_exceeded > 0:
        return {
            "recommended_followup_profile_kind": "vmec2000",
            "recommended_followup_reason": "widen_mgrid_before_backend_comparison",
        }
    if evidence == "diagnostic_underresolved":
        return {
            "recommended_followup_profile_kind": "resolution-preflight",
            "recommended_followup_reason": "fix_projection_nzeta_or_mgrid_gate_first",
        }
    if next_action_text in {
        "scan_delt_stage_budget_or_pressure_acceleration",
        "scan_delt_or_stage_budget",
        "let_current_run_finish_then_scan_delt_or_stage_budget",
        "scan_delt_or_promote_native_spline_controls",
        "let_current_run_finish_then_scan_delt_or_control_basis",
    }:
        if "direct" in backend:
            if not edge_enabled:
                kind = "direct-gpu-edge-polish"
            elif edge_capture == "basis_underfit" and edge_next_basis == "stellarator":
                return {
                    "recommended_followup_profile_kind": "direct-gpu-edge-stellarator-polish",
                    "recommended_followup_reason": "square_edge_basis_underfits_force_direction",
                }
            elif edge_capture == "basis_underfit" and edge_next_basis == "full":
                return {
                    "recommended_followup_profile_kind": "direct-gpu-edge-full-polish",
                    "recommended_followup_reason": "stellarator_edge_basis_underfits_force_direction",
                }
            elif edge_capture == "basis_underfit" and edge_next_basis == "native_spline_controls":
                if edge_update_mode != "native_coordinate":
                    if edge_basis == "full":
                        native_kind = "direct-gpu-edge-full-native-polish"
                    elif edge_basis == "stellarator":
                        native_kind = "direct-gpu-edge-stellarator-native-polish"
                    else:
                        native_kind = "direct-gpu-edge-native-polish"
                    return {
                        "recommended_followup_profile_kind": native_kind,
                        "recommended_followup_reason": (
                            f"{edge_basis or 'reduced'}_edge_basis_underfits_force_direction"
                        ),
                    }
                return {
                    "recommended_followup_profile_kind": "native-spline-control-prototype",
                    "recommended_followup_reason": (
                        f"{edge_basis or 'reduced'}_native_edge_update_still_underfits_force_direction"
                    ),
                }
            elif bool(freeb_jax_nestor_operator):
                if edge_update_mode != "native_coordinate":
                    return {
                        "recommended_followup_profile_kind": "direct-gpu-edge-stellarator-native-polish",
                        "recommended_followup_reason": (
                            "edge_control_and_jax_nestor_still_stalled_promote_native_spline_controls"
                        ),
                    }
                return {
                    "recommended_followup_profile_kind": "native-spline-control-prototype",
                    "recommended_followup_reason": (
                        "edge_control_and_jax_nestor_still_stalled_promote_native_spline_controls"
                    ),
                }
            elif edge_basis == "full":
                kind = "direct-gpu-edge-full-jax-nestor-polish"
            elif edge_basis == "stellarator":
                kind = "direct-gpu-edge-stellarator-jax-nestor-polish"
            else:
                kind = "direct-gpu-edge-jax-nestor-polish"
        elif backend == "vmec2000_mgrid":
            kind = "vmec2000"
        else:
            kind = "provider-parity"
        return {
            "recommended_followup_profile_kind": kind,
            "recommended_followup_reason": next_action_text,
        }
    if accepted_status not in {"completed", "not_applicable"} and backend in {
        "vmec_jax_direct",
        "vmec_jax_mgrid",
        "vmec_jax_direct_live",
        "vmec_jax_mgrid_live",
        "vmec2000_mgrid",
    }:
        return {
            "recommended_followup_profile_kind": "provider-parity",
            "recommended_followup_reason": "accepted_lcfs_provider_parity_missing",
        }
    if backend == "vmec2000_mgrid":
        return {
            "recommended_followup_profile_kind": "provider-parity",
            "recommended_followup_reason": "vmec2000_reference_not_strict",
        }
    return {
        "recommended_followup_profile_kind": "provider-parity",
        "recommended_followup_reason": next_action_text or "inspect_backend_comparison",
    }


def _control_projection_delta_text(payload: dict[str, Any]) -> str | None:
    values = payload.get("radius_delta_by_label")
    if isinstance(values, dict):
        items: list[str] = []
        for key, value in values.items():
            finite = _finite_float(value)
            if finite is not None:
                items.append(f"{key}:{finite:.6g}")
        return ",".join(items) if items else None
    values = payload.get("radius_delta")
    if isinstance(values, list):
        items = []
        for value in values:
            finite = _finite_float(value)
            if finite is not None:
                items.append(f"{finite:.6g}")
        return ",".join(items) if items else None
    return None


def _control_projection_candidate(payload: dict[str, Any], symmetry: str) -> dict[str, Any]:
    candidates = payload.get("candidate_bases")
    if not isinstance(candidates, dict):
        return {}
    candidate = candidates.get(symmetry)
    return candidate if isinstance(candidate, dict) else {}


def _vmec2000_tail_projection(rows: list[Any], *, length: int = 12) -> dict[str, Any]:
    """Estimate residual decay per VMEC2000 iteration from the current stage tail."""

    pairs: list[tuple[int, float]] = []
    last_it: int | None = None
    for row in reversed(list(rows)):
        try:
            if isinstance(row, dict):
                it = int(row.get("it"))
                total = _finite_float(row.get("total"))
                if total is None:
                    total = sum(float(row.get(key)) for key in ("fsqr", "fsqz", "fsql"))
            else:
                it = int(row.it)
                total = float(row.fsqr) + float(row.fsqz) + float(row.fsql)
        except Exception:
            continue
        if not np.isfinite(total) or total <= 0.0:
            continue
        if last_it is not None and it >= last_it:
            break
        pairs.append((it, total))
        last_it = it
        if len(pairs) >= int(length):
            break
    pairs.reverse()
    out: dict[str, Any] = {
        "window": len(pairs),
        "first_iter": None,
        "last_iter": None,
        "first": None,
        "last": None,
        "min": None,
        "max": None,
        "monotone_decrease_fraction": None,
        "per_iter_log_slope": None,
        "per_iter_factor": None,
        "estimated_additional_iterations_to_target": {},
    }
    if len(pairs) < 2:
        if pairs:
            out.update(
                {
                    "first_iter": int(pairs[0][0]),
                    "last_iter": int(pairs[-1][0]),
                    "first": float(pairs[0][1]),
                    "last": float(pairs[-1][1]),
                    "min": float(pairs[-1][1]),
                    "max": float(pairs[-1][1]),
                }
            )
        return out
    iters = np.asarray([pair[0] for pair in pairs], dtype=float)
    totals = np.asarray([pair[1] for pair in pairs], dtype=float)
    out.update(
        {
            "first_iter": int(iters[0]),
            "last_iter": int(iters[-1]),
            "first": float(totals[0]),
            "last": float(totals[-1]),
            "min": float(np.nanmin(totals)),
            "max": float(np.nanmax(totals)),
        }
    )
    diffs = np.diff(totals)
    out["monotone_decrease_fraction"] = float(np.mean(diffs < 0.0)) if diffs.size else None
    try:
        slope, _intercept = np.polyfit(iters, np.log(totals), 1)
    except Exception:
        return out
    if not np.isfinite(slope):
        return out
    out["per_iter_log_slope"] = float(slope)
    out["per_iter_factor"] = float(np.exp(slope))
    last = float(totals[-1])
    estimates: dict[str, int | None] = {}
    for target in (1.0e-12,):
        key = f"{target:.0e}"
        if last <= target:
            estimates[key] = 0
        elif slope < 0.0:
            estimates[key] = _ceil_tail_iteration_estimate(np.log(target / last) / slope)
        else:
            estimates[key] = None
    out["estimated_additional_iterations_to_target"] = estimates
    return out


def _vmec2000_component_tail_projections(rows: list[Any]) -> dict[str, Any]:
    """Return VMEC2000 residual-tail projections by force component."""

    return {
        component: _vmec2000_tail_projection(
            [
                {
                    "it": int(row.get("it") if isinstance(row, dict) else row.it),
                    "total": float(row.get(component) if isinstance(row, dict) else getattr(row, component)),
                    "fsqr": float(row.get(component) if isinstance(row, dict) else getattr(row, component)),
                    "fsqz": 0.0,
                    "fsql": 0.0,
                    "max_component": float(row.get(component) if isinstance(row, dict) else getattr(row, component)),
                }
                for row in rows
                if _finite_float(row.get(component) if isinstance(row, dict) else getattr(row, component, None))
                is not None
            ]
        )
        for component in FORCE_COMPONENTS
    }


def _row_total_and_max(row: Any) -> tuple[int, float, float] | None:
    try:
        if isinstance(row, dict):
            iteration = int(row.get("it"))
            total = _finite_float(row.get("total"))
            if total is None:
                parts = [_finite_float(row.get(key)) for key in ("fsqr", "fsqz", "fsql")]
                if not all(value is not None for value in parts):
                    return None
                total = float(sum(parts))  # type: ignore[arg-type]
            max_component = _finite_float(row.get("max_component"))
            if max_component is None:
                parts = [_finite_float(row.get(key)) for key in ("fsqr", "fsqz", "fsql")]
                if not all(value is not None for value in parts):
                    return None
                max_component = float(max(parts))  # type: ignore[arg-type]
        else:
            iteration = int(row.it)
            parts = [float(row.fsqr), float(row.fsqz), float(row.fsql)]
            total = float(sum(parts))
            max_component = float(max(parts))
    except Exception:
        return None
    if not (np.isfinite(total) and total > 0.0 and np.isfinite(max_component)):
        return None
    return iteration, float(total), float(max_component)


def _tail_plateau_payload(
    rows: list[Any],
    *,
    stage_ftol: float | None,
    length: int = 12,
    rel_span_tol: float = 0.02,
) -> dict[str, Any]:
    """Classify whether the recent residual tail is flat above tolerance."""

    parsed = [item for item in (_row_total_and_max(row) for row in list(rows)[-int(length) :]) if item]
    out: dict[str, Any] = {
        "window": int(len(parsed)),
        "status": "insufficient_tail",
        "stage_ftol": None if stage_ftol is None else float(stage_ftol),
        "total_rel_span": None,
        "total_last_over_min": None,
        "monotone_decrease_fraction": None,
    }
    if len(parsed) < 3:
        return out
    totals = np.asarray([item[1] for item in parsed], dtype=float)
    max_components = np.asarray([item[2] for item in parsed], dtype=float)
    diffs = np.diff(totals)
    total_min = float(np.min(totals))
    total_max = float(np.max(totals))
    total_last = float(totals[-1])
    rel_span = float((total_max - total_min) / max(total_min, np.finfo(float).tiny))
    stage = None if stage_ftol is None else float(stage_ftol)
    above_stage = bool(stage is not None and np.isfinite(stage) and total_last > stage)
    flat = bool(rel_span <= float(rel_span_tol))
    if flat and above_stage:
        status = "flat_above_stage_ftol"
    elif flat:
        status = "flat_near_stage_ftol"
    elif np.all(diffs < 0.0):
        status = "monotone_decreasing"
    else:
        status = "oscillatory"
    out.update(
        {
            "status": status,
            "first_iter": int(parsed[0][0]),
            "last_iter": int(parsed[-1][0]),
            "total_first": float(totals[0]),
            "total_last": total_last,
            "total_min": total_min,
            "total_max": total_max,
            "total_rel_span": rel_span,
            "total_last_over_min": float(total_last / max(total_min, np.finfo(float).tiny)),
            "max_component_last": float(max_components[-1]),
            "monotone_decrease_fraction": float(np.mean(diffs < 0.0)),
        }
    )
    return out


_VMEC_STYLE_STAGE_RE = re.compile(
    r"^\s*NS\s*=\s*(?P<ns>\d+)\s+NO\.\s+FOURIER\s+MODES\s*=\s*(?P<modes>\d+)"
    r"\s+FTOLV\s*=\s*(?P<ftolv>[+\-0-9.Ee]+)\s+NITER\s*=\s*(?P<niter>\d+)"
)
_VMEC_STYLE_ROW_RE = re.compile(
    r"^\s*(?P<it>\d+)\s+"
    r"(?P<fsqr>[+\-0-9.Ee]+)\s+"
    r"(?P<fsqz>[+\-0-9.Ee]+)\s+"
    r"(?P<fsql>[+\-0-9.Ee]+)\s+"
    r"(?P<raxis>[+\-0-9.Ee]+)\s+"
    r"(?P<delt>[+\-0-9.Ee]+)\s+"
    r"(?P<wmhd>[+\-0-9.Ee]+)"
)


def _vmec_style_row_payload(match: re.Match[str], *, stage_index: int | None) -> dict[str, Any]:
    fsqr = float(match.group("fsqr"))
    fsqz = float(match.group("fsqz"))
    fsql = float(match.group("fsql"))
    return {
        "it": int(match.group("it")),
        "stage_index": stage_index,
        "fsqr": fsqr,
        "fsqz": fsqz,
        "fsql": fsql,
        "total": float(fsqr + fsqz + fsql),
        "max_component": float(max(fsqr, fsqz, fsql)),
        "raxis": float(match.group("raxis")),
        "delt": float(match.group("delt")),
        "wmhd": float(match.group("wmhd")),
    }


def _vmec_style_log_payload(path: Path) -> dict[str, Any]:
    """Parse live vmec_jax verbose-solver rows from a launcher log."""

    stages: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    vacuum_turn_on_iter: int | None = None
    saw_axis_repair = False
    current_stage_index: int | None = None
    backend_name = "vmec_jax_live"
    for line in path.read_text(errors="replace").splitlines():
        if "running vmec_jax direct-coil backend" in line:
            backend_name = "vmec_jax_direct_live"
        elif "running vmec_jax generated-mgrid backend" in line:
            backend_name = "vmec_jax_mgrid_live"
        stage_match = _VMEC_STYLE_STAGE_RE.match(line)
        if stage_match is not None:
            current_stage_index = len(stages)
            stages.append(
                {
                    "ns": int(stage_match.group("ns")),
                    "mode_count": int(stage_match.group("modes")),
                    "ftolv": float(stage_match.group("ftolv")),
                    "niter": int(stage_match.group("niter")),
                    "iteration_row_count": 0,
                    "last_row": None,
                    "min_total": None,
                }
            )
            continue
        if "INITIAL JACOBIAN CHANGED SIGN" in line:
            saw_axis_repair = True
        turn_on_match = re.search(r"VACUUM PRESSURE TURNED ON AT\s+(\d+)\s+ITERATIONS", line)
        if turn_on_match is not None:
            vacuum_turn_on_iter = int(turn_on_match.group(1))
        row_match = _VMEC_STYLE_ROW_RE.match(line)
        if row_match is None:
            continue
        row = _vmec_style_row_payload(row_match, stage_index=current_stage_index)
        rows.append(row)
        if current_stage_index is not None and 0 <= current_stage_index < len(stages):
            stage = stages[current_stage_index]
            stage["iteration_row_count"] = int(stage["iteration_row_count"]) + 1
            stage["last_row"] = row
            min_total = _finite_float(stage.get("min_total"))
            stage["min_total"] = row["total"] if min_total is None else min(min_total, row["total"])

    last = rows[-1] if rows else None
    totals = [float(row["total"]) for row in rows]
    progress_phase = (
        "force_iterations"
        if rows
        else "axis_repair_or_pre_iteration_output"
        if saw_axis_repair
        else "startup_or_pre_iteration_output"
    )
    stage_ftol = _finite_float(stages[-1].get("ftolv")) if stages else None
    payload: dict[str, Any] = {
        "launcher_log": path,
        **_file_status_payload(path, prefix="launcher_log"),
        "status": "running_partial",
        "backend_name": backend_name,
        "progress_phase": progress_phase,
        "force_rows_started": bool(rows),
        "stage_summaries": stages,
        "iteration_row_count": len(rows),
        "tail_rows": rows[-12:],
        "last_row": last,
        "min_total": None if not totals else float(np.nanmin(np.asarray(totals, dtype=float))),
        "history": {
            "fsq_component_sum_tail_projection": _vmec2000_tail_projection(rows),
            "fsq_component_tail_projection_by_component": _vmec2000_component_tail_projections(rows),
        },
        "tail_plateau": _tail_plateau_payload(rows, stage_ftol=stage_ftol),
        "initial_jacobian_changed_sign": bool(saw_axis_repair),
        "vacuum_pressure_turn_on_iter": vacuum_turn_on_iter,
        "n_iter": None if last is None else int(last["it"]),
        "final_fsqr": None if last is None else float(last["fsqr"]),
        "final_fsqz": None if last is None else float(last["fsqz"]),
        "final_fsql": None if last is None else float(last["fsql"]),
        "final_fsq_component_sum": None if last is None else float(last["total"]),
        "best_scored_fsq": None if not totals else float(np.nanmin(np.asarray(totals, dtype=float))),
        "final_max_component": None if last is None else float(last["max_component"]),
    }
    return payload


def _summary_row(
    *,
    path: Path,
    backend_name: str,
    backend: dict[str, Any],
    cfg: dict[str, Any],
    projection: dict[str, Any],
    resolution_deck: dict[str, Any] | None = None,
    resolution_guardrail: dict[str, Any] | None = None,
    strict_convergence_assessment: dict[str, Any] | None = None,
    spline_bridge: dict[str, Any] | None = None,
    vmec_free_boundary_scale: dict[str, Any] | None = None,
    native_spline_vector_residual_profile: dict[str, Any] | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    case = _case_name(path)
    cfg = {**_config_from_case_name(case), **cfg}
    if not projection and not resolution_deck:
        projection, inferred_resolution = _infer_square_axis_resolution(cfg, case=case)
        if inferred_resolution:
            resolution_deck = inferred_resolution
            cfg.setdefault("recommended_nzeta", inferred_resolution.get("recommended_nzeta"))
            cfg.setdefault(
                "max_boundary_projection_error",
                inferred_resolution.get("projection_target_max_component_error"),
            )
    backend_for_projection = backend
    if backend_name == "vmec2000_mgrid" and not isinstance(backend.get("history"), dict):
        tail_rows = backend.get("tail_rows")
        if isinstance(tail_rows, list) and tail_rows:
            backend_for_projection = {
                **backend,
                "history": {
                    "fsq_component_sum_tail_projection": _vmec2000_tail_projection(tail_rows),
                    "fsq_component_tail_projection_by_component": _vmec2000_component_tail_projections(
                        tail_rows
                    ),
                },
            }
    requested_ftol = _finite_float(cfg.get("ftol"))
    if requested_ftol is None:
        requested_ftol = _finite_float(backend.get("requested_ftol"))
    if requested_ftol is None:
        stage_summaries = backend.get("stage_summaries")
        if isinstance(stage_summaries, list) and stage_summaries:
            last_stage = stage_summaries[-1]
            if isinstance(last_stage, dict):
                requested_ftol = _finite_float(last_stage.get("ftolv"))
    tail_plateau = backend.get("tail_plateau")
    if not isinstance(tail_plateau, dict):
        tail_rows = backend.get("tail_rows")
        tail_plateau = (
            _tail_plateau_payload(tail_rows, stage_ftol=requested_ftol)
            if isinstance(tail_rows, list)
            else {}
        )
    if backend_name == "vmec2000_mgrid":
        final_total, best_total, final_iter = _vmec2000_total(backend)
        final_max_component = _vmec2000_final_max_component(backend) or _finite_float(
            backend.get("final_max_component")
        )
    else:
        final_total, best_total, final_iter = _jax_total(backend)
        final_max_component = _jax_final_max_component(backend)
    virtual_casing = _virtual_casing_payload(backend)
    solver_overrides = _solver_overrides_payload(backend)
    step_status_counts = _history_counts(backend, "step_status_counts")
    restart_path_counts = _history_counts(backend, "restart_path_counts")
    freeb_jax_nestor_operator = solver_overrides.get(
        "freeb_jax_nestor_operator", cfg.get("freeb_jax_nestor_operator")
    )
    freeb_jax_nestor_jit_operator = solver_overrides.get(
        "freeb_jax_nestor_jit_operator", cfg.get("freeb_jax_nestor_jit_operator")
    )
    freeb_edge_control_projection = solver_overrides.get("freeb_edge_control_projection")
    if not isinstance(freeb_edge_control_projection, dict):
        requested = cfg.get("freeb_edge_control_projection")
        edge_enabled = _edge_projection_enabled_from_requested(requested)
        freeb_edge_control_projection = {
            "requested": requested,
            "enabled": edge_enabled,
            "status": "enabled" if edge_enabled else "disabled",
        }
    freeb_edge_control_runtime = backend.get("free_boundary_edge_control_projection")
    if not isinstance(freeb_edge_control_runtime, dict):
        freeb_edge_control_runtime = {}
    freeb_edge_control_state_residual = freeb_edge_control_runtime.get("state_residual")
    if not isinstance(freeb_edge_control_state_residual, dict):
        freeb_edge_control_state_residual = {}
    freeb_edge_control_state_coordinates = freeb_edge_control_runtime.get("state_coordinates")
    if not isinstance(freeb_edge_control_state_coordinates, dict):
        freeb_edge_control_state_coordinates = {}
    freeb_edge_control_reduced_unknown = freeb_edge_control_runtime.get("reduced_unknown_vector")
    if not isinstance(freeb_edge_control_reduced_unknown, dict):
        freeb_edge_control_reduced_unknown = {}
    freeb_edge_control_force_direction = freeb_edge_control_runtime.get("force_direction")
    if not isinstance(freeb_edge_control_force_direction, dict):
        freeb_edge_control_force_direction = freeb_edge_control_runtime.get("update_direction")
    if not isinstance(freeb_edge_control_force_direction, dict):
        freeb_edge_control_force_direction = {}
    freeb_edge_control_update_direction = freeb_edge_control_runtime.get("update_direction")
    if not isinstance(freeb_edge_control_update_direction, dict):
        freeb_edge_control_update_direction = {}
    freeb_edge_control_reduced_force_direction = freeb_edge_control_runtime.get(
        "reduced_force_direction"
    )
    if not isinstance(freeb_edge_control_reduced_force_direction, dict):
        freeb_edge_control_reduced_force_direction = freeb_edge_control_runtime.get(
            "reduced_update_direction"
        )
    if not isinstance(freeb_edge_control_reduced_force_direction, dict):
        freeb_edge_control_reduced_force_direction = {}
    freeb_edge_control_reduced_update_direction = freeb_edge_control_runtime.get("reduced_update_direction")
    if not isinstance(freeb_edge_control_reduced_update_direction, dict):
        freeb_edge_control_reduced_update_direction = {}
    freeb_edge_control_native_last_step = freeb_edge_control_runtime.get("native_last_step")
    if not isinstance(freeb_edge_control_native_last_step, dict):
        freeb_edge_control_native_last_step = {}
    freeb_edge_control_native_state = freeb_edge_control_runtime.get("native_control_state")
    if not isinstance(freeb_edge_control_native_state, dict):
        freeb_edge_control_native_state = {}
    freeb_edge_control_native_spline_state = freeb_edge_control_runtime.get("native_spline_state")
    if not isinstance(freeb_edge_control_native_spline_state, dict):
        freeb_edge_control_native_spline_state = {}
    freeb_edge_control_native_unknown = freeb_edge_control_runtime.get("native_unknown_vector")
    if not isinstance(freeb_edge_control_native_unknown, dict):
        freeb_edge_control_native_unknown = {}
    hot_restart = backend.get("hot_restart")
    hot_restart = hot_restart if isinstance(hot_restart, dict) else {}
    hot_restart_stages = hot_restart.get("stages")
    hot_restart_last_stage = (
        hot_restart_stages[-1]
        if isinstance(hot_restart_stages, list)
        and hot_restart_stages
        and isinstance(hot_restart_stages[-1], dict)
        else {}
    )
    hot_restart_stage_summaries = (
        [stage for stage in hot_restart_stages if isinstance(stage, dict)]
        if isinstance(hot_restart_stages, list)
        else []
    )
    hot_restart_resume_summaries = [
        stage.get("restart_solver_state")
        for stage in hot_restart_stage_summaries
        if isinstance(stage.get("restart_solver_state"), dict)
    ]
    hot_restart_last_resume = (
        hot_restart_resume_summaries[-1] if hot_restart_resume_summaries else {}
    )
    hot_restart_runtime_flags = [
        bool(summary.get("freeb_nestor_runtime_present"))
        for summary in hot_restart_resume_summaries
        if isinstance(summary, dict)
    ]
    strict_met = _strict_components_met(final_max_component, requested_ftol)
    strict_gap = _strict_gap(final_max_component, requested_ftol)
    promotion = _promotion_payload(
        backend_name=backend_name,
        backend=backend,
        cfg=cfg,
        strict_components_met=strict_met,
        virtual_casing=virtual_casing,
    )
    promotion_blockers = promotion.get("promotion_blockers")
    if isinstance(promotion_blockers, list):
        promotion_blockers_text = ",".join(str(item) for item in promotion_blockers)
    else:
        promotion_blockers_text = promotion_blockers
    control_projection = backend.get("boundary_reduced_control_projection")
    if not isinstance(control_projection, dict):
        control_projection = {}
    stellarator_projection = _control_projection_candidate(control_projection, "stellarator")
    full_projection = _control_projection_candidate(control_projection, "full")
    control_projection_state = control_projection.get("state_coordinates")
    if not isinstance(control_projection_state, dict):
        control_projection_state = {}
    stellarator_projection_state = stellarator_projection.get("state_coordinates")
    if not isinstance(stellarator_projection_state, dict):
        stellarator_projection_state = {}
    full_projection_state = full_projection.get("state_coordinates")
    if not isinstance(full_projection_state, dict):
        full_projection_state = {}
    iters_to_target = _tail_projection(backend_for_projection, "", target=1.0e-12)
    component_values = _final_force_components(backend)
    limiting_component = _limiting_component(backend_for_projection, component_values)
    strict_tail_projection = _strict_tail_projection_payload(
        backend_for_projection,
        limiting_component=limiting_component,
        target=1.0e-12,
    )
    strict_tail_statuses = strict_tail_projection.get("component_statuses")
    strict_tail_statuses_text = (
        ",".join(f"{key}:{value}" for key, value in sorted(strict_tail_statuses.items()))
        if isinstance(strict_tail_statuses, dict)
        else ""
    )
    max_iter = cfg.get("max_iter")
    if max_iter is None:
        max_iter = _last_stage_value(backend, "niter")
    remaining_iterations = _remaining_iterations(max_iter, final_iter)
    stage_schedule = _stage_schedule_payload(backend, final_iter=final_iter)
    vacuum_grid_exceeded_count = backend.get("vacuum_grid_exceeded_count")
    accepted_parity = _accepted_provider_parity_payload(backend)
    accepted_parity_field = accepted_parity.get("field_vector")
    accepted_parity_field = accepted_parity_field if isinstance(accepted_parity_field, dict) else {}
    accepted_parity_vac = accepted_parity.get("vacuum_channels")
    accepted_parity_vac = accepted_parity_vac if isinstance(accepted_parity_vac, dict) else {}
    accepted_parity_bnormal = accepted_parity_vac.get("bnormal")
    accepted_parity_bnormal = accepted_parity_bnormal if isinstance(accepted_parity_bnormal, dict) else {}
    edge_force_capture_status = _edge_force_capture_status(freeb_edge_control_force_direction)
    edge_force_capture_next_basis = _edge_force_capture_next_basis(
        edge_enabled=bool(freeb_edge_control_projection.get("enabled")),
        basis=freeb_edge_control_projection.get("basis_symmetry"),
        capture_status=edge_force_capture_status,
    )
    status_value = status if status is not None else backend.get("status")
    next_action = _recommended_next_action(
        status=status_value,
        progress_phase=backend.get("progress_phase"),
        force_rows_started=backend.get("force_rows_started"),
        strict_components_met=strict_met,
        tail_plateau_status=tail_plateau.get("status"),
        iters_to_target=iters_to_target,
        remaining_iterations=remaining_iterations,
        vacuum_grid_exceeded_count=vacuum_grid_exceeded_count,
        strict_tail_projection_status=strict_tail_projection.get("status"),
        strict_tail_iters_to_target=_finite_float(
            strict_tail_projection.get("limiting_component_iters_to_target")
        ),
    )
    resolution = resolution_deck if isinstance(resolution_deck, dict) else {}
    guardrail = resolution_guardrail if isinstance(resolution_guardrail, dict) else {}
    guardrail_requested = guardrail.get("requested_deck")
    guardrail_requested = guardrail_requested if isinstance(guardrail_requested, dict) else {}
    guardrail_effective = guardrail.get("effective_deck")
    guardrail_effective = guardrail_effective if isinstance(guardrail_effective, dict) else {}
    guardrail_auto_bumps = guardrail.get("auto_bumps")
    guardrail_auto_bumps = guardrail_auto_bumps if isinstance(guardrail_auto_bumps, dict) else {}
    guardrail_baseline = guardrail.get("validated_minimum_production_deck")
    guardrail_baseline = guardrail_baseline if isinstance(guardrail_baseline, dict) else {}
    strict_evidence = _strict_evidence_payload(
        status=status_value,
        backend_name=backend_name,
        requested_ftol=requested_ftol,
        strict_components_met=strict_met,
        strict_gap=strict_gap,
        next_action=next_action,
        production_candidate=promotion.get("production_candidate"),
        promotion_blockers=promotion_blockers_text,
        resolution_deck=resolution,
        cfg=cfg,
        tail_plateau_status=tail_plateau.get("status"),
    )
    recommended_followup = _recommended_followup_payload(
        backend_name=backend_name,
        status=status_value,
        next_action=next_action,
        strict_evidence_status=strict_evidence.get("strict_evidence_status"),
        accepted_provider_parity_status=accepted_parity.get("status"),
        freeb_jax_nestor_operator=freeb_jax_nestor_operator,
        freeb_edge_control_projection=freeb_edge_control_projection,
        freeb_edge_force_capture_status=edge_force_capture_status,
        freeb_edge_force_capture_next_basis=edge_force_capture_next_basis,
        vacuum_grid_exceeded_count=vacuum_grid_exceeded_count,
    )
    convergence_assessment = (
        strict_convergence_assessment if isinstance(strict_convergence_assessment, dict) else {}
    )
    spline_bridge = spline_bridge if isinstance(spline_bridge, dict) else {}
    scale = vmec_free_boundary_scale if isinstance(vmec_free_boundary_scale, dict) else {}
    native_vector_profile = (
        native_spline_vector_residual_profile
        if isinstance(native_spline_vector_residual_profile, dict)
        else {}
    )
    native_matrix_free_profile = native_vector_profile.get("matrix_free_normal_step_profile")
    if not isinstance(native_matrix_free_profile, dict):
        native_matrix_free_profile = {}
    return {
        "case": case,
        "backend": backend_name,
        **strict_evidence,
        **recommended_followup,
        "status": status_value,
        "progress_phase": backend.get("progress_phase"),
        "force_rows_started": backend.get("force_rows_started"),
        "launcher_log_size_bytes": backend.get("launcher_log_size_bytes"),
        "launcher_log_mtime_unix_s": _finite_float(backend.get("launcher_log_mtime_unix_s")),
        "threed1_size_bytes": backend.get("threed1_size_bytes"),
        "threed1_mtime_unix_s": _finite_float(backend.get("threed1_mtime_unix_s")),
        "initial_jacobian_changed_sign": backend.get("initial_jacobian_changed_sign"),
        "vacuum_pressure_turn_on_iter": backend.get("vacuum_pressure_turn_on_iter"),
        "mpol": cfg.get("mpol"),
        "ntor": cfg.get("ntor"),
        "ns": cfg.get("ns"),
        "nzeta": cfg.get("nzeta"),
        "resolution_guardrail_status": guardrail.get("status"),
        "resolution_guardrail_action": guardrail.get("action"),
        "resolution_guardrail_requested_vs_effective_changed": guardrail.get(
            "requested_vs_effective_changed"
        ),
        "requested_mpol": guardrail_requested.get("mpol", cfg.get("requested_mpol")),
        "requested_ntor": guardrail_requested.get("ntor", cfg.get("requested_ntor")),
        "requested_ntheta": guardrail_requested.get("ntheta", cfg.get("requested_ntheta")),
        "requested_nzeta": guardrail_requested.get("nzeta", cfg.get("requested_nzeta")),
        "effective_mpol": guardrail_effective.get("mpol", cfg.get("mpol")),
        "effective_ntor": guardrail_effective.get("ntor", cfg.get("ntor")),
        "effective_ntheta": guardrail_effective.get("ntheta", cfg.get("ntheta")),
        "effective_nzeta": guardrail_effective.get("nzeta", cfg.get("nzeta")),
        "resolution_guardrail_mode_deck_auto_bumped": guardrail_auto_bumps.get(
            "mode_deck_auto_bumped_to_recommended"
        ),
        "resolution_guardrail_ntheta_auto_bumped": guardrail_auto_bumps.get(
            "ntheta_auto_bumped_to_recommended"
        ),
        "resolution_guardrail_nzeta_auto_bumped": guardrail_auto_bumps.get(
            "nzeta_auto_bumped_to_recommended"
        ),
        "validated_minimum_mpol": guardrail_baseline.get("mpol"),
        "validated_minimum_ntor": guardrail_baseline.get("ntor"),
        "validated_minimum_ntheta": guardrail_baseline.get("ntheta"),
        "validated_minimum_nzeta": guardrail_baseline.get("nzeta"),
        "nzeta_auto": cfg.get("nzeta_auto"),
        "recommended_nzeta": cfg.get("recommended_nzeta"),
        "nvacskip": cfg.get("nvacskip"),
        "solver_mode": cfg.get("solver_mode"),
        "strict_backtracking": solver_overrides.get("strict_backtracking", cfg.get("strict_backtracking")),
        "jax_hot_restart_requested_count": solver_overrides.get(
            "jax_hot_restart_count", cfg.get("jax_hot_restart_count")
        ),
        "jax_hot_restart_executed_count": hot_restart.get("executed_count"),
        "jax_hot_restart_iters": solver_overrides.get(
            "jax_hot_restart_iters", cfg.get("jax_hot_restart_iters")
        ),
        "jax_hot_restart_policy": solver_overrides.get(
            "jax_hot_restart_policy", cfg.get("jax_hot_restart_policy")
        ),
        "jax_hot_restart_always": solver_overrides.get(
            "jax_hot_restart_always", cfg.get("jax_hot_restart_always")
        ),
        "jax_initial_restart_wout": solver_overrides.get(
            "jax_initial_restart_wout", cfg.get("jax_initial_restart_wout")
        ),
        "jax_hot_restart_stopped_after_strict": hot_restart.get("stopped_after_strict_convergence"),
        "jax_hot_restart_last_status": hot_restart_last_stage.get("strict_status"),
        "jax_hot_restart_last_component_max": _finite_float(
            hot_restart_last_stage.get("component_max")
        ),
        "jax_hot_restart_resume_present_last": hot_restart_last_resume.get("present"),
        "jax_hot_restart_resume_freeb_runtime_present_last": hot_restart_last_resume.get(
            "freeb_nestor_runtime_present"
        ),
        "jax_hot_restart_resume_freeb_runtime_present_any": any(hot_restart_runtime_flags)
        if hot_restart_runtime_flags
        else None,
        "jax_hot_restart_resume_freeb_model_last": hot_restart_last_resume.get("freeb_model"),
        "multigrid_resume_enabled": backend.get("multigrid_resume_enabled"),
        "multigrid_resume_env_default": backend.get("multigrid_resume_env_default"),
        "multigrid_resume_applied_any": _bool_any(backend.get("multigrid_resume_stage_applied")),
        "multigrid_resume_applied_last": _bool_last(backend.get("multigrid_resume_stage_applied")),
        "multigrid_resume_freeb_runtime_present_any": _bool_any(
            backend.get("multigrid_resume_freeb_runtime_present")
        ),
        "multigrid_resume_freeb_runtime_present_last": _bool_last(
            backend.get("multigrid_resume_freeb_runtime_present")
        ),
        "multigrid_resume_freeb_model_last": _last_sequence_value(
            backend.get("multigrid_resume_freeb_model")
        ),
        "freeb_jax_nestor_operator": freeb_jax_nestor_operator,
        "freeb_jax_nestor_jit_operator": freeb_jax_nestor_jit_operator,
        "freeb_edge_control_projection_status": freeb_edge_control_projection.get("status"),
        "freeb_edge_control_projection_requested": freeb_edge_control_projection.get("requested"),
        "freeb_edge_control_projection_basis": freeb_edge_control_projection.get("basis_symmetry"),
        "freeb_edge_control_projection_control_count": freeb_edge_control_projection.get("control_count"),
        "freeb_edge_control_projection_rcond": _finite_float(
            freeb_edge_control_projection.get("rcond")
        ),
        "freeb_edge_control_projection_ridge": _finite_float(
            freeb_edge_control_projection.get("ridge")
        ),
        "freeb_edge_control_projection_trust_radius": _finite_float(
            freeb_edge_control_projection.get("trust_radius")
        ),
        "freeb_edge_control_projection_update_mode": freeb_edge_control_projection.get("update_mode"),
        "freeb_edge_control_projection_apply_count": freeb_edge_control_runtime.get("apply_count"),
        "freeb_edge_control_projection_delta_projection_count": freeb_edge_control_runtime.get(
            "delta_projection_count"
        ),
        "freeb_edge_control_projection_coordinate_update_count": freeb_edge_control_runtime.get(
            "coordinate_update_count"
        ),
        "freeb_edge_control_projection_native_coordinate_update_count": freeb_edge_control_runtime.get(
            "native_coordinate_update_count"
        ),
        "freeb_edge_control_projection_native_velocity_reset_count": freeb_edge_control_runtime.get(
            "native_velocity_reset_count"
        ),
        "freeb_edge_control_projection_native_resync_count": freeb_edge_control_runtime.get("native_resync_count"),
        "freeb_edge_control_projection_native_force_l2": _finite_float(
            freeb_edge_control_native_last_step.get("control_force_l2")
        ),
        "freeb_edge_control_projection_native_velocity_l2": _finite_float(
            freeb_edge_control_native_last_step.get("control_velocity_l2")
        ),
        "freeb_edge_control_projection_native_update_l2": _finite_float(
            freeb_edge_control_native_last_step.get("control_update_l2")
        ),
        "freeb_edge_control_projection_native_trust_scale": _finite_float(
            freeb_edge_control_native_last_step.get("trust_scale")
        ),
        "freeb_edge_control_projection_native_decoded_edge_update_l2": _finite_float(
            freeb_edge_control_native_last_step.get("decoded_edge_update_l2")
        ),
        "freeb_edge_control_projection_native_decoded_edge_update_linf": _finite_float(
            freeb_edge_control_native_last_step.get("decoded_edge_update_linf")
        ),
        "freeb_edge_control_projection_native_source_update_l2": _finite_float(
            freeb_edge_control_native_last_step.get("source_edge_update_l2")
        ),
        "freeb_edge_control_projection_native_source_update_residual_l2": _finite_float(
            freeb_edge_control_native_last_step.get("source_update_residual_l2")
        ),
        "freeb_edge_control_projection_native_source_update_residual_linf": _finite_float(
            freeb_edge_control_native_last_step.get("source_update_residual_linf")
        ),
        "freeb_edge_control_projection_native_source_update_residual_rel": _finite_float(
            freeb_edge_control_native_last_step.get("source_update_residual_rel")
        ),
        "freeb_edge_control_projection_native_source_update_captured_fraction": _finite_float(
            freeb_edge_control_native_last_step.get("source_update_captured_fraction")
        ),
        "freeb_edge_control_projection_native_state_status": freeb_edge_control_native_state.get(
            "status"
        ),
        "freeb_edge_control_projection_native_state_schema": freeb_edge_control_native_state.get(
            "native_state_schema"
        ),
        "freeb_edge_control_projection_native_state_mode": freeb_edge_control_native_state.get(
            "mode"
        ),
        "freeb_edge_control_projection_native_state_unknown_l2": _finite_float(
            freeb_edge_control_native_state.get("unknown_l2")
        ),
        "freeb_edge_control_projection_native_state_unknown_linf": _finite_float(
            freeb_edge_control_native_state.get("unknown_linf")
        ),
        "freeb_edge_control_projection_native_state_decoded_edge_linf": _finite_float(
            freeb_edge_control_native_state.get("decoded_edge_linf")
        ),
        "freeb_edge_control_projection_native_state_fit_residual_linf": _finite_float(
            freeb_edge_control_native_state.get("fit_residual_linf")
        ),
        "freeb_edge_control_projection_native_state_fit_residual_rel": _finite_float(
            freeb_edge_control_native_state.get("fit_residual_rel")
        ),
        "freeb_edge_control_projection_native_spline_state_status": (
            freeb_edge_control_native_spline_state.get("status")
        ),
        "freeb_edge_control_projection_native_spline_state_schema": (
            freeb_edge_control_native_spline_state.get("native_state_schema")
        ),
        "freeb_edge_control_projection_native_spline_state_mode": (
            freeb_edge_control_native_spline_state.get("mode")
        ),
        "freeb_edge_control_projection_native_spline_state_full_edge_size": (
            freeb_edge_control_native_spline_state.get("full_edge_size")
        ),
        "freeb_edge_control_projection_native_spline_state_reduced_unknown_size": (
            freeb_edge_control_native_spline_state.get("reduced_unknown_size")
        ),
        "freeb_edge_control_projection_native_spline_state_unknown_linf": _finite_float(
            freeb_edge_control_native_spline_state.get("unknown_linf")
        ),
        "freeb_edge_control_projection_native_unknown_status": (
            freeb_edge_control_native_unknown.get("status")
        ),
        "freeb_edge_control_projection_native_unknown_schema": (
            freeb_edge_control_native_unknown.get("native_state_schema")
        ),
        "freeb_edge_control_projection_native_unknown_size": (
            freeb_edge_control_native_unknown.get("native_unknown_size")
        ),
        "freeb_edge_control_projection_native_unknown_full_vmec_size": (
            freeb_edge_control_native_unknown.get("full_vmec_size")
        ),
        "freeb_edge_control_projection_native_unknown_interior_size": (
            freeb_edge_control_native_unknown.get("interior_unknown_size")
        ),
        "freeb_edge_control_projection_native_unknown_edge_control_size": (
            freeb_edge_control_native_unknown.get("edge_control_size")
        ),
        "freeb_edge_control_projection_native_unknown_removed_edge_dofs": (
            freeb_edge_control_native_unknown.get("removed_fourier_edge_dofs")
        ),
        "freeb_edge_control_projection_native_unknown_reduction_fraction": _finite_float(
            freeb_edge_control_native_unknown.get("unknown_reduction_fraction")
        ),
        "freeb_edge_control_projection_native_unknown_edge_control_linf": _finite_float(
            freeb_edge_control_native_unknown.get("edge_control_linf")
        ),
        "freeb_edge_control_projection_native_unknown_reconstruction_residual_linf": _finite_float(
            freeb_edge_control_native_unknown.get("edge_reconstruction_residual_linf")
        ),
        "freeb_edge_control_projection_native_unknown_reconstruction_residual_rel": _finite_float(
            freeb_edge_control_native_unknown.get("edge_reconstruction_residual_rel")
        ),
        "freeb_edge_control_projection_zero_velocity_count": freeb_edge_control_runtime.get(
            "zero_velocity_count"
        ),
        "freeb_edge_control_projection_state_residual_status": freeb_edge_control_state_residual.get(
            "status"
        ),
        "freeb_edge_control_projection_state_residual_linf": _finite_float(
            freeb_edge_control_state_residual.get("residual_linf")
        ),
        "freeb_edge_control_projection_state_residual_rms": _finite_float(
            freeb_edge_control_state_residual.get("residual_rms")
        ),
        "freeb_edge_control_projection_state_residual_rel": _finite_float(
            freeb_edge_control_state_residual.get("residual_rel")
        ),
        "freeb_edge_control_projection_state_captured_fraction": _finite_float(
            freeb_edge_control_state_residual.get("captured_fraction")
        ),
        "freeb_edge_control_projection_state_coordinates_status": freeb_edge_control_state_coordinates.get(
            "status"
        ),
        "freeb_edge_control_projection_state_coordinate_linf": _finite_float(
            freeb_edge_control_state_coordinates.get("coordinate_linf")
        ),
        "freeb_edge_control_projection_state_coordinate_l2": _finite_float(
            freeb_edge_control_state_coordinates.get("coordinate_l2")
        ),
        "freeb_edge_control_projection_state_reconstruction_residual_linf": _finite_float(
            freeb_edge_control_state_coordinates.get("reconstruction_residual_linf")
        ),
        "freeb_edge_control_projection_state_reconstruction_residual_rms": _finite_float(
            freeb_edge_control_state_coordinates.get("reconstruction_residual_rms")
        ),
        "freeb_edge_control_projection_state_reconstruction_residual_rel": _finite_float(
            freeb_edge_control_state_coordinates.get("reconstruction_residual_rel")
        ),
        "freeb_edge_control_projection_reduced_unknown_status": freeb_edge_control_reduced_unknown.get(
            "status"
        ),
        "freeb_edge_control_projection_reduced_unknown_size": freeb_edge_control_reduced_unknown.get(
            "reduced_unknown_size"
        ),
        "freeb_edge_control_projection_full_edge_size": freeb_edge_control_reduced_unknown.get(
            "full_edge_size"
        ),
        "freeb_edge_control_projection_unknown_reduction_fraction": _finite_float(
            freeb_edge_control_reduced_unknown.get("reduction_fraction")
        ),
        "freeb_edge_control_projection_unknown_decoded_residual_linf": _finite_float(
            freeb_edge_control_reduced_unknown.get("decoded_residual_linf")
        ),
        "freeb_edge_control_projection_unknown_decoded_residual_rel": _finite_float(
            freeb_edge_control_reduced_unknown.get("decoded_residual_rel")
        ),
        "freeb_edge_control_projection_update_direction_status": freeb_edge_control_update_direction.get(
            "status"
        ),
        "freeb_edge_control_projection_update_direction_linf": _finite_float(
            freeb_edge_control_update_direction.get("residual_linf")
        ),
        "freeb_edge_control_projection_update_direction_rms": _finite_float(
            freeb_edge_control_update_direction.get("residual_rms")
        ),
        "freeb_edge_control_projection_update_direction_rel": _finite_float(
            freeb_edge_control_update_direction.get("residual_rel")
        ),
        "freeb_edge_control_projection_update_direction_captured_fraction": _finite_float(
            freeb_edge_control_update_direction.get("captured_fraction")
        ),
        "freeb_edge_control_projection_update_direction_trust_scale": _finite_float(
            freeb_edge_control_update_direction.get("trust_scale")
        ),
        "freeb_edge_control_projection_force_direction_status": freeb_edge_control_force_direction.get(
            "status"
        ),
        "freeb_edge_control_projection_force_direction_linf": _finite_float(
            freeb_edge_control_force_direction.get("residual_linf")
        ),
        "freeb_edge_control_projection_force_direction_rms": _finite_float(
            freeb_edge_control_force_direction.get("residual_rms")
        ),
        "freeb_edge_control_projection_force_direction_rel": _finite_float(
            freeb_edge_control_force_direction.get("residual_rel")
        ),
        "freeb_edge_control_projection_force_direction_captured_fraction": _finite_float(
            freeb_edge_control_force_direction.get("captured_fraction")
        ),
        "freeb_edge_control_projection_force_direction_trust_scale": _finite_float(
            freeb_edge_control_force_direction.get("trust_scale")
        ),
        "freeb_edge_control_projection_force_capture_status": edge_force_capture_status,
        "freeb_edge_control_projection_force_capture_next_basis": edge_force_capture_next_basis,
        "freeb_edge_control_projection_reduced_update_status": freeb_edge_control_reduced_update_direction.get(
            "status"
        ),
        "freeb_edge_control_projection_reduced_update_size": freeb_edge_control_reduced_update_direction.get(
            "reduced_update_size"
        ),
        "freeb_edge_control_projection_full_update_size": freeb_edge_control_reduced_update_direction.get(
            "full_update_size"
        ),
        "freeb_edge_control_projection_reduced_update_linf": _finite_float(
            freeb_edge_control_reduced_update_direction.get("update_linf")
        ),
        "freeb_edge_control_projection_reduced_update_decoded_residual_linf": _finite_float(
            freeb_edge_control_reduced_update_direction.get("decoded_residual_linf")
        ),
        "freeb_edge_control_projection_reduced_update_decoded_residual_rel": _finite_float(
            freeb_edge_control_reduced_update_direction.get("decoded_residual_rel")
        ),
        "freeb_edge_control_projection_reduced_update_captured_fraction": _finite_float(
            freeb_edge_control_reduced_update_direction.get("captured_fraction")
        ),
        "freeb_edge_control_projection_reduced_force_status": freeb_edge_control_reduced_force_direction.get(
            "status"
        ),
        "freeb_edge_control_projection_reduced_force_size": freeb_edge_control_reduced_force_direction.get(
            "reduced_update_size"
        ),
        "freeb_edge_control_projection_reduced_force_linf": _finite_float(
            freeb_edge_control_reduced_force_direction.get("update_linf")
        ),
        "freeb_edge_control_projection_reduced_force_decoded_residual_linf": _finite_float(
            freeb_edge_control_reduced_force_direction.get("decoded_residual_linf")
        ),
        "freeb_edge_control_projection_reduced_force_decoded_residual_rel": _finite_float(
            freeb_edge_control_reduced_force_direction.get("decoded_residual_rel")
        ),
        "freeb_edge_control_projection_reduced_force_captured_fraction": _finite_float(
            freeb_edge_control_reduced_force_direction.get("captured_fraction")
        ),
        "free_boundary_jax_nestor_operator_applied": backend.get(
            "free_boundary_jax_nestor_operator_applied"
        ),
        "free_boundary_jax_nestor_operator_reason": backend.get(
            "free_boundary_jax_nestor_operator_reason"
        ),
        "free_boundary_jax_nestor_operator_jitted": backend.get(
            "free_boundary_jax_nestor_operator_jitted"
        ),
        "free_boundary_jax_nestor_operator_cache_hit": backend.get(
            "free_boundary_jax_nestor_operator_cache_hit"
        ),
        "free_boundary_jax_nestor_operator_time_s": _finite_float(
            backend.get("free_boundary_jax_nestor_operator_time_s")
        ),
        "side_power": cfg.get("side_power"),
        "corner_power": cfg.get("corner_power"),
        "max_iter": max_iter,
        **stage_schedule,
        "requested_ftol": requested_ftol,
        "strict_gap": strict_gap,
        "remaining_iterations": remaining_iterations,
        "next_action": next_action,
        "strict_assessment_full_fourier_status": convergence_assessment.get(
            "full_fourier_strict_profile_status"
        ),
        "strict_assessment_reduced_control_status": convergence_assessment.get(
            "reduced_control_profile_status"
        ),
        "strict_assessment_solver_native_spline_status": convergence_assessment.get(
            "solver_native_spline_status"
        ),
        "strict_assessment_solver_native_spline_edge_controls": convergence_assessment.get(
            "solver_native_spline_edge_controls"
        ),
        "strict_assessment_solver_native_spline_scope": convergence_assessment.get(
            "solver_native_spline_scope"
        ),
        "strict_assessment_full_native_spline_state_required": convergence_assessment.get(
            "full_native_spline_state_required_for_less_fourier_pressure"
        ),
        "strict_assessment_vmec2000_fix_fourier_bottleneck": convergence_assessment.get(
            "vmec2000_expected_to_fix_fourier_bottleneck"
        ),
        "spline_bridge_status": spline_bridge.get("status"),
        "spline_bridge_nonlinear_solver_boundary_basis": spline_bridge.get(
            "nonlinear_solver_boundary_basis"
        ),
        "spline_bridge_solver_native_spline_controls": spline_bridge.get(
            "solver_native_spline_controls"
        ),
        "spline_bridge_solver_native_spline_edge_controls": spline_bridge.get(
            "solver_native_spline_edge_controls"
        ),
        "spline_bridge_solver_native_spline_scope": spline_bridge.get(
            "solver_native_spline_scope"
        ),
        "spline_bridge_solver_edge_control_projection_enabled": spline_bridge.get(
            "solver_edge_control_projection_enabled"
        ),
        "spline_bridge_solver_edge_control_update_mode": spline_bridge.get(
            "solver_edge_control_update_mode"
        ),
        "spline_bridge_can_reduce_input_shape_dofs": spline_bridge.get(
            "can_reduce_input_shape_dofs"
        ),
        "spline_bridge_can_project_free_boundary_edge_updates": spline_bridge.get(
            "can_project_free_boundary_edge_updates"
        ),
        "spline_bridge_can_reduce_free_boundary_edge_dofs": spline_bridge.get(
            "can_reduce_free_boundary_edge_dofs"
        ),
        "spline_bridge_can_reduce_full_nonlinear_solver_dofs": spline_bridge.get(
            "can_reduce_full_nonlinear_solver_dofs"
        ),
        "spline_bridge_requires_native_spline_state_for_reduced_nonlinear_dofs": spline_bridge.get(
            "requires_native_spline_state_for_reduced_nonlinear_dofs"
        ),
        "native_spline_vector_residual_profile_status": native_vector_profile.get("status"),
        "native_spline_vector_residual_profile_native_unknown_size": native_vector_profile.get(
            "native_unknown_size"
        ),
        "native_spline_vector_residual_profile_full_vmec_size": native_vector_profile.get(
            "full_vmec_size"
        ),
        "native_spline_vector_residual_profile_removed_edge_dofs": native_vector_profile.get(
            "removed_fourier_edge_dofs"
        ),
        "native_spline_vector_residual_profile_reduction_fraction": _finite_float(
            native_vector_profile.get("unknown_reduction_fraction")
        ),
        "native_spline_vector_residual_profile_decode_parity_linf": _finite_float(
            native_vector_profile.get("decode_parity_linf")
        ),
        "native_spline_vector_residual_profile_projected_residual_parity_linf": _finite_float(
            native_vector_profile.get("projected_residual_host_parity_linf")
        ),
        "native_spline_vector_residual_profile_projected_residual_parity_rel": _finite_float(
            native_vector_profile.get("projected_residual_host_parity_rel")
        ),
        "native_spline_vector_residual_profile_projected_residual_jvp_wall_s": _finite_float(
            native_vector_profile.get("projected_residual_jvp_wall_s")
        ),
        "native_spline_vector_residual_profile_jvp_linf": _finite_float(
            native_vector_profile.get("jvp_linf")
        ),
        "native_spline_vector_residual_profile_matrix_free_status": (
            native_matrix_free_profile.get("status")
        ),
        "native_spline_vector_residual_profile_matrix_free_method": (
            native_matrix_free_profile.get("method")
        ),
        "native_spline_vector_residual_profile_matrix_free_vjp_wall_s": _finite_float(
            native_matrix_free_profile.get("vjp_wall_s")
        ),
        "native_spline_vector_residual_profile_matrix_free_normal_matvec_wall_s": _finite_float(
            native_matrix_free_profile.get("normal_matvec_wall_s")
        ),
        "native_spline_vector_residual_profile_matrix_free_cg_step_wall_s": _finite_float(
            native_matrix_free_profile.get("cg_step_wall_s")
        ),
        "native_spline_vector_residual_profile_matrix_free_linear_maxiter": (
            native_matrix_free_profile.get("linear_maxiter")
        ),
        "native_spline_vector_residual_profile_matrix_free_residual_reduction_factor": _finite_float(
            native_matrix_free_profile.get("residual_reduction_factor")
        ),
        "native_spline_vector_residual_profile_matrix_free_step_l2": _finite_float(
            native_matrix_free_profile.get("step_l2")
        ),
        "native_spline_vector_residual_profile_matrix_free_next_action": (
            native_matrix_free_profile.get("next_action")
        ),
        "native_spline_vector_residual_profile_next_action": native_vector_profile.get(
            "next_action"
        ),
        "resolution_deck_status": resolution.get("status"),
        "resolution_deck_reasons": ",".join(_as_text_list(resolution.get("reasons"))),
        "vmec_scale_status": scale.get("status"),
        "vmec_scale_phiedge": _finite_float(scale.get("phiedge")),
        "vmec_scale_external_r_bphi_rms": _finite_float(scale.get("external_r_bphi_rms")),
        "vmec_scale_proxy_over_external_r_bphi_rms": _finite_float(
            scale.get("phiedge_proxy_over_external_r_bphi_rms")
        ),
        "vmec_scale_suggested_phiedge": _finite_float(
            scale.get("suggested_phiedge_for_external_r_bphi_rms")
        ),
        "vmec_scale_suggested_phiedge_over_current": _finite_float(
            scale.get("suggested_phiedge_over_current")
        ),
        "accepted_provider_parity_status": accepted_parity.get("status"),
        "accepted_provider_parity_sample": accepted_parity.get("sample"),
        "accepted_provider_parity_field_diff_rms_rel": _finite_float(
            accepted_parity_field.get("diff_rms_rel")
        ),
        "accepted_provider_parity_bnormal_diff_rms_rel": _finite_float(
            accepted_parity_bnormal.get("diff_rms_rel")
        ),
        "accepted_provider_parity_field_lt_5pct": accepted_parity.get("field_rms_rel_lt_5pct"),
        "accepted_provider_parity_bnormal_lt_10pct": accepted_parity.get("bnormal_rms_rel_lt_10pct"),
        "boundary_mode_count": projection.get("mode_count"),
        "boundary_recommended_nzeta": projection.get("recommended_nzeta"),
        "max_boundary_projection_error": cfg.get("max_boundary_projection_error"),
        "boundary_proj_max": _finite_float(projection.get("max_abs_component_error")),
        "boundary_proj_rel": _finite_float(projection.get("max_abs_component_error_rel")),
        "boundary_coeff_delta_l2": _finite_float(backend.get("boundary_coeff_delta_l2")),
        "boundary_coeff_delta_linf": _finite_float(backend.get("boundary_coeff_delta_linf")),
        "boundary_coeff_delta_rel": _finite_float(backend.get("boundary_coeff_delta_rel")),
        "boundary_sample_displacement_rms": _finite_float(
            backend.get("boundary_sample_displacement_rms")
        ),
        "boundary_sample_displacement_max": _finite_float(
            backend.get("boundary_sample_displacement_max")
        ),
        "boundary_sample_displacement_rel": _finite_float(
            backend.get("boundary_sample_displacement_rel")
        ),
        "boundary_control_projection_status": control_projection.get("status"),
        "boundary_control_projection_residual_rel": _finite_float(
            control_projection.get("residual_rel")
        ),
        "boundary_control_projection_captured_fraction": _finite_float(
            control_projection.get("captured_fraction")
        ),
        "boundary_control_projection_radius_delta": _control_projection_delta_text(control_projection),
        "boundary_control_projection_state_coordinate_linf": _finite_float(
            control_projection_state.get("coordinate_linf")
        ),
        "boundary_control_projection_state_reconstruction_residual_rel": _finite_float(
            control_projection_state.get("reconstruction_residual_rel")
        ),
        "boundary_control_projection_stellarator_residual_rel": _finite_float(
            stellarator_projection.get("residual_rel")
        ),
        "boundary_control_projection_stellarator_captured_fraction": _finite_float(
            stellarator_projection.get("captured_fraction")
        ),
        "boundary_control_projection_stellarator_control_count": stellarator_projection.get("control_count"),
        "boundary_control_projection_stellarator_state_reconstruction_residual_rel": _finite_float(
            stellarator_projection_state.get("reconstruction_residual_rel")
        ),
        "boundary_control_projection_full_residual_rel": _finite_float(full_projection.get("residual_rel")),
        "boundary_control_projection_full_captured_fraction": _finite_float(
            full_projection.get("captured_fraction")
        ),
        "boundary_control_projection_full_control_count": full_projection.get("control_count"),
        "boundary_control_projection_full_state_reconstruction_residual_rel": _finite_float(
            full_projection_state.get("reconstruction_residual_rel")
        ),
        "final_iter": final_iter,
        "final_total": final_total,
        "final_max_component": final_max_component,
        "final_fsqr": component_values["fsqr"],
        "final_fsqz": component_values["fsqz"],
        "final_fsql": component_values["fsql"],
        "limiting_component": limiting_component,
        "fsqr_strict_gap": _strict_gap_for_component(component_values["fsqr"], requested_ftol),
        "fsqz_strict_gap": _strict_gap_for_component(component_values["fsqz"], requested_ftol),
        "fsql_strict_gap": _strict_gap_for_component(component_values["fsql"], requested_ftol),
        "strict_components_met": strict_met,
        "boundary_condition_mode": promotion.get("boundary_condition_mode"),
        "coil_bnormal_role": promotion.get("coil_bnormal_role"),
        "production_candidate": promotion.get("production_candidate"),
        "promotion_blockers": promotion_blockers_text,
        "virtual_casing_required": promotion.get("virtual_casing_required"),
        "virtual_casing_available": promotion.get("virtual_casing_available"),
        "best_total": best_total,
        "best_component_max": _finite_float(backend.get("best_scored_component_max")),
        "returned_best_scored_state": backend.get("returned_best_scored_state"),
        "best_scored_full_boundary_count": backend.get("best_scored_full_boundary_count"),
        "best_scored_fresh_boundary_count": backend.get("best_scored_fresh_boundary_count"),
        "best_scored_drift_restart_count": backend.get("best_scored_drift_restart_count"),
        "best_scored_drift_streak": backend.get("best_scored_drift_streak"),
        "best_scored_drift_last_restart_iter": backend.get("best_scored_drift_last_restart_iter"),
        "best_scored_drift_last_ratio": _finite_float(backend.get("best_scored_drift_last_ratio")),
        "final_residual_recomputed_on_accepted_state": backend.get(
            "final_residual_recomputed_on_accepted_state"
        ),
        "fresh_convergence_gate": backend.get("free_boundary_fresh_convergence_gate"),
        "fresh_convergence_rechecks": backend.get("free_boundary_fresh_convergence_recheck_count"),
        "fresh_convergence_rejects": backend.get("free_boundary_fresh_convergence_reject_count"),
        "fresh_convergence_failures": backend.get("free_boundary_fresh_convergence_failed_count"),
        "freeb_convergence_blocked_count": backend.get("free_boundary_convergence_blocked_count"),
        "update_delta_rms": _finite_float(backend.get("update_delta_rms")),
        "update_delta_to_velocity_rms_ratio": _finite_float(
            backend.get("update_delta_to_velocity_rms_ratio")
        ),
        "trial_ratio_last": _stat(backend, "w_try_ratio_stats", "last"),
        "trial_ratio_min": _stat(backend, "w_try_ratio_stats", "min"),
        "trial_ratio_max": _stat(backend, "w_try_ratio_stats", "max"),
        "trial_ratio_mean": _stat(backend, "w_try_ratio_stats", "mean"),
        "step_status_counts": _counts_text(step_status_counts),
        "step_momentum_count": _count_sum(step_status_counts, "momentum"),
        "step_rejected_count": _count_sum(step_status_counts, "rejected", "restart_pending"),
        "step_restart_count": _count_sum(step_status_counts, prefix="restart_"),
        "restart_path_counts": _counts_text(restart_path_counts),
        "restart_path_trial_rejected_count": _count_sum(restart_path_counts, "trial_rejected"),
        "restart_path_momentum_accept_count": _count_sum(restart_path_counts, "momentum_accept"),
        "dt_eff_last": _stat(backend, "dt_eff_stats", "last"),
        "dt_eff_min": _stat(backend, "dt_eff_stats", "min"),
        "time_step_last": _stat(backend, "time_step_stats", "last"),
        "freeb_full_update_count": _stat(backend, "freeb_full_update_stats", "sum"),
        "nestor_reuse_count": _stat(backend, "freeb_nestor_reused_stats", "sum"),
        "nestor_reuse_last": _stat(backend, "freeb_nestor_reused_stats", "last"),
        "nestor_source_reuse_count": _stat(backend, "freeb_nestor_source_reused_stats", "sum"),
        "nestor_source_reuse_last": _stat(backend, "freeb_nestor_source_reused_stats", "last"),
        "nestor_provider_source_reuse_allowed_last": _stat(
            backend, "freeb_nestor_provider_allows_source_reuse_stats", "last"
        ),
        "nestor_sample_time_last": _stat(backend, "freeb_nestor_sample_time_stats", "last"),
        "nestor_sample_time_mean": _stat(backend, "freeb_nestor_sample_time_stats", "mean"),
        "nestor_sample_time_max": _stat(backend, "freeb_nestor_sample_time_stats", "max"),
        "nestor_solve_time_last": _stat(backend, "freeb_nestor_solve_time_stats", "last"),
        "nestor_solve_time_mean": _stat(backend, "freeb_nestor_solve_time_stats", "mean"),
        "nestor_solve_time_max": _stat(backend, "freeb_nestor_solve_time_stats", "max"),
        "nestor_trial_reuse_count": _stat(backend, "freeb_nestor_trial_reused_stats", "sum"),
        "nestor_trial_failed_count": _stat(backend, "freeb_nestor_trial_failed_stats", "sum"),
        "nestor_trial_sample_time_mean": _stat(
            backend, "freeb_nestor_trial_sample_time_stats", "mean"
        ),
        "nestor_trial_sample_time_max": _stat(backend, "freeb_nestor_trial_sample_time_stats", "max"),
        "nestor_trial_solve_time_mean": _stat(backend, "freeb_nestor_trial_solve_time_stats", "mean"),
        "nestor_trial_solve_time_max": _stat(backend, "freeb_nestor_trial_solve_time_stats", "max"),
        "include_edge_count": _stat(backend, "include_edge_stats", "sum"),
        "include_edge_last": _stat(backend, "include_edge_stats", "last"),
        "anderson_pressure_enabled": backend.get("free_boundary_anderson_pressure_enabled"),
        "anderson_pressure_applied_count": _stat(backend, "freeb_anderson_pressure_applied_stats", "sum"),
        "anderson_pressure_last_theta": _finite_float(
            backend.get("free_boundary_anderson_pressure_last_theta")
        ),
        "bad_jacobian_count": _stat(backend, "bad_jacobian_stats", "sum"),
        "bnormal_rms_last": _stat(backend, "freeb_nestor_bnormal_rms_stats", "last"),
        "bnormal_rms_min": _stat(backend, "freeb_nestor_bnormal_rms_stats", "min"),
        "virtual_casing_status": virtual_casing.get("status"),
        "virtual_casing_grid_adequacy_status": virtual_casing.get("grid_adequacy_status"),
        "virtual_casing_surface_ntheta": _finite_float(virtual_casing.get("ntheta")),
        "virtual_casing_surface_nzeta": _finite_float(virtual_casing.get("nzeta")),
        "virtual_casing_quad_ntheta": _finite_float(virtual_casing.get("quad_ntheta")),
        "virtual_casing_quad_nzeta": _finite_float(virtual_casing.get("quad_nzeta")),
        "virtual_casing_quad_factor_theta": _finite_float(virtual_casing.get("quad_factor_theta")),
        "virtual_casing_quad_factor_zeta": _finite_float(virtual_casing.get("quad_factor_zeta")),
        "virtual_casing_external_bnormal_residual_rms": _finite_float(
            virtual_casing.get("external_bnormal_residual_rms")
        ),
        "virtual_casing_external_bnormal_residual_max": _finite_float(
            virtual_casing.get("external_bnormal_residual_max")
        ),
        "virtual_casing_pressure_balance_rms": _finite_float(virtual_casing.get("pressure_balance_rms")),
        "virtual_casing_pressure_balance_max": _finite_float(virtual_casing.get("pressure_balance_max")),
        "virtual_casing_required_external_b_rms": _finite_float(virtual_casing.get("required_external_b_rms")),
        "virtual_casing_target_external_b_rms": _finite_float(virtual_casing.get("target_external_b_rms")),
        "virtual_casing_wall_s": _finite_float(virtual_casing.get("wall_s")),
        "tail_decay_factor": _tail_projection(backend_for_projection, "per_iter_factor"),
        "iters_to_1e-12_est": iters_to_target,
        "strict_tail_projection_status": strict_tail_projection.get("status"),
        "strict_tail_projection_target": _finite_float(strict_tail_projection.get("target")),
        "strict_tail_component_statuses": strict_tail_statuses_text,
        "strict_tail_limiting_component": strict_tail_projection.get("limiting_component"),
        "strict_tail_limiting_component_status": strict_tail_projection.get("limiting_component_status"),
        "strict_tail_limiting_component_factor": _finite_float(
            strict_tail_projection.get("limiting_component_factor")
        ),
        "strict_tail_limiting_iters_to_1e-12_est": _finite_float(
            strict_tail_projection.get("limiting_component_iters_to_target")
        ),
        "fsqr_tail_decay_factor": _component_tail_projection(
            backend_for_projection, "fsqr", "per_iter_factor"
        ),
        "fsqz_tail_decay_factor": _component_tail_projection(
            backend_for_projection, "fsqz", "per_iter_factor"
        ),
        "fsql_tail_decay_factor": _component_tail_projection(
            backend_for_projection, "fsql", "per_iter_factor"
        ),
        "fsqr_iters_to_1e-12_est": _component_tail_projection(
            backend_for_projection, "fsqr", "", target=1.0e-12
        ),
        "fsqz_iters_to_1e-12_est": _component_tail_projection(
            backend_for_projection, "fsqz", "", target=1.0e-12
        ),
        "fsql_iters_to_1e-12_est": _component_tail_projection(
            backend_for_projection, "fsql", "", target=1.0e-12
        ),
        "tail_plateau_status": tail_plateau.get("status"),
        "tail_plateau_window": tail_plateau.get("window"),
        "tail_total_rel_span": _finite_float(tail_plateau.get("total_rel_span")),
        "tail_last_over_min": _finite_float(tail_plateau.get("total_last_over_min")),
        "wall_s": _finite_float(backend.get("wall_s")),
        "vacuum_grid_exceeded_count": vacuum_grid_exceeded_count,
    }


def rows_from_profile(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    cfg = data.get("configuration", {})
    cfg = cfg if isinstance(cfg, dict) else {}
    projection = data.get("boundary_projection", {})
    projection = projection if isinstance(projection, dict) else {}
    resolution_deck = data.get("resolution_deck", {})
    resolution_deck = resolution_deck if isinstance(resolution_deck, dict) else {}
    resolution_guardrail = data.get("resolution_guardrail", {})
    resolution_guardrail = resolution_guardrail if isinstance(resolution_guardrail, dict) else {}
    strict_convergence_assessment = data.get("strict_convergence_assessment", {})
    strict_convergence_assessment = (
        strict_convergence_assessment if isinstance(strict_convergence_assessment, dict) else {}
    )
    spline_bridge = data.get("spline_bridge", {})
    spline_bridge = spline_bridge if isinstance(spline_bridge, dict) else {}
    scale = data.get("vmec_free_boundary_scale", {})
    scale = scale if isinstance(scale, dict) else {}
    native_vector_profile = data.get("native_spline_vector_residual_profile", {})
    native_vector_profile = native_vector_profile if isinstance(native_vector_profile, dict) else {}
    rows: list[dict[str, Any]] = []
    for backend_name, backend in sorted((data.get("backends", {}) or {}).items()):
        if isinstance(backend, dict):
            rows.append(
                _summary_row(
                    path=path,
                    backend_name=str(backend_name),
                    backend=backend,
                    cfg=cfg,
                    projection=projection,
                    resolution_deck=resolution_deck,
                    resolution_guardrail=resolution_guardrail,
                    strict_convergence_assessment=strict_convergence_assessment,
                    spline_bridge=spline_bridge,
                    vmec_free_boundary_scale=scale,
                    native_spline_vector_residual_profile=native_vector_profile,
                )
            )
    if not rows and (projection or resolution_deck or scale or native_vector_profile):
        rows.append(
            _summary_row(
                path=path,
                backend_name="preflight_diagnostics",
                backend={"status": "preflight_only"},
                cfg=cfg,
                projection=projection,
                resolution_deck=resolution_deck,
                resolution_guardrail=resolution_guardrail,
                strict_convergence_assessment=strict_convergence_assessment,
                spline_bridge=spline_bridge,
                vmec_free_boundary_scale=scale,
                native_spline_vector_residual_profile=native_vector_profile,
            )
        )
    return rows


def _vmec2000_row_payload(row: Any) -> dict[str, Any]:
    total = float(row.fsqr) + float(row.fsqz) + float(row.fsql)
    return {
        "it": int(row.it),
        "fsqr": float(row.fsqr),
        "fsqz": float(row.fsqz),
        "fsql": float(row.fsql),
        "total": total,
        "max_component": float(max(float(row.fsqr), float(row.fsqz), float(row.fsql))),
    }


def _vmec2000_stage_payload(stage: Any) -> dict[str, Any]:
    rows = list(getattr(stage, "rows", []) or [])
    totals = [float(row.fsqr) + float(row.fsqz) + float(row.fsql) for row in rows]
    last = rows[-1] if rows else None
    return {
        "ns": int(getattr(stage, "ns", -1)),
        "niter": int(getattr(stage, "niter", -1)),
        "ftolv": _finite_float(getattr(stage, "ftolv", None)),
        "iteration_row_count": len(rows),
        "min_total": None if not totals else float(np.nanmin(np.asarray(totals, dtype=float))),
        "last_row": None if last is None else _vmec2000_row_payload(last),
    }


def _vmec2000_partial_payload_from_threed1(path: Path) -> dict[str, Any]:
    stages = _parse_vmec2000_threed1(path)
    rows = [row for stage in stages for row in stage.rows]
    totals = [float(row.fsqr) + float(row.fsqz) + float(row.fsql) for row in rows]
    last = rows[-1] if rows else None
    progress_phase = "force_iterations" if rows else "startup_or_pre_iteration_output"
    stage_ftol = _finite_float(getattr(stages[-1], "ftolv", None)) if stages else None
    return {
        "threed1": path,
        **_file_status_payload(path, prefix="threed1"),
        "progress_phase": progress_phase,
        "force_rows_started": bool(rows),
        "stage_summaries": [_vmec2000_stage_payload(stage) for stage in stages],
        "iteration_row_count": len(rows),
        "last_row": None if last is None else _vmec2000_row_payload(last),
        "min_total": None if not totals else float(np.nanmin(np.asarray(totals, dtype=float))),
        "history": {
            "fsq_component_sum_tail_projection": _vmec2000_tail_projection(rows),
            "fsq_component_tail_projection_by_component": _vmec2000_component_tail_projections(rows),
        },
        "tail_plateau": _tail_plateau_payload(rows, stage_ftol=stage_ftol),
        "final_max_component": None
        if last is None
        else float(max(float(last.fsqr), float(last.fsqz), float(last.fsql))),
        "vacuum_grid_exceeded_count": sum(
            1
            for line in path.read_text(errors="replace").splitlines()
            if "Plasma Boundary exceeded Vacuum Grid Size" in line
        ),
    }


def _looks_like_vmec2000_partial_payload(payload: dict[str, Any]) -> bool:
    return any(key in payload for key in ("stage_summaries", "last_row", "tail_rows", "progress_phase"))


def rows_from_vmec2000_partial(path: Path) -> list[dict[str, Any]]:
    if path.name == PARTIAL_VMEC2000_NAME or path.suffix.lower() == ".json":
        backend = json.loads(path.read_text())
        if not isinstance(backend, dict) or not _looks_like_vmec2000_partial_payload(backend):
            return []
        if isinstance(backend, dict) and backend.get("progress_phase") is None:
            threed1_matches = sorted((path.parent / "vmec2000_mgrid").glob("threed1*"))
            if threed1_matches:
                from_threed1 = _vmec2000_partial_payload_from_threed1(threed1_matches[0])
                for key in (
                    "progress_phase",
                    "force_rows_started",
                    "threed1_size_bytes",
                    "threed1_mtime_unix_s",
                ):
                    backend.setdefault(key, from_threed1.get(key))
    else:
        backend = _vmec2000_partial_payload_from_threed1(path)
    if not isinstance(backend, dict):
        return []
    return [
        _summary_row(
            path=path,
            backend_name="vmec2000_mgrid",
            backend=backend,
            cfg={},
            projection={},
            status="running_partial",
        )
    ]


def rows_from_launcher_log(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(errors="replace")
    backend = _vmec_style_log_payload(path)
    return [
        _summary_row(
            path=path,
            backend_name=str(backend.get("backend_name", "vmec_jax_live")),
            backend=backend,
            cfg=_parse_square_build_config(text),
            projection={},
            status="running_partial",
        )
    ]


def rows_from_source(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.name == FINAL_REPORT_NAME:
        return rows_from_profile(path)
    if path.name == PARTIAL_VMEC2000_NAME or path.name.startswith("threed1"):
        return rows_from_vmec2000_partial(path)
    if path.suffix.lower() == ".json":
        return rows_from_vmec2000_partial(path)
    if path.name == LAUNCHER_LOG_NAME:
        return rows_from_launcher_log(path)
    return []


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _print_rows(rows: list[dict[str, Any]], *, markdown: bool, fields: list[str]) -> None:
    if markdown:
        print("| " + " | ".join(fields) + " |")
        print("| " + " | ".join("---" for _ in fields) + " |")
        for row in rows:
            print("| " + " | ".join(_format_value(row.get(field)) for field in fields) + " |")
        return
    print("\t".join(fields))
    for row in rows:
        print("\t".join(_format_value(row.get(field)) for field in fields))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    rows: list[dict[str, Any]] = []
    for path in _profile_paths(list(args.paths)):
        rows.extend(rows_from_source(path))
    fields = [
        "case",
        "backend",
        "backend_role",
        "strict_evidence_status",
        "strict_evidence_blockers",
        "status",
        "progress_phase",
        "force_rows_started",
        "launcher_log_size_bytes",
        "launcher_log_mtime_unix_s",
        "threed1_size_bytes",
        "threed1_mtime_unix_s",
        "initial_jacobian_changed_sign",
        "vacuum_pressure_turn_on_iter",
        "mpol",
        "ntor",
        "ns",
        "nzeta",
        "resolution_guardrail_status",
        "resolution_guardrail_action",
        "resolution_guardrail_requested_vs_effective_changed",
        "requested_mpol",
        "requested_ntor",
        "requested_ntheta",
        "requested_nzeta",
        "effective_mpol",
        "effective_ntor",
        "effective_ntheta",
        "effective_nzeta",
        "resolution_guardrail_mode_deck_auto_bumped",
        "resolution_guardrail_ntheta_auto_bumped",
        "resolution_guardrail_nzeta_auto_bumped",
        "validated_minimum_mpol",
        "validated_minimum_ntor",
        "validated_minimum_ntheta",
        "validated_minimum_nzeta",
        "nzeta_auto",
        "recommended_nzeta",
        "nvacskip",
        "solver_mode",
        "strict_backtracking",
        "jax_hot_restart_requested_count",
        "jax_hot_restart_executed_count",
        "jax_hot_restart_iters",
        "jax_hot_restart_policy",
        "jax_hot_restart_always",
        "jax_initial_restart_wout",
        "jax_hot_restart_stopped_after_strict",
        "jax_hot_restart_last_status",
        "jax_hot_restart_last_component_max",
        "jax_hot_restart_resume_present_last",
        "jax_hot_restart_resume_freeb_runtime_present_last",
        "jax_hot_restart_resume_freeb_runtime_present_any",
        "jax_hot_restart_resume_freeb_model_last",
        "multigrid_resume_enabled",
        "multigrid_resume_env_default",
        "multigrid_resume_applied_any",
        "multigrid_resume_applied_last",
        "multigrid_resume_freeb_runtime_present_any",
        "multigrid_resume_freeb_runtime_present_last",
        "multigrid_resume_freeb_model_last",
        "freeb_jax_nestor_operator",
        "freeb_jax_nestor_jit_operator",
        "freeb_edge_control_projection_status",
        "freeb_edge_control_projection_requested",
        "freeb_edge_control_projection_basis",
        "freeb_edge_control_projection_control_count",
        "freeb_edge_control_projection_rcond",
        "freeb_edge_control_projection_ridge",
        "freeb_edge_control_projection_trust_radius",
        "freeb_edge_control_projection_update_mode",
        "freeb_edge_control_projection_apply_count",
        "freeb_edge_control_projection_delta_projection_count",
        "freeb_edge_control_projection_coordinate_update_count",
        "freeb_edge_control_projection_native_coordinate_update_count",
        "freeb_edge_control_projection_native_velocity_reset_count",
        "freeb_edge_control_projection_native_resync_count",
        "freeb_edge_control_projection_native_force_l2",
        "freeb_edge_control_projection_native_velocity_l2",
        "freeb_edge_control_projection_native_update_l2",
        "freeb_edge_control_projection_native_trust_scale",
        "freeb_edge_control_projection_native_decoded_edge_update_l2",
        "freeb_edge_control_projection_native_decoded_edge_update_linf",
        "freeb_edge_control_projection_native_source_update_l2",
        "freeb_edge_control_projection_native_source_update_residual_l2",
        "freeb_edge_control_projection_native_source_update_residual_linf",
        "freeb_edge_control_projection_native_source_update_residual_rel",
        "freeb_edge_control_projection_native_source_update_captured_fraction",
        "freeb_edge_control_projection_native_state_status",
        "freeb_edge_control_projection_native_state_schema",
        "freeb_edge_control_projection_native_state_mode",
        "freeb_edge_control_projection_native_state_unknown_l2",
        "freeb_edge_control_projection_native_state_unknown_linf",
        "freeb_edge_control_projection_native_state_decoded_edge_linf",
        "freeb_edge_control_projection_native_state_fit_residual_linf",
        "freeb_edge_control_projection_native_state_fit_residual_rel",
        "freeb_edge_control_projection_native_spline_state_status",
        "freeb_edge_control_projection_native_spline_state_schema",
        "freeb_edge_control_projection_native_spline_state_mode",
        "freeb_edge_control_projection_native_spline_state_full_edge_size",
        "freeb_edge_control_projection_native_spline_state_reduced_unknown_size",
        "freeb_edge_control_projection_native_spline_state_unknown_linf",
        "freeb_edge_control_projection_native_unknown_status",
        "freeb_edge_control_projection_native_unknown_schema",
        "freeb_edge_control_projection_native_unknown_size",
        "freeb_edge_control_projection_native_unknown_full_vmec_size",
        "freeb_edge_control_projection_native_unknown_interior_size",
        "freeb_edge_control_projection_native_unknown_edge_control_size",
        "freeb_edge_control_projection_native_unknown_removed_edge_dofs",
        "freeb_edge_control_projection_native_unknown_reduction_fraction",
        "freeb_edge_control_projection_native_unknown_edge_control_linf",
        "freeb_edge_control_projection_native_unknown_reconstruction_residual_linf",
        "freeb_edge_control_projection_native_unknown_reconstruction_residual_rel",
        "native_spline_vector_residual_profile_status",
        "native_spline_vector_residual_profile_native_unknown_size",
        "native_spline_vector_residual_profile_full_vmec_size",
        "native_spline_vector_residual_profile_removed_edge_dofs",
        "native_spline_vector_residual_profile_reduction_fraction",
        "native_spline_vector_residual_profile_decode_parity_linf",
        "native_spline_vector_residual_profile_projected_residual_parity_linf",
        "native_spline_vector_residual_profile_projected_residual_parity_rel",
        "native_spline_vector_residual_profile_projected_residual_jvp_wall_s",
        "native_spline_vector_residual_profile_jvp_linf",
        "native_spline_vector_residual_profile_matrix_free_status",
        "native_spline_vector_residual_profile_matrix_free_method",
        "native_spline_vector_residual_profile_matrix_free_vjp_wall_s",
        "native_spline_vector_residual_profile_matrix_free_normal_matvec_wall_s",
        "native_spline_vector_residual_profile_matrix_free_cg_step_wall_s",
        "native_spline_vector_residual_profile_matrix_free_linear_maxiter",
        "native_spline_vector_residual_profile_matrix_free_residual_reduction_factor",
        "native_spline_vector_residual_profile_matrix_free_step_l2",
        "native_spline_vector_residual_profile_matrix_free_next_action",
        "native_spline_vector_residual_profile_next_action",
        "freeb_edge_control_projection_zero_velocity_count",
        "freeb_edge_control_projection_state_residual_status",
        "freeb_edge_control_projection_state_residual_linf",
        "freeb_edge_control_projection_state_residual_rms",
        "freeb_edge_control_projection_state_residual_rel",
        "freeb_edge_control_projection_state_captured_fraction",
        "freeb_edge_control_projection_state_coordinates_status",
        "freeb_edge_control_projection_state_coordinate_linf",
        "freeb_edge_control_projection_state_coordinate_l2",
        "freeb_edge_control_projection_state_reconstruction_residual_linf",
        "freeb_edge_control_projection_state_reconstruction_residual_rms",
        "freeb_edge_control_projection_state_reconstruction_residual_rel",
        "freeb_edge_control_projection_reduced_unknown_status",
        "freeb_edge_control_projection_reduced_unknown_size",
        "freeb_edge_control_projection_full_edge_size",
        "freeb_edge_control_projection_unknown_reduction_fraction",
        "freeb_edge_control_projection_unknown_decoded_residual_linf",
        "freeb_edge_control_projection_unknown_decoded_residual_rel",
        "freeb_edge_control_projection_update_direction_status",
        "freeb_edge_control_projection_update_direction_linf",
        "freeb_edge_control_projection_update_direction_rms",
        "freeb_edge_control_projection_update_direction_rel",
        "freeb_edge_control_projection_update_direction_captured_fraction",
        "freeb_edge_control_projection_update_direction_trust_scale",
        "freeb_edge_control_projection_force_direction_status",
        "freeb_edge_control_projection_force_direction_linf",
        "freeb_edge_control_projection_force_direction_rms",
        "freeb_edge_control_projection_force_direction_rel",
        "freeb_edge_control_projection_force_direction_captured_fraction",
        "freeb_edge_control_projection_force_direction_trust_scale",
        "freeb_edge_control_projection_force_capture_status",
        "freeb_edge_control_projection_force_capture_next_basis",
        "freeb_edge_control_projection_reduced_update_status",
        "freeb_edge_control_projection_reduced_update_size",
        "freeb_edge_control_projection_full_update_size",
        "freeb_edge_control_projection_reduced_update_linf",
        "freeb_edge_control_projection_reduced_update_decoded_residual_linf",
        "freeb_edge_control_projection_reduced_update_decoded_residual_rel",
        "freeb_edge_control_projection_reduced_update_captured_fraction",
        "freeb_edge_control_projection_reduced_force_status",
        "freeb_edge_control_projection_reduced_force_size",
        "freeb_edge_control_projection_reduced_force_linf",
        "freeb_edge_control_projection_reduced_force_decoded_residual_linf",
        "freeb_edge_control_projection_reduced_force_decoded_residual_rel",
        "freeb_edge_control_projection_reduced_force_captured_fraction",
        "free_boundary_jax_nestor_operator_applied",
        "free_boundary_jax_nestor_operator_reason",
        "free_boundary_jax_nestor_operator_jitted",
        "free_boundary_jax_nestor_operator_cache_hit",
        "free_boundary_jax_nestor_operator_time_s",
        "side_power",
        "corner_power",
        "max_iter",
        "stage_count",
        "stage_ns_array",
        "stage_niter_array",
        "stage_ftol_array",
        "stage_budget_total",
        "stage_budget_final",
        "current_stage_index",
        "current_stage_niter",
        "current_stage_ftol",
        "current_stage_last_iter",
        "current_stage_iteration_row_count",
        "remaining_stage_budget",
        "remaining_total_stage_budget",
        "requested_ftol",
        "strict_gap",
        "remaining_iterations",
        "next_action",
        "strict_assessment_full_fourier_status",
        "strict_assessment_reduced_control_status",
        "strict_assessment_solver_native_spline_status",
        "strict_assessment_solver_native_spline_edge_controls",
        "strict_assessment_solver_native_spline_scope",
        "strict_assessment_full_native_spline_state_required",
        "strict_assessment_vmec2000_fix_fourier_bottleneck",
        "spline_bridge_status",
        "spline_bridge_nonlinear_solver_boundary_basis",
        "spline_bridge_solver_native_spline_controls",
        "spline_bridge_solver_native_spline_edge_controls",
        "spline_bridge_solver_native_spline_scope",
        "spline_bridge_solver_edge_control_projection_enabled",
        "spline_bridge_solver_edge_control_update_mode",
        "spline_bridge_can_reduce_input_shape_dofs",
        "spline_bridge_can_project_free_boundary_edge_updates",
        "spline_bridge_can_reduce_free_boundary_edge_dofs",
        "spline_bridge_can_reduce_full_nonlinear_solver_dofs",
        "spline_bridge_requires_native_spline_state_for_reduced_nonlinear_dofs",
        "recommended_followup_profile_kind",
        "recommended_followup_reason",
        "resolution_deck_status",
        "resolution_deck_reasons",
        "vmec_scale_status",
        "vmec_scale_phiedge",
        "vmec_scale_external_r_bphi_rms",
        "vmec_scale_proxy_over_external_r_bphi_rms",
        "vmec_scale_suggested_phiedge",
        "vmec_scale_suggested_phiedge_over_current",
        "accepted_provider_parity_status",
        "accepted_provider_parity_sample",
        "accepted_provider_parity_field_diff_rms_rel",
        "accepted_provider_parity_bnormal_diff_rms_rel",
        "accepted_provider_parity_field_lt_5pct",
        "accepted_provider_parity_bnormal_lt_10pct",
        "boundary_mode_count",
        "boundary_recommended_nzeta",
        "max_boundary_projection_error",
        "boundary_proj_max",
        "boundary_proj_rel",
        "boundary_coeff_delta_l2",
        "boundary_coeff_delta_linf",
        "boundary_coeff_delta_rel",
        "boundary_sample_displacement_rms",
        "boundary_sample_displacement_max",
        "boundary_sample_displacement_rel",
        "boundary_control_projection_status",
        "boundary_control_projection_residual_rel",
        "boundary_control_projection_captured_fraction",
        "boundary_control_projection_radius_delta",
        "boundary_control_projection_state_coordinate_linf",
        "boundary_control_projection_state_reconstruction_residual_rel",
        "boundary_control_projection_stellarator_residual_rel",
        "boundary_control_projection_stellarator_captured_fraction",
        "boundary_control_projection_stellarator_control_count",
        "boundary_control_projection_stellarator_state_reconstruction_residual_rel",
        "boundary_control_projection_full_residual_rel",
        "boundary_control_projection_full_captured_fraction",
        "boundary_control_projection_full_control_count",
        "boundary_control_projection_full_state_reconstruction_residual_rel",
        "final_iter",
        "final_total",
        "final_max_component",
        "final_fsqr",
        "final_fsqz",
        "final_fsql",
        "limiting_component",
        "fsqr_strict_gap",
        "fsqz_strict_gap",
        "fsql_strict_gap",
        "strict_components_met",
        "boundary_condition_mode",
        "coil_bnormal_role",
        "production_candidate",
        "promotion_blockers",
        "virtual_casing_required",
        "virtual_casing_available",
        "best_total",
        "best_component_max",
        "returned_best_scored_state",
        "best_scored_full_boundary_count",
        "best_scored_fresh_boundary_count",
        "best_scored_drift_restart_count",
        "best_scored_drift_streak",
        "best_scored_drift_last_restart_iter",
        "best_scored_drift_last_ratio",
        "final_residual_recomputed_on_accepted_state",
        "fresh_convergence_gate",
        "fresh_convergence_rechecks",
        "fresh_convergence_rejects",
        "fresh_convergence_failures",
        "freeb_convergence_blocked_count",
        "update_delta_rms",
        "update_delta_to_velocity_rms_ratio",
        "dt_eff_last",
        "dt_eff_min",
        "time_step_last",
        "trial_ratio_last",
        "trial_ratio_min",
        "trial_ratio_max",
        "trial_ratio_mean",
        "step_status_counts",
        "step_momentum_count",
        "step_rejected_count",
        "step_restart_count",
        "restart_path_counts",
        "restart_path_trial_rejected_count",
        "restart_path_momentum_accept_count",
        "freeb_full_update_count",
        "nestor_reuse_count",
        "nestor_reuse_last",
        "nestor_source_reuse_count",
        "nestor_source_reuse_last",
        "nestor_provider_source_reuse_allowed_last",
        "nestor_sample_time_last",
        "nestor_sample_time_mean",
        "nestor_sample_time_max",
        "nestor_solve_time_last",
        "nestor_solve_time_mean",
        "nestor_solve_time_max",
        "nestor_trial_reuse_count",
        "nestor_trial_failed_count",
        "nestor_trial_sample_time_mean",
        "nestor_trial_sample_time_max",
        "nestor_trial_solve_time_mean",
        "nestor_trial_solve_time_max",
        "include_edge_count",
        "include_edge_last",
        "anderson_pressure_enabled",
        "anderson_pressure_applied_count",
        "anderson_pressure_last_theta",
        "bad_jacobian_count",
        "bnormal_rms_last",
        "bnormal_rms_min",
        "virtual_casing_status",
        "virtual_casing_grid_adequacy_status",
        "virtual_casing_surface_ntheta",
        "virtual_casing_surface_nzeta",
        "virtual_casing_quad_ntheta",
        "virtual_casing_quad_nzeta",
        "virtual_casing_quad_factor_theta",
        "virtual_casing_quad_factor_zeta",
        "virtual_casing_external_bnormal_residual_rms",
        "virtual_casing_external_bnormal_residual_max",
        "virtual_casing_pressure_balance_rms",
        "virtual_casing_pressure_balance_max",
        "virtual_casing_required_external_b_rms",
        "virtual_casing_target_external_b_rms",
        "virtual_casing_wall_s",
        "tail_decay_factor",
        "iters_to_1e-12_est",
        "strict_tail_projection_status",
        "strict_tail_projection_target",
        "strict_tail_component_statuses",
        "strict_tail_limiting_component",
        "strict_tail_limiting_component_status",
        "strict_tail_limiting_component_factor",
        "strict_tail_limiting_iters_to_1e-12_est",
        "fsqr_tail_decay_factor",
        "fsqz_tail_decay_factor",
        "fsql_tail_decay_factor",
        "fsqr_iters_to_1e-12_est",
        "fsqz_iters_to_1e-12_est",
        "fsql_iters_to_1e-12_est",
        "tail_plateau_status",
        "tail_plateau_window",
        "tail_total_rel_span",
        "tail_last_over_min",
        "wall_s",
        "vacuum_grid_exceeded_count",
    ]
    if args.csv is not None:
        _write_csv(args.csv, rows, fields)
    _print_rows(rows, markdown=bool(args.markdown), fields=fields)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
