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


DEFAULT_GLOB = "results/square_coil_freeb_backend_profile_*/square_coil_free_boundary_backend_profile.json"
FINAL_REPORT_NAME = "square_coil_free_boundary_backend_profile.json"
PARTIAL_VMEC2000_NAME = "_partial_vmec2000_payload.json"


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


def _case_name(path: Path) -> str:
    for parent in (path.parent, path.parent.parent):
        name = parent.name
        if name.startswith("square_coil_freeb_backend_profile_"):
            return name.replace("square_coil_freeb_backend_profile_", "")
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
    return cfg


def _stat(backend: dict[str, Any], history_key: str, stat_key: str) -> float | None:
    history = backend.get("history")
    if not isinstance(history, dict):
        return None
    stats = history.get(history_key)
    if not isinstance(stats, dict):
        return None
    return _finite_float(stats.get(stat_key))


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
        "monotone_decrease_fraction": None,
        "per_iter_log_slope": None,
        "per_iter_factor": None,
        "estimated_additional_iterations_to_target": {},
    }
    if len(pairs) < 2:
        return out
    iters = np.asarray([pair[0] for pair in pairs], dtype=float)
    totals = np.asarray([pair[1] for pair in pairs], dtype=float)
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
            estimates[key] = int(np.ceil(np.log(target / last) / slope))
        else:
            estimates[key] = None
    out["estimated_additional_iterations_to_target"] = estimates
    return out


def _summary_row(
    *,
    path: Path,
    backend_name: str,
    backend: dict[str, Any],
    cfg: dict[str, Any],
    projection: dict[str, Any],
    status: str | None = None,
) -> dict[str, Any]:
    case = _case_name(path)
    cfg = {**_config_from_case_name(case), **cfg}
    backend_for_projection = backend
    if backend_name == "vmec2000_mgrid" and not isinstance(backend.get("history"), dict):
        tail_rows = backend.get("tail_rows")
        if isinstance(tail_rows, list) and tail_rows:
            backend_for_projection = {
                **backend,
                "history": {
                    "fsq_component_sum_tail_projection": _vmec2000_tail_projection(tail_rows),
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
    if backend_name == "vmec2000_mgrid":
        final_total, best_total, final_iter = _vmec2000_total(backend)
        final_max_component = _vmec2000_final_max_component(backend) or _finite_float(
            backend.get("final_max_component")
        )
    else:
        final_total, best_total, final_iter = _jax_total(backend)
        final_max_component = _jax_final_max_component(backend)
    return {
        "case": case,
        "backend": backend_name,
        "status": status if status is not None else backend.get("status"),
        "mpol": cfg.get("mpol"),
        "ntor": cfg.get("ntor"),
        "ns": cfg.get("ns"),
        "nzeta": cfg.get("nzeta"),
        "nzeta_auto": cfg.get("nzeta_auto"),
        "recommended_nzeta": cfg.get("recommended_nzeta"),
        "nvacskip": cfg.get("nvacskip"),
        "solver_mode": cfg.get("solver_mode"),
        "side_power": cfg.get("side_power"),
        "corner_power": cfg.get("corner_power"),
        "max_iter": cfg.get("max_iter"),
        "requested_ftol": requested_ftol,
        "boundary_mode_count": projection.get("mode_count"),
        "boundary_recommended_nzeta": projection.get("recommended_nzeta"),
        "max_boundary_projection_error": cfg.get("max_boundary_projection_error"),
        "boundary_proj_max": _finite_float(projection.get("max_abs_component_error")),
        "boundary_proj_rel": _finite_float(projection.get("max_abs_component_error_rel")),
        "final_iter": final_iter,
        "final_total": final_total,
        "final_max_component": final_max_component,
        "strict_components_met": _strict_components_met(final_max_component, requested_ftol),
        "best_total": best_total,
        "returned_best_scored_state": backend.get("returned_best_scored_state"),
        "best_scored_full_boundary_count": backend.get("best_scored_full_boundary_count"),
        "best_scored_fresh_boundary_count": backend.get("best_scored_fresh_boundary_count"),
        "final_residual_recomputed_on_accepted_state": backend.get(
            "final_residual_recomputed_on_accepted_state"
        ),
        "fresh_convergence_gate": backend.get("free_boundary_fresh_convergence_gate"),
        "fresh_convergence_rechecks": backend.get("free_boundary_fresh_convergence_recheck_count"),
        "fresh_convergence_rejects": backend.get("free_boundary_fresh_convergence_reject_count"),
        "fresh_convergence_failures": backend.get("free_boundary_fresh_convergence_failed_count"),
        "freeb_convergence_blocked_count": backend.get("free_boundary_convergence_blocked_count"),
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
        "tail_decay_factor": _tail_projection(backend_for_projection, "per_iter_factor"),
        "iters_to_1e-12_est": _tail_projection(backend_for_projection, "", target=1.0e-12),
        "wall_s": _finite_float(backend.get("wall_s")),
        "vacuum_grid_exceeded_count": backend.get("vacuum_grid_exceeded_count"),
    }


def rows_from_profile(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    cfg = data.get("configuration", {})
    cfg = cfg if isinstance(cfg, dict) else {}
    projection = data.get("boundary_projection", {})
    projection = projection if isinstance(projection, dict) else {}
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
    return {
        "threed1": path,
        "stage_summaries": [_vmec2000_stage_payload(stage) for stage in stages],
        "iteration_row_count": len(rows),
        "last_row": None if last is None else _vmec2000_row_payload(last),
        "min_total": None if not totals else float(np.nanmin(np.asarray(totals, dtype=float))),
        "history": {
            "fsq_component_sum_tail_projection": _vmec2000_tail_projection(rows),
        },
        "final_max_component": None
        if last is None
        else float(max(float(last.fsqr), float(last.fsqz), float(last.fsql))),
        "vacuum_grid_exceeded_count": sum(
            1
            for line in path.read_text(errors="replace").splitlines()
            if "Plasma Boundary exceeded Vacuum Grid Size" in line
        ),
    }


def rows_from_vmec2000_partial(path: Path) -> list[dict[str, Any]]:
    if path.name == PARTIAL_VMEC2000_NAME:
        backend = json.loads(path.read_text())
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


def rows_from_source(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.name == FINAL_REPORT_NAME:
        return rows_from_profile(path)
    if path.name == PARTIAL_VMEC2000_NAME or path.name.startswith("threed1"):
        return rows_from_vmec2000_partial(path)
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
        "status",
        "mpol",
        "ntor",
        "ns",
        "nzeta",
        "nzeta_auto",
        "recommended_nzeta",
        "nvacskip",
        "solver_mode",
        "side_power",
        "corner_power",
        "max_iter",
        "requested_ftol",
        "boundary_mode_count",
        "boundary_recommended_nzeta",
        "max_boundary_projection_error",
        "boundary_proj_max",
        "boundary_proj_rel",
        "final_iter",
        "final_total",
        "final_max_component",
        "strict_components_met",
        "best_total",
        "returned_best_scored_state",
        "best_scored_full_boundary_count",
        "best_scored_fresh_boundary_count",
        "final_residual_recomputed_on_accepted_state",
        "fresh_convergence_gate",
        "fresh_convergence_rechecks",
        "fresh_convergence_rejects",
        "fresh_convergence_failures",
        "freeb_convergence_blocked_count",
        "dt_eff_last",
        "dt_eff_min",
        "time_step_last",
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
        "tail_decay_factor",
        "iters_to_1e-12_est",
        "wall_s",
        "vacuum_grid_exceeded_count",
    ]
    if args.csv is not None:
        _write_csv(args.csv, rows, fields)
    _print_rows(rows, markdown=bool(args.markdown), fields=fields)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
