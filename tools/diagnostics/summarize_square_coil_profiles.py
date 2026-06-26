#!/usr/bin/env python
"""Summarize square-coil free-boundary backend profile JSON files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any


DEFAULT_GLOB = "results/square_coil_freeb_backend_profile_*/square_coil_free_boundary_backend_profile.json"


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
            candidate = path / "square_coil_free_boundary_backend_profile.json"
            if candidate.exists():
                out.append(candidate)
                continue
            out.extend(sorted(path.glob("**/square_coil_free_boundary_backend_profile.json")))
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


def rows_from_profile(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    cfg = data.get("configuration", {})
    projection = data.get("boundary_projection", {})
    projection = projection if isinstance(projection, dict) else {}
    rows: list[dict[str, Any]] = []
    for backend_name, backend in sorted((data.get("backends", {}) or {}).items()):
        if not isinstance(backend, dict):
            continue
        requested_ftol = _finite_float(cfg.get("ftol"))
        if backend_name == "vmec2000_mgrid":
            final_total, best_total, final_iter = _vmec2000_total(backend)
            final_max_component = _vmec2000_final_max_component(backend)
        else:
            final_total, best_total, final_iter = _jax_total(backend)
            final_max_component = _jax_final_max_component(backend)
        rows.append(
            {
                "case": path.parent.name.replace("square_coil_freeb_backend_profile_", ""),
                "backend": backend_name,
                "status": backend.get("status"),
                "mpol": cfg.get("mpol"),
                "ntor": cfg.get("ntor"),
                "ns": cfg.get("ns"),
                "nzeta": cfg.get("nzeta"),
                "nzeta_auto": cfg.get("nzeta_auto"),
                "recommended_nzeta": cfg.get("recommended_nzeta"),
                "nvacskip": cfg.get("nvacskip"),
                "solver_mode": cfg.get("solver_mode"),
                "max_iter": cfg.get("max_iter"),
                "requested_ftol": requested_ftol,
                "boundary_mode_count": projection.get("mode_count"),
                "boundary_recommended_nzeta": projection.get("recommended_nzeta"),
                "boundary_proj_max": _finite_float(projection.get("max_abs_component_error")),
                "boundary_proj_rel": _finite_float(projection.get("max_abs_component_error_rel")),
                "final_iter": final_iter,
                "final_total": final_total,
                "final_max_component": final_max_component,
                "strict_components_met": _strict_components_met(final_max_component, requested_ftol),
                "best_total": best_total,
                "dt_eff_last": _stat(backend, "dt_eff_stats", "last"),
                "dt_eff_min": _stat(backend, "dt_eff_stats", "min"),
                "time_step_last": _stat(backend, "time_step_stats", "last"),
                "freeb_full_update_count": _stat(backend, "freeb_full_update_stats", "sum"),
                "anderson_pressure_enabled": backend.get("free_boundary_anderson_pressure_enabled"),
                "anderson_pressure_applied_count": _stat(
                    backend, "freeb_anderson_pressure_applied_stats", "sum"
                ),
                "anderson_pressure_last_theta": _finite_float(
                    backend.get("free_boundary_anderson_pressure_last_theta")
                ),
                "bad_jacobian_count": _stat(backend, "bad_jacobian_stats", "sum"),
                "bnormal_rms_last": _stat(backend, "freeb_nestor_bnormal_rms_stats", "last"),
                "bnormal_rms_min": _stat(backend, "freeb_nestor_bnormal_rms_stats", "min"),
                "tail_decay_factor": _tail_projection(backend, "per_iter_factor"),
                "iters_to_1e-12_est": _tail_projection(backend, "", target=1.0e-12),
                "wall_s": _finite_float(backend.get("wall_s")),
                "vacuum_grid_exceeded_count": backend.get("vacuum_grid_exceeded_count"),
            }
        )
    return rows


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
        rows.extend(rows_from_profile(path))
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
        "max_iter",
        "requested_ftol",
        "boundary_mode_count",
        "boundary_recommended_nzeta",
        "boundary_proj_max",
        "boundary_proj_rel",
        "final_iter",
        "final_total",
        "final_max_component",
        "strict_components_met",
        "best_total",
        "dt_eff_last",
        "dt_eff_min",
        "time_step_last",
        "freeb_full_update_count",
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
