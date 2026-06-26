#!/usr/bin/env python
"""Profile square-coil free-boundary solves through direct, mgrid, and VMEC2000 paths."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.toroidal_stellarator_mirror_hybrid_square_coils_free_boundary import (
    ExampleConfig,
    _boundary_projection_payload as _example_boundary_projection_payload,
    _case_label,
    _run_budget,
    _square_axis_sample_kwargs,
    _stage_values,
    build_square_coils,
    make_free_boundary_indata,
)
from vmec_jax.boundary import BoundaryCoeffs, boundary_from_indata
from vmec_jax.driver import run_free_boundary, write_wout_from_fixed_boundary_run
from vmec_jax.external_fields import build_coil_field_geometry, write_mgrid_from_coils
from vmec_jax.fourier import eval_fourier
from vmec_jax.free_boundary import _sample_external_boundary_arrays
from vmec_jax.solvers.free_boundary.validation import (
    free_boundary_promotion_status,
    virtual_casing_diagnostics_from_run,
)
from vmec_jax.namelist import write_indata
from vmec_jax.toroidal_hybrid import (
    SquareAxisSplineControls,
    evaluate_toroidal_hybrid_indata_boundary,
    recommend_square_axis_stellarator_mirror_hybrid_resolution,
    recommended_square_axis_nzeta,
    square_axis_free_boundary_edge_control_projection_payload,
    square_axis_resolution_deck_status,
    square_axis_spline_control_fourier_map_status,
    square_axis_spline_control_fourier_matrix,
    square_axis_spline_symmetric_control_basis,
)
from vmec_jax.vmec2000_exec import _parse_vmec2000_threed1, find_vmec2000_exec, run_xvmec2000


DEFAULT_OUTDIR = REPO_ROOT / "results" / "square_coil_freeb_backend_profile"
TINY = 1.0e-300
STRICT_COMPONENT_FTOL = 1.0e-12
LOOSE_COMPONENT_FTOL = 1.0e-8


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _log_step(message: str) -> None:
    print(f"[square-coil-profile] {message}", flush=True)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--beta-percent", type=float, default=0.0)
    p.add_argument("--mpol", type=int, default=ExampleConfig().mpol)
    p.add_argument("--ntor", type=int, default=ExampleConfig().ntor)
    p.add_argument("--ns", type=int, default=9)
    p.add_argument(
        "--nzeta",
        type=_parse_optional_positive_int,
        default=None,
        help="VMEC zeta grid size. Omit, 0, or 'auto' to use the square-axis recommendation for NTOR.",
    )
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--ftol", type=float, default=ExampleConfig().ftol)
    p.add_argument(
        "--solver-mode",
        default="parity",
        choices=("auto", "default", "parity", "accelerated"),
        help="vmec_jax solver mode for JAX backends; 'auto' leaves run_free_boundary policy unchanged.",
    )
    p.add_argument("--ns-array", default=None, help="Comma-separated VMEC multigrid NS_ARRAY override.")
    p.add_argument("--niter-array", default=None, help="Comma-separated VMEC multigrid NITER_ARRAY override.")
    p.add_argument("--ftol-array", default=None, help="Comma-separated VMEC multigrid FTOL_ARRAY override.")
    p.add_argument("--phiedge", type=float, default=None)
    p.add_argument("--delt", type=float, default=ExampleConfig().delt)
    p.add_argument("--activate-fsq", type=float, default=1.0e-3)
    p.add_argument("--nvacskip", type=int, default=1, help="Initial/floor NVACSKIP for VMEC free-boundary updates.")
    p.add_argument(
        "--axis-kind",
        default=ExampleConfig().plasma_axis_kind,
        choices=("spline", "superellipse", "control_spline"),
    )
    p.add_argument("--axis-corner-factor", type=float, default=1.14)
    p.add_argument(
        "--side-power",
        type=float,
        default=ExampleConfig().side_power,
        help="Square-axis side localization power; 1.0 is the low-Fourier production default.",
    )
    p.add_argument(
        "--corner-power",
        type=float,
        default=ExampleConfig().corner_power,
        help="Square-axis corner localization power; values above 1.0 are sharper stress cases.",
    )
    p.add_argument(
        "--max-boundary-projection-error",
        type=_parse_optional_positive_float,
        default=ExampleConfig().max_boundary_projection_error,
        help=(
            "Optional production gate on the Fourier boundary projection max component error. "
            "Use 'none' to keep diagnostic underresolved profiles runnable."
        ),
    )
    p.add_argument("--enforce-recommended-nzeta", action="store_true")
    p.add_argument(
        "--nstep",
        type=int,
        default=ExampleConfig().nstep,
        help="VMEC NSTEP diagnostic cadence; use 1 for strict VMEC2000 profiling.",
    )
    p.add_argument("--n-coils-per-side", type=int, default=4)
    p.add_argument("--coil-segments", type=int, default=96)
    p.add_argument(
        "--coil-chunk-size",
        type=_parse_optional_positive_int,
        default=512,
        help="Direct-coil sampling batch size. Use 0 or 'none' for the full cached JIT sampler.",
    )
    p.add_argument("--mgrid-nr", type=int, default=36)
    p.add_argument("--mgrid-nz", type=int, default=28)
    p.add_argument("--mgrid-nphi", type=int, default=None)
    p.add_argument("--mgrid-padding-fraction", type=float, default=0.75)
    p.add_argument("--mgrid-min-padding", type=float, default=0.35)
    p.add_argument("--skip-direct", action="store_true")
    p.add_argument("--skip-mgrid", action="store_true")
    p.add_argument("--run-vmec2000", action="store_true")
    p.add_argument("--vmec2000-exec", type=Path, default=None)
    p.add_argument("--vmec2000-timeout", type=float, default=600.0)
    p.add_argument("--jit-forces", action="store_true")
    p.add_argument(
        "--direct-static-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cache direct-coil geometry outside the solve. This is faster for CLI profiling but not differentiable.",
    )
    p.add_argument(
        "--jit-direct-sampler",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the cached JIT direct-coil sampler when --coil-chunk-size is 0/none.",
    )
    p.add_argument(
        "--direct-trial-bsqvac-resample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Resample direct-coil vacuum pressure on trial/backtracking boundaries. "
            "Disable to benchmark VMEC2000-style committed-pressure scoring."
        ),
    )
    p.add_argument(
        "--skip-provider-parity",
        action="store_true",
        help="Skip the initial-boundary direct-coil/generated-mgrid field parity diagnostic.",
    )
    p.add_argument(
        "--accepted-provider-parity",
        action="store_true",
        help=(
            "Also compare direct-coil and generated-mgrid fields on each accepted JAX backend LCFS. "
            "This requires writing the generated mgrid even for direct-backend profiles."
        ),
    )
    p.add_argument(
        "--return-best-scored-state",
        action="store_true",
        help="Return the lowest fresh free-boundary residual state if max_iter is exhausted.",
    )
    p.add_argument(
        "--jax-hot-restart-count",
        type=int,
        default=0,
        help=(
            "Number of final-grid vmec_jax hot-restart passes to run after the initial solve "
            "if strict 1e-12 force components are not reached."
        ),
    )
    p.add_argument(
        "--jax-hot-restart-iters",
        type=_parse_optional_positive_int,
        default=None,
        help="Per-pass hot-restart iteration budget. Omit/auto to reuse the final-grid NITER.",
    )
    p.add_argument(
        "--jax-hot-restart-policy",
        choices=("state", "freeb", "full"),
        default="freeb",
        help=(
            "State carried into hot restarts: accepted state only, accepted state plus free-boundary "
            "cadence/runtime state, or the full nonlinear controller resume state."
        ),
    )
    p.add_argument(
        "--jax-hot-restart-always",
        action="store_true",
        help="Run every requested hot-restart pass even if an earlier pass satisfies strict convergence.",
    )
    p.add_argument(
        "--jax-initial-restart-wout",
        type=Path,
        default=None,
        help=(
            "Optional existing wout_*.nc used to seed the first vmec_jax pass on the final grid. "
            "This avoids rerunning the full multigrid ladder when profiling strict hot restarts."
        ),
    )
    p.add_argument(
        "--freeb-anderson-pressure",
        action="store_true",
        help="Enable opt-in Anderson(1) mixing for free-boundary vacuum pressure in vmec_jax backends.",
    )
    p.add_argument(
        "--freeb-jax-nestor-operator",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use the experimental JAX-assembled NESTOR boundary operator. "
            "The profiler explicitly disables it unless this flag is set."
        ),
    )
    p.add_argument(
        "--freeb-jax-nestor-jit-operator",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="JIT-cache the experimental JAX NESTOR operator when it is enabled.",
    )
    p.add_argument(
        "--freeb-include-edge",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override VMEC_JAX_FREEB_INCLUDE_EDGE for backend A/B profiles.",
    )
    p.add_argument(
        "--freeb-dense-solve-mode",
        choices=("mode", "grid"),
        default=None,
        help="Override VMEC_JAX_FREEB_DENSE_SOLVE_MODE for the dense free-boundary solve.",
    )
    p.add_argument(
        "--freeb-experimental-fouri-matrix",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override VMEC_JAX_FREEB_EXPERIMENTAL_FOURI_MATRIX for NESTOR operator profiling.",
    )
    p.add_argument(
        "--freeb-add-analytic-bvec",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override VMEC_JAX_FREEB_ADD_ANALYTIC_BVEC for NESTOR operator profiling.",
    )
    p.add_argument(
        "--freeb-edge-control-projection",
        choices=("none", "square", "stellarator"),
        default="none",
        help=(
            "Constrain vmec_jax free-boundary LCFS edge motion to the selected square-axis spline-control "
            "basis. This is an opt-in research profile for long straight hybrid axes."
        ),
    )
    p.add_argument(
        "--freeb-edge-control-rcond",
        type=float,
        default=1.0e-12,
        help="Pseudo-inverse cutoff used by --freeb-edge-control-projection.",
    )
    p.add_argument(
        "--verbose-solver",
        action="store_true",
        help="Print VMEC-style vmec_jax iteration progress for long direct/mgrid backend profiles.",
    )
    p.add_argument(
        "--virtual-casing-diagnostics",
        action="store_true",
        help="Add optional finite-beta virtual-casing postsolve diagnostics for the direct-coil backend.",
    )
    p.add_argument(
        "--virtual-casing-quad-factor",
        type=int,
        default=2,
        help="Virtual-casing source quadrature multiplier for finite-beta diagnostics.",
    )
    p.add_argument(
        "--virtual-casing-chunk-size",
        type=_parse_virtual_casing_chunk_arg,
        default="auto",
        help="Chunk size passed to virtual_casing_jax compute_external_B_functional; use auto or none.",
    )
    p.add_argument(
        "--virtual-casing-target-chunk-size",
        type=_parse_virtual_casing_chunk_arg,
        default="auto",
        help="Target chunk size passed to virtual_casing_jax; use auto or none.",
    )
    p.add_argument(
        "--virtual-casing-pythonpath",
        type=Path,
        default=None,
        help=(
            "Optional path to a virtual_casing_jax source checkout. The profiler prepends it to "
            "sys.path and exports VMEC_JAX_VIRTUAL_CASING_JAX_PATH for finite-beta diagnostics."
        ),
    )
    p.add_argument(
        "--resolution-diagnostics-only",
        action="store_true",
        help=(
            "Write Fourier projection, NZETA, and mgrid compatibility diagnostics, "
            "then exit before coil/mgrid generation or equilibrium solves."
        ),
    )
    return p


def _parse_int_list(raw: str | None, *, name: str) -> tuple[int, ...] | None:
    if raw is None or str(raw).strip() == "":
        return None
    values = tuple(int(tok.strip()) for tok in str(raw).replace(";", ",").split(",") if tok.strip())
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    return values


def _parse_float_list(raw: str | None, *, name: str) -> tuple[float, ...] | None:
    if raw is None or str(raw).strip() == "":
        return None
    values = tuple(float(tok.strip()) for tok in str(raw).replace(";", ",").split(",") if tok.strip())
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    return values


def _parse_optional_positive_int(raw: str) -> int | None:
    value = str(raw).strip().lower()
    if value in {"", "0", "auto", "none", "null", "false", "no"}:
        return None
    parsed = int(value)
    if parsed <= 0:
        return None
    return parsed


def _parse_optional_positive_float(raw: str) -> float | None:
    value = str(raw).strip().lower()
    if value in {"", "0", "auto", "none", "null", "false", "no"}:
        return None
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive, finite, or 'none'")
    return parsed


def _parse_virtual_casing_chunk_arg(raw: str) -> int | str | None:
    value = str(raw).strip().lower()
    if value in {"", "auto"}:
        return "auto"
    if value in {"0", "none", "null", "false", "no"}:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("virtual-casing chunk sizes must be positive, auto, or none")
    return parsed


def _configure_virtual_casing_pythonpath(path: Path | None) -> str | None:
    """Prepend an optional virtual_casing_jax source path for this process."""

    if path is None:
        return None
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists() or not resolved.is_dir():
        raise ValueError(f"--virtual-casing-pythonpath must be an existing directory: {resolved}")
    text = str(resolved)
    if text not in sys.path:
        sys.path.insert(0, text)
    existing = os.environ.get("VMEC_JAX_VIRTUAL_CASING_JAX_PATH")
    parts = [part for part in (existing or "").split(os.pathsep) if part]
    if text not in parts:
        os.environ["VMEC_JAX_VIRTUAL_CASING_JAX_PATH"] = os.pathsep.join([text, *parts])
    return text


def _resolve_schedule(args: argparse.Namespace) -> tuple[tuple[int, ...], tuple[int, ...], tuple[float, ...]]:
    ns_array = _parse_int_list(args.ns_array, name="ns-array")
    niter_array = _parse_int_list(args.niter_array, name="niter-array")
    ftol_array = _parse_float_list(args.ftol_array, name="ftol-array")
    provided = [value is not None for value in (ns_array, niter_array, ftol_array)]
    if any(provided) and not all(provided):
        raise ValueError("--ns-array, --niter-array, and --ftol-array must be provided together")
    if ns_array is None or niter_array is None or ftol_array is None:
        return (int(args.ns),), (int(args.max_iter),), (float(args.ftol),)
    if not (len(ns_array) == len(niter_array) == len(ftol_array)):
        raise ValueError("--ns-array, --niter-array, and --ftol-array must have matching lengths")
    return ns_array, niter_array, ftol_array


def _tail_lines(path: Path | None, *, lines: int = 60) -> list[str]:
    if path is None or not Path(path).exists():
        return []
    return Path(path).read_text(errors="replace").splitlines()[-int(lines) :]


def _vacuum_grid_exceeded_count(path: Path | None) -> int:
    if path is None or not Path(path).exists():
        return 0
    return sum(
        1
        for line in Path(path).read_text(errors="replace").splitlines()
        if "Plasma Boundary exceeded Vacuum Grid Size" in line
    )


def _file_status_payload(path: Path | None, *, prefix: str) -> dict[str, Any]:
    if path is None or not Path(path).exists():
        return {f"{prefix}_size_bytes": None, f"{prefix}_mtime_unix_s": None}
    stat = Path(path).stat()
    return {f"{prefix}_size_bytes": int(stat.st_size), f"{prefix}_mtime_unix_s": float(stat.st_mtime)}


def _partial_vmec2000_payload(workdir: Path) -> dict[str, Any]:
    matches = sorted(Path(workdir).glob("threed1*"))
    threed1 = matches[0] if matches else None
    stages = []
    rows = []
    if threed1 is not None and threed1.exists():
        try:
            stages = _parse_vmec2000_threed1(threed1)
            rows = [row for stage in stages for row in stage.rows]
        except Exception:
            stages = []
            rows = []
    totals = [float(row.fsqr) + float(row.fsqz) + float(row.fsql) for row in rows]
    last = rows[-1] if rows else None
    final_ftol = float(stages[-1].ftolv) if stages else None
    tail_plateau = _vmec2000_tail_plateau_payload(rows, stage_ftol=final_ftol)
    progress_phase = (
        "waiting_for_threed1"
        if threed1 is None
        else ("force_iterations" if rows else "startup_or_pre_iteration_output")
    )
    return {
        "workdir": workdir,
        "updated_unix_s": float(time.time()),
        "threed1": threed1,
        **_file_status_payload(threed1, prefix="threed1"),
        "progress_phase": progress_phase,
        "force_rows_started": bool(rows),
        "threed1_tail": _tail_lines(threed1, lines=80),
        "iteration_row_count": len(rows),
        "stage_summaries": [_vmec2000_stage_payload(stage) for stage in stages],
        "tail_rows": [_vmec2000_row_payload(row) for row in rows[-12:]],
        "last_row": None if last is None else _vmec2000_row_payload(last),
        "min_total": None if not totals else float(np.nanmin(np.asarray(totals, dtype=float))),
        "final_max_component": None if last is None else _vmec2000_max_component(last),
        "strict_components_met": None if last is None else _vmec2000_strict_components_met(last, final_ftol),
        "strict_convergence": _strict_convergence_verdict(
            components=(
                None if last is None else float(last.fsqr),
                None if last is None else float(last.fsqz),
                None if last is None else float(last.fsql),
            ),
            requested_ftol=final_ftol,
            free_boundary_active=True if rows else None,
            final_residual_recomputed=True if rows else None,
            tail_plateau_status=tail_plateau.get("status"),
            fresh_residual_required=False,
        ),
        "tail_plateau": tail_plateau,
        "vacuum_grid_exceeded_count": _vacuum_grid_exceeded_count(threed1),
    }


def _write_partial_vmec2000_payload(*, outdir: Path, workdir: Path) -> Path:
    """Write a live VMEC2000 progress report using an atomic file replace."""

    path = Path(outdir) / "_partial_vmec2000_payload.json"
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = _partial_vmec2000_payload(Path(workdir))
    tmp.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)
    return path


def _start_vmec2000_progress_monitor(
    *,
    outdir: Path,
    workdir: Path,
    interval_s: float = 30.0,
) -> tuple[threading.Event, threading.Thread]:
    """Refresh partial VMEC2000 progress while the external executable runs."""

    stop = threading.Event()
    outdir = Path(outdir)
    workdir = Path(workdir)

    def _monitor() -> None:
        while not stop.wait(max(0.1, float(interval_s))):
            try:
                _write_partial_vmec2000_payload(outdir=outdir, workdir=workdir)
            except Exception:
                # Progress monitoring must never affect the VMEC2000 solve.
                continue

    thread = threading.Thread(target=_monitor, name="vmec2000-progress-monitor", daemon=True)
    thread.start()
    return stop, thread


def _last_finite(values: Any) -> float | None:
    try:
        arr = np.asarray(values, dtype=float).reshape(-1)
    except Exception:
        return None
    finite = arr[np.isfinite(arr)]
    return None if finite.size == 0 else float(finite[-1])


def _history_tail(values: Any, *, length: int = 12, dtype: type = float) -> list[Any]:
    try:
        arr = np.asarray(values, dtype=dtype).reshape(-1)
    except Exception:
        return []
    if arr.size == 0:
        return []
    out: list[Any] = []
    for value in arr[-int(length) :]:
        if dtype is int:
            out.append(int(value))
            continue
        value_f = float(value)
        out.append(value_f if np.isfinite(value_f) else None)
    return out


def _history_stats(values: Any, *, dtype: type = float) -> dict[str, Any]:
    """Compact numeric summary for long solver histories."""

    try:
        arr = np.asarray(values, dtype=dtype).reshape(-1)
    except Exception:
        return {"count": 0, "finite_count": 0}
    if arr.size == 0:
        return {"count": 0, "finite_count": 0}
    arr_f = np.asarray(arr, dtype=float)
    finite = arr_f[np.isfinite(arr_f)]
    out: dict[str, Any] = {
        "count": int(arr_f.size),
        "finite_count": int(finite.size),
        "nonzero_count": int(np.count_nonzero(arr_f[np.isfinite(arr_f)])),
    }
    if finite.size == 0:
        return out
    first = arr_f[0]
    last = arr_f[-1]
    out.update(
        {
            "first": float(first) if np.isfinite(first) else None,
            "last": float(last) if np.isfinite(last) else None,
            "min": float(np.nanmin(finite)),
            "max": float(np.nanmax(finite)),
            "mean": float(np.nanmean(finite)),
            "sum": float(np.nansum(finite)),
        }
    )
    return out


def _finite_float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if np.isfinite(out) else None


def _strict_convergence_verdict(
    *,
    components: tuple[Any, Any, Any],
    requested_ftol: Any,
    free_boundary_active: bool | None,
    final_residual_recomputed: Any = None,
    bad_resets: Any = None,
    n_iter: Any = None,
    tail_plateau_status: Any = None,
    fresh_residual_required: bool = True,
) -> dict[str, Any]:
    """Classify a backend row against the strict component-wise force target."""

    values = [_finite_float_or_none(value) for value in components]
    finite_values = [value for value in values if value is not None]
    requested = _finite_float_or_none(requested_ftol)
    if not finite_values:
        return {
            "status": "no_force_rows",
            "strict_component_target": STRICT_COMPONENT_FTOL,
            "loose_component_target": LOOSE_COMPONENT_FTOL,
            "requested_ftol": requested,
            "component_max": None,
            "component_sum": None,
            "component_max_over_strict_target": None,
            "component_max_over_requested_ftol": None,
            "strict_components_met": None,
            "requested_components_met": None,
            "loose_components_met": None,
            "blockers": ["missing_force_components"],
        }

    component_max = float(max(finite_values))
    component_sum = float(sum(finite_values))
    requested_den = STRICT_COMPONENT_FTOL if requested is None else max(float(requested), TINY)
    strict_met = bool(component_max <= STRICT_COMPONENT_FTOL)
    requested_met = bool(component_max <= requested_den)
    loose_met = bool(component_max <= LOOSE_COMPONENT_FTOL)
    blockers: list[str] = []
    if requested is None:
        blockers.append("requested_ftol_unknown")
    elif requested > STRICT_COMPONENT_FTOL * (1.0 + 1.0e-10):
        blockers.append("requested_ftol_above_1e-12")
    if free_boundary_active is False:
        blockers.append("free_boundary_not_active")
    if fresh_residual_required and final_residual_recomputed is False:
        blockers.append("final_residual_not_recomputed_on_accepted_state")
    if not strict_met:
        blockers.append("component_max_above_1e-12")
    if requested is not None and not requested_met:
        blockers.append("component_max_above_requested_ftol")
    bad_reset_count = _finite_float_or_none(bad_resets)
    iter_count = _finite_float_or_none(n_iter)
    if (
        bad_reset_count is not None
        and iter_count is not None
        and iter_count > 0.0
        and bad_reset_count >= max(5.0, 0.5 * iter_count)
    ):
        blockers.append("bad_jacobian_or_restart_limited")
    tail_status = "" if tail_plateau_status is None else str(tail_plateau_status)
    if tail_status == "flat_above_stage_ftol":
        blockers.append("flat_tail_above_stage_ftol")

    strict_gap = float(component_max / STRICT_COMPONENT_FTOL)
    if requested is not None and requested > STRICT_COMPONENT_FTOL * (1.0 + 1.0e-10):
        status = "non_strict_ftol"
    elif free_boundary_active is False:
        status = "free_boundary_not_active"
    elif strict_met and not blockers:
        status = "strict_components_met"
    elif tail_status == "flat_above_stage_ftol":
        status = "stalled_above_strict_ftol"
    elif "bad_jacobian_or_restart_limited" in blockers:
        status = "bad_jacobian_or_restart_limited"
    elif strict_gap <= 100.0:
        status = "near_strict_not_met"
    elif loose_met:
        status = "loose_only_above_strict"
    else:
        status = "underconverged"

    return {
        "status": status,
        "strict_component_target": STRICT_COMPONENT_FTOL,
        "loose_component_target": LOOSE_COMPONENT_FTOL,
        "requested_ftol": requested,
        "component_max": component_max,
        "component_sum": component_sum,
        "component_max_over_strict_target": strict_gap,
        "component_max_over_requested_ftol": float(component_max / requested_den),
        "strict_components_met": strict_met,
        "requested_components_met": requested_met if requested is not None else None,
        "loose_components_met": loose_met,
        "free_boundary_active": free_boundary_active,
        "fresh_residual_required": bool(fresh_residual_required),
        "final_residual_recomputed_on_accepted_state": final_residual_recomputed,
        "tail_plateau_status": tail_status or None,
        "blockers": list(dict.fromkeys(blockers)),
    }


def _tail_decay_projection(
    values: Any,
    *,
    length: int = 128,
    targets: tuple[float, ...] = (1.0e-8, 1.0e-10, 1.0e-12),
) -> dict[str, Any]:
    """Estimate residual-tail decay without storing a long solver history."""

    try:
        arr = np.asarray(values, dtype=float).reshape(-1)
    except Exception:
        return {"count": 0, "finite_positive_count": 0}
    finite_positive = arr[np.isfinite(arr) & (arr > 0.0)]
    out: dict[str, Any] = {
        "count": int(arr.size),
        "finite_positive_count": int(finite_positive.size),
        "window": int(min(max(0, int(length)), finite_positive.size)),
    }
    if finite_positive.size == 0:
        return out
    tail = finite_positive[-out["window"] :] if out["window"] else finite_positive
    last = float(tail[-1])
    out.update(
        {
            "first": float(tail[0]),
            "last": last,
            "min": float(np.nanmin(tail)),
            "max": float(np.nanmax(tail)),
        }
    )
    if tail.size < 2:
        out["estimated_additional_iterations_to_target"] = {
            f"{target:.0e}": 0 if last <= float(target) else None for target in targets
        }
        return out

    previous = tail[:-1]
    current = tail[1:]
    valid = np.isfinite(previous) & np.isfinite(current) & (previous > 0.0) & (current > 0.0)
    ratios = current[valid] / previous[valid]
    ratios = ratios[np.isfinite(ratios) & (ratios > 0.0)]
    out["ratio_count"] = int(ratios.size)
    out["monotone_decrease_fraction"] = float(np.mean(current[valid] < previous[valid])) if np.any(valid) else None
    if ratios.size == 0:
        out["estimated_additional_iterations_to_target"] = {
            f"{target:.0e}": 0 if last <= float(target) else None for target in targets
        }
        return out

    log_slope = float(np.nanmean(np.log(ratios)))
    factor = float(np.exp(log_slope))
    out.update(
        {
            "per_iter_factor": factor,
            "log_slope_per_iter": log_slope,
            "ratio_min": float(np.nanmin(ratios)),
            "ratio_max": float(np.nanmax(ratios)),
        }
    )
    estimates: dict[str, int | None] = {}
    for target in targets:
        target_f = float(target)
        if last <= target_f:
            estimates[f"{target_f:.0e}"] = 0
        elif log_slope < 0.0:
            estimates[f"{target_f:.0e}"] = int(np.ceil(np.log(target_f / last) / log_slope))
        else:
            estimates[f"{target_f:.0e}"] = None
    out["estimated_additional_iterations_to_target"] = estimates
    return out


def _component_sum_history(run: Any) -> np.ndarray:
    result = None if run is None else getattr(run, "result", None)
    if result is None:
        return np.asarray([], dtype=float)
    histories = []
    for attr in ("fsqr2_history", "fsqz2_history", "fsql2_history"):
        try:
            arr = np.asarray(getattr(result, attr), dtype=float).reshape(-1)
        except Exception:
            return np.asarray([], dtype=float)
        if arr.size == 0:
            return np.asarray([], dtype=float)
        histories.append(arr)
    n = min(arr.size for arr in histories)
    if n == 0:
        return np.asarray([], dtype=float)
    return histories[0][-n:] + histories[1][-n:] + histories[2][-n:]


def _component_tail_projection_payload(run: Any) -> dict[str, Any]:
    """Return component-wise residual tail projections for strict FTOL checks."""

    result = None if run is None else getattr(run, "result", None)
    components: dict[str, dict[str, Any]] = {}
    limiting_name: str | None = None
    limiting_value: float | None = None
    for name, attr in (
        ("fsqr", "fsqr2_history"),
        ("fsqz", "fsqz2_history"),
        ("fsql", "fsql2_history"),
    ):
        values = [] if result is None else getattr(result, attr, [])
        projection = _tail_decay_projection(values)
        components[name] = projection
        try:
            last = float(projection.get("last"))
        except Exception:
            last = float("nan")
        if np.isfinite(last) and (limiting_value is None or last > limiting_value):
            limiting_name = name
            limiting_value = last
    return {
        "components": components,
        "limiting_component": limiting_name,
        "limiting_component_value": limiting_value,
    }


def _internal_to_physical_mode_scale(static: Any) -> np.ndarray:
    """Return the VMEC-internal to physical Fourier coefficient scale."""

    modes = getattr(static, "modes", None)
    m = np.asarray(getattr(modes, "m"), dtype=float)
    n = np.asarray(getattr(modes, "n"), dtype=float)
    sqrt2 = np.sqrt(2.0)
    return np.where(m == 0.0, 1.0, sqrt2) * np.where(np.abs(n) == 0.0, 1.0, sqrt2)


def _boundary_control_projection_for_symmetry(
    *,
    config: ExampleConfig,
    deltas: dict[str, np.ndarray],
    symmetry: str,
) -> dict[str, Any] | None:
    """Project a solved boundary move onto one reduced-control symmetry basis."""

    axis_kind = str(config.plasma_axis_kind).strip().lower()
    try:
        basis, matrix = _square_control_fourier_matrix(config, symmetry=symmetry)
        jacobian = matrix.stacked_jacobian()
        target_size = int(sum(np.asarray(value, dtype=float).size for value in deltas.values()))
        if jacobian.shape[0] != target_size:
            return {
                "status": "shape_mismatch",
                "axis_kind": axis_kind,
                "basis_symmetry": basis.symmetry,
                "labels": list(basis.labels),
                "control_count": int(matrix.control_count),
                "jacobian_shape": [int(value) for value in jacobian.shape],
                "target_size": target_size,
            }
        if jacobian.shape[1] == 0:
            return {
                "status": "empty_control_basis",
                "axis_kind": axis_kind,
                "basis_symmetry": basis.symmetry,
                "control_count": int(matrix.control_count),
                "jacobian_shape": [int(value) for value in jacobian.shape],
                "target_size": target_size,
            }
        projection = matrix.project_boundary_delta(
            BoundaryCoeffs(
                R_cos=np.asarray(deltas["R_cos"], dtype=float),
                R_sin=np.asarray(deltas["R_sin"], dtype=float),
                Z_cos=np.asarray(deltas["Z_cos"], dtype=float),
                Z_sin=np.asarray(deltas["Z_sin"], dtype=float),
            )
        )
        return {
            "status": "available" if projection.target_l2 > TINY else "zero_boundary_motion",
            "axis_kind": axis_kind,
            "basis_symmetry": basis.symmetry,
            "labels": list(projection.labels),
            "control_count": int(matrix.control_count),
            "radius_delta": [float(value) for value in projection.radius_delta],
            "radius_delta_by_label": projection.radius_delta_by_label,
            "rank": int(projection.rank),
            "singular_values": [float(value) for value in projection.singular_values],
            "condition_number": projection.condition_number,
            "jacobian_shape": [int(value) for value in jacobian.shape],
            "target_l2": projection.target_l2,
            "predicted_l2": projection.predicted_l2,
            "residual_l2": projection.residual_l2,
            "residual_linf": projection.residual_linf,
            "residual_rms": projection.residual_rms,
            "residual_rel": projection.residual_rel,
            "captured_fraction": projection.captured_fraction,
        }
    except Exception as exc:
        return {
            "status": f"failed:{type(exc).__name__}",
            "axis_kind": axis_kind,
            "basis_symmetry": str(symmetry),
            "error": repr(exc),
        }


def _boundary_reduced_control_projection_payload(
    *,
    config: ExampleConfig | None,
    deltas: dict[str, np.ndarray],
) -> dict[str, Any] | None:
    """Project a solved boundary move onto the square side/corner controls."""

    if config is None:
        return None
    axis_kind = str(config.plasma_axis_kind).strip().lower()
    if not _square_axis_uses_spline_controls(config):
        return {
            "status": "not_applicable_for_axis_kind",
            "axis_kind": axis_kind,
        }
    primary = _boundary_control_projection_for_symmetry(
        config=config,
        deltas=deltas,
        symmetry="square",
    )
    if primary is None:
        return None
    primary["candidate_bases"] = {
        symmetry: _boundary_control_projection_for_symmetry(
            config=config,
            deltas=deltas,
            symmetry=symmetry,
        )
        for symmetry in ("square", "stellarator")
    }
    return primary


def _boundary_motion_payload(run: Any, *, config: ExampleConfig | None = None) -> dict[str, Any] | None:
    """Measure how far the accepted LCFS moved from the input boundary."""

    try:
        state = getattr(run, "state")
        static = getattr(run, "static")
        indata = getattr(run, "indata")
        initial = boundary_from_indata(indata, static.modes)
        scale = _internal_to_physical_mode_scale(static)
        final = {
            "R_cos": np.asarray(state.Rcos, dtype=float)[-1] * scale,
            "R_sin": np.asarray(state.Rsin, dtype=float)[-1] * scale,
            "Z_cos": np.asarray(state.Zcos, dtype=float)[-1] * scale,
            "Z_sin": np.asarray(state.Zsin, dtype=float)[-1] * scale,
        }
        initial_components = {
            "R_cos": np.asarray(initial.R_cos, dtype=float),
            "R_sin": np.asarray(initial.R_sin, dtype=float),
            "Z_cos": np.asarray(initial.Z_cos, dtype=float),
            "Z_sin": np.asarray(initial.Z_sin, dtype=float),
        }
        coeff_norm_sq = 0.0
        coeff_ref_norm_sq = 0.0
        coeff_max = 0.0
        deltas: dict[str, np.ndarray] = {}
        for name, final_values in final.items():
            initial_values = initial_components[name]
            if final_values.shape != initial_values.shape:
                return None
            delta = final_values - initial_values
            deltas[name] = delta
            coeff_norm_sq += float(np.sum(delta * delta))
            coeff_ref_norm_sq += float(np.sum(initial_values * initial_values))
            if delta.size:
                coeff_max = max(coeff_max, float(np.max(np.abs(delta))))

        R0 = np.asarray(eval_fourier(initial.R_cos, initial.R_sin, static.basis), dtype=float)
        Z0 = np.asarray(eval_fourier(initial.Z_cos, initial.Z_sin, static.basis), dtype=float)
        R1 = np.asarray(
            eval_fourier(state.Rcos[-1], state.Rsin[-1], static.basis, coeffs_internal=True),
            dtype=float,
        )
        Z1 = np.asarray(
            eval_fourier(state.Zcos[-1], state.Zsin[-1], static.basis, coeffs_internal=True),
            dtype=float,
        )
        if R0.shape != R1.shape or Z0.shape != Z1.shape:
            return None
        displacement = np.sqrt((R1 - R0) ** 2 + (Z1 - Z0) ** 2)
        ref_radius = np.sqrt(R0 * R0 + Z0 * Z0)
        rms = float(np.sqrt(np.mean(displacement * displacement)))
        max_abs = float(np.max(displacement))
        ref_rms = float(np.sqrt(np.mean(ref_radius * ref_radius)))
        coeff_norm = float(np.sqrt(coeff_norm_sq))
        coeff_ref_norm = float(np.sqrt(coeff_ref_norm_sq))
        payload: dict[str, Any] = {
            "boundary_coeff_delta_l2": coeff_norm,
            "boundary_coeff_delta_linf": coeff_max,
            "boundary_coeff_delta_rel": float(coeff_norm / max(coeff_ref_norm, TINY)),
            "boundary_sample_displacement_rms": rms,
            "boundary_sample_displacement_max": max_abs,
            "boundary_sample_displacement_rel": float(rms / max(ref_rms, TINY)),
        }
        projection = _boundary_reduced_control_projection_payload(config=config, deltas=deltas)
        if projection is not None:
            payload["boundary_reduced_control_projection"] = projection
        return payload
    except Exception:
        return None


def _jax_history_payload(run: Any, diag: dict[str, Any], *, length: int = 12) -> dict[str, Any]:
    result = None if run is None else getattr(run, "result", None)
    component_sum = _component_sum_history(run)
    component_projection = _component_tail_projection_payload(run)
    return {
        "length": None if result is None else int(getattr(result, "n_iter", -1)),
        "w_tail": _history_tail([] if result is None else getattr(result, "w_history", []), length=length),
        "fsqr_tail": _history_tail([] if result is None else getattr(result, "fsqr2_history", []), length=length),
        "fsqz_tail": _history_tail([] if result is None else getattr(result, "fsqz2_history", []), length=length),
        "fsql_tail": _history_tail([] if result is None else getattr(result, "fsql2_history", []), length=length),
        "fsq_component_sum_tail": _history_tail(component_sum, length=length),
        "fsq_component_sum_stats": _history_stats(component_sum),
        "fsq_component_sum_tail_projection": _tail_decay_projection(component_sum),
        "fsq_component_tail_projection_by_component": component_projection["components"],
        "fsq_limiting_component": component_projection["limiting_component"],
        "fsq_limiting_component_value": component_projection["limiting_component_value"],
        "freeb_ivac_tail": _history_tail(diag.get("freeb_ivac_history"), length=length, dtype=int),
        "freeb_ivacskip_tail": _history_tail(diag.get("freeb_ivacskip_history"), length=length, dtype=int),
        "freeb_full_update_tail": _history_tail(diag.get("freeb_full_update_history"), length=length, dtype=int),
        "freeb_full_update_stats": _history_stats(diag.get("freeb_full_update_history"), dtype=int),
        "freeb_nestor_reused_tail": _history_tail(diag.get("freeb_nestor_reused_history"), length=length, dtype=int),
        "freeb_nestor_reused_stats": _history_stats(diag.get("freeb_nestor_reused_history"), dtype=int),
        "freeb_nestor_source_reused_tail": _history_tail(
            diag.get("freeb_nestor_source_reused_history"), length=length, dtype=int
        ),
        "freeb_nestor_source_reused_stats": _history_stats(
            diag.get("freeb_nestor_source_reused_history"), dtype=int
        ),
        "freeb_nestor_provider_allows_source_reuse_tail": _history_tail(
            diag.get("freeb_nestor_provider_allows_source_reuse_history"), length=length, dtype=int
        ),
        "freeb_nestor_provider_allows_source_reuse_stats": _history_stats(
            diag.get("freeb_nestor_provider_allows_source_reuse_history"), dtype=int
        ),
        "freeb_nestor_solve_time_tail": _history_tail(
            diag.get("freeb_nestor_solve_time_history"), length=length
        ),
        "freeb_nestor_solve_time_stats": _history_stats(diag.get("freeb_nestor_solve_time_history")),
        "freeb_nestor_sample_time_tail": _history_tail(
            diag.get("freeb_nestor_sample_time_history"), length=length
        ),
        "freeb_nestor_sample_time_stats": _history_stats(diag.get("freeb_nestor_sample_time_history")),
        "freeb_nestor_trial_reused_tail": _history_tail(
            diag.get("freeb_nestor_trial_reused_history"), length=length, dtype=int
        ),
        "freeb_nestor_trial_reused_stats": _history_stats(
            diag.get("freeb_nestor_trial_reused_history"), dtype=int
        ),
        "freeb_nestor_trial_failed_tail": _history_tail(
            diag.get("freeb_nestor_trial_failed_history"), length=length, dtype=int
        ),
        "freeb_nestor_trial_failed_stats": _history_stats(
            diag.get("freeb_nestor_trial_failed_history"), dtype=int
        ),
        "freeb_nestor_trial_solve_time_tail": _history_tail(
            diag.get("freeb_nestor_trial_solve_time_history"), length=length
        ),
        "freeb_nestor_trial_solve_time_stats": _history_stats(
            diag.get("freeb_nestor_trial_solve_time_history")
        ),
        "freeb_nestor_trial_sample_time_tail": _history_tail(
            diag.get("freeb_nestor_trial_sample_time_history"), length=length
        ),
        "freeb_nestor_trial_sample_time_stats": _history_stats(
            diag.get("freeb_nestor_trial_sample_time_history")
        ),
        "freeb_nestor_bnormal_rms_tail": _history_tail(
            diag.get("freeb_nestor_bnormal_rms_history"), length=length
        ),
        "freeb_nestor_bnormal_rms_stats": _history_stats(diag.get("freeb_nestor_bnormal_rms_history")),
        "freeb_nestor_bsqvac_rms_tail": _history_tail(
            diag.get("freeb_nestor_bsqvac_rms_history"), length=length
        ),
        "freeb_nestor_bsqvac_rms_stats": _history_stats(diag.get("freeb_nestor_bsqvac_rms_history")),
        "freeb_anderson_pressure_applied_tail": _history_tail(
            diag.get("freeb_anderson_pressure_applied_history"), length=length, dtype=int
        ),
        "freeb_anderson_pressure_applied_stats": _history_stats(
            diag.get("freeb_anderson_pressure_applied_history"), dtype=int
        ),
        "freeb_anderson_pressure_theta_tail": _history_tail(
            diag.get("freeb_anderson_pressure_theta_history"), length=length
        ),
        "freeb_anderson_pressure_residual_norm_tail": _history_tail(
            diag.get("freeb_anderson_pressure_residual_norm_history"), length=length
        ),
        "include_edge_tail": _history_tail(diag.get("include_edge_history"), length=length, dtype=int),
        "include_edge_stats": _history_stats(diag.get("include_edge_history"), dtype=int),
        "bcovar_update_tail": _history_tail(diag.get("bcovar_update_history"), length=length, dtype=int),
        "bcovar_update_stats": _history_stats(diag.get("bcovar_update_history"), dtype=int),
        "bad_jacobian_tail": _history_tail(diag.get("bad_jacobian_history"), length=length, dtype=int),
        "bad_jacobian_stats": _history_stats(diag.get("bad_jacobian_history"), dtype=int),
        "time_step_tail": _history_tail(diag.get("time_step_history"), length=length),
        "time_step_stats": _history_stats(diag.get("time_step_history")),
        "dt_eff_tail": _history_tail(diag.get("dt_eff_history"), length=length),
        "dt_eff_stats": _history_stats(diag.get("dt_eff_history")),
        "update_rms_tail": _history_tail(diag.get("update_rms_history"), length=length),
        "update_rms_stats": _history_stats(diag.get("update_rms_history")),
    }


def _classify_run(diag: dict[str, Any], residuals: dict[str, Any]) -> str:
    if bool(residuals.get("converged_strict", False)):
        return "converged_strict"
    if not bool(residuals.get("free_boundary_active", False)):
        return "free_boundary_not_activated"
    bad_resets = int(residuals.get("bad_resets") or 0)
    n_iter = int(residuals.get("n_iter") or 0)
    if n_iter > 0 and bad_resets >= max(5, n_iter // 2):
        return "bad_jacobian_or_restart_limited"
    component_sum = residuals.get("final_fsq_component_sum")
    requested = residuals.get("requested_ftol")
    if component_sum is not None and requested is not None:
        try:
            if float(component_sum) > 100.0 * float(requested):
                return "underconverged"
        except Exception:
            pass
    return "incomplete"


def _final_residuals(run: Any, *, config: ExampleConfig | None = None) -> dict[str, Any]:
    diag = run.result.diagnostics if run.result is not None else {}
    diag = diag if isinstance(diag, dict) else {}
    fsqr = diag.get("final_fsqr")
    fsqz = diag.get("final_fsqz")
    fsql = diag.get("final_fsql")
    values = [v for v in (fsqr, fsqz, fsql) if v is not None and np.isfinite(float(v))]
    freeb = diag.get("free_boundary", {}) if isinstance(diag.get("free_boundary", {}), dict) else {}
    nestor = (
        freeb.get("last_nestor_diagnostics", {})
        if isinstance(freeb.get("last_nestor_diagnostics", {}), dict)
        else {}
    )
    model = str(freeb.get("nestor_model", "none"))
    out = {
        "n_iter": None if run.result is None else int(getattr(run.result, "n_iter", -1)),
        "solver_mode": diag.get("solver_mode"),
        "use_scan": diag.get("use_scan"),
        "performance_mode": diag.get("performance_mode"),
        "converged": bool(diag.get("converged", False)),
        "converged_strict": bool(diag.get("converged_strict", False)),
        "requested_ftol": diag.get("requested_ftol"),
        "final_fsqr": fsqr,
        "final_fsqz": fsqz,
        "final_fsql": fsql,
        "final_fsq_component_sum": float(sum(float(v) for v in values)) if values else None,
        "pre_update_final_fsqr": diag.get("pre_update_final_fsqr"),
        "pre_update_final_fsqz": diag.get("pre_update_final_fsqz"),
        "pre_update_final_fsql": diag.get("pre_update_final_fsql"),
        "return_best_scored_state": diag.get("return_best_scored_state"),
        "returned_best_scored_state": diag.get("returned_best_scored_state"),
        "best_scored_iter": diag.get("best_scored_iter"),
        "best_scored_fsq": diag.get("best_scored_fsq"),
        "best_scored_fsqr": diag.get("best_scored_fsqr"),
        "best_scored_fsqz": diag.get("best_scored_fsqz"),
        "best_scored_fsql": diag.get("best_scored_fsql"),
        "best_scored_full_boundary_count": diag.get("best_scored_full_boundary_count"),
        "best_scored_fresh_boundary_count": diag.get("best_scored_fresh_boundary_count"),
        "free_boundary_convergence_blocked_count": diag.get("free_boundary_convergence_blocked_count"),
        "free_boundary_fresh_convergence_gate": diag.get("free_boundary_fresh_convergence_gate"),
        "free_boundary_fresh_convergence_recheck_count": diag.get("free_boundary_fresh_convergence_recheck_count"),
        "free_boundary_fresh_convergence_reject_count": diag.get("free_boundary_fresh_convergence_reject_count"),
        "free_boundary_fresh_convergence_failed_count": diag.get("free_boundary_fresh_convergence_failed_count"),
        "final_iter2_for_recompute": diag.get("final_iter2_for_recompute"),
        "bad_resets": diag.get("bad_resets"),
        "ijacob": diag.get("ijacob"),
        "final_residual_recomputed_on_accepted_state": diag.get("final_residual_recomputed_on_accepted_state"),
        "free_boundary_nestor_model": model,
        "free_boundary_active": bool(model.strip() and model != "none"),
        "free_boundary_jax_nestor_operator_applied": nestor.get("jax_nestor_operator_applied"),
        "free_boundary_jax_nestor_operator_reason": nestor.get("jax_nestor_operator_reason"),
        "free_boundary_jax_nestor_operator_time_s": nestor.get("jax_nestor_operator_time_s"),
        "free_boundary_jax_nestor_operator_jitted": nestor.get("jax_nestor_operator_jitted"),
        "free_boundary_jax_nestor_operator_cache_hit": nestor.get("jax_nestor_operator_cache_hit"),
        "free_boundary_nestor_matrix_override_applied": nestor.get("matrix_override_applied"),
        "free_boundary_nestor_rhs_mode": nestor.get("rhs_mode"),
        "free_boundary_bnormal_rms": nestor.get("bnormal_rms"),
        "free_boundary_bsqvac_rms": nestor.get("bsqvac_rms"),
        "free_boundary_couple_edge": freeb.get("couple_edge"),
        "free_boundary_edge_control_projection": freeb.get("edge_control_projection"),
        "free_boundary_anderson_pressure_enabled": freeb.get("anderson_pressure_enabled"),
        "free_boundary_anderson_pressure_last_applied": _last_finite(
            diag.get("freeb_anderson_pressure_applied_history")
        ),
        "free_boundary_anderson_pressure_last_theta": _last_finite(
            diag.get("freeb_anderson_pressure_theta_history")
        ),
        "free_boundary_activate_fsq": freeb.get("activate_fsq"),
        "free_boundary_last_ivac": freeb.get("ivac"),
        "free_boundary_last_ivacskip": freeb.get("ivacskip"),
        "free_boundary_last_nvacskip": freeb.get("nvacskip"),
        "free_boundary_last_nestor_solve_time_s": _last_finite(diag.get("freeb_nestor_solve_time_history")),
        "free_boundary_last_nestor_sample_time_s": _last_finite(diag.get("freeb_nestor_sample_time_history")),
        "history": _jax_history_payload(run, diag),
    }
    out["strict_convergence"] = _strict_convergence_verdict(
        components=(fsqr, fsqz, fsql),
        requested_ftol=diag.get("requested_ftol"),
        free_boundary_active=bool(model.strip() and model != "none"),
        final_residual_recomputed=diag.get("final_residual_recomputed_on_accepted_state"),
        bad_resets=diag.get("bad_resets"),
        n_iter=None if run.result is None else int(getattr(run.result, "n_iter", -1)),
        fresh_residual_required=True,
    )
    boundary_motion = _boundary_motion_payload(run, config=config)
    if boundary_motion is not None:
        out.update(boundary_motion)
    out["stall_classification"] = _classify_run(diag, out)
    return out


def _vector_surface_rms(values: Any) -> float | None:
    try:
        arr = np.asarray(values, dtype=float)
    except Exception:
        return None
    if arr.ndim != 3 or arr.shape[0] != 3:
        return None
    mag2 = np.sum(arr * arr, axis=0)
    finite = mag2[np.isfinite(mag2)]
    if finite.size == 0:
        return None
    return float(np.sqrt(np.mean(finite)))


def _virtual_casing_profile_payload(
    *,
    run: Any,
    direct_params: Any | None,
    coil_geometry: Any | None = None,
    quad_factor: int = 2,
    chunk_size: int | str | None = "auto",
    target_chunk_size: int | str | None = "auto",
) -> dict[str, Any]:
    """Return optional DESC-style finite-beta postsolve boundary diagnostics."""

    quad = int(quad_factor)
    if quad < 1:
        raise ValueError("virtual-casing quad_factor must be at least one")
    if direct_params is None:
        return {"status": "skipped_requires_direct_coils"}
    try:
        __import__("virtual_casing_jax.functional")
    except ImportError:
        return {
            "status": "skipped_missing_virtual_casing_jax",
            "quad_factor": quad,
            "chunk_size": chunk_size,
            "target_chunk_size": target_chunk_size,
        }

    t0 = time.perf_counter()
    try:
        diagnostics = virtual_casing_diagnostics_from_run(
            run,
            coil_params=direct_params,
            coil_geometry=coil_geometry,
            quad_factor=quad,
            chunk_size=chunk_size,
            target_chunk_size=target_chunk_size,
        )
    except Exception as exc:
        return {
            "status": f"failed:{type(exc).__name__}",
            "error": repr(exc),
            "wall_s": float(time.perf_counter() - t0),
            "quad_factor": quad,
            "chunk_size": chunk_size,
            "target_chunk_size": target_chunk_size,
        }
    ntheta = int(np.asarray(diagnostics.external_bnormal_residual).shape[0])
    nzeta = int(np.asarray(diagnostics.external_bnormal_residual).shape[1])
    return {
        "status": "computed",
        "wall_s": float(time.perf_counter() - t0),
        "quad_factor": quad,
        "chunk_size": chunk_size,
        "target_chunk_size": target_chunk_size,
        "quad_ntheta": max(quad * ntheta, ntheta),
        "quad_nzeta": max(quad * nzeta, nzeta),
        "external_bnormal_residual_rms": diagnostics.external_bnormal_residual_rms,
        "external_bnormal_residual_max": diagnostics.external_bnormal_residual_max,
        "pressure_balance_rms": diagnostics.pressure_balance_rms,
        "pressure_balance_max": diagnostics.pressure_balance_max,
        "required_external_b_rms": _vector_surface_rms(diagnostics.required_external_b),
        "target_external_b_rms": _vector_surface_rms(diagnostics.target_external_b),
        "ntheta": ntheta,
        "nzeta": nzeta,
    }


def _mgrid_bounds(indata: Any, *, padding_fraction: float, min_padding: float) -> dict[str, float]:
    samples = evaluate_toroidal_hybrid_indata_boundary(indata, ntheta=96, nzeta=128)
    rmin = float(np.min(samples.R))
    rmax = float(np.max(samples.R))
    zmin = float(np.min(samples.Z))
    zmax = float(np.max(samples.Z))
    rpad = max(float(min_padding), float(padding_fraction) * max(rmax - rmin, 1.0e-6))
    zpad = max(float(min_padding), float(padding_fraction) * max(zmax - zmin, 1.0e-6))
    return {
        "rmin": max(1.0e-3, rmin - rpad),
        "rmax": rmax + rpad,
        "zmin": zmin - zpad,
        "zmax": zmax + zpad,
        "boundary_rmin": rmin,
        "boundary_rmax": rmax,
        "boundary_zmin": zmin,
        "boundary_zmax": zmax,
    }


def _bool_env(value: bool) -> str:
    return "1" if bool(value) else "0"


def _set_backend_env(overrides: dict[str, str | None]) -> dict[str, str | None]:
    previous = {name: os.environ.get(name) for name in overrides}
    for name, value in overrides.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = str(value)
    return previous


def _restore_backend_env(previous: dict[str, str | None]) -> None:
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


_FREEB_HOT_RESTART_RESUME_KEYS = (
    "freeb_ivac",
    "freeb_ivacskip",
    "freeb_nvacskip",
    "freeb_nvskip0",
    "freeb_last_model",
    "freeb_nestor_runtime",
    "prev_rz_fsq",
)


def _run_diagnostics(run: Any) -> dict[str, Any]:
    result = None if run is None else getattr(run, "result", None)
    diagnostics = None if result is None else getattr(result, "diagnostics", None)
    return diagnostics if isinstance(diagnostics, dict) else {}


def _jax_hot_restart_solver_state(run: Any, *, policy: str) -> dict[str, Any] | None:
    """Return the optional resume-state payload for a final-grid hot restart."""

    policy_key = str(policy).strip().lower()
    if policy_key == "state":
        return None
    resume = _run_diagnostics(run).get("resume_state")
    if not isinstance(resume, dict):
        return None
    if policy_key == "full":
        return dict(resume)
    if policy_key != "freeb":
        raise ValueError(f"unknown hot-restart policy {policy!r}")
    out = {key: resume[key] for key in _FREEB_HOT_RESTART_RESUME_KEYS if key in resume}
    return out or None


def _strict_residual_met(residuals: dict[str, Any]) -> bool:
    strict = residuals.get("strict_convergence")
    if isinstance(strict, dict) and strict.get("strict_components_met") is not None:
        return bool(strict.get("strict_components_met"))
    return bool(residuals.get("converged_strict", False))


def _jax_hot_restart_stage_payload(
    *,
    stage_index: int,
    kind: str,
    budget: int,
    restart_policy: str | None,
    run: Any,
    residuals: dict[str, Any],
) -> dict[str, Any]:
    """Return compact per-stage convergence data for hot-restart profiling."""

    strict = residuals.get("strict_convergence")
    strict = strict if isinstance(strict, dict) else {}
    return {
        "stage_index": int(stage_index),
        "kind": str(kind),
        "budget": int(budget),
        "restart_policy": None if restart_policy is None else str(restart_policy),
        "n_iter": residuals.get("n_iter"),
        "converged": bool(residuals.get("converged", False)),
        "converged_strict": bool(residuals.get("converged_strict", False)),
        "strict_status": strict.get("status"),
        "strict_components_met": strict.get("strict_components_met"),
        "component_max": strict.get("component_max"),
        "component_sum": strict.get("component_sum"),
        "component_max_over_strict_target": strict.get("component_max_over_strict_target"),
        "final_fsq_component_sum": residuals.get("final_fsq_component_sum"),
        "final_residual_recomputed_on_accepted_state": residuals.get(
            "final_residual_recomputed_on_accepted_state"
        ),
        "free_boundary_last_ivac": residuals.get("free_boundary_last_ivac"),
        "free_boundary_last_ivacskip": residuals.get("free_boundary_last_ivacskip"),
        "free_boundary_last_nvacskip": residuals.get("free_boundary_last_nvacskip"),
        "returned_best_scored_state": residuals.get("returned_best_scored_state"),
        "best_scored_iter": residuals.get("best_scored_iter"),
        "history_length": None if getattr(run, "result", None) is None else int(getattr(run.result, "n_iter", -1)),
    }


def _run_jax_backend(
    *,
    input_path: Path,
    wout_path: Path,
    config: ExampleConfig,
    beta_percent: float = 0.0,
    direct_params: Any | None,
    solver_mode: str | None,
    return_best_scored_state: bool,
    freeb_anderson_pressure: bool = False,
    freeb_jax_nestor_operator: bool = False,
    freeb_jax_nestor_jit_operator: bool = True,
    freeb_include_edge: bool | None = None,
    freeb_dense_solve_mode: str | None = None,
    freeb_experimental_fouri_matrix: bool | None = None,
    freeb_add_analytic_bvec: bool | None = None,
    freeb_edge_control_projection: str = "none",
    freeb_edge_control_rcond: float = 1.0e-12,
    direct_static_cache: bool = True,
    jit_direct_sampler: bool = False,
    direct_trial_bsqvac_resample: bool = True,
    jax_hot_restart_count: int = 0,
    jax_hot_restart_iters: int | None = None,
    jax_hot_restart_policy: str = "freeb",
    jax_hot_restart_always: bool = False,
    jax_initial_restart_wout: Path | None = None,
    verbose_solver: bool = False,
    virtual_casing_diagnostics: bool = False,
    virtual_casing_quad_factor: int = 2,
    virtual_casing_chunk_size: int | str | None = "auto",
    virtual_casing_target_chunk_size: int | str | None = "auto",
    accepted_provider_parity: bool = False,
    accepted_provider_parity_mgrid_input: Path | None = None,
    accepted_provider_parity_coil_params: Any | None = None,
    accepted_provider_parity_bounds: dict[str, float] | None = None,
    accepted_provider_parity_mgrid_nphi: int | None = None,
    accepted_provider_parity_sample: str = "accepted_boundary_coil_field_only",
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    coil_geometry = None
    if direct_params is not None:
        kwargs = {
            "external_field_provider_kind": "direct_coils",
            "external_field_provider_params": direct_params,
        }
        if bool(direct_static_cache):
            coil_geometry = build_coil_field_geometry(direct_params)
            kwargs["external_field_provider_static"] = {
                "coil_geometry": coil_geometry,
                "regularization_epsilon": getattr(direct_params, "regularization_epsilon", 0.0),
                "chunk_size": getattr(direct_params, "chunk_size", None),
                "cache_scope": "square_coil_profile_direct_solve",
                "jit_sampler": bool(jit_direct_sampler),
                "resample_trial_bsqvac": bool(direct_trial_bsqvac_resample),
            }
    edge_control_projection_payload = _freeb_edge_control_projection_solver_payload(
        config,
        symmetry=str(freeb_edge_control_projection),
        rcond=float(freeb_edge_control_rcond),
    )
    edge_control_projection_summary = _freeb_edge_control_projection_summary(
        edge_control_projection_payload,
        requested=str(freeb_edge_control_projection),
    )
    hot_restart_count = max(0, int(jax_hot_restart_count))
    hot_restart_iters_eff = (
        int(jax_hot_restart_iters) if jax_hot_restart_iters is not None else int(config.max_iter)
    )
    hot_restart_policy = str(jax_hot_restart_policy).strip().lower()
    if hot_restart_policy not in {"state", "freeb", "full"}:
        raise ValueError(f"unknown --jax-hot-restart-policy value: {jax_hot_restart_policy!r}")

    t0 = time.perf_counter()
    env_overrides: dict[str, str | None] = {
        "VMEC_JAX_RETURN_BEST_SCORED_STATE": _bool_env(return_best_scored_state),
        "VMEC_JAX_FREEB_ANDERSON_PRESSURE": _bool_env(freeb_anderson_pressure),
        "VMEC_JAX_FREEB_JAX_NESTOR_OPERATOR": _bool_env(freeb_jax_nestor_operator),
        "VMEC_JAX_FREEB_JAX_NESTOR_JIT_OPERATOR": _bool_env(freeb_jax_nestor_jit_operator),
    }
    if freeb_include_edge is not None:
        env_overrides["VMEC_JAX_FREEB_INCLUDE_EDGE"] = _bool_env(freeb_include_edge)
    if freeb_dense_solve_mode is not None:
        env_overrides["VMEC_JAX_FREEB_DENSE_SOLVE_MODE"] = str(freeb_dense_solve_mode)
    if freeb_experimental_fouri_matrix is not None:
        env_overrides["VMEC_JAX_FREEB_EXPERIMENTAL_FOURI_MATRIX"] = _bool_env(freeb_experimental_fouri_matrix)
    if freeb_add_analytic_bvec is not None:
        env_overrides["VMEC_JAX_FREEB_ADD_ANALYTIC_BVEC"] = _bool_env(freeb_add_analytic_bvec)
    previous_env = _set_backend_env(env_overrides)
    try:
        def _run_solver_pass(
            *,
            max_iter: int,
            restart_state: Any | None,
            restart_wout_path: Path | None,
            restart_solver_state: dict[str, Any] | None,
            multigrid: bool,
        ) -> Any:
            pass_kwargs = dict(kwargs)
            if restart_state is not None:
                pass_kwargs["restart_state"] = restart_state
            if restart_wout_path is not None:
                pass_kwargs["restart_wout_path"] = restart_wout_path
            if restart_solver_state is not None:
                pass_kwargs["restart_solver_state"] = restart_solver_state
            if edge_control_projection_payload is not None:
                pass_kwargs["free_boundary_edge_control_projection"] = edge_control_projection_payload
            return run_free_boundary(
                input_path,
                max_iter=int(max_iter),
                multigrid=bool(multigrid),
                multigrid_use_input_niter=True,
                verbose=bool(verbose_solver),
                jit_forces=config.jit_forces,
                solver_mode=solver_mode,
                free_boundary_activate_fsq=None
                if config.free_boundary_activate_fsq is None
                else float(config.free_boundary_activate_fsq),
                **pass_kwargs,
            )

        initial_budget = (
            int(hot_restart_iters_eff)
            if jax_initial_restart_wout is not None
            else _run_budget(config, restart_state=None)
        )
        run = _run_solver_pass(
            max_iter=initial_budget,
            restart_state=None,
            restart_wout_path=jax_initial_restart_wout,
            restart_solver_state=None,
            multigrid=bool(config.use_multigrid_schedule) and jax_initial_restart_wout is None,
        )
        residuals = _final_residuals(run, config=config)
        hot_restart_stages = [
            _jax_hot_restart_stage_payload(
                stage_index=0,
                kind="initial_restart_wout" if jax_initial_restart_wout is not None else "initial",
                budget=int(initial_budget),
                restart_policy="wout" if jax_initial_restart_wout is not None else None,
                run=run,
                residuals=residuals,
            )
        ]
        for restart_index in range(1, hot_restart_count + 1):
            if _strict_residual_met(residuals) and not bool(jax_hot_restart_always):
                break
            _log_step(
                "running vmec_jax hot restart "
                f"{restart_index}/{hot_restart_count} "
                f"(policy={hot_restart_policy}, niter={hot_restart_iters_eff})"
            )
            restart_solver_state = _jax_hot_restart_solver_state(run, policy=hot_restart_policy)
            run = _run_solver_pass(
                max_iter=int(hot_restart_iters_eff),
                restart_state=run.state,
                restart_wout_path=None,
                restart_solver_state=restart_solver_state,
                multigrid=False,
            )
            residuals = _final_residuals(run, config=config)
            hot_restart_stages.append(
                _jax_hot_restart_stage_payload(
                    stage_index=int(restart_index),
                    kind="hot_restart",
                    budget=int(hot_restart_iters_eff),
                    restart_policy=hot_restart_policy,
                    run=run,
                    residuals=residuals,
                )
            )
    finally:
        _restore_backend_env(previous_env)
    wall_s = time.perf_counter() - t0
    write_wout_from_fixed_boundary_run(wout_path, run, include_fsq=True)
    accepted_parity = (
        _accepted_provider_parity_payload(
            run=run,
            mgrid_input=accepted_provider_parity_mgrid_input,
            coil_params=accepted_provider_parity_coil_params,
            config=config,
            bounds={} if accepted_provider_parity_bounds is None else accepted_provider_parity_bounds,
            mgrid_nphi=int(accepted_provider_parity_mgrid_nphi or config.nzeta),
            sample=str(accepted_provider_parity_sample),
        )
        if bool(accepted_provider_parity)
        else {"status": "disabled"}
    )
    vc_payload = (
        _virtual_casing_profile_payload(
            run=run,
            direct_params=direct_params,
            coil_geometry=coil_geometry,
            quad_factor=int(virtual_casing_quad_factor),
            chunk_size=virtual_casing_chunk_size,
            target_chunk_size=virtual_casing_target_chunk_size,
        )
        if bool(virtual_casing_diagnostics)
        else {"status": "disabled"}
    )
    return {
        "status": "completed",
        "wall_s": float(wall_s),
        "input": input_path,
        "wout": wout_path,
        **residuals,
        "free_boundary_solver_overrides": {
            "return_best_scored_state": bool(return_best_scored_state),
            "freeb_anderson_pressure": bool(freeb_anderson_pressure),
            "freeb_jax_nestor_operator": bool(freeb_jax_nestor_operator),
            "freeb_jax_nestor_jit_operator": bool(freeb_jax_nestor_jit_operator),
            "freeb_include_edge": None if freeb_include_edge is None else bool(freeb_include_edge),
            "freeb_dense_solve_mode": None if freeb_dense_solve_mode is None else str(freeb_dense_solve_mode),
            "freeb_experimental_fouri_matrix": None
            if freeb_experimental_fouri_matrix is None
            else bool(freeb_experimental_fouri_matrix),
            "freeb_add_analytic_bvec": None
            if freeb_add_analytic_bvec is None
            else bool(freeb_add_analytic_bvec),
            "freeb_edge_control_projection": edge_control_projection_summary,
            "jax_hot_restart_count": int(hot_restart_count),
            "jax_hot_restart_iters": int(hot_restart_iters_eff),
            "jax_hot_restart_policy": str(hot_restart_policy),
            "jax_hot_restart_always": bool(jax_hot_restart_always),
            "jax_initial_restart_wout": None
            if jax_initial_restart_wout is None
            else str(jax_initial_restart_wout),
        },
        "hot_restart": {
            "enabled": bool(hot_restart_count > 0),
            "requested_count": int(hot_restart_count),
            "executed_count": max(0, len(hot_restart_stages) - 1),
            "iters_per_restart": int(hot_restart_iters_eff),
            "resume_policy": str(hot_restart_policy),
            "initial_restart_wout": None
            if jax_initial_restart_wout is None
            else str(jax_initial_restart_wout),
            "always": bool(jax_hot_restart_always),
            "stopped_after_strict_convergence": bool(
                hot_restart_count > 0
                and len(hot_restart_stages) <= hot_restart_count
                and _strict_residual_met(residuals)
                and not bool(jax_hot_restart_always)
            ),
            "stages": hot_restart_stages,
        },
        "accepted_provider_parity": accepted_parity,
        "virtual_casing": vc_payload,
        "free_boundary_promotion": free_boundary_promotion_status(
            beta_percent=float(beta_percent),
            strict_components_met=residuals.get("converged_strict"),
            final_residual_recomputed=residuals.get("final_residual_recomputed_on_accepted_state"),
            virtual_casing_status=vc_payload.get("status"),
            direct_coil_backend=direct_params is not None,
        ),
    }


def _rms(value: np.ndarray) -> float | None:
    arr = np.asarray(value, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(np.sqrt(np.mean(arr * arr)))


def _max_abs(value: np.ndarray) -> float | None:
    arr = np.asarray(value, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(np.max(np.abs(arr)))


def _difference_stats(candidate: Any, reference: Any) -> dict[str, float | None]:
    cand = np.asarray(candidate, dtype=float)
    ref = np.asarray(reference, dtype=float)
    if cand.shape != ref.shape:
        raise ValueError(f"shape mismatch in provider parity comparison: {cand.shape} != {ref.shape}")
    diff = cand - ref
    ref_rms = _rms(ref)
    ref_max = _max_abs(ref)
    diff_rms = _rms(diff)
    diff_max = _max_abs(diff)
    return {
        "reference_rms": ref_rms,
        "reference_max_abs": ref_max,
        "candidate_rms": _rms(cand),
        "candidate_max_abs": _max_abs(cand),
        "diff_rms": diff_rms,
        "diff_max_abs": diff_max,
        "diff_rms_rel": None if diff_rms is None or ref_rms is None else float(diff_rms / max(ref_rms, TINY)),
        "diff_max_rel": None if diff_max is None or ref_max is None else float(diff_max / max(ref_max, TINY)),
    }


def _vector_difference_stats(
    candidate_components: tuple[Any, ...],
    reference_components: tuple[Any, ...],
) -> dict[str, float | None]:
    cand = [np.asarray(component, dtype=float) for component in candidate_components]
    ref = [np.asarray(component, dtype=float) for component in reference_components]
    if len(cand) != len(ref):
        raise ValueError("candidate/reference component counts differ")
    for c_arr, r_arr in zip(cand, ref, strict=True):
        if c_arr.shape != r_arr.shape:
            raise ValueError(f"shape mismatch in vector parity comparison: {c_arr.shape} != {r_arr.shape}")
    cand_mag = np.sqrt(sum(c_arr * c_arr for c_arr in cand))
    ref_mag = np.sqrt(sum(r_arr * r_arr for r_arr in ref))
    diff_mag = np.sqrt(sum((c_arr - r_arr) ** 2 for c_arr, r_arr in zip(cand, ref, strict=True)))
    ref_rms = _rms(ref_mag)
    ref_max = _max_abs(ref_mag)
    diff_rms = _rms(diff_mag)
    diff_max = _max_abs(diff_mag)
    return {
        "reference_rms": ref_rms,
        "reference_max_abs": ref_max,
        "candidate_rms": _rms(cand_mag),
        "candidate_max_abs": _max_abs(cand_mag),
        "diff_rms": diff_rms,
        "diff_max_abs": diff_max,
        "diff_rms_rel": None if diff_rms is None or ref_rms is None else float(diff_rms / max(ref_rms, TINY)),
        "diff_max_rel": None if diff_max is None or ref_max is None else float(diff_max / max(ref_max, TINY)),
    }


def _mgrid_domain_payload(sample: Any, bounds: dict[str, float]) -> dict[str, Any]:
    R = np.asarray(sample.R, dtype=float)
    Z = np.asarray(sample.Z, dtype=float)
    margins = {
        "rmin_margin": float(np.nanmin(R) - float(bounds["rmin"])),
        "rmax_margin": float(float(bounds["rmax"]) - np.nanmax(R)),
        "zmin_margin": float(np.nanmin(Z) - float(bounds["zmin"])),
        "zmax_margin": float(float(bounds["zmax"]) - np.nanmax(Z)),
    }
    return {
        "boundary_rmin": float(np.nanmin(R)),
        "boundary_rmax": float(np.nanmax(R)),
        "boundary_zmin": float(np.nanmin(Z)),
        "boundary_zmax": float(np.nanmax(Z)),
        **margins,
        "contained": bool(all(value >= 0.0 for value in margins.values())),
    }


def _provider_parity_payload(
    *,
    mgrid_input: Path,
    coil_params: Any,
    config: ExampleConfig,
    bounds: dict[str, float],
    mgrid_nphi: int,
) -> dict[str, Any]:
    """Compare generated-mgrid and direct-coil fields on the initial VMEC boundary."""

    t0 = time.perf_counter()
    try:
        run = run_free_boundary(
            mgrid_input,
            use_initial_guess=True,
            verbose=False,
            jit_forces=False,
            solver_mode="parity",
        )
        direct_static = {
            "coil_geometry": build_coil_field_geometry(coil_params),
            "regularization_epsilon": getattr(coil_params, "regularization_epsilon", 0.0),
            "chunk_size": getattr(coil_params, "chunk_size", None),
            "cache_scope": "square_coil_profile_provider_parity",
            "jit_sampler": False,
        }
        mgrid_sample = _sample_external_boundary_arrays(
            state=run.state,
            static=run.static,
            plascur=0.0,
        )
        direct_sample = _sample_external_boundary_arrays(
            state=run.state,
            static=run.static,
            plascur=0.0,
            external_field_provider_kind="direct_coils",
            external_field_provider_static=direct_static,
            external_field_provider_params=coil_params,
        )
        return _provider_parity_stats(
            status="completed",
            reference_provider="direct_coils",
            candidate_provider="generated_mgrid",
            sample="initial_boundary_coil_field_only",
            wall_s=float(time.perf_counter() - t0),
            mgrid_sample=mgrid_sample,
            direct_sample=direct_sample,
            config=config,
            bounds=bounds,
            mgrid_nphi=mgrid_nphi,
        )
    except Exception as exc:
        return {
            "status": "failed",
            "error": repr(exc),
            "wall_s": float(time.perf_counter() - t0),
            "mgrid_nphi": int(mgrid_nphi),
            "mgrid_kp_divisible_by_nzeta": bool(int(mgrid_nphi) % max(1, int(config.nzeta)) == 0),
        }


def _provider_parity_stats(
    *,
    status: str,
    reference_provider: str,
    candidate_provider: str,
    sample: str,
    wall_s: float,
    mgrid_sample: Any,
    direct_sample: Any,
    config: ExampleConfig,
    bounds: dict[str, float],
    mgrid_nphi: int,
) -> dict[str, Any]:
    """Return compact direct-coil/generated-mgrid field parity metrics."""

    component_stats = {
        name: _difference_stats(
            getattr(mgrid_sample, name),
            getattr(direct_sample, name),
        )
        for name in ("br_mgrid", "bp_mgrid", "bz_mgrid")
    }
    vacuum_stats = {
        name: _difference_stats(
            getattr(mgrid_sample.vac_ext, name),
            getattr(direct_sample.vac_ext, name),
        )
        for name in ("bnormal", "bnormal_unit", "bu", "bv", "bsqvac")
    }
    field_vector = _vector_difference_stats(
        (mgrid_sample.br_mgrid, mgrid_sample.bp_mgrid, mgrid_sample.bz_mgrid),
        (direct_sample.br_mgrid, direct_sample.bp_mgrid, direct_sample.bz_mgrid),
    )
    bnormal_rel = vacuum_stats["bnormal"]["diff_rms_rel"]
    field_rel = field_vector["diff_rms_rel"]
    return {
        "status": str(status),
        "reference_provider": str(reference_provider),
        "candidate_provider": str(candidate_provider),
        "sample": str(sample),
        "wall_s": float(wall_s),
        "ntheta": int(np.asarray(mgrid_sample.R).shape[0]),
        "nzeta": int(np.asarray(mgrid_sample.R).shape[1]),
        "mgrid_nphi": int(mgrid_nphi),
        "mgrid_kp_divisible_by_nzeta": bool(int(mgrid_nphi) % max(1, int(config.nzeta)) == 0),
        "domain": _mgrid_domain_payload(mgrid_sample, bounds),
        "field_vector": field_vector,
        "components": component_stats,
        "vacuum_channels": vacuum_stats,
        "field_rms_rel_lt_5pct": bool(field_rel is not None and float(field_rel) < 5.0e-2),
        "bnormal_rms_rel_lt_10pct": bool(bnormal_rel is not None and float(bnormal_rel) < 1.0e-1),
    }


def _accepted_provider_parity_payload(
    *,
    run: Any,
    mgrid_input: Path | None,
    coil_params: Any | None,
    config: ExampleConfig,
    bounds: dict[str, float],
    mgrid_nphi: int,
    sample: str,
) -> dict[str, Any]:
    """Compare generated-mgrid and direct-coil fields on an accepted LCFS."""

    if mgrid_input is None or coil_params is None:
        return {"status": "skipped_requires_mgrid_and_direct_coils"}
    t0 = time.perf_counter()
    try:
        mgrid_seed = run_free_boundary(
            mgrid_input,
            use_initial_guess=True,
            verbose=False,
            jit_forces=False,
            solver_mode="parity",
        )
        direct_static = {
            "coil_geometry": build_coil_field_geometry(coil_params),
            "regularization_epsilon": getattr(coil_params, "regularization_epsilon", 0.0),
            "chunk_size": getattr(coil_params, "chunk_size", None),
            "cache_scope": "square_coil_profile_accepted_provider_parity",
            "jit_sampler": False,
        }
        mgrid_sample = _sample_external_boundary_arrays(
            state=run.state,
            static=mgrid_seed.static,
            plascur=0.0,
        )
        direct_sample = _sample_external_boundary_arrays(
            state=run.state,
            static=run.static,
            plascur=0.0,
            external_field_provider_kind="direct_coils",
            external_field_provider_static=direct_static,
            external_field_provider_params=coil_params,
        )
        return _provider_parity_stats(
            status="completed",
            reference_provider="direct_coils",
            candidate_provider="generated_mgrid",
            sample=sample,
            wall_s=float(time.perf_counter() - t0),
            mgrid_sample=mgrid_sample,
            direct_sample=direct_sample,
            config=config,
            bounds=bounds,
            mgrid_nphi=mgrid_nphi,
        )
    except Exception as exc:
        return {
            "status": "failed",
            "sample": str(sample),
            "error": repr(exc),
            "wall_s": float(time.perf_counter() - t0),
            "mgrid_nphi": int(mgrid_nphi),
            "mgrid_kp_divisible_by_nzeta": bool(int(mgrid_nphi) % max(1, int(config.nzeta)) == 0),
        }


def _boundary_projection_payload(config: ExampleConfig) -> dict[str, Any]:
    """Return the Fourier truncation error for the profile boundary deck."""

    return _example_boundary_projection_payload(config)


def _square_axis_uses_spline_controls(config: ExampleConfig) -> bool:
    axis_kind = str(config.plasma_axis_kind).strip().lower()
    return axis_kind in {"spline", "control_spline", "spline_controls", "periodic_spline"}


def _square_axis_controls(config: ExampleConfig) -> SquareAxisSplineControls:
    return (
        config.plasma_axis_spline_controls
        if config.plasma_axis_spline_controls is not None
        else SquareAxisSplineControls.rounded_square(
            axis_half_width=float(config.plasma_axis_half_width),
            corner_radius_factor=float(config.plasma_axis_spline_corner_radius_factor),
        )
    ).validate()


def _square_control_fourier_matrix(config: ExampleConfig, *, symmetry: str = "square") -> tuple[Any, Any]:
    """Build a symmetry-reduced spline-control to Fourier map."""

    controls = _square_axis_controls(config)
    basis = square_axis_spline_symmetric_control_basis(controls, symmetry=symmetry)
    sample_kwargs = {
        key: value
        for key, value in _square_axis_sample_kwargs(config).items()
        if key not in {"axis_kind", "axis_spline_controls"}
    }
    matrix = square_axis_spline_control_fourier_matrix(
        control_basis=basis,
        nfp=int(config.nfp),
        mpol=int(config.mpol),
        ntor=int(config.ntor),
        ntheta_fit=max(64, 4 * int(config.mpol)),
        nzeta_fit=max(128, 8 * int(config.ntor)),
        **sample_kwargs,
    )
    return basis, matrix


def _freeb_edge_control_projection_solver_payload(
    config: ExampleConfig,
    *,
    symmetry: str,
    rcond: float,
) -> dict[str, Any] | None:
    """Return the generic solver payload for square-axis edge controls."""

    symmetry = str(symmetry).strip().lower()
    if symmetry in {"", "none", "off", "false"}:
        return None
    if symmetry not in {"square", "stellarator"}:
        raise ValueError(f"unsupported free-boundary edge-control projection: {symmetry!r}")
    if not _square_axis_uses_spline_controls(config):
        raise ValueError("--freeb-edge-control-projection requires --axis-kind control_spline")
    sample_kwargs = {
        key: value
        for key, value in _square_axis_sample_kwargs(config).items()
        if key not in {"axis_kind", "axis_spline_controls"}
    }
    return square_axis_free_boundary_edge_control_projection_payload(
        controls=_square_axis_controls(config),
        symmetry=symmetry,
        rcond=float(rcond),
        source="profile_square_coil_free_boundary",
        nfp=int(config.nfp),
        mpol=int(config.mpol),
        ntor=int(config.ntor),
        ntheta_fit=max(64, 4 * int(config.mpol)),
        nzeta_fit=max(128, 8 * int(config.ntor)),
        **sample_kwargs,
    )


def _freeb_edge_control_projection_summary(payload: dict[str, Any] | None, *, requested: str) -> dict[str, Any]:
    """Return JSON-safe metadata for an edge-control projection request."""

    requested = str(requested).strip().lower()
    if payload is None:
        return {
            "requested": requested,
            "enabled": False,
            "status": "disabled",
        }
    jacobian = np.asarray(payload.get("control_jacobian"), dtype=float)
    return {
        "requested": requested,
        "enabled": bool(payload.get("enabled", False)),
        "status": "enabled" if bool(payload.get("enabled", False)) else "disabled",
        "basis_symmetry": payload.get("basis_symmetry"),
        "labels": list(payload.get("labels", [])),
        "control_count": int(jacobian.shape[1]) if jacobian.ndim == 2 else None,
        "jacobian_shape": [int(value) for value in jacobian.shape],
        "rcond": float(payload.get("rcond", 1.0e-12)),
    }


def _control_basis_payload(config: ExampleConfig) -> dict[str, Any]:
    """Return compact square-axis spline-control metadata for diagnostics."""

    axis_kind = str(config.plasma_axis_kind).strip().lower()
    if not _square_axis_uses_spline_controls(config):
        return {
            "status": "not_applicable_for_axis_kind",
            "axis_kind": axis_kind,
        }
    controls = _square_axis_controls(config)
    bases: dict[str, Any] = {}
    for symmetry in ("square", "stellarator"):
        basis = square_axis_spline_symmetric_control_basis(controls, symmetry=symmetry)
        reduced_radius = basis.project_radius(controls.radius)
        bases[symmetry] = {
            "symmetry": basis.symmetry,
            "labels": list(basis.labels),
            "reduced_count": len(basis.labels),
            "reduced_radius": [float(value) for value in reduced_radius],
            "expansion_matrix_shape": [int(value) for value in basis.matrix.shape],
            "expansion_matrix": np.asarray(basis.matrix, dtype=float),
        }
    return {
        "status": "available",
        "axis_kind": axis_kind,
        "control_count": int(np.asarray(controls.radius).size),
        "control_zeta": np.asarray(controls.zeta, dtype=float),
        "control_radius": np.asarray(controls.radius, dtype=float),
        "bases": bases,
    }


def _control_fourier_map_for_symmetry(config: ExampleConfig, *, symmetry: str) -> dict[str, Any]:
    axis_kind = str(config.plasma_axis_kind).strip().lower()
    try:
        controls = _square_axis_controls(config)
        sample_kwargs = {
            key: value
            for key, value in _square_axis_sample_kwargs(config).items()
            if key not in {"axis_kind", "axis_spline_controls"}
        }
        return square_axis_spline_control_fourier_map_status(
            controls=controls,
            symmetry=symmetry,
            nfp=int(config.nfp),
            mpol=int(config.mpol),
            ntor=int(config.ntor),
            ntheta_fit=max(64, 4 * int(config.mpol)),
            nzeta_fit=max(128, 8 * int(config.ntor)),
            **sample_kwargs,
        )
    except Exception as exc:
        return {
            "status": f"failed:{type(exc).__name__}",
            "axis_kind": axis_kind,
            "basis_symmetry": str(symmetry),
            "error": repr(exc),
        }


def _control_fourier_map_payload(config: ExampleConfig) -> dict[str, Any]:
    """Return conditioning diagnostics for the reduced control-to-Fourier map."""

    axis_kind = str(config.plasma_axis_kind).strip().lower()
    if not _square_axis_uses_spline_controls(config):
        return {
            "status": "not_applicable_for_axis_kind",
            "axis_kind": axis_kind,
        }
    payload = _control_fourier_map_for_symmetry(config, symmetry="square")
    if payload is not None:
        payload["candidate_bases"] = {
            symmetry: _control_fourier_map_for_symmetry(config, symmetry=symmetry)
            for symmetry in ("square", "stellarator")
        }
    return payload


def _spline_bridge_payload(
    *,
    config: ExampleConfig,
    projection: dict[str, Any],
    resolution_deck: dict[str, Any],
    control_basis: dict[str, Any],
    control_fourier_map: dict[str, Any],
) -> dict[str, Any]:
    """State whether the square-axis spline controls are solver-native."""

    axis_kind = str(config.plasma_axis_kind).strip().lower()
    uses_controls = bool(_square_axis_uses_spline_controls(config))
    mode_count = int(
        resolution_deck.get(
            "mode_count",
            projection.get("mode_count", control_fourier_map.get("mode_count", 0)),
        )
        or 0
    )
    bases = control_basis.get("bases", {}) if isinstance(control_basis, dict) else {}
    square_basis = bases.get("square", {}) if isinstance(bases, dict) else {}
    stellarator_basis = bases.get("stellarator", {}) if isinstance(bases, dict) else {}
    square_count = square_basis.get("reduced_count")
    stellarator_count = stellarator_basis.get("reduced_count")
    status = "spline_control_to_fourier_bridge" if uses_controls else "fourier_boundary_only"
    next_action = (
        "profile_freeb_edge_control_projection"
        if resolution_deck.get("status") == "production_ready"
        else "repair_projection_or_zeta_deck_before_solver_profiling"
    )
    return {
        "status": status,
        "axis_kind": axis_kind,
        "real_space_axis_basis": "periodic_spline_controls" if uses_controls else "sampled_fourier_target",
        "nonlinear_solver_boundary_basis": "vmec_fourier_coefficients",
        "solver_native_spline_controls": False,
        "optional_solver_edge_control_projection": uses_controls,
        "requires_fourier_projection": True,
        "reduced_square_control_count": None if square_count is None else int(square_count),
        "reduced_stellarator_control_count": None if stellarator_count is None else int(stellarator_count),
        "fourier_mode_count": mode_count,
        "fourier_boundary_channel_count": 4 * mode_count,
        "projection_status": resolution_deck.get("status"),
        "projection_max_abs_component_error": projection.get("max_abs_component_error"),
        "projection_target_max_component_error": resolution_deck.get("projection_target_max_component_error"),
        "control_map_status": control_fourier_map.get("status") if isinstance(control_fourier_map, dict) else None,
        "can_reduce_input_shape_dofs": uses_controls,
        "can_reduce_nonlinear_solver_dofs": uses_controls,
        "recommended_next_action": next_action,
        "interpretation": (
            "The square-axis spline path still enters VMEC through Fourier coefficients, but the vmec_jax "
            "free-boundary edge update can now be constrained to a reduced spline-control subspace for "
            "A/B convergence profiles."
        ),
    }


def _resolution_deck_payload(
    *,
    config: ExampleConfig,
    projection: dict[str, Any],
    mgrid_nphi: int,
    target_error: float | None,
    include_recommendation: bool = False,
) -> dict[str, Any]:
    """Summarize cheap pre-solve checks for a square-hybrid Fourier deck."""

    payload = square_axis_resolution_deck_status(
        projection=projection,
        mpol=int(config.mpol),
        ntor=int(config.ntor),
        ns=int(config.ns),
        nzeta=int(config.nzeta),
        mgrid_nphi=int(mgrid_nphi),
        target_max_component_error=target_error,
    )
    if include_recommendation and target_error is not None and payload.get("projection_meets_gate") is False:
        recommendation = recommend_square_axis_stellarator_mirror_hybrid_resolution(
            target_max_component_error=float(target_error),
            mpol=int(config.mpol),
            ntor=int(config.ntor),
            max_mpol=max(8, int(config.mpol) + 2),
            max_ntor=max(32, int(config.ntor) + 8),
            nfp=int(config.nfp),
            ns_array=[int(value) for value in config.ns_array],
            niter_array=[int(value) for value in config.niter_array],
            ftol_array=[float(value) for value in config.ftol_array],
            phiedge=float(config.phiedge),
            **_square_axis_sample_kwargs(config),
        )
        payload["recommended_deck"] = recommendation.get("recommended")
    return payload


def _enforce_boundary_projection_gate(
    *,
    config: ExampleConfig,
    projection: dict[str, Any],
    resolution_deck: dict[str, Any],
    limit: float | None,
) -> None:
    if limit is None:
        return
    observed = float(projection.get("max_abs_component_error", np.inf))
    if not np.isfinite(observed):
        raise ValueError("square-hybrid boundary projection error is not finite")
    if observed > float(limit):
        recommendation = recommend_square_axis_stellarator_mirror_hybrid_resolution(
            target_max_component_error=float(limit),
            mpol=int(config.mpol),
            ntor=int(config.ntor),
            max_mpol=max(8, int(config.mpol) + 2),
            max_ntor=max(32, int(config.ntor) + 8),
            nfp=int(config.nfp),
            ns_array=[int(value) for value in config.ns_array],
            niter_array=[int(value) for value in config.niter_array],
            ftol_array=[float(value) for value in config.ftol_array],
            phiedge=float(config.phiedge),
            **_square_axis_sample_kwargs(config),
        )
        suggested = recommendation["recommended"]
        raise ValueError(
            "square-hybrid boundary projection error is too large for this production profile: "
            f"max_abs_component_error={observed:.3e} exceeds {float(limit):.3e} "
            f"for MPOL={int(config.mpol)}, NTOR={int(config.ntor)}, NZETA={int(config.nzeta)}. "
            "Suggested finite Fourier closure for the current spline-smoothed target: "
            f"MPOL={int(suggested['mpol'])}, NTOR={int(suggested['ntor'])}, "
            f"NZETA>={int(suggested['recommended_nzeta'])} "
            f"(projection error {float(suggested['max_abs_component_error']):.3e}). "
            "Increase MPOL/NTOR/NZETA or pass --max-boundary-projection-error none for a diagnostic-only run."
        )
    reasons = [str(reason) for reason in resolution_deck.get("reasons", [])]
    if not reasons:
        return
    recommended_nzeta = int(resolution_deck.get("recommended_nzeta", recommended_square_axis_nzeta(config.ntor)))
    raise ValueError(
        "square-hybrid resolution deck is not production-ready for a finite projection gate: "
        f"reasons={','.join(reasons)} for MPOL={int(config.mpol)}, NTOR={int(config.ntor)}, "
        f"NZETA={int(config.nzeta)}, MGRID_NPHI={int(resolution_deck.get('mgrid_nphi', config.nzeta))}. "
        f"Use NZETA>={recommended_nzeta}, keep --mgrid-nphi a multiple of NZETA, "
        "or pass --max-boundary-projection-error none for a diagnostic-only run."
    )


def _vmec2000_row_payload(row: Any) -> dict[str, Any]:
    total = float(row.fsqr) + float(row.fsqz) + float(row.fsql)
    max_component = _vmec2000_max_component(row)
    return {
        "it": int(row.it),
        "fsqr": float(row.fsqr),
        "fsqz": float(row.fsqz),
        "fsql": float(row.fsql),
        "total": total,
        "max_component": max_component,
        "delt0r": row.delt0r,
        "delbsq": row.delbsq,
        "fedge": row.fedge,
    }


def _vmec2000_max_component(row: Any) -> float:
    return float(max(float(row.fsqr), float(row.fsqz), float(row.fsql)))


def _vmec2000_strict_components_met(row: Any, requested_ftol: float | None) -> bool | None:
    if requested_ftol is None:
        return None
    requested = float(requested_ftol)
    if not np.isfinite(requested) or requested <= 0.0:
        return None
    return bool(_vmec2000_max_component(row) <= requested)


def _vmec2000_tail_plateau_payload(
    rows: list[Any],
    *,
    stage_ftol: float | None,
    length: int = 12,
    rel_span_tol: float = 0.02,
) -> dict[str, Any]:
    """Classify the most recent VMEC2000 residual tail."""

    tail: list[dict[str, float | int]] = []
    for row in list(rows)[-int(length) :]:
        try:
            payload = row if isinstance(row, dict) else _vmec2000_row_payload(row)
            total = float(payload["total"])
            max_component = float(payload["max_component"])
            iteration = int(payload["it"])
        except Exception:
            continue
        if np.isfinite(total) and total > 0.0 and np.isfinite(max_component):
            tail.append({"it": iteration, "total": total, "max_component": max_component})
    out: dict[str, Any] = {
        "window": int(len(tail)),
        "status": "insufficient_tail",
        "stage_ftol": None if stage_ftol is None else float(stage_ftol),
        "total_rel_span": None,
        "total_last_over_min": None,
        "monotone_decrease_fraction": None,
    }
    if len(tail) < 3:
        return out
    totals = np.asarray([float(row["total"]) for row in tail], dtype=float)
    max_components = np.asarray([float(row["max_component"]) for row in tail], dtype=float)
    diffs = np.diff(totals)
    total_min = float(np.min(totals))
    total_max = float(np.max(totals))
    total_last = float(totals[-1])
    rel_span = float((total_max - total_min) / max(total_min, TINY))
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
            "first_iter": int(tail[0]["it"]),
            "last_iter": int(tail[-1]["it"]),
            "total_first": float(totals[0]),
            "total_last": total_last,
            "total_min": total_min,
            "total_max": total_max,
            "total_rel_span": rel_span,
            "total_last_over_min": float(total_last / max(total_min, TINY)),
            "max_component_last": float(max_components[-1]),
            "monotone_decrease_fraction": float(np.mean(diffs < 0.0)),
        }
    )
    return out


def _vmec2000_stage_payload(stage: Any) -> dict[str, Any]:
    rows = list(getattr(stage, "rows", []) or [])
    totals = [float(row.fsqr) + float(row.fsqz) + float(row.fsql) for row in rows]
    last = rows[-1] if rows else None
    ftolv = float(getattr(stage, "ftolv", float("nan")))
    return {
        "ns": int(getattr(stage, "ns", -1)),
        "niter": int(getattr(stage, "niter", -1)),
        "ftolv": ftolv,
        "iteration_row_count": len(rows),
        "min_total": None if not totals else float(np.nanmin(np.asarray(totals, dtype=float))),
        "last_row": None if last is None else _vmec2000_row_payload(last),
        "final_max_component": None if last is None else _vmec2000_max_component(last),
        "strict_components_met": None if last is None else _vmec2000_strict_components_met(last, ftolv),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    virtual_casing_pythonpath = _configure_virtual_casing_pythonpath(args.virtual_casing_pythonpath)
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    recommended_nzeta = recommended_square_axis_nzeta(int(args.ntor))
    resolved_nzeta = int(recommended_nzeta if args.nzeta is None else args.nzeta)
    mgrid_nphi = int(resolved_nzeta if args.mgrid_nphi is None else args.mgrid_nphi)
    if mgrid_nphi % max(1, resolved_nzeta) != 0 and not bool(args.resolution_diagnostics_only):
        raise ValueError(
            f"--mgrid-nphi={mgrid_nphi} is incompatible with --nzeta={resolved_nzeta} for VMEC-plane "
            "mgrid sampling; omit --mgrid-nphi or use a multiple of --nzeta."
        )
    ns_array, niter_array, ftol_array = _resolve_schedule(args)
    if bool(args.enforce_recommended_nzeta) and resolved_nzeta < recommended_nzeta:
        raise ValueError(
            f"NZETA={resolved_nzeta} is underresolved for NTOR={int(args.ntor)}; use at least {recommended_nzeta}"
        )
    config = ExampleConfig(
        outdir=outdir,
        betas_percent=(float(args.beta_percent),),
        n_coils_per_side=int(args.n_coils_per_side),
        coil_segments=int(args.coil_segments),
        coil_chunk_size=args.coil_chunk_size,
        plasma_axis_kind=str(args.axis_kind),
        plasma_axis_spline_corner_radius_factor=float(args.axis_corner_factor),
        side_power=float(args.side_power),
        corner_power=float(args.corner_power),
        mpol=int(args.mpol),
        ntor=int(args.ntor),
        ns=int(ns_array[-1]),
        ns_array=ns_array,
        nzeta=resolved_nzeta,
        max_iter=int(niter_array[-1]),
        ftol=float(ftol_array[-1]),
        phiedge=float(args.phiedge) if args.phiedge is not None else ExampleConfig().phiedge,
        niter_array=niter_array,
        ftol_array=ftol_array,
        use_multigrid_schedule=len(ns_array) > 1,
        nstep=int(args.nstep),
        delt=float(args.delt),
        nvacskip=int(args.nvacskip),
        free_boundary_activate_fsq=float(args.activate_fsq),
        beta_continuation_restart=False,
        jit_forces=bool(args.jit_forces),
        write_plots=False,
    )
    solver_mode = None if str(args.solver_mode).strip().lower() == "auto" else str(args.solver_mode)
    ns_values, niter_values, ftol_values = _stage_values(config)
    _log_step(
        "building square-coil configuration "
        f"beta={float(args.beta_percent):g}%, mpol={int(args.mpol)}, ntor={int(args.ntor)}, "
        f"ns={ns_values}, nzeta={resolved_nzeta}, "
        f"side_power={float(args.side_power):g}, corner_power={float(args.corner_power):g}"
    )
    boundary_projection = _boundary_projection_payload(config)
    control_basis = _control_basis_payload(config)
    control_fourier_map = _control_fourier_map_payload(config)
    resolution_deck = _resolution_deck_payload(
        config=config,
        projection=boundary_projection,
        mgrid_nphi=mgrid_nphi,
        target_error=args.max_boundary_projection_error,
        include_recommendation=bool(args.resolution_diagnostics_only),
    )
    spline_bridge = _spline_bridge_payload(
        config=config,
        projection=boundary_projection,
        resolution_deck=resolution_deck,
        control_basis=control_basis,
        control_fourier_map=control_fourier_map,
    )
    if bool(args.resolution_diagnostics_only):
        payload = {
            "schema": "square_coil_free_boundary_backend_profile",
            "configuration": {
                "beta_percent": float(args.beta_percent),
                "mpol": int(args.mpol),
                "ntor": int(args.ntor),
                "ns": int(ns_array[-1]),
                "nzeta": resolved_nzeta,
                "nzeta_auto": bool(args.nzeta is None),
                "recommended_nzeta": int(recommended_nzeta),
                "nzeta_underrecommended": bool(resolved_nzeta < int(recommended_nzeta)),
                "max_iter": int(niter_array[-1]),
                "ftol": float(ftol_array[-1]),
                "axis_kind": str(args.axis_kind),
                "side_power": float(args.side_power),
                "corner_power": float(args.corner_power),
                "max_boundary_projection_error": None
                if args.max_boundary_projection_error is None
                else float(args.max_boundary_projection_error),
                "use_multigrid_schedule": bool(len(ns_array) > 1),
                "ns_array": ns_values,
                "niter_array": niter_values,
                "ftol_array": ftol_values,
                "resolution_diagnostics_only": True,
                "accepted_provider_parity": bool(args.accepted_provider_parity),
                "freeb_anderson_pressure": bool(args.freeb_anderson_pressure),
                "freeb_jax_nestor_operator": bool(args.freeb_jax_nestor_operator),
                "freeb_jax_nestor_jit_operator": bool(args.freeb_jax_nestor_jit_operator),
                "freeb_include_edge": None if args.freeb_include_edge is None else bool(args.freeb_include_edge),
                "freeb_dense_solve_mode": None
                if args.freeb_dense_solve_mode is None
                else str(args.freeb_dense_solve_mode),
                "freeb_experimental_fouri_matrix": None
                if args.freeb_experimental_fouri_matrix is None
                else bool(args.freeb_experimental_fouri_matrix),
                "freeb_add_analytic_bvec": None
                if args.freeb_add_analytic_bvec is None
                else bool(args.freeb_add_analytic_bvec),
                "freeb_edge_control_projection": str(args.freeb_edge_control_projection),
                "freeb_edge_control_rcond": float(args.freeb_edge_control_rcond),
                "jax_hot_restart_count": int(args.jax_hot_restart_count),
                "jax_hot_restart_iters": None
                if args.jax_hot_restart_iters is None
                else int(args.jax_hot_restart_iters),
                "jax_hot_restart_policy": str(args.jax_hot_restart_policy),
                "jax_hot_restart_always": bool(args.jax_hot_restart_always),
                "jax_initial_restart_wout": None
                if args.jax_initial_restart_wout is None
                else str(args.jax_initial_restart_wout),
                "virtual_casing_quad_factor": int(args.virtual_casing_quad_factor),
                "virtual_casing_chunk_size": args.virtual_casing_chunk_size,
                "virtual_casing_target_chunk_size": args.virtual_casing_target_chunk_size,
                "virtual_casing_pythonpath": virtual_casing_pythonpath,
            },
            "mgrid": {
                "created": False,
                "nphi": int(mgrid_nphi),
            },
            "boundary_projection": boundary_projection,
            "control_basis": control_basis,
            "control_fourier_map": control_fourier_map,
            "spline_bridge": spline_bridge,
            "resolution_deck": resolution_deck,
            "provider_parity": None,
            "backends": {},
        }
        report = outdir / "square_coil_free_boundary_backend_profile.json"
        _log_step(f"writing resolution diagnostics {report}")
        report.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")
        print(report)
        return 0
    _enforce_boundary_projection_gate(
        config=config,
        projection=boundary_projection,
        resolution_deck=resolution_deck,
        limit=args.max_boundary_projection_error,
    )
    coils = build_square_coils(config)
    label = _case_label(float(args.beta_percent))
    direct_input = outdir / f"input.square_{label}_direct"
    mgrid_input = outdir / f"input.square_{label}_mgrid"
    direct_wout = outdir / f"wout_square_{label}_direct.nc"
    mgrid_wout = outdir / f"wout_square_{label}_mgrid.nc"
    mgrid_path = outdir / "mgrid_square_coils.nc"
    need_mgrid = (
        (not bool(args.skip_mgrid)) or (not bool(args.skip_provider_parity)) or bool(args.run_vmec2000)
        or bool(args.accepted_provider_parity)
    )

    base_indata = make_free_boundary_indata(config, beta_percent=float(args.beta_percent))
    write_indata(direct_input, base_indata)
    _log_step(f"wrote direct input {direct_input}")
    bounds = _mgrid_bounds(
        base_indata,
        padding_fraction=float(args.mgrid_padding_fraction),
        min_padding=float(args.mgrid_min_padding),
    )
    if need_mgrid:
        _log_step(f"writing generated mgrid {mgrid_path}")
        mgrid_write_chunk_size = 512 if args.coil_chunk_size is None else args.coil_chunk_size
        write_mgrid_from_coils(
            mgrid_path,
            coils.params,
            rmin=bounds["rmin"],
            rmax=bounds["rmax"],
            zmin=bounds["zmin"],
            zmax=bounds["zmax"],
            nr=int(args.mgrid_nr),
            nz=int(args.mgrid_nz),
            nphi=mgrid_nphi,
            nfp=int(config.nfp),
            chunk_size=mgrid_write_chunk_size,
        )
        mgrid_indata = deepcopy(base_indata)
        mgrid_indata.scalars["MGRID_FILE"] = mgrid_path.name
        write_indata(mgrid_input, mgrid_indata)
        _log_step(f"wrote mgrid input {mgrid_input}")
    else:
        _log_step("skipping generated mgrid path")

    payload: dict[str, Any] = {
        "schema": "square_coil_free_boundary_backend_profile",
        "configuration": {
            "beta_percent": float(args.beta_percent),
            "mpol": int(args.mpol),
            "ntor": int(args.ntor),
            "ns": int(ns_array[-1]),
            "nzeta": resolved_nzeta,
            "nzeta_auto": bool(args.nzeta is None),
            "recommended_nzeta": int(recommended_nzeta),
            "nzeta_underrecommended": bool(resolved_nzeta < int(recommended_nzeta)),
            "max_iter": int(niter_array[-1]),
            "ftol": float(ftol_array[-1]),
            "nstep": int(config.nstep),
            "solver_mode": None if solver_mode is None else str(solver_mode),
            "return_best_scored_state": bool(args.return_best_scored_state),
            "freeb_anderson_pressure": bool(args.freeb_anderson_pressure),
            "freeb_jax_nestor_operator": bool(args.freeb_jax_nestor_operator),
            "freeb_jax_nestor_jit_operator": bool(args.freeb_jax_nestor_jit_operator),
            "freeb_include_edge": None if args.freeb_include_edge is None else bool(args.freeb_include_edge),
            "freeb_dense_solve_mode": None
            if args.freeb_dense_solve_mode is None
            else str(args.freeb_dense_solve_mode),
            "freeb_experimental_fouri_matrix": None
            if args.freeb_experimental_fouri_matrix is None
            else bool(args.freeb_experimental_fouri_matrix),
            "freeb_add_analytic_bvec": None
            if args.freeb_add_analytic_bvec is None
            else bool(args.freeb_add_analytic_bvec),
            "freeb_edge_control_projection": str(args.freeb_edge_control_projection),
            "freeb_edge_control_rcond": float(args.freeb_edge_control_rcond),
            "jax_hot_restart_count": int(args.jax_hot_restart_count),
            "jax_hot_restart_iters": None
            if args.jax_hot_restart_iters is None
            else int(args.jax_hot_restart_iters),
            "jax_hot_restart_policy": str(args.jax_hot_restart_policy),
            "jax_hot_restart_always": bool(args.jax_hot_restart_always),
            "jax_initial_restart_wout": None
            if args.jax_initial_restart_wout is None
            else str(args.jax_initial_restart_wout),
            "direct_static_cache": bool(args.direct_static_cache),
            "jit_direct_sampler": bool(args.jit_direct_sampler),
            "direct_trial_bsqvac_resample": bool(args.direct_trial_bsqvac_resample),
            "verbose_solver": bool(args.verbose_solver),
            "virtual_casing_diagnostics": bool(args.virtual_casing_diagnostics),
            "virtual_casing_quad_factor": int(args.virtual_casing_quad_factor),
            "virtual_casing_chunk_size": args.virtual_casing_chunk_size,
            "virtual_casing_target_chunk_size": args.virtual_casing_target_chunk_size,
            "virtual_casing_pythonpath": virtual_casing_pythonpath,
            "resolution_diagnostics_only": bool(args.resolution_diagnostics_only),
            "accepted_provider_parity": bool(args.accepted_provider_parity),
            "phiedge": float(config.phiedge),
            "delt": float(args.delt),
            "activate_fsq": float(args.activate_fsq),
            "nvacskip": int(config.nvacskip),
            "axis_kind": str(args.axis_kind),
            "axis_corner_factor": float(args.axis_corner_factor),
            "side_power": float(args.side_power),
            "corner_power": float(args.corner_power),
            "max_boundary_projection_error": None
            if args.max_boundary_projection_error is None
            else float(args.max_boundary_projection_error),
            "coil_chunk_size": None if args.coil_chunk_size is None else int(args.coil_chunk_size),
            "use_multigrid_schedule": bool(len(ns_array) > 1),
            "ns_array": ns_values,
            "niter_array": niter_values,
            "ftol_array": ftol_values,
        },
        "mgrid": {
            "created": bool(need_mgrid),
            "path": mgrid_path,
            "nr": int(args.mgrid_nr),
            "nz": int(args.mgrid_nz),
            "nphi": int(mgrid_nphi),
            "write_chunk_size": None
            if not need_mgrid
            else int(512 if args.coil_chunk_size is None else args.coil_chunk_size),
            **bounds,
        },
        "boundary_projection": boundary_projection,
        "control_basis": control_basis,
        "control_fourier_map": control_fourier_map,
        "spline_bridge": spline_bridge,
        "resolution_deck": resolution_deck,
        "provider_parity": None,
        "backends": {},
    }
    if not bool(args.skip_provider_parity):
        _log_step("running provider-parity diagnostic")
        payload["provider_parity"] = _provider_parity_payload(
            mgrid_input=mgrid_input,
            coil_params=coils.params,
            config=config,
            bounds=bounds,
            mgrid_nphi=mgrid_nphi,
        )
    if not args.skip_direct:
        _log_step("running vmec_jax direct-coil backend")
        payload["backends"]["vmec_jax_direct"] = _run_jax_backend(
            input_path=direct_input,
            wout_path=direct_wout,
            config=config,
            beta_percent=float(args.beta_percent),
            direct_params=coils.params,
            solver_mode=solver_mode,
            return_best_scored_state=bool(args.return_best_scored_state),
            freeb_anderson_pressure=bool(args.freeb_anderson_pressure),
            freeb_jax_nestor_operator=bool(args.freeb_jax_nestor_operator),
            freeb_jax_nestor_jit_operator=bool(args.freeb_jax_nestor_jit_operator),
            freeb_include_edge=args.freeb_include_edge,
            freeb_dense_solve_mode=args.freeb_dense_solve_mode,
            freeb_experimental_fouri_matrix=args.freeb_experimental_fouri_matrix,
            freeb_add_analytic_bvec=args.freeb_add_analytic_bvec,
            freeb_edge_control_projection=str(args.freeb_edge_control_projection),
            freeb_edge_control_rcond=float(args.freeb_edge_control_rcond),
            direct_static_cache=bool(args.direct_static_cache),
            jit_direct_sampler=bool(args.jit_direct_sampler),
            direct_trial_bsqvac_resample=bool(args.direct_trial_bsqvac_resample),
            jax_hot_restart_count=int(args.jax_hot_restart_count),
            jax_hot_restart_iters=args.jax_hot_restart_iters,
            jax_hot_restart_policy=str(args.jax_hot_restart_policy),
            jax_hot_restart_always=bool(args.jax_hot_restart_always),
            jax_initial_restart_wout=args.jax_initial_restart_wout,
            verbose_solver=bool(args.verbose_solver),
            virtual_casing_diagnostics=bool(args.virtual_casing_diagnostics),
            virtual_casing_quad_factor=int(args.virtual_casing_quad_factor),
            virtual_casing_chunk_size=args.virtual_casing_chunk_size,
            virtual_casing_target_chunk_size=args.virtual_casing_target_chunk_size,
            accepted_provider_parity=bool(args.accepted_provider_parity),
            accepted_provider_parity_mgrid_input=mgrid_input,
            accepted_provider_parity_coil_params=coils.params,
            accepted_provider_parity_bounds=bounds,
            accepted_provider_parity_mgrid_nphi=mgrid_nphi,
            accepted_provider_parity_sample="accepted_boundary_direct_backend",
        )
    if not args.skip_mgrid:
        _log_step("running vmec_jax generated-mgrid backend")
        payload["backends"]["vmec_jax_mgrid"] = _run_jax_backend(
            input_path=mgrid_input,
            wout_path=mgrid_wout,
            config=config,
            beta_percent=float(args.beta_percent),
            direct_params=None,
            solver_mode=solver_mode,
            return_best_scored_state=bool(args.return_best_scored_state),
            freeb_anderson_pressure=bool(args.freeb_anderson_pressure),
            freeb_jax_nestor_operator=bool(args.freeb_jax_nestor_operator),
            freeb_jax_nestor_jit_operator=bool(args.freeb_jax_nestor_jit_operator),
            freeb_include_edge=args.freeb_include_edge,
            freeb_dense_solve_mode=args.freeb_dense_solve_mode,
            freeb_experimental_fouri_matrix=args.freeb_experimental_fouri_matrix,
            freeb_add_analytic_bvec=args.freeb_add_analytic_bvec,
            freeb_edge_control_projection=str(args.freeb_edge_control_projection),
            freeb_edge_control_rcond=float(args.freeb_edge_control_rcond),
            jax_hot_restart_count=int(args.jax_hot_restart_count),
            jax_hot_restart_iters=args.jax_hot_restart_iters,
            jax_hot_restart_policy=str(args.jax_hot_restart_policy),
            jax_hot_restart_always=bool(args.jax_hot_restart_always),
            jax_initial_restart_wout=args.jax_initial_restart_wout,
            verbose_solver=bool(args.verbose_solver),
            virtual_casing_diagnostics=False,
            accepted_provider_parity=bool(args.accepted_provider_parity),
            accepted_provider_parity_mgrid_input=mgrid_input,
            accepted_provider_parity_coil_params=coils.params,
            accepted_provider_parity_bounds=bounds,
            accepted_provider_parity_mgrid_nphi=mgrid_nphi,
            accepted_provider_parity_sample="accepted_boundary_mgrid_backend",
        )
    if bool(args.run_vmec2000):
        exe = args.vmec2000_exec or find_vmec2000_exec()
        if exe is None:
            _log_step("skipping VMEC2000 backend because xvmec2000 was not found")
            payload["backends"]["vmec2000_mgrid"] = {"status": "skipped_missing_xvmec2000"}
        else:
            _log_step(f"running VMEC2000 backend with {exe}")
            t0 = time.perf_counter()
            vmec2000_workdir = outdir / "vmec2000_mgrid"
            monitor_stop, monitor_thread = _start_vmec2000_progress_monitor(
                outdir=outdir,
                workdir=vmec2000_workdir,
            )
            try:
                run = run_xvmec2000(
                    mgrid_input,
                    exec_path=exe,
                    workdir=vmec2000_workdir,
                    timeout_s=float(args.vmec2000_timeout),
                    keep_workdir=True,
                )
                rows = [row for stage in run.stages for row in stage.rows]
                last = rows[-1] if rows else None
                totals = [float(row.fsqr) + float(row.fsqz) + float(row.fsql) for row in rows]
                final_max_component = None if last is None else _vmec2000_max_component(last)
                stage_ftol = float(run.stages[-1].ftolv) if run.stages else float(config.ftol)
                strict_components_met = (
                    None if last is None else _vmec2000_strict_components_met(last, float(config.ftol))
                )
                tail_plateau = _vmec2000_tail_plateau_payload(rows, stage_ftol=stage_ftol)
                payload["backends"]["vmec2000_mgrid"] = {
                    "status": "completed" if run.returncode == 0 else "nonzero_exit",
                    "returncode": int(run.returncode),
                    "wall_s": float(time.perf_counter() - t0),
                    "exec": exe,
                    "workdir": run.workdir,
                    "threed1": run.threed1_path,
                    "stdout_tail": run.stdout.splitlines()[-40:],
                    "stderr_tail": run.stderr.splitlines()[-40:],
                    "threed1_tail": _tail_lines(run.threed1_path, lines=80),
                    "vacuum_grid_exceeded_count": _vacuum_grid_exceeded_count(run.threed1_path),
                    "stage_summaries": [_vmec2000_stage_payload(stage) for stage in run.stages],
                    "iteration_row_count": len(rows),
                    "first_rows": [_vmec2000_row_payload(row) for row in rows[:8]],
                    "tail_rows": [_vmec2000_row_payload(row) for row in rows[-12:]],
                    "last_row": None if last is None else _vmec2000_row_payload(last),
                    "min_total": None if not totals else float(np.nanmin(np.asarray(totals, dtype=float))),
                    "final_max_component": final_max_component,
                    "strict_components_met": strict_components_met,
                    "strict_convergence": _strict_convergence_verdict(
                        components=(
                            None if last is None else float(last.fsqr),
                            None if last is None else float(last.fsqz),
                            None if last is None else float(last.fsql),
                        ),
                        requested_ftol=float(config.ftol),
                        free_boundary_active=True if rows else None,
                        final_residual_recomputed=True if rows else None,
                        tail_plateau_status=tail_plateau.get("status"),
                        fresh_residual_required=False,
                    ),
                    "free_boundary_promotion": free_boundary_promotion_status(
                        beta_percent=float(args.beta_percent),
                        strict_components_met=strict_components_met,
                        final_residual_recomputed=True,
                        direct_coil_backend=False,
                        require_fresh_residual=False,
                    ),
                    "tail_plateau": tail_plateau,
                }
            except Exception as exc:
                payload["backends"]["vmec2000_mgrid"] = {
                    "status": "failed",
                    "error": repr(exc),
                    **_partial_vmec2000_payload(vmec2000_workdir),
                }
            finally:
                monitor_stop.set()
                monitor_thread.join(timeout=2.0)
                try:
                    _write_partial_vmec2000_payload(outdir=outdir, workdir=vmec2000_workdir)
                except Exception:
                    pass

    report = outdir / "square_coil_free_boundary_backend_profile.json"
    _log_step(f"writing profile report {report}")
    report.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
