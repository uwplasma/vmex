"""Run a small convergence grid for the toroidal hybrid boundary."""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import os
from pathlib import Path
import sys
import tempfile
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import vmec_jax as vj
from vmec_jax.namelist import write_indata
from vmec_jax.toroidal_hybrid import (
    evaluate_toroidal_hybrid_indata_boundary,
    sample_toroidal_stellarator_mirror_hybrid_boundary,
    toroidal_hybrid_cross_section_anisotropy,
    toroidal_hybrid_cross_section_orientation,
    toroidal_stellarator_mirror_hybrid_indata,
    toroidal_stellarator_mirror_hybrid_metrics,
)
from vmec_jax.vmec2000_exec import flatten_threed1, run_xvmec2000, threed1_fsq_total
from vmec_jax.wout import read_wout


def _parse_ints(text: str) -> list[int]:
    return [int(item.strip()) for item in str(text).split(",") if item.strip()]


def _parse_case_filters(text: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(text).split(",") if item.strip())


def _parse_path_args(values: list[str] | None) -> list[Path]:
    paths: list[Path] = []
    for value in values or []:
        for item in str(value).split(","):
            item = item.strip()
            if item:
                paths.append(Path(item).expanduser())
    return paths


def _case_matches_filters(case: str, filters: tuple[str, ...]) -> bool:
    return not filters or any(fnmatch.fnmatchcase(case, pattern) for pattern in filters)


def _parse_mode_pairs(text: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        left, sep, right = item.partition(":")
        if not sep:
            raise ValueError("mode pairs must use MPOL:NTOR, for example 5:4,6:5")
        pairs.append((int(left), int(right)))
    return pairs


_SHAPE_CASE_PRESETS = {
    "default": {},
    "sharp": {
        "side_minor_modulation": 0.16,
        "side_elongation": 0.35,
        "side_power": 2.0,
        "corner_amplitude": 0.025,
        "corner_ellipticity": 0.22,
        "corner_rotation": 0.42,
        "corner_power": 2.0,
    },
}
_RESOLUTION_PRESETS = {
    "manual": {
        "description": "Use --ns-array and --mode-pairs exactly as provided.",
        "ns_array": None,
        "mode_pairs": None,
        "target_resolution_ladder": False,
    },
    "smoke": {
        "description": "Low-cost geometry and plotting smoke ladder.",
        "ns_array": (7, 9),
        "mode_pairs": ((5, 20),),
        "target_resolution_ladder": False,
    },
    "promotion": {
        "description": "Moderate promotion ladder before expensive solved rows.",
        "ns_array": (7, 9, 15),
        "mode_pairs": ((5, 20),),
        "target_resolution_ladder": False,
    },
    "target": {
        "description": "Target ladder for solved/parity convergence campaigns.",
        "ns_array": (7, 9, 15),
        "mode_pairs": ((5, 20), (6, 24)),
        "target_resolution_ladder": True,
    },
}

_VMEC_JAX_INITIALIZATION_POLICY = "vmec_jax_default_input_boundary"
_VMEC2000_INITIALIZATION_POLICY = "vmec2000_default_input_boundary"
_VMEC_JAX_AXIS_RAW_POLICY = "raw_input_axis_or_zero"
_VMEC_JAX_AXIS_INFERRED_POLICY = "boundary_inferred_missing_axis"


def _parse_shape_cases(text: str) -> list[str]:
    names = [item.strip() for item in str(text).split(",") if item.strip()]
    unknown = [name for name in names if name not in _SHAPE_CASE_PRESETS]
    if unknown:
        choices = ", ".join(sorted(_SHAPE_CASE_PRESETS))
        raise ValueError(f"unknown shape case(s) {unknown}; choices are {choices}")
    return names


def _resolution_preset(name: str) -> dict[str, object]:
    key = str(name).strip().lower()
    try:
        return dict(_RESOLUTION_PRESETS[key])
    except KeyError as exc:
        choices = ", ".join(sorted(_RESOLUTION_PRESETS))
        raise ValueError(f"unknown resolution preset {name!r}; choices are {choices}") from exc


def _vmec_jax_axis_initialization_policy(solver_mode: str) -> str:
    """Return the VMEC/JAX axis branch used by this runner's fixed-boundary call."""
    mode = str(solver_mode).strip().lower()
    infer_axis = mode != "parity"
    enable_env = os.getenv("VMEC_JAX_ENABLE_AXIS_INFER", "").strip().lower()
    disable_env = os.getenv("VMEC_JAX_DISABLE_AXIS_INFER", "").strip().lower()
    if enable_env in ("1", "true", "yes", "on"):
        infer_axis = True
    if disable_env in ("1", "true", "yes", "on"):
        infer_axis = False
    return _VMEC_JAX_AXIS_INFERRED_POLICY if infer_axis else _VMEC_JAX_AXIS_RAW_POLICY


def _base_sample_kwargs(args: argparse.Namespace) -> dict[str, float | int]:
    return {
        "major_radius": float(args.major_radius),
        "minor_radius": float(args.minor_radius),
        "axis_oval": float(args.axis_oval),
        "side_minor_modulation": float(args.side_minor_modulation),
        "side_elongation": float(args.side_elongation),
        "side_power": float(args.side_power),
        "corner_amplitude": float(args.corner_amplitude),
        "corner_helicity": int(args.corner_helicity),
        "corner_ellipticity": float(args.corner_ellipticity),
        "corner_rotation": float(args.corner_rotation),
        "corner_power": float(args.corner_power),
    }


def _shape_case_kwargs(args: argparse.Namespace) -> list[tuple[str, dict[str, float | int]]]:
    base = _base_sample_kwargs(args)
    names = _parse_shape_cases(args.shape_cases)
    if not names:
        return [("custom", base)]
    cases = []
    for name in names:
        kwargs = dict(base)
        kwargs.update(_SHAPE_CASE_PRESETS[name])
        cases.append((name, kwargs))
    return cases


def _import_matplotlib():
    try:
        mpl_cache = Path(tempfile.gettempdir()) / "vmec_jax_mplconfig"
        mpl_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise SystemExit("This example requires matplotlib for plots.") from exc
    return plt


_CSV_COLUMNS = (
    "case",
    "shape_case",
    "resolution_preset",
    "target_resolution_ladder",
    "target_resolution_promotion_claim",
    "ns",
    "mpol",
    "ntor",
    "nstep",
    "rbc_count",
    "zbs_count",
    "max_boundary_fit_error",
    "max_orientation_fit_error",
    "orientation_fit_valid_fraction",
    "major_radius",
    "minor_radius",
    "axis_oval",
    "side_minor_modulation",
    "side_elongation",
    "side_power",
    "corner_amplitude",
    "corner_helicity",
    "corner_ellipticity",
    "corner_rotation",
    "corner_power",
    "cross_section_orientation_span",
    "side_orientation_span",
    "corner_orientation_span",
    "orientation_valid_fraction",
    "valid_cross_section_orientation_span",
    "valid_side_orientation_span",
    "valid_corner_orientation_span",
    "side_corner_weight_overlap_max",
    "fitted_cross_section_orientation_span",
    "fitted_side_orientation_span",
    "fitted_corner_orientation_span",
    "fitted_orientation_valid_fraction",
    "fitted_valid_cross_section_orientation_span",
    "fitted_valid_side_orientation_span",
    "fitted_valid_corner_orientation_span",
    "fitted_side_corner_weight_overlap_max",
    "cross_section_anisotropy_min",
    "cross_section_anisotropy_max",
    "fitted_cross_section_anisotropy_min",
    "fitted_cross_section_anisotropy_max",
    "initialization_policy",
    "vmec_jax_axis_initialization_policy",
    "ran_solve",
    "solver_mode",
    "use_scan",
    "cli_finish",
    "cli_fixed_boundary_mode",
    "cli_fixed_boundary_initial_policy",
    "cli_fixed_boundary_finish_attempts",
    "cli_fixed_boundary_finish_budgets",
    "cli_fixed_boundary_finish_fsq",
    "cli_fixed_boundary_finish_converged",
    "cli_fixed_boundary_finish_modes",
    "cli_fixed_boundary_finish_best_fsq",
    "cli_fixed_boundary_finish_budget_cap",
    "cli_fixed_boundary_finish_budget_exhausted",
    "cli_fixed_boundary_full_parity_fallback",
    "cli_fixed_boundary_partial_parity_fallback",
    "cli_fixed_boundary_staged_followup_used",
    "full_solver_diagnostics",
    "diagnostic_light_history",
    "diagnostic_resume_state_mode",
    "diagnostic_scan_path",
    "diagnostic_scan_minimal",
    "diagnostic_scan_light",
    "diagnostic_scan_use_precomputed",
    "diagnostic_scan_use_lax_tridi",
    "diagnostic_stage_modes",
    "diagnostic_stage_niter",
    "diagnostic_stage_offsets",
    "diagnostic_step_history_size",
    "diagnostic_time_step_history_size",
    "diagnostic_step_status_counts",
    "diagnostic_restart_reason_counts",
    "diagnostic_bcovar_updates",
    "diagnostic_initial_time_step",
    "diagnostic_final_time_step",
    "diagnostic_min_time_step",
    "diagnostic_max_time_step",
    "diagnostic_initial_bcovar_update",
    "diagnostic_final_dt_eff",
    "diagnostic_max_update_rms",
    "diagnostic_final_update_rms",
    "diagnostic_initial_axis_reset_attempted",
    "diagnostic_initial_axis_reset_reset",
    "diagnostic_initial_axis_reset_bad_jacobian",
    "diagnostic_initial_axis_reset_force_reset",
    "diagnostic_initial_axis_reset_fsq",
    "diagnostic_initial_axis_reset_ptau_min",
    "diagnostic_initial_axis_reset_ptau_max",
    "diagnostic_initial_axis_reset_state_tau_min",
    "diagnostic_initial_axis_reset_state_tau_max",
    "diagnostic_initial_axis_reset_error",
    "requested_ftol",
    "fsq_total_target",
    "seconds",
    "n_iter",
    "direct_initial_residual_requested",
    "direct_initial_residual_source",
    "direct_initial_axis_initialization_policy",
    "direct_initial_fsq",
    "direct_initial_fsqr",
    "direct_initial_fsqz",
    "direct_initial_fsql",
    "direct_initial_max_component",
    "direct_initial_max_component_name",
    "direct_initial_max_component_over_ftol",
    "direct_initial_fsq_ratio_vmec2000",
    "direct_initial_fsqr_ratio_vmec2000",
    "direct_initial_fsqz_ratio_vmec2000",
    "direct_initial_fsql_ratio_vmec2000",
    "initial_fsq_ratio_direct_initial",
    "vmec2000_initial_fsq_ratio_direct_initial",
    "direct_initial_error",
    "initial_residual_source",
    "initial_fsq",
    "best_fsq",
    "best_iter",
    "fsq_reduction",
    "final_fsq",
    "initial_fsqr",
    "initial_fsqz",
    "initial_fsql",
    "initial_max_component",
    "initial_max_component_name",
    "initial_max_component_over_ftol",
    "final_fsqr",
    "final_fsqz",
    "final_fsql",
    "final_max_component",
    "final_max_component_name",
    "final_max_component_over_ftol",
    "best_fsqr",
    "best_fsqz",
    "best_fsql",
    "best_max_component",
    "best_max_component_name",
    "best_max_component_over_ftol",
    "strict_component_pass",
    "strict_component_bottleneck",
    "strict_component_margin",
    "converged",
    "converged_strict",
    "converged_by_total_fsq",
    "aspect",
    "mean_iota",
    "magnetic_well",
    "ran_vmec2000",
    "vmec2000_initialization_policy",
    "vmec2000_returncode",
    "vmec2000_runtime_s",
    "vmec2000_n_rows",
    "vmec2000_initial_residual_source",
    "vmec2000_initial_fsq",
    "vmec2000_best_fsq",
    "vmec2000_best_iter",
    "vmec2000_fsq_reduction",
    "vmec2000_final_fsq",
    "vmec2000_initial_fsqr",
    "vmec2000_initial_fsqz",
    "vmec2000_initial_fsql",
    "vmec2000_initial_max_component",
    "vmec2000_initial_max_component_name",
    "vmec2000_initial_max_component_over_ftol",
    "vmec2000_final_fsqr",
    "vmec2000_final_fsqz",
    "vmec2000_final_fsql",
    "vmec2000_final_max_component",
    "vmec2000_final_max_component_name",
    "vmec2000_final_max_component_over_ftol",
    "vmec2000_aspect",
    "vmec2000_mean_iota",
    "initial_fsq_ratio_vmec2000",
    "initial_fsqr_ratio_vmec2000",
    "initial_fsqz_ratio_vmec2000",
    "initial_fsql_ratio_vmec2000",
    "input",
    "wout",
    "vmec2000_wout",
    "vmec2000_threed1",
    "vmec2000_error",
    "aggregate_source_json",
)


def _csv_cell(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def _write_rows_csv(rows: list[dict[str, object]], *, outdir: Path) -> str:
    path = outdir / "toroidal_stellarator_mirror_hybrid_convergence.csv"
    with path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(_CSV_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_cell(row.get(name)) for name in _CSV_COLUMNS})
    return str(path)


def _summarize_fsq_history(
    values: np.ndarray, *, iterations: np.ndarray | None = None
) -> dict[str, float | int | None]:
    history = np.asarray(values, dtype=float).reshape(-1)
    if history.size == 0:
        return {
            "initial_fsq": None,
            "best_fsq": None,
            "best_iter": None,
            "fsq_reduction": None,
            "final_fsq": None,
        }
    if iterations is None:
        iter_values = np.arange(history.size, dtype=int)
    else:
        iter_values = np.asarray(iterations, dtype=int).reshape(-1)
        if iter_values.size != history.size:
            iter_values = np.arange(history.size, dtype=int)
    out: dict[str, float | int | None] = {
        "initial_fsq": float(history[0]),
        "best_fsq": None,
        "best_iter": None,
        "fsq_reduction": None,
        "final_fsq": float(history[-1]),
    }
    finite = np.isfinite(history)
    if np.any(finite):
        finite_values = np.where(finite, history, np.inf)
        best_idx = int(np.argmin(finite_values))
        best_fsq = float(finite_values[best_idx])
        out["best_fsq"] = best_fsq
        out["best_iter"] = int(iter_values[best_idx])
        out["fsq_reduction"] = float(history[0]) / best_fsq if best_fsq > 0.0 else None
    return out


def _safe_ratio(numerator: object, denominator: object) -> float | None:
    if numerator is None or denominator is None:
        return None
    num = float(numerator)
    den = float(denominator)
    if not np.isfinite(num) or not np.isfinite(den) or den == 0.0:
        return None
    return num / den


def _orientation_fit_diagnostics(
    reference,
    fitted,
    *,
    anisotropy_rtol: float = 1.0e-8,
    anisotropy_atol: float = 1.0e-14,
) -> dict[str, float | None]:
    """Compare fitted principal-axis angles where the axis is well-defined."""
    reference_angle = toroidal_hybrid_cross_section_orientation(reference)
    fitted_angle = toroidal_hybrid_cross_section_orientation(fitted)
    if reference_angle.shape != fitted_angle.shape:
        raise ValueError("reference and fitted orientation arrays must have the same shape")
    reference_anisotropy = toroidal_hybrid_cross_section_anisotropy(reference)
    fitted_anisotropy = toroidal_hybrid_cross_section_anisotropy(fitted)
    anisotropy_scale = max(
        float(np.max(reference_anisotropy)) if reference_anisotropy.size else 0.0,
        float(np.max(fitted_anisotropy)) if fitted_anisotropy.size else 0.0,
    )
    threshold = float(anisotropy_atol) + float(anisotropy_rtol) * anisotropy_scale
    valid = (reference_anisotropy > threshold) & (fitted_anisotropy > threshold)
    valid_fraction = float(np.mean(valid)) if valid.size else 0.0
    if not np.any(valid):
        return {
            "max_orientation_fit_error": None,
            "orientation_fit_valid_fraction": valid_fraction,
        }
    wrapped_delta = 0.5 * np.angle(np.exp(2.0j * (fitted_angle - reference_angle)))
    return {
        "max_orientation_fit_error": float(np.max(np.abs(wrapped_delta[valid]))),
        "orientation_fit_valid_fraction": valid_fraction,
    }


def _attach_initial_residual_comparison(row: dict[str, object]) -> None:
    """Attach VMEC/JAX-to-VMEC2000 first-row residual ratios when available."""
    row["initial_fsq_ratio_vmec2000"] = _safe_ratio(row.get("initial_fsq"), row.get("vmec2000_initial_fsq"))
    row["direct_initial_fsq_ratio_vmec2000"] = _safe_ratio(
        row.get("direct_initial_fsq"),
        row.get("vmec2000_initial_fsq"),
    )
    row["initial_fsq_ratio_direct_initial"] = _safe_ratio(
        row.get("initial_fsq"),
        row.get("direct_initial_fsq"),
    )
    row["vmec2000_initial_fsq_ratio_direct_initial"] = _safe_ratio(
        row.get("vmec2000_initial_fsq"),
        row.get("direct_initial_fsq"),
    )
    for name in ("fsqr", "fsqz", "fsql"):
        row[f"initial_{name}_ratio_vmec2000"] = _safe_ratio(
            row.get(f"initial_{name}"),
            row.get(f"vmec2000_initial_{name}"),
        )
        row[f"direct_initial_{name}_ratio_vmec2000"] = _safe_ratio(
            row.get(f"direct_initial_{name}"),
            row.get(f"vmec2000_initial_{name}"),
        )


def _row_history_iterations(row: dict[str, object], history_size: int) -> np.ndarray:
    """Return stored iteration labels for a row, or a one-based fallback."""
    labels = np.asarray(row.get("iter_history", []), dtype=int).reshape(-1)
    if labels.size != int(history_size):
        return np.arange(1, int(history_size) + 1, dtype=int)
    return labels


def _diag_float_list(diag: dict[str, object], key: str) -> list[float]:
    values = np.asarray(diag.get(key, []), dtype=float).reshape(-1)
    return [float(value) for value in values]


def _diag_int_list(diag: dict[str, object], key: str) -> list[int]:
    values = np.asarray(diag.get(key, []), dtype=int).reshape(-1)
    return [int(value) for value in values]


def _diag_str_list(diag: dict[str, object], key: str) -> list[str]:
    values = np.asarray(diag.get(key, []), dtype=object).reshape(-1)
    return [str(value) for value in values]


def _diag_bool_list(diag: dict[str, object], key: str) -> list[bool]:
    values = np.asarray(diag.get(key, []), dtype=bool).reshape(-1)
    return [bool(value) for value in values]


def _diag_optional_bool(diag: dict[str, object], key: str) -> bool | None:
    return None if key not in diag or diag.get(key) is None else bool(diag.get(key))


def _diag_optional_int(diag: dict[str, object], key: str) -> int | None:
    if key not in diag or diag.get(key) is None:
        return None
    return int(diag[key])


def _diag_optional_float(diag: dict[str, object], key: str) -> float | None:
    if key not in diag or diag.get(key) is None:
        return None
    value = float(diag[key])
    return value if np.isfinite(value) else None


def _counts_json(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


def _cli_finish_diagnostic_fields(diag: dict[str, object]) -> dict[str, object]:
    budgets = _diag_int_list(diag, "cli_fixed_boundary_finish_budgets")
    fsq = _diag_float_list(diag, "cli_fixed_boundary_finish_fsq")
    converged = _diag_bool_list(diag, "cli_fixed_boundary_finish_converged")
    modes = _diag_str_list(diag, "cli_fixed_boundary_finish_modes")
    finite_fsq = [value for value in fsq if np.isfinite(float(value))]
    return {
        "cli_fixed_boundary_mode": _diag_optional_bool(diag, "cli_fixed_boundary_mode"),
        "cli_fixed_boundary_initial_policy": None
        if diag.get("cli_fixed_boundary_initial_policy") is None
        else str(diag.get("cli_fixed_boundary_initial_policy")),
        "cli_fixed_boundary_finish_attempts": max(len(budgets), len(fsq), len(converged), len(modes)),
        "cli_fixed_boundary_finish_budgets": budgets,
        "cli_fixed_boundary_finish_fsq": fsq,
        "cli_fixed_boundary_finish_converged": converged,
        "cli_fixed_boundary_finish_modes": modes,
        "cli_fixed_boundary_finish_best_fsq": None if not finite_fsq else float(min(finite_fsq)),
        "cli_fixed_boundary_finish_budget_cap": _diag_optional_int(
            diag,
            "cli_fixed_boundary_finish_budget_cap",
        ),
        "cli_fixed_boundary_finish_budget_exhausted": _diag_optional_bool(
            diag,
            "cli_fixed_boundary_finish_budget_exhausted",
        ),
        "cli_fixed_boundary_full_parity_fallback": _diag_optional_bool(
            diag,
            "cli_fixed_boundary_full_parity_fallback",
        ),
        "cli_fixed_boundary_partial_parity_fallback": _diag_optional_bool(
            diag,
            "cli_fixed_boundary_partial_parity_fallback",
        ),
        "cli_fixed_boundary_staged_followup_used": _diag_optional_bool(
            diag,
            "cli_fixed_boundary_staged_followup_used",
        ),
    }


def _solver_diagnostic_fields(diag: dict[str, object], *, fallback_size: int) -> dict[str, object]:
    """Return compact JSON-safe solver histories for trajectory audits."""
    step_status = _diag_str_list(diag, "step_status_history")
    restart_reason = _diag_str_list(diag, "restart_reason_history")
    pre_restart_reason = _diag_str_list(diag, "pre_restart_reason_history")
    dt_eff = _diag_float_list(diag, "dt_eff_history")
    update_rms = _diag_float_list(diag, "update_rms_history")
    w_curr = _diag_float_list(diag, "w_curr_history")
    w_try = _diag_float_list(diag, "w_try_history")
    w_try_ratio = _diag_float_list(diag, "w_try_ratio_history")
    time_step = _diag_float_list(diag, "time_step_history")
    terminal_size = max(
        len(step_status),
        len(restart_reason),
        len(pre_restart_reason),
        len(time_step),
        len(dt_eff),
        len(update_rms),
        len(w_curr),
        len(w_try),
        len(w_try_ratio),
        0,
    )
    iter2 = _diag_int_list(diag, "iter2_history")
    if terminal_size and len(iter2) != terminal_size:
        iter2 = [int(value) for value in range(1, terminal_size + 1)]
    elif not terminal_size and int(fallback_size) > 0:
        iter2 = []
    bcovar = _diag_int_list(diag, "bcovar_update_history")
    stage_modes = _diag_str_list(diag, "multigrid_stage_modes")
    stage_niter = _diag_int_list(diag, "multigrid_niter_stages")
    stage_offsets = _diag_int_list(diag, "multigrid_stage_offsets")
    return {
        "diagnostic_light_history": None if "light_history" not in diag else bool(diag.get("light_history")),
        "diagnostic_resume_state_mode": None
        if diag.get("resume_state_mode") is None
        else str(diag.get("resume_state_mode")),
        "diagnostic_scan_path": None if diag.get("scan_path") is None else str(diag.get("scan_path")),
        "diagnostic_scan_minimal": _diag_optional_bool(diag, "scan_minimal"),
        "diagnostic_scan_light": _diag_optional_bool(diag, "light_history"),
        "diagnostic_scan_use_precomputed": _diag_optional_bool(diag, "scan_use_precomputed"),
        "diagnostic_scan_use_lax_tridi": _diag_optional_bool(diag, "scan_use_lax_tridi"),
        "diagnostic_stage_modes": stage_modes,
        "diagnostic_stage_niter": stage_niter,
        "diagnostic_stage_offsets": stage_offsets,
        "diagnostic_step_history_size": int(terminal_size),
        "diagnostic_step_iter_history": iter2,
        "diagnostic_step_status_history": step_status,
        "diagnostic_restart_reason_history": restart_reason,
        "diagnostic_pre_restart_reason_history": pre_restart_reason,
        "diagnostic_time_step_history": time_step,
        "diagnostic_time_step_history_size": int(len(time_step)),
        "diagnostic_dt_eff_history": dt_eff,
        "diagnostic_update_rms_history": update_rms,
        "diagnostic_w_curr_history": w_curr,
        "diagnostic_w_try_history": w_try,
        "diagnostic_w_try_ratio_history": w_try_ratio,
        "diagnostic_bcovar_update_history": bcovar,
        "diagnostic_step_status_counts": _counts_json(step_status),
        "diagnostic_restart_reason_counts": _counts_json(restart_reason),
        "diagnostic_bcovar_updates": int(sum(1 for value in bcovar if int(value) != 0)),
        "diagnostic_initial_time_step": None if not time_step else float(time_step[0]),
        "diagnostic_final_time_step": None if not time_step else float(time_step[-1]),
        "diagnostic_min_time_step": None if not time_step else float(np.nanmin(time_step)),
        "diagnostic_max_time_step": None if not time_step else float(np.nanmax(time_step)),
        "diagnostic_initial_bcovar_update": None if not bcovar else bool(int(bcovar[0])),
        "diagnostic_final_dt_eff": None if not dt_eff else float(dt_eff[-1]),
        "diagnostic_max_update_rms": None if not update_rms else float(np.nanmax(update_rms)),
        "diagnostic_final_update_rms": None if not update_rms else float(update_rms[-1]),
        "diagnostic_initial_axis_reset_attempted": _diag_optional_bool(diag, "initial_axis_reset_attempted"),
        "diagnostic_initial_axis_reset_reset": _diag_optional_bool(diag, "initial_axis_reset_reset"),
        "diagnostic_initial_axis_reset_bad_jacobian": _diag_optional_bool(diag, "initial_axis_reset_bad_jacobian"),
        "diagnostic_initial_axis_reset_force_reset": _diag_optional_bool(diag, "initial_axis_reset_force_reset"),
        "diagnostic_initial_axis_reset_fsq": _diag_optional_float(diag, "initial_axis_reset_fsq"),
        "diagnostic_initial_axis_reset_ptau_min": _diag_optional_float(diag, "initial_axis_reset_ptau_min"),
        "diagnostic_initial_axis_reset_ptau_max": _diag_optional_float(diag, "initial_axis_reset_ptau_max"),
        "diagnostic_initial_axis_reset_state_tau_min": _diag_optional_float(
            diag,
            "initial_axis_reset_state_tau_min",
        ),
        "diagnostic_initial_axis_reset_state_tau_max": _diag_optional_float(
            diag,
            "initial_axis_reset_state_tau_max",
        ),
        "diagnostic_initial_axis_reset_error": None
        if diag.get("initial_axis_reset_error") is None
        else str(diag.get("initial_axis_reset_error")),
    }


def _compute_direct_initial_residual(
    input_path: Path,
    *,
    solver_mode: str,
    use_scan: bool | None,
) -> dict[str, object]:
    """Evaluate force residual scalars on the VMEC/JAX initial state."""
    mode = str(solver_mode).strip().lower()
    run = vj.run_fixed_boundary(
        input_path,
        solver="vmec2000_iter",
        solver_mode=str(solver_mode),
        use_scan=use_scan,
        max_iter=1,
        use_initial_guess=True,
        cli_fixed_boundary_mode=True,
        verbose=False,
    )
    wout = vj.wout_from_fixed_boundary_run(
        run,
        include_fsq=True,
        fast_bcovar=False if mode == "parity" else True,
    )
    fsqr = float(wout.fsqr)
    fsqz = float(wout.fsqz)
    fsql = float(wout.fsql)
    return {
        "direct_initial_residual_source": "vmec_jax_initial_guess_residual_scalars",
        "direct_initial_axis_initialization_policy": _vmec_jax_axis_initialization_policy(str(solver_mode)),
        "direct_initial_fsq": fsqr + fsqz + fsql,
        "direct_initial_fsqr": fsqr,
        "direct_initial_fsqz": fsqz,
        "direct_initial_fsql": fsql,
        "direct_initial_error": None,
    }


def _write_summary_plot(rows: list[dict[str, object]], *, outdir: Path) -> str:
    plt = _import_matplotlib()
    outdir.mkdir(parents=True, exist_ok=True)
    labels = [f"ns={row['ns']}, {row['mpol']}:{row['ntor']}" for row in rows]
    solved_rows = any(row.get("best_fsq") is not None or row.get("final_fsq") is not None for row in rows)
    if solved_rows:
        values = [
            float(row["best_fsq"] if row.get("best_fsq") is not None else row["final_fsq"])
            if row.get("final_fsq") is not None
            else float(row["max_boundary_fit_error"])
            for row in rows
        ]
        ylabel = "best fsq"
    else:
        values = [float(row["max_boundary_fit_error"]) for row in rows]
        ylabel = "max boundary fit error"
    fig, ax = plt.subplots(1, 1, figsize=(max(7.0, 0.6 * len(rows)), 4.2), constrained_layout=True)
    ax.semilogy(np.arange(len(rows)), np.maximum(values, 1.0e-300), "o-", lw=1.5)
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title("Toroidal hybrid convergence grid")
    ax.grid(True, which="both", alpha=0.25)
    path = outdir / "toroidal_hybrid_convergence.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def _write_orientation_plot(rows: list[dict[str, object]], *, outdir: Path) -> str | None:
    if not rows:
        return None
    plt = _import_matplotlib()
    outdir.mkdir(parents=True, exist_ok=True)
    labels = [str(row["case"]) for row in rows]
    x = np.arange(len(rows), dtype=float)
    fig, ax0 = plt.subplots(1, 1, figsize=(max(7.0, 0.62 * len(rows)), 4.4), constrained_layout=True)
    error = np.asarray(
        [
            np.nan if row.get("max_orientation_fit_error") is None else float(row["max_orientation_fit_error"])
            for row in rows
        ],
        dtype=float,
    )
    valid_fraction = np.asarray(
        [
            np.nan
            if row.get("orientation_fit_valid_fraction") is None
            else float(row["orientation_fit_valid_fraction"])
            for row in rows
        ],
        dtype=float,
    )
    if np.any(np.isfinite(error)):
        ax0.semilogy(x, np.maximum(error, 1.0e-300), "o-", lw=1.4, ms=4, label="valid max fit error")
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, rotation=30, ha="right")
    ax0.set_ylabel("angle error [rad]")
    ax0.set_title("Toroidal hybrid orientation preservation")
    ax0.grid(True, which="both", alpha=0.25)
    ax1 = ax0.twinx()
    if np.any(np.isfinite(valid_fraction)):
        ax1.plot(x, valid_fraction, "s--", color="tab:orange", lw=1.3, ms=4, label="valid fraction")
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_ylabel("valid-axis fraction")
    lines0, labels0 = ax0.get_legend_handles_labels()
    lines1, labels1 = ax1.get_legend_handles_labels()
    if lines0 or lines1:
        ax0.legend(lines0 + lines1, labels0 + labels1, loc="best", fontsize=8)
    path = outdir / "toroidal_hybrid_orientation_preservation.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def _write_fsq_history_plot(rows: list[dict[str, object]], *, outdir: Path) -> str | None:
    history_rows = [row for row in rows if row.get("fsq_history") or row.get("vmec2000_fsq_history")]
    if not history_rows:
        return None
    plt = _import_matplotlib()
    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.2), constrained_layout=True)
    for row in history_rows:
        history = np.asarray(row.get("fsq_history", []), dtype=float).reshape(-1)
        vmec2000_history = np.asarray(row.get("vmec2000_fsq_history", []), dtype=float).reshape(-1)
        finite_history = np.concatenate(
            [
                history[np.isfinite(history) & (history > 0.0)],
                vmec2000_history[np.isfinite(vmec2000_history) & (vmec2000_history > 0.0)],
            ]
        )
        visible_scale = float(np.max(finite_history)) if finite_history.size else 0.0
        direct_initial = row.get("direct_initial_fsq")
        if direct_initial is not None:
            direct_value = float(direct_initial)
            plot_value = direct_value
            label = f"{row['case']} VMEC/JAX direct initial"
            if visible_scale > 0.0 and direct_value > 1.0e4 * visible_scale:
                plot_value = 10.0 * visible_scale
                label = f"{label} (off-scale {direct_value:.2e})"
            ax.semilogy(
                [0],
                [max(plot_value, 1.0e-300)],
                "*",
                ms=8,
                label=label,
            )
        if history.size:
            iters = _row_history_iterations(row, int(history.size))
            ax.semilogy(
                iters,
                np.maximum(history, 1.0e-300),
                "o-",
                lw=1.3,
                ms=3,
                label=f"{row['case']} VMEC/JAX",
            )
        if vmec2000_history.size:
            vmec2000_iters = np.asarray(row.get("vmec2000_iter_history", []), dtype=int).reshape(-1)
            if vmec2000_iters.size != vmec2000_history.size:
                vmec2000_iters = np.arange(vmec2000_history.size, dtype=int)
            ax.semilogy(
                vmec2000_iters,
                np.maximum(vmec2000_history, 1.0e-300),
                "s--",
                lw=1.2,
                ms=3,
                label=f"{row['case']} VMEC2000",
            )
    ax.set_xlabel("iteration (0 is VMEC/JAX direct initial)")
    ax.set_ylabel("fsq")
    ax.set_title("Toroidal hybrid residual history")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    path = outdir / "toroidal_hybrid_fsq_history.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def _write_step_diagnostic_plot(rows: list[dict[str, object]], *, outdir: Path) -> str | None:
    step_rows = [
        row
        for row in rows
        if row.get("diagnostic_dt_eff_history")
        or row.get("diagnostic_time_step_history")
        or row.get("diagnostic_update_rms_history")
        or row.get("diagnostic_w_try_ratio_history")
    ]
    if not step_rows:
        return None
    plt = _import_matplotlib()
    outdir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.4), sharex=True, constrained_layout=True)
    for row in step_rows:
        label = str(row["case"])
        iters = np.asarray(row.get("diagnostic_step_iter_history", []), dtype=int).reshape(-1)
        for ax, key, ylabel in (
            (
                axes[0],
                "diagnostic_dt_eff_history"
                if row.get("diagnostic_dt_eff_history")
                else "diagnostic_time_step_history",
                "dt effective / scan dt",
            ),
            (axes[1], "diagnostic_update_rms_history", "update RMS"),
            (axes[2], "diagnostic_w_try_ratio_history", "trial/current fsq"),
        ):
            values = np.asarray(row.get(key, []), dtype=float).reshape(-1)
            if values.size == 0:
                continue
            x = iters if iters.size == values.size else np.arange(1, values.size + 1, dtype=int)
            ax.semilogy(x, np.maximum(values, 1.0e-300), ".-", lw=1.1, ms=3, label=label)
            ax.set_ylabel(ylabel)
            ax.grid(True, which="both", alpha=0.25)
    axes[-1].set_xlabel("iteration")
    for ax in axes:
        if ax.lines:
            ax.legend(loc="best", fontsize=8)
    axes[0].set_title("Toroidal hybrid solver step diagnostics")
    path = outdir / "toroidal_hybrid_step_diagnostics.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def _write_profile_plots(rows: list[dict[str, object]], *, outdir: Path) -> str | None:
    wout_rows = [row for row in rows if row.get("wout")]
    if not wout_rows:
        return None
    plt = _import_matplotlib()
    outdir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), constrained_layout=True)
    plotted = False
    for row in wout_rows:
        try:
            wout = read_wout(str(row["wout"]))
        except Exception:
            continue
        ns = int(getattr(wout, "ns", 0))
        if ns <= 0:
            continue
        s = np.linspace(0.0, 1.0, ns)
        label = str(row["case"])
        iotas = np.asarray(getattr(wout, "iotas", np.zeros((0,))), dtype=float).reshape(-1)
        if iotas.size == ns:
            axes[0].plot(s, iotas, ".-", lw=1.2, ms=3, label=label)
            plotted = True
        dwell = np.asarray(getattr(wout, "Dwell", np.zeros((0,))), dtype=float).reshape(-1)
        if dwell.size == ns:
            axes[1].plot(s, dwell, ".-", lw=1.2, ms=3, label=label)
            plotted = True
    if not plotted:
        plt.close(fig)
        return None
    axes[0].set_xlabel("s")
    axes[0].set_ylabel("iota")
    axes[0].set_title("iota profile")
    axes[1].set_xlabel("s")
    axes[1].set_ylabel("DWell")
    axes[1].set_title("Mercier well term")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
    path = outdir / "toroidal_hybrid_profiles.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def _write_parity_component_plot(rows: list[dict[str, object]], *, outdir: Path) -> str | None:
    parity_rows = [
        row
        for row in rows
        if row.get("final_fsqr") is not None
        and row.get("vmec2000_final_fsqr") is not None
        and row.get("vmec2000_final_fsqz") is not None
        and row.get("vmec2000_final_fsql") is not None
    ]
    if not parity_rows:
        return None
    plt = _import_matplotlib()
    outdir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        1,
        len(parity_rows),
        figsize=(max(5.0, 4.4 * len(parity_rows)), 4.0),
        squeeze=False,
        constrained_layout=True,
    )
    components = ("fsqr", "fsqz", "fsql")
    x = np.arange(len(components), dtype=float)
    width = 0.36
    for ax, row in zip(axes.ravel(), parity_rows, strict=False):
        jax_values = np.asarray([float(row[f"final_{name}"]) for name in components], dtype=float)
        vmec_values = np.asarray([float(row[f"vmec2000_final_{name}"]) for name in components], dtype=float)
        ax.bar(x - width / 2.0, np.maximum(jax_values, 1.0e-300), width=width, label="VMEC/JAX")
        ax.bar(x + width / 2.0, np.maximum(vmec_values, 1.0e-300), width=width, label="VMEC2000")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(components)
        ax.set_ylabel("final residual component")
        ax.set_title(str(row["case"]))
        ax.grid(True, axis="y", which="both", alpha=0.25)
        ax.legend(loc="best", fontsize=8)
    path = outdir / "toroidal_hybrid_parity_components.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def _write_convergence_figures(rows: list[dict[str, object]], *, outdir: Path) -> dict[str, str]:
    figures: dict[str, str] = {}
    summary_rows = [
        row
        for row in rows
        if row.get("best_fsq") is not None
        or row.get("final_fsq") is not None
        or row.get("max_boundary_fit_error") is not None
    ]
    if summary_rows:
        figures["convergence"] = _write_summary_plot(summary_rows, outdir=outdir)
    orientation_plot = _write_orientation_plot(rows, outdir=outdir)
    if orientation_plot is not None:
        figures["orientation"] = orientation_plot
    fsq_history_plot = _write_fsq_history_plot(rows, outdir=outdir)
    if fsq_history_plot is not None:
        figures["fsq_history"] = fsq_history_plot
    step_plot = _write_step_diagnostic_plot(rows, outdir=outdir)
    if step_plot is not None:
        figures["step_diagnostics"] = step_plot
    profile_plot = _write_profile_plots(rows, outdir=outdir)
    if profile_plot is not None:
        figures["profiles"] = profile_plot
    parity_plot = _write_parity_component_plot(rows, outdir=outdir)
    if parity_plot is not None:
        figures["parity_components"] = parity_plot
    return figures


def _finite_row_values(rows: list[dict[str, object]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            values.append(number)
    return values


def _range_or_none(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "max": None}
    return {"min": float(min(values)), "max": float(max(values))}


_RESIDUAL_COMPONENTS = ("fsqr", "fsqz", "fsql")


def _row_float(row: dict[str, object], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _component_max(row: dict[str, object], prefix: str) -> tuple[str | None, float | None]:
    values: list[tuple[str, float]] = []
    for name in _RESIDUAL_COMPONENTS:
        value = _row_float(row, f"{prefix}_{name}")
        if value is not None:
            values.append((name, value))
    if not values:
        return None, None
    name, value = max(values, key=lambda item: item[1])
    return name, float(value)


def _attach_component_max_diagnostics(row: dict[str, object]) -> None:
    """Attach strict-component bottleneck metrics for any available residuals."""
    ftol = _row_float(row, "requested_ftol")
    for prefix in ("direct_initial", "initial", "best", "final", "vmec2000_initial", "vmec2000_final"):
        name, value = _component_max(row, prefix)
        row[f"{prefix}_max_component"] = value
        row[f"{prefix}_max_component_name"] = name
        row[f"{prefix}_max_component_over_ftol"] = (
            None if value is None or ftol is None or ftol <= 0.0 else float(value / ftol)
        )

    final_name = row.get("final_max_component_name")
    final_value = _row_float(row, "final_max_component")
    if final_value is None or ftol is None:
        row["strict_component_pass"] = None
        row["strict_component_bottleneck"] = final_name
        row["strict_component_margin"] = None
        return
    row["strict_component_pass"] = bool(final_value <= ftol)
    row["strict_component_bottleneck"] = None if final_value <= ftol else final_name
    row["strict_component_margin"] = float(ftol - final_value)


def _row_sort_key(row: dict[str, object]) -> tuple[str, int, int, int, str]:
    def _int_value(name: str) -> int:
        try:
            return int(row.get(name))
        except (TypeError, ValueError):
            return 10**9

    return (
        str(row.get("shape_case", "")),
        _int_value("ns"),
        _int_value("mpol"),
        _int_value("ntor"),
        str(row.get("case", "")),
    )


def _aggregate_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    direct_ratio = _range_or_none(_finite_row_values(rows, "direct_initial_fsq_ratio_vmec2000"))
    best_fsq = _range_or_none(_finite_row_values(rows, "best_fsq"))
    final_fsq = _range_or_none(_finite_row_values(rows, "final_fsq"))
    vmec2000_final_fsq = _range_or_none(_finite_row_values(rows, "vmec2000_final_fsq"))
    final_component = _range_or_none(_finite_row_values(rows, "final_max_component"))
    final_component_over_ftol = _range_or_none(_finite_row_values(rows, "final_max_component_over_ftol"))
    vmec2000_final_component_over_ftol = _range_or_none(
        _finite_row_values(rows, "vmec2000_final_max_component_over_ftol")
    )
    vmec2000_rows = [row for row in rows if bool(row.get("ran_vmec2000"))]
    strict_blockers = [
        str(row.get("strict_component_bottleneck"))
        for row in rows
        if row.get("strict_component_bottleneck") is not None
    ]
    return {
        "row_count": len(rows),
        "ran_solve_rows": sum(1 for row in rows if bool(row.get("ran_solve"))),
        "vmec2000_rows": len(vmec2000_rows),
        "vmec2000_returncode_zero_rows": sum(
            1 for row in vmec2000_rows if row.get("vmec2000_returncode") == 0
        ),
        "vmec_jax_converged_rows": sum(1 for row in rows if bool(row.get("converged"))),
        "vmec_jax_strict_converged_rows": sum(1 for row in rows if bool(row.get("converged_strict"))),
        "vmec_jax_total_fsq_converged_rows": sum(
            1 for row in rows if bool(row.get("converged_by_total_fsq"))
        ),
        "vmec_jax_strict_component_pass_rows": sum(
            1 for row in rows if row.get("strict_component_pass") is True
        ),
        "vmec_jax_strict_component_known_rows": sum(
            1 for row in rows if row.get("strict_component_pass") is not None
        ),
        "vmec_jax_strict_component_blocker_counts": _counts_json(strict_blockers),
        "vmec_jax_final_max_component_min": final_component["min"],
        "vmec_jax_final_max_component_max": final_component["max"],
        "vmec_jax_final_max_component_over_ftol_min": final_component_over_ftol["min"],
        "vmec_jax_final_max_component_over_ftol_max": final_component_over_ftol["max"],
        "vmec2000_final_max_component_over_ftol_min": vmec2000_final_component_over_ftol["min"],
        "vmec2000_final_max_component_over_ftol_max": vmec2000_final_component_over_ftol["max"],
        "direct_initial_fsq_ratio_vmec2000_min": direct_ratio["min"],
        "direct_initial_fsq_ratio_vmec2000_max": direct_ratio["max"],
        "best_fsq_min": best_fsq["min"],
        "best_fsq_max": best_fsq["max"],
        "final_fsq_min": final_fsq["min"],
        "final_fsq_max": final_fsq["max"],
        "vmec2000_final_fsq_min": vmec2000_final_fsq["min"],
        "vmec2000_final_fsq_max": vmec2000_final_fsq["max"],
    }


def _aggregate_convergence_jsons(
    paths: list[Path],
    *,
    outdir: Path,
    no_plots: bool,
) -> Path:
    if not paths:
        raise ValueError("at least one --aggregate-json path is required")
    outdir.mkdir(parents=True, exist_ok=True)
    rows_by_case: dict[str, dict[str, object]] = {}
    duplicate_cases: list[str] = []
    source_summaries: list[dict[str, object]] = []
    for source_path in paths:
        path = source_path.expanduser().resolve()
        with path.open() as file_obj:
            source = json.load(file_obj)
        if not isinstance(source, dict):
            raise ValueError(f"{path} must contain a JSON object")
        source_rows = source.get("rows")
        if not isinstance(source_rows, list):
            raise ValueError(f"{path} must contain a rows list")
        source_summaries.append(
            {
                "path": str(path),
                "row_count": len(source_rows),
                "resolution_preset": source.get("resolution_preset"),
                "target_resolution_ladder": source.get("target_resolution_ladder"),
                "case_filters": source.get("case_filters", []),
            }
        )
        for index, source_row in enumerate(source_rows):
            if not isinstance(source_row, dict):
                raise ValueError(f"{path} row {index} must be a JSON object")
            case = str(source_row.get("case", "")).strip()
            if not case:
                raise ValueError(f"{path} row {index} is missing a case name")
            if case in rows_by_case:
                duplicate_cases.append(case)
            row = dict(source_row)
            row["aggregate_source_json"] = str(path)
            _attach_component_max_diagnostics(row)
            rows_by_case[case] = row
    rows = sorted(rows_by_case.values(), key=_row_sort_key)
    if not rows:
        raise ValueError("aggregated convergence JSONs contained no rows")
    csv_path = _write_rows_csv(rows, outdir=outdir)
    summary = {
        "aggregate_schema": "toroidal_stellarator_mirror_hybrid_convergence_aggregate.v1",
        "source_jsons": [str(path.expanduser().resolve()) for path in paths],
        "source_summaries": source_summaries,
        "duplicate_cases_replaced": sorted(set(duplicate_cases)),
        "resolution_presets": sorted({str(row.get("resolution_preset", "")) for row in rows}),
        "target_resolution_ladder": all(bool(row.get("target_resolution_ladder")) for row in rows),
        "target_resolution_promotion_claim": all(
            bool(row.get("target_resolution_promotion_claim")) for row in rows
        ),
        "case_count": len(rows),
        "rows": rows,
        "aggregate_metrics": _aggregate_metrics(rows),
        "csv": csv_path,
        "figures": {},
    }
    if not no_plots:
        summary["figures"] = _write_convergence_figures(rows, outdir=outdir / "figures")
    summary_path = outdir / "toroidal_stellarator_mirror_hybrid_convergence_aggregate.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary_path


def _row_case_name(*, ns: int, mpol: int, ntor: int, shape_case: str = "custom") -> str:
    base = f"ns{int(ns):03d}_mpol{int(mpol):02d}_ntor{int(ntor):02d}"
    return base if shape_case == "custom" else f"{shape_case}_{base}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=str, default="results/toroidal_stellarator_mirror_hybrid_convergence")
    parser.add_argument("--ns-array", type=str, default="9,15")
    parser.add_argument("--mode-pairs", type=str, default="5:4")
    parser.add_argument(
        "--resolution-preset",
        choices=tuple(sorted(_RESOLUTION_PRESETS)),
        default="manual",
        help="Named ns/mode-pair ladder. Use manual to honor --ns-array and --mode-pairs.",
    )
    parser.add_argument(
        "--case-filter",
        type=str,
        default="",
        help="Comma-separated shell patterns for generated case names, for example '*ns015*' or 'sharp_*'.",
    )
    parser.add_argument("--nfp", type=int, default=2)
    parser.add_argument("--niter", type=int, default=80)
    parser.add_argument(
        "--nstep",
        type=int,
        default=25,
        help="VMEC print cadence written into NSTEP; use 1 for full VMEC2000 threed1 trajectories.",
    )
    parser.add_argument("--ftol", type=float, default=1.0e-9)
    parser.add_argument("--max-iter", type=int, default=3)
    parser.add_argument("--major-radius", type=float, default=1.15)
    parser.add_argument("--minor-radius", type=float, default=0.18)
    parser.add_argument("--axis-oval", type=float, default=0.10)
    parser.add_argument("--side-minor-modulation", type=float, default=0.10)
    parser.add_argument("--side-elongation", type=float, default=0.28)
    parser.add_argument("--side-power", type=float, default=1.0)
    parser.add_argument("--corner-amplitude", type=float, default=0.035)
    parser.add_argument("--corner-helicity", type=int, default=1)
    parser.add_argument("--corner-ellipticity", type=float, default=0.18)
    parser.add_argument("--corner-rotation", type=float, default=0.35)
    parser.add_argument("--corner-power", type=float, default=1.0)
    parser.add_argument(
        "--shape-cases",
        type=str,
        default="",
        help="Comma-separated preset shape cases to scan; choices: default,sharp. Empty uses the explicit CLI shape.",
    )
    parser.add_argument("--ntheta-fit", type=int, default=64)
    parser.add_argument("--nzeta-fit", type=int, default=64)
    parser.add_argument("--run-solve", action="store_true")
    parser.add_argument("--solver-mode", choices=("default", "parity", "accelerated"), default="accelerated")
    parser.add_argument("--use-scan", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--cli-finish",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the vmec_jax CLI finish/fallback policy; disable for raw VMEC-style trajectory parity.",
    )
    parser.add_argument(
        "--direct-initial-residual",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When solving, also evaluate VMEC/JAX residual scalars on the pre-iteration initial state.",
    )
    parser.add_argument(
        "--full-solver-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Store full per-step VMEC/JAX solver histories instead of the quiet light-history path.",
    )
    parser.add_argument("--run-vmec2000", action="store_true")
    parser.add_argument("--vmec2000-exec", type=str, default="")
    parser.add_argument("--vmec2000-timeout-s", type=float, default=120.0)
    parser.add_argument(
        "--aggregate-json",
        nargs="+",
        default=None,
        metavar="PATH",
        help="Aggregate one or more existing convergence JSON files instead of running new cases.",
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    aggregate_paths = _parse_path_args(args.aggregate_json)
    if args.aggregate_json is not None:
        summary_path = _aggregate_convergence_jsons(
            aggregate_paths,
            outdir=outdir,
            no_plots=bool(args.no_plots),
        )
        print(summary_path)
        return
    resolution_preset = _resolution_preset(args.resolution_preset)
    if args.resolution_preset == "manual":
        ns_values = _parse_ints(args.ns_array)
        mode_pairs = _parse_mode_pairs(args.mode_pairs)
    else:
        ns_values = list(resolution_preset["ns_array"])
        mode_pairs = list(resolution_preset["mode_pairs"])
    vmec2000_exec = Path(args.vmec2000_exec).expanduser() if str(args.vmec2000_exec).strip() else None
    shape_cases = _shape_case_kwargs(args)
    case_filters = _parse_case_filters(args.case_filter)
    rows: list[dict[str, object]] = []

    for shape_case, sample_kwargs in shape_cases:
        samples = sample_toroidal_stellarator_mirror_hybrid_boundary(
            ntheta=int(args.ntheta_fit),
            nzeta=int(args.nzeta_fit),
            **sample_kwargs,
        )
        reference_metrics = toroidal_stellarator_mirror_hybrid_metrics(samples)
        for ns in ns_values:
            for mpol, ntor in mode_pairs:
                case = _row_case_name(ns=ns, mpol=mpol, ntor=ntor, shape_case=shape_case)
                if not _case_matches_filters(case, case_filters):
                    continue
                case_dir = outdir / case
                case_dir.mkdir(parents=True, exist_ok=True)
                indata = toroidal_stellarator_mirror_hybrid_indata(
                    nfp=int(args.nfp),
                    mpol=int(mpol),
                    ntor=int(ntor),
                    ntheta_fit=int(args.ntheta_fit),
                    nzeta_fit=int(args.nzeta_fit),
                    ns_array=int(ns),
                    niter_array=int(args.niter),
                    ftol_array=float(args.ftol),
                    **sample_kwargs,
                )
                indata.scalars["NSTEP"] = int(args.nstep)
                input_path = case_dir / "input.toroidal_stellarator_mirror_hybrid"
                write_indata(input_path, indata)
                fitted = evaluate_toroidal_hybrid_indata_boundary(
                    indata,
                    ntheta=int(args.ntheta_fit),
                    nzeta=int(args.nzeta_fit),
                )
                max_fit_error = max(
                    float(np.max(np.abs(fitted.R - samples.R))),
                    float(np.max(np.abs(fitted.Z - samples.Z))),
                )
                fitted_metrics = toroidal_stellarator_mirror_hybrid_metrics(fitted)
                orientation_fit = _orientation_fit_diagnostics(samples, fitted)
                row: dict[str, object] = {
                    "case": case,
                    "shape_case": shape_case,
                    "resolution_preset": str(args.resolution_preset),
                    "target_resolution_ladder": bool(resolution_preset["target_resolution_ladder"]),
                    "target_resolution_promotion_claim": False,
                    "ns": int(ns),
                    "mpol": int(mpol),
                    "ntor": int(ntor),
                    "nstep": int(args.nstep),
                    "input": str(input_path),
                    "rbc_count": len(indata.indexed.get("RBC", {})),
                    "zbs_count": len(indata.indexed.get("ZBS", {})),
                    "max_boundary_fit_error": max_fit_error,
                    "max_orientation_fit_error": orientation_fit["max_orientation_fit_error"],
                    "orientation_fit_valid_fraction": orientation_fit["orientation_fit_valid_fraction"],
                    "major_radius": sample_kwargs["major_radius"],
                    "minor_radius": sample_kwargs["minor_radius"],
                    "axis_oval": sample_kwargs["axis_oval"],
                    "side_minor_modulation": sample_kwargs["side_minor_modulation"],
                    "side_elongation": sample_kwargs["side_elongation"],
                    "side_power": sample_kwargs["side_power"],
                    "corner_amplitude": sample_kwargs["corner_amplitude"],
                    "corner_helicity": sample_kwargs["corner_helicity"],
                    "corner_ellipticity": sample_kwargs["corner_ellipticity"],
                    "corner_rotation": sample_kwargs["corner_rotation"],
                    "corner_power": sample_kwargs["corner_power"],
                    "cross_section_orientation_span": reference_metrics["cross_section_orientation_span"],
                    "side_orientation_span": reference_metrics["side_orientation_span"],
                    "corner_orientation_span": reference_metrics["corner_orientation_span"],
                    "orientation_valid_fraction": reference_metrics["orientation_valid_fraction"],
                    "valid_cross_section_orientation_span": reference_metrics["valid_cross_section_orientation_span"],
                    "valid_side_orientation_span": reference_metrics["valid_side_orientation_span"],
                    "valid_corner_orientation_span": reference_metrics["valid_corner_orientation_span"],
                    "side_corner_weight_overlap_max": reference_metrics["side_corner_weight_overlap_max"],
                    "fitted_cross_section_orientation_span": fitted_metrics["cross_section_orientation_span"],
                    "fitted_side_orientation_span": fitted_metrics["side_orientation_span"],
                    "fitted_corner_orientation_span": fitted_metrics["corner_orientation_span"],
                    "fitted_orientation_valid_fraction": fitted_metrics["orientation_valid_fraction"],
                    "fitted_valid_cross_section_orientation_span": fitted_metrics[
                        "valid_cross_section_orientation_span"
                    ],
                    "fitted_valid_side_orientation_span": fitted_metrics["valid_side_orientation_span"],
                    "fitted_valid_corner_orientation_span": fitted_metrics["valid_corner_orientation_span"],
                    "fitted_side_corner_weight_overlap_max": fitted_metrics["side_corner_weight_overlap_max"],
                    "cross_section_anisotropy_min": reference_metrics["cross_section_anisotropy_min"],
                    "cross_section_anisotropy_max": reference_metrics["cross_section_anisotropy_max"],
                    "fitted_cross_section_anisotropy_min": fitted_metrics["cross_section_anisotropy_min"],
                    "fitted_cross_section_anisotropy_max": fitted_metrics["cross_section_anisotropy_max"],
                    "initialization_policy": _VMEC_JAX_INITIALIZATION_POLICY,
                    "vmec_jax_axis_initialization_policy": _vmec_jax_axis_initialization_policy(args.solver_mode),
                    "min_R": reference_metrics["min_R"],
                    "stellsym_R_error": reference_metrics["stellsym_R_error"],
                    "stellsym_Z_error": reference_metrics["stellsym_Z_error"],
                    "ran_solve": bool(args.run_solve),
                    "solver_mode": str(args.solver_mode),
                    "use_scan": None if args.use_scan is None else bool(args.use_scan),
                    "cli_finish": bool(args.cli_finish),
                    "cli_fixed_boundary_mode": None,
                    "cli_fixed_boundary_initial_policy": None,
                    "cli_fixed_boundary_finish_attempts": 0,
                    "cli_fixed_boundary_finish_budgets": [],
                    "cli_fixed_boundary_finish_fsq": [],
                    "cli_fixed_boundary_finish_converged": [],
                    "cli_fixed_boundary_finish_modes": [],
                    "cli_fixed_boundary_finish_best_fsq": None,
                    "cli_fixed_boundary_finish_budget_cap": None,
                    "cli_fixed_boundary_finish_budget_exhausted": None,
                    "cli_fixed_boundary_full_parity_fallback": None,
                    "cli_fixed_boundary_partial_parity_fallback": None,
                    "cli_fixed_boundary_staged_followup_used": None,
                    "full_solver_diagnostics": bool(args.full_solver_diagnostics),
                    "diagnostic_light_history": None,
                    "diagnostic_resume_state_mode": None,
                    "diagnostic_scan_path": None,
                    "diagnostic_scan_minimal": None,
                    "diagnostic_scan_light": None,
                    "diagnostic_scan_use_precomputed": None,
                    "diagnostic_scan_use_lax_tridi": None,
                    "diagnostic_stage_modes": [],
                    "diagnostic_stage_niter": [],
                    "diagnostic_stage_offsets": [],
                    "diagnostic_step_history_size": 0,
                    "diagnostic_step_iter_history": [],
                    "diagnostic_step_status_history": [],
                    "diagnostic_restart_reason_history": [],
                    "diagnostic_pre_restart_reason_history": [],
                    "diagnostic_time_step_history": [],
                    "diagnostic_time_step_history_size": 0,
                    "diagnostic_dt_eff_history": [],
                    "diagnostic_update_rms_history": [],
                    "diagnostic_w_curr_history": [],
                    "diagnostic_w_try_history": [],
                    "diagnostic_w_try_ratio_history": [],
                    "diagnostic_bcovar_update_history": [],
                    "diagnostic_step_status_counts": {},
                    "diagnostic_restart_reason_counts": {},
                    "diagnostic_bcovar_updates": 0,
                    "diagnostic_initial_time_step": None,
                    "diagnostic_final_time_step": None,
                    "diagnostic_min_time_step": None,
                    "diagnostic_max_time_step": None,
                    "diagnostic_initial_bcovar_update": None,
                    "diagnostic_final_dt_eff": None,
                    "diagnostic_max_update_rms": None,
                    "diagnostic_final_update_rms": None,
                    "diagnostic_initial_axis_reset_attempted": None,
                    "diagnostic_initial_axis_reset_reset": None,
                    "diagnostic_initial_axis_reset_bad_jacobian": None,
                    "diagnostic_initial_axis_reset_force_reset": None,
                    "diagnostic_initial_axis_reset_fsq": None,
                    "diagnostic_initial_axis_reset_ptau_min": None,
                    "diagnostic_initial_axis_reset_ptau_max": None,
                    "diagnostic_initial_axis_reset_state_tau_min": None,
                    "diagnostic_initial_axis_reset_state_tau_max": None,
                    "diagnostic_initial_axis_reset_error": None,
                    "requested_ftol": float(args.ftol),
                    "fsq_total_target": None,
                    "seconds": None,
                    "direct_initial_residual_requested": bool(args.direct_initial_residual),
                    "direct_initial_residual_source": None,
                    "direct_initial_axis_initialization_policy": None,
                    "direct_initial_fsq": None,
                    "direct_initial_fsqr": None,
                    "direct_initial_fsqz": None,
                    "direct_initial_fsql": None,
                    "direct_initial_max_component": None,
                    "direct_initial_max_component_name": None,
                    "direct_initial_max_component_over_ftol": None,
                    "direct_initial_fsq_ratio_vmec2000": None,
                    "direct_initial_fsqr_ratio_vmec2000": None,
                    "direct_initial_fsqz_ratio_vmec2000": None,
                    "direct_initial_fsql_ratio_vmec2000": None,
                    "initial_fsq_ratio_direct_initial": None,
                    "vmec2000_initial_fsq_ratio_direct_initial": None,
                    "direct_initial_error": None,
                    "initial_residual_source": None,
                    "initial_fsq": None,
                    "best_fsq": None,
                    "best_iter": None,
                    "fsq_reduction": None,
                    "final_fsq": None,
                    "initial_fsqr": None,
                    "initial_fsqz": None,
                    "initial_fsql": None,
                    "initial_max_component": None,
                    "initial_max_component_name": None,
                    "initial_max_component_over_ftol": None,
                    "final_fsqr": None,
                    "final_fsqz": None,
                    "final_fsql": None,
                    "final_max_component": None,
                    "final_max_component_name": None,
                    "final_max_component_over_ftol": None,
                    "best_fsqr": None,
                    "best_fsqz": None,
                    "best_fsql": None,
                    "best_max_component": None,
                    "best_max_component_name": None,
                    "best_max_component_over_ftol": None,
                    "strict_component_pass": None,
                    "strict_component_bottleneck": None,
                    "strict_component_margin": None,
                    "converged": None,
                    "converged_strict": None,
                    "converged_by_total_fsq": None,
                    "n_iter": None,
                    "aspect": None,
                    "mean_iota": None,
                    "magnetic_well": None,
                    "fsq_history": [],
                    "iter_history": [],
                    "fsqr_history": [],
                    "fsqz_history": [],
                    "fsql_history": [],
                    "wout": None,
                    "ran_vmec2000": bool(args.run_vmec2000),
                    "vmec2000_initialization_policy": _VMEC2000_INITIALIZATION_POLICY,
                    "vmec2000_returncode": None,
                    "vmec2000_runtime_s": None,
                    "vmec2000_n_rows": None,
                    "vmec2000_initial_residual_source": None,
                    "vmec2000_initial_fsq": None,
                    "vmec2000_best_fsq": None,
                    "vmec2000_best_iter": None,
                    "vmec2000_fsq_reduction": None,
                    "vmec2000_final_fsq": None,
                    "vmec2000_initial_fsqr": None,
                    "vmec2000_initial_fsqz": None,
                    "vmec2000_initial_fsql": None,
                    "vmec2000_initial_max_component": None,
                    "vmec2000_initial_max_component_name": None,
                    "vmec2000_initial_max_component_over_ftol": None,
                    "vmec2000_final_fsqr": None,
                    "vmec2000_final_fsqz": None,
                    "vmec2000_final_fsql": None,
                    "vmec2000_final_max_component": None,
                    "vmec2000_final_max_component_name": None,
                    "vmec2000_final_max_component_over_ftol": None,
                    "vmec2000_aspect": None,
                    "vmec2000_mean_iota": None,
                    "vmec2000_iter_history": [],
                    "vmec2000_fsq_history": [],
                    "vmec2000_fsqr_history": [],
                    "vmec2000_fsqz_history": [],
                    "vmec2000_fsql_history": [],
                    "initial_fsq_ratio_vmec2000": None,
                    "initial_fsqr_ratio_vmec2000": None,
                    "initial_fsqz_ratio_vmec2000": None,
                    "initial_fsql_ratio_vmec2000": None,
                    "vmec2000_threed1": None,
                    "vmec2000_wout": None,
                    "vmec2000_error": None,
                }
                if bool(args.run_solve) and bool(args.direct_initial_residual):
                    try:
                        row.update(
                            _compute_direct_initial_residual(
                                input_path,
                                solver_mode=str(args.solver_mode),
                                use_scan=args.use_scan,
                            )
                        )
                    except Exception as exc:
                        row["direct_initial_error"] = str(exc)
                if bool(args.run_solve):
                    t0 = perf_counter()
                    run = vj.run_fixed_boundary(
                        input_path,
                        solver="vmec2000_iter",
                        solver_mode=str(args.solver_mode),
                        use_scan=args.use_scan,
                        max_iter=int(args.max_iter),
                        light_history=False if bool(args.full_solver_diagnostics) else None,
                        cli_fixed_boundary_mode=bool(args.cli_finish),
                        verbose=False,
                    )
                    row["seconds"] = float(perf_counter() - t0)
                    diag = dict(run.result.diagnostics) if run.result is not None else {}
                    row["converged"] = bool(diag.get("converged", False))
                    row["converged_strict"] = bool(diag.get("converged_strict", False))
                    row["converged_by_total_fsq"] = bool(diag.get("converged_by_total_fsq", False))
                    if diag.get("requested_ftol") is not None:
                        row["requested_ftol"] = float(diag["requested_ftol"])
                    if diag.get("fsq_total_target") is not None:
                        row["fsq_total_target"] = float(diag["fsq_total_target"])
                    row.update(_cli_finish_diagnostic_fields(diag))
                    row["n_iter"] = int(getattr(run.result, "n_iter", -1)) if run.result is not None else None
                    best_component_index = None
                    fsq_history = np.zeros((0,), dtype=float)
                    if run.result is not None and getattr(run.result, "w_history", None) is not None:
                        fsq_history = np.asarray(run.result.w_history, dtype=float).reshape(-1)
                        row["fsq_history"] = [float(value) for value in fsq_history]
                        iter_history = np.asarray(diag.get("iter2_history", []), dtype=int).reshape(-1)
                        if iter_history.size == fsq_history.size:
                            row["iter_history"] = [int(value) for value in iter_history]
                        else:
                            row["iter_history"] = [int(value) for value in range(1, fsq_history.size + 1)]
                        row.update(_summarize_fsq_history(fsq_history))
                        if fsq_history.size and np.any(np.isfinite(fsq_history)):
                            best_component_index = int(
                                np.argmin(np.where(np.isfinite(fsq_history), fsq_history, np.inf))
                            )
                        row["initial_residual_source"] = "vmec_jax_solve_history_first_stored_row"
                        for source, history_key, initial_key, final_key, best_key in (
                            ("fsqr2_history", "fsqr_history", "initial_fsqr", "final_fsqr", "best_fsqr"),
                            ("fsqz2_history", "fsqz_history", "initial_fsqz", "final_fsqz", "best_fsqz"),
                            ("fsql2_history", "fsql_history", "initial_fsql", "final_fsql", "best_fsql"),
                        ):
                            component = np.asarray(getattr(run.result, source, []), dtype=float).reshape(-1)
                            row[history_key] = [float(value) for value in component]
                            if component.size:
                                row[initial_key] = float(component[0])
                                row[final_key] = float(component[-1])
                                if best_component_index is not None and 0 <= int(best_component_index) < component.size:
                                    row[best_key] = float(component[int(best_component_index)])
                    row.update(_solver_diagnostic_fields(diag, fallback_size=int(fsq_history.size)))
                    try:
                        row["aspect"] = float(
                            vj.equilibrium_aspect_ratio_from_state(state=run.state, static=run.static)
                        )
                    except Exception:
                        row["aspect"] = None
                    try:
                        _chips, iotas, _iotaf = vj.equilibrium_iota_profiles_from_state(
                            state=run.state,
                            static=run.static,
                            indata=run.indata,
                            signgs=int(run.signgs),
                        )
                        row["mean_iota"] = float(np.nanmean(np.asarray(iotas, dtype=float)))
                    except Exception:
                        row["mean_iota"] = None
                    try:
                        row["magnetic_well"] = float(
                            vj.magnetic_well_from_state(
                                state=run.state,
                                static=run.static,
                                indata=run.indata,
                                signgs=int(run.signgs),
                            )
                        )
                    except Exception:
                        row["magnetic_well"] = None
                    wout_path = case_dir / "wout_toroidal_stellarator_mirror_hybrid.nc"
                    vj.write_wout_from_fixed_boundary_run(wout_path, run, include_fsq=True)
                    row["wout"] = str(wout_path)
                if bool(args.run_vmec2000):
                    try:
                        vmec2000 = run_xvmec2000(
                            input_path,
                            exec_path=vmec2000_exec,
                            workdir=case_dir / "vmec2000",
                            timeout_s=float(args.vmec2000_timeout_s),
                            keep_workdir=True,
                        )
                        row["vmec2000_returncode"] = int(vmec2000.returncode)
                        row["vmec2000_runtime_s"] = float(vmec2000.runtime_s)
                        row["vmec2000_threed1"] = (
                            str(vmec2000.threed1_path) if vmec2000.threed1_path is not None else None
                        )
                        vmec2000_wouts = sorted(vmec2000.workdir.glob("wout*.nc"))
                        row["vmec2000_wout"] = str(vmec2000_wouts[0]) if vmec2000_wouts else None
                        if vmec2000_wouts:
                            try:
                                vmec2000_wout = read_wout(vmec2000_wouts[0])
                                row["vmec2000_final_fsqr"] = float(vmec2000_wout.fsqr)
                                row["vmec2000_final_fsqz"] = float(vmec2000_wout.fsqz)
                                row["vmec2000_final_fsql"] = float(vmec2000_wout.fsql)
                                row["vmec2000_aspect"] = float(vmec2000_wout.aspect)
                                row["vmec2000_mean_iota"] = float(
                                    np.nanmean(np.asarray(vmec2000_wout.iotas, dtype=float))
                                )
                            except Exception:
                                pass
                        vmec2000_rows = flatten_threed1(vmec2000.stages)
                        row["vmec2000_n_rows"] = len(vmec2000_rows)
                        if vmec2000_rows:
                            vmec2000_iters = np.asarray([item.it for item in vmec2000_rows], dtype=int)
                            vmec2000_fsq = threed1_fsq_total(vmec2000_rows)
                            row["vmec2000_iter_history"] = [int(value) for value in vmec2000_iters]
                            row["vmec2000_fsq_history"] = [float(value) for value in vmec2000_fsq]
                            vmec2000_summary = _summarize_fsq_history(vmec2000_fsq, iterations=vmec2000_iters)
                            row["vmec2000_initial_fsq"] = vmec2000_summary["initial_fsq"]
                            row["vmec2000_best_fsq"] = vmec2000_summary["best_fsq"]
                            row["vmec2000_best_iter"] = vmec2000_summary["best_iter"]
                            row["vmec2000_fsq_reduction"] = vmec2000_summary["fsq_reduction"]
                            row["vmec2000_final_fsq"] = vmec2000_summary["final_fsq"]
                            row["vmec2000_fsqr_history"] = [float(item.fsqr) for item in vmec2000_rows]
                            row["vmec2000_fsqz_history"] = [float(item.fsqz) for item in vmec2000_rows]
                            row["vmec2000_fsql_history"] = [float(item.fsql) for item in vmec2000_rows]
                            row["vmec2000_initial_residual_source"] = "vmec2000_threed1_first_row"
                            row["vmec2000_initial_fsqr"] = float(vmec2000_rows[0].fsqr)
                            row["vmec2000_initial_fsqz"] = float(vmec2000_rows[0].fsqz)
                            row["vmec2000_initial_fsql"] = float(vmec2000_rows[0].fsql)
                    except Exception as exc:
                        row["vmec2000_error"] = str(exc)
                _attach_initial_residual_comparison(row)
                _attach_component_max_diagnostics(row)
                rows.append(row)

    if not rows:
        raise ValueError("case_filter selected no toroidal hybrid rows")

    summary = {
        "shape_cases": [{"name": name, "sample_parameters": kwargs} for name, kwargs in shape_cases],
        "resolution_preset": str(args.resolution_preset),
        "resolution_preset_description": str(resolution_preset["description"]),
        "target_resolution_ladder": bool(resolution_preset["target_resolution_ladder"]),
        "target_resolution_promotion_claim": False,
        "case_filters": list(case_filters),
        "rows": rows,
        "csv": _write_rows_csv(rows, outdir=outdir),
        "figures": {},
    }
    if not bool(args.no_plots):
        summary["figures"] = _write_convergence_figures(rows, outdir=outdir / "figures")

    summary_path = outdir / "toroidal_stellarator_mirror_hybrid_convergence.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(summary_path)


if __name__ == "__main__":  # pragma: no cover
    main()
