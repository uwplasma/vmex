"""Sample circular-coil fields for the mirror free-boundary planning lane."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vmec_jax.mirror import (
    IPrimeProfile,
    MirrorBoundary,
    MirrorCircularCoils,
    MirrorConfig,
    MirrorFreeBoundaryLoopState,
    MirrorResolution,
    MirrorSolveOptions,
    PressureProfile,
    PsiPrimeProfile,
    initial_mirror_boundary_from_circular_coil_scan,
    load_mirror_output,
    make_mirror_free_boundary_circular_coil_scan,
    make_mirror_grid,
    mirror_external_bnormal,
    mirror_external_pressure_balance_response,
    mirror_free_boundary_guarded_least_squares_loop,
    mirror_free_boundary_least_squares_step,
    mirror_free_boundary_residual,
    mirror_lcfs_diagnostic,
    mirror_lcfs_diagnostic_from_arrays,
    mirror_lcfs_merit,
    mirror_lcfs_residual,
    plot_mirror_output,
    propose_axisymmetric_mirror_lcfs_candidate_set,
    run_mirror_fixed_boundary,
    sample_mirror_axis_external_field,
    sample_mirror_boundary_external_field,
    two_coil_on_axis_bz,
    write_mirror_output,
    write_mirror_free_boundary_circular_coil_scan,
)


CIRCULAR_COIL_BETA_SCAN_SCHEMA = "mirror_free_boundary_circular_coil_beta_scan"
CIRCULAR_COIL_BETA_SCAN_SCHEMA_VERSION = "0.6"
CIRCULAR_COIL_BETA_SCAN_LS_LINE_SEARCH_FACTORS = (1.0, 0.5, 0.25, 0.125)
CIRCULAR_COIL_BETA_SCAN_TOP_LEVEL_FIELDS = (
    "metrics_schema",
    "metrics_schema_version",
    "workflow_status",
    "free_boundary_solve_status",
    "external_field_provider_kind",
    "coil_format",
    "coil_radius",
    "separation",
    "current",
    "midplane_radius",
    "ns",
    "ntheta",
    "nxi",
    "n_segments",
    "axis_bz_relative_linf",
    "boundary_bmag_min",
    "boundary_bmag_max",
    "setup_json",
    "summary_csv",
    "summary_rows",
    "beta_scan_requested_percent",
    "beta_cases",
    "fixed_boundary_baseline_count",
    "fixed_boundary_baseline_rows",
    "lcfs_pilot_requested",
    "lcfs_pilot_steps_requested",
    "lcfs_pilot_target_merit",
    "lcfs_pilot_stagnation_rtol",
    "lcfs_pilot_fsq_growth_limit",
    "lcfs_pilot_rows_total",
    "lcfs_pilot_accepted_rows_total",
    "lcfs_pilot_skipped_rows_total",
    "lcfs_pilot_stop_reason_counts",
    "ls_boundary_step_requested",
    "ls_boundary_finite_difference_step",
    "ls_boundary_damping",
    "ls_boundary_max_relative_step",
    "ls_boundary_ridge",
    "ls_boundary_step_rows_total",
    "ls_boundary_coupled_trial_requested",
    "ls_boundary_coupled_trial_rows_total",
    "ls_boundary_coupled_loop_requested",
    "ls_boundary_coupled_loop_steps_requested",
    "ls_boundary_coupled_loop_target_merit",
    "ls_boundary_coupled_loop_stagnation_rtol",
    "ls_boundary_coupled_loop_fsq_growth_limit",
    "ls_boundary_coupled_loop_rows_total",
    "ls_boundary_coupled_loop_accepted_rows_total",
    "ls_boundary_coupled_loop_stop_reason_counts",
    "figures",
)
CIRCULAR_COIL_BETA_SCAN_ROW_FIELDS = (
    "beta_percent",
    "beta_fraction",
    "pressure_scale",
    "mout",
    "optimizer",
    "optimizer_success",
    "optimizer_nit",
    "final_residual_norm",
    "final_fsq",
    "final_normalized_force",
    "lcfs_external_bnormal_rms",
    "lcfs_pressure_balance_rms",
    "lcfs_merit",
    "lcfs_update_strategy",
    "lcfs_update_candidate_summaries",
    "lcfs_update_allowed_strategies",
    "lcfs_update_rejection_reason",
    "lcfs_pilot_status",
    "lcfs_pilot_rows_count",
    "lcfs_pilot_accepted_rows",
    "lcfs_pilot_skipped_rows",
    "lcfs_pilot_final_fsq_growth_ratio",
    "lcfs_pilot_best_fsq_growth_ratio",
    "lcfs_pilot_stop_reason",
    "lcfs_pilot_last_accepted_step",
    "lcfs_pilot_last_accepted_merit",
    "lcfs_pilot_last_accepted_pressure_balance_rms",
    "lcfs_pilot_last_accepted_fsq",
    "lcfs_pilot_last_accepted_fsq_growth_ratio",
    "lcfs_pilot_last_accepted_normalized_force",
    "lcfs_pilot_rows",
    "ls_boundary_step",
    "ls_boundary_coupled_loop_status",
    "ls_boundary_coupled_loop_rows_count",
    "ls_boundary_coupled_loop_accepted_rows",
    "ls_boundary_coupled_loop_stop_reason",
    "ls_boundary_coupled_loop_final_merit",
    "ls_boundary_coupled_loop_final_fsq_growth_ratio",
    "ls_boundary_coupled_loop_last_accepted_step",
    "ls_boundary_coupled_loop_last_accepted_merit",
    "ls_boundary_coupled_loop_last_accepted_fsq_growth_ratio",
    "ls_boundary_coupled_loop_rows",
    "figures",
)
CIRCULAR_COIL_BETA_SCAN_LS_STEP_FIELDS = (
    "accepted",
    "line_search_factor",
    "coefficients_initial",
    "coefficients_new",
    "raw_step",
    "limited_step",
    "finite_difference_steps",
    "jacobian_shape",
    "residual_value_before",
    "residual_value_after",
    "predicted_value",
    "equilibrium_rms_before",
    "equilibrium_rms_after",
    "lcfs_value_before",
    "lcfs_value_after",
    "max_relative_radius_change",
    "trial_rows",
    "coupled_trial",
    "figure",
)
CIRCULAR_COIL_BETA_SCAN_LS_COUPLED_TRIAL_FIELDS = (
    "status",
    "mout",
    "accepted_by_merit",
    "rejection_reason",
    "final_residual_norm",
    "final_fsq",
    "final_normalized_force",
    "fsq_growth_ratio",
    "lcfs_external_bnormal_rms",
    "lcfs_pressure_balance_rms",
    "lcfs_merit",
    "lcfs_merit_ratio",
    "figures",
)
CIRCULAR_COIL_BETA_SCAN_LS_COUPLED_LOOP_ROW_FIELDS = (
    "step",
    "status",
    "accepted",
    "rejection_reason",
    "stop_reason",
    "merit_improvement_fraction",
    "fsq_growth_ratio",
    "final_fsq",
    "final_normalized_force",
    "lcfs_merit",
    "lcfs_merit_ratio",
    "mout",
    "ls_boundary_step",
    "figures",
)
CIRCULAR_COIL_BETA_SCAN_PILOT_ROW_FIELDS = (
    "step",
    "mout",
    "accepted",
    "rejection_reason",
    "stop_reason",
    "lcfs_merit_improvement_fraction",
    "fsq_growth_ratio",
    "final_residual_norm",
    "final_fsq",
    "final_normalized_force",
    "lcfs_external_bnormal_rms",
    "lcfs_pressure_balance_rms",
    "lcfs_merit",
    "lcfs_update_strategy_next",
    "lcfs_update_candidate_summaries_next",
    "lcfs_update_allowed_strategies_next",
    "lcfs_update_rejection_reason_next",
    "figures",
)
CIRCULAR_COIL_BETA_SCAN_PILOT_STATUSES = ("not_requested", "accepted", "rejected", "skipped")
CIRCULAR_COIL_BETA_SCAN_WORKFLOW_STATUSES = (
    "setup_only",
    "fixed_boundary_baseline",
    "lcfs_pilot",
    "ls_boundary_coupled_loop",
)
CIRCULAR_COIL_BETA_SCAN_FREE_BOUNDARY_STATUSES = (
    "not_run",
    "lcfs_pilot_not_converged_free_boundary",
    "ls_boundary_coupled_loop_not_converged_free_boundary",
)
CIRCULAR_COIL_BETA_SCAN_STOP_REASONS = (
    "fsq_growth_guard",
    "ls_step_not_accepted",
    "max_steps",
    "merit_stagnation",
    "noop_candidate",
    "rejected_merit_increase",
    "target_merit",
)
CIRCULAR_COIL_BETA_SCAN_REJECTION_REASONS = (
    "fsq_growth_guard",
    "ls_step_not_accepted",
    "merit_increase",
    "noop_candidate_selected",
    "normal_field_guard_no_candidate",
)
CIRCULAR_COIL_BETA_SCAN_REPORT_FIELDS = (
    "beta_percent",
    "baseline_final_fsq",
    "baseline_final_normalized_force",
    "baseline_lcfs_merit",
    "baseline_pressure_balance_rms",
    "baseline_external_bnormal_rms",
    "pilot_status",
    "pilot_accepted_rows",
    "pilot_stop_reason",
    "last_accepted_step",
    "last_accepted_fsq",
    "last_accepted_fsq_growth_ratio",
    "last_accepted_lcfs_merit",
    "last_accepted_pressure_balance_rms",
    "last_accepted_normalized_force",
    "final_trial_fsq",
    "final_trial_fsq_growth_ratio",
    "final_trial_lcfs_merit",
    "final_trial_pressure_balance_rms",
    "final_trial_normalized_force",
)


def circular_coil_beta_scan_schema() -> dict[str, object]:
    """Return the compact JSON contract written by this planning fixture."""
    return {
        "metrics_schema": CIRCULAR_COIL_BETA_SCAN_SCHEMA,
        "metrics_schema_version": CIRCULAR_COIL_BETA_SCAN_SCHEMA_VERSION,
        "top_level_required_fields": list(CIRCULAR_COIL_BETA_SCAN_TOP_LEVEL_FIELDS),
        "beta_row_required_fields": list(CIRCULAR_COIL_BETA_SCAN_ROW_FIELDS),
        "pilot_row_required_fields": list(CIRCULAR_COIL_BETA_SCAN_PILOT_ROW_FIELDS),
        "ls_boundary_step_fields": list(CIRCULAR_COIL_BETA_SCAN_LS_STEP_FIELDS),
        "ls_boundary_coupled_trial_fields": list(CIRCULAR_COIL_BETA_SCAN_LS_COUPLED_TRIAL_FIELDS),
        "ls_boundary_coupled_loop_row_fields": list(CIRCULAR_COIL_BETA_SCAN_LS_COUPLED_LOOP_ROW_FIELDS),
        "pilot_status_values": list(CIRCULAR_COIL_BETA_SCAN_PILOT_STATUSES),
        "workflow_status_values": list(CIRCULAR_COIL_BETA_SCAN_WORKFLOW_STATUSES),
        "free_boundary_status_values": list(CIRCULAR_COIL_BETA_SCAN_FREE_BOUNDARY_STATUSES),
        "pilot_stop_reasons": list(CIRCULAR_COIL_BETA_SCAN_STOP_REASONS),
        "pilot_rejection_reasons": list(CIRCULAR_COIL_BETA_SCAN_REJECTION_REASONS),
        "report_fields": list(CIRCULAR_COIL_BETA_SCAN_REPORT_FIELDS),
    }


def validate_circular_coil_beta_scan_metrics(metrics: dict[str, object]) -> None:
    """Raise ``ValueError`` if the metrics JSON does not follow the schema."""
    _require_fields(metrics, CIRCULAR_COIL_BETA_SCAN_TOP_LEVEL_FIELDS, "top-level metrics")
    if metrics["metrics_schema"] != CIRCULAR_COIL_BETA_SCAN_SCHEMA:
        raise ValueError(f"unexpected metrics schema {metrics['metrics_schema']!r}")
    if metrics["metrics_schema_version"] != CIRCULAR_COIL_BETA_SCAN_SCHEMA_VERSION:
        raise ValueError(f"unexpected metrics schema version {metrics['metrics_schema_version']!r}")
    workflow_status = str(metrics["workflow_status"])
    if workflow_status not in CIRCULAR_COIL_BETA_SCAN_WORKFLOW_STATUSES:
        raise ValueError(f"unknown workflow_status {workflow_status!r}")
    free_boundary_status = str(metrics["free_boundary_solve_status"])
    if free_boundary_status not in CIRCULAR_COIL_BETA_SCAN_FREE_BOUNDARY_STATUSES:
        raise ValueError(f"unknown free_boundary_solve_status {free_boundary_status!r}")
    beta_requested = metrics.get("beta_scan_requested_percent", [])
    beta_cases = metrics.get("beta_cases", [])
    if len(beta_requested) != len(beta_cases):
        raise ValueError("beta_scan_requested_percent and beta_cases must have the same length")
    baseline_rows = metrics.get("fixed_boundary_baseline_rows", [])
    if int(metrics["fixed_boundary_baseline_count"]) != len(baseline_rows):
        raise ValueError("fixed_boundary_baseline_count does not match fixed_boundary_baseline_rows")
    summary_rows = metrics.get("summary_rows", [])
    if len(summary_rows) != len(baseline_rows):
        raise ValueError("summary_rows must have one row per fixed-boundary baseline row")
    for index, report_row in enumerate(summary_rows):
        if not isinstance(report_row, dict):
            raise ValueError(f"summary row {index} must be a JSON object")
        _require_fields(report_row, CIRCULAR_COIL_BETA_SCAN_REPORT_FIELDS, f"summary row {index}")

    pilot_rows = [pilot for row in baseline_rows if isinstance(row, dict) for pilot in row.get("lcfs_pilot_rows", [])]
    ls_rows = [
        row.get("ls_boundary_step") for row in baseline_rows if isinstance(row, dict) and row.get("ls_boundary_step")
    ]
    ls_trial_rows = [
        row["coupled_trial"] for row in ls_rows if isinstance(row, dict) and row.get("coupled_trial") is not None
    ]
    ls_loop_rows = [
        loop_row
        for row in baseline_rows
        if isinstance(row, dict)
        for loop_row in row.get("ls_boundary_coupled_loop_rows", [])
    ]
    ls_loop_accepted_rows = sum(bool(row.get("accepted", False)) for row in ls_loop_rows)
    accepted_rows = sum(bool(row.get("accepted", False)) and not bool(row.get("skipped", False)) for row in pilot_rows)
    skipped_rows = sum(bool(row.get("skipped", False)) for row in pilot_rows)
    if int(metrics["lcfs_pilot_rows_total"]) != len(pilot_rows):
        raise ValueError("lcfs_pilot_rows_total does not match nested pilot rows")
    if int(metrics["lcfs_pilot_accepted_rows_total"]) != int(accepted_rows):
        raise ValueError("lcfs_pilot_accepted_rows_total does not match nested pilot rows")
    if int(metrics["lcfs_pilot_skipped_rows_total"]) != int(skipped_rows):
        raise ValueError("lcfs_pilot_skipped_rows_total does not match nested pilot rows")
    if int(metrics["ls_boundary_step_rows_total"]) != len(ls_rows):
        raise ValueError("ls_boundary_step_rows_total does not match nested LS rows")
    if int(metrics["ls_boundary_coupled_trial_rows_total"]) != len(ls_trial_rows):
        raise ValueError("ls_boundary_coupled_trial_rows_total does not match nested LS trial rows")
    if int(metrics["ls_boundary_coupled_loop_rows_total"]) != len(ls_loop_rows):
        raise ValueError("ls_boundary_coupled_loop_rows_total does not match nested LS loop rows")
    if int(metrics["ls_boundary_coupled_loop_accepted_rows_total"]) != int(ls_loop_accepted_rows):
        raise ValueError("ls_boundary_coupled_loop_accepted_rows_total does not match nested LS loop rows")
    if metrics["ls_boundary_coupled_loop_stop_reason_counts"] != _counts_json(
        [str(row.get("stop_reason")) for row in ls_loop_rows]
    ):
        raise ValueError("ls_boundary_coupled_loop_stop_reason_counts does not match nested LS loop rows")
    if metrics["lcfs_pilot_stop_reason_counts"] != _counts_json([str(row.get("stop_reason")) for row in pilot_rows]):
        raise ValueError("lcfs_pilot_stop_reason_counts does not match nested pilot rows")

    for index, row in enumerate(baseline_rows):
        if not isinstance(row, dict):
            raise ValueError(f"baseline row {index} must be a JSON object")
        _require_fields(row, CIRCULAR_COIL_BETA_SCAN_ROW_FIELDS, f"baseline row {index}")
        case_pilot_rows = row.get("lcfs_pilot_rows", [])
        if not isinstance(case_pilot_rows, list):
            raise ValueError(f"baseline row {index} lcfs_pilot_rows must be a JSON array")
        case_loop_rows = row.get("ls_boundary_coupled_loop_rows", [])
        if not isinstance(case_loop_rows, list):
            raise ValueError(f"baseline row {index} ls_boundary_coupled_loop_rows must be a JSON array")
        status = str(row["lcfs_pilot_status"])
        if status not in CIRCULAR_COIL_BETA_SCAN_PILOT_STATUSES:
            raise ValueError(f"baseline row {index} has unknown lcfs_pilot_status {status!r}")
        ls_step = row.get("ls_boundary_step")
        if ls_step is not None:
            if not isinstance(ls_step, dict):
                raise ValueError(f"baseline row {index} ls_boundary_step must be a JSON object or null")
            _require_fields(ls_step, CIRCULAR_COIL_BETA_SCAN_LS_STEP_FIELDS, f"baseline row {index} ls_boundary_step")
            coupled_trial = ls_step.get("coupled_trial")
            if coupled_trial is not None:
                if not isinstance(coupled_trial, dict):
                    raise ValueError(
                        f"baseline row {index} ls_boundary_step coupled_trial must be a JSON object or null"
                    )
                _require_fields(
                    coupled_trial,
                    CIRCULAR_COIL_BETA_SCAN_LS_COUPLED_TRIAL_FIELDS,
                    f"baseline row {index} ls_boundary_step coupled_trial",
                )
        for step_index, pilot in enumerate(case_pilot_rows):
            if not isinstance(pilot, dict):
                raise ValueError(f"pilot row {index}.{step_index} must be a JSON object")
            _require_fields(pilot, CIRCULAR_COIL_BETA_SCAN_PILOT_ROW_FIELDS, f"pilot row {index}.{step_index}")
            stop_reason = pilot.get("stop_reason")
            if stop_reason is not None and str(stop_reason) not in CIRCULAR_COIL_BETA_SCAN_STOP_REASONS:
                raise ValueError(f"pilot row {index}.{step_index} has unknown stop_reason {stop_reason!r}")
            rejection_reason = pilot.get("rejection_reason")
            if rejection_reason is not None and str(rejection_reason) not in CIRCULAR_COIL_BETA_SCAN_REJECTION_REASONS:
                raise ValueError(f"pilot row {index}.{step_index} has unknown rejection_reason {rejection_reason!r}")
        for step_index, loop_row in enumerate(case_loop_rows):
            if not isinstance(loop_row, dict):
                raise ValueError(f"LS loop row {index}.{step_index} must be a JSON object")
            _require_fields(
                loop_row,
                CIRCULAR_COIL_BETA_SCAN_LS_COUPLED_LOOP_ROW_FIELDS,
                f"LS loop row {index}.{step_index}",
            )
            stop_reason = loop_row.get("stop_reason")
            if stop_reason is not None and str(stop_reason) not in CIRCULAR_COIL_BETA_SCAN_STOP_REASONS:
                raise ValueError(f"LS loop row {index}.{step_index} has unknown stop_reason {stop_reason!r}")
            rejection_reason = loop_row.get("rejection_reason")
            if rejection_reason is not None and str(rejection_reason) not in CIRCULAR_COIL_BETA_SCAN_REJECTION_REASONS:
                raise ValueError(f"LS loop row {index}.{step_index} has unknown rejection_reason {rejection_reason!r}")
            ls_loop_step = loop_row.get("ls_boundary_step")
            if not isinstance(ls_loop_step, dict):
                raise ValueError(f"LS loop row {index}.{step_index} ls_boundary_step must be a JSON object")
            _require_fields(
                ls_loop_step,
                CIRCULAR_COIL_BETA_SCAN_LS_STEP_FIELDS,
                f"LS loop row {index}.{step_index} ls_boundary_step",
            )
            coupled_trial = ls_loop_step.get("coupled_trial")
            if coupled_trial is not None:
                if not isinstance(coupled_trial, dict):
                    raise ValueError(f"LS loop row {index}.{step_index} coupled_trial must be a JSON object or null")
                _require_fields(
                    coupled_trial,
                    CIRCULAR_COIL_BETA_SCAN_LS_COUPLED_TRIAL_FIELDS,
                    f"LS loop row {index}.{step_index} coupled_trial",
                )
        for field, expected_value in _lcfs_pilot_summary(case_pilot_rows).items():
            if row.get(field) != expected_value:
                raise ValueError(f"baseline row {index} field {field} does not match nested pilot rows")
        for field, expected_value in _ls_boundary_coupled_loop_summary(case_loop_rows).items():
            if row.get(field) != expected_value:
                raise ValueError(f"baseline row {index} field {field} does not match nested LS loop rows")

    for index, expected_row in enumerate(circular_coil_beta_scan_report_rows(metrics)):
        report_row = summary_rows[index]
        for field, expected_value in expected_row.items():
            if report_row.get(field) != expected_value:
                raise ValueError(f"summary row {index} field {field} does not match fixed-boundary baseline row")


def _require_fields(row: dict[str, object], fields: tuple[str, ...], label: str) -> None:
    missing = [field for field in fields if field not in row]
    if missing:
        raise ValueError(f"{label} is missing required fields: {', '.join(missing)}")


def circular_coil_beta_scan_report_rows(metrics: dict[str, object]) -> list[dict[str, object]]:
    """Return compact table rows for beta-scan reports and ESSOS comparisons."""
    rows = []
    for row in metrics.get("fixed_boundary_baseline_rows", []):
        if not isinstance(row, dict):
            continue
        rows.append(
            {
                "beta_percent": row.get("beta_percent"),
                "baseline_final_fsq": row.get("final_fsq"),
                "baseline_final_normalized_force": row.get("final_normalized_force"),
                "baseline_lcfs_merit": row.get("lcfs_merit"),
                "baseline_pressure_balance_rms": row.get("lcfs_pressure_balance_rms"),
                "baseline_external_bnormal_rms": row.get("lcfs_external_bnormal_rms"),
                "pilot_status": row.get("lcfs_pilot_status"),
                "pilot_accepted_rows": row.get("lcfs_pilot_accepted_rows"),
                "pilot_stop_reason": row.get("lcfs_pilot_stop_reason"),
                "last_accepted_step": row.get("lcfs_pilot_last_accepted_step"),
                "last_accepted_fsq": row.get("lcfs_pilot_last_accepted_fsq"),
                "last_accepted_fsq_growth_ratio": row.get("lcfs_pilot_last_accepted_fsq_growth_ratio"),
                "last_accepted_lcfs_merit": row.get("lcfs_pilot_last_accepted_merit"),
                "last_accepted_pressure_balance_rms": row.get("lcfs_pilot_last_accepted_pressure_balance_rms"),
                "last_accepted_normalized_force": row.get("lcfs_pilot_last_accepted_normalized_force"),
                "final_trial_fsq": row.get("lcfs_pilot_final_fsq"),
                "final_trial_fsq_growth_ratio": row.get("lcfs_pilot_final_fsq_growth_ratio"),
                "final_trial_lcfs_merit": row.get("lcfs_pilot_final_merit"),
                "final_trial_pressure_balance_rms": row.get("lcfs_pilot_final_pressure_balance_rms"),
                "final_trial_normalized_force": row.get("lcfs_pilot_final_normalized_force"),
            }
        )
    return rows


def _write_beta_scan_report_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CIRCULAR_COIL_BETA_SCAN_REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


@dataclass(frozen=True)
class _LCFSProposalSelection:
    proposal: object
    candidates: tuple[object, ...]
    candidate_summaries: list[dict[str, object]]
    allowed_strategies: tuple[str, ...]
    rejection_reason: str | None


@dataclass(frozen=True)
class _LCFSPilotStepResult:
    row: dict[str, object]
    lcfs: object
    merit: object
    selection: _LCFSProposalSelection
    accepted: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("results/mirror/free_boundary_circular_coils"))
    parser.add_argument("--coil-radius", type=float, default=0.35)
    parser.add_argument("--separation", type=float, default=2.0)
    parser.add_argument("--current", type=float, default=1.0e6)
    parser.add_argument("--midplane-radius", type=float, default=0.3)
    parser.add_argument("--ns", type=int, default=7)
    parser.add_argument("--ntheta", type=int, default=32)
    parser.add_argument("--nxi", type=int, default=33)
    parser.add_argument("--n-segments", type=int, default=256)
    parser.add_argument("--betas", type=str, default="1,3,10")
    parser.add_argument("--pressure-scale-one-percent", type=float, default=1.0)
    parser.add_argument("--run-fixed-boundary-baseline", action="store_true")
    parser.add_argument("--baseline-maxiter", type=int, default=0)
    parser.add_argument("--baseline-psi-prime", type=float, default=0.01)
    parser.add_argument("--lcfs-update-damping", type=float, default=0.25)
    parser.add_argument("--lcfs-update-max-relative-step", type=float, default=0.05)
    parser.add_argument("--lcfs-update-cap-taper-power", type=float, default=2.0)
    parser.add_argument("--lcfs-update-smoothing-passes", type=int, default=1)
    parser.add_argument("--lcfs-merit-bnormal-weight", type=float, default=1.0)
    parser.add_argument(
        "--lcfs-proposal-mode",
        choices=("best_predicted", "coupled", "local", "scale", "bnormal", "mixed"),
        default="best_predicted",
    )
    parser.add_argument("--lcfs-coupled-fsq-weight", type=float, default=1.0)
    parser.add_argument("--lcfs-require-bnormal-nonincrease", action="store_true")
    parser.add_argument("--run-lcfs-pilot", action="store_true")
    parser.add_argument("--lcfs-pilot-steps", type=int, default=1)
    parser.add_argument("--lcfs-pilot-target-merit", type=float, default=0.0)
    parser.add_argument("--lcfs-pilot-stagnation-rtol", type=float, default=0.0)
    parser.add_argument("--lcfs-pilot-fsq-growth-limit", type=float, default=0.0)
    parser.add_argument("--run-ls-boundary-step", action="store_true")
    parser.add_argument("--run-ls-boundary-coupled-trial", action="store_true")
    parser.add_argument("--run-ls-boundary-coupled-loop", action="store_true")
    parser.add_argument("--ls-boundary-coupled-loop-steps", type=int, default=1)
    parser.add_argument("--ls-boundary-coupled-loop-target-merit", type=float, default=0.0)
    parser.add_argument("--ls-boundary-coupled-loop-stagnation-rtol", type=float, default=0.0)
    parser.add_argument("--ls-boundary-coupled-loop-fsq-growth-limit", type=float, default=0.0)
    parser.add_argument("--ls-boundary-finite-difference-step", type=float, default=1.0e-5)
    parser.add_argument("--ls-boundary-damping", type=float, default=1.0)
    parser.add_argument("--ls-boundary-max-relative-step", type=float, default=0.1)
    parser.add_argument("--ls-boundary-ridge", type=float, default=1.0e-8)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def _parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in str(value).replace(",", " ").split() if item.strip())


def _write_axis_plot(z, direct_bz, analytic_bz, *, outdir: Path) -> Path:
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(z, analytic_bz, "k-", linewidth=1.8, label="analytic two-coil")
    ax.plot(z, direct_bz, "o", markersize=4.0, label="direct-coil bridge")
    ax.set_xlabel("z")
    ax.set_ylabel("Bz on axis")
    ax.set_title("free-boundary circular-coil bridge")
    ax.legend(fontsize="small")
    fig.tight_layout()
    path = outdir / "free_boundary_circular_coils_axis_bz.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_boundary_bmag_plot(boundary_sample, *, outdir: Path) -> Path:
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    mesh = ax.pcolormesh(
        np.asarray(boundary_sample.z),
        np.asarray(boundary_sample.theta),
        np.asarray(boundary_sample.bmag),
        shading="auto",
    )
    ax.set_xlabel("z")
    ax.set_ylabel("theta")
    ax.set_title("external |B| on sampled mirror boundary")
    fig.colorbar(mesh, ax=ax, label="|B|")
    fig.tight_layout()
    path = outdir / "free_boundary_circular_coils_boundary_bmag.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_geometry_plot(grid, boundary, coils: MirrorCircularCoils, *, outdir: Path) -> Path:
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    theta = grid.theta
    radius = boundary.radius_on_grid_3d(grid)
    z = np.broadcast_to(grid.z[None, :], radius.shape)
    x = radius * np.cos(theta[:, None])
    y = radius * np.sin(theta[:, None])
    coil_theta = np.linspace(0.0, 2.0 * np.pi, 160)

    fig = plt.figure(figsize=(6.5, 4.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(z, x, y, color="lightgray", alpha=0.5, linewidth=0.0)
    for radius_m, z0 in zip(coils.radii_m, coils.z_centers_m, strict=True):
        ax.plot(
            np.full_like(coil_theta, z0),
            radius_m * np.cos(coil_theta),
            radius_m * np.sin(coil_theta),
            color="tab:orange",
            linewidth=2.0,
        )
    ax.set_xlabel("z")
    ax.set_ylabel("x")
    ax.set_zlabel("y")
    ax.set_title("mirror boundary and circular coils")
    ax.set_box_aspect([max(1.0, float(np.ptp(grid.z))), 1, 1])
    ax.view_init(elev=18, azim=-62)
    fig.tight_layout()
    path = outdir / "free_boundary_circular_coils_geometry.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_lcfs_diagnostic_plot(diagnostic, proposal=None, *, outdir: Path, name: str) -> Path:
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    z = np.asarray(diagnostic.z)
    theta = np.asarray(diagnostic.theta)
    bnormal = np.asarray(diagnostic.external_bnormal)
    pressure_balance = np.asarray(diagnostic.pressure_balance)
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 5.2), sharex=True)
    if theta.size == 1:
        axes[0].plot(z, bnormal[0], "o-", markersize=3.0)
        axes[1].plot(z, pressure_balance[0], "o-", markersize=3.0, label="before")
        if proposal is not None:
            axes[1].plot(
                np.asarray(proposal.z),
                np.asarray(proposal.pressure_balance_predicted),
                "s--",
                markersize=3.0,
                label="predicted update",
            )
    else:
        mesh0 = axes[0].pcolormesh(z, theta, bnormal, shading="auto")
        fig.colorbar(mesh0, ax=axes[0], label="B_ext . n")
        mesh1 = axes[1].pcolormesh(z, theta, pressure_balance, shading="auto")
        fig.colorbar(mesh1, ax=axes[1], label="pressure balance")
    axes[0].axhline(0.0, color="k", linewidth=0.8, alpha=0.6)
    axes[1].axhline(0.0, color="k", linewidth=0.8, alpha=0.6)
    axes[0].set_ylabel("B_ext . n")
    axes[1].set_ylabel("pressure balance")
    axes[1].set_xlabel("z")
    axes[0].set_title("LCFS target diagnostic")
    if proposal is not None and theta.size == 1:
        axes[1].legend(fontsize="small")
    fig.tight_layout()
    path = outdir / f"{name}_lcfs_diagnostic.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_beta_scan_summary_plot(rows: list[dict[str, object]], *, outdir: Path) -> Path:
    """Plot baseline and pilot LCFS metrics across requested beta cases."""
    import matplotlib.pyplot as plt

    if not rows:
        raise ValueError("at least one beta row is required")
    outdir.mkdir(parents=True, exist_ok=True)
    beta = np.asarray([float(row["beta_percent"]) for row in rows], dtype=float)
    order = np.argsort(beta)
    beta = beta[order]

    def _ordered(name: str) -> np.ndarray:
        return np.asarray([float(rows[index][name]) for index in order], dtype=float)

    def _ordered_optional(name: str) -> np.ndarray:
        values = []
        for index in order:
            value = rows[index].get(name)
            values.append(np.nan if value is None else float(value))
        return np.asarray(values, dtype=float)

    baseline_pressure = _ordered("lcfs_pressure_balance_rms")
    baseline_bnormal = _ordered("lcfs_external_bnormal_rms")
    baseline_merit = _ordered("lcfs_merit")
    baseline_fsq = _ordered("final_fsq")
    pilot_pressure = _ordered_optional("lcfs_pilot_last_accepted_pressure_balance_rms")
    pilot_merit = _ordered_optional("lcfs_pilot_last_accepted_merit")
    pilot_fsq = _ordered_optional("lcfs_pilot_last_accepted_fsq")

    fig, axes = plt.subplots(4, 1, figsize=(6.6, 8.2), sharex=True)
    axes[0].plot(beta, baseline_pressure, "o-", label="baseline")
    if np.isfinite(pilot_pressure).any():
        axes[0].plot(beta, pilot_pressure, "s--", label="last accepted pilot")
    axes[0].set_ylabel("pressure RMS")
    axes[0].legend(fontsize="small")

    axes[1].plot(beta, baseline_bnormal, "o-", color="tab:green")
    axes[1].set_ylabel("B_ext . n RMS")

    axes[2].plot(beta, baseline_merit, "o-", label="baseline")
    if np.isfinite(pilot_merit).any():
        axes[2].plot(beta, pilot_merit, "s--", label="last accepted pilot")
    axes[2].set_ylabel("LCFS merit")
    axes[2].legend(fontsize="small")

    axes[3].plot(beta, baseline_fsq, "o-", label="baseline")
    if np.isfinite(pilot_fsq).any():
        axes[3].plot(beta, pilot_fsq, "s--", label="last accepted pilot")
    axes[3].set_ylabel("final fsq")
    axes[3].set_xlabel("nominal beta (%)")
    axes[3].legend(fontsize="small")

    for ax in axes:
        finite_positive = [line.get_ydata() for line in ax.lines if np.all(np.asarray(line.get_ydata()) > 0.0)]
        if finite_positive:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
    fig.suptitle("circular-coil mirror beta scan LCFS metrics", y=0.995)
    fig.tight_layout()
    path = outdir / "free_boundary_circular_coils_beta_scan_summary.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _fit_polynomial_boundary_coefficients(grid, boundary) -> np.ndarray:
    """Fit ``r0 * (1 + a2*xi**2 + a4*xi**4)`` to an axisymmetric boundary."""

    xi = np.asarray(grid.xi, dtype=float)
    radius = np.asarray(boundary.radius_on_grid(grid), dtype=float)
    basis = np.column_stack([np.ones_like(xi), xi**2, xi**4])
    linear_coefficients, *_ = np.linalg.lstsq(basis, radius, rcond=1.0e-12)
    r0 = float(linear_coefficients[0])
    if r0 <= 0.0:
        r0 = float(np.mean(radius))
    return np.asarray([r0, linear_coefficients[1] / r0, linear_coefficients[2] / r0], dtype=float)


def _polynomial_boundary_from_coefficients(coefficients) -> MirrorBoundary:
    coefficients = np.asarray(coefficients, dtype=float).ravel()
    if coefficients.size != 3:
        raise ValueError("polynomial boundary coefficients must be [r0, a2, a4]")
    return MirrorBoundary.polynomial_radius(
        r0=float(coefficients[0]),
        a2=float(coefficients[1]),
        a4=float(coefficients[2]),
    )


def _write_ls_boundary_step_plot(summary: dict[str, object], *, outdir: Path, name: str) -> Path:
    """Plot LS residual components over the tried backtracking factors."""

    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    rows = sorted(
        (row for row in summary["trial_rows"] if row.get("residual_value") is not None),
        key=lambda row: float(row["factor"]),
    )
    factors = np.asarray([float(row["factor"]) for row in rows], dtype=float)
    values = np.asarray([float(row["residual_value"]) for row in rows], dtype=float)
    equilibrium = np.asarray([float(row["equilibrium_rms"]) for row in rows], dtype=float)
    lcfs = np.asarray([float(row["lcfs_value"]) for row in rows], dtype=float)
    selected = float(summary["line_search_factor"])

    fig, axes = plt.subplots(2, 1, figsize=(6.6, 5.2), sharex=True)
    axes[0].plot(factors, values, "o-", label="combined")
    axes[0].axvline(selected, color="tab:red", linewidth=1.1, linestyle="--", label="selected")
    axes[0].set_ylabel("combined value")
    axes[0].legend(fontsize="small")
    axes[1].plot(factors, equilibrium, "o-", label="equilibrium RMS")
    axes[1].plot(factors, lcfs, "s--", label="LCFS value")
    axes[1].axvline(selected, color="tab:red", linewidth=1.1, linestyle="--")
    axes[1].set_xlabel("line-search factor")
    axes[1].set_ylabel("component values")
    axes[1].legend(fontsize="small")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle("free-boundary LS boundary step", y=0.995)
    fig.tight_layout()
    path = outdir / f"{name}_ls_boundary_step.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _build_ls_boundary_residual_function(
    *,
    grid,
    coils,
    output,
    final_trace,
    reference_merit,
) -> object:
    """Build the frozen-output residual function used by one LS boundary step."""

    edge_internal_bmag = np.asarray(output.field.bmag, dtype=float)[-1]
    theta = np.asarray(output.theta, dtype=float)
    z = np.asarray(output.z, dtype=float)
    edge_pressure = float(np.asarray(output.profiles.pressure, dtype=float)[-1])
    equilibrium_value = float(final_trace.normalized_force)
    equilibrium_scale = max(abs(equilibrium_value), 1.0e-300)
    pressure_scale = max(float(reference_merit.pressure_scale), 1.0e-300)
    bnormal_scale = max(float(reference_merit.bnormal_scale), 1.0e-300)
    bnormal_weight = float(reference_merit.bnormal_weight)

    def residual_function(items: np.ndarray):
        trial_boundary = _polynomial_boundary_from_coefficients(items)
        boundary_r = trial_boundary.radius_on_grid_3d(grid)
        external_sample = sample_mirror_boundary_external_field(grid, trial_boundary, coils)
        diagnostic = mirror_lcfs_diagnostic_from_arrays(
            theta=theta,
            z=z,
            boundary_r=boundary_r,
            edge_internal_bmag=edge_internal_bmag,
            external_sample=external_sample,
            edge_pressure=edge_pressure,
            mu0=1.0,
        )
        lcfs_residual = mirror_lcfs_residual(
            diagnostic,
            pressure_scale=pressure_scale,
            bnormal_scale=bnormal_scale,
            bnormal_weight=bnormal_weight,
        )
        return mirror_free_boundary_residual(
            np.asarray([equilibrium_value]),
            lcfs_residual,
            equilibrium_scale=equilibrium_scale,
        )

    return residual_function


def _ls_boundary_step_summary_from_step(
    *,
    step,
    residual_function,
    grid,
    line_search_factors,
    write_plots: bool,
    figure_dir: Path,
    name: str,
) -> dict[str, object]:
    """Return the JSON/plot summary for a computed LS boundary step."""

    trial_rows = []
    for factor in (0.0, *tuple(line_search_factors)):
        trial_coefficients = step.coefficients + float(factor) * step.limited_step
        trial_residual = residual_function(trial_coefficients)
        trial_rows.append(
            {
                "factor": float(factor),
                "selected": bool(np.isclose(float(factor), step.line_search_factor)),
                "accepted": bool(trial_residual.value <= step.residual.value + 1.0e-12),
                "residual_value": float(trial_residual.value),
                "equilibrium_rms": float(trial_residual.equilibrium_rms),
                "lcfs_value": float(trial_residual.lcfs_value),
            }
        )

    old_radius = _polynomial_boundary_from_coefficients(step.coefficients).radius_on_grid(grid)
    new_radius = _polynomial_boundary_from_coefficients(step.new_coefficients).radius_on_grid(grid)
    summary: dict[str, object] = {
        "accepted": bool(step.accepted),
        "line_search_factor": float(step.line_search_factor),
        "coefficients_initial": [float(value) for value in step.coefficients],
        "coefficients_new": [float(value) for value in step.new_coefficients],
        "raw_step": [float(value) for value in step.raw_step],
        "limited_step": [float(value) for value in step.limited_step],
        "finite_difference_steps": [float(value) for value in step.finite_difference_steps],
        "jacobian_shape": [int(value) for value in step.jacobian.shape],
        "residual_value_before": float(step.residual.value),
        "residual_value_after": float(step.trial_residual.value),
        "predicted_value": float(step.predicted_value),
        "equilibrium_rms_before": float(step.residual.equilibrium_rms),
        "equilibrium_rms_after": float(step.trial_residual.equilibrium_rms),
        "lcfs_value_before": float(step.residual.lcfs_value),
        "lcfs_value_after": float(step.trial_residual.lcfs_value),
        "max_relative_radius_change": float(np.max(np.abs(new_radius - old_radius) / np.maximum(old_radius, 1.0e-300))),
        "trial_rows": trial_rows,
        "coupled_trial": None,
        "figure": None,
    }
    if write_plots:
        summary["figure"] = str(_write_ls_boundary_step_plot(summary, outdir=figure_dir, name=name))
    return summary


def _run_ls_boundary_step(
    *,
    grid,
    boundary,
    coils,
    output,
    final_trace,
    reference_merit,
    finite_difference_step: float,
    damping: float,
    max_relative_step: float,
    ridge: float,
    write_plots: bool,
    figure_dir: Path,
    name: str,
) -> dict[str, object]:
    """Run one diagnostic LS update over polynomial side-boundary coefficients."""

    coefficients = _fit_polynomial_boundary_coefficients(grid, boundary)
    residual_function = _build_ls_boundary_residual_function(
        grid=grid,
        coils=coils,
        output=output,
        final_trace=final_trace,
        reference_merit=reference_merit,
    )
    step = mirror_free_boundary_least_squares_step(
        coefficients,
        residual_function,
        finite_difference_step=finite_difference_step,
        damping=damping,
        max_relative_step=max_relative_step,
        ridge=ridge,
        line_search_factors=CIRCULAR_COIL_BETA_SCAN_LS_LINE_SEARCH_FACTORS,
    )
    return _ls_boundary_step_summary_from_step(
        step=step,
        residual_function=residual_function,
        grid=grid,
        line_search_factors=CIRCULAR_COIL_BETA_SCAN_LS_LINE_SEARCH_FACTORS,
        write_plots=write_plots,
        figure_dir=figure_dir,
        name=name,
    )


def _run_ls_boundary_coupled_trial(
    *,
    step_summary: dict[str, object],
    outdir: Path,
    label: str,
    config,
    grid,
    coils,
    psi_prime_value: float,
    pressure,
    solve_options,
    baseline_final_fsq: float,
    reference_merit,
    write_plots: bool,
) -> dict[str, object]:
    """Evaluate the LS-selected boundary with a realized fixed-boundary solve."""

    if not bool(step_summary["accepted"]):
        return {
            "status": "skipped",
            "mout": None,
            "accepted_by_merit": False,
            "rejection_reason": "ls_step_not_accepted",
            "final_residual_norm": None,
            "final_fsq": None,
            "final_normalized_force": None,
            "fsq_growth_ratio": None,
            "lcfs_external_bnormal_rms": None,
            "lcfs_pressure_balance_rms": None,
            "lcfs_merit": None,
            "lcfs_merit_ratio": None,
            "figures": {},
        }

    trial_boundary = _polynomial_boundary_from_coefficients(step_summary["coefficients_new"])
    result = run_mirror_fixed_boundary(
        config,
        trial_boundary,
        psi_prime=PsiPrimeProfile.constant(float(psi_prime_value)),
        i_prime=IPrimeProfile.zero(),
        pressure=pressure,
        options=solve_options,
    )
    mout_path = outdir / f"mout_free_boundary_circular_coils_beta_{label}_ls_boundary_trial.nc"
    if mout_path.exists():
        mout_path.unlink()
    mout = write_mirror_output(mout_path, result)
    output = load_mirror_output(mout)
    external_sample = sample_mirror_boundary_external_field(grid, trial_boundary, coils)
    lcfs = mirror_lcfs_diagnostic(output, external_sample, mu0=1.0)
    merit = mirror_lcfs_merit(
        lcfs,
        pressure_scale=reference_merit.pressure_scale,
        bnormal_scale=reference_merit.bnormal_scale,
        bnormal_weight=reference_merit.bnormal_weight,
    )
    final = result.final_trace
    merit_ratio = float(merit.value) / max(float(reference_merit.value), 1.0e-300)
    fsq_growth_ratio = float(final.fsq) / max(float(baseline_final_fsq), 1.0e-300)
    accepted_by_merit = merit_ratio <= 1.0
    plot_paths: dict[str, str] = {}
    if write_plots:
        figure_dir = outdir / "figures" / f"fixed_boundary_beta_{label}_ls_boundary_trial"
        plot_paths = {name: str(path) for name, path in plot_mirror_output(mout, outdir=figure_dir).items()}
        plot_paths["lcfs_diagnostic"] = str(
            _write_lcfs_diagnostic_plot(
                lcfs,
                None,
                outdir=figure_dir,
                name=f"free_boundary_circular_coils_beta_{label}_ls_boundary_trial",
            )
        )
    return {
        "status": "accepted" if accepted_by_merit else "rejected",
        "mout": str(mout),
        "accepted_by_merit": bool(accepted_by_merit),
        "rejection_reason": None if accepted_by_merit else "merit_increase",
        "final_residual_norm": float(final.residual_norm),
        "final_fsq": float(final.fsq),
        "final_normalized_force": float(final.normalized_force),
        "fsq_growth_ratio": fsq_growth_ratio,
        "lcfs_external_bnormal_rms": float(lcfs.external_bnormal_rms),
        "lcfs_pressure_balance_rms": float(lcfs.pressure_balance_rms),
        "lcfs_merit": float(merit.value),
        "lcfs_merit_ratio": merit_ratio,
        "figures": plot_paths,
    }


def _run_ls_boundary_coupled_loop(
    *,
    outdir: Path,
    label: str,
    config,
    grid,
    initial_boundary,
    initial_output,
    initial_final,
    coils,
    psi_prime_value: float,
    pressure,
    solve_options,
    baseline_final_fsq: float,
    reference_merit,
    max_steps: int,
    target_merit: float,
    stagnation_rtol: float,
    fsq_growth_limit: float,
    finite_difference_step: float,
    damping: float,
    max_relative_step: float,
    ridge: float,
    write_plots: bool,
) -> list[dict[str, object]]:
    """Run a guarded multi-step coupled LS boundary loop."""

    def payload_for(boundary, output, final_trace):
        return SimpleNamespace(boundary=boundary, output=output, final=final_trace)

    def residual_for_state(state, coefficients):
        payload = state.payload
        residual_function = _build_ls_boundary_residual_function(
            grid=grid,
            coils=coils,
            output=payload.output,
            final_trace=payload.final,
            reference_merit=reference_merit,
        )
        return residual_function(coefficients)

    def step_summary_for(step_index: int, state, step):
        step_label = f"{label}_ls_loop_step_{step_index}"
        residual_function = _build_ls_boundary_residual_function(
            grid=grid,
            coils=coils,
            output=state.payload.output,
            final_trace=state.payload.final,
            reference_merit=reference_merit,
        )
        return _ls_boundary_step_summary_from_step(
            step=step,
            residual_function=residual_function,
            grid=grid,
            line_search_factors=CIRCULAR_COIL_BETA_SCAN_LS_LINE_SEARCH_FACTORS,
            write_plots=write_plots,
            figure_dir=outdir / "figures" / f"fixed_boundary_beta_{label}_ls_loop_step_{step_index}",
            name=f"free_boundary_circular_coils_beta_{step_label}",
        )

    step_summaries: dict[int, dict[str, object]] = {}

    def trial_for_state(state, step):
        step_index = len(step_summaries) + 1
        step_label = f"{label}_ls_loop_step_{step_index}"
        step_summary = step_summary_for(step_index, state, step)
        trial = _run_ls_boundary_coupled_trial(
            step_summary=step_summary,
            outdir=outdir,
            label=step_label,
            config=config,
            grid=grid,
            coils=coils,
            psi_prime_value=psi_prime_value,
            pressure=pressure,
            solve_options=solve_options,
            baseline_final_fsq=baseline_final_fsq,
            reference_merit=reference_merit,
            write_plots=write_plots,
        )
        step_summary["coupled_trial"] = trial
        step_summaries[step_index] = step_summary
        trial_boundary = _polynomial_boundary_from_coefficients(step_summary["coefficients_new"])
        trial_output = load_mirror_output(str(trial["mout"]))
        trial_final = SimpleNamespace(normalized_force=float(trial["final_normalized_force"]))
        return MirrorFreeBoundaryLoopState(
            coefficients=step.new_coefficients,
            residual=step.trial_residual,
            merit=float(trial["lcfs_merit"]),
            equilibrium_value=float(trial["final_fsq"]),
            payload=payload_for(trial_boundary, trial_output, trial_final),
        )

    initial_coefficients = _fit_polynomial_boundary_coefficients(grid, initial_boundary)
    initial_residual_function = _build_ls_boundary_residual_function(
        grid=grid,
        coils=coils,
        output=initial_output,
        final_trace=initial_final,
        reference_merit=reference_merit,
    )
    initial_residual = initial_residual_function(initial_coefficients)
    initial_state = MirrorFreeBoundaryLoopState(
        coefficients=initial_coefficients,
        residual=initial_residual,
        merit=float(reference_merit.value),
        equilibrium_value=float(baseline_final_fsq),
        payload=payload_for(initial_boundary, initial_output, initial_final),
    )
    loop = mirror_free_boundary_guarded_least_squares_loop(
        initial_state,
        residual_for_state,
        trial_function=trial_for_state,
        max_steps=max_steps,
        target_merit=target_merit,
        stagnation_rtol=stagnation_rtol,
        equilibrium_growth_limit=fsq_growth_limit,
        equilibrium_growth_reference=baseline_final_fsq,
        finite_difference_step=finite_difference_step,
        damping=damping,
        max_relative_step=max_relative_step,
        ridge=ridge,
        line_search_factors=CIRCULAR_COIL_BETA_SCAN_LS_LINE_SEARCH_FACTORS,
    )

    rows: list[dict[str, object]] = []
    for loop_row in loop.rows:
        step_summary = step_summaries.get(loop_row.step)
        if step_summary is None:
            step_summary = step_summary_for(loop_row.step, loop_row.state_before, loop_row.ls_step)
            step_summaries[loop_row.step] = step_summary
        trial = step_summary.get("coupled_trial")
        rejection_reason = loop_row.rejection_reason
        stop_reason = loop_row.stop_reason
        if rejection_reason == "equilibrium_growth_guard":
            rejection_reason = "fsq_growth_guard"
        if stop_reason == "equilibrium_growth_guard":
            stop_reason = "fsq_growth_guard"
        row: dict[str, object] = {
            "step": int(loop_row.step),
            "status": str(loop_row.status),
            "accepted": bool(loop_row.accepted),
            "rejection_reason": rejection_reason,
            "stop_reason": stop_reason,
            "merit_improvement_fraction": loop_row.merit_improvement_fraction if loop_row.accepted else None,
            "fsq_growth_ratio": None if trial is None else trial["fsq_growth_ratio"],
            "final_fsq": None if trial is None else trial["final_fsq"],
            "final_normalized_force": None if trial is None else trial["final_normalized_force"],
            "lcfs_merit": None if trial is None else trial["lcfs_merit"],
            "lcfs_merit_ratio": None if trial is None else trial["lcfs_merit_ratio"],
            "mout": None if trial is None else trial["mout"],
            "ls_boundary_step": step_summary,
            "figures": {} if trial is None else trial["figures"],
        }
        rows.append(row)
    return rows


def _beta_label(beta_percent: float) -> str:
    return f"{float(beta_percent):g}".replace(".", "p")


def _lcfs_pilot_summary(pilot_rows: list[dict[str, object]]) -> dict[str, object]:
    """Return scalar status fields for a beta-case LCFS pilot sequence."""
    if not pilot_rows:
        return {
            "lcfs_pilot_status": "not_requested",
            "lcfs_pilot_rows_count": 0,
            "lcfs_pilot_accepted_rows": 0,
            "lcfs_pilot_skipped_rows": 0,
            "lcfs_pilot_final_merit": None,
            "lcfs_pilot_best_merit": None,
            "lcfs_pilot_final_pressure_balance_rms": None,
            "lcfs_pilot_final_fsq": None,
            "lcfs_pilot_best_fsq": None,
            "lcfs_pilot_final_fsq_growth_ratio": None,
            "lcfs_pilot_best_fsq_growth_ratio": None,
            "lcfs_pilot_final_normalized_force": None,
            "lcfs_pilot_stop_reason": None,
            "lcfs_pilot_last_accepted_step": None,
            "lcfs_pilot_last_accepted_merit": None,
            "lcfs_pilot_last_accepted_pressure_balance_rms": None,
            "lcfs_pilot_last_accepted_fsq": None,
            "lcfs_pilot_last_accepted_fsq_growth_ratio": None,
            "lcfs_pilot_last_accepted_normalized_force": None,
        }
    accepted = sum(bool(row.get("accepted", False)) and not bool(row.get("skipped", False)) for row in pilot_rows)
    skipped = sum(bool(row.get("skipped", False)) for row in pilot_rows)
    accepted_rows = [
        row for row in pilot_rows if bool(row.get("accepted", False)) and not bool(row.get("skipped", False))
    ]
    merit_values = [float(row["lcfs_merit"]) for row in pilot_rows if row.get("lcfs_merit") is not None]
    fsq_values = [float(row["final_fsq"]) for row in pilot_rows if row.get("final_fsq") is not None]
    fsq_growth_ratios = [
        float(row["fsq_growth_ratio"]) for row in pilot_rows if row.get("fsq_growth_ratio") is not None
    ]
    final = pilot_rows[-1]
    if bool(final.get("skipped", False)):
        status = "skipped"
    elif bool(final.get("accepted", False)):
        status = "accepted"
    else:
        status = "rejected"
    last_accepted = accepted_rows[-1] if accepted_rows else None
    return {
        "lcfs_pilot_status": status,
        "lcfs_pilot_rows_count": len(pilot_rows),
        "lcfs_pilot_accepted_rows": int(accepted),
        "lcfs_pilot_skipped_rows": int(skipped),
        "lcfs_pilot_final_merit": None if final.get("lcfs_merit") is None else float(final["lcfs_merit"]),
        "lcfs_pilot_best_merit": None if not merit_values else float(min(merit_values)),
        "lcfs_pilot_final_pressure_balance_rms": None
        if final.get("lcfs_pressure_balance_rms") is None
        else float(final["lcfs_pressure_balance_rms"]),
        "lcfs_pilot_final_fsq": None if final.get("final_fsq") is None else float(final["final_fsq"]),
        "lcfs_pilot_best_fsq": None if not fsq_values else float(min(fsq_values)),
        "lcfs_pilot_final_fsq_growth_ratio": None
        if final.get("fsq_growth_ratio") is None
        else float(final["fsq_growth_ratio"]),
        "lcfs_pilot_best_fsq_growth_ratio": None if not fsq_growth_ratios else float(min(fsq_growth_ratios)),
        "lcfs_pilot_final_normalized_force": None
        if final.get("final_normalized_force") is None
        else float(final["final_normalized_force"]),
        "lcfs_pilot_stop_reason": final.get("stop_reason"),
        "lcfs_pilot_last_accepted_step": None if last_accepted is None else int(last_accepted["step"]),
        "lcfs_pilot_last_accepted_merit": None
        if last_accepted is None or last_accepted.get("lcfs_merit") is None
        else float(last_accepted["lcfs_merit"]),
        "lcfs_pilot_last_accepted_pressure_balance_rms": None
        if last_accepted is None or last_accepted.get("lcfs_pressure_balance_rms") is None
        else float(last_accepted["lcfs_pressure_balance_rms"]),
        "lcfs_pilot_last_accepted_fsq": None
        if last_accepted is None or last_accepted.get("final_fsq") is None
        else float(last_accepted["final_fsq"]),
        "lcfs_pilot_last_accepted_fsq_growth_ratio": None
        if last_accepted is None or last_accepted.get("fsq_growth_ratio") is None
        else float(last_accepted["fsq_growth_ratio"]),
        "lcfs_pilot_last_accepted_normalized_force": None
        if last_accepted is None or last_accepted.get("final_normalized_force") is None
        else float(last_accepted["final_normalized_force"]),
    }


def _ls_boundary_coupled_loop_summary(loop_rows: list[dict[str, object]]) -> dict[str, object]:
    """Return scalar status fields for a beta-case coupled LS loop."""

    if not loop_rows:
        return {
            "ls_boundary_coupled_loop_status": "not_requested",
            "ls_boundary_coupled_loop_rows_count": 0,
            "ls_boundary_coupled_loop_accepted_rows": 0,
            "ls_boundary_coupled_loop_stop_reason": None,
            "ls_boundary_coupled_loop_final_merit": None,
            "ls_boundary_coupled_loop_final_fsq_growth_ratio": None,
            "ls_boundary_coupled_loop_last_accepted_step": None,
            "ls_boundary_coupled_loop_last_accepted_merit": None,
            "ls_boundary_coupled_loop_last_accepted_fsq_growth_ratio": None,
        }

    accepted_rows = [row for row in loop_rows if bool(row.get("accepted", False))]
    final = loop_rows[-1]
    if bool(final.get("accepted", False)):
        status = "accepted"
    elif final.get("rejection_reason") == "ls_step_not_accepted":
        status = "skipped"
    else:
        status = "rejected"
    last_accepted = accepted_rows[-1] if accepted_rows else None
    return {
        "ls_boundary_coupled_loop_status": status,
        "ls_boundary_coupled_loop_rows_count": len(loop_rows),
        "ls_boundary_coupled_loop_accepted_rows": len(accepted_rows),
        "ls_boundary_coupled_loop_stop_reason": final.get("stop_reason"),
        "ls_boundary_coupled_loop_final_merit": None if final.get("lcfs_merit") is None else float(final["lcfs_merit"]),
        "ls_boundary_coupled_loop_final_fsq_growth_ratio": None
        if final.get("fsq_growth_ratio") is None
        else float(final["fsq_growth_ratio"]),
        "ls_boundary_coupled_loop_last_accepted_step": None if last_accepted is None else int(last_accepted["step"]),
        "ls_boundary_coupled_loop_last_accepted_merit": None
        if last_accepted is None or last_accepted.get("lcfs_merit") is None
        else float(last_accepted["lcfs_merit"]),
        "ls_boundary_coupled_loop_last_accepted_fsq_growth_ratio": None
        if last_accepted is None or last_accepted.get("fsq_growth_ratio") is None
        else float(last_accepted["fsq_growth_ratio"]),
    }


def _counts_json(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


def _beta_scan_summary(
    baseline_rows: list[dict[str, object]],
    *,
    run_fixed_boundary_baseline: bool,
    run_lcfs_pilot: bool,
    lcfs_pilot_steps: int,
    lcfs_pilot_target_merit: float,
    lcfs_pilot_stagnation_rtol: float,
    lcfs_pilot_fsq_growth_limit: float,
    run_ls_boundary_step: bool,
    run_ls_boundary_coupled_trial: bool,
    run_ls_boundary_coupled_loop: bool,
    ls_boundary_coupled_loop_steps: int,
    ls_boundary_coupled_loop_target_merit: float,
    ls_boundary_coupled_loop_stagnation_rtol: float,
    ls_boundary_coupled_loop_fsq_growth_limit: float,
    ls_boundary_finite_difference_step: float,
    ls_boundary_damping: float,
    ls_boundary_max_relative_step: float,
    ls_boundary_ridge: float,
) -> dict[str, object]:
    """Return top-level status fields for the circular-coil beta scan."""
    pilot_rows = [pilot for row in baseline_rows for pilot in row.get("lcfs_pilot_rows", [])]
    ls_rows = [row for row in baseline_rows if row.get("ls_boundary_step") is not None]
    ls_trial_rows = [
        row["ls_boundary_step"]["coupled_trial"]
        for row in ls_rows
        if row["ls_boundary_step"].get("coupled_trial") is not None
    ]
    ls_loop_rows = [loop_row for row in baseline_rows for loop_row in row.get("ls_boundary_coupled_loop_rows", [])]
    if run_lcfs_pilot:
        workflow_status = "lcfs_pilot"
    elif run_ls_boundary_coupled_loop:
        workflow_status = "ls_boundary_coupled_loop"
    elif run_fixed_boundary_baseline:
        workflow_status = "fixed_boundary_baseline"
    else:
        workflow_status = "setup_only"
    if run_lcfs_pilot:
        free_boundary_status = "lcfs_pilot_not_converged_free_boundary"
    elif run_ls_boundary_coupled_loop:
        free_boundary_status = "ls_boundary_coupled_loop_not_converged_free_boundary"
    else:
        free_boundary_status = "not_run"
    return {
        "workflow_status": workflow_status,
        "free_boundary_solve_status": free_boundary_status,
        "external_field_provider_kind": "direct_coils",
        "coil_format": "essos_compatible_circular_fourier",
        "fixed_boundary_baseline_count": len(baseline_rows),
        "lcfs_pilot_requested": bool(run_lcfs_pilot),
        "lcfs_pilot_steps_requested": int(lcfs_pilot_steps) if run_lcfs_pilot else 0,
        "lcfs_pilot_target_merit": float(lcfs_pilot_target_merit) if run_lcfs_pilot else None,
        "lcfs_pilot_stagnation_rtol": float(lcfs_pilot_stagnation_rtol) if run_lcfs_pilot else None,
        "lcfs_pilot_fsq_growth_limit": float(lcfs_pilot_fsq_growth_limit) if run_lcfs_pilot else None,
        "lcfs_pilot_rows_total": len(pilot_rows),
        "lcfs_pilot_accepted_rows_total": sum(
            bool(row.get("accepted", False)) and not bool(row.get("skipped", False)) for row in pilot_rows
        ),
        "lcfs_pilot_skipped_rows_total": sum(bool(row.get("skipped", False)) for row in pilot_rows),
        "lcfs_pilot_stop_reason_counts": _counts_json([str(row.get("stop_reason")) for row in pilot_rows]),
        "ls_boundary_step_requested": bool(run_ls_boundary_step),
        "ls_boundary_finite_difference_step": float(ls_boundary_finite_difference_step)
        if run_ls_boundary_step
        else None,
        "ls_boundary_damping": float(ls_boundary_damping) if run_ls_boundary_step else None,
        "ls_boundary_max_relative_step": float(ls_boundary_max_relative_step) if run_ls_boundary_step else None,
        "ls_boundary_ridge": float(ls_boundary_ridge) if run_ls_boundary_step else None,
        "ls_boundary_step_rows_total": len(ls_rows),
        "ls_boundary_coupled_trial_requested": bool(run_ls_boundary_coupled_trial),
        "ls_boundary_coupled_trial_rows_total": len(ls_trial_rows),
        "ls_boundary_coupled_loop_requested": bool(run_ls_boundary_coupled_loop),
        "ls_boundary_coupled_loop_steps_requested": int(ls_boundary_coupled_loop_steps)
        if run_ls_boundary_coupled_loop
        else 0,
        "ls_boundary_coupled_loop_target_merit": float(ls_boundary_coupled_loop_target_merit)
        if run_ls_boundary_coupled_loop
        else None,
        "ls_boundary_coupled_loop_stagnation_rtol": float(ls_boundary_coupled_loop_stagnation_rtol)
        if run_ls_boundary_coupled_loop
        else None,
        "ls_boundary_coupled_loop_fsq_growth_limit": float(ls_boundary_coupled_loop_fsq_growth_limit)
        if run_ls_boundary_coupled_loop
        else None,
        "ls_boundary_coupled_loop_rows_total": len(ls_loop_rows),
        "ls_boundary_coupled_loop_accepted_rows_total": sum(bool(row.get("accepted", False)) for row in ls_loop_rows),
        "ls_boundary_coupled_loop_stop_reason_counts": _counts_json(
            [str(row.get("stop_reason")) for row in ls_loop_rows]
        ),
    }


def _proposal_predicted_metrics(proposal, *, grid, coils, baseline_merit) -> dict[str, object]:
    sample = sample_mirror_boundary_external_field(grid, proposal.boundary, coils)
    boundary_r = proposal.boundary.radius_on_grid_3d(grid)
    bnormal = mirror_external_bnormal(boundary_r, grid.z, sample)
    bnormal_rms = float(np.sqrt(np.mean(np.asarray(bnormal, dtype=float) ** 2)))
    pressure_rms = float(proposal.pressure_balance_rms_predicted)
    pressure_term = pressure_rms / max(float(baseline_merit.pressure_scale), 1.0e-300)
    bnormal_term = bnormal_rms / max(float(baseline_merit.bnormal_scale), 1.0e-300)
    merit = float(np.sqrt(pressure_term**2 + float(baseline_merit.bnormal_weight) * bnormal_term**2))
    return {
        "strategy": str(proposal.strategy),
        "predicted_merit": merit,
        "predicted_pressure_balance_rms": pressure_rms,
        "predicted_external_bnormal_rms": bnormal_rms,
        "max_relative_delta_radius": float(
            np.max(np.abs(proposal.delta_radius) / np.maximum(proposal.old_radius, 1.0e-300))
        ),
    }


def _select_lcfs_proposal(
    *,
    lcfs,
    pressure_response,
    grid,
    coils,
    external_sample,
    baseline_merit,
    mode: str,
    damping: float,
    max_relative_step: float,
    cap_taper_power: float,
    smoothing_passes: int,
    require_bnormal_nonincrease: bool,
) -> _LCFSProposalSelection:
    candidates = list(
        propose_axisymmetric_mirror_lcfs_candidate_set(
            lcfs,
            external_sample,
            pressure_response,
            damping=damping,
            max_relative_step=max_relative_step,
            radius_floor=1.0e-4,
            preserve_caps=True,
            cap_taper_power=cap_taper_power,
            smoothing_passes=smoothing_passes,
            bnormal_weight=baseline_merit.bnormal_weight,
        )
    )
    summaries = [
        _proposal_predicted_metrics(candidate, grid=grid, coils=coils, baseline_merit=baseline_merit)
        for candidate in candidates
    ]
    mode_to_strategy = {
        "local": "local_pressure",
        "scale": "scale_pressure",
        "bnormal": "bnormal_slope",
        "mixed": "mixed_scale_bnormal",
    }
    if mode in mode_to_strategy:
        strategy = mode_to_strategy[mode]
        proposal = next(candidate for candidate in candidates if candidate.strategy == strategy)
        return _LCFSProposalSelection(
            proposal=proposal,
            candidates=tuple(candidates),
            candidate_summaries=summaries,
            allowed_strategies=tuple(summary["strategy"] for summary in summaries),
            rejection_reason=None,
        )
    allowed = np.ones(len(summaries), dtype=bool)
    if require_bnormal_nonincrease:
        baseline_bnormal = float(baseline_merit.external_bnormal_rms)
        allowed = np.asarray(
            [
                summary["strategy"] == "noop" or summary["predicted_external_bnormal_rms"] <= baseline_bnormal + 1.0e-14
                for summary in summaries
            ],
            dtype=bool,
        )
    merit_values = [summary["predicted_merit"] if allowed[index] else np.inf for index, summary in enumerate(summaries)]
    best_index = int(np.argmin(merit_values))
    allowed_strategies = tuple(
        str(summary["strategy"]) for index, summary in enumerate(summaries) if bool(allowed[index])
    )
    nonzero_allowed = any(strategy != "noop" for strategy in allowed_strategies)
    selected = candidates[best_index]
    rejection_reason = (
        "normal_field_guard_no_candidate"
        if require_bnormal_nonincrease and selected.strategy == "noop" and not nonzero_allowed
        else None
    )
    return _LCFSProposalSelection(
        proposal=selected,
        candidates=tuple(candidates),
        candidate_summaries=summaries,
        allowed_strategies=allowed_strategies,
        rejection_reason=rejection_reason,
    )


def _next_proposal_fields(selection: _LCFSProposalSelection) -> dict[str, object]:
    proposal = selection.proposal
    return {
        "lcfs_update_pressure_balance_rms_predicted_next": float(proposal.pressure_balance_rms_predicted),
        "lcfs_update_strategy_next": str(proposal.strategy),
        "lcfs_update_candidate_summaries_next": selection.candidate_summaries,
        "lcfs_update_allowed_strategies_next": list(selection.allowed_strategies),
        "lcfs_update_rejection_reason_next": selection.rejection_reason,
        "lcfs_update_cap_taper_power_next": float(proposal.cap_taper_power),
        "lcfs_update_smoothing_passes_next": int(proposal.smoothing_passes),
        "lcfs_update_max_relative_delta_radius_next": float(
            np.max(np.abs(proposal.delta_radius) / np.maximum(proposal.old_radius, 1.0e-300))
        ),
    }


def _skipped_lcfs_pilot_row(
    *,
    step: int,
    current_lcfs,
    current_merit,
    selection: _LCFSProposalSelection,
) -> dict[str, object]:
    row = {
        "step": int(step),
        "mout": None,
        "accepted": False,
        "skipped": True,
        "rejection_reason": selection.rejection_reason or "noop_candidate_selected",
        "stop_reason": "noop_candidate",
        "lcfs_merit_improvement_fraction": None,
        "fsq_growth_ratio": None,
        "final_residual_norm": None,
        "final_fsq": None,
        "final_normalized_force": None,
        "min_sqrtg": None,
        "mirror_ratio": None,
        "lcfs_external_bnormal_rms": float(current_lcfs.external_bnormal_rms),
        "lcfs_external_bnormal_max": float(current_lcfs.external_bnormal_max),
        "lcfs_pressure_balance_rms": float(current_lcfs.pressure_balance_rms),
        "lcfs_pressure_balance_max": float(current_lcfs.pressure_balance_max),
        "lcfs_pressure_balance_rms_change_fraction": 0.0,
        "lcfs_merit": float(current_merit.value),
        "lcfs_merit_change_fraction": 0.0,
        "lcfs_merit_bnormal_weight": float(current_merit.bnormal_weight),
        "figures": {},
    }
    row.update(_next_proposal_fields(selection))
    row["lcfs_update_max_relative_delta_radius_next"] = 0.0
    return row


def _completed_lcfs_pilot_row(
    *,
    step: int,
    mout,
    result,
    lcfs,
    merit,
    reference_lcfs,
    reference_merit,
    next_selection: _LCFSProposalSelection,
    accepted: bool,
    figures: dict[str, str],
) -> dict[str, object]:
    final = result.final_trace
    row = {
        "step": int(step),
        "mout": str(mout),
        "accepted": bool(accepted),
        "rejection_reason": None,
        "stop_reason": None,
        "lcfs_merit_improvement_fraction": None,
        "fsq_growth_ratio": None,
        "final_residual_norm": float(final.residual_norm),
        "final_fsq": float(final.fsq),
        "final_normalized_force": float(final.normalized_force),
        "min_sqrtg": float(final.min_sqrtg),
        "mirror_ratio": float(final.mirror_ratio),
        "lcfs_external_bnormal_rms": float(lcfs.external_bnormal_rms),
        "lcfs_external_bnormal_max": float(lcfs.external_bnormal_max),
        "lcfs_pressure_balance_rms": float(lcfs.pressure_balance_rms),
        "lcfs_pressure_balance_max": float(lcfs.pressure_balance_max),
        "lcfs_pressure_balance_rms_change_fraction": float(
            1.0 - lcfs.pressure_balance_rms / max(reference_lcfs.pressure_balance_rms, 1.0e-300)
        ),
        "lcfs_merit": float(merit.value),
        "lcfs_merit_change_fraction": float(1.0 - merit.value / max(reference_merit.value, 1.0e-300)),
        "lcfs_merit_bnormal_weight": float(merit.bnormal_weight),
        "figures": figures,
    }
    row.update(_next_proposal_fields(next_selection))
    return row


def _run_lcfs_pilot_step(
    *,
    step: int,
    outdir: Path,
    label: str,
    config,
    boundary,
    grid,
    coils,
    psi_prime_value: float,
    pressure,
    solve_options,
    reference_lcfs,
    reference_merit,
    accepted_merit_value: float,
    lcfs_merit_bnormal_weight: float,
    lcfs_proposal_mode: str,
    lcfs_update_damping: float,
    lcfs_update_max_relative_step: float,
    lcfs_update_cap_taper_power: float,
    lcfs_update_smoothing_passes: int,
    lcfs_require_bnormal_nonincrease: bool,
    write_plots: bool,
) -> _LCFSPilotStepResult:
    result = run_mirror_fixed_boundary(
        config,
        boundary,
        psi_prime=PsiPrimeProfile.constant(float(psi_prime_value)),
        i_prime=IPrimeProfile.zero(),
        pressure=pressure,
        options=solve_options,
    )
    mout_path = outdir / f"mout_free_boundary_circular_coils_beta_{label}_lcfs_step_{step}.nc"
    if mout_path.exists():
        mout_path.unlink()
    mout = write_mirror_output(mout_path, result)
    output = load_mirror_output(mout)
    external_sample = sample_mirror_boundary_external_field(grid, boundary, coils)
    lcfs = mirror_lcfs_diagnostic(output, external_sample, mu0=1.0)
    merit = mirror_lcfs_merit(
        lcfs,
        pressure_scale=reference_merit.pressure_scale,
        bnormal_scale=reference_merit.bnormal_scale,
        bnormal_weight=lcfs_merit_bnormal_weight,
    )
    pressure_response = mirror_external_pressure_balance_response(lcfs, coils, mu0=1.0)
    selection = _select_lcfs_proposal(
        lcfs=lcfs,
        pressure_response=pressure_response,
        grid=grid,
        coils=coils,
        external_sample=external_sample,
        baseline_merit=reference_merit,
        mode=lcfs_proposal_mode,
        damping=lcfs_update_damping,
        max_relative_step=lcfs_update_max_relative_step,
        cap_taper_power=lcfs_update_cap_taper_power,
        smoothing_passes=lcfs_update_smoothing_passes,
        require_bnormal_nonincrease=lcfs_require_bnormal_nonincrease,
    )
    plot_paths: dict[str, str] = {}
    if write_plots:
        figure_dir = outdir / "figures" / f"fixed_boundary_beta_{label}_lcfs_step_{step}"
        plot_paths = {name: str(path) for name, path in plot_mirror_output(mout, outdir=figure_dir).items()}
        plot_paths["lcfs_diagnostic"] = str(
            _write_lcfs_diagnostic_plot(
                lcfs,
                selection.proposal,
                outdir=figure_dir,
                name=f"free_boundary_circular_coils_beta_{label}_lcfs_step_{step}",
            )
        )
    accepted = bool(merit.value <= accepted_merit_value)
    return _LCFSPilotStepResult(
        row=_completed_lcfs_pilot_row(
            step=step,
            mout=mout,
            result=result,
            lcfs=lcfs,
            merit=merit,
            reference_lcfs=reference_lcfs,
            reference_merit=reference_merit,
            next_selection=selection,
            accepted=accepted,
            figures=plot_paths,
        ),
        lcfs=lcfs,
        merit=merit,
        selection=selection,
        accepted=accepted,
    )


def _strategy_slug(strategy: object) -> str:
    return str(strategy).replace(" ", "_").replace("/", "_")


def _coupled_trial_summary(
    *,
    strategy: object,
    mout: object | None,
    accepted: bool,
    skipped: bool,
    final_fsq: float | None,
    lcfs_merit: float,
    merit_ratio: float,
    fsq_growth_ratio: float,
    fsq_weight: float,
) -> dict[str, object]:
    fsq_penalty = max(float(fsq_growth_ratio) - 1.0, 0.0)
    score = float(merit_ratio) + float(fsq_weight) * fsq_penalty
    return {
        "strategy": str(strategy),
        "mout": None if mout is None else str(mout),
        "selected": False,
        "accepted_by_merit": bool(accepted),
        "skipped": bool(skipped),
        "final_fsq": None if final_fsq is None else float(final_fsq),
        "lcfs_merit": float(lcfs_merit),
        "merit_ratio": float(merit_ratio),
        "fsq_growth_ratio": float(fsq_growth_ratio),
        "fsq_penalty": float(fsq_penalty),
        "score": score,
    }


def _run_lcfs_coupled_pilot_step(
    *,
    step: int,
    outdir: Path,
    label: str,
    config,
    current_lcfs,
    current_merit,
    current_selection: _LCFSProposalSelection,
    grid,
    coils,
    psi_prime_value: float,
    pressure,
    solve_options,
    reference_lcfs,
    reference_merit,
    accepted_merit_value: float,
    baseline_final_fsq: float,
    lcfs_merit_bnormal_weight: float,
    lcfs_update_damping: float,
    lcfs_update_max_relative_step: float,
    lcfs_update_cap_taper_power: float,
    lcfs_update_smoothing_passes: int,
    lcfs_require_bnormal_nonincrease: bool,
    lcfs_coupled_fsq_weight: float,
    write_plots: bool,
) -> _LCFSPilotStepResult:
    allowed = set(current_selection.allowed_strategies)
    trial_summaries = [
        _coupled_trial_summary(
            strategy="noop",
            mout=None,
            accepted=False,
            skipped=True,
            final_fsq=baseline_final_fsq,
            lcfs_merit=float(current_merit.value),
            merit_ratio=1.0,
            fsq_growth_ratio=1.0,
            fsq_weight=lcfs_coupled_fsq_weight,
        )
    ]
    best_key = (1.0, 1.0, 1.0, 0)
    best_result: _LCFSPilotStepResult | None = None
    best_summary_index = 0
    for order, candidate in enumerate(current_selection.candidates, start=1):
        strategy = str(candidate.strategy)
        if strategy == "noop" or strategy not in allowed:
            continue
        trial = _run_lcfs_pilot_step(
            step=step,
            outdir=outdir,
            label=f"{label}_coupled_{_strategy_slug(strategy)}",
            config=config,
            boundary=candidate.boundary,
            grid=grid,
            coils=coils,
            psi_prime_value=psi_prime_value,
            pressure=pressure,
            solve_options=solve_options,
            reference_lcfs=reference_lcfs,
            reference_merit=reference_merit,
            accepted_merit_value=accepted_merit_value,
            lcfs_merit_bnormal_weight=lcfs_merit_bnormal_weight,
            lcfs_proposal_mode="best_predicted",
            lcfs_update_damping=lcfs_update_damping,
            lcfs_update_max_relative_step=lcfs_update_max_relative_step,
            lcfs_update_cap_taper_power=lcfs_update_cap_taper_power,
            lcfs_update_smoothing_passes=lcfs_update_smoothing_passes,
            lcfs_require_bnormal_nonincrease=lcfs_require_bnormal_nonincrease,
            write_plots=False,
        )
        fsq_growth_ratio = float(trial.row["final_fsq"]) / max(float(baseline_final_fsq), 1.0e-300)
        merit_ratio = float(trial.merit.value) / max(float(accepted_merit_value), 1.0e-300)
        summary = _coupled_trial_summary(
            strategy=strategy,
            mout=trial.row["mout"],
            accepted=trial.accepted,
            skipped=False,
            final_fsq=float(trial.row["final_fsq"]),
            lcfs_merit=float(trial.merit.value),
            merit_ratio=merit_ratio,
            fsq_growth_ratio=fsq_growth_ratio,
            fsq_weight=lcfs_coupled_fsq_weight,
        )
        trial_summaries.append(summary)
        key = (float(summary["score"]), merit_ratio, fsq_growth_ratio, order)
        if key < best_key:
            best_key = key
            best_result = trial
            best_summary_index = len(trial_summaries) - 1

    trial_summaries[best_summary_index]["selected"] = True
    if best_result is None:
        row = _skipped_lcfs_pilot_row(
            step=step,
            current_lcfs=current_lcfs,
            current_merit=current_merit,
            selection=current_selection,
        )
        row["coupled_trial_rows"] = trial_summaries
        row["coupled_score"] = 1.0
        row["coupled_merit_ratio"] = 1.0
        row["coupled_fsq_penalty"] = 0.0
        row["coupled_fsq_weight"] = float(lcfs_coupled_fsq_weight)
        return _LCFSPilotStepResult(
            row=row,
            lcfs=current_lcfs,
            merit=current_merit,
            selection=current_selection,
            accepted=False,
        )

    selected_summary = trial_summaries[best_summary_index]
    best_result.row["fsq_growth_ratio"] = selected_summary["fsq_growth_ratio"]
    best_result.row["coupled_trial_rows"] = trial_summaries
    best_result.row["coupled_score"] = selected_summary["score"]
    best_result.row["coupled_merit_ratio"] = selected_summary["merit_ratio"]
    best_result.row["coupled_fsq_penalty"] = selected_summary["fsq_penalty"]
    best_result.row["coupled_fsq_weight"] = float(lcfs_coupled_fsq_weight)
    if write_plots:
        figure_dir = outdir / "figures" / f"fixed_boundary_beta_{label}_lcfs_step_{step}"
        plot_paths = {
            name: str(path) for name, path in plot_mirror_output(best_result.row["mout"], outdir=figure_dir).items()
        }
        plot_paths["lcfs_diagnostic"] = str(
            _write_lcfs_diagnostic_plot(
                best_result.lcfs,
                best_result.selection.proposal,
                outdir=figure_dir,
                name=f"free_boundary_circular_coils_beta_{label}_lcfs_step_{step}",
            )
        )
        best_result.row["figures"] = plot_paths
    return best_result


def _run_fixed_boundary_baseline_cases(
    *,
    outdir: Path,
    grid,
    boundary,
    scan,
    maxiter: int,
    psi_prime_value: float,
    lcfs_update_damping: float,
    lcfs_update_max_relative_step: float,
    lcfs_update_cap_taper_power: float,
    lcfs_update_smoothing_passes: int,
    lcfs_merit_bnormal_weight: float,
    lcfs_proposal_mode: str,
    lcfs_coupled_fsq_weight: float,
    lcfs_require_bnormal_nonincrease: bool,
    run_lcfs_pilot: bool,
    lcfs_pilot_steps: int,
    lcfs_pilot_target_merit: float,
    lcfs_pilot_stagnation_rtol: float,
    lcfs_pilot_fsq_growth_limit: float,
    run_ls_boundary_step: bool,
    run_ls_boundary_coupled_trial: bool,
    run_ls_boundary_coupled_loop: bool,
    ls_boundary_coupled_loop_steps: int,
    ls_boundary_coupled_loop_target_merit: float,
    ls_boundary_coupled_loop_stagnation_rtol: float,
    ls_boundary_coupled_loop_fsq_growth_limit: float,
    ls_boundary_finite_difference_step: float,
    ls_boundary_damping: float,
    ls_boundary_max_relative_step: float,
    ls_boundary_ridge: float,
    write_plots: bool,
) -> list[dict[str, object]]:
    config = MirrorConfig(
        MirrorResolution(ns=grid.ns, ntheta=1, nxi=grid.nxi, mpol=0),
        z_min=float(grid.z[0]),
        z_max=float(grid.z[-1]),
    )
    baseline_grid = make_mirror_grid(
        ns=grid.ns,
        ntheta=1,
        nxi=grid.nxi,
        mpol=0,
        z_min=float(grid.z[0]),
        z_max=float(grid.z[-1]),
    )
    external_sample = sample_mirror_boundary_external_field(baseline_grid, boundary, scan.coils)
    rows = []
    for case in scan.beta_cases:
        label = _beta_label(case.beta_percent)
        pressure = PressureProfile.polynomial([case.pressure_scale, -case.pressure_scale], gamma=2.0)
        solve_options = MirrorSolveOptions(optimizer="lbfgs", maxiter=int(maxiter), tolerance=1.0e-10, mu0=1.0)
        result = run_mirror_fixed_boundary(
            config,
            boundary,
            psi_prime=PsiPrimeProfile.constant(float(psi_prime_value)),
            i_prime=IPrimeProfile.zero(),
            pressure=pressure,
            options=solve_options,
        )
        final = result.final_trace
        baseline_final_fsq = float(final.fsq)
        mout_path = outdir / f"mout_free_boundary_circular_coils_beta_{label}.nc"
        if mout_path.exists():
            mout_path.unlink()
        mout = write_mirror_output(mout_path, result)
        output = load_mirror_output(mout)
        lcfs = mirror_lcfs_diagnostic(output, external_sample, mu0=1.0)
        lcfs_merit = mirror_lcfs_merit(lcfs, bnormal_weight=lcfs_merit_bnormal_weight)
        pressure_response = mirror_external_pressure_balance_response(lcfs, scan.coils, mu0=1.0)
        proposal_selection = _select_lcfs_proposal(
            lcfs=lcfs,
            pressure_response=pressure_response,
            grid=baseline_grid,
            coils=scan.coils,
            external_sample=external_sample,
            baseline_merit=lcfs_merit,
            mode=lcfs_proposal_mode,
            damping=lcfs_update_damping,
            max_relative_step=lcfs_update_max_relative_step,
            cap_taper_power=lcfs_update_cap_taper_power,
            smoothing_passes=lcfs_update_smoothing_passes,
            require_bnormal_nonincrease=lcfs_require_bnormal_nonincrease,
        )
        proposal = proposal_selection.proposal
        pilot_rows: list[dict[str, object]] = []
        accepted_merit_value = float(lcfs_merit.value)
        current_lcfs = lcfs
        current_merit = lcfs_merit
        candidate_proposal = proposal
        candidate_boundary = proposal.boundary
        candidate_selection = proposal_selection
        if run_lcfs_pilot:
            for step in range(1, int(lcfs_pilot_steps) + 1):
                if lcfs_proposal_mode != "coupled" and candidate_proposal.strategy == "noop":
                    pilot_rows.append(
                        _skipped_lcfs_pilot_row(
                            step=step,
                            current_lcfs=current_lcfs,
                            current_merit=current_merit,
                            selection=candidate_selection,
                        )
                    )
                    break
                if lcfs_proposal_mode == "coupled":
                    pilot_step = _run_lcfs_coupled_pilot_step(
                        step=step,
                        outdir=outdir,
                        label=label,
                        config=config,
                        current_lcfs=current_lcfs,
                        current_merit=current_merit,
                        current_selection=candidate_selection,
                        grid=baseline_grid,
                        coils=scan.coils,
                        psi_prime_value=psi_prime_value,
                        pressure=pressure,
                        solve_options=solve_options,
                        reference_lcfs=lcfs,
                        reference_merit=lcfs_merit,
                        accepted_merit_value=accepted_merit_value,
                        baseline_final_fsq=baseline_final_fsq,
                        lcfs_merit_bnormal_weight=lcfs_merit_bnormal_weight,
                        lcfs_update_damping=lcfs_update_damping,
                        lcfs_update_max_relative_step=lcfs_update_max_relative_step,
                        lcfs_update_cap_taper_power=lcfs_update_cap_taper_power,
                        lcfs_update_smoothing_passes=lcfs_update_smoothing_passes,
                        lcfs_require_bnormal_nonincrease=lcfs_require_bnormal_nonincrease,
                        lcfs_coupled_fsq_weight=lcfs_coupled_fsq_weight,
                        write_plots=write_plots,
                    )
                else:
                    pilot_step = _run_lcfs_pilot_step(
                        step=step,
                        outdir=outdir,
                        label=label,
                        config=config,
                        boundary=candidate_boundary,
                        grid=baseline_grid,
                        coils=scan.coils,
                        psi_prime_value=psi_prime_value,
                        pressure=pressure,
                        solve_options=solve_options,
                        reference_lcfs=lcfs,
                        reference_merit=lcfs_merit,
                        accepted_merit_value=accepted_merit_value,
                        lcfs_merit_bnormal_weight=lcfs_merit_bnormal_weight,
                        lcfs_proposal_mode=lcfs_proposal_mode,
                        lcfs_update_damping=lcfs_update_damping,
                        lcfs_update_max_relative_step=lcfs_update_max_relative_step,
                        lcfs_update_cap_taper_power=lcfs_update_cap_taper_power,
                        lcfs_update_smoothing_passes=lcfs_update_smoothing_passes,
                        lcfs_require_bnormal_nonincrease=lcfs_require_bnormal_nonincrease,
                        write_plots=write_plots,
                    )
                pilot_rows.append(pilot_step.row)
                if bool(pilot_step.row.get("skipped", False)):
                    break
                fsq_growth_ratio = float(pilot_step.row["final_fsq"]) / max(baseline_final_fsq, 1.0e-300)
                pilot_rows[-1]["fsq_growth_ratio"] = fsq_growth_ratio
                if not pilot_step.accepted:
                    pilot_rows[-1]["rejection_reason"] = "merit_increase"
                    pilot_rows[-1]["stop_reason"] = "rejected_merit_increase"
                    break
                if float(lcfs_pilot_fsq_growth_limit) > 0.0 and fsq_growth_ratio > float(lcfs_pilot_fsq_growth_limit):
                    pilot_rows[-1]["accepted"] = False
                    pilot_rows[-1]["rejection_reason"] = "fsq_growth_guard"
                    pilot_rows[-1]["stop_reason"] = "fsq_growth_guard"
                    break
                merit_improvement_fraction = float(1.0 - pilot_step.merit.value / max(accepted_merit_value, 1.0e-300))
                pilot_rows[-1]["lcfs_merit_improvement_fraction"] = merit_improvement_fraction
                if pilot_step.merit.value <= float(lcfs_pilot_target_merit):
                    pilot_rows[-1]["stop_reason"] = "target_merit"
                    break
                if merit_improvement_fraction <= float(lcfs_pilot_stagnation_rtol):
                    pilot_rows[-1]["stop_reason"] = "merit_stagnation"
                    break
                if step == int(lcfs_pilot_steps):
                    pilot_rows[-1]["stop_reason"] = "max_steps"
                accepted_merit_value = float(pilot_step.merit.value)
                current_lcfs = pilot_step.lcfs
                current_merit = pilot_step.merit
                candidate_proposal = pilot_step.selection.proposal
                candidate_boundary = candidate_proposal.boundary
                candidate_selection = pilot_step.selection
        plot_paths: dict[str, str] = {}
        if write_plots:
            figure_dir = outdir / "figures" / f"fixed_boundary_beta_{label}"
            plot_paths = {name: str(path) for name, path in plot_mirror_output(mout, outdir=figure_dir).items()}
            plot_paths["lcfs_diagnostic"] = str(
                _write_lcfs_diagnostic_plot(
                    lcfs,
                    proposal,
                    outdir=figure_dir,
                    name=f"free_boundary_circular_coils_beta_{label}",
                )
            )
        ls_boundary_step = None
        if run_ls_boundary_step:
            ls_boundary_step = _run_ls_boundary_step(
                grid=baseline_grid,
                boundary=boundary,
                coils=scan.coils,
                output=output,
                final_trace=final,
                reference_merit=lcfs_merit,
                finite_difference_step=ls_boundary_finite_difference_step,
                damping=ls_boundary_damping,
                max_relative_step=ls_boundary_max_relative_step,
                ridge=ls_boundary_ridge,
                write_plots=write_plots,
                figure_dir=outdir / "figures" / f"fixed_boundary_beta_{label}",
                name=f"free_boundary_circular_coils_beta_{label}",
            )
            if run_ls_boundary_coupled_trial:
                ls_boundary_step["coupled_trial"] = _run_ls_boundary_coupled_trial(
                    step_summary=ls_boundary_step,
                    outdir=outdir,
                    label=label,
                    config=config,
                    grid=baseline_grid,
                    coils=scan.coils,
                    psi_prime_value=psi_prime_value,
                    pressure=pressure,
                    solve_options=solve_options,
                    baseline_final_fsq=baseline_final_fsq,
                    reference_merit=lcfs_merit,
                    write_plots=write_plots,
                )
            if ls_boundary_step["figure"] is not None:
                plot_paths["ls_boundary_step"] = str(ls_boundary_step["figure"])
        ls_boundary_coupled_loop_rows: list[dict[str, object]] = []
        if run_ls_boundary_coupled_loop:
            ls_boundary_coupled_loop_rows = _run_ls_boundary_coupled_loop(
                outdir=outdir,
                label=label,
                config=config,
                grid=baseline_grid,
                initial_boundary=boundary,
                initial_output=output,
                initial_final=final,
                coils=scan.coils,
                psi_prime_value=psi_prime_value,
                pressure=pressure,
                solve_options=solve_options,
                baseline_final_fsq=baseline_final_fsq,
                reference_merit=lcfs_merit,
                max_steps=ls_boundary_coupled_loop_steps,
                target_merit=ls_boundary_coupled_loop_target_merit,
                stagnation_rtol=ls_boundary_coupled_loop_stagnation_rtol,
                fsq_growth_limit=ls_boundary_coupled_loop_fsq_growth_limit,
                finite_difference_step=ls_boundary_finite_difference_step,
                damping=ls_boundary_damping,
                max_relative_step=ls_boundary_max_relative_step,
                ridge=ls_boundary_ridge,
                write_plots=write_plots,
            )
        summary = result.optimizer_summaries[-1] if result.optimizer_summaries else None
        row = {
            "beta_percent": float(case.beta_percent),
            "beta_fraction": float(case.beta_fraction),
            "pressure_scale": float(case.pressure_scale),
            "mout": str(mout),
            "optimizer": str(summary.optimizer if summary is not None else "lbfgs"),
            "optimizer_success": bool(summary.success) if summary is not None else False,
            "optimizer_nit": int(summary.nit) if summary is not None else 0,
            "final_residual_norm": float(final.residual_norm),
            "final_fsq": float(final.fsq),
            "final_normalized_force": float(final.normalized_force),
            "min_sqrtg": float(final.min_sqrtg),
            "mirror_ratio": float(final.mirror_ratio),
            "lcfs_external_bnormal_rms": float(lcfs.external_bnormal_rms),
            "lcfs_external_bnormal_max": float(lcfs.external_bnormal_max),
            "lcfs_pressure_balance_rms": float(lcfs.pressure_balance_rms),
            "lcfs_pressure_balance_max": float(lcfs.pressure_balance_max),
            "lcfs_edge_pressure": float(lcfs.edge_pressure),
            "lcfs_merit": float(lcfs_merit.value),
            "lcfs_merit_pressure_scale": float(lcfs_merit.pressure_scale),
            "lcfs_merit_bnormal_scale": float(lcfs_merit.bnormal_scale),
            "lcfs_merit_bnormal_weight": float(lcfs_merit.bnormal_weight),
            "lcfs_pressure_response_min": float(np.min(proposal.pressure_response)),
            "lcfs_pressure_response_max": float(np.max(proposal.pressure_response)),
            "lcfs_update_pressure_balance_rms_predicted": float(proposal.pressure_balance_rms_predicted),
            "lcfs_update_strategy": str(proposal.strategy),
            "lcfs_update_candidate_summaries": proposal_selection.candidate_summaries,
            "lcfs_update_allowed_strategies": list(proposal_selection.allowed_strategies),
            "lcfs_update_rejection_reason": proposal_selection.rejection_reason,
            "lcfs_update_normal_field_guard": bool(lcfs_require_bnormal_nonincrease),
            "lcfs_update_pressure_balance_rms_reduction_fraction": float(
                1.0 - proposal.pressure_balance_rms_predicted / max(proposal.pressure_balance_rms_before, 1.0e-300)
            ),
            "lcfs_update_cap_taper_power": float(proposal.cap_taper_power),
            "lcfs_update_smoothing_passes": int(proposal.smoothing_passes),
            "lcfs_update_max_abs_delta_radius": float(np.max(np.abs(proposal.delta_radius))),
            "lcfs_update_max_relative_delta_radius": float(
                np.max(np.abs(proposal.delta_radius) / np.maximum(proposal.old_radius, 1.0e-300))
            ),
            "lcfs_pilot_rows": pilot_rows,
            "ls_boundary_step": ls_boundary_step,
            "ls_boundary_coupled_loop_rows": ls_boundary_coupled_loop_rows,
            "figures": plot_paths,
        }
        row.update(_lcfs_pilot_summary(pilot_rows))
        row.update(_ls_boundary_coupled_loop_summary(ls_boundary_coupled_loop_rows))
        rows.append(row)
    return rows


def run_case(
    outdir: Path,
    *,
    coil_radius: float = 0.35,
    separation: float = 2.0,
    current: float = 1.0e6,
    midplane_radius: float = 0.3,
    ns: int = 7,
    ntheta: int = 32,
    nxi: int = 33,
    n_segments: int = 256,
    betas: tuple[float, ...] = (1.0, 3.0, 10.0),
    pressure_scale_one_percent: float = 1.0,
    run_fixed_boundary_baseline: bool = False,
    baseline_maxiter: int = 0,
    baseline_psi_prime: float = 0.01,
    lcfs_update_damping: float = 0.25,
    lcfs_update_max_relative_step: float = 0.05,
    lcfs_update_cap_taper_power: float = 2.0,
    lcfs_update_smoothing_passes: int = 1,
    lcfs_merit_bnormal_weight: float = 1.0,
    lcfs_proposal_mode: str = "best_predicted",
    lcfs_coupled_fsq_weight: float = 1.0,
    lcfs_require_bnormal_nonincrease: bool = False,
    run_lcfs_pilot: bool = False,
    lcfs_pilot_steps: int = 1,
    lcfs_pilot_target_merit: float = 0.0,
    lcfs_pilot_stagnation_rtol: float = 0.0,
    lcfs_pilot_fsq_growth_limit: float = 0.0,
    run_ls_boundary_step: bool = False,
    run_ls_boundary_coupled_trial: bool = False,
    run_ls_boundary_coupled_loop: bool = False,
    ls_boundary_coupled_loop_steps: int = 1,
    ls_boundary_coupled_loop_target_merit: float = 0.0,
    ls_boundary_coupled_loop_stagnation_rtol: float = 0.0,
    ls_boundary_coupled_loop_fsq_growth_limit: float = 0.0,
    ls_boundary_finite_difference_step: float = 1.0e-5,
    ls_boundary_damping: float = 1.0,
    ls_boundary_max_relative_step: float = 0.1,
    ls_boundary_ridge: float = 1.0e-8,
    write_plots: bool = True,
) -> Path:
    if run_lcfs_pilot and int(lcfs_pilot_steps) < 1:
        raise ValueError("lcfs_pilot_steps must be at least 1 when run_lcfs_pilot is enabled")
    if float(lcfs_pilot_target_merit) < 0.0:
        raise ValueError("lcfs_pilot_target_merit must be nonnegative")
    if float(lcfs_pilot_stagnation_rtol) < 0.0:
        raise ValueError("lcfs_pilot_stagnation_rtol must be nonnegative")
    if float(lcfs_pilot_fsq_growth_limit) < 0.0:
        raise ValueError("lcfs_pilot_fsq_growth_limit must be nonnegative")
    if float(lcfs_coupled_fsq_weight) < 0.0:
        raise ValueError("lcfs_coupled_fsq_weight must be nonnegative")
    if run_ls_boundary_coupled_trial and not run_ls_boundary_step:
        raise ValueError("run_ls_boundary_coupled_trial requires run_ls_boundary_step")
    if run_ls_boundary_coupled_loop and int(ls_boundary_coupled_loop_steps) < 1:
        raise ValueError("ls_boundary_coupled_loop_steps must be at least 1 when the loop is enabled")
    if float(ls_boundary_coupled_loop_target_merit) < 0.0:
        raise ValueError("ls_boundary_coupled_loop_target_merit must be nonnegative")
    if float(ls_boundary_coupled_loop_stagnation_rtol) < 0.0:
        raise ValueError("ls_boundary_coupled_loop_stagnation_rtol must be nonnegative")
    if float(ls_boundary_coupled_loop_fsq_growth_limit) < 0.0:
        raise ValueError("ls_boundary_coupled_loop_fsq_growth_limit must be nonnegative")
    if float(ls_boundary_finite_difference_step) <= 0.0:
        raise ValueError("ls_boundary_finite_difference_step must be positive")
    if float(ls_boundary_damping) <= 0.0:
        raise ValueError("ls_boundary_damping must be positive")
    if float(ls_boundary_max_relative_step) <= 0.0:
        raise ValueError("ls_boundary_max_relative_step must be positive")
    if float(ls_boundary_ridge) < 0.0:
        raise ValueError("ls_boundary_ridge must be nonnegative")
    outdir.mkdir(parents=True, exist_ok=True)
    grid = make_mirror_grid(
        ns=ns, ntheta=ntheta, nxi=nxi, mpol=max(0, (ntheta - 1) // 2), z_min=-0.5 * separation, z_max=0.5 * separation
    )
    coils = MirrorCircularCoils.symmetric_pair(
        coil_radius_m=coil_radius,
        separation_m=separation,
        current_a=current,
        n_segments=n_segments,
    )
    scan = make_mirror_free_boundary_circular_coil_scan(
        coils,
        betas,
        pressure_scale_for_one_percent=pressure_scale_one_percent,
    )
    analytic_bz = two_coil_on_axis_bz(
        grid.z,
        coil_radius_m=coil_radius,
        separation_m=separation,
        current_a=current,
    )
    boundary = initial_mirror_boundary_from_circular_coil_scan(
        grid,
        scan,
        midplane_radius=midplane_radius,
    )
    axis_sample = sample_mirror_axis_external_field(grid, coils)
    boundary_sample = sample_mirror_boundary_external_field(grid, boundary, coils)
    direct_bz = np.asarray(axis_sample.bz, dtype=float)
    relative_error = np.max(np.abs(direct_bz - analytic_bz) / np.maximum(np.abs(analytic_bz), np.finfo(float).tiny))
    setup_path = write_mirror_free_boundary_circular_coil_scan(
        outdir / "free_boundary_circular_coils_setup.json",
        scan,
    )

    figure_paths: dict[str, str] = {}
    if write_plots:
        figure_dir = outdir / "figures"
        figure_paths["axis_bz"] = str(_write_axis_plot(grid.z, direct_bz, analytic_bz, outdir=figure_dir))
        figure_paths["boundary_bmag"] = str(_write_boundary_bmag_plot(boundary_sample, outdir=figure_dir))
        figure_paths["geometry"] = str(_write_geometry_plot(grid, boundary, coils, outdir=figure_dir))
    baseline_rows = (
        _run_fixed_boundary_baseline_cases(
            outdir=outdir,
            grid=grid,
            boundary=boundary,
            scan=scan,
            maxiter=baseline_maxiter,
            psi_prime_value=baseline_psi_prime,
            lcfs_update_damping=lcfs_update_damping,
            lcfs_update_max_relative_step=lcfs_update_max_relative_step,
            lcfs_update_cap_taper_power=lcfs_update_cap_taper_power,
            lcfs_update_smoothing_passes=lcfs_update_smoothing_passes,
            lcfs_merit_bnormal_weight=lcfs_merit_bnormal_weight,
            lcfs_proposal_mode=lcfs_proposal_mode,
            lcfs_coupled_fsq_weight=lcfs_coupled_fsq_weight,
            lcfs_require_bnormal_nonincrease=lcfs_require_bnormal_nonincrease,
            run_lcfs_pilot=run_lcfs_pilot,
            lcfs_pilot_steps=lcfs_pilot_steps,
            lcfs_pilot_target_merit=lcfs_pilot_target_merit,
            lcfs_pilot_stagnation_rtol=lcfs_pilot_stagnation_rtol,
            lcfs_pilot_fsq_growth_limit=lcfs_pilot_fsq_growth_limit,
            run_ls_boundary_step=run_ls_boundary_step,
            run_ls_boundary_coupled_trial=run_ls_boundary_coupled_trial,
            run_ls_boundary_coupled_loop=run_ls_boundary_coupled_loop,
            ls_boundary_coupled_loop_steps=ls_boundary_coupled_loop_steps,
            ls_boundary_coupled_loop_target_merit=ls_boundary_coupled_loop_target_merit,
            ls_boundary_coupled_loop_stagnation_rtol=ls_boundary_coupled_loop_stagnation_rtol,
            ls_boundary_coupled_loop_fsq_growth_limit=ls_boundary_coupled_loop_fsq_growth_limit,
            ls_boundary_finite_difference_step=ls_boundary_finite_difference_step,
            ls_boundary_damping=ls_boundary_damping,
            ls_boundary_max_relative_step=ls_boundary_max_relative_step,
            ls_boundary_ridge=ls_boundary_ridge,
            write_plots=write_plots,
        )
        if run_fixed_boundary_baseline
        else []
    )
    if write_plots and baseline_rows:
        figure_paths["beta_scan_summary"] = str(
            _write_beta_scan_summary_plot(
                baseline_rows,
                outdir=outdir / "figures",
            )
        )

    summary_csv_path = outdir / "free_boundary_circular_coils_beta_scan_summary.csv"
    metrics = {
        "metrics_schema": CIRCULAR_COIL_BETA_SCAN_SCHEMA,
        "metrics_schema_version": CIRCULAR_COIL_BETA_SCAN_SCHEMA_VERSION,
        **_beta_scan_summary(
            baseline_rows,
            run_fixed_boundary_baseline=run_fixed_boundary_baseline,
            run_lcfs_pilot=run_lcfs_pilot,
            lcfs_pilot_steps=lcfs_pilot_steps,
            lcfs_pilot_target_merit=lcfs_pilot_target_merit,
            lcfs_pilot_stagnation_rtol=lcfs_pilot_stagnation_rtol,
            lcfs_pilot_fsq_growth_limit=lcfs_pilot_fsq_growth_limit,
            run_ls_boundary_step=run_ls_boundary_step,
            run_ls_boundary_coupled_trial=run_ls_boundary_coupled_trial,
            run_ls_boundary_coupled_loop=run_ls_boundary_coupled_loop,
            ls_boundary_coupled_loop_steps=ls_boundary_coupled_loop_steps,
            ls_boundary_coupled_loop_target_merit=ls_boundary_coupled_loop_target_merit,
            ls_boundary_coupled_loop_stagnation_rtol=ls_boundary_coupled_loop_stagnation_rtol,
            ls_boundary_coupled_loop_fsq_growth_limit=ls_boundary_coupled_loop_fsq_growth_limit,
            ls_boundary_finite_difference_step=ls_boundary_finite_difference_step,
            ls_boundary_damping=ls_boundary_damping,
            ls_boundary_max_relative_step=ls_boundary_max_relative_step,
            ls_boundary_ridge=ls_boundary_ridge,
        ),
        "coil_radius": float(coil_radius),
        "separation": float(separation),
        "current": float(current),
        "midplane_radius": float(midplane_radius),
        "ns": int(ns),
        "ntheta": int(ntheta),
        "nxi": int(nxi),
        "n_segments": int(n_segments),
        "axis_bz_relative_linf": float(relative_error),
        "axis_bz_min": float(np.min(np.abs(direct_bz))),
        "axis_bz_max": float(np.max(np.abs(direct_bz))),
        "boundary_bmag_min": float(np.min(np.asarray(boundary_sample.bmag))),
        "boundary_bmag_max": float(np.max(np.asarray(boundary_sample.bmag))),
        "lcfs_coupled_fsq_weight": float(lcfs_coupled_fsq_weight),
        "setup_json": str(setup_path),
        "summary_csv": str(summary_csv_path),
        "summary_rows": [],
        "beta_scan_requested_percent": [float(case.beta_percent) for case in scan.beta_cases],
        "beta_cases": [case.to_dict() for case in scan.beta_cases],
        "fixed_boundary_baseline_rows": baseline_rows,
        "figures": figure_paths,
    }
    metrics["summary_rows"] = circular_coil_beta_scan_report_rows(metrics)
    validate_circular_coil_beta_scan_metrics(metrics)
    _write_beta_scan_report_csv(summary_csv_path, metrics["summary_rows"])
    metrics_path = outdir / "free_boundary_circular_coils_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics_path


def main() -> None:
    args = build_parser().parse_args()
    path = run_case(
        args.outdir,
        coil_radius=args.coil_radius,
        separation=args.separation,
        current=args.current,
        midplane_radius=args.midplane_radius,
        ns=args.ns,
        ntheta=args.ntheta,
        nxi=args.nxi,
        n_segments=args.n_segments,
        betas=_parse_float_list(args.betas),
        pressure_scale_one_percent=args.pressure_scale_one_percent,
        run_fixed_boundary_baseline=args.run_fixed_boundary_baseline,
        baseline_maxiter=args.baseline_maxiter,
        baseline_psi_prime=args.baseline_psi_prime,
        lcfs_update_damping=args.lcfs_update_damping,
        lcfs_update_max_relative_step=args.lcfs_update_max_relative_step,
        lcfs_update_cap_taper_power=args.lcfs_update_cap_taper_power,
        lcfs_update_smoothing_passes=args.lcfs_update_smoothing_passes,
        lcfs_merit_bnormal_weight=args.lcfs_merit_bnormal_weight,
        lcfs_proposal_mode=args.lcfs_proposal_mode,
        lcfs_coupled_fsq_weight=args.lcfs_coupled_fsq_weight,
        lcfs_require_bnormal_nonincrease=args.lcfs_require_bnormal_nonincrease,
        run_lcfs_pilot=args.run_lcfs_pilot,
        lcfs_pilot_steps=args.lcfs_pilot_steps,
        lcfs_pilot_target_merit=args.lcfs_pilot_target_merit,
        lcfs_pilot_stagnation_rtol=args.lcfs_pilot_stagnation_rtol,
        lcfs_pilot_fsq_growth_limit=args.lcfs_pilot_fsq_growth_limit,
        run_ls_boundary_step=args.run_ls_boundary_step,
        run_ls_boundary_coupled_trial=args.run_ls_boundary_coupled_trial,
        run_ls_boundary_coupled_loop=args.run_ls_boundary_coupled_loop,
        ls_boundary_coupled_loop_steps=args.ls_boundary_coupled_loop_steps,
        ls_boundary_coupled_loop_target_merit=args.ls_boundary_coupled_loop_target_merit,
        ls_boundary_coupled_loop_stagnation_rtol=args.ls_boundary_coupled_loop_stagnation_rtol,
        ls_boundary_coupled_loop_fsq_growth_limit=args.ls_boundary_coupled_loop_fsq_growth_limit,
        ls_boundary_finite_difference_step=args.ls_boundary_finite_difference_step,
        ls_boundary_damping=args.ls_boundary_damping,
        ls_boundary_max_relative_step=args.ls_boundary_max_relative_step,
        ls_boundary_ridge=args.ls_boundary_ridge,
        write_plots=not args.no_plots,
    )
    print(path)


if __name__ == "__main__":
    main()
