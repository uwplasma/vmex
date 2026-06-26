from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tools.diagnostics import summarize_square_coil_profiles as summary


def test_square_coil_profile_summary_reads_jax_and_vmec2000_rows(tmp_path: Path):
    case_dir = tmp_path / "square_coil_freeb_backend_profile_case_a"
    case_dir.mkdir()
    report = case_dir / "square_coil_free_boundary_backend_profile.json"
    report.write_text(
        json.dumps(
            {
                "configuration": {
                    "mpol": 5,
                    "ntor": 12,
                    "ns": 9,
                    "nzeta": 32,
                    "nzeta_auto": True,
                    "recommended_nzeta": 32,
                    "nvacskip": 1,
                    "solver_mode": "parity",
                    "side_power": 1.25,
                    "corner_power": 1.5,
                    "max_boundary_projection_error": 1.0e-4,
                    "max_iter": 1000,
                    "ftol": 1.0e-6,
                },
                "boundary_projection": {
                    "mode_count": 65,
                    "recommended_nzeta": 32,
                    "max_abs_component_error": 1.2e-4,
                    "max_abs_component_error_rel": 4.5e-4,
                },
                "backends": {
                    "vmec_jax_mgrid": {
                        "status": "completed",
                        "n_iter": 999,
                        "final_fsqr": 2.0e-7,
                        "final_fsqz": 3.0e-7,
                        "final_fsql": 4.0e-7,
                        "final_fsq_component_sum": 9.0e-7,
                        "best_scored_fsq": 8.0e-7,
                        "returned_best_scored_state": True,
                        "best_scored_full_boundary_count": 10,
                        "best_scored_fresh_boundary_count": 8,
                        "final_residual_recomputed_on_accepted_state": True,
                        "free_boundary_fresh_convergence_gate": True,
                        "free_boundary_fresh_convergence_recheck_count": 3,
                        "free_boundary_fresh_convergence_reject_count": 2,
                        "free_boundary_fresh_convergence_failed_count": 1,
                        "free_boundary_convergence_blocked_count": 4,
                        "free_boundary_anderson_pressure_enabled": True,
                        "free_boundary_anderson_pressure_last_theta": 0.25,
                        "boundary_coeff_delta_l2": 0.04,
                        "boundary_coeff_delta_linf": 0.02,
                        "boundary_coeff_delta_rel": 1.0e-2,
                        "boundary_sample_displacement_rms": 0.03,
                        "boundary_sample_displacement_max": 0.07,
                        "boundary_sample_displacement_rel": 8.0e-3,
                        "virtual_casing": {
                            "status": "computed",
                            "external_bnormal_residual_rms": 1.0e-8,
                            "external_bnormal_residual_max": 3.0e-8,
                            "pressure_balance_rms": 2.0e-6,
                            "pressure_balance_max": 5.0e-6,
                            "required_external_b_rms": 0.8,
                            "target_external_b_rms": 0.82,
                            "wall_s": 4.5,
                        },
                        "accepted_provider_parity": {
                            "status": "completed",
                            "sample": "accepted_boundary_mgrid_backend",
                            "field_vector": {"diff_rms_rel": 1.5e-3},
                            "vacuum_channels": {"bnormal": {"diff_rms_rel": 2.5e-3}},
                            "field_rms_rel_lt_5pct": True,
                            "bnormal_rms_rel_lt_10pct": True,
                        },
                        "history": {
                            "dt_eff_stats": {"last": 0.02, "min": 0.01},
                            "time_step_stats": {"last": 0.019},
                            "freeb_full_update_stats": {"sum": 997.0},
                            "freeb_nestor_reused_stats": {"sum": 100.0, "last": 0.0},
                            "freeb_nestor_source_reused_stats": {"sum": 95.0, "last": 1.0},
                            "freeb_nestor_provider_allows_source_reuse_stats": {"last": 1.0},
                            "freeb_nestor_sample_time_stats": {
                                "last": 0.12,
                                "mean": 0.2,
                                "max": 0.5,
                            },
                            "freeb_nestor_solve_time_stats": {
                                "last": 0.03,
                                "mean": 0.04,
                                "max": 0.08,
                            },
                            "freeb_nestor_trial_reused_stats": {"sum": 8.0},
                            "freeb_nestor_trial_failed_stats": {"sum": 2.0},
                            "freeb_nestor_trial_sample_time_stats": {"mean": 0.7, "max": 1.3},
                            "freeb_nestor_trial_solve_time_stats": {"mean": 0.09, "max": 0.2},
                            "include_edge_stats": {"sum": 50.0, "last": 0.0},
                            "freeb_anderson_pressure_applied_stats": {"sum": 12.0},
                            "bad_jacobian_stats": {"sum": 1.0},
                            "freeb_nestor_bnormal_rms_stats": {"last": 4.0e-3, "min": 3.0e-3},
                            "fsq_component_sum_tail_projection": {
                                "per_iter_factor": 0.98,
                                "estimated_additional_iterations_to_target": {"1e-12": 1234},
                            },
                            "fsq_component_tail_projection_by_component": {
                                "fsqr": {
                                    "per_iter_factor": 0.97,
                                    "estimated_additional_iterations_to_target": {"1e-12": 111},
                                },
                                "fsqz": {
                                    "per_iter_factor": 0.96,
                                    "estimated_additional_iterations_to_target": {"1e-12": 222},
                                },
                                "fsql": {
                                    "per_iter_factor": 0.95,
                                    "estimated_additional_iterations_to_target": {"1e-12": 333},
                                },
                            },
                            "fsq_limiting_component": "fsql",
                        },
                        "wall_s": 3.5,
                    },
                    "vmec2000_mgrid": {
                        "status": "completed",
                        "tail_rows": [
                            {
                                "it": 800,
                                "fsqr": 2.5e-6,
                                "fsqz": 3.0e-6,
                                "fsql": 5.0e-7,
                            },
                            {
                                "it": 1000,
                                "fsqr": 2.0e-6,
                                "fsqz": 3.0e-6,
                                "fsql": 4.0e-7,
                            },
                        ],
                        "last_row": {
                            "it": 1000,
                            "fsqr": 2.0e-6,
                            "fsqz": 3.0e-6,
                            "fsql": 4.0e-7,
                        },
                        "min_total": 5.0e-6,
                        "vacuum_grid_exceeded_count": 2,
                        "wall_s": 2.0,
                    },
                },
            }
        )
    )

    rows = summary.rows_from_profile(report)

    assert [row["backend"] for row in rows] == ["vmec2000_mgrid", "vmec_jax_mgrid"]
    assert rows[0]["backend_role"] == "vmec2000_mgrid_reference"
    assert rows[0]["strict_evidence_status"] == "non_strict_ftol"
    assert "requested_ftol_above_1e-12" in rows[0]["strict_evidence_blockers"]
    assert rows[0]["final_total"] == pytest.approx(5.4e-6)
    assert rows[0]["requested_ftol"] == pytest.approx(1.0e-6)
    assert rows[0]["final_max_component"] == pytest.approx(3.0e-6)
    assert rows[0]["final_fsqr"] == pytest.approx(2.0e-6)
    assert rows[0]["final_fsqz"] == pytest.approx(3.0e-6)
    assert rows[0]["final_fsql"] == pytest.approx(4.0e-7)
    assert rows[0]["limiting_component"] == "fsqz"
    assert rows[0]["fsqz_strict_gap"] == pytest.approx(3.0)
    assert rows[0]["strict_components_met"] is False
    assert rows[0]["best_total"] == pytest.approx(5.0e-6)
    assert rows[0]["tail_decay_factor"] < 1.0
    assert rows[0]["iters_to_1e-12_est"] > 0
    assert rows[0]["fsqr_tail_decay_factor"] < 1.0
    assert rows[0]["fsqz_iters_to_1e-12_est"] is None
    assert rows[1]["final_total"] == pytest.approx(9.0e-7)
    assert rows[1]["final_max_component"] == pytest.approx(4.0e-7)
    assert rows[1]["final_fsqr"] == pytest.approx(2.0e-7)
    assert rows[1]["final_fsqz"] == pytest.approx(3.0e-7)
    assert rows[1]["final_fsql"] == pytest.approx(4.0e-7)
    assert rows[1]["limiting_component"] == "fsql"
    assert rows[1]["fsql_strict_gap"] == pytest.approx(0.4)
    assert rows[1]["fsqr_tail_decay_factor"] == pytest.approx(0.97)
    assert rows[1]["fsqz_iters_to_1e-12_est"] == pytest.approx(222)
    assert rows[1]["strict_components_met"] is True
    assert rows[1]["backend_role"] == "vmec_jax_mgrid_parity"
    assert rows[1]["strict_evidence_status"] == "non_strict_ftol"
    assert "requested_ftol_above_1e-12" in rows[1]["strict_evidence_blockers"]
    assert rows[1]["boundary_condition_mode"] == "vacuum_coil_normal"
    assert rows[1]["coil_bnormal_role"] == "vacuum_boundary_condition"
    assert rows[1]["production_candidate"] is True
    assert rows[1]["promotion_blockers"] == ""
    assert rows[1]["virtual_casing_required"] is False
    assert rows[1]["best_total"] == pytest.approx(8.0e-7)
    assert rows[1]["returned_best_scored_state"] is True
    assert rows[1]["best_scored_full_boundary_count"] == 10
    assert rows[1]["best_scored_fresh_boundary_count"] == 8
    assert rows[1]["final_residual_recomputed_on_accepted_state"] is True
    assert rows[1]["fresh_convergence_gate"] is True
    assert rows[1]["fresh_convergence_rechecks"] == 3
    assert rows[1]["fresh_convergence_rejects"] == 2
    assert rows[1]["fresh_convergence_failures"] == 1
    assert rows[1]["freeb_convergence_blocked_count"] == 4
    assert rows[1]["solver_mode"] == "parity"
    assert rows[1]["side_power"] == pytest.approx(1.25)
    assert rows[1]["corner_power"] == pytest.approx(1.5)
    assert rows[1]["nzeta_auto"] is True
    assert rows[1]["recommended_nzeta"] == 32
    assert rows[1]["boundary_mode_count"] == 65
    assert rows[1]["boundary_recommended_nzeta"] == 32
    assert rows[1]["max_boundary_projection_error"] == pytest.approx(1.0e-4)
    assert rows[1]["boundary_proj_max"] == pytest.approx(1.2e-4)
    assert rows[1]["boundary_proj_rel"] == pytest.approx(4.5e-4)
    assert rows[1]["boundary_coeff_delta_l2"] == pytest.approx(0.04)
    assert rows[1]["boundary_coeff_delta_linf"] == pytest.approx(0.02)
    assert rows[1]["boundary_coeff_delta_rel"] == pytest.approx(1.0e-2)
    assert rows[1]["boundary_sample_displacement_rms"] == pytest.approx(0.03)
    assert rows[1]["boundary_sample_displacement_max"] == pytest.approx(0.07)
    assert rows[1]["boundary_sample_displacement_rel"] == pytest.approx(8.0e-3)
    assert rows[1]["dt_eff_last"] == pytest.approx(0.02)
    assert rows[1]["dt_eff_min"] == pytest.approx(0.01)
    assert rows[1]["time_step_last"] == pytest.approx(0.019)
    assert rows[1]["freeb_full_update_count"] == pytest.approx(997.0)
    assert rows[1]["nestor_reuse_count"] == pytest.approx(100.0)
    assert rows[1]["nestor_reuse_last"] == pytest.approx(0.0)
    assert rows[1]["nestor_source_reuse_count"] == pytest.approx(95.0)
    assert rows[1]["nestor_source_reuse_last"] == pytest.approx(1.0)
    assert rows[1]["nestor_provider_source_reuse_allowed_last"] == pytest.approx(1.0)
    assert rows[1]["nestor_sample_time_last"] == pytest.approx(0.12)
    assert rows[1]["nestor_sample_time_mean"] == pytest.approx(0.2)
    assert rows[1]["nestor_sample_time_max"] == pytest.approx(0.5)
    assert rows[1]["nestor_solve_time_last"] == pytest.approx(0.03)
    assert rows[1]["nestor_solve_time_mean"] == pytest.approx(0.04)
    assert rows[1]["nestor_solve_time_max"] == pytest.approx(0.08)
    assert rows[1]["nestor_trial_reuse_count"] == pytest.approx(8.0)
    assert rows[1]["nestor_trial_failed_count"] == pytest.approx(2.0)
    assert rows[1]["nestor_trial_sample_time_mean"] == pytest.approx(0.7)
    assert rows[1]["nestor_trial_sample_time_max"] == pytest.approx(1.3)
    assert rows[1]["nestor_trial_solve_time_mean"] == pytest.approx(0.09)
    assert rows[1]["nestor_trial_solve_time_max"] == pytest.approx(0.2)
    assert rows[1]["include_edge_count"] == pytest.approx(50.0)
    assert rows[1]["include_edge_last"] == pytest.approx(0.0)
    assert rows[1]["anderson_pressure_enabled"] is True
    assert rows[1]["anderson_pressure_applied_count"] == pytest.approx(12.0)
    assert rows[1]["anderson_pressure_last_theta"] == pytest.approx(0.25)
    assert rows[1]["bad_jacobian_count"] == pytest.approx(1.0)
    assert rows[1]["bnormal_rms_last"] == pytest.approx(4.0e-3)
    assert rows[1]["bnormal_rms_min"] == pytest.approx(3.0e-3)
    assert rows[1]["virtual_casing_status"] == "computed"
    assert rows[1]["virtual_casing_external_bnormal_residual_rms"] == pytest.approx(1.0e-8)
    assert rows[1]["virtual_casing_external_bnormal_residual_max"] == pytest.approx(3.0e-8)
    assert rows[1]["virtual_casing_pressure_balance_rms"] == pytest.approx(2.0e-6)
    assert rows[1]["virtual_casing_pressure_balance_max"] == pytest.approx(5.0e-6)
    assert rows[1]["virtual_casing_required_external_b_rms"] == pytest.approx(0.8)
    assert rows[1]["virtual_casing_target_external_b_rms"] == pytest.approx(0.82)
    assert rows[1]["virtual_casing_wall_s"] == pytest.approx(4.5)
    assert rows[1]["accepted_provider_parity_status"] == "completed"
    assert rows[1]["accepted_provider_parity_sample"] == "accepted_boundary_mgrid_backend"
    assert rows[1]["accepted_provider_parity_field_diff_rms_rel"] == pytest.approx(1.5e-3)
    assert rows[1]["accepted_provider_parity_bnormal_diff_rms_rel"] == pytest.approx(2.5e-3)
    assert rows[1]["accepted_provider_parity_field_lt_5pct"] is True
    assert rows[1]["accepted_provider_parity_bnormal_lt_10pct"] is True
    assert rows[1]["tail_decay_factor"] == pytest.approx(0.98)
    assert rows[1]["iters_to_1e-12_est"] == pytest.approx(1234)
    assert rows[0]["vacuum_grid_exceeded_count"] == 2


def test_square_coil_profile_summary_marks_strict_direct_row_as_evidence(tmp_path: Path):
    case_dir = tmp_path / "square_coil_freeb_backend_profile_strict_direct"
    case_dir.mkdir()
    report = case_dir / "square_coil_free_boundary_backend_profile.json"
    report.write_text(
        json.dumps(
            {
                "configuration": {
                    "beta_percent": 0.0,
                    "mpol": 5,
                    "ntor": 28,
                    "ns": 17,
                    "nzeta": 64,
                    "ftol": 1.0e-12,
                    "nzeta_underrecommended": False,
                },
                "resolution_deck": {
                    "status": "production_ready",
                    "reasons": [],
                    "mgrid_nphi_multiple_of_nzeta": True,
                },
                "backends": {
                    "vmec_jax_direct": {
                        "status": "completed",
                        "n_iter": 200,
                        "final_fsqr": 4.0e-13,
                        "final_fsqz": 5.0e-13,
                        "final_fsql": 6.0e-13,
                        "final_residual_recomputed_on_accepted_state": True,
                    }
                },
            }
        )
    )

    row = summary.rows_from_profile(report)[0]

    assert row["backend_role"] == "vmec_jax_direct_research"
    assert row["strict_components_met"] is True
    assert row["production_candidate"] is True
    assert row["strict_evidence_status"] == "strict_production_evidence"
    assert row["strict_evidence_blockers"] == ""
    assert row["resolution_deck_status"] == "production_ready"
    assert row["recommended_followup_profile_kind"] == "none"
    assert row["recommended_followup_reason"] == "strict_evidence"


def test_square_coil_profile_summary_blocks_underresolved_resolution_deck(tmp_path: Path):
    case_dir = tmp_path / "square_coil_freeb_backend_profile_underresolved"
    case_dir.mkdir()
    report = case_dir / "square_coil_free_boundary_backend_profile.json"
    report.write_text(
        json.dumps(
            {
                "configuration": {
                    "beta_percent": 0.0,
                    "mpol": 5,
                    "ntor": 28,
                    "ns": 17,
                    "nzeta": 32,
                    "ftol": 1.0e-12,
                    "nzeta_underrecommended": True,
                },
                "resolution_deck": {
                    "status": "diagnostic_underresolved",
                    "reasons": [
                        "nzeta_below_square_axis_recommendation",
                        "mgrid_nphi_not_multiple_of_nzeta",
                    ],
                    "mgrid_nphi_multiple_of_nzeta": False,
                },
                "backends": {
                    "vmec_jax_direct": {
                        "status": "completed",
                        "n_iter": 200,
                        "final_fsqr": 4.0e-13,
                        "final_fsqz": 5.0e-13,
                        "final_fsql": 6.0e-13,
                        "final_residual_recomputed_on_accepted_state": True,
                    }
                },
            }
        )
    )

    row = summary.rows_from_profile(report)[0]

    assert row["strict_components_met"] is True
    assert row["strict_evidence_status"] == "diagnostic_underresolved"
    assert row["resolution_deck_status"] == "diagnostic_underresolved"
    assert row["resolution_deck_reasons"] == (
        "nzeta_below_square_axis_recommendation,mgrid_nphi_not_multiple_of_nzeta"
    )
    assert "resolution_deck_diagnostic_underresolved" in row["strict_evidence_blockers"]
    assert "mgrid_nphi_not_multiple_of_nzeta" in row["strict_evidence_blockers"]
    assert row["recommended_followup_profile_kind"] == "resolution-preflight"
    assert row["recommended_followup_reason"] == "fix_projection_nzeta_or_mgrid_gate_first"


def test_square_coil_profile_summary_recommends_provider_parity_when_missing(
    tmp_path: Path,
):
    case_dir = tmp_path / "square_coil_freeb_backend_profile_missing_parity"
    case_dir.mkdir()
    report = case_dir / "square_coil_free_boundary_backend_profile.json"
    report.write_text(
        json.dumps(
            {
                "configuration": {
                    "mpol": 5,
                    "ntor": 28,
                    "ns": 17,
                    "nzeta": 64,
                    "ftol": 1e-12,
                    "jax_hot_restart_count": 2,
                    "jax_hot_restart_iters": 4000,
                    "jax_hot_restart_policy": "freeb",
                    "jax_hot_restart_always": False,
                    "jax_initial_restart_wout": "results/seed/wout.nc",
                },
                "resolution_deck": {
                    "status": "production_ready",
                    "reasons": [],
                    "mgrid_nphi_multiple_of_nzeta": True,
                },
                "backends": {
                    "vmec_jax_direct": {
                        "status": "completed",
                        "n_iter": 1000,
                        "final_fsqr": 2.0e-9,
                        "final_fsqz": 3.0e-9,
                        "final_fsql": 4.0e-10,
                        "final_residual_recomputed_on_accepted_state": True,
                    }
                },
            }
        )
    )

    row = summary.rows_from_profile(report)[0]

    assert row["strict_evidence_status"] == "underconverged"
    assert row["accepted_provider_parity_status"] == "not_run"
    assert row["recommended_followup_profile_kind"] == "provider-parity"
    assert row["recommended_followup_reason"] == "accepted_lcfs_provider_parity_missing"


def test_square_coil_profile_summary_recommends_vmec2000_when_grid_exceeded(
    tmp_path: Path,
):
    case_dir = tmp_path / "square_coil_freeb_backend_profile_grid_exceeded_completed"
    case_dir.mkdir()
    report = case_dir / "square_coil_free_boundary_backend_profile.json"
    report.write_text(
        json.dumps(
            {
                "configuration": {"mpol": 5, "ntor": 28, "ns": 17, "nzeta": 64, "ftol": 1e-12},
                "backends": {
                    "vmec2000_mgrid": {
                        "status": "completed",
                        "tail_rows": [
                            {"it": 1, "total": 2.0e-9, "max_component": 1.0e-9},
                            {"it": 2, "total": 2.1e-9, "max_component": 1.0e-9},
                            {"it": 3, "total": 2.2e-9, "max_component": 1.0e-9},
                        ],
                        "vacuum_grid_exceeded_count": 1,
                    }
                },
            }
        )
    )

    row = summary.rows_from_profile(report)[0]

    assert row["vacuum_grid_exceeded_count"] == 1
    assert row["next_action"] == "widen_mgrid_before_interpreting_residual"
    assert row["recommended_followup_profile_kind"] == "vmec2000"
    assert row["recommended_followup_reason"] == "widen_mgrid_before_backend_comparison"


def test_square_coil_profile_summary_recommends_vmec2000_for_failed_flat_tail(
    tmp_path: Path,
):
    case_dir = tmp_path / "square_coil_freeb_backend_profile_vmec2000_timeout_flat"
    case_dir.mkdir()
    report = case_dir / "square_coil_free_boundary_backend_profile.json"
    report.write_text(
        json.dumps(
            {
                "configuration": {
                    "mpol": 5,
                    "ntor": 28,
                    "ns": 17,
                    "nzeta": 64,
                    "ftol": 1e-12,
                    "max_iter": 24000,
                },
                "backends": {
                    "vmec2000_mgrid": {
                        "status": "failed",
                        "last_row": {
                            "it": 11957,
                            "fsqr": 1.12e-11,
                            "fsqz": 8.12e-12,
                            "fsql": 3.81e-12,
                            "total": 2.313e-11,
                            "max_component": 1.12e-11,
                        },
                        "tail_plateau": {"status": "flat_above_stage_ftol"},
                        "vacuum_grid_exceeded_count": 0,
                    }
                },
            }
        )
    )

    row = summary.rows_from_profile(report)[0]

    assert row["status"] == "failed"
    assert row["strict_gap"] == pytest.approx(11.2)
    assert row["remaining_iterations"] == 12043
    assert row["next_action"] == "scan_delt_or_stage_budget"
    assert row["recommended_followup_profile_kind"] == "vmec2000"
    assert row["recommended_followup_reason"] == "scan_delt_or_stage_budget"


def test_square_coil_profile_summary_recommends_direct_gpu_for_stalled_direct(
    tmp_path: Path,
):
    case_dir = tmp_path / "square_coil_freeb_backend_profile_stalled_direct"
    case_dir.mkdir()
    report = case_dir / "square_coil_free_boundary_backend_profile.json"
    report.write_text(
        json.dumps(
            {
                "configuration": {"mpol": 5, "ntor": 28, "ns": 17, "nzeta": 64, "ftol": 1e-12},
                "resolution_deck": {
                    "status": "production_ready",
                    "reasons": [],
                    "mgrid_nphi_multiple_of_nzeta": True,
                },
                "backends": {
                    "vmec_jax_direct": {
                        "status": "completed",
                        "n_iter": 1000,
                        "final_fsqr": 2.0e-9,
                        "final_fsqz": 3.0e-9,
                        "final_fsql": 4.0e-10,
                        "tail_plateau": {"status": "oscillatory"},
                        "accepted_provider_parity": {"status": "completed"},
                        "final_residual_recomputed_on_accepted_state": True,
                    }
                },
            }
        )
    )

    row = summary.rows_from_profile(report)[0]

    assert row["next_action"] == "scan_delt_stage_budget_or_pressure_acceleration"
    assert row["recommended_followup_profile_kind"] == "direct-gpu-edge-polish"
    assert row["recommended_followup_reason"] == "scan_delt_stage_budget_or_pressure_acceleration"
    assert row["freeb_edge_control_projection_status"] == "disabled"


def test_square_coil_profile_summary_recommends_edge_jax_nestor_for_stalled_edge_direct(
    tmp_path: Path,
):
    case_dir = tmp_path / "square_coil_freeb_backend_profile_stalled_edge_direct"
    case_dir.mkdir()
    report = case_dir / "square_coil_free_boundary_backend_profile.json"
    report.write_text(
        json.dumps(
            {
                "configuration": {"mpol": 5, "ntor": 28, "ns": 17, "nzeta": 64, "ftol": 1e-12},
                "resolution_deck": {
                    "status": "production_ready",
                    "reasons": [],
                    "mgrid_nphi_multiple_of_nzeta": True,
                },
                "backends": {
                    "vmec_jax_direct": {
                        "status": "completed",
                        "n_iter": 1000,
                        "final_fsqr": 2.0e-9,
                        "final_fsqz": 3.0e-9,
                        "final_fsql": 4.0e-10,
                        "update_delta_rms": 2.0e-5,
                        "update_delta_to_velocity_rms_ratio": 0.4,
                        "tail_plateau": {"status": "oscillatory"},
                        "accepted_provider_parity": {"status": "completed"},
                        "final_residual_recomputed_on_accepted_state": True,
                        "free_boundary_edge_control_projection": {
                            "apply_count": 7,
                            "delta_projection_count": 8,
                            "zero_velocity_count": 6,
                            "state_residual": {
                                "status": "measured",
                                "residual_linf": 2.5e-14,
                                "residual_rms": 1.0e-14,
                                "residual_rel": 4.0e-13,
                                "captured_fraction": 1.0,
                            },
                            "update_direction": {
                                "status": "measured",
                                "residual_linf": 3.0e-11,
                                "residual_rms": 9.0e-12,
                                "residual_rel": 0.25,
                                "captured_fraction": 0.9682458365518543,
                            },
                        },
                        "free_boundary_solver_overrides": {
                            "freeb_edge_control_projection": {
                                "requested": "square",
                                "enabled": True,
                                "status": "enabled",
                                "basis_symmetry": "square",
                                "control_count": 2,
                                "rcond": 1.0e-12,
                            },
                        },
                    }
                },
            }
        )
    )

    row = summary.rows_from_profile(report)[0]

    assert row["recommended_followup_profile_kind"] == "direct-gpu-edge-jax-nestor-polish"
    assert row["recommended_followup_reason"] == "scan_delt_stage_budget_or_pressure_acceleration"
    assert row["freeb_edge_control_projection_status"] == "enabled"
    assert row["freeb_edge_control_projection_requested"] == "square"
    assert row["freeb_edge_control_projection_basis"] == "square"
    assert row["freeb_edge_control_projection_control_count"] == 2
    assert row["freeb_edge_control_projection_rcond"] == pytest.approx(1.0e-12)
    assert row["freeb_edge_control_projection_apply_count"] == 7
    assert row["freeb_edge_control_projection_delta_projection_count"] == 8
    assert row["freeb_edge_control_projection_zero_velocity_count"] == 6
    assert row["freeb_edge_control_projection_state_residual_status"] == "measured"
    assert row["freeb_edge_control_projection_state_residual_linf"] == pytest.approx(2.5e-14)
    assert row["freeb_edge_control_projection_state_residual_rms"] == pytest.approx(1.0e-14)
    assert row["freeb_edge_control_projection_state_residual_rel"] == pytest.approx(4.0e-13)
    assert row["freeb_edge_control_projection_state_captured_fraction"] == pytest.approx(1.0)
    assert row["freeb_edge_control_projection_update_direction_status"] == "measured"
    assert row["freeb_edge_control_projection_update_direction_linf"] == pytest.approx(3.0e-11)
    assert row["freeb_edge_control_projection_update_direction_rms"] == pytest.approx(9.0e-12)
    assert row["freeb_edge_control_projection_update_direction_rel"] == pytest.approx(0.25)
    assert row["freeb_edge_control_projection_update_direction_captured_fraction"] == pytest.approx(
        0.9682458365518543
    )
    assert row["update_delta_rms"] == pytest.approx(2.0e-5)
    assert row["update_delta_to_velocity_rms_ratio"] == pytest.approx(0.4)


def test_square_coil_profile_summary_infers_resolution_deck_for_live_launcher_log(
    tmp_path: Path,
):
    case_dir = tmp_path / "square_coil_direct_gpu_ns17_mpol5_ntor28_nzeta64_niter8k_control_spline"
    case_dir.mkdir()
    launcher = case_dir / "launcher.log"
    launcher.write_text(
        "\n".join(
            [
                "[square-coil-profile] building square-coil configuration beta=0%, "
                "mpol=5, ntor=28, ns=[9, 13, 17], nzeta=64, side_power=1, corner_power=1",
                "[square-coil-profile] running vmec_jax direct-coil backend",
                "  NS =   17 NO. FOURIER MODES =  257 FTOLV =  1.000E-12 NITER =   8000",
                "  ITER    FSQR      FSQZ      FSQL    RAX(v=0)    DELT       WMHD",
                "    1  7.37E-08  6.27E-08  3.44E-11  1.457E+00  2.00E-02  3.2171E-01",
            ]
        )
    )

    row = summary.rows_from_source(launcher)[0]

    assert row["backend"] == "vmec_jax_direct_live"
    assert row["resolution_deck_status"] == "production_ready"
    assert row["resolution_deck_reasons"] == ""
    assert row["boundary_proj_max"] == pytest.approx(3.480773921149713e-12)
    assert row["recommended_nzeta"] == 64
    assert row["boundary_recommended_nzeta"] == 64
    assert row["max_boundary_projection_error"] == pytest.approx(5.0e-12)


def test_square_coil_profile_summary_infers_underresolved_live_launcher_log(
    tmp_path: Path,
):
    case_dir = tmp_path / "square_coil_direct_gpu_ns17_mpol5_ntor28_nzeta48_niter8k_control_spline"
    case_dir.mkdir()
    launcher = case_dir / "launcher.log"
    launcher.write_text(
        "\n".join(
            [
                "[square-coil-profile] building square-coil configuration beta=0%, "
                "mpol=5, ntor=28, ns=[9, 13, 17], nzeta=48, side_power=1, corner_power=1",
                "[square-coil-profile] running vmec_jax direct-coil backend",
                "  NS =   17 NO. FOURIER MODES =  257 FTOLV =  1.000E-12 NITER =   8000",
                "  ITER    FSQR      FSQZ      FSQL    RAX(v=0)    DELT       WMHD",
                "    1  7.37E-08  6.27E-08  3.44E-11  1.457E+00  2.00E-02  3.2171E-01",
            ]
        )
    )

    row = summary.rows_from_source(launcher)[0]

    assert row["resolution_deck_status"] == "diagnostic_underresolved"
    assert row["resolution_deck_reasons"] == "nzeta_below_square_axis_recommendation"
    assert "resolution_deck_diagnostic_underresolved" in row["strict_evidence_blockers"]
    assert row["strict_evidence_status"] == "diagnostic_underresolved"


def test_square_coil_profile_summary_recommends_direct_gpu_after_jax_nestor_probe(
    tmp_path: Path,
):
    case_dir = tmp_path / "square_coil_freeb_backend_profile_stalled_direct_jax_nestor"
    case_dir.mkdir()
    report = case_dir / "square_coil_free_boundary_backend_profile.json"
    report.write_text(
        json.dumps(
            {
                "configuration": {"mpol": 5, "ntor": 28, "ns": 17, "nzeta": 64, "ftol": 1e-12},
                "resolution_deck": {
                    "status": "production_ready",
                    "reasons": [],
                    "mgrid_nphi_multiple_of_nzeta": True,
                },
                "backends": {
                    "vmec_jax_direct": {
                        "status": "completed",
                        "n_iter": 1000,
                        "final_fsqr": 2.0e-9,
                        "final_fsqz": 3.0e-9,
                        "final_fsql": 4.0e-10,
                        "tail_plateau": {"status": "oscillatory"},
                        "accepted_provider_parity": {"status": "completed"},
                        "final_residual_recomputed_on_accepted_state": True,
                        "free_boundary_solver_overrides": {
                            "freeb_jax_nestor_operator": True,
                            "freeb_jax_nestor_jit_operator": False,
                            "freeb_edge_control_projection": {
                                "requested": "square",
                                "enabled": True,
                                "status": "enabled",
                                "basis_symmetry": "square",
                                "control_count": 2,
                                "rcond": 1.0e-12,
                            },
                            "jax_hot_restart_count": 2,
                            "jax_hot_restart_iters": 4000,
                            "jax_hot_restart_policy": "freeb",
                            "jax_hot_restart_always": False,
                            "jax_initial_restart_wout": "results/seed/wout.nc",
                        },
                        "hot_restart": {
                            "requested_count": 2,
                            "executed_count": 1,
                            "stopped_after_strict_convergence": False,
                            "stages": [
                                {"strict_status": "underconverged", "component_max": 3.0e-9},
                                {"strict_status": "stalled_above_strict_ftol", "component_max": 3.0e-9},
                            ],
                        },
                        "free_boundary_jax_nestor_operator_applied": True,
                        "free_boundary_jax_nestor_operator_reason": "applied",
                        "free_boundary_jax_nestor_operator_jitted": False,
                        "free_boundary_jax_nestor_operator_cache_hit": False,
                        "free_boundary_jax_nestor_operator_time_s": 0.25,
                    }
                },
            }
        )
    )

    row = summary.rows_from_profile(report)[0]

    assert row["freeb_jax_nestor_operator"] is True
    assert row["freeb_jax_nestor_jit_operator"] is False
    assert row["jax_hot_restart_requested_count"] == 2
    assert row["jax_hot_restart_executed_count"] == 1
    assert row["jax_hot_restart_iters"] == 4000
    assert row["jax_hot_restart_policy"] == "freeb"
    assert row["jax_hot_restart_always"] is False
    assert row["jax_initial_restart_wout"] == "results/seed/wout.nc"
    assert row["jax_hot_restart_stopped_after_strict"] is False
    assert row["jax_hot_restart_last_status"] == "stalled_above_strict_ftol"
    assert row["jax_hot_restart_last_component_max"] == pytest.approx(3.0e-9)
    assert row["free_boundary_jax_nestor_operator_applied"] is True
    assert row["free_boundary_jax_nestor_operator_reason"] == "applied"
    assert row["free_boundary_jax_nestor_operator_jitted"] is False
    assert row["free_boundary_jax_nestor_operator_cache_hit"] is False
    assert row["free_boundary_jax_nestor_operator_time_s"] == pytest.approx(0.25)
    assert row["recommended_followup_profile_kind"] == "direct-gpu"
    assert row["recommended_followup_reason"] == "scan_delt_stage_budget_or_pressure_acceleration"


def test_square_coil_profile_summary_markdown_includes_virtual_casing_columns(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    case_dir = tmp_path / "square_coil_freeb_backend_profile_vc_case"
    case_dir.mkdir()
    report = case_dir / "square_coil_free_boundary_backend_profile.json"
    report.write_text(
        json.dumps(
            {
                "configuration": {
                    "beta_percent": 3.0,
                    "mpol": 5,
                    "ntor": 28,
                    "ns": 9,
                    "nzeta": 64,
                    "ftol": 1.0e-12,
                },
                "backends": {
                    "vmec_jax_direct": {
                        "status": "completed",
                        "n_iter": 2,
                        "final_fsqr": 1.0e-13,
                        "final_fsqz": 2.0e-13,
                        "final_fsql": 3.0e-13,
                        "final_fsq_component_sum": 6.0e-13,
                        "final_residual_recomputed_on_accepted_state": True,
                        "virtual_casing": {
                            "status": "computed",
                            "external_bnormal_residual_rms": 4.0e-8,
                            "pressure_balance_rms": 5.0e-6,
                        },
                    }
                },
            }
        )
    )

    assert summary.main([str(report), "--markdown"]) == 0
    out = capsys.readouterr().out

    assert "virtual_casing_status" in out
    assert "virtual_casing_pressure_balance_rms" in out
    assert "finite_beta_total_field" in out
    assert "diagnostic_only" in out
    assert "strict_production_evidence" in out
    assert "computed" in out
    assert "5e-06" in out


def test_square_coil_profile_summary_reads_active_vmec2000_threed1(tmp_path: Path):
    case_dir = tmp_path / "square_coil_freeb_backend_profile_vmec2000_ns9_13_17_mpol7_ntor28_nzeta64_niter24k_fg"
    workdir = case_dir / "vmec2000_mgrid"
    workdir.mkdir(parents=True)
    threed1 = workdir / "threed1.square_beta_00p000_mgrid"
    threed1.write_text(
        "\n".join(
            [
                " NS =    17 NO. FOURIER MODES =  371 FTOLV =  1.000E-12 NITER =  24000",
                " ITER    FSQR      FSQZ      FSQL      fsqr      fsqz      fsql      DELT",
                "  200   4.00E-11  2.00E-11  1.00E-11  1.00E-12  2.00E-12  3.00E-12  2.00E-02",
                "  400   3.00E-11  2.50E-11  8.00E-12  1.00E-12  2.00E-12  3.00E-12  2.00E-02",
            ]
        )
        + "\n"
    )

    assert summary._profile_paths([case_dir]) == [threed1]
    rows = summary.rows_from_source(threed1)

    assert len(rows) == 1
    row = rows[0]
    assert row["case"] == "vmec2000_ns9_13_17_mpol7_ntor28_nzeta64_niter24k_fg"
    assert row["backend"] == "vmec2000_mgrid"
    assert row["status"] == "running_partial"
    assert row["progress_phase"] == "force_iterations"
    assert row["force_rows_started"] is True
    assert row["threed1_size_bytes"] > 0
    assert row["threed1_mtime_unix_s"] > 0.0
    assert row["mpol"] == 7
    assert row["ntor"] == 28
    assert row["ns"] == 17
    assert row["nzeta"] == 64
    assert row["max_iter"] == 24000
    assert row["requested_ftol"] == pytest.approx(1.0e-12)
    assert row["final_iter"] == 400
    assert row["final_total"] == pytest.approx(6.3e-11)
    assert row["final_max_component"] == pytest.approx(3.0e-11)
    assert row["final_fsqr"] == pytest.approx(3.0e-11)
    assert row["final_fsqz"] == pytest.approx(2.5e-11)
    assert row["final_fsql"] == pytest.approx(8.0e-12)
    assert row["limiting_component"] == "fsqr"
    assert row["fsqr_strict_gap"] == pytest.approx(30.0)
    assert row["strict_components_met"] is False
    assert row["tail_decay_factor"] == pytest.approx(0.9994733361578259)
    assert row["iters_to_1e-12_est"] == pytest.approx(7865)
    assert row["fsqr_iters_to_1e-12_est"] is not None
    assert row["recommended_followup_profile_kind"] == "wait_current_run"
    assert row["recommended_followup_reason"] == "active_profile_still_running"


def test_square_coil_profile_summary_prefers_active_partial_sidecar(tmp_path: Path):
    case_dir = tmp_path / "square_coil_freeb_backend_profile_live_sidecar"
    case_dir.mkdir()
    partial = case_dir / "_partial_vmec2000_payload.json"
    partial.write_text(
        json.dumps(
            {
                "stage_summaries": [{"ns": 17, "ftolv": 1.0e-12}],
                "last_row": {"it": 600, "fsqr": 8.0e-13, "fsqz": 9.0e-13, "fsql": 7.0e-13},
                "min_total": 2.4e-12,
                "vacuum_grid_exceeded_count": 0,
            }
        )
    )

    assert summary._profile_paths([case_dir]) == [partial]
    row = summary.rows_from_source(partial)[0]

    assert row["case"] == "live_sidecar"
    assert row["status"] == "running_partial"
    assert row["requested_ftol"] == pytest.approx(1.0e-12)
    assert row["final_max_component"] == pytest.approx(9.0e-13)
    assert row["strict_components_met"] is True


def test_square_coil_profile_summary_accepts_renamed_copied_sidecar(tmp_path: Path):
    sidecar = tmp_path / "copied_active_payload.json"
    sidecar.write_text(
        json.dumps(
            {
                "stage_summaries": [{"ns": 17, "niter": 24000, "ftolv": 1.0e-12}],
                "last_row": {"it": 5200, "fsqr": 2.48e-11, "fsqz": 5.45e-12, "fsql": 2.35e-12},
                "tail_rows": [
                    {"it": 5198, "total": 2.69e-11, "max_component": 2.49e-11},
                    {"it": 5199, "total": 2.68e-11, "max_component": 2.48e-11},
                    {"it": 5200, "total": 2.68e-11, "max_component": 2.48e-11},
                ],
                "vacuum_grid_exceeded_count": 0,
            }
        )
    )

    rows = summary.rows_from_source(sidecar)

    assert len(rows) == 1
    row = rows[0]
    assert row["case"] == "copied_active_payload"
    assert row["backend"] == "vmec2000_mgrid"
    assert row["status"] == "running_partial"
    assert row["max_iter"] == 24000
    assert row["requested_ftol"] == pytest.approx(1.0e-12)
    assert row["final_iter"] == 5200
    assert row["final_max_component"] == pytest.approx(2.48e-11)
    assert row["strict_gap"] == pytest.approx(24.8)
    assert row["remaining_iterations"] == 18800


def test_square_coil_profile_summary_reports_vmec2000_tail_plateau_from_sidecar(tmp_path: Path):
    case_dir = tmp_path / "square_coil_freeb_backend_profile_plateau_sidecar"
    case_dir.mkdir()
    partial = case_dir / "_partial_vmec2000_payload.json"
    partial.write_text(
        json.dumps(
            {
                "stage_summaries": [{"ns": 17, "niter": 17, "ftolv": 1.0e-10}],
                "last_row": {"it": 13, "fsqr": 2.53e-10, "fsqz": 2.42e-10, "fsql": 6.8e-11},
                "min_total": 5.60e-10,
                "tail_rows": [
                    {"it": 10, "total": 5.60e-10, "max_component": 2.50e-10},
                    {"it": 11, "total": 5.61e-10, "max_component": 2.51e-10},
                    {"it": 12, "total": 5.62e-10, "max_component": 2.52e-10},
                    {"it": 13, "total": 5.63e-10, "max_component": 2.53e-10},
                ],
            }
        )
    )

    row = summary.rows_from_source(partial)[0]

    assert row["tail_plateau_status"] == "flat_above_stage_ftol"
    assert row["tail_plateau_window"] == 4
    assert row["tail_last_over_min"] == pytest.approx(5.63e-10 / 5.60e-10)
    assert row["tail_total_rel_span"] == pytest.approx((5.63e-10 - 5.60e-10) / 5.60e-10)
    assert row["strict_gap"] == pytest.approx(2.53)
    assert row["max_iter"] == 17
    assert row["remaining_iterations"] == 4
    assert row["next_action"] == "let_current_run_finish_then_scan_delt_or_stage_budget"


def test_square_coil_profile_summary_prioritizes_vacuum_grid_over_tail(tmp_path: Path):
    case_dir = tmp_path / "square_coil_freeb_backend_profile_grid_exceeded_ns17_niter24k"
    case_dir.mkdir()
    partial = case_dir / "_partial_vmec2000_payload.json"
    partial.write_text(
        json.dumps(
            {
                "status": "running_partial",
                "progress_phase": "force_iterations",
                "force_rows_started": True,
                "vacuum_grid_exceeded_count": 2,
                "stage_summaries": [{"ns": 17, "ftolv": 1.0e-12}],
                "last_row": {"it": 40, "fsqr": 2.0e-10, "fsqz": 1.0e-10, "fsql": 1.0e-11},
                "tail_rows": [
                    {"it": 38, "total": 3.0e-10, "max_component": 2.0e-10},
                    {"it": 39, "total": 3.1e-10, "max_component": 2.0e-10},
                    {"it": 40, "total": 3.2e-10, "max_component": 2.0e-10},
                ],
            }
        )
    )

    row = summary.rows_from_source(partial)[0]

    assert row["vacuum_grid_exceeded_count"] == 2
    assert row["next_action"] == "widen_mgrid_before_interpreting_residual"


def test_square_coil_profile_summary_labels_vmec2000_startup_before_force_rows(tmp_path: Path):
    case_dir = tmp_path / "square_coil_freeb_backend_profile_vmec2000_ns17_mpol8_ntor32_nzeta72"
    workdir = case_dir / "vmec2000_mgrid"
    workdir.mkdir(parents=True)
    threed1 = workdir / "threed1.square_beta_00p000_mgrid"
    threed1.write_text(
        "\n".join(
            [
                " THIS IS PARVMEC (PARALLEL VMEC), VERSION 9.0",
                " COMPUTATION PARAMETERS: (u = theta, v = zeta)",
                "     ns     nu     nv     mu     mv",
                "     17     22     72      8     32",
                " R-Z FOURIER BOUNDARY COEFFICIENTS AND MAGNETIC AXIS INITIAL GUESS",
                "   nb  mb     rbc         rbs         zbc         zbs",
                "    0   0  1.6050E+00  0.0000E+00  0.0000E+00  0.0000E+00",
            ]
        )
        + "\n"
    )

    row = summary.rows_from_source(threed1)[0]

    assert row["progress_phase"] == "startup_or_pre_iteration_output"
    assert row["force_rows_started"] is False
    assert row["threed1_size_bytes"] > 0
    assert row["final_iter"] is None
    assert row["final_total"] is None


def test_square_coil_profile_summary_enriches_legacy_partial_sidecar_from_threed1(tmp_path: Path):
    case_dir = tmp_path / "square_coil_freeb_backend_profile_legacy_partial"
    workdir = case_dir / "vmec2000_mgrid"
    workdir.mkdir(parents=True)
    partial = case_dir / "_partial_vmec2000_payload.json"
    partial.write_text(json.dumps({"iteration_row_count": 0, "stage_summaries": []}))
    (workdir / "threed1.square_beta_00p000_mgrid").write_text(
        " R-Z FOURIER BOUNDARY COEFFICIENTS AND MAGNETIC AXIS INITIAL GUESS\n"
    )

    row = summary.rows_from_source(partial)[0]

    assert row["progress_phase"] == "startup_or_pre_iteration_output"
    assert row["force_rows_started"] is False
    assert row["threed1_size_bytes"] > 0


def test_square_coil_profile_summary_script_uses_repo_local_vmec_parser(tmp_path: Path):
    """Running the script path should not import a stale installed vmec_jax."""

    stale_pkg = tmp_path / "stale_pkg" / "vmec_jax"
    stale_pkg.mkdir(parents=True)
    (stale_pkg / "__init__.py").write_text("")
    (stale_pkg / "vmec2000_exec.py").write_text(
        "raise RuntimeError('stale vmec_jax.vmec2000_exec imported')\n"
    )

    case_dir = tmp_path / "square_coil_freeb_backend_profile_vmec2000_ns17_mpol7_ntor28_nzeta64_niter24k"
    workdir = case_dir / "vmec2000_mgrid"
    workdir.mkdir(parents=True)
    (workdir / "threed1.square_beta_00p000_mgrid").write_text(
        "\n".join(
            [
                " NS =   17 NO. FOURIER MODES =  371 FTOLV =  1.000E-12 NITER =  24000",
                " ITER    FSQR      FSQZ      FSQL      fsqr      fsqz      fsql      DELT",
                "  200   4.00E-11  2.00E-11  1.00E-11  1.00E-12  2.00E-12  3.00E-12  2.00E-02",
            ]
        )
        + "\n"
    )

    script = Path(__file__).resolve().parents[1] / "tools" / "diagnostics" / "summarize_square_coil_profiles.py"
    env = {**os.environ, "PYTHONPATH": str(stale_pkg.parent)}
    proc = subprocess.run(
        [sys.executable, str(script), str(case_dir), "--markdown"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )

    assert "vmec2000_ns17_mpol7_ntor28_nzeta64_niter24k" in proc.stdout
    assert "4e-11" in proc.stdout


def test_square_coil_profile_summary_parses_live_direct_launcher_log(tmp_path: Path):
    case_dir = tmp_path / "square_coil_freeb_backend_profile_direct_chunked_verbose_ns9_13_17_mpol5_ntor28_nzeta64_niter12k"
    case_dir.mkdir()
    launcher_log = case_dir / "launcher.log"
    launcher_log.write_text(
        "\n".join(
            [
                "[square-coil-profile] running vmec_jax direct-coil backend",
                "  NS =    9 NO. FOURIER MODES =  257 FTOLV =  1.000E-08 NITER =   4000",
                "  ITER    FSQR      FSQZ      FSQL    RAX(v=0)    DELT       WMHD",
                " INITIAL JACOBIAN CHANGED SIGN!",
                "    1  5.57E-03  1.14E-03  3.77E-03  1.500E+00  2.00E-02  2.5837E+00",
                "    2  4.00E-03  9.00E-04  2.00E-03  1.500E+00  2.00E-02  2.5836E+00",
                "    3  2.00E-03  5.00E-04  1.00E-03  1.500E+00  2.00E-02  2.5835E+00",
                "  VACUUM PRESSURE TURNED ON AT    3 ITERATIONS",
            ]
        )
    )

    assert summary._profile_paths([case_dir]) == [launcher_log]
    rows = summary.rows_from_source(launcher_log)

    assert len(rows) == 1
    row = rows[0]
    assert row["case"] == "direct_chunked_verbose_ns9_13_17_mpol5_ntor28_nzeta64_niter12k"
    assert row["backend"] == "vmec_jax_direct_live"
    assert row["status"] == "running_partial"
    assert row["progress_phase"] == "force_iterations"
    assert row["force_rows_started"] is True
    assert row["launcher_log_size_bytes"] > 0
    assert row["launcher_log_mtime_unix_s"] > 0.0
    assert row["initial_jacobian_changed_sign"] is True
    assert row["vacuum_pressure_turn_on_iter"] == 3
    assert row["mpol"] == 5
    assert row["ntor"] == 28
    assert row["ns"] == 17
    assert row["nzeta"] == 64
    assert row["max_iter"] == 12000
    assert row["requested_ftol"] == pytest.approx(1.0e-8)
    assert row["final_iter"] == 3
    assert row["final_total"] == pytest.approx(3.5e-3)
    assert row["final_max_component"] == pytest.approx(2.0e-3)
    assert row["best_total"] == pytest.approx(3.5e-3)
    assert row["strict_components_met"] is False
    assert row["tail_decay_factor"] is not None
    assert row["iters_to_1e-12_est"] is not None


def test_square_coil_profile_summary_reports_multistage_budget(tmp_path: Path):
    case_dir = tmp_path / "square_coil_direct_gpu_ns9_13_17_mpol5_ntor28_nzeta64_niter8k"
    case_dir.mkdir()
    launcher_log = case_dir / "launcher.log"
    launcher_log.write_text(
        "\n".join(
            [
                "[square-coil-profile] running vmec_jax direct-coil backend",
                "  NS =    9 NO. FOURIER MODES =  257 FTOLV =  1.000E-08 NITER =   1000",
                "  ITER    FSQR      FSQZ      FSQL    RAX(v=0)    DELT       WMHD",
                "    1  1.00E-03  1.00E-03  1.00E-04  1.500E+00  2.00E-02  2.5837E+00",
                "  NS =   13 NO. FOURIER MODES =  257 FTOLV =  1.000E-10 NITER =   2000",
                "  ITER    FSQR      FSQZ      FSQL    RAX(v=0)    DELT       WMHD",
                "    1  1.00E-06  1.00E-06  1.00E-07  1.500E+00  2.00E-02  2.5837E+00",
                "  NS =   17 NO. FOURIER MODES =  257 FTOLV =  1.000E-12 NITER =   8000",
                "  ITER    FSQR      FSQZ      FSQL    RAX(v=0)    DELT       WMHD",
                "    1  1.00E-08  8.00E-09  1.00E-09  1.500E+00  2.00E-02  2.5837E+00",
                "    2  9.00E-09  7.00E-09  9.00E-10  1.500E+00  2.00E-02  2.5837E+00",
            ]
        )
        + "\n"
    )

    row = summary.rows_from_source(launcher_log)[0]

    assert row["stage_count"] == 3
    assert row["stage_ns_array"] == "9,13,17"
    assert row["stage_niter_array"] == "1000,2000,8000"
    assert row["stage_ftol_array"] == "1e-08,1e-10,1e-12"
    assert row["stage_budget_total"] == 11000
    assert row["stage_budget_final"] == 8000
    assert row["current_stage_index"] == 2
    assert row["current_stage_niter"] == 8000
    assert row["current_stage_ftol"] == pytest.approx(1.0e-12)
    assert row["current_stage_last_iter"] == 2
    assert row["current_stage_iteration_row_count"] == 2
    assert row["remaining_stage_budget"] == 7998
    assert row["remaining_total_stage_budget"] == 7998


def test_square_coil_profile_summary_labels_live_mgrid_launcher_log(tmp_path: Path):
    case_dir = tmp_path / "square_coil_freeb_backend_profile_jax_mgrid_verbose_ns9_mpol5_ntor28_nzeta64"
    case_dir.mkdir()
    launcher_log = case_dir / "launcher.log"
    launcher_log.write_text(
        "\n".join(
            [
                "[square-coil-profile] running vmec_jax generated-mgrid backend",
                "  NS =    9 NO. FOURIER MODES =  257 FTOLV =  1.000E-08 NITER =   1200",
                "  ITER    FSQR      FSQZ      FSQL    RAX(v=0)    DELT       WMHD",
                "    1  5.57E-03  1.14E-03  3.77E-03  1.500E+00  2.00E-02  2.5837E+00",
            ]
        )
    )

    row = summary.rows_from_source(launcher_log)[0]

    assert row["backend"] == "vmec_jax_mgrid_live"


def test_square_coil_profile_summary_parses_direct_gpu_launcher_dir_name(tmp_path: Path):
    case_dir = (
        tmp_path
        / "square_coil_direct_gpu_ns9_13_17_mpol5_ntor28_nzeta64_niter8k_control_spline_baseline"
    )
    case_dir.mkdir()
    launcher_log = case_dir / "launcher.log"
    launcher_log.write_text(
        "\n".join(
            [
                "[square-coil-profile] running vmec_jax direct-coil backend",
                "  NS =    9 NO. FOURIER MODES =  257 FTOLV =  1.000E-12 NITER =   8000",
                "  ITER    FSQR      FSQZ      FSQL    RAX(v=0)    DELT       WMHD",
                " 1059  5.37E-08  3.57E-08  4.06E-09  1.458E+00  2.00E-02  3.2171E-01",
            ]
        )
    )

    row = summary.rows_from_source(launcher_log)[0]

    assert row["case"] == "ns9_13_17_mpol5_ntor28_nzeta64_niter8k_control_spline_baseline"
    assert row["backend"] == "vmec_jax_direct_live"
    assert row["mpol"] == 5
    assert row["ntor"] == 28
    assert row["ns"] == 17
    assert row["nzeta"] == 64
    assert row["max_iter"] == 8000
    assert row["requested_ftol"] == pytest.approx(1.0e-12)
    assert row["final_iter"] == 1059
    assert row["final_max_component"] == pytest.approx(5.37e-8)
    assert row["strict_gap"] == pytest.approx(5.37e4)


def test_square_coil_profile_summary_marks_live_direct_log_before_force_rows(tmp_path: Path):
    case_dir = tmp_path / "square_coil_freeb_backend_profile_direct_mpol6_ntor23"
    case_dir.mkdir()
    launcher_log = case_dir / "launcher.log"
    launcher_log.write_text(
        "\n".join(
            [
                "[square-coil-profile] running vmec_jax direct-coil backend",
                "  NS =    9 NO. FOURIER MODES =  257 FTOLV =  1.000E-08 NITER =   4000",
                " INITIAL JACOBIAN CHANGED SIGN!",
                " TRYING TO IMPROVE INITIAL MAGNETIC AXIS GUESS",
            ]
        )
    )

    row = summary.rows_from_source(launcher_log)[0]

    assert row["status"] == "running_partial"
    assert row["progress_phase"] == "axis_repair_or_pre_iteration_output"
    assert row["force_rows_started"] is False
    assert row["initial_jacobian_changed_sign"] is True
    assert row["final_iter"] is None
    assert row["final_total"] is None
