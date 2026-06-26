from __future__ import annotations

import json
from pathlib import Path

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
                        "free_boundary_anderson_pressure_enabled": True,
                        "free_boundary_anderson_pressure_last_theta": 0.25,
                        "history": {
                            "dt_eff_stats": {"last": 0.02, "min": 0.01},
                            "time_step_stats": {"last": 0.019},
                            "freeb_full_update_stats": {"sum": 997.0},
                            "freeb_anderson_pressure_applied_stats": {"sum": 12.0},
                            "bad_jacobian_stats": {"sum": 1.0},
                            "freeb_nestor_bnormal_rms_stats": {"last": 4.0e-3, "min": 3.0e-3},
                            "fsq_component_sum_tail_projection": {
                                "per_iter_factor": 0.98,
                                "estimated_additional_iterations_to_target": {"1e-12": 1234},
                            },
                        },
                        "wall_s": 3.5,
                    },
                    "vmec2000_mgrid": {
                        "status": "completed",
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
    assert rows[0]["final_total"] == pytest.approx(5.4e-6)
    assert rows[0]["requested_ftol"] == pytest.approx(1.0e-6)
    assert rows[0]["final_max_component"] == pytest.approx(3.0e-6)
    assert rows[0]["strict_components_met"] is False
    assert rows[0]["best_total"] == pytest.approx(5.0e-6)
    assert rows[1]["final_total"] == pytest.approx(9.0e-7)
    assert rows[1]["final_max_component"] == pytest.approx(4.0e-7)
    assert rows[1]["strict_components_met"] is True
    assert rows[1]["best_total"] == pytest.approx(8.0e-7)
    assert rows[1]["solver_mode"] == "parity"
    assert rows[1]["nzeta_auto"] is True
    assert rows[1]["recommended_nzeta"] == 32
    assert rows[1]["boundary_mode_count"] == 65
    assert rows[1]["boundary_recommended_nzeta"] == 32
    assert rows[1]["boundary_proj_max"] == pytest.approx(1.2e-4)
    assert rows[1]["boundary_proj_rel"] == pytest.approx(4.5e-4)
    assert rows[1]["dt_eff_last"] == pytest.approx(0.02)
    assert rows[1]["dt_eff_min"] == pytest.approx(0.01)
    assert rows[1]["time_step_last"] == pytest.approx(0.019)
    assert rows[1]["freeb_full_update_count"] == pytest.approx(997.0)
    assert rows[1]["anderson_pressure_enabled"] is True
    assert rows[1]["anderson_pressure_applied_count"] == pytest.approx(12.0)
    assert rows[1]["anderson_pressure_last_theta"] == pytest.approx(0.25)
    assert rows[1]["bad_jacobian_count"] == pytest.approx(1.0)
    assert rows[1]["bnormal_rms_last"] == pytest.approx(4.0e-3)
    assert rows[1]["bnormal_rms_min"] == pytest.approx(3.0e-3)
    assert rows[1]["tail_decay_factor"] == pytest.approx(0.98)
    assert rows[1]["iters_to_1e-12_est"] == pytest.approx(1234)
    assert rows[0]["vacuum_grid_exceeded_count"] == 2
