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
    assert rows[0]["final_total"] == pytest.approx(5.4e-6)
    assert rows[0]["requested_ftol"] == pytest.approx(1.0e-6)
    assert rows[0]["final_max_component"] == pytest.approx(3.0e-6)
    assert rows[0]["strict_components_met"] is False
    assert rows[0]["best_total"] == pytest.approx(5.0e-6)
    assert rows[0]["tail_decay_factor"] < 1.0
    assert rows[0]["iters_to_1e-12_est"] > 0
    assert rows[1]["final_total"] == pytest.approx(9.0e-7)
    assert rows[1]["final_max_component"] == pytest.approx(4.0e-7)
    assert rows[1]["strict_components_met"] is True
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
    assert rows[1]["tail_decay_factor"] == pytest.approx(0.98)
    assert rows[1]["iters_to_1e-12_est"] == pytest.approx(1234)
    assert rows[0]["vacuum_grid_exceeded_count"] == 2


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
    assert row["strict_components_met"] is False
    assert row["tail_decay_factor"] == pytest.approx(0.9994733361578259)
    assert row["iters_to_1e-12_est"] == pytest.approx(7865)


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
