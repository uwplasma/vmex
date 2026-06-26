from __future__ import annotations

import json
import os
import sys
import types
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from tools.diagnostics import profile_square_coil_free_boundary as profile
from vmec_jax.config import VMECConfig
from vmec_jax.namelist import InData
from vmec_jax.state import StateLayout, VMECState
from vmec_jax.static import build_static
from vmec_jax.toroidal_hybrid import SquareAxisControlFourierMatrix


def test_square_coil_profile_parser_accepts_control_spline_axis_kind(tmp_path: Path):
    args = profile._parser().parse_args(
        [
            "--outdir",
            str(tmp_path),
            "--axis-kind",
            "control_spline",
            "--verbose-solver",
            "--virtual-casing-diagnostics",
            "--virtual-casing-quad-factor",
            "3",
            "--virtual-casing-chunk-size",
            "128",
            "--virtual-casing-target-chunk-size",
            "none",
            "--accepted-provider-parity",
            "--freeb-jax-nestor-operator",
            "--no-freeb-jax-nestor-jit-operator",
            "--freeb-include-edge",
            "--freeb-dense-solve-mode",
            "grid",
            "--no-freeb-experimental-fouri-matrix",
            "--freeb-add-analytic-bvec",
            "--resolution-diagnostics-only",
        ]
    )

    assert args.axis_kind == "control_spline"
    assert args.ftol == pytest.approx(profile.ExampleConfig().ftol)
    assert args.delt == pytest.approx(profile.ExampleConfig().delt)
    assert args.verbose_solver is True
    assert args.virtual_casing_diagnostics is True
    assert args.virtual_casing_quad_factor == 3
    assert args.virtual_casing_chunk_size == 128
    assert args.virtual_casing_target_chunk_size is None
    assert args.accepted_provider_parity is True
    assert args.freeb_jax_nestor_operator is True
    assert args.freeb_jax_nestor_jit_operator is False
    assert args.freeb_include_edge is True
    assert args.freeb_dense_solve_mode == "grid"
    assert args.freeb_experimental_fouri_matrix is False
    assert args.freeb_add_analytic_bvec is True
    assert args.resolution_diagnostics_only is True


def test_square_coil_profile_residual_payload_keeps_solver_mode_and_history_tails():
    diagnostics = {
        "solver_mode": "parity",
        "use_scan": True,
        "performance_mode": False,
        "converged": False,
        "converged_strict": False,
        "requested_ftol": 1.0e-12,
        "final_fsqr": 1.0e-5,
        "final_fsqz": 2.0e-5,
        "final_fsql": 3.0e-6,
        "bad_resets": 0,
        "ijacob": 1,
        "free_boundary": {
            "nestor_model": "vmec2000_like_dense_integral",
            "couple_edge": True,
            "activate_fsq": 1.0e-3,
            "ivac": 3,
            "ivacskip": 0,
            "nvacskip": 2,
            "anderson_pressure_enabled": True,
            "last_nestor_diagnostics": {
                "bnormal_rms": 4.0e-4,
                "bsqvac_rms": 1.5e-2,
            },
        },
        "freeb_ivac_history": np.array([1, 2, 3]),
        "freeb_ivacskip_history": np.array([0, 1, 0]),
        "freeb_full_update_history": np.array([1, 0, 1]),
        "freeb_nestor_reused_history": np.array([0, 1, 0]),
        "freeb_nestor_source_reused_history": np.array([0, 1, 0]),
        "freeb_nestor_provider_allows_source_reuse_history": np.array([1, 1, 1]),
        "freeb_nestor_solve_time_history": np.array([0.3, 0.1, 0.2]),
        "freeb_nestor_sample_time_history": np.array([0.8, 0.7, 0.6]),
        "freeb_nestor_trial_reused_history": np.array([0, 0, 1]),
        "freeb_nestor_trial_failed_history": np.array([0, 1, 0]),
        "freeb_nestor_trial_solve_time_history": np.array([0.4, 0.0, 0.2]),
        "freeb_nestor_trial_sample_time_history": np.array([1.1, 0.0, 0.9]),
        "freeb_nestor_bnormal_rms_history": np.array([1.0e-3, 7.0e-4, 4.0e-4]),
        "freeb_anderson_pressure_applied_history": np.array([0, 0, 1]),
        "freeb_anderson_pressure_theta_history": np.array([np.nan, np.nan, 0.4]),
        "freeb_anderson_pressure_residual_norm_history": np.array([1.0, 0.8, 0.6]),
        "include_edge_history": np.array([0, 1, 1]),
        "bad_jacobian_history": np.array([0, 0, 0]),
        "time_step_history": np.array([0.05, 0.05, 0.04]),
        "dt_eff_history": np.array([0.05, 0.05, 0.04]),
        "update_rms_history": np.array([1.0e-2, 4.0e-3, 2.0e-3]),
    }
    result = SimpleNamespace(
        n_iter=3,
        diagnostics=diagnostics,
        w_history=np.array([1.0, 0.5, 0.25, 0.125]),
        fsqr2_history=np.array([1.0e-3, 1.0e-4, 1.0e-5]),
        fsqz2_history=np.array([2.0e-3, 2.0e-4, 2.0e-5]),
        fsql2_history=np.array([3.0e-4, 3.0e-5, 3.0e-6]),
    )
    run = SimpleNamespace(result=result)

    payload = profile._final_residuals(run)

    assert payload["solver_mode"] == "parity"
    assert payload["use_scan"] is True
    assert payload["free_boundary_active"] is True
    assert payload["free_boundary_anderson_pressure_enabled"] is True
    assert payload["free_boundary_anderson_pressure_last_applied"] == pytest.approx(1.0)
    assert payload["free_boundary_anderson_pressure_last_theta"] == pytest.approx(0.4)
    assert payload["final_fsq_component_sum"] == pytest.approx(3.3e-5)
    assert payload["history"]["fsq_component_sum_tail"] == pytest.approx([0.0033, 0.00033, 3.3e-5])
    assert payload["history"]["fsq_component_sum_stats"]["min"] == pytest.approx(3.3e-5)
    assert payload["history"]["fsq_component_sum_tail_projection"]["per_iter_factor"] == pytest.approx(0.1)
    assert payload["history"]["fsq_component_sum_tail_projection"][
        "estimated_additional_iterations_to_target"
    ]["1e-12"] == 8
    assert payload["history"]["freeb_full_update_stats"]["sum"] == pytest.approx(2.0)
    assert payload["history"]["freeb_nestor_source_reused_stats"]["sum"] == pytest.approx(1.0)
    assert payload["history"]["freeb_nestor_provider_allows_source_reuse_stats"]["last"] == pytest.approx(1.0)
    assert payload["history"]["freeb_nestor_sample_time_stats"]["mean"] == pytest.approx(0.7)
    assert payload["history"]["freeb_nestor_solve_time_stats"]["max"] == pytest.approx(0.3)
    assert payload["history"]["freeb_nestor_trial_reused_stats"]["sum"] == pytest.approx(1.0)
    assert payload["history"]["freeb_nestor_trial_failed_stats"]["sum"] == pytest.approx(1.0)
    assert payload["history"]["freeb_nestor_trial_sample_time_stats"]["max"] == pytest.approx(1.1)
    assert payload["history"]["freeb_nestor_trial_solve_time_stats"]["mean"] == pytest.approx(0.2)
    assert payload["history"]["freeb_anderson_pressure_applied_stats"]["sum"] == pytest.approx(1.0)
    assert payload["history"]["freeb_anderson_pressure_theta_tail"][-1] == pytest.approx(0.4)
    assert payload["history"]["bad_jacobian_stats"]["nonzero_count"] == 0
    assert payload["history"]["time_step_stats"]["last"] == pytest.approx(0.04)
    assert payload["history"]["update_rms_stats"]["max"] == pytest.approx(1.0e-2)
    assert payload["history"]["freeb_ivac_tail"] == [1, 2, 3]
    assert payload["history"]["include_edge_tail"] == [0, 1, 1]


def test_square_coil_profile_tail_decay_projection_estimates_remaining_iterations():
    values = np.asarray([1.0e-6, 3.0e-7, 9.0e-8, 2.7e-8])

    projection = profile._tail_decay_projection(values, length=4, targets=(1.0e-8, 1.0e-12))

    assert projection["window"] == 4
    assert projection["monotone_decrease_fraction"] == pytest.approx(1.0)
    assert projection["per_iter_factor"] == pytest.approx(0.3)
    assert projection["estimated_additional_iterations_to_target"]["1e-08"] == 1
    assert projection["estimated_additional_iterations_to_target"]["1e-12"] == 9


def test_square_coil_profile_provider_parity_stats_reports_compact_thresholds():
    config = SimpleNamespace(nzeta=4)
    bounds = {"rmin": 0.0, "rmax": 4.0, "zmin": -1.0, "zmax": 1.0}
    mgrid = SimpleNamespace(
        R=np.ones((2, 2)),
        Z=np.zeros((2, 2)),
        br_mgrid=np.ones((2, 2)),
        bp_mgrid=np.ones((2, 2)) * 2.0,
        bz_mgrid=np.ones((2, 2)) * 3.0,
        vac_ext=SimpleNamespace(
            bnormal=np.ones((2, 2)),
            bnormal_unit=np.ones((2, 2)),
            bu=np.ones((2, 2)) * 2.0,
            bv=np.ones((2, 2)) * 3.0,
            bsqvac=np.ones((2, 2)) * 4.0,
        ),
    )
    direct = SimpleNamespace(
        R=np.ones((2, 2)),
        Z=np.zeros((2, 2)),
        br_mgrid=np.ones((2, 2)) * 1.01,
        bp_mgrid=np.ones((2, 2)) * 2.02,
        bz_mgrid=np.ones((2, 2)) * 3.03,
        vac_ext=SimpleNamespace(
            bnormal=np.ones((2, 2)) * 1.02,
            bnormal_unit=np.ones((2, 2)) * 1.02,
            bu=np.ones((2, 2)) * 2.04,
            bv=np.ones((2, 2)) * 3.06,
            bsqvac=np.ones((2, 2)) * 4.08,
        ),
    )

    payload = profile._provider_parity_stats(
        status="completed",
        reference_provider="direct_coils",
        candidate_provider="generated_mgrid",
        sample="accepted_boundary_direct_backend",
        wall_s=1.25,
        mgrid_sample=mgrid,
        direct_sample=direct,
        config=config,
        bounds=bounds,
        mgrid_nphi=8,
    )

    assert payload["sample"] == "accepted_boundary_direct_backend"
    assert payload["mgrid_kp_divisible_by_nzeta"] is True
    assert payload["field_vector"]["diff_rms_rel"] == pytest.approx(0.01 / 1.01)
    assert payload["vacuum_channels"]["bnormal"]["diff_rms_rel"] == pytest.approx(0.02 / 1.02)
    assert payload["field_rms_rel_lt_5pct"] is True
    assert payload["bnormal_rms_rel_lt_10pct"] is True


def test_square_coil_profile_vmec2000_tail_plateau_classifies_flat_above_stage_ftol():
    rows = [
        {"it": 10, "total": 5.60e-10, "max_component": 2.50e-10},
        {"it": 11, "total": 5.61e-10, "max_component": 2.51e-10},
        {"it": 12, "total": 5.62e-10, "max_component": 2.52e-10},
        {"it": 13, "total": 5.63e-10, "max_component": 2.53e-10},
    ]

    plateau = profile._vmec2000_tail_plateau_payload(rows, stage_ftol=1.0e-10)

    assert plateau["status"] == "flat_above_stage_ftol"
    assert plateau["window"] == 4
    assert plateau["total_last_over_min"] == pytest.approx(5.63e-10 / 5.60e-10)
    assert plateau["max_component_last"] == pytest.approx(2.53e-10)


def test_square_coil_profile_boundary_motion_payload_measures_edge_displacement():
    cfg = VMECConfig(
        mpol=2,
        ntor=0,
        ns=3,
        nfp=1,
        lasym=False,
        lthreed=False,
        lconm1=True,
        ntheta=8,
        nzeta=1,
    )
    static = build_static(cfg)
    indata = InData(
        scalars={
            "MPOL": 2,
            "NTOR": 0,
            "NS_ARRAY": [3],
            "NFP": 1,
            "LASYM": False,
            "LCONM1": True,
        },
        indexed={"RBC": {(0, 0): 3.0}},
    )
    layout = StateLayout(ns=3, K=static.modes.K, lasym=False)
    zeros = np.zeros((3, static.modes.K), dtype=float)
    rcos = zeros.copy()
    rcos[:, 0] = 3.0
    rcos[-1, 0] += 0.1
    state = VMECState(
        layout=layout,
        Rcos=rcos,
        Rsin=zeros.copy(),
        Zcos=zeros.copy(),
        Zsin=zeros.copy(),
        Lcos=zeros.copy(),
        Lsin=zeros.copy(),
    )

    payload = profile._boundary_motion_payload(
        SimpleNamespace(state=state, static=static, indata=indata)
    )

    assert payload is not None
    assert payload["boundary_coeff_delta_l2"] == pytest.approx(0.1)
    assert payload["boundary_coeff_delta_linf"] == pytest.approx(0.1)
    assert payload["boundary_coeff_delta_rel"] == pytest.approx(0.1 / 3.0)
    assert payload["boundary_sample_displacement_rms"] == pytest.approx(0.1)
    assert payload["boundary_sample_displacement_max"] == pytest.approx(0.1)
    assert payload["boundary_sample_displacement_rel"] == pytest.approx(0.1 / 3.0)


def test_square_coil_profile_boundary_reduced_control_projection_payload(monkeypatch):
    basis = SimpleNamespace(symmetry="square", labels=["side", "corner"])
    matrix = SquareAxisControlFourierMatrix(
        controls=SimpleNamespace(),
        m=np.asarray([0]),
        n=np.asarray([0]),
        R_cos=np.asarray([[1.0, 0.0]]),
        R_sin=np.asarray([[0.0, 2.0]]),
        Z_cos=np.asarray([[0.0, 0.0]]),
        Z_sin=np.asarray([[1.0, 1.0]]),
        control_basis=basis,
    )

    monkeypatch.setattr(
        profile,
        "_square_control_fourier_matrix",
        lambda _config, *, symmetry="square": (basis, matrix),
    )
    config = SimpleNamespace(plasma_axis_kind="control_spline")
    payload = profile._boundary_reduced_control_projection_payload(
        config=config,
        deltas={
            "R_cos": np.asarray([1.0]),
            "R_sin": np.asarray([4.0]),
            "Z_cos": np.asarray([0.0]),
            "Z_sin": np.asarray([3.0]),
        },
    )

    assert payload is not None
    assert payload["status"] == "available"
    assert payload["labels"] == ["side", "corner"]
    assert payload["radius_delta"] == pytest.approx([1.0, 2.0])
    assert payload["radius_delta_by_label"]["side"] == pytest.approx(1.0)
    assert payload["radius_delta_by_label"]["corner"] == pytest.approx(2.0)
    assert payload["residual_rel"] == pytest.approx(0.0, abs=1.0e-14)
    assert payload["captured_fraction"] == pytest.approx(1.0)
    assert payload["candidate_bases"]["square"]["captured_fraction"] == pytest.approx(1.0)
    assert payload["candidate_bases"]["stellarator"]["captured_fraction"] == pytest.approx(1.0)


def test_square_coil_profile_partial_vmec2000_payload_reads_timeout_rows(tmp_path: Path):
    workdir = tmp_path / "vmec2000_mgrid"
    workdir.mkdir()
    (workdir / "threed1.case").write_text(
        "\n".join(
            [
                " NS =    9 NO. FOURIER MODES =  113 FTOLV =  1.000E-08 NITER =   5000",
                " ITER    FSQR      FSQZ      FSQL      fsqr      fsqz      fsql      DELT    RAX(v=0)      WMHD      BETA      <M>   DEL-BSQ   FEDGE",
                "    1   4.00E-03  2.00E-03  1.00E-03  1.00E-04  2.00E-04  3.00E-04  5.00E-02  1.50E+00  3.00E-01  0.000E+00  1.0 1.00E-02 2.00E-03",
                " 'Plasma Boundary exceeded Vacuum Grid Size         '",
                "  200   4.00E-06  2.00E-06  1.00E-06  1.00E-08  2.00E-08  3.00E-08  5.00E-02  1.50E+00  3.00E-01  0.000E+00  1.0 1.00E-02 2.00E-06",
            ]
        )
        + "\n"
    )

    payload = profile._partial_vmec2000_payload(workdir)

    assert payload["progress_phase"] == "force_iterations"
    assert payload["force_rows_started"] is True
    assert payload["threed1_size_bytes"] > 0
    assert payload["threed1_mtime_unix_s"] > 0.0
    assert payload["iteration_row_count"] == 2
    assert payload["last_row"]["it"] == 200
    assert payload["last_row"]["total"] == pytest.approx(7.0e-6)
    assert payload["last_row"]["max_component"] == pytest.approx(4.0e-6)
    assert payload["min_total"] == pytest.approx(7.0e-6)
    assert payload["final_max_component"] == pytest.approx(4.0e-6)
    assert payload["strict_components_met"] is False
    assert payload["tail_plateau"]["status"] == "insufficient_tail"
    assert payload["stage_summaries"][0]["strict_components_met"] is False
    assert payload["vacuum_grid_exceeded_count"] == 1


def test_square_coil_profile_writes_partial_vmec2000_sidecar(tmp_path: Path):
    workdir = tmp_path / "vmec2000_mgrid"
    workdir.mkdir()
    (workdir / "threed1.case").write_text(
        "\n".join(
            [
                " NS =    9 NO. FOURIER MODES =  113 FTOLV =  1.000E-12 NITER =   5000",
                " ITER    FSQR      FSQZ      FSQL      fsqr      fsqz      fsql      DELT",
                "  200   4.00E-11  2.00E-11  1.00E-11  1.00E-12  2.00E-12  3.00E-12  2.00E-02",
            ]
        )
        + "\n"
    )

    path = profile._write_partial_vmec2000_payload(outdir=tmp_path, workdir=workdir)

    data = json.loads(path.read_text())
    assert path.name == "_partial_vmec2000_payload.json"
    assert data["progress_phase"] == "force_iterations"
    assert data["force_rows_started"] is True
    assert data["iteration_row_count"] == 1
    assert data["updated_unix_s"] > 0.0
    assert data["final_max_component"] == pytest.approx(4.0e-11)
    assert data["strict_components_met"] is False
    assert data["tail_plateau"]["stage_ftol"] == pytest.approx(1.0e-12)


def test_square_coil_profile_provider_parity_payload_reports_field_and_bnormal_error(monkeypatch, tmp_path: Path):
    vac_direct = SimpleNamespace(
        bnormal=np.array([[2.0, 4.0], [6.0, 8.0]]),
        bnormal_unit=np.array([[0.2, 0.4], [0.6, 0.8]]),
        bu=np.array([[1.0, 2.0], [3.0, 4.0]]),
        bv=np.array([[1.5, 2.5], [3.5, 4.5]]),
        bsqvac=np.array([[10.0, 11.0], [12.0, 13.0]]),
    )
    vac_mgrid = SimpleNamespace(
        bnormal=vac_direct.bnormal + 0.5,
        bnormal_unit=vac_direct.bnormal_unit + 0.05,
        bu=vac_direct.bu + 0.1,
        bv=vac_direct.bv + 0.1,
        bsqvac=vac_direct.bsqvac + 0.2,
    )
    direct_sample = SimpleNamespace(
        R=np.array([[1.0, 1.1], [1.2, 1.3]]),
        Z=np.array([[-0.1, 0.0], [0.1, 0.2]]),
        br_mgrid=np.ones((2, 2)),
        bp_mgrid=2.0 * np.ones((2, 2)),
        bz_mgrid=3.0 * np.ones((2, 2)),
        vac_ext=vac_direct,
    )
    mgrid_sample = SimpleNamespace(
        R=direct_sample.R,
        Z=direct_sample.Z,
        br_mgrid=direct_sample.br_mgrid + 0.01,
        bp_mgrid=direct_sample.bp_mgrid + 0.02,
        bz_mgrid=direct_sample.bz_mgrid + 0.03,
        vac_ext=vac_mgrid,
    )

    monkeypatch.setattr(
        profile,
        "run_free_boundary",
        lambda *args, **kwargs: SimpleNamespace(state=object(), static=object()),
    )
    monkeypatch.setattr(profile, "build_coil_field_geometry", lambda _params: "geometry")

    def fake_sample(**kwargs):
        return direct_sample if kwargs.get("external_field_provider_kind") == "direct_coils" else mgrid_sample

    monkeypatch.setattr(profile, "_sample_external_boundary_arrays", fake_sample)
    config = SimpleNamespace(nzeta=2)
    coil_params = SimpleNamespace(regularization_epsilon=0.0, chunk_size=None)

    payload = profile._provider_parity_payload(
        mgrid_input=tmp_path / "input.square_mgrid",
        coil_params=coil_params,
        config=config,
        bounds={"rmin": 0.8, "rmax": 1.5, "zmin": -0.3, "zmax": 0.4},
        mgrid_nphi=4,
    )

    assert payload["status"] == "completed"
    assert payload["mgrid_kp_divisible_by_nzeta"] is True
    assert payload["domain"]["contained"] is True
    assert payload["components"]["br_mgrid"]["diff_rms"] == pytest.approx(0.01)
    assert payload["field_vector"]["diff_rms"] == pytest.approx(np.sqrt(0.01**2 + 0.02**2 + 0.03**2))
    assert payload["vacuum_channels"]["bnormal"]["diff_rms"] == pytest.approx(0.5)


def test_square_coil_profile_rejects_mgrid_nphi_not_multiple_of_nzeta(tmp_path: Path):
    with pytest.raises(ValueError, match="mgrid-nphi=.*incompatible.*nzeta"):
        profile.main(
            [
                "--outdir",
                str(tmp_path / "profile"),
                "--nzeta",
                "4",
                "--mgrid-nphi",
                "6",
                "--skip-direct",
                "--skip-mgrid",
                "--skip-provider-parity",
                "--max-boundary-projection-error",
                "none",
            ]
        )


def test_square_coil_profile_records_boundary_projection_payload(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(profile, "write_mgrid_from_coils", lambda *args, **kwargs: None)

    outdir = tmp_path / "profile"
    profile.main(
        [
            "--outdir",
            str(outdir),
            "--mpol",
            "3",
            "--ntor",
            "4",
            "--ns",
            "5",
            "--nzeta",
            "16",
            "--max-iter",
            "2",
            "--nstep",
            "3",
            "--skip-direct",
            "--skip-mgrid",
            "--skip-provider-parity",
            "--max-boundary-projection-error",
            "1.0",
        ]
    )

    report = outdir / "square_coil_free_boundary_backend_profile.json"
    data = json.loads(report.read_text())
    assert data["configuration"]["max_boundary_projection_error"] == pytest.approx(1.0)
    assert data["configuration"]["nstep"] == 3
    projection = data["boundary_projection"]
    assert projection["mpol"] == 3
    assert projection["ntor"] == 4
    assert projection["recommended_nzeta"] == 16
    assert projection["mode_count"] > 0
    assert np.isfinite(float(projection["max_abs_error"]))
    control_basis = data["control_basis"]
    assert control_basis["status"] == "available"
    assert control_basis["control_count"] == 8
    assert control_basis["bases"]["square"]["labels"] == ["side", "corner"]
    assert control_basis["bases"]["square"]["reduced_count"] == 2
    assert control_basis["bases"]["square"]["expansion_matrix_shape"] == [8, 2]
    assert control_basis["bases"]["stellarator"]["reduced_count"] == 5
    control_map = data["control_fourier_map"]
    assert control_map["status"] == "available"
    assert control_map["labels"] == ["side", "corner"]
    assert control_map["control_count"] == 2
    assert control_map["jacobian_shape"] == [4 * projection["mode_count"], 2]
    assert len(control_map["singular_values"]) == 2
    assert control_map["condition_number"] > 0.0
    candidate_maps = control_map["candidate_bases"]
    assert candidate_maps["square"]["control_count"] == 2
    assert candidate_maps["stellarator"]["basis_symmetry"] == "stellarator"
    assert candidate_maps["stellarator"]["control_count"] == 5
    assert candidate_maps["stellarator"]["jacobian_shape"] == [4 * projection["mode_count"], 5]
    spline_bridge = data["spline_bridge"]
    assert spline_bridge["status"] == "spline_control_to_fourier_bridge"
    assert spline_bridge["solver_native_spline_controls"] is False
    assert spline_bridge["requires_fourier_projection"] is True
    assert spline_bridge["reduced_square_control_count"] == 2
    assert spline_bridge["can_reduce_input_shape_dofs"] is True
    assert spline_bridge["can_reduce_nonlinear_solver_dofs"] is False
    deck = data["resolution_deck"]
    assert deck["status"] == "production_ready"
    assert deck["projection_meets_gate"] is True
    assert deck["mgrid_nphi_multiple_of_nzeta"] is True


def test_square_coil_profile_resolution_diagnostics_only_reports_incompatible_mgrid(tmp_path: Path):
    outdir = tmp_path / "profile_resolution_only"

    assert (
        profile.main(
            [
                "--outdir",
                str(outdir),
                "--mpol",
                "3",
                "--ntor",
                "4",
                "--ns",
                "5",
                "--nzeta",
                "16",
                "--mgrid-nphi",
                "24",
                "--max-boundary-projection-error",
                "none",
                "--resolution-diagnostics-only",
            ]
        )
        == 0
    )

    data = json.loads((outdir / "square_coil_free_boundary_backend_profile.json").read_text())
    assert data["configuration"]["resolution_diagnostics_only"] is True
    assert data["backends"] == {}
    assert data["mgrid"]["created"] is False
    deck = data["resolution_deck"]
    assert deck["status"] == "diagnostic_underresolved"
    assert deck["mgrid_nphi_multiple_of_nzeta"] is False
    assert "mgrid_nphi_not_multiple_of_nzeta" in deck["reasons"]
    assert "projection_gate_disabled" in deck["reasons"]


def test_square_coil_profile_projection_gate_fails_before_backend_work(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        profile,
        "_boundary_projection_payload",
        lambda config: {
            "max_abs_component_error": 2.0e-3,
            "max_abs_error": 2.0e-3,
        },
    )
    monkeypatch.setattr(
        profile,
        "recommend_square_axis_stellarator_mirror_hybrid_resolution",
        lambda **kwargs: {
            "recommended": {
                "mpol": 6,
                "ntor": 20,
                "recommended_nzeta": 48,
                "max_abs_component_error": 5.0e-5,
            }
        },
    )

    with pytest.raises(ValueError, match="boundary projection error is too large"):
        profile.main(
            [
                "--outdir",
                str(tmp_path / "profile_projection_gate"),
                "--mpol",
                "3",
                "--ntor",
                "4",
                "--ns",
                "5",
                "--nzeta",
                "16",
                "--max-iter",
                "2",
                "--max-boundary-projection-error",
                "1e-4",
                "--skip-direct",
                "--skip-mgrid",
                "--skip-provider-parity",
            ]
        )


def test_square_coil_profile_production_gate_rejects_underrecommended_nzeta(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setattr(
        profile,
        "_boundary_projection_payload",
        lambda config: {
            "max_abs_component_error": 1.0e-13,
            "max_abs_error": 1.0e-13,
            "rms_error": 1.0e-14,
            "mode_count": 1,
        },
    )

    with pytest.raises(ValueError, match="resolution deck is not production-ready.*nzeta_below"):
        profile.main(
            [
                "--outdir",
                str(tmp_path / "profile_underrecommended_nzeta"),
                "--mpol",
                "5",
                "--ntor",
                "28",
                "--ns",
                "5",
                "--nzeta",
                "32",
                "--mgrid-nphi",
                "32",
                "--max-iter",
                "2",
                "--max-boundary-projection-error",
                "5e-12",
                "--skip-direct",
                "--skip-mgrid",
                "--skip-provider-parity",
            ]
        )


def test_square_coil_profile_projection_gate_disabled_allows_diagnostic_underrecommended_nzeta(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setattr(
        profile,
        "_boundary_projection_payload",
        lambda config: {
            "max_abs_component_error": 1.0e-13,
            "max_abs_error": 1.0e-13,
            "rms_error": 1.0e-14,
            "mode_count": 1,
        },
    )

    outdir = tmp_path / "profile_underrecommended_diagnostic"
    assert (
        profile.main(
            [
                "--outdir",
                str(outdir),
                "--mpol",
                "5",
                "--ntor",
                "28",
                "--ns",
                "5",
                "--nzeta",
                "32",
                "--mgrid-nphi",
                "32",
                "--max-iter",
                "2",
                "--max-boundary-projection-error",
                "none",
                "--skip-direct",
                "--skip-mgrid",
                "--skip-provider-parity",
            ]
        )
        == 0
    )

    data = json.loads((outdir / "square_coil_free_boundary_backend_profile.json").read_text())
    deck = data["resolution_deck"]
    assert deck["status"] == "diagnostic_underresolved"
    assert "projection_gate_disabled" in deck["reasons"]
    assert "nzeta_below_square_axis_recommendation" in deck["reasons"]
    assert data["control_basis"]["bases"]["square"]["reduced_count"] == 2


def test_square_coil_profile_defaults_nzeta_to_square_axis_recommendation(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(profile, "write_mgrid_from_coils", lambda *args, **kwargs: None)

    outdir = tmp_path / "profile_auto_nzeta"
    profile.main(
        [
            "--outdir",
            str(outdir),
            "--mpol",
            "5",
            "--ntor",
            "12",
            "--ns",
            "5",
            "--max-iter",
            "2",
            "--skip-direct",
            "--skip-mgrid",
            "--skip-provider-parity",
            "--max-boundary-projection-error",
            "none",
        ]
    )

    data = json.loads((outdir / "square_coil_free_boundary_backend_profile.json").read_text())
    assert data["configuration"]["nzeta"] == profile.recommended_square_axis_nzeta(12)
    assert data["configuration"]["nzeta_auto"] is True
    assert data["configuration"]["nzeta_underrecommended"] is False


def test_square_coil_profile_records_direct_coil_chunk_size(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(profile, "write_mgrid_from_coils", lambda *args, **kwargs: None)

    outdir = tmp_path / "profile_jit_sampler"
    profile.main(
        [
            "--outdir",
            str(outdir),
            "--mpol",
            "3",
            "--ntor",
            "4",
            "--ns",
            "5",
            "--nzeta",
            "16",
            "--max-iter",
            "2",
            "--coil-chunk-size",
            "0",
            "--skip-direct",
            "--skip-mgrid",
            "--skip-provider-parity",
            "--max-boundary-projection-error",
            "none",
        ]
    )

    data = json.loads((outdir / "square_coil_free_boundary_backend_profile.json").read_text())
    assert data["configuration"]["coil_chunk_size"] is None
    assert data["mgrid"]["created"] is False
    assert data["mgrid"]["write_chunk_size"] is None


def test_square_coil_profile_direct_only_skips_mgrid_write(monkeypatch, tmp_path: Path):
    write_calls = []

    def fake_write_mgrid(*args, **kwargs):
        write_calls.append((args, kwargs))

    monkeypatch.setattr(profile, "write_mgrid_from_coils", fake_write_mgrid)

    outdir = tmp_path / "profile_direct_only"
    profile.main(
        [
            "--outdir",
            str(outdir),
            "--mpol",
            "3",
            "--ntor",
            "4",
            "--ns",
            "5",
            "--nzeta",
            "16",
            "--max-iter",
            "2",
            "--coil-chunk-size",
            "0",
            "--skip-direct",
            "--skip-mgrid",
            "--skip-provider-parity",
            "--max-boundary-projection-error",
            "none",
        ]
    )

    data = json.loads((outdir / "square_coil_free_boundary_backend_profile.json").read_text())
    assert write_calls == []
    assert data["mgrid"]["created"] is False


def test_square_coil_profile_chunks_mgrid_when_direct_sampler_is_jit(monkeypatch, tmp_path: Path):
    captured = {}
    monkeypatch.setattr(profile, "_run_jax_backend", lambda **kwargs: {"status": "skipped_test"})

    def fake_write_mgrid(*args, **kwargs):
        captured["kwargs"] = kwargs

    monkeypatch.setattr(profile, "write_mgrid_from_coils", fake_write_mgrid)

    outdir = tmp_path / "profile_mgrid_chunked"
    profile.main(
        [
            "--outdir",
            str(outdir),
            "--mpol",
            "3",
            "--ntor",
            "4",
            "--ns",
            "5",
            "--nzeta",
            "16",
            "--max-iter",
            "2",
            "--coil-chunk-size",
            "0",
            "--skip-direct",
            "--skip-provider-parity",
            "--max-boundary-projection-error",
            "none",
        ]
    )

    data = json.loads((outdir / "square_coil_free_boundary_backend_profile.json").read_text())
    assert captured["kwargs"]["chunk_size"] == 512
    assert data["mgrid"]["created"] is True
    assert data["mgrid"]["write_chunk_size"] == 512


def test_square_coil_profile_passes_direct_sampler_cache_flags(monkeypatch, tmp_path: Path):
    captured = []
    monkeypatch.setattr(
        profile,
        "_run_jax_backend",
        lambda **kwargs: captured.append(kwargs) or {"status": "skipped_test"},
    )

    outdir = tmp_path / "profile_direct_sampler_flags"
    profile.main(
        [
            "--outdir",
            str(outdir),
            "--mpol",
            "3",
            "--ntor",
            "4",
            "--ns",
            "5",
            "--nzeta",
            "16",
            "--max-iter",
            "2",
            "--coil-chunk-size",
            "0",
            "--side-power",
            "1.25",
            "--corner-power",
            "1.5",
            "--no-direct-static-cache",
            "--jit-direct-sampler",
            "--no-direct-trial-bsqvac-resample",
            "--virtual-casing-diagnostics",
            "--virtual-casing-quad-factor",
            "4",
            "--virtual-casing-chunk-size",
            "64",
            "--virtual-casing-target-chunk-size",
            "128",
            "--freeb-jax-nestor-operator",
            "--no-freeb-jax-nestor-jit-operator",
            "--freeb-include-edge",
            "--freeb-dense-solve-mode",
            "grid",
            "--no-freeb-experimental-fouri-matrix",
            "--freeb-add-analytic-bvec",
            "--skip-mgrid",
            "--skip-provider-parity",
            "--max-boundary-projection-error",
            "none",
        ]
    )

    data = json.loads((outdir / "square_coil_free_boundary_backend_profile.json").read_text())
    assert data["configuration"]["side_power"] == pytest.approx(1.25)
    assert data["configuration"]["corner_power"] == pytest.approx(1.5)
    assert data["configuration"]["direct_static_cache"] is False
    assert data["configuration"]["jit_direct_sampler"] is True
    assert data["configuration"]["direct_trial_bsqvac_resample"] is False
    assert data["configuration"]["virtual_casing_diagnostics"] is True
    assert data["configuration"]["virtual_casing_quad_factor"] == 4
    assert data["configuration"]["virtual_casing_chunk_size"] == 64
    assert data["configuration"]["virtual_casing_target_chunk_size"] == 128
    assert data["configuration"]["freeb_jax_nestor_operator"] is True
    assert data["configuration"]["freeb_jax_nestor_jit_operator"] is False
    assert data["configuration"]["freeb_include_edge"] is True
    assert data["configuration"]["freeb_dense_solve_mode"] == "grid"
    assert data["configuration"]["freeb_experimental_fouri_matrix"] is False
    assert data["configuration"]["freeb_add_analytic_bvec"] is True
    assert captured[0]["config"].side_power == pytest.approx(1.25)
    assert captured[0]["config"].corner_power == pytest.approx(1.5)
    assert captured[0]["direct_static_cache"] is False
    assert captured[0]["jit_direct_sampler"] is True
    assert captured[0]["direct_trial_bsqvac_resample"] is False
    assert captured[0]["virtual_casing_diagnostics"] is True
    assert captured[0]["virtual_casing_quad_factor"] == 4
    assert captured[0]["virtual_casing_chunk_size"] == 64
    assert captured[0]["virtual_casing_target_chunk_size"] == 128
    assert captured[0]["freeb_jax_nestor_operator"] is True
    assert captured[0]["freeb_jax_nestor_jit_operator"] is False
    assert captured[0]["freeb_include_edge"] is True
    assert captured[0]["freeb_dense_solve_mode"] == "grid"
    assert captured[0]["freeb_experimental_fouri_matrix"] is False
    assert captured[0]["freeb_add_analytic_bvec"] is True


def test_square_coil_profile_virtual_casing_payload_uses_cached_geometry(monkeypatch):
    monkeypatch.setitem(sys.modules, "virtual_casing_jax", types.ModuleType("virtual_casing_jax"))
    monkeypatch.setitem(
        sys.modules,
        "virtual_casing_jax.functional",
        types.ModuleType("virtual_casing_jax.functional"),
    )
    captured = {}

    def fake_diagnostics_from_run(*args, **kwargs):
        captured["run"] = args[0]
        captured.update(kwargs)
        return SimpleNamespace(
            external_bnormal_residual_rms=1.0e-9,
            external_bnormal_residual_max=2.0e-9,
            pressure_balance_rms=3.0e-6,
            pressure_balance_max=4.0e-6,
            required_external_b=np.ones((3, 2, 3)),
            target_external_b=2.0 * np.ones((3, 2, 3)),
            external_bnormal_residual=np.zeros((2, 3)),
        )

    monkeypatch.setattr(profile, "virtual_casing_diagnostics_from_run", fake_diagnostics_from_run)
    direct_params = SimpleNamespace()

    payload = profile._virtual_casing_profile_payload(
        run=object(),
        direct_params=direct_params,
        coil_geometry="cached-geometry",
        quad_factor=3,
        chunk_size=64,
        target_chunk_size=128,
    )

    assert payload["status"] == "computed"
    assert payload["quad_factor"] == 3
    assert payload["chunk_size"] == 64
    assert payload["target_chunk_size"] == 128
    assert payload["quad_ntheta"] == 6
    assert payload["quad_nzeta"] == 9
    assert payload["external_bnormal_residual_rms"] == pytest.approx(1.0e-9)
    assert payload["pressure_balance_max"] == pytest.approx(4.0e-6)
    assert payload["required_external_b_rms"] == pytest.approx(np.sqrt(3.0))
    assert payload["target_external_b_rms"] == pytest.approx(np.sqrt(12.0))
    assert payload["ntheta"] == 2
    assert payload["nzeta"] == 3
    assert captured["coil_params"] is direct_params
    assert captured["coil_geometry"] == "cached-geometry"
    assert captured["quad_factor"] == 3
    assert captured["chunk_size"] == 64
    assert captured["target_chunk_size"] == 128


def test_square_coil_profile_run_jax_backend_uses_static_direct_sampler(monkeypatch, tmp_path: Path):
    captured = {}
    geometry = ("gamma", "gamma_dash", "currents")

    monkeypatch.setattr(profile, "build_coil_field_geometry", lambda params: geometry)
    monkeypatch.setattr(profile, "write_wout_from_fixed_boundary_run", lambda *args, **kwargs: None)
    env_names = [
        "VMEC_JAX_RETURN_BEST_SCORED_STATE",
        "VMEC_JAX_FREEB_ANDERSON_PRESSURE",
        "VMEC_JAX_FREEB_JAX_NESTOR_OPERATOR",
        "VMEC_JAX_FREEB_JAX_NESTOR_JIT_OPERATOR",
        "VMEC_JAX_FREEB_INCLUDE_EDGE",
        "VMEC_JAX_FREEB_DENSE_SOLVE_MODE",
        "VMEC_JAX_FREEB_EXPERIMENTAL_FOURI_MATRIX",
        "VMEC_JAX_FREEB_ADD_ANALYTIC_BVEC",
    ]
    for name in env_names:
        monkeypatch.setenv(name, f"previous_{name}")

    def fake_run_free_boundary(*args, **kwargs):
        captured.update(kwargs)
        captured["env"] = {name: os.environ.get(name) for name in env_names}
        result = SimpleNamespace(
            n_iter=1,
            diagnostics={
                "requested_ftol": 1.0e-12,
                "final_fsqr": 1.0e-6,
                "final_fsqz": 2.0e-6,
                "final_fsql": 3.0e-6,
                "free_boundary": {"nestor_model": "vmec2000_like_dense_integral"},
            },
            w_history=[],
            fsqr2_history=[],
            fsqz2_history=[],
            fsql2_history=[],
        )
        return SimpleNamespace(result=result)

    monkeypatch.setattr(profile, "run_free_boundary", fake_run_free_boundary)

    direct_params = SimpleNamespace(chunk_size=None, regularization_epsilon=1.0e-6)
    config = SimpleNamespace(
        use_multigrid_schedule=False,
        max_iter=1,
        jit_forces=True,
        free_boundary_activate_fsq=1.0e99,
    )

    out = profile._run_jax_backend(
        input_path=tmp_path / "input.case",
        wout_path=tmp_path / "wout_case.nc",
        config=config,
        direct_params=direct_params,
        solver_mode="parity",
        return_best_scored_state=False,
        freeb_anderson_pressure=True,
        freeb_jax_nestor_operator=True,
        freeb_jax_nestor_jit_operator=False,
        freeb_include_edge=True,
        freeb_dense_solve_mode="grid",
        freeb_experimental_fouri_matrix=False,
        freeb_add_analytic_bvec=True,
        direct_static_cache=True,
        jit_direct_sampler=True,
        direct_trial_bsqvac_resample=False,
    )

    assert out["status"] == "completed"
    assert out["free_boundary_solver_overrides"]["return_best_scored_state"] is False
    assert out["free_boundary_solver_overrides"]["freeb_anderson_pressure"] is True
    assert out["free_boundary_solver_overrides"]["freeb_jax_nestor_operator"] is True
    assert out["free_boundary_solver_overrides"]["freeb_jax_nestor_jit_operator"] is False
    assert out["free_boundary_solver_overrides"]["freeb_include_edge"] is True
    assert out["free_boundary_solver_overrides"]["freeb_dense_solve_mode"] == "grid"
    assert out["free_boundary_solver_overrides"]["freeb_experimental_fouri_matrix"] is False
    assert out["free_boundary_solver_overrides"]["freeb_add_analytic_bvec"] is True
    assert out["virtual_casing"]["status"] == "disabled"
    assert out["free_boundary_promotion"]["boundary_condition_mode"] == "vacuum_coil_normal"
    assert out["free_boundary_promotion"]["coil_bnormal_role"] == "vacuum_boundary_condition"
    assert out["free_boundary_promotion"]["production_candidate"] is False
    assert "strict_force_components_not_met" in out["free_boundary_promotion"]["promotion_blockers"]
    assert captured["external_field_provider_kind"] == "direct_coils"
    assert captured["external_field_provider_params"] is direct_params
    static = captured["external_field_provider_static"]
    assert static["coil_geometry"] == geometry
    assert static["regularization_epsilon"] == pytest.approx(1.0e-6)
    assert static["chunk_size"] is None
    assert static["jit_sampler"] is True
    assert static["resample_trial_bsqvac"] is False
    assert captured["env"]["VMEC_JAX_RETURN_BEST_SCORED_STATE"] == "0"
    assert captured["env"]["VMEC_JAX_FREEB_ANDERSON_PRESSURE"] == "1"
    assert captured["env"]["VMEC_JAX_FREEB_JAX_NESTOR_OPERATOR"] == "1"
    assert captured["env"]["VMEC_JAX_FREEB_JAX_NESTOR_JIT_OPERATOR"] == "0"
    assert captured["env"]["VMEC_JAX_FREEB_INCLUDE_EDGE"] == "1"
    assert captured["env"]["VMEC_JAX_FREEB_DENSE_SOLVE_MODE"] == "grid"
    assert captured["env"]["VMEC_JAX_FREEB_EXPERIMENTAL_FOURI_MATRIX"] == "0"
    assert captured["env"]["VMEC_JAX_FREEB_ADD_ANALYTIC_BVEC"] == "1"
    for name in env_names:
        assert os.environ.get(name) == f"previous_{name}"
