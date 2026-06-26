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
from vmec_jax.solvers.free_boundary.types import NestorRuntimeState
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
            "--virtual-casing-pythonpath",
            str(tmp_path),
            "--accepted-provider-parity",
            "--jax-hot-restart-count",
            "2",
            "--jax-hot-restart-iters",
            "500",
            "--jax-hot-restart-policy",
            "freeb",
            "--jax-hot-restart-always",
            "--jax-initial-restart-wout",
            str(tmp_path / "seed.nc"),
            "--freeb-jax-nestor-operator",
            "--no-freeb-jax-nestor-jit-operator",
            "--freeb-include-edge",
            "--freeb-dense-solve-mode",
            "grid",
            "--no-freeb-experimental-fouri-matrix",
            "--freeb-add-analytic-bvec",
            "--freeb-edge-control-projection",
            "square",
            "--freeb-edge-control-rcond",
            "1e-10",
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
    assert args.virtual_casing_pythonpath == tmp_path
    assert args.accepted_provider_parity is True
    assert args.jax_hot_restart_count == 2
    assert args.jax_hot_restart_iters == 500
    assert args.jax_hot_restart_policy == "freeb"
    assert args.jax_hot_restart_always is True
    assert args.jax_initial_restart_wout == tmp_path / "seed.nc"
    assert args.freeb_jax_nestor_operator is True
    assert args.freeb_jax_nestor_jit_operator is False
    assert args.freeb_include_edge is True
    assert args.freeb_dense_solve_mode == "grid"
    assert args.freeb_experimental_fouri_matrix is False
    assert args.freeb_add_analytic_bvec is True
    assert args.freeb_edge_control_projection == "square"
    assert args.freeb_edge_control_rcond == pytest.approx(1.0e-10)
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
        "update_delta_rms": 2.0e-4,
        "update_delta_to_velocity_rms_ratio": 0.25,
        "best_scored_component_max": 7.0e-6,
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
                "jax_nestor_operator_applied": True,
                "jax_nestor_operator_reason": "applied",
                "jax_nestor_operator_time_s": 0.125,
                "jax_nestor_operator_jitted": False,
                "jax_nestor_operator_cache_hit": False,
                "matrix_override_applied": True,
                "rhs_mode": "vmec_mode",
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
    assert payload["free_boundary_jax_nestor_operator_applied"] is True
    assert payload["free_boundary_jax_nestor_operator_reason"] == "applied"
    assert payload["free_boundary_jax_nestor_operator_time_s"] == pytest.approx(0.125)
    assert payload["free_boundary_jax_nestor_operator_jitted"] is False
    assert payload["free_boundary_jax_nestor_operator_cache_hit"] is False
    assert payload["free_boundary_nestor_matrix_override_applied"] is True
    assert payload["free_boundary_nestor_rhs_mode"] == "vmec_mode"
    assert payload["free_boundary_anderson_pressure_enabled"] is True
    assert payload["free_boundary_anderson_pressure_last_applied"] == pytest.approx(1.0)
    assert payload["free_boundary_anderson_pressure_last_theta"] == pytest.approx(0.4)
    assert payload["final_fsq_component_sum"] == pytest.approx(3.3e-5)
    assert payload["update_delta_rms"] == pytest.approx(2.0e-4)
    assert payload["update_delta_to_velocity_rms_ratio"] == pytest.approx(0.25)
    assert payload["best_scored_component_max"] == pytest.approx(7.0e-6)
    assert payload["history"]["fsq_component_sum_tail"] == pytest.approx([0.0033, 0.00033, 3.3e-5])
    assert payload["history"]["fsq_component_sum_stats"]["min"] == pytest.approx(3.3e-5)
    assert payload["history"]["fsq_component_sum_tail_projection"]["per_iter_factor"] == pytest.approx(0.1)
    assert payload["history"]["fsq_component_sum_tail_projection"][
        "estimated_additional_iterations_to_target"
    ]["1e-12"] == 8
    component_projection = payload["history"]["fsq_component_tail_projection_by_component"]
    assert payload["history"]["fsq_limiting_component"] == "fsqz"
    assert payload["history"]["fsq_limiting_component_value"] == pytest.approx(2.0e-5)
    assert component_projection["fsqr"]["per_iter_factor"] == pytest.approx(0.1)
    assert component_projection["fsqz"]["estimated_additional_iterations_to_target"]["1e-12"] == 8
    assert component_projection["fsql"]["estimated_additional_iterations_to_target"]["1e-12"] == 7
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
    strict = payload["strict_convergence"]
    assert strict["status"] == "underconverged"
    assert strict["strict_component_target"] == pytest.approx(1.0e-12)
    assert strict["loose_component_target"] == pytest.approx(1.0e-8)
    assert strict["component_max"] == pytest.approx(2.0e-5)
    assert strict["component_sum"] == pytest.approx(3.3e-5)
    assert strict["component_max_over_strict_target"] == pytest.approx(2.0e7)
    assert strict["strict_components_met"] is False
    assert strict["requested_components_met"] is False
    assert "component_max_above_1e-12" in strict["blockers"]
    assert "component_max_above_requested_ftol" in strict["blockers"]


def test_square_coil_profile_tail_decay_projection_estimates_remaining_iterations():
    values = np.asarray([1.0e-6, 3.0e-7, 9.0e-8, 2.7e-8])

    projection = profile._tail_decay_projection(values, length=4, targets=(1.0e-8, 1.0e-12))

    assert projection["window"] == 4
    assert projection["monotone_decrease_fraction"] == pytest.approx(1.0)
    assert projection["per_iter_factor"] == pytest.approx(0.3)
    assert projection["estimated_additional_iterations_to_target"]["1e-08"] == 1
    assert projection["estimated_additional_iterations_to_target"]["1e-12"] == 9


def test_square_coil_profile_rejects_loose_production_schedule():
    schedule = {
        "requested_final_ftol": 1.0e-8,
        "requested_final_ftol_meets_target": False,
        "reasons": ["final_ftol_above_strict_target"],
    }

    with pytest.raises(ValueError, match="production profiles require a final component-wise FTOL"):
        profile._enforce_strict_schedule_gate(schedule=schedule, limit=5.0e-12)

    profile._enforce_strict_schedule_gate(schedule=schedule, limit=None)
    profile._enforce_strict_schedule_gate(
        schedule={"requested_final_ftol": 1.0e-12, "requested_final_ftol_meets_target": True},
        limit=5.0e-12,
    )


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


def test_free_boundary_edge_control_projection_removes_uncontrolled_edge_modes():
    from vmec_jax.solve import (
        _freeb_edge_control_delta_tuple_projection_metrics,
        _freeb_edge_control_state_residual_metrics,
        _prepare_freeb_edge_control_projection,
        _project_freeb_edge_control_delta_tuple,
        _project_freeb_edge_control_state,
    )

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
    state0 = VMECState(
        layout=layout,
        Rcos=zeros.copy(),
        Rsin=zeros.copy(),
        Zcos=zeros.copy(),
        Zsin=zeros.copy(),
        Lcos=zeros.copy(),
        Lsin=zeros.copy(),
    )
    jacobian = np.zeros((4 * static.modes.K, 1), dtype=float)
    jacobian[0, 0] = 1.0
    projection = _prepare_freeb_edge_control_projection(
        {
            "enabled": True,
            "basis_symmetry": "test",
            "labels": ["R00"],
            "control_jacobian": jacobian,
        },
        indata=indata,
        static=static,
        state0=state0,
        free_boundary_enabled=True,
    )

    assert projection["enabled"] is True
    assert projection["info"]["control_count"] == 1

    rcos = zeros.copy()
    zsin = zeros.copy()
    rcos[-1, 0] = 3.2
    rcos[-1, 1] = 0.5
    zsin[-1, 1] = 0.25
    trial = VMECState(
        layout=layout,
        Rcos=rcos,
        Rsin=zeros.copy(),
        Zcos=zeros.copy(),
        Zsin=zsin,
        Lcos=zeros.copy(),
        Lsin=zeros.copy(),
    )

    projected = _project_freeb_edge_control_state(trial, projection, host_update=True)
    raw_metrics = _freeb_edge_control_state_residual_metrics(trial, projection)
    projected_metrics = _freeb_edge_control_state_residual_metrics(projected, projection)
    direction_rcos = zeros.copy()
    direction_zsin = zeros.copy()
    direction_rcos[-1, 0] = 0.2
    direction_rcos[-1, 1] = 0.5
    direction_zsin[-1, 1] = 0.25
    direction_metrics = _freeb_edge_control_delta_tuple_projection_metrics(
        (direction_rcos, zeros.copy(), zeros.copy(), direction_zsin, zeros.copy(), zeros.copy()),
        projection,
    )
    direction_lcos = zeros.copy()
    direction_lcos[-1, 0] = 2.0
    projected_direction = _project_freeb_edge_control_delta_tuple(
        (
            direction_rcos,
            zeros.copy(),
            zeros.copy(),
            direction_zsin,
            direction_lcos,
            zeros.copy(),
        ),
        projection,
        host_update=True,
    )
    projected_direction_metrics = _freeb_edge_control_delta_tuple_projection_metrics(
        projected_direction,
        projection,
    )

    assert np.asarray(projected.Rcos)[-1, 0] == pytest.approx(3.2)
    assert np.asarray(projected.Rcos)[-1, 1] == pytest.approx(0.0)
    assert np.asarray(projected.Zsin)[-1, 1] == pytest.approx(0.0)
    assert raw_metrics["status"] == "measured"
    assert raw_metrics["residual_linf"] > 0.1
    assert raw_metrics["control_delta_by_label"]["R00"] == pytest.approx(0.2)
    assert projected_metrics["residual_linf"] == pytest.approx(0.0, abs=1.0e-12)
    assert projected_metrics["residual_rms"] == pytest.approx(0.0, abs=1.0e-12)
    assert direction_metrics["status"] == "measured"
    assert direction_metrics["residual_linf"] > 0.1
    assert direction_metrics["control_delta_by_label"]["R00"] == pytest.approx(0.2)
    assert 0.0 < direction_metrics["captured_fraction"] < 1.0
    assert np.asarray(projected_direction[0])[-1, 0] == pytest.approx(0.2)
    assert np.asarray(projected_direction[0])[-1, 1] == pytest.approx(0.0)
    assert np.asarray(projected_direction[3])[-1, 1] == pytest.approx(0.0)
    assert np.asarray(projected_direction[4])[-1, 0] == pytest.approx(2.0)
    assert projected_direction_metrics["residual_linf"] == pytest.approx(0.0, abs=1.0e-12)
    assert projected_direction_metrics["captured_fraction"] == pytest.approx(1.0)


def test_square_coil_profile_hot_restart_solver_state_filters_freeb_resume_keys():
    resume_state = {
        "time_step": 0.02,
        "freeb_ivac": 4,
        "freeb_ivacskip": 0,
        "freeb_nvacskip": 7,
        "freeb_nvskip0": 1,
        "freeb_last_model": "vmec2000_like_dense_integral",
        "freeb_nestor_runtime": "runtime",
        "prev_rz_fsq": 1.0e-9,
    }
    run = SimpleNamespace(result=SimpleNamespace(diagnostics={"resume_state": resume_state}))

    assert profile._jax_hot_restart_solver_state(run, policy="state") is None
    freeb = profile._jax_hot_restart_solver_state(run, policy="freeb")
    assert freeb == {
        "freeb_ivac": 4,
        "freeb_ivacskip": 0,
        "freeb_nvacskip": 7,
        "freeb_nvskip0": 1,
        "freeb_model": "vmec2000_like_dense_integral",
        "freeb_nestor_runtime": "runtime",
        "prev_rz_fsq": pytest.approx(1.0e-9),
    }
    assert "time_step" not in freeb
    assert profile._jax_hot_restart_solver_state(run, policy="full") == resume_state


def test_square_coil_profile_json_ready_summarizes_nestor_runtime():
    runtime = NestorRuntimeState(
        operator_cache=SimpleNamespace(tag="cache"),
        phi=np.zeros((2, 3)),
        bsqvac=np.zeros((4, 5)),
        mode="dense",
        update_count=6,
        reuse_count=7,
        source_cache_iter=8,
    )

    payload = profile._json_ready({"resume_state": {"freeb_nestor_runtime": runtime}})

    summary = payload["resume_state"]["freeb_nestor_runtime"]
    assert summary == {
        "type": "NestorRuntimeState",
        "mode": "dense",
        "update_count": 6,
        "reuse_count": 7,
        "source_cache_iter": 8,
        "phi_shape": [2, 3],
        "bsqvac_shape": [4, 5],
        "operator_cache_type": "SimpleNamespace",
    }


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
    assert payload["strict_convergence"]["status"] == "non_strict_ftol"
    assert payload["strict_convergence"]["component_max"] == pytest.approx(4.0e-6)
    assert payload["strict_convergence"]["component_max_over_strict_target"] == pytest.approx(4.0e6)
    assert payload["strict_convergence"]["fresh_residual_required"] is False
    assert "requested_ftol_above_1e-12" in payload["strict_convergence"]["blockers"]
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
    assert data["strict_convergence"]["status"] == "near_strict_not_met"
    assert data["strict_convergence"]["component_max_over_strict_target"] == pytest.approx(40.0)
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


def test_square_coil_profile_vmec_scale_payload_compares_phiedge_to_external_r_bphi(monkeypatch):
    monkeypatch.setattr(profile, "build_coil_field_geometry", lambda _params: "geometry")

    captured = {}

    def fake_sample(provider_kind, provider_static, provider_params, R, Z, phi):
        captured["provider_kind"] = provider_kind
        captured["provider_static"] = provider_static
        captured["shape"] = np.asarray(R).shape
        return np.zeros_like(R), 2.0 * np.ones_like(R), np.zeros_like(R)

    monkeypatch.setattr(profile, "sample_external_field_cylindrical", fake_sample)
    config = profile.ExampleConfig(
        mpol=3,
        ntor=4,
        ns=5,
        ns_array=(5,),
        niter_array=(2,),
        ftol_array=(1.0e-8,),
        max_iter=2,
        ntheta=32,
        nzeta=32,
        max_boundary_projection_error=None,
        coil_segments=8,
    )
    indata = profile.make_free_boundary_indata(config, beta_percent=0.0)
    payload = profile._vmec_free_boundary_scale_payload(
        indata=indata,
        coil_params=SimpleNamespace(regularization_epsilon=1.0e-6, chunk_size=None),
        config=config,
    )

    assert captured["provider_kind"] == "direct_coils"
    assert captured["provider_static"]["coil_geometry"] == "geometry"
    assert captured["shape"] == (32, 32)
    assert payload["status"] in {"scale_mismatch", "severe_scale_mismatch", "scale_reasonable"}
    assert payload["external_r_bphi_rms"] > 0.0
    assert payload["phiedge_over_r1_z1_abs"] > 0.0
    assert payload["suggested_phiedge_for_external_r_bphi_rms"] < 0.0


def test_square_coil_profile_scale_diagnostics_only_writes_scale_payload(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(profile, "build_coil_field_geometry", lambda _params: "geometry")

    def fail_mgrid(*_args, **_kwargs):
        raise AssertionError("scale diagnostics should not write an mgrid")

    def fake_sample(_provider_kind, _provider_static, _provider_params, R, Z, _phi):
        return np.zeros_like(R), np.ones_like(R), np.zeros_like(Z)

    monkeypatch.setattr(profile, "write_mgrid_from_coils", fail_mgrid)
    monkeypatch.setattr(profile, "sample_external_field_cylindrical", fake_sample)

    outdir = tmp_path / "profile_scale_only"
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
            "--ntheta",
            "32",
            "--nzeta",
            "16",
            "--max-iter",
            "2",
            "--coil-segments",
            "8",
            "--max-boundary-projection-error",
            "none",
            "--scale-diagnostics-only",
        ]
    )

    data = json.loads((outdir / "square_coil_free_boundary_backend_profile.json").read_text())
    assert data["configuration"]["scale_diagnostics_only"] is True
    assert data["configuration"]["resolution_diagnostics_only"] is False
    assert data["mgrid"]["created"] is False
    assert data["provider_parity"] is None
    assert data["backends"] == {}
    assert data["vmec_free_boundary_scale"]["external_r_bphi_rms"] > 0.0
    assert data["vmec_free_boundary_scale"]["status"] in {
        "scale_reasonable",
        "scale_mismatch",
        "severe_scale_mismatch",
    }


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
    assert data["configuration"]["ntheta"] == profile.recommended_square_axis_ntheta(3)
    assert data["configuration"]["recommended_ntheta"] == profile.recommended_square_axis_ntheta(3)
    assert data["configuration"]["ntheta_auto"] is True
    assert data["configuration"]["ntheta_underrecommended"] is False
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
    assert spline_bridge["optional_solver_edge_control_projection"] is True
    assert spline_bridge["requires_fourier_projection"] is True
    assert spline_bridge["reduced_square_control_count"] == 2
    assert spline_bridge["can_reduce_input_shape_dofs"] is True
    assert spline_bridge["can_project_free_boundary_edge_updates"] is True
    assert spline_bridge["can_reduce_free_boundary_edge_dofs"] is True
    assert spline_bridge["can_reduce_nonlinear_solver_dofs"] is False
    assert spline_bridge["requires_native_spline_state_for_reduced_nonlinear_dofs"] is True
    assessment = data["strict_convergence_assessment"]
    assert assessment["schema"] == "square_axis_strict_convergence_assessment.v1"
    assert assessment["full_fourier_strict_profile_status"] == "ready_to_attempt"
    assert assessment["reduced_control_profile_status"] == "not_enabled"
    assert assessment["solver_native_spline_status"] == "not_implemented"
    assert assessment["vmec2000_expected_to_fix_fourier_bottleneck"] is False
    assert "spline_control_updates_not_enabled" in assessment["blockers"]
    deck = data["resolution_deck"]
    assert deck["status"] == "production_ready"
    assert deck["ntheta"] == profile.recommended_square_axis_ntheta(3)
    assert deck["ntheta_underrecommended"] is False
    assert deck["projection_meets_gate"] is True
    assert deck["mgrid_nphi_multiple_of_nzeta"] is True
    assert data["vmec_free_boundary_scale"]["status"] == "skipped_all_backends_disabled"


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


def test_square_coil_profile_production_gate_auto_bumps_underrecommended_nzeta(
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

    outdir = tmp_path / "profile_auto_bumped_nzeta"
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
                "--ntheta",
                "16",
                "--nzeta",
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
        == 0
    )

    data = json.loads((outdir / "square_coil_free_boundary_backend_profile.json").read_text())
    assert data["configuration"]["ntheta"] == profile.recommended_square_axis_ntheta(5)
    assert data["configuration"]["ntheta_auto"] is False
    assert data["configuration"]["ntheta_auto_bumped_to_recommended"] is True
    assert data["configuration"]["ntheta_underrecommended"] is False
    assert data["configuration"]["nzeta"] == profile.recommended_square_axis_nzeta(28)
    assert data["configuration"]["nzeta_auto"] is False
    assert data["configuration"]["nzeta_auto_bumped_to_recommended"] is True
    assert data["configuration"]["nzeta_underrecommended"] is False
    assert data["resolution_deck"]["status"] == "production_ready"


def test_square_coil_profile_production_gate_rejects_underrecommended_nzeta_without_auto_bump(
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
                "--ntheta",
                "16",
                "--nzeta",
                "32",
                "--mgrid-nphi",
                "32",
                "--no-auto-bump-nzeta-to-recommended",
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
                "--ntheta",
                "16",
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
    assert "ntheta_below_square_axis_recommendation" in deck["reasons"]
    assert "nzeta_below_square_axis_recommendation" in deck["reasons"]
    assert data["configuration"]["ntheta"] == 16
    assert data["configuration"]["ntheta_underrecommended"] is True
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
    assert data["configuration"]["ntheta"] == profile.recommended_square_axis_ntheta(5)
    assert data["configuration"]["ntheta_auto"] is True
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
            "--virtual-casing-pythonpath",
            str(tmp_path),
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
    assert data["configuration"]["virtual_casing_pythonpath"] == str(tmp_path.resolve())
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


def test_square_coil_profile_run_jax_backend_passes_edge_control_projection(
    monkeypatch, tmp_path: Path
):
    captured = {}
    projection_payload = {
        "enabled": True,
        "basis_symmetry": "square",
        "labels": ["side", "corner"],
        "control_jacobian": np.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]),
        "rcond": 1.0e-10,
    }

    monkeypatch.setattr(
        profile,
        "_freeb_edge_control_projection_solver_payload",
        lambda config, *, symmetry, rcond: projection_payload,
    )
    monkeypatch.setattr(profile, "write_wout_from_fixed_boundary_run", lambda *args, **kwargs: None)

    def fake_run_free_boundary(*args, **kwargs):
        captured.update(kwargs)
        result = SimpleNamespace(
            n_iter=1,
            diagnostics={
                "requested_ftol": 1.0e-12,
                "final_fsqr": 1.0e-6,
                "final_fsqz": 2.0e-6,
                "final_fsql": 3.0e-6,
                "free_boundary": {
                    "nestor_model": "vmec2000_like_dense_integral",
                    "edge_control_projection": {
                        "enabled": True,
                        "basis_symmetry": "square",
                        "control_count": 2,
                        "apply_count": 3,
                        "delta_projection_count": 5,
                        "zero_velocity_count": 4,
                    },
                },
            },
            w_history=[],
            fsqr2_history=[],
            fsqz2_history=[],
            fsql2_history=[],
        )
        return SimpleNamespace(result=result, state=SimpleNamespace(name="state"))

    monkeypatch.setattr(profile, "run_free_boundary", fake_run_free_boundary)
    config = SimpleNamespace(
        use_multigrid_schedule=False,
        max_iter=1,
        jit_forces=True,
        free_boundary_activate_fsq=1.0e-3,
    )

    out = profile._run_jax_backend(
        input_path=tmp_path / "input.case",
        wout_path=tmp_path / "wout_case.nc",
        config=config,
        direct_params=None,
        solver_mode="parity",
        return_best_scored_state=False,
        freeb_edge_control_projection="square",
        freeb_edge_control_rcond=1.0e-10,
    )

    assert captured["free_boundary_edge_control_projection"] is projection_payload
    summary = out["free_boundary_solver_overrides"]["freeb_edge_control_projection"]
    assert summary["status"] == "enabled"
    assert summary["basis_symmetry"] == "square"
    assert summary["control_count"] == 2
    assert out["free_boundary_edge_control_projection"]["apply_count"] == 3
    assert out["free_boundary_edge_control_projection"]["delta_projection_count"] == 5
    assert out["free_boundary_edge_control_projection"]["zero_velocity_count"] == 4


def test_square_coil_profile_run_jax_backend_hot_restarts_from_freeb_state(
    monkeypatch, tmp_path: Path
):
    calls = []
    written = {}
    states = [SimpleNamespace(name="initial"), SimpleNamespace(name="restart")]

    monkeypatch.setattr(
        profile,
        "write_wout_from_fixed_boundary_run",
        lambda path, run, **kwargs: written.update({"path": path, "state": run.state, "kwargs": kwargs}),
    )

    def fake_result(*, fsqr: float, fsqz: float, fsql: float, n_iter: int):
        return SimpleNamespace(
            n_iter=n_iter,
            diagnostics={
                "requested_ftol": 1.0e-12,
                "final_fsqr": fsqr,
                "final_fsqz": fsqz,
                "final_fsql": fsql,
                "final_residual_recomputed_on_accepted_state": True,
                "resume_state": {
                    "time_step": 0.02,
                    "freeb_ivac": 4,
                    "freeb_ivacskip": 0,
                    "freeb_nvacskip": 5,
                    "freeb_nvskip0": 1,
                    "freeb_last_model": "vmec2000_like_dense_integral",
                    "prev_rz_fsq": fsqr + fsqz,
                },
                "free_boundary": {
                    "nestor_model": "vmec2000_like_dense_integral",
                    "ivac": 4,
                    "ivacskip": 0,
                    "nvacskip": 5,
                },
            },
            w_history=[],
            fsqr2_history=[],
            fsqz2_history=[],
            fsql2_history=[],
        )

    def fake_run_free_boundary(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        if len(calls) == 1:
            return SimpleNamespace(
                result=fake_result(fsqr=2.0e-9, fsqz=1.0e-9, fsql=5.0e-10, n_iter=100),
                state=states[0],
            )
        return SimpleNamespace(
            result=fake_result(fsqr=2.0e-13, fsqz=3.0e-13, fsql=4.0e-13, n_iter=5),
            state=states[1],
        )

    monkeypatch.setattr(profile, "run_free_boundary", fake_run_free_boundary)
    config = SimpleNamespace(
        use_multigrid_schedule=True,
        max_iter=8000,
        niter_array=[1000, 2000, 8000],
        jit_forces=True,
        free_boundary_activate_fsq=1.0e-3,
    )

    out = profile._run_jax_backend(
        input_path=tmp_path / "input.case",
        wout_path=tmp_path / "wout_case.nc",
        config=config,
        direct_params=None,
        solver_mode="parity",
        return_best_scored_state=True,
        jax_hot_restart_count=2,
        jax_hot_restart_iters=5,
        jax_hot_restart_policy="freeb",
    )

    assert len(calls) == 2
    assert calls[0]["kwargs"]["max_iter"] == 11000
    assert calls[0]["kwargs"]["multigrid"] is True
    assert "restart_state" not in calls[0]["kwargs"]
    assert calls[1]["kwargs"]["max_iter"] == 5
    assert calls[1]["kwargs"]["multigrid"] is False
    assert calls[1]["kwargs"]["restart_state"] is states[0]
    restart_solver_state = calls[1]["kwargs"]["restart_solver_state"]
    assert restart_solver_state["freeb_ivac"] == 4
    assert restart_solver_state["freeb_nvacskip"] == 5
    assert "time_step" not in restart_solver_state
    assert out["strict_convergence"]["strict_components_met"] is True
    assert out["hot_restart"]["requested_count"] == 2
    assert out["hot_restart"]["executed_count"] == 1
    assert out["hot_restart"]["resume_policy"] == "freeb"
    assert out["hot_restart"]["stopped_after_strict_convergence"] is True
    assert out["hot_restart"]["stages"][0]["strict_status"] == "loose_only_above_strict"
    assert out["hot_restart"]["stages"][1]["strict_status"] == "strict_components_met"
    assert out["free_boundary_solver_overrides"]["jax_hot_restart_count"] == 2
    assert out["free_boundary_solver_overrides"]["jax_hot_restart_iters"] == 5
    assert out["free_boundary_solver_overrides"]["jax_hot_restart_policy"] == "freeb"
    assert written["state"] is states[1]


def test_square_coil_profile_run_jax_backend_can_seed_initial_pass_from_wout(
    monkeypatch, tmp_path: Path
):
    calls = []
    seed = tmp_path / "seed_wout.nc"
    monkeypatch.setattr(profile, "write_wout_from_fixed_boundary_run", lambda *args, **kwargs: None)

    def fake_run_free_boundary(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        result = SimpleNamespace(
            n_iter=3,
            diagnostics={
                "requested_ftol": 1.0e-12,
                "final_fsqr": 2.0e-13,
                "final_fsqz": 3.0e-13,
                "final_fsql": 4.0e-13,
                "final_residual_recomputed_on_accepted_state": True,
                "free_boundary": {"nestor_model": "vmec2000_like_dense_integral"},
            },
            w_history=[],
            fsqr2_history=[],
            fsqz2_history=[],
            fsql2_history=[],
        )
        return SimpleNamespace(result=result, state=SimpleNamespace(name="seeded"))

    monkeypatch.setattr(profile, "run_free_boundary", fake_run_free_boundary)
    config = SimpleNamespace(
        use_multigrid_schedule=True,
        max_iter=8000,
        niter_array=[1000, 2000, 8000],
        jit_forces=True,
        free_boundary_activate_fsq=1.0e-3,
    )

    out = profile._run_jax_backend(
        input_path=tmp_path / "input.case",
        wout_path=tmp_path / "wout_case.nc",
        config=config,
        direct_params=None,
        solver_mode="parity",
        return_best_scored_state=True,
        jax_hot_restart_count=0,
        jax_hot_restart_iters=5000,
        jax_initial_restart_wout=seed,
    )

    assert len(calls) == 1
    assert calls[0]["kwargs"]["max_iter"] == 5000
    assert calls[0]["kwargs"]["multigrid"] is False
    assert calls[0]["kwargs"]["restart_wout_path"] == seed
    assert out["hot_restart"]["stages"][0]["kind"] == "initial_restart_wout"
    assert out["hot_restart"]["stages"][0]["restart_policy"] == "wout"
    assert out["hot_restart"]["initial_restart_wout"] == str(seed)
    assert out["free_boundary_solver_overrides"]["jax_initial_restart_wout"] == str(seed)
