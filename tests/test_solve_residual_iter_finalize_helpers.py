from __future__ import annotations

import numpy as np
import pytest

from vmec_jax.solvers.fixed_boundary.residual.finalize import (
    attach_residual_iter_timing_diagnostics,
    build_residual_iter_resume_state_from_namespace,
    build_residual_iter_resume_state_payload,
    finalize_residual_iter_result,
    vmec2000_state_only_scan_result,
    vmec2000_traced_scan_result,
)
from vmec_jax.solvers.fixed_boundary.results import SolveVmecResidualResult


def _timing_stats() -> dict[str, float]:
    return {
        "setup_total": 0.25,
        "setup_static_grid_rebuild": 0.01,
        "setup_freeb_policy": 0.02,
        "setup_boundary_profiles": 0.03,
        "setup_cache_key_hash": 0.04,
        "setup_ptau_constants": 0.005,
        "setup_index_constants": 0.006,
        "setup_update_constants": 0.009,
        "setup_axis_reset": 0.05,
        "setup_axis_reset_compute_forces": 0.02,
        "iteration_loop": 1.0,
        "iteration_prepare": 0.1,
        "iteration_residual_metrics": 0.2,
        "iteration_post_update": 0.05,
        "finalize": 0.03,
        "finalize_nestor_recompute": 0.006,
        "finalize_residual_recompute": 0.010,
        "finalize_residual_device_get": 0.004,
        "finalize_diag_build": 0.008,
        "compute_forces": 0.4,
        "compute_forces_first": 0.15,
        "compute_forces_rest": 0.25,
        "compute_forces_calls": 3,
        "compute_forces_main": 0.4,
        "compute_forces_main_calls": 3,
        "compute_forces_auto_flip": 0.03,
        "compute_forces_auto_flip_calls": 2,
        "compute_forces_trial": 0.05,
        "compute_forces_trial_calls": 1,
        "compute_forces_backtracking": 0.07,
        "compute_forces_backtracking_calls": 1,
        "iteration_control": 0.25,
        "iteration_control_fsq1": 0.11,
        "iteration_control_fsq1_precond_norm": 0.02,
        "iteration_control_fsq1_scalar_build": 0.01,
        "iteration_control_fsq1_payload_get": 0.04,
        "iteration_control_fsq1_direct_get": 0.02,
        "iteration_control_badjac": 0.08,
        "iteration_control_badjac_ptau_get": 0.03,
        "iteration_control_badjac_state_jacobian": 0.02,
        "iteration_control_vmec_time": 0.01,
        "iteration_control_restart": 0.02,
        "iteration_control_evolve": 0.01,
        "preconditioner": 0.12,
        "precond_apply": 0.08,
        "precond_mode_scale": 0.01,
        "precond_refresh_seed": 0.015,
        "precond_refresh_calls": 3,
        "precond_reassemble_calls": 1,
        "precond_cache_hit_count": 5,
        "precond_refresh_seed_reuse_count": 2,
        "update": 0.09,
        "update_state": 0.07,
        "update_state_ready": 0.025,
        "update_trace_build": 0.0,
        "update_trace_finalize": 0.0,
        "precond_refresh": 0.04,
        "iterations": 2,
    }


def _resume_base_kwargs() -> dict[str, object]:
    return {
        "time_step": 0.9,
        "inv_tau": [1.0, 2.0],
        "fsq_prev": 3.0,
        "fsq0_prev": 4.0,
        "flip_sign": -1.0,
        "iter1": 7,
        "last_iter2": 8,
        "ijacob": 2,
        "bad_resets": 1,
        "res0": 0.5,
        "res1": 0.25,
        "prev_rz_fsq": 0.125,
        "bad_growth_streak": 3,
        "huge_force_restart_count": 4,
        "vmec2000_cache_valid": True,
        "freeb_ivac": 2,
        "freeb_ivacskip": 1,
        "freeb_nvacskip": 5,
        "freeb_nvskip0": 5,
        "freeb_last_model": "jax_nestor",
        "freeb_nestor_runtime": None,
    }


def test_attach_residual_iter_timing_diagnostics_updates_report_without_printing() -> None:
    diagnostics: dict[str, object] = {}
    stats = _timing_stats()
    got = attach_residual_iter_timing_diagnostics(
        diagnostics,
        stats,
        timing_enabled=True,
        timing_detail_enabled=True,
        finalize_diag_build_start=None,
        iteration_loop_start=10.0,
        finalize_start=11.5,
        solve_wall_start=9.0,
        print_timing=False,
    )

    assert got is diagnostics
    assert "timing" in diagnostics
    assert diagnostics["timing"]["force_eval_s"] == pytest.approx(0.4)
    assert stats["iteration_loop"] == pytest.approx(1.5)
    assert stats["iteration_loop_unattributed"] == pytest.approx(
        diagnostics["timing"]["iteration_loop_unattributed_s"]
    )


def test_build_residual_iter_resume_state_payload_modes() -> None:
    assert build_residual_iter_resume_state_payload(
        resume_state_mode="none",
        base_kwargs=_resume_base_kwargs(),
    ) is None

    light = build_residual_iter_resume_state_payload(
        resume_state_mode="light",
        base_kwargs=_resume_base_kwargs(),
    )
    assert light["iter_offset"] == 8
    assert light["freeb_model"] == "jax_nestor"
    assert "cache_norms" not in light

    full = build_residual_iter_resume_state_payload(
        resume_state_mode="full",
        base_kwargs=_resume_base_kwargs(),
        heavy_payload={"cache_norms": np.asarray([1.0, 2.0])},
    )
    assert full["freeb_nvacskip"] == 5
    np.testing.assert_allclose(full["cache_norms"], [1.0, 2.0])


def test_build_residual_iter_resume_state_from_namespace_selects_payload() -> None:
    namespace = _resume_base_kwargs()
    namespace.update(
        {key: [1.0, 2.0] for key in "vRcc vRss vZsc vZcs vLsc vLcs vRsc vRcs vZcc vZss vLcc vLss".split()}
    )
    namespace.update(
        {
            "state_checkpoint": "checkpoint",
            "cache_precond_diag": None,
            "cache_tcon": None,
            "cache_norms": np.asarray([3.0, 4.0]),
            "cache_rz_scale": None,
            "cache_l_scale": None,
            "cache_rz_norm": None,
            "cache_f_norm1": None,
            "cache_prec_rz_mats": None,
            "cache_prec_rz_jmax": None,
            "cache_prec_lam_prec": None,
            "cache_prec_faclam": None,
            "cache_prec_lam_debug": None,
            "cache_constraint_rcon0": None,
            "cache_constraint_zcon0": None,
            "unrelated_large_local": object(),
        }
    )

    assert build_residual_iter_resume_state_from_namespace(namespace, resume_state_mode="none") is None
    light = build_residual_iter_resume_state_from_namespace(namespace, resume_state_mode="light")
    assert light["iter_offset"] == 8
    assert "vRcc" not in light

    full = build_residual_iter_resume_state_from_namespace(namespace, resume_state_mode="full")
    np.testing.assert_allclose(full["vRcc"], [1.0, 2.0])
    np.testing.assert_allclose(full["cache_norms"], [3.0, 4.0])
    assert full["state_checkpoint"] == "checkpoint"
    assert "unrelated_large_local" not in full


def test_finalize_residual_iter_result_attaches_free_boundary_and_force_payload() -> None:
    def attach(result):
        result.diagnostics["attached"] = True
        return result

    payload = object()
    result = finalize_residual_iter_result(
        result_type=SolveVmecResidualResult,
        state="state",
        w_history=[0.0, 1.0, 2.0],
        fsqr2_history=[1.0, 0.5],
        fsqz2_history=[2.0, 0.25],
        fsql2_history=[3.0, 0.125],
        grad_rms_history=[4.0, 0.0625],
        step_history=[0.1, 0.2],
        diagnostics={},
        attach_free_boundary_diagnostics=attach,
        return_final_force_payload=True,
        converged=True,
        final_force_payload=payload,
    )

    assert result.n_iter == 2
    assert result.diagnostics["attached"] is True
    assert getattr(result, "_final_force_payload") is payload
    np.testing.assert_allclose(result.w_history, [0.0, 1.0, 2.0])


def test_vmec2000_state_only_scan_result_builds_empty_history_result() -> None:
    empty = np.asarray([], dtype=float)
    carry = type("Carry", (), {"state": "state"})()

    def attach(result):
        result.diagnostics["attached"] = True
        return result

    result = vmec2000_state_only_scan_result(
        result_type=SolveVmecResidualResult,
        carry_final=carry,
        empty_history=empty,
        max_iter=7,
        diagnostics={"state_only": True},
        attach_free_boundary_diagnostics=attach,
    )

    assert result.state == "state"
    assert result.n_iter == 7
    assert result.diagnostics == {"state_only": True, "attached": True}
    assert result.w_history is empty
    assert result.fsqr2_history is empty
    assert result.step_history is empty


def test_vmec2000_traced_scan_result_builds_resume_diagnostics() -> None:
    empty = np.asarray([], dtype=float)
    carry = type("Carry", (), {"state": "traced-state"})()
    resume_state = {"iter_offset": 3}
    attach_calls = []

    def attach(result):
        attach_calls.append(result)
        return result

    def diagnostics_func(*, resume_state, scan_use_precomputed, scan_use_lax_tridi):
        return {
            "resume_state": resume_state,
            "scan_use_precomputed": scan_use_precomputed,
            "scan_use_lax_tridi": scan_use_lax_tridi,
        }

    result = vmec2000_traced_scan_result(
        result_type=SolveVmecResidualResult,
        carry_final=carry,
        empty_history=empty,
        max_iter=5,
        resume_state=resume_state,
        scan_use_precomputed=True,
        scan_use_lax_tridi=False,
        attach_free_boundary_diagnostics=attach,
        traced_diagnostics_func=diagnostics_func,
    )

    assert attach_calls == [result]
    assert result.state == "traced-state"
    assert result.n_iter == 5
    assert result.diagnostics["resume_state"] is resume_state
    assert result.diagnostics["scan_use_precomputed"] is True
    assert result.diagnostics["scan_use_lax_tridi"] is False
    assert result.w_history is empty
