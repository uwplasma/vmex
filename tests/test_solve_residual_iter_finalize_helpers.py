from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax.solvers.fixed_boundary.residual.finalize import (
    _edge_control_native_control_state_payload,
    _edge_control_native_spline_state_payload,
    _namespace_with_best_scored_state,
    attach_residual_iter_timing_diagnostics,
    build_residual_iter_resume_state_from_namespace,
    build_residual_iter_resume_state_payload,
    final_free_boundary_residual_reports_from_namespace,
    finalize_residual_iter_from_namespace,
    finalize_residual_iter_result,
    vmec2000_state_only_scan_result,
    vmec2000_traced_scan_result,
)
from vmec_jax.solvers.free_boundary import FreeBoundaryNativeSplineState
from vmec_jax.solvers.fixed_boundary.results import SolveVmecResidualResult
from vmec_jax.state import StateLayout, VMECState


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


def test_edge_control_native_state_payload_uses_reduced_edge_state_schema() -> None:
    projection = {
        "enabled": True,
        "mode_count": 1,
        "jacobian_np": np.asarray([[1.0], [0.0], [0.0], [0.0]]),
        "initial_np": {
            "R_cos": np.asarray([3.0]),
            "R_sin": np.asarray([0.0]),
            "Z_cos": np.asarray([0.0]),
            "Z_sin": np.asarray([0.0]),
        },
        "info": {
            "enabled": True,
            "labels": ["R00"],
            "rcond": 1.0e-12,
        },
    }
    ns = {
        "freeb_edge_control_projection_enabled": True,
        "freeb_edge_control_projection": projection,
        "freeb_edge_control_projection_native_control_coordinates": np.asarray([0.25]),
    }

    payload = _edge_control_native_control_state_payload(ns)

    assert payload["enabled"] is True
    assert payload["status"] == "tracked"
    assert payload["mode"] == "native_reduced_lcfs_edge_state"
    assert payload["native_state_schema"] == "FreeBoundaryReducedEdgeState.v1"
    assert payload["full_edge_size"] == 4
    assert payload["reduced_unknown_size"] == 1
    assert payload["reduction_fraction"] == pytest.approx(0.25)
    assert payload["unknown_by_label"] == {"R00": pytest.approx(0.25)}
    assert payload["decoded_edge_linf"] == pytest.approx(3.25)


def test_edge_control_native_spline_state_payload_reports_adapter_schema() -> None:
    projection = {
        "enabled": True,
        "mode_count": 1,
        "mode_scale_np": np.asarray([1.0]),
        "jacobian_np": np.asarray([[1.0], [0.0], [0.0], [0.0]]),
        "initial_np": {
            "R_cos": np.asarray([3.0]),
            "R_sin": np.asarray([0.0]),
            "Z_cos": np.asarray([0.0]),
            "Z_sin": np.asarray([0.0]),
        },
        "info": {
            "enabled": True,
            "labels": ["R00"],
            "rcond": 1.0e-12,
        },
    }
    layout = StateLayout(ns=2, K=1, lasym=False)
    zeros = np.zeros((2, 1))
    state = VMECState(
        layout=layout,
        Rcos=np.asarray([[3.0], [3.25]]),
        Rsin=zeros.copy(),
        Zcos=zeros.copy(),
        Zsin=zeros.copy(),
        Lcos=zeros.copy(),
        Lsin=zeros.copy(),
    )
    native_state = FreeBoundaryNativeSplineState.from_vmec_state(state, projection)
    ns = {
        "freeb_edge_control_projection_enabled": True,
        "freeb_edge_control_projection": projection,
        "freeb_edge_control_projection_native_spline_state": native_state,
    }

    payload = _edge_control_native_spline_state_payload(ns)

    assert payload["enabled"] is True
    assert payload["status"] == "tracked"
    assert payload["native_state_schema"] == "FreeBoundaryNativeSplineState.v1"
    assert payload["mode"] == "free_boundary_native_spline_state"
    assert payload["nonlinear_boundary_basis"] == "reduced_lcfs_spline_controls"
    assert payload["force_evaluation_boundary_basis"] == "vmec_fourier_coefficients"
    assert payload["full_edge_size"] == 4
    assert payload["reduced_unknown_size"] == 1
    assert payload["unknown_linf"] == pytest.approx(0.25)


def test_best_scored_namespace_restores_matching_free_boundary_bundle() -> None:
    runtime = object()
    bsqvac = np.asarray([[1.0, 2.0]])
    namespace = {
        "return_best_scored_state": True,
        "state": "last-state",
        "fsqr_f": 1.0,
        "fsqz_f": 2.0,
        "fsql_f": 3.0,
        "prev_rz_fsq": 3.0,
        "freeb_bsqvac_half_current": "last-bsqvac",
        "freeb_nestor_runtime": "last-runtime",
        "freeb_last_model": "last-model",
        "freeb_last_diagnostics": {"last": True},
        "freeb_ivac": 4,
        "freeb_ivacskip": 5,
        "freeb_nvacskip": 6,
        "freeb_nvskip0": 7,
        "freeb_plascur": 8.0,
        "best_scored": {
            "state": "best-state",
            "fsqr": 1.0e-12,
            "fsqz": 2.0e-12,
            "fsql": 3.0e-12,
            "freeb_bsqvac_half_current": bsqvac,
            "freeb_nestor_runtime": runtime,
            "freeb_last_model": "jax_nestor",
            "freeb_last_diagnostics": {"bnormal_rms": 4.0e-6},
            "freeb_ivac": 2,
            "freeb_ivacskip": 0,
            "freeb_nvacskip": 3,
            "freeb_nvskip0": 3,
            "freeb_plascur": 0.25,
        },
    }

    restored, returned = _namespace_with_best_scored_state(namespace)

    assert returned is True
    assert restored["state"] == "best-state"
    assert restored["fsqr_f"] == pytest.approx(1.0e-12)
    assert restored["prev_rz_fsq"] == pytest.approx(3.0e-12)
    assert restored["freeb_bsqvac_half_current"] is bsqvac
    assert restored["freeb_nestor_runtime"] is runtime
    assert restored["freeb_last_model"] == "jax_nestor"
    assert restored["freeb_last_diagnostics"] == {"bnormal_rms": 4.0e-6}
    assert restored["freeb_ivac"] == 2
    assert restored["freeb_ivacskip"] == 0
    assert restored["freeb_nvacskip"] == 3
    assert restored["freeb_nvskip0"] == 3
    assert restored["freeb_plascur"] == pytest.approx(0.25)


def test_final_free_boundary_report_compares_cached_and_fresh_vacuum_residuals() -> None:
    calls: list[float] = []

    def compute_forces(_state, **kwargs):
        marker = float(np.asarray(kwargs["freeb_bsqvac_half"]).sum())
        calls.append(marker)
        return None, None, marker, 2.0 * marker, 3.0 * marker, None, None, None

    def residuals(_norms, *, gcr2, gcz2, gcl2):
        return gcr2, gcz2, gcl2

    class _FreshNestor:
        model = "fresh-direct"
        sample_time_s = 0.25
        solve_time_s = 0.5
        diagnostics = {"bnormal_rms": 4.0e-6}
        vac_total = SimpleNamespace(bsqvac=np.asarray([[5.0, 6.0]]))

    namespace = {
        "timing_stats": _timing_stats(),
        "timing_enabled": False,
        "freeb_bsqvac_half_current": np.asarray([[1.0, 2.0]]),
        "fsqr_f": 1.0e-9,
        "fsqz_f": 2.0e-9,
        "fsql_f": 3.0e-9,
        "freeb_last_model": "cached-direct",
        "freeb_last_diagnostics": {"bnormal_rms": 1.0e-6},
        "free_boundary_enabled": True,
        "freeb_couple_edge": True,
        "state": "state",
        "static": SimpleNamespace(cfg=SimpleNamespace(nzeta=2), free_boundary_extcur=()),
        "freeb_nestor_runtime": "runtime",
        "freeb_plascur": 0.25,
        "external_field_provider_kind": "direct",
        "external_field_provider_static": {},
        "external_field_provider_params": {},
        "include_edge": False,
        "zero_m1": 1.0,
        "constraint_precond_diag": None,
        "constraint_tcon_override": None,
        "constraint_precond_active": False,
        "constraint_tcon_active": False,
        "last_iter2": 9,
        "return_best_scored_state": True,
        "_compute_forces_iter": compute_forces,
    }

    report = final_free_boundary_residual_reports_from_namespace(
        namespace,
        nestor_external_only_step_func=lambda **_: (_FreshNestor(), "runtime"),
        residual_fsq_from_norms_func=residuals,
        device_get_floats_func=lambda *values: tuple(float(value) for value in values),
    )

    assert calls == [3.0, 11.0]
    assert report["final_cached_vacuum_residual_recomputed"] is True
    assert report["final_cached_vacuum_fsqr_report"] == pytest.approx(3.0)
    assert report["final_fsqr_report"] == pytest.approx(11.0)
    assert report["final_fresh_minus_cached_vacuum_fsqr"] == pytest.approx(8.0)
    assert report["final_nestor_model"] == "fresh-direct"
    assert report["final_nestor_diagnostics"] == {"bnormal_rms": 4.0e-6}


def test_finalize_residual_iter_from_namespace_builds_diagnostics_and_result() -> None:
    def attach(result):
        result.diagnostics["attached"] = True
        return result

    def convergence_flags(*, fsqr, fsqz, fsql, ftol, fsq_total_target):
        assert fsq_total_target is None
        return fsqr <= ftol and fsqz <= ftol and fsql <= ftol, False, fsqr + fsqz + fsql

    payload = object()
    namespace = {
        "timing_enabled": False,
        "timing_stats": _timing_stats(),
        "timing_detail_enabled": False,
        "t_iteration_loop_start": None,
        "_solve_wall_start": 0.0,
        "freeb_bsqvac_half_current": None,
        "fsqr_f": 1.0e-8,
        "fsqz_f": 2.0e-8,
        "fsql_f": 3.0e-8,
        "freeb_last_model": "none",
        "freeb_last_diagnostics": {},
        "free_boundary_enabled": False,
        "freeb_couple_edge": False,
        "ftol": 1.0e-7,
        "gamma": 1.0,
        "step_size": 0.9,
        "precond_radial_alpha": 0.5,
        "precond_lambda_alpha": 0.25,
        "strict_update": True,
        "reference_mode": False,
        "use_restart_triggers": True,
        "use_direct_fallback": False,
        "max_update_rms": 5.0e-3,
        "converged": True,
        "badjac_use_state": False,
        "badjac_mode": "ptau",
        "badjac_state_probe": False,
        "badjac_initial_state_probe_iters": 0,
        "light_history": False,
        "resume_state_mode": "none",
        "fsq_total_target": None,
        "ijacob": 0,
        "bad_resets": 0,
        "setup_axis_reset_applied": True,
        "axis_reset_done": True,
        "setup_axis_force_probe": object(),
        "setup_axis_force_probe_reused": True,
        "iter1": 1,
        "res0": 0.0,
        "freeb_nvacskip": 1,
        "freeb_nvskip0": 1,
        "freeb_ivac": -1,
        "freeb_ivacskip": 0,
        "free_boundary_activate_fsq": None,
        "freeb_plascur": 0.0,
        "state": "state",
        "return_best_scored_state": True,
        "best_scored": {
            "state": "best-state",
            "iter": 6,
            "fsq": 6.0e-9,
            "fsqr": 1.0e-9,
            "fsqz": 2.0e-9,
            "fsql": 3.0e-9,
            "component_max": 3.0e-9,
            "full_boundary_count": 4,
            "fresh_boundary_count": 3,
        },
        "w_history": [0.0, 1.0],
        "fsqr2_history": [1.0e-8],
        "fsqz2_history": [2.0e-8],
        "fsql2_history": [3.0e-8],
        "grad_rms_history": [4.0e-8],
        "step_history": [0.1],
        "k": payload,
    }

    result = finalize_residual_iter_from_namespace(
        namespace,
        result_type=SolveVmecResidualResult,
        nestor_external_only_step_func=lambda **_: (_ for _ in ()).throw(AssertionError("unused")),
        residual_fsq_from_norms_func=lambda *_, **__: (_ for _ in ()).throw(AssertionError("unused")),
        device_get_floats_func=lambda *values: tuple(float(v) for v in values),
        residual_convergence_flags_func=convergence_flags,
        residual_iter_history_diagnostics_func=lambda ns: {"history_marker": len(ns["w_history"])},
        attach_free_boundary_diagnostics=attach,
        return_final_force_payload=True,
    )

    assert result.state == "best-state"
    assert result.diagnostics["attached"] is True
    assert result.diagnostics["history_marker"] == 2
    assert result.diagnostics["converged_strict"] is True
    assert result.diagnostics["return_best_scored_state"] is True
    assert result.diagnostics["returned_best_scored_state"] is True
    assert result.diagnostics["best_scored_iter"] == 6
    assert result.diagnostics["best_scored_component_max"] == pytest.approx(3.0e-9)
    assert result.diagnostics["best_scored_full_boundary_count"] == 4
    assert result.diagnostics["best_scored_fresh_boundary_count"] == 3
    assert result.diagnostics["final_fsqr"] == pytest.approx(1.0e-9)
    assert result.diagnostics["setup_axis_reset_applied"] is True
    assert result.diagnostics["setup_axis_reset_done"] is True
    assert result.diagnostics["setup_axis_force_probe_available"] is True
    assert result.diagnostics["setup_axis_force_probe_reused"] is True
    assert result.diagnostics["free_boundary"]["enabled"] is False
    assert result.diagnostics["resume_state"] is None
    assert getattr(result, "_final_force_payload") is payload


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
