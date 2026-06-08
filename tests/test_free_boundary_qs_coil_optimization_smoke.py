from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "examples" / "optimization" / "free_boundary_QS_coil_optimization.py"


def _load_example_module():
    spec = importlib.util.spec_from_file_location("free_boundary_qs_coil_optimization_example", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _same_branch_replay_gate_stub() -> dict[str, object]:
    fingerprint = {
        "n_steps": 1,
        "n_freeb_steps": 1,
        "freeb_sizes": [2],
    }
    trace_diag = {
        "differentiates_adaptive_controller": False,
        "n_steps": 1,
        "branch_fingerprint": fingerprint,
        "masks": {
            "active": [True],
            "accepted": [True],
            "rejected": [False],
            "done": [True],
            "has_active_freeb_replay": [True],
        },
        "replay_diagnostics": {
            "scalar_controls_stackable": True,
            "array_controls_stackable": True,
            "preconditioner_policy_n_segments": 1,
        },
    }
    return {
        "branch": {
            "same_branch": True,
            "same_accepted_trace_branch": True,
            "same_residual_branch": True,
            "base_fingerprint": fingerprint,
            "plus_fingerprint": fingerprint,
            "minus_fingerprint": fingerprint,
            "base_residual_fingerprint": {"n_iter": 1, "final_fsq_total_bucket": "small"},
            "plus_residual_fingerprint": {"n_iter": 1, "final_fsq_total_bucket": "small"},
            "minus_residual_fingerprint": {"n_iter": 1, "final_fsq_total_bucket": "small"},
        },
        "trace_replay_diagnostics": {
            "base": trace_diag,
            "plus": trace_diag,
            "minus": trace_diag,
        },
    }


def test_objective_terms_report_weighted_proxy_components():
    module = _load_example_module()

    summary = {
        "residual_proxy": 2.0,
        "qs_total": 0.25,
        "qs_helicity_m": 1,
        "qs_helicity_n": 0,
        "qs_surfaces": [0.25, 0.5],
        "aspect": 5.5,
        "target_aspect": 6.0,
        "mean_iota": 0.3,
        "target_iota": 0.4,
    }

    terms = module.objective_terms_from_summary(
        summary,
        residual_weight=3.0,
        qs_weight=4.0,
        aspect_weight=0.5,
        iota_weight=10.0,
    )

    assert terms["residual"]["contribution"] == pytest.approx(6.0)
    assert terms["quasisymmetry"]["contribution"] == pytest.approx(1.0)
    assert terms["quasisymmetry"]["helicity_m"] == 1
    assert terms["quasisymmetry"]["surfaces"] == [0.25, 0.5]
    assert terms["aspect"]["error"] == pytest.approx(-0.5)
    assert terms["aspect"]["contribution"] == pytest.approx(0.125)
    assert terms["mean_iota"]["contribution"] == pytest.approx(0.1)
    assert terms["total"] == pytest.approx(7.225)
    assert module.objective_from_summary(
        summary,
        residual_weight=3.0,
        qs_weight=4.0,
        aspect_weight=0.5,
        iota_weight=10.0,
    ) == pytest.approx(terms["total"])


def test_objective_terms_report_missing_unweighted_proxy_components():
    module = _load_example_module()

    terms = module.objective_terms_from_summary(
        {
            "residual_proxy": 2.0,
            "qs_total": None,
            "aspect": None,
            "target_aspect": 6.0,
            "mean_iota": None,
            "target_iota": 0.4,
        },
        residual_weight=3.0,
        qs_weight=4.0,
        aspect_weight=0.5,
        iota_weight=10.0,
    )

    assert terms["total"] == pytest.approx(6.0)
    assert terms["missing_unweighted_terms"] == ["qs_total", "aspect", "mean_iota"]
    assert terms["quasisymmetry"]["contribution"] == pytest.approx(0.0)
    assert terms["aspect"]["contribution"] == pytest.approx(0.0)
    assert terms["mean_iota"]["contribution"] == pytest.approx(0.0)


def test_circle_variable_manifest_and_apply_are_coil_only():
    module = _load_example_module()
    base_params, _metadata = module.make_circle_provider(current_scale=1.0)
    _x0, variables = module.select_coil_variables(
        base_params,
        max_current_vars=1,
        max_fourier_vars=1,
    )

    manifest = module.variable_records(
        variables,
        base_params,
        current_step=0.1,
        dof_step=0.5,
    )
    perturbed = module.apply_coil_variables(
        base_params,
        np.asarray([1.0, -2.0]),
        variables,
        current_step=0.1,
        dof_step=0.5,
    )

    assert [record["kind"] for record in manifest] == ["current", "fourier_dof"]
    assert all(record["kind"] in {"current", "fourier_dof"} for record in manifest)
    assert manifest[0]["parameterization"] == "multiplicative"
    assert manifest[0]["unit_x_delta"] == pytest.approx(0.2)
    assert manifest[1]["parameterization"] == "additive"
    assert manifest[1]["unit_x_delta"] == pytest.approx(0.5)
    assert float(np.asarray(perturbed.base_currents)[0]) == pytest.approx(2.2)
    assert float(np.asarray(base_params.base_currents)[0]) == pytest.approx(2.0)
    assert float(np.asarray(perturbed.base_curve_dofs)[variables[1][1]]) == pytest.approx(0.4)
    assert float(np.asarray(base_params.base_curve_dofs)[variables[1][1]]) == pytest.approx(1.4)
    assert perturbed.n_segments == base_params.n_segments
    assert perturbed.nfp == base_params.nfp
    assert perturbed.stellsym == base_params.stellsym


def test_circle_provider_honors_chunk_size_and_smoke_default():
    module = _load_example_module()
    params, metadata = module.make_circle_provider(current_scale=1.0, chunk_size=32)

    assert params.chunk_size == 32
    assert metadata["chunk_size"] == 32

    args = module.apply_smoke_defaults(
        module.build_parser().parse_args(["--smoke", "--provider", "circle"])
    )
    assert args.chunk_size is None

    explicit = module.apply_smoke_defaults(
        module.build_parser().parse_args(["--smoke", "--provider", "circle", "--chunk-size", "0"])
    )
    assert explicit.chunk_size is None


def test_same_branch_direction_selects_current_and_fourier_variables():
    module = _load_example_module()
    base_params, _metadata = module.make_circle_provider(current_scale=1.0)

    direction = module.same_branch_direction_from_variables(
        [
            ("current", (0,)),
            ("fourier_dof", (0, 0, 2)),
            ("fourier_dof", (0, 1, 1)),
        ]
    )

    np.testing.assert_array_equal(direction, np.asarray([1.0, 1.0, 0.0]))
    tangent = module.coil_param_direction_from_variables(
        base_params,
        direction,
        [
            ("current", (0,)),
            ("fourier_dof", (0, 0, 2)),
            ("fourier_dof", (0, 1, 1)),
        ],
        current_step=0.02,
        dof_step=1.0e-3,
    )
    np.testing.assert_allclose(np.asarray(tangent.base_currents), np.asarray([0.04]))
    assert float(np.asarray(tangent.base_curve_dofs)[0, 0, 2]) == pytest.approx(1.0e-3)
    assert float(np.asarray(tangent.base_curve_dofs)[0, 1, 1]) == pytest.approx(0.0)


def test_same_branch_vector_key_parser_accepts_bnormal_alias():
    module = _load_example_module()

    keys = module.parse_same_branch_vector_keys("qs_total,bnormal_rms")

    assert keys == ("qs_total", "accepted_bnormal_rms")


def test_same_branch_vector_key_parser_defaults_to_promoted_state_scalars():
    module = _load_example_module()

    keys = module.parse_same_branch_vector_keys(None)

    assert keys == ("aspect", "qs_total", "mean_iota", "lcfs_boundary_moment")


def test_branch_local_scalar_report_adapter_records_gate_evidence():
    pytest.importorskip("jax")
    from vmec_jax._compat import jnp
    from vmec_jax.free_boundary_adjoint import (
        direct_coil_branch_local_scalars_report_from_complete_fd,
        direct_coil_same_branch_physical_scalar_gate_report,
    )

    gate = _same_branch_replay_gate_stub()
    complete_report = {
        "branch_compatibility": gate["branch"],
        "trace_replay_diagnostics": gate["trace_replay_diagnostics"],
        "objective_values": {
            "aspect": {"base": 6.0, "plus": 6.1, "minus": 5.9, "central_fd_directional": 0.2},
            "qs_total": {"base": 0.4, "plus": 0.42, "minus": 0.38, "central_fd_directional": 0.1},
        },
    }
    branch_local = {
        "uses_production_forward": True,
        "differentiates_adaptive_controller": False,
        "differentiates_run_free_boundary": False,
        "differentiates_fixed_accepted_branch": True,
        "derivative_mode": "directional_jvp",
        "replay_ad_mode": "direct",
        "scalar_keys": ("aspect", "qs_total"),
        "values": {"aspect": 6.0, "qs_total": 0.4},
        "replay_value_map": {"aspect": jnp.asarray(6.0), "qs_total": jnp.asarray(0.4)},
        "base_abs_delta": {"aspect": 0.0, "qs_total": 0.0},
        "directional_derivatives": {"aspect": jnp.asarray(0.2), "qs_total": jnp.asarray(0.1)},
        "replay_option_flags": {"use_stacked_step_controls": True, "use_accepted_only_fast_path": True},
        "replay_branch_metadata": {
            "n_steps": 1,
            "accepted_mask": [True],
            "rejected_mask": [False],
            "done_mask": [True],
        },
        "controller_slot_summary": {"accepted_slots": 1, "rejected_slots": 0},
    }

    report = direct_coil_branch_local_scalars_report_from_complete_fd(
        complete_report,
        branch_local,
        json_safe=True,
    )

    assert report["passed"] is True
    assert report["uses_production_forward"] is True
    assert report["differentiates_adaptive_controller"] is False
    assert report["differentiates_run_free_boundary"] is False
    assert report["scalar_reports"]["aspect"]["abs_error"] == pytest.approx(0.0)
    physical_gate = direct_coil_same_branch_physical_scalar_gate_report(
        complete_report,
        report,
        scalar_keys=("aspect", "qs_total"),
        json_safe=True,
    )
    assert physical_gate["passed"] is True
    assert physical_gate["same_branch"] is True

    failed_report = direct_coil_branch_local_scalars_report_from_complete_fd(
        complete_report,
        {
            **branch_local,
            "uses_production_forward": False,
            "directional_derivatives": {"aspect": jnp.asarray(10.0), "qs_total": jnp.asarray(0.1)},
        },
    )
    assert failed_report["passed"] is False
    assert "branch-local report did not use production forward values" in failed_report["errors"]
    assert failed_report["scalar_reports"]["aspect"]["passed"] is False


def test_branch_local_scalar_report_adapter_records_failure_modes():
    pytest.importorskip("jax")
    from vmec_jax._compat import jnp
    from vmec_jax.free_boundary_adjoint import (
        direct_coil_adaptive_full_loop_same_branch_gate_report,
        direct_coil_branch_local_scalars_report_from_complete_fd,
        direct_coil_same_branch_physical_scalar_gate_report,
    )

    gate = _same_branch_replay_gate_stub()
    complete_report = {
        "branch_compatibility": gate["branch"],
        "trace_replay_diagnostics": gate["trace_replay_diagnostics"],
        "objective_values": {
            "aspect": {"base": 6.0, "plus": 6.1, "minus": 5.9, "central_fd_directional": 0.2},
            "qs_total": {"base": 0.4, "plus": 0.42, "minus": 0.38, "central_fd_directional": 0.1},
        },
    }

    valid_branch_local = {
        "uses_production_forward": True,
        "differentiates_adaptive_controller": False,
        "differentiates_run_free_boundary": False,
        "differentiates_fixed_accepted_branch": True,
        "derivative_mode": "directional_jvp",
        "replay_ad_mode": "direct",
        "scalar_keys": ("aspect", "qs_total"),
        "values": {"aspect": jnp.asarray(6.0), "qs_total": jnp.asarray(0.4)},
        "replay_value_map": {"aspect": jnp.asarray(6.0), "qs_total": jnp.asarray(0.4)},
        "directional_derivatives": {"aspect": jnp.asarray(0.2), "qs_total": jnp.asarray(0.1)},
        "replay_option_flags": {"use_stacked_step_controls": False, "use_accepted_only_fast_path": True},
        "replay_branch_metadata": {"accepted_mask": [True], "rejected_mask": [False]},
        "controller_slot_summary": {"accepted_slots": 1, "rejected_slots": 0},
    }

    with pytest.raises(ValueError, match="scalar_keys"):
        direct_coil_branch_local_scalars_report_from_complete_fd(
            complete_report,
            {**valid_branch_local, "scalar_keys": ()},
        )

    incomplete_report = direct_coil_branch_local_scalars_report_from_complete_fd(
        complete_report,
        {
            **valid_branch_local,
            "uses_production_forward": False,
            "differentiates_adaptive_controller": True,
            "differentiates_run_free_boundary": True,
            "differentiates_fixed_accepted_branch": False,
            "directional_derivatives": None,
            "values": {"aspect": jnp.asarray(6.0)},
        },
        scalar_keys=("aspect", "qs_total", "missing"),
        json_safe=True,
    )
    assert incomplete_report["passed"] is False
    assert "branch-local report does not contain directional derivatives" in incomplete_report["errors"]
    assert "branch-local report did not use production forward values" in incomplete_report["errors"]
    assert "branch-local report unexpectedly claims adaptive-controller differentiation" in incomplete_report["errors"]
    assert "branch-local report unexpectedly claims run_free_boundary differentiation" in incomplete_report["errors"]
    assert "branch-local report does not differentiate the fixed accepted branch" in incomplete_report["errors"]
    assert "aspect: missing branch-local directional derivative" in incomplete_report["errors"]
    assert "qs_total: missing branch-local directional derivative" in incomplete_report["errors"]
    assert "missing: missing complete-solve objective values" in incomplete_report["errors"]
    assert incomplete_report["scalar_reports"] == {}

    replay_delta_report = direct_coil_branch_local_scalars_report_from_complete_fd(
        complete_report,
        {
            **valid_branch_local,
            "values": {"aspect": jnp.asarray(6.0), "qs_total": jnp.asarray(0.4)},
            "replay_value_map": {"aspect": jnp.asarray(6.01), "qs_total": jnp.asarray(0.4)},
            "base_abs_delta": {},
        },
        scalar_keys=("aspect",),
        base_value_atol={"aspect": 1.0e-3},
    )
    assert replay_delta_report["passed"] is False
    assert replay_delta_report["scalar_reports"]["aspect"]["base_abs_delta"] == pytest.approx(1.0e-2)

    complete_delta_report = direct_coil_branch_local_scalars_report_from_complete_fd(
        complete_report,
        {
            **valid_branch_local,
            "values": {"aspect": jnp.asarray(6.03), "qs_total": jnp.asarray(0.4)},
            "replay_value_map": {},
            "base_abs_delta": {},
        },
        scalar_keys=("aspect",),
        base_value_atol=1.0e-3,
    )
    assert complete_delta_report["passed"] is False
    assert complete_delta_report["scalar_reports"]["aspect"]["base_abs_delta"] == pytest.approx(3.0e-2)

    missing_scalar_gate = direct_coil_same_branch_physical_scalar_gate_report(
        complete_report,
        replay_delta_report,
        scalar_keys=("aspect", "qs_total", "missing"),
        json_safe=True,
    )
    assert missing_scalar_gate["passed"] is False
    assert "aspect: scalar AD-vs-FD report failed" in missing_scalar_gate["errors"]
    assert "qs_total: missing scalar report" in missing_scalar_gate["errors"]
    assert "missing: missing scalar report" in missing_scalar_gate["errors"]

    changed_branch = {
        **complete_report,
        "branch_compatibility": {
            **gate["branch"],
            "same_branch": False,
            "same_accepted_trace_branch": False,
            "same_residual_branch": False,
        },
    }
    changed_branch_report = direct_coil_branch_local_scalars_report_from_complete_fd(
        changed_branch,
        valid_branch_local,
        scalar_keys=("aspect",),
    )
    changed_gate = direct_coil_same_branch_physical_scalar_gate_report(
        changed_branch,
        changed_branch_report,
        scalar_keys=("aspect",),
    )
    assert changed_gate["passed"] is False
    assert "same-branch replay gate failed" in changed_gate["errors"]
    assert "scalar report is not same-branch" in changed_gate["errors"]
    assert "accepted-trace branch fingerprint changed" in changed_gate["errors"]
    assert "residual-controller branch fingerprint changed" in changed_gate["errors"]

    adaptive_gate = direct_coil_adaptive_full_loop_same_branch_gate_report(
        complete_report,
        replay_delta_report,
        scalar_keys=("aspect",),
        require_stacked_step_controls=True,
        require_fixed_rejected_controller_slot=True,
        require_status_derived_rejected_controller_slot=True,
        json_safe=True,
    )
    assert adaptive_gate["passed"] is False
    assert "stacked step-control replay was not used" in adaptive_gate["errors"]
    assert "fixed rejected controller slot was not replayed" in adaptive_gate["errors"]
    assert "accepted-only fast path was used for a rejected-slot replay gate" in adaptive_gate["errors"]
    assert "rejected controller slot was not derived from trace step_status" in adaptive_gate["errors"]
    assert "base: missing complete-solve payload" in adaptive_gate["errors"]
    assert "stacked step-policy branch changed" in adaptive_gate["errors"]


def test_same_branch_report_anchor_uses_best_or_initial_coil_point():
    module = _load_example_module()
    base_params, _metadata = module.make_circle_provider(current_scale=1.0)
    _x0, variables = module.select_coil_variables(
        base_params,
        max_current_vars=1,
        max_fourier_vars=1,
    )
    args = SimpleNamespace(
        current_step=0.1,
        dof_step=0.5,
        same_branch_report_anchor="best",
    )

    best_params, anchor = module.same_branch_report_anchor_params(
        base_params,
        {"x": np.asarray([1.0, -2.0])},
        variables,
        args,
    )
    assert anchor == "best"
    assert float(np.asarray(best_params.base_currents)[0]) == pytest.approx(2.2)
    assert float(np.asarray(best_params.base_curve_dofs)[variables[1][1]]) == pytest.approx(0.4)

    args.same_branch_report_anchor = "initial"
    initial_params, anchor = module.same_branch_report_anchor_params(base_params, {"x": np.asarray([1.0, -2.0])}, variables, args)
    assert anchor == "initial"
    assert float(np.asarray(initial_params.base_currents)[0]) == pytest.approx(float(np.asarray(base_params.base_currents)[0]))
    assert float(np.asarray(initial_params.base_curve_dofs)[variables[1][1]]) == pytest.approx(
        float(np.asarray(base_params.base_curve_dofs)[variables[1][1]])
    )

    args.same_branch_report_anchor = "best"
    fallback_params, anchor = module.same_branch_report_anchor_params(base_params, None, variables, args)
    assert anchor == "initial_no_best_available"
    assert float(np.asarray(fallback_params.base_currents)[0]) == pytest.approx(float(np.asarray(base_params.base_currents)[0]))


def test_same_branch_derivative_proposal_uses_gated_directional_report():
    module = _load_example_module()
    report = {
        "branch_compatibility": {"same_branch": True},
        "direction_x": [1.0, 0.0, -1.0],
        "branch_local_vector_jacobian": {
            "available": True,
            "uses_production_forward": True,
            "differentiates_adaptive_controller": False,
            "differentiates_run_free_boundary": False,
            "differentiates_fixed_accepted_branch": True,
            "replay_ad_mode": "direct",
            "derivative_mode": "directional_jvp",
            "max_base_abs_delta": 0.0,
            "scalars": {
                "qs_total": {
                    "value": 0.2,
                    "exact_directional": 3.0,
                    "base_abs_delta": 0.0,
                },
                "aspect": {
                    "value": 5.5,
                    "exact_directional": -4.0,
                    "base_abs_delta": 0.0,
                },
                "mean_iota": {
                    "value": 0.35,
                    "exact_directional": -1.0,
                    "base_abs_delta": 0.0,
                },
            },
        },
    }
    objective_model = {
        "residual_weight": 0.75,
        "qs_weight": 2.0,
        "aspect_weight": 0.5,
        "target_aspect": 6.0,
        "iota_weight": 10.0,
        "target_iota": 0.4,
    }

    proposal = module.same_branch_derivative_proposal_from_report(
        report,
        objective_model,
        {"x": [0.1, 0.2, 0.3]},
        step_size=0.25,
    )

    assert proposal["available"] is True
    assert proposal["same_branch"] is True
    assert proposal["uses_production_forward"] is True
    assert proposal["replay_ad_mode"] == "direct"
    assert proposal["derivative_mode"] == "directional_jvp"
    assert proposal["differentiates_adaptive_controller"] is False
    assert proposal["differentiates_run_free_boundary"] is False
    assert proposal["differentiates_fixed_accepted_branch"] is True
    assert proposal["complete_solve_acceptance_authority"] is True
    assert proposal["max_base_abs_delta"] == pytest.approx(0.0)
    assert proposal["max_base_abs_delta_allowed"] == pytest.approx(2.0e-3)
    assert "complete solve decides acceptance" in proposal["scope"]
    assert proposal["directional_derivative"] == pytest.approx(9.0)
    assert proposal["contributions"]["qs_total"]["contribution"] == pytest.approx(6.0)
    assert proposal["contributions"]["aspect"]["contribution"] == pytest.approx(2.0)
    assert proposal["contributions"]["mean_iota"]["contribution"] == pytest.approx(1.0)
    assert proposal["contributions"]["mean_iota"]["target"] == pytest.approx(0.4)
    assert proposal["objective_terms_used"] == ["aspect", "mean_iota", "qs_total"]
    assert proposal["objective_terms_omitted"]["residual_proxy"]["weight"] == pytest.approx(0.75)
    assert "complete free-boundary solve" in proposal["objective_terms_omitted"]["residual_proxy"]["reason"]
    assert proposal["alpha"] == pytest.approx(-0.25)
    np.testing.assert_allclose(proposal["trial_x"], [-0.15, 0.2, 0.55])


def test_same_branch_derivative_proposal_rejects_adaptive_claims():
    module = _load_example_module()

    proposal = module.same_branch_derivative_proposal_from_report(
        {
            "direction_x": [1.0],
            "branch_compatibility": {"same_branch": True},
            "branch_local_vector_jacobian": {
                "available": True,
                "uses_production_forward": True,
                "differentiates_adaptive_controller": True,
                "differentiates_run_free_boundary": False,
                "differentiates_fixed_accepted_branch": True,
                "replay_ad_mode": "direct",
                "derivative_mode": "directional_jvp",
                "max_base_abs_delta": 0.0,
                "scalars": {"qs_total": {"value": 0.2, "exact_directional": 1.0, "base_abs_delta": 0.0}},
            },
        },
        {"qs_weight": 1.0},
        {"x": [0.0]},
        step_size=0.1,
    )

    assert proposal["available"] is False
    assert "adaptive-controller" in proposal["reason"]


def test_same_branch_derivative_proposal_rejects_failed_vector_gate():
    module = _load_example_module()

    proposal = module.same_branch_derivative_proposal_from_report(
        {
            "direction_x": [1.0],
            "branch_compatibility": {"same_branch": True},
            "branch_local_vector_gate": {
                "available": True,
                "passed": False,
                "physical_scalar_gate": {"passed": False},
            },
            "branch_local_vector_jacobian": {
                "available": True,
                "uses_production_forward": True,
                "differentiates_adaptive_controller": False,
                "differentiates_run_free_boundary": False,
                "differentiates_fixed_accepted_branch": True,
                "replay_ad_mode": "direct",
                "derivative_mode": "directional_jvp",
                "max_base_abs_delta": 0.0,
                "scalars": {"qs_total": {"value": 0.2, "exact_directional": 1.0, "base_abs_delta": 0.0}},
            },
        },
        {"qs_weight": 1.0},
        {"x": [0.0]},
        step_size=0.1,
    )

    assert proposal["available"] is False
    assert "vector gate" in proposal["reason"]


def test_same_branch_derivative_proposal_rejects_failed_rejected_slot_gate():
    module = _load_example_module()

    proposal = module.same_branch_derivative_proposal_from_report(
        {
            "direction_x": [1.0],
            "branch_compatibility": {"same_branch": True},
            "accepted_rejected_controller_slot_gate": {
                "requested": True,
                "available": True,
                "passed": False,
            },
            "branch_local_vector_jacobian": {
                "available": True,
                "uses_production_forward": True,
                "differentiates_adaptive_controller": False,
                "differentiates_run_free_boundary": False,
                "differentiates_fixed_accepted_branch": True,
                "replay_ad_mode": "direct",
                "derivative_mode": "directional_jvp",
                "max_base_abs_delta": 0.0,
                "scalars": {"qs_total": {"value": 0.2, "exact_directional": 1.0, "base_abs_delta": 0.0}},
            },
        },
        {"qs_weight": 1.0},
        {"x": [0.0]},
        step_size=0.1,
    )

    assert proposal["available"] is False
    assert "accepted/rejected controller-slot gate" in proposal["reason"]


def test_same_branch_derivative_proposal_requires_direct_jvp_and_fresh_replay():
    module = _load_example_module()
    base_report = {
        "branch_compatibility": {"same_branch": True},
        "direction_x": [1.0],
        "branch_local_vector_jacobian": {
            "available": True,
            "uses_production_forward": True,
            "differentiates_adaptive_controller": False,
            "differentiates_run_free_boundary": False,
            "differentiates_fixed_accepted_branch": True,
            "replay_ad_mode": "direct",
            "derivative_mode": "directional_jvp",
            "max_base_abs_delta": 0.0,
            "scalars": {"qs_total": {"value": 0.2, "exact_directional": 1.0, "base_abs_delta": 0.0}},
        },
    }
    objective_model = {
        "residual_weight": 0.0,
        "qs_weight": 1.0,
        "aspect_weight": 0.0,
        "iota_weight": 0.0,
    }
    best = {"x": [0.0]}

    custom_vjp_report = json.loads(json.dumps(base_report))
    custom_vjp_report["branch_local_vector_jacobian"]["replay_ad_mode"] = "custom_vjp"
    proposal = module.same_branch_derivative_proposal_from_report(
        custom_vjp_report,
        objective_model,
        best,
        step_size=0.1,
    )
    assert proposal["available"] is False
    assert "direct JVP" in proposal["reason"]

    stale_report = json.loads(json.dumps(base_report))
    stale_report["branch_local_vector_jacobian"]["max_base_abs_delta"] = 1.0e-2
    proposal = module.same_branch_derivative_proposal_from_report(
        stale_report,
        objective_model,
        best,
        step_size=0.1,
        max_base_abs_delta=1.0e-3,
    )
    assert proposal["available"] is False
    assert "exceeds proposal cap" in proposal["reason"]

    changed_branch_report = json.loads(json.dumps(base_report))
    changed_branch_report["branch_compatibility"]["same_branch"] = False
    proposal = module.same_branch_derivative_proposal_from_report(
        changed_branch_report,
        objective_model,
        best,
        step_size=0.1,
    )
    assert proposal["available"] is False
    assert "branch fingerprint" in proposal["reason"]


def test_nestor_profile_policy_requires_size_and_speedup_thresholds():
    module = _load_example_module()

    low_modes = module.nestor_profile_policy_from_results(
        [
            {"available": True, "nestor_solve_mode": "dense", "wall_s": 10.0},
            {
                "available": True,
                "nestor_solve_mode": "matrix_free",
                "nestor_operator_solver": "gmres",
                "wall_s": 5.0,
            },
        ],
        mode_count=32,
        min_mode_count=96,
        min_speedup=1.15,
    )
    assert low_modes["promote_matrix_free"] is False
    assert "below threshold" in low_modes["reason"]

    slow_matrix_free = module.nestor_profile_policy_from_results(
        [
            {"available": True, "nestor_solve_mode": "dense", "wall_s": 10.0},
            {
                "available": True,
                "nestor_solve_mode": "matrix_free",
                "nestor_operator_solver": "bicgstab",
                "wall_s": 9.5,
            },
        ],
        mode_count=144,
        min_mode_count=96,
        min_speedup=1.15,
    )
    assert slow_matrix_free["promote_matrix_free"] is False
    assert "speedup" in slow_matrix_free["reason"]

    promoted = module.nestor_profile_policy_from_results(
        [
            {"available": True, "nestor_solve_mode": "dense", "wall_s": 10.0},
            {
                "available": True,
                "nestor_solve_mode": "matrix_free",
                "nestor_operator_solver": "gmres",
                "wall_s": 6.0,
            },
        ],
        mode_count=144,
        min_mode_count=96,
        min_speedup=1.15,
    )
    assert promoted["promote_matrix_free"] is True
    assert promoted["matrix_free_best_solver"] == "gmres"

    assert module.parse_profile_matrix_free_solvers("gmres,bicgstab") == ("gmres", "bicgstab")
    with pytest.raises(ValueError, match="unsupported"):
        module.parse_profile_matrix_free_solvers("cg")


def test_same_branch_report_writer_uses_source_helper(tmp_path, monkeypatch):
    module = _load_example_module()
    base_params, _metadata = module.make_circle_provider(current_scale=1.0)
    _x0, variables = module.select_coil_variables(
        base_params,
        max_current_vars=1,
        max_fourier_vars=1,
    )
    args = SimpleNamespace(
        current_step=0.02,
        dof_step=1.0e-3,
        target_aspect=6.0,
        target_iota=0.4,
        helicity_m=1,
        helicity_n=0,
        qs_surfaces="0.25,0.5",
        qs_ntheta=15,
        qs_nphi=16,
        residual_weight=1.0,
        qs_weight=2.0,
        aspect_weight=1.0e-2,
        iota_weight=1.0,
        same_branch_report_eps=1.0e-4,
        same_branch_report_mode="none",
        same_branch_report_max_iter=3,
        vmec_max_iter=2,
        ftol=1.0e-8,
        jit_forces=False,
        activate_fsq=1.0e99,
    )
    calls = []

    def fake_report(input_path, params, *, params_for, objective_fn, eps, solve_kwargs):
        calls.append(
            {
                "input_path": input_path,
                "params": params,
                "plus_current": float(np.asarray(params_for(eps).base_currents)[0]),
                "minus_current": float(np.asarray(params_for(-eps).base_currents)[0]),
                "eps": eps,
                "solve_kwargs": solve_kwargs,
            }
        )
        return {
            "branch_compatibility": {
                "same_branch": True,
                "plus": {
                    "changed_fields": (),
                    "max_abs_scalar_delta": 0.0,
                    "max_rel_scalar_delta": 0.0,
                },
                "minus": {
                    "changed_fields": (),
                    "max_abs_scalar_delta": 0.0,
                    "max_rel_scalar_delta": 0.0,
                },
            },
            "values": {
                "base": 1.0,
                "plus": 1.1,
                "minus": 0.9,
                "central_fd_directional": 1000.0,
            },
            "objective_values": {
                "objective": {
                    "base": 1.0,
                    "plus": 1.1,
                    "minus": 0.9,
                    "central_fd_directional": 1000.0,
                },
                "qs_total": {
                    "base": 0.5,
                    "plus": 0.6,
                    "minus": 0.4,
                    "central_fd_directional": 1000.0,
                },
                "aspect": {
                    "base": 6.0,
                    "plus": 6.1,
                    "minus": 5.9,
                    "central_fd_directional": 1000.0,
                },
            },
            "primary_objective": "objective",
        }

    import vmec_jax.free_boundary_adjoint as freeb_adj

    monkeypatch.setattr(freeb_adj, "direct_coil_same_branch_complete_solve_fd_report", fake_report)
    path = module.write_same_branch_validation_report(
        input_path=tmp_path / "input.direct",
        base_params=base_params,
        variables=variables,
        args=args,
        outdir=tmp_path,
    )

    assert path == tmp_path / "same_branch_complete_solve_report.json"
    assert calls
    assert calls[0]["solve_kwargs"]["max_iter"] == 3
    assert calls[0]["solve_kwargs"]["jit_forces"] is False
    assert calls[0]["plus_current"] > float(np.asarray(base_params.base_currents)[0])
    assert calls[0]["minus_current"] < float(np.asarray(base_params.base_currents)[0])
    report = json.loads(path.read_text())
    assert report["branch_compatibility"]["same_branch"] is True
    assert report["values"]["central_fd_directional"] == pytest.approx(1000.0)
    assert set(report["objective_values"]) == {"objective", "qs_total", "aspect"}
    assert report["primary_objective"] == "objective"
    assert report["branch_local_scalar_gradient"]["available"] is False
    assert report["branch_local_vector_jacobian"]["available"] is False
    assert "adaptive host branch" in report["branch_local_vector_jacobian"]["scope"]
    assert [record["kind"] for record in report["direction_variables"]] == ["current", "fourier_dof"]
    assert report["timings"]["complete_solve_fd_wall_s"] >= 0.0
    assert "branch_local_scalar_wall_s" not in report["timings"]
    assert "branch_local_vector_wall_s" not in report["timings"]


def test_same_branch_report_writer_records_branch_local_scalar_gradient(tmp_path, monkeypatch):
    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp

    module = _load_example_module()
    base_params, _metadata = module.make_circle_provider(current_scale=1.0)
    _x0, variables = module.select_coil_variables(
        base_params,
        max_current_vars=1,
        max_fourier_vars=1,
    )
    args = SimpleNamespace(
        current_step=0.02,
        dof_step=1.0e-3,
        target_aspect=6.0,
        target_iota=0.4,
        helicity_m=1,
        helicity_n=0,
        qs_surfaces="0.25,0.5",
        qs_ntheta=15,
        qs_nphi=16,
        residual_weight=1.0,
        qs_weight=2.0,
        aspect_weight=1.0e-2,
        iota_weight=1.0,
        same_branch_report_eps=1.0e-4,
        same_branch_report_mode="scalar",
        same_branch_report_scalar_key="aspect",
        same_branch_report_max_iter=3,
        vmec_max_iter=2,
        ftol=1.0e-8,
        jit_forces=False,
        activate_fsq=1.0e99,
    )

    def fake_report(*_args, **_kwargs):
        gate = _same_branch_replay_gate_stub()
        return {
            "base": {"traces": ("synthetic-trace",)},
            "branch_compatibility": {
                **gate["branch"],
                "same_branch": True,
                "plus": {"changed_fields": (), "max_abs_scalar_delta": 0.0, "max_rel_scalar_delta": 0.0},
                "minus": {"changed_fields": (), "max_abs_scalar_delta": 0.0, "max_rel_scalar_delta": 0.0},
            },
            "trace_replay_diagnostics": gate["trace_replay_diagnostics"],
            "values": {"base": 1.0, "plus": 1.1, "minus": 0.9, "central_fd_directional": 1000.0},
            "objective_values": {
                "objective": {"base": 1.0, "plus": 1.1, "minus": 0.9, "central_fd_directional": 1000.0},
                "qs_total": {"base": 0.4, "plus": 0.42, "minus": 0.38, "central_fd_directional": 0.4},
                "aspect": {"base": 6.0, "plus": 6.07, "minus": 5.93, "central_fd_directional": 0.7},
            },
            "primary_objective": "objective",
        }

    direction_x = module.same_branch_direction_from_variables(variables)
    direction_params = module.coil_param_direction_from_variables(
        base_params,
        direction_x,
        variables,
        current_step=args.current_step,
        dof_step=args.dof_step,
    )
    grad = jax.tree_util.tree_map(lambda leaf: jnp.ones_like(jnp.asarray(leaf)), direction_params)

    def fake_branch_local_scalar(*_args, **kwargs):
        assert kwargs["scalar_key"] == "aspect"
        assert kwargs["replay_ad_mode"] == "direct"
        assert kwargs["include_trace_replay_diagnostics"] is False
        assert kwargs["include_payload"] is False
        assert kwargs["include_replay_graph_metadata"] is False
        assert kwargs["replay_kwargs"]["state_only_replay"] is True
        assert kwargs["replay_kwargs"]["include_analytic"] is True
        assert kwargs["replay_kwargs"]["include_mode_diagnostics"] is False
        assert kwargs["replay_kwargs"]["nestor_solve_mode"] == "dense"
        assert kwargs["replay_kwargs"]["nestor_operator_solver"] == "gmres"
        assert kwargs["replay_kwargs"]["nestor_operator_tol"] == pytest.approx(1.0e-11)
        assert kwargs["replay_kwargs"]["nestor_operator_atol"] == pytest.approx(1.0e-13)
        assert kwargs["replay_kwargs"]["nestor_operator_maxiter"] is None
        assert kwargs["replay_kwargs"]["nestor_operator_restart"] is None
        assert kwargs["replay_kwargs"]["freeze_vacuum_field"] is False
        assert kwargs["replay_kwargs"]["freeze_freeb_bsqvac"] is False
        return {
            "uses_production_forward": True,
            "differentiates_adaptive_controller": False,
            "differentiates_run_free_boundary": False,
            "differentiates_fixed_accepted_branch": True,
            "replay_ad_mode": "direct",
            "scalar_key": "aspect",
            "includes_payload": False,
            "includes_replay_graph_metadata": False,
            "replay_option_flags": {
                "use_stacked_step_controls": True,
                "state_only_replay": True,
                "include_mode_diagnostics": False,
                "nestor_solve_mode": "dense",
                "nestor_operator_solver": "gmres",
                "nestor_operator_tol": 1.0e-11,
                "nestor_operator_atol": 1.0e-13,
                "nestor_operator_maxiter": None,
                "nestor_operator_restart": None,
            },
            "replay_graph_metadata": {
                "omitted": True,
                "differentiates_adaptive_controller": False,
            },
            "value": 6.0,
            "replay_value": jnp.asarray(6.0),
            "base_abs_delta": 0.0,
            "grad": grad,
            "timings": {
                "production_scalar_eval_wall_s": 0.01,
                "replay_value_and_grad_dispatch_s": 0.02,
                "replay_value_and_grad_ready_s": 0.03,
                "replay_value_and_grad_wall_s": 0.05,
                "trace_replay_diagnostics_wall_s": 0.004,
                "total_wall_s": 0.07,
            },
        }

    import vmec_jax.free_boundary_adjoint as freeb_adj

    monkeypatch.setattr(freeb_adj, "direct_coil_same_branch_complete_solve_fd_report", fake_report)
    monkeypatch.setattr(
        freeb_adj,
        "direct_coil_run_free_boundary_branch_local_scalar_value_and_grad_jax",
        fake_branch_local_scalar,
    )

    path = module.write_same_branch_validation_report(
        input_path=tmp_path / "input.direct",
        base_params=base_params,
        variables=variables,
        args=args,
        outdir=tmp_path,
    )

    report = json.loads(path.read_text())
    scalar = report["branch_local_scalar_gradient"]
    expected_directional = module._pytree_directional_vdot(grad, direction_params)
    assert scalar["available"] is True
    assert "fixed accepted branch only" in scalar["scope"]
    assert scalar["mode"] == "scalar"
    assert scalar["uses_production_forward"] is True
    assert scalar["differentiates_adaptive_controller"] is False
    assert scalar["differentiates_run_free_boundary"] is False
    assert scalar["differentiates_fixed_accepted_branch"] is True
    assert scalar["replay_ad_mode"] == "direct"
    assert scalar["scalar_key"] == "aspect"
    assert scalar["state_only_replay"] is True
    assert scalar["includes_payload"] is False
    assert scalar["includes_replay_graph_metadata"] is False
    assert scalar["replay_option_flags"]["state_only_replay"] is True
    assert scalar["replay_option_flags"]["nestor_solve_mode"] == "dense"
    assert scalar["replay_option_flags"]["nestor_operator_solver"] == "gmres"
    assert scalar["replay_graph_metadata"]["omitted"] is True
    assert scalar["value"] == pytest.approx(6.0)
    assert scalar["replay_value"] == pytest.approx(6.0)
    assert scalar["base_abs_delta"] == pytest.approx(0.0)
    assert scalar["exact_directional"] == pytest.approx(expected_directional)
    assert scalar["complete_fd_directional"] == pytest.approx(0.7)
    assert scalar["abs_error"] == pytest.approx(abs(float(expected_directional) - 0.7))
    assert report["branch_local_vector_jacobian"]["available"] is False
    assert report["timings"]["complete_solve_fd_wall_s"] >= 0.0
    assert report["timings"]["branch_local_scalar_wall_s"] >= 0.0
    assert scalar["timings"]["replay_value_and_grad_wall_s"] == pytest.approx(0.05)
    assert report["timings"]["branch_local_scalar_replay_value_and_grad_wall_s"] == pytest.approx(0.05)
    assert "branch_local_vector_wall_s" not in report["timings"]


def test_same_branch_report_writer_records_branch_local_vector_jacobian(tmp_path, monkeypatch):
    pytest.importorskip("jax")
    from vmec_jax._compat import jnp

    module = _load_example_module()
    base_params, _metadata = module.make_circle_provider(current_scale=1.0)
    _x0, variables = module.select_coil_variables(
        base_params,
        max_current_vars=1,
        max_fourier_vars=1,
    )
    args = SimpleNamespace(
        current_step=0.02,
        dof_step=1.0e-3,
        target_aspect=6.0,
        target_iota=0.4,
        helicity_m=1,
        helicity_n=0,
        qs_surfaces="0.25,0.5",
        qs_ntheta=15,
        qs_nphi=16,
        residual_weight=1.0,
        qs_weight=2.0,
        aspect_weight=1.0e-2,
        iota_weight=1.0,
        same_branch_report_eps=1.0e-4,
        same_branch_report_mode="vector",
        same_branch_report_vector_keys="aspect,qs_total,mean_iota,lcfs_boundary_moment",
        same_branch_report_max_iter=3,
        same_branch_report_disable_analytic=True,
        same_branch_report_freeze_vacuum_field=True,
        same_branch_report_freeze_bsqvac=True,
        same_branch_report_nestor_solve_mode="matrix_free",
        same_branch_report_nestor_operator_solver="bicgstab",
        same_branch_report_nestor_operator_tol=2.0e-9,
        same_branch_report_nestor_operator_atol=3.0e-12,
        same_branch_report_nestor_operator_maxiter=17,
        same_branch_report_nestor_operator_restart=5,
        vmec_max_iter=2,
        ftol=1.0e-8,
        jit_forces=False,
        activate_fsq=1.0e99,
    )

    def fake_report(*_args, **_kwargs):
        gate = _same_branch_replay_gate_stub()
        return {
            "base": {"traces": ("synthetic-trace",)},
            "branch_compatibility": {
                **gate["branch"],
                "same_branch": True,
                "plus": {"changed_fields": (), "max_abs_scalar_delta": 0.0, "max_rel_scalar_delta": 0.0},
                "minus": {"changed_fields": (), "max_abs_scalar_delta": 0.0, "max_rel_scalar_delta": 0.0},
            },
            "trace_replay_diagnostics": gate["trace_replay_diagnostics"],
            "values": {"base": 1.0, "plus": 1.1, "minus": 0.9, "central_fd_directional": 1000.0},
            "objective_values": {
                "objective": {"base": 1.0, "plus": 1.1, "minus": 0.9, "central_fd_directional": 1000.0},
                "aspect": {"base": 6.0, "plus": 6.1, "minus": 5.9, "central_fd_directional": 0.1},
                "mean_iota": {"base": 0.4, "plus": 0.41, "minus": 0.39, "central_fd_directional": 0.2},
                "qs_total": {"base": 0.4, "plus": 0.42, "minus": 0.38, "central_fd_directional": 0.4},
                "lcfs_boundary_moment": {
                    "base": 0.2,
                    "plus": 0.21,
                    "minus": 0.19,
                    "central_fd_directional": 0.2,
                },
                "accepted_bnormal_rms": {
                    "base": 0.3,
                    "plus": 0.31,
                    "minus": 0.29,
                    "central_fd_directional": 0.3,
                },
            },
            "primary_objective": "objective",
        }

    direction_x = module.same_branch_direction_from_variables(variables)
    direction_params = module.coil_param_direction_from_variables(
        base_params,
        direction_x,
        variables,
        current_step=args.current_step,
        dof_step=args.dof_step,
    )
    def fake_branch_local_vector(*_args, **kwargs):
        assert kwargs["replay_ad_mode"] == "direct"
        assert kwargs["include_trace_replay_diagnostics"] is False
        assert kwargs["include_payload"] is False
        assert kwargs["include_replay_graph_metadata"] is False
        assert kwargs["replay_kwargs"]["state_only_replay"] is True
        assert kwargs["replay_kwargs"]["include_analytic"] is False
        assert kwargs["replay_kwargs"]["include_mode_diagnostics"] is False
        assert kwargs["replay_kwargs"]["nestor_solve_mode"] == "matrix_free"
        assert kwargs["replay_kwargs"]["nestor_operator_solver"] == "bicgstab"
        assert kwargs["replay_kwargs"]["nestor_operator_tol"] == pytest.approx(2.0e-9)
        assert kwargs["replay_kwargs"]["nestor_operator_atol"] == pytest.approx(3.0e-12)
        assert kwargs["replay_kwargs"]["nestor_operator_maxiter"] == 17
        assert kwargs["replay_kwargs"]["nestor_operator_restart"] == 5
        assert kwargs["replay_kwargs"]["freeze_vacuum_field"] is True
        assert kwargs["replay_kwargs"]["freeze_freeb_bsqvac"] is True
        assert kwargs["direction_params"] is not None
        assert kwargs["direction_params"].n_segments == direction_params.n_segments
        assert kwargs["scalar_keys"] == ("aspect", "qs_total", "mean_iota", "lcfs_boundary_moment")
        return {
            "uses_production_forward": True,
            "differentiates_adaptive_controller": False,
            "differentiates_run_free_boundary": False,
            "differentiates_fixed_accepted_branch": True,
            "replay_ad_mode": "direct",
            "derivative_mode": "directional_jvp",
            "scalar_keys": ("aspect", "qs_total", "mean_iota", "lcfs_boundary_moment"),
            "includes_payload": False,
            "includes_replay_graph_metadata": False,
            "replay_option_flags": {
                "use_stacked_step_controls": True,
                "state_only_replay": True,
                "include_mode_diagnostics": False,
                "nestor_solve_mode": "matrix_free",
                "nestor_operator_solver": "bicgstab",
                "nestor_operator_tol": 2.0e-9,
                "nestor_operator_atol": 3.0e-12,
                "nestor_operator_maxiter": 17,
                "nestor_operator_restart": 5,
                "directional_jvp_fast_path": "current_only",
                "directional_uses_fixed_coil_geometry": True,
            },
            "replay_graph_metadata": {
                "omitted": True,
                "differentiates_adaptive_controller": False,
            },
            "replay_branch_metadata": {
                "n_steps": 1,
                "n_free_boundary_replay_steps": 1,
                "accepted_mask": [True],
                "rejected_mask": [False],
                "done_mask": [True],
                "has_active_freeb_replay": [True],
            },
            "max_base_abs_delta": 0.0,
            "values": {
                "aspect": 6.0,
                "qs_total": 0.4,
                "mean_iota": 0.4,
                "lcfs_boundary_moment": 0.2,
            },
            "replay_value_map": {
                "aspect": jnp.asarray(6.0),
                "qs_total": jnp.asarray(0.4),
                "mean_iota": jnp.asarray(0.4),
                "lcfs_boundary_moment": jnp.asarray(0.2),
            },
            "base_abs_delta": {
                "aspect": 0.0,
                "qs_total": 0.0,
                "mean_iota": 0.0,
                "lcfs_boundary_moment": 0.0,
            },
            "jacobian": None,
            "directional_derivatives": {
                "aspect": jnp.asarray(0.1),
                "qs_total": jnp.asarray(0.4),
                "mean_iota": jnp.asarray(0.2),
                "lcfs_boundary_moment": jnp.asarray(0.2),
            },
            "timings": {
                "production_scalar_eval_wall_s": 0.01,
                "replay_jvp_wall_s": 0.02,
                "replay_vjp_wall_s": 0.0,
                "replay_pullbacks_wall_s": 0.0,
                "jacobian_stack_ready_s": 0.004,
                "total_wall_s": 0.06,
            },
        }

    import vmec_jax.free_boundary_adjoint as freeb_adj

    monkeypatch.setattr(freeb_adj, "direct_coil_same_branch_complete_solve_fd_report", fake_report)
    monkeypatch.setattr(
        freeb_adj,
        "direct_coil_run_free_boundary_branch_local_scalars_value_and_jacobian_jax",
        fake_branch_local_vector,
    )

    path = module.write_same_branch_validation_report(
        input_path=tmp_path / "input.direct",
        base_params=base_params,
        variables=variables,
        args=args,
        outdir=tmp_path,
    )

    report = json.loads(path.read_text())
    vector = report["branch_local_vector_jacobian"]
    assert vector["available"] is True
    assert "fixed accepted branch only" in vector["scope"]
    assert vector["uses_production_forward"] is True
    assert vector["differentiates_adaptive_controller"] is False
    assert vector["differentiates_run_free_boundary"] is False
    assert vector["differentiates_fixed_accepted_branch"] is True
    assert vector["replay_ad_mode"] == "direct"
    assert vector["derivative_mode"] == "directional_jvp"
    assert vector["directional_jvp_fast_path"] == "current_only"
    assert vector["directional_uses_fixed_coil_geometry"] is True
    assert vector["scalar_keys"] == ["aspect", "qs_total", "mean_iota", "lcfs_boundary_moment"]
    assert vector["state_only_replay"] is True
    assert vector["replay_option_flags"]["use_stacked_step_controls"] is True
    assert vector["replay_option_flags"]["state_only_replay"] is True
    assert vector["replay_option_flags"]["nestor_solve_mode"] == "matrix_free"
    assert vector["replay_option_flags"]["nestor_operator_solver"] == "bicgstab"
    assert vector["replay_option_flags"]["nestor_operator_maxiter"] == 17
    assert vector["includes_payload"] is False
    assert vector["includes_replay_graph_metadata"] is False
    assert vector["replay_graph_metadata"]["omitted"] is True
    assert vector["controller_slot_summary"]["accepted_slots"] == 1
    assert vector["controller_slot_summary"]["rejected_slots"] == 0
    assert vector["max_base_abs_delta"] == pytest.approx(0.0)
    expected_directionals = {
        "aspect": 0.1,
        "qs_total": 0.4,
        "mean_iota": 0.2,
        "lcfs_boundary_moment": 0.2,
    }
    for key, expected_directional in expected_directionals.items():
        scalar_evidence = vector["scalars"][key]
        assert scalar_evidence["exact_directional"] == pytest.approx(expected_directional)
        assert scalar_evidence["complete_fd_directional"] == pytest.approx(expected_directional)
        assert scalar_evidence["abs_error"] == pytest.approx(0.0)
        assert scalar_evidence["base_abs_delta"] == pytest.approx(0.0)
    assert report["timings"]["complete_solve_fd_wall_s"] >= 0.0
    assert report["timings"]["branch_local_vector_wall_s"] >= 0.0
    assert vector["timings"]["replay_jvp_wall_s"] == pytest.approx(0.02)
    assert report["timings"]["branch_local_vector_replay_jvp_wall_s"] == pytest.approx(0.02)
    assert report["timings"]["branch_local_vector_replay_pullbacks_wall_s"] == pytest.approx(0.0)
    assert "branch_local_scalar_wall_s" not in report["timings"]
    vector_gate = report["branch_local_vector_gate"]
    assert vector_gate["available"] is True
    assert vector_gate["passed"] is True
    assert vector_gate["differentiates_adaptive_controller"] is False
    assert vector_gate["differentiates_run_free_boundary"] is False
    assert vector_gate["differentiates_fixed_accepted_branch"] is True
    assert vector_gate["scalar_report"]["passed"] is True
    assert vector_gate["physical_scalar_gate"]["passed"] is True
    assert vector_gate["physical_scalar_gate"]["same_branch"] is True


def test_same_branch_report_profiles_nestor_and_rejected_slot(tmp_path, monkeypatch):
    pytest.importorskip("jax")
    from vmec_jax._compat import jnp

    module = _load_example_module()
    base_params, _metadata = module.make_circle_provider(current_scale=1.0)
    _x0, variables = module.select_coil_variables(
        base_params,
        max_current_vars=1,
        max_fourier_vars=1,
    )
    args = SimpleNamespace(
        current_step=0.02,
        dof_step=1.0e-3,
        target_aspect=6.0,
        target_iota=0.4,
        helicity_m=1,
        helicity_n=0,
        qs_surfaces="0.25,0.5",
        qs_ntheta=15,
        qs_nphi=16,
        residual_weight=1.0,
        qs_weight=2.0,
        aspect_weight=1.0e-2,
        iota_weight=1.0,
        same_branch_report_eps=1.0e-4,
        same_branch_report_mode="vector",
        same_branch_report_vector_keys="aspect,qs_total",
        same_branch_report_max_iter=3,
        same_branch_report_disable_analytic=False,
        same_branch_report_freeze_vacuum_field=False,
        same_branch_report_freeze_bsqvac=False,
        same_branch_report_nestor_solve_mode="dense",
        same_branch_report_nestor_operator_solver="gmres",
        same_branch_report_nestor_operator_tol=1.0e-11,
        same_branch_report_nestor_operator_atol=1.0e-13,
        same_branch_report_nestor_operator_maxiter=None,
        same_branch_report_nestor_operator_restart=None,
        same_branch_report_replay_max_mode_count=220,
        same_branch_report_profile_nestor="dense-vs-matrix-free",
        same_branch_report_profile_matrix_free_solvers="gmres,bicgstab",
        same_branch_report_profile_min_mode_count=96,
        same_branch_report_profile_min_speedup=1.15,
        same_branch_report_profile_max_mode_count=220,
        same_branch_report_rejected_slot_gate=True,
        vmec_max_iter=2,
        ftol=1.0e-8,
        jit_forces=False,
        activate_fsq=1.0e99,
    )

    trace = {"state_pre": np.zeros(3), "freeb_bsqvac_half": np.ones(2)}
    init = SimpleNamespace(static=SimpleNamespace(modes=SimpleNamespace(m=np.arange(144))))

    def fake_report(*_args, **_kwargs):
        gate = _same_branch_replay_gate_stub()
        return {
            "base": {"traces": [trace], "init": init},
            "branch_compatibility": {
                **gate["branch"],
                "same_branch": True,
                "plus": {"changed_fields": (), "max_abs_scalar_delta": 0.0, "max_rel_scalar_delta": 0.0},
                "minus": {"changed_fields": (), "max_abs_scalar_delta": 0.0, "max_rel_scalar_delta": 0.0},
            },
            "trace_replay_diagnostics": gate["trace_replay_diagnostics"],
            "values": {"base": 1.0, "plus": 1.1, "minus": 0.9, "central_fd_directional": 1000.0},
            "objective_values": {
                "objective": {"base": 1.0, "plus": 1.1, "minus": 0.9, "central_fd_directional": 1000.0},
                "aspect": {"base": 6.0, "plus": 6.1, "minus": 5.9, "central_fd_directional": 0.1},
                "qs_total": {"base": 0.4, "plus": 0.42, "minus": 0.38, "central_fd_directional": 0.4},
            },
            "primary_objective": "objective",
        }

    calls: list[dict[str, object]] = []

    def fake_branch_local_vector(*_args, **kwargs):
        replay_kwargs = dict(kwargs["replay_kwargs"])
        calls.append(replay_kwargs)
        accept_mask = replay_kwargs.get("accept_mask")
        traces = tuple(replay_kwargs.get("traces", (trace,)))
        if accept_mask is None:
            statuses = [
                str(item.get("step_status", "accepted")).strip().lower() or "accepted"
                for item in traces
            ]
            accepted_mask = [status != "rejected" and not status.startswith("restart_") for status in statuses]
            rejected_mask = [not item for item in accepted_mask]
        else:
            statuses = [
                str(item.get("step_status", "accepted")).strip().lower() or "accepted"
                for item in traces
            ]
            accepted_mask = [bool(item) for item in np.asarray(accept_mask, dtype=bool)]
            rejected_mask = [not item for item in accepted_mask]
        return {
            "uses_production_forward": True,
            "differentiates_adaptive_controller": False,
            "differentiates_run_free_boundary": False,
            "differentiates_fixed_accepted_branch": True,
            "replay_ad_mode": "direct",
            "derivative_mode": "directional_jvp",
            "scalar_keys": ("aspect", "qs_total"),
            "includes_payload": False,
            "includes_replay_graph_metadata": bool(kwargs["include_replay_graph_metadata"]),
            "replay_option_flags": {
                "use_stacked_step_controls": bool(replay_kwargs["use_stacked_step_controls"]),
                "use_accepted_only_fast_path": bool(replay_kwargs["use_accepted_only_fast_path"]),
                "state_only_replay": bool(replay_kwargs["state_only_replay"]),
                "include_mode_diagnostics": False,
                "nestor_solve_mode": str(replay_kwargs["nestor_solve_mode"]),
                "nestor_operator_solver": str(replay_kwargs["nestor_operator_solver"]),
                "nestor_operator_tol": float(replay_kwargs["nestor_operator_tol"]),
                "nestor_operator_atol": float(replay_kwargs["nestor_operator_atol"]),
                "nestor_operator_maxiter": replay_kwargs["nestor_operator_maxiter"],
                "nestor_operator_restart": replay_kwargs["nestor_operator_restart"],
                "directional_jvp_fast_path": "current_only",
                "directional_uses_fixed_coil_geometry": True,
            },
            "replay_graph_metadata": {"omitted": not bool(kwargs["include_replay_graph_metadata"])},
            "replay_branch_metadata": {
                "n_steps": len(accepted_mask),
                "n_free_boundary_replay_steps": int(np.count_nonzero(accepted_mask)),
                "status_masks": {
                    "step_status": statuses,
                    "accept_mask": accepted_mask,
                    "status_acceptance_source": "trace_step_status",
                },
                "status_acceptance_source": "trace_step_status",
                "accepted_mask": accepted_mask,
                "rejected_mask": rejected_mask,
            },
            "max_base_abs_delta": 0.0,
            "values": {
                "aspect": 6.0,
                "qs_total": 0.4,
            },
            "replay_value_map": {
                "aspect": jnp.asarray(6.0),
                "qs_total": jnp.asarray(0.4),
            },
            "base_abs_delta": {
                "aspect": 0.0,
                "qs_total": 0.0,
            },
            "jacobian": None,
            "directional_derivatives": {
                "aspect": jnp.asarray(0.1),
                "qs_total": jnp.asarray(0.4),
            },
            "timings": {
                "production_scalar_eval_wall_s": 0.01,
                "replay_jvp_wall_s": 0.02,
                "replay_vjp_wall_s": 0.0,
                "replay_pullbacks_wall_s": 0.0,
                "jacobian_stack_ready_s": 0.0,
                "total_wall_s": 0.03,
            },
        }

    import vmec_jax.free_boundary_adjoint as freeb_adj

    monkeypatch.setattr(freeb_adj, "direct_coil_same_branch_complete_solve_fd_report", fake_report)
    monkeypatch.setattr(
        freeb_adj,
        "direct_coil_run_free_boundary_branch_local_scalars_value_and_jacobian_jax",
        fake_branch_local_vector,
    )

    path = module.write_same_branch_validation_report(
        input_path=tmp_path / "input.direct",
        base_params=base_params,
        variables=variables,
        args=args,
        outdir=tmp_path,
    )

    report = json.loads(path.read_text())
    assert report["mode_count"] == 144
    assert len(calls) == 5
    assert calls[0]["nestor_solve_mode"] == "dense"
    assert calls[1]["use_accepted_only_fast_path"] is False
    assert "accept_mask" not in calls[1]
    assert [item.get("step_status", "accepted") for item in calls[1]["traces"]] == ["accepted", "rejected"]
    profiled = {(call["nestor_solve_mode"], call["nestor_operator_solver"]) for call in calls[2:]}
    assert profiled == {("dense", "gmres"), ("matrix_free", "gmres"), ("matrix_free", "bicgstab")}

    rejected_gate = report["accepted_rejected_controller_slot_gate"]
    assert rejected_gate["available"] is True
    assert rejected_gate["passed"] is True
    assert rejected_gate["same_branch"] is True
    assert rejected_gate["differentiates_adaptive_controller"] is False
    assert rejected_gate["differentiates_run_free_boundary"] is False
    assert rejected_gate["same_stacked_step_policy_branch"] is True
    assert rejected_gate["fixed_rejected_controller_slot_present"] is True
    assert rejected_gate["fixed_rejected_controller_slots"] == 1
    assert rejected_gate["directional_jvp_fast_path"] == "current_only"
    assert rejected_gate["directional_uses_fixed_coil_geometry"] is True
    assert rejected_gate["controller_slot_summary"]["accepted_slots"] == 1
    assert rejected_gate["controller_slot_summary"]["rejected_slots"] == 1
    assert rejected_gate["replay_option_flags"]["use_accepted_only_fast_path"] is False
    assert rejected_gate["replay_branch_metadata"]["status_acceptance_source"] == "trace_step_status"
    assert rejected_gate["replay_branch_metadata"]["status_masks"]["step_status"] == ["accepted", "rejected"]

    profile = report["nestor_replay_profile"]
    assert profile["enabled"] is True
    assert len(profile["results"]) == 3
    assert {item["nestor_solve_mode"] for item in profile["results"]} == {"dense", "matrix_free"}
    assert profile["policy"]["mode_count"] == 144


def test_same_branch_report_profile_skips_above_mode_count_cap(tmp_path, monkeypatch):
    pytest.importorskip("jax")
    from vmec_jax._compat import jnp

    module = _load_example_module()
    base_params, _metadata = module.make_circle_provider(current_scale=1.0)
    _x0, variables = module.select_coil_variables(
        base_params,
        max_current_vars=1,
        max_fourier_vars=1,
    )
    args = SimpleNamespace(
        current_step=0.02,
        dof_step=1.0e-3,
        target_aspect=6.0,
        target_iota=0.4,
        helicity_m=1,
        helicity_n=0,
        qs_surfaces="0.25,0.5",
        qs_ntheta=15,
        qs_nphi=16,
        residual_weight=1.0,
        qs_weight=2.0,
        aspect_weight=1.0e-2,
        iota_weight=1.0,
        same_branch_report_eps=1.0e-4,
        same_branch_report_mode="vector",
        same_branch_report_vector_keys="aspect,qs_total",
        same_branch_report_max_iter=3,
        same_branch_report_disable_analytic=False,
        same_branch_report_freeze_vacuum_field=False,
        same_branch_report_freeze_bsqvac=False,
        same_branch_report_nestor_solve_mode="dense",
        same_branch_report_nestor_operator_solver="gmres",
        same_branch_report_nestor_operator_tol=1.0e-11,
        same_branch_report_nestor_operator_atol=1.0e-13,
        same_branch_report_nestor_operator_maxiter=None,
        same_branch_report_nestor_operator_restart=None,
        same_branch_report_replay_max_mode_count=100,
        same_branch_report_profile_nestor="dense-vs-matrix-free",
        same_branch_report_profile_matrix_free_solvers="gmres,bicgstab",
        same_branch_report_profile_min_mode_count=96,
        same_branch_report_profile_min_speedup=1.15,
        same_branch_report_profile_max_mode_count=100,
        same_branch_report_rejected_slot_gate=False,
        vmec_max_iter=2,
        ftol=1.0e-8,
        jit_forces=False,
        activate_fsq=1.0e99,
    )

    trace = {"state_pre": np.zeros(3), "freeb_bsqvac_half": np.ones(2)}
    init = SimpleNamespace(static=SimpleNamespace(modes=SimpleNamespace(m=np.arange(144))))

    def fake_report(*_args, **_kwargs):
        return {
            "base": {"traces": [trace], "init": init},
            "branch_compatibility": {
                "same_branch": True,
                "plus": {"changed_fields": (), "max_abs_scalar_delta": 0.0, "max_rel_scalar_delta": 0.0},
                "minus": {"changed_fields": (), "max_abs_scalar_delta": 0.0, "max_rel_scalar_delta": 0.0},
            },
            "values": {"base": 1.0, "plus": 1.1, "minus": 0.9, "central_fd_directional": 1000.0},
            "objective_values": {
                "objective": {"base": 1.0, "plus": 1.1, "minus": 0.9, "central_fd_directional": 1000.0},
                "aspect": {"base": 6.0, "plus": 6.1, "minus": 5.9, "central_fd_directional": 0.1},
                "qs_total": {"base": 0.4, "plus": 0.42, "minus": 0.38, "central_fd_directional": 0.4},
            },
            "primary_objective": "objective",
        }

    calls: list[dict[str, object]] = []

    def fake_branch_local_vector(*_args, **kwargs):
        replay_kwargs = dict(kwargs["replay_kwargs"])
        calls.append(replay_kwargs)
        return {
            "uses_production_forward": True,
            "differentiates_adaptive_controller": False,
            "differentiates_run_free_boundary": False,
            "differentiates_fixed_accepted_branch": True,
            "replay_ad_mode": "direct",
            "derivative_mode": "directional_jvp",
            "scalar_keys": ("aspect", "qs_total"),
            "includes_payload": False,
            "includes_replay_graph_metadata": bool(kwargs["include_replay_graph_metadata"]),
            "replay_option_flags": {
                "use_stacked_step_controls": bool(replay_kwargs["use_stacked_step_controls"]),
                "use_accepted_only_fast_path": bool(replay_kwargs["use_accepted_only_fast_path"]),
                "state_only_replay": bool(replay_kwargs["state_only_replay"]),
                "include_mode_diagnostics": False,
                "nestor_solve_mode": str(replay_kwargs["nestor_solve_mode"]),
                "nestor_operator_solver": str(replay_kwargs["nestor_operator_solver"]),
                "nestor_operator_tol": float(replay_kwargs["nestor_operator_tol"]),
                "nestor_operator_atol": float(replay_kwargs["nestor_operator_atol"]),
                "nestor_operator_maxiter": replay_kwargs["nestor_operator_maxiter"],
                "nestor_operator_restart": replay_kwargs["nestor_operator_restart"],
            },
            "replay_graph_metadata": {"omitted": not bool(kwargs["include_replay_graph_metadata"])},
            "replay_branch_metadata": {
                "n_steps": 1,
                "n_free_boundary_replay_steps": 1,
                "accepted_mask": [True],
                "rejected_mask": [False],
            },
            "max_base_abs_delta": 0.0,
            "values": {"aspect": 6.0, "qs_total": 0.4},
            "replay_value_map": {"aspect": jnp.asarray(6.0), "qs_total": jnp.asarray(0.4)},
            "base_abs_delta": {"aspect": 0.0, "qs_total": 0.0},
            "jacobian": None,
            "directional_derivatives": {"aspect": jnp.asarray(0.1), "qs_total": jnp.asarray(0.4)},
            "timings": {
                "production_scalar_eval_wall_s": 0.01,
                "replay_jvp_wall_s": 0.02,
                "replay_vjp_wall_s": 0.0,
                "replay_pullbacks_wall_s": 0.0,
                "jacobian_stack_ready_s": 0.0,
                "total_wall_s": 0.03,
            },
        }

    import vmec_jax.free_boundary_adjoint as freeb_adj

    monkeypatch.setattr(freeb_adj, "direct_coil_same_branch_complete_solve_fd_report", fake_report)
    monkeypatch.setattr(
        freeb_adj,
        "direct_coil_run_free_boundary_branch_local_scalars_value_and_jacobian_jax",
        fake_branch_local_vector,
    )

    path = module.write_same_branch_validation_report(
        input_path=tmp_path / "input.direct",
        base_params=base_params,
        variables=variables,
        args=args,
        outdir=tmp_path,
    )

    report = json.loads(path.read_text())
    profile = report["nestor_replay_profile"]
    assert len(calls) == 0
    assert report["same_branch_replay_mode_count_guard"]["triggered"] is True
    assert report["branch_local_vector_jacobian"]["available"] is False
    assert profile["enabled"] is True
    assert profile["results"] == []
    assert profile["skipped_due_to_replay_mode_count_cap"] is True
    assert profile["policy"]["promote_matrix_free"] is False
    assert "mode-count cap" in profile["policy"]["reason"]


def test_circle_dry_run_writes_configuration_without_solves(tmp_path, monkeypatch):
    module = _load_example_module()

    def fake_make_free_boundary_indata(_input_path, output_path, **_kwargs):
        output_path.write_text("&INDATA\n/\n")
        return output_path

    def fail_run_direct_free_boundary(*_args, **_kwargs):
        raise AssertionError("dry-run must not call run_direct_free_boundary")

    def fail_minimize(*_args, **_kwargs):
        raise AssertionError("dry-run must not call scipy.optimize.minimize")

    def fail_write_wout(*_args, **_kwargs):
        raise AssertionError("dry-run must not write a best wout")

    monkeypatch.setattr(module, "make_free_boundary_indata", fake_make_free_boundary_indata)
    monkeypatch.setattr(module, "run_direct_free_boundary", fail_run_direct_free_boundary)
    monkeypatch.setattr(module, "write_wout_from_fixed_boundary_run", fail_write_wout)
    fake_scipy_optimize = ModuleType("scipy.optimize")
    fake_scipy_optimize.minimize = fail_minimize
    monkeypatch.setitem(sys.modules, "scipy.optimize", fake_scipy_optimize)

    exit_code = module.main(
        [
            "--smoke",
            "--dry-run",
            "--provider",
            "circle",
            "--helicity-n",
            "-1",
            "--qs-surfaces",
            "0.3,0.7",
            "--outdir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert not (tmp_path / "history.json").exists()
    assert not (tmp_path / "wout_best_direct_coil_qs.nc").exists()
    assert (tmp_path / "input.direct_coil_qs").read_text() == "&INDATA\n/\n"

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert "optimizer" not in summary
    assert "best" not in summary
    assert summary["flow"] == "single_stage_direct_coil_no_mgrid"
    assert summary["workflow"]["field_backend"] == "direct_coils"
    assert summary["workflow"]["python_provider_required"] is True
    assert summary["workflow"]["uses_mgrid_file"] is False
    assert summary["workflow"]["plasma_boundary_optimized"] is False
    assert summary["scope"] == "deterministic coil-only direct-coil free-boundary QS optimization example"
    assert summary["dry_run"] is True
    assert summary["plasma_boundary_optimized"] is False
    assert any("Boozer-space" in limitation for limitation in summary["single_stage_limitations"])
    assert any("full-loop" in limitation for limitation in summary["single_stage_limitations"])
    assert summary["provider"]["provider"] == "circle"
    assert summary["baseline_coils"]["n_base_coils"] == 1
    assert summary["vmec_config"]["external_field_provider_kind"] == "direct_coils"
    assert summary["vmec_config"]["mgrid_file"] == "DIRECT_COILS"
    assert summary["vmec_config"]["uses_generated_mgrid"] is False
    assert summary["vmec_config"]["python_provider_required"] is True
    assert summary["vmec_config"]["uses_mgrid_file"] is False
    assert "Python-provider tag" in summary["vmec_config"]["vmec_input_replay"]
    assert summary["vmec_config"]["vmec_max_iter"] == 2
    assert summary["vmec_config"]["jit_forces"] is True
    assert [record["kind"] for record in summary["optimized_variables"]] == ["current", "fourier_dof"]
    assert summary["optimized_variables"][0]["parameterization"] == "multiplicative"
    assert summary["optimized_variables"][1]["parameterization"] == "additive"
    assert summary["objective_model"]["target_aspect"] == pytest.approx(6.0)
    assert summary["objective_model"]["helicity_n"] == -1
    assert summary["objective_model"]["qs_surfaces"] == [0.3, 0.7]
    assert summary["same_branch_report_config"]["enabled"] is False
    assert summary["same_branch_report_config"]["mode"] == "vector"
    assert summary["same_branch_report_config"]["ad_mode"] == "direct"
    assert summary["same_branch_report_config"]["vector_keys"] == [
        "aspect",
        "qs_total",
        "mean_iota",
        "lcfs_boundary_moment",
    ]


def test_essos_provider_skip_returns_code_77_without_solves(tmp_path, monkeypatch, capsys):
    module = _load_example_module()

    def fake_load_essos_provider(*_args, **_kwargs):
        raise module.SkipExample("synthetic missing ESSOS assets")

    def fail_make_free_boundary_indata(*_args, **_kwargs):
        raise AssertionError("ESSOS skip must happen before input generation")

    def fail_run_direct_free_boundary(*_args, **_kwargs):
        raise AssertionError("ESSOS skip must happen before free-boundary solves")

    def fail_minimize(*_args, **_kwargs):
        raise AssertionError("ESSOS skip must happen before optimizer import/use")

    def fail_write_wout(*_args, **_kwargs):
        raise AssertionError("ESSOS skip must happen before WOUT writes")

    monkeypatch.setattr(module, "load_essos_provider", fake_load_essos_provider)
    monkeypatch.setattr(module, "make_free_boundary_indata", fail_make_free_boundary_indata)
    monkeypatch.setattr(module, "run_direct_free_boundary", fail_run_direct_free_boundary)
    monkeypatch.setattr(module, "write_wout_from_fixed_boundary_run", fail_write_wout)
    fake_scipy_optimize = ModuleType("scipy.optimize")
    fake_scipy_optimize.minimize = fail_minimize
    monkeypatch.setitem(sys.modules, "scipy.optimize", fake_scipy_optimize)

    exit_code = module.main(
        [
            "--smoke",
            "--dry-run",
            "--provider",
            "essos",
            "--outdir",
            str(tmp_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == module.SKIP_EXIT_CODE
    assert "SKIP: synthetic missing ESSOS assets" in captured.err
    assert not (tmp_path / "summary.json").exists()
    assert not (tmp_path / "input.direct_coil_qs").exists()
    assert not (tmp_path / "history.json").exists()
    assert not (tmp_path / "wout_best_direct_coil_qs.nc").exists()


def test_essos_dry_run_writes_direct_coil_configuration_without_mgrid(tmp_path, monkeypatch):
    module = _load_example_module()
    synthetic_params, _metadata = module.make_circle_provider(current_scale=1.0)

    def fake_load_essos_provider(coils_json, *, chunk_size, current_scale):
        assert coils_json is None
        assert chunk_size == 128
        assert current_scale == pytest.approx(1.25)
        return synthetic_params, {
            "provider": "essos",
            "coils_json": "/synthetic/ESSOS_biot_savart_LandremanPaulQA.json",
            "n_base_coils": 1,
            "n_segments": int(synthetic_params.n_segments),
            "nfp": int(synthetic_params.nfp),
            "stellsym": bool(synthetic_params.stellsym),
            "current_scale_multiplier": 1.25,
        }

    def fake_make_free_boundary_indata(_input_path, output_path, **kwargs):
        output_path.write_text("&INDATA\n  LFREEB = T\n  MGRID_FILE = 'DIRECT_COILS'\n/\n")
        assert kwargs["vmec_max_iter"] == 2
        assert kwargs["ftol"] == pytest.approx(1.0e-8)
        return output_path

    def fail_run_direct_free_boundary(*_args, **_kwargs):
        raise AssertionError("dry-run must not call run_direct_free_boundary")

    def fail_minimize(*_args, **_kwargs):
        raise AssertionError("dry-run must not call scipy.optimize.minimize")

    def fail_write_wout(*_args, **_kwargs):
        raise AssertionError("dry-run must not write a best wout")

    monkeypatch.setattr(module, "load_essos_provider", fake_load_essos_provider)
    monkeypatch.setattr(module, "make_free_boundary_indata", fake_make_free_boundary_indata)
    monkeypatch.setattr(module, "run_direct_free_boundary", fail_run_direct_free_boundary)
    monkeypatch.setattr(module, "write_wout_from_fixed_boundary_run", fail_write_wout)
    fake_scipy_optimize = ModuleType("scipy.optimize")
    fake_scipy_optimize.minimize = fail_minimize
    monkeypatch.setitem(sys.modules, "scipy.optimize", fake_scipy_optimize)

    exit_code = module.main(
        [
            "--smoke",
            "--dry-run",
            "--provider",
            "essos",
            "--chunk-size",
            "128",
            "--current-scale",
            "1.25",
            "--max-current-vars",
            "1",
            "--max-fourier-vars",
            "1",
            "--outdir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert not (tmp_path / "history.json").exists()
    assert not (tmp_path / "wout_best_direct_coil_qs.nc").exists()
    generated_input = tmp_path / "input.direct_coil_qs"
    assert "DIRECT_COILS" in generated_input.read_text()

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["dry_run"] is True
    assert summary["plasma_boundary_optimized"] is False
    assert summary["flow"] == "single_stage_direct_coil_no_mgrid"
    assert summary["workflow"]["mgrid_compatibility_example"].endswith("free_boundary_essos_mgrid_forward.py")
    assert summary["provider"]["provider"] == "essos"
    assert summary["provider"]["coils_json"].endswith("ESSOS_biot_savart_LandremanPaulQA.json")
    assert summary["baseline_coils"]["n_base_coils"] == 1
    assert summary["vmec_config"]["generated_input"].endswith("input.direct_coil_qs")
    assert summary["vmec_config"]["generated_input"] == str(generated_input)
    assert summary["vmec_config"]["external_field_provider_kind"] == "direct_coils"
    assert summary["vmec_config"]["mgrid_file"] == "DIRECT_COILS"
    assert summary["vmec_config"]["uses_generated_mgrid"] is False
    assert summary["vmec_config"]["python_provider_required"] is True
    assert summary["vmec_config"]["uses_mgrid_file"] is False
    assert summary["vmec_config"]["mgrid_compatibility_example"].endswith("free_boundary_essos_mgrid_forward.py")
    assert "generated_mgrid" not in summary["vmec_config"]
    assert [record["kind"] for record in summary["optimized_variables"]] == ["current", "fourier_dof"]
    assert all(record["kind"] != "boundary" for record in summary["optimized_variables"])


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Full coil -> direct-coil free-boundary solve -> Boozer/QS exact "
        "gradient validation is phase 2; current tests validate provider, "
        "projection, dense-vacuum, and dense mode-space adjoint pieces."
    ),
)
def test_full_free_boundary_qs_exact_gradient_validation_phase2_marker():
    raise NotImplementedError("production NESTOR/QS exact-gradient validation is not promoted yet")


def test_deterministic_circle_smoke_records_qs_terms(tmp_path, monkeypatch):
    module = _load_example_module()
    calls = []

    def fake_make_free_boundary_indata(_input_path, output_path, **_kwargs):
        output_path.write_text("&INDATA\n/\n")
        return output_path

    def fake_run_direct_free_boundary(input_path, params, *, vmec_max_iter, activate_fsq, jit_forces=True):
        calls.append(
            {
                "input_path": input_path,
                "current": float(np.asarray(params.base_currents)[0]),
                "vmec_max_iter": vmec_max_iter,
                "activate_fsq": activate_fsq,
                "jit_forces": bool(jit_forces),
            }
        )
        return SimpleNamespace(), 0.01

    def fake_summarize_run(
        _run,
        params,
        *,
        objective,
        wall_s,
        target_aspect,
        target_iota,
        helicity_m,
        helicity_n,
        qs_surfaces,
        qs_ntheta,
        qs_nphi,
    ):
        current = float(np.asarray(params.base_currents)[0])
        return {
            "objective": objective,
            "wall_s": wall_s,
            "vmec_n_iter": 1,
            "fsqr": current,
            "fsqz": 0.0,
            "fsql": 0.0,
            "residual_proxy": current,
            "qs_total": 0.25,
            "qs_helicity_m": helicity_m,
            "qs_helicity_n": helicity_n,
            "qs_surfaces": qs_surfaces,
            "qs_ntheta": qs_ntheta,
            "qs_nphi": qs_nphi,
            "aspect": target_aspect,
            "target_aspect": target_aspect,
            "mean_iota": target_iota,
            "target_iota": target_iota,
            "coil_current_norm": abs(current),
            "mean_coil_length": 1.0,
            "vmec_history": {"w": [], "fsqr2": [], "fsqz2": [], "fsql2": []},
        }

    def fake_write_wout(path, _run, *, include_fsq):
        path.write_text(f"include_fsq={include_fsq}\n")

    monkeypatch.setattr(module, "make_free_boundary_indata", fake_make_free_boundary_indata)
    monkeypatch.setattr(module, "run_direct_free_boundary", fake_run_direct_free_boundary)
    monkeypatch.setattr(module, "summarize_run", fake_summarize_run)
    monkeypatch.setattr(module, "write_wout_from_fixed_boundary_run", fake_write_wout)

    exit_code = module.main(
        [
            "--smoke",
            "--provider",
            "circle",
            "--max-evals",
            "1",
            "--max-iter",
            "1",
            "--qs-weight",
            "4.0",
            "--helicity-n",
            "-1",
            "--outdir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert len(calls) == 1
    assert all(call["jit_forces"] is True for call in calls)
    assert calls[0]["current"] == pytest.approx(2.0)

    history = json.loads((tmp_path / "history.json").read_text())
    summary = json.loads((tmp_path / "summary.json").read_text())

    assert len(history) == 1
    assert history[0]["variables"][0]["parameterization"] == "multiplicative"
    assert history[0]["coil_diagnostics"]["n_base_coils"] == 1
    assert history[0]["summary"]["objective_terms"]["residual"]["contribution"] == pytest.approx(2.0)
    assert history[0]["summary"]["objective_terms"]["quasisymmetry"]["contribution"] == pytest.approx(1.0)
    assert history[0]["summary"]["objective_terms"]["total"] == pytest.approx(3.0)
    assert summary["dry_run"] is False
    assert summary["flow"] == "single_stage_direct_coil_no_mgrid"
    assert summary["workflow"]["optimized_dofs"] == "coil currents and selected coil Fourier coefficients only"
    assert summary["scope"] == "deterministic coil-only direct-coil free-boundary QS optimization example"
    assert summary["baseline_coils"]["n_base_coils"] == 1
    assert summary["optimized_variables"][0]["unit_x_delta"] == pytest.approx(0.04)
    assert summary["objective_model"]["qs_weight"] == pytest.approx(4.0)
    assert summary["objective_model"]["helicity_n"] == -1
    assert (tmp_path / "wout_best_direct_coil_qs.nc").read_text() == "include_fsq=True\n"


def test_derivative_proposal_summary_marks_report_stale_when_trial_is_accepted(tmp_path, monkeypatch):
    module = _load_example_module()
    calls = []

    def fake_make_free_boundary_indata(_input_path, output_path, **_kwargs):
        output_path.write_text("&INDATA\n/\n")
        return output_path

    def fake_run_direct_free_boundary(_input_path, params, *, vmec_max_iter, activate_fsq, jit_forces=True):
        calls.append(
            {
                "current": float(np.asarray(params.base_currents)[0]),
                "vmec_max_iter": int(vmec_max_iter),
                "activate_fsq": float(activate_fsq),
                "jit_forces": bool(jit_forces),
            }
        )
        return SimpleNamespace(), 0.01

    def fake_summarize_run(
        _run,
        params,
        *,
        objective,
        wall_s,
        target_aspect,
        target_iota,
        helicity_m,
        helicity_n,
        qs_surfaces,
        qs_ntheta,
        qs_nphi,
    ):
        current = float(np.asarray(params.base_currents)[0])
        return {
            "objective": objective,
            "wall_s": wall_s,
            "vmec_n_iter": 1,
            "fsqr": current,
            "fsqz": 0.0,
            "fsql": 0.0,
            "residual_proxy": current,
            "qs_total": 0.25,
            "qs_helicity_m": helicity_m,
            "qs_helicity_n": helicity_n,
            "qs_surfaces": qs_surfaces,
            "qs_ntheta": qs_ntheta,
            "qs_nphi": qs_nphi,
            "aspect": target_aspect,
            "target_aspect": target_aspect,
            "mean_iota": target_iota,
            "target_iota": target_iota,
            "coil_current_norm": abs(current),
            "mean_coil_length": 1.0,
            "vmec_history": {"w": [], "fsqr2": [], "fsqz2": [], "fsql2": []},
        }

    def fake_write_wout(path, _run, *, include_fsq):
        path.write_text(f"include_fsq={include_fsq}\n")

    def fake_write_same_branch_validation_report(**kwargs):
        path = Path(kwargs["outdir"]) / "same_branch_complete_solve_report.json"
        path.write_text(
            json.dumps(
                {
                    "direction_x": [1.0, 0.0],
                    "branch_compatibility": {"same_branch": True},
                    "branch_local_vector_jacobian": {
                        "available": True,
                        "uses_production_forward": True,
                        "differentiates_adaptive_controller": False,
                        "differentiates_run_free_boundary": False,
                        "differentiates_fixed_accepted_branch": True,
                        "replay_ad_mode": "direct",
                        "derivative_mode": "directional_jvp",
                        "max_base_abs_delta": 0.0,
                        "scalars": {
                            "qs_total": {"value": 0.25, "exact_directional": 1.0, "base_abs_delta": 0.0},
                            "aspect": {"value": 6.0, "exact_directional": 0.0, "base_abs_delta": 0.0},
                        },
                    },
                }
            )
            + "\n"
        )
        return path

    monkeypatch.setattr(module, "make_free_boundary_indata", fake_make_free_boundary_indata)
    monkeypatch.setattr(module, "run_direct_free_boundary", fake_run_direct_free_boundary)
    monkeypatch.setattr(module, "summarize_run", fake_summarize_run)
    monkeypatch.setattr(module, "write_wout_from_fixed_boundary_run", fake_write_wout)
    monkeypatch.setattr(module, "write_same_branch_validation_report", fake_write_same_branch_validation_report)

    exit_code = module.main(
        [
            "--smoke",
            "--provider",
            "circle",
            "--max-evals",
            "1",
            "--max-iter",
            "1",
            "--qs-weight",
            "4.0",
            "--write-same-branch-report",
            "--same-branch-derivative-proposal",
            "--same-branch-proposal-step",
            "1.0",
            "--outdir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert len(calls) == 2
    assert calls[0]["current"] == pytest.approx(2.0)
    assert calls[1]["current"] == pytest.approx(1.96)

    summary = json.loads((tmp_path / "summary.json").read_text())
    proposal = summary["same_branch_derivative_proposal"]
    assert proposal["available"] is True
    assert proposal["accepted_by_complete_solve"] is True
    assert proposal["rejected_by_complete_solve"] is False
    assert proposal["acceptance_decision_source"] == "complete_solve_objective"
    assert proposal["best_eval_before_trial"] == 0
    assert proposal["best_eval_after_trial"] == 1
    assert proposal["trial_objective"] < proposal["previous_best_objective"]
    assert summary["best"]["eval"] == 1
    report_status = summary["same_branch_complete_solve_report_final_best_status"]
    assert report_status["report_generated_before_derivative_proposal"] is True
    assert report_status["final_best_changed_after_report"] is True
    assert report_status["report_matches_final_best"] is False


def test_derivative_proposal_summary_records_rejected_trial_as_complete_solve_rejection(tmp_path, monkeypatch):
    module = _load_example_module()
    calls = []

    def fake_make_free_boundary_indata(_input_path, output_path, **_kwargs):
        output_path.write_text("&INDATA\n/\n")
        return output_path

    def fake_run_direct_free_boundary(_input_path, params, *, vmec_max_iter, activate_fsq, jit_forces=True):
        calls.append(
            {
                "current": float(np.asarray(params.base_currents)[0]),
                "vmec_max_iter": int(vmec_max_iter),
                "activate_fsq": float(activate_fsq),
                "jit_forces": bool(jit_forces),
            }
        )
        return SimpleNamespace(), 0.01

    def fake_summarize_run(
        _run,
        params,
        *,
        objective,
        wall_s,
        target_aspect,
        target_iota,
        helicity_m,
        helicity_n,
        qs_surfaces,
        qs_ntheta,
        qs_nphi,
    ):
        current = float(np.asarray(params.base_currents)[0])
        return {
            "objective": objective,
            "wall_s": wall_s,
            "vmec_n_iter": 1,
            "fsqr": current,
            "fsqz": 0.0,
            "fsql": 0.0,
            "residual_proxy": 4.0 - current,
            "qs_total": 0.25,
            "qs_helicity_m": helicity_m,
            "qs_helicity_n": helicity_n,
            "qs_surfaces": qs_surfaces,
            "qs_ntheta": qs_ntheta,
            "qs_nphi": qs_nphi,
            "aspect": target_aspect,
            "target_aspect": target_aspect,
            "mean_iota": target_iota,
            "target_iota": target_iota,
            "coil_current_norm": abs(current),
            "mean_coil_length": 1.0,
            "vmec_history": {"w": [], "fsqr2": [], "fsqz2": [], "fsql2": []},
        }

    def fake_write_wout(path, _run, *, include_fsq):
        path.write_text(f"include_fsq={include_fsq}\n")

    def fake_write_same_branch_validation_report(**kwargs):
        path = Path(kwargs["outdir"]) / "same_branch_complete_solve_report.json"
        path.write_text(
            json.dumps(
                {
                    "direction_x": [1.0, 0.0],
                    "branch_compatibility": {"same_branch": True},
                    "branch_local_vector_jacobian": {
                        "available": True,
                        "uses_production_forward": True,
                        "differentiates_adaptive_controller": False,
                        "differentiates_run_free_boundary": False,
                        "differentiates_fixed_accepted_branch": True,
                        "replay_ad_mode": "direct",
                        "derivative_mode": "directional_jvp",
                        "max_base_abs_delta": 0.0,
                        "scalars": {
                            "qs_total": {"value": 0.25, "exact_directional": 1.0, "base_abs_delta": 0.0},
                            "aspect": {"value": 6.0, "exact_directional": 0.0, "base_abs_delta": 0.0},
                        },
                    },
                }
            )
            + "\n"
        )
        return path

    monkeypatch.setattr(module, "make_free_boundary_indata", fake_make_free_boundary_indata)
    monkeypatch.setattr(module, "run_direct_free_boundary", fake_run_direct_free_boundary)
    monkeypatch.setattr(module, "summarize_run", fake_summarize_run)
    monkeypatch.setattr(module, "write_wout_from_fixed_boundary_run", fake_write_wout)
    monkeypatch.setattr(module, "write_same_branch_validation_report", fake_write_same_branch_validation_report)

    exit_code = module.main(
        [
            "--smoke",
            "--provider",
            "circle",
            "--max-evals",
            "1",
            "--max-iter",
            "1",
            "--qs-weight",
            "4.0",
            "--write-same-branch-report",
            "--same-branch-derivative-proposal",
            "--same-branch-proposal-step",
            "1.0",
            "--outdir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert len(calls) == 2
    assert calls[0]["current"] == pytest.approx(2.0)
    assert calls[1]["current"] == pytest.approx(1.96)

    summary = json.loads((tmp_path / "summary.json").read_text())
    proposal = summary["same_branch_derivative_proposal"]
    assert proposal["available"] is True
    assert proposal["accepted_by_complete_solve"] is False
    assert proposal["rejected_by_complete_solve"] is True
    assert proposal["acceptance_decision_source"] == "complete_solve_objective"
    assert proposal["trial_objective"] > proposal["previous_best_objective"]
    assert proposal["best_eval_before_trial"] == 0
    assert proposal["best_eval_after_trial"] == 0
    assert summary["best"]["eval"] == 0
    report_status = summary["same_branch_complete_solve_report_final_best_status"]
    assert report_status["report_generated_before_derivative_proposal"] is True
    assert report_status["final_best_changed_after_report"] is False
    assert report_status["report_matches_final_best"] is True


def test_essos_provider_non_dry_run_uses_direct_coils_without_mgrid(tmp_path, monkeypatch):
    module = _load_example_module()
    synthetic_params, _metadata = module.make_circle_provider(current_scale=1.0)
    calls = []

    def fake_load_essos_provider(coils_json, *, chunk_size, current_scale):
        assert coils_json is None
        assert chunk_size == 128
        assert current_scale == pytest.approx(1.0)
        return synthetic_params, {
            "provider": "essos",
            "coils_json": "/synthetic/ESSOS_biot_savart_LandremanPaulQA.json",
            "n_base_coils": 1,
            "n_segments": int(synthetic_params.n_segments),
            "nfp": int(synthetic_params.nfp),
            "stellsym": bool(synthetic_params.stellsym),
            "current_scale_multiplier": 1.0,
        }

    def fake_make_free_boundary_indata(_input_path, output_path, **_kwargs):
        output_path.write_text("&INDATA\n  LFREEB = T\n  MGRID_FILE = 'DIRECT_COILS'\n/\n")
        return output_path

    def fake_run_direct_free_boundary(input_path, params, *, vmec_max_iter, activate_fsq, jit_forces=True):
        calls.append(
            {
                "input_path": input_path,
                "params": params,
                "vmec_max_iter": int(vmec_max_iter),
                "activate_fsq": float(activate_fsq),
                "jit_forces": bool(jit_forces),
            }
        )
        return SimpleNamespace(), 0.02

    def fake_summarize_run(
        _run,
        params,
        *,
        objective,
        wall_s,
        target_aspect,
        target_iota,
        helicity_m,
        helicity_n,
        qs_surfaces,
        qs_ntheta,
        qs_nphi,
    ):
        current = float(np.asarray(params.base_currents)[0])
        return {
            "objective": objective,
            "wall_s": wall_s,
            "vmec_n_iter": 1,
            "fsqr": current,
            "fsqz": 0.0,
            "fsql": 0.0,
            "residual_proxy": current,
            "qs_total": 0.125,
            "qs_helicity_m": helicity_m,
            "qs_helicity_n": helicity_n,
            "qs_surfaces": qs_surfaces,
            "qs_ntheta": qs_ntheta,
            "qs_nphi": qs_nphi,
            "aspect": target_aspect,
            "target_aspect": target_aspect,
            "mean_iota": target_iota,
            "target_iota": target_iota,
            "coil_current_norm": abs(current),
            "mean_coil_length": 1.0,
            "vmec_history": {"w": [], "fsqr2": [], "fsqz2": [], "fsql2": []},
        }

    def fake_write_wout(path, _run, *, include_fsq):
        path.write_text(f"include_fsq={include_fsq}\n")

    monkeypatch.setattr(module, "load_essos_provider", fake_load_essos_provider)
    monkeypatch.setattr(module, "make_free_boundary_indata", fake_make_free_boundary_indata)
    monkeypatch.setattr(module, "run_direct_free_boundary", fake_run_direct_free_boundary)
    monkeypatch.setattr(module, "summarize_run", fake_summarize_run)
    monkeypatch.setattr(module, "write_wout_from_fixed_boundary_run", fake_write_wout)

    exit_code = module.main(
        [
            "--smoke",
            "--provider",
            "essos",
            "--chunk-size",
            "128",
            "--max-evals",
            "1",
            "--max-iter",
            "1",
            "--outdir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert len(calls) == 1
    assert calls[0]["input_path"] == tmp_path / "input.direct_coil_qs"
    assert calls[0]["params"].n_segments == synthetic_params.n_segments
    assert (tmp_path / "input.direct_coil_qs").read_text().count("DIRECT_COILS") == 1
    assert not any(tmp_path.glob("mgrid*"))

    history = json.loads((tmp_path / "history.json").read_text())
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert len(history) == 1
    assert summary["dry_run"] is False
    assert summary["provider"]["provider"] == "essos"
    assert summary["flow"] == "single_stage_direct_coil_no_mgrid"
    assert summary["workflow"]["uses_mgrid_file"] is False
    assert summary["vmec_config"]["external_field_provider_kind"] == "direct_coils"
    assert summary["vmec_config"]["mgrid_file"] == "DIRECT_COILS"
    assert summary["vmec_config"]["uses_generated_mgrid"] is False
    assert summary["vmec_config"]["python_provider_required"] is True
    assert "generated_mgrid" not in summary["vmec_config"]
    assert all(record["kind"] in {"current", "fourier_dof"} for record in summary["optimized_variables"])
    assert all(record["kind"] != "boundary" for record in summary["optimized_variables"])
    assert (tmp_path / "wout_best_direct_coil_qs.nc").read_text() == "include_fsq=True\n"
