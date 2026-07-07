from __future__ import annotations

import numpy as np
import pytest

from conftest import circular_coil_params
from vmec_jax._compat import jnp
from vmec_jax.external_fields import CoilFieldParams
from vmec_jax.solvers.free_boundary import derivatives


def _params() -> CoilFieldParams:
    return circular_coil_params(current=2.0, radius=1.2, n_segments=8)


def _fake_registry(*, args, qs_surfaces, qs_angle_cache_for_static):
    value_fns = {
        "aspect": lambda payload: 5.0,
        "mean_iota": lambda payload: 0.41,
        "qs_total": lambda payload: 1.0e-3,
        "lcfs_boundary_moment": lambda payload: 2.0e-2,
        "accepted_bnormal_rms": lambda payload: 3.0e-2,
        "state_norm": lambda payload: 4.0,
        "betatotal": lambda payload: 0.0,
        "boozer_qs_total": lambda payload: 1.5e-3,
    }
    replay_fns = {key: (lambda replay, payload, key=key: value_fns[key](payload)) for key in value_fns}
    return value_fns, replay_fns


def test_canonical_free_boundary_output_keys_support_user_facing_aliases():
    keys, aliases = derivatives.canonical_free_boundary_output_keys(
        ("aspect", "iota", "boundary_displacement", "bnormal_rms", "qs_residual")
    )

    assert keys == (
        "aspect",
        "mean_iota",
        "lcfs_boundary_moment",
        "accepted_bnormal_rms",
        "qs_total",
    )
    assert aliases["iota"] == "mean_iota"
    assert aliases["boundary_displacement"] == "lcfs_boundary_moment"
    assert aliases["bnormal_rms"] == "accepted_bnormal_rms"
    assert aliases["qs_residual"] == "qs_total"


def test_free_boundary_value_and_jvp_projects_public_names(monkeypatch):
    base = _params()
    direction = derivatives.coil_direction(base, current=0.1, curve_dof=1.0e-3, curve_index=(0, 0, 2))
    calls: dict[str, object] = {}

    def fake_branch_local(**kwargs):
        calls["scalar_keys"] = tuple(kwargs["scalar_keys"])
        return {
            "values": {
                "aspect": 5.0,
                "mean_iota": 0.41,
                "lcfs_boundary_moment": 2.0e-2,
                "accepted_bnormal_rms": 3.0e-2,
                "qs_total": 1.0e-3,
            },
            "replay_value_map": {
                "aspect": jnp.asarray(5.0),
                "mean_iota": jnp.asarray(0.41),
                "lcfs_boundary_moment": jnp.asarray(2.0e-2),
                "accepted_bnormal_rms": jnp.asarray(3.0e-2),
                "qs_total": jnp.asarray(1.0e-3),
            },
            "base_abs_delta": {key: 0.0 for key in kwargs["scalar_keys"]},
            "base_rel_delta": {key: 0.0 for key in kwargs["scalar_keys"]},
            "directional_derivatives": {
                "aspect": jnp.asarray(0.2),
                "mean_iota": jnp.asarray(-0.3),
                "lcfs_boundary_moment": jnp.asarray(0.4),
                "accepted_bnormal_rms": jnp.asarray(0.5),
                "qs_total": jnp.asarray(-0.6),
            },
            "jacobian": None,
            "grads": {},
            "derivative_mode": "directional_jvp",
        }

    monkeypatch.setattr(derivatives, "same_branch_scalar_function_registry", _fake_registry)
    monkeypatch.setattr(
        derivatives._branch_local_derivatives,
        "direct_coil_run_free_boundary_branch_local_scalars_value_and_jacobian_jax",
        fake_branch_local,
    )

    report = derivatives.free_boundary_value_and_jvp(
        "input.test",
        base,
        direction_params=direction,
        outputs=("aspect", "iota", "boundary_displacement", "bnormal_rms", "qs_residual"),
    )

    assert calls["scalar_keys"] == (
        "aspect",
        "mean_iota",
        "lcfs_boundary_moment",
        "accepted_bnormal_rms",
        "qs_total",
    )
    assert report["values"]["iota"] == 0.41
    assert float(np.asarray(report["directional_derivatives"]["qs_residual"])) == -0.6
    assert report["differentiates_adaptive_controller"] is False
    assert report["differentiates_fixed_accepted_branch"] is True


def test_free_boundary_value_and_jvp_can_attach_complete_solve_fd_validation(monkeypatch):
    base = _params()
    direction = derivatives.coil_direction(base, current=0.1)

    def fake_branch_local(**kwargs):
        return {
            "values": {"aspect": 5.0, "qs_total": 1.0e-3},
            "replay_value_map": {"aspect": jnp.asarray(5.0), "qs_total": jnp.asarray(1.0e-3)},
            "base_abs_delta": {"aspect": 0.0, "qs_total": 0.0},
            "base_rel_delta": {"aspect": 0.0, "qs_total": 0.0},
            "directional_derivatives": {"aspect": jnp.asarray(0.2), "qs_total": jnp.asarray(-0.6)},
            "jacobian": None,
            "grads": {},
            "derivative_mode": "directional_jvp",
        }

    def fake_fd_report(*args, **kwargs):
        assert kwargs["eps"] == 1.0e-5
        return {
            "objective_values": {
                "aspect": {"central_fd_directional": 0.2},
                "qs_total": {"central_fd_directional": -0.6},
            },
            "branch_compatibility": {"same_branch": True},
        }

    def fake_scalar_report(complete_report, branch_local, *, scalar_keys, **kwargs):
        return {
            "passed": True,
            "scalar_reports": {
                key: {
                    "passed": True,
                    "exact_directional": float(branch_local["directional_derivatives"][key]),
                    "complete_fd_directional": complete_report["objective_values"][key][
                        "central_fd_directional"
                    ],
                    "abs_error": 0.0,
                    "rel_error": 0.0,
                }
                for key in scalar_keys
            },
        }

    monkeypatch.setattr(derivatives, "same_branch_scalar_function_registry", _fake_registry)
    monkeypatch.setattr(
        derivatives._branch_local_derivatives,
        "direct_coil_run_free_boundary_branch_local_scalars_value_and_jacobian_jax",
        fake_branch_local,
    )
    monkeypatch.setattr(derivatives._branch_local_derivatives, "direct_coil_same_branch_complete_solve_fd_report", fake_fd_report)
    monkeypatch.setattr(
        derivatives._branch_local_derivatives,
        "direct_coil_branch_local_scalars_report_from_complete_fd",
        fake_scalar_report,
    )

    report = derivatives.free_boundary_value_and_jvp(
        "input.test",
        base,
        direction_params=direction,
        outputs=("aspect", "qs_residual"),
        cotangent={"aspect": 2.0, "qs_residual": -1.0},
        validate_fd=True,
        fd_epsilon=1.0e-5,
    )

    assert report["fd_validation"]["scalar_report"]["passed"] is True
    assert report["fd_validation"]["public_scalar_report"]["qs_residual"]["abs_error"] == 0.0
    assert report["validation_summary"] == {
        "available": True,
        "same_branch": True,
        "scalar_passed": True,
        "cotangent_passed": True,
        "all_passed": True,
        "n_scalars": 2,
        "n_scalar_passes": 2,
        "max_abs_error": 0.0,
        "max_rel_error": 0.0,
    }
    cotangent_check = report["cotangent_vjp_fd_check"]
    assert cotangent_check["passed"] is True
    assert cotangent_check["ad_cotangent_directional"] == 2.0 * 0.2 + (-1.0) * (-0.6)
    assert cotangent_check["fd_cotangent_directional"] == 2.0 * 0.2 + (-1.0) * (-0.6)


def test_free_boundary_value_and_jvp_rejects_unrequested_cotangent_key(monkeypatch):
    base = _params()
    direction = derivatives.coil_direction(base, current=0.1)

    monkeypatch.setattr(derivatives, "same_branch_scalar_function_registry", _fake_registry)

    with pytest.raises(ValueError, match="was not requested"):
        derivatives.free_boundary_value_and_jvp(
            "input.test",
            base,
            direction_params=direction,
            outputs=("aspect",),
            cotangent={"qs_residual": 1.0},
        )


def test_contract_free_boundary_vjp_contracts_row_stacked_pytree_jacobian():
    jacobian = {
        "base_currents": jnp.asarray([[1.0, 2.0], [3.0, 4.0]]),
        "base_curve_dofs": jnp.asarray([[[1.0]], [[-1.0]]]),
    }

    cotangent = derivatives.contract_free_boundary_vjp(jacobian, [2.0, -1.0])

    np.testing.assert_allclose(np.asarray(cotangent["base_currents"]), np.asarray([-1.0, 0.0]))
    np.testing.assert_allclose(np.asarray(cotangent["base_curve_dofs"]), np.asarray([[3.0]]))
