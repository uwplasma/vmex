from __future__ import annotations

import numpy as np
import pytest

from vmec_jax._compat import enable_x64
from vmec_jax.external_fields import CoilFieldParams, sample_coil_field_cylindrical
from vmec_jax.free_boundary import (
    _build_vmec_mode_basis,
    _ensure_vmec_nonsingular_kernel_tables,
    _vmec_analytic_terms_from_geometry,
    _vmec_bvec_from_gsource,
    _vmec_mode_matrix_from_grpmn,
    _vmec_nonsingular_terms_from_bexni,
    _vmec_source_from_gsource,
    ExternalBoundarySample,
    VacuumBoundaryFields,
    vacuum_boundary_fields_from_cylindrical,
)
from vmec_jax.free_boundary_adjoint import (
    dense_fixed_point_solve_jax,
    dense_mode_vacuum_solve_jax,
    dense_nonlinear_solve_jax,
    dense_vmec_nestor_mode_solve_jax,
    dense_vacuum_residual,
    dense_vacuum_solve_jax,
    jax_visible_accepted_only_nonlinear_controller_jax,
    jax_visible_accepted_nonlinear_controller_directional_check_jax,
    jax_visible_accepted_nonlinear_controller_jax,
    jax_visible_masked_nonlinear_controller_directional_check_jax,
    jax_visible_masked_nonlinear_controller_jax,
    jax_visible_nonlinear_controller_directional_check_jax,
    jax_visible_nonlinear_controller_jax,
    jax_visible_segmented_accepted_nonlinear_controller_jax,
    jax_visible_segmented_state_only_accepted_nonlinear_controller_jax,
    jax_visible_state_only_accepted_nonlinear_controller_jax,
    jax_visible_state_only_accepted_only_nonlinear_controller_jax,
    jax_visible_unrolled_accepted_only_nonlinear_controller_jax,
    jax_visible_unrolled_state_only_accepted_only_nonlinear_controller_jax,
    direct_coil_projected_mode_fixed_point_directional_check_jax,
    direct_coil_projected_mode_fixed_point_objective_jax,
    mode_matrix_matvec_from_grpmn_jax,
    mode_matrix_from_grpmn_jax,
    mode_operator_vacuum_solve_jax,
    mode_rhs_from_gsource_jax,
    pytree_directional_derivative_check_jax,
    vacuum_boundary_fields_from_cylindrical_jax,
    vmec_analytic_terms_from_geometry_jax,
    vmec_nonsingular_terms_from_bexni_jax,
    vmec_source_from_gsource_jax,
)


def _well_conditioned_matrix():
    from vmec_jax._compat import jnp

    A = jnp.asarray(
        [
            [3.0, 0.2, -0.1],
            [0.4, 2.5, 0.3],
            [-0.2, 0.1, 2.2],
        ]
    )
    b = jnp.asarray([1.0, -0.4, 0.7])
    return A, b


def test_dense_vacuum_solve_matches_jnp_linalg_solve():
    from vmec_jax._compat import jnp

    enable_x64(True)
    A, b = _well_conditioned_matrix()

    actual = dense_vacuum_solve_jax(A, b)
    expected = jnp.linalg.solve(A, b)

    np.testing.assert_allclose(actual, expected, rtol=1.0e-14, atol=1.0e-14)
    np.testing.assert_allclose(dense_vacuum_residual(A, actual, b), np.zeros_like(np.asarray(b)), atol=1.0e-14)


def test_dense_vacuum_vjp_wrt_b_matches_transpose_solve():
    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp

    enable_x64(True)
    A, b = _well_conditioned_matrix()
    cotangent = jnp.asarray([0.3, -0.2, 0.5])

    def objective(rhs):
        x = dense_vacuum_solve_jax(A, rhs)
        return jnp.vdot(cotangent, x)

    grad_b = jax.grad(objective)(b)
    expected = jnp.linalg.solve(A.T, cotangent)

    np.testing.assert_allclose(grad_b, expected, rtol=1.0e-13, atol=1.0e-13)


def test_dense_vacuum_gradient_wrt_rhs_parameter_matches_finite_difference():
    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp

    enable_x64(True)
    A, b = _well_conditioned_matrix()
    direction = jnp.asarray([0.2, -0.1, 0.4])
    cotangent = jnp.asarray([0.3, -0.2, 0.5])

    def objective(scale):
        x = dense_vacuum_solve_jax(A, b + scale * direction)
        return jnp.vdot(cotangent, x)

    exact = jax.grad(objective)(0.0)
    eps = 1.0e-6
    fd = (objective(eps) - objective(-eps)) / (2.0 * eps)

    np.testing.assert_allclose(exact, fd, rtol=2.0e-9, atol=1.0e-11)


def test_dense_vacuum_gradient_wrt_matrix_parameter_matches_finite_difference():
    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp

    enable_x64(True)
    A, b = _well_conditioned_matrix()
    dA = jnp.asarray(
        [
            [0.0, 0.2, 0.0],
            [-0.1, 0.0, 0.3],
            [0.0, 0.1, 0.0],
        ]
    )
    cotangent = jnp.asarray([0.3, -0.2, 0.5])

    def objective(scale):
        x = dense_vacuum_solve_jax(A + scale * dA, b)
        return jnp.vdot(cotangent, x)

    exact = jax.grad(objective)(0.0)
    eps = 1.0e-6
    fd = (objective(eps) - objective(-eps)) / (2.0 * eps)

    np.testing.assert_allclose(exact, fd, rtol=2.0e-9, atol=1.0e-11)


def test_dense_vacuum_symmetric_mode_uses_symmetric_transpose_solve():
    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp

    enable_x64(True)
    A = jnp.asarray([[3.0, 0.2], [0.2, 2.0]])
    b = jnp.asarray([0.7, -0.1])
    cotangent = jnp.asarray([0.4, 0.5])

    def objective(rhs):
        return jnp.vdot(cotangent, dense_vacuum_solve_jax(A, rhs, symmetric=True))

    grad_b = jax.grad(objective)(b)
    expected = jnp.linalg.solve(A, cotangent)

    np.testing.assert_allclose(grad_b, expected, rtol=1.0e-13, atol=1.0e-13)


def _nonlinear_residual(x, params):
    from vmec_jax._compat import jnp

    rhs = params["rhs"]
    return jnp.asarray(
        [
            x[0] + 0.2 * x[0] ** 3 + 0.1 * x[1] - rhs[0],
            x[1] + 0.1 * x[0] ** 2 + 0.3 * jnp.sin(x[1]) - rhs[1],
        ]
    )


def test_dense_nonlinear_solve_drives_residual_to_zero():
    pytest.importorskip("jax")
    from vmec_jax._compat import jnp

    enable_x64(True)
    params = {"rhs": jnp.asarray([0.4, -0.2], dtype=float)}
    root = dense_nonlinear_solve_jax(
        _nonlinear_residual,
        jnp.asarray([0.0, 0.0], dtype=float),
        params,
        max_iter=12,
    )

    residual = _nonlinear_residual(root, params)
    np.testing.assert_allclose(residual, np.zeros(2), rtol=0.0, atol=1.0e-12)


def test_dense_nonlinear_implicit_adjoint_matches_finite_difference_for_rhs():
    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp

    enable_x64(True)
    rhs0 = jnp.asarray([0.4, -0.2], dtype=float)
    direction = jnp.asarray([0.3, -0.5], dtype=float)
    weights = jnp.asarray([1.2, 0.7], dtype=float)

    def objective(scale):
        root = dense_nonlinear_solve_jax(
            _nonlinear_residual,
            jnp.asarray([0.0, 0.0], dtype=float),
            {"rhs": rhs0 + scale * direction},
            max_iter=12,
        )
        return 0.5 * jnp.vdot(weights * root, root)

    exact = jax.grad(objective)(0.0)
    eps = 1.0e-6
    fd = (objective(eps) - objective(-eps)) / (2.0 * eps)

    np.testing.assert_allclose(exact, fd, rtol=3.0e-8, atol=1.0e-10)


def test_dense_fixed_point_implicit_adjoint_matches_finite_difference_for_rhs():
    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp

    enable_x64(True)
    rhs0 = jnp.asarray([0.25, -0.18], dtype=float)
    direction = jnp.asarray([0.2, -0.4], dtype=float)
    weights = jnp.asarray([0.8, 1.1], dtype=float)

    def update(state, params):
        rhs = params["rhs"]
        return jnp.asarray(
            [
                0.12 + 0.18 * jnp.tanh(state[0]) + 0.07 * state[1] + rhs[0],
                -0.09 + 0.05 * state[0] ** 2 + 0.16 * jnp.sin(state[1]) + rhs[1],
            ]
        )

    def objective(scale):
        root = dense_fixed_point_solve_jax(
            update,
            jnp.asarray([0.0, 0.0], dtype=float),
            {"rhs": rhs0 + scale * direction},
            max_iter=12,
        )
        return 0.5 * jnp.vdot(weights * root, root)

    exact = jax.grad(objective)(0.0)
    eps = 1.0e-6
    fd = (objective(eps) - objective(-eps)) / (2.0 * eps)

    np.testing.assert_allclose(exact, fd, rtol=5.0e-8, atol=1.0e-10)


def test_pytree_directional_derivative_check_can_skip_finite_difference():
    pytest.importorskip("jax")
    from vmec_jax._compat import jnp

    params = {"x": jnp.asarray([1.5, -0.4]), "y": jnp.asarray(0.25)}
    direction = {"x": jnp.asarray([0.2, -0.3]), "y": jnp.asarray(-0.1)}

    def objective(values):
        return jnp.sum(values["x"] ** 2) + 0.5 * values["y"] ** 2

    check = pytree_directional_derivative_check_jax(
        objective,
        params,
        direction,
        compute_fd=False,
    )

    expected = 2.0 * jnp.vdot(params["x"], direction["x"]) + params["y"] * direction["y"]
    np.testing.assert_allclose(np.asarray(check["exact_directional"]), np.asarray(expected), rtol=1.0e-14, atol=1.0e-14)
    assert bool(jnp.isnan(check["fd_directional"]))
    assert bool(jnp.isnan(check["abs_error"]))
    assert bool(jnp.isnan(check["rel_error"]))


def test_direct_coil_trace_directional_helpers_can_skip_finite_difference(monkeypatch):
    """Cover high-level replay helper contracts without an expensive VMEC trace."""

    pytest.importorskip("jax")
    from vmec_jax._compat import jnp
    import vmec_jax.free_boundary_adjoint as freeb_adjoint

    enable_x64(True)

    def fake_replay(coil_params, initial_state, **kwargs):
        del initial_state, kwargs
        x = jnp.asarray(coil_params)
        return {
            "objective": jnp.sum(x * x) + 0.25 * x[0],
            "objective_components": {"toy": jnp.sum(x * x)},
        }

    monkeypatch.setattr(
        freeb_adjoint,
        "direct_coil_accepted_trace_replay_objective_jax",
        fake_replay,
    )
    monkeypatch.setattr(
        freeb_adjoint,
        "direct_coil_accepted_trace_controller_replay_objective_jax",
        fake_replay,
    )

    params = jnp.asarray([1.0, -2.0, 0.5])
    direction = jnp.asarray([0.25, 0.5, -0.75])

    replay_check = freeb_adjoint.direct_coil_accepted_trace_directional_check_jax(
        params,
        direction,
        initial_state=None,
        eps=1.0e-5,
    )
    assert replay_check["objective_components"]["toy"] > 0.0
    np.testing.assert_allclose(
        np.asarray(replay_check["exact_directional"]),
        np.asarray(replay_check["fd_directional"]),
        rtol=1.0e-8,
        atol=1.0e-10,
    )

    controller_check = freeb_adjoint.direct_coil_accepted_trace_controller_directional_check_jax(
        params,
        direction,
        initial_state=None,
        eps=1.0e-5,
        compute_fd=False,
    )
    assert controller_check["objective_components"]["toy"] > 0.0
    assert np.isfinite(float(np.asarray(controller_check["exact_directional"])))
    assert np.isnan(float(np.asarray(controller_check["fd_directional"])))


def test_jax_visible_nonlinear_controller_matches_manual_scan_and_fd():
    pytest.importorskip("jax")
    from vmec_jax._compat import jnp

    enable_x64(True)
    controls = {
        "gain": jnp.asarray([0.21, 0.18, 0.16, 0.13], dtype=float),
        "bias": jnp.asarray(
            [
                [0.010, -0.020],
                [0.015, -0.012],
                [-0.005, 0.018],
                [0.008, 0.004],
            ],
            dtype=float,
        ),
    }
    params = {
        "matrix": jnp.asarray([[0.18, -0.05], [0.07, 0.14]], dtype=float),
        "drive": jnp.asarray([0.30, -0.22], dtype=float),
    }
    direction = {
        "matrix": jnp.asarray([[0.02, -0.01], [0.015, 0.005]], dtype=float),
        "drive": jnp.asarray([0.04, -0.025], dtype=float),
    }

    def step(state, prm, control):
        drive = prm["matrix"] @ state + prm["drive"] + control["bias"]
        next_state = 0.72 * state + control["gain"] * jnp.tanh(drive)
        return next_state, {"drive": drive, "norm": jnp.vdot(next_state, next_state)}

    run = jax_visible_nonlinear_controller_jax(
        step,
        jnp.asarray([0.05, -0.03], dtype=float),
        params,
        controls,
    )
    state = jnp.asarray([0.05, -0.03], dtype=float)
    norms = []
    for k in range(4):
        state, aux = step(
            state,
            params,
            {"gain": controls["gain"][k], "bias": controls["bias"][k]},
        )
        norms.append(aux["norm"])
    np.testing.assert_allclose(run["state"], state, rtol=1.0e-14, atol=1.0e-14)
    np.testing.assert_allclose(run["history"]["norm"], jnp.asarray(norms), rtol=1.0e-14, atol=1.0e-14)

    def objective_from_run(controller_run):
        return 0.5 * jnp.vdot(controller_run["state"], controller_run["state"]) + 0.01 * jnp.sum(
            controller_run["history"]["norm"]
        )

    check = jax_visible_nonlinear_controller_directional_check_jax(
        step,
        objective_from_run,
        params,
        direction,
        jnp.asarray([0.05, -0.03], dtype=float),
        controls,
        eps=1.0e-5,
    )
    assert np.isfinite(float(check["value"]))
    assert float(check["abs_error"]) < 1.0e-9
    np.testing.assert_allclose(check["exact_directional"], check["fd_directional"], rtol=1.0e-6, atol=1.0e-10)


def test_jax_visible_masked_controller_keeps_final_state_and_gradient_stable():
    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp

    enable_x64(True)
    controls = {
        "gain": jnp.asarray([0.20, 0.30, 0.25, 0.80, 0.80], dtype=float),
        "stop": jnp.asarray([False, False, True, False, False]),
    }
    params = {"target": jnp.asarray([0.85, -0.35], dtype=float)}
    direction = {"target": jnp.asarray([0.03, -0.02], dtype=float)}
    initial_state = jnp.asarray([0.0, 0.0], dtype=float)

    def step(state, prm, control):
        delta = prm["target"] - state
        next_state = state + control["gain"] * delta
        return next_state, {"err": jnp.linalg.norm(delta)}

    def converged_fn(_next_state, _prm, control, _aux):
        return control["stop"]

    run = jax_visible_masked_nonlinear_controller_jax(
        step,
        converged_fn,
        initial_state,
        params,
        controls,
    )

    manual = initial_state
    active = []
    done = False
    for k in range(int(controls["gain"].shape[0])):
        active.append(not done)
        proposed, _aux = step(
            manual,
            params,
            {"gain": controls["gain"][k], "stop": controls["stop"][k]},
        )
        if not done:
            manual = proposed
        done = done or bool(controls["stop"][k])

    np.testing.assert_allclose(run["state"], manual, rtol=1.0e-14, atol=1.0e-14)
    np.testing.assert_array_equal(np.asarray(run["history"]["active"]), np.asarray(active))
    assert bool(run["done"]) is True

    def objective(controller_params):
        out = jax_visible_masked_nonlinear_controller_jax(
            step,
            converged_fn,
            initial_state,
            controller_params,
            controls,
            checkpoint_steps=True,
        )
        return 0.5 * jnp.vdot(out["state"], out["state"]) + 0.01 * jnp.sum(
            jnp.asarray(out["history"]["active"], dtype=float) * out["history"]["err"]
        )

    exact = jnp.vdot(jax.grad(objective)(params)["target"], direction["target"])

    def shifted(scale):
        return {"target": params["target"] + scale * direction["target"]}

    eps = 1.0e-5
    fd = (objective(shifted(eps)) - objective(shifted(-eps))) / (2.0 * eps)
    np.testing.assert_allclose(exact, fd, rtol=2.0e-6, atol=1.0e-10)


def test_jax_visible_controller_plain_step_outputs_and_segment_validation():
    pytest.importorskip("jax")
    from vmec_jax._compat import jnp
    from vmec_jax.free_boundary_adjoint_controller import _pytree_vdot_jax

    enable_x64(True)

    def step_plain(state, params, control):
        return state + params["scale"] * control

    params = {"scale": jnp.asarray(0.5)}
    controls = jnp.asarray([1.0, 2.0, 3.0])

    run = jax_visible_nonlinear_controller_jax(step_plain, jnp.asarray(0.0), params, controls)
    np.testing.assert_allclose(np.asarray(run["state"]), 3.0)

    masked = jax_visible_masked_nonlinear_controller_jax(
        step_plain,
        lambda state, _params, _control, _aux: state > 1.0,
        jnp.asarray(0.0),
        params,
        controls,
    )
    np.testing.assert_allclose(np.asarray(masked["state"]), 1.5)
    np.testing.assert_array_equal(np.asarray(masked["history"]["active"]), [True, True, False])

    accepted = jax_visible_accepted_nonlinear_controller_jax(
        step_plain,
        lambda _state, proposed, _params, _control, _aux: proposed < 2.0,
        lambda state, _params, _control, _aux: state > 1.0,
        jnp.asarray(0.0),
        params,
        controls,
    )
    np.testing.assert_allclose(np.asarray(accepted["state"]), 1.5)
    np.testing.assert_array_equal(np.asarray(accepted["history"]["accepted"]), [True, True, False])

    state_only_accepted = jax_visible_state_only_accepted_nonlinear_controller_jax(
        step_plain,
        lambda _state, proposed, _params, _control, _aux: proposed < 2.0,
        lambda state, _params, _control, _aux: state > 1.0,
        jnp.asarray(0.0),
        params,
        controls,
    )
    np.testing.assert_allclose(np.asarray(state_only_accepted["state"]), np.asarray(accepted["state"]))
    assert bool(state_only_accepted["done"]) == bool(accepted["done"])
    assert state_only_accepted["history"] == {}

    accepted_only = jax_visible_accepted_only_nonlinear_controller_jax(
        step_plain,
        lambda state, _params, _control, _aux: state > 1.0,
        jnp.asarray(0.0),
        params,
        controls[:2],
    )
    np.testing.assert_allclose(np.asarray(accepted_only["state"]), 1.5)
    np.testing.assert_array_equal(np.asarray(accepted_only["history"]["active"]), [True, True])
    np.testing.assert_array_equal(np.asarray(accepted_only["history"]["accepted"]), [True, True])
    np.testing.assert_array_equal(np.asarray(accepted_only["history"]["rejected"]), [False, False])
    np.testing.assert_array_equal(np.asarray(accepted_only["history"]["done"]), [False, True])

    unrolled_accepted_only = jax_visible_unrolled_accepted_only_nonlinear_controller_jax(
        step_plain,
        lambda state, _params, _control, _aux: state > 1.0,
        jnp.asarray(0.0),
        params,
        controls[:2],
    )
    np.testing.assert_allclose(np.asarray(unrolled_accepted_only["state"]), np.asarray(accepted_only["state"]))
    for key in ("active", "accepted", "rejected", "done"):
        np.testing.assert_array_equal(
            np.asarray(unrolled_accepted_only["history"][key]),
            np.asarray(accepted_only["history"][key]),
        )

    state_only_accepted_only = jax_visible_state_only_accepted_only_nonlinear_controller_jax(
        step_plain,
        lambda state, _params, _control, _aux: state > 1.0,
        jnp.asarray(0.0),
        params,
        controls[:2],
    )
    np.testing.assert_allclose(np.asarray(state_only_accepted_only["state"]), np.asarray(accepted_only["state"]))
    assert bool(state_only_accepted_only["done"]) == bool(accepted_only["done"])
    assert state_only_accepted_only["history"] == {}

    unrolled_state_only_accepted_only = jax_visible_unrolled_state_only_accepted_only_nonlinear_controller_jax(
        step_plain,
        lambda state, _params, _control, _aux: state > 1.0,
        jnp.asarray(0.0),
        params,
        controls[:2],
    )
    np.testing.assert_allclose(
        np.asarray(unrolled_state_only_accepted_only["state"]),
        np.asarray(accepted_only["state"]),
    )
    assert bool(unrolled_state_only_accepted_only["done"]) == bool(accepted_only["done"])
    assert unrolled_state_only_accepted_only["history"] == {}

    accepted_only_initial_done = jax_visible_accepted_only_nonlinear_controller_jax(
        step_plain,
        lambda state, _params, _control, _aux: state > 1.0,
        jnp.asarray(0.0),
        params,
        controls[:1],
        initial_done=True,
    )
    np.testing.assert_allclose(np.asarray(accepted_only_initial_done["state"]), 0.0)
    np.testing.assert_array_equal(np.asarray(accepted_only_initial_done["history"]["active"]), [False])
    np.testing.assert_array_equal(np.asarray(accepted_only_initial_done["history"]["accepted"]), [False])
    np.testing.assert_array_equal(np.asarray(accepted_only_initial_done["history"]["rejected"]), [False])
    np.testing.assert_array_equal(np.asarray(accepted_only_initial_done["history"]["done"]), [True])

    unrolled_accepted_only_initial_done = jax_visible_unrolled_accepted_only_nonlinear_controller_jax(
        step_plain,
        lambda state, _params, _control, _aux: state > 1.0,
        jnp.asarray(0.0),
        params,
        controls[:1],
        initial_done=True,
    )
    np.testing.assert_allclose(np.asarray(unrolled_accepted_only_initial_done["state"]), 0.0)
    np.testing.assert_array_equal(np.asarray(unrolled_accepted_only_initial_done["history"]["active"]), [False])
    np.testing.assert_array_equal(np.asarray(unrolled_accepted_only_initial_done["history"]["done"]), [True])

    state_only_accepted_initial_done = jax_visible_state_only_accepted_nonlinear_controller_jax(
        step_plain,
        lambda *_args: True,
        lambda *_args: False,
        jnp.asarray(0.0),
        params,
        controls[:1],
        initial_done=True,
    )
    np.testing.assert_allclose(np.asarray(state_only_accepted_initial_done["state"]), 0.0)
    assert bool(state_only_accepted_initial_done["done"]) is True
    assert state_only_accepted_initial_done["history"] == {}

    state_only_accepted_only_initial_done = jax_visible_state_only_accepted_only_nonlinear_controller_jax(
        step_plain,
        lambda state, _params, _control, _aux: state > 1.0,
        jnp.asarray(0.0),
        params,
        controls[:1],
        initial_done=True,
    )
    np.testing.assert_allclose(np.asarray(state_only_accepted_only_initial_done["state"]), 0.0)
    assert bool(state_only_accepted_only_initial_done["done"]) is True
    assert state_only_accepted_only_initial_done["history"] == {}

    unrolled_state_only_initial_done = jax_visible_unrolled_state_only_accepted_only_nonlinear_controller_jax(
        step_plain,
        lambda state, _params, _control, _aux: state > 1.0,
        jnp.asarray(0.0),
        params,
        controls[:1],
        initial_done=True,
    )
    np.testing.assert_allclose(np.asarray(unrolled_state_only_initial_done["state"]), 0.0)
    assert bool(unrolled_state_only_initial_done["done"]) is True
    assert unrolled_state_only_initial_done["history"] == {}

    with pytest.raises(ValueError, match="at least one segment"):
        jax_visible_segmented_accepted_nonlinear_controller_jax(
            step_plain,
            lambda *_args: True,
            lambda *_args: False,
            jnp.asarray(0.0),
            params,
            (),
        )

    with pytest.raises(ValueError, match="step_fns length"):
        jax_visible_segmented_accepted_nonlinear_controller_jax(
            (step_plain,),
            lambda *_args: True,
            lambda *_args: False,
            jnp.asarray(0.0),
            params,
            (controls[:1], controls[1:]),
        )

    with pytest.raises(ValueError, match="accepted_only_segments length"):
        jax_visible_segmented_accepted_nonlinear_controller_jax(
            (step_plain, step_plain),
            lambda *_args: True,
            lambda *_args: False,
            jnp.asarray(0.0),
            params,
            (controls[:1], controls[1:]),
            accepted_only_segments=(True,),
        )

    with pytest.raises(ValueError, match="at least one segment"):
        jax_visible_segmented_state_only_accepted_nonlinear_controller_jax(
            step_plain,
            lambda *_args: True,
            lambda *_args: False,
            jnp.asarray(0.0),
            params,
            (),
        )

    with pytest.raises(ValueError, match="step_fns length"):
        jax_visible_segmented_state_only_accepted_nonlinear_controller_jax(
            (step_plain,),
            lambda *_args: True,
            lambda *_args: False,
            jnp.asarray(0.0),
            params,
            (controls[:1], controls[1:]),
        )

    with pytest.raises(ValueError, match="accepted_only_segments length"):
        jax_visible_segmented_state_only_accepted_nonlinear_controller_jax(
            (step_plain, step_plain),
            lambda *_args: True,
            lambda *_args: False,
            jnp.asarray(0.0),
            params,
            (controls[:1], controls[1:]),
            accepted_only_segments=(True,),
        )

    np.testing.assert_allclose(np.asarray(_pytree_vdot_jax({}, {})), 0.0)


def test_segmented_accepted_controller_matches_monolithic_scan_and_gradient():
    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp, tree_util

    enable_x64(True)
    controls = {
        "gain": jnp.asarray([0.16, 2.4, 0.13, 0.11, 0.45], dtype=float),
        "bias": jnp.asarray(
            [
                [0.010, -0.012],
                [3.000, -2.700],
                [-0.018, 0.014],
                [0.012, 0.009],
                [0.020, -0.015],
            ],
            dtype=float,
        ),
        "accept": jnp.asarray([True, False, True, True, True]),
        "stop": jnp.asarray([False, False, False, True, False]),
    }
    control_segments = (
        {key: value[:2] for key, value in controls.items()},
        {key: value[2:] for key, value in controls.items()},
    )
    fast_control_segments = (
        {**{key: value[:1] for key, value in controls.items()}, "accept_fn_forbidden": jnp.asarray([True])},
        {key: value[1:2] for key, value in controls.items()},
        {**{key: value[2:4] for key, value in controls.items()}, "accept_fn_forbidden": jnp.asarray([True, True])},
        {key: value[4:] for key, value in controls.items()},
    )
    initial_state = jnp.asarray([0.06, -0.02], dtype=float)
    params = {
        "matrix": jnp.asarray([[0.22, -0.04], [0.08, 0.18]], dtype=float),
        "drive": jnp.asarray([0.40, -0.31], dtype=float),
    }
    direction = {
        "matrix": jnp.asarray([[0.015, -0.006], [0.010, 0.004]], dtype=float),
        "drive": jnp.asarray([0.035, -0.020], dtype=float),
    }

    def step(state, prm, control):
        drive = prm["matrix"] @ state + prm["drive"] + control["bias"]
        proposed = 0.70 * state + control["gain"] * jnp.tanh(drive)
        return proposed, {
            "drive_norm": jnp.vdot(drive, drive),
            "proposal_norm": jnp.linalg.norm(proposed - state),
        }

    def accept_fn(_state, _proposed_state, _params, control, _aux):
        return control["accept"]

    def guarded_accept_fn(_state, _proposed_state, _params, control, _aux):
        if "accept_fn_forbidden" in control:
            raise AssertionError("accepted-only segments must not evaluate accept_fn")
        return control["accept"]

    def converged_fn(_accepted_state, _params, control, _aux):
        return control["stop"]

    monolithic = jax_visible_accepted_nonlinear_controller_jax(
        step,
        accept_fn,
        converged_fn,
        initial_state,
        params,
        controls,
        checkpoint_steps=True,
    )
    segmented = jax_visible_segmented_accepted_nonlinear_controller_jax(
        step,
        accept_fn,
        converged_fn,
        initial_state,
        params,
        control_segments,
        checkpoint_steps=True,
    )
    fast_segmented = jax_visible_segmented_accepted_nonlinear_controller_jax(
        step,
        guarded_accept_fn,
        converged_fn,
        initial_state,
        params,
        fast_control_segments,
        checkpoint_steps=True,
        accepted_only_segments=(True, False, True, False),
    )
    unrolled_fast_segmented = jax_visible_segmented_accepted_nonlinear_controller_jax(
        step,
        guarded_accept_fn,
        converged_fn,
        initial_state,
        params,
        fast_control_segments,
        checkpoint_steps=True,
        accepted_only_segments=(True, False, True, False),
        unroll_accepted_only_segments_below=2,
    )

    tree_util.tree_map(
        lambda actual, expected: np.testing.assert_allclose(
            np.asarray(actual),
            np.asarray(expected),
            rtol=1.0e-14,
            atol=1.0e-14,
        ),
        segmented["state"],
        monolithic["state"],
    )
    tree_util.tree_map(
        lambda actual, expected: np.testing.assert_allclose(
            np.asarray(actual),
            np.asarray(expected),
            rtol=1.0e-14,
            atol=1.0e-14,
        ),
        fast_segmented["state"],
        monolithic["state"],
    )
    tree_util.tree_map(
        lambda actual, expected: np.testing.assert_allclose(
            np.asarray(actual),
            np.asarray(expected),
            rtol=1.0e-14,
            atol=1.0e-14,
        ),
        unrolled_fast_segmented["state"],
        monolithic["state"],
    )
    assert bool(segmented["done"]) == bool(monolithic["done"])
    assert bool(fast_segmented["done"]) == bool(monolithic["done"])
    assert bool(unrolled_fast_segmented["done"]) == bool(monolithic["done"])
    assert segmented["n_segments"] == 2
    assert fast_segmented["n_segments"] == 4
    assert unrolled_fast_segmented["n_segments"] == 4
    for key in ("active", "accepted", "rejected", "done"):
        np.testing.assert_array_equal(np.asarray(segmented["history"][key]), np.asarray(monolithic["history"][key]))
        np.testing.assert_array_equal(
            np.asarray(fast_segmented["history"][key]),
            np.asarray(monolithic["history"][key]),
        )
        np.testing.assert_array_equal(
            np.asarray(unrolled_fast_segmented["history"][key]),
            np.asarray(monolithic["history"][key]),
        )
    np.testing.assert_allclose(
        segmented["history"]["drive_norm"],
        monolithic["history"]["drive_norm"],
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        fast_segmented["history"]["drive_norm"],
        monolithic["history"]["drive_norm"],
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        unrolled_fast_segmented["history"]["drive_norm"],
        monolithic["history"]["drive_norm"],
        rtol=1.0e-14,
        atol=1.0e-14,
    )

    state_only_segmented = jax_visible_segmented_state_only_accepted_nonlinear_controller_jax(
        step,
        accept_fn,
        converged_fn,
        initial_state,
        params,
        control_segments,
        checkpoint_steps=True,
    )
    state_only_fast_segmented = jax_visible_segmented_state_only_accepted_nonlinear_controller_jax(
        step,
        guarded_accept_fn,
        converged_fn,
        initial_state,
        params,
        fast_control_segments,
        checkpoint_steps=True,
        accepted_only_segments=(True, False, True, False),
    )
    state_only_unrolled_fast_segmented = jax_visible_segmented_state_only_accepted_nonlinear_controller_jax(
        step,
        guarded_accept_fn,
        converged_fn,
        initial_state,
        params,
        fast_control_segments,
        checkpoint_steps=True,
        accepted_only_segments=(True, False, True, False),
        unroll_accepted_only_segments_below=2,
    )
    for state_only_run in (
        state_only_segmented,
        state_only_fast_segmented,
        state_only_unrolled_fast_segmented,
    ):
        tree_util.tree_map(
            lambda actual, expected: np.testing.assert_allclose(
                np.asarray(actual),
                np.asarray(expected),
                rtol=1.0e-14,
                atol=1.0e-14,
            ),
            state_only_run["state"],
            monolithic["state"],
        )
        assert bool(state_only_run["done"]) == bool(monolithic["done"])
        assert state_only_run["history"] == {}
        assert state_only_run["n_segments"] in (2, 4)

    def objective_from_run(run):
        accepted = jnp.asarray(run["history"]["accepted"], dtype=float)
        active = jnp.asarray(run["history"]["active"], dtype=float)
        return (
            0.5 * jnp.vdot(jnp.asarray([1.1, 0.8], dtype=float) * run["state"], run["state"])
            + 0.02 * jnp.sum(accepted * run["history"]["proposal_norm"])
            + 0.004 * jnp.sum(active * run["history"]["drive_norm"])
        )

    def monolithic_objective(controller_params):
        return objective_from_run(
            jax_visible_accepted_nonlinear_controller_jax(
                step,
                accept_fn,
                converged_fn,
                initial_state,
                controller_params,
                controls,
                checkpoint_steps=True,
            )
        )

    def segmented_objective(controller_params):
        return objective_from_run(
            jax_visible_segmented_accepted_nonlinear_controller_jax(
                step,
                accept_fn,
                converged_fn,
                initial_state,
                controller_params,
                control_segments,
                checkpoint_steps=True,
            )
        )

    def fast_segmented_objective(controller_params):
        return objective_from_run(
            jax_visible_segmented_accepted_nonlinear_controller_jax(
                step,
                guarded_accept_fn,
                converged_fn,
                initial_state,
                controller_params,
                fast_control_segments,
                checkpoint_steps=True,
                accepted_only_segments=(True, False, True, False),
            )
        )

    def state_only_objective_from_run(run):
        return 0.5 * jnp.vdot(jnp.asarray([1.1, 0.8], dtype=float) * run["state"], run["state"])

    def state_only_segmented_objective(controller_params):
        return state_only_objective_from_run(
            jax_visible_segmented_state_only_accepted_nonlinear_controller_jax(
                step,
                accept_fn,
                converged_fn,
                initial_state,
                controller_params,
                control_segments,
                checkpoint_steps=True,
            )
        )

    def state_only_fast_segmented_objective(controller_params):
        return state_only_objective_from_run(
            jax_visible_segmented_state_only_accepted_nonlinear_controller_jax(
                step,
                guarded_accept_fn,
                converged_fn,
                initial_state,
                controller_params,
                fast_control_segments,
                checkpoint_steps=True,
                accepted_only_segments=(True, False, True, False),
                unroll_accepted_only_segments_below=2,
            )
        )

    monolithic_grad = jax.grad(monolithic_objective)(params)
    segmented_grad = jax.grad(segmented_objective)(params)
    fast_segmented_grad = jax.grad(fast_segmented_objective)(params)
    state_only_segmented_grad = jax.grad(state_only_segmented_objective)(params)
    state_only_fast_segmented_grad = jax.grad(state_only_fast_segmented_objective)(params)
    monolithic_dir = tree_util.tree_reduce(
        lambda acc, leaf: acc + leaf,
        tree_util.tree_map(lambda grad_leaf, dir_leaf: jnp.vdot(grad_leaf, dir_leaf), monolithic_grad, direction),
        0.0,
    )
    segmented_dir = tree_util.tree_reduce(
        lambda acc, leaf: acc + leaf,
        tree_util.tree_map(lambda grad_leaf, dir_leaf: jnp.vdot(grad_leaf, dir_leaf), segmented_grad, direction),
        0.0,
    )
    fast_segmented_dir = tree_util.tree_reduce(
        lambda acc, leaf: acc + leaf,
        tree_util.tree_map(lambda grad_leaf, dir_leaf: jnp.vdot(grad_leaf, dir_leaf), fast_segmented_grad, direction),
        0.0,
    )
    state_only_segmented_dir = tree_util.tree_reduce(
        lambda acc, leaf: acc + leaf,
        tree_util.tree_map(
            lambda grad_leaf, dir_leaf: jnp.vdot(grad_leaf, dir_leaf),
            state_only_segmented_grad,
            direction,
        ),
        0.0,
    )
    state_only_fast_segmented_dir = tree_util.tree_reduce(
        lambda acc, leaf: acc + leaf,
        tree_util.tree_map(
            lambda grad_leaf, dir_leaf: jnp.vdot(grad_leaf, dir_leaf),
            state_only_fast_segmented_grad,
            direction,
        ),
        0.0,
    )
    np.testing.assert_allclose(segmented_objective(params), monolithic_objective(params), rtol=1.0e-14, atol=1.0e-14)
    np.testing.assert_allclose(
        fast_segmented_objective(params),
        monolithic_objective(params),
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        state_only_fast_segmented_objective(params),
        state_only_segmented_objective(params),
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(segmented_dir, monolithic_dir, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(fast_segmented_dir, monolithic_dir, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(state_only_fast_segmented_dir, state_only_segmented_dir, rtol=1.0e-12, atol=1.0e-12)

    eps = 1.0e-5
    params_plus = tree_util.tree_map(lambda value, step_value: value + eps * step_value, params, direction)
    params_minus = tree_util.tree_map(lambda value, step_value: value - eps * step_value, params, direction)
    fd = (segmented_objective(params_plus) - segmented_objective(params_minus)) / (2.0 * eps)
    np.testing.assert_allclose(segmented_dir, fd, rtol=1.0e-6, atol=1.0e-10)


def test_jax_visible_controller_direct_coil_gradient_matches_fd():
    pytest.importorskip("jax")
    from vmec_jax._compat import jnp

    enable_x64(True)
    radius = 1.27
    dofs = jnp.zeros((1, 3, 3), dtype=float)
    dofs = dofs.at[0, 0, 2].set(radius)
    dofs = dofs.at[0, 1, 1].set(radius)
    coil_params = CoilFieldParams(
        base_curve_dofs=dofs,
        base_currents=jnp.asarray([4.8e6], dtype=float),
        n_segments=32,
        regularization_epsilon=1.0e-9,
    )
    dofs_direction = jnp.zeros_like(coil_params.base_curve_dofs)
    dofs_direction = dofs_direction.at[0, 0, 2].set(0.012)
    dofs_direction = dofs_direction.at[0, 1, 1].set(-0.010)
    direction = coil_params.with_arrays(
        base_curve_dofs=dofs_direction,
        base_currents=0.006 * coil_params.base_currents,
    )
    controls = {
        "gain": jnp.asarray([0.08, 0.06, 0.05], dtype=float),
        "phase": jnp.asarray([0.10, 0.38, 0.71], dtype=float),
    }

    def step(state, params, control):
        R = jnp.asarray(
            [
                [0.76 + 0.025 * state[0], 0.84 - 0.015 * state[1]],
                [0.91 + 0.010 * state[1], 0.80 + 0.018 * state[0]],
            ],
            dtype=float,
        )
        Z = jnp.asarray(
            [
                [0.11 + 0.020 * state[1], -0.12 + 0.010 * state[0]],
                [0.16 - 0.018 * state[0], -0.18 - 0.012 * state[1]],
            ],
            dtype=float,
        )
        phi = control["phase"] + jnp.asarray([[0.0, 0.17], [0.31, 0.49]], dtype=float)
        br, bp, bz = sample_coil_field_cylindrical(params, R, Z, phi)
        drive = jnp.asarray(
            [
                1.0e2 * jnp.mean(br + 0.15 * bp),
                1.0e2 * jnp.mean(bz - 0.10 * br),
            ],
            dtype=float,
        )
        next_state = 0.68 * state + control["gain"] * jnp.tanh(drive)
        return next_state, {"drive": drive, "bmean": jnp.mean(br * br + bp * bp + bz * bz)}

    def objective_from_run(controller_run):
        state = controller_run["state"]
        return 0.5 * jnp.vdot(state, state) + 1.0e-3 * jnp.sum(controller_run["history"]["bmean"])

    check = jax_visible_nonlinear_controller_directional_check_jax(
        step,
        objective_from_run,
        coil_params,
        direction,
        jnp.asarray([0.01, -0.02], dtype=float),
        controls,
        eps=1.0e-4,
        checkpoint_steps=True,
    )
    assert np.isfinite(float(check["value"]))
    assert abs(float(check["exact_directional"])) > 1.0e-12
    np.testing.assert_allclose(check["exact_directional"], check["fd_directional"], rtol=2.0e-5, atol=1.0e-10)


def test_masked_controller_direct_coil_projected_mode_ad_matches_fd_for_current_and_fourier():
    """Validate a JAX-visible masked free-boundary surrogate controller.

    The surrogate keeps the production dependency claim narrow:
    direct-coil parameters move through Biot-Savart sampling on a moving
    boundary, boundary projection, a dense mode-space vacuum solve, and an
    unrolled masked nonlinear controller.  It does not exercise or claim a VJP
    for production ``run_free_boundary``.
    """

    pytest.importorskip("jax")
    from vmec_jax._compat import jnp

    enable_x64(True)
    radius = 1.31
    dofs = jnp.zeros((1, 3, 3), dtype=float)
    dofs = dofs.at[0, 0, 2].set(radius)
    dofs = dofs.at[0, 1, 1].set(radius)
    coil_params = CoilFieldParams(
        base_curve_dofs=dofs,
        base_currents=jnp.asarray([4.6e6], dtype=float),
        n_segments=32,
        regularization_epsilon=1.0e-9,
    )
    controls = {
        "gain": jnp.asarray([0.12, 0.10, 0.09, 0.08, 0.75], dtype=float),
        "phase": jnp.asarray([0.06, 0.24, 0.43, 0.67, 0.91], dtype=float),
        "bias": jnp.asarray(
            [
                [0.004, -0.003, 0.002],
                [0.002, -0.001, 0.003],
                [-0.003, 0.002, 0.001],
                [0.001, 0.004, -0.002],
                [0.70, -0.60, 0.50],
            ],
            dtype=float,
        ),
        "stop": jnp.asarray([False, False, False, True, False]),
    }
    sin_basis = jnp.asarray(
        [
            [0.00, 0.21, -0.28],
            [0.37, -0.13, 0.48],
            [-0.24, 0.58, 0.16],
            [0.65, 0.27, -0.36],
        ],
        dtype=float,
    )
    mode_matrix = jnp.asarray(
        [
            [3.15, 0.14, -0.07],
            [0.11, 2.85, 0.19],
            [-0.05, 0.22, 3.05],
        ],
        dtype=float,
    )
    mode_to_state = jnp.asarray(
        [
            [0.10, -0.04, 0.03],
            [-0.03, 0.08, 0.05],
            [0.04, 0.02, -0.07],
        ],
        dtype=float,
    )
    phi_offsets = jnp.asarray([[0.00, 0.17], [0.34, 0.53]], dtype=float)
    imirr = jnp.asarray([1, 0, 3, 2])
    initial_state = jnp.asarray([0.015, -0.010, 0.012], dtype=float)

    def boundary_from_state(state, control):
        shape = jnp.asarray([[0.24, -0.16], [0.32, -0.21]], dtype=float)
        return {
            "R": jnp.asarray([[0.76, 0.84], [0.90, 0.79]], dtype=float)
            + 0.026 * state[0]
            + 0.014 * state[2] * shape,
            "Z": jnp.asarray([[0.11, -0.12], [0.17, -0.18]], dtype=float)
            + 0.024 * state[1]
            - 0.010 * state[2] * shape,
            "phi": control["phase"] + phi_offsets,
            "Ru": jnp.asarray([[0.028, -0.038], [0.020, 0.048]], dtype=float) + 0.006 * state[2],
            "Zu": jnp.asarray([[0.205, 0.218], [0.192, 0.211]], dtype=float) - 0.005 * state[2],
            "Rv": jnp.asarray([[0.038, 0.012], [-0.028, 0.046]], dtype=float),
            "Zv": jnp.asarray([[0.018, -0.030], [0.055, -0.013]], dtype=float),
        }

    def step(state, params, control):
        boundary = boundary_from_state(state, control)
        br, bp, bz = sample_coil_field_cylindrical(
            params,
            boundary["R"],
            boundary["Z"],
            boundary["phi"],
        )
        vac = vacuum_boundary_fields_from_cylindrical_jax(
            br=br,
            bp=bp,
            bz=bz,
            R=boundary["R"],
            Ru=boundary["Ru"],
            Zu=boundary["Zu"],
            Rv=boundary["Rv"],
            Zv=boundary["Zv"],
        )
        rhs_mode = mode_rhs_from_gsource_jax(
            vac["bnormal"],
            sin_basis=sin_basis,
            xmpot=jnp.asarray([0, 1, 1]),
            n_raw=jnp.asarray([0, 0, 1]),
            onp=1.0,
            lasym=False,
            imirr=imirr,
            nuv3=4,
            nuv_full=4,
        )
        response = dense_mode_vacuum_solve_jax(mode_matrix, rhs_mode, sin_basis)
        mode_coeffs = jnp.asarray(response["mode_coeffs"])
        drive = mode_to_state @ mode_coeffs + control["bias"]
        next_state = 0.64 * state + control["gain"] * jnp.tanh(drive)
        return next_state, {
            "mode_norm": jnp.vdot(mode_coeffs, mode_coeffs),
            "bnormal_norm": jnp.vdot(vac["bnormal"], vac["bnormal"]),
            "step_norm": jnp.linalg.norm(next_state - state),
        }

    def converged_fn(_next_state, _params, control, _aux):
        return control["stop"]

    def objective_from_run(controller_run):
        active = jnp.asarray(controller_run["history"]["active"], dtype=float)
        state = controller_run["state"]
        return (
            0.5 * jnp.vdot(jnp.asarray([1.0, 0.8, 1.1], dtype=float) * state, state)
            + 3.0e-4 * jnp.sum(active * controller_run["history"]["mode_norm"])
            + 5.0e-8 * jnp.sum(active * controller_run["history"]["bnormal_norm"])
        )

    zero_dofs = jnp.zeros_like(coil_params.base_curve_dofs)
    zero_currents = jnp.zeros_like(coil_params.base_currents)
    current_direction = coil_params.with_arrays(
        base_curve_dofs=zero_dofs,
        base_currents=0.02 * coil_params.base_currents,
    )
    fourier_direction = coil_params.with_arrays(
        base_curve_dofs=zero_dofs.at[0, 0, 2].set(0.018),
        base_currents=zero_currents,
    )

    current_check = jax_visible_masked_nonlinear_controller_directional_check_jax(
        step,
        converged_fn,
        objective_from_run,
        coil_params,
        current_direction,
        initial_state,
        controls,
        eps=1.0e-3,
        checkpoint_steps=True,
    )
    fourier_check = jax_visible_masked_nonlinear_controller_directional_check_jax(
        step,
        converged_fn,
        objective_from_run,
        coil_params,
        fourier_direction,
        initial_state,
        controls,
        eps=1.0e-3,
        checkpoint_steps=True,
    )

    np.testing.assert_array_equal(
        np.asarray(current_check["run"]["history"]["active"]),
        np.asarray([True, True, True, True, False]),
    )
    assert bool(current_check["run"]["done"]) is True
    for check in (current_check, fourier_check):
        assert np.isfinite(float(check["value"]))
        assert abs(float(check["exact_directional"])) > 1.0e-12
        np.testing.assert_allclose(check["exact_directional"], check["fd_directional"], rtol=7.0e-5, atol=1.0e-10)


def test_accepted_controller_direct_coil_projected_mode_ad_matches_fd_and_rejects_bad_step():
    """Validate accepted/rejected JAX-visible control flow for direct coils.

    This is a production-controller bridge: a deliberately large proposal is
    rejected by a JAX-visible accept mask, later accepted steps continue from
    the previous state, and the coil current/Fourier directional derivatives
    still match central finite differences.  The test does not claim a VJP for
    production ``run_free_boundary``; it validates the static-scan structure
    that production accepted/rejected control should move toward.
    """

    pytest.importorskip("jax")
    from vmec_jax._compat import jnp, tree_util

    enable_x64(True)
    radius = 1.28
    dofs = jnp.zeros((1, 3, 3), dtype=float)
    dofs = dofs.at[0, 0, 2].set(radius)
    dofs = dofs.at[0, 1, 1].set(radius)
    coil_params = CoilFieldParams(
        base_curve_dofs=dofs,
        base_currents=jnp.asarray([4.4e6], dtype=float),
        n_segments=32,
        regularization_epsilon=1.0e-9,
    )
    controls = {
        "gain": jnp.asarray([0.10, 4.0, 0.09, 0.08, 0.75], dtype=float),
        "phase": jnp.asarray([0.04, 0.21, 0.40, 0.63, 0.88], dtype=float),
        "bias": jnp.asarray(
            [
                [0.003, -0.002, 0.002],
                [8.0, -7.0, 6.0],
                [-0.002, 0.003, 0.001],
                [0.002, 0.003, -0.002],
                [0.70, -0.60, 0.50],
            ],
            dtype=float,
        ),
        "accept": jnp.asarray([True, False, True, True, True]),
        "stop": jnp.asarray([False, False, False, True, False]),
    }
    sin_basis = jnp.asarray(
        [
            [0.00, 0.20, -0.27],
            [0.34, -0.12, 0.46],
            [-0.22, 0.55, 0.15],
            [0.61, 0.25, -0.34],
        ],
        dtype=float,
    )
    mode_matrix = jnp.asarray(
        [
            [3.25, 0.12, -0.06],
            [0.10, 2.95, 0.17],
            [-0.04, 0.20, 3.15],
        ],
        dtype=float,
    )
    mode_to_state = jnp.asarray(
        [
            [0.09, -0.035, 0.025],
            [-0.025, 0.075, 0.045],
            [0.035, 0.018, -0.065],
        ],
        dtype=float,
    )
    phi_offsets = jnp.asarray([[0.00, 0.16], [0.33, 0.51]], dtype=float)
    imirr = jnp.asarray([1, 0, 3, 2])
    initial_state = jnp.asarray([0.012, -0.011, 0.010], dtype=float)

    def boundary_from_state(state, control):
        shape = jnp.asarray([[0.21, -0.15], [0.30, -0.20]], dtype=float)
        return {
            "R": jnp.asarray([[0.75, 0.83], [0.89, 0.78]], dtype=float)
            + 0.024 * state[0]
            + 0.012 * state[2] * shape,
            "Z": jnp.asarray([[0.10, -0.12], [0.16, -0.17]], dtype=float)
            + 0.022 * state[1]
            - 0.009 * state[2] * shape,
            "phi": control["phase"] + phi_offsets,
            "Ru": jnp.asarray([[0.027, -0.036], [0.019, 0.046]], dtype=float) + 0.005 * state[2],
            "Zu": jnp.asarray([[0.202, 0.216], [0.190, 0.209]], dtype=float) - 0.004 * state[2],
            "Rv": jnp.asarray([[0.036, 0.011], [-0.027, 0.044]], dtype=float),
            "Zv": jnp.asarray([[0.017, -0.028], [0.052, -0.012]], dtype=float),
        }

    def step(state, params, control):
        boundary = boundary_from_state(state, control)
        br, bp, bz = sample_coil_field_cylindrical(
            params,
            boundary["R"],
            boundary["Z"],
            boundary["phi"],
        )
        vac = vacuum_boundary_fields_from_cylindrical_jax(
            br=br,
            bp=bp,
            bz=bz,
            R=boundary["R"],
            Ru=boundary["Ru"],
            Zu=boundary["Zu"],
            Rv=boundary["Rv"],
            Zv=boundary["Zv"],
        )
        rhs_mode = mode_rhs_from_gsource_jax(
            vac["bnormal"],
            sin_basis=sin_basis,
            xmpot=jnp.asarray([0, 1, 1]),
            n_raw=jnp.asarray([0, 0, 1]),
            onp=1.0,
            lasym=False,
            imirr=imirr,
            nuv3=4,
            nuv_full=4,
        )
        response = dense_mode_vacuum_solve_jax(mode_matrix, rhs_mode, sin_basis)
        mode_coeffs = jnp.asarray(response["mode_coeffs"])
        drive = mode_to_state @ mode_coeffs + control["bias"]
        proposed = 0.62 * state + control["gain"] * jnp.tanh(drive)
        return proposed, {
            "mode_norm": jnp.vdot(mode_coeffs, mode_coeffs),
            "bnormal_norm": jnp.vdot(vac["bnormal"], vac["bnormal"]),
            "proposal_norm": jnp.linalg.norm(proposed - state),
        }

    def accept_fn(_state, _proposed_state, _params, control, _aux):
        return control["accept"]

    def converged_fn(_accepted_state, _params, control, _aux):
        return control["stop"]

    def objective_from_run(controller_run):
        history = controller_run["history"]
        accepted = jnp.asarray(history["accepted"], dtype=float)
        state = controller_run["state"]
        return (
            0.5 * jnp.vdot(jnp.asarray([1.0, 0.9, 1.2], dtype=float) * state, state)
            + 2.0e-4 * jnp.sum(accepted * history["mode_norm"])
            + 4.0e-8 * jnp.sum(accepted * history["bnormal_norm"])
        )

    zero_dofs = jnp.zeros_like(coil_params.base_curve_dofs)
    zero_currents = jnp.zeros_like(coil_params.base_currents)
    current_direction = coil_params.with_arrays(
        base_curve_dofs=zero_dofs,
        base_currents=0.018 * coil_params.base_currents,
    )
    fourier_direction = coil_params.with_arrays(
        base_curve_dofs=zero_dofs.at[0, 0, 2].set(0.016),
        base_currents=zero_currents,
    )

    current_check = jax_visible_accepted_nonlinear_controller_directional_check_jax(
        step,
        accept_fn,
        converged_fn,
        objective_from_run,
        coil_params,
        current_direction,
        initial_state,
        controls,
        eps=1.0e-3,
        checkpoint_steps=True,
    )
    fourier_check = jax_visible_accepted_nonlinear_controller_directional_check_jax(
        step,
        accept_fn,
        converged_fn,
        objective_from_run,
        coil_params,
        fourier_direction,
        initial_state,
        controls,
        eps=1.0e-3,
        checkpoint_steps=True,
    )

    history = current_check["run"]["history"]
    np.testing.assert_array_equal(np.asarray(history["active"]), np.asarray([True, True, True, True, False]))
    np.testing.assert_array_equal(np.asarray(history["accepted"]), np.asarray([True, False, True, True, False]))
    np.testing.assert_array_equal(np.asarray(history["rejected"]), np.asarray([False, True, False, False, False]))
    assert bool(current_check["run"]["done"]) is True

    changed_rejected_controls = {
        **controls,
        "gain": controls["gain"].at[1].set(25.0),
        "bias": controls["bias"].at[1].set(jnp.asarray([-30.0, 25.0, -20.0], dtype=float)),
    }
    changed_rejected_run = jax_visible_accepted_nonlinear_controller_jax(
        step,
        accept_fn,
        converged_fn,
        initial_state,
        coil_params,
        changed_rejected_controls,
        checkpoint_steps=True,
    )
    tree_util.tree_map(
        lambda actual, expected: np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=0.0, atol=0.0),
        changed_rejected_run["state"],
        current_check["run"]["state"],
    )

    for check in (current_check, fourier_check):
        assert np.isfinite(float(check["value"]))
        assert abs(float(check["exact_directional"])) > 1.0e-12
        np.testing.assert_allclose(check["exact_directional"], check["fd_directional"], rtol=7.0e-5, atol=1.0e-10)


def _mode_basis_for_rhs_tests(*, lasym: bool = False):
    ntheta, nzeta = 4, 5
    wint = np.full((ntheta, nzeta), 1.0 / float(ntheta * nzeta))
    return _build_vmec_mode_basis(
        ntheta=ntheta,
        nzeta=nzeta,
        nfp=2,
        mf=2,
        nf=1,
        lasym=lasym,
        wint=wint,
    )


def _mode_rhs_from_basis(gsource, basis):
    return mode_rhs_from_gsource_jax(
        gsource,
        sin_basis=basis["sinmni"],
        cos_basis=basis["cosmni"],
        xmpot=basis["xmpot"],
        n_raw=basis["n_raw"],
        onp=float(basis["onp"]),
        lasym=bool(basis["lasym"]),
        nuv3=int(basis["nuv3"]),
        nuv_full=int(basis["nuv_full"]),
        imirr=basis["imirr"],
        imirr_full=basis["imirr_full"],
    )


def _mode_matrix_from_basis(grpmn, basis):
    return mode_matrix_from_grpmn_jax(
        grpmn,
        sin_basis=basis["sinmni"],
        cos_basis=basis["cosmni"],
        xmpot=basis["xmpot"],
        n_raw=basis["n_raw"],
        lasym=bool(basis["lasym"]),
        mn0=int(basis["mn0"]),
    )


def _nonsingular_boundary_sample(*, radius_shift: float = 0.0, lasym: bool = True):
    basis = _mode_basis_for_rhs_tests(lasym=lasym)
    nzeta = 5
    ntheta = int(basis["nuv3"]) // nzeta
    theta = np.asarray(basis["theta"], dtype=float).reshape(ntheta, nzeta)
    zeta = np.asarray(basis["zeta"], dtype=float).reshape(ntheta, nzeta)
    R = 1.25 + 0.04 * radius_shift + 0.05 * np.cos(theta) + 0.02 * np.cos(theta - zeta)
    Z = 0.18 * np.sin(theta) + 0.03 * np.sin(theta + zeta)
    Ru = -0.05 * np.sin(theta) - 0.02 * np.sin(theta - zeta)
    Zu = 0.18 * np.cos(theta) + 0.03 * np.cos(theta + zeta)
    Rv = 0.02 * np.sin(theta - zeta)
    Zv = 0.03 * np.cos(theta + zeta)
    ruu = -0.05 * np.cos(theta) - 0.02 * np.cos(theta - zeta)
    ruv = 0.02 * np.cos(theta - zeta)
    rvv = -0.02 * np.cos(theta - zeta)
    zuu = -0.18 * np.sin(theta) - 0.03 * np.sin(theta + zeta)
    zuv = -0.03 * np.sin(theta + zeta)
    zvv = -0.03 * np.sin(theta + zeta)
    zeros = np.zeros_like(R)
    ones = np.ones_like(R)
    vac = VacuumBoundaryFields(
        bu=zeros,
        bv=zeros,
        bsupu=zeros,
        bsupv=zeros,
        bsqvac=zeros,
        bnormal=zeros,
        bnormal_unit=zeros,
        g_uu=ones,
        g_uv=zeros,
        g_vv=ones,
        det_guv=ones,
    )
    sample = ExternalBoundarySample(
        mgrid_path="synthetic",
        R=R,
        Z=Z,
        Ru=Ru,
        Zu=Zu,
        Rv=Rv,
        Zv=Zv,
        phi=zeta / float(basis["nfp"]),
        br=zeros,
        bp=zeros,
        bz=zeros,
        br_mgrid=zeros,
        bp_mgrid=zeros,
        bz_mgrid=zeros,
        br_axis=zeros,
        bp_axis=zeros,
        bz_axis=zeros,
        axis_r=np.zeros((1,), dtype=float),
        axis_z=np.zeros((1,), dtype=float),
        vac_ext=vac,
        ruu=ruu,
        ruv=ruv,
        rvv=rvv,
        zuu=zuu,
        zuv=zuv,
        zvv=zvv,
    )
    return basis, sample


def _jax_nonsingular_terms(sample, basis, bexni):
    tables = _ensure_vmec_nonsingular_kernel_tables(basis=basis, nv=sample.R.shape[1], nvper=2)
    return vmec_nonsingular_terms_from_bexni_jax(
        R=sample.R,
        Z=sample.Z,
        Ru=sample.Ru,
        Zu=sample.Zu,
        Rv=sample.Rv,
        Zv=sample.Zv,
        ruu=sample.ruu,
        ruv=sample.ruv,
        rvv=sample.rvv,
        zuu=sample.zuu,
        zuv=sample.zuv,
        zvv=sample.zvv,
        bexni=bexni,
        basis=basis,
        tables=tables,
        signgs=1,
        nvper=2,
    )


def _jax_analytic_terms(sample, basis, bexni):
    return vmec_analytic_terms_from_geometry_jax(
        R=sample.R,
        Ru=sample.Ru,
        Rv=sample.Rv,
        Zu=sample.Zu,
        Zv=sample.Zv,
        ruu=sample.ruu,
        ruv=sample.ruv,
        rvv=sample.rvv,
        zuu=sample.zuu,
        zuv=sample.zuv,
        zvv=sample.zvv,
        bexni=bexni,
        basis=basis,
        signgs=1,
    )


@pytest.mark.parametrize("lasym", [False, True])
def test_jax_vmec_source_and_mode_rhs_match_numpy_reference(lasym):
    enable_x64(True)
    basis = _mode_basis_for_rhs_tests(lasym=lasym)
    gsource = np.linspace(-0.8, 1.3, int(basis["nuv_full"]), dtype=float)

    actual_source = vmec_source_from_gsource_jax(
        gsource,
        onp=float(basis["onp"]),
        lasym=bool(basis["lasym"]),
        nuv3=int(basis["nuv3"]),
        nuv_full=int(basis["nuv_full"]),
        imirr=basis["imirr"],
        imirr_full=basis["imirr_full"],
    )
    expected_source = np.asarray(_vmec_source_from_gsource(gsource=gsource, basis=basis))
    np.testing.assert_allclose(actual_source, expected_source, rtol=1.0e-13, atol=1.0e-13)

    actual_rhs = _mode_rhs_from_basis(gsource, basis)
    expected_rhs = _vmec_bvec_from_gsource(gsource=gsource, basis=basis)
    np.testing.assert_allclose(actual_rhs, expected_rhs, rtol=1.0e-13, atol=1.0e-13)


@pytest.mark.parametrize("lasym", [False, True])
def test_jax_vmec_mode_rhs_gradient_wrt_gsource_matches_finite_difference(lasym):
    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp

    enable_x64(True)
    basis = _mode_basis_for_rhs_tests(lasym=lasym)
    gsource = jnp.asarray(np.linspace(-0.8, 1.3, int(basis["nuv_full"]), dtype=float))
    direction = jnp.asarray(np.cos(np.arange(int(basis["nuv_full"]), dtype=float)))
    rhs0 = _mode_rhs_from_basis(gsource, basis)
    weights = jnp.asarray(np.linspace(0.3, 1.1, int(rhs0.shape[0]), dtype=float))

    def objective(scale):
        rhs = _mode_rhs_from_basis(gsource + scale * direction, basis)
        return jnp.vdot(weights, rhs)

    exact = jax.grad(objective)(0.0)
    eps = 1.0e-6
    fd = (objective(eps) - objective(-eps)) / (2.0 * eps)
    np.testing.assert_allclose(exact, fd, rtol=3.0e-9, atol=1.0e-11)


@pytest.mark.parametrize("lasym", [False, True])
def test_jax_vmec_mode_matrix_matches_numpy_reference(lasym):
    enable_x64(True)
    basis = _mode_basis_for_rhs_tests(lasym=lasym)
    mnpd2 = int(basis["mnpd2"])
    nuv3 = int(basis["nuv3"])
    rows = np.arange(mnpd2, dtype=float)[:, None]
    cols = np.arange(nuv3, dtype=float)[None, :]
    grpmn = 0.04 * np.sin(0.3 + 0.2 * rows + 0.1 * cols) + 0.02 * np.cos(0.4 * rows - 0.3 * cols)

    actual = _mode_matrix_from_basis(grpmn, basis)
    expected = _vmec_mode_matrix_from_grpmn(grpmn=grpmn, basis=basis)

    np.testing.assert_allclose(actual, expected, rtol=1.0e-13, atol=1.0e-13)


@pytest.mark.parametrize("lasym", [False, True])
def test_jax_vmec_mode_matrix_gradient_wrt_grpmn_matches_finite_difference(lasym):
    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp

    enable_x64(True)
    basis = _mode_basis_for_rhs_tests(lasym=lasym)
    mnpd2 = int(basis["mnpd2"])
    nuv3 = int(basis["nuv3"])
    rows = np.arange(mnpd2, dtype=float)[:, None]
    cols = np.arange(nuv3, dtype=float)[None, :]
    grpmn = jnp.asarray(
        0.03 * np.sin(0.2 + 0.15 * rows + 0.1 * cols)
        + 0.01 * np.cos(0.25 * rows - 0.35 * cols),
        dtype=float,
    )
    direction = jnp.asarray(np.cos(0.4 * rows + 0.2 * cols), dtype=float)
    weights = jnp.asarray(np.sin(0.1 + np.arange(int(_mode_matrix_from_basis(grpmn, basis).size))), dtype=float)
    weights = jnp.reshape(weights, _mode_matrix_from_basis(grpmn, basis).shape)

    def objective(scale):
        matrix = _mode_matrix_from_basis(grpmn + scale * direction, basis)
        return jnp.vdot(weights, matrix)

    exact = jax.grad(objective)(0.0)
    eps = 1.0e-6
    fd = (objective(eps) - objective(-eps)) / (2.0 * eps)

    np.testing.assert_allclose(exact, fd, rtol=3.0e-9, atol=1.0e-11)


@pytest.mark.parametrize("lasym", [False, True])
def test_jax_vmec_source_matrix_solve_chain_gradients_match_finite_difference(lasym):
    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp

    enable_x64(True)
    basis = _mode_basis_for_rhs_tests(lasym=lasym)
    gsource = jnp.asarray(np.linspace(-0.2, 0.35, int(basis["nuv_full"]), dtype=float))
    g_direction = jnp.asarray(np.sin(np.arange(int(basis["nuv_full"]), dtype=float) + 0.2))
    mnpd2 = int(basis["mnpd2"])
    nuv3 = int(basis["nuv3"])
    rows = np.arange(mnpd2, dtype=float)[:, None]
    cols = np.arange(nuv3, dtype=float)[None, :]
    grpmn = jnp.asarray(0.01 * np.sin(0.4 + 0.2 * rows + 0.15 * cols), dtype=float)
    grpmn_direction = jnp.asarray(0.02 * np.cos(0.2 * rows - 0.1 * cols), dtype=float)
    phi_weights = jnp.asarray(np.cos(0.3 + np.arange(nuv3, dtype=float)), dtype=float)

    def response(source_scale, matrix_scale):
        rhs = _mode_rhs_from_basis(gsource + source_scale * g_direction, basis)
        matrix = _mode_matrix_from_basis(grpmn + matrix_scale * grpmn_direction, basis)
        out = dense_mode_vacuum_solve_jax(
            matrix,
            rhs,
            basis["sinmni"],
            basis["cosmni"] if lasym else None,
        )
        return 0.5 * jnp.vdot(out["mode_coeffs"], out["mode_coeffs"]) + 0.1 * jnp.vdot(
            phi_weights,
            out["phi_flat"],
        )

    exact_source = jax.grad(lambda scale: response(scale, 0.0))(0.0)
    exact_matrix = jax.grad(lambda scale: response(0.0, scale))(0.0)
    eps = 1.0e-6
    fd_source = (response(eps, 0.0) - response(-eps, 0.0)) / (2.0 * eps)
    fd_matrix = (response(0.0, eps) - response(0.0, -eps)) / (2.0 * eps)

    np.testing.assert_allclose(exact_source, fd_source, rtol=5.0e-8, atol=1.0e-10)
    np.testing.assert_allclose(exact_matrix, fd_matrix, rtol=5.0e-8, atol=1.0e-10)


def test_jax_vmec_nonsingular_green_terms_match_numpy_reference():
    enable_x64(True)
    basis, sample = _nonsingular_boundary_sample()
    bexni = np.linspace(-0.18, 0.24, int(basis["nuv3"]), dtype=float)

    actual_gsource, actual_grpmn = _jax_nonsingular_terms(sample, basis, bexni)
    expected_gsource, expected_grpmn = _vmec_nonsingular_terms_from_bexni(
        sample=sample,
        basis=basis,
        bexni=bexni,
        signgs=1,
        nvper=2,
    )

    np.testing.assert_allclose(actual_gsource, expected_gsource, rtol=2.0e-12, atol=2.0e-12)
    np.testing.assert_allclose(actual_grpmn, expected_grpmn, rtol=2.0e-12, atol=2.0e-12)


def test_jax_vmec_nonsingular_green_scan_matches_unrolled_loop():
    enable_x64(True)
    basis, sample = _nonsingular_boundary_sample()
    tables = _ensure_vmec_nonsingular_kernel_tables(basis=basis, nv=sample.R.shape[1], nvper=2)
    bexni = np.linspace(-0.18, 0.24, int(basis["nuv3"]), dtype=float)

    kwargs = dict(
        R=sample.R,
        Z=sample.Z,
        Ru=sample.Ru,
        Zu=sample.Zu,
        Rv=sample.Rv,
        Zv=sample.Zv,
        ruu=sample.ruu,
        ruv=sample.ruv,
        rvv=sample.rvv,
        zuu=sample.zuu,
        zuv=sample.zuv,
        zvv=sample.zvv,
        bexni=bexni,
        basis=basis,
        signgs=1,
        nvper=2,
    )

    scan_gsource, scan_grpmn = vmec_nonsingular_terms_from_bexni_jax(
        **kwargs,
        tables=tables,
    )
    loop_gsource, loop_grpmn = vmec_nonsingular_terms_from_bexni_jax(
        **kwargs,
        tables={**tables, "use_ip_scan": False},
    )

    np.testing.assert_allclose(scan_gsource, loop_gsource, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(scan_grpmn, loop_grpmn, rtol=1.0e-13, atol=1.0e-13)


def test_jax_vmec_nonsingular_green_solve_chain_gradients_match_finite_difference():
    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp

    enable_x64(True)
    basis, sample = _nonsingular_boundary_sample()
    base_bex = jnp.asarray(np.linspace(-0.18, 0.24, int(basis["nuv3"]), dtype=float))
    bex_direction = jnp.asarray(np.cos(0.2 + np.arange(int(basis["nuv3"]), dtype=float)))
    phi_weights = jnp.asarray(np.sin(0.1 + np.arange(int(basis["nuv3"]), dtype=float)), dtype=float)

    def response(source_scale, geometry_scale):
        tables = _ensure_vmec_nonsingular_kernel_tables(basis=basis, nv=sample.R.shape[1], nvper=2)
        gsource, grpmn = vmec_nonsingular_terms_from_bexni_jax(
            R=jnp.asarray(sample.R) + 0.04 * geometry_scale,
            Z=sample.Z,
            Ru=sample.Ru,
            Zu=sample.Zu,
            Rv=sample.Rv,
            Zv=sample.Zv,
            ruu=sample.ruu,
            ruv=sample.ruv,
            rvv=sample.rvv,
            zuu=sample.zuu,
            zuv=sample.zuv,
            zvv=sample.zvv,
            bexni=base_bex + source_scale * bex_direction,
            basis=basis,
            tables=tables,
            signgs=1,
            nvper=2,
        )
        rhs = _mode_rhs_from_basis(gsource, basis)
        matrix = _mode_matrix_from_basis(grpmn, basis)
        out = dense_mode_vacuum_solve_jax(matrix, rhs, basis["sinmni"], basis["cosmni"])
        return 0.5 * jnp.vdot(out["mode_coeffs"], out["mode_coeffs"]) + 0.1 * jnp.vdot(
            phi_weights,
            out["phi_flat"],
        )

    exact_source = jax.grad(lambda scale: response(scale, 0.0))(0.0)
    exact_geometry = jax.grad(lambda scale: response(0.0, scale))(0.0)
    eps = 1.0e-6
    fd_source = (response(eps, 0.0) - response(-eps, 0.0)) / (2.0 * eps)
    fd_geometry = (response(0.0, eps) - response(0.0, -eps)) / (2.0 * eps)

    np.testing.assert_allclose(exact_source, fd_source, rtol=7.0e-7, atol=1.0e-10)
    np.testing.assert_allclose(exact_geometry, fd_geometry, rtol=7.0e-7, atol=1.0e-10)


@pytest.mark.parametrize("lasym", [False, True])
def test_jax_vmec_analytic_terms_match_numpy_reference(lasym):
    enable_x64(True)
    basis, sample = _nonsingular_boundary_sample()
    if not lasym:
        basis = _mode_basis_for_rhs_tests(lasym=False)
    bexni = np.linspace(-0.11, 0.29, int(basis["nuv3"]), dtype=float)

    actual_bvec, actual_grpmn = _jax_analytic_terms(sample, basis, bexni)
    expected_bvec, expected_grpmn = _vmec_analytic_terms_from_geometry(
        sample=sample,
        basis=basis,
        bexni=bexni,
        signgs=1,
    )

    np.testing.assert_allclose(actual_bvec, expected_bvec, rtol=4.0e-12, atol=4.0e-12)
    np.testing.assert_allclose(actual_grpmn, expected_grpmn, rtol=4.0e-12, atol=4.0e-12)


def test_jax_vmec_analytic_terms_validate_geometry_basis_and_source_shapes():
    enable_x64(True)
    basis, sample = _nonsingular_boundary_sample()
    bexni = np.linspace(-0.11, 0.29, int(basis["nuv3"]), dtype=float)

    with pytest.raises(ValueError, match="R must be a 2D"):
        vmec_analytic_terms_from_geometry_jax(
            R=np.ravel(sample.R),
            Ru=sample.Ru,
            Rv=sample.Rv,
            Zu=sample.Zu,
            Zv=sample.Zv,
            ruu=sample.ruu,
            ruv=sample.ruv,
            rvv=sample.rvv,
            zuu=sample.zuu,
            zuv=sample.zuv,
            zvv=sample.zvv,
            bexni=bexni,
            basis=basis,
            signgs=1,
        )
    with pytest.raises(ValueError, match="Ru must match R shape"):
        vmec_analytic_terms_from_geometry_jax(
            R=sample.R,
            Ru=sample.Ru[:, :-1],
            Rv=sample.Rv,
            Zu=sample.Zu,
            Zv=sample.Zv,
            ruu=sample.ruu,
            ruv=sample.ruv,
            rvv=sample.rvv,
            zuu=sample.zuu,
            zuv=sample.zuv,
            zvv=sample.zvv,
            bexni=bexni,
            basis=basis,
            signgs=1,
        )
    bad_basis = dict(basis)
    bad_basis["theta"] = np.asarray(basis["theta"])[:-1]
    with pytest.raises(ValueError, match="basis theta/zeta"):
        _jax_analytic_terms(sample, bad_basis, bexni)
    with pytest.raises(ValueError, match="bexni"):
        _jax_analytic_terms(sample, basis, bexni[:2])


@pytest.mark.parametrize("lasym", [False, True])
def test_jax_vmec_analytic_mode_solve_chain_gradients_match_finite_difference(lasym):
    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp

    enable_x64(True)
    basis, sample = _nonsingular_boundary_sample()
    if not lasym:
        basis = _mode_basis_for_rhs_tests(lasym=False)
    base_bex = jnp.asarray(np.linspace(-0.11, 0.29, int(basis["nuv3"]), dtype=float))
    bex_direction = jnp.asarray(np.sin(0.17 + np.arange(int(basis["nuv3"]), dtype=float)))
    phi_weights = jnp.asarray(np.cos(0.19 + np.arange(int(basis["nuv3"]), dtype=float)), dtype=float)

    def response(source_scale, geometry_scale):
        bvec, grpmn = vmec_analytic_terms_from_geometry_jax(
            R=jnp.asarray(sample.R) + 0.02 * geometry_scale,
            Ru=sample.Ru,
            Rv=sample.Rv,
            Zu=sample.Zu,
            Zv=sample.Zv,
            ruu=sample.ruu,
            ruv=sample.ruv,
            rvv=sample.rvv,
            zuu=sample.zuu,
            zuv=sample.zuv,
            zvv=sample.zvv,
            bexni=base_bex + source_scale * bex_direction,
            basis=basis,
            signgs=1,
        )
        matrix = _mode_matrix_from_basis(grpmn, basis)
        out = dense_mode_vacuum_solve_jax(
            matrix,
            bvec,
            basis["sinmni"],
            basis["cosmni"] if lasym else None,
        )
        return 0.5 * jnp.vdot(out["mode_coeffs"], out["mode_coeffs"]) + 0.1 * jnp.vdot(
            phi_weights,
            out["phi_flat"],
        )

    exact_source = jax.grad(lambda scale: response(scale, 0.0))(0.0)
    exact_geometry = jax.grad(lambda scale: response(0.0, scale))(0.0)
    eps = 1.0e-6
    fd_source = (response(eps, 0.0) - response(-eps, 0.0)) / (2.0 * eps)
    fd_geometry = (response(0.0, eps) - response(0.0, -eps)) / (2.0 * eps)

    np.testing.assert_allclose(exact_source, fd_source, rtol=2.0e-6, atol=1.0e-10)
    np.testing.assert_allclose(exact_geometry, fd_geometry, rtol=2.0e-6, atol=1.0e-10)


@pytest.mark.py311_slow_coverage
def test_jax_vmec_combined_analytic_nonsingular_solve_chain_gradients_match_finite_difference():
    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp

    enable_x64(True)
    basis, sample = _nonsingular_boundary_sample()
    base_bex = jnp.asarray(np.linspace(-0.11, 0.29, int(basis["nuv3"]), dtype=float))
    bex_direction = jnp.asarray(np.cos(0.23 + np.arange(int(basis["nuv3"]), dtype=float)))
    phi_weights = jnp.asarray(np.sin(0.27 + np.arange(int(basis["nuv3"]), dtype=float)), dtype=float)
    tables = _ensure_vmec_nonsingular_kernel_tables(basis=basis, nv=sample.R.shape[1], nvper=2)

    def response(source_scale, geometry_scale):
        R = jnp.asarray(sample.R) + 0.015 * geometry_scale
        bex = base_bex + source_scale * bex_direction
        gsource_nonsing, grpmn_nonsing = vmec_nonsingular_terms_from_bexni_jax(
            R=R,
            Z=sample.Z,
            Ru=sample.Ru,
            Zu=sample.Zu,
            Rv=sample.Rv,
            Zv=sample.Zv,
            ruu=sample.ruu,
            ruv=sample.ruv,
            rvv=sample.rvv,
            zuu=sample.zuu,
            zuv=sample.zuv,
            zvv=sample.zvv,
            bexni=bex,
            basis=basis,
            tables=tables,
            signgs=1,
            nvper=2,
        )
        bvec_analytic, grpmn_analytic = vmec_analytic_terms_from_geometry_jax(
            R=R,
            Ru=sample.Ru,
            Rv=sample.Rv,
            Zu=sample.Zu,
            Zv=sample.Zv,
            ruu=sample.ruu,
            ruv=sample.ruv,
            rvv=sample.rvv,
            zuu=sample.zuu,
            zuv=sample.zuv,
            zvv=sample.zvv,
            bexni=bex,
            basis=basis,
            signgs=1,
        )
        rhs = _mode_rhs_from_basis(gsource_nonsing, basis) + bvec_analytic
        matrix = _mode_matrix_from_basis(grpmn_nonsing + grpmn_analytic, basis)
        out = dense_mode_vacuum_solve_jax(matrix, rhs, basis["sinmni"], basis["cosmni"])
        return 0.5 * jnp.vdot(out["mode_coeffs"], out["mode_coeffs"]) + 0.1 * jnp.vdot(
            phi_weights,
            out["phi_flat"],
        )

    exact_source = jax.grad(lambda scale: response(scale, 0.0))(0.0)
    exact_geometry = jax.grad(lambda scale: response(0.0, scale))(0.0)
    eps = 1.0e-6
    fd_source = (response(eps, 0.0) - response(-eps, 0.0)) / (2.0 * eps)
    fd_geometry = (response(0.0, eps) - response(0.0, -eps)) / (2.0 * eps)

    np.testing.assert_allclose(exact_source, fd_source, rtol=3.0e-6, atol=1.0e-10)
    np.testing.assert_allclose(exact_geometry, fd_geometry, rtol=3.0e-6, atol=1.0e-10)


def test_dense_vmec_nestor_mode_solve_matches_manual_combined_operator():
    enable_x64(True)
    basis, sample = _nonsingular_boundary_sample()
    bex = np.linspace(-0.11, 0.29, int(basis["nuv3"]), dtype=float)
    tables = _ensure_vmec_nonsingular_kernel_tables(basis=basis, nv=sample.R.shape[1], nvper=2)

    actual = dense_vmec_nestor_mode_solve_jax(
        R=sample.R,
        Z=sample.Z,
        Ru=sample.Ru,
        Zu=sample.Zu,
        Rv=sample.Rv,
        Zv=sample.Zv,
        ruu=sample.ruu,
        ruv=sample.ruv,
        rvv=sample.rvv,
        zuu=sample.zuu,
        zuv=sample.zuv,
        zvv=sample.zvv,
        bexni=bex,
        basis=basis,
        tables=tables,
        signgs=1,
        nvper=2,
    )
    gsource_nonsing, grpmn_nonsing = _jax_nonsingular_terms(sample, basis, bex)
    bvec_analytic, grpmn_analytic = _jax_analytic_terms(sample, basis, bex)
    rhs = _mode_rhs_from_basis(gsource_nonsing, basis) + bvec_analytic
    matrix = _mode_matrix_from_basis(grpmn_nonsing + grpmn_analytic, basis)
    expected = dense_mode_vacuum_solve_jax(matrix, rhs, basis["sinmni"], basis["cosmni"])

    np.testing.assert_allclose(actual["rhs_mode"], rhs, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(actual["mode_matrix"], matrix, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(actual["mode_coeffs"], expected["mode_coeffs"], rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(actual["phi_flat"], expected["phi_flat"], rtol=1.0e-13, atol=1.0e-13)


def test_dense_vmec_nestor_mode_solve_matches_host_reduced_symmetric_grid():
    """Reduced stellarator-symmetric samples should match the host full-grid reconstruction path."""

    enable_x64(True)
    basis, sample = _nonsingular_boundary_sample(lasym=False)
    bex = np.linspace(-0.11, 0.29, int(basis["nuv3"]), dtype=float)
    tables = _ensure_vmec_nonsingular_kernel_tables(basis=basis, nv=sample.R.shape[1], nvper=2)

    actual = dense_vmec_nestor_mode_solve_jax(
        R=sample.R,
        Z=sample.Z,
        Ru=sample.Ru,
        Zu=sample.Zu,
        Rv=sample.Rv,
        Zv=sample.Zv,
        ruu=sample.ruu,
        ruv=sample.ruv,
        rvv=sample.rvv,
        zuu=sample.zuu,
        zuv=sample.zuv,
        zvv=sample.zvv,
        bexni=bex,
        basis=basis,
        tables=tables,
        signgs=1,
        nvper=2,
    )
    gsource_nonsing, grpmn_nonsing = _vmec_nonsingular_terms_from_bexni(
        sample=sample,
        basis=basis,
        bexni=bex,
        signgs=1,
        nvper=2,
    )
    bvec_analytic, grpmn_analytic = _vmec_analytic_terms_from_geometry(
        sample=sample,
        basis=basis,
        bexni=bex,
        signgs=1,
    )
    rhs = _vmec_bvec_from_gsource(gsource=gsource_nonsing, basis=basis) + bvec_analytic
    matrix = _vmec_mode_matrix_from_grpmn(grpmn=grpmn_nonsing + grpmn_analytic, basis=basis)
    expected = dense_mode_vacuum_solve_jax(matrix, rhs, basis["sinmni"])

    np.testing.assert_allclose(actual["gsource_nonsing"], gsource_nonsing, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(actual["grpmn"], grpmn_nonsing + grpmn_analytic, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(actual["rhs_mode"], rhs, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(actual["mode_matrix"], matrix, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(actual["mode_coeffs"], expected["mode_coeffs"], rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(actual["phi_flat"], expected["phi_flat"], rtol=1.0e-12, atol=1.0e-12)


@pytest.mark.parametrize("lasym", [False, True], ids=["stellsym", "lasym"])
@pytest.mark.py311_slow_coverage
def test_mode_matrix_matvec_matches_dense_mode_matrix(lasym: bool) -> None:
    """Matrix-free VMEC/NESTOR mode application must match dense assembly."""

    enable_x64(True)
    basis, sample = _nonsingular_boundary_sample(lasym=lasym)
    bex = np.linspace(-0.11, 0.29, int(basis["nuv3"]), dtype=float)
    tables = _ensure_vmec_nonsingular_kernel_tables(basis=basis, nv=sample.R.shape[1], nvper=2)
    response = dense_vmec_nestor_mode_solve_jax(
        R=sample.R,
        Z=sample.Z,
        Ru=sample.Ru,
        Zu=sample.Zu,
        Rv=sample.Rv,
        Zv=sample.Zv,
        ruu=sample.ruu,
        ruv=sample.ruv,
        rvv=sample.rvv,
        zuu=sample.zuu,
        zuv=sample.zuv,
        zvv=sample.zvv,
        bexni=bex,
        basis=basis,
        tables=tables,
        signgs=1,
        nvper=2,
        include_analytic=False,
        include_phi_flat=False,
        include_residual=False,
    )
    grpmn = response["grpmn"]
    matrix = mode_matrix_from_grpmn_jax(
        grpmn,
        sin_basis=basis["sinmni"],
        cos_basis=basis["cosmni"] if bool(basis["lasym"]) else None,
        xmpot=basis["xmpot"],
        n_raw=basis["n_raw"],
        lasym=bool(basis["lasym"]),
        mn0=int(basis["mn0"]),
    )
    size = int(matrix.shape[0])
    vector = np.sin(0.31 + np.arange(size, dtype=float))

    actual = mode_matrix_matvec_from_grpmn_jax(
        vector,
        grpmn,
        sin_basis=basis["sinmni"],
        cos_basis=basis["cosmni"] if bool(basis["lasym"]) else None,
        xmpot=basis["xmpot"],
        n_raw=basis["n_raw"],
        lasym=bool(basis["lasym"]),
        mn0=int(basis["mn0"]),
    )
    actual_t = mode_matrix_matvec_from_grpmn_jax(
        vector,
        grpmn,
        sin_basis=basis["sinmni"],
        cos_basis=basis["cosmni"] if bool(basis["lasym"]) else None,
        xmpot=basis["xmpot"],
        n_raw=basis["n_raw"],
        lasym=bool(basis["lasym"]),
        mn0=int(basis["mn0"]),
        transpose=True,
    )

    np.testing.assert_allclose(actual, matrix @ vector, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(actual_t, matrix.T @ vector, rtol=1.0e-12, atol=1.0e-12)


@pytest.mark.py311_slow_coverage
def test_matrix_free_mode_operator_solve_matches_dense_response_and_gradients() -> None:
    """GMRES mode response matches dense solve and AD-vs-FD source gradients."""

    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp

    enable_x64(True)
    basis, sample = _nonsingular_boundary_sample(lasym=True)
    bex = jnp.asarray(np.linspace(-0.11, 0.29, int(basis["nuv3"]), dtype=float))
    direction = jnp.asarray(np.cos(0.19 + np.arange(int(basis["nuv3"]), dtype=float)))
    gsource, grpmn = _jax_nonsingular_terms(sample, basis, bex)
    rhs = _mode_rhs_from_basis(gsource, basis)
    matrix = _mode_matrix_from_basis(grpmn, basis)
    dense = dense_mode_vacuum_solve_jax(matrix, rhs, basis["sinmni"], basis["cosmni"])
    matrix_free = mode_operator_vacuum_solve_jax(
        grpmn,
        rhs,
        sin_basis=basis["sinmni"],
        cos_basis=basis["cosmni"],
        xmpot=basis["xmpot"],
        n_raw=basis["n_raw"],
        lasym=True,
        mn0=int(basis["mn0"]),
        tol=1.0e-13,
        atol=1.0e-15,
        maxiter=64,
    )

    assert matrix_free["solve_mode"] == "matrix_free_gmres"
    assert matrix_free["mode_matrix_materialized"] is False
    np.testing.assert_allclose(matrix_free["mode_coeffs"], dense["mode_coeffs"], rtol=2.0e-11, atol=2.0e-11)
    np.testing.assert_allclose(matrix_free["phi_flat"], dense["phi_flat"], rtol=2.0e-11, atol=2.0e-11)
    np.testing.assert_allclose(matrix_free["residual"], np.zeros_like(np.asarray(rhs)), atol=2.0e-10)

    weights = jnp.asarray(np.sin(0.23 + np.arange(int(basis["nuv3"]), dtype=float)))

    def objective(scale):
        source = bex + scale * direction
        src, gmat = _jax_nonsingular_terms(sample, basis, source)
        rhs_scaled = _mode_rhs_from_basis(src, basis)
        out = mode_operator_vacuum_solve_jax(
            gmat,
            rhs_scaled,
            sin_basis=basis["sinmni"],
            cos_basis=basis["cosmni"],
            xmpot=basis["xmpot"],
            n_raw=basis["n_raw"],
            lasym=True,
            mn0=int(basis["mn0"]),
            include_phi_flat=True,
            include_residual=False,
            tol=1.0e-13,
            atol=1.0e-15,
            maxiter=64,
        )
        return jnp.vdot(weights, out["phi_flat"])

    exact = jax.grad(objective)(0.0)
    eps = 1.0e-6
    fd = (objective(eps) - objective(-eps)) / (2.0 * eps)
    np.testing.assert_allclose(exact, fd, rtol=5.0e-6, atol=1.0e-10)


@pytest.mark.py311_slow_coverage
def test_dense_vmec_nestor_mode_solve_can_use_matrix_free_response() -> None:
    """The combined JAX NESTOR solve can opt into matrix-free mode response."""

    enable_x64(True)
    basis, sample = _nonsingular_boundary_sample(lasym=False)
    bex = np.linspace(-0.11, 0.29, int(basis["nuv3"]), dtype=float)
    tables = _ensure_vmec_nonsingular_kernel_tables(basis=basis, nv=sample.R.shape[1], nvper=2)
    dense = dense_vmec_nestor_mode_solve_jax(
        R=sample.R,
        Z=sample.Z,
        Ru=sample.Ru,
        Zu=sample.Zu,
        Rv=sample.Rv,
        Zv=sample.Zv,
        ruu=sample.ruu,
        ruv=sample.ruv,
        rvv=sample.rvv,
        zuu=sample.zuu,
        zuv=sample.zuv,
        zvv=sample.zvv,
        bexni=bex,
        basis=basis,
        tables=tables,
        signgs=1,
        nvper=2,
    )
    matrix_free = dense_vmec_nestor_mode_solve_jax(
        R=sample.R,
        Z=sample.Z,
        Ru=sample.Ru,
        Zu=sample.Zu,
        Rv=sample.Rv,
        Zv=sample.Zv,
        ruu=sample.ruu,
        ruv=sample.ruv,
        rvv=sample.rvv,
        zuu=sample.zuu,
        zuv=sample.zuv,
        zvv=sample.zvv,
        bexni=bex,
        basis=basis,
        tables=tables,
        signgs=1,
        nvper=2,
        solve_mode="matrix_free",
        operator_tol=1.0e-13,
        operator_atol=1.0e-15,
        operator_maxiter=64,
    )

    assert matrix_free["solve_mode"] == "matrix_free_gmres"
    assert matrix_free["mode_matrix"] is None
    assert matrix_free["mode_matrix_materialized"] is False
    np.testing.assert_allclose(matrix_free["rhs_mode"], dense["rhs_mode"], rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(matrix_free["mode_coeffs"], dense["mode_coeffs"], rtol=2.0e-11, atol=2.0e-11)
    np.testing.assert_allclose(matrix_free["phi_flat"], dense["phi_flat"], rtol=2.0e-11, atol=2.0e-11)

    bicgstab = dense_vmec_nestor_mode_solve_jax(
        R=sample.R,
        Z=sample.Z,
        Ru=sample.Ru,
        Zu=sample.Zu,
        Rv=sample.Rv,
        Zv=sample.Zv,
        ruu=sample.ruu,
        ruv=sample.ruv,
        rvv=sample.rvv,
        zuu=sample.zuu,
        zuv=sample.zuv,
        zvv=sample.zvv,
        bexni=bex,
        basis=basis,
        tables=tables,
        signgs=1,
        nvper=2,
        solve_mode="bicgstab",
        operator_tol=1.0e-13,
        operator_atol=1.0e-15,
        operator_maxiter=64,
        include_phi_flat=False,
        include_residual=False,
    )

    assert bicgstab["solve_mode"] == "matrix_free_bicgstab"
    assert bicgstab["mode_matrix"] is None
    assert "phi_flat" not in bicgstab
    assert "residual" not in bicgstab
    np.testing.assert_allclose(bicgstab["mode_coeffs"], dense["mode_coeffs"], rtol=2.0e-11, atol=2.0e-11)

    with pytest.raises(ValueError, match="solve_mode"):
        dense_vmec_nestor_mode_solve_jax(
            R=sample.R,
            Z=sample.Z,
            Ru=sample.Ru,
            Zu=sample.Zu,
            Rv=sample.Rv,
            Zv=sample.Zv,
            ruu=sample.ruu,
            ruv=sample.ruv,
            rvv=sample.rvv,
            zuu=sample.zuu,
            zuv=sample.zuv,
            zvv=sample.zvv,
            bexni=bex,
            basis=basis,
            tables=tables,
            signgs=1,
            nvper=2,
            solve_mode="unknown",
        )


@pytest.mark.py311_slow_coverage
def test_matrix_free_mode_operator_alternate_solvers_and_validation_paths() -> None:
    """Exercise cheap matrix-free Krylov branches and explicit validation errors."""

    pytest.importorskip("jax")
    from vmec_jax._compat import jnp

    enable_x64(True)
    sin_basis = jnp.asarray(
        [
            [0.0, 0.2, -0.1],
            [0.4, -0.3, 0.5],
            [-0.2, 0.6, 0.1],
            [0.7, 0.1, -0.4],
        ],
        dtype=float,
    )
    xmpot = jnp.asarray([0, 1, 2])
    n_raw = jnp.asarray([0, 0, 1])
    grpmn = jnp.zeros((3, 4), dtype=float)
    rhs = jnp.asarray([1.0, -0.5, 0.25], dtype=float)
    expected_coeffs = rhs / float(4.0 * np.pi**3)

    restarted_gmres = mode_operator_vacuum_solve_jax(
        grpmn,
        rhs,
        sin_basis=sin_basis,
        xmpot=xmpot,
        n_raw=n_raw,
        lasym=False,
        solver="gmres",
        tol=1.0e-13,
        atol=1.0e-15,
        maxiter=8,
        restart=2,
    )
    bicgstab = mode_operator_vacuum_solve_jax(
        grpmn,
        rhs,
        sin_basis=sin_basis,
        xmpot=xmpot,
        n_raw=n_raw,
        lasym=False,
        solver="bicgstab",
        tol=1.0e-13,
        atol=1.0e-15,
        maxiter=8,
        include_phi_flat=False,
        include_residual=False,
    )

    assert restarted_gmres["solve_mode"] == "matrix_free_gmres"
    assert bicgstab["solve_mode"] == "matrix_free_bicgstab"
    assert "phi_flat" not in bicgstab
    assert "residual" not in bicgstab
    np.testing.assert_allclose(restarted_gmres["mode_coeffs"], expected_coeffs, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(restarted_gmres["phi_flat"], sin_basis @ expected_coeffs, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(restarted_gmres["residual"], np.zeros_like(np.asarray(rhs)), atol=1.0e-13)
    np.testing.assert_allclose(bicgstab["mode_coeffs"], expected_coeffs, rtol=1.0e-13, atol=1.0e-13)

    with pytest.raises(ValueError, match="solver must be"):
        mode_operator_vacuum_solve_jax(
            grpmn,
            rhs,
            sin_basis=sin_basis,
            xmpot=xmpot,
            n_raw=n_raw,
            lasym=False,
            solver="cg",
        )
    with pytest.raises(ValueError, match="1D rhs_mode"):
        mode_operator_vacuum_solve_jax(
            grpmn,
            rhs[:, None],
            sin_basis=sin_basis,
            xmpot=xmpot,
            n_raw=n_raw,
            lasym=False,
        )
    with pytest.raises(ValueError, match="sin_basis must be a 2D array"):
        mode_operator_vacuum_solve_jax(
            grpmn,
            rhs,
            sin_basis=sin_basis[:, 0],
            xmpot=xmpot,
            n_raw=n_raw,
            lasym=False,
        )
    with pytest.raises(ValueError, match="grpmn must be a 2D array"):
        mode_matrix_matvec_from_grpmn_jax(
            rhs,
            jnp.zeros(4),
            sin_basis=sin_basis,
            xmpot=xmpot,
            n_raw=n_raw,
            lasym=False,
        )
    with pytest.raises(ValueError, match="invalid_grpmn_shape_lasym"):
        mode_matrix_matvec_from_grpmn_jax(
            jnp.ones(2 * sin_basis.shape[1]),
            grpmn,
            sin_basis=sin_basis,
            xmpot=xmpot,
            n_raw=n_raw,
            lasym=True,
            cos_basis=sin_basis,
        )
    grpmn_lasym = jnp.zeros((2 * sin_basis.shape[1], sin_basis.shape[0]), dtype=float)
    with pytest.raises(ValueError, match="cos_basis is required"):
        mode_matrix_matvec_from_grpmn_jax(
            jnp.ones(2 * sin_basis.shape[1]),
            grpmn_lasym,
            sin_basis=sin_basis,
            xmpot=xmpot,
            n_raw=n_raw,
            lasym=True,
        )
    with pytest.raises(ValueError, match="cos_basis must match"):
        mode_matrix_matvec_from_grpmn_jax(
            jnp.ones(2 * sin_basis.shape[1]),
            grpmn_lasym,
            sin_basis=sin_basis,
            xmpot=xmpot,
            n_raw=n_raw,
            lasym=True,
            cos_basis=sin_basis[:, :2],
        )
    with pytest.raises(ValueError, match="vector size"):
        mode_matrix_matvec_from_grpmn_jax(
            jnp.ones(4),
            grpmn,
            sin_basis=sin_basis,
            xmpot=xmpot,
            n_raw=n_raw,
            lasym=False,
        )
    with pytest.raises(ValueError, match="invalid_grpmn_shape"):
        mode_matrix_matvec_from_grpmn_jax(
            rhs,
            jnp.zeros((2, 4)),
            sin_basis=sin_basis,
            xmpot=xmpot,
            n_raw=n_raw,
            lasym=False,
        )
    with pytest.raises(ValueError, match="cos_basis is required"):
        mode_matrix_matvec_from_grpmn_jax(
            jnp.ones(6),
            jnp.zeros((6, 4)),
            sin_basis=sin_basis,
            xmpot=xmpot,
            n_raw=n_raw,
            lasym=True,
        )
    with pytest.raises(ValueError, match="cos_basis must match"):
        mode_matrix_matvec_from_grpmn_jax(
            jnp.ones(6),
            jnp.zeros((6, 4)),
            sin_basis=sin_basis,
            cos_basis=sin_basis[:, :2],
            xmpot=xmpot,
            n_raw=n_raw,
            lasym=True,
        )


@pytest.mark.py311_slow_coverage
def test_dense_vmec_nestor_mode_solve_gradients_match_finite_difference():
    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp

    enable_x64(True)
    basis, sample = _nonsingular_boundary_sample()
    base_bex = jnp.asarray(np.linspace(-0.11, 0.29, int(basis["nuv3"]), dtype=float))
    bex_direction = jnp.asarray(np.cos(0.23 + np.arange(int(basis["nuv3"]), dtype=float)))
    phi_weights = jnp.asarray(np.sin(0.27 + np.arange(int(basis["nuv3"]), dtype=float)), dtype=float)
    tables = _ensure_vmec_nonsingular_kernel_tables(basis=basis, nv=sample.R.shape[1], nvper=2)

    def response(source_scale, geometry_scale):
        out = dense_vmec_nestor_mode_solve_jax(
            R=jnp.asarray(sample.R) + 0.015 * geometry_scale,
            Z=sample.Z,
            Ru=sample.Ru,
            Zu=sample.Zu,
            Rv=sample.Rv,
            Zv=sample.Zv,
            ruu=sample.ruu,
            ruv=sample.ruv,
            rvv=sample.rvv,
            zuu=sample.zuu,
            zuv=sample.zuv,
            zvv=sample.zvv,
            bexni=base_bex + source_scale * bex_direction,
            basis=basis,
            tables=tables,
            signgs=1,
            nvper=2,
        )
        return 0.5 * jnp.vdot(out["mode_coeffs"], out["mode_coeffs"]) + 0.1 * jnp.vdot(
            phi_weights,
            out["phi_flat"],
        )

    exact_source = jax.grad(lambda scale: response(scale, 0.0))(0.0)
    exact_geometry = jax.grad(lambda scale: response(0.0, scale))(0.0)
    eps = 1.0e-6
    fd_source = (response(eps, 0.0) - response(-eps, 0.0)) / (2.0 * eps)
    fd_geometry = (response(0.0, eps) - response(0.0, -eps)) / (2.0 * eps)

    np.testing.assert_allclose(exact_source, fd_source, rtol=3.0e-6, atol=1.0e-10)
    np.testing.assert_allclose(exact_geometry, fd_geometry, rtol=3.0e-6, atol=1.0e-10)


def test_free_boundary_adjoint_operator_validation_errors_are_explicit():
    """Guard the public validation contract of the JAX NESTOR operator blocks."""

    basis, sample = _nonsingular_boundary_sample()
    tables = _ensure_vmec_nonsingular_kernel_tables(basis=basis, nv=sample.R.shape[1], nvper=2)
    bexni = np.linspace(-0.18, 0.24, int(basis["nuv3"]), dtype=float)

    with pytest.raises(ValueError, match="square dense matrix"):
        dense_vacuum_solve_jax(np.ones((2, 3)), np.ones(2))
    with pytest.raises(ValueError, match="leading dimension"):
        dense_vacuum_solve_jax(np.eye(2), np.ones(3))
    with pytest.raises(ValueError, match="requires imirr"):
        vmec_source_from_gsource_jax(np.ones(4), onp=1.0, lasym=False)

    with pytest.raises(ValueError, match="sin_basis must be a 2D array"):
        mode_rhs_from_gsource_jax(np.ones(3), sin_basis=np.ones(3), xmpot=np.arange(3), n_raw=np.arange(3), onp=1.0, lasym=True)
    with pytest.raises(ValueError, match="cos_basis is required"):
        mode_rhs_from_gsource_jax(
            np.ones(3),
            sin_basis=np.ones((3, 2)),
            xmpot=np.arange(2),
            n_raw=np.arange(2),
            onp=1.0,
            lasym=True,
        )
    with pytest.raises(ValueError, match="cos_basis must match"):
        mode_rhs_from_gsource_jax(
            np.ones(3),
            sin_basis=np.ones((3, 2)),
            cos_basis=np.ones((3, 1)),
            xmpot=np.arange(2),
            n_raw=np.arange(2),
            onp=1.0,
            lasym=True,
        )

    with pytest.raises(ValueError, match="grpmn must be a 2D array"):
        mode_matrix_from_grpmn_jax(np.ones(4), sin_basis=np.ones((3, 2)), xmpot=np.arange(2), n_raw=np.arange(2), lasym=False)
    with pytest.raises(ValueError, match="sin_basis must be a 2D array"):
        mode_matrix_from_grpmn_jax(np.ones((2, 3)), sin_basis=np.ones(2), xmpot=np.arange(2), n_raw=np.arange(2), lasym=False)
    with pytest.raises(ValueError, match="invalid_grpmn_shape"):
        mode_matrix_from_grpmn_jax(np.ones((1, 3)), sin_basis=np.ones((3, 2)), xmpot=np.arange(2), n_raw=np.arange(2), lasym=False)
    with pytest.raises(ValueError, match="invalid_grpmn_shape_lasym"):
        mode_matrix_from_grpmn_jax(np.ones((2, 3)), sin_basis=np.ones((3, 2)), xmpot=np.arange(2), n_raw=np.arange(2), lasym=True, cos_basis=np.ones((3, 2)))
    with pytest.raises(ValueError, match="cos_basis is required"):
        mode_matrix_from_grpmn_jax(np.ones((4, 3)), sin_basis=np.ones((3, 2)), xmpot=np.arange(2), n_raw=np.arange(2), lasym=True)
    with pytest.raises(ValueError, match="cos_basis must match"):
        mode_matrix_from_grpmn_jax(np.ones((4, 3)), sin_basis=np.ones((3, 2)), cos_basis=np.ones((3, 1)), xmpot=np.arange(2), n_raw=np.arange(2), lasym=True)

    with pytest.raises(ValueError, match="R must be a 2D"):
        vmec_nonsingular_terms_from_bexni_jax(
            R=np.ones(3),
            Z=sample.Z,
            Ru=sample.Ru,
            Zu=sample.Zu,
            Rv=sample.Rv,
            Zv=sample.Zv,
            ruu=sample.ruu,
            ruv=sample.ruv,
            rvv=sample.rvv,
            zuu=sample.zuu,
            zuv=sample.zuv,
            zvv=sample.zvv,
            bexni=bexni,
            basis=basis,
            tables=tables,
            signgs=1,
            nvper=2,
        )
    with pytest.raises(ValueError, match="Z must match R shape"):
        vmec_nonsingular_terms_from_bexni_jax(
            R=sample.R,
            Z=sample.Z[:, :-1],
            Ru=sample.Ru,
            Zu=sample.Zu,
            Rv=sample.Rv,
            Zv=sample.Zv,
            ruu=sample.ruu,
            ruv=sample.ruv,
            rvv=sample.rvv,
            zuu=sample.zuu,
            zuv=sample.zuv,
            zvv=sample.zvv,
            bexni=bexni,
            basis=basis,
            tables=tables,
            signgs=1,
            nvper=2,
        )
    bad_basis = dict(basis)
    bad_basis["nu_full"] = int(basis["nu_full"]) + 1
    with pytest.raises(ValueError, match="nu_full"):
        _jax_nonsingular_terms(sample, bad_basis, bexni)
    bad_tables = dict(tables)
    bad_tables["cosui"] = np.empty((1, 0), dtype=float)
    with pytest.raises(ValueError, match="table shape"):
        vmec_nonsingular_terms_from_bexni_jax(
            R=sample.R,
            Z=sample.Z,
            Ru=sample.Ru,
            Zu=sample.Zu,
            Rv=sample.Rv,
            Zv=sample.Zv,
            ruu=sample.ruu,
            ruv=sample.ruv,
            rvv=sample.rvv,
            zuu=sample.zuu,
            zuv=sample.zuv,
            zvv=sample.zvv,
            bexni=bexni,
            basis=basis,
            tables=bad_tables,
            signgs=1,
            nvper=2,
        )
    with pytest.raises(ValueError, match="bexni"):
        _jax_nonsingular_terms(sample, basis, bexni[:2])

    with pytest.raises(ValueError, match="sin_basis must be a 2D array"):
        dense_mode_vacuum_solve_jax(np.eye(2), np.ones(2), np.ones(2))
    with pytest.raises(ValueError, match="must match sin_basis columns"):
        dense_mode_vacuum_solve_jax(np.eye(2), np.ones(2), np.ones((3, 3)))
    with pytest.raises(ValueError, match="cos_basis must match"):
        dense_mode_vacuum_solve_jax(np.eye(4), np.ones(4), np.ones((3, 2)), np.ones((3, 1)))
    with pytest.raises(ValueError, match="2 \\* sin_basis columns"):
        dense_mode_vacuum_solve_jax(np.eye(3), np.ones(3), np.ones((3, 2)), np.ones((3, 2)))


def _mode_vacuum_inputs(*, lasym: bool = False):
    from vmec_jax._compat import jnp

    sin_basis = jnp.asarray(
        [
            [0.0, 0.2, -0.3],
            [0.4, -0.1, 0.5],
            [-0.2, 0.6, 0.1],
            [0.7, 0.3, -0.4],
        ],
        dtype=float,
    )
    mode_matrix = jnp.asarray(
        [
            [3.0, 0.2, -0.1],
            [0.4, 2.6, 0.3],
            [-0.2, 0.1, 2.4],
        ],
        dtype=float,
    )
    rhs = jnp.asarray([0.5, -0.2, 0.4], dtype=float)
    if not lasym:
        return mode_matrix, rhs, sin_basis, None

    cos_basis = jnp.asarray(
        [
            [0.5, -0.3, 0.1],
            [-0.2, 0.4, -0.6],
            [0.3, 0.2, 0.7],
            [-0.1, -0.5, 0.2],
        ],
        dtype=float,
    )
    top = jnp.concatenate([mode_matrix + 0.8 * jnp.eye(3), 0.1 * jnp.eye(3)], axis=1)
    bottom = jnp.concatenate([-0.05 * jnp.eye(3), mode_matrix + 1.1 * jnp.eye(3)], axis=1)
    return jnp.concatenate([top, bottom], axis=0), jnp.concatenate([rhs, -0.3 * rhs]), sin_basis, cos_basis


def test_dense_mode_vacuum_solve_reconstructs_grid_potential():
    from vmec_jax._compat import jnp

    enable_x64(True)
    A, rhs, sin_basis, _cos_basis = _mode_vacuum_inputs()

    actual = dense_mode_vacuum_solve_jax(A, rhs, sin_basis)
    coeffs = jnp.linalg.solve(A, rhs)

    np.testing.assert_allclose(actual["mode_coeffs"], coeffs, rtol=1.0e-14, atol=1.0e-14)
    np.testing.assert_allclose(actual["phi_flat"], sin_basis @ coeffs, rtol=1.0e-14, atol=1.0e-14)
    np.testing.assert_allclose(actual["residual"], np.zeros_like(np.asarray(rhs)), atol=1.0e-14)


def test_dense_mode_vacuum_solve_can_skip_residual_diagnostics():
    from vmec_jax._compat import jnp

    enable_x64(True)
    A, rhs, sin_basis, _cos_basis = _mode_vacuum_inputs()

    actual = dense_mode_vacuum_solve_jax(A, rhs, sin_basis, include_residual=False)
    coeffs = jnp.linalg.solve(A, rhs)

    np.testing.assert_allclose(actual["mode_coeffs"], coeffs, rtol=1.0e-14, atol=1.0e-14)
    np.testing.assert_allclose(actual["phi_flat"], sin_basis @ coeffs, rtol=1.0e-14, atol=1.0e-14)
    assert "residual" not in actual


def test_dense_mode_vacuum_solve_can_skip_grid_potential_reconstruction():
    from vmec_jax._compat import jnp

    enable_x64(True)
    A, rhs, sin_basis, _cos_basis = _mode_vacuum_inputs()

    actual = dense_mode_vacuum_solve_jax(A, rhs, sin_basis, include_phi_flat=False, include_residual=False)
    coeffs = jnp.linalg.solve(A, rhs)

    np.testing.assert_allclose(actual["mode_coeffs"], coeffs, rtol=1.0e-14, atol=1.0e-14)
    assert "phi_flat" not in actual
    assert "residual" not in actual


def test_dense_mode_vacuum_solve_reconstructs_lasym_grid_potential():
    from vmec_jax._compat import jnp

    enable_x64(True)
    A, rhs, sin_basis, cos_basis = _mode_vacuum_inputs(lasym=True)

    actual = dense_mode_vacuum_solve_jax(A, rhs, sin_basis, cos_basis)
    coeffs = jnp.linalg.solve(A, rhs)
    nmodes = sin_basis.shape[1]

    np.testing.assert_allclose(actual["mode_coeffs"], coeffs, rtol=1.0e-14, atol=1.0e-14)
    np.testing.assert_allclose(
        actual["phi_flat"],
        sin_basis @ coeffs[:nmodes] + cos_basis @ coeffs[nmodes:],
        rtol=1.0e-14,
        atol=1.0e-14,
    )


def test_dense_mode_vacuum_gradient_wrt_rhs_matches_finite_difference():
    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp

    enable_x64(True)
    A, rhs, sin_basis, _cos_basis = _mode_vacuum_inputs()
    direction = jnp.asarray([0.2, -0.3, 0.1], dtype=float)
    weights = jnp.asarray([0.7, -0.2, 0.4, 0.1], dtype=float)

    def objective(scale):
        response = dense_mode_vacuum_solve_jax(A, rhs + scale * direction, sin_basis)
        return jnp.vdot(weights, response["phi_flat"])

    exact = jax.grad(objective)(0.0)
    eps = 1.0e-6
    fd = (objective(eps) - objective(-eps)) / (2.0 * eps)

    np.testing.assert_allclose(exact, fd, rtol=2.0e-9, atol=1.0e-11)


def test_dense_mode_vacuum_gradient_wrt_matrix_matches_finite_difference():
    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp

    enable_x64(True)
    A, rhs, sin_basis, _cos_basis = _mode_vacuum_inputs()
    dA = jnp.asarray(
        [
            [0.0, 0.1, -0.2],
            [0.05, 0.0, 0.1],
            [-0.1, 0.2, 0.0],
        ],
        dtype=float,
    )
    weights = jnp.asarray([0.7, -0.2, 0.4, 0.1], dtype=float)

    def objective(scale):
        response = dense_mode_vacuum_solve_jax(A + scale * dA, rhs, sin_basis)
        return jnp.vdot(weights, response["phi_flat"])

    exact = jax.grad(objective)(0.0)
    eps = 1.0e-6
    fd = (objective(eps) - objective(-eps)) / (2.0 * eps)

    np.testing.assert_allclose(exact, fd, rtol=3.0e-9, atol=1.0e-11)


def _toy_coil_vacuum_response(*, current_scale: float = 0.0, radius_shift: float = 0.0):
    """Small direct-coil -> vacuum-linear-solve chain for adjoint checks."""

    from vmec_jax._compat import jnp

    radius = 1.15 + 0.02 * radius_shift
    dofs = jnp.zeros((1, 3, 3), dtype=float)
    dofs = dofs.at[0, 0, 2].set(radius)
    dofs = dofs.at[0, 1, 1].set(radius)
    params = CoilFieldParams(
        base_curve_dofs=dofs,
        base_currents=jnp.asarray([3.0e7 * (1.0 + 0.01 * current_scale)], dtype=float),
        n_segments=96,
        regularization_epsilon=1.0e-9,
    )
    R = jnp.asarray([0.24, 0.37, 0.51], dtype=float)
    Z = jnp.asarray([0.11, -0.17, 0.23], dtype=float)
    phi = jnp.asarray([0.0, 0.4, 0.9], dtype=float)
    br, bphi, bz = sample_coil_field_cylindrical(params, R, Z, phi)
    rhs = jnp.stack(
        (
            br[0] + 0.3 * bphi[1],
            bz[1] - 0.2 * br[2],
            bphi[2] + 0.5 * bz[0],
        )
    )
    A = jnp.asarray(
        [
            [2.7, 0.2, -0.1],
            [0.1, 2.2, 0.3],
            [-0.2, 0.4, 2.5],
        ],
        dtype=float,
    )
    x = dense_vacuum_solve_jax(A, rhs)
    return 0.5 * jnp.vdot(x, x) + 0.1 * jnp.vdot(rhs, rhs)


def _toy_coil_nonlinear_response(*, current_scale: float = 0.0, radius_shift: float = 0.0):
    """Small direct-coil -> nonlinear implicit-root chain for phase-2 checks."""

    from vmec_jax._compat import jnp

    radius = 1.25 + 0.03 * radius_shift
    dofs = jnp.zeros((1, 3, 3), dtype=float)
    dofs = dofs.at[0, 0, 2].set(radius)
    dofs = dofs.at[0, 1, 1].set(radius)
    params = CoilFieldParams(
        base_curve_dofs=dofs,
        base_currents=jnp.asarray([6.0e6 * (1.0 + 0.02 * current_scale)], dtype=float),
        n_segments=96,
        regularization_epsilon=1.0e-9,
    )
    R = jnp.asarray([0.32, 0.47, 0.61], dtype=float)
    Z = jnp.asarray([0.09, -0.16, 0.19], dtype=float)
    phi = jnp.asarray([0.0, 0.35, 0.8], dtype=float)
    br, bphi, bz = sample_coil_field_cylindrical(params, R, Z, phi)
    rhs = 0.04 * jnp.stack(
        (
            br[0] + 0.2 * bphi[1] - 0.1 * bz[2],
            bz[1] - 0.3 * br[2] + 0.1 * bphi[0],
        )
    )

    def residual(x, residual_params):
        local_rhs = residual_params["rhs"]
        return jnp.asarray(
            [
                x[0] + 0.1 * x[0] ** 3 + 0.08 * x[1] - local_rhs[0],
                0.9 * x[1] + 0.05 * x[0] ** 2 + 0.15 * jnp.sin(x[1]) - local_rhs[1],
            ]
        )

    root = dense_nonlinear_solve_jax(
        residual,
        jnp.asarray([0.0, 0.0], dtype=float),
        {"rhs": rhs},
        max_iter=14,
    )
    weights = jnp.asarray([0.9, 1.3], dtype=float)
    return 0.5 * jnp.vdot(weights * root, root) + 0.02 * jnp.vdot(rhs, rhs)


def _toy_coil_free_boundary_fixed_point_response(
    *,
    current_scale: float = 0.0,
    radius_shift: float = 0.0,
):
    """Direct coils -> state-dependent boundary -> vacuum response fixed point."""

    from vmec_jax._compat import jnp

    radius = 1.28 + 0.02 * radius_shift
    dofs = jnp.zeros((1, 3, 3), dtype=float)
    dofs = dofs.at[0, 0, 2].set(radius)
    dofs = dofs.at[0, 1, 1].set(radius)
    coil_params = CoilFieldParams(
        base_curve_dofs=dofs,
        base_currents=jnp.asarray([4.5e6 * (1.0 + 0.015 * current_scale)], dtype=float),
        n_segments=64,
        regularization_epsilon=1.0e-9,
    )
    theta_shape = (2, 2)
    phi = jnp.asarray([[0.0, 0.4], [0.8, 1.2]], dtype=float)
    Ru = jnp.asarray([[0.02, -0.03], [0.04, -0.02]], dtype=float)
    Zu = jnp.asarray([[0.19, 0.21], [0.18, 0.20]], dtype=float)
    Rv = jnp.asarray([[0.03, 0.01], [-0.02, 0.04]], dtype=float)
    Zv = jnp.asarray([[0.02, -0.01], [0.03, -0.02]], dtype=float)
    weights = jnp.asarray([[0.4, -0.2], [0.6, -0.3]], dtype=float)
    A = jnp.asarray([[2.6, 0.18], [-0.12, 2.4]], dtype=float)

    def update(state, params):
        R = jnp.asarray([[0.72, 0.81], [0.86, 0.76]], dtype=float) + 0.04 * state[0]
        Z = jnp.asarray([[0.11, -0.12], [0.17, -0.15]], dtype=float) + 0.03 * state[1]
        br, bphi, bz = sample_coil_field_cylindrical(params["coil"], R, Z, phi)
        vac = vacuum_boundary_fields_from_cylindrical_jax(
            br=br,
            bp=bphi,
            bz=bz,
            R=R,
            Ru=Ru,
            Zu=Zu,
            Rv=Rv,
            Zv=Zv,
        )
        rhs = 0.03 * jnp.asarray(
            [
                jnp.mean(vac["bnormal_unit"] * weights),
                jnp.mean((vac["bsqvac"] - jnp.mean(vac["bsqvac"])) * weights),
            ]
        )
        response = dense_vacuum_solve_jax(A, rhs)
        update_state = jnp.asarray([0.03, -0.02], dtype=float) + 0.45 * jnp.tanh(response)
        if update_state.shape != (2,):  # pragma: no cover - defensive shape guard.
            raise AssertionError(f"unexpected update shape {update_state.shape} for boundary grid {theta_shape}")
        return update_state

    root = dense_fixed_point_solve_jax(
        update,
        jnp.asarray([0.0, 0.0], dtype=float),
        {"coil": coil_params},
        max_iter=12,
    )
    final = update(root, {"coil": coil_params})
    return 0.5 * jnp.vdot(jnp.asarray([1.1, 0.9], dtype=float) * root, root) + 0.1 * jnp.vdot(final, final)


def _toy_coil_projected_mode_fixed_point_response(
    *,
    current_scale: float = 0.0,
    radius_shift: float = 0.0,
):
    """Direct coils -> moving boundary -> mode vacuum response -> fixed point."""

    from vmec_jax._compat import jnp

    radius = 1.34 + 0.025 * radius_shift
    dofs = jnp.zeros((1, 3, 3), dtype=float)
    dofs = dofs.at[0, 0, 2].set(radius)
    dofs = dofs.at[0, 1, 1].set(radius)
    coil_params = CoilFieldParams(
        base_curve_dofs=dofs,
        base_currents=jnp.asarray([5.5e6 * (1.0 + 0.012 * current_scale)], dtype=float),
        n_segments=64,
        regularization_epsilon=1.0e-9,
    )
    phi = jnp.asarray([[0.05, 0.45], [0.85, 1.25]], dtype=float)
    Ru_base = jnp.asarray([[0.03, -0.04], [0.02, 0.05]], dtype=float)
    Zu_base = jnp.asarray([[0.20, 0.22], [0.19, 0.21]], dtype=float)
    Rv_base = jnp.asarray([[0.04, 0.01], [-0.03, 0.05]], dtype=float)
    Zv_base = jnp.asarray([[0.02, -0.03], [0.06, -0.01]], dtype=float)
    sin_basis = jnp.asarray(
        [
            [0.0, 0.2, -0.3],
            [0.4, -0.1, 0.5],
            [-0.2, 0.6, 0.1],
            [0.7, 0.3, -0.4],
        ],
        dtype=float,
    )
    mode_matrix = jnp.asarray(
        [
            [3.2, 0.12, -0.06],
            [0.16, 2.7, 0.21],
            [-0.09, 0.24, 2.9],
        ],
        dtype=float,
    )
    mode_to_state = jnp.asarray(
        [
            [0.09, -0.04, 0.03],
            [-0.02, 0.08, 0.05],
            [0.04, 0.03, -0.07],
        ],
        dtype=float,
    )

    def boundary_from_state(state):
        base_R = jnp.asarray([[0.74, 0.83], [0.89, 0.78]], dtype=float)
        base_Z = jnp.asarray([[0.10, -0.13], [0.18, -0.16]], dtype=float)
        shape = jnp.asarray([[0.25, -0.15], [0.35, -0.20]], dtype=float)
        R = base_R + 0.035 * state[0] + 0.018 * state[2] * shape
        Z = base_Z + 0.028 * state[1] - 0.012 * state[2] * shape
        return {
            "R": R,
            "Z": Z,
            "phi": phi,
            "Ru": Ru_base + 0.01 * state[2],
            "Zu": Zu_base - 0.008 * state[2],
            "Rv": Rv_base,
            "Zv": Zv_base,
        }

    def update_from_response(_state, response, _vac, _boundary, _params):
        mode_coeffs = jnp.asarray(response["mode_coeffs"])
        pressure_like = jnp.asarray([0.02, -0.015, 0.01], dtype=float)
        return pressure_like + 0.22 * jnp.tanh(mode_to_state @ mode_coeffs)

    solved = direct_coil_projected_mode_fixed_point_objective_jax(
        coil_params,
        jnp.asarray([0.0, 0.0, 0.0], dtype=float),
        boundary_from_state=boundary_from_state,
        update_from_response=update_from_response,
        mode_matrix=mode_matrix,
        sin_basis=sin_basis,
        xmpot=jnp.asarray([0, 1, 1]),
        n_raw=jnp.asarray([0, 0, 1]),
        imirr=jnp.asarray([1, 0, 3, 2]),
        nuv3=4,
        nuv_full=4,
        max_iter=14,
        state_weights=jnp.asarray([1.0, 0.8, 1.2], dtype=float),
        update_weights=0.08,
        mode_weights=0.02,
        fixed_point_residual_weight=10.0,
    )
    return solved["objective"]


def _assert_scalar_directional_check(objective, *, eps, rtol, min_abs):
    """Run the reusable AD-vs-FD gate for a scalar control parameter."""

    from vmec_jax._compat import jnp

    check = pytree_directional_derivative_check_jax(
        objective,
        jnp.asarray(0.0, dtype=float),
        jnp.asarray(1.0, dtype=float),
        eps=eps,
    )
    assert abs(float(check["exact_directional"])) > min_abs
    np.testing.assert_allclose(check["exact_directional"], check["fd_directional"], rtol=rtol, atol=1.0e-10)


def test_dense_vacuum_adjoint_chain_wrt_coil_current_matches_finite_difference():
    """Validate a direct-coil field feeding an implicit vacuum solve."""

    pytest.importorskip("jax")

    enable_x64(True)
    _assert_scalar_directional_check(
        lambda scale: _toy_coil_vacuum_response(current_scale=scale),
        eps=1.0e-4,
        rtol=2.0e-6,
        min_abs=1.0e-8,
    )


def test_dense_vacuum_adjoint_chain_wrt_coil_geometry_matches_finite_difference():
    """Validate the same chain for a Fourier curve coefficient perturbation."""

    pytest.importorskip("jax")

    enable_x64(True)
    _assert_scalar_directional_check(
        lambda shift: _toy_coil_vacuum_response(radius_shift=shift),
        eps=1.0e-4,
        rtol=2.0e-6,
        min_abs=1.0e-8,
    )


def test_dense_nonlinear_adjoint_chain_wrt_coil_current_matches_finite_difference():
    """Validate direct-coil controls through a nonlinear implicit root."""

    pytest.importorskip("jax")

    enable_x64(True)
    _assert_scalar_directional_check(
        lambda scale: _toy_coil_nonlinear_response(current_scale=scale),
        eps=1.0e-4,
        rtol=5.0e-6,
        min_abs=1.0e-8,
    )


@pytest.mark.py311_coverage_only
def test_dense_nonlinear_adjoint_chain_wrt_coil_geometry_matches_finite_difference():
    """Validate a coil Fourier perturbation through a nonlinear implicit root."""

    pytest.importorskip("jax")

    enable_x64(True)
    _assert_scalar_directional_check(
        lambda shift: _toy_coil_nonlinear_response(radius_shift=shift),
        eps=1.0e-4,
        rtol=5.0e-6,
        min_abs=1.0e-8,
    )


@pytest.mark.py311_coverage_only
def test_dense_fixed_point_direct_coil_loop_wrt_current_matches_finite_difference():
    """Validate a miniature complete free-boundary fixed-point coil loop."""

    pytest.importorskip("jax")

    enable_x64(True)
    _assert_scalar_directional_check(
        lambda scale: _toy_coil_free_boundary_fixed_point_response(current_scale=scale),
        eps=1.0e-4,
        rtol=8.0e-6,
        min_abs=1.0e-9,
    )


@pytest.mark.py311_coverage_only
def test_dense_fixed_point_direct_coil_loop_wrt_geometry_matches_finite_difference():
    """Validate the fixed-point loop for one coil Fourier geometry coefficient."""

    pytest.importorskip("jax")

    enable_x64(True)
    _assert_scalar_directional_check(
        lambda shift: _toy_coil_free_boundary_fixed_point_response(radius_shift=shift),
        eps=1.0e-4,
        rtol=8.0e-6,
        min_abs=1.0e-9,
    )


@pytest.mark.py311_coverage_only
def test_dense_fixed_point_projected_mode_loop_wrt_current_matches_finite_difference():
    """Validate moving-boundary direct-coil fixed point through mode response."""

    pytest.importorskip("jax")

    enable_x64(True)
    _assert_scalar_directional_check(
        lambda scale: _toy_coil_projected_mode_fixed_point_response(current_scale=scale),
        eps=1.0e-4,
        rtol=1.0e-5,
        min_abs=1.0e-9,
    )


@pytest.mark.py311_coverage_only
def test_dense_fixed_point_projected_mode_loop_wrt_geometry_matches_finite_difference():
    """Validate moving-boundary fixed point for one coil Fourier coefficient."""

    pytest.importorskip("jax")

    enable_x64(True)
    _assert_scalar_directional_check(
        lambda shift: _toy_coil_projected_mode_fixed_point_response(radius_shift=shift),
        eps=1.0e-4,
        rtol=1.0e-5,
        min_abs=1.0e-9,
    )


@pytest.mark.py311_slow_coverage
def test_projected_mode_fixed_point_objective_exposes_components():
    """Check the scalar objective wrapper returns usable diagnostics."""

    from vmec_jax._compat import jnp

    enable_x64(True)
    value = _toy_coil_projected_mode_fixed_point_response()
    assert np.isfinite(float(value))
    assert float(value) > 0.0

    radius = 1.34
    dofs = jnp.zeros((1, 3, 3), dtype=float)
    dofs = dofs.at[0, 0, 2].set(radius)
    dofs = dofs.at[0, 1, 1].set(radius)
    coil_params = CoilFieldParams(
        base_curve_dofs=dofs,
        base_currents=jnp.asarray([5.5e6], dtype=float),
        n_segments=32,
        regularization_epsilon=1.0e-9,
    )
    phi = jnp.asarray([[0.05, 0.45], [0.85, 1.25]], dtype=float)
    sin_basis = jnp.asarray(
        [
            [0.0, 0.2, -0.3],
            [0.4, -0.1, 0.5],
            [-0.2, 0.6, 0.1],
            [0.7, 0.3, -0.4],
        ],
        dtype=float,
    )

    def boundary_from_state(state):
        return {
            "R": jnp.asarray([[0.74, 0.83], [0.89, 0.78]], dtype=float) + 0.02 * state[0],
            "Z": jnp.asarray([[0.10, -0.13], [0.18, -0.16]], dtype=float) + 0.02 * state[1],
            "phi": phi,
            "Ru": jnp.asarray([[0.03, -0.04], [0.02, 0.05]], dtype=float),
            "Zu": jnp.asarray([[0.20, 0.22], [0.19, 0.21]], dtype=float),
            "Rv": jnp.asarray([[0.04, 0.01], [-0.03, 0.05]], dtype=float),
            "Zv": jnp.asarray([[0.02, -0.03], [0.06, -0.01]], dtype=float),
        }

    def update_from_response(_state, response, _vac, _boundary, _params):
        return 0.1 * jnp.tanh(jnp.asarray(response["mode_coeffs"])[:2])

    solved = direct_coil_projected_mode_fixed_point_objective_jax(
        coil_params,
        jnp.asarray([0.0, 0.0], dtype=float),
        boundary_from_state=boundary_from_state,
        update_from_response=update_from_response,
        mode_matrix=jnp.asarray([[3.2, 0.12, -0.06], [0.16, 2.7, 0.21], [-0.09, 0.24, 2.9]]),
        sin_basis=sin_basis,
        xmpot=jnp.asarray([0, 1, 1]),
        n_raw=jnp.asarray([0, 0, 1]),
        imirr=jnp.asarray([1, 0, 3, 2]),
        nuv3=4,
        nuv_full=4,
        max_iter=10,
        state_weights=1.0,
        mode_weights=0.01,
    )

    assert {"state", "mode", "fixed_point_residual"}.issubset(solved["objective_components"])
    np.testing.assert_allclose(solved["fixed_point_residual"], np.zeros(2), atol=1.0e-11)
    assert float(solved["objective"]) >= float(solved["objective_components"]["state"])


@pytest.mark.py311_slow_coverage
def test_projected_mode_fixed_point_objective_value_and_grad_wrt_coil_pytree():
    """Validate the optimizer-facing scalar objective has coil pytree gradients."""

    pytest.importorskip("jax")
    from vmec_jax._compat import jnp

    enable_x64(True)
    radius = 1.34
    dofs = jnp.zeros((1, 3, 3), dtype=float)
    dofs = dofs.at[0, 0, 2].set(radius)
    dofs = dofs.at[0, 1, 1].set(radius)
    coil_params = CoilFieldParams(
        base_curve_dofs=dofs,
        base_currents=jnp.asarray([5.5e6], dtype=float),
        n_segments=32,
        regularization_epsilon=1.0e-9,
    )
    phi = jnp.asarray([[0.05, 0.45], [0.85, 1.25]], dtype=float)
    sin_basis = jnp.asarray(
        [
            [0.0, 0.2, -0.3],
            [0.4, -0.1, 0.5],
            [-0.2, 0.6, 0.1],
            [0.7, 0.3, -0.4],
        ],
        dtype=float,
    )
    mode_matrix = jnp.asarray(
        [[3.2, 0.12, -0.06], [0.16, 2.7, 0.21], [-0.09, 0.24, 2.9]],
        dtype=float,
    )
    mode_to_state = jnp.asarray([[0.08, -0.04, 0.02], [-0.03, 0.06, 0.05]], dtype=float)

    def boundary_from_state(state):
        shape = jnp.asarray([[0.25, -0.15], [0.35, -0.20]], dtype=float)
        return {
            "R": jnp.asarray([[0.74, 0.83], [0.89, 0.78]], dtype=float)
            + 0.02 * state[0]
            + 0.01 * state[1] * shape,
            "Z": jnp.asarray([[0.10, -0.13], [0.18, -0.16]], dtype=float)
            + 0.025 * state[1],
            "phi": phi,
            "Ru": jnp.asarray([[0.03, -0.04], [0.02, 0.05]], dtype=float),
            "Zu": jnp.asarray([[0.20, 0.22], [0.19, 0.21]], dtype=float),
            "Rv": jnp.asarray([[0.04, 0.01], [-0.03, 0.05]], dtype=float),
            "Zv": jnp.asarray([[0.02, -0.03], [0.06, -0.01]], dtype=float),
        }

    def update_from_response(_state, response, _vac, _boundary, _params):
        return jnp.asarray([0.01, -0.015], dtype=float) + 0.12 * jnp.tanh(
            mode_to_state @ jnp.asarray(response["mode_coeffs"])
        )

    current_direction = coil_params.base_currents * 0.01
    dofs_direction = jnp.zeros_like(coil_params.base_curve_dofs)
    dofs_direction = dofs_direction.at[0, 0, 2].set(0.02)
    dofs_direction = dofs_direction.at[0, 1, 1].set(-0.015)
    direction = coil_params.with_arrays(
        base_curve_dofs=dofs_direction,
        base_currents=current_direction,
    )
    check = direct_coil_projected_mode_fixed_point_directional_check_jax(
        coil_params,
        direction,
        jnp.asarray([0.0, 0.0], dtype=float),
        boundary_from_state=boundary_from_state,
        update_from_response=update_from_response,
        mode_matrix=mode_matrix,
        sin_basis=sin_basis,
        xmpot=jnp.asarray([0, 1, 1]),
        n_raw=jnp.asarray([0, 0, 1]),
        imirr=jnp.asarray([1, 0, 3, 2]),
        nuv3=4,
        nuv_full=4,
        max_iter=10,
        state_weights=jnp.asarray([1.0, 0.7], dtype=float),
        mode_weights=0.02,
        rhs_mode_weights=0.01,
        bnormal_weight=0.005,
        fixed_point_residual_weight=10.0,
    )
    value = check["value"]
    grad_params = check["grad"]

    assert np.isfinite(float(value))
    assert np.all(np.isfinite(np.asarray(grad_params.base_currents)))
    assert np.all(np.isfinite(np.asarray(grad_params.base_curve_dofs)))
    assert float(jnp.linalg.norm(grad_params.base_currents)) > 1.0e-18
    assert float(jnp.linalg.norm(grad_params.base_curve_dofs)) > 1.0e-10
    assert {"state", "mode", "fixed_point_residual"}.issubset(check["objective_components"])
    np.testing.assert_allclose(check["value"], check["solved"]["objective"], rtol=0.0, atol=1.0e-14)
    assert float(check["abs_error"]) < 1.0e-8
    np.testing.assert_allclose(check["exact_directional"], check["fd_directional"], rtol=2.0e-5, atol=1.0e-10)

    with pytest.raises(ValueError, match="eps"):
        direct_coil_projected_mode_fixed_point_directional_check_jax(
            coil_params,
            direction,
            jnp.asarray([0.0, 0.0], dtype=float),
            boundary_from_state=boundary_from_state,
            update_from_response=update_from_response,
            mode_matrix=mode_matrix,
            sin_basis=sin_basis,
            xmpot=jnp.asarray([0, 1, 1]),
            n_raw=jnp.asarray([0, 0, 1]),
            eps=0.0,
        )


@pytest.mark.py311_coverage_only
def test_lasym_projected_mode_fixed_point_objective_ad_matches_central_fd_for_coil_pytree():
    """Validate the asymmetric direct-coil fixed-point chain with central FD.

    This phase-2 gate exercises a JAX-visible surrogate for the production
    free-boundary dependency graph:

    direct coil dofs/current -> moving boundary samples -> boundary-normal
    source -> LASYM sine/cosine mode projection -> dense vacuum solve ->
    fixed-point update -> scalar objective.

    It intentionally stays at tiny dense scale and does not claim exact
    gradients through production ``run_free_boundary``.
    """

    pytest.importorskip("jax")
    from vmec_jax._compat import jnp

    enable_x64(True)
    radius = 1.29
    dofs = jnp.zeros((1, 3, 3), dtype=float)
    dofs = dofs.at[0, 0, 2].set(radius)
    dofs = dofs.at[0, 1, 1].set(radius)
    dofs = dofs.at[0, 2, 0].set(0.035)
    coil_params = CoilFieldParams(
        base_curve_dofs=dofs,
        base_currents=jnp.asarray([4.2e6], dtype=float),
        n_segments=32,
        regularization_epsilon=1.0e-9,
    )
    phi = jnp.asarray([[0.08, 0.47], [0.91, 1.31]], dtype=float)
    sin_basis = jnp.asarray(
        [
            [0.0, 0.24],
            [0.35, -0.16],
            [-0.22, 0.51],
            [0.62, 0.19],
        ],
        dtype=float,
    )
    cos_basis = jnp.asarray(
        [
            [0.72, -0.11],
            [0.18, 0.43],
            [-0.34, 0.27],
            [0.26, -0.49],
        ],
        dtype=float,
    )
    mode_matrix = jnp.asarray(
        [
            [3.4, 0.11, 0.07, -0.04],
            [0.09, 3.0, -0.05, 0.08],
            [0.06, -0.03, 3.2, 0.10],
            [-0.02, 0.07, 0.12, 3.5],
        ],
        dtype=float,
    )
    mode_to_state = jnp.asarray(
        [
            [0.07, -0.03, 0.04, 0.02],
            [-0.02, 0.06, 0.03, -0.04],
            [0.05, 0.02, -0.06, 0.03],
        ],
        dtype=float,
    )

    def boundary_from_state(state):
        shape = jnp.asarray([[0.20, -0.13], [0.31, -0.22]], dtype=float)
        return {
            "R": jnp.asarray([[0.73, 0.84], [0.88, 0.79]], dtype=float)
            + 0.018 * state[0]
            + 0.012 * state[2] * shape,
            "Z": jnp.asarray([[0.12, -0.14], [0.20, -0.17]], dtype=float)
            + 0.020 * state[1]
            - 0.010 * state[2] * shape,
            "phi": phi,
            "Ru": jnp.asarray([[0.025, -0.035], [0.018, 0.045]], dtype=float) + 0.004 * state[2],
            "Zu": jnp.asarray([[0.19, 0.22], [0.18, 0.21]], dtype=float) - 0.003 * state[2],
            "Rv": jnp.asarray([[0.035, 0.012], [-0.026, 0.044]], dtype=float),
            "Zv": jnp.asarray([[0.018, -0.028], [0.052, -0.014]], dtype=float),
        }

    def update_from_response(_state, response, _vac, _boundary, _params):
        return jnp.asarray([0.012, -0.017, 0.009], dtype=float) + 0.11 * jnp.tanh(
            mode_to_state @ jnp.asarray(response["mode_coeffs"])
        )

    dofs_direction = jnp.zeros_like(coil_params.base_curve_dofs)
    dofs_direction = dofs_direction.at[0, 0, 2].set(0.018)
    dofs_direction = dofs_direction.at[0, 1, 1].set(-0.014)
    dofs_direction = dofs_direction.at[0, 2, 0].set(0.006)
    direction = coil_params.with_arrays(
        base_curve_dofs=dofs_direction,
        base_currents=0.008 * coil_params.base_currents,
    )

    check = direct_coil_projected_mode_fixed_point_directional_check_jax(
        coil_params,
        direction,
        jnp.zeros(3, dtype=float),
        boundary_from_state=boundary_from_state,
        update_from_response=update_from_response,
        mode_matrix=mode_matrix,
        sin_basis=sin_basis,
        cos_basis=cos_basis,
        xmpot=jnp.asarray([0, 1]),
        n_raw=jnp.asarray([0, 1]),
        lasym=True,
        nuv3=4,
        nuv_full=4,
        max_iter=12,
        state_weights=jnp.asarray([1.0, 0.8, 1.1], dtype=float),
        update_weights=0.03,
        mode_weights=0.015,
        rhs_mode_weights=0.008,
        bnormal_weight=0.004,
        fixed_point_residual_weight=8.0,
    )
    assert np.isfinite(float(check["value"]))
    assert abs(float(check["exact_directional"])) > 1.0e-9
    np.testing.assert_allclose(check["exact_directional"], check["fd_directional"], rtol=2.5e-5, atol=1.0e-10)


def _boundary_projection_inputs():
    from vmec_jax._compat import jnp

    br = jnp.asarray([[0.11, -0.07], [0.05, 0.09]], dtype=float)
    bp = jnp.asarray([[0.31, 0.22], [-0.18, 0.14]], dtype=float)
    bz = jnp.asarray([[-0.12, 0.08], [0.16, -0.05]], dtype=float)
    R = jnp.asarray([[1.2, 1.1], [0.9, 1.05]], dtype=float)
    Ru = jnp.asarray([[0.03, -0.04], [0.02, 0.05]], dtype=float)
    Zu = jnp.asarray([[0.25, 0.23], [0.21, 0.24]], dtype=float)
    Rv = jnp.asarray([[0.07, 0.02], [-0.05, 0.04]], dtype=float)
    Zv = jnp.asarray([[0.01, -0.03], [0.06, -0.02]], dtype=float)
    return br, bp, bz, R, Ru, Zu, Rv, Zv


def test_jax_boundary_projection_matches_numpy_reference():
    enable_x64(True)
    br, bp, bz, R, Ru, Zu, Rv, Zv = _boundary_projection_inputs()

    actual = vacuum_boundary_fields_from_cylindrical_jax(
        br=br,
        bp=bp,
        bz=bz,
        R=R,
        Ru=Ru,
        Zu=Zu,
        Rv=Rv,
        Zv=Zv,
    )
    expected = vacuum_boundary_fields_from_cylindrical(
        br=np.asarray(br),
        bp=np.asarray(bp),
        bz=np.asarray(bz),
        R=np.asarray(R),
        Ru=np.asarray(Ru),
        Zu=np.asarray(Zu),
        Rv=np.asarray(Rv),
        Zv=np.asarray(Zv),
    )

    for key in ("bu", "bv", "bsupu", "bsupv", "bsqvac", "bnormal", "bnormal_unit", "det_guv"):
        np.testing.assert_allclose(actual[key], getattr(expected, key), rtol=1.0e-13, atol=1.0e-13)


def test_jax_boundary_projection_can_skip_contravariant_channels():
    enable_x64(True)
    br, bp, bz, R, Ru, Zu, Rv, Zv = _boundary_projection_inputs()

    full = vacuum_boundary_fields_from_cylindrical_jax(
        br=br,
        bp=bp,
        bz=bz,
        R=R,
        Ru=Ru,
        Zu=Zu,
        Rv=Rv,
        Zv=Zv,
        include_bnormal_unit=False,
    )
    compact = vacuum_boundary_fields_from_cylindrical_jax(
        br=br,
        bp=bp,
        bz=bz,
        R=R,
        Ru=Ru,
        Zu=Zu,
        Rv=Rv,
        Zv=Zv,
        include_bnormal_unit=False,
        include_contravariant=False,
    )

    assert set(compact) == {"bu", "bv", "bnormal", "g_uu", "g_uv", "g_vv", "det_guv"}
    for key in compact:
        np.testing.assert_allclose(compact[key], full[key], rtol=1.0e-13, atol=1.0e-13)


def test_jax_boundary_projection_gradient_wrt_field_matches_finite_difference():
    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp

    enable_x64(True)
    br, bp, bz, R, Ru, Zu, Rv, Zv = _boundary_projection_inputs()
    weights = jnp.asarray([[0.4, -0.2], [0.7, -0.5]], dtype=float)
    direction = jnp.asarray([[0.3, -0.1], [0.2, 0.5]], dtype=float)

    def objective(scale):
        vac = vacuum_boundary_fields_from_cylindrical_jax(
            br=br + scale * direction,
            bp=bp,
            bz=bz,
            R=R,
            Ru=Ru,
            Zu=Zu,
            Rv=Rv,
            Zv=Zv,
        )
        return jnp.sum(weights * vac["bsqvac"]) + 0.2 * jnp.sum(vac["bnormal_unit"] ** 2)

    exact = jax.grad(objective)(0.0)
    eps = 1.0e-6
    fd = (objective(eps) - objective(-eps)) / (2.0 * eps)

    np.testing.assert_allclose(exact, fd, rtol=5.0e-8, atol=1.0e-10)


def test_jax_boundary_projection_gradient_wrt_geometry_matches_finite_difference():
    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp

    enable_x64(True)
    br, bp, bz, R, Ru, Zu, Rv, Zv = _boundary_projection_inputs()
    weights = jnp.asarray([[0.4, -0.2], [0.7, -0.5]], dtype=float)
    direction = jnp.asarray([[0.1, 0.2], [-0.3, 0.4]], dtype=float)

    def objective(scale):
        vac = vacuum_boundary_fields_from_cylindrical_jax(
            br=br,
            bp=bp,
            bz=bz,
            R=R + scale * direction,
            Ru=Ru,
            Zu=Zu,
            Rv=Rv,
            Zv=Zv,
        )
        return jnp.sum(weights * vac["bsqvac"]) + 0.2 * jnp.sum(vac["bnormal"] ** 2)

    exact = jax.grad(objective)(0.0)
    eps = 1.0e-6
    fd = (objective(eps) - objective(-eps)) / (2.0 * eps)

    np.testing.assert_allclose(exact, fd, rtol=5.0e-8, atol=1.0e-10)


def _toy_coil_projected_vacuum_response(*, current_scale: float = 0.0, radius_shift: float = 0.0):
    """Direct coils -> boundary projection -> implicit vacuum solve."""

    from vmec_jax._compat import jnp

    radius = 1.45 + 0.03 * radius_shift
    dofs = jnp.zeros((1, 3, 3), dtype=float)
    dofs = dofs.at[0, 0, 2].set(radius)
    dofs = dofs.at[0, 1, 1].set(radius)
    params = CoilFieldParams(
        base_curve_dofs=dofs,
        base_currents=jnp.asarray([2.5e7 * (1.0 + 0.02 * current_scale)], dtype=float),
        n_segments=128,
        regularization_epsilon=1.0e-9,
    )
    R = jnp.asarray([[0.78, 0.86], [0.92, 0.81]], dtype=float)
    Z = jnp.asarray([[0.16, -0.13], [0.22, -0.19]], dtype=float)
    phi = jnp.asarray([[0.05, 0.45], [0.9, 1.25]], dtype=float)
    Ru = jnp.asarray([[0.03, -0.04], [0.02, 0.05]], dtype=float)
    Zu = jnp.asarray([[0.22, 0.24], [0.21, 0.23]], dtype=float)
    Rv = jnp.asarray([[0.04, 0.01], [-0.03, 0.05]], dtype=float)
    Zv = jnp.asarray([[0.02, -0.03], [0.06, -0.01]], dtype=float)

    br, bphi, bz = sample_coil_field_cylindrical(params, R, Z, phi)
    vac = vacuum_boundary_fields_from_cylindrical_jax(
        br=br,
        bp=bphi,
        bz=bz,
        R=R,
        Ru=Ru,
        Zu=Zu,
        Rv=Rv,
        Zv=Zv,
    )
    weights = jnp.asarray([[0.4, -0.2], [0.7, -0.5]], dtype=float)
    rhs = jnp.stack(
        (
            jnp.mean(vac["bsqvac"]),
            jnp.mean(vac["bnormal_unit"] * weights),
            jnp.mean((vac["bu"] - 0.3 * vac["bv"]) * weights),
        )
    )
    A = jnp.asarray(
        [
            [2.8, 0.15, -0.08],
            [0.2, 2.4, 0.25],
            [-0.1, 0.3, 2.6],
        ],
        dtype=float,
    )
    x = dense_vacuum_solve_jax(A, rhs)
    return 0.5 * jnp.vdot(x, x) + 0.05 * jnp.mean(vac["bnormal"] ** 2)


def _toy_coil_projected_mode_vacuum_response(*, current_scale: float = 0.0, radius_shift: float = 0.0):
    """Direct coils -> projection -> mode-space vacuum solve."""

    from vmec_jax._compat import jnp

    radius = 1.45 + 0.03 * radius_shift
    dofs = jnp.zeros((1, 3, 3), dtype=float)
    dofs = dofs.at[0, 0, 2].set(radius)
    dofs = dofs.at[0, 1, 1].set(radius)
    params = CoilFieldParams(
        base_curve_dofs=dofs,
        base_currents=jnp.asarray([2.5e7 * (1.0 + 0.02 * current_scale)], dtype=float),
        n_segments=128,
        regularization_epsilon=1.0e-9,
    )
    R = jnp.asarray([[0.78, 0.86], [0.92, 0.81]], dtype=float)
    Z = jnp.asarray([[0.16, -0.13], [0.22, -0.19]], dtype=float)
    phi = jnp.asarray([[0.05, 0.45], [0.9, 1.25]], dtype=float)
    Ru = jnp.asarray([[0.03, -0.04], [0.02, 0.05]], dtype=float)
    Zu = jnp.asarray([[0.22, 0.24], [0.21, 0.23]], dtype=float)
    Rv = jnp.asarray([[0.04, 0.01], [-0.03, 0.05]], dtype=float)
    Zv = jnp.asarray([[0.02, -0.03], [0.06, -0.01]], dtype=float)
    sin_basis = jnp.asarray(
        [
            [0.0, 0.2, -0.3],
            [0.4, -0.1, 0.5],
            [-0.2, 0.6, 0.1],
            [0.7, 0.3, -0.4],
        ],
        dtype=float,
    )
    mode_matrix = jnp.asarray(
        [
            [3.1, 0.15, -0.08],
            [0.2, 2.5, 0.25],
            [-0.1, 0.3, 2.7],
        ],
        dtype=float,
    )

    br, bphi, bz = sample_coil_field_cylindrical(params, R, Z, phi)
    vac = vacuum_boundary_fields_from_cylindrical_jax(
        br=br,
        bp=bphi,
        bz=bz,
        R=R,
        Ru=Ru,
        Zu=Zu,
        Rv=Rv,
        Zv=Zv,
    )
    rhs_mode = mode_rhs_from_gsource_jax(
        vac["bnormal"],
        sin_basis=sin_basis,
        xmpot=jnp.asarray([0, 1, 1]),
        n_raw=jnp.asarray([0, 0, 1]),
        onp=1.0,
        lasym=False,
        imirr=jnp.asarray([1, 0, 3, 2]),
        nuv3=4,
        nuv_full=4,
    )
    response = dense_mode_vacuum_solve_jax(mode_matrix, rhs_mode, sin_basis)
    weights = jnp.asarray([0.7, -0.2, 0.4, 0.1], dtype=float)
    return 0.5 * jnp.vdot(response["mode_coeffs"], response["mode_coeffs"]) + 0.1 * jnp.vdot(
        weights,
        response["phi_flat"],
    )


def test_dense_vacuum_adjoint_chain_through_projection_wrt_current_matches_finite_difference():
    """Validate the next rung in the coil-to-vacuum adjoint chain."""

    pytest.importorskip("jax")
    from vmec_jax._compat import jax

    enable_x64(True)

    exact = jax.grad(lambda scale: _toy_coil_projected_vacuum_response(current_scale=scale))(0.0)
    eps = 1.0e-4
    fd = (
        _toy_coil_projected_vacuum_response(current_scale=eps)
        - _toy_coil_projected_vacuum_response(current_scale=-eps)
    ) / (2.0 * eps)

    assert abs(float(exact)) > 1.0e-8
    np.testing.assert_allclose(exact, fd, rtol=3.0e-6, atol=1.0e-10)


def test_dense_vacuum_adjoint_chain_through_projection_wrt_geometry_matches_finite_difference():
    """Validate projected vacuum sensitivity to one coil Fourier coefficient."""

    pytest.importorskip("jax")
    from vmec_jax._compat import jax

    enable_x64(True)

    exact = jax.grad(lambda shift: _toy_coil_projected_vacuum_response(radius_shift=shift))(0.0)
    eps = 1.0e-4
    fd = (
        _toy_coil_projected_vacuum_response(radius_shift=eps)
        - _toy_coil_projected_vacuum_response(radius_shift=-eps)
    ) / (2.0 * eps)

    assert abs(float(exact)) > 1.0e-8
    np.testing.assert_allclose(exact, fd, rtol=3.0e-6, atol=1.0e-10)


def test_dense_mode_vacuum_chain_through_projection_wrt_current_matches_finite_difference():
    """Validate the mode-space scaffold in a direct-coil projected chain."""

    pytest.importorskip("jax")
    from vmec_jax._compat import jax

    enable_x64(True)

    exact = jax.grad(lambda scale: _toy_coil_projected_mode_vacuum_response(current_scale=scale))(0.0)
    eps = 1.0e-4
    fd = (
        _toy_coil_projected_mode_vacuum_response(current_scale=eps)
        - _toy_coil_projected_mode_vacuum_response(current_scale=-eps)
    ) / (2.0 * eps)

    assert abs(float(exact)) > 1.0e-8
    np.testing.assert_allclose(exact, fd, rtol=4.0e-6, atol=1.0e-10)


def test_dense_mode_vacuum_chain_through_projection_wrt_geometry_matches_finite_difference():
    """Validate the mode-space scaffold for a coil Fourier perturbation."""

    pytest.importorskip("jax")
    from vmec_jax._compat import jax

    enable_x64(True)

    exact = jax.grad(lambda shift: _toy_coil_projected_mode_vacuum_response(radius_shift=shift))(0.0)
    eps = 1.0e-4
    fd = (
        _toy_coil_projected_mode_vacuum_response(radius_shift=eps)
        - _toy_coil_projected_mode_vacuum_response(radius_shift=-eps)
    ) / (2.0 * eps)

    assert abs(float(exact)) > 1.0e-8
    np.testing.assert_allclose(exact, fd, rtol=4.0e-6, atol=1.0e-10)
