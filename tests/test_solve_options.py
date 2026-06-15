from __future__ import annotations

import math

import pytest

from vmec_jax.solvers.fixed_boundary.options import (
    validate_fixed_boundary_gd_options,
    validate_fixed_boundary_lbfgs_options,
    validate_lambda_gd_options,
    validate_pressure_shape,
    validate_residual_gn_options,
    validate_residual_iteration_options,
    validate_residual_lbfgs_options,
)


def test_lambda_gd_options_normalize_scalars_and_preconditioner():
    opts = validate_lambda_gd_options(
        max_iter="5",
        max_backtracks="3",
        bt_factor="0.25",
        preconditioner=" Mode_Diag ",
        precond_exponent="2.0",
    )

    assert opts.max_iter == 5
    assert opts.max_backtracks == 3
    assert opts.bt_factor == pytest.approx(0.25)
    assert opts.preconditioner == "mode_diag"
    assert opts.precond_exponent == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"max_iter": 0}, "max_iter"),
        ({"max_backtracks": -1}, "max_backtracks"),
        ({"bt_factor": 1.0}, "bt_factor"),
        ({"preconditioner": "radial_tridi"}, "Unknown preconditioner"),
        ({"preconditioner": "mode_diag", "precond_exponent": 0.0}, "precond_exponent"),
    ],
)
def test_lambda_gd_options_reject_invalid_inputs(override, match):
    kwargs = {
        "max_iter": 1,
        "max_backtracks": 0,
        "bt_factor": 0.5,
        "preconditioner": "none",
        "precond_exponent": 1.0,
    }
    kwargs.update(override)

    with pytest.raises(ValueError, match=match):
        validate_lambda_gd_options(**kwargs)


def test_fixed_boundary_gd_options_validate_line_search_and_gamma():
    opts = validate_fixed_boundary_gd_options(
        max_iter=2,
        max_backtracks=4,
        bt_factor=0.5,
        gamma=0,
    )

    assert opts.max_iter == 2
    assert opts.max_backtracks == 4
    assert opts.bt_factor == pytest.approx(0.5)
    assert opts.gamma == pytest.approx(0.0)

    with pytest.raises(ValueError, match="gamma=1"):
        validate_fixed_boundary_gd_options(
            max_iter=1,
            max_backtracks=0,
            bt_factor=0.5,
            gamma=1.0,
        )


def test_fixed_boundary_lbfgs_options_validate_history_size():
    opts = validate_fixed_boundary_lbfgs_options(
        history_size="7",
        max_iter="8",
        max_backtracks="9",
        bt_factor="0.4",
        gamma="0.0",
    )

    assert opts.history_size == 7
    assert opts.max_iter == 8
    assert opts.max_backtracks == 9
    assert opts.bt_factor == pytest.approx(0.4)
    assert opts.gamma == pytest.approx(0.0)

    with pytest.raises(ValueError, match="history_size"):
        validate_fixed_boundary_lbfgs_options(
            history_size=0,
            max_iter=1,
            max_backtracks=0,
            bt_factor=0.5,
            gamma=0.0,
        )


def test_pressure_shape_validation_is_pure_and_message_matches_solvers():
    validate_pressure_shape((3,), (3,))

    with pytest.raises(ValueError, match=r"pressure must have shape \(3,\), got \(2,\)"):
        validate_pressure_shape((2,), (3,))


def test_residual_lbfgs_options_normalize_all_solver_scalars():
    opts = validate_residual_lbfgs_options(
        w_rz="1.5",
        w_l="2.5",
        objective_scale="0.25",
        scale_rz="3.0",
        scale_l="4.0",
        history_size="5",
        max_iter="6",
        max_backtracks="7",
        bt_factor="0.8",
    )

    assert opts.w_rz == pytest.approx(1.5)
    assert opts.w_l == pytest.approx(2.5)
    assert opts.objective_scale == pytest.approx(0.25)
    assert opts.scale_rz == pytest.approx(3.0)
    assert opts.scale_l == pytest.approx(4.0)
    assert opts.history_size == 5
    assert opts.max_iter == 6
    assert opts.max_backtracks == 7
    assert opts.bt_factor == pytest.approx(0.8)


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"w_rz": -1.0}, "nonnegative"),
        ({"w_l": -1.0}, "nonnegative"),
        ({"objective_scale": 0.0}, "objective_scale"),
        ({"scale_rz": 0.0}, "scale_rz and scale_l"),
        ({"scale_l": 0.0}, "scale_rz and scale_l"),
        ({"history_size": 0}, "history_size"),
        ({"max_iter": 0}, "max_iter"),
        ({"max_backtracks": -1}, "max_backtracks"),
        ({"bt_factor": 0.0}, "bt_factor"),
    ],
)
def test_residual_lbfgs_options_reject_invalid_inputs(override, match):
    kwargs = {
        "w_rz": 1.0,
        "w_l": 1.0,
        "objective_scale": None,
        "scale_rz": 1.0,
        "scale_l": 1.0,
        "history_size": 1,
        "max_iter": 1,
        "max_backtracks": 0,
        "bt_factor": 0.5,
    }
    kwargs.update(override)

    with pytest.raises(ValueError, match=match):
        validate_residual_lbfgs_options(**kwargs)


def test_residual_gn_options_preserve_current_optional_defaults():
    opts = validate_residual_gn_options(
        damping=None,
        damping_increase="2.0",
        damping_decrease="1.0",
        max_damping=None,
        max_retries="3",
        zero_m1_iters=None,
        zero_m1_fsqz_thresh=None,
        w_rz="1.0",
        w_l="2.0",
        max_iter="5",
        cg_maxiter="6",
        max_backtracks="-1",
        bt_factor="0.5",
        objective_scale=None,
    )

    assert opts.damping is None
    assert opts.damping_increase == pytest.approx(2.0)
    assert opts.damping_decrease == pytest.approx(1.0)
    assert math.isinf(opts.max_damping_eff)
    assert opts.max_retries == 3
    assert opts.zero_m1_iters_eff == 0
    assert opts.zero_m1_fsqz_thresh is None
    assert opts.w_rz == pytest.approx(1.0)
    assert opts.w_l == pytest.approx(2.0)
    assert opts.max_iter == 5
    assert opts.cg_maxiter == 6
    assert opts.max_backtracks == -1
    assert opts.bt_factor == pytest.approx(0.5)
    assert opts.objective_scale is None


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"damping": -1.0}, "damping"),
        ({"damping_increase": 1.0}, "damping_increase"),
        ({"damping_decrease": 0.0}, "damping_decrease"),
        ({"damping_decrease": 1.1}, "damping_decrease"),
        ({"max_damping": 0.0}, "max_damping"),
        ({"max_retries": -1}, "max_retries"),
        ({"zero_m1_iters": -1}, "zero_m1_iters"),
        ({"zero_m1_fsqz_thresh": -1.0}, "zero_m1_fsqz_thresh"),
        ({"w_l": -1.0}, "nonnegative"),
        ({"max_iter": 0}, "max_iter"),
        ({"cg_maxiter": 0}, "cg_maxiter"),
        ({"bt_factor": 1.0}, "bt_factor"),
        ({"objective_scale": 0.0}, "objective_scale"),
    ],
)
def test_residual_gn_options_reject_invalid_inputs(override, match):
    kwargs = {
        "damping": 1e-3,
        "damping_increase": 10.0,
        "damping_decrease": 0.5,
        "max_damping": 1e6,
        "max_retries": 6,
        "zero_m1_iters": 50,
        "zero_m1_fsqz_thresh": 1e-6,
        "w_rz": 1.0,
        "w_l": 1.0,
        "max_iter": 1,
        "cg_maxiter": 1,
        "max_backtracks": 0,
        "bt_factor": 0.5,
        "objective_scale": None,
    }
    kwargs.update(override)

    with pytest.raises(ValueError, match=match):
        validate_residual_gn_options(**kwargs)


def test_residual_iteration_options_normalize_scalars_and_booleans():
    opts = validate_residual_iteration_options(
        max_iter="2",
        step_size="0.75",
        precompile_only=0,
        signgs="-1",
        lambda_update_scale="1.25",
        enforce_vmec_lambda_axis=1,
        vmec2000_control="",
        reference_mode=True,
        limit_dt_from_force=1,
        limit_update_rms=0,
        backtracking=[],
        strict_update=[1],
        jit_precompile=1,
        use_scan=0,
    )

    assert opts.max_iter == 2
    assert opts.step_size == pytest.approx(0.75)
    assert opts.precompile_only is False
    assert opts.signgs == -1
    assert opts.lambda_update_scale == pytest.approx(1.25)
    assert opts.enforce_vmec_lambda_axis is True
    assert opts.vmec2000_control is False
    assert opts.reference_mode is True
    assert opts.limit_dt_from_force is True
    assert opts.limit_update_rms is False
    assert opts.backtracking is False
    assert opts.strict_update is True
    assert opts.jit_precompile is True
    assert opts.use_scan is False


def test_residual_iteration_options_allow_zero_iterations_for_precompile_only():
    opts = validate_residual_iteration_options(
        max_iter=0,
        step_size=1.0,
        precompile_only=True,
        signgs=1,
        lambda_update_scale=1.0,
        enforce_vmec_lambda_axis=False,
        vmec2000_control=False,
        reference_mode=False,
        limit_dt_from_force=False,
        limit_update_rms=False,
        backtracking=False,
        strict_update=True,
        jit_precompile=False,
        use_scan=False,
    )

    assert opts.max_iter == 1
    assert opts.precompile_only is True


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"max_iter": 0}, "max_iter"),
        ({"step_size": 0.0}, "step_size"),
    ],
)
def test_residual_iteration_options_reject_invalid_inputs(override, match):
    kwargs = {
        "max_iter": 1,
        "step_size": 1.0,
        "precompile_only": False,
        "signgs": 1,
        "lambda_update_scale": 1.0,
        "enforce_vmec_lambda_axis": False,
        "vmec2000_control": False,
        "reference_mode": False,
        "limit_dt_from_force": False,
        "limit_update_rms": False,
        "backtracking": False,
        "strict_update": True,
        "jit_precompile": False,
        "use_scan": False,
    }
    kwargs.update(override)

    with pytest.raises(ValueError, match=match):
        validate_residual_iteration_options(**kwargs)
