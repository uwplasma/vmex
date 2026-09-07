"""Polish homotopy contracts; nonlinear integration stays in the full suite."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vmex.core.errors import (
    StrongForceContinuationError,
)
from vmex.core.polish import (
    apply_high_order_correction,
    make_strong_structured_chart,
    strong_root_rank,
    strong_root_residual,
)
from vmex.core.polish_driver import (
    PolishConfig,
    _IdentityPreconditioner,
    _arclength_to_target,
    _bordered_preconditioner,
    _branch_tangent,
    _build_mode_block_preconditioner,
    _continuation_precondition,
    _low_inverse,
    _normalized_low_residual_norm,
    _ptc_config,
    _residual_evaluations,
    _supports_keyword,
    polish_strong_root,
)

jax.config.update("jax_enable_x64", True)

from tests.test_polish_preconditioner import (
    DATA,
    _tree_dot, small_adapter as small_adapter, small_strong_root as small_strong_root,
)

pytestmark = pytest.mark.full


def test_solvax_continuation_api_compatibility_helpers():
    def legacy_preconditioner(state, rhs, dtau):
        del state, dtau
        return rhs

    def parameterized_preconditioner(state, rhs, dtau, parameter):
        del state, dtau, parameter
        return rhs

    assert not _supports_keyword(legacy_preconditioner, "parameter")
    assert _supports_keyword(parameterized_preconditioner, "parameter")
    np.testing.assert_array_equal(
        legacy_preconditioner(None, jnp.ones((2,)), None), jnp.ones((2,))
    )
    np.testing.assert_array_equal(
        parameterized_preconditioner(None, jnp.ones((2,)), None, None),
        jnp.ones((2,)),
    )
    assert _residual_evaluations(
        SimpleNamespace(nonlinear_steps=3, residual_evaluations=9)
    ) == 9
    assert _residual_evaluations(SimpleNamespace(nonlinear_steps=3)) == 4
    assert _residual_evaluations(SimpleNamespace(steps=2)) == 3
    assert not _supports_keyword(1, "parameter")


def test_parameterized_continuation_preconditioner_switches_at_half(monkeypatch):
    rhs = jnp.asarray([1.0, -2.0])
    monkeypatch.setattr(
        "vmex.core.polish_driver._low_inverse", lambda value, runtime: 2.0 * value
    )
    block = SimpleNamespace(apply=lambda value, alpha, dtau: 3.0 * value)
    np.testing.assert_array_equal(
        _continuation_precondition(rhs, 0.25, 1.0, SimpleNamespace(), block),
        2.0 * rhs,
    )
    np.testing.assert_array_equal(
        _continuation_precondition(rhs, 0.75, 1.0, SimpleNamespace(), block),
        3.0 * rhs,
    )
    np.testing.assert_array_equal(
        _continuation_precondition(
            rhs, 0.25, 1.0, SimpleNamespace(), block, SimpleNamespace()
        ),
        3.0 * rhs,
    )
    identity = _IdentityPreconditioner()
    np.testing.assert_array_equal(identity.apply(rhs), rhs)
    np.testing.assert_array_equal(
        _continuation_precondition(
            rhs,
            0.25,
            1.0,
            SimpleNamespace(),
            identity,
        ),
        rhs,
    )


def test_arclength_crossing_runs_target_correction_and_counts_work(monkeypatch):
    zero = jnp.zeros((2,))
    target_vector = jnp.asarray([0.25, -0.5])

    def corrector(
        residual,
        initial,
        *,
        tangent,
        predictor,
        config,
        admissible,
        parameterized_precond,
    ):
        del residual, initial, config
        assert bool(admissible(*predictor))
        np.testing.assert_array_equal(
            parameterized_precond(predictor, predictor, 1.0, tangent, predictor)[0],
            predictor[0],
        )
        return SimpleNamespace(
            x=predictor,
            steps=2,
            linear_iterations=3,
            residual_evaluations=4,
            converged=True,
            linear_converged=True,
        )

    def target(residual, initial, *, precond, admissible, config):
        del residual, initial, config
        assert bool(admissible(target_vector))
        np.testing.assert_array_equal(precond(target_vector, target_vector, 1.0), target_vector)
        return SimpleNamespace(
            x=target_vector,
            steps=5,
            linear_iterations=6,
            residual_evaluations=7,
            converged=True,
            linear_converged=True,
        )

    monkeypatch.setattr(
        "vmex.core.polish_driver._solvax_continuation_api",
        lambda: (None, None, None, corrector, target),
    )
    monkeypatch.setattr(
        "vmex.core.polish_driver._ptc_config", lambda config, **kwargs: object()
    )
    monkeypatch.setattr(
        "vmex.core.polish_driver._branch_tangent",
        lambda *args, **kwargs: (jnp.zeros_like(zero), jnp.asarray(1.0)),
    )
    monkeypatch.setattr(
        "vmex.core.polish_driver._apply_bordered_preconditioner",
        lambda state, rhs, dtau, tangent, runtime, block, chart=None: rhs,
    )
    monkeypatch.setattr(
        "vmex.core.polish_driver._low_inverse", lambda rhs, runtime: rhs
    )
    monkeypatch.setattr(
        "vmex.core.polish_driver.strong_root_residual",
        lambda vector, runtime, alpha: vector + alpha,
    )
    result = _arclength_to_target(
        zero,
        0.95,
        SimpleNamespace(layout=SimpleNamespace(size=2), operator_balance=1.0),
        PolishConfig(max_arclength_steps=1, arclength_step=0.1),
        lambda vector, alpha: jnp.all(jnp.isfinite(vector)) & jnp.isfinite(alpha),
        None,
        None,
    )
    np.testing.assert_array_equal(result[0], target_vector)
    assert result[1:] == (1.0, 1, 7, 9, 11)


def test_bordered_tangent_uses_previous_orientation(monkeypatch):
    zero = jnp.zeros((2,))
    previous = (jnp.asarray([-1.0, -1.0]), jnp.asarray(-1.0))

    def fake_gmres(operator, rhs, *, precond, **kwargs):
        del kwargs
        physical, normalization = operator(rhs)
        assert physical.shape == zero.shape
        assert np.isfinite(float(normalization))
        for actual, expected in zip(
            jax.tree.leaves(precond(rhs)), jax.tree.leaves(rhs), strict=True
        ):
            np.testing.assert_array_equal(actual, expected)
        return SimpleNamespace(
            x=(jnp.asarray([0.5, 0.25]), jnp.asarray(0.5)),
            converged=True,
            residual_norm=jnp.asarray(0.0),
            iterations=1,
        )

    monkeypatch.setattr("vmex.core.polish_driver.gmres", fake_gmres)
    monkeypatch.setattr(
        "vmex.core.polish_driver._bordered_preconditioner",
        lambda *args, **kwargs: lambda state, rhs, dtau: rhs,
    )
    monkeypatch.setattr(
        "vmex.core.polish_driver.strong_root_residual",
        lambda vector, runtime, alpha: vector + alpha * jnp.ones_like(vector),
    )
    tangent = _branch_tangent(
        zero,
        0.5,
        SimpleNamespace(),
        PolishConfig(),
        previous,
        None,
    )
    np.testing.assert_allclose(
        jnp.vdot(tangent[0], tangent[0]).real + tangent[1] ** 2,
        1.0,
        rtol=2.0e-13,
    )
    assert float(jnp.vdot(tangent[0], previous[0]) + tangent[1] * previous[1]) > 0.0


def test_square_strong_root_endpoint_jvp_boundary_and_rank(small_strong_root):
    runtime = small_strong_root
    zero = jnp.zeros((runtime.layout.size,), dtype=jnp.float64)
    radial_matrix = runtime.native.radial_basis.basis_matrix(runtime.radial_nodes**2)
    assert runtime.radial_nodes.size > runtime.native.radial_basis.size
    assert runtime.theta.size >= 4 * int(np.max(np.abs(runtime.native.m))) + 5
    assert runtime.zeta.size == 1
    for mode, mode_m in enumerate(np.asarray(runtime.native.m)):
        regularized_matrix = (
            runtime.radial_nodes[:, None] ** abs(int(mode_m)) * radial_matrix
        )
        np.testing.assert_allclose(
            runtime.radial_fit[mode] @ regularized_matrix,
            np.eye(runtime.native.radial_basis.size),
            rtol=5.0e-10,
            atol=5.0e-10,
        )
    low_endpoint = strong_root_residual(zero, runtime, 0.0)
    strong_endpoint = strong_root_residual(zero, runtime, 1.0)
    # The alpha = 0 endpoint is legacy_residual(x0) - legacy_defect: two
    # evaluations of one nonlinear function in two separately compiled
    # programs.  XLA does not promise bit-identical fusion across programs or
    # platforms, so the cancellation bottoms out at round-off (measured
    # 2.8e-14 on the Linux CI runner, exact zero on arm64).  The bound is the
    # cancellation floor of the row-scaled O(1) residual, not a physics
    # tolerance.
    np.testing.assert_allclose(low_endpoint, 0.0, atol=1.0e-12)
    assert strong_endpoint.shape == zero.shape
    assert np.all(np.isfinite(np.asarray(strong_endpoint)))
    # The initial force RMS is divided by the measured low-inverse stiffness.
    np.testing.assert_allclose(
        jnp.linalg.norm(strong_endpoint),
        np.sqrt(runtime.layout.size) / runtime.operator_balance,
        rtol=3.0e-13,
    )
    assert float(runtime.operator_balance) >= 1.0
    assert runtime.coordinate_scale.shape == zero.shape
    assert runtime.equation_scale.shape == zero.shape
    assert np.all(np.asarray(runtime.coordinate_scale) > 0.0)
    assert np.all(np.asarray(runtime.equation_scale) > 0.0)
    assert runtime.strong_block_sign.shape == (3,)
    np.testing.assert_array_equal(jnp.abs(runtime.strong_block_sign), 1.0)

    probe = jnp.linspace(-0.01, 0.015, runtime.layout.size)
    low_probe = strong_root_residual(probe, runtime, 0.0)
    strong_probe = strong_root_residual(probe, runtime, 1.0)
    alpha = 0.37
    np.testing.assert_allclose(
        strong_root_residual(probe, runtime, alpha),
        low_probe + alpha * (strong_probe - low_probe),
        rtol=2.0e-13,
        atol=2.0e-13,
    )

    direction = jnp.linspace(-0.2, 0.3, runtime.layout.size)
    _, tangent = jax.jvp(
        lambda value: strong_root_residual(value, runtime, 1.0),
        (zero,),
        (direction,),
    )
    step = 2.0e-5
    finite_difference = (
        strong_root_residual(step * direction, runtime, 1.0)
        - strong_root_residual(-step * direction, runtime, 1.0)
    ) / (2.0 * step)
    np.testing.assert_allclose(tangent, finite_difference, rtol=2.0e-6, atol=2.0e-7)

    correction = runtime.layout.unpack(0.01 * direction)
    corrected = apply_high_order_correction(runtime.native, correction)
    for name in ("R_cos", "R_sin", "Z_cos", "Z_sin"):
        np.testing.assert_array_equal(
            np.asarray(getattr(corrected, name)[:, -1]),
            np.asarray(getattr(runtime.native, name)[:, -1]),
        )
    assert corrected.source.endswith("strong-root correction")

    rank, singular_values = strong_root_rank(runtime, relative_tolerance=1.0e-8)
    assert rank == runtime.layout.size
    assert float(singular_values[-1]) > 0.0


def test_low_vector_preconditioner_is_finite_on_native_coordinates(
    small_strong_root,
):
    runtime = small_strong_root
    zero = jnp.zeros((runtime.layout.size,), dtype=jnp.float64)
    direction = jnp.linspace(-0.1, 0.2, runtime.layout.size)
    _, response = jax.jvp(
        lambda value: strong_root_residual(value, runtime, 0.0),
        (zero,),
        (direction,),
    )
    recovered = _low_inverse(response, runtime)
    assert np.all(np.isfinite(np.asarray(recovered)))
    assert float(jnp.linalg.norm(recovered)) > 0.0
    assert float(jnp.linalg.norm(recovered)) < 10.0 * float(
        jnp.linalg.norm(direction)
    )


def test_scaled_low_inverse_and_transpose_are_exact_duals(small_strong_root):
    runtime = small_strong_root
    left = runtime.transfer.restrict(
        runtime.layout.unpack(jnp.linspace(-0.2, 0.1, runtime.layout.size))
    )
    right = runtime.transfer.restrict(
        runtime.layout.unpack(jnp.linspace(0.3, -0.15, runtime.layout.size))
    )
    forward = runtime.low_preconditioner.solve_scaled(left)
    transpose = runtime.low_preconditioner.solve_scaled_transpose(right)
    np.testing.assert_allclose(
        _tree_dot(forward, right),
        _tree_dot(left, transpose),
        rtol=3.0e-12,
        atol=3.0e-12,
    )


def test_arclength_tangent_and_bordered_preconditioner_are_finite(
    small_strong_root,
):
    runtime = small_strong_root
    zero = jnp.zeros((runtime.layout.size,), dtype=jnp.float64)
    block_preconditioner = _build_mode_block_preconditioner(runtime)
    direction = jnp.linspace(-0.15, 0.25, runtime.layout.size)
    _, response = jax.jvp(
        lambda value: strong_root_residual(value, runtime, 1.0),
        (zero,),
        (direction,),
    )
    recovered = block_preconditioner.apply(response, 1.0)
    np.testing.assert_allclose(recovered, direction, rtol=3.0e-8, atol=3.0e-8)
    _, pullback = jax.vjp(
        lambda value: strong_root_residual(value, runtime, 1.0), zero
    )
    transpose_response = pullback(direction)[0]
    transpose_recovered = block_preconditioner.apply_transpose(
        transpose_response, 1.0
    )
    np.testing.assert_allclose(
        transpose_recovered, direction, rtol=3.0e-8, atol=3.0e-8
    )
    tangent = _branch_tangent(
        zero,
        0.0,
        runtime,
        PolishConfig(),
        None,
        block_preconditioner,
    )
    np.testing.assert_allclose(
        jnp.vdot(tangent[0], tangent[0]).real + tangent[1] ** 2,
        1.0,
        rtol=2.0e-13,
    )
    assert float(tangent[1]) > 0.0
    rhs = (jnp.linspace(-0.2, 0.3, runtime.layout.size), jnp.asarray(0.4))
    corrected = _bordered_preconditioner(
        runtime, tangent, block_preconditioner
    )((zero, 0.0), rhs, 1.0e6)
    assert corrected[0].shape == zero.shape
    assert np.all(np.isfinite(np.asarray(corrected[0])))
    assert np.isfinite(float(corrected[1]))


def test_polish_driver_records_bounded_unpolished_return(
    small_strong_root, monkeypatch
):
    class InitialCertificate:
        normalized_l2 = jnp.asarray(2.0)
        radial_refinement_difference = jnp.asarray(0.0)
        minimum_signed_jacobian = jnp.asarray(0.5)
        # the driver now reports the non-saturating window normalizations
        # beside eps_F, so a stand-in certificate has to carry them
        window_normalizations = SimpleNamespace(
            volume_average_force=jnp.asarray(1.0),
            relative_force_error=jnp.asarray(1.0),
            magnetic_relative_force_error=jnp.asarray(1.0),
            s_min=0.1, s_max=0.99,
        )

    config = PolishConfig(
        max_continuation_stages=1,
        alpha_initial_step=1.0e-3,
        alpha_min_step=1.0e-3,
        alpha_max_step=1.0e-3,
        max_nonlinear_iterations=12,
        preconditioner="legacy",
        use_pseudo_arclength=True,
        fail_policy="return_unpolished",
    )

    def fail_tangent(*args, **kwargs):
        del args, kwargs
        raise StrongForceContinuationError("test tangent failure")

    def endpoint(residual, initial, **kwargs):
        del residual, kwargs
        return SimpleNamespace(
            x=initial,
            steps=2,
            linear_iterations=3,
            residual_evaluations=4,
            converged=True,
            linear_converged=True,
        )

    def continuation(residual, initial, *, accept_stage, **kwargs):
        del residual, kwargs
        alpha = 1.0e-3
        accept_stage(initial, alpha, None)
        stage = SimpleNamespace(
            nonlinear_steps=5,
            linear_iterations=6,
            residual_evaluations=7,
            accepted=True,
        )
        return SimpleNamespace(
            x=initial,
            alpha=alpha,
            steps=(stage,),
            converged=False,
        )

    monkeypatch.setattr(
        "vmex.core.polish_driver._solvax_continuation_api",
        lambda: (lambda **kwargs: object(), None, continuation, None, endpoint),
    )
    monkeypatch.setattr(
        "vmex.core.polish_driver._ptc_config", lambda config, **kwargs: object()
    )
    monkeypatch.setattr(
        "vmex.core.polish_driver._arclength_to_target", fail_tangent
    )
    chart = make_strong_structured_chart(small_strong_root)
    result = polish_strong_root(
        small_strong_root,
        config=config,
        initial_certificate=InitialCertificate(),
        chart=chart,
    )
    report = result.polish_report
    assert not report.converged
    assert report.termination_reason == "pseudo-arclength-tangent-failed"
    assert report.final_alpha == pytest.approx(1.0e-3)
    assert report.continuation_accepted == 1
    assert report.continuation_rejected == 0
    assert report.nonlinear_iterations > 0
    assert report.linear_iterations > 0
    assert report.minimum_signed_jacobian > 0.0
    np.testing.assert_array_equal(result.correction, 0.0)
    assert result.native_equilibrium is small_strong_root.native


def test_legacy_polish_announces_refinement_and_certificate_phases():
    """The legacy driver's setup phases each emit a notice before starting.

    Small solovev case; the raw lift never certifies at the default bar, so
    the run passes through every phase (refinement, initial certificate,
    preconditioner/root-runtime build) into one bounded Gauss-Newton step.
    """
    from vmex import VmecInput
    from vmex.core.polish_driver import polish_legacy_solution
    from vmex.core.solver import resolution_from_input, solve

    inp = VmecInput.from_file(
        str(DATA / "input.solovev")
    ).change_resolution(mpol=3, ntor=0, ntheta=12, nzeta=4)
    inp = dataclasses.replace(
        inp, ns_array=np.asarray([5]), ftol_array=np.asarray([1.0e-9]),
        niter_array=np.asarray([2000]),
    )
    result = solve(inp)
    lines: list[str] = []

    def capture(text="", **kwargs):
        lines.append(str(text))

    polish_legacy_solution(
        inp,
        resolution_from_input(inp, ns=5),
        result.state,
        config=PolishConfig(
            max_nonlinear_iterations=1,
            collocation_scale_probes=2,
            fail_policy="return_unpolished",
        ),
        verbose=True,
        emit=capture,
    )
    text = "\n".join(lines)
    assert "refining the converged state" in text
    assert "evaluating the initial force certificate" in text
    assert "building the polish preconditioner and root runtime" in text
    # eps_F alone is unreadable on a low-beta case: the console must name
    # its ceiling and print the measures that can actually move.
    assert "EPS-F is bounded by 2 by construction" in text
    assert "<|F|>  [N m^-3]" in text
    assert "<|F|>/<|grad B^2/2mu0|>" in text
    assert "|F| L2 near axis [N m^-3]" in text
    assert "(volume averages over s in [0.10, 0.99])" in text


def test_polish_ptc_stopping_is_invariant_to_positive_residual_scaling():
    tolerance = 2.0e-7
    config = _ptc_config(PolishConfig(tolerance=tolerance), residual_scale=3.0e-4)
    rescaled = _ptc_config(
        PolishConfig(tolerance=tolerance), residual_scale=7.0 * 3.0e-4
    )
    assert config.rtol == tolerance
    assert config.atol == pytest.approx(tolerance * 3.0e-4)
    assert rescaled.atol == pytest.approx(7.0 * config.atol)


def test_low_endpoint_check_ignores_numerical_row_equilibration():
    residual = jnp.asarray([2.0e-9, -6.0e-9])
    runtime = SimpleNamespace(
        equation_scale=jnp.asarray([2.0, 3.0]),
        layout=SimpleNamespace(size=2),
    )
    rescaled_runtime = SimpleNamespace(
        equation_scale=7.0 * runtime.equation_scale,
        layout=runtime.layout,
    )
    expected = jnp.linalg.norm(residual / runtime.equation_scale) / jnp.sqrt(2.0)
    np.testing.assert_allclose(
        _normalized_low_residual_norm(residual, runtime),
        expected,
        rtol=2.0e-13,
    )
    np.testing.assert_allclose(
        _normalized_low_residual_norm(7.0 * residual, rescaled_runtime),
        expected,
        rtol=2.0e-13,
    )

