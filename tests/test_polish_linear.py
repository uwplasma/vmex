"""Polish linear contracts; nonlinear integration stays in the full suite."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vmex.core.errors import (
    StrongForceCertificationError,
    StrongForceLinearSolveError,
)
from vmex.core.omnigenity import boozer_spectrum_high_order
from vmex.core.polish import (
    make_strong_structured_chart,
)
from vmex.core.polish_driver import (
    PolishConfig,
    PolishContext,
    polish_collocation_least_squares,
)
from vmex.core.polish_implicit import (
    PolishLinearConfig,
    PolishLinearReport,
    _checked_solution,
    _linear_report,
    _solve_linear,
    _collocation_stationarity,
    _tree_norm as _implicit_tree_norm,
    collocation_polish_adjoint,
    collocation_polish_tangent,
    implicit_collocation_polished_state,
)

jax.config.update("jax_enable_x64", True)

from tests.test_polish_preconditioner import (
    _random_like, _tree_dot, _tree_norm, small_adapter as small_adapter, small_strong_root as small_strong_root,
)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"rtol": 0.0}, "rtol"),
        ({"atol": -1.0}, "atol"),
        ({"restart": 0}, "restart"),
        ({"max_restarts": 0}, "max_restarts"),
        ({"fail_policy": "ignore"}, "fail_policy"),
        ({"stationarity_rtol": 0.0}, "stationarity_rtol"),
        ({"stationarity_rtol": np.inf}, "stationarity_rtol"),
        ({"stationarity_atol": -1.0}, "stationarity_atol"),
        ({"stationarity_atol": np.nan}, "stationarity_atol"),
    ],
)
def test_polish_linear_config_validation(updates, message):
    with pytest.raises(ValueError, match=message):
        PolishLinearConfig(**updates)


def test_polish_linear_failure_policy_is_explicit():
    value = jnp.ones((2,))
    report = PolishLinearReport(
        residual_norm=jnp.asarray(2.0),
        tolerance=jnp.asarray(1.0),
        iterations=jnp.asarray(3),
        converged=jnp.asarray(False),
    )
    with pytest.raises(StrongForceLinearSolveError, match="did not converge"):
        _checked_solution(value, report, PolishLinearConfig(), "test")
    result = _checked_solution(
        value,
        report,
        PolishLinearConfig(fail_policy="nan"),
        "test",
    )
    assert np.isnan(result).all()
    assert _implicit_tree_norm((jnp.asarray([3.0, 4.0]),)) == pytest.approx(5.0)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize(
    ("rhs", "value", "applied", "flag", "accepted"),
    [
        (1.0, 0.0, 0.0, True, False),
        (1.0, 0.0, 0.0, False, False),
        (1.0, 1.0, 1.0, False, True),
        (1.0, np.nan, 1.0, True, False),
        (1.0, np.inf, 1.0, True, False),
        (np.nan, 1.0, 1.0, True, False),
        (np.inf, 1.0, 1.0, True, False),
        (1.0, 1.0, np.nan, True, False),
        (1.0, 1.0, np.inf, True, False),
        (1.0e200, 1.0e200, 1.0e200, True, False),
        (1.0, 1.0e200, 1.0e200, True, False),
        (0.0, 0.0, 0.0, False, True),
        (0.0, 1.0e-12, 1.0e-12, False, True),
        (0.0, 1.0e-9, 1.0e-9, True, False),
    ],
)
def test_polish_linear_true_certificate(compiled, rhs, value, applied, flag, accepted):
    def report_for(rhs, value, applied):
        return _linear_report(
            lambda _: applied,
            rhs,
            SimpleNamespace(x=value, converged=flag, iterations=jnp.asarray(30)),
            PolishLinearConfig(),
        )

    with jax.disable_jit(False):
        report = (jax.jit(report_for) if compiled else report_for)(
            *[jnp.asarray([item, item]) for item in (rhs, value, applied)]
        )
    assert bool(report.converged) == accepted
    assert int(report.iterations) == 30


@pytest.mark.parametrize("policy", ["raise", "nan"])
def test_polish_linear_compiled_failure_returns_nan_and_status(policy):
    config = PolishLinearConfig(fail_policy=policy)

    def checked(value):
        report = _linear_report(
            lambda x: x, jnp.ones(1),
            SimpleNamespace(x=value, converged=True, iterations=jnp.asarray(1)),
            config,
        )
        return _checked_solution(value, report, config, "tangent"), report

    with jax.disable_jit(False):
        value, report = jax.jit(checked)(jnp.zeros(1))
    assert not bool(report.converged)
    assert bool(jnp.all(jnp.isnan(value)))


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_polish_linear_krylov_true_residual(transpose, compiled):
    matrix = jnp.asarray([[4.0, 1.0], [-2.0, 3.0]])
    if transpose:
        matrix = matrix.T
    rhs = jnp.asarray([2.0, -1.0])

    def solve(rhs):
        return _solve_linear(
            lambda x: matrix @ x, rhs, lambda x: x / 4.0,
            PolishLinearConfig(), "adjoint" if transpose else "tangent",
        )

    with jax.disable_jit(False):
        value, report = (jax.jit(solve) if compiled else solve)(rhs)
    assert bool(report.converged)
    np.testing.assert_allclose(value, np.linalg.solve(matrix, rhs), atol=1.0e-11)
    assert float(jnp.linalg.norm(rhs - matrix @ value)) <= float(report.tolerance)


@pytest.mark.parametrize("compiled", [False, True])
def test_polish_linear_iteration_exhaustion(compiled):
    matrix = jnp.asarray([[4.0, 1.0], [-2.0, 3.0]])
    rhs = jnp.asarray([2.0, -1.0])

    def solve(rhs):
        return _solve_linear(
            lambda x: matrix @ x, rhs, lambda x: x,
            PolishLinearConfig(restart=1, max_restarts=1, fail_policy="nan"),
            "tangent",
        )

    with jax.disable_jit(False):
        value, report = (jax.jit(solve) if compiled else solve)(rhs)
    assert not bool(report.converged)
    assert float(report.residual_norm) > float(report.tolerance)
    assert bool(jnp.all(jnp.isnan(value)))


@pytest.mark.parametrize("compiled", [False, True])
def test_polish_vjp_uses_primal_native_input(monkeypatch, compiled):
    """An analytic stationary root detects reuse of a stale native parameter."""
    from vmex.core import polish_implicit as pi

    @dataclasses.dataclass(frozen=True, eq=False)
    class Runtime:
        native: jax.Array

    # g(c,q)=c-q^2=0, output=q+c, hence d(output)/dq=1+2q.
    # Keep the actual adjoint/Krylov/custom-VJP chain; replace only the physics.
    runtime = Runtime(jnp.asarray([1.0]))
    context = PolishContext(runtime, SimpleNamespace(size=1), jnp.ones(1), jnp.ones(1))
    monkeypatch.setattr(pi, "_collocation_stationarity", lambda c, q, runtime, chart: c - q*q)
    monkeypatch.setattr(pi, "_collocation_corrected_state", lambda q, c, runtime, chart: q + c)

    def objective(q):
        stationary = context._replace(correction=jax.lax.stop_gradient(q*q))
        return jnp.sum(implicit_collocation_polished_state(q, stationary))

    with jax.disable_jit(False):
        derivative = jax.grad(objective)
        if compiled:
            derivative = jax.jit(derivative)
        # One compiled function must handle changed native data correctly.
        for q in (1.0, 2.0, -0.5):
            np.testing.assert_allclose(
                derivative(jnp.asarray([q])), [1.0 + 2.0*q], atol=1.0e-11)


@pytest.fixture
def analytic_stationarity_context(monkeypatch):
    from vmex.core import polish_implicit as pi

    @dataclasses.dataclass(frozen=True, eq=False)
    class Runtime:
        native: jax.Array

    # r=[c^2-q,c] has a nonzero-residual stationary root c=sqrt(q-1/2).
    # Its exact Hessian is 4q-2, whereas the GN approximation is 4q-1.
    monkeypatch.setattr(pi, "strong_collocation_residual_at_native",
                        lambda c, q, runtime, chart: jnp.concatenate((c*c-q, c)))
    monkeypatch.setattr(pi, "_collocation_corrected_state", lambda q, c, runtime, chart: q+c)
    return PolishContext(Runtime(jnp.asarray([1.5])), SimpleNamespace(size=1),
                         jnp.ones(1), jnp.asarray([3.0]), 2.0, 1.0)


@pytest.mark.parametrize("compiled", [False, True])
def test_polish_stationary_nonzero_residual_derivatives(analytic_stationarity_context, compiled):
    context = analytic_stationarity_context

    def responses(q):
        root = jnp.sqrt(q-0.5)
        current = context._replace(
            runtime=dataclasses.replace(context.runtime, native=q), correction=root)
        tangent = collocation_polish_tangent(current, jnp.ones_like(q))
        adjoint = collocation_polish_adjoint(current, jnp.ones_like(q))
        custom = jax.grad(lambda native: jnp.sum(implicit_collocation_polished_state(
            native, context._replace(correction=root))))(q)
        return tangent, adjoint, custom

    with jax.disable_jit(False):
        evaluate = jax.jit(responses) if compiled else responses
        for q in (1.5, 4.5):
            tangent, adjoint, custom = evaluate(jnp.asarray([q]))
            expected = 1.0 + 0.5/np.sqrt(q-0.5)
            for result in (tangent, adjoint):
                assert bool(result.report.converged)
                assert bool(result.report.stationarity_converged)
                assert bool(result.report.linear_converged)
                assert float(result.report.stationarity_norm) <= float(result.report.stationarity_tolerance)
            for actual in (tangent.native_tangent, adjoint.native_cotangent, custom):
                np.testing.assert_allclose(actual, expected, atol=1.0e-10)


@pytest.mark.parametrize("mode", ["tangent", "adjoint", "vjp"])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("policy", ["raise", "nan"])
def test_polish_nonstationary_derivative_failure(analytic_stationarity_context, mode, compiled, policy):
    context = analytic_stationarity_context
    config = PolishLinearConfig(fail_policy=policy)

    def response(correction):
        current = context._replace(correction=correction)
        if mode == "vjp":
            return jax.grad(lambda native: jnp.sum(implicit_collocation_polished_state(
                native, current, config)))(current.runtime.native)
        function = collocation_polish_tangent if mode == "tangent" else collocation_polish_adjoint
        result = function(current, jnp.ones(1), config=config)
        return result[0], result.report

    with jax.disable_jit(False):
        evaluate = jax.jit(response) if compiled else response
        if not compiled and policy == "raise":
            with pytest.raises(StrongForceCertificationError) as failure:
                evaluate(jnp.asarray([1.1]))
            assert failure.value.stationarity_norm == pytest.approx(0.3465)
            assert failure.value.stationarity_norm > failure.value.stationarity_tolerance
        else:
            result = evaluate(jnp.asarray([1.1]))
            value = result if mode == "vjp" else result[0]
            assert bool(jnp.all(jnp.isnan(value)))
            if mode != "vjp":
                report = result[1]
                assert not bool(report.converged)
                assert not bool(report.stationarity_converged)
                assert not bool(report.linear_converged)
                assert int(report.iterations) == 0
                assert float(report.stationarity_norm) == pytest.approx(0.3465)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize(
    ("field", "value"),
    [("residual_scale", v) for v in (0.0, -1.0, np.nan, np.inf)]
    + [("stationarity_reference", v) for v in (-1.0, np.nan, np.inf)]
    + [("variable_scale", jnp.asarray([v])) for v in (0.0, -1.0, np.nan, np.inf)]
    + [("correction", jnp.asarray([np.nan]))],
)
def test_polish_stationarity_rejects_invalid_scaling(analytic_stationarity_context, compiled, field, value):
    from vmex.core.polish_implicit import _stationarity_certificate

    context = analytic_stationarity_context._replace(**{field: value})
    certificate = lambda g: _stationarity_certificate(g, context, PolishLinearConfig())  # noqa: E731
    with jax.disable_jit(False):
        result = (jax.jit(certificate) if compiled else certificate)(jnp.zeros(1))
    assert not bool(result[2])


def test_polish_stationarity_scaling_and_shapes(analytic_stationarity_context):
    from vmex.core.polish_implicit import _stationarity_certificate

    context = analytic_stationarity_context._replace(variable_scale=jnp.asarray([2.0]), residual_scale=4.0)
    config = PolishLinearConfig(stationarity_atol=1.0)
    norm, tolerance, valid = _stationarity_certificate(jnp.asarray([8.0]), context, config)
    assert float(norm) == 1.0
    assert float(tolerance) == 1.0
    assert bool(valid)
    assert not bool(_stationarity_certificate(jnp.asarray([8.001]), context, config)[2])
    for gradient in (jnp.asarray([np.nan]), jnp.asarray([np.inf])):
        assert not bool(_stationarity_certificate(gradient, context, config)[2])
    with pytest.raises(ValueError, match="scalars"):
        _stationarity_certificate(jnp.ones(1), context._replace(residual_scale=jnp.ones(1)), config)
    with pytest.raises(ValueError, match="shape"):
        _stationarity_certificate(jnp.ones(1), context._replace(variable_scale=jnp.ones(2)), config)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("policy", ["raise", "nan"])
def test_stationary_polish_reports_a_failed_linear_solve(compiled, policy):
    from vmex.core.polish_implicit import _solve_stationary_linear

    matrix = jnp.asarray([[4.0, 1.0], [-2.0, 3.0]])
    config = PolishLinearConfig(restart=1, max_restarts=1, fail_policy=policy)

    def solve(rhs):
        return _solve_stationary_linear(
            lambda x: matrix @ x, rhs, lambda x: x, config, "tangent",
            (jnp.asarray(0.0), jnp.asarray(1.0e-8), jnp.asarray(True)))

    with jax.disable_jit(False):
        evaluate = jax.jit(solve) if compiled else solve
        if not compiled and policy == "raise":
            with pytest.raises(StrongForceLinearSolveError):
                evaluate(jnp.asarray([2.0, -1.0]))
        else:
            value, report = evaluate(jnp.asarray([2.0, -1.0]))
            assert bool(report.stationarity_converged)
            assert not bool(report.linear_converged)
            assert not bool(report.converged)
            assert int(report.iterations) > 0
            assert bool(jnp.all(jnp.isnan(value)))


def test_collocation_certification_error_retains_both_failure_gates():
    error = StrongForceCertificationError(
        "not certified",
        solver_converged=True,
        normalized_l2=0.2,
        tolerance=0.1,
        radial_refinement=0.03,
        radial_refinement_tolerance=0.01,
    )

    assert error.solver_converged
    assert error.normalized_l2 > error.tolerance
    assert error.radial_refinement > error.radial_refinement_tolerance


def test_implicit_polish_rejects_mismatched_inputs(small_strong_root):
    chart = make_strong_structured_chart(small_strong_root)
    good = PolishContext(
        small_strong_root,
        chart,
        jnp.zeros((chart.size,)),
        jnp.ones((chart.size,)),
    )
    bad = good._replace(correction=jnp.zeros((chart.size + 1,)))
    with pytest.raises(ValueError, match="correction has shape"):
        collocation_polish_tangent(bad, small_strong_root.native)
    with pytest.raises(ValueError, match="correction has shape"):
        collocation_polish_adjoint(bad, small_strong_root.native)
    with pytest.raises(ValueError, match="native_tangent"):
        collocation_polish_tangent(good, jnp.asarray(0.0))
    with pytest.raises(ValueError, match="polished_cotangent"):
        collocation_polish_adjoint(good, jnp.asarray(0.0))
    with pytest.raises(ValueError, match="native must have"):
        implicit_collocation_polished_state(jnp.asarray(0.0), good)


@pytest.mark.full
def test_physics_accepted_polish_can_fail_derivative_stationarity(small_strong_root):
    chart = make_strong_structured_chart(small_strong_root, balance_iterations=1, balance_probes=2)
    with jax.disable_jit(False):
        result = polish_collocation_least_squares(
            small_strong_root, chart=chart,
            config=PolishConfig(tolerance=2.0, validation_tolerance=10.0,
                                radial_refinement_tolerance=10.0, collocation_scale_probes=2,
                                max_nonlinear_iterations=1))
        assert result.polish_report.converged
        # This loose solver bar accepts the initial point, whose exact gradient
        # is not within the default derivative stationarity threshold.
        assert result.polish_report.nonlinear_iterations == 0
        with pytest.raises(StrongForceCertificationError, match="stationary"):
            collocation_polish_tangent(result.context, _random_like(small_strong_root.native, 51))


@pytest.mark.full
def test_collocation_polish_primal_and_derivatives(small_strong_root):
    # Compile this numerical integration explicitly; the suite disables JIT.
    with jax.disable_jit(False):
        chart = make_strong_structured_chart(
            small_strong_root, balance_iterations=1, balance_probes=2
        )
        result = polish_collocation_least_squares(
            small_strong_root,
            chart=chart,
            config=PolishConfig(
                tolerance=1.0e-10,
                validation_tolerance=10.0,
                radial_refinement_tolerance=10.0,
                collocation_scale_probes=2,
                max_nonlinear_iterations=40,
                fail_policy="return_unpolished",
            ),
        )
        assert result.correction.shape == (small_strong_root.layout.size,)
        assert result.polish_report.least_squares_success is not None
        assert result.polish_report.variable_scale_probes == 2
        assert result.context is not None
        assert result.polish_report.converged
        # The 1e-10 primal stopping flag is not derivative eligibility.
        # Check the public stationarity and linear certificates below.
        assert result.polish_report.nonlinear_iterations > 0

        native_tangent = _random_like(small_strong_root.native, 51)
        output_cotangent = _random_like(small_strong_root.native, 52)
        linear_config = PolishLinearConfig(
            rtol=2.0e-10,
            atol=2.0e-11,
            restart=chart.size,
            max_restarts=5,
        )
        tangent = collocation_polish_tangent(
            result.context, native_tangent, config=linear_config
        )
        adjoint = collocation_polish_adjoint(
            result.context, output_cotangent, config=linear_config
        )
        assert bool(tangent.report.converged)
        assert bool(adjoint.report.converged)
        assert bool(tangent.report.stationarity_converged)
        assert bool(adjoint.report.stationarity_converged)
        for report in (tangent.report, adjoint.report):
            assert np.isfinite(float(report.stationarity_norm))
            assert float(report.stationarity_norm) <= float(report.stationarity_tolerance)
        np.testing.assert_allclose(tangent.report.stationarity_norm,
                                   result.polish_report.least_squares_optimality, rtol=1.0e-4, atol=1.0e-8)
        np.testing.assert_allclose(
            _tree_dot(output_cotangent, tangent.native_tangent),
            _tree_dot(adjoint.native_cotangent, native_tangent),
            rtol=2.0e-5,
            atol=2.0e-6,
        )

        def objective(native):
            polished = implicit_collocation_polished_state(
                native, result.context, linear_config
            )
            return _tree_dot(polished, output_cotangent)

        custom_gradient = jax.grad(objective)(small_strong_root.native)
        difference = jax.tree.map(
            jnp.subtract, custom_gradient, adjoint.native_cotangent
        )
        assert _tree_norm(difference) <= 2.0e-5 * max(
            _tree_norm(adjoint.native_cotangent), 1.0
        )

        polished = implicit_collocation_polished_state(
            small_strong_root.native,
            result.context,
            linear_config,
        )

        def boozer_objective(native):
            spectrum = boozer_spectrum_high_order(
                native,
                surfaces=[0.49],
                mboz=4,
                nboz=2,
                ntheta=12,
                nzeta=8,
            )
            return jnp.sum(spectrum["bmnc_b"][:, 1:] ** 2)

        boozer_cotangent = jax.grad(boozer_objective)(polished)
        boozer_adjoint = collocation_polish_adjoint(
            result.context,
            boozer_cotangent,
            config=linear_config,
        )
        boozer_gradient = jax.grad(
            lambda native: boozer_objective(
                implicit_collocation_polished_state(
                    native,
                    result.context,
                    linear_config,
                )
            )
        )(small_strong_root.native)
        boozer_difference = jax.tree.map(
            jnp.subtract,
            boozer_gradient,
            boozer_adjoint.native_cotangent,
        )
        assert _tree_norm(boozer_difference) <= 2.0e-5 * max(
            _tree_norm(boozer_adjoint.native_cotangent),
            1.0,
        )

        base_stationarity = _collocation_stationarity(
            result.context.correction,
            small_strong_root.native,
            result.context.runtime,
            result.context.chart,
        )

        def stationarity_remainder(step):
            perturbed_native = jax.tree.map(
                lambda value, direction: value + step * direction,
                small_strong_root.native,
                native_tangent,
            )
            perturbed_correction = (
                result.context.correction + step * tangent.correction_tangent
            )
            return jnp.linalg.norm(
                _collocation_stationarity(
                    perturbed_correction,
                    perturbed_native,
                    result.context.runtime,
                    result.context.chart,
                )
                - base_stationarity
            )

        coarse = stationarity_remainder(2.0e-5)
        fine = stationarity_remainder(1.0e-5)
        assert fine < 0.35 * coarse


def test_polish_derivative_failure_is_nan_under_tracing_and_raises_outside():
    """``fail_policy="raise"`` cannot raise on a traced convergence flag.

    Outside jit the policy raises; under jit the flag is a tracer, so both
    policies return NaN. The docstring says so, and this pins it: a jitted
    gradient that fails comes back as NaN, never as an exception.
    """
    import jax
    import jax.numpy as jnp

    from vmex.core.polish_implicit import (
        PolishLinearConfig, PolishLinearReport, StrongForceLinearSolveError,
        _checked_solution,
    )

    def report(converged):
        return PolishLinearReport(
            residual_norm=jnp.asarray(1.0), tolerance=jnp.asarray(1.0e-8),
            iterations=jnp.asarray(30), converged=converged)

    value = jnp.ones(3)
    raising = PolishLinearConfig(fail_policy="raise")

    # eager, failed: the documented exception
    with pytest.raises(StrongForceLinearSolveError):
        _checked_solution(value, report(jnp.asarray(False)), raising, "tangent")
    # eager, converged: the value passes through
    ok = _checked_solution(value, report(jnp.asarray(True)), raising, "tangent")
    assert bool(jnp.all(ok == value))

    # traced, failed: NaN, no exception, whatever the policy says.  The
    # conftest disables jit suite-wide, so ask for it explicitly here --
    # which is also why the interpreted suite never exercised this path.
    def traced(flag):
        return _checked_solution(value, report(flag), raising, "tangent")

    with jax.disable_jit(False):
        assert bool(jnp.all(jnp.isnan(jax.jit(traced)(jnp.asarray(False)))))
        assert bool(jnp.all(jax.jit(traced)(jnp.asarray(True)) == value))

