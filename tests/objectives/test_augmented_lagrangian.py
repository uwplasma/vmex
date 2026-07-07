from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax import optimization_workflow as workflow
from vmec_jax._compat import jnp
from vmec_jax.quasi_isodynamic import optimization_terms as qi_objectives


class _StateViolation:
    name = "positive_x"

    def J(self, _ctx, state):
        return jnp.asarray([state.x], dtype=jnp.float64)

    def to_objective_term(self, *, target, residual_weight: float):
        return workflow.ObjectiveTerm(
            self.name,
            self.J,
            target=target,
            weight=residual_weight,
            metadata={"synthetic": True},
        )


class _QIFieldViolation:
    name = "qi_field_violation"
    requires_qi_field = True

    def __init__(self, options):
        self.options = options

    def J(self, _ctx, _state):
        raise RuntimeError("QI field test violation must be assembled through LeastSquaresProblem.")

    def to_qi_term(self, residual_weight: float):
        def _evaluate(_ctx, _state, field):
            residual = jnp.asarray(field["violation"], dtype=jnp.float64) * float(residual_weight)
            return residual, jnp.sum(residual * residual)

        return workflow.QIObjectiveTerm(self.name, _evaluate, qi_options=self.options)

    def to_constraint_qi_term(self):
        def _evaluate(_ctx, _state, field):
            residual = jnp.asarray(field["violation"], dtype=jnp.float64)
            return residual, jnp.sum(jnp.maximum(residual, 0.0) ** 2)

        return workflow.QIObjectiveTerm(f"{self.name}_constraint", _evaluate, qi_options=self.options)


class _StateSignedViolation:
    name = "signed_x"

    def to_constraint_term(self):
        def _residual(_ctx, state):
            return jnp.asarray([state.x], dtype=jnp.float64)

        return workflow.ObjectiveTerm(
            self.name,
            _residual,
            target=0.0,
            weight=1.0,
            total=lambda ctx, state: jnp.sum(_residual(ctx, state) ** 2),
            metadata={"signed": True},
        )


class _PreparedStateSignedViolation:
    name = "prepared_signed_x"

    def to_constraint_term(self):
        def _prepare(ctx):
            offset = float(ctx.offset)

            def _residual(_ctx, state):
                return jnp.asarray([state.x + offset], dtype=jnp.float64)

            return workflow.ObjectiveTerm(
                self.name,
                _residual,
                target=0.0,
                weight=1.0,
                metadata={"prepared": True},
            )

        return workflow.ObjectiveTerm(
            self.name,
            lambda _ctx, _state: (_ for _ in ()).throw(RuntimeError("unprepared constraint was used")),
            target=0.0,
            weight=1.0,
            prepare=_prepare,
        )


class _QIFieldPositiveViolation:
    name = "qi_positive_violation"
    requires_qi_field = True

    def __init__(self, options):
        self.options = options

    def to_qi_term(self, residual_weight: float):
        def _evaluate(_ctx, _state, field):
            residual = jnp.asarray(field["violation"], dtype=jnp.float64) * float(residual_weight)
            return residual, jnp.sum(residual * residual)

        return workflow.QIObjectiveTerm(self.name, _evaluate, qi_options=self.options)


def test_augmented_lagrangian_state_constraint_projects_shifted_violation() -> None:
    constraint = workflow.AugmentedLagrangianConstraint(
        _StateViolation(),
        multiplier=0.2,
        penalty=4.0,
        softness=0.0,
    )
    problem = workflow.LeastSquaresProblem.from_tuples([(constraint.J, 0.0, 1.0)])
    term = problem.objective_terms[0]
    ctx = SimpleNamespace()

    np.testing.assert_allclose(np.asarray(term.residual(ctx, SimpleNamespace(x=-0.10))), [0.0])
    np.testing.assert_allclose(np.asarray(term.residual(ctx, SimpleNamespace(x=0.20))), [0.5])
    assert term.name == "al_positive_x"
    assert term.metadata["synthetic"] is True


def test_augmented_lagrangian_direct_methods_and_invalid_objective_branches() -> None:
    constraint = workflow.AugmentedLagrangianConstraint(
        _StateViolation(),
        multiplier=0.0,
        penalty=4.0,
        softness=0.0,
    )
    with pytest.raises(RuntimeError, match="assembled through LeastSquaresProblem"):
        constraint.J(SimpleNamespace(), SimpleNamespace(x=1.0))

    capped = constraint.updated(violation=1.0, penalty_growth=10.0, max_penalty=5.0)
    assert capped.multiplier == pytest.approx(4.0)
    assert capped.penalty == pytest.approx(5.0)

    class NoObjectiveHooks:
        pass

    with pytest.raises(ValueError, match="must expose to_objective_term"):
        workflow.AugmentedLagrangianConstraint(NoObjectiveHooks()).to_objective_term(target=0.0, residual_weight=1.0)
    with pytest.raises(ValueError, match="must expose to_qi_term"):
        workflow.AugmentedLagrangianConstraint(NoObjectiveHooks()).to_qi_term(1.0)


def test_augmented_lagrangian_signed_state_and_qi_fallback_paths() -> None:
    state_term = workflow.AugmentedLagrangianConstraint(
        _StateSignedViolation(),
        multiplier=0.0,
        penalty=4.0,
        softness=0.0,
        name="bounded_x",
    ).to_objective_term(target=0.0, residual_weight=2.0)

    residual = state_term.residual(SimpleNamespace(), SimpleNamespace(x=0.25))
    np.testing.assert_allclose(np.asarray(residual), [1.0])
    assert float(state_term.total(SimpleNamespace(), SimpleNamespace(x=0.25))) == pytest.approx(1.0)
    assert state_term.metadata["signed"] is True

    options = workflow.QuasiIsodynamicOptions(surfaces=np.asarray([0.5]))
    qi_term = workflow.AugmentedLagrangianConstraint(
        _QIFieldPositiveViolation(options),
        multiplier=0.0,
        penalty=9.0,
        softness=0.0,
    ).to_qi_term(2.0)
    residual, total = qi_term.residual_and_total(
        SimpleNamespace(),
        SimpleNamespace(),
        {"violation": jnp.asarray([0.25], dtype=jnp.float64)},
    )
    np.testing.assert_allclose(np.asarray(residual), [1.5])
    assert float(total) == pytest.approx(2.25)


def test_augmented_lagrangian_preserves_prepared_scalar_constraint() -> None:
    term = workflow.AugmentedLagrangianConstraint(
        _PreparedStateSignedViolation(),
        multiplier=0.0,
        penalty=4.0,
        softness=0.0,
    ).to_objective_term(target=0.0, residual_weight=1.0)

    bound = term.bind(SimpleNamespace(offset=0.2))
    residual = bound.residual(SimpleNamespace(offset=99.0), SimpleNamespace(x=0.3))
    np.testing.assert_allclose(np.asarray(residual), [1.0])
    assert bound.metadata["prepared"] is True


def test_augmented_lagrangian_multiplier_update_is_projected_and_can_grow_penalty() -> None:
    constraint = workflow.AugmentedLagrangianConstraint(
        _StateViolation(),
        multiplier=0.2,
        penalty=4.0,
    )

    updated = constraint.updated(violation=0.3, penalty_growth=2.0, max_penalty=6.0)
    assert updated.multiplier == pytest.approx(1.4)
    assert updated.penalty == pytest.approx(6.0)

    projected = constraint.updated(violation=-1.0)
    assert projected.multiplier == pytest.approx(0.2)


def test_augmented_lagrangian_qi_constraint_wraps_shared_field_objective() -> None:
    options = workflow.QuasiIsodynamicOptions(surfaces=np.asarray([0.5]))
    constraint = workflow.AugmentedLagrangianConstraint(
        _QIFieldViolation(options),
        multiplier=0.3,
        penalty=9.0,
        softness=0.0,
    )
    problem = workflow.LeastSquaresProblem.from_tuples([(constraint.J, 0.0, 1.0)])

    assert problem.is_qi
    assert problem.qi_options is options
    term = problem.qi_objective_terms[0]
    residual, total = term.residual_and_total(
        SimpleNamespace(),
        SimpleNamespace(),
        {"violation": jnp.asarray([-0.05, 0.2], dtype=jnp.float64)},
    )

    np.testing.assert_allclose(np.asarray(residual), [0.0, 0.7])
    assert float(total) == pytest.approx(0.49)


def test_augmented_lagrangian_tuple_rejects_nonzero_target() -> None:
    constraint = workflow.AugmentedLagrangianConstraint(_StateViolation())

    with pytest.raises(ValueError, match="target=0"):
        workflow.LeastSquaresProblem.from_tuples([(constraint.J, 1.0, 1.0)])


def test_qi_mirror_and_elongation_constraints_expose_signed_residuals(monkeypatch) -> None:
    options = workflow.QuasiIsodynamicOptions(surfaces=np.asarray([0.5]))
    ctx = SimpleNamespace(static=SimpleNamespace(cfg=SimpleNamespace(nfp=2)))

    def fake_mirror_ratio_penalty_from_boozer_output(_booz, **kwargs):
        assert kwargs["threshold"] == 0.3
        return {
            "mirror_ratio": jnp.asarray([0.2, 0.5], dtype=jnp.float64),
            "residuals1d": jnp.asarray([0.0, 0.2], dtype=jnp.float64),
            "total": jnp.asarray(0.04, dtype=jnp.float64),
        }

    monkeypatch.setattr(
        qi_objectives,
        "mirror_ratio_penalty_from_boozer_output",
        fake_mirror_ratio_penalty_from_boozer_output,
    )
    mirror = workflow.MirrorRatio(threshold=0.3, qi_options=options)
    mirror_term = workflow.AugmentedLagrangianConstraint(
        mirror,
        multiplier=0.0,
        penalty=4.0,
    ).to_qi_term(1.0)
    residual, _total = mirror_term.residual_and_total(
        ctx,
        SimpleNamespace(),
        {"booz": {"bmnc_b": np.zeros((2, 3))}},
    )
    np.testing.assert_allclose(np.asarray(residual), [0.0, 2.0 * 0.2 / np.sqrt(2.0)])

    def fake_max_elongation_penalty_from_state(**kwargs):
        assert kwargs["smooth_extrema"] == 0.1
        return {
            "max_elongation": jnp.asarray(8.5, dtype=jnp.float64),
            "residuals1d": jnp.asarray([0.5], dtype=jnp.float64),
            "total": jnp.asarray(0.25, dtype=jnp.float64),
        }

    monkeypatch.setattr(qi_objectives, "max_elongation_penalty_from_state", fake_max_elongation_penalty_from_state)
    elongation = workflow.MaxElongation(threshold=8.0, smooth_extrema=0.1, qi_options=options)
    elongation_term = workflow.AugmentedLagrangianConstraint(elongation, penalty=9.0).to_qi_term(1.0)
    residual, _total = elongation_term.residual_and_total(ctx, "state", {})
    np.testing.assert_allclose(np.asarray(residual), [1.5])
