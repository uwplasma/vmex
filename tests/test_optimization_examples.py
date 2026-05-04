from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_fixed_boundary_qs_examples_are_standalone_workflows() -> None:
    scripts = [
        ROOT / "examples" / "optimization" / "QH_optimization.py",
        ROOT / "examples" / "optimization" / "QA_optimization.py",
        ROOT / "examples" / "optimization" / "QP_optimization.py",
    ]
    for script in scripts:
        text = script.read_text()
        assert 'if __name__ == "__main__"' not in text
        assert "FixedBoundaryQSConfig" not in text
        assert "run_quasi_isodynamic_objective_optimization(" not in text
        assert "run_fixed_boundary_objective_optimization(" not in text
        assert "build_qs_stage(" not in text
        assert "run_qs_stage(" not in text
        assert "run_qs_optimization(" not in text
        assert "FixedBoundaryVMEC.from_input(" in text
        assert "LeastSquaresProblem.from_tuples(" in text
        assert "least_squares_solve(" in text
        assert "problem =" in text


def test_qi_example_uses_qi_problem_api() -> None:
    text = (ROOT / "examples" / "optimization" / "QI_optimization.py").read_text()
    assert "run_quasi_isodynamic_objective_optimization(" not in text
    assert "QuasiIsodynamicOptions(" in text
    assert "QuasiIsodynamicResidual()" in text
    assert "LeastSquaresProblem.from_tuples(" in text
    assert "least_squares_solve(" in text


def test_custom_objective_term_residual_shape() -> None:
    from vmec_jax.optimization_workflow import ObjectiveTerm

    term = ObjectiveTerm(
        "custom",
        evaluate=lambda _ctx, _state: [1.0, 3.0],
        target=1.0,
        weight=2.0,
    )

    residual = term.residual(None, None)

    assert residual.shape == (2,)
    assert [float(x) for x in residual] == [0.0, 4.0]


def test_least_squares_problem_uses_simsopt_weight_semantics() -> None:
    from vmec_jax.optimization_workflow import LeastSquaresProblem

    problem = LeastSquaresProblem.from_tuples([(lambda _ctx, _state: 2.0, 1.0, 4.0)])
    residual = problem.objective_terms[0].residual(None, None)

    assert residual.shape == (1,)
    assert float(residual[0]) == 2.0


def test_least_squares_problem_routes_qi_terms() -> None:
    from vmec_jax.optimization_workflow import LeastSquaresProblem, QuasiIsodynamicResidual

    qi = QuasiIsodynamicResidual()
    problem = LeastSquaresProblem.from_tuples([(qi.J, 0.0, 9.0)])

    assert problem.is_qi
    assert len(problem.objective_terms) == 0
    assert len(problem.qi_objective_terms) == 1
    assert problem.qi_objective_terms[0].name == "qi"


def test_lgradb_tuple_stays_regular_state_objective() -> None:
    from vmec_jax.optimization_workflow import LeastSquaresProblem, LgradB

    lgradb = LgradB(threshold=0.30)
    problem = LeastSquaresProblem.from_tuples([(lgradb.J, 0.0, 0.01)])

    assert not problem.is_qi
    assert len(problem.objective_terms) == 1
    assert problem.objective_terms[0].name == "LgradB"
    assert len(problem.qi_objective_terms) == 0
