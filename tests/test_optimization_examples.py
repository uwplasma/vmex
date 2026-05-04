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
        assert "target_aspect=" not in text
        assert "target_iota=" not in text
        assert "iota_abs_min=" not in text
        assert "qi_options=" not in text
        assert "plot=" not in text
        assert "print_optimization_outputs" not in text
        assert "result.final_result" in text
        assert "vmecplot2_bmag_grid(" in text


def test_qi_example_uses_qi_problem_api() -> None:
    text = (ROOT / "examples" / "optimization" / "QI_optimization.py").read_text()
    assert "run_quasi_isodynamic_objective_optimization(" not in text
    assert "QuasiIsodynamicOptions(" in text
    assert "QuasiIsodynamicResidual(QI_OPTIONS)" in text
    assert "LeastSquaresProblem.from_tuples(" in text
    assert "least_squares_solve(" in text
    assert "target_aspect=" not in text
    assert "iota_abs_min=" not in text
    assert "qi_options=" not in text
    assert "plot=" not in text
    assert "print_optimization_outputs" not in text
    assert "result.final_result" in text
    assert "vmecplot2_bmag_grid(" in text


def test_finite_beta_examples_plot_explicitly_after_solve() -> None:
    scripts = [
        ROOT / "examples" / "optimization" / "qa_optimization_finite_beta.py",
        ROOT / "examples" / "optimization" / "qh_optimization_finite_beta.py",
        ROOT / "examples" / "optimization" / "qi_optimization_finite_beta.py",
    ]
    for script in scripts:
        text = script.read_text()
        assert "save_final_outputs(" in text
        assert "plot=" not in text
        assert "vj.plot_qh_optimization(" in text


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
    from vmec_jax.optimization_workflow import LeastSquaresProblem, QuasiIsodynamicOptions, QuasiIsodynamicResidual

    qi_options = QuasiIsodynamicOptions(surfaces=[0.5])
    qi = QuasiIsodynamicResidual(qi_options)
    problem = LeastSquaresProblem.from_tuples([(qi.J, 0.0, 9.0)])

    assert problem.is_qi
    assert len(problem.objective_terms) == 0
    assert len(problem.qi_objective_terms) == 1
    assert problem.qi_objective_terms[0].name == "qi"
    assert problem.qi_options is qi_options


def test_least_squares_problem_collects_problem_metadata() -> None:
    from vmec_jax.optimization_workflow import AbsMeanIotaFloor, AspectRatio, LeastSquaresProblem, MeanIota

    problem = LeastSquaresProblem.from_tuples(
        [
            (AspectRatio().J, 5.0, 1.0),
            (MeanIota().J, 0.42, 100.0),
            (AbsMeanIotaFloor(0.41).J, 0.0, 1.0),
        ]
    )

    assert problem.metadata == {
        "target_aspect": 5.0,
        "target_iota": 0.42,
        "iota_abs_min": 0.41,
    }


def test_lgradb_tuple_stays_regular_state_objective() -> None:
    from vmec_jax.optimization_workflow import LeastSquaresProblem, LgradB

    lgradb = LgradB(threshold=0.30)
    problem = LeastSquaresProblem.from_tuples([(lgradb.J, 0.0, 0.01)])

    assert not problem.is_qi
    assert len(problem.objective_terms) == 1
    assert problem.objective_terms[0].name == "LgradB"
    assert len(problem.qi_objective_terms) == 0
