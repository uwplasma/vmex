from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


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
        assert "plot_3d_boundary_comparison(" in text
        assert "plot_bmag_contours(" in text
        assert "plot_objective_history(" in text


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
    assert "plot_3d_boundary_comparison(" in text
    assert "plot_bmag_contours(" in text
    assert "plot_objective_history(" in text


def test_qi_objective_comparison_is_top_level_diagnostic() -> None:
    text = (ROOT / "examples" / "optimization" / "compare_omnigenity_qi_objective.py").read_text()

    assert "argparse" not in text
    assert "QI_VARIANTS" in text
    assert "PHIMIN_FACTORS" in text
    assert "QuasiIsodynamicResidual" in text
    assert "legacy_qi_branch_shuffle_diagnostic_from_boozer_output" in text
    assert "quasi_isodynamic_residual_from_state(" in text


def test_qs_sweep_reports_true_legacy_qi_metric() -> None:
    text = (ROOT / "examples" / "optimization" / "generate_qs_ess_sweep.py").read_text()

    assert "legacy_qi_branch_shuffle_diagnostic_from_boozer_output" in text
    assert '"qi_legacy_total": qi_total' not in text


def test_policy_matrix_plots_single_problem(tmp_path, monkeypatch) -> None:
    pytest.importorskip("matplotlib")

    from examples.optimization import compare_qs_policy_matrix as matrix

    monkeypatch.setattr(matrix, "PROBLEMS", ("qa",))

    outpath = tmp_path / "one_problem_matrix.png"
    matrix._plot_policy_matrix_all([], outpath=outpath)

    assert outpath.exists()


def test_qs_sweep_history_merge_preserves_stage_profiles_and_traces() -> None:
    from examples.optimization.generate_qs_ess_sweep import PROBLEM_CONFIGS, _merge_stage_histories

    def stage_result(label: str, wall: float) -> dict:
        return {
            "_history_dump": {
                "history": [
                    {"wall_time_s": 0.0, "objective": 2.0, "qs_objective": 1.0, "aspect": 5.0},
                    {"wall_time_s": wall, "objective": 1.0, "qs_objective": 0.5, "aspect": 5.1},
                ],
                "nfev": 2,
                "njev": 1,
                "success": True,
                "message": label,
                "objective_initial": 2.0,
                "objective_final": 1.0,
                "qs_initial": 1.0,
                "qs_final": 0.5,
                "aspect_initial": 5.0,
                "aspect_final": 5.1,
                "max_nfev": 3,
                "profile": {
                    "exact_tape_build": {"count": 1, "wall_time_s": wall, "mean_wall_time_s": wall},
                    "trial_solve": {"count": 2, "wall_time_s": 2.0 * wall, "mean_wall_time_s": wall},
                },
                "callback_trace": {
                    "enabled": True,
                    "events": [{"index": 0, "kind": "jacobian", "source": "exact_tape_replay", "wall_time_s": wall}],
                    "summary": {"jacobian:exact_tape_replay": {"count": 1, "wall_time_s": wall}},
                },
            }
        }

    merged = _merge_stage_histories(
        [
            ("stage 1", 1, stage_result("one", 0.25)),
            ("stage 2", 2, stage_result("two", 0.5)),
        ],
        problem_cfg=PROBLEM_CONFIGS["qa"],
    )

    assert merged["profile"]["exact_tape_build"]["count"] == 2
    assert merged["profile"]["exact_tape_build"]["wall_time_s"] == 0.75
    assert merged["profile"]["trial_solve"]["count"] == 4
    assert merged["profile"]["trial_solve"]["wall_time_s"] == 1.5
    assert len(merged["stage_profiles"]) == 2
    assert merged["callback_trace"]["summary"]["jacobian:exact_tape_replay"]["count"] == 2
    assert merged["callback_trace"]["summary"]["jacobian:exact_tape_replay"]["wall_time_s"] == 0.75
    assert [event["stage"] for event in merged["callback_trace"]["events"]] == ["stage 1", "stage 2"]


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
        assert "vj.plot_3d_boundary_comparison(" in text
        assert "vj.plot_bmag_contours(" in text
        assert "vj.plot_objective_history(" in text


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


def test_dmerc_tuple_stays_regular_state_objective() -> None:
    from vmec_jax.optimization_workflow import DMerc, LeastSquaresProblem

    problem = LeastSquaresProblem.from_tuples([(DMerc().J, 0.0, 0.25)])

    assert not problem.is_qi
    assert len(problem.objective_terms) == 1
    assert problem.objective_terms[0].name == "DMerc"
    assert len(problem.qi_objective_terms) == 0


def test_jxbforce_profile_tuple_stays_regular_state_objective() -> None:
    from vmec_jax.optimization_workflow import (
        BDotB,
        BDotGradV,
        JDotB,
        LeastSquaresProblem,
        ToroidalCurrent,
        ToroidalCurrentGradient,
    )

    problem = LeastSquaresProblem.from_tuples(
        [
            (JDotB(surfaces=(0.25, 0.75)).J, 0.0, 0.25),
            (BDotB(surfaces=(0.5,)).J, 1.0, 0.10),
            (BDotGradV().J, 0.0, 0.05),
            (ToroidalCurrent(surfaces=(0.75,)).J, 0.0, 0.20),
            (ToroidalCurrentGradient().J, 0.0, 0.30),
        ]
    )

    assert not problem.is_qi
    assert [term.name for term in problem.objective_terms] == [
        "jdotb",
        "bdotb",
        "bdotgradv",
        "torcur",
        "torcur_prime",
    ]
    assert len(problem.qi_objective_terms) == 0


def test_finite_beta_workflow_objectives_are_jax_differentiable(monkeypatch) -> None:
    pytest.importorskip("jax")
    import jax

    from vmec_jax._compat import jnp
    import vmec_jax.optimization_workflow as workflow
    from vmec_jax.optimization_workflow import (
        BDotB,
        BDotGradV,
        BetaTotal,
        DMerc,
        JDotB,
        MagneticWell,
        ToroidalCurrent,
        ToroidalCurrentGradient,
        VolavgB,
    )

    def fake_scalars_from_state(*, state, **_kwargs):
        scale = jnp.asarray(state, dtype=jnp.float64)
        return {
            "volavgB": 2.0 + 0.5 * scale,
            "betatotal": 0.03 + 0.01 * scale,
            "vp": jnp.asarray([0.0, 1.0 + scale, 0.8, 0.6], dtype=jnp.float64),
        }

    def fake_mercier_terms_from_state(*, state, **_kwargs):
        scale = jnp.asarray(state, dtype=jnp.float64)
        return {
            "DMerc": jnp.asarray(
                [0.0, 0.02 + 0.01 * scale, -0.03 + 0.02 * scale, 0.0],
                dtype=jnp.float64,
            ),
            "jdotb": jnp.asarray([0.0, 0.10 + 0.01 * scale, 0.20 + 0.02 * scale, 0.0], dtype=jnp.float64),
            "bdotb": jnp.asarray([0.0, 1.00 + 0.10 * scale, 1.20 + 0.20 * scale, 0.0], dtype=jnp.float64),
            "bdotgradv": jnp.asarray([0.0, 2.00 + 0.20 * scale, 2.20 + 0.30 * scale, 0.0], dtype=jnp.float64),
            "torcur": jnp.asarray([0.0, 0.40 + 0.04 * scale, 0.60 + 0.06 * scale, 0.0], dtype=jnp.float64),
            "ip": jnp.asarray([0.0, 1.40 + 0.14 * scale, 1.60 + 0.16 * scale, 0.0], dtype=jnp.float64),
        }

    monkeypatch.setattr(workflow, "finite_beta_scalars_from_state", fake_scalars_from_state)
    monkeypatch.setattr(workflow, "mercier_terms_from_state", fake_mercier_terms_from_state)
    ctx = SimpleNamespace(static=SimpleNamespace(s=np.asarray([0.0, 0.25, 0.75, 1.0])), indata=None, signgs=1)

    vol_value, vol_grad = jax.value_and_grad(lambda x: VolavgB().J(ctx, x))(jnp.asarray(1.0))
    beta_value, beta_grad = jax.value_and_grad(lambda x: BetaTotal().J(ctx, x))(jnp.asarray(1.0))
    well = MagneticWell(minimum=0.7, softness=1.0e-2)
    well_value, well_grad = jax.value_and_grad(lambda x: well.J(ctx, x))(jnp.asarray(1.0))
    dmerc = DMerc(minimum=0.0, softness=1.0e-2)
    dmerc_value, dmerc_grad = jax.value_and_grad(lambda x: jnp.sum(dmerc.J(ctx, x)))(jnp.asarray(1.0))
    jdotb_value, jdotb_grad = jax.value_and_grad(lambda x: jnp.sum(JDotB(surfaces=(0.25, 0.75)).J(ctx, x)))(
        jnp.asarray(1.0)
    )
    bdotb_value, bdotb_grad = jax.value_and_grad(lambda x: jnp.sum(BDotB().J(ctx, x)))(jnp.asarray(1.0))
    bdotgradv_value, bdotgradv_grad = jax.value_and_grad(lambda x: jnp.sum(BDotGradV().J(ctx, x)))(
        jnp.asarray(1.0)
    )
    torcur_value, torcur_grad = jax.value_and_grad(lambda x: jnp.sum(ToroidalCurrent().J(ctx, x)))(jnp.asarray(1.0))
    torcur_prime_value, torcur_prime_grad = jax.value_and_grad(
        lambda x: jnp.sum(ToroidalCurrentGradient(surfaces=(0.25, 0.75)).J(ctx, x))
    )(jnp.asarray(1.0))

    np.testing.assert_allclose(np.asarray(vol_value), 2.5)
    np.testing.assert_allclose(np.asarray(vol_grad), 0.5)
    np.testing.assert_allclose(np.asarray(beta_value), 0.04)
    np.testing.assert_allclose(np.asarray(beta_grad), 0.01)
    assert np.isfinite(np.asarray(well_value))
    assert np.isfinite(np.asarray(well_grad))
    assert np.isfinite(np.asarray(dmerc_value))
    assert np.isfinite(np.asarray(dmerc_grad))
    assert abs(float(np.asarray(dmerc_grad))) > 0.0
    np.testing.assert_allclose(np.asarray(jdotb_value), 0.33)
    np.testing.assert_allclose(np.asarray(jdotb_grad), 0.03)
    np.testing.assert_allclose(np.asarray(bdotb_value), 2.5)
    np.testing.assert_allclose(np.asarray(bdotb_grad), 0.3)
    np.testing.assert_allclose(np.asarray(bdotgradv_value), 4.7)
    np.testing.assert_allclose(np.asarray(bdotgradv_grad), 0.5)
    np.testing.assert_allclose(np.asarray(torcur_value), 1.10)
    np.testing.assert_allclose(np.asarray(torcur_grad), 0.10)
    np.testing.assert_allclose(np.asarray(torcur_prime_value), 3.30)
    np.testing.assert_allclose(np.asarray(torcur_prime_grad), 0.30)


def test_jxbforce_and_current_objective_gradients_match_finite_difference(monkeypatch) -> None:
    pytest.importorskip("jax")

    from vmec_jax._compat import jnp
    import vmec_jax.optimization_workflow as workflow
    from vmec_jax.optimization_workflow import JDotB, ToroidalCurrent

    def fake_mercier_terms_from_state(*, state, **_kwargs):
        x = jnp.asarray(state, dtype=jnp.float64)
        return {
            "jdotb": jnp.asarray([0.0, 0.1 + 0.02 * x**2, 0.2 + 0.03 * x**2, 0.0], dtype=jnp.float64),
            "torcur": jnp.asarray([0.0, 0.4 + 0.04 * x**2, 0.6 + 0.06 * x**2, 0.0], dtype=jnp.float64),
        }

    monkeypatch.setattr(workflow, "mercier_terms_from_state", fake_mercier_terms_from_state)
    ctx = SimpleNamespace(static=SimpleNamespace(s=np.asarray([0.0, 0.25, 0.75, 1.0])), indata=None, signgs=1)

    def centered_fd(fn, x0, eps=1.0e-6):
        return (float(fn(x0 + eps)) - float(fn(x0 - eps))) / (2.0 * eps)

    import jax

    for objective in (JDotB(surfaces=(0.25, 0.75)), ToroidalCurrent(surfaces=(0.25, 0.75))):
        fn = lambda x, objective=objective: jnp.sum(objective.J(ctx, jnp.asarray(x, dtype=jnp.float64)))
        ad_grad = float(jax.grad(fn)(jnp.asarray(1.3, dtype=jnp.float64)))
        fd_grad = centered_fd(fn, 1.3)
        np.testing.assert_allclose(ad_grad, fd_grad, rtol=1e-6, atol=1e-8)
