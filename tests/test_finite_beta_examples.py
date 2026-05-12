from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = ROOT / "examples" / "optimization"

FINITE_BETA_SCRIPTS = (
    EXAMPLE_DIR / "qa_optimization_finite_beta.py",
    EXAMPLE_DIR / "qh_optimization_finite_beta.py",
    EXAMPLE_DIR / "qi_optimization_finite_beta.py",
)


def _literal_assignments(path: Path) -> dict[str, object]:
    tree = ast.parse(path.read_text())
    values: dict[str, object] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                continue
    return values


def test_finite_beta_examples_are_standalone_visible_workflows() -> None:
    for script in FINITE_BETA_SCRIPTS:
        text = script.read_text()

        assert "FiniteBetaStage1Config" not in text
        assert "run_stage1(" not in text
        assert 'if __name__ == "__main__"' not in text

        assert "vj.load_config(" in text
        assert "vj.build_static(" in text
        assert "vj.FiniteBetaTargets(" in text
        assert "def residuals_from_state" in text
        assert "finite_beta_global_residuals_from_state" in text
        assert "vj.FixedBoundaryExactOptimizer(" in text
        assert "optimizer.run(" in text
        assert "stage1_result = finite_beta_stage1_result(stage_records)" in text
        assert "stage_summaries = stage1_result.stage_summaries" in text
        assert "final_summary = stage1_result.final_summary" in text
        assert "save_final_outputs(" in text


def test_finite_beta_examples_wire_top_level_solver_controls() -> None:
    for script in FINITE_BETA_SCRIPTS:
        text = script.read_text()

        assert "inner_max_iter=INNER_MAX_ITER" in text
        assert "inner_ftol=INNER_FTOL" in text
        assert "trial_max_iter=TRIAL_MAX_ITER" in text
        assert "trial_ftol=TRIAL_FTOL" in text
        assert "solver_device=SOLVER_DEVICE" in text


def test_qi_finite_beta_example_uses_diagnostic_default_grid() -> None:
    values = _literal_assignments(EXAMPLE_DIR / "qi_optimization_finite_beta.py")

    assert values["QI_MBOZ"] == 10
    assert values["QI_NBOZ"] == 10
    assert values["QI_NPHI"] == 32
    assert values["QI_NALPHA"] == 8
    assert values["QI_N_BOUNCE"] == 12


def test_finite_beta_stage1_result_adapts_raw_stage_tuples() -> None:
    from examples.optimization.finite_beta_stage1_common import finite_beta_stage1_result

    initial_optimizer = object()
    final_optimizer = object()
    initial_result = {
        "x": np.asarray([0.1]),
        "nfev": 1,
        "success": True,
        "_history_dump": {
            "method": "scipy",
            "solver_device": "cpu",
            "max_nfev": 2,
            "nfev": 1,
            "njev": 1,
            "success": True,
            "message": "initial done",
            "objective_initial": 8.0,
            "objective_final": 4.0,
            "qs_initial": 3.0,
            "qs_final": 2.0,
            "aspect_initial": 5.0,
            "aspect_final": 5.2,
            "iota_initial": 0.3,
            "iota_final": 0.35,
            "total_wall_time_s": 1.25,
            "selected_best_exact_point": False,
        },
    }
    final_result = {
        "x": np.asarray([0.2, 0.3]),
        "nfev": 3,
        "njev": 2,
        "success": False,
        "status": 0,
        "message": "budget",
        "_history_dump": {
            "method": "scipy",
            "solver_device": "default",
            "max_nfev": 4,
            "nfev": 3,
            "njev": 2,
            "success": False,
            "message": "budget",
            "objective_initial": 4.0,
            "objective_final": 1.5,
            "qs_initial": 2.0,
            "qs_final": 0.5,
            "aspect_initial": 5.2,
            "aspect_final": 5.4,
            "iota_initial": 0.35,
            "iota_final": 0.42,
            "total_wall_time_s": 2.5,
            "selected_best_exact_point": True,
        },
    }

    result = finite_beta_stage1_result(
        [
            (1, initial_optimizer, np.asarray([0.0]), initial_result),
            (2, final_optimizer, np.asarray([0.1, 0.0]), final_result),
        ]
    )

    assert result.final_optimizer is final_optimizer
    assert result.final_result is final_result
    np.testing.assert_allclose(result.final_params, [0.2, 0.3])
    assert [summary.mode for summary in result.stage_summaries] == [1, 2]
    assert result.final_summary.objective_final == 1.5
    assert result.final_summary.aspect_final == 5.4
    assert result.final_summary.iota_final == 0.42
    assert result.final_summary.n_params == 2
    assert result.summary["final"]["selected_best_exact_point"] is True


def test_finite_beta_stage_summary_uses_optimizer_result_fallbacks() -> None:
    from examples.optimization.finite_beta_stage1_common import finite_beta_stage_summary

    summary = finite_beta_stage_summary(
        (
            3,
            object(),
            np.asarray([0.0, 0.1, 0.2]),
            {
                "x": np.asarray([0.3, 0.4, 0.5]),
                "objective": np.float64(2.25),
                "nfev": np.int64(5),
                "njev": np.int64(4),
                "nit": np.int64(2),
                "success": np.bool_(True),
                "status": np.int64(1),
                "message": "fallback done",
                "_history_dump": {
                    "history": [
                        {"objective": 4.0, "iota": 0.33},
                        {"objective": 2.25, "iota": 0.44},
                    ],
                },
            },
        )
    )

    assert summary.mode == 3
    assert summary.objective_final == 2.25
    assert summary.iota_final == 0.44
    assert summary.nfev == 5
    assert summary.njev == 4
    assert summary.nit == 2
    assert summary.success is True
    assert summary.status == 1
    assert summary.message == "fallback done"
    assert summary.n_params == 3


def test_finite_beta_stage1_result_rejects_empty_stage_records() -> None:
    from examples.optimization.finite_beta_stage1_common import finite_beta_stage1_result

    with pytest.raises(ValueError, match="requires at least one stage record"):
        finite_beta_stage1_result([])
