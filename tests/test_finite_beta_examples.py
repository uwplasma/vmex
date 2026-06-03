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


def test_finite_beta_examples_share_standard_pressure_and_bootstrap_profiles() -> None:
    for script in FINITE_BETA_SCRIPTS:
        text = script.read_text()

        assert "BETA_PERCENT = 2.5" in text
        assert "STANDARD_PROFILES = vj.standard_finite_beta_profiles(BETA_PERCENT)" in text
        assert "vj.profile_to_power_series_coeffs(STANDARD_PROFILES.ne)" in text
        assert "vj.profile_to_power_series_coeffs(STANDARD_PROFILES.Te)" in text
        assert "apply_finite_beta_pressure_profile(indata, beta_percent=BETA_PERCENT)" in text
        assert "redl_bootstrap_mismatch_from_state" in text


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


def test_finite_beta_save_final_outputs_uses_selected_best_exact_result(tmp_path) -> None:
    from examples.optimization.finite_beta_stage1_common import save_final_outputs

    saved = []

    class RecordingOptimizer:
        def __init__(self, label: str):
            self.label = label

        def save_input(self, path, params):
            saved.append((self.label, "input", path.name, np.asarray(params, dtype=float).copy(), None))

        def save_wout(self, path, params, *, state=None):
            saved.append((self.label, "wout", path.name, np.asarray(params, dtype=float).copy(), state))

        def save_history(self, path, result):
            saved.append((self.label, "history", path.name, np.asarray(result["x"], dtype=float).copy(), result))

    initial_state = object()
    final_state = object()
    initial_optimizer = RecordingOptimizer("initial")
    final_optimizer = RecordingOptimizer("final")
    initial_params = np.asarray([0.0, 0.0])
    final_params = np.asarray([1.0, 1.0e-5])
    initial_result = {
        "x": np.asarray([99.0, 99.0]),
        "_state_initial": initial_state,
        "_state_final": object(),
        "_history_dump": {"selected_best_exact_point": False},
    }
    final_result = {
        "x": final_params,
        "_state_final": final_state,
        "_history_dump": {"selected_best_exact_point": True},
    }

    save_final_outputs(
        output_dir=tmp_path,
        stage_records=[(1, initial_optimizer, initial_params, initial_result)],
        final_optimizer=final_optimizer,
        final_result=final_result,
    )

    initial_wout = next(item for item in saved if item[:3] == ("initial", "wout", "wout_initial.nc"))
    final_input = next(item for item in saved if item[:3] == ("final", "input", "input.final"))
    final_wout = next(item for item in saved if item[:3] == ("final", "wout", "wout_final.nc"))
    final_history = next(item for item in saved if item[:3] == ("final", "history", "history.json"))

    np.testing.assert_allclose(initial_wout[3], initial_params)
    assert initial_wout[4] is initial_state
    np.testing.assert_allclose(final_input[3], final_params)
    np.testing.assert_allclose(final_wout[3], final_params)
    assert final_wout[4] is final_state
    assert final_history[4] is final_result


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
