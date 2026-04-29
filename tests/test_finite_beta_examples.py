from __future__ import annotations

import ast
from pathlib import Path


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
