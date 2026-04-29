from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_fixed_boundary_qs_examples_are_standalone_workflows() -> None:
    scripts = [
        ROOT / "examples" / "optimization" / "qh_fixed_resolution_jax.py",
        ROOT / "examples" / "optimization" / "qa_fixed_resolution_jax_ess.py",
        ROOT / "examples" / "optimization" / "qp_fixed_resolution_jax_ess.py",
    ]
    for script in scripts:
        text = script.read_text()
        assert 'if __name__ == "__main__"' not in text
        assert "FixedBoundaryQSConfig" not in text
        assert "build_qs_stage(" not in text
        assert "run_qs_stage(" not in text
        assert "run_qs_optimization(" not in text
        assert "OBJECTIVES = [" in text
        assert "cfg, indata = vj.load_config" in text
        assert "vj.build_static(" in text
        assert "vj.boundary_param_specs(" in text
        assert "def residuals_from_state" in text
        assert "vj.FixedBoundaryExactOptimizer(" in text
        assert "optimizer.run(" in text
        assert "for stage_mode in stage_modes:" in text
        assert "save_qs_final_outputs(" in text


def test_custom_objective_term_residual_shape() -> None:
    from examples.optimization.fixed_boundary_qs_common import ObjectiveTerm

    term = ObjectiveTerm(
        "custom",
        evaluate=lambda _ctx, _state: [1.0, 3.0],
        target=1.0,
        weight=2.0,
    )

    residual = term.residual(None, None)

    assert residual.shape == (2,)
    assert [float(x) for x in residual] == [0.0, 4.0]
