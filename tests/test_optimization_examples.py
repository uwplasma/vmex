from __future__ import annotations

import importlib


def test_fixed_boundary_qs_examples_import_without_running() -> None:
    modules = [
        "examples.optimization.qh_fixed_resolution_jax",
        "examples.optimization.qa_fixed_resolution_jax_ess",
        "examples.optimization.qp_fixed_resolution_jax_ess",
    ]
    for name in modules:
        module = importlib.import_module(name)
        assert module.CONFIG.max_mode == module.MAX_MODE
        assert module.CONFIG.max_nfev == module.MAX_NFEV
        assert len(module.OBJECTIVES) >= 2


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
