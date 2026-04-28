from __future__ import annotations


def test_finite_beta_examples_wire_top_level_solver_controls() -> None:
    from examples.optimization import qa_optimization_finite_beta as qa
    from examples.optimization import qh_optimization_finite_beta as qh
    from examples.optimization import qi_optimization_finite_beta as qi

    for module in (qa, qh, qi):
        assert module.CONFIG.inner_max_iter == module.INNER_MAX_ITER
        assert module.CONFIG.inner_ftol == module.INNER_FTOL
        assert module.CONFIG.trial_max_iter == module.TRIAL_MAX_ITER
        assert module.CONFIG.trial_ftol == module.TRIAL_FTOL
        assert module.CONFIG.solver_device == module.SOLVER_DEVICE


def test_qi_finite_beta_example_uses_diagnostic_default_grid() -> None:
    from examples.optimization import qi_optimization_finite_beta as qi

    assert qi.CONFIG.qi_mboz == qi.QI_MBOZ == 10
    assert qi.CONFIG.qi_nboz == qi.QI_NBOZ == 10
    assert qi.CONFIG.qi_nphi == qi.QI_NPHI == 32
    assert qi.CONFIG.qi_nalpha == qi.QI_NALPHA == 8
    assert qi.CONFIG.qi_n_bounce == qi.QI_N_BOUNCE == 12
