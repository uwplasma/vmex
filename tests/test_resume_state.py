import numpy as np
import pytest

import vmec_jax.api as vj


@pytest.mark.py311_slow_coverage
def test_resume_state_matches_continuous():
    input_path = "examples/data/input.circular_tokamak"
    max_iter = 4
    split_iter = 2

    full = vj.run_fixed_boundary(
        input_path,
        solver="vmec2000_iter",
        max_iter=max_iter,
        multigrid=False,
        ns_override=13,
        verbose=False,
        performance_mode=False,
        use_scan=False,
    )
    run1 = vj.run_fixed_boundary(
        input_path,
        solver="vmec2000_iter",
        max_iter=split_iter,
        multigrid=False,
        ns_override=13,
        verbose=False,
        performance_mode=False,
        use_scan=False,
    )
    resume_state = run1.result.diagnostics["resume_state"]
    run2 = vj.run_fixed_boundary(
        input_path,
        solver="vmec2000_iter",
        max_iter=max_iter - split_iter,
        multigrid=False,
        ns_override=13,
        restart_state=run1.state,
        restart_solver_state=resume_state,
        verbose=False,
        performance_mode=False,
        use_scan=False,
    )

    def _cat(a, b):
        return np.concatenate([np.asarray(a), np.asarray(b)], axis=0)

    np.testing.assert_allclose(
        np.asarray(full.result.w_history),
        _cat(run1.result.w_history, run2.result.w_history),
        rtol=1e-8,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.asarray(full.result.fsqr2_history),
        _cat(run1.result.fsqr2_history, run2.result.fsqr2_history),
        rtol=1e-8,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.asarray(full.result.fsqz2_history),
        _cat(run1.result.fsqz2_history, run2.result.fsqz2_history),
        rtol=1e-8,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.asarray(full.result.fsql2_history),
        _cat(run1.result.fsql2_history, run2.result.fsql2_history),
        rtol=1e-8,
        atol=1e-10,
    )


def test_accelerated_resume_state_is_minimal_and_restartable():
    input_path = "examples/data/input.circular_tokamak"

    run1 = vj.run_fixed_boundary(
        input_path,
        solver="vmec2000_iter",
        solver_mode="accelerated",
        max_iter=2,
        multigrid=False,
        ns_override=13,
        verbose=False,
    )
    resume_state = run1.result.diagnostics["resume_state"]
    assert run1.result.diagnostics["resume_state_mode"] == "minimal"
    assert "vRcc" not in resume_state
    assert "cache_precond_diag" not in resume_state

    run2 = vj.run_fixed_boundary(
        input_path,
        solver="vmec2000_iter",
        solver_mode="accelerated",
        max_iter=2,
        multigrid=False,
        ns_override=13,
        restart_state=run1.state,
        restart_solver_state=resume_state,
        verbose=False,
    )

    assert np.isfinite(np.asarray(run2.result.w_history)).all()
    assert np.isfinite(np.asarray(run2.result.fsqr2_history)).all()
    assert run2.result.diagnostics["resume_state_mode"] == "minimal"
