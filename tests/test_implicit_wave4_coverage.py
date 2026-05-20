from __future__ import annotations

import numpy as np
import pytest

from vmec_jax._compat import has_jax


pytestmark = pytest.mark.skipif(not has_jax(), reason="implicit helpers require JAX")


def test_flatten_unflatten_lambda_blocks_round_trip_in_cos_then_sin_order():
    from vmec_jax._compat import jnp
    from vmec_jax.implicit import _flatten_L, _unflatten_L

    lcos = jnp.arange(6.0).reshape(2, 3)
    lsin = lcos + 10.0

    flat = _flatten_L(lcos, lsin)
    out_lcos, out_lsin = _unflatten_L(flat, shape=(2, 3))

    assert flat.shape == (12,)
    np.testing.assert_allclose(np.asarray(flat[:6]), np.asarray(lcos).reshape(-1))
    np.testing.assert_allclose(np.asarray(flat[6:]), np.asarray(lsin).reshape(-1))
    np.testing.assert_allclose(np.asarray(out_lcos), np.asarray(lcos))
    np.testing.assert_allclose(np.asarray(out_lsin), np.asarray(lsin))


def test_cg_solve_matches_spd_solution_and_accepts_exact_initial_guess():
    from vmec_jax._compat import jnp
    from vmec_jax.implicit import _cg_solve

    matrix = jnp.asarray([[4.0, 1.0], [1.0, 3.0]])
    rhs = jnp.asarray([1.0, 2.0])
    expected = np.linalg.solve(np.asarray(matrix), np.asarray(rhs))

    out = _cg_solve(lambda x: matrix @ x, rhs, tol=1e-14, max_iter=8)
    exact_x0 = _cg_solve(lambda x: matrix @ x, rhs, x0=jnp.asarray(expected), tol=1e-14, max_iter=8)

    np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(exact_x0), expected, rtol=1e-12, atol=1e-12)


def test_implicit_entry_points_report_jax_import_guard(monkeypatch):
    import vmec_jax.implicit as implicit

    monkeypatch.setattr(implicit, "has_jax", lambda: False)

    with pytest.raises(ImportError, match="solve_lambda_state_implicit requires JAX"):
        implicit.solve_lambda_state_implicit(object(), object(), phipf=1.0, chipf=1.0, signgs=1, lamscale=1.0)

    with pytest.raises(ImportError, match="solve_fixed_boundary_state_implicit requires JAX"):
        implicit.solve_fixed_boundary_state_implicit(
            object(),
            object(),
            phipf=1.0,
            chipf=1.0,
            signgs=1,
            lamscale=1.0,
            pressure=0.0,
        )


def test_fixed_boundary_implicit_rejects_unknown_solver_before_building_state():
    from vmec_jax.implicit import solve_fixed_boundary_state_implicit

    with pytest.raises(ValueError, match="solver must be 'gd' or 'lbfgs'"):
        solve_fixed_boundary_state_implicit(
            object(),
            object(),
            phipf=1.0,
            chipf=1.0,
            signgs=1,
            lamscale=1.0,
            pressure=0.0,
            solver="newton",
        )


def test_cg_solve_computes_adjoint_sensitivity_for_linear_implicit_system():
    from vmec_jax._compat import jnp
    from vmec_jax.implicit import _cg_solve

    hessian = jnp.asarray([[5.0, 1.0], [1.0, 2.0]])
    param_jacobian = jnp.asarray([[2.0, -1.0], [0.5, 3.0]])
    cotangent = jnp.asarray([4.0, -2.0])

    adjoint = _cg_solve(lambda x: hessian @ x, cotangent, tol=1e-14, max_iter=8)
    actual_grad = -(param_jacobian.T @ adjoint)
    expected_grad = -(np.asarray(param_jacobian).T @ np.linalg.solve(np.asarray(hessian), np.asarray(cotangent)))

    np.testing.assert_allclose(np.asarray(actual_grad), expected_grad, rtol=1e-12, atol=1e-12)
