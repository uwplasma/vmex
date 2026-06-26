from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax._compat import has_jax


pytestmark = pytest.mark.skipif(not has_jax(), reason="implicit residual adjoint helpers require JAX")


def _dense_host(J, b, damping):
    J_np = np.asarray(J)
    b_np = np.asarray(b)
    damping_np = float(np.asarray(damping))
    lhs = J_np @ J_np.T + damping_np * np.eye(int(J_np.shape[0]))
    return np.linalg.solve(lhs, J_np @ b_np)


def _not_traced(*_args):
    return False


def test_linear_map_jacobian_columns_chunks_rectangular_map():
    from vmec_jax._compat import jnp
    from vmec_jax.solvers.fixed_boundary.adjoint.residual_linear_algebra import linear_map_jacobian_columns

    matrix = jnp.asarray([[1.0, -2.0, 0.5], [0.25, 3.0, -1.5]])
    jac = linear_map_jacobian_columns(
        lambda x: matrix @ x,
        input_size=3,
        output_size=2,
        dtype=matrix.dtype,
        chunk_size=2,
    )

    np.testing.assert_allclose(np.asarray(jac), np.asarray(matrix), rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        linear_map_jacobian_columns(lambda x: x, input_size=1, output_size=1, dtype=matrix.dtype, chunk_size=0)


def test_lineax_bicgstab_wrapper_marks_missing_and_host_read_failure(monkeypatch):
    from vmec_jax._compat import jnp
    from vmec_jax.solvers.fixed_boundary.adjoint.residual_linear_algebra import lineax_bicgstab_solve

    value, success, stats = lineax_bicgstab_solve(lambda x: x, jnp.ones(2), tol=1e-8, max_iter=3)
    assert value is None
    assert success is False
    assert stats == {}

    calls = {}

    class FakeLineax:
        class FunctionLinearOperator:
            def __init__(self, matvec, input_structure):
                calls["shape"] = tuple(input_structure.shape)
                calls["dtype"] = input_structure.dtype
                self.matvec = matvec

        class BiCGStab:
            def __init__(self, *, rtol, atol, max_steps):
                calls["solver"] = (rtol, atol, max_steps)

        @staticmethod
        def linear_solve(operator, b, *, solver, options, throw):
            del operator, solver
            calls["options"] = dict(options)
            calls["throw"] = bool(throw)
            return SimpleNamespace(value=jnp.asarray([1.0, -1.0]), stats={"num_steps": 4})

    fake_jax = SimpleNamespace(
        ShapeDtypeStruct=__import__("jax").ShapeDtypeStruct,
        device_get=lambda _value: (_ for _ in ()).throw(RuntimeError("host read")),
    )

    value, success, stats = lineax_bicgstab_solve(
        lambda x: x,
        jnp.asarray([2.0, 3.0]),
        x0=jnp.asarray([0.1, 0.2]),
        tol=1e-7,
        max_iter=9,
        lineax_module=FakeLineax,
        jax_module=fake_jax,
    )

    np.testing.assert_allclose(np.asarray(value), [1.0, -1.0])
    assert success is False
    assert stats == {"num_steps": 4}
    assert calls["shape"] == (2,)
    np.testing.assert_allclose(np.asarray(calls["options"]["y0"]), [0.1, 0.2])
    assert calls["solver"] == (1e-7, 0.0, 9)
    assert calls["throw"] is False


def test_active_residual_adjoint_routes_dense_chunked_path():
    from vmec_jax._compat import jax, jnp
    from vmec_jax.solvers.fixed_boundary.adjoint.residual_linear_algebra import solve_active_residual_adjoint_linearized

    matrix = jnp.asarray([[1.0, 0.0, 2.0], [0.5, -1.0, 0.25]])
    x0 = jnp.asarray([0.1, -0.2, 0.3])
    residual_star, residual_jvp = jax.linearize(lambda x: matrix @ x, x0)
    residual_vjp = jax.linear_transpose(residual_jvp, x0)
    b_active = jnp.asarray([3.0, -2.0, 1.0])

    result = solve_active_residual_adjoint_linearized(
        residual_jvp,
        residual_vjp,
        residual_star_active=residual_star,
        b_active=b_active,
        x_active_star=x0,
        residual_adjoint_mode="chunked",
        damping=0.2,
        cg_tol=1e-8,
        cg_max_iter=5,
        jac_chunk_size=1,
        dense_transpose_lstsq_host=_dense_host,
        is_traced=_not_traced,
        cg_solve=lambda *_args, **_kwargs: pytest.fail("chunked path should not call CG"),
    )

    expected = _dense_host(matrix, b_active, 0.2)
    assert result.route == "dense"
    np.testing.assert_allclose(np.asarray(result.lam), expected, rtol=1e-6, atol=1e-6)


def test_active_residual_adjoint_routes_bicgstab_and_lineax_successes():
    from vmec_jax._compat import jax, jnp
    from vmec_jax.solvers.fixed_boundary.adjoint.residual_linear_algebra import solve_active_residual_adjoint_linearized

    diag = jnp.asarray([2.0, -3.0])
    x0 = jnp.asarray([0.1, -0.2])
    residual_star, residual_jvp = jax.linearize(lambda x: diag * x, x0)
    residual_vjp = jax.linear_transpose(residual_jvp, x0)
    b_active = jnp.asarray([4.0, 9.0])

    bicgstab_calls = {}

    def fake_bicgstab(matvec, b, *, tol, atol, maxiter):
        bicgstab_calls["matvec"] = np.asarray(matvec(jnp.ones_like(b)))
        bicgstab_calls["params"] = (tol, atol, maxiter)
        return jnp.asarray([10.0, 11.0]), None

    direct = solve_active_residual_adjoint_linearized(
        residual_jvp,
        residual_vjp,
        residual_star_active=residual_star,
        b_active=b_active,
        x_active_star=x0,
        residual_adjoint_mode="direct",
        damping=0.5,
        cg_tol=1e-7,
        cg_max_iter=12,
        jac_chunk_size=None,
        dense_transpose_lstsq_host=_dense_host,
        is_traced=_not_traced,
        cg_solve=lambda *_args, **_kwargs: pytest.fail("direct path should not call CG"),
        bicgstab_solve=fake_bicgstab,
    )

    assert direct.route == "bicgstab"
    np.testing.assert_allclose(np.asarray(direct.lam), [10.0, 11.0])
    np.testing.assert_allclose(bicgstab_calls["matvec"], np.asarray(diag + 0.5))
    assert bicgstab_calls["params"] == (1e-7, 0.0, 12)

    lineax = solve_active_residual_adjoint_linearized(
        residual_jvp,
        residual_vjp,
        residual_star_active=residual_star,
        b_active=b_active,
        x_active_star=x0,
        residual_adjoint_mode="lineax",
        damping=0.25,
        cg_tol=1e-6,
        cg_max_iter=8,
        jac_chunk_size=None,
        dense_transpose_lstsq_host=_dense_host,
        is_traced=_not_traced,
        cg_solve=lambda *_args, **_kwargs: pytest.fail("lineax path should not call CG"),
        lineax_solve=lambda matvec, b, **_kwargs: (matvec(jnp.ones_like(b)), True, {"num_steps": 2}),
    )

    assert lineax.route == "lineax"
    assert lineax.info == {"num_steps": 2}
    np.testing.assert_allclose(np.asarray(lineax.lam), np.asarray(diag + 0.25))


def test_active_residual_adjoint_falls_back_to_cg_after_failed_square_solvers():
    from vmec_jax._compat import jax, jnp
    from vmec_jax.solvers.fixed_boundary.adjoint.residual_linear_algebra import solve_active_residual_adjoint_linearized

    diag = jnp.asarray([2.0, 3.0])
    x0 = jnp.asarray([0.1, -0.2])
    residual_star, residual_jvp = jax.linearize(lambda x: diag * x, x0)
    residual_vjp = jax.linear_transpose(residual_jvp, x0)
    b_active = jnp.asarray([4.0, 9.0])

    def fake_cg(matvec, rhs, *, tol, max_iter):
        np.testing.assert_allclose(np.asarray(matvec(jnp.ones_like(rhs))), np.asarray(diag * diag + 0.5))
        np.testing.assert_allclose(np.asarray(rhs), np.asarray(diag * b_active))
        assert (tol, max_iter) == (1e-5, 7)
        return jnp.asarray([-1.0, 2.0])

    result = solve_active_residual_adjoint_linearized(
        residual_jvp,
        residual_vjp,
        residual_star_active=residual_star,
        b_active=b_active,
        x_active_star=x0,
        residual_adjoint_mode="direct",
        damping=0.5,
        cg_tol=1e-5,
        cg_max_iter=7,
        jac_chunk_size=None,
        dense_transpose_lstsq_host=_dense_host,
        is_traced=_not_traced,
        cg_solve=fake_cg,
        bicgstab_solve=lambda *_args, **_kwargs: (jnp.zeros_like(b_active), 1),
    )

    assert result.route == "cg"
    np.testing.assert_allclose(np.asarray(result.lam), [-1.0, 2.0])


def test_full_residual_adjoint_routes_matrix_free_cg_and_jvp():
    from vmec_jax._compat import jnp
    from vmec_jax.solvers.fixed_boundary.adjoint.residual_linear_algebra import solve_full_residual_adjoint_linearized

    calls = {}
    st_star = SimpleNamespace(layout="layout-token")
    b = jnp.asarray([1.0, 2.0, 3.0])
    residual_star = jnp.asarray([0.1, 0.2, 0.3])

    def fake_make_full_normal_map(residual_jvp, residual_vjp, *, unpack_state, pack_state, project_state, layout, damping):
        del residual_jvp, residual_vjp, unpack_state, pack_state, project_state
        calls["normal_args"] = (layout, damping)
        return lambda x: 2.0 * x

    def fake_cg(matvec, rhs, *, tol, max_iter):
        calls["cg"] = (np.asarray(matvec(jnp.asarray([1.0, 1.0, 1.0]))), np.asarray(rhs), tol, max_iter)
        return jnp.asarray([4.0, 5.0, 6.0])

    result = solve_full_residual_adjoint_linearized(
        lambda state: jnp.asarray(state.values) + 1.0,
        lambda _x: _x,
        residual_star=residual_star,
        b=b,
        st_star=st_star,
        damping=0.125,
        cg_tol=1e-6,
        cg_max_iter=11,
        cg_solve=fake_cg,
        unpack_state=lambda u, layout: SimpleNamespace(values=u, layout=layout),
        pack_state=lambda state: state.values,
        project_state=lambda state: SimpleNamespace(values=state.values * 0.5, layout=state.layout),
        make_full_normal_map_func=fake_make_full_normal_map,
        validate_full_shapes=lambda residual, rhs: calls.setdefault("validated", (np.shape(residual), np.shape(rhs))),
    )

    assert result.route == "cg"
    assert calls["normal_args"] == ("layout-token", 0.125)
    np.testing.assert_allclose(calls["cg"][0], [2.0, 2.0, 2.0])
    np.testing.assert_allclose(calls["cg"][1], [1.0, 2.0, 3.0])
    assert calls["cg"][2:] == (1e-6, 11)
    assert calls["validated"] == ((3,), (3,))
    np.testing.assert_allclose(np.asarray(result.state_update.values), [2.0, 2.5, 3.0])
    np.testing.assert_allclose(np.asarray(result.lam), [3.0, 3.5, 4.0])
