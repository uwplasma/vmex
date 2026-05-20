from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax._compat import has_jax


pytestmark = pytest.mark.skipif(not has_jax(), reason="implicit helpers require JAX")


def test_profile_log_helpers_include_elapsed_payloads_when_enabled(monkeypatch, capsys):
    import vmec_jax.implicit as implicit

    monkeypatch.setenv("VMEC_JAX_PROFILE_BACKWARD", "yes")
    monkeypatch.setenv("VMEC_JAX_PROFILE_RESIDUAL", "yes")
    monkeypatch.setattr(implicit.time, "perf_counter", lambda: 12.5)

    implicit._vmec_backward_profile_log("adjoint", start=10.0, size=3)
    implicit._vmec_residual_profile_log("residual", start=11.0, projected=True)

    out = capsys.readouterr().out
    assert "[vmec_jax backward]" in out
    assert "'elapsed_s': 2.5" in out
    assert "'size': 3" in out
    assert "[vmec_jax residual]" in out
    assert "'elapsed_s': 1.5" in out
    assert "'projected': True" in out


def test_dense_transpose_lstsq_host_matches_tikhonov_normal_equations():
    from vmec_jax.implicit import _dense_transpose_lstsq_host

    jac = np.asarray([[1.0, 2.0, -1.0], [0.5, -0.25, 1.5]])
    rhs = np.asarray([0.75, -1.25, 0.5])
    damping = 0.2

    lam = _dense_transpose_lstsq_host(jac, rhs, damping)
    lhs = jac @ jac.T + damping * np.eye(jac.shape[0])
    expected = np.linalg.solve(lhs, jac @ rhs)

    np.testing.assert_allclose(lam, expected, rtol=1e-12, atol=1e-12)


def test_linear_map_jacobian_columns_chunks_exact_columns():
    from vmec_jax._compat import jnp
    from vmec_jax.implicit import _linear_map_jacobian_columns

    matrix = jnp.asarray([[1.0, -2.0, 0.5], [0.25, 3.0, -1.5]])
    jac = _linear_map_jacobian_columns(
        lambda x: matrix @ x,
        input_size=3,
        output_size=2,
        dtype=matrix.dtype,
        chunk_size=2,
    )

    np.testing.assert_allclose(np.asarray(jac), np.asarray(matrix), rtol=1e-12, atol=1e-12)

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        _linear_map_jacobian_columns(lambda x: x, input_size=1, output_size=1, dtype=matrix.dtype, chunk_size=0)


def test_lineax_bicgstab_wrapper_handles_optional_x0_and_device_get_failure(monkeypatch):
    import vmec_jax.implicit as implicit
    from vmec_jax._compat import jnp

    calls = {}

    class FakeLineax:
        class FunctionLinearOperator:
            def __init__(self, matvec, input_structure):
                calls["operator_shape"] = tuple(input_structure.shape)
                calls["operator_dtype"] = input_structure.dtype
                self.matvec = matvec

        class BiCGStab:
            def __init__(self, *, rtol, atol, max_steps):
                calls["solver"] = (rtol, atol, max_steps)

        @staticmethod
        def linear_solve(operator, b, *, solver, options, throw):
            del operator, solver
            calls["b"] = np.asarray(b)
            calls["options"] = dict(options)
            calls["throw"] = bool(throw)
            return SimpleNamespace(value=jnp.asarray([1.0, -1.0]), stats={"num_steps": 2})

    monkeypatch.setattr(implicit, "lx", FakeLineax)
    monkeypatch.setattr(implicit.jax, "device_get", lambda _value: (_ for _ in ()).throw(RuntimeError("host read")))

    value, success, stats = implicit._lineax_bicgstab_solve(
        lambda x: x,
        jnp.asarray([2.0, 3.0]),
        x0=jnp.asarray([0.1, 0.2]),
        tol=1e-7,
        max_iter=9,
    )

    np.testing.assert_allclose(np.asarray(value), [1.0, -1.0])
    assert success is False
    assert stats == {"num_steps": 2}
    np.testing.assert_allclose(calls["options"]["y0"], [0.1, 0.2])
    assert calls["solver"] == (1e-7, 0.0, 9)
    assert calls["throw"] is False
