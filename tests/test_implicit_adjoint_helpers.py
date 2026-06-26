from __future__ import annotations

import numpy as np
import pytest

from vmec_jax._compat import has_jax


pytestmark = pytest.mark.skipif(not has_jax(), reason="implicit adjoint helpers require JAX")


def test_active_vjp_jvp_wrappers_route_diagonal_map():
    from vmec_jax._compat import jax, jnp
    from vmec_jax.solvers.fixed_boundary.adjoint.implicit_linear_algebra import (
        active_normal_rhs,
        make_active_normal_map,
        make_damped_transpose_map,
    )

    diag = jnp.asarray([2.0, -3.0, 0.5])
    x0 = jnp.asarray([0.1, -0.2, 0.3])
    _residual_star, residual_jvp = jax.linearize(lambda x: diag * x, x0)
    residual_vjp = jax.linear_transpose(residual_jvp, x0)
    damping = 0.25

    v = jnp.asarray([1.5, -2.0, 4.0])
    transpose_map = make_damped_transpose_map(residual_vjp, damping=damping)
    np.testing.assert_allclose(
        np.asarray(transpose_map(v)),
        np.asarray(diag * v + damping * v),
        rtol=1e-12,
        atol=1e-12,
    )

    normal_map = make_active_normal_map(residual_jvp, residual_vjp, damping=damping)
    np.testing.assert_allclose(
        np.asarray(normal_map(v)),
        np.asarray(diag * diag * v + damping * v),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(active_normal_rhs(residual_jvp, v)),
        np.asarray(diag * v),
        rtol=1e-12,
        atol=1e-12,
    )


def test_full_normal_map_routes_identity_state_packers():
    from vmec_jax._compat import jax, jnp
    from vmec_jax.solvers.fixed_boundary.adjoint.implicit_linear_algebra import make_full_normal_map

    x0 = jnp.asarray([0.5, -1.0, 2.0])
    _residual_star, residual_jvp = jax.linearize(lambda x: x, x0)
    residual_vjp = jax.linear_transpose(residual_jvp, x0)
    damping = 0.1

    normal_map = make_full_normal_map(
        residual_jvp,
        residual_vjp,
        unpack_state=lambda flat, _layout: flat,
        pack_state=lambda state: state,
        project_state=lambda state: state,
        layout=None,
        damping=damping,
    )

    u = jnp.asarray([3.0, -4.0, 5.0])
    np.testing.assert_allclose(
        np.asarray(normal_map(u)),
        np.asarray((1.0 + damping) * u),
        rtol=1e-12,
        atol=1e-12,
    )


def test_active_mode_selection_preserves_solver_fallback_branches():
    from vmec_jax.solvers.fixed_boundary.adjoint.implicit_linear_algebra import select_active_adjoint_mode

    direct_square = select_active_adjoint_mode("direct", active_is_square=True)
    assert direct_square.use_direct_stellsym is True
    assert direct_square.falls_back_to_cg is False

    bicgstab_square = select_active_adjoint_mode("bicgstab", active_is_square=True)
    assert bicgstab_square.use_direct_stellsym is True
    assert bicgstab_square.requested_mode == "bicgstab"

    direct_rect = select_active_adjoint_mode("direct", active_is_square=False)
    assert direct_rect.use_direct_stellsym is False
    assert direct_rect.falls_back_to_cg is True

    lineax_rect = select_active_adjoint_mode("lineax", active_is_square=False)
    assert lineax_rect.use_lineax_active is False
    assert lineax_rect.falls_back_to_cg is True

    dense_rect = select_active_adjoint_mode("dense", active_is_square=False)
    assert dense_rect.use_chunked_active is True
    assert dense_rect.falls_back_to_cg is False

    auto_square = select_active_adjoint_mode(" auto ", active_is_square=True)
    assert auto_square.requested_mode == "auto"
    assert auto_square.falls_back_to_cg is True


def test_dense_adjoint_from_jacobian_covers_dense_and_chunked_paths():
    from vmec_jax._compat import jnp
    from vmec_jax.solvers.fixed_boundary.adjoint.implicit_linear_algebra import dense_adjoint_from_jacobian

    jac = jnp.asarray([[1.0, 0.0, 2.0], [0.5, -1.0, 0.25]])
    rhs = jnp.asarray([3.0, -2.0, 1.0])
    damping = jnp.asarray(0.2)

    dense_calls = []

    def dense_host(J, b, damp):
        dense_calls.append((np.asarray(J).shape, np.asarray(b).shape, float(np.asarray(damp))))
        lhs = np.asarray(J) @ np.asarray(J).T + float(np.asarray(damp)) * np.eye(int(np.asarray(J).shape[0]))
        return np.linalg.solve(lhs, np.asarray(J) @ np.asarray(b))

    dense_lam = dense_adjoint_from_jacobian(
        jac,
        rhs,
        damping=damping,
        mode="dense",
        dense_transpose_lstsq_host=dense_host,
        is_traced=lambda *_args: False,
    )
    chunked_lam = dense_adjoint_from_jacobian(
        jac,
        rhs,
        damping=damping,
        mode="chunked",
        dense_transpose_lstsq_host=dense_host,
        is_traced=lambda *_args: False,
    )

    expected = dense_host(jac, rhs, damping)
    np.testing.assert_allclose(np.asarray(dense_lam), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(chunked_lam), expected, rtol=1e-12, atol=1e-12)
    assert dense_calls[0] == ((2, 3), (3,), 0.2)


def test_active_packing_chunk_size_and_shape_checks():
    from vmec_jax._compat import jnp
    from vmec_jax.solvers.fixed_boundary.adjoint.implicit_linear_algebra import (
        default_jac_chunk_size,
        full_active_keep_indices,
        select_active_packing_strategy,
        validate_active_adjoint_shapes,
        validate_full_adjoint_shapes,
    )

    assert select_active_packing_strategy(keep_all_active=True) == "full"
    assert select_active_packing_strategy(keep_all_active=False) == "reduced"
    np.testing.assert_array_equal(np.asarray(full_active_keep_indices(4)), np.arange(4))
    assert default_jac_chunk_size(jnp.ones(100), None) == 64
    assert default_jac_chunk_size(jnp.ones(3), 2) == 2

    assert validate_active_adjoint_shapes(jnp.ones(3), jnp.ones(3), jnp.ones(3)) is True
    assert validate_active_adjoint_shapes(jnp.ones(2), jnp.ones(3), jnp.ones(3)) is False
    validate_full_adjoint_shapes(jnp.ones(2), jnp.ones(3))

    with pytest.raises(ValueError, match="one-dimensional"):
        full_active_keep_indices(jnp.ones((2, 2)))
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        default_jac_chunk_size(jnp.ones(3), 0)
    with pytest.raises(ValueError, match="b_active must be one-dimensional"):
        validate_active_adjoint_shapes(jnp.ones(2), jnp.ones((1, 2)), jnp.ones(2))
    with pytest.raises(ValueError, match="full cotangent vector must be one-dimensional"):
        validate_full_adjoint_shapes(jnp.ones(2), jnp.ones((1, 2)))
