"""M0/M1 contracts and spectral identities for the mirror backend."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import vmec_jax.mirror as mirror_api  # noqa: E402

from vmec_jax.mirror import (  # noqa: E402
    EndCondition,
    MirrorBoundary,
    MirrorConfig,
    MirrorResolution,
    MirrorState,
)
from vmec_jax.mirror.basis import ChebyshevBasis, ThetaBasis  # noqa: E402
from vmec_jax.mirror.model import (  # noqa: E402
    MIRROR_INPUT_SCHEMA,
    MIRROR_OUTPUT_SCHEMA,
)


def test_public_api_keeps_numerical_kernels_in_owning_modules() -> None:
    required = {
        "MirrorConfig",
        "MirrorState",
        "solve_fixed_boundary_cli",
        "solve_free_boundary_cli",
        "solve_beta_scan_cli",
        "solve_fixed_boundary_implicit",
        "spline_fixed_boundary_adjoint",
        "spline_fixed_boundary_tangent",
        "write_mout",
        "plot_mout",
    }
    internal = {
        "ChebyshevBasis",
        "SeparableMirrorPreconditioner",
        "isotropic_force_residual",
        "solve_reduced_exterior_laplace_neumann",
    }
    assert required <= set(mirror_api.__all__)
    assert internal.isdisjoint(mirror_api.__all__)
    assert len(mirror_api.__all__) == 24


def test_mirror_config_freezes_supported_end_and_convergence_contract() -> None:
    config = MirrorConfig()
    assert config.end_condition is EndCondition.FIXED_FLUX_CUT
    assert config.ftol == 1.0e-12
    assert config.max_iterations == 2000
    assert MIRROR_INPUT_SCHEMA == "vmec_jax.mirror.input/1"
    assert MIRROR_OUTPUT_SCHEMA == "vmec_jax.mirror.mout/1"

    with pytest.raises(ValueError, match="ntheta=4 cannot resolve mpol=2"):
        MirrorResolution(ns=5, mpol=2, ntheta=4, nxi=9)
    with pytest.raises(ValueError, match="axisymmetric"):
        MirrorResolution(ns=5, mpol=0, ntheta=3, nxi=9)
    with pytest.raises(ValueError, match="z_max"):
        MirrorConfig(z_min=1.0, z_max=1.0)
    with pytest.raises(ValueError, match="ftol"):
        MirrorConfig(ftol=0.0)


def test_grid_and_state_shapes_are_explicit_and_pytree_compatible() -> None:
    config = MirrorConfig(
        resolution=MirrorResolution(ns=7, mpol=2, ntheta=7, nxi=13),
        z_min=-2.0,
        z_max=3.0,
    )
    grid = config.build_grid()
    assert grid.shape == (7, 7, 13)
    assert grid.xi[0] == -1.0 and grid.xi[-1] == 1.0
    assert np.all(np.diff(grid.xi) > 0.0)
    assert np.isclose(grid.z[0], -2.0) and np.isclose(grid.z[-1], 3.0)
    assert np.isclose(np.sum(grid.radial_weights), 1.0)
    assert np.isclose(np.sum(grid.theta_basis.weights), 2.0 * np.pi)
    assert np.isclose(np.sum(grid.axial_basis.weights), 2.0)

    radius = 0.3 * (1.0 + 0.1 * jnp.asarray(grid.xi) ** 2)
    boundary = MirrorBoundary.from_radius(radius, grid)
    state = MirrorState.from_boundary(boundary, grid)
    state.validate_shape(grid)
    assert state.radius_scale.shape == grid.shape
    assert state.lambda_stream.shape == grid.shape
    leaves, structure = jax.tree_util.tree_flatten(state)
    assert len(leaves) == 2
    rebuilt = jax.tree_util.tree_unflatten(structure, leaves)
    np.testing.assert_allclose(rebuilt.radius_scale, state.radius_scale)


def test_cgl_derivative_and_quadrature_are_polynomial_exact() -> None:
    basis = ChebyshevBasis.build(18)
    x = basis.nodes
    for power in range(basis.size):
        values = jnp.asarray(x**power)
        expected = np.zeros_like(x) if power == 0 else power * x ** (power - 1)
        np.testing.assert_allclose(
            basis.differentiate(values), expected, rtol=2.0e-11, atol=2.0e-11
        )

    assert np.all(basis.weights > 0.0)
    assert np.isclose(np.sum(basis.weights), 2.0)
    for power in range(basis.size):
        expected = 0.0 if power % 2 else 2.0 / (power + 1)
        assert np.isclose(
            float(basis.integrate(jnp.asarray(x**power))),
            expected,
            rtol=4.0e-13,
            atol=4.0e-13,
        )


def test_cgl_operators_obey_integration_by_parts_and_spectral_interpolation() -> None:
    coarse = ChebyshevBasis.build(17)
    x = jnp.asarray(coarse.nodes)
    f = 1.0 + x + 0.3 * x**5
    g = 0.7 - 0.2 * x**2 + x**4
    lhs = coarse.integrate(coarse.differentiate(f) * g + f * coarse.differentiate(g))
    rhs = f[-1] * g[-1] - f[0] * g[0]
    np.testing.assert_allclose(lhs, rhs, rtol=2.0e-13, atol=2.0e-13)

    fine = ChebyshevBasis.build(65)
    values = jnp.exp(x) + 0.25 * jnp.cos(3.0 * x)
    interpolated = coarse.interpolate(values, fine.nodes)
    expected = np.exp(fine.nodes) + 0.25 * np.cos(3.0 * fine.nodes)
    np.testing.assert_allclose(interpolated, expected, rtol=5.0e-11, atol=5.0e-11)
    roundtrip = fine.interpolate(interpolated, coarse.nodes)
    np.testing.assert_allclose(roundtrip, values, rtol=2.0e-13, atol=2.0e-13)

    rng = np.random.default_rng(17)
    left = jnp.asarray(rng.normal(size=coarse.size))
    right = jnp.asarray(rng.normal(size=coarse.size))
    np.testing.assert_allclose(
        jnp.vdot(left, coarse.differentiate(right)),
        jnp.vdot(coarse.differentiate_transpose(left), right),
        rtol=2.0e-14,
        atol=2.0e-14,
    )


def test_theta_fft_derivative_and_quadrature_resolve_requested_modes() -> None:
    basis = ThetaBasis.build(ntheta=13, mpol=5)
    theta = jnp.asarray(basis.nodes)
    values = 0.7 + 0.4 * jnp.cos(3.0 * theta) - 0.2 * jnp.sin(5.0 * theta)
    expected = -1.2 * jnp.sin(3.0 * theta) - 1.0 * jnp.cos(5.0 * theta)
    np.testing.assert_allclose(basis.differentiate(values), expected, rtol=2.0e-13, atol=2.0e-13)
    np.testing.assert_allclose(basis.integrate(values), 1.4 * np.pi, rtol=2.0e-13, atol=2.0e-13)

    left = 0.3 + jnp.sin(2.0 * theta) - 0.1 * jnp.cos(4.0 * theta)
    np.testing.assert_allclose(
        jnp.vdot(left, basis.differentiate(values)),
        jnp.vdot(basis.differentiate_transpose(left), values),
        rtol=2.0e-13,
        atol=2.0e-13,
    )

    axisym = ThetaBasis.build(ntheta=1, mpol=0)
    np.testing.assert_array_equal(axisym.differentiate(jnp.asarray([3.0])), jnp.asarray([0.0]))
