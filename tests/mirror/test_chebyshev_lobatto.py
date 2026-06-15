from __future__ import annotations

import numpy as np
import pytest

from vmec_jax.mirror import MirrorConfig, MirrorResolution
from vmec_jax.mirror.core.basis import ChebyshevLobattoBasis
from vmec_jax.mirror.kernels.chebyshev import (
    apply_chebyshev_filter,
    chebyshev_lobatto_derivative_matrix,
    chebyshev_lobatto_nodes,
    clenshaw_curtis_weights,
    cosine_to_increasing_permutation,
    interpolate_chebyshev_values,
)

pytestmark = pytest.mark.mirror


def test_cgl_nodes_are_public_increasing_order_with_exact_endpoints():
    nodes = chebyshev_lobatto_nodes(17)
    assert np.all(np.diff(nodes) > 0.0)
    assert nodes[0] == -1.0
    assert nodes[-1] == 1.0

    cosine_nodes = chebyshev_lobatto_nodes(17, increasing=False)
    permutation = cosine_to_increasing_permutation(17)
    assert np.allclose(nodes, cosine_nodes[permutation])
    inverse = np.argsort(permutation)
    assert np.allclose(cosine_nodes, nodes[inverse])


def test_cgl_first_derivative_is_exact_for_resolved_monomials():
    num_nodes = 18
    nodes = chebyshev_lobatto_nodes(num_nodes)
    derivative = chebyshev_lobatto_derivative_matrix(num_nodes)
    for power in range(num_nodes):
        values = nodes**power
        expected = np.zeros_like(nodes) if power == 0 else power * nodes ** (power - 1)
        assert np.allclose(derivative @ values, expected, atol=2.0e-11, rtol=2.0e-11)


def test_cgl_second_derivative_is_exact_for_resolved_monomials():
    num_nodes = 16
    nodes = chebyshev_lobatto_nodes(num_nodes)
    derivative = chebyshev_lobatto_derivative_matrix(num_nodes)
    second = derivative @ derivative
    for power in range(num_nodes):
        values = nodes**power
        expected = np.zeros_like(nodes) if power < 2 else power * (power - 1) * nodes ** (power - 2)
        assert np.allclose(second @ values, expected, atol=2.0e-9, rtol=2.0e-10)


def test_clenshaw_curtis_weights_integrate_monomials():
    num_nodes = 33
    nodes = chebyshev_lobatto_nodes(num_nodes)
    weights = clenshaw_curtis_weights(num_nodes)
    assert np.all(weights > 0.0)
    assert np.isclose(np.sum(weights), 2.0)
    for power in range(num_nodes):
        expected = 0.0 if power % 2 else 2.0 / (power + 1)
        assert np.isclose(weights @ (nodes**power), expected, atol=3.0e-14, rtol=3.0e-14)


def test_cgl_resolution_change_is_spectral_for_smooth_functions():
    coarse = chebyshev_lobatto_nodes(17)
    fine = chebyshev_lobatto_nodes(65)
    values = np.exp(coarse) + 0.25 * np.cos(3.0 * coarse)
    interpolated = interpolate_chebyshev_values(values, coarse, fine)
    expected = np.exp(fine) + 0.25 * np.cos(3.0 * fine)
    assert np.max(np.abs(interpolated - expected)) < 5.0e-11

    roundtrip = interpolate_chebyshev_values(interpolated, fine, coarse)
    assert np.max(np.abs(roundtrip - values)) < 2.0e-13


def test_chebyshev_filter_reduces_high_modes_and_preserves_low_modes():
    nodes = chebyshev_lobatto_nodes(33)
    low = np.polynomial.chebyshev.chebval(nodes, [1.0, -0.25, 0.5])
    high = 0.2 * np.polynomial.chebyshev.chebval(nodes, [0.0] * 24 + [1.0])
    filtered = apply_chebyshev_filter(low + high, nodes=nodes, alpha=80.0, order=8, cutoff=3)
    low_filtered = apply_chebyshev_filter(low, nodes=nodes, alpha=80.0, order=8, cutoff=3)
    assert np.max(np.abs(low_filtered - low)) < 1.0e-12
    assert np.linalg.norm(filtered - low) < 0.35 * np.linalg.norm(high)


def test_chebyshev_basis_and_mirror_config_build_static_grid():
    basis = ChebyshevLobattoBasis.from_num_nodes(9)
    assert basis.nodes.shape == (9,)
    assert basis.derivative_matrix.shape == (9, 9)
    assert basis.second_derivative_matrix.shape == (9, 9)
    stacked = np.stack([basis.nodes**2, 2.0 * basis.nodes**2], axis=0)
    differentiated = basis.differentiate(stacked, axis=1)
    assert np.allclose(differentiated[0], 2.0 * basis.nodes)
    assert np.allclose(differentiated[1], 4.0 * basis.nodes)

    grid = MirrorConfig(resolution=MirrorResolution(ns=5, ntheta=1, nxi=9, mpol=0), z_min=-2.0, z_max=2.0).build_grid()
    assert grid.quadrature_shape == (5, 1, 9)
    assert np.isclose(grid.z_xi, 2.0)
    assert np.allclose(grid.z, 2.0 * grid.xi)
    assert np.isclose(np.sum(grid.w_s), 1.0)
    assert np.isclose(np.sum(grid.w_theta), 2.0 * np.pi)
    assert np.isclose(np.sum(grid.w_xi), 2.0)
