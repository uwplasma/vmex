from __future__ import annotations

import numpy as np
import pytest

import vmec_jax as vj
from vmec_jax.mirror.core.basis import ThetaFourierBasis
from vmec_jax.mirror.kernels.fourier import evaluate_real_fourier, evaluate_real_fourier_derivative

pytestmark = pytest.mark.mirror


def test_theta_basis_evaluates_real_fourier_series_and_derivatives():
    basis = ThetaFourierBasis.from_resolution(ntheta=32, mpol=5)
    cos_coeffs = np.zeros(6)
    sin_coeffs = np.zeros(6)
    cos_coeffs[2] = 1.5
    sin_coeffs[3] = -0.25

    values = basis.evaluate(cos_coeffs, sin_coeffs)
    expected = 1.5 * np.cos(2.0 * basis.theta) - 0.25 * np.sin(3.0 * basis.theta)
    assert np.allclose(values, expected)

    derivative = basis.evaluate_derivative(cos_coeffs, sin_coeffs)
    expected_derivative = -3.0 * np.sin(2.0 * basis.theta) - 0.75 * np.cos(3.0 * basis.theta)
    assert np.allclose(derivative, expected_derivative)


def test_fft_theta_derivative_is_exact_for_resolved_modes():
    basis = ThetaFourierBasis.from_resolution(ntheta=33, mpol=8)
    for mode in range(1, 8):
        cos_values = np.cos(mode * basis.theta)
        sin_values = np.sin(mode * basis.theta)
        assert np.allclose(basis.differentiate(cos_values), -mode * sin_values, atol=2.0e-13)
        assert np.allclose(basis.differentiate(sin_values), mode * cos_values, atol=2.0e-13)


def test_theta_quadrature_orthogonality_for_resolved_modes():
    basis = ThetaFourierBasis.from_resolution(ntheta=64, mpol=12)
    assert np.isclose(np.sum(basis.weights), 2.0 * np.pi)
    for mode in range(1, 12):
        cos_values = np.cos(mode * basis.theta)
        sin_values = np.sin(mode * basis.theta)
        assert np.isclose(basis.weights @ (cos_values**2), np.pi, atol=2.0e-14)
        assert np.isclose(basis.weights @ (sin_values**2), np.pi, atol=2.0e-14)
        assert abs(basis.weights @ (sin_values * cos_values)) < 2.0e-14


def test_axisymmetric_theta_basis_has_zero_derivative():
    basis = ThetaFourierBasis.from_resolution(ntheta=1, mpol=0)
    assert np.allclose(basis.theta, [0.0])
    assert np.allclose(basis.differentiate(np.array([3.0])), [0.0])
    assert np.isclose(basis.weights[0], 2.0 * np.pi)


def test_function_helpers_match_basis_methods():
    basis = ThetaFourierBasis.from_resolution(ntheta=16, mpol=4)
    cos_coeffs = np.array([1.0, 0.0, 0.5, 0.0, -0.2])
    sin_coeffs = np.array([0.0, 0.7, 0.0, -0.1, 0.0])
    assert np.allclose(basis.evaluate(cos_coeffs, sin_coeffs), evaluate_real_fourier(basis.theta, cos_coeffs, sin_coeffs=sin_coeffs))
    assert np.allclose(
        basis.evaluate_derivative(cos_coeffs, sin_coeffs),
        evaluate_real_fourier_derivative(basis.theta, cos_coeffs, sin_coeffs=sin_coeffs),
    )


def test_vmec_jax_exposes_lazy_mirror_module():
    assert vj.mirror.MirrorResolution(ntheta=1, mpol=0).ntheta == 1
