"""Uniform-theta Fourier helpers for mirror geometry."""

from __future__ import annotations

from typing import Any

import numpy as np


TWOPI = 2.0 * np.pi


def theta_nodes(ntheta: int, *, dtype: Any = float, endpoint: bool = False) -> np.ndarray:
    """Return a uniform theta grid on ``[0, 2*pi)`` by default."""
    ntheta = int(ntheta)
    if ntheta < 1:
        raise ValueError("ntheta must be >= 1")
    return np.linspace(0.0, TWOPI, ntheta, endpoint=endpoint, dtype=dtype)


def theta_weights(ntheta: int, *, dtype: Any = float) -> np.ndarray:
    """Return uniform quadrature weights for theta integration."""
    ntheta = int(ntheta)
    if ntheta < 1:
        raise ValueError("ntheta must be >= 1")
    return np.full(ntheta, TWOPI / float(ntheta), dtype=dtype)


def real_fourier_modes(mpol: int) -> np.ndarray:
    """Return non-negative real Fourier mode numbers ``0..mpol``."""
    mpol = int(mpol)
    if mpol < 0:
        raise ValueError("mpol must be >= 0")
    return np.arange(mpol + 1, dtype=int)


def evaluate_real_fourier(theta, cos_coeffs, *, sin_coeffs=None) -> np.ndarray:
    """Evaluate ``sum_m c_m cos(m theta) + s_m sin(m theta)``."""
    theta = np.asarray(theta)
    cos_coeffs = np.asarray(cos_coeffs)
    if sin_coeffs is None:
        sin_coeffs = np.zeros_like(cos_coeffs)
    else:
        sin_coeffs = np.asarray(sin_coeffs)
    if cos_coeffs.shape[-1] != sin_coeffs.shape[-1]:
        raise ValueError("cos_coeffs and sin_coeffs must have the same final dimension")
    modes = np.arange(cos_coeffs.shape[-1], dtype=theta.dtype)
    phase = theta[:, None] * modes[None, :]
    values = np.tensordot(cos_coeffs, np.cos(phase), axes=([-1], [1]))
    values += np.tensordot(sin_coeffs, np.sin(phase), axes=([-1], [1]))
    return values


def evaluate_real_fourier_derivative(theta, cos_coeffs, *, sin_coeffs=None) -> np.ndarray:
    """Evaluate the theta derivative of a real Fourier series."""
    theta = np.asarray(theta)
    cos_coeffs = np.asarray(cos_coeffs)
    if sin_coeffs is None:
        sin_coeffs = np.zeros_like(cos_coeffs)
    else:
        sin_coeffs = np.asarray(sin_coeffs)
    if cos_coeffs.shape[-1] != sin_coeffs.shape[-1]:
        raise ValueError("cos_coeffs and sin_coeffs must have the same final dimension")
    modes = np.arange(cos_coeffs.shape[-1], dtype=theta.dtype)
    phase = theta[:, None] * modes[None, :]
    values = np.tensordot(-cos_coeffs * modes, np.sin(phase), axes=([-1], [1]))
    values += np.tensordot(sin_coeffs * modes, np.cos(phase), axes=([-1], [1]))
    return values


def _integer_wavenumbers(ntheta: int) -> np.ndarray:
    return np.fft.fftfreq(int(ntheta), d=1.0 / float(ntheta))


def fourier_derivative(values, *, axis: int = -1) -> np.ndarray:
    """Differentiate periodic nodal values using FFT wavenumbers."""
    values = np.asarray(values)
    axis = axis % values.ndim
    ntheta = values.shape[axis]
    if ntheta == 1:
        return np.zeros_like(values)
    wavenumbers = _integer_wavenumbers(ntheta)
    shape = [1] * values.ndim
    shape[axis] = ntheta
    transformed = np.fft.fft(values, axis=axis)
    derivative = np.fft.ifft(1j * wavenumbers.reshape(shape) * transformed, axis=axis)
    return np.real_if_close(derivative, tol=1000)


def fourier_second_derivative(values, *, axis: int = -1) -> np.ndarray:
    """Return the second theta derivative of periodic nodal values."""
    values = np.asarray(values)
    axis = axis % values.ndim
    ntheta = values.shape[axis]
    if ntheta == 1:
        return np.zeros_like(values)
    wavenumbers = _integer_wavenumbers(ntheta)
    shape = [1] * values.ndim
    shape[axis] = ntheta
    transformed = np.fft.fft(values, axis=axis)
    derivative = np.fft.ifft(-(wavenumbers.reshape(shape) ** 2) * transformed, axis=axis)
    return np.real_if_close(derivative, tol=1000)
