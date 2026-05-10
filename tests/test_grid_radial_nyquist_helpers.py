from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax.grids import angle_steps, make_angle_grid
from vmec_jax.nyquist import nyquist_basis_from_wout, nyquist_mode_table
from vmec_jax.radial import d_ds_coeffs


def test_radial_derivative_handles_single_and_two_surface_grids():
    single = np.asarray([[3.0, 4.0]])
    np.testing.assert_allclose(np.asarray(d_ds_coeffs(single, np.asarray([0.0]))), np.zeros_like(single))

    two = np.asarray([[1.0, 3.0], [5.0, 11.0]])
    got = np.asarray(d_ds_coeffs(two, np.asarray([0.0, 0.5])))
    np.testing.assert_allclose(got, [[8.0, 16.0], [8.0, 16.0]])


def test_angle_steps_validate_periodic_grid_sizes():
    np.testing.assert_allclose(angle_steps(ntheta=4, nzeta=2), (0.5 * np.pi, np.pi))

    with pytest.raises(ValueError, match="ntheta"):
        angle_steps(ntheta=0, nzeta=1)
    with pytest.raises(ValueError, match="nzeta"):
        angle_steps(ntheta=1, nzeta=0)


def test_nyquist_mode_and_basis_caches_reuse_objects():
    xm = np.asarray([0, 1, 1])
    xn = np.asarray([0, -4, 4])
    first = nyquist_mode_table(xm_nyq=xm, xn_nyq=xn, nfp=2)
    second = nyquist_mode_table(xm_nyq=xm.copy(), xn_nyq=xn.copy(), nfp=2)

    assert first is second
    np.testing.assert_array_equal(first.n, [0, -2, 2])

    grid = make_angle_grid(ntheta=4, nzeta=3, nfp=2)
    wout = SimpleNamespace(xm_nyq=xm, xn_nyq=xn, nfp=2)
    basis_first = nyquist_basis_from_wout(wout=wout, grid=grid)
    basis_second = nyquist_basis_from_wout(wout=wout, grid=grid)

    assert basis_first is basis_second
