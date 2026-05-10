import numpy as np

from vmec_jax.modes import default_grid_sizes, nyquist_mode_table, nyquist_mode_table_from_grid, vmec_mode_table


def test_vmec_mode_table_uses_vmec_m0_n_convention_and_abs_inputs():
    modes = vmec_mode_table(mpol=-3, ntor=-1)

    assert modes.K == 8
    np.testing.assert_array_equal(modes.m, np.array([0, 0, 1, 1, 1, 2, 2, 2]))
    np.testing.assert_array_equal(modes.n, np.array([0, 1, -1, 0, 1, -1, 0, 1]))
    assert vmec_mode_table(3, 1) is modes


def test_nyquist_mode_tables_cover_axisymmetric_and_grid_derived_limits():
    axisym = nyquist_mode_table(mpol=2, ntor=0)
    assert axisym.K == 6
    assert np.all(axisym.n == 0)

    padded = nyquist_mode_table(mpol=2, ntor=1)
    assert int(np.max(padded.m)) == 5
    assert int(np.max(np.abs(padded.n))) == 3

    grid = nyquist_mode_table_from_grid(mpol=2, ntor=1, ntheta=10, nzeta=8)
    assert int(np.max(grid.m)) == 5
    assert int(np.max(np.abs(grid.n))) == 4


def test_default_grid_sizes_follow_vmec_axisymmetric_collapse():
    assert default_grid_sizes(mpol=-2, ntor=-1, ntheta=0, nzeta=0) == (10, 6)
    assert default_grid_sizes(mpol=2, ntor=0, ntheta=9, nzeta=99) == (8, 1)
    assert default_grid_sizes(mpol=2, ntor=3, ntheta=11, nzeta=13) == (10, 13)
