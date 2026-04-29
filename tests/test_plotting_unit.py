from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax.plotting import (
    _case_from_input_path,
    _default_example_outdir,
    _extent_from_grids,
    axis_rz_from_wout,
    axis_rz_from_wout_physical,
    bmag_from_wout,
    bmag_from_wout_physical,
    bsub_from_wout,
    bsup_from_wout,
    closed_theta_grid,
    fix_matplotlib_3d,
    profiles_from_wout,
    select_zeta_slices,
    surface_data_from_wout,
    surface_rz_from_wout,
    surface_rz_from_wout_physical,
    surface_stack,
    vmecplot2_bmag_grid,
    vmecplot2_cross_section_indices,
    vmecplot2_lcfs_3d_grid,
    vmecplot2_surface_grid,
    zeta_grid,
    zeta_grid_field_period,
)


def _toy_wout(*, lasym: bool = False):
    ns = 2
    main = np.asarray([0.0, 1.0])
    nyq = np.asarray([0.0, 1.0])
    rmnc = np.asarray([[1.0, 0.0], [1.0, 0.1]])
    rmns = np.asarray([[0.0, 0.0], [0.0, 0.05 if lasym else 0.0]])
    zmns = np.asarray([[0.0, 0.0], [0.0, 0.2]])
    zmnc = np.asarray([[0.0, 0.0], [0.0, 0.07 if lasym else 0.0]])
    bmnc = np.asarray([[2.0, 0.0], [2.0, 0.3]])
    bmns = np.asarray([[0.0, 0.0], [0.0, 0.4]])
    return SimpleNamespace(
        ns=ns,
        nfp=2,
        lasym=lasym,
        xm=main,
        xn=np.asarray([0.0, 0.0]),
        xm_nyq=nyq,
        xn_nyq=np.asarray([0.0, 0.0]),
        rmnc=rmnc,
        rmns=rmns,
        zmns=zmns,
        zmnc=zmnc,
        bmnc=bmnc,
        bmns=bmns,
        bsupumnc=bmnc + 1.0,
        bsupumns=bmns + 0.2,
        bsupvmnc=bmnc + 2.0,
        bsupvmns=bmns + 0.3,
        bsubumnc=bmnc + 3.0,
        bsubumns=bmns + 0.4,
        bsubvmnc=bmnc + 4.0,
        bsubvmns=bmns + 0.5,
        iotaf=np.asarray([0.0, 0.4]),
        iotas=np.asarray([0.0, 0.35]),
        presf=np.asarray([1.0, 0.0]),
        pres=np.asarray([0.9, 0.1]),
        raxis_cc=np.asarray([1.0, 0.1]),
        raxis_cs=np.asarray([0.0, 0.2]),
        zaxis_cs=np.asarray([0.0, 0.3]),
        zaxis_cc=np.asarray([0.0, 0.4]),
    )


def test_grids_slices_and_path_helpers():
    np.testing.assert_allclose(closed_theta_grid(3), [0.0, np.pi, 2.0 * np.pi])
    np.testing.assert_allclose(zeta_grid(3, endpoint=True), [0.0, np.pi, 2.0 * np.pi])
    np.testing.assert_allclose(zeta_grid_field_period(3, nfp=2), [0.0, np.pi / 3.0, 2.0 * np.pi / 3.0])
    np.testing.assert_array_equal(vmecplot2_cross_section_indices(8), [0, 2, 4, 6])
    with pytest.raises(ValueError, match="nzeta>=8"):
        vmecplot2_cross_section_indices(6)
    with pytest.raises(ValueError, match="positive"):
        select_zeta_slices(np.arange(4), n=0)

    zeta = np.asarray([0.0, 0.5, 1.0, 1.5])
    np.testing.assert_allclose(select_zeta_slices(zeta, n=3), [0.0, 1.0, 1.5])
    assert _case_from_input_path("/tmp/input.nfp4_QH") == "nfp4_QH"
    assert _case_from_input_path("/tmp/wout_test.nc") == "wout_test"
    assert _default_example_outdir("sub", "case", "/tmp/out") == Path("/tmp/out")
    assert _extent_from_grids(np.asarray([2.0]), np.asarray([3.0])) == (2.5, 3.5, 1.5, 2.5)


def test_wout_surface_and_field_helpers_respect_lasym():
    theta = np.asarray([0.0, 0.5 * np.pi])
    zeta = np.asarray([0.0, 0.3])
    wout = _toy_wout(lasym=False)

    R, Z = surface_rz_from_wout(wout, theta=theta, zeta=zeta, s_index=1)
    np.testing.assert_allclose(R[:, 0], [1.1, 1.0])
    np.testing.assert_allclose(Z[:, 0], [0.0, 0.2])
    B = bmag_from_wout(wout, theta=theta, zeta=zeta, s_index=1)
    np.testing.assert_allclose(B[:, 0], [2.3, 2.0])

    wout_asym = _toy_wout(lasym=True)
    R_asym, Z_asym = surface_rz_from_wout(wout_asym, theta=theta, zeta=zeta, s_index=1)
    np.testing.assert_allclose(R_asym[:, 0], [1.1, 1.05])
    np.testing.assert_allclose(Z_asym[:, 0], [0.07, 0.2])
    B_asym = bmag_from_wout(wout_asym, theta=theta, zeta=zeta, s_index=1)
    np.testing.assert_allclose(B_asym[:, 0], [2.3, 2.4])

    bsupu, bsupv = bsup_from_wout(wout_asym, theta=theta, zeta=zeta, s_index=1)
    bsubu, bsubv = bsub_from_wout(wout_asym, theta=theta, zeta=zeta, s_index=1)
    assert bsupu.shape == bsupv.shape == bsubu.shape == bsubv.shape == (2, 2)
    assert float(bsupu[1, 0]) > float(bmag_from_wout(wout, theta=theta, zeta=zeta, s_index=1)[1, 0])


def test_physical_angle_helpers_match_field_period_helpers_for_axisymmetric_data():
    wout = _toy_wout(lasym=True)
    theta = np.asarray([0.0, 0.25 * np.pi, 0.5 * np.pi])
    phi = np.asarray([0.0, 0.2])
    zeta = phi * float(wout.nfp)

    R_phys, Z_phys = surface_rz_from_wout_physical(wout, theta=theta, phi=phi, s_index=1)
    R_zeta, Z_zeta = surface_rz_from_wout(wout, theta=theta, zeta=zeta, s_index=1)
    np.testing.assert_allclose(R_phys, R_zeta)
    np.testing.assert_allclose(Z_phys, Z_zeta)

    B_phys = bmag_from_wout_physical(wout, theta=theta, phi=phi, s_index=1)
    B_zeta = bmag_from_wout(wout, theta=theta, zeta=zeta, s_index=1)
    np.testing.assert_allclose(B_phys, B_zeta)


def test_axis_profiles_surface_stack_and_surface_data():
    wout = _toy_wout(lasym=True)
    zeta = np.asarray([0.0, np.pi / 4.0])
    R_axis, Z_axis = axis_rz_from_wout(wout, zeta=zeta)
    R_axis_phys, Z_axis_phys = axis_rz_from_wout_physical(wout, phi=zeta)
    np.testing.assert_allclose(R_axis, R_axis_phys)
    np.testing.assert_allclose(Z_axis, Z_axis_phys)
    np.testing.assert_allclose(R_axis, [1.1, 0.8])
    np.testing.assert_allclose(Z_axis, [0.4, -0.3])

    fallback = SimpleNamespace(rmnc=np.asarray([[1.25]]))
    np.testing.assert_allclose(axis_rz_from_wout(fallback, zeta=zeta)[0], [1.25, 1.25])

    profiles = profiles_from_wout(wout)
    assert set(profiles) == {"s", "s_half", "iotaf", "iotas", "presf", "pres", "buco", "bvco", "jcuru", "jcurv"}
    np.testing.assert_allclose(profiles["s"], [0.0, 1.0])
    np.testing.assert_allclose(profiles["s_half"], [0.5])
    np.testing.assert_allclose(profiles["buco"], [0.0, 0.0])

    theta = np.asarray([0.0, np.pi / 2.0])
    R_stack, Z_stack = surface_stack(wout, theta=theta, zeta_list=[0.0, 0.1, 0.2], s_index=1)
    assert R_stack.shape == Z_stack.shape == (2, 3)
    data = surface_data_from_wout(wout, theta=theta, zeta=np.asarray([0.0]), s_index=1, with_bmag=True)
    assert data.R.shape == data.Z.shape == data.B.shape == (2, 1)
    assert surface_data_from_wout(wout, theta=theta, zeta=np.asarray([0.0]), s_index=1).B is None


def test_vmecplot2_grid_helpers_return_vmecplot2_shapes():
    wout = _toy_wout(lasym=True)
    theta, zeta, B = vmecplot2_bmag_grid(wout, s_index=1, ntheta=4, nzeta=5, zeta_max=np.pi)
    assert theta.shape == (4,)
    assert zeta.shape == (5,)
    assert B.shape == (4, 5)

    theta_s, zeta_s, R_s, Z_s = vmecplot2_surface_grid(wout, s_index=1, ntheta=6, nzeta=3)
    assert theta_s.shape == (6,)
    assert zeta_s.shape == (3,)
    assert R_s.shape == Z_s.shape == (6, 3)

    theta_3d, phi_3d, R_3d, Z_3d, B_3d = vmecplot2_lcfs_3d_grid(wout, s_index=1, ntheta=5, nzeta=7)
    assert theta_3d.shape == (5,)
    assert phi_3d.shape == (7,)
    assert R_3d.shape == Z_3d.shape == B_3d.shape == (5, 7)


def test_fix_matplotlib_3d_sets_equal_radius_limits():
    class _Axis:
        def __init__(self):
            self.xlim = (0.0, 2.0)
            self.ylim = (-2.0, 2.0)
            self.zlim = (10.0, 11.0)

        def get_xlim3d(self):
            return self.xlim

        def get_ylim3d(self):
            return self.ylim

        def get_zlim3d(self):
            return self.zlim

        def set_xlim3d(self, value):
            self.xlim = tuple(value)

        def set_ylim3d(self, value):
            self.ylim = tuple(value)

        def set_zlim3d(self, value):
            self.zlim = tuple(value)

    axis = _Axis()
    fix_matplotlib_3d(axis)
    assert axis.xlim == (-1.0, 3.0)
    assert axis.ylim == (-2.0, 2.0)
    assert axis.zlim == (8.5, 12.5)
