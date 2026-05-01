from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from vmec_jax.plotting import (
    _mode_table_from_wout,
    surface_rz_from_wout,
    surface_rz_from_wout_physical,
)


def test_surface_rz_from_wout_respects_lasym_geometry_parity() -> None:
    theta = np.asarray([0.5 * np.pi])
    zeta = np.asarray([0.0])
    common = dict(
        nfp=1,
        xm=np.asarray([0, 1]),
        xn=np.asarray([0, 0]),
        xm_nyq=np.asarray([0, 1]),
        xn_nyq=np.asarray([0, 0]),
        rmnc=np.asarray([[1.0, 2.0]]),
        rmns=np.asarray([[10.0, 20.0]]),
        zmnc=np.asarray([[100.0, 200.0]]),
        zmns=np.asarray([[0.0, 3.0]]),
    )

    symmetric = SimpleNamespace(**common, lasym=False)
    R_sym, Z_sym = surface_rz_from_wout(symmetric, theta=theta, zeta=zeta, s_index=0)
    np.testing.assert_allclose(R_sym, [[1.0]])
    np.testing.assert_allclose(Z_sym, [[3.0]])

    asymmetric = SimpleNamespace(**common, lasym=True)
    R_asym, Z_asym = surface_rz_from_wout(asymmetric, theta=theta, zeta=zeta, s_index=0)
    np.testing.assert_allclose(R_asym, [[21.0]])
    np.testing.assert_allclose(Z_asym, [[103.0]])


def test_surface_rz_from_wout_physical_uses_vmec_xn_scaling() -> None:
    wout = SimpleNamespace(
        nfp=2,
        lasym=False,
        xm=np.asarray([0, 0]),
        xn=np.asarray([0, 2]),
        xm_nyq=np.asarray([0, 0]),
        xn_nyq=np.asarray([0, 2]),
        rmnc=np.asarray([[1.0, 2.0]]),
        rmns=np.asarray([[0.0, 0.0]]),
        zmnc=np.asarray([[0.0, 0.0]]),
        zmns=np.asarray([[0.0, 0.0]]),
    )

    R, Z = surface_rz_from_wout_physical(
        wout,
        theta=np.asarray([0.0]),
        phi=np.asarray([0.25 * np.pi]),
        s_index=0,
    )

    np.testing.assert_allclose(R, [[1.0]], atol=1e-14)
    np.testing.assert_allclose(Z, [[0.0]], atol=1e-14)
    np.testing.assert_array_equal(_mode_table_from_wout(wout, nyq=False, physical=False).n, [0, 1])
    np.testing.assert_array_equal(_mode_table_from_wout(wout, nyq=False, physical=True).n, [0, 2])
