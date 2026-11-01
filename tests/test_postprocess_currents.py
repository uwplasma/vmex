"""Focused current-density regressions for VMEC/C++ VMEC WOUT parity."""

from __future__ import annotations

import numpy as np

from vmex.core.postprocess import MU0, compute_currents


def test_lasym_odd_m_uses_inner_half_mesh_sqrt_s() -> None:
    """Pin the C++ VMEC 0.7.1/PARVMEC correction for ``currvmns``.

    The historical asymmetric ``read_wout_mod.f90`` branch divided the
    inner ``bsubumns`` value by the *outer* half-mesh sqrt(s).  That makes
    the two neighboring terms use the same denominator.  The physical
    radial derivative, and C++ VMEC's PARVMEC-calibrated implementation, use
    the corresponding inner and outer denominators.
    """
    ns = 5
    zeros = np.zeros((ns, 1))
    bsubumns = np.asarray([[0.0], [2.0], [9.0], [20.0], [35.0]])
    _currumnc, _currvmnc, currumns, currvmns = compute_currents(
        bsubsmns=zeros,
        bsubumnc=zeros,
        bsubvmnc=zeros,
        xm_nyq=np.asarray([1]),
        xn_nyq=np.asarray([0]),
        bsubsmnc=zeros,
        bsubumns=bsubumns,
        bsubvmns=zeros,
        lasym=True,
    )

    ohs = float(ns - 1)
    hs = 1.0 / ohs
    j = 1
    sqrt_s_inner = np.sqrt(hs * (j - 0.5))
    sqrt_s_outer = np.sqrt(hs * (j + 0.5))
    sqrt_s_full = np.sqrt(hs * j)
    bu0 = bsubumns[j, 0] / sqrt_s_inner
    bu1 = bsubumns[j + 1, 0] / sqrt_s_outer
    expected = (
        ohs * (bu1 - bu0) * sqrt_s_full
        + 0.25 * (bu0 + bu1) / sqrt_s_full
    ) / MU0

    legacy_bu0 = bsubumns[j, 0] / sqrt_s_outer
    legacy = (
        ohs * (bu1 - legacy_bu0) * sqrt_s_full
        + 0.25 * (legacy_bu0 + bu1) / sqrt_s_full
    ) / MU0

    assert currumns is not None and currvmns is not None
    assert currumns[j, 0] == 0.0
    np.testing.assert_allclose(currvmns[j, 0], expected, rtol=2.0e-15)
    assert not np.isclose(currvmns[j, 0], legacy, rtol=1.0e-8)


def test_full_mesh_bsubs_is_not_averaged_as_half_mesh() -> None:
    """A full-mesh ``bsubsmns`` enters on its own surface, without a half-cell shift."""
    ns, s = 11, np.linspace(0, 1, 11)
    zeros = np.zeros((ns, 2))
    bsubsmns = np.stack((np.sqrt(s), s), axis=1)  # m = 1 and m = 2 radial scalings
    _, currvmnc, _, _ = compute_currents(
        bsubsmns=bsubsmns, bsubumnc=zeros, bsubvmnc=zeros,
        xm_nyq=np.asarray([1, 2]), xn_nyq=np.asarray([0, 0]))
    np.testing.assert_allclose(currvmnc[2:-1]*MU0, -bsubsmns[2:-1]*[1, 2], rtol=1e-13)
