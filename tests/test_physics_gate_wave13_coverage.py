from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vmec_jax.modes import ModeTable
from vmec_jax.kernels.parity import (
    _mn_cos_to_signed_host,
    _mn_sin_to_signed_host,
    _signed_to_mn_cos_host,
    _signed_to_mn_sin_host,
    signed_maps_from_modes,
    vmec_m1_internal_to_physical_signed,
    vmec_m1_physical_to_internal_signed,
)
from vmec_jax.kernels.residue import vmec_pwint_from_trig, vmec_scalxc_from_s, vmec_wint_from_trig
from vmec_jax.kernels.tomnsp import vmec_trig_tables
from vmec_jax.wout import read_wout


WOUT_FLUX_CASES = (
    ("axisym_vacuum", "examples/data/wout_circular_tokamak.nc"),
    ("axisym_finite_beta", "examples/data/wout_shaped_tokamak_pressure.nc"),
    ("qh_warm_start", "examples/data/wout_nfp4_QH_warm_start.nc"),
    ("qi_fixed_resolution", "examples/data/wout_nfp3_QI_fixed_resolution_final.nc"),
    (
        "lasym_finite_beta",
        "examples/data/single_grid/wout_basic_non_stellsym_pressure_reference.nc",
    ),
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(("case_name", "wout_rel"), WOUT_FLUX_CASES)
def test_bundled_wout_flux_mesh_and_rotational_transform_closure(case_name: str, wout_rel: str) -> None:
    """Bundled VMEC WOUTs should preserve toroidal/poloidal flux mesh conventions."""
    pytest.importorskip("netCDF4")

    wout = read_wout(_repo_root() / wout_rel)
    ns = int(wout.ns)
    assert ns >= 3

    phi = np.asarray(wout.phi, dtype=float)
    phipf = np.asarray(wout.phipf, dtype=float)
    phips = np.asarray(wout.phips, dtype=float)
    chipf = np.asarray(wout.chipf, dtype=float)
    iotaf = np.asarray(wout.iotaf, dtype=float)

    assert phi.shape == (ns,)
    assert phipf.shape == (ns,)
    assert phips.shape == (ns,)
    assert chipf.shape == (ns,)
    assert iotaf.shape == (ns,)
    assert np.all(np.isfinite([phi, phipf, phips, chipf, iotaf]))

    dphi_ds_half = (phi[1:] - phi[:-1]) * float(ns - 1)
    np.testing.assert_allclose(
        phipf[1:],
        dphi_ds_half,
        rtol=3.0e-13,
        atol=5.0e-13,
        err_msg=f"{case_name}: phipf must be the half-mesh dphi/ds",
    )
    np.testing.assert_allclose(
        phipf[0],
        phipf[1],
        rtol=0.0,
        atol=0.0,
        err_msg=f"{case_name}: VMEC copies phipf(js=2) to the axis",
    )
    np.testing.assert_allclose(
        phips[1:],
        phipf[1:] / (2.0 * np.pi * float(wout.signgs)),
        rtol=1.0e-13,
        atol=1.0e-13,
        err_msg=f"{case_name}: phips must use VMEC signgs/twopi normalization",
    )
    np.testing.assert_allclose(
        chipf,
        iotaf * phipf,
        rtol=3.0e-13,
        atol=5.0e-13,
        err_msg=f"{case_name}: chipf must close with iotaf*phipf",
    )


def test_vmec_quadrature_and_scalxc_exact_rules() -> None:
    """Protect VMEC angular weights, axis masking, and odd-m radial force scaling."""
    trig = vmec_trig_tables(ntheta=6, nzeta=3, nfp=2, mmax=4, nmax=1, lasym=True, cache=False)

    wint = np.asarray(vmec_wint_from_trig(trig, nzeta=3), dtype=float)
    expected_wtheta = np.asarray(trig.cosmui3[:, 0], dtype=float) / float(np.asarray(trig.mscale[0]))
    np.testing.assert_allclose(wint, expected_wtheta[:, None] * np.ones((1, 3)), rtol=0.0, atol=0.0)

    pwint = np.asarray(vmec_pwint_from_trig(trig, ns=4, nzeta=3), dtype=float)
    assert pwint.shape == (4, int(trig.ntheta3), 3)
    np.testing.assert_allclose(pwint[0], 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(pwint[1:], np.broadcast_to(wint, pwint[1:].shape), rtol=0.0, atol=0.0)
    with pytest.raises(ValueError, match="ns must be >= 1"):
        vmec_pwint_from_trig(trig, ns=0, nzeta=3)

    s = np.asarray([0.0, 0.25, 0.81, 1.0], dtype=float)
    scalxc = np.asarray(vmec_scalxc_from_s(s=s, mpol=5), dtype=float)
    expected_odd = 1.0 / np.maximum(np.sqrt(s), np.sqrt(s[1]))
    expected = np.ones((s.size, 5), dtype=float)
    expected[:, 1::2] = expected_odd[:, None]
    np.testing.assert_allclose(scalxc, expected, rtol=1.0e-15, atol=1.0e-15)
    np.testing.assert_allclose(np.asarray(vmec_scalxc_from_s(s=[], mpol=3)), np.zeros((0, 3)))
    np.testing.assert_allclose(np.asarray(vmec_scalxc_from_s(s=s, mpol=0)), np.zeros((s.size, 0)))


def test_vmec_m1_constrained_basis_applies_exact_symmetric_and_lasym_signs() -> None:
    """The VMEC m=1 internal basis must rotate only the paired R/Z parity blocks."""
    modes = ModeTable(
        m=np.asarray([0, 1, 1, 2, 2], dtype=int),
        n=np.asarray([0, -1, 1, -1, 1], dtype=int),
    )
    maps = signed_maps_from_modes(modes)
    ns = 2
    shape = (ns, maps.mpol, maps.nrange)

    rcc = np.zeros(shape)
    rss = np.zeros(shape)
    zsc = np.zeros(shape)
    zcs = np.zeros(shape)
    rsc = np.zeros(shape)
    rcs = np.zeros(shape)
    zcc = np.zeros(shape)
    zss = np.zeros(shape)

    rss[:, 1, 1] = [2.0, 4.0]
    zcs[:, 1, 1] = [0.5, 1.5]
    rsc[:, 1, 1] = [-3.0, 5.0]
    zcc[:, 1, 1] = [1.0, -2.0]

    # Populate m=2 sentinels to prove non-m=1 physics modes pass through unchanged.
    rcc[:, 2, 1] = [7.0, 8.0]
    rss[:, 2, 1] = [0.3, 0.4]
    zsc[:, 2, 1] = [9.0, 10.0]
    zcs[:, 2, 1] = [0.6, 0.7]
    rsc[:, 2, 1] = [-1.0, -1.5]
    rcs[:, 2, 1] = [1.2, 1.4]
    zcc[:, 2, 1] = [2.0, 2.5]
    zss[:, 2, 1] = [-0.2, -0.3]

    rcos = _mn_cos_to_signed_host(rcc, rss, maps=maps, ncoeff=modes.m.size)
    zsin = _mn_sin_to_signed_host(zsc, zcs, maps=maps, ncoeff=modes.m.size)
    rsin = _mn_sin_to_signed_host(rsc, rcs, maps=maps, ncoeff=modes.m.size)
    zcos = _mn_cos_to_signed_host(zcc, zss, maps=maps, ncoeff=modes.m.size)

    internal = vmec_m1_physical_to_internal_signed(
        Rcos=rcos,
        Zsin=zsin,
        Rsin=rsin,
        Zcos=zcos,
        modes=modes,
        lthreed=True,
        lasym=True,
        lconm1=True,
    )

    _rcc_i, rss_i = _signed_to_mn_cos_host(np.asarray(internal[0]), maps=maps)
    _zsc_i, zcs_i = _signed_to_mn_sin_host(np.asarray(internal[1]), maps=maps)
    rsc_i, rcs_i = _signed_to_mn_sin_host(np.asarray(internal[2]), maps=maps)
    zcc_i, zss_i = _signed_to_mn_cos_host(np.asarray(internal[3]), maps=maps)

    np.testing.assert_allclose(rss_i[:, 1, 1], 0.5 * (rss[:, 1, 1] + zcs[:, 1, 1]))
    np.testing.assert_allclose(zcs_i[:, 1, 1], 0.5 * (rss[:, 1, 1] - zcs[:, 1, 1]))
    np.testing.assert_allclose(rsc_i[:, 1, 1], 0.5 * (rsc[:, 1, 1] + zcc[:, 1, 1]))
    np.testing.assert_allclose(zcc_i[:, 1, 1], 0.5 * (rsc[:, 1, 1] - zcc[:, 1, 1]))
    np.testing.assert_allclose(rss_i[:, 2, 1], rss[:, 2, 1])
    np.testing.assert_allclose(zcs_i[:, 2, 1], zcs[:, 2, 1])
    np.testing.assert_allclose(rcs_i[:, 2, 1], rcs[:, 2, 1])
    np.testing.assert_allclose(zss_i[:, 2, 1], zss[:, 2, 1])

    physical = vmec_m1_internal_to_physical_signed(
        Rcos=internal[0],
        Zsin=internal[1],
        Rsin=internal[2],
        Zcos=internal[3],
        modes=modes,
        lthreed=True,
        lasym=True,
        lconm1=True,
    )
    for expected, got in zip((rcos, zsin, rsin, zcos), physical):
        np.testing.assert_allclose(np.asarray(got), expected, rtol=1.0e-12, atol=1.0e-12)
