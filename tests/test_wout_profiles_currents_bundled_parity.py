from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vmec_jax.config import load_config
from vmec_jax.energy import _iotaf_from_iotas, flux_profiles_from_indata
from vmec_jax.integrals import cumrect_s_halfmesh
from vmec_jax.profiles import MU0, eval_profiles
from vmec_jax.wout import read_wout


PROFILE_CASES = (
    (
        "axisymmetric_finite_beta",
        "input.shaped_tokamak_pressure",
        "wout_shaped_tokamak_pressure.nc",
        True,
    ),
    (
        "current_driven_3d",
        "input.nfp4_QH_warm_start",
        "wout_nfp4_QH_warm_start.nc",
        False,
    ),
    (
        "qi_stel_seed_3127",
        "input.QI_stel_seed_3127",
        "wout_QI_stel_seed_3127.nc",
        False,
    ),
    (
        "finite_beta_3d",
        "input.li383_low_res",
        "wout_li383_low_res.nc",
        True,
    ),
    (
        "cth",
        "input.cth_like_fixed_bdy",
        "wout_cth_like_fixed_bdy.nc",
        True,
    ),
    (
        "lasym_3d",
        "input.basic_non_stellsym_simsopt",
        "wout_basic_non_stellsym_simsopt.nc",
        False,
    ),
)

CURRENT_CASES = (
    ("axisymmetric", "wout_circular_tokamak.nc"),
    ("axisymmetric_finite_beta", "wout_shaped_tokamak_pressure.nc"),
    ("current_driven_3d", "wout_nfp4_QH_warm_start.nc"),
    ("qi_stel_seed_3127", "wout_QI_stel_seed_3127.nc"),
    ("finite_beta_3d", "wout_li383_low_res.nc"),
    ("cth", "wout_cth_like_fixed_bdy.nc"),
    ("lasym_3d", "wout_basic_non_stellsym_simsopt.nc"),
)


def _data_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "examples" / "data"


def _s_full(ns: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, ns)


def _s_half(s: np.ndarray) -> np.ndarray:
    if s.size < 2:
        return s.copy()
    return np.concatenate([s[:1], 0.5 * (s[1:] + s[:-1])])


def _vmec_half_from_full(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return values.copy()

    out = np.zeros_like(values)
    if values.size >= 3:
        out[0] = 1.5 * values[1] - 0.5 * values[2]
    else:
        out[0] = values[1]
    out[1:-1] = 0.5 * (values[1:-1] + values[2:])
    out[-1] = 1.5 * values[-1] - 0.5 * values[-2]
    return out


@pytest.mark.parametrize(
    ("case_name", "input_name", "wout_name", "expect_pressure"),
    PROFILE_CASES,
    ids=[case[0] for case in PROFILE_CASES],
)
def test_bundled_wout_flux_pressure_iota_profiles_follow_vmec_radial_mesh(
    case_name: str,
    input_name: str,
    wout_name: str,
    expect_pressure: bool,
) -> None:
    """Solved bundled wouts should preserve VMEC input profiles and radial staggering."""
    pytest.importorskip("jax")
    pytest.importorskip("netCDF4")
    from vmec_jax._compat import enable_x64

    enable_x64(True)

    data_dir = _data_dir()
    _cfg, indata = load_config(str(data_dir / input_name))
    wout = read_wout(data_dir / wout_name)
    ns = int(wout.ns)
    s = _s_full(ns)

    flux = flux_profiles_from_indata(indata, s, signgs=int(wout.signgs))
    phipf_expected = np.asarray(flux.phipf, dtype=float) * (2.0 * np.pi * int(wout.signgs))
    phi_expected = np.asarray(cumrect_s_halfmesh(phipf_expected, s), dtype=float)

    np.testing.assert_allclose(
        np.asarray(wout.phipf, dtype=float),
        phipf_expected,
        rtol=1.0e-13,
        atol=1.0e-13,
        err_msg=f"{case_name}: phipf does not match input PHIEDGE/APHI profile",
    )
    np.testing.assert_allclose(
        np.asarray(wout.phi, dtype=float),
        phi_expected,
        rtol=1.0e-13,
        atol=1.0e-13,
        err_msg=f"{case_name}: phi is not the VMEC half-mesh integral of phipf",
    )

    profiles = eval_profiles(indata, _s_half(s))
    pressure_expected = np.array(profiles.get("pressure", np.zeros((ns,))), dtype=float, copy=True)
    if pressure_expected.size:
        pressure_expected[0] = 0.0

    if expect_pressure:
        assert np.max(np.abs(np.asarray(wout.pres, dtype=float))) > 0.0
    np.testing.assert_allclose(
        np.asarray(wout.pres, dtype=float),
        pressure_expected,
        rtol=1.0e-12,
        atol=1.0e-14,
        err_msg=f"{case_name}: pressure profile drifted from VMEC input profile",
    )
    np.testing.assert_allclose(
        np.asarray(wout.presf, dtype=float),
        _vmec_half_from_full(pressure_expected),
        rtol=1.0e-12,
        atol=1.0e-14,
        err_msg=f"{case_name}: presf does not follow VMEC full-to-half radial stencil",
    )

    hs = 1.0 / float(ns - 1)
    wp_expected = hs * float(np.sum(np.asarray(wout.vp, dtype=float)[1:] * np.asarray(wout.pres, dtype=float)[1:]))
    np.testing.assert_allclose(
        float(wout.wp),
        wp_expected,
        rtol=1.0e-13,
        atol=1.0e-14,
        err_msg=f"{case_name}: wp is not the VMEC half-mesh pressure integral",
    )
    assert float(wout.wb) > 0.0, f"{case_name}: bundled fixture lacks positive magnetic energy"
    np.testing.assert_allclose(
        float(wout.betatotal),
        float(wout.wp) / float(wout.wb),
        rtol=1.0e-13,
        atol=1.0e-14,
        err_msg=f"{case_name}: betatotal no longer matches wp / wb",
    )
    beta_scalars = np.asarray([wout.betatotal, wout.betapol, wout.betator, wout.betaxis], dtype=float)
    assert np.all(np.isfinite(beta_scalars)), f"{case_name}: beta scalars must be finite"
    if expect_pressure:
        assert np.all(beta_scalars > 0.0), f"{case_name}: finite-pressure fixture lost positive beta scalars"
    else:
        np.testing.assert_allclose(
            beta_scalars,
            0.0,
            rtol=0.0,
            atol=0.0,
            err_msg=f"{case_name}: zero-pressure fixture gained finite beta scalars",
        )

    iotaf_expected = np.asarray(
        _iotaf_from_iotas(
            np.asarray(wout.iotas, dtype=float),
            lrfp=bool(indata.get_bool("LRFP", False)),
        ),
        dtype=float,
    )
    np.testing.assert_allclose(
        np.asarray(wout.iotaf, dtype=float),
        iotaf_expected,
        rtol=1.0e-13,
        atol=1.0e-13,
        err_msg=f"{case_name}: iotaf does not follow VMEC iotas radial stencil",
    )

    if int(indata.get_int("NCURR", 0)) == 0:
        iotas_expected = np.array(profiles.get("iota", np.zeros((ns,))), dtype=float, copy=True)
        if iotas_expected.size:
            iotas_expected[0] = 0.0
        np.testing.assert_allclose(
            np.asarray(wout.iotas, dtype=float),
            iotas_expected,
            rtol=1.0e-13,
            atol=1.0e-13,
            err_msg=f"{case_name}: iotas drifted from VMEC input profile",
        )


@pytest.mark.parametrize("case_name,wout_name", CURRENT_CASES, ids=[case[0] for case in CURRENT_CASES])
def test_bundled_wout_surface_averaged_currents_follow_ampere_radial_difference(
    case_name: str,
    wout_name: str,
) -> None:
    """Stored J profiles should match VMEC's finite-difference of buco/bvco."""
    pytest.importorskip("netCDF4")

    wout = read_wout(_data_dir() / wout_name)
    ns = int(wout.ns)
    assert ns >= 4

    hs = 1.0 / float(ns - 1)
    signgs = float(wout.signgs)
    buco = np.asarray(wout.buco, dtype=float)
    bvco = np.asarray(wout.bvco, dtype=float)

    jcuru_expected = -signgs * (bvco[2:] - bvco[1:-1]) / (hs * MU0)
    jcurv_expected = signgs * (buco[2:] - buco[1:-1]) / (hs * MU0)
    jcuru = np.asarray(wout.jcuru, dtype=float)[1:-1]
    jcurv = np.asarray(wout.jcurv, dtype=float)[1:-1]

    assert np.max(np.abs(jcuru)) > 0.0 or np.max(np.abs(jcurv)) > 0.0
    np.testing.assert_allclose(
        jcuru,
        jcuru_expected,
        rtol=1.0e-12,
        atol=1.0e-9,
        err_msg=f"{case_name}: jcuru no longer matches -d(bvco)/ds / mu0",
    )
    np.testing.assert_allclose(
        jcurv,
        jcurv_expected,
        rtol=1.0e-12,
        atol=1.0e-9,
        err_msg=f"{case_name}: jcurv no longer matches d(buco)/ds / mu0",
    )
