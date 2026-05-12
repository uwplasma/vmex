from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from vmec_jax.config import load_config
from vmec_jax.static import build_static
from vmec_jax.wout import equilibrium_aspect_ratio_from_state, read_wout, state_from_wout


GEOMETRY_CASES = (
    ("circular", "input.circular_tokamak", "wout_circular_tokamak.nc"),
    ("finite_beta", "input.shaped_tokamak_pressure", "wout_shaped_tokamak_pressure.nc"),
    ("qh", "input.nfp4_QH_warm_start", "wout_nfp4_QH_warm_start.nc"),
    ("qi", "input.nfp3_QI_fixed_resolution_final", "wout_nfp3_QI_fixed_resolution_final.nc"),
    ("dshape", "input.DSHAPE", "wout_DSHAPE.nc"),
    ("cth", "input.cth_like_fixed_bdy", "wout_cth_like_fixed_bdy.nc"),
)

JXB_MERCIER_CASES = (
    "wout_circular_tokamak.nc",
    "wout_shaped_tokamak_pressure.nc",
    "wout_nfp4_QH_warm_start.nc",
    "wout_LandremanPaul2021_QA_lowres.nc",
    "wout_nfp3_QI_fixed_resolution_final.nc",
    "wout_DSHAPE.nc",
    "wout_cth_like_fixed_bdy.nc",
    "wout_li383_low_res.nc",
)


def _data_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "examples" / "data"


def _static_aligned_to_wout(cfg, wout):
    cfg = replace(
        cfg,
        ns=int(wout.ns),
        mpol=int(wout.mpol),
        ntor=int(wout.ntor),
        nfp=int(wout.nfp),
        lasym=bool(wout.lasym),
        lthreed=bool(int(wout.ntor) > 0),
    )
    return build_static(cfg)


@pytest.mark.parametrize(("case_name", "input_name", "wout_name"), GEOMETRY_CASES)
def test_bundled_wout_aspectratio_matches_state_geometry(
    case_name: str,
    input_name: str,
    wout_name: str,
) -> None:
    """VMEC2000 aspect-ratio scalars should match geometry reconstructed from the solved state."""
    pytest.importorskip("jax")
    pytest.importorskip("netCDF4")

    data_dir = _data_dir()
    cfg, _indata = load_config(str(data_dir / input_name))
    wout = read_wout(data_dir / wout_name)
    state = state_from_wout(wout)
    aspect = float(np.asarray(equilibrium_aspect_ratio_from_state(state=state, static=_static_aligned_to_wout(cfg, wout))))

    assert float(wout.aspect) > 0.0, f"{case_name}: bundled fixture lacks populated aspect scalar"
    assert float(wout.Aminor_p) > 0.0, f"{case_name}: bundled fixture lacks populated Aminor_p scalar"
    assert float(wout.Rmajor_p) > 0.0, f"{case_name}: bundled fixture lacks populated Rmajor_p scalar"
    assert float(wout.volume_p) > 0.0, f"{case_name}: bundled fixture lacks populated volume_p scalar"

    np.testing.assert_allclose(aspect, float(wout.aspect), rtol=5.0e-13, atol=5.0e-13)
    np.testing.assert_allclose(
        float(wout.Rmajor_p) / float(wout.Aminor_p),
        float(wout.aspect),
        rtol=5.0e-13,
        atol=5.0e-13,
    )


@pytest.mark.parametrize("wout_name", JXB_MERCIER_CASES)
def test_bundled_wout_mercier_and_jxbforce_profile_closures(wout_name: str) -> None:
    """Bundled Mercier/JXBFORCE profiles should obey VMEC decomposition and endpoint rules."""
    pytest.importorskip("netCDF4")

    wout = read_wout(_data_dir() / wout_name)
    dmerc = np.asarray(wout.DMerc, dtype=float)
    parts = (
        np.asarray(wout.Dshear, dtype=float)
        + np.asarray(wout.Dcurr, dtype=float)
        + np.asarray(wout.Dwell, dtype=float)
        + np.asarray(wout.Dgeod, dtype=float)
    )
    np.testing.assert_allclose(dmerc, parts, rtol=0.0, atol=0.0)
    assert np.all(np.isfinite(dmerc))
    assert np.any(np.abs(dmerc[1:-1]) > 0.0)

    for name in ("jdotb", "bdotb", "bdotgradv"):
        profile = np.asarray(getattr(wout, name), dtype=float)
        assert profile.shape == (int(wout.ns),)
        assert np.all(np.isfinite(profile))
        assert np.any(np.abs(profile[1:-1]) > 0.0)

    jdotb = np.asarray(wout.jdotb, dtype=float)
    bdotb = np.asarray(wout.bdotb, dtype=float)
    bdotgradv = np.asarray(wout.bdotgradv, dtype=float)
    np.testing.assert_allclose(jdotb[0], 2.0 * jdotb[1] - jdotb[2], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(jdotb[-1], 2.0 * jdotb[-2] - jdotb[-3], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(bdotb[0], 2.0 * bdotb[2] - bdotb[1], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(bdotb[-1], 2.0 * bdotb[-2] - bdotb[-3], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(bdotgradv[0], 2.0 * bdotgradv[1] - bdotgradv[2], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(bdotgradv[-1], 2.0 * bdotgradv[-2] - bdotgradv[-3], rtol=0.0, atol=0.0)
