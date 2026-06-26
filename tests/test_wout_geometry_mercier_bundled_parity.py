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
    ("qi_stel_seed_3127", "input.QI_stel_seed_3127", "wout_QI_stel_seed_3127.nc"),
    ("dshape", "input.DSHAPE", "wout_DSHAPE.nc"),
    ("cth", "input.cth_like_fixed_bdy", "wout_cth_like_fixed_bdy.nc"),
)

JXB_MERCIER_CASES = (
    "wout_circular_tokamak.nc",
    "wout_shaped_tokamak_pressure.nc",
    "wout_nfp4_QH_warm_start.nc",
    "wout_LandremanPaul2021_QA_lowres.nc",
    "wout_nfp3_QI_fixed_resolution_final.nc",
    "wout_QI_stel_seed_3127.nc",
    "wout_DSHAPE.nc",
    "wout_cth_like_fixed_bdy.nc",
    "wout_li383_low_res.nc",
)


AXIS_MODE_CASES = (
    ("axisym", "examples/data/wout_circular_tokamak.nc"),
    ("nonaxis", "examples/data/wout_nfp4_QH_warm_start.nc"),
    ("lasym_nonaxis", "examples/data/wout_basic_non_stellsym_simsopt.nc"),
    ("single_grid_lasym_axisym", "examples/data/single_grid/wout_up_down_asymmetric_tokamak_reference.nc"),
    (
        "single_grid_lasym_nonaxis_finite_beta",
        "examples/data/single_grid/wout_basic_non_stellsym_pressure_reference.nc",
    ),
)


JACOBIAN_VOLUME_CASES = (
    ("axisym", "examples/data/wout_circular_tokamak.nc"),
    ("finite_beta_axisym", "examples/data/wout_shaped_tokamak_pressure.nc"),
    ("nonaxis", "examples/data/wout_nfp4_QH_warm_start.nc"),
    ("lasym_nonaxis", "examples/data/wout_basic_non_stellsym_simsopt.nc"),
    (
        "single_grid_lasym_nonaxis_finite_beta",
        "examples/data/single_grid/wout_basic_non_stellsym_pressure_reference.nc",
    ),
)


def _data_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "examples" / "data"


def _single_grid_data_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "examples/data/single_grid"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_wout_or_skip(path: Path):
    if not path.exists():
        pytest.skip(f"optional bundled WOUT fixture is not available: {path}")
    return read_wout(path)


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


@pytest.mark.parametrize(("case_name", "wout_rel"), AXIS_MODE_CASES)
def test_bundled_wout_axis_metadata_matches_axis_fourier_modes(case_name: str, wout_rel: str) -> None:
    """Axis metadata should stay identical to the m=0 axis Fourier coefficients."""
    pytest.importorskip("netCDF4")

    wout = _read_wout_or_skip(_repo_root() / wout_rel)
    ntor = int(wout.ntor)
    nfp = int(wout.nfp)

    expected_raxis_cc = np.zeros(ntor + 1)
    expected_raxis_cs = np.zeros(ntor + 1)
    expected_zaxis_cc = np.zeros(ntor + 1)
    expected_zaxis_cs = np.zeros(ntor + 1)
    for mode_idx, (m_mode, n_mode) in enumerate(zip(np.asarray(wout.xm, dtype=int), np.asarray(wout.xn, dtype=int))):
        if int(m_mode) != 0:
            continue
        n_idx = abs(int(n_mode)) // nfp
        if n_idx > ntor:
            continue
        expected_raxis_cc[n_idx] = float(wout.rmnc[0, mode_idx])
        expected_raxis_cs[n_idx] = float(wout.rmns[0, mode_idx])
        expected_zaxis_cc[n_idx] = float(wout.zmnc[0, mode_idx])
        expected_zaxis_cs[n_idx] = float(wout.zmns[0, mode_idx])

    np.testing.assert_allclose(np.asarray(wout.raxis_cc), expected_raxis_cc, rtol=0.0, atol=0.0, err_msg=case_name)
    np.testing.assert_allclose(np.asarray(wout.raxis_cs), expected_raxis_cs, rtol=0.0, atol=0.0, err_msg=case_name)
    np.testing.assert_allclose(np.asarray(wout.zaxis_cc), expected_zaxis_cc, rtol=0.0, atol=0.0, err_msg=case_name)
    np.testing.assert_allclose(np.asarray(wout.zaxis_cs), expected_zaxis_cs, rtol=0.0, atol=0.0, err_msg=case_name)


@pytest.mark.parametrize(("case_name", "wout_rel"), JACOBIAN_VOLUME_CASES)
def test_bundled_wout_vp_matches_jacobian_constant_mode(case_name: str, wout_rel: str) -> None:
    """The VMEC volume derivative is signgs times the sqrt(g) m=n=0 mode."""
    pytest.importorskip("netCDF4")

    wout = _read_wout_or_skip(_repo_root() / wout_rel)
    mask = (np.asarray(wout.xm_nyq, dtype=int) == 0) & (np.asarray(wout.xn_nyq, dtype=int) == 0)
    assert np.count_nonzero(mask) == 1, case_name
    mode_idx = int(np.flatnonzero(mask)[0])

    np.testing.assert_allclose(
        np.asarray(wout.gmns, dtype=float)[:, mode_idx],
        0.0,
        rtol=0.0,
        atol=0.0,
        err_msg=f"{case_name}: sine sqrt(g) constant mode must stay zero",
    )
    np.testing.assert_allclose(
        np.asarray(wout.vp, dtype=float),
        float(wout.signgs) * np.asarray(wout.gmnc, dtype=float)[:, mode_idx],
        rtol=1.0e-14,
        atol=5.0e-14,
        err_msg=f"{case_name}: vp no longer closes with the gmnc m=n=0 Jacobian mode",
    )


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
    for name in ("DMerc", "Dshear", "Dcurr", "Dwell", "Dgeod"):
        values = np.asarray(getattr(wout, name), dtype=float)
        assert values.shape == (int(wout.ns),), name
        assert np.all(np.isfinite(values)), name
        np.testing.assert_allclose(values[[0, -1]], 0.0, rtol=0.0, atol=0.0, err_msg=name)
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


def test_single_grid_lasym_finite_beta_wout_diagnostic_closures() -> None:
    """The bundled LASYM=T finite-beta fixture should exercise asymmetric JXB/Mercier profiles."""
    pytest.importorskip("netCDF4")

    wout = read_wout(_single_grid_data_dir() / "wout_basic_non_stellsym_pressure_reference.nc")
    assert bool(wout.lasym)
    assert float(wout.wp) > 0.0
    assert float(wout.wb) > 0.0
    np.testing.assert_allclose(float(wout.betatotal), float(wout.wp) / float(wout.wb), rtol=1.0e-13)

    dmerc = np.asarray(wout.DMerc, dtype=float)
    dmerc_parts = (
        np.asarray(wout.Dshear, dtype=float)
        + np.asarray(wout.Dcurr, dtype=float)
        + np.asarray(wout.Dwell, dtype=float)
        + np.asarray(wout.Dgeod, dtype=float)
    )
    np.testing.assert_allclose(dmerc, dmerc_parts, rtol=0.0, atol=0.0)
    for name in ("DMerc", "Dshear", "Dcurr", "Dwell", "Dgeod"):
        np.testing.assert_allclose(np.asarray(getattr(wout, name), dtype=float)[[0, -1]], 0.0, rtol=0.0, atol=0.0)
    assert np.linalg.norm(dmerc[1:-1]) > 0.0

    for name in ("rmns", "zmnc", "lmnc", "bmns", "bsubumns", "bsubvmns"):
        values = np.asarray(getattr(wout, name), dtype=float)
        assert values.ndim == 2
        assert np.all(np.isfinite(values)), name
        assert np.linalg.norm(values) > 0.0, name

    for name in ("jdotb", "bdotb", "bdotgradv"):
        profile = np.asarray(getattr(wout, name), dtype=float)
        assert profile.shape == (int(wout.ns),)
        assert np.all(np.isfinite(profile))
        assert np.any(np.abs(profile[1:-1]) > 0.0)
