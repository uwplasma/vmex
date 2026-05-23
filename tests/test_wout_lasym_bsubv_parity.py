from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax.config import load_config
from vmec_jax.static import build_static
from vmec_jax.vmec_tomnsp import vmec_angle_grid
from vmec_jax.wout import (
    read_wout,
    state_from_wout,
    wout_minimal_from_fixed_boundary,
    write_wout,
)


def _rms(value) -> float:
    arr = np.asarray(value, dtype=float)
    return float(np.sqrt(np.mean(arr * arr)))


def _abs_rms(got, expected) -> float:
    return _rms(np.asarray(got, dtype=float) - np.asarray(expected, dtype=float))


@pytest.fixture(scope="module")
def up_down_lasym_reference_generation(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    pytest.importorskip("jax")
    pytest.importorskip("netCDF4")

    root = Path(__file__).resolve().parents[1]
    input_path = root / "examples_single_grid/data/input.up_down_asymmetric_tokamak"
    wout_path = root / "examples_single_grid/data/wout_up_down_asymmetric_tokamak_reference.nc"
    if not input_path.exists() or not wout_path.exists():
        pytest.skip("Missing bundled up_down_asymmetric_tokamak VMEC2000 reference")

    cfg, indata = load_config(str(input_path))
    wref = read_wout(wout_path)
    cfg = replace(
        cfg,
        ns=int(wref.ns),
        mpol=int(wref.mpol),
        ntor=int(wref.ntor),
        nfp=int(wref.nfp),
        lasym=bool(wref.lasym),
        lthreed=bool(int(wref.ntor) > 0),
    )
    grid = vmec_angle_grid(
        ntheta=int(cfg.ntheta),
        nzeta=int(cfg.nzeta),
        nfp=int(cfg.nfp),
        lasym=bool(cfg.lasym),
    )
    static = build_static(cfg, grid=grid)
    state = state_from_wout(wref)

    out_path = tmp_path_factory.mktemp("wout_lasym_updown") / "wout_up_down_asymmetric_tokamak_vmec_jax.nc"
    wnew = wout_minimal_from_fixed_boundary(
        path=out_path,
        state=state,
        static=static,
        indata=indata,
        signgs=int(wref.signgs),
        fsqr=0.0,
        fsqz=0.0,
        fsql=0.0,
    )
    write_wout(out_path, wnew, overwrite=True)

    return SimpleNamespace(wref=wref, wnew=wnew, roundtrip=read_wout(out_path))


def test_up_down_lasym_reference_state_bsubvmns_uses_iequi_asym_source(
    up_down_lasym_reference_generation: SimpleNamespace,
) -> None:
    wref = up_down_lasym_reference_generation.wref
    wnew = up_down_lasym_reference_generation.wnew

    assert bool(wnew.lasym) is True
    assert _rms(wref.bsubvmns) < 1.0e-5
    assert _abs_rms(wnew.bsubvmns, wref.bsubvmns) < 5.0e-5

    # Guard channels that regressed when the full LASYM bsubu/bsubv output was
    # switched to the IEQUI source instead of only the asymmetric bsubv channel.
    assert _abs_rms(wnew.bsubvmnc, wref.bsubvmnc) < 5.0e-4
    assert _abs_rms(wnew.bsubumns, wref.bsubumns) < 1.0e-12
    assert _abs_rms(wnew.bsupumns, wref.bsupumns) < 1.0e-12


def test_up_down_lasym_generated_wout_roundtrips_jxbforce_profiles(
    up_down_lasym_reference_generation: SimpleNamespace,
) -> None:
    wref = up_down_lasym_reference_generation.wref
    wnew = up_down_lasym_reference_generation.wnew
    reread = up_down_lasym_reference_generation.roundtrip

    assert bool(reread.lasym) is True
    assert int(reread.ns) == int(wref.ns)
    assert int(reread.mnmax_nyq) == int(wref.mnmax_nyq)

    for name in ("bdotb", "bdotgradv"):
        np.testing.assert_allclose(
            np.asarray(getattr(wnew, name), dtype=float),
            np.asarray(getattr(wref, name), dtype=float),
            rtol=5.0e-13,
            atol=5.0e-13,
        )
        np.testing.assert_allclose(
            np.asarray(getattr(reread, name), dtype=float),
            np.asarray(getattr(wnew, name), dtype=float),
            rtol=0.0,
            atol=0.0,
        )

    for name in ("gmns", "bmns", "bsubumns", "bsubvmns", "bsupumns", "bsupvmns"):
        arr = np.asarray(getattr(reread, name), dtype=float)
        assert arr.shape == (int(wref.ns), int(wref.mnmax_nyq)), name
        assert np.all(np.isfinite(arr)), name

    np.testing.assert_allclose(
        np.asarray(reread.DMerc, dtype=float),
        np.asarray(reread.Dshear, dtype=float)
        + np.asarray(reread.Dwell, dtype=float)
        + np.asarray(reread.Dcurr, dtype=float)
        + np.asarray(reread.Dgeod, dtype=float),
        rtol=0.0,
        atol=0.0,
    )


@pytest.fixture(scope="module")
def basic_non_stellsym_pressure_reference_generation(
    tmp_path_factory: pytest.TempPathFactory,
) -> SimpleNamespace:
    pytest.importorskip("jax")
    pytest.importorskip("netCDF4")

    root = Path(__file__).resolve().parents[1]
    input_path = root / "examples_single_grid/data/input.basic_non_stellsym_pressure"
    wout_path = root / "examples_single_grid/data/wout_basic_non_stellsym_pressure_reference.nc"
    if not input_path.exists() or not wout_path.exists():
        pytest.skip("Missing bundled basic_non_stellsym_pressure VMEC2000 reference")

    cfg, indata = load_config(str(input_path))
    wref = read_wout(wout_path)
    cfg = replace(
        cfg,
        ns=int(wref.ns),
        mpol=int(wref.mpol),
        ntor=int(wref.ntor),
        nfp=int(wref.nfp),
        lasym=bool(wref.lasym),
        lthreed=bool(int(wref.ntor) > 0),
    )
    grid = vmec_angle_grid(
        ntheta=int(cfg.ntheta),
        nzeta=int(cfg.nzeta),
        nfp=int(cfg.nfp),
        lasym=bool(cfg.lasym),
    )
    static = build_static(cfg, grid=grid)
    state = state_from_wout(wref)

    out_path = tmp_path_factory.mktemp("wout_lasym_pressure") / "wout_basic_non_stellsym_pressure_vmec_jax.nc"
    wnew = wout_minimal_from_fixed_boundary(
        path=out_path,
        state=state,
        static=static,
        indata=indata,
        signgs=int(wref.signgs),
        fsqr=0.0,
        fsqz=0.0,
        fsql=0.0,
    )
    write_wout(out_path, wnew, overwrite=True)

    return SimpleNamespace(wref=wref, wnew=wnew, roundtrip=read_wout(out_path))


def test_basic_non_stellsym_pressure_generation_preserves_finite_beta_scalars(
    basic_non_stellsym_pressure_reference_generation: SimpleNamespace,
) -> None:
    wref = basic_non_stellsym_pressure_reference_generation.wref
    wnew = basic_non_stellsym_pressure_reference_generation.wnew

    assert bool(wnew.lasym) is True
    assert float(wnew.wp) > 0.0
    assert float(wnew.betatotal) > 0.0
    assert np.max(np.abs(np.asarray(wnew.pres, dtype=float)[1:])) > 0.0

    for name in ("pres", "presf"):
        np.testing.assert_allclose(
            np.asarray(getattr(wnew, name), dtype=float),
            np.asarray(getattr(wref, name), dtype=float),
            rtol=1.0e-12,
            atol=1.0e-14,
            err_msg=name,
        )

    for name in (
        "wb",
        "wp",
        "betatotal",
        "betapol",
        "betator",
        "betaxis",
        "volume_p",
        "aspect",
        "Aminor_p",
        "Rmajor_p",
    ):
        np.testing.assert_allclose(
            float(getattr(wnew, name)),
            float(getattr(wref, name)),
            rtol=5.0e-12,
            atol=1.0e-14,
            err_msg=name,
        )
    np.testing.assert_allclose(
        float(wnew.betatotal),
        float(wnew.wp) / float(wnew.wb),
        rtol=1.0e-13,
        atol=1.0e-14,
    )

    np.testing.assert_allclose(
        np.asarray(wnew.bdotb, dtype=float),
        np.asarray(wref.bdotb, dtype=float),
        rtol=5.0e-13,
        atol=5.0e-11,
    )
    np.testing.assert_allclose(
        np.asarray(wnew.bdotgradv, dtype=float),
        np.asarray(wref.bdotgradv, dtype=float),
        rtol=5.0e-13,
        atol=5.0e-13,
    )

    np.testing.assert_allclose(
        np.asarray(wnew.DMerc, dtype=float),
        np.asarray(wnew.Dshear, dtype=float)
        + np.asarray(wnew.Dwell, dtype=float)
        + np.asarray(wnew.Dcurr, dtype=float)
        + np.asarray(wnew.Dgeod, dtype=float),
        rtol=0.0,
        atol=0.0,
    )


def test_basic_non_stellsym_pressure_generation_preserves_lasym_channels(
    basic_non_stellsym_pressure_reference_generation: SimpleNamespace,
) -> None:
    wref = basic_non_stellsym_pressure_reference_generation.wref
    wnew = basic_non_stellsym_pressure_reference_generation.wnew
    reread = basic_non_stellsym_pressure_reference_generation.roundtrip

    assert bool(reread.lasym) is True
    assert int(reread.ns) == int(wref.ns)
    assert int(reread.mnmax) == int(wref.mnmax)
    assert int(reread.mnmax_nyq) == int(wref.mnmax_nyq)

    for name in ("rmns", "zmnc", "lmnc"):
        arr = np.asarray(getattr(wnew, name), dtype=float)
        assert arr.shape == (int(wref.ns), int(wref.mnmax)), name
        assert np.all(np.isfinite(arr)), name
        assert _rms(arr[1:]) > 0.0, name
        np.testing.assert_allclose(
            arr,
            np.asarray(getattr(wref, name), dtype=float),
            rtol=1.0e-13,
            atol=2.0e-13,
        )
        np.testing.assert_allclose(
            np.asarray(getattr(reread, name), dtype=float),
            arr,
            rtol=0.0,
            atol=0.0,
        )

    strict_nyquist_channels = (
        "gmns",
        "bmns",
        "bsupumns",
        "bsupvmns",
        "bsubumns",
        "bsubvmns",
        "bsubumnc",
    )
    for name in strict_nyquist_channels:
        arr = np.asarray(getattr(wnew, name), dtype=float)
        assert arr.shape == (int(wref.ns), int(wref.mnmax_nyq)), name
        assert np.all(np.isfinite(arr)), name
        assert _rms(arr[1:]) > 0.0, name
        np.testing.assert_allclose(
            arr,
            np.asarray(getattr(wref, name), dtype=float),
            rtol=1.0e-10,
            atol=5.0e-12,
        )
        np.testing.assert_allclose(
            np.asarray(getattr(reread, name), dtype=float),
            arr,
            rtol=0.0,
            atol=0.0,
        )

    # The asymmetric-v covariant cosine channel is sensitive to a small IEQUI
    # synthesis difference, but should stay close enough to catch channel swaps
    # or accidental LASYM zeroing in the finite-beta path.
    np.testing.assert_allclose(
        np.asarray(wnew.bsubvmnc, dtype=float),
        np.asarray(wref.bsubvmnc, dtype=float),
        rtol=5.0e-6,
        atol=3.0e-3,
    )
    np.testing.assert_allclose(
        np.asarray(reread.bsubvmns, dtype=float),
        np.asarray(wnew.bsubvmns, dtype=float),
        rtol=0.0,
        atol=0.0,
    )
    assert _rms(np.asarray(wnew.bsubvmns, dtype=float)[1:]) > 0.0
