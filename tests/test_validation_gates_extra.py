from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from vmec_jax.namelist import read_indata
from vmec_jax.wout import read_wout


@dataclass(frozen=True)
class MaterializedWoutCase:
    name: str
    wout_rel: str
    expect_lasym: bool
    expect_finite_beta: bool


MATERIALIZED_WOUT_CASES = (
    MaterializedWoutCase(
        "axisymmetric_zero_beta",
        "examples/data/wout_circular_tokamak.nc",
        expect_lasym=False,
        expect_finite_beta=False,
    ),
    MaterializedWoutCase(
        "axisymmetric_finite_beta",
        "examples/data/wout_shaped_tokamak_pressure.nc",
        expect_lasym=False,
        expect_finite_beta=True,
    ),
    MaterializedWoutCase(
        "three_d_finite_beta",
        "examples/data/wout_li383_low_res.nc",
        expect_lasym=False,
        expect_finite_beta=True,
    ),
    MaterializedWoutCase(
        "lasym_finite_beta",
        "examples_single_grid/data/wout_basic_non_stellsym_pressure_reference.nc",
        expect_lasym=True,
        expect_finite_beta=True,
    ),
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _max_abs(value: object) -> float:
    arr = np.asarray(value, dtype=float)
    return float(np.max(np.abs(arr)))


@pytest.mark.parametrize("case", MATERIALIZED_WOUT_CASES, ids=[case.name for case in MATERIALIZED_WOUT_CASES])
def test_materialized_wouts_close_volume_pressure_and_beta_identities(case: MaterializedWoutCase) -> None:
    """Cheap geometry/energy gates for shipped VMEC2000-produced equilibria."""
    pytest.importorskip("netCDF4")

    wout_path = _repo_root() / case.wout_rel
    if not wout_path.exists():
        pytest.skip(f"Missing materialized wout fixture: {case.name}")

    wout = read_wout(wout_path)
    ns = int(wout.ns)
    assert ns >= 3
    assert bool(wout.lasym) is case.expect_lasym

    vp = np.asarray(wout.vp, dtype=float)
    pres = np.asarray(wout.pres, dtype=float)
    phi = np.asarray(wout.phi, dtype=float)
    phipf = np.asarray(wout.phipf, dtype=float)

    for name, arr in (("vp", vp), ("pres", pres), ("phi", phi), ("phipf", phipf)):
        assert arr.shape == (ns,), name
        assert np.all(np.isfinite(arr)), name

    assert float(wout.wb) > 0.0
    assert vp[0] == pytest.approx(0.0, abs=0.0)
    assert np.all(vp[1:] > 0.0)

    volume_from_half_mesh = (4.0 * np.pi**2) * float(np.sum(vp[1:]) / float(ns - 1))
    np.testing.assert_allclose(
        volume_from_half_mesh,
        float(wout.volume_p),
        rtol=2.0e-6,
        atol=1.0e-12,
        err_msg=f"{case.name}: volume_p is not the VMEC half-mesh integral of vp",
    )

    wp_from_profile = float(np.sum(vp[1:] * pres[1:]) / float(ns - 1))
    np.testing.assert_allclose(
        wp_from_profile,
        float(wout.wp),
        rtol=2.0e-11,
        atol=1.0e-13,
        err_msg=f"{case.name}: pressure energy wp drifted from vp*pres integral",
    )
    np.testing.assert_allclose(
        float(wout.betatotal),
        float(wout.wp) / float(wout.wb),
        rtol=1.0e-13,
        atol=1.0e-15,
        err_msg=f"{case.name}: betatotal no longer closes as wp/wb",
    )

    assert phi[0] == pytest.approx(0.0, abs=1.0e-14)
    if abs(phi[-1]) > 0.0:
        assert np.all(np.diff(phi) * np.sign(phi[-1]) >= -1.0e-12)
        assert np.all(phipf * np.sign(phi[-1]) >= 0.0)

    beta_scalars = np.asarray([wout.betapol, wout.betator, wout.betatotal, wout.betaxis], dtype=float)
    assert np.all(np.isfinite(beta_scalars))
    if case.expect_finite_beta:
        assert float(wout.wp) > 0.0
        assert np.max(pres[1:]) > 0.0
        assert np.all(beta_scalars > 0.0)
    else:
        np.testing.assert_allclose(beta_scalars, 0.0, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(pres, 0.0, rtol=0.0, atol=0.0)


def test_lasym_finite_beta_reference_fixture_has_asymmetric_pressure_signal() -> None:
    """Asset gate for the no-solve LASYM=true finite-beta reference."""
    pytest.importorskip("netCDF4")

    repo_root = _repo_root()
    input_path = repo_root / "examples_single_grid/data/input.basic_non_stellsym_pressure"
    wout_path = repo_root / "examples_single_grid/data/wout_basic_non_stellsym_pressure_reference.nc"
    if not input_path.exists() or not wout_path.exists():
        pytest.skip("Missing bundled LASYM finite-beta reference input/wout")

    indata = read_indata(input_path)
    wout = read_wout(wout_path)

    assert bool(indata.get_bool("LASYM")) is True
    assert bool(wout.lasym) is True
    assert int(wout.ntor) > 0
    assert int(wout.nfp) == int(indata.get_int("NFP"))
    assert int(wout.mpol) == int(indata.get_int("MPOL"))
    assert int(wout.ntor) == int(indata.get_int("NTOR"))

    assert _max_abs(wout.rmns) > 1.0e-3
    assert _max_abs(wout.zmnc) > 1.0e-3
    assert _max_abs(wout.bmns) > 1.0e-2
    assert _max_abs(wout.gmns) > 1.0e-2
    assert _max_abs(wout.bsubumns) > 1.0e-2
    assert _max_abs(wout.bsubvmns) > 1.0e-2

    pres = np.asarray(wout.pres, dtype=float)
    assert pres[0] == pytest.approx(0.0, abs=0.0)
    assert np.all(pres[1:] > 0.0)
    assert np.all(np.diff(pres[1:]) < 0.0)

    np.testing.assert_allclose(
        float(wout.phi[-1]),
        float(indata.get_float("PHIEDGE")),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert float(wout.wp) > 0.0
    assert float(wout.betaxis) > float(wout.betatotal) > 0.0


@pytest.mark.parametrize(
    "wout_rel,min_abs_d_r",
    (
        ("examples/data/wout_shaped_tokamak_pressure.nc", 1.0e-7),
        ("examples/data/wout_li383_low_res.nc", 1.0e-2),
        ("examples_single_grid/data/wout_basic_non_stellsym_pressure_reference.nc", 1.0e-4),
    ),
    ids=("axisymmetric_pressure", "three_d_finite_beta", "lasym_finite_beta"),
)
def test_materialized_finite_beta_glasser_profiles_are_self_consistent(
    wout_rel: str,
    min_abs_d_r: float,
) -> None:
    """No-solve gate for persisted/fallback Glasser profiles on finite-beta wouts."""

    pytest.importorskip("netCDF4")
    wout_path = _repo_root() / wout_rel
    if not wout_path.exists():
        pytest.skip(f"Missing materialized finite-beta wout fixture: {wout_rel}")

    wout = read_wout(wout_path)
    ns = int(wout.ns)
    dmerc = np.asarray(wout.DMerc, dtype=float)
    d_r = np.asarray(wout.D_R, dtype=float)
    h_glasser = np.asarray(wout.H, dtype=float)
    correction = np.asarray(wout.glasser_correction, dtype=float)
    valid = np.asarray(wout.glasser_shear_valid, dtype=bool)

    for name, arr in (
        ("DMerc", dmerc),
        ("D_R", d_r),
        ("HGlasser", h_glasser),
        ("GlasserCorrection", correction),
    ):
        assert arr.shape == (ns,), name
        assert np.all(np.isfinite(arr)), name
    assert valid.shape == (ns,)
    assert int(np.count_nonzero(valid[1:-1])) >= max(ns - 4, 1)

    np.testing.assert_allclose(
        d_r,
        -dmerc + correction,
        rtol=1.0e-11,
        atol=1.0e-13,
        err_msg=f"{wout_rel}: D_R no longer equals -DMerc + GlasserCorrection",
    )
    assert np.all(correction[1:-1] >= -1.0e-14)
    assert float(np.nanmax(np.abs(d_r[1:-1]))) > min_abs_d_r
    assert float(wout.betatotal) > 0.0
