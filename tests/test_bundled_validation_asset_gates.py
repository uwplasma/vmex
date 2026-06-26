from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from vmec_jax.namelist import read_indata
from vmec_jax.wout import read_wout, write_wout


@dataclass(frozen=True)
class FreeBoundaryAssetCase:
    input_rel: str
    expect_lasym: bool
    expect_mgrid_mode: str


@dataclass(frozen=True)
class RoundtripCase:
    name: str
    wout_rel: str
    expect_lasym: bool


FREE_BOUNDARY_ASSET_CASES = (
    FreeBoundaryAssetCase(
        "examples/data/input.cth_like_free_bdy",
        expect_lasym=False,
        expect_mgrid_mode="R",
    ),
    FreeBoundaryAssetCase(
        "examples/data/input.cth_like_free_bdy_lasym_small",
        expect_lasym=True,
        expect_mgrid_mode="S",
    ),
    FreeBoundaryAssetCase(
        "examples/data/single_grid/input.cth_like_free_bdy",
        expect_lasym=False,
        expect_mgrid_mode="R",
    ),
    FreeBoundaryAssetCase(
        "examples/data/single_grid/input.cth_like_free_bdy_lasym_small",
        expect_lasym=True,
        expect_mgrid_mode="S",
    ),
)


ROUNDTRIP_CASES = (
    RoundtripCase("axisymmetric_vacuum", "examples/data/wout_circular_tokamak.nc", expect_lasym=False),
    RoundtripCase("stellarator_asymmetric", "examples/data/wout_basic_non_stellsym_simsopt.nc", expect_lasym=True),
    pytest.param(
        RoundtripCase("free_boundary", "examples/data/single_grid/wout_cth_like_free_bdy.nc", expect_lasym=False),
        marks=pytest.mark.full,
    ),
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _decode_nc_scalar(value: object) -> str:
    arr = np.asarray(value)
    if arr.dtype.kind in {"S", "U"}:
        return b"".join(np.asarray(arr, dtype="S").ravel()).decode().strip()
    return str(arr.ravel()[0]).strip()


def _assert_allclose_field(name: str, expected: object, actual: object) -> None:
    expected_arr = np.asarray(expected)
    actual_arr = np.asarray(actual)
    assert actual_arr.shape == expected_arr.shape, name
    if expected_arr.dtype.kind in {"b", "i", "u"}:
        np.testing.assert_array_equal(actual_arr, expected_arr, err_msg=name)
    else:
        np.testing.assert_allclose(actual_arr, expected_arr, rtol=0.0, atol=0.0, err_msg=name)


@pytest.mark.parametrize("case", FREE_BOUNDARY_ASSET_CASES, ids=lambda case: Path(case.input_rel).parent.name + ":" + Path(case.input_rel).name)
@pytest.mark.full
def test_free_boundary_inputs_reference_local_mgrid_assets_with_consistent_metadata(case: FreeBoundaryAssetCase) -> None:
    """Free-boundary validation fixtures must be self-contained and metadata-consistent."""
    netcdf4 = pytest.importorskip("netCDF4")

    repo = _repo_root()
    input_path = repo / case.input_rel
    assert input_path.exists(), case.input_rel

    indata = read_indata(input_path)
    assert bool(indata.get_bool("LFREEB")) is True
    assert bool(indata.get_bool("LASYM")) is case.expect_lasym
    assert int(indata.get_int("NFP")) > 0
    assert int(indata.get_int("MPOL")) >= 2
    assert int(indata.get_int("NTOR")) > 0

    mgrid_name = str(indata.scalars["MGRID_FILE"]).strip().strip("'\"")
    mgrid_path = input_path.parent / mgrid_name
    if not mgrid_path.exists():
        pytest.skip(f"{case.input_rel} references missing {mgrid_name}; run tools/fetch_assets.py")
    extcur = np.asarray(indata.scalars["EXTCUR"], dtype=float)
    assert extcur.ndim == 1
    assert extcur.size > 0
    assert np.any(np.abs(extcur) > 0.0)

    with netcdf4.Dataset(mgrid_path) as ds:
        nfp = int(np.asarray(ds.variables["nfp"][:]).ravel()[0])
        nextcur = int(np.asarray(ds.variables["nextcur"][:]).ravel()[0])
        mode = _decode_nc_scalar(ds.variables["mgrid_mode"][:])
        raw_coil_current = np.asarray(ds.variables["raw_coil_cur"][:], dtype=float)
        dims = {name: len(dim) for name, dim in ds.dimensions.items()}

        assert nfp == int(indata.get_int("NFP"))
        assert nextcur >= 1
        assert extcur.size >= nextcur
        assert mode == case.expect_mgrid_mode
        assert raw_coil_current.shape == (nextcur,)
        assert np.all(np.isfinite(raw_coil_current))
        assert dims["rad"] == int(np.asarray(ds.variables["ir"][:]).ravel()[0])
        assert dims["zee"] == int(np.asarray(ds.variables["jz"][:]).ravel()[0])
        assert dims["phi"] == int(np.asarray(ds.variables["kp"][:]).ravel()[0])

        for group in range(1, nextcur + 1):
            suffix = f"{group:03d}"
            for component in ("br", "bp", "bz"):
                name = f"{component}_{suffix}"
                assert name in ds.variables
                field = np.asarray(ds.variables[name][:], dtype=float)
                assert field.shape == (dims["phi"], dims["zee"], dims["rad"])
                assert np.all(np.isfinite(field))
                assert float(np.max(np.abs(field))) > 0.0


@pytest.mark.parametrize("case", ROUNDTRIP_CASES, ids=lambda case: case.name)
def test_representative_bundled_wouts_roundtrip_without_losing_physics_metadata(tmp_path: Path, case: RoundtripCase) -> None:
    """Read/write/read parity should preserve the shipped validation equilibria exactly."""
    pytest.importorskip("netCDF4")

    src = _repo_root() / case.wout_rel
    if not src.exists():
        pytest.skip(f"{case.wout_rel} is a fetched validation WOUT; run tools/fetch_assets.py")
    original = read_wout(src)
    assert bool(original.lasym) is case.expect_lasym

    out_path = tmp_path / Path(case.wout_rel).name
    write_wout(out_path, original, overwrite=True)
    roundtrip = read_wout(out_path)

    for name in (
        "ns",
        "mpol",
        "ntor",
        "nfp",
        "lasym",
        "signgs",
        "mnmax",
        "mnmax_nyq",
        "mpol_nyq",
        "ntor_nyq",
        "pcurr_type",
        "piota_type",
    ):
        assert getattr(roundtrip, name) == getattr(original, name), name

    for name in (
        "xm",
        "xn",
        "xm_nyq",
        "xn_nyq",
        "rmnc",
        "rmns",
        "zmnc",
        "zmns",
        "lmnc",
        "lmns",
        "gmnc",
        "gmns",
        "bmnc",
        "bmns",
        "bsupumnc",
        "bsupumns",
        "bsupvmnc",
        "bsupvmns",
        "bsubumnc",
        "bsubumns",
        "bsubvmnc",
        "bsubvmns",
        "bsubsmns",
        "bsubsmnc",
        "phipf",
        "phips",
        "chipf",
        "iotaf",
        "iotas",
        "phi",
        "vp",
        "pres",
        "presf",
        "fsqt",
        "equif",
        "buco",
        "bvco",
        "jcuru",
        "jcurv",
        "DMerc",
        "Dshear",
        "Dwell",
        "Dcurr",
        "Dgeod",
        "jdotb",
        "bdotb",
        "bdotgradv",
        "ac",
        "ac_aux_s",
        "ac_aux_f",
    ):
        _assert_allclose_field(name, getattr(original, name), getattr(roundtrip, name))

    for name in (
        "wb",
        "volume_p",
        "gamma",
        "wp",
        "fsqr",
        "fsqz",
        "fsql",
        "Aminor_p",
        "Rmajor_p",
        "aspect",
        "betatotal",
        "betapol",
        "betator",
        "betaxis",
        "ctor",
    ):
        assert float(getattr(roundtrip, name)) == float(getattr(original, name)), name


@pytest.mark.parametrize("case", ROUNDTRIP_CASES, ids=lambda case: case.name)
def test_wout_residual_trace_shows_converged_final_state(case: RoundtripCase) -> None:
    """Bundled WOUT traces must retain enough information to prove convergence."""
    pytest.importorskip("netCDF4")

    src = _repo_root() / case.wout_rel
    if not src.exists():
        pytest.skip(f"{case.wout_rel} is a fetched validation WOUT; run tools/fetch_assets.py")
    wout = read_wout(src)
    fsqt = np.asarray(wout.fsqt, dtype=float)
    active = fsqt[fsqt > 0.0]
    assert active.size >= 2
    assert np.all(np.isfinite(active))
    assert np.all(active >= 0.0)

    residual_components = np.asarray([wout.fsqr, wout.fsqz, wout.fsql], dtype=float)
    assert np.all(np.isfinite(residual_components))
    assert np.all(residual_components >= 0.0)
    final_total = float(np.sum(residual_components))
    assert final_total < float(active[0]) * 1.0e-3
    assert final_total < 1.0e-9
