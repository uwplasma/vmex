from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vmec_jax.config import config_from_indata
from vmec_jax.free_boundary import MGridData, MGridMetadata, load_mgrid
from vmec_jax.namelist import read_indata
from vmec_jax.static import build_static


def test_free_boundary_config_vmec2000_defaults(tmp_path: Path):
    txt = """
&INDATA
  NFP = 3
  MPOL = 5
  NTOR = 2
  NS = 11
  LASYM = F
  LFREEB = T
  MGRID_FILE = 'NONE'
  NVACSKIP = 0
/
"""
    p = tmp_path / "input.fb_defaults"
    p.write_text(txt)
    indata = read_indata(p)
    cfg = config_from_indata(indata)

    # VMEC read_indata/readin behavior.
    assert cfg.lfreeb is False
    assert cfg.mgrid_file.upper() == "NONE"
    assert cfg.nvacskip == 3
    assert cfg.extcur == ()

    static = build_static(cfg)
    assert static.free_boundary_state0 is None


def test_free_boundary_config_extcur_indexed(tmp_path: Path):
    txt = """
&INDATA
  NFP = 2
  MPOL = 5
  NTOR = 1
  NS = 9
  LFREEB = T
  MGRID_FILE = 'mgrid_test.nc'
  NVACSKIP = -5
  EXTCUR(1) = 12.0
  EXTCUR(3) = -4.5
/
"""
    p = tmp_path / "input.fb_extcur"
    p.write_text(txt)
    indata = read_indata(p)
    cfg = config_from_indata(indata)

    assert cfg.lfreeb is True
    assert cfg.mgrid_file == "mgrid_test.nc"
    # NVACSKIP<=0 falls back to NFP.
    assert cfg.nvacskip == 2
    assert cfg.extcur == (12.0, 0.0, -4.5)

    static = build_static(cfg)
    assert static.free_boundary_state0 is not None
    assert static.free_boundary_state0.nvacskip == 2


def test_mgrid_loader_skeleton(tmp_path: Path):
    netCDF4 = pytest.importorskip("netCDF4", reason="netCDF4 required for mgrid loader test")

    p = tmp_path / "mgrid_test.nc"
    with netCDF4.Dataset(str(p), mode="w", format="NETCDF3_CLASSIC") as ds:
        ds.createDimension("stringsize", 8)
        ds.createDimension("external_coil_groups", 2)
        ds.createDimension("dim_00001", 1)
        ds.createDimension("external_coils", 2)
        ds.createDimension("rad", 3)
        ds.createDimension("zee", 4)
        ds.createDimension("phi", 2)

        for name, value in (
            ("ir", 3),
            ("jz", 4),
            ("kp", 2),
            ("nfp", 5),
            ("nextcur", 2),
        ):
            v = ds.createVariable(name, "i4", ())
            v.assignValue(value)
        for name, value in (
            ("rmin", 1.0),
            ("rmax", 2.0),
            ("zmin", -0.3),
            ("zmax", 0.7),
        ):
            v = ds.createVariable(name, "f8", ())
            v.assignValue(value)

        mgrid_mode = ds.createVariable("mgrid_mode", "S1", ("dim_00001",))
        mgrid_mode[:] = np.asarray(list("S"), dtype="S1")
        coil_group = ds.createVariable("coil_group", "S1", ("external_coil_groups", "stringsize"))
        names = np.full((2, 8), b" ", dtype="S1")
        names[0, :5] = np.asarray(list("coilA"), dtype="S1")
        names[1, :5] = np.asarray(list("coilB"), dtype="S1")
        coil_group[:] = names
        raw = ds.createVariable("raw_coil_cur", "f8", ("external_coils",))
        raw[:] = np.asarray([10.0, -7.0], dtype=np.float64)

        for i in (1, 2):
            for prefix in ("br", "bp", "bz"):
                v = ds.createVariable(f"{prefix}_{i:03d}", "f8", ("phi", "zee", "rad"))
                v[:] = (100 * i) + np.arange(2 * 4 * 3, dtype=np.float64).reshape(2, 4, 3)

    meta = load_mgrid(p, load_fields=False)
    assert isinstance(meta, MGridMetadata)
    assert meta.ir == 3
    assert meta.jz == 4
    assert meta.kp == 2
    assert meta.nfp == 5
    assert meta.nextcur == 2
    assert meta.coil_groups == ("coilA", "coilB")
    assert meta.raw_coil_cur == (10.0, -7.0)

    data = load_mgrid(p, load_fields=True)
    assert isinstance(data, MGridData)
    assert data.br.shape == (2, 2, 4, 3)
    assert data.bp.shape == (2, 2, 4, 3)
    assert data.bz.shape == (2, 2, 4, 3)
    # Check one value round-trips correctly.
    assert float(data.br[1, 1, 3, 2]) == pytest.approx(200 + (1 * 12 + 3 * 3 + 2))
