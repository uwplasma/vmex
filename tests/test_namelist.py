from __future__ import annotations

from pathlib import Path

from vmec_jax.namelist import InData, read_indata, write_indata


def test_indata_basic(tmp_path: Path):
    txt = """
&INDATA
  NFP = 1
  MPOL = 5
  NTOR = 0
  LASYM = F
  RBC(0,0) = 6.0
  RBC(0,1) = 2.0
  ZBS(0,1) = 2.0
/
"""
    p = tmp_path / "input.test"
    p.write_text(txt)

    indata = read_indata(p)
    assert indata.get_int("NFP") == 1
    assert indata.get_int("MPOL") == 5
    assert indata.get_bool("LASYM") is False

    assert indata.indexed["RBC"][(0,0)] == 6.0
    assert indata.indexed["RBC"][(0,1)] == 2.0
    assert indata.indexed["ZBS"][(0,1)] == 2.0


def test_fortran_repeat_syntax_is_expanded(tmp_path: Path):
    # VMEC inputs commonly use Fortran repeat counts like `11*0.0`.
    txt = """
&INDATA
  AI = 11*0.0, 0.8
/
"""
    p = tmp_path / "input.repeat"
    p.write_text(txt)

    indata = read_indata(p)
    ai = indata.get("AI")
    assert isinstance(ai, list)
    assert len(ai) == 12
    assert all(not isinstance(v, str) for v in ai)
    assert ai[:11] == [0.0] * 11
    assert ai[11] == 0.8


def test_write_indata_roundtrips_scalars_lists_and_indexed_values(tmp_path: Path):
    source = InData(
        scalars={
            "NFP": 2,
            "LASYM": False,
            "MGRID_FILE": "mgrid_cth_like_lasym_small.nc",
            "FTOL_ARRAY": [1e-10, 1e-13],
        },
        indexed={
            "RBC": {(0, 0): 1.0, (1, 1): 1e-5},
            "ZBS": {(0, 1): 0.2, (1, 1): 1e-5},
        },
        source_path=None,
    )
    path = tmp_path / "input.roundtrip"

    write_indata(path, source)
    got = read_indata(path)

    assert got.get_int("NFP") == 2
    assert got.get_bool("LASYM") is False
    assert got.get("MGRID_FILE") == "mgrid_cth_like_lasym_small.nc"
    assert got.get("FTOL_ARRAY") == [1e-10, 1e-13]
    assert got.indexed["RBC"][(0, 0)] == 1.0
    assert got.indexed["RBC"][(1, 1)] == 1e-5
    assert got.indexed["ZBS"][(0, 1)] == 0.2
    assert got.indexed["ZBS"][(1, 1)] == 1e-5
