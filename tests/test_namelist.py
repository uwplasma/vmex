from __future__ import annotations

from pathlib import Path

import pytest

from vmec_jax.namelist import InData, minimal_fixed_boundary_indata, read_indata, write_indata


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


def test_indata_parser_handles_comments_slices_repeats_and_raw_tokens(tmp_path: Path):
    txt = """
&INDATA
  MGRID_FILE = 'grid!literal.nc' ! comment outside the quoted string
  RAXIS_CC(:) = 1.0, 2.5D+0
  ZERO_REPEAT = 0*7.0
  EMPTY_REPEAT = 2*
  RAW_TOKEN = not_a_number
  FLAG_LIST = .TRUE., .FALSE.
  EMPTY_VALUE =
  RBC(0,-1) = 3.0
/
"""
    path = tmp_path / "input.parser_edges"
    path.write_text(txt)

    indata = read_indata(path)

    assert indata.get("MGRID_FILE") == "grid!literal.nc"
    assert indata.get("RAXIS_CC") == [1.0, 2.5]
    assert indata.get("ZERO_REPEAT") == "0*7.0"
    assert indata.get("EMPTY_REPEAT") == "2*"
    assert indata.get("RAW_TOKEN") == "not_a_number"
    assert indata.get_bool("FLAG_LIST") is True
    assert "EMPTY_VALUE" not in indata.scalars
    assert indata.indexed["RBC"][(0, -1)] == 3.0


def test_indata_getters_and_parser_error_paths(tmp_path: Path):
    indata = InData(
        scalars={
            "EMPTY_LIST": [],
            "BAD_INT": "not-int",
            "BAD_FLOAT": "not-float",
        },
        indexed={},
    )

    assert indata.get_int("EMPTY_LIST", default=7) == 7
    assert indata.get_float("EMPTY_LIST", default=1.25) == 1.25
    assert indata.get_bool("EMPTY_LIST") is False
    assert indata.get_int("BAD_INT", default=9) == 9
    assert indata.get_float("BAD_FLOAT", default=2.5) == 2.5

    no_block = tmp_path / "input.no_block"
    no_block.write_text("&OTHER\n  NFP = 1\n/\n")
    with pytest.raises(ValueError, match="No &INDATA found"):
        read_indata(no_block)

    no_end = tmp_path / "input.no_end"
    no_end.write_text("&INDATA\n  NFP = 1\n")
    with pytest.raises(ValueError, match="No terminating '/'"):
        read_indata(no_end)

    empty = tmp_path / "input.empty"
    empty.write_text("&INDATA\n  ! no assignments\n/\n")
    assert read_indata(empty).scalars == {}

    bad_indexed = tmp_path / "input.bad_indexed"
    bad_indexed.write_text("&INDATA\n  RBC(0,0) = 1.0, 2.0\n/\n")
    with pytest.raises(ValueError, match="Indexed assignment RBC"):
        read_indata(bad_indexed)


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


def test_minimal_fixed_boundary_seed_roundtrips_three_boundary_coefficients(tmp_path: Path):
    source = minimal_fixed_boundary_indata(nfp=3, r0=1.1, rbc01=0.17, zbs01=0.19)
    path = tmp_path / "input.minimal_nfp3"

    write_indata(path, source)
    got = read_indata(path)

    assert got.get_int("NFP") == 3
    assert got.get_int("MPOL") == 5
    assert got.get_int("NTOR") == 5
    assert got.get_bool("LASYM") is False
    assert got.indexed["RBC"] == {(0, 0): 1.1, (0, 1): 0.17}
    assert got.indexed["ZBS"] == {(0, 1): 0.19}
    assert "RBS" not in got.indexed
    assert "ZBC" not in got.indexed


@pytest.mark.parametrize("nfp", [1, 2, 3, 4])
def test_bundled_minimal_seed_inputs_match_factory_contract(nfp: int):
    """Guard the common far-from-target optimization seeds used in docs/examples."""

    root = Path(__file__).resolve().parents[1]
    input_path = root / "examples" / "data" / f"input.minimal_seed_nfp{nfp}"

    got = read_indata(input_path)
    expected = minimal_fixed_boundary_indata(nfp=nfp)

    assert got.scalars == expected.scalars
    assert got.indexed == expected.indexed
    assert got.get_bool("LFREEB") is False
    assert got.get_bool("LASYM") is False
    assert got.get_int("NFP") == nfp
    assert got.get_int("MPOL") >= 5
    assert got.get_int("NTOR") >= 5
    assert set(got.indexed) == {"RBC", "ZBS"}
    assert got.indexed["RBC"] == {(0, 0): 1.0, (0, 1): 0.2}
    assert got.indexed["ZBS"] == {(0, 1): 0.2}


def test_bundled_nfp2_target_helicity_seed_has_documented_high_mode_perturbations():
    """Guard the reviewed QI panel seed separately from the bare minimal seed."""

    root = Path(__file__).resolve().parents[1]
    input_path = root / "examples" / "data" / "input.minimal_seed_nfp2_target_helicity"
    got = read_indata(input_path)

    assert got.get_int("NFP") == 2
    assert got.get_int("MPOL") == 6
    assert got.get_int("NTOR") == 6
    assert got.indexed["RBC"][(0, 0)] == 1.0
    assert got.indexed["RBC"][(0, 1)] == 0.2
    assert got.indexed["ZBS"][(0, 1)] == 0.2
    assert set(got.indexed) == {"RBC", "ZBS"}
    for family in ("RBC", "ZBS"):
        modes = set(got.indexed[family])
        assert all(max(abs(m), abs(n)) <= 3 for m, n in modes)
        assert any(max(abs(m), abs(n)) == 3 for m, n in modes)
        assert len(modes) == (25 if family == "RBC" else 24)
