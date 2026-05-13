from __future__ import annotations

from pathlib import Path

from vmec_jax.config import config_from_indata, load_config
from vmec_jax.namelist import InData


def test_config_from_indata_scalar_extcur_and_quoted_mgrid() -> None:
    indata = InData(
        scalars={
            "MPOL": 2,
            "NTOR": 0,
            "NS_ARRAY": 5,
            "NFP": 3,
            "LFREEB": True,
            "MGRID_FILE": "'mgrid_rel.nc'",
            "EXTCUR": 3.5,
            "NVACSKIP": 0,
        },
        indexed={},
    )

    cfg = config_from_indata(indata)

    assert cfg.lfreeb is True
    assert cfg.mgrid_file == "mgrid_rel.nc"
    assert cfg.extcur == (3.5,)
    assert cfg.nvacskip == 3


def test_config_from_indata_indexed_extcur_ignores_invalid_indices() -> None:
    indata = InData(
        scalars={
            "MPOL": 2,
            "NTOR": 0,
            "NS_ARRAY": 5,
            "NFP": 1,
            "LFREEB": True,
            "MGRID_FILE": "mgrid.nc",
        },
        indexed={
            "EXTCUR": {
                (0,): 99.0,
                (2,): 4.0,
                (2, 1): 88.0,
            }
        },
    )

    cfg = config_from_indata(indata)

    assert cfg.extcur == (0.0, 4.0)


def test_load_config_resolves_relative_mgrid_paths(tmp_path: Path) -> None:
    input_path = tmp_path / "input.freeb"
    input_path.write_text(
        """
&INDATA
  MPOL = 2
  NTOR = 0
  NS_ARRAY = 5
  NFP = 1
  LFREEB = T
  MGRID_FILE = 'subdir/mgrid.nc'
/
"""
    )

    cfg, _indata = load_config(input_path)

    assert cfg.lfreeb is True
    assert cfg.mgrid_file == str((tmp_path / "subdir" / "mgrid.nc").resolve())
