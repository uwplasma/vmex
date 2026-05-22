from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from vmec_jax.namelist import read_indata
from vmec_jax.wout import read_wout
from vmec_jax.wout_schema import assert_main_modes_match_wout


@dataclass(frozen=True)
class WoutInventoryCase:
    wout_rel: str
    input_rel: str
    optional_reason: str = ""


REQUIRED_WOUT_FIXTURES = (
    WoutInventoryCase("examples/data/wout_DSHAPE.nc", "examples/data/input.DSHAPE"),
    WoutInventoryCase("examples/data/wout_LandremanPaul2021_QA_lowres.nc", "examples/data/input.LandremanPaul2021_QA_lowres"),
    WoutInventoryCase("examples/data/wout_QI_stel_seed_3127.nc", "examples/data/input.QI_stel_seed_3127"),
    WoutInventoryCase("examples/data/wout_basic_non_stellsym_simsopt.nc", "examples/data/input.basic_non_stellsym_simsopt"),
    WoutInventoryCase("examples/data/wout_circular_tokamak.nc", "examples/data/input.circular_tokamak"),
    WoutInventoryCase("examples/data/wout_cth_like_fixed_bdy.nc", "examples/data/input.cth_like_fixed_bdy"),
    WoutInventoryCase("examples/data/wout_li383_low_res.nc", "examples/data/input.li383_low_res"),
    WoutInventoryCase("examples/data/wout_nfp3_QI_fixed_resolution_final.nc", "examples/data/input.nfp3_QI_fixed_resolution_final"),
    WoutInventoryCase("examples/data/wout_nfp4_QH_warm_start.nc", "examples/data/input.nfp4_QH_warm_start"),
    WoutInventoryCase("examples/data/wout_purely_toroidal_field.nc", "examples/data/input.purely_toroidal_field"),
    WoutInventoryCase("examples/data/wout_shaped_tokamak_pressure.nc", "examples/data/input.shaped_tokamak_pressure"),
    WoutInventoryCase("examples_single_grid/data/wout_ITERModel_reference.nc", "examples_single_grid/data/input.ITERModel"),
    WoutInventoryCase(
        "examples_single_grid/data/wout_LandremanPaul2021_QA_lowres_reference.nc",
        "examples_single_grid/data/input.LandremanPaul2021_QA_lowres",
    ),
    WoutInventoryCase(
        "examples_single_grid/data/wout_LandremanPaul2021_QA_reactorScale_lowres_reference.nc",
        "examples_single_grid/data/input.LandremanPaul2021_QA_reactorScale_lowres",
    ),
    WoutInventoryCase(
        "examples_single_grid/data/wout_LandremanPaul2021_QH_reactorScale_lowres_reference.nc",
        "examples_single_grid/data/input.LandremanPaul2021_QH_reactorScale_lowres",
    ),
    WoutInventoryCase(
        "examples_single_grid/data/wout_LandremanSengupta2019_section5.4_B2_A80_reference.nc",
        "examples_single_grid/data/input.LandremanSengupta2019_section5.4_B2_A80",
    ),
    WoutInventoryCase(
        "examples_single_grid/data/wout_LandremanSenguptaPlunk_section5p3_low_res_reference.nc",
        "examples_single_grid/data/input.LandremanSenguptaPlunk_section5p3_low_res",
    ),
    WoutInventoryCase(
        "examples_single_grid/data/wout_basic_non_stellsym_pressure_reference.nc",
        "examples_single_grid/data/input.basic_non_stellsym_pressure",
    ),
    WoutInventoryCase(
        "examples_single_grid/data/wout_circular_tokamak_aspect_100_reference.nc",
        "examples_single_grid/data/input.circular_tokamak_aspect_100",
    ),
    WoutInventoryCase("examples_single_grid/data/wout_circular_tokamak_reference.nc", "examples_single_grid/data/input.circular_tokamak"),
    WoutInventoryCase("examples_single_grid/data/wout_cth_like_free_bdy.nc", "examples_single_grid/data/input.cth_like_free_bdy"),
    WoutInventoryCase(
        "examples_single_grid/data/wout_purely_toroidal_field_reference.nc",
        "examples_single_grid/data/input.purely_toroidal_field",
    ),
    WoutInventoryCase(
        "examples_single_grid/data/wout_shaped_tokamak_pressure_reference.nc",
        "examples_single_grid/data/input.shaped_tokamak_pressure",
    ),
    WoutInventoryCase("examples_single_grid/data/wout_solovev_reference.nc", "examples_single_grid/data/input.solovev"),
    WoutInventoryCase(
        "examples_single_grid/data/wout_up_down_asymmetric_tokamak_reference.nc",
        "examples_single_grid/data/input.up_down_asymmetric_tokamak",
    ),
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_every_bundled_reference_wout_is_declared() -> None:
    repo = _repo_root()
    discovered = {
        path.relative_to(repo).as_posix()
        for folder in (repo / "examples/data", repo / "examples_single_grid/data")
        for path in folder.glob("wout*.nc")
    }
    declared = {case.wout_rel for case in REQUIRED_WOUT_FIXTURES}

    assert discovered == declared


@pytest.mark.parametrize("case", REQUIRED_WOUT_FIXTURES, ids=lambda case: Path(case.wout_rel).stem)
def test_bundled_reference_wout_inventory_matches_input(case: WoutInventoryCase) -> None:
    pytest.importorskip("netCDF4")

    repo = _repo_root()
    input_path = repo / case.input_rel
    wout_path = repo / case.wout_rel
    assert input_path.exists(), case.input_rel
    assert wout_path.exists(), case.wout_rel

    indata = read_indata(input_path)
    wout = read_wout(wout_path)
    assert_main_modes_match_wout(wout=wout)

    assert int(wout.nfp) == int(indata.get_int("NFP", int(wout.nfp)))
    assert int(wout.mpol) == int(indata.get_int("MPOL", int(wout.mpol)))
    ntor = int(indata.get_int("NTOR", 0))
    if ntor > 0:
        assert int(wout.ntor) == ntor
    assert bool(wout.lasym) is bool(indata.get_bool("LASYM", False))
    np.testing.assert_allclose(float(wout.phi[-1]), indata.get_float("PHIEDGE", float(wout.phi[-1])), rtol=1.0e-12, atol=1.0e-12)
    assert np.isfinite([wout.fsqr, wout.fsqz, wout.fsql]).all()
