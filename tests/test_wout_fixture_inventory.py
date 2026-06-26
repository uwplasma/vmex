from __future__ import annotations

import subprocess
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from vmec_jax.namelist import read_indata
from vmec_jax.wout import read_wout
from vmec_jax.io.wout_files.schema import assert_main_modes_match_wout


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
    WoutInventoryCase(
        "examples/data/single_grid/wout_basic_non_stellsym_pressure_reference.nc",
        "examples/data/single_grid/input.basic_non_stellsym_pressure",
    ),
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _tracked_wout_fixtures(repo: Path) -> set[str]:
    output = subprocess.check_output(
        ["git", "ls-files", "examples/data/wout*.nc", "examples/data/single_grid/wout*.nc"],
        cwd=repo,
        text=True,
    )
    return {line.strip() for line in output.splitlines() if line.strip()}


def _load_fetch_assets_module(repo: Path):
    spec = importlib.util.spec_from_file_location("fetch_assets", repo / "tools" / "fetch_assets.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_wout_fixtures_are_not_tracked_in_git() -> None:
    repo = _repo_root()
    discovered = _tracked_wout_fixtures(repo)

    assert discovered == set()


def test_wout_fixture_asset_manifest_covers_required_cases() -> None:
    repo = _repo_root()
    module = _load_fetch_assets_module(repo)
    fixture_bundle = module.BUNDLES_BY_NAME["wout-fixtures"]
    declared = {case.wout_rel for case in REQUIRED_WOUT_FIXTURES}
    common_paths = set(fixture_bundle.common_paths)

    assert "examples/data/wout_*.nc" in common_paths
    assert "examples/data/single_grid/wout_*.nc" in common_paths
    assert declared


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
