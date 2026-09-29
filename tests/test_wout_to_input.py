"""Reconstruct an INDATA deck from saved VMEC geometry and profiles."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from vmex.core.cli import main
from vmex.core.input import VmecInput
from vmex.core.wout import read_wout
from tests.conftest import resolve_golden_dir


DATA = Path(__file__).resolve().parents[1] / "examples" / "data"


@pytest.mark.parametrize("name", ["wout_shaped_tokamak_pressure.nc",
                                   "wout_shaped_tokamak_pressure_polished.nc"])
def test_reconstructed_deck_roundtrips_and_preserves_equilibrium(name, tmp_path):
    wout = read_wout(DATA / name)
    deck = VmecInput.from_wout(wout)
    reread = VmecInput.from_file(deck.to_indata(tmp_path / "input.case"))
    assert reread == deck
    assert deck.ns_array[-1] == wout.ns
    assert deck.ftol_array[-1] == pytest.approx(wout.ftolv)
    assert deck.niter_array[-1] >= wout.niter
    assert deck.phiedge == pytest.approx(wout.phi[-1], rel=1e-14)
    assert deck.pres_scale == pytest.approx(1e4, rel=1e-12)
    assert deck.ncurr == 0
    np.testing.assert_allclose(deck.rbc[deck.ntor], wout.rmnc[-1], rtol=0, atol=0)
    np.testing.assert_allclose(deck.zbs[deck.ntor], wout.zmns[-1], rtol=0, atol=0)


def test_lasym_free_boundary_modes_and_coils(tmp_path):
    base = read_wout(DATA / "wout_shaped_tokamak_pressure.nc")
    wout = replace(base, lasym=True, lfreeb=True, mgrid_file="field.nc",
                   nextcur=2, extcur=np.array([1.2, -3.4]),
                   rmns=np.full_like(base.rmnc, 0.012),
                   zmnc=np.full_like(base.zmns, -0.034),
                   raxis_cs=np.array([0.0]), zaxis_cc=np.array([0.3]))
    deck = VmecInput.from_wout(wout)
    assert VmecInput.from_file(deck.to_indata(tmp_path / "input.asym")) == deck
    assert deck.lasym and deck.lfreeb
    assert deck.mgrid_file == "field.nc"
    np.testing.assert_array_equal(deck.extcur, [1.2, -3.4])
    np.testing.assert_allclose(deck.rbs, 0.012)
    np.testing.assert_allclose(deck.zbc, -0.034)


def test_cli_writes_input_without_solving(tmp_path):
    source = DATA / "wout_shaped_tokamak_pressure.nc"
    assert main([str(source), "--to-input", "--outdir", str(tmp_path)]) == 0
    deck = VmecInput.from_file(tmp_path / "input.shaped_tokamak_pressure")
    assert deck.phiedge == pytest.approx(67.86)
    assert main([str(source), "--to-input", "--outdir", str(tmp_path)]) == 0
    assert (tmp_path / "input.shaped_tokamak_pressure_from_wout").exists()
    assert main([str(source), "--to-input", "--outdir", str(tmp_path)]) != 0


def test_reject_unrecoverable_adiabatic_pressure():
    wout = read_wout(DATA / "wout_shaped_tokamak_pressure.nc")
    with pytest.raises(ValueError, match="GAMMA"):
        VmecInput.from_wout(replace(wout, gamma=5 / 3))


@pytest.mark.parametrize("case", ["solovev", "up_down_asymmetric_tokamak",
                                  "cth_like_fixed_bdy",
                                  "LandremanPaul2021_QA_lowres",
                                  "cth_like_free_bdy_lasym_small"])
def test_vmec2000_wout_geometry_and_input_roundtrip(case, tmp_path):
    golden = resolve_golden_dir()
    if golden is None:
        pytest.skip("VMEC2000 WOUT fixtures unavailable")
    source = golden / case / f"wout_{case}.nc"
    wout = read_wout(source)
    deck = VmecInput.from_wout(source)
    assert VmecInput.from_file(deck.to_indata(tmp_path / f"input.{case}")) == deck
    assert deck.lasym == wout.lasym
    assert deck.lfreeb == wout.lfreeb
    assert deck.phiedge == pytest.approx(wout.phi[-1], rel=1e-14)
    for col, (m, xn) in enumerate(zip(wout.xm, wout.xn)):
        row = int(round(xn / wout.nfp)) + deck.ntor
        m = int(m)
        assert deck.rbc[row, m] == wout.rmnc[-1, col]
        assert deck.zbs[row, m] == wout.zmns[-1, col]
        if wout.lasym:
            assert deck.rbs[row, m] == wout.rmns[-1, col]
            assert deck.zbc[row, m] == wout.zmnc[-1, col]
    if case.startswith("cth_like"):
        assert deck.ncurr == 1
    if wout.lfreeb:
        np.testing.assert_array_equal(deck.extcur, wout.extcur[:wout.nextcur])
