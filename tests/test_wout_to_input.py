"""Reconstruct an INDATA deck from saved VMEC geometry and profiles."""

from dataclasses import replace

import numpy as np
import pytest

from vmex.core.cli import main
from vmex.core import profiles
from vmex.core.input import VmecInput
from vmex.core.wout import read_wout, write_wout
from tests.conftest import resolve_golden_dir


def _golden(case):
    root = resolve_golden_dir()
    if root is None:
        pytest.skip("VMEC2000 WOUT fixtures unavailable")
    return root / case / f"wout_{case}.nc"


@pytest.mark.parametrize("case,scale", [("solovev", 1.0),
                                       ("cth_like_fixed_bdy", 432.29080924603676)])
def test_reconstructed_deck_roundtrips_and_preserves_equilibrium(case, scale, tmp_path):
    wout = read_wout(_golden(case))
    deck = VmecInput.from_wout(wout)
    reread = VmecInput.from_file(deck.to_indata(tmp_path / "input.case"))
    assert reread == deck
    assert deck.ns_array[-1] == wout.ns
    assert deck.ftol_array[-1] == pytest.approx(wout.ftolv)
    assert deck.niter_array[-1] >= wout.niter
    assert deck.phiedge == pytest.approx(wout.phi[-1], rel=1e-14)
    assert deck.pres_scale == pytest.approx(scale, rel=1e-12)
    assert deck.ncurr == (1 if case.startswith("cth_like") else 0)
    np.testing.assert_allclose(deck.rbc[deck.ntor], wout.rmnc[-1], rtol=0, atol=0)
    np.testing.assert_allclose(deck.zbs[deck.ntor], wout.zmns[-1], rtol=0, atol=0)


def test_lasym_free_boundary_modes_and_coils(tmp_path):
    base = read_wout(_golden("solovev"))
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
    source = _golden("solovev")
    assert main([str(source), "--to-input", "--outdir", str(tmp_path)]) == 0
    deck = VmecInput.from_file(tmp_path / "input.solovev")
    assert deck.phiedge == pytest.approx(read_wout(source).phi[-1])
    assert main([str(source), "--to-input", "--outdir", str(tmp_path)]) == 0
    assert (tmp_path / "input.solovev_from_wout").exists()
    assert main([str(source), "--to-input", "--outdir", str(tmp_path)]) != 0


def test_reject_unrecoverable_adiabatic_pressure():
    wout = read_wout(_golden("solovev"))
    with pytest.raises(ValueError, match="GAMMA"):
        VmecInput.from_wout(replace(wout, gamma=5 / 3))


@pytest.mark.parametrize("changes,reason", [
    ({"ns": 2}, "invalid resolution"),
    ({"lfreeb": True, "mgrid_file": "NONE"}, "no MGRID_FILE"),
    ({"lfreeb": True, "mgrid_file": ""}, "no MGRID_FILE"),
    ({"nextcur": -1}, "invalid external current count"),
    ({"xn": np.array([0.5, 0, 0, 0, 0, 0])}, "invalid Fourier mode"),
    ({"xn": np.array([6, 0, 0, 0, 0, 0])}, "invalid Fourier mode"),
])
def test_reject_invalid_wout_metadata(changes, reason):
    wout = read_wout(_golden("solovev"))
    with pytest.raises(ValueError, match=reason):
        VmecInput.from_wout(replace(wout, **changes))


def test_resolves_mgrid_beside_wout(tmp_path, monkeypatch):
    source = tmp_path / "wout_case.nc"
    field = tmp_path / "field.nc"
    field.touch()
    wout = replace(read_wout(_golden("solovev")),
                   lfreeb=True, mgrid_file=field.name)
    monkeypatch.setattr("vmex.core.wout.read_wout", lambda path: wout)
    assert VmecInput.from_wout(source).mgrid_file == str(field)


def test_fallback_profiles_use_solved_values():
    base = read_wout(_golden("solovev"))
    s = np.linspace(0, 1, base.ns)
    pressure = np.asarray(base.presf) + 5.0 * s**2
    wout = replace(base, presf=pressure,
                   pres=np.r_[0.0, (pressure[:-1] + pressure[1:]) / 2],
                   iotaf=np.asarray(base.iotaf) + 0.1,
                   iotas=np.asarray(base.iotas) + 0.1)
    deck = VmecInput.from_wout(wout)
    assert deck.pmass_type == "cubic_spline"
    assert deck.piota_type == "cubic_spline"
    assert deck.ncurr == 0
    half = (np.arange(1, base.ns) - 0.5) / (base.ns - 1)
    np.testing.assert_allclose(profiles.pressure(
        deck.pmass_type, deck.am, deck.am_aux_s, deck.am_aux_f, half),
        wout.pres[1:], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(profiles.iota(
        deck.piota_type, deck.ai, deck.ai_aux_s, deck.ai_aux_f, half),
        wout.iotas[1:], rtol=1e-10, atol=1e-10)


def test_sharp_pressure_fallback_preserves_half_mesh():
    base = read_wout(_golden("DSHAPE"))
    full = np.linspace(0, 1, base.ns)
    half = (np.arange(1, base.ns) - 0.5) / (base.ns - 1)
    def pedestal(s):
        return 1e4 * (1 - np.tanh((s - 0.8) / 0.003)) / 2
    wout = replace(base, pmass_type="unavailable", presf=pedestal(full),
                   pres=np.r_[0, pedestal(half)])
    deck = VmecInput.from_wout(wout)
    solved = profiles.pressure(deck.pmass_type, deck.am, deck.am_aux_s,
                               deck.am_aux_f, half)
    np.testing.assert_allclose(solved, wout.pres[1:], rtol=0, atol=1e-2)
    assert len(deck.am_aux_s) <= 101


@pytest.mark.parametrize("kind", ["", "unavailable"])
def test_missing_profile_metadata_uses_solved_profiles(kind, tmp_path):
    base = read_wout(_golden("solovev"))
    wout = replace(base, pmass_type=kind, piota_type=kind, pcurr_type=kind,
                   ai=np.zeros_like(base.ai), ac=np.ones_like(base.ac))
    deck = VmecInput.from_wout(wout)
    assert deck.pmass_type == ("power_series" if not kind else "cubic_spline")
    assert deck.piota_type == "cubic_spline"
    assert deck.pcurr_type == "power_series"
    assert deck.ncurr == 0
    assert VmecInput.from_file(deck.to_indata(tmp_path / "input.legacy")) == deck


def test_cli_rejects_wrong_operation_and_unrecoverable_wout(tmp_path):
    source = _golden("solovev")
    with pytest.raises(SystemExit):
        main([str(source), "--to-input", "--plot"])
    invalid = tmp_path / "wout_adiabatic.nc"
    write_wout(invalid, replace(read_wout(source), gamma=5 / 3))
    assert main([str(invalid), "--to-input", "--outdir", str(tmp_path)]) != 0
    assert not (tmp_path / "input.adiabatic").exists()


@pytest.mark.parametrize("case", ["solovev", "up_down_asymmetric_tokamak",
                                  "cth_like_fixed_bdy",
                                  "LandremanPaul2021_QA_lowres",
                                  "cth_like_free_bdy_lasym_small",
                                  "circular_tokamak", "DSHAPE", "li383_low_res",
                                  "nfp4_QH_warm_start"])
def test_vmec2000_wout_geometry_and_input_roundtrip(case, tmp_path):
    source = _golden(case)
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
