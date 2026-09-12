"""DESC interoperability contracts: analytic geometry/profiles and reproducible solves."""

from dataclasses import replace
import sys

import numpy as np
import pytest

from vmex.core import cli
from vmex.core.desc import _compact_boundary, is_desc_file, write_desc_input
from vmex.core.errors import VmecInputError
from vmex.core.input import VmecInput


@pytest.mark.parametrize("text,expected", [
    ("# DESC\nsym = 1\nM = 4", True), ("Psi = 1", True),
    ("&INDATA\nMPOL=4\n/", False), ('{"mpol": 4}', False),
    ("# sym = 1\n! M = 4\ninvalid", False),
])
def test_detection(tmp_path, text, expected):
    path = tmp_path / "deck"
    path.write_text(text)
    assert is_desc_file(path) is expected
    assert is_desc_file(tmp_path / "output.HDF5")


@pytest.mark.parametrize("tol", [-1, 0.02, float("nan"), float("inf")])
def test_invalid_tolerance(tmp_path, tol):
    with pytest.raises(VmecInputError, match="tolerance"):
        write_desc_input(tmp_path / "eq.h5", tolerance=tol)


def test_missing_optional_dependency(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "desc.equilibrium", None)
    with pytest.raises(VmecInputError, match="desc-opt"):
        write_desc_input(tmp_path / "eq.h5")


@pytest.mark.parametrize("asymmetric", [False, True])
def test_truncation_is_minimal_and_bounds_off_grid_geometry(asymmetric):
    rng = np.random.default_rng(392)
    inp = VmecInput(mpol=9, ntor=7, lasym=asymmetric)
    arrays = {key: np.array(getattr(inp, key)) for key in ("rbc", "zbs", "rbs", "zbc")}
    arrays["rbc"][7, 0], arrays["rbc"][7, 1], arrays["zbs"][7, 1] = 10, 1, 1
    for key in ("rbc", "zbs", "rbs", "zbc")[:4 if asymmetric else 2]:
        arrays[key] += rng.normal(size=inp.rbc.shape) * 1e-7
    arrays["rbc"][9, 3] = 0.1
    inp = replace(inp, **arrays)
    compact = _compact_boundary(inp, 0.01)
    assert (compact.mpol, compact.ntor) == (4, 2)
    exact = _compact_boundary(inp, 0)
    assert (exact.mpol, exact.ntor) == (9, 7)
    # Independently reconstruct both coordinate families at random angles;
    # derivatives make this sensitive to lost small, high-frequency modes.
    u, v = rng.uniform(0, 2 * np.pi, (2, 301))
    def evaluate(deck, derivative):
        m = np.arange(deck.mpol)
        n = np.arange(-deck.ntor, deck.ntor + 1)[:, None]
        phase = m * u[:, None, None] - n * v[:, None, None]
        weight = 1 if derivative == 0 else (m if derivative == 1 else -n)
        c = np.cos(phase + (np.pi / 2 if derivative else 0)) * weight
        s = np.sin(phase + (np.pi / 2 if derivative else 0)) * weight
        return np.stack([np.sum(deck.rbc*c + deck.rbs*s, axis=(1, 2)),
                         np.sum(deck.zbc*c + deck.zbs*s, axis=(1, 2))])
    for derivative in range(3):
        assert np.max(abs(evaluate(inp, derivative) - evaluate(compact, derivative))) < 0.01


@pytest.fixture
def desc_equilibrium():
    pytest.importorskip("desc")
    import jax
    from desc.equilibrium import Equilibrium
    previous = jax.config.jax_disable_jit
    jax.config.update("jax_disable_jit", False)
    yield Equilibrium
    jax.config.update("jax_disable_jit", previous)


@pytest.mark.parametrize("family,current,asymmetric", [(False, False, False), (True, True, True)])
def test_hdf5_preserves_physical_profiles_and_surface(tmp_path, desc_equilibrium, family, current, asymmetric):
    from desc.equilibrium import EquilibriaFamily
    from desc.geometry import FourierRZToroidalSurface
    from vmex.core.profiles import current as evaluate_current, iota, pressure
    surface = FourierRZToroidalSurface(
        R_lmn=[10, 1, 0.07], modes_R=[[0, 0], [1, 0], [1, 1]],
        Z_lmn=[-1, 0.05], modes_Z=[[-1, 0], [1 if asymmetric else -1, 1]],
        NFP=3, sym=not asymmetric,
    )
    eq = desc_equilibrium(L=4, M=4, N=3, surface=surface, Psi=2.3,
                          pressure=np.array([[0, 1200], [2, -1000]]),
                          **({"current": np.array([[2, 8e3], [4, 3e3]])} if current else {"iota": np.array([[0, 0.4], [2, 0.1]])}))
    source = tmp_path / "eq.h5"
    (EquilibriaFamily(desc_equilibrium(), eq) if family else eq).save(str(source))
    inp = VmecInput.from_file(write_desc_input(source))
    assert (inp.nfp, inp.lasym, inp.phiedge) == (3, asymmetric, 2.3)
    assert (inp.mpol, inp.ntor) == (3, 2)
    s = np.linspace(0.01, 1, 41)
    np.testing.assert_allclose(pressure(inp.pmass_type, inp.am, inp.am_aux_s, inp.am_aux_f, s), 1200 - 1000*s)
    if current:
        values = evaluate_current(inp.pcurr_type, inp.ac, inp.ac_aux_s, inp.ac_aux_f, s)
        np.testing.assert_allclose(values, 8e3*s + 3e3*s**2)
        assert inp.curtor == 11000
    else:
        np.testing.assert_allclose(iota(inp.piota_type, inp.ai, inp.ai_aux_s, inp.ai_aux_f, s), 0.4 + 0.1*s)
    # Direct independent DESC evaluation checks Fourier signs, NFP and LASYM.
    from desc.grid import Grid
    rng = np.random.default_rng(742)
    theta, zeta = rng.uniform(0, 2*np.pi, (2, 83))
    data = eq.surface.compute(["R", "Z"], grid=Grid(np.column_stack([np.ones(83), theta, zeta]), sort=False))
    phase = np.arange(inp.mpol)*theta[:, None, None] - np.arange(-inp.ntor, inp.ntor+1)[None, :, None]*inp.nfp*zeta[:, None, None]
    for key, c, ss in [("R", inp.rbc, inp.rbs), ("Z", inp.zbc, inp.zbs)]:
        np.testing.assert_allclose(np.sum(c*np.cos(phase) + ss*np.sin(phase), axis=(1, 2)), data[key], atol=1e-8)


@pytest.mark.parametrize("saved", [False, True])
def test_cli_and_reproduction(tmp_path, desc_equilibrium, saved):
    from vmex.core.wout import read_wout
    source = tmp_path / "input.native"
    source.write_text("sym = 1\nNFP = 1\nPsi = 1\nM_pol = 4\nN_tor = 0\n"
                      "l: 0 p = 0 i = 0.4\nn: 0 R0 = 10 Z0 = 0\n"
                      "m: 0 n: 0 R1 = 10 Z1 = 0\nm: 1 n: 0 R1 = 1 Z1 = 0\n"
                      "m: -1 n: 0 R1 = 0 Z1 = -1\n")
    if saved:
        eq = desc_equilibrium.from_input_file(str(source))
        source = tmp_path / "input.native.h5"
        eq.save(str(source))
    original = source.read_bytes()
    outdir = tmp_path / "converted"
    assert cli.main([str(source), "--outdir", str(outdir)]) == 0
    deck = outdir / "input.input.native"
    first = read_wout(outdir / "wout_input.native.nc")
    np.testing.assert_allclose(first.iotaf, -0.4, atol=1e-12)
    assert source.read_bytes() == original
    assert cli.main([str(deck), "--outdir", str(tmp_path / "rerun"), "--quiet"]) == 0
    second = read_wout(tmp_path / "rerun" / "wout_input.native.nc")
    for field in ("rmnc", "zmns", "bmnc", "iotaf", "presf"):
        np.testing.assert_allclose(getattr(first, field), getattr(second, field), atol=1e-12, rtol=1e-12)
    assert write_desc_input(source).name == "input.input.native"


def test_invalid_desc_object(tmp_path, desc_equilibrium):
    from desc.geometry import FourierRZToroidalSurface
    path = tmp_path / "bad.h5"
    FourierRZToroidalSurface().save(str(path))
    with pytest.raises(VmecInputError, match="expected a DESC"):
        write_desc_input(path)
    path.write_bytes(b"not HDF5")
    with pytest.raises(VmecInputError, match="Cannot convert"):
        write_desc_input(path)


@pytest.mark.parametrize("kind", ["current", "iota"])
def test_spline_profiles_preserve_values_between_knots(tmp_path, desc_equilibrium, kind):
    from desc.profiles import SplineProfile
    from vmex.core.profiles import current, iota, pressure
    rho = np.linspace(0, 1, 17)
    prof = SplineProfile(1e4*rho**2 + 2e3*rho**4 if kind == "current" else 0.4+0.1*rho**2,
                         knots=rho, method="cubic2")
    p = SplineProfile(1000*(1-rho**2), knots=rho, method="cubic2")
    eq = desc_equilibrium(L=4, M=4, N=0, pressure=p, **{kind: prof})
    source = tmp_path / "splines.h5"
    eq.save(str(source))
    inp = VmecInput.from_file(write_desc_input(source))
    s = np.linspace(0, 1, 257)
    actual = (current(inp.pcurr_type, inp.ac, inp.ac_aux_s, inp.ac_aux_f, s) if kind == "current"
              else iota(inp.piota_type, inp.ai, inp.ai_aux_s, inp.ai_aux_f, s))
    expected = prof(np.sqrt(s))
    if kind == "current":
        actual = actual * inp.curtor / actual[-1]
    assert np.max(abs(np.asarray(actual-expected))) < 1e-3*np.max(abs(np.asarray(expected)))
    np.testing.assert_allclose(pressure(inp.pmass_type, inp.am, inp.am_aux_s, inp.am_aux_f,s),
                               p(np.sqrt(s)),atol=1,rtol=0)


def test_unreadable_input_is_typed(tmp_path):
    with pytest.raises(VmecInputError, match="Cannot read"):
        is_desc_file(tmp_path)


@pytest.mark.parametrize("extension", ["HDF5", "pkl", "pickle"])
def test_saved_formats_and_failure_do_not_replace_input(tmp_path, desc_equilibrium, extension):
    eq = desc_equilibrium(L=2, M=2, N=0)
    source = tmp_path / f"saved.{extension}"
    eq.save(str(source), file_format="hdf5" if extension == "HDF5" else "pickle")
    target = write_desc_input(source)
    before = target.read_bytes()
    assert target.name == "input.saved"
    source.write_bytes(b"corrupted equilibrium")
    assert cli.main([str(source), "--quiet"]) == 5
    assert target.read_bytes() == before


def test_truncation_retains_small_rapid_variations_and_is_scale_invariant():
    coeff = np.zeros((1, 21))
    coeff[0, :2] = [10, 1]
    coeff[0, 20] = 0.001
    z = np.zeros_like(coeff)
    z[0, 1] = 1
    inp = VmecInput(mpol=21, ntor=0, rbc=coeff, zbs=z)
    # Position alone permits removal, but its derivative exceeds the bound.
    assert _compact_boundary(inp, 0.01).mpol == 21
    for scale in (1e-3, 1e3):
        scaled = replace(inp, rbc=scale*coeff, zbs=scale*z)
        assert _compact_boundary(scaled, 0.01).mpol == 21


def test_native_final_continuation_stage(tmp_path, desc_equilibrium):
    source = tmp_path / "continuation.desc"
    source.write_text("sym = 1\nNFP = 3\nPsi = 2\nM_pol = 2, 4\nN_tor = 1\n"
                      "pres_ratio = 0, 1\nbdry_ratio = 0, 1\n"
                      "l: 0 p = 1200 i = 0.4\nl: 2 p = -1000 i = 0.1\n"
                      "n: 0 R0 = 10 Z0 = 0\n"
                      "m: 0 n: 0 R1 = 10 Z1 = 0\nm: 1 n: 0 R1 = 1 Z1 = 0\n"
                      "m: -1 n: 0 R1 = 0 Z1 = -1\nm: 1 n: 1 R1 = 0.1 Z1 = 0\n")
    inp = VmecInput.from_file(write_desc_input(source))
    np.testing.assert_allclose(inp.am[:2], [1200, -1000])
    assert inp.ntor == 2
    assert inp.nfp == 3
    assert np.max(abs(inp.rbc[[1, 3]])) > 0.01
