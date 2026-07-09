"""A/B tests: new clean-core input parsing and profiles vs the legacy code.

For every bundled input deck, :class:`vmec_jax.core.input.VmecInput` is
compared field-by-field against the historical parser stack
(``vmec_jax.namelist`` + ``vmec_jax.config`` + ``vmec_jax.profiles``), both
writers are round-tripped, and the profile evaluators are compared on a
radial grid at rtol 1e-13.  A VMEC++ JSON example (``data/solovev.json``,
copied verbatim from the vmecpp repository) validates JSON-schema
compatibility.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vmec_jax.config import config_from_indata
from vmec_jax.core import profiles as core_profiles
from vmec_jax.core.input import VmecInput
from vmec_jax.namelist import InData, read_indata
from vmec_jax.profiles import eval_profiles, profiles_from_indata

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "examples" / "data"
DECKS = sorted(DATA.glob("input.*"))
FIXTURES = Path(__file__).parent / "data"

assert DECKS, f"no input decks found under {DATA}"

# Decks covering the distinct pmass/piota/pcurr parameterizations in the repo:
# power_series (solovev, DSHAPE), two_power pressure+current (cth_like),
# cubic_spline pressure+iota (profile_splines), cubic_spline_ip current
# (nfp2_QA_finite_beta).
PROFILE_DECKS = [
    "input.solovev",
    "input.DSHAPE",
    "input.cth_like_free_bdy",
    "input.profile_splines",
    "input.nfp2_QA_finite_beta",
]


def _as_list(value) -> list:
    if value is None:
        return []
    return list(value) if isinstance(value, list) else [value]


def _dense_boundary(indata: InData, name: str, mpol: int, ntor: int) -> np.ndarray:
    """Old-parser indexed boundary coefficients densified to [n+ntor, m]."""
    grid = np.zeros((2 * ntor + 1, mpol))
    for idx, value in indata.indexed.get(name, {}).items():
        if len(idx) != 2:
            continue
        n, m = idx
        if -ntor <= n <= ntor and 0 <= m < mpol:
            grid[n + ntor, m] = float(value)
    return grid


def _dense_axis(indata: InData, name: str, ntor: int, legacy: str | None = None) -> np.ndarray:
    """Expected dense axis array per VMEC2000 read_indata_namelist."""
    out = np.zeros(ntor + 1)
    given = [float(v) for v in _as_list(indata.get(name, []))]
    out[: min(len(given), ntor + 1)] = given[: ntor + 1]
    for idx, value in indata.indexed.get(name, {}).items():
        if len(idx) == 1 and 0 <= idx[0] <= ntor:
            out[idx[0]] = float(value)
    if legacy is not None:
        old = np.zeros(ntor + 1)
        given = [float(v) for v in _as_list(indata.get(legacy, []))]
        old[: min(len(given), ntor + 1)] = given[: ntor + 1]
        out = np.where(old != 0.0, old, out)
    return out


@pytest.mark.parametrize("deck", DECKS, ids=lambda p: p.name)
def test_indata_matches_legacy_parser(deck: Path) -> None:
    """New VmecInput fields equal the legacy namelist/config/profile values."""
    old = read_indata(deck)
    cfg = config_from_indata(old)
    prof = profiles_from_indata(old)
    new = VmecInput.from_file(deck)

    # Resolution / symmetry (raw INDATA values with VMEC2000 defaults).
    assert new.nfp == old.get_int("NFP", 1)
    assert new.mpol == old.get_int("MPOL", 6)
    assert new.ntor == old.get_int("NTOR", 0)
    assert new.lasym == old.get_bool("LASYM", False)
    assert new.ntheta == old.get_int("NTHETA", 0)
    assert new.nzeta == old.get_int("NZETA", 0)

    # Scalars.
    assert new.delt == old.get_float("DELT", 1.0)
    assert new.tcon0 == old.get_float("TCON0", 1.0)
    assert new.phiedge == old.get_float("PHIEDGE", 1.0)
    assert new.nstep == old.get_int("NSTEP", 10)
    assert new.gamma == old.get_float("GAMMA", 0.0)
    assert new.curtor == old.get_float("CURTOR", 0.0)
    assert new.ncurr == old.get_int("NCURR", 0)
    assert new.bloat == prof.bloat
    assert new.spres_ped == old.get_float("SPRES_PED", 1.0)
    assert new.pres_scale == prof.pres_scale
    assert new.mfilter_fbdy == old.get_int("MFILTER_FBDY", -1)
    assert new.nfilter_fbdy == old.get_int("NFILTER_FBDY", -1)
    assert new.precon_type == str(old.get("PRECON_TYPE", "NONE")).strip()
    assert new.prec2d_threshold == old.get_float("PREC2D_THRESHOLD", 1e-30)

    # Free-boundary block (legacy config applies the readin.f normalizations).
    assert new.lfreeb == cfg.lfreeb
    assert new.mgrid_file == cfg.mgrid_file
    # Known default divergence: legacy vmec_jax defaults NVACSKIP to nfp,
    # while VMEC2000 (vmec_input.f) defaults it to 1 and readin.f only
    # replaces non-positive values.  The new code follows VMEC2000.
    if old.get("NVACSKIP") is not None:
        assert new.nvacskip == cfg.nvacskip
    else:
        assert new.nvacskip == 1
    np.testing.assert_array_equal(new.extcur, np.asarray(cfg.extcur))

    # Multigrid ladder: the deck-provided prefix must match exactly, and the
    # three arrays must share one length.
    assert len(new.ns_array) == len(new.ftol_array) == len(new.niter_array)
    ns_given = [int(v) for v in _as_list(old.get("NS_ARRAY", []))]
    if ns_given:
        n = len(new.ns_array)
        np.testing.assert_array_equal(new.ns_array, ns_given[:n])
    for name, arr in (("FTOL_ARRAY", new.ftol_array), ("NITER_ARRAY", new.niter_array)):
        given = [float(v) for v in _as_list(old.get(name, []))]
        if given:
            k = min(len(given), len(arr))
            np.testing.assert_array_equal(arr[:k], given[:k])

    # Profile types and coefficient arrays (legacy values are padded to >=21,
    # exactly like the new dense storage).
    assert new.pmass_type == prof.pmass_type
    assert new.pcurr_type == prof.pcurr_type
    assert new.piota_type == prof.piota_type
    for name in ("am", "ac", "ai"):
        np.testing.assert_array_equal(getattr(new, name), np.asarray(getattr(prof, name)))
    for name in ("am_aux_s", "am_aux_f", "ac_aux_s", "ac_aux_f", "ai_aux_s", "ai_aux_f"):
        np.testing.assert_array_equal(getattr(new, name), np.asarray(getattr(prof, name)))

    # Axis arrays (RAXIS/ZAXIS backwards compatibility included).  The n = 0
    # sine coefficients are zeroed by read_indata_namelist after the merge
    # (raxis_cs(0) = 0; zaxis_cs(0) = 0) since sin(0) terms are meaningless.
    np.testing.assert_array_equal(new.raxis_c, _dense_axis(old, "RAXIS_CC", new.ntor, "RAXIS"))
    expected_zaxis_s = _dense_axis(old, "ZAXIS_CS", new.ntor, "ZAXIS")
    expected_zaxis_s[0] = 0.0
    np.testing.assert_array_equal(new.zaxis_s, expected_zaxis_s)
    expected_raxis_s = _dense_axis(old, "RAXIS_CS", new.ntor)
    expected_raxis_s[0] = 0.0
    np.testing.assert_array_equal(new.raxis_s, expected_raxis_s)
    np.testing.assert_array_equal(new.zaxis_c, _dense_axis(old, "ZAXIS_CC", new.ntor))

    # Boundary coefficient grids, [n + ntor, m].
    for name in ("RBC", "ZBS", "RBS", "ZBC"):
        np.testing.assert_array_equal(
            getattr(new, name.lower()),
            _dense_boundary(old, name, new.mpol, new.ntor),
            err_msg=name,
        )


@pytest.mark.parametrize("deck", DECKS, ids=lambda p: p.name)
def test_json_round_trip(deck: Path, tmp_path: Path) -> None:
    """VmecInput -> to_json -> from_file reproduces every field exactly."""
    original = VmecInput.from_file(deck)
    reread = VmecInput.from_file(original.to_json(tmp_path / "roundtrip.json"))
    assert reread == original


@pytest.mark.parametrize("deck", DECKS, ids=lambda p: p.name)
def test_indata_round_trip(deck: Path, tmp_path: Path) -> None:
    """VmecInput -> to_indata -> from_file reproduces every field exactly."""
    original = VmecInput.from_file(deck)
    reread = VmecInput.from_file(original.to_indata(tmp_path / "input.roundtrip"))
    assert reread == original


@pytest.mark.parametrize("deck_name", PROFILE_DECKS)
def test_profiles_match_legacy(deck_name: str) -> None:
    """New pressure/iota/current evaluators match legacy eval_profiles."""
    deck = DATA / deck_name
    s = np.linspace(0.0, 1.0, 50)
    old = eval_profiles(read_indata(deck), s)
    new = VmecInput.from_file(deck)

    def check(new_values, old_values) -> None:
        old_values = np.asarray(old_values)
        atol = 1e-13 * (1.0 + float(np.max(np.abs(old_values))))
        np.testing.assert_allclose(np.asarray(new_values), old_values,
                                   rtol=1e-13, atol=atol)

    check(
        core_profiles.pressure(
            new.pmass_type, new.am, new.am_aux_s, new.am_aux_f, s,
            pres_scale=new.pres_scale, bloat=new.bloat, spres_ped=new.spres_ped,
        ),
        old["pressure_pa"],
    )
    # Legacy "pressure" is the VMEC-internal (mu0 * Pa) variant.
    check(
        core_profiles.MU0
        * core_profiles.pressure(
            new.pmass_type, new.am, new.am_aux_s, new.am_aux_f, s,
            pres_scale=new.pres_scale, bloat=new.bloat, spres_ped=new.spres_ped,
        ),
        old["pressure"],
    )
    check(
        core_profiles.iota(new.piota_type, new.ai, new.ai_aux_s, new.ai_aux_f, s,
                           bloat=new.bloat),
        old["iota"],
    )
    # Integrated current parameterizations ('two_power'/'gauss_trunc') now use
    # VMEC2000's exact 10-point Gauss-Legendre rule (profile_functions.f
    # gln = 10); the legacy evaluator used a 16-point rule, which deviates
    # from VMEC2000 by ~2e-6 relative (the bug was exposed by the end-to-end
    # cth_like_fixed_bdy wout parity test).  Compare quadrature-based lanes at
    # the known legacy deviation; everything else stays strict.
    quadrature_lane = new.pcurr_type in ("two_power", "gauss_trunc")
    current_new = core_profiles.current(
        new.pcurr_type, new.ac, new.ac_aux_s, new.ac_aux_f, s, bloat=new.bloat
    )
    if quadrature_lane:
        np.testing.assert_allclose(np.asarray(current_new),
                                   np.asarray(old["current"]),
                                   rtol=5e-6, atol=1e-12)
    else:
        check(current_new, old["current"])


def test_vmecpp_json_example() -> None:
    """The VMEC++ solovev.json example parses with VMEC++ schema semantics."""
    new = VmecInput.from_file(FIXTURES / "solovev.json")
    assert new.mpol == 6
    assert new.ntor == 0
    assert new.nfp == 1  # default
    assert new.lasym is False  # default
    assert new.ncurr == 0
    assert new.nstep == 250
    assert new.delt == 0.9
    np.testing.assert_array_equal(new.ns_array, [5, 11, 55])
    np.testing.assert_array_equal(new.niter_array, [1000, 2000, 2000])
    np.testing.assert_array_equal(new.ftol_array, [1e-16, 1e-16, 1e-16])
    np.testing.assert_array_equal(new.am[:3], [0.125, -0.125, 0.0])
    assert not np.any(new.am[3:])
    np.testing.assert_array_equal(new.ai[:2], [1.0, 0.0])
    np.testing.assert_array_equal(new.raxis_c, [4.0])
    np.testing.assert_array_equal(new.zaxis_s, [0.0])
    # Sparse boundary lists land at [n + ntor, m] = [0, m] for ntor = 0.
    np.testing.assert_array_equal(new.rbc[0, :3], [3.999, 1.026, -0.068])
    np.testing.assert_array_equal(new.zbs[0, :3], [0.0, 1.58, 0.01])
    assert not np.any(new.rbs) and not np.any(new.zbc)
    assert new.lfreeb is False  # VMEC++ default
    assert new.gamma == 0.0  # default (JSON alias: adiabatic_index)
    assert new.mgrid_file == "NONE"

    # And it survives both writers.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        assert VmecInput.from_file(new.to_json(Path(tmp) / "x.json")) == new
        assert VmecInput.from_file(new.to_indata(Path(tmp) / "input.x")) == new
