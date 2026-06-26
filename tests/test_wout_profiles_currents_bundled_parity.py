from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vmec_jax.boundary import boundary_from_indata
from vmec_jax.config import load_config
from vmec_jax.energy import _iotaf_from_iotas, flux_profiles_from_indata
from vmec_jax.integrals import cumrect_s_halfmesh
from vmec_jax.modes import vmec_mode_table
from vmec_jax.profiles import MU0, eval_profiles
from vmec_jax.static import build_static
from vmec_jax.wout import _chipf_from_chips, equilibrium_iota_profiles_from_state, read_wout, state_from_wout


PROFILE_CASES = (
    (
        "axisymmetric_finite_beta",
        "examples/data/input.shaped_tokamak_pressure",
        "examples/data/wout_shaped_tokamak_pressure.nc",
        True,
    ),
    (
        "current_driven_3d",
        "examples/data/input.nfp4_QH_warm_start",
        "examples/data/wout_nfp4_QH_warm_start.nc",
        False,
    ),
    (
        "qi_stel_seed_3127",
        "examples/data/input.QI_stel_seed_3127",
        "examples/data/wout_QI_stel_seed_3127.nc",
        False,
    ),
    (
        "finite_beta_3d",
        "examples/data/input.li383_low_res",
        "examples/data/wout_li383_low_res.nc",
        True,
    ),
    (
        "cth",
        "examples/data/input.cth_like_fixed_bdy",
        "examples/data/wout_cth_like_fixed_bdy.nc",
        True,
    ),
    (
        "lasym_3d",
        "examples/data/input.basic_non_stellsym_simsopt",
        "examples/data/wout_basic_non_stellsym_simsopt.nc",
        False,
    ),
    (
        "single_grid_lasym_pressure",
        "examples/data/single_grid/input.basic_non_stellsym_pressure",
        "examples/data/single_grid/wout_basic_non_stellsym_pressure_reference.nc",
        True,
    ),
)

CURRENT_CASES = (
    ("axisymmetric", "examples/data/wout_circular_tokamak.nc"),
    ("axisymmetric_finite_beta", "examples/data/wout_shaped_tokamak_pressure.nc"),
    ("current_driven_3d", "examples/data/wout_nfp4_QH_warm_start.nc"),
    ("qi_stel_seed_3127", "examples/data/wout_QI_stel_seed_3127.nc"),
    ("finite_beta_3d", "examples/data/wout_li383_low_res.nc"),
    ("cth", "examples/data/wout_cth_like_fixed_bdy.nc"),
    ("lasym_3d", "examples/data/wout_basic_non_stellsym_simsopt.nc"),
    ("single_grid_lasym_pressure", "examples/data/single_grid/wout_basic_non_stellsym_pressure_reference.nc"),
)

CURRENT_DRIVEN_STATE_CASES = (
    (
        "LandremanPaul2021_QA_lowres",
        "examples/data/input.LandremanPaul2021_QA_lowres",
        "examples/data/wout_LandremanPaul2021_QA_lowres.nc",
    ),
    (
        "nfp4_QH_warm_start",
        "examples/data/input.nfp4_QH_warm_start",
        "examples/data/wout_nfp4_QH_warm_start.nc",
    ),
    (
        "QI_stel_seed_3127",
        "examples/data/input.QI_stel_seed_3127",
        "examples/data/wout_QI_stel_seed_3127.nc",
    ),
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _s_full(ns: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, ns)


def _s_half(s: np.ndarray) -> np.ndarray:
    if s.size < 2:
        return s.copy()
    return np.concatenate([s[:1], 0.5 * (s[1:] + s[:-1])])


def _vmec_half_from_full(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return values.copy()

    out = np.zeros_like(values)
    if values.size >= 3:
        out[0] = 1.5 * values[1] - 0.5 * values[2]
    else:
        out[0] = values[1]
    out[1:-1] = 0.5 * (values[1:-1] + values[2:])
    out[-1] = 1.5 * values[-1] - 0.5 * values[-2]
    return out


@pytest.mark.parametrize(
    ("case_name", "input_name", "wout_name"),
    CURRENT_DRIVEN_STATE_CASES,
    ids=[case[0] for case in CURRENT_DRIVEN_STATE_CASES],
)
def test_current_driven_iota_profiles_recompute_from_bundled_state(
    case_name: str,
    input_name: str,
    wout_name: str,
) -> None:
    """Current-driven accepted states must regenerate VMEC add-fluxes profiles.

    These are no-solve parity gates: the geometry is read from a VMEC2000 WOUT,
    then vmec_jax recomputes ``chips``, ``iotas``, ``iotaf``, and WOUT
    half-mesh ``chipf`` from that state.  This catches regressions where output
    writers accidentally use input-only zero-current profiles for ``NCURR=1``.
    """

    pytest.importorskip("jax")
    pytest.importorskip("netCDF4")
    from vmec_jax._compat import enable_x64

    enable_x64(True)

    repo_root = _repo_root()
    input_path = repo_root / input_name
    wout_path = repo_root / wout_name
    if not input_path.exists() or not wout_path.exists():
        pytest.skip(f"Missing bundled current-driven fixture: {case_name}")

    cfg, indata = load_config(str(input_path))
    if int(indata.get_int("NCURR", 0)) != 1:
        pytest.skip(f"{case_name} is not current-driven")

    wout = read_wout(wout_path)
    static = build_static(cfg)
    state = state_from_wout(wout)

    chips, iotas, iotaf = equilibrium_iota_profiles_from_state(
        state=state,
        static=static,
        indata=indata,
        signgs=int(wout.signgs),
    )
    chips_np = np.asarray(chips, dtype=float)
    iotas_np = np.asarray(iotas, dtype=float)
    iotaf_np = np.asarray(iotaf, dtype=float)
    chipf_internal = np.asarray(_chipf_from_chips(chips_np), dtype=float)
    chipf_wout = chipf_internal * (2.0 * np.pi * int(wout.signgs))

    assert np.all(np.isfinite(chips_np))
    assert np.all(np.isfinite(iotas_np))
    assert np.all(np.isfinite(iotaf_np))
    np.testing.assert_allclose(
        iotas_np,
        np.asarray(wout.iotas, dtype=float),
        rtol=2.5e-11,
        atol=2.5e-11,
        err_msg=f"{case_name}: recomputed current-driven iotas drifted",
    )
    np.testing.assert_allclose(
        iotaf_np,
        np.asarray(wout.iotaf, dtype=float),
        rtol=2.5e-11,
        atol=2.5e-11,
        err_msg=f"{case_name}: recomputed current-driven iotaf drifted",
    )
    np.testing.assert_allclose(
        chipf_wout,
        np.asarray(wout.chipf, dtype=float),
        rtol=2.5e-11,
        atol=2.5e-11,
        err_msg=f"{case_name}: recomputed current-driven chipf drifted",
    )


def _input_r00(indata) -> float:
    modes = vmec_mode_table(int(indata.get_int("MPOL", 1)), int(indata.get_int("NTOR", 0)))
    boundary = boundary_from_indata(indata, modes)
    mask = (np.asarray(modes.m, dtype=int) == 0) & (np.asarray(modes.n, dtype=int) == 0)
    index = int(np.where(mask)[0][0]) if np.any(mask) else 0
    return float(np.asarray(boundary.R_cos, dtype=float)[index])


def _expected_internal_pressure(indata, wout, s_half: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Return VMEC internal pressure and optional mass expected from input profiles.

    For ``GAMMA = 0`` VMEC writes the pressure profile directly.  For nonzero
    ``GAMMA`` VMEC stores the mass profile from the input deck and reconstructs
    pressure using the solved volume derivative, so the profile gate must follow
    that same convention instead of comparing against the raw ``AM`` polynomial.
    """

    profiles = eval_profiles(indata, s_half)
    pressure_pa = np.array(profiles.get("pressure_pa", np.zeros_like(s_half)), dtype=float, copy=True)
    pressure_internal = np.array(profiles.get("pressure", np.zeros_like(s_half)), dtype=float, copy=True)
    if pressure_pa.size:
        pressure_pa[0] = 0.0
    if pressure_internal.size:
        pressure_internal[0] = 0.0

    gamma = float(indata.get_float("GAMMA", 0.0))
    if gamma == 0.0:
        return pressure_internal, None

    lrfp = bool(indata.get_bool("LRFP", False))
    vnorm = np.asarray(wout.chipf if lrfp else wout.phips, dtype=float)
    mass_expected = pressure_pa * (np.abs(vnorm) * _input_r00(indata)) ** gamma
    if mass_expected.size:
        mass_expected[0] = 0.0
    vp = np.asarray(wout.vp, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        pressure_expected_pa = np.where(vp != 0.0, mass_expected / np.where(vp != 0.0, vp, 1.0) ** gamma, 0.0)
    pressure_expected = MU0 * pressure_expected_pa
    if pressure_expected.size:
        pressure_expected[0] = 0.0
    return pressure_expected, mass_expected


@pytest.mark.parametrize(
    ("case_name", "input_name", "wout_name", "expect_pressure"),
    PROFILE_CASES,
    ids=[case[0] for case in PROFILE_CASES],
)
def test_bundled_wout_flux_pressure_iota_profiles_follow_vmec_radial_mesh(
    case_name: str,
    input_name: str,
    wout_name: str,
    expect_pressure: bool,
) -> None:
    """Solved bundled wouts should preserve VMEC input profiles and radial staggering."""
    pytest.importorskip("jax")
    pytest.importorskip("netCDF4")
    from vmec_jax._compat import enable_x64

    enable_x64(True)

    repo_root = _repo_root()
    input_path = repo_root / input_name
    wout_path = repo_root / wout_name
    if not input_path.exists() or not wout_path.exists():
        pytest.skip(f"Missing bundled profile fixture: {case_name}")

    _cfg, indata = load_config(str(input_path))
    wout = read_wout(wout_path)
    ns = int(wout.ns)
    s = _s_full(ns)

    flux = flux_profiles_from_indata(indata, s, signgs=int(wout.signgs))
    phipf_expected = np.asarray(flux.phipf, dtype=float) * (2.0 * np.pi * int(wout.signgs))
    phi_expected = np.asarray(cumrect_s_halfmesh(phipf_expected, s), dtype=float)

    np.testing.assert_allclose(
        np.asarray(wout.phipf, dtype=float),
        phipf_expected,
        rtol=1.0e-13,
        atol=1.0e-13,
        err_msg=f"{case_name}: phipf does not match input PHIEDGE/APHI profile",
    )
    np.testing.assert_allclose(
        np.asarray(wout.phi, dtype=float),
        phi_expected,
        rtol=1.0e-13,
        atol=1.0e-13,
        err_msg=f"{case_name}: phi is not the VMEC half-mesh integral of phipf",
    )
    np.testing.assert_allclose(
        np.asarray(wout.chipf, dtype=float),
        np.asarray(wout.iotaf, dtype=float) * np.asarray(wout.phipf, dtype=float),
        rtol=1.0e-13,
        atol=1.0e-13,
        err_msg=f"{case_name}: chipf is not consistent with iotaf * phipf on the half mesh",
    )

    s_half = _s_half(s)
    profiles = eval_profiles(indata, s_half)
    pressure_expected, mass_expected = _expected_internal_pressure(indata, wout, s_half)

    if expect_pressure:
        assert np.max(np.abs(np.asarray(wout.pres, dtype=float))) > 0.0
    if mass_expected is not None:
        import netCDF4

        with netCDF4.Dataset(repo_root / wout_name) as ds:
            if "mass" in ds.variables:
                np.testing.assert_allclose(
                    np.asarray(ds.variables["mass"][:], dtype=float),
                    mass_expected,
                    rtol=1.0e-12,
                    atol=1.0e-10,
                    err_msg=f"{case_name}: mass profile drifted from VMEC input pressure profile",
                )
    np.testing.assert_allclose(
        np.asarray(wout.pres, dtype=float),
        pressure_expected,
        rtol=1.0e-12,
        atol=1.0e-14,
        err_msg=f"{case_name}: pressure profile drifted from VMEC input profile",
    )
    np.testing.assert_allclose(
        np.asarray(wout.presf, dtype=float),
        _vmec_half_from_full(pressure_expected),
        rtol=1.0e-12,
        atol=1.0e-14,
        err_msg=f"{case_name}: presf does not follow VMEC full-to-half radial stencil",
    )

    hs = 1.0 / float(ns - 1)
    wp_expected = hs * float(np.sum(np.asarray(wout.vp, dtype=float)[1:] * np.asarray(wout.pres, dtype=float)[1:]))
    np.testing.assert_allclose(
        float(wout.wp),
        wp_expected,
        rtol=1.0e-13,
        atol=1.0e-14,
        err_msg=f"{case_name}: wp is not the VMEC half-mesh pressure integral",
    )
    assert float(wout.wb) > 0.0, f"{case_name}: bundled fixture lacks positive magnetic energy"
    np.testing.assert_allclose(
        float(wout.betatotal),
        float(wout.wp) / float(wout.wb),
        rtol=1.0e-13,
        atol=1.0e-14,
        err_msg=f"{case_name}: betatotal no longer matches wp / wb",
    )
    volume_expected = 4.0 * np.pi**2 * hs * float(np.sum(np.asarray(wout.vp, dtype=float)[1:]))
    np.testing.assert_allclose(
        float(wout.volume_p),
        volume_expected,
        rtol=2.0e-6,
        atol=1.0e-8,
        err_msg=f"{case_name}: volume_p is not the VMEC radial integral of vp",
    )
    beta_scalars = np.asarray([wout.betatotal, wout.betapol, wout.betator, wout.betaxis], dtype=float)
    assert np.all(np.isfinite(beta_scalars)), f"{case_name}: beta scalars must be finite"
    if expect_pressure:
        assert np.all(beta_scalars > 0.0), f"{case_name}: finite-pressure fixture lost positive beta scalars"
    else:
        np.testing.assert_allclose(
            beta_scalars,
            0.0,
            rtol=0.0,
            atol=0.0,
            err_msg=f"{case_name}: zero-pressure fixture gained finite beta scalars",
        )

    iotaf_expected = np.asarray(
        _iotaf_from_iotas(
            np.asarray(wout.iotas, dtype=float),
            lrfp=bool(indata.get_bool("LRFP", False)),
        ),
        dtype=float,
    )
    np.testing.assert_allclose(
        np.asarray(wout.iotaf, dtype=float),
        iotaf_expected,
        rtol=1.0e-13,
        atol=1.0e-13,
        err_msg=f"{case_name}: iotaf does not follow VMEC iotas radial stencil",
    )

    if int(indata.get_int("NCURR", 0)) == 0:
        iotas_expected = np.array(profiles.get("iota", np.zeros((ns,))), dtype=float, copy=True)
        if iotas_expected.size:
            iotas_expected[0] = 0.0
        np.testing.assert_allclose(
            np.asarray(wout.iotas, dtype=float),
            iotas_expected,
            rtol=1.0e-13,
            atol=1.0e-13,
            err_msg=f"{case_name}: iotas drifted from VMEC input profile",
        )


@pytest.mark.parametrize("case_name,wout_name", CURRENT_CASES, ids=[case[0] for case in CURRENT_CASES])
def test_bundled_wout_surface_averaged_currents_follow_ampere_radial_difference(
    case_name: str,
    wout_name: str,
) -> None:
    """Stored J profiles should match VMEC's finite-difference of buco/bvco."""
    pytest.importorskip("netCDF4")

    wout_path = _repo_root() / wout_name
    if not wout_path.exists():
        pytest.skip(f"Missing bundled current-profile fixture: {case_name}")

    wout = read_wout(wout_path)
    ns = int(wout.ns)
    assert ns >= 4

    hs = 1.0 / float(ns - 1)
    signgs = float(wout.signgs)
    buco = np.asarray(wout.buco, dtype=float)
    bvco = np.asarray(wout.bvco, dtype=float)

    jcuru_expected = -signgs * (bvco[2:] - bvco[1:-1]) / (hs * MU0)
    jcurv_expected = signgs * (buco[2:] - buco[1:-1]) / (hs * MU0)
    jcuru = np.asarray(wout.jcuru, dtype=float)[1:-1]
    jcurv = np.asarray(wout.jcurv, dtype=float)[1:-1]

    assert np.max(np.abs(jcuru)) > 0.0 or np.max(np.abs(jcurv)) > 0.0
    np.testing.assert_allclose(
        jcuru,
        jcuru_expected,
        rtol=1.0e-12,
        atol=1.0e-9,
        err_msg=f"{case_name}: jcuru no longer matches -d(bvco)/ds / mu0",
    )
    np.testing.assert_allclose(
        jcurv,
        jcurv_expected,
        rtol=1.0e-12,
        atol=1.0e-9,
        err_msg=f"{case_name}: jcurv no longer matches d(buco)/ds / mu0",
    )
