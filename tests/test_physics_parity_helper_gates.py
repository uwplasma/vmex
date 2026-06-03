from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from vmec_jax._compat import enable_x64, jnp
from vmec_jax.booz_input import booz_xform_inputs_from_state
from vmec_jax.config import load_config
from vmec_jax.finite_beta import finite_beta_scalars_from_state, magnetic_well_from_vp
from vmec_jax.profiles import MU0, eval_profiles
from vmec_jax.static import build_static
from vmec_jax.state import zeros_state
from vmec_jax.wout import read_wout, state_from_wout


def _data_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "examples" / "data"


def _single_grid_data_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "examples_single_grid" / "data"


def _full_and_half_mesh(ns: int) -> tuple[np.ndarray, np.ndarray]:
    s_full = np.linspace(0.0, 1.0, int(ns))
    if int(ns) < 2:
        return s_full, s_full
    return s_full, np.concatenate([s_full[:1], 0.5 * (s_full[1:] + s_full[:-1])])


def _normalized_equif_from_wout_profiles(wout) -> np.ndarray:
    """Reconstruct VMEC's normalized radial force-balance profile from wout fields."""
    ns = int(wout.ns)
    equif = np.zeros(ns, dtype=float)
    if ns < 3:
        return equif

    ohs = float(ns - 1)
    flux_scale = 2.0 * np.pi * float(wout.signgs)
    phipf = np.asarray(wout.phipf, dtype=float) / flux_scale
    chipf = np.asarray(wout.chipf, dtype=float) / flux_scale
    jcuru = np.asarray(wout.jcuru, dtype=float) * MU0
    jcurv = np.asarray(wout.jcurv, dtype=float) * MU0
    pres = np.asarray(wout.pres, dtype=float)
    vp = np.asarray(wout.vp, dtype=float)

    for js in range(1, ns - 1):
        vpphi = 0.5 * (vp[js + 1] + vp[js])
        presgrad = (pres[js + 1] - pres[js]) * ohs
        denom = abs(jcurv[js] * chipf[js]) + abs(jcuru[js] * phipf[js]) + abs(presgrad * vpphi)
        if denom != 0.0 and vpphi != 0.0:
            raw = ((-phipf[js] * jcuru[js] + chipf[js] * jcurv[js]) / vpphi) + presgrad
            equif[js] = raw * vpphi / denom

    equif[0] = 2.0 * equif[1] - equif[2]
    equif[-1] = 2.0 * equif[-2] - equif[-3]
    return equif


@pytest.mark.parametrize(
    ("case_name", "input_name", "wout_name"),
    (
        ("circular", "input.circular_tokamak", "wout_circular_tokamak.nc"),
        ("finite_beta_axisym", "input.shaped_tokamak_pressure", "wout_shaped_tokamak_pressure.nc"),
    ),
)
def test_small_wout_profile_jxbforce_mercier_and_finite_beta_gates(
    case_name: str,
    input_name: str,
    wout_name: str,
) -> None:
    """Small bundled fixtures should satisfy coupled profile/B/J/Mercier gates."""
    pytest.importorskip("netCDF4")

    _cfg, indata = load_config(str(_data_dir() / input_name))
    wout = read_wout(_data_dir() / wout_name)
    s_full, s_half = _full_and_half_mesh(int(wout.ns))

    profiles_full = eval_profiles(indata, s_full)
    profiles_half = eval_profiles(indata, s_half)
    np.testing.assert_allclose(
        np.asarray(profiles_full["pressure"]),
        np.asarray(wout.presf),
        rtol=1.0e-13,
        atol=1.0e-14,
        err_msg=case_name,
    )
    np.testing.assert_allclose(
        np.asarray(profiles_half["pressure"])[1:],
        np.asarray(wout.pres)[1:],
        rtol=1.0e-13,
        atol=1.0e-14,
        err_msg=case_name,
    )
    assert float(wout.pres[0]) == 0.0

    dmerc_parts = (
        np.asarray(wout.Dshear)
        + np.asarray(wout.Dcurr)
        + np.asarray(wout.Dwell)
        + np.asarray(wout.Dgeod)
    )
    np.testing.assert_allclose(
        np.asarray(wout.DMerc),
        dmerc_parts,
        rtol=0.0,
        atol=0.0,
        err_msg=case_name,
    )

    bdotgradv = np.zeros(int(wout.ns), dtype=float)
    bdotgradv[1:-1] = (
        float(wout.signgs)
        * (np.asarray(wout.phips)[1:-1] + np.asarray(wout.phips)[2:])
        / (np.asarray(wout.vp)[1:-1] + np.asarray(wout.vp)[2:])
    )
    bdotgradv[0] = 2.0 * bdotgradv[1] - bdotgradv[2]
    bdotgradv[-1] = 2.0 * bdotgradv[-2] - bdotgradv[-3]
    np.testing.assert_allclose(np.asarray(wout.bdotgradv), bdotgradv, rtol=1.0e-13, atol=1.0e-13)

    for name in ("bmnc", "jdotb", "bdotb", "bdotgradv", "DMerc"):
        values = np.asarray(getattr(wout, name), dtype=float)
        assert np.all(np.isfinite(values)), f"{case_name}.{name}"
        assert np.any(np.abs(values[1:-1]) > 0.0), f"{case_name}.{name}"

    dvol = np.abs(np.asarray(wout.vp, dtype=float))[1:]
    well_expected = (1.5 * dvol[0] - 0.5 * dvol[1] - (1.5 * dvol[-1] - 0.5 * dvol[-2])) / (
        1.5 * dvol[0] - 0.5 * dvol[1]
    )
    np.testing.assert_allclose(
        np.asarray(magnetic_well_from_vp(wout.vp)),
        well_expected,
        rtol=1.0e-13,
        atol=1.0e-13,
    )


@pytest.mark.parametrize(
    ("case_name", "wout_name"),
    (
        ("circular", "wout_circular_tokamak.nc"),
        ("finite_beta_axisym", "wout_shaped_tokamak_pressure.nc"),
    ),
)
def test_small_wout_current_scalar_and_force_balance_gate(case_name: str, wout_name: str) -> None:
    """Converged fixtures should preserve VMEC current and force-balance scalar identities."""
    pytest.importorskip("netCDF4")

    wout = read_wout(_data_dir() / wout_name)
    assert int(wout.ns) >= 3

    buco = np.asarray(wout.buco, dtype=float)
    ctor_expected = (
        float(wout.signgs) * (2.0 * np.pi) * (1.5 * float(buco[-1]) - 0.5 * float(buco[-2])) / MU0
    )
    np.testing.assert_allclose(
        float(wout.ctor),
        ctor_expected,
        rtol=1.0e-13,
        atol=1.0e-8,
        err_msg=f"{case_name}: ctor is not the VMEC edge-current extrapolation from buco",
    )

    equif_expected = _normalized_equif_from_wout_profiles(wout)
    np.testing.assert_allclose(
        np.asarray(wout.equif, dtype=float),
        equif_expected,
        rtol=1.0e-12,
        atol=2.0e-14,
        err_msg=f"{case_name}: equif no longer matches the normalized radial force-balance identity",
    )
    assert np.max(np.abs(np.asarray(wout.equif, dtype=float)[1:-1])) < 0.02, case_name


@pytest.mark.parametrize(
    ("case_name", "data_root", "input_name", "wout_name", "wp_rtol", "volume_rtol"),
    (
        (
            "finite_beta_axisym",
            "examples",
            "input.shaped_tokamak_pressure",
            "wout_shaped_tokamak_pressure.nc",
            1.0e-12,
            5.0e-9,
        ),
        (
            "finite_beta_lasym",
            "examples_single_grid",
            "input.basic_non_stellsym_pressure",
            "wout_basic_non_stellsym_pressure_reference.nc",
            5.0e-12,
            1.0e-8,
        ),
    ),
)
def test_finite_beta_state_scalars_match_wout_energy_gate(
    case_name: str,
    data_root: str,
    input_name: str,
    wout_name: str,
    wp_rtol: float,
    volume_rtol: float,
) -> None:
    """Differentiable finite-beta scalars should reproduce bundled VMEC2000 WOUT diagnostics."""
    pytest.importorskip("jax")
    pytest.importorskip("netCDF4")
    enable_x64(True)

    data_dir = _single_grid_data_dir() if data_root == "examples_single_grid" else _data_dir()
    cfg, indata = load_config(str(data_dir / input_name))
    wout = read_wout(data_dir / wout_name)
    cfg = replace(
        cfg,
        ns=int(wout.ns),
        mpol=int(wout.mpol),
        ntor=int(wout.ntor),
        nfp=int(wout.nfp),
        lasym=bool(wout.lasym),
        lthreed=bool(int(wout.ntor) > 0),
    )

    scalars = finite_beta_scalars_from_state(
        state=state_from_wout(wout),
        static=build_static(cfg),
        indata=indata,
        signgs=int(wout.signgs),
    )

    np.testing.assert_allclose(
        np.asarray(scalars["aspect"]),
        float(wout.aspect),
        rtol=1.0e-13,
        atol=1.0e-13,
        err_msg=case_name,
    )
    np.testing.assert_allclose(
        np.asarray(scalars["iotas"]),
        np.asarray(wout.iotas),
        rtol=1.0e-13,
        atol=1.0e-13,
        err_msg=case_name,
    )
    np.testing.assert_allclose(
        np.asarray(scalars["iotaf"]),
        np.asarray(wout.iotaf),
        rtol=1.0e-13,
        atol=1.0e-13,
        err_msg=case_name,
    )
    np.testing.assert_allclose(
        np.asarray(scalars["vp"]),
        np.asarray(wout.vp),
        rtol=1.0e-12,
        atol=1.0e-11,
        err_msg=case_name,
    )
    np.testing.assert_allclose(
        np.asarray(scalars["wb"]),
        float(wout.wb),
        rtol=1.0e-13,
        atol=1.0e-13,
        err_msg=case_name,
    )
    np.testing.assert_allclose(
        np.asarray(scalars["wp"]),
        float(wout.wp),
        rtol=wp_rtol,
        atol=1.0e-13,
        err_msg=case_name,
    )
    np.testing.assert_allclose(
        np.asarray(scalars["betatotal"]),
        float(wout.wp) / float(wout.wb),
        rtol=wp_rtol,
        atol=1.0e-15,
        err_msg=case_name,
    )
    np.testing.assert_allclose(
        np.asarray(scalars["volume"]) * (4.0 * np.pi**2),
        float(wout.volume_p),
        rtol=volume_rtol,
        atol=1.0e-12,
        err_msg=case_name,
    )
    np.testing.assert_allclose(
        np.asarray(scalars["volavgB"]),
        np.sqrt(2.0 * float(wout.wb) / (float(wout.volume_p) / (4.0 * np.pi**2))),
        rtol=volume_rtol,
        atol=1.0e-13,
        err_msg=case_name,
    )
    np.testing.assert_allclose(
        np.asarray(magnetic_well_from_vp(scalars["vp"])),
        np.asarray(magnetic_well_from_vp(wout.vp)),
        rtol=1.0e-12,
        atol=1.0e-13,
        err_msg=case_name,
    )
    assert float(np.asarray(scalars["wp"])) > 0.0
    assert 0.0 < float(np.asarray(scalars["betatotal"])) < 1.0e-2


def test_accepted_finite_beta_wout_scalar_jvp_matches_finite_difference() -> None:
    """Finite-beta accepted-state scalars should have AD derivatives matching finite differences."""
    jax = pytest.importorskip("jax")
    pytest.importorskip("netCDF4")
    enable_x64(True)

    cfg, indata = load_config(str(_data_dir() / "input.shaped_tokamak_pressure"))
    wout = read_wout(_data_dir() / "wout_shaped_tokamak_pressure.nc")
    cfg = replace(
        cfg,
        ns=int(wout.ns),
        mpol=int(wout.mpol),
        ntor=int(wout.ntor),
        nfp=int(wout.nfp),
        lasym=bool(wout.lasym),
        lthreed=bool(int(wout.ntor) > 0),
        ntheta=2 * int(wout.mpol) + 6,
        nzeta=1,
    )
    static = build_static(cfg)
    state_np = state_from_wout(wout)
    state = replace(
        state_np,
        Rcos=jnp.asarray(state_np.Rcos, dtype=jnp.float64),
        Rsin=jnp.asarray(state_np.Rsin, dtype=jnp.float64),
        Zcos=jnp.asarray(state_np.Zcos, dtype=jnp.float64),
        Zsin=jnp.asarray(state_np.Zsin, dtype=jnp.float64),
        Lcos=jnp.asarray(state_np.Lcos, dtype=jnp.float64),
        Lsin=jnp.asarray(state_np.Lsin, dtype=jnp.float64),
    )
    tangent0 = zeros_state(state.layout, like=state.Rcos)
    mode_idx = int(np.flatnonzero((np.asarray(static.modes.m) == 1) & (np.asarray(static.modes.n) == 0))[0])
    tangent = replace(tangent0, Rcos=tangent0.Rcos.at[-1, mode_idx].set(1.0))

    def scalar_vector(trial_state):
        scalars = finite_beta_scalars_from_state(
            state=trial_state,
            static=static,
            indata=indata,
            signgs=int(wout.signgs),
        )
        return jnp.asarray([scalars["betatotal"], scalars["volavgB"], scalars["volume"]], dtype=jnp.float64)

    def state_at(alpha):
        return replace(state, Rcos=state.Rcos + alpha * tangent.Rcos)

    value, tangent_ad = jax.jvp(scalar_vector, (state,), (tangent,))
    eps = jnp.asarray(1.0e-5, dtype=jnp.float64)
    tangent_fd = (scalar_vector(state_at(eps)) - scalar_vector(state_at(-eps))) / (2.0 * eps)

    assert np.all(np.isfinite(np.asarray(value)))
    np.testing.assert_allclose(np.asarray(tangent_ad), np.asarray(tangent_fd), rtol=5.0e-6, atol=1.0e-8)
    assert np.linalg.norm(np.asarray(tangent_ad)) > 0.0


def test_single_grid_lasym_finite_beta_force_balance_and_asymmetry_gate() -> None:
    """LASYM=T finite-beta WOUTs should preserve scalar identities and asymmetric spectra."""
    pytest.importorskip("netCDF4")

    wout = read_wout(_single_grid_data_dir() / "wout_basic_non_stellsym_pressure_reference.nc")
    assert bool(wout.lasym)
    assert float(wout.wp) > 0.0
    assert float(wout.wb) > 0.0
    np.testing.assert_allclose(float(wout.betatotal), float(wout.wp) / float(wout.wb), rtol=1.0e-13)
    residual_scalars = np.asarray([wout.fsqr, wout.fsqz, wout.fsql], dtype=float)
    assert np.all(np.isfinite(residual_scalars))
    assert np.max(np.abs(residual_scalars)) < 1.0e-9

    dmerc_parts = (
        np.asarray(wout.Dshear)
        + np.asarray(wout.Dcurr)
        + np.asarray(wout.Dwell)
        + np.asarray(wout.Dgeod)
    )
    np.testing.assert_allclose(np.asarray(wout.DMerc), dmerc_parts, rtol=0.0, atol=0.0)
    assert np.linalg.norm(np.asarray(wout.DMerc, dtype=float)[1:-1]) > 0.0

    equif_expected = _normalized_equif_from_wout_profiles(wout)
    np.testing.assert_allclose(
        np.asarray(wout.equif, dtype=float),
        equif_expected,
        rtol=1.0e-12,
        atol=2.0e-14,
    )
    assert np.max(np.abs(np.asarray(wout.equif, dtype=float)[1:-1])) > 0.1

    bdotgradv = np.zeros(int(wout.ns), dtype=float)
    bdotgradv[1:-1] = (
        float(wout.signgs)
        * (np.asarray(wout.phips)[1:-1] + np.asarray(wout.phips)[2:])
        / (np.asarray(wout.vp)[1:-1] + np.asarray(wout.vp)[2:])
    )
    bdotgradv[0] = 2.0 * bdotgradv[1] - bdotgradv[2]
    bdotgradv[-1] = 2.0 * bdotgradv[-2] - bdotgradv[-3]
    np.testing.assert_allclose(np.asarray(wout.bdotgradv), bdotgradv, rtol=1.0e-13, atol=1.0e-13)

    for name in ("rmns", "zmnc", "lmnc", "bmns", "bsubumns", "bsubvmns"):
        values = np.asarray(getattr(wout, name), dtype=float)
        assert np.all(np.isfinite(values)), name
        assert np.linalg.norm(values) > 0.0, name


def test_circular_fixture_boozer_inputs_preserve_wout_spectral_conventions() -> None:
    """Boozer input helpers should preserve fixture mode order and B spectra."""
    pytest.importorskip("jax")
    pytest.importorskip("netCDF4")
    enable_x64(True)

    cfg, indata = load_config(str(_data_dir() / "input.circular_tokamak"))
    wout = read_wout(_data_dir() / "wout_circular_tokamak.nc")
    cfg = replace(
        cfg,
        ns=int(wout.ns),
        mpol=int(wout.mpol),
        ntor=int(wout.ntor),
        nfp=int(wout.nfp),
        lasym=bool(wout.lasym),
        lthreed=bool(int(wout.ntor) > 0),
        ntheta=2 * int(wout.mpol) + 6,
        nzeta=1 if int(wout.ntor) == 0 else 2 * int(wout.ntor) + 4,
    )
    static = build_static(cfg)

    inputs = booz_xform_inputs_from_state(
        state=state_from_wout(wout),
        static=static,
        indata=indata,
        signgs=int(wout.signgs),
        use_nyq_from_grid=False,
    )

    np.testing.assert_array_equal(np.asarray(inputs.xm), np.asarray(wout.xm))
    np.testing.assert_array_equal(np.asarray(inputs.xn), np.asarray(wout.xn))
    np.testing.assert_array_equal(np.asarray(inputs.xm_nyq), np.asarray(wout.xm_nyq))
    np.testing.assert_array_equal(np.asarray(inputs.xn_nyq), np.asarray(wout.xn_nyq))
    np.testing.assert_allclose(
        np.asarray(inputs.iota),
        np.asarray(wout.iotas)[1:],
        rtol=5.0e-13,
        atol=5.0e-13,
    )
    np.testing.assert_allclose(
        np.asarray(inputs.lmns),
        np.asarray(wout.lmns)[1:],
        rtol=5.0e-13,
        atol=5.0e-13,
    )

    assert inputs.rmns is None
    assert inputs.zmnc is None
    assert inputs.lmnc is None
    assert inputs.bmns is None
    assert inputs.bsubumns is None
    assert inputs.bsubvmns is None

    np.testing.assert_allclose(np.asarray(inputs.bmnc), np.asarray(wout.bmnc)[1:], rtol=3.0e-9, atol=5.0e-8)
    np.testing.assert_allclose(
        np.asarray(inputs.bsubumnc),
        np.asarray(wout.bsubumnc)[1:],
        rtol=5.0e-13,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        np.asarray(inputs.bsubvmnc),
        np.asarray(wout.bsubvmnc)[1:],
        rtol=3.0e-5,
        atol=1.0e-3,
    )
