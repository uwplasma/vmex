from __future__ import annotations

import numpy as np
import pytest

from vmec_jax._compat import jnp
from vmec_jax.fourier import (
    HelicalBasis,
    build_helical_basis,
    eval_fourier,
    eval_fourier_dtheta,
    eval_fourier_dzeta_phys,
    project_to_modes,
)
from vmec_jax.grids import make_angle_grid
from vmec_jax.integrals import (
    cumrect_s_halfmesh,
    cumtrapz_s,
    dvds_from_sqrtg,
    dvds_from_sqrtg_zeta,
    volume_from_sqrtg,
    volume_from_sqrtg_vmec,
)
from vmec_jax.modes import vmec_mode_table
from vmec_jax.namelist import InData
from vmec_jax.profiles import (
    MU0,
    ProfileInputs,
    _as_float_list,
    _cubic_spline_profile,
    _lower,
    eval_profiles,
    profiles_from_indata,
)


def test_fourier_projection_roundtrip_and_raw_inner_products() -> None:
    modes = vmec_mode_table(mpol=3, ntor=1)
    grid = make_angle_grid(ntheta=18, nzeta=16, nfp=2)
    basis = build_helical_basis(modes, grid, cache=False)

    coeff_cos = np.linspace(0.1, 0.8, modes.K)
    coeff_sin = np.linspace(-0.2, 0.3, modes.K)
    coeff_sin[0] = 0.0

    field = eval_fourier(coeff_cos, coeff_sin, basis)
    cos_back, sin_back = project_to_modes(field, basis)

    np.testing.assert_allclose(np.asarray(cos_back), coeff_cos, atol=1e-13)
    np.testing.assert_allclose(np.asarray(sin_back)[1:], coeff_sin[1:], atol=1e-13)
    assert float(np.asarray(sin_back)[0]) == pytest.approx(0.0, abs=1e-14)

    raw_cos, raw_sin = project_to_modes(field, basis, normalize=False)
    np.testing.assert_allclose(np.asarray(raw_cos)[0], coeff_cos[0] * grid.ntheta * grid.nzeta, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(raw_cos)[1:],
        coeff_cos[1:] * grid.ntheta * grid.nzeta / 2.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(raw_sin)[1:],
        coeff_sin[1:] * grid.ntheta * grid.nzeta / 2.0,
        atol=1e-12,
    )


def test_fourier_fallback_without_phase_stack_matches_stacked_path() -> None:
    modes = vmec_mode_table(mpol=3, ntor=1)
    grid = make_angle_grid(ntheta=12, nzeta=10, nfp=3)
    basis = build_helical_basis(modes, grid, cache=False)
    basis_no_stack = HelicalBasis(
        cos_phase=basis.cos_phase,
        sin_phase=basis.sin_phase,
        phase_stack=None,
        m=basis.m,
        n=basis.n,
        nfp=basis.nfp,
    )

    coeff_cos = np.linspace(1.0, 2.0, modes.K)
    coeff_sin = np.linspace(-0.5, 0.25, modes.K)

    np.testing.assert_allclose(
        np.asarray(eval_fourier(coeff_cos, coeff_sin, basis_no_stack, coeffs_internal=True)),
        np.asarray(eval_fourier(coeff_cos, coeff_sin, basis, coeffs_internal=True)),
        atol=1e-13,
    )
    np.testing.assert_allclose(
        np.asarray(eval_fourier_dtheta(coeff_cos, coeff_sin, basis_no_stack, coeffs_internal=True)),
        np.asarray(eval_fourier_dtheta(coeff_cos, coeff_sin, basis, coeffs_internal=True)),
        atol=1e-13,
    )
    np.testing.assert_allclose(
        np.asarray(eval_fourier_dzeta_phys(coeff_cos, coeff_sin, basis_no_stack, coeffs_internal=True)),
        np.asarray(eval_fourier_dzeta_phys(coeff_cos, coeff_sin, basis, coeffs_internal=True)),
        atol=1e-13,
    )


def test_integral_helpers_validate_shapes_and_match_closed_forms() -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False)
    zeta = np.linspace(0.0, 2.0 * np.pi, 5, endpoint=False)
    sqrtg = np.full((3, theta.size, zeta.size), 2.0)
    s = np.asarray([0.0, 0.25, 1.0])

    dvds = dvds_from_sqrtg(sqrtg, theta, zeta, nfp=2)
    np.testing.assert_allclose(np.asarray(dvds), np.full(3, 4.0 * np.pi**2), atol=1e-13)
    _, volume = volume_from_sqrtg(sqrtg, s, theta, zeta, nfp=2)
    np.testing.assert_allclose(np.asarray(volume), [0.0, np.pi**2, 4.0 * np.pi**2], atol=1e-13)

    dvds_vmec = dvds_from_sqrtg_zeta(sqrtg, theta, zeta, signgs=-1)
    np.testing.assert_allclose(np.asarray(dvds_vmec), np.full(3, -8.0 * np.pi**2), atol=1e-13)
    _, volume_vmec = volume_from_sqrtg_vmec(sqrtg, s, theta, zeta, signgs=1)
    np.testing.assert_allclose(np.asarray(volume_vmec), [0.0, 2.0 * np.pi**2, 8.0 * np.pi**2], atol=1e-13)

    np.testing.assert_allclose(np.asarray(cumtrapz_s(jnp.asarray([]), jnp.asarray([]))), [])
    np.testing.assert_allclose(np.asarray(cumtrapz_s([3.0], [0.0])), [0.0])
    np.testing.assert_allclose(np.asarray(cumrect_s_halfmesh([1.0, 2.0, 4.0], s)), [0.0, 0.5, 3.5])

    with pytest.raises(ValueError, match="nfp must be positive"):
        dvds_from_sqrtg(sqrtg, theta, zeta, nfp=0)
    with pytest.raises(ValueError, match="non-empty"):
        dvds_from_sqrtg(sqrtg[:, :0, :], theta[:0], zeta, nfp=1)
    with pytest.raises(ValueError, match="must be 1D"):
        cumtrapz_s(np.zeros((2, 2)), s)
    with pytest.raises(ValueError, match="same length"):
        cumrect_s_halfmesh([1.0, 2.0], s)


def test_profile_two_power_pedestal_lrfp_and_integrated_current() -> None:
    cfg = ProfileInputs(
        pmass_type="two_power",
        piota_type="power_series",
        pcurr_type="two_power",
        am=jnp.asarray([10.0, 2.0, 1.0]),
        ai=jnp.asarray([0.0, 2.0]),
        ac=jnp.asarray([3.0, 2.0, 1.0]),
        ac_aux_s=jnp.asarray([]),
        ac_aux_f=jnp.asarray([]),
        pres_scale=2.0,
        bloat=1.0,
        spres_ped=0.5,
        lrfp=True,
        ncurr=1,
    )
    s = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])

    prof = eval_profiles(cfg, s)
    pressure_pa = 20.0 * (1.0 - s**2)
    pressure_pa = np.where(s > 0.5, 20.0 * (1.0 - 0.5**2), pressure_pa)
    current = 3.0 * (s - s**3 / 3.0)

    np.testing.assert_allclose(np.asarray(prof["pressure_pa"]), pressure_pa, atol=1e-13)
    np.testing.assert_allclose(np.asarray(prof["pressure"]), MU0 * pressure_pa, atol=1e-18)
    np.testing.assert_allclose(np.asarray(prof["current"]), current, atol=1e-13)
    assert np.isinf(np.asarray(prof["iota"])[0])
    np.testing.assert_allclose(np.asarray(prof["iota"])[1:], 1.0 / (2.0 * s[1:]), atol=1e-13)
    assert prof["ncurr"] == 1


def test_profile_spline_current_variants_and_indata_normalization() -> None:
    x = np.linspace(0.0, 1.0, 6)
    knots = jnp.asarray([0.0, 0.5, 1.0])
    values = jnp.asarray([1.0, 2.0, 3.0])
    np.testing.assert_allclose(np.asarray(_cubic_spline_profile(knots, values, x, integrate=False)), 1.0 + 2.0 * x)
    np.testing.assert_allclose(np.asarray(_cubic_spline_profile(knots, values, x, integrate=True)), x + x**2)
    np.testing.assert_allclose(np.asarray(_cubic_spline_profile(jnp.asarray([0.0]), jnp.asarray([4.0]), x, integrate=False)), 4.0)
    np.testing.assert_allclose(np.asarray(_cubic_spline_profile(jnp.asarray([0.0]), jnp.asarray([4.0]), x, integrate=True)), 4.0 * x)
    np.testing.assert_allclose(np.asarray(_cubic_spline_profile(jnp.asarray([]), jnp.asarray([]), x, integrate=True)), np.zeros_like(x))

    indata = InData(
        scalars={
            "PMASS_TYPE": "'two_power'",
            "PIOTA_TYPE": ['"power_series"'],
            "PCURR_TYPE": "cubic_spline_i",
            "AM": [1.0, 2.0, 1.0],
            "AI": [0.4],
            "AC": [1.0],
            "AC_AUX_S": [0.0, 0.4, 0.8, 0.7, 1.0],
            "AC_AUX_F": [1.0, 1.8, 2.6, 9.0, 3.0],
            "PRES_SCALE": 2.0,
            "BLOAT": 1.0,
            "SPRES_PED": -0.75,
            "LRFP": False,
            "NCURR": 1,
        },
        indexed={},
    )
    cfg = profiles_from_indata(indata)
    assert cfg.pmass_type == "two_power"
    assert cfg.piota_type == "power_series"
    assert cfg.pcurr_type == "cubic_spline_i"
    assert cfg.spres_ped == pytest.approx(0.75)
    np.testing.assert_allclose(np.asarray(cfg.ac_aux_s), [0.0, 0.4, 0.8])
    np.testing.assert_allclose(np.asarray(cfg.ac_aux_f), [1.0, 1.8, 2.6])

    prof = eval_profiles(cfg, [0.0, 0.4, 0.8])
    np.testing.assert_allclose(np.asarray(prof["current"]), [1.0, 1.8, 2.6], atol=1e-13)

    empty_spline_cfg = ProfileInputs(
        pmass_type="power_series",
        piota_type="power_series",
        pcurr_type="cubic_spline_ip",
        am=jnp.asarray([0.0]),
        ai=jnp.asarray([]),
        ac=jnp.asarray([1.0]),
        ac_aux_s=jnp.asarray([]),
        ac_aux_f=jnp.asarray([]),
        pres_scale=1.0,
        bloat=1.0,
        spres_ped=1.0,
        lrfp=False,
        ncurr=1,
    )
    np.testing.assert_allclose(np.asarray(eval_profiles(empty_spline_cfg, x)["current"]), np.zeros_like(x))


def test_profile_parser_edges_and_unsupported_types() -> None:
    assert _as_float_list(None) == []
    assert _as_float_list(3) == [3.0]
    assert _as_float_list(["1.5", 2]) == [1.5, 2.0]
    assert _lower(None, "default") == "default"
    assert _lower(["'Two_Power'"], "default") == "two_power"

    base = dict(
        am=jnp.asarray([1.0]),
        ai=jnp.asarray([0.4]),
        ac=jnp.asarray([1.0]),
        ac_aux_s=jnp.asarray([]),
        ac_aux_f=jnp.asarray([]),
        pres_scale=1.0,
        bloat=1.0,
        spres_ped=1.0,
        lrfp=False,
        ncurr=0,
    )
    with pytest.raises(NotImplementedError, match="pmass_type"):
        eval_profiles(ProfileInputs(pmass_type="bad", piota_type="power_series", pcurr_type="power_series", **base), [0.0])
    with pytest.raises(NotImplementedError, match="piota_type"):
        eval_profiles(ProfileInputs(pmass_type="power_series", piota_type="bad", pcurr_type="power_series", **base), [0.0])
    with pytest.raises(NotImplementedError, match="pcurr_type"):
        eval_profiles(ProfileInputs(pmass_type="power_series", piota_type="power_series", pcurr_type="bad", **base), [0.0])
