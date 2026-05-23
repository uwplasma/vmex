from __future__ import annotations

import numpy as np

import vmec_jax.energy as energy_mod
from vmec_jax.energy import (
    _as_float_list,
    _iotaf_from_iotas,
    _make_torflux_jit,
    _poly_no_const,
    _poly_no_const_deriv,
    flux_profiles_from_indata,
    integrate_volume_density,
)
from vmec_jax.namelist import InData


def test_polynomial_and_iotaf_edge_branches():
    x = np.asarray([0.0, 0.5, 1.0])
    assert _as_float_list(None) == []
    assert _as_float_list([1, "2.5"]) == [1.0, 2.5]
    assert _as_float_list("3.5") == [3.5]

    np.testing.assert_allclose(np.asarray(_poly_no_const([], x)), np.zeros_like(x))
    np.testing.assert_allclose(np.asarray(_poly_no_const_deriv([], x)), np.zeros_like(x))
    np.testing.assert_allclose(np.asarray(_poly_no_const([2.0, 3.0], x)), 2.0 * x + 3.0 * x**2)
    np.testing.assert_allclose(np.asarray(_poly_no_const_deriv([2.0, 3.0], x)), 2.0 + 6.0 * x)

    np.testing.assert_allclose(np.asarray(_iotaf_from_iotas(np.asarray([0.2]), lrfp=False)), [0.2])
    np.testing.assert_allclose(np.asarray(_iotaf_from_iotas(np.asarray([0.0, 0.3]), lrfp=False)), [0.3, 0.3])
    np.testing.assert_allclose(
        np.asarray(_iotaf_from_iotas(np.asarray([0.0, 0.2, 0.4, 0.8]), lrfp=False)),
        [0.1, 0.3, 0.6, 1.0],
    )
    np.testing.assert_allclose(
        np.asarray(_iotaf_from_iotas(np.asarray([0.0, 0.2, 0.4, 0.8]), lrfp=True)),
        [1.0 / (1.5 / 0.2 - 0.5 / 0.4), 2.0 / (1.0 / 0.2 + 1.0 / 0.4), 2.0 / (1.0 / 0.4 + 1.0 / 0.8), 1.0 / (1.5 / 0.8 - 0.5 / 0.4)],
    )
    np.testing.assert_allclose(
        np.asarray(_iotaf_from_iotas(np.asarray([0.0, 0.0, 0.0]), lrfp=True)),
        [0.0, 0.0, 0.0],
    )


def test_torflux_cache_and_flux_profiles_nonrfp_and_rfp():
    deriv, torflux = _make_torflux_jit((1.0, 0.5), False, 123)
    deriv_again, torflux_again = _make_torflux_jit((1.0, 0.5), False, 123)
    assert deriv is deriv_again
    assert torflux is torflux_again
    np.testing.assert_allclose(np.asarray(deriv(np.asarray([0.0, 1.0]))), [1.0, 2.0])
    assert _make_torflux_jit((1.0,), True, 123) == (None, None)

    s = np.asarray([0.0, 0.5, 1.0])
    indata = InData(
        scalars={
            "PHIEDGE": 2.0,
            "APHI": [1.0],
            "AI": [0.25],
            "AC": [0.5],
        },
        indexed={},
    )
    flux = flux_profiles_from_indata(indata, s, signgs=-1)
    assert flux.signgs == -1
    assert np.asarray(flux.phipf).shape == (3,)
    assert np.asarray(flux.chipf).shape == (3,)
    assert np.asarray(flux.phips)[0] == 0.0
    assert float(np.asarray(flux.phipf[-1])) < 0.0

    rfp = InData(
        scalars={
            "PHIEDGE": 1.0,
            "LRFP": True,
            "AI": [2.0],
        },
        indexed={},
    )
    rfp_flux = flux_profiles_from_indata(rfp, s, signgs=1)
    np.testing.assert_allclose(np.asarray(rfp_flux.chipf), np.full(3, 1.0 / (4.0 * np.pi)))
    assert np.all(np.asarray(rfp_flux.phipf) > 0.0)
    assert np.asarray(rfp_flux.phips)[0] == 0.0


def test_flux_profiles_default_torflux_skips_unneeded_profile_evaluation(monkeypatch):
    s = np.asarray([0.0, 0.25, 0.5, 1.0])
    current_driven = InData(
        scalars={
            "PHIEDGE": 2.0,
            "APHI": [1.0],
            "AI": [],
            "AC": [1.0],
            "PCURR_TYPE": "cubic_spline_ip",
            "AC_AUX_S": [0.0, 1.0],
            "AC_AUX_F": [0.0, 2.0],
        },
        indexed={},
    )

    def fail_eval_profiles(_indata, _s):
        raise AssertionError("default torflux with no AI should not evaluate profiles")

    monkeypatch.setattr(energy_mod, "eval_profiles", fail_eval_profiles)

    flux = flux_profiles_from_indata(current_driven, s, signgs=1)

    np.testing.assert_allclose(np.asarray(flux.phipf), np.full_like(s, 1.0 / np.pi))
    np.testing.assert_allclose(np.asarray(flux.chipf), np.zeros_like(s))
    assert np.asarray(flux.phips)[0] == 0.0


def test_flux_profiles_default_torflux_uses_iota_profile_when_present(monkeypatch):
    s = np.asarray([0.0, 0.5, 1.0])
    iota_driven = InData(
        scalars={
            "PHIEDGE": 2.0,
            "APHI": [1.0],
            "AI": [0.25],
            "AC": [],
        },
        indexed={},
    )
    calls = {"n": 0}

    def fake_eval_profiles(_indata, s_grid):
        calls["n"] += 1
        return {"iota": np.full_like(np.asarray(s_grid), 0.25, dtype=float)}

    monkeypatch.setattr(energy_mod, "eval_profiles", fake_eval_profiles)

    flux = flux_profiles_from_indata(iota_driven, s, signgs=1)

    assert calls["n"] == 1
    np.testing.assert_allclose(np.asarray(flux.chipf), np.full_like(s, 0.25 / np.pi))


def test_integrate_volume_density_single_surface_and_sign():
    density = np.ones((1, 2, 3))
    sqrtg = np.full((1, 2, 3), 2.0)
    s = np.asarray([0.0])
    theta = np.asarray([0.0, np.pi])
    zeta = np.asarray([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0])

    val = integrate_volume_density(density, sqrtg, s, theta, zeta, nfp=2, signgs=-1)

    expected = -np.sum(density * sqrtg) * (2.0 * np.pi / 2) * ((2.0 * np.pi / 3.0) / 2) * 2
    np.testing.assert_allclose(np.asarray(val), expected)
